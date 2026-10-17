---
paths:
  - "aquaflux/parallel/**"
---

# Rules — `aquaflux/parallel/` (domain decomposition & the sharded residual)

> **Provenance boundary (binding).** This file may cite the C++/Fortran precursors; none of that may
> reach the shipped surface (`.py`, `docs/`) — root `CLAUDE.md` Comment Convention.

Split the mesh across devices and evaluate the cell residual with `jax.shard_map`, differentiably. Dated
measurements and incident history: `.claude/notes/parallel-log.md`.

## The binding architectural decision — invert MPI, do not port it

The Fortran precursor is multi-controller MPI with explicit halo messages and a hand-derived transpose of
each for the adjoint. We keep the ghost-cell playbook and drop both costs.

**`shard_map`, not `jit` + `NamedSharding`/GSPMD — decided.** GSPMD degrades to all-to-all or replication
on an unstructured `segment_sum`; `shard_map` lets us write the per-partition program and name the
collectives. **The point is differentiability:** the transpose of `ppermute` is `ppermute`, of `psum` a
broadcast, so the halo exchange's adjoint is derived automatically — no hand-written adjoint communication.

## Module layout

- `partitioner.py` — `Partitioner` (`partition(mesh, n_parts) -> labels`) over `aquaflux.mesh.cell_adjacency_csr`.
  `BlockPartitioner` (dependency-free default: a reverse Cuthill–McKee ordering cut into balanced
  contiguous blocks); `ScotchCLIPartitioner` (min-cut via the `gpart` CLI, cross-platform);
  `ScotchPartitioner` (`scotchpy` binding, pip `scotchpy64`, Linux x86_64 wheels only — prefer the CLI on macOS).
- `partition.py` — `partition_mesh(mesh, labels) -> PartitionedMesh`, `LocalPartition`: owned cells + an
  appended halo ring, the local mesh an ordinary `Mesh`.
- `padding.py` — `PaddedLayout`, `pad_partition`: uniform per-shard shapes; operator-independent.
- `halo.py` — `HaloExchange` (`plan(layout) -> HaloPlan` at setup; `fill(owned_local, plan, axis_name)` in
  the sharded body, collective inside `fill`). `AllToAllHaloExchange` (default) moves only ghosted cells
  per neighbour in one `all_to_all`; `AllGatherHaloExchange` is O(P × owned) per device, kept only as the
  correctness reference the all-to-all is checked against.
- `distributed.py` — `build_distributed_residual` / `DistributedResidual`: the `shard_map` driver;
  operator-independent.

## Decided: the halo lives in a wrapper, never on `Mesh`

A partition's local mesh (owned + halo) **is** an ordinary `Mesh`, so everything runs on it unchanged;
ghost identity and send/recv lists live in `PartitionedMesh`/`LocalPartition`. This keeps the serial path
untouched. Facts the builder depends on:

- **Domain-boundary faces never cross a partition** (`neighbour = -1`, one owner), so a BC lives in one
  partition; only interior faces become partition-boundary faces.
- **Geometry is gathered, never recomputed** (`local_face_geometry`/`local_cell_geometry`): a ghost's
  stencil is incomplete locally. This is what buys bit-for-bit agreement with serial.
- **Global owner/neighbour roles are preserved**, so normals stay owner-outward.
- ⚠️ **A periodic seam keeps its per-face `neighbour_offset`** through both face renumberings via
  `mesh.face_cells.gather_neighbour_offset(face_index)` (padding faces take zero). Losing it raises
  nothing — it shows up only as a wrong displacement inside an operator. Pinned by
  `test_partition.py::test_local_meshes_keep_the_periodic_seam_displacement` and
  `test_padding.py::test_padding_keeps_the_periodic_seam_displacement`.
- **Each local mesh carries only its own nodes** (remapped face-node connectivity). `node_coords` is a
  differentiable leaf stacked to every device, so the global node array would otherwise be replicated P
  times for data the sharded residual never reads.

## Decided: the per-device body runs a real assembler — never a re-implementation

**Hold the line on this.** A hand-built per-device residual once drifted silently from
`ResidualAssembler` and could only express constant Dirichlet (history in the log). Now the caller injects
`assemble(mesh, geometry) -> assembler`, applied once per padded partition; the P assemblers are stacked
into one pytree with a leading partition axis and passed to `shard_map` sharded; the body refreshes the
halo and calls `assembler.residual(...)`. `parallel/` names **no** physical operator; its only dependency
outside `mesh/` is `BoundaryConditions`.

The injected assembler's contract is three members, all load-bearing: `residual(field, *,
gradient_hook=None)` (called as `residual(local_field, gradient_hook=hook)` — `TypeError` otherwise), a
`gradient_scheme` attribute (`None` if no gradient; read at build as `assemblers[0].gradient_scheme`), and
a mesh-bound `boundary`. Scalar and coupled-flow assemblers both satisfy it.

**A distributed capability extends the injected builder or the layout — never add physics to
`parallel/`.** Importing an operator here means the seam is wrong.

## Decided: how padding makes shapes uniform

`shard_map` needs identical local shapes; `PaddedLayout` is the one home for reconciling cell, ghost, face
and node counts. Two reserved slots keep it safe for any operator:

- **The null cell** (`n_owned_max + n_ghost_max`) owns every padding face, so padding can only scatter into
  a discarded row.
- **The null face** (`n_faces_max`) fills padded boundary-patch index lists.
- **Padding faces carry the reserved `"padding"` patch, not `"boundary"`** (binding). They have
  `neighbour = -1`; `BoundaryConditions.resolve` refuses uncovered boundary faces except `"padding"`
  (`.claude/rules/boundary.md`). The name is reserved in `FacePatches.from_dict`; every partition appends
  it, so the static `names` tuple is uniform. Likewise `FacePatches.patch_types`/`patch_groups`:
  `partition_mesh` keeps the **global** record (`eqx.tree_at` on `label` only) and `pad_partition` copies
  it — a per-partition record would make treedefs differ and break stacking.

Padding is inert through three independent mechanisms: zero area ⇒ zero flux; null-cell ownership; benign
geometry (normal along the first axis, face centroid one unit from the null-cell centroid, so the normal
distance is exactly 1 and nothing divides by zero). Traps that must stay closed:

- **Padding cells carry unit volume; real cells keep real volumes** (a transient term or volume source
  would otherwise be silently wrong). Pinned by `test_padding.py::test_real_geometry_survives_padding_unchanged`.
- **No `NaN` anywhere, including discarded rows** — a `NaN` in a padding row poisons the cotangent of every
  row sharing its reduction.
- **Boundary-patch bindings are padded, not labels.** `_uniform_boundary_faces` pads each patch's index
  array to the max over partitions with the null face. The boundary fold is one scatter over concatenated
  patch indices, so the null face repeats and which write lands is unspecified — inert, and why `apply`
  must never pass `unique_indices=True`.
- **The padded mesh is deliberately not `validate()`-d** (padding faces list no nodes; padding cells touch
  no face). Do not "fix" this.
- Ragged face-node array: the first padding face absorbs the padded tail (row pointers stay well-formed),
  the rest list no nodes; those incidences point at node 0. Ragged node array: padded to `n_nodes_max` with
  copies of node 0 (finite, inert).

## Halo depth — one layer, exchanging derived fields

**One ghost layer suffices because the gradient and limiter are exchanged, not recomputed.** A face flux
reads only its two cells' `phi`, `grad`, coefficient and limiter; recomputing a ghost's gradient is what
would need two layers (the Fortran precursor's reason). So: exchange `phi` → gradient on owned cells →
exchange gradient → flux. Ghost values are then serial-exact, so the distributed Jacobian equals serial,
and every exchange is a differentiable collective inside `R(phi)`.

- **Built via `gradient_hook`.** `ResidualAssembler.residual` applies an optional hook to the reconstructed
  gradient (identity serially). `DistributedResidual` passes one that overwrites ghost rows using the same
  `HaloExchange` and plan as `phi` (`fill` is trailing-axis generic). It runs only when a gradient scheme is
  injected (`DistributedResidual.exchange_gradient`, static). Boundary values need no correction (they read
  owner gradients only).
- **Iterative gradient schemes get a per-sweep exchange through the same hook**, threaded via
  `_gradient` → `GradientScheme.gradients(..., operator_hook=…)` → `GradientSolve.solve(..., operator_hook=…)`,
  which wraps the operator as `operator(operator_hook(x))`. Single-pass `CompactGreenGauss` ignores it;
  `CorrectedGreenGauss` + `SweptGradientSolve` is serial-exact sweep by sweep.
- **This requires a reduction-free solve — `SweptGradientSolve`** (preconditioned Richardson with
  cell-local `GradientPreconditioner`s). `GmresGradientSolve` **raises** on `operator_hook` (its inner
  products would double-count ghosts and are not `psum`-reduced); `HessianCorrectedGradient` **raises**
  (nested solves read ghost gradients and Hessians the hook does not refresh). Lifting this is issue #134.
  The distributed swept solve drops the host-side under-resolution diagnostic (it would need an owned-only
  reduction).
- ⚠️ **The library default gradient does not run here — a partitioned build must name a scheme.**
  `MomentumContinuity.build`/`SSTTurbulence.build` default to `schemes.DEFAULT_GRADIENT_SCHEME`
  (`MultipleCorrectionGradient`), which raises on `operator_hook`: its two passes need an exchange between
  them that no seam supplies (#134).
- The limiter `psi` exchange is deferred until distributed advection exists; it reuses the same hook.

## Decided: collective pattern is `all_to_all`, not `ppermute`

A min-cut partition has an irregular, non-uniform neighbour set. `ppermute` would need edge-colouring that
data-dependent graph into up to `max_degree` rounds; `all_to_all` does the whole exchange in one collective
with uniform per-pair buffers. `ppermute` only suits a structured (Cartesian brick) decomposition, which is
not this regime.

## Testing

Serial-equivalence in value **and** gradient is the standard: `test_partition.py` (loop-emulated,
2/3/4 partitions, node remap), `test_padding.py` (no devices; padded residual matches unpadded row for row),
`test_shard_map_smoke.py`, `test_distributed.py` (mixed `Dirichlet`/`Neumann`/`ZeroGradient` so pre-baked
values cannot pass), `test_halo_all_to_all.py` (bit-for-bit vs all-gather), `test_distributed_gradient.py`
and `test_distributed_swept_gradient.py` (skewed mesh; controls with the gradient / per-sweep exchange
turned off must **fail**, proving each exchange load-bearing; GMRES and Hessian solves must raise).

- Device tests simulate 4 CPU devices in a subprocess spawned **only** through
  `tests/support/devices.py::run_on_simulated_devices`. ⚠️ `XLA_FLAGS` must be **appended** to, never
  assigned (assignment drops the workflow's `--xla_cpu_multi_thread_eigen=false` and starves another
  worker); pinned by `tests/unit/test_simulated_devices.py`.
- ⚠️ **They are the fast tier's most expensive tests and their cost is set by machine contention**, not by
  the test (an order of magnitude with no log line saying so). They carry `pytestmark =
  pytest.mark.distributed` and run in their own CI job at `-n 2` (see root `CLAUDE.md`, "What a fast-tier
  test may spend"); `tests/unit/test_distributed_marker.py` enforces the marker. A worker killed with
  `node down: Not properly terminated` is a timeout under contention, not a crash — raising the timeout
  does not fix it. Before adding a device test, account for its cost; do not raise `-n`.

## Not yet built (each waits on a real need)
The core — decomposition, padding, injected-assembler `shard_map` residual, all-to-all halo, local node
sets, derived-field and per-sweep gradient exchange — matches serial in value and adjoint.
1. **Distributed GMRES / Hessian gradient solve** (owned-only `psum` inner products + nested-solve
   exchanges) — issue #134; not a blocker, `SweptGradientSolve` is the production path.
2. **Limiter `psi` exchange** — blocked on a distributed-advection path.
3. **Per-partition reverse Cuthill–McKee** (`permute_cells` + `ReverseCuthillMcKee`) — no consumer yet
  (no per-partition preconditioner or matvec benchmark; `BlockPartitioner` is already RCM-ordered) — #135.
4. **Multi-node** via `jax.distributed.initialize()`; multi-node CPU leans on Gloo and is materially slower
   than MPI — re-verify before targeting a CPU cluster.

## Open questions
- Load balance vs halo size in the Scotch objective for the coupled block.
- Distributed AMG: the hierarchy is built off-jit from global connectivity and frozen, so only the
  matrix-free V-cycle runs sharded; a partition-local AMG with a cross-partition smoother is the fallback.
