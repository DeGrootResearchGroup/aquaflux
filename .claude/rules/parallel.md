---
paths:
  - "aquaflux/parallel/**"
---

# Rules — `aquaflux/parallel/` (domain decomposition & the sharded residual)

> **Provenance boundary (binding).** This file cites the C++/Fortran precursors to inform *your*
> understanding — that is its job, and why it loads into your context. Per the root `CLAUDE.md`
> **Comment Convention**, none of that provenance may reach the shipped surface (`.py`
> comments/docstrings, `docs/`): cite the *math*, never the reference code, the `.claude/` rules,
> the design notes, or the author's own papers.

Distributed-memory execution: split the mesh across devices and evaluate the cell residual with
`jax.shard_map`, keeping the whole thing differentiable. Governed by the root `CLAUDE.md`
Engineering Principles.

## The binding architectural decision — invert MPI, do not port it

The Fortran precursor is multi-controller MPI: each rank owns cells plus ghosts, issues explicit
`MPI_Send`/`MPI_Recv` halo exchanges, and needs a **hand-derived transpose of every message-passing
step** to get an adjoint. We keep the ghost-cell playbook and drop both of its costs.

`shard_map` (not `jit` + `NamedSharding`/GSPMD) is the chosen model, and this is **decided, not
open**. GSPMD is an auto-partitioner tuned for dense/structured arrays; an unstructured FVM
`segment_sum` over arbitrary face→cell connectivity is exactly the case it degrades to all-to-all or
full replication. `shard_map` is the MPI-style escape hatch *inside* the compiled, differentiable
world: we write the per-partition program and name the collectives.

**The differentiability win is the point.** The transpose of a `ppermute` is another `ppermute`; the
transpose of a `psum` is a broadcast. JAX derives the adjoint of the halo exchange automatically, so
the distributed solve stays differentiable across partitions with **zero hand-written adjoint
communication** — the step that is a notorious, error-prone hand-derivation in an MPI adjoint code.

## Module layout

- `partitioner.py` — **`Partitioner`** strategy (`partition(mesh, n_parts) -> labels`) over the
  shared cell graph (`aquaflux.mesh.cell_adjacency_csr`, the same graph the reordering code uses).
  `BlockPartitioner` (dependency-free default: cut a reverse Cuthill–McKee ordering into balanced
  contiguous blocks — space-filling-curve style, always valid, perfectly balanced);
  `ScotchCLIPartitioner` (min-cut via the `gpart` command-line tool — the **cross-platform** Scotch
  path, `brew install scotch` / the `scotch` package); `ScotchPartitioner` (same via the `scotchpy`
  binding — pip package `scotchpy64`, **Linux x86_64 wheels only**, so prefer the CLI on macOS).
- `partition.py` — **`partition_mesh(mesh, labels) -> PartitionedMesh`** and `LocalPartition`. The
  decomposition wrapper: owned cells plus an appended halo ring, with the local mesh built as an
  ordinary `Mesh` so every operator/scheme/BC runs on it verbatim.
- `padding.py` — **`PaddedLayout`** and **`pad_partition`**: uniform per-shard shapes. Deliberately
  **operator-independent**.
- `halo.py` — **`HaloExchange`** strategy; `AllGatherHaloExchange` is the current implementation.
- `distributed.py` — **`build_distributed_residual`** / **`DistributedResidual`**: the `shard_map`
  driver. Also operator-independent.

## Decided: the halo lives in a wrapper, never on `Mesh`

`Mesh` stays a pure contiguous-block container. A partition's *local* mesh — owned cells plus the
halo ring — **is** an ordinary `Mesh`, so everything runs on it unchanged; who is a ghost, which
remote cell it mirrors, and the send/recv index lists are a **separate responsibility** and live in
`PartitionedMesh`/`LocalPartition`. This is what keeps the entire serial path untouched by
distribution.

Three correctness facts the builder depends on:

- **Domain-boundary faces never cross a partition** (a boundary face has `neighbour = -1` and one
  owner), so a boundary condition lives entirely within one partition. Only *interior* faces become
  partition-boundary faces.
- **Geometry is gathered, never recomputed.** A ghost cell's face stencil is incomplete locally, so
  recomputing its centroid/volume would be wrong. `local_face_geometry` / `local_cell_geometry`
  gather from the global geometry, which is what buys bit-for-bit agreement with serial.
- **Global owner/neighbour roles are preserved** by the remap, so owner-outward normals stay
  consistent — no re-orientation.

## Decided: the per-device body runs a real assembler — never a re-implementation

**This is the rule to hold the line on.** An earlier version of `distributed.py` hand-built
`FaceCellConnectivity` + `MeshGeometry` + `FaceContext` and called `DiffusionFlux` directly — a
second implementation of `ResidualAssembler`. It drifted silently (the `MaterialModel` →
`PropertyModel` rename broke it and no test caught it, because there was no shared code path to
break), it hardcoded one operator, and it substituted a **pre-baked constant per-face
`boundary_value` array** for the boundary closures — which can only express constant Dirichlet, so
`Neumann`, `ZeroGradient`, `Convective`, and wall functions were all silently out of reach.

The structure now is: the caller injects `assemble(mesh, geometry) -> assembler`, applied once per
padded partition; the P assemblers are stacked into one pytree with a leading partition axis and
passed to `shard_map` as a sharded input; the per-device body refreshes the halo and calls
`assembler.residual(...)`. `parallel/` therefore names **no** physical operator, and its only
dependency outside `mesh/` is `BoundaryConditions` (for the patch-binding padding below).

The contract the injected assembler must satisfy is exactly two members — `residual(field)` and a
mesh-bound `boundary`. Both the scalar and coupled-flow assemblers already have these.

**When adding a distributed capability, extend the injected builder or the layout — never add
physics to `parallel/`.** If you find yourself importing an operator here, the seam is wrong.

## Decided: how padding makes shapes uniform

`shard_map` requires identical local shapes on every shard, but partitions differ in cell, ghost,
and face counts. `PaddedLayout` is the single home for reconciling that, and holds no physics.

Two reserved slots make the padding safe for *any* operator, which is what lets the layout stay
operator-independent:

- **The null cell** (`n_owned_max + n_ghost_max`) is past every partition's real cells. Every
  padding face names it as owner, so a padding face can only ever scatter into a discarded row.
- **The null face** (`n_faces_max`) is past every partition's real faces; it is the fill value for
  padded boundary-patch index lists.

Padding is inert through **three independent mechanisms** — deliberately belt-and-braces, since the
layout no longer knows which operator will run: (1) zero area ⇒ zero flux; (2) null-cell ownership
⇒ even a non-zero flux cannot reach a real row; (3) benign geometry — normal along the first unit
axis, face centroid one unit along it from the null-cell centroid, so the owner-to-face normal
distance is exactly `1` and nothing divides by zero.

Specific traps that are now closed, and must stay closed:

- **Padding cells carry unit volume; real cells keep their real volumes.** The earlier code gave
  *every* cell volume `1.0`. Diffusion never reads volume, so it was inert — but the first transient
  term or volume source would have been silently wrong, with no error and no `NaN`.
  `tests/unit/test_padding.py::test_real_geometry_survives_padding_unchanged` pins this.
- **No `NaN` anywhere, including in discarded rows.** A `NaN` in a padding row still poisons the
  cotangent of every row sharing its reduction, so the gradient breaks even though the value looks
  fine. Padding geometry is finite and non-degenerate for this reason.
- **Boundary-patch bindings are padded, not the patch labels.** A named patch holds a different
  number of faces per partition (possibly zero). `_uniform_boundary_faces` pads each resolved index
  array to the max over partitions, filling with the null face. This is safe because the boundary
  fold writes with `.at[faces].set(...)` — the padded entries write a value at a zero-area,
  null-cell-owned face.
- **The padded mesh is deliberately not `validate()`-d.** Padding faces list no nodes and padding
  cells are touched by no face; `Mesh.validate()` rejects both, correctly, for a mesh describing
  real geometry. A padded mesh is device-axis bookkeeping and its geometry is *gathered*, so the
  node-level invariants do not apply. Do not "fix" this by calling `validate()`.
- **The face-node array is ragged across partitions**, so it is padded too: the first padding face
  absorbs the whole padded tail (keeping the row pointers well-formed) and the rest list no nodes.

## Halo depth — one layer, exchanging derived fields (decided)

**One ghost layer suffices — provided the gradient and limiter are *exchanged*, not recomputed.**
The residual at an owned cell reads, at each face, only quantities at that cell and its immediate
neighbour: `phi`, `grad`, the coefficient, and (for limited advection) the limiter at the upwind
cell. The flux never reaches past one layer. The *only* thing that would force a second layer is
**recomputing** the neighbour's gradient/limiter locally, which needs that neighbour's own stencil —
two layers out. The Fortran precursor took the second layer for exactly this reason.

aquaflux already materializes the gradient and limiter as per-cell fields before the flux, so it
exchanges them instead: exchange `phi` → compute `grad`/limiter on **owned** cells (correct: an
owned cell's stencil is within owned + the one-layer `phi` halo) → exchange `grad`/limiter → flux.
The ghost values are then bit-for-bit what they would be serially, so the distributed Jacobian
equals the serial Jacobian. Every exchange is a differentiable collective *inside* `R(phi)`, so the
AD-linearized limiter and the exact adjoint are preserved automatically.

## Testing

- `tests/unit/test_partition.py` — the loop-emulated distributed residual matches serial for
  2/3/4 partitions; the halo plan reproduces a direct global gather; a parameter gradient through
  the decomposition matches serial.
- `tests/unit/test_padding.py` — padding in isolation: **no devices, no `shard_map`**. Uniform
  shapes, real geometry preserved (volumes explicitly), padding inert and finite, and the load-
  bearing one: a residual on a padded local mesh matches the unpadded one row for row.
- `tests/unit/test_shard_map_smoke.py` — the `shard_map` + `all_gather` mechanism matches a serial
  reference in **value and gradient** (adjoint exact to 0).
- `tests/unit/test_distributed.py` — the full sharded residual and its gradient match serial. Uses a
  **mix of boundary closures** (`Dirichlet`, `Neumann`, `ZeroGradient`) precisely because the
  solution-dependent ones cannot be pre-baked — they only pass if the real closures run per device.

The device tests simulate 4 CPU devices via `--xla_force_host_platform_device_count=4`, which must
be set **before JAX initializes** — hence the subprocess.

## Not yet built (in priority order)

1. **`HaloExchange` should own its plan and its collective.** Current `fill(owned_local, all_owned,
   ghost_src_partition, ghost_src_owned_index)` takes the all-gather *result*, and the
   `all_gather` is issued in the `shard_map` body outside the strategy. A neighbour-only `ppermute`
   never receives an `(n_partitions, n_owned_max)` array and **cannot implement this signature**.
   Reshape to `plan(pmesh)` at setup plus `fill(owned_local, plan)` with the collective *inside*,
   **before** writing `ppermute` — not during.
2. **The derived-field halo** (`grad`, limiter) per the one-layer scheme above — an exchange hook
   between gradient reconstruction and the flux loop. Needed the moment a non-orthogonal correction
   or a limiter runs distributed; the current gate is orthogonal-grid, where the correction vanishes.
3. **Node remapping.** Each local mesh keeps the **full global** `node_coords`, so the stacked
   sharded arrays replicate it P times. Correct but wasteful; remap to a partition-local node set.
4. **Per-partition reverse Cuthill–McKee**, applied as a transform over a built `PartitionedMesh`
   (not woven into the build loop) so it stays contained.
5. **Multi-node** via `jax.distributed.initialize()`. Note the honest gap: multi-host is
   production-grade on GPU/TPU, but multi-node **CPU** leans on Gloo and is materially slower than
   MPI — re-verify before committing to a CPU cluster.

## Open questions

- **Collective pattern:** `ppermute` (structured neighbour set) vs `all_to_all` (general partition
  graph) — depends on how irregular real partition adjacency turns out to be.
- **Load balance vs halo size** in the Scotch objective for the coupled block.
- **Distributed AMG setup.** The smoothed-aggregation hierarchy is built off-jit from the global
  connectivity once and frozen, so partitioning does not disturb the *setup*; only the frozen
  matrix-free V-cycle application runs sharded, and its collectives transpose the same way. A
  partition-local AMG with a cross-partition smoother is the fallback if that proves insufficient.
