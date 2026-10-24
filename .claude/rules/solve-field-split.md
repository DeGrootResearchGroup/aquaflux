---
paths:
  - "aquaflux/precondition/field_split.py"
  - "aquaflux/precondition/traced_field_split.py"
  - "aquaflux/precondition/traced_cycle.py"
---

# Rules — `aquaflux/precondition/field_split.py` (the block-triangular field split)

> Split out of `solve.md`; see it for the package-wide contracts. Dated measurements, A/B tables and the
> investigation history live in `.claude/notes/solve-field-split-log.md` — read it before re-measuring an
> arm. The coupled march is `coupled_step` given a `MaterializedJacobian(FieldSplit(...))` value and run on
> a session (`open_session`); see `turbulence.md`'s rename table for the deleted continuation builders.

## The field split — a saddle plus two transported scalars

**What it is.** `FieldGroups` / `FieldSplitInverse` / `field_split_inverse` / `FieldSplitPreconditioner`
(`field_split.py`): a block-lower-triangular inverse over `[u,v,w,p]` (leading) and `[k,ω]` (trailing),
always leading first, keeping the `[k,ω] ← [u,v,w,p]` coupling. `leading_inverse` and `trailing_inverse`
are required. Both flagship cases ship `SimpleSmoothed` leading and `JacobiSmoothed` trailing
(`case.yaml`, both `frozen_coarsening: true` on the leading block and `equilibrate: false` on the
trailing one). The operator stays monolithic, so the AD Jacobian and adjoint are untouched.

**Why a split.** The coupled block is six fields, two of them advection-dominated scalars (ω in a log
variable); the saddle-point literature (Vanka, SIMPLE pre-transforms, monolithic saddle AMG, Uroić–Jasak)
targets `(u,p)` only and has not transferred. The hard direction of the degenerate cell blocks is pure ω,
driven by a large ω-row coupling to the cell's own `p`/`u` (`solve-refuted-directions.md`, the WHICH-row
entry) — which is exactly the split's retained lower block. A block-*diagonal* arm (coupling dropped)
was refuted: the coupling is load-bearing. On `bfs3d` the split marched 31 % faster than the
monolithic V-cycle while taking 11 % more cycles (log, 2026-08).

**Contracts.**
- **The partition is free because the state is field-major**: dof `(cell i, field f)` is at
  `f·n_cells + i`, so groups are contiguous ranges and blocks are contiguous submatrices. `FieldGroups`
  is a view over the state's `FieldLayout` (`solve/state.py`); build it with
  `FieldGroups.split_before(layout, "k")` (a probe holding a raw matrix uses `FieldGroups.by_counts`).
  A layout with a `GlobalDofs` block is refused. Never re-derive `f·n_cells + i` inline.
- **`factors.n_dofs` + `factors.apply(r, transpose=)` is the whole surface the split satisfies** for
  `HostPreconditioner.matvec()`. ⚠️ A base reading anything else off `self.inverse` raises on the split,
  and `getattr(pc, name, False)` turns that raise into a plausible `False` — test such a property by
  reading it directly.
- **Refresh**: `FieldSplitInverse.refactor_block` makes the split a `RefactorableInverse`;
  `FieldSplitPreconditioner` inherits `MaterializedJacobianPreconditioner.refresh_in_place`
  (`materialized_preconditioner.py`). A block inverse without `refactor_block` is refused with a
  `TypeError` before either block is touched. **A session's split never sees the whole matrix**: its
  probe is laid out by the split's blocks (`JacobianProbe.build(groups=)`, `probe_layout.BlockLayout`),
  so a fit and every refit probe one block-major value array, add the shift on its stored diagonal in
  place (`ProbedBlocks.add_shift`) and hand the blocks to `FieldSplitInverse.refactor_blocks` /
  `_field_split_of_blocks` — no whole CSR, shifted copy or cut (bfs3d 2-step peak 9.42 -> 8.47 GiB, held
  host 3.12 -> 2.74 GiB without the cut's maps, 2026-10-18).
  - **All of it on the device (2026-10-22)**: the values, the shift and the pivot floor stay there, the
    blocks are `entry_map.DeviceMatrix` slices (the whole array is let go before the fit, so the two are
    not held through it together), and `hierarchy_inverse.block_for` hands a `HierarchyBlockInverse` its block as it is
    (`FrozenCoarsening.fit` fits it in place) and any other inverse a host matrix (`as_scipy`), as
    `RefactorableInverse` promises. The first fit reads the two diagonal blocks back once (a factory
    coarsens on the host). `_set_coupling` holds the coupling as a `DeviceMatrix` in the split's
    precision (a copy, never a view of the probe's values); `FieldSplitInverse.coupling` is a host copy
    made on first read after a refit and dropped by the next. A refit reads back only the pivot floor's
    row set and the fit's checks. Pinned by
    `test_a_refit_probed_into_blocks_stays_on_the_device_and_fits_what_the_host_cut_fits` (it patches
    `as_scipy` to raise). Cost on pitzDaily: log, "The refit's probe and coupling on the device
    (2026-10-22)".
  `refactor_block`
  (a whole operator, for other callers) still cuts by held gathers (`FieldGroups.blocks(matrix, maps)`),
  and `traced_field_split` packs the coupling through the split's maps
  (`FieldSplitInverse.padded_coupling()`); all bit-identical to slicing and `from_scipy`
  (`solve-amg-multigrid.md`, "entry maps"). Pinned by
  `test_a_block_layout_materializes_the_blocks_cut_from_the_whole_matrix`.
- **Transpose is closed-form** (reverse the block solves, use `Cᵀ`), so the adjoint is served. `Cᵀ` is
  formed on first use (`_transposed_coupling`) and dropped by `_set_coupling`, the one place a coupling is
  stored; do not re-derive it per apply (`A.T` on CSR is a CSC view that converts on every product).
- **Only the dropped triangle is not materialized.** `FieldGroups.active_rows()` gives the
  `(n_fields, n_fields)` table of field-pair blocks to keep; `column_probe_plan(active_rows=)`
  (`sparse_jacobian.py`) filters the pattern before the gather map is built; a probe takes the split's
  `FieldGroups` (`JacobianProbe.build(groups=)`, `coupled_jacobian_probe(..., groups=)`,
  `MaterializedProblem.probe(settings, groups)`), which gives both this table and the block layout.
  `None` is byte-identical to the full pattern.
  On `bfs3d` this cut stored `nnz` 47.2M → 36.7M (−22 %) with `apply` unchanged forward and transpose.
  `active_rows` (a static per-field-pair table) and `in_reach` (a per-entry reach question) are separate
  mechanisms; keep them so.

**Traps that transfer.**
- ⚠️ **Rank Krylov-accelerated preconditioners by convergence on the true residual, not one
  application's contraction**, and only at a state that discriminates (a state every arm solves in one
  cycle ranks nothing). Rank on step-initial states weighted by the march's own solve mix; a hard
  iterate only screens.
- ⚠️ **Measure a coupled preconditioner at β = 0 on a converged state** as well as on the march — that
  is the adjoint's operator. Do not strip the shift off a mid-march iterate.
- ⚠️ **Two march arms that differ only in the preconditioner must have tracking residuals for their
  first steps**; if they part at step 1 the comparison is confounded.
  `validation/bfs3d_openfoam/march_log_compare.py` reports where two logs part. Report the refresh
  count beside the wall: `refresh_on_cycles` is denominated in cycles, so a weaker arm is penalized twice.
- ⚠️ **A cycle count is not a cost proxy once the smoother changes** (one trailing sweep against four:
  fewer seconds, more cycles).
- ⚠️ **Checkpoint names are reused across marches** (`StateCheckpointer` keeps
  `BFS3D_CHECKPOINT_KEEP`, default 3, numbered per run). `field_split_probe.load_state` checks the
  recorded shift; raise the keep count for a study needing a trajectory.
- **The `[k,ω]` cell block is effectively lower-triangular** (ω depends strongly on same-cell k; k
  barely on ω where the SST limiter is active), which is why a cell-ordered sweep or a cell-block inverse
  handles it and point Jacobi does not. Re-ordering fields to `(ω, k)` makes forward sweeps worse.
- **A hierarchy on the raw `[k,ω]` Jacobian slice must be field-aware.** A field-blind scalar
  aggregation merged k and ω dofs from different cells and produced a negative coarse diagonal at
  level 1 (the fine slice is clean). A cell-block aggregation (`JacobiSmoothed`, block lAIR with
  `_cell_block_inverse`) builds on it; see `solve-amg-multigrid.md`.
- Closed arms (Vanka on the four-field saddle, Chebyshev on either block, flat block Jacobi as the
  whole trailing inverse, the PETSc split blocks, turbulence-first ordering): `solve-refuted-directions.md`
  and the log.

## `TracedFieldSplit`: the composition on device (#290)

`traced_field_split.py` holds `TracedFieldSplit`, `traced_field_split(inverse)` and
`offers_traced_split(inverse)` (submodule-only plumbing). Each block inverse offers `traced_cycle() ->
TracedCycle` (`traced_cycle.py`; `TracedCycle` / `OffersTracedCycle` are exported from `precondition`,
for a user's own block inverse). Every shipped block inverse (`SimpleSmoothed`, `JacobiSmoothed`,
`AirReduction`) offers one, so every spec-built `FieldSplit` is traced; the host form survives only for
a block without a traced cycle (an `inverse_wrapper`).

**`TracedForm` (`traced_cycle.py`) is what a step carries, whichever shape the inverse has**: an abstract
`equinox.Module` owning `in_precision` and `matvec()`, its subclasses supplying `apply`, `n_dofs` and a
static `precision`. Two subclasses: `TracedFieldSplit`, and `TracedSingleBlock(cycle, n_dofs)` for a
bare `BlockInverse` over the whole state (the laminar `MaterializedJacobian(SimpleSmoothed())`). The
session's `_traced_form(inverse, precision)` (`materialized_session.py`) is the one choice
of form, returning `None` for an inverse with a host block. Pinned for the single block by
`test_a_single_block_inverse_is_carried_in_the_step_and_each_refit_hands_back_a_new_one`
(`tests/integration/test_flow_march.py`). ⚠️ The base does not declare `n_dofs`/`precision` as
`eqx.AbstractVar`: equinox then orders `precision` (which has a default) before a subclass's own fields,
and the dataclass refuses to build.

**Why.** The host split's numpy `apply` behind a `pure_callback` forces six whole-vector host↔device
transfers per application even when both blocks are traced cycles. On CPU the traced form is
bit-identical and at parity (a pitzDaily march: same 20 steps / 154 cycles / root, 5.6 % slower, one run
each); the case for it is the GPU, which is unmeasured — harness `validation/traced_split_benchmark.py`.

**Binding design.**
- **The hierarchies are array leaves, not constants.** `TracedCycle.state` holds the hierarchy as
  leaves; its `step` is static and compares by value (`_HierarchyStep` in `hierarchy_inverse.py`,
  `_AirStep` in `field_split.py`). A static bound method closed over the hierarchy made a refreshed
  split hit the jit cache and apply the stale hierarchy with no error. Pinned by
  `test_a_split_rebuilt_after_a_refresh_reuses_the_compiled_solve_and_applies_the_refresh` and
  `test_both_hierarchies_ride_as_leaves_not_as_constants`. The inverses' eager `_solve` routes through
  the same object.
- **No `transpose` argument**: `jax.linear_transpose` derives `Mᵀ` (tested against the host transpose).
- ⚠️ **A bound method of an `equinox.Module` holding arrays cannot be passed to `jax.jit`** (unhashable
  cache key); `matvec()` returns a plain closure.

**How it is wired.**
1. `MaterializedSession._build` fits the host preconditioner, then, if `_traced_form(...)` returns a
   form, uses `TracedFormShiftPolicy(base, session._form)` (`monolithic_policy.py`) instead of
   `MonolithicFactorShiftPolicy`. **One traced form per fit**: the
   session takes it once at the fit and after each re-fit, and every step it builds (the stopping step
   included, re-pointed at each re-fit) carries those arrays — a split per step kept a copy of the whole
   hierarchy alive per held step (`solve.md`, `materialized_session`). The problem's
   `build_step(state, policy, march, base_regime)` receives the glued policy. `session.preconditioner`
   is the host object either way.
2. **A re-fit is a new step, not a mutation.** `BetaTrackingRefresh` re-fits in place and returns whether
   it did; the session returns an update `step -> with_traced_form(step, form)`, which `newton_march`
   applies to both `strategy` and `active_step`, and reports on `MarchResult.strategy` for the next
   segment and the adjoint. The superseded split's own arrays are released (`solve.md`,
   `materialized_session`), so a step kept across a re-fit raises if applied.
3. **The form rides in two places, both data**: the policy and the adjoint factory (`_TracedFormAdjoint`;
   the step's `adjoint_preconditioner_factory` is a data field). `with_traced_form` replaces both in one
   `tree_at`. ⚠️ Updating only the policy leaves the adjoint on the starting split.
4. **A re-fit is a cache hit only at unchanged array shapes.** A block whose coarsening reads values
   must set `frozen_coarsening` (both cases do on the leading block), and every derived sparse piece is
   stored on its structural pattern (`solve-amg-multigrid.md`, `solve-flow-block.md`).
   `validation/traced_split_refresh_shapes.py` checks it: 0 moved signatures, 1 trace across 40 builds.

## `FieldSplit(precision="float32")`: the split applied in single precision

**The block inverses hold their hierarchies in the split's precision** (`HierarchyBlockInverse(stored_
precision=)`, passed by the session through `BlockInverse.bound(stored_precision=)` where the family's
`_stores_precision` allows; lAIR does not, and stays double). Each fit and refit is computed in float64
and rounded once, prolongation values excepted (a frozen refit reads them back, and the held Galerkin maps
key on them), so the split shares the inverse's arrays and applies exactly what narrowing a double
hierarchy gave. Each SIMPLE piece is rounded as it is formed (`_simple_pieces(precision=)`, the host
Schur it returns stays double) and the levels a level at a time after `derive_extras`, so no whole double
copy sits beside a whole rounded one. **The host split holds its coupling in the split's precision too**
(`FieldSplitInverse(stored_precision=)`, from the session's `FieldSplitPreconditioner.build`), held on
the device as a `DeviceMatrix` rounded by `in_precision` (JAX's conversion, which flushes
single-precision subnormals as the traced split's does) and packed into single-precision slots
(`_PaddedLayout.operator` keeps its operand's type); `FieldSplitInverse.coupling` reads a host copy back
on first use after a refit, for the host `apply` and diagnostics. All of this is bit-identical in every array the solve applies — the traced split is
composed from the same rounded entries — pinned by
`test_a_split_holding_its_coupling_in_single_precision_composes_the_same_single_split`,
`test_pieces_held_in_single_precision_are_the_double_pieces_rounded_once`, and on bfs3d by the
`BFS3D_MEMORY_CHECKSUMS=1` leaf digests of `memory_attribution.py`. Host `apply` stays double (a float64 residual promotes every operation;
an explicit widening was measured bit-identical and deleted). bfs3d 2-step peak 10.82 -> 10.15 GiB, steady
device float64 1.23 -> 0.15 GiB (2026-10-18; `solve-direct-preconditioners-log.md`, "bfs3d memory by
owner"). Pinned by `test_an_inverse_held_in_single_precision_holds_its_double_fit_rounded_once` and
`test_a_single_precision_split_applies_the_hierarchies_its_inverses_hold_not_a_copy`.

The library default is `"float64"`; pitzDaily's and bfs3d's `case.yaml` set `precision: float32` (owner's
decision, #732: same steps, cycles and root; pitzDaily −24 %, bfs3d −17 % wall;
`solve-open-directions.md` §7). ⚠️ Split-apply or march timings on either case recorded before
2026-10-16 are float64; `PITZ_SPLIT_PRECISION` / `BFS3D_SPLIT_PRECISION=float64` reproduce them.

- `FieldSplit.precision` (`SPLIT_PRECISIONS`, `materialized_spec.py`) reaches the split through
  `MaterializedSession._precision` at both the first fit and a refit. The fit is computed in float64
  (hierarchies, inverted diagonal blocks, coupling); `TracedForm.in_precision` rounds every floating leaf once,
  and `apply` narrows its input and widens its output, so residual, Krylov and stop stay float64. A
  host-composed split refuses float32 at the fit.
- ⚠️ **The adjoint must be preconditioned in double.** float32 arithmetic is linear only to ~1e-7, and
  GMRES needs a fixed linear preconditioner, so a tight transpose solve stagnates. `_TracedFormAdjoint`
  widens the split (`in_precision("float64")`) once per transpose solve — the rounded entries are
  harmless, the rounded arithmetic is not. Pinned by
  `test_the_adjoint_applies_a_single_precision_split_in_double_and_so_exactly_linearly`.

## Materialization is still a scaling wall

`FieldSplitPreconditioner` materializes the coupled Jacobian in one coloured-probe pass over all `dim+3`
fields at reach 3 (`_materialize_jacobian`) and slices the blocks from it. With the dropped triangle
excluded, the kept blocks are ~78 % of the full pattern (leading 44 %, trailing 11 %, retained coupling
22 % on `bfs3d`); linearly extrapolated to 1M cells that is ~1.2 billion stored entries. **Unbuilt
cheaper route:** probe `[k,ω]` on its own with a 2-field colouring at reach 2 (k and ω enter pointwise;
only `ν_t`'s strain-rate dependence needs reach 3), keeping the true k↔ω coupling without materializing
the leading or coupling blocks. The block-diagonal `BlockDiagonal(ScalarTwoLevel, ...)` path
materializes nothing but was insufficient on `bfs3d`.
