---
paths:
  - "aquaflux/precondition/multigrid.py"
  - "aquaflux/precondition/hierarchy_inverse.py"
  - "aquaflux/precondition/entry_map.py"
  - "aquaflux/precondition/padded_rows.py"
  - "aquaflux/precondition/level_fit.py"
  - "aquaflux/precondition/kronecker_pattern.py"
---

# Rules — `aquaflux/precondition/` the traced multigrid (and the record of the deleted monolithic AMG)

This file governs the traced multigrid (`precondition/multigrid.py`, `precondition/hierarchy_inverse.py`)
the padded-row level operator (`precondition/padded_rows.py`), the entry maps a refit applies
(`precondition/entry_map.py`) and the device fit of a hierarchy's level values (`precondition/level_fit.py`).
Package-wide contracts are in `solve.md`; the materialized base `MaterializedJacobianPreconditioner` is in
`solve-direct-preconditioners.md`. Dated measurements and investigation behind every entry are in
`.claude/notes/solve-amg-multigrid-log.md` (never auto-loads).

**Deleted, so do not grep for them as live:** the monolithic PETSc AMG (`amg_preconditioner.py`,
`MonolithicVCycle*`, `monolithic_vcycle_inverse`, `ShiftedCellMajorOperator`, with the `petsc` dependency,
2026-10-10), the zero-fill `Ilu0` / `hostilu` smoother and the elimination orderings (#371), Vanka, the
threshold ILU, the complete LU, `PerFieldNativeInverse`, `_CsrOperator`, `_AGGREGATE_STATS`,
`build_convection_air_hierarchy` and every coupled AMG builder (`coupled_amg_continuation` and kin; the
march is `coupled_step`, see `turbulence.md`). Why each lost: `.claude/notes/solve-refuted-directions.md`
("PETSc ... DELETED as dominated", "Complete LU ...", "Incomplete-LU preconditioning of the pitzDaily flow
block", "Coupled preconditioner options deleted as dominated (#371)", "Low-β directions already measured
out"). Both flagship cases run `MaterializedJacobian(FieldSplit(SimpleSmoothed, JacobiSmoothed))`; the
field split's adjoint gate is `tests/integration/test_coupled_field_split.py`.

## The deleted monolithic AMG — traps that transfer to live code

- **A preconditioner that takes its pattern from the stored sparsity is decided by which entries are
  stored, and at a degenerate state (near-zero fluxes, a closure switch on its kink) rounding decides
  that** — treat a preconditioner measured at such a state as unverified.
- **Two independent implementations agreeing is evidence about a shared cause, not about which shared
  thing it is** (a zero-fill failure blamed on fill was the shared elimination order).
- **Vary interacting knobs together** — stencil reach and fill each "did nothing" alone on pitzDaily and
  together took one matvec.
- **A zero-shift measurement is meaningless without its state** — at pitzDaily's cold start the Jacobian
  is nearly singular; at the converged root the field split converges. Harness:
  `validation/bfs3d_openfoam/zero_shift_adjoint.py`.
- **Nothing cheap computable at build time ranked preconditioner arms** (`condest`, diagonal dominance,
  stationary-sweep contraction, a pivot census; the correlation flipped sign with the shift) — settle it on
  a true-residual solve, and finally on a march.
- **Pressure-last elimination is an ordering for a factorization that keeps fill**; re-ask it only for a
  level-of-fill factorization.
- **Sweep preconditioners at the march's hard states** (checkpoints with high cycle counts or clipped α);
  an easy operator does not discriminate.
- **With a smoothed prolongator a strength threshold gave dense coarse operators and NaN V-cycles; with
  plain aggregation it built a 5-level hierarchy** — do not refute strength-of-connection on
  smoothed-aggregation evidence.
- **Don't under-materialize** — a fresher Jacobian cut a march's Krylov cycles ~23 % despite more refresh
  work; the lever is a cheaper probe, not a rarer one.
- **A cycle-count threshold encodes which preconditioner is installed** — read a `cycles` retry as a
  statement about the threshold before reading it as one about the step.
- **Do not instrument a long run with a per-step notification stream** (it took ~70 % CPU from the solve).
- **March code first recorded here** (cost guard vs shift escalation, `refit_beta_floor` gates, `RefreshTiming`): now `solve-march.md`; originals in the log, "Condensed out of the rule (2026-10-17)".

## The traced multigrid — in progress

**Current status (2026-10-17).** The traced nodal hierarchy (`JacobiSmoothedInverse` over
`HierarchyBlockInverse`) is built, differentiable, and is the `[k, ω]` trailing inverse of both flagship
cases' field splits. On the bfs3d `[k, ω]` block it matched PETSc GAMG's matched-smoother arm (2 restart
cycles). Both case files set `equilibrate: false`; `JacobiSmoothedInverse`'s default is still `True`,
whose original reason (the Frobenius singularity guard) is gone — flipping it is an open shipped-default
decision. The march failures this section once chased belonged to the global `k` positivity cap, deleted
2026-10-14 (`solve-march.md`). Dated record: log, "The traced multigrid — in progress".

- **What closed the gap to GAMG, each found by reading the reference rather than tuning:** one aggressive
  first level over the squared graph (GAMG's default; plain MIS at distance 1 on the squared graph); an
  undamped Richardson level smoother (`spectral_damping=False`: `D⁻¹A` has a unit diagonal, so
  `omega/lam_max` can only under-relax — 10 → 2 cycles); a row-walk level operator instead of a COO
  scatter (now `_PaddedRowOperator`, below); and a correct singularity guard (below).
- **`_cell_block_inverse`'s singularity test is `level_fit.block_singularity`: `|det B|` over the largest
  `∏ᵢ |B[i, σ(i)]|` among permutations σ, invariant under any row or column scaling** (the fields' units
  are arbitrary); refused below 1e-12. Both earlier bars refused invertible `[k, ω]` blocks mid-march: a
  Frobenius bar (rows ~1e8 apart), and a row-norm (Hadamard) bound, invariant under row scaling only, on
  a bfs3d corner cell at `k` ~1e-17 (lower-triangular, det 9.5e-06, ratio 6.8e-16; 2026-10-19). A block
  with no nonzero permutation product (an empty row) scores 0. Cost `b!` passes over the blocks, fine for
  a cell's fields. Pinned by `test_block_singularity_is_unmoved_by_scaling_any_row_or_column` and
  `test_an_invertible_block_with_a_vanishing_variable_is_not_refused`. It decides invertibility only;
  such a block's inverse is ~1e20 in places, which a float32 split then has to carry.
- **Equilibration** (symmetric `D B D`) improves median per-cell block conditioning, so it is kept, but:
  it cannot change whether a block is singular (`det B̂ = 0 ⟺ det B = 0`); it does not help the worst
  cells (it moves the imbalance into the off-diagonal); and it is a similarity transform, so the spectral
  half of the setup (`lam_max`, the damped-Jacobi smoother) is invariant under it while the 0/1 tentative
  prolongation is not.
  - ⚠️ **Float32 needs independent row/column scaling of a cell block** (two-sided gave cond 2.41 where
    symmetric left 1.22e12 on a bfs3d worst cell); unnecessary at float64 for 2×2 blocks.
  - The root cause is upstream: the k and ω rows are ~8 orders apart (8.8e-06 vs 1.4e+03) before any
    preconditioner sees them; a large `‖B⁻¹‖` there is ω slaved to k, not an error.
- **Methodological traps (each produced a wrong write-up):** probe the state the failure happens at (a
  re-fit before a step, not a checkpoint; wrap `refactor_block`, not only the factory); pair the operator
  with the β of the failing refresh, or sweep β; never quote an arm at one smoother-sweep count; a
  block-alone probe ties where a march separates; measure a field split's block on the block alone first
  (the split hides its ~11 %-of-nonzeros trailing half); a default-on guard on a shared seam reaches every
  caller of the seam, so enumerate callers, not files.
- **Harnesses** (`validation/bfs3d_openfoam/`): `trailing_hierarchy_sweep.py` (the block alone, every arm),
  `cell_block_scaling.py`, `trailing_depth_probe.py`; `compare.py`'s `BFS3D_TURBULENCE_INVERSE` and banner
  record the inverse and its settings. Read shared harness records by name, not position (six harnesses
  once died when one grew a field).

## Faithful smoothed aggregation — matching PETSc GAMG

**Current status (2026-10-17).** `multigrid.py`'s aggregation path is a port of PETSc GAMG-AGG: the
aggressive first level, undamped Richardson smoothing, the fix-up pass, magnitude before symmetrization and
the coarse-size stopping rule are built, and the matched arm equals GAMG on the bfs3d `[k, ω]` block. The
investigation, tables and corrections: log, "Faithful smoothed aggregation — matching PETSc GAMG" and
"Faithful smoothed aggregation — the rule body as it stood". Read this before touching the aggregation
path.

- **When a reimplementation of a published, available algorithm underperforms it, read its source before
  running another sweep**, and do not call a port faithful until something has checked it line by line.
  GAMG-AGG defaults: `nsmooths 1`, `aggressive_coarsening_levels 1`, `use_aggressive_square_graph TRUE`,
  random order, `aggressive_mis_k 2`, `graph_symmetrize TRUE` (source: `src/ksp/pc/impls/gamg/{agg,gamg}.c`,
  `src/mat/graphops/coarsen/impls/misk/misk.c`).
- **Compare like with like:** an ours-vs-reference table must carry a matched-smoother row (block Jacobi
  against `pbjacobi`, same sweeps and aggregation); an ILU row measures the smoother, not the coarsening.
  Block Jacobi is a good smoother class here given enough sweeps, at the same CPU cost per apply.
- **Built and kept:** nodal aggregation (`_cell_graph` collapses the dof graph to cells; `_block_tentative`
  gives each field its own coarse unknown) with a block smoother (`_cell_block_inverse`) — all three
  required, no pair suffices; MIS aggregation (`_mis_aggregate`, randomized order, returns its roots, which
  the fix-up pass needs); `prolongation_smoothing` as its own parameter (`"symmetric-part"` default,
  `"standard"`, `"none"` = plain aggregation; unknown raises).
- **The fix-up pass, `_reattach_to_adjacent_root`:** each root, in ascending order, steals distance-1
  members of other aggregates; only members move. Its guarantee is conditional — a member whose neighbours
  are all members keeps its distant root.
- **Magnitude before symmetrization** (`|A_ij|` summed, then symmetrized). A no-op on an M-matrix at
  threshold 0, but not at a threshold: the weights differ, and the coupled flow block runs at 0.25
  (`turbulence/coupled.py`).
- **Stopping rule:** `max_coarse` only fires if `max_levels` has not bound first; the two are not
  equivalents.
- **Still differs from GAMG, unmeasured:** strength-of-connection semantics (GAMG thresholds absolutely on
  the diagonally scaled graph; ours is row-max-relative), so threshold arms are not comparable across the
  two.
- **Refuted / dominated:** QR-orthonormal tentative columns (`orthonormal_prolongation`) are inert at two
  levels and 17–30× worse deeper (ledger, "QR-orthonormalized tentative prolongation"); the standard
  prolongator is under-smoothed at 4 sweeps on the scalar problem, not wrong (best at 8), but both
  smoothed prolongators are refuted on the flow saddle under the SIMPLE smoother
  (`solve-flow-block-log.md`, "Smoothed aggregation on the flow saddle"). Squaring the graph on the first
  level is GAMG's default coarsening, not over-coarsening. Singletons: `avoid_singletons` drops a
  neighbourless vertex from the coarse space as GAMG does (`solve-flow-block-log.md`, "Depth, singletons,
  and the prolongation — where the AMG method itself was wrong").
- **A field-blind aggregation manufactures a negative coarse diagonal** by merging a k-dof with another
  cell's ω-dof; the fine `[k, ω]` slice is clean. Aggregation over the cell graph is robust to explicit-zero
  pruning; a zero-fill factorization is not — ask which consumer sees a pattern change.
- **The coarse solve is a factorization, not a pseudo-inverse (`level_fit.dense_inverse`, traced; the
  host `_dense_inverse` wraps it for lAIR)**: an LU where the reciprocal 1-norm condition of the computed
  inverse clears `n · eps` (the pseudo-inverse's own truncation), the pseudo-inverse otherwise (a
  `lax.cond`, so only the taken branch runs) — a factorization of a numerically singular matrix returns
  finite nonsense rather than raising. ~2× on a whole build, growing with coarse size up to
  `_MAX_DENSE_COARSE_DOFS`. Pinned by
  `test_the_coarse_solve_is_a_factorization_where_the_operator_admits_one`.
- **`JacobiSmoothedInverse` defaults to the matched bundle** (`aggressive_levels=1`,
  `prolongation_smoothing="none"`, `spectral_damping=False`, `equilibrate=True`, `max_coarse=16`). Whether
  it transfers to the scalar-transport and velocity blocks is unmeasured. Still open if more is needed: a
  sparse-LU coarse solve, Chebyshev with the SA-cached bounds.
- **Trailing-block scalability (bfs3d).** The library's two-level dense-coarse default does not scale (a
  77× ratio is ~26k coarse dofs at 1M cells). bfs3d's `case.yaml` sets the trailing `JacobiSmoothed` to
  `max_levels: 20, max_coarse: 200, strength_threshold: 0.25, aggressive_levels: 0, frozen_coarsening:
  true`; depth or threshold alone did nothing, together they converge. `frozen_coarsening` is
  load-bearing: a nonzero threshold reads values, so an unfrozen refresh re-coarsens and retraces (~50 s
  against ~4 s per refresh). Open: validated at 23040 cells only, and `_cell_graph`'s edge weights are not
  normalized per row field (ω rows ~8 orders above k rows), so a raw threshold may act as an ω-only
  measure on a bigger mesh — re-run `trailing_depth_probe.py` there. The field split's whole-Jacobian
  materialize is a separate, likely larger wall (`solve-field-split.md`). GPU smoother cost cannot be
  measured in this (CPU-only) environment.

## Binding decisions — `precondition/multigrid.py`

- **A hierarchy may be held narrow: `hierarchy_inverse._narrowed(levels, equilibration, precision)` rounds
  every floating array except the prolongation values**, which are the very arrays the held
  `FrozenCoarsening`'s plan holds (a rounded copy would sit beside them, not replace them). `HierarchyBlockInverse(stored_precision="float32")` applies it after `derive_extras` on
  every fit and refit (the extras are derived from the double levels first). **It consumes its list**:
  the inverse lets go of the double hierarchy first and each double level is replaced by its rounded copy
  as that copy is made, so the two whole copies are never alive together (pinned by
  `test_narrowing_a_hierarchy_replaces_each_double_level_in_the_list_it_is_given`). It lives beside its one
  user, not on `SmoothedHierarchy`, because `multigrid.py` is held at its size ratchet. The cast helper is
  `traced_cycle.in_precision`, shared with `TracedForm.in_precision`; a leaf already in the target
  type is returned as the same array. ⚠️ **Never round with numpy's cast**: it keeps a single-precision
  subnormal that JAX's conversion flushes to zero, so it is not the traced split's rounding (the coupling,
  the last host-rounded array, is rounded on the device by `in_precision` since 2026-10-22). Why and measurements: `solve-field-split.md`, "the
  split applied in single precision".

- **A level operator is a padded-row (ELLPACK) operator, `_PaddedRowOperator`** (`padded_rows.py`, with
  `_padded_rows`, the one slot rule, and `_PaddedLayout`/`_padded_operator`; moved out of `multigrid.py`
  to keep it under its size budget, 2026-10-18). `columns` / `values` are
  `(n_rows, width)`, `row_lengths` marks stored slots, `apply` is `sum(values * x[columns], axis=1)` (a
  gather fusion then a row-sum kernel, not one fused loop). Measurements on both machines: log, "The
  padded-row level operator — measurements".
  - **The index half is a `PaddedColumns` (2026-10-24, the second stage of `solve-open-directions.md`
    §14d): `columns()`, `row_lengths()` and `entries()` (the stored entry each slot is packed from,
    `nnz` for padding), held (`_HeldColumns`) or computed at cell level (`_KroneckerColumns`, rows only:
    a transpose is packed held, below).** The operator's `rows` and the layout's are the same objects;
    `columns`, `row_lengths` are properties. `_PaddedLayout.of(pattern, ..., fields=(r, c))` holds a pattern at cell
    level when `KroneckerPattern.of_dof` finds it one cell pattern over every field pair, else as arrays;
    the callers that know their fields pass them (`level_plan`: `(block_size, block_size)`; `simple_plan`:
    `F`, `G`, `D`, `M G`, `S`, the block inverse; `FieldSplitInverse.padded_coupling`: trailing × leading).
    The decode is `_padded_rows`'s own slot rule (slot `s` of row `(f, d)`: other field `s // len(d)`,
    place `s % len(d)`, padding at column 0), so every gather, product and row sum is unchanged; pinned
    array for array and the products exactly by
    `test_a_layout_held_at_cell_level_is_the_layout_held_as_arrays`. ⚠️ **The field is counted, never
    divided** (`Σ_t [s ≥ t·len(d)]`): a vector integer division made an isolated apply 1.6× (4×4) to 8×
    (2×2) the held form's; counted, 1.08× / 1.8× / 1.2× (4×4 / 2×2 / 2×4 forward, transposed 1.0 / 1.8 /
    0.9; the transposed decode is deleted since). Inside a loop (a Krylov or V-cycle loop, as the march applies them) XLA hoists the decode out
    and the two forms tie (ten 4×4 applies 168 vs 176 ms, 2×2 25.3 vs 25.4), the decoded columns then
    living as the loop's temporary. bfs3d-sized lattice (23400 cells, reach 3, float32), 4-core Linux,
    jax 0.12.0.1, 2026-10-24 (`solve-direct-preconditioners-log.md`, "Cell-level pattern (§14d), stages
    1–3").
    - ⚠️ **Shape stability:** the cell arrays are padded to the dof budget over the other side's field
      count, and the counts the decode needs (`cell_nnz`, `n_in`) are 0-d leaves, not static: either one
      sized or static by the actual pattern retraced a budgeted rebuild
      (`test_shape_budget_makes_a_repartitioning_rebuild_a_compilation_cache_hit` caught both). The
      layout's own static `nnz` is what `simple_fit` sizes its outputs by. A level whose pattern stops
      being one cell pattern between two rebuilds (a nodal level gaining `landing_with_diagonal`'s unit
      diagonals) changes form and so retraces; a frozen refit keeps its layout and cannot.
    - ⚠️ **Not bit-identical to the held form at default compiler flags, and that is FMA placement, not a
      defect (accepted by the project owner, 2026-10-24).** Values and slots are equal, but a changed
      program contracts multiply-adds elsewhere: pitzDaily's hybrid start moved ≤ 6.2e-15 (the
      potential-flow solve's scalar hierarchy), identical under `XLA_FLAGS=--xla_cpu_max_isa=AVX`; the
      pitzDaily march's step rows and bfs3d's split leaves were unchanged. Pin identity under that flag,
      never at default flags.
    - ⚠️ `SimpleSmoothedInverse._simple_plans` keys its plans on the identity of every leaf of each level
      operator's `rows`, never on `columns` (a computed property, new on every read).
  - **The Galerkin map of a nodal level is held at cell level (2026-10-24, the third stage of §14d).**
    `level_fit._galerkin` takes `kronecker_pattern.KroneckerGalerkin` when the level stores one cell
    pattern over every field pair and `nodal_cell_transfer` finds `P = I ⊗ T` (a nodal aggregation's
    tentative prolongation; both shipped blocks run `prolongation_smoothing: "none"`): it builds the
    cell product `Tᵀ C T` once, holds its terms (source, source row, target; sorted by target) and
    places every field pair by `entry_position` on both sides, summing each target's terms in source
    order as the dof map's segment sum does. Pinned bit for bit against `EntryMap.product`'s map, landing
    and values, by `test_the_cell_level_galerkin_map_is_the_degree_of_freedom_map_exactly`; any other
    level keeps `EntryMap.product`. `_LevelPlan.galerkin` is a `ValueMap`.
  - **The transpose is packed only for an adjoint (2026-10-24, §14d's lazy transposes), and applied as a
    gather** through `jax.custom_derivatives.linear_call`, so `jax.linear_transpose` gives a forward
    product, not a scatter-add over every slot. A fit and a refit hold rows alone (`transposed`,
    `t_values` are `None`: no leaves, no compile shape); `_PaddedRowOperator.with_transpose()` packs `Aᵀ`
    on the host from `to_scipy()` by `_padded_rows` (held columns, the widest column, the values' own
    precision), and **`solve.with_transposes(tree)` asks every `solve.PacksTranspose` leaf of a pytree**.
    `root_adjoint` applies it to the adjoint's preconditioner factory at the start of each transpose
    solve, so the copy lives only as long as that solve and reaches every residual (traced split, single
    block, block-SIMPLE, scalar transport); the host inverses' `apply(transpose=True)` packs into their
    lazy `_transpose_fn`, dropped at a refit. **Transposing an operator without its transpose raises
    `RuntimeError` naming `with_transposes`**, never a scatter fallback. ⚠️ An adjoint factory must hold
    its operators as pytree leaves, not close over them (`BlockPreconditioner.factory()` and
    `bordered_preconditioner` return modules for this reason); a closure raises. ⚠️ Reverse mode through
    a preconditioned `solve_linear` outside `root_adjoint` (`newton_step` under `jax.grad`) is transposed
    by lineax, which asks nothing: that caller passes its factory through `with_transposes` itself
    (`test_preconditioned_solve_is_differentiable`). ⚠️ Packing needs concrete values (it raises under a
    trace); `root_adjoint`'s backward rule runs eagerly under `jax.grad`. `apply` has **no `jax.vmap`
    rule** — nothing vmaps a preconditioner apply today. Pinned by
    `test_a_padded_row_operator_transposes_by_gathering_its_stored_transpose`,
    `test_a_padded_row_operator_holds_no_transpose_until_asked_and_then_packs_its_own_entries` and
    `test_a_fitted_split_holds_no_transposes_and_an_adjoint_packs_them_for_its_solve_only`. Saved (both
    case files' configuration, CPU, jax 0.12.0.1, 2026-10-24): pitzDaily march identical row for row,
    peak 3.085 → 2.900 GiB; bfs3d: `solve-direct-preconditioners-log.md`, "Lazy transposes (§14d)". The
    adjoint pays the packing instead: 22.9 s on bfs3d, 19 transposed applications, ~1.3 % of a
    transpose solve (`adjoint_transpose_cost.py`, 2026-10-25).
  - ⚠️ **On macOS the row-sum kernel (a single-threaded `__ynn_fusion`) is the bottleneck for wide rows**
    (bfs3d, up to ~364 wide), and no rewrite of the sum avoids it: `einsum`, a chunked sum, a column-major
    layout, a reduce-free halving sum and an unrolled sum were all measured slower. The XLA flag
    `--xla_cpu_experimental_ynn_fusion_type=-reduce` fixes the Mac and is neutral-to-worse on Linux, so it
    cannot be global; float32 (shipped by both cases, `solve-field-split.md`) fixes the Mac's forward
    apply. Narrow rows (pitzDaily, ≤ ~85) win on both platforms. Kernel cost is machine-dependent:
    `validation/pitzdaily_openfoam/csr_kernel_precision.py` and `split_apply_timing.py` time the forms.
  - The shape budget is a width per coarse level (`ShapeBudget.operator_width`); `from_scipy(a, width=)`
    refuses a row wider than the budget rather than dropping entries; `_require_matching_structure`
    compares widths (the compilation signature). Padding costs `n_rows * width` (~1.1× on pitzDaily's split,
    ~1.3× on bfs3d's); an operator with a few very wide rows would pay that width everywhere.
  - `to_scipy` returns exactly the stored sparsity, explicit zeros included, padding excluded; `nnz` is
    the stored-entry count. Pinned by `test_a_padded_row_operator_reproduces_the_matrix_it_froze` and
    siblings.
- **Reading a frozen level back to the host has one home per array:** `_PaddedRowOperator.to_scipy()`
  (inverse of `from_scipy`) and `_SparseLevel` / `_AirLevel.prolongation_scipy()` (per class:
  `_SparseLevel` stores a COO triple doubling as the transposed restriction, `_AirLevel` an independent
  CSR prolongation and restriction).
- **`multigrid.py` is a pure operator-coarsening library — operator in, uniformly.** Every builder
  (`build_smoothed_hierarchy`, `build_convection_hierarchy`, `build_air_hierarchy`) takes an assembled
  `a: sp.csr_matrix`, never a mesh, edges or flow quantities. Assembly is
  `frozen_operator.ConvectionDiffusionStencil(owner, nb, coefficient, n, *, flux=None,
  boundary_diagonal=None).assemble()` (plus `decouple_dof` for a pressure pin), the one assembler for all
  four consumers. Do not reintroduce an `(owner, nb, coeff, …)` signature or a second assembler. The
  first-order-upwind stencil is the preconditioner's choice (an M-matrix aggregation can coarsen),
  whatever the residual's advection scheme.
- **Aggregate-size diagnostics are an explicit output parameter**: `_build_aggregation_hierarchy` and its
  public wrappers take `stats: list[dict] | None`. `HierarchyBlockInverse._coarsen` owns a fresh list per
  build and keeps it on the inverse (`self._aggregate_stats`), never on the hierarchy — a hierarchy is a
  jit argument, and a varying field there would retrace (static) or cannot be traced (a list of dicts). A
  frozen-coarsening refresh reports this inverse's own last aggregation. Pinned by
  `test_a_frozen_coarsening_refresh_reports_its_own_last_aggregation_not_a_strangers`.
- **The V-cycle recursion (`_frozen_v_cycle`) and its fixed-cycle driver (`_fixed_cycles`) are written
  once**; a family contributes only its `_VCycleOps` (restriction, prolongation, smoother) and a thin
  entry point. The fixed outer loop is what makes `b -> x` a constant linear operator, which the frozen
  preconditioner and the adjoint transpose depend on.
- **A level is static indices + traced values, so a refresh is a jit cache hit.** In `_SparseLevel` /
  `_AirLevel` only `n` and `n_coarse` are static; everything else, `lam_max` included (a 0-d array, not a
  float), is a traced leaf. This holds because aggregation at `strength_threshold=0` is a pure function of
  the graph. Pinned by `test_aggregation_hierarchy_structure_is_value_independent` and
  `test_refreshing_a_hierarchy_is_a_compilation_cache_hit`.
  - ⚠️ **scipy drops an entry that sums to exactly zero from a sparse product or sum**, and closure
    switches zero `[k, ω]` derivatives, so shapes moved with values until every such site stored its result
    on its structural pattern: the Galerkin product (`_galerkin_coarse`, pattern from the same product over
    ones), the threshold-0 aggregation graph (`_ones(graph)`), and smoothed prolongations
    (`_smoothing_pattern`). Do not "simplify" any back to a bare product. A nonzero threshold still reads
    values, so a rebuild there is shape-stable only under `frozen_coarsening`. Pinned by
    `test_a_nodal_hierarchy_keeps_its_shapes_when_a_coupling_becomes_exactly_zero`,
    `test_a_threshold_free_mis_coarsening_keeps_its_shapes_when_a_coupling_becomes_exactly_zero`,
    `test_galerkin_coarse_pattern_does_not_move_when_a_weight_becomes_zero`; harness
    `validation/traced_split_refresh_shapes.py`.
  - **A frozen coarsening's level values are fitted on the device, by one function for the first build
    and every refit (`level_fit.py`, 2026-10-22).** The host decides structure (aggregates, prolongations,
    every level's pattern) and holds it as a `CoarseningPlan` of `_LevelPlan`s, each carrying device entry
    maps (`entry_map.ValueMap`, a `DeviceEntryMap` or a cell-level map): the padded-row packing
    (`_PaddedLayout`), the diagonal and cell-block gathers, the Galerkin map
    `Pᵀ A P` to the next level's values, the coarsest level's dense gather, and the start vector of the
    spectral estimate. `fit_levels(plan, values, scale)` (one `eqx.filter_jit`) derives from the fine
    operator's stored values every level's operator, diagonal, cell-block inverse (`invert_cell_blocks`,
    with the singularity count), `lam_max` (20 power steps on `D⁻¹A`, `spectral_estimate`) and coarse
    inverse, plus `LevelChecks` the host reads back once and raises on (the messages of
    `_require_positive_diagonal` / `_refuse_singular_blocks`). `multigrid.FrozenCoarsening(plan, pattern)`
    is the host face: `fit(a, where)` (`a` a scipy matrix or an `entry_map.DeviceMatrix`, a host
    `SparsePattern` with device values, fitted where they are; `HierarchyBlockInverse.refactor_block`
    passes either through and reads one back, `as_scipy`, only to re-coarsen) checks shape and pattern ("cannot refit ..."), re-derives an
    equilibration scale by the host rule (`equilibration_scale` of the device diagonal, one round trip),
    fits, and assembles the `SmoothedHierarchy`. `_build_aggregation_hierarchy` plans level by level
    (`level_plan`; a budget's unit diagonals on empty coarse slots by `landing_with_diagonal`, applied as an
    embed plus a constant), carries the coarse values with the plan's own Galerkin map, and ends in
    `fit`; the builders hand the coarsening back through `coarsening: list | None` (like `stats`), and
    `HierarchyBlockInverse` keeps it (`self._coarsening`) and refits by `self._coarsening.fit(matrix)`.
    There is no `SmoothedHierarchy.refit`, `_sparse_level` or `_spectral_radius`.
    - **Why**: a refit moved every level through the host (scipy products, packing, LU, power iteration),
      which on a GPU is a transfer per piece; now a refit uploads the fine values once and reads back the
      checks. Prolongation smoothing's `lam_smooth` (`matrix_spectral_estimate`) and the "standard"
      singular value stay host-side: they decide structure, at build only.
    - ⚠️ **Not bit-identical to the old host fit**: the Galerkin map sums each coarse entry's terms in
      another order (bounded by `2 · n_terms · eps · Σ|w_j a_j|`, `tests/unit/test_entry_map.py`), and the
      LU, power iteration and block inverses are JAX's. The build and a refit share the function, so a
      refit at the built operator reproduces the build exactly.
    - **Cost** (pitzDaily, CPU, 2026-10-22): same march; steady refactor ~0.7 s against ~0.83 s; the first
      fit ~6-8 s dearer (map building moved from the first refit, plus the fit program's per-process
      compilation); memory held +29 MiB, but peak resident +0.06-0.32 GiB, freed memory XLA's worker-thread
      arenas keep (`malloc_trim` returns it). A compiled function's outputs are new buffers, so the fit
      returns slot values and the operator is assembled on the layout's own index arrays -- keep it so.
      Log, "The device fit (2026-10-22)".
  - **Entry maps (`entry_map.py`)**: on a pattern that does not move, every sparse operation linear in one
    varying operand is a fixed map `out_t = Σ w_tj a_j`: `EntryMap` stores it as (source entry, target
    slot, weight), applies it on the host as one gather/multiply/`bincount`, and `on_device()` gives a
    `DeviceEntryMap` (a gather with a sentinel slot when every target takes one unit-weight source, else a
    target-sorted `segment_sum`). Two builders: `EntryMap.gathering(pattern, operation)` runs an
    entry-moving operation once over `SparsePattern.numbered()` (cuts, transposes, `_padded_rows`,
    `cell_blocks` -- so each operation's index rule keeps its one home), and `EntryMap.product(L, pattern,
    R)` gives `M -> L M R` for fixed `L`, `R` onto the product's structural pattern (the same set
    `_galerkin_coarse` stores on). `EntryMaps` (held by `FieldSplitInverse._maps`) keys one gathering map per slot and rebuilds it when the operand's pattern
    moved; `EntryMap.apply` itself refuses an operand of another pattern. **A pattern is held once however many owners
    read it**: `SparsePattern.of` returns the live pattern matching the operand by content (`matches`),
    used by `EntryMaps.pattern_of`, both landings and the coarsening's plan, since the split cuts a block
    and its hierarchy then fits a copy of it (a scaling or shift copies the index arrays, so identity alone
    misses it). Pinned by
    `test_two_owners_reading_one_pattern_hold_one_copy_and_another_pattern_is_never_given_it`. Host-wired: `FieldGroups.blocks(maps=)` and
    `FieldSplitInverse.padded_coupling()` (`traced_field_split`). The SIMPLE pieces are fitted on the
    device from their own plan (`solve-flow-block.md`).
    - **Why the Galerkin map is linear on both shipped blocks:** both run `prolongation_smoothing: "none"`
      (the class default; neither case file sets it), so `P` is the aggregates' 0/1 indicator, and a refit
      reuses the held `P` in any case. A smoothed `P` would still give a linear map at refit (the refit
      freezes `P`); only a rebuild re-smooths it.
    - Index arrays are int32, all-ones weights and in-order sources/targets are not stored, a host gather
      filling its slots in ascending order (every `placing` map) holds a byte-a-slot mask in place of its
      targets (`EntryMap._filled`; with the shared patterns, bfs3d's peak 9.25 -> 8.75 GiB, log, "Shared
      patterns and filled-slot masks (2026-10-22)"), and the numbered pattern is not kept -- keep it so. **`EntryMap.product` is built to hold one term list at a
      time** (no all-ones weight list, the slots by a sorted copy of the keys and `searchsorted`, not
      `np.unique`'s inverse): the leading block's fine Galerkin map set bfs3d's peak, 9.67 -> 8.71 GiB with
      the maps' terms unchanged in content and order (log, "The product map's build transient
      (2026-10-22)"; pinned by `test_the_product_map_holds_the_terms_of_the_plain_construction_in_its_order`).
      Host-map measurements: log, "Refit by entry maps (2026-10-17)"; harness
      `validation/pitzdaily_openfoam/refit_profile.py`.
- **A coarsening that reads no values is refitted, not re-aggregated** (`HierarchyBlockInverse.
  refactor_block`, `_coarsening_reads_values()`): at `strength_threshold` 0, `prolongation_smoothing`
  `"none"` and no `shape_headroom`, a rebuild provably reproduces the held aggregates and `P`, so the
  refresh refits whether or not `frozen_coarsening` is set. pitzDaily's trailing `JacobiSmoothed` is this
  case (it re-aggregated at every refit before). The refit equals a fresh build to the rounding above.
  Pinned by `test_a_coarsening_that_reads_no_values_is_refitted_rather_than_aggregated_again`.
  - **Strength-of-connection aggregation is opt-in (`strength_threshold`) and makes aggregation
    value-dependent** (it forfeits the free refresh): edges kept iff `|A_ij| ≥ θ·max_k|A_ik|` from either
    endpoint (`_aggregation_edges`). It keeps a V-cycle contracting on anisotropic operators (flat 3 cycles
    at every aspect ratio, `test_strength_of_connection_aggregation_fixes_an_anisotropic_operator`). `θ=0`
    is the default and byte-identical to isotropic; the coupled flow block runs 0.25 (frozen, so no refresh
    cost). The refreshed scalar k/ω AMGs stay at 0 to keep their cache hit.
- **⚠️ THE lAIR RESTRICTION WALKED THE OPERATOR'S FULL SPARSITY PATTERN — fixed: it walks the strength
  graph** (Manteuffel, Ruge & Southworth, SISC 40(6), 2018), which bounds coarse fill; the old walk
  compounded fill until the build did not finish on any real operator. The neighbourhood is a knob,
  `build_air_hierarchy(restriction_theta=…)` (default `theta`): wider buys contraction and costs density.
  `restriction_theta=0.0` reproduces the full walk and is the only setting for which the exact-Schur
  property holds (`test_lair_restriction_reproduces_the_exact_schur_complement`).
  `_require_bounded_coarsening` refuses a hierarchy growing stored entries per row by more than
  `_MAX_COARSENING_GROWTH` above `max_coarse`, because densification has no symptom but a build that does
  not finish. Table and the recalibrated contraction tests: log, "The lAIR restriction walk".
  - **lAIR is block-aware (`block_size=`)**: the C/F split on the cell graph, one right-hand side per field
    in the local solve, and a cell-block FC-Jacobi smoother — all three needed. A block smoother needs only
    invertible cell blocks, not a positive scalar diagonal, which is why it builds where the scalar build
    refuses (`test_block_air_builds_where_the_scalar_diagonal_is_negative`). `block_size=1` is
    byte-identical.
  - **`AirReductionInverse` (`field_split.py`)** wires it as a trailing inverse (`BFS3D_TURBULENCE_INVERSE=air`);
    not a `HierarchyBlockInverse`, and its transpose is `jax.linear_transpose` of the fixed-cycle map
    because `R != Pᵀ`. Whether lAIR is any good on bfs3d is unmeasured; it is opt-in everywhere
    (`ScalarAir`, `ConvectionAir`).
  - **`refresh_air_hierarchy(hierarchy, a_new)` refreshes values on a frozen coarsening**: it reuses each
    level's C/F split, prolongation and restriction neighbourhoods (`_frozen_neighbourhoods`, recovered from
    the stored `R`) and re-solves only the local restriction — all three are value-dependent under the
    strength-graph walk, so all must be carried. `_require_matching_structure` still checks the result.
    On a uniformly scaled operator a plain rebuild is shape-stable too; the refresh matters when the
    convection/diffusion balance moves. Pinned by
    `test_refresh_air_hierarchy_keeps_the_structure_and_is_a_cache_hit` and siblings.
- **Degenerate-mesh guard, at build time:** `frozen_operator.require_valid_graph` runs when a
  `ConvectionDiffusionStencil` is constructed, and every level's diagonal is checked when it is fitted (the
  aggregation build through `FrozenCoarsening.fit`'s `LevelChecks`, lAIR's loop by
  `_require_positive_diagonal`; after boundary stiffness is folded in) — which is why the runtime smoothers need no
  per-apply floor. A nodal level (`block_size > 1`) skips the scalar check; only its cell blocks need be
  invertible.
- **`_coo_apply(sorted_segments=)` is a promise, not a hint** — a backend acting on a false one returns
  wrong output. The smoothed-aggregation restriction (the prolongation transposed, segments = its column
  array) passes `False`; lAIR's independent CSR transfers are sorted. Level operators do not go through
  `_coo_apply`.
- **Frozen level diagonals are inverted once, in `__post_init__`, through one shared
  `_level_inv_diagonal(diagonal, block_inverse)`** for both level kinds, which guards the nodal case (a
  nodal level's scalar diagonal may be zero; its smoother reads `block_inverse`). `smoother_weight` stays a
  solve-time argument. Pinned by `test_block_sparse_level_stores_a_finite_inv_diagonal_on_a_zero_scalar_diagonal`.
- **The damped-Jacobi convection hierarchy is TWO-LEVEL by design (library default,
  `_CONVECTION_LEVELS = 2`):** a smoothed fine level and a dense direct coarse solve. A deeper Galerkin
  recursion over a structural (value-blind, threshold-0) coarsening gives coarse-of-coarse operators with
  near-imaginary eigenvalues no single-factor damped-Jacobi smoother can damp (ρ(S) ≈ 1.0–1.36). Depth with
  a strength threshold does not reproduce it (bfs3d's trailing bundle above, a case-level override). Deep
  threshold-free convection coarsening is lAIR's job.
