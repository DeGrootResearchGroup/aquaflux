---
paths:
  - "aquaflux/precondition/sparse_jacobian.py"
  - "aquaflux/precondition/host_preconditioner.py"
  - "aquaflux/precondition/materialized_preconditioner.py"
---

# Rules — `aquaflux/precondition/` the frozen-host contract and Jacobian materialization (and the record of the deleted direct preconditioners)

This file governs three live modules: the frozen-host contract (`host_preconditioner.py`), the shared
materialized base (`materialized_preconditioner.py`) and the coloured-probe Jacobian
(`sparse_jacobian.py`). Package-wide contracts are in `solve.md`. Dated measurements and the
investigation behind each entry are in `.claude/notes/solve-direct-preconditioners-log.md` (never
auto-loads).

**Deleted, so do not grep for them as live:** the complete LU (`CompleteLu*`, `lu_preconditioner.py`,
2026-10-11) and the PETSc monolithic AMG (`MonolithicVCycle*`, `amg_preconditioner.py`,
`ShiftedCellMajorOperator`, 2026-10-10). Why each lost: `.claude/notes/solve-refuted-directions.md`,
"Complete LU (SciPy SuperLU) — DELETED as dominated" and "PETSc ... DELETED as dominated"; the LU tests'
assertions live on in `tests/integration/test_coupled_field_split.py`. The coupled march is one builder,
`coupled_step`, given `BlockDiagonal` or `MaterializedJacobian(FieldSplit | BlockInverse)`; the rename
table for the old continuation builders is in `turbulence.md`.

## Preconditioner — the frozen host family (shared contract)

- **`equilibrate_cell_major` and `cell_major_permutation` live in `frozen_operator.py`, unexported.** They
  are the reorder half of the transform whose rescale half (`symmetrically_equilibrate`,
  `equilibration_scale`, `apply_symmetric_scale`, `row_chunks`) sits beside them. No library code uses
  them since PETSc went; they remain for the study harnesses (`cell_block_conditioning.py`,
  `column_reach_zero_pruning.py`, `host_ilu_levels.py`, reached via `test_solve_api.py`'s
  `VALIDATION_INTERNAL_REACHES`) and are a deletion candidate if those go. There is no elimination-ordering
  family (deleted with the zero-fill ILU, #371).
- **`HostPreconditioner` owns `__init__` and `matvec()`; each subclass supplies only `build` /
  `refresh_in_place`.** Fitting genuinely differs per subclass and is deliberately not unified behind a
  union signature; applying does not.
- **`FrozenInverse` is exactly `n_dofs` + `apply(residual, *, transpose=…)`**, satisfied structurally by
  `HierarchyBlockInverse` and both `FieldSplitInverse`s. `MaterializedJacobianPreconditioner`
  (`materialized_preconditioner.py`) is the shared base for fitting any inverse to the materialized
  Jacobian (the probe, the shift-diagonal add, teardown); its subclasses are `FieldSplitPreconditioner`
  and `MaterializedBlockPreconditioner`.
  - ⚠️ **Anything a base reads off `self.inverse` beyond that pair is a requirement on all of them.** A
    capability flag once read off only one inverse class raised inside a property on the others, and
    `getattr(pc, ..., False)` swallowed the `AttributeError`. Answer a non-contract capability on the
    subclass. Pinned by an AST check of the base's source
    (`test_the_base_asks_its_factors_for_nothing_beyond_the_declared_contract`), because the lookup is on
    paths a test would not drive.
- **Capabilities only some inverses have are `runtime_checkable` protocols asked by `isinstance`
  (`host_preconditioner.py`, exported):**
  - `RefactorableInverse(FrozenInverse)`: `refactor_block(block)`, re-fit in place to the raw field-major
    operator of the built shape. One spelling across the family — a split is itself refactorable (there is
    no `split.refactor`). `require_refactorable(factors, owner)` raises a `TypeError` naming the inverse;
    a split checks both blocks before re-fitting either (no half re-fit) and refuses a non-`FrozenInverse`
    block at construction.
  - `ReleasableInverse`: `destroy()`, asked only by `release(inverse)`, which `HostPreconditioner.destroy()`
    calls, so every member releases its inverse.
  - The traced capability is `OffersTracedCycle` / `TracedCycle` in `traced_cycle.py`; there is no
    `TracedFactors` / `apply_traced`.
  - Factory types say so: `leading_inverse` / `trailing_inverse` / `inverse` are
    `Callable[[sp.csr_matrix, int], FrozenInverse]`; `BlockInverse.__call__` returns `RefactorableInverse`.
  - Pinned by `test_each_inverse_declares_exactly_the_capabilities_it_has`,
    `test_destroying_a_preconditioner_releases_its_inverse_when_it_holds_anything` and the split's
    refit/atomicity tests in `test_field_split.py`.
- **`refresh_in_place(..., diagonal_correction=)`**: a callback given the unshifted Jacobian's diagonal
  that returns a per-row amount added to the shift before the fit (or `None`); it changes the fit only,
  never the operator solved. One user: the pivot floor (`solve.md`, `PivotFloor`). Default `None` is
  byte-identical. **Both re-fit paths add the shift through `_fitted_shift`** (the whole matrix, and a
  field split probed straight into its blocks, which never forms it and reads the diagonal through
  `ProbedBlocks.diagonal()`); a test observes the fit by spying on it. ⚠️ A correction wired on the
  whole-matrix path alone silently never reaches a field split, the only inverse the step cases ship
  (#763 merging #766). Pinned by `test_a_diagonal_correction_reaches_the_split_s_fit_on_either_refit_path`.
- **`refresh_in_place` returns one shape across the family**: the `("probe", s), ("assemble", s),
  ("refactor", s)` tuple. Every refreshed preconditioner is a `MaterializedJacobianPreconditioner`, so the
  session passes `batched_matvec`, `probe_batch_size` and `structure` unconditionally.
- **The pseudo-transient shift has one home: `sparse_jacobian.shifted_jacobian`, which adds `β d` by
  pattern-preserving `setdiag`.** Never `a + sp.diags(shift)`: a sparse addition drops the explicit zeros
  a fixed-pattern probe deliberately stores.

## Materializing the coupled Jacobian (`sparse_jacobian.py`)

The coloured-probe materialization of the coupled Jacobian from the same residual the solver uses (no
re-derived assembly), consumed by the materialized-Jacobian multigrid inverses (the field split and a
bare block inverse).

- **API.** `block_stencil_colouring(owner, nb, n, reach)` (pure NumPy: the cell-block pattern at a stencil
  `reach` and a collision-free CPR colouring whose conflict graph is the pattern squared), then
  `materialize_block_jacobian(matvec, plan)` (one `jax.jvp` per colour × column field). Both take a
  `ColumnProbePlan`: `ColumnProbePlan.uniform(colouring, n_fields)` for one reach, `column_probe_plan(owner,
  nb, n, column_reach, pattern_reach)` for a reach per column (built by `jacobian_probe.py`).
  `block_stencil_gather_map(plan)` returns a `ProbeGather`. Field-major DOF layout `(cell i, field f) =
  f·n + i`. The coupled RANS residual reaches distance 3 on an unskewed mesh; `jacobian_relative_error`
  checks the reach covers it.
- **The colouring is by saturation degree** (`_saturation_colouring`, vertex selection by a bucket queue,
  `O(n + m)`): fewer colours means fewer probes, and any collision-free colouring de-compresses to the same
  matrix, so changing the colouring changes cost, never the answer. Ties break by recency of eligibility,
  so colourings differ from a full-scan argmax but stay valid (`test_saturation_colouring_*`). Colour
  counts and the bucket-queue crossover (slower than the scan below ~100k–200k cells, faster above):
  log, "Saturation colouring and its bucket queue".
- **A probe feeding a field split is laid out by the split's blocks** (`block_stencil_gather_map(plan,
  leading_fields=)` -> `ProbeGather.layout`, a `BlockLayout`; `materialize_blocks` -> `ProbedBlocks`):
  one block-major value array, each block a CSR view of its slice (assigned, not passed to scipy's
  constructor, which copies a view), the shift added in place at the stored diagonal (refused if one is
  missing). `materialize_block_jacobian` refuses such a structure; `refresh_in_place` takes the block
  path when `structure.layout` is set and calls `_refactor_blocks`, which only `FieldSplitPreconditioner`
  implements.
- **A materialize holds the matrix plus one chunk.** Seeds are filled one chunk at a time into a reused
  buffer (`ColumnProbePlan.seed_block(start, stop, out=)` for whole-state seeds, `field_seeds` for one
  field's; one shape per program, so each batched map compiles once), and **a chunk never spans two
  column fields** (`ColumnProbePlan.field_chunks`; each field's last chunk may be short and is padded, so
  a materialize makes up to `n_fields − 1` more calls than `ceil(n_probes / batch)`; `probe_batch_size=None`
  is one batch per field),
  and `ProbeGather` scatters each chunk's responses as they are computed; out-of-reach entries need no
  sentinel. The gather map's one-time build (`_index_dtype`, COO→CSR grouping) runs at case construction
  and peaks above any single materialize. Measurements: log, "Chunked seeds and per-chunk
  de-compression" and "The gather map's build peak".
  - **Batch size:** `PROBE_BATCH_SIZE = 8` (`materialized_session.py`), where processor time bottomed on
    bfs3d (M3 Pro, 2026-08-11) and where the per-tangent time bottoms on bfs3d on a 4-core Linux
    container in every allocator arm (2026-10-18, `refit_profile.py` `REFIT_JVP_BATCHES`; table in
    `.claude/notes/validation-log.md`, "Why bfs3d's probe tangent costs 5.5× pitzDaily's"): no evidence
    the optimum is a machine property. The 2026-08 sweep's peak grew ~2.5 MB per unit of batch, so
    auto-sizing the batch to free memory was judged not worth building; that reading predates a census
    of the compiled program, whose temporary is ~100 MiB per tangent plus ~100 MiB on bfs3d (816 MiB at
    8, Linux, 2026-10-18), transient per call and faulted in afresh at glibc defaults. A run's peak at
    batch 16 against 8 is unmeasured. On bfs3d a run's memory ceiling is set by the fits and re-fits of the materialized
    Jacobian and the compiled step, not by building the case (0.84 GiB): 8.46 GiB over the shipped 17-step
    march against 12.25 GiB on `main` `1b68caf6`, identical step tables (2026-10-19, 4-core Linux, jax
    0.12.0): log, "bfs3d memory by owner" and "Where a run's memory goes".
  - **One program per column field (2026-10-20).** A problem whose assembler implements
    `residual_with_field(state, field, values)` (`FieldwiseResidual`) sets `probes_by_field = True`
    (`_CoupledProblem`, `_FlowProblem`); the session (`_probe_products`, used by both the fit and the
    re-fit) then passes a `FieldwiseMatvec` wrapping `batched_field_jacobian_matvec(assembler, layout,
    state, field, tangents)` — `field` static, one compiled program per field, seeds `(batch, n_cells)`.
    The other fields are closed over as constants, so JAX never forms their tangents (a whole-state seed
    is zeros, which are values: every kernel downstream ran on them). The default (`False`) is the
    whole-state `batched_jacobian_matvec`. The probe count is unchanged; what each tangent computes is
    not. ⚠️ **The pruning dies silently if the residual repacks its fields into a flat state** anywhere on
    the path (the repack's derivative instantiates every zero), and a constant-zero seed is no symbolic
    zero either; verify on the jaxpr (`test_fieldwise_probe.py` does).
    - **Not bit-identical at default compiler flags, and that is rounding, not a defect.** On an x86 CPU
      with fused multiply-add, removing an add of zero changes where XLA contracts multiply-adds: per-field
      against whole-state matrices differ by ≤ 2.4e-16 of each (row field, column field) block's largest
      entry, on 2–8 % of entries, none non-finite (bfs3d and pitzDaily hybrid starts, jax 0.12.0, 4-core
      Xeon, 2026-10-20). Under `XLA_FLAGS=--xla_cpu_max_isa=AVX` (no FMA instructions) the two are
      **identical** (`np.array_equal` on `data`/`indices`/`indptr`) on both cases. Instrument:
      `validation/fieldwise_probe_identity.py`. ⚠️ The batched whole-state product was never bit-identical
      to a per-tangent loop either (up to 2.7e-11 relative on small entries, pitzDaily, same flags and
      without FMA alike), whatever its old docstring said: never pin a probe change on exact equality at
      default flags.
    - **Ready for per-column precision, not built:** `FieldwiseMatvec.seed_dtype(field)` (float64 for
      every field today) types each field's seed buffer and the tangents' dtype keys each program; a
      float32 field program must also cast the closed-over state and assembler.
    - **Measured (4-core Xeon, jax 0.12.0, glibc defaults, hybrid start, batch 8):** bfs3d's per-field
      programs touch 2.8 (p) to 6.9 GB (each velocity) per call against 10.4 for the whole state, mean
      27.5 ms per tangent against 59.3; a re-fit's probe 43.1 → 18.5 s (β 0.005). pitzDaily 12.1 → 5.5 ms
      per tangent, probe 1.60 → 0.94 s. bfs3d's march (owner's arm64 Mac, 2026-10-21): same 17 steps / 149
      cycles / `x_r/h` 8.361, probe over the march 125–132 → 71 s warm. The price: ~10 s per process to trace and lower the extra programs
      (pitzDaily, warm cache), and +0.4 GiB VmHWM after a bfs3d re-fit. Tables and marches:
      `.claude/notes/validation-log.md`, "One probe program per column field".
  - **Batched probing** (`batched_matvec`, a `jax.vmap` of the jvp built once) and the **fixed full-pattern
    gather** (`structure=`) are what the materialized path passes. The full pattern (no `eliminate_zeros`)
    is a superset of the live nonzeros at any state, so the structure never moves cold→developed — that
    fixed structure is the gather's real value (in-place refits need it), not its wall clock.
- **Probe colouring reach is a per-case choice, set by measurement; the pattern stays at reach 3.**
  pitzDaily probes every column at reach 2 inside a reach-3 pattern (`column_reach: [2, 2, 2, 2, 2]`,
  2026-10-14: same 18 steps / 123 cycles / `x_r/h` 8.0686, probe per refit 4.3–5.8 → 2.5–3.4 s; at zero
  shift 7 % more forward applies, none transposed). bfs3d keeps uniform reach 3 because the same colouring
  cost it 34 % of its wall in one run per arm -- a single draw from a march whose step count spread 22–82 under last-bit perturbations (2026-10-17/18) (`validation.md`) -- (measured by the project owner, PR #722, `validation/bfs3d_openfoam/compare.py`
  with `BFS3D_COLUMN_REACH=2,2,2,2,2,2`, shipped `case.yaml`, `77b03c18`, Apple M3 Pro, jax 0.10.2):

  | probe colouring (pattern reach 3) | steps | restart cycles | final `\|R\|` | refit (probe / refactor) | solve |
  |---|---|---|---|---|---|
  | uniform reach 3 (shipped) | 18 | 137 | 5.1e-06 | 11.9–12.7 s (8.2–8.5 / 3.6–4.2) | 522 s |
  | reach 2 | 21 | 216 | 5.0e-06 | 6.0–6.7 s (3.4–3.6 / 2.4–3.0) | 697 s |

  The shipped arm on a 4-core Linux container (Xeon 2.1 GHz, jax 0.11.2, `b1648725`, glibc defaults,
  cold cache, 2026-10-17): 29 steps / 173 cycles / 3.7e-06 / re-fit 56–64 s (probe 37.7–41.8 / refactor
  16.9–20.5) / 3155 s, `x_r/h` 8.3611, peak 12.1 GiB — a different march (rounding), the same root; the
  re-fit is 58 % of it. Why the probe is slow there: its batched JVP touches 4.4x pitzDaily's bytes per
  tangent at ~23 GB/s on 2-way-partitioned kernels, and faults its 816 MiB temporary in afresh every call
  (~a quarter of the probe); `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` removes the faults at an unchanged
  footprint (re-fit 48.7 → 39.7 s at `state-00032`; a cadence-5 march without the pivot floor 2439 → 2180 s, peak 12.63 → 12.15 GiB), the retaining allocators are not usable for a march
  in 15 GiB: `.claude/notes/validation-log.md`, "bfs3d on a 4-core Linux container" and the subsection after it.

  What decides it is the far mass the colouring folds (`probe_reach_audit.py`: bfs3d's p carries 1.1e-3 of
  its norm beyond reach 2, against ~4e-4 folded on pitzDaily). Untested: shortening only some of bfs3d's
  columns. pitzDaily arms and the reach-4 pattern: log, "Probe colouring reach 2 — pitzDaily and bfs3d".
- **Keep the materialized pattern at reach 3 even where a column is probed shorter.** A reach-2 pattern
  changes the aggregation's coarse space (3 levels / 480 coarse equations against 2 / 1296 on bfs3d) and
  diverged the then-shipped pattern-dependent smoother; positions, not values, were causal, and a stored
  exact zero is not a substitute (sparse arithmetic prunes it). Re-measure under the traced hierarchy
  before lowering it; ledger: `solve-refuted-directions.md`, reach-2 under "Low-β directions". Under the
  traced split, zeroing the VALUES beyond reach 2 (positions kept) costs 1.2–1.9× the applications on
  bfs3d, and beyond reach 1 does not converge at all (ledger, "TRUNCATING the Jacobian to reach 1").
- ⚠️ **A short reach corrupts rather than truncates.** A colouring is collision-free only for the pattern
  it was built at, so two cells sharing a reach-2 colour can both couple to a row at distance 3; the
  response holds the sum and the de-compression charges it to the near entry. So pattern entries outside a
  column's own reach are written as explicit zeros (`ColumnProbePlan.in_reach`), never gathered. Pinned by
  `test_a_short_probed_column_zeroes_its_out_of_reach_entries` and
  `test_per_column_plan_recovers_a_mixed_reach_matrix_exactly_and_more_cheaply`.
- ⚠️ **The residual can reach past `stencil_reach`, and then every column aliases.** `CorrectedGreenGauss`
  applies its face-coupling operator once per Richardson sweep, so `k` sweeps put a scalar residual's
  stencil at `k + 1` and the coupled RANS residual's at `k + 2` — at the shipped 4 sweeps, past reach 3 on
  any skewed mesh. bfs3d is skew-free (latent); pitzDaily is skewed and its reach-3 probe is measurably
  inexact (1.99e-07 against the true jvp, floor only at reach 5), almost all in the pressure column.
  - **The fix is to probe a narrowed residual, not to widen the reach**: `JacobianProbeSpec(gradient_sweeps=n)`
    (the coupled probe's `gradient_sweeps`) caps the sweeps for the coloured probe only, so the recovered
    matrix is exact for the residual it was taken from while the Krylov matvec, the root and the adjoint
    stay on the full residual. `None` is byte-identical. See `schemes.md` for `narrow_gradient_sweeps`.
  - The cap relocates the far mass inward, it does not reduce it (that is set by skewness).
  - Preconditioning the gradient solve with GMRES instead moves reach the wrong way (implicit
    differentiation runs the sweep series to round-off: reach 9–11) — refuted.
  - **Read a reach or exactness measurement per (row field, column field) pair**, never over a whole column
    or matrix: row norms differ by ~8 orders (k 8.8e-06, ω 1.4e+03), and `jacobian_relative_error` (one
    random vector over the whole matrix) cannot see a wrong pressure block. `probe_reach_audit.py`'s
    `block_shell_fractions` is per pair. Detail: log, "The residual reaching past stencil_reach".
- **Why the velocity columns reach 3 and p/k/ω reach 2: the eddy viscosity reads the strain rate** (a
  velocity-dependent coefficient of a gradient-based flux). Refuted explanations, do not re-propose: the
  flux limiter / second-order reconstruction, the gradient scheme's own stencil, the velocity gradient in
  `a_P`'s lagged flux. A `stop_gradient` on the strain rate is a diagnostic (it changes the Jacobian, not
  the residual), not a route to a reach-2 pattern. Arms table (`column_reach_probe.py`): log, "Why the
  velocity columns reach further".
- **Judging preconditioner arms:** a pivot census, `condest` or a diagonal-dominance census did not
  predict convergence on this operator — judge on a true-residual solve; and at bfs3d's march shift every
  arm once tied, separating only at zero shift, so a march cannot validate a change only the adjoint's
  operator exercises.

## The deleted monolithic complete LU — traps that transfer

- **A factor frozen at one shift mis-preconditions the operator once β moves** — why a march keeps its
  preconditioner current (`BetaTrackingRefresh`) rather than freezing it.
- **Judge a Krylov stop on the true residual** — an apparently faster incomplete-LU arm was a
  preconditioned-norm artifact (‖Mr‖, not ‖Ax−b‖).
