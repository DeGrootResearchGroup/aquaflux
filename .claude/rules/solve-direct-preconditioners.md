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
- **A materialize holds the matrix plus one chunk.** Seeds are filled one chunk at a time into a reused
  buffer (`ColumnProbePlan.seed_block(start, stop, out=)`, one shape, so the batched map compiles once),
  and `ProbeGather` scatters each chunk's responses as they are computed; out-of-reach entries need no
  sentinel. The gather map's one-time build (`_index_dtype`, COO→CSR grouping) runs at case construction
  and peaks above any single materialize. Measurements: log, "Chunked seeds and per-chunk
  de-compression" and "The gather map's build peak".
  - **Batch size:** `PROBE_BATCH_SIZE = 8` (`materialized_session.py`), where processor time bottomed on
    bfs3d (2026-08-11). The probe's memory barely scales with it, so auto-sizing the batch to free memory
    is not worth building; a run's memory ceiling is set by building the case, not by probing it.
  - **Batched probing** (`batched_matvec`, a `jax.vmap` of the jvp built once) and the **fixed full-pattern
    gather** (`structure=`) are what the materialized path passes. The full pattern (no `eliminate_zeros`)
    is a superset of the live nonzeros at any state, so the structure never moves cold→developed — that
    fixed structure is the gather's real value (in-place refits need it), not its wall clock.
- **Probe colouring reach is a per-case choice, set by measurement; the pattern stays at reach 3.**
  pitzDaily probes every column at reach 2 inside a reach-3 pattern (`column_reach: [2, 2, 2, 2, 2]`,
  2026-10-14: same 18 steps / 123 cycles / `x_r/h` 8.0686, probe per refit 4.3–5.8 → 2.5–3.4 s; at zero
  shift 7 % more forward applies, none transposed). bfs3d keeps uniform reach 3 because the same colouring
  cost it 34 % of its wall (measured by the project owner, PR #722, `validation/bfs3d_openfoam/compare.py`
  with `BFS3D_COLUMN_REACH=2,2,2,2,2,2`, shipped `case.yaml`, `77b03c18`, Apple M3 Pro, jax 0.10.2):

  | probe colouring (pattern reach 3) | steps | restart cycles | final `\|R\|` | refit (probe / refactor) | solve |
  |---|---|---|---|---|---|
  | uniform reach 3 (shipped) | 18 | 137 | 5.1e-06 | 11.9–12.7 s (8.2–8.5 / 3.6–4.2) | 522 s |
  | reach 2 | 21 | 216 | 5.0e-06 | 6.0–6.7 s (3.4–3.6 / 2.4–3.0) | 697 s |

  What decides it is the far mass the colouring folds (`probe_reach_audit.py`: bfs3d's p carries 1.1e-3 of
  its norm beyond reach 2, against ~4e-4 folded on pitzDaily). Untested: shortening only some of bfs3d's
  columns. pitzDaily arms and the reach-4 pattern: log, "Probe colouring reach 2 — pitzDaily and bfs3d".
- **Keep the materialized pattern at reach 3 even where a column is probed shorter.** A reach-2 pattern
  changes the aggregation's coarse space (3 levels / 480 coarse equations against 2 / 1296 on bfs3d) and
  diverged the then-shipped pattern-dependent smoother; positions, not values, were causal, and a stored
  exact zero is not a substitute (sparse arithmetic prunes it). Re-measure under the traced hierarchy
  before lowering it; ledger: `solve-refuted-directions.md`, reach-2 under "Low-β directions".
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
