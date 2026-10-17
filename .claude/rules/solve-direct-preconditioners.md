---
paths:
  - "aquaflux/precondition/sparse_jacobian.py"
  - "aquaflux/precondition/host_preconditioner.py"
  - "aquaflux/precondition/materialized_preconditioner.py"
---

# Rules — `aquaflux/precondition/` the frozen-host contract and Jacobian materialization (and the record of the deleted direct preconditioners)

> ⚠️ **THE COMPLETE LU IS DELETED (2026-10-11), one day after PETSc took its UMFPACK backend and the
> monolithic AMG (2026-10-10).** There is no `lu_preconditioner.py`, `CompleteLu`, `CompleteLuPreconditioner`,
> `CompleteLuInverse`, `complete_lu_inverse`, `FACTORIZATION_LINEAR_SOLVE`, `BetaTrackingRefresh(every_step=)`,
> `LU_BACKENDS` or `backend` anywhere; nor `amg_preconditioner.py`, `MonolithicVCyclePreconditioner`,
> `MonolithicVCycleInverse`, `monolithic_vcycle_inverse` or `ShiftedCellMajorOperator`. What this file
> still governs is live: the frozen-host contract (`host_preconditioner.py`: `FrozenInverse`,
> `RefactorableInverse`, `ReleasableInverse`, `HostPreconditioner`), the shared materialized base
> (`materialized_preconditioner.py`: `MaterializedJacobianPreconditioner`, whose subclasses are
> `FieldSplitPreconditioner` and `MaterializedBlockPreconditioner`) and the coloured-probe Jacobian
> (`sparse_jacobian.py`). **Every entry below that measures the complete LU, UMFPACK, a PETSc V-cycle or
> the ILUT is dated history and cannot be re-run from this tree.** Why the LU was dominated, and the
> evidence, is in
> `.claude/notes/solve-refuted-directions.md`, "Complete LU (SciPy SuperLU) — DELETED as dominated"; the LU tests' assertions live on in
> `tests/integration/test_coupled_field_split.py`, ported onto the split.

> ⚠️ **`coupled_continuation`, `coupled_lu_continuation`, `coupled_amg_continuation`, `lu_beta_tracking_refresh` and `amg_beta_tracking_refresh` no longer exist (deleted 2026-09-14, #371).** Entries below that name them are dated history. The coupled march is now one builder, `coupled_step`, given a preconditioner value (`BlockDiagonal` or `MaterializedJacobian(FieldSplit | BlockInverse)` (a `CompleteLu` inverse existed until 2026-10-11)), and a march that keeps its preconditioner current runs on a session (`open_session`) — see the rename table in `.claude/rules/turbulence.md`.

> Split out of `solve.md` (2026-08-18) to keep routine `aquaflux/solve/` work from loading the
> full complete-LU and coupled-Jacobian-materialization investigation narrative. See `solve.md` for
> the package-wide contracts, current configuration, and binding decisions this file assumes.
>
> **Dated investigation lives in `.claude/notes/solve-direct-preconditioners-log.md`** (split out 2026-10-17; it never auto-loads). This
> file keeps the current contract, the shipped defaults and the traps that still apply to live code. The
> measurements of the deleted preconditioners (complete LU, UMFPACK, the PETSc V-cycle and its incomplete-LU
> smoother, the monolithic ILUT) were deleted rather than moved; why each lost is in
> `.claude/notes/solve-refuted-directions.md` ("Incomplete-LU preconditioning of the pitzDaily flow block",
> "PETSc ... DELETED as dominated", "Complete LU (SciPy SuperLU) — DELETED as dominated").

## Preconditioner — the frozen host family (shared contract)

- **`equilibrate_cell_major` and `cell_major_permutation` live in `frozen_operator.py` (binding).**
  ⚠️ There is no `solve/ordering.py`, no `equilibrate_ordered` and no elimination-ordering strategy
  family (`EliminationOrdering` / `CellMajor` / `CellOrder` / `NaturalCells` /
  `ReverseCuthillMcKeeCells` / `AscendingRowLengthCells`): they existed for the zero-fill ILU(0) kernel
  and were deleted with it (2026-09-13, #371); the one ordering left is the natural cell-major one.
  They are the reorder half of one transform whose rescale half
  (`symmetrically_equilibrate`, `equilibration_scale`, `apply_symmetric_scale`, `row_chunks`) was
  already there, and every consumer applies the two together -- a factorization or a coarsening wants
  the matrix both unit-diagonal and grouped by cell. Its library consumer was the deleted PETSc V-cycle;
  it stays for the study harnesses (`cell_block_conditioning.py`, `column_reach_zero_pruning.py`,
  `host_ilu_levels.py`) and its unit test, and is a deletion candidate if those go.
  - **Neither is exported (since #284 part 2).** The harnesses import them from
    `precondition/frozen_operator.py`, a reach listed in `test_solve_api.py`'s
    `VALIDATION_INTERNAL_REACHES`. The permutation's unit test moved with the function, into
    `test_frozen_operator_scaling.py` beside the rescale half it belongs with.


- **The host preconditioners share ONE application path and ONE declared contract —
  `precondition/host_preconditioner.py` (BUILT, 2026-08-14).** The (since deleted) complete LU and the AMG V-cycle
  differ entirely in how the inverse is *fitted* and not at all in how it is *applied*, so
  `HostPreconditioner` owns `__init__` and `matvec()` and each subclass supplies only `build` /
  `refresh_in_place`. Those two genuinely differ (different inputs, different refresh costs) and are
  deliberately **not** unified behind a signature that would be the union of both.
  - **`FrozenInverse` is the contract, and it is exactly `n_dofs` + `apply(residual, *, transpose=…)`.**
    That pair is a real structural contract satisfied by `HierarchyBlockInverse` and both
    `FieldSplitInverse`s (four further members — the complete-LU factors, the PETSc
    `MonolithicVCycleInverse`, the monolithic ILUT and the Vanka smoother — have since been deleted) — and declared by none of them individually, so `matvec` would otherwise be written out
    once per class. `FieldSplitPreconditioner` used to obtain it by subclassing a *concrete sibling*,
    `MonolithicVCyclePreconditioner` — which is what let the `has_exact_solve` hazard below happen at all.
    **⚠️ That is fixed structurally (2026-09-11, #287): `MaterializedJacobianPreconditioner`
    (now `materialized_preconditioner.py`; it was in the deleted `amg_preconditioner.py`) is the shared
    base**, holding only what
    is genuinely common to fitting *any* inverse to the coloured-probe materialized coupled Jacobian
    (the probe itself, the shift-diagonal add, teardown) —
    `MonolithicVCyclePreconditioner` is no longer `FieldSplitPreconditioner`'s base, so it no longer
    inherits the monolithic-only state (the fixed-pattern cell-major assembler) it never used. See `solve-amg-multigrid.md` and `solve-field-split.md`.
  - **⚠️ Anything a base reads off `self.inverse` beyond that pair is a requirement on ALL of them
    (binding).** This is not hypothetical: an exact-solve capability flag (`has_exact_solve`, deleted with
    the host exact forward solve on 2026-09-13, #371) read `self.inverse.has_exact_solve`, which only
    `MonolithicVCycleInverse` had, so the property **raised** on the field split — and the call sites asked through
    `getattr(pc, ..., False)`, whose default swallows an `AttributeError` raised inside a property body
    exactly as it swallows a missing name. If a capability is not in `FrozenInverse`, answer it on the
    subclass. Pinned by an AST check on the base's own source
    (`test_the_base_asks_its_factors_for_nothing_beyond_the_declared_contract`) — read off the source
    rather than exercised, because the failure is a lookup that is *never taken* on the paths a test
    would naturally drive, which is why the original went unseen.
  - **The capabilities only SOME inverses have are declared protocols, asked by `isinstance` (BUILT
    2026-10-10, #281).** Beside `FrozenInverse`, `host_preconditioner.py` declares two, each
    `runtime_checkable` and exported from `aquaflux.precondition`:
    - **`RefactorableInverse(FrozenInverse)`**: `refactor_block(block)`, re-fit IN PLACE to the raw
      field-major operator of the built shape. **One spelling across the family**: the split's
      `FieldSplitInverse.refactor` is renamed `refactor_block` (there is no `split.refactor`), so a
      split is itself refactorable and refreshes through the same path as a single block inverse.
      (The deleted `CompleteLuInverse` had one too, re-factoring its SuperLU factor afresh; the deleted PETSc
      `MonolithicVCycleInverse.refactor(cell_major, scale, perm)` had a different signature and was deliberately not
      this protocol.)
    - **`ReleasableInverse`**: `destroy()`, with `release(factors)` the one place that asks.
      `HostPreconditioner.destroy()` calls it, so every family member releases its inverse. ⚠️ **Before
      this, `MaterializedJacobianPreconditioner.destroy` had an EMPTY body**, so destroying a
      `MonolithicVCyclePreconditioner` or a `FieldSplitPreconditioner` released nothing — only
      `MaterializedBlockPreconditioner` overrode it with a `getattr` probe. Every validation harness that
      calls `pc.destroy()` "to keep one preconditioner in memory at a time" was relying on a no-op.
    - **The traced capability is NOT here**: it is `OffersTracedCycle` / `TracedCycle` in
      `traced_cycle.py` (#665), which holds the hierarchy as array leaves. A `TracedFactors` /
      `apply_traced` protocol briefly existed on this PR's branch and was dropped in its merge with
      main; neither name exists.
    - `require_refactorable(factors, owner)` raises a `TypeError` naming the inverse; the split checks
      BOTH blocks before re-fitting either, so a refusal never leaves it half re-fitted. A split also
      refuses a non-`FrozenInverse` block at construction (`TypeError`) rather than at first apply.
    - **The factory types say so**: `leading_inverse` / `trailing_inverse` / `inverse` are
      `Callable[[sp.csr_matrix, int], FrozenInverse]` (were `... object`), and `BlockInverse.__call__`
      returns `RefactorableInverse`.
    Pinned by `test_each_inverse_declares_exactly_the_capabilities_it_has` (both directions, real
    types), `test_destroying_a_preconditioner_releases_its_inverse_when_it_holds_anything`, and the
    split's refit/atomicity tests in `test_field_split.py`; each was mutation-checked.
  - **`refresh_in_place` has ONE return type across the family (#281):** the `("probe", s),
    ("assemble", s), ("refactor", s)` tuple. Its signature had two shapes while the complete LU existed
    (the LU took no probe keywords, so the session chose them by
    `isinstance(pc, MaterializedJacobianPreconditioner)`); with the LU deleted (2026-10-11) every refreshed
    preconditioner is a `MaterializedJacobianPreconditioner` and the session passes `batched_matvec`,
    `probe_batch_size` and `structure` unconditionally.
  - **The pseudo-transient shift has one home: `sparse_jacobian.shifted_jacobian`.** Every host
    preconditioner adds `β d` before factoring, and two spellings once disagreed: a pattern-preserving
    `setdiag` against `a + sp.diags(shift)` — the latter is wrong, since a sparse *addition* stores only
    entries whose result is nonzero and so drops the explicit zeros a fixed-pattern probe deliberately
    kept. **Measured:** the two spellings are identical in values *and* pattern on a full
    diagonal and on a matrix with diagonal entries missing (both create them), and differ only where
    explicit zeros are stored — so adopting `setdiag` everywhere is a correctness fix in general. The
    whole refactor was **bit-identical** end to end, measured at the time on a complete LU (since deleted)
    built and refreshed under both implementations: byte-equal `matvec` and transpose output.

## Materializing the coupled Jacobian (`sparse_jacobian.py`)

**This section documents `sparse_jacobian.py`'s coloured-probe materialization of the coupled Jacobian —
shared infrastructure consumed today by the materialized-Jacobian multigrid inverses (the field split
and a bare block inverse).** It originally grew up beside the now-deleted monolithic ILUT (the alternative to the
block-triangular SIMPLE preconditioner for the coupled saddle, which formed the true Schur coupling
`B F⁻¹ G` through its incomplete-LU fill rather than approximating it); that preconditioner is gone
(dominated by the complete LU at 2D and by the algebraic multigrid at 3D, per its own docstring and per
the "nothing selects the monolithic ILUT or complete LU for the adjoint" finding, deleted with the
monolithic-AMG record — its zero-shift verdict is the "Adjoint (zero shift)" bullet of
`.claude/notes/solve-refuted-directions.md`, "Complete LU (SciPy SuperLU)"), but the Jacobian-materialization machinery below did not go with it — the
coloured probe still depends on it (as the complete LU did until it was deleted, 2026-10-11).

- **`sparse_jacobian.py`** materializes the coupled Jacobian from the *same* residual the solver uses
    (no re-derived assembly): `block_stencil_colouring(owner, nb, n, reach)` (pure NumPy — the cell-block
    pattern at a stencil `reach` and a collision-free CPR colouring, the conflict graph is the pattern
    squared) then `materialize_block_jacobian(matvec, plan)` (one `jax.jvp` per colour×column-field).
    **The `(colouring, n_fields)` pair is GONE — both take a `ColumnProbePlan`** (`ColumnProbePlan.uniform(
    colouring, n_fields)` for one reach throughout, `column_probe_plan(owner, nb, n, column_reach,
    pattern_reach)` to give each column its own); `block_stencil_gather_map(plan)` likewise takes one
    argument. `jacobian_relative_error` guards that `reach` covers the stencil (coupled RANS reaches
    distance **3**). Field-major DOF layout `(cell i, field f) = f·n + i`.
    - **The colouring is by SATURATION degree, not by degree — 112 → 94 colours on `bfs3d` (BUILT,
      2026-08-11).** `_saturation_colouring` picks the uncoloured vertex whose neighbours already show
      the most *distinct* colours. The colour count **is** the probe count, so this is 108 fewer
      directional derivatives per materialize for free; the build costs 0.6 s against the degree-ordered
      greedy's 0.2 s, paid a handful of times per march. **Any collision-free colouring de-compresses to
      the same matrix**, so this changes what a materialize costs and not what it returns — which is why
      it needed no re-validation of anything downstream. Colour counts on this mesh: reach 1 **16 → 11**,
      reach 2 **60 → 39**, reach 3 **112 → 94**. Pinned by
      `test_saturation_colouring_uses_fewer_colours_than_degree_ordering` on a 3D lattice.
    - **Vertex selection is a BUCKET QUEUE, not a full scan — the scan alone was quadratic (issue
      #243, fixed 2026-08-18).** `_saturation_colouring` used to pick the next vertex by
      `np.argmax(rank)` on every one of the `n` rounds — an `O(n)` scan repeated `n` times, `O(n²)`
      total, even though the neighbour-update half of the same round was already degree-bounded and
      vectorized. `rank` is bounded by `degree.max() * (degree.max() + 2)`, a property of the mesh's
      *local* connectivity and not of `n`, so a doubly-linked-list bucket per reachable rank value
      turns extraction into an `O(1)`-amortized pop and a rank increase into an `O(1)` unlink/relink,
      giving `O(n + m)` total for `m` conflict-graph edges. **Measured on synthetic cubic-lattice
      conflict graphs at stencil reach 3 (plain Python lists for the bucket linked lists, not numpy
      arrays — this loop is scalar-access-bound and numpy's per-element scalar wrapping loses to plain
      list indexing here):** ~2× SLOWER than the scan at `bfs3d`'s own scale (21,952-cell cubic lattice:
      old 0.53 s, bucket-queue 1.08 s) but **crosses over and wins by the scale the issue actually
      reports the scan dominating** (262,144 cells, full `block_stencil_colouring` pipeline: old 18.3 s,
      bucket-queue 15.7 s; the pure colouring step alone: old 17.3 s, bucket-queue 10.8 s, ~1.6×
      faster). So this is a deliberate trade: a modest, bounded regression at the project's current
      working scale in exchange for removing what would otherwise become the dominant cost well before
      a million-cell mesh. **Ties within a bucket are broken by recency of eligibility, not by vertex
      index** — a different, still-deterministic choice from the old `np.argmax`'s first-occurrence
      convention, so the two algorithms can colour the same graph in a different order; this does not
      change collision-freedom or the colour count in any way that matters (`test_sparse_jacobian.py`'s
      `test_saturation_colouring_is_valid_and_comparable_on_*` tests validity and a colour-count bound
      against the old full-scan implementation, kept in the test file as a reference, not equality with
      it). *(The wall-clock figures above name only the mesh size, reach and Python-list-vs-array
      choice — no case, preconditioner or machine load; re-measure before treating the exact seconds
      as load-bearing. The qualitative crossover — worse below roughly 100k–200k cells, better above —
      is the reproducible part, and it lands where the issue's own synthetic-scaling evidence put the
      `O(n²)` term starting to dominate.)*
    - **✅ SEEDS ARE BUILT ONE CHUNK AT A TIME, not all at once (BUILT, 2026-08-11).** The probe seeds
      are `n_probes x n_fields x n_cells` floats — **441 MB on `bfs3d` at 399 probes, 624 MB at 564** —
      and the old code materialized the whole set before chunking it, holding it for the entire
      materialize. `ColumnProbePlan.seed_block(start, stop, out=)` fills a **reused** buffer of one chunk
      instead, so the seeds cost `probe_batch_size x nf` (a few MB) rather than `n_probes x nf`. Measured
      on an 18³ lattice at 546 probes: **153 MB → 1.1 MB, 136x less.** The buffer keeps one shape across
      every chunk, so the batched map still compiles a single time, and a final short chunk is padded by
      the rows `seed_block` leaves zero — the previous `np.vstack` pad is gone with it.
    - **✅ THE DE-COMPRESSION RUNS PER CHUNK TOO, so a materialize costs the matrix and nothing else
      (BUILT, 2026-08-11).** `block_stencil_gather_map` now returns a **`ProbeGather`** rather than an
      `(indptr, indices, gather_map)` tuple, and it sorts the CSR entries by the probe that feeds them.
      Each chunk's responses are scattered in as they are computed and then dropped, so the full
      `n_probes x nf` response array is gone — and with it the `np.concatenate` that appended a zero
      sentinel, which was a **second** full copy. Measured on an 18³ lattice, 546 probes, 11.5M nnz:

      | | |
      |---|---|
      | matrix data alone (irreducible) | 92 MB |
      | responses + sentinel copy (old, on top) | 306 MB |
      | **peak now** | **95 MB** |

      Scaled to `bfs3d` at 47.2M nnz that is roughly **1260 MB → 380 MB**. Out-of-reach entries need no
      sentinel: having no source probe, they never enter the sorted order, so a zero-initialized `data`
      leaves them zero. The index arrays are `int32` (entry counts and `n_probes x nf` both sit well
      inside its range), so the permanent structure is no larger than the `int64` gather map it replaced.
      **Verified against the TRUE matrix-free jvp on the real case** — `jacobian_relative_error` 2.5e-16
      (uniform) and 6.5e-16 (per-column), i.e. the float64 floor, which is an independent check rather
      than the two paths agreeing with each other.

    - **Current status — the gather map's one-time build.** `ProbeGather` chooses its index width from a bound
      known before the array is formed (`_index_dtype`: `int32` unless the bound overflows it), applies the
      out-of-reach mask in place, and groups entries by probe with a `scipy.sparse` COO→CSR conversion rather
      than an `argsort` permutation. That build runs once, at case construction, and its transient peak is
      larger than any single materialize's. Measurements and verification: `.claude/notes/solve-direct-preconditioners-log.md`, "Materializing the
      coupled Jacobian" → "The gather map's build peak" and "The sort permutation removed".

      **⚠️ Read this before tuning `_PROBE_BATCH_SIZE` against memory.** Most of the peak the batch size
      was being traded against was never the batch: seeds and responses are both `n_probes x nf` and
      neither scaled with `probe_batch_size`. Both are now gone, so the peak is the matrix plus one chunk
      — which is what makes a memory-budgeted batch size a meaningful knob rather than a knob on the
      small term. The recorded "16 vs 4: ~2.2 GB vs ~0.7 GB" was measured against the old allocations and
      **no longer describes this code**; re-measure before using it.

    - **✅ CURRENT (2026-10-14): pitzDaily probes EVERY column at reach 2 inside its reach-3 pattern
      (`column_reach: [2, 2, 2, 2, 2]`); bfs3d ships a uniform reach 3 (since #712), and measured the same
      colouring as a 34 % loss (below).** Everything in the
      bullet below is the bfs3d history under a zero-fill incomplete factorization, which neither case
      runs any more. On pitzDaily (`compare.py` / `case.yaml` shipped otherwise, one run each, 4-core Linux,
      jax 0.11.2; mid-span `x_r/h` 8.0686 and 18 steps in every arm):

      | probe colouring / pattern | restart cycles | final `\|R\|` | probe per refit | solve |
      |---|---|---|---|---|
      | reach 3 / 3 (was shipped) | 123 | 9.07e-06 | 4.3–5.8 s | 599 s ¹ |
      | **reach 2 / 3 (now shipped)** | 123 | 9.57e-06 | 2.5–3.4 s | 573 s ¹ |
      | reach 4 / 4 (exact for this residual) | 123 | 4.62e-06 | void ² | void ² |
      | reach 2 / 4 | 121 | 9.12e-06 | 2.6–3.3 s | 600 s ³ |

      ¹ `main` `403af71d`; ² contended by a test run; ³ `main` `a5d61d49`, where the reach-3/3 march took 543 s
      — so a reach-4 pattern is ~10 % dearer overall (each refit's refactor ~1.2 → ~2 s). The folded mass
      is ~4e-4 of the Jacobian's norm (distance-3 entries ~48 % of the pattern's positions). Subtracting
      the previous refit's far values from the folded near entries was tried statically and is NOT worth
      building: between refits the far values change about as much as they are large, so the correction
      cut the per-block error by at most ~2× and in some blocks not at all.
      **At ZERO shift — the adjoint's operator — the shortened probe costs 7 % forward and nothing
      transposed** (`zero_shift_arms.py` at the converged root `state-00018` of the shipped march, operator
      probed exact at uniform reach 3, preconditioner as the case configures it, GMRES restart 30 to 1e-8 on
      the true residual, same machine): uniform 262 / 283 applies (forward / transposed), reach-2 colouring
      280 / 284, both converged. ⚠️ The harness's `splitfloor` arm fails identically in both (621 applies,
      true 3e-2 to 1e-1): `compare.PC_BETA_FLOOR` is `None` since #712, so that arm is built at an unset
      shift and measures nothing about a floor.
      **⚠️ bfs3d does NOT take it: the same colouring costs that case 34 % of its wall (measured
      2026-10-15 by the project owner, PR #722).** `validation/bfs3d_openfoam/compare.py`, shipped
      `case.yaml`, the arm `BFS3D_COLUMN_REACH=2,2,2,2,2,2` against the unset uniform reach 3, both on
      `77b03c18`; Apple M3 Pro, 11 cores, macOS, jax 0.10.2, one run each, machine idle:

      | probe colouring (pattern reach 3) | steps | restart cycles | final `\|R\|` | refit (probe / refactor) | solve |
      |---|---|---|---|---|---|
      | uniform reach 3 (shipped) | 18 | 137 | 5.1e-06 | 11.9–12.7 s (8.2–8.5 / 3.6–4.2) | 522 s |
      | reach 2 | 21 | 216 | 5.0e-06 | 6.0–6.7 s (3.4–3.6 / 2.4–3.0) | 697 s |

      Same `x_r/h` 8.361. The probe halves, but the march takes 3 more steps and 58 % more cycles. Noise
      reference: `main` `a5d61d49` ran the shipped arm in 529 s (1.3 % apart). At zero shift
      (`zero_shift_adjoint.py`, the reach-3 march's `state-00018`, operator probed at uniform reach 3,
      restart 30 to 1e-8 on the true residual) it is nearly free: 373 / 403 applies (forward / transposed)
      uniform against 393 / 418, all converged. So the reach-2 colouring is a **per-case** choice and the
      far mass it folds is what decides it: `probe_reach_audit.py` finds bfs3d's p, k and omega carrying
      1.1e-3 / 6.7e-5 / 3.9e-5 of their norm beyond reach 2, against ~4e-4 folded on pitzDaily.
      Untested: shortening only some columns on bfs3d (the old `(3,3,3,3,2,2)`, or velocities alone).
      **The pattern stays at reach 3 for the reason the bullet below gives** — the coarse space reads it;
      only the colouring is shortened. Harnesses: `validation/pitzdaily_openfoam/compare.py` (the arms are
      `case.yaml` edits; a reach-4 pattern is `PITZ_STENCIL_REACH=4`).
    - **Per-column probing reach — the traps that still apply.** bfs3d's per-column history (shipped
      `(3,3,3,3,2,2)` until #712, issue #191, the colour-aliasing census, the pattern-pruning fix) is in
      `.claude/notes/solve-direct-preconditioners-log.md`, "Per-column probing reach — bfs3d's history".
      **⚠️ A SHORT REACH CORRUPTS RATHER THAN TRUNCATES, and this is the trap the design turns on.** A
      colouring is collision-free only for the pattern it was built at, so two cells sharing a reach-2
      colour may still both couple to a common row at distance 3; the response then holds the *sum* and
      the de-compression charges all of it to the near entry. So a column with far couplings has its
      **near** entries perturbed — it is not simply missing the far ones. Correspondingly, every pattern
      entry outside its column's own reach is written as an **explicit zero** rather than gathered
      (`ColumnProbePlan.in_reach` plus a zero sentinel row): gathering there would deposit a *different*
      cell's near coupling into a position the matrix has nothing in. Both halves are pinned
      (`test_a_short_probed_column_zeroes_its_out_of_reach_entries`,
      `test_per_column_plan_recovers_a_mixed_reach_matrix_exactly_and_more_cheaply`).

      **⚠️ THE OTHER SIDE OF THE SAME TRAP: the RESIDUAL can reach past `stencil_reach`, and then EVERY
      column aliases — the corrected gradient's sweep count is what moves it (measured 2026-08-16,
      harness `validation/gradient_stencil_reach.py`).** Everything above is about probing a column
      *shorter* than the pattern. The converse is a residual assembled so that it reaches *longer* than
      the pattern, which no `column_reach` choice can fix and which the exactness gate cannot see
      (`jacobian_relative_error` compares the recovered matrix against the same over-reaching operator on
      one random vector, and collapses over row fields besides — the blindness recorded above).
      `CorrectedGreenGauss` applies `A_g` once per Richardson sweep and `A_g` couples a cell to its face
      neighbours, so **`k` sweeps put a scalar residual's stencil at `k + 1`** and the coupled RANS
      residual's at `k + 2` (measured on a small skewed cavity: reach **6** at the shipped `sweeps=4`,
      4 at 2). **So the shipped `SweptGradientSolve(sweeps=4)` and the shipped `stencil_reach=3` are
      mutually inconsistent the moment a mesh is skewed.**
      - **⚠️ LATENT ON `bfs3d`, LIVE ON pitzDaily — "latent on every case that exists" was written here
        and is FALSE (corrected 2026-08-16).** `bfs3d` is skew-free to round-off (0 of 66368 interior
        faces above 1e-6 relative), which is the same fact as the recorded "swapping Corrected→Compact
        Green-Gauss leaves it bit-identical". **pitzDaily is not** (11567 of 24170 faces, max 7.5e-2),
        and it pays for it now: `jacobian_relative_error` against the true jvp is **1.99e-07 at reach 3**
        there against `bfs3d`'s 2.34e-16, reaching the float64 floor only at **reach 5** — so the shipped
        `stencil_reach=3` materializes a measurably wrong matrix on a shipped 2D case. The error is
        carried almost entirely by the **pressure column** (it enters the residual only through
        gradients), which a whole-matrix `jacobian_relative_error` cannot see — the same
        collapse-over-row-fields blindness recorded throughout this section.
      - **The fix is to probe a NARROWED residual, not to widen the reach** (the colour count climbs
        11 → 39 → 94 from reach 1 → 3 on `bfs3d`, so reach 6 is not purchasable):
        `JacobianProbe(gradient_sweeps=n)` / the coupled builders' `probe_gradient_sweeps=n` cap
        the sweeps **for the coloured probe only**, leaving the Krylov matvec the exact jvp of the full
        residual — so the recovered matrix is exact for what it was taken from, and the converged state
        and its adjoint are untouched. `None` (default) is byte-identical. See
        `.claude/rules/schemes.md` for `narrow_gradient_sweeps`, and `.claude/notes/schemes-log.md`,
        "`narrow_gradient_sweeps`: latent on bfs3d, live on pitzDaily" for the reach/accuracy tables.
      - **⚠️ Do NOT read the cap as "less far coupling".** Measured, the mass beyond a given distance is
        set by the **skewness**, not the sweep count (at 25 % perturbation, `|dR/dφ|` beyond distance 2
        is 9.80e-4 at 2 sweeps and 1.004e-3 at both 4 sweeps and the exact solve). A shorter sweep
        *relocates* that mass inward. What the cap buys is that the probe is no longer aliasing — the
        same distinction as everywhere else in this section.
      - **❌ "Precondition the gradient solve and use GMRES instead" is REFUTED as a reach lever, do not
        re-propose it.** With implicit differentiation the tangent is `A_g⁻¹B`, i.e. the sweep series run
        to round-off — measured reach **9** at 25 % skew and **11** at 40 %, against 5 for `sweeps=4`. It
        moves reach the wrong way, and it does not touch the reason GMRES is not the default (the nested
        implicit-diff tangent re-entered by every jvp, ≈180× per coupled-residual eval on pitzDaily).
      - ⚠️ **How it got through, which is the transferable part: the audit that licensed it AND the gate
        that verified it both collapse over row fields**, on a matrix whose row norms differ by ~8 orders
        (k row 8.8e-06, ω row 1.4e+03). `column_reach_ladder` takes a whole column block;
        `jacobian_relative_error` is one random vector over the whole matrix. Neither can see a wrong
        pressure block, and `probe_reach_audit.py` already had the right function — `block_shell_fractions`,
        per (row field, column field) pair. **Read a reach or exactness measurement per PAIR, never over a
        whole column or a whole matrix.**
      - **Do not trust a pivot census, `condest` or a diagonal-dominance census as a predictor of
        preconditioner convergence on this operator** — all three were measured blind to what separated
        converging from failing arms (deleted incomplete-LU smoothers, 2026-08); judge on the TRUE residual.
      - **The only operating point that separated preconditioner arms on bfs3d was zero shift** — at the
        march's own shift every arm tied (2026-08-12, deleted PETSc V-cycle), so a march cannot validate a
        change that only the adjoint's operator exercises.
      - **The materialized-reach default is set by measurement, per case** (pitzDaily colouring reach 2,
        bfs3d uniform 3 — above). Older entries that shipped a different per-column reach are history in the log.

      **✅ WHY the velocity columns reach further than the scalars: the EDDY VISCOSITY's dependence on
      the strain rate (measured 2026-08-11, `validation/bfs3d_openfoam/column_reach_probe.py`).** The
      momentum and scalar transport equations look alike, so the split invites an explanation, and the
      two obvious ones are both wrong. The harness perturbs one degree of freedom and takes a single
      directional derivative — that *is* the column, and how far the response spreads on the cell graph
      is its reach — so it costs megabytes rather than a materialize's gigabytes and can sweep arms
      freely. At `state-00077` (step 28, ‖R‖ 3.586e-06, shift 0.0064), from the four deepest interior
      cells:

      | arm | u | v | w | p | k | ω |
      |---|---|---|---|---|---|---|
      | Venkatakrishnan-limited linear upwind (the case) | 3 | 3 | 3 | 2 | 2 | 2 |
      | unlimited second-order upwind | 3 | 3 | 3 | 2 | 2 | 2 |
      | first-order upwind on momentum | 3 | 3 | 3 | 2 | 2 | 2 |
      | compact (one-shot) Green-Gauss gradient | 3 | 3 | 3 | 2 | 2 | 2 |
      | `a_P`'s lagged flux with no velocity gradient | 3 | 3 | 3 | 2 | 2 | 2 |
      | **`nu_t` with `stop_gradient` on the strain rate** | **2** | **2** | **2** | 2 | 2 | 2 |

      **`nu_t = a1 k / max(a1 omega, S F2)` reads the strain-rate magnitude, which is built from the
      velocity gradient.** That makes the eddy viscosity a *velocity-dependent coefficient* of a flux
      that is already gradient-based, and the composition of the two is what spends the extra ring.
      It also accounts for the columns that do NOT reach three: `k` and `omega` enter `nu_t` **pointwise**
      (no gradient), and pressure does not enter it at all.
      **Three explanations are REFUTED — do not re-propose them:** the flux limiter and the second-order
      reconstruction (first-order upwind on momentum, the very scheme k/ω use, still reaches three); the
      gradient scheme's own stencil (a one-shot compact Green-Gauss still reaches three); and the
      velocity gradient inside `a_P`'s lagged convective flux. The last was *my* account, argued from the
      source and killed by the arm built to test it — the lesson being that the ring is not where the
      pressure-velocity coupling is, it is in the closure.
      **⚠️ The arm is a DIAGNOSTIC, not a proposal.** `stop_gradient` on the strain rate changes the
      *Jacobian*, not the residual, so it is a quasi-Newton linearization rather than a model change.
      It is worth knowing that it would take the whole operator to reach two — but the probe feeds the
      **preconditioner**, and a reach-2 *pattern* is separately measured to break the hierarchy (below),
      so this is not a route to 234 probes without re-opening that.

      **This does NOT reopen the reach-2 lever below, and must not be read as contradicting it.** That
      entry is about probing *every* column at reach 2 **and coarsening the reach-2 pattern**, which
      changes the hierarchy (3 levels / 480 coarse equations against 2 / 1296) and fails. Here the
      pattern stays at reach 3, so the coarse space and the smoother see exactly what they see today; only
      columns with *nothing* beyond reach 2 are shortened, and the velocities — which do carry content
      there — are not.
    - **Materialize efficiency — two shipped speedups, both AMG-path-only, bit-identical (BUILT).** The
      probe dominates a refresh, so `materialize_block_jacobian` takes two optional accelerators the AMG
      preconditioner passes (the deleted complete LU kept the plain loop). **(1) Batched
      probing** — `batched_matvec` (a `jax.vmap` of the jvp, **built once and reused** so it compiles a
      single time; `probe_batch_size` chunks it for memory) runs the coloured probes as a few fused passes
      instead of a Python loop of separate calls. Measured 22.4→14.0 s (~1.6×) on `bfs3d` — modest because
      CPU forward-AD does not vectorize across the batch like a GPU (the win is dispatch amortization). For
      the SAME reason the chunk was once kept **small** (`_PROBE_BATCH_SIZE` was 4, not 16): a larger
      batch holds more simultaneous forward-AD tapes. **The default is now 8** — see the sweep below,
      which is what retired that reasoning.
      **Current status:** `PROBE_BATCH_SIZE = 8` (`materialized_session.py`), where processor time bottomed in a
      2026-08-11 sweep on bfs3d; the probe's memory barely scales with the batch, and per-column probing cuts
      CPU time 1:1 with the probe count. Sweep table, attribution of a run's memory (the case build, not the
      probe, sets the ceiling) and the timing method: `.claude/notes/solve-direct-preconditioners-log.md`, "Materialize efficiency".
      **⚠️ AUTO-SIZING THE BATCH TO AVAILABLE MEMORY IS NOT WORTH BUILDING, and this is why.** The
      motivation was that the right chunk depends on machine state and case size. At 2.5 MB per unit it
      does not: any sane fixed value is safe, and the constraint that motivated the mechanism was an
      artifact of allocations that have since been removed. Fix the allocation, not the knob.
      **(2) Gather de-compression** — `block_stencil_gather_map(plan)` precomputes, once, the
      **fixed full-pattern** CSR structure and the de-compression that fills it (no scatter loop, no
      per-materialize CSR re-sort), passed as `structure=`. ⚠️ It used to return an
      `(indptr, indices, gather_map)` tuple consumed as one vectorized `data = responses.ravel()[gather_map]`;
      it now returns a `ProbeGather` that scatters **chunk by chunk**, so no such expression exists and the
      full response array it indexed is gone (see the chunked-de-compression entry above). The **full** pattern (no `eliminate_zeros`) is the fixed mesh distance-3 colouring graph
      — a superset of the Jacobian's live nonzeros at *any* state — so the structure stays fixed
      cold→developed and no entry is ever dropped as values change: a guaranteed-fixed structure the in-place
      GAMG refactor needs. On `bfs3d` the full pattern is ~47.2M positions, but that is a **structural
      over-estimate**, mostly explicit zeros at every state — LIVE nnz is ~constant (**38.7M cold / 39.0M
      developed**; there is *no* cold→developed nnz collapse — an earlier "47.2M cold → 39.0M developed"
      reading conflated the fixed pattern with live nnz). **The stored zeros must be pruned before the
      operator reaches an incomplete factorization, and `MonolithicVCycleInverse._live` is where that now happens** — see
      *"stored exactly-zero positions break the ZERO-SHIFT V-cycle"* below, which measures what they cost and
      corrects the two mechanisms this bullet used to assert for it. De-compression 22.4→11.2 s (2.0× vs
      loop); full build (materialize + GAMG) only 56.0→54.2 s (**1.03×** — GAMG-dominated), so the gather's
      real value is the fixed-structure invariant, not the wall-clock.
      ⚠️ There is no `MonolithicVCycleInverse._live` and no stored-zeros entry any more (deleted with PETSc,
      2026-10-10); the trap it carried is the reach-3 bullet below: sparse arithmetic prunes stored exact zeros.
    - **Keep the materialized PATTERN at reach 3 even where a column is probed shorter.** A reach-2 pattern
      changes the aggregation's coarse space (3 levels / 480 coarse equations against 2 / 1296 on bfs3d) and
      diverged the then-shipped pattern-dependent smoother, where padding the distance-3 positions with
      ~1e-30 (nonzero) recovered convergence bit-identically — positions, not values, were causal; and a
      stored EXACT zero is not that padding (sparse arithmetic prunes it). Re-measure under the traced
      hierarchy before lowering the pattern; ledger: `.claude/notes/solve-refuted-directions.md` (reach-2).

## The deleted monolithic complete LU — traps that transfer

Deleted 2026-10-11 as dominated; why, and what was lost, is
`.claude/notes/solve-refuted-directions.md` → "Complete LU (SciPy SuperLU) — DELETED as dominated".
- **A factor frozen at one shift mis-preconditions the operator once β moves** — the reason a march keeps its
  preconditioner current (`BetaTrackingRefresh`) rather than freezing it.
- **Judge a Krylov stop on the TRUE residual** — an apparently faster incomplete-LU arm was a
  preconditioned-norm artifact (‖Mr‖, not ‖Ax−b‖).
