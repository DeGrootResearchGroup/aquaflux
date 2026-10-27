---
paths:
  - "aquaflux/precondition/sparse_jacobian.py"
  - "aquaflux/precondition/probe_layout.py"
  - "aquaflux/precondition/kronecker_pattern.py"
  - "aquaflux/precondition/host_preconditioner.py"
  - "aquaflux/precondition/materialized_preconditioner.py"
---

# Rules — `aquaflux/precondition/` the frozen-host contract and Jacobian materialization (and the record of the deleted direct preconditioners)

This file governs the frozen-host contract (`host_preconditioner.py`), the shared materialized base
(`materialized_preconditioner.py`), the coloured-probe Jacobian (`sparse_jacobian.py`: colouring, plan,
gather-map builders, seeds, materialize), the probe's stored layout and de-compression
(`probe_layout.py`: `BlockLayout`, `ProbeGather`, `ProbedBlocks`, split out of `sparse_jacobian.py` at
its size ratchet on 2026-10-24) and the cell-level pattern (`kronecker_pattern.py`). Package-wide contracts are in `solve.md`. Dated measurements and the
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
  `ProbedBlocks.diagonal()`, a device array; the shift is a device array from `frozen_shift_diagonal`); a test observes the fit by spying on it. ⚠️ A correction wired on the
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
  `block_stencil_gather_map(plan)` returns a `ProbeGather` (`probe_layout.py`). Field-major DOF layout `(cell i, field f) =
  f·n + i`. The coupled RANS residual reaches distance 3 on an unskewed mesh; `jacobian_relative_error`
  checks the reach covers it.
- **The colouring is by saturation degree** (`_saturation_colouring`, vertex selection by a bucket queue,
  `O(n + m)`): fewer colours means fewer probes, and any collision-free colouring de-compresses to the same
  matrix, so changing the colouring changes cost, never the answer. Ties break by recency of eligibility,
  so colourings differ from a full-scan argmax but stay valid (`test_saturation_colouring_*`). Colour
  counts and the bucket-queue crossover (slower than the scan below ~100k–200k cells, faster above):
  log, "Saturation colouring and its bucket queue".
- **Every probe structure has a `BlockLayout`** (`block_stencil_gather_map(plan, leading_fields=)` ->
  `ProbeGather.layout`; `materialize_blocks` -> `ProbedBlocks`). A field split's is laid out by the
  split's blocks; with no `leading_fields` it is the one group (`leading_dofs == n_dofs`,
  `BlockLayout.whole` its pattern, the other three blocks empty, `ProbeGather.indptr`/`indices` read off
  that pattern), so a single block inverse is fitted by the same device path (2026-10-23).
  The layout gives:
  one block-major value array **on the device** (2026-10-22), each block an `entry_map.DeviceMatrix` of
  its slice over a host `SparsePattern` the layout owns (a `CellLevelPattern` on the cell path, below;
  `SparsePattern.of_arrays` on the entry-by-entry one; either way a hierarchy fitted to that block shares
  it rather than holding a copy), the shift added at the stored diagonal by one
  donated scatter (`_add_at`; refused if a diagonal entry is missing). `materialize_block_jacobian` (the
  public host-matrix form) refuses a split layout and reads a whole one back once; `refresh_in_place`
  takes the block path whenever a `structure` is given and calls `_refactor_blocks`:
  `FieldSplitPreconditioner`'s cuts the split, `MaterializedBlockPreconditioner`'s refuses a split layout
  and refits its inverse to the whole block, each through `hierarchy_inverse.block_for` (a
  `HierarchyBlockInverse` gets the `DeviceMatrix`, any other inverse `as_scipy` of it). The first fit
  reads the fitted block(s) back once, since a factory coarsens on the host. A refit reads nothing back
  but the pivot floor's row set (`solve-field-split.md`, "Refresh"). Pinned for one group by
  `test_a_single_block_refit_probed_on_the_device_fits_what_the_whole_host_matrix_fits`.
- **The probed pattern is a cell pattern over full blocks of field pairs, and the probe's index arrays are
  held at cell level (2026-10-24, the first stage of `solve-open-directions.md` §14d).** Every pattern
  entry keeps all its field pairs (`in_reach` zeros stay stored), and `active_rows` drops whole blocks, so
  each block is `C ⊗ (r×c)`: `kronecker_pattern.KroneckerPattern(cells, row_fields, column_fields)`, with
  `entry_position` the one formula placing entry `(a, cell entry, b)` in the degree-of-freedom CSR order
  (`a·c·nnz_C + c·indptr[i] + b·len(i) + k`), so the **values keep exactly the order the dof pattern gives
  them**. `block_stencil_gather_map` takes the cell path (`_cell_probe_gather`: `BlockLayout.of_cells`, the
  dof block patterns expanded from `C` without a sort; `_CellEntries`, per distinct colouring the in-reach
  cell entries grouped by column colour, plus each entry's row and `C`'s `indptr`; `_scatter_cell_chunk`
  places a chunk for every stored row field) whenever `_stored_blocks` finds each of the four blocks kept
  or dropped whole, else the entry-by-entry path (`_dof_probe_gather`, `_DofEntries` holding `_source` /
  `_position`, two indices per stored entry). Pinned equal, layout array for array and values exactly, by
  `test_the_cell_level_probe_order_lays_out_and_fills_what_the_entry_order_does`; the fallback by
  `test_a_partition_keeping_part_of_a_block_is_probed_entry_by_entry`. On a 21952-cell lattice at 6
  fields and reach 3 (45.5M entries) the structure retains 193 MB against 546 MB, the rest being the host
  whole pattern's CSR arrays (`gather_map_memory.py --side 28 --reach 3 --fields 6`, 2026-10-24, 4-core
  Linux, jax 0.12.0.1, before the block patterns below were held at cell level). The padded layouts,
  Galerkin maps and SIMPLE cuts downstream are cell level too (`solve-amg-multigrid.md`,
  `solve-flow-block.md`).
  - **The layout's block patterns are `kronecker_pattern.CellLevelPattern`s (2026-10-24):** a
    `SparsePattern` held as its `KroneckerPattern`, whose `indptr`/`indices` are formed on each read
    (`KroneckerPattern.dof_arrays`) and never kept, whose `matches` is identity, then cell patterns, and
    expands only against a pattern held as arrays. These were the session's host `SparsePattern`s (140
    MiB on bfs3d, 22 → 4.9 MiB of live host patterns on pitzDaily). Every refit checks its blocks against
    them by identity, so only the first fit's host read-back (`as_scipy`) and `SparsePattern.of` at
    build expand them; `ProbeGather.indptr`/`indices` are properties for the same reason. ⚠️ A caller
    reading both arrays should read them once (each read expands). Pinned by
    `test_a_refit_never_expands_the_probe_s_cell_level_block_patterns` (a spy on the expansion) and
    `test_a_cell_level_pattern_matches_exactly_the_patterns_storing_its_entries`.
- **A materialize holds the matrix plus one chunk.** Seeds are formed one chunk at a time **on the
  device** (2026-10-23): `ColumnProbePlan.seed_block(start, stop, rows=)` (whole-state) and
  `field_seeds(start, stop, rows=, dtype=)` (one field's) are one jitted indicator, `_indicator_seeds`,
  over the plan's colours and each probe's (field, colour), moved to the device once (`_device`); `rows`
  pads a short chunk with zero rows (one shape per program, so each batched map compiles once). Pinned by
  `test_the_seeds_formed_on_the_device_are_each_probes_colour_indicator`. Cost (pitzDaily and the
  laminar pitzDaily of `validation/pitzdaily_openfoam/laminar/`, whose `refit_cost.py` times the
  single-block refit): log, "Seeds on the device, and the single-block refit on the device (2026-10-23)".
  **A chunk never spans two
  column fields** (`ColumnProbePlan.field_chunks`; each field's last chunk may be short and is padded, so
  a materialize makes up to `n_fields − 1` more calls than `ceil(n_probes / batch)`; `probe_batch_size=None`
  is one batch per field),
  and `ProbeGather` scatters each chunk's responses as they are computed, **on the device** (2026-10-22):
  its entry order (above) lives there, `_ChunkProbe.responses` returns device arrays, and one jitted
  kernel per order (`_scatter_cell_chunk` / `_scatter_chunk`, `donate_argnums=0`) places a chunk through
  `ProbeGather.width` lanes (the most cell entries, or entries, any chunk reads), so one program serves
  every chunk; a lane past the chunk's end is dropped (`mode="drop"`). `ProbeGather.values(probing)` is the loop; the whole-matrix path reads
  the values back once at its end. Out-of-reach entries need no sentinel. The gather map's one-time build (`_index_dtype`, COO→CSR grouping) runs at case construction
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
    After the device fit (#804–#808) the 2-step peak is 8.87 GiB in a later re-fit's leading block re-fit,
    and the device holds 2.19 GiB of int32 pattern arrays against 0.66 GiB of float32 values (2026-10-24,
    `7fe27d07`; log, "bfs3d memory by owner after the device fit, probe and seeds"). Current: 5.58 GiB
    with the tracer, set by the step program's first call (2026-10-25, after #817--#825 and the early
    release of the superseded split; `solve.md`); the whole shipped march peaks at 4.43 GiB `VmHWM`
    against 8.81 GiB on `1290aa11`, same machine and step table (2026-10-25, `af118f90`; validation
    log, "A week of memory and time work, paired on one machine").
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
  shift 7 % more forward applies, none transposed). bfs3d probes pressure at reach 3 and every other
  column at 2 (`column_reach: [2, 2, 2, 3, 2, 2]`, since 2026-10-26): three perturbed draws per arm
  (`BFS3D_NU_PERTURBATION` 0 / ±1e-14, pivot floor, 4-core Linux, jax 0.12.0.1, `45a9059`) gave 17 steps
  / 146 cycles every draw against uniform reach 3's 17 / 143, probe ~23 → ~10 s per fit, march
  1068–1203 s against 1276–1467 s (10–18 % shorter), all to `x_r/h` 8.361. Every column at reach 2 took
  21 / 23 / 40 steps, 194–384 cycles (2026-10-25). Tables: log, "Probe colouring reach 2 — pitzDaily and
  bfs3d" and "bfs3d: pressure alone at reach 3". The zero-shift adjoint ties: 331 / 373 applications
  (forward / transposed) to 1e-8 against uniform's 335 / 383 (`zero_shift_adjoint.py`, `state-00017`).
  `JacobianProbeSpec.strong_couplings` (pressure at reach 2 plus the couplings beyond it the first fit
  shows to be strong, 243 seeds) kept the 17 steps on three draws and saved ~1 %, a wash, so bfs3d
  does not set it (log, "`StrongCouplings` (library, 2026-10-26)").
- **`StrongCouplings` (`strong_couplings.py`; `JacobianProbeSpec.strong_couplings`, off when unset)
  probes one column at its `column_reach` plus the couplings beyond it that the Jacobian shows to be
  strong.** At the session's first fit (`MaterializedSession._with_strong_couplings`) the column is
  probed once at the plan's `pattern_reach`; beyond its own reach a pair `(i, j)` is kept when its
  magnitude, in some judged row field (`rows`, default the column's own), is at least `fraction` times
  row `i`'s smallest **nonzero** face-neighbour coupling to the column. Every fit from then on probes the
  column for that pattern (`ColumnProbePlan.with_column_pattern`, coloured collision-free for it); the
  pattern is held for the session, like a frozen coarsening. It needs the plan to carry its cell graph
  (`ColumnProbePlan.edges`, set by `column_probe_plan`) and refuses `column_reach` unset. The selection
  is `MaterializedSession.coupling_selection` and is described to the `"probe"` report sink.
  `JacobianProbeSpec.build_settings()` is what reaches a problem's `probe` (all but this field).
  ⚠️ Judge the pressure column by its continuity rows (the default), not the momentum rows: a momentum
  component's pressure coupling across a face parallel to that component is near zero without being
  zero, so its weakest face coupling is a poor yardstick (the option 1 audit's static-bar failure).

  The shipped arm on a 4-core Linux container (Xeon 2.1 GHz, jax 0.11.2, `b1648725`, glibc defaults,
  cold cache, 2026-10-17): 29 steps / 173 cycles / 3.7e-06 / re-fit 56–64 s (probe 37.7–41.8 / refactor
  16.9–20.5) / 3155 s, `x_r/h` 8.3611, peak 12.1 GiB — a different march (rounding), the same root; the
  re-fit is 58 % of it. Why the probe is slow there: its batched JVP touches 4.4x pitzDaily's bytes per
  tangent at ~23 GB/s on 2-way-partitioned kernels, and faults its 816 MiB temporary in afresh every call
  (~a quarter of the probe); `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` removes the faults at an unchanged
  footprint (re-fit 48.7 → 39.7 s at `state-00032`; a cadence-5 march without the pivot floor 2439 → 2180 s, peak 12.63 → 12.15 GiB), the retaining allocators are not usable for a march
  in 15 GiB: `.claude/notes/validation-log.md`, "bfs3d on a 4-core Linux container" and the subsection after it.

  **What decides it is cell aspect ratio, through the pressure column** (2026-10-26, `validation/reach_folding.py`);
  the distance-3 couplings exist at all because bfs3d is graded (the gradient's second pass, `schemes.md`).
  Reach is graph distance; across a stretched cell the cell three faces away can couple more strongly
  than the face neighbour along its long side, so dropping the distance-3 pressure couplings removes some
  as large as couplings kept. bfs3d's largest dropped `|C_ij|` is 1.3x the row's weakest face coupling at
  the median in cells whose spacing ratio exceeds 10:1 (41 % of its cells) and 0.23x in 2–10:1 ones (58 %);
  pitzDaily's is 0.3–0.9 % in its 98.5 % of cells below 10:1 and 0.5x in its 183 stretched ones. Only p matters: probing p alone
  at 2 reproduces bfs3d's failure, which is a stagnant solve at the harder inner iterates, not a slower
  one; the drop alone stagnates it, the fold alone does not, and neither the coarsening nor float32 is
  involved. A block-norm measure of the fold (bfs3d 2x pitzDaily) cannot predict this. Re-measure on
  any case with strongly graded cells before shortening p. Why a short pressure probe fails only
  in the march: a fit made from it is as good as reach 3's at the state it was fitted at, but loses
  heavily one or two inner solves later, and the part that loses is its coarse levels (Galerkin sums
  over pressure rows that nearly cancel), not the fine smoother; no fine-level smoother setting rescues
  it (log, "Why a reach-2 pressure fit fails in the march"; `validation/reach_research/`). pitzDaily arms and the
  reach-4 pattern: log, "Probe colouring reach 2 — pitzDaily and bfs3d"; the aspect-ratio measurements:
  same section, "WHY bfs3d NEEDS REACH 3".
- **Keep the materialized pattern at reach 3 even where a column is probed shorter.** A reach-2 pattern
  changes the aggregation's coarse space (3 levels / 480 coarse equations against 2 / 1296 on bfs3d) and
  diverged the then-shipped pattern-dependent smoother; positions, not values, were causal, and a stored
  exact zero is not a substitute (sparse arithmetic prunes it). Re-measure under the traced hierarchy
  before lowering it; ledger: `solve-refuted-directions.md`, reach-2 under "Low-β directions". Under the
  traced split, zeroing the VALUES beyond reach 2 (positions kept) costs 1.2–1.9× the applications on
  bfs3d, and beyond reach 1 does not converge at all (ledger, "TRUNCATING the Jacobian to reach 1").
- ⚠️ **A short reach both truncates and corrupts.** A colouring is collision-free only for the pattern
  it was built at, so two cells sharing a reach-2 colour can both couple to a row at distance 3; the
  response holds the sum and the de-compression charges it to the near entry. So pattern entries outside a
  column's own reach are written as explicit zeros (`ColumnProbePlan.in_reach`), never gathered: each far
  coupling is removed from its own position AND added onto a near one. On bfs3d the removal is what
  breaks the solve (above). Pinned by
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
