---
paths:
  - "aquaflux/solve/lu_preconditioner.py"
  - "aquaflux/solve/sparse_jacobian.py"
  - "aquaflux/solve/ilu0.py"
  - "aquaflux/solve/_ilu0.pyx"
---

# Rules — `aquaflux/solve/` direct preconditioners (complete-LU) and Jacobian materialization

> Split out of `solve.md` (2026-08-18) to keep routine `aquaflux/solve/` work from loading the
> full complete-LU and coupled-Jacobian-materialization investigation narrative. See `solve.md` for
> the package-wide contracts, current configuration, and binding decisions this file assumes.
>
> **The monolithic ILUT preconditioner (`ilut_preconditioner.py`) was DELETED (2026-08-18) — dominated
> by the complete LU at 2D and by the algebraic multigrid at 3D, per its own docstring and per the
> "nothing selects the monolithic ILUT or complete LU for the adjoint" finding in
> `solve-amg-multigrid.md`, and selected by no shipped case bundle.** `MonolithicIlutPreconditioner`,
> `IlutFactors`, `factorize_ilut` and the coupled builders `coupled_ilut_continuation` /
> `coupled_ilut_refreshing_continuation` / `ilut_beta_tracking_refresh` no longer exist; the shared
> `_beta_tracking_refresh` skeleton and `MonolithicFactorShiftPolicy` they used are unchanged and now
> serve only the complete LU and the algebraic multigrid. If you are looking for any of those symbols,
> they are gone rather than renamed — the code is in git history.
>
> **This file has no `-log.md` sibling yet — current facts and dated investigation entries sit
> together.** If you are about to push it past ~1,800 lines, split it first: peel the dated/historical
> content into a new `solve-direct-preconditioners-log.md` (no `paths:` frontmatter) and leave a
> current-status summary here, following the pattern in `solve-flow-block.md` /
> `solve-flow-block-log.md`. See `solve.md`'s "Where new content goes".

## Preconditioner — the frozen host family (shared contract)

- **`equilibrate_cell_major` / `equilibrate_ordered` live in `frozen_operator.py` (binding, moved
  2026-08-15). ⚠️ `cell_major_permutation` moved AGAIN on 2026-08-17, to the new
  `solve/ordering.py`** — it is one elimination ordering among several now, and sat in `frozen_operator`
  only because that is where it was first written. `frozen_operator` imports it (one direction, no
  cycle); nothing re-exports it from its old home, so a stale import fails loudly.
  They are the reorder half of one transform whose rescale half
  (`symmetrically_equilibrate`, `equilibration_scale`, `apply_symmetric_scale`, `row_chunks`) was
  already there, and every consumer applies the two together -- a factorization or a coarsening wants
  the matrix both unit-diagonal and grouped by cell. Consumed by the multigrid V-cycle
  (`amg_preconditioner.py`, `host_vcycle.py`) and the block field split (`field_split.py`); the complete
  LU needs neither (its own fill-reducing pivoting and ordering already handle the indefinite saddle).
  - **Both are exported from `aquaflux.solve`.** They were internal by `__all__` yet deep-imported
    by study harnesses, i.e. public in practice and unguarded in principle; the harnesses now
    take them from the package surface. The permutation's unit test moved with the function, into
    `test_frozen_operator_scaling.py` beside the rescale half it belongs with.


- **The host preconditioners share ONE application path and ONE declared contract —
  `solve/host_preconditioner.py` (BUILT, 2026-08-14).** The complete LU and the AMG V-cycle
  differ entirely in how the inverse is *fitted* and not at all in how it is *applied*, so
  `HostPreconditioner` owns `__init__` and `matvec()` and each subclass supplies only `build` /
  `refresh_in_place`. Those two genuinely differ (different inputs, different refresh costs) and are
  deliberately **not** unified behind a signature that would be the union of both.
  - **`HostFactors` is the contract, and it is exactly `n_dofs` + `apply(residual, *, transpose=…)`.**
    That pair is a real structural contract satisfied by **five** classes — the complete-LU factors,
    `AmgVCycle`, `NativeHierarchyInverse` and both `BlockTriangularFieldSplit`s (two further members, the
    monolithic ILUT and the Vanka smoother, have since been deleted, both dominated on every arm
    measured) — and declared by none of them individually, so `matvec` would otherwise be written out
    once per class and
    `FieldSplitAmgPreconditioner` obtained it by subclassing a *concrete sibling*.
  - **⚠️ Anything a base reads off `self.factors` beyond that pair is a requirement on ALL of them
    (binding).** This is not hypothetical: `has_native_solve` read `self.factors.has_native_solve`,
    which only `AmgVCycle` has, so the property **raised** on the field split — and both call sites ask
    through `getattr(pc, "is_exact_native", False)`, whose default swallows an `AttributeError` raised
    inside a property body exactly as it swallows a missing name. The answer it produced was
    accidentally the correct `False`. If a capability is not in `HostFactors`, answer it on the
    subclass. Pinned by an AST check on the base's own source
    (`test_the_base_asks_its_factors_for_nothing_beyond_the_declared_contract`) — read off the source
    rather than exercised, because the failure is a lookup that is *never taken* on the paths a test
    would naturally drive, which is why the original went unseen.
  - **The pseudo-transient shift has one home: `sparse_jacobian.shifted_jacobian`.** Every host
    preconditioner adds `β d` before factoring, and two spellings once disagreed: a pattern-preserving
    `setdiag` against `a + sp.diags(shift)` — the latter is wrong, since a sparse *addition* stores only
    entries whose result is nonzero and so drops the explicit zeros a fixed-pattern probe deliberately
    kept. **Measured:** the two spellings are identical in values *and* pattern on a full
    diagonal and on a matrix with diagonal entries missing (both create them), and differ only where
    explicit zeros are stored — so adopting `setdiag` everywhere is a correctness fix in general. The
    whole refactor is **bit-identical** end to end: a complete LU built and refreshed under both
    implementations returns byte-equal `matvec` and transpose output.

## Materializing the coupled Jacobian (`sparse_jacobian.py`)

**This section documents `sparse_jacobian.py`'s coloured-probe materialization of the coupled Jacobian —
shared infrastructure consumed today by the complete LU and (mostly) by the monolithic algebraic
multigrid.** It originally grew up beside the now-deleted monolithic ILUT (the alternative to the
block-triangular SIMPLE preconditioner for the coupled saddle, which formed the true Schur coupling
`B F⁻¹ G` through its incomplete-LU fill rather than approximating it); that preconditioner is gone
(dominated by the complete LU at 2D and by the algebraic multigrid at 3D, per its own docstring and per
the "nothing selects the monolithic ILUT or complete LU for the adjoint" finding in
`solve-amg-multigrid.md`), but the Jacobian-materialization machinery below did not go with it — the
complete LU and the AMG's coloured probe both still depend on it.

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

    - **⚠️ THE GATHER MAP'S OWN CONSTRUCTION PEAKS HIGHER THAN ANYTHING ELSE IN A RUN, and the entry
      above is about the MATERIALIZE, not the build (measured 2026-08-14).** The de-compression entry
      says "the peak is the matrix plus one chunk", which is true *per materialize* and was read as
      though it described the whole path. It does not: `block_stencil_gather_map` plus
      `ProbeGather.__init__` run **once**, at case construction, and cost **100 bytes per pattern
      entry** in transient peak. The retained `ProbeGather` is ~12 B/entry, so the recorded "gather map
      ~1.2 GB" is the *retained* delta and understates the peak by roughly four times.

      *Configuration:* synthetic 14³ = 2744-cell lattice, reach-3 block-stencil pattern, 6 fields,
      143 464 cell blocks, 5 164 704 entries (`bfs3d`'s shape at ~1/9 its size); peak resident set via
      `ru_maxrss`, **one arm per process**, 5 interleaved repetitions on an idle machine, minimum
      reported. The pre-change arm is the previous commit's module imported side by side with the
      shipped one, so both are the real code rather than a re-typed stand-in. Harness
      `validation/bfs3d_openfoam/gather_map_memory.py`. ⚠️ The first run of a session reads low (a cold
      allocator gave 47 B/entry where the same arm repeated at 71) — discard it, as with every other
      probe in this file.

      | | before | after |
      |---|---|---|
      | transient peak | 517 MB (100.1 B/entry) | **339 MB (65.7 B/entry), −34%** |
      | retained | 62 MB | 62 MB, unchanged |
      | build time | 0.31 s / 0.26 s | 0.27 s / 0.24 s (**0.85× / 0.93×**) |

      Scaled to `bfs3d`'s 47.2M entries that is roughly **4.7 GB → 3.1 GB**, which makes this the
      largest single allocation in a run either way — the case build after the blocked wall-distance
      search is ~850 MB. All five repetitions of the pre-change arm read **100.1 B/entry to the
      decimal**, and reproduce the value measured earlier while a march was competing for the machine:
      peak resident set is an allocation high-water mark, so unlike wall clock it is insensitive to
      contention. Use it, not timing, when a probe must run beside other work.

      Two causes, both incidental rather than structural. The build routed an **integer** index array
      through `float64` purely to use `coo_matrix.tocsr()` as a sorter and converted back to `int64` —
      four full-length arrays at double the necessary width — and it applied the out-of-reach mask with
      `np.where`, allocating a second copy of a grid that can be written in place. Every index here has
      a bound known before it is formed (`zero_slot = n_probes · nf`), which is what lets the width be
      chosen up front (`_index_dtype`) so the wide form never materializes. **32-bit is not assumed:**
      the type falls back to `int64` above `iinfo(int32).max`, which no mesh this path is used on
      reaches (`bfs3d` is 78M against a 2.1e9 ceiling) but a much larger one would. Choosing from the
      bound also removed a latent wrap: `ProbeGather` narrowed to `int32` *unconditionally*, which
      silently truncates on a mesh that does not fit rather than falling back.

      **⚠️ FIXING THE BUILD ALONE WOULD HAVE LEFT MOST OF THIS ON THE TABLE, because
      `ProbeGather.__init__` has its own ~57 B/entry peak that was HIDDEN under the build's.** Measured
      per phase before the change: the compressed-sparse-row build 56.0 B/entry and the regrouping
      ~57 B/entry, the second invisible end to end because it fitted under the first. That is the
      transferable part — a phase whose peak sits under a larger one cannot be seen from outside, and
      becomes the ceiling the moment the larger one is reduced — and it is why both were narrowed in
      one change.

      **Verified identical, not merely convergent.** Against the pre-change module on the same plans:
      `indptr`, `indices`, `_position`, `_source` and `_probe_start` all `array_equal`, **and** the
      `scatter` output agrees on random responses — on the uniform-reach path *and* on the
      `(3,3,3,3,2,2)` per-column path, which is the one that exercises the out-of-reach mask where the
      in-place write lives. Fast (1043 passed), `slow` (46) and `validation` (18) tiers all green.

      **✅ THE SORT PERMUTATION IS GONE — grouped via a `scipy.sparse` COO→CSR conversion instead of
      `np.argsort` (BUILT).** `np.argsort(probe_of)` forced a full-entry-count `int64` array regardless
      of every other array's chosen width — the one remaining full-width allocation the narrowing above
      could not reach, since numpy's `argsort` has no narrower output form. Treating `probe_of` as the
      row index of a throwaway `scipy.sparse.coo_matrix` (shape `(n_probes, nf)`) and converting to CSR
      makes the grouping a counting sort over `probe_of`'s own range — a few hundred values — done in
      `scipy`'s compiled conversion rather than a numpy-level sort of the whole entry count; `grouped.
      indptr` is exactly `_probe_start`, so the separate `np.searchsorted` call is gone too. Order
      *within* a probe is free (`scatter` writes to unique positions), so the unstable grouping this
      gives is exact, not an approximation, and there are no duplicate `(probe, position)` pairs to sum
      (`position` values are already unique) — the same "no duplicates, so `tocsr` only reorders" property
      `block_stencil_gather_map`'s own `coo_matrix(...).tocsr()` call already relies on.
      **Verified by scatter *output*, not index-array equality — the check this entry itself said would
      be needed once this was taken.** A byte-for-byte copy of the pre-change `__init__` run
      side by side with the new one on the identical `gather_map` input, at a real reach-3, 6-field,
      27,000-cell pattern (`n_probes=582`, 56,272,032 pattern entries): `scatter()` on random responses
      agrees **to the last bit** (`max abs diff = 0.0`) between the two, while the old code's `by_probe`
      permutation alone is a measured **450.2 MB `int64` array that the new code never forms at all** —
      exactly `8 B × 56,272,032`, the full predicted size, at a scale where `_source`/`_position`
      together (each now `int32`) retain 450.2 MB total, i.e. removing this one array is worth as much
      as everything `ProbeGather` otherwise keeps, combined. Fast-gate `sparse_jacobian`/`amg_precondi-
      tioner`/`lu_preconditioner`/`frozen_operator_scaling` unit tests and the
      not-slow `coupled_amg`/`coupled_field_split`/`coupled_rans` integration tests all pass unchanged
      (harness not in the repository for the old-vs-new comparison script; the dtype/scatter-output
      tests that *are* in the repository, `test_gather_map_index_arrays_are_narrow_on_a_realistic_
      pattern` and `test_gather_de_compression_matches_the_scatter_loop`, needed no changes and still
      pass — they were already output-based rather than index-array-based).
      **Why this matters at production scale, not just as a percentage:** an independent scaling
      investigation (synthetic cubic lattices, reach 3, 6 fields, up to 262,144 cells, array-level
      `.nbytes` accounting to stay immune to the RSS/allocator noise recorded throughout this file) found
      this array's *share* of the gather-map build's transient peak growing from ~11–12% at `bfs3d`-scale
      (≤27,000 cells) to **91.5% of the transient at 262,144 cells**, extrapolating to **~17.6 GB at
      1,000,000 cells** — a number the 27,000-cell measurement above independently corroborates by simple
      linear scaling (450.2 MB × ~37 ≈ 16.7 GB). So this lever, recorded here for a long time as a minor
      deferred optimization, is at 1M-cell scale the single largest identifiable, deterministically-
      quantifiable transient contributor to the gather-map build — comparable in size to the entire
      retained `ProbeGather` structure itself.

      **⚠️ Read this before tuning `_PROBE_BATCH_SIZE` against memory.** Most of the peak the batch size
      was being traded against was never the batch: seeds and responses are both `n_probes x nf` and
      neither scaled with `probe_batch_size`. Both are now gone, so the peak is the matrix plus one chunk
      — which is what makes a memory-budgeted batch size a meaningful knob rather than a knob on the
      small term. The recorded "16 vs 4: ~2.2 GB vs ~0.7 GB" was measured against the old allocations and
      **no longer describes this code**; re-measure before using it.

    - **⚠️⚠️ PER-COLUMN PROBING REACH — shipped at `(3,3,3,3,2,2)`; `p` MUST stay at reach 3.**
      `(3,3,3,2,2,2)` diverges this case on its first step (issue #191), and the cause is not the reach's
      accuracy — the matrix is exact to the floating-point floor — but the *sparsity* the shortening
      leaves behind. Keeping those positions as stored zeros does cure the divergence and is **not
      available**, because the zero-fill incomplete factorization cannot be handed stored zeros; see
      *"stored exactly-zero positions break the ZERO-SHIFT V-cycle"* below. Read the root-cause section at
      the end of this bullet before using anything in it. The 564 → 399 figure and the 5.7e-16 Frobenius agreement below are both real and both fail
      to predict the divergence, which is the instructive part — the matrix was exact and the *sparsity*
      was not.

      The probe
      is charged per (colour, **column** field), so the reach is worth choosing per column rather than
      once. Measured with `validation/bfs3d_openfoam/probe_reach_audit.py` at three states — a cold inner
      iterate (`inner-00003-02`), a developed step (`state-00072`, step 33, ‖R‖ 3.1e-6, shift 0.0068) and
      the march's hardest solve (`inner-00050-03`, 15 cycles, α → 0) — the share of each column's norm
      lying **beyond reach 2**:

      | arm | stored pattern | exact zeros | after equilibration | lost |
      |---|---|---|---|---|
      | uniform reach 3 | 47 209 392 | 8 487 718 | **38 721 674** | 18.0 % |
      | `column_reach=(3,3,3,2,2,2)` | 47 209 392 | 14 899 552 | **32 309 840** | 31.6 % |

      **6 411 834 positions — 16.6 % — that the uniform arm keeps are stripped from the operator the
      smoother factorizes**, and a zero-fill incomplete factorization takes its pattern from exactly that.
      (38.7M also matches the independently recorded live-nnz figure, which corroborates the reading.)
      **This retro-explains the padding experiment that established the reach-3 requirement:** it padded
      the distance-3 shell with **~1e-30, not exact zero** — precisely what survives the product. That is
      why positions read as causal and values did not.
      It equally explains the p/kω asymmetry: pressure is in the incomplete-LU-smoothed **leading** block,
      while k/ω are in the trailing block, whose native cell-block inverse has no factorization pattern to
      weaken.
      ⚠️ **It also violated the "full pattern, no `eliminate_zeros`, so the structure is truly fixed"
      invariant the in-place GAMG refactor relies on — for EVERY arm**, since the pruned set is whichever
      entries happen to be exactly zero at that state, which moves with the state.
      **✅ FIXED — pattern-preserving assembly, and there were TWO pruning sites, not one (BUILT).**
      Sparse *arithmetic* is what prunes, so both the shift add and the equilibration had to change:
      - `frozen_operator.apply_symmetric_scale(data, indptr, indices, scale, chunks=…)` scales the stored
        values in place (row-chunked, so the row factor's per-nonzero expansion never allocates an array
        the size of the values). `symmetrically_equilibrate` uses it in place of `diags(s) @ a @ diags(s)`.
      - `MonolithicAmgPreconditioner._shifted` adds the shift by **diagonal assignment**, not
        `a + sp.diags(shift)` — a sparse **addition** drops explicit zeros just as the product does, so
        the zeros were being lost before the equilibration ever ran. Found only because the bit-identity
        test kept failing after the first fix.
      - `ShiftedCellMajorOperator.assemble` already had the correct chunked form privately; it now calls
        the shared helper, and `_row_chunks` moved to `frozen_operator.row_chunks` rather than being a
        second copy.
      Strictly cheaper than the sparse products it replaces. Pinned by `test_frozen_operator_scaling.py`
      (pattern preserved including explicit zeros, values bit-identical to the product where it stores
      them, **and an explicit test that the old product form would have dropped them**, so it cannot
      quietly return).
      ⚠️ **The bit-identity test had SEEN this and worked around it:** it compared the two paths as dense
      arrays under the comment *"the fast path keeps the full pattern's explicit zeros, so compare as
      dense rather than by nnz"* — a dense comparison passes whether or not a path drops positions. It now
      compares `indptr`/`indices`/`data`, and against the production `_shifted` rather than a raw
      `jacobian + sp.diags(shift)` written in the test. **A workaround in a test is a defect report;
      read it as one.**
      **✅ CONFIRMED ON THE CASE — the acceptance test passes bit for bit.** With the fix and the case
      back at its `COLUMN_REACH=(3,3,3,2,2,2)` default, `bfs3d` rung-1 step 1 reproduces the uniform-reach
      baseline in **every reported digit**: `‖R₀‖` 3.2901e-01, β 0.5000, 4 inner, **3 cycles**,
      `‖R‖` 2.046e-01, α 1.000, no flags — against the diverged arm's 44 cycles, `‖R‖` 3.758e-01, α 0.000.
      So the values were never the problem and the stored pattern always was.
      **Expect a cost shift, not only a win:** the trailing block carries ~37 % more stored entries
      (5 245 488 against 3 839 276) into its factorization than it did when equilibration pruned them, so
      per-cycle cost there may rise even where cycle counts fall. Judge on march wall clock, not cycles —
      and **not on the confirming run above, whose wall clock is void** (forced past the load guard with
      the test tiers running; cycles and step counts are unaffected, timings absorbed the contention).
      Still open: a full march to mid-span `x_r/h` 8.361, and what the restored pattern does to the
      `equilibrate=True` arm, which is a different question from this one.
      **THE MECHANISM IS COLOUR ALIASING, and it is the MAJORITY of a shortened column, not an edge
      case (measured structurally 2026-08-12, `validation/bfs3d_openfoam/column_reach_collisions.py` —
      mesh graph only, no state, no Jacobian, no solve).** A colouring is collision-free only on the
      pattern it was built at, so under the reach-3 assembly pattern two cells sharing a reach-2 colour
      can both couple to one row; the response holds their **sum** and de-compression charges all of it
      to whichever lies in reach. On `bfs3d` (23040 cells, 1 311 372 pattern cell-blocks):

      | column | reach | entries in reach | corrupted | share | far entries folded in | rows touched |
      |---|---|---|---|---|---|---|
      | u, v, w | 3 | 1311372 | 0 | 0% | 0 | 0 |
      | p, k, ω | 2 | 537896 | 287315 | **53.4%** | 369396 | **23025 of 23040** |

      **By this measure `p`, `k` and `ω` all close inside reach 2 and the velocities do not.** The
      table is kept because a column-norm measure does **not** license shortening a column — it collapses
      over row fields, and on this system the ω rows set it alone — so read it as evidence about the
      method, not as the licence. The shipped value is `(3,3,3,3,2,2)` — `p` at reach 3 with only k and ω
      shortened, the pattern held at reach 3 — giving **454 probes against 564** and a **−16 % probe cost
      per build**.
      **Why it is a case property and not a library constant:** it follows from the *schemes*. First-order
      upwind on k/ω plus a non-orthogonal diffusion correction reaches 2; the velocity columns carry the
      limited second-order upwind reconstruction out to 3. Re-measure for any case that changes them —
      the library default (`column_reach=None`) probes every column at `stencil_reach`, unchanged.

      **⚠️ THE MEASUREMENT THAT SETS THE DEFAULT (2026-08-12).** At `(3,3,3,2,2,2)` the `bfs3d` march took **44 restart cycles
      at step 1 instead of 3**, α collapses to 0.000, and by step 2 β is at its 16.0 ceiling with ‖R‖ an
      order of magnitude ABOVE its start. The p row is the signature: **1.402e-01 after step one against
      6.217e-07**. Restoring p to reach 3 — `(3,3,3,3,2,2)` — reproduces the uniform-reach trajectory
      *exactly* (identical ‖R‖, cycles, β and α over the first 24 steps to the log's four figures) and
      keeps a **−16 % probe cost per build** (8.4 s against 10.0 s; 13.9 s at the pre-#188 baseline).
      Verified 67 steps / 319 cycles / 1960 s, mid-span `x_r/h` 8.36 unchanged.

      **Why the audit above did not catch it, as a HYPOTHESIS and not a measurement.** The shell norm is
      a **column-relative** quantity — the share of a column's norm beyond the reach — while the damage
      from folding is **entry-relative**: the aliased mass lands on particular positions, and what
      matters is its size against *the entry it lands on*. In an incompressible collocated saddle the
      (p,p) block is near-zero by construction (it carries only the Rhie–Chow damping), so folding even
      1e-15 of column mass onto it can be a large relative perturbation of exactly the entries an
      incomplete LU turns into pivots. That would explain why p audits as clean at 9.8e-16 and still
      breaks the smoother, while k and ω — whose columns are read only by the turbulence rows, under a
      per-cell block inverse — are genuinely inert. **This is not established.** What would settle it is
      an entry-relative audit (folded mass over the magnitude of the receiving entry, per block) rather
      than a column-norm one; `probe_reach_audit.py` reports the column-relative measure only.

      **The safe reading until then: a shell norm licenses shortening a column only when that column's
      entries are not themselves near-zero.** Shortening `k`/`ω` is measured safe on this case;
      shortening `p` is measured catastrophic on this case; the two are not distinguished by any number
      in the table above.
      **⚠️ "Safe" here means AT POSITIVE SHIFT, which is the only regime a march visits. At β = 0 —
      the adjoint's operator — shortening `k`/`ω` is measured to DOUBLE the cost** (22 restart cycles
      against uniform reach's 11 to the same 1e-11 floor; see the stored-zeros entry below for the full
      configuration). It still converges, so this is a cost to know about rather than a reason to widen
      the march's reach, but do not carry "proven safe" across the β boundary.
      **⚠️ AND `(3,3,3,2,2,2)` IS ONLY SOUND IF THE FACTORIZATION KEEPS STORED ZEROS, WHICH IT MUST NOT.**
      Shortening `p` is safe only when the dropped positions are held in the pattern as stored zeros —
      and stored zeros are exactly what `AmgVCycle._live` removes, because the incomplete factorization
      cannot take them. So with the operator pruned at the factorization the `p` column is back in the
      configuration that diverges the case, and the case default must stay `(3,3,3,3,2,2)`. **Predicted
      from the mechanism, not re-measured on a march** — the divergence itself is the measured #191
      result, at β > 0 under a pruning assembly.
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
        `CoupledJacobianProbe(gradient_sweeps=n)` / the coupled builders' `probe_gradient_sweeps=n` cap
        the sweeps **for the coloured probe only**, leaving the Krylov matvec the exact jvp of the full
        residual — so the recovered matrix is exact for what it was taken from, and the converged state
        and its adjoint are untouched. `None` (default) is byte-identical. See
        `.claude/rules/schemes.md` for `narrow_gradient_sweeps` and the reach/accuracy tables.
      - **⚠️ Do NOT read the cap as "less far coupling".** Measured, the mass beyond a given distance is
        set by the **skewness**, not the sweep count (at 25 % perturbation, `|dR/dφ|` beyond distance 2
        is 9.80e-4 at 2 sweeps and 1.004e-3 at both 4 sweeps and the exact solve). A shorter sweep
        *relocates* that mass inward. What the cap buys is that the probe is no longer aliasing — the
        same distinction as everywhere else in this section.
      - **⚠️ AND THE CAP IS A TRADE, NOT A FREE WIN — it is byte-identical at `None` and a genuine
        exchange otherwise, because narrowing makes a ZERO-FILL elimination harder.** On pitzDaily, an
        ILU(0) pivot census of the equilibrated cell-major leading `[u,v,p]` block against the narrowed
        operator (`hybrid_initialize` seed, field split, same everything else):

        | probe sweeps / reach | negative pivots | min \|pivot\| | nnz |
        |---|---|---|---|
        | full (4) / 3 | 7 | 5.02e-02 | 6.32M |
        | full (4) / 5 | 9 | 2.18e-02 | 13.32M |
        | 2 / 5 | 34 | 2.76e-02 | 7.34M |
        | **1 / 3** | **263** | **3.59e-04** | **4.72M** |

        At `sweeps=1` / reach 3 the matrix is **exact to the float64 floor with 2.8× fewer nonzeros**
        than the reach-5 matrix the case otherwise needs — a bigger prize than the aliasing fix alone —
        while the negative-pivot count goes 9 → 34 → **263**. That is consistent with the "truncation
        relocates far coupling inward" finding above: the mass lands on the near entries an incomplete
        factorization turns into pivots. **So `probe_gradient_sweeps` is preconditioner-family-dependent.**
        ⚠️ **A pivot census is a PROXY and the direct test — a one-apply or a march at `sweeps=1` — has
        NOT been run.** This file separately records a case where a pivot census was *identical* across
        arms that converged five-fold apart, so do not treat it as decisive; here the arms do separate,
        which is why the direction is worth carrying.
      - **The family it is FOR is the SIMPLE-smoothed hierarchy, which takes the saving for nothing.**
        The mechanism is which preconditioners inherit the stored *sparsity*: an incomplete factorization
        takes its pattern from it, so a corrupted or narrowed pattern gives a correspondingly different
        factor, while `native_saddle_inverse` relaxes through diagonal and Schur approximations and takes
        no pattern at all. Measured on pitzDaily, that arm is **reach-insensitive** — the same 71 steps at
        reach 3 and reach 5, cycles within 3 % (395 vs 408), identical final residual — while the Jacobian
        halves and the march is 31 % shorter. **The untested pairing worth running is
        `probe_gradient_sweeps=1` + `stencil_reach=3` + a SIMPLE-smoothed hierarchy:** exact Jacobian,
        4.72M nonzeros against 13.32M, and a preconditioner indifferent to the pattern.
      - **🛑🛑 SCOPE BANNER — READ BEFORE CITING ANY ZERO-FILL RESULT BELOW. THE 2026-08-16 SWEEPS ARE
        MONOLITHIC AND THE CASE IS FIELD-SPLIT, SO THEY DO NOT DESCRIBE THE SHIPPED PRECONDITIONER.**
        `validation/pitzdaily_openfoam/ilu0_remedy_sweep.py` and its siblings build **one `AmgVCycle`
        over all five fields interleaved cell-major**. The pitzDaily case runs `field_split=True`, and
        with `FLOW_INVERSE = "petsc"` (its default) the split sends:
        - the **`[u, v, p]` saddle** to the PETSc AMG V-cycle — *this* is the only block `FILL_LEVELS`
          governs, and the only place an incomplete factorization happens at all;
        - the **`[k, omega]` pair** to `native_nodal_inverse`, **which is not an ILU**.

        Three consequences, all binding:
        1. **Every arm in the sweeps below — the fill ladder, the orderings, the shifts, the reach arms,
           `condest`, the diagonal-dominance census — was measured on a preconditioner this case does not
           use.** They are results about a monolithic five-field ILU. Do not quote them as properties of
           pitzDaily's solver.
        2. **The `omega` findings from that work are irrelevant to this case**, because no ILU ever
           factorizes an `omega` row here. The 2026-08-16 census reporting "100 % of `omega` rows are not
           diagonally dominant, median ~5×, tail past 400×" is a fact about a matrix the shipped
           preconditioner never factorizes. It is **not** evidence for the ω-is-the-lever thread.
        3. **A "field split would fix it" hypothesis is already refuted by the case itself** — pitzDaily
           *is* split and still sets `FILL_LEVELS = 1`. Whatever forces fill 1 does so **inside the
           `[u, v, p]` block**.

        **What survives, correctly scoped to that block:** the same census puts `p` at **0 % of rows
        non-dominant at every shift** and `u` at **61 % (β = 0.5) rising to 95 % (β = 0)**, with `v`
        between. So the saddle is not where this block is weak — the **streamwise momentum rows** are,
        and momentum advection here is `LimitedUpwind` (second order), which is the regime Elman (1986)
        identifies for incomplete-factorization instability. **The open, correctly-posed question is
        therefore: within `[u, v, p]` alone, does ILU(0) become viable if the preconditioner's operator
        uses first-order upwind advection while the Krylov operator stays the exact second-order
        Jacobian?** That keeps fill at zero and is the substitution the frozen AMG operator already makes
        elsewhere for this reason. **Not measured.**
      - **⚠️ The ILU fill ranking INVERTS between the two cases** — `bfs3d` wants zero fill (its ILU(1)
        diverges at low shift, 303 negative pivots against zero), pitzDaily wants fill 1, where **two
        independent zero-fill implementations fail identically** (PETSc ILU(0) and the native
        `HostVCycleInverse`, both α → 0 by step 3–4), putting it on the fill rather than on anything
        PETSc-specific. pitzDaily's converging arms all land `x_r/h` 8.0686, so those are cost
        comparisons, not accuracy ones. ⚠️ Both cases set `field_split=True`, so this comparison is
        between two **flow-block** factorizations — which is the one thing about it the monolithic sweeps
        above cannot speak to.
        **⚠️⚠️ BUT THE SKEWNESS AND THE PROBE ALIASING ARE NOT WHY, AND AN EARLIER VERSION OF THIS ENTRY
        IMPLIED THEY WERE (corrected 2026-08-16; two independent measurements, one reproduced by hand).**
        A one-variable skewness sweep — coupled RANS on `perturbed_grid_2d`, 384 cells, pitzDaily's whole
        scheme bundle, *only* the interior-node displacement moving, PETSc GMRES(30) at
        `KSP_NORM_UNPRECONDITIONED` with the true residual recomputed in `scipy` — is **flat to the
        iteration** across and beyond pitzDaily's own skewness:

        | perturbation | 0.0 | 0.05 | 0.3 |
        |---|---|---|---|
        | max \|skew\|/d | 1.5e-13 | **9.3e-2** (above pitzDaily's 7.5e-2) | 5.4e-1 |
        | ILU(0) its, β = 0 / 0.05 / 0.5 | 17 / 16 / 14 | **17 / 16 / 14** | cap / cap / 18 |
        | ILU(1) its, β = 0 / 0.05 / 0.5 | 11 / 10 / 9 | 10 / 10 / 9 | cap / cap / 13 |

        The first amplitude that breaks anything is **4× pitzDaily's max**, and there *both fills fail
        identically* — so it is not fill-selective at any amplitude. Independently: swapping
        `CorrectedGreenGauss`→`CompactGreenGauss` on pitzDaily takes the reach-3 probe error on the
        **pressure column** from 6.3e-7 to **2.6e-16** — the matrix becomes exact, no aliasing left — and
        the fill ranking does not move (β = 0.05: ILU(0) 208→177, ILU(1) 52→52).
        **So the reach finding above stands on its own** (pitzDaily really does materialize a wrong
        matrix at reach 3) **and explains nothing about the fill ranking.** Keep the two apart.
        **⚠️ NOR IS IT THE PIVOTS, which is the account this file carried.** At the *developed* state
        (the OpenFOAM time-accurate field on the same mesh) **both factorizations have zero negative
        pivots and identical min |pivot| ≈ 2.7e-1 … 3.2e-1 at every β** — and still converge **12×
        apart** (β = 0: ILU(0) caps at 1.9e-4, ILU(1) 479; β = 0.05: ILU(0) caps, ILU(1) 52). The
        recorded "pitzDaily's ILU(0) has negative pivots at every shift" is real at the *cold seed* only,
        and is a coincidence: remove the negative pivots entirely and the gap **widens**. This is the
        third entry in this file where a pivot census was measured blind to what separates the arms.
        **And the literature says the same thing from theory, which is why the census keeps failing us
        here:** Chow & Saad (*J. Comput. Appl. Math.* 86(2), 1997) found that unstable triangular solves
        — `‖L⁻¹‖` large — **occur without the presence of small pivots**, and proposed
        `condest = ‖(LU)⁻¹e‖_∞` (one triangular solve on the all-ones vector) as the trigger instead.
        Elman (*Math. Comp.* 47, 1986) is the origin for convection-dominated problems: ILU fails through
        *instability of the factors*, not through pivot loss. **So stop censusing pivots on this operator
        and measure `condest` per fill level.**
        **⚠️⚠️ "NEGATIVE PIVOTS AT EVERY SHIFT" ALSO NEVER TESTED A SHIFT ON THE ROWS THAT MATTER — two
        source-verified facts (2026-08-16).** (i) `CoupledShiftPolicy`'s base diagonal is **identically
        zero on the continuity row** (`turbulence/coupled.py`: `self.momentum.pack(…, jnp.zeros(n_cells))`,
        and the comment says so — "0 on pressure"). The operator is then equilibrated to a unit diagonal,
        so raising β changes the *velocity/k/ω* rows' relative dominance and leaves the pressure row
        exactly as it was; the PC-only `beta_floor` inherits the same blind spot. (ii) **PETSc's `PCILU`
        defaults to `MAT_SHIFT_NONE`** — verified by `pc.view()`, which prints no shift line for `ilu`
        while `icc` prints "using Manteuffel shift [POSITIVE_DEFINITE]" — and the default `zeropivot` is
        **2.22e-14**, so `NONZERO` and `INBLOCKS` (whose triggers are `|pivot| ≤ zeropivot·rowsum` and
        `|pivot| ≤ zeropivot`) are **inert** against pitzDaily's min |pivot| of 5e-2. **A
        factorization-level shift is the only mechanism in the stack that can reach the continuity row,
        and it has never been switched on** (`extra_options={"mg_levels_pc_factor_shift_type":
        "positive_definite"}`, no code change).
        **⚠️ BUT DO NOT REACH FOR THE SHIFT FIRST — the primary sources rank a scalar diagonal shift LAST
        for an indefinite matrix, and say why.** Chow & Saad (§5, read from the UMSI-97-95 preprint) reject
        it outright: a shift `A + αI` "may shift the eigenvalues of A arbitrarily close to the origin", and
        **a shift of α may *decrease* the magnitude of the pivot** — their remedy is instead a *dynamic,
        per-row, sign-following* perturbation (replace any pivot below a threshold by the threshold
        **carrying the pivot's own sign**), which is the same sign-awareness IFPACK ships as
        `sgn(a_ii)·α`. Manteuffel's existence theorem assumes positive definiteness, which this operator
        lacks, and Saad's Fig. 10.18 shows the iteration-vs-shift curve is U-shaped with **both** endpoints
        of the obvious search wrong (the diagonally-dominant shift is "too large in general"; the smallest
        admissible one is "not a viable alternative"). The published ranking for this matrix class is
        **nonsymmetric permutation + two-sided scaling first** (Benzi's Table V: ILUT solves 0 of 5, ILUTP
        still fails 2 of 5, **MC64 + ILUT solves 5 of 5** at modest fill; Konshin–Olshanskii–Vassilevski
        independently call two-side Sinkhorn scaling "the most important" remedy for Navier–Stokes), then
        constraint-aware **ordering**, then pivoting, then sign-following perturbation, then the shift.
      - **⚠️⚠️ THE MOST ACTIONABLE THEORY RESULT: ELMAN (1986) PROVES ILU's STABILITY IS A MESH-PÉCLET
        CONDITION ON THE *DISCRETIZATION*, AND THAT FIRST-ORDER UPWINDING REMOVES IT ENTIRELY.** Verified
        against the primary (Math. Comp. 47(175):191–217, Table 1 and §4), on `−Δu + 2P₁u_x + 2P₂u_y`
        with `p_i = P_i h` the cell Péclet numbers:

        | discretization | ILU | MILU |
        |---|---|---|
        | centred, co-signed convection | stable ⟺ `p₁p₂ ≤ 1` | **unconditionally stable** |
        | centred, opposite-signed (recirculating / cross-flow) | `p ≤ 2+√3` | unstable already at `\|p\| > 1` |
        | **first-order upwind** | **always stable** | **always stable** |

        His Table 1 is the same story empirically: ILU *fails* on `P₁=P₂=50` where MILU takes 7 iterations,
        and MILU *fails* on `P₁=−50, P₂=50` where ILU takes 32 — each failure coinciding with an indefinite
        symmetric part of the preconditioned operator, though `A`'s own symmetric part is the definite
        Laplacian. **So "which fill is right" may be the wrong question and "what is the probed operator's
        cell Péclet number" the right one.**
        **This bears directly on a seam this codebase already has, and on an asymmetry nobody has read as
        one.** `frozen_operator.convection_diffusion_operator` **always upwinds first order** — a deliberate
        preconditioner-only choice, recorded in `CLAUDE.md` as what makes it an M-matrix a hierarchy can
        coarsen — so the *block* preconditioner factorizes an operator Elman proves is unconditionally
        stable. The **monolithic AMG path does not**: it materializes the *true* Jacobian by coloured
        probing, which carries pitzDaily's Venkatakrishnan-limited **second-order** upwind reconstruction —
        squarely in the regime where Elman's bound bites, and pitzDaily's recirculation is exactly the
        opposite-signed case with the tighter `2+√3` limit. **The experiment this suggests is cheap and
        architecturally free: probe a FIRST-ORDER-UPWIND variant of the residual for the preconditioner
        only** — the identical seam `probe_gradient_sweeps` uses (`CoupledJacobianProbe.narrow`), with the
        Krylov matvec still the exact jvp, so the root and the adjoint are untouched. **Not yet run.** Note
        it also predicts the case split: whichever case sits at higher cell Péclet in its probed operator is
        the one whose zero-fill factorization goes unstable.
      - **❌❌ BLOCK ILU(0) AT `bs = n_fields` IS REFUTED — by a published measurement on EXACTLY this
        problem class, which settles the repo's two contradictory memory notes.** Chapman, Saad & Wigton
        (TR umsi-96-14, 1996; *Int. J. Numer. Meth. Fluids* 33(6):767–788, 2000) ran precisely this
        substitution on Barth's **2D Navier–Stokes-with-turbulence Jacobians at block size 5**, verbatim:
        *"ILU(k) with padded blocks gives virtually identical results to BILU(k). This is to be expected,
        as the underlying preconditioners are the same in exact arithmetic."* Their tables: where point
        ILU(0) works, BILU(0) matches within 1 %; **where point ILU(0) fails, BILU(0) fails identically.**
        Independently reproduced here on a synthetic bs=5 operator — the two preconditioners' action
        agrees to **5.6e-16** — which is expected, since this codebase's stored pattern is already a union
        of *dense per-cell blocks* in cell-major order, and for such a pattern point and block ILU(0)
        coincide. So "BILU was properly refuted" and "block-ILU(0) is the predicted fix" are both
        explicable: blocking buys **only** intra-block partial pivoting, not a cure for the saddle.
        Blocking *does* win for **threshold** ILU, which is a different method. Two residues worth
        knowing: BAIJ storage makes CSW's *padding* free (a block is dense by definition), which is a
        genuinely different object from what ships, since `AmgVCycle._live` prunes stored zeros — but CSW
        measured padded vs unpadded as similar, and unpadded *better* on one case; and PETSc's
        fixed-block-size kernels carry a maintainer's own warning that the bs=5-adjacent un-permutation
        code "may also be buggy". **Shipping it is also blocked**: PCGAMG rejects a SeqBAIJ `Pmat`
        outright (`No method creategraph for Mat of type seqbaij`), and the level-PC workaround is
        incompatible with `refactor`'s `setUp()` on every refresh. **Do not build this.**
      - **✅ WHAT CSW MEASURED INSTEAD IS THE LARGEST EFFECT IN THEIR TABLES, AND IT IS THE LEVER THIS
        CODEBASE ALREADY HAS A SEAM FOR: build the preconditioner from a WIDER SOURCE STENCIL.** Same
        paper, same bs=5 N-S+turbulence matrices: building BILU(0) from the **distance-2** matrix instead
        of distance-1 took BARTHT2A from **545 → 62** GMRES steps and turned the **failing** BARTHS2A into
        **73**. That maps directly onto the reach-3-vs-reach-5 question here, and pitzDaily's reach-3
        matrix is separately measured to be *wrong in the pressure column* (1.99e-07). ⚠️
        Counter-evidence to put in the same table: `probe_gradient_sweeps=1` gives a float64-exact matrix
        at 4.72M nnz with **263** negative pivots, so *exactness* is not the mechanism and *sparsity* is.
      - **❌ CSW'S WIDER-STENCIL RESULT HAS NOW BEEN TESTED HERE, AND IT DOES NOT TRANSFER: THE DIRECTION
        REVERSES WITH THE SHIFT (2026-08-16).** The full text was read and their numbers above are
        accurate as quoted. But their comparison is between *two genuine discretizations*, and it is run
        at one operating point; ours is a shift-parameterized family, and across it the sign of the effect
        is not stable. Same sweep, same state, same everything but the reach the **preconditioner** is
        built from (the Krylov operator is the exact Jacobian-vector product in every arm):

        | β | PC at reach 5 (matched) | PC at reach 3 (narrowed) |
        |---|---|---|
        | 0.5 | 38 cycles, 3.63e+00 — **fails** | **3 cycles, 3.8e-11** |
        | 0.05 | **1 cycle, 2.3e-11** | 38 cycles, 3.08e-01 — **fails** |
        | 0 | 38, 9.99e-01 — fails | 38, 1.68e+00 — fails |

        At β = 0.05 CSW's direction holds and the matched stencil is the one that works; **at β = 0.5 it
        is exactly inverted**, and the narrowed preconditioner is the only zero-fill arm that converges.
        Neither reach spans the shift range a march traverses, so "match the preconditioner's stencil to
        the operator" is **not** a fix here, and neither is its opposite.
      - **⚠️ AND THE REACH ARMS ARE NOT ONE-VARIABLE — do not read the table above as isolating the
        smoother's pattern.** The aggregation is built from the same materialized matrix, so a reach-3 arm
        also carries a different coarse space; the coarse size is visibly different (1615 against 670).
        Smoother pattern and coarse space move together in those arms, and nothing run so far separates
        them. The observation (the answer flips with reach) stands; the *mechanism* does not follow from
        it. Separating them needs an arm that narrows only the smoother's factorization pattern while
        holding the hierarchy fixed, which has not been built.
      - **⚠️⚠️ READ THE TWO SUBSECTIONS ABOVE AND BELOW AT THEIR OWN REACH — the sweep below shows the
        answer FLIPS with it, so a number from one is not comparable with a number from the other.**
        Everything in the root-cause subsection above (the fill ladder, the compact-vs-corrected arms, the
        developed-state pivot comparison, the ordering and GAMG-size arms) was measured at
        **`stencil_reach=3`**, where pitzDaily's matrix is known to be *wrong in the pressure column*
        (1.99e-07). The sweep below runs at **reach 5**, where it is exact (1.50e-15). At reach 5 and
        β = 0.05, ILU(0) converges in **one cycle** — so the flat statement "pitzDaily wants fill 1" holds
        at reach 3 and **not** at reach 5, and the root-cause subsection's conclusion that zero fill is an
        insufficient approximation of the saddle is **confounded by the reach it was measured at**. It is
        not thereby refuted: the developed-state arm there (zero negative pivots, 12× apart) is a
        different and still-unexplained observation. **Both are kept, both are labelled; do not merge
        them, and do not cite either without its reach and its state.**
      - **✅ THE SWEEP HAS NOW BEEN RUN, AND NOTHING IS A FIX (2026-08-16). Every candidate that rescues
        one shift breaks another, and the shipped cell-major order is the best single choice.** Harness:
        `validation/pitzdaily_openfoam/ilu0_remedy_sweep.py` (kept in the repository so this can be
        re-asked). Configuration, stated in full because two of these have moved before: the case's own
        **self-start** seed (potential-flow + Laplace-smoothed turbulence, `‖R‖` 2.8629e+02), real
        right-hand side `−R(state)`, reach **5** unless stated, `smoother_sweeps=4`,
        `coarse_eq_limit=2000`, plain aggregation, **operator and V-cycle at the SAME β — no PC-only
        floor**, restarted GMRES(15) to rtol 1e-6 on the **true** residual, 38-cycle cap. `38` below
        always means the cap, i.e. a failure.

        | arm | β = 0.5 | β = 0.05 | β = 0 |
        |---|---|---|---|
        | ILU(1) ×4 — control | **1** cyc, 1.8e-15 | **1**, 1.8e-11 | 38, 1.95e+01 |
        | ILU(0) ×4 — control | 38, 3.63e+00 | **1**, 2.3e-11 | 38, 9.99e-01 |
        | ILU(0) `shift_type nonzero` (1e-4, 1e-2) | 38, 3.63e+00 | 1 | 38, 9.99e-01 |
        | ILU(0) `shift_type inblocks` | 38, 3.63e+00 | 1 | 38, 9.99e-01 |
        | ILU(0) `shift_type positive_definite` | 38, **1.01e-05** | 38, 1.92e-01 | 38, 2.18e-01 |
        | ILU(0) ×1 / ×8 / ×16 | 9.9e+01 / 8.6e+00 / 1.00e+00 | 7 cyc / 2 cyc / 38, 4.09e+02 | 9.99e-01 / 1.37e+00 / 1.40e+00 |
        | ILU(0) order `rcm` | 38, 2.27e+05 | 3 cyc | 38, 5.40e+01 |
        | ILU(0) order `nd` | **19**, 8.3e-08 | 38, 3.8e-08 | 38, 1.36e+00 |
        | ILU(0) order `qmd` | **4**, 2.0e-12 | 38, 1.35e+01 | 38, 1.38e+00 |
        | ILU(0) order `rowlength` | **1**, 6.2e-14 | 8 cyc | 38, 1.52e+00 |
        | ILU(0) no equilibration | 38, 1.01e+00 | 1 | 38, 9.99e-01 |
        | ILU(0) reach 3 | **3**, 3.8e-11 | 38, 3.08e-01 | 38, 1.68e+00 |
        | ILU(1) reach 3 | 38, 4.79e-05 | 4 cyc | 38, 3.16e+00 |
        | block ILU(0) ×4 / ×8 (BAIJ level operator) | 1.56e+00 / 1.08e+00 | 1 / 1 | 2.24e+00 / 1.36e+00 |

        What that settles:
        - **The ORDERING is the one lever with a real effect, and it is a trade, not a fix.** At β = 0.5
          `rowlength` **ties ILU(1) exactly** (1 cycle) and `qmd` takes 4, from a control that diverges —
          the largest swing anywhere in the sweep, and consistent with the published ranking above putting
          permutation ahead of the shift. But the *same* orderings wreck β = 0.05, where the shipped
          cell-major order converges in one cycle and `qmd` **diverges** (1.35e+01) — so no single ordering
          is admissible for a march that crosses both. `rcm` is the worst arm in the whole sweep at
          β = 0.5 (2.27e+05), which matters because RCM is what the MC64 literature pairs its permutation
          with: **do not carry RCM over from that recipe untested.**
        - **The factorization shift is inert or blunt.** `nonzero` and `inblocks` are **bit-identical to
          the control at every β** — confirming from the outside that their triggers never fire here.
          `positive_definite` does reach: negative pivots 34 → **0**, and β = 0.5 improves 3.63 → 1.01e-05.
          It still does not converge, and it **degrades a healthy factorization** (β = 0.05, 1 cycle → cap
          at 1.92e-01), leaving every pivot at ≈ 1.9e-02 — the over-shift Chow & Saad warn about, observed.
        - **ILU(0) is AMPLIFYING here, not weak.** Sweeps are non-monotone at every β (β = 0.5:
          9.9e+01 → 3.63 → 8.56 → 1.00 for 1/4/8/16), and at β = 0.05 sixteen sweeps **diverge** a system
          four sweeps solve in one cycle. More smoother is the wrong direction.
        - **The equilibration is exonerated.** Removing it is neutral at β = 0.05 and no better elsewhere,
          with the same negative-pivot count — it is not what makes the zero-fill pivots small.
        - **Block ILU(0) does not rescue it either**, matching Chapman–Saad–Wigton above: better than point
          ILU(0) at β = 0.5 (1.56 vs 3.63) but still a failure, and *worse* at β = 0 (2.24 vs 1.00).
        - **⚠️ "pitzDaily's ILU(0) has negative pivots at every shift" is FALSE at this seed and the
          entry saying so should not be read as covering β = 0.05.** The counts are 34 / **4** / 6 at
          β = 0.5 / 0.05 / 0, and at β = 0.05 the zero-fill control **converges in one cycle** — at this
          state the zero-fill failure is a *large*-shift phenomenon, the opposite of `bfs3d`.
        - **⚠️ β = 0 at the COLD SEED discriminates nothing: every arm fails, ILU(1) included** (1.95e+01,
          the worst control in the table). That is a property of the unshifted Jacobian at a seed nowhere
          near the root, **not** of the adjoint's operator, which is taken at a converged state. Do not
          cite the β = 0 column as an adjoint result.
        - **Not run:** any `bfs3d` cross-check — the brief's precondition was a candidate that succeeds on
          pitzDaily, and none does; a developed pitzDaily state (only the self-start was probed, so every
          row above is seed-specific, and the developed-state measurement two bullets up already shows the
          pivots behave differently there); MC64's maximum-product **permutation**, which the sweep never
          touched (its accompanying plain two-sided scaling is separately demoted by CSW's Table IV — see
          the next bullet — but the permutation is a different mechanism and stands); the
          first-order-upwind probed operator Elman's bound points at; and — the highest-value gap, since
          every ordering swept here is saddle-blind — a **static-deferring** order that pushes the small
          diagonals (the pressure rows) to the end of the elimination. (The `condest` ranking **has** since
          been run, and is refuted — two bullets down.)
      - **✅ THE CSW FULL TEXT WAS READ (2026-08-16). Every number quoted from it above checks out. What
        it adds is an INSTRUMENT, not a remedy — and it independently corroborates two of this sweep's
        negative results.** Scope first, because the transfer is partial: their matrices are *compressible*
        N-S with an energy equation, so they share our **block size 5, our turbulence row, and our
        distance-1/distance-2 stencil question**, but they have **no incompressible saddle and no weak
        pressure block**. Nothing about the (p,p) block transfers from them.
        - **The instrument: `condest = ‖(LU)⁻¹e‖∞`**, `e` the vector of ones — one forward-and-back
          substitution, **no Krylov solve**. CSW score every factorization in the paper by it and decline
          to even attempt arms above 1e20. It measures the thing a pivot census structurally cannot: an
          incomplete factorization fails by its triangular *recurrences growing*, which happens with no
          small pivot anywhere. That is the exact blind spot this sweep hit — 34 negative pivots at
          β = 0.5 against 4 at β = 0.05, with the 4-pivot arm being the one that converges.
        - **⚠️ IT IS NECESSARY, NOT SUFFICIENT, AND CSW SHOW BOTH WAYS IT MISLEADS.** Their Table XVIII
          perturbs the diagonal blocks (SVD, smallest singular values lifted): `condest` falls in *every*
          test, and on the N-S-with-turbulence matrices convergence gets **worse** — BARTHT1A 94 steps →
          fails, BARTHT2A 545 → 449 → fails. Their banded-ILU experiment does the same thing (`condest`
          under 1e4, convergence stalls at ~600 steps). **So a low `condest` may never be read as a
          verdict**; it is a filter for the arms that cannot work, not a ranking of the ones that can.
        - **✅ Which independently corroborates our `positive_definite` arm.** Ours: pivots 34 → 0,
          β = 0.5 improves 3.63 → 1.01e-05 but still fails, and it *degrades* the healthy β = 0.05 arm
          from 1 cycle to a cap. That is Table XVIII's shape exactly, on the closest published matrix
          class. **Our result is the published behaviour of diagonal perturbation, not a PETSc artifact** —
          stop treating it as a suspicious local finding.
        - **✅ And corroborates the block-ILU refutation** already recorded above, from the same tables.
        - **⚠️ IT ALSO DEMOTES THE MC64 PLAN'S SCALING HALF — read the next bullet against this.** CSW's
          Table IV is the controlled test of row-then-column 2-norm scaling under ILU(0): it helps on
          three matrices, **hurts on two**, and tracks the change in *normality* `‖AᵀA − AAᵀ‖/‖AAᵀ‖`
          rather than anything about the pivots; they report it produced no significant change elsewhere
          and left the rest of the paper unscaled. Our own `no equilibration` arm agrees (neutral at
          β = 0.05, no better anywhere). **Plain two-sided scaling is not the lever here.** This says
          nothing about MC64's *permutation*, which is a different mechanism and remains untested.
        - **Their conclusion on dropping strategy, which bears on any threshold-ILU choice here:** for the
          BARTH matrices
          *"the threshold dropping method is not suitable … it is beneficial to keep all entries in the L
          and U factors that correspond to the ILU(0) pattern"*, and level-of-fill beats threshold by more
          as the entries per row grow. Recorded as a caution, **not** transferred: our own incomplete-LU
          results are on a different matrix and are not contradicted by this.
        - **The question this motivated — and it is now ANSWERED, negatively; see the next bullet.** The
          sweep's own finding is that **the winning configuration moves with β and no static choice spans
          a march**, so a per-matrix *selector* is the only shape a fix can take, and `condest` was the
          first candidate instrument for one costing no solve. The falsifiable form was: does
          `argmin condest` pick the arm that converges, at each β? **It does not**, and neither does any
          threshold or window on it.
        - **❌❌ MEASURED, AND `condest` CARRIES NO USABLE SIGNAL ON THIS PROBLEM (2026-08-16). Do not
          build a selector on it, and do not re-run this.** Every arm above was rebuilt with
          `ILU0_SWEEP_CONDEST_ONLY=1` and its `condest` joined against the cycle counts already in the
          table — same harness, same self-start state (`‖R‖` 2.8629e+02), same reach 5, same
          `smoother_sweeps=4`, `coarse_eq_limit=2000`, plain aggregation, operator and V-cycle at the same
          β. The factorizations reproduce **bit-for-bit** (every pivot census identical to the earlier
          run), so the join is legitimate.

          | arm | β=0.5 condest | β=0.5 | β=0.05 condest | β=0.05 |
          |---|---|---|---|---|
          | ILU(1) control | 5.92e+03 | **1** | 2.72e+07 | **1** |
          | ILU(0) control | 1.14e+08 | fail 3.63 | 2.43e+07 | **1** |
          | ILU(0) ×16 | 1.14e+08 | fail | **2.43e+07** | **diverges 4.09e+02** |
          | shift posdef | 1.72e-01 | fail | 1.46e-01 | fail |
          | order rcm | 2.13e+05 | **diverges 2.3e+05** | 5.21e+07 | **3** |
          | order nd | 2.99e+03 | 19 | 3.07e+04 | fail |
          | order qmd | 2.61e+03 | 4 | 1.06e+05 | fail |
          | order rowlength | 4.21e+03 | **1** | 4.10e+04 | 8 |
          | no equilibration | 4.09e+10 | fail | **1.15e+08** | **1** |
          | ILU(0) reach 3 | 6.07e+04 | 3 | 2.42e+07 | fail |

          **Four independent ways it fails, any one of which is disqualifying:**
          1. **Same `condest`, opposite outcome, same β.** At β=0.05 the ILU(0) control and ×16 share
             **one factorization** (2.43e+07) and land on 1 cycle and *divergence*. `condest` scores the
             factor and is blind by construction to how the smoother is applied — and to the coarse space,
             which the reach arms move.
          2. **The threshold is not stable across β.** `no equilibration` converges in 1 cycle at
             **1.15e+08** (β=0.05); the ILU(0) control fails at **1.14e+08** (β=0.5) — the same value to
             two figures, opposite outcomes, same case and state.
          3. **The sign of the correlation reverses between blocks.** At β=0.5 the *lowest*-condest ILU(0)
             orderings converge (qmd 2.61e+03 → 4) and the highest fail; at β=0.05 the lowest fail
             (nd 3.07e+04) and the highest converges (rcm 5.21e+07 → 3). `rcm` is the same arm at
             **2.13e+05 diverging** and **5.21e+07 converging**.
          4. **No window exists at β=0.05, two-sided or otherwise.** At β=0.5 a band [2.6e+03, 5.9e+03]
             does separate the like-for-like arms perfectly, which is why this looked promising mid-run;
             at β=0.05 failing arms at 3.07e+04 and 1.06e+05 **bracket** a converging arm at 4.10e+04.
             ⚠️ Do not cite the β=0.5 band on its own — it is real and it is an artifact of one shift.
        - **✅ The ONE robust reading, and it is worth keeping: `condest ≪ 1` means the shift has
          over-damped the factor into uselessness.** `positive_definite` measures **1.72e-01 / 1.46e-01 /
          1.42e-01** at β = 0.5 / 0.05 / 0 and **fails at all three**, with min pivot = median pivot
          (≈1.9e-02), i.e. the whole diagonal flattened to one constant. `LU` is then large, `(LU)⁻¹` is
          small and perfectly stable, and it approximates nothing — the preconditioned system gets tiny
          eigenvalues and stalls. This is CSW's Table XVIII effect reproduced on our matrix, and it makes
          their framing precise: they require `LU` to be an **accurate** representation of `A` *and*
          `(LU)⁻¹` to be well conditioned. `condest` tests only the second. **A one-sided
          over-stabilization check is all it is good for here.**
        - **Harness note:** `getFactorMatrix` raises PETSc error 56 on a Python PC, so the two
          `block ILU(0)` arms report no census. That is the instrument not reaching inside a custom PC,
          **not** a result about those arms.
      - **✅✅ HILUCSI (Chen, Ghai & Jiao, arXiv:1911.10139, 2019) EXPLAINS THE `condest` FAILURE ABOVE,
        AND SUPPLIES THE ONE ORDERING FAMILY THIS SWEEP NEVER TESTED. Read in full 2026-08-16.** A
        multilevel incomplete-LDU with mixed symmetric/unsymmetric processing, benchmarked on 2D and 3D
        Navier–Stokes and on symmetric saddle-point systems from Stokes and mixed Poisson.
        - **⚠️ Why our `condest` measurement came back useless, stated as a published mechanism.** Their
          §3.2 derives `ρ(AM⁻¹ − I) ≤ ‖M⁻¹‖·‖δ_A‖`, so what must be bounded is `‖M⁻¹‖`, achieved by
          bounding the norms of **`D⁻¹`, `L⁻¹` AND `U⁻¹`** — and they state flatly that bounding `‖L⁻¹‖`
          and `‖U⁻¹‖` **alone is insufficient**, attributing HILUCSI's robustness advantage over ILUPACK
          to exactly that difference. Our `condest = ‖(LU)⁻¹e‖∞` is precisely the insufficient quantity:
          it omits the diagonal. **So the negative result above is the predicted one**, and the repair
          (score `κ(D)` alongside it — the census already reports min and median pivot) is cheap and
          untested.
        - **⚠️ SCOPE — our refutation is of `condest` as an ARM SELECTOR, not of `condest` as a dropping
          threshold.** ILUPACK and HILUCSI use it *inside* the factorization to decide what to drop or
          defer (ILUPACK's default `κ = 5`); we scored finished factorizations with it and ranked them.
          Those are different uses and our measurement says nothing about the second. Do not cite the
          refutation against inverse-based dropping.
        - **⚠️ Their Definition 1 names the criterion our architecture actually needs, with a caveat that
          must travel with it.** `M` is accurate if `ρ(AM⁻¹ − I) ≤ ρ₀`; and *"it is unnecessary for `ρ₀`
          to be less than 1 for the convergence of a KSP method. However … if `ρ₀ < 1`, then `M` can be
          used in place of a stationary iterative method as a smoother in multigrid methods."* Our ILU is
          a **stationary Richardson smoother inside a V-cycle**, so it is held to the stronger standard,
          which is a candidate explanation for the sweep's "ILU(0) is amplifying, not weak" (×16 diverges
          a system ×4 solves in one cycle). **⚠️ Do NOT go measure `ρ` naively:** a multigrid smoother is
          *not* required to contract the smooth modes — that is the coarse space's job — and this project
          has already retracted one conclusion built on a smoother spectral radius that turned out to be
          the largest, smoothest eigenmode. The criterion is suggestive here, not directly applicable.
        - **✅ THE ACTIONABLE GAP: static deferring is a SADDLE-AWARE SYMMETRIC permutation, and every
          ordering this sweep tested was saddle-blind.** RCM, ND, QMD and rowlength are all
          bandwidth- or fill-reducing and know nothing about which rows are the pressure rows. HILUCSI
          instead *"symmetrically permutes the zero and tiny diagonal entries to the lower-right corner,
          to be factorized in the next level"*, reporting it *"improves robustness and efficiency …
          especially for (nearly) symmetric saddle-point problems"* and removes the need for 2×2 pivots,
          because *"zero or tiny pivots tend to lead to fast growth of `‖L⁻¹‖∞` and `‖U⁻¹‖₁`"*. It being
          **symmetric** is what makes it admissible here, where an unsymmetric row permutation is not.
          Our Rhie–Chow (p,p) block is nonzero so we have no *zero* diagonals, but we do have small ones
          and the probe already localizes the bad pivots to the **pressure rows**. **This is the cheapest
          untested high-value arm on the list** and it fits the existing `assemble()` permutation seam.
        - **✅ It also revives MC64 in a form our recorded objection does not cover.** The objection above
          is that an unsymmetric row permutation breaks the cell-block structure `setBlockSize(n_fields)`
          needs. HILUCSI §2.4 uses the **symmetrized** HSL_MC64 (`P_r = P_c`, `D̃_r = D̃_c = √(D_r D_c)`),
          which is a symmetric permutation. Re-examine the objection against that form; do not carry it
          over unexamined.
        - **⚠️ AND A DIRECT N-S DATA POINT AGAINST TUNING `condest`:** on PR02R, a 2D Navier–Stokes
          matrix, SuperLU's ILUTP fails and ILUPACK fails **"regardless of how we tuned condest"**, while
          HILUCSI succeeds. Independent of our own sweep, on our own equation set.
        - **❌ What it does NOT do is rescue ILU(0).** HILUCSI is a *multilevel* ILU whose levels come
          from deferring; it replaces zero-fill single-level ILU rather than repairing it. Adopting it
          wholesale is a different architecture from the GAMG V-cycle that ships (and their own listed
          limitation is that it is serial and not block-aware for vector-valued PDEs). Take the ordering
          idea and the `D⁻¹` correction; do not read this as a drop-in.
      - **✅ THE STRONGEST PUBLISHED PRECEDENT FOR THIS EXACT FAILURE MODE IS MC64 + SCALING.** Benzi
        (*J. Comput. Phys.* 182, 2002, §3.3) on **LNS3937, a linearized Navier–Stokes matrix**: *"Because
        of zero pivots, ILUT cannot solve any of these problems. Even for large amounts of fill, ILUT with
        partial pivoting (ILUTP) still cannot solve LNS3937 … After preprocessing with the maximum
        diagonal product permutation and corresponding scalings, all these problems can be easily
        solved by ILUT"* — 24 iterations, with RCM as the accompanying symmetric permutation. Duff & Koster
        (*SIMAX* 20(4) 1999; 22(4) 2001). ⚠️ For us an unsymmetric row permutation **breaks the cell-block
        structure GAMG's `setBlockSize(n_fields)` aggregation depends on**, so it would have to be applied
        to the smoother alone. ⚠️ The symmetrized HSL_MC64 form (`P_r = P_c`) that HILUCSI uses is a
        **symmetric** permutation, so re-examine that objection against it rather than carrying it over.
      - **✅✅ THE MC64 PRE-CHECK HAS BEEN RUN, AND ITS OWN PREMISE WAS WRONG: THE WEAK ROWS ARE `omega`
        AND `u`, NOT `p` (2026-08-16).** This entry previously said *"if the pressure rows already carry
        their maximum on the diagonal after equilibration, MC64 has nothing to fix"*. The pressure rows
        **do** carry it — and the gate passes anyway, on the transport rows. Harness:
        `validation/pitzdaily_openfoam/mc64_precheck.py`, no solve. Configuration: pitzDaily, 12225 cells,
        5 fields, the case's own self-start seed (`‖R‖` 2.8629e+02), reach 5, the matrix **as the smoother
        factorizes it** (shift, symmetric square-root-diagonal equilibration, cell-major reorder). The
        ratio is (largest off-diagonal magnitude)/|diagonal| per row; equilibration puts every diagonal at
        magnitude 1, so it reads as `|a_ij| / sqrt(d_ii·d_jj)` — a scale-free coupling strength. Percent
        is the share of that field's 12225 rows whose diagonal is **beaten**:

          | β | u | v | p | k | omega |
          |---|---|---|---|---|---|
          | 0.5 | 60.9 % (med 1.09) | 1.4 % | **0.0 %** (med 0.50) | 0.0 % | **100 %** (med 4.71, max 341) |
          | 0.05 | 93.4 % (med 1.43) | 27.9 % | **0.0 %** (med 0.50) | 0.6 % | **100 %** (med 5.65, max 406) |
          | 0 | 95.2 % (med 1.48) | 39.1 % | **0.0 %** (med 0.50) | 1.8 % | **100 %** (med 5.79, max 415) |

        - **The pressure rows are diagonally dominant at every shift** — 0, 1 and 4 rows of 12225 beaten,
          median ratio 0.50 throughout. **This matrix is not weak at the saddle.** A large amount of
          effort recorded in this file is aimed at the (p,p) block and the Schur complement; on this case
          that is not where the zero-fill elimination is in trouble.
        - **`omega` is the weakest field by a wide margin** — *every* row at *every* shift, median
          coupling ~5× its own diagonal and a tail past 400×. This is a **third independent leg** for the
          recorded "ω is the problem field", the other two being the per-field V-cycle smoothing
          measurement and the near-null-direction analysis of the worst per-cell blocks. Three different
          quantities, same field.
        - **The u/v asymmetry is physical, which is what makes the measurement credible:** 60.9 % against
          1.4 % at β = 0.5 in a predominantly streamwise flow is convection beating diffusion in the
          streamwise coupling — a high cell Péclet number, exactly the regime Elman (1986) identifies as
          where an incomplete factorization goes unstable, and where first-order upwinding restores it.
        - **⚠️⚠️ BUT DIAGONAL DOMINANCE DOES NOT EXPLAIN THE β-DEPENDENT FAILURE, AND MUST NOT BE CITED AS
          IF IT DID.** Dominance degrades **monotonically** as the shift falls (u: 60.9 → 93.4 → 95.2 %),
          while ILU(0)'s outcome is **non-monotone** (β = 0.5 fails, β = 0.05 converges in one cycle,
          β = 0 fails). The *least* dominant of the two solvable shifts is the one that converges. So this
          probe localizes **where** the matrix is weak; it is the third quantity in a row — after the pivot
          census and `condest` — that fails to track the outcome. **The β-dependence remains unexplained.**
        - **Consequence for MC64:** it has genuine work to do (~7.4k–11.6k `u` rows and all 12.2k `omega`
          rows), so it is no longer speculative — but it would act mostly on **`omega`**, and since
          dominance does not predict convergence here, a dominance-based argument is not a prediction that
          it will help. Try it as an experiment, not as a fix.
        - **And it promotes a cheaper, better-targeted candidate that keeps fill at zero:** build the
          preconditioner's operator with **first-order upwind advection** while the Krylov operator stays
          the exact second-order Jacobian. That is what Elman's bound prescribes for these rows, it is the
          same preconditioner-only substitution seam the reach narrowing already uses, and this codebase
          **already upwinds first-order in the frozen AMG operator for exactly this reason**. Untested on
          the coupled smoother.
      - **⚠️ ORDERING IS DEMOTED to a rate lever, and Duff & Meurant explains our OWN measurement.** No
        paper was found in which a reordering converts a *breaking* ILU into a non-breaking one; where
        breakdown is provably impossible (M-matrices) it holds for **every** symmetric permutation. The
        RCM recommendation (Benzi, Szyld & van Duin, *SISC* 20(5), 1999) is explicitly scoped to *"if some
        amount of fill-in is allowed"*. **At zero fill the classic result runs the other way**: Duff &
        Meurant (*BIT* 29, 1989) found fill-reducing orderings *significantly worse* than natural for
        no-fill IC, because *"the average size of the fill-ins is much larger, so that the norm of the
        remainder matrix `R = A − L̄L̄ᵀ` ends up being larger"* — which is exactly the shape of this repo's
        own finding that pitzDaily's ILU(0) converges under the mesh's own ordering and **caps under RCM,
        streamwise and random**. Streamline/downwind ordering additionally does not transfer: it needs an
        acyclic digraph, and recirculation is the whole point of this case. **Corollary: report
        `‖A − LU‖_F` alongside the pivots — Duff & Meurant say that, not the pivots, is what tracks the
        iteration count at zero fill.**
      - **❌ MILU / Gustafsson is CLOSED for this operator — do not spend a run on it.** The `O(h⁻¹)`
        condition-number result is proven only for the **perturbed** method (`Ae + τh²e = LUe`, τ > 0;
        Dupont–Kendall–Rachford 1968, Gustafsson 1978, unperturbed case later by Beauwens) on **symmetric
        M-matrix elliptic** operators, and Elman's table above shows it is *worse than ILU* in the
        opposite-signed convection regime pitzDaily's recirculation occupies. Structurally it is wrong here
        twice over: the discrete gradient/divergence rows **already have zero row sums**, so compensation on
        them adds nothing (Wubs & Thies), and unperturbed MILU is singular exactly when `A` is (Notay 1992),
        so a closed-domain constant-pressure null mode is transferred into the preconditioner by
        construction. The revealed preference agrees — Ifpack2 `RILUK` relax value defaults to 0, MATLAB's
        `milu` defaults off, hypre has no such option, and **PETSc has no MILU parameter at all**. A grep of
        Benzi–Golub–Liesen's 137-page saddle-point survey finds **zero** occurrences of "MILU", "modified
        incomplete", "Gustafsson" or "Dupont".
      - **What it IS, as far as measured: zero fill is an insufficient APPROXIMATION of pitzDaily's
        velocity–pressure saddle, at healthy pivots** — a smoothing-quality deficit, not a breakdown.
        Supporting: the fill ladder's whole jump is 0→1 (ILU(0) caps, ILU(1) 52, ILU(2) 25, ILU(3) 21 at
        the developed state, β = 0.05); the unresolved ILU(0) residual sits in `v` 3.7e-3, `p` 2.2e-3,
        `u` 1.0e-3 against `k` 1.0e-4, `ω` 2.0e-4; the **`[u,v,p]` slice alone reproduces the entire
        gap** (ILU(0) cap / ILU(1) 52) while `[k,ω]` alone is healthy (16 / 11), so **a field split does
        not rescue it**; and the failure is **delocalized** — the worst 200 rows match the mesh median on
        skewness (1.24e-4 vs 3.15e-4, i.e. *lower* than typical), wall adjacency (0 vs 0) and aspect
        ratio (1.65 vs 1.66). Two things make it fatal rather than merely expensive: ILU(0) converges
        **only under the mesh's own cell ordering** (208 its; RCM, streamwise and random all cap, while
        ILU(1) converges under all four), so it is riding on an accident; and **the multigrid cannot
        repair it** — on a clean orthogonal channel GAMG makes the smoother choice irrelevant and
        size-independent (ILU(0)×4 3/5/6 against ILU(1)×4 3/8/11 at 384/3456/13824 cells), where on
        pitzDaily the same two arms are **87 against 19**.
      - **⚠️ WHY `bfs3d` IS THE OPPOSITE IS STILL UNEXPLAINED — and the leading hypothesis is REFUTED by
        its own measurement.** Cell anisotropy looked decisive in a controlled sweep (grading a
        zero-skew channel under a fixed poor ordering: ILU(0) 36 → 43 → 178 → 329 its as median aspect
        ratio goes 2.7 → 10.4, while ILU(1) holds 18 → 16). But measured on the two shipped meshes
        (geometry only), the per-cell conductance ratio is median **1.72** on pitzDaily and **8.59** on
        `bfs3d` — **`bfs3d` is 5× the more anisotropic mesh and is the case where zero fill works**, so
        anisotropy points the wrong way. Two candidates remain undistinguished: the 3D distance-3
        stencil's density (280 nnz/row against pitzDaily's 122, so ILU(1) fill there is far denser and
        its 303 negative pivots may be a fill-explosion with no 2D analogue), and Reynolds number, which
        was not swept. **The experiment that separates them: this same single-state probe — PETSc ILU(0)
        vs ILU(1), unpreconditioned norm, true residual, equilibrated cell-major operator at matched β —
        on a `bfs3d` checkpoint, reading the residual per field block and the factor nnz.** Not yet run
        (the `bfs3d` case build is ~7 GB and the machine was shared).
      - **❌ "Precondition the gradient solve and use GMRES instead" is REFUTED as a reach lever, do not
        re-propose it.** With implicit differentiation the tangent is `A_g⁻¹B`, i.e. the sweep series run
        to round-off — measured reach **9** at 25 % skew and **11** at 40 %, against 5 for `sweeps=4`. It
        moves reach the wrong way, and it does not touch the reason GMRES is not the default (the nested
        implicit-diff tangent re-entered by every jvp, ≈180× per coupled-residual eval on pitzDaily).

      **✅ ROOT CAUSE AND FIX (2026-08-12) — it was NOT the aliasing, and it was NOT the reach. Sparse
      arithmetic was pruning the explicit zeros out of the smoother's pattern.** Two readings of #191
      preceded this and both are refuted; the falsifying harnesses are in `validation/bfs3d_openfoam/`.
      - **Aliasing is real and pervasive but is not the cause.** A reach-2 colouring is collision-free
        only on its own pattern, so under the reach-3 assembly pattern two cells sharing a colour can
        both couple to one row and the response is charged entirely to the near one:
        `column_reach_collisions.py` (mesh graph only, no state, no solve) finds **287 315 of 537 896
        entries — 53.4 % — corrupted in every shortened column**, touching 23 025 of 23 040 rows.
        Yet `trailing_column_reach_probe.py`, which measures the assembled error *exactly* (seed the
        colour class, minus seed the one cell), puts the pressure column's error at **1.4e-17 … 4.0e-16
        at the rung ICs — the float64 floor, and the divergence is at step 1, so no other state is
        involved.** A perturbation that size cannot take a solve from 3 cycles to 44.
      - **The cause: `a + sp.diags(shift)` and `diags(s) @ a @ diags(s)` are sparse arithmetic, which
        stores only entries whose result is nonzero.** `column_reach` writes out-of-reach entries as
        **exact `0.0`**; a uniform build stores the true value there (~1e-26, nonzero). Exact zeros were
        stripped, 1e-26 survived. Measured (`column_reach_zero_pruning.py`): same 47 209 392 stored
        pattern in both arms, but **38 721 674 against 32 309 840 after assembly — 6 411 834 positions,
        16.6 %, removed from what the ILU(0) smoother factorizes.** A zero-fill factorization takes its
        pattern from exactly that.
      - **Why `p` and not `k`/`ω`:** pressure is in the incomplete-LU-smoothed **leading** block; k and ω
        are in the trailing block, whose cell-block inverse has no factorization pattern to weaken. That
        also retro-explains the padding experiment which established the reach-3 requirement — it padded
        with **~1e-30, not exact zero**, precisely what survives sparse arithmetic.
      - **⚠️ CURED BY PATTERN-PRESERVING ASSEMBLY, AND THAT CURE IS NOT AVAILABLE — so `p` STAYS at reach
        3.** Preserving the pattern (`frozen_operator.apply_symmetric_scale`, plus a diagonal assignment
        for the shift) does exactly what this entry says: `(3,3,3,2,2,2)` then reproduces the uniform-reach
        baseline in **every reported digit** at rung-1 step 1 — `‖R₀‖` 3.2901e-01, β 0.5000, 4 inner,
        **3 cycles**, `‖R‖` 2.046e-01, α 1.000, no flags — and the **full march** converges to the recorded
        root: 67 steps, 320 cycles, 4 escalations, final ‖R‖ **3.586e-06**, mid-span `x_r/h` **8.3611**,
        ν_t peak **150.1071**, with the trailing inverse's `equilibrate` setting inert (both settings
        step-for-step identical). ⚠️ That run's **wall clock is void** (its early steps ran against the
        test tiers); its step and cycle counts are not.
        **But what it preserves are stored EXACT ZEROS, and the zero-fill incomplete factorization cannot
        take those** — measured at zero shift, carrying them costs 58 restart cycles at a true relative
        residual of 2.299e-02 against 11 cycles to 8.474e-11 without (see *"stored exactly-zero positions
        break the ZERO-SHIFT V-cycle"*). They are therefore pruned at the factorization boundary
        (`AmgVCycle._live`), which puts `p` at reach 2 back in the configuration above. **The shipped
        default is `(3,3,3,3,2,2)`** — 454 probes against 564, −16 % — and `p` at reach 3 IS a correctness
        constraint after all. The march numbers here stand as a measurement of the preserving arm; they are
        not a licence for the shortened default under the shipped one.
      - ⚠️ **How it got through, which is the transferable part: the audit that licensed it AND the gate
        that verified it both collapse over row fields**, on a matrix whose row norms differ by ~8 orders
        (k row 8.8e-06, ω row 1.4e+03). `column_reach_ladder` takes a whole column block;
        `jacobian_relative_error` is one random vector over the whole matrix. Neither can see a wrong
        pressure block, and `probe_reach_audit.py` already had the right function — `block_shell_fractions`,
        per (row field, column field) pair. **Read a reach or exactness measurement per PAIR, never over a
        whole column or a whole matrix.**

      **✅ MEASURED, and CPU time tracks the probe count 1:1** (`state-00077`, at the then-default
      `_PROBE_BATCH_SIZE = 4` — since moved to 8, so the RATIO below stands and the absolute seconds
      are ~10 % lower now,
      four interleaved repetitions after a discarded warm-up, minimum reported):

      | | uniform reach 3 | per-column | |
      |---|---|---|---|
      | probes | 564 | 399 | −29% |
      | **process CPU** | 65.74 s | 46.74 s | **−29%** |
      | wall | 10.36 s | 7.11 s | −31% |

      Per-arm spread was ≤5% and the arms do not overlap (worst per-column 7.50 s beats best uniform
      10.36 s). **The CPU row is the load-bearing one:** it is nearly insensitive to other processes
      competing for cores, and it moves exactly with the probe count, which is what establishes that the
      saving is work removed rather than a scheduling artifact.
      **⚠️ Against the ORIGINAL shipped probe (672, degree-ordered colouring at uniform reach 3) the
      combined saving is 672 → 399 = −41%, but that arm was NOT timed in this run** — it is a probe-count
      ratio extrapolated on the 1:1 CPU-vs-probes relation above. Time it before quoting a second.
      **⚠️ THREE EARLIER ATTEMPTS AT THIS NUMBER WERE EACH CONFOUNDED, in three different ways, and the
      method is the finding.** (1) The batched jvp compiles on its first call, so whichever arm ran first
      carried the trace and the second looked cheap — 12.4 s vs 6.3 s, which is the compile, not the
      probe. (2) A concurrent 4.4 GB march drove the machine 8 GB into swap and the result came out
      *inverted*, 10.4 s vs 11.8 s. (3) The machine is shared and rarely idle. The method that works:
      discard a warm-up, interleave the arms so a load excursion hits both, repeat, report the minimum,
      and carry CPU time beside wall as the cross-check.
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
      preconditioner passes (the complete LU keeps the plain loop, which any NumPy matvec supports). **(1) Batched
      probing** — `batched_matvec` (a `jax.vmap` of the jvp, **built once and reused** so it compiles a
      single time; `probe_batch_size` chunks it for memory) runs the coloured probes as a few fused passes
      instead of a Python loop of separate calls. Measured 22.4→14.0 s (~1.6×) on `bfs3d` — modest because
      CPU forward-AD does not vectorize across the batch like a GPU (the win is dispatch amortization). For
      the SAME reason the chunk was once kept **small** (`_PROBE_BATCH_SIZE` was 4, not 16): a larger
      batch holds more simultaneous forward-AD tapes. **The default is now 8** — see the sweep below,
      which is what retired that reasoning.
      **✅ RE-MEASURED, and the memory half of that trade NO LONGER EXISTS (2026-08-11,
      `validation/bfs3d_openfoam/probe_batch_sweep.py`; `bfs3d` `state-00077`, 399 probes at the
      per-column reach, 47.2M nnz, warm-up discarded per chunk shape, min of 3):**

      | batch | wall (s) | cpu (s) | python peak | vs batch 1 |
      |---|---|---|---|---|
      | 1 | 11.73 | 91.37 | 380 MB | 1.00× |
      | 2 | 8.42 | 70.29 | 383 MB | 1.39× |
      | **4 (shipped)** | 6.69 | 53.89 | 388 MB | 1.75× |
      | **8** | 5.94 | **46.35** | 399 MB | 1.98× |
      | 16 | **5.81** | 49.13 | 419 MB | 2.02× |
      | 32 | 6.47 | 52.33 | 460 MB | 1.81× |

      - **Memory barely moves: 380 → 460 MB across batch 1 → 32**, about 2.5 MB per unit of batch. The
        old "16 vs 4: ~2.2 GB vs ~0.7 GB" was the seed set and the response array, and **neither ever
        scaled with the batch**; both are gone. So the memory argument for a small chunk is void.
      - **The curve has an interior optimum and TURNS: 32 is worse than 16** (6.47 s against 5.81 s), so
        "bigger is better up to memory" was never the shape. CPU time — the cleaner axis on a shared
        machine — bottoms at **8**.
      - **The default is now `_PROBE_BATCH_SIZE = 8`** (was 4), which is where processor time bottoms.
        16 is a hair faster in wall and slower in CPU; on a shared machine CPU is the more trustworthy
        axis, and the wall difference between 8 and 16 is 0.13 s on a 6 s probe.
      **⚠️ AUTO-SIZING THE BATCH TO AVAILABLE MEMORY IS NOT WORTH BUILDING, and this is why.** The
      motivation was that the right chunk depends on machine state and case size. At 2.5 MB per unit it
      does not: any sane fixed value is safe, and the constraint that motivated the mechanism was an
      artifact of allocations that have since been removed. Fix the allocation, not the knob.
      **⚠️ WHERE THE MEMORY ACTUALLY GOES, measured by attribution rather than assumed:** peak RSS is
      149 MB after imports, **6889 MB after `build_case`**, 7154 MB after the colouring and plan, 8359 MB
      after the gather map. So the case assembly is ~6.7 GB, the gather map ~1.2 GB, and a whole
      materialize 0.4 GB. **A run's memory ceiling is set by building the case, not by probing it** — the
      probe is now a rounding error against it, and an initial reading that blamed the structure build was
      wrong.
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
      operator reaches an incomplete factorization, and `AmgVCycle._live` is where that now happens** — see
      *"stored exactly-zero positions break the ZERO-SHIFT V-cycle"* below, which measures what they cost and
      corrects the two mechanisms this bullet used to assert for it. De-compression 22.4→11.2 s (2.0× vs
      loop); full build (materialize + GAMG) only 56.0→54.2 s (**1.03×** — GAMG-dominated), so the gather's
      real value is the fixed-structure invariant, not the wall-clock.
    - **⚠️⚠️ STORED EXACTLY-ZERO POSITIONS BREAK THE ZERO-SHIFT V-CYCLE, and BOTH obvious mechanisms are
      REFUTED (measured 2026-08-12, harness `validation/bfs3d_openfoam/zero_pattern_pivots.py`).** The
      coloured probe stores the full block-stencil pattern, of which **8.03M of 47.21M positions are exactly
      zero**; whether they survive into the preconditioner depends only on how the shift and the
      equilibration are *spelled*. A sparse product or addition stores only nonzero results and deletes them;
      an in-place diagonal assignment and value scale cannot, and keeps them.

      *Configuration:* `bfs3d` `state-00067` (converged, ‖R‖ 3.586e-06), **operator and preconditioner both
      at β = 0**, right-hand side the real steady residual `−R(state)`, monolithic V-cycle, plain
      aggregation, ILU(0) ×4, `coarse_eq_limit` 2000, pattern reach 3, GMRES restart 15 to rtol 1e-8 judged
      on the **TRUE** residual.

      | column reach | shift / equilibration | nnz | stored zeros | cycles | TRUE rel |
      |---|---|---|---|---|---|
      | uniform 3 | prune / prune | 39.18M | 0 | **11** | **8.474e-11** |
      | uniform 3 | preserve / preserve | 47.21M | 8.03M | 58 | **2.299e-02** |
      | uniform 3 | **preserve / prune** | 39.18M | 0 | **11** | **8.474e-11** |
      | 3/3/3/3/2/2 | prune / prune | 36.97M | 0 | 22 | 8.545e-11 |
      | 3/3/3/3/2/2 | preserve / preserve | 47.21M | 10.24M | 58 | 2.299e-02 |

      - **⚠️ AT THE FORWARD MARCH'S OWN OPERATING POINT EVERY ARM TIES, so a march cannot reveal any of
        this.** Same harness at `state-00066` (step-initial, operator β 0.0096 with the V-cycle at the 0.05
        floor — the shipped mismatch): **all five arms give 4 restart cycles at a true relative residual of
        1.435e-13** (1.444e-13 at the shortened reach). Not a cycle of difference, from either the stored
        zeros or the reach. So the damage is **confined to the zero-shift adjoint**, and the pattern-preserving
        change's clean march revalidation was correct rather than lucky — its validation and its failure mode
        genuinely do not overlap. This is the "a benign operating point cannot discriminate" rule biting on
        exactly the axis that was being changed: β = 0 is the only state on this case that separates these arms.
      - **Pruning at EITHER stage restores the good operator bit-identically** (11 cycles, 8.474e-11), so the
        fix is not "which stage prunes" but "prune before the factorization". Shipped as
        **`AmgVCycle._live`**, at the boundary where the operator reaches PETSc, so the assemblers can stay
        pattern-preserving (which is what makes the fixed-pattern fast path and the generic sparse path agree
        entry for entry) while the factorization still gets the live pattern.
      - **⚠️ A PIVOT CENSUS CANNOT DETECT THIS — all four arms are IDENTICAL: min |pivot| 1.546e-01, zero
        negative pivots, median 1.020.** So is the coarse space's size (2 levels, 1296 coarse equations in
        every arm). Both were the natural diagnoses, both were measured, and both are blind here; do not
        spend a run on either again. **The blindness holds at BOTH states, i.e. over nine arms:** at
        `state-00066` every arm reads min |pivot| 2.145e-01 and median 1.017 — the shift lifts the smallest
        pivot from 1.546e-01, which is the shift doing its job on the diagonal — while the arms there tie in
        convergence and at β = 0 they differ five-fold. A census that is constant across arms that converge
        identically *and* across arms that do not is not measuring the thing that separates them. What *is* demonstrated (small-scale, PETSc directly) is that the stored
        zeros are genuinely kept by PETSc (`nz_used` counts them, in the `Mat` and in the ILU factor) and
        that the smoother's **action** differs, i.e. the elimination does deposit fill into them.
      - **⚠️ Why a denser incomplete factor is a WORSE smoother on this operator at zero shift is NOT
        established.** Retaining fill can only make an incomplete factorization a closer approximation to
        `A⁻¹`, and the pivots stay healthy, so the ILU(1)-style "fill produces negative pivots" account does
        **not** transfer. The open instrument is to compare the two V-cycles' action per level, or to degrade
        the coarse solve to `jacobi` in both arms and see whether the gap survives.
      - **Column shortening is a separate and much milder effect: it DEGRADES but does not break.** At the
        tight stop `3/3/3/3/2/2` converges in 22 cycles against uniform's 11 — 2× the cost, same 1e-11 floor
        — where preserving does not converge at all. A loose march-solver stop reads its 4.779e-03 as a
        failure, but that is the stop, not divergence. Under preservation the reach is genuinely irrelevant
        (58 cycles / 2.299e-02 at both reaches), which is the one claim of the pattern-preserving change that
        the measurement fully confirms.
      - **The reach-3-positions padding experiment is consistent with all of this, and the distinction is now
        load-bearing:** that experiment padded with **~1e-30**, which is *nonzero* and therefore survives a
        pruning assembly. An exact zero does not. Do not read "explicit zeros are required in the pattern" as
        licensing stored exact zeros.
    - **Probe REACH is a preconditioner choice — reach-2 is NOT a safe drop-in (measured, SHELVED — a
      GENUINE failure, not a build artifact).** The materialized `J` is only the preconditioner's operator
      (the solve matvec is always the exact matrix-free jvp), so a lower `stencil_reach` gives an approximate
      PC. On the orthogonal `bfs3d` mesh reach-2 is numerically near-*exact* at *every* state
      (‖A2−A3‖_F/‖A3‖ = 6e-6 cold, ~1e-15 developed — the dropped distance-3 shell is negligible; swapping
      Corrected→Compact Green-Gauss leaves it bit-identical, so the non-orthogonal skew correction
      contributes ~0 here) and ~2.2x cheaper to probe (60 vs 112 colours under the degree-ordered colouring this was measured with; 39 vs 94 under the saturation colouring now shipped, ~half the nnz). **Yet GAMG(reach-2)
      DIVERGES as a preconditioner** (true residual 1e3–1e8) at cold AND developed, on its own operator, with
      a verified-correct build — so it is genuine, not a build/scaling bug. **The cause is the ILU(1)
      smoother, whose fill is PATTERN-dependent (a symbolic/graph operation), not value-dependent:** halving
      the graph gives a structurally weaker incomplete factorization that is non-convergent on the indefinite
      saddle. The ≤6e-6 magnitude argument bounds only the value-based *aggregation* (consistent
      reach-2↔reach-3); it does not touch the smoother. Proof: padding reach-2's values onto the reach-3
      *positions* (distance-3 shell as ~**1e-30**, which is *nonzero*) recovers reach-3 convergence
      **bit-identically** (32 matvecs) — so the distance-3 *positions* are causal, their values are not.
      **This is why the reach-3 pattern is REQUIRED for smoother convergence (not merely a
      structure-invariance nicety), and why you must not lower the default reach.** ⚠️ **Read "positions
      carrying a tiny NONZERO", not "stored exact zeros" — the two are opposite in effect and the padding here
      was 1e-30.** A stored *exact* zero survives no pruning assembly and, when it does reach the
      factorization, is measured to break the zero-shift V-cycle outright (see the stored-zeros entry above).
      Recovery is not cheap:
      `smoother_sweeps=3` / ILU(2) still diverge *and* erase the setup win; restoring the reach-3 pattern
      converges but forfeits the GAMG-setup half of the economy (the larger half — refresh is
      setup-dominated), leaving only the marginal probe-colour saving. The one open lever is a fundamentally
      different smoother that tolerates the sparser graph (a smoother-design task). On a genuinely SKEWED mesh
      reach-2 would additionally be *lossy* (the non-orthogonal ring pushes real content to distance-3) — a
      second reason to keep reach-3.
      **One correction worth carrying, because it was believed for a while:** an apparent "47.2M → 39M
      nnz decay" in the pattern was a conflation of a *pattern* count with a *live* count — the live
      Jacobian holds ~38.7–39.0M nnz throughout, roughly constant. The reach-3 requirement above rests on
      the padding experiment, which is sound; it does not rest on any decay.
## Preconditioner — monolithic complete-LU

- **Monolithic COMPLETE-LU preconditioner — BUILT (`lu_preconditioner.py`), the preferred 2D/moderate
  coupled preconditioner.** It factors the assembled coupled Jacobian
  *completely* (`MonolithicLuPreconditioner`), so it is the operator's **exact** inverse and a Krylov
  solve converges in **one** iteration. Verified on the real forward operator and the β=0 adjoint
  (true-residual checked). Because the fill is pattern-determined it is also **state-robust**
  (no drop-tolerance tail that shifts with the flow). Built on `HostPreconditioner` (`build` /
  `refresh_in_place` / `matvec`), a host object applied via `pure_callback`, riding as a static field;
  the adjoint reuses the factorization's transpose. **No equilibration / cell-major reordering** — the
  complete factorization's own pivoting + fill-reducing ordering handle the indefinite saddle on the raw
  field-major matrix; equilibrating + cell-major actually *hurt* it, measured.
  - **Pluggable backend (`factorize_lu(backend=…)`):** `"umfpack"` (SuiteSparse via the optional
    `petsc4py` dep) is the fast path — a fill-reducing (nested-dissection/AMD) ordering + a multifrontal
    BLAS-3 numeric kernel. A refresh **re-factors from scratch** (NOT a fixed-pattern numeric-only
    refactor): the coupled Jacobian's sparsity *grows* as the flow develops — cross-coupling entries that
    are exactly zero at the cold reference become nonzero — so a frozen-pattern refactor is both wrong and
    a shape error; the full factor is fast enough (~1 s at 2D/moderate) that re-analysing each refresh is
    cheap. `"scipy"` (`scipy.sparse.linalg.splu`, SuperLU) is the always-available fallback: exact and
    correct (what the tests run under) but, lacking nested dissection, slower to factor than UMFPACK.
    `"auto"` (default) picks UMFPACK when importable, else SciPy. So the module imports with no optional
    dependency; the faster factorization is opt-in via `pip install aquaflux[petsc]`.
  - **SCOPE — a 2D / moderate-mesh tool (binding).** A complete LU's fill is `O(n log n)` in 2D but
    `O(n^{4/3})` in 3D, so **memory is the wall in 3D** — measured (synthetic block grids): 2D factor time
    ~`dof^1.37` (comfortable to ~10⁵ cells, seconds, <10 GB), but 3D hit **out-of-memory at ~10⁴ cells**.
    So this preconditioner is the fast, exact choice for 2D / moderate meshes; large 3D needs the
    algebraic multigrid path (`amg_preconditioner.py`, below), or a **rank-structured direct solver**
    (MUMPS-BLR / STRUMPACK — the fill-taming way to keep this exact-factor paradigm in 3D, reachable via
    the same PETSc dep).
  - **A level-based ILU(k) via PETSc, tried as a cheaper alternative, was a MEASUREMENT ARTIFACT.** It
    looked faster but was a **preconditioned-norm artifact** (PETSc's KSP converges on ‖Mr‖, not the true
    ‖Ax−b‖); it is weaker, not stronger — always verify the TRUE residual.
  - **FROZEN is wrong for the β-ramping dual-time march — track β (binding, measured).** A complete LU is
    *exact* only for the operator it factored, `J + β d`. In a dual-time march β ramps (0.5 → 0.005), so a
    factorization frozen at one β **mis-preconditions** the operator actually solved — measured on rung2:
    a LU frozen at β = 0.05 needs 25 / 111 / 217 / **474** GMRES iters at β = 0.1 / 0.5 / 1 / 2 (vs **1**
    when factored at the matching β), and on a real cold ramp the frozen LU **NaN'd** on the overshot
    low-β state (215 cycles → failure). Because the LU factor is cheap (~1 s), the fix is to
    **re-factor at the current `(state, β)` every step** (`forward_march`'s `precondition_step` seam +
    `lu_beta_tracking_refresh`, `.claude/rules/turbulence.md`): exact each step (1 Krylov iter), and robust
    through overshoots (measured: completes the cold ramp where the frozen LU failed, cyc ≤ 18). The
    finishing solve and adjoint keep the last frozen factorization (exact enough at the converged β → 0).
  - **Coupled builders (`coupled_lu_continuation` / `coupled_lu_refreshing_continuation`, and the
    β-tracking `lu_beta_tracking_refresh`) live in `.claude/rules/turbulence.md`;** they share the
    `MonolithicFactorShiftPolicy` and the `_monolithic_factor_step` builder tail with the algebraic
    multigrid (one implementation, parameterized by the factorization).

