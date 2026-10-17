# Investigation log — `aquaflux/precondition/` the materialized Jacobian (`solve-direct-preconditioners.md`)

> Split out of `.claude/rules/solve-direct-preconditioners.md` (2026-10-17). **Lives in `.claude/notes/`,
> outside the auto-loaded `.claude/rules/` tree, so it never auto-loads.** It holds the dated measurements
> and investigation behind the coloured-probe materialization of the coupled Jacobian (`sparse_jacobian.py`)
> — moved verbatim, under headings mirroring where they came from. See `solve-direct-preconditioners.md`
> for the current contract and status. Several entries name since-deleted code (`MonolithicVCycleInverse`,
> `MonolithicVCyclePreconditioner`, `ShiftedCellMajorOperator`, PETSc, the incomplete-LU smoothers); they
> are history and cannot be re-run from this tree. The measurements of the deleted preconditioners
> themselves were deleted, not moved — see `solve-refuted-directions.md`.

## Materializing the coupled Jacobian

### The gather map's build peak (2026-08-14)

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

### The sort permutation removed

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

## Per-column probing reach — bfs3d's history

    - **⚠️⚠️ PER-COLUMN PROBING REACH — bfs3d's history (shipped `(3,3,3,3,2,2)` until #712); `p` had to stay at reach 3.**
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
      while k/ω are in the trailing block, whose traced cell-block inverse has no factorization pattern to
      weaken.
      ⚠️ **It also violated the "full pattern, no `eliminate_zeros`, so the structure is truly fixed"
      invariant the in-place GAMG refactor relies on — for EVERY arm**, since the pruned set is whichever
      entries happen to be exactly zero at that state, which moves with the state.
      **✅ FIXED — pattern-preserving assembly, and there were TWO pruning sites, not one (BUILT).**
      Sparse *arithmetic* is what prunes, so both the shift add and the equilibration had to change:
      - `frozen_operator.apply_symmetric_scale(data, indptr, indices, scale, chunks=…)` scales the stored
        values in place (row-chunked, so the row factor's per-nonzero expansion never allocates an array
        the size of the values). `symmetrically_equilibrate` uses it in place of `diags(s) @ a @ diags(s)`.
      - `MonolithicVCyclePreconditioner._shifted` adds the shift by **diagonal assignment**, not
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
      and stored zeros are exactly what `MonolithicVCycleInverse._live` removes, because the incomplete factorization
      cannot take them. So with the operator pruned at the factorization the `p` column is back in the
      configuration that diverges the case, and the case default must stay `(3,3,3,3,2,2)`. **Predicted
      from the mechanism, not re-measured on a march** — the divergence itself is the measured #191
      result, at β > 0 under a pruning assembly.

### Root cause of #191 (2026-08-12)

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
        (`MonolithicVCycleInverse._live`), which puts `p` at reach 2 back in the configuration above. **The shipped
        default is `(3,3,3,3,2,2)`** — 454 probes against 564, −16 % — and `p` at reach 3 IS a correctness
        constraint after all. The march numbers here stand as a measurement of the preserving arm; they are
        not a licence for the shortened default under the shipped one.

### The SIMPLE-smoothed hierarchy and the narrowed probe (open)

      - **The family it is FOR is the SIMPLE-smoothed hierarchy, which takes the saving for nothing.**
        The mechanism is which preconditioners inherit the stored *sparsity*: an incomplete factorization
        takes its pattern from it, so a corrupted or narrowed pattern gives a correspondingly different
        factor, while `SimpleSmoothedInverse` relaxes through diagonal and Schur approximations and takes
        no pattern at all. Measured on pitzDaily, that arm is **reach-insensitive** — the same 71 steps at
        reach 3 and reach 5, cycles within 3 % (395 vs 408), identical final residual — while the Jacobian
        halves and the march is 31 % shorter. **The untested pairing worth running is
        `probe_gradient_sweeps=1` + `stencil_reach=3` + a SIMPLE-smoothed hierarchy:** exact Jacobian,
        4.72M nonzeros against 13.32M, and a preconditioner indifferent to the pattern.

## Materialize efficiency

### Per-column probe cost

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

### Probe batch sweep (2026-08-11)

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

### Where a run's memory goes

      **⚠️ WHERE THE MEMORY ACTUALLY GOES, measured by attribution rather than assumed:** peak RSS is
      149 MB after imports, **6889 MB after `build_case`**, 7154 MB after the colouring and plan, 8359 MB
      after the gather map. So the case assembly is ~6.7 GB, the gather map ~1.2 GB, and a whole
      materialize 0.4 GB. **A run's memory ceiling is set by building the case, not by probing it** — the
      probe is now a rounding error against it, and an initial reading that blamed the structure build was
      wrong.
