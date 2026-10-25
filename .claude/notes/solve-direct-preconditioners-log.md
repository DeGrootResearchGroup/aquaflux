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

      **bfs3d by stage, 2026-10-17** (`validation/bfs3d_openfoam/memory_by_stage.py`, one process, Linux
      `VmRSS`/`VmHWM` plus a 20 ms sampler for each stage's own peak; commit `9efc5759` plus this branch's
      uncommitted mesh move, no solver code changed; jax 0.11.2, 4-core Xeon 2.1 GHz, 15 GiB, no swap, glibc
      allocator defaults, compilation cache cold; shipped `case.yaml`). GiB:

      | stage | stage peak | RSS after | ΔRSS | HWM |
      |---|---|---|---|---|
      | imports | 0.43 | 0.43 | +0.42 | 0.43 |
      | mesh read (23040 cells) | 0.46 | 0.46 | +0.03 | 0.47 |
      | geometry | 0.53 | 0.50 | +0.04 | 0.54 |
      | case build (its own mesh + geometry, blocked wall distance) | 0.84 | 0.83 | +0.32 | 0.84 |
      | `hybrid_initialize` | 1.53 | 1.49 | +0.66 | 1.53 |
      | first residual, compiled and run | 1.62 | 1.59 | +0.10 | 1.62 |
      | first JVP, compiled and run | 1.83 | 1.80 | +0.21 | 1.83 |
      | freed, then the march's own build + initialization to its first fit | 1.71 | 1.65 | +0.23 | 1.83 |
      | probe plan, colouring and gather map | **3.19** | 2.06 | +0.40 | 3.19 |
      | fit 1 (probe responses + block inverses) | **5.67** | 4.56 | +2.50 | 5.67 |
      | from the first fit to the first re-fit | 6.00 | 5.90 | +1.34 | 6.00 |
      | re-fit 1 (β 0.5) | 8.61 | 6.94 | +1.04 | 8.64 |
      | step 1 (march step compiled) | 9.71 | 8.76 | +1.82 | 9.71 |
      | re-fit 2 (β 0.167) | 11.28 | 9.52 | +0.76 | 11.28 |
      | step 2 | 10.40 | 9.45 | −0.07 | 11.28 |
      | re-fit 3 (β 0.0556) | 11.73 | 9.81 | +0.36 | 11.74 |
      | step 3 | 10.56 | 9.60 | −0.21 | 11.74 |

      The case build is 0.84 GiB (the PETSc-era ~6.7 GB is gone); **the ceiling is set by the materialized
      Jacobian's fits and re-fits and the compiled march step**, each re-fit transiently ~1.5–2.7 GiB above
      the resident set, which grew by a smaller amount per re-fit (+1.04, +0.76, +0.36). The whole shipped
      march (29 steps, 29 re-fits) peaked at **12.1 GiB** (`VmHWM` of the case process, sampled every 15 s):
      it fits 15 GiB with ~3 GiB to spare, and two of it would not.

### bfs3d memory by owner (2026-10-18)

      `validation/bfs3d_openfoam/memory_attribution.py` (per stage: `VmRSS`/`VmHWM`, a 20 ms sampled peak,
      `tracemalloc` current and peak, JAX live arrays by dtype, smaps, glibc `mallinfo2`; censuses naming
      every array over 1 MiB). 4-core Xeon 2.1 GHz, 15.7 GiB, no swap, jax 0.11.2, glibc defaults, shipped
      `case.yaml`, 2-step marches (`BFS3D_MEMORY_STEPS=2`), warm compilation cache.

      - **Before #757 (`cf80be32`)**, HWM 10.90 GiB, set inside each re-fit's leading-block re-fit and
        rising per re-fit. At re-fit 2: device 4.9 GiB (float64 hierarchy 1.23 — `values` and the stored
        transpose `t_values` both float64 — three float32 traced splits at 0.63 each, int32 indices 1.82
        including a generation built per re-fit); host transient 3.8 GiB at peak over a 1.6 baseline
        (probed CSR, shifted copy and the four cut blocks at 0.41 GiB each, plus the re-fit's products);
        the rest runtime, the step program's 984 MiB of temporaries, allocator retention. One full CSR copy
        is 36.7M stored entries: 280 MiB float64 values + 140 MiB int32 indices. `ProbeGather` keeps
        3 x 140 MiB; its build's transient (~1.1 GiB) and the case build (0.87 GiB) are below the peak.
        Compiled code is ~20 MiB. A `gc.collect()` at every stage freed nothing: every copy is held (once the march returns, a
        collection does free its session's cycles: 2026-10-24 entry "What the resident memory holds ...").
      - **After #757 (`fe772f59`)**, HWM **11.91 GiB**: the held entry maps add ~1.5 GiB of host memory
        (traced host after step 1 3.07 against 1.57) and the first re-fit's host transient peaks at 5.9 GiB;
        int32 no longer grows per re-fit except one generation (the first re-fit's held layouts beside the
        fit's `from_scipy` packing).
      - **One traced split per fit** (`MaterializedSession._split`): HWM 11.91 -> **10.81 GiB**, steady
        device 4.10 -> 3.48 GiB. Two float32 splits remain: the march's current one and the fit-time one,
        held by `solve.staged_march`'s local `strategy` (the step it built, kept for the whole segment;
        found by scanning stack locals, since a running frame's locals are invisible to
        `gc.get_referrers`) and, until 2026-10-18, by `compare.py`'s per-rung residual reporter.
      - **Hierarchies held at the split's precision** (`stored_precision`): HWM 10.82 -> **10.15 GiB**,
        steady resident 9.38 -> 8.19 GiB, device float64 1.23 -> 0.15 GiB. The peak falls less than the
        steady state because each re-fit still forms its new hierarchy in float64 before rounding it.
      - **A re-fit releases the superseded split** (`_release_superseded`): HWM 10.15 -> **9.42 GiB**,
        steady resident 8.19 -> 7.14 GiB, device float32 1.25 -> 0.63 and int32 0.99 -> 0.65 GiB (the
        fit-time index generation the stale split pinned: what was left of "lever 2"). Both steps ran with
        nothing applying a released array. Cumulative over the main baseline: 11.91 -> 9.42 GiB peak,
        10.58 -> 7.14 GiB steady.
      - **The split's probe laid out by its blocks** (no whole CSR, shifted copy or cut): HWM 9.42 ->
        **8.47 GiB**; held host 3.12 -> 2.74 GiB (the block-cut entry maps are gone); re-fit 2's own peak
        10.9 -> 8.4 GiB. Cumulative over the main baseline: **11.91 -> 8.47 GiB** (2 steps). What sets
        the peak now is the first re-fit's leading-block re-fit, where the entry maps (#757) are built
        (host transient 4.8 GiB over 2.6), and the step program's ~1 GiB of temporaries.
      - **Whole shipped march, branch against `main` (`fe772f59`)**, `run_case.sh`, `VmHWM` sampled every 15 s:
        step tables identical (shift, inner, cycles, `R`, `a_min`, `rho1`, flags) through all 34 steps both
        reached; peak **8.54 GiB against 12.21 GiB**; 3439 s against 3636 s to step 34; re-fits 31, mean
        42.7 s (probe 32.6, refactor 10.0) against 47.4 s (33.8 / 12.2). Both died at step 35 on the same
        singular `[k, omega]` cell block (`validation-log.md`, "The shipped bfs3d march fails at step 35").
      - **pitzDaily, branch against `main`**: step tables identical, 21 steps / 106 cycles / `R` 2.618e-06;
        warm-cache wall 173 s against 181 s; peak `VmHWM` 3.17 against 3.16 GiB (its peak is not set by
        these copies).
      - **The split's coupling and SIMPLE pieces held in float32 as formed, the levels narrowed one at a
        time (2026-10-21, `claude/float32-storage-c15425` against `9ffe291b`)**: 11-core M-series Mac,
        18 GiB, jax 0.10.2, `MallocLargeCache=0`, shipped `case.yaml` (split `float32`, `PivotFloor(0.15)`),
        `BFS3D_MEMORY_STEPS=2`, warm cache, one run per arm. Peak footprint **9.70 -> 9.67 GiB**; block
        re-fit #1's stage peak 9.68 -> 9.62, its `tracemalloc` peak 5.03 -> 4.99 GiB; held host after the
        first re-fit 2.85 -> 2.81 GiB (the coupling's 80 MiB of float64 values now 40). Every traced split
        (fit and both re-fits, 293 leaves each) has the same SHA-256 leaf digests on both arms
        (`BFS3D_MEMORY_CHECKSUMS=1`), and the march tables agree except wall time. **So float64
        formation of the hierarchy and pieces is not what sets the peak**: the first re-fit's ~2 GiB host
        transient is the entry maps' build (#757; `EntryMap._target` 469 MiB, `_source` 171 MiB,
        `SparsePattern.indices` 178 MiB held at its end) and scipy's temporaries, and the device float64
        transient freed by per-level narrowing is a few tens of MiB at that moment. What is still float64
        and is not stored in float32, because the fit reads it in double: the probe's value array (280 MiB,
        transient per fit), the level operators and diagonals until `derive_extras` has read them
        (`to_scipy`), the prolongation values (a refit reads them back), and the entry maps' weights
        (`None` for the 0/1 transfers both blocks use).

### bfs3d memory by owner after the device fit, probe and seeds (2026-10-24)

      `memory_attribution.py` on `7fe27d07` (`main` after #809: the fit, the probe's values, shift, pivot
      floor, coupling and seeds all on the device, #804–#808, measured on pitzDaily only until now). 4-core
      Xeon 2.8 GHz container, 15.7 GiB, no swap, THP `madvise`, glibc 2.39 defaults, jax 0.12.0.1,
      compilation cache cold, shipped `case.yaml`, `BFS3D_MEMORY_STEPS=2`, census on. ⚠️ The container
      restarted during step 2 (no exit record, cause unknown; the record's earlier restart was at 12.3 GiB
      resident), so step 2's row and the final census are missing; the JSON record is written per stage
      and everything through re-fit #2 survived. GiB:

      | stage | s | stage peak | resident after | device | host traced | malloc free |
      |---|---|---|---|---|---|---|
      | case build | 58 | 1.00 | 1.00 | 0.10 | 0.13 | 0.15 |
      | `hybrid_initialize` | 74 | 1.63 | 1.52 | 0.13 | 0.15 | 0.34 |
      | probe plan (the gather map's build) | 61 | **4.04** | 2.19 | 0.41 | 0.45 | 0.35 |
      | fit, first build (block inverse build #1 sets it) | 754 | 6.41 | 5.63 | 2.72 | 0.59 | 0.57 |
      | re-fit #1 | 47 | 7.55 | 6.50 | 3.52 | 0.62 | 0.51 |
      | march step, first call (compile + run) | 177 | 8.18 | 7.23 | 2.98 | 0.70 | 1.40 |
      | re-fit #2 (its leading `block re-fit` sets it) | 50 | **8.87** | 7.75 | 3.52 | 0.76 | 1.26 |

      - **The peak-setter moved.** The first re-fit's entry-map build (the 2026-10-22 peak-setter, a 4.8
        GiB host transient) is gone: the maps are built at the first fit, on the device, and re-fit #1's
        host transient is 0.64 GB. The peak is now a later re-fit's leading block re-fit on top of a
        baseline the step program's first call raised (retained free memory 0.51 → 1.40 GiB across it).
        The peak is 0.12 GiB above the 2026-10-22 record (8.75) with the tracer (0.32) included, so level.
      - **The memory moved from the host to the device, and the device holds 3.3× more int32 than
        float32.** At step 1's end: device 2.98 GiB = int32 **2.19** + float32 0.66 + float64 0.13; host
        traced arrays 0.42 beside the tracer. By owner (device, MiB): `DeviceEntryMap` 1082 over 66
        arrays; `_PaddedLayout` 591 over 40 (`columns`, `t_columns`, the packing maps); `ProbeGather` 280
        (two int32 arrays over the 36.7M entries); `_SimplePieces` 193; the hierarchies' `values` 171 and
        `t_values` 171; `SimpleSmoothedInverse` 131; `SimplePlan` 108; `Corrections` 89 over 28; host
        `SparsePattern` 140. Every int32 array describes the one reach-3 block-stencil pattern on the cell
        graph at dof level (36.7M entries against ~1.0M cell pairs).
      - **The stored transposes are about half of every operator**: `t_values`/`t_columns` beside
        `values`/`columns` on all 9 hierarchy levels, the SIMPLE pieces (Schur 48 + 48 MiB, divergence and
        velocity correction 24 + 24 each) and the coupling (44). The forward march applies only `A x`.
      - **Not arrays (at step 1's end, resident 7.24)**: freed memory glibc retains 1.38 (0.75 of it
        kept after the step program's first call, in the XLA worker threads' arenas), compiled code 0.03,
        and ~2.1 that is JAX's compiled-program cache (named in "What the resident memory holds beyond
        live arrays, on two meshes", below). The case build (0.98) and `hybrid_initialize` (+0.52) are
        almost all of this kind (arrays 0.26 of 1.52): the programs they compile.
      - A re-fit holds the old and new leading fine operator together (`float32(92160, 252)` ×4 against ×2
        at a step's end, +177 MiB) until `_release_superseded` runs.
      - **The compiled programs' temporaries** (`memory_analysis()`, a 1-step census-off run of the same
        configuration with the cache warm; two runs, peaks 8.07 and 8.46 GiB, fits 325 and 265 s): the
        **march step 974 MiB** of temporaries over 1373 MiB of arguments (the split's leaves and the
        state); batched JVP of 8 tangents **829 MiB** (arguments 98); single JVP 202; residual 106. So the
        probe's temporary is ~100 MiB a tangent plus ~30, as the 2026-10-18 census read it; a Krylov
        application's JVP needs ~200 MiB and the residual ~100; and ~770 MiB of the step's temporary is
        the split's padded products and the solve's own vectors, which is the 14a answer: the step's
        temporaries follow the split's storage, not the residual's. (The harness armed the step's capture
        only after step `STEPS − 1`, so a 1-step run printed no step size; fixed the same day.) Two 1-step
        runs of one configuration spread 0.4 GiB in peak, so a bfs3d peak is not pinned closer than that.
      - Record: `validation/bfs3d_openfoam/memory_attribution.json` and the run log (both git-ignored),
        copied into this entry; the 2026-10-18 `memory_by_stage.py` configuration is otherwise unchanged.

### Cell-level pattern (§14d), stages 1–3 (2026-10-24)

      Branch `claude/cell-level-pattern` from `78a308f3` (= `7fe27d07` + the memory record): stage 1
      `538b41c0` (the probe's entry order, `_CellEntries`), stage 2 `487b1400` (`PaddedColumns`, cell-level
      padded layouts), stage 3 `d044e4d1` (`KroneckerGalerkin`, `KroneckerGather` for the SIMPLE cuts and
      `unpack`), counted decode `d7c6751c`. 4-core Xeon container, 15.7 GiB, jax
      0.12.0.1, glibc defaults, shipped `case.yaml`s.

      **bfs3d, `memory_attribution.py`, `BFS3D_MEMORY_STEPS=1 BFS3D_MEMORY_CENSUS=0
      BFS3D_MEMORY_CHECKSUMS=1`** (one run per arm; base ran with a cold compilation cache, the others
      partly warm). GiB, device = int32 / total:

      | stage | base `78a308f3` | stage 1 `538b41c0` | stages 1–3 `d044e4d1` |
      |---|---|---|---|
      | probe plan, stage peak | 4.03 | 1.79 | 1.74 |
      | fit (first build): device int32 / total | 2.02 / 2.72 | 1.75 / 2.45 | 0.39 / 1.09 |
      | re-fit #1: stage peak | 7.43 | 7.60 | 6.76 |
      | march step 1, first call: stage peak | 8.07 | 8.12 | 7.19 |
      | step 1 end: device int32 / total | 2.19 / 2.98 | 1.93 / 2.72 | 0.40 / 1.20 |
      | step 1 end: resident | 7.13 | 7.17 | 6.11 |

      - Every float leaf of both traced splits (322) has the base's digest at stages 1–3; step 1's inner
        rows are identical. Stage 1's whole-tree digests equal the base's (stages 2–3 add integer leaves,
        so only the float leaves compare).
      - Stage 1 alone: device int32 −0.26 GiB (`ProbeGather`), yet resident at step 1's end level (+0.04):
        malloc's arena `in_use` rose ~0.4 GiB at the fit against base with tracemalloc's numpy total
        equal; not attributed (no census), and the compilation cache differed between the two runs.
      - The compiled programs' sizes the harness prints (batched JVP 828.9 MiB temp, single JVP 201.5,
        residual 105.8) are unchanged; the step program is not among them.

      **pitzDaily `march_cost.py`, `AQUAFLUX_DISABLE_COMPILATION_CACHE=1`, one run per arm:** base 21
      steps / 102 cycles / final R 5.736e-06 / `x_r/h` 8.0686 / ν_t/ν peak 418.09, VmHWM 3.163 GiB,
      solve 222.4 s (refits 26.7 s: probe 13.9, refactor 12.3); stages 1–3 the same counts with every
      step row identical (β, inner steps, cycles, R, α), VmHWM 3.016 GiB, solve 212.5 s (refits 23.6 s:
      probe 12.7, refactor 10.4).

      **Not bit-identical at default compiler flags on pitzDaily, and that is fused multiply-add (FMA)
      placement:** the hybrid start differs from base by ≤ 6.2e-15 (and so the split's leaves, ≤ 1.7e-15
      relative). Bisected: the cell-level padded layouts alone move it (the Galerkin and SIMPLE maps do
      not); under `XLA_FLAGS=--xla_cpu_max_isa=AVX` (no FMA) base and branch starts are identical; held
      and cell-level operators built from one start give identical leaves and bit-equal applies in
      isolation, in a loop and transposed. The values are the same; a changed program contracts
      multiply-adds elsewhere in the potential-flow solve. bfs3d's start did not move.

      **Apply time.** Isolated pitzDaily split apply (`split_apply_timing.py`, hybrid start, float64
      traced split, best of 15): base 71.7–71.9 ms forward / 66.5–67.5 transposed; stages 1–3 with the
      division decode 81.1–82.7 / 82.7–82.9; with the counted decode 86.9–96.6 / 83.9–84.4 (the counted
      decode is faster per operator below, so the split-level number moved within its noise). On a
      bfs3d-sized lattice (23400 cells, reach 3, float32), per padded operator, isolated, held → cell:
      division 4×4 33.7 → 52.1 ms, 2×2 2.4 → 20.2, 2×4 16.5 → 38.9; counted 32.4 → 35.0, 2.4 → 4.4,
      16.3 → 19.8 (transposed 33.4 → 33.4, 2.4 → 4.4, 16.3 → 14.7). Ten applies inside a `fori_loop`:
      4×4 176 → 168 ms, 2×2 25.4 → 25.3 (XLA hoists the decode out of the loop, so the decoded columns
      are the loop's temporary). Index storage per operator (both directions, with the packing entries):
      4×4 360.6 → 17.2 MiB, 2×2 90.3 → 17.2, 2×4 180.5 → 17.2.

      **By owner at step 1's end (census, `a287287d`, `BFS3D_MEMORY_STEPS=1`, census on, 2026-10-24;
      arrays of at least 1 MiB; device 1.12 GiB, host traced 0.15):** `_KroneckerColumns` 264.0 MiB over
      41 arrays (the cell-level padded columns, both directions, every level and piece: was
      `_PaddedLayout` 591 with the packing maps under `DeviceEntryMap`); level operators' `values` /
      `t_values` 171.3 + 171.3; host `SparsePattern` 140.1 over 3 (unchanged); `Corrections` 88.6;
      `_SimplePieces` Schur 48.2 + 48.2; the adjoint form's coupling 44.3 + 44.3; `KroneckerGalerkin`
      40.6 over 14; `KroneckerGather` 30.6 over 10 (was `SimplePlan` 108); `_CellEntries` 10.0 (was
      `ProbeGather` 280); `SimplePlan` 6.1. No `DeviceEntryMap` array of 1 MiB remains (was 1082 over 66).
      Run peak 6.87 GiB (step 1's first call), against 8.07 in the census-free base run above.

### A `malloc_trim(0)` before each re-fit, on bfs3d (2026-10-24)

      `memory_attribution.py` on `042e5107` (`7f62a3c4` plus the trim, then opt-in: `MaterializedSession(memory_trim=)`
      handed to `BetaTrackingRefresh`, which calls `aquaflux.allocator.GlibcTrim` once per re-fit, before its
      probe; the solver is otherwise `7fe27d07`'s). 4-core Linux container, 15.7 GiB, no swap, glibc 2.39
      defaults (no `MALLOC_*`), jax 0.12.0.1, shipped `case.yaml`, `BFS3D_MEMORY_STEPS=2
      BFS3D_MEMORY_CENSUS=0`, `BFS3D_MEMORY_TRIM` 0/1, one arm per process, arms alternated; run 1 compiled
      cold, the rest warm. GiB:

      | run | trim | peak (`HWM`) | set by | re-fit #1 stage peak | step 1's first call | resident before re-fit #2 -> after the trim | `malloc free` at step 1's end | resident at step 2's end |
      |---|---|---|---|---|---|---|---|---|
      | 1 | off | 8.92 | re-fit #2's `block re-fit #3` | 7.41 | 8.24 | 7.30 | 1.49 | 7.47 |
      | 2 | on | 8.53 | re-fit #2's `block re-fit #3` | 7.67 | 8.31 | 7.38 -> 6.74 | 1.03 | 7.32 |
      | 3 | off | 9.16 | re-fit #2's `block re-fit #3` | 7.81 | 8.49 | 7.55 | 1.12 | 7.62 |
      | 4 | on | 8.51 | re-fit #2's `block re-fit #3` | 7.63 | 8.31 | 7.40 -> 6.73 | 0.99 | 7.26 |
      | 5 | off | 9.04 | re-fit #2's `block re-fit #3` | 7.80 | 8.33 | 7.39 | 0.96 | 7.63 |
      | 6 | on | 8.51 | re-fit #2's `block re-fit #3` | 7.64 | 8.27 | 7.35 -> 6.73 | 0.95 | 7.19 |

      - **Peak: off 8.92 / 9.16 / 9.04 (mean 9.04, spread 0.24); on 8.53 / 8.51 / 8.51 (mean 8.52, spread
        0.02).** Means 0.52 apart; the closest pair (runs 1 and 2) 0.39. The trim also took out the run-to-run
        spread: what varied between off runs was how much freed memory the arenas happened to keep.
      - **The peak stays where it was** (re-fit #2's leading block re-fit) and is now 0.2--0.3 above the step
        program's first call (8.27--8.31 on), the next thing a lower re-fit peak would meet.
      - **The trim before re-fit #2 gives back 0.61--0.67 GiB of resident memory** (0.28--0.37 before re-fit
        #1). `mallinfo2`'s `fordblks` does not move across it: `malloc_trim` returns free pages to the kernel
        (`madvise`) and glibc still counts them as free, so the harness's `malloc free` column is not what a
        trim lowers; read the resident set.
      - Held resident at step 2's end 7.19--7.32 on against 7.47--7.63 off. No cost in time: re-fit #2's stage
        30.3--33.5 s on against 31.4--34.2 off, the 2-step march 578--622 s on against 598--611 off (warm).
      - Run 6 ran on `817c95a5` (this entry's first commit; no code change from `042e5107`). Records: the
        `memory_attribution_{off,on}{1,2,3}.json` and `run-20261024-02*.log`/`-03*.log` (git-ignored), copied here.
      - **Shipped as the session's default** the same day (the owner's decision: the mean drop met the 0.5 GiB
        bar, the closest pair did not). `memory_attribution.py` now trims by default; `BFS3D_MEMORY_TRIM=0`
        is the off arm. Not yet measured: a whole march's peak.
      - **Re-measured after #817** (cell-level pattern, entry above) on `2e7949e8` (`main` with #817 and the
        trim), same machine, jax, allocator, case and harness settings; run 1 recompiled #817's changed
        programs, the rest warm. #817's 1-step runs peak in step 1's first call; in a 2-step run re-fit #2's
        leading block re-fit (`block re-fit #3`) still sets the peak, about 1 GiB above that first call
        (6.68--6.87), so the trim still lands on the peak. GiB:

        | run | trim | peak (`HWM`) | re-fit #1 stage peak | resident before re-fit #2 -> after the trim | `malloc free` at step 1's end | resident at step 2's end |
        |---|---|---|---|---|---|---|
        | 1 | off | 7.88 | 6.43 | 5.88 | 1.38 | 6.07 |
        | 2 | on | 7.31 | 6.32 | 5.89 -> 5.14 | 1.13 | 5.74 |
        | 3 | off | 7.67 | 6.47 | 5.69 | 0.90 | 5.97 |
        | 4 | on | 7.27 | 6.30 | 5.75 -> 5.12 | 0.99 | 5.70 |
        | 5 | off | 7.78 | 6.47 | 5.79 | 0.99 | 6.03 |
        | 6 | on | 7.33 | 6.33 | 5.84 -> 5.15 | 1.05 | 5.74 |

        Peak: off mean **7.78** (spread 0.21), on mean **7.30** (spread 0.06): 0.48 apart, closest pair 0.34.
        The trim before re-fit #2 gives back 0.62--0.76 GiB of resident memory. The 2-step march took
        537--563 s on against 539--621 s off (run 1 recompiling), so no cost in time. Records:
        `memory_attribution_r817_{off,on}{1,2,3}.json` and `run-20261024-{195807,2*}.log` (git-ignored).

### The probe's host block patterns at cell level (2026-10-24)

      `CellLevelPattern` (`d9fe819d`, branch `claude/host-cell-patterns`) against `main` `0c76b4ec` (#817):
      bfs3d `memory_attribution.py`, `BFS3D_MEMORY_STEPS=1 BFS3D_MEMORY_CENSUS=0
      BFS3D_MEMORY_CHECKSUMS=1`, one run per arm, `main` first (its compilation cache warmer for the
      second), 4-core Xeon container, jax 0.12.0.1. GiB, `main` → branch:

      | stage | stage peak | resident after | host traced |
      |---|---|---|---|
      | probe plan | 1.79 → 1.65 | 1.78 → 1.56 | 0.321 → 0.183 |
      | fit (first build) | 5.55 → 5.37 | 4.36 → 4.10 | 0.608 → 0.471 |
      | re-fit #1 | 6.54 → 6.30 | 5.19 → 4.85 | 0.639 → 0.502 |
      | march step 1, first call | 7.05 → 6.71 | 6.05 → 5.72 | 0.717 → 0.580 |
      | step 1 end | 6.07 → 5.73 | 6.07 → 5.73 | 0.729 → 0.592 |

      - Host traced arrays fall 0.137–0.138 GiB at every stage: the three block patterns (140 MiB). The
        resident and peak readings fall more (0.24–0.34 GiB), which is within the 0.4 GiB spread two
        1-step runs of one configuration showed (entry above); read only the traced column as this
        change's.
      - Device memory unchanged (1.20 GiB at step 1's end); all 1130 split leaf digests (both splits,
        integer leaves included) and step 1's inner rows identical to `main`'s.
      - pitzDaily, live host `SparsePattern`s after the session's build: 22.1 → 4.9 MiB (the three block
        patterns now one shared 1.2 MiB cell pattern).
      - ⚠️ Measured before #818 made `malloc_trim(0)` before each re-fit the session's default; both
        arms ran without it, so the resident and peak columns are not today's. The host traced column does
        not depend on it.

### What the resident memory holds beyond live arrays, on two meshes (2026-10-24)

      `validation/memory_residue.py` (runs `memory_attribution.py`'s march and stages on either step case)
      on `13fa17d4` (`main` `faea4a64` + the harness; library unchanged since `7fe27d07` but for #811's
      march log), same 4-core Linux container as the entry above (15.7 GiB, no swap, glibc 2.39 defaults,
      jax 0.12.0.1), compilation cache warm for both meshes (59 entries), shipped `case.yaml`s,
      `BFS3D_MEMORY_STEPS=1`, census off, tracer on (0.17 / 0.30 GiB), `RESIDUE_READ=1 RESIDUE_TRIM=1
      RESIDUE_CLEAR=end` (a `malloc_trim(0)` at every stage's end, so the peaks are not the record's).
      pitzDaily 12 225 cells (2D), bfs3d 23 040 (3D). Readings at `end`, after the measured step and its
      return from the march, GiB:

      | | pitzDaily | bfs3d | slope per cell | at zero cells |
      |---|---|---|---|---|
      | resident at `end`, after a trim | 2.52 | 6.62 | | |
      | freed by `gc.collect()` + trim (live arrays freed) | 0.20 (0.18) | 1.75 (1.60) | run-end only | |
      | freed by `jax.clear_caches()` + collect + trim | **1.06** | **2.09** | ~100 KB | ~0 |
      | of which traced by `tracemalloc` (Python: jaxprs, `SourceInfo`) | 0.19 | 0.38 | ~19 KB | ~0 |
      | of which untraced (XLA executables, compiler state) | 0.87 | 1.71 | ~83 KB | ~0 |
      | loaded executables (before the clear) | 1072 | 1242 | | |
      | left: live arrays (the harness's captured step arguments) | 0.17 | 1.38 | | |
      | left: file-backed / anonymous holding no array | 0.22 / 0.42 | 0.22 / 0.54 | | |
      | glibc retention given back after the step program's first call | 0.38 | 0.68 | | |

      - **The record's "~2.1 GiB the census does not see" is JAX's compiled-program cache**: 2.09 GiB on
        bfs3d (1.71 untraced + 0.38 already inside the traced total), held as long as the executables
        are. It grows by stage with the executable count (bfs3d 539 after the case build, 692 after
        `hybrid_initialize`, 1215 after the fit, 1242 at the end; pitzDaily 514 / 675 / 1045 / 1072), so
        the case build's and `hybrid_initialize`'s "not arrays" growth is the op-by-op programs they
        compile. glibc's retention is separate and smaller once trimmed.
      - **Two parts, and only one grows with the case.** The serialized HLO of every live executable
        (`RESIDUE_HLO`; for bfs3d a fifth run on `605d2337` with only the HLO readings and the clear, whose
        clear freed 2.08, reproducing 2.09) totals 0.246 GiB on bfs3d against 0.055 on pitzDaily, and 16 /
        13 modules over 1 MiB hold 0.242 / 0.052 of it -- sizes only an embedded constant explains (a
        program taking its data as arguments is kilobytes). bfs3d against pitzDaily, MiB:
        `batched_field_jacobian_matvec` (one per column field) 6 × 17.9 against 5 × 3.3, `newton_step`
        3 × 19.8 against 3 × 5.8, `_march_step` 43.8 against 12.0, `norm_of_residual` 17.0 against 2.7,
        `_eddy_viscosity_drift` 14.5 against 1.7, `fit_simple_level` 4 × 1.4 against 3 × 1.2.
        - **Programs carrying arrays as constants** grow 4.5× for 1.9× the cells, and each program embeds
          its own copy (six batched-JVP programs). **What they are** (`validation/folded_constants.py`, the
          compiled coupled residual, `main` `50fe3be0`): XLA's constant folding of the unit tangents
          `BoundaryClosure.gradient_weight` builds as `jnp.zeros_like(gradient).at[:, k].set(1.0)` -- a
          scatter of constants, evaluated at compile time and stored. bfs3d 15.87 MiB of constants in
          the residual (nine `f64[23040, 3, 3]`, three `f64[23040, 3]`), pitzDaily 1.87 (four
          `f64[12225, 2, 2]`, two `f64[12225, 2]`): cells × dim², which with one more batched-JVP program
          is the 4.5×; not the stencil. The same tangent as `jnp.broadcast_to(unit_k, shape)` (not
          folded) leaves 0.05 / 0.01 MiB and a bit-identical residual on both cases. On a march their resident cost
          is small (next bullet).
        - **Measured on a march, the folded constants cost ~0.05 GiB resident, not the ~0.8 estimated**
          (bfs3d, `memory_residue.py` with `RESIDUE_CLEAR=end` and `RESIDUE_HLO`, 1 step, census off,
          tracer on, warm cache, 2026-10-25, one run per arm). `main` `39786723`: HLO 0.234 GiB, clear +
          trim freed 1.61 GiB, peak 6.25. With broadcast tangents (`3636ab49`): HLO 0.086, freed 1.56,
          peak 6.43 -- the peaks within the 0.4 GiB spread two runs of one configuration show; the clear's
          reading reproduced to 0.01 between two earlier runs. The estimate came from a standalone probe
          (four jitted functions closing over 32 MiB of int32 each: 432 MiB resident for 114 MiB of HLO);
          that multiplier does not transfer to the case.
        - **So the cache is not proportional to its programs' HLO**: ~1.5 GiB of bfs3d's is per-program
          runtime and compiler state over ~1300 executables. What doubled it from pitzDaily to bfs3d (1.06
          against 2.09 on `13fa17d4`, with 16 % more executables) is not separated, and no projection of it
          to 300k cells stands. `newton_step` still carries 3 × 16 MiB of HLO with broadcast tangents
          (`laplace_field` hands `eqx.filter_jit(newton_step)` closures, whose captured arrays compile in);
          its resident cost is unmeasured and, by the measurement above, small.
      - **After the march returns, its session and splits are cyclic garbage** until a collection: 1.6 GiB
        of arrays on bfs3d. During the march a collection frees nothing (entry of 2026-10-18), so this is
        not in the march's footprint; it is in the footprint of anything run after a solve in the same
        process (the adjoint, a second solve) until Python's collector reaches its next full pass.
      - Records: `memory_residue.<case>.json` and the run logs (git-ignored), copied into this entry. Two
        runs per mesh with the clear's live-array readings (the first pair lacked them); a third pitzDaily
        run (with the HLO ranking) reproduced the second (end 2.55, clear + trim freed 1.06).
      - ⚠️ **Measured before #817 and #819** (the cell-level pattern on the device and on the host, entries
        above) and before #818's trim by default. If the arrays compiled in as constants are dof-level
        pattern arrays, #817 shrank them. They are not: the constants are folded tangents (above),
        measured on `50fe3be0` (after #817--#819) by `folded_constants.py`; the cache totals are not
        re-taken there.

### bfs3d memory by owner after #817–#819 and broadcast tangents (2026-10-25)

      `memory_attribution.py` on `3636ab49` (`main` `39786723` + broadcast tangents, #824): 4-core Linux
      container, 15.7 GiB, no swap, glibc 2.39 defaults, jax 0.12.0.1, warm cache, shipped `case.yaml`
      (re-fit trim on by default since #818), `BFS3D_MEMORY_STEPS=1`, census on (tracer 0.31 GiB at the
      end). **Peak 6.35 GiB, set in the step program's first call** (stage peak; re-fit #1's leading
      block re-fit 5.94, the fit's block inverse build 5.21). Resident at step 1's end 5.38 GiB:

      | owner | GiB |
      |---|---|
      | live device arrays (int32 0.40, float32 0.66, float64 0.13) | 1.20 |
      | -- cell-level padded columns `_KroneckerColumns` (41 arrays) | 0.26 |
      | -- hierarchy level operators, `values` + stored `t_values` (9 levels) | 0.17 + 0.17 |
      | -- SIMPLE pieces (Schur, divergence, dg; each with its transpose) | 0.19 |
      | -- gradient `Corrections` 0.09, coupling 0.04 + 0.04, `KroneckerGalerkin` + `KroneckerGather` 0.07, probe `DeviceMatrix` 0.04 | 0.28 |
      | host traced by `tracemalloc` (Python objects, jaxprs; arrays ≥ 1 MiB only 0.02) | 0.59 |
      | `tracemalloc`'s own records | 0.31 |
      | compiled-program cache outside Python (freed by `clear_caches` + trim, less its traced 0.40; `memory_residue.py` run of the same commit) | ~1.16 |
      | glibc free-but-retained (`mallinfo2` 1.05; the step program's first call adds most of it) | ~1.05 |
      | file-backed (libraries) | 0.21 |
      | rest (interpreter, XLA runtime, thread stacks, arena fragmentation) | ~0.8 |

      - The peak is the step program's 1.02 GiB of temporaries (`memory_analysis`, beside 1.01 GiB of
        arguments that are the split's leaves and the state) on a 4.6 GiB baseline. Of that baseline,
        live arrays are 1.2; the cache, retention and tracer about 2.5.
      - Every peak number in this entry carries the tracer; a census-free run of the same commit peaked
        at 6.43 and `main` without the tangents at 6.25 (`memory_residue.py`, entry above), one run each.
        So bfs3d's 1-step peak on current `main` is ~6.3 ± 0.2 GiB, against 7.19 after #817 alone and
        8.07 on `7fe27d07`.

### What the march step's temporary holds: split against no preconditioner (2026-10-25)

      `validation/step_temporaries.py` (§14a's step half; `validation/hlo_text.py` reads the dumps,
      shared with `fusion_profile.py`) on `1bd8d292` (`main`): the case's own march with `_march_step`
      stubbed, so its first step's arguments are captured after the fit and the forced first re-fit and no
      step runs (state: the ramp's anchor station at the hybrid start; a `DualTimeStep`); the step compiled
      twice with the persistent cache off, (1) as built, (2) with its shift policy wrapped so the
      `ShiftTerm` keeps diagonal and row scale and `make_preconditioner` returns `None` (same shift, line
      search, dual-time loop, GMRES restart 15). Read per arm: `memory_analysis()`, XLA's buffer assignment
      and live ranges (`--xla_dump_to`); a temp slot (offset, size) is live while any of its values is, the
      peak is the schedule position with most temp bytes live, each slot named by the value live there and
      owned by the innermost `aquaflux` frame of its HLO metadata (rule in the docstring); "hoisted" =
      defined in fewer `while` loops than its `op_name` was traced in. 4-core Linux container, 15.7 GiB,
      jax 0.12.0.1, shipped `case.yaml`s (split `SimpleSmoothed` / `JacobiSmoothed`, float32, pivot floor
      0.15, `linear_solve` restart 15).

      **pitzDaily** (12 225 cells, 61 125 dof), MiB:

      | | split arm | identity arm |
      |---|---|---|
      | temp (`memory_analysis`) / peak live in it | 165.0 / 162.1 | 90.2 / 89.8 |
      | argument | 132.8 | 18.0 |
      | split: decoded padded columns, `s32[n_rows, width]` `add_select` (hoisted, 31 slots) | 55.2 | -- |
      | split: decoded columns not hoisted (9) + cell gathers of the decode `s32[n·w]` (2) | 6.2 + 5.8 | -- |
      | split: gathered `x[columns]`, `f32[2750625]` = 36675 × 75 (hoisted, 1) | 10.5 | -- |
      | residual JVP: hoisted linearization (357 slots) | 34.1 | 34.1 |
      | residual JVP: tangent working set | 23.9 | 29.2 |
      | copies without metadata (loop-carried values into the restart branch) | 13.4 | 13.9 |
      | hoisted state-independent residual pieces (traced in the dual-time loop's line search) | 10.3 | 10.3 |
      | Krylov vectors live at the peak (the 7.5 MiB basis is not live there) | 1.4 | 1.4 |

      - **The method check fails, for a found reason.** A standalone JVP is 42.9 MiB of temp (residual
        22.8, `jacobian_matvec` at the same state); the identity arm is 90.2, not ~52 (a JVP + the basis +
        a few vectors). The identity peak sits in the restart branch's true-residual JVP, and on top of
        that JVP's working set it holds 44 MiB XLA hoisted out of the loops for the whole step: the JVP's
        linearization at the fixed state (34.1) and state-independent pieces of the line search's residual
        (10.3), plus 13.9 of copies of loop-carried values. A standalone JVP frees those after one use.
      - **The split's share is 74.8 MiB (165.0 − 90.2), and its peak slots sum to 78.2**: 67 of it is int32
        decoded columns. Every padded operator's `x[rows.columns()]` (`padded_rows.py`,
        `_padded_product`) decodes the cell-level columns and passes them through the gather's index
        normalization (`select(c < 0, c + n, c)`); XLA hoists that out of the Krylov and V-cycle loops and
        holds an `s32[n_rows, width]` array per operator for the step, **twice**: the split is traced at
        two call sites (`M v` in the Arnoldi loop, `x = M V y` at the restart boundary) and each has its
        own copy (`s32[36675,75]` ×2, `s32[12225,85]` ×2, ...). The gathered values `x[columns]` are
        materialized too (one `f32[n_rows·width]` live at the peak, 10.5); the multiply and row sum are one
        kernel after it, so no `(n_rows, width)` product array exists.
      - Record: `validation/step_temporaries.pitzdaily.json` and `run-20261025-021615.log` (git-ignored).

## Condensed out of the rule (2026-10-17)

> Moved verbatim from the rule file on 2026-10-17 when it was condensed to current state; the
> rule keeps the live contract and a pointer here. Symbols named below may since have moved;
> check the source before treating an entry as current.

### The frozen host family — contract history (#281, #287)

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

### Saturation colouring and its bucket queue (2026-08-11, 2026-08-18)

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

### Chunked seeds and per-chunk de-compression (2026-08-11)

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

### Probe colouring reach 2 — pitzDaily and bfs3d (2026-10-14/15)

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

### The residual reaching past stencil_reach — the gradient-sweep trap (2026-08-16)

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


### Why the velocity columns reach further (2026-08-11)

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

### Materialize efficiency — batched probing and gather de-compression

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

### Seeds on the device, and the single-block refit on the device (2026-10-23)

Branch `claude/aquaflux-host-device-review-ef4ffs` (`6fc9948c`) against `main` `c58bfd1d`, 4-core Linux
container, jax 0.12.0.1, CPU, arms alternated.

- **pitzDaily (`case.yaml` as shipped; the field split, so only the device seeds apply;
  `validation/pitzdaily_openfoam/march_cost.py` under `run_case.sh`)**: same march on all four runs (21
  steps, 102 cycles, final R 5.736e-06, `x_r/h` 8.0686). Solve main 241.1, 243.9 s; branch 246.2,
  241.3 s. Refits summed main 31.5, 32.1 s; branch 32.0, 30.1 s. VmHWM main 3.106, 3.125 GiB; branch
  2.975, 3.097 GiB.
- **Laminar pitzDaily (`validation/pitzdaily_openfoam/laminar/case.yaml`, Re_h ~ 250, one
  `SimpleSmoothed` over the whole state; both changes apply).**
  - At `refresh_on_cycles` 3 the march never refits (no laminar solve reaches 3 cycles), so it exercises
    only the first fit: 21 steps / 104 cycles on all ten runs; `aquaflux run` wall main 146.8 (cold),
    130.0, 132.4, 137.6, 134.9 s against branch 134.5 (cold), 131.4, 132.4, 134.1, 133.3 s; peak resident
    (the child's `ru_maxrss`) main 2.133 (cold), 2.045, 2.051, 2.050, 2.013 GiB against branch 1.942
    (cold), 2.002, 1.967, 2.015, 2.003 GiB.
  - At `refresh_on_cycles` 1 (the shipped file; `refit_cost.py`, in-process): 24 steps and 24 refits on
    all four runs. Build and solve main 116.7 (first), 104.3 s; branch 100.5, 102.4 s. Refits summed
    main 25.7, 23.2 s (assemble 0.8, 0.3); branch 21.9, 22.5 s (assemble 0.0). Peak resident main 2.287,
    2.176 GiB; branch 2.052, 2.079 GiB.
- `aquaflux run`'s `history.csv` reported `refits` 0 on a laminar `FlowMarch` that did refit, because only
  `CoupledMarch` wired the session's refit observer (fixed by #809, `case.md`), so these runs were timed by
  `refit_cost.py`, which wraps the hook.
