---
paths:
  - "aquaflux/solve/amg_preconditioner.py"
  - "aquaflux/solve/multigrid.py"
  - "aquaflux/solve/native_inverse.py"
  - "aquaflux/solve/host_vcycle.py"
---

# Rules — `aquaflux/solve/` monolithic AMG and the JAX-native multigrid

> Split out of `solve.md` (2026-08-18) to keep routine `aquaflux/solve/` work from loading the full
> AMG/multigrid investigation narrative. See `solve.md` for the package-wide contracts, current
> configuration, and general binding decisions this file assumes.
>
> **⚠️ This file is already past this project's ~1,800-line outer bound (2,088 lines as of 2026-08-18)
> and has no `-log.md` sibling.** Split it on its next substantial edit: peel the dated/historical
> content into a new `solve-amg-multigrid-log.md` (no `paths:` frontmatter) and leave a current-status
> summary here, following the pattern in `solve-flow-block.md` / `solve-flow-block-log.md`. See
> `solve.md`'s "Where new content goes".

## Preconditioner — monolithic AMG (the coupled PC)

- **Monolithic ALGEBRAIC-MULTIGRID preconditioner — BUILT (`amg_preconditioner.py`), the coupled PC for
  large 3D.** The third member of the family: instead of factoring the assembled coupled Jacobian it applies
  **one smoothed-aggregation multigrid V-cycle** (`MonolithicAmgPreconditioner`, PETSc `PCGAMG`) whose only
  exact solve is a **direct LU on the small coarsest grid**, so the heavy fill never lives on the fine grid.
  This is the answer to both factorizations' 3D wall: the complete LU's fill OOMs (`O(n^{4/3})`), and the
  threshold-ILU's `spilu` is *prohibitively slow to build* on a distance-3 3D coupled Jacobian — measured on
  the 23k-cell `bfs3d` case the assembled Jacobian is **38.7M nnz (~280/row)** and a single `spilu` at
  `fill_factor=30` ran **>7.5 min and never finished** (RSS <2 GB — the 3D wall here is *time*, not memory,
  and the earlier "~11 min XLA compile" reading was a CPU-contention artifact; on idle CPU the probe compile
  is fast and `spilu` dominates). The V-cycle instead **builds in ~seconds with bounded memory** and scales.
  - **The V-cycle is a fixed LINEAR operator (one `pc.apply`, not an inner Krylov solve), so it is a
    drop-in for the same callback-matvec interface as the ILUT/LU and — being linear and transposable —
    serves the adjoint's transpose solve through the multigrid's own transpose (`pc.applyTranspose`), with
    no flexible outer Krylov.** It preconditions the **equilibrated, cell-major** matrix via the shared
    `equilibrate_cell_major` (**in `frozen_operator.py`** -- the one home for the sqrt-diagonal
    equilibration + cell-major reorder, moved there 2026-08-15; see the placement note below); the
    `Mat` block size is `n_fields` so GAMG aggregates cell-blocks. Host object, applied via `pure_callback`,
    riding as a static field; `build`/`refresh_in_place` its own and `matvec` inherited from the shared
    `HostPreconditioner`, so it plugs into
    `MonolithicFactorShiftPolicy` unchanged. **It is the one family member that needs `petsc4py`** (no
    pure-SciPy AMG fallback); the module lazily imports PETSc and raises a clear install hint otherwise.
  - **⚠️ SUPERSEDED BELOW — "ILU(0) stalls" is a HIGH-β measurement and does NOT hold at low β.** The
    original comparison was made near the OpenFOAM-converged state at a *large* shift, where the operator
    is diagonally dominant and extra fill is harmless. At the low shifts the march's tail actually runs
    at, the ranking **inverts**: ILU(1) breaks down and ILU(0) is the one that converges. See
    *"Zero-fill is the low-β smoother"* below, which is the current default-setting evidence. Read the
    two together: neither is wrong, they are measurements at opposite ends of the β range, and the march
    lives at the low end.
  - **The smoother is the research variable, and the first measured MVP config was a STATIONARY ILU(1) level
    smoother (`richardson`) + direct-LU coarse.** Measured on the `bfs3d` shifted coupled Jacobian
    (true 2-norm residual, `KSP_NORM_UNPRECONDITIONED`, at 2 sweeps): plain GMRES + stationary **ILU(1)**
    reached 1e-8, **ILU(0) stalled** and **SOR diverged** (measured, configuration not recorded — no shift,
    aggregation, coarse-eq limit or state, and both the smoother-fill and aggregation defaults have since
    moved; re-measure, and read it with the superseding low-β result below). A Krylov-accelerated (GMRES)
    smoother is a few iterations *faster* but makes the V-cycle **nonlinear** — it
    needs flexible GMRES and has no clean transpose, so it is a deferred forward-only optimization, **not** the
    adjoint path. ⚠️ **Name the forward solver's PATH — there are three and they differ.**
    `coupled_amg_continuation` builds its own inline: `forward_rtol = 0.3` in the **row-scaled**
    `coupled_scaled_norm`, `restart=15`, `max_restarts=60`. (`_COUPLED_FORWARD_SOLVER`, block-SIMPLE 2D:
    `relative_residual_gmres(1e-2)`, 2-norm, restart 120. `_COUPLED_ILUT_FORWARD_SOLVER`, 2D ILUT: 1e-2
    2-norm, restart 10.) There is no `_COUPLED_AMG_FORWARD_SOLVER` symbol.
  - **Per-step cost tuning (measured): `smoother_sweeps=2` default and the forward restart 15 (from 40).**
    The restart-15 forward loop stops as soon as the ~1% inexact-Newton tolerance is met instead of running
    out a 40-vector subspace (the dominant per-step saving). The **smoother-sweeps knob is the second lever,
    and more is better on this saddle**: the outer Krylov cost is governed by the *smoother work* per V-cycle,
    and adding a second incomplete-LU Richardson sweep — one extra cheap triangular back-solve — roughly
    quarters the outer iteration count on the low-shift operator the march's tail runs at, and is worth much
    less at a high shift where the operator is already diagonally dominant (measured on the `bfs3d` coupled
    Jacobian to a 1% stop — configuration not recorded: no β value, aggregation or coarse-eq limit, and it
    was tuned against ILU(1), which is no longer the validated smoother). Each outer iteration pays a full
    Jacobian-vector product (and, on the JAX-side `lineax` path, a `pure_callback` into PETSc), so trading one
    cheap extra sweep for far fewer outer iterations is a large net win — `sweeps=2` is the sweet spot
    (`sweeps=3` helps a little more at low shift but costs at high shift). Adding *fill* to the smoother
    (`smoother_fill_levels`) instead would cut iterations too, but it is the expensive incomplete-factorization
    build the ILUT hits in three dimensions; sweeps add smoother work without that build cost, and the
    coarsening choice (selective vs smoothed-aggregation) is a minor knob by comparison — but do not read that
    as covering `pc_gamg_agg_nsmooths`: plain-vs-smoothed *prolongator* smoothing is measured below as the
    largest preconditioner win found on this case. (The whole-march wall figure that used to sit here
    predated several march-wide wins and is deleted.) An **experimental, opt-in native-PETSc forward path**
    (`coupled_amg_continuation(native_forward_solve=True)`) is a far larger per-step lever — a native KSP
    whose shell matvec calls the eager JAX jvp (true Newton), 1 native GMRES iteration vs the JAX-side
    lineax path's ~90 on the identical system — but it currently under-converges the *march* (the lineax
    path over-solves each step to ~machine zero and the pseudo-transient globalization leans on those
    near-exact steps; the native honest-tolerance step descends slower and overruns the step budget), so it
    stays off by default. The follow-up to make it the default is a **β-tracking GAMG refresh** (the AMG
    analogue of `lu_beta_tracking_refresh`), so the frozen V-cycle matches the ramping β and the native
    solve stays accurate at low β.
  - **`coarse_eq_limit` — grow the coarsest-grid direct LU (BUILT).** GAMG's default coarsens to a tiny
    (~50-equation) coarse grid, whose direct LU captures only the crudest global mode; the indefinite
    saddle's wall is exactly that global pressure coupling. `build_amg_vcycle(coarse_eq_limit=K)` /
    `coupled_amg_continuation(coarse_eq_limit=K)` (default `None` = PETSc's ~50, byte-identical) stops
    coarsening at `K` equations so the coarse LU inverts more of the global coupling **exactly** — a
    stronger V-cycle *and* transpose V-cycle (so the adjoint benefits). Measured on the `bfs3d` hard state
    (honest right-PC cycles to 1e-6): **baseline ~50 → 652 cycles at β=0.5, `K=2000` → 24 (~27×)**, and it
    **saturates by ~2000** (`K=8000` identical), at negligible extra build cost (the 2000-eq coarse LU is
    trivial against the materialize). The `bfs3d` combo-sweep. *(A `cycle_type` V/W knob was tried and
    **dropped**: the W-cycle came back byte-identical to V — GAMG did not honour `pc_mg_cycle_type` here —
    and `coarse_eq_limit` dominates regardless.)*
  - **Zero-fill is the low-β smoother: ILU(0), 4 sweeps, `coarse_eq_limit=2000`, PC-only `beta_floor`
    (the validated bundle).** The shifted operator is `J + β d`; as the march's shift falls the diagonal
    weakens and the factorization, not the coarse space, is what fails first. Measured on the `bfs3d`
    coupled Jacobian at **adjoint-grade rtol 1e-8 on the TRUE residual**:

    | β | ILU(1) (was default) | ILU(0) |
    |---|---|---|
    | 0.10 | converges | 30 its |
    | 0.05 | converges | 51 its |
    | 0.02 | **DIVERGES** (534 its, true rel. 3.7) | 97 its |

    **Ground truth, not inference:** a pivot census of both factorizations at β=0.02 found **303 negative
    pivots in ILU(1)** (min 6.2e-4) and **zero** in ILU(0) (min pivot 0.29–0.36 at every β tested). The
    *fill* is what destroys the factorization — dropping it is not an approximation here, it is the fix.
    The worst pivots sit in **velocity rows, not pressure rows**, which is why "the pressure block is the
    problem" intuitions kept failing. ILU(0) is also **3–4× cheaper to build**. This is consistent with the
    literature: every saddle-point-AMG ILU smoother in the published work we surveyed is **zero-fill**
    (ILUC0 / DILU); the ILU(1) default was ours alone.

    The bundle members were each measured alone and then together on the same state:
    - `smoother_fill_levels=0` — the table above.
    - `smoother_sweeps=4` — ILU(0) is a *weaker* smoother than ILU(1), so extra sweeps pay more than they
      did for ILU(1): 390 → 69 its at β=0.01. (This is why the `sweeps=2` sweet spot recorded above does
      not carry over: it was tuned against ILU(1).)
    - `coarse_eq_limit=2000` — `None` **stalls at every low β** (552 its, true rel. 1.3–67). Not optional.
    - **PC-only `beta_floor=0.05`** — the V-cycle is built at `max(β, 0.05)` while the march still solves
      at its own β: 43/69/95 → 29/45/47 its at β = 0.02/0.01/0.005. The *operator* is untouched, so the
      converged root and the adjoint are unchanged and the mismatch saturates at `beta_floor·d` rather than
      growing as β → 0. Flooring the k/ω rows *alone* hurt; flooring all rows was best.

    **`alpha_u` and a velocity-row `beta_floor` are the same knob.** A velocity under-relaxation `α_u=0.95`
    adds `(1−α_u)/α_u = 0.0526` of the diagonal — a β-equivalent of ~0.05 applied to the velocity rows only.
    Worth knowing before adding a second spelling of it: prefer the floor, which is explicit about being
    preconditioner-only.

    **⚠️⚠️ THE FILL RANKING INVERTS BETWEEN CASES — `bfs3d` WANTS ZERO FILL AND `pitzDaily` WANTS ONE,
    AND COPYING THE VALUE ACROSS STOPS THE OTHER CASE DEAD (measured 2026-08-16).** Everything in the
    bundle above was measured on `bfs3d`, and this one does not transfer. On the 2D `pitzDaily` leading
    `[u, v, p]` block, at the cold seed the march fails on, right-preconditioned GMRES on the TRUE
    residual, matrix exact to 1.5e-15:

    | β | ILU(0) ×4 (the `bfs3d` value) | ILU(1) ×4 |
    |---|---|---|
    | 2.0 | one-apply 1.36e+28 → 300 matvecs, true **3.50** | one-apply 7.4e-02 → **1 matvec** to the 0.3 stop, 10 to 1.7e-09 |
    | 0.5 | one-apply 1.14e+14 → true **0.984** | **1 matvec** |

    **At zero fill the level sweep is not a contraction on that block — it AMPLIFIES, and the sweeps
    compound it: one apply reads 9.88e+05 at one sweep and 1.56e+31 at four.** Read "more smoothing
    makes it worse" as the signature; a merely weak smoother improves with sweeps.

    **The discriminator is a pivot census, and it inverts.** `pitzDaily`'s ILU(0) carries **negative
    pivots at every shift** (27/25/9 of 36675 at β 2/0.5/0), min |pivot| 5.9e-03 — about twenty times
    smaller than `bfs3d`'s, whose ILU(0) has **zero** negatives at any shift (min |pivot| 1.2e-01 …
    1.9e-01, median 1.02, max factor entry 8.0). So on one case the FILL produces the bad pivots and
    dropping it is the fix; on the other, dropping it produces them and the fill is the fix.

    **✅ AND THE SIMPLE-SMOOTHED HIERARCHY ESCAPES BOTH THE FILL AND THE REACH — the fastest arm on
    `pitzDaily`, at HALF the Jacobian (2026-08-16).** Same case, same everything but the leading
    inverse and the probe's reach; three Reynolds rungs; `x_r/h` identical to four decimals in every
    converging arm, so this is a cost comparison and not an accuracy one:

    | leading inverse | reach 3 | reach 5 |
    |---|---|---|
    | PETSc **ILU(1)** | **fails** (300 matvecs, true 3.36) | converges — 74 steps, 321 cycles, **628 s** |
    | PETSc **ILU(0)** | fails | fails (α 0.000, NaN by step 4) |
    | our **ILU(0)** (`hostilu`) | fails | fails (α 0.000, inf by step 3) |
    | **native SIMPLE** | **converges — 71 steps, 395 cycles, 550 s** | converges — 71, 408, 799 s |

    **SIMPLE is REACH-INSENSITIVE, and that is the load-bearing observation.** Between reach 3 and
    reach 5 it takes the **same 71 steps**, cycles within 3 % (395 against 408) and a final residual
    identical to four figures (5.095e-06 against 5.097e-06) — while the Jacobian halves (6.3M nonzeros
    against 13.3M) and the march is **31 % shorter**. It simply absorbs the ~2e-07 perturbation that a
    short reach leaves in the pressure column.

    **The mechanism is which preconditioners inherit the stored SPARSITY.** An incomplete
    factorization takes its pattern from it, so a corrupted pattern gives a corrupted factor and
    ILU(1) *requires* reach 5 here. SIMPLE relaxes through diagonal and Schur approximations and takes
    no pattern at all, so the same corruption is merely a slightly wrong operator. **Shortening the
    reach is free for SIMPLE and fatal for an ILU** — which is the argument for the reach as a
    preconditioner-family choice rather than a case constant.

    At reach 3 SIMPLE also beats the incumbent outright, 550 s against 628 s (12 %).

    ⚠️ **One run per arm, and this case has no measured march-level noise floor.** The step counts and
    cycle counts are deterministic and carry the result; the 12 % wall margin over PETSc is the softer
    number. ⚠️ **And SIMPLE's own settings are still the sibling case's** (`strength_threshold` 0.25,
    2 sweeps, 5 levels) on a mesh that coarsens about 3× per level where that one manages 24×, so the
    arm is not at its best here — `PITZ_FLOW_SWEEPS` and the threshold are unexplored.

    **✅ CONFIRMED ON A SECOND, INDEPENDENT ZERO-FILL IMPLEMENTATION (2026-08-16).**
    `HostVCycleInverse` is this package's own hierarchy smoothed by its own `Ilu0`: different
    coarsening, different factorization code, different language even. Run as the leading inverse on
    `pitzDaily` it fails identically to PETSc's zero-fill smoother — step 1 alpha 0.000 with the
    residual above its own starting value, step 2 at the shift ceiling, **non-finite by step 3** —
    while PETSc's ILU(1) converges the same case in 74 steps and 628 s.

    **🛑 BUT ITS CONCLUSION — "the mechanism is the FILL" — WAS WRONG, AND THE ERROR IS INSTRUCTIVE
    (corrected 2026-08-17).** The two implementations were NOT "differing only in fill". They also
    **shared the elimination ordering**: at the time both took it from `equilibrate_cell_major` --
    cell-major over the mesh's own cell order. (The host V-cycle now takes an injected ordering through
    `equilibrate_ordered` and defaults to that same one, so the confound is a choice rather than a
    given.) Fill and order were confounded, and the confound is the one that
    mattered — **on this block the ordering is the larger lever, and the shipped cell order is the
    thing that fails.** Two independent implementations agreeing is evidence about a *shared* cause;
    it does not identify which shared thing is the cause, and here the argument named the wrong one.
    See *"Ordering, not fill, is what fails zero-fill on `pitzDaily`"* below for the measurements.

    **⚠️ THEREFORE THE CONSEQUENCE DRAWN FROM IT IS ALSO WITHDRAWN.** It said `Ilu0` is zero-fill by
    construction with no fill parameter, so `HostVCycleInverse` "cannot serve a case that needs one" —
    and offered a level-of-fill factorization as the specifiable gap. `pitzDaily` is not shown to need
    fill. It is shown to need a different cell order, which the host V-cycle now takes
    (`aquaflux/solve/ordering.py`). A level-of-fill `Ilu0` may still be wanted some day; this case is
    no longer the evidence for it.

    **⚠️ Three mechanisms were refuted on the way, each of which had a plausible story:**
    - **NOT the pressure/momentum scale split.** The 3D block's split is *comparable or worse* at
      matched β (pressure min |diag| 3.42e-07 against 2D's 4.19e-05; equilibrated max entry 81.9 at
      β = 0 against 24.3) and it converges. Decisively: an exact LU of the *same* equilibrated 2D matrix
      solves it in 1 matvec to 9.9e-15 while ILU(0) of it diverges — two factorizations of one matrix
      ranking oppositely cannot be a scaling effect.
    - **NOT the field split.** The monolithic five-field V-cycle fails too (one-apply 1.94e+12).
    - **NOT the coarse space.** `coarse_eq_limit` 2000 (2 levels, 402 coarse equations) and `None`
      (4 levels, 27) give **identical** one-apply 1.557e+31.
    - **NOT a `spilu`-style broken factor.** The 1e+38 resembles the recorded threshold-ILU blow-up but
      is not one: max |ILU(0) factor entry| is 110–338, not 1e+23. It is stationary-sweep amplification.

    **⚠️ AND THE REACH AND THE FILL MUST BE VARIED TOGETHER.** Neither alone helps on `pitzDaily`:
    reach 3 + fill 1 fails (300 matvecs, true 3.36), reach 5 + fill 0 fails (true 3.50), reach 5 +
    fill 1 takes ONE matvec. A one-variable sweep measured reach 5 as "step-for-step identical, 35 %
    dearer, buys nothing" and it was reverted on that basis — a correct measurement of the wrong pair.

    **✅ RESOLVED (2026-08-17) — the β = 0 failure was the STATE, not the preconditioner, and the
    adjoint is safe on both cases.** An earlier entry here reported that both fills fail at zero shift on
    `pitzDaily` and left it open; it recorded no state, and it was measured at the cold self-start.
    Re-measured at the **converged root** (`state-00071`, reach 5, fill 1, `petsc` leading inverse, true
    residual through GMRES at rtol 1e-8), the shipped field split **converges**:

    | case / arm | forward `M` | transpose `Mᵀ` |
    |---|---|---|
    | `pitzDaily` field split @ β = 0 | 326 applies, **2.34e-09** | 299, **9.83e-09** |
    | `pitzDaily` field split @ floor 0.05 | 205 applies, **7.39e-09** | 213, **6.53e-09** |
    | `bfs3d` field split @ β = 0 (`petsc` leading) | 116 applies, **6.07e-09** | 117, **2.67e-09** |
    | `bfs3d` field split @ floor 0.05 | 105 applies, **6.54e-09** | 104, **6.72e-09** |

    On both cases the **floored** preconditioner is cheaper than the one built at β = 0, so the shipped
    pairing is not merely adequate at zero shift, it is the better one.

    **⚠️ The trap this cost, which is the part worth keeping: a zero-shift measurement is meaningless
    without its state.** At `pitzDaily`'s cold self-start the same field split *diverges* to 100–800×
    the right-hand side — and that is not a preconditioner property, because the cold Jacobian is nearly
    singular there (smallest pivot `1.3e-12` against a matrix 1-norm of `278`) and **a complete LU of it
    is not an accurate inverse either** (one apply `7.8e-04`). The control is monotone: as β goes
    2 → 0.05 → 0, one-apply accuracy degrades `7.4e-15 → 7.8e-11 → 7.8e-04`. So the pseudo-transient
    shift is not only globalization — it is what makes this Jacobian factorizable at all. Harnesses:
    `validation/pitzdaily_openfoam/zero_shift_arms.py`, `validation/bfs3d_openfoam/zero_shift_adjoint.py`.

    **Consequence: nothing selects the monolithic ILUT or complete LU for the adjoint**, which was the
    last regime either had. Both are already dominated on the forward march, the LU does not fit in 3D,
    and no validation case selects the ILUT. That closes the case for deleting them.

    **What a deletion is giving up, measured, so the decision is not made on a guess (2026-08-17).** The
    complete LU *does* work on `pitzDaily` and needs no PETSc — SciPy SuperLU alone, at the
    `hybrid_initialize` state, 61125 dofs:

    | probe | matrix nnz | materialize (**every arm pays**) | factor (**the LU's own**) | factor L+U | peak RSS | ‖Ax−b‖/‖b‖ |
    |---|---|---|---|---|---|---|
    | reach 5, full sweeps | 13.32 M | 6.0 s | **72.7 s** | 174.0 M | 3.60 GB | 1.35e-09 |
    | reach 3, `sweeps=1` | 4.72 M | 2.8 s | **17.9 s** | 89.8 M | 3.14 GB | 4.96e-10 |

    So it is **86–92 % factorization** — essentially all cost the field split does not pay — and 4.5 GB
    peak for a 12 k-cell 2D mesh, which is why it stays 2D-only whatever else is true. ⚠️ The often-cited
    "UMFPACK is ~26× faster than the threshold-ILU" is a **PETSc-only** figure; on SuperLU the record
    already says it is no faster to factor than the ILUT, and these timings agree. Eliminating PETSc
    therefore *weakens* the complete LU's case rather than strengthening it.

    **⚠️ Narrowing the probe's gradient sweeps helps a COMPLETE factorization and hurts a ZERO-FILL one —
    opposite signs, same knob.** At `probe_gradient_sweeps=1` the LU's factor cost falls 4.1× (72.7 → 17.9 s)
    on an exactly-probed matrix, while an ILU(0) pivot census on the equilibrated leading block goes
    **9 → 34 → 263** negative pivots at full / 2 / 1 sweeps, minimum |pivot| falling `2.18e-02 → 3.59e-04`.
    ⚠️ And measured against the **true** Jacobian rather than the narrowed one, `sweeps=1` at reach 3 is
    `1.07e-03` away where the *aliased* full-sweep probe at the same reach is only `1.99e-07` — so
    narrowing trades a small structured corruption for a larger smooth approximation on this mesh. It is a
    preconditioner-family-dependent knob, not a free win: right where sweeps are inert (`bfs3d`, skew-free),
    questionable where they are live (`pitzDaily`).

    **Naming trap for whoever does the deletion: `_COUPLED_ILUT_FORWARD_SOLVER` is the default forward
    solver on the COMPLETE-LU path too**, not only the ILUT's — `_monolithic_factor_step` falls back to it
    for both. Deleting the ILUT without renaming it leaves a solver named after a preconditioner that no
    longer exists.

### ⭐ Ordering, not fill, is what fails zero-fill on `pitzDaily` (2026-08-17)

**The correctly-scoped question.** Everything above was measured MONOLITHICALLY — one V-cycle over all
five fields — and the case is FIELD SPLIT. The split sends `[u, v, p]` to the V-cycle whose level
smoother is the incomplete factorization (the only block a fill level governs) and `[k, omega]` to
`native_nodal_inverse`, which is not a factorization at all, **so no `k` or `omega` row is ever
eliminated by an ILU in the shipped solver.** Everything in this section is on `[u, v, p]` alone, taken
from the assembled operator by the same `FieldGroups` the split uses.

This **agrees with** the zero-shift resolution immediately above rather than competing with it. That one
found the β = 0 failure was the *state* — at the converged root the shipped split converges. This one
finds that on the *split block* β = 0 converges under most orderings **even at the cold self-start**, the
shipped order included. Both point the same way: the recorded "both fills fail at β = 0" was monolithic,
and neither the adjoint's operator nor the split's block is the thing that was failing.

*Configuration (all arms):* `pitzDaily`, 12225 cells; flow block 36675 of 61125 dofs, **reach 5** (the
exact one on this mesh), block nnz 6.44 M; symmetric sqrt-diagonal equilibration ON; `Ilu0` **zero
fill**, `COMPILED=True`; real right-hand side `-R(state)[leading]`; operator and factorization at the
**same** β (no preconditioner-only floor). Judged on the **TRUE** relative residual from
right-preconditioned GMRES, rtol 1e-8, restart 30, ≤ 20 restarts (so 621 applies = hit the cap).
Two states: the case's **self-start** (`|R|` 2.89e+02 — where the `hostilu` march dies at step 1 *under
the shipped cell order*) and
the **converged root** (`R` 5.095e-06; marched with the *native SIMPLE* leading inverse at reach 3,
which is reach-insensitive — the state is a root of the exact residual either way, and it is re-probed
here at reach 5 because an ILU inherits the stored pattern).
Harness: `validation/pitzdaily_openfoam/flow_block_ordering.py`.

⚠️ **The converged root is a GITIGNORED run artifact, not a checked-in fixture.** It was
`validation/pitzdaily_openfoam/checkpoints/state-00071.npz`, written by `StateCheckpointer` during a
full march; `checkpoints/` is ignored, so that file is not in the repository and will not be in a fresh
clone. To re-adjudicate the converged-root half of this table, **re-run the case to regenerate it** —
`PITZ_FLOW_INVERSE=native validation/run_case.sh validation/pitzdaily_openfoam/compare.py`, about nine
minutes — and the harness picks it up automatically. Absent one it prints a line saying so and measures
the self-start only, so a run missing the artifact reports a *smaller* table rather than a wrong one.

GMRES applications; **FAIL** = stalled above 1e-6 true:

| cell ordering | self-start β 0.5 | β 0.05 | β 0 | root β 0.5 | β 0.05 | β 0 |
|---|---|---|---|---|---|---|
| `cell_major` — **SHIPPED** | **FAIL** | 55 | 125 | **FAIL** | 80 | 404 |
| `cell_major_rowlength` | **113** | 152 | 140 | **121** | 58 | 275 |
| `cell_major_rcm` | 140 | **32** | **55** | **FAIL** | **51** | **66** |
| `pointwise_rowlength` | 113 | 152 | 140 | 121 | 58 | 275 |
| `pointwise_rcm` | 203 | 32 | 52 | **FAIL** | 51 | 63 |
| `mc64_symmetric` (permutation only) | **FAIL** | 58 | 107 | **FAIL** | 77 | 339 |
| `defer_small_diagonal` 1 % / 5 % | **FAIL** | 61 / 93 | 125 / 152 | **FAIL** | 89 / 164 | 308 / 435 |
| `cell_major_reversed` — **CONTROL** | **FAIL** | 60 | 104 | **FAIL** | 66 | 362 |
| `field_major` | 460 | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** |
| `pressure_last` | 586 | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** |
| `pressure_first` | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** |

- **The shipped cell order is the failure, and it fails exactly where the march starts.** At β = 0.5 —
  the march's own `beta0` — the shipped order **amplifies from the FIRST stationary sweep** (true
  residual ×5.53 after one, ×2.17e+03 after two, ×5.97e+08 after four) and stalls GMRES. Two other
  cell orders converge on the identical matrix. That is a one-variable change: same block, same
  equilibration, same zero fill, same factorization code.
- **`cell_major_reversed` is the control that makes this a result.** Merely relabelling the cells the
  other way round fails just like the shipped order, so the wins are not "any permutation shakes it
  loose" — RCM and row-length are doing something specific.
- **Neither winner spans everything, and the failing corners differ.** Row-length is the only ordering
  that converges at all six points. RCM is much the better where it works (2–6× fewer applies at every
  small-β point) and fails only at **converged root + β = 0.5** — a corner no march and no adjoint
  visits, since β is small by the time the state is converged. The two (state, β) pairs that actually
  occur are hot-state/large-β and converged-state/small-β, and RCM handles both.
- **⚠️ The stationary-sweep contraction does NOT predict the Krylov verdict — a FOURTH quantity that
  fails to rank these arms**, after `condest`, diagonal dominance and the pivot census. It fails in
  *both* directions: at root β = 0.5 the shipped order's sweeps **contract** (2.65e-2, 2.29e-2, 5.74e-2)
  and GMRES still stalls at 3.25e-4; at self-start β = 0.5 RCM's sweeps **amplify** (2.7e-1 → 1.18e+01
  → 2.67e+04) and GMRES converges in 140. Report both, rank on neither, and settle it on a march.
- **⚠️ THE FIRST CENSUS WAS AN ARTIFACT — the harness read the diagonal of the equilibrated *operator*
  rather than the *factor*, and the symmetric square-root equilibration forces that to magnitude
  exactly 1.** It reported "zero negatives, min |pivot| 1.00" for all twelve orderings at all six
  points, including arms that diverge by 1e+59. Fixed by exposing `Ilu0.pivots` (the stored diagonal
  *is* the pivot there — unlike PETSc, which stores its reciprocal); re-runnable with
  `FLOW_BLOCK_CENSUS_ONLY=1`. The verdicts above never touched it.

- **⭐ AND THE REAL CENSUS IS A FIFTH FAILED PREDICTOR — it reproduces `condest`'s exact failure mode:
  THE SIGN OF THE CORRELATION REVERSES WITH THE SHIFT.** Negative pivots of the ILU(0) factor, at the
  self-start, against the verdicts in the table above (same state, same matrix, so this is a fair join):

  | ordering | β 0.5 verdict / neg | β 0.05 verdict / neg | β 0 verdict / neg |
  |---|---|---|---|
  | `cell_major` | **FAIL** / 22 | 55 / 3 | 125 / 9 |
  | `cell_major_rcm` | 140 / 12 | **32** / 3 | **55** / 1 |
  | `cell_major_rowlength` | **113** / 1 | 152 / **11** | 140 / 5 |
  | `cell_major_reversed` | **FAIL** / 20 | 60 / 9 | 104 / 2 |
  | `field_major` | 460 / **0** | **FAIL** / 1 | **FAIL** / 1 |
  | `pressure_last` | 586 / **0** | **FAIL** / 4 | **FAIL** / 13 |
  | `pressure_first` | **FAIL** / 5815 | **FAIL** / 7292 | **FAIL** / 7138 |

  At **β = 0.5** it looks predictive: every failing arm carries 20+ negatives and every converging one
  carries 12 or fewer. At **β = 0.05 and 0 it inverts** — `field_major` fails with **one** negative
  pivot and a comfortable min |pivot| of 2e-01, while `cell_major_rowlength` converges with **eleven**.
  A quantity whose correlation changes sign between operating points cannot select an arm, which is
  precisely what was already recorded for `condest`.

  **So: `condest`, diagonal dominance, stationary-sweep contraction, and now the pivot census have each
  failed to rank these arms. Nothing cheap computable at BUILD time has yet predicted the verdict on
  this operator class** — which is the standing argument for settling an ordering question on a
  true-residual solve rather than on a factor inspection.

  Two consistency checks fall out of it: `defer_small_diagonal` at 1 % and 5 % has a census **identical
  to `cell_major`** at every point, and `mc64_symmetric` is identical to it at β = 0.5 — corroborating
  from the factor's side that both are near-no-ops here, which their verdicts already implied.

  ⚠️ *Census configuration:* the root half was taken at `state-00082.npz`, a **different** converged
  root from the `state-00071.npz` the verdict table used (the case checkpoints on a rolling keep-3 and
  the later marches replaced it). Both are converged to ~5e-06, but they are not the same matrix, so
  only the self-start columns are joined above. The harness now takes the **latest** checkpoint rather
  than a hard-coded name, which is what made this visible instead of silent.


**⚠️ The saddle-point literature's ordering is WRONG for zero fill, and the reason is structural.**
`pressure_last` — velocities first, pressure last, the Konshin/Olshanskii/Vassilevski direction — fails
at five of six points, and `pressure_first` is catastrophic everywhere (a sweep growing by 1e+59 in one
application). This does **not** contradict that literature: what makes pressure-last work is that
eliminating the velocities **fills** the pressure block with the Schur complement, and a zero-fill
factorization discards precisely that fill, leaving the pressure block to be eliminated against its own
bare, near-singular Rhie–Chow diagonal. Pressure-last is an ordering for a factorization that KEEPS
fill. Do not port it to `Ilu0`; if a level-of-fill `Ilu0` is ever built, re-ask it there.

**Two well-motivated leads measured out, so they need not be re-tried:**
- **HILUCSI static deferring** (Chen, Ghai & Jiao, arXiv:1911.10139 — symmetrically permute the
  smallest-diagonal rows to the lower-right, criterion taken PRE-equilibration since the equilibration
  forces every nonzero diagonal to magnitude 1): never better than the shipped order at any point, and
  clearly worse at 5 %. It does not rescue β = 0.5.
- **MC64 symmetrized, PERMUTATION ONLY** (`min_weight_full_bipartite_matching` on `−log|a_ij|`, applied
  as `P_r = P_c`): essentially a no-op — bit-identical sweep numbers to the shipped order at β = 0.5,
  because the diagonal is already the matched entry in nearly every row. It also costs 70–150 s to
  build. ⚠️ This is not a test of MC64: the method's other half is the pair of dual potentials that
  rescale the matched matrix, and `scipy` does not expose them.

**✅ AND IT CARRIES A REYNOLDS RUNG ON THE REAL MARCH — the block-level result is not an artifact of
measuring a block.** `PITZ_FLOW_INVERSE=hostilu PITZ_FLOW_ORDER=rcm`, reach 5, `beta_start` 0.5, fill/
sweeps/coarse 1/4/2000 (the `hostilu` path ignores the fill), field split on, trailing `native_nodal`,
`Ilu0` **compiled**, 3 Reynolds points, log `run-20260817-101242.log`:

| | shipped `natural` order | `rcm` order |
|---|---|---|
| step 1 | α = 0.000, residual ABOVE its start | α = **1.000**, R 5.69e-02 |
| steps 1–4 | β at the ceiling by 2, **non-finite by step 3** | full steps, β relaxing 0.5 → 0.148 |
| rung 1 (Re/100) | never reached | **converged, 28 steps, 270 s, R 7.25e-06** |

So the ordering turns a march that dies at step 3 into one that completes an entire continuation rung at
full Newton steps. **That is the headline, and it is the answer to "can a zero-fill smoother work on this
case": yes.**

**🛑 BUT IT DOES NOT CARRY THE WHOLE CASE, AND WHERE IT STOPS IS THE INFORMATIVE PART.** Rung 2 (Re/10)
retried at step 29 (β → 1.0, recovered), then step 30 clipped to α = 0.000, step 31 diverged twice with
the ladder escalating β → 8 → 16 and reported `inf`, and steps 31–32 ground with α ≈ 0. **At the point of
failure the linear solves were healthy — 12 cycles each — while α collapsed.** A preconditioner that
returns a 12-cycle solve is not the thing failing; the line search is. Run stopped by hand at step 32.

**🛑 AND THE MECHANISM IS NAMED IN THE LOG: THE COST THRESHOLD IS POSITIVE FEEDBACK ON THIS CASE.**
(Spelled `retry.on_cycles` at the time; there is no such field now — it is `retry.abort_above_cycles`
and it no longer escalates, which is what this section led to.) The
retry ladder escalates β when a solve costs more than `on_cycles` restart cycles, on the usual reasoning
that a stiffer pseudo-timestep makes the block easier. **On this case that implication is false** — under
`rcm` the block takes 140 applications at β = 0.5, 32 at 0.05 and 55 at 0 — so the rule closes a loop
that runs the wrong way: *more cycles → higher β → a harder block → more cycles.* The log gives each
redo's reason, and the first three are all `cycles`, not divergence:

```
step 29 attempt 2: cycles,   beta -> 1.0000
step 30 attempt 2: cycles,   beta -> 1.3333
step 30 attempt 3: cycles,   beta -> 2.6667
step 31 attempt 2: diverged, beta -> 8.0000
step 31 attempt 3: diverged, beta -> 16.0000
```

The divergence is the *consequence* of three cycle-triggered escalations, not the cause. And the trigger
is a **mis-calibration rather than a fault**: `on_cycles = 10` was set for the PETSc ILU(1) arm, while
this arm's healthy cost at rung 2 is about 12 — so the ladder fires on an arm that is working. **Read a
`cycles` escalation as a statement about the threshold before reading it as one about the step.**

**✅ AND RAISING THE THRESHOLD CARRIES THE WHOLE CASE — the zero-fill host smoother now marches
`pitzDaily` end to end (2026-08-17).** One variable changed from the run above, `PITZ_RETRY_ON_CYCLES`
10 → 25, everything else identical (`hostilu`, `rcm`, reach 5, `beta_start` 0.5, `Ilu0` compiled;
log `run-20260817-111002.log`). The 12-cycle solve at step 29 that previously triggered the first
escalation is simply accepted, and the march never looks back:

| step | threshold **10** | threshold **25** |
|---|---|---|
| 29 | β **1.0** (escalated on `cycles`), R 5.53e-02 | β **0.50**, R 4.55e-02 |
| 30 | β **2.67**, α **0.000**, R **rising** 1.00e-01 | β **0.33**, α 1.000, R 2.96e-02 |
| 31 | β **16.0**, R **inf** | β **0.22**, α 1.000, R 2.19e-02 |
| … | dead | converges, 3 rungs |

**Result: 82 steps, 466 cycles, 743 s, `x_r/h` = 8.0686 against OpenFOAM's 7.7409.** Seven retries, every
one on `alpha` and **none on `cycles`** — with the spurious trigger gone, only genuine line-search retries
remain, and their β escalations stay inside the 0.04–0.16 band this block finds easy.

**The root is the same one.** `x_r/h` 8.0686 matches the native-SIMPLE arm to four decimals
(`nut_peak` 417.54 against 417.51), so this is a cost and robustness result, not an accuracy one — two
very different preconditioners landing on one converged state.

⚠️ **It is the SLOWEST of the three working arms, and it is not a recommendation to change a default:**

| arm on `pitzDaily` | steps | cycles | wall |
|---|---|---|---|
| native SIMPLE, reach 3 | 71 | 395 | **550 s** |
| PETSc ILU(1), reach 5 | 74 | 321 | 628 s |
| **`hostilu` + `rcm`, reach 5, threshold 25** | 82 | 466 | 743 s |

⚠️ **And the three are NOT a controlled comparison.** This arm ran the cost threshold at 25 where the other
two ran 10, and the SIMPLE arm ran reach 3 where the other two ran reach 5 (it is reach-insensitive; an
ILU is not). One run each, and this case has no measured march-level noise floor. What the row
establishes is **that a zero-fill host smoother now completes this case at all**, which it could not
before at any setting — not where it places on cost. A matched re-run of all three at one threshold is
the obvious follow-up, and it is untried.

⚠️ **This is also the first attributable `hostilu` wall time on this case**, because it is the first run
recorded as having the compiled kernel live (see the kernel-provenance warning elsewhere in this file).

This also re-frames the independently recorded `beta_start = 4` run being **worse** than
`beta_start = 0.5`: consistent with β being the wrong direction on this case rather than with anything
about the preconditioner. `PITZ_BETA_START` remains untried in the low direction.

### ✅ RESOLVED — the cost guard and the shift escalation are now separate responses (2026-08-17)

**Binding, and it changed shipped behaviour on both cases.** `RetryPolicy.on_cycles` was one number
doing two jobs: it stopped the dual-time inner loop *and* escalated β. It is now
**`abort_above_cycles`**, a cost guard only. `retry_reason` returns `"diverged" | "alpha" | "cycles"`,
and `ESCALATING_REASONS = {"diverged", "alpha"}` — one home, read by both the march and the log — says
which of them raise the shift. A `"cycles"` retry **redoes the step at the shift it already had**, on
the factorization the dual-time loop's mid-step refresh has by then rebuilt.

- **Why cost must not escalate.** A cycle count says the frozen preconditioner is struggling with this
  operator, not that the step is stiff. The march already has the right response — `refresh_on_cycles`
  rebuilds it mid-step — and escalating on the same observation additionally assumes a stiffer operator
  is an easier one, which is false on this case (140 applications at β 0.5 against 32 at 0.05).
  **Stiffness is what `on_alpha` measures, and a step length is dimensionless**, so it needs no per-arm
  calibration. The abort threshold remains a number, but it is now a *resource* decision rather than a
  diagnosis.
- **⚠️ The redo is capped at ONE per step** (`redone_on_fresh` in `march.py`). It differs from the first
  attempt only in starting on the refreshed factorization, so a second at the same shift would repeat
  it exactly — burning the escalation budget on identical attempts and spinning to `cycles_limit` every
  step.
- **⚠️ `cycle_budget` DEPENDS on this redo and would otherwise silently accept a bad iterate.** It
  deliberately truncates a grinding solve and returns a partial, non-converged iterate *expecting it to
  be discarded*. Pair it with `abort_above_cycles < cycle_budget` or the truncation is simply accepted.
  That pairing was the reason the two jobs were fused in the first place, and it is what made "just
  delete the cost trigger" wrong.
- **`escalates` now means `on_alpha is not None`**, so a cost-only policy needs no readable β leaf and
  `require_shifted` no longer rejects one.

**✅ `bfs3d` RE-MARCHED UNDER THE SPLIT, AND IT IS UNCHANGED TO EVERY RECORDED FIGURE (2026-08-17).**
Shipped defaults, `hostilu`, `Ilu0` compiled, `abort_above_cycles` 10, `cycle_budget` 42; log
`bfs3d_openfoam/march.log`:

| | recorded (pre-split) | re-marched (post-split) |
|---|---|---|
| steps | 61 | **61** |
| Krylov cycles | 208 | **208** |
| final ‖R‖ | 1.858e-06 | **1.858e-06** |
| mid-span / full-span `x_r/h` | 8.361 / 12.53 | **8.3611 / 12.53** |

**And the reason it is unchanged is the useful part: this case never takes the cost path at all.** Its
two retries are both `alpha`, so there was no `cycles` escalation for the split to remove. The
behaviour change is therefore confined to configurations where a solve actually exceeds the budget
without reaching target — on the evidence so far, `pitzDaily` under the zero-fill smoother, which is
where the defect was found. ⚠️ Wall was 1408 s against a recorded 1246 s, which is the soft number: one
run each, different machine conditions, and the recorded figure has no kernel provenance (this one is
logged as compiled).

**The reasoning that produced this, kept because it generalizes.** The problem was that
the cost threshold is an absolute count and therefore **arm-dependent**, which conflicts with this
project's knob-free-robustness goal: a cycle count is `preconditioner strength × operator difficulty`,
so **any constant encodes an assumption about which preconditioner is installed.** Four directions were
written down; (4) is what shipped, and the other three remain available if the abort threshold itself
proves to need calibrating:

1. **Make the trigger relative to the arm's own history** — an anomaly test against a running median of
   recent accepted solves (say 3×) rather than against a constant. Self-calibrating; needs a cold-start
   rule for the first few steps, where there is no history and the march is at its most fragile.
2. **Trigger on the cycle BUDGET instead.** A solve that exhausts `cycle_budget` was truncated and its
   direction is genuinely suspect; one that finished under it converged. That is a resource decision
   rather than a diagnosis, so it does not need per-arm calibration — but it is blunt, firing only at
   the extreme.
3. **Make the criterion dimensionless: cost per unit residual reduction**, not cost alone. The rule
   wants "this step is going badly", and many cycles that buy a large drop is not that. Something like
   `cycles / log(‖G_in‖/‖G_out‖)` measures what is actually wanted; the present rule reads the numerator
   only.
4. **✅ SHIPPED — a COST signal does not drive the β ladder at all.**
   A high cycle count is evidence about the **preconditioner** (stale factorization, weak smoother), not
   about the **step's stiffness**, and this case already routes it to the right responder:
   `refresh_on_cycles = 3` rebuilds the frozen preconditioner on exactly this signal. Sending the same
   observation to β-escalation as well conflates two diagnoses — and on this case the escalation is not
   even monotone in the right direction. The honest stiffness signal is **α**, which is dimensionless
   and needs no calibration.

**⚠️ AND `on_cycles` DID TWO JOBS — the discovery that shaped the fix.** Besides triggering the
β-escalating retry, `RetryPolicy.thresholded_step` pushes it down as `abort_above_inner_cycles`, which
**cuts the dual-time inner loop short** the moment a solve exceeds it (`retry.py`). It does not truncate
the Krylov solve — the log shows the over-threshold solves running their full 12 cycles — it stops the
step taking further inner iterations. So on the failing run, step 29 ran 3 inners and was discarded;
on the converging one the identical first two inners (`1.663e-01 → 7.841e-03 → 3.865e-03`, bit-identical)
were followed by a 4th and the step was accepted. The threshold therefore withheld *convergence the step
was in the middle of achieving* and then escalated β for the trouble.

**There is weak evidence for (4) already.** In the converging run no solve reached 25 cycles, so the
trigger **never fired** — that run is observationally what "no cycles trigger for this arm" looks like,
and it converged on `alpha` retries alone. ⚠️ Weak because it is one arm on one case. And removing the
trigger outright would **not** have been a pure subtraction — it would have taken the early abort with
it, which `cycle_budget` depends on. That is why what shipped splits the two rather than deleting one.

**Two structural observations worth keeping:**
- **`pointwise_rowlength` is EXACTLY `cell_major_rowlength`** — identical at all six points, not merely
  close. The coloured probe assembles against a fixed block pattern, so every row of a cell has the same
  stored nonzero count, and a stable sort by row length therefore groups them by cell on its own.
- **Preserving the cell blocks helps RCM**: applied per-cell it beats the pointwise form at the hard
  point (140 against 203 at self-start β = 0.5) and ties elsewhere. Consistent with the interleave being
  load-bearing, which the two field-major arms confirm from the other side.

    **PLAIN aggregation, not smoothed — `pc_gamg_agg_nsmooths = 0` (measured, and the largest
    preconditioner win found on this case).** Smoothing the tentative prolongator with a Jacobi step is
    GAMG's default and is right for an M-matrix-like operator; on a strongly indefinite saddle it
    degrades the coarse correction. Measured with the march's **own** states, right-hand sides
    (`-R(state)`) and shift pairing (operator at the march's β, V-cycle at `max(β, floor)`):

    | state | smoothed (was) | plain (now) |
    |---|---|---|
    | entering a step below the shift floor (β 0.0154, α clipped to 0.200) | 22 cyc, 3.2e-8 | **9 cyc, 4.7e-10** |
    | entering the retried step (α → 0) | 4 cyc, 1.2e-12 | **3 cyc, 5.7e-14** |
    | the converged tail | 6 cyc | 6 cyc |

    **The tie at the converged tail is the methodological point:** an easy operator does not discriminate
    between preconditioners, so the first sweep — taken there — reported no difference on any arm and
    nearly closed the question. Sweep preconditioners at the march's *hard* states, which the checkpoints
    identify by cycle count and clipped α. Plain aggregation is also marginally cheaper to set up.
    **It is also what makes a DEEP hierarchy usable:** with smoothing ON, adding levels via
    `pc_gamg_threshold` gives dense coarse operators, builds of 96–1224 s, and a V-cycle returning
    **NaN** at every threshold tried (0.01/0.05/0.1/0.25); with it OFF, `threshold=0.05` builds a
    **5-level** hierarchy in 26 s at the same cycle count. That matters for scaling — the shipped
    2-level design leans on a direct LU at `coarse_eq_limit` equations to carry the global modes, which
    is cheap at 23k cells and will not be at 10× — so the threshold is left OFF as the default (it buys
    nothing on *this* mesh) while being known to work.
    **Corollary worth keeping: the NaN is the smoothing/threshold INTERACTION, not the threshold.** Do
    not re-refute strength-of-connection on the smoothed-aggregation evidence.
    **VALIDATED ON THE FULL 3-RUNG MARCH** — 4050 s / 62 steps / **347 cycles** / 3 retry cascades →
    3474 s / 69 steps / **290 cycles** / **1** cascade, converging deeper (2.6e-6 vs 8.9e-6) with the
    answer unchanged (mid-span `x_r/h` 8.36). Per rung the effect lands exactly where the sweep said:
    rung 1 (the easy high-β anchor) **37 → 37, identical**, rung 2 110 → 75, rung 3 200 → 178. Note the
    march takes *more* steps (62 → 69): cheaper solves hold α higher, so the Courant control grows β
    differently and the trajectory diverges from step 24 — this is a whole-march total, not a per-step
    improvement, and part of the wall saving is the two retry cascades that stop happening. That same march
    ran to the target Reynolds number with no breakdown and β reaching **0.0077** at 6–11 cycles per solve,
    well past the 0.02 where ILU(1) diverged — the qualitative claim that matters.

    **⚠️ CONFLICTING CYCLE TOTALS for this one 62-step march, unresolved.** It is recorded here as **347
    cycles** and elsewhere as **883 "raw" cycles**, with "raw" nowhere defined. Neither is recoverable from
    source. Treat the *ratios* as the finding and the absolute total as unestablished — re-measure with the
    counter's definition stated in the same breath if a cycle total ever becomes decisive.
  - **β-diagonal split — track β without re-materializing the Jacobian (BUILT).** The operator is
    `J(φ) + β d`, and the shift `β d` touches only the **diagonal**, so a β-tracking refresh does **not**
    need the coloured-probe materialization of `J` (the dominant refresh cost — hundreds of jvps).
    `MonolithicAmgPreconditioner.refresh_shift_in_place(shift)` reuses the **cached** Jacobian (stored at
    the last `build` / `refresh_in_place`), re-adds the new `β d` diagonal (`O(nnz)` numpy) and re-factors.
    Measured on the `bfs3d` hard state: **full `refresh_in_place` 36 s vs shift-only 18 s (2×)** — the 18 s
    saved is the materialize, the remaining 18 s the equilibrate + GAMG refactor. The frozen `J` does not
    track *state* drift, so the full materialize is **gated** (`_materialize_gate`, mirroring
    `_staleness_beta_gate`): `amg_beta_tracking_refresh(materialize_drift=τ, materialize_every=K)` does the
    cheap shift-only refresh in between and a full materialize when the ν_t drift since the last one exceeds
    `τ` OR after `K` steps (both `None` = full every refresh, unchanged). Prefer `materialize_drift` (the
    honest state-staleness signal via `eddy_viscosity_drift`) with a large `K` as the safety cap — the fixed
    step count was the "fixed cadence" antipattern. **Measured caveat: a *fresh* Jacobian is worth its cost
    on a fast-developing flow** — driving the materialize *more* often (via a tight `τ`) cut the march's
    Krylov cycles ~23 % (fewer/cheaper steps) despite more refresh, so under-materializing was costing more
    in solve than the materialize saves; the lever is a *cheaper* materialize (batched probe + gather
    de-compression above), not a rarer one. Forward-march only. Pinned by `test_amg_refresh_shift_in_place_*`
    (`test_amg_preconditioner.py`) and `test_materialize_gate_*` / `test_batched_probing_*` /
    `test_gather_de_compression_*`.
    - **⚠️ THE TWO GATES ARE COMBINED, NOT NESTED — the β floor used to make the drift trigger UNREACHABLE
      (fixed; `_refresh_branch`).** `_beta_tracking_refresh` asked the β gate first and the materialize gate
      only *inside* it. With a PC-only `beta_floor` the gate's input is `max(β, floor)`, so once the march
      drops below the floor that input is **pinned** and the β gate answers "no change" on every step
      forever — taking the drift gate down with it. That is exactly the low-shift tail where the flow
      develops fastest. Measured on the 3-rung `bfs3d` cold march (56 steps, `beta_floor = 0.05`):

      | | steps | refresh declined | mean cycles |
      |---|---|---|---|
      | β ≥ floor | 34 | 12 % | 6.9 |
      | β < floor | 22 | **91 %** | 7.5 |

      13 steps had >5 % ν_t drift *and* no refresh, and they carried **189 of the march's 399 Krylov
      cycles (47 %)** — steps 29–31 ran 24/23/34 cycles on a V-cycle nothing was allowed to refresh, and
      the step after them blew up into a 3-attempt β-escalation retry costing 380 s. The decision is now
      the total function `_refresh_branch(stale_state, moved_beta, split)`: **state drift ⇒ `full`**
      whatever β says, β move alone ⇒ `shift`, neither ⇒ `none`. Note the shift branch is real work below
      the floor too — the shift is `pc_beta · d(state)` and the per-cell `d` tracks the state even where
      `pc_beta` is clamped, so the old comment's "would rebuild an identical V-cycle" was only true of the
      β factor. **Consequence to know:** the materialize gate is now consulted every step rather than only
      on refresh steps, so its `materialize_every` cap counts **steps**, not refreshes.
      **MEASURED END-TO-END on the 3-rung `bfs3d` cold march, and the trade is strongly favourable:**

      | | before | after |
      |---|---|---|
      | outer steps | 69 | **61** |
      | **Krylov cycles** | **480** | **348 (−27 %)** |
      | starved steps (>5 % drift, no refresh) | 15, holding 201 cycles (42 %) | **0** |
      | sub-floor steps declining a refresh | 89 % | **19 %** |
      | full refreshes | 32 @ 23.1 s | 48 @ **17.3 s** |
      | refresh total | 803 s (15.7 % of wall) | 865 s (19.1 %) |
      | β-escalation retry steps | 5, **1745 s** | 2, **655 s** |
      | final ‖R‖ / mid-span `x_r/h` | 2.65e-6 / 8.36 | 2.42e-6 / **8.36** |

      Read the refresh row correctly: the preconditioner now costs **more** in absolute terms (865 s vs
      803 s) because it refreshes 50 % more often — that is the trade working, not a regression. It buys
      132 fewer Krylov cycles and, far larger, removes three of the five retry cascades (−1090 s), which
      were the single biggest line item in the march. The converged answer is **unchanged** (`x_r/h` 8.36
      mid-span against OpenFOAM's 7.24, exactly as before), which is the constraint that matters: this is
      a path change, not a solution change.
    - **Where a refresh's time now goes (whole-run aggregate, same march): probe 632 s, refactor 195 s,
      assemble 28 s, other 12 s — the probe is 73 %.** The non-probe tail is close to floor, so any
      further work on refresh cost has to attack the coloured probe itself (amortizing colours across
      steps, or a cheaper stencil), not the assembly or the multigrid setup.
    - **⚠️ Wall-clock from that comparison is a LOWER bound, and the reason is a trap worth naming.** The
      "after" run was watched by a per-refresh log monitor whose notifications drove the desktop UI to
      ~70 % CPU against the solve; its first ~20 minutes are inflated (steps 1–3 were *bit-identical* in
      cycles and residual to the baseline while taking 254 s against 224 s). Cycles, phase times and
      step/retry counts are unaffected — which is exactly why the cycle count, not the clock, is this
      project's cost measure. **Do not instrument a long run with a per-step notification stream.**
    - **Where a refresh's time actually goes — REPORTED, not inferred (`RefreshTiming`, `solve/refresh_timing.py`).**
      The observer used to receive `(kind, seconds)`, so a breakdown had to be recovered by differencing
      the `full` and `shift` branches across a run. It now receives a record carrying ordered
      `(phase, seconds)` pairs — AMG: `probe` / `assemble` / `refactor` — and `MarchLogger` renders them
      under the `"pc"` detail (`pc full 23.0s (probe 14.6 assemble 3.2 refactor 5.2)`), with any
      unattributed remainder shown as `other` so a breakdown cannot silently fail to add up. The
      factorization preconditioners report no phases (empty tuple), which the record documents as valid.
      Differencing the same 56-step march gave probe ≈ 14.6 s of a 23.0 s `full` (63 %) against 8.4 s for
      `shift` — consistent with the older ~40-of-60 s reading, and the reason the probe is the lever.
    - **⚠️ EVERY refresh between two step rows is counted and summed, because more than one fires
      (fixed 2026-08-17).** `on_refresh` used to *overwrite* its record, which was safe only while the
      hook ran once per step. A rebuild triggered by solve cost (`refresh_on_cycles`) fires **inside** a
      step, and a **retried** step runs its inner loop again, so several land between step rows — six in
      one observed step. The old rendering therefore showed one refresh per step however many fired, and
      printed one refresh's seconds as the step's whole preconditioner cost: on that step,
      `pc inner 15.2s` for what was really **65.4 s, 22 % of a 301 s step** rather than 5 %. Seconds and
      phases now sum across the step and a branch that ran more than once is marked `Nx` —
      `pc full inner 3x none 2x 65.4s (probe 11.8 assemble 0.5 refactor 52.0)`. A single refresh renders
      exactly as before. **Any refresh-versus-step cost comparison drawn from a log written before this
      fix understates the refresh side, by however many refreshes that step happened to fire.**
    - **The per-refresh sparse-matrix work is precomputed (`ShiftedCellMajorOperator`).** The `assemble`
      phase — add `β d` to the diagonal, symmetrically equilibrate, reorder to cell-major — was a sparse
      add plus two sparse products plus two fancy-index permutations, each allocating and re-sorting a
      matrix the size of the coupled Jacobian, **repeated identically every refresh** because for a fixed
      stencil reach the pattern never changes and only the values do. The class hoists the
      pattern-dependent part (which base nonzero feeds each output nonzero, where the diagonals sit, the
      cell-major CSR structure) into a one-time build, leaving one gather + an `O(n_dofs)` diagonal add +
      a symmetric scale into a **preallocated** buffer. Bit-identical to `equilibrate_cell_major(J +
      diags(shift))` — pinned by `tests/unit/test_cell_major_operator.py`, which runs without `petsc4py`
      because the class holds none. Two details worth keeping: the returned matrix **aliases the reused
      buffer** (consume it immediately; `AmgVCycle._build`/`refactor` both copy), and the symmetric scale
      is applied **chunked over rows** so it never allocates a second Jacobian-sized temporary — the
      point of the exercise being to stop allocating those. It engages only when a precomputed
      `structure` was used, which is precisely the guarantee the pattern is fixed; otherwise the generic
      path runs unchanged. `AmgVCycle.refactor`'s pattern re-check is likewise memoized on the index
      arrays' identity, since comparing tens of millions of indices twice per refresh re-confirms
      something fixed by construction.
    - **Two redundant full copies of the operator's VALUES are gone (2026-08-14).** `_build` wrote
      `cell_major.data.astype(ScalarType).copy()` — `astype` already returns a fresh array, which is the
      ownership the persistent `Mat` needs, so the `.copy()` allocated the values a second time — and
      `refactor` wrote `self._data[:] = cell_major.data.astype(ScalarType)`, whose `astype` builds a full
      temporary that the assignment then copies out of, on **every** refresh. A slice assignment casts to
      the destination's type as it copies, so it needs no `astype` at all. Each is one array as long as
      the Jacobian's nonzeros. Behaviour-identical; not measured.
    - **Already done, do not re-propose:** `refactor` reuses the aggregation and prolongation
      (`pc_gamg_reuse_interpolation` + smoother `reuse_ordering`) and overwrites the operator values in
      place, so only the Galerkin coarse operators and the incomplete-LU factor values recompute.
    - **⛔ THE VANKA PATCH SMOOTHER IS DELETED (2026-08-15) — it never won on any arm, and it was kept
      only on the condition that it would.** It was built to let the "the coarse space is the wall"
      verdict be re-adjudicated; that re-adjudication happened and went against it (see the low-β
      bullet below: measured against a *working* coarse space it stagnates on its own, at a state where
      the shipped incomplete-LU converges in two cycles). It was never exported, never a default, and
      never beat the incumbent, so it went rather than be carried — the module, its tests and the probe
      arms are all recoverable from git history if the question ever reopens.
      **⚠️ Everything below is kept deliberately, as the JUSTIFICATION for that deletion and as the
      record of what the route costs, and it is no longer re-runnable without restoring the module.**
      Three facts worth keeping:
      - **Route: PETSc's *shell* preconditioner, not `PCPATCH`.** `PCPATCH` with
        `pc_patch_construct_type = vanka` needs a `DM` the plain-AIJ path does not supply, which is why
        the earlier plan flagged it as a risk. `pc_type python` + `pc_python_type
        aquaflux.solve.vanka.VankaPC` needs **nothing but the assembled matrix** — verified on a plain
        `createAIJWithArrays` matrix with no `DM`, and it reaches every level (block size propagates to
        the Galerkin operators, so the coarse levels are configurable too). Reached through the
        `extra_options` seam, so no change to `AmgVCycle` was needed. Options carry the level prefix:
        `mg_levels_pc_type python`, `mg_levels_vanka_neighbours`, `vanka_neighbour_fields`
        (`before_centre`|`all`), `vanka_centre_field`, `vanka_damping`. On the shipped 2-level hierarchy
        `mg_levels_` *is* the fine level (level 0 is `mg_coarse_`), so no level-index bookkeeping.
      - **The patch is chosen ALGEBRAICALLY, by coupling strength, and it has to be.** The classical
        collocated Vanka patch is "the cell plus the cells its continuity row couples to" — but this
        Jacobian's stencil is distance-3, ~47 cells per row, so that patch would be a 280-unknown dense
        solve. `CellStarPatches` instead ranks the pressure row's cell-collapsed `Σ|a_ij|` and keeps the
        `n_neighbours` strongest. That also keeps the module mesh-free (`numpy`/`scipy` only, no `jax`),
        so it unit-tests on a three-cell matrix. **The strength is summed only over the fields the patch
        will actually contain** — with `before_centre` (the classical choice: the neighbours contribute
        velocities, each neighbour's pressure staying in its own patch) that is the divergence entries
        alone. Ranking on the whole row instead lets the collocated pressure-pressure damping term pick
        the patch's cells, which are not the cells the patch exists to invert.
      - **Additive with overlap averaging, deliberately — the recombination is the part that failed
        before.** The recorded "additive Vanka + Richardson diverges (ρ 9e4)" arm was an *unweighted*
        additive sum, which over-corrects every unknown that several patches share; the undamped
        block-ILU/inexact-Uzawa arm ("1-apply reduction 5.10") amplified for the same reason. The weight
        is `W^½ (Σ RᵀA_pp⁻¹R) W^½` with `W` the reciprocal coverage — split symmetrically across
        restriction and prolongation, which is the normalization Schöberl–Zulehner (2003) use and what
        Metsch's algebraic Vanka (dissertation §4.6) implements. It is a partition of unity, so on
        non-overlapping patches the smoother is exactly the block inverse (pinned by a unit test), and
        one application is a **bounded, fixed linear** operator — which is what the non-flexible outer
        GMRES and the adjoint's transpose both require, and what an inner-Krylov ("Krylov-Vanka")
        smoother is not.
        **Two deviations to state with any result, both making this weaker than the textbook smoother:**
        the classical Vanka sweep is *multiplicative* (patches solved in sequence against a residual
        updated as it goes) and this is additive; and the patch is truncated by strength rather than
        taking the whole continuity row. A null result from it is therefore *not* on its own evidence
        that patch relaxation cannot help — which is why the harness also runs a sweeps ladder on the
        Vanka arm, not only on the incomplete-LU one.
        `VankaSmoother.worst_patch_gain` (max `|A_p⁻¹|`) is printed at setup, because a near-singular
        patch produces an enormous additive correction and looks from the outside exactly like "patch
        relaxation does not work here". Cost at the `bfs3d` shape (23k cells, 6 fields, 280 nnz/row),
        extrapolated from a 5k-cell synthetic of the same density: patch selection ~0.1 s, factorization
        0.5–3 s, one apply 7–110 ms, and the stored inverses 0.1–1.1 GB depending on patch width — which
        is why the harness's widest arm is 12 velocity neighbours (~325 MB) and not 12 full cells.
    - **`AmgVCycle.destroy()` / `.levels`.** The V-cycle owns several PETSc objects and a factored
      hierarchy; `destroy()` releases them on the caller's schedule rather than the collector's, which a
      loop that builds one preconditioner per arm needs (two live copies of a 3D coupled operator plus
      factors is enough to exhaust a workstation — the standing "one heavy probe at a time" rule).
      `refactor`'s rebuild branch calls it too, so the teardown has one home.

- **Three small clones removed (BUILT 2026-08-15), and one deliberately left.**
  - The SIMPLE smoother's sweep loop is `sweeps_from(level, rhs, guess, count)` in
    `saddle_multigrid._native_saddle_cycle`. `smooth` and `smooth_zero` each declared the identical
    10-line `fori_loop` body; they differ only in where they start and how many sweeps that leaves.
    Bit-identical at 1, 2 and 4 sweeps.
  - `block_stencil_colouring` calls `frozen_operator.require_valid_graph` instead of re-inlining three
    of its four checks. That helper takes the caller's name for its messages, which is precisely why it
    is shared; only the `reach` check is the colouring's own. It also loses its underscore, since it now
    crosses a module boundary.
  - `checkpoint._atomically(path, write)` is the one home for write-then-rename. It was written twice —
    once with the reasoning attached and once without, and the copy without it is the one a reader would
    have had to reconstruct the argument for. `write` takes a **path**, matching the serializer contract
    ("write exactly to this path"), so an injected serializer needs no wrapping. Verified that a crashed
    write leaves **no** file that reads as a checkpoint, which is the whole point of the idiom.
  - **NOT done: `vanka.colour_patches`' per-vertex Python-set loop**, which duplicates the algorithm
    `sparse_jacobian._saturation_colouring` vectorizes and whose docstring records the difference as
    "seconds and minutes" on a high-degree graph. Deliberately skipped: patch relaxation is measured
    closed on this operator and `vanka.py` is a deletion candidate, so optimizing it is work spent on
    code slated to go.


## The JAX-native multigrid — in progress

- **⚠️ IN PROGRESS (2026-08-10): the native trailing V-cycle now MATCHES PETSc on quality AND cost;
  what blocks it is the POSITIVITY LIMITER — read that as the PROXIMATE failure, not the cause.**
  ⚠️ The same limiter, defaults and case run fine under the PETSc ILU(0) control (58 steps to 9.588e-06,
  recorded below), so the limiter is necessary and not sufficient: the native arm dies at a state the
  control never reaches, and what drives that arm's direction into the boundary is **unexplained**. Read this before the
  2026-08-09 section below, which it supersedes in several places.

  **Where it stands. Configuration, stated here rather than 500 lines away, because the sweep count
  is load-bearing:** `bfs3d` `state-00057`, PC β 0.05, the `[k, ω]` block **ALONE** (46080 dofs,
  4.20M nnz), GMRES restart 15 to rtol 1e-8 on the TRUE residual, **4 smoother sweeps**, the PETSc
  side on its **matched-smoother** arm (plain aggregation, point-block Jacobi ×4 — against ILU(0) ×4
  PETSc does 1 cycle, not 2); harness `trailing_hierarchy_sweep.py`. On that arm the JAX-native nodal
  hierarchy reaches **2 restart cycles against PETSc GAMG's 2**, on a 438-equation coarse space
  against 432. Quality parity, on CPU. Per-apply cost came out at parity too, but that pair of
  timings was recorded with no sweep count and no state and is deleted — measured 2026-08-10,
  configuration not recorded, re-measure before relying on it. The full march converges
  rung 1 (14 steps, 43 cycles, 304 s against the PETSc control's 14 / 45 / 359) and then dies on
  rung 2.

  **Four things closed the gap, and each was found by reading the reference rather than tuning:**
  1. **The aggressive first level.** `build_amg_vcycle` never sets `pc_gamg_aggressive_coarsening`,
     so GAMG applies its default of one aggressive level over the SQUARED graph. Ours had none:
     21× coarsening against GAMG's 107×, the whole 5× coarse-space difference. (`use_aggressive_
     square_graph` and `aggressive_mis_k` are ALTERNATIVES; at the default the coarsener is plain
     MIS at distance 1 on the squared graph.)
  2. **PETSc's level smoother is UNDAMPED** — `richardson` at its default scale of 1. Ours relaxed
     by `omega/lam_max`, and `D⁻¹A` has a unit diagonal so `lam_max ≥ 1` always: the spectral factor
     can only ever under-relax. Worth **10 → 2 cycles**. Reachable as `spectral_damping=False`.
  3. **A CSR matvec instead of the COO `segment_sum`.** A scatter-add collides on output rows; a CSR
     row walk does not. Measured on the 4.2M-nnz block: `segment_sum` 13.3 ms, `BCOO @ x` 13.4 ms,
     scipy CSR on the host 2.6 ms, **`BCSR @ x` 1.4 ms**. The level operator is applied ~10× per
     V-cycle, so this alone took the apply from **117.8 ms to 13.3 ms (8.8×)**. Landed as
     `_CsrOperator`, which owns its matvec and replaced the loose `row`/`col`/`val`. **Not** a jit
     or marshalling problem — both were measured out (1 trace everywhere; a numpy↔jnp round trip is
     0.02 ms), which also retires the older claim that "marshalling is ~half the gap".
  4. **The singularity guard was WRONG, and it was aborting the march.** `_cell_block_inverse`
     tested `|det| < 1e-12 · ‖B‖_F^b`. On the coupled turbulence pair a cell reads
     `[[8.8e-06, 1.7e-12], [-1.3e+03, 1.5e-01]]` — rows differing by **1.5e8** — so `‖B‖_F²` ≈ 1.6e6
     is set entirely by the ω row and the bar lands at 1.6e-06, just above the determinant of
     1.35e-06. The block is **not singular**: on the row-norm (Hadamard) bound `|det| ≤ ∏‖rowᵢ‖` it
     scores **1.2e-04**, eight orders clear. Now tested that way, which is invariant under rescaling
     any row or column where the Frobenius form is not. A structurally empty row is named
     explicitly, since it makes the bound zero and no determinant compares below it.

  **⚠️ THE α COLLAPSE IN THESE MARCHES IS THE POSITIVITY LIMITER, AND THE STEP TABLE SAYS SO — READ
  `limit` BEFORE ATTRIBUTING ANYTHING TO THE PRECONDITIONER.** `a_min` and the `limit` aside are the
  **same number** wherever both appear (0.579/5.79e-01, 0.651/6.51e-01, 0.004/3.76e-03, …). Those
  steps are not failing to descend; they are being allowed almost no movement because `k` would go
  negative. Two consequences, both of which cost hours today:
  - **A capped step is not a bad step.** Two arms differed in α at rung-2 step 16 (1.000 against 0.579)
    and reached an **identical** residual, 1.271e-01. Attributing the α difference to preconditioner
    quality was wrong. ⚠️ **Those two numbers are NOT a raw-vs-equilibrated pair** — they are `march.log`
    (**petsc** trailing inverse) against `march-20260810-223702.log` (**native + `equilibrate=True`**),
    both under the `dirichlet` k wall BC. The genuine `equilibrate` A/B at step 16 is **0.699 against
    1.000**, under `zerogradient` (`march-20260810-221936.log` / `march-20260811-003915.log`, both at
    ‖R‖ 1.382e-01). The point about a capped step survives either way; the labelling did not.
  - **⚠️ DO NOT GATE `retry.on_alpha` ON `binding_limit == 1`. It was proposed, built and reverted
    the same day (2026-08-10).** More damping shrinks the correction, so it *widens*
    `room = k/|dk|`. The measured evidence is already in this file: the single escalation of an
       entire `bfs3d` march fired at step 51, whose cap was **4.37e-10** — constraint-bound — and it
       was worth **8 steps and 199 s** end to end. Gating `retry.on_alpha` on `binding_limit == 1`
       would have suppressed exactly that one useful escalation.

    **What damping genuinely cannot do is un-pin a cell already ON the boundary**, and that is the
    real rung-2 failure — see the lock-up section immediately below, which is what replaced this.

  **⚠️ (2026-08-10, LATER) THE RUNG-2 DEATH IS A FRACTION-TO-THE-BOUNDARY LOCK-UP, AND β ESCALATION
  IS A SYMPTOM RATHER THAN THE CAUSE.** Read this before acting on the two bullets above; it does not
  contradict them but it re-ranks them, and the "~100 dead steps" is now measured rather than
  estimated. Taken from the archived `march.log` of the native run (banner: `turbulence inverse:
  native`, `equilibrate=True`, `sweeps=4`, `aggressive_levels=1`, `spectral_damping=False`,
  `refresh on cycles 3`, `retry on cycles / alpha 10 / 0.01`, `pc beta floor 0.05`, rung 2 of 3):

  | step | β | cyc | ‖R‖ | a_min | `limit` |
  |---|---|---|---|---|---|
  | 24 | 0.1756 | 5 | 1.123e-01 | 0.694 | 6.94e-01 |
  | 25 | 0.4682 | 2 | 7.567e-02 | 0.004 | 3.76e-03 |
  | 26 | 3.7458 | 0 | 7.316e-02 | 0.000 | 1.00e-05 |
  | 27 | 16.0000 | 0 | 7.316e-02 | 0.000 | 1.95e-06 |
  | 28 | 16.0000 | 0 | 7.316e-02 | 0.000 | 1.95e-08 |
  | … | 16.0000 | 0 | 7.316e-02 | 0.000 | ÷100 every step |
  | 122 | 16.0000 | 0 | 7.316e-02 | 0.000 | 1.95e-196 |

  **The 100× per step IS the rule, not a coincidence.** Taking `α = τ·room` with `τ = 0.99` leaves the
  binding cell at 1% of its `k`, so the next step's room is a hundredth of this one's — forever, for
  as long as the direction keeps pointing at the boundary there. Ninety-six consecutive steps, residual
  frozen to every reported digit, **zero** Krylov cycles, β pinned at its 16.0 ceiling. So:
  - **β is irrelevant from step 27 on, and that is a statement about a PINNED cell, not about damping.**
    Damping is *argued* to widen a fraction-to-the-boundary cap (a smaller correction means more room),
    and that argument is **not measured and cannot be from a log** — only the accepted attempt's cap is
    recorded. ⚠️ An earlier version of this bullet offered "the cap falls only 5.1× across 26→27" as
    evidence; that is the **same cross-step comparison withdrawn above**, used to argue the opposite
    conclusion, and it is struck. The decision rests on the A/B alone. What damping cannot do is recover
    a cell whose `k` is already ~0: the
    room is then small however small the correction gets. **Do not read this as "stop escalating on a
    constraint-bound step"** — that was tried, and it deletes the one escalation on this case that is
    measured to pay (step 51, cap 4.37e-10, 8 steps and 199 s).
  - **Nothing in the ordinary stopping tests can see this.** The state is finite, the residual is finite,
    and the tolerance is simply never reached — so the segment spends its whole budget.
  - **⚠️ Extracting this from a march log needs per-step BLOCKS, not two independent `findall`s.** The
    `limit` line is written only when `binding_limit < 1`, so there are fewer of them than steps (105
    against 108 here) and zipping the two lists silently misaligns them. The first pass at this table
    read step 25's cap as 1.95e-08 instead of 3.76e-03 — a three-step shift, invisible because the
    shifted table is just as smooth. Split on the summary-block delimiter and parse within each block.

  **SHIPPED in response:**
  1. `forward_march(stop_on_limit_stall=3)` (**a new default-on guard**) ends the segment after three
     consecutive steps that are constraint-bound, non-widening, and **changed the residual by less
     than 1e-3 relative, in either direction** (`_limit_collapsing`).

     **⚠️ The residual half of that predicate was wrong twice, and the fix for the first attempt caused
     the second. Do not re-derive it from the failing run alone.**
     - *"the residual did not fall"* — **never fires.** A locked-up step is not a bit-exact no-op: it
       still moves the state by `α·δ`, so at a cap of ~1e-6 the residual genuinely falls, by ~1e-6
       relative, and the counter resets every step. Shipped, and observed doing nothing on a march that
       had otherwise reproduced the lock-up step for step (it reached step 31 and was still going).
     - *"the residual did not fall by 0.1%"* — **fires on a converging rung.** `march-20260810-094635`
       rung 2 steps 19–21 ran caps 0.983 → 0.928 → 0.253 with the residual climbing
       1.293e-01 → 1.335e-01 → 1.489e-01, then recovered to 9.241e-02 and converged to 4.994e-06. A
       *rising* residual means the step did something; a pseudo-transient path is a march in pseudo-time,
       not a descent method. A one-sided rule ends that rung at its worst moment.
     - **The failure is a step that changes NOTHING, so the test is two-sided.** The two populations sit
       orders apart: null steps move the residual by ~1e-6 relative, productive ones by percents.

     **Validated by replaying the predicate over every march log on disk** — `march_stall_replay.py`,
     kept in `validation/bfs3d_openfoam/`, because a march that recovers looks exactly like one that does
     not until it does, so a candidate rule cannot be judged on the run that motivated it. Across 11 logs
     / 22 rungs it fires on exactly the three locked-up rungs (090711 rung 2 at step 21, cutting a
     73-step stall; 130032 rung 2 at step 29, cutting 96) and on **nothing** that went on to converge —
     including the shipped `x_r/h` 8.36 baseline rung. Both cases are pinned as unit tests with the real
     recorded numbers inlined, since the logs themselves are not tracked.

     The cap *narrowing* is the third condition and is what separates a lock-up from a march working
     productively along a constraint. `None` disables the guard.

  The march now fails fast and honestly instead of grinding; **what drives `k` to the boundary in those
  cells is still open**, and the global-scalar-cap design (one cell throttling all 23040) is the thing
  to reconsider.

  **Capturing the step DIRECTION, which no checkpoint holds.** The cap is a property of `delta`, so
  "which cells does it bind on" cannot be answered from a state — a step checkpoint holds where a step
  ended and the inner-iterate dump holds where an inner iteration reached, and neither carries a
  direction. `compare.py` gained `BFS3D_DUMP_STEP_LIMIT=<cap>` (with `BFS3D_DUMP_STEP_LIMIT_KEEP`),
  which wraps the engine's `step_limit` in a `_DumpingStepLimit` — a **frozen dataclass**, not a
  closure, because the limiter rides in a *static* field compared by `__eq__` and a closure there
  recompiles the whole coupled solve on every rebuild. It returns the real cap unchanged, so the march
  it instruments is the march that would have run. Dumps **stop** after `KEEP` rather than wrapping, so
  the first binding event survives the escalation ladder that follows it. `singular_cell_probe.py`
  reads such a dump and reports the exact per-cell room, the binding set, and the binding cells'
  conditioning against the **base rate** for the mesh.

  **⚠️ SUPERSEDED THE SAME DAY — equilibration is a TRIGGER, not the cause. Read the floored-limiter
  entry above before using anything here.** The A/B below is real and reproducible, but it holds **only
  at `positivity_floor = 0`**. With the floor at 1e-8 the flag stops mattering: `equilibrate=True`
  converges all three rungs in 69 steps and `equilibrate=False` in 67, to the **same root in every
  reported digit**. So rescaling never caused the failure — it pushed a cell into the numerically-dead
  zone a few steps earlier than the unscaled arm did, and the global positivity limiter's geometric
  ratchet did the killing. The defect was in the limiter. Keep the case default at `False` (it is free,
  and it is what the converging arms were measured with), but do not describe this flag as deciding
  convergence.

  **⚠️ EQUILIBRATION DECIDES WHETHER THE `bfs3d` NATIVE MARCH CONVERGES — true ONLY at floor 0 (see
  above). Measured 2026-08-11.** An A/B differing in **one flag** was run end to end:

  *Configuration, both arms:* `bfs3d` (23040 cells), 3-rung Reynolds continuation (`N_POINTS=2` →
  Re/100, Re/10, target), `BFS3D_TURBULENCE_INVERSE=native`, `field_split=True`; native trailing
  `cycles=1, sweeps=4, max_coarse=2000, aggressive_levels=1, prolongation_smoothing=none,
  spectral_damping=False`; monolithic smoother **ILU(0) ×4**, `coarse_eq_limit` 2000, plain aggregation,
  reach 3, forward restart 15, `refresh_on_cycles` 3, `retry on cycles / alpha` 10 / 0.01, cycle budget
  42, PC β floor 0.05, stop `(rtol, atol) = (0.0, 1e-5)`, `k` wall BC `zerogradient`.

  | arm | outcome | steps | final ‖R‖ | mid-span `x_r/h` |
  |---|---|---|---|---|
  | `equilibrate=True` (archived `march-20260810-221936.log`) | **locked up, rung 2 step 34** (α 0.001; ‖R‖ frozen from step 35; killed by hand at 39, not solver-terminated) | 39 | frozen 1.257e-02 | — |
  | **`equilibrate=False`** (`march-20260811-003915.log`) | **converged, all 3 rungs, 2081 s** | 77 | **3.586e-06** | **8.361** |

  **It REPRODUCES — the claim does not rest on one run each.** Two further `equilibrate=True` runs,
  `march-20260810-130032.log` and `march-20260810-223702.log` (the latter dump-free), both stall at the
  **identical** ‖R‖ 7.316e-02 with the same β ladder — and they ran under the **`dirichlet`** k wall BC,
  where the pair above ran `zerogradient`. So the flag's effect survives a change of wall closure. The
  march is also deterministic: four runs at identical configuration agree in every printed field.

  **⚠️ The dump wrapper is NOT the confound (checked, not argued).** `221936` ran with
  `BFS3D_DUMP_STEP_LIMIT` and `003915` did not. A dump-ON/dump-OFF control pair at otherwise equal
  settings differs in **zero** content lines over 16 steps, against the equilibrate flag's **77**.
  `_DumpingStepLimit` returns the inner cap unchanged and preserves static-field value equality, so it
  neither alters the step nor forces a retrace.

  **⚠️ The separation is a GROWING PERTURBATION, not a threshold event — an earlier version of this
  entry said "bit-identical for 15 steps, then separated on α alone" and BOTH halves are wrong.** The
  arms differ at **step 1** (p-block residual 6.217e-07 vs 6.272e-07) and accumulate 77 differing
  content lines before step 17. They agree only to the **4th printed digit of the summary row**, which
  is insensitive to the diverging component: at step 16 the p-block residuals differ by **19×**
  (1.545e-07 vs 2.938e-06) while the reported ‖R‖ matches to four digits. Step 16 differs in the cycle
  count (5 vs 4) and the retry flag as well as α (1.000 vs 0.699 `L`). The correct statement is: **a
  small preconditioner-dependent difference is present from the first step and amplifies for fifteen
  steps until it crosses the positivity limiter**, after which the clipped α trips `retry.on_alpha` →
  β escalation → more clipping → the 16.0 ceiling at zero cycles. Do not describe this as a clean
  bifurcation; that framing implies a threshold the flag crosses cleanly and the data do not show one.

  **⚠️ `x_r/h` 8.361 against OpenFOAM's 7.243 is TWO GRID STATIONS, not a 15 % discrepancy.**
  `reattachment_length` returns the `x` of the last reversed wall cell — a grid station — and the
  stations near reattachment are `… 6.728, 7.243, 7.787, 8.361, 8.966 …`, spacing ~0.55 h. The
  comparison to the shipped PETSc run (`march.log`: 58 steps, 282 cycles, `x_r/h` 8.3611) is **not
  like-for-like**: it differs in the k wall BC (`dirichlet`) as well as the inverse (`petsc`), and it
  lands on the identical station — i.e. the metric did not resolve either change. Quote the sub-cell
  interpolated crossing from `wall_layer_comparison.py` if this number has to bear weight.
  - **The positivity limiter is NOT the failure — failing to recover from it is.** The converged arm hits
    the same constraint repeatedly (`L` at steps 24, 26, 27, 30, 32, 35, 39, 40, 41, 49, 51, 53, 54, 56,
    57, 60, 61, **including α 0.000 at step 61**) and recovers from every one, α returning to 1.000.
  - **A constraint-free α collapse appeared, and nothing reacts to it.** Step 68: α 0.031 with **no `L`
    flag** and 15 cycles (the run's highest) — a poor *direction*, not a clipped step. α 0.031 is above
    `retry.on_alpha` 0.01, no `RefreshTrigger` reads α or `binding_limit`, and this bundle sets
    `beta_rel_change=inf`, so no refresh fires. It cost a few steps here, not the run, but it is the
    first live evidence that the refresh gap is a real cost.
  - **⚠️ ONE RUN EACH, and one instrumentation difference:** the archived equilibrated arm ran with
    `BFS3D_DUMP_STEP_LIMIT=0.05/12`, the converged arm with the dumps off. The dump wrapper returns the
    real cap unchanged by construction, so it *should* be neutral, but it is not a matched pair.
  - **The stated reason for the `equilibrate=True` default no longer exists.** `native_nodal_inverse`
    defaults it on because "the per-cell block solve is not otherwise safe" — raw, 4 of 23040 cell blocks
    were flagged singular. That count came from the **Frobenius** guard (`|det| < 1e-12·‖B‖_F`), which is
    not invariant under row scaling and is **the guard that was found wrong and replaced** by the
    Hadamard row-norm bound, which is invariant. The default is **unchanged pending a decision**; flipping
    it is a shipped-default change.

  **✅ THE FLOOR RESCUES A CONFIGURATION THAT PREVIOUSLY DIED — which is the real case for it, not the
  step count (measured 2026-08-11).** `equilibrate=True` had failed on rung 2 **twice**, under both wall
  closures (`march-20260810-221936.log`, ‖R‖ frozen at 1.257e-02 from step 34; `march-20260810-223702.log`,
  frozen at 7.316e-02 from step 26). Re-run with `positivity_floor=1e-8` and **nothing else changed**, it
  converges all three rungs.

  | arm | `equilibrate` | floor | outcome | steps | wall | final ‖R‖ | `x_r/h` | escalations |
  |---|---|---|---|---|---|---|---|---|
  | `221936` | True | 0 | **died, rung 2 step 34** | 39 | — | frozen 1.257e-02 | — | 13 |
  | `223702` | True | 0 | **died, rung 2 step 26** | — | — | frozen 7.316e-02 | — | — |
  | `003915` | False | 0 | converged | 77 | 2081 s | 3.586e-06 | 8.3611 | 8 |
  | — | False | **1e-8** | converged | **67** | 2133 s | 3.586e-06 | 8.3611 | 4 |
  | — | **True** | **1e-8** | **converged** | **69** | 2188 s | 3.586e-06 | 8.3611 | 5 |

  **The chain the floor breaks, visible step by step.** `221936` clipped at rung-2 step 16 (α 0.699,
  `L`), which tripped `retry.on_alpha`, which escalated β, which clipped harder, up to the 16.0 ceiling
  at zero cycles. With the floor, that **same step 16 takes α 1.000 with no flag**, matching the
  unscaled arm exactly — the binding cell there was numerically dead, and buying it out removes the
  first link. From step 22 on, the floored equilibrated and floored unscaled arms run the *same*
  trajectory field for field.

  **What it does NOT fix, confirmed on all three converging arms.** The rung-3 hard point reproduces
  bit-for-bit regardless of floor or flag: α 0.010 at 15 cycles, then α 0.000, then β 0.0293 → 0.9364
  (**32×**) across two steps and six steps of SER walking it back, then a constraint-free α 0.031 with
  **no `L` flag**. That is the bad-direction mode, and the escalation ladder's **overshoot** is now the
  clearest remaining cost on this case.

  **Where the wall time goes** (equilibrated + floored run, per-step sums, one run on a shared machine —
  proportions not a benchmark): preconditioner **19 %** (304 s of 1561 s through step 48), of which the
  coloured Jacobian probe is 243 s and the refactor 56 s; 17 of 48 steps carried a refresh and only 2
  were scheduled — the rest fired on the reactive 3-cycle rule. Rung 3 costs ~56 s/step against ~25–30 s
  on rungs 1–2. The four most expensive steps are all first-of-rung (115–171 s), carrying a compilation
  on top of the PC build.

  **⚠️ THE α COLLAPSES ARE TWO DIFFERENT FAILURE MODES, and only one is the limiter's fault — measured
  2026-08-11 by a floored-limiter A/B.** `PositiveBlockLimit` gained a `floor`: the room becomes
  `(phi_i + floor)/|delta_i|`, so an entry that is numerically zero stops setting the step for all of
  them. `floor=0` (the library default) is bit-identical to the plain rule, and the limiter is inactive
  at a root for **any** floor (there `delta = 0`), which is what keeps it out of the converged state and
  therefore out of the implicit-function-theorem adjoint — the adjoint never sees it in any case, since
  `_implicit_solve_bwd` reads only `jax.vjp(residual_fn, phi_star)` and a transpose solve, never
  `forward_step_fn`.

  *Configuration, both arms:* `bfs3d`, native trailing inverse with **`equilibrate=False`**, `k` wall BC
  `zerogradient`, 3-rung Reynolds continuation (`N_POINTS=2`), ILU(0) ×4, `coarse_eq_limit` 2000, plain
  aggregation, reach 3, forward restart 15, `refresh_on_cycles` 3, `retry on cycles / alpha` 10 / 0.01,
  PC β floor 0.05, stop `(0.0, 1e-5)`. Floor **1e-8**, calibrated by replaying the recorded clips
  (`positivity_floor_calibration.py`).

  | | floor 0 | **floor 1e-8** |
  |---|---|---|
  | steps | 77 | **67** |
  | escalations (all `alpha`-triggered) | 8 | **4** |
  | final ‖R‖ | 3.586e-06 | 3.586e-06 |
  | mid-span `x_r/h` | 8.3611 | 8.3611 |
  | `ux`/`uy`/`uz` rel-L2 | 0.0616 / 0.0072 / 0.0061 | identical |
  | ν_t peak | 150.1071 | 150.1071 |
  | wall | 2081 s | 2133 s |

  **The root is unchanged in every reported digit, and the wall clock is a WASH — do not sell this as a
  speed-up.** The floored arm's rung-2 steps run at low β costing 4–9 cycles where the unfloored arm
  escalated into cheap 2-cycle solves, and that cancels the ten steps. (One run each, and this case has
  no measured run-to-run spread, so 2.5% is not resolvable.)

  - **Mode 1, the RATCHET — the floor eliminates it.** A numerically-dead cell with a *tiny* correction:
    `tau` leaves it at `1 - tau` of its value, so the cap falls ×100 per step while its `delta` never
    moves. Rung 2 went 34 steps → 24, three escalations → one, and the clips stopped binding at all by
    step 33.
  - **Mode 2, a BAD DIRECTION — the floor cannot and should not touch it.** Rung 3 steps 50–51
    reproduce bit-for-bit with the floor in place (α 0.000, ‖R‖ 6.191e-03 vs 6.192e-03), and step 58's
    α 0.031 carries **no `L` flag at all**. For α to collapse with a 1e-8 floor the binding cell needs
    `|delta_k| >> 1e-8` — a large correction on a live cell, not a tiny `k`. No physically-sized floor
    rescues a step whose correction dwarfs the field; the escalation ladder is the right response, and
    the open question there is its **32× overshoot** (β 0.0293 → 0.9364 across two steps, then six
    steps of SER walking it back), not its trigger.
  - **Healthy clips survive, which the dumps could NOT show** (they were written only below cap 0.05).
    Steps 7 and 8 are preserved exactly; steps 9 and 44 shift by 0.1–0.3 % — exactly the `floor/k`
    perturbation the softened form predicts for a live cell. That is also how to read a clip: a live
    cell moves by `floor/k`, a dead one is bought out entirely.
  - **⚠️ Choose the floor by REPLAY, not by guess — the dead cells are a graded population, not one
    outlier.** Exempting the worst promotes the next: worst recorded cap 1.05e-09 unfloored, 1.6e-02 at
    a 1e-12 floor, 6.9e-02 at 1e-10, ~0.35 at 1e-08. An earlier estimate of "1e-12 should do it" was
    wrong by four orders. At 1e-06 the binding cell is a live one (`k` 4.7e-03, cap 0.84), so 1e-08
    keeps two orders of margin below anything physical here (inlet `k` 0.375, mesh median ~3e-2).
  - **Safe only because every consumer of the solved `k` clamps at zero.** The limiter's original
    justification — a negative `k` reaching a bare `sqrt` and poisoning the residual — no longer holds:
    `f1`, `f2`, `eddy_viscosity`, the production cap and the wall closures all clamp, and at `k < 0` the
    destruction term runs on the `k`-independent viscous ω branch, whose sign pushes `k` back up.
    **Re-check that before floating this limiter on another field.**

  **✅ BUILT AND MEASURED (2026-08-11): the per-cell PROJECTION removes the cap's failure mode entirely
  and does NOT make the case faster. Read this before proposing anything else aimed at the limiter.**
  `PositiveBlockProjection` clips each cell's own correction, `delta_i <- max(delta_i, -tau(k_i + floor))`,
  instead of scaling the whole step by the worst cell; applied before the cap, which then computes
  exactly `1`. It answers Mode 1 structurally rather than by decades of headroom, since a floored *cap*
  only postpones the ratchet — `(k_new + floor) = (1 - tau)(k + floor)` per capped step, whatever the
  floor.

  *Configuration, both arms:* `bfs3d`, native trailing inverse, `zerogradient` k wall, floor 1e-08,
  `equilibrate=False`, `refresh_on_cycles` 3, ILU(0) ×4, trailing sweeps 1, `coarse_eq_limit` 2000,
  restart 15, two rungs. **Both predate the probe/memory work merged in #188/#189**, so their wall
  clocks are not comparable to anything measured after it.

  | | projection | cap only |
  |---|---|---|
  | steps | 65 | 67 |
  | **Krylov cycles** | **329** | **329** |
  | wall | 2021 s | 2124 s |
  | positivity-limited (`L`) steps | **0** | 22 |
  | escalations | 5 | 4 |
  | mid-span `x_r/h` | 8.36 | 8.36 |

  **The cycle count is IDENTICAL, which is the number to read.** Wall differs by 4.8 % between runs
  taken at different machine loads and this case has no measured run-to-run spread; the linear-solve
  work did not move. **Do not cite this as a speed-up.**

  **Why it bought nothing: the cap was one DOOR into the cascade, not its cause.** The mechanism worked
  exactly as designed — the cap never binds once, against 22 times — and the same cascade reappeared
  through the line search's descent test instead. Rung 3 steps 53–55: α 0.000 twice with **no `L`
  flag**, β driven 0.0548 → 0.8769 (**16×**), cured at 0.8769, then six steps walking back to 0.0770 —
  the walk-back law again, `log(0.8769/0.0770)/log(1.5)` = **6.00** predicted, **6** observed. This is
  Mode 2 above, and it confirms that entry empirically: no treatment of the *limiter* reaches it.

  **So the lever is the RETURN, not the trigger and not the constraint.** Three separate attacks on the
  trigger side are now measured or refuted — gating `retry.on_alpha` on `binding_limit`, raising the
  floor, and removing the cap's global coupling altogether — and none moved the cycle count.

  **Keep the projection for ROBUSTNESS, not speed.** It eliminates the positivity lock-up outright (the
  mode that killed the 77-step march and the β = 16 runaways), and it is off by default and
  byte-identical off (`positivity_projection=False`, `BFS3D_K_POSITIVITY_PROJECTION` unset). It is not
  *dominated* — it removes a failure mode nothing else does — so it is not a deletion candidate, but it
  must not be sold as a performance feature.

  **⚠️ REFUTED — "rescaling promotes collapsed-`k` rows and inflates their corrections" is FALSE. Do not
  re-propose it (measured 2026-08-11, `k_row_scale_probe.py`).** The proposed explanation for why the
  `equilibrate` flag changes the step length was: symmetric rescaling divides row `i` by `sqrt(A_ii)`; a
  cell whose `k` has collapsed has a tiny diagonal there; so rescaling promotes that row to unit weight
  and un-scaling inflates the correction in exactly the cells the cap (a **minimum** over cells) is
  decided by. **The first clause is false**, so the rest cannot hold.

  *Configuration:* `bfs3d`, states `step-limit-04`/`-11` (both from `march-20260810-223702.log`: native
  trailing inverse, `equilibrate=True`, **`k` wall BC `dirichlet`**, rung 2 = Re/10, β 0.468 and 4).
  Exact `∂R_k/∂k` per cell by one-hot Jacobian-vector product — no materialization — 40 cells per decile
  of `k` plus the three cells the limiter is observed to bind on.

  | decile of `k` | median `k` | median `∂R_k/∂k` | scale `1/sqrt(|diag|)` |
  |---|---|---|---|
  | 0 | 8.974e-13 | 3.543e-05 | 168 |
  | 3 | 1.702e-03 | 1.246e-05 | 283 |
  | 9 | 4.893e-01 | 3.176e-05 | 177 |

  Across **twelve orders of magnitude in `k`** the diagonal moves ~1.1× and the scale 1.7×, peaking in
  the MIDDLE deciles — lowest-decile/highest-decile scale is **0.95×**. The binding cells sit **below**
  the median scale (12800 at 0.89×, 3181 at 0.51×, 22400 at 0.89×), i.e. rescaling mildly *demotes*
  them. The reason is structural: `∂R_k/∂k` is set by the destruction `β* ω V`, face transport and the
  pseudo-transient shift, **none of which vanish as `k -> 0`** — a collapsing `k` does not weaken its own
  equation.

  **The within-cell control is stronger than the table and removes the last confound.** Cell 12800's `k`
  differs by **six orders of magnitude** between the two dumps (3.082e-16 at β 0.468, 3.082e-22 at β 4)
  and its diagonal is **identical to four digits in both, 2.864e-05**. Same cell, same position, `k`
  varying a millionfold, `∂R_k/∂k` unchanged — so the flat decile trend is not an artifact of which
  cells populate the low deciles. (Cell 22400 reports the same 2.864e-05, so the two share a structural
  situation; cell 3181 differs at 8.753e-05.)

  **⚠️ The probe that first "measured" this question was structurally incapable of answering it — check
  for this failure mode before trusting any arm comparison.** It preconditioned with
  `CoupledShiftPolicy.make_preconditioner`, which is block-SIMPLE on `[u,v,w,p]` plus (with
  `method=None`) **identity on `k` and `ω`**. `equilibrate` lives only inside the engine's
  `FieldSplitAmgPreconditioner` (via `trailing_inverse`), so **both arms ran identical code** and
  returned identical corrections — reported as "no effect". Two ~2 GB Jacobians were built and discarded
  to produce it. The faithfulness gate could not catch it: the gate forms `operator(δ) − b`, which
  contains **no preconditioner at all**. *An A/B needs an assertion that the arms actually differ
  (`assert not array_equal(...)`), not only a gate that the system is right.*

  **⚠️ A Euclidean gate cannot establish solve accuracy on this system.** That probe read its
  8.4e-07 **2-norm** gate as "the march over-delivers by orders of magnitude against its 0.3 stop". The
  coupled Euclidean residual is ~100 % `ω` — the reason the row-scaled stop exists — so 8.4e-07 there is
  consistent with the `k` and velocity rows sitting at 0.1–0.3. Report the gate **per block**, in the
  measure the solve actually stopped in.

  **What the dumps DO show, and it points away from the preconditioner.** At fixed β, `δk` at the
  binding cell is unchanged to 7 digits between dumps while `k` falls ×100 per clipped step —
  `3.0816e-16 -> 3.0816e-22`, mantissa preserved, exactly `(1 − τ)` at `τ = 0.99`. The correction is not
  driving the collapse; **the limiter is ratcheting one cell toward zero and the global `min` lets that
  one cell of 23040 throttle the march.** The binding component is also ~13 orders below its block's
  norm — order 40× the round-off floor — so *which* arm's correction clips is likely not a reproducible
  quantity at any tolerance. The live target is the limiter's design (a `k`-relative floor, `τ` tapering,
  or a per-cell rather than global cap), not the hierarchy.

  **Equilibration: KEEP it, for conditioning, and stop expecting it to fix anything else.**
  - It improves per-cell block conditioning by orders of magnitude in the median, which is why it is kept.
  (Measured 2026-08-10 with `cell_block_scaling.py`; state, β and bundle not recorded, so the figures that
  quantified it are deleted — re-measure before relying on the size of the gain.)
  - It **cannot** change whether a cell block is singular. `det B̂ = det B / |b₁₁b₂₂|`, so
    `det B̂ = 0 ⟺ det B = 0`, and the coupling ratio `|a_kω a_ωk| / |a_kk a_ωω|` measures **identical**
    raw and rescaled (7.369e-04 both). Using it as the fix for the guard was wrong from the start;
    the "4 singular → 2 singular" reading that seemed to support it compared *different states*.
  - It does **not** help the worst cells: on `bfs3d` `state-00057`, symmetric `D B D` leaves cond at 1.22e12
  because it moves the imbalance from the rows into the subdiagonal (`[[1, 1.5e-9], [-1.1e6, 1]]`). The
  next two bullets are the same worst cell at that state.
  - **Independent row/column scaling would**: row-equilibrating the cell block gives cond 1.68e4,
    two-sided gives **2.41**, and both are *exact* rebracketings of `B⁻¹`. Unnecessary at float64
    and 2×2 (relative error is already 2.5e-16), but it becomes mandatory in **float32** — which is
    the GPU case this whole exercise is for.
  - The deeper issue is upstream: the k row has norm 8.8e-06 and the ω row 1.4e+03, so the
    **equations** are eight orders apart before any preconditioner sees them. That is what
    `RowScaledNorm` already recognizes in the convergence measure. Fixing the residual's row scaling
    would help the smoother, the coarsening and any factorization at once.

  **`‖B⁻¹‖ = 9.5e8` (`bfs3d` `state-00057`, symmetrically equilibrated) is not an error and no rescaling
  removes it.** The block is essentially
  lower-triangular (`∂R_k/∂ω ≈ 1.7e-12`), so ω is slaved to k there and a k correction legitimately
  produces one ~1e9 times larger in ω. Whether a *cell-local* smoother should apply that is the real
  open question, and it is the same "ω is not locally determined" theme as the Vanka campaign.

  **The tree is verified NEUTRAL on the shipped path.** A control march (PETSc ILU(0)×1) on all of
  this measured **58 steps / 282 cycles / final ‖R‖ 9.588e-06 / mid-span `x_r/h` 8.36** — identical
  in every reported digit to the recorded baseline. Wall was 1809 s against 1636 s — **10.6%, unexplained**, and
  on one run each. ⚠️ This case has **no measured run-to-run spread** to judge that against; the "~2%" it
  was previously compared to was a remembered figure with no configuration and has been deleted. The
  neutrality conclusion rests on the cycle count and the reported digits, not on the wall.

  ⚠️ **SIX OF THESE HARNESSES WERE DEAD ON ARRIVAL UNTIL 2026-08-14, AND NOTHING ANNOUNCED IT.** Each
  opened with `march_beta, _, description = STATES[name]` — a **positional** unpack of a record that has
  since grown to five fields — so every one of them raised `ValueError: too many values to unpack` before
  reaching its first line of work: `trailing_hierarchy_sweep.py`, `cell_block_scaling.py`,
  `trailing_block_conditioning.py`, `turbulence_smoother_sweep.py`, `zero_pattern_pivots.py` and
  `field_coupling.py`. All six now read the fields **by name**, which is what makes them survive the next
  field. **The lesson is about the keep-the-harness rule rather than about these six**: a harness kept in
  the repository is only re-adjudicable if it still *runs*, and nothing in the test suite exercises these,
  so a shared record can grow a field and silently retire the entire probe fleet. Every measurement in
  this file whose harness is named above was taken before that breakage and is unaffected — but any
  attempt to re-take one between the field's addition and this fix would have failed at startup.

  **Harnesses kept (all in `validation/bfs3d_openfoam/`):** `trailing_hierarchy_sweep.py` (the block
  alone, every arm), `cell_block_scaling.py` (per-cell conditioning, raw vs equilibrated),
  `singular_cell_probe.py` (which cells, from a checkpoint **or** a dumped block),
  `k_row_scale_probe.py` (exact `∂R_k/∂k` per cell by one-hot Jacobian-vector product, stratified by
  `k` — the harness that refuted the row-promotion explanation above, and the cheap way to ask whether
  any field's rows are what a rescaling promotes).
  `compare.py` gained `BFS3D_TURBULENCE_INVERSE`, `BFS3D_NATIVE_EQUILIBRATE`,
  `BFS3D_DUMP_TRAILING_BLOCK`, a `pbjacobi1` (undamped) smoother arm, march-log **archival**, and a
  banner that records the inverse and all its settings — **including the three knobs that install
  wrappers or change retention (`CHECKPOINT_KEEP`, `INNER_DUMP_ABOVE`, `DUMP_TRAILING_BLOCK`), which it
  used to read but never print, so two differently-configured runs could produce identical banners.**

  **⚠️ FOUR METHODOLOGICAL TRAPS, each of which produced a wrong write-up today:**
  1. **Probe the state the failure happens at.** The refusal fires from a *mid-step* refresh; step
     checkpoints and the inner-iterate dump both miss it (the inner observer writes only after an
     iteration succeeds). Three capture runs were wasted before dumping the operator *before* the
     build, which needs no state and no shift pairing — and even that missed twice, first by
     wrapping only the factory when the refresh goes through `refactor_block`.
  2. **Pair the operator with the right β.** Probing state-N with state-N's β when the failing
     refresh uses state-N+1's is the recorded trap; sweep β instead.
  3. **Never quote an arm at one smoother-sweep count.** The standard prolongator was recorded as
     "void" from its 4-sweep numbers; at 8 it is the best native arm on the scalar problem. The rule
     cuts both ways — on the flow saddle it fails at 4 sweeps and fails *worse* at 8, which is what
     distinguishes an under-smoothed arm from an amplified one.
  4. **A block-alone probe ties where a march separates.** Raw and equilibrated are both 2 cycles on
     the block and behave differently in a march.

  **✅ ANSWERED: the cap binds on ONE cell, it is a step-corner cell whose `k` is already numerically
  zero, and it is NOT one of the ill-conditioned ones.** Measured from twelve `BFS3D_DUMP_STEP_LIMIT`
  dumps taken on the native march at rung 2 (`equilibrate=True`, `sweeps=4`, `aggressive_levels=1`,
  `spectral_damping=False`, `refresh on cycles 3`, `retry on cycles / alpha 10 / 0.01`, PC β floor
  0.05), the operator swept over β 0 … 0.4. The probe reproduces the recorded cap exactly
  (3.761982e-03 both ways), so the capture is reading the quantity the march acted on.

  | dump | cap | cell | room | `k` | `dk` | next cell | next room |
  |---|---|---|---|---|---|---|---|
  | 04 | 3.7620e-03 | **12800** | 3.80e-03 | 3.08e-16 | −8.11e-14 | 22400 | 2.60e-01 |
  | 05 | 1.4517e-04 | **12800** | 1.47e-04 | 3.08e-18 | −2.10e-14 | 12840 | 2.09e+00 |
  | 08 | 1.0516e-07 | **12800** | 1.06e-07 | 3.08e-20 | −2.90e-13 | 12840 | 1.12e-01 |
  | 11 | 1.0516e-09 | **12800** | 1.06e-09 | 3.08e-22 | −2.90e-13 | 12840 | 1.12e-01 |

  - **One cell out of 23040 throttles the whole march,** and it is not a crowd: at dump 04 exactly
    **one** cell sits within 100× of the tightest room, the runner-up 68× behind; by dump 11 the gap is
    **eight orders**. Cell 12800 owns ten of the twelve dumps and every one from step 25 on.
  - **`k` there ratchets by exactly 100× per step** (the per-step evidence is the checkpoint mantissa
    below; these dumps are non-consecutive) — 3.08e-16 → 3.08e-18 → 3.08e-20 → 3.08e-22 — which
    is `1 − τ` at `τ = 0.99`, the same factor the cap collapses by. Against a mesh median `k` of
    **2.97e-02**, so the binding cell's `k` is ~20 orders below typical. It is not small; it is zero.
  - **`dk` stays ~1e-13 throughout while `k` collapses.** ⚠️ **The inference drawn from this — "the
    cell's `k` equation has no root at `k ≥ 0`" — is WRONG, and was refuted the same day by direct
    measurement. It is the opposite.** Holding every other field at the `step-limit-11` iterate and
    scanning `R_k[12800]` over `k_P` gives a straight line to seven digits,
    `R_k = A·k − b` with `A = 2.8740e-06`, `b = 5.7236e-20`, so the **root is `k* = +1.99e-14`,
    strictly positive** — and the iterate sits **eight decades BELOW its own root**. Newton should be
    pushing `k` *up*.

    Production, destruction and the `Dirichlet(0)` wall term all vanish linearly at `k = 0`, so `k = 0`
    solves the row exactly when the neighbours are 0. (⚠️ An earlier version called the corner `k` system
    "**homogeneous and an M-matrix**" — the M-matrix half is **false**: `A_kk` carries **14 non-positive
    diagonals**, min −9.03e-07.)

    **Why the returned correction is wrong there, corrected — it is Krylov truncation, NOT roundoff.**
    An adversarial re-derivation (independent colour-recovery of `A_kk`, verified row-wise against a
    `jax.vjp` to 5e-16) settles the mechanism:
    - `R_k[12800] = −5.72e-20` is **deterministic and 13 orders ABOVE its own row's floating-point
      floor** (~1.3e-33). "Roundoff floor" was the wrong description.
    - What is true is stronger: `|R_row| / ‖R_k‖₂ = 2.94e-16 ≈ machine epsilon`, and
      `|R_row| / ‖R‖₂ = 2.15e-22`. **Any Krylov solve stopping on a relative-residual test would need
      ~1e-22 relative accuracy in float64 to resolve this row — unreachable in principle, not merely
      under-solved.** Measured: `‖A_kk·dk − rhs_k‖/‖rhs_k‖ = 1.0055`, i.e. the returned correction does
      not reduce the k-block linear residual at all, and dumps 08/09/10 sit at a **bit-identical state**
      yet return `dk[12800]` of −2.901e-13, −8.825e-14, −1.567e-14, an **18× spread**.
    - The returned `dk` also provably violates its own row for *every* admissible shift: the row demands
      `βD·(k_ref + 2.901e-13) = −3.958e-19`, and the left side is ≥ 0 while the right is < 0.
    - ⚠️ **"14× larger than the exact local step" was wrong** — that froze the neighbours. Given the
      dumped neighbours the row-consistent `dk` is −1.52e-13, still negative; it is the **exact k-BLOCK
      solve** that comes out positive, at `+3.90e-10` (β = 0) through `+1.90e-15` (β = 10), i.e.
      **positive at every β from 0 to 10** while the dumped value is negative and 3–5 orders off.

    `positive_block_limit` then computes a purely **relative** room `0.99·k/|dk|` and lets that cell veto
    the global step.
    - **⚠️ THE MOST USEFUL LEAD OF THE LOT: at β ≥ 0.5 an exact k-block solve produces NO binding cell at
      all (cap = 1.0).** Only at β ≤ 0.05 does an exact solve give a tight cap, and then on a *different*
      cell (7.6e-14 at cell 2679). So the cap is **not intrinsic to the `k` equations at a healthy
      shift** — it is a low-β-plus-inexact-solve artifact, which points back at the low-β conditioning
      wall rather than at the closure.

    **The ratchet is proved by the mantissa.** `k[12800]` reads `3.0816e-22` at `step-limit-11` and
    `3.0816e-208 / e-210 / e-212` at checkpoints 120 / 121 / 122 — **the same five digits, 190 decades
    apart**, i.e. ninety-five successive ×0.01 cuts, while `u/v/w/p` report a relative change of
    exactly 0.
  - **It is not one cell, either: 1876 of 23040 cells sit below `k = 1e-6`** at this state, against an
    OpenFOAM global minimum of 1.672e-6. ⚠️ **BOTH HALVES OF THAT COMPARISON WERE WRONG, AND THE
    CONVERGED ROOT HAS NO SUCH DEFICIT — see the resolution below.**
  - **Geometry: it is a step-corner cell, and its mirror is the runner-up.** 12800 sits at
    `(0.0007, −0.0099, 0.0009)` and 22400 at `(0.0007, −0.0099, 0.0391)` — bottom wall (`y ≈ −h`),
    immediately behind the step (`x ≈ 0`), against each side wall (span `4h = 0.04`). The stagnant
    bottom/side-wall corner, i.e. the same corner-separation region that makes the full-span
    reattachment (16.14 here) disagree with the mid-span one (5.34). ω there is 4.01e+05, a wall value.
  - **The tightness ranking tracks the wall-face count — on n = 4 cells, one dump, one state.** Counted
    off `face_patches`, the four tightest cells carry **3, 3, 2, 1** no-slip faces out of six:

    | cell | room (dump 04) | boundary faces |
    |---|---|---|
    | 12800 | 3.80e-03 | `lowerWall`, `lowerWall`, `sideWalls` |
    | 22400 | 2.60e-01 | `lowerWall`, `lowerWall`, `sideWalls` |
    | 12840 | 1.38e+00 | `lowerWall`, `sideWalls` |
    | 13129 | 2.17e+00 | `sideWalls` |

    The two cells that own the cap are **trihedral wall corners** — half of every face is a no-slip
    wall, the two `lowerWall` faces being the floor and the vertical step face. **Hypothesis, not yet
    measured:** the near-wall `k` closure is per-wall-cell, and its production is area-averaged over a
    cell's wall faces while the destruction `β*kω_wall` is not obviously averaged the same way — so a
    three-wall-face cell could take up to 3× the destruction against one cell's worth of production.
    **❌ REFUTED the same day by reading both sources** — our reduction is already OpenFOAM's:
    `wall_cells` is a `jnp.unique`, so a cell appears once however many wall faces it has, and
    `wall_shear_rate` area-averages `Σ|S_f|r_f / Σ|S_f|` over them, which is exactly
    `patchFieldsToWallCellField` in OpenFOAM 13's `wallCellWallFunctionFvPatchScalarField`. Nothing is
    summed per face. The ranking most likely reads **stagnation** — three wall faces means the deepest
    dead zone and so the least production — not double counting. See `.claude/rules/turbulence.md` for
    the two differences that are real, of which one bears directly on the lever below.
  - **NOT the ill-conditioned cells — the standing hypothesis is refuted at this state.** **Zero**
    singular blocks at every β from 0 to 0.4. The binding cells run cond 3.6e5 … 2.9e7, ranks
    1088–2165 of 23040, and **none** of them is among the twelve worst-conditioned. `cond > 1e6`
    catches 50% of them against an 8.7% base rate — mildly enriched — but `cond > 1e9` catches none,
    and the separately characterised cond ~1e12 / ‖B⁻¹‖ ~9.5e8 cells are absent entirely (the worst
    here is 2.9e7 / 6.6e6). **Caveat on comparing those two numbers:** the 1e12 figure was measured on
    `state-00057` under *symmetric equilibration*, and this is a different state, raw — so this
    refutes the coincidence at this state rather than retiring the 1e12 finding.

  **✅ RESOLVED (2026-08-10, by re-running the shipped march end to end): the `k` trough is a REYNOLDS-
  CONTINUATION TRANSIENT, and the converged root does not have it.** At the converged target-Re root
  (step 58, `R` 9.588e-06, `x_r/h` 8.36) there are **ZERO** cells below `k = 1e-6`: `k_min` 1.2997e-05,
  median 0.7017 against OpenFOAM's 0.7468, a ratio of **0.98**. Cells below 1e-6 go 190 (step 37) → 130
  → **0** (step 46) → 0 at convergence.
  - **Both halves of the "1876 vs 1.672e-6" comparison were wrong.** It is **cross-Reynolds** —
    `state-00122` is rung 2 (ν = 1e-4), the OpenFOAM field is target Re — and 1.672e-6 is the **steady**
    OpenFOAM run's minimum, the run this case's own docstring rules out as a reference. The valid
    transient reference has `k_min` **6.2247e-03**.
  - **The mechanism is the continuation's own doing.** `y* = β*^0.25 √k d/ν` scales as **1/ν**, so at the
    Re/100 anchor `wall_function_weight = tanh((y*/y*_lam)⁴)` is ~1e-7 — **the log-layer wall production
    is switched off even at fully turbulent `k`** — while `ω_wall = 6ν/(β₁d²)` is **100×** larger, so
    `β*kω` destruction is 100× the target's. Homogeneous, linear, destruction-dominated ⇒ `k` decays.
    Raising Re reverses both: the weight median goes 5.6e-20 (rung-1 root) → 1.9e-10 → **1.000** (target).
  - The rung-1 root is **correctly** laminar (`ν_t/ν` median 1.7e-5 at `Re_h = 100`), and **1732 of the
    1876 cells (92%) are inherited from it**.
  - **Population is pre-stall, depth is stall-caused**: `#k<1e-6` is 1876 by dump 04 and frozen there for
    all 97 remaining steps, while `k[12800]` falls 3.08e-16 → 3.08e-212 over the same span.
  - **The stalled arm is `BFS3D_TURBULENCE_INVERSE=native`; the default `petsc` arm walks the same trough
    and recovers** — agreeing with the exact-solve finding that at β ≥ 0.5 there is no binding cell.
  - ❌ **REFUTED — the initialization.** `hybrid_initialize` gives a **uniform** `k = 0.2489585`, zero
    cells below 1e-6.
  - ❌ **REFUTED — the "ω is exactly 10×, still at its 60ν seed" lead.** `ω[12800] = 400910.4` **is**
    `6ν/(β₁d²)` at **rung 2's own** ν = 1e-4 to 16 digits (all 4490 wall cells within 4.8e-5). The three
    rungs give 4.0091e6 / 4.0091e5 / 4.0091e4 — the "10×" compared a rung-2 state against the
    **target-Re** formula. Those rows are converged; the `60ν` seed was replaced by `omega_wall` in
    `c324021` (2026-07-23), well before these checkpoints.
  - **The sustaining/ambient source has no motivating evidence left**: it targets a defect absent from
    the converged root. The defensible levers are the absolute floor in the limiter, and not anchoring
    the ladder below the Reynolds number at which the wall function turns itself on.
  - **Genuinely open, and now the real physics question — but BOTH numbers needed a definition before
    they meant anything (harness: `validation/bfs3d_openfoam/wall_layer_comparison.py`).**
    - "**First wall layer**" spans a 4× range of defensible meanings, and the choice matters more than
      the discrepancy: **all wall-adjacent (4490 cells) is 1.054**, the **finest layer (1600) is 0.459**,
      the **floor behind the step (640) is 0.342**. The recorded 0.164/0.46× is the *finest layer*. Side
      walls, 64% of wall-adjacent cells, agree at **1.113** and do not participate.
    - `x_r/h` **7.24 and 8.36 are both grid stations, two apart.** The floor's stations near
      reattachment are `… 6.728, 7.243, 7.787, 8.361, 8.966 …`, local spacing **0.375–0.749 h**, so the
      metric's resolution is about half a step height and "15%" is a two-cell offset. A sub-cell
      interpolated crossing puts the reference at **~7.6**, i.e. the quoted 7.24 carries a systematic
      −0.36 h truncation, and the interior spanwise columns scatter by **sd 0.13 h**. A defensible
      reference is **x_r/h ≈ 7.6 ± 0.3**; the gap survives it (~1.0 h, ~13%) but is smaller than quoted.
    - **The wall closure is NOT the cause.** The `Dirichlet(0)` vs `kqRWallFunction` (zero-gradient)
      difference is real but worth only ~7.4% of local destruction (k ×0.93–0.98); `nut_wall` is
      *algebraically identical* to `nutkWallFunction`; every verified wall-closure difference multiplies
      to **~0.87–0.93, not 0.46**. The wall-cell `k` budget is **transport-dominated**
      (|transport|/production ≈ 1.16), so that `k` is inherited, not locally made.
    - **Where it is inherited from, and this is the causally clean part:** the deficit is already there
      **upstream of the step** (x/h −2.85 … −0.15, no recirculation), first-cell `k` **0.54–0.68×** with
      the channel core at parity (0.987), and the lip-shed rows carry the same ratios. In wall units
      `k⁺` ≈ 1.3–2.4 against OpenFOAM's 2.4–4.0, DNS channel ≈3.9–4.4, and the SST log-layer equilibrium
      `1/√Cμ` = 3.33 — **low against both references**, which is what makes it a defect rather than a
      difference. It seeds a shear layer carrying 20–25% less `ν_t` through x/h 0.3–2.
    - ❌ **Exonerated with evidence:** the momentum scheme (aquaflux is *more* dissipative yet its shear
      layer is *thinner* — wrong sign); the SST shear limiter (identical branch in both); the mesh
      (cell-for-cell identical, max centroid distance 4.7e-9 m); and first-order upwind on k/ω (false
      diffusion ~5% of `ν_t S²` at x/h 0.4, <1% beyond — an order of magnitude too small).
    - ⚠️ **THE EXPANSION RATIO CHANGES WHICH NUMBER LOOKS WRONG, AND NEITHER OF US HAD ACCOUNTED FOR
      IT.** The famous benchmarks are ER 1.125–1.2 (Driver & Seegmiller 1985: 6.26 ± 0.10; Le, Moin &
      Kim 1997 DNS: 6.28) and **must not be used as the target here — this case is ER = 2**, and the
      published trend is that larger ER *lengthens* the normalized bubble (Armaly et al. 1983, quoting
      Durst & Tropea 1981). **At ER = 2 the 2D references cluster at 8.0–8.8**: Pont-Vílchez, Trias,
      Gorobets & Oliva (2019, *JFM* 863) DNS gives **X_r = 8.8h** at Re_τ = 395 (verified independently
      of the agent that reported it — a later LES cites it as its reference, calling its own 8.15h a
      7.3% under-estimate); Durst & Tropea (1981) 8.5 at Re_H 1.5e4; Rothe & Johnston (1975) 7.8.
      **So aquaflux's 8.36 sits INSIDE the ER = 2 band and OpenFOAM's 7.24 sits below it** — the
      opposite of the "aquaflux over-predicts by 15%" framing this section started from.
    - ⚠️ **"SST is a known under-predictor" is also wrong.** On Driver & Seegmiller, NASA's turbulence-
      model resource puts SST at x/H ≈ **6.50** against 6.26 ± 0.10 — a 4% **over**-prediction,
      reproduced across four codes; Menter (1994) reports 6.5 himself. The classic under-prediction is
      **k-ε** (Menter's Jones–Launder 5.5), and even the widely-repeated "20–25%" is disowned by
      Thangam & Speziale (ICASE 91-23) as a resolution artifact.
    - **What is genuinely unresolved is the FINITE SPAN.** span/h = 4 is 2.5× below the span/h > 10
      that de Brederode & Bradshaw (1972) require for two-dimensionality at the centreline (quoted
      verbatim in Jovic & Driver 1994, NASA TM 108807). Lower aspect ratio **shortens** reattachment —
      direction unanimous across every source found — which would pull both codes below the 2D band.
      **No open-access `x_r`-vs-aspect-ratio numbers exist**, so the magnitude is unknown and neither
      7.24 nor 8.36 can be called correct.
    - ⚠️ **One anomaly worth chasing: our spanwise profile has the WRONG SIGN against the literature.**
      Both codes here reattach *later* at the outer slabs (10.28) than in the interior (7.24), but the
      two published spanwise measurements go the other way — Sugiyama et al. (2013) find near-side-wall
      reattachment ~60% of the centreline value at AR 16, and Armaly, Li & Nie (2003) find a *minimum*
      near the side wall. Both are laminar/transitional, so the transfer is uncertain, but this is the
      one place our result contradicts published structure rather than merely differing in magnitude.
    - **Untested and the largest remaining unknown: grid convergence.** One mesh exists.

  **⚠️ THE LEVER, as corrected earlier — with the scan's reach stated.** The measurement below is a **1-D scan of
  one row at one iterate with every other field frozen**. It establishes a strictly positive root of cell
  12800's frozen-field `k` row, which removes the motivation for a projection *at this state*. It does
  **not** establish the sign of the coupled Newton direction, nor anything about the other 1875 collapsed
  cells, nor that no state has a negative root. Two defects remain, and they are different from
  each other:
  1. **`positive_block_limit` is a purely RELATIVE rule** — `room = tau·k/|dk|` — applied to a field
     whose physical floor is 0. A relative rule has no lower bound, so a roundoff-level `dk` in an
     already-collapsed cell produces an arbitrarily small cap. An **absolute floor**
     (`room = tau·(k + k_abs)/|dk|`) fixes it, is pure globalization, sits off the IFT path entirely,
     and — *provided the cap is inactive at the converged state* — changes nothing at the solution.
     Scale `k_abs` off the block (`eps·max(k)`), not a constant, so it carries `k`'s units.
     **⚠️ IT IS NOT SAFE ALONE — clamping `k` in the closure is a PREREQUISITE, in the same change.**
     Relaxing the cap lets `k` go slightly negative, and two sites then misbehave badly (verified):
     `SSTModel.f1`/`.f2` (`sst.py:136,178`) take a raw `jnp.sqrt(k)` and NaN the whole residual, where
     every closure in `boundary.py` uses `safe_sqrt(jnp.maximum(k, 0.0))`; and `OmegaProduction`
     (`sources.py:288-294`) divides by `jnp.maximum(nu_t, 1e-30)`, so a negative `ν_t` selects the floor
     and the cap becomes ~**−1e23** at the measured corner values. `KProduction`'s cap
     (`sources.py:209`) also flips sign, turning production into a sink. **Unbuilt and unmeasured.**
  2. **A large part of the near-wall field has laminarized** (1876 cells below 1e-6). That is the real
     physics defect and the absolute floor does not address it. The candidate is a **sustaining /
     ambient source**. ⚠️ **Both the form and the citation first recorded here were WRONG; corrected:**
     - The literature terms are **constant**, and there are **two** of them, one per equation:
       `k: + β* ω_amb k_amb` and `ω: + β ω_amb²`, where `ω_amb` is an **ambient constant, not the local
       ω**. Source: **Rumsey & Spalart, AIAA J. 47(4), 2009, 982–993** (NASA calls it `SST-sust`).
       **Spalart & Rumsey 2007** — cited here originally — proposes *floor values*, not source terms.
     - The form written here first, `+β* k_amb ω` with the **local** ω, is a different term: it makes
       the destruction `−β* ω (k − k_amb)`, pinning `k ≥ k_amb` *uniformly including deep in the
       boundary layer* where ω is O(1e5). At the `bfs3d` corner that is ~80× the literature term and
       would override the `Dirichlet(0)` wall condition the case sets — which is exactly what Rumsey &
       Spalart warn against. It also is **not** diagonal-free: `∂R_k/∂ω = −β* k_amb` is an off-diagonal
       the frozen AMG operator would not see. The literature (constant) form does have both properties.
     - **Motivating evidence is weak.** The `k` deficit is measured at an *unconverged* state whose ω
       rows have not moved off initialization, and OpenFOAM reaches min `k` 1.672e-6 on this mesh with
       **no** sustaining term at all. It would also make two existing assertions tautological
       (`test_coupled_mass_flow.py:138`, `test_coupled_periodic_channel.py:124`, both of which assert
       `min(k) > 1e-6` and document themselves as *non*-tautological).
     - **Order of work: do the step-limiter fix first and re-measure.** If the march then converges with
       `min(k)` near OpenFOAM's, this has no motivating evidence left. Unbuilt and unmeasured.

  Two further leads found while measuring this, both unverified: `ω[12800] = 4.0091e5` is **exactly
  10×** the `omega_wall` value `6ν/(β₁d²)` its own residual imposes, which is the `60ν` initialization
  value never relaxed (so this state's `ω` rows are unconverged too); and `SSTModel.f1`/`.f2`
  (`sst.py:136,178`) use **plain unclamped `jnp.sqrt(k)`** where every closure in `boundary.py` uses
  `safe_sqrt(jnp.maximum(k, 0.0))` — clamping those two would make a transiently negative `k`
  survivable, which is what an absolute-floor cap needs.

  **The rest of this paragraph is the superseded framing, kept because the OpenFOAM comparison in it
  still stands on its own:** `kOmegaSSTBase.C` ends every `k` solve with
  `bound(k_, kMin_)`, and `bound` replaces a negative cell by the **average of its neighbours** before
  flooring at `kMin` (`isf = max(max(isf, fvc::average(max(vsf,min))*pos0(-isf)), min)`). It also
  *replaces* the `ω` row in wall cells (`matrix.setValues`) and lags `G`, so its wall-cell `k` equation
  is linear with a non-negative source. We do the opposite on both counts: both terms stay live
  functions of `k` in one Newton residual (deliberately — a frozen `ω` degenerates the row), and we
  constrain the **step** rather than projecting the state. Constraining the step is what locks up; a
  projection cannot. That reframes the choice, and it is the one to make with the user:
  a positivity **projection** after the step (OpenFOAM's answer, and it changes the forward path but
  not the root provided the floor is inactive there) against keeping the step constraint and finding
  why that cell's row is hard to satisfy (its root is **positive**, at `+1.99e-14`). Neither is built.

  **⚠️ (2026-08-10, LATER STILL) THE `k` WALL BC A/B, RUN AS A CONTROLLED PAIR — and the crash is NOT
  fixed by any of it.** Two full 3-rung marches from the initial state, `BFS3D_TURBULENCE_INVERSE=native`,
  everything identical but `BFS3D_K_WALL`, both carrying the four negative-`k` clamps and
  `stop_on_limit_stall=3`:

  | | `dirichlet` (control) | `zerogradient` |
  |---|---|---|
  | rung 1 | 14 steps, 43 cycles, ‖R‖ 5.882e-06 | 14 steps, 45 cycles, ‖R‖ 7.821e-06 |
  | rung-2 step it locks at | **25** | **35** |
  | rung-2 ‖R‖ at lock | **7.316e-02** | **1.257e-02** |
  | rung-2 steps before the guard ends it | 15 | 24 |
  | outcome | `RuntimeError` from `solve_reynolds_continuation` | same |

  - **❌ Zero-gradient does NOT cure the lock-up.** The same fraction-to-the-boundary ratchet appears,
    same β ceiling of 16.0, same ~100×-per-step cap collapse — just later. Consistent with the
    term-by-term account that put the whole wall closure at ~0.87–0.93.
  - **✅ But it is worth keeping on the evidence: a 5.8× lower residual and 10 more steps of rung 2**,
    and the attribution is clean because the clamps are verified neutral (below) and both arms had the
    guard. It also produced five consecutive *uncapped* full steps in rung 2, which the control never
    does.
  - **✅ THE CLAMPS ARE EXACTLY NEUTRAL, end to end.** The control's rung 1 is **14 steps / 43 cycles /
    5.882e-06 / ‖R₀‖ 3.3078e-01 — identical in every digit to the pre-clamp recorded baseline** — and its
    rung-2 steps 25–29 reproduce that baseline's trajectory exactly (step 25 β 0.4682, ‖R‖ 7.567e-02,
    `a_min` 0.004, cap 3.76e-03; then 1.00e-05 → 1.95e-06 → 1.95e-08 → 1.95e-10). The per-guard
    `array_equal` unit tests said this; a real coupled march now confirms it.
  - **✅ `stop_on_limit_stall` works, on two independent arms.** The control's rung 2 ends at **15 steps
    instead of the recorded 108**; the zero-gradient arm's at 24. ~90 dead steps saved each time, and the
    run now fails with a `RuntimeError` naming the rung instead of grinding out `MAX_STEPS`.
  - **⚠️ THE DEAD GRIND MOVED RATHER THAN VANISHED — a real coverage gap.** Once the guard ends the
    segment, `solve_coupled` falls through to the finishing solve (`ImplicitNewtonSolver`), which has **no
    equivalent guard**, and that grinds at α 0.000 / 0 cycles with the residual frozen until `max_steps`.
    `stop_on_limit_stall` covers `forward_march` only. Fixing that is the next march-level item.
  - The step-limit dumps now carry **β and the anchor** (`cap 2.2923e-06 beta 1.754 anchor yes`), so the
    falsifier named against the earlier diagnosis is closed: these dumps can be paired with the linear
    system that produced them.

  **✅ GATES GREEN on all of the above** (CSR level operator, native trailing inverse, and both march
  changes), with the tiers named because a default-on march guard reaches further than the fast gate:
  - fast gate **967 passed / 1 skipped** (899 unit `-n auto`, 68 integration `-n 1`);
  - `test_coupled_rans` + `test_coupled_amg` + `test_coupled_field_split` + `test_reynolds_continuation`
    — 33 tests, 18 of them `slow` — **33 passed**;
  - `test_coupled_lu` + `test_coupled_ilut` — **15 passed**. These two matter and were nearly missed:
    they drive `forward_march` with `step_control` + `precondition_step`, so they pick up
    `stop_on_limit_stall` exactly as the four above do, and they are in neither the fast gate nor the
    list a first pass would think to run;
  - **`-m validation` — 18 passed.**

  **The trap, recorded because it nearly landed:** a default-on guard on a *shared* seam is not covered
  by "the tests for the subsystem I changed". `forward_march` has one production caller, but everything
  that reaches `solve_coupled` with an observer picks the new default up. Enumerate by **who calls the
  seam**, not by which file the change is in.


## Faithful smoothed aggregation — matching PETSc GAMG

- **⚠️ (2026-08-09): making the JAX-native multigrid a FAITHFUL smoothed aggregation, so a
  comparison against PETSc GAMG means something. Uncommitted work sits on `claude/block-aware-aggregation`.**
  Read this before touching `solve/multigrid.py`'s aggregation path.

  **⚠️ THE EQUILIBRATION HYPOTHESIS WAS WRONG, AND MEASURING IT IS WHAT SETTLED THE COMPARISON
  (2026-08-09).** The suspicion recorded here was that ours and PETSc's were not built on the same
  matrix — `AmgVCycle` calls `equilibrate_cell_major` before handing the operator to PETSc, so GAMG
  coarsens a **unit-diagonal** matrix, while `build_convection_hierarchy` coarsened the raw block,
  whose diagonal on `bfs3d`'s `[k, ω]` slice spans **7.96e5×** (1.26e-06 … 1.0). That description of
  the code is accurate, and the σ_max figures reproduce exactly (**4.365** raw against **2.832e3**
  equilibrated, so the standard prolongator step `1.4/σ_max` is ~0.32 for us and ~5e-4 for PETSc).
  The **inference** from it — that no ours-vs-PETSc number meant anything until it was fixed — does
  not survive. `equilibrate=True` is now built and measured, and it makes our hierarchy **worse**:
  8 → 11 cycles on the RCM arm, 5 → 10 on the plain-aggregation MIS arm.

  **A symmetric equilibration is a SIMILARITY TRANSFORM, so the SPECTRAL half of the setup is
  invariant under it.** `D̂⁻¹Â = S⁻¹(D⁻¹A)S`, so both spectral estimates are unchanged and the
  damped-Jacobi smoother `x += (ω/λ)D⁻¹r` is unchanged with them. Measured on a badly scaled chain:
  the **fine** level's `lam_max` is 1.966 raw against 1.976 equilibrated (equal to power-iteration
  accuracy) while the **coarse** level's is 1.81 against 15.3. The **tentative prolongation** — a
  fixed 0/1 aggregate indicator, which does not transform with the operator — is non-equivariant.

  **❌ "EQUILIBRATION CHANGES THE GRAPH THE COARSENING SEES" — PROPOSED AND REFUTED BY MEASUREMENT
  (2026-08-12). Do not re-propose it; the falsifying run is cheap and is in the tree.** The reasoning
  was sound as far as it went: `symmetrically_equilibrate` returns `(diagonal @ a @ diagonal).tocsr()`,
  **a sparse product stores only entries whose result is nonzero, so it DROPS every explicit zero**
  (verified in isolation), and at `strength_threshold = 0` the aggregation reads the graph and nothing
  else. The dropped counts are large and real — on `bfs3d` at the target-Re cold initial field, the
  trailing `[k, ω]` block loses **26.8 %** of its stored entries to equilibration (5 245 488 → 3 839 276,
  1 406 212 exact zeros) while the leading block loses **0.0 %** (5 662 of 20 981 952).
  **It does not reach the coarsening, because the cell collapse absorbs it**
  (`validation/bfs3d_openfoam/equilibration_graph_effect.py`, running the real `_cell_graph` /
  `_aggregation_edges` / `_mis_aggregate`):

  | trailing `[k, ω]` | cell edges | aggregates | max aggregate |
  |---|---|---|---|
  | raw | 644 166 | 2300 | 45 |
  | equilibrated | 643 967 | 2302 | 45 |

  **199 of 644 166 edges — 0.03 %.** `_cell_graph` sums `\|A_ij\|` over each `block_size²` block, so a
  cell pair keeps its edge if **any** of its entries is nonzero; the dropped degrees of freedom are
  scattered across blocks rather than clustered within them, so almost every edge survives. The coarse
  space is effectively identical. The leading-block control loses **1** edge and its partition is
  bit-identical.
  **So the similarity-transform argument stands, and the recorded explanation — the non-equivariant
  tentative prolongation and the coarse operator it builds — remains the live one for the 5 → 10.**
  **What survives, and it is worth keeping:** *the aggregation is robust to explicit-zero pruning while
  an incomplete factorization is not.* The smoother takes its pattern from the stored pattern directly,
  with no cell collapse to absorb the loss — which is why the same pruning is fatal to `column_reach`'s
  pressure column (see the per-column reach entry under `sparse_jacobian.py`) and inert here. When
  reasoning about a pattern change, **ask which consumer sees it**: a nodal coarsening and a zero-fill
  factorization have opposite sensitivities.

  **Reference numbers, re-measured on the current code** — `bfs3d` `state-00057`, PC β 0.05, the
  `[k, ω]` block **ALONE** (46080 dofs, 4.20M nnz, 91/row), GMRES restart 15 to rtol 1e-8 on the
  TRUE residual, random right-hand side; harness
  `validation/bfs3d_openfoam/trailing_hierarchy_sweep.py`:

  | preconditioner | coarse eq | cycles |
  |---|---|---|
  | PETSc GAMG, plain aggregation, ILU(0) ×4 | 432 | **1** |
  | PETSc GAMG, plain aggregation, ILU(0) ×1 | 432 | **2** |
  | **PETSc GAMG, plain aggregation, point-block Jacobi ×4** | **432** | **2** |
  | ours, MIS, standard prolongator, ×8, EQUILIBRATED | 2150 | 4 |
  | ours, MIS, plain, ×4, raw | 2150 | 5 |
  | ours, RCM, symmetric-part prolongator, ×4, raw | 598 | 8 |
  | ours, MIS, plain, ×4, EQUILIBRATED | 2150 | 10 |
  | ours, RCM, symmetric-part prolongator, ×4, EQUILIBRATED | 598 | 11 |
  | ours, MIS, standard prolongator, ×4, raw / EQUILIBRATED | 2150 | 44 / 44 — both fail (true rel 1.0) |

  **Read the THIRD row, not the first — the like-for-like arm.** Comparing our Jacobi-class smoother
  against PETSc's incomplete-LU measures the smoother, not the hierarchy, and an ILU sweep is a much
  stronger and much less parallel unit of work.

  **✅ THE GAP IS CLOSED: matched to PETSc's algorithm, our V-cycle equals it — 2 cycles against 2,
  on a 436-equation coarse space against 432.** Two differences accounted for the whole of it, and
  neither was scaling. Both were found by reading `agg.c`/`misk.c`/`rich.c` against our code rather
  than by tuning:

  | arm, `[k, ω]` block alone, matched smoother class | coarse eq | cycles |
  |---|---|---|
  | PETSc GAMG, plain aggregation, point-block Jacobi ×4 | 432 | **2** |
  | ours, MIS ×4, damped, no aggressive level | 2150 | 5 |
  | ours + **aggressive level** ×4, damped | **436** | 10 |
  | ours + aggressive level ×8, damped | 436 | 3 |
  | ours + aggressive level + **undamped** ×2 | 436 | 11 |
  | **ours + aggressive level + undamped ×4** | **436** | **2** |

  1. **The aggressive first level, which PETSc applies BY DEFAULT and we did not.**
     `build_amg_vcycle` never sets `pc_gamg_aggressive_coarsening`, and GAMG's default is
     `aggressive_coarsening_levels 1` with `use_aggressive_square_graph` — so on level 0 it coarsens
     the **squared** graph. That is the entire 5× coarse-space difference: ours coarsened 21×, GAMG
     107×, and with `aggressive_levels=1` ours lands at 436 equations against PETSc's 432.
     **Note `use_aggressive_square_graph` and `aggressive_mis_k` are ALTERNATIVES, not a pair** —
     `mis_k` is read only in the non-squared branch (`agg.c:1314`), so at the default the coarsener
     is plain MIS at distance **1** on the squared graph.
  2. **PETSc's level smoother is UNDAMPED.** `mg_levels_ksp_type richardson` at its default
     `scale = 1.0` (`rich.c:277`) is `x += D⁻¹(b − Ax)`, while ours relaxed by `omega/lam_max` with
     `omega = 0.8`. That is never a *smaller* factor than 0.8 and usually much smaller: `D⁻¹A` has
     unit diagonal blocks, so its eigenvalues average one and `lam_max ≥ 1` always. Dividing by it
     under-relaxes every mode except the extreme one. Worth **10 → 2 cycles** at four sweeps.
     Reachable as `convection_multigrid_solve(..., omega=1.0, spectral_damping=False)`; the
     spectral default is unchanged, because an undamped sweep is not a contraction standing alone —
     it does not need to be under a coarse correction and an outer Krylov.

  **Equilibration is NOT among the causes**, and the two that are were invisible until the sources
  were read side by side. The earlier "ours is behind on both axes" reading was correct as a
  measurement and wrong as a diagnosis: it described a hierarchy that was simply not running the
  same algorithm.

  **The three remaining differences, now BUILT and measured (same block, same state, matched arm):**

  | arm | coarse eq | ×1 | ×2 | ×4 |
  |---|---|---|---|---|
  | matched, before | 436 | 58 | 11 | **2** |
  | + fix-up pass + magnitude-first graph | 438 | 52 | 10 | **2** |
  | + coarse-size stopping rule | 438 | 52 | 10 | **2** |

  - **The fix-up pass — BUILT, `_reattach_to_adjacent_root`.** PETSc's `fixAggregatesWithSquare`
    (`agg.c:1032-1075`): after the squared-graph MIS, each selected root **in ascending index
    order** steals every distance-1 neighbour in the *unsquared* graph that belongs to another
    aggregate. Only members move, never roots, so the count is unchanged and no aggregate empties.
    **Its guarantee is conditional and must not be stated as absolute** — a member whose neighbours
    are all themselves members has no adjacent root to move to and keeps its distant one. Worth a
    cycle at ×2 and six at ×1; nothing at ×4, where the arm was already at PETSc's 2.
  - **Magnitude before symmetrization — BUILT.** PETSc block-sums `|A_ij|` and *then* symmetrizes;
    we took the symmetric part first, so an edge with `A_ij ≈ −A_ji` **cancelled out of the graph
    entirely**. On an M-matrix (every frozen upwind transport operator) the sparsity is identical
    either way, so at `strength_threshold = 0` — the default, and what every scalar hierarchy runs —
    this is a no-op. **It is NOT a no-op at a threshold**, because the weights differ (`|A_ij|`
    against `|A_ij + A_ji|/2`) and the strong-connection set is chosen from them: the coupled flow
    block runs at **0.25** (`turbulence/coupled.py`), so its hierarchy genuinely moves. Do not quote
    the M-matrix equivalence without that caveat — it is a statement about the pattern only.
  - **The stopping rule — measured INERT here, and that is a real result rather than a null one.**
    PETSc coarsens until the grid is under `coarse_eq_limit`; we stop at `max_levels`, at which
    point `max_coarse` can never fire. Setting `max_coarse=2000, max_levels=20` gives a
    **bit-identical** hierarchy on this block, because one aggregation already lands at 438 < 2000
    and both rules then stop. So it changes nothing *at this size* and remains the correct rule for
    a mesh where it would bind. The two knobs are still not equivalents and must not be quoted as
    matched.

  **Still different, verified by reading the source, and NOT measured:**
  - **Strength-of-connection semantics differ** — PETSc thresholds absolutely on the
    diagonally-scaled graph (Vanek), we use the row-max-relative classical criterion. Inert at
    threshold 0, so it affects no measurement here, but the recorded threshold arms on the two sides
    are not comparable.
  - **Tentative prolongation — NOW MEASURED, and it is REFUTED as a deep-hierarchy fix.** PETSc
    orthonormalizes each aggregate's block by QR (`agg.c:690-714`), giving unit-2-norm columns where
    ours are 0/1 — i.e. `P_ours = P_petsc · diag(√|agg|)`. The inertness claim is confirmed exactly (a
    2-level pair is bit-identical), but deeper it is **17–30× worse**, not better. Built as
    `orthonormal_prolongation`; see *"Depth, singletons, and the prolongation"* below for the arms and
    for the mechanism that was proposed and refuted.
  - **Singletons — NOW MEASURED, and this one was a real defect worth 1.7×.** PETSc drops a
    neighbourless vertex from the coarse space entirely (zero row in `P`, left to the smoother); we gave
    it its own aggregate. On this operator most such vertices are not neighbourless at all — they are
    an artifact of the random sweep's arrival order — and a three-level hierarchy came out with 49
    singletons of 161 aggregates on its second level. Fixed behind `avoid_singletons`; see below.

  **What equilibration IS worth (keep it). ⚠️ "Default off" was a scope error — settled from source
  2026-08-10:** `build_convection_hierarchy` and `_build_aggregation_hierarchy` default `equilibrate=False`,
  so "default off" is true of the **JAX-native builder only**; `native_nodal_inverse` overrides it to `True`
  (its per-cell block solve is not otherwise safe); and the PETSc `AmgVCycle` path equilibrates
  **unconditionally** via `equilibrate_cell_major`. Three different objects, no contradiction.

  **⚠️ THE SECOND, COARSENING-INDEPENDENT BENEFIT RECORDED HERE WAS A PSEUDO-INVERSE ARTIFACT, AND IT IS
  GONE (2026-08-14).** The claim was that the coarse solve's singular-value truncation is what a badly
  scaled operator defeats first, so rescaling buys accuracy there: on the badly-scaled chain fixture
  (condition ~1.6e8) the one-level direct solve went from 1.2e-09 unscaled to 5.3e-13 rescaled. That is a
  property of the **pseudo-inverse**, which truncates small singular values, and not of the operator. The
  coarse solve is now a factorization (`_dense_inverse`), which the scaling does not defeat, and the same
  fixture reads **2.8e-13 unscaled against 3.9e-13 rescaled** — both at the floating-point floor, the
  unscaled one four thousand times better than it was, and their ordering now noise. The unit test that
  pinned the gap asserted `scaled < raw` and duly failed on the change; it now pins only what remains
  true, that both paths invert `A`. **The surviving case for rescaling is per-cell block conditioning**
  (measured separately, orders of magnitude in the median) and its effect on the coarse operator the
  tentative prolongation builds — not coarse-solve accuracy.

  **✅ THE COARSE SOLVE IS A FACTORIZATION, NOT A PSEUDO-INVERSE — `_dense_inverse` (2026-08-14).** A
  pseudo-inverse is a singular value decomposition, and it was **59 % of a whole hierarchy build**
  (1.37 s of 2.32 s on a 92160-dof, 8.7M-nnz flow block coarsening to 2156). The generality is only
  needed for a *singular* coarse operator, which this module works to avoid producing — an empty
  aggregate is given a unit diagonal precisely so no coarse row is structurally zero. So the LU is taken
  where it is valid, falling back to the decomposition where it is not. Cost at the coarse sizes that
  arise, and at the ceiling `_MAX_DENSE_COARSE_DOFS` permits:

  | n | pinv | inv | LU factor alone |
  |---|---|---|---|
  | 868 | 0.090 s | 0.013 s (7.0×) | 0.011 s |
  | 2000 | 1.027 s | 0.118 s (8.7×) | 0.032 s |
  | 4096 | 12.06 s | 1.154 s (10.5×) | 0.263 s |
  | **8192** (the cap) | **101.8 s** | **9.7 s** | 2.0 s |

  Whole-build effect **1.97×** (2.30 s → 1.17 s), with the coarse inverse agreeing to 6.4e-15 and the
  cycle output to 1.1e-16. **The gap widens with coarse size**, so this matters more as the mesh grows,
  which is the direction 3D is heading.
  ⚠️ **Singularity is judged by LAPACK's reciprocal-condition estimate, not by whether the factorization
  raised.** An operator can be numerically singular without being exactly so, and a factorization of one
  returns finite nonsense rather than failing — measured, `np.linalg.inv` of a matrix with a 1e-18
  diagonal entry returns finite garbage without complaint. The threshold is the same `n · eps` the
  pseudo-inverse itself uses when truncating, so the two paths agree on where invertibility stops
  instead of drawing the line in two places. All three cases are pinned
  (`test_the_coarse_solve_is_a_factorization_where_the_operator_admits_one`); a fallback that never
  fires and one that always does look identical from the outside.

  **What is BUILT and measured good (keep):**
  - **Nodal (block-aware) aggregation** — `_cell_graph` collapses the dof graph to cell connectivity
    (exact, since field-major means `index % n_cells` *is* the cell) and `_block_tentative` gives each
    field its own coarse unknown. With a **block smoother** (`_cell_block_inverse`,
    `_block_diagonal_inverse_operator`) this makes the two-field slice buildable and convergent.
    **All three are required; no pair suffices** (measured 2×2).
  - **MIS aggregation** (`_mis_aggregate`) — greedy maximal independent set over a **randomized** visit
    order, single sweep, selector-claims-neighbours, faithful to `MatCoarsenApply_MISK_private`. Worth
    **8 → 5 cycles** over the two-pass RCM scheme. The old scheme only seeded from a fully-free
    neighbourhood, so it seeded few aggregates and left most vertices to a ragged cleanup pass.
  - `jax.jit` on the native applies (they were dispatching eagerly, ~18 % of apply cost) and
    `indices_are_sorted=True` in `_coo_apply` (CSR→COO is row-sorted by construction).
  - **There is no `PerFieldNativeInverse` — deleted 2026-08-15 (binding).** It had no production caller
    (only its own tests and one sweep arm), its own docstring recorded it as superseded by
    `NodalNativeInverse` wherever that works, and it was never ported onto `NativeHierarchyInverse` — so
    it had neither `refactor_block` nor `refactor` and `BlockTriangularFieldSplit.refactor` **raised**
    the first time a march refreshed with it, while being exported from `__all__` as public API. It also
    re-committed two costs the shared base was written to remove (eager per-field transposes; a fresh
    closure per apply). Its measurements never transferred to the nodal inverse in any case — a
    per-field pair is a weaker object than one block-aware hierarchy — so the `nativeN` arm in
    `turbulence_smoother_sweep.py` went with it, leaving `nodal[N][cM]`.
  - `NodalNativeInverse` (`solve/field_split.py`, over the shared
    `solve/native_inverse.NativeHierarchyInverse` base), both transposable in
    closed form and fixed linear operators, so adjoint-legal. **⚠️ "Neither is wired into production" was
    wrong — settled from source 2026-08-10.** The nodal inverse is reachable through
    `native_nodal_inverse` and is the `BFS3D_TURBULENCE_INVERSE=native` arm; what is true is that it is not a
    *default*.

  - **`prolongation_smoothing` is its own parameter, no longer welded to `mis_aggregation`.** The old
    flag chose the aggregation *and* the prolongator formula together, so "MIS with and without
    prolongator smoothing" was not expressible and the two effects could not be separated — which is
    how the smoothed prolongator came to be recorded as void when what it actually needs is more
    smoother sweeps. `"symmetric-part"` (default, the historical formula), `"standard"` (the textbook
    σ_max form on the true operator and scalar diagonal), `"none"` (plain aggregation, what the
    shipped PETSc bundle runs). An unknown value raises.
  - **`_mis_aggregate` returns its ROOTS**, not just the aggregate index per vertex — the fix-up pass
    cannot be written without knowing which vertex seeded each aggregate, and re-deriving it
    afterwards is not possible (an aggregate's root is not recoverable from the labelling).

  **What is BUILT and measured BAD (revert or gate):**
  - The **standard prolongator at 4 sweeps** fails outright (44 cycles, true rel 1.0) — raw *and*
    equilibrated, so this is not the scaling. At **8 sweeps equilibrated it is the best native arm
    (4 cycles)**, so it is under-smoothed rather than wrong: the earlier "VOID / much worse" reading
    conflated a smoothing deficit with a broken formula. Treat sweeps as part of that arm's
    specification, never quote it at a single sweep count.
    ⚠️ **Measured on the 2150-equation scalar problem under a point-block-Jacobi / ILU smoother. It does
    NOT carry to the flow saddle**, where both smoothed prolongators are refuted under the SIMPLE
    smoother at 4 *and* 8 sweeps — see *Smoothed aggregation on the flow saddle*. "Best native arm"
    below always means best on the operator it was measured on.
  - **⚠️ GRAPH SQUARING ON THE FIRST LEVEL IS GAMG'S DEFAULT COARSENING, NOT A SIZE KNOB — and
    recording it as harmful was the single thing holding the comparison open.** Squaring the graph on the first level is not a size knob, it is
    GAMG's *default coarsening* (`aggressive_coarsening_levels 1`), and its 106× ratio is not
    over-coarsening — it is PETSc's 107×, i.e. the target. The 5 → 10 reading is real but was taken
    with the damped smoother; at the matched undamped smoother the same arm is **2 cycles**. A knob
    measured harmful in one configuration was recorded as harmful in general, and that closed off
    the one change that mattered.

  **Two corrections to older entries in this file, both of which cost real time:**
  - The recorded *"the builders refuse the `[k,ω]` slice because its diagonal is negative from the live
    source linearizations"* is **wrong**. The fine slice is clean — **0 of 23040** cells non-positive at
    β = 0, at the march shift, and at the floor. The refusal names `level 1`, a **coarse** operator, and
    is manufactured by **field-blind aggregation** merging a k-dof with an ω-dof of another cell. The
    error text said `level 1` all along; an inference was stapled to a verbatim quote and only the quote
    was ever checked.
  - **A field split HIDES the quality of its trailing half.** That half is ~11 % of the nonzeros, so the
    flow block carries the solve: a trailing inverse that does not converge *at all* in isolation still
    produced a plausible 1.36× blended march estimate. **Measure a block's preconditioner on the block
    alone first**, then in situ.

  - **A THIRD correction, from this round: an arm quoted at one smoother-sweep count is not a
    result about the method.** The standard prolongator was written down as "much worse" from its
    4-sweep numbers; at 8 it is the best native arm. Two of the three wrong conclusions in this
    section came from holding one axis fixed while attributing the outcome to another.

  **Also established:** block Jacobi is a perfectly good smoother class here given enough sweeps (PETSc
  ×4 = 2 cycles against ILU ×1's 2), and on CPU the two cost the *same* per apply (102 vs 101 ms) — so
  the penalty for a GPU-friendly smoother is quality, buyable with sweeps, not cost. The remaining
  question is a GPU one and **cannot be measured in the current environment** (CPU-only JAX).

  **⚠️ COMPARE LIKE WITH LIKE, or the arm measures the smoother and gets attributed to the hierarchy
  (binding for this campaign).** Our multigrid has only Jacobi-class smoothers; PETSc's default here
  is an incomplete-LU sweep, which is both far stronger and the least parallelizable piece in it. An
  ours-vs-PETSc table whose PETSc row is ILU therefore cannot say anything about the *coarsening*,
  which is the thing under development. Every such table must carry a **matched-smoother** row —
  PETSc `pbjacobi` against our block Jacobi, at the same sweep count and the same aggregation — and
  that row is the one to quote. The ILU rows stay as the incumbent's absolute bar, not as the
  comparison.

  **⚠️ HOW THE GAP WAS ACTUALLY FOUND, because the method generalizes and the alternative cost
  days.** Three rounds of tuning our own knobs (equilibration, prolongator variants, sweep ladders)
  moved nothing and produced two wrong write-ups. What worked was **an independent agent reading
  `agg.c` / `misk.c` / `rich.c` against `multigrid.py` and reporting differences with citations on
  both sides** — it found the undamped Richardson scale, the default aggressive level, and the
  fix-up pass in one pass, none of which was visible from any measurement we had. When a
  reimplementation of a *published, available* algorithm underperforms it, read the source before
  running another sweep; and do not describe a port as faithful until something has checked it
  line by line, which is exactly the claim that was wrong here.

  **NEXT STEPS, in order:**
  1. **Decide whether the matched configuration becomes the DEFAULT, and for which consumers.**
     ⚠️ **This shipped: `native_nodal_inverse` now defaults to the whole matched bundle**
     (`aggressive_levels=1`, `prolongation_smoothing="none"`, `spectral_damping=False`,
     `equilibrate=True`) and is the `BFS3D_TURBULENCE_INVERSE=native` arm. The open part is whether it
     transfers to other consumers; the library `build_amg_vcycle` defaults are untouched. The measurement says the matched
     bundle is 5 → 2 cycles on the turbulence block; whether that transfers to the scalar transport
     and velocity blocks is unmeasured, and flipping a default is a march-level decision, not a
     block-probe one.
  2. Then, if still short: sparse-LU coarse solve (ours is a dense inverse, `_dense_inverse`),
     QR-orthonormalized
     tentative columns (inert at 2 levels, not deeper), and Chebyshev with the SA-cached bounds.
  3. **Scalability, independent of all the above and unaddressed:** the convection hierarchy is capped
     at 2 levels (`_CONVECTION_LEVELS`) with a **dense inverse** coarse solve. At a 77× ratio that is
     ~26k coarse dofs and 5.4 GB to store at 1M cells — infeasible. Depth now *builds* (3 levels, no
     refusal) via the new `max_levels` argument, so the fix is cheap once the method itself works.
     PETSc reaches only 2 levels here because it coarsens until under `coarse_eq_limit`; our
     `max_coarse` is NOT the equivalent knob and has no effect at 2 levels.

  **Local PETSc source** for continuing the port (shallow sparse clone, may need re-cloning):
  `src/ksp/pc/impls/gamg/{agg,gamg}.c` and `src/mat/graphops/coarsen/impls/misk/misk.c` — note
  `coarsen` moved under `graphops` in current PETSc. GAMG-AGG defaults: `nsmooths 1`,
  `aggressive_coarsening_levels 1`, `use_aggressive_square_graph TRUE`,
  `use_minimum_degree_ordering FALSE` (so: random order), `aggressive_mis_k 2`, `graph_symmetrize TRUE`.



## The coupled AMG builder

- **Coupled builder `coupled_amg_continuation`** (`.claude/rules/turbulence.md`) shares
  `MonolithicFactorShiftPolicy` + `_monolithic_factor_step` with the ILUT/LU. Verified: converges to the
  block PC's fixed point AND passes the **coupled-adjoint FD gate** (the transpose V-cycle serves the
  gradient), `tests/integration/test_coupled_amg.py`; V-cycle mechanics in `tests/unit/test_amg_preconditioner.py`.
  Follow-ups: a refreshing/β-tracking variant (the frozen build serves the forward + adjoint; a developing
  3D march would want the refresh), and the FGMRES forward optimization.


## Binding decisions — `solve/multigrid.py`

> The rest of `solve.md`'s "Binding decisions" carries the general Newton/adjoint decisions and a
> short pointer to this section; these are the `multigrid.py`-specific ones, moved here in full.

- **`solve/multigrid.py` is a pure operator-coarsening library — operator-in, uniformly (binding).**
  **Every** builder takes an assembled `a: sp.csr_matrix` — `build_smoothed_hierarchy(a)`,
  `build_convection_hierarchy(a)`, `build_air_hierarchy(a)` — and none takes a mesh, edge arrays, or
  flow quantities. Assembly lives beside it in `aquaflux/solve/frozen_operator.py`
  (numpy + scipy only — no mesh, no field, no `jax`):
  `convection_diffusion_operator(owner, nb, coefficient, n, *, flux=None, boundary_diagonal=None)` —
  symmetric graph Laplacian when `flux is None`, first-order-upwind convection-diffusion otherwise —
  plus `decouple_dof(a, index)` for the closed-domain pressure pin. It is the **one** assembler for
  all four consumers (pressure Schur, viscous velocity block, convection velocity block, k/ω scalar
  transport). Do not reintroduce a `(owner, nb, coeff, …)` signature into `multigrid.py`, and do not
  add a second stencil assembler — the old `_laplacian_csr` was exactly `convection_diffusion_operator`
  at `flux=None` and was deleted. `build_convection_air_hierarchy` was likewise **deleted**: once it
  took `a` it was a pure alias for `build_air_hierarchy`.
  *Why it is a solver concern, not a flow one:* the first-order-upwind stencil is the
  **preconditioner's** choice, independent of what the residual discretizes advection with — it is
  chosen to give a diagonally dominant M-matrix an aggregation hierarchy can coarsen. Its parameters
  are a weighted graph (`coefficient`, `flux`), not flow quantities. Keeping it in `solve/` also adds
  no new dependency edge: every consumer already imports `solve.multigrid`.
- **The V-cycle recursion AND its outer fixed-cycle driver are single-homed (binding, #52).** A
  family (`smoothed_multigrid_solve`, `convection_multigrid_solve`, `air_multigrid_solve`) contributes
  **only** its `_VCycleOps` — restriction, prolongation, smoother. The recursion is `_frozen_v_cycle`
  and the outer loop (zero initial guess, `cycles` residual-correction passes,
  `x += _frozen_v_cycle(levels, b - A x, …)`) is `_fixed_cycle_solve`; both are written once. That
  outer loop is what makes `b -> x` a constant linear operator — the property the frozen-left-PC and
  the adjoint transpose depend on — so it must not be re-typed per family where one copy could drift.
  A new family adds a `_VCycleOps` builder and a thin entry point that calls `_fixed_cycle_solve`; do
  **not** re-write the cycle loop in it.
- **A level is STATIC indices + TRACED values, so a hierarchy refresh is a jit cache hit (binding).**
  `_SparseLevel` / `_AirLevel` are `equinox.Module`s in which **only `n` and `n_coarse` are static** —
  they size the sparse matvec output (`_coo_apply`'s `n_out`), so they must be concrete. Everything
  else (`val`, `diagonal`, `coarse_inv`, the prolongation/restriction values, and **`lam_max`**) is a
  traced leaf. `lam_max` is deliberately a **0-d array, not a Python `float`**: it is only smoother
  arithmetic, and as a static field any refreshed value would be a new compilation-cache key —
  defeating the point. Consequence: a hierarchy passed as a **jit argument** survives a refresh as a
  *cache hit* (one compiled V-cycle), which is what lets a frozen preconditioner track a developing
  flow without paying a recompile per refresh (the ~2.6× scalar-AMG staleness win above). This is only
  sound because **the aggregation coarsening is a pure function of the graph at the default
  `strength_threshold=0`** — `_aggregate` reads `owner`/`nb`/`n` and never the coefficients — so on a
  fixed mesh a hierarchy re-derived at a new operator has identical aggregates, coarse sizes and array
  shapes, and only values differ (this is why the refreshed *scalar k/ω* AMGs keep `strength_threshold=0`
  — the flow block is frozen, so it can afford the value-dependent SoC below; the scalars cannot without
  a value-refresh). Both properties are pinned in `tests/unit/test_multigrid.py`
  (`test_aggregation_hierarchy_structure_is_value_independent`,
  `test_refreshing_a_hierarchy_is_a_compilation_cache_hit`). **Caveat — lAIR does NOT get this for
  free, and this is measured, not hypothetical.** Its C/F split comes from `_strength_classical`, which
  thresholds on `|A_ij|`, so re-deriving a reduction hierarchy at a new operator changes the split and
  the shapes: on a 600-cell chain, cold vs developed coefficients (5000× flux, 1000× viscosity ramp)
  gave **identical L0–L2 but divergent L3–L5** (`n_coarse` 37→38, then `n` 37→38 / `nnz` 109→112,
  18→19 / 52→55), i.e. a different jit signature and therefore a recompile anyway. The aggregation
  path was invariant at *every* level in the same comparison. **Consequence:** for `method="air"` (and
  `velocity="convection-air"`) a cheap refresh requires **reusing the reference's frozen C/F split and
  prolongation and recomputing only the values on it** — legitimate, since any valid split gives a
  valid preconditioner. That is **`refresh_air_hierarchy(hierarchy, a_new, degree=…)`** (below),
  whereas the aggregation path gets it for free by rebuilding.
  - **Strength-of-connection (SoC) aggregation IS available now — opt-in via `strength_threshold`,
    and it makes the aggregation path value-dependent (so it forfeits the free refresh above).**
    `build_smoothed_hierarchy(a, strength_threshold=θ)` / `build_convection_hierarchy(a, …)` (and
    `BlockPreconditioner.build(…, strength_threshold=θ)`) aggregate on the **strong** subgraph only —
    edge `(i,j)` kept iff `|A_ij| ≥ θ·max_k|A_ik|` from either endpoint (`_aggregation_edges`, reusing
    `_strength_classical`, symmetrized) — instead of the full graph. **Why it exists:** isotropic
    aggregation coarsens *across* the stiff direction of a high-aspect-ratio / skewed operator and the
    V-cycle stalls; SoC coarsens *along* the strong coupling and keeps contracting. Measured on a
    uniformly-anisotropic Poisson (cycles to true 1%): plain `>20` and rising toward failure as the
    aspect ratio grows (contraction 0.22→0.56→0.90→0.97 at AR 1→10→100→1000), **SoC a flat 3 cycles at
    every AR, mesh-independent** — the difference between a converging and a non-converging solve at
    AR≥100 (`test_multigrid.py::test_strength_of_connection_aggregation_fixes_an_anisotropic_operator`).
    **`θ=0` (default) is byte-identical to the historical isotropic build.** It is turned **on (θ=0.25)
    for the coupled flow block** (`_coupled_shift_policy`: the convection velocity AMG + MSIMPLER Schur),
    which is frozen at the reference state so the value-dependence costs no refresh; it is a **no-op on
    the low-aspect-ratio pitzDaily case**, and the payoff is the future wall-resolved / skewed regime.
    It does **not** apply to the reduction-based `air`/`lsc` blocks (already strength-based), and the
    refreshed scalar k/ω AMGs stay `θ=0` to keep their refresh cache-hit — a value-refresh (à la
    `refresh_air_hierarchy`) to let them use SoC too is the tracked follow-up.
- **`refresh_air_hierarchy` — the lAIR refresh that keeps the compilation signature (BUILT).** It
  re-derives an lAIR hierarchy's **values** at a new operator while holding the coarsening fixed: each
  level reuses its stored C/F split (recovered from the level's own masks) and its stored
  prolongation, and re-solves only the local approximate-ideal restriction against the new `a`. The
  prolongation must be *carried over*, not re-derived, because `_one_point_interpolation` picks each
  F-point's strongest C-neighbour by `argmax |a_ij|` — a value-dependent column choice. The
  restriction's sparsity, by contrast, depends only on the split and on `a`'s *pattern* (the
  degree-`d` neighbourhood walk), so it is invariant, and the Galerkin `R A P` patterns below follow
  inductively. The result is verified before returning (`_require_matching_structure`) and a
  mismatched operator raises rather than silently returning a hierarchy that would recompile. Pinned
  by `test_refresh_air_hierarchy_keeps_the_structure_and_is_a_cache_hit` (shapes preserved, values
  changed, jitted V-cycle traces once), `test_refreshed_air_hierarchy_preconditions_the_new_operator`
  (a refreshed cycle beats the stale one on the new operator, so the reused split is a real trade and
  not a no-op), and `test_refresh_air_hierarchy_rejects_a_mismatched_operator`. **Why it matters:**
  measured on the separated pitzDaily state with the production lAIR scalars, refreshing the k/ω AMGs
  is worth ~2.4× in outer cycles (30 → 13 at β=2; the flow block 30 → 29, i.e. nothing), and this is
  the only way to take that win without paying a recompile per refresh.
- **Degenerate-mesh guard (binding — validated where the graph is consumed).** Because the
  hierarchies are built once off-jit and then frozen, a degenerate mesh must fail *there*, not as a
  silently stalling runtime V-cycle. Now that the builders are operator-in, the **graph** check lives
  with the assembler: `frozen_operator.require_valid_graph` (`n ≥ 1`, matched `owner`/`nb`, in-range
  endpoints) runs inside `convection_diffusion_operator`; the two build loops
  (`_build_aggregation_hierarchy` for smoothed/convection, `build_air_hierarchy` for lAIR) call
  `_require_positive_diagonal` on **every** level's operator diagonal before inverting/freezing it,
  so a zero diagonal (disconnected component, isolated/zero-volume cell, degenerate `R A P` row)
  raises `ValueError` at setup instead of baking `inf` into the frozen operator. The diagonal is
  checked *after* boundary stiffness is folded into the operator, so a boundary-only cell is
  correctly allowed. This one build-time guard is why the runtime smoothers (`_chebyshev_smooth`,
  `_jacobi_smooth`, `_fc_jacobi`) and the block-preconditioner rescales, which invert the frozen
  diagonal / the positive momentum `a_P`, need no per-apply floor.
- **The damped-Jacobi convection hierarchy is TWO-LEVEL by design (binding — do not add a depth
  knob).** `build_convection_hierarchy(a)` builds exactly a smoothed fine level + a single **direct**
  (dense inverse) coarse solve. **`max_levels` exists** (`multigrid.py`, default
  `_CONVECTION_LEVELS = 2`); raising it re-opens the defect below, so leave it at 2. On the fine level the
  upwind operator is a diagonally dominant M-matrix, so one damping factor `ω/λ_max` contracts
  (`_jacobi_smooth`, ρ ≈ 0.7 at high cell Peclet). A *deeper* Galerkin recursion is deliberately not
  built: a coarse-of-coarse operator of a strongly convection-dominated problem acquires
  near-imaginary-axis eigenvalues that **no single-factor damped-Jacobi smoother can damp** — the
  smoother becomes non-contractive (measured ρ(S) ≈ 1.0–1.36 on such levels), so the coarse level
  must be an exact solve. Deep, mesh-independent convection coarsening is the job of the
  reduction-based lAIR hierarchy (`build_air_hierarchy` + `_fc_jacobi`) instead. Both
  production callers (the flow `SmoothedAmgConvectionVelocity` two-level path and the turbulence
  preconditioner) already used two levels, so this is behaviour-neutral; the deep damped-Jacobi
  build it removed was dominated on both ends (worse than two-level shallow, worse than lAIR deep)
  and was the sole source of the non-contractive-smoother defect.
