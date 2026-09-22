# Investigation log — issue #435, the multiple-correction gradient on tetrahedra

> Lives in `.claude/notes/`, outside the auto-loaded `.claude/rules/` tree, so it never auto-loads.
> It holds the full chronological investigation into why `MultipleCorrectionGradient` cannot march a
> coupled RANS problem on the tetrahedral duct in `validation/tetrahedral_gradient_ab/`, including
> the reasoning behind ideas that were tried and refuted — kept so a refuted idea does not get
> re-derived. See `.claude/rules/schemes.md`'s `boundary_gradient_weight` entry for the current,
> load-bearing status of the fix that came out of this investigation (the singular first-pass
> extrapolation on corner tetrahedra); this log is everything found *after* that fix, chasing why the
> march still fails.
>
> ⚠️ **THIS IS A POINT-IN-TIME SNAPSHOT, NOT A MAINTAINED RECORD.** Everything below was measured on
> the branch this investigation ran on, whose base is commit `480c8c6` (itself on top of the `main`
> that existed at PR #460). It predates, and has **not** been reconciled with, at least these later
> changes to `main`:
>
> - **#467 / #468 / #469** — binding `MultipleCorrectionGradient` per field against that field's own
>   boundary conditions (rather than geometry-only), the corner-cell regression that binding first
>   caused, and its own repair.
> - **A `compare.py` harness bug, found and fixed independently of this log** (`e7dc4af`):
>   `run_march_ab` called `solve_coupled` with `rtol=`/`atol=` keywords the API had since replaced
>   with a `Convergence` object, and caught every exception, so for a time its report of "the march
>   fails" was actually an un-run `TypeError`. This log's OWN measurements do not go through
>   `compare.py`'s `run_march_ab` — they use the probe scripts named throughout, whose own march
>   helper (`warm_start_probe.py`'s `march()`) uses the identical `rtol=`/`atol=` calling convention.
>   That convention was verified, empirically, to be live and correctly forwarded at this log's base
>   commit (real per-step diagnostics were observed evolving over dozens of steps, which an immediate
>   `TypeError` cannot produce) — so the measurements below are genuine marches, not silently-skipped
>   ones. But the same calling convention would hit the now-fixed bug if run unmodified against
>   current `main`, so the probe scripts landed alongside this log are **not** guaranteed to run
>   against `main` as it now stands, and have not been re-verified against it.
> - Several docstring and stale-record corrections to `aquaflux/schemes/multiple_correction.py` and
>   `.claude/rules/schemes.md` (#468), made by an independent review of this same investigation.
>
> Read this as "what was tried, on what code, and what it showed" — not as a current statement about
> `main`. Where a later fix might resolve or change a finding recorded here, that is unverified.

---

  | 0 boundary faces (1374) | 15 | 15 |
  | 1 boundary face (912) | 1.25e16 (29 cells > 1e4) | 8.3 |
  | 2 boundary faces (176) | 4.1e18 (all 176 > 1e4) | 2.5 |

  **This singularity was symptom 1 of #435, and it was closure-independent.** It gave the potential-flow
  seed's Laplace Jacobian `cond` **3.29e19** (owner) / **1.69e19** (repaired) -- the "close to each other
  despite a twelve-order M2 gap" puzzle -- and after the fix **1.8e5** / **1.0e6**. It also inflated the
  coupled Jacobian's largest entry to 2.4e12 (0.007 mesh) and 2.8e25 (0.005 mesh). Pinned by
  `test_corner_tetrahedra_are_determined_by_their_boundary_conditions`. Mutation-checked: dropping the
  weight from the first pass, from the closure, or from the assembler, or restoring `w = d`, each fails a
  test.

  ⚠️ **It CHANGES THE DISCRETIZATION of every case on the default scheme with a gradient-type patch on a
  skewed boundary cell -- pitzDaily included**, which since #361 runs `MultipleCorrectionGradient` by
  default. The fast tier passes (1835); the slow and validation tiers were **not** run on it. **pitzDaily
  does not move** (2026-09-16, shipped defaults: `continuous` ramp, 16 stations x 1, `flow` scaling,
  turbulence damping 3, `MultipleCorrectionGradient` at reach 3, `simplesmooth` / `jacobi_smoothed`,
  `N_POINTS` 2, stop `(0, 1e-5)`): **31 steps / 191 cycles, one line-search clip, `x_r/h` 8.0686,
  `nut` peak 417.98, `ux` 0.0190, final `|R|` 7.815e-06, 305 s**, against the recorded arm at the same
  configuration (turbulence.md's damping table) of 31 steps / 196 cycles / `x_r/h` 8.0686. Not a true
  A/B -- the record predates several refactors -- but the root and the step count are unchanged. The old pitzDaily march A/B (skew + projection + `boundary_chain`: 707.6 s / 456 cycles /
  68 steps / `x_r/h` 8.069, against 660.4 s / 467 / 67 / 8.069 without it) was measured under the removed
  extrapolation and does not describe the current code.

  ⚠️ **It reaches the first pass only.** `M2` and the gradient defect are still probed against exact face
  values, and a zero-gradient face value is itself only a linear extrapolation tangentially, so a
  quadratic keeps a second-order inconsistency at boundary cells. Only `MultipleCorrectionGradient` reads
  the weight; the other schemes accept and ignore it.

  ⚠️ **It does NOT make the tetrahedral duct march -- symptom 2 of #435 is a second problem, traced
  2026-09-16 to the VELOCITY reconstruction at the wall-function cells.** All measured on the 2462-cell
  duct at the Re/10 anchor (`y+` ~10 at the wall cells), repaired closure unless stated:

  * **Not the plug initial condition.** A uniform plug does make the Hessian correction invent interior
    strain (median S/omega 0.47, max 73, the omega-production cap binding in 618 cells, against exactly 0
    for the first pass), but converging the anchor with `CorrectedGreenGauss` first and handing that smooth
    state over still fails: owner closure `|R|` 0.75 -> 3.9e3 then frozen at `alpha = 0`, repaired
    `alpha` 0.004--0.25 at 126--163 cycles with an exact LU (`warm_start_probe.py`).
  * **Not k/omega.** At that smooth state (`mixed_scheme_probe.py`), multiple correction on k/omega only
    with corrected Green--Gauss on velocity: `|R|` 3.2, S/omega max 0.90, cap binds nowhere, no
    non-positive omega diagonal. On velocity only: `|R|` **359**, S/omega max **18.8**, cap binds in **75**
    cells, **11** non-positive omega diagonals -- the same as the scheme on both blocks. (The omega wall
    value is already kept out by `ImposedGradient`.)
  * **Mechanism:** the negative omega diagonal is the closure's live `omega` inside the capped omega
    production (`10 beta* k omega / nu_t` grows with omega), activated by a spuriously large strain rate;
    the frozen-closure omega operator is positive under every scheme (advection dominates). Every such
    cell is in the **first ring** off the wall cells. A wall cell's velocity gradient reads the no-slip
    wall value -- a jump of several m/s across ~1 mm where the true profile is logarithmic -- and the
    first ring's Hessian correction differentiates that gradient.
  * **Localized:** first pass (no Hessian correction) on the wall cells only takes `|R|` 359 -> 15.7 but
    leaves the ring-1 strain (max 18.8, cap in 75, 14 negative diagonals); on wall cells **and** ring 1
    (no velocity-gradient model, just excluding both from the Hessian correction): `|R|` 1.01, S/omega
    max 0.88, cap nowhere, no negative diagonal at the seed. Marched from that seed it runs at 6--17
    cycles mostly (two steps near 90; measured before `refresh_on_cycles` existed, so the counts carry
    some preconditioner staleness -- see the next bullet for how much) but still does **not** converge
    in 12 steps (`|R|` 0.073 -> 0.70, `limited_hessian_march_probe.py`).
  * **Imposing a wall-model velocity gradient at the wall cells instead of excluding them reaches the
    same clean seed by a different route, and marching it exposes where the real problem is.**
    `grad u_i = t_i s n`, `s = (1-f)|U_t|/d + f u_tau/(kappa d)` -- the same
    `wall_function_weight`/`log_layer_shear_rate` blend the closure's own strain rate already uses, `k`
    read live from the coupled state, imposed via `ImposedGradient` exactly as `omega`'s wall value
    already is ("option (b)", `WallModelVelocityMomentum` in `wall_velocity_gradient_probe.py`) --
    combined with first pass on ring 1 ("option 1"): `|R|` 65.97, S/omega max 0.88, cap nowhere, no
    negative omega diagonal. **Every local symptom either variant was built to clear is gone at the
    seed.** Marched from that seed with `refresh_on_cycles=3` (matching pitzDaily's shipped default --
    without it the preconditioner is never refit mid-march and cycle counts inflate, but the outcome is
    unchanged), the residual still grows without bound: 0.25 (step 0) -> 1.2 (step 9) -> 44.0 (step 19),
    never converging in 20 steps; from a uniform plug it grows the same way (0.60 -> 1.78 by step 4, two
    escalations).
  * **⚠️⚠️ REFUTED BY MEASUREMENT — "the mechanism is squarely in `omega`, and within it, ring 0" was a
    raw-magnitude artifact, exactly the one this file's `omega_transform` entry warns about (found
    2026-09-17, the same day it was written, on this very case).** The bullet this replaces decomposed
    the diverging march by plain Euclidean per-block/per-ring norms of the physical residual and read
    `omega` (and ring 0 within it) as dominant. That measurement was taken on a build of this case with
    no `omega_transform=LogScalars()` (since fixed, see `compare.py`) and, independently of the
    transform, never used the march's own row-scaled measure -- so it inherited the exact defect this
    file already records for pitzDaily: a raw `omega` comparison is dominated by units and by the
    physical scale of `omega_wall` near the wall, not by which equation is actually least satisfied.
  * **Corrected: capture the march's own `RowScaledNorm` (row-equilibrated by each row's diagonal, then
    field-normalized) and decompose THAT, not the raw residual** (`divergence_diagnosis_probe.py`,
    wrapping `coupled_scaled_norm` rather than reconstructing its shift-policy dependency). With
    `scaled_norm` at its shipped default (`False`), the measure is built once, at the seed, and held
    fixed for the whole march -- captured once, applied to every checkpoint. **Under it, `omega` is
    consistently one of the SMALLEST blocks, never the largest, at every step**: step 0 `u0` 0.187 `u1`
    0.096 `u2` 0.095 `p` 0.0022 `k` 0.090 `omega` 0.0133; step 19 `u0` 23.1 `u1` 16.3 `u2` 11.4 `p` 0.98
    `k` 55.0 `omega` 4.05. **`k` overtakes everything to become the single largest block by the end, and
    grows fastest of all in absolute terms (~611x over 19 steps, against velocity's ~120--170x and
    `omega`'s ~300x)**; `p`, the smallest block throughout, also grows faster in relative terms
    (~446x) than `omega` does. Velocity, not `omega`, dominates the early steps. **Within `omega`
    itself, ring 0 -- the wall-fixation cells -- carries an ESSENTIALLY ZERO fractional residual at
    every step (1e-6 to 1e-2, growing only because the state itself is diverging), while rings 1, 2 and
    3+ are comparable to each other and carry the whole of `omega`'s already-small share.** This is
    exactly what the fixation row's construction predicts: it is linear in the solved variable with
    unit derivative, so a Newton step satisfies it almost exactly regardless of how far the rest of the
    state has moved. **The earlier "ring 0 is the driver" claim is retracted in full: it is not
    dominant, not fastest-growing, and not where the row-scaled evidence points.** Open: the leading
    candidates are now `k` transport and the velocity block, not `omega`'s wall-fixation cells --
    neither has yet been decomposed by ring or traced to a mechanism.
  * **Traced further: `k`'s own Jacobian diagonal goes non-positive at the FIRST step of the march, at
    wall-adjacent cells only -- earlier than any other symptom measured on this case.** At step 0, 2
    cells have `d R_k / dk <= 0`, both ring 0; by step 19 (still diverging), 29 cells, spread across
    every ring. This matches `KProduction`'s own documented mechanism exactly: its cap
    `min(nu_t S^2, 10 beta* k omega)` is an increasing function of `k` where the cap binds, and
    differentiating that cap exactly is "a negative diagonal contribution... that can destroy the
    diagonal dominance of the k-equation's Jacobian and stall its linear solve at high Reynolds
    number" (`aquaflux/turbulence/sources.py`). Both other validation cases already carry the
    documented fix, `explicit_production_limiter=True` (a forward-solve-only Patankar freeze of the
    cap's `k`; the residual value is untouched, only the Jacobian the linear solve sees) --
    `tetrahedral_gradient_ab/compare.py` never had it. **Adding it (now the case default,
    `TET_PRODUCTION_LIMITER=0` for the exact operator) measurably improves but does not fix the
    march**: negative-`k`-diagonal cells at step 19 fall from 29 to 6, the final (still non-converged)
    scaled `|R|` from 63.0 to 9.16, and the character changes from unbounded growth to a stall (`alpha`
    collapsing to 0.004--0.008 by steps 18--19, `|R|` plateauing near 9.15 rather than climbing
    further). **The remaining 2--6 negative-diagonal cells (present from step 0 onward, always
    including ring 0) point at the OTHER documented negative-diagonal source in the same file**:
    `KProduction`'s live production viscosity (`nu_t` proportional to `k`, so `nu_t S^2` differentiates
    to another negative diagonal term, addressed by `explicit_production_viscosity` /
    `frozen_production_viscosity`) -- untried here, and not what either shipped case's own `compare.py`
    uses (they reach it only through a separate Jacobian-probe path in a *different* case's harness,
    `pitzdaily_gradient_ab/run_ab.py`, not as a standing default), so adding it here would be a new
    experiment rather than restoring parity with a sibling.
  * **Adding `explicit_production_viscosity=True` too (`TET_PRODUCTION_VISCOSITY=0` for the exact
    operator, on by default alongside the limiter) shrinks the negative-diagonal count further -- 1 cell
    at step 0 (down from 2), 4 at step 19 (down from 6, 29 with neither fix) -- but the march still does
    not converge, and the final scaled `|R|` barely moves (9.16 -> 8.20).** Both forward-solve devices
    are doing exactly what they claim: reducing (not eliminating) a Jacobian-conditioning defect,
    without touching the residual value. **That the residual keeps growing regardless is the tell: the
    remaining driver is not a linearization artifact these freezes can reach, but something growing in
    the residual itself.** The strain-rate symptom option 1 was built to clear supports this directly --
    S/omega max is 0.88 AT THE SEED, but grows back to 1.40 by step 0 of the march and to 79.8 (ring 3+)
    by step 14, with the production cap binding in more and more cells throughout (26 -> 292). Option
    1's wall-model velocity gradient and ring-1 exclusion were only ever measured to hold AT the smooth
    seed; this is direct evidence they do not keep holding as the state moves away from it. **Open, and
    the redirect is back to the velocity/strain-rate mechanism, not further Jacobian conditioning**:
    whatever spatial pattern option 1 was tuned to clear at the seed apparently does not stay fixed to
    ring 1 as the flow develops.
  * **⚠️ It is NOT specific to `MultipleCorrectionGradient` -- `CompactGreenGauss`, a completely
    different, single-pass, non-Hessian scheme with no wall-cell treatment of any kind, ALSO fails to
    march this exact problem, and fails FASTER than either multcorr arm** (`scheme_march_comparison_probe.py`,
    same seed, same Re, `refresh_on_cycles=3`, 15 steps, scaled `|R|`): `compact GG` reaches `|R|` 0.20
    at step 0 (ratio 3.1x its own reference already) and 22.6 by step 14, cycles pegged at the 163 cap,
    `alpha` collapsing to 0.004--0.03 -- a faster, more violent failure than plain `multcorr repaired`
    (2.53 -> 47.1) and far worse than `option 1` (0.24 -> 2.66, still the best-behaved failing arm).
    **Only `CorrectedGreenGauss` -- marching the SAME seed AGAIN -- converges cleanly**: `alpha = 1`
    every single step, no damping needed at all, `CONVERGED` at step 12. This is the one scheme this
    case's seed itself uses, so it is not being warm-started FROM a different scheme's converged state,
    unlike every other arm. **The clean split is not "multiple-correction vs. everything else" -- it is
    "the scheme the seed was built with vs. every other scheme," and `CorrectedGreenGauss` is also the
    only arm here whose gradient is found by iterating a non-orthogonal correction to self-consistency
    (`SweptGradientSolve`) rather than computed in one shot from the current field.** Open: the next
    question is whether this is a warm-start artifact (swapping FROM a different scheme's root produces
    too large an initial gradient mismatch for a non-iterative scheme to absorb) or an intrinsic
    property of the non-iterative schemes on this mesh at this Reynolds number, independent of which
    scheme built the seed -- distinguishable by building and marching each scheme's OWN low-Re anchor
    from the plug, the way the seed itself is built, rather than warm-starting it from another scheme's
    root.
  * **⚠️ SETTLED: NOT a warm-start artifact. Every non-`CorrectedGreenGauss` scheme fails to converge
    its OWN anchor from the plug too** (`own_anchor_march_probe.py`, `refresh_on_cycles=3`, 25 steps,
    scaled `|R|`, no scheme is ever warm-started from another's state here): `compact GG` collapses
    immediately (`|R|` 0.55 -> 41.0 by step 12, then FREEZES bit-identical through step 24 with
    `alpha = 0` every step -- the line search rejects every trial outright); `multcorr repaired` is
    noisy but trends the same way (4.84 -> 199.0, one escalation cascade at step 21 spiking it to 382);
    `option 1` is the interesting case -- it starts as cleanly as `corrected GG` (`alpha = 1` at step 0,
    `|R|` 0.26, briefly `alpha` up to 0.5) and stays roughly bounded through step ~15 (0.26--0.55), but
    then starts growing again from step 16 onward and ends at 18.8, still diverging. **Of these four
    arms only `CorrectedGreenGauss` converges, from the plug exactly as from the seed: `alpha = 1` every
    step, no damping needed, `CONVERGED` at step 13** (the multiple-correction first pass, tried later,
    also converges -- see the last bullet of this list). So the split holds even with the warm-start
    variable removed, independent of how the march is started.
  * **⚠️ REFUTED: "the default's under-resolved 4-sweep correction is masking a divergence a resolved
    `CorrectedGreenGauss` would also show."** The default `SweptGradientSolve(sweeps=4)` warns on this
    mesh that its correction is under-resolved (relative residual above 5e-2). Tested by marching the
    same own-anchor problem (plug start, Re/10, `refresh_on_cycles=3`, 25 steps, production limiter and
    viscosity freezes on, `omega_transform=LogScalars()`) with the correction solved EXACTLY
    (`CorrectedGreenGauss(solver=GmresGradientSolve())`): **it converges in 13 steps with `alpha = 1`
    every step, and its residual trajectory matches the 4-sweep default's to about 1--3 % at every step**
    (step 0: 0.407 against 0.404; step 13: 4.28e-5 against 4.13e-5). Resolving the correction changes
    nothing, so the partial correction was not masking anything.
  * **⚠️ THE `SweptGradientSolve` ARMS WITH MORE SWEEPS ARE NOT A "MORE RESOLVED" TEST -- THE
    ITERATION DIVERGES ON THIS MESH.** `sweeps=12` still warns (residual above 5e-2 after 12 sweeps)
    and does not march (`|R|` stays 0.5--1.1 for 25 steps at 5--200 Krylov cycles per step, `alpha`
    0--0.125); `sweeps=40` blows up outright (`|R|` 2.2e6 at step 0 growing to 5e8, then a singular
    factorization). A Richardson iteration that gets WORSE with more sweeps is not converging to the
    system's solution here, so those arms measure the iteration's instability, not a resolved
    `CorrectedGreenGauss`; only the exact Krylov arm answers the question above. **A side finding worth
    its own follow-up: the swept solve's default of 4 sweeps works on this mesh only because it stops
    before the diverging mode grows** -- `relaxation` (the solve's damping factor) was not tried.
  * **What this leaves open, honestly:** the "single-shot schemes fall short because this mesh needs an
    iterated correction" hypothesis recorded earlier in this list is not supported -- exact and
    4-sweep `CorrectedGreenGauss` behave identically, so iteration count is not the operative
    difference. What IS established is the split itself (`CorrectedGreenGauss` and, later, the
    multiple-correction first pass march; `CompactGreenGauss`, full `MultipleCorrectionGradient` and
    option 1 do not) and that it is not a warm-start artifact, a
    Jacobian-conditioning artifact, or a k/omega production-cap artifact. Notably `MultipleCorrectionGradient`
    is the MORE exact reconstruction (quadratic-exact against `CorrectedGreenGauss`'s linear), so raw
    reconstruction accuracy does not explain the split either.

  * **Where the gradients differ, at ONE state** (`gradient_difference_probe.py`: the Re/10 anchor
    converged with `CorrectedGreenGauss`, 2462 cells, every scheme evaluated on that same flow, relative
    difference from `CorrectedGreenGauss` per cell, production limiter/viscosity freezes and
    `LogScalars` on). **Multiple-correction's Hessian second pass INFLATES the gradient in two places,
    and `CorrectedGreenGauss` is not smoothing anything.**
    - **The 176 corner cells (ring 0, own >=2 boundary faces): velocity-gradient magnitude is 2.7x
      `CorrectedGreenGauss`'s at the median (strain rate 2.8x, `grad k` 3.1x), and the relative difference
      tracks `max|M2^-1|` monotonically** -- median 0.5 where `max|M2^-1| <= 10`, 1.8 at 10--100, 49 at
      100--1e3, **347 at > 1e3 (worst cell 1652, at `max|M2^-1|` 8.4e3)**. The first pass at the same cells
      agrees with `CorrectedGreenGauss` to ~6 % and compact GG to ~19 %, so this is entirely the second
      pass. It is not a property of this flow: against the EXACT gradient of a smooth analytic field
      the same scheme's error at those cells is a median 11x at the three worst (p90 181x, max 224x)
      while its error over all cells is the best of the four (median 0.032, against 0.119 for
      `CorrectedGreenGauss`, 0.344 compact, 0.106 first pass) -- quadratic-exact, so higher-order content
      is amplified by `M2^-1` and nothing else is wrong.
    - **Ring 1 (731 cells, next to the wall cells): the second pass roughly doubles the first pass's
      magnitude** -- velocity gradient 1.64x `CorrectedGreenGauss` (strain rate 1.71x, `grad k` 2.44x)
      against 0.81x for the multiple-correction first pass and 0.79x for compact GG. Median relative
      difference from `CorrectedGreenGauss`: 1.39 full, 0.40 first pass, 0.50 compact. At ring 0
      (non-corner) and rings 2--3 the schemes agree far better (medians 0.05--0.6, and 0.13--0.3 by
      ring 3).
    - **`CorrectedGreenGauss` sits BETWEEN the others at ring 1 (compact and first pass ~20 % below it,
      full multiple-correction ~64 % above), so "the scheme that marches is the smoother one" is
      refuted**; that also matches its ~12 % median error on the smooth field, which is worse than
      multiple-correction's. Rank correlations of the difference with cell properties are weak for
      the full scheme (ring +0.17, `max|M2^-1|` -0.23, non-orthogonality +0.31) because the two
      places above are small populations against 2462 cells.
    - **What this does NOT explain:** `CompactGreenGauss` fails to march and is NOT inflating -- it is
      ~20 % low at ring 1 and ~5 % low at the corner cells, with a 34 % median error on a smooth field
      -- so a single snapshot cannot say the inflation is why multiple-correction fails, and compact GG's
      failure is evidently something else (or the same instability reached from the other side).
      Untested: marching the multiple-correction FIRST PASS alone (no Hessian, no `M2`), and the full
      scheme with the Hessian withheld only where `max|M2^-1| > 10` for EVERY field including pressure
      and `k`/`omega` (option 1 treats the velocity block only). The first says whether the second
      pass is the culprit; the second isolates the corner-cell amplification.

  * **Reconstructing EXACT functions on this mesh** (`exact_function_probe.py`: exact values at cell and
    boundary-face centroids, all 2462 cells, error = max per-cell gradient error over the 95th-percentile
    exact gradient magnitude). **`MultipleCorrectionGradient` (repaired) is exact where it is designed
    to be, in every cell including the 176 corner cells and the worst one (`max|M2^-1|` 8.4e3):**
    constant 7e-11, linear 8.3e-12, quadratic 6.4e-12 (first pass: linear 5e-14, quadratic 0.13 as
    designed). The unrepaired closure (`fallback=None`) fails exactly at the corner cells (linear 3.9,
    quadratic 1.8), which is the #432 defect and is what the repair removes. So the reconstruction is
    correct: no implementation error, no boundary-treatment error, no residual singularity.
    **What is not bounded is its response to content beyond quadratic, and it grows with `max|M2^-1|`:**
    cubic field, max error by `max|M2^-1|` group 0.06 (<=10), 0.39 (10--100), 0.93 (100--1e3), **12.0
    (>1e3, 3 cells)**; smooth sin/cos field 0.06, 0.20, 0.86, 2.1. Every other scheme stays bounded at
    those cells (0.05--0.3), and away from them multiple-correction is the most accurate scheme here
    (0.06--0.2 against 0.14--0.18 for the first pass). The wall cells of a wall-function mesh carry a
    log-layer profile, which is exactly the strongly non-polynomial content this amplifies, and the
    corner cells sit in that ring (the 2.7x median inflation of the velocity gradient recorded above).
  * **A side result that qualifies an earlier statement: the DEFAULT `CorrectedGreenGauss` (4 sweeps,
    under-resolved) is not linear-exact** -- max error 10.4 (linear), 3.7 (quadratic), 4.75 (smooth) in
    interior cells, with the same iteration that diverges at more sweeps -- while `CorrectedGreenGauss`
    with the exact Krylov solve IS linear-exact (1.5e-13) and otherwise behaves like the first pass
    (0.14--0.18). `CompactGreenGauss` is not linear-exact either (3.0). The `CorrectedGreenGauss` march
    is nonetheless indistinguishable between the two (see the exact-solve result above), so exactness of
    the reconstruction is not what separates the schemes that march from those that do not.
  * **REFUTED: withholding the Hessian where `max|M2^-1| > 10` (the ~184 ill-conditioned cells), for
    EVERY field, does not make the march converge** (`own_anchor_march_probe.py`, arm `multcorr M2<=10`,
    `FirstPassWhereIllConditioned(limit=10)` on velocity, pressure, `k` and `omega`; plug start, Re/10,
    `refresh_on_cycles=3`, 25 steps, freezes and `LogScalars` on): scaled `|R|` 0.53 at step 0 (`alpha`
    0.25, 89 cycles), 0.47--1.2 through step 8, then growing steadily to 56.2 at step 18 and **NaN at
    step 19**. It starts no better than plain `multcorr repaired` and is worse than `option 1`
    (0.26 -> 18.8), which treats only velocity at the wall cells and ring 1. So the corner-cell
    amplification found above is real but is not, by itself, what stops the march; nor is the ring-1
    velocity inflation by itself (option 1 relapses without the corner treatment). Not tested: both
    together (option 1 plus the `M2` limit on every field), and the first pass alone for every field.
  * **Probe correction affecting older records:** `FirstPassWhereIllConditioned` and `FirstPassOnCells`
    (`diffusion_operator_probe.py`) previously dropped the `imposed` gradient before calling the inner
    scheme, so any arm built from them lost omega's imposed wall-cell gradient from the Hessian; they now
    forward it. Results recorded earlier in this list from those two wrappers on the `omega` block
    (e.g. the ring-0/ring-1 first-pass variants of `mixed_scheme_probe.py`) did not include it.

  * **✅ THE MULTIPLE-CORRECTION FIRST PASS ALONE MARCHES -- THE HESSIAN SECOND PASS IS WHAT STOPS THE
    FULL SCHEME.** `multcorr first pass` for EVERY field (`FirstPassOnly(bound repaired)`: linear-exact,
    with the boundary-condition first-pass fix, no Hessian, no `M2`; `own_anchor_march_probe.py`, plug
    start, Re/10, `refresh_on_cycles=3`, freezes and `LogScalars` on): **`CONVERGED` in 13 steps,
    `alpha = 1` at every step, 6--12 cycles per step, scaled `|R|` 0.383 -> 3.77e-5 -- the same
    trajectory as `CorrectedGreenGauss` (0.404 -> 4.13e-5) to within a few per cent at every step.** The
    identical case, mesh, closure, freezes and boundary treatment with the second pass switched on
    diverges (4.84 -> 199 over 25 steps). This overturns the earlier reading that the failing schemes
    share a cause: `CompactGreenGauss` fails for a separate reason (it is not linear-exact, 3.0 on a
    linear field, and reads ~20 % low in the near-wall ring), and the full scheme fails because of the
    second pass. It also means the two partial treatments tried above (Hessian withheld at the ~184
    ill-conditioned cells for every field; wall-model velocity gradient plus ring-1 exclusion) each leave
    the second pass active in cells where it still destabilizes the march. Open: WHICH cells need it
    withheld. Cheapest discriminator: withhold it by wall ring (rings 0--1, 0--2, ... for every field)
    and find where the march starts to converge; the gradient comparison above put the largest
    inflation at ring 1 (1.64x velocity, 2.44x `grad k`) and the corners, but the `M2 <= 10` arm shows
    the corners alone are not enough.

  * **RING BISECTION: the Hessian has to be withheld almost everywhere -- the failure is NOT localized to
    the wall or the corners.** `own_anchor_march_probe.py`, arms `hessian off rings<=N`
    (`FirstPassOnCells`, every field, `imposed` forwarded; plug start, Re/10, `refresh_on_cycles=3`, 25
    steps, freezes and `LogScalars` on; ring sizes 1040/731/428/217/46, ring 0 = the wall-patch cells):

    | Hessian withheld through | cells still on | outcome (scaled `|R|`, `alpha`) |
    |---|---|---|
    | ring 0 | 1422 | diverges: 1.9 -> 14 (step 10) -> 5.4, `alpha` mostly < 0.06, many `alpha = 0` |
    | ring 1 | 691 | drifts: 0.38 -> 0.17 (step 2) -> 0.56 (step 10) -> 1.8, `alpha` 0.03--1, no convergence |
    | ring 2 | 263 | drifts: 0.53 -> 0.23 (step 4) -> 2.4, `alpha` 0.03--1, no convergence |
    | ring 3 | 46 | **`CONVERGED` in 13 steps, `alpha = 1` every step, identical to the first-pass-only arm** |

    Each ring withheld improves the early behaviour (fewer cycles, larger `alpha`, slower growth), and
    only leaving the Hessian on in ring 4 alone (the 46 cells furthest from the walls) converges. **Read
    "the core" with care: the duct is only about 6 cells across, so rings 0--3 are 98 % of the cells and
    every cell is within a few cells of a wall -- this arm shows the Hessian must be off wherever there is
    near-wall content, i.e. almost everywhere here, not that it fails in resolved core flow.** Combined
    with the `M2 <= 10` arm failing, no small population of cells is responsible.
  * **The exact Jacobian at the plug separates the arms cleanly** (`coupled_operator_probe.py`, uniform
    plug at Re/10, coloured probe at reach 3/5/7 against the true Jacobian, exact LU Newton solve):
    **the full scheme's Jacobian is numerically singular** -- `|x|` of the exact Newton step 5.3e12,
    smallest singular value 3e-14 against `|J|max` 26, the near-null direction is pure `omega` (share
    1.00), and 43 of 1422 free `omega` rows have a non-positive diagonal; withholding the Hessian only
    where `max|M2^-1| > 10` leaves it singular (`|x|` 1e15, same 43 rows), while `CorrectedGreenGauss`,
    the first pass alone, and `hessian off rings<=2` all give a regular Jacobian (`|x|` ~5e2, exact LU
    solve to 1e-16, 0 negative `omega` diagonals). **The plug-state singularity is therefore removed by
    withholding the Hessian in rings 0--2, yet that arm still fails to march** -- so there are at least
    two mechanisms: the plug-state `omega` singularity (the near-wall velocity-jump strain already
    traced above, gone with rings 0--2 off) and a slower one from the Hessian in the core that only
    shows as the march develops. Not yet identified.
  * **The reach-3 probe is not the story.** At reach 3 the probe reproduces the true Jacobian to 7e-3
    (`CorrectedGreenGauss`) and 9e-4 (full multiple-correction), and exactly (1e-16) at reach 5 and 7 for
    every arm, first pass and reach-3 columns included. Not tested: whether the 5--10x higher Krylov
    cycle counts of the failing arms (50--126 against 9--12) come from that residual 1e-3 probe error or
    from the operator itself.
  * **Consequence for #435, stated plainly:** on this mesh the configuration that marches is the
    linear-exact first pass (with the boundary-condition first-pass fix), and localized repairs of the
    quadratic-exact scheme do not recover it. Whether the scheme should offer the first pass as a
    supported mode, and whether the mesh (coarse wall-function tetrahedra) is simply outside what the
    second pass tolerates in a coupled RANS march -- pitzDaily's hexahedra march with it -- is a decision,
    not something the measurements settle.

  * **MESH REFINEMENT (5x the cells, same mesher): the first pass and `CorrectedGreenGauss` still march
    identically; the full scheme's first step still shows the failure signature, but the run died
    before it could be followed.** A 12577-cell duct (`MESH_SIZE` 0.004 in `of_case/make_mesh.py`, 3127
    nodes, otherwise the same geometry, generated outside the repository and selected with
    `TET_POLYMESH`) -- cell-max non-orthogonality median 27.2 / max 67.1 deg (WORSE than the 2462-cell
    mesh's 51 deg: uniform-size Delaunay refinement does not improve cell quality), 306 corner cells,
    max `|M2^-1|` 2.3e3 (interior) and 2.1e3 (corner), 510 cells above 10. Own-anchor marches from the
    plug at Re/10 (`refresh_on_cycles=3`, freezes and `LogScalars` on): **`corrected GG` CONVERGED in
    15 steps and `multcorr first pass` in 16, `alpha = 1` at every step, 9--15 cycles, residual
    trajectories matching to a few per cent** (0.492 -> 6.9e-5 and 0.492 -> 2.6e-5). **`multcorr
    repaired` (full scheme): step 0 `|R|` 1.60, `alpha` 0.0156, 56 cycles, two escalations -- the same
    signature as on the 2462-cell mesh (4.84, 0.0156, 52) -- and then the process vanished with no
    exception and no `ended:` line after 749 s for that one step (cause not identified; another session
    was loading the machine, load average 27 afterwards); step 0 took 749 s against 114--160 s for the
    other two arms.** So refinement alone has not made the full scheme march, but this is one step of
    evidence, not a march. Two other refinements (6590 and 8849 cells) were generated and rejected: the
    first has one corner cell the repair leaves at `max|M2^-1|` 4.7e13, the second an interior cell at
    9.3e3, i.e. the near-coplanar-sliver hazard `make_mesh.py` documents.

  * **✅ THE SAME DUCT ON AN ORTHOGONAL HEXAHEDRAL MESH MARCHES WITH EVERY SCHEME, THE FULL
    MULTIPLE-CORRECTION SCHEME INCLUDED -- so the failure is the tetrahedral mesh's cell shape, not the
    geometry, resolution, wall treatment, closure or start.** `of_case/make_hex_mesh.py` (transfinite,
    60 x 6 x 6 = 2160 cells of 4.2 mm cubes, same patches; non-orthogonality, skewness and neighbour
    volume ratio all exactly 0 / 0 / 1) against the 2462-cell tetrahedral duct's 18 deg mean / 51 deg max
    non-orthogonality; wall-cell distance 2.08 mm against 2.04 mm; own-anchor marches from the plug at
    Re/10, `refresh_on_cycles=3`, freezes and `LogScalars` on, identical case and wall model:
    **`corrected GG`, `multcorr first pass` and full `multcorr repaired` all `CONVERGED` in 13 steps with
    `alpha = 1` at every step and 9 Krylov cycles per step**, final scaled `|R|` 2.75e-5, 2.75e-5 and
    2.67e-5 (the first two identical to five figures on an orthogonal mesh, as the correction terms
    vanish). The identical case on the tetrahedral mesh: full scheme diverges, first pass and
    `CorrectedGreenGauss` converge. The `omega` wall fixation, the production-cap and wall-model
    machinery, the `k`/velocity mechanisms traced above, and the corner cells (hexahedra also own two
    boundary faces at the duct's edges) are therefore all exonerated as the cause; what distinguishes
    the tetrahedral case is non-orthogonality / skew (or tetrahedral cell shape), acting through the
    second pass. Open: how much skew it takes -- a controlled perturbation of this hexahedral mesh would
    give the threshold directly.

  * **SKEW ALONE DOES NOT BREAK IT: the perturbed hexahedral duct marches with the full scheme up to
    the tetrahedral mesh's non-orthogonality.** `of_case/perturb_mesh.py` displaces the nodes of the
    60 x 6 x 6 hexahedral duct by a random fraction of a cell (interior nodes in 3D, face nodes tangentially,
    edge nodes along the edge, corner nodes fixed, so the duct is unchanged; one realization, seed 0).
    Full `multcorr repaired`, own anchor from the plug at Re/10, `refresh_on_cycles=3`, freezes and
    `LogScalars` on:

    | perturbation | non-orth mean / p99 / max | skewness mean / max | full scheme |
    |---|---|---|---|
    | 0.2 cell | 7.7 / 17.3 / 22.8 deg | 0.045 / 0.15 | `CONVERGED`, 13 steps, `alpha = 1`, 6--9 cycles |
    | 0.3 cell | 11.6 / 26.9 / 37.4 deg | 0.068 / 0.24 | `CONVERGED`, 13 steps, `alpha = 1`, 6--9 cycles |
    | 0.4 cell | 15.6 / 37.7 / 53.6 deg | 0.093 / 0.42 | `CONVERGED`, 13 steps, `alpha = 1`, 6--9 cycles |
    | tetrahedral duct | 18.2 / 41.4 / 51.0 deg | 0.20 / 1.09 | diverges |

    The first pass and `CorrectedGreenGauss` also converge at 0.4 (identical trajectories to a few per
    cent). 0.45 is unusable (a face angle of 126 deg: near-inverted cells). **Caveats: the perturbed hex
    mesh reaches the tetrahedral mesh's non-orthogonality but only about half its skewness (mean 0.09
    against 0.20, max 0.42 against 1.09), so this does not show that skew of the tetrahedral mesh's
    size is harmless.** Its edge cells own two boundary faces but are NOT underdetermined
    (`max|M2^-1|` 2--3, no repair needed at any level), unlike the tetrahedral corner cells.

  * **✅ NOT THE CORNER CELLS: tetrahedral meshes with NO corner cells and NO underdetermined cell still
    fail with the full scheme, while the first pass converges on them.** `TET_MESH_OPTIMIZE=netgen` in
    `of_case/make_mesh.py` runs gmsh's Netgen optimizer, which removes every cell owning two or more
    boundary faces (the corner cells the case was built to have). Same duct, same generator settings,
    own anchor from the plug at Re/10, `refresh_on_cycles=3`, freezes and `LogScalars` on:

    | mesh | cells | corner cells | non-orth mean / max | skewness mean / p99 | `max|M2^-1|` (unrepaired closure) |
    |---|---|---|---|---|---|
    | original tetrahedra | 2462 | 176 | 18.2 / 51.0 deg | 0.20 / 0.60 | 3.3e16 (8.4e3 repaired) |
    | optimized, `MESH_SIZE` 0.007 | 3015 | **0** | 14.4 / 48.5 deg | 0.19 / 0.54 | 4.5e2 (43 cells above 10) |
    | optimized, `MESH_SIZE` 0.008 | 2369 | **0** | 14.4 / 58.4 deg | 0.20 / 0.61 | 9.0e2 (53 cells above 10) |

    **Full `multcorr repaired` fails on both**: 3015 cells -- `|R|` 0.80 -> 0.37 (step 3) -> 2.06 ->
    230 by step 24, `alpha` collapsing to 0.001--0.03; 2369 cells -- `|R|` 1.12 -> 0.85 by step 11 and then
    FROZEN bit-identical (0.85363) with `alpha = 0` through step 24. **`multcorr first pass` on the 3015-cell
    mesh converges in 13 steps with `alpha = 1` after a half step** (0.62 -> 2.1e-5). The mesh-quality
    and `M2` figures for the two optimized meshes were computed before a machine crash and the meshes
    regenerated afterwards (cell counts identical, 3015 and 2369); they were not recomputed. So the
    underdetermined corner cells (and their repair) are exonerated, together with the hexahedral result
    above: the failing ingredient is tetrahedral cell shape or its skewness (0.19--0.20 mean against the
    perturbed hexahedra's 0.09), acting through the second pass, in the mesh interior. Still unseparated:
    skewness itself against the tetrahedral cell type (a hexahedral mesh with tetrahedral-level skewness,
    or a structured tetrahedral mesh, would separate them).

  * **The first-order gradient error the second pass has to cancel is several times larger on tetrahedra**
    (measured, no march; per-cell `|gradient_defect| / h`, the gradient error of the linear-exact first
    pass on a quadratic field per unit Hessian, over the cell size; median / p90 / max, with the mesh's
    `max|M2^-1|` p90 / max): orthogonal hexahedra 0.13 / 0.18 / 0.22 (2.3 / 2.3); hexahedra perturbed by
    0.4 cell 0.28 / 0.39 / 0.78 (2.5 / 3.1); corner-free tetrahedra (3015 and 2369 cells) 0.77--0.79 / 0.93
    / 1.4 (3.7--4.1 / 450--900); original tetrahedra 0.80 / 0.90 / 1.2 (5.7 / 8.4e3). The reason is
    structural: on a hexahedron the faces come in opposite pairs, so the linear-interpolation error of the
    raw sum cancels at leading order (zero for a uniform orthogonal interior cell), while a tetrahedron has
    no opposite face, so the error is first order in `h` and comparable to the whole within-cell variation
    of the gradient it is meant to resolve. The second pass then cancels it through a Hessian built from
    DIFFERENCES of neighbouring cells' own (rough, first-order-wrong) gradients.

  **LAMINAR CONTROL: the second pass stops the march on tetrahedra even for a resolved, near-quadratic
  field -- the "quadratic exactness makes it harmless" reading is REFUTED**
  (`laminar_duct_probe.py`; `solve_flow_march`, the flow-only march on the same staged driver as the
  turbulent case; Re_Dh 50, U 1, mu 5e-4, FirstOrderUpwind, no turbulence; complete-LU materialized
  preconditioner, `DualTimeLoop(5, 1e-2)`, `RetryPolicy(on_alpha=0.01, beta_factor=2)`, wall-tapered seed
  and a mu x 10 anchor rung, row-scaled convergence `rtol` 1e-6; 30 steps per rung). Original tets
  (2462 cells): corrected Green-Gauss converges (50 s), multiple-correction FIRST PASS converges
  (41 s), compact Green-Gauss fails, the full scheme fails (`|R|` to `inf` at anchor step 1). Corner-free
  Netgen tets (3015 cells, 0 corner cells): corrected, compact and first pass converge, the full scheme
  fails the same way. Orthogonal hex (2160 cells): all four converge, the full scheme included. So the
  failure does not need turbulence, a wall model, nonquadratic content or corner cells; it needs the
  Hessian pass on tetrahedral cells. What it does NOT distinguish: which property of the second pass
  (the `M2^-1` amplification of the rough gradient differences, or the unbounded linear map itself)
  does it; one seed and one Reynolds number only. Materialization is NOT the cause
  (coloured-probe Jacobian against `jvp`, and compiled against eager evaluation; measured from a separate session, harness not kept in the repository on this branch's gradient
  code, same 2462-tet duct, laminar flow residual, full scheme with `SkewCorrectedGradient` fallback,
  rest and noisy-plug states, three random vectors): compiled and eager evaluation agree to ~1e-16 in
  residual, `jvp` and `vjp`; the reach-3 coloured-probe Jacobian is exact to 9e-16 at rest and off by
  3e-4 at the noisy plug (exact at reach 6). On the code before `boundary_gradient_weight` the same
  checks gave a 3e-1 jit-vs-eager gap and a 3e-1 probe error on 276 rows, from inverting a numerically
  singular `M1 - B` (condition up to 4e18) -- the whole of that non-determinism. One mesh, two states.

  **THE SECOND PASS MAKES THE EXACT JACOBIAN ~300-600x WORSE CONDITIONED AND GIVES IT NEGATIVE DIAGONALS**
  (dense exact Jacobian from `jvp` on unit vectors, measured from a separate session on this branch's
  gradient code: original 2462-tet duct, 9848 dofs, laminar Re_Dh 50, `FirstOrderUpwind`, mu 5e-4; rest
  state and a noisy plug = plug + 0.05 N(0,1) on u and p, seed 1; corrected = `CorrectedGreenGauss`
  default 4 sweeps; first pass = `FirstPassOnly` of the bound `MultipleCorrectionGradient(OwnerGradient,
  fallback=SkewCorrectedGradient)`; full = that scheme; Schur complement `Jpp - Jpu Juu^-1 Jup` unscaled;
  one matrix per cell, one seed):

  | arm, state | negative diag u/v/w | negative p diag | sv max | sv min | cond | negative Schur eigenvalues |
  |---|---|---|---|---|---|---|
  | corrected, rest | 0 | 0 | 5.2e-4 | 1.2e-7 | 4.5e3 | 14 of 2462 |
  | corrected, plug | 0 | 0 | 5.0e-4 | 7.0e-8 | 7.1e3 | 2 |
  | first pass, rest | 0 | 0 | 3.8e-4 | 1.2e-7 | 3.3e3 | 0 |
  | first pass, plug | 0 | 0 | 2.5e-4 | 7.1e-8 | 3.6e3 | 0 |
  | **full, rest** | **43 each** | **103** | **6.6e-2** | 3.9e-8 | **1.7e6** | **97** |
  | **full, plug** | **30 / 16 / 15** | **105** | **5.8e-2** | 2.8e-8 | **2.1e6** | **98** |

  The first pass matches the converging corrected scheme on every measure, so the second pass alone
  produces this. The conditioning comes from the LARGEST singular value growing ~130x (entries grow
  ~130-300x) while the smallest is comparable: amplification of a few rows, consistent with the `M2^-1`
  picture. What this does NOT show: which cells the 43 negative diagonals are (near-wall, repaired or
  corner cells are not yet identified), that the negative diagonals are what stops the march (a
  dual-time shift can cover a small negative diagonal), or anything on the hex mesh or about the linear
  prediction `R(x + a d)` versus `R + a J d` at a failing step -- those arms are still to be run.

  **Who the negative-diagonal cells are** (same configuration, full scheme at rest; histograms indexed by
  ring from the wall cells, or by count): the 43 cells with a negative u-momentum diagonal sit in rings
  [34, 9] (only the wall-owning cells and ring 1; the mesh has [1040, 731, 428, 217, 46]); 27 are among
  the 176 repaired cells and 16 are not; they own boundary faces 0:9, 1:7, 2:27 (mesh 1374, 912, 176),
  all wall faces; `max|M2^-1|` median 55 (mesh median 1.8), max 2008, and only 19 of 43 exceed 100. The
  103 cells with a negative pressure diagonal are also wall-adjacent (rings [67, 36]), 43 in the repaired
  set, with median `max|M2^-1|` 3.1 -- ordinary; 33 are negative in both u and p. So the culprits are
  wall-adjacent and enriched in two-boundary-face and repaired cells and in large `M2^-1`, but a large
  `max|M2^-1|` alone does not predict the sign flip, and the repaired set does not contain them all.
  One rest state, one seed.

  **Causal replacement check on the operator** (same matrices; `FirstPassOnCells` = first pass only on the
  named cells, the full scheme elsewhere; the fallback closure's boundary handling kept for the first
  pass; rest, then noisy plug in brackets): full scheme = 43 (30/16/15) negative u-diagonals, 103 (105)
  negative p-diagonals, sv max 6.6e-2 (5.8e-2), cond 1.7e6 (2.1e6), 97 (98) negative Schur eigenvalues.
  (a) first pass on exactly the 43 negative-u cells: negative u/v/w 16 (22/3/3), p 29 (33); largest u
  diagonal and sv max UNCHANGED; cond 1.7e6 (9.7e5); Schur negatives 77 (70). Those 43 cells are not the
  source of the growth. (b) first pass on the 176 repaired cells: negative u/v/w 6 (2/1/1), p 9 (10);
  largest u diagonal 7.0e-5 (3.5e-4), about 2x the corrected scheme's instead of 240-300x; sv max 1.1e-3
  (1.6e-3); cond 3.0e4 (2.3e4), about 60x (90x) better than full and 7x (3x) the corrected scheme's; Schur
  negatives 57 (51), against 14 (2) corrected and 0 first pass only. So most of the operator damage
  (entry growth, conditioning, most sign flips) comes through the second pass ON the 176 repaired cells,
  but not all of it. This is the operator; the march with (b) is a separate measurement.

  **Laminar march with the second pass withheld on subsets does NOT follow the operator result**
  (`laminar_duct_probe.py`, `TET_ARMS`; original 2462 tets, Re_Dh 50, `solve_flow_march` anchored, same
  configuration as the laminar control above; first pass only, every field, on the named cells; the 176
  cells the fallback repaired and the 176 owning two or more boundary faces were verified to be the same
  set, 176 of 176): repaired/corner cells FAILED (`|R|` to inf at anchor step 1); `max|M2^-1| > 10` cells
  FAILED; rings <= 0, <= 1 and <= 2 FAILED; **rings <= 3 CONVERGED** (`|R|` 5.8e-10, 55 s). Ring
  populations [1040, 731, 428, 217, 46], so rings <= 3 is 98 % of the mesh. Withholding the second pass
  on the cells where the exact Jacobian is worst (b above) therefore does not make the march converge:
  the failure is not localized to the corner/repaired cells or the walls.

  **Linear-prediction test at failing steps: the Jacobian is right, the trouble is nonlinearity along a
  near-singular direction** (dense `J` from `jvp` on unit vectors, then `R(x + a d)` against `R + a J d`; measured from a separate session, harness not kept in the repository on this branch's
  gradient code; 2462-tet duct, Re_Dh 50, full scheme, REST start, Euclidean measure,
  `MaterializedJacobian(CompleteLu)` + `DualTimeLoop(3)`; states at march steps 0, 5, 20 of the failing
  march, `|R|` 1.6e-3, 5.0e-3, 0.78; dense exact `J`; `d = -J^-1 R`, the PURE Newton direction, not the
  march's shifted step; one march, no corrected-GG control): `|J d + R| / |R|` about 1e-12, so the solve
  is consistent; `|d|` = 3.2e3, 8.0e2, 6.9e4 against a state norm of about 50 (`J`'s smallest singular
  value about 4e-8 dominates `d`); the actual `|R(x + a d)|` at `a` = 1 / 0.5 / 0.1 / 0.01 is
  45 / 11 / 0.44 / 4.2e-3 at step 0 (quadratic in `a` all the way down), 1.4 / 0.36 / 1.9e-2 / 4.98e-3
  at step 5 and 930 / 239 / 12.4 / 0.770 at step 20, against a linear prediction of about 0 at `a` = 1;
  the linear model is accurate at `a = 0.01` at steps 5 and 20 and breaks by `a = 0.1`. A finite-difference
  check of `J` on the Newton direction agrees to 1e-5..1e-3 at `e = 1e-4` (the error grows as `e` shrinks:
  round-off in `R`, not a Jacobian mismatch). What it does NOT show: where the near-null mode lives; that
  the shifted march step (as opposed to the pure Newton step) meets the same mode; and whether corrected
  Green-Gauss has a comparable smallest singular value (its dense `sv min` is 7e-8..1.2e-7, the same order
  as the full scheme's 3e-8..4e-8, so a small smallest singular value alone does not distinguish them).

  **Corrected Green-Gauss control of the same test: `|d|` does not discriminate, the nonlinearity does**
  (same configuration, `CorrectedGreenGauss` default 4 sweeps, rest start, march states at steps 0, 3, 6,
  10 of a march that converges in 13, dense `J`, pure Newton `d`, one march; different marches, so only
  step 0 has a comparable `|R|`, 2.1e-3 against 1.6e-3): the Newton direction is also huge at step 0
  (`|d|` 4.7e3 against the full scheme's 3.2e3, both from a smallest singular value of 1e-7..4e-8) but
  the residual along it is essentially LINEAR out to `a = 1`: actual `|R(x + d)|` 5.7e-5 (Newton reduces
  `|R|` about 38x), `|actual - linear| / |R|` 2.6e-2 at step 0 and 1e-3 by step 10, and the `J` finite
  difference agrees to 1e-6. The full scheme's residual along its Newton direction is strongly quadratic
  (`|R(x + a d)|` about 45 `a^2` at step 0, 28000x `|R|` at `a = 1` against corrected GG's 0.03x; 277x at
  step 5), a second derivative 1e4..1e5 larger, which a step-limited march can cross only by taking
  tiny steps.

  **The near-null mode and the march's OWN step** (smallest right singular vector by inverse iteration on the dense LU, harness not kept in the repository; same configuration, rest start,
  Euclidean, `CompleteLu` + `DualTimeLoop(3)`, dense exact `J` at kept march states, smallest right
  singular vector by inverse iteration on the dense LU, march step `d = x_{k+1} - x_k` the actual
  shifted, line-searched step; one march per scheme, so the states differ and the comparison is per
  trajectory, not at equal `|R|`). Full scheme: at step 5 (`|R|` 5.0e-3) `sigma_min` 1.4e-8, vector
  energy u/v/w/p 0.12/0.10/0.13/0.66, by ring 0..4 [0.51, 0.25, 0.19, 0.05, 0.01] against the cell
  fractions [0.42, 0.30, 0.17, 0.09, 0.02], 13 % on the 176 repaired cells (7.1 % of cells), Spearman
  correlation with `max|M2^-1|` 0.02, one cell holds 10 %; at step 20 (`|R|` 0.78) `sigma_min` 6.8e-10,
  0.04/0.04/0.04/0.88, 15.6 % on repaired cells, one cell 12 %. Corrected Green-Gauss: at steps 3 and 6
  `sigma_min` 6.5e-8 and 6.4e-8, energy 98 % pressure and following the cell fractions exactly, its
  top cell 0.1 %. The march's actual step is 30-1000x larger for the full scheme (`|d|` 427 at step 5
  and 1.4e4 at step 20, against 12 and 4.9) and follows the linear model with a much larger error
  (0.2-0.5 `|R|`, against 0.02 `|R|`); at step 5 the step RAISES `|R|` even by the linear model. So the
  failure is neither a Jacobian error nor an overshoot of a wrong linearization: the shifted step itself
  has huge components along a pressure near-null direction. The full scheme's smallest singular value at
  rest (3.9e-8) is not much below corrected GG's (1.2e-7); it falls to 7e-10 ALONG the march, so the
  singularity grows with the state rather than sitting in the initial operator, and the mode is
  concentrated in one cell (10-12 %), mildly biased to the wall/repaired cells, and unrelated to where
  the second-pass correction is largest.

  **The concentrated cell** (per-cell share of that singular vector, harness not kept in the repository; same configuration, full scheme, rest-start march, dense `J`,
  smoothness ratio = sum over interior faces of `v_i v_j` / sum of `(v_i^2 + v_j^2) / 2`, +1 smooth,
  negative alternating): the SAME cell carries the top share of the vector at both steps, cell 2168:
  ring 0, two boundary faces (both wall), in the repaired set, `max|M2^-1|` 130 (mesh median 1.8, 99th
  percentile 186, max 8432), volume 0.86x its mean neighbour, only 2 interior neighbours (both ring 1,
  not repaired, `max|M2^-1|` about 1). Step 5: top-5 cells hold 10.3 / 9.1 / 9.0 / 6.4 / 2.7 % (38 %);
  the cell's diagonals u/v/w/p are +2.0e-5 / -4.2e-5 / -2.2e-5 / -1.0e-5, and its two neighbours carry
  almost none of the vector. Step 20: 12.1 / 5.6 / 3.8 / 3.8 / 3.7 %; diagonals +3.4e-3 / +3.1e-3 /
  +2.8e-3 / -5.3e-7. Smoothness of the pressure part +0.46 then +0.29 (velocity +0.41 then +0.17): a
  localized bump, not a checkerboard. Corrected Green-Gauss: every cell holds about 0.1 %, smoothness
  +1.000 (pressure) and +0.98 (velocity), the benign near-constant-pressure mode of an incompressible
  operator with a weak reference. What this does NOT show: causality (first pass on this cell or on the
  top five was not tested), and it is one cell in one march. Note the laminar march with the second pass
  withheld on ALL 176 repaired cells, which contains cell 2168, still failed, so this cell alone cannot
  be the whole mechanism.

  **The near-null mode MIGRATES when the second pass is withheld on the corner cells**
  (the march with `FirstPassOnCells` on the repaired cells, then the same null-vector map, harness not kept in the repository; same configuration, first pass only on the 176 repaired cells, rest start,
  Euclidean, `CompleteLu` + `DualTimeLoop(3)`, 60-step cap, one run): the march FAILS (`|R|` 8.1e-4 at step
  0, 5.2e-3 at step 5, a jump to 0.135 at step 6, 25 by step 59 -- a slow rise, linear-solve cycles 9-83,
  never at the cap). Near-null vector of the dense Jacobian at kept states: step 5 (`|R|` 5.2e-3)
  `sigma_min` 4.6e-8, 90 % pressure, 5.9 % on the repaired cells (7.1 % of cells: no longer enriched),
  top cell 302 (ring 0, one wall face, NOT repaired, `max|M2^-1|` 29); step 20 (`|R|` 0.32) `sigma_min`
  1.9e-9, 92.5 % pressure, top cell 1984 (ring 0, one wall face, not repaired, `max|M2^-1|` 3.4); step 40
  (`|R|` 3.6) `sigma_min` 3.4e-10, 96.8 % pressure, ring shares [0.35, 0.37, 0.20, 0.06, 0.01], top cell
  169 (ring 1, no boundary face). Pressure-part smoothness +0.36..+0.76 (corrected Green-Gauss +1.000);
  Spearman with `max|M2^-1|` about 0 throughout. So `sigma_min` still falls along the march (4.6e-8 ->
  3.4e-10; full scheme 1.4e-8 -> 6.8e-10 by step 20), and the mode moves from a corner cell to ordinary
  non-repaired wall cells 4-5 hops away and then to an interior cell: the mechanism is generic to the
  second pass on tetrahedra, not specific to corner cells. What this does NOT show: that the growth of
  `|R|` is caused by the mode (only that the two move together); states along a diverging march are
  comparable only as a trend; one march.

  **Orthogonal hexahedral control of the dense Jacobian** (same configuration and scheme definitions,
  2160-cell duct, 8640 dofs, rest and noisy plug, one seed; no cells repaired): corrected Green-Gauss,
  first pass and full scheme all have 0 negative diagonals, singular values max / min / cond 2.86e-4 /
  1.25e-7 / 2.28e3 (corrected and first pass identical, as expected) against 2.83e-4 / 1.25e-7 / 2.27e3
  for the full scheme at rest (plug 1.91e3 against 1.89e3), and 0 negative Schur eigenvalues in all
  six cases (tets, full scheme: cond 1.7e6 and 2.1e6). The full scheme's smallest mode at rest-start march
  states 3 and 6 is the smooth constant-pressure mode (`sigma_min` 7.9e-8 and 7.4e-8, smoothness +1.000
  pressure and +0.97 velocity, every cell 0.1 %). So on the orthogonal mesh the second pass is inert on
  the operator and the damage is specific to the tetrahedra. What this does NOT separate: skewness,
  corner cells and tetrahedral connectivity (`max|M2^-1|` is 2.3 everywhere on the hex). The hex march
  itself was not run in that session; its states came from a march that ran past step 6.

  **Withholding the second pass from ONE field's gradient does not rescue the laminar march**
  (`laminar_duct_probe.py` arms `second pass on velocity only` / `second pass on pressure only`, a
  `MomentumContinuity` subclass that reconstructs the other field's gradient by the first pass alone;
  original 2462 tets, Re_Dh 50, `solve_flow_march` anchored, same configuration as the laminar control):
  both FAILED (`|R|` to inf at anchor step 1). The second pass on either the velocity gradient or the
  pressure gradient alone is enough to break the march, so the two have to come off together. Residual
  smoke test at a perturbed plug: the full and first-pass residuals differ by 1.6e-3, withholding the
  pressure second pass moves it by 1.3e-4 and withholding the velocity one by 1.5e-3.

  **A smooth cap on the correction does NOT rescue the laminar march** (`CorrectionCapped` in
  `diffusion_operator_probe.py`: `g = g_first + theta (g_full - g_first)`, `theta = 1 / sqrt(1 + |g_full -
  g_first|^2 / (kappa^2 rho^2 + floor))`, `rho^2` = sum over a cell's interior faces of the squared
  difference of the neighbour's first-pass gradient from its own, `floor` = 1e-18 x the mean squared
  first-pass gradient; applied to every field; original 2462 tets, Re_Dh 50, `solve_flow_march`
  anchored, same configuration as the laminar control): `kappa` = 0.25, 0.5, 1 and 2 ALL FAILED (`|R|` to
  inf at anchor step 1 or by step 6). Even at `kappa = 0.25`, where the correction cannot exceed a quarter
  of the local gradient spread, the march diverges, so bounding the correction's size relative to the
  local spread is not sufficient. What this does NOT show: that no limiter can work (this one is smooth
  and relative to the local spread; an absolute or a face-value limiter was not tried), or which part of
  the surviving small correction does the damage.

  **First pass in the two cancelling-difference terms only does NOT rescue the laminar march -- the
  mechanism reviewer's stated falsifier fired** (`laminar_duct_probe.py` arms `stabilized damping only` /
  `stabilized diffusion only` / `stabilized both`: a `MomentumContinuity` subclass giving the multiple-
  correction FIRST-pass pressure gradient to the Rhie-Chow damping `(p_N - p_P) - interp(grad p) . d` and/or
  the first-pass velocity gradient to the momentum viscous flux, while every face VALUE -- face pressure,
  the momentum interpolation in the mass flux, the boundary closures, the momentum diagonal -- keeps the
  full scheme; original 2462 tets, Re_Dh 50, `solve_flow_march` anchored, same configuration as the
  laminar control; with both flags off the residual is bit-identical to the full scheme, and on the
  orthogonal hex the diffusion swap changes it by 1e-20): damping only FAILED (`|R|` to inf at anchor step
  1), diffusion only FAILED (step 1), **both FAILED** (`|R|` 5.0 -> 0.91 over two steps, then inf at anchor
  step 2). The reviewer's screen predicted both to converge and, if not, that the raw reconstruction gain
  is fatal in the face-value reconstructions too. What this does NOT show: which of those face-value
  consumers (the face pressure of the pressure force, the momentum interpolation in the mass flux, the
  boundary closures, the momentum diagonal's gradient) carries it -- they were not separated; nor that the
  Rhie-Chow and diffusion sign flips are innocent (they are removed in this arm and something else remains).

  **The consumer bisect, and why it CANNOT be read as attribution** (`laminar_duct_probe.py` arms
  `bisect: ...`: the base residual reconstructs every gradient by the multiple-correction first pass, which
  marches, and ONE consumer of the gradient reads the full scheme's; original 2462 tets, Re_Dh 50,
  `solve_flow_march` anchored, same configuration as the laminar control; every override changes the
  residual at a perturbed plug, by 4e-7 to 1.4e-3 against 1.6e-3 for full versus first pass): the base
  converged (`|R|` 5.8e-10, 53 s); full for the Rhie-Chow damping, the momentum viscous flux, the face
  pressure of the pressure force, the momentum interpolation inside the mass flux, the boundary mass-flux
  closure (residual change 1.7e-6) and the boundary pressure all FAILED; full for the momentum diagonal
  (4e-7) and the boundary velocity (4e-6) CONVERGED. **What it does NOT show, and the reason it must not be
  read as "these six consumers are at fault": every arm is a MIXED configuration, in which one gradient
  serves two consumers that other code pairs -- the face pressure in the momentum equation and the
  damping in continuity, the interpolated momentum and the viscous flux -- so a failure may come from the
  inconsistency between them and not from the consumer.** The same objection applies to the `stabilized ...`
  arms above (they give the first-pass gradient to the damping and the viscous flux but the full one to the
  face pressure and the momentum interpolation), so their failure does not refute the mechanism cleanly
  either. The interpretable arms are the consistent ones: whole cell subsets (the rings) and whole fields
  (the split above).

  **The outlet-cell confound in the ring ladder is refuted** (same configuration): the outlet's mass-flux
  closure and boundary pressure read the full pressure gradient only at the 40 outlet-owning cells, of
  which 12 sit at rings 3-4 from the wall-owning cells (rings [16, 12, 0, 8, 4]), so the wall-ring arms
  withheld them late. First pass on the outlet cells only FAILED, on the inlet and outlet cells FAILED,
  rings <= 2 plus the inlet and outlet cells FAILED, and rings <= 3 with the outlet cells kept FULL
  CONVERGED (`|R|` 5.8e-10, 57 s). So the second pass at the outlet cells neither breaks the march nor is
  what rings <= 2 leaves behind.

  **Also found on the way, and real on its own terms (all-Dirichlet scalar Laplace,
  `diffusion_operator_probe.py`):** the Hessian correction multiplies gradient sensitivity by `M2^-1`, so
  the repaired corner cells (`max|M2^-1|` 600--8400) push the diffusion correction to 398x the orthogonal
  diagonal and turn 43 diagonals negative (owner: 13, via neighbours of singular corner cells); first
  pass where `max|M2^-1| > 10` removes all of them. That is a Laplace-operator result; it is **not** what
  stalls the coupled march (the velocity mechanism above is, and limiting on M2 alone left all 43 omega
  diagonals negative). Mesh refinement (to 48005 cells) did not remove either.
