---
paths:
  - "aquaflux/solve/saddle_multigrid.py"
  - "aquaflux/solve/shift_basis.py"
---

# Rules — `aquaflux/solve/` the flow block (native `[u, v, w, p]` saddle preconditioning)

> Split out of `solve.md` (2026-08-18). See `solve.md` for the package-wide contracts, current
> configuration, and general binding decisions this file assumes.

## The flow block — native preconditioning of the `[u, v, w, p]` saddle

**Current status (2026-08-18): the native path is BUILT and validated for differentiation, but is
NOT the shipped default on a march — `FLOW_INVERSE` stays `petsc`.** The two subsections kept here are
the durable ones: that `jax.grad` runs and is validated through this block, and the final honest
verdict on why the native hierarchy does not currently win end to end. The full chronological
investigation between those two points — several rounds of "found a win", each later qualified or
retracted by a tighter measurement — is preserved in full in `solve-flow-block-log.md`; read it before
proposing to revisit the native flow block, so a re-litigated idea is not mistaken for a new one.

## The flow block — native preconditioning of the `[u, v, w, p]` saddle

### ✅ `jax.grad` RUNS ON THIS CASE, AND IS VALIDATED — the first gradient ever taken here (2026-08-15)

**Read this before citing any zero-shift result in this section as an adjoint result.** The whole
section is justified by the transpose solve behind every gradient meeting the unshifted operator — but
every other number in it is a **linear probe**, driven by the right-hand side `-R(state)`
(`field_split_probe.py`). The actual adjoint had never been executed. It now has been, it works, and it
agrees with a finite difference. Harness `validation/bfs3d_openfoam/adjoint_probe.py`.

*Configuration, every run below:* `state-00067` started from its **physical** fields
(`coupled.physical_fields`, not `layout.unpack` — the checkpoint holds `log(omega)`), `forward_rtol`
0.3, β = 0 throughout, field split, **shipped** column reach `(3,3,3,3,2,2)`, native nodal trailing
inverse, `coarse_eq_limit` 2000, positivity floor 1e-8, preconditioner built once on concrete
parameters, adjoint solver `relative_residual_gmres(1e-6, restart=120, max_restarts=150)`. Objective
`sum(k^2)`; the differentiated parameter scales the molecular viscosity.

**TWO things blocked it, and only the first was known.**

1. **The API gap.** `solve_coupled` did not expose `adjoint_solver`, so the transpose solve ran at
   `default_linear_solver()` = lineax's stock restart and stagnation budget and raised *"A stagnation in
   an iterative linear solve has occurred. Try increasing `stagnation_iters` or `restart`"* — the remedy
   the error names was unreachable from the coupled entry point. Threading it (`solve_coupled(
   adjoint_solver=…)`) removes that failure mode outright.
2. **The BUDGET, which is the part nobody had measured.** With the knob threaded, the transpose solve
   needs **~1450 preconditioner applications**; the first attempt gave it 60 restart-15 cycles ≈ 900 and
   died on `The maximum number of solver steps was reached`. The operator was never intractable — it was
   under-resourced by ~1.6×, and the budget had been sized from the stale zero-shift figures below.

**THE GRADIENT, and the finite-difference check.** The mismatch at a loose root is severe and is
entirely the *finite difference's* fault — the adjoint barely moves while the difference walks onto it:

| forward root `rtol` | central difference (`eps` 1e-4) | adjoint | gap |
|---|---|---|---|
| 2e-2 | −2.5807e+03 | −3.1793295e+03 | **23.2 %** |
| 1e-3 | −3.1964e+03 | −3.1793668e+03 | 0.53 % |
| **1e-4** | **−3.1787556e+03** | **−3.1793669e+03** | **1.9e-04 — agrees** |

- **The adjoint is stable to 6e-8 relative across the last two rows** while the difference moves 24 %.
  So the gradient was right throughout, and every "mismatch" was a difference taken between two solves
  that were **not at their roots** — the implicit-function-theorem adjoint is only valid at `R = 0`, and
  a 2 % relative residual is nowhere near it.
- ⚠️ **A LARGER `eps` MAKES IT WORSE, NOT BETTER — so "the finite difference is noisy" is the wrong
  mechanism, and the arithmetic that seems to confirm it is a coincidence.** At `rtol` 2e-2 the gap goes
  **23.2 % at `eps` 1e-4 → 64.2 % at `eps` 1e-3**. Solve *noise* would fall as `1/eps`; what actually
  happens is that a bigger parameter step moves the two solves' stopping points further apart. The
  seductive trap: the `eps` 1e-4 discrepancy (599) matches `objective-noise / 2 eps` to three figures,
  which reads as proof of noise and is not. **Fix the root, not the step size.**

**THE ADJOINT'S COST, measured — and this is the yardstick the campaign lacked.**

| arm at β = 0 (`rtol` 1e-3, otherwise identical) | adjoint applications | derived cycles | per application |
|---|---|---|---|
| **PETSc ILU(0) flow block (incumbent)** | **1454** | **12.0** | ~195 ms |
| native SIMPLE flow block, shipped settings (**4** sweeps) | 1696 | 14.0 | ~383 ms |
| native SIMPLE flow block, **8** sweeps | **1696** | **14.0** | ~607 ms |

- **The native flow block does NOT beat the incumbent on the real adjoint** — ~17 % more applications
  *and* ~2× the cost per application, so ~2.3× the work. This is the one arm the native-preconditioner
  programme exists to test and it had never been run; the recorded 7-against-11 zero-shift win is a
  **linear-probe** result at right-hand side `-R`, not this.
- **⚠️⚠️ THE COUNT IS A PROPERTY OF THE COARSENING, NOT THE SMOOTHER — THREE DIFFERENT SMOOTHERS ON THE
  SAME HIERARCHY ALL GIVE 1696 (measured 2026-08-16).** A fourth arm — the same native hierarchy applied
  on the host and smoothed by a **zero-fill incomplete factorization** (`solve/ilu0.py`, one sweep,
  `BFS3D_FLOW_INVERSE=hostilu`) — returns **1696 applications, 14.0 derived cycles**, identical to the
  SIMPLE-smoothed arm at 4 sweeps and at 8. So on the adjoint's operator and right-hand side, SIMPLE ×4,
  SIMPLE ×8 and an incomplete factorization ×1 are indistinguishable, and all three sit ~8 % above
  PETSc's 1575 at the matched `rtol` 1e-4 (see the iteration-count-independence table below for that
  baseline; the 1454 above is the looser `rtol` 1e-3 root). **What separates the native arms from PETSc
  here is the aggregation, and ours is the worse one for this right-hand side.**
  *Configuration:* `state-00067`, β = 0, uniform column reach, `rtol` 1e-4, `forward_rtol` 0.3, adjoint
  solver `relative_residual_gmres(1e-6, restart=120, max_restarts=150)`, native coarsening
  `strength_threshold` 0.25 / 5 levels / `max_coarse` 500 / no singletons. Per application **~170 ms**,
  measured by differencing heartbeats (100→800 applications at a flat 17 s per 100), so this arm is
  marginally *cheaper* per application than the incumbent's ~195 ms and the ~8 % extra applications make
  it roughly a wash on wall clock — inside this case's noise floor either way.
- **⚠️⚠️ AND THIS IS THE CASE THAT PROVES A LINEAR PROBE CANNOT RANK ADJOINT PRECONDITIONERS.** The same
  two arms, at the same state and the same β = 0, measured through `field_split_probe.py` at right-hand
  side `−R`: native **4 restart cycles against PETSc's 11**, a 2.75× win. Measured on the actual
  gradient: **1696 against 1575, a 1.08× loss.** The ranking inverts and the magnitude is out by ~3×.
  The reason is already recorded a few lines above and is now demonstrated rather than argued: **the
  adjoint's right-hand side is the cotangent `dL/dphi*`, localized in one field block, not the full
  steady residual** — and the restart differs too (120 against 15). **Do not promote a `−R` linear probe
  to an adjoint result, in either direction.**
- **✅ The gradient is IDENTICAL to every printed digit — −3.179366936e+03 from both arms**, which is the
  correctness check behaving exactly as it must: a preconditioner changes how the transpose solve reaches
  the answer, never where it lands. It is also the first end-to-end exercise of the hand-written
  `HostVCycleInverse` transpose and `Ilu0.solve(transpose=True)` on a real adjoint rather than on a unit
  fixture, and they reproduce PETSc's gradient exactly.
- **⚠️ DOUBLING THE SMOOTHER SWEEPS BUYS EXACTLY NOTHING HERE — 1696 applications either way, not one
  cycle different, at 1.59× the cost per application. So 8 sweeps is STRICTLY DOMINATED by 4 on this
  operator.** That is worth stating loudly because this file's standing rule is the opposite one —
  *"never quote an arm at one smoother-sweep count"*, which earned itself twice when a sweep ladder
  reversed a verdict. On the adjoint's operator the ladder is flat, so the rule's usual remedy (run more
  sweeps before believing a native arm lost) does not apply.
  **The setting is verified to have taken effect, which matters because a no-op produces the same
  identical count:** the coarsening line is unchanged (sweeps do not touch the coarse space, as
  expected) while the measured per-application cost moves 383 → 607 ms (400→800 applications in 153 s
  against 243 s). Cost changed, convergence did not.
- **All three arms' gradients agree to 9–10 significant figures** (−3.179366754e+03 for PETSc and for
  native-8, −3.179366759e+03 for native-4), which is the correctness check behaving exactly as it must:
  a preconditioner changes how the transpose solve reaches the answer, never where it lands.
- ⚠️ **Applications, not cycles, and not wall clock.** A cycle count is a fair proxy only when the
  candidates share a per-application cost, and these arms explicitly do not — the same trap that nearly
  killed the field split (31 % faster end to end *while taking 11 % more cycles*). Wall clock across
  these runs is **contended** (another session ran a test tier throughout) and is not quotable; the
  application counts are contention-immune.
- **Untested, and the honest remaining scope:** only the sweep count was varied on the native side. The
  splitting, `pressure_sweeps`, `omega`, the strength threshold and the level count are all at the
  case's shipped values, and this says nothing about them. What it does establish is that the *cheapest*
  lever on that list — more smoothing — is spent.

**⚠️⚠️ DO NOT SIZE AN ADJOINT SOLVE FROM THE ZERO-SHIFT FIGURES BELOW — treat them as stale until
re-measured.** The 60-restart budget that failed was chosen *from* them (11 cycles at uniform reach, 22
at the shipped one) on the reasoning that 60 leaves ample headroom. That reasoning is unsound twice
over: they are **linear-probe** numbers at a different right-hand side, and they predate much of the
preconditioner work this section records — the smoother, the aggregation, the field split, the trailing
inverse and the column reach have all moved since parts of them were taken.

**The adjoint's right-hand side is NOT `-R`** — it is the cotangent `dL/dphi*`, here `2k`, **localized in
a single field block**, where every linear probe in this file uses the full steady residual. That
difference is why the adjoint's 12 cycles and the probes' 11 are not the same measurement even when the
numbers look alike.

**Instrumentation, and why it counts applications rather than cycles.** The restart-cycle count
`solve_linear` returns is **discarded inside `_implicit_solve_bwd`**, which has no observer, so the
adjoint's cost is unreachable from outside. `adjoint_probe.TransposeApplyCounter` swaps the host
preconditioner's `factors` attribute for a delegating proxy (the same mutation an in-place refresh
performs, seen by an already-compiled solve for the same reason — the callback reads the attribute
rather than capturing it) and counts **transposed** applications. Forward and adjoint share one
factorization, so counting the transpose makes the split exact and free: the forward march never applies
it. Cycles are then *derived*, not measured. A heartbeat every N applications makes the solve watchable;
it separates running from hung, **not** converging from stagnating.

**✅ ITERATION-COUNT INDEPENDENCE IS DEMONSTRATED (2026-08-15) — the coupling analogue of Gate C, and the
one thing a finite-difference check cannot substitute for.** A matching finite difference shows the
derivative is *right*, not that it came from the implicit-function-theorem solve rather than the march
taped onto the tape. The designed test is to repeat the gradient from a **different starting iterate** —
which changes the forward step count while leaving the root, and therefore the gradient, alone. Run at
identical settings (`rtol` 1e-4, `forward_rtol` 0.3, PETSc flow block, adjoint restart 120 to 1e-6),
varying only which checkpoint the solve starts from — `state-00067` is step 28 of the target rung and
`state-00066` is step 27, one step further out on the same trajectory:

| start | forward applications | adjoint applications | adjoint cycles | gradient |
|---|---|---|---|---|
| `state-00067` | 1944 | 1575 | 13.0 | −3.179366936e+03 |
| `state-00066` | **2262 (+16 %)** | **1454 (−8 %)** | 12.0 | −3.179366945e+03 |

- **The forward path got LONGER and the adjoint got CHEAPER.** The cost moved *opposite* to the step
  count, which is stronger than a flat number would have been: a taped reverse pass cannot do that, since
  its cost is the forward pass's by construction.
- **The gradients agree to 9 significant figures from two different starting iterates**, confirming both
  solves landed on the same root — which is what makes the cost comparison a comparison of the *same*
  transpose problem rather than of two different ones.
- **Over all four runs on this case the separation is wide:** forward applications span 1070 → 2262
  (2.1×) while the adjoint stays within 1454–1575 (±8 %) and is **non-monotone** in the forward count.

⚠️ **What the ±8 % is, so it is not misread as drift.** The adjoint's operator is the Jacobian *at the
root*, and `rtol` is relative to each solve's own `‖R₀‖` — so a start further out stops at a slightly
different absolute residual and the operator differs slightly. That is a root-quality effect, not an
iteration-count one, and it is bounded by the same ±8 % seen when the *same* start is converged to
`rtol` 1e-3 against 1e-4.

**Standing configuration for every measurement below**, because none of them mean anything without it:
`bfs3d` `state-00067` (converged, `|R|` 3.586e-06, written at march shift 0.0064), **operator and
preconditioner both at `beta = 0`** — the operator the implicit-function-theorem adjoint solves, and the
only state in this set that discriminates between candidates. Real right-hand side `-R(state)`, GMRES
restart 15 judged on the **TRUE** residual, uniform stencil reach 3, field split with ILU(0) on the
trailing half, harness `validation/bfs3d_openfoam/field_split_probe.py`.

⚠️ **THE INCUMBENT IS THE FIELD SPLIT, NOT THE MONOLITHIC ARM — and calling the monolithic one "shipped"
here cost a day of comparisons against a bar 45 % too slow.** The shipped bundle runs `field split True`
(`compare.py`, and the bundle table at the top of this file), so the arm that differs from a native
leading-block candidate in exactly one place is `split flow/ilu0`. Both reach 11 restart cycles at this
state, but they are **not** interchangeable as a baseline:

| ILU(0) arm at `state-00067`, β = 0 | cycles | solve |
|---|---|---|
| monolithic, six fields | 11 | 42–46 s |
| **`split flow/ilu0` — the matched incumbent** | 11 | **29 s** |

Quote the split arm. A candidate that replaces only the leading block must be measured against the arm
that differs only in the leading block; the monolithic number flatters it by ~45 %.

⚠️ **Two restart caps are in play and residuals across them are NOT comparable:** `max_restarts` 60
(reported as 58 cycles) and 20 (reported as 18). A failing arm runs its cap out, so it costs 3–4× a
converging one — measured, the Krylov solves are ~80 % of a run's wall clock and the fixed setup (case
build, materialize, compile) is a constant ~95 s. Each run therefore carries its own controls.

### Equation (39): the Frobenius-optimal diagonal approximate inverse — the one transferable result

Jemcov & Maruszewski (ECCOMAS/WCCM 2008) minimize `||I - F~^-1 F||_F` over **diagonal** approximate
inverses and obtain the closed form `F~^-1_ii = F_ii / ||F_i||^2`, against Jacobi's `1 / F_ii`. Written
as a ratio the two differ by `F_ii^2 / ||F_i||^2`, which is at most one and is the fraction of row `i`'s
energy sitting on its diagonal — so **the optimal choice is Jacobi with an automatic, per-row
under-relaxation.** That is the derived form of a quantity this solver otherwise sets by hand in at least
three places: a velocity under-relaxation, a preconditioner-only velocity-row shift floor, and the
relative velocity-row relaxation the closest published work on this discretization never manages to drop.

On the `bfs3d` flow block that per-row factor runs min 3.9e-03, **median 0.53**, max 0.998 — a median
47 % under-relaxation. On a small 2D coupled channel it is 0.88–1.0, i.e. nearly inert, which is why a
small case cannot screen this.

**As the velocity predictor inside a SIMPLE smoother it is worth four orders AND changes the sign of the
sweep** (58-cycle cap):

| smoother's `F~^-1` | 2 sweeps | 4 sweeps |
|---|---|---|
| Jacobi `1 / F_ii` | 4.015e-01 | 9.758e-01 |
| Frobenius `F_ii / ||F_i||^2` | **4.281e-05** | **9.399e-06** |

9400× at two sweeps, with nothing else changed and the same 5 s build. **The qualitative change matters
more than the magnitude: under Jacobi more sweeps make it worse (the sweep amplifies), under Frobenius
more sweeps help (it contracts).** A relaxation is exactly what an amplifying sweep lacks, and this
supplies it without a tuned constant.

**On the SCHUR relaxation the answer is sweep-count dependent, and one count gets it backwards** — at 2
sweeps Frobenius is 5.200e-05 against Jacobi's 4.281e-05 (1.2× **worse**); at 4 sweeps it is 3.990e-06
against 9.399e-06 (2.4× **better**). Recorded because the 2-sweep number was written up as the verdict
before the 4-sweep one landed; the file's standing rule against quoting an arm at a single sweep count
earned itself again here.

**The two damping sources are one knob and must move together** (4 sweeps, 20-cycle cap, so not
comparable with the table above):

| Schur diagonal | `pressure_omega` 0.7 | `pressure_omega` 1.0 |
|---|---|---|
| Jacobi | 6.774e-05 | **3.435e-01** |
| Frobenius | 6.828e-05 | **4.199e-05** |

An undamped **Jacobi** pressure sweep blows up; an undamped **Frobenius** one is the best of the four. So
the hand-set 0.7 was standing in for the derived relaxation, and stacking the two over-damps — but the
constant cannot simply be dropped, only replaced.

**Set from this:** Frobenius on **both** the velocity predictor and the Schur relaxation, with
`pressure_omega = 1.0`.

### What does NOT transfer from that paper

- **Multi-step is nearly inert here.** `N` = 1 / 3 / 10 gives 7.399e-03 / 6.690e-03 / 3.823e-03 while the
  solve goes 91 s → 262 s: **1.9× for 10× the work**, where the paper reports `N = 10` working across all
  its cases.
- **Algorithm 2 is WORSE than Algorithm 1 by 45×** (7.399e-03 against 1.632e-04), where the paper reports
  it much better. Untested hypothesis: its velocity seed is one algebraic-multigrid cycle on `F`, which is
  a good solve on a gentle case and may be poor on a developed high-Reynolds velocity block, making a
  seeded start worse than a zero one.
- **The structural mismatch is the likely cause.** Their system has a `(p, p)` block that is exactly zero
  and `D = G^T` (inf-sup-stable unequal-order finite elements); ours has a nonzero Rhie–Chow damping block
  and a nonsymmetric `D`. Their optimality result — two distinct eigenvalues, hence a Krylov method
  converging in two iterations — rests on that zero block and does not carry over. Their cases are a
  driven cavity and a bent duct: no turbulence, no separation, no pseudo-transient shift.


### ⛔ THE NATIVE FLOW BLOCK IS STILL NOT FASTER ON A MARCH — and the comparison that said otherwise was unfair

**Read this before quoting any native-versus-incumbent march number.** With the inner tolerance at its
old 1e-3 the incumbent stood at 1957 s and the native block at 1510 s, which reads as the native arm
finally winning by 23 %. It is not a comparison: **only the native arm had the `inner_tol` improvement.**
Given the same change the incumbent gains *more*, and the ordering is unchanged:

| arm at `inner_tol` 1e-2 | steps | wall |
|---|---|---|
| **petsc — the shipped default** | **59** | **1197 s** |
| native, 2 sweeps | 63 | 1510 s |

**The incumbent is 1.26× faster, so `FLOW_INVERSE` stays `petsc`.** The native direction's case rests
where it always did — on the **adjoint at β = 0**, which is a different operator and where the incumbent's
advantage (an incomplete factorization becoming near-exact as the shift raises diagonal dominance)
disappears. That case cannot be closed until `jax.grad` runs on this case at all.

**The general trap, which has now cost two wrong readings in this file:** when a change lands on one arm
of an A/B, every previously-recorded number for the *other* arm is stale, and comparing across that
boundary measures the change rather than the arms. Re-measure the baseline before ranking.

⚠️ **THIS DOES NOT OVERTURN THE RECORDED "TIGHTEN `inner_tol`" FINDING — it bounds it.** Elsewhere in
this file a *pitzDaily* (2D, ILUT, large-Δτ dual-time) march is recorded as hitting a residual floor and
over-developing because "at `inner_tol = 0.05` the implicit step is only 5 %-solved, so a large-Δτ
backward-Euler step on a half-solved system overshoots", with the fix being to tighten it. **That
mechanism is real and this sweep shows its onset:** the 0.05 arm is precisely where an extra outer step
appears. What the two together establish is that the threshold is **case- and Δτ-dependent, and that
"tight" meant about 1e-2 rather than 1e-3** — on `bfs3d` at this Courant ramp, 1e-3 bought nothing over
1e-2 while costing a third of the march. Neither number transfers to the other case; the *mechanism*
does. (That pitzDaily entry's own numbers were deleted as unrecorded, so only its mechanism was ever
citable.)

⚠️ **`INNER_STEPS` (5) is INERT and always has been** — the loop exits on the tolerance long before the
cap, at every arm above. It becomes a real knob only if the tolerance is loosened far enough for the cap
to start catching, which none of these arms reach.

