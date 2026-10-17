# Investigation log — `aquaflux/solve/` the observed march (`newton_march`, triggers, controls, logging)

> Split out of `solve-march.md` (2026-10-17). **Lives in `.claude/notes/`, outside the auto-loaded
> `.claude/rules/` tree, so it never auto-loads.** It holds the dated measurements and investigation
> steps behind the march's homotopy, retry, refresh-trigger and contract-placement entries, moved
> verbatim; each heading names the `solve-march.md` entry it came from. See `solve-march.md` for the
> current, load-bearing status. Symbols named below may since have been renamed or deleted (for
> example `amg_beta_tracking_refresh` and `lu_beta_tracking_refresh` went with #371) — check the
> source before quoting one as current.

## `RampSchedule` / `RampWalk` — on bfs3d it does not replace the station count (2026-10-11, #678)

("Same log" below means `.claude/notes/solve-globalization-log.md`.)

    - **On bfs3d it does not replace the station count (2026-10-11, #678).** At `grow` 3, release 1e-4,
      pace 24, `finish` 3 the ramp ended itself after step 9 (settled at step 6, 9 station changes
      against 13) and cost 187 accepted / 267 all-attempt cycles in 19 steps with five steps redone,
      against the shipped fixed `grow` 1.5 x 12's 174 / 193 in 28 with one -- better than a fixed 12 at
      `grow` 3 (214 / 459), worse than the shipped schedule. `rho1` gave no lead on the settle there at
      either `grow` (above one at `grow` 3, below one from the first floor step at the shipped 1.5). Same log, "bfs3d: the release, `grow` 3 and the settle-ended ramp".

## `ResidualHomotopy` — what it is for, and the four-arm result (pitzDaily, 2026-09-09)

  - **What it is for, quantified on pitzDaily's shipped Reynolds ladder (2026-09-09, `simplesmooth`
    flow inverse, `N_POINTS=2`, `RATIO=10`, `BETA_START=0.5`, `beta_min=0.005`, `grow=1.5`,
    `INNER_STEPS=5`, cold compile cache; 69 outer steps / 417 restart cycles / 1325 s total of which
    **396 s was XLA compilation before step 1**, so 929 s of marching; `x_r/h` 8.07 vs OpenFOAM 7.74).**
    Solving each rung as its own march pays two costs a single march does not, and BOTH are visible in
    that one log:
      * **Every rung is converged and the next rung's viscosity jump immediately undoes it.** Rung 1
        spent **12 of its 28 steps** taking `|R|` 4.5e-04 → 7.2e-06; rung 2 opened at 4.5e-02 — **six
        thousand times worse**. Rung 2 converged to 4.6e-06 and the target opened at 3.0e-03. A seed
        does not need to be a root, and those polishing steps are discarded work.
      * **Every rung restarts the pseudo-timestep ramp.** `beta` reopens at `BETA_START` per rung and
        walks down one `1/grow` notch per **outer step** — `ceil(ln(100)/ln(1.5)) = 12` steps at these
        values, measured with `a_min = 1.000` on all eleven comfortable ones, i.e. the control had no
        reason for caution and crawled anyway. Rungs 1/2/3 spent 12/15/16 steps there.
    Per rung, on the quiet re-run: 28 steps / 123 cycles / 157 s, 21 / 139 / 160 s, 20 / 155 / 170 s.
    (⚠️ The wall clocks above are from an **uncontended** run; a first attempt at the same march was
    contended by a parallel test tier and read 1325 s against 555 s for a **bit-identical** trajectory
    — same 69/417, same final `|R|` 5.771e-06. Counts survive contention; clocks do not.)
  - **⚠️ THE MEASURED RESULT, four arms, all reaching `x_r/h` 8.0686 and `nu_t` peak 417.8 (2026-09-09,
    commit 72c9a96, uncontended, warm compile cache, compiled ILU live, `simplesmooth` flow inverse,
    `PITZ_RAMP_STATIONS=4`, `PITZ_RAMP_STEPS=3`).** Read the **cycle** column: it is the cost measure
    that survives a bad clock.

    | arm | steps | cycles | wall |
    |---|---|---|---|
    | Reynolds ladder (shipped) | 69 | 417 | 555 s |
    | ramp, station brake left mis-firing | 36 | 269 | 356 s |
    | ramp, `rebase` only | 50 | 301 | 453 s |
    | **ramp, `rebase` + `redamp`** | **40** | **261** | **363 s** |

    Against the ladder that is **1.7x steps, 1.60x cycles, 1.53x wall**. The last two rows are the
    whole argument for `redamp` and are covered in its own entry below.

## `ResidualHomotopy` — "a station spans several steps on purpose" retracted (2026-09-10)

  - **⚠️ "A STATION SPANS SEVERAL STEPS ON PURPOSE" IS RETRACTED (2026-09-10). Per-step viscosity is the
    CHEAPEST schedule measured, on both cases that have run it.** The claim here was that a station
    change re-points the refresh hook (`amg_beta_tracking_refresh(...).rebind`), whose
    `BetaTrackingRefresh._full_rebuild_pending` (`forced_full["pending"]` before #661) overrides **both** gates and forces a FULL coloured-probe re-materialize —
    "the most expensive single operation in the march" — so moving the parameter every step would pay a
    rebuild per step for a handful of saved steps. The mechanism is real; **every quantitative half of
    the argument is wrong.**
    - **A full re-materialize is a FRACTION of an outer step, not the march's most expensive operation**:
      1.2–1.6 s against ~9 s on pitzDaily, 11.5 s against a ~34 s mean outer step on `bfs3d`.
    - **`24 x 1` beats `4 x 3`** — 191 restart cycles against 261 on pitzDaily (11 arms; the numbers and
      the mechanism are in `.claude/notes/turbulence-log.md`, "ViscosityRampHomotopy — schedule, per-station refresh and the anchor seed").
    - **⚠️ The rebuilds are largely SUBSTITUTIVE, not additive** — the arithmetic "n stations cost n
      rebuilds of pure overhead" double-counts, because the ladder rebuilds too, on
      `refresh_on_cycles`, and does so on nearly every step of its long target rung. Counted from both
      `bfs3d` march logs' own `pc …` lines (23040 cells, field split on, `simplesmooth`, column reach
      3/3/3/3/2/2, ILU(0)/4 sweeps, `REFRESH_ON_CYCLES=3`): the **ladder** did 33 builds / 391 s over 59
      steps (28 of them cost-triggered inner refreshes), the **ramp** 37 builds / 490 s over 32 steps.
      Comparable totals per march — the station rebuilds mostly *replace* the cost-triggered ones.
    `steps_per_station` remains the knob; what it is for is a schedule that has been *measured on its
    case*, not a standing preference for coarse stations.

## `redamp` — the spurious brake `rebase` removed was load-bearing (pitzDaily, 2026-09-09)

    - **⚠️ THE SPURIOUS BRAKE `rebase` REMOVES WAS LOAD-BEARING.** This is the case's instance of a
      general globalization lesson — *a defect that fires for the wrong reason can be the only thing
      holding the march out of a bad regime, so fixing it correctly is what breaks the case* — which is
      stated once, with what to do about it, at the top of `.claude/rules/solve-globalization.md`. The
      numbers are here; the lesson is there. Before `rebase`, the ratio rule
      mis-read each station change as divergence and doubled β at two of four changes. That was wrong
      *and* was the only thing holding β out of a regime where the frozen preconditioner stops tracking
      the operator. Removing it,
      β walked 0.5 → 0.148 → 0.044 → **0.013** (`grow ** steps_per_station` = `1.5³` = 3.375 per
      station, unopposed) into a wall at **β ≈ 0.012**: the line search collapsed to `alpha = 0`, `|R|`
      went 4.4e-03 → 4.4e-01, and the retry ladder escalated β to ~2 to recover. Cost: 50 steps / 301
      cycles against 36 / 269 for the march that never went there.

## `redamp` — corollary: the refit floor `PC_BETA_FLOOR`, not `beta_min` (2026-09 .. 2026-10-13)

    - **⚠️ Corollary, CORRECTED: the suspicious constant is `PC_BETA_FLOOR`, not `beta_min`.** An
      earlier draft of this entry said no single `beta_min` could be right here. That was wrong, and it
      came from conflating the two floors above: `beta_min = 0.005` is demonstrably *fine* — the march
      converges there, comfortably, at the settled state. What the collapse locates is the level at
      which the preconditioner **stops following** the operator, and that is `PC_BETA_FLOOR = 0.05`, a
      value chosen against no recorded evidence on this case. **Measured 2026-10-13 and dropped from
      pitzDaily's file:** with no refit floor the shipped march takes 19 steps / 144 cycles against 20 /
      154 at 0.05, same `x_r/h` (`.claude/notes/solve-open-directions.md` 11a). bfs3d dropped it too
      (2026-10-13, with pitzDaily's other march settings): without it its march took 33 steps / 155 cycles
      against 36 / 166 (#709).
      ⚠️ **"Fine" means stable, not fast (measured 2026-10-10).** `beta_min = 0.005` is reached at ramp
      step 13 and then sets the whole target station's rate: a dual-time step at a fixed shift contracts
      each mode by `beta / (lambda + beta)`, and on pitzDaily the slowest mode sits at 0.5-0.8 of the
      floor, so the station converges linearly at 0.55-0.84 per step for 15 steps. From a settled target
      state (steps 21 and 27) ONE zero-shift step reaches the stopping bar; from the arrival state it does
      not descend at all. `ShiftStrengthControl.release_floor` (opt-in) releases it after the first
      full-length step at the floor on the target: 31 -> 22 steps, 161 -> 141 cycles at the same
      `x_r/h`. See `solve-globalization.md` and `.claude/notes/solve-globalization-log.md`.

## `station_step` — its first measured verdict is negative

    - **⚠️ ITS FIRST MEASURED VERDICT IS NEGATIVE, AND THE RESULT IS WORTH MORE THAN THE SEAM.** The
      motivating use — damp a closure block harder while a viscosity ramp walks, release at the target —
      does not pay, because **the phases are not separable budgets**. The station-keyed arm got its cheap
      ramp exactly as asked (102 cycles, matching the constant-ratio arm *to the cycle*) and switching
      the target station's ratio then bought **one** cycle. The target's cost is set by the state the
      ramp hands it, and a harder-damped ramp hands over a worse one. Full data in
      `.claude/notes/turbulence-log.md`, "ViscosityRampHomotopy — momentum-only scaling and `turbulence_damping`". The seam is kept because it is the
      instrument that established that, and because nothing else can express a per-station setting.

## Reactive β-escalation — why escalation leads (the reorder, bfs3d)

    - **Why escalation leads and `retry.solver` is the FALLBACK (the reorder, measured on `bfs3d`).** The
      two retries used to run divergence-first: a NaN'd step ran the tight `retry.solver` (a 1e-4 Krylov
      solve, restart-40) and *then*, seeing its high count, the cycle bailout escalated β. On the 3D march
      that order was the single worst cost — measured on the cold `bfs3d` cold-continuation, step 28's
      primary NaN'd (α collapsed), the tight retry ground ~325 matvecs (~40 min) to recover it to finite,
      and *then* the β-escalation re-damped the same step to a clean ~5-cycle solve — so the entire tight
      grind was wasted work the escalation superseded. Reordered, the escalation fires first on the NaN,
      recovers the step cheaply, and the tight `retry.solver` fires only as a **fallback** for a non-finite
      step escalation could *not* fix — the genuine inexact-preconditioner case (loose Krylov → non-finite δ
      that a tighter Krylov, not more damping, cures), where the cost threshold is typically `None` anyway so
      escalation is absent and the divergence retry is the sole, original mechanism.

## Reactive β-escalation — the deleted proactive β-mismatch refresh and the Step-0 diagnostic (bfs3d)

    - **The proactive β-mismatch refresh this was once paired with
      (`amg_beta_tracking_refresh(beta_rel_change=…)`) is DELETED (2026-09-13, #371)**; the bailout
      escalates β *after* a solve reveals a hard operator. **The bailout is REACTIVE by necessity: a Step-0 diagnostic on `bfs3d`
      (42-step instrumented capture) showed no cheap STATIC operator property predicts a bad step** — the
      diagonal-dominance defect of the frozen shifted operator does not separate bad from good (the rung-1
      trio 10/11/12 have near-identical DD but 302 vs 12 matvecs; the hardness is non-monotone in β and
      refresh-invariant), so a predict-then-avoid β-chooser was refuted and detect-then-react is the honest
      design. Refresh for staleness, escalate β for stiffness — a cost spike does not distinguish the two on
      its own, and neither does any static probe of the operator.

## `DualTimeStep(abort_above_inner_cycles=…)` — measured on the next bfs3d march

      **MEASURED on the next march, and the prediction held where it applied:**

      | step | discarded attempt before | after | cycles saved |
      |---|---|---|---|
      | 48 | `[2, 9, 5, 43]` | unchanged | 0 (predicted 0 — the trip is on the last inner) |
      | 51 | `[2, 12, 4, 4, 4]` | `[2, 12]` | 12 (predicted 12) |
      | 52 | `[2, 15, 39]` | `[2, 19]` | 35 (predicted 39) |

      Discarded-attempt cycles 141 → 107 (a new retry at step 53 cost 13 of the 47 saved), and the
      retry region's wall fell **250 s** (step 51 −77 s, step 52 −165 s).
      **⚠️ MEASURE IT IN WALL, NOT IN THE CYCLE TOTAL.** The march's reported `cyc` is the **accepted**
      attempt's count only, so discarded work was never in it: total cycles moved 348 → 347 while a real
      250 s came out. A prediction phrased against the cycle total would read as a total miss.
      A further ~31 cycles are available if the per-solve `stagnation_iters`/`max_restarts` are brought
      down toward the threshold (still ~4–6× it, which is why one solve can eat the step budget); step 48
      is the case that needs it, since its trip lands on the last inner where the abort cannot help.

## `retry.on_alpha` / `abort_below_alpha` — the bfs3d steps 51-54 collapse (2026-08-08)

      Caught on the shipped 3D `bfs3d` field-split march (`refresh_on_cycles=3`, ILU(0)×4, plain
      aggregation, `coarse_eq_limit` 2000, `N_POINTS=2`), target rung, four consecutive steps 51–54:

      | step | β | cyc | R | a_min | flg |
      |---|---|---|---|---|---|
      | 51 | 0.0390 | 12 | 7.061e-03 | 0.000 | L |
      | 52 | 0.0780 | 10 | 6.409e-03 | 0.000 | L |
      | 53 | 0.1561 | 7 | 8.224e-03 | 0.000 | L |
      | 54 | 0.3121 | 9 | 6.365e-03 | 0.000 | L |
      | 55 | 0.6243 | 5 | 5.530e-03 | **1.000** | |

      Step 51's inner table shows what the step row cannot: inner 0 descends (α 0.974, rate 0.779), then
      **inners 1–4 all report rate 1.000 at α 0.000** — four re-solves from an unchanged iterate,
      returning it unchanged — with `limit 4.37e-10`, i.e. the **positivity cap**, not the descent test,
      is what admits nothing. The four steps cost ~233 s to cross half a decade.

## `retry.on_alpha` / `abort_below_alpha` — measured end to end (bfs3d, 2026-08-08)

      **MEASURED END TO END, and ONE retry is worth 8 steps and 199 s.** Same case and configuration as
      the baseline above (field split, `refresh_on_cycles=3`, ILU(0)×4, plain aggregation,
      `coarse_eq_limit` 2000, `N_POINTS=2`), `retry.on_alpha = 0.01`:

      | | baseline | with the trigger | |
      |---|---|---|---|
      | **wall** | 2161 s | **1959 s** | **−9.3%** |
      | steps | 66 | 58 | −8 |
      | Krylov cycles | 324 | 277 | −15% |
      | mid-span `x_r/h` | 8.361 | **8.361** | identical |

      **The run is its own control, which is what makes the attribution safe:** the two lower rungs come
      out *identical to the cycle* — Re/100 14 steps / 359 s / 45 cycles in both, Re/10 23 steps / 99
      cycles, 668 s against 671 s — because the trigger is inert wherever the line search is healthy. The
      whole difference is the target rung, 29 steps / 1131 s / 180 cycles → **21 / 932 / 133**. The
      trigger fired **once in the entire march** (step 51, `alpha`, β 0.0390 → 0.0780), and the inner
      abort cut that attempt from 5 inner iterations / 12 cycles to 2 / 6.
      **Note the cycle count DOES show this one, unlike the cost abort** — there the wasted work sat in
      *discarded* attempts, which the total never counted (348 → 347 cycles while 250 s came out); here
      it sat in **accepted** steps, so both measures agree. Which measure can see a saving depends on
      whether the work being removed was accepted or discarded, so decide that before quoting either.

## `retry.on_alpha` / `abort_below_alpha` — known waste under the old mid-step refresh

      **Known waste left on the table, as measured under the old mid-step refresh:** the mid-step cost
      refresh fired in the *same* body call as the collapse, so a doomed attempt paid it (~14.7 s) and the
      escalated retry then re-matched the preconditioner anyway. ⚠️ **Since 2026-10-10 the shape is
      different and unmeasured:** an attempt that asks for a re-fit stops there, and the escalated redo
      takes the re-fit (`due=True`) at the pre-step state and the escalated β — one re-fit, at the operator
      the redo solves, rather than one mid-attempt plus a gated re-match. While the global k cap existed,
      suppressing it would have needed to tell a constraint-bound step (where no preconditioner can help)
      from a non-descending one (where a refresh might be exactly the cure); with the cap and the
      whole-step-limit seam deleted (2026-10-14) every collapse is the non-descending kind. Measured
      evidence, under that cap, that the refresh cannot rescue a constraint-bound step: at baseline
      step 51 the mid-step refresh fired at inner 1 and took the following solves from 5 cycles to 2,
      while α stayed 0.000 and `‖G‖` did not move for three more iterations.

## `CoefficientDriftTrigger` — calibration on a cold-IC pitzDaily march (2026-07-25)

    - **CALIBRATED, and the premise validated, on an instrumented cold-IC pitzDaily march (2026-07-25 —
      this closes #17 for this case).** One logged march with no refresh trigger + `on_step`, which
      still records drift because `solve_coupled` observes whenever an observer is supplied:

      | step | 5 | 10 | 12 | 14 | **15** | 17 | 19 | 20 | 22 | 23 |
      |---|---|---|---|---|---|---|---|---|---|---|
      | drift | 0.038 | 0.057 | 0.073 | 0.092 | **0.106** | 0.137 | 0.172 | 0.189 | 0.226 | 0.243 |
      | cycles | 10 | 12 | 13 | 17 | **21** | 28 | 45 | 53 | 85 | 84 |

      **The premise holds:** the cycle count sits flat at its 9–13 floor while drift climbs steadily,
      then rises monotonically with it — so `ν_t` movement is what makes the frozen preconditioner
      expensive, and drift *leads* cost early (drift is already 5× its step-0 value at step 5 while
      cycles are still at the floor). `threshold = 0.1` fires at step 15, where cost has just doubled
      off the floor and the recirculation has formed (`x_r/h` 0.32) — before the steep part
      (45→53→85) and clear of the pre-separation regime where a rebuild was measured to make things
      *worse*.

## `CoefficientDriftTrigger` — end-to-end result at `threshold = 0.1` (pitzDaily, 2026-07-25)

    - **END-TO-END RESULT at `threshold = 0.1`, `refresh.limit = 8`, against the logged control (same
      cold IC, same everything, no refresh trigger) — a 5–8× cost win, sustained and repeatable:**

      | global step | 16 | 18 | 20 | 21 | 22 | 23 |
      |---|---|---|---|---|---|---|
      | refreshed cycles | 13 | 11 | **10** | 10 | 10 | 11 |
      | control cycles | 24 | 33 | **53** | 74 | 85 | 84 |

      ~21 s/step versus ~190 s/step, and the refreshed march was simultaneously **ahead on residual**
      (rel 2.67e-2 vs 3.03e-2 at step 23). Three further observations worth keeping:
      - **A refresh repays itself inside one step, which is why `refresh.limit` can be generous rather than
      hoarded.** The absolute figure that used to sit here, and the "~60–240 s recompile" it was contradicting,
      were both recorded without a configuration and are deleted; the recompile question is settled from the
      code above — a hierarchy refresh is a jit cache hit.
      - **It repeats across segments.** Refreshes fired at steps 15 and 30, each time on that segment's
        *own* drift accumulating from ~0 to 0.10 — the production confirmation of the per-segment
        re-basing.
      - **α improved 0.5 → 1.0** across the refresh (see the correction to the "preconditioner cannot
        change α" claim above).
    - **What the refresh does NOT fix — state this when reporting it.** The march still grinds: after the
      second refresh the residual moved ~0.3 %/step (rel ~1.77e-2) with cheap (11-cycle) full (α = 1)
      steps, and `x_r/h` crept 0.48 → 0.71 against OpenFOAM's 7.74. So the refresh solved the **cost**
      problem, not the **step-productivity** problem (#22). Its real value beyond the speedup is that the
      cost confound is now *gone*, so a β/α-control experiment finally measures what it claims to.

## `solve/strategy.py` — why the forward-step contracts moved (2026-08-15)

    - **The placement had a cost that came due twice.** `StepControl` was declared in `march.py` with
      **zero** implementations there, so `step_control.py` had to import `march` — which forbade the
      reverse, so a defaulting rule about two `solve/` objects could not be written in `solve/` at all
      and ended up in `turbulence/coupled.py`, a package away. And `implicit.py`, named for the Newton
      solver, was a de-facto contract module handing the step protocol (then a private
      `_ForwardStep`, now `NewtonStrategy`), `_within_tolerance` and
      `backtracking_line_search` across boundaries either privately or absent from `__all__` — which
      under this package's own boundary rule read as violations.

## Moved from `solve-march.md` by the 2026-10-17 prune (verbatim; the rule keeps a condensed current statement)

### `redamp` — the two floors, the `(state, β)` wall, the default `redamping = 2.0`, the refuted neighbours (pitzDaily, 2026-09-09)

    - **⚠️ TWO DIFFERENT FLOORS, and conflating them mis-states the mechanism.** `beta_min`
      (on the step control) floors the shift the **operator** is solved with; `refit_beta_floor` (on the
      preconditioner refresh, `PC_BETA_FLOOR = 0.05` on this case when measured; pitzDaily's file sets none since
      2026-10-13) floors only the preconditioner's
      **copy** — `pc_beta = max(beta, refit_beta_floor)` — so the V-cycle stays in a regime it inverts well
      while the solved system keeps the small shift it needs for pseudo-transient progress. During the
      collapse the preconditioner was built at **0.05 throughout and never saw 0.012**, so this is not
      "a shift the preconditioner cannot invert": it is the **operator/preconditioner mismatch** that
      opens once the operator's β falls below that floor.
      And the mismatch's *size* is not the discriminator either — the same march later converged
      comfortably at β = 0.005 against the same 0.05 preconditioner, a **10x** mismatch, where **4.2x**
      was fatal mid-ramp. That is the `(state, β)` point again, and it is the reason a floor on either
      quantity is the wrong repair.
    - **⚠️ THE WALL BELONGS TO `(state, β)`, NOT TO β — do NOT build a learned floor under β.** The same
      march converged at **β = 0.005** with `alpha` 1.000 at 3-5 cycles a step once the ramp had
      arrived and the state had settled — a shift that was fatal at step 13 and comfortable at step 47.
      A floor learned from the collapse would forbid exactly the low-shift convergence that finishes
      the march. Damping *while the problem moves*, and only then, is the distinction.
    - **Default `redamping = 2.0`** — one `backoff`'s worth, the control's own calibrated response to
      "this β is too aggressive". It damps at **every** change where the accident happened to brake at
      two of four, so it is the more conservative arm; measured, that is the better trade: β sits near
      1.0 through the ramp at **2 restart cycles a step** (against 21-22 for the collapsed arm), and the
      cheap steps more than pay for the four extra. **261 cycles beats the accident's 269** — the
      principled fix is not merely as good as the accident, it is better, on the measure that counts.
    - **⚠️ TWO NEIGHBOURING MECHANISMS ARE REFUTED, and this is why `redamp` is preferred rather than
      merely chosen.** A **floor under β learned from the collapse** and a **cap on the growth RATE**
      (a "ceiling", from the sibling accelerating-ramp work) both *infer* a property of the shift from
      the march's behaviour — "a collapse here means β is too small", "a brake here means the rate is
      too fast". Such an inference decays as the state moves, because the evidence was gathered at a
      state the march has since left, and the `(state, β)` datum above is a direct counterexample to
      the floor. `redamp` infers nothing: it damps during exactly the interval in which the problem is
      changing and stops when it stops, so the control is never asked to tell a wrong step from a
      changed problem — it is told, from outside, at the one moment the answer is certain.

### The observed march — nothing in the refresh machinery reads the line-search α (bfs3d analysis)

  - **NOTHING in the refresh machinery reads the line-search α, and on `bfs3d` almost nothing reads
    anything else either (source-verified against the current defaults).** Two independent refresh paths
    exist and they key on different things: the post-step `RefreshTrigger`s (`CycleGrowthTrigger` →
    `cycles` + `residual_ratio`; `CoefficientDriftTrigger` → `ν_t` drift) and the per-attempt `refresh_preconditioner` hook (`amg_beta_tracking_refresh`, which since 2026-09-13
    (#371) re-fits only on its first call and after `rebind` — its scheduled gates were deleted).
    **Neither reads `alpha`**, so a collapsed line search can only ever escalate β — it
    can never buy a rebuild — and the **only** live cost trigger is the reactive "one solve
    reached `refresh_on_cycles` restart cycles" (since 2026-10-10 that ENDS the step and the march re-fits
    before the next one, passing `due=True`; it used to re-fit mid-step). Two consequences worth holding: (a) an α-triggered
    refresh needs **no new trigger** — `refresh_preconditioner` is already called once per *attempt*, after the
    control has set β, so a rebuild on escalation would be a condition inside that hook (the deleted
    `beta_rel_change` gate was one, never measured in that role); (b) that would **not** have addressed
    the lock-ups this case hit under the global k cap, which were constraint-bound (the positivity ratchet)
    while the direction was measured accurate and the solve over-delivered against its tolerance. That cap
    was deleted 2026-10-14 and the whole-step-limit seam after it (`step_limit`, `binding_limit`), so every
    collapse now is the **constraint-free** kind (direction genuinely bad) -- the one where α *is* the right
    refresh signal, and that case is invisible to every trigger today.

### Resuming a march — the shift through `resumed_at`, and the 2026-10-08 measurement

      - **The shift goes through `ShiftStrengthControl.resumed_at(shift)` → `(clamp(shift), None)`**, which
        `staged_march` calls on the control it ends up with (the default Courant ramp for a dual-time march) when the
        `Resumption` has a shift and the control has the hook. The first step then HOLDS the shift (there
        is no previous report to adapt from — the refresh boundary's treatment) and the ramp continues from it;
        the memo is dropped, so a ratio-keyed control forms its first ratio a step later. A march with no
        control (the single-step SER march) has no shift to seed; a custom control without `resumed_at`
        starts afresh. **Measured 2026-10-08** (24 x 16 laminar channel, `mu = 5e-3`, `DualTimeLoop(inner_steps=3)`,
        default `DualTimeControl` — `beta_start` 2, `beta_min` 0.02, grow 1.5 —, row-scaled `atol = 1e-9`): fresh
        17 steps, shift 2.0 → 0.02; resumed at step 3 / 6 / 10 with the reference only, 16 / 16 / 14 steps
        (14 / 11 / 7 left), each opening at 2.0; with the shift, the first step runs at the shift the stopped
        march's last step ran at and the next two follow its ramp.
      - Refused beside a `homotopy` (its anchor is the first station's). Carried by `Checkpoint` from a step
        record (`case.md`). ⚠️ **Not carried:** the refresh budget (`refresh.limit` restarts) and a retry's
        escalation history beyond the shift the march ended on. Pinned in `tests/integration/test_flow_march.py`,
        `test_flow_march_resume.py` (the dual-time shift),
        `tests/unit/test_coupled_rans.py`, `test_root_solver.py` and `test_step_control.py`.

### Reactive divergence retry — `retry.solver` (the pitzDaily measurement)

  - **Reactive divergence retry — `retry.solver` recovers a step an INEXACT preconditioner poisons,
    without tightening every step (BUILT).** An *inexact* preconditioner can return a
    non-finite correction on the stiff operator an aggressive Courant overshoot produces, where the
    *exact* complete-LU returns a finite one — the loose default Krylov tolerance is what leaves that
    correction too inaccurate. `newton_march(retry=RetryPolicy(solver=…, divergence_cap=inf))` redoes a diverged
    step **from the same pre-step state** at the tighter `retry.solver`; the trigger is
    `RetryPolicy.has_diverged` (non-finite, or `> divergence_cap·reference` — default `inf`, i.e. non-finite only,
    because the residual legitimately *rises* during development via `β×travel`, so a tight cap would
    false-fire on the reachability descent). **The preconditioner is NOT re-refreshed on retry** — under a
    β-tracking refresh (`lu_beta_tracking_refresh`, since deleted — see the rename table in `.claude/rules/turbulence.md`) the factor is already
    fresh at this `(state, β)`, and re-factoring the deterministic factorization at the same point is a
    no-op; the failure is an under-converged *Krylov* solve, not a stale PC, so only the Krylov tolerance
    is tightened. This is orthogonal to the refresh *gating*: the retry recovers a diverged step whatever
    cadence the preconditioner was refreshed at. One retry: a still-diverged
    step breaks as before. Default a policy with no `retry.solver` is **byte-identical**, and the exact-LU path never
    triggers it. **Why it beats tightening every step:** measured on an aggressive pitzDaily ramp with an
    inexact monolithic-factorization preconditioner (since deleted; its record is in git history of `solve-direct-preconditioners.md` before 2026-10-17),
    rung-1 steps 1–7 ran on the cheap loose solver and *only* the diverged step 8 retried tight —
    recovering to the exact-LU value (ratio 9.72e-2) and tracking the LU on — instead of paying the tight
    solve on every step. Threaded through `solve_coupled(retry=RetryPolicy(solver=…))`; forward-only (raises under
    `jax.grad`, same guard as the refresh/control). Pinned by `test_newton_march.py`
    (`test_march_retries_a_diverged_step_with_the_tighter_solver`, `test_march_does_not_retry_a_finite_step`).
    On 2D the exact LU is cheaper *and* robust for free, so this is really a 3D-readiness lever (where the
    LU's fill is the wall and the algebraic multigrid is the option).

### `DualTimeStep(cycle_budget=…)` and `abort_above_inner_cycles` — design account and bfs3d measurements

    - **`DualTimeStep(cycle_budget=…)` makes the escalation CHEAP — cap the primary grind, don't run it to
      completion (BUILT).** The escalation above is reactive-after-the-solve, so its cost is set by how
      expensive the doomed primary is *before* it returns. The dual-time step runs up to `inner_steps` inner
      Newton iterations, each a restart-capped GMRES; on a grinding primary every inner ran to the
      stagnation cap, so the step burned `inner_steps ×` a full stagnation before the escalation could fire
      (measured ~5× the necessary cost on `bfs3d` — steps 11/25/27 ground ~300 matvecs where one capped
      inner is ~60). `cycle_budget` stops the inner loop once its *accumulated* Krylov count reaches the
      budget (`cond` gains `& (cycles < cycle_budget)`, elided at trace time when `None`), so a grinding
      primary is cut after ~one over-budget inner iteration and the partial iterate is handed to the
      escalation, which redoes it at a larger β where it converges cheaply. **Pair it with
      `retry.abort_above_cycles < cycle_budget`** so a capped primary's reported count trips the redo (else the
      partial non-converged step would be accepted). Good steps converge well under the budget, so they are
      byte-identical; only a grinding primary hits it. `cycle_budget=None` (default) is unbounded and
      byte-identical. Reached through `dual_time=DualTimeLoop(cycle_budget=…)` on every coupled builder
      (since #388; it was a loose keyword before); forward-only, like the escalation it
      feeds. This is Agent C's "small-budget primary + inner abort" realized as an inner-loop cost cap rather
      than a non-attainment flag threaded through every solve layer — same effect (a doomed primary costs
      ~`cycle_budget` matvecs, not `inner_steps ×` a stagnation), far smaller blast radius. Pinned by
      `test_dual_time.py` (`…cycle_budget_caps_the_inner_loop`, `…none_is_the_unbounded_step`).
      **⚠️ CORRECTION: "a doomed primary costs ~`cycle_budget` matvecs" is MEASURABLY FALSE, because the
      budget is checked BETWEEN inner iterations.** A single inner solve is bounded only by its own
      `stagnation_iters=40` / `max_restarts=60`, both ≈ the whole budget, so one solve can blow through
      it. Measured on the 3-rung `bfs3d` march at `cycle_budget=42`: the three discarded attempts cost
      **26 / 56 / 59** cycles, entering their last inner having spent only 14 / 17 / 16. The budget did
      bind — just a whole stagnating solve too late.
    - **`DualTimeStep(abort_above_inner_cycles=…)` — stop the moment the attempt is KNOWN to be
      discarded (BUILT).** `retry.abort_above_cycles` is a **per-solve** quantity, so the instant one solve
      exceeds it with the inner target unmet, `newton_march` is going to bin the whole attempt and redo
      it at a larger β. Yet the check lived only in `newton_march`, *after* the step returned — so the
      step kept running inner iterations whose results were already destined for the bin. The same
      predicate now sits in the inner loop's `cond`, and `newton_march` pushes its own `retry.abort_above_cycles`
      down via `RetryPolicy.with_inner_abort`, which calls the step's own `with_inner_abort` when the
      step is an `AbortsInnerLoop` (#281; `DualTimeStep` implements it with `dataclasses.replace`, since the
      fields are static and live in the treedef), so there is **one** number rather than two to keep
      in step.
      **It cannot bin an expensive success**, and the ordering is what guarantees that: `cond` tests the
      convergence target *before* either cost bailout, so a costly solve that brings `‖G‖` under the
      target exits normally with `reached_target` set and is kept. The empirical backing is strong on
      this case — across 64 attempts, **no kept attempt ever had a single inner solve above 10 cycles**,
      while all three discarded ones ran 12/15/43 — so the threshold separates them perfectly.
      Measured on `bfs3d`, it saved the predicted cycles wherever the trip was not on the last inner: `.claude/notes/solve-march-log.md`, "`DualTimeStep(abort_above_inner_cycles=…)` — measured on the next bfs3d march".
      **⚠️ It is NOT purely a cost change.** `refresh_preconditioner` runs per *attempt*, so truncating a
      discarded attempt changes the refresh sequence and hence the V-cycle the next step sees: the two
      marches agree step-for-step through 52 and then diverge (61 vs 62 steps, same `x_r/h` 8.36, both
      converged). Benign here, but do not describe the abort as trajectory-neutral.
      `None` (default) is byte-identical. Forward-only. Pinned by
      `test_dual_time.py::test_abort_above_inner_cycles_{stops_a_doomed_attempt_early,
      never_bins_an_expensive_success, none_is_the_unbounded_step}` and the `RetryPolicy.with_inner_abort` plumbing
      tests beside them.

### `CycleGrowthTrigger` — why the residual is a gate (pitzDaily measurements)

  - **`CycleGrowthTrigger` — cost growth is the trigger, the residual is the GATE.** Fires only when: the
    `warmup` is past; `residual_ratio <= max_residual_ratio`; and the last `patience` steps each measured
    `>= growth ×` the segment's **running-minimum** non-zero count. **Why the residual is demoted to a
    gate:** the cycle count rises for two reasons, and on this case the *wrong* one is larger. From the
    measurements above — staleness at fixed β=2 is 17 → 31 cycles (**1.8×**), while β alone at a fixed
    pre-separation state is 17 → 43 (**2.5×**). So a bare "cost has doubled" rule fires from the SER ramp
    before the flow separates, and a mis-fire is not neutral: a pre-separation refresh measured
    **43 → 83 cycles**, plus a wasted scipy rebuild and a recompile. Since `β = β₀(‖R‖/‖R₀‖)^p` is a
    function of the residual ratio alone, gating on the ratio normalizes the confound **without**
    re-deriving the schedule or widening the `stepper()` contract to return β.

### `StepOutcome.residual_norm` — the trace-count measurement and the fallback-rung account

    - **`residual_norm`** — `norm(R(phi))` at the step's own returned iterate, in the step's own measure.
      **Both drivers used to evaluate this for themselves** (the deleted `_forward`'s loop body,
      `_march_step`'s return), spending a full residual per outer iteration to recompute a number the step was already
      holding: a globalized step ends in a line search that evaluated the measure at every rung it
      walked, so the value at the rung it kept is in hand when the step returns. Measured on
      `_march_step`, **4 residual traces per compiled step down to 3** — the Newton correction's
      `R(phi)`, its Jacobian--vector product, and the ladder's body; the fourth was the re-evaluation.
      Small beside the 15--30 Krylov matvecs a step spends, so this is tidiness with a measurable edge
      rather than a speed lever, and it should not be quoted as one.
      **It rests on ONE invariant: the step's measure and the driver's are the same object.**
      The march takes its measure from `strategy.norm()`, and rebuilds the
      *step's* `residual_norm` field through `norm_builder`, so the search, the acceptance test and the
      reported norm are one measure by construction. Break that and the convergence test runs against a
      residual history measured in a different scale from the reference it is compared with.
      **The fallback rung is the trap, and it is pinned.** The ladder walks longest-first and stops at
      the first admissible rung; when *nothing* is admissible it falls back to the longest rung that is
      finite **and** within `fallback_growth` (default `100.0`) of `reference_norm`, passed several
      evaluations earlier — so `LineSearchStep` carries that rung's measure alongside the accepted one.
      Reporting the accepted rung's norm on a fallback describes a step that was not taken
      (`test_the_line_search_reports_the_fallback_rungs_own_measure`, which fails with `inf` if the
      fallback carry is dropped). Where **no** finite rung comes in under that bound the search instead
      returns `alpha = 0` and reports `reference_norm` — the iterate untouched, which is exactly its
      measure there — so the poisoned rung is never handed on as the next anchor. A search with no
      finite rung *at all* is a third case and is deliberately unchanged: it still reports a non-finite
      measure, because the divergence guard, the acceptance policy and the convergence guard all test
      that number with `isfinite`.
      **`DualTimeStep` forms it from the `G` VECTOR its last search kept (since 2026-10-17)**: its inner
      loop converges the *shifted* `G` at a held reference while the driver judges the step by the
      steady `R`, and `G(phi_next) != R(phi_next)` whenever the step moved, but `R = G − β d (phi − φⁿ)`,
      so the measure is one subtraction (`transient_term`, the same expression `G` was built with). It
      used to evaluate `R(phi_next)` afresh ("costs what the driver used to pay and saves nothing" —
      no longer true). The same carried vector is each inner iteration's Newton right-hand side, so a
      dual-time `_march_step` traces the residual **3 times** (the anchor's `R(φⁿ)`, the ladder body, the
      Krylov JVP), not 5 — pinned by `test_a_dual_time_march_step_traces_the_residual_only_where_it_must`;
      the contract and the pitzDaily compile measurement are in `solve-globalization.md` ("The line search
      returns its residual VECTOR").

### Positivity — the bfs3d two-cell `k < 0` failure, and the deleted whole-step limit

  - **Positivity is NOT carried by the shift and the divergence guard (binding — a stated invariant that
    was wrong).** The direct-scalar path documented positivity as following from the pseudo-transient
    shift plus the guard. It does not: the guard fires on a *non-finite residual*, and by the time
    `sqrt` of a negative value has produced one, the state is already poisoned. Measured on `bfs3d`: a
    march ran 62 healthy steps and died because **two cells out of 23040** took `k` to `-3.3e-4` — every
    field still finite, all of `u, p, k, omega` moving by ~1e-4, only the derived `nu_t` NaN, because
    the SST closure takes `sqrt(k)`. The line search had crawled along that boundary for four inner
    iterations (`alpha` halving `0.008 → 0.001`, residual falling under 1% each) with no way to see it.
    - **What guards it now (2026-10-14): a PER-CELL step projection, not a whole-step cap.** Every
      coupled builder installs `_closure_step_projection(coupled, omega_step_ratio)` as `step_projection`:
      `solve.composed_projection` of the k clip (`turbulence.positive_k_projection` →
      `solve.positive_block_projection`, each cell's decreasing `k` correction clipped to `tau = 0.99` of its
      own room; unconditional for a directly-solved `k`) and the ω bound (`turbulence.omega_step_bound` →
      `solve.bounded_block_projection`, `|Δ log ω| ≤ log(omega_step_ratio)` per cell; default
      `DEFAULT_OMEGA_STEP_RATIO = e` under `LogScalars`, none for a direct ω, `math.inf` = unbounded, a
      finite ratio with direct ω raises). **The bound is part of the guard, not an option:** clipping a
      cell's `k` alone leaves its ω correction solved for a `k` move no longer taken, and on `bfs3d` that
      gave full-length trials with residual ratios up to 4e91.
    - **There is no `positive_block_limit` / `PositiveBlockLimit` / `turbulence.positive_k_limit` any
      more** — the fraction-to-the-boundary *global* cap, deleted 2026-10-14. Its trap, worth keeping: a
      global cap lets one near-zero cell throttle the whole step (on `bfs3d` one or two near-zero-k
      side-wall corner cells set it to 0.001–0.002 and caused all of its redone attempts). Measurements and
      the A/B are in `.claude/notes/solve-globalization-log.md`, "bfs3d's retried steps: the global k cap,
      replaced by a per-cell clip with an omega bound (2026-10-14)".
    - **There is no whole-step limit seam either (deleted 2026-10-14, the same day, as unused):** no
      `step_limit` on `PseudoTransientStep` / `DualTimeStep` / `shifted_step`, no
      `backtracking_line_search(max_alpha=)`, no `StepOutcome`/`StepReport.binding_limit`, no
      `newton_march(stop_on_limit_stall=)` and no `_limit_collapsing` (the fraction-to-the-boundary
      lock-up stop, which only a narrowing cap could trip). A constraint the residual cannot express goes
      in `step_projection`, per entry. If a whole-step cap is ever wanted again, restore it from git
      history with its two traps: its "no cap" must not be `1.0` if anything can ask for a step longer
      than the full one (a `1.0` default is what silently disabled the line search's growth rungs from
      2026-08-15 until they were deleted as inert, 2026-10-14); and a fraction-to-the-boundary cap collapses
      geometrically once a cell sits on the boundary, which is what `stop_on_limit_stall` existed to end.
    - **BOTH strategies carry `step_projection` (fixed 2026-08-15).** It was a `DualTimeStep` field
      only, so choosing `PseudoTransientStep` silently gave up the guard. The guard protects the
      **state**, not the march, so `PseudoTransientStep` applies it identically. Default `None`, the
      unconstrained step.
    - Why not a log variable for `k`: `k = 0` is the physical no-slip wall condition, so `log k` is
      singular exactly where the mesh is finest (recorded and measured in `.claude/rules/turbulence.md`
      — the full-log form descends then freezes). `log(k+1)` is regular there but bounds `k > −1`, not
      `k > 0`, so it does **not** prevent this failure. `k = w²` would give both properties at the cost
      of a vanishing Jacobian scale at the wall.

### A self-rescaling measure — the 62-step bfs3d comparison

  - **A self-rescaling measure means two "same" residuals are NOT the same number (binding trap).**
    `newton_march(norm_builder=…)` (a `(step, state)` builder) re-derives the `RowScaledNorm` at the state each outer iteration
    *begins from* and holds it for that whole iteration. So the `R` reported at the end of step N and
    the `‖G‖` entering step N+1 measure the **identical state** in **different scales**. Measured over
    one 62-step `bfs3d` march: they differed on **every** step — up to 2× early on, converging to 1 as
    the state settled and the scales stopped moving (the convergence-to-1 is the signature that
    identifies rescaling rather than a state difference; a rung boundary shows a ~2.8e5 jump, since the
    Reynolds number changed too). Nothing is wrong here, but **never compare two residuals across an
    outer-iteration boundary** and never print them adjacent — the march log deliberately dropped a
    "from |R|=…" field from its inner-block title for exactly this reason.

### `stopping_step` / `stopping_reference` — full entry as of 2026-10-14 (#701), with the arms and test inventory

  - **`stopping_step` / `stopping_reference` — judge the STOP in another step's measure (BUILT
    2026-10-14, #701).** `newton_march(stopping_step=step -> step)` (threaded through `staged_march` and
    `solve_coupled`): once arrived, the stopping test (the loop condition, the retry loop's
    "already converged" check, `MarchResult.converged`) reads `norm_builder(stopping_step(step),
    state)(R(state))` instead of the step's own measure; every step is still steered, line-searched and
    REPORTED in its own (so `StepReport.residual_norm`, the step controls and the triggers are untouched
    -- a march with one is step-for-step identical to one without, up to where it stops). The tolerance
    is `atol + rtol * stopping_reference`, the starting residual measured in the stopping measure
    (`staged_march` takes it once at its initial state and passes it on; `newton_march` takes it at
    `phi0` otherwise); reports keep `reference_norm`. Needs `norm_builder`; `staged_march` refuses it
    beside a `resume` (whose reference was taken in the steering measure). Costs a measure build and a
    residual evaluation per step once arrived. **What the stop read is reported too (2026-10-14):**
    `StepReport.stopping_norm` / `stopping_ratio` carry the norm the stopping test compared and its ratio
    to `judged_reference()` (the starting residual in that measure) on every step -- equal to
    `residual_norm` / `residual_ratio` without a stopping step, `nan` on a hand-built report.
    `MarchLogger`'s `R` column and its `reference |R0| ... stopping at` banner read them (falling back to
    the steered pair when `nan`), so the banner re-heads at arrival in the stopping measure, and a step
    whose two norms differ gets an aside `steered R ...; R is the stopping measure` -- the per-equation
    grid stays the steered measure's terms. `case/run.py` records `stopping_norm` as the run's
    `residual`. ⚠️ Before this, a log's final `R` (bfs3d E6: 2.55e-6) was the steered norm, not the
    number compared with `atol`. ⚠️ **`aquaflux_ui`'s Run section still plots and judges
    `residual_norm`** (`run_monitor.py`); the new columns reach `history.csv` (`StepHistory.COLUMNS`
    follows `StepReport._fields`) but the page does not read them yet. Pinned by
    `test_each_report_carries_the_residual_the_stopping_test_read` (`test_newton_march.py`),
    `test_once_a_march_judges_its_stop_in_another_measure_the_log_reports_that_measure`
    (`test_march_log.py`) and `test_case_run.py`'s step count; six mutations each turn one red. **What it is for:** a materialized viscosity ramp holds
    the anchor's shift diagonal for the whole march (deliberately -- `solve.md`'s `materialized_session`
    entry), so the case was judged in the anchor's row scales, which under-read its residual by
    1.4-4x at bfs3d's end state; `solve_reynolds_ramp` passes the session's
    `stopping_step(case)`. Measured on both flagship cases (configurations in `.claude/notes/turbulence-log.md`, "#701 — the arms"):
    pitzDaily 19 / 144 identical to main on every row; bfs3d (its pre-#712 file) identical on every row
    to main's stop, then 2 more steps at `refit_beta_floor` 0.05 (176 against 166 cycles) and 1 without
    (157 against 155), to the same `x_r/h`; on #712's bfs3d file 20 / 134 against main's 19 / 126, the
    same way. **Steering in the stopping measure as well (prototype E8) lost on #712's file under the
    global k cap, and ties under the per-cell clip + omega bound that replaced it** (17 / 133 against merged
    main's 18 / 137, `.claude/notes/turbulence-log.md`, "#701 — the arms"), so it is not
    built; the march steers in the step's own measure (`.claude/notes/turbulence-log.md`, "#701 — the arms"). Pinned by `test_a_stopping_step_decides_when_the_march_stops_and_the_step_s_measure_how_it_steps`,
    `..._relative_tolerance_is_relative_to_the_residual_in_its_own_measure`,
    `test_the_staged_march_judges_its_root_in_the_measure_the_march_stopped_by` and
    `..._scales_its_relative_tolerance_by_the_stopping_measure` (`test_newton_march.py`; mutations
    caught: the stop read in the steering measure, the stop reference taken in it, the driver's final
    check in it, and the driver's relative target scaled by it). **Dismissed, not tested:** the retry
    loop's converged check reading the steering measure instead (reachable only when a retry reason
    fires on a step one measure calls converged and the other does not), and the driver handing
    `newton_march` its reference rather than letting it measure the same value at `phi0`.
    Two consumers: `MaterializedSession.refresh_preconditioner` over `BetaTrackingRefresh`, which
    re-fits on its first call, after a `rebind`, and when the march passes `due` (a cost-ended step, a
    trigger); and `solve.RebuildingSession.refresh_preconditioner` (the block-diagonal family, #386),
    which hands back an update only after a `rebind`, grafting a step rebuilt on the new problem
    (`ShiftedStep.preconditioned_as`), and refuses at build a step with `refresh_on_cycles`. Distinct from the trigger's `refresh.builder`, which restarts a *segment* with a new step.
    Pinned by `test_an_update_the_refresh_hook_returns_reaches_every_later_step_and_the_result` and
    `test_an_update_returned_before_a_cost_redo_reaches_the_redo_and_every_later_step`
    (`tests/unit/test_newton_march.py`, both mutation-checked). There is no `lu_beta_tracking_refresh` or
    `amg_beta_tracking_refresh`; both went with #371.

### `ShiftStrengthControl` / `CflResidualDualTimeControl` — the refactor verification and the 3D measurements

    - **`ShiftStrengthControl` is the shared base, and a subclass supplies ONLY `_adapt` (binding).** It
      owns the first-step seed, the hold-across-a-refresh rule, the `[beta_min, beta_max]` clamp, the
      `tree_at` swap of a `ConstantRelaxation` onto a **dynamic** β leaf (the compilation-cache-hit
      property), and `carry_beta`. The carried state is **`ShiftControlState(beta, memo, release)`** for every
      control — β is universal, `memo` is whatever the rule remembers (`None` for the memoryless Courant
      rule, the previous residual for the two ratio rules), and `release` is the base's own memory of a
      failed release of the shift floor. **Why it exists:** written three times, the bookkeeping
      drifted — `carry_beta` was byte-identical in two controls and *absent* from the third, which
      `newton_march` then probed for with `hasattr`, so that control silently dropped its escalation
      feedback; and the same class reset β at a refresh boundary where the others held it, i.e. the
      sawtooth defect fixed for `DualTimeControl` never reached it. Both were invisible because each
      class carried its own `next_step`. The refactor is verified **bit-for-bit** against the previous
      implementation over 3096 β transitions spanning all three bands, rising/falling/flat residuals, the
      clamps and both boundary paths. `CflResidualDualTimeControl` reduces exactly to `DualTimeControl`
      at infinite ratio thresholds, which is now **pinned by a test** rather than asserted in prose.
  - **`CflResidualDualTimeControl` — the combined control, grows on α but brakes on a rising residual
    (built for the 3D inexact-AMG march the α-only control NaNs).** The two single-signal dual-time
    controls fail in **disjoint** ways: `DualTimeControl` grows Δτ on the inner-loop factor α (fast) but
    is **blind to the trajectory** — it grows into an overshoot while α = 1, which diverges unless the
    linear solve is near-exact (measured: on the 3D `bfs3d` Re-continuation the α-control runs `x_r/h`
    away to 15–19 and NaNs, because the AMG V-cycle is inexact — the same "aggressive control's overshoot
    is tolerated only by a near-exact solve" rule the 2D complete-LU satisfies); `ResidualRatioDualTimeControl`
    keys growth on the steady-residual ratio (safe against overshoot) but is **blind to productive
    development** — on the `β × travel` plateau the residual is flat, so it pins β and crawls (measured:
    the 3D march *survives* the overshoot with it but takes 4 h against OpenFOAM's ~15 min). The combined
    control grows **only when both signals are comfortable** (α ≥ `grow_above` **and** the residual ratio
    ≤ `hold_ratio`) and brakes on **either** wall (α < `backoff_below` **or** ratio > `rise_ratio`), with a
    hold band between the two ratio thresholds so a noisy plateau does not oscillate — so it grows on α
    through the flat-residual development (the residual-only rule's stall) yet the residual-rise term brakes
    the overshoot the α-only rule is blind to. This is the "pair α with a step-productivity signal" lever the
    single-step α-targeter's non-convergence ceiling flagged, applied to the dual-time controls. State is
    `(β, previous residual)` — the `beta` and `memo` of the shared `ShiftControlState` carry — across refreshes like
    `ResidualRatioDualTimeControl`; opt-in via
    `solve_coupled(step_control=…)`. Unit-tested in `test_step_control.py` (grows on α at a flat residual,
    brakes on a rising residual at α = 1, brakes on an inner clip, holds in the band, carries β). The ratio
    thresholds are march-calibrated numbers — set them from a logged march, not intuition (the 3D
    development overshoot shows ratios ~1.14).
