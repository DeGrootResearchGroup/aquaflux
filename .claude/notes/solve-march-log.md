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
