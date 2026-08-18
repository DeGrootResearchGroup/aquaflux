---
paths:
  - "aquaflux/solve/march.py"
  - "aquaflux/solve/march_log.py"
  - "aquaflux/solve/checkpoint.py"
---

# Rules — `aquaflux/solve/` the observed march (`forward_march`, triggers, controls, logging)

> Split out of `solve.md` (2026-08-18). See `solve.md` for the package-wide contracts, current
> configuration, and binding decisions this file assumes.
>
> **This file has no `-log.md` sibling yet.** If you are about to push it past ~1,800 lines, split it
> first: peel the dated/historical content into a new `solve-march-log.md` (no `paths:` frontmatter) and
> leave a current-status summary here, following the pattern in `solve-flow-block.md` /
> `solve-flow-block-log.md`. See `solve.md`'s "Where new content goes".

## The observed march — forward_march, triggers, controls, logging

- **⚠️ `MarchLogger.on_step` and `.on_checkpoint` are MUTUALLY EXCLUSIVE renderings of one event —
  wire one, never both (measured 2026-08-15).** Both call `_log`; they differ only in whether the
  injected case metrics are appended (`on_checkpoint` has the state, `on_step` does not). But
  `forward_march` calls its `observer` and `checkpoint` seams **unconditionally on every step**, so
  handing the same logger to both logs every step **twice** and double-counts the logger's own step
  and cumulative-cycle totals — measured, 2 steps produce 4 rows and `cum 8` against the correct
  `cum 4`. The shipped driver wires `on_checkpoint` and deliberately leaves `on_step` unwired; prefer
  that, since `on_checkpoint` is the strictly more informative of the two. Both docstrings now carry
  the warning.
  **This is why the planned "observation bundle" (one `MarchObserver` replacing `on_step` /
  `on_checkpoint` / `on_retry`) is NOT built and must not be planned from the premise that
  `MarchLogger` already implements it.** The method names match; the semantics do not. A bundle needs
  the two renderings reconciled first — one row per step whichever seam fires — which is a change to
  the log's output format, not a signature refactor.


- **`march.py` — BUILT (`forward_march`, `StepReport`/`MarchResult`, `RefreshTrigger`/`CycleGrowthTrigger`):
  the observed, forward-only march that drives a mid-march preconditioner refresh.**
  - **Two marches, ONE decision layer (binding — this is the shape to hold).** `_forward` (traced,
    inside `custom_vjp`, has the root guard, cannot stop early, cannot be observed) and `forward_march`
    (eager Python loop, forward-only, **no guard by design**, stops on an injected trigger, reports every
    step). They are not duplicates: `forward_march` calls the **same** `forward_step.stepper()`, the same
    `forward_step.norm()`, and the same `_within_tolerance`. The only residue is a ~6-line loop shell,
    pinned against drift by a test that both marches reach the same state on the same residual.
  - **NOTHING in the refresh machinery reads the line-search α, and on `bfs3d` almost nothing reads
    anything else either (source-verified against the current defaults).** Two independent refresh paths
    exist and they key on different things: the post-step `RefreshTrigger`s (`CycleGrowthTrigger` →
    `cycles` + `residual_ratio`; `CoefficientDriftTrigger` → `ν_t` drift) and the per-attempt
    `precondition_step` hook (`amg_beta_tracking_refresh` → `beta_rel_change` / `materialize_drift` /
    `materialize_every` / `refresh_every`). **Neither reads `alpha` or `binding_limit`**, so a collapsed
    line search can only ever escalate β — it can never buy a rebuild. And in the shipped `bfs3d` bundle
    the proactive arms are switched off by construction: with the default `BFS3D_REFRESH_ON_CYCLES=3`,
    `compare.py` passes `beta_rel_change=inf`, `refresh_every=10**9`, `materialize_drift=None`,
    `materialize_every=None`, so the **only** live trigger is the reactive mid-step "one solve reached 3
    restart cycles". Two consequences worth holding: (a) an α-triggered refresh needs **no new trigger** —
    `precondition_step` is already called once per *attempt*, after the control has set β, so a finite
    `beta_rel_change` makes a β escalation pull a matched rebuild for free; (b) that would **not** address
    the lock-ups this case actually hits, which run at `binding_limit < 1` (the positivity ratchet) where
    the direction is measured accurate and the solve already over-delivers against its tolerance. Where α
    *is* the right refresh signal is the **constraint-free** collapse (`binding_limit == 1`, direction
    genuinely bad), and that case is invisible to every trigger today.
  - **Why the early-stop could NOT go inside `ImplicitNewtonSolver` (binding — do not "simplify" it back).**
    `_forward`'s guard raises whenever the terminal state is not a root, and a trigger-stopped segment
    exits un-converged *by design*. Injecting a count-based early stop would therefore require an
    **exemption** in that guard — creating a production path that returns a non-root without raising,
    which is exactly the silent-wrong-gradient hole the guard exists to close. Chunking `_forward` with
    `max_steps=1` fails independently: it recomputes `residual_norm_0` per chunk, pinning the SER ramp at
    β₀ forever.
  - **The eager march NEVER returns the answer.** It is a pure accelerator; every staged solve ends with
    a real `ImplicitNewtonSolver.solve()` that owns the guard, the `custom_vjp`, and the result. So the
    guard has exactly one home and is unconditionally on the path that produces the returned state.
  - **Two reference norms, and conflating them freezes the march (binding).** `residual_norm_0` is
    **segment-local** (recomputed at each `forward_march` entry, handed to `stepper()` for the SER ramp);
    `reference_norm` is **global** (fixed across segments, used for the convergence test and the reported
    ratio). Substituting the second for the first pairs a refreshed, larger shift diagonal with the small
    β belonging to the pre-refresh residual — the over-damping freeze documented in `turbulence.md`.
  - **Per-step jit cache hit is mandatory, not an optimization (top implementation risk).** The per-step
    call goes through the module-level `eqx.filter_jit`'d `_march_step`, taking the `ForwardStep` **and**
    the residual as *arguments*. Two caller obligations: pass the **same** `forward_step` object across a
    segment (a rebuilt one is the intended one-off recompile per refresh), and pass a **bound module
    method** (`coupled.residual`) rather than a freshly-built `lambda`, which `filter_jit` hashes by
    identity. Retracing per step would cost the 60–240 s compile *every step* and dominate the march it
    accelerates. Pinned by a trace-counting test (extra steps add zero traces). Note the residual is
    invoked several times *within one trace* (step, line-search ladder, norm), so trace count ≠ compile
    count — assert that further steps add none, not that the total is 1.
  - **`solve_coupled` does NOT declare `grow` / `descent_backoff` / `descent_test`, and must not be
    "fixed" to (binding).** They are `coupled_continuation` parameters and reach it through
    `**continuation_kwargs`, which `solve_coupled` already splats into that same function at both of
    its build sites. Declaring them as well — which it used to — meant three extra parameters on an
    already-wide signature forwarding what was forwarded anyway. Their documented home is
    `coupled_continuation`, where the parameters actually live; pinned call-for-call by
    `test_globalization_knobs_still_reach_the_continuation_builder`.
  - **The four refresh settings are ONE injected object — `RefreshPolicy` (`solve/refresh.py`),
    exported from `aquaflux.solve` (BUILT).** `trigger` / `limit` / `builder` / `precondition_step`
    were four keyword arguments on `solve_coupled`, meaningless apart: a `limit` with no trigger bounds
    a loop that never runs, a `builder` with no trigger is called once. The derived predicates and the
    one validation are on the object — `refreshes` (trigger AND budget), `observes` (does this force
    the eager march), `segments` (= `limit + 1`), `is_last_segment`, and `require_rebuildable`, which
    raises when a caller-supplied step has no builder to rebuild it with.
    - **It is a DRIVER-level object and `forward_march` does NOT take it (binding).** The march uses
      only `trigger` and `precondition_step`; `limit` and `builder` govern the *sequence of segments*,
      which is the driver's loop. Passing the whole policy down would hand the march two fields it
      cannot act on — the "smallest sufficient collaborator" rule, and the reason this differs from
      `RetryPolicy`, which the march consumes whole.
    - **`NO_REFRESH` is the shared default instance**, byte-identical to the old all-`None` defaults.
    - **`builder` alone does NOT make a march observed**, and that is deliberate: without a trigger it
      is called once, for the initial build, which the single-stage solve does just as well. Getting
      this wrong forces the observed path — with its doubled `max_steps` budget and its ban under
      `jax.grad` — on a solve that only wanted a custom way to construct its step.
  - **The six retry settings are ONE injected object — `RetryPolicy` (`solve/retry.py`), exported from
    `aquaflux.solve` (BUILT).** `solver` / `divergence_cap` / `on_cycles` / `on_alpha` / `beta_factor` /
    `cycles_limit` used to be six parallel keyword arguments on **both** `forward_march` and
    `solve_coupled`, and they are meaningless apart: a `beta_factor` with no threshold escalates nothing,
    a `cycles_limit` bounds a loop that never runs. The three decisions taken from them read three or
    four each, so they are **methods on the policy** rather than free functions taking a subset —
    `retry_reason` (named `escalation_reason` until the 2026-08-17 split; before that
    `_escalation_reason`), `has_diverged` (was `_has_diverged`),
    `with_inner_abort` (was `_with_inner_abort`), plus `escalate`, which exists solely to keep the
    "scale the β leaf, never rebuild it from a float" rule in one place (rebuilding changes the leaf's
    dtype/weak type and recompiles the whole coupled solve on every retry).
    - **`NO_RETRIES` is the shared default instance**, so both signatures' defaults name what the default
      *means* rather than how it is spelled; it is byte-identical to the old all-`None` defaults.
    - **The ORDER stays on `forward_march`, not on the policy** — escalation first, `solver` as the
      fallback — because it is a property of the loop, not of the settings. Likewise `on_retry` stays a
      separate argument: it is *reporting*, and it belongs with the other observation seams rather than
      with the thresholds.
    - **`forward_march` takes the whole policy; a refresh policy would NOT be passed the same way.** The
      march uses all six, so the object is the smallest sufficient collaborator there. Do not extend this
      to arguments a callee does not need in full.
  - **Reactive divergence retry — `retry.solver` recovers a step an INEXACT preconditioner poisons,
    without tightening every step (BUILT).** An *inexact* preconditioner can return a
    non-finite correction on the stiff operator an aggressive Courant overshoot produces, where the
    *exact* complete-LU returns a finite one — the loose default Krylov tolerance is what leaves that
    correction too inaccurate. `forward_march(retry=RetryPolicy(solver=…, divergence_cap=inf))` redoes a diverged
    step **from the same pre-step state** at the tighter `retry.solver`; the trigger is
    `RetryPolicy.has_diverged` (non-finite, or `> divergence_cap·reference` — default `inf`, i.e. non-finite only,
    because the residual legitimately *rises* during development via `β×travel`, so a tight cap would
    false-fire on the reachability descent). **The preconditioner is NOT re-refreshed on retry** — under a
    β-tracking refresh (`lu_beta_tracking_refresh`, `.claude/rules/turbulence.md`) the factor is already
    fresh at this `(state, β)`, and re-factoring the deterministic factorization at the same point is a
    no-op; the failure is an under-converged *Krylov* solve, not a stale PC, so only the Krylov tolerance
    is tightened. This is orthogonal to the refresh *gating*: the retry recovers a diverged step whatever
    cadence the preconditioner was refreshed at. One retry: a still-diverged
    step breaks as before. Default a policy with no `retry.solver` is **byte-identical**, and the exact-LU path never
    triggers it. **Why it beats tightening every step:** measured on an aggressive pitzDaily ramp with an
    inexact monolithic-factorization preconditioner (since deleted; see `solve-direct-preconditioners.md`),
    rung-1 steps 1–7 ran on the cheap loose solver and *only* the diverged step 8 retried tight —
    recovering to the exact-LU value (ratio 9.72e-2) and tracking the LU on — instead of paying the tight
    solve on every step. Threaded through `solve_coupled(retry=RetryPolicy(solver=…))`; forward-only (raises under
    `jax.grad`, same guard as the refresh/control). Pinned by `test_forward_march.py`
    (`test_march_retries_a_diverged_step_with_the_tighter_solver`, `test_march_does_not_retry_a_finite_step`).
    On 2D the exact LU is cheaper *and* robust for free, so this is really a 3D-readiness lever (where the
    LU's fill is the wall and the algebraic multigrid is the option).
  - **⚠️ SUPERSEDED 2026-08-17: THE COST THRESHOLD NO LONGER ESCALATES, and the field is now
    `retry.abort_above_cycles`; `retry.on_cycles` does not exist. Escalation is `retry.on_alpha` alone —
    see `solve-amg-multigrid.md`'s "the cost guard and the shift escalation are now separate responses".
    The entry below describes the design before that split.**
  - **Reactive β-escalation bailout — the cost threshold ESCALATED β for a bad step, tried BEFORE the tight
    divergence retry (BUILT).** A step goes bad two ways — a *finite-but-expensive* solve (count `> N`) or a
    *non-finite* one — and on the stiff low-β saddle **both have the same cheap cure: more damping.** A
    larger β lifts the correction out of the NaN regime *and* cuts the cycle count (a stronger pseudo-time
    shift makes the same frozen preconditioner more diagonally dominant), and it is far cheaper than the
    tight-Krylov divergence retry. So `forward_march(retry=RetryPolicy(on_cycles=N, beta_factor=2.0,
    cycles_limit=2))` redoes a step whose count exceeds `N` **or** that diverged (non-finite / over
    `retry.divergence_cap`) **from the same pre-step state** with β escalated (`×retry.beta_factor`,
    re-applying `precondition_step` at the new β so a β-tracking refresh re-shifts), up to
    `retry.cycles_limit` times or until it converges/drops below `N`. It reads β off
    `active_step.relaxation_schedule.beta` (a `ConstantRelaxation` / `DualTimeStep`), so it requires a
    readable β and is inert on the default switched-evolution schedule. an unset threshold (the default) is
    **byte-identical** (and a diverged step then falls straight to `retry.solver`, the pre-reorder
    behaviour). Forward-only; threaded through `solve_coupled(retry=RetryPolicy(on_cycles=…))`. Pinned by
    `test_forward_march.py` (`test_a_cycle_spike_redoes_the_step_ONCE_and_does_NOT_escalate`,
    `…_does_not_escalate_below_the_cycle_cap`, `…_escalates_beta_before_the_tight_divergence_retry`,
    `…_falls_back_to_the_tight_retry_when_escalation_cannot_fix_divergence`).
    - **The escalation must keep `_march_step` a compile-cache HIT (binding — a measured recompile
      hazard).** The retried step redoes the (coupled, minutes-to-compile) `_march_step` at the escalated
      β, so a treedef **or aval** change on that step recompiles the whole solve every retry — which on a
      stiff region that retries most steps is ~half the march wall. β is a dynamic 0-d leaf, so the *value*
      change is fine; the trap is the leaf's **abstract value**. Escalate by **scaling the existing leaf**
      (`beta * retry.beta_factor`), never by rebuilding it from a Python float (`jnp.asarray(float(beta) *
      f)`): the latter yields a *weak*-typed float64 array whose dtype/weak_type need not match the leaf
      the step control set, and any mismatch is a cache miss. Scaling preserves the aval exactly, so the
      retried step is a hit for **any** β dtype (weak/strong f64, f32). The shipped controls happen to set
      weak-f64 (so the old form was accidentally a hit), but that was an unpinned coincidence one JAX
      weak-type-promotion change from breaking. Pinned by
      `test_a_forced_escalation_adds_no_march_step_compilations` (a strong-typed β leaf, the case the
      rebuild recompiled on). Note the `retry.solver` (divergence) fallback is a *separate*, one-off
      recompile — a distinct solver object (restart-40 vs the forward restart-15) is a genuinely different
      static key, compiled once and reused; it is not a per-step cost.
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
    - **This is the PROACTIVE β-mismatch refresh's reactive twin** — the refresh
      (`amg_beta_tracking_refresh(beta_rel_change=…)`, `.claude/rules/turbulence.md`) re-freezes the PC
      *before* a solve when β has drifted (the stale-PC cause of a spike); the bailout escalates β *after* a
      solve reveals a hard operator. **The bailout is REACTIVE by necessity: a Step-0 diagnostic on `bfs3d`
      (42-step instrumented capture) showed no cheap STATIC operator property predicts a bad step** — the
      diagonal-dominance defect of the frozen shifted operator does not separate bad from good (the rung-1
      trio 10/11/12 have near-identical DD but 302 vs 12 matvecs; the hardness is non-monotone in β and
      refresh-invariant), so a predict-then-avoid β-chooser was refuted and detect-then-react is the honest
      design. Refresh for staleness, escalate β for stiffness — a cost spike does not distinguish the two on
      its own, and neither does any static probe of the operator.
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
      byte-identical. Threaded through `coupled_amg_continuation(cycle_budget=…)` (and the shared
      `_monolithic_factor_step`, so the LU steps can take it too); forward-only, like the escalation it
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
      exceeds it with the inner target unmet, `forward_march` is going to bin the whole attempt and redo
      it at a larger β. Yet the check lived only in `forward_march`, *after* the step returned — so the
      step kept running inner iterations whose results were already destined for the bin. The same
      predicate now sits in the inner loop's `cond`, and `forward_march` pushes its own `retry.abort_above_cycles`
      down via `RetryPolicy.with_inner_abort` (using `dataclasses.replace`, not `eqx.tree_at` — the field is static,
      so it is in the treedef, not among the leaves), so there is **one** number rather than two to keep
      in step.
      **It cannot bin an expensive success**, and the ordering is what guarantees that: `cond` tests the
      convergence target *before* either cost bailout, so a costly solve that brings `‖G‖` under the
      target exits normally with `reached_target` set and is kept. The empirical backing is strong on
      this case — across 64 attempts, **no kept attempt ever had a single inner solve above 10 cycles**,
      while all three discarded ones ran 12/15/43 — so the threshold separates them perfectly.
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
      **⚠️ It is NOT purely a cost change.** `precondition_step` runs per *attempt*, so truncating a
      discarded attempt changes the refresh sequence and hence the V-cycle the next step sees: the two
      marches agree step-for-step through 52 and then diverge (61 vs 62 steps, same `x_r/h` 8.36, both
      converged). Benign here, but do not describe the abort as trajectory-neutral.
      A further ~31 cycles are available if the per-solve `stagnation_iters`/`max_restarts` are brought
      down toward the threshold (still ~4–6× it, which is why one solve can eat the step budget); step 48
      is the case that needs it, since its trip lands on the last inner where the abort cannot help.
      `None` (default) is byte-identical. Forward-only. Pinned by
      `test_dual_time.py::test_abort_above_inner_cycles_{stops_a_doomed_attempt_early,
      never_bins_an_expensive_success, none_is_the_unbounded_step}` and the `RetryPolicy.with_inner_abort` plumbing
      tests beside them.
    - **A COLLAPSED STEP LENGTH is the third way a step dies, and neither the cost bailout nor the
      escalation could see it — `retry.on_alpha` / `abort_below_alpha` (BUILT 2026-08-08).** The
      escalation's two triggers are cost-with-the-target-unmet and divergence. A step whose *solves are
      cheap* and whose *residual is finite* but whose **line search cannot move** is neither, so it was
      accepted as taken, and the only thing raising β was the step control's own backoff — **one doubling
      per outer step, each doubling paying a full step to discover it was not enough.**
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
      Both halves are now built and are the same predicate in two places, as the cost bailout already is:
      **`forward_march(retry=RetryPolicy(on_alpha=α))`** escalates β (reason `"alpha"` on `on_retry`), and it is pushed
      into **`DualTimeStep.abort_below_alpha`** by `RetryPolicy.with_inner_abort` so the inner loop exits at the
      collapse instead of iterating on. `RetryPolicy.retry_reason` now owns which of the three reasons applies,
      so the decision and the string reported for it cannot disagree.
      **Both cost and step-length reasons require the target unmet; divergence does not** — a non-finite
      residual is not a result to keep because the loop happened to meet its tolerance. Both `None`
      (default) is byte-identical.
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
      **The threshold was calibrated from the baseline's own step table, not chosen:** no productive step
      went below `a_min` 0.191, all four dead ones reported 0.000 (inner collapses at 0.001 and 0.003),
      so 0.01 sits an order of magnitude clear of both. Recalibrate on another case rather than porting
      the number.
      **Known waste left on the table (deliberate, not yet fixed):** the mid-step cost refresh fires in
      the *same* body call as the collapse, so a doomed attempt still pays it (~14.7 s) and the escalated
      retry then re-matches the preconditioner anyway. Suppressing it would need to distinguish a
      constraint-bound step (`binding_limit < 1`, where no preconditioner can help) from a non-descending
      one (where a refresh might be exactly the cure) — `binding_limit` exists for precisely that
      distinction. Measured evidence that the refresh cannot rescue a constraint-bound step: at baseline
      step 51 the mid-step refresh fired at inner 1 and took the following solves from 5 cycles to 2,
      while α stayed 0.000 and `‖G‖` did not move for three more iterations.
    - **The escalated β is CARRIED into the control — so a static β floor can be dropped and the *controller*
      decides how low is safe (BUILT).** β is inverse to the pseudo-timestep, so a static `beta_min` is a cap
      on the *largest* timestep the march may take, applied everywhere — which slows convergence in regions
      that could safely take a bigger step. The escalation is the per-region feedback for "how low is safe
      *here*": it fires exactly where β went too low. But without carrying it back, the next outer step's
      `step_control.next_step` recomputes β from the control's own (floor-ward) trajectory and **re-pays the
      escalation every step** — the observed low-β tail (β pinned at the floor, each step re-escalating). So
      after an escalation `forward_march` seeds the control's carried β with the escalated value via
      `step_control.carry_beta(state, β)` — **one implementation on `ShiftStrengthControl`, over the shared
      `(beta, memo)` state, so no control can be missing it** (it once was: the deleted single-step
      α-targeter had none, and the `hasattr` guard below meant its escalation feedback vanished in
      silence). The memo is preserved across the seed, so a ratio-keyed control does not lose its
      reference and misread the next step as a huge reduction. The control then continues its grow/brake dynamics *from* the discovered-safe β, so
      `beta_min` can be driven toward zero and the controller — with escalation as the safety net and the
      carry as the memory — finds how large a timestep each region tolerates, rather than a global floor
      capping it. Only fires when β was actually escalated and the control exposes `carry_beta`; no
      escalation ⇒ byte-identical. Pinned by `test_forward_march.py`
      (`…carries_the_escalated_beta_into_the_control`) and `test_step_control.py`
      (`test_carry_beta_seeds_the_carried_state`).
  - **`CoefficientDriftTrigger` — the PREFERRED staleness trigger: measure the drift, don't infer it
    from cost (binding for new work).** A frozen preconditioner is stale exactly when the operator it
    approximates has moved, so the honest signal is that movement itself. `StepReport.drift` carries a
    **scalar** relative drift produced by `forward_march(drift_measure=…)`; the coupled RANS measure is
    `turbulence.eddy_viscosity_drift(coupled, reference_state)` — `‖Δν_t‖/‖ν_t,ref‖` — because `ν_t` is
    what the frozen k/ω transport operators are assembled from.
    - **Why it beats `CycleGrowthTrigger` (which it supersedes for this job).** The cycle count rises
      from staleness **and** from `β → 0` ill-conditioning the shifted system, and on a separating flow
      the second is the *larger* — hence that class's `max_residual_ratio` gate and its `patience`.
      Drift has neither confound: it does not respond to `β`, so **no gate**, and it moves smoothly with
      the flow instead of jumping on one stiff solve, so **no patience**. It also cannot fire before the
      flow develops — the regime where a refresh was measured to be actively *harmful* (43 → 83 cycles)
      — because an undeveloped flow is by definition one whose `ν_t` has not moved. This is the fix for
      the confound recorded as #19.
    - **The scalar is the whole design (binding).** Drift needs the *state*, but the state must not
      reach a trigger: `checkpoint` carries state and `observer` carries only numbers precisely so a
      trigger stays a pure function replayable against one logged march (see below). Reducing drift to a
      number on the report keeps that property *and* lets the trigger see the physics — so calibration
      is still "log one march with `trigger=None` and a `drift_measure`, replay thresholds offline".
      Do **not** widen the trigger interface to take the state.
    - **The measure must be RE-BASED at every refresh (binding — a real trap).** `solve_coupled` builds
      a fresh `eddy_viscosity_drift` per segment against that segment's starting state, which *is* the
      state the current preconditioner was frozen at. Carrying one measure across segments would keep
      reporting drift the refresh had already absorbed, so the trigger would re-fire on the next step
      and burn the whole `refresh.limit` in consecutive steps. Same discipline as the segment-local
      `residual_norm_0`. Pinned by
      `test_the_drift_measure_is_rebased_at_every_refresh`.
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
    - **The shipped default was originally 0.5 and would never have fired on this march** (drift
      reaches only 0.24 by step 23). A "conservative" placeholder chosen by intuition was not
      conservative — it was inert. Trigger numerics must come from a logged march, not from judgement
      about what sounds safe; the replay procedure exists precisely because that judgement is unreliable.
      Calibrated on **one** geometry, so treat 0.1 as a starting point elsewhere, not a constant.
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
  - **Zero-count trap (real, pinned).** `stepper()` reports `0` for a fully-rejected step, and a direct
    solver reports `0` too. A running-minimum baseline of `0` makes `cycles >= growth*0` always true and
    latches the trigger on permanently — so the trigger **ignores zero-count reports** for both the
    baseline and the growth test, and stays disarmed until one positive count exists.
  - **`refresh.limit` lives on the driver, not the trigger.** That keeps the trigger a **pure function of
    one segment's history** — which is what lets `warmup`/`patience` re-apply correctly after each
    refresh, lets it be unit-tested on synthetic histories with no solve, and (the big one) lets it be
    **calibrated offline**: log one march with no refresh trigger and an `on_step` observer, then
    replay candidate parameters against the log. No numeric default here is calibrated — they are chosen
    conservative (late rather than early) and must be set from an instrumented full-mesh run.
  - **Observation does NOT require a refresh (binding — this was a real bug).** `solve_coupled` runs the
    observed pre-march when the caller wants a refresh **or** merely wants to watch
    (`observing = refreshing or on_step or on_checkpoint`). Gating it on the trigger alone makes an
    *instrumented reference march* — no refresh trigger plus an observer, which is exactly the run a
    trigger is calibrated against, and the longest-running one — produce **no output at all** and sit
    silent for hours. Consequence to keep in mind: an observed solve spends `max_steps` on the pre-march
    and `max_steps` again on the finishing solve, so the budget is larger but *split*; instrumenting a
    solve already near its limit can turn a pass into a convergence-guard raise. Pinned by
    `test_the_march_reports_progress_without_a_refresh.trigger`.
  - **`checkpoint` is a SECOND seam, separate from `observer` (binding).** `checkpoint(report, state)`
    carries the state; `observer(report)` carries only numbers. Keeping the state off the report history
    is what keeps a `RefreshTrigger` a pure function that can be replayed offline against a logged march
    — put the state on that seam and a trigger could read the physics, and trigger calibration would cost
    one full solve per candidate instead of one logged run for all of them.
  - **Reporting seam.** `StepReport(step, cycles, residual_norm, residual_ratio, alpha, drift,
    inner_iterations, shift, escalations, diverged_retry)` + `MarchResult`,
    plus an optional streaming `observer` (a long march must not withhold all logging until it finishes).
    The trigger and a future logger consume the identical objects, so there is no second reporting path.
    Per-step observation exists only where the march is eager — the traced `_forward` would need
    `jax.debug.callback`, a separate decision; do not promise per-step reporting on the differentiable path.
  - **`shift` / `escalations` / `diverged_retry` on the report, and `MarchLogger` (`solve/march_log.py`)
    — the reporting half of the `on_step` seam (BUILT).** Every driver used to write its own
    `on_step`/`on_checkpoint` formatter, so the copies drifted and a gap fixed in one persisted in the
    others; `MarchLogger` owns everything derivable from a `StepReport` and takes an injected
    `metrics: state -> {name: value}` for the case quantities the solver cannot know (a reattachment
    length — `compare.reattachment_metrics` is the bfs3d one). `note()` writes an arbitrary line to the
    log's **own** stream (a driver reaching for `print` alongside it splits the run across two
    destinations, and a file log then loses whichever lines went to stdout); `phase(label, total)`
    auto-numbers, so a caller keeps no counter.
    Three report fields exist because the log could not otherwise carry them. **`shift`** is the `beta`
    the step was taken at — already read at the construction site for `carry_beta`, and otherwise
    recovered by every driver wrapping the step control. **`escalations` / `diverged_retry`** record
    whether the step was redone: `cycles` counts only the **accepted** attempt, so a redone step is
    indistinguishable from a cheap one, **and a retry mechanism left unconfigured never announces its
    absence** — which is not hypothetical (a bfs3d march ran with `cycle_budget` set but
    the cost threshold at its `None` default, so the beta-escalation never fired and nothing in the log
    said so; the cap exists to trip that trigger).
    The logger also reports the **reference norm and the stopping target**: the test is
    `‖R‖ <= atol + rtol·‖R₀‖`, and the march reports `‖R‖` and `‖R‖/‖R₀‖` but not `‖R₀‖`, so the target
    was invisible. Pinned by `tests/unit/test_march_log.py` (synthetic reports, no solve).
  - **Solve cost is reported offset-corrected, split by inner iteration (BUILT).** The raw count is
    lineax's `num_steps`, which carries **+2 per solve**, so a two-inner step reporting `cycles = 6` did
    two ideal single-cycle solves — the raw number is *entirely* offset and overstates the work
    threefold, worst exactly on the cheap near-root steps where the march's economics are decided.
    `restart_cycles(raw, solves)` in `solve/linear.py` is the one place that offset is stripped;
    `StepReport.restart_cycles` and `MarchLogger` both call it (it was previously open-coded as
    `cycles - 2*inner_iterations` on the report, and the logger printed the raw count instead).
    The step line reports `in` (inner count), `cyc` (corrected total) and `c/in` together, because one
    summed number conflates *how many* solves the step needed with how hard each was.
  - **`MarchLogger.on_inner` + `TextTable` (`aquaflux/text_table.py`) — the per-inner table (BUILT).**
    `DualTimeStep.inner_observer` already emitted `(index, ‖G‖ before, ‖G‖ after, cycles, alpha)` per
    inner Newton iteration via `jax.debug.callback`; nothing formatted it. `on_inner` matches that
    signature (`inner_observer=logger.on_inner`) and renders each iteration as a row — its own solve
    cost and the inner contraction `rate = ‖G‖out/‖G‖in` — inside a ruled table opened at `index == 0`
    and closed by the step line.
    Two ordering consequences, both deliberate: the **summary is a footer, not a header**, because the
    step's outcome is not known until it returns and buffering the block to lead with it would cost the
    live progress the rows exist to give (the title carries what *is* known — the step number, the
    elapsed time, and the residual the step inherits); and a **redone step opens a fresh block per
    attempt**, so the extra blocks are the record of what a retry cost, while the step line still
    reports only the accepted attempt.
    `TextTable`/`Column` is a **root leaf** (no package imports, like `vectors.py`) so any subsystem can
    format a report with it. It is built for streaming — rule, headings, `row` and `spanning` are
    separate methods each returning one line — which is what lets a table appear in a log being tailed.
    `rule(title=None, *, fill="-", segmented=True)` draws all three kinds the framed step block needs:
    the light **segmented** rule that divides one grid, the heavy `fill="="` one that **brackets** a
    block containing several grids, and the `segmented=False` span for the boundary between two grids
    that share a width but not a column layout — where a segmented rule would appear to belong to
    whichever grid its ticks happened to line up with. A step therefore renders as **one framed block**:
    step row, then the per-equation grid, then the asides **one concern per line** (preconditioner /
    case metrics / `limit` / `cum`). Run together on a single line those three had to be read in full to
    find any one of them.
    An over-wide value **widens its row rather than being truncated**: a cut-off number is a wrong
    number. Pinned by `tests/unit/test_text_table.py`.
  - **`StepOutcome` — the forward step's return is a record, not a tuple (BUILT).** It grew to six
    values (`phi, cycles, alpha, inner_iterations, reached_target, max_inner_cycles, binding_limit`),
    which is the missing-object smell: a positional tuple is where a consumer silently mis-unpacks one
    field for another, and every growth broke all five test doubles separately. Three of the fields are
    there because a march could not otherwise act correctly:
    - **`reached_target`** — did the step run to its OWN stopping criterion, or was it cut short? A
      cost-only escalation cannot tell an expensive success from a grind and **discards the success**:
      measured, an inner loop that reached `‖G‖ = 3.0e-6` against a `1.0e-5` target was thrown away for
      costing 54 raw cycles, wasting the work *and* replacing it with a shorter step. `forward_march`
      now fires only when `cycles > retry.abort_above_cycles` **and not** `reached_target`.
    - **`max_inner_cycles`** — the offset-corrected cost of the step's most expensive SINGLE solve, and
      what `retry.abort_above_cycles` triggers on. A **summed** threshold is not a difficulty signal: it
      grows with how many times the step solved, so at a threshold of 40 a 5-inner step trips at 6
      corrected cycles per solve and a 1-inner step at 38 — a 6× difference in sensitivity decided by a
      count that says nothing about conditioning. (Measured impact on one 63-step march: the two
      triggers agree on 62 steps and disagree on one — the expensive step, which per-inner catches and
      summed misses. The change is correctness, not speed.) **`cycle_budget` stays SUMMED** and should:
      it is a cost cap, and total cost is exactly what it caps.
    - **`binding_limit`** — the step cap where it was the *binding* constraint, else 1. A small `alpha`
      has two opposite causes (the direction overshot; a constraint stopped it being followed further),
      so `alpha` alone cannot be acted on or reported honestly.
  - **Positivity is NOT carried by the shift and the divergence guard (binding — a stated invariant that
    was wrong).** The direct-scalar path documented positivity as following from the pseudo-transient
    shift plus the guard. It does not: the guard fires on a *non-finite residual*, and by the time
    `sqrt` of a negative value has produced one, the state is already poisoned. Measured on `bfs3d`: a
    march ran 62 healthy steps and died because **two cells out of 23040** took `k` to `-3.3e-4` — every
    field still finite, all of `u, p, k, omega` moving by ~1e-4, only the derived `nu_t` NaN, because
    the SST closure takes `sqrt(k)`. The line search had crawled along that boundary for four inner
    iterations (`alpha` halving `0.008 → 0.001`, residual falling under 1% each) with no way to see it.
    - **`positive_block_limit(start, stop, tau)` COMPUTES the admissible fraction, it does not search
      for one** — the fraction-to-the-boundary rule, `alpha_max = tau·min(phi_i / −delta_i)` over
      decreasing entries. Rejecting violating rungs is **not** sufficient: `alpha` was already at the
      ladder's shortest rung when the field crossed zero, so there was nothing shorter to fall back to.
      `backtracking_line_search(max_alpha=…)` caps every rung **including the growth rungs**, and its
      default is `inf`, **not 1** — a default of 1 silently disables the growth rungs, a regression the
      growth tests caught.
    - `turbulence.positive_k_limit(coupled)` returns the limiter for a directly-solved `k` and `None`
      for a log-solved one (positive by construction there, so a cap would only throttle);
      `coupled_amg_continuation` wires it automatically.
    - **BOTH strategies carry `step_limit` / `step_projection` (fixed 2026-08-15).** They were
      `DualTimeStep` fields only, so choosing `PseudoTransientStep` **silently gave up the guard** —
      and the guard exists because its absence is a recorded march death (two cells of 23040 took `k`
      negative and NaN'd the whole residual through a bare `sqrt`, every field still finite, nothing in
      the ordinary stopping tests able to see it). The guard protects the **state**, not the march: a
      field that must stay positive must stay positive whichever strategy steps it. `PseudoTransientStep`
      now applies the identical pair in the identical order — project per entry first, then read the cap,
      which then finds nothing binding — so the two compose the same way on both. Both default `None`,
      which is the unconstrained step exactly, so the default path is unchanged.
    - **The cap is GLOBAL, so one entry near zero throttles the whole step** — a real risk on a field
      spanning `1e-5` to `4.5`. Measured, it does not bite, because the escape is not the cap: the cap
      forces `alpha → 0`, `CflResidualDualTimeControl` reads that as "shift too weak" and escalates
      `beta` `0.5 → 1.0 → 2.0`, and at the larger shift the implicit step is short enough to fit inside
      the constraint — full `alpha = 1`, residual descending. **The two mechanisms compose without
      having been designed together**, and the march then converged in 20 steps where two previous runs
      had died. Do not "fix" the throttling without re-measuring; the globalization already handles it.
    - Why not a log variable for `k`: `k = 0` is the physical no-slip wall condition, so `log k` is
      singular exactly where the mesh is finest (recorded and measured in `.claude/rules/turbulence.md`
      — the full-log form descends then freezes). `log(k+1)` is regular there but bounds `k > −1`, not
      `k > 0`, so it does **not** prevent this failure. `k = w²` would give both properties at the cost
      of a vanishing Jacobian scale at the wall.
  - **`StateCheckpointer` (`solve/checkpoint.py`) — rolling state checkpoints (BUILT).** A march that
    raises at its last step loses everything: the exception propagates out, so a driver saving only on a
    successful return has nothing for the hours before. It cost one `bfs3d` run its two converged rungs.
    Plugs into the same `on_checkpoint` seam, `every` steps, keeping `keep` files. Two properties that
    matter for a job that may be killed: it **writes to a staging name and renames** (a kill mid-write
    would otherwise leave a truncated file that still reads as a checkpoint — and with a small `keep`,
    the only one left), and **retention deletes only paths it wrote**, tracked in a deque rather than by
    globbing, so it cannot remove another run's state. The default serializer hands `numpy` a *file
    object*, because `np.savez` appends `.npz` to any path lacking it and would otherwise silently write
    somewhere other than asked. `combine_observers` fans the single `on_checkpoint` out to logger +
    checkpointer so a driver does not hand-roll a lambda one of them can be dropped from.
    **Known defect, not yet fixed:** the checkpointer writes whatever the march reports *including the
    failed step*, so the newest file can be the poisoned state — and a driver calling it "last good
    state" is then lying. Skip a non-finite report, or do not claim "good".
  - **`on_retry(reason, attempt, beta)` — say WHY a step is being redone (BUILT).** `forward_march`
  calls it immediately before a redo with `"diverged"`, `"cycles"` or `"alpha"` — the three
  `RetryPolicy.retry_reason` returns (⚠️ only `"diverged"`/`"alpha"` escalate β since 2026-08-17;
  `"cycles"` redoes at the same shift) — or `"solver"` for the tight-Krylov
  divergence retry. Without it a log shows the same step's work two or three times with nothing between
  the blocks, and the four reasons call for completely different responses. `MarchLogger.on_retry`
  writes the explanation between the abandoned attempt's block and the retry's, and numbers the attempt.
    - **`beta` is the shift the RETRIED attempt will run at — escalated on the three escalation
      reasons, unchanged on `"solver"` (binding, fixed 2026-08-14).** The call therefore sits *after*
      `escalated = beta * retry.beta_factor` is formed and before it is written onto the step. It used
      to fire before the escalation and hand over the abandoned attempt's β, which left `MarchLogger`
      reconstructing the real one as **`beta * 2`** — `retry.beta_factor`'s *default*, hardcoded — so a
      march configured with any other factor logged a shift it never ran at, and the `"solver"` path,
      where β is not touched at all, logged an escalation that never happened. **The trap generalizes:
      a callback that reports a value the caller is about to change makes its consumer re-derive the
      change, and a consumer re-deriving a caller's arithmetic will encode a default as a constant.**
      Pinned by `test_on_retry_reports_the_beta_the_retried_attempt_will_run_at` (at a *non-default*
      factor, since 2 cannot catch this) and two `test_march_log.py` tests.
  - **A self-rescaling measure means two "same" residuals are NOT the same number (binding trap).**
    `forward_march(norm_builder=…)` re-derives the `RowScaledNorm` at the state each outer iteration
    *begins from* and holds it for that whole iteration. So the `R` reported at the end of step N and
    the `‖G‖` entering step N+1 measure the **identical state** in **different scales**. Measured over
    one 62-step `bfs3d` march: they differed on **every** step — up to 2× early on, converging to 1 as
    the state settled and the scales stopped moving (the convergence-to-1 is the signature that
    identifies rescaling rather than a state difference; a rung boundary shows a ~2.8e5 jump, since the
    Reynolds number changed too). Nothing is wrong here, but **never compare two residuals across an
    outer-iteration boundary** and never print them adjacent — the march log deliberately dropped a
    "from |R|=…" field from its inner-block title for exactly this reason.
  - **The step grid stays NARROW; only scan-down quantities get columns (binding).** The step table is
    `step, t(s), beta, in, cyc, R, a_min, flg` — fixed, ~61 characters, comparable to the nested inner
    table so the two read as one document. Everything else — the case metrics, the preconditioner
    branch, the cumulative cycles, the per-field changes — rides in **spanning rows beneath the row it
    belongs to**. The first version put them all in columns and reached 112 characters with
    case-dependent column names; at that width, with the heading several rows up, it stopped being a
    table and became a line of numbers that could not be paired with their labels. **Do not widen the
    grid to add a quantity** — add an aside line, or a column only if it is worth scanning down the
    whole run. A row that follows an inner block is re-headed **compactly** (the label row alone, no
    rules): labelling one row must not cost three lines.
  - **The step summary is a table row, and the stopping test is stated once (BUILT).** The step line had
    grown to ~20 free-text `key=value` fields and ~150 characters — unreadable in a tail, and half of it
    was `‖R₀‖` and the target repeated on every row when both are **constants within a rung**. They now
    ride in a banner emitted once (and again whenever a continuation rung re-bases `‖R₀‖`), and the
    per-step values are a `TextTable` row so a quantity can be scanned *down* the run. Headings re-emit
    every `HEADINGS_EVERY` rows (fewer when the inner table is on, since its blocks push the last
    headings further up-screen), and the inner block is **indented** under the step row it belongs to.
  - **No column heading may contain `|` (binding).** `R`, not `‖R‖`; `G in`, not `‖G‖ in`. A heading
    carrying the grid delimiter makes the log unparseable by any column-splitting tool — not
    hypothetical: it forced regex workarounds in the analysis scripts written against the first version
    of this format. Say what the quantity is in the banner or the docstring, not in a heading that
    breaks the grid.
  - **`a_min` (step) vs `alpha` (inner) — different quantities, deliberately named apart.** An inner
    iteration's `alpha` is its own backtracking factor, and it reads **1.0 even when the line search
    failed to descend** (the non-descent fallback returns the longest finite trial step). The step
    reports the **minimum over its iterations with any non-descending one folded in as 0**. So
    `a_min=0.000` beside `alpha=1.000` is a real state — a full step that did not reduce `‖G‖` — not a
    contradiction. In the inner table **`rate ≥ 1` identifies those iterations exactly**, since the
    search is monotone.
  - **Diagnostics are opt-in through one `detail` argument (BUILT).** `MarchLogger(detail=…)` selects
    from `{"inner", "fields", "residuals", "pc"}`; empty (the default) is the plain one-row-per-step log. They are
    debugging/profiling instruments — several lines per step — not something a routine run should pay
    for. **Every hook stays safe to wire regardless** and no-ops when its name is absent, so a driver
    connects the instrumentation once and switches verbosity with one argument rather than by rewiring
    (which is how a driver ends up hand-rolling conditional plumbing). An **unknown name raises**: a
    silently-ignored typo means losing a diagnostic you believed was enabled.
    - `"fields"` — `MarchLogger(fields=…)` takes a `state -> named arrays` extractor
      (`turbulence.coupled_fields`) and reports each field's relative change per step via
      `field_change_metrics`. **The residual says the equations are unsatisfied; this says whether the
      *solution* still moves** — which a scalar case metric cannot: a mesh-quantized quantity like a
      reattachment length can sit perfectly still while the fields behind it drift several per cent, so
      "the metric stopped moving" is partly a statement about the instrument's resolution. The logger
      builds the (stateful) change measure itself so `detail` alone decides whether it runs — it must be
      called once per step in order, and a caller passing it directly would invite calling it from a
      probe and corrupting the sequence. `field_change_metrics` keys its output by the **field's own
      name** (not a decorated `d<name>/<name>`), so it joins against the per-equation residual below;
      how a quantity is *labelled* is the report's business, not the measure's. `coupled_fields` returns
      pressure **gauge-free** (mean removed; incompressible `p` is defined up to a constant unless a
      boundary pins it), **splits velocity per component** (`u`/`v`/`w`, so each lines up with its own
      momentum equation — a single vector entry averages them and hides a component that has stopped
      moving), and includes `ν_t`, which is derived rather than solved but is what the momentum
      equations actually see.
    - `"residuals"` — `MarchLogger(residuals=…)` takes a `state -> {equation: residual}` extractor
      (`turbulence.coupled_residuals`) and reports **the per-equation residual and its step-on-step
      contraction `rate`** beside each field's relative change, in one grid under the step row. **The
      scalar residual says the solve stopped improving; only this says which equation stopped it.**
      The numbers are `RowScaledNorm.per_block` — the very per-block values the march's scalar measure
      is the Euclidean combination of (`__call__` is now literally `norm(per_block(r))`, one formula),
      so a row near the total *owns* the residual. Names come from `coupled_equation_names(dim)` —
      `(u, v, w, p, k, omega)`, the flat layout's block order — the single home shared with
      `coupled_fields`, so the two grids join. Costs **one extra residual evaluation per logged step**,
      which is why it is opt-in. **The rows add up to the `R` printed above them, exactly** — pinned at
      `rel=1e-12`. That requires equilibrating at the **previous** state, not the logged one:
      `forward_march` re-derives the measure at the state each outer iteration *starts* from and holds
      it for the whole iteration (every trial step, the acceptance test, and the reported norm), so a
      step's residual is `norm_at_start(R(state_at_end))`. Scaling at the end state measures the right
      residual vector in the *wrong* scales and the rows stop adding up. `coupled_residuals` is
      therefore **stateful and order-dependent** — the same once-per-step-in-order contract
      `field_change_metrics` already carries — and takes a `reference_state` seed for the first step,
      which a per-rung reporter must pass (its rung's `seed_state`) or that step alone is scaled at its
      own end state. `ν_t` has no equation, so it gets a row with `--` in the residual cells rather than
      an invented number.
    - `"pc"` — `amg_beta_tracking_refresh(observer=…)` reports which branch a refresh took (`full` /
      `shift` / `none`) and its wall time; `MarchLogger.on_refresh` renders it on the step it preceded.
      **This closes a real gap:** which branch ran was previously invisible, so preconditioner behaviour
      had to be inferred from wall-clock — and was, wrongly, with an occasional expensive re-materialize
      read as a fixed per-step overhead.
    Pinned by `tests/unit/test_march_log.py`, whose assertions read **cells** rather than substrings, so
    a column reordering is not a false failure while a wrong value still is.

  - **`precondition_step` — per-step refresh of the step's frozen host preconditioner (binding,
    forward-only).** `forward_march(precondition_step=…)` calls `precondition_step(active_step, state)`
    before each `_march_step`, *after* the control has set β on `active_step`, to re-derive the step's
    **static** host preconditioner from the current `(state, β)`. It runs in the eager loop (a host op
    outside the jitted step) and mutates the preconditioner in place, so `_march_step` stays a
    compilation-cache hit. Two consumers (`.claude/rules/turbulence.md`), sharing one
    `_beta_tracking_refresh` skeleton: `lu_beta_tracking_refresh` re-factors the complete LU at the current
    `(state, β)` **every step** (cheap + exact → 1 Krylov iter), the fix for the frozen-LU β-mismatch above;
    `amg_beta_tracking_refresh` re-materializes the V-cycle **gated** (β-move OR staleness cap) instead,
    because rebuilding it is far more expensive and only an approximate preconditioner to begin with — the
    β-move trigger is what averts the α=0 /
    no-drift stall a *drift* trigger would hit on an overshoot. Distinct from the trigger's `refresh.builder`
    (which fires occasionally, restarts a *segment*, and returns a *new* step): this fires every step (the
    consumer may itself no-op) and mutates in place. Forward-only (impure), folded into the same `observing`
    gate and `jax.grad` guard as the trigger/control; `None` is byte-identical to before.
  - **The forward-step CONTRACTS live in `solve/forward_step.py`, not in whichever module needed them
    first (binding, 2026-08-15).** `ForwardStep`, `ShiftedForwardStep`, `StepOutcome`, `StepReport`,
    `StepControl`, the `StepFn` callable alias and `within_tolerance` are what travel between the Newton
    driver, the eager march, the globalization strategies, the step controls and the retry policy. None
    belongs to any one of them, and every one of them was living wherever it was first written.
    - **The placement had a cost that came due twice.** `StepControl` was declared in `march.py` with
      **zero** implementations there, so `step_control.py` had to import `march` — which forbade the
      reverse, so a defaulting rule about two `solve/` objects could not be written in `solve/` at all
      and ended up in `turbulence/coupled.py`, a package away. And `implicit.py`, named for the Newton
      solver, was a de-facto contract module handing `_ForwardStep`, `_within_tolerance` and
      `backtracking_line_search` across boundaries either privately or absent from `__all__` — which
      under this package's own boundary rule read as violations.
    - **`forward_step.py` is a LEAF and must stay one.** It imports `linear`, `norm` and `relaxation`,
      none of which import it back; `implicit`, `march`, `continuation`, `retry`, `step_control`,
      `march_log` and `checkpoint` all depend on it. Adding an import here that points at any of those
      re-creates exactly the cycle it exists to remove.
    - **What did NOT move, and why.** `backtracking_line_search` stays in `implicit.py`: it is
      behaviour, not a contract, and moving it would make the contract module carry a line search.
      `MarchResult` stays in `march.py` — only `forward_march` produces it. The concrete strategies,
      triggers and controls stay with their own modules; a contract module holds contracts.
    - The rule the cycle was blocking is now `solve/step_control.py`'s `default_dual_time_control`,
      beside the controls it chooses between, and exported.
  - **`StepControl` — stateful, feedback-driven step reshaping on the eager march only (binding — the
    twin of `RelaxationSchedule`, deliberately NOT one interface).** A `RelaxationSchedule` is memoryless
    and lives on the differentiable step; a `StepControl` reads the *previous* `StepReport` (α, cost) —
    feedback available only after a step — to reshape the next step, and may raise under `jax.grad`, so it
    lives here beside `RefreshTrigger`, never on the traced path. `forward_march(step_control=…)` calls
    `next_step(base, previous, state) -> (ForwardStep, new_state)`, threading the control's own state; the
    march stays β-ignorant (the control returns a ready-to-run step, typically `base` with a
    `ConstantRelaxation` β leaf via `tree_at`, so `_march_step` stays a cache hit). **The control state
    survives across preconditioner refreshes (issue #156):** `forward_march` takes an incoming
    `control_state` and returns the final one on `MarchResult`, and `solve_coupled` threads it from each
    segment into the next — so a control that climbs β over many steps continues past a refresh instead of
    resetting to `beta_start` at every segment (the α-controller and the refresh were co-designed but
    could not compose before this). This is the *global*-lifetime carry, deliberately opposite the
    segment-local SER `residual_norm_0` and `drift_measure`. Unifying the two into
    one `(rn, rn0, α, state) -> (β, state)` interface was rejected: it would union SER's needs with the
    control's (dead α/state args for SER), drag α onto the differentiable core where the line search
    cannot even produce it before the step, and risk the byte-identity of the default path. Four concrete
    `StepControl`s live in `solve/step_control.py`, all three sharing one body: **`DualTimeControl`** (the
    Courant β-ramp, the **default** for a dual-time observed march — carries β across refreshes, see the
    DualTimeStep bullet above), **`ResidualRatioDualTimeControl`** (the opt-in residual-keyed
    alternative), and **`CflResidualDualTimeControl`** (see the bullet below). There is **no
    `AlphaTargetingControl`** — deleted 2026-08-14, see the "SER β schedule runs backwards" bullet.
    - **`ShiftStrengthControl` is the shared base, and a subclass supplies ONLY `_adapt` (binding).** It
      owns the first-step seed, the hold-across-a-refresh rule, the `[beta_min, beta_max]` clamp, the
      `tree_at` swap of a `ConstantRelaxation` onto a **dynamic** β leaf (the compilation-cache-hit
      property), and `carry_beta`. The carried state is **`(beta, memo)`** for every control — β is
      universal and `memo` is whatever the rule remembers (`None` for the memoryless Courant rule, the
      previous residual for the two ratio rules). **Why it exists:** written three times, the bookkeeping
      drifted — `carry_beta` was byte-identical in two controls and *absent* from the third, which
      `forward_march` probes for with `hasattr`, so that control silently dropped its escalation
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
    `(β, previous residual)` — the shared `(beta, memo)` carry — across refreshes like
    `ResidualRatioDualTimeControl`; opt-in via
    `solve_coupled(step_control=…)`. Unit-tested in `test_step_control.py` (grows on α at a flat residual,
    brakes on a rising residual at α = 1, brakes on an inner clip, holds in the band, carries β). The ratio
    thresholds are march-calibrated numbers — set them from a logged march, not intuition (the 3D
    development overshoot shows ratios ~1.14).

