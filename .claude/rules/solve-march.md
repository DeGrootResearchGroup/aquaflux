---
paths:
  - "aquaflux/solve/march.py"
  - "aquaflux/solve/march_log.py"
  - "aquaflux/solve/march_history.py"
  - "aquaflux/solve/checkpoint.py"
  - "aquaflux/solve/step_control.py"
  - "aquaflux/solve/ramp_schedule.py"
---

# Rules — `aquaflux/solve/` the observed march (`newton_march`, triggers, controls, logging)

> Split out of `solve.md`, which holds the package-wide contracts this file assumes. Dated
> measurements and investigation steps are in `.claude/notes/solve-march-log.md` (never auto-loads);
> a new measurement goes there.
>
> **Deleted names (#371, 2026-09-14):** there is no `coupled_continuation`, `coupled_lu_continuation`,
> `coupled_amg_continuation`, `lu_beta_tracking_refresh` or `amg_beta_tracking_refresh`. The coupled
> march is one builder, `coupled_step`, given a preconditioner value (`BlockDiagonal` or
> `MaterializedJacobian(FieldSplit | BlockInverse)`); a march that keeps its preconditioner current runs
> on a session (`open_session`). Rename table: `.claude/rules/turbulence.md`.

## `StepHistory` — the per-step record for a program to read

`march_history.py::StepHistory(path, clock=)` is a host-side observer: `on_checkpoint(report, state)`
writes a row; `on_refresh(timing)`, `on_retry(reason, attempt, beta)` and `on_residuals(terms)`, called
while a step is under way, fill in the row it then writes and are reset after it.
- **What a recorder offers is declared by `MarchRecorder`** (same module, exported, `runtime_checkable`,
  #669): `on_checkpoint`, `on_retry`, `on_refresh`, and an `on_residuals` that may be `None` (the march
  then skips its cost). The case runner fans hooks out by `isinstance(recorder, MarchRecorder)`; a
  recorder offering only some hooks gets none of the extras. A `StateCheckpointer` is not one. The one
  attribute probe left, `hasattr(value, "item")` in `_cell`, recognizes NumPy/JAX scalars without
  importing either.
- Columns: `step` (1-based across continuation segments), `seconds`, every `StepReport` field under its
  own name with the report's `step` renamed `segment_step` (it restarts per segment), `restart_cycles`,
  `refits` / `refit_seconds` (refreshes of any kind but `none`), `retry_reasons` (`;`-joined), and one
  `residual_of_<equation>` per equation the march's measure names. `COLUMNS` is derived from
  `StepReport._fields`, so a new report field reaches the file with no edit here.
- Floats by `repr`, booleans 0/1, NumPy/JAX scalars unwrapped with `.item()` first — `np.float64` is a
  `float` subclass whose NumPy-2 repr is `np.float64(0.1)`.
- ⚠️ **The header is written with the first row**, since the equation columns are known only once the
  march reports them; until then the file exists and is empty (`aquaflux_ui.ConvergenceHistory.read`
  reads that as zero steps). A step naming other equations than the header raises `ValueError`.
- ⚠️ **The prefix is `residual_of_`, not `residual_`**, which would also match `residual_norm` /
  `residual_ratio`.
- It is the machine-readable sibling of `MarchLogger`, not a replacement: the case runner writes both
  (`Outputs.log`, `Outputs.history`, default `history.csv`). Pinned by `tests/unit/test_march_history.py`.

## `StepReport.station` / `.arrived` and `newton_march(on_residuals=)`

- **`station`** (the homotopy's station index, `0` without one; only equality between adjacent steps
  means anything) and **`arrived`** (whether the step drove the target problem; `True` without a
  homotopy) let a record mark where a continuation ended.
- **`on_residuals({equation: term})`** is called after each step, before `observer` and `checkpoint`,
  with each named block's term of the measure the step was judged in (`active_step.norm()`) at the
  station residual it drove — the terms whose Euclidean combination is `residual_norm` (pinned to
  `rel=1e-12`). From `solve.norm.named_blocks(measure, residual_fn, state)` (one compiled call per block
  structure, `_block_terms`). Skipped unless the measure is a `NamedBlockMeasure` (`norm.py`,
  `runtime_checkable`: `names` + `per_block`) with named blocks. Only the two row-scaled builders name
  them: `coupled_scaled_norm` (`turbulence/measures.py`, `coupled_equation_names`) and
  `FlowMeasures.row_scaled` (`flow.flow_equation_names`); the block-scaled coupled measure stays unnamed.
  One residual evaluation per step; nothing when `None`. Threaded through `staged_march`,
  `solve_coupled`, `solve_flow_march`; `solve_reynolds_ramp` reaches it through `**solve_kwargs`.
- **Open question:** this overlaps `turbulence.coupled_residuals` + `MarchLogger(detail={"residuals"})`,
  which re-derives the same per-block numbers from outside the march (stateful, coupled RANS only; its
  one caller is `validation/bfs3d_openfoam/compare.py`). Whether to feed the logger from the hook and
  delete `coupled_residuals` is undecided.

## `newton_march(adaptive_tolerance=)` and `StepReport.linear_tolerance` (default since 2026-10-15, #645)

`AdaptiveLinearTolerance` (`DEFAULT_ADAPTIVE_TOLERANCE`; `None` switches it off) varies each step's
linear-solve relative tolerance; the rule and why it is not Eisenstat–Walker are in `solve.md`'s
contract entry. March side: the solver's base tolerance becomes a float64 array leaf once; each step (and
redo) runs `with_relative_tolerance(solver, tolerance)`; the next tolerance is chosen from the step's
`alpha`, its reported residual over the previous report's (deliberately across the measure rebuild) and
the stopping test's norm and target. `linear_tolerance` is the tolerance the step ran at (`nan` without
the setting), so replaying the rule over the reports reproduces it (pinned). `tight_solver` is untouched.

## The observed march — newton_march, triggers, controls, logging

### `ResidualHomotopy` — a continuation walked inside one march

A `Protocol` on `newton_march(homotopy=…)`: `enter(step, previous)` makes the station that step solves
current and returns its residual; `arrived(step)` says whether that station is the target.
- **`enter` is handed the previous step's report**, so a homotopy may decide where a step runs from how
  the march went. The first step is entered twice with `None` (damping anchor, then the step). The
  march's look-backs (did the previous step enter a station? the `rebase` test) read
  `reports[-1].station` / `reports[-2].station`, so an implementation need not remember history.
- ⚠️ **The stopping test is gated on arrival** — the load-bearing correctness property. A small residual
  at an intermediate station says nothing about the target, so the march may not stop and
  `MarchResult.converged` may not be `True` until `arrived`. Pinned by starting a march at a station's
  own exact root.
- **The damping anchor `‖R₀‖` is taken at the first station**; `residual_fn` stays the target and is what
  `reference_norm` defaults to, so the stopping bar is the target's throughout.
- **The row-scaled measure and the drift measure stay on the target assembler**: one yardstick down the
  log; the arrival gate makes it safe.
- ⚠️ Reachability is pinned explicitly (an unwired `homotopy` leaves the march bit-identical): a ramp whose
  stations have distinct roots, budget cut at arrival, must end on the last station's root.
- **Current status:** walking the continuation inside one march (`rebase` + `redamp`) beat pitzDaily's
  ladder of separate marches on steps, cycles and wall at the same `x_r/h` (log, "`ResidualHomotopy` —
  what it is for, and the four-arm result"). Per-step viscosity (`steps_per_station=1`) is the cheapest
  schedule measured on both flagship cases; set `steps_per_station` from a schedule measured on the case
  (log, "a station spans several steps on purpose" retracted).

**`RampSchedule` / `RampWalk` (`solve/ramp_schedule.py`)** — one frozen value (`stations`,
`steps_per_station`, `end_when: StepReport -> bool`, `finish`); `schedule.walk()` hands each march a
mutable `RampWalk` recording a `RampPosition(station, progress)` per entered step (`progress` 0 at the
anchor, 1 at the target; a homotopy maps it to `anchor ** (1 - progress)`). `ViscosityRampHomotopy`,
`solve_reynolds_ramp` and the case file's `ViscosityRamp.schedule(step_control)` all take it.
- **Early end:** the first report off the target for which `end_when` holds ends the walk; the rest is
  walked in `finish` equal steps of progress, the last the target. A report from the target never ends
  anything. Refused: `finish` without `end_when`; `end_when` with `steps_per_station != 1`.
- `end_when` is the march's own `step_control.settled` (`solve-globalization.md`), so the ramp ends on the
  evidence the floor release waits for. **Opt-in.** pitzDaily's file still ships 12 fixed stations (111
  cycles against 114 settle-ended); on bfs3d the settle-ended ramp also lost to the shipped fixed count
  (#678; log, "`RampSchedule` / `RampWalk`").
- The fixed walk numbers stations `0..stations`; finishing steps continue from the station the end fired
  at, so an ended walk's target id is not `stations`. Pinned by `tests/unit/test_ramp_schedule.py`.

**`redamp` — re-damp on entering a station; the march breaks without it.**
`ShiftStrengthControl.redamp(state, factor)` multiplies β by `ResidualHomotopy.shift_factor(step)` on the
step that first faces a new station, and `newton_march` holds the control for that step. It is
`rebase`'s companion, firing on the adjacent step (this on the entering step, `rebase` on the next, the
first whose ratio would straddle the boundary).
- Why: the ratio rule's spurious brake at station changes, which `rebase` removed, had been holding β out
  of a collapse regime (log, "`redamp` — the spurious brake `rebase` removed was load-bearing"; the
  general lesson heads `solve-globalization.md`).
- **The hold is load-bearing**: an entering step forgoes its `/grow`, so a change is worth
  `factor * grow`, which reproduces what the old brake did.
- **Default `redamping = 2.0`** (one `backoff`'s worth); it damps at every change and measured better
  than the accident it replaced (261 against 269 cycles on pitzDaily).
- ⚠️ **Two different floors.** `beta_min` (step control) floors the shift the **operator** is solved
  with; `refit_beta_floor` (the refresh) floors only the preconditioner's copy,
  `pc_beta = max(beta, refit_beta_floor)`. The collapse was an operator/preconditioner mismatch, not a
  shift the preconditioner could not invert — and the mismatch's size does not discriminate either (10x
  converged at the target where 4.2x was fatal mid-ramp). Both flagship case files dropped the refit floor
  on 2026-10-13.
- ⚠️ **The wall belongs to `(state, β)`, not to β — do not build a learned floor under β, or a cap on β's
  growth rate.** The same march converged at β = 0.005 once arrived that was fatal at step 13; both
  alternatives infer a property of the shift from evidence gathered at a state the march has left.
  `redamp` infers nothing: it damps exactly while the problem moves. (Detail: log, "`redamp` — the two
  floors, …".)
- `shift_factor` defaults to `1.0` on the protocol, so a homotopy indifferent to the shift is
  byte-identical.

**`station_step(step, station, arrived) -> step`** reshapes the step per station (the counterpart of what
a homotopy does to the problem); called once per outer step while a homotopy runs, before the step
control; `None` is byte-identical. It cannot live in the shift policy, which sees only progress
quantities (iterate, residual, β), and those come apart from the station index on a short ramp.
- ⚠️ **Return a step differing only in array leaf values** (`eqx.tree_at` over a fixed structure):
  `filter_jit` compares non-array leaves by value, so a swapped Python scalar recompiles the solve per
  station.
- Its first measured use (damp a closure block harder during the ramp) did not pay; kept as the only way
  to express a per-station setting (log, "`station_step`").

### `march.py` — the one march

- **`newton_march` is the only Newton loop in the package (binding, 2026-09-15).** `RootSolver.solve`
  marches with it on `stop_array_gradients` copies, owns the root guard, and hands the root to
  `root_adjoint`; there is no traced finishing solve. A Python loop cannot run under `jit`/`vmap`, so
  every entry point refuses them up front through `refuse_a_transform_the_march_cannot_run_in(pytree,
  caller=…)` (in `march.py`; called by `RootSolver.solve`, `solve_coupled`, `solve_flow_march`) — decided,
  since nothing `vmap`s a solve and `_march_step`'s `filter_jit` gives the compile-cache reuse.
- **The march is forward-only and carries no guard by design**: stopping short on a trigger is its
  purpose, so every caller reads `MarchResult.converged` before treating the state as a result. **The
  early stop is the driver's, not the solver's (binding)** — putting it behind the root guard would need
  an exemption returning a non-root without raising, the silent-wrong-gradient hole the guard closes. So
  `RootSolver.solve` passes no trigger; `solve_coupled` does and owns its own convergence test. Chunking
  a march with `max_steps=1` fails independently: it recomputes `residual_norm_0` per chunk, pinning the
  SER ramp at β₀.
- **Two reference norms; conflating them freezes the march (binding).** `residual_norm_0` is
  segment-local (recomputed at each `newton_march` entry, handed to `stepper()` for the SER ramp);
  `reference_norm` is global (fixed across segments; the convergence test and reported ratio).
  Substituting the second for the first pairs a refreshed shift diagonal with a stale small β (the
  over-damping freeze in `turbulence.md`).
- **Per-step jit cache hit is mandatory (top implementation risk).** The per-step call is the
  module-level `eqx.filter_jit`'d `_march_step`, taking the `NewtonStrategy` and the residual as
  arguments. Callers pass the same `strategy` across a segment (a rebuilt one is the intended one-off
  recompile) and a bound module method rather than a fresh `lambda` (hashed by identity). Pinned by a
  trace-counting test: assert further steps add no traces, not that the total is 1 (the residual is
  traced several times within one step).
- **Nothing in the refresh machinery reads the line-search α.** The post-step `RefreshTrigger`s read
  cycles/residual ratio (`CycleGrowthTrigger`) or `ν_t` drift (`CoefficientDriftTrigger`); the
  per-attempt `refresh_preconditioner` re-fits on first call, after `rebind`, and when the march passes
  `due=True`. So a collapsed line search can only escalate β, never buy a rebuild. An α-keyed rebuild
  would need no new trigger — a condition inside `refresh_preconditioner`, which runs once per attempt
  after the control sets β. Unbuilt and unmeasured (log, "nothing in the refresh machinery reads the
  line-search α").
- **`solve_coupled` declares no globalization setting, and must not be "fixed" to (binding).** They ride
  `**strategy_kwargs` (one `Globalization`) to the step builder. Pinned by
  `test_globalization_knobs_still_reach_the_continuation_builder`.

### Resuming a march — `Resumption`

`resume=Resumption(...)` on `staged_march`, `solve_flow_march`, `solve_coupled` and `RootSolver.solve`,
built on `newton_march(reference_norm=, damping_reference=)`. Without it a resumed march re-measures
`residual_norm_0` at the restart state, so the ratio is ~1 and β reopens at `beta0`.
- Carries `reference_residual` (the stopping scale `atol + rtol * reference`, and the first segment's
  damping anchor unless given), `damping_reference` (the anchor of the segment the march was in — differs
  once it has refreshed) and `shift`. The march still starts from the residual it measures at `phi0`.
  `StepReport.damping_reference` records each step's anchor and `report_record` writes it, so a checkpoint
  carries it.
- The shift goes through `ShiftStrengthControl.resumed_at(shift)` → `(clamp(shift), None)`, which
  `staged_march` calls on the control it ends up with; the first step holds it and the ramp continues;
  the memo is dropped, so a ratio-keyed control forms its first ratio a step later. A march with no
  control has no shift to seed.
- Refused beside a `homotopy`. Carried by a case's `Checkpoint` (`case.md`). ⚠️ **Not carried:** the
  refresh budget and retry escalation history beyond the final shift. Measurement: log, "Resuming a march".

### `RefreshPolicy` and `RetryPolicy` — the injected settings objects

- **`RefreshPolicy` (`solve/refresh.py`)** bundles `trigger` / `limit` / `builder` /
  `refresh_preconditioner`, meaningless apart. Predicates on it: `refreshes`, `segments` (`limit + 1`),
  `is_last_segment`, `require_rebuildable`. **It is driver-level; `newton_march` does not take it
  (binding)** — the march uses only `trigger` and `refresh_preconditioner`, and `limit` / `builder`
  govern the driver's segment loop. `NO_REFRESH` is the shared default. There is no
  `RefreshPolicy.observes`.
- **`RetryPolicy` (`solve/retry.py`)** bundles `solver` / `divergence_cap` / `on_alpha` /
  `abort_above_cycles` / `beta_factor` / `max_retries`, and carries the decisions over them as methods:
  `retry_reason`, `has_diverged`, `with_inner_abort`, and `escalate` (the one home of "scale the β leaf,
  never rebuild it from a float"). `newton_march` takes the whole policy (it uses all of it). `NO_RETRIES`
  is the shared default. **The order stays on `newton_march`** (escalation first, `solver` as fallback) —
  a property of the loop; `on_retry` stays a separate argument (reporting).
- `retry.solver` is a `LinearSolverSpec`, built once per march into `tight_solver`
  (`solve-globalization.md`).

### Retries — what redoes a step, and at what shift

`retry_reason` returns `"diverged" | "alpha" | "cycles"`; `ESCALATING_REASONS = {"diverged", "alpha"}`
(`retry.py`, read by the march and the log) says which raise β. The tight-Krylov fallback reports
`"solver"`. (Origin of the split: `solve-amg-multigrid.md`, "the cost guard and the shift escalation are
now separate responses".)
- **Escalation**: a step that diverges or whose line search collapses (`retry.on_alpha`) is redone from
  the same pre-step state at β × `retry.beta_factor`, up to `retry.max_retries`, before the tight
  `retry.solver` fallback. Escalation leads because, run divergence-first, the tight retry's grind on
  bfs3d was work the escalation then superseded (log, "why escalation leads"). The bailout is reactive by
  necessity: no cheap static operator property predicted a bad step on bfs3d (log; ledger).
- **Cost (`"cycles"`)**: `retry.abort_above_cycles` redoes an expensive step **at the same shift**, on a
  re-fitted preconditioner (the march passes `due=True` first). A cycle count says the frozen
  preconditioner is struggling, not that the step is stiff, so cost must not escalate. ⚠️ **One cost redo
  per step** (`redone_on_fresh` in `march.py`): a second at the same shift would repeat it exactly.
- **Divergence fallback (`retry.solver`)**: redoes a diverged step at the tighter solver, without
  re-refreshing the preconditioner (the failure is an under-converged Krylov solve, not a stale
  preconditioner). One retry. `divergence_cap` defaults to `inf` (non-finite only), because the residual
  legitimately rises during development. Measurement: log, "Reactive divergence retry".
- **Both cost and step-length reasons require the target unmet; divergence does not.** `reached_target`
  keeps an expensive success (one that met its own tolerance) from being discarded.
- **The thresholds are per-solve and calibrated from a logged march.** `abort_above_cycles` triggers on
  `max_inner_cycles` (the costliest single solve; a summed count would vary 6x in sensitivity with the
  inner count). `retry.on_alpha = 0.01` was set from bfs3d's own step table (no productive step below
  `a_min` 0.191; dead ones 0.000) — recalibrate on another case (log, "`retry.on_alpha`").
- ⚠️ **The escalation must keep `_march_step` a compile-cache hit (binding).** Escalate by scaling the
  existing leaf (`beta * retry.beta_factor`), never by rebuilding from a Python float — a weak-typed
  rebuild can change the aval and recompile the whole solve on every retry. Pinned by
  `test_a_forced_escalation_adds_no_march_step_compilations` (strong-typed β). The `retry.solver`
  fallback is a separate one-off compile.
- **The escalated β is carried into the control** via `step_control.carry_beta(state, β)` (one
  implementation on `ShiftStrengthControl`; the march checks `isinstance(control,
  ShiftCarryingControl)`), memo preserved. Without it the next step recomputes β from the control's own
  trajectory and re-pays the escalation every step. This is what lets `beta_min` go toward zero with the
  escalation as the safety net.
- **`on_retry(reason, attempt, beta)`** is called immediately before a redo; `beta` is the shift the
  **retried** attempt will run at (escalated, or unchanged for `"cycles"`/`"solver"`). ⚠️ Trap that
  transfers: a callback reporting a value the caller is about to change makes its consumer re-derive the
  change, and it will hard-code a default (the logger once printed `beta * 2`). Pinned at a non-default
  factor.
- **Inner-loop counterparts on `DualTimeStep`** (pushed down by `RetryPolicy.with_inner_abort` when the
  step is an `AbortsInnerLoop`, so there is one number, not two):
  - `abort_above_inner_cycles` stops the inner loop the moment one solve exceeds `abort_above_cycles`
    with the target unmet. `cond` tests the convergence target first, so it cannot bin an expensive
    success. ⚠️ It is not trajectory-neutral: `refresh_preconditioner` runs per attempt, so truncating
    an attempt changes the refresh sequence.
  - `abort_below_alpha` exits the inner loop at a step-length collapse.
  - `cycle_budget` caps the inner loop's **summed** Krylov count (`dual_time=DualTimeLoop(cycle_budget=…)`
    on every coupled builder). ⚠️ It is checked **between** inner iterations, so one stagnating solve can
    overshoot it by its own `stagnation_iters` / `max_restarts`. ⚠️ **It returns a partial iterate
    expecting it to be discarded: pair it with `retry.abort_above_cycles < cycle_budget`**, or the
    truncation is accepted. Measurements: log, "`DualTimeStep(cycle_budget=…)` and
    `abort_above_inner_cycles`".
- `None` for each threshold is byte-identical; all are forward-only. A retry mechanism left unconfigured
  never announces its absence — check the log's retry lines when a cap is set.

### Refresh — re-fitting the frozen preconditioner during a march

- **A cost re-fit ends the step; the march takes it before the next one (binding, 2026-10-10, #290).**
  `DualTimeStep(refresh_on_cycles=N)` stops its inner loop after the solve that reached `N` restart
  cycles, keeping that iterate, and reports `StepOutcome.refresh_due`; the march then calls
  `refresh_preconditioner(active_step, state, due=True)` before the next step. Why: a mid-step host
  callback synchronized every inner iteration, and an on-device preconditioner cannot be swapped
  mid-step at all. There is no `inner_refresh` hook.
  - **`due`** is keyword-only on every hook: `False` = offered (the hook gates its own cadence), `True` =
    asked (it must re-fit; the session reports kind `"cost"`). The march passes `due=True` after a
    `refresh_due` step, before a cost redo, and before an escalated redo of an attempt that itself asked.
  - **A `refresh_due` step is never a cost redo** (`retry_reason` skips `"cycles"` for it); divergence and
    step-length collapse still redo it.
  - **Refused, not ignored:** a `StopsForRefresh` strategy with `refresh_on_cycles` set and no
    `refresh_preconditioner` raises before the first step.
  - Every march recorded under `refresh_on_cycles` before 2026-10-10 predates this.
- **`refresh_preconditioner(active_step, state, due=…)`** runs in the eager loop before each
  `_march_step`, after the control has set β, and again before a redo. It returns `None` or an update
  `step -> step`:
  - `None` — nothing re-fitted, or a host preconditioner in a static field re-fitted in place
    (`MonolithicFactorShiftPolicy`) — a cache hit by construction.
  - an update — a preconditioner carried as data (`TracedSplitShiftPolicy`, `with_traced_split(step,
    split)`). The march applies it to **both** `strategy` and `active_step` (updating only the active
    step lasts one step) and reports the last on `MarchResult.strategy`, which `staged_march` carries to
    the next segment and the adjoint. Fixed shapes keep it a cache hit (`solve-field-split.md`).
  - **With `norm_builder` set, the march rebuilds the step's measure from the updated step** (#701): an
    update can carry a new shift policy (`RebuildingSession` at a `rebind`), and the row-scaled measure
    reads its scales from it. The redo path applies an update without rebuilding the measure (a redo
    keeps the attempt's measure).
  - Consumers: `MaterializedSession.refresh_preconditioner` over `BetaTrackingRefresh` (re-fits on first
    call, after `rebind`, and on `due`), and `RebuildingSession.refresh_preconditioner` (block-diagonal
    family, #386: an update only after a `rebind`, via `ShiftedStep.preconditioned_as`; refuses a step
    with `refresh_on_cycles`). Distinct from `refresh.builder`, which restarts a segment with a new step.
- **`CoefficientDriftTrigger` — the preferred staleness trigger (binding for new work).** A frozen
  preconditioner is stale when the operator moved, so measure that: `StepReport.drift` carries a scalar
  from `newton_march(drift_measure=…)`; coupled RANS uses `turbulence.eddy_viscosity_drift(coupled,
  reference_state)` = `‖Δν_t‖/‖ν_t,ref‖`. It needs no residual gate and no patience (drift does not
  respond to β and cannot fire before the flow develops, where a refresh was measured harmful).
  - **The scalar is the whole design (binding):** the state must not reach a trigger, so triggers stay
    pure functions replayable offline against one logged march. Do not widen the trigger interface to
    take the state.
  - **Re-base the measure at every refresh (binding):** `solve_coupled` builds a fresh measure per
    segment against that segment's starting state; one carried across segments would re-fire every step
    and burn `refresh.limit`. Pinned by `test_the_drift_measure_is_rebased_at_every_refresh`.
  - `threshold = 0.1` was calibrated on one cold-IC pitzDaily march (log); the first default, 0.5, was
    inert there. Set trigger numerics from a logged march, not judgement. End to end it cut per-step cost
    but not step productivity (log).
- **`CycleGrowthTrigger` — cost growth is the trigger, the residual ratio the gate.** Fires once
  `warmup` is past, `residual_ratio <= max_residual_ratio`, and the last `patience` steps each cost
  `>= growth ×` the segment's running-minimum non-zero count. The gate exists because the cycle count
  also rises with β → 0, more than with staleness on pitzDaily, and a pre-separation refresh was harmful
  (log, "`CycleGrowthTrigger`"). ⚠️ **Zero-count reports are ignored** (a rejected step or a direct
  solver reports `0`; a `0` baseline would latch the trigger on). Its defaults are uncalibrated.
- **`refresh.limit` lives on the driver**, keeping the trigger a pure function of one segment's history
  (so `warmup`/`patience` re-apply after a refresh, it unit-tests without a solve, and it calibrates
  offline).

### Observation seams

- **Observation changes nothing about a solve (binding, #369).** `solve_coupled` has one march;
  observers, refreshes, controls and retries all run in it on `stop_gradient` copies. Pinned by
  `test_observing_a_solve_changes_nothing_about_it`. Trap from the deleted eager pre-march: when
  observing selects a different path, an instrumented run is not the run under investigation.
- **`checkpoint(report, state)` is a separate seam from `observer(report)` (binding)** — keeping the
  state off the report history is what keeps a `RefreshTrigger` replayable offline.
- **`StepReport` + `MarchResult`** are the one reporting path; trigger and logger consume the same
  objects. `RootSolver` does not expose `observer`; `solve_coupled` does.
- ⚠️ **`MarchLogger.on_step` and `.on_checkpoint` are mutually exclusive renderings of one event — wire
  one.** `newton_march` calls both seams every step, so the same logger on both logs every step twice and
  double-counts its totals. The shipped driver wires `on_checkpoint` (it also has the state for case
  metrics). A planned single `MarchObserver` bundle is not built and needs the two renderings reconciled
  first, which changes the log format.
- **`StateCheckpointer` (`solve/checkpoint.py`)** — rolling state checkpoints on the `on_checkpoint` seam,
  `every` steps, keeping `keep` files. It writes to a staging name and renames (a kill mid-write cannot
  leave a truncated file), and retention deletes only paths it wrote (tracked, not globbed). The default
  serializer hands NumPy a file object (`np.savez` appends `.npz` to a path). `combine_observers` fans
  `on_checkpoint` out to several consumers. ⚠️ **Known defect, unfixed:** it writes the failed step too,
  so the newest file can be the poisoned state; do not call it "last good state" (a case's `initial:
  Checkpoint` refuses non-finite fields on read). `find_checkpoint(directory, step="latest", prefix=…)`
  finds the highest step by name (not mtime), ignoring `.partial`; `checkpoint_name` holds the
  `<prefix>-<step:05d>.npz` pattern and `report_record(report)` the list of a step's numbers, both shared
  with the case layer (`aquaflux/case/restart_file.py` injects its own `save=`).

### `stopping_step` — judge the stop in another step's measure (#701)

`newton_march(stopping_step=step -> step)` (through `staged_march`, `solve_coupled`): once arrived, the
stopping test (loop condition, the retry loop's converged check, `MarchResult.converged`) reads
`norm_builder(stopping_step(step), state)(R(state))`; every step is still steered, line-searched and
reported in its own measure, so a march with one is step-for-step identical to one without up to where it
stops. Tolerance `atol + rtol * stopping_reference` (the starting residual in the stopping measure).
Needs `norm_builder`; refused beside a `resume`.
- `StepReport.stopping_norm` / `stopping_ratio` carry what the stop compared, on every step (equal to the
  steered pair without a stopping step; `nan` on a hand-built report). `MarchLogger`'s `R` column and
  banner read them (falling back to the steered pair when `nan`); a step whose two norms differ gets a
  `steered R …` aside; the per-equation grid stays in the steered measure. `case/run.py` records
  `stopping_norm` as the run's residual. ⚠️ **`aquaflux_ui`'s Run section still plots and judges
  `residual_norm`** (`run_monitor.py`).
- Why: a materialized viscosity ramp holds the anchor's shift diagonal by design (`solve.md`,
  `materialized_session`), so the case was judged in the anchor's row scales; `solve_reynolds_ramp`
  passes the session's `stopping_step(case)`. Steering in the stopping measure as well was measured and
  is not built. Arms: `.claude/notes/turbulence-log.md`, "#701 — the arms"; full entry: log,
  "`stopping_step`".

### `StepOutcome` — the strategy's return is a record

`StepOutcome(phi, residual_norm, raw_cycles, alpha, inner_iterations, reached_target, max_inner_cycles,
full_step_ratio, refresh_due)` — a `NamedTuple` built in tests by one `_outcome` helper; never unpack it
positionally (an inserted field once silently shifted a test's unpack).
- **`residual_norm`** is `norm(R(phi))` at the returned iterate in the step's own measure, taken from the
  line search rather than recomputed. **It rests on one invariant: the step's measure and the driver's
  are the same object** (the march takes its measure from `strategy.norm()` and rebuilds the step's via
  `norm_builder`). On a fallback rung the search reports that rung's own measure; with no finite rung
  under `fallback_growth` (default 100) of `reference_norm` it returns `alpha = 0` at `reference_norm`; a
  search with no finite rung at all still reports a non-finite measure, which the guards test.
  `DualTimeStep` forms it from the `G` vector its last search kept, `R = G − β d (phi − φⁿ)`
  (`transient_term`); contract in `solve-globalization.md` ("The line search returns its residual
  VECTOR").
- **`reached_target`** — did the step meet its own stopping criterion. A cost redo fires only when
  `max_inner_cycles > retry.abort_above_cycles` and not `reached_target`.
- **`max_inner_cycles`** — the offset-corrected cost of the step's costliest single solve; `cycle_budget`
  stays summed (a cost cap).
- **`full_step_ratio`** — `norm(G(phi + delta)) / norm(G(phi))` of the first full-length trial
  (`LineSearchStep.full_step_norm`, recorded at rung 0 whatever the test decided). The dual-time step
  reports its first inner iteration's (so it compares step to step); `nan` only when every
  pseudo-transient attempt was rejected. Log column `rho1`. It shows the settle a step or two ahead,
  where `alpha` (a halving ladder) only says whether the full step was taken.
- ⚠️ There is no `binding_limit` (deleted 2026-10-14 with the whole-step limit); archived
  `history.csv` / `march.log` files still carry its column and `L` flags.

### Positivity — a per-cell step projection

The shift and the divergence guard do **not** carry positivity: the guard fires on a non-finite
residual, which `sqrt(k < 0)` produces only after the state is poisoned (bfs3d died at step 62 from two
cells of 23040 with `k < 0`; log, "Positivity").
- **Every coupled builder installs `_closure_step_projection(coupled, omega_step_ratio)` as
  `step_projection`**: `solve.composed_projection` of the k clip (`turbulence.positive_k_projection` →
  `solve.positive_block_projection`, each cell's decreasing `k` correction clipped to `tau = 0.99` of its
  room) and the ω bound (`turbulence.omega_step_bound` → `solve.bounded_block_projection`,
  `|Δ log ω| ≤ log(omega_step_ratio)`; default `DEFAULT_OMEGA_STEP_RATIO = e` under `LogScalars`, none
  for a direct ω; a finite ratio with direct ω raises). **The ω bound is part of the guard:** clipping
  `k` alone leaves ω's correction solved for a `k` move not taken (residual ratios to 4e91 on bfs3d).
- Both `PseudoTransientStep` and `DualTimeStep` carry `step_projection` (default `None`).
- There is no global fraction-to-the-boundary cap (`positive_block_limit`, `positive_k_limit`) and no
  whole-step limit seam (`step_limit`, `max_alpha=`, `stop_on_limit_stall`), both deleted 2026-10-14. A
  constraint the residual cannot express goes in `step_projection`. Traps if a whole-step cap returns:
  one near-zero cell throttles a global cap; its "no cap" must not be `1.0` if anything asks for a step
  longer than full; and a fraction-to-the-boundary cap collapses geometrically once a cell sits on the
  boundary. A/B: `.claude/notes/solve-globalization-log.md`, "bfs3d's retried steps: the global k cap,
  replaced by a per-cell clip with an omega bound (2026-10-14)".
- Not a log variable for `k`: `k = 0` is the wall condition, so `log k` is singular there
  (`turbulence.md`); `log(k+1)` bounds `k > −1`, not `k > 0`.

### The forward-step contracts live in `solve/strategy.py`

`NewtonStrategy`, `ShiftedNewtonStrategy`, `StepOutcome`, `StepReport`, `StepControl`, `StepFn` and
`within_tolerance` travel between the driver, the march, the strategies, the controls and the retry
policy, so they live in `strategy.py`. **It is a leaf and must stay one**: it imports `linear`, `norm`
and `relaxation`; `implicit`, `march`, `continuation`, `retry`, `step_control`, `march_log` and
`checkpoint` depend on it — an import back re-creates the cycle it removed. `backtracking_line_search`
stays in `implicit.py` (behaviour, not contract) and `MarchResult` in `march.py`.

### Step controls (`solve/step_control.py`)

- **`StepControl` — stateful, feedback-driven step reshaping on the eager march only (binding; the twin
  of `RelaxationSchedule`, deliberately not one interface).** It reads the previous `StepReport` and may
  raise under `jax.grad`. `newton_march(step_control=…)` calls `next_step(base, previous, state) ->
  (NewtonStrategy, new_state)`; the control returns a ready step (a `ConstantRelaxation` β leaf via
  `tree_at`, keeping `_march_step` a cache hit), so the march stays β-ignorant. **The control state
  survives refreshes** (#156): `newton_march` takes `control_state` and returns the final one on
  `MarchResult`, and `solve_coupled` threads it across segments — the opposite lifetime to the
  segment-local `residual_norm_0` and `drift_measure`. Unifying it with the SER schedule into one
  interface was rejected (it would union their needs and drag α onto the differentiable core).
- **`ShiftStrengthControl` is the shared base; a subclass supplies only `_adapt` (binding).** The base
  owns the first-step seed, the hold across a refresh, the `[beta_min, beta_max]` clamp, the `tree_at`
  swap onto a dynamic β leaf, `carry_beta`, `resumed_at`, `redamp`, `settled` and the floor release. State
  is `ShiftControlState(beta, memo, release)` for every control (`memo`: `None` for the Courant rule, the
  previous residual for the ratio rules). Written three times, the bookkeeping drifted (a missing
  `carry_beta`, a β reset at a refresh in one class only). `CflResidualDualTimeControl` reduces exactly to
  `DualTimeControl` at infinite ratio thresholds (pinned).
- Concrete controls, with `default_dual_time_control` choosing the default beside them:
  - **`DualTimeControl`** — the Courant β-ramp on the inner-loop factor α; the default for a dual-time
    march; carries β across refreshes.
  - **`ResidualRatioDualTimeControl`** — opt-in, keyed on the steady-residual ratio; safe against
    overshoot but pins β on a flat-residual plateau.
  - **`CflResidualDualTimeControl`** — opt-in; grows only when α ≥ `grow_above` **and** ratio ≤
    `hold_ratio`, brakes when α < `backoff_below` **or** ratio > `rise_ratio`, holds between. It exists
    because the α-only control grows into an overshoot an inexact (AMG) solve cannot tolerate, and the
    ratio-only control crawls on a plateau. Set the ratio thresholds from a logged march. Measurements:
    log, "`ShiftStrengthControl` / `CflResidualDualTimeControl`".
- There is no `AlphaTargetingControl` (the single-step α-targeter, deleted 2026-08-14):
  `.claude/notes/solve-globalization-log.md`, "THE SER β SCHEDULE RUNS BACKWARDS".

### A self-rescaling measure — never compare residuals across an outer-iteration boundary

`newton_march(norm_builder=…)` (a `(step, state)` builder) re-derives the `RowScaledNorm` at the state
each outer iteration begins from and holds it for that iteration. So the `R` reported at the end of step
N and the `‖G‖` entering step N+1 measure the same state in different scales (up to 2x apart early on
bfs3d; log). Never compare or print them adjacent.

### `MarchLogger` (`solve/march_log.py`) and `TextTable`

`MarchLogger` owns everything derivable from a `StepReport` (one formatter for every driver); case
quantities come from an injected `metrics: state -> {name: value}` (bfs3d: `compare.reattachment_metrics`).
`note()` writes to the log's own stream (never `print` beside it); `phase(label, total)` auto-numbers.
Pinned by `tests/unit/test_march_log.py`, which reads cells, not substrings.
- **Report fields that exist for the log:** `shift` (the β the step ran at), `escalations` /
  `diverged_retry` (`raw_cycles` counts only the accepted attempt, so a redone step otherwise looks
  cheap). The banner states `‖R₀‖` and the stopping target once per rung (and again whenever a rung
  re-bases), not per row.
- **Cost is reported offset-corrected:** lineax's `num_steps` carries +2 per solve;
  `restart_cycles(raw, solves)` in `solve/linear.py` is the one place the offset is stripped. The row
  shows `in` (inner count), `cyc` (corrected total) and `c/in`.
- **The step grid stays narrow (binding).** Columns `step, t(s), beta, in, cyc, R, a_min, rho1, flg`,
  pinned at `<= 70` characters by `test_the_grid_stays_narrow_whatever_is_switched_on`. Everything else
  rides in spanning aside lines beneath its row, one concern per line (preconditioner / case metrics /
  `cum`). Do not widen the grid; add an aside, or a column only if worth scanning down the run. Headings
  re-emit every `HEADINGS_EVERY` rows; a row after an inner block is re-headed with the label row alone.
- **No column heading may contain `|` (binding)** — `R`, not `‖R‖`; it breaks column-splitting tools.
- **`a_min` (step) vs `alpha` (inner).** An inner `alpha` reads 1.0 even when the search failed to
  descend (fallback to the longest rung within `fallback_growth`), and 0.000 when no finite rung came in
  under it. The step reports the minimum with any non-descending iteration folded in as 0, so
  `a_min=0.000` beside `alpha=1.000` is a full step that did not reduce `‖G‖`. In the inner table
  `rate ≥ 1` identifies every non-descending iteration exactly.
- **`on_inner` + `TextTable`** (`aquaflux/text_table.py`, a root leaf with no package imports):
  `inner_observer=logger.on_inner` renders `DualTimeStep`'s per-inner `(index, ‖G‖ in, ‖G‖ out, cycles,
  alpha)` as rows with `rate = ‖G‖out/‖G‖in`, opened at `index == 0` and closed by the step line (a
  footer: the outcome is unknown until the step returns). A redone step opens a fresh block per attempt.
  `TextTable` streams (rule, headings, `row`, `spanning` each return one line);
  `rule(title=None, *, fill="-", segmented=True)` draws the segmented grid rule, the `fill="="` bracket
  and the unsegmented span between two grids. An over-wide value widens its row rather than being
  truncated. Pinned by `tests/unit/test_text_table.py`.
- **Diagnostics are opt-in through `detail`** (`MarchLogger.DETAIL = {"inner", "fields", "residuals",
  "pc"}`); every hook stays safe to wire and no-ops when its name is absent; an unknown name raises.
  - `"fields"` — `fields=` a `state -> named arrays` extractor (`turbulence.coupled_fields`: pressure
    gauge-free, velocity per component, `ν_t` included); each field's relative change per step via
    `field_change_metrics`, keyed by the field's own name so it joins the residual grid. It shows whether
    the solution still moves, which a mesh-quantized case metric cannot. The logger builds the stateful
    change measure itself (it must be called once per step in order).
  - `"residuals"` — `residuals=` a `state -> {equation: residual}` extractor
    (`turbulence.coupled_residuals`): per-equation residual and its contraction `rate`, from
    `RowScaledNorm.per_block`, so the rows add up to the steered `R` exactly (pinned `rel=1e-12`). That
    requires equilibrating at the **previous** state (the march's measure is built at each step's start),
    so `coupled_residuals` is stateful and takes a `reference_state` seed a per-rung reporter must pass.
    `ν_t` gets `--` residual cells. One extra residual evaluation per step.
  - `"pc"` — a session's `observer=logger.on_refresh` reports each refresh's kind (`full` / `cost` /
    `none`) and its `RefreshTiming` phases; rendered on the step it preceded, summed across every refresh
    between two step rows.
