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

> ⚠️ **`coupled_continuation`, `coupled_lu_continuation`, `coupled_amg_continuation`, `lu_beta_tracking_refresh` and `amg_beta_tracking_refresh` no longer exist (deleted 2026-09-14, #371).** Entries below that name them are dated history. The coupled march is now one builder, `coupled_step`, given a preconditioner value (`BlockDiagonal` or `MaterializedJacobian(FieldSplit | BlockInverse)` (a `CompleteLu` inverse existed until 2026-10-11)), and a march that keeps its preconditioner current runs on a session (`open_session`) — see the rename table in `.claude/rules/turbulence.md`.

> Split out of `solve.md` (2026-08-18). See `solve.md` for the package-wide contracts, current
> configuration, and binding decisions this file assumes.
>
> **Dated investigation lives in `.claude/notes/solve-march-log.md`** (split out 2026-10-17; outside the auto-loaded tree, so
> it never auto-loads). This file keeps what is built, the defaults and the binding decisions; a new
> dated measurement goes in the log, per `solve.md`'s "Where new content goes".

## `StepHistory` — the per-step record for a program to read (BUILT 2026-10-05)

`march_history.py::StepHistory(path, clock=)` is a host-side observer with the march hooks'
names — `on_checkpoint(report, state)` writes a row; `on_refresh(timing)`, `on_retry(reason, attempt,
beta)` and `on_residuals(terms)`, called while the step is under way, fill in the row it then writes
and are reset after it. **What it offers is declared: `MarchRecorder` (same module, exported, `runtime_checkable`,
#669)** — `on_checkpoint`, `on_retry`, `on_refresh` and an `on_residuals` that may be `None` (no taker,
so the march skips their cost). A `StateCheckpointer` is not one (it records steps only). The case
runner fans the hooks out by `isinstance(recorder, MarchRecorder)` and the solver specs type their
recorder as one, so no hook is looked up by name; a recorder offering only some hooks gets none of the
extras (pinned per missing hook in `test_case_run.py`). The one remaining attribute probe here,
`hasattr(value, "item")` in `_cell`, deliberately recognizes NumPy/JAX array scalars without importing
either; it is not a project contract. Columns: `step` (1-based, counting on across continuation segments), `seconds`
(since construction, injected clock), every `StepReport` field under its own name with the report's own
`step` renamed `segment_step` (it restarts per segment), `restart_cycles`, then (2026-10-07, for the
browser interface's Run section) `refits` / `refit_seconds` (refreshes of kind `full` or `inner`; a
`none` reused the standing preconditioner and is not counted), `retry_reasons` (`;`-joined, empty for a
step taken as-is), and one `residual_of_<equation>` per equation the march's measure names. Floats are
written by `repr` (exact round trip), booleans as 0/1, and NumPy/JAX scalars are unwrapped with
`.item()` first — **a `np.float64` IS a `float` subclass and its repr in NumPy 2 is `np.float64(0.1)`**,
which is what that unwrap prevents. `COLUMNS` is derived from `StepReport._fields`, so a field added to
the report reaches the file without an edit here.
- **⚠️ The header is written with the FIRST ROW, not at construction** (changed 2026-10-07): the
  equation columns are known only once the march reports them, so until the first step the file exists
  and is empty. `aquaflux_ui.ConvergenceHistory.read` reads an empty file as zero steps. A step naming
  other equations than the header is refused (`ValueError`); one naming none leaves its cells empty.
- **⚠️ The prefix is `residual_of_`, NOT `residual_`** — the first version used `residual_`, and the
  existing `residual_norm` / `residual_ratio` columns matched it, so a reader took them for two more
  equations. Caught by `test_the_residual_columns_of_the_total_are_not_taken_for_equations` (UI side).
- It is the machine-readable sibling of `MarchLogger` (which formats for a person), not a replacement:
  the case runner writes both (`Outputs.log`, `Outputs.history`, default `history.csv`), and
  `aquaflux_ui` plots the history (Results' convergence plot, Run's five plots and events).
Pinned by `tests/unit/test_march_history.py`.

## `StepReport.station` / `.arrived` and `newton_march(on_residuals=)` (BUILT 2026-10-07)

- **`station`** (the homotopy's station index, `0` without one; only equality between adjacent steps
  means anything) and **`arrived`** (whether the step drove the TARGET problem; `True` throughout a march
  without a homotopy) are set from the same `homotopy.station` / `arrived` the march already reads, so a
  record can mark where a continuation ended. Pinned by
  `test_each_report_says_which_station_its_step_drove_and_whether_that_was_the_target`.
- **`on_residuals({equation: term})`** is called after each step, before `observer` and `checkpoint`,
  with each named block's term of **the measure the step was judged in** (`active_step.norm()`) at the
  station residual the step drove — the numbers whose Euclidean combination IS the report's
  `residual_norm` (pinned to `rel=1e-12` on a real flow march,
  `test_each_steps_residual_is_reported_by_equation_and_the_terms_make_up_its_norm`). It comes from
  `solve.norm.named_blocks(measure, residual_fn, state)`, one compiled call per block structure
  (`_block_terms`, `filter_jit`), and is skipped for a measure that is not a `NamedBlockMeasure` (`norm.py`, exported, `runtime_checkable`:
`names` + `per_block`, #669 — `named_blocks` asks with `isinstance`, not `getattr(measure, "names", ())`)
or whose blocks are unnamed: `RowScaledNorm` /
  `BlockScaledNorm` gained a static `names` (empty by default, refused unless it names each block once),
  and only the two row-scaled builders set it — `coupled_scaled_norm` (`coupled_equation_names`; both in
  `turbulence/measures.py`) and `FlowMeasures.row_scaled` (`flow.flow_equation_names`). The block-scaled
  coupled measure's flow block holds every flow equation, so it stays unnamed. Costs one residual
  evaluation per step, and nothing when `None`. Threaded through `staged_march`, `solve_coupled` and
  `solve_flow_march` beside `on_retry`; `solve_reynolds_ramp` reaches it through `**solve_kwargs`.
  - **This overlaps `turbulence.coupled_residuals` + `MarchLogger(detail={"residuals"})`**, which
    computes the same per-block numbers by re-deriving the measure from outside the march (stateful,
    needs the continuation's shift policy and a reference state, coupled RANS only, no caller in the
    package or `validation/`). The hook is the residual-agnostic form; whether the logger's `residuals`
    detail should be fed from it and `coupled_residuals` deleted as dominated is an open question, not
    decided here.

## `newton_march(adaptive_tolerance=)` and `StepReport.linear_tolerance` (BUILT 2026-10-15, #645; default)

`AdaptiveLinearTolerance` (the default on every march since 2026-10-15, `DEFAULT_ADAPTIVE_TOLERANCE`; `None` switches it off) varies the relative tolerance each step's linear solve stops at —
the rule, its measurements and why it is not Eisenstat–Walker are in `solve.md`'s contract entry (measurements: `.claude/notes/solve-log.md`, "Contracts: `adaptive_tolerance.py` — measured on the built setting (#645)"). On the
march side: the solver's base tolerance is read and converted to a float64 array leaf once, before the
first step; each step (and a redo of it) runs `with_relative_tolerance(solver, tolerance)`; after the
step, the next tolerance is chosen from its `alpha`, its reported residual over the previous report's
(deliberately across the measure rebuild — see `solve.md`) and the stopping test's norm and target.
`StepReport.linear_tolerance` is the tolerance the step RAN at (`nan` without the setting), so a replay of
the rule over the reports reproduces it (pinned). `tight_solver` is untouched.

## The observed march — newton_march, triggers, controls, logging

- **`ResidualHomotopy` — a continuation walked INSIDE one march, not as a ladder of marches (BUILT
  2026-09-09).** A `Protocol` on `newton_march(homotopy=…)` with two methods: `enter(step, previous)`
  makes the station that step solves current and returns its residual, and `arrived(step)` says whether
  that station is the **target**.
  - **`RampSchedule` / `RampWalk` (`solve/ramp_schedule.py`, BUILT 2026-10-11) — where each step of a
    geometric ramp runs, and what may END it early.** The settings (`stations`, `steps_per_station`,
    `end_when: StepReport -> bool`, `finish`) are one frozen value; `schedule.walk()` hands each march a
    mutable `RampWalk` that records a `RampPosition(station, progress)` per entered step (`progress` 0 at
    the anchor, 1 at the target; a homotopy turns it into its parameter, `anchor ** (1 - progress)`).
    Generic, so it lives in `solve/` (Principle 3.6): `ViscosityRampHomotopy` takes `ramp=RampSchedule`,
    and so do `solve_reynolds_ramp` and the case file's `ViscosityRamp.schedule(step_control)` -- one
    configuration object instead of `stations`/`steps_per_station` keywords copied onto each builder.
    - **The early end:** the first report OFF the target for which `end_when` is true ends the walk; the
      rest of the span is walked in `finish` equal steps of progress (equal geometric steps of the
      parameter), the last being the target. Read once; a report from the target never ends anything
      (a ramp that arrived first just arrives). Refused: `finish` without `end_when`, and `end_when`
      with `steps_per_station != 1` (measured only at one; it would cut a station short).
    - **`end_when` is the march's own `step_control.settled`** (`solve-globalization.md`), so the ramp
      ends on the same evidence the release waits for. Why it exists and what it buys -- the ramp's best
      length is the shift's descent plus the settle, not the viscosity step; ending on the settle made
      paces 16 and 24 cost 124 and 118 cycles against the hand-tuned 111, and a 3-step finish 114 -- is
      in `.claude/notes/solve-globalization-log.md` ("Ending the ramp on the settle signal"). **Opt-in;
      pitzDaily's file still ships 12 fixed stations** (111 < 114).
    - **On bfs3d the settle-ended ramp did not beat the shipped fixed station count (#678);** the measured arms are in `.claude/notes/solve-march-log.md`, "`RampSchedule` / `RampWalk` — on bfs3d it does not replace the station count".
    - Station ids: the fixed walk numbers stations `0..stations` exactly as before (byte-identical
      reports); finishing steps continue from the station the end fired at, so the target's id is not
      `stations` on an ended walk -- only equality between adjacent steps means anything.
    - Pinned by `tests/unit/test_ramp_schedule.py`; six mutations of the walk (the arrived guard, the
      finish's start, the new-station-per-finish-step, an off-by-one finish, firing twice, the order
      check) each fail at least one test.
  - **`enter` is handed the PREVIOUS STEP'S REPORT (2026-10-11), so a homotopy may decide where a step
    runs from how the march went** -- the seam a ramp ending on the settle needed
    (`solve.RampSchedule`, `solve.md`). The first step is entered twice with `None` (once for the
    damping anchor, once to take it). ⚠️ The march's look-backs ("did the previous step enter a
    station?", the `rebase` test) now read `reports[-1].station` / `reports[-2].station` rather than
    asking the homotopy about past steps, so `station(step)` is asked only of the step just entered and
    an implementation need not remember history. Pinned by
    `test_each_station_is_entered_with_the_report_of_the_step_before_it` (mutation: handing an older
    report fails it). `None` (the default) is byte-identical and is pinned by a test, since an
  added seam that moves the incumbent path is a defect rather than a feature.
  - **Current status:** walking the continuation inside one march (`rebase` + `redamp`) beat pitzDaily's Reynolds ladder of separate marches on steps, restart cycles and wall at the same `x_r/h` (2026-09-09: 40 / 261 / 363 s against 69 / 417 / 555 s). The ladder's per-rung costs and the four-arm table are in `.claude/notes/solve-march-log.md`, "`ResidualHomotopy` — what it is for, and the four-arm result".
  - **⚠️ THE STOPPING TEST IS GATED ON ARRIVAL, and this is the load-bearing correctness property.** A
    small residual at an intermediate station says nothing about the target, so `newton_march` may not
    stop — and `MarchResult.converged` may not be `True` — until `arrived`. Both directions are pinned
    by starting a march **at a station's own exact root**, where the tolerance test passes on step one:
    without the gate it returns there and reports success at a state that does not solve the target.
  - **Per-step viscosity (`steps_per_station=1`) is the cheapest schedule measured on both flagship cases;** the earlier claim that a station should span several steps to save re-materializations was retracted. `steps_per_station` remains the knob, set from a schedule measured on its case. Trail: `.claude/notes/solve-march-log.md`, "`ResidualHomotopy` — \"a station spans several steps on purpose\" retracted".
  - **⚠️ `redamp` — RE-DAMP ON ENTERING A STATION, AND THE MARCH BREAKS WITHOUT IT (BUILT 2026-09-09).**
    `ShiftStrengthControl.redamp(state, factor)` multiplies β by `ResidualHomotopy.shift_factor(step)`
    on the step that first faces a new station, and `newton_march` **holds** the control for that step.
    It is the companion of `rebase` and they fire on **adjacent steps for different reasons**: this on
    the entering step, that on the next one, which is the first whose ratio would straddle the boundary.
    - The ratio rule's spurious brake at station changes, which `rebase` removed, had been holding β out of a collapse regime; without `redamp` the march fell into it. Measured march: `.claude/notes/solve-march-log.md`, "`redamp` — the spurious brake `rebase` removed was load-bearing" (the general lesson is at the top of `solve-globalization.md`).
    - **The hold is load-bearing and reproduces the accident's arithmetic exactly.** An entering step
      forgoes its usual `/grow`, so a change is worth `factor * grow`, not `factor` — which is what the
      old brake did, since `_adapt` chose brake *instead of* grow rather than in addition to it. A test
      written expecting `factor ** changes` measured `(factor * grow) ** changes`; the implementation
      was right and the expectation wrong.
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
    - **The collapse located the refit floor `PC_BETA_FLOOR`, not `beta_min`;** both flagship case files dropped the refit floor on 2026-10-13, and `beta_min = 0.005` is stable but slow at the target (`ShiftStrengthControl.release_floor`, `solve-globalization.md`). Measurements: `.claude/notes/solve-march-log.md`, "`redamp` — corollary: the refit floor".
    - `shift_factor` defaults to `1.0` on the protocol, so a homotopy indifferent to the shift pays
      nothing and the march is byte-identical to one that never asks.
  - **`station_step(step, station, arrived) -> step` — reshape the STEP for the station, the counterpart
    of what a homotopy does to the PROBLEM (BUILT 2026-09-11).** Called once per outer step while a
    homotopy is running, before the step control; `None` (the default) is byte-identical, and it is not
    consulted at all without a homotopy, since there are no stations then.
    - **Why it cannot live in the shift policy.** A policy is handed the iterate, the residual and `β` —
      all three measure *progress*, which is a different quantity from *which station this is*, and on a
      short ramp they come apart completely. Measured on a 16-station pitzDaily ramp: a per-block damping
      keyed on `β` has released fully by step 12 (the control floors `β` there) and one keyed on the
      closure residual by step **6** (it falls 100x in six steps, being dominated by the initial
      transient). Both spend their whole release inside the ramp and hand the target station nothing.
      The station index is known only to the march, which is why it arrives from here.
    - ⚠️ **Return a step differing only in ARRAY leaf values** (`eqx.tree_at` over a fixed structure),
      exactly as the march's own per-step measure swap does. `equinox.filter_jit` compares non-array
      leaves by value on the static side, so swapping a Python scalar recompiles the whole solve once
      per station — turning the cheapest setting in the march into its dominant cost. That is not
      hypothetical: the same trap, via a `float` molecular viscosity, cost ~10 % of a `bfs3d` march.
    - Its first measured use (damp a closure block harder while a viscosity ramp walks, release at the target) did not pay; the seam is kept as the only way to express a per-station setting. See `.claude/notes/solve-march-log.md`, "`station_step` — its first measured verdict is negative".
  - **The damping anchor `‖R₀‖` is taken at the FIRST STATION, not at the target.** It is the scale the
    first step's inner loop is judged against, so it has to be the problem that step runs; with no
    homotopy it is `residual_fn` and nothing moves. `residual_fn` stays the **target** residual and is
    what `reference_norm` defaults to, so the *stopping bar* is the target problem's throughout rather
    than a moving one.
  - **The row-scaled measure and the drift measure stay on the TARGET assembler, deliberately.** One
    yardstick across every station makes progress comparable down the log; the arrival gate is what
    makes it safe, since the march can only *stop* where the residual really is the target's.
  - **⚠️ Reachability is easy to fake here and is pinned explicitly.** An unwired `homotopy` leaves the
    march bit-identical, and every property asserted of it then holds for an unrelated reason. The test
    marches a ramp whose stations have visibly different roots and cuts the budget off exactly at
    arrival, so the state must sit on the **last ramp station's** root — which only happens if the
    stations were the residuals actually being driven.


- **⚠️ `MarchLogger.on_step` and `.on_checkpoint` are MUTUALLY EXCLUSIVE renderings of one event —
  wire one, never both (measured 2026-08-15).** Both call `_log`; they differ only in whether the
  injected case metrics are appended (`on_checkpoint` has the state, `on_step` does not). But
  `newton_march` calls its `observer` and `checkpoint` seams **unconditionally on every step**, so
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


- **`march.py` — BUILT (`newton_march`, `StepReport`/`MarchResult`, `RefreshTrigger`/`CycleGrowthTrigger`):
  the observed, forward-only march that drives a mid-march preconditioner refresh.**
  - **ONE march (binding — 2026-09-15, phase 3 of the unification; this replaces "two marches, one
    decision layer").** `newton_march` is the only Newton loop in the package. The traced `_forward`
    (a `lax.while_loop` inside `RootSolver`) is **deleted**; `RootSolver.solve` now
    marches with `newton_march` on `stop_array_gradients` copies, owns the root guard, and hands the
    root to `root_adjoint`. What that buys is that a capability can no longer exist on one loop and not
    the other — the shape of #369, which phase 2 fixed inside `solve_coupled` and this removes the
    remaining home for. What it costs is `jit`/`vmap`: a Python loop cannot run under either, so every
    entry point refuses them up front through
    `refuse_a_transform_the_march_cannot_run_in(pytree, caller=…)`, which lives **here** in `march.py`
    and is called by `RootSolver.solve`, `solve_coupled` and `solve_flow_march`. That
    was decided, not discovered: nothing in the repository `vmap`s a solve, and the four library builders
    that jitted one did so for compile-cache reuse, which the `filter_jit`-compiled `_march_step` gives
    them anyway (see `solve.md`'s `implicit.py` entry for the residual-identity condition that keeps).
    The march is still **forward-only and still carries no guard by design** — stopping short on a
    trigger is its purpose — so every caller reads `MarchResult.converged` before treating the state as
    a result.
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
  - **Why the early stop is the DRIVER's, not the solver's (binding — do not "simplify" it back).**
    The root guard raises whenever the terminal state is not a root, and a trigger-stopped segment exits
    un-converged *by design*. Putting the early stop behind the guard would require an **exemption** in
    it — a production path returning a non-root without raising, which is exactly the
    silent-wrong-gradient hole the guard exists to close. So the trigger stays a `newton_march`
    argument that `RootSolver.solve` does not pass, and the driver that does pass one
    (`solve_coupled`) owns its own convergence test at the end. Chunking a march with `max_steps=1`
    fails independently: it recomputes `residual_norm_0` per chunk, pinning the SER ramp at β₀ forever.
  - ⚠️ There is no traced finishing solve: the march's state is the answer, and the root guard is each driver's explicit test on `MarchResult.converged` before `root_adjoint`.
  - **Two reference norms, and conflating them freezes the march (binding).** `residual_norm_0` is
    **segment-local** (recomputed at each `newton_march` entry, handed to `stepper()` for the SER ramp);
    `reference_norm` is **global** (fixed across segments, used for the convergence test and the reported
    ratio). Substituting the second for the first pairs a refreshed, larger shift diagonal with the small
    β belonging to the pre-refresh residual — the over-damping freeze documented in `turbulence.md`.
    - **Resuming a march (2026-10-07; shift and anchor 2026-10-08): `resume=Resumption(...)` on `staged_march`,
      `solve_flow_march`, `solve_coupled` and `RootSolver.solve`, built on
      `newton_march(reference_norm=, damping_reference=)`.** A march resumed at a state re-measures
      `residual_norm_0` there, so the ratio `‖R‖/‖R₀‖` is ~1 and beta reopens at `beta0` — the reason a
      resumed default march did not continue the interrupted one (the memoryless SER schedule holds no other
      history; a report's `shift` reads 0.0 for it, which looked like a reset and was not). A `Resumption`
      carries `reference_residual` (the stopping scale, `atol + rtol * reference`, and the first segment's
      damping anchor unless `damping_reference` is given), `damping_reference` (the anchor of the segment the
      march was in — differs from the reference once it has refreshed, since a later segment re-bases) and
      `shift` (what a dual-time march's `ShiftStrengthControl` had walked to). The march still STARTS from
      the residual it measures at `phi0` (`current`, the loop's own test) — only the schedule's anchor and,
      via `reference_norm`, the stopping bar change. `StepReport.damping_reference` records the anchor each
      step was damped against, and `report_record` writes it, so a checkpoint carries it.
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
  - **Per-step jit cache hit is mandatory, not an optimization (top implementation risk).** The per-step
    call goes through the module-level `eqx.filter_jit`'d `_march_step`, taking the `NewtonStrategy` **and**
    the residual as *arguments*. Two caller obligations: pass the **same** `strategy` object across a
    segment (a rebuilt one is the intended one-off recompile per refresh), and pass a **bound module
    method** (`coupled.residual`) rather than a freshly-built `lambda`, which `filter_jit` hashes by
    identity. Retracing per step would cost the 60–240 s compile *every step* and dominate the march it
    accelerates. Pinned by a trace-counting test (extra steps add zero traces). Note the residual is
    invoked several times *within one trace* (step, line-search ladder, norm), so trace count ≠ compile
    count — assert that further steps add none, not that the total is 1.
  - **`solve_coupled` declares no globalization setting, and must not be "fixed" to (binding).** They
    ride `**strategy_kwargs` (as one `Globalization`) to the step builder, which `solve_coupled` already
    splats at its build sites; declaring one as well -- `grow` once was, and `grow` itself was deleted
    2026-10-14 -- only forwards what is forwarded anyway. Pinned call-for-call by
    `test_globalization_knobs_still_reach_the_continuation_builder`. (`descent_backoff` /
    `descent_test` are deleted too — see `.claude/notes/solve-refuted-directions.md`.)
  - **The four refresh settings are ONE injected object — `RefreshPolicy` (`solve/refresh.py`),
    exported from `aquaflux.solve` (BUILT).** `trigger` / `limit` / `builder` / `refresh_preconditioner`
    were four keyword arguments on `solve_coupled`, meaningless apart: a `limit` with no trigger bounds
    a loop that never runs, a `builder` with no trigger is called once. The derived predicates and the
    one validation are on the object — `refreshes` (trigger AND budget), `observes` (does this force
    the eager march), `segments` (= `limit + 1`), `is_last_segment`, and `require_rebuildable`, which
    raises when a caller-supplied step has no builder to rebuild it with.
    - **It is a DRIVER-level object and `newton_march` does NOT take it (binding).** The march uses
      only `trigger` and `refresh_preconditioner`; `limit` and `builder` govern the *sequence of segments*,
      which is the driver's loop. Passing the whole policy down would hand the march two fields it
      cannot act on — the "smallest sufficient collaborator" rule, and the reason this differs from
      `RetryPolicy`, which the march consumes whole.
    - **`NO_REFRESH` is the shared default instance**, byte-identical to the old all-`None` defaults.
    - **There is no `RefreshPolicy.observes` (deleted 2026-09-15).** It decided whether `solve_coupled`
      ran an observed pre-march; that solve now has one march, so nothing selects between marches.
  - **The six retry settings are ONE injected object — `RetryPolicy` (`solve/retry.py`), exported from
    `aquaflux.solve` (BUILT).** `solver` / `divergence_cap` / `on_cycles` / `on_alpha` / `beta_factor` /
    `max_retries` used to be six parallel keyword arguments on **both** `newton_march` and
    `solve_coupled`, and they are meaningless apart: a `beta_factor` with no threshold escalates nothing,
    a `max_retries` bounds a loop that never runs. The three decisions taken from them read three or
    four each, so they are **methods on the policy** rather than free functions taking a subset —
    `retry_reason` (named `escalation_reason` until the 2026-08-17 split; before that
    `_escalation_reason`), `has_diverged` (was `_has_diverged`),
    `with_inner_abort` (was `_with_inner_abort`), plus `escalate`, which exists solely to keep the
    "scale the β leaf, never rebuild it from a float" rule in one place (rebuilding changes the leaf's
    dtype/weak type and recompiles the whole coupled solve on every retry).
    - **`NO_RETRIES` is the shared default instance**, so both signatures' defaults name what the default
      *means* rather than how it is spelled; it is byte-identical to the old all-`None` defaults.
    - **The ORDER stays on `newton_march`, not on the policy** — escalation first, `solver` as the
      fallback — because it is a property of the loop, not of the settings. Likewise `on_retry` stays a
      separate argument: it is *reporting*, and it belongs with the other observation seams rather than
      with the thresholds.
    - **`newton_march` takes the whole policy; a refresh policy would NOT be passed the same way.** The
      march uses all six, so the object is the smallest sufficient collaborator there. Do not extend this
      to arguments a callee does not need in full.
  - ⚠️ `retry.solver` is a `LinearSolverSpec` (settings), built once per march into `tight_solver`
    — see `solve-globalization.md`. Every `retry.solver` below means the solver it builds.
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
  - **Reactive β-escalation (BUILT).** A step whose line search collapses (`retry.on_alpha`) or that diverges is redone from the same pre-step state at β × `retry.beta_factor`, up to `retry.max_retries` times, before the tight `retry.solver` fallback; `retry.abort_above_cycles` redoes an expensive step at the SAME shift. There is no `retry.on_cycles` (the cost threshold stopped escalating 2026-08-17 — `solve-amg-multigrid.md`, "the cost guard and the shift escalation are now separate responses"). The sub-entries below are its live parts.
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
    - **Escalation leads and `retry.solver` is the FALLBACK:** run divergence-first, the tight retry's grind on `bfs3d` was wasted work the escalation then superseded; the tight solve is kept for a non-finite step escalation cannot fix. Measurement: `.claude/notes/solve-march-log.md`, "Reactive β-escalation — why escalation leads (the reorder, bfs3d)".
    - **The bailout is reactive by necessity:** on `bfs3d` no cheap static operator property predicted a bad step, so the design is detect-then-react. Refresh for staleness, escalate β for stiffness. Trail: `.claude/notes/solve-march-log.md`, "Reactive β-escalation — the deleted proactive β-mismatch refresh and the Step-0 diagnostic"; ledger: `.claude/notes/solve-refuted-directions.md`.
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
    - **A COLLAPSED STEP LENGTH is the third way a step dies, and neither the cost bailout nor the
      escalation could see it — `retry.on_alpha` / `abort_below_alpha` (BUILT 2026-08-08).** The
      escalation's two triggers are cost-with-the-target-unmet and divergence. A step whose *solves are
      cheap* and whose *residual is finite* but whose **line search cannot move** is neither, so it was
      accepted as taken, and the only thing raising β was the step control's own backoff — **one doubling
      per outer step, each doubling paying a full step to discover it was not enough.**
      Caught on the 3D `bfs3d` field-split march: four consecutive target-rung steps at `a_min` 0.000 while the control doubled β once per step. Table: `.claude/notes/solve-march-log.md`, "`retry.on_alpha` / `abort_below_alpha` — the bfs3d steps 51-54 collapse".
      Both halves are now built and are the same predicate in two places, as the cost bailout already is:
      **`newton_march(retry=RetryPolicy(on_alpha=α))`** escalates β (reason `"alpha"` on `on_retry`), and it is pushed
      into **`DualTimeStep.abort_below_alpha`** by `RetryPolicy.with_inner_abort` so the inner loop exits at the
      collapse instead of iterating on. `RetryPolicy.retry_reason` now owns which of the three reasons applies,
      so the decision and the string reported for it cannot disagree.
      **Both cost and step-length reasons require the target unmet; divergence does not** — a non-finite
      residual is not a result to keep because the loop happened to meet its tolerance. Both `None`
      (default) is byte-identical.
      End to end on `bfs3d` (2026-08-08, `retry.on_alpha = 0.01`) one retry saved 8 steps and ~9% wall at identical `x_r/h`; table and attribution in `.claude/notes/solve-march-log.md`, "`retry.on_alpha` / `abort_below_alpha` — measured end to end".
      **The threshold was calibrated from the baseline's own step table, not chosen:** no productive step
      went below `a_min` 0.191, all four dead ones reported 0.000 (inner collapses at 0.001 and 0.003),
      so 0.01 sits an order of magnitude clear of both. Recalibrate on another case rather than porting
      the number.
      What a doomed attempt's re-fit costs is unmeasured since a cost re-fit ends the step (2026-10-10); the earlier measurement is in `.claude/notes/solve-march-log.md`, "`retry.on_alpha` / `abort_below_alpha` — known waste under the old mid-step refresh".
    - **The escalated β is CARRIED into the control — so a static β floor can be dropped and the *controller*
      decides how low is safe (BUILT).** β is inverse to the pseudo-timestep, so a static `beta_min` is a cap
      on the *largest* timestep the march may take, applied everywhere — which slows convergence in regions
      that could safely take a bigger step. The escalation is the per-region feedback for "how low is safe
      *here*": it fires exactly where β went too low. But without carrying it back, the next outer step's
      `step_control.next_step` recomputes β from the control's own (floor-ward) trajectory and **re-pays the
      escalation every step** — the observed low-β tail (β pinned at the floor, each step re-escalating). So
      after an escalation `newton_march` seeds the control's carried β with the escalated value via
      `step_control.carry_beta(state, β)` — **one implementation on `ShiftStrengthControl`, over the shared
      `ShiftControlState(beta, memo, release)` state, so no control can be missing it** (it once was: the deleted single-step
      α-targeter had none, and the `hasattr` guard the march used then meant its escalation feedback
      vanished in silence; since #281 the march asks `isinstance(control, ShiftCarryingControl)`). The memo is preserved across the seed, so a ratio-keyed control does not lose its
      reference and misread the next step as a huge reduction. The control then continues its grow/brake dynamics *from* the discovered-safe β, so
      `beta_min` can be driven toward zero and the controller — with escalation as the safety net and the
      carry as the memory — finds how large a timestep each region tolerates, rather than a global floor
      capping it. Only fires when β was actually escalated and the control is a `ShiftCarryingControl`;
      no escalation ⇒ byte-identical. Pinned by `test_newton_march.py`
      (`…carries_the_escalated_beta_into_the_control`) and `test_step_control.py`
      (`test_carry_beta_seeds_the_carried_state`).
  - **`CoefficientDriftTrigger` — the PREFERRED staleness trigger: measure the drift, don't infer it
    from cost (binding for new work).** A frozen preconditioner is stale exactly when the operator it
    approximates has moved, so the honest signal is that movement itself. `StepReport.drift` carries a
    **scalar** relative drift produced by `newton_march(drift_measure=…)`; the coupled RANS measure is
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
    - **`threshold = 0.1` (the default) was calibrated on one instrumented cold-IC pitzDaily march (2026-07-25),** where drift leads cost and 0.1 fires after the recirculation forms. Data: `.claude/notes/solve-march-log.md`, "`CoefficientDriftTrigger` — calibration on a cold-IC pitzDaily march".
    - **The shipped default was originally 0.5 and would never have fired on this march** (drift
      reaches only 0.24 by step 23). A "conservative" placeholder chosen by intuition was not
      conservative — it was inert. Trigger numerics must come from a logged march, not from judgement
      about what sounds safe; the replay procedure exists precisely because that judgement is unreliable.
      Calibrated on **one** geometry, so treat 0.1 as a starting point elsewhere, not a constant.
    - End to end on that march the refresh cut per-step cost 5-8x and repeated per segment, but did not fix step productivity — state that when reporting it. See `.claude/notes/solve-march-log.md`, "`CoefficientDriftTrigger` — end-to-end result at `threshold = 0.1`".
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
  - **Observation changes nothing about a coupled solve (binding, 2026-09-15, #369).** `solve_coupled`
    has one march: observers, refreshes, controls and retries all run in it, on `stop_gradient` copies,
    and `root_adjoint` attaches the adjoint at the root. `on_step` / `on_checkpoint` therefore report
    without selecting anything — pinned by `test_observing_a_solve_changes_nothing_about_it` (bit-identical
    fields with and without an observer) and `test_the_march_reports_progress_without_a_refresh_trigger`.
    ⚠️ Trap from the deleted eager pre-march: when observing a solve selects a different path through it, an instrumented run is not the run under investigation.
  - **`checkpoint` is a SECOND seam, separate from `observer` (binding).** `checkpoint(report, state)`
    carries the state; `observer(report)` carries only numbers. Keeping the state off the report history
    is what keeps a `RefreshTrigger` a pure function that can be replayed offline against a logged march
    — put the state on that seam and a trigger could read the physics, and trigger calibration would cost
    one full solve per candidate instead of one logged run for all of them.
  - **Reporting seam.** `StepReport(step, raw_cycles, residual_norm, residual_ratio, alpha, drift,
    inner_iterations, shift, escalations, diverged_retry)` + `MarchResult`,
    plus an optional streaming `observer` (a long march must not withhold all logging until it finishes).
    The trigger and a future logger consume the identical objects, so there is no second reporting path.
    Per-step observation is available wherever a driver passes an `observer`, which since phase 3 means
    every march including the differentiable one — the state it reports is a `stop_gradient` copy, so an
    observer sees concrete values under `jax.grad` too. `RootSolver` does not *expose* the
    argument (its surface was left alone in that change); `solve_coupled` does.
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
    whether the step was redone: `raw_cycles` counts only the **accepted** attempt, so a redone step is
    indistinguishable from a cheap one, **and a retry mechanism left unconfigured never announces its
    absence** — which is not hypothetical (a bfs3d march ran with `cycle_budget` set but
    the cost threshold at its `None` default, so the beta-escalation never fired and nothing in the log
    said so; the cap exists to trip that trigger).
    The logger also reports the **reference norm and the stopping target**: the test is
    `‖R‖ <= atol + rtol·‖R₀‖`, and the march reports `‖R‖` and `‖R‖/‖R₀‖` but not `‖R₀‖`, so the target
    was invisible. Pinned by `tests/unit/test_march_log.py` (synthetic reports, no solve).
  - **Solve cost is reported offset-corrected, split by inner iteration (BUILT).** The raw count is
    lineax's `num_steps`, which carries **+2 per solve**, so a two-inner step reporting `raw_cycles = 6` did
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
    case metrics / `cum`; a `limit` line, and the `L` flag, went with the whole-step limit 2026-10-14). Run together on a single line those three had to be read in full to
    find any one of them.
    An over-wide value **widens its row rather than being truncated**: a cut-off number is a wrong
    number. Pinned by `tests/unit/test_text_table.py`.
  - **`StepOutcome` — the strategy's return is a record, not a tuple (BUILT).** It grew to ten
    values (`binding_limit`, the ninth, left 2026-10-14 with the whole-step limit; `refresh_due` joined
    2026-10-11 with `StopsForRefresh`, #290; it is nine now: `phi, residual_norm, raw_cycles, alpha,
    inner_iterations, reached_target, max_inner_cycles, full_step_ratio, refresh_due`), which is the missing-object smell: a positional tuple is where a consumer silently
    mis-unpacks one field for another, and every growth broke all five test doubles separately — which
    is why they are built by a single `_outcome` helper now. **The growth to eight demonstrated exactly
    that failure:** `residual_norm` was inserted *second*, and the one test still unpacking positionally
    (`phi_next, cycles, alpha, *_ = ...`) began reading the norm as its cycle count and failed on the
    dtype. Several of the fields are there because a march could not otherwise act correctly:
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
    - **`reached_target`** — did the step run to its OWN stopping criterion, or was it cut short? A
      cost-only escalation cannot tell an expensive success from a grind and **discards the success**:
      measured, an inner loop that reached `‖G‖ = 3.0e-6` against a `1.0e-5` target was thrown away for
      costing 54 raw cycles, wasting the work *and* replacing it with a shorter step. `newton_march`
      now fires only when `cycles > retry.abort_above_cycles` **and not** `reached_target`.
    - **`max_inner_cycles`** — the offset-corrected cost of the step's most expensive SINGLE solve, and
      what `retry.abort_above_cycles` triggers on. A **summed** threshold is not a difficulty signal: it
      grows with how many times the step solved, so at a threshold of 40 a 5-inner step trips at 6
      corrected cycles per solve and a 1-inner step at 38 — a 6× difference in sensitivity decided by a
      count that says nothing about conditioning. (Measured impact on one 63-step march: the two
      triggers agree on 62 steps and disagree on one — the expensive step, which per-inner catches and
      summed misses. The change is correctness, not speed.) **`cycle_budget` stays SUMMED** and should:
      it is a cost cap, and total cost is exactly what it caps.
    - ⚠️ **There is no `binding_limit`** (on `StepOutcome`, `StepReport`, the march history or the log;
      deleted 2026-10-14 with the whole-step limit it reported). An archived `history.csv` still carries
      the column and an archived `march.log` its `limit` lines and `L` flags; nothing reads either back.
    - **`full_step_ratio` (2026-10-11)** — the step's first full-length trial over the state it started
      from, `norm(G(phi + delta)) / norm(G(phi))`: what the line search compares with one to accept the
      full step, and used to discard when it rejected it. The DUAL-TIME step reports its FIRST inner
      iteration's (whose `G` at the anchor is the steady `R`, so it compares step to step); the two
      single-step strategies the accepted attempt's. It comes from `LineSearchStep.full_step_norm`,
      which the ladder records at rung 0 whatever the test decided (the ladder starts at the full step
      since `grow` was deleted, 2026-10-14; `nan` only for a pseudo-transient step whose every attempt
      was rejected). On the report and in `StepHistory`, and the
      march log's `rho1` column. **Why:** `alpha` is a halving ladder (1 or 0.5 at the floor), so it
      says only WHETHER the full step was taken; on pitzDaily this ratio walked 1.23, 1.13, 1.04, 0.90
      over the four steps before the settle, so the settle is visible a step or two ahead
      (`.claude/notes/solve-globalization-log.md`, "Ending the ramp on the settle signal"). Pinned
      against closed forms (a diagonal shifted Jacobian on `phi**3 - theta`) for the dual-time step
      (accepted AND rejected first steps), the pseudo-transient step and the damped Newton step through
      the march; mutation-checked (wrong inner, the kept rung reported instead, the march dropping it).
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
    state" is then lying. Skip a non-finite report, or do not claim "good". (A case's `initial:
    Checkpoint` refuses non-finite fields when it reads one, so a restart does not start from poison;
    the writer itself is unchanged.)
    **Finding one again (2026-10-02):** a new process has no `StateCheckpointer.latest`, so
    `find_checkpoint(directory, step="latest", prefix=...)` looks files up by name — the highest observed
    step (not the newest by mtime), ignoring the `.partial` staging name, matching the prefix exactly.
    `checkpoint_name` is the one place the `<prefix>-<step:05d>.npz` pattern lives, shared by writer and
    finder, and `report_record(report)` the one place a step's numbers are listed, shared by the default
    serializer and the case layer's. The default serializer still writes a bare `state` array; a case's run
    injects its own `save=` (physical fields + header, `aquaflux/case/restart_file.py`).
  - **`on_retry(reason, attempt, beta)` — say WHY a step is being redone (BUILT).** `newton_march`
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
    `newton_march(norm_builder=…)` (a `(step, state)` builder) re-derives the `RowScaledNorm` at the state each outer iteration
    *begins from* and holds it for that whole iteration. So the `R` reported at the end of step N and
    the `‖G‖` entering step N+1 measure the **identical state** in **different scales**. Measured over
    one 62-step `bfs3d` march: they differed on **every** step — up to 2× early on, converging to 1 as
    the state settled and the scales stopped moving (the convergence-to-1 is the signature that
    identifies rescaling rather than a state difference; a rung boundary shows a ~2.8e5 jump, since the
    Reynolds number changed too). Nothing is wrong here, but **never compare two residuals across an
    outer-iteration boundary** and never print them adjacent — the march log deliberately dropped a
    "from |R|=…" field from its inner-block title for exactly this reason.
  - **The step grid stays NARROW; only scan-down quantities get columns (binding).** The step table is
    `step, t(s), beta, in, cyc, R, a_min, rho1, flg` — fixed, 70 characters (pinned at `<= 70` by
    `test_the_grid_stays_narrow_whatever_is_switched_on`; `rho1` was added 2026-10-11 at width 4,
    `.2f`, to stay inside it, and the per-equation grid widened to match), comparable to the nested inner
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
    failed to descend** (the non-descent fallback keeps the longest rung within `fallback_growth` of
    the norm it was handed, usually the full step). The step reports the **minimum over its iterations
    with any non-descending one folded in as 0**. So `a_min=0.000` beside `alpha=1.000` is a real
    state — a full step that did not reduce `‖G‖` — not a contradiction. Since 2026-09-09 an inner
    `alpha` may also read **0.000**, the case where no finite rung came in under `fallback_growth` and
    the search declined to move at all; that iterate is reported at the norm it started from. In the
    inner table **`rate ≥ 1` identifies every non-descending iteration exactly**, since the search is
    monotone — the `alpha=0.000` case lands at `rate = 1`.
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
      which is why it is opt-in. **The rows add up to the `R` printed above them, exactly** (the steered `R`: after a `stopping_step` march arrives they add up to the `steered R` aside instead) — pinned at
      `rel=1e-12`. That requires equilibrating at the **previous** state, not the logged one:
      `newton_march` re-derives the measure at the state each outer iteration *starts* from and holds
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

  - **A cost re-fit ENDS the step; the march takes it before the next one (binding, 2026-10-10, #290).**
    `DualTimeStep(refresh_on_cycles=N)` stops its inner loop after the solve that reached `N` restart
    cycles, **keeping** that iterate, and reports `StepOutcome.refresh_due`. The march then calls
    `refresh_preconditioner(active_step, state, due=True)` before the next step, at the kept iterate and
    the next step's shift. Before, the loop fired a `jax.debug.callback` (`inner_refresh` → the session's
    `refresh_at`) and carried on inside the same step; that hook, its `since_refresh`/`spent` carry and the
    `MaterializedProblem.bind_march` "must carry `inner_refresh`" contract are deleted.
    - **Why:** the callback was an ordered host synchronization inside the compiled step on every inner
      iteration whenever `refresh_on_cycles` was set, and an on-device (traced) preconditioner cannot be
      swapped mid-step at all — its arrays are inputs to the running step. One mechanism now serves both;
      the traced field split (#290) needs exactly this shape.
    - **`due`** is keyword-only on every hook (session, `RefreshPolicy`, wrappers): `False` = offered (the
      hook gates its own cadence — the multigrid session reports `"none"`), `True` = asked (it must
      re-fit; the session reports kind **`"cost"`**, which replaces the old `"inner"`). The march passes
      `due=True` before the step after a `refresh_due` step, before a **cost redo** (`"cycles"`, so the
      redo is no longer on the same inverse — it repeated the first attempt exactly for any gated hook),
      and before an escalated redo of an attempt that itself asked.
    - **A `refresh_due` step is never a cost redo** (`RetryPolicy.retry_reason` skips `"cycles"` for it):
      redoing it would discard the iterate the re-fit exists to keep. Divergence and step-length collapse
      still redo it.
    - **Refused, not ignored:** a strategy that `StopsForRefresh` with `refresh_on_cycles` set and no
      `refresh_preconditioner` raises `ValueError` before the first step (`newton_march`) — the build-time
      refusal in `shifted_step` went with `inner_refresh`, since a builder cannot know what the march holds.
    - **What it changes in a march:** the rest of a step that used to continue after the mid-step re-fit
      becomes the next outer step — re-anchored pseudo-time, the step control adapting the shift, the
      measure rebuilt — so trajectories move and every recorded march under `refresh_on_cycles` predates
      it. Pinned by `test_refresh_on_cycles_ends_the_step_keeping_the_iterate_it_reached`,
      `test_a_step_only_asks_for_a_refit_once_a_solve_reaches_the_threshold` (`test_dual_time.py`),
      `test_a_cost_redo_asks_the_refresh_hook_for_a_refit`,
      `test_a_step_that_stopped_for_a_refit_is_kept_and_the_next_step_asks_for_it`,
      `test_a_step_that_can_stop_for_a_refit_is_refused_without_a_hook_to_take_it` (`test_newton_march.py`),
      and the two session tests in `test_preconditioner_session.py`.
  - **`refresh_preconditioner` — per-step re-fit of the step's frozen preconditioner (binding,
    forward-only; contract rewritten 2026-10-12, #290).** `newton_march(refresh_preconditioner=…)` calls
    `refresh_preconditioner(active_step, state, due=…)` in the eager loop before each `_march_step`,
    *after* the control has set β on `active_step`, and again before a redo. It returns **`None` or an
    update `step -> step`**:
    - `None` — nothing to hand back: either nothing re-fitted, or the preconditioner is a host object in
      a **static** field re-fitted in place (`MonolithicFactorShiftPolicy`), which leaves `_march_step` a
      compile-cache hit by construction.
    - an update — the preconditioner rides in the step as **data** (`TracedSplitShiftPolicy`) and the
      re-fit produced new arrays; the update is `with_traced_split(step, split)`. The march applies it to
      **both** `strategy` and `active_step` (the active step is rebuilt from `strategy` each iteration,
      so updating only it lasts one step) and reports the last on **`MarchResult.strategy`**, which
      `staged_march` carries into the next segment and the adjoint. At unchanged array shapes it is a
      compile-cache hit; see `solve-field-split.md` for what keeps the shapes fixed.
    **When `norm_builder` is set, the march builds the step's measure AGAIN from the updated step
    (2026-10-13, #701)**: an update can carry a new shift policy (`RebuildingSession`, the block-diagonal
    family, grafts the station's policy at a `rebind`), and the row-scaled measure reads its scales from
    the step's policy, so without the rebuild the step entering a new problem was judged in the previous
    one's scales. A materialized session's update carries only a re-fitted traced split, whose rebuild
    gives the same scales. Pinned by
    `test_a_step_an_update_moves_is_judged_in_the_measure_built_from_the_updated_step` (mutation-checked).
    The escalation's redo path applies an update without rebuilding the measure (a redo keeps the
    attempt's measure; only a `rebind`'s first step can carry a new policy, and that is the top path).
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
  - **The forward-step CONTRACTS live in `solve/strategy.py`, not in whichever module needed them
    first (binding, 2026-08-15).** `NewtonStrategy`, `ShiftedNewtonStrategy`, `StepOutcome`, `StepReport`,
    `StepControl`, the `StepFn` callable alias and `within_tolerance` are what travel between the Newton
    driver, the eager march, the globalization strategies, the step controls and the retry policy. None
    belongs to any one of them, and every one of them was living wherever it was first written.
    - Why they moved (a contract declared where it had no implementation forced an import cycle): `.claude/notes/solve-march-log.md`, "`solve/strategy.py` — why the forward-step contracts moved".
    - **`strategy.py` is a LEAF and must stay one.** It imports `linear`, `norm` and `relaxation`,
      none of which import it back; `implicit`, `march`, `continuation`, `retry`, `step_control`,
      `march_log` and `checkpoint` all depend on it. Adding an import here that points at any of those
      re-creates exactly the cycle it exists to remove.
    - **What did NOT move, and why.** `backtracking_line_search` stays in `implicit.py`: it is
      behaviour, not a contract, and moving it would make the contract module carry a line search.
      `MarchResult` stays in `march.py` — only `newton_march` produces it. The concrete strategies,
      triggers and controls stay with their own modules; a contract module holds contracts.
    - The rule the cycle was blocking is now `solve/step_control.py`'s `default_dual_time_control`,
      beside the controls it chooses between, and exported.
  - **`StepControl` — stateful, feedback-driven step reshaping on the eager march only (binding — the
    twin of `RelaxationSchedule`, deliberately NOT one interface).** A `RelaxationSchedule` is memoryless
    and lives on the differentiable step; a `StepControl` reads the *previous* `StepReport` (α, cost) —
    feedback available only after a step — to reshape the next step, and may raise under `jax.grad`, so it
    lives here beside `RefreshTrigger`, never on the traced path. `newton_march(step_control=…)` calls
    `next_step(base, previous, state) -> (NewtonStrategy, new_state)`, threading the control's own state; the
    march stays β-ignorant (the control returns a ready-to-run step, typically `base` with a
    `ConstantRelaxation` β leaf via `tree_at`, so `_march_step` stays a cache hit). **The control state
    survives across preconditioner refreshes (issue #156):** `newton_march` takes an incoming
    `control_state` and returns the final one on `MarchResult`, and `solve_coupled` threads it from each
    segment into the next — so a control that climbs β over many steps continues past a refresh instead of
    resetting to `beta_start` at every segment (the α-controller and the refresh were co-designed but
    could not compose before this). This is the *global*-lifetime carry, deliberately opposite the
    segment-local SER `residual_norm_0` and `drift_measure`. Unifying the two into
    one `(rn, rn0, α, state) -> (β, state)` interface was rejected: it would union SER's needs with the
    control's (dead α/state args for SER), drag α onto the differentiable core where the line search
    cannot even produce it before the step, and risk the byte-identity of the default path. Four concrete
    `StepControl`s live in `solve/step_control.py`, all three sharing one body: **`DualTimeControl`** (the
    Courant β-ramp, the **default** for a dual-time march — carries β across refreshes, see the
    DualTimeStep bullet above), **`ResidualRatioDualTimeControl`** (the opt-in residual-keyed
    alternative), and **`CflResidualDualTimeControl`** (see the bullet below). There is **no
    `AlphaTargetingControl`** — deleted 2026-08-14, see the "SER β schedule runs backwards" bullet.
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

