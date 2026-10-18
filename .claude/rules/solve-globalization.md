---
paths:
  - "aquaflux/solve/strategy.py"
  - "aquaflux/solve/continuation.py"
  - "aquaflux/solve/step_control.py"
  - "aquaflux/solve/retry.py"
  - "aquaflux/solve/relaxation.py"
  - "aquaflux/solve/line_search_growth.py"
---

# Rules — `aquaflux/solve/` globalization (Newton step, continuation, line search)

> Split out of `solve.md`; see it for the package-wide contracts and regime table, and `solve-march.md`
> for `newton_march`, `StepReport` and the homotopy. Dated measurements, test inventories and the full
> reasoning behind each entry are in `.claude/notes/solve-globalization-log.md` (the pre-prune rule is
> preserved there under "Moved verbatim from `solve-globalization.md`"); read it before re-proposing
> something that sounds tried. New dated content goes there, not here. The coupled march is
> `coupled_step` plus a preconditioner value and a session; the old `coupled_*_continuation` builders are
> gone (`turbulence.md`'s rename table).

## ⚠️ A defect that fires for the wrong reason can be load-bearing (binding)

A globalization defect can be the only thing holding the march out of a bad regime, so fixing it
correctly can break the case, one or more steps after the change. The instance: a residual-ratio control
mis-read a homotopy station change as divergence and braked; removing the brake (`rebase`) was right, and
let the shift walk into a regime where the frozen preconditioner stops tracking the operator. The fix,
`redamp`, had to reproduce the accidental brake's arithmetic (`solve-march.md`, `ResidualHomotopy` /
`redamp`; measurement in `solve-march-log.md`).

- Before deleting a damping behaviour, ask what it is holding, not only whether it is right. This applies
  to undesigned constants too (the preconditioner's `refit_beta_floor` decided silently where the frozen
  operator stopped following the real one). Do not confuse that refit floor with the march's `beta_min`
  (0.005 is stable on the flagship cases).
- A correct fix that makes the case worse is evidence about the system: supply what the defect was doing
  deliberately; do not revert.
- Measure the removal alone before combining it with a replacement.

## Globalization — Newton step, continuation, line search

### The strategy seam

- **Forward globalization is one injected `NewtonStrategy`** (`strategy.py`): `stepper()` → one outer step
  returning a `StepOutcome`; `linear_solver()`; `adjoint_preconditioner()`; `norm()`. Concrete:
  `DampedNewtonStep` (backtracking only, no shift), and the shifted `PseudoTransientStep` / `DualTimeStep`.
  The march calls the step unconditionally; there are no separate `line_search` / `preconditioner` /
  `continuation` constructor arguments on a solver. A new globalization is a new `NewtonStrategy`, never a
  branch in the march. Every shift vanishes at the fixed point, so the root and the IFT adjoint are
  strategy-independent.
- **Optional capabilities are declared protocols asked by `isinstance`, never `getattr(x, name,
  default)`** (binding, #281) — a probe answers "no" silently when a class offers the capability under
  another name. Pinned both ways in `tests/unit/test_strategy_capabilities.py`; remaining `getattr`
  sites outside this family: #669.
  - `NewtonStrategy.with_norm(norm)`: how the march swaps the per-iteration measure in (a data-leaf
    `tree_at`, so a cache hit).
  - `CarriesRelaxationSchedule` / `ReadableShift`; `ShiftedNewtonStrategy = NewtonStrategy +
    CarriesRelaxationSchedule`. **`shift_of(strategy)` is the one reader of a shift** (`None` where there
    is none).
  - `AbortsInnerLoop.with_inner_abort(*, above_cycles, below_alpha)` (`DualTimeStep`), called by
    `RetryPolicy.with_inner_abort`.
  - `StopsForRefresh` (`refresh_on_cycles`): a step that ends early for a re-fit, reporting
    `StepOutcome.refresh_due`; `newton_march` refuses one with `refresh_on_cycles` set and no
    `refresh_preconditioner` (`solve-march.md`).
  - `ShiftCarryingControl(StepControl)` declares `carry_beta`, `resumed_at`, `redamp`, `rebase`
    **together — all four or none (owner's decision)**: they are one capability, so a control offering
    some would be silently wrong across the other boundaries. Only `ShiftStrengthControl` implements them.
- **The escalation guard checks `active_step`, the step the control produced, not the step handed in**
  (`RetryPolicy.require_shifted`, first iteration). The default `SwitchedEvolutionRelaxation` exposes no
  `beta`; a `ShiftStrengthControl` installs a `ConstantRelaxation` that does, so a pre-loop check rejected
  every shipped `bfs3d` march. Pinned in both directions
  (`test_the_escalation_guard_accepts_a_shift_a_step_control_installs` /
  `..._still_rejects_a_step_that_can_never_escalate`). It tests the shift's presence, not
  `isinstance(..., ShiftedNewtonStrategy)`.
  - Gate only what is genuinely silent: the divergence retry (re-solves tighter, never touches β) and a
    step-agnostic `StepControl` are not gated. `ShiftStrengthControl.next_step` refuses a step with no
    `relaxation_schedule` itself, with a `TypeError` naming both.
  - ⚠️ **No test tier drives `bfs3d`'s own driver** (`validation.md`): a refactor can stop the case from
    starting with every gate green. Run it after touching these seams.
- **`ShiftedStep` (`continuation.py`) is the shared body of `PseudoTransientStep` and `DualTimeStep`**,
  which differ only in `stepper()`. Shared fields: `shift_policy`, `relaxation_schedule`, `line_search`,
  `step_projection`, `krylov_solver`, `residual_norm`, `adjoint_preconditioner_factory`,
  `jacobian_residual`. `DampedNewtonStep` is deliberately not a subclass (no shift). ⚠️ `DualTimeStep`
  redeclares `line_search` with default **10** against the base's **0**.
  - `adjoint_preconditioner_factory` is a **data** field (#290): a traced split (`_TracedSplitAdjoint`) or
    `solve.UnshiftedPreconditioner(policy)` rides as array leaves so a re-fit needs no recompile; a value
    with no arrays (a closure, `FrozenTransposeFactory`, `None`) is still static.
  - `ShiftedStep.preconditioned_as(other)` swaps in `other`'s `shift_policy`,
    `adjoint_preconditioner_factory` and `jacobian_residual` and keeps every setting, the shift included
    (`RebuildingSession`; `tree_at` with `is_leaf=None` so unset fields swap too).
- **`ShiftedStep.jacobian_residual`: the residual and the Jacobian may be different functions.** An
  optional `phi -> R~(phi)` whose JVP forms the shifted operator; `None` differentiates the residual and is
  byte-identical. Read by both steppers via `_shifted_solve(jacobian_fn=)`. `R` decides the answer, `J`
  only the rate (consumer: the gradient reconstruction's sweep count, `schemes.md`).
  - The adjoint stays exact by construction: `root_adjoint` forms `jax.vjp(residual_fn)` at the root and
    never consults the step (`test_a_stand_in_jacobian_leaves_the_adjoint_exact`). The descent test still
    reads the true residual. It is a data field (a bound assembler method is a pytree).
  - A reachability test is mandatory (`test_a_stand_in_jacobian_genuinely_changes_the_step_it_produces`):
    an unwired field leaves everything bit-identical.
  - ⚠️ **A Jacobian scaled by 0.5 is exactly Newton's stability boundary** (fixed-point derivative
    `1 − 1/s = −1`: oscillates forever once the shift eases). Use 0.8 for a "wrong operator" fixture.

### The pseudo-transient engine (`continuation.py`)

- **`PseudoTransientStep` is residual-agnostic**: an injected `RelaxationSchedule` for β, the shifted
  solve `(J + diag(β d)) δ = −R` (`solve_linear(throw=False)`), and the accept/escalate `while_loop`.
  Physics comes from a **`ShiftPolicy`** (`shift_term(φ, residual=None) -> ShiftTerm`) and acceptance
  from a **`StepAcceptance`**; the escalation mechanics (grow β, cap at `max_escalations`, carry the best
  candidate) stay in the engine. Default acceptance `DivergenceGuard(divergence_cap=10.0)`: accept unless
  non-finite or above `divergence_cap·‖R₀‖` (the march is non-monotone, so not a descent test).
  Applications: `flow/continuation.py`'s `MomentumShiftPolicy` via the `momentum_continuation` factory
  (no wrapper class), and `turbulence/continuation.py`'s `ScalarShiftPolicy` via
  `scalar_pseudo_transient_solve` (the only scalar path the SST driver supports). A new nonlinear residual
  needing pseudo-time writes a `ShiftPolicy`; it never re-implements the march.
- **`stepper()` returns a `StepOutcome`** (`phi_next`, its `residual_norm`, `cycles`, `alpha`, …;
  `solve-march.md`). The count is the accepted attempt's; **a step whose every attempt was rejected
  reports `0` cycles — treat it as no measurement, not as free.** A fully rejected step returns `phi` and
  its own measure.
- **The β schedule is an injected `RelaxationSchedule` (`relaxation.py`); do not re-inline it.** Its
  method is `shift_strength(‖R‖, ‖R₀‖)` (β damps more as it grows; the opposite of an under-relaxation
  factor — #373). Default `SwitchedEvolutionRelaxation(beta0=2, exponent=1, beta_floor=0)`. A schedule is
  memoryless, which keeps it traceable; `ConstantRelaxation(β)` holds β as a dynamic 0-d leaf so a control
  can vary it per step as a cache hit. A stateful or α-driven rule is a `StepControl`, not a schedule.

### `Globalization`: one settings object for every builder (binding, #372)

- `Globalization` (`continuation.py`, exported) carries every setting that says *how a march damps*:
  `beta0`, `exponent`, `beta_floor`, `max_escalations`, `escalation_factor`, `divergence_cap`,
  `line_search`, `line_search_growth`. `step(policy, **fields)` and `dual_time_step(policy, **fields)` are
  the only places a globalization becomes a step; per-problem fields (`adjoint_preconditioner_factory`,
  `krylov_solver`, `residual_norm`, `step_projection`, `jacobian_residual`) pass through.
- **Every field is unset (`None`) and falls through** to the builder's base, then to the step class's
  default. `_coupled_step` uses `globalization.with_defaults(line_search=_COUPLED_LINE_SEARCH)` (10), so
  `Globalization(beta0=1.5)` keeps the coupled line search and an explicit `line_search=0` survives.
  `DEFAULT_GLOBALIZATION` is the empty override; there are no presets (a preset-based first version
  silently reset the coupled line search).
- **Refusals**: `dual_time_step` raises on any set ladder field (`max_escalations`, `escalation_factor`,
  `divergence_cap`, `line_search_growth`) — the inner loop has no ladder. A step field the target class
  does not declare raises even as `None` (`_supplied` checks names before dropping unset values); a
  setting given both on the object and as a field raises.
- **Membership test: can the setting's reason be stated without naming a residual or a preconditioner?**
  The closure's step projection (`omega_step_ratio`, built by `_closure_step_projection`) fails it and
  stays on the coupled builders. `shift_basis`, the velocity shift parts and `turbulence_damping` pass it
  but describe the shift, so they are `turbulence.CoupledShiftSettings` (#387). Open: `divergence_cap`
  and `line_search_growth.basin` are residual ratios in a measure the object excludes (#370); a non-SER
  schedule or non-`DivergenceGuard` acceptance reaches a step only via `step(**fields)`.
- The flow-only and scalar builders still cannot reach the problem-specific step fields. The segregated
  `bulk_velocity_flow_solve` is a `DampedNewtonStep` and takes no `Globalization`. March keywords are
  bound against `coupled_step`'s signature (`_march_keywords`), so an unknown name is a `TypeError`.
  Pinned by `tests/unit/test_globalization_reach.py`.

### `ShiftStrengthControl`: boundary responses and the floor release (`step_control.py`)

Carried state: `ShiftControlState(beta, memo, release)`. Each boundary response keeps what still applies:

| method | called when | keeps | drops |
|---|---|---|---|
| `carry_beta` | a β-escalation retry | the memo | — (β replaced) |
| `rebase` | a `ResidualHomotopy` station change, one step later | β | the ratio reference |
| `redamp` | that station change, on the entering step | the memo | — (β scaled) |

All three keep `release`; only `resumed_at` starts a state without one.

- ⚠️ **`rebase`'s contract is "drop the ratio reference, keep the rest", not "clear the memo".** The base
  clears the whole memo only because its memo is the reference; a subclass with a richer memo must
  override it, or the ratio silently defaults to 1.0 and the brake never fires. Call these methods
  rather than unpacking the state at a call site.
- **`redamp` is not optional**: without it the shift descends by `grow ** steps_per_station` unopposed
  and walks through the level the next station can carry (`solve-march.md`).
- **`release_floor` (opt-in, `None` by default; both flagship cases ship 1e-4 with `release_backoff`
  0.5).** At a fixed shift a dual-time step contracts each mode by `β/(λ+β)`, so a march parked at
  `beta_min` converges linearly at a rate the floor sets. `_floored`, applied after every `_adapt`, is the
  one place that decides: released iff the previous step `arrived`, the rule did not raise β, and either
  that step was `settled(report)` (ran at or below `beta_min` with `alpha ≥ settle_alpha`, default 1.0)
  or it already ran below `beta_min`. Anything else puts `beta_min` back in one step. `_clamp`'s lower
  bound is `release_floor` when set.
  - **`settled(report)` is the one definition of settled**, read by the release and by
    `RampSchedule(end_when=control.settled)` (`solve-march.md`). It ignores `arrived`. There is no
    `release_alpha`.
  - The gate is measured: a zero-shift step from the state the ramp arrived at raised the residual; after
    one full-length step at the floor it held. The shift a state can take belongs to the state.
  - `release_floor` must be positive (a retry escalates β by scaling it).
  - **`release_backoff`**: a failed release records `bar = release_backoff × residual it fired from` in
    `ReleaseMemory`; the floor is released again only from a settled step at or under it. A release that
    holds sets no bar (bit-identical). It does not make a deep ramp start competitive: the residual a
    release needs belongs to the state the ramp leaves. ⚠️ The bar compares residuals across steps in a
    measure rebuilt every step — adequate for a factor-2 bar, not a fine one. `rho1` does not predict
    which release holds.
  - ⚠️ The release's pitzDaily/bfs3d step and cycle counts were taken under different refresh cadences
    and stops; see the log ("The shift floor sets the target station's rate", "bfs3d: the release…",
    "The release backoff") before quoting one. pitzDaily's pairing with `grow` 3 transferred to bfs3d only
    with the residual inner stop. Pinned by `test_step_control.py` and
    `test_newton_march.py::test_a_released_shift_turns_a_linear_tail_into_newton_steps`.

### The shift's spatial basis and parts (`shift_basis.py`, `ShiftTerm`)

- **`ShiftBasis` sets the per-cell base diagonal `d`** from the operator's buckets:
  `local_diagonal(convective, dissipative)`. The one concrete is `LocalCourantBasis(dissipative_weight=w)`,
  `d = convective + w·dissipative`. `w = 1` (default) is `d = a_P`: spatially uniform under-relaxation
  `1/(1+β)` — **on the momentum rows this is exactly a segregated solver's implicit under-relaxation at
  `α_u = 1/(1+β)`** (both sides match), so the momentum treatment is standard and not where a cold-start
  gap hides. `w = 0` is a local convective (Courant) time step. Buckets come from
  `rhie_chow.momentum_diagonal_parts` and `scalar_transport_shift_diagonal_parts`. With a non-`a_P` basis
  the shifted diagonal is `a_P + β d`, and `make_preconditioner` must invert that. Pressure keeps zero
  shift. A new basis is a new `ShiftBasis`; do not branch the policies.
- **The velocity buckets' source is injected (`VelocityShiftParts`)**, because the shift (current local
  time scale) and the frozen preconditioner have different lifetimes. The protocol lives in
  `shift_basis.py`; `FrozenViscosityVelocityParts` (default) in `flow/continuation.py`;
  `LiveViscosityVelocityParts` (re-forms the closure, maps solved scalars back to physical) in
  `turbulence/coupled.py`. `parts(flow, k_solved=None, omega_solved=None)`. `a_p` still comes from the
  preconditioner, since that is what the velocity block inverts. It reaches every coupled builder.
  ⚠️ A `Protocol` is not runtime-checked: write the protocol's defaults on every implementation, even one
  that then rejects the call.
- **`ShiftTerm` has three parts**: `diagonal`, `make_preconditioner(β)`, optional `row_shift_scale(β)`.
  `ShiftTerm.shift(β) = β · row_shift_scale(β) · diagonal` is the one place they combine, and every
  consumer (including a preconditioner for the same operator) goes through it.
  - ⚠️ **Never fold a block factor into `diagonal`**: β arrives clamped at `beta_min`, so the factor
    multiplies the floor forever; and `diagonal` is the row scale of the row-equilibrated measure
    (`coupled_scaled_norm`), so it would move the bar the march is judged by (`turbulence.md`,
    `turbulence_damping`).
  - ⚠️ **A wrapper policy that rebuilds a `ShiftTerm` from `.diagonal` alone silently drops the rest.**
    Forward every part (`MonolithicFactorShiftPolicy` and `flow.BorderedShiftPolicy` once did not); find
    wrappers structurally (AST), not by grepping the callee.
  - ⚠️ **`residual` is optional, so omitting it does not fail**: a residual-dependent policy silently
    falls back to a constant. Both steps pass it (`DualTimeStep` once did not).

### The line search

- **`line_search` backtracks the shifted step before escalating β** (`backtracking_line_search`, shared
  with `DampedNewtonStep`): rungs `{1, 1/2, …, 1/2**line_search}`, a `lax.while_loop` stopping at the first
  reducing rung (one residual evaluation when the full step descends). Forward-only: it runs on
  `stop_gradient` inputs and must never be called on a differentiated path. Why: from a hybrid IC the full
  coupled step overshoots by orders of magnitude, and escalating β (a full re-solve) did not descend; a
  line search on one solve does. The coupled path uses 10 (`_COUPLED_LINE_SEARCH`); flow and scalar
  default to 0. There is no `grow` rung above the full step (it had reached nothing since a `max_alpha =
  1.0` default capped it; deleted).
- **The line search returns its residual vector (`LineSearchStep.residual`), and the dual-time step runs
  on it.** At `alpha = 0` the iterate is the start point, so the vector is the caller's own
  `backtracking_line_search(..., residual=R(phi))` (`None` if none was passed; all three strategies pass
  it); with no finite rung it is a NaN seed. `DualTimeStep` carries `G` as the next inner right-hand side
  and reports `R(phi_next) = G − transient_term(phi_next)`. Not bit-identical to re-evaluating `R`
  (rounds at the size of `β d (phi − φⁿ)`, zero at the root).
- **A step is handed `R(phi)` and must not evaluate it (`StepFn` = `(residual_fn, phi, residual,
  residual_norm_0, solver)`, #649).** The march carries each step's `StepOutcome.residual` into the next
  (`solve-march.md`); `DampedNewtonStep` passes it to `newton_correction(residual=)`,
  `PseudoTransientStep` uses it as the right-hand side and the policy's residual (and returns it from a
  fully rejected step), `DualTimeStep` as its anchor's `R(φⁿ)`. A caller driving `stepper()` directly
  (tests, `validation/` probes) passes `residual_fn(phi)`. Pinned by
  `test_the_dual_time_step_evaluates_the_residual_once_per_inner_iteration` (`n`, not `1 + n`) and the
  trace counts in `test_newton_march.py` (two residual copies per compiled step on every strategy).

### The linear solve and the residual measure

- **`RetryPolicy.solver` is a settings value, `LinearSolverSpec`** (`linear_solver_spec.py`, #437):
  `GmresSolve(rtol, restart, stagnation_iters, max_restarts)` builds `relative_residual_gmres(...)`;
  `DirectSolve()` builds `lx.AutoLinearSolver(well_posed=True)`. `newton_march` builds it once per march
  (`tight_solver`). Only this field moved: `RootSolveSettings.linear_solver` / `.adjoint_solver` and
  `solve_coupled(adjoint_solver=)` still take `lineax` solvers (the case layer builds them from specs).
- **`krylov_solver` overrides `_INEXACT_CONTINUATION_SOLVER`** (restart-40 GMRES); the coupled path uses
  restart 120 and stops in the march's own row-scaled measure at `rtol = 0.3` (`LinearSolveSettings`,
  `solve.md`'s regime table). **Why a relative stop (`relative_residual_gmres`)**: `lineax`'s stock stop
  is componentwise, so near-wall ω wall-fixation rows with ~0 right-hand side collapse onto `atol` and
  hold the solve ~9 orders past the asked tolerance. ⚠️ The stopped residual is the **true** residual
  (right preconditioning: `b − A M y = b − A x`); no "solution accuracy ≈ rtol" follows from it.
- **The residual measure is an injected `ResidualNorm`, owned by the strategy** (`solve/norm.py`): the
  outer stop, the SER ratio, the line search (`norm=`) and the `DivergenceGuard` all read the same
  measure. Default `jnp.linalg.norm`. Why a scaled measure exists: on `[flow, k, ω]` the Euclidean norm
  is ~100 % ω, so the line search can neither see nor protect k (k collapsed and the march froze).
  `BlockScaledNorm` / `RowScaledNorm` judge each block on its own scale. The adjoint never forms a
  measure, so the measure is forward-only.
- **`norm_builder`: the measure is rebuilt every outer iteration and held fixed within one (binding).**
  `newton_march(norm_builder=)` takes `(step, state) -> ResidualNorm` (a solve builds it from its
  `Convergence` measure via `ResidualMeasure._builder`), swaps it on with `eqx.tree_at`, and re-measures
  `residual_norm_0`. Rebuilding per trial step would let a candidate win by shrinking its own denominator.
  `residual_norm` is therefore a **data** field, so the swap is a cache hit.
  - ⚠️ **The step's linear solve must read the swapped measure**: `relative_residual_gmres(norm=None)`
    means "the measure of the step that runs me", and all three steppers call
    `solve.linear.in_progress_measure(solver, self.residual_norm)` before solving; a `norm=None` solver
    run outside a step raises. (Before #370 the inner stop used the build-time measure on both flagship
    cases.) Pinned by `test_each_step_stops_a_measure_following_linear_solve_in_its_own_measure`.
  - Whether `RootSolver` accepts a rebuilt `RowScaledNorm` is untested.
- **A `ShiftPolicy`'s preconditioner is data — the #105 "not a pytree" rule is reversed (#386).**
  `ScalarTransportPreconditioner` and its strategies are `equinox.Module`s with hierarchies as leaves, and
  the block policy's adjoint factory is `solve.UnshiftedPreconditioner(policy)`. Why: a viscosity ramp
  re-points the session every station, and static preconditioner objects recompiled the coupled solve
  each time. A rebuild with unchanged coarse shapes is a cache hit; one whose shapes move is a miss, which
  is why `RebuildingSession._repointed` derives the new step from the held one. The old
  `UnexpectedTracerError` hazard is gone because the march runs on `stop_array_gradients` copies. Pinned by
  `test_a_re_pointed_block_session_step_is_a_compilation_cache_hit` and
  `test_the_scalar_solve_compiles_once_whether_its_preconditioner_is_carried_or_rebuilt`.
