---
paths:
  - "aquaflux/solve/**"
---

# Rules — `aquaflux/solve/` (Newton + implicitly-differentiated linear solve)

> **Provenance boundary (binding).** This file cites the C++/Fortran precursors to inform
> *your* understanding — that is its job, and why it loads into your
> context. Per the root `CLAUDE.md` **Comment Convention**, none of that provenance may
> reach the shipped surface (`.py` comments/docstrings, `docs/`): cite the *math*, never the
> reference code, the `.claude/` rules, the design notes, or the author's own papers.

Drive the residual to zero and expose an exact, iteration-count-independent adjoint.
Governed by the root `CLAUDE.md` Engineering Principles.

## Responsibility
- A Newton driver on `R(state, params) = 0` using the AD Jacobian (JVP/VJP), and a
  linear solve wrapped so its gradient comes from **implicit differentiation**, not by
  unrolling Krylov iterations onto the tape.

> **`solve/__init__.py` is the API boundary (binding, #48).** Everything consumable from this
> package is re-exported there, and **library code imports `from aquaflux.solve import …`, never
> `from aquaflux.solve.<submodule> import …`**. A name absent from `__all__` is internal (reach for it
> only from that submodule's own unit tests, which are exempt). When you add a public entry point,
> export it in the *same* change — a partial surface is what pushes consumers into deep imports and
> makes `__init__` stop describing the package (the block preconditioner once pulled nine names
> straight out of `solve.multigrid` while `__all__` advertised only the smoothed-aggregation third of
> the AMG toolkit). `tests/unit/test_solve_api.py` pins both halves and fails with the offending
> file named, so this cannot erode silently.
- Milestone 0: a single scalar diffusion system; the plumbing must generalize to the
  coupled p–U block later without redesign.

## Status — BUILT (Stage A, linear)
- **`linear.py` — BUILT.** `solve_linear(matvec, b, solver, preconditioner=None)` is a
  matrix-free wrapper over `lineax` (default restarted GMRES); `lineax` supplies the
  **implicit-diff of the linear solve** (the Krylov loop is not taped). This is the load-bearing
  adjoint primitive. The optional **left preconditioner** `M` (a matvec ≈ `A⁻¹`) hands the solver
  `M∘A` and `Mb`; since the caller `stop_gradient`s `M`'s coefficients, it changes only Krylov
  convergence, not the solution or its gradient — **verified transparent** in
  `test_preconditioning.py` (solution and gradient identical with/without `M`). This is the seam
  the **outer block preconditioner** (below) attaches to.
  - **`solve_linear` returns `(x, cycles)` — there is ONE linear-solve entry point, not a counted/
    uncounted pair (binding, do not re-split).** A caller that only wants the answer writes
    `x, _ = solve_linear(...)`. A `solve_linear_counted` sibling existed briefly and was **deleted**: it
    held the real body while `solve_linear` forwarded to it and dropped the count, i.e. the old shape
    preserved across a refactor — the delegating-wrapper form the pre-release no-shims policy bans. It
    also duplicated the whole signature and `Parameters` block (its docstring had already degenerated to
    "the arguments mean exactly what they do there", which cannot stand alone), and it was *dominated*:
    it did `solve_linear`'s job plus more. The blast radius of collapsing was ~9 lines — the function has
    exactly **two** library call sites (`newton.py`'s `newton_correction`, `implicit.py`'s adjoint
    transpose solve); the "it has too many callers to change" intuition is false, so do not resurrect the
    pair on that argument. **The count is restart CYCLES, not matvecs** (each cycle is up to `restart`
    matvecs — the standing misreading of `lineax`'s `num_steps`), pinned to `int32` so a caller can carry
    it through a `lax.while_loop` (whose carry structure must be invariant — exactly one call site does,
    the pseudo-transient escalation loop), and `0` for a solver that reports none (a direct
    factorization). **Why the count:** a frozen preconditioner going stale shows up first as a *rising
    cycle count on an otherwise-unchanged system*, before the residual history shows anything — so it is
    the honest trigger for re-freezing the preconditioner mid-march, and a robust one. Wall-clock time is
    the tempting proxy and a bad one: it moves with machine load or a suspended process while the linear
    algebra has not changed at all. Pinned in `test_preconditioning.py`, including the load-bearing
    behavioural check that **a better preconditioner strictly lowers the count** (otherwise it measures
    nothing).
- **`newton.py` — BUILT: `newton_step` / `newton_correction`, one correction each. There is NO
  `NewtonSolver` class — it was deleted (binding, #102).** Each forms `J` matrix-free via `jax.jvp`
  and calls `solve_linear`; no hand-derived Jacobian.
  - **Why it went.** The class was `newton_step` plus a **fixed-count, unchecked loop**, which is
    redundant at `iterations=1` (19 of its 28 call sites — a *linear* residual, where one correction
    is exact) and **forbidden** above it (a fixed count cannot tell convergence from exhaustion, and
    taping the unrolled steps is the gradient path the two-level implicit differentiation exists to
    avoid). Its one production use, `laplace_field`, was `iterations=1` — not Newton at all, just an
    exact linear solve. The turbulence scalars had already migrated off it for exactly these reasons
    (see `turbulence/continuation.py`).
  - **The split to hold to.** *Linear* residual → `newton_step`, exact in one call. *Nonlinear*
    residual → `ImplicitNewtonSolver` (converges, globalizes, IFT adjoint). Do **not** reintroduce a
    fixed-count loop over `newton_step` in library code. A *test* may write one inline when the point
    is to show unglobalized Newton is insufficient (`test_scalar_continuation.py`) or to isolate a
    preconditioner (`test_turbulent_channel.py`) — that is 2 lines and self-documenting, not a class.
  - **`newton_step` is the only path that differentiates in FORWARD mode.** `ImplicitNewtonSolver` is
    a `jax.custom_vjp`, which registers only the reverse rule, so `jacfwd`/`jvp` through it raises
    `TypeError` — a JAX API consequence, not a mathematical one (the IFT gives the tangent just as
    readily; a `custom_jvp` would serve both, at the cost of the separate tight `adjoint_solver` the
    current design deliberately controls). `newton_step` is plain traced operations, so both modes
    work. This matters: the plane-wall sensitivity gate takes `jacfwd` through the whole transient
    march — one linear solve per input, the efficient direction for a scalar parameter against a
    whole field. Pinned in `tests/unit/test_newton.py`.
- **Neither function jits internally — the caller owns the jit boundary**, matching
  `ImplicitNewtonSolver`. Wrap calls in `eqx.filter_jit`; un-jitted, every operation dispatches
  eagerly. For a caller that re-solves in a loop, pass the assembler as an `equinox.Module`
  **argument** to the jitted function (`reused_flow_solve`'s pattern) so its arrays are dynamic
  leaves and the compiled solve is a cache hit; a bare captured closure is hashed by identity and
  misses every time.
- **`implicit.py` — BUILT (`ImplicitNewtonSolver`).** The nonlinear counterpart: Newton to
  convergence (`lax.while_loop`, data-dependent stop) with a reverse-mode **IFT adjoint** via
  `custom_vjp` — one transpose linear solve at the converged state, `dphi*/dtheta =
  -(dR/dphi)^{-1}(dR/dtheta)`, no Newton loop taped. `solve(residual_fn, phi0, theta)` takes the
  differentiable params `theta` explicit so the adjoint returns their cotangents. Reverse-mode
  only (`jax.grad`), which is what a scalar objective through the solver needs. This is the
  "IFT on the converged Newton state" half of the two-level scheme; it activates with the first
  nonlinear residual (the flux limiter). `newton_step` is shared with `NewtonSolver`. Verified
  (`test_implicit_solve.py`): converges a nonlinear root, gradient matches the closed form to
  1e-10, and is iteration-count-independent. Used by the limited-advection solve.
  - **Convergence guard (binding — the IFT adjoint is only valid at a root).** `_forward` carries the
    terminal residual norm out of the `while_loop` and wraps the returned field in `eqx.error_if`: if
    the residual is non-finite or above `atol + rtol·‖R₀‖` (exhausted `max_steps`, or a `NaN`/`Inf`
    that used to make `residual_norm > tol` short-circuit to `False` and exit with a poisoned field),
    it **raises `eqx.EquinoxRuntimeError`** instead of returning. The guard sits in `_forward`, so it
    fires for both the forward value and the `jax.grad` path (the fwd pass saves the guarded field),
    closing the silent-wrong-gradient hole where the transpose solve at a non-root stays well-posed
    and raises no `NaN`. The stopping test is one helper, `_within_tolerance`, shared by the loop
    `cond` and the guard. A `NaN` mid-iteration is often caught first by `lineax`'s own non-finite
    guard at the next linear solve — both are hard errors, neither is silent.
- **Forward globalization is ONE injected strategy — `forward_step: ForwardStep`.** The forward
  Newton loop has a single point of variation: `ImplicitNewtonSolver` takes one `forward_step`
  implementing the `ForwardStep` protocol (`stepper()` → the per-step
  `(residual_fn, phi, ‖R₀‖, solver) -> phi_next`; `default_solver()` → the inexact-Newton forward
  GMRES for that march; `adjoint_preconditioner()` → the converged-state transpose preconditioner).
  Two concrete strategies: **`DampedNewtonStep`** (default — the backtracking line search, holding
  the forward/adjoint preconditioner and the line-search count) and **`PseudoTransientStep`**
  (`aquaflux/solve/`, the residual-agnostic diagonally-shifted march; the flow configures it via
  `aquaflux/flow/`'s `momentum_continuation` factory — no wrapper class). `_forward` calls the
  injected step unconditionally — there is **no `if continuation is None` branch**, and **no separate
  `line_search`/`preconditioner`/`continuation` constructor args** (they were unified here; do not
  reintroduce them). Each strategy's shift vanishes at the fixed point, so the converged state and
  the IFT adjoint are strategy-independent. When adding a globalization (e.g. a monotone/forcing
  acceptance), add a `ForwardStep` — do **not** grow a branch in `_forward`.
- **`continuation.py` — BUILT (`PseudoTransientStep`, residual-agnostic).** The pseudo-transient
  continuation engine lives **here in `solve/`, not in `flow/`** — it is a `ForwardStep`
  (`stepper`/`default_solver`/`adjoint_preconditioner`) that runs an **injected**
  `RelaxationSchedule` for the shift strength β, the diagonally-shifted solve `(J + diag(βd))δ = −R`
  (`solve_linear(throw=False)`), and the closed-loop accept/escalate `while_loop`. **`stepper()`
  returns `(φ_next, cycles, alpha)`** — the accepted attempt's cycle count *and* its line-search
  factor α (the step-quality signal, ≤1; `_forward` drops both off the `custom_vjp` primal, a march
  reads them).
  - **The β schedule is an injected `RelaxationSchedule` (`solve/relaxation.py`), SER extracted as the
    default (binding — do not re-inline the β rule).** The old `beta0`/`exponent`/`beta_floor` fields on
    `PseudoTransientStep` are gone; the field is `relaxation_schedule: RelaxationSchedule`, defaulting to
    `SwitchedEvolutionRelaxation(beta0=2, exponent=1, beta_floor=0)` — byte-identical to the old inline
    `max(beta_floor, β₀(‖R‖/‖R₀‖)^p)`. It is the direct twin of the injected `ResidualNorm`
    (`solve/norm.py`). A `RelaxationSchedule` is **memoryless** (β from the two residual norms only), which
    is what keeps it on the differentiable traced path. `ConstantRelaxation(β)` carries β as a **dynamic
    0-d leaf** so an external control can vary it per step as a `filter_jit` cache hit (the `lam_max`
    precedent). The five builder factories (`momentum_continuation`, `coupled_continuation`,
    `mass_flow_coupled_continuation`, the two scalar builders) keep their public `beta0=/exponent=` knobs
    and translate them into `SwitchedEvolutionRelaxation(...)` at the one construction line — a factory
    building the real object, not a shim. A *stateful* or α-driven damping rule is **not** a schedule; it
    is a `StepControl` on the eager march (see `march.py`).
  - **Two injected seams**, both `Protocol`s: the physics comes from a **`ShiftPolicy`**
  (`shift_term(φ) -> ShiftTerm(diagonal, make_preconditioner)`; `ShiftTerm.diagonal` is the full-state
  base shift, `make_preconditioner(β)` the frozen shifted `M`), and the per-attempt accept/reject
  decision from a **`StepAcceptance`** (`accept(candidate_norm, residual_norm, residual_norm_0,
  attempt) -> bool`). The escalation-loop *mechanics* (grow `β`, cap at `max_escalations`, carry the
  best candidate) stay in the engine; only the decision is delegated. Default acceptance is
  **`DivergenceGuard(divergence_cap=10.0)`** — accept unless the candidate is non-finite or exceeds
  `divergence_cap·‖R₀‖` (a divergence guard, not a descent test, since the march is non-monotone); a
  monotone / forcing rule is a drop-in `StepAcceptance` — do **not** hardwire an acceptance test into
  the `while_loop`. So the engine is
  reusable for **any** nonlinear residual (reaction/energy/turbulence), not just the coupled flow —
  verified in `tests/unit/test_pseudo_transient.py`, which drives it on a scalar root with a trivial
  policy (no mesh, no flow). The flow application is `aquaflux/flow/continuation.py`'s
  `MomentumShiftPolicy` (velocity-block `a_P` shift + shifted SIMPLE preconditioner), configured into a
  `PseudoTransientStep` by the `momentum_continuation(assembler, …)` **factory** (which builds the
  block preconditioner and injects the `DivergenceGuard` + adjoint factory) — **no wrapper/adapter
  class**, since `PseudoTransientStep` is itself the `ForwardStep`. The scalar application is
  `aquaflux/turbulence/continuation.py`'s `ScalarShiftPolicy` (the transport operator diagonal — the
  scalar `a_P` analogue from `scalar_transport_shift_diagonal` — as the base shift, with the frozen
  scalar-transport AMG reused **unshifted** as `M`, since the shift only adds positive diagonal),
  globalizing the stiff k/omega solves via `scalar_pseudo_transient_solve` — the **only** scalar path
  the SST driver supports (the fixed-count Newton sub-solve was removed). When a new nonlinear residual
  needs pseudo-time globalization, write a `ShiftPolicy` — do **not** re-implement the march.
  - **`stepper()` returns `(phi_next, cycles)` — ONE step method on the whole `ForwardStep` protocol,
    counted/uncounted pair deleted (binding).** Every strategy reports its step's restart-cycle count
    (`DampedNewtonStep` gets it from `newton_correction`, which now returns `(delta, r, cycles)`); a
    consumer with no use for it drops it (`phi, _ = step(…)`). A `counted_stepper()` sibling existed
    briefly, with `stepper()` forwarding to it and dropping the count — deleted for the same reason as
    `solve_linear_counted` above, and note it had **no production consumer at all** while it existed.
    The reported count is the **accepted** attempt's, not the sum over rejected escalation attempts —
    the cost of the step actually taken. **A step whose every attempt was rejected reports `0`**
    (`best_cycles` is only written on acceptance): a consumer must treat `0` as *no measurement*, not
    as *free*, or a rejected step reads as the cheapest in the march. Consumed by `forward_march`
    (below); dropped by `_forward`.
  - **The count is NOT carried out of `_forward`'s `while_loop` (binding).** Two reasons, both concrete:
    it would put an `int32` in the primal output of `_implicit_solve`'s `custom_vjp`, so the reverse rule
    would have to handle a `float0` cotangent leaf in the most correctness-critical function in the
    package, for a number the differentiated path can never use; and it would force the *generic* Newton
    loop to pick which step's count survives (last / max / sum), which is a reporting policy the solver
    has no business owning. Per-step cost is observed eagerly instead, by `forward_march`.
  - **`line_search` — backtrack the shifted step before escalating β (binding, the coupled-RANS fix).**
    The step optionally scales the shifted correction `δ` back along `{1, 1/2, …, 1/2**line_search}`
    (`backtracking_line_search`, extracted from `implicit.py` and shared with `DampedNewtonStep` — one
    home for the ladder) and keeps the largest length that reduces the residual, **before** the
    accept/escalate test. The ladder is a **`lax.while_loop` that stops at the first (largest) reducing
    rung** — a full step that already descends (the common case near the root) costs one residual
    evaluation, not `line_search+1`, and the loop body compiles once instead of unrolling `line_search+1`
    residual copies into the graph. It is safe as a non-differentiable `while_loop` because the search is
    **forward-only**: it runs inside `ImplicitNewtonSolver`'s `custom_vjp` forward pass, whose reverse
    rule is the IFT transpose solve at the root and never differentiates the iteration (every caller is a
    `ForwardStep`; nothing differentiates through it — audited). Do **not** call it on a differentiated
    path. `line_search=0` (default) is the old behaviour: take the full step `φ+δ`, and
    the **only** recourse to an overshoot is escalating β — a *full re-solve*. This was measured to be
    the dominant coupled-RANS cost: from the hybrid IC the full coupled Newton step overshoots by
    ~10⁷× (‖R‖ 220 → 5.8e9), so every step burned ~4–7 expensive re-solves and, worse, escalating β
    (β=16/64) still did **not** descend (rel ≈ 1.0 — the march stalled). A line search on the **one**
    β₀ solve finds α≈¼ → rel≈0.48 (residual halved) in a few cheap residual evaluations. So the
    coupled path sets `line_search>0` (`coupled_continuation`, `_COUPLED_LINE_SEARCH=10`); β escalation
    stays the fallback for a genuinely bad *direction* (an ill-conditioned shifted solve), not an
    overshoot. Like the shift, the search only reshapes the forward path — converged state and IFT
    adjoint unchanged. The flow path leaves `line_search=0`, so it is bit-identical.
  - **`forward_solver` overrides the shared `_INEXACT_CONTINUATION_SOLVER`.** `default_solver()` returns
    the injected `forward_solver` when set, else the shared restart-40 GMRES. The coupled path injects a
    larger-restart GMRES (`_COUPLED_FORWARD_SOLVER`, restart 120): the stiff coupled saddle system needs
    hundreds of restart-40 cycles (a 40-vector subspace discards too much Arnoldi history), whereas a
    120-vector subspace reaches the same tight solution ~1.4× faster and tighter. Tolerances stay tight
    — an *inexact* linear solve is unsafe under log-ω (an inaccurate step in the log variable is
    exponentiated and diverges), so the accuracy is load-bearing, not wasteful.
  - **THE SER β SCHEDULE RUNS BACKWARDS FOR STIFF COUPLED RANS (measured, pitzDaily — the dominant
    cost, and it is the globalization, not the preconditioner).** The switched-evolution-relaxation
    schedule `β = β₀(‖R‖/‖R₀‖)^p` *lowers* β as the residual falls, on the premise that a smaller shift
    means a more Newton-like, more productive step near the root. **On this problem the premise is false:
    the efficiency-optimal β *rises* as ‖R‖ falls, so SER drives β the wrong way and the coupled march
    grinds instead of entering the quadratic basin.** Two independent measurements on E1's checkpoints
    (`solve_coupled`, twolevel, corrected hybrid IC; each re-solving one frozen step across fixed β, PC
    rebuilt at that state):
    - **Efficiency (residual reduction per second):** optimum β ≈ 2 at rel 0.38, **≥ 5 at rel 0.05**,
      while SER's β *fell* 0.76 → 0.10. At the developed state SER's β is ~50× below the optimum, a **~190×**
      step-efficiency gap (0.003 vs 0.56 %/s).
    - **The mechanism is line-search CLIPPING, seen directly via the step-length factor α.** α (the
      fraction of the shifted step the backtracking search keeps) is a clean monotone signal: it rises
      with β and hits **α = 1 exactly at the efficiency-optimal β** — the point where the full damped step
      *just stops overshooting*. Below it the step overshoots and is clipped to near-nothing; at it the
      step is full and productive.

      | β | α @ rel 0.38 | α @ rel 0.05 |
      |---|---|---|
      | 0.10 (≈SER in the tail) | 0.016 | **0.031** |
      | 1.0 | 0.50 | 0.25 |
      | 2.0 | **1.00** | 0.50 |
      | 5.0 | 1.00 | **1.00** |

      **SER operates at α ≈ 0.03 in the tail:** the full Newton step overshoots by ~33× (at β=0.05, ~80×),
      and the line search salvages a ~0.4% crawl from it. *That* is the grind — not near-convergence, not
      preconditioner cost. The α = 1 boundary is the controller target (raise β until the full step is
      marginally accepted); α is far less noisy than the per-step residual reduction ρ (which swings
      37%↔6% at fixed β and wrecked a first, ρ-driven controller that ratcheted β into a runaway).
    - **Caveat — β-schedule and PC-refresh are COUPLED; the optimal-β numbers above use a PC rebuilt at
      each state.** In a real march the preconditioner is frozen at the cold IC, and a bolder β moves the
      state faster, staling that frozen PC faster (the ρ-controller runaway hit 119 cycles at β=10.4 — high
      β should be *cheaper*, so that was PC staleness, not the shift). So an α-targeting β schedule and the
      scalar-AMG refresh (below) must be co-designed, not tuned in isolation. A **β-independent staleness
      indicator** — the drift of the frozen operator's coefficients, `‖Δν_t‖`/`‖Δṁ‖` relative to the
      freeze state — is the clean refresh trigger this motivates (it fixes the `CycleGrowthTrigger`
      confound, #19: cycle count rises from β→0 *and* staleness, drift rises only from staleness).
    - **VALIDATED end-to-end (α-targeting controller + PC refresh strictly dominates SER on pitzDaily).**
      A prototype controller — raise β toward the α=1 boundary (`β ← β/α`, capped), ease gently when
      α=1 — with the k/ω AMGs refreshed every 5 steps and the step `filter_jit`'d (to match SER's
      compiled `while_loop` footing, ~2.2 s/cyc), A/B'd from the cold hybrid IC against E1's SER march:

      | reach | SER (E1) | α-controller + refresh |
      |---|---|---|
      | rel 0.10 | 15.5 min | 11.4 min |
      | rel 0.054 | **64 min** | **24 min (2.6×)** |
      | deepest | **rel 0.052** (67 min, then stalled) | **rel 0.032** (41 min) |

      Faster at every overlapping residual, the lead *widens* into the tail (1.3× → 2.6×), and it
      reaches residuals SER never touched. The mechanism is the diagnosis playing out live: as the
      state stiffens α drops below 1 and the controller *raises* β into the 2–5 band (refresh holding
      cycles ~16) while SER collapses to β≈0.10 and grinds. Two prior arms confirm the attribution:
      (a) the **frozen-PC** α-controller *lost* (0.65×) — cycles rose with β (25 vs SER's ≤14),
      the β↔PC-refresh coupling biting, so the refresh is load-bearing; (b) the **eager** (un-jitted)
      version was handicapped ~1.4×/cyc — the jit is needed for a fair comparison, not for the physics.
    - **The controller has a CEILING — it does not converge either (it stalls at rel ~0.03, deeper than
      SER's ~0.05, not at a root).** The cause is its own **over-damped hunting**: the `β/α` raise
      overshoots *past* the α=1 boundary to where the full step is tiny (α=1, ρ~2%), then eases slowly;
      α saturates at 1 above the boundary, so the controller is blind there and cannot sit at the
      productive edge (the sweep's 20–60%/step β). So the direction is right and the win is real, but a
      dynamics rework is needed: approach α=1 *from below* without overshooting, or pair α with a
      step-productivity signal.
    - **PRODUCTIONIZED as an injected strategy pair (the direction is shipped, opt-in).** The β schedule
      is now the injected `RelaxationSchedule` (SER = `SwitchedEvolutionRelaxation`, the default; see the
      `continuation.py` bullet), and the α-targeting control is `AlphaTargetingControl`, a `StepControl`
      on the eager `forward_march` (opt-in via `solve_coupled(step_control=…)`, composes with
      `refresh_trigger`). It is **never a default** — it does not converge standalone and loses without
      the refresh — and its numeric gains (`beta_start`/`growth_cap`/`ease`) are placeholders, like the
      trigger's. The dynamics rework above is the open follow-up. Study harnesses in the scratchpad
      (`beta_sweep.py`, `alpha_probe.py`, `alpha_controller_march.py` = frozen-PC, `alpha_refresh_march.py`
      = the winning arm) remain as the calibration/replay tools.
  - **Where the coupled-solve cost actually is (settled by measurement).** As the SER ramp drives `β → 0`
    through the march, the *unshifted* coupled saddle Jacobian is severely ill-conditioned, so the
    diagonally-shifted GMRES burns thousands of matvecs per solve (measured: one shifted solve ≈ 36 s at
    β=2, 127 s at β=0.2 on ~12k-cell pitzDaily — note lineax `num_steps` counts restart **cycles**
    ×`restart`, not iterations). **The `β → 0` here is SER-induced and correctable, not inevitable — see
    the schedule-runs-backwards finding above.** Several levers were probed: two are wired but **off by
    default** (kept for further evaluation, not the fix), one is dead, and one — refreshing the **scalar**
    k/ω AMGs after the flow separates — is a real ~2.6× win, now BUILT (see below):
    - **Flooring the SER `β` below (`β = max(beta_floor, β₀(‖R‖/‖R₀‖)^p)`, `PseudoTransientStep.beta_floor`,
      default 0 = off) — correctness-safe, a measured WASH, kept off-by-default.** It never moves the
      converged root (the shift `β d` scales the correction `δ`, which vanishes at `R=0`; it only damps the
      *path*, linear instead of quadratic terminal steps) and it does make each late solve cheaper. But
      end-to-end it is a net wash: floor 0.0 vs 0.3 reached the same tolerance in the same wall time on
      `solve_coupled`, because the cheaper late solves cancel the extra Newton steps. Wired through
      `coupled_continuation(beta_floor=…)` for further evaluation; not a default because it is a wash.
    - **The block-scaled per-field residual measure (`block_scaled_norm=True`) — kept off-by-default
      because it *stalls* the march.** A `BlockScaledNorm` over `[flow, k, ω]` weighs every field rather
      than the `ω` block that dominates ‖R‖, but the per-block relative norm plateaus long before the
      fields converge, so `coupled_continuation` defaults to the Euclidean `jnp.linalg.norm` and exposes
      `block_scaled_norm` (default `False`) to request the block measure for experimentation. The
      `BlockScaledNorm` class and its `_coupled_residual_norm` builder are kept as that opt-in path.
    - **A block-*triangular* preconditioner (forward-substituting `∂R_turb/∂flow·δ_flow`) — tried, WORSE,
      dead.** It made the channel worse (85 vs 51 outer cycles at β=0.5) and on recirculating pitzDaily was
      so bad GMRES could not converge at all: stronger flow↔turbulence coupling *amplifies* the inexact
      diagonal blocks' inversion error it propagates downstream. So the missing cross-coupling is **not**
      the bottleneck.
    - **The real cost is the pressure-Schur *approximation* at high Reynolds number — and strengthening
      the inner solve CANNOT fix it (measured; do not re-attempt).** The block-diagonal conv+MSIMPLER
      preconditioner is *excellent* at low Re (4 outer cycles on a Re=2500 channel) and weak only at high
      Re / recirculation (17 cycles on a Re=1e5 channel). The weak block is the **flow saddle**, not the
      k/ω scalars (per-block error operator `E_b = I − A_b·M_b` on a developed Re=1e5 channel: flow
      ρ=34.0 / one-shot 24.1, vs ω 13.9 / **2.4** and k 8.5 / 7.9 — ω's high ρ with a low one-shot is an
      isolated outlier eigenvalue GMRES kills in one iteration, a red herring). But every lever *inside*
      that block is dead:
      - **More velocity-AMG V-cycles (×2/×4/×8): ρ 34.019 → 33.995 → 34.031 → 34.046 — no effect at all.**
      - **More Schur V-cycles (×2/×4/×8): ρ 41.6 / 48.7 / 48.5 — strictly worse.** Inverting `Ŝ` *more
        accurately* making the preconditioner *worse* is the signature that `Ŝ` is the **wrong operator**:
        the error is the Schur *approximation*, not its inversion (a partial V-cycle was accidentally
        regularizing it). Driving both sub-solves toward exact never beats the 1-cycle baseline.
      - **Rebuilding the preconditioner at the developed state (staleness) does not help *the flow
        block*** (ρ 34.0 → 31.6 on the channel; 49.9 → 91.9, i.e. worse, on pitzDaily, with an identical
        one-shot). The frozen *flow* reference is fine — the convective linearization is Peclet-robust and
        MSIMPLER's Schur is velocity-independent. **Confirmed on the real solve:** refreshing only the flow
        block at a separated pitzDaily state made it slightly *worse* (31 → 34 outer cycles at β=2).
      - **BUT refreshing the *scalar* k/ω AMGs is a real 2.6× cycle win once the flow separates — the one
        staleness lever that does pay (measured on the real solve, not ρ).** The scalars were noted above
        as going stale (ω ρ 13.9 → 3.3 rebuilt) but dismissed as "not the cycle bottleneck" on the ρ /
        one-shot proxy; on the **real coupled shifted solve** that dismissal does not hold. Marching
        pitzDaily to a genuinely separated state (25 pseudo-transient steps, rel 3.0e-2, 70 recirculation
        cells, `x_r/h` 0.87) and re-solving the **same** shifted system with the preconditioner refreshed
        block-by-block (operator held fixed; every solve converged, `‖Aδ−b‖/‖b‖` ~1e-8):

        | refreshed | cycles | matvecs | wall |
        |---|---|---|---|
        | nothing (all frozen at the cold IC) | 31 | 3720 | 68.9 s |
        | **k/ω scalar AMGs only** | **12** | **1440** | **27.4 s** |
        | flow block only | 34 | 4080 | 71.8 s |
        | everything | 13 | 1560 | 30.4 s |

        So the entire gain is the **scalars** (31 → 12), the flow refresh contributes nothing (everything
        ≈ scalars-only), and this is a textbook instance of the ρ caution above — the scalars' low one-shot
        made them look harmless while they were worth 2.6× on the real iteration. **The benefit only
        appears once the flow has separated**: at a *pre-separation* state (4 march steps, no recirculation)
        a full refresh is worthless (17 → 14 cycles at β=2, and *worse* at β=0.2, 43 → 83), which is why an
        early measurement gives the wrong answer. Full-refresh gains were confirmed at β ∈ {2, 0.5, 0.2}
        (31→13, 19→12, 31→18); the block-by-block isolation above was run at β=2. **Implication for
        implementation: refresh only the two `ScalarTransportPreconditioner`s and leave the flow block
        frozen** — much cheaper than a whole-policy rebuild, and it avoids the flow refresh's small
        regression. It is adjoint-safe (the preconditioner is `stop_gradient`-ed whatever it is frozen at,
        so a refresh changes only the forward Krylov count, never the converged state or its IFT adjoint).
        **BUILT** — `forward_march` + `CycleGrowthTrigger` (see the `march.py` section) segment the march
        around the off-jit rebuild, which is required because the traced solve is one `lax.while_loop` and
        scipy AMG assembly cannot run inside it; `solve_coupled(refresh_trigger=…)` is the driver. A
        refresh still forces a full recompile (~60–240 s) because these are non-pytrees hashed by
        identity, which is why `refresh_limit` bounds how often it may happen. That recompile is
        avoidable in principle — **the coarsening structure is value-independent** (`_aggregate` takes only
        `(owner, nb, n)`, pure graph topology, so for a fixed mesh the aggregates, `n_coarse` and every
        sparsity pattern are invariant), so only `val`/`diagonal`/`lam_max`/`coarse_inv` change; making
        those traced leaves over a static index structure would turn a refresh into a cache hit.
      - **Rescaling the MSIMPLER `k` is a ρ mirage — validate on the real march, never on ρ.** Growing `k`
        collapses ρ (34.0 → 9.6) but barely moves the one-shot error (24.1 → 22.6), and the ρ-minimizing
        `k` sits ~40× *above the maximum* of the whole per-cell `ρV/a_P` distribution — i.e. the degenerate
        limit `schur_a_p → 0`, `Ŝ⁻¹ → 0`, which simply switches the pressure correction off. On the real
        production march it is **slower**: shipped auto-`k` 348 s / 8 steps vs `k×4` 447 s (28% slower) at
        an identical residual trajectory. **The shipped per-apply `mean(ρV/a_P)` calibration is
        near-optimal — do not "fix" it**, and do not make the Schur "shift-consistent" with the
        pseudo-transient `a_P(1+β)` either (that direction is strictly worse at every β).
      **Root cause:** the MSIMPLER Schur is a *constant-coefficient* (scaled pressure-mass-matrix) Poisson,
      which is a near-Stokes/low-Re approximation and degrades as convection strengthens — exactly the
      high-Re/recirculating regime here. **The fix is a better Schur approximation, not a better solve of
      this one:** the stabilized least-squares-commutator (LSC) of Elman, Howle, Shadid, Silvester &
      Tuminaro (2007), which needs only momentum-operator applies, `diag(V)`, and the assembled pressure
      Poisson `B Q̂⁻¹ Bᵀ` this file's Schur already builds. Use the **stabilized** (2007) variant — a
      Rhie–Chow collocated discretization is equal-order stabilized, so the original (2006) LSC
      underperforms on it — and re-derive its boundary treatment for cell-centred FVM. Prefer LSC over
      pressure-convection–diffusion (PCD), whose auxiliary operator carries finite-element boundary
      recipes that do not transfer cleanly to FVM.
  - **The residual measure is an injected `ResidualNorm`, owned by the `ForwardStep` (`solve/norm.py`).**
    Every `ForwardStep` exposes `norm()`; `ImplicitNewtonSolver` reads it for the outer stopping test
    (threaded through `_forward`/`_implicit_solve` as the extra nondiff arg `norm_fn`) and the strategy
    uses the *same* measure for its own globalization — so the convergence test, the SER ramp
    `β = β₀(‖R‖/‖R₀‖)^p`, `backtracking_line_search` (which now takes a `norm=` kwarg), and the
    `DivergenceGuard` all agree on one scale. Default is `jnp.linalg.norm` (`DampedNewtonStep.norm()` and
    `PseudoTransientStep`'s `residual_norm` field both default to it), so **the flow path is
    bit-identical**. The non-trivial impl is `BlockScaledNorm(sizes, scales)`: it splits the flat
    residual into contiguous blocks, divides each by its own reference magnitude, and returns the L2 of
    those per-block relative residuals — `sqrt(Σ_b (‖R_b‖/scale_b)²)`. **Why it exists (the coupled-RANS
    fix):** the plain Euclidean ‖R‖ on `[flow, k, ω]` is ~100% ω (ω residual O(1e5), k O(1e-3)), so the
    line search can neither *see* nor *protect* the k block — a step that collapses k is accepted (barely
    moves the ω-dominated norm) while one that reduces k is vetoed because ω ticked up, and k gets
    starved (measured: k median collapses to ~7e-5 vs a physical ~0.5, and the march freezes). `coupled.py`
    builds a `BlockScaledNorm` over `[flow, k, ω]` (and `[…, β]` for the mass-flow bordered march) with
    per-field scales `‖R0_field‖` at the reference state, so the whole system is judged. The adjoint never
    forms a residual norm, so `norm_fn` is a **forward-only** device — the converged state and IFT
    gradient are norm-independent (the bwd pass takes it as a `del`-ed nondiff arg). Since it is a static
    field holding an `eqx.Module` with static tuple fields, it stays hashable for the `custom_vjp` nondiff
    slot (like the `lineax` solver already carried there).
  - **A `ShiftPolicy`'s preconditioner must stay a non-pytree (binding, #105).** `ScalarTransportPreconditioner`
    (`turbulence/preconditioner.py`) is a plain `dataclasses.dataclass(frozen=True, eq=False)` ABC with
    `ConvectionAmgPreconditioner` / `AirAmgPreconditioner` concrete strategies — deliberately **not** an
    `equinox.Module`. Two things break if it is made a pytree: (i) a solve taking it as an argument traces
    its hierarchy arrays, which then reach `_implicit_solve`'s `custom_vjp` as tracers in a
    `nondiff_argnums` slot and JAX raises `UnexpectedTracerError`; (ii) it is *because* the object is opaque
    to JAX that carrying one instance across outer sweeps is a `filter_jit` cache **hit** (non-array
    arguments go to the static side, hashed by identity). Both were hit and fixed while building #105 —
    do not "modernize" these into `equinox.Module`s.
- **`march.py` — BUILT (`forward_march`, `StepReport`/`MarchResult`, `RefreshTrigger`/`CycleGrowthTrigger`):
  the observed, forward-only march that drives a mid-march preconditioner refresh.**
  - **Two marches, ONE decision layer (binding — this is the shape to hold).** `_forward` (traced,
    inside `custom_vjp`, has the root guard, cannot stop early, cannot be observed) and `forward_march`
    (eager Python loop, forward-only, **no guard by design**, stops on an injected trigger, reports every
    step). They are not duplicates: `forward_march` calls the **same** `forward_step.stepper()`, the same
    `forward_step.norm()`, and the same `_within_tolerance`. The only residue is a ~6-line loop shell,
    pinned against drift by a test that both marches reach the same state on the same residual.
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
  - **`refresh_limit` lives on the driver, not the trigger.** That keeps the trigger a **pure function of
    one segment's history** — which is what lets `warmup`/`patience` re-apply correctly after each
    refresh, lets it be unit-tested on synthetic histories with no solve, and (the big one) lets it be
    **calibrated offline**: log one march with `refresh_trigger=None` and an `on_step` observer, then
    replay candidate parameters against the log. No numeric default here is calibrated — they are chosen
    conservative (late rather than early) and must be set from an instrumented full-mesh run.
  - **Observation does NOT require a refresh (binding — this was a real bug).** `solve_coupled` runs the
    observed pre-march when the caller wants a refresh **or** merely wants to watch
    (`observing = refreshing or on_step or on_checkpoint`). Gating it on the trigger alone makes an
    *instrumented reference march* — `refresh_trigger=None` plus an observer, which is exactly the run a
    trigger is calibrated against, and the longest-running one — produce **no output at all** and sit
    silent for hours. Consequence to keep in mind: an observed solve spends `max_steps` on the pre-march
    and `max_steps` again on the finishing solve, so the budget is larger but *split*; instrumenting a
    solve already near its limit can turn a pass into a convergence-guard raise. Pinned by
    `test_the_march_reports_progress_without_a_refresh_trigger`.
  - **`checkpoint` is a SECOND seam, separate from `observer` (binding).** `checkpoint(report, state)`
    carries the state; `observer(report)` carries only numbers. Keeping the state off the report history
    is what keeps a `RefreshTrigger` a pure function that can be replayed offline against a logged march
    — put the state on that seam and a trigger could read the physics, and trigger calibration would cost
    one full solve per candidate instead of one logged run for all of them.
  - **Reporting seam.** `StepReport(step, cycles, residual_norm, residual_ratio, alpha)` + `MarchResult`,
    plus an optional streaming `observer` (a long march must not withhold all logging until it finishes).
    The trigger and a future logger consume the identical objects, so there is no second reporting path.
    Per-step observation exists only where the march is eager — the traced `_forward` would need
    `jax.debug.callback`, a separate decision; do not promise per-step reporting on the differentiable path.
  - **`StepControl` — stateful, feedback-driven step reshaping on the eager march only (binding — the
    twin of `RelaxationSchedule`, deliberately NOT one interface).** A `RelaxationSchedule` is memoryless
    and lives on the differentiable step; a `StepControl` reads the *previous* `StepReport` (α, cost) —
    feedback available only after a step — to reshape the next step, and may raise under `jax.grad`, so it
    lives here beside `RefreshTrigger`, never on the traced path. `forward_march(step_control=…)` calls
    `next_step(base, previous, state) -> (ForwardStep, new_state)`, threading the control's own state; the
    march stays β-ignorant (the control returns a ready-to-run step, typically `base` with a
    `ConstantRelaxation` β leaf via `tree_at`, so `_march_step` stays a cache hit). Unifying the two into
    one `(rn, rn0, α, state) -> (β, state)` interface was rejected: it would union SER's needs with the
    control's (dead α/state args for SER), drag α onto the differentiable core where the line search
    cannot even produce it before the step, and risk the byte-identity of the default path. The one
    concrete `StepControl` is `AlphaTargetingControl` (`solve/step_control.py`) — experimental, opt-in,
    see the "SER β schedule runs backwards" bullet.
- **Gate C — PASSED (`tests/integration/test_skewed_diffusion.py`).** With
  `CorrectedGreenGauss` injected into the residual on a 25%-skewed mesh, one Newton step
  drives `‖R‖` ~24 → ~1e-12 and reproduces a harmonic linear field to ~5e-13 (linear-exact
  on a skewed grid). The reference's lagged correction is emulated with `stop_gradient` on
  the gradient (residual value real, Jacobian omits the correction) and needs ~8
  deferred-correction sweeps — the concrete before/after. The nested gradient GMRES is
  differentiated through cleanly (forward-mode `jvp` inside the outer Newton).

## Binding decisions
- **Two-level implicit differentiation**: IFT on the converged
  Newton state (skip Newton iterations) + `custom_vjp`/adjoint on each linear solve
  (skip Krylov iterations). **Neither loop is unrolled onto the tape.** Say "no loops on
  the differentiation path," not "no loops."
- Prefer **lineax** (or `jax.scipy.sparse.linalg`) for the solve with built-in implicit
  diff; add a `custom_vjp` only where the library's differentiation is not exact through
  the converged solve. **Verify** the adjoint is a single transpose solve, not an
  unrolled iteration — this is the whole correctness claim.
- The **preconditioner is the top research risk.** Literature synthesis is done; the chosen
  direction and the traps follow. Headline: a
  **block-triangular SIMPLE-type** preconditioner using the lagged `a_P` for the Schur approximation,
  with a **fixed-cycle multigrid inner** pressure solve built once off-jit and frozen; keep the inner
  *fixed* (constant operator) so plain GMRES + the verified transparent-left-PC suffices (a *variable*
  inner would force FGMRES). On **`jaxamg`**: the search confirmed it is **NVIDIA/AmgX-locked and
  scalar-only** (no coupled/saddle-point, no AMD/TPU) — usable at most as a pressure-Poisson *inner*
  escape-hatch on NVIDIA hardware, **not** the coupled solver or an architectural commitment. Do not
  adopt it on the README's word. **`LSC` original / `PCD` carry equal-order/FEM traps** (use stabilized
  LSC for Rhie–Chow; PCD needs FEM-BC re-derivation).
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
    sound because **the aggregation coarsening is a pure function of the graph** — `_aggregate` reads
    `owner`/`nb`/`n` and never the coefficients — so on a fixed mesh a hierarchy re-derived at a new
    operator has identical aggregates, coarse sizes and array shapes, and only values differ. Both
    properties are pinned in `tests/unit/test_multigrid.py`
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
    whereas the aggregation path gets it for free by rebuilding. Also: do not add a
    strength-of-connection filter to `_aggregate` without revisiting this, as that would make the
    aggregation path value-dependent too.
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
    with the assembler: `frozen_operator._require_valid_graph` (`n ≥ 1`, matched `owner`/`nb`, in-range
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
    (dense pseudo-inverse) coarse solve; it has no `max_levels` parameter. On the fine level the
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
- **Where preconditioning must attach — measured, do not repeat the wrong lever.** For the
  skewed lid-driven cavity (`CorrectedGreenGauss`, `FirstOrderUpwind`) the per-Newton-step cost
  splits cleanly: the **outer coupled saddle-point GMRES takes 67 steps at 432 dof, 127 at 768
  dof — growing ~O(N)**, while the **inner gradient `A_g` solve is a flat 4 steps** regardless of
  mesh (it is volume-dominated and inherently well-conditioned). So the outer block solve is the
  whole bottleneck. **Preconditioning the *inner gradient* solve was built, measured, and
  reverted:** an inverse-volume-Jacobi `M≈A_g⁻¹` took the inner solve 4→3 steps and was
  *net-negative* end-to-end (132 s vs 121 s/step — the extra matvec per iteration outweighs the
  one iteration saved). This is the same outcome as the block-Jacobi velocity-diagonal experiment
  (`flow.md`): the cheap diagonal is not the missing physics. The real lever is an **outer**
  pressure-Schur / SIMPLE-style block preconditioner on the coupled `(u,p)` system, attaching via
  the `solve_linear(preconditioner=…)` seam.
- **The "gradient Schur elimination" is already exact and free from AD — it was never a numerical
  gap.** Feeding a gradient scheme (nested `lineax` solve `A_g g = Bφ`) into the flow residual and
  taking `jax.jvp` makes `lineax`'s implicit-diff form the exact Schur complement
  `S = ∂R/∂x + (∂R/∂g)A_g⁻¹B` *without unrolling* the inner Krylov loop. The skewed cavity with
  `CorrectedGreenGauss` **converges quadratically** (‖R‖ → 6e-12, `u_min=-0.204` vs Ghia −0.211),
  full Newton, differentiable. What remained was purely performance — not correctness or
  convergence of the absorbed gradient.
- **The efficient realization of the absorbed gradient — `SweptCorrectedGradient` (built, measured,
  a ~5× win).** Two costs of applying `A_g⁻¹` inside every outer matvec are separable from the outer
  iteration count above: the *per-matvec* cost and the *compile* cost of a nested implicit-diff GMRES.
  Both collapse if the constant, well-conditioned `A_g` is inverted by a **fixed number of matrix-free
  Richardson sweeps, unrolled** (no `lineax`, no implicit-diff tangent solve, no dense matrix). On the
  N=32 skewed cavity this cut a coupled Newton step from **112 s → 23 s run and 96 s → 23 s compile**
  (the compile collapse localizes the earlier blow-up to the nested Krylov + control flow), staying an
  exact drop-in (3.8e-10). Sweep count is **mesh-independent** ⇒ `O(n)`. (A *dense* LU of `A_g` was also
  built and measured but **removed** — exact yet `O((n·dim)²)`, so strictly dominated by the swept apply
  at every size; see `schemes.md`, do not rebuild.) The remaining lever is still the **outer** block
  preconditioner (the 67→127 outer iterations), which is independent of the gradient scheme.
- **Gate C (the improvement-over-reference claim):** on a non-orthogonal mesh the AD-exact
  Jacobian must converge the linear problem in **one** Newton step, where the reference
  needed several. Guard this with a test.

## Testability seam
- The Newton solver is a class constructed with an **injected residual object and
  linear-solver strategy** (CLAUDE Principle 1), so it is tested against a trivial
  analytic residual (e.g. a quadratic) with a known root and known Jacobian — no FVM
  mesh required.
- The adjoint is tested by finite-difference agreement of `jax.grad` through `solve()`
  on a small problem, plus the AD-correctness / no-NaN gate every integration suite
  carries (CLAUDE Testing Architecture).
