---
paths:
  - "aquaflux/solve/forward_step.py"
  - "aquaflux/solve/continuation.py"
  - "aquaflux/solve/step_control.py"
  - "aquaflux/solve/retry.py"
  - "aquaflux/solve/relaxation.py"
  - "aquaflux/solve/line_search_growth.py"
---

# Rules — `aquaflux/solve/` globalization (forward step, continuation, line search)

> Split out of `solve.md` (2026-08-18). See `solve.md` for the package-wide contracts, current
> configuration, and general binding decisions this file assumes. The dated investigation behind
> these architecture/binding statements — several rounds of measurement on the residual-measure
> choice, the shift basis, and the SER schedule, some later corrected or superseded — is preserved in
> `solve-globalization-log.md`; read it before re-proposing something that sounds already tried.
>
> **Adding new content: a dated measurement or investigation step goes in `solve-globalization-log.md`,
> not here.** Update this file's architecture/binding prose only when an investigation reaches a
> durable verdict (see `solve.md`'s "Where new content goes").

## Globalization — forward step, continuation, line search

- **`ShiftedStep` is the shared body of the two shifted forward steps (`solve/continuation.py`, BUILT
  2026-08-15).** `PseudoTransientStep` and `DualTimeStep` differ entirely in `stepper()` — what one
  *outer step* means — and not at all in how they are configured or interrogated. The eight fields they
  share (`shift_policy`, `relaxation_schedule`, `line_search`, `step_limit`, `step_projection`,
  `forward_solver`, `residual_norm`, `adjoint_preconditioner_factory`) and all three `ForwardStep`
  accessors (`norm`, `default_solver`, `adjoint_preconditioner`) were written out **twice, identically**,
  field comments included. A subclass now supplies `stepper` plus the fields its own step shape needs.
  `DampedNewtonStep` is deliberately NOT a subclass: it has no shift, and its `default_solver` returns a
  different constant, so folding it in would mean inventing a `relaxation_schedule` it does not possess.
  **Verified as a pure refactor**: no field and no default changed on either class, compared field by
  field against the previous implementation (`DualTimeStep.line_search` stays **10** against the base's
  **0** — it redeclares it, which is why that difference has to be checked rather than assumed).

- **`_TrailingFirstFieldSplit` supplies only what differs, and `apply` has ONE body (BUILT 2026-08-15).**
  The two orderings were mirrored copies — 14 lines differing in 5 — and the copy had dropped the base's
  explanation of why the transposed coupling is formed once. The class docstring justified the split as
  avoiding "a branch on ordering inside `apply`, on a path that runs once per Krylov iteration", which is
  a real cost and the wrong conclusion: the ordering **cannot change after construction**, so
  `_set_order(first=…)` resolves it there and `apply` reads a pair of `(inverse, dofs)` records. The
  remaining branch is on `transpose`, which the old body already had — transposing a block-triangular
  inverse reverses the order and uses `Cᵀ`, and that is the whole of the difference between the four
  cases it used to spell out. **Bit-identical** across all four (both orderings × both directions), each
  compared as a full dense action rather than on one vector.

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
  - **`ShiftedForwardStep` is the SECOND contract, and the eager march's beta machinery requires it
    (binding, 2026-08-15).** `ForwardStep` says what every strategy must *do*; `ShiftedForwardStep`
    says what one must additionally *carry* — a `relaxation_schedule` holding `beta` as a readable,
    replaceable **dynamic** leaf. It is separate rather than folded in because not every strategy has
    a shift: `DampedNewtonStep` globalizes by backtracking alone, and demanding a `relaxation_schedule`
    of it would be inventing a quantity it does not possess.
    **What it replaces, and why it matters more than it looks:** the requirement was enforced by
    `hasattr` probes scattered through `forward_march`, which fail **silently**. A `DampedNewtonStep`
    satisfies `ForwardStep` in full, so a march configured with `RetryPolicy.on_cycles` accepted it and then
    never escalated — from the log, indistinguishable from a march that never needed to. One reporting
    path failed the *opposite* way and read `active_step.relaxation_schedule` unguarded, so the same
    conforming step raised `AttributeError` mid-march. `forward_march` now checks **once, on the first
    iteration, against the step the CONTROL produced** (`RetryPolicy.require_shifted`), naming the
    feature and what it needs.
    **⚠️ THE OBJECT CHECKED IS `active_step`, NOT THE STEP HANDED IN, AND THAT IS LOAD-BEARING (fixed
    2026-08-16).** Checking before the loop looks equivalent and is not: the default schedule is
    `SwitchedEvolutionRelaxation`, which is **memoryless and exposes no `beta`**, while a
    `ShiftStrengthControl` swaps a `ConstantRelaxation` (which does) onto the step every iteration. So
    the readable beta the escalation needs is supplied **by the control**, and a pre-loop gate sees a
    step that never runs. It rejected the shipped configuration outright — every `bfs3d` march runs a
    shift control *and* both retry thresholds, so **not one could start**, under any flow inverse, with
    a `TypeError` whose own message names `DualTimeStep` as acceptable while refusing one. The check
    still precedes any escalation, which is the property it exists for, and with no control
    `active_step` is the base step so the ungated path is unchanged. Pinned in **both** directions by
    `test_the_escalation_guard_accepts_a_shift_a_step_control_installs` and
    `test_the_escalation_guard_still_rejects_a_step_that_can_never_escalate` — the pair matters,
    because a gate can be "fixed" by deleting it and only the second test notices.
    ⚠️ **This was found by RUNNING a march, not by the suite, and it was the second such break in one
    sitting** (the other: `compare.py` handed `solve_coupled` a bare `precondition_step` callable where
    a `RefreshPolicy` was expected, so `refresh.observes` raised before step 1). **No test tier drives
    this case's own driver**, so a refactor can tighten a seam, take the case out entirely, and leave
    every gate green. Treat "the fast gate passes" as saying nothing about whether `bfs3d` can march.
    - **Gate only what is genuinely silent (binding).** The escalation is gated. The divergence retry
      (`retry_solver` / `retry_divergence_cap`) is **not** — it re-solves at a tighter tolerance and
      never touches beta, so it works on any step; its *reporting* goes through `_shift_of`, which
      returns `None` for a step with no shift rather than inventing one. A `StepControl` is **not**
      gated either: the protocol only asks it to return a ready-to-run step, a step-agnostic one is
      legitimate (and exercised), and a control that *does* drive beta already fails loudly from its
      own `tree_at`. Gating those would reject what the protocol permits.
    - **The runtime check tests the SHIFT, not `isinstance(..., ShiftedForwardStep)`.** The argument is
      already typed `ForwardStep`, so re-testing those four methods at runtime would reject a
      legitimate duck-typed step for a reason unrelated to the feature asked for — which it did, on
      every test double, when first written that way.
- **`continuation.py` — BUILT (`PseudoTransientStep`, residual-agnostic).** The pseudo-transient
  continuation engine lives **here in `solve/`, not in `flow/`** — it is a `ForwardStep`
  (`stepper`/`default_solver`/`adjoint_preconditioner`) that runs an **injected**
  `RelaxationSchedule` for the shift strength β, the diagonally-shifted solve `(J + diag(βd))δ = −R`
  (`solve_linear(throw=False)`), and the closed-loop accept/escalate `while_loop`. **`stepper()`
  returns a `StepOutcome`** carrying the accepted attempt's cycle count, its line-search factor α (the
  step-quality signal, ≤1; `_forward` drops both off the `custom_vjp` primal, a march reads them), and
  **its `residual_norm`** — the measure at the accepted candidate, which the escalation carry now
  transports beside the candidate it belongs to (a fully-rejected step returns `phi` untouched, so it
  reports `phi`'s own measure, the `residual_norm` the caller handed in). The attempt already formed it
  for the acceptance test, so both drivers read it instead of re-evaluating the residual at the same
  iterate.
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
  - **The shift's SPATIAL distribution is an injected `ShiftBasis` (`solve/shift_basis.py`) — the
    spatial twin of the `RelaxationSchedule` (binding).** `RelaxationSchedule` sets *how much* damping
    (the scalar β); `ShiftBasis` sets the per-cell base diagonal `d` the shift `β d` is built on, from
    the operator's two diagonal buckets: `local_diagonal(convective, dissipative) -> d`. The one concrete
    is `LocalCourantBasis(dissipative_weight=w)`: `d = convective + w·dissipative`. **`w = 1` (default) is
    the full operator diagonal `a_P`** — and because `d = a_P` (the same diagonal the operator carries),
    `β a_P` is spatially-*uniform* under-relaxation (relaxation `1/(1+β)` in every cell), byte-compatible
    with the historical shift. **That equivalence is exact and worth stating plainly: on the momentum
    rows, this shift IS the implicit under-relaxation a segregated pressure-correction solver applies,
    at `α_u = 1/(1+β)`.** Both sides match, not just the diagonal — the shift acts on `δ = φ − φ_k` and
    so contributes `β a_P φ_k` to the right-hand side, which is exactly the `((1−α_u)/α_u) a_P φ_old`
    that implicit under-relaxation adds. At the pitzDaily march's operating point `β ≈ 1.9`, that is
    `α_u ≈ 0.345`, against the 0.3 that established segregated codes use as their default momentum
    relaxation. **Consequence (load-bearing for the cold-start work): the momentum treatment is the
    industry-standard one in different notation, so it is NOT where a cold-start reachability gap can
    be hiding, and "our globalization is exotic / uncovered by theory" is false.** Look instead at what
    differs — the coupled pressure/continuity treatment, the turbulence relaxation and limiters, and the
    fact that the reference which reliably reaches this root is a *transient* run. **`w = 0` is a genuine local convective time step** (`d = Σ_f max(mdot_f,0)`
    = ½Σ|mdot|), the non-uniform per-cell `Δt` a Courant condition implies — OF's `Co = ½Δt Σ|φ|/V` and
    Fluent's *segregated* local pseudo-time step are this same convective basis. The buckets are supplied
    by each block: `rhie_chow.momentum_diagonal_parts` (velocity) and
    `preconditioner.scalar_transport_shift_diagonal_parts` (k/ω), each summing to the block's total shift
    diagonal to rounding. **Consequence for the preconditioner (binding):** with a non-`a_P` basis the
    shifted diagonal is `a_P + β d`, *not* `a_P(1+β)`, so `make_preconditioner` must invert `a_P + β d` —
    the velocity block's `apply_at` is fed exactly that (was `a_P(1+β)`; identical when `w=1`). Threaded
    through `momentum_continuation`/`coupled_continuation`/`solve_coupled(shift_basis=…)` and the k/ω
    shift policies; **pressure keeps zero shift regardless** (elliptic), which is why a *local* basis is
    defensible on this coupled solver where Fluent uses a global time scale for its coupled path.
    - **The velocity buckets' SOURCE is injected (`VelocityShiftParts`), because the shift and the
      preconditioner have different lifetimes (binding).** `CoupledShiftPolicy` used to take them from
      the frozen flow preconditioner, which welded two unrelated lifetimes together: the preconditioner
      is deliberately frozen for many steps (re-freezing the flow block is measured unhelpful), while the
      shift is a *local time scale* that should describe the operator being solved now. Worse, the
      borrowed quantity was **live in velocity but frozen in viscosity**, so in a coupled solve `μ_eff =
      ρ(ν + ν_t)` was stuck at its freeze-state value and the velocity time scale ignored the eddy
      viscosity that develops — nobody chose that, it fell out of reusing the preconditioner's assembler.
      `FrozenViscosityVelocityParts` (the default, `None`) reproduces the old diagonal **exactly**;
      `LiveViscosityVelocityParts` re-forms the closure at the state being stepped, costing one closure
      evaluation per *step* (milliseconds against a tens-of-seconds solve). It receives the scalar blocks
      **as solved**, so it holds the transforms and maps back to physical — passing `log ω` to the closure
      would silently build the shift from the wrong field. Note `a_p` still comes from the preconditioner
      and must: it is what the velocity block inverts, so the two would otherwise disagree.
      *Measured:* live vs frozen viscosity at a developed state is a modest p50 0.96 / max 1.82 with **no
      tail** — a correctness fix, not a large lever.
      - **Where the pieces live (binding, #156 seam 3).** The `VelocityShiftParts` **protocol** lives in
        `solve/shift_basis.py`, beside `ShiftBasis` — the natural sibling: `ShiftBasis` says *how* to
        combine the buckets, `VelocityShiftParts` says *where they come from*. `FrozenViscosityVelocityParts`
        (needs only a `BlockPreconditioner`) lives in `flow/continuation.py`; only `LiveViscosityVelocityParts`
        (needs `SSTTurbulence` + the transforms) stays in `turbulence/coupled.py`. This is what lets the
        **flow-only** `MomentumShiftPolicy` carry the *same* `velocity_shift_parts` seam (it could not
        before — the protocol lived in `turbulence`, and `flow` importing it would be a cycle). Its
        `parts(flow, k_solved=None, omega_solved=None)` makes the turbulence blocks optional, so the
        flow-only policy calls `parts(flow)` while the coupled one passes all three. The flow-only path is
        still behaviour-unaffected (constant μ ⇒ frozen and live coincide); the seam exists for symmetry
        and a future variable-viscosity flow, and to delete the byte-for-byte duplicated inline pattern.
      - **⚠️ `LiveViscosityVelocityParts.parts` declared the two turbulence blocks WITHOUT the protocol's
        defaults until 2026-08-20 (#282), so it did not satisfy the arity its own protocol promises.**
        Latent — nothing hands the coupled implementation to the flow-only policy today — but it would
        have become a `TypeError` from inside a shift policy the moment the monolithic builders started
        accepting a `velocity_shift_parts`, which is the same change that fixed it. It now takes the
        call and raises a message naming `FrozenViscosityVelocityParts`, because unlike its sibling it
        genuinely cannot do the job without the turbulence context. **A `Protocol` is not checked at
        runtime, so an implementation narrower than its protocol is invisible until the call that needs
        the wider arity — write the defaults even where the implementation will reject them.**
      - **`velocity_shift_parts` reaches all FOUR coupled builders since 2026-08-20 (#282).** It was on
        the two block builders only; `_monolithic_shift_source` now takes and forwards it. Nothing
        structural had excluded it — a live source needs momentum + turbulence + the transforms, not a
        flow preconditioner — and the configuration it was built for, a dual-time low-shift march whose
        shift tracks the developing `ν_t`, is a *monolithic* one. So it was absent from precisely the
        path it was written for.
    Adding a
    basis (e.g. a Fluent-style global min-physical-time-scale) is a new `ShiftBasis`; do not branch the
    policies.

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
  - **`stepper()` returns a `StepOutcome` (`phi_next`, its `residual_norm`, `cycles`, `alpha`, and
    four more — see `solve-march.md`) — ONE step method on the whole `ForwardStep` protocol,
    counted/uncounted pair deleted (binding).** Every strategy reports its step's restart-cycle count
    (`DampedNewtonStep` gets it from `newton_correction`, which now returns `(delta, r, cycles)`); a
    consumer with no use for it drops it (`phi, _ = step(…)`). A `counted_stepper()` sibling existed
    briefly, with `stepper()` forwarding to it and dropping the count — deleted for the same reason as
    `solve_linear_counted` above, and note it had **no production consumer at all** while it existed.
    The reported count is the **accepted** attempt's, not the sum over rejected escalation attempts —
    the cost of the step actually taken. **A step whose every attempt was rejected reports `0`**
    (`best_cycles` is only written on acceptance): a consumer must treat `0` as *no measurement*, not
    as *free*, or a rejected step reads as the cheapest in the march. Consumed by `forward_march`
    (`solve-march.md`); dropped by `_forward`.
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
  - **`forward_solver` overrides the shared `_INEXACT_CONTINUATION_SOLVER`; the coupled default stops on a
    relative residual in an INJECTED norm (`relative_residual_gmres`, `solve/linear.py`).**
    `default_solver()` returns the injected `forward_solver` when set, else the shared restart-40 GMRES.
    The coupled path injects restart 120 (the stiff saddle needs hundreds of restart-40 cycles; a
    40-vector subspace discards too much Arnoldi history). ⚠️ **The norm it stops in has MOVED since this
    was written — the "global 2-norm" below describes the arrangement these measurements were taken
    under, not the current default.** Since #282 every coupled family stops in the march's own row-scaled
    measure at `forward_rtol = 0.3`, built by `_coupled_step` rather than by any builder; see
    `solve.md`'s regime table. The mechanism below is unaffected and is why the componentwise stock stop
    was abandoned in the first place. **The dominant waste was the TERMINATION, not the restart.** The
    old `GMRES(rtol=1e-3, atol=1e-10)` reached true_rel ~4e-12 in ~15 restart cycles / 1800 matvecs on the
    cold-IC pitzDaily solve although only rtol=1e-3 was asked — because `lineax`'s stock stop is
    **componentwise** (`|r_i| ≤ atol + rtol|b_i|` under max_norm), and the ~470 near-wall ω wall-fixation
    rows start satisfied (right-hand side ~0), so their per-row scale collapses onto the absolute `atol`
    floor and a handful of them hold the whole solve to ~1e-10 (~9 orders past 1e-3). `relative_residual_gmres`
    scales the system to unit right-hand-side 2-norm and runs GMRES at `rtol=0, atol=target, norm=2-norm`,
    so it stops on `norm(r)/norm(b) ≤ target` — immune to those rows. ⚠️ **That residual is the TRUE
    one, not `‖Mr‖`: `_shifted_solve` takes `solve_linear`'s `preconditioner_side="right"` default, so the
    Krylov residual is `b − A M y = b − A x`. No "solution accuracy" follows from it — an earlier version of
    this line inferred "≈1% solution accuracy since M≈A⁻¹", which holds only under LEFT preconditioning and
    misled a later reader into a wrong hypothesis.** Measured on the real cold-IC march: ~3-5 cycles (often 2-3/step), ~4× fewer
    matvecs to the same `x_r/h`, trajectory unchanged.

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
- **`norm_builder` — the residual measure is re-derived every outer iteration, and held FIXED within
  one (binding).** `forward_march(norm_builder=…)` takes a `state -> ResidualNorm` and, at the top of
  each iteration, swaps the rebuilt measure onto the step with `eqx.tree_at` (the same mechanism the
  α-control uses for β) and re-measures `residual_norm_0` against it so the SER ratio stays on one
  scale. Every line-search trial step, the acceptance test and the reported norm within that iteration
  then use the *same* measure — **rebuilding per trial step would let a candidate win by shrinking its
  own denominator rather than its residual**, so the search would stop comparing like with like.
  - **This is why `residual_norm` is a DATA field on both `ForwardStep`s, not a static one.** A static
    field lives in the treedef, so swapping it would be a new compilation *every step*. As data, and
    with the measure carrying its scales as traced leaves over a fixed block structure
    (`RowScaledNorm`), the swap is a cache hit. A plain callable (the default) has no array leaves and
    is filtered to the static side regardless, so the default path is byte-identical.
  - **⚠️ `RowScaledNorm` is MARCH-ONLY today.** `ImplicitNewtonSolver` passes `forward.norm()` into
    `custom_vjp`'s `nondiff_argnums`, which requires a hashable object, and a pytree holding arrays is
    not hashable there. So the finishing solve keeps whatever measure it was constructed with. Letting
    the traced solver use it requires reworking that slot — not done.

