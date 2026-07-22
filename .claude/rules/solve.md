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
  (`stepper`/`default_solver`/`adjoint_preconditioner`) that owns the switched-evolution-relaxation
  schedule `β = β₀(‖R‖/‖R₀‖)^p`, the diagonally-shifted solve `(J + diag(βd))δ = −R`
  (`solve_linear(throw=False)`), and the closed-loop accept/escalate `while_loop`. **Two injected
  seams**, both `Protocol`s: the physics comes from a **`ShiftPolicy`**
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
  - **`line_search` — backtrack the shifted step before escalating β (binding, the coupled-RANS fix).**
    The step optionally scales the shifted correction `δ` back along `{1, 1/2, …, 1/2**line_search}`
    (`backtracking_line_search`, extracted from `implicit.py` and shared with `DampedNewtonStep` — one
    home for the ladder) and keeps the largest length that reduces the residual, **before** the
    accept/escalate test. `line_search=0` (default) is the old behaviour: take the full step `φ+δ`, and
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
  - **Where the coupled-solve cost actually is (settled by measurement).** As the SER ramp drives `β → 0`
    through the march, the *unshifted* coupled saddle Jacobian is severely ill-conditioned, so the
    diagonally-shifted GMRES burns thousands of matvecs per solve (measured: one shifted solve ≈ 36 s at
    β=2, 127 s at β=0.2 on ~12k-cell pitzDaily — note lineax `num_steps` counts restart **cycles**
    ×`restart`, not iterations). Three levers were probed; two are wired but **off by default** (kept for
    further evaluation, not the fix) and one is dead:
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
    - **The real cost is diagonal-block-preconditioner inexactness at high Reynolds number.** The
      block-diagonal conv+MSIMPLER preconditioner is *excellent* at low Re (4 outer cycles on a Re=2500
      channel) and weak only at high Re / recirculation (17 cycles on a Re=1e5 channel). The remaining
      lever is strengthening the weakest diagonal block's inner solve (more AMG V-cycles / tighter inner
      tolerance on that block), not adding coupling structure.
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
