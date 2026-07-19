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
- **`newton.py` — BUILT.** `NewtonSolver(iterations, solver)` is an `equinox.Module` taking
  an **injected residual closure** and linear solver. Each step forms `J` matrix-free via
  `jax.jvp` and calls `solve_linear`; no hand-derived Jacobian. Verified (`test_newton.py`):
  one step is exact on a linear residual, a nonlinear residual converges, and `jax.grad`
  through the converged solve matches finite difference.
- **Deliberate Stage-A staging (not a shortcut):** for the *linear* transient-diffusion
  residual, one Newton step is exact, so the driver differentiates directly through that
  single step while `lineax` implicit-diffs the linear solve — the "no unrolled Krylov"
  half of the two-level scheme is already in place. The **IFT-over-Newton** half (a
  `custom_vjp` skipping *many* Newton iterations) is only needed once the residual is
  genuinely nonlinear (convection / coupled p–U); it is the documented next increment, and
  the injected-closure interface is unchanged when it lands.
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
  class**, since `PseudoTransientStep` is itself the `ForwardStep`. When a new nonlinear residual needs
  pseudo-time globalization, write a `ShiftPolicy` — do **not** re-implement the march.
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
