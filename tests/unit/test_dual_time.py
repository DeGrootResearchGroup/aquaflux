"""The dual-time (backward-Euler) forward step drives a non-flow residual and keeps the IFT adjoint.

``DualTimeStep`` holds a reference ``phi^n`` and runs an inner Newton loop on the transient residual
``G = R + beta d (phi - phi^n)`` each outer timestep, so the shift sits in the residual (not only the
Jacobian) and the measured steady residual is the honest discrete time derivative rather than
``beta x travel``. These tests exercise it on a small scalar nonlinear root with a trivial shift
policy -- no mesh, no flow -- proving: it converges, its gradient is the exact
implicit-function-theorem sensitivity (independent of the iteration count), and with a single inner
step it reduces to the ordinary pseudo-transient step.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.solve import (
    DivergenceGuard,
    DualTimeStep,
    ImplicitNewtonSolver,
    PseudoTransientStep,
    ShiftTerm,
    SwitchedEvolutionRelaxation,
    positive_block_limit,
)
from aquaflux.solve.implicit import backtracking_line_search


class UniformShiftPolicy(eqx.Module):
    """A minimal non-flow shift policy: a uniform pseudo-time shift on every DOF, unpreconditioned."""

    strength: float = eqx.field(static=True, default=1.0)

    def shift_term(self, phi: jnp.ndarray) -> ShiftTerm:
        diagonal = self.strength * jnp.ones_like(phi)
        return ShiftTerm(diagonal, lambda relaxation: None)


def _residual(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """A nonlinear residual with root ``phi = cbrt(theta)`` (per component)."""
    return phi**3 - theta


def _solver(step: DualTimeStep, max_steps: int = 200) -> ImplicitNewtonSolver:
    return ImplicitNewtonSolver(rtol=1e-10, atol=1e-10, max_steps=max_steps, forward_step=step)


def test_dual_time_converges_without_flow() -> None:
    """The dual-time step converges a nonlinear root using only an injected scalar shift policy."""
    theta = jnp.array([8.0, 27.0, 64.0])
    step = DualTimeStep(
        UniformShiftPolicy(strength=1.0),
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0),
        inner_steps=4,
    )
    phi = _solver(step).solve(_residual, jnp.ones_like(theta), theta)

    assert jnp.all(jnp.isfinite(phi))
    assert jnp.allclose(phi, jnp.cbrt(theta), atol=1e-6)


def test_dual_time_is_differentiable() -> None:
    """Reverse-mode gradient through the converged dual-time solve matches the closed form."""
    theta = jnp.array([8.0])
    step = DualTimeStep(
        UniformShiftPolicy(strength=1.0),
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0),
        inner_steps=4,
    )

    def solved_sum(t: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(_solver(step).solve(_residual, jnp.ones_like(t), t))

    grad = jax.grad(solved_sum)(theta)

    # d/dtheta cbrt(theta) = (1/3) theta^(-2/3); the IFT adjoint is independent of the shift and of the
    # inner/outer iteration structure.
    assert jnp.allclose(grad, (1.0 / 3.0) * theta ** (-2.0 / 3.0), atol=1e-6)


def test_dual_time_gradient_is_iteration_count_independent() -> None:
    """The adjoint is the transpose solve at the root, so it does not depend on how the march got there.

    Varying the outer step cap and the inner-loop depth changes the forward path but not the converged
    root, so the gradient must be identical -- the signature of an implicit-function-theorem adjoint
    rather than an unrolled iteration.
    """
    theta = jnp.array([8.0, 27.0])

    def grad_with(inner_steps: int, max_steps: int) -> jnp.ndarray:
        step = DualTimeStep(
            UniformShiftPolicy(strength=1.0),
            relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0),
            inner_steps=inner_steps,
        )

        def solved_sum(t: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(_solver(step, max_steps=max_steps).solve(_residual, jnp.ones_like(t), t))

        return jax.grad(solved_sum)(theta)

    reference = grad_with(inner_steps=2, max_steps=200)
    assert jnp.allclose(grad_with(inner_steps=5, max_steps=200), reference, atol=1e-8)
    assert jnp.allclose(grad_with(inner_steps=2, max_steps=500), reference, atol=1e-8)


def test_dual_time_inner_loop_iterates_and_sums_cost() -> None:
    """inner_steps > 1 runs multiple inner Newton steps: it converges G further and sums the cycles.

    Guards the per-step signals a StepControl reads: a broken accumulation (cycles overwritten instead
    of summed, or the inner loop not iterating) would still pass the convergence/gradient tests, which
    only check the root. Here the shared anchor is phi0 and beta = 1 (the residual ratio is 1 at the
    anchor), so the transient residual is G(p) = R(p) + (p - phi0); more inner iterations must drive it
    lower and cost more linear-solve cycles.
    """
    theta = jnp.array([8.0, 27.0, 64.0])
    schedule = SwitchedEvolutionRelaxation(beta0=1.0)
    policy = UniformShiftPolicy(strength=1.0)
    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(_residual(phi0, theta))

    def residual_theta(p: jnp.ndarray) -> jnp.ndarray:
        return _residual(p, theta)

    def run(inner_steps: int):
        step = DualTimeStep(
            policy, relaxation_schedule=schedule, inner_steps=inner_steps, inner_tol=1e-6
        )
        return step.stepper()(residual_theta, phi0, r0, step.default_solver())

    def gnorm(p: jnp.ndarray) -> float:
        return float(jnp.linalg.norm(_residual(p, theta) + (p - phi0)))

    phi1, cyc1 = (lambda o: (o.phi, o.cycles))(run(1))
    phi4, cyc4 = (lambda o: (o.phi, o.cycles))(run(4))
    assert gnorm(phi4) < gnorm(phi1)  # more inner iterations converge the implicit step further
    assert int(cyc4) > int(cyc1)  # cycles are summed over the inner iterations, not overwritten


def test_dual_time_inner_observer_surfaces_the_trajectory_without_changing_the_step() -> None:
    """The opt-in inner_observer fires once per inner iteration with its G-norm before/after, cycle
    count and line-search factor -- and setting it leaves the computed step byte-identical."""
    theta = jnp.array([8.0, 27.0, 64.0])
    schedule = SwitchedEvolutionRelaxation(beta0=1.0)
    policy = UniformShiftPolicy(strength=1.0)
    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(_residual(phi0, theta))

    def residual_theta(p: jnp.ndarray) -> jnp.ndarray:
        return _residual(p, theta)

    records: list[tuple[int, float, float, int, float]] = []

    def observer(inner, g_before, g_after, cycles, alpha) -> None:
        records.append((int(inner), float(g_before), float(g_after), int(cycles), float(alpha)))

    observed = DualTimeStep(
        policy, relaxation_schedule=schedule, inner_steps=4, inner_tol=1e-8, inner_observer=observer
    )
    plain = DualTimeStep(policy, relaxation_schedule=schedule, inner_steps=4, inner_tol=1e-8)
    outcome = observed.stepper()(residual_theta, phi0, r0, observed.default_solver())
    phi_obs, n_inner = outcome.phi, outcome.inner_iterations
    phi_obs.block_until_ready()  # flush the ordered debug callbacks
    phi_plain = plain.stepper()(residual_theta, phi0, r0, plain.default_solver()).phi

    assert len(records) == int(n_inner) >= 1  # one record per inner iteration
    assert [r[0] for r in records] == list(range(len(records)))  # indices 0,1,2,... in order
    assert records[-1][2] < records[0][1]  # G-norm falls across the inner loop
    assert jnp.allclose(phi_obs, phi_plain)  # the observer does not perturb the step


def test_dual_time_one_inner_step_is_a_single_shifted_step() -> None:
    """With one inner step the dual-time step is exactly one shifted Newton step.

    At the anchor the transient term is zero, so ``G(phi^n) = R(phi^n)`` and one inner solve is the
    same attempt ``PseudoTransientStep`` forms. Compared against a pseudo-transient step with its
    escalation disabled and a permissive guard (so it, too, takes its raw first attempt), the single
    inner iterate must match. Both strategies must also converge to the same root -- their overshoot
    handling differs (inner loop vs escalation) but the fixed point does not.
    """
    theta = jnp.array([8.0, 27.0, 64.0])
    schedule = SwitchedEvolutionRelaxation(beta0=1.0)
    policy = UniformShiftPolicy(strength=1.0)

    dual = DualTimeStep(policy, relaxation_schedule=schedule, inner_steps=1, line_search=0)
    # No escalation and a permissive guard: PseudoTransientStep then takes its raw first attempt, which
    # is the same single shifted step DualTimeStep takes when the inner loop is one step and unclipped.
    raw_shifted = PseudoTransientStep(
        policy,
        relaxation_schedule=schedule,
        line_search=0,
        max_escalations=0,
        acceptance=DivergenceGuard(divergence_cap=1e12),
    )

    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(_residual(phi0, theta))

    def residual_theta(p: jnp.ndarray) -> jnp.ndarray:
        return _residual(p, theta)

    dual_next = dual.stepper()(residual_theta, phi0, r0, dual.default_solver()).phi
    shifted_next = raw_shifted.stepper()(residual_theta, phi0, r0, raw_shifted.default_solver()).phi
    assert jnp.allclose(dual_next, shifted_next, atol=1e-10)

    # And the full dual-time march reaches the same root as the escalating pseudo-transient march.
    pseudo = PseudoTransientStep(policy, relaxation_schedule=schedule)
    root_dual = _solver(dual).solve(_residual, phi0, theta)
    root_pseudo = ImplicitNewtonSolver(
        rtol=1e-10, atol=1e-10, max_steps=200, forward_step=pseudo
    ).solve(_residual, phi0, theta)
    assert jnp.allclose(root_dual, root_pseudo, atol=1e-8)


def test_dual_time_cycle_budget_caps_the_inner_loop() -> None:
    """A ``cycle_budget`` stops the inner loop once its accumulated linear-solve count reaches the budget.

    With ``inner_tol = 0`` the inner target is unreachable, so an unbounded step runs the full
    ``inner_steps`` and accumulates their summed solve count; the budgeted step instead stops after the
    first inner iteration that carries the running count to the budget -- the cost bailout that keeps a
    grinding primary solve from running every inner iteration into the restart cap. ``cycle_budget=None``
    leaves the step byte-identical.
    """
    theta = jnp.array([8.0, 27.0, 64.0])

    def residual_fn(phi: jnp.ndarray) -> jnp.ndarray:
        return _residual(phi, theta)

    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(residual_fn(phi0))
    common = dict(
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0), inner_steps=20, inner_tol=0.0
    )
    unbounded = DualTimeStep(UniformShiftPolicy(strength=1.0), **common)
    budgeted = DualTimeStep(UniformShiftPolicy(strength=1.0), cycle_budget=5, **common)

    out_u = unbounded.stepper()(residual_fn, phi0, r0, unbounded.default_solver())
    out_b = budgeted.stepper()(residual_fn, phi0, r0, budgeted.default_solver())
    cyc_u, inner_u = out_u.cycles, out_u.inner_iterations
    cyc_b, inner_b = out_b.cycles, out_b.inner_iterations

    assert int(inner_u) == 20  # unreachable target -> ran the full inner budget
    assert int(cyc_u) > 5  # accumulated more than the budget without a cap (sanity for the test)
    assert int(inner_b) < int(inner_u)  # the budget cut the loop short
    # Stops at the first inner iteration whose running total reaches the budget: at most one over-budget
    # solve's worth beyond it.
    assert 5 <= int(cyc_b) <= 5 + int(cyc_u) // int(inner_u) + 1


def test_dual_time_cycle_budget_none_is_the_unbounded_step() -> None:
    """The default ``cycle_budget=None`` reproduces the step with no budget field set, bit for bit."""
    theta = jnp.array([8.0, 27.0, 64.0])

    def residual_fn(phi: jnp.ndarray) -> jnp.ndarray:
        return _residual(phi, theta)

    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(residual_fn(phi0))
    common = dict(relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0), inner_steps=6)
    default = DualTimeStep(UniformShiftPolicy(strength=1.0), **common)
    explicit_none = DualTimeStep(UniformShiftPolicy(strength=1.0), cycle_budget=None, **common)

    a = default.stepper()(residual_fn, phi0, r0, default.default_solver())
    b = explicit_none.stepper()(residual_fn, phi0, r0, explicit_none.default_solver())
    assert jnp.allclose(a[0], b[0]) and int(a[1]) == int(b[1]) and int(a[3]) == int(b[3])


def test_a_cut_short_step_reports_that_it_did_not_reach_its_target() -> None:
    """A cost-only escalation cannot tell an expensive success from a grind, and would bin the success.

    ``max_inner_cycles`` is reported alongside because the summed count is not an inner-count-invariant
    difficulty signal: it grows with how many times the step solved.
    """
    theta = jnp.array([8.0, 27.0, 64.0])

    def residual_fn(phi: jnp.ndarray) -> jnp.ndarray:
        return _residual(phi, theta)

    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(residual_fn(phi0))
    common = dict(relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0), inner_steps=20)
    cut = DualTimeStep(UniformShiftPolicy(strength=1.0), inner_tol=0.0, cycle_budget=5, **common)
    met = DualTimeStep(UniformShiftPolicy(strength=1.0), inner_tol=1e-2, **common)

    cut_out = cut.stepper()(residual_fn, phi0, r0, cut.default_solver())
    met_out = met.stepper()(residual_fn, phi0, r0, met.default_solver())

    assert not bool(cut_out.reached_target)  # an unreachable target, stopped by the budget
    assert bool(met_out.reached_target)  # a loose target the inner loop actually met
    assert int(met_out.max_inner_cycles) <= int(met_out.cycles)  # one solve, not their sum


def test_positive_block_limit_keeps_a_constrained_field_off_zero() -> None:
    """Fraction-to-the-boundary: the largest step that keeps the block strictly positive.

    Rejecting violating rungs is not enough -- on the march that motivated this, the search was
    already at its shortest rung when the field crossed zero, so there was no shorter rung to take.
    Capping makes an admissible step reachable by construction.
    """
    limit = positive_block_limit(2, 5)
    phi = jnp.array([1.0, 1.0, 1.0e-5, 2.0, 3.0])
    delta = jnp.array([9.0, 9.0, -1.0e-3, 0.5, -1.0])

    alpha = limit(phi, delta)

    assert float(alpha) == pytest.approx(0.99 * 1.0e-5 / 1.0e-3)  # set by the one near-zero entry
    assert bool(((phi + alpha * delta)[2:5] > 0).all())
    # Entries outside the block are unconstrained: the cap ignores them however far they move.
    assert float(limit(phi, jnp.array([-1e3, 0.0, 1.0, 1.0, 1.0]))) == 1.0


def test_a_capped_line_search_never_exceeds_the_cap() -> None:
    """The cap applies to every rung, including the growth rungs above one."""
    phi, delta = jnp.array([1.0]), jnp.array([-1.0])
    stepped, alpha = backtracking_line_search(
        lambda p: p * 0.0, phi, delta, jnp.asarray(1.0), steps=4, grow=2, max_alpha=0.1
    )

    assert float(alpha) <= 0.1
    assert float(stepped[0]) >= 0.9  # phi - alpha, with alpha capped


def test_the_inner_line_search_honours_an_injected_step_limit() -> None:
    """A step that would drive the constrained block negative is shortened, not taken."""
    theta = jnp.array([8.0, 27.0, 64.0])

    def residual_fn(phi: jnp.ndarray) -> jnp.ndarray:
        return _residual(phi, theta)

    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(residual_fn(phi0))
    common = dict(relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0), inner_steps=3)
    free = DualTimeStep(UniformShiftPolicy(strength=1.0), **common)
    capped = DualTimeStep(
        UniformShiftPolicy(strength=1.0), step_limit=lambda p, d: jnp.asarray(0.01), **common
    )

    free_out = free.stepper()(residual_fn, phi0, r0, free.default_solver())
    capped_out = capped.stepper()(residual_fn, phi0, r0, capped.default_solver())

    assert float(capped_out.alpha) <= 0.01 < float(free_out.alpha)
    # The capped step still moves -- a cap shortens the step, it does not null it.
    assert not jnp.allclose(capped_out.phi, phi0)
