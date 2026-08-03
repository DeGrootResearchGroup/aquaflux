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
from aquaflux.solve import (
    DivergenceGuard,
    DualTimeStep,
    ImplicitNewtonSolver,
    PseudoTransientStep,
    ShiftTerm,
    SwitchedEvolutionRelaxation,
)


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

    phi1, cyc1, _, _ = run(1)
    phi4, cyc4, _, _ = run(4)
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
    phi_obs, _, _, n_inner = observed.stepper()(residual_theta, phi0, r0, observed.default_solver())
    phi_obs.block_until_ready()  # flush the ordered debug callbacks
    phi_plain, _, _, _ = plain.stepper()(residual_theta, phi0, r0, plain.default_solver())

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

    dual_next, _, _, _ = dual.stepper()(residual_theta, phi0, r0, dual.default_solver())
    shifted_next, _, _, _ = raw_shifted.stepper()(
        residual_theta, phi0, r0, raw_shifted.default_solver()
    )
    assert jnp.allclose(dual_next, shifted_next, atol=1e-10)

    # And the full dual-time march reaches the same root as the escalating pseudo-transient march.
    pseudo = PseudoTransientStep(policy, relaxation_schedule=schedule)
    root_dual = _solver(dual).solve(_residual, phi0, theta)
    root_pseudo = ImplicitNewtonSolver(
        rtol=1e-10, atol=1e-10, max_steps=200, forward_step=pseudo
    ).solve(_residual, phi0, theta)
    assert jnp.allclose(root_dual, root_pseudo, atol=1e-8)
