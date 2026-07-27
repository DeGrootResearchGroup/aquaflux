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

    dual_next, _, _ = dual.stepper()(residual_theta, phi0, r0, dual.default_solver())
    shifted_next, _, _ = raw_shifted.stepper()(
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
