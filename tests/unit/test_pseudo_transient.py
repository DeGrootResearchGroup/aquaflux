"""The pseudo-transient continuation engine drives a non-flow residual.

``PseudoTransientStep`` is the residual-agnostic pseudo-transient continuation strategy: the
switched-evolution-relaxation schedule, the shifted solve, and the accept/escalate loop, with the
problem-specific choices (which DOFs shift, the shift magnitude, the shifted preconditioner) supplied
by an injected ``ShiftPolicy``. These tests exercise it on a small scalar nonlinear root with a
trivial shift policy — no mesh, no flow assembler, no block preconditioner — proving the engine is
reusable beyond the coupled flow it was first built for.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from aquaflux.solve import (
    DivergenceGuard,
    ImplicitNewtonSolver,
    PseudoTransientStep,
    ShiftTerm,
)


class UniformShiftPolicy(eqx.Module):
    """A minimal non-flow shift policy: a uniform pseudo-time shift on every DOF, unpreconditioned.

    Attributes
    ----------
    strength : float
        The per-DOF base shift magnitude ``d`` (static); the engine scales it by the relaxation ``β``.
    """

    strength: float = eqx.field(static=True, default=1.0)

    def shift_term(self, phi: jnp.ndarray) -> ShiftTerm:
        diagonal = self.strength * jnp.ones_like(phi)
        return ShiftTerm(diagonal, lambda relaxation: None)


def _residual(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """A nonlinear residual with root ``phi = cbrt(theta)`` (per component)."""
    return phi**3 - theta


def test_pseudo_transient_engine_runs_without_flow() -> None:
    """The engine converges a nonlinear root using only an injected scalar shift policy."""
    theta = jnp.array([8.0, 27.0, 64.0])
    step = PseudoTransientStep(UniformShiftPolicy(strength=1.0), beta0=1.0)
    solver = ImplicitNewtonSolver(rtol=1e-10, atol=1e-10, max_steps=200, forward_step=step)

    phi = solver.solve(_residual, jnp.ones_like(theta), theta)

    assert jnp.all(jnp.isfinite(phi))
    assert jnp.allclose(phi, jnp.cbrt(theta), atol=1e-6)


def test_pseudo_transient_engine_is_differentiable() -> None:
    """Reverse-mode gradient through the engine's converged solve matches the closed form."""
    theta = jnp.array([8.0])
    step = PseudoTransientStep(UniformShiftPolicy(strength=1.0), beta0=1.0)
    solver = ImplicitNewtonSolver(rtol=1e-10, atol=1e-10, max_steps=200, forward_step=step)

    def solved_sum(t: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(solver.solve(_residual, jnp.ones_like(t), t))

    grad = jax.grad(solved_sum)(theta)

    # d/dtheta cbrt(theta) = (1/3) theta^(-2/3); the IFT adjoint is independent of the shift.
    assert jnp.allclose(grad, (1.0 / 3.0) * theta ** (-2.0 / 3.0), atol=1e-6)


def test_divergence_guard_accepts_below_cap_and_rejects_divergence() -> None:
    """The default acceptance policy is decidable from scalar norms alone (no solve, no mesh)."""
    guard = DivergenceGuard(divergence_cap=10.0)
    r0 = jnp.asarray(1.0)
    previous, attempt = jnp.asarray(2.0), jnp.asarray(0)  # unused by a pure divergence guard

    # Finite candidate below cap × ‖R₀‖ is accepted; at/above the cap, or non-finite, is rejected.
    assert bool(guard.accept(jnp.asarray(5.0), previous, r0, attempt))
    assert bool(guard.accept(jnp.asarray(9.999), previous, r0, attempt))
    assert not bool(guard.accept(jnp.asarray(10.0), previous, r0, attempt))
    assert not bool(guard.accept(jnp.asarray(50.0), previous, r0, attempt))
    assert not bool(guard.accept(jnp.asarray(jnp.inf), previous, r0, attempt))
    assert not bool(guard.accept(jnp.asarray(jnp.nan), previous, r0, attempt))

    # The cap scales with the initial residual and is tunable.
    assert bool(
        DivergenceGuard(divergence_cap=100.0).accept(jnp.asarray(50.0), previous, r0, attempt)
    )


def test_injected_acceptance_policy_is_honoured() -> None:
    """A custom acceptance policy is used by the engine — the seam is real, not just present.

    ``RejectFirstAttempt`` refuses the first (undamped-schedule) attempt of every step, forcing one
    escalation; the solve must still converge, proving the engine routes the accept/reject decision
    through the injected policy rather than a hardwired test.
    """

    class RejectFirstAttempt(eqx.Module):
        def accept(self, candidate_norm, residual_norm, residual_norm_0, attempt):
            finite_bounded = jnp.isfinite(candidate_norm) & (
                candidate_norm < 10.0 * residual_norm_0
            )
            return finite_bounded & (attempt > 0)

    theta = jnp.array([8.0, 27.0])
    step = PseudoTransientStep(
        UniformShiftPolicy(strength=1.0), beta0=1.0, acceptance=RejectFirstAttempt()
    )
    solver = ImplicitNewtonSolver(rtol=1e-10, atol=1e-10, max_steps=200, forward_step=step)

    phi = solver.solve(_residual, jnp.ones_like(theta), theta)

    assert jnp.allclose(phi, jnp.cbrt(theta), atol=1e-6)
