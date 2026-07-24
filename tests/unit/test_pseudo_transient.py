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
import pytest
from aquaflux.solve import (
    DivergenceGuard,
    ImplicitNewtonSolver,
    PseudoTransientStep,
    ShiftTerm,
    SwitchedEvolutionRelaxation,
)
from aquaflux.solve.implicit import backtracking_line_search


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
    step = PseudoTransientStep(
        UniformShiftPolicy(strength=1.0), relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0)
    )
    solver = ImplicitNewtonSolver(rtol=1e-10, atol=1e-10, max_steps=200, forward_step=step)

    phi = solver.solve(_residual, jnp.ones_like(theta), theta)

    assert jnp.all(jnp.isfinite(phi))
    assert jnp.allclose(phi, jnp.cbrt(theta), atol=1e-6)


def test_pseudo_transient_engine_is_differentiable() -> None:
    """Reverse-mode gradient through the engine's converged solve matches the closed form."""
    theta = jnp.array([8.0])
    step = PseudoTransientStep(
        UniformShiftPolicy(strength=1.0), relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0)
    )
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
        UniformShiftPolicy(strength=1.0),
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0),
        acceptance=RejectFirstAttempt(),
    )
    solver = ImplicitNewtonSolver(rtol=1e-10, atol=1e-10, max_steps=200, forward_step=step)

    phi = solver.solve(_residual, jnp.ones_like(theta), theta)

    assert jnp.allclose(phi, jnp.cbrt(theta), atol=1e-6)


def test_backtracking_line_search_picks_largest_descending_rung() -> None:
    """The shared backtracking helper keeps the largest step length that reduces the residual, and
    falls back to the smallest rung when none does. Physics-free: ``R(x) = x`` so ``||R|| = |x|``."""
    residual = lambda x: x  # noqa: E731
    phi = jnp.array([1.0])
    reference = jnp.asarray(1.0)  # ||R(phi)||

    # delta = -4: full step x = -3 (|R| = 3, overshoot); alpha = 1/2 -> x = -1 (|R| = 1, not < 1);
    # alpha = 1/4 -> x = 0 (|R| = 0 < 1). Largest descending rung is 1/4.
    out, alpha = backtracking_line_search(residual, phi, jnp.array([-4.0]), reference, steps=4)
    assert jnp.allclose(out, 0.0)
    assert jnp.allclose(alpha, 0.25)  # the kept fraction is reported

    # steps = 0 takes the full (overshooting) step unchanged, and reports alpha = 1.
    full, full_alpha = backtracking_line_search(
        residual, phi, jnp.array([-4.0]), reference, steps=0
    )
    assert jnp.allclose(full, -3.0)
    assert jnp.allclose(full_alpha, 1.0)

    # delta = +4: every rung increases the residual, so it falls back to the smallest, 1/16.
    fallback, fb_alpha = backtracking_line_search(
        residual, phi, jnp.array([4.0]), reference, steps=4
    )
    assert jnp.allclose(fallback, 1.0 + (1.0 / 16.0) * 4.0)
    assert jnp.allclose(fb_alpha, 1.0 / 16.0)


def test_line_search_recovers_an_overshooting_step_without_escalation() -> None:
    """With the escalation fallback disabled, the line search alone rescues a step whose full shifted
    correction overshoots -- the stiff-first-step regime the coupled RANS solve hits.

    From ``phi = 1`` toward the root ``phi = 10`` (``theta = 1000``) with only a weak shift, the full
    Newton correction lands near ``phi ~ 334`` and the cubic residual explodes. A backtracking search
    scales it back to a descent; without it (and without escalation) the step is rejected every
    iteration and the solve never converges.
    """
    theta = jnp.array([1000.0])
    policy = UniformShiftPolicy(strength=1.0)

    searched = ImplicitNewtonSolver(
        rtol=1e-8,
        atol=1e-10,
        max_steps=200,
        forward_step=PseudoTransientStep(
            policy,
            relaxation_schedule=SwitchedEvolutionRelaxation(beta0=0.01),
            max_escalations=0,
            line_search=40,
        ),
    )
    phi = searched.solve(_residual, jnp.ones_like(theta), theta)
    assert jnp.allclose(phi, jnp.cbrt(theta), atol=1e-5)

    # No line search and no escalation: the overshoot is never tamed, so the solve cannot converge.
    unsearched = ImplicitNewtonSolver(
        rtol=1e-8,
        atol=1e-10,
        max_steps=50,
        forward_step=PseudoTransientStep(
            policy,
            relaxation_schedule=SwitchedEvolutionRelaxation(beta0=0.01),
            max_escalations=0,
            line_search=0,
        ),
    )
    with pytest.raises(Exception):  # noqa: B017  (EquinoxRuntimeError, raised at solve time)
        jax.block_until_ready(unsearched.solve(_residual, jnp.ones_like(theta), theta))


def test_stepper_returns_the_step_and_its_linear_solve_cycle_count() -> None:
    """``stepper()`` returns ``(phi_next, cycles)`` -- the step, and what its shifted solve cost.

    The count is the cost of the *accepted* attempt's shifted solve, the signal an observed march
    watches to decide a frozen preconditioner has gone stale. There is one stepper: a caller with no
    use for the count drops it, rather than there being a second count-free method to drift from.
    """
    theta = jnp.array([8.0, 27.0, 64.0])
    phi0 = jnp.ones_like(theta)
    step = PseudoTransientStep(
        UniformShiftPolicy(strength=1.0), relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0)
    )
    residual_norm_0 = jnp.linalg.norm(_residual(phi0, theta))
    solver = step.default_solver()

    def residual_fn(phi):
        return _residual(phi, theta)

    phi_next, cycles, alpha = step.stepper()(residual_fn, phi0, residual_norm_0, solver)

    # A real shifted solve was taken: the iterate moved, and stayed finite. Deliberately not a
    # descent assertion -- the pseudo-transient march is non-monotone (which is why its acceptance
    # policy is a divergence guard rather than a descent test), so one step need not reduce ‖R‖.
    assert not jnp.allclose(phi_next, phi0)
    assert bool(jnp.all(jnp.isfinite(phi_next)))
    assert int(cycles) > 0
    assert cycles.dtype == jnp.int32  # invariant carry dtype for a lax.while_loop
    # This step has no line search (default line_search=0), so the full shifted step is taken: alpha=1.
    assert jnp.allclose(alpha, 1.0)
