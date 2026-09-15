"""Unit tests for the implicit-function-theorem nonlinear solver.

Exercised on analytic nonlinear roots (no operators), so the IFT adjoint is checked against a
closed-form derivative — the seam the solve rule requires.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.solve import ImplicitNewtonSolver
from aquaflux.solve.forward_step import within_tolerance
from aquaflux.solve.implicit import DampedNewtonStep


def _residual(x, theta):
    """x^3 + x - theta = 0; root x*(theta) with dx*/dtheta = 1/(3 x*^2 + 1)."""
    return x**3 + x - theta


def test_converges_to_nonlinear_root() -> None:
    theta = jnp.array([2.0, -5.0, 0.3])
    x = ImplicitNewtonSolver().solve(_residual, jnp.zeros(3), theta)
    assert jnp.allclose(_residual(x, theta), 0.0, atol=1e-9)


def test_ift_gradient_matches_closed_form() -> None:
    """Reverse-mode gradient through the converged root equals the analytical derivative."""
    theta = jnp.array([2.0, -5.0, 0.3])
    solver = ImplicitNewtonSolver()
    x_star = solver.solve(_residual, jnp.zeros(3), theta)
    grad = jax.grad(lambda th: jnp.sum(solver.solve(_residual, jnp.zeros(3), th)))(theta)
    analytic = 1.0 / (3.0 * x_star**2 + 1.0)
    assert jnp.allclose(grad, analytic, atol=1e-10)


def _newton_steps_taken(solver, residual_fn, phi0, theta):
    """Directly measure how many Newton iterations ``solver`` actually takes to reach ``theta``'s
    root from ``phi0`` -- an undifferentiated replay of ``_forward``'s own loop.

    ``_forward`` computes this count internally (the ``step`` carried alongside the iterate) but
    discards it before returning, and the count is data-dependent (``lax.while_loop``), so it
    cannot be read back through a traced ``jax.grad`` call. This calls the same production step
    function (``forward_step.stepper()``) and the same stopping test (``within_tolerance``) in a
    plain, eager Python loop instead, which is the only way to observe the real trip count.
    """
    forward_step_fn = solver.forward_step.stepper()
    lin_solver = (
        solver.solver if solver.solver is not None else solver.forward_step.default_solver()
    )
    norm_fn = solver.forward_step.norm()
    residual_norm_0 = norm_fn(residual_fn(phi0, theta))
    phi, residual_norm, step = phi0, residual_norm_0, 0
    while step < solver.max_steps and not within_tolerance(
        residual_norm, residual_norm_0, solver.rtol, solver.atol
    ):
        outcome = forward_step_fn(lambda p: residual_fn(p, theta), phi, residual_norm_0, lin_solver)
        phi, residual_norm = outcome.phi, outcome.residual_norm
        step += 1
    return step, phi


def test_ift_gradient_is_iteration_count_independent() -> None:
    """Two Newton paths that reach the same root by a different number of steps must give the SAME
    gradient -- the property that separates the implicit-function-theorem adjoint from the forward
    iteration unrolled onto the tape, and matching finite differences does not establish it: a
    fully unrolled march would also match, at the root it happened to reach, while carrying a
    gradient that depends on how it got there.

    Loosening ``atol`` does not exercise this: Newton's quadratic convergence blows through both a
    tight and a loose tolerance in the same 5 iterations at this fixture (``theta = 1.5`` from
    ``phi0 = 0``), so that comparison compares one path against itself. Varying the initial guess
    instead genuinely changes the number of steps Newton takes to the same root, which is the
    distinguishing experiment -- the same lever `test_the_coupled_adjoint_is_independent_of_the_
    forward_iteration_count` uses (there, ``inner_steps``) for the identical property on the
    coupled march.

    The step counts are measured directly, in a separate, undifferentiated run, and asserted to
    differ before the gradients are ever compared -- without that, a test that happened to vary
    nothing real would pass no matter what the adjoint did.
    """
    theta = jnp.array([1.5])
    solver = ImplicitNewtonSolver()
    near, far = jnp.zeros(1), jnp.array([50.0])

    steps_near, root_near = _newton_steps_taken(solver, _residual, near, theta)
    steps_far, root_far = _newton_steps_taken(solver, _residual, far, theta)
    assert steps_near != steps_far, (
        f"both starting points took {steps_near} Newton iterations, so this test compares one "
        "path against itself and cannot detect a taped adjoint"
    )
    # The paths converge to the same root despite taking different numbers of steps to get there.
    assert jnp.allclose(root_near, root_far, atol=1e-8)

    grad_near = jax.grad(lambda th: jnp.sum(solver.solve(_residual, near, th)))(theta)
    grad_far = jax.grad(lambda th: jnp.sum(solver.solve(_residual, far, th)))(theta)
    assert jnp.allclose(grad_near, grad_far, atol=1e-8)


def _sqrt_residual(x, theta):
    """sqrt(x) - theta = 0. A full Newton step from a moderate x with theta < 0 overshoots to
    x < 0, so the next residual is non-finite — a deterministic mid-iteration NaN."""
    return jnp.sqrt(x) - theta


def test_non_convergence_within_max_steps_raises_instead_of_returning_a_poisoned_field() -> None:
    """Exhausting ``max_steps`` short of tolerance must raise, not silently return a non-root whose
    implicit-function-theorem adjoint would be a wrong gradient with no NaN to flag it."""
    theta = jnp.array([50.0])  # Newton from 0 needs many steps; two is far short of the root
    solver = ImplicitNewtonSolver(max_steps=2)
    with pytest.raises(eqx.EquinoxRuntimeError, match="did not converge"):
        solver.solve(_residual, jnp.zeros(1), theta).block_until_ready()


def test_non_finite_residual_raises_instead_of_exiting_silently() -> None:
    """Pins the NaN case: ``within_tolerance``'s ``<=`` comparison already evaluates ``False`` for a
    NaN residual norm on its own (NaN comparisons are always false), so this exercises the guard's
    redundant-but-harmless overlap with that, not its unique purpose. The genuinely load-bearing
    case -- a residual norm *and* its own threshold both diverging to ``+inf``, where ``inf <= inf``
    is ``True`` and only ``jnp.isfinite`` still catches it -- is
    ``test_a_residual_and_its_threshold_that_both_diverge_to_infinity_still_raises`` below."""
    # Pure full Newton (line_search=0): the backtracking search would otherwise recover by
    # shrinking the step back into the domain, so disable it to reach the non-finite iterate. One
    # step lands at x < 0, so the loop exits on the step count with a non-finite residual norm — the
    # finiteness guard turns that into the hard error (before any further linear solve on the NaN).
    solver = ImplicitNewtonSolver(max_steps=1, forward_step=DampedNewtonStep(line_search=0))
    with pytest.raises(eqx.EquinoxRuntimeError, match="did not converge"):
        solver.solve(_sqrt_residual, jnp.array([4.0]), jnp.array([-1.0])).block_until_ready()


def test_a_residual_and_its_threshold_that_both_diverge_to_infinity_still_raises() -> None:
    """``within_tolerance`` alone cannot catch this case: it compares with ``<=``, and when the
    residual norm AND its own convergence threshold have both diverged to ``+inf``,
    ``inf <= inf`` evaluates ``True`` -- a false "converged". Only the explicit
    ``jnp.isfinite(residual_norm)`` guard turns this into an error instead.

    A residual that is already infinite at ``phi0`` makes ``residual_norm_0`` (and hence the
    threshold ``atol + rtol * residual_norm_0``) infinite before the loop runs even once, so the
    loop's own stopping test is satisfied trivially and it exits immediately at ``step = 0`` with a
    non-finite residual norm -- exactly the state the isfinite guard exists to reject.
    """

    def _infinite_residual(x, theta):
        # 1/x at x=0 overflows to +inf. Written as a genuine function of x (not a bare `jnp.inf`
        # fill unrelated to x) so the Newton correction's right-hand side stays a value the trace
        # carries through the loop rather than one that folds to a compile-time constant.
        del theta
        return 1.0 / x

    solver = ImplicitNewtonSolver()
    with pytest.raises(eqx.EquinoxRuntimeError, match="did not converge"):
        solver.solve(_infinite_residual, jnp.zeros(1), jnp.array([1.0])).block_until_ready()


def test_non_convergence_raises_on_the_grad_path_too() -> None:
    """The whole point: the silently-wrong output is a *gradient*, so the guard must also fire when
    the solve is reached only through ``jax.grad`` (the backward pass linearizes the non-root)."""
    theta = jnp.array([50.0])
    solver = ImplicitNewtonSolver(max_steps=2)
    with pytest.raises(eqx.EquinoxRuntimeError, match="did not converge"):
        jax.grad(lambda th: jnp.sum(solver.solve(_residual, jnp.zeros(1), th)))(
            theta
        ).block_until_ready()
