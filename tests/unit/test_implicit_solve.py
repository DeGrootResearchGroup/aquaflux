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


def test_ift_gradient_is_iteration_count_independent() -> None:
    """Loosening the forward tolerance (more/fewer steps) does not change the adjoint — it is a
    single transpose solve at the converged state, not an unrolled loop."""
    theta = jnp.array([1.5])
    tight = jax.grad(
        lambda th: jnp.sum(ImplicitNewtonSolver(atol=1e-14).solve(_residual, jnp.zeros(1), th))
    )(theta)
    loose = jax.grad(
        lambda th: jnp.sum(ImplicitNewtonSolver(atol=1e-10).solve(_residual, jnp.zeros(1), th))
    )(theta)
    assert jnp.allclose(tight, loose, atol=1e-8)


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
    """A NaN/Inf residual makes ``residual_norm > tol`` evaluate ``False`` and exits the loop at
    once; the finiteness guard must turn that into a hard error rather than a poisoned field."""
    # Pure full Newton (line_search=0): the backtracking search would otherwise recover by
    # shrinking the step back into the domain, so disable it to reach the non-finite iterate. One
    # step lands at x < 0, so the loop exits on the step count with a non-finite residual norm — the
    # finiteness guard turns that into the hard error (before any further linear solve on the NaN).
    solver = ImplicitNewtonSolver(max_steps=1, forward_step=DampedNewtonStep(line_search=0))
    with pytest.raises(eqx.EquinoxRuntimeError, match="did not converge"):
        solver.solve(_sqrt_residual, jnp.array([4.0]), jnp.array([-1.0])).block_until_ready()


def test_non_convergence_raises_on_the_grad_path_too() -> None:
    """The whole point: the silently-wrong output is a *gradient*, so the guard must also fire when
    the solve is reached only through ``jax.grad`` (the backward pass linearizes the non-root)."""
    theta = jnp.array([50.0])
    solver = ImplicitNewtonSolver(max_steps=2)
    with pytest.raises(eqx.EquinoxRuntimeError, match="did not converge"):
        jax.grad(lambda th: jnp.sum(solver.solve(_residual, jnp.zeros(1), th)))(
            theta
        ).block_until_ready()
