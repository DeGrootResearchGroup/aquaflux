"""Unit tests for the implicit-function-theorem nonlinear solver.

Exercised on analytic nonlinear roots (no operators), so the IFT adjoint is checked against a
closed-form derivative — the seam the solve rule requires.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
from aquaflux.solve import ImplicitNewtonSolver


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
