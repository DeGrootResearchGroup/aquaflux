"""Unit tests for the Newton driver, on analytic residuals (no operator imports)."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
from aquaflux.solve import NewtonSolver, solve_linear


def test_linear_residual_solved_in_one_iteration() -> None:
    """For a linear residual, a single Newton step is exact from any start."""
    a = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    b = jnp.array([1.0, 2.0])
    phi = NewtonSolver(iterations=1).solve(lambda x: a @ x - b, jnp.array([9.0, -9.0]))
    assert jnp.allclose(phi, jnp.linalg.solve(a, b), atol=1e-10)


def test_nonlinear_residual_converges() -> None:
    """A monotone nonlinear residual phi^3 + phi - b converges with a few iterations."""
    b = jnp.array([2.0, -5.0, 0.0])
    solver = NewtonSolver(iterations=30)
    phi = solver.solve(lambda x: x**3 + x - b, jnp.zeros(3))
    assert jnp.allclose(phi**3 + phi, b, atol=1e-9)


def test_solution_is_differentiable_through_parameter() -> None:
    """Gradient of the converged solution w.r.t. a residual parameter (implicit-diff solve)."""

    def solved_sum(k):
        a = jnp.array([[k, 1.0], [1.0, 3.0]])
        phi = NewtonSolver(iterations=1).solve(
            lambda x: a @ x - jnp.array([1.0, 2.0]), jnp.zeros(2)
        )
        return jnp.sum(phi)

    g = jax.grad(solved_sum)(4.0)
    fd = (solved_sum(4.0 + 1e-4) - solved_sum(4.0 - 1e-4)) / 2e-4
    assert abs(float(g) - float(fd)) < 1e-6


def test_solve_linear_matches_dense() -> None:
    """The matrix-free linear solve agrees with a dense solve."""
    a = jnp.array([[5.0, 2.0, 0.0], [2.0, 4.0, 1.0], [0.0, 1.0, 3.0]])
    b = jnp.array([1.0, -2.0, 0.5])
    assert jnp.allclose(solve_linear(lambda x: a @ x, b), jnp.linalg.solve(a, b), atol=1e-9)
