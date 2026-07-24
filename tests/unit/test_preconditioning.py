"""Unit tests for the linear-solve preconditioning seam.

Left preconditioning must be **transparent**: it accelerates the Krylov iteration but changes
neither the converged solution nor its gradient. (Its convergence acceleration is exercised on
the real coupled-flow system, where the unpreconditioned block solve genuinely scales badly —
a synthetic matrix is a poor and fragile proxy for that.)
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from aquaflux.solve import solve_linear


def _system(theta):
    """A well-posed dense system A(theta) x = b with a widely spread diagonal."""
    n = 40
    rng = np.random.default_rng(0)
    a = jnp.diag(jnp.asarray(np.logspace(0.0, 3.0, n))) + 0.05 * jnp.asarray(
        rng.standard_normal((n, n))
    )
    a = a + theta * jnp.eye(n)
    b = jnp.asarray(rng.standard_normal(n))
    return a, b


def _jacobi(a):
    inverse_diagonal = 1.0 / jax.lax.stop_gradient(jnp.diag(a))
    return lambda v: inverse_diagonal * v


def test_preconditioner_does_not_change_solution() -> None:
    a, b = _system(2.0)
    plain, _ = solve_linear(lambda v: a @ v, b)
    preconditioned, _ = solve_linear(lambda v: a @ v, b, preconditioner=_jacobi(a))
    assert jnp.allclose(plain, preconditioned, atol=1e-8)


def test_preconditioner_is_gradient_transparent() -> None:
    """The gradient of a functional of the solution is identical with and without the preconditioner."""

    def loss(theta, use_preconditioner):
        a, b = _system(theta)
        precond = _jacobi(a) if use_preconditioner else None
        return jnp.sum(solve_linear(lambda v: a @ v, b, preconditioner=precond)[0])

    plain = jax.grad(lambda t: loss(t, False))(2.0)
    preconditioned = jax.grad(lambda t: loss(t, True))(2.0)
    assert jnp.allclose(plain, preconditioned, atol=1e-8)


def test_solve_returns_the_solution_and_a_positive_cycle_count() -> None:
    """``solve_linear`` returns ``(x, cycles)``: the solution, and what the solve cost."""
    a, b = _system(2.0)
    value, cycles = solve_linear(lambda v: a @ v, b)
    assert jnp.allclose(value, jnp.linalg.solve(a, b), atol=1e-8)
    assert int(cycles) > 0
    # Pinned dtype: a caller carries this through a lax.while_loop, whose carry must be invariant.
    assert cycles.dtype == jnp.int32


def test_counted_solve_reports_zero_for_a_direct_solver() -> None:
    """A solver that reports no iteration count (a direct factorization) yields 0, not an error."""
    a, b = _system(2.0)
    _, cycles = solve_linear(lambda v: a @ v, b, solver=lx.AutoLinearSolver(well_posed=True))
    assert int(cycles) == 0


def test_the_cycle_count_falls_when_the_preconditioner_improves() -> None:
    """The count measures how hard the *preconditioned* system was -- the staleness signal.

    This is the property a mid-march preconditioner refresh triggers on: on a fixed system the count
    rises as the frozen preconditioner drifts from the operator, and falls when it matches it again.
    Here the same system is solved with and without a matching Jacobi preconditioner; the
    well-preconditioned solve must take strictly fewer iterations, or the count would be measuring
    nothing useful.
    """
    a, b = _system(2.0)
    _, plain = solve_linear(lambda v: a @ v, b)
    _, preconditioned = solve_linear(lambda v: a @ v, b, preconditioner=_jacobi(a))
    assert int(preconditioned) < int(plain)
