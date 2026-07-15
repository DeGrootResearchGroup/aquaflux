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
    plain = solve_linear(lambda v: a @ v, b)
    preconditioned = solve_linear(lambda v: a @ v, b, preconditioner=_jacobi(a))
    assert jnp.allclose(plain, preconditioned, atol=1e-8)


def test_preconditioner_is_gradient_transparent() -> None:
    """The gradient of a functional of the solution is identical with and without the preconditioner."""

    def loss(theta, use_preconditioner):
        a, b = _system(theta)
        precond = _jacobi(a) if use_preconditioner else None
        return jnp.sum(solve_linear(lambda v: a @ v, b, preconditioner=precond))

    plain = jax.grad(lambda t: loss(t, False))(2.0)
    preconditioned = jax.grad(lambda t: loss(t, True))(2.0)
    assert jnp.allclose(plain, preconditioned, atol=1e-8)
