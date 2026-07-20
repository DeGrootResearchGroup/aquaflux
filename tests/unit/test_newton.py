"""Unit tests for the Newton correction, on analytic residuals (no operator imports).

``newton_step`` is one matrix-free correction, not a driver: it is *exact in a single call* for a
linear residual, which is what the linear paths (transient diffusion, Stokes flow, the Laplace
initializer) rely on. Because it is plain traced operations rather than a ``custom_vjp``, it also
differentiates in **both** modes -- the forward-mode ``jacfwd`` that a scalar-parameter sensitivity
over a whole field wants, as well as reverse-mode. Iterating a nonlinear residual to convergence is
:class:`~aquaflux.solve.ImplicitNewtonSolver`'s job and is tested in ``test_implicit_solve.py``.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
from aquaflux.solve import newton_step, solve_linear

A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
B = jnp.array([1.0, 2.0])


def test_linear_residual_solved_in_one_step() -> None:
    """For a linear residual, a single Newton correction is exact from any start."""
    phi = newton_step(lambda x: A @ x - B, jnp.array([9.0, -9.0]))
    assert jnp.allclose(phi, jnp.linalg.solve(A, B), atol=1e-10)


def _solved_sum(k):
    """The solved field's sum as a function of a residual parameter."""
    a = jnp.array([[k, 1.0], [1.0, 3.0]])
    return jnp.sum(newton_step(lambda x: a @ x - B, jnp.zeros(2)))


def test_reverse_mode_differentiable_through_the_step() -> None:
    """``jax.grad`` through the exact linear solve matches central finite differences."""
    g = jax.grad(_solved_sum)(4.0)
    fd = (_solved_sum(4.0 + 1e-4) - _solved_sum(4.0 - 1e-4)) / 2e-4
    assert abs(float(g) - float(fd)) < 1e-6


def test_forward_mode_differentiable_through_the_step() -> None:
    """``jacfwd`` also works -- one linear solve per input, the efficient direction for a scalar
    parameter against a whole field, and the mode a ``custom_vjp`` solver cannot offer."""

    def solved(k):
        a = jnp.array([[k, 1.0], [1.0, 3.0]])
        return newton_step(lambda x: a @ x - B, jnp.zeros(2))

    forward = jax.jacfwd(solved)(4.0)
    step = 1e-5
    fd = (solved(4.0 + step) - solved(4.0 - step)) / (2.0 * step)
    assert jnp.allclose(forward, fd, atol=1e-6)


def test_newton_correction_returns_the_step_and_the_residual() -> None:
    """``newton_correction`` exposes ``(delta, R(phi))`` so a line search need not recompute R."""
    from aquaflux.solve.newton import newton_correction

    phi = jnp.array([9.0, -9.0])
    delta, r = newton_correction(lambda x: A @ x - B, phi)
    assert jnp.allclose(r, A @ phi - B, atol=1e-12)
    assert jnp.allclose(phi + delta, jnp.linalg.solve(A, B), atol=1e-10)


def test_solve_linear_matches_dense() -> None:
    """The matrix-free linear solve agrees with a dense solve."""
    a = jnp.array([[5.0, 2.0, 0.0], [2.0, 4.0, 1.0], [0.0, 1.0, 3.0]])
    b = jnp.array([1.0, -2.0, 0.5])
    assert jnp.allclose(solve_linear(lambda x: a @ x, b), jnp.linalg.solve(a, b), atol=1e-9)
