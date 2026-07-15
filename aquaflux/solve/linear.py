"""Differentiable matrix-free linear solve, with optional left preconditioning.

A thin wrapper over ``lineax`` that solves ``A x = b`` given only a matrix-vector product
``matvec(x) = A x`` — never a materialized matrix. ``lineax`` differentiates the solve by
**implicit differentiation** (it differentiates the solution of ``A x = b`` directly, rather
than unrolling the iterative solver onto the tape), so a gradient taken through
:func:`solve_linear` costs one extra solve and is independent of the iteration count. This is
the linear-solve primitive the Newton driver and the gradient schemes build on.

An optional **preconditioner** ``M`` (a matvec approximating ``A^{-1}``) is applied on the
left: the solver is handed ``M A`` and ``M b`` instead of ``A`` and ``b``. Since the solution
of ``(M A) x = M b`` is exactly ``A^{-1} b``, and ``M``'s coefficients are treated as constant
(the caller ``stop_gradient``s them), preconditioning changes only the Krylov convergence, not
the solution or its gradient — it is implicit-diff-transparent.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import lineax as lx


def default_linear_solver() -> lx.AbstractLinearSolver:
    """A general-purpose matrix-free solver (restarted GMRES) with tight tolerances."""
    return lx.GMRES(rtol=1e-10, atol=1e-10)


def solve_linear(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    solver: lx.AbstractLinearSolver | None = None,
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Solve ``A x = b`` for ``x`` given the linear map ``matvec(x) = A x``.

    Parameters
    ----------
    matvec : callable
        The linear operator, mapping ``x`` of shape ``b.shape`` to ``A x`` of the same shape.
        Must be linear in its argument.
    b : jnp.ndarray
        Right-hand side.
    solver : lineax.AbstractLinearSolver, optional
        The linear solver; defaults to :func:`default_linear_solver`.
    preconditioner : callable, optional
        A left preconditioner ``M`` (a matvec approximating ``A^{-1}``). The solver is handed
        ``x -> M(A(x))`` and ``M(b)``. ``M``'s internal coefficients must be constant with
        respect to any outer differentiation (``stop_gradient``-ed by the caller), so that
        preconditioning accelerates convergence without perturbing the solution or its gradient.

    Returns
    -------
    jnp.ndarray
        The solution ``x``, of shape ``b.shape``.
    """
    if solver is None:
        solver = default_linear_solver()
    if preconditioner is None:
        preconditioned_matvec, rhs = matvec, b
    else:

        def preconditioned_matvec(x):
            return preconditioner(matvec(x))

        rhs = preconditioner(b)
    operator = lx.FunctionLinearOperator(
        preconditioned_matvec, jax.ShapeDtypeStruct(b.shape, b.dtype)
    )
    return lx.linear_solve(operator, rhs, solver=solver).value
