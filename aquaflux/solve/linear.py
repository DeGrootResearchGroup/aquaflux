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
    *,
    throw: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve ``A x = b`` for ``x`` given the linear map ``matvec(x) = A x``, with the solve's cost.

    Returns the solution **and** the solver's reported iteration count. The count is the *cost* of
    the solve rather than part of its result — how hard the preconditioned system was this time — so
    a caller that only wants the answer drops it at the call site (``x, _ = solve_linear(...)``).
    There is deliberately no count-free variant to wrap this one: a second entry point would mean a
    second signature and a second parameter docstring to keep in step.

    **The count is restart cycles, not matrix-vector products (binding — the easy misreading).** For a
    restarted GMRES ``stats["num_steps"]`` counts *cycles*, each of which is up to ``restart``
    matvecs, so a "17" is ~17x``restart`` matvecs. A solver that reports no iteration count (a direct
    factorization) yields ``0``.

    **Why the count is worth returning:** a frozen preconditioner going stale shows up first as a
    *rising cycle count* on an otherwise-unchanged system, well before it shows up in the residual
    history. That makes the cycle count the honest trigger for re-freezing the preconditioner
    mid-march — and a robust one, unlike wall-clock time, which a suspended or loaded machine
    perturbs without the linear algebra having changed at all.

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
    throw : bool
        If ``True`` (default), a non-convergent solve raises. If ``False``, it instead returns the
        solver's last iterate without raising — for a caller that tests the result and recovers (an
        adaptive continuation that escalates damping when the shifted solve fails to converge). The
        returned iterate may not solve the system; the caller must check it.

    Returns
    -------
    x : jnp.ndarray
        The solution, of shape ``b.shape``.
    cycles : jnp.ndarray
        The solver's iteration count (restart **cycles** for a restarted GMRES), an ``int32`` scalar.
        The dtype is pinned so a caller can carry it through a ``lax.while_loop`` whose carry
        structure must be invariant (the escalation loop in the pseudo-transient step does).
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
    solution = lx.linear_solve(operator, rhs, solver=solver, throw=throw)
    return solution.value, jnp.asarray(solution.stats.get("num_steps", 0), dtype=jnp.int32)
