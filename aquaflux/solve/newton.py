"""The Newton correction on the cell residual.

One Newton step takes ``phi`` to ``phi + delta``, where the correction solves the linearized system
``J delta = -R(phi)`` and ``J = dR/dphi`` is applied **matrix-free** via a forward-mode directional
derivative (``jax.jvp``) — the Jacobian is never assembled, and no hand-derived linearization
coefficients exist. The linear solve is the differentiable
:func:`~aquaflux.solve.linear.solve_linear`.

The residual is supplied as a closure ``residual_fn(phi)``, so these functions are testable on any
analytic residual and never import a specific operator.

**This module is one step, not a driver.** For a *linear* residual — transient diffusion, Stokes
flow, a Laplace initializer — a single :func:`newton_step` is exact, and being plain traced
operations it differentiates in **both** modes: the forward-mode ``jacfwd`` that a scalar-parameter
sensitivity over a whole field wants, as well as reverse-mode. For a *nonlinear* residual, iterate
with :class:`~aquaflux.solve.ImplicitNewtonSolver`, which stops on a convergence test, globalizes
the march, and carries the implicit-function-theorem adjoint. Do **not** write a fixed-count loop
over :func:`newton_step` for a nonlinear residual: it cannot tell convergence from exhaustion, and
taping the unrolled steps is exactly the gradient path the two-level implicit differentiation exists
to avoid.

Neither function jits internally — the caller owns the jit boundary, so a step composes into
whatever the caller compiles. Wrap the call in ``equinox.filter_jit``; un-jitted, every operation
dispatches eagerly.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import lineax as lx

from .linear import solve_linear


def newton_step(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    phi: jnp.ndarray,
    solver: lx.AbstractLinearSolver | None = None,
    preconditioner: Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]] | None = None,
) -> jnp.ndarray:
    """One Newton correction ``phi -> phi + delta``, ``J delta = -R(phi)``.

    The Jacobian ``J = dR/dphi`` is applied matrix-free via a forward-mode directional
    derivative (``jax.jvp``); the linear solve is the differentiable
    :func:`~aquaflux.solve.linear.solve_linear`. Shared with the implicit-function-theorem solver.
    A globalized driver instead takes the raw correction from :func:`newton_correction` and damps it
    with a line search.

    Exact in one call for a linear residual. For a nonlinear one, iterate with
    :class:`~aquaflux.solve.ImplicitNewtonSolver` rather than calling this a fixed number of times
    (see the module docstring).

    Parameters
    ----------
    residual_fn : callable
        Maps ``phi`` of shape ``(n_cells,)`` to the residual, same shape.
    phi : jnp.ndarray
        Current iterate, shape ``(n_cells,)``.
    solver : lineax.AbstractLinearSolver, optional
        Linear solver for the Newton step; defaults to the package default.
    preconditioner : callable, optional
        A factory ``phi -> M`` giving the left preconditioner ``M`` (a matvec approximating
        ``J^{-1}``) for the step's linear solve; built at the current iterate. ``M``'s
        coefficients must be ``stop_gradient``-ed so preconditioning stays gradient-transparent.
    """
    delta, _ = newton_correction(residual_fn, phi, solver=solver, preconditioner=preconditioner)
    return phi + delta


def newton_correction(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    phi: jnp.ndarray,
    solver: lx.AbstractLinearSolver | None = None,
    preconditioner: Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The raw Newton correction ``delta`` (solving ``J delta = -R(phi)``) and the residual ``R(phi)``.

    Separated from :func:`newton_step` so a globalized driver can damp the step (a line search on
    ``phi + alpha delta``) using the same matrix-free, differentiable solve. Returns ``(delta, r)``
    with ``r = R(phi)`` so a caller doing a line search need not recompute it.
    """
    r = residual_fn(phi)

    def jacobian_vector_product(v, _phi=phi):
        return jax.jvp(residual_fn, (_phi,), (v,))[1]

    preconditioner_matvec = None if preconditioner is None else preconditioner(phi)
    delta = solve_linear(
        jacobian_vector_product, -r, solver=solver, preconditioner=preconditioner_matvec
    )
    return delta, r
