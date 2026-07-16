"""Newton driver on the cell residual.

Solves ``R(phi) = 0`` by Newton's method: at each iterate the correction ``delta`` solves the
linearized system ``J delta = -R(phi)``, where ``J = dR/dphi`` is applied **matrix-free** via
a forward-mode directional derivative (``jax.jvp``) — the Jacobian is never assembled, and no
hand-derived linearization coefficients exist. The linear solve is the differentiable
:func:`~aquaflux.solve.linear.solve_linear`.

The residual is supplied as a closure ``residual_fn(phi)``, so the driver is testable on any
analytic residual and never imports a specific operator. For a linear residual (e.g. transient
diffusion) a single iteration is exact; ``iterations`` is fixed and small, and the driver is
differentiated directly through those unrolled steps. (A convergence-based stop with an
implicit-function-theorem adjoint is the upgrade for genuinely nonlinear residuals, where many
iterations would otherwise be taped; it is not needed while the physics is linear.)
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
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
    :func:`~aquaflux.solve.linear.solve_linear`. Shared by :class:`NewtonSolver` and the
    implicit-function-theorem solver.

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
    r = residual_fn(phi)

    def jacobian_vector_product(v, _phi=phi):
        return jax.jvp(residual_fn, (_phi,), (v,))[1]

    preconditioner_matvec = None if preconditioner is None else preconditioner(phi)
    return phi + solve_linear(
        jacobian_vector_product, -r, solver=solver, preconditioner=preconditioner_matvec
    )


class NewtonSolver(eqx.Module):
    """Fixed-iteration Newton solver with a matrix-free, differentiable linear step.

    The Newton step is compiled **once** (with the iterate as the only traced argument) and the
    compiled graph is reused across iterations. Iterating an un-jitted :func:`newton_step` rebuilds
    the Jacobian-vector-product operator and the preconditioner as fresh closures every step, which
    forces XLA to recompile the whole preconditioned linear solve each iteration — the dominant cost
    of the naive loop.

    Attributes
    ----------
    iterations : int
        Number of Newton iterations (static). One is exact for a linear residual.
    solver : lineax.AbstractLinearSolver or None
        The linear solver for each Newton step; ``None`` uses the package default.
    preconditioner : callable or None
        A factory ``phi -> M`` giving the left preconditioner ``M`` (a matvec approximating
        ``J^{-1}``) for each step's linear solve, built at the current iterate (e.g.
        :meth:`aquaflux.flow.BlockPreconditioner.factory`). ``None`` solves unpreconditioned. Static
        (its coefficients are ``stop_gradient``-ed by the factory, so it stays gradient-transparent).
    """

    iterations: int = eqx.field(static=True, default=1)
    solver: lx.AbstractLinearSolver | None = None
    preconditioner: Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]] | None = (
        eqx.field(static=True, default=None)
    )

    def solve(
        self,
        residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
        phi0: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve ``residual_fn(phi) = 0`` starting from ``phi0``.

        Parameters
        ----------
        residual_fn : callable
            Maps a cell field ``phi`` of shape ``(n_cells,)`` to the residual, same shape.
        phi0 : jnp.ndarray
            Initial guess, shape ``(n_cells,)``.

        Returns
        -------
        jnp.ndarray
            The converged field, shape ``(n_cells,)``.
        """

        @eqx.filter_jit
        def step(phi: jnp.ndarray) -> jnp.ndarray:
            return newton_step(
                residual_fn, phi, solver=self.solver, preconditioner=self.preconditioner
            )

        phi = phi0
        for _ in range(self.iterations):
            phi = step(phi)
        return phi
