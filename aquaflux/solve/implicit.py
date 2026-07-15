"""Nonlinear Newton solve with an implicit-function-theorem (IFT) adjoint.

For a genuinely nonlinear residual (e.g. a flux-limited advection scheme) Newton takes many
iterations, and differentiating through the unrolled iterations would tape every step. Instead
the converged state ``phi*(theta)`` — defined implicitly by ``R(phi*, theta) = 0`` — is
differentiated by the **implicit function theorem**:

    dphi*/dtheta = -(dR/dphi)^{-1} (dR/dtheta),

so the reverse-mode gradient of a loss ``L(phi*)`` with cotangent ``v = dL/dphi*`` is

    dL/dtheta = -(dR/dtheta)^T lambda,   where   (dR/dphi)^T lambda = v.

This is **one transpose linear solve**, independent of the iteration count — no Newton loop is
placed on the tape. The forward iteration may therefore use a data-dependent stopping criterion
(``lax.while_loop``); the custom VJP supplies the derivative in its place.

The adjoint is defined only for reverse mode (``jax.grad`` / ``jax.vjp``), which is what a
scalar objective through the solver needs.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from .linear import solve_linear
from .newton import newton_step


def _forward(residual_fn, phi0, theta, rtol, atol, max_steps, solver):
    """Newton iterate to convergence (``lax.while_loop``); returns the converged field."""
    residual_norm_0 = jnp.linalg.norm(residual_fn(phi0, theta))

    def cond(carry):
        _, step, residual_norm = carry
        return (step < max_steps) & (residual_norm > atol + rtol * residual_norm_0)

    def body(carry):
        phi, step, _ = carry
        phi = newton_step(lambda p: residual_fn(p, theta), phi, solver=solver)
        return phi, step + 1, jnp.linalg.norm(residual_fn(phi, theta))

    phi, _, _ = jax.lax.while_loop(cond, body, (phi0, 0, residual_norm_0))
    return phi


@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4, 5, 6))
def _implicit_solve(residual_fn, phi0, theta, rtol, atol, max_steps, solver):
    return _forward(residual_fn, phi0, theta, rtol, atol, max_steps, solver)


def _implicit_solve_fwd(residual_fn, phi0, theta, rtol, atol, max_steps, solver):
    phi_star = _forward(residual_fn, phi0, theta, rtol, atol, max_steps, solver)
    return phi_star, (phi_star, theta)


def _implicit_solve_bwd(residual_fn, rtol, atol, max_steps, solver, residuals, cotangent):
    phi_star, theta = residuals
    # Transpose Jacobian solve: (dR/dphi)^T lambda = cotangent.
    _, vjp_phi = jax.vjp(lambda p: residual_fn(p, theta), phi_star)
    lam = solve_linear(lambda u: vjp_phi(u)[0], cotangent, solver=solver)
    # Parameter cotangent -(dR/dtheta)^T lambda: negate lambda so no pytree (float0) negation.
    _, vjp_theta = jax.vjp(lambda th: residual_fn(phi_star, th), theta)
    (theta_cotangent,) = vjp_theta(-lam)
    return jnp.zeros_like(phi_star), theta_cotangent


_implicit_solve.defvjp(_implicit_solve_fwd, _implicit_solve_bwd)


class ImplicitNewtonSolver(eqx.Module):
    """Newton solve to convergence with a reverse-mode IFT adjoint.

    Use for nonlinear residuals where the forward iteration count is data-dependent and the
    gradient must not unroll it. The residual is passed as ``residual_fn(phi, theta)`` with the
    differentiable parameters ``theta`` explicit, so the adjoint can return their cotangents.

    Attributes
    ----------
    rtol, atol : float
        Relative / absolute stopping tolerances on the residual norm (static).
    max_steps : int
        Maximum Newton iterations (static).
    solver : lineax.AbstractLinearSolver or None
        Linear solver for the Newton and adjoint solves; ``None`` uses the package default.
    """

    rtol: float = eqx.field(static=True, default=1e-10)
    atol: float = eqx.field(static=True, default=1e-12)
    max_steps: int = eqx.field(static=True, default=50)
    solver: lx.AbstractLinearSolver | None = None

    def solve(
        self,
        residual_fn: Callable[[jnp.ndarray, object], jnp.ndarray],
        phi0: jnp.ndarray,
        theta: object,
    ) -> jnp.ndarray:
        """Solve ``residual_fn(phi, theta) = 0``; reverse-differentiable in ``theta`` by IFT.

        Parameters
        ----------
        residual_fn : callable
            Maps ``(phi, theta)`` to the residual of shape ``(n_cells,)``.
        phi0 : jnp.ndarray
            Initial guess, shape ``(n_cells,)``.
        theta : pytree
            Differentiable parameters the residual depends on.

        Returns
        -------
        jnp.ndarray
            The converged field, shape ``(n_cells,)``.
        """
        return _implicit_solve(
            residual_fn, phi0, theta, self.rtol, self.atol, self.max_steps, self.solver
        )
