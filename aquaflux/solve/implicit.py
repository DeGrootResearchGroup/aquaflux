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
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from .linear import default_linear_solver, solve_linear
from .newton import newton_correction

# The step a continuation strategy supplies: given the (single-argument) residual, the current
# iterate, the starting residual norm, and the linear solver, return the next iterate.
_ContinuationStep = Callable[
    [Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray, jnp.ndarray, lx.AbstractLinearSolver],
    jnp.ndarray,
]


class Continuation(Protocol):
    """A globalization strategy that replaces the forward Newton step (e.g. pseudo-transient).

    Structural interface only, so the generic solver stays free of any flow specifics. A concrete
    continuation (e.g. :class:`aquaflux.flow.PseudoTransientContinuation`) supplies the per-step
    :meth:`stepper` used in the forward loop and an :meth:`adjoint_preconditioner` for the converged
    transpose solve.
    """

    def stepper(self) -> _ContinuationStep:
        """The forward step ``(residual_fn, phi, residual_norm_0, solver) -> phi_next``."""

    def adjoint_preconditioner(
        self,
    ) -> Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
        """The ``state -> M`` preconditioner factory for the adjoint (transpose) solve."""


# Inexact-Newton forward solver: each Newton step's linear solve need only make Newton progress,
# not be exact — the next step corrects the leftover. A loose relative tolerance cuts the GMRES
# matvec count per step several-fold; the few extra Newton steps it costs still net a large
# speedup, and the converged state is unchanged (the outer loop drives the residual to the
# nonlinear tolerance regardless of how accurately each inner solve was taken). The adjoint solve,
# by contrast, is taken once at the converged state and sets the gradient accuracy directly, so it
# defaults to the tight :func:`default_linear_solver`.
_INEXACT_FORWARD_SOLVER = lx.GMRES(rtol=1e-3, atol=1e-3)

# Inexact-Newton forward solver for a continuation march: same loose relative tolerance, but a
# *tight* absolute floor and generous restart/stagnation budget. Continuation drives the residual
# far below the ``1e-3`` absolute floor the plain inexact solver uses, and once ``‖R‖`` nears that
# floor the linear solve would stop taking a step and the outer march would stall short of the
# nonlinear tolerance — so the absolute term must not cap the terminal convergence. The looser
# stagnation budget also rides out the stiffer shifted operators a graded, high-Reynolds mesh
# produces.
_INEXACT_CONTINUATION_SOLVER = lx.GMRES(rtol=1e-3, atol=1e-10, restart=40, stagnation_iters=40)


def _damped_newton_step(residual_fn, phi, solver, preconditioner, line_search_steps):
    """One Newton step with a monotone backtracking line search on the residual norm.

    ``line_search_steps == 0`` recovers the undamped full step ``phi + delta``. Otherwise the step
    length ``alpha`` is halved (up to ``line_search_steps`` times) until
    ``||R(phi + alpha delta)|| < ||R(phi)||`` — the globalization a convection-dominated open flow
    needs, where the full Newton step from a uniform field overshoots and diverges. A full step is
    kept unchanged whenever it already reduces the residual, so a well-behaved iterate (near the
    root, or a linear residual) is unaffected. The search only reshapes the forward path; the IFT
    adjoint depends solely on the converged state, so it stays gradient-transparent.
    """
    delta, r = newton_correction(residual_fn, phi, solver=solver, preconditioner=preconditioner)
    if line_search_steps == 0:
        return phi + delta
    r_norm = jnp.linalg.norm(r)

    # Backtracking over a fixed ladder alpha in {1, 1/2, ..., 1/2**line_search_steps}: keep the
    # *largest* rung that reduces the residual (locked in by ``accepted``), falling back to the
    # smallest if none does. A fixed (unrolled) count keeps the step a constant-length operation, so
    # it composes with the ``while_loop`` and the IFT ``custom_vjp`` without a data-dependent branch.
    alpha = 1.0
    chosen = 0.5**line_search_steps  # smallest rung, used if nothing reduces the residual
    accepted = jnp.asarray(False)
    for _ in range(line_search_steps + 1):
        reduces = (jnp.linalg.norm(residual_fn(phi + alpha * delta)) < r_norm) & ~accepted
        chosen = jnp.where(reduces, alpha, chosen)
        accepted = accepted | reduces
        alpha = 0.5 * alpha
    return phi + chosen * delta


def _forward(
    residual_fn,
    phi0,
    theta,
    rtol,
    atol,
    max_steps,
    solver,
    preconditioner,
    line_search,
    continuation_step,
):
    """Newton iterate to convergence (``lax.while_loop``); returns the converged field.

    With ``continuation_step`` set, each iteration is the pseudo-transient (diagonally shifted) step
    it supplies instead of the line-searched Newton step — the globalization a convection-dominated
    open flow needs at high Reynolds number, where the block preconditioner would otherwise stall
    the inner GMRES. The shift vanishes at the fixed point, so the converged field solves the same
    unshifted ``R(phi, theta) = 0`` and the stopping test is unchanged.
    """
    residual_norm_0 = jnp.linalg.norm(residual_fn(phi0, theta))

    def cond(carry):
        _, step, residual_norm = carry
        return (step < max_steps) & (residual_norm > atol + rtol * residual_norm_0)

    def body(carry):
        phi, step, _ = carry

        def residual_theta(p):
            return residual_fn(p, theta)

        if continuation_step is None:
            phi = _damped_newton_step(residual_theta, phi, solver, preconditioner, line_search)
        else:
            phi = continuation_step(residual_theta, phi, residual_norm_0, solver)
        return phi, step + 1, jnp.linalg.norm(residual_fn(phi, theta))

    phi, _, _ = jax.lax.while_loop(cond, body, (phi0, 0, residual_norm_0))
    return phi


@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4, 5, 6, 7, 8, 9, 10))
def _implicit_solve(
    residual_fn,
    phi0,
    theta,
    rtol,
    atol,
    max_steps,
    solver,
    adjoint_solver,
    preconditioner,
    line_search,
    continuation_step,
):
    return _forward(
        residual_fn,
        phi0,
        theta,
        rtol,
        atol,
        max_steps,
        solver,
        preconditioner,
        line_search,
        continuation_step,
    )


def _implicit_solve_fwd(
    residual_fn,
    phi0,
    theta,
    rtol,
    atol,
    max_steps,
    solver,
    adjoint_solver,
    preconditioner,
    line_search,
    continuation_step,
):
    phi_star = _forward(
        residual_fn,
        phi0,
        theta,
        rtol,
        atol,
        max_steps,
        solver,
        preconditioner,
        line_search,
        continuation_step,
    )
    return phi_star, (phi_star, theta)


def _adjoint_preconditioner(preconditioner, phi_star, example):
    """Transpose ``M^T`` of the forward preconditioner, as a left preconditioner for the adjoint.

    The forward ``M = preconditioner(phi*)`` approximates ``J^{-1}``; the adjoint solves the
    transpose system ``J^T lambda = v``, whose consistent left preconditioner is ``M^T ~ J^{-T}``,
    obtained by transposing the (linear) preconditioner matvec with :func:`jax.linear_transpose`.
    It is mesh-independent wherever ``M`` is -- the adjoint GMRES iteration count stays flat under
    refinement instead of growing with the system size. ``None`` in, ``None`` out.
    """
    if preconditioner is None:
        return None
    m = preconditioner(phi_star)
    transpose = jax.linear_transpose(m, example)
    return lambda u: transpose(u)[0]


def _implicit_solve_bwd(
    residual_fn,
    rtol,
    atol,
    max_steps,
    solver,
    adjoint_solver,
    preconditioner,
    line_search,
    continuation_step,
    residuals,
    cotangent,
):
    phi_star, theta = residuals
    # Transpose Jacobian solve: (dR/dphi)^T lambda = cotangent, left-preconditioned by M^T so the
    # adjoint solve is mesh-independent (unpreconditioned it grows with the system size). This solve
    # sets the gradient accuracy, so it uses the (tight) adjoint solver, not the inexact forward one.
    _, vjp_phi = jax.vjp(lambda p: residual_fn(p, theta), phi_star)
    adjoint_precond = _adjoint_preconditioner(preconditioner, phi_star, cotangent)
    lam = solve_linear(
        lambda u: vjp_phi(u)[0], cotangent, solver=adjoint_solver, preconditioner=adjoint_precond
    )
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
        Linear solver for the forward Newton steps. ``None`` uses an **inexact-Newton** default (a
        loose-tolerance GMRES): each step's solve need only make Newton progress, since the next
        step corrects it, which cuts the matvec count per step several-fold. The converged state is
        unaffected — the loop still drives the residual to ``rtol``/``atol``. When ``continuation``
        is set, the ``None`` default instead uses a variant with a tight *absolute* floor, so the
        pseudo-transient march is not capped short of the nonlinear tolerance near convergence.
    adjoint_solver : lineax.AbstractLinearSolver or None
        Linear solver for the adjoint (transpose) solve. ``None`` uses the tight
        :func:`default_linear_solver`, because this single solve at the converged state sets the
        gradient accuracy directly and should not be loosened along with the forward steps.
    preconditioner : callable or None
        A factory ``phi -> M`` giving the left preconditioner ``M`` (a matvec approximating
        ``J^{-1}``), built at the current iterate (e.g.
        :meth:`aquaflux.flow.BlockPreconditioner.factory`). Used for **both** the forward Newton
        solves and, transposed to ``M^T``, the adjoint (transpose) solve, so gradients are
        mesh-independent too. ``None`` solves unpreconditioned — usable only on small or
        well-conditioned systems; the coupled flow saddle-point needs one. Static.
    line_search : int
        Maximum step-halvings in the forward step's backtracking line search (static). Each Newton
        step takes the largest ``alpha in {1, 1/2, 1/4, ...}`` (up to this many halvings) that
        reduces the residual norm — the globalization a convection-dominated open flow needs, where
        the undamped full step overshoots from a uniform initial field and diverges. A full step is
        kept whenever it already reduces the residual, so a well-behaved or linear solve is
        unchanged. ``0`` disables it (pure full Newton). The IFT adjoint is unaffected either way.
        Ignored when ``continuation`` is set (which supplies its own globalization).
    continuation : Continuation or None
        A continuation strategy (e.g. :class:`aquaflux.flow.PseudoTransientContinuation`) whose
        pseudo-transient step replaces the line-searched Newton step in the forward loop (static).
        This is the globalization a *high-Reynolds* convective flow needs — the line search alone
        leaves the inner GMRES stalling on the block preconditioner once convection dominates. Its
        diagonal shift vanishes at the fixed point, so the converged state and the IFT adjoint are
        the same as without it. When set and ``preconditioner`` is ``None``, the adjoint solve uses
        the continuation's own (unshifted) preconditioner. ``None`` keeps the plain line-searched
        Newton loop.
    """

    rtol: float = eqx.field(static=True, default=1e-10)
    atol: float = eqx.field(static=True, default=1e-12)
    max_steps: int = eqx.field(static=True, default=50)
    solver: lx.AbstractLinearSolver | None = None
    adjoint_solver: lx.AbstractLinearSolver | None = None
    preconditioner: Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]] | None = (
        eqx.field(static=True, default=None)
    )
    line_search: int = eqx.field(static=True, default=10)
    continuation: Continuation | None = None

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
        if self.solver is not None:
            solver = self.solver
        elif self.continuation is not None:
            solver = _INEXACT_CONTINUATION_SOLVER  # tight absolute floor for the terminal march
        else:
            solver = _INEXACT_FORWARD_SOLVER
        adjoint_solver = (
            self.adjoint_solver if self.adjoint_solver is not None else default_linear_solver()
        )
        continuation_step = None if self.continuation is None else self.continuation.stepper()
        # The adjoint solve (at the converged, unshifted state) reuses the continuation's own
        # preconditioner when the caller supplies only a continuation, so a high-Re solve needs just
        # one strategy for both the forward globalization and the mesh-independent adjoint.
        preconditioner = self.preconditioner
        if preconditioner is None and self.continuation is not None:
            preconditioner = self.continuation.adjoint_preconditioner()
        return _implicit_solve(
            residual_fn,
            phi0,
            theta,
            self.rtol,
            self.atol,
            self.max_steps,
            solver,
            adjoint_solver,
            preconditioner,
            self.line_search,
            continuation_step,
        )
