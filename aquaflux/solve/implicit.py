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

# The step a forward-step strategy supplies: given the (single-argument) residual, the current
# iterate, the starting residual norm, and the linear solver, return the next iterate.
_ForwardStep = Callable[
    [Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray, jnp.ndarray, lx.AbstractLinearSolver],
    jnp.ndarray,
]


class ForwardStep(Protocol):
    """A globalized Newton forward-step strategy (line search, pseudo-transient continuation, ...).

    The single point of variation in the forward loop: given the residual, the current iterate, the
    starting residual norm, and the linear solver, a strategy returns the next iterate. Every
    strategy must reduce to the undamped Newton step near the root and impose no shift at the fixed
    point, so the converged state solves the unshifted ``R = 0`` and the implicit-function-theorem
    adjoint is independent of which strategy produced the forward path.

    Structural interface only (a ``Protocol``), so the generic solver stays free of any flow
    specifics. The concrete strategies are :class:`DampedNewtonStep` (the default backtracking line
    search) and :class:`PseudoTransientStep` (the residual-agnostic pseudo-transient march;
    :func:`aquaflux.flow.momentum_continuation` configures it for the high-Reynolds flow).
    """

    def stepper(self) -> _ForwardStep:
        """The forward step ``(residual_fn, phi, residual_norm_0, solver) -> phi_next``."""

    def default_solver(self) -> lx.AbstractLinearSolver:
        """The forward-loop linear solver to use when the caller supplies none (an inexact-Newton
        default whose tolerances suit this strategy's march)."""

    def adjoint_preconditioner(
        self,
    ) -> Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]] | None:
        """The ``state -> M`` preconditioner factory for the adjoint (transpose) solve, or ``None``."""


# Inexact-Newton forward solver: each Newton step's linear solve need only make Newton progress,
# not be exact — the next step corrects the leftover. A loose relative tolerance cuts the GMRES
# matvec count per step several-fold; the few extra Newton steps it costs still net a large
# speedup, and the converged state is unchanged (the outer loop drives the residual to the
# nonlinear tolerance regardless of how accurately each inner solve was taken). The adjoint solve,
# by contrast, is taken once at the converged state and sets the gradient accuracy directly, so it
# defaults to the tight :func:`default_linear_solver`.
_INEXACT_FORWARD_SOLVER = lx.GMRES(rtol=1e-3, atol=1e-3)


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


class DampedNewtonStep(eqx.Module):
    """Backtracking-line-searched Newton step — the default forward-step strategy.

    Each step takes the largest ``alpha in {1, 1/2, ..., 2**-line_search}`` that reduces the
    residual norm (a full step whenever the full step already reduces it), so a well-behaved or
    linear solve runs undamped while a convection-dominated open flow — whose full Newton step from
    a uniform initial field overshoots and diverges — is damped back onto a descent path.
    ``line_search == 0`` is pure full Newton. The search only reshapes the forward path; the
    implicit-function-theorem adjoint depends solely on the converged state, so it is unaffected.

    Attributes
    ----------
    preconditioner : callable or None
        A factory ``phi -> M`` giving the left preconditioner ``M`` (a matvec approximating
        ``J^{-1}``) for each Newton step's linear solve, built at the current iterate (e.g.
        :meth:`aquaflux.flow.BlockPreconditioner.factory`). Used for the forward steps and,
        transposed, for the adjoint (transpose) solve, so gradients are mesh-independent too.
        ``None`` solves unpreconditioned — usable only on small or well-conditioned systems; the
        coupled flow saddle-point needs one. Static.
    line_search : int
        Maximum step-halvings in the backtracking line search (static). ``0`` disables it (pure
        full Newton).
    """

    preconditioner: Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]] | None = (
        eqx.field(static=True, default=None)
    )
    line_search: int = eqx.field(static=True, default=10)

    def stepper(self) -> _ForwardStep:
        """The line-searched Newton step ``(residual_fn, phi, ‖R₀‖, solver) -> phi_next``."""
        preconditioner = self.preconditioner
        line_search = self.line_search

        def step(residual_fn, phi, residual_norm_0, solver):
            # The starting norm is unused: each step's line search is decided from the residual at
            # the current iterate, not the initial one.
            del residual_norm_0
            return _damped_newton_step(residual_fn, phi, solver, preconditioner, line_search)

        return step

    def default_solver(self) -> lx.AbstractLinearSolver:
        """The inexact-Newton forward solver (loose relative tolerance; the next step corrects the
        leftover, cutting the matvec count per step several-fold with the converged state unchanged)."""
        return _INEXACT_FORWARD_SOLVER

    def adjoint_preconditioner(
        self,
    ) -> Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]] | None:
        """The forward preconditioner, reused (transposed) for the adjoint solve."""
        return self.preconditioner


def _within_tolerance(residual_norm, residual_norm_0, rtol, atol):
    """The Newton stopping test: the residual norm has dropped to the absolute/relative floor."""
    return residual_norm <= atol + rtol * residual_norm_0


def _forward(residual_fn, phi0, theta, rtol, atol, max_steps, solver, forward_step_fn):
    """Newton iterate to convergence (``lax.while_loop``); return the converged field or error.

    Each iteration applies the injected ``forward_step_fn`` — the globalized Newton step the
    :class:`ForwardStep` strategy supplies (a backtracking line search by default, a pseudo-transient
    continuation for a high-Reynolds convective flow). Every strategy's shift vanishes at the fixed
    point, so the converged field solves the same unshifted ``R(phi, theta) = 0`` and the stopping
    test is unchanged.

    The loop can exit without converging in two ways: it exhausts ``max_steps`` short of tolerance,
    or the residual norm becomes non-finite (``NaN``/``Inf``), which makes the ``residual_norm > tol``
    test ``False`` and exits at once. Both leave a field that does *not* solve ``R = 0``. The
    implicit-function-theorem adjoint linearizes the residual at whatever field this returns, so a
    non-converged field would yield a **silently wrong gradient** — the transpose solve is still
    well-posed and raises no ``NaN``. Guard against that here: if the terminal residual is non-finite
    or above tolerance, raise instead of returning a poisoned field, so neither the forward value nor
    the gradient built on it can be used unknowingly.
    """
    residual_norm_0 = jnp.linalg.norm(residual_fn(phi0, theta))

    def cond(carry):
        _, step, residual_norm = carry
        return (step < max_steps) & ~_within_tolerance(residual_norm, residual_norm_0, rtol, atol)

    def body(carry):
        phi, step, _ = carry

        def residual_theta(p):
            return residual_fn(p, theta)

        phi = forward_step_fn(residual_theta, phi, residual_norm_0, solver)
        return phi, step + 1, jnp.linalg.norm(residual_fn(phi, theta))

    phi, _, residual_norm = jax.lax.while_loop(cond, body, (phi0, 0, residual_norm_0))
    converged = jnp.isfinite(residual_norm) & _within_tolerance(
        residual_norm, residual_norm_0, rtol, atol
    )
    return eqx.error_if(
        phi,
        ~converged,
        "ImplicitNewtonSolver did not converge: the Newton residual norm did not reach "
        "atol + rtol*||R0|| within max_steps, or became non-finite. The implicit-function-theorem "
        "adjoint is only valid at a converged root, so the returned field and any gradient built on "
        "it would be silently wrong. Raise max_steps, loosen the tolerances, or use a stronger "
        "globalization (e.g. pseudo-transient continuation for a high-Reynolds flow).",
    )


@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4, 5, 6, 7, 8, 9))
def _implicit_solve(
    residual_fn,
    phi0,
    theta,
    rtol,
    atol,
    max_steps,
    solver,
    adjoint_solver,
    adjoint_preconditioner,
    forward_step_fn,
):
    return _forward(residual_fn, phi0, theta, rtol, atol, max_steps, solver, forward_step_fn)


def _implicit_solve_fwd(
    residual_fn,
    phi0,
    theta,
    rtol,
    atol,
    max_steps,
    solver,
    adjoint_solver,
    adjoint_preconditioner,
    forward_step_fn,
):
    phi_star = _forward(residual_fn, phi0, theta, rtol, atol, max_steps, solver, forward_step_fn)
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
    adjoint_preconditioner,
    forward_step_fn,
    residuals,
    cotangent,
):
    phi_star, theta = residuals
    # Transpose Jacobian solve: (dR/dphi)^T lambda = cotangent, left-preconditioned by M^T so the
    # adjoint solve is mesh-independent (unpreconditioned it grows with the system size). This solve
    # sets the gradient accuracy, so it uses the (tight) adjoint solver, not the inexact forward one.
    _, vjp_phi = jax.vjp(lambda p: residual_fn(p, theta), phi_star)
    adjoint_precond = _adjoint_preconditioner(adjoint_preconditioner, phi_star, cotangent)
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
        Linear solver for the forward Newton steps. ``None`` uses the forward-step strategy's own
        default (:meth:`ForwardStep.default_solver`) — an **inexact-Newton** GMRES whose tolerances
        suit that strategy's march (a loose relative tolerance for the line search, plus a tight
        *absolute* floor for the pseudo-transient continuation so its march is not capped short of
        the nonlinear tolerance near convergence). The converged state is unaffected — the loop
        still drives the residual to ``rtol``/``atol``.
    adjoint_solver : lineax.AbstractLinearSolver or None
        Linear solver for the adjoint (transpose) solve. ``None`` uses the tight
        :func:`default_linear_solver`, because this single solve at the converged state sets the
        gradient accuracy directly and should not be loosened along with the forward steps.
    forward_step : ForwardStep
        The globalized forward-step strategy that supplies each Newton iteration (static): a
        :class:`DampedNewtonStep` backtracking line search by default, or a :class:`PseudoTransientStep`
        (e.g. from :func:`aquaflux.flow.momentum_continuation`) for a high-Reynolds convective flow. The
        strategy also owns the forward preconditioner and, transposed, the adjoint preconditioner
        (via :meth:`ForwardStep.adjoint_preconditioner`), so gradients are mesh-independent too.
        Every strategy's shift vanishes at the fixed point, so the converged state and the IFT
        adjoint are the same regardless of which is used. Defaults to an unpreconditioned
        ``DampedNewtonStep`` — pass ``DampedNewtonStep(preconditioner=...)`` for a coupled flow,
        which needs one.
    """

    rtol: float = eqx.field(static=True, default=1e-10)
    atol: float = eqx.field(static=True, default=1e-12)
    max_steps: int = eqx.field(static=True, default=50)
    solver: lx.AbstractLinearSolver | None = None
    adjoint_solver: lx.AbstractLinearSolver | None = None
    forward_step: ForwardStep = eqx.field(default_factory=DampedNewtonStep)

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

        Raises
        ------
        equinox.EquinoxRuntimeError
            If the Newton iteration does not converge — it exhausts ``max_steps`` short of
            ``atol + rtol*||R0||`` or the residual norm becomes non-finite. The
            implicit-function-theorem adjoint is valid only at a converged root, so a
            non-converged field is rejected rather than returned (its gradient would be silently
            wrong). Raised at solve time, and equally on the ``jax.grad`` path.
        """
        forward = self.forward_step
        solver = self.solver if self.solver is not None else forward.default_solver()
        adjoint_solver = (
            self.adjoint_solver if self.adjoint_solver is not None else default_linear_solver()
        )
        # The strategy owns both the forward step and the adjoint preconditioner (the same
        # preconditioner it applies forward, transposed at the converged state), so a high-Re solve
        # needs a single strategy for both the forward globalization and the mesh-independent adjoint.
        return _implicit_solve(
            residual_fn,
            phi0,
            theta,
            self.rtol,
            self.atol,
            self.max_steps,
            solver,
            adjoint_solver,
            forward.adjoint_preconditioner(),
            forward.stepper(),
        )
