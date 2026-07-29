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
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from .linear import default_linear_solver, solve_linear
from .newton import newton_correction
from .norm import ResidualNorm

# The step a forward-step strategy supplies: given the (single-argument) residual, the current
# iterate, the starting residual norm, and the linear solver, return the next iterate, **the
# restart-cycle count of the linear solve that produced it, and the line-search factor alpha**. The
# count and alpha are the step's cost/quality, not part of its result: an observed march watches the
# count rise to detect a frozen preconditioner going stale and reads alpha to control the shift, while
# the plain Newton loop drops both at the call site.
_ForwardStep = Callable[
    [Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray, jnp.ndarray, lx.AbstractLinearSolver],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
]


class ForwardStep(Protocol):
    """A globalized Newton forward-step strategy (line search, pseudo-transient continuation, ...).

    The single point of variation in the forward loop: given the residual, the current iterate, the
    starting residual norm, and the linear solver, a strategy returns the next iterate and the cost
    of the linear solve that produced it. Every strategy must reduce to the undamped Newton step
    near the root and impose no shift at the fixed point, so the converged state solves the
    unshifted ``R = 0`` and the implicit-function-theorem adjoint is independent of which strategy
    produced the forward path.

    Structural interface only (a ``Protocol``), so the generic solver stays free of any flow
    specifics. The concrete strategies are :class:`DampedNewtonStep` (the default backtracking line
    search) and :class:`PseudoTransientStep` (the residual-agnostic pseudo-transient march;
    :func:`aquaflux.flow.momentum_continuation` configures it for the high-Reynolds flow).
    """

    def stepper(self) -> _ForwardStep:
        """The forward step ``(residual_fn, phi, residual_norm_0, solver) -> (phi_next, cycles, alpha)``.

        ``cycles`` is the restart-cycle count of the linear solve behind the accepted step (its cost,
        which an observed march reads to detect a stale preconditioner); ``alpha`` is the line-search
        factor of that step (its quality — ``1`` if the full shifted step descended, smaller if it was
        clipped, the signal a step controller drives the shift by). There is no variant of this method
        that drops them; a caller that wants neither writes ``phi, _, _ = step(…)``.
        """

    def default_solver(self) -> lx.AbstractLinearSolver:
        """The forward-loop linear solver to use when the caller supplies none (an inexact-Newton
        default whose tolerances suit this strategy's march)."""

    def norm(self) -> ResidualNorm:
        """The residual measure ``R -> scalar`` this strategy judges progress by.

        Owns the norm so the outer convergence test and this strategy's own globalization (the
        line search / switched-evolution-relaxation ramp / divergence guard) use **one** consistent
        measure. Defaults to the Euclidean norm; a heterogeneous block system returns a
        :class:`~aquaflux.solve.BlockScaledNorm` so no single large-magnitude block dominates the
        stopping test or the globalization."""

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


def backtracking_line_search(
    residual_fn, phi, delta, reference_norm, steps, norm=jnp.linalg.norm, growth=1.0, grow=0
):
    """Backtrack the step length: the largest ``alpha`` in ``{1, 1/2, ..., 1/2**steps}`` with
    ``norm(R(phi + alpha delta)) < reference_norm``, falling back to the smallest rung if none reduces
    the residual.

    The ladder is walked by a ``lax.while_loop`` that **stops at the first (largest) reducing rung**,
    so a step whose full length already descends (the common case near the root) costs a single
    residual evaluation rather than ``steps + 1``. The loop compiles its body once, which keeps this
    off the compile-time cost of an unrolled ladder. It is a **forward-only** device — the search
    lives inside :class:`ImplicitNewtonSolver`'s ``custom_vjp`` forward pass, whose reverse rule is
    the implicit-function-theorem transpose solve at the converged root and never differentiates the
    iteration — so the non-differentiability of ``lax.while_loop`` is not a constraint here. Do not
    call it on a path that is itself differentiated. ``steps == 0`` returns the undamped full step
    ``phi + delta`` unchanged, so a well-behaved iterate (near the root, or a linear residual) is
    unaffected. The search only reshapes the forward path — the converged state, and hence the IFT
    adjoint, is unchanged. Shared by the line-searched Newton step and the pseudo-transient march, so
    an accurate direction that overshoots is scaled back cheaply (residual evaluations) rather than
    re-solved.

    Parameters
    ----------
    residual_fn : callable
        The single-argument residual ``phi -> R(phi)``.
    phi : jnp.ndarray
        The current iterate.
    delta : jnp.ndarray
        The (full) correction direction; the search scales it by ``alpha``.
    reference_norm : jnp.ndarray
        The residual norm the candidate must beat (typically ``norm(R(phi))``), measured with the
        same ``norm`` passed here.
    steps : int
        Maximum step-halvings (static). ``0`` disables the search.
    growth : float or jnp.ndarray, optional
        How much the residual may **grow** and still be accepted, as a multiple of ``reference_norm``
        (default ``1.0`` = strict descent, the classical monotone search). A value above one makes the
        search **non-monotone**: it keeps the largest step whose residual is below
        ``growth * reference_norm`` rather than below ``reference_norm``.

        **Why a monotone search is wrong far from the root here.** A pseudo-transient step is a
        pseudo-time march, not a descent method: the steady residual is legitimately non-monotone along
        a transient path, so a strict-descent test *vetoes physically correct steps*. Measured on a
        separating RANS case, the rejected step was improving the momentum and ``k`` balances
        monotonically and growing the recirculation while the ``omega`` block -- which dominates the
        norm -- rose; the search then returned its smallest rung (a near-null step), the divergence
        guard accepted it as finite, and the march reported progress while standing still. Allowing
        controlled growth far from the root admits those steps. Near the root the monotone test is
        wanted again, for the terminal quadratic phase -- hence a *schedule* rather than a constant
        (see :class:`~aquaflux.solve.LineSearchGrowth`).
    norm : callable, optional
        The residual measure ``R -> scalar`` the acceptance is judged by (default the Euclidean
        norm). A heterogeneous block system passes a :class:`~aquaflux.solve.BlockScaledNorm` so the
        search judges every block, not only the largest-magnitude one — otherwise a step that lets a
        small-scale block (e.g. ``k`` in a coupled RANS state) blow up is accepted because the norm
        is dominated by another block (``omega``).

    Returns
    -------
    stepped : jnp.ndarray
        ``phi + alpha delta`` for the kept ``alpha``.
    alpha : jnp.ndarray
        The kept step-length fraction (a scalar): ``1`` when the full step descended, smaller when it
        had to be shortened. A staleness / step-control signal — ``alpha = 1`` means the shifted step
        was not clipped, ``alpha < 1`` that it overshot. ``steps == 0`` returns ``alpha = 1`` (the
        full step is taken unconditionally).
    """
    if steps == 0:
        return phi + delta, jnp.asarray(1.0)

    # Walk the ladder from the LONGEST step down and keep the first admissible one -- the largest
    # step the acceptance tolerance allows, not the one that minimizes the residual.
    #
    # Taking the largest rather than the best is deliberate, and was measured. A minimizing search
    # reaches a lower residual per step but travels much less far, and on a marching solve distance is
    # the point: the same case run both ways developed its recirculation 9x more slowly under the
    # minimizing search, while reporting *better* residuals at every early step. Residual depth per
    # step and progress per step are different objectives here, and progress is the one that matters
    # while the solution is still forming.
    #
    # The ladder extends ABOVE one (``grow`` rungs of doubling) because the admissible step is often
    # longer than the full step. Measured on a developed state: the full step moved the reattachment
    # not at all, while alpha ~ 5.7 moved it four times further and still sat inside the tolerance the
    # acceptance rule already allowed -- it was simply unreachable from a ladder that starts at one.
    #
    # When NOTHING is admissible the fallback is the longest finite rung **no longer than the full
    # step**, not the shortest rung. The shortest is a near-null step that changes nothing and which
    # the divergence guard then accepts as finite, so the march reports a step and stands still -- a
    # guaranteed stall rather than a slow one, and the observed failure mode on a stiff coupled march.
    #
    # The cap at one is not incidental. Without it, extending the ladder upward also extends the
    # FALLBACK upward, so a step with no admissible length quietly becomes a multiple of the full step:
    # measured with two growth rungs, a march took alpha = 4 as a fallback and multiplied its residual
    # measure by 4.6 in a single step. A growth rung must only ever be reachable by PASSING the
    # acceptance test, never by falling back onto it -- the fallback exists to avoid a null step, not
    # to license an excursion.
    #
    # The carry (index, chosen, found, longest_finite, seen_finite) is fixed-shape, so the body
    # compiles once. Walking longest-first means the first finite rung encountered IS the longest
    # finite one, so the fallback needs no second pass.
    admissible = growth * reference_norm
    lowest = -grow  # rung index; alpha = 0.5**index, so negative indices are steps longer than one

    def cond(carry):
        index, _, found, _, _ = carry
        return (~found) & (index <= steps)

    def body(carry):
        index, chosen, _, longest_finite, seen_finite = carry
        alpha = 0.5**index
        value = norm(residual_fn(phi + alpha * delta))
        finite = jnp.isfinite(value)
        accepted = finite & (value < admissible)
        # The fallback candidate ignores rungs longer than the full step (index < 0).
        eligible = finite & (index >= 0)
        return (
            index + 1,
            jnp.where(accepted, alpha, chosen),
            accepted,
            jnp.where(eligible & ~seen_finite, alpha, longest_finite),
            seen_finite | eligible,
        )

    shortest = jnp.asarray(0.5**steps)
    _, chosen, found, longest_finite, _ = jax.lax.while_loop(
        cond,
        body,
        (jnp.asarray(lowest), shortest, jnp.asarray(False), shortest, jnp.asarray(False)),
    )
    chosen = jnp.where(found, chosen, longest_finite)
    return phi + chosen * delta, chosen


def _damped_newton_step(
    residual_fn, phi, solver, preconditioner, line_search_steps, norm=jnp.linalg.norm
):
    """One Newton step with a monotone backtracking line search on the residual norm.

    ``line_search_steps == 0`` recovers the undamped full step ``phi + delta``. Otherwise the step
    length ``alpha`` is halved (up to ``line_search_steps`` times) until
    ``norm(R(phi + alpha delta)) < norm(R(phi))`` — the globalization a convection-dominated open
    flow needs, where the full Newton step from a uniform field overshoots and diverges. A full step
    is kept unchanged whenever it already reduces the residual, so a well-behaved iterate (near the
    root, or a linear residual) is unaffected. The search only reshapes the forward path; the IFT
    adjoint depends solely on the converged state, so it stays gradient-transparent. ``norm`` is the
    residual measure the search is judged by (default Euclidean).

    Returns ``(phi_next, cycles, alpha)`` — the stepped iterate, the restart-cycle count of the one
    linear solve behind it, and the line-search factor. The line search itself costs only residual
    evaluations, so the step's linear-solve cost is exactly that single solve's.
    """
    delta, r, cycles = newton_correction(
        residual_fn, phi, solver=solver, preconditioner=preconditioner
    )
    stepped, alpha = backtracking_line_search(
        residual_fn, phi, delta, norm(r), line_search_steps, norm=norm
    )
    return stepped, cycles, alpha


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
    residual_norm : ResidualNorm
        The residual measure this step judges progress by, shared with the outer stopping test so both
        agree on one scale. Defaults to the Euclidean norm.
    line_search : int
        Maximum step-halvings in the backtracking line search (static). ``0`` disables it (pure
        full Newton).
    """

    preconditioner: Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]] | None = (
        eqx.field(static=True, default=None)
    )
    line_search: int = eqx.field(static=True, default=10)
    # Data rather than static, matching `PseudoTransientStep`: an observed march re-derives the
    # measure each outer iteration and swaps it in with `tree_at`, which needs it to be a leaf. A
    # plain callable has no arrays, so it is filtered to the static side anyway and the default path
    # is unchanged.
    residual_norm: ResidualNorm = eqx.field(default=jnp.linalg.norm)

    def stepper(self) -> _ForwardStep:
        """The line-searched Newton step ``(residual_fn, phi, ‖R₀‖, solver) -> (phi_next, cycles, alpha)``."""
        preconditioner = self.preconditioner
        line_search = self.line_search
        norm = self.residual_norm

        def step(residual_fn, phi, residual_norm_0, solver):
            # The starting norm is unused: each step's line search is decided from the residual at
            # the current iterate, not the initial one.
            del residual_norm_0
            return _damped_newton_step(
                residual_fn, phi, solver, preconditioner, line_search, norm=norm
            )

        return step

    def default_solver(self) -> lx.AbstractLinearSolver:
        """The inexact-Newton forward solver (loose relative tolerance; the next step corrects the
        leftover, cutting the matvec count per step several-fold with the converged state unchanged)."""
        return _INEXACT_FORWARD_SOLVER

    def norm(self) -> ResidualNorm:
        """The residual measure the line search is judged by (the injected :attr:`residual_norm`)."""
        return self.residual_norm

    def adjoint_preconditioner(
        self,
    ) -> Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]] | None:
        """The forward preconditioner, reused (transposed) for the adjoint solve."""
        return self.preconditioner


def _within_tolerance(residual_norm, residual_norm_0, rtol, atol):
    """The Newton stopping test: the residual norm has dropped to the absolute/relative floor."""
    return residual_norm <= atol + rtol * residual_norm_0


def _forward(residual_fn, phi0, theta, rtol, atol, max_steps, solver, forward_step_fn, norm_fn):
    """Newton iterate to convergence (``lax.while_loop``); return the converged field or error.

    Each iteration applies the injected ``forward_step_fn`` — the globalized Newton step the
    :class:`ForwardStep` strategy supplies (a backtracking line search by default, a pseudo-transient
    continuation for a high-Reynolds convective flow). Every strategy's shift vanishes at the fixed
    point, so the converged field solves the same unshifted ``R(phi, theta) = 0`` and the stopping
    test is unchanged.

    ``norm_fn`` is the residual measure the stopping test uses — the **same** measure the forward
    step judges its own globalization by (the strategy owns it, via :meth:`ForwardStep.norm`), so a
    heterogeneous block system's convergence and its line search agree on one scale. The default is
    the Euclidean norm.

    The loop can exit without converging in two ways: it exhausts ``max_steps`` short of tolerance,
    or the residual norm becomes non-finite (``NaN``/``Inf``), which makes the ``residual_norm > tol``
    test ``False`` and exits at once. Both leave a field that does *not* solve ``R = 0``. The
    implicit-function-theorem adjoint linearizes the residual at whatever field this returns, so a
    non-converged field would yield a **silently wrong gradient** — the transpose solve is still
    well-posed and raises no ``NaN``. Guard against that here: if the terminal residual is non-finite
    or above tolerance, raise instead of returning a poisoned field, so neither the forward value nor
    the gradient built on it can be used unknowingly.
    """
    residual_norm_0 = norm_fn(residual_fn(phi0, theta))

    def cond(carry):
        _, step, residual_norm = carry
        return (step < max_steps) & ~_within_tolerance(residual_norm, residual_norm_0, rtol, atol)

    def body(carry):
        phi, step, _ = carry

        def residual_theta(p):
            return residual_fn(p, theta)

        # The step's cycle count and line-search factor are dropped here, deliberately. Carrying
        # either out of this loop would put a forward-only scalar in the primal output of the
        # surrounding `custom_vjp`, so the reverse rule would have to handle a float0 cotangent leaf
        # for a number the differentiated path can never use; and it would force this generic loop to
        # choose which step's value survives (last / max / sum), which is a reporting/control policy
        # the Newton solver has no business owning. A march that wants per-step cost or the line-search
        # factor observes them eagerly instead (`forward_march`).
        phi, _, _ = forward_step_fn(residual_theta, phi, residual_norm_0, solver)
        return phi, step + 1, norm_fn(residual_fn(phi, theta))

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
    adjoint_preconditioner,
    forward_step_fn,
    norm_fn,
):
    return _forward(
        residual_fn, phi0, theta, rtol, atol, max_steps, solver, forward_step_fn, norm_fn
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
    adjoint_preconditioner,
    forward_step_fn,
    norm_fn,
):
    phi_star = _forward(
        residual_fn, phi0, theta, rtol, atol, max_steps, solver, forward_step_fn, norm_fn
    )
    return phi_star, (phi_star, theta)


class TransposedPreconditioner:
    """An adjoint-preconditioner factory whose output is **already** the transpose ``M^T``.

    The generic adjoint machinery derives the transpose preconditioner from the forward one with
    :func:`jax.linear_transpose`, which works only when the forward preconditioner is a traceable
    JAX operation (an algebraic-multigrid V-cycle is). A preconditioner applied through a host
    callback -- the monolithic incomplete-LU factorization, whose triangular solve runs in ``scipy``
    via :func:`jax.pure_callback` -- cannot be transposed that way; instead it supplies its own
    transpose directly (the same factorization applied with a transposed triangular solve). Wrapping
    the factory in this marker tells :func:`_adjoint_preconditioner` to apply its output as-is rather
    than transpose it.

    Parameters
    ----------
    factory : callable
        The ``state -> M^T`` factory, returning the transpose preconditioner matvec directly.
    """

    def __init__(self, factory: Callable[[Any], Callable[[Any], Any]]) -> None:
        self.factory = factory

    def __call__(self, state: Any) -> Callable[[Any], Any]:
        return self.factory(state)


def _adjoint_preconditioner(preconditioner, phi_star, example):
    """Transpose ``M^T`` of the forward preconditioner, as a left preconditioner for the adjoint.

    The forward ``M = preconditioner(phi*)`` approximates ``J^{-1}``; the adjoint solves the
    transpose system ``J^T lambda = v``, whose consistent left preconditioner is ``M^T ~ J^{-T}``,
    obtained by transposing the (linear) preconditioner matvec with :func:`jax.linear_transpose`.
    It is mesh-independent wherever ``M`` is -- the adjoint GMRES iteration count stays flat under
    refinement instead of growing with the system size. ``None`` in, ``None`` out. A
    :class:`TransposedPreconditioner` factory already returns ``M^T`` (a callback preconditioner that
    :func:`jax.linear_transpose` cannot handle), so it is applied directly.
    """
    if preconditioner is None:
        return None
    if isinstance(preconditioner, TransposedPreconditioner):
        return preconditioner(phi_star)
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
    norm_fn,
    residuals,
    cotangent,
):
    # norm_fn is a forward-only measure (stopping test + globalization); the adjoint never forms a
    # residual norm, so it is unused here.
    del norm_fn
    phi_star, theta = residuals
    # Transpose Jacobian solve: (dR/dphi)^T lambda = cotangent, left-preconditioned by M^T so the
    # adjoint solve is mesh-independent (unpreconditioned it grows with the system size). This solve
    # sets the gradient accuracy, so it uses the (tight) adjoint solver, not the inexact forward one.
    _, vjp_phi = jax.vjp(lambda p: residual_fn(p, theta), phi_star)
    adjoint_precond = _adjoint_preconditioner(adjoint_preconditioner, phi_star, cotangent)
    lam, _ = solve_linear(
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
        The globalized forward-step strategy that supplies each Newton iteration. A **pytree field,
        deliberately not static**: a step control varies the shift strength by swapping in a
        :class:`~aquaflux.solve.ConstantRelaxation` carrying ``beta`` as a dynamic leaf, and that is
        what keeps a controlled march a compilation-cache hit rather than a recompile per step. It is a
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
            forward.norm(),
        )
