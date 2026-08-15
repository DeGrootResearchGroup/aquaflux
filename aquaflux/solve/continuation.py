"""Pseudo-transient continuation as a residual-agnostic forward-step strategy.

Pseudo-transient continuation globalizes Newton on a stiff nonlinear residual by solving, each
step, a *diagonally shifted* system

    (J(φ) + diag(s)) δ = -R(φ),    s = β d(φ),

and taking ``φ ← φ + δ``. The shift ``s`` is a residual-ramped pseudo-time term
(``β = β₀ (‖R‖/‖R₀‖)^p``, switched-evolution-relaxation): strong damping while the residual is
large (robust from a cold start) and none as it vanishes (``β → 0`` recovers the undamped Newton
step and its terminal quadratic rate). Because the shift **vanishes at the fixed point** — ``δ = 0``
forces ``R(φ*) = 0``, the unshifted steady residual — the implicit-function-theorem adjoint (which
linearizes ``R`` at ``φ*``, never the shifted operator) is untouched: continuation only reshapes the
forward path, like the line search it replaces.

Each step is **closed-loop**: it accepts its shifted correction only if the residual stays finite
and bounded, and otherwise **escalates the damping and retries** (a smaller pseudo-timestep) until
the step is accepted. This turns ``β₀`` into a starting guess (too small is recovered by escalation;
too large only slows the march) rather than a per-case knob, and it cannot diverge to a non-finite
iterate. The retry does not change the fixed point, so the converged state and its adjoint are
unchanged.

Everything above is independent of *what* is being solved. The only problem-specific choices — which
degrees of freedom carry the shift, how large the base shift ``d(φ)`` is, and the shifted-operator
preconditioner — are supplied by an injected :class:`ShiftPolicy` (for the coupled flow, the
velocity-block ``a_P`` shift and the matching SIMPLE preconditioner; see
:class:`aquaflux.flow.MomentumShiftPolicy`). :class:`PseudoTransientStep` is therefore reusable for
any nonlinear residual, not only the flow.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from .implicit import StepOutcome, _ForwardStep, backtracking_line_search
from .line_search_growth import LineSearchGrowth, MonotoneLineSearch
from .linear import corrected_cycles, solve_linear
from .norm import ResidualNorm
from .relaxation import RelaxationSchedule, SwitchedEvolutionRelaxation

# Inexact-Newton forward solver for the pseudo-transient march: a loose *relative* tolerance (each
# shifted step need only make Newton progress; the next step corrects the leftover) but a *tight*
# absolute floor and a generous restart/stagnation budget. The march drives the residual far below
# the ``1e-3`` absolute floor the plain inexact solver uses, and once ``‖R‖`` nears that floor the
# linear solve would stop taking a step and the outer march would stall short of the nonlinear
# tolerance — so the absolute term must not cap the terminal convergence. The looser stagnation
# budget also rides out the stiffer shifted operators a graded, high-Reynolds mesh produces.
_INEXACT_CONTINUATION_SOLVER = lx.GMRES(rtol=1e-3, atol=1e-10, restart=40, stagnation_iters=40)


def _shifted_solve(residual_fn, phi, rhs, shift, preconditioner, solver):
    """Solve the diagonally-shifted Newton system ``(J(phi) + diag(shift)) delta = -rhs``, matrix-free.

    The shifted-Newton correction shared by the pseudo-transient and dual-time marches: the true
    Jacobian-vector product ``J(phi) v`` (from ``residual_fn``) plus the pseudo-time diagonal ``shift``
    on the shifted degrees of freedom, solved against ``-rhs`` with the frozen shifted preconditioner.
    ``rhs`` is the residual the step drives down -- the steady ``R(phi)`` for pseudo-transient
    continuation, the transient ``R(phi) + shift*(phi - phi_ref)`` for the dual-time inner loop; the
    Jacobian is ``J(phi) + shift`` either way (the transient term's derivative in ``phi`` is ``shift``).
    Non-throwing, so a non-convergent shifted system returns a candidate for the caller to judge rather
    than raising.

    Parameters
    ----------
    residual_fn : callable
        The single-argument steady residual ``phi -> R(phi)``; its Jacobian-vector product forms the
        unshifted part of the operator.
    phi : jnp.ndarray
        The iterate the Jacobian is linearized at, shape ``(n,)``.
    rhs : jnp.ndarray
        The residual the shifted system is solved against (``-rhs`` is the right-hand side), shape
        ``(n,)``.
    shift : jnp.ndarray
        The pseudo-time diagonal ``β d`` added to the Jacobian, shape ``(n,)`` (zeros off the shifted
        degrees of freedom).
    preconditioner : callable or None
        The frozen preconditioner for the shifted operator, applied on the **RIGHT** -- this call passes
        no ``preconditioner_side``, so it takes :func:`~aquaflux.solve.solve_linear`'s ``"right"``
        default. That is deliberate and load-bearing: under right preconditioning the Krylov residual is
        ``b - A M y = b - A x``, the **true** residual, so the relative-residual stop stays honest even
        where ``M`` is a poor inverse -- the shifted coupled saddle at low shift, where a
        left-preconditioned solve would report convergence while returning a step that does not solve
        the system. Do not read the loose ``forward_rtol`` as a preconditioned-residual stop; it is not.
    solver : lineax.AbstractLinearSolver
        The Krylov solver for the shifted system.

    Returns
    -------
    delta : jnp.ndarray
        The shifted-Newton correction, shape ``(n,)``.
    cycles : jnp.ndarray
        The restart-cycle count of the linear solve (int32 scalar).
    """

    # A preconditioner tagged ``is_exact_native`` runs the whole shifted solve natively on the host (PETSc
    # GMRES + native GAMG, its operator a shell over the exact jvp at ``phi``) -- apply it directly rather
    # than wrapping the moderate V-cycle in a JAX-side Krylov iteration, which is far slower (the JAX-side
    # GMRES needs tens of per-matvec callbacks where the native solve reaches the stop in ~1). One "cycle"
    # by convention, so the staleness signal stays well defined. Forward-only (the adjoint uses the
    # differentiable single-V-cycle transpose).
    if getattr(preconditioner, "is_exact_native", False):
        return preconditioner.exact_solve(phi, -rhs, shift), jnp.asarray(1, dtype=jnp.int32)

    def shifted_jacobian(tangent: jnp.ndarray) -> jnp.ndarray:
        return jax.jvp(residual_fn, (phi,), (tangent,))[1] + shift * tangent

    return solve_linear(
        shifted_jacobian, -rhs, solver=solver, preconditioner=preconditioner, throw=False
    )


class _Attempt(NamedTuple):
    """One shifted solve, line-searched and measured — everything a step learns for one ``β``.

    These five values are produced together by a single (expensive) shifted linear solve and are
    consumed together by the accept/reject decision, so they travel as one record rather than as a
    loose tuple. Carrying the record is what lets a probing loop hand its result to the escalation
    loop instead of discarding it and paying for the same solve twice.

    Attributes
    ----------
    candidate : jnp.ndarray
        The trial iterate ``φ + α δ``, shape ``(n,)``.
    residual_norm : jnp.ndarray
        The measure at :attr:`candidate` — a scalar, in whatever measure the step is judged by.
    cycles : jnp.ndarray
        Krylov cycles the shifted solve took — a scalar, the staleness signal a refresh trigger reads.
    alpha : jnp.ndarray
        The line-search factor actually taken — a scalar.
    directional : jnp.ndarray
        ``d/ds measure(R(φ + s δ))`` at ``s = 0`` — a scalar. Negative means the correction descends
        in the measure the solve is judged by.
    """

    candidate: jnp.ndarray
    residual_norm: jnp.ndarray
    cycles: jnp.ndarray
    alpha: jnp.ndarray
    directional: jnp.ndarray


class ShiftTerm(NamedTuple):
    """The per-step data a :class:`ShiftPolicy` produces at one iterate.

    Attributes
    ----------
    diagonal : jnp.ndarray
        The **base** pseudo-time diagonal ``d(φ)`` over the *full* state vector, shape ``(n_dof,)``,
        with zeros on the degrees of freedom that receive no shift. The step scales it by the
        relaxation to form the shift ``β d`` added to the Jacobian diagonal.
    make_preconditioner : callable
        ``relaxation -> M`` giving the frozen left preconditioner ``M`` (a matvec approximating the
        *shifted* operator's inverse) for a given ``β``, or ``None`` for an unpreconditioned solve.
        Passed the same ``β`` the diagonal is scaled by, so ``M`` inverts the same shifted operator.
    """

    diagonal: jnp.ndarray
    make_preconditioner: Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray] | None]


class ShiftPolicy(Protocol):
    """The problem-specific part of pseudo-transient continuation (structural interface only).

    A policy decides which degrees of freedom carry the pseudo-time shift, how large the base shift
    is, and how the shifted operator is preconditioned — everything :class:`PseudoTransientStep`
    needs that depends on the physics. The generic march owns the schedule, the shifted solve, and
    the acceptance/escalation loop and never imports any problem specifics.
    """

    def shift_term(self, phi: jnp.ndarray) -> ShiftTerm:
        """The base shift diagonal and the ``β -> M`` preconditioner factory at iterate ``phi``."""


class StepAcceptance(Protocol):
    """The accept/reject decision for one shifted-step attempt (structural interface only).

    The engine's escalation-loop mechanics — grow ``β`` on rejection, cap at ``max_escalations``,
    carry the best candidate — are fixed; a policy supplies only *whether* a given candidate is
    accepted. It sees pure scalars (residual norms and the attempt index), so it is unit-testable
    with no solve. The default :class:`DivergenceGuard` is a divergence guard, not a descent test,
    because the pseudo-transient march is legitimately non-monotone; a monotone / sufficient-decrease
    or forcing rule is a drop-in alternative.
    """

    def accept(
        self,
        candidate_norm: jnp.ndarray,
        residual_norm: jnp.ndarray,
        residual_norm_0: jnp.ndarray,
        attempt: jnp.ndarray,
    ) -> jnp.ndarray:
        """Whether to accept a candidate (a boolean array).

        Parameters
        ----------
        candidate_norm : jnp.ndarray
            ``‖R(candidate)‖`` of the shifted-step candidate under test.
        residual_norm : jnp.ndarray
            ``‖R(φ)‖`` at the current iterate (before the step) — for a descent / monotone test.
        residual_norm_0 : jnp.ndarray
            ``‖R(φ₀)‖`` at the initial iterate — the scale a divergence guard measures against.
        attempt : jnp.ndarray
            The 0-based attempt index within this step's escalation loop.
        """


class DivergenceGuard(eqx.Module):
    """Accept unless the candidate diverges — the default acceptance policy.

    Rejects only a non-finite candidate or one that has blown up past ``divergence_cap × ‖R₀‖``, and
    accepts everything else. Measured against the *initial* residual because the pseudo-transient
    march is non-monotone (it oscillates around and below ``‖R₀‖``), so this catches a genuine blow-up
    without rejecting a healthy transient — it is a divergence guard, not a descent test.

    Attributes
    ----------
    divergence_cap : float
        The divergence threshold (static): an attempt is rejected if its residual is non-finite or
        exceeds ``divergence_cap × ‖R₀‖``. Lenient by default; lower it to intervene on divergence
        sooner.
    """

    divergence_cap: float = eqx.field(static=True, default=10.0)

    def accept(
        self,
        candidate_norm: jnp.ndarray,
        residual_norm: jnp.ndarray,
        residual_norm_0: jnp.ndarray,
        attempt: jnp.ndarray,
    ) -> jnp.ndarray:
        # A pure divergence guard needs neither the previous-iterate norm nor the attempt index.
        del residual_norm, attempt
        return jnp.isfinite(candidate_norm) & (
            candidate_norm < self.divergence_cap * residual_norm_0
        )


class PseudoTransientStep(eqx.Module):
    """Pseudo-transient continuation as a :class:`~aquaflux.solve.ForwardStep` (see the module docstring).

    The residual-agnostic engine: it forms the switched-evolution-relaxation shift, solves the
    shifted Newton system, and runs the closed-loop accept/escalate loop, delegating every
    problem-specific choice to an injected :class:`ShiftPolicy`. Plug it into
    :class:`~aquaflux.solve.ImplicitNewtonSolver` as its ``forward_step``.

    Attributes
    ----------
    shift_policy : ShiftPolicy
        Supplies the base shift diagonal and the shifted-operator preconditioner at each iterate
        (e.g. :class:`aquaflux.flow.MomentumShiftPolicy` for the coupled flow).
    relaxation_schedule : RelaxationSchedule
        Sets the shift strength ``β`` each step from the current and reference residual norms.
        Defaults to :class:`~aquaflux.solve.SwitchedEvolutionRelaxation` (SER,
        ``β = max(beta_floor, β₀(‖R‖/‖R₀‖)^p)``) — its ``beta0``/``exponent``/``beta_floor`` were the
        old fields here and are now that class's constructor arguments. Memoryless by contract, so it
        stays on the differentiable path; a stateful/feedback damping rule is a forward-only
        :class:`~aquaflux.solve.StepControl` on the eager march, not a schedule.
    line_search_growth : LineSearchGrowth
        How far the residual may rise and still be accepted by the backtracking ladder
        (:class:`~aquaflux.solve.LineSearchGrowth`). Defaults to strict descent
        (:class:`~aquaflux.solve.MonotoneLineSearch`); a pseudo-time march far from the root may want
        :class:`~aquaflux.solve.RelaxedFarFromRoot`. **Not exposed by any builder** -- set it by
        constructing this class directly.
    max_escalations : int
        Maximum damping escalations per step (static). If a step's shifted solve fails to descend (an
        ill-conditioned shifted system, or an overshoot), ``β`` is multiplied by
        :attr:`escalation_factor` and the step retried, up to this many times. A well-behaved step is
        accepted on the first attempt (no extra cost). ``0`` disables escalation.
    escalation_factor : float
        Factor ``> 1`` by which ``β`` grows on each rejected attempt (static).
    acceptance : StepAcceptance
        The accept/reject policy for each shifted-step attempt. Defaults to a
        :class:`DivergenceGuard` (accept unless the candidate is non-finite or exceeds
        ``divergence_cap × ‖R₀‖``) — the divergence guard the non-monotone march needs. Swap in a
        monotone / sufficient-decrease or forcing rule without touching the escalation loop.
    line_search : int
        Maximum backtracking step-halvings applied to the shifted correction *before* the step is
        judged (static). ``0`` (the default) takes the full shifted step ``φ + δ``, so escalating
        the damping ``β`` — a full re-solve — is the only recourse when that step overshoots. A
        positive value first scales ``δ`` back along the ladder ``{1, 1/2, …, 1/2**line_search}``
        (:func:`~aquaflux.solve.implicit.backtracking_line_search`, cheap residual evaluations, no
        re-solve), keeping the largest length that reduces the residual. When the shifted direction
        is accurate but the *full* step overshoots — the stiff coupled-RANS regime, where a full step
        blows up while a quarter-step descends — this recovers a descent from the **one** expensive
        solve instead of re-solving at larger ``β`` (which changes the direction and, measured, does
        not descend). The ``β`` escalation remains the fallback for a genuinely bad direction (an
        ill-conditioned shifted solve). Like the shift, it only reshapes the forward path, so the
        converged state and the IFT adjoint are unchanged.
    forward_solver : lineax.AbstractLinearSolver or None
        The linear solver for the shifted forward solves, overriding the shared
        :data:`_INEXACT_CONTINUATION_SOLVER` when set. A stiff coupled system whose shifted
        operator needs a larger Krylov subspace to converge without restarting can pass a
        larger-``restart`` GMRES here; ``None`` uses the shared default.
    residual_norm : ResidualNorm
        The residual measure ``R -> scalar`` the march judges progress by (static, default the
        Euclidean norm): the switched-evolution-relaxation ramp ``β = β₀(‖R‖/‖R₀‖)^p``, the line
        search, and the acceptance/divergence guard all use it, and :class:`ImplicitNewtonSolver`
        reads it (via :meth:`norm`) for the outer stopping test, so one measure governs the whole
        solve. A heterogeneous block system (e.g. coupled RANS, where ``omega`` is O(1e5) and ``k``
        O(1e-3)) needs a scaled measure so the march *sees* every block — with the plain norm the
        ‖R‖ is ~100% ``omega`` and the line search neither judges nor protects the ``k`` block. The
        coupled-RANS builders therefore inject a per-state, row-equilibrated
        :class:`~aquaflux.solve.RowScaledNorm` (``coupled_scaled_norm``, rebuilt each outer iteration
        and held fixed across a line search), overriding this field's plain-norm class default; the
        coarser :class:`~aquaflux.solve.BlockScaledNorm` is an off-by-default alternative. Like the
        shift, it only reshapes the forward path; the IFT adjoint never forms a norm, so the converged
        state and its gradient are unchanged.
    adjoint_preconditioner_factory : callable or None
        The ``state -> M`` preconditioner factory for the converged transpose (adjoint) solve, or
        ``None`` for an unpreconditioned adjoint (static). At ``φ*`` the operator is the
        well-conditioned steady Jacobian (``β → 0``), so the adjoint needs no shift — the ordinary
        (unshifted) preconditioner is the consistent choice.
    """

    shift_policy: ShiftPolicy
    relaxation_schedule: RelaxationSchedule = eqx.field(default_factory=SwitchedEvolutionRelaxation)
    line_search_growth: LineSearchGrowth = eqx.field(default_factory=MonotoneLineSearch)
    max_escalations: int = eqx.field(static=True, default=6)
    escalation_factor: float = eqx.field(static=True, default=2.0)
    acceptance: StepAcceptance = eqx.field(default_factory=DivergenceGuard)
    line_search: int = eqx.field(static=True, default=0)
    # The positivity constraint, the same pair `DualTimeStep` carries and applied identically. It lives
    # on both because it guards the *state*, not the march: a field that must stay positive must stay
    # positive whichever strategy is stepping it, and having it on only one meant choosing a strategy
    # silently gave up a guard whose absence is a recorded march death (two cells of 23040 took `k`
    # negative and NaN'd the whole residual through a bare `sqrt`). Both default `None`, which is the
    # unconstrained step exactly.
    step_limit: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = eqx.field(
        static=True, default=None
    )
    step_projection: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = eqx.field(
        static=True, default=None
    )
    # Data, not static (the same reasoning as `residual_norm` below): a solver configured with a
    # row-scaled stopping measure CARRIES that measure's scale arrays, and equinox rightly warns when
    # arrays are put in a static field -- static leaves take part in the jit cache key by equality, so
    # array-valued ones make the key ill-defined and, if the scales were ever rebuilt, would retrace
    # every step. As data the arrays are ordinary traced leaves and the cache key is shape/dtype only.
    # A solver with no array leaves (the default, and any plain-tolerance GMRES) filters to the static
    # side regardless, so this changes nothing for those.
    forward_solver: lx.AbstractLinearSolver | None = None
    # Not a static field: an observed march rebuilds the measure each outer iteration and swaps it in
    # with `tree_at`, and that is only a compilation-cache hit if the measure rides as data. A plain
    # callable (the default) has no array leaves, so it is filtered to the static side anyway and the
    # default path is unchanged.
    # Back the shift off until the correction descends in the measure the solve is judged by, then
    # escalate from there as usual. Off by default (0), because each backoff costs one shifted solve.
    #
    # Why it exists: the shifted correction is not a descent direction by construction. For the exact
    # Newton direction it would be -- with J delta = -R the derivative of the measure along delta is
    # -norm(R) for any positive weighting -- but the shifted direction satisfies J delta = -R -
    # beta D delta, whose second term has no fixed sign and grows with beta. Measured on a stiff
    # coupled state, the derivative was negative at beta <= 1 and changed sign between beta = 1 and
    # beta = 2, with the march running at about 1.9: every step length then increased the measure, the
    # line search could only pick the least-harmful rung, and the march sat still while reporting
    # steps. Backing the shift off restores descent; escalating -- the response to an overshoot --
    # makes it strictly worse.
    # Rungs the line search may try ABOVE the full step (alpha = 2**grow ... 2, 1, 1/2 ...). Zero is
    # a one-sided ladder, the historical behaviour. The admissible step is often longer than the full
    # one: measured at a cold start, alpha = 2 sat inside the acceptance tolerance and travelled twice
    # as far, but a ladder starting at one could not express it -- so a march reporting alpha = 1 every
    # step may simply never have been asked whether more was allowed.
    grow: int = eqx.field(static=True, default=0)
    descent_backoff: int = eqx.field(static=True, default=0)
    # Reject a correction that does not descend in the measure, rather than judging it on the
    # candidate's norm alone. Independent of the backoff: with the backoff off, this makes a
    # non-descent direction fail so the caller sees it instead of a step that quietly went nowhere.
    descent_test: bool = eqx.field(static=True, default=False)
    residual_norm: ResidualNorm = eqx.field(default=jnp.linalg.norm)
    adjoint_preconditioner_factory: (
        Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]] | None
    ) = eqx.field(static=True, default=None)

    def norm(self) -> ResidualNorm:
        """The residual measure the march and the outer stopping test share (:attr:`residual_norm`)."""
        return self.residual_norm

    def default_solver(self) -> lx.AbstractLinearSolver:
        """The forward-loop solver for the pseudo-transient march when the caller supplies none.

        The injected :attr:`forward_solver` when set, else the shared
        :data:`_INEXACT_CONTINUATION_SOLVER` — a loose relative tolerance with a tight absolute floor
        and a generous restart/stagnation budget, so the march is not capped short of the nonlinear
        tolerance and rides out the stiffer shifted operators a graded, high-Reynolds mesh produces.
        """
        return (
            self.forward_solver if self.forward_solver is not None else _INEXACT_CONTINUATION_SOLVER
        )

    def adjoint_preconditioner(
        self,
    ) -> Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]] | None:
        """The (unshifted) ``state -> M`` factory for the adjoint solve at the converged state."""
        return self.adjoint_preconditioner_factory

    def stepper(self) -> _ForwardStep:
        """The accepted shifted-Newton step and its linear solve's cycle count.

        ``(residual_fn, φ, ‖R₀‖, solver) -> (φ_next, cycles, alpha, inner_iterations)``. ``cycles`` is the
        raw solver count of the **accepted** attempt's shifted linear solve — the cost of the step that
        was actually taken, not the sum over rejected escalation attempts; ``inner_iterations`` is ``1``
        (a single-step attempt has no inner loop); ``alpha`` is that attempt's
        line-search factor (``1`` if the full shifted step descended, smaller if clipped), the
        step-quality signal a :class:`~aquaflux.solve.StepControl` drives the next shift by.

        **A step in which every attempt was rejected reports ``0`` cycles (and ``alpha`` = 1).** The
        count is only recorded on
        acceptance, so a fully-rejected step (the escalation ladder exhausted without the acceptance
        policy admitting anything) carries the initial zero rather than the cost of the attempts it
        burned. A consumer that reads the count as a cost signal must therefore treat ``0`` as "no
        measurement", not as "free" — otherwise a rejected step looks like the cheapest in the march.

        **Why the count is worth carrying.** On a *fixed* system the cycle count rises as a frozen
        preconditioner goes stale, and it does so before the residual history shows anything. That
        makes it the honest trigger for re-freezing the preconditioner mid-march, and a robust one:
        unlike elapsed wall-clock time (a tempting proxy), it is unaffected by machine load or a
        suspended process, and it measures the linear algebra rather than the wall clock.

        Each step forms the shifted-Newton correction at ``β = β₀ (‖R‖/‖R₀‖)^p`` and **accepts it
        only if the injected :attr:`acceptance` policy admits it** (by default, unless it diverges);
        otherwise it escalates the damping (``β *= escalation_factor``) and retries, up to
        :attr:`max_escalations`. A cold-start step whose shifted system is ill-conditioned (or whose
        full step overshoots) is re-damped until it is accepted, so the march cannot diverge to a
        non-finite iterate — while a well-behaved step is accepted on the first attempt at no extra
        solve. The retry uses a non-throwing linear solve so a non-convergent attempt is *rejected
        and re-damped* rather than raising. ``β`` still vanishes at the fixed point, so the converged
        state and the IFT adjoint are unchanged.
        """
        policy = self.shift_policy
        schedule = self.relaxation_schedule
        growth_schedule = self.line_search_growth
        max_escalations, escalation_factor = self.max_escalations, self.escalation_factor
        acceptance = self.acceptance
        line_search = self.line_search
        descent_backoff, descent_test = self.descent_backoff, self.descent_test
        grow = self.grow
        norm = self.residual_norm
        step_limit, step_projection = self.step_limit, self.step_projection

        def step(
            residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
            phi: jnp.ndarray,
            residual_norm_0: jnp.ndarray,
            solver: lx.AbstractLinearSolver,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            residual = residual_fn(phi)
            residual_norm = norm(residual)
            term = policy.shift_term(phi)  # base diagonal + β -> M, from the same iterate
            # The injected schedule sets the base shift strength for this step's first attempt (SER by
            # default: strong damping while ‖R‖ is large, easing to zero at the root). Escalation below
            # only grows it from here on a rejected attempt.
            base_relaxation = schedule.relaxation(residual_norm, residual_norm_0)
            # How far the residual may rise and still count as progress. A pseudo-time march is not a
            # descent method -- the steady residual is non-monotone along a transient path -- so a
            # strict-descent ladder vetoes correct steps far from the root. The schedule relaxes that
            # far away and restores strict descent in the basin (monotone by default).
            growth_factor = growth_schedule.growth(residual_norm, residual_norm_0)

            def attempt(relaxation: jnp.ndarray) -> _Attempt:
                # The shift only reshapes the forward path (like the preconditioner it damps), so it
                # is detached: it never perturbs the converged state or its adjoint.
                shift = jax.lax.stop_gradient(relaxation * term.diagonal)  # β d over the full state
                # Preconditioner inverts the *same* shifted operator it is damped by. The solve does
                # not throw: a non-convergent shifted system yields a candidate the acceptance test
                # rejects (triggering more damping), rather than raising.
                preconditioner = term.make_preconditioner(relaxation)
                delta, cycles = _shifted_solve(
                    residual_fn, phi, residual, shift, preconditioner, solver
                )
                # Backtrack the step length before judging it: when the shifted direction is accurate
                # but the full step overshoots, a scaled-back step descends from this one solve,
                # sparing a re-solve at larger beta. `line_search == 0` takes the full step (alpha = 1).
                # Clip the correction per entry BEFORE the cap reads it: an entry that would cross
                # zero is held back on its own, rather than shortening the step for every entry. The
                # cap below then finds nothing binding and returns 1, so the two compose -- the same
                # ordering, and the same reasoning, as the dual-time step's inner loop.
                if step_projection is not None:
                    delta = step_projection(phi, delta)
                max_alpha = 1.0 if step_limit is None else step_limit(phi, delta)
                candidate, alpha = backtracking_line_search(
                    residual_fn,
                    phi,
                    delta,
                    residual_norm,
                    line_search,
                    norm=norm,
                    growth=growth_factor,
                    grow=grow,
                    max_alpha=max_alpha,
                )
                # The directional derivative of the measure along the correction, d/ds norm(R(phi +
                # s delta)) at s = 0. Negative means the direction descends in the measure the solve
                # is judged by; non-negative means no step length along it can help, and the line
                # search will be reduced to picking the least-harmful rung.
                #
                # This is not guaranteed by construction here. For the exact Newton direction
                # (J delta = -R) it would be -- the derivative is then -norm(R) for any positive
                # weighting -- but the shifted direction satisfies J delta = -R - beta D delta, and
                # that second term carries no fixed sign. Its damage grows with beta: measured on a
                # stiff coupled state, the derivative was negative for beta <= 1 and changed sign
                # between beta = 1 and beta = 2, exactly where that march had been running.
                directional = jax.jvp(lambda x: norm(residual_fn(x)), (phi,), (delta,))[1]
                return _Attempt(
                    candidate=candidate,
                    residual_norm=norm(residual_fn(candidate)),
                    cycles=cycles,
                    alpha=alpha,
                    directional=directional,
                )

            def admits(trial: _Attempt, attempts: jnp.ndarray) -> jnp.ndarray:
                """Whether the injected acceptance policy (and the descent test) admit ``trial``."""
                accept = acceptance.accept(
                    trial.residual_norm, residual_norm, residual_norm_0, attempts
                )
                # A direction that does not descend in the measure is rejected outright, however the
                # candidate scores. Escalating on it would be worse than useless: more shift makes
                # the derivative *less* negative, so the loop would drive itself further from a
                # usable direction while paying for a solve each time. Rejecting instead lets
                # `descent_backoff` below take over, which moves β the other way.
                if descent_test:  # a static choice, so the branch is resolved at trace time
                    accept = accept & (trial.directional < 0.0)
                return accept

            # Escalate the damping on a rejected attempt, taking the first the acceptance policy
            # admits. The loop *mechanics* — grow β, cap at max_escalations, carry the best candidate
            # — are fixed here; only the accept/reject decision is the injected policy's, so a
            # divergence guard (the default) or a monotone/forcing rule slots in without touching this
            # loop. More shift (a smaller pseudo-timestep) is what a rejected step needs. The loop
            # exits as soon as an attempt is accepted, so a healthy first attempt costs a single solve;
            # only a rejected step pays for extra, more-damped attempts.
            def cond(state: tuple) -> jnp.ndarray:
                _, _, attempts, accepted, _, _ = state
                return (~accepted) & (attempts <= max_escalations)

            def record(state: tuple, trial: _Attempt, accept: jnp.ndarray) -> tuple:
                """Fold one judged attempt into an escalation carry (the shared accept bookkeeping)."""
                relaxation, best, attempts, _, best_cycles, best_alpha = state
                return (
                    relaxation * escalation_factor,
                    jnp.where(accept, trial.candidate, best),
                    attempts + 1,
                    accept,
                    # Report the cycles and line-search factor of the attempt actually taken, not the
                    # rejected escalations'. (A fully-rejected step keeps the initial 0 / 1.)
                    jnp.where(accept, trial.cycles, best_cycles),
                    jnp.where(accept, trial.alpha, best_alpha),
                )

            def body(state: tuple) -> tuple:
                relaxation, _, attempts, *_ = state
                trial = attempt(relaxation)
                return record(state, trial, admits(trial, attempts))

            def fresh(start: jnp.ndarray) -> tuple:
                """The escalation carry for a step that has not yet solved anything."""
                return (
                    start,
                    phi,
                    0,
                    jnp.asarray(False),
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(1.0),
                )

            # Back the shift OFF until the direction descends, then escalate from there as usual.
            #
            # These two loops move beta in opposite directions on purpose, because they answer
            # different failures. Escalation answers "this step overshot or the shifted system was
            # ill-conditioned" -- more damping is the cure. The backoff answers "no step length along
            # this direction can help", which more damping makes strictly worse: the shift term is what
            # spoils descent, so the derivative only becomes less negative as beta grows. Running
            # escalation against a non-descent direction therefore spends solves making the direction
            # worse, which is what a march does when it sits at the shortest rung and does not move.
            #
            # Each backoff costs one shifted solve, so it is off by default and bounded when on. The
            # probe that finally descends is a *complete* attempt at the relaxation the escalation
            # loop is about to start from, so it is carried out of the loop and folded straight into
            # the escalation carry. Re-solving it there would double the cost of every step on the
            # common path — the one where the first probe already descends and nothing is backed off.
            def backoff_cond(state: tuple) -> jnp.ndarray:
                _, tries, descends, _ = state
                return (~descends) & (tries < descent_backoff)

            def backoff_body(state: tuple) -> tuple:
                relaxation, tries, _, _ = state
                trial = attempt(relaxation)
                descends = trial.directional < 0.0
                return (
                    jnp.where(descends, relaxation, relaxation / escalation_factor),
                    tries + 1,
                    descends,
                    trial,
                )

            start_relaxation = base_relaxation
            if descent_backoff > 0:
                start_relaxation, _, probed, trial = jax.lax.while_loop(
                    backoff_cond,
                    backoff_body,
                    (
                        base_relaxation,
                        jnp.asarray(0, dtype=jnp.int32),
                        jnp.asarray(False),
                        _Attempt(
                            candidate=phi,
                            residual_norm=residual_norm,
                            cycles=jnp.asarray(0, dtype=jnp.int32),
                            alpha=jnp.ones_like(residual_norm),
                            directional=jnp.zeros_like(residual_norm),
                        ),
                    ),
                )
                # `probed` is the loop's own descent flag, so the seeded carry is used only when the
                # carried attempt really was taken at `start_relaxation`. When the backoff instead
                # exhausted its tries it exits at a *lower*, unprobed relaxation, and the escalation
                # loop starts there from scratch — the same relaxation ladder as before this fast path.
                cold = fresh(start_relaxation)
                seeded = record(cold, trial, probed & admits(trial, 0))
                start = jax.tree.map(lambda s, c: jnp.where(probed, s, c), seeded, cold)
            else:
                start = fresh(start_relaxation)

            _, phi_next, _, _, step_cycles, step_alpha = jax.lax.while_loop(cond, body, start)
            # A single-step pseudo-transient attempt has no inner Newton loop; report 1 inner iteration
            # so a consumer can offset-correct the raw solver count uniformly with the dual-time path.
            # A single damped step has no inner loop, so nothing could have been cut short and the
            # one solve is trivially the most expensive one.
            return StepOutcome(
                phi_next,
                step_cycles,
                step_alpha,
                1,
                jnp.asarray(True),
                corrected_cycles(step_cycles),
                jnp.asarray(1.0),
            )

        return step


class DualTimeStep(eqx.Module):
    """Dual-time (backward-Euler) forward-step strategy: an inner Newton loop per outer timestep.

    :class:`PseudoTransientStep` adds the shift ``β d`` to the Jacobian only and measures the bare
    steady residual ``R(φ)``. That residual is a poor convergence signal along a pseudo-time march:
    after a shifted step ``(J + β D) δ = −R`` the steady residual is ``R(φ + δ) = −β D δ``, i.e. ``β``
    times how far the step travelled, not a distance to the root — so a row-scaled measure of it
    stalls at a ``β``-proportional floor while the step is productive.

    This strategy instead takes a true backward-Euler timestep. Each call **holds a reference**
    ``φⁿ`` (the iterate it is given) and runs an **inner Newton loop** on the transient residual

        ``G(φ) = R(φ) + β d · (φ − φⁿ)``,

    solving ``(J(φ) + β d) δ = −G(φ)`` and line-searching on ``‖G‖`` until ``‖G‖ ≤ inner_tol · ‖R(φⁿ)‖``
    or ``inner_steps`` iterations are spent. Because the shift ``β d · (φ − φⁿ)`` now sits in the
    residual as well as the Jacobian, the leftover ``−β D δ`` cancels: the returned iterate satisfies
    the backward-Euler equation, and the steady residual the outer loop measures at the *next* anchor is
    the discrete unsteady term ``β d · (φⁿ⁺¹ − φⁿ)`` — the physical time derivative, which falls to zero
    as the transient settles. A row-scaled / block-scaled measure of it is then well-behaved.

    With ``inner_steps = 1`` a step is a **single shifted Newton step** — the transient term is zero at
    the anchor, so ``G(φⁿ) = R(φⁿ)`` and the one inner **solve** ``(J + β d) δ = −R(φⁿ)`` is exactly the
    one :class:`PseudoTransientStep` forms (the resulting iterate coincides too when neither line-searches,
    i.e. ``line_search = 0``). The two strategies differ in how they handle an overshoot:
    ``PseudoTransientStep`` escalates ``β`` (a re-solve at more damping) and guards divergence with an
    injected acceptance policy, whereas this strategy takes **more inner iterations at fixed ``β``** and
    line-searches each on ``‖G‖`` — the inner loop *is* the globalization, so there is no escalation
    ladder or acceptance policy here. ``inner_steps > 1`` is what lets a larger pseudo-timestep (a
    smaller ``β``) be taken stably from a cold start: the inner iterations converge the implicit step a
    single shifted step would overshoot, so ``β`` can be driven below the level at which the single-step
    march diverges. The larger the step the more the outer march accelerates, so a
    :class:`~aquaflux.solve.StepControl` on the eager march is the natural driver of ``β`` here.

    The shift still **vanishes at the fixed point** — as ``‖R‖ → 0`` the schedule ramps ``β → 0``, so
    ``G → R`` and the inner loop is a single undamped Newton step. The converged state therefore solves
    the unshifted ``R = 0`` and the implicit-function-theorem adjoint is identical to the line-searched
    or pseudo-transient march: this strategy only reshapes the forward path.

    Attributes
    ----------
    shift_policy : ShiftPolicy
        Supplies the base shift diagonal ``d(φ)`` and the ``β -> M`` preconditioner factory at the
        reference iterate — the same injected policy :class:`PseudoTransientStep` uses.
    relaxation_schedule : RelaxationSchedule
        Sets ``β`` from ``‖R(φⁿ)‖ / ‖R₀‖`` (switched-evolution-relaxation by default). Memoryless, so it
        stays on the differentiable path; a stateful ``β`` ramp is a
        :class:`~aquaflux.solve.StepControl` on the eager march (it swaps in a
        :class:`~aquaflux.solve.ConstantRelaxation`), not a schedule.
    inner_steps : int
        Maximum inner Newton iterations per outer timestep (static). ``1`` recovers
        :class:`PseudoTransientStep`; a few (2–5) converge the backward-Euler step at a larger
        pseudo-timestep.
    inner_tol : float
        The inner loop stops once ``‖G‖`` has fallen to this fraction of ``‖R(φⁿ)‖`` (static). A loose
        value (e.g. ``0.05``) is enough — the outer march re-solves each timestep anyway.
    line_search : int
        Maximum backtracking step-halvings applied to each inner correction, judged on ``‖G‖`` (static).
        ``G = 0`` is a well-posed fixed-``φⁿ`` solve, so the inner line search is strict-descent
        (monotone) on ``‖G‖`` — unlike the non-monotone steady residual the outer march tolerates.
        Default ``10``.
    forward_solver : lineax.AbstractLinearSolver or None
        The shifted-solve Krylov solver, overriding the shared :data:`_INEXACT_CONTINUATION_SOLVER`
        when set.
    residual_norm : ResidualNorm
        The measure ``R -> scalar`` used for the inner target, the inner line search, and (via
        :meth:`norm`) the outer stopping test — one consistent measure. Defaults to the Euclidean norm;
        a heterogeneous block system can pass a row-/block-scaled measure, which this strategy makes
        well-behaved (see the class summary).
    adjoint_preconditioner_factory : callable or None
        The ``state -> M`` factory for the converged transpose (adjoint) solve, or ``None`` (static). At
        ``φ*`` the operator is the unshifted steady Jacobian (``β → 0``), so the ordinary preconditioner
        is the consistent choice.
    inner_observer : callable or None
        An optional profiling hook ``(inner_index, g_before, g_after, cycles, alpha, iterate) -> None``
        called **once per inner Newton iteration** with that iteration's ``‖G‖`` before and after the
        step, the solver's raw cycle count for its shifted solve, its line-search factor, and the
        iterate it reached (static). It surfaces the inner trajectory the outer
        :class:`~aquaflux.solve.StepReport` only summarizes (it reports the inner *count* and the
        *summed* cycles).

        The **iterate** is there because the inner loop is where a march's expensive linear solves
        actually live, and nothing else exposes the state at which one happened. A checkpoint is written
        at the end of a step, so it holds the state the *next* step starts from — and a step's first
        solve is the easy one, taken from a settled state with a freshly rebuilt preconditioner. On the
        three-dimensional coupled march every one of the 70 step-initial solves cost at most 2 restart
        cycles while solves later in the inner loop reached 15, so a study that probes checkpoints alone
        sees none of the hard operators and reports that every preconditioner performs identically.

        Emitted through :func:`jax.debug.callback`, so it is forward-only and transform-transparent;
        ``None`` (default) elides the call entirely, leaving the step byte-identical. Do not set it on a
        differentiated solve.
    step_limit : callable or None
        ``(phi, delta) -> alpha_max``, capping every inner line search (static). The seam for a
        constraint the residual cannot express -- chiefly a field that must stay positive, via
        :func:`~aquaflux.solve.positive_block_limit`. Without it a direct-variable field can cross
        zero and reach a ``sqrt`` in the closure, which turns a healthy state into NaN with no warning
        the guard can act on. ``None`` (default) is byte-identical.
    step_projection : callable or None
        ``(phi, delta) -> delta'``, clipping the correction itself before the line search (static). The
        per-entry form of the same positivity constraint ``step_limit`` caps globally, via
        :func:`~aquaflux.solve.positive_block_projection`. Applied FIRST, so a step carrying both then
        computes an unbinding cap of exactly ``1`` and the diagnostics keyed on the cap keep working.
        Prefer it wherever one near-zero entry would otherwise set the step length for the whole state.
        ``None`` (default) is byte-identical.
    refresh_on_cycles : int or None
        Refresh the preconditioner **inside** the step once a single solve reaches this many restart
        cycles, by calling :attr:`inner_refresh` at the iterate it reached (static). ``None`` (default)
        never refreshes mid-step and is byte-identical.

        This is a *control* seam, not the profiling :attr:`inner_observer`, and the decision is taken in
        the loop rather than by the hook so that one rule both fires the refresh and forgives the abort.
        The pairing is the point: a march's expensive inner solves are stale-preconditioner effects
        rather than hard operators — measured on a three-dimensional coupled march, a preconditioner
        rebuilt at the very iterate of the hardest solve converged in **one** cycle where the march's own
        took fifteen — so a refresh here can rescue an attempt that :attr:`abort_above_inner_cycles`
        would otherwise discard along with its pseudo-timestep. Only one refresh fires per step; a second
        expensive solve after it means the operator really is hard, and the abort then does its job.
    inner_refresh : callable or None
        ``(iterate) -> None``, rebuilding the step's frozen preconditioner at that iterate (static).
        Impure and forward-only — it mutates host state, so never set it on a differentiated solve.
    cycle_budget : int or None
        An optional cap on the inner loop's **accumulated** linear-solve count (static). When set, the
        inner loop stops as soon as its summed cycle count reaches ``cycle_budget``, so a primary solve
        grinding on a stiff low-β operator is cut after ~one over-budget inner iteration rather than
        running the full ``inner_steps`` into the restart cap (~5× the cost on the 3D coupled march). The
        partial, non-converged iterate it then returns is meant to be discarded by the march's β-escalation
        (:func:`~aquaflux.solve.forward_march` with ``retry.on_cycles < cycle_budget``), which redoes the
        step at a larger β where it converges cheaply -- so the two are paired. ``None`` (default) is
        unbounded and byte-identical. Forward-only, like the escalation it pairs with.
    abort_above_inner_cycles : int or None
        Stop the inner loop as soon as any **single** solve has cost more than this (static). Set by
        :func:`~aquaflux.solve.forward_march` from its own ``retry.on_cycles``, so the two are one number
        rather than two that must be kept in step; a caller driving this class directly may set it itself.

        This is the *same* predicate the march applies after the step returns — cost above the threshold
        with the target unmet means the whole attempt is discarded and redone at a larger shift. Applied
        only at the end, every inner iteration after the threshold is crossed is work that is thrown
        away: measured on a three-dimensional coupled march, three discarded attempts ran 26, 56 and 59
        cycles where the threshold was crossed at 14, 17 and 16.

        It cannot bin an expensive success. :func:`cond` tests the convergence target first, so a costly
        solve that *does* bring ``‖G‖`` under the target exits normally with ``reached_target`` set and is
        kept — exactly as when the march decided this after the fact. ``None`` (default) is
        byte-identical. Forward-only.
    abort_below_alpha : float or None
        Stop the inner loop once the running minimum line-search factor has fallen to this or below
        (static). The cost sibling of ``abort_above_inner_cycles``, for the *other* way an attempt is
        known to be doomed: a step length of essentially zero.

        Once the ladder cannot move — because the correction does not descend, or because a constraint
        the residual cannot express (positivity of a directly-solved field) caps the admissible length
        at nearly nothing — every further inner iteration re-solves from an unchanged iterate and
        returns the same unchanged iterate. Measured on a three-dimensional coupled march: with the
        admissible length capped at 4.4e-10, four consecutive inner iterations moved ``‖G‖`` from
        4.442e-03 to 4.440e-03 for eleven restart cycles, and the outer control then had to spend a
        whole further step per doubling of the shift to escape. The cure for both causes is the same
        one the march's escalation applies — a larger shift, which shortens the implicit step until it
        fits — so the loop exits and hands the attempt back to be redone.

        Like the cost bailout it cannot bin a success: :func:`cond` tests the convergence target first.
        ``None`` (default) is byte-identical. Forward-only.
    """

    shift_policy: ShiftPolicy
    relaxation_schedule: RelaxationSchedule = eqx.field(default_factory=SwitchedEvolutionRelaxation)
    inner_steps: int = eqx.field(static=True, default=5)
    inner_tol: float = eqx.field(static=True, default=0.05)
    line_search: int = eqx.field(static=True, default=10)
    # Data, not static (the same reasoning as `residual_norm` below): a solver configured with a
    # row-scaled stopping measure CARRIES that measure's scale arrays, and equinox rightly warns when
    # arrays are put in a static field -- static leaves take part in the jit cache key by equality, so
    # array-valued ones make the key ill-defined and, if the scales were ever rebuilt, would retrace
    # every step. As data the arrays are ordinary traced leaves and the cache key is shape/dtype only.
    # A solver with no array leaves (the default, and any plain-tolerance GMRES) filters to the static
    # side regardless, so this changes nothing for those.
    forward_solver: lx.AbstractLinearSolver | None = None
    # Data, not static (matching PseudoTransientStep): an observed march re-derives the measure each
    # outer iteration and swaps it in with `tree_at`, which needs it to ride as a leaf. The default
    # plain callable has no array leaves, so it filters to the static side and the default is unchanged.
    residual_norm: ResidualNorm = eqx.field(default=jnp.linalg.norm)
    adjoint_preconditioner_factory: (
        Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]] | None
    ) = eqx.field(static=True, default=None)
    inner_observer: Callable[..., None] | None = eqx.field(static=True, default=None)
    refresh_on_cycles: int | None = eqx.field(static=True, default=None)
    inner_refresh: Callable[[jnp.ndarray], None] | None = eqx.field(static=True, default=None)
    cycle_budget: int | None = eqx.field(static=True, default=None)
    step_limit: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = eqx.field(
        static=True, default=None
    )
    step_projection: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = eqx.field(
        static=True, default=None
    )
    abort_above_inner_cycles: int | None = eqx.field(static=True, default=None)
    abort_below_alpha: float | None = eqx.field(static=True, default=None)

    def norm(self) -> ResidualNorm:
        """The residual measure the inner loop and the outer stopping test share (:attr:`residual_norm`)."""
        return self.residual_norm

    def default_solver(self) -> lx.AbstractLinearSolver:
        """The shifted-solve Krylov solver: the injected :attr:`forward_solver`, else the shared default."""
        return (
            self.forward_solver if self.forward_solver is not None else _INEXACT_CONTINUATION_SOLVER
        )

    def adjoint_preconditioner(
        self,
    ) -> Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]] | None:
        """The (unshifted) ``state -> M`` factory for the adjoint solve at the converged state."""
        return self.adjoint_preconditioner_factory

    def stepper(self) -> _ForwardStep:
        """One backward-Euler outer timestep: the inner-converged iterate and its total solve cost.

        ``(residual_fn, φ, ‖R₀‖, solver) -> (φ_next, cycles, alpha, inner_iterations)``. The reference
        ``φⁿ`` is ``φ``; ``cycles`` is the **summed** raw solver count over the inner Newton iterations
        and ``inner_iterations`` is that count (so a consumer can recover the per-solve cost); ``alpha``
        is the **smallest** inner line-search factor (``1`` when every inner step took the full length, smaller
        when the implicit step had to be clipped, and ``0`` if any inner step failed to reduce ``‖G‖`` —
        the line search's non-descent fallback, which a step control must read as "struggling", not as a
        clean full step). It is the step-quality signal a :class:`~aquaflux.solve.StepControl` drives
        ``β`` by: raise the pseudo-timestep while ``α = 1``, back off when it clips or fails to descend.
        """
        policy = self.shift_policy
        schedule = self.relaxation_schedule
        inner_steps = self.inner_steps
        inner_tol = self.inner_tol
        line_search = self.line_search
        norm = self.residual_norm
        inner_observer = self.inner_observer
        refresh_on_cycles = self.refresh_on_cycles
        inner_refresh = self.inner_refresh
        cycle_budget = self.cycle_budget

        def _refresh_if_due(due, iterate) -> None:
            """Host side of the mid-step refresh: the loop has already decided, this just obeys."""
            if bool(due):
                inner_refresh(iterate)

        step_limit = self.step_limit
        step_projection = self.step_projection
        abort_above = self.abort_above_inner_cycles
        abort_below = self.abort_below_alpha

        def step(
            residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
            phi: jnp.ndarray,
            residual_norm_0: jnp.ndarray,
            solver: lx.AbstractLinearSolver,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            reference = phi  # φⁿ, held across the inner loop
            # ‖R(φⁿ)‖ = ‖G(φⁿ)‖: the transient term β d (φ − φⁿ) is zero at the anchor, so the honest
            # steady residual at the anchor and the inner loop's starting G-norm are the same number.
            reference_norm = norm(residual_fn(reference))
            relaxation = schedule.relaxation(reference_norm, residual_norm_0)
            term = policy.shift_term(reference)
            # The shift only reshapes the forward path (like the preconditioner it damps), so detach it:
            # it never perturbs the converged state or its adjoint.
            shift = jax.lax.stop_gradient(relaxation * term.diagonal)  # β d over the full state
            preconditioner = term.make_preconditioner(relaxation)
            target = inner_tol * reference_norm

            def transient_residual(p: jnp.ndarray) -> jnp.ndarray:
                # G(p) = R(p) + β d (p − φⁿ): the backward-Euler residual whose root is the implicit
                # timestep. Its Jacobian is J_R(p) + β d, formed matrix-free below.
                return residual_fn(p) + shift * (p - reference)

            def cond(carry: tuple) -> jnp.ndarray:
                _, inner, gnorm, cycles, min_alpha, _, since_refresh, _, _ = carry
                # The convergence target is tested FIRST, and that ordering is what makes the cost
                # bailouts below safe: a solve that was expensive but brought ‖G‖ under the target exits
                # here with `reached_target` set and is kept, never binned for its cost.
                keep = (inner < inner_steps) & (gnorm > target)
                # Cost bailout: stop the inner loop once its accumulated linear-solve count reaches
                # `cycle_budget`, so a primary solve that is grinding on a stiff low-β operator is cut off
                # after ~one over-budget inner iteration (~`cycle_budget` matvecs) instead of running the
                # full `inner_steps` into the restart cap (measured ~5× the cost on the 3D coupled march).
                # The partial, non-converged iterate this returns is meant to be discarded by the march's
                # β-escalation (`forward_march(retry.on_cycles<cycle_budget)`), which redoes the step at a
                # larger β where it converges cheaply -- so pair the two. `cycle_budget=None` (default) is
                # byte-identical (the budget term is elided at trace time, as `cycle_budget` is static).
                if cycle_budget is not None:
                    keep = keep & (cycles < cycle_budget)
                # Doomed-attempt bailout: one solve has already cost more than the march's discard
                # threshold, and the target is still unmet (the test above), so this attempt WILL be
                # thrown away and redone at a larger shift. Every further inner iteration is work that
                # is discarded. Checking it here rather than after the step is the whole point: the
                # threshold is a per-solve quantity, so it can be known the moment a solve returns.
                # Measured on cycles since the LAST REFRESH, not since the step began. The march's
                # expensive inner solves are stale-preconditioner effects rather than hard operators, so
                # when `inner_refresh` has just rebuilt at this iterate the attempt deserves one more
                # solve before being written off -- otherwise the refresh is paid for and then discarded
                # along with the step, which is what happened before this counter was split out. With no
                # refresh hook the two counters are identical and this is byte-identical to testing the
                # step's maximum.
                if abort_above is not None:
                    keep = keep & (since_refresh <= abort_above)
                # Doomed-attempt bailout, the other way an attempt dies: the ladder can no longer move.
                # `min_alpha` is zero when an iteration failed to descend and tiny when a positivity cap
                # bound the length, and neither recovers within the step -- the iterate is unchanged, so
                # the next solve re-derives the same correction and the same cap. Every further iteration
                # is therefore a re-solve that returns where it started. The march redoes the attempt at a
                # larger shift, which is the cure for both causes.
                if abort_below is not None:
                    keep = keep & (min_alpha > abort_below)
                return keep

            def body(carry: tuple) -> tuple:
                p, inner, gnorm, cycles, min_alpha, max_inner, since_refresh, spent, binding = carry
                delta, step_cycles = _shifted_solve(
                    residual_fn, p, transient_residual(p), shift, preconditioner, solver
                )
                # Strict-descent (monotone) line search on ‖G‖: G = 0 is a well-posed fixed-φⁿ solve, so
                # a clipped inner step is a signal the pseudo-timestep is too large, read out via alpha.
                # Cap the ladder by whatever constraint the residual cannot express (positivity of a
                # directly-solved field). `None` leaves the cap at 1, i.e. no cap.
                # Clip the correction per entry BEFORE the cap reads it: an entry that would cross
                # zero is held back on its own, rather than shortening the step for every entry. The
                # cap below then finds nothing binding and returns 1 (see
                # `positive_block_projection`), so the two compose and the cap's diagnostics survive.
                if step_projection is not None:
                    delta = step_projection(p, delta)
                max_alpha = 1.0 if step_limit is None else step_limit(p, delta)
                candidate, alpha = backtracking_line_search(
                    transient_residual, p, delta, gnorm, line_search, norm=norm, max_alpha=max_alpha
                )
                new_gnorm = norm(transient_residual(candidate))
                # An inner step that does NOT reduce ‖G‖ took the line search's non-descent fallback,
                # which still reports alpha = 1 (the longest finite rung ≤ the full step). Fold that into
                # the reported min-alpha as 0, so a step control reads it as "struggling" (shrink the
                # pseudo-timestep) rather than "comfortable" (grow it). The monotone search means no
                # descent ⇔ the fallback fired, so `new_gnorm < gnorm` detects it exactly.
                descended = new_gnorm < gnorm
                if inner_observer is not None:
                    # Surface this inner iteration's trajectory (‖G‖ before/after, its solve's raw cycle
                    # count, its line-search factor) *and the iterate it reached*. `jax.debug.callback`
                    # fires during execution and is a no-op under differentiation; `None` (the default)
                    # elides this branch at trace time.
                    jax.debug.callback(
                        inner_observer,
                        inner,
                        gnorm,
                        new_gnorm,
                        step_cycles,
                        alpha,
                        candidate,
                        ordered=True,
                    )
                corrected = corrected_cycles(step_cycles)
                # Refresh the preconditioner mid-step once a solve gets expensive, and give the rebuilt
                # one a fair hearing by restarting the abort's counter. The decision is made HERE, in the
                # traced loop, and handed to the host hook -- so the rule that fires the refresh and the
                # rule that forgives the abort are one rule, not two that can drift apart.
                if refresh_on_cycles is not None and inner_refresh is not None:
                    due = (corrected >= refresh_on_cycles) & jnp.logical_not(spent)
                    jax.debug.callback(_refresh_if_due, due, candidate, ordered=True)
                else:
                    due = jnp.asarray(False)
                return (
                    candidate,
                    inner + 1,
                    new_gnorm,
                    cycles + step_cycles,
                    jnp.minimum(min_alpha, jnp.where(descended, alpha, 0.0)),
                    # The most expensive SINGLE solve, which is the inner-count-invariant difficulty
                    # signal: the summed count above also counts how many times the step solved. Kept
                    # un-reset by a refresh, so the step still REPORTS how hard it really was -- that is
                    # the signal a study picks its probe states by, and hiding a refresh in it would make
                    # the hardest steps look benign.
                    jnp.maximum(max_inner, corrected),
                    # ...whereas the abort's counter restarts, so the refreshed preconditioner is judged
                    # on its own solve rather than on the one that provoked it.
                    jnp.where(due, 0, jnp.maximum(since_refresh, corrected)),
                    spent | due,
                    # The cap only where it was the BINDING constraint: `alpha` reaching it means the
                    # ladder wanted a longer step and the limit, not the descent test, stopped it.
                    jnp.minimum(binding, jnp.where(alpha >= max_alpha, max_alpha, 1.0)),
                )

            (
                phi_next,
                inner_iterations,
                final_gnorm,
                cycles,
                alpha,
                max_inner,
                _,
                _,
                binding,
            ) = jax.lax.while_loop(
                cond,
                body,
                (
                    reference,
                    jnp.asarray(0, dtype=jnp.int32),
                    reference_norm,
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(1.0),
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(False),
                    jnp.asarray(1.0),
                ),
            )
            # `cycles` is the SUM of the inner solves' raw counts; report the inner-iteration count
            # alongside it so the two costs (nonlinear inner work vs linear solve cost) are not conflated.
            # `reached_target` says the loop exited on its OWN tolerance rather than being cut short by
            # `cycle_budget` or `inner_steps`. Without it a march escalating on cost alone discards a step
            # that converged expensively, which is pure waste: it throws away a good iterate AND takes a
            # shorter step than the work already earned.
            return StepOutcome(
                phi_next, cycles, alpha, inner_iterations, final_gnorm <= target, max_inner, binding
            )

        return step
