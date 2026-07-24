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

from .implicit import _ForwardStep, backtracking_line_search
from .linear import solve_linear
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
        :data:`_INEXACT_CONTINUATION_SOLVER` when set (static). A stiff coupled system whose shifted
        operator needs a larger Krylov subspace to converge without restarting can pass a
        larger-``restart`` GMRES here; ``None`` uses the shared default.
    residual_norm : ResidualNorm
        The residual measure ``R -> scalar`` the march judges progress by (static, default the
        Euclidean norm): the switched-evolution-relaxation ramp ``β = β₀(‖R‖/‖R₀‖)^p``, the line
        search, and the acceptance/divergence guard all use it, and :class:`ImplicitNewtonSolver`
        reads it (via :meth:`norm`) for the outer stopping test, so one measure governs the whole
        solve. A heterogeneous block system (e.g. coupled RANS, where ``omega`` is O(1e5) and ``k``
        O(1e-3)) *can* pass a :class:`~aquaflux.solve.BlockScaledNorm` so the march *sees* every block
        — with the plain norm the ‖R‖ is ~100% ``omega`` and the line search neither judges nor
        protects the ``k`` block. (In practice the block-scaled measure stalled the coupled march, so
        coupled RANS defaults to the Euclidean norm; the block-scaled option remains for
        experimentation.) Like the shift, it only reshapes the forward path; the IFT adjoint never
        forms a norm, so the converged state and its gradient are unchanged.
    adjoint_preconditioner_factory : callable or None
        The ``state -> M`` preconditioner factory for the converged transpose (adjoint) solve, or
        ``None`` for an unpreconditioned adjoint (static). At ``φ*`` the operator is the
        well-conditioned steady Jacobian (``β → 0``), so the adjoint needs no shift — the ordinary
        (unshifted) preconditioner is the consistent choice.
    """

    shift_policy: ShiftPolicy
    relaxation_schedule: RelaxationSchedule = eqx.field(default_factory=SwitchedEvolutionRelaxation)
    max_escalations: int = eqx.field(static=True, default=6)
    escalation_factor: float = eqx.field(static=True, default=2.0)
    acceptance: StepAcceptance = eqx.field(default_factory=DivergenceGuard)
    line_search: int = eqx.field(static=True, default=0)
    forward_solver: lx.AbstractLinearSolver | None = eqx.field(static=True, default=None)
    residual_norm: ResidualNorm = eqx.field(static=True, default=jnp.linalg.norm)
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

        ``(residual_fn, φ, ‖R₀‖, solver) -> (φ_next, cycles, alpha)``. ``cycles`` is the restart-cycle
        count of the **accepted** attempt's shifted linear solve — the cost of the step that was
        actually taken, not the sum over rejected escalation attempts; ``alpha`` is that attempt's
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
        max_escalations, escalation_factor = self.max_escalations, self.escalation_factor
        acceptance = self.acceptance
        line_search = self.line_search
        norm = self.residual_norm

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

            def attempt(
                relaxation: jnp.ndarray,
            ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                # The shift only reshapes the forward path (like the preconditioner it damps), so it
                # is detached: it never perturbs the converged state or its adjoint.
                shift = jax.lax.stop_gradient(relaxation * term.diagonal)  # β d over the full state

                def shifted_jacobian(tangent: jnp.ndarray) -> jnp.ndarray:
                    # True Jacobian-vector product plus the pseudo-transient diagonal on shifted DOFs.
                    jvp = jax.jvp(residual_fn, (phi,), (tangent,))[1]
                    return jvp + shift * tangent

                # Preconditioner inverts the *same* shifted operator it is damped by. The solve does
                # not throw: a non-convergent shifted system yields a candidate the acceptance test
                # rejects (triggering more damping), rather than raising.
                preconditioner = term.make_preconditioner(relaxation)
                delta, cycles = solve_linear(
                    shifted_jacobian,
                    -residual,
                    solver=solver,
                    preconditioner=preconditioner,
                    throw=False,
                )
                # Backtrack the step length before judging it: when the shifted direction is accurate
                # but the full step overshoots, a scaled-back step descends from this one solve,
                # sparing a re-solve at larger beta. `line_search == 0` takes the full step (alpha = 1).
                candidate, alpha = backtracking_line_search(
                    residual_fn, phi, delta, residual_norm, line_search, norm=norm
                )
                return candidate, norm(residual_fn(candidate)), cycles, alpha

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

            def body(state: tuple) -> tuple:
                relaxation, best, attempts, _, best_cycles, best_alpha = state
                candidate, candidate_norm, cycles, alpha = attempt(relaxation)
                accept = acceptance.accept(candidate_norm, residual_norm, residual_norm_0, attempts)
                best = jnp.where(accept, candidate, best)
                # Report the cycles and line-search factor of the attempt actually taken, not the
                # rejected escalations'. (A fully-rejected step keeps the initial 0 / 1.)
                best_cycles = jnp.where(accept, cycles, best_cycles)
                best_alpha = jnp.where(accept, alpha, best_alpha)
                return (
                    relaxation * escalation_factor,
                    best,
                    attempts + 1,
                    accept,
                    best_cycles,
                    best_alpha,
                )

            _, phi_next, _, _, step_cycles, step_alpha = jax.lax.while_loop(
                cond,
                body,
                (
                    base_relaxation,
                    phi,
                    0,
                    jnp.asarray(False),
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(1.0),
                ),
            )
            return phi_next, step_cycles, step_alpha

        return step
