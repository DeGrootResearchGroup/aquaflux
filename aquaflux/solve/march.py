"""An observed, forward-only Newton march, and the staleness trigger that watches it.

:class:`~aquaflux.solve.ImplicitNewtonSolver` runs its Newton march inside a ``lax.while_loop`` and
returns only the converged field. That is exactly right for the differentiable solve — the loop is
never taped, and the implicit-function-theorem adjoint is one transpose solve at the root — but it
makes the march *opaque*: nothing outside can see what each step cost, and nothing can stop the loop
part way to do work that cannot run under ``jit``.

Refreshing a frozen algebraic-multigrid (AMG) preconditioner mid-march needs both. The rebuild
assembles ``scipy`` sparse matrices, which cannot happen inside a traced loop, and the *decision* to
rebuild is made from the per-step linear-solve cost. So this module adds a second march — an eager
Python loop, :func:`forward_march` — that steps the **same** injected
:class:`~aquaflux.solve.ForwardStep`, judges convergence with the **same** tolerance test, and
measures progress with the **same** residual norm, but observes every step and may stop early.

**The eager march never returns the answer.** It is a pure accelerator: a driver uses it to reach a
better-preconditioned state, and then finishes with a real ``ImplicitNewtonSolver.solve()``, which
owns the convergence guard, the ``custom_vjp``, and the returned field. That is why
:func:`forward_march` deliberately has **no** non-convergence guard of its own — stopping short is
its purpose, and a state it hands back is an intermediate, never a result. Keeping the guard in one
place means a march that ends short of a root can never be mistaken for a converged one.

**Two reference residual norms, and conflating them breaks the march.** Each call to
:func:`forward_march` computes its own ``residual_norm_0`` from the state it is handed, and passes
*that* to the step. The pseudo-transient schedule ramps its damping as ``beta = beta_0 (‖R‖/‖R₀‖)^p``,
so a segment restarted after a refresh must restart its ramp too. (A refresh **carries** the shift
diagonals rather than rebuilding them -- rebuilding them was measured to freeze the march -- so the
reason is not a grown diagonal: it is that the ramp is defined relative to where the segment began,
and feeding it a residual ratio measured against a different state makes ``beta`` mean something
else. Note the consequence, which is easy to miss: with frequent refreshes the ratio never falls
far below one, so ``beta`` stays pinned near ``beta_0`` for the whole march.) The separate ``reference_norm`` is the *global* scale
progress is reported and tested against, held fixed across every segment so that "converged" and the
reported ratio mean the same thing throughout. The first must never be substituted for the second.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple, Protocol

import equinox as eqx
import jax.numpy as jnp
import lineax as lx

from .implicit import ForwardStep, _within_tolerance
from .norm import ResidualNorm


class StepReport(NamedTuple):
    """What one march step cost and where it left the residual.

    Attributes
    ----------
    step : int
        The 0-based index of the step within its march segment.
    cycles : int
        The **raw** solver iteration count behind the accepted step, summed over the step's inner Newton
        iterations (:attr:`inner_iterations`). This is lineax's ``num_steps``, which carries a **+2
        offset and is blind within a restart cycle**: a solve that converges in 1, a few, or up to
        ``restart`` matvecs all report ``3``, and it only climbs once a solve genuinely spills past a
        restart cycle. So ``cycles`` is a raw cost primitive, **not** a literal cycle count — read
        :attr:`restart_cycles` (offset-corrected) for the honest per-step cycle count and
        :meth:`matvecs` for the matvec estimate. **``0`` means "no measurement", not "free":** a
        pseudo-transient step records its count only on acceptance, so a step whose every damping
        attempt was rejected reports ``0`` despite having burned several solves — skip zeros.
    inner_iterations : int
        How many inner Newton iterations the step took: the backward-Euler inner-loop count for a
        :class:`~aquaflux.solve.DualTimeStep` (what the summed :attr:`cycles` is spread over), and ``1``
        for a single-step (pseudo-transient / damped-Newton) march. Reporting it separately from
        :attr:`cycles` is what keeps the two costs — nonlinear inner work vs linear solve cost — from
        being conflated into one misleading number.
    residual_norm : float
        The residual measure at the state the step produced.
    residual_ratio : float
        ``residual_norm`` divided by the march's global reference norm — how far the solve has come,
        on the same scale for every segment.
    alpha : float
        The line-search factor of the accepted step: ``1`` if the full shifted step descended, smaller
        if it was clipped. The step-quality signal a :class:`StepControl` drives the next shift by
        (``α < 1`` means the step overshot — the shift is too weak). ``1`` for a step with no line
        search, and for a fully-rejected step.
    drift : float
        How far the frozen operator's coefficients have moved since the segment's reference state, as
        a relative measure, from the march's injected ``drift_measure``. **The staleness signal a
        :class:`CoefficientDriftTrigger` fires on**, and the one quantity here that reports on the
        *preconditioner* rather than on the step. ``0.0`` when no measure was supplied — "not
        measured", like ``cycles = 0``, and it fails closed because a drift trigger compares against a
        positive threshold.

        It is a **scalar**, deliberately: computing it needs the state, but putting the state on the
        report would cost the replay property that makes trigger calibration cheap (see
        ``forward_march``'s ``checkpoint``). Reducing it to a number here keeps a trigger a pure
        function of numbers while still letting it see the physics.
    """

    step: int
    cycles: int
    residual_norm: float
    residual_ratio: float
    alpha: float
    drift: float = 0.0
    inner_iterations: int = 1

    @property
    def restart_cycles(self) -> int:
        """The offset-corrected restart-cycle count.

        ``cycles`` with lineax's +2-per-inner-solve offset removed, so an ideal one-cycle solve reads as
        ``1`` and a dual-time step as its real total cycles over the inner loop. Clamped at ``0`` (a
        no-measurement ``cycles = 0`` step stays ``0``).
        """
        return max(self.cycles - 2 * self.inner_iterations, 0)

    def matvecs(self, restart: int) -> int:
        """Upper-bound matvec estimate: :attr:`restart_cycles` times the GMRES ``restart`` length.

        An upper bound because ``num_steps`` is cycle-granular -- blind within the final partial cycle,
        which holds anywhere from 1 to ``restart`` matvecs.
        """
        return self.restart_cycles * restart


class MarchResult(NamedTuple):
    """The outcome of one :func:`forward_march` segment.

    Attributes
    ----------
    state : jnp.ndarray
        The state the march reached. An **intermediate** unless :attr:`converged` is ``True``; it
        carries no guarantee of solving the residual, so it must be finished by a real solve.
    reports : tuple of StepReport
        One report per step taken, in order.
    converged : bool
        Whether the march reached the requested tolerance against the global reference norm.
    triggered : bool
        Whether the march stopped early because the injected trigger fired.
    control_state : object
        The injected :class:`StepControl`'s carried state as of the last step (``None`` if no control
        was used). Returned so a driver running one segment per preconditioner refresh can **thread** it
        into the next segment's :func:`forward_march`, rather than each segment restarting the control
        from scratch — a stateful control (e.g. one climbing the shift strength over many steps) would
        otherwise throw its progress away at every refresh.
    """

    state: jnp.ndarray
    reports: tuple[StepReport, ...]
    converged: bool
    triggered: bool
    control_state: object = None


class RefreshTrigger(Protocol):
    """Decides, from a march's step history, whether the frozen preconditioner should be rebuilt.

    Structural interface only (a ``Protocol``). A trigger is a **pure function of the history**: it
    holds no state across calls, so the same history always yields the same answer. That is what
    lets a candidate trigger be replayed offline against a march that was logged once, instead of
    each parameter change costing another full solve. How *many* times a march may refresh is the
    driver's choice, not the trigger's.
    """

    def should_refresh(self, history: Sequence[StepReport]) -> bool:
        """Whether to rebuild the preconditioner, given every step of the current segment so far."""


class CycleGrowthTrigger(eqx.Module):
    """Fire when the per-step linear-solve cost has grown, once the flow is developed.

    A frozen preconditioner going stale shows up as a **rising restart-cycle count on a system that
    is otherwise unchanged**, which is why the cycle count is the signal rather than the residual
    history or the wall clock (the clock moves with machine load and tells you nothing about the
    linear algebra).

    **The confound this class exists to handle.** The cycle count rises for two independent reasons:
    the preconditioner drifting from the operator (the signal), and the pseudo-transient damping
    ``beta`` ramping toward zero as the residual falls, which ill-conditions the shifted system and
    raises the count *whether or not anything is stale*. On a backward-facing step the second effect
    was measured to be the **larger** of the two, so a bare "cost has doubled" rule fires early, from
    damping alone. Firing early is not merely wasted work: rebuilding before the flow separates was
    measured to roughly **double** the cycle count, on top of the rebuild and recompilation it costs.

    Since ``beta`` is a function of the residual ratio alone, the ratio is used as the **gate** that
    the flow has actually developed — demoted from being the trigger to guarding it — while the cost
    growth remains the trigger. Both must hold, and hold for :attr:`patience` steps running.

    Attributes
    ----------
    growth : float
        Fire only when a step's cycle count reaches this multiple of the segment's cheapest step
        (static). The baseline is the **running minimum** over the segment's non-zero counts, which
        is the most conservative available and is not anchored on an atypical first solve.
    max_residual_ratio : float
        Fire only once the residual has fallen to this fraction of the global reference (static) —
        the developed-flow gate. The refresh pays only after the flow has separated; before that it
        is worthless at best and a large regression at worst.
    warmup : int
        Ignore this many leading steps of a segment (static). The opening steps run at the largest
        damping, from an initial condition where the preconditioner is fresh by construction, so
        their cost is not representative.
    patience : int
        Require the growth condition to hold on this many consecutive most-recent steps (static).
        A single expensive step — a transiently stiff state, or one that escalated its damping —
        must not buy a rebuild and a recompilation.

    Notes
    -----
    The defaults are **provisional**: they are shaped to be conservative (late rather than early),
    not calibrated. The cycle count as a function of damping and staleness has no closed form, so
    the numbers have to come from an instrumented march. Because this trigger is a pure function of
    a :class:`StepReport` history, that calibration is done by logging one march with
    ``trigger=None`` and replaying candidate parameters against the log, with no further solves.
    """

    growth: float = eqx.field(static=True, default=2.0)
    max_residual_ratio: float = eqx.field(static=True, default=5e-2)
    warmup: int = eqx.field(static=True, default=5)
    patience: int = eqx.field(static=True, default=2)

    def should_refresh(self, history: Sequence[StepReport]) -> bool:
        """Whether the segment's history shows a sustained, developed-flow cost rise.

        Parameters
        ----------
        history : sequence of StepReport
            Every step of the current march segment, in order.

        Returns
        -------
        bool
            ``True`` when all of: the warmup is past; the latest step is at or below
            :attr:`max_residual_ratio`; and the last :attr:`patience` steps each measured at least
            :attr:`growth` times the segment's cheapest measured step.
        """
        if len(history) <= self.warmup or len(history) < self.patience:
            return False
        if history[-1].residual_ratio > self.max_residual_ratio:
            return False
        # Zero counts are "no measurement" (a step whose every damping attempt was rejected, or a
        # direct solver reporting nothing). They must not set the baseline: a zero minimum would
        # make every subsequent step "grown" and latch the trigger on permanently.
        measured = [report.cycles for report in history if report.cycles > 0]
        if not measured:
            return False
        threshold = self.growth * min(measured)
        recent = history[-self.patience :]
        return all(report.cycles > 0 and report.cycles >= threshold for report in recent)


class CoefficientDriftTrigger(eqx.Module):
    """Fire when the frozen operator's coefficients have drifted from the state they were frozen at.

    The **direct** staleness signal, and the reason it is preferred to
    :class:`CycleGrowthTrigger`: a preconditioner is stale exactly when the operator it approximates
    has moved, so measuring that movement asks the question directly instead of inferring it from
    cost. On a coupled RANS march the moving coefficient is the eddy viscosity ``nu_t`` -- it is what
    the frozen scalar-transport operators are built from, and it changes by orders of magnitude as a
    recirculation develops.

    **Why this is a better trigger than watching the cycle count.** The cycle count rises for two
    independent reasons -- the preconditioner drifting (the signal) and the pseudo-transient damping
    ``beta`` ramping toward zero, which ill-conditions the shifted system whether or not anything is
    stale -- and on a backward-facing step the second was measured to be the *larger*. So
    :class:`CycleGrowthTrigger` needs a residual-ratio **gate** to suppress the confound, plus
    ``patience`` because a single expensive step must not buy a rebuild. Coefficient drift has
    neither problem: it does not respond to ``beta`` at all, so it needs no gate, and it moves
    smoothly with the flow rather than jumping with one stiff solve, so it needs no patience. It is
    also *unlikely* to fire before the flow develops -- which is when a refresh was measured to be
    actively harmful -- because an undeveloped flow is largely one whose coefficients have not moved.
    That is an argument, not a guarantee: nothing enforces it, and an initial condition being repaired
    early in a march can move ``nu_t`` without the flow separating, so :attr:`warmup` is the only
    actual guard.

    Attributes
    ----------
    threshold : float
        Fire once the reported drift reaches this value (static). The measure is relative, so ``0.1``
        means "the coefficients have moved by a tenth of their magnitude at the freeze state".
    warmup : int
        Ignore this many leading steps of a segment (static). A segment's opening steps are measured
        against a preconditioner that is fresh by construction.

    Notes
    -----
    The default threshold is **calibrated on one instrumented march** -- a backward-facing step at
    Reynolds ~ 4.7e4 from a cold start -- rather than guessed, but it is one case, so treat it as a
    starting point for a new geometry rather than a universal constant. On that march the restart-cycle
    count sat flat near its floor while the drift climbed, then rose steeply with it:

    ==========  =====  =====  =====  =====  =====  =====
    drift        0.04   0.07   0.11   0.15   0.19   0.24
    cycles         10     13     21     33     53     84
    ==========  =====  =====  =====  =====  =====  =====

    ``0.1`` is where the cost has just doubled off its floor and the recirculation has formed, which
    is early enough to skip the steep part entirely and late enough to stay out of the pre-separation
    regime where a rebuild was measured to make the solve *worse*. Re-calibrating is deliberately
    cheap: because this is a pure function of a :class:`StepReport` history, log one march with
    ``trigger=None`` and a ``drift_measure``, then replay candidate thresholds against the log with no
    further solves.
    """

    threshold: float = eqx.field(static=True, default=0.1)
    warmup: int = eqx.field(static=True, default=3)

    def should_refresh(self, history: Sequence[StepReport]) -> bool:
        """Whether the coefficients have drifted past :attr:`threshold` since the segment began.

        Parameters
        ----------
        history : sequence of StepReport
            Every step of the current march segment, in order.

        Returns
        -------
        bool
            ``True`` once the warmup is past and the latest step's drift reaches the threshold. A
            march with no ``drift_measure`` reports ``0.0`` and so never fires.
        """
        if len(history) <= self.warmup:
            return False
        return history[-1].drift >= self.threshold


class StepControl(Protocol):
    """Reshapes the forward step each iteration from the march's own feedback (forward-only).

    Where a :class:`~aquaflux.solve.RelaxationSchedule` is a *memoryless* rule that lives on the
    differentiable step, a step control is **stateful and reads the previous step's outcome** — the
    line-search factor α, the cost, the residual — to decide the next step. That feedback is only
    available *after* a step, and a control may raise under ``jax.grad``, so it lives here on the eager
    march, alongside :class:`RefreshTrigger`, never on the traced Newton path.

    ``next_step`` returns a ready-to-run :class:`~aquaflux.solve.ForwardStep` (typically ``base_step``
    with its shift strength replaced, via :class:`~aquaflux.solve.ConstantRelaxation` on a dynamic β
    leaf so :func:`_march_step` stays a compilation-cache hit) plus its own updated state. The march
    threads that state and stays ignorant of what the control adjusts, so it works for any
    ``ForwardStep`` — the control, not the march, knows about β.
    """

    def next_step(
        self, base_step: ForwardStep, previous: StepReport | None, state: object
    ) -> tuple[ForwardStep, object]:
        # NOTE: the shipped control reshapes the shift strength, so it requires a
        # `PseudoTransientStep` specifically -- the annotation is wider than the real contract, and
        # passing a `DampedNewtonStep` raises `AttributeError` inside the march loop rather than
        # being rejected at the seam. Narrow this if a second, step-agnostic control ever appears.
        """The step to run next, and the control's carried state.

        Parameters
        ----------
        base_step : ForwardStep
            The march's base step, whose non-shift configuration (preconditioner, line search, norm)
            the control reuses.
        previous : StepReport or None
            The report of the step just taken (``None`` before the first step) — the feedback the
            control adapts on.
        state : object
            The control's own state from the previous call (``None`` on the first call).
        """


def _has_diverged(residual_norm: jnp.ndarray, reference: float, divergence_cap: float) -> bool:
    """Whether a step's residual norm signals a diverged step the retry should redo.

    True if the norm is non-finite, or -- when a finite ``divergence_cap`` is set -- if it exceeds
    ``divergence_cap * reference``. A non-finite norm is the load-bearing case (an inexact
    preconditioner returning a poisoned correction on a stiff operator); the cap is an optional extra
    for a finite blow-up, and defaults to off (``inf``) so only non-finiteness triggers a retry.
    """
    if not bool(jnp.isfinite(residual_norm)):
        return True
    return (
        divergence_cap < float("inf")
        and reference > 0.0
        and (float(residual_norm) > divergence_cap * reference)
    )


@eqx.filter_jit
def _march_step(
    forward_step: ForwardStep,
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    phi: jnp.ndarray,
    residual_norm_0: jnp.ndarray,
    solver: lx.AbstractLinearSolver,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One observed step: the next state, its raw cycle count, line-search factor, inner-iteration count, and new residual norm.

    Compiled as a unit, and — this is the load-bearing part — ``forward_step`` and ``residual_fn``
    are **arguments, not captured values**, so repeated steps hit the compilation cache instead of
    retracing the shifted solve every iteration (which would dominate the whole march). Two things
    are required of the caller for that to hold:

    * pass the **same** ``forward_step`` object for every step of a segment (a rebuilt one is a new
      compilation, which is the intended one-off cost of a refresh); and
    * pass a **bound method** of a module as ``residual_fn`` (e.g. ``coupled.residual``), which is a
      pytree whose arrays ride as dynamic leaves. A freshly-created ``lambda`` is hashed by identity,
      so building one per step misses the cache every time.

    The next residual norm is returned from inside this same compiled call so the march does not pay
    a second, separate residual evaluation per step.
    """
    phi_next, cycles, alpha, inner = forward_step.stepper()(
        residual_fn, phi, residual_norm_0, solver
    )
    return phi_next, cycles, alpha, inner, forward_step.norm()(residual_fn(phi_next))


def forward_march(
    forward_step: ForwardStep,
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    phi0: jnp.ndarray,
    *,
    max_steps: int,
    rtol: float,
    atol: float,
    reference_norm: float | None = None,
    trigger: RefreshTrigger | None = None,
    step_control: StepControl | None = None,
    control_state: object = None,
    observer: Callable[[StepReport], None] | None = None,
    checkpoint: Callable[[StepReport, jnp.ndarray], None] | None = None,
    drift_measure: Callable[[jnp.ndarray], float] | None = None,
    norm_builder: Callable[[jnp.ndarray], ResidualNorm] | None = None,
    precondition_step: Callable[[ForwardStep, jnp.ndarray], None] | None = None,
    solver: lx.AbstractLinearSolver | None = None,
    retry_solver: lx.AbstractLinearSolver | None = None,
    retry_divergence_cap: float = float("inf"),
    retry_on_cycles: int | None = None,
    retry_beta_factor: float = 2.0,
    retry_cycles_limit: int = 2,
) -> MarchResult:
    """March the residual eagerly, reporting each step and stopping early if the trigger fires.

    A forward-only counterpart to :class:`~aquaflux.solve.ImplicitNewtonSolver`'s traced march,
    for a driver that must observe per-step cost or interpose work that cannot run under ``jit``
    (rebuilding a frozen preconditioner). It applies the same injected ``forward_step``, the same
    residual measure (``forward_step.norm()``), and the same stopping test, so the two marches take
    the same path on the same problem.

    **This function may return a state that does not solve the residual, without raising** — that is
    the point of a march that can stop early. It carries no convergence guard; the caller must
    finish with an ``ImplicitNewtonSolver.solve()``, which does, and which produces the actual
    result and its adjoint. Do not differentiate through this march.

    Parameters
    ----------
    forward_step : ForwardStep
        The globalized step strategy to apply. The **same object** must be used for every step of a
        segment, or each step recompiles.
    residual_fn : callable
        The single-argument residual ``phi -> R(phi)``. Pass a bound module method rather than a
        freshly-built closure (see :func:`_march_step`).
    phi0 : jnp.ndarray
        The state to march from.
    max_steps : int
        Maximum steps this segment may take.
    rtol, atol : float
        Stopping tolerances, tested as ``‖R‖ <= atol + rtol * reference_norm``.
    reference_norm : float, optional
        The **global** residual scale to judge progress against, held fixed across every segment of
        a staged solve. Defaults to the norm at ``phi0``, which is correct for a single segment.
        This is deliberately *not* the same quantity as the damping schedule's reference, which is
        always recomputed per segment from ``phi0`` (see the module docstring).
    trigger : RefreshTrigger, optional
        Consulted after every step; when it fires the march stops and reports ``triggered=True``.
        ``None`` marches to convergence or ``max_steps``.
    step_control : StepControl, optional
        Reshapes the step each iteration from the previous step's report (e.g. driving the shift
        strength β toward a line-search-factor target). ``None`` runs ``forward_step`` unchanged, so
        the march is byte-identical to an uncontrolled one. Forward-only, like ``trigger``.
    control_state : object, optional
        The initial state for ``step_control`` (``None`` on a fresh march). A driver that runs one
        segment per refresh passes the previous segment's :attr:`MarchResult.control_state` here, so a
        stateful control continues across the refresh instead of resetting — the same discipline the
        *global* ``reference_norm`` follows, and the opposite of the deliberately segment-local damping
        reference and ``drift_measure``. Ignored when ``step_control is None``.
    observer : callable, optional
        Called with each :class:`StepReport` as it is produced, for streaming progress out of a long
        march. The full history is also returned, so an observer is only needed for live reporting.
    checkpoint : callable, optional
        Called with ``(report, state)`` after each step — the same report the observer sees, plus the
        state that produced it. For saving intermediate states of a long march, so a later study can
        re-solve at a chosen point without re-marching to it, and so a crash costs steps rather than
        the whole run.

        **Deliberately separate from** ``observer``, rather than putting the state on the report.
        A :class:`RefreshTrigger` reads the report history, and keeping that history purely numeric is
        what makes a trigger a pure function that can be replayed offline against a logged march. If
        the state travelled on the same seam, a trigger could reach into the physics and that replay
        property — the reason trigger calibration costs one logged run instead of one run per
        candidate — would be lost.
    drift_measure : callable, optional
        ``state -> float``, a relative measure of how far the frozen operator's coefficients have
        moved from the ones this segment's preconditioner was built at. Evaluated once per step and
        reported as :attr:`StepReport.drift`, which is what a
        :class:`CoefficientDriftTrigger` fires on. ``None`` reports ``0.0`` throughout.

        **It must be re-based per segment**, against the state the current preconditioner was frozen
        at — the same discipline as the segment-local damping reference. Measuring drift from a state
        older than the last refresh would report movement that has already been absorbed and refresh
        again immediately.
    precondition_step : callable, optional
        ``(active_step, state) -> None``, called before each step (after the control has set the shift
        strength on ``active_step``) to refresh that step's frozen host preconditioner from the current
        state and shift. It runs in this eager loop -- a host operation outside the jitted ``_march_step``
        -- and mutates the step's *static* preconditioner in place, so ``_march_step`` stays a
        compilation-cache hit. The use case is a **complete-LU preconditioner re-factored at the current
        ``(state, β)``** so it is the exact inverse of the operator actually solved (a frozen factorization
        mis-preconditions the shifted operator once the march's β leaves the value it was built at). Like
        the trigger and the control it is **forward-only** -- an impure mutation that must never be on a
        differentiated path. ``None`` (the default) leaves the preconditioner untouched, byte-identical to
        before.
    solver : lineax.AbstractLinearSolver, optional
        The linear solver for each step; defaults to ``forward_step.default_solver()``.
    retry_solver : lineax.AbstractLinearSolver, optional
        A **tighter** linear solver used as the **fallback** for a step that is *still* diverged after the
        ``β``-escalation below (or when escalation is unavailable -- no ``retry_on_cycles`` / no ``β`` leaf,
        the pure-ILUT configuration). An *inexact* preconditioner -- a threshold-incomplete-LU (ILUT) --
        can return a correction that goes non-finite where the loose default Krylov tolerance left it too
        inaccurate; that failure is cured by a tighter Krylov solve, not by more damping, so it is the
        fallback when escalation (the cheaper cure, tried first) does not recover the step. It redoes the
        **same** step from the **same** pre-step state with ``retry_solver`` -- the preconditioner is
        unchanged and *not* re-refreshed (already fresh at this ``(state, β)``); only the Krylov solve is
        tightened. A single retry: if it still diverges, the march breaks as it would have without a retry.
        Forward-only. ``None`` (default) never retries, so the loop is byte-identical -- and the exact-LU
        path, which never diverges, needs none.
    retry_divergence_cap : float, optional
        With a ``retry_solver`` set, a step is treated as diverged (and retried) when its residual norm is
        non-finite **or**, if this cap is finite, exceeds ``retry_divergence_cap * reference``. Defaults to
        ``inf`` (only non-finiteness triggers a retry), because the load-bearing failure is a poisoned
        (non-finite) correction, and a residual can legitimately *rise* during development (the
        ``β × travel`` identity), so a tight cap would false-fire on the reachability descent.
    retry_on_cycles : int or None
        A ``β``-escalation bailout, tried **before** ``retry_solver`` as the cheaper cure for a bad step.
        When a step's linear-solve count exceeds this **or** the step diverged (non-finite / over
        ``retry_divergence_cap``), it is redone from the pre-step state with ``β`` **escalated**
        (``β *= retry_beta_factor``, re-matching the preconditioner via ``precondition_step``). On the stiff
        low-``β`` saddle both a cost spike and a non-finite correction have the same fix -- more damping --
        and escalating is far cheaper than the tight-Krylov divergence retry, so it leads; the divergence
        retry becomes the fallback only for a non-finite step escalation cannot fix (the inexact-ILUT case).
        ``None`` (default) disables escalation (byte-identical) and a diverged step falls straight to
        ``retry_solver``. Escalation needs a ``β`` leaf (a step control's ``ConstantRelaxation``); it no-ops
        otherwise.
    retry_beta_factor : float
        The factor ``β`` is multiplied by on each cycle-count retry (default ``2``).
    retry_cycles_limit : int
        The maximum number of successive ``β`` escalations for one step (default ``2``). After them the step
        is accepted whatever its count.

    Returns
    -------
    MarchResult
        The state reached, the per-step reports, and whether the march converged or was triggered.
    """
    if solver is None:
        solver = forward_step.default_solver()
    # When the measure is rebuilt each iteration, the segment reference must be taken in that same
    # measure -- otherwise the damping schedule divides two differently-scaled quantities.
    norm = norm_builder(phi0) if norm_builder is not None else forward_step.norm()

    # The segment-local reference: what the step's damping schedule ramps against. Recomputed here,
    # never inherited, so a segment resumed after a refresh restarts its ramp. It is fixed for the
    # whole segment: recomputing it per step would hold the ratio at one, pinning the shift at its
    # starting strength and freezing the march.
    residual_norm_0 = jnp.asarray(norm(residual_fn(phi0)))
    reference = float(residual_norm_0) if reference_norm is None else float(reference_norm)

    state = phi0
    current = float(residual_norm_0)
    reports: list[StepReport] = []
    triggered = False
    # `control_state` is a parameter (the initial state), threaded and returned so a multi-segment
    # driver can continue a stateful control across a refresh instead of restarting it.

    def converged_at(residual_norm: float) -> bool:
        return bool(_within_tolerance(jnp.asarray(residual_norm), reference, rtol, atol))

    while len(reports) < max_steps and not converged_at(current) and not triggered:
        # A step control reshapes the base step from the previous report (None runs it unchanged, so
        # the loop is byte-identical). It threads its own state; the march stays ignorant of β.
        active_step = forward_step
        if norm_builder is not None:
            # Re-derive the residual measure at the state this outer iteration starts from, and hold
            # it for the whole iteration -- every trial step of the line search, the acceptance test
            # and the reported norm all use this one. Rebuilding it per trial step instead would let a
            # candidate win by shrinking its own denominator rather than its residual, so the search
            # would stop comparing like with like. The swap is a compilation-cache hit as long as the
            # measure carries its scales as data over a fixed block structure.
            active_step = eqx.tree_at(lambda s: s.residual_norm, active_step, norm_builder(state))
        if step_control is not None:
            active_step, control_state = step_control.next_step(
                active_step, reports[-1] if reports else None, control_state
            )
        if precondition_step is not None:
            # Refresh the step's (frozen, host) preconditioner from the state and shift strength this
            # step is about to run at -- e.g. re-factoring a complete LU at the current (state, β) so it
            # is the exact inverse of the operator actually solved. Runs HERE, in the eager loop (a host
            # op outside the jitted `_march_step`), after the control has set β on `active_step`. It
            # mutates the step's static preconditioner in place, so `_march_step` stays a compilation
            # cache hit. Forward-only, like the trigger and the control.
            precondition_step(active_step, state)
        prestep_state = state
        state, cycles, alpha, inner, residual_norm = _march_step(
            active_step, residual_fn, prestep_state, residual_norm_0, solver
        )
        # A step can go bad two ways -- a non-finite / diverging correction, or a finite solve whose cost
        # spikes past `retry_on_cycles` -- and on the stiff low-β saddle BOTH have the same cheap cure:
        # MORE damping. Escalate β FIRST (redo from the pre-step state at `β *= retry_beta_factor`,
        # re-matching the frozen preconditioner via `precondition_step`), because a larger β both lifts the
        # correction out of the non-finite regime AND cuts the cycle count, and it is far cheaper than the
        # tight-Krylov divergence retry below -- on the coupled AMG march a NaN'd low-β step recovers in a
        # handful of cycles at 2β, where the tight retry would grind hundreds of matvecs to recover the same
        # step the escalation then re-damps anyway. β vanishes at the root, so the escalation reshapes only
        # the forward path; it is not carried into the control, so a persistently hard region re-triggers
        # each step. Needs a readable β leaf (a `ConstantRelaxation` set by a step control) and
        # `retry_on_cycles is not None`; otherwise it no-ops and a diverged step falls straight through to
        # the divergence retry. `retry_on_cycles=None` (default) is byte-identical.
        retries = 0
        while (
            retry_on_cycles is not None
            and retries < retry_cycles_limit
            and not converged_at(float(residual_norm))
            and (
                int(cycles) > retry_on_cycles
                or _has_diverged(residual_norm, reference, retry_divergence_cap)
            )
            and hasattr(active_step.relaxation_schedule, "beta")
        ):
            retries += 1
            escalated = float(active_step.relaxation_schedule.beta) * retry_beta_factor
            active_step = eqx.tree_at(
                lambda s: s.relaxation_schedule.beta, active_step, jnp.asarray(escalated)
            )
            if precondition_step is not None:
                precondition_step(active_step, prestep_state)  # re-match the PC to the escalated β
            state, cycles, alpha, inner, residual_norm = _march_step(
                active_step, residual_fn, prestep_state, residual_norm_0, solver
            )
        # Divergence retry -- the FALLBACK for a non-finite correction β-escalation could not fix. An
        # inexact preconditioner (a threshold-ILU) can return a non-finite correction where the loose
        # default Krylov tolerance left it too inaccurate; the cure there is a tighter Krylov solve, not
        # more damping (the factors are already fresh at this (state, β), so re-preconditioning is a no-op).
        # This fires only if the step is STILL diverged after the escalation loop -- or if escalation was
        # unavailable (`retry_on_cycles is None` or no β leaf, the pure-ILUT configuration, where it is the
        # sole and original retry) -- redoing the SAME step from the SAME pre-step state with `retry_solver`.
        # One retry; a still-diverged step breaks below as it would without a retry. `retry_solver=None`
        # (default) is byte-identical, and the exact-LU path never triggers this.
        if retry_solver is not None and _has_diverged(
            residual_norm, reference, retry_divergence_cap
        ):
            state, cycles, alpha, inner, residual_norm = _march_step(
                active_step, residual_fn, prestep_state, residual_norm_0, retry_solver
            )
        # Carry an escalated β forward into the control. The escalation raised β because the control had
        # driven it too low for this operator; without carrying that back, the next `next_step` recomputes
        # from the control's own (floor-ward) β and re-pays the escalation every step on a persistently hard
        # region -- the low-β reachability tail, where β sits at its floor and each step re-escalates. The
        # escalation IS the feedback for "how low is safe here", so it replaces a static β floor: seeding the
        # control's carried β with the escalated value lets the control continue from the discovered-safe
        # level (and adapt on from there), so β_min can be driven toward zero and the *controller* decides how
        # large a pseudo-timestep is safe. Only when β was actually escalated, and only for a control that
        # carries β (the dual-time family exposes `carry_beta`); no escalation ⇒ byte-identical.
        if (
            retries
            and step_control is not None
            and hasattr(step_control, "carry_beta")
            and hasattr(active_step.relaxation_schedule, "beta")
        ):
            control_state = step_control.carry_beta(
                control_state, float(active_step.relaxation_schedule.beta)
            )
        current = float(residual_norm)
        report = StepReport(
            step=len(reports),
            cycles=int(cycles),
            residual_norm=current,
            residual_ratio=current / reference if reference > 0.0 else 0.0,
            alpha=float(alpha),
            drift=0.0 if drift_measure is None else float(drift_measure(state)),
            inner_iterations=int(inner),
        )
        reports.append(report)
        if observer is not None:
            observer(report)
        if checkpoint is not None:
            checkpoint(report, state)
        # A non-finite residual can never satisfy the tolerance test, so without this the march
        # would spend its whole budget stepping a poisoned state. Stop and let the finishing solve
        # report the failure, which is where non-convergence is diagnosed.
        if not jnp.isfinite(residual_norm):
            break
        if trigger is not None and not converged_at(current):
            triggered = trigger.should_refresh(reports)

    return MarchResult(
        state=state,
        reports=tuple(reports),
        converged=converged_at(current),
        triggered=triggered,
        control_state=control_state,
    )
