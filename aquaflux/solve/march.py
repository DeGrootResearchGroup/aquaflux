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

**The eager march's state is an answer only when it reports ``converged``.** Short of that it is a
pure accelerator: a driver uses it to reach a better-preconditioned state, and then finishes with a
real ``ImplicitNewtonSolver.solve()``, which owns the convergence guard, the ``custom_vjp``, and the
returned field. That is why :func:`forward_march` deliberately has **no** non-convergence guard of
its own — stopping short is its purpose, and a state it hands back carries no guarantee beyond what
:attr:`MarchResult.converged` states. Keeping the guard in one place means a march that ends short of
a root can never be mistaken for a converged one.

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
from typing import Any, NamedTuple, Protocol

import equinox as eqx
import jax.numpy as jnp
import lineax as lx

from .forward_step import ForwardStep, StepControl, StepOutcome, StepReport, within_tolerance
from .norm import ResidualNorm
from .retry import NO_RETRIES, RetryPolicy


def combine_observers(*callbacks: Callable[..., None]) -> Callable[..., None]:
    """Fan one march callback out to several observers, called in order with the same arguments.

    ``forward_march`` takes a single ``on_step`` / ``on_checkpoint``, but a run usually wants more than
    one thing to happen per step -- log it *and* checkpoint it. Composing them here keeps a driver from
    writing its own lambda, which is where one observer silently gets dropped from a later edit.

    An observer that raises propagates: a checkpoint that cannot be written is a real failure, and
    swallowing it would leave a run believing it was protected when it was not.

    Parameters
    ----------
    *callbacks : callable
        Each accepting whatever the seam supplies (``(report)`` for ``on_step``, ``(report, state)``
        for ``on_checkpoint``).

    Returns
    -------
    callable
        One callback invoking each in turn.

    Examples
    --------
    >>> seen = []
    >>> both = combine_observers(lambda r: seen.append(("a", r)), lambda r: seen.append(("b", r)))
    >>> both(1); seen
    [('a', 1), ('b', 1)]
    """

    def combined(*args: Any) -> None:
        for callback in callbacks:
            callback(*args)

    return combined


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


def _shift_of(forward_step: ForwardStep) -> float | None:
    """The step's current shift strength, for **reporting**, or ``None`` if it has no shift.

    Reporting must never demand a shift: a plain damped-Newton step legitimately has none, and a march
    of one is a perfectly ordinary thing to run and to log. This is the read that belongs on every
    reporting path -- ``StepReport.shift``, and the retry announcement, which reached for
    ``forward_step.relaxation_schedule`` unguarded and raised ``AttributeError`` on exactly such a step.

    It stays on the march rather than moving to :class:`~aquaflux.solve.RetryPolicy` with the retry
    decisions: reporting a step's shift is not a retry concern, and the step summary reads it on every
    step whether or not any retry is configured.
    """
    return getattr(getattr(forward_step, "relaxation_schedule", None), "beta", None)


def _limit_collapsing(
    previous: StepReport | None, report: StepReport, progress: float = 1e-3
) -> bool:
    """Whether this step continues a constraint-bound sequence that is buying nothing.

    True when the step's length was decided by an injected constraint (``binding_limit < 1``), the cap
    is no wider than the step before it, and the residual **changed** by less than the fraction
    ``progress`` -- in either direction. One such step is ordinary -- a capped step is a legitimate
    short step, and a pseudo-transient path is allowed to be non-monotone -- so this is a per-step
    predicate that a caller counts, never a verdict on its own.

    **The failure it names.** A fraction-to-the-boundary rule takes ``tau`` of the distance to the
    constraint, so a step that runs into it leaves the binding entry at ``1 - tau`` of its value: at
    ``tau = 0.99`` the next step's room is a hundredth of this one's, whether or not anything else
    changed. Once the direction keeps pointing at the boundary in that entry, the cap collapses
    geometrically and every step is arithmetically a no-op. Measured on a coupled RANS march: the cap
    fell by exactly 100x per step from 1.95e-06 to 1.95e-196 over ninety-six consecutive steps, with the
    residual frozen to every reported digit and the linear solve costing nothing, because there was no
    step left to take. Nothing in the ordinary stopping tests sees that -- the state is finite, the
    residual is finite, and the tolerance is simply never reached -- so the march runs its whole budget.

    **Both halves of the residual test were got wrong once each, and the archived march logs are what
    caught them.** Replaying a candidate predicate over every logged march -- the locked-up ones and the
    healthy ones together -- is the only cheap way to see a false positive, since a march that recovers
    looks exactly like one that does not until it does.

    * *Not* "did not fall", strictly. A locked-up step is not a bit-exact no-op: it still moves the
      state by ``alpha`` times the correction, so at a cap of ~1e-6 the residual genuinely falls, by
      ~1e-6 relative. A strict test resets on every step and never fires, which is what it did on a
      march that had otherwise reproduced the lock-up step for step.
    * *Not* "did not fall by ``progress``" either. That still admits a residual that **rose**, and a
      rising residual is the signature of a healthy pseudo-transient excursion rather than a stall. On
      a march that converges, three consecutive steps ran caps of 0.983, 0.928, 0.253 at residuals
      1.310e-01, 1.335e-01, 1.489e-01 -- climbing from the 1.293e-01 the run of three began at -- and
      then recovered to 9.241e-02 and converged: a one-sided test ends that rung at its worst moment.

    The failure is a step that changes **nothing**, so the test is two-sided. The two populations sit
    orders apart: null steps move the residual by ~1e-6 relative, productive ones by percents.

    All three conditions are needed. The residual test alone fires on an ordinary non-monotone pair of
    steps, and the cap test alone fires wherever a step is legitimately short; the collapse is the
    conjunction, and requiring the cap to be *narrowing* is what separates it from a march that is
    working along a constraint and making progress.

    Parameters
    ----------
    previous, report : StepReport or None
        The step before this one (``None`` before the first, which can continue nothing), and this one.
    progress : float
        The relative residual change a step must exceed, in either direction, to count as having done
        something. Default ``1e-3``.
    """
    return (
        previous is not None
        and report.binding_limit < 1.0
        and report.binding_limit <= previous.binding_limit
        and abs(report.residual_norm - previous.residual_norm)
        <= progress * abs(previous.residual_norm)
    )


@eqx.filter_jit
def _march_step(
    forward_step: ForwardStep,
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    phi: jnp.ndarray,
    residual_norm_0: jnp.ndarray,
    solver: lx.AbstractLinearSolver,
) -> tuple[StepOutcome, jnp.ndarray]:
    """One observed step: the step's :class:`StepOutcome`, and the residual norm at the state it produced.

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
    outcome = forward_step.stepper()(residual_fn, phi, residual_norm_0, solver)
    return outcome, forward_step.norm()(residual_fn(outcome.phi))


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
    retry: RetryPolicy = NO_RETRIES,
    stop_on_limit_stall: int | None = 3,
    on_retry: Callable[[str, int, float], None] | None = None,
) -> MarchResult:
    """March the residual eagerly, reporting each step and stopping early if the trigger fires.

    A forward-only counterpart to :class:`~aquaflux.solve.ImplicitNewtonSolver`'s traced march,
    for a driver that must observe per-step cost or interpose work that cannot run under ``jit``
    (rebuilding a frozen preconditioner). It applies the same injected ``forward_step``, the same
    residual measure (``forward_step.norm()``), and the same stopping test, so the two marches take
    the same path on the same problem.

    **This function may return a state that does not solve the residual, without raising** — that is
    the point of a march that can stop early. It carries no convergence guard, so a caller must read
    :attr:`MarchResult.converged` before treating the state as a result; a march that stopped short
    must be finished with an ``ImplicitNewtonSolver.solve()``, which does carry the guard, and which
    produces the result and its adjoint. Do not differentiate through this march.

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
    norm_builder : callable, optional
        ``state -> ResidualNorm``, re-deriving the residual measure at the state each outer iteration
        starts from and holding it for that whole iteration — every trial step of the line search, the
        acceptance test and the reported norm. Rebuilding it per trial step instead would let a
        candidate win by shrinking its own denominator rather than its residual. The segment reference
        the damping schedule ramps against is taken in this same measure, so the ratio divides two
        comparably-scaled quantities. ``None`` (the default) uses ``forward_step.norm()`` throughout.
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
    retry : RetryPolicy
        When and how to redo a step: the three escalation triggers (cost, step length, divergence),
        the shift factor and escalation limit, and the optional tighter linear solver. See
        :class:`~aquaflux.solve.RetryPolicy` for what each setting means and why the defaults are what
        they are. The default policy retries nothing, so the loop is byte-identical to a march without
        retries.

        **The order is the load-bearing part, and it lives here rather than on the policy.** Shift
        escalation is tried **first**, because on the stiff low-``β`` saddle a cost spike, a collapsed
        step length and a non-finite correction all have the same cheap cure -- more damping -- and a
        doubling is far cheaper than a tight Krylov solve. ``retry.solver`` is the **fallback**, for a
        step that is *still* diverged afterwards, or when escalation is unavailable (no threshold set,
        or no ``β`` leaf to escalate -- the configuration where the tighter solve is the sole and
        original retry). Each escalation re-matches the preconditioner through ``precondition_step``;
        the divergence retry does not, because the factorization is already fresh at this
        ``(state, β)`` and only the Krylov tolerance is at fault.
    stop_on_limit_stall : int or None
        End the segment after this many **consecutive** steps that are constraint-bound, narrowing, and
        not reducing the residual (see :func:`_limit_collapsing`), default ``3``. That pattern is a
        fraction-to-the-boundary lock-up, and it does not recover on its own: the cap shrinks by a fixed
        factor per step for as long as the march is allowed to run, so without this the segment spends
        its entire ``max_steps`` budget taking arithmetically null steps. Ending it hands the caller a
        state that is honestly unconverged instead, which the finishing solve reports. ``None`` disables
        the test. The count is deliberately not ``1``: an isolated capped step is an ordinary short step,
        and a pseudo-transient path is allowed to be non-monotone.
    on_retry : callable, optional
        ``(reason, attempt, beta) -> None``, called immediately before a step is redone. ``reason`` is
        ``"cycles"`` (the cost trigger, and the step was cut short), ``"alpha"`` (the step-length
        trigger, likewise cut short), ``"diverged"`` (the β-escalation firing on a non-finite or runaway
        residual) or ``"solver"`` (the tighter-solver fallback).
        ``beta`` is **the shift the retried attempt will run at**, not the one the abandoned attempt
        used: on the three escalation reasons that is the already-escalated value, and on ``"solver"``
        — which retries at a tighter Krylov tolerance and does not touch the shift — it is the
        unchanged one. Reporting the pre-escalation value instead would leave a consumer to re-derive
        the real shift from ``retry.beta_factor``, which is exactly what a log must not have to guess.
        Without it a log shows a step's work twice with nothing saying why, leaving a reader to infer
        the trigger from the numbers. ``None`` (default) elides the call.

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

    # Both thresholds are knowable INSIDE a step -- the cost one the moment a solve returns, the
    # step-length one the moment a line search collapses -- but the reaction below only runs once the
    # whole step is back. Push them down to the step so it can stop there and then, rather than
    # finishing inner iterations whose results this loop is about to discard. One number each, set in
    # one place: a step that took its own copy would be a second spelling to keep in step with this one.
    forward_step = retry.with_inner_abort(forward_step)

    state = phi0
    current = float(residual_norm_0)
    reports: list[StepReport] = []
    triggered = False
    stalled = 0
    # `control_state` is a parameter (the initial state), threaded and returned so a multi-segment
    # driver can continue a stateful control across a refresh instead of restarting it.

    def converged_at(residual_norm: float) -> bool:
        return bool(within_tolerance(jnp.asarray(residual_norm), reference, rtol, atol))

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
        # Check the step can support what was ASKED FOR -- loudly, and once. The escalation drives the
        # pseudo-transient shift, which `ForwardStep` does not promise (see `RetryPolicy.require_shifted`);
        # the `hasattr` this replaces sat in the escalation's own loop condition and failed **silently**,
        # so a march configured to escalate simply never did.
        #
        # It must run HERE, on `active_step`, not on the base step before the loop. A `StepControl` is
        # what INSTALLS the readable shift on the dual-time family: the builder hands over a step whose
        # schedule is a `SwitchedEvolutionRelaxation` (which has no `beta`), and the control swaps in a
        # `ConstantRelaxation` (which does) at this point of every iteration. Checking before the loop
        # therefore rejects the shipped coupled configuration -- a step that would have escalated
        # perfectly well -- while checking here sees what the escalation will actually be handed.
        # Two neighbours are deliberately NOT gated, for opposite reasons: the tight-solver fallback
        # re-solves and never touches beta, so it works on any step; and a control that drives beta
        # already fails loudly from inside itself when its `tree_at` cannot find the field.
        if retry.escalates and not reports:
            retry.require_shifted(active_step)
        if precondition_step is not None:
            # Refresh the step's (frozen, host) preconditioner from the state and shift strength this
            # step is about to run at -- e.g. re-factoring a complete LU at the current (state, β) so it
            # is the exact inverse of the operator actually solved. Runs HERE, in the eager loop (a host
            # op outside the jitted `_march_step`), after the control has set β on `active_step`. It
            # mutates the step's static preconditioner in place, so `_march_step` stays a compilation
            # cache hit. Forward-only, like the trigger and the control.
            precondition_step(active_step, state)
        prestep_state = state
        outcome, residual_norm = _march_step(
            active_step, residual_fn, prestep_state, residual_norm_0, solver
        )
        # A step can go bad three ways -- a non-finite / diverging correction, a finite solve whose cost
        # spikes past `retry.on_cycles`, or a step length collapsed to `retry.on_alpha` -- and on the
        # stiff low-β saddle ALL THREE have the same cheap cure: MORE damping. The policy owns which of
        # them applies. Escalate β FIRST (redo from the pre-step state at `β *= retry.beta_factor`,
        # re-matching the frozen preconditioner via `precondition_step`), because a larger β lifts the
        # correction out of the non-finite regime, cuts the cycle count AND shortens the implicit step until
        # it fits inside whatever was clipping it, and it is far cheaper than the tight-Krylov divergence
        # retry below -- on the coupled AMG march a NaN'd low-β step recovers in a handful of cycles at 2β,
        # where the tight retry would grind hundreds of matvecs to recover the same step the escalation then
        # re-damps anyway. β vanishes at the root, so the escalation reshapes only the forward path; the
        # discovered-safe β IS then carried into the control (see the carry below), so a persistently hard
        # region continues from it rather than re-paying the escalation every step. Needs a
        # readable β leaf (a `ConstantRelaxation` set by a step control) and at least one threshold set;
        # otherwise it no-ops and a diverged step falls straight through to the divergence retry. Both
        # thresholds `None` (the default) is byte-identical.
        retries = 0
        while (
            (reason := retry.escalation_reason(outcome, residual_norm, reference)) is not None
            and retries < retry.cycles_limit
            and not converged_at(float(residual_norm))
        ):
            retries += 1
            # `RetryPolicy.escalate` SCALES the existing β leaf rather than rebuilding one; its
            # docstring carries why (a rebuilt leaf's dtype/weak type need not match, and any mismatch
            # recompiles the whole coupled solve on every retry).
            escalated = retry.escalate(active_step.relaxation_schedule.beta)
            # Report the escalated β, and so only once it exists: `on_retry` promises the shift the
            # retried attempt will RUN at. Reporting the pre-escalation leaf here instead left the one
            # consumer reconstructing it as `beta * 2` -- correct only at the default `beta_factor`,
            # and wrong at every other, in the log a long march is read back from.
            if on_retry is not None:
                on_retry(reason, retries, float(escalated))
            active_step = eqx.tree_at(lambda s: s.relaxation_schedule.beta, active_step, escalated)
            if precondition_step is not None:
                # Re-match the preconditioner to the escalated β. Whether this actually rebuilds is the
                # HOOK'S decision, not this call's: a gated refresh may judge the move too small to be
                # worth its cost and reuse the standing factorization. That is worth stating, because a
                # hook gated so tightly that it never fires makes every escalated attempt solve against
                # a factorization built for the PRE-escalation β, which is not what this call site
                # intends. Observed on a coupled RANS march whose β-mismatch gate was effectively off:
                # a step's three escalated attempts all ran with no rebuild and all returned a step
                # length of 0.000, and the following step took a full step once a rebuild happened to
                # be triggered by its solve cost. That is a correlation across three runs, NOT an
                # established cause -- the same steps were also positivity-cap-bound, and the two
                # explanations are not separated by that data. Whichever it is, a refresh hook used
                # with escalation should let a doubling through: re-matching is what this call asks for.
                precondition_step(active_step, prestep_state)
            outcome, residual_norm = _march_step(
                active_step, residual_fn, prestep_state, residual_norm_0, solver
            )
        # Divergence retry -- the FALLBACK for a non-finite correction β-escalation could not fix. An
        # inexact preconditioner (a threshold-ILU) can return a non-finite correction where the loose
        # default Krylov tolerance left it too inaccurate; the cure there is a tighter Krylov solve, not
        # more damping (the factors are already fresh at this (state, β), so re-preconditioning is a no-op).
        # This fires only if the step is STILL diverged after the escalation loop -- or if escalation was
        # unavailable (no threshold set, or no β leaf: the pure-ILUT configuration, where it is the sole
        # and original retry) -- redoing the SAME step from the SAME pre-step state with `retry.solver`.
        # One retry; a still-diverged step breaks below as it would without a retry. A policy with no
        # `solver` (the default) is byte-identical, and the exact-LU path never triggers this.
        diverged_retry = retry.solver is not None and retry.has_diverged(residual_norm, reference)
        if diverged_retry:
            if on_retry is not None:
                # `_shift_of`, not a direct read: the divergence retry needs no shift, so it runs on
                # steps that have none, and this reported the shift by reaching straight through
                # `relaxation_schedule` -- raising `AttributeError` on a step that satisfies
                # `ForwardStep` in full. Unchanged beta here in any case; nothing escalated.
                on_retry("solver", retries + 1, float(_shift_of(active_step) or 0.0))
            outcome, residual_norm = _march_step(
                active_step, residual_fn, prestep_state, residual_norm_0, retry.solver
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
        if retries and step_control is not None and hasattr(step_control, "carry_beta"):
            control_state = step_control.carry_beta(
                control_state, float(active_step.relaxation_schedule.beta)
            )
        state = outcome.phi
        current = float(residual_norm)
        # Not every ForwardStep carries a relaxation schedule (a plain damped-Newton step has none), and
        # a schedule need not expose a readable beta -- report 0 rather than demanding either.
        step_shift = _shift_of(active_step)
        report = StepReport(
            step=len(reports),
            cycles=int(outcome.cycles),
            residual_norm=current,
            residual_ratio=current / reference if reference > 0.0 else 0.0,
            alpha=float(outcome.alpha),
            drift=0.0 if drift_measure is None else float(drift_measure(state)),
            inner_iterations=int(outcome.inner_iterations),
            max_inner_cycles=int(outcome.max_inner_cycles),
            binding_limit=float(outcome.binding_limit),
            shift=0.0 if step_shift is None else float(step_shift),
            escalations=int(retries),
            diverged_retry=bool(diverged_retry),
        )
        stalled = stalled + 1 if _limit_collapsing(reports[-1] if reports else None, report) else 0
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
        # The same argument for a step that is finite but null: a collapsing constraint cap can never
        # satisfy the tolerance test either, and unlike a stiff region it does not recover, so the rest
        # of the budget is spent on steps that move nothing. Stop on the same terms.
        if stop_on_limit_stall is not None and stalled >= stop_on_limit_stall:
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
