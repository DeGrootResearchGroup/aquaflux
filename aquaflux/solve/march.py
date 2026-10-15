"""The observed, forward-only Newton march, and the staleness trigger that watches it.

:func:`newton_march` is the loop every Newton solve in this package runs on. It steps an injected
:class:`~aquaflux.solve.NewtonStrategy` until the residual meets a tolerance, and between steps — in
plain Python, outside the compiled step — it reports what the step cost, lets a control reshape the
next one, lets a policy redo a bad one, and can stop early so a driver can do work no traced program
could.

Refreshing a frozen algebraic-multigrid (AMG) preconditioner mid-march is the motivating case and
needs all of that: the rebuild assembles ``scipy`` sparse matrices, which cannot happen inside a
traced loop, and the *decision* to rebuild is made from the per-step linear-solve cost.

Stepping in Python is also the one thing this march cannot do inside a traced program -- ``jax.jit``,
``jax.vmap``, or a traced loop such as ``jax.lax.scan`` -- where the values it reads back are still
abstract. Every driver therefore refuses those transforms up
front with :func:`refuse_a_transform_the_march_cannot_run_in`, rather than failing part way with a
message about a tracer. ``jax.grad`` is unaffected and is the mode that matters here: the march runs
on ``stop_gradient`` copies, so it sees concrete values, and the derivative is attached to the
converged state afterwards by :func:`~aquaflux.solve.root_adjoint` — one transpose solve at the root,
independent of how the march reached it, so nothing about the iteration is taped.

**The march's state is an answer only when it reports ``converged``.** :func:`newton_march`
deliberately has **no** non-convergence guard of its own — stopping short, on a trigger or out of
steps, is part of its purpose — so the driver that calls it owns that test, and refuses a state that is
not a root before attaching the adjoint to it. A state the march hands back carries no guarantee
beyond what :attr:`MarchResult.converged` states.

**Two reference residual norms, and conflating them breaks the march.** Each call to
:func:`newton_march` computes its own ``residual_norm_0`` from the state it is handed, and passes
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

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from .adaptive_tolerance import AdaptiveLinearTolerance
from .convergence import MeasureBuilder
from .linear import relative_tolerance, with_relative_tolerance
from .norm import named_blocks, norm_of_residual
from .retry import ESCALATING_REASONS, NO_RETRIES, RetryPolicy
from .root_adjoint import stop_array_gradients
from .strategy import (
    NewtonStrategy,
    ShiftCarryingControl,
    StepControl,
    StepOutcome,
    StepReport,
    StopsForRefresh,
    shift_of,
    within_tolerance,
)


def refuse_a_transform_the_march_cannot_run_in(pytree: object, *, caller: str) -> None:
    """Raise if ``pytree`` holds abstract values, i.e. the march is inside a traced program.

    The march steps in Python on concrete values: it reads residual norms back as floats, runs host
    work between steps, and decides how many steps to take from what it sees. Under ``jax.grad`` a
    ``stop_gradient`` copy of an input is concrete, so the march runs and the derivative is attached
    at the root afterwards; inside any **traced** program it is still abstract, and the march would
    fail part way with an error about converting a tracer to a Python value -- far from the call that
    caused it. Refuse up front instead, naming the entry point and saying what to do.

    ⚠️ **"Traced" is wider than ``jit`` and ``vmap``.** ``jax.lax.scan`` and ``jax.lax.fori_loop``
    trace their bodies too, so a transient march that steps a nonlinear solve inside one is refused
    here as well; write that time loop in Python and give each step's residual a stable identity (a
    small ``equinox.Module`` holding the previous states) so the later steps run on the compiled Newton
    step the opening ones build.

    Parameters
    ----------
    pytree : pytree
        The solve's inputs, whose leaves carry the abstractness of any transform in force.
    caller : str
        The entry point to name in the message, e.g. ``"solve_coupled"``.

    Raises
    ------
    ValueError
        If any leaf is a tracer.
    """
    stopped = stop_array_gradients(pytree)
    if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(stopped)):
        raise ValueError(
            f"{caller} steps its march in Python on concrete values, so it cannot run inside a traced "
            "program: not under jax.jit or jax.vmap, and not inside jax.lax.scan or another traced "
            "loop. Call it outside those; jax.grad through it is supported."
        )


def combine_observers(*callbacks: Callable[..., None]) -> Callable[..., None]:
    """Fan one march callback out to several observers, called in order with the same arguments.

    ``newton_march`` takes a single ``on_step`` / ``on_checkpoint``, but a run usually wants more than
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
    """The outcome of one :func:`newton_march` segment.

    Attributes
    ----------
    state : jnp.ndarray
        The state the march reached. An **intermediate** unless :attr:`converged` is ``True``; it
        carries no guarantee of solving the residual, so it must be finished by a real solve.
    reports : tuple of StepReport
        One report per step taken, in order.
    converged : bool
        Whether the march reached the requested tolerance against the global reference norm, with a
        finite residual norm. A non-finite one is never reported as converged, however the tolerance
        test compares (see :func:`newton_march`).
    triggered : bool
        Whether the march stopped early because the injected trigger fired.
    control_state : object
        The injected :class:`StepControl`'s carried state as of the last step (``None`` if no control
        was used). Required, like :attr:`strategy`: a producer that leaves either out fails here rather
        than handing a driver ``None`` to march or differentiate with. Returned so a driver running one segment per preconditioner refresh can **thread** it
        into the next segment's :func:`newton_march`, rather than each segment restarting the control
        from scratch — a stateful control (e.g. one climbing the shift strength over many steps) would
        otherwise throw its progress away at every refresh.
    strategy : NewtonStrategy
        The base step as of the last step: the one passed in, with every update a
        ``refresh_preconditioner`` returned applied to it. A driver that marches again or hands the step
        to the adjoint takes this one, which carries the re-fitted preconditioner.
    """

    state: jnp.ndarray
    reports: tuple[StepReport, ...]
    converged: bool
    triggered: bool
    control_state: object
    strategy: NewtonStrategy


class ResidualHomotopy(Protocol):
    """Which problem each outer step of a march solves, on a path ending at the target problem.

    Structural interface only (a ``Protocol``). A march normally drives **one** residual to zero. A
    homotopy replaces that with a short sequence of *stations* -- residuals of the same shape,
    differing in a physical parameter -- walked within a **single** march, the last of which is the
    target problem itself.

    **What this buys over solving each station as its own march.** Solving a sequence of related
    problems one full solve at a time (a Reynolds-number ladder, say) pays two costs per station that
    a within-march walk does not. Each solve converges its station to a stopping tolerance even though
    the station is only a *seed* for the next one -- and the parameter jump that follows immediately
    undoes that convergence, so the polishing is discarded work. And each solve restarts its step
    control, so a pseudo-timestep ramp walks back up from its starting shift at every station
    regardless of how good the seed it was handed is. Marching the stations in one loop keeps the
    state, the shift and the preconditioner, and converges only the last one.

    **The stopping test is gated on arrival.** A march running a homotopy may not stop on its residual
    tolerance until :meth:`arrived` is true, because a small residual at an intermediate station says
    nothing about the target problem. :func:`newton_march` enforces this; an implementation only has
    to answer honestly.

    **Stations are coarser than steps, deliberately.** :meth:`enter` is called once per outer step, but
    an implementation is expected to hold a station for several steps: whatever a station change costs
    downstream -- refitting a frozen preconditioner to the new parameter is the motivating case -- is
    paid once per *change*, not once per step, so moving the parameter every step can cost far more
    than the steps it saves.
    """

    def enter(self, step: int, previous: StepReport | None) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Make the station outer step ``step`` runs current, and return its residual ``phi -> R(phi)``.

        Called once per outer step, before the step is taken, with the report of the step before it
        (``None`` for the first) -- so a homotopy may decide its next station from how the march is
        going, rather than from the step index alone (ending a ramp once the march has settled, say).
        Steps are entered in order, and the first may be entered more than once (the march measures its
        damping anchor at the first station before taking the step); an implementation answers a
        repeated step as it did the first time. **May have side effects** -- that is what "enter" names:
        an implementation that must re-point a preconditioner refresh at the new station does it here,
        so the operator the step solves against and the residual it drives agree.

        Return a **bound method of a module** rather than a freshly-built closure, for the same reason
        :func:`newton_march`'s own ``residual_fn`` must be one: its arrays then ride as dynamic leaves
        and each step stays a compilation-cache hit instead of recompiling per station.
        """

    def arrived(self, step: int) -> bool:
        """Whether outer step ``step`` runs the **target** problem, so the march may stop on tolerance.

        Asked of a step after it has been entered, as are :meth:`station` and :meth:`shift_factor`.
        """

    def shift_factor(self, step: int) -> float:
        """What to multiply the pseudo-transient shift by when outer step ``step`` **enters** a station.

        ``1.0`` within a station (the ordinary case, and the value a homotopy that wants no shift
        response returns always). Greater than one at a change, asking the march to re-damp because the
        problem just got harder -- the same response a continuation makes at a rung boundary, which a
        within-march homotopy has nowhere else to hang.

        Read **only** on a step where :meth:`station` differs from the previous step's, so a value
        returned elsewhere is ignored.
        """
        return 1.0

    def station(self, step: int) -> int:
        """Which station outer step ``step`` runs, as an index that only ever increases.

        The march needs this to spot a **change**, because a change is a discontinuity in the residual
        and some march machinery is only valid within one station. Concretely, a step control that
        keys on the residual *ratio* forms that ratio across consecutive steps, and the ratio is
        meaningful only when both residuals came from the same station -- a station change raises the
        residual because the problem got harder, which such a control would otherwise read as
        divergence and brake on. :func:`newton_march` therefore rebases the control at a change (see
        :meth:`~aquaflux.solve.ShiftStrengthControl.rebase`).

        Only equality between adjacent steps is read, never the value or the spacing, so an
        implementation may number stations however it likes as long as consecutive steps within one
        station compare equal. The march asks it of the step just entered and records the answer on that
        step's report (:attr:`~aquaflux.solve.StepReport.station`), and compares a step with the ones
        before it through those reports -- so an implementation need not remember past steps.
        """


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
        measured = [report.raw_cycles for report in history if report.raw_cycles > 0]
        if not measured:
            return False
        threshold = self.growth * min(measured)
        recent = history[-self.patience :]
        return all(report.raw_cycles > 0 and report.raw_cycles >= threshold for report in recent)


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


@eqx.filter_jit
def _march_step(
    strategy: NewtonStrategy,
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    phi: jnp.ndarray,
    residual_norm_0: jnp.ndarray,
    solver: lx.AbstractLinearSolver,
) -> tuple[StepOutcome, jnp.ndarray]:
    """One observed step: the step's :class:`StepOutcome`, and the residual norm at the state it produced.

    Compiled as a unit, and — this is the load-bearing part — ``strategy`` and ``residual_fn``
    are **arguments, not captured values**, so repeated steps hit the compilation cache instead of
    retracing the shifted solve every iteration (which would dominate the whole march). Two things
    are required of the caller for that to hold:

    * pass the **same** ``strategy`` object for every step of a segment (a rebuilt one is a new
      compilation, which is the intended one-off cost of a refresh); and
    * pass a **bound method** of a module as ``residual_fn`` (e.g. ``coupled.residual``), which is a
      pytree whose arrays ride as dynamic leaves. A freshly-created ``lambda`` is hashed by identity,
      so building one per step misses the cache every time.

    The next residual norm comes from the step itself (:attr:`StepOutcome.residual_norm`) rather than
    from a fresh evaluation here. A globalized step ends in a line search, which already formed the
    measure at the rung it kept, so evaluating the residual again at that same point cost a full
    residual per march step to recompute a number the step was holding. It is the step's own measure,
    which is this march's measure: when ``norm_builder`` is given, `newton_march` rebuilds the *step's*
    ``residual_norm`` at each outer iteration, so the search, the acceptance test and the reported norm
    are one measure by construction -- that is the invariant this relies on.
    """
    outcome = strategy.stepper()(residual_fn, phi, residual_norm_0, solver)
    return outcome, outcome.residual_norm


def newton_march(
    strategy: NewtonStrategy,
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    phi0: jnp.ndarray,
    *,
    max_steps: int,
    rtol: float,
    atol: float,
    reference_norm: float | None = None,
    damping_reference: float | None = None,
    trigger: RefreshTrigger | None = None,
    step_control: StepControl | None = None,
    control_state: object = None,
    observer: Callable[[StepReport], None] | None = None,
    checkpoint: Callable[[StepReport, jnp.ndarray], None] | None = None,
    drift_measure: Callable[[jnp.ndarray], float] | None = None,
    norm_builder: MeasureBuilder | None = None,
    refresh_preconditioner: Callable[..., None] | None = None,
    solver: lx.AbstractLinearSolver | None = None,
    retry: RetryPolicy = NO_RETRIES,
    adaptive_tolerance: AdaptiveLinearTolerance | None = None,
    on_retry: Callable[[str, int, float], None] | None = None,
    homotopy: ResidualHomotopy | None = None,
    station_step: Callable[[NewtonStrategy, int, bool], NewtonStrategy] | None = None,
    on_residuals: Callable[[Mapping[str, float]], None] | None = None,
    stopping_step: Callable[[NewtonStrategy], NewtonStrategy] | None = None,
    stopping_reference: float | None = None,
) -> MarchResult:
    """March the residual eagerly, reporting each step and stopping early if the trigger fires.

    The one Newton loop in this package: :class:`~aquaflux.solve.RootSolver` runs it, and
    so does every coupled driver. Stepping eagerly is what lets a driver observe per-step cost or
    interpose work that cannot run under ``jit`` (rebuilding a frozen preconditioner) -- and, for the
    same reason, is why a solve cannot be called inside ``jit`` or ``vmap`` (see
    :func:`refuse_a_transform_the_march_cannot_run_in`).

    **This function may return a state that does not solve the residual, without raising** — that is
    the point of a march that can stop early. It carries no convergence guard, so a caller must read
    :attr:`MarchResult.converged` before treating the state as a result. Do not differentiate through
    this march: run it on ``stop_gradient`` inputs and hand a converged state to
    :func:`~aquaflux.solve.root_adjoint`, which attaches the derivative at the root.

    Parameters
    ----------
    strategy : NewtonStrategy
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
    damping_reference : float, optional
        The scale the damping schedule ramps against, in place of the norm measured at ``phi0``. It is
        for a march **resuming** one that stopped short: the interrupted march's schedule was ramping
        against the norm at *its* first state, and measuring afresh at the resumed state re-bases the
        ramp there, which opens it at its starting strength again. Giving the interrupted march's own
        reference continues the ramp where it left off. The march still starts from the residual it
        actually measures at ``phi0`` -- only the schedule's anchor is replaced. ``None`` (the default)
        measures at ``phi0``, byte-identical to a march that was never resumed.
    trigger : RefreshTrigger, optional
        Consulted after every step; when it fires the march stops and reports ``triggered=True``.
        ``None`` marches to convergence or ``max_steps``.
    step_control : StepControl, optional
        Reshapes the step each iteration from the previous step's report (e.g. driving the shift
        strength β toward a line-search-factor target). ``None`` runs ``strategy`` unchanged, so
        the march is byte-identical to an uncontrolled one. Forward-only, like ``trigger``.
    control_state : object, optional
        The initial state for ``step_control`` (``None`` on a fresh march). A driver that runs one
        segment per refresh passes the previous segment's :attr:`MarchResult.control_state` here, so a
        stateful control continues across the refresh instead of resetting — the same discipline the
        *global* ``reference_norm`` follows, and the opposite of the deliberately segment-local damping
        reference and ``drift_measure``. A march resuming an interrupted one passes the control's own
        ``resumed_at(shift)`` here (see :class:`~aquaflux.solve.Resumption`). Ignored when
        ``step_control is None``.
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
        ``(step, state) -> ResidualNorm``, re-deriving the residual measure at the state each outer
        iteration starts from and holding it for that whole iteration — every trial step of the line search, the
        acceptance test and the reported norm. Rebuilding it per trial step instead would let a
        candidate win by shrinking its own denominator rather than its residual. The segment reference
        the damping schedule ramps against is taken in this same measure, so the ratio divides two
        comparably-scaled quantities. ``step`` is ``strategy``, for a measure that reads its scales from the
        step's shift; a solve builds one from its :class:`~aquaflux.solve.ResidualMeasure`.
        ``None`` (the default) uses ``strategy.norm()`` throughout.
    refresh_preconditioner : callable, optional
        ``(active_step, state, *, due) -> update or None``, called before each step (after the control
        has set the shift strength on ``active_step``) to refresh that step's frozen preconditioner from
        the current state and shift. It runs in this eager loop, outside the jitted ``_march_step``. A
        host preconditioner rides in a *static* field and is re-fitted in place, and the hook returns
        ``None``. A preconditioner carried as **data** cannot be mutated, so the hook returns an
        ``update``, ``step -> step``, which the march applies to the base step and the active one alike
        and reports on :attr:`MarchResult.strategy`. Either way the step's structure is unchanged, so
        ``_march_step`` stays a compilation-cache hit.

        ``due`` is ``True`` when the march is ASKING for a re-fit rather than offering one: the previous
        step stopped for one (:attr:`~aquaflux.solve.StepOutcome.refresh_due`, the dual-time loop's
        ``refresh_on_cycles``), or the step is being redone for its cost. A hook that gates its own
        cadence must re-fit when ``due`` is set; with ``due`` unset it decides for itself. The use case
        is a **materialized-Jacobian preconditioner re-fitted at the current ``(state, β)``**, since a
        frozen inverse mis-preconditions the shifted operator once the march's β leaves the value it was
        built at. Like
        the trigger and the control it is **forward-only** -- an impure mutation that must never be on a
        differentiated path. ``None`` (the default) leaves the preconditioner untouched, byte-identical to
        before.
    homotopy : ResidualHomotopy, optional
        Walk a sequence of related problems within this one march, ending at the target. When given,
        each outer step drives ``homotopy.enter(step, previous)`` instead of ``residual_fn``, and the march may
        **not** stop on its residual tolerance until ``homotopy.arrived(step)`` -- a converged
        intermediate station is not a converged answer. ``residual_fn`` stays the **target** residual
        and is what ``reference_norm`` defaults to, so the stopping bar is the target problem's
        throughout rather than a moving one. ``None`` (the default) marches ``residual_fn`` alone and
        is byte-identical.

        The damping anchor ``‖R₀‖`` is taken at the **first station** rather than at the target, since
        it is the scale the first step's inner loop is judged against; with no homotopy that is
        ``residual_fn`` and nothing changes.
    station_step : callable, optional
        ``(step, station, arrived) -> step``, letting the caller reshape the Newton step for the
        station it is about to run -- the counterpart, on the *step*, of what a homotopy does to the
        *problem*. Only consulted when a ``homotopy`` is given; ``None`` (the default) is
        byte-identical.

        It exists because a station change is a change of problem, and how hard a step should be
        damped is a property of the problem: a path whose intermediate stations are easier than the
        target may want a different damping on the way than at the end, and nothing else in the march
        can express that. Every signal a shift policy can read on its own -- the shift strength, the
        residual -- measures *progress*, which is a different thing from *which station this is* and
        on a short ramp saturates long before the last one begins.

        ⚠️ **Return a step differing only in ARRAY leaf values.** ``equinox.filter_jit`` compares
        non-array leaves by value on the static side, so swapping a Python scalar recompiles the whole
        solve at every station -- turning the cheapest possible setting into the dominant cost of the
        march. Swap with ``equinox.tree_at`` over a fixed structure, as the measure above is swapped.
    solver : lineax.AbstractLinearSolver, optional
        The linear solver for each step; defaults to ``strategy.linear_solver()``.
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
        original retry). Each escalation re-matches the preconditioner through ``refresh_preconditioner``;
        the divergence retry does not, because the factorization is already fresh at this
        ``(state, β)`` and only the Krylov tolerance is at fault.
    adaptive_tolerance : AdaptiveLinearTolerance or None
        Vary the relative tolerance each step's linear solve stops at, from how the previous step went:
        tighter after a step the line search cut back or whose residual rose, and tighter in a tail of
        full-length steps limited by their own solve (see :class:`~aquaflux.solve.AdaptiveLinearTolerance`).
        Each step starts from the base tolerance the march's ``solver`` was built with, which must be a
        relative-residual GMRES. A step's reduction is its reported residual over the one reported before
        it. ``None`` (default) solves every step to the solver's own tolerance.
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
    on_residuals : callable, optional
        ``{equation: term} -> None``, called after each step, before ``observer`` and ``checkpoint``,
        with each equation's term of the measure the step was judged in -- the numbers whose Euclidean
        combination is the report's ``residual_norm`` -- so a record of the march can say which
        equation is holding it up. Called only when that measure names its blocks
        (:func:`~aquaflux.solve.norm.named_blocks`); a plain Euclidean measure has nothing to split.
        Costs one residual evaluation per step, at the station the step drove; ``None`` (default)
        costs nothing.
    stopping_step : callable, optional
        ``step -> step``, the step whose measure the **stopping test** reads, from ``norm_builder``, once
        the march drives the target problem; every step is still steered, line-searched and reported in
        its own measure. For a march whose steps keep something built on another problem -- a
        continuation that holds its first station's shift diagonal, from which a row-scaled measure
        takes its row scales -- so that the target is judged in its own scales without changing a step
        (a session that offers it is a :class:`~aquaflux.solve.StoppingStepSource`). The tolerance is
        then ``atol + rtol * stopping_reference``. Costs a measure build and a residual evaluation per
        step once arrived. Needs ``norm_builder``. ``None`` (default) stops in the step's own measure.
    stopping_reference : float, optional
        The reference residual a ``stopping_step``'s tolerance is relative to, in its measure; by
        default ``residual_fn`` at ``phi0``, measured there. A driver marching in segments passes the one
        it measured at the start, as it does ``reference_norm``. Reports, and the controls and triggers
        that read them, keep ``reference_norm``.

    Returns
    -------
    MarchResult
        The state reached, the per-step reports, and whether the march converged or was triggered.

    Raises
    ------
    ValueError
        If the step can stop for a preconditioner re-fit (a :class:`~aquaflux.solve.StopsForRefresh`
        with ``refresh_on_cycles`` set) and no ``refresh_preconditioner`` is given to take it.
    """
    if solver is None:
        solver = strategy.linear_solver()
    # The tolerance an adaptive rule varies is the solver's own, carried as an array from the first step
    # so that every later value is a change of data and the compiled step a cache hit.
    base_tolerance = None if adaptive_tolerance is None else relative_tolerance(solver)
    if adaptive_tolerance is not None:
        if base_tolerance is None:
            raise TypeError(
                "adaptive_tolerance varies the relative tolerance of a relative-residual GMRES "
                "(relative_residual_gmres or residual_stop_gmres), and this march's linear solver is a "
                f"{type(solver).__name__}, which has none to vary."
            )
        solver = with_relative_tolerance(solver, base_tolerance)
    tolerance = base_tolerance
    if stopping_step is not None and norm_builder is None:
        raise ValueError(
            "a stopping_step is judged in the measure norm_builder builds from it, so it needs a "
            "norm_builder."
        )
    # When the measure is rebuilt each iteration, the segment reference must be taken in that same
    # measure -- otherwise the damping schedule divides two differently-scaled quantities.
    norm = norm_builder(strategy, phi0) if norm_builder is not None else strategy.norm()

    # The segment-local reference: what the step's damping schedule ramps against. Recomputed here,
    # never inherited, so a segment resumed after a refresh restarts its ramp. It is fixed for the
    # whole segment: recomputing it per step would hold the ratio at one, pinning the shift at its
    # starting strength and freezing the march.
    # The anchor is the scale the FIRST step's inner loop is judged against, so it is taken at the
    # station that step actually runs. With no homotopy that is `residual_fn` and this is unchanged.
    measured_norm_0 = norm_of_residual(
        norm, residual_fn if homotopy is None else homotopy.enter(0, None), phi0
    )
    residual_norm_0 = (
        measured_norm_0 if damping_reference is None else jnp.asarray(float(damping_reference))
    )
    reference = float(measured_norm_0) if reference_norm is None else float(reference_norm)
    # A tolerance relative to a residual is relative to that residual in the measure it is judged in.
    if stopping_step is None:
        stop_reference = reference
    elif stopping_reference is not None:
        stop_reference = float(stopping_reference)
    else:
        stop_reference = float(
            norm_of_residual(norm_builder(stopping_step(strategy), phi0), residual_fn, phi0)
        )

    # Both thresholds are knowable INSIDE a step -- the cost one the moment a solve returns, the
    # step-length one the moment a line search collapses -- but the reaction below only runs once the
    # whole step is back. Push them down to the step so it can stop there and then, rather than
    # finishing inner iterations whose results this loop is about to discard. One number each, set in
    # one place: a step that took its own copy would be a second spelling to keep in step with this one.
    strategy = retry.with_inner_abort(strategy)
    # A step that stops for a re-fit only helps if something takes the re-fit. Refused here, before the
    # first step, rather than at the first hard step -- which may be an hour into a march.
    if (
        isinstance(strategy, StopsForRefresh)
        and strategy.refresh_on_cycles is not None
        and refresh_preconditioner is None
    ):
        raise ValueError(
            f"the step stops for a preconditioner re-fit (refresh_on_cycles="
            f"{strategy.refresh_on_cycles}) but this march has no refresh_preconditioner to take it, so "
            "every hard step would end early and the preconditioner would never be refreshed. March with a "
            "preconditioner session (a MaterializedJacobian preconditioner supplies the hook), or leave "
            "refresh_on_cycles unset."
        )
    # Built once, so every divergence retry of the march hands the step the same solver.
    tight_solver = None if retry.solver is None else retry.solver.build()

    state = phi0
    current = float(measured_norm_0)
    reports: list[StepReport] = []
    triggered = False
    # Whether the step just taken stopped to have its preconditioner re-fitted: the next refresh is then
    # asked for rather than offered, at the iterate that step kept.
    refresh_due = False
    # `control_state` is a parameter (the initial state), threaded and returned so a multi-segment
    # driver can continue a stateful control across a refresh instead of restarting it.

    def judged_reference() -> float:
        # The starting residual the stopping test scales its relative tolerance by: the steering
        # measure's, or -- once arrived with a stopping step -- the stopping measure's.
        return stop_reference if stopping_step is not None and arrived else reference

    def converged_at(residual_norm: float) -> bool:
        # Finiteness is tested separately because `within_tolerance` alone cannot reject a diverged
        # march: it compares with `<=`, and when the residual norm AND the threshold it is judged
        # against have both run away to `+inf`, `inf <= inf` is True -- a false "converged" on a state
        # that solves nothing. A NaN norm already fails the comparison on its own (every NaN
        # comparison is false); `+inf` is the case that needs this.
        return bool(
            jnp.isfinite(jnp.asarray(residual_norm))
            and within_tolerance(
                jnp.asarray(residual_norm),
                judged_reference(),
                rtol,
                atol,
            )
        )

    # Whether the step just set up runs the TARGET problem. A homotopy march may not stop on the
    # residual tolerance before it does -- a small residual at an intermediate station says nothing
    # about the target -- so this starts False and the loop always takes at least one step. Without a
    # homotopy it is True throughout and the stopping test is unchanged.
    arrived = homotopy is None

    def judged(step: NewtonStrategy, phi: jnp.ndarray, residual: Callable, steered: float) -> float:
        # The residual norm the stopping test reads: the step's own (`steered`), or -- once arrived --
        # the one `stopping_step` measures the target in.
        if stopping_step is None or not arrived:
            return steered
        return float(norm_of_residual(norm_builder(stopping_step(step), phi), residual, phi))

    stop_norm = judged(strategy, phi0, residual_fn, current)

    # A non-finite residual at the state the march is handed stops it before it steps: the step's
    # linear solve would be asked to factor a NaN/inf right-hand side and raise from inside the Krylov
    # solver, which reports a bug upstream of itself rather than the fact the caller needs -- that this
    # march never left a state solving nothing. The post-step break below is the same rule one step
    # later, for a residual that goes non-finite on the way.
    while (
        len(reports) < max_steps
        and math.isfinite(current)
        and not (converged_at(stop_norm) and arrived)
        and not triggered
    ):
        # The station this step drives. `enter` is where a homotopy re-points whatever must follow the
        # parameter (a preconditioner refresh), so it runs before the control and the refresh below.
        step_residual = residual_fn
        if homotopy is not None:
            step_residual = homotopy.enter(len(reports), reports[-1] if reports else None)
            arrived = homotopy.arrived(len(reports))
        # A step control reshapes the base step from the previous report (None runs it unchanged, so
        # the loop is byte-identical). It threads its own state; the march stays ignorant of β.
        active_step = strategy
        if norm_builder is not None:
            # Re-derive the residual measure at the state this outer iteration starts from, and hold
            # it for the whole iteration -- every trial step of the line search, the acceptance test
            # and the reported norm all use this one. Rebuilding it per trial step instead would let a
            # candidate win by shrinking its own denominator rather than its residual, so the search
            # would stop comparing like with like. The swap is a compilation-cache hit as long as the
            # measure carries its scales as data over a fixed block structure.
            active_step = active_step.with_norm(norm_builder(strategy, state))
        if station_step is not None and homotopy is not None:
            # Let the caller reshape the step for the station it is about to run -- the seam by which
            # anything a homotopy changes about the PROBLEM can be matched by a change to how the step
            # is DAMPED. `enter` already re-points what must follow the parameter downstream; this is
            # its counterpart on the step itself, and it is here rather than inside the homotopy
            # because the march owns the step and a homotopy knows nothing about a step's internals.
            #
            # ⚠️ Whatever it swaps must be an ARRAY leaf over a fixed structure, exactly as the measure
            # above must be: `filter_jit` compares non-array leaves by value on the static side, so
            # swapping a Python scalar recompiles the whole solve once per station and turns a
            # station-varying setting into the most expensive thing in the march.
            active_step = station_step(active_step, homotopy.station(len(reports)), bool(arrived))
        # Entering a new station: re-damp deliberately, and HOLD the control for this step so it does
        # not immediately adapt the re-damping away. The hold is the same treatment a refresh boundary
        # gets, for the neighbouring reason -- there the preconditioner moved under the march, here the
        # residual did. Without the re-damping the shift walks straight through the level the new
        # station can carry: measured on a pitzDaily ramp, into a line-search collapse and a
        # two-order residual excursion the retry ladder then had to climb back out of.
        entering = (
            homotopy is not None
            and reports
            and homotopy.station(len(reports)) != reports[-1].station
        )
        previous_report = reports[-1] if reports else None
        if entering and isinstance(step_control, ShiftCarryingControl):
            factor = float(homotopy.shift_factor(len(reports)))
            if factor != 1.0:
                control_state = step_control.redamp(control_state, factor)
                previous_report = None  # hold: the re-damping is this step's adaptation
        if step_control is not None:
            # A residual-ratio rule divides the last step's residual by the one before it, and that
            # ratio only means something when both came from the SAME station. The previous step began
            # a new one here, so the pair straddles a change of problem: drop the remembered residual
            # and let the control fall back to alpha for one step, exactly as it does on the first step
            # of a march. Without this the control reads the residual rise that a harder new station
            # causes as an overshoot and brakes on it -- measured firing at two of four station changes
            # on a pitzDaily viscosity ramp, both with a fully comfortable inner loop.
            # Not a hold (`previous=None`, the refresh boundary's treatment): there the residual
            # function is unchanged and its reference is still worth keeping, and here it is not.
            if (
                homotopy is not None
                and len(reports) >= 2
                and reports[-1].station != reports[-2].station
                and isinstance(step_control, ShiftCarryingControl)
            ):
                control_state = step_control.rebase(control_state)
            active_step, control_state = step_control.next_step(
                active_step, previous_report, control_state
            )
        # Check the step can support what was ASKED FOR -- loudly, and once. The escalation drives the
        # pseudo-transient shift, which `NewtonStrategy` does not promise (see `RetryPolicy.require_shifted`);
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
        if refresh_preconditioner is not None:
            # Refresh the step's frozen preconditioner from the state and shift strength this step is
            # about to run at -- e.g. re-fitting a materialized-Jacobian inverse at the current
            # (state, β), so it preconditions the operator actually solved. Runs HERE, in the eager loop (a host
            # op outside the jitted `_march_step`), after the control has set β on `active_step`. A host
            # preconditioner is mutated in place; one carried as data comes back as a step update,
            # applied to the base step as well so every later step starts from it. Either way the
            # step's structure is unchanged and `_march_step` stays a compilation-cache hit.
            # Forward-only, like the trigger and the control. `due` when the previous step stopped for
            # exactly this: `state` is then the iterate it kept, so the re-fit is taken where that
            # step's expensive solve left off.
            update = refresh_preconditioner(active_step, state, due=refresh_due)
            if update is not None:
                strategy, active_step = update(strategy), update(active_step)
                if norm_builder is not None:
                    # An update may carry a new shift policy (a session re-pointed at another problem),
                    # and the measure is built from the step's policy, so build it again from the step
                    # as updated: otherwise the step that enters a new problem is judged -- its line
                    # search, its inner stop, the residual it reports -- in the previous one's scales.
                    active_step = active_step.with_norm(norm_builder(strategy, state))
        prestep_state = state
        step_solver = solver if tolerance is None else with_relative_tolerance(solver, tolerance)
        outcome, residual_norm = _march_step(
            active_step, step_residual, prestep_state, residual_norm_0, step_solver
        )
        # A step can go bad three ways -- a non-finite / diverging correction, a step length collapsed
        # to `retry.on_alpha`, or a solve truncated by `retry.abort_above_cycles` -- and they DO NOT
        # share one response. The first two are stiffness and are cured by MORE damping; the third is a
        # statement about the frozen preconditioner, whose cure is a FRESH one (re-fitted below before
        # the redo), so it is redone at the shift it already had. `RetryPolicy.retry_reason` names the reason and `ESCALATING_REASONS`
        # says which of them raise β. Escalate β FIRST (redo from the pre-step state at `β *= retry.beta_factor`,
        # re-matching the frozen preconditioner via `refresh_preconditioner`), because a larger β lifts the
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
        redone_on_fresh = False
        while (
            (reason := retry.retry_reason(outcome, residual_norm, reference)) is not None
            and retries < retry.max_retries
            and not converged_at(
                judged(active_step, outcome.phi, step_residual, float(residual_norm))
            )
        ):
            escalating = reason in ESCALATING_REASONS
            if not escalating:
                # The fresh-preconditioner redo is worth exactly ONE attempt. It differs from the first
                # only in starting on the preconditioner re-fitted for it, so a second redo
                # at the same shift on the same factorization would repeat it exactly -- burning the
                # escalation budget on identical attempts and, with no other reason ever firing,
                # spinning to `max_retries` every step.
                if redone_on_fresh:
                    break
                redone_on_fresh = True
            retries += 1
            # `RetryPolicy.escalate` SCALES the existing β leaf rather than rebuilding one; its
            # docstring carries why (a rebuilt leaf's dtype/weak type need not match, and any mismatch
            # recompiles the whole coupled solve on every retry).
            escalated = (
                retry.escalate(active_step.relaxation_schedule.beta)
                if escalating
                else shift_of(active_step)
            )
            # Report the escalated β, and so only once it exists: `on_retry` promises the shift the
            # retried attempt will RUN at. Reporting the pre-escalation leaf here instead left the one
            # consumer reconstructing it as `beta * 2` -- correct only at the default `beta_factor`,
            # and wrong at every other, in the log a long march is read back from.
            if on_retry is not None:
                on_retry(reason, retries, float(escalated or 0.0))
            if escalating:
                active_step = eqx.tree_at(
                    lambda s: s.relaxation_schedule.beta, active_step, escalated
                )
            if refresh_preconditioner is not None:
                # Re-match the preconditioner to the escalated β. A redo for COST asks for a re-fit
                # (`due`): the cost said the frozen preconditioner is struggling, and redoing the
                # attempt on the same one would repeat it exactly. So does a redo of an attempt that
                # had itself stopped for a re-fit. Otherwise, whether this actually rebuilds is the
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
                update = refresh_preconditioner(
                    active_step,
                    prestep_state,
                    due=not escalating or bool(outcome.refresh_due),
                )
                if update is not None:
                    strategy, active_step = update(strategy), update(active_step)
            outcome, residual_norm = _march_step(
                active_step, step_residual, prestep_state, residual_norm_0, step_solver
            )
        # Divergence retry -- the FALLBACK for a non-finite correction β-escalation could not fix. An
        # inexact preconditioner can return a non-finite correction where the loose
        # default Krylov tolerance left it too inaccurate; the cure there is a tighter Krylov solve, not
        # more damping (the factors are already fresh at this (state, β), so re-preconditioning is a no-op).
        # This fires only if the step is STILL diverged after the escalation loop -- or if escalation was
        # unavailable (no threshold set, or no β leaf) -- redoing the SAME step from the SAME pre-step
        # state with the tighter solver `retry.solver` names.
        # One retry; a still-diverged step breaks below as it would without a retry. A policy with no
        # `solver` (the default) is byte-identical.
        diverged_retry = retry.solver is not None and retry.has_diverged(residual_norm, reference)
        if diverged_retry:
            if on_retry is not None:
                # `shift_of`, not a direct read: the divergence retry needs no shift, so it runs on
                # steps that have none, and this reported the shift by reaching straight through
                # `relaxation_schedule` -- raising `AttributeError` on a step that satisfies
                # `NewtonStrategy` in full. Unchanged beta here in any case; nothing escalated.
                on_retry("solver", retries + 1, float(shift_of(active_step) or 0.0))
            outcome, residual_norm = _march_step(
                active_step, step_residual, prestep_state, residual_norm_0, tight_solver
            )
        # Carry an escalated β forward into the control. The escalation raised β because the control had
        # driven it too low for this operator; without carrying that back, the next `next_step` recomputes
        # from the control's own (floor-ward) β and re-pays the escalation every step on a persistently hard
        # region -- the low-β reachability tail, where β sits at its floor and each step re-escalates. The
        # escalation IS the feedback for "how low is safe here", so it replaces a static β floor: seeding the
        # control's carried β with the escalated value lets the control continue from the discovered-safe
        # level (and adapt on from there), so β_min can be driven toward zero and the *controller* decides how
        # large a pseudo-timestep is safe. Only when β was actually escalated, and only for a control that
        # carries β (a `ShiftCarryingControl`); no escalation ⇒ byte-identical.
        if (
            retries
            and isinstance(step_control, ShiftCarryingControl)
            and (carried := shift_of(active_step)) is not None
        ):
            control_state = step_control.carry_beta(control_state, float(carried))
        state = outcome.phi
        previous = current
        current = float(residual_norm)
        stop_norm = judged(active_step, state, step_residual, current)
        step_tolerance = tolerance
        if adaptive_tolerance is not None:
            # The reduction is the march's own, report to report, although the measure is rebuilt each
            # outer iteration so its two ends are in different scales. That is deliberate and measured: a
            # residual that rose in the record is the setback signal the rule acts on, and re-measuring
            # the step's start in its own scales instead hid exactly that rise on a Reynolds-continuation
            # march, which then failed where this converged.
            tolerance = adaptive_tolerance.next_tolerance(
                base_tolerance,
                alpha=float(outcome.alpha),
                reduction=current / previous if previous > 0.0 else math.inf,
                stopping_norm=stop_norm,
                target=atol + rtol * judged_reference(),
            )
        refresh_due = bool(outcome.refresh_due)
        # Not every NewtonStrategy carries a relaxation schedule (a plain damped-Newton step has none), and
        # a schedule need not expose a readable beta -- report 0 rather than demanding either.
        step_shift = shift_of(active_step)
        report = StepReport(
            step=len(reports),
            raw_cycles=int(outcome.raw_cycles),
            residual_norm=current,
            residual_ratio=current / reference if reference > 0.0 else 0.0,
            alpha=float(outcome.alpha),
            drift=0.0 if drift_measure is None else float(drift_measure(state)),
            inner_iterations=int(outcome.inner_iterations),
            max_inner_cycles=int(outcome.max_inner_cycles),
            shift=0.0 if step_shift is None else float(step_shift),
            escalations=int(retries),
            diverged_retry=bool(diverged_retry),
            damping_reference=float(residual_norm_0),
            station=0 if homotopy is None else int(homotopy.station(len(reports))),
            arrived=bool(arrived),
            full_step_ratio=float(outcome.full_step_ratio),
            stopping_norm=stop_norm,
            stopping_ratio=stop_norm / judged_reference() if judged_reference() > 0.0 else 0.0,
            linear_tolerance=math.nan if step_tolerance is None else float(step_tolerance),
        )
        reports.append(report)
        if on_residuals is not None:
            terms = named_blocks(active_step.norm(), step_residual, state)
            if terms is not None:
                on_residuals(terms)
        if observer is not None:
            observer(report)
        if checkpoint is not None:
            checkpoint(report, state)
        # A non-finite residual can never satisfy the tolerance test, so without this the march
        # would spend its whole budget stepping a poisoned state. Stop and let the caller's
        # convergence test report the failure, which is where non-convergence is diagnosed.
        if not jnp.isfinite(residual_norm):
            break
        if trigger is not None and not converged_at(stop_norm):
            triggered = trigger.should_refresh(reports)

    return MarchResult(
        state=state,
        reports=tuple(reports),
        converged=converged_at(stop_norm) and arrived,
        triggered=triggered,
        control_state=control_state,
        strategy=strategy,
    )
