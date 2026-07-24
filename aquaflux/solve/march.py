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
so a segment restarted after a refresh must restart its ramp too: a refresh rebuilds the shift
diagonals, and under a logarithmic solve variable those grow at a developed state, so pairing a
grown diagonal with the small ``beta`` belonging to the *pre-refresh* residual over-damps the step
and the march silently stops descending. The separate ``reference_norm`` is the *global* scale
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


class StepReport(NamedTuple):
    """What one march step cost and where it left the residual.

    Attributes
    ----------
    step : int
        The 0-based index of the step within its march segment.
    cycles : int
        The restart-cycle count of the linear solve behind the accepted step — the step's cost.
        **``0`` means "no measurement", not "free":** a pseudo-transient step records its count only
        on acceptance, so a step whose every damping attempt was rejected reports ``0`` despite
        having burned several solves. A consumer reading this as a cost signal must skip zeros.
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
    """

    step: int
    cycles: int
    residual_norm: float
    residual_ratio: float
    alpha: float


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
    """

    state: jnp.ndarray
    reports: tuple[StepReport, ...]
    converged: bool
    triggered: bool


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


@eqx.filter_jit
def _march_step(
    forward_step: ForwardStep,
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    phi: jnp.ndarray,
    residual_norm_0: jnp.ndarray,
    solver: lx.AbstractLinearSolver,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One observed step: the next state, its cycle count, line-search factor, and new residual norm.

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
    phi_next, cycles, alpha = forward_step.stepper()(residual_fn, phi, residual_norm_0, solver)
    return phi_next, cycles, alpha, forward_step.norm()(residual_fn(phi_next))


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
    observer: Callable[[StepReport], None] | None = None,
    checkpoint: Callable[[StepReport, jnp.ndarray], None] | None = None,
    solver: lx.AbstractLinearSolver | None = None,
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
    solver : lineax.AbstractLinearSolver, optional
        The linear solver for each step; defaults to ``forward_step.default_solver()``.

    Returns
    -------
    MarchResult
        The state reached, the per-step reports, and whether the march converged or was triggered.
    """
    if solver is None:
        solver = forward_step.default_solver()
    norm = forward_step.norm()

    # The segment-local reference: what the step's damping schedule ramps against. Recomputed here,
    # never inherited, so a segment resumed after a refresh restarts its ramp.
    residual_norm_0 = jnp.asarray(norm(residual_fn(phi0)))
    reference = float(residual_norm_0) if reference_norm is None else float(reference_norm)

    state = phi0
    current = float(residual_norm_0)
    reports: list[StepReport] = []
    triggered = False
    control_state: object = None

    def converged_at(residual_norm: float) -> bool:
        return bool(_within_tolerance(jnp.asarray(residual_norm), reference, rtol, atol))

    while len(reports) < max_steps and not converged_at(current) and not triggered:
        # A step control reshapes the base step from the previous report (None runs it unchanged, so
        # the loop is byte-identical). It threads its own state; the march stays ignorant of β.
        active_step = forward_step
        if step_control is not None:
            active_step, control_state = step_control.next_step(
                forward_step, reports[-1] if reports else None, control_state
            )
        state, cycles, alpha, residual_norm = _march_step(
            active_step, residual_fn, state, residual_norm_0, solver
        )
        current = float(residual_norm)
        report = StepReport(
            step=len(reports),
            cycles=int(cycles),
            residual_norm=current,
            residual_ratio=current / reference if reference > 0.0 else 0.0,
            alpha=float(alpha),
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
    )
