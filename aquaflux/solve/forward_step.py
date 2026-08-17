"""The contracts the forward march is written against, in one place.

Four things travel between the Newton driver, the eager march, the globalization strategies, the step
controls and the retry policy: what a **strategy** must provide (:class:`ForwardStep`, and
:class:`ShiftedForwardStep` for the shift-driven half), what a **step** returns
(:class:`StepOutcome`), what a **march** reports (:class:`StepReport`), and what a **control** does with
that report (:class:`StepControl`). None of them belongs to any one of those modules, and every one of
them was living in whichever module happened to need it first.

**That is not tidiness; the placement had a cost that came due twice.** ``StepControl`` was declared in
``march.py`` with *zero* implementations there, so ``step_control.py`` had to import ``march`` -- which
forbade the reverse, so a defaulting rule about two ``solve/`` objects could not be written in
``solve/`` at all and ended up a package away in the turbulence driver. And ``implicit.py``, named for
the Newton solver, was a de-facto contract module handing three names across boundaries with underscore
prefixes, which under this package's own convention read as violations.

Nothing here imports from a sibling that imports it back, so this module is a leaf: the strategies, the
march, the controls and the retry policy all depend on it and it depends on none of them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Protocol, runtime_checkable

import jax.numpy as jnp
import lineax as lx

from .linear import restart_cycles as _strip_step_offset
from .norm import ResidualNorm
from .relaxation import RelaxationSchedule

__all__ = [
    "ForwardStep",
    "ShiftedForwardStep",
    "StepControl",
    "StepFn",
    "StepOutcome",
    "StepReport",
    "within_tolerance",
]


class StepOutcome(NamedTuple):
    """What one forward step produced, what it cost, and how it ended.

    A record rather than a widening tuple: these seven values travel together through every stepper and
    both consumers, and a positional 7-tuple is where a caller silently mis-unpacks one for another.

    Attributes
    ----------
    phi : jnp.ndarray
        The stepped iterate -- the only part that is the step's *result*; the rest is cost and quality.
    cycles : jnp.ndarray
        The **raw** solver count summed over the step's inner iterations (lineax's ``num_steps``, with
        its per-solve offset still in). Total cost, and what a summed cost cap is measured against.
    alpha : jnp.ndarray
        The line-search factor: for an inner loop, the **minimum** over its iterations with any
        non-descending one folded in as ``0``.
    inner_iterations : jnp.ndarray
        How many inner iterations ran. ``1`` for a step with no inner loop.
    reached_target : jnp.ndarray
        Whether the step ran to its **own** stopping criterion rather than being cut short. A cost-only
        trigger cannot tell a step that did its job expensively from one that ground and gave up, and
        would discard the former -- throwing away a good iterate for a shorter step.
    max_inner_cycles : jnp.ndarray
        The **offset-corrected** cost of the step's most expensive single solve. This is the
        inner-count-invariant difficulty signal: the summed :attr:`cycles` grows with how many times the
        step solved, so a threshold on it is ~6x more sensitive for a 5-iteration step than a
        1-iteration one, and the same per-solve difficulty trips it or not depending on a count that
        says nothing about conditioning.
    binding_limit : jnp.ndarray
        The step cap **where it was the binding constraint**, else ``1``. A small ``alpha`` has two
        completely different causes -- the direction overshot (shorten it, escalate the shift) or a
        constraint bound (the direction is fine, it just cannot be followed that far) -- and they call
        for opposite responses, so ``alpha`` alone cannot be acted on. Below ``1`` means an injected
        limit, not the descent test, decided the step length; the value is how tight it was.
    """

    phi: jnp.ndarray
    cycles: jnp.ndarray
    alpha: jnp.ndarray
    inner_iterations: jnp.ndarray
    reached_target: jnp.ndarray
    max_inner_cycles: jnp.ndarray
    binding_limit: jnp.ndarray


StepFn = Callable[
    [Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray, jnp.ndarray, lx.AbstractLinearSolver],
    StepOutcome,
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

    def stepper(self) -> StepFn:
        """The forward step ``(residual_fn, phi, residual_norm_0, solver) -> StepOutcome``.

        ``StepOutcome.cycles`` is the restart-cycle count of the linear solve behind the accepted step
        (its cost, which an observed march reads to detect a stale preconditioner);
        ``StepOutcome.alpha`` is the line-search factor of that step (its quality — ``1`` if the full
        shifted step descended, smaller if it was clipped, the signal a step controller drives the
        shift by). There is no variant of this method that drops them; a caller that wants only the
        iterate writes ``outcome = step(…)`` and reads ``outcome.phi``.
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


@runtime_checkable
class ShiftedForwardStep(ForwardStep, Protocol):
    """A :class:`ForwardStep` whose globalization is a **shift strength an external control can drive**.

    :class:`ForwardStep` says what every strategy must *do*. This says what a strategy must additionally
    *carry* for the eager march's feedback machinery to work on it: a ``relaxation_schedule`` holding the
    pseudo-transient shift ``beta`` as a readable, replaceable leaf. The schedule that exposes such a
    leaf is :class:`~aquaflux.solve.ConstantRelaxation`, which a :class:`StepControl` swaps in once per
    iteration; the :class:`~aquaflux.solve.SwitchedEvolutionRelaxation` a shifted step is built with by
    default computes ``beta`` from the residual ratio and exposes nothing to read or replace.

    **Why it is a separate protocol rather than more of `ForwardStep`.** Not every strategy has a shift.
    :class:`DampedNewtonStep` globalizes by backtracking alone and has no ``relaxation_schedule`` at all,
    and requiring one of it would be inventing a quantity it does not possess. But
    :func:`~aquaflux.solve.forward_march`'s beta escalation and every
    :class:`~aquaflux.solve.StepControl` *do* need one -- they raise beta on a bad step and drive it
    between steps -- so the requirement is real and belongs written down.

    **Why an explicit up-front check rather than a ``hasattr`` probe at the point of use.** A probe fails
    *silently*: a `DampedNewtonStep` satisfies `ForwardStep` completely, so passing one with
    ``RetryPolicy.on_alpha`` set is accepted and then simply never escalates -- and a march that quietly
    declines to escalate looks exactly like one that never needed to. Reading
    ``active_step.relaxation_schedule`` unguarded fails the opposite way, raising ``AttributeError``
    mid-march on a step that conforms. The march therefore checks this once, before the first step, and
    names which feature needs what.

    Notes
    -----
    ``beta`` must be a **dynamic** array leaf, not a static field: the march escalates it by *scaling*
    the existing leaf so its dtype and weak-type are preserved, which is what keeps the jitted march step
    a compilation-cache hit rather than recompiling the whole coupled solve on every retry.
    """

    relaxation_schedule: RelaxationSchedule


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
        :attr:`restart_cycles` (offset-corrected) for the honest per-step cycle count. **``0`` means "no measurement", not "free":** a
        pseudo-transient step records its count only on acceptance, so a step whose every damping
        attempt was rejected reports ``0`` despite having burned several solves — skip zeros.
    inner_iterations : int
        How many inner Newton iterations the step took: the backward-Euler inner-loop count for a
        :class:`~aquaflux.solve.DualTimeStep` (what the summed :attr:`cycles` is spread over), and ``1``
        for a single-step (pseudo-transient / damped-Newton) march. Reporting it separately from
        :attr:`cycles` is what keeps the two costs — nonlinear inner work vs linear solve cost — from
        being conflated into one misleading number.
    max_inner_cycles : int
        The offset-corrected cost of the step's most expensive **single** solve -- the
        inner-count-invariant difficulty signal, and what the escalation triggers on. ``cycles`` sums
        over the inner iterations, so a threshold on it is ~6x more sensitive for a 5-iteration step
        than a 1-iteration one and answers partly a question about nonlinear difficulty rather than
        conditioning. ``0`` when not measured.
    binding_limit : float
        The step cap where it was the **binding** constraint, else ``1``. A small ``alpha`` means
        either that the direction overshot or that a constraint stopped the step being followed
        further -- opposite diagnoses, so ``alpha`` alone cannot be acted on or reported honestly.
    residual_norm : float
        The residual measure at the state the step produced.
    residual_ratio : float
        ``residual_norm`` divided by the march's global reference norm — how far the solve has come,
        on the same scale for every segment.
    escalations : int
        How many times the step was redone with ``beta`` escalated before it was accepted (the
        escalation bailout). ``0`` for a step taken as-is.
    diverged_retry : bool
        Whether the step was redone by the retry policy's tighter solver after diverging. Recorded
        alongside
        ``escalations`` because ``cycles`` reports only the **accepted** attempt: without them a redone
        step is indistinguishable from a cheap one, and a retry mechanism that never fires (because it
        was left unconfigured) is invisible in the log.
    shift : float
        The pseudo-transient shift strength ``beta`` the step was taken at, read from the step's
        relaxation schedule (``0`` for a schedule that exposes none). Recorded because ``beta`` is what
        a :class:`StepControl` steers and what a preconditioner's staleness is a function of, so a log
        or a diagnostic that omits it cannot explain why a step cost what it did -- and every driver
        that wants it would otherwise wrap the control to capture it.
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
    max_inner_cycles: int = 0
    binding_limit: float = 1.0
    shift: float = 0.0
    escalations: int = 0
    diverged_retry: bool = False

    @property
    def restart_cycles(self) -> int:
        """The offset-corrected restart-cycle count.

        ``cycles`` with lineax's +2-per-inner-solve offset removed, so an ideal one-cycle solve reads as
        ``1`` and a dual-time step as its real total cycles over the inner loop. Clamped at ``0`` (a
        no-measurement ``cycles = 0`` step stays ``0``).
        """
        return _strip_step_offset(self.cycles, self.inner_iterations)


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


def within_tolerance(residual_norm, residual_norm_0, rtol, atol):
    """The Newton stopping test: the residual norm has dropped to the absolute/relative floor."""
    return residual_norm <= atol + rtol * residual_norm_0
