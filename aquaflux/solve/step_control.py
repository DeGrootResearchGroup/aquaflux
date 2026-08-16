"""Feedback step controls for the eager march (forward-only).

A :class:`~aquaflux.solve.StepControl` reshapes the forward step each iteration from the previous
step's outcome. All three members here drive the pseudo-transient shift strength β, and all three are
:class:`ShiftStrengthControl` subclasses supplying nothing but their adaptation rule:

* :class:`DualTimeControl` ramps the *dual-time* pseudo-timestep by a Courant rule (grow it while the
  backward-Euler inner loop is comfortable, ``α = 1``; shrink it when an inner step clips), which lets β
  be driven well below the single-step divergence floor because the inner loop keeps the larger implicit
  step stable. This is the **default control for the dual-time coupled march**: measured on a cold-start
  pitzDaily Reynolds-continuation ramp it reaches the developed recirculation in ~4× fewer outer steps
  than the residual-keyed alternative, and it converges standalone (to the requested tolerance, not a
  short plateau) — the pseudo-timestep is bounded by its ``beta_min`` floor, and the shift vanishes at
  the root, so the finishing solve owns the converged root and the adjoint regardless.
* :class:`ResidualRatioDualTimeControl` ramps the same pseudo-timestep by the *steady-residual* reduction
  ratio instead of α (switched evolution relaxation). Opt-in: it is the safer rule when the steady
  residual is a reliable progress signal, but on the pitzDaily ramp the row-scaled residual is nearly
  flat while the flow develops (the ``β × travel`` identity), so it pins β and stalls the pseudo-timestep.
* :class:`CflResidualDualTimeControl` combines the two: it grows the pseudo-timestep on the Courant
  signal α (:class:`DualTimeControl`'s speed) but brakes it on a rising steady residual
  (:class:`ResidualRatioDualTimeControl`'s overshoot safety), so it is fast on the flat-residual
  development where the residual-only rule stalls *and* safe on the overshoot where the α-only rule
  diverges — the two controls' failure modes are disjoint, and this grows only when both signals are
  comfortable. It reduces exactly to :class:`DualTimeControl` at infinite ratio thresholds, which is
  pinned by a test rather than left as a claim.

All three are forward-only accelerators on the eager march — they read the previous step's report and may
raise under ``jax.grad`` — so they live here rather than on the differentiable Newton path.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .continuation import DualTimeStep
from .forward_step import ForwardStep, StepControl, StepReport
from .relaxation import ConstantRelaxation


class ShiftStrengthControl(eqx.Module):
    """The shared body of a step control that drives the pseudo-transient shift strength β.

    A control is a small rule wrapped in a lot of identical bookkeeping: seed β on the first step of a
    march, hold it across a preconditioner refresh, adapt it within a segment, clamp it, swap it onto the
    step, and accept an externally-escalated β from the march. **Only the adaptation rule differs between
    the concrete controls**, and it is two to five lines in each; everything else is written once here.

    Solving it three times is what let them drift. :meth:`carry_beta` was byte-identical in two of them
    and **absent from a third**, which :func:`~aquaflux.solve.forward_march` probes for with ``hasattr``
    — so that control silently could not receive the escalation feedback, and nothing reported it. The
    same class also reset β to ``beta_start`` at every refresh boundary rather than holding it, which is
    the sawtooth defect recorded and fixed for the dual-time controls; it never got the fix, because the
    fix landed on the other side of a duplicated seam.

    **The carried state is ``(beta, memo)``.** β is universal — every control adapts it and the march
    escalates it — and ``memo`` is whatever else the rule needs to remember between steps: ``None`` for a
    memoryless rule, the previous residual measure for the two that key on a reduction ratio. Keeping one
    shape is what lets :meth:`next_step` and :meth:`carry_beta` be written once; a subclass never touches
    the state directly, only its own ``memo``.

    A subclass supplies :meth:`_adapt` and its own fields, nothing else.

    Attributes
    ----------
    beta_start : float
        β for the first step of the whole march (static).
    beta_min, beta_max : float
        Clamps on β (static). ``beta_min`` bounds how large the pseudo-timestep may grow.
    """

    beta_start: float = eqx.field(static=True, default=2.0)
    beta_min: float = eqx.field(static=True, default=0.02)
    beta_max: float = eqx.field(static=True, default=4.0)

    def _adapt(self, beta: float, previous: StepReport, memo: object) -> tuple[float, object]:
        """The control's rule: the next β and the next memo, from the step just taken.

        Parameters
        ----------
        beta : float
            The shift strength the previous step ran at.
        previous : StepReport
            The report of that step -- the feedback the rule reads (``alpha``, ``residual_norm``).
        memo : object
            Whatever this control remembered last call (``None`` on the first adaptation).

        Returns
        -------
        tuple
            ``(beta, memo)``. Clamp the β through :meth:`_clamp`; the base does not clamp for you,
            because a rule may want to decline to adapt at all and return its input unchanged.
        """
        raise NotImplementedError

    def _clamp(self, beta: float) -> float:
        """β held inside ``[beta_min, beta_max]``, as a plain ``float``.

        Kept a method rather than inlined per rule because the bound is the same for every control and
        this was one of the three verbatim copies the base exists to remove.
        """
        return float(min(max(beta, self.beta_min), self.beta_max))

    def next_step(
        self, base_step: ForwardStep, previous: StepReport | None, state: object
    ) -> tuple[ForwardStep, tuple[float, object]]:
        """The base step carrying a constant β, and the ``(beta, memo)`` state to carry forward.

        ``state`` is ``None`` only on the very first step of the whole march; β is :attr:`beta_start`
        there. The **first step of a refresh segment** (``previous is None`` with a carried ``state``)
        **holds** β so the ramp continues across the preconditioner refresh rather than resetting --
        without this, a march that refreshes every few steps (the common case, a drift-triggered refresh
        on a developing flow) sawtooths β back to the start each segment and the pseudo-timestep never
        actually grows. Otherwise the rule adapts it from the previous step's report.

        β rides as a :class:`~aquaflux.solve.ConstantRelaxation` on a **dynamic** leaf, so a controlled
        step differs from the base one only in that leaf's *value* and
        :func:`~aquaflux.solve.forward_march`'s jitted step stays a compilation-cache hit. ``base_step``
        must therefore carry a ``relaxation_schedule`` to replace -- these are shift-strength controls.
        """
        if state is None:  # the very first step of the whole march
            beta, memo = self.beta_start, None
        else:
            beta, memo = state
            if previous is not None:  # within a segment; a refresh boundary holds instead
                beta, memo = self._adapt(beta, previous, memo)
        controlled = eqx.tree_at(
            lambda s: s.relaxation_schedule, base_step, ConstantRelaxation(jnp.asarray(beta))
        )
        return controlled, (beta, memo)

    def carry_beta(self, state: object, beta: float) -> tuple[float, object]:
        """Seed the carried β with an externally-chosen value, keeping the memo.

        Called by :func:`~aquaflux.solve.forward_march` after a β-escalation retry, so the ramp continues
        from the escalation's discovered-safe β instead of the control's own last β. The memo is
        preserved so a rule keying on a residual ratio does not lose its reference and mis-read the next
        step as a huge reduction.
        """
        memo = state[1] if isinstance(state, tuple) else None
        return (float(beta), memo)


class DualTimeControl(ShiftStrengthControl):
    """Ramp the pseudo-timestep of a :class:`~aquaflux.solve.DualTimeStep` march by a Courant rule.

    The dual-time step's reported α is the **smallest inner line-search factor** of its backward-Euler
    timestep: α = 1 means every inner Newton step took the full length (the pseudo-timestep was
    comfortable), α < 1 that an inner step had to be clipped (the pseudo-timestep is too large). This
    control grows the pseudo-timestep (lowers β) while the inner loop is comfortable and shrinks it
    (raises β) when it clips — the classic explicit-CFL ramp, but around an *implicit* timestep the
    inner loop keeps stable, so β can be driven well below the level at which a single-step
    pseudo-transient march diverges.

    - **α ≥ :attr:`grow_above`:** the inner loop was comfortable — grow the pseudo-timestep,
      ``β ← β / grow``.
    - **α < :attr:`backoff_below`:** an inner step clipped hard — shrink it, ``β ← β * backoff``.
    - **otherwise:** hold β (a dead band, so the ramp does not chatter around the boundary).

    The well-behaved backward-Euler residual is what lets it keep growing the step as the flow develops.
    β is **carried across a preconditioner refresh** (see
    :meth:`ShiftStrengthControl.next_step`), which is what makes this the **default step control for the
    dual-time coupled march**: a refresh fires every few steps, and a non-carrying ramp would sawtooth β
    back to the start each segment and never grow the pseudo-timestep. Measured on a cold-start
    pitzDaily Reynolds-continuation ramp, the carrying ramp reaches the developed recirculation in
    **~4× fewer outer steps** than the residual-keyed :class:`ResidualRatioDualTimeControl`, which pins β
    near its start because the (row-scaled) steady residual is nearly flat while the transient develops.
    :attr:`ShiftStrengthControl.beta_min` bounds the pseudo-timestep, so the ramp does not run away; the
    shift still vanishes at the root, so the finishing solve owns the converged root and the adjoint
    either way.

    This control is **memoryless** — its memo is always ``None``, since α alone drives it.

    Attributes
    ----------
    grow : float
        Factor ``> 1`` the pseudo-timestep is grown by (β divided by) on a comfortable step (static).
    backoff : float
        Factor ``> 1`` the pseudo-timestep is shrunk by (β multiplied by) on a clipped step (static).
    grow_above, backoff_below : float
        The α thresholds bounding the grow / back-off / hold bands (static).
    """

    grow: float = eqx.field(static=True, default=1.5)
    backoff: float = eqx.field(static=True, default=2.0)
    grow_above: float = eqx.field(static=True, default=0.5)
    backoff_below: float = eqx.field(static=True, default=0.25)

    def _adapt(self, beta: float, previous: StepReport, memo: object) -> tuple[float, object]:
        alpha = float(previous.alpha)
        if alpha < self.backoff_below:  # an inner step clipped hard -> pseudo-timestep too large
            beta = beta * self.backoff
        elif alpha >= self.grow_above:  # comfortable -> grow the pseudo-timestep
            beta = beta / self.grow
        return self._clamp(beta), None


class ResidualRatioDualTimeControl(ShiftStrengthControl):
    """Ramp the dual-time pseudo-timestep by the steady-residual reduction ratio (residual-based PTC).

    The convergent form of pseudo-transient control -- switched evolution relaxation (Mulder & Van Leer
    1985), with the convergence theory of Kelley & Keyes (1998) and its differential-algebraic extension
    (Coffey, Kelley & Keyes 2003). The pseudo-timestep grows in proportion to how much the *steady*
    residual fell over the previous step and shrinks when it rose. In terms of the shift ``β`` -- which
    is inversely the pseudo-timestep -- the update is

        ``β ← β · (‖R_n‖ / ‖R_{n-1}‖)``

    so a residual drop (ratio ``< 1``) lowers ``β`` and grows the step, while a residual rise
    (ratio ``> 1``) raises ``β`` and shrinks it.

    **An alternative to the α-based :class:`DualTimeControl`, with a different failure guard.** That
    control grows the step on the inner line-search factor ``α`` alone -- a proxy for the *inner* solve's
    comfort at fixed timestep, blind to the *outer* steady residual -- and bounds runaway only with its
    ``beta_min`` floor on the pseudo-timestep. This control instead **earns** growth by residual
    reduction: a rising residual *automatically* shrinks the step. The trade measured on a cold-start
    pitzDaily Reynolds-continuation ramp is the reverse of what that guard suggests: because the
    row-scaled steady residual is nearly flat while the recirculation develops (the shifted step leaves
    ``R(φ+δ) ≈ −β d δ`` — ``β`` times how far the step travelled, the physical unsteady term, not a
    distance to the root), this control keeps ``β`` pinned near its start and the pseudo-timestep never
    grows, so it reaches the developed bubble in **~4× more outer steps** than the α-based ramp. It
    remains the choice when the steady residual *is* a reliable progress signal (a genuinely
    monotone-converging transient), where earned growth is the safer rule.

    The per-step change is clipped to ``[1 / max_change, max_change]`` so one anomalous step cannot fling
    the timestep. A hard inner-loop clip (``α < backoff_below``) forces an extra shrink regardless of the
    ratio -- an inner step that had to be clipped means the implicit step was too large this iteration,
    whatever the residual did. **Before a ratio is available** (the first adaptation of a march, whose
    memo is still ``None``) this control declines to adapt at all, so its first move is never made on a
    fabricated ratio; :class:`CflResidualDualTimeControl` deliberately differs there.

    The residual it reads is whatever measure the march steers by (``previous.residual_norm``); with the
    default row-equilibrated norm that is a fractional change per equation, so the ratio is a meaningful
    reduction factor across steps. Its memo is that residual.

    **Opt-in alternative to the default :class:`DualTimeControl`.** The finishing solve, running the
    default schedule, still owns the converged root and the adjoint.

    Attributes
    ----------
    max_change : float
        The most the pseudo-timestep may grow or shrink in one step, as a factor ``> 1`` bounding the
        residual-ratio update to ``[1 / max_change, max_change]`` (static).
    backoff : float
        Extra factor ``> 1`` the pseudo-timestep is shrunk by (β multiplied by) on a hard inner clip
        (static).
    backoff_below : float
        The α threshold below which the hard inner-clip shrink fires (static).
    """

    beta_start: float = eqx.field(static=True, default=0.5)
    max_change: float = eqx.field(static=True, default=1.3)
    backoff: float = eqx.field(static=True, default=2.0)
    backoff_below: float = eqx.field(static=True, default=0.5)

    def _adapt(self, beta: float, previous: StepReport, memo: object) -> tuple[float, object]:
        residual = float(previous.residual_norm)
        if memo is None or memo <= 0.0:  # no ratio yet: remember the residual, do not adapt on it
            return beta, residual
        change = min(max(residual / memo, 1.0 / self.max_change), self.max_change)
        beta = beta * change
        if float(previous.alpha) < self.backoff_below:  # clipped hard -> implicit step too large
            beta = beta * self.backoff
        return self._clamp(beta), residual


class CflResidualDualTimeControl(ShiftStrengthControl):
    """Grow the dual-time pseudo-timestep on the Courant signal α, but brake it on a rising residual.

    The two single-signal controls fail in **disjoint** ways, so combining them recovers the strengths of
    both. :class:`DualTimeControl` grows Δτ on the inner line-search factor α (a *local* step-health signal:
    is this Δτ small enough for the backward-Euler inner loop to take a full step?) — fast, but **blind to
    the trajectory**: it happily grows Δτ into an overshoot while the inner loop stays comfortable (α = 1),
    which then diverges unless the linear solve is near-exact. :class:`ResidualRatioDualTimeControl` grows Δτ
    on the steady-residual reduction ratio (a *global* trajectory-health signal: is the march converging or
    diverging?) — safe against overshoot, but **blind to productive development**: while the residual sits on
    its ``β × travel`` plateau (nearly flat as the slow transient develops) it cannot tell "developing" from
    "stalled", so it pins β and the pseudo-timestep stalls to a crawl.

    A step can be *locally* healthy (α = 1) yet *globally* diverging (residual rising) — the overshoot — and
    that is the one regime where both signals are needed at once. This control therefore grows Δτ **only when
    both are comfortable** and shrinks it when **either** wall is hit ("grow until the first wall"):

    - **α < :attr:`backoff_below` (local wall) or ratio > :attr:`rise_ratio` (global wall):** shrink,
      ``β ← β · backoff``. The α term catches a Δτ too large for the inner loop; the ratio term is the
      overshoot governor α lacks — it fires when the steady residual rises even though α is still 1.
    - **α ≥ :attr:`grow_above` and ratio ≤ :attr:`hold_ratio`:** the step is comfortable *and* the
      trajectory is not diverging — grow, ``β ← β / grow``. Because the plateau ratio ≈ 1 satisfies
      ``ratio ≤ hold_ratio``, this grows on α through the flat-residual development where the residual-only
      rule stalls.
    - otherwise (``hold_ratio < ratio ≤ rise_ratio``): hold. The band between the two ratio thresholds
      keeps the mildly-noisy plateau from oscillating between grow and brake.

    So on a case with no dangerous overshoot the residual never crosses ``rise_ratio`` and it grows on α like
    :class:`DualTimeControl` (its speed); on a case with a sharp overshoot the ratio term brakes right at the
    excursion where α is blind (:class:`ResidualRatioDualTimeControl`'s safety) — without either control's
    blind spot. **At infinite ratio thresholds it IS `DualTimeControl`**, since neither ratio clause can then
    fire; a test pins that reduction, so the relationship is checked rather than asserted in prose.

    Before a residual ratio is available (the first adaptation, whose memo is ``None``) the ratio defaults to
    ``1`` so α alone drives that step -- deliberately unlike :class:`ResidualRatioDualTimeControl`, which
    declines to adapt at all until it has a real ratio, because there α is not a growth signal on its own.

    Attributes
    ----------
    grow : float
        Factor ``> 1`` the pseudo-timestep is grown by (β divided by) on a comfortable, non-diverging step
        (static).
    backoff : float
        Factor ``> 1`` the pseudo-timestep is shrunk by (β multiplied by) on either wall (static).
    grow_above, backoff_below : float
        The α thresholds: at or above ``grow_above`` the inner step is comfortable; below ``backoff_below``
        it clipped hard (static).
    hold_ratio, rise_ratio : float
        The steady-residual ratio thresholds, with ``hold_ratio < rise_ratio``: growth is allowed only when
        ``ratio ≤ hold_ratio`` (residual flat or falling), braking fires when ``ratio > rise_ratio`` (residual
        rising), and the band between holds β (static).
    """

    grow: float = eqx.field(static=True, default=1.5)
    backoff: float = eqx.field(static=True, default=2.0)
    grow_above: float = eqx.field(static=True, default=0.5)
    backoff_below: float = eqx.field(static=True, default=0.25)
    hold_ratio: float = eqx.field(static=True, default=1.05)
    rise_ratio: float = eqx.field(static=True, default=1.10)

    def _adapt(self, beta: float, previous: StepReport, memo: object) -> tuple[float, object]:
        residual = float(previous.residual_norm)
        ratio = residual / memo if memo is not None and memo > 0.0 else 1.0
        alpha = float(previous.alpha)
        if alpha < self.backoff_below or ratio > self.rise_ratio:  # either wall -> shrink
            beta = beta * self.backoff
        elif alpha >= self.grow_above and ratio <= self.hold_ratio:  # both comfortable -> grow
            beta = beta / self.grow
        return self._clamp(beta), residual


def default_dual_time_control(
    step_control: StepControl | None, observing: bool, continuation: ForwardStep
) -> StepControl | None:
    """The step control for an observed march: the caller's, or the default Courant ramp for a dual-time
    march that was given none.

    It lives here, beside the controls it chooses between, rather than in the turbulence driver that
    calls it. That is where it was: `StepControl` was declared in `march.py` with no implementations
    there, so `step_control.py` had to import `march` -- which forbade the reverse, so a rule about two
    `solve/` objects could not be expressed in `solve/` at all. With the contract in
    :mod:`~aquaflux.solve.forward_step` that constraint is gone and the rule comes home.

    A **dual-time** march (a :class:`~aquaflux.solve.DualTimeStep`, whose reported ``alpha`` is the
    backward-Euler inner-loop comfort a Courant ramp reads) that is **already observing** (a
    a refresh or an observer set ``observing``) but was handed **no** ``step_control`` defaults to
    :class:`~aquaflux.solve.DualTimeControl`. That ramp grows the pseudo-timestep while the inner loop
    stays comfortable, reaching a developed recirculation in far fewer outer steps than the residual-keyed
    schedule (which pins ``beta`` because the row-scaled steady residual is nearly flat while the flow
    develops). ``step_control`` is returned **unchanged** for a single-step march, a caller-supplied
    control, or a march that is not observing — so the default is injected only where a control actually
    runs, and injecting it never turns observation on (which would wrongly make the differentiable
    single-stage solve raise the forward-only guard).

    Parameters
    ----------
    step_control : StepControl or None
        The caller-supplied control (``None`` if none was given).
    observing : bool
        Whether the march runs the observed eager path (a refresh or observer is active).
    continuation : ForwardStep
        The globalization step the march applies.

    Returns
    -------
    StepControl or None
        ``DualTimeControl()`` when defaulting applies; ``step_control`` otherwise.
    """
    if step_control is None and observing and isinstance(continuation, DualTimeStep):
        return DualTimeControl()
    return step_control
