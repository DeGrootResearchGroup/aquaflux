"""Feedback step controls for the eager march (experimental, forward-only).

A :class:`~aquaflux.solve.StepControl` reshapes the forward step each iteration from the previous
step's outcome. Two members drive the pseudo-transient shift strength β from the line-search factor α:

* :class:`AlphaTargetingControl` drives the *single-step* shift toward the boundary where the full
  shifted step just stops being clipped — the measured efficiency optimum for a stiff coupled solve,
  which the default switched-evolution-relaxation schedule misses by ramping β the wrong way.
* :class:`DualTimeControl` ramps the *dual-time* pseudo-timestep by a Courant rule (grow it while the
  backward-Euler inner loop is comfortable, shrink it when it clips), which lets β be driven well below
  the single-step divergence floor because the inner loop keeps the larger implicit step stable.

**Both are experimental, opt-in, never a default.** Each was measured to help a stiff coupled RANS
march but neither converges to a root by itself (they plateau short), and their numeric gains are
hand-set placeholders, not calibrated. They are forward-only accelerators on the eager march; the
finishing solve (running the default schedule) still owns the converged root and the adjoint. Do not
promote either to a default until it converges standalone.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .implicit import ForwardStep
from .march import StepReport
from .relaxation import ConstantRelaxation


class AlphaTargetingControl(eqx.Module):
    """Drive the shift strength β toward the line-search-factor α = 1 boundary.

    The line-search factor α (fraction of the shifted step the backtracking search keeps) rises with
    β and reaches 1 exactly at the efficiency-optimal shift: below it the full step overshoots and is
    clipped (α < 1, wasteful); at it the full step is taken. So the target is the α = 1 edge.

    - **α < 1 (clipped):** β is too weak — raise it. Since α is roughly proportional to β in the
      clipped regime, ``β ← β / α`` lands near the boundary, capped at :attr:`growth_cap` per step so
      a tiny α cannot fling β to the ceiling.
    - **α = 1 (full step):** β is at or above the boundary — ease it down gently (÷ :attr:`ease`) to
      probe toward a larger productive step. As the state stiffens the boundary rises, re-clips the
      eased β, and the raise fires again — so the control hunts the moving boundary from both sides.

    **Known ceiling (see the module docstring):** the ``β/α`` raise overshoots *past* the boundary
    into over-damping (α saturates at 1 above it, giving no gradient), so the control plateaus rather
    than converging. This is the open item; the class is shipped as the validated *direction*, not a
    finished solver.

    Attributes
    ----------
    beta_start : float
        The shift strength for the first step (static).
    growth_cap : float
        The most β may grow in one step (static), bounding the ``β/α`` raise.
    ease : float
        The factor β is divided by when the full step is taken (static); gentle, so the equilibrium
        hugs just below the α = 1 boundary.
    beta_min, beta_max : float
        Clamps on β (static).
    """

    beta_start: float = eqx.field(static=True, default=2.0)
    growth_cap: float = eqx.field(static=True, default=3.0)
    ease: float = eqx.field(static=True, default=1.1)
    beta_min: float = eqx.field(static=True, default=0.1)
    beta_max: float = eqx.field(static=True, default=50.0)

    def _adapt(self, beta: float, alpha: float) -> float:
        if alpha < 0.999:  # the full step was clipped -> β too weak
            beta = min(beta / max(alpha, 0.05), self.growth_cap * beta)
        else:  # the full step descended -> probe a little lower
            beta = beta / self.ease
        return float(min(max(beta, self.beta_min), self.beta_max))

    def next_step(
        self, base_step: ForwardStep, previous: StepReport | None, state: object
    ) -> tuple[ForwardStep, float]:
        """The base step with a constant shift strength β, and the new β to carry.

        ``state`` is the previous step's β (``None`` on the first step). β for the step about to run
        is derived from the *previous* step's α; the first step uses :attr:`beta_start`. ``base_step``
        must be a :class:`~aquaflux.solve.PseudoTransientStep` (it is the only step with a
        ``relaxation_schedule`` to replace) — α-targeting is a shift-strength control.
        """
        beta = (
            self.beta_start
            if state is None or previous is None
            else self._adapt(state, previous.alpha)
        )
        controlled = eqx.tree_at(
            lambda s: s.relaxation_schedule, base_step, ConstantRelaxation(jnp.asarray(beta))
        )
        return controlled, beta


class DualTimeControl(eqx.Module):
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

    Unlike :class:`AlphaTargetingControl` (which drives the *single-step* shift toward the α = 1
    boundary on the steady residual, and plateaus there), this drives the *dual-time* pseudo-timestep,
    and the well-behaved backward-Euler residual is what lets it keep growing the step as the flow
    develops. **Experimental, opt-in, never a default** — its numeric gains are hand-set placeholders
    (calibration is the open item, tied to the cost of the low-β linear solves); the finishing solve,
    running the default schedule, still owns the converged root and the adjoint.

    Attributes
    ----------
    beta_start : float
        The pseudo-transient shift strength for the first step (static).
    grow : float
        Factor ``> 1`` the pseudo-timestep is grown by (β divided by) on a comfortable step (static).
    backoff : float
        Factor ``> 1`` the pseudo-timestep is shrunk by (β multiplied by) on a clipped step (static).
    grow_above, backoff_below : float
        The α thresholds bounding the grow / back-off / hold bands (static).
    beta_min, beta_max : float
        Clamps on β (static). ``beta_min`` bounds how large the pseudo-timestep may grow.
    """

    beta_start: float = eqx.field(static=True, default=2.0)
    grow: float = eqx.field(static=True, default=1.5)
    backoff: float = eqx.field(static=True, default=2.0)
    grow_above: float = eqx.field(static=True, default=0.5)
    backoff_below: float = eqx.field(static=True, default=0.25)
    beta_min: float = eqx.field(static=True, default=0.02)
    beta_max: float = eqx.field(static=True, default=4.0)

    def _adapt(self, beta: float, alpha: float) -> float:
        if alpha < self.backoff_below:  # an inner step clipped hard -> pseudo-timestep too large
            beta = beta * self.backoff
        elif alpha >= self.grow_above:  # comfortable -> grow the pseudo-timestep
            beta = beta / self.grow
        return float(min(max(beta, self.beta_min), self.beta_max))

    def next_step(
        self, base_step: ForwardStep, previous: StepReport | None, state: object
    ) -> tuple[ForwardStep, float]:
        """The base dual-time step with a constant β, and the new β to carry.

        ``state`` is the previous step's β (``None`` on the first step); β for the step about to run is
        derived from the *previous* step's α (its smallest inner line-search factor). ``base_step`` must
        be a :class:`~aquaflux.solve.DualTimeStep` (the step whose reported α is an inner-loop factor and
        whose ``relaxation_schedule`` this replaces).
        """
        beta = (
            self.beta_start
            if state is None or previous is None
            else self._adapt(state, previous.alpha)
        )
        controlled = eqx.tree_at(
            lambda s: s.relaxation_schedule, base_step, ConstantRelaxation(jnp.asarray(beta))
        )
        return controlled, beta


class ResidualRatioDualTimeControl(eqx.Module):
    """Ramp the dual-time pseudo-timestep by the steady-residual reduction ratio (residual-based PTC).

    The convergent form of pseudo-transient control -- switched evolution relaxation (Mulder & Van Leer
    1985), with the convergence theory of Kelley & Keyes (1998) and its differential-algebraic extension
    (Coffey, Kelley & Keyes 2003). The pseudo-timestep grows in proportion to how much the *steady*
    residual fell over the previous step and shrinks when it rose. In terms of the shift ``β`` -- which
    is inversely the pseudo-timestep -- the update is

        ``β ← β · (‖R_n‖ / ‖R_{n-1}‖)``

    so a residual drop (ratio ``< 1``) lowers ``β`` and grows the step, while a residual rise
    (ratio ``> 1``) raises ``β`` and shrinks it.

    **This is what prevents the runaway of :class:`DualTimeControl`.** That control grows the step on the
    inner line-search factor ``α`` alone -- a proxy for the *inner* solve's comfort at fixed timestep,
    blind to the *outer* steady residual -- so when the flow over-develops (each inner solve stays
    comfortable, ``α = 1``, while the steady residual climbs) it keeps growing the step and drives the
    transient away from the steady manifold. Here growth is **earned** by residual reduction: a rising
    residual *automatically* shrinks the step, so the ramp cannot run away.

    The per-step change is clipped to ``[1 / max_change, max_change]`` so one anomalous step cannot fling
    the timestep, and ``β`` is clamped to ``[beta_min, beta_max]``. A hard inner-loop clip
    (``α < backoff_below``) forces an extra shrink regardless of the ratio -- an inner step that had to be
    clipped means the implicit step was too large this iteration, whatever the residual did. The shift is
    **carried across a preconditioner refresh** (the ramp is continuous) rather than resetting to
    ``beta_start`` at each segment as the α-based control does.

    The residual it reads is whatever measure the march steers by (``previous.residual_norm``); with the
    default row-equilibrated norm that is a fractional change per equation, so the ratio is a meaningful
    reduction factor across steps.

    **Experimental, opt-in, never a default.** The finishing solve, running the default schedule, still
    owns the converged root and the adjoint.

    Attributes
    ----------
    beta_start : float
        The pseudo-transient shift strength for the first step (static).
    max_change : float
        The most the pseudo-timestep may grow or shrink in one step, as a factor ``> 1`` bounding the
        residual-ratio update to ``[1 / max_change, max_change]`` (static).
    backoff : float
        Extra factor ``> 1`` the pseudo-timestep is shrunk by (β multiplied by) on a hard inner clip
        (static).
    backoff_below : float
        The α threshold below which the hard inner-clip shrink fires (static).
    beta_min, beta_max : float
        Clamps on β (static). ``beta_min`` bounds how large the pseudo-timestep may grow.
    """

    beta_start: float = eqx.field(static=True, default=0.5)
    max_change: float = eqx.field(static=True, default=1.3)
    backoff: float = eqx.field(static=True, default=2.0)
    backoff_below: float = eqx.field(static=True, default=0.5)
    beta_min: float = eqx.field(static=True, default=0.02)
    beta_max: float = eqx.field(static=True, default=4.0)

    def _adapt(self, beta: float, alpha: float, ratio: float) -> float:
        change = min(max(ratio, 1.0 / self.max_change), self.max_change)
        beta = beta * change
        if alpha < self.backoff_below:  # an inner step clipped hard -> implicit step too large
            beta = beta * self.backoff
        return float(min(max(beta, self.beta_min), self.beta_max))

    def next_step(
        self, base_step: ForwardStep, previous: StepReport | None, state: object
    ) -> tuple[ForwardStep, tuple[float, float | None]]:
        """The base dual-time step with a constant β, and the ``(β, ‖R‖)`` state to carry.

        ``state`` is the carried ``(β, previous residual norm)`` (``None`` on the very first step). β is
        updated from the ratio of the *previous* step's residual to the one before it; the first step of
        a refresh segment (``previous is None`` with a carried ``state``) holds β so the ramp continues
        across the refresh rather than resetting. ``base_step`` must be a
        :class:`~aquaflux.solve.DualTimeStep`.
        """
        if state is None:
            beta, prev_residual = self.beta_start, None
        else:
            beta, prev_residual = state
            if previous is not None:
                if prev_residual is not None and prev_residual > 0.0:
                    ratio = float(previous.residual_norm) / prev_residual
                    beta = self._adapt(beta, float(previous.alpha), ratio)
                prev_residual = float(previous.residual_norm)
        controlled = eqx.tree_at(
            lambda s: s.relaxation_schedule, base_step, ConstantRelaxation(jnp.asarray(beta))
        )
        return controlled, (beta, prev_residual)
