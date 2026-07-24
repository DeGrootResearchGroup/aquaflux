"""Feedback step controls for the eager march (experimental, forward-only).

A :class:`~aquaflux.solve.StepControl` reshapes the forward step each iteration from the previous
step's outcome. The one member here, :class:`AlphaTargetingControl`, drives the pseudo-transient
shift strength β toward the boundary where the full shifted step just stops being clipped by the
line search — the measured efficiency optimum for a stiff coupled solve, which the default
switched-evolution-relaxation schedule misses by ramping β the wrong way.

**Experimental, opt-in, never a default.** The α-targeting control was measured to strictly beat SER
on a stiff coupled RANS march (reaching a given residual far faster and deeper) *when paired with a
mid-march preconditioner refresh*, but it does **not** by itself converge to a root — it plateaus
short — and its numeric gains are hand-set placeholders, not calibrated. It is a forward-only
accelerator on the eager march; the finishing solve (running the default schedule) still owns the
converged root and the adjoint. Do not promote it to a default until it converges standalone.
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
