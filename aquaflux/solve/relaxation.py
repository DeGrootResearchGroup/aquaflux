"""The pseudo-transient damping schedule: how the shift strength beta is set each step.

Pseudo-transient continuation adds a diagonal shift ``beta * d(phi)`` to the Newton Jacobian to
damp each step far from the root, easing it to zero at the fixed point (where the shift vanishes and
the undamped Newton step, with its quadratic rate, is recovered). *How* beta is chosen each step is a
first-class, swappable strategy -- the analogue of the injected :class:`~aquaflux.solve.ResidualNorm`
-- because the choice materially changes the march.

The default is switched-evolution-relaxation (SER): ``beta = beta0 (||R||/||R0||)^p``, strong damping
while the residual is large and none as it vanishes. SER is **memoryless** -- beta depends only on the
current residual norm and the reference norm -- which is what lets it live on the differentiable Newton
path (inside the traced loop and the ``custom_vjp``). A schedule that needs history or step feedback
(e.g. the line-search factor) is a *forward-only* concern and belongs on the eager march instead (see
:class:`~aquaflux.solve.StepControl`), not here.
"""

from __future__ import annotations

from typing import Protocol

import equinox as eqx
import jax.numpy as jnp


class RelaxationSchedule(Protocol):
    """The rule that sets the pseudo-transient shift strength beta from the residual norms.

    Structural interface only (a ``Protocol``). A schedule is **memoryless**: it maps the current
    residual norm and the march's reference norm to a shift strength, with no state carried across
    steps. That is what keeps it usable on the differentiable path -- it is a pure function of two
    in-scope scalars, so it traces inside the Newton loop and never reaches the ``custom_vjp`` primal.
    A stateful or feedback-driven damping rule is not a ``RelaxationSchedule``; it is a
    :class:`~aquaflux.solve.StepControl` on the eager march.
    """

    def relaxation(self, residual_norm: jnp.ndarray, residual_norm_0: jnp.ndarray) -> jnp.ndarray:
        """The shift strength beta for a step whose residual norm is ``residual_norm``.

        Parameters
        ----------
        residual_norm : jnp.ndarray
            The residual measure at the current iterate, a scalar.
        residual_norm_0 : jnp.ndarray
            The reference residual measure the ramp is relative to, a scalar.

        Returns
        -------
        jnp.ndarray
            The shift strength ``beta`` (a scalar), scaling the base pseudo-time diagonal.
        """


class SwitchedEvolutionRelaxation(eqx.Module):
    """Switched-evolution-relaxation (SER): ``beta = max(beta_floor, beta0 (||R||/||R0||)^p)``.

    Strong damping while the residual is large, easing as it vanishes -- ``beta -> 0`` at the root
    recovers the undamped Newton step and its terminal quadratic rate. The default schedule.

    Attributes
    ----------
    beta0 : float
        *Initial* damping strength ``beta0`` (static) -- the shift a step at ``||R|| = ||R0||`` uses.
        With the pseudo-transient step's escalation, ``beta0`` is a starting guess, not a per-case
        knob: too small is recovered by escalation, too large only costs a slower march.
    exponent : float
        The ramp exponent ``p`` (static). ``1`` ramps the shift linearly with the residual norm.
    beta_floor : float
        A lower bound on ``beta`` (static, default ``0`` = no floor). It keeps the shifted forward
        solve out of the ill-conditioned low-beta regime: the *unshifted* coupled saddle Jacobian
        (``beta -> 0`` far from the root) is severely ill-conditioned, so a diagonally-shifted GMRES
        that lets ``beta`` ramp to zero stagnates and burns many matrix-vector products per step.
        Holding ``beta >= beta_floor`` keeps each linear solve cheap. **Correctness-safe:** the shift
        scales the correction, which vanishes at the fixed point, so a non-zero floor never moves the
        converged root -- it only damps the path (roughly linear terminal steps rather than quadratic).
        Off by default pending further evaluation (early end-to-end measurements were a wash -- the
        cheaper late solves cancelled the extra Newton steps).

    Notes
    -----
    On stiff coupled RANS this schedule was measured to run the *wrong* way -- the efficiency-optimal
    beta rises as the residual falls while SER lowers it, so the march grinds. That does not make SER
    wrong in general (it is the correct memoryless default and the only schedule on the differentiable
    path); the alternative is a forward-only :class:`~aquaflux.solve.StepControl` on the eager march.
    """

    beta0: float = eqx.field(static=True, default=2.0)
    exponent: float = eqx.field(static=True, default=1.0)
    beta_floor: float = eqx.field(static=True, default=0.0)

    def relaxation(self, residual_norm: jnp.ndarray, residual_norm_0: jnp.ndarray) -> jnp.ndarray:
        return jnp.maximum(
            self.beta_floor, self.beta0 * (residual_norm / residual_norm_0) ** self.exponent
        )


class ConstantRelaxation(eqx.Module):
    """A fixed shift strength beta, independent of the residual -- the seam an external controller sets.

    ``beta`` is a **dynamic** leaf (a 0-d array, not a static field), so a caller that varies it per
    step -- e.g. a forward-only :class:`~aquaflux.solve.StepControl` driving beta toward its own target
    -- does so as a ``filter_jit`` cache hit rather than a recompile. (Same reason the multigrid
    ``lam_max`` is a traced 0-d array, not a Python float.)

    Attributes
    ----------
    beta : jnp.ndarray
        The constant shift strength, a 0-d array.
    """

    beta: jnp.ndarray

    def relaxation(self, residual_norm: jnp.ndarray, residual_norm_0: jnp.ndarray) -> jnp.ndarray:
        # The norms are ignored: this schedule holds beta at whatever the controller last set.
        del residual_norm, residual_norm_0
        return self.beta
