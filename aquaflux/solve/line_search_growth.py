"""How much the residual may grow and still be accepted: the line search's monotonicity schedule.

A pseudo-transient step is a **pseudo-time march**, not a descent method. The steady residual is
legitimately non-monotone along a transient path -- a growing recirculation raises the ``omega``
imbalance long before it falls -- so a *strict-descent* line search vetoes physically correct steps.
Measured on a separating RANS case: the vetoed step was reducing the momentum and ``k`` residuals
monotonically and growing the bubble, while ``omega`` (which is ~100 % of the Euclidean norm) rose;
the search then returned its smallest rung, a near-null step, which the divergence guard accepted as
finite. The march reported "accepted" every step and stood still.

Near the root the monotone test is wanted again: that is what delivers the terminal quadratic phase,
and admitting growth there would let the iterate wander off a root it has already found. So the
right object is a **schedule** that relaxes monotonicity far from the root and restores it in the
basin -- the direct analogue of the pseudo-transient shift itself, which damps hard far away and
vanishes at the fixed point.

Like :class:`~aquaflux.solve.RelaxationSchedule`, a growth schedule is **memoryless**: it maps the
current and reference residual measures to a factor, with no state carried across steps. That is what
lets it live on the differentiable traced path rather than on the eager march.
"""

from __future__ import annotations

from typing import Protocol

import equinox as eqx
import jax.numpy as jnp


class LineSearchGrowth(Protocol):
    """The rule setting how far the residual may rise and still be accepted.

    Structural interface only (a ``Protocol``). An implementation is a pure function of the two
    residual measures, returning a multiplier ``>= 1`` applied to the reference norm: ``1`` is the
    classical monotone (strict-descent) search, and larger values admit a controlled increase.
    """

    def growth(self, residual_norm: jnp.ndarray, residual_norm_0: jnp.ndarray) -> jnp.ndarray:
        """The admissible growth factor for a step at ``residual_norm``.

        Parameters
        ----------
        residual_norm : jnp.ndarray
            The residual measure at the current iterate, a scalar.
        residual_norm_0 : jnp.ndarray
            The reference residual measure the march is judged against, a scalar.

        Returns
        -------
        jnp.ndarray
            The multiplier on the reference norm (a scalar, ``>= 1``).
        """


class MonotoneLineSearch(eqx.Module):
    """Strict descent: the residual must fall. The default, and the classical behaviour.

    Correct near a root -- it is what gives the terminal quadratic phase -- and the wrong test far
    from one on a pseudo-time march (see the module docstring).
    """

    def growth(self, residual_norm: jnp.ndarray, residual_norm_0: jnp.ndarray) -> jnp.ndarray:
        del residual_norm, residual_norm_0
        return jnp.asarray(1.0)


class RelaxedFarFromRoot(eqx.Module):
    """Admit growth while far from the root, restore strict descent inside the basin.

    ``growth = max_growth`` while ``||R|| / ||R_0|| > basin``, easing to ``1`` below it. The transition
    is smooth in the residual ratio rather than a switch, so the accepted step length does not jump as
    the march crosses the threshold.

    Attributes
    ----------
    max_growth : float
        The largest admissible residual increase far from the root, as a multiple (static). ``1``
        reduces this to :class:`MonotoneLineSearch`; ``2`` lets the residual double on a step that the
        pseudo-time march considers progress.
    basin : float
        The residual ratio below which strict descent is restored (static). Choose it as the ratio at
        which the march is expected to be in the quadratic basin; above it the transient path is still
        being traversed and monotonicity is not a meaningful requirement.
    """

    max_growth: float = eqx.field(static=True, default=2.0)
    basin: float = eqx.field(static=True, default=1e-2)

    def growth(self, residual_norm: jnp.ndarray, residual_norm_0: jnp.ndarray) -> jnp.ndarray:
        ratio = residual_norm / residual_norm_0
        # Smoothly interpolate 1 -> max_growth as the ratio rises through `basin`; clamped so the
        # factor is never below one (which would be *stricter* than descent) nor above max_growth.
        excess = jnp.clip(jnp.log10(jnp.maximum(ratio, 1e-300) / self.basin), 0.0, 1.0)
        return 1.0 + (self.max_growth - 1.0) * excess
