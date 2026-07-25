"""How the pseudo-transient shift's *spatial* distribution is built from a cell's operator parts.

Pseudo-transient continuation adds ``beta d(phi)`` to the Newton Jacobian diagonal (see
:mod:`~aquaflux.solve.continuation`). The strength ``beta`` (an injected
:class:`~aquaflux.solve.RelaxationSchedule`) sets *how much* damping; this module sets the per-cell
**base diagonal** ``d`` -- the local pseudo-time reciprocal ``V / dt`` -- from the transport operator's
own coefficients, and it is a first-class swappable strategy because the spatial shape of ``d``
materially changes the march.

The base diagonal is assembled from two per-cell buckets a block already computes:

- ``convective`` -- the first-order-upwind outflow sum ``Sum_f max(mdot_f, 0)`` (equivalently
  ``1/2 Sum_f |mdot_f|`` on a divergence-free flux). This is the local **convective** Courant scale
  ``|u| / dx`` in flux form -- the same quantity OpenFOAM's ``Co = 1/2 dt Sum_f|phi_f| / V`` and
  Fluent's local pseudo-time step are built on.
- ``dissipative`` -- everything else on the diagonal: the diffusion stiffness, plus (for a scalar) the
  reaction and boundary diagonal, plus (for momentum) any transient ``V/dt``.

A :class:`ShiftBasis` combines the two into ``d``. The default combines them one-to-one, which is the
full operator diagonal (the momentum ``a_P`` / its scalar analogue): because that equals the operator
diagonal, ``beta d`` is a spatially-*uniform* under-relaxation (every cell relaxed by ``1/(1+beta)``).
Dropping the dissipative bucket instead gives a genuine **local convective time step** -- the
non-uniform per-cell ``dt`` a Courant condition implies, which advances slow recirculation/near-wall
cells faster than the fast shear layer, something the uniform relaxation cannot do.
"""

from __future__ import annotations

from typing import Protocol

import equinox as eqx
import jax.numpy as jnp


class ShiftBasis(Protocol):
    """Combine a cell's convective and dissipative diagonal buckets into the base shift ``d``.

    Structural interface only (a ``Protocol``). A basis is a **pure, per-cell** map from the two
    non-negative operator-diagonal buckets to the base pseudo-time diagonal ``d`` the engine scales by
    ``beta``. It carries no state and reads nothing global, so it traces inside the Newton loop exactly
    like the shift itself.
    """

    def local_diagonal(self, convective: jnp.ndarray, dissipative: jnp.ndarray) -> jnp.ndarray:
        """The base shift diagonal ``d`` per cell from the two operator-diagonal buckets.

        Parameters
        ----------
        convective : jnp.ndarray
            The first-order-upwind convective outflow sum per cell, shape ``(n_cells,)`` (``>= 0``).
        dissipative : jnp.ndarray
            The rest of the operator diagonal per cell -- diffusion stiffness plus any
            reaction/boundary/transient contribution, shape ``(n_cells,)`` (``>= 0``).

        Returns
        -------
        jnp.ndarray
            The non-negative base shift diagonal ``d``, shape ``(n_cells,)``.
        """


class LocalCourantBasis(eqx.Module):
    """``d = convective + w * dissipative`` -- a local Courant time step with a tunable dissipative weight.

    The base shift is the local convective Courant scale plus a fraction ``w`` of the dissipative
    stiffness. The two endpoints are the useful ones:

    - ``w = 1`` (default) -- ``d`` is the **full operator diagonal** (the momentum ``a_P`` / its scalar
      analogue). Since it equals the operator diagonal, ``beta d`` is spatially-uniform under-relaxation
      (relaxation factor ``1/(1+beta)`` in every cell). This is byte-for-byte the historical shift, so
      it is the default and leaves every existing march unchanged.
    - ``w = 0`` -- ``d`` is the **pure convective** local time step, ``Sum_f max(mdot_f, 0)``. The
      per-cell pseudo-time ``dt = Co* V / d`` then follows the local convective condition, so a
      low-velocity recirculation or near-wall cell is damped far less than a fast shear-layer cell --
      the non-uniform behaviour a global under-relaxation cannot reproduce (the OpenFOAM / Fluent local
      time-step construction).

    Intermediate ``w`` down-weights the dissipative contribution without dropping it (e.g. to keep some
    near-wall damping while freeing the convective interior).

    Attributes
    ----------
    dissipative_weight : float
        The weight ``w >= 0`` on the dissipative bucket (static). ``1`` is the full operator diagonal
        (uniform relaxation); ``0`` is the pure convective local time step.
    """

    dissipative_weight: float = eqx.field(static=True, default=1.0)

    def local_diagonal(self, convective: jnp.ndarray, dissipative: jnp.ndarray) -> jnp.ndarray:
        return convective + self.dissipative_weight * dissipative
