"""Registers the coupled RANS problem with ``aquaflux.initialization.hybrid_initialize``.

Its own module because the initializer it registers (:func:`~aquaflux.turbulence.sst_initial_fields`) is
imported by :mod:`~aquaflux.turbulence.coupled`, so the registration -- which needs the class from there --
cannot live in either without a cycle.
"""

from __future__ import annotations

import jax.numpy as jnp

from aquaflux.initialization import hybrid_initialize

from .coupled import CoupledRANS
from .initialization import sst_initial_fields


@hybrid_initialize.register(CoupledRANS)
def _initialize_coupled_rans(
    coupled: CoupledRANS, **settings: object
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """The coupled solve's hybrid start: potential flow plus the closure's fields, physical ``(flow, k, omega)``.

    The settings are those of :func:`~aquaflux.turbulence.sst_initial_fields` (``k_floor``,
    ``omega_floor``, ``length_scale_factor``).
    """
    return sst_initial_fields(coupled.momentum, coupled.turbulence, **settings)
