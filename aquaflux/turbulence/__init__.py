"""Turbulence models: RANS closures that supply an eddy viscosity to the flow solve.

The k-omega SST constants and the closed-form quantities they define -- the blending functions, the
constant blend, and the eddy viscosity -- live in :class:`~aquaflux.turbulence.sst.SSTModel`; the
strain-rate magnitude they consume in
:func:`~aquaflux.turbulence.strain.strain_rate_magnitude`; the k and omega volumetric source terms
(production, destruction, cross-diffusion) as :mod:`~aquaflux.turbulence.sources` operators; and the
boundary values (near-wall omega, inlet k and omega) in :mod:`~aquaflux.turbulence.boundary`.
"""

from __future__ import annotations

from .boundary import inlet_k, inlet_omega, omega_wall_value
from .sources import (
    KDestruction,
    KProduction,
    OmegaCrossDiffusion,
    OmegaDestruction,
    OmegaProduction,
)
from .sst import SSTModel
from .strain import strain_rate_magnitude

__all__ = [
    "KDestruction",
    "KProduction",
    "OmegaCrossDiffusion",
    "OmegaDestruction",
    "OmegaProduction",
    "SSTModel",
    "inlet_k",
    "inlet_omega",
    "omega_wall_value",
    "strain_rate_magnitude",
]
