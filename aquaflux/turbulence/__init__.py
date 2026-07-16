"""Turbulence models: RANS closures that supply an eddy viscosity to the flow solve.

The k-omega SST constants and the closed-form quantities they define -- the blending functions, the
constant blend, and the eddy viscosity -- live in :class:`~aquaflux.turbulence.sst.SSTModel`; the
strain-rate magnitude they consume in
:func:`~aquaflux.turbulence.strain.strain_rate_magnitude`; and the k and omega volumetric source
terms (production, destruction, cross-diffusion) as :mod:`~aquaflux.turbulence.sources` operators.
"""

from __future__ import annotations

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
    "strain_rate_magnitude",
]
