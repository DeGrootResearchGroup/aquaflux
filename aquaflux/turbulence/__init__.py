"""Turbulence models: RANS closures that supply an eddy viscosity to the flow solve.

The k-omega SST constants and the closed-form quantities they define -- the blending functions, the
constant blend, and the eddy viscosity -- live in :class:`~aquaflux.turbulence.sst.SSTModel`; the
strain-rate magnitude they consume in
:func:`~aquaflux.turbulence.strain.strain_rate_magnitude`; the k and omega volumetric source terms
(production, destruction, cross-diffusion) as :mod:`~aquaflux.turbulence.sources` operators; and the
boundary values (near-wall omega, inlet k and omega) in :mod:`~aquaflux.turbulence.boundary`; and
the assembly of the k and omega transport equations in :mod:`~aquaflux.turbulence.transport`; and
the segregated outer loop coupling the flow and turbulence solves in
:func:`~aquaflux.turbulence.driver.solve_segregated`.
"""

from __future__ import annotations

from .boundary import inlet_k, inlet_omega, omega_wall_value
from .driver import solve_segregated
from .sources import (
    KDestruction,
    KProduction,
    OmegaCrossDiffusion,
    OmegaDestruction,
    OmegaProduction,
)
from .sst import SSTModel
from .strain import strain_rate_magnitude
from .transport import SSTClosureFields, SSTTurbulence

__all__ = [
    "KDestruction",
    "KProduction",
    "OmegaCrossDiffusion",
    "OmegaDestruction",
    "OmegaProduction",
    "SSTClosureFields",
    "SSTModel",
    "SSTTurbulence",
    "inlet_k",
    "inlet_omega",
    "omega_wall_value",
    "solve_segregated",
    "strain_rate_magnitude",
]
