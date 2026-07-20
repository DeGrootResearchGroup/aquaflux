"""Turbulence models: RANS closures that supply an eddy viscosity to the flow solve.

The k-omega SST constants and the closed-form quantities they define -- the blending functions, the
constant blend, and the eddy viscosity -- live in :class:`~aquaflux.turbulence.sst.SSTModel`; the
strain-rate magnitude they consume in
:func:`~aquaflux.turbulence.strain.strain_rate_magnitude`; the k and omega volumetric source terms
(production, destruction, cross-diffusion) as :mod:`~aquaflux.turbulence.sources` operators; and the
boundary values (near-wall omega, inlet k and omega) in :mod:`~aquaflux.turbulence.boundary`; and
the assembly of the k and omega transport equations in :mod:`~aquaflux.turbulence.transport`; and
the segregated outer loop coupling the flow and turbulence solves in
:func:`~aquaflux.turbulence.driver.solve_segregated`; and the monolithic coupled residual
``R(u, p, k, omega)`` and its single Newton solve in :mod:`~aquaflux.turbulence.coupled`.
"""

from __future__ import annotations

from .boundary import equilibrium_k, inlet_k, inlet_omega, omega_wall_value
from .continuation import ScalarShiftPolicy, scalar_pseudo_transient_solve
from .coupled import (
    CoupledRANS,
    CoupledRANSLayout,
    CoupledShiftPolicy,
    coupled_continuation,
    solve_coupled,
)
from .driver import bulk_velocity, solve_segregated
from .initialization import hybrid_initialize
from .preconditioner import (
    AirAmgPreconditioner,
    ConvectionAmgPreconditioner,
    ScalarTransportPreconditioner,
    scalar_transport_preconditioner,
    scalar_transport_shift_diagonal,
)
from .sources import (
    KDestruction,
    KProduction,
    OmegaCrossDiffusion,
    OmegaDestruction,
    OmegaProduction,
)
from .sst import SSTModel
from .strain import strain_rate_magnitude
from .transport import SSTClosureFields, SSTTurbulence, WallFixedResidual

__all__ = [
    "AirAmgPreconditioner",
    "ConvectionAmgPreconditioner",
    "CoupledRANS",
    "CoupledRANSLayout",
    "CoupledShiftPolicy",
    "KDestruction",
    "KProduction",
    "OmegaCrossDiffusion",
    "OmegaDestruction",
    "OmegaProduction",
    "SSTClosureFields",
    "SSTModel",
    "SSTTurbulence",
    "ScalarShiftPolicy",
    "ScalarTransportPreconditioner",
    "WallFixedResidual",
    "bulk_velocity",
    "coupled_continuation",
    "equilibrium_k",
    "hybrid_initialize",
    "inlet_k",
    "inlet_omega",
    "omega_wall_value",
    "scalar_pseudo_transient_solve",
    "scalar_transport_preconditioner",
    "scalar_transport_shift_diagonal",
    "solve_coupled",
    "solve_segregated",
    "strain_rate_magnitude",
]
