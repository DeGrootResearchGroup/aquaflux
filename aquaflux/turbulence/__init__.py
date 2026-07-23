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

from .boundary import (
    equilibrium_k,
    log_layer_shear_rate,
    inlet_k,
    inlet_omega,
    k_wall_production,
    nut_wall,
    omega_wall,
    omega_wall_value,
    wall_function_weight,
    wall_k_diffusivity,
    wall_shear_stress,
    wall_y_star,
)
from .continuation import ScalarShiftPolicy, scalar_pseudo_transient_solve
from .coupled import (
    CoupledRANS,
    CoupledRANSLayout,
    CoupledShiftPolicy,
    DirectScalars,
    LogScalars,
    ScalarVariableTransform,
    coupled_continuation,
    solve_coupled,
)
from .driver import bulk_velocity, solve_segregated
from .initialization import hybrid_initialize
from .preconditioner import (
    AirAmgPreconditioner,
    ConvectionAmgPreconditioner,
    ScalarTransportPreconditioner,
    ScaledScalarPreconditioner,
    scalar_transport_preconditioner,
    scalar_transport_shift_diagonal,
)
from .sources import (
    KDestruction,
    KProduction,
    NearWallKClosure,
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
    "DirectScalars",
    "KDestruction",
    "KProduction",
    "LogScalars",
    "NearWallKClosure",
    "OmegaCrossDiffusion",
    "OmegaDestruction",
    "OmegaProduction",
    "SSTClosureFields",
    "SSTModel",
    "SSTTurbulence",
    "ScalarShiftPolicy",
    "ScalarTransportPreconditioner",
    "ScalarVariableTransform",
    "ScaledScalarPreconditioner",
    "WallFixedResidual",
    "bulk_velocity",
    "coupled_continuation",
    "equilibrium_k",
    "hybrid_initialize",
    "inlet_k",
    "inlet_omega",
    "k_wall_production",
    "log_layer_shear_rate",
    "nut_wall",
    "omega_wall",
    "omega_wall_value",
    "scalar_pseudo_transient_solve",
    "scalar_transport_preconditioner",
    "scalar_transport_shift_diagonal",
    "solve_coupled",
    "solve_segregated",
    "strain_rate_magnitude",
    "wall_function_weight",
    "wall_k_diffusivity",
    "wall_shear_stress",
    "wall_y_star",
]
