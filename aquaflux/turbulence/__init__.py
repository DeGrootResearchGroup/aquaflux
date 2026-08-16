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
    omega_wall_gradient,
    omega_wall_value,
    wall_function_weight,
    wall_k_diffusivity,
    wall_shear_stress,
    wall_y_star,
)
from .continuation import ScalarShiftPolicy, scalar_pseudo_transient_solve
from .diagnostics import coupled_equation_names, coupled_fields, coupled_residuals
from .coupled import (
    production_cap_active,
    positive_k_limit,
    positive_k_projection,
    CoupledJacobianProbe,
    CoupledRANS,
    CoupledRANSLayout,
    CoupledShiftPolicy,
    LiveViscosityVelocityParts,
    DirectScalars,
    LogScalars,
    MonolithicFactorShiftPolicy,
    ScalarVariableTransform,
    amg_beta_tracking_refresh,
    coupled_amg_continuation,
    coupled_continuation,
    coupled_ilut_continuation,
    coupled_ilut_refreshing_continuation,
    coupled_lu_continuation,
    coupled_lu_refreshing_continuation,
    eddy_viscosity_drift,
    ilut_beta_tracking_refresh,
    lu_beta_tracking_refresh,
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
from .reynolds import (
    ReynoldsPoint,
    GeometricReynoldsSchedule,
    ReynoldsSchedule,
    solve_reynolds_continuation,
)
from .sources import (
    production_and_limit,
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
    "CoupledJacobianProbe",
    "CoupledRANS",
    "CoupledRANSLayout",
    "CoupledShiftPolicy",
    "DirectScalars",
    "GeometricReynoldsSchedule",
    "KDestruction",
    "KProduction",
    "LiveViscosityVelocityParts",
    "LogScalars",
    "MonolithicFactorShiftPolicy",
    "NearWallKClosure",
    "OmegaCrossDiffusion",
    "OmegaDestruction",
    "OmegaProduction",
    "ReynoldsPoint",
    "ReynoldsSchedule",
    "SSTClosureFields",
    "SSTModel",
    "SSTTurbulence",
    "ScalarShiftPolicy",
    "ScalarTransportPreconditioner",
    "ScalarVariableTransform",
    "ScaledScalarPreconditioner",
    "WallFixedResidual",
    "amg_beta_tracking_refresh",
    "bulk_velocity",
    "coupled_amg_continuation",
    "coupled_continuation",
    "coupled_equation_names",
    "coupled_fields",
    "coupled_ilut_continuation",
    "coupled_ilut_refreshing_continuation",
    "coupled_lu_continuation",
    "coupled_lu_refreshing_continuation",
    "coupled_residuals",
    "eddy_viscosity_drift",
    "equilibrium_k",
    "hybrid_initialize",
    "ilut_beta_tracking_refresh",
    "inlet_k",
    "inlet_omega",
    "k_wall_production",
    "log_layer_shear_rate",
    "lu_beta_tracking_refresh",
    "nut_wall",
    "omega_wall",
    "omega_wall_gradient",
    "omega_wall_value",
    "positive_k_limit",
    "positive_k_projection",
    "production_and_limit",
    "production_cap_active",
    "scalar_pseudo_transient_solve",
    "scalar_transport_preconditioner",
    "scalar_transport_shift_diagonal",
    "solve_coupled",
    "solve_reynolds_continuation",
    "solve_segregated",
    "strain_rate_magnitude",
    "wall_function_weight",
    "wall_k_diffusivity",
    "wall_shear_stress",
    "wall_y_star",
]
