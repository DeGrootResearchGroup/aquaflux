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
    BetaTaperedDamping,
    ConstantDamping,
    ResidualTaperedDamping,
    TurbulenceDamping,
    turbulence_residual_norm,
    production_cap_active,
    positive_k_limit,
    positive_k_projection,
    coupled_jacobian_probe,
    CoupledRANS,
    coupled_rans_layout,
    CoupledShiftPolicy,
    LiveViscosityVelocityParts,
    DirectScalars,
    LogScalars,
    PreconditionerSession,
    ScalarVariableTransform,
    coupled_step,
    open_session,
    wall_consistent_state,
    eddy_viscosity_drift,
    solve_coupled,
)
from .driver import bulk_velocity, solve_segregated
from .initialization import hybrid_initialize, wall_consistent_omega
from .march_settings import ShiftSettings
from .preconditioner import (
    AirAmgPreconditioner,
    ConvectionAmgPreconditioner,
    ScalarAir,
    ScalarBlock,
    ScalarTransportPreconditioner,
    ScalarTwoLevel,
    ScaledScalarPreconditioner,
    UnpreconditionedScalars,
    scalar_transport_preconditioner,
    scalar_transport_shift_diagonal,
)
from .preconditioner_spec import (
    BlockDiagonal,
    CompleteLu,
    FieldSplit,
    JacobianProbeSpec,
    MaterializedJacobian,
    MonolithicVCycle,
    preconditioner_spec_from_mapping,
    preconditioner_spec_to_mapping,
)
from .reynolds import (
    AdaptiveReynoldsSchedule,
    GeometricReynoldsSchedule,
    ViscosityRampHomotopy,
    scale_both_blocks,
    scale_momentum_only,
    ReynoldsPoint,
    ReynoldsSchedule,
    solve_reynolds_continuation,
    solve_reynolds_ramp,
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
    "AdaptiveReynoldsSchedule",
    "AirAmgPreconditioner",
    "BetaTaperedDamping",
    "BlockDiagonal",
    "CompleteLu",
    "ConstantDamping",
    "ConvectionAmgPreconditioner",
    "CoupledRANS",
    "CoupledShiftPolicy",
    "DirectScalars",
    "FieldSplit",
    "GeometricReynoldsSchedule",
    "JacobianProbeSpec",
    "KDestruction",
    "KProduction",
    "LiveViscosityVelocityParts",
    "LogScalars",
    "MaterializedJacobian",
    "MonolithicVCycle",
    "NearWallKClosure",
    "OmegaCrossDiffusion",
    "OmegaDestruction",
    "OmegaProduction",
    "PreconditionerSession",
    "ResidualTaperedDamping",
    "ReynoldsPoint",
    "ReynoldsSchedule",
    "SSTClosureFields",
    "SSTModel",
    "SSTTurbulence",
    "ScalarAir",
    "ScalarBlock",
    "ScalarShiftPolicy",
    "ScalarTransportPreconditioner",
    "ScalarTwoLevel",
    "ScalarVariableTransform",
    "ScaledScalarPreconditioner",
    "ShiftSettings",
    "TurbulenceDamping",
    "UnpreconditionedScalars",
    "ViscosityRampHomotopy",
    "WallFixedResidual",
    "bulk_velocity",
    "coupled_equation_names",
    "coupled_fields",
    "coupled_jacobian_probe",
    "coupled_rans_layout",
    "coupled_residuals",
    "coupled_step",
    "eddy_viscosity_drift",
    "equilibrium_k",
    "hybrid_initialize",
    "inlet_k",
    "inlet_omega",
    "k_wall_production",
    "log_layer_shear_rate",
    "nut_wall",
    "omega_wall",
    "omega_wall_gradient",
    "omega_wall_value",
    "open_session",
    "positive_k_limit",
    "positive_k_projection",
    "preconditioner_spec_from_mapping",
    "preconditioner_spec_to_mapping",
    "production_and_limit",
    "production_cap_active",
    "scalar_pseudo_transient_solve",
    "scalar_transport_preconditioner",
    "scalar_transport_shift_diagonal",
    "scale_both_blocks",
    "scale_momentum_only",
    "solve_coupled",
    "solve_reynolds_continuation",
    "solve_reynolds_ramp",
    "solve_segregated",
    "strain_rate_magnitude",
    "turbulence_residual_norm",
    "wall_consistent_omega",
    "wall_consistent_state",
    "wall_function_weight",
    "wall_k_diffusivity",
    "wall_shear_stress",
    "wall_y_star",
]
