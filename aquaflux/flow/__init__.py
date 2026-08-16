"""Coupled pressure--velocity (flow) solver: momentum + Rhie--Chow continuity.

The block ``(u, v[, w], p)`` system, solved monolithically by the same differentiable Newton /
implicit-diff machinery as a scalar field. Momentum reuses the scalar advection and diffusion
operators (viscosity as the diffusion coefficient); continuity uses the Rhie--Chow face mass
flux to couple pressure implicitly. See :class:`MomentumContinuity`.
"""

from __future__ import annotations

from .block_preconditioner import BlockPreconditioner, frozen_momentum_diagonal_parts
from .boundary import FlowBoundary, MovingWall, NoSlipWall, PressureOutlet, VelocityInlet
from .continuation import (
    FrozenViscosityVelocityParts,
    MomentumShiftPolicy,
    momentum_continuation,
    reused_flow_solve,
)
from .initialization import bernoulli_pressure, laplace_field, potential_flow
from .mean_velocity import bulk_velocity_flow_solve
from .scales import body_force_velocity, characteristic_velocity
from .momentum import FlowFields, MomentumContinuity, PressureForce, VelocityFields
from .source import MomentumSource, UniformBodyForce
from .preconditioner import damped_jacobi_solve, pressure_schur_laplacian
from .rhie_chow import interior_mass_flux, momentum_diagonal

__all__ = [
    "BlockPreconditioner",
    "FlowBoundary",
    "FlowFields",
    "FrozenViscosityVelocityParts",
    "MomentumContinuity",
    "MomentumShiftPolicy",
    "MomentumSource",
    "MovingWall",
    "NoSlipWall",
    "PressureForce",
    "PressureOutlet",
    "UniformBodyForce",
    "VelocityFields",
    "VelocityInlet",
    "bernoulli_pressure",
    "body_force_velocity",
    "bulk_velocity_flow_solve",
    "characteristic_velocity",
    "damped_jacobi_solve",
    "frozen_momentum_diagonal_parts",
    "interior_mass_flux",
    "laplace_field",
    "momentum_continuation",
    "momentum_diagonal",
    "potential_flow",
    "pressure_schur_laplacian",
    "reused_flow_solve",
]
