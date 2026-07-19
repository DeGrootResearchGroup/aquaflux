"""Coupled pressure--velocity (flow) solver: momentum + Rhie--Chow continuity.

The block ``(u, v[, w], p)`` system, solved monolithically by the same differentiable Newton /
implicit-diff machinery as a scalar field. Momentum reuses the scalar advection and diffusion
operators (viscosity as the diffusion coefficient); continuity uses the Rhie--Chow face mass
flux to couple pressure implicitly. See :class:`MomentumContinuity`.
"""

from __future__ import annotations

from .block_preconditioner import BlockPreconditioner
from .boundary import FlowBoundary, MovingWall, NoSlipWall, PressureOutlet, VelocityInlet
from .continuation import MomentumShiftPolicy, momentum_continuation, reused_flow_solve
from .initialization import laplace_field, potential_flow
from .momentum import MomentumContinuity
from .preconditioner import damped_jacobi_solve, pressure_schur_laplacian
from .rhie_chow import interior_mass_flux, momentum_diagonal

__all__ = [
    "BlockPreconditioner",
    "FlowBoundary",
    "MomentumContinuity",
    "MomentumShiftPolicy",
    "MovingWall",
    "NoSlipWall",
    "PressureOutlet",
    "VelocityInlet",
    "damped_jacobi_solve",
    "interior_mass_flux",
    "laplace_field",
    "momentum_continuation",
    "momentum_diagonal",
    "potential_flow",
    "pressure_schur_laplacian",
    "reused_flow_solve",
]
