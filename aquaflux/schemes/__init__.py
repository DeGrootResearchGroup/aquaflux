"""First-class swappable numerics: face interpolation, gradient reconstruction, slope limiting.

Schemes are strategy classes (``equinox.Module``) with a known order of accuracy, tested in
isolation and consumed by operators via injection, so the numerics can be swapped (compact
Green–Gauss → corrected → implicit gradient; unlimited → Venkatakrishnan-limited) without
touching physics.
"""

from __future__ import annotations

from .gradient import (
    AveragedInteriorHessian,
    AveragedNeighbourHessian,
    CellBlockJacobi,
    CellPreconditioner,
    CoupledBlockSweep,
    CompactGreenGauss,
    ContractionRate,
    CorrectedGreenGauss,
    GmresGradientSolve,
    GradientPreconditioner,
    GradientScheme,
    GradientSolve,
    GradientSystem,
    HessianBoundaryClosure,
    HessianCorrectedGradient,
    PackedSystemSolve,
    NestedHessianSolve,
    HessianSolve,
    ExactCellBlock,
    InverseCellVolume,
    InverseVolume,
    OwnerHessian,
    PreparedBoundaryClosure,
    SweepCalibration,
    SweptGradientSolve,
    cell_diagonal_block,
    contraction_rate,
    narrow_gradient_sweeps,
)
from .interpolation import (
    blend_owner_neighbour,
    interpolate_owner_neighbour,
    interpolation_factor,
)
from .limiter import Limiter, VenkatakrishnanLimiter

__all__ = [
    "AveragedInteriorHessian",
    "AveragedNeighbourHessian",
    "CellBlockJacobi",
    "CellPreconditioner",
    "CompactGreenGauss",
    "ContractionRate",
    "CorrectedGreenGauss",
    "CoupledBlockSweep",
    "ExactCellBlock",
    "GmresGradientSolve",
    "GradientPreconditioner",
    "GradientScheme",
    "GradientSolve",
    "GradientSystem",
    "HessianBoundaryClosure",
    "HessianCorrectedGradient",
    "HessianSolve",
    "InverseCellVolume",
    "InverseVolume",
    "Limiter",
    "NestedHessianSolve",
    "OwnerHessian",
    "PackedSystemSolve",
    "PreparedBoundaryClosure",
    "SweepCalibration",
    "SweptGradientSolve",
    "VenkatakrishnanLimiter",
    "blend_owner_neighbour",
    "cell_diagonal_block",
    "contraction_rate",
    "interpolate_owner_neighbour",
    "interpolation_factor",
    "narrow_gradient_sweeps",
]
