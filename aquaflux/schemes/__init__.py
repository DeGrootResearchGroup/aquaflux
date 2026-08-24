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
    fastest_boundary_closure,
    narrow_gradient_sweeps,
)
from .multiple_correction import (
    Corrections,
    GradientBoundaryClosure,
    MultipleCorrectionGradient,
    OwnerGradient,
    SkewCorrectedGradient,
)
from .interpolation import (
    blend_owner_neighbour,
    interpolate_owner_neighbour,
    interpolation_factor,
    non_orthogonal_correction,
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
    "Corrections",
    "CoupledBlockSweep",
    "ExactCellBlock",
    "GmresGradientSolve",
    "GradientBoundaryClosure",
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
    "MultipleCorrectionGradient",
    "NestedHessianSolve",
    "OwnerGradient",
    "OwnerHessian",
    "PackedSystemSolve",
    "PreparedBoundaryClosure",
    "SkewCorrectedGradient",
    "SweepCalibration",
    "SweptGradientSolve",
    "VenkatakrishnanLimiter",
    "blend_owner_neighbour",
    "cell_diagonal_block",
    "contraction_rate",
    "fastest_boundary_closure",
    "interpolate_owner_neighbour",
    "interpolation_factor",
    "narrow_gradient_sweeps",
    "non_orthogonal_correction",
]
