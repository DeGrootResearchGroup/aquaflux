"""First-class swappable numerics: face interpolation, gradient reconstruction, slope limiting.

Schemes are strategy classes (``equinox.Module``) with a known order of accuracy, tested in
isolation and consumed by operators via injection, so the numerics can be swapped (compact
Green–Gauss → corrected → implicit gradient; unlimited → Venkatakrishnan-limited) without
touching physics.
"""

from __future__ import annotations

from .gradient import (
    CellBlockJacobi,
    CellPreconditioner,
    CompactGreenGauss,
    ContractionRate,
    CorrectedGreenGauss,
    GmresGradientSolve,
    GradientPreconditioner,
    GradientScheme,
    GradientSolve,
    GradientSystem,
    HessianCorrectedGradient,
    ExactCellBlock,
    InverseCellVolume,
    InverseVolume,
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
    "CellBlockJacobi",
    "CellPreconditioner",
    "CompactGreenGauss",
    "ContractionRate",
    "CorrectedGreenGauss",
    "ExactCellBlock",
    "GmresGradientSolve",
    "GradientPreconditioner",
    "GradientScheme",
    "GradientSolve",
    "GradientSystem",
    "HessianCorrectedGradient",
    "InverseCellVolume",
    "InverseVolume",
    "Limiter",
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
