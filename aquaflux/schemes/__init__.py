"""First-class swappable numerics: face interpolation, gradient reconstruction, slope limiting.

Schemes are strategy classes (``equinox.Module``) with a known order of accuracy, tested in
isolation and consumed by operators via injection, so the numerics can be swapped (compact
Green–Gauss → corrected → implicit gradient; unlimited → Venkatakrishnan-limited) without
touching physics.
"""

from __future__ import annotations

from .gradient import (
    CompactGreenGauss,
    CorrectedGreenGauss,
    GmresGradientSolve,
    GradientScheme,
    GradientSolve,
    HessianCorrectedGradient,
    SweptGradientSolve,
)
from .interpolation import interpolate_owner_neighbour, interpolation_factor
from .limiter import Limiter, VenkatakrishnanLimiter

__all__ = [
    "CompactGreenGauss",
    "CorrectedGreenGauss",
    "GmresGradientSolve",
    "GradientScheme",
    "GradientSolve",
    "HessianCorrectedGradient",
    "Limiter",
    "SweptGradientSolve",
    "VenkatakrishnanLimiter",
    "interpolate_owner_neighbour",
    "interpolation_factor",
]
