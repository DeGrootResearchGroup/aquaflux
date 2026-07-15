"""Weak boundary-face-value closures (a BC is a special face interpolator).

Patch-based Dirichlet / flux / zero-gradient / convective conditions imposed weakly
through the boundary-face flux, sharing the interior face-interpolation interface.
Pure functions of boundary-cell state + face geometry + BC parameters.
"""

from __future__ import annotations

from .conditions import (
    BoundaryCondition,
    Convective,
    Dirichlet,
    DirichletField,
    Neumann,
    ZeroGradient,
)
from .patch import apply_per_patch

__all__ = [
    "BoundaryCondition",
    "Convective",
    "Dirichlet",
    "DirichletField",
    "Neumann",
    "ZeroGradient",
    "apply_per_patch",
]
