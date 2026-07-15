"""Weak boundary-face-value closures (a BC is a special face interpolator).

Patch-based Dirichlet / flux / zero-gradient / convective conditions imposed weakly
through the boundary-face flux, sharing the interior face-interpolation interface.
Pure functions of boundary-cell state + face geometry + BC parameters.
"""

from __future__ import annotations

from .collection import BoundaryConditions
from .conditions import (
    BoundaryCondition,
    Convective,
    Dirichlet,
    DirichletField,
    Neumann,
    ZeroGradient,
)

__all__ = [
    "BoundaryCondition",
    "BoundaryConditions",
    "Convective",
    "Dirichlet",
    "DirichletField",
    "Neumann",
    "ZeroGradient",
]
