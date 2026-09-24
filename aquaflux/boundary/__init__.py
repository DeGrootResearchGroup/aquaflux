"""Weak boundary-face-value closures (a BC is a special face interpolator).

Patch-based Dirichlet / flux / zero-gradient / convective conditions imposed weakly
through the boundary-face flux, sharing the interior face-interpolation interface.
Pure functions of boundary-cell state + face geometry + BC parameters.
"""

from __future__ import annotations

from .collection import BoundaryConditions, refuse_a_closure_that_closes_other_fields
from .conditions import (
    HOST_EQUATION_FIELD,
    BoundaryCondition,
    Convective,
    Dirichlet,
    DirichletField,
    Neumann,
    ZeroGradient,
)

__all__ = [
    "HOST_EQUATION_FIELD",
    "BoundaryCondition",
    "BoundaryConditions",
    "Convective",
    "Dirichlet",
    "DirichletField",
    "Neumann",
    "ZeroGradient",
    "refuse_a_closure_that_closes_other_fields",
]
