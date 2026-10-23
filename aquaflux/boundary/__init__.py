"""Weak boundary-face-value closures (a BC is a special face interpolator).

Patch-based Dirichlet / flux / zero-gradient / convective conditions imposed weakly
through the boundary-face flux, sharing the interior face-interpolation interface.
Pure functions of boundary-cell state + face geometry + BC parameters. A value that varies over a
patch is a function of face centroids; :class:`RadialProfile` is one given against distance from an
axis, the profile of a fully developed pipe flow.
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
    as_float_leaf,
    as_position_function,
)
from .radial import AxialVector, RadialProfile, radial_average

__all__ = [
    "HOST_EQUATION_FIELD",
    "AxialVector",
    "BoundaryCondition",
    "BoundaryConditions",
    "Convective",
    "Dirichlet",
    "DirichletField",
    "Neumann",
    "RadialProfile",
    "ZeroGradient",
    "as_float_leaf",
    "as_position_function",
    "radial_average",
    "refuse_a_closure_that_closes_other_fields",
]
