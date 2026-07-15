"""Layer-0 residual substrate: gather → compute → scatter assembly of `R(state, params)`.

The residual is assembled by `segment_sum` scatter over face→cell index arrays from
injected per-operator flux/source closures; the Jacobian and adjoint come from AD.
No hand-derived linearization coefficients live here.
"""

from __future__ import annotations

from .advection import (
    AdvectionFlux,
    AdvectionScheme,
    FirstOrderUpwind,
    LimitedUpwind,
)
from .diffusion import DiffusionFlux
from .face_flux import FaceContext, FaceFluxOperator
from .transient import TransientTerm

__all__ = [
    "AdvectionFlux",
    "AdvectionScheme",
    "DiffusionFlux",
    "FaceContext",
    "FaceFluxOperator",
    "FirstOrderUpwind",
    "LimitedUpwind",
    "TransientTerm",
]
