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
from .diffusion import (
    DiffusionFlux,
    flux_continuous_conductance,
    flux_continuous_denominator,
)
from .face_flux import FaceContext, FaceFluxOperator
from .fixed_value import DifferenceRow, FixationRow, FixedValueCells, LogRatioRow
from .residual import CellBalance, ResidualAssembler
from .source import VolumeSource
from .transient import TransientTerm

__all__ = [
    "AdvectionFlux",
    "AdvectionScheme",
    "CellBalance",
    "DifferenceRow",
    "DiffusionFlux",
    "FaceContext",
    "FaceFluxOperator",
    "FirstOrderUpwind",
    "FixationRow",
    "FixedValueCells",
    "LimitedUpwind",
    "LogRatioRow",
    "ResidualAssembler",
    "TransientTerm",
    "VolumeSource",
    "flux_continuous_conductance",
    "flux_continuous_denominator",
]
