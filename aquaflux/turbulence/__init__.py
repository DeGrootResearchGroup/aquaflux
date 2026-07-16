"""Turbulence models: RANS closures that supply an eddy viscosity to the flow solve.

The k-omega SST constants and the closed-form quantities they define -- the blending functions, the
constant blend, and the eddy viscosity -- live in :class:`~aquaflux.turbulence.sst.SSTModel`; the
strain-rate magnitude they consume in
:func:`~aquaflux.turbulence.strain.strain_rate_magnitude`.
"""

from __future__ import annotations

from .sst import SSTModel
from .strain import strain_rate_magnitude

__all__ = [
    "SSTModel",
    "strain_rate_magnitude",
]
