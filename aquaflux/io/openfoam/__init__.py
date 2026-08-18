"""OpenFOAM polyMesh reader.

Reads an ASCII OpenFOAM ``constant/polyMesh`` directory into an aquaflux
:class:`~aquaflux.mesh.Mesh`, collapsing a one-cell-thick ``empty``-capped case to a genuine 2D
mesh. The public entry points are :class:`OpenFOAMReader` and the :func:`read_openfoam` convenience.
"""

from __future__ import annotations

from .fields import parse_scalar_field, read_surface_scalar_field, read_volume_scalar_field
from .reader import OpenFOAMReader, read_openfoam

__all__ = [
    "OpenFOAMReader",
    "parse_scalar_field",
    "read_openfoam",
    "read_surface_scalar_field",
    "read_volume_scalar_field",
]
