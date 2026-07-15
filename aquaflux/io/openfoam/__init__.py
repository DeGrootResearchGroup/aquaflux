"""OpenFOAM polyMesh reader.

Reads an ASCII OpenFOAM ``constant/polyMesh`` directory into an aquaflux
:class:`~aquaflux.mesh.Mesh`, collapsing a one-cell-thick ``empty``-capped case to a genuine 2D
mesh. The public entry points are :class:`OpenFOAMReader` and the :func:`read_openfoam` convenience.
"""

from __future__ import annotations

from .reader import OpenFOAMReader, read_openfoam

__all__ = [
    "OpenFOAMReader",
    "read_openfoam",
]
