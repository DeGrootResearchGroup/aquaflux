"""Mesh import/export: reading external mesh formats into an aquaflux :class:`~aquaflux.mesh.Mesh`.

Separates file-format concerns from mesh representation: :mod:`aquaflux.mesh` owns the mesh
structure, and this package owns the readers that build one from an external source. Every reader
implements the format-agnostic :class:`MeshReader` contract (``read() -> Mesh``); the first is the
OpenFOAM polyMesh reader, exposed as :class:`OpenFOAMReader` and the :func:`read_openfoam`
convenience.
"""

from __future__ import annotations

from .openfoam import OpenFOAMReader, read_openfoam
from .reader import MeshReader

__all__ = [
    "MeshReader",
    "OpenFOAMReader",
    "read_openfoam",
]
