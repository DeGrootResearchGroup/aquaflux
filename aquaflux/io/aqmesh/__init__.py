"""aquaflux's own mesh file, ``.aqmesh``: a mesh's arrays as it holds them, read back exactly.

See :mod:`aquaflux.io.aqmesh.mesh_file`.
"""

from __future__ import annotations

from .mesh_file import (
    AQMESH_SUFFIX,
    AQMESH_VERSION,
    AquafluxMeshHeader,
    AquafluxMeshReader,
    AquafluxMeshWriter,
    read_aqmesh,
    read_aqmesh_header,
    write_aqmesh,
)

__all__ = [
    "AQMESH_SUFFIX",
    "AQMESH_VERSION",
    "AquafluxMeshHeader",
    "AquafluxMeshReader",
    "AquafluxMeshWriter",
    "read_aqmesh",
    "read_aqmesh_header",
    "write_aqmesh",
]
