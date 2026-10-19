"""Mesh generation from computer-aided design (CAD) geometry.

Builds the meshes aquaflux solves on from a solid read from a CAD file. This package holds the
generators and the external mesher they call; the result is an ordinary
:class:`~aquaflux.mesh.Mesh`, written out for other codes by :mod:`aquaflux.io`.

:class:`NetgenProcess` runs the Netgen mesher -- surface triangulation of a solid's faces, and a
tetrahedral fill that keeps a supplied boundary surface exactly -- in a child process, because its
copy of the geometry kernel cannot share an interpreter with the CAD reader's. It needs the optional
mesher, ``pip install aquaflux[meshing]``; :func:`netgen_available` says whether it is installed
without loading it.
"""

from __future__ import annotations

from .netgen_process import (
    MeshingError,
    NetgenProcess,
    SurfaceTriangulation,
    TetrahedralMesh,
    check_closed_surface,
    netgen_available,
)

__all__ = [
    "MeshingError",
    "NetgenProcess",
    "SurfaceTriangulation",
    "TetrahedralMesh",
    "check_closed_surface",
    "netgen_available",
]
