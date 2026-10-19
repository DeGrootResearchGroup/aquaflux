"""Mesh generation from computer-aided design (CAD) geometry.

Builds the meshes aquaflux solves on from a solid read from a CAD file. This package holds the
generators and the external mesher they call; the result is an ordinary
:class:`~aquaflux.mesh.Mesh`, written out for other codes by :mod:`aquaflux.io`.

:class:`NetgenProcess` runs the Netgen mesher -- surface triangulation of a solid's faces, and a
tetrahedral fill that keeps a supplied boundary surface exactly -- in a child process, because its
copy of the geometry kernel cannot share an interpreter with the CAD reader's. It needs the optional
mesher, ``pip install aquaflux[meshing]``; :func:`netgen_available` says whether it is installed
without loading it.

:class:`PolyhedralDual` turns a conformal mesh of any cell kinds into its polyhedral dual -- one cell
per primal point, cut into convex pieces at a sharp concave edge -- keeping the boundary's corners,
edges and patch borders sharp.
"""

from __future__ import annotations

from .dual import PolyhedralDual
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
    "PolyhedralDual",
    "SurfaceTriangulation",
    "TetrahedralMesh",
    "check_closed_surface",
    "netgen_available",
]
