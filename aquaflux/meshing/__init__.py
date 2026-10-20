"""Mesh generation from computer-aided design (CAD) geometry.

Builds the meshes aquaflux solves on from a solid read from a CAD file. This package holds the
generators and the external mesher they call; the result is an ordinary
:class:`~aquaflux.mesh.Mesh`, written out for other codes by :mod:`aquaflux.io`.

:class:`NetgenProcess` runs the Netgen mesher -- surface triangulation of a solid's faces, and a
tetrahedral fill that keeps a supplied boundary surface exactly -- in a child process, because its
copy of the geometry kernel cannot share an interpreter with the CAD reader's. It needs the optional
mesher, ``pip install aquaflux[meshing]``; :func:`netgen_available` says whether it is installed
without loading it.

:class:`CadMesher` meshes a region of a CAD model -- some solids' union, less others -- as polyhedra:
Netgen's tetrahedra of it, their boundary in patches chosen by face (:class:`FacesNear`,
:class:`FacesOn`), turned into their dual. :class:`PrismLayers` grows layers of prisms inward from the wall before the fill, sized for the dual,
so a wall-resolved mesh's layers join its core conformally. :func:`element_mesh` makes a mesh of
any tetrahedra, triangular prisms, hexahedra and pyramids.

:class:`OctreeCore` builds a core of hexahedra inside a region from a balanced :class:`Octree`,
refined where asked (:class:`NearSurface`): cubes beside smaller ones are cut into pyramids and
tetrahedra about their centres so no point hangs on an edge, and the core's boundary quadrilaterals
are capped with pyramids, leaving a closed triangle surface for a tetrahedral fill.

:class:`PolyhedralDual` turns a conformal mesh of any cell kinds into its polyhedral dual -- one cell
per primal point, cut into convex pieces at a sharp concave edge -- keeping the boundary's corners,
edges and patch borders sharp.
"""

from __future__ import annotations

from .cad_mesher import CadMesher, FaceSelection, FacesNear, FacesOn, RegionFaces
from .dual import PolyhedralDual, TracedDual
from .netgen_process import (
    MeshingError,
    NetgenProcess,
    SurfaceTriangulation,
    TetrahedralMesh,
    check_closed_surface,
    netgen_available,
)
from .elements import element_mesh
from .hex_core import HexCore, OctreeCore
from .layers import LayerExtrusion, PrismLayers
from .octree import NearSurface, Octree, RefinementCriterion

__all__ = [
    "CadMesher",
    "FaceSelection",
    "FacesNear",
    "FacesOn",
    "HexCore",
    "LayerExtrusion",
    "MeshingError",
    "NearSurface",
    "NetgenProcess",
    "Octree",
    "OctreeCore",
    "PolyhedralDual",
    "PrismLayers",
    "RefinementCriterion",
    "RegionFaces",
    "SurfaceTriangulation",
    "TetrahedralMesh",
    "TracedDual",
    "check_closed_surface",
    "element_mesh",
    "netgen_available",
]
