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
:class:`FacesOn`), turned into their dual. :class:`PrismLayers` grows layers inward from the wall
before the fill, sized for the dual, so a wall-resolved mesh's layers join its core conformally;
over a wall that :class:`MappedQuads` has meshed with structured quadrilaterals (a band, such as a
pipe's cylinder, or a four-sided face) the layers are hexahedra. :class:`Polygons` carries a surface
of triangles and quadrilaterals between these steps, and :class:`PyramidCaps` caps the
quadrilaterals a tetrahedral fill meets. :class:`PipeCore` fills a pipe's core, inside its layers,
with a structured O-grid; the mesher finds pipes by itself. :func:`element_mesh` makes a mesh of any
tetrahedra, triangular prisms, hexahedra and pyramids.

:class:`OctreeCore` builds a core of hexahedra inside a region from a balanced :class:`Octree`,
refined where asked (:class:`NearSurface`): cubes beside smaller ones are cut into pyramids and
tetrahedra about their centres so no point hangs on an edge, and the core's boundary quadrilaterals
are capped with pyramids, leaving a closed triangle surface for a tetrahedral fill.

:class:`HexBlock` is a curved hexahedron cut into hexahedra by transfinite interpolation from its
edges (straight, an :class:`Arc`, a :class:`SampledCurve` or any :class:`Curve`) and faces (any
:class:`Surface`), its cells spread by a count, a :class:`Graded` progression or node fractions, its
faces optionally projected onto a true surface (:class:`PlaneSurface`, :class:`CylinderSurface`,
:class:`ConeSurface`, :class:`SphereSurface`, or a face of a CAD model), and movable as a whole
(:meth:`HexBlock.placed`). A :class:`BlockLattice` lays out a grid of blocks by coordinates in a
:class:`CartesianFrame` or :class:`CylindricalFrame` and is shaped by editing its arrays.
:meth:`BlockMesh.assemble` joins blocks sharing faces into one conformal mesh of hexahedra.
:class:`PipeOGrid`, :class:`Annulus` and :class:`CubedSphereCap` lay out the blocks of a pipe, the
space about a lamp sleeve, and the hemispherical shell about its end.

:class:`PolyhedralDual` turns a conformal mesh of any cell kinds into its polyhedral dual -- one cell
per primal point, cut into convex pieces at a sharp concave edge -- keeping the boundary's corners,
edges and patch borders sharp. :class:`FacePlanarizer` moves a mesh's free vertices to flatten its
faces without making the mesh's worst cells or faces worse.
"""

from __future__ import annotations

from .block_assembly import BlockMesh
from .block_topologies import Annulus, CubedSphereCap, PipeOGrid
from .blocks import Arc, Curve, HexBlock, SampledCurve, Surface
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
from .divisions import Graded, cell_counts
from .layers import LayerExtrusion, PrismLayers
from .lattices import BlockLattice, CartesianFrame, CylindricalFrame, Straight
from .mapped import MappedQuads, MappedSurface
from .octree import NearSurface, Octree, RefinementCriterion
from .pipe_core import PipeCore, PipeCoreFill
from .polygons import Polygons, PyramidCaps
from .surfaces import ConeSurface, CylinderSurface, PlaneSurface, SphereSurface
from .planarize import FacePlanarizer, Planarization

__all__ = [
    "Annulus",
    "Arc",
    "BlockLattice",
    "BlockMesh",
    "CadMesher",
    "CartesianFrame",
    "ConeSurface",
    "CubedSphereCap",
    "Curve",
    "CylinderSurface",
    "CylindricalFrame",
    "FacePlanarizer",
    "FaceSelection",
    "FacesNear",
    "FacesOn",
    "Graded",
    "HexBlock",
    "HexCore",
    "LayerExtrusion",
    "MappedQuads",
    "MappedSurface",
    "MeshingError",
    "NearSurface",
    "NetgenProcess",
    "Octree",
    "OctreeCore",
    "PipeCore",
    "PipeCoreFill",
    "PipeOGrid",
    "Planarization",
    "PlaneSurface",
    "Polygons",
    "PolyhedralDual",
    "PrismLayers",
    "PyramidCaps",
    "RefinementCriterion",
    "RegionFaces",
    "SampledCurve",
    "SphereSurface",
    "Straight",
    "Surface",
    "SurfaceTriangulation",
    "TetrahedralMesh",
    "TracedDual",
    "cell_counts",
    "check_closed_surface",
    "element_mesh",
    "netgen_available",
]
