"""Static mesh: connectivity, node coordinates, and derived face/cell geometry.

``Mesh`` (an ``equinox.Module``) holds only connectivity + node coordinates;
``Mesh.geometry()`` derives face and cell geometry via the dimension's
:class:`FaceGeometryScheme` strategy.
"""

from __future__ import annotations

from .cell import CellGeometry
from .collapse import collapse_extruded_direction
from .connectivity import (
    FaceCellConnectivity,
    FaceNodeConnectivity,
    interior_mask,
)
from .distance import distance_to_patches
from .face import (
    EdgeFaceGeometry,
    FaceGeometry,
    FaceGeometryScheme,
    PolygonFaceGeometry,
    face_geometry_scheme,
)
from .geometry import MeshGeometry
from .graph import cell_adjacency_coo, cell_adjacency_csr
from .groups import CellZones, FacePatches, LabelledGroups
from .mesh import Mesh
from .quality import centroid_iteration_shift, closed_cell_residual, face_planarity
from .reorder import (
    CellReordering,
    IdentityReordering,
    RandomReordering,
    ReverseCuthillMcKee,
    permute_cells,
)
from .structured import graded_nodes, structured_grid_2d, structured_grid_3d

__all__ = [
    "CellGeometry",
    "CellReordering",
    "CellZones",
    "EdgeFaceGeometry",
    "FaceCellConnectivity",
    "FaceGeometry",
    "FaceGeometryScheme",
    "FaceNodeConnectivity",
    "FacePatches",
    "IdentityReordering",
    "LabelledGroups",
    "Mesh",
    "MeshGeometry",
    "PolygonFaceGeometry",
    "RandomReordering",
    "ReverseCuthillMcKee",
    "cell_adjacency_coo",
    "cell_adjacency_csr",
    "centroid_iteration_shift",
    "closed_cell_residual",
    "collapse_extruded_direction",
    "distance_to_patches",
    "face_geometry_scheme",
    "face_planarity",
    "graded_nodes",
    "interior_mask",
    "permute_cells",
    "structured_grid_2d",
    "structured_grid_3d",
]
