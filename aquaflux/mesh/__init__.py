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
    index_dtype,
    interior_mask,
)
from .digest import mesh_digest
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
from .groups import PADDING_PATCH, RESERVED_PATCH_NAMES, CellZones, FacePatches, LabelledGroups
from .mesh import Mesh
from .periodic import (
    PERIODIC_MATCH_TOLERANCE,
    PeriodicJoin,
    face_vertex_means,
    matched_by_translation,
    without_faces,
)
from .quality import (
    CONCAVE_COSINE,
    MeshQuality,
    QualitySummary,
    cell_concavity,
    cell_non_convexity,
    centroid_iteration_shift,
    closed_cell_residual,
    face_non_orthogonality,
    face_planarity,
    face_skewness,
    face_warp,
    triangulation_volume_change,
)
from .reorder import (
    CellReordering,
    IdentityReordering,
    RandomReordering,
    ReverseCuthillMcKee,
    permute_cells,
)
from .structured import graded_nodes, structured_grid_2d, structured_grid_3d
from .surface import PatchTriangles, patch_triangles
from .winding import stored_ring_is_outward

__all__ = [
    "CONCAVE_COSINE",
    "PADDING_PATCH",
    "PERIODIC_MATCH_TOLERANCE",
    "RESERVED_PATCH_NAMES",
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
    "MeshQuality",
    "PatchTriangles",
    "PeriodicJoin",
    "PolygonFaceGeometry",
    "QualitySummary",
    "RandomReordering",
    "ReverseCuthillMcKee",
    "cell_adjacency_coo",
    "cell_adjacency_csr",
    "cell_concavity",
    "cell_non_convexity",
    "centroid_iteration_shift",
    "closed_cell_residual",
    "collapse_extruded_direction",
    "distance_to_patches",
    "face_geometry_scheme",
    "face_non_orthogonality",
    "face_planarity",
    "face_skewness",
    "face_vertex_means",
    "face_warp",
    "graded_nodes",
    "index_dtype",
    "interior_mask",
    "matched_by_translation",
    "mesh_digest",
    "patch_triangles",
    "permute_cells",
    "stored_ring_is_outward",
    "structured_grid_2d",
    "structured_grid_3d",
    "triangulation_volume_change",
    "without_faces",
]
