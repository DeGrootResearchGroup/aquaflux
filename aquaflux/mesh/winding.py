"""Which way each face's stored node ring runs, relative to its owner cell.

A mesh accepts a face's nodes in either direction around the perimeter and never reorders them: it
computes the normal the ring implies and then flips *the normal* to point out of the owner, as a
separate array. So the stored ring's own direction is arbitrary, and anything that writes the rings
out in a format that gives the direction a meaning -- a polyhedron listed face by face, or a file
whose face order says which cell owns it -- has to recover that direction first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aquaflux.vectors import dot

from .cell import CellGeometry
from .face import face_geometry_scheme

if TYPE_CHECKING:  # pragma: no cover
    from .mesh import Mesh


def stored_ring_is_outward(mesh: Mesh) -> np.ndarray:
    """Whether each face's stored node ring already winds outward from its owner cell.

    The flip the mesh applies to the ring's normal is recovered by running the mesh's own
    face-geometry strategy, rather than by re-deriving the orientation test here.

    Parameters
    ----------
    mesh : Mesh
        The mesh whose face rings are in question.

    Returns
    -------
    np.ndarray of bool, shape ``(n_faces,)``
        ``True`` where the stored ring is owner-outward: its right-hand normal points out of the
        owner cell. A degenerate face, whose ring implies no normal at all, reports ``True``; its
        winding cannot be determined and does not matter, because a face of zero area bounds no
        volume.
    """
    scheme = face_geometry_scheme(mesh.dim)
    _, centroid, node_order_normal = scheme.unoriented_geometry(mesh.node_coords, mesh.face_nodes)
    approx = CellGeometry.approx_centroids(centroid, mesh.face_cells)
    outward = scheme.orient_owner_outward(
        node_order_normal, centroid, approx[mesh.face_cells.owner]
    )
    # `orient_owner_outward` returns the ring's own normal times +-1, so this projection is
    # +-|n|^2: its sign is the flip, and nothing else about the magnitude is used.
    return np.asarray(dot(outward, node_order_normal)) >= 0.0
