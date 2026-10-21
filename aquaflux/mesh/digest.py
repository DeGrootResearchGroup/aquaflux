"""A mesh's digest: one string for one mesh, wherever and however often it is read.

Two readings of the same mesh -- the same file read twice, or read on another machine whose rounding
differs in a coordinate's last bits -- give the same digest; a renumbered or moved mesh does not. A
checkpoint records it to refuse fields on another mesh, and a mesh file to find out it has been
changed since it was written.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .mesh import Mesh

__all__ = ["mesh_digest"]

#: How finely a node's position is resolved when a mesh is digested, as a fraction of the mesh's
#: largest extent. Coarse enough that rounding in the last bits of a coordinate -- the same mesh read
#: on another machine -- lands in the same bin, fine enough that no mesh anyone would call the same
#: shares a digest with a moved one.
_DIGEST_RESOLUTION = 1.0e-6


def mesh_digest(mesh: Mesh) -> str:
    """A digest of a mesh's topology and node positions, the same for the same mesh wherever it is read.

    Parameters
    ----------
    mesh : Mesh
        The mesh. Only its face-to-cell connectivity and node coordinates are read, so no geometry
        has to have been computed.

    Returns
    -------
    str
        A hexadecimal digest. Two meshes with the same cell numbering and nodes within a millionth of
        the mesh's extent of each other share one.
    """
    owner = np.asarray(mesh.face_cells.owner, dtype=np.int64)
    neighbour = np.asarray(mesh.face_cells.neighbour, dtype=np.int64)
    nodes = np.asarray(mesh.node_coords, dtype=np.float64)
    extent = float(np.max(nodes.max(axis=0) - nodes.min(axis=0)))
    resolution = _DIGEST_RESOLUTION * (extent if extent > 0.0 else 1.0)
    binned = np.round(nodes / resolution).astype(np.int64)
    digest = hashlib.sha256()
    for array in (owner, neighbour, binned):
        digest.update(repr(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()
