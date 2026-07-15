"""A hand-built two-cube ``PolyMeshData`` for testing the OpenFOAM assembler in isolation.

Two unit cubes sharing one interior face at ``x = 1`` — the smallest mesh with both an interior
face and boundary patches. Built directly as the parsed record so the assembler can be tested
without any files or text parsing.
"""

from __future__ import annotations

import numpy as np
from aquaflux.io.openfoam.records import CellZone, FoamPatch, PolyMeshData

# Node lattice: x in {0, 1, 2}, y in {0, 1}, z in {0, 1}; index = x + 3*y + 6*z.
_POINTS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [2, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [2, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [2, 1, 1],
    ],
    dtype=np.float64,
)

# Faces, interior first (OpenFOAM upper-triangular ordering), each in perimeter order.
_FACES = [
    [1, 4, 10, 7],  # 0: interior, x = 1 (owner 0 | neighbour 1)
    [0, 3, 9, 6],  # 1: inlet, x = 0
    [2, 5, 11, 8],  # 2: outlet, x = 2
    [0, 1, 7, 6],  # 3: wall, y = 0 (cell 0)
    [3, 4, 10, 9],  # 4: wall, y = 1 (cell 0)
    [0, 1, 4, 3],  # 5: wall, z = 0 (cell 0)
    [6, 7, 10, 9],  # 6: wall, z = 1 (cell 0)
    [1, 2, 8, 7],  # 7: wall, y = 0 (cell 1)
    [4, 5, 11, 10],  # 8: wall, y = 1 (cell 1)
    [1, 2, 5, 4],  # 9: wall, z = 0 (cell 1)
    [7, 8, 11, 10],  # 10: wall, z = 1 (cell 1)
]
_OWNER = np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
_NEIGHBOUR_INTERNAL = np.array([1], dtype=np.int64)


def two_cube_polymesh_data() -> PolyMeshData:
    """Return the two-cube mesh as a parsed :class:`~aquaflux.io.openfoam.records.PolyMeshData`.

    Boundary patches: ``inlet`` (x=0), ``outlet`` (x=2), ``walls`` (the eight y/z faces). Cell zones:
    ``left`` = cell 0, ``right`` = cell 1.
    """
    offsets = np.arange(len(_FACES) + 1, dtype=np.int64) * 4
    indices = np.array([node for face in _FACES for node in face], dtype=np.int64)
    return PolyMeshData(
        points=_POINTS,
        face_node_offsets=offsets,
        face_node_indices=indices,
        owner=_OWNER,
        neighbour_internal=_NEIGHBOUR_INTERNAL,
        patches=(
            FoamPatch("inlet", "patch", 1, 1),
            FoamPatch("outlet", "patch", 2, 1),
            FoamPatch("walls", "wall", 3, 8),
        ),
        cell_zones=(
            CellZone("left", np.array([0], dtype=np.int64)),
            CellZone("right", np.array([1], dtype=np.int64)),
        ),
    )
