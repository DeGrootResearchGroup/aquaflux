"""Hand-built ``PolyMeshData`` fixtures for testing the OpenFOAM assembler in isolation.

Two unit cubes sharing one interior face at ``x = 1`` — the smallest mesh with both an interior
face and boundary patches — plus a cyclic-patch variant of it, and a one-cell-thick structured
slab whose x = 0 / x = lx planes are a raw (un-fused) ``cyclic`` pair. Built directly as the
parsed record so the assembler can be tested without any files or text parsing.
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


def cyclic_two_cube_polymesh_data() -> PolyMeshData:
    """The two-cube mesh with ``inlet``/``outlet`` re-declared as a matched ``cyclic`` pair.

    Same points, faces, owners and ``walls`` patch as :func:`two_cube_polymesh_data` — only the
    two x-normal boundary patches change type, so fusing them should reproduce a mesh whose sole
    interior face carries a ``+2`` periodic-image offset along x (the two cubes wrapped end to
    end), leaving ``walls`` untouched.
    """
    return two_cube_polymesh_data()._replace(
        patches=(
            FoamPatch("inlet", "cyclic", 1, 1, "outlet"),
            FoamPatch("outlet", "cyclic", 2, 1, "inlet"),
            FoamPatch("walls", "wall", 3, 8),
        )
    )


def cyclic_slab_polymesh_data(
    nx: int, ny: int, lx: float = 2.0, ly: float = 1.0, lz: float = 0.4
) -> PolyMeshData:
    """A one-cell-thick 3D slab as raw, un-fused polyMesh data with a ``cyclic`` patch pair.

    The x = 0 and x = lx planes are two separate boundary patches (``"left"`` / ``"right"``),
    declared cyclic and paired via ``neighbourPatch`` rather than pre-fused into interior faces —
    the shape of data :func:`~aquaflux.io.openfoam.assembler.assemble` has to recover the
    periodicity from, mirroring what a real OpenFOAM cyclic mesh's ``boundary`` file declares. The
    z = 0 / z = lz planes are a single ``"frontAndBack"`` ``empty`` patch, so assembling and then
    collapsing that axis should reproduce ``structured_grid_2d(nx, ny, periodic=("x",))``.
    """
    x, y, z = np.linspace(0.0, lx, nx + 1), np.linspace(0.0, ly, ny + 1), np.array([0.0, lz])

    def nid(i, j, k):  # node index (k slowest, i fastest), two node planes in z
        return (k * (ny + 1) + j) * (nx + 1) + i

    def cid(i, j):  # cell index; the layer is one cell thick, so there is no k
        return j * nx + i

    kk, jj, ii = np.meshgrid(np.arange(2), np.arange(ny + 1), np.arange(nx + 1), indexing="ij")
    coords = np.stack([x[ii].ravel(), y[jj].ravel(), z[kk].ravel()], axis=1)

    node_blocks: list[np.ndarray] = []
    owner_blocks: list[np.ndarray] = []
    neighbour_blocks: list[np.ndarray] = []  # interior faces only

    # Interior x-normal faces: i in [1, nx - 1] (only when nx >= 2 — a single column has none).
    if nx >= 2:
        fi, fj = (a.ravel() for a in np.meshgrid(np.arange(1, nx), np.arange(ny), indexing="ij"))
        node_blocks.append(
            np.stack(
                [nid(fi, fj, 0), nid(fi, fj + 1, 0), nid(fi, fj + 1, 1), nid(fi, fj, 1)], axis=1
            )
        )
        owner_blocks.append(cid(fi - 1, fj))
        neighbour_blocks.append(cid(fi, fj))

    # Interior y-normal faces: j in [1, ny - 1] (only when ny >= 2).
    if ny >= 2:
        fi, fj = (a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(1, ny), indexing="ij"))
        node_blocks.append(
            np.stack(
                [nid(fi, fj, 0), nid(fi + 1, fj, 0), nid(fi + 1, fj, 1), nid(fi, fj, 1)], axis=1
            )
        )
        owner_blocks.append(cid(fi, fj - 1))
        neighbour_blocks.append(cid(fi, fj))

    n_interior = sum(block.shape[0] for block in node_blocks)

    def boundary_block(face_nodes, cell) -> int:
        node_blocks.append(np.stack(face_nodes, axis=1))
        owner_blocks.append(cell)
        return cell.size

    start = n_interior
    fj = np.arange(ny)
    n = boundary_block(
        [nid(0, fj, 0), nid(0, fj + 1, 0), nid(0, fj + 1, 1), nid(0, fj, 1)], cid(0, fj)
    )
    left = FoamPatch("left", "cyclic", start, n, "right")
    start += n
    n = boundary_block(
        [nid(nx, fj, 0), nid(nx, fj + 1, 0), nid(nx, fj + 1, 1), nid(nx, fj, 1)], cid(nx - 1, fj)
    )
    right = FoamPatch("right", "cyclic", start, n, "left")
    start += n

    fi = np.arange(nx)
    n = boundary_block(
        [nid(fi, 0, 0), nid(fi + 1, 0, 0), nid(fi + 1, 0, 1), nid(fi, 0, 1)], cid(fi, 0)
    )
    bottom = FoamPatch("bottom", "wall", start, n)
    start += n
    n = boundary_block(
        [nid(fi, ny, 0), nid(fi + 1, ny, 0), nid(fi + 1, ny, 1), nid(fi, ny, 1)], cid(fi, ny - 1)
    )
    top = FoamPatch("top", "wall", start, n)
    start += n

    fi, fj, fk = (
        a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(2), indexing="ij")
    )
    n = boundary_block(
        [nid(fi, fj, fk), nid(fi + 1, fj, fk), nid(fi + 1, fj + 1, fk), nid(fi, fj + 1, fk)],
        cid(fi, fj),
    )
    front_and_back = FoamPatch("frontAndBack", "empty", start, n)

    all_nodes = np.concatenate(node_blocks, axis=0)
    n_faces = all_nodes.shape[0]
    neighbour_internal = (
        np.concatenate(neighbour_blocks).astype(np.int64)
        if neighbour_blocks
        else np.zeros(0, dtype=np.int64)
    )
    return PolyMeshData(
        points=coords,
        face_node_offsets=np.arange(n_faces + 1, dtype=np.int64) * 4,
        face_node_indices=all_nodes.ravel().astype(np.int64),
        owner=np.concatenate(owner_blocks).astype(np.int64),
        neighbour_internal=neighbour_internal,
        patches=(left, right, bottom, top, front_and_back),
        cell_zones=(),
    )
