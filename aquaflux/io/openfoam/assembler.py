"""Assemble parsed polyMesh arrays into an aquaflux :class:`~aquaflux.mesh.Mesh`.

The bridge between the text parsers and the mesh: it applies the OpenFOAM conventions that are
*semantic* rather than syntactic — padding the interior-only neighbour list with the boundary
sentinel, deriving the cell count, fusing matched ``cyclic`` patch pairs into periodic seam faces
(:mod:`.cyclic`), and mapping the surviving boundary patches and cell zones onto their aquaflux
names — then hands the result to :meth:`~aquaflux.mesh.Mesh.from_csr`, which owns all topological
validation. This is a pure function of the parsed record, so it is testable with a hand-built
:class:`~aquaflux.io.openfoam.records.PolyMeshData` and no files.
"""

from __future__ import annotations

import numpy as np

from aquaflux.mesh import Mesh
from aquaflux.mesh.groups import RESERVED_PATCH_NAMES

from .cyclic import DEFAULT_MATCH_TOLERANCE, fuse_cyclic_patches
from .records import CellZone, FoamPatch, PolyMeshData, patch_face_range


def assemble(
    data: PolyMeshData, *, cyclic_match_tolerance: float = DEFAULT_MATCH_TOLERANCE
) -> Mesh:
    """Assemble parsed polyMesh arrays into a validated 3D :class:`~aquaflux.mesh.Mesh`.

    Parameters
    ----------
    data : PolyMeshData
        The parsed points, CSR face connectivity, owner, interior-only neighbour, boundary patches
        and cell zones.
    cyclic_match_tolerance : float
        Passed to :func:`~aquaflux.io.openfoam.cyclic.fuse_cyclic_patches` for matching a
        ``cyclic`` patch pair's faces (see
        :data:`~aquaflux.io.openfoam.cyclic.DEFAULT_MATCH_TOLERANCE`).

    Returns
    -------
    Mesh
        The 3D mesh (a polyMesh is inherently three-dimensional). Collapsing a one-cell-thick
        two-dimensional case is a separate step in the reader. A matched ``cyclic`` patch pair is
        fused into interior periodic seam faces rather than kept as boundary faces.

    Raises
    ------
    ValueError
        If the mesh has no faces, the neighbour list is longer than the face count, a boundary
        patch reuses a reserved name, or a ``cyclic`` patch cannot be fused with its partner (see
        :func:`~aquaflux.io.openfoam.cyclic.fuse_cyclic_patches`). Topological problems (index
        ranges, degenerate faces, orphan cells) are raised by :meth:`~aquaflux.mesh.Mesh.from_csr`.
    """
    owner = np.asarray(data.owner)
    neighbour_internal = np.asarray(data.neighbour_internal)
    n_faces = owner.shape[0]
    if n_faces == 0:
        raise ValueError("polyMesh has no faces")
    if neighbour_internal.shape[0] > n_faces:
        raise ValueError(
            f"neighbour list ({neighbour_internal.shape[0]}) is longer than the face count "
            f"({n_faces})"
        )

    neighbour = _full_neighbour(neighbour_internal, n_faces)
    n_cells = _cell_count(owner, neighbour_internal)

    fused = fuse_cyclic_patches(
        data.points,
        data.face_node_offsets,
        data.face_node_indices,
        owner,
        neighbour,
        data.patches,
        match_tolerance=cyclic_match_tolerance,
    )

    return Mesh.from_csr(
        data.points,
        fused.face_node_offsets,
        fused.face_node_indices,
        fused.owner,
        fused.neighbour,
        n_cells=n_cells,
        cell_zones=_cell_zones_to_dict(data.cell_zones),
        face_patches=_patches_to_face_patches(fused.patches),
        neighbour_offset=fused.neighbour_offset,
    )


def _full_neighbour(neighbour_internal: np.ndarray, n_faces: int) -> np.ndarray:
    """Pad the interior-only neighbour list to full length with the boundary sentinel ``-1``.

    OpenFOAM orders faces so the interior faces come first, so the interior neighbours fill the
    front of the array and every boundary face after them gets ``-1``.
    """
    boundary = np.full(n_faces - neighbour_internal.shape[0], -1, dtype=np.int64)
    return np.concatenate([neighbour_internal.astype(np.int64), boundary])


def _cell_count(owner: np.ndarray, neighbour_internal: np.ndarray) -> int:
    """Number of cells: one past the highest cell index referenced by any face."""
    max_cell = int(owner.max())
    if neighbour_internal.size:
        max_cell = max(max_cell, int(neighbour_internal.max()))
    return max_cell + 1


def _patches_to_face_patches(patches: tuple[FoamPatch, ...]) -> dict[str, np.ndarray] | None:
    """Map boundary patches to ``{name: contiguous face indices}`` for the mesh constructor.

    Boundary faces not covered by any patch fall into aquaflux's automatic ``"boundary"`` patch.

    Raises
    ------
    ValueError
        If a patch reuses a reserved name (``"interior"`` / ``"boundary"``).
    """
    if not patches:
        return None
    collisions = [p.name for p in patches if p.name in RESERVED_PATCH_NAMES]
    if collisions:
        raise ValueError(
            f"boundary patch name(s) {collisions} collide with reserved aquaflux patch names "
            f"{list(RESERVED_PATCH_NAMES)}; rename them in the polyMesh boundary file"
        )
    return {p.name: patch_face_range(p) for p in patches}


def _cell_zones_to_dict(zones: tuple[CellZone, ...]) -> dict[str, np.ndarray] | None:
    """Map cell zones to ``{name: cell indices}`` for the mesh constructor (``None`` if none)."""
    if not zones:
        return None
    return {zone.name: np.asarray(zone.cell_labels) for zone in zones}
