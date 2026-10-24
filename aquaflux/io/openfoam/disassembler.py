"""Lay an aquaflux :class:`~aquaflux.mesh.Mesh` out as the arrays of an OpenFOAM polyMesh.

The inverse of :func:`.assembler.assemble`, and like it a pure function of its input, so the
ordering and winding rules below are testable without a filesystem. The format fixes three things
the mesh leaves free, and each is restored here:

* **Face order.** The interior faces come first, in *upper-triangular* order -- each face's owner
  below its neighbour, the faces sorted by owner and then by neighbour -- and the boundary faces
  follow, each patch's faces as one contiguous block, since a patch is recorded only as a start
  face and a count. The mesh keeps faces in whatever order they were built.
* **Which cell owns an interior face.** The lower-numbered one; the mesh accepts either.
* **Which way a face's nodes run.** Their right-hand normal points out of the owner (for a
  boundary face, out of the domain). The mesh stores a ring in either direction and orients its
  normal separately, so each ring's direction is recovered with
  :func:`~aquaflux.mesh.stored_ring_is_outward` and reversed where it disagrees.

**Cells are not renumbered.** Only faces move, so cell ``i`` of the mesh is cell ``i`` of the
written polyMesh, and a cell field written against it with
:func:`~aquaflux.io.openfoam.write_openfoam_time` needs no permutation.
"""

from __future__ import annotations

import numpy as np

from aquaflux import ragged
from aquaflux.mesh import RESERVED_PATCH_NAMES, Mesh, stored_ring_is_outward

from .records import CellZone, FoamPatch, PolyMeshData

#: The type a patch is written with when its mesh declared none.
DEFAULT_PATCH_TYPE = "patch"


def disassemble(mesh: Mesh) -> PolyMeshData:
    """The polyMesh arrays of a three-dimensional mesh, in the order and winding the format requires.

    Parameters
    ----------
    mesh : Mesh
        A three-dimensional, non-periodic mesh whose boundary faces all belong to named patches and
        whose named patches hold boundary faces only.

    Returns
    -------
    PolyMeshData
        The points, the faces in upper-triangular order with every ring owner-outward, owner, the
        interior neighbours, one boundary patch per named patch (in the mesh's patch order, typed
        as declared, or :data:`DEFAULT_PATCH_TYPE`, with its groups), and one cell zone per named
        zone. :func:`.assembler.assemble` of the result rebuilds the mesh, with its faces reordered
        and its cells unchanged.

    Raises
    ------
    ValueError
        If the mesh is two-dimensional (a polyMesh is always three-dimensional; a two-dimensional
        case is a one-cell-thick extrusion, which this does not build), if some boundary face is in
        no named patch (the format has no unnamed boundary), or if a named patch holds an interior
        face (which the format records as a face zone, not written here).
    NotImplementedError
        If the mesh is periodic: its seam would be written as a pair of ``cyclic`` patches, which is
        not built. Also if a patch is declared ``cyclic``, which needs that pairing too.
    """
    _check_writable(mesh)
    owner = np.asarray(mesh.face_cells.owner, dtype=np.int64)
    neighbour = np.asarray(mesh.face_cells.neighbour, dtype=np.int64)
    interior = neighbour >= 0

    swap = interior & (owner > neighbour)
    owner, neighbour = np.where(swap, neighbour, owner), np.where(swap, owner, neighbour)
    # A ring outward from the old owner must be reversed when it is kept, and kept when the
    # face changes hands: then the old neighbour, which it points into, is the new owner.
    reverse = stored_ring_is_outward(mesh) == swap

    patches = mesh.face_patches
    label = np.asarray(patches.label)
    named = [name for name in patches.names if name not in RESERVED_PATCH_NAMES]
    interior_faces = np.flatnonzero(interior)
    interior_faces = interior_faces[np.lexsort((neighbour[interior_faces], owner[interior_faces]))]
    blocks = [np.flatnonzero(label == patches.id_of(name)) for name in named]
    order = np.concatenate([interior_faces, *blocks])

    offsets = np.asarray(mesh.face_nodes.offsets, dtype=np.int64)
    indices = np.asarray(mesh.face_nodes.face_node_indices, dtype=np.int64)
    flat, _, counts = ragged.rows(offsets, indices, order, reverse[order])

    starts = interior_faces.size + np.concatenate([[0], np.cumsum([b.size for b in blocks])])
    types = dict(patches.patch_types)
    foam_patches = tuple(
        FoamPatch(
            name=name,
            type_=types.get(name, DEFAULT_PATCH_TYPE),
            start_face=int(start),
            n_faces=int(block.size),
            in_groups=tuple(g for g, members in patches.patch_groups if name in members),
        )
        for name, start, block in zip(named, starts[:-1], blocks, strict=True)
    )
    zones = mesh.cell_zones
    return PolyMeshData(
        points=np.asarray(mesh.node_coords, dtype=np.float64),
        face_node_offsets=np.concatenate([[0], np.cumsum(counts)]),
        face_node_indices=flat,
        owner=owner[order],
        neighbour_internal=neighbour[interior_faces],
        patches=foam_patches,
        cell_zones=tuple(
            CellZone(name=name, cell_labels=np.flatnonzero(np.asarray(zones.mask(name))))
            for name in zones.names[1:]
        ),
    )


def _check_writable(mesh: Mesh) -> None:
    """Refuse a mesh the polyMesh format cannot hold as written, naming what is in the way."""
    if mesh.dim != 3:
        raise ValueError(
            f"a polyMesh is three-dimensional; this mesh has dim {mesh.dim}, and writing it would "
            "need an extrusion one cell thick between two empty patches, which is not built"
        )
    if mesh.face_cells.neighbour_offset is not None:
        raise NotImplementedError(
            "this mesh is periodic; its seam would be written as a pair of cyclic patches, which "
            "is not built"
        )
    patches = mesh.face_patches
    cyclic = [name for name, kind in patches.patch_types if kind == "cyclic"]
    if cyclic:
        raise NotImplementedError(
            f"patch(es) {cyclic} are declared cyclic, which needs a paired neighbour patch; "
            "writing cyclic pairs is not built"
        )
    unnamed = patches.uncovered_boundary_faces(
        [name for name in patches.names if name not in RESERVED_PATCH_NAMES],
        mesh.face_cells,
    )
    if unnamed:
        raise ValueError(
            f"{sum(unnamed.values())} boundary face(s) belong to no named patch; a polyMesh has no "
            "unnamed boundary, so put them in a named patch first"
        )
    label = np.asarray(patches.label)
    interior = np.asarray(mesh.face_cells.neighbour) >= 0
    on_interior = [
        name
        for name in patches.names
        if name not in RESERVED_PATCH_NAMES and np.any(interior & (label == patches.id_of(name)))
    ]
    if on_interior:
        raise ValueError(
            f"patch(es) {on_interior} hold interior faces; the format records those as a face zone, "
            "which is not written"
        )
