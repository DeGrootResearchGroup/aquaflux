"""Bridge the face-based mesh storage to VTK cell connectivity. Pure: no filesystem.

An aquaflux mesh stores nodes, a face->cell incidence (owner / neighbour) and ragged
compressed-sparse-row (CSR) face->node rings. Cells are *implicit* -- a cell is the set of faces
that reference it -- and there is no stored cell->vertex list. VTK's unstructured grid has two cell
types that take exactly that shape:

- **``VTK_POLYHEDRON`` (type 42)**, in three dimensions, is defined *by its face-node lists*, which
  is what the mesh already holds. No reconstruction into standard element types is needed, and none
  would be reliable for a general polyhedron.
- **``VTK_POLYGON`` (type 7)**, in two dimensions, is the cell's ordered vertex ring -- a short walk
  around the cell's two-node edge faces.

This module turns a :class:`~aquaflux.mesh.Mesh` into the arrays those cell types are written from.
It is eager NumPy throughout: writing a file never happens inside a compiled solve, and the per-cell
face counts are ragged, so the index arithmetic is build-time work like the mesh assembly itself.

Winding
-------
VTK requires every polyhedron face to be listed **outward from the cell listing it**, so a face
shared by two cells appears in opposite orders in the two. Two separate facts decide the order here,
and conflating them writes a mesh whose faces point inward on roughly half the cells:

1. **The stored ring's own direction is arbitrary.** A mesh accepts a face's nodes in either
   direction around the perimeter and never reorders them; what it orients owner-outward is the
   *normal*, computed from that ring and then flipped, as a separate array. So "as stored" is not a
   synonym for "owner-outward" -- measured on a structured hexahedral grid, 20 of 36 face rings wind
   owner-outward and the rest do not. :func:`stored_ring_is_outward` recovers which is which, by
   asking the mesh's own face-geometry strategy for the flip it applies.
2. **Owner-outward is neighbour-inward.** Having established the ring's direction relative to the
   owner, a cell listing the face as its *neighbour* needs the opposite one.

The ring is therefore reversed exactly when those two disagree: an owner-outward ring listed under
the neighbour, or an owner-inward ring listed under the owner. A boundary face is listed by its
owner only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from aquaflux.mesh.cell import CellGeometry
from aquaflux.mesh.connectivity import index_dtype
from aquaflux.mesh.face import face_geometry_scheme
from aquaflux.vectors import dot

if TYPE_CHECKING:  # pragma: no cover
    from aquaflux.mesh import Mesh

#: VTK cell type ids. A two-dimensional cell is an arbitrary polygon, a three-dimensional one an
#: arbitrary polyhedron -- the two types that are defined by exactly what the mesh stores.
VTK_POLYGON = 7
VTK_POLYHEDRON = 42


class VtkCells(NamedTuple):
    """The connectivity arrays of one VTK unstructured-grid piece.

    Attributes
    ----------
    points : np.ndarray
        Node coordinates, shape ``(n_nodes, 3)``. VTK points always carry three components, so a
        two-dimensional mesh is padded with ``z = 0`` (see :func:`build_vtk_cells` for why that
        padding is the *last* axis regardless of which axis a two-dimensional case was collapsed
        along).
    connectivity : np.ndarray of int
        Each cell's point ids, concatenated. For a polygon these are the ordered vertex ring; for a
        polyhedron they are the cell's point *set* (ascending), since its topology is carried by
        :attr:`faces` instead.
    offsets : np.ndarray of int, shape ``(n_cells,)``
        The **end** index of each cell's slice of :attr:`connectivity` -- the convention the VTK XML
        format uses, which is a cumulative count rather than the ``n + 1`` row pointer a CSR array
        would carry.
    types : np.ndarray of uint8, shape ``(n_cells,)``
        The VTK cell type per cell: :data:`VTK_POLYGON` or :data:`VTK_POLYHEDRON`.
    faces : np.ndarray of int or None
        The polyhedron face stream, three dimensions only (``None`` in two). Per cell: the number of
        faces, then for each face its node count followed by that face's point ids, wound outward
        from this cell. The ids are **global point ids**, not positions within
        :attr:`connectivity`.
    face_offsets : np.ndarray of int or None
        The end index of each cell's slice of :attr:`faces`, same convention as :attr:`offsets`
        (``None`` in two dimensions). Written under the name ``faceoffsets``.
    """

    points: np.ndarray
    connectivity: np.ndarray
    offsets: np.ndarray
    types: np.ndarray
    faces: np.ndarray | None
    face_offsets: np.ndarray | None

    @property
    def n_points(self) -> int:
        """Number of points."""
        return self.points.shape[0]

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return self.types.shape[0]


def stored_ring_is_outward(mesh: Mesh) -> np.ndarray:
    """Whether each face's stored node ring already winds outward from its owner cell.

    A mesh does not normalize the direction a face's nodes are listed in: it computes the normal
    that ring implies and then flips *the normal* to point out of the owner, leaving the ring alone.
    Recovering the flip is therefore the only way to know which direction the stored ring runs, and
    it is recovered by running the mesh's own face-geometry strategy rather than by re-deriving the
    orientation test here.

    Parameters
    ----------
    mesh : Mesh
        The mesh whose face rings are in question.

    Returns
    -------
    np.ndarray of bool, shape ``(n_faces,)``
        ``True`` where the stored ring is owner-outward. A degenerate face, whose ring implies no
        normal at all, reports ``True``; its winding cannot be determined and does not matter,
        because a face of zero area bounds no volume.
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


class _CellFaceEntries(NamedTuple):
    """The (cell, face) incidences, grouped by cell -- every cell paired with each face bounding it.

    An interior face produces two entries (one per incident cell) and a boundary face one. Sorted by
    cell, so a cell's entries are contiguous and the group sizes are a plain count.

    Attributes
    ----------
    cell : np.ndarray of int, shape ``(n_entries,)``
        The cell of each entry, non-decreasing.
    face : np.ndarray of int, shape ``(n_entries,)``
        The face of each entry.
    reversed_ring : np.ndarray of bool, shape ``(n_entries,)``
        Whether this entry must list the face's stored node ring backwards to wind outward from
        *its* cell.
    counts : np.ndarray of int, shape ``(n_cells,)``
        Number of faces bounding each cell.
    """

    cell: np.ndarray
    face: np.ndarray
    reversed_ring: np.ndarray
    counts: np.ndarray

    @property
    def first(self) -> np.ndarray:
        """Index of each cell's first entry, shape ``(n_cells,)``."""
        return np.cumsum(self.counts) - self.counts


def _cell_face_entries(mesh: Mesh) -> _CellFaceEntries:
    """Group faces by the cells they bound, carrying each entry's ring direction."""
    face_cells = mesh.face_cells
    owner = np.asarray(face_cells.owner)
    neighbour = np.asarray(face_cells.neighbour)
    interior = np.flatnonzero(np.asarray(face_cells.interior))

    face = np.concatenate([np.arange(owner.shape[0], dtype=interior.dtype), interior])
    cell = np.concatenate([owner, neighbour[interior]])
    by_owner = np.concatenate(
        [np.ones(owner.shape[0], dtype=bool), np.zeros(interior.shape[0], dtype=bool)]
    )

    order = np.argsort(cell, kind="stable")
    cell, face, by_owner = cell[order], face[order], by_owner[order]
    # Outward for this entry when the ring's own direction and the side listing it agree: an
    # owner-outward ring under the owner, or an owner-inward ring under the neighbour.
    outward = stored_ring_is_outward(mesh)[face] == by_owner
    counts = np.bincount(cell, minlength=mesh.n_cells)
    return _CellFaceEntries(cell, face, ~outward, counts)


class _EntryRings(NamedTuple):
    """Every entry's face-node ring, already reversed where the entry needs it, flattened.

    Attributes
    ----------
    node : np.ndarray of int, shape ``(n_ring_slots,)``
        The node ids of every entry's ring, concatenated in entry order.
    entry : np.ndarray of int, shape ``(n_ring_slots,)``
        Which entry each slot belongs to.
    position : np.ndarray of int, shape ``(n_ring_slots,)``
        The slot's position within its entry's ring.
    counts : np.ndarray of int, shape ``(n_entries,)``
        Nodes per entry.
    """

    node: np.ndarray
    entry: np.ndarray
    position: np.ndarray
    counts: np.ndarray


def _entry_rings(
    mesh: Mesh, entries: _CellFaceEntries, index: type[np.signedinteger]
) -> _EntryRings:
    """Read each entry's face ring out of the CSR store, reversing it where the entry says to.

    ``index`` is the integer width every array built here is formed at -- see
    :func:`build_vtk_cells`, which chooses it once for the whole reconstruction.
    """
    face_nodes = mesh.face_nodes
    ring_start = np.asarray(face_nodes.offsets)[:-1].astype(index)
    indices = np.asarray(face_nodes.face_node_indices)

    counts = np.asarray(face_nodes.counts)[entries.face].astype(index)
    entry = np.repeat(np.arange(counts.shape[0], dtype=index), counts)
    position = (
        np.arange(int(counts.sum()), dtype=index) - (np.cumsum(counts, dtype=index) - counts)[entry]
    )
    # A reversed ring reads its own slots back to front; the CSR slice it reads from is the same.
    source = np.where(entries.reversed_ring[entry], counts[entry] - 1 - position, position)
    return _EntryRings(indices[ring_start[entries.face][entry] + source], entry, position, counts)


def _polyhedron_faces(
    entries: _CellFaceEntries,
    rings: _EntryRings,
    n_cells: int,
    index: type[np.signedinteger],
) -> tuple[np.ndarray, np.ndarray]:
    """The ``faces`` / ``faceoffsets`` stream: per cell, its face count then each face's ring.

    ``index`` is the integer width the stream and its offsets are formed at (see
    :func:`build_vtk_cells`).
    """
    entry_block = rings.counts + 1  # the face's node count, then its nodes
    cell_block = np.bincount(entries.cell, weights=entry_block, minlength=n_cells).astype(index)
    cell_block += 1  # the cell's face count leads its block
    face_offsets = np.cumsum(cell_block, dtype=index)
    cell_start = face_offsets - cell_block

    # Entries are grouped by cell, so an entry's offset inside its cell's block is the running sum
    # of block lengths since that cell's first entry.
    within = np.cumsum(entry_block, dtype=index) - entry_block
    entry_start = cell_start[entries.cell] + 1 + (within - within[entries.first][entries.cell])

    stream = np.empty(int(face_offsets[-1]) if n_cells else 0, dtype=index)
    stream[cell_start] = entries.counts
    stream[entry_start] = rings.counts
    stream[entry_start[rings.entry] + 1 + rings.position] = rings.node
    return stream, face_offsets


def _polyhedron_connectivity(
    entries: _CellFaceEntries, rings: _EntryRings, n_cells: int, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Each cell's point *set*, ascending: a polyhedron's topology is in its face stream instead."""
    # One integer key per (cell, node) incidence makes the per-cell de-duplication a single global
    # sort, rather than a Python loop over cells.
    key = np.unique(entries.cell[rings.entry].astype(np.int64) * n_nodes + rings.node)
    connectivity = key % n_nodes
    counts = np.bincount(key // n_nodes, minlength=n_cells)
    return connectivity, np.cumsum(counts)


def _polygon_connectivity(
    entries: _CellFaceEntries, rings: _EntryRings, n_cells: int, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Each cell's ordered vertex ring, walked around its two-node edge faces.

    Every entry's ring is already wound outward from its own cell, so the edges of one cell form a
    single closed *directed* cycle: each edge ends where exactly one other begins. Chaining them is
    therefore a permutation chase -- ``n`` vectorized steps for a mesh whose largest cell has ``n``
    edges -- and needs no geometric tie-break.
    """
    start, end = rings.node[0::2], rings.node[1::2]
    # Within a cell each node starts exactly one edge, so (cell, start node) is a unique key and a
    # sorted search on it answers "which edge continues this one".
    key = entries.cell.astype(np.int64) * n_nodes + start
    order = np.argsort(key)
    sorted_key = key[order]
    wanted = entries.cell.astype(np.int64) * n_nodes + end
    found = np.searchsorted(sorted_key, wanted)
    if start.shape[0] and (
        np.any(found >= sorted_key.shape[0])
        or np.any(sorted_key[np.minimum(found, sorted_key.shape[0] - 1)] != wanted)
    ):
        raise ValueError(
            "a 2D cell's edges do not form a closed ring: an edge ends at a node no other edge of "
            "the same cell starts from, so the mesh's face->cell incidence is inconsistent"
        )
    following = order[found]

    width = int(entries.counts.max()) if n_cells else 0
    ring = np.zeros((n_cells, width), dtype=start.dtype)
    at = entries.first.copy()
    # A permutation always returns to where it started; what makes the cell a simple polygon is that
    # it takes *every* one of the cell's edges to get back. Returning sooner means the edges split
    # into two or more rings -- which a walk of a fixed number of steps would otherwise traverse
    # twice and report as a plausible ring.
    returned_after = np.zeros(n_cells, dtype=np.int64)
    for step in range(width):
        ring[:, step] = start[at]
        at = following[at]
        returned_after = np.where(
            (returned_after == 0) & (at == entries.first), step + 1, returned_after
        )
    if n_cells and np.any(returned_after != entries.counts):
        raise ValueError(
            "a 2D cell's edges form more than one closed ring, so the cell is not a simple polygon"
        )
    valid = np.arange(width) < entries.counts[:, None]
    return ring[valid], np.cumsum(entries.counts)


def build_vtk_cells(mesh: Mesh) -> VtkCells:
    """Reconstruct VTK cell connectivity from a mesh's face-based storage.

    Parameters
    ----------
    mesh : Mesh
        The mesh, two- or three-dimensional.

    Returns
    -------
    VtkCells
        Points and the connectivity arrays for one unstructured-grid piece.

    Raises
    ------
    ValueError
        If a two-dimensional cell's edges do not form exactly one closed ring.

    Notes
    -----
    A two-dimensional mesh's points are padded with a third coordinate of zero. The axis a
    two-dimensional case was originally extruded along is deliberately **not** consulted: the
    coordinates written here are the mesh's own two, laid in the plane ``z = 0``, so a vector field
    written beside them must be padded on the same last axis to stay aligned with the geometry.
    Restoring the original axis would put the geometry in one plane and the vectors in another.
    """
    coords = np.asarray(mesh.node_coords, dtype=float)
    points = coords if mesh.dim == 3 else np.pad(coords, ((0, 0), (0, 1)))

    entries = _cell_face_entries(mesh)
    n_cells, n_nodes = mesh.n_cells, mesh.n_nodes
    # One width for every index array built below. The face stream is the largest thing here -- a
    # cell's face count, then each face's node count and its nodes -- so its length bounds every
    # offset, and the node ids it holds are bounded by the node count. On a mesh of a few million
    # cells this choice is the difference between a few hundred megabytes and a gigabyte or more.
    ring_slots = int(np.asarray(mesh.face_nodes.counts)[entries.face].sum())
    index = index_dtype(max(n_cells + entries.cell.shape[0] + ring_slots, n_nodes, 1))

    rings = _entry_rings(mesh, entries, index)
    if mesh.dim == 3:
        connectivity, offsets = _polyhedron_connectivity(entries, rings, n_cells, n_nodes)
        faces, face_offsets = _polyhedron_faces(entries, rings, n_cells, index)
        cell_type = VTK_POLYHEDRON
    else:
        connectivity, offsets = _polygon_connectivity(entries, rings, n_cells, n_nodes)
        faces, face_offsets = None, None
        cell_type = VTK_POLYGON

    types = np.full(n_cells, cell_type, dtype=np.uint8)
    return VtkCells(points, connectivity, offsets, types, faces, face_offsets)
