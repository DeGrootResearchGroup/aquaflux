"""The polyhedral dual of a conformal mesh, keeping the boundary's features sharp.

Every point of a three-dimensional mesh (the *primal*) becomes one cell of the dual, and every edge
the face between its two end points' cells. The dual of a conformal mesh is conformal, and each
primal cell type has a dual of a useful kind -- a block of hexahedra gives hexahedra (shifted half a
cell), tetrahedra give general polyhedra -- so a mesh made of several kinds of cell, joined
conformally, has a dual whose pieces are joined conformally too, with no tetrahedra or pyramids left
in it. Dual cell ``i`` is the cell of primal point ``i``.

**Vertices of the dual.** The centroid of every primal cell; the centroid of every primal boundary
face; the midpoint of every *feature* edge; and every primal point on two or more feature edges.

**Faces of the dual.**

* For each primal edge, the face between the dual cells of its two ends. Its ring runs through the
  centroids of the cells around the edge, in order. Around an edge on the boundary the ring is not
  closed by cells; it starts and ends at the centroids of the edge's two boundary faces, and passes
  through the edge's midpoint when the edge is a feature.
* For each primal boundary point, its share of the boundary: the polygon through the centroids of
  the boundary faces around it (and the midpoints of the feature edges between them). Where the
  point lies on two or more feature edges the polygon is cut along them, one face per sector, each
  with the point itself as a vertex -- so a corner of the boundary stays a corner of the dual and an
  edge stays an edge. Every sector lies in one patch, because a patch's border is always a feature.

**Features** are the boundary edges where the boundary folds by more than an angle, every edge
between two patches, and any edges named explicitly (a mesher passes the edges of the geometry it was
built from). Away from them the dual's boundary runs through face centroids, inside a curved primal
boundary by the chord of a face.

**A sharp concave fold is refused, for now.** The dual cell of a point on a concave edge wraps around
the fold; on tetrahedra it can fold far enough that its approximate centroid lies beyond one of its
own faces, and the mesh -- which orients every face's normal away from that centroid -- then turns
the face inward. The standard remedy is to split such a cell into convex pieces, which is not built.
A gently concave wall (a lamp sleeve, seen from the water around it) folds by less than the feature
angle and is accepted.
"""

from __future__ import annotations

import dataclasses
from itertools import pairwise

import numba
import numpy as np

from aquaflux import ragged
from aquaflux.mesh import RESERVED_PATCH_NAMES, Mesh

#: The automatic patch names, which a primal's named patches are distinguished from.
_AUTOMATIC = set(RESERVED_PATCH_NAMES)

# Status codes of the compiled walks.
_OK, _NOT_MANIFOLD = 0, 1


@dataclasses.dataclass(frozen=True)
class PolyhedralDual:
    """Build the feature-preserving polyhedral dual of a three-dimensional mesh.

    Attributes
    ----------
    feature_angle : float
        The fold, in radians, beyond which a boundary edge is a feature: the angle between the
        outward normals of its two boundary faces. There is no default: it decides whether a
        curved wall is faceted or a corner is rounded off.
    feature_edges : np.ndarray of int, shape ``(n, 2)``, optional
        Further edges to keep, as pairs of primal point indices (in either order); each must be an
        edge of the primal boundary. Edges between two patches are features whatever is given.
    """

    feature_angle: float
    feature_edges: np.ndarray | None = None

    def build(self, primal: Mesh) -> Mesh:
        """The dual of ``primal``.

        Parameters
        ----------
        primal : Mesh
            A three-dimensional, non-periodic mesh with one cell zone and no patch on interior
            faces, whose boundary is a closed manifold surface (every boundary edge on exactly two
            boundary faces, every boundary point's faces one fan).

        Returns
        -------
        Mesh
            One cell per primal point, in the same order. The dual's boundary faces are in the
            patches of the primal faces they cover, with the same types and groups.

        Raises
        ------
        ValueError
            If the primal is not a mesh this construction accepts (see ``primal``), or a named
            feature edge is not a primal boundary edge.
        """
        _check_primal(primal)
        tables = _PrimalTables.of(primal)
        features = self._features(primal, tables)
        return _assemble(primal, tables, features)

    def _features(self, primal: Mesh, tables: _PrimalTables) -> np.ndarray:
        """Which boundary edges are features, shape ``(n_edges,)``.

        Raises
        ------
        NotImplementedError
            If the boundary folds concavely by more than :attr:`feature_angle` at some edge.
        """
        on_boundary = tables.boundary_faces[:, 0] >= 0
        fold = boundary_folds(primal, tables.boundary_faces[on_boundary])
        concave = fold < -self.feature_angle
        if np.any(concave):
            raise NotImplementedError(
                f"the boundary folds concavely by more than the feature angle at "
                f"{int(np.count_nonzero(concave))} edge(s) (sharpest {np.degrees(-fold.min()):.0f} "
                "degrees): a dual cell there wraps around the fold and is not convex enough for the "
                "mesh to orient its faces; splitting such cells is not built yet"
            )
        label = np.asarray(primal.face_patches.label)
        first, second = tables.boundary_faces[on_boundary, 0], tables.boundary_faces[on_boundary, 1]
        feature = np.zeros(tables.n_edges, dtype=bool)
        feature[on_boundary] = (np.abs(fold) > self.feature_angle) | (label[first] != label[second])
        if self.feature_edges is not None and len(self.feature_edges):
            named = np.sort(np.asarray(self.feature_edges, dtype=np.int64).reshape(-1, 2), axis=1)
            index = tables.edge_index(named)
            if np.any(index < 0) or not np.all(on_boundary[np.maximum(index, 0)]):
                raise ValueError("a named feature edge is not an edge of the primal boundary")
            feature[index] = True
        return feature


def boundary_folds(primal: Mesh, pairs: np.ndarray) -> np.ndarray:
    """The signed fold between pairs of boundary faces: positive convex, negative concave.

    Parameters
    ----------
    primal : Mesh
    pairs : np.ndarray of int, shape ``(n, 2)``
        Pairs of boundary faces sharing an edge.

    Returns
    -------
    np.ndarray, shape ``(n,)``
        The angle, in radians, between the two faces' outward normals, negative where the second
        face's centroid lies outside the first's plane (the domain is concave there).
    """
    geometry = primal.geometry().face
    normal, centroid = np.asarray(geometry.normal), np.asarray(geometry.centroid)
    first, second = pairs[:, 0], pairs[:, 1]
    cosine = np.clip(np.einsum("ij,ij->i", normal[first], normal[second]), -1.0, 1.0)
    rises = np.einsum("ij,ij->i", centroid[second] - centroid[first], normal[first]) > 0.0
    return np.where(rises, -1.0, 1.0) * np.arccos(cosine)


def _check_primal(primal: Mesh) -> None:
    """Refuse a primal mesh outside what the construction handles, naming the reason."""
    if primal.dim != 3:
        raise ValueError(
            f"the dual is built for a three-dimensional mesh; this one has dim {primal.dim}"
        )
    if primal.face_cells.neighbour_offset is not None:
        raise ValueError("the dual of a periodic mesh is not built")
    if primal.cell_zones.n_groups > 1:
        raise ValueError(
            "the primal has several cell zones; a point on a zone border would need its dual cell "
            "split, which is not built"
        )
    patches = primal.face_patches
    label = np.asarray(patches.label)
    interior = np.asarray(primal.face_cells.neighbour) >= 0
    baffles = [
        name
        for name in patches.names
        if name not in _AUTOMATIC and np.any(interior & (label == patches.id_of(name)))
    ]
    if baffles:
        raise ValueError(f"patch(es) {baffles} hold interior faces, which the dual does not keep")
    used = np.zeros(primal.n_nodes, dtype=bool)
    used[np.asarray(primal.face_nodes.face_node_indices)] = True
    if not used.all():
        raise ValueError(
            f"{int((~used).sum())} primal point(s) are on no face, so have no dual cell"
        )


@dataclasses.dataclass(frozen=True)
class _PrimalTables:
    """The primal's edges and the incidences the walks need, as arrays."""

    edge_nodes: np.ndarray  # (n_edges, 2), ascending
    incidence_edge: np.ndarray  # (n_incidences,)
    edge_face_offsets: np.ndarray  # CSR, edge -> faces
    edge_faces: np.ndarray
    boundary_faces: np.ndarray  # (n_edges, 2): an edge's two boundary faces, or -1
    previous_incidence: np.ndarray  # (n_incidences,)

    @property
    def n_edges(self) -> int:
        return self.edge_nodes.shape[0]

    def edge_index(self, pairs: np.ndarray) -> np.ndarray:
        """The index of each ascending node pair among the edges, or -1."""
        bound = int(self.edge_nodes.max()) + 1
        keys = self.edge_nodes[:, 0] * bound + self.edge_nodes[:, 1]
        wanted = pairs[:, 0] * bound + pairs[:, 1]
        position = np.searchsorted(keys, wanted)
        position = np.minimum(position, keys.size - 1)
        return np.where(keys[position] == wanted, position, -1)

    @classmethod
    def of(cls, primal: Mesh) -> _PrimalTables:
        nodes = primal.face_nodes
        edge_nodes, incidence_edge = nodes.edges()
        face_of = np.asarray(nodes.face_of_incidence, dtype=np.int64)
        offsets, order = ragged.group(incidence_edge, edge_nodes.shape[0])
        edge_faces = face_of[order]
        boundary = np.asarray(primal.face_cells.neighbour)[edge_faces] < 0
        n_edges = edge_nodes.shape[0]
        counts = np.bincount(
            np.repeat(np.arange(n_edges), np.diff(offsets))[boundary], minlength=n_edges
        )
        if np.any((counts != 0) & (counts != 2)):
            raise ValueError(
                f"{int(np.count_nonzero((counts != 0) & (counts != 2)))} boundary edge(s) are on "
                "other than two boundary faces: the boundary is not a closed manifold surface"
            )
        pairs = np.full((n_edges, 2), -1, dtype=np.int64)
        on = np.flatnonzero(counts == 2)
        found = (
            edge_faces[boundary].reshape(-1, 2) if boundary.any() else np.zeros((0, 2), np.int64)
        )
        pairs[on] = found
        next_incidence = np.asarray(nodes.next_incidence, dtype=np.int64)
        previous = np.empty_like(next_incidence)
        previous[next_incidence] = np.arange(next_incidence.size)
        return cls(edge_nodes, incidence_edge, offsets, edge_faces, pairs, previous)


@numba.njit(cache=True)
def _walk_edges(edge_face_offsets, edge_faces, owner, neighbour, ring_offsets, ring_cells, ends):
    """Order the cells around every edge, and find a boundary edge's first and last faces.

    For an edge inside the domain the faces and cells around it form a cycle; for one on the
    boundary, a chain from one boundary face to the other. ``ring_cells[ring_offsets[e]:...]``
    receives the cells in order and ``ends[e]`` the chain's two boundary faces (``-1`` inside).
    """
    for edge in range(edge_face_offsets.size - 1):
        first, last = edge_face_offsets[edge], edge_face_offsets[edge + 1]
        start = -1
        for k in range(first, last):
            if neighbour[edge_faces[k]] < 0:
                start = edge_faces[k]
                break
        if start < 0:
            start = edge_faces[first]
        cell = owner[start]
        previous = start
        out = ring_offsets[edge]
        ring_cells[out] = cell
        filled = 1
        ends[edge, 0] = start if neighbour[start] < 0 else -1
        ends[edge, 1] = -1
        for _ in range(last - first):
            following = -1
            for k in range(first, last):
                face = edge_faces[k]
                if face != previous and (owner[face] == cell or neighbour[face] == cell):
                    following = face
                    break
            if following < 0:
                return _NOT_MANIFOLD
            if neighbour[following] < 0:
                ends[edge, 1] = following
                break
            if following == start:
                break
            cell = neighbour[following] if owner[following] == cell else owner[following]
            if filled >= ring_offsets[edge + 1] - out:
                return _NOT_MANIFOLD
            ring_cells[out + filled] = cell
            filled += 1
            previous = following
        if filled != ring_offsets[edge + 1] - out:
            return _NOT_MANIFOLD
    return _OK


@numba.njit(cache=True)
def _walk_points(
    point_offsets, point_incidences, face_offsets, face_nodes, incidence_edge,
    previous_incidence, face_of_incidence, boundary_faces, fan_edges, fan_faces,
):  # fmt: skip
    """Order the boundary edges and faces around every boundary point.

    For point ``p`` the boundary faces containing it form one fan; ``fan_faces`` receives them in
    order and ``fan_edges`` the boundary edge before each (between it and the previous face), both
    in ``point_offsets`` rows.
    """
    for point in range(point_offsets.size - 1):
        first, last = point_offsets[point], point_offsets[point + 1]
        if first == last:
            continue
        incidence = point_incidences[first]
        face = face_of_incidence[incidence]
        entering = incidence_edge[previous_incidence[incidence]]
        start_face = face
        for k in range(last - first):
            fan_faces[first + k] = face
            fan_edges[first + k] = entering
            leaving = -1
            for j in range(face_offsets[face], face_offsets[face + 1]):
                if face_nodes[j] == point:
                    a, b = incidence_edge[j], incidence_edge[previous_incidence[j]]
                    leaving = b if a == entering else a
                    break
            if leaving < 0:
                return _NOT_MANIFOLD
            pair = boundary_faces[leaving]
            face = pair[1] if pair[0] == face else pair[0]
            entering = leaving
            # The fan closes on its first face exactly when every face around the point is in it.
            if (face == start_face) != (k == last - first - 1):
                return _NOT_MANIFOLD
    return _OK


def _assemble(primal: Mesh, tables: _PrimalTables, feature: np.ndarray) -> Mesh:
    """Number the dual's vertices, build its faces, and hand them to ``Mesh.from_csr``."""
    geometry = primal.geometry()
    owner = np.asarray(primal.face_cells.owner, dtype=np.int64)
    neighbour = np.asarray(primal.face_cells.neighbour, dtype=np.int64)
    points = np.asarray(primal.node_coords, dtype=np.float64)
    n_cells, n_points = primal.n_cells, primal.n_nodes
    boundary_face = np.flatnonzero(neighbour < 0)
    boundary_rank = np.full(primal.n_faces, -1, dtype=np.int64)
    boundary_rank[boundary_face] = np.arange(boundary_face.size)

    # The walks.
    on_boundary = tables.boundary_faces[:, 0] >= 0
    n_around = np.diff(tables.edge_face_offsets)
    ring_offsets = np.concatenate([[0], np.cumsum(n_around - on_boundary)])
    ring_cells = np.empty(ring_offsets[-1], dtype=np.int64)
    ends = np.empty((tables.n_edges, 2), dtype=np.int64)
    status = _walk_edges(
        tables.edge_face_offsets,
        tables.edge_faces,
        owner,
        neighbour,
        ring_offsets,
        ring_cells,
        ends,
    )
    if status != _OK:
        raise ValueError(
            "the cells around some edge do not form one fan: the primal is not manifold"
        )

    face_offsets = np.asarray(primal.face_nodes.offsets, dtype=np.int64)
    face_nodes = np.asarray(primal.face_nodes.face_node_indices, dtype=np.int64)
    face_of = np.asarray(primal.face_nodes.face_of_incidence, dtype=np.int64)
    boundary_incidences = np.flatnonzero(neighbour[face_of] < 0)
    point_offsets, order = ragged.group(face_nodes[boundary_incidences], n_points)
    point_incidences = boundary_incidences[order]
    fan_edges = np.empty(point_incidences.size, dtype=np.int64)
    fan_faces = np.empty(point_incidences.size, dtype=np.int64)
    status = _walk_points(
        point_offsets, point_incidences, face_offsets, face_nodes, tables.incidence_edge,
        tables.previous_incidence, face_of, tables.boundary_faces, fan_edges, fan_faces,
    )  # fmt: skip
    if status != _OK:
        raise ValueError(
            "the boundary faces around some point do not form one fan: the boundary is not manifold"
        )

    # Dual vertices: cell centroids, boundary-face centroids, feature-edge midpoints, feature points.
    feature_edge = np.flatnonzero(feature)
    feature_rank = np.full(tables.n_edges, -1, dtype=np.int64)
    feature_rank[feature_edge] = np.arange(feature_edge.size)
    on_features = np.bincount(tables.edge_nodes[feature_edge].reshape(-1), minlength=n_points)
    corner = np.flatnonzero(on_features >= 2)
    corner_rank = np.full(n_points, -1, dtype=np.int64)
    corner_rank[corner] = np.arange(corner.size)
    face_base = n_cells
    middle_base = face_base + boundary_face.size
    corner_base = middle_base + feature_edge.size
    vertices = np.concatenate(
        [
            np.asarray(geometry.cell.centroid, dtype=np.float64),
            np.asarray(geometry.face.centroid, dtype=np.float64)[boundary_face],
            points[tables.edge_nodes[feature_edge]].mean(axis=1),
            points[corner],
        ]
    )

    # One face per primal edge, between its two ends' cells: [first boundary face, the cells around,
    # last boundary face, the midpoint] on the boundary, the cells alone inside.
    lead = on_boundary.astype(np.int64)
    n_ring = np.diff(ring_offsets)
    edge_counts = n_ring + 2 * lead + feature.astype(np.int64)
    edge_starts = np.concatenate([[0], np.cumsum(edge_counts)[:-1]])
    edge_rings = np.empty(int(edge_counts.sum()), dtype=np.int64)
    of_cell = np.repeat(np.arange(tables.n_edges), n_ring)
    position = np.arange(ring_cells.size) - np.repeat(ring_offsets[:-1], n_ring)
    edge_rings[edge_starts[of_cell] + lead[of_cell] + position] = ring_cells
    bounded = np.flatnonzero(on_boundary)
    edge_rings[edge_starts[bounded]] = face_base + boundary_rank[ends[bounded, 0]]
    edge_rings[edge_starts[bounded] + 1 + n_ring[bounded]] = (
        face_base + boundary_rank[ends[bounded, 1]]
    )
    edge_rings[edge_starts[feature_edge] + 2 + n_ring[feature_edge]] = (
        middle_base + feature_rank[feature_edge]
    )
    rings: list[np.ndarray] = []
    counts_parts: list[np.ndarray] = [edge_counts]
    dual_owner, dual_neighbour, dual_patch = (
        [tables.edge_nodes[:, 0]],
        [tables.edge_nodes[:, 1]],
        [],
    )
    dual_patch.append(np.full(tables.n_edges, -1, dtype=np.int64))

    # Each boundary point's share of the boundary, cut along the feature edges through it.
    label = np.asarray(primal.face_patches.label)
    sector_owner, sector_patch = [], []
    for point in np.flatnonzero(np.diff(point_offsets)):
        edges = fan_edges[point_offsets[point] : point_offsets[point + 1]]
        faces = fan_faces[point_offsets[point] : point_offsets[point + 1]]
        cuts = np.flatnonzero(feature[edges])
        if cuts.size == 0:
            rings.append(face_base + boundary_rank[faces])
            sector_owner.append(point)
            sector_patch.append(label[faces[0]])
            continue
        shift = cuts[0]
        edges, faces = np.roll(edges, -shift), np.roll(faces, -shift)
        cuts = cuts - shift
        if cuts.size == 1:
            rings.append(
                np.concatenate(
                    [[middle_base + feature_rank[edges[0]]], face_base + boundary_rank[faces]]
                )
            )
            sector_owner.append(point)
            sector_patch.append(label[faces[0]])
            continue
        bounds = [*cuts.tolist(), edges.size]
        for begin, end in pairwise(bounds):
            closing = edges[end % edges.size]
            rings.append(
                np.concatenate(
                    [
                        [
                            corner_base + corner_rank[point],
                            middle_base + feature_rank[edges[begin]],
                        ],
                        face_base + boundary_rank[faces[begin:end]],
                        [middle_base + feature_rank[closing]],
                    ]
                )
            )
            sector_owner.append(point)
            sector_patch.append(label[faces[begin]])
    dual_owner.append(np.asarray(sector_owner, dtype=np.int64))
    dual_neighbour.append(np.full(len(sector_owner), -1, dtype=np.int64))
    dual_patch.append(np.asarray(sector_patch, dtype=np.int64))

    counts = np.concatenate([*counts_parts, [ring.size for ring in rings]]).astype(np.int64)
    patch_of = np.concatenate(dual_patch)
    patches = primal.face_patches
    named = [name for name in patches.names if name not in _AUTOMATIC]
    return Mesh.from_csr(
        vertices,
        np.concatenate([[0], np.cumsum(counts)]),
        np.concatenate([edge_rings, *rings]),
        np.concatenate(dual_owner),
        np.concatenate(dual_neighbour),
        n_points,
        face_patches={name: np.flatnonzero(patch_of == patches.id_of(name)) for name in named},
        patch_types=dict(patches.patch_types),
        patch_groups=dict(patches.patch_groups),
    )
