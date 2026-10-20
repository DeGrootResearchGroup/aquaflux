"""The polyhedral dual of a conformal mesh, keeping the boundary's features sharp.

Every point of a three-dimensional mesh (the *primal*) becomes one cell of the dual, and every edge
the face between its two end points' cells. The dual of a conformal mesh is conformal, and each
primal cell type has a dual of a useful kind -- a block of hexahedra gives hexahedra (shifted half a
cell), tetrahedra give general polyhedra -- so a mesh made of several kinds of cell, joined
conformally, has a dual whose pieces are joined conformally too, with no tetrahedra or pyramids left
in it. Dual cell ``i`` is the cell of primal point ``i`` (its first piece, for a point on a sharp
concave edge; the further pieces follow the points' cells, below).

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

**A cell at a sharp concave fold is cut into pieces.** The dual cell of a point on an edge folding
concavely by more than the feature angle wraps around the fold, so far from convex that one of its
own vertices lies a whole cell size beyond one of its faces -- a poor cell for a finite-volume
discretization, whose fluxes assume a face separates its two cells. So the primal cells
around such a point are shared among the walls meeting at its concave edges, each going to the wall
its centroid lies deepest behind, and each wall's share is one dual cell. Where two pieces meet,
the dual faces are cut through vertices put on the surface between them (for two walls, the plane
bisecting the fold), so the face between the pieces is flat and each piece is the part of the cell
on its side: the dual edge between two cells in different pieces gains the vertex where it crosses
that surface, a dual face spanning two pieces is cut into one face per piece, meeting at a vertex
on that surface, and each primal face between the pieces gives a face between them through the
point. A gently concave wall (a lamp sleeve, seen from the water around it) folds by less than the
feature angle and is not cut.
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
        """The dual of ``primal`` -- the mesh of :meth:`build_traced`.

        Parameters
        ----------
        primal : Mesh
            A three-dimensional, non-periodic mesh with one cell zone and no patch on interior
            faces, whose boundary is a closed manifold surface (every boundary edge on exactly two
            boundary faces, every boundary point's faces one fan).

        Returns
        -------
        Mesh
            One cell per primal point, in the same order, then the further pieces of the points on
            sharp concave edges, by point. The dual's boundary faces are in the patches of the
            primal faces they cover, with the same types and groups.

        Raises
        ------
        ValueError
            If the primal is not a mesh this construction accepts (see ``primal``), or a named
            feature edge is not a primal boundary edge.
        NotImplementedError
            If the primal cells around a point on a sharp concave edge that go to one wall are not
            face-connected there, so would make a dual cell in two parts.
        """
        return self.build_traced(primal).mesh

    def build_traced(self, primal: Mesh) -> TracedDual:
        """The dual of ``primal``, with where each vertex on its boundary came from.

        Parameters
        ----------
        primal : Mesh
            As :meth:`build`.

        Returns
        -------
        TracedDual

        Raises
        ------
        ValueError, NotImplementedError
            As :meth:`build`.
        """
        _check_primal(primal)
        tables = _PrimalTables.of(primal)
        feature, concave = self._features(primal, tables)
        walks = _Walks.of(primal, tables)
        pieces = _Pieces.of(primal, tables, walks, feature, concave)
        return _assemble(primal, tables, walks, feature, pieces)

    def _features(self, primal: Mesh, tables: _PrimalTables) -> tuple[np.ndarray, np.ndarray]:
        """Which edges are features, and which fold concavely by more than the feature angle.

        Both of shape ``(n_edges,)``; every such concave edge is a feature.
        """
        on_boundary = tables.boundary_faces[:, 0] >= 0
        fold = boundary_folds(primal, tables.boundary_faces[on_boundary])
        concave = np.zeros(tables.n_edges, dtype=bool)
        concave[on_boundary] = fold < -self.feature_angle
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
        return feature, concave


@dataclasses.dataclass(frozen=True)
class TracedDual:
    """A dual mesh, and the primal boundary each of its boundary vertices stands for.

    A vertex on the dual's boundary is a primal boundary face's centroid, the midpoint of a primal
    boundary edge (or, where a cell is cut into pieces, a point on that edge's two boundary faces'
    centroids), or a primal boundary point itself. A mesher that knows the true surface the primal
    boundary approximates moves the first two onto it; a primal point is already there.

    Attributes
    ----------
    mesh : Mesh
        The dual.
    vertex_face : np.ndarray of int, shape ``(n_vertices,)``
        The primal boundary face whose centroid each vertex is, or ``-1``.
    vertex_edge : np.ndarray of int, shape ``(n_vertices, 2)``
        The two points of the primal boundary edge each vertex stands for (ascending), or ``-1``.
    """

    mesh: Mesh
    vertex_face: np.ndarray
    vertex_edge: np.ndarray


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
def _walk_edges(
    edge_face_offsets, edge_faces, owner, neighbour, ring_offsets, ring_cells, ring_faces, ends
):
    """Order the cells around every edge, and find a boundary edge's first and last faces.

    For an edge inside the domain the faces and cells around it form a cycle; for one on the
    boundary, a chain from one boundary face to the other. ``ring_cells[ring_offsets[e]:...]``
    receives the cells in order, ``ring_faces`` (same rows) the face from each cell to the next
    (left as it is after a chain's last cell), and ``ends[e]`` the chain's two boundary faces
    (``-1`` inside).
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
            ring_faces[out + filled - 1] = following
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


@dataclasses.dataclass(frozen=True)
class _Walks:
    """The cells around every primal edge and the boundary faces around every boundary point.

    Around edge ``e``, ``ring_cells[ring_offsets[e]:ring_offsets[e + 1]]`` are its cells in order
    and ``ring_faces`` (same rows) the face from each cell to the next -- cyclically inside the
    domain; on the boundary the last entry is ``-1`` and ``ends[e]`` are the chain's first and last
    boundary faces. Around boundary point ``p``, ``fan_faces[point_offsets[p]:...]`` are its
    boundary faces in order and ``fan_edges`` the boundary edge before each.
    """

    ring_offsets: np.ndarray
    ring_cells: np.ndarray
    ring_faces: np.ndarray
    ends: np.ndarray
    point_offsets: np.ndarray
    fan_edges: np.ndarray
    fan_faces: np.ndarray

    def ring(self, edge: int) -> tuple[np.ndarray, np.ndarray]:
        """The cells around ``edge`` and the face after each."""
        rows = slice(self.ring_offsets[edge], self.ring_offsets[edge + 1])
        return self.ring_cells[rows], self.ring_faces[rows]

    def fan(self, point: int) -> tuple[np.ndarray, np.ndarray]:
        """The boundary edges and faces around ``point``, the edge before each face."""
        rows = slice(self.point_offsets[point], self.point_offsets[point + 1])
        return self.fan_edges[rows], self.fan_faces[rows]

    @classmethod
    def of(cls, primal: Mesh, tables: _PrimalTables) -> _Walks:
        owner = np.asarray(primal.face_cells.owner, dtype=np.int64)
        neighbour = np.asarray(primal.face_cells.neighbour, dtype=np.int64)
        on_boundary = tables.boundary_faces[:, 0] >= 0
        n_around = np.diff(tables.edge_face_offsets)
        ring_offsets = np.concatenate([[0], np.cumsum(n_around - on_boundary)])
        ring_cells = np.empty(ring_offsets[-1], dtype=np.int64)
        ring_faces = np.full(ring_offsets[-1], -1, dtype=np.int64)
        ends = np.empty((tables.n_edges, 2), dtype=np.int64)
        status = _walk_edges(
            tables.edge_face_offsets, tables.edge_faces, owner, neighbour,
            ring_offsets, ring_cells, ring_faces, ends,
        )  # fmt: skip
        if status != _OK:
            raise ValueError(
                "the cells around some edge do not form one fan: the primal is not manifold"
            )
        face_offsets = np.asarray(primal.face_nodes.offsets, dtype=np.int64)
        face_nodes = np.asarray(primal.face_nodes.face_node_indices, dtype=np.int64)
        face_of = np.asarray(primal.face_nodes.face_of_incidence, dtype=np.int64)
        boundary_incidences = np.flatnonzero(neighbour[face_of] < 0)
        point_offsets, order = ragged.group(face_nodes[boundary_incidences], primal.n_nodes)
        point_incidences = boundary_incidences[order]
        fan_edges = np.empty(point_incidences.size, dtype=np.int64)
        fan_faces = np.empty(point_incidences.size, dtype=np.int64)
        status = _walk_points(
            point_offsets, point_incidences, face_offsets, face_nodes, tables.incidence_edge,
            tables.previous_incidence, face_of, tables.boundary_faces, fan_edges, fan_faces,
        )  # fmt: skip
        if status != _OK:
            raise ValueError(
                "the boundary faces around some point do not form one fan: the boundary is not "
                "manifold"
            )
        return cls(ring_offsets, ring_cells, ring_faces, ends, point_offsets, fan_edges, fan_faces)


@dataclasses.dataclass(frozen=True)
class _Pieces:
    """The dual cells of points on sharp concave edges, cut into pieces.

    Such a point's dual cell wraps around the fold. The walls meeting at its concave edges are
    each given the plane through the point normal to the wall's mean normal there, and each primal
    cell around the point goes to the wall its centroid lies deepest behind; the cells going to one
    wall make one piece. Two pieces meet on the surface where the two walls' depths are equal: for
    two walls, the plane through the point bisecting the domain's angle at the fold. The vertices
    added between pieces are put on that surface (:meth:`crossing`), so the faces between them are
    flat and each piece is the part of the cell on one side of it.

    Attributes
    ----------
    keys : np.ndarray of int
        Sorted ``point * n_cells + cell`` for every cell around every cut point.
    piece : np.ndarray of int
        The dual cell each key's cell belongs to: the point itself for its first piece, a cell
        numbered from ``n_points`` up for each further one.
    points : np.ndarray of bool, shape ``(n_points,)``
        Which points are cut.
    n_dual_cells : int
    cuts : np.ndarray of int, shape ``(n, 2)``
        ``(point, face)`` for every interior primal face around a cut point whose two cells are in
        different pieces of it.
    pieces, origin, normal : np.ndarray
        Every piece of a cut point, sorted, shape ``(n,)``; its point's coordinates and its wall's
        unit outward normal, shape ``(n, 3)`` each.
    """

    n_cells: int
    keys: np.ndarray
    piece: np.ndarray
    points: np.ndarray
    n_dual_cells: int
    cuts: np.ndarray
    pieces: np.ndarray
    origin: np.ndarray
    normal: np.ndarray

    def interface(self, first: int, second: int) -> tuple[np.ndarray, float]:
        """The plane two pieces of one point meet on, as ``(row, value)``: ``row @ x == value``."""
        a, b = np.searchsorted(self.pieces, [first, second])
        row = self.normal[b] - self.normal[a]
        return row, float(row @ self.origin[a])

    def crossing(self, first, second, start, end) -> np.ndarray:
        """Where segments cross the surface between two pieces of one point, as a fraction.

        Parameters
        ----------
        first, second : np.ndarray of int, shape ``(n,)``
            Pieces of one point each.
        start, end : np.ndarray, shape ``(n, 3)``
            The segments' ends, ``start`` on the side of ``first``.

        Returns
        -------
        np.ndarray, shape ``(n,)``
            The fraction of the way from ``start`` to ``end`` at which the two walls' depths are
            equal, kept to ``[0.1, 0.9]`` so that no added vertex sits on an end.
        """
        a = np.searchsorted(self.pieces, first)
        b = np.searchsorted(self.pieces, second)
        towards = self.normal[b] - self.normal[a]  # the first wall's depth less the second's
        rise = np.einsum("ij,ij->i", start - self.origin[a], towards)
        fall = np.einsum("ij,ij->i", end - self.origin[a], towards)
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = rise / (rise - fall)
        return np.clip(np.nan_to_num(fraction, nan=0.5), 0.1, 0.9)

    def dual_cells(self, point: int | np.ndarray, cells: np.ndarray) -> np.ndarray:
        """The dual cell of ``point`` holding each of ``cells`` (each a cell around it)."""
        result = np.broadcast_to(np.asarray(point, dtype=np.int64), cells.shape).copy()
        if self.keys.size:
            wanted = result * self.n_cells + cells
            position = np.minimum(np.searchsorted(self.keys, wanted), self.keys.size - 1)
            hit = self.keys[position] == wanted
            result[hit] = self.piece[position[hit]]
        return result

    @classmethod
    def of(
        cls, primal: Mesh, tables: _PrimalTables, walks: _Walks, feature: np.ndarray,
        concave: np.ndarray,
    ) -> _Pieces:  # fmt: skip
        """Cut the dual cell of every point on a ``concave`` edge (shape ``(n_edges,)``).

        Raises
        ------
        NotImplementedError
            If the cells going to one wall around a point are not face-connected around it, so
            would make a dual cell in two parts.
        """
        n_points, n_cells = primal.n_nodes, primal.n_cells
        cut_points = np.unique(tables.edge_nodes[concave])
        geometry = primal.geometry()
        centroid = np.asarray(geometry.cell.centroid)
        area_vector = np.asarray(geometry.face.normal) * np.asarray(geometry.face.area)[:, None]
        points = np.asarray(primal.node_coords, dtype=np.float64)
        owner = np.asarray(primal.face_cells.owner, dtype=np.int64)
        neighbour = np.asarray(primal.face_cells.neighbour, dtype=np.int64)
        face_nodes = np.asarray(primal.face_nodes.face_node_indices, dtype=np.int64)
        face_of = np.asarray(primal.face_nodes.face_of_incidence, dtype=np.int64)
        incidence_offsets, incidence_order = ragged.group(face_nodes, n_points)
        keys, pieces, cuts, walls_of = [], [], [], []
        cut = np.zeros(n_points, dtype=bool)
        next_cell = n_points
        for point in cut_points:
            edges, fan = walks.fan(point)
            sector = np.cumsum(feature[edges]) % max(int(np.count_nonzero(feature[edges])), 1)
            at_fold = np.flatnonzero(concave[edges])
            walls = np.unique(np.concatenate([sector[at_fold], sector[at_fold - 1]]))
            normal = np.stack([area_vector[fan[sector == wall]].sum(axis=0) for wall in walls])
            normal /= np.linalg.norm(normal, axis=1, keepdims=True)
            faces = face_of[
                incidence_order[incidence_offsets[point] : incidence_offsets[point + 1]]
            ]
            around = np.unique(np.concatenate([owner[faces], neighbour[faces]]))
            around = around[around >= 0]
            depth = -(centroid[around] - points[point]) @ normal.T
            used, group = np.unique(np.argmax(depth, axis=1), return_inverse=True)
            n_groups = int(group.max()) + 1
            if n_groups < 2:
                continue
            inner = faces[neighbour[faces] >= 0]
            first = group[np.searchsorted(around, owner[inner])]
            second = group[np.searchsorted(around, neighbour[inner])]
            _check_connected(around, n_groups, owner[inner], neighbour[inner], first == second)
            cut[point] = True
            keys.append(point * n_cells + around)
            pieces.append(np.where(group == 0, point, next_cell + group - 1))
            walls_of.append(
                (np.r_[point, next_cell + np.arange(n_groups - 1)], points[point], normal[used])
            )
            next_cell += n_groups - 1
            between = inner[first != second]
            cuts.append(np.stack([np.full(between.size, point), between], axis=1))
        if not keys:
            empty, none = np.zeros(0, dtype=np.int64), np.zeros((0, 3))
            pairs = np.zeros((0, 2), dtype=np.int64)
            return cls(n_cells, empty, empty, cut, n_points, pairs, empty, none, none)
        ids = np.concatenate([own for own, _, _ in walls_of])
        origin = np.concatenate([np.tile(at, (own.size, 1)) for own, at, _ in walls_of])
        order = np.argsort(ids)
        return cls(
            n_cells, np.concatenate(keys), np.concatenate(pieces), cut, next_cell,
            np.concatenate(cuts), ids[order], origin[order],
            np.concatenate([normals for _, _, normals in walls_of])[order],
        )  # fmt: skip


def _check_connected(cells, n_groups, first, second, joined) -> None:
    """Refuse a piece whose cells are not face-connected around its point."""
    parent = np.arange(cells.size)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(
        np.searchsorted(cells, first[joined]), np.searchsorted(cells, second[joined]), strict=True
    ):
        parent[root(a)] = root(b)
    roots = np.array([root(i) for i in range(cells.size)])
    if np.unique(roots).size != n_groups:
        raise NotImplementedError(
            "the cells around a point on a concave edge that lie deepest behind one wall are not "
            "connected, so would make a dual cell in two parts"
        )


@dataclasses.dataclass(frozen=True)
class _Vertices:
    """The numbering of the dual's vertices, kind by kind, with -1 where a kind has none."""

    coordinates: np.ndarray
    boundary_face: np.ndarray  # (n_faces,): the vertex at a boundary face's centroid
    cut_face: np.ndarray  # (n_faces,): the vertex on the dual edge of a face cut between pieces
    middle: np.ndarray  # (n_edges,): the vertex at an edge's midpoint
    point: np.ndarray  # (n_points,): the vertex at a primal point

    @classmethod
    def of(
        cls, primal: Mesh, tables: _PrimalTables, feature: np.ndarray, middle: np.ndarray,
        pieces: _Pieces,
    ) -> _Vertices:  # fmt: skip
        """Cell centroids, boundary-face centroids, a vertex per face cut between pieces,
        midpoints of ``middle`` edges, and the points on two feature edges or cut into pieces.

        The vertices of cut faces and of edges cut between pieces start at face centroids and edge
        midpoints and are moved onto the surfaces between pieces as the faces are built."""
        geometry = primal.geometry()
        neighbour = np.asarray(primal.face_cells.neighbour)
        points = np.asarray(primal.node_coords, dtype=np.float64)
        face_centroid = np.asarray(geometry.face.centroid, dtype=np.float64)
        on_features = np.bincount(tables.edge_nodes[feature].reshape(-1), minlength=primal.n_nodes)
        kinds = [
            ("boundary_face", neighbour < 0, face_centroid),
            ("cut_face", np.isin(np.arange(primal.n_faces), pieces.cuts[:, 1]), face_centroid),
            ("middle", middle, points[tables.edge_nodes].mean(axis=1)),
            ("point", (on_features >= 2) | pieces.points, points),
        ]
        parts = [np.asarray(geometry.cell.centroid, dtype=np.float64)]
        base = primal.n_cells
        numbering = {}
        for name, chosen, where in kinds:
            index = np.flatnonzero(chosen)
            numbering[name] = np.full(chosen.size, -1, dtype=np.int64)
            numbering[name][index] = base + np.arange(index.size)
            parts.append(where[index])
            base += index.size
        return cls(np.concatenate(parts), **numbering)


class _Faces:
    """The dual's faces as they are built: rings, owner, neighbour and the patch of each."""

    def __init__(self) -> None:
        self.rings: list[np.ndarray] = []
        self.owner: list[int] = []
        self.neighbour: list[int] = []
        self.patch: list[int] = []

    def add(self, ring, owner: int, neighbour: int = -1, patch: int = -1) -> None:
        self.rings.append(np.asarray(ring, dtype=np.int64))
        self.owner.append(int(owner))
        self.neighbour.append(int(neighbour))
        self.patch.append(int(patch))


def _assemble(
    primal: Mesh, tables: _PrimalTables, walks: _Walks, feature: np.ndarray, pieces: _Pieces
) -> TracedDual:
    """Number the dual's vertices, build its faces, and hand them to ``Mesh.from_csr``."""
    on_boundary = tables.boundary_faces[:, 0] >= 0
    # The edges whose faces see a piece: those at a cut point, and those of a face cut between
    # pieces (whose ring gains that face's vertex).
    touched = pieces.points[tables.edge_nodes].any(axis=1)
    face_of = np.asarray(primal.face_nodes.face_of_incidence)
    touched[tables.incidence_edge[np.isin(face_of, pieces.cuts[:, 1])]] = True
    touched_edges = np.flatnonzero(touched)
    split = np.zeros(tables.n_edges, dtype=bool)
    split[touched_edges] = [
        _breaks(walks, tables, pieces, edge, on_boundary[edge]).size > 0 for edge in touched_edges
    ]
    vertices = _Vertices.of(primal, tables, feature, feature | split, pieces)
    _place_cut_faces(primal, pieces, vertices)

    faces = _Faces()
    plain = _plain_edge_faces(np.flatnonzero(~touched), walks, tables, feature, vertices)
    for edge in touched_edges:
        _pieced_edge_faces(edge, walks, tables, feature, pieces, vertices, on_boundary[edge], faces)
    _separating_faces(primal, tables, pieces, vertices, faces)
    _boundary_faces(primal, walks, feature, feature | split, pieces, vertices, faces)

    counts = np.concatenate([plain[0], [ring.size for ring in faces.rings]]).astype(np.int64)
    patch_of = np.concatenate([np.full(plain[0].size, -1), faces.patch])
    patches = primal.face_patches
    named = [name for name in patches.names if name not in _AUTOMATIC]
    n_vertices = vertices.coordinates.shape[0]
    vertex_face = np.full(n_vertices, -1, dtype=np.int64)
    bounding = np.flatnonzero(vertices.boundary_face >= 0)
    vertex_face[vertices.boundary_face[bounding]] = bounding
    vertex_edge = np.full((n_vertices, 2), -1, dtype=np.int64)
    halved = np.flatnonzero((vertices.middle >= 0) & on_boundary)
    vertex_edge[vertices.middle[halved]] = tables.edge_nodes[halved]
    mesh = Mesh.from_csr(
        vertices.coordinates,
        np.concatenate([[0], np.cumsum(counts)]),
        np.concatenate([plain[1], *faces.rings]),
        np.concatenate([plain[2], faces.owner]).astype(np.int64),
        np.concatenate([plain[3], faces.neighbour]).astype(np.int64),
        pieces.n_dual_cells,
        face_patches={name: np.flatnonzero(patch_of == patches.id_of(name)) for name in named},
        patch_types=dict(patches.patch_types),
        patch_groups=dict(patches.patch_groups),
    )
    return TracedDual(mesh, vertex_face, vertex_edge)


def _place_cut_faces(primal, pieces, vertices) -> None:
    """Put each cut face's vertex where the segment between its cells' centroids crosses between
    the pieces (the mean, for a face cut around more than one point)."""
    if not pieces.cuts.size:
        return
    centroid = np.asarray(primal.geometry().cell.centroid)
    point, face = pieces.cuts[:, 0], pieces.cuts[:, 1]
    first = np.asarray(primal.face_cells.owner)[face]
    second = np.asarray(primal.face_cells.neighbour)[face]
    start, end = centroid[first], centroid[second]
    fraction = pieces.crossing(
        pieces.dual_cells(point, first), pieces.dual_cells(point, second), start, end
    )
    where = start + fraction[:, None] * (end - start)
    target = vertices.cut_face[face]
    total = np.zeros((vertices.coordinates.shape[0], 3))
    np.add.at(total, target, where)
    count = np.bincount(target, minlength=total.shape[0])
    chosen = np.unique(target)
    vertices.coordinates[chosen] = total[chosen] / count[chosen, None]


def _plain_edge_faces(edges, walks, tables, feature, vertices):
    """The faces of edges no piece touches: one each, between its two ends' cells.

    The ring is ``[first boundary face, the cells around, last boundary face, the midpoint]`` on
    the boundary (the midpoint only for a feature) and the cells alone inside. Returns the counts,
    the concatenated rings, the owners and the neighbours.
    """
    on_boundary = tables.boundary_faces[edges, 0] >= 0
    lead = on_boundary.astype(np.int64)
    n_ring = walks.ring_offsets[edges + 1] - walks.ring_offsets[edges]
    counts = n_ring + 2 * lead + feature[edges].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rings = np.empty(int(counts.sum()), dtype=np.int64)
    of_cell = np.repeat(np.arange(edges.size), n_ring)
    position = np.arange(of_cell.size) - np.repeat(np.cumsum(n_ring) - n_ring, n_ring)
    rings[starts[of_cell] + lead[of_cell] + position] = ragged.rows(
        walks.ring_offsets, walks.ring_cells, edges
    )[0]
    bounded = np.flatnonzero(on_boundary)
    rings[starts[bounded]] = vertices.boundary_face[walks.ends[edges[bounded], 0]]
    rings[starts[bounded] + 1 + n_ring[bounded]] = vertices.boundary_face[
        walks.ends[edges[bounded], 1]
    ]
    kept = np.flatnonzero(feature[edges])
    rings[starts[kept] + 2 + n_ring[kept]] = vertices.middle[edges[kept]]
    return counts, rings, tables.edge_nodes[edges, 0], tables.edge_nodes[edges, 1]


def _breaks(walks, tables, pieces, edge, bounded) -> np.ndarray:
    """The positions around ``edge`` after which the next cell is in another pair of pieces."""
    cells, _ = walks.ring(edge)
    pair = np.stack([pieces.dual_cells(node, cells) for node in tables.edge_nodes[edge]], axis=1)
    following = np.roll(pair, -1, axis=0)
    changes = np.any(pair != following, axis=1)
    if bounded:
        changes[-1] = False
    return np.flatnonzero(changes)


def _pieced_edge_faces(edge, walks, tables, feature, pieces, vertices, bounded, faces) -> None:
    """The faces of an edge a piece touches.

    The ring of cells gains the vertex of every face cut between pieces. Where the pair of pieces
    either side changes, the face is cut there into one face per run of cells, each closed through
    the edge's own vertex (its midpoint, on a feature edge): the face between those two pieces.
    """
    cells, after = walks.ring(edge)
    ends = [pieces.dual_cells(node, cells) for node in tables.edge_nodes[edge]]
    ring, at = [], []  # vertices, and the vertex position of each break
    breaks = set(_breaks(walks, tables, pieces, edge, bounded).tolist())
    if bounded:
        ring.append(vertices.boundary_face[walks.ends[edge, 0]])
    for k, cell in enumerate(cells):
        ring.append(cell)
        if after[k] >= 0 and vertices.cut_face[after[k]] >= 0:
            ring.append(vertices.cut_face[after[k]])
            if k in breaks:
                at.append(len(ring) - 1)
    if bounded:
        ring.append(vertices.boundary_face[walks.ends[edge, 1]])
    first = [0, *(k + 1 for k in sorted(breaks))]  # the first cell of each run
    middle = vertices.middle[edge]
    if not at:
        faces.add([*ring, *([middle] if middle >= 0 else [])], ends[0][0], ends[1][0])
        return
    if not feature[edge]:
        _place_middle(middle, ring, at, ends, sorted(breaks), bounded, pieces, vertices.coordinates)
    if bounded:
        bounds = [0, *at, len(ring) - 1]
        runs = [ring[a : b + 1] for a, b in pairwise(bounds)]
    else:
        bounds = [*at, at[0] + len(ring)]
        doubled = ring + ring
        runs = [doubled[a : b + 1] for a, b in pairwise(bounds)]
        first = [(k + 1) % len(cells) for k in sorted(breaks)]
    for run, cell in zip(runs, first, strict=True):
        faces.add([*run, middle], ends[0][cell], ends[1][cell])


def _place_middle(middle, ring, at, ends, breaks, bounded, pieces, coordinates) -> None:
    """Put the vertex a cut edge face's pieces meet at on the surfaces between them.

    Inside the domain: the point nearest the mean of the cut faces' vertices around the edge (each
    on the surface between the pieces either side of it) that lies on every one of those surfaces
    and in the plane through that mean normal to the dual face -- so, where three pieces meet, on
    the line all three share. On the boundary it must stay on the segment between
    the edge's two boundary-face centroids, which every face around it shares: where that segment
    crosses between the pieces of an end whose first and last cells are in different pieces, else at
    its middle.
    """
    if not bounded:
        ring = np.asarray(ring)
        mean = coordinates[ring[at]].mean(axis=0)
        polygon = coordinates[ring] - mean
        normal = np.cross(polygon, np.roll(polygon, -1, axis=0)).sum(axis=0)
        rows, values = [normal], [normal @ mean]
        for k in breaks:
            for pieces_of_end in ends:
                before, after = pieces_of_end[k], pieces_of_end[(k + 1) % pieces_of_end.size]
                if before != after:
                    row, value = pieces.interface(before, after)
                    rows.append(row)
                    values.append(value)
        rows, values = np.asarray(rows), np.asarray(values)
        scale = np.linalg.norm(rows, axis=1)
        rows, values = rows / scale[:, None], values / scale
        coordinates[middle] = mean + np.linalg.pinv(rows, rcond=1e-8) @ (values - rows @ mean)
        return
    start, end = coordinates[ring[0]], coordinates[ring[-1]]
    fraction = 0.5
    for pieces_of_end in ends:
        if pieces_of_end[0] != pieces_of_end[-1]:
            fraction = pieces.crossing(
                pieces_of_end[:1], pieces_of_end[-1:], start[None], end[None]
            )[0]
            break
    coordinates[middle] = start + fraction * (end - start)


def _separating_faces(primal, tables, pieces, vertices, faces) -> None:
    """The faces between two pieces of one point: one per primal face cut between them.

    The quadrilateral through the point, the vertices of the face's two edges at it and the face's
    own vertex, all on the surface between the two pieces.
    """
    face_offsets = np.asarray(primal.face_nodes.offsets, dtype=np.int64)
    face_nodes = np.asarray(primal.face_nodes.face_node_indices, dtype=np.int64)
    owner = np.asarray(primal.face_cells.owner, dtype=np.int64)
    neighbour = np.asarray(primal.face_cells.neighbour, dtype=np.int64)
    for point, face in pieces.cuts:
        incidence = face_offsets[face] + int(
            np.flatnonzero(face_nodes[face_offsets[face] : face_offsets[face + 1]] == point)[0]
        )
        leaving = tables.incidence_edge[incidence]
        entering = tables.incidence_edge[tables.previous_incidence[incidence]]
        sides = pieces.dual_cells(point, np.array([owner[face], neighbour[face]]))
        faces.add(
            [
                vertices.point[point],
                vertices.middle[leaving],
                vertices.cut_face[face],
                vertices.middle[entering],
            ],
            sides[0],
            sides[1],
        )


def _boundary_faces(primal, walks, feature, middle, pieces, vertices, faces) -> None:
    """Each boundary point's share of the boundary, one face per sector of it.

    The polygon through the centroids of the boundary faces around the point, passing through the
    midpoint of every edge between them that has one, and cut into sectors along the feature edges
    and where the next face's cell is in another piece of the point. With two or more sectors each
    has the point as a vertex.
    """
    label = np.asarray(primal.face_patches.label)
    owner = np.asarray(primal.face_cells.owner, dtype=np.int64)
    for point in np.flatnonzero(np.diff(walks.point_offsets)):
        edges, fan = walks.fan(point)
        cuts = feature[edges]
        piece = np.full(fan.size, point, dtype=np.int64)
        if pieces.points[point]:
            piece = pieces.dual_cells(point, owner[fan])
            cuts = cuts | (piece != np.roll(piece, 1))
        cut = np.flatnonzero(cuts)
        shift = int(cut[0]) if cut.size else 0
        edges, fan, piece = np.roll(edges, -shift), np.roll(fan, -shift), np.roll(piece, -shift)
        through = middle[edges]

        def run(begin, end, edges=edges, fan=fan, through=through):
            ring = []
            for k in range(begin, end):
                if k > begin and through[k]:
                    ring.append(vertices.middle[edges[k]])
                ring.append(vertices.boundary_face[fan[k]])
            return ring

        if cut.size == 0:
            opening = [vertices.middle[edges[0]]] if through[0] else []
            faces.add([*opening, *run(0, fan.size)], piece[0], patch=label[fan[0]])
        elif cut.size == 1:
            faces.add([vertices.middle[edges[0]], *run(0, fan.size)], piece[0], patch=label[fan[0]])
        else:
            bounds = [*(cut - shift).tolist(), fan.size]
            for begin, end in pairwise(bounds):
                faces.add(
                    [
                        vertices.point[point],
                        vertices.middle[edges[begin]],
                        *run(begin, end),
                        vertices.middle[edges[end % fan.size]],
                    ],
                    piece[begin],
                    patch=label[fan[begin]],
                )
