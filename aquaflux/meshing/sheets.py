"""A closed surface with sheets standing in it, opened along the sheets into one closed surface.

A baffle is a sheet with the fluid on both sides of it: a face of the region that is not a wall
between the region and the outside but a cut through the region. A triangulation of such a region
is not a closed surface -- a sheet's edge free in the fluid is used by one triangle, its edge on a
wall by three -- and nothing downstream (prism layers, a hexahedral core, a tetrahedral fill, the
polyhedral dual) takes one.

:func:`opened` makes it one, as if the sheet were a slit of no thickness. Each face of a sheet is
given twice, once wound each way, one for the fluid on each side. A point of a sheet becomes one
point per side wherever the two sides part: inside the sheet and where it meets a wall, but not
along the sheet's free edge, round which the fluid of one side runs into the other's. Which faces
around a point belong to one side is read from the order of the faces around each edge: going
round an edge, every face lets the fluid through on one side of it, and two faces facing each other
across fluid are neighbours on the opened surface. A point's faces that are neighbours, edge by
edge, form one side of it, and each side gets a point of its own.

An edge inside a sheet, or along a wall the sheet stands on, whose two ends both lie on the sheet's
free edge would be left as one edge used by four faces, and a triangle with all three corners there
would come out as two faces on the same three points. So each such edge is first cut in two at its
midpoint, moved onto the sheet, until none is left.

A tetrahedral fill cannot take the two sides of a sheet as two surfaces at one place, so
:class:`SlitSurface` gives it the sheet once, as triangles standing inside the region, and
:func:`parted` hands the tetrahedra on each side the points of that side again: the tetrahedra round
a point that reach one another without crossing the sheet are one side of it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .polygons import Polygons

__all__ = ["OpenedSurface", "SlitSurface", "opened", "parted"]

#: Moves points onto faces of the drawing: ``onto(points, faces) -> points``.
Onto = Callable[[np.ndarray, np.ndarray], np.ndarray]

#: How many rounds of cutting the edges between two points of a free edge are allowed: each round
#: cuts one such edge of every face, and a face has at most three.
_MAX_CUTTING_ROUNDS = 8


@dataclasses.dataclass(frozen=True)
class OpenedSurface:
    """A surface opened along its sheets.

    Attributes
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
        The surface's points, then those added: the midpoints of cut edges, then a further copy of a
        point for each further side of a sheet it lies on.
    faces : Polygons
        The faces, every edge used by exactly two of them once each way. A sheet's face is given
        once per side: wound as given, among the triangles in their place, and wound the other way,
        after every given triangle.
    face : np.ndarray of int, shape ``(n_faces,)``
        The drawing's face under each face.
    origin : np.ndarray of int, shape ``(n_points,)``
        The point each point was copied from: itself, unless it is a further copy.
    """

    points: np.ndarray
    faces: Polygons
    face: np.ndarray
    origin: np.ndarray


def opened(
    points: np.ndarray, faces: Polygons, face: np.ndarray, sheets: np.ndarray, onto: Onto
) -> OpenedSurface:
    """A closed surface with sheets standing in it, opened along the sheets.

    Parameters
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
    faces : Polygons
        The surface: a closed surface wound out of the region, with the sheets' faces (any winding)
        standing in it, each sharing its edges with the faces around it.
    face : np.ndarray of int, shape ``(n_faces,)``
        The drawing's face under each face.
    sheets : np.ndarray of int
        The drawing's faces that are sheets: the region lies on both sides of them.
    onto : callable
        ``onto(points, faces) -> points``: each point moved onto the drawing's face given for it,
        for the midpoints of cut edges.

    Returns
    -------
    OpenedSurface

    Raises
    ------
    ValueError
        If a sheet's face is a quadrilateral, two faces meet an edge at the same angle without being
        the two sides of one sheet's face, or the faces round an edge do not alternate in the way
        a surface bounding the region on one side of each does.
    """
    points = np.asarray(points, dtype=np.float64)
    face = np.asarray(face, dtype=np.int64)
    on_sheet = np.isin(face, np.asarray(sheets, dtype=np.int64))
    if not on_sheet.any():
        return OpenedSurface(points, faces, face, np.arange(points.shape[0]))
    if on_sheet[faces.triangles.shape[0] :].any():
        raise ValueError("a sheet is meshed with quadrilaterals; it must be triangles")
    points, faces, face = _cut_free_chords(points, faces, face, on_sheet, onto)
    on_sheet = np.isin(face, np.asarray(sheets, dtype=np.int64))
    n_triangles = faces.triangles.shape[0]
    doubled = Polygons(
        np.concatenate([faces.triangles, faces.triangles[on_sheet[:n_triangles]][:, ::-1]]),
        faces.quads,
    )
    triangle_face, quad_face = faces.split(face)
    face = np.concatenate([triangle_face, triangle_face[on_sheet[:n_triangles]], quad_face])
    twin = np.full(doubled.n_faces, -1, dtype=np.int64)
    given = np.flatnonzero(on_sheet[:n_triangles])
    copies = n_triangles + np.arange(given.size)
    twin[given], twin[copies] = copies, given
    rings = doubled.padded()
    side = _sides(points, doubled, rings, twin)
    # One point per side: the side met first keeps the point's number, the others get new ones.
    used = rings >= 0
    keys = np.stack([rings[used], side[used]], axis=1)
    unique, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    point = unique[order, 0]  # each side's point, sides in the order first met
    keeps = np.zeros(point.size, dtype=bool)
    keeps[np.unique(point, return_index=True)[1]] = True
    number = np.where(keeps, point, points.shape[0] + np.cumsum(~keeps) - 1)
    origin = np.concatenate([np.arange(points.shape[0]), point[~keeps]])
    renumbered = rings.copy()
    renumbered[used] = number[rank[inverse.reshape(-1)]]
    n_doubled = doubled.triangles.shape[0]
    return OpenedSurface(
        points=points[origin],
        faces=Polygons(renumbered[:n_doubled, :3], renumbered[n_doubled:]),
        face=face,
        origin=origin,
    )


def _sides(points, faces: Polygons, rings: np.ndarray, twin: np.ndarray) -> np.ndarray:
    """Each corner's side: corners of one point are on one side when their faces are neighbours
    round the point, edge by edge, on the opened surface. Shape ``(n_faces, 4)``, ``-1`` past a
    triangle's third corner."""
    pairs, owner = faces.edges()
    # Each use's place in its face's ring, in the order Polygons.edges lists them.
    slot = np.concatenate(
        [np.tile(np.arange(kind.shape[1]), kind.shape[0]) for kind in faces.kinds()]
    )
    size = faces.corners()[owner]
    first = owner * 4 + slot
    second = owner * 4 + (slot + 1) % size
    keys = np.sort(pairs, axis=1)
    order = np.lexsort(keys.T[::-1])
    keys = keys[order]
    starts = np.flatnonzero(np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)])
    counts = np.diff(np.r_[starts, keys.shape[0]])
    if np.any(counts < 2):
        raise ValueError(f"{int(np.count_nonzero(counts < 2))} edge(s) bound one face only")
    left, right = [], []
    plain = starts[counts == 2]
    left.append(order[plain])
    right.append(order[plain + 1])
    centroids = faces.centroids(points)
    for start, count in zip(starts[counts > 2], counts[counts > 2], strict=True):
        uses = order[start : start + count]
        a, b = _neighbours_round(points, keys[start], pairs[uses], owner[uses], centroids, twin)
        left.append(uses[a])
        right.append(uses[b])
    left, right = np.concatenate(left), np.concatenate(right)
    if np.any(np.all(pairs[left] == pairs[right], axis=1)):
        raise ValueError("two faces meeting at an edge run along it the same way")
    # The use a -> b and its neighbour's b -> a: corner a of one is joined to corner a of the other.
    n = rings.size
    joined = coo_matrix(
        (
            np.ones(2 * left.size),
            (np.r_[first[left], second[left]], np.r_[second[right], first[right]]),
        ),
        shape=(n, n),
    )
    _, label = connected_components(joined, directed=False)
    side = label.reshape(rings.shape)
    return np.where(rings >= 0, side, -1)


def _neighbours_round(points, key, pairs, owner, centroids, twin):
    """The faces round one edge used by more than two, paired as neighbours across the fluid.

    Going round the edge from its lower-numbered point to its higher, a face running along it that
    way has the region on its far side as the angle decreases, one running the other way as it
    increases; so each face of the second kind is the neighbour of the next face round. Two sides
    of one sheet's face lie at one angle, the one with nothing between them first.
    """
    origin, axis = points[key[0]], points[key[1]] - points[key[0]]
    axis = axis / np.linalg.norm(axis)
    across = centroids[owner] - origin
    across -= np.outer(across @ axis, axis)
    u = across[0] / np.linalg.norm(across[0])
    w = np.cross(axis, u)
    angle = np.mod(np.arctan2(across @ w, across @ u), 2.0 * np.pi)
    forward = pairs[:, 0] == key[0]
    order = np.lexsort((~forward, angle))
    angle, forward, owner_sorted = angle[order], forward[order], owner[order]
    same = np.isclose(np.diff(np.r_[angle, angle[0] + 2.0 * np.pi]), 0.0, atol=1e-9)
    following = np.roll(owner_sorted, -1)
    if np.any(same & (twin[owner_sorted] != following)):
        raise ValueError(
            f"two faces meet the edge between points {key.tolist()} at one angle, and they are not "
            "the two sides of one sheet"
        )
    if forward.sum() * 2 != forward.size or np.any(forward == np.roll(forward, -1)):
        raise ValueError(
            f"the faces round the edge between points {key.tolist()} do not alternate as a surface "
            "bounding the region does"
        )
    backward = np.flatnonzero(~forward)
    after = (backward + 1) % forward.size
    return order[backward], order[after]


def _cut_free_chords(points, faces: Polygons, face, on_sheet, onto):
    """The surface with every edge cut in two that a sheet's face uses and whose two ends lie on a
    sheet's free edge, though it is not one -- until none is left."""
    for _ in range(_MAX_CUTTING_ROUNDS):
        pairs, owner = faces.edges()
        keys = np.sort(pairs, axis=1)
        unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(-1)
        free = np.zeros(points.shape[0], dtype=bool)
        free[unique[counts == 1].reshape(-1)] = True
        sheet_edge = np.zeros(unique.shape[0], dtype=bool)
        sheet_edge[inverse[on_sheet[owner]]] = True
        chords = np.flatnonzero(sheet_edge & (counts >= 2) & free[unique].all(axis=1))
        if not chords.size:
            return points, faces, face
        # At most one chord per face in a round: a face's chosen chord and the faces using it.
        chosen = np.zeros(unique.shape[0], dtype=bool)
        busy = np.zeros(faces.n_faces, dtype=bool)
        users = [owner[inverse == c] for c in chords.tolist()]
        for chord, using in zip(chords.tolist(), users, strict=True):
            if not busy[using].any():
                chosen[chord] = True
                busy[using] = True
        chords = np.flatnonzero(chosen)
        under = np.array(
            [face[owner[(inverse == c) & on_sheet[owner]][0]] for c in chords.tolist()],
            dtype=np.int64,
        )
        middle = onto(points[unique[chords]].mean(axis=1), under)
        midpoint = np.full(unique.shape[0], -1, dtype=np.int64)
        midpoint[chords] = points.shape[0] + np.arange(chords.size)
        points = np.vstack([points, middle])
        faces, face, on_sheet = _bisected(faces, face, on_sheet, pairs, owner, inverse, midpoint)
    raise ValueError("edges between points of a sheet's free edge are still left after cutting")


def _bisected(faces: Polygons, face, on_sheet, pairs, owner, inverse, midpoint):
    """Each face with a cut edge as two faces, the triangle ``a b c`` cut on ``a b`` at ``m`` as
    ``a m c`` and ``m b c``; a quadrilateral ``a b c d`` as the triangle ``a m d`` and the
    quadrilateral ``m b c d``."""
    cut = np.flatnonzero(midpoint[inverse] >= 0)
    keep = np.ones(faces.n_faces, dtype=bool)
    triangle_face, quad_face = list(faces.split(face))
    triangle_sheet, quad_sheet = list(faces.split(on_sheet))
    new_triangles, new_triangle_face, new_triangle_sheet = [], [], []
    new_quads, new_quad_face, new_quad_sheet = [], [], []
    rings = faces.padded()
    size = faces.corners()
    for use in cut.tolist():
        f = owner[use]
        a, b = pairs[use]
        m = midpoint[inverse[use]]
        ring = rings[f, : size[f]]
        k = int(np.flatnonzero(ring == a)[0])
        rolled = np.roll(ring, -k)  # a, b, ...
        keep[f] = False
        if size[f] == 3:
            c = rolled[2]
            new_triangles += [[a, m, c], [m, b, c]]
            new_triangle_face += [face[f]] * 2
            new_triangle_sheet += [on_sheet[f]] * 2
        else:
            c, d = rolled[2], rolled[3]
            new_triangles.append([a, m, d])
            new_triangle_face.append(face[f])
            new_triangle_sheet.append(on_sheet[f])
            new_quads.append([m, b, c, d])
            new_quad_face.append(face[f])
            new_quad_sheet.append(on_sheet[f])
    kept_triangles, kept_quads = faces.split(keep)
    triangles = np.concatenate(
        [faces.triangles[kept_triangles], np.reshape(new_triangles, (-1, 3))]
    )
    quads = np.concatenate([faces.quads[kept_quads], np.reshape(new_quads, (-1, 4))])
    face = np.concatenate([
        triangle_face[kept_triangles], np.asarray(new_triangle_face, np.int64),
        quad_face[kept_quads], np.asarray(new_quad_face, np.int64),
    ])  # fmt: skip
    on_sheet = np.concatenate([
        triangle_sheet[kept_triangles], np.asarray(new_triangle_sheet, bool),
        quad_sheet[kept_quads], np.asarray(new_quad_sheet, bool),
    ])  # fmt: skip
    return Polygons(triangles.astype(np.int64), quads.astype(np.int64)), face, on_sheet


#: The four faces of a tetrahedron, each by the corners other than the one it faces.
_TETRAHEDRON_FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


@dataclasses.dataclass(frozen=True)
class SlitSurface:
    """A surface opened along sheets, as a fill takes it: each point at the place it shares with its
    copies, the sheets' two sides given once, standing in the region.

    Attributes
    ----------
    triangles : np.ndarray of int, shape ``(n, 3)``
        The surface's other triangles, as given, their corners each replaced by its twin.
    internal : np.ndarray of int, shape ``(k, 3)``
        One of each pair of sides of a sheet's triangle, corners likewise.
    """

    triangles: np.ndarray
    internal: np.ndarray

    @classmethod
    def of(cls, triangles: np.ndarray, twins: np.ndarray) -> SlitSurface:
        """The surface ``triangles`` with each point replaced by ``twins[point]``.

        Parameters
        ----------
        triangles : np.ndarray of int, shape ``(n, 3)``
            A closed surface opened along sheets, every face wound out of the region.
        twins : np.ndarray of int, shape ``(n_points,)``
            For each point, the point standing for every copy at its place (itself if none).

        Raises
        ------
        ValueError
            If more than two triangles lie on one place, or two lie there wound the same way.
        """
        merged = np.asarray(twins, dtype=np.int64)[np.asarray(triangles, dtype=np.int64)]
        keys = np.sort(merged, axis=1)
        _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(-1)
        if np.any(counts > 2):
            raise ValueError("more than two triangles of the surface lie at one place")
        paired = counts[inverse] == 2
        rows = np.flatnonzero(paired)
        by_place = rows[np.argsort(inverse[rows], kind="stable")].reshape(-1, 2)
        one, other = merged[by_place[:, 0]], merged[by_place[:, 1]]
        if not np.array_equal(_starting_lowest(one[:, ::-1]), _starting_lowest(other)):
            raise ValueError("two triangles at one place are wound the same way")
        first = np.zeros(merged.shape[0], dtype=bool)
        first[by_place[:, 0]] = True
        return cls(merged[~paired], merged[first])


def _starting_lowest(triangles: np.ndarray) -> np.ndarray:
    """Each triangle's corners turned round to start at the lowest, keeping the winding."""
    shift = np.argmin(triangles, axis=1)
    return np.take_along_axis(triangles, (shift[:, None] + np.arange(3)) % 3, axis=1)


def parted(
    tetrahedra: np.ndarray, points: np.ndarray, surface: np.ndarray, twins: np.ndarray
) -> np.ndarray:
    """Tetrahedra filled about a :class:`SlitSurface`, given the points of their own side again.

    Parameters
    ----------
    tetrahedra : np.ndarray of int, shape ``(n, 4)``
        The fill's tetrahedra, numbered as the slit surface is (each copied point by its twin).
    points : np.ndarray, shape ``(n_points, 3)``
        Every point, the fill's included.
    surface : np.ndarray of int, shape ``(m, 3)``
        The opened surface the slit surface was made of, every face wound out of the region on
        its side.
    twins : np.ndarray of int, shape ``(n_points,)``
        As given to :meth:`SlitSurface.of`; points the fill added are their own.

    Returns
    -------
    np.ndarray of int, shape ``(n, 4)``
        The tetrahedra, each corner at a copied point replaced by the copy on the tetrahedron's side.

    Raises
    ------
    ValueError
        If the tetrahedra round a copied point that reach one another without crossing the sheet
        hold no face of the surface there, or hold faces of two of its copies.
    """
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    surface = np.asarray(surface, dtype=np.int64)
    twins = np.asarray(twins, dtype=np.int64)
    copied = np.bincount(twins, minlength=points.shape[0]) > 1
    if not copied.any():
        return tetrahedra
    n = tetrahedra.shape[0]
    corners = tetrahedra[:, _TETRAHEDRON_FACES]  # (n, 4, 3)
    keys = np.sort(corners, axis=2).reshape(-1, 3)
    order = np.lexsort(keys.T[::-1])
    keys = keys[order]
    same = np.all(keys[1:] == keys[:-1], axis=1)
    sheet = SlitSurface.of(surface, twins).internal
    internal = np.sort(sheet, axis=1)
    row = [("", np.int64)] * 3
    crossing = np.isin(
        np.ascontiguousarray(keys[:-1]).view(row).reshape(-1),
        np.ascontiguousarray(internal).view(row).reshape(-1),
    )
    joined = np.flatnonzero(same & ~crossing)
    first, second = order[joined], order[joined + 1]  # faces as tetrahedron * 4 + face
    # Two tetrahedra sharing a face away from the sheet are on one side at each of its corners.
    left, right = [], []
    for j in range(3):
        point = tetrahedra[first // 4, _TETRAHEDRON_FACES[first % 4, j]]
        keep = copied[point]
        a, b, p = first[keep] // 4, second[keep] // 4, point[keep]
        left.append(a * 4 + np.argmax(tetrahedra[a] == p[:, None], axis=1))
        right.append(b * 4 + np.argmax(tetrahedra[b] == p[:, None], axis=1))
    left, right = np.concatenate(left), np.concatenate(right)
    graph = coo_matrix((np.ones(left.size), (left, right)), shape=(4 * n, 4 * n))
    _, side = connected_components(graph, directed=False)
    # Each face of the opened surface names the copies on its side: the tetrahedron holding it there.
    seed_side, seed_copy = _seeds(tetrahedra, points, surface, twins, copied, keys, order)
    at = tetrahedra.reshape(-1)
    copy = np.full(side.max() + 1, -1, dtype=np.int64)
    copy[side[seed_side]] = seed_copy
    if np.any(copy[side[seed_side]] != seed_copy):
        raise ValueError(
            "the tetrahedra on one side of a sheet hold faces of two copies of a point"
        )
    needed = np.flatnonzero(copied[at])
    if np.any(copy[side[needed]] < 0):
        raise ValueError("tetrahedra on one side of a sheet hold no face of the surface there")
    result = at.copy()
    result[needed] = copy[side[needed]]
    return result.reshape(n, 4)


def _seeds(tetrahedra, points, surface, twins, copied, keys, order):
    """For each corner of a face of the opened surface at a copied point: the corner of the
    tetrahedron holding that face on its side (as tetrahedron * 4 + corner), and the copy."""
    merged = twins[surface]
    wanted = np.sort(merged, axis=1)
    position = np.searchsorted(
        np.ascontiguousarray(keys).view([("", np.int64)] * 3).reshape(-1),
        np.ascontiguousarray(wanted).view([("", np.int64)] * 3).reshape(-1),
    )
    holders = np.stack([order[position], order[np.minimum(position + 1, order.size - 1)]], axis=1)
    found = np.all(keys[np.minimum(position + 1, keys.shape[0] - 1)] == wanted, axis=1)
    if not np.all(np.all(keys[position] == wanted, axis=1)):
        raise ValueError("a face of the surface is no tetrahedron's face")
    # A face of a sheet is held on both sides: take the tetrahedron behind it, away from its normal.
    corner = points[merged]
    normal = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
    tetrahedron = holders // 4
    apex = points[tetrahedra[tetrahedron, holders % 4]]
    behind = np.einsum("ijk,ik->ij", apex - corner[:, None, 0], normal) < 0.0
    choice = np.where(found & ~behind[:, 0], 1, 0)
    holder = tetrahedron[np.arange(surface.shape[0]), choice]
    sides, copies = [], []
    for j in range(3):
        keep = copied[merged[:, j]]
        t, p = holder[keep], merged[keep, j]
        sides.append(t * 4 + np.argmax(tetrahedra[t] == p[:, None], axis=1))
        copies.append(surface[keep, j])
    return np.concatenate(sides), np.concatenate(copies)
