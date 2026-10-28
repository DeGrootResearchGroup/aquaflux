"""A surface whose layered faces fold right back over each other, split so layers can fan round.

Where two layered faces fold right back onto each other -- the two sides of a sheet standing in the
region, along its free edge -- no one direction clears both: their normals are opposite. So every
point on such an edge is given three copies at its place: one for the faces on each side, each
growing layers its own way, and one between them, growing ahead of the edge. Faces of no area join
the copies along the edge, and the layers grown over them are the fan that wraps round it: a wedge
of cells from each side's column to the middle one, and on to the other side's.

Where such an edge turns -- at a sheet's corner -- one column between the sides cannot stand ahead
of both edges, and the cells along each would reach back along it with nothing in the corner: so
there each edge has a middle copy of its own, standing ahead of it, and two triangles of no area join
the four copies, split between the two middle ones: their first cells are tetrahedra filling the
corner. (One quadrilateral would make a pyramid, but a flat one: the two sides' columns grow
straight off the sheet, so the point lies between their tops.)

Round a point of such an edge the faces fall into two runs, one each side. Where the edge ends on a
wall, the wall's faces join the two sides' faces into one run; it is parted at the edge splitting
its angle round the point in half, and the fan closes down onto the wall along that edge: there the
faces of no area are triangles, one end of the edge being a single point.

:func:`fanned` makes the split surface. The layers are grown on it as on any surface, and then the
copies are merged back into their point: the first layer's cells over the faces of no area are then
wedges, tetrahedra in a corner, and where the fan closes down onto a wall, one tent over the edge it
closes along; the rest as they were.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .dual import FOLDED_BACK
from .polygons import Polygons

__all__ = ["FannedSurface", "MergedLayers", "fanned", "folded_edges", "merged"]

#: How far, in radians, a folded-back edge must turn at a point for each of its two sides there to
#: be given a middle copy of its own.
TURNED = np.radians(45.0)


@dataclasses.dataclass(frozen=True)
class FannedSurface:
    """A surface split at the points where its layered faces fold back.

    Attributes
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
        The surface's points, then a copy of each point split for its second side (the first side
        keeps the point), then the copies between the two sides (*middles*): one per split point,
        or where the folded-back edge turns, one per edge there.
    faces : Polygons
        The surface's faces, those of the second side round a split point renumbered to its copy,
        then the faces of no area joining the copies: the triangles after the surface's triangles,
        the quadrilaterals after its quadrilaterals.
    point : np.ndarray of int, shape ``(n_points,)``
        The surface's point each point stands at: itself, for a point of the surface.
    face : np.ndarray of int, shape ``(n_faces,)``
        The surface's face each face lies in: itself, or for a face of no area, the face beside it
        on its first side.
    added : np.ndarray of bool, shape ``(n_faces,)``
        Whether the face is one of no area joining copies.
    middles : np.ndarray of int, shape ``(n_middles,)``
        The middle copies.
    seconds : np.ndarray of int, shape ``(n_middles,)``
        The copy for the second side of each middle copy's point.
    split : np.ndarray of int, shape ``(n_middles,)``
        The surface point each middle copy belongs to.
    landing : np.ndarray of int, shape ``(n_middles,)``
        Where the fan closes down onto a wall at a split point, the far end of the edge it closes
        along; ``-1`` at a point between two folded-back edges.
    ahead_of : np.ndarray of int, shape ``(n_middles,)``
        Where a folded-back edge turns, the far end of the edge each middle copy stands ahead of;
        ``-1`` for a middle copy standing ahead of every folded-back edge at its point.
    """

    points: np.ndarray
    faces: Polygons
    point: np.ndarray
    face: np.ndarray
    added: np.ndarray
    middles: np.ndarray
    seconds: np.ndarray
    split: np.ndarray
    landing: np.ndarray
    ahead_of: np.ndarray


def folded_edges(points: np.ndarray, faces: Polygons, layered: np.ndarray) -> np.ndarray:
    """The edges between two layered faces folded right back onto each other, as ascending point
    pairs, shape ``(n, 2)``.

    Parameters
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
    faces : Polygons
        A closed surface, every edge used by two faces.
    layered : np.ndarray of bool, shape ``(n_faces,)``
    """
    pairs, owner = faces.edges()
    keys = np.sort(pairs, axis=1)
    order = np.lexsort(keys.T[::-1])
    keys, owner = keys[order][0::2], np.stack([owner[order][0::2], owner[order][1::2]], axis=1)
    normals = faces.normals(points)
    cosine = np.einsum("ij,ij->i", normals[owner[:, 0]], normals[owner[:, 1]])
    folded = layered[owner[:, 0]] & layered[owner[:, 1]] & (cosine <= FOLDED_BACK - 1.0)
    return keys[folded]


def fanned(points: np.ndarray, faces: Polygons, layered: np.ndarray) -> FannedSurface | None:
    """The surface split at the points of its folded-back layered edges, or ``None`` if it has none.

    Parameters
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
    faces : Polygons
        A closed surface wound out of the region, every edge used by two faces once each way.
    layered : np.ndarray of bool, shape ``(n_faces,)``

    Returns
    -------
    FannedSurface or None

    Raises
    ------
    NotImplementedError
        If a point lies on more than two folded-back edges, or the edge parting a run of faces round
        a point lies between a layered face and one without layers.
    """
    points = np.asarray(points, dtype=np.float64)
    layered = np.asarray(layered, dtype=bool)
    folds = folded_edges(points, faces, layered)
    if not folds.size:
        return None
    folded = {tuple(edge) for edge in folds.tolist()}
    rings = faces.padded()
    size = faces.corners()
    pairs, owner = faces.edges()
    across = {(int(a), int(b)): int(f) for (a, b), f in zip(pairs.tolist(), owner, strict=True)}
    n_points = points.shape[0]
    split = np.unique(folds)
    second = n_points + np.arange(split.size)
    corners = rings.copy()
    # Each middle copy's point, landing and edge it stands ahead of; the middle copy serving each
    # bridged edge at each of its ends, keyed (end, other end).
    middles: list[tuple[int, int, int]] = []
    mid_of: dict[tuple[int, int], int] = {}
    turned: list[tuple[int, int]] = []  # the points with a middle copy per edge, and a face there
    bridged: dict[tuple[int, int], None] = {}  # the edges whose two faces part, in order met
    for k, p in enumerate(split.tolist()):
        walk = _walk(p, rings, size, across)
        # Turn the walk to start just after a folded edge: each face, with the edge after it.
        start = next(i for i, (_, nxt) in enumerate(walk) if (min(p, nxt), max(p, nxt)) in folded)
        walk = walk[start + 1 :] + walk[: start + 1]
        cut = [(min(p, nxt), max(p, nxt)) in folded for _, nxt in walk]
        if sum(cut) > 2:
            raise NotImplementedError(
                f"point {p} lies on {sum(cut)} edges where layered faces fold back; at most two "
                "are built"
            )
        landing = -1
        if sum(cut) == 2:
            parting = cut.index(True) + 1  # the first face of the second side
        else:
            parting = _half_way(p, [f for f, _ in walk], rings, size, points)
            if layered[walk[parting - 1][0]] != layered[walk[parting][0]]:
                raise NotImplementedError(
                    f"the faces round point {p} would be parted between a layered face and one "
                    "without layers"
                )
            landing = walk[parting - 1][1]
        for f, _ in walk[parting:]:
            corners[f][rings[f] == p] = second[k]
        ends = (walk[parting - 1][1], walk[-1][1])
        if landing < 0 and _turn(points, p, *ends) > TURNED:
            turned.append((p, walk[0][0]))
            for end in ends:
                mid_of[(p, end)] = n_points + split.size + len(middles)
                middles.append((p, -1, end))
        else:
            for end in ends:
                mid_of[(p, end)] = n_points + split.size + len(middles)
            middles.append((p, landing, -1))
        for end in ends:
            bridged[(min(p, end), max(p, end))] = None
    added_triangles, added_quads, parent_triangles, parent_quads = [], [], [], []
    for a, b in bridged:
        # Face f1 runs along the edge a -> b, f2 back; each joins it at the copies it uses.
        f1, f2 = across[(a, b)], across[(b, a)]
        a1, b1 = _copy_in(corners, rings, f1, a), _copy_in(corners, rings, f1, b)
        a2, b2 = _copy_in(corners, rings, f2, b), _copy_in(corners, rings, f2, a)
        ends_a = [a1, mid_of[(a, b)], b2] if a1 != b2 else [a1, a1, a1]
        ends_b = [b1, mid_of[(b, a)], a2] if b1 != a2 else [b1, b1, b1]
        for j in range(2):
            ring = [ends_b[j], ends_a[j], ends_a[j + 1], ends_b[j + 1]]
            ring = [c for i, c in enumerate(ring) if c != ring[i - 1]]
            if len(ring) == 4:
                added_quads.append(ring)
                parent_quads.append(f1)
            else:
                added_triangles.append(ring)
                parent_triangles.append(f1)
    for p, beside in turned:
        added_triangles.extend(_corner(p, mid_of, added_quads))
        parent_triangles.extend([beside, beside])
    n_triangles = faces.triangles.shape[0]
    triangles = corners[:n_triangles, :3]
    quads = corners[n_triangles:]
    new_triangles = np.reshape(np.asarray(added_triangles, dtype=np.int64), (-1, 3))
    new_quads = np.reshape(np.asarray(added_quads, dtype=np.int64), (-1, 4))
    all_faces = Polygons(np.vstack([triangles, new_triangles]), np.vstack([quads, new_quads]))
    face = np.concatenate([
        np.arange(n_triangles), np.asarray(parent_triangles, np.int64),
        n_triangles + np.arange(quads.shape[0]), np.asarray(parent_quads, np.int64),
    ])  # fmt: skip
    added = np.concatenate([
        np.zeros(n_triangles, bool), np.ones(len(parent_triangles), bool),
        np.zeros(quads.shape[0], bool), np.ones(len(parent_quads), bool),
    ])  # fmt: skip
    of_middle, landing, ahead_of = np.reshape(np.asarray(middles, dtype=np.int64), (-1, 3)).T
    copies = np.concatenate([split, of_middle])
    return FannedSurface(
        points=np.vstack([points, points[copies]]),
        faces=all_faces,
        point=np.concatenate([np.arange(n_points), copies]),
        face=face,
        added=added,
        middles=n_points + split.size + np.arange(of_middle.size),
        seconds=second[np.searchsorted(split, of_middle)],
        split=of_middle,
        landing=landing,
        ahead_of=ahead_of,
    )


def _turn(points: np.ndarray, p: int, before: int, after: int) -> float:
    """How far, in radians, a line through ``p`` from ``before`` to ``after`` turns there."""
    a, b = points[p] - points[before], points[after] - points[p]
    cosine = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _corner(p: int, mid_of: dict[tuple[int, int], int], quads: list[list[int]]) -> list[list[int]]:
    """The two triangles of no area closing the hole the faces of no area leave round a point
    whose folded-back edge turns: the hole's corners are the copies of the two sides and the two
    edges' middle copies, each of its edges used once the other way round, and it is split
    between the middle copies."""
    middles = {m for (q, _), m in mid_of.items() if q == p}
    used = set()
    for ring in quads:
        for u, v in zip(ring, ring[1:] + ring[:1], strict=True):
            if u in middles or v in middles:
                used.add((u, v))
    # The hole runs round each edge of a copy used one way only, the other way.
    following = {v: u for u, v in used if (v, u) not in used}
    if len(following) != 4:
        raise ValueError(f"the faces of no area round point {p} leave no four-sided hole")
    start = next(iter(middles))
    ring = [start]
    while following[ring[-1]] != start:
        ring.append(following[ring[-1]])
    return [ring[:3], [ring[2], ring[3], ring[0]]]


def _walk(p: int, rings, size, across) -> list[tuple[int, int]]:
    """The faces round point ``p`` in turn, each with the point after ``p`` in its ring: the edge
    to the next face round."""
    faces_at = np.flatnonzero(np.any(rings == p, axis=1))
    start = int(faces_at[0])
    walk, f = [], start
    while True:
        ring = rings[f, : size[f]]
        k = int(np.flatnonzero(ring == p)[0])
        nxt = int(ring[(k + 1) % size[f]])
        walk.append((f, nxt))
        f = across[(nxt, p)]
        if f == start:
            break
        if len(walk) > faces_at.size:
            raise ValueError(f"the faces round point {p} do not make one fan")
    if len(walk) != faces_at.size:
        raise ValueError(f"the faces round point {p} make more than one fan")
    return walk


def _half_way(p: int, order: list[int], rings, size, points) -> int:
    """Where to part a run of faces round ``p``: the position of the first face whose middle lies
    past half the run's angle round the point."""
    angles = []
    for f in order:
        ring = rings[f, : size[f]]
        k = int(np.flatnonzero(ring == p)[0])
        before = points[ring[k - 1]] - points[p]
        after = points[ring[(k + 1) % size[f]]] - points[p]
        cosine = before @ after / (np.linalg.norm(before) * np.linalg.norm(after))
        angles.append(float(np.arccos(np.clip(cosine, -1.0, 1.0))))
    middles = np.cumsum(angles) - 0.5 * np.asarray(angles)
    position = int(np.searchsorted(middles, 0.5 * sum(angles)))
    return min(max(position, 1), len(order) - 1)


def _copy_in(corners, rings, f: int, p: int) -> int:
    """The copy of point ``p`` that face ``f`` uses."""
    return int(corners[f][int(np.flatnonzero(rings[f] == p)[0])])


@dataclasses.dataclass(frozen=True)
class MergedLayers:
    """Layers grown on a :class:`FannedSurface`, its copies merged back into their points.

    Attributes
    ----------
    renumber : np.ndarray of int
        Each point of the layers grown on the fanned surface, numbered among the merged ones: a
        copy at the wall becomes its point, the layers' points follow the surface's.
    prisms, hexahedra : np.ndarray of int
        The cells: those over the surface's faces, layer by layer as grown, then the fan's.
    tetrahedra : np.ndarray of int, shape ``(n_tetrahedra, 4)``
        The fan's first cells in the corners of a folded-back edge.
    tents : np.ndarray of int, shape ``(n_tents, 6)``
        The fan's first cells where it closes down onto a wall, as
        :func:`~aquaflux.meshing.element_mesh` takes them.
    """

    renumber: np.ndarray
    prisms: np.ndarray
    hexahedra: np.ndarray
    tetrahedra: np.ndarray
    tents: np.ndarray


def merged(
    fan: FannedSurface, n_grown: int, layered: np.ndarray, prisms: np.ndarray, hexahedra: np.ndarray
) -> MergedLayers:
    """The cells grown on ``fan``'s layered faces, its copies merged back into their points.

    Parameters
    ----------
    fan : FannedSurface
    n_grown : int
        The number of points of the grown layers, the fanned surface's first.
    layered : np.ndarray of bool, shape ``(fan.faces.n_faces,)``
    prisms, hexahedra : np.ndarray of int
        The cells over the layered triangles and quadrilaterals, layer by layer, each layer in the
        fanned surface's order; each cell's lower face, then its upper one.

    Returns
    -------
    MergedLayers
        Over a face of no area the first layer's cell is a wedge (over a quadrilateral, between
        two copies at each end) or a tetrahedron (over a corner's triangle, all three copies at
        one point); over the two triangles where the fan closes down onto a wall, one tent; the
        layers above are cells as grown.
    """
    n_fanned, n_points = fan.points.shape[0], int(fan.point.max()) + 1
    renumber = np.concatenate([fan.point, np.arange(n_fanned, n_grown) - (n_fanned - n_points)])
    middle = np.zeros(n_fanned, dtype=bool)
    middle[fan.middles] = True
    tri_added, quad_added = fan.faces.split(fan.added)
    tri_layered, quad_layered = fan.faces.split(layered)
    kept, fanned_cells = [], []
    for cells, added, corners in (
        (prisms, tri_added[tri_layered], 3),
        (hexahedra, quad_added[quad_layered], 4),
    ):
        per_layer = cells.reshape(-1, added.size, 2 * corners)
        kept.append(renumber[per_layer[:, ~added].reshape(-1, 2 * corners)])
        fanned_cells.append((per_layer[0, added], per_layer[1:, added].reshape(-1, 2 * corners)))
    (first_tri, upper_tri), (first_quad, upper_quad) = fanned_cells
    wedges = np.array([_wedge(cell, renumber) for cell in first_quad], dtype=np.int64)
    bottoms = renumber[first_tri[:, :3]]
    in_corner = np.all(bottoms == bottoms[:, :1], axis=1)
    tetrahedra = np.column_stack([bottoms[in_corner, 0], renumber[first_tri[in_corner, 3:]]])
    return MergedLayers(
        renumber=renumber,
        prisms=np.vstack([kept[0], wedges.reshape(-1, 6), renumber[upper_tri]]),
        hexahedra=np.vstack([kept[1], renumber[upper_quad]]),
        tetrahedra=tetrahedra.reshape(-1, 4),
        tents=_tents(first_tri[~in_corner], renumber, middle),
    )


def _wedge(cell: np.ndarray, renumber: np.ndarray) -> list[int]:
    """The first layer's cell over a quadrilateral of no area, as a wedge: the two copies at each
    end are one point, so the quadrilateral's sides between them are its triangles."""
    bottom, top = renumber[cell[:4]], renumber[cell[4:]]
    for i in range(4):
        if bottom[i] == bottom[(i + 1) % 4] and bottom[(i + 2) % 4] == bottom[(i + 3) % 4]:
            j, k, m = (i + 1) % 4, (i + 2) % 4, (i + 3) % 4
            return [bottom[i], top[i], top[j], bottom[m], top[m], top[k]]
    raise ValueError(f"the cell over a face of no area {cell[:4].tolist()} is not a wedge")


def _tents(cells: np.ndarray, renumber: np.ndarray, middle: np.ndarray) -> np.ndarray:
    """The first layer's cells over the triangles of no area, as tents, one per middle copy.

    Where the fan closes down onto a wall along an edge ``p q``, two triangles of no area join
    ``q`` to ``p``'s copies, one each side of the middle one. The two prisms over them, their
    copies merged, would share two faces (the edge ``p q`` runs through both), which no mesh of
    cells allows; together they are one cell standing on that edge, its top the columns' four tops.
    """
    by_middle: dict[int, list[tuple[int, int, np.ndarray]]] = {}
    for cell in cells:
        bottom = cell[:3]
        k = int(np.flatnonzero(middle[bottom])[0])
        side = next(i for i in range(3) if i != k and renumber[bottom[i]] == renumber[bottom[k]])
        q = 3 - k - side
        by_middle.setdefault(int(bottom[k]), []).append((side, q, cell))
    tents = []
    for runs in by_middle.values():
        if len(runs) != 2:
            raise ValueError(f"a fan closes onto a wall with {len(runs)} cell(s), not two")
        (side_a, q_a, a), (side_b, q_b, b) = runs
        if renumber[a[q_a]] != renumber[b[q_b]]:
            raise ValueError("a fan's two cells closing onto a wall stand on different edges")
        k = 3 - side_a - q_a
        tents.append([a[side_a], a[q_a], a[3 + q_a], a[3 + side_a], a[3 + k], b[3 + side_b]])
    return renumber[np.array(tents, dtype=np.int64).reshape(-1, 6)]
