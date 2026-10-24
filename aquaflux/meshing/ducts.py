"""Rectangular ducts in a region: found from its faces, divided for a structured core, and their rings.

A duct is a part of the region bounded round by four four-sided faces -- its sides -- each meeting
the next along an edge where the region folds convexly by more than the feature angle (a *rail*,
one of the duct's four long corners), and each running from one *rim* of four edges (a cross-section
of the duct) to another. One such ring of four faces is a *segment*. Segments that share a rim are
one duct, so a duct may run along a curve as a chain of segments -- a straight leg, a bend, another
leg, as a sweep along a path is drawn -- or as one segment of curved faces. Nothing else may lie
inside it.

A duct's sides are meshed as structured grids (:class:`~aquaflux.meshing.MappedQuads`), so its wall
carries hexahedral layers, and the core the layers leave is filled cross-section by cross-section
(:class:`~aquaflux.meshing.DuctCore`); its ends are open or junctions as any conduit's
(:mod:`~aquaflux.meshing.conduits`). For the grids to join round the duct and line up along it, every
rail of a segment has one count of segments, and each pair of opposite sides one count across, all
along the duct; and every one of those edges is divided at equal arc length, so the rows of the four
sides meet at the same cross-sections however much longer the outer rail of a bend is than the
inner. :func:`duct_divisions` says how to divide them, never more coarsely than the triangulation
chose.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from aquaflux.io.cad import CadTopology

from .mapped import _loops, _resampled
from .netgen_process import EdgePoints
from .regions import parity_votes

__all__ = ["Duct", "duct_divisions", "duct_rings", "find_ducts"]

#: How near a point of a face's loop must lie to one of the drawing's vertices to be that vertex,
#: relative to the triangulation's extent.
CORNER_TOLERANCE = 1e-9

#: How many times a segment's centreline is taken again from the crossings of its cross-sections.
STATION_PASSES = 3

#: Samples per segment along each rail at which its crossings with the cross-sections are sought.
STATION_SAMPLES = 32


@dataclasses.dataclass(frozen=True)
class Duct:
    """A rectangular duct: its sides, segment by segment along it.

    Attributes
    ----------
    sides : np.ndarray of int, shape ``(n_segments, 4)``
        Each segment's four faces, in order round the duct; side ``i`` of one segment continues
        side ``i`` of the one before.
    rails : np.ndarray of int, shape ``(n_segments, 4)``
        Each segment's edge between its sides ``i`` and ``i + 1`` (the last and the first for
        ``i = 3``).
    rims : np.ndarray of int, shape ``(n_segments + 1, 4)``
        Each side's edge across the duct at the start of each segment, then at the end of the last:
        row ``k`` the rim between segments ``k - 1`` and ``k``, rows ``0`` and ``n_segments`` the
        duct's two ends.
    """

    sides: np.ndarray
    rails: np.ndarray
    rims: np.ndarray

    @property
    def faces(self) -> list[int]:
        """Its faces, ascending."""
        return sorted(self.sides.reshape(-1).tolist())


@dataclasses.dataclass(frozen=True)
class _Segment:
    """A ring of four faces, as :class:`Duct` holds one of its segments: its sides, its rails, and
    each side's edge across at the segment's start and at its end."""

    sides: tuple[int, ...]
    rails: tuple[int, ...]
    start: tuple[int, ...]
    end: tuple[int, ...]

    def key(self) -> tuple[frozenset, frozenset]:
        """What makes two segments the same, however their sides are numbered."""
        return frozenset(self.sides), frozenset(self.rails)

    def starting_at(self, rim: tuple[int, ...], topology: CadTopology) -> _Segment:
        """The same segment, its sides renumbered and its ends swapped so that side ``i``'s edge at
        its start is ``rim[i]``."""
        by_edge = {}
        for k, side in enumerate(self.sides):
            by_edge[self.start[k]] = (side, self.end[k])
            by_edge[self.end[k]] = (side, self.start[k])
        sides = tuple(by_edge[edge][0] for edge in rim)
        end = tuple(by_edge[edge][1] for edge in rim)
        rails = tuple(_shared(topology, sides[i], sides[(i + 1) % 4]) for i in range(4))
        return _Segment(sides, rails, tuple(rim), end)

    def length(self, topology: CadTopology) -> float:
        """The mean length of its rails."""
        return float(np.mean(topology.edge_length[list(self.rails)]))


def find_ducts(
    topology: CadTopology, points: np.ndarray, triangles: np.ndarray, face: np.ndarray,
    candidates: np.ndarray, feature_angle: float, edge_counts: bool = True,
) -> tuple[list[Duct], dict[int, str]]:  # fmt: skip
    """The ducts whose every side is one of ``candidates``, and why each other candidate is not a
    duct's side.

    Parameters
    ----------
    topology : CadTopology
    points, triangles, face : np.ndarray
        The region's surface triangulation, each triangle tagged by its face.
    candidates : np.ndarray of int
        The faces that may be ducts' sides.
    feature_angle : float
        The fold, in radians, beyond which an edge is sharp: a rail folds convexly by more.
    edge_counts : bool
        Whether a duct's edges must already be divided as its grids need (:func:`duct_divisions`).
        Off, a duct is one whatever its triangulation's divisions, as one triangulated again with
        its edges divided so would be.

    Returns
    -------
    ducts : list of Duct
    refused : dict of {int: str}
        For each candidate that is not a duct's side, the reason.

    Notes
    -----
    Where a face is a side of ducts running two ways -- a box's four walls round one axis, or
    round another -- the duct longest for its width is taken.
    """
    candidates = set(np.asarray(candidates, dtype=np.int64).tolist())
    refused: dict[int, str] = {}
    edges_of = {}
    for f in sorted(candidates):
        edges, reason = _four_edges(topology, f)
        if edges is None:
            refused[f] = reason
        else:
            edges_of[f] = edges
    segments: dict[tuple, _Segment] = {}
    for f in edges_of:
        for first in (0, 1):
            segment = _ring(topology, edges_of, f, first, feature_angle)
            if segment is not None:
                segments.setdefault(segment.key(), segment)
    chains, closed = _chains(topology, list(segments.values()))
    for chain in closed:
        for segment in chain:
            for side in segment.sides:
                refused.setdefault(side, "its duct closes on itself, with no end")
    ducts, taken = [], set()
    extent = _extent(points)
    for chain in sorted(chains, key=lambda c: (-_slenderness(topology, c), min(_faces(c)))):
        faces = _faces(chain)
        if faces & taken:
            for side in faces - taken:
                refused.setdefault(side, "it is a side of a duct running another way, taken")
            continue
        duct = _duct(chain)
        if edge_counts and not _divided(topology, points, triangles, face, duct, extent):
            for side in faces:
                refused[side] = (
                    "its duct's rails or opposite sides are not divided alike, as its grids need"
                )
            continue
        if not _hollow(topology, points, triangles, face, duct):
            for side in faces:
                refused[side] = "something else lies inside its duct"
            continue
        ducts.append(duct)
        taken |= faces
    refused = {f: reason for f, reason in refused.items() if f not in taken}
    for f in edges_of:
        if f not in taken:
            refused.setdefault(
                f,
                "its edges folding convexly beyond the feature angle do not make it one of four "
                "faces round a duct",
            )
    return ducts, refused


def duct_divisions(
    topology: CadTopology, points: np.ndarray, triangles: np.ndarray, face: np.ndarray,
    ducts: list[Duct],
) -> dict[int, int | EdgePoints]:  # fmt: skip
    """How to divide every edge of the ducts so that their grids join round each duct and line up
    along it.

    Each pair of opposite sides gets, all along the duct, the most segments the triangulation gave
    any of their edges across, each edge divided at equal arc length. Each segment's four rails get
    the most segments it gave any of them, their points where the segment's cross-sections normal
    to its centreline cross them (:func:`_stations`), so that the grids' rows are those
    cross-sections however the rails curve; where a rail's points cannot be read from the
    triangulation, the segment's rails are divided at equal arc length. An edge across two ducts
    takes the larger count.

    Parameters
    ----------
    topology : CadTopology
    points, triangles, face : np.ndarray
        The region's surface triangulation, each triangle tagged by its face.
    ducts : list of Duct

    Returns
    -------
    dict of {int: int or EdgePoints}
        Segments per edge across, and points per rail, by the edge's index in ``topology``.
    """
    extent = _extent(points)
    divisions: dict[int, int | EdgePoints] = {}
    for duct in ducts:
        lines = _edge_lines(topology, points, triangles, face, duct.faces, extent)
        for pair in ((0, 2), (1, 3)):
            across = duct.rims[:, pair].reshape(-1).tolist()
            count = max(_segments(lines, e) for e in across)
            for edge in across:
                held = divisions.get(edge, 0)
                divisions[edge] = max(held if isinstance(held, int) else 0, count)
        for k, rails in enumerate(duct.rails.tolist()):
            # Each rail's line, from the end on the segment's first rim.
            at_start = {v for e in duct.rims[k].tolist() for v in topology.edge_vertices[e]}
            starts = [next(v for v in topology.edge_vertices[e].tolist() if v in at_start)
                      for e in rails]  # fmt: skip
            count = max(_segments(lines, e) for e in rails)
            if not all(e in lines for e in rails):
                # A rail the triangulation could not be read along: equal arc length, by count.
                divisions.update(dict.fromkeys(rails, count))
                continue
            along = [_from(topology, lines[e], e, v) for e, v in zip(rails, starts, strict=True)]
            for edge, start, fractions in zip(rails, starts, _stations(along, count), strict=True):
                divisions[edge] = EdgePoints(tuple(fractions), tuple(topology.vertices[start]))
    return divisions


def _stations(rails: list[np.ndarray], segments: int) -> list[np.ndarray]:
    """Where a duct segment's ``segments - 1`` interior cross-sections cross each of its rails, as
    fractions of each rail's length: cross-sections at equal steps along the segment's centreline,
    each normal to it -- but turned, within a width of either rim, toward that rim's own plane, so
    that a rim cut obliquely across the duct is met by cross-sections that lean as it does.

    The centreline is the mean of the four rails, first at equal fractions of their lengths and then
    at the crossings found, :data:`STATION_PASSES` times. Where a cross-section does not cross a rail
    in order (a segment too sharply curved for its width), every rail keeps its points at equal arc
    length.

    Parameters
    ----------
    rails : list of np.ndarray, each shape ``(n, 3)``
        The segment's four rails as polylines, in order round it, each from the segment's first rim
        to its last.
    segments : int

    Returns
    -------
    list of np.ndarray
        For each rail, its interior points' fractions of its length, ``segments - 1`` increasing.
    """
    equal = np.linspace(0.0, 1.0, segments + 1)
    if segments < 2:
        return [equal[1:-1] for _ in rails]
    dense = np.linspace(0.0, 1.0, STATION_SAMPLES * segments + 1)
    sampled = np.stack([_resampled(rail, dense, closed=False) for rail in rails])
    corners = [np.array([rail[0] for rail in rails]), np.array([rail[-1] for rail in rails])]
    width = float(np.mean(np.linalg.norm(corners[0] - np.roll(corners[0], -1, axis=0), axis=1)))
    rims = [np.linalg.svd(c - c.mean(axis=0))[2][-1] for c in corners]
    crossing = np.tile(equal, (len(rails), 1))
    for _ in range(STATION_PASSES):
        centre = np.mean(
            [_resampled(rail, at, closed=False) for rail, at in zip(rails, crossing, strict=True)],
            axis=0,
        )
        length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1))])
        along = equal * length[-1]
        stations = np.stack([np.interp(along, length, centre[:, k]) for k in range(3)], axis=1)
        tangent = np.gradient(stations, axis=0, edge_order=2)  # second order at the ends too
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        normal = tangent.copy()
        for rim, at, distance in ((rims[0], 0, along), (rims[1], -1, length[-1] - along)):
            rim = rim if rim @ tangent[at] > 0.0 else -rim
            share = np.clip(1.0 - distance / width, 0.0, 1.0)[:, None]
            normal += share * (rim - tangent[at])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        height = (
            np.einsum("rsk,tk->rts", sampled, normal)
            - np.sum(stations * normal, axis=1)[None, :, None]
        )
        found = np.array([
            [0.0, *(_crossing(dense, height[r, k], crossing[r, k]) for k in range(1, segments)), 1.0]
            for r in range(len(rails))
        ])  # fmt: skip
        if not np.all(np.diff(found, axis=1) > 0.0):
            return [equal[1:-1] for _ in rails]
        crossing = found
    return [at[1:-1] for at in crossing]


def _crossing(fractions: np.ndarray, height: np.ndarray, guess: float) -> float:
    """Where ``height``, sampled at ``fractions``, crosses zero nearest ``guess``; ``guess`` if it
    does not."""
    changes = np.flatnonzero(np.sign(height[:-1]) != np.sign(height[1:]))
    if not changes.size:
        return guess
    roots = fractions[changes] - height[changes] * (
        (fractions[changes + 1] - fractions[changes]) / (height[changes + 1] - height[changes])
    )
    return float(roots[np.argmin(np.abs(roots - guess))])


def duct_rings(duct: Duct, grids) -> tuple[np.ndarray, int]:
    """A duct's rows of points round it, from its sides' grids.

    Parameters
    ----------
    duct : Duct
    grids : mapping of {int: np.ndarray of int}
        Each side's grid of points (:attr:`~aquaflux.meshing.MappedSurface.grids`).

    Returns
    -------
    rings : np.ndarray of int, shape ``(rows + 1, 2 (width + height))``
        One ring per row along the duct, from its first rim to its last, each starting at the
        corner between sides 3 and 0 and running round side 0 (``width`` segments), side 1
        (``height``), side 2 and side 3, each in the same direction.
    width : int

    Raises
    ------
    ValueError
        If a side was not meshed as a grid, or the grids do not join round the duct.
    """
    rows, width = [], None
    for k, sides in enumerate(duct.sides.tolist()):
        missing = [f for f in sides if f not in grids]
        if missing:
            raise ValueError(f"the duct's side(s) {missing} were not meshed as structured grids")
        oriented = _round(sides, grids)
        if width is None:
            width = oriented[0].shape[1] - 1
        ring_rows = np.concatenate([g[:, :-1] for g in oriented], axis=1)
        if k == 0 and len(duct.sides) > 1:
            following = set(np.concatenate([np.asarray(grids[f]).reshape(-1)
                                            for f in duct.sides[1].tolist()]).tolist())  # fmt: skip
            if set(ring_rows[0].tolist()) <= following:
                ring_rows = ring_rows[::-1]
        elif k > 0:
            if set(ring_rows[-1].tolist()) == set(rows[-1][-1].tolist()):
                ring_rows = ring_rows[::-1]
            if not np.array_equal(ring_rows[0], rows[-1][-1]):
                raise ValueError("a duct's segments' grids do not meet row for row at their rim")
            ring_rows = ring_rows[1:]
        rows.append(ring_rows)
    return np.concatenate(rows), width


def _round(sides: list[int], grids) -> list[np.ndarray]:
    """A segment's four grids, each turned so that its rows run across the duct -- from the corner
    it shares with the side before to the corner it shares with the next -- and its first row at
    the same end of the segment as the others'."""
    point_sets = [set(np.asarray(grids[f]).reshape(-1).tolist()) for f in sides]
    oriented = []
    for i, f in enumerate(sides):
        grid = np.asarray(grids[f])
        before, after = point_sets[i - 1], point_sets[(i + 1) % 4]
        for candidate in (grid, grid.T):
            if set(candidate[:, 0].tolist()) <= before and set(candidate[:, -1].tolist()) <= after:
                grid = candidate
                break
            if set(candidate[:, -1].tolist()) <= before and set(candidate[:, 0].tolist()) <= after:
                grid = candidate[:, ::-1]
                break
        else:
            raise ValueError(f"the grid of the duct's side {f} does not meet its neighbours'")
        if i > 0 and grid[0, 0] != oriented[-1][0, -1]:
            grid = grid[::-1]
        if i > 0 and grid[0, 0] != oriented[-1][0, -1]:
            raise ValueError(f"the grid of the duct's side {f} does not meet its neighbours'")
        oriented.append(grid)
    if oriented[-1][0, -1] != oriented[0][0, 0] or oriented[0].shape[0] != oriented[2].shape[0]:
        raise ValueError("the grids of a duct's four sides do not close round it")
    return oriented


def _four_edges(topology: CadTopology, f: int):
    """Face ``f``'s four edges in order round it, each sharing a vertex with the next, or None and
    why not."""
    edges = [
        e
        for e in topology.edges_of_face(f).tolist()
        if not topology.seam[e] and topology.edge_length[e] > 0.0
    ]
    if len(edges) != 4:
        return None, f"it is bounded by {len(edges)} edges, not the four of a duct's side"
    ends = {e: set(topology.edge_vertices[e].tolist()) for e in edges}
    if any(len(v) != 2 for v in ends.values()) or len(set().union(*ends.values())) != 4:
        return None, "its four edges do not make one loop of four corners"
    loop = [edges[0]]
    while len(loop) < 4:
        here = loop[-1]
        following = [e for e in edges if e not in loop and ends[e] & ends[here]]
        if not following:
            return None, "its four edges do not make one loop of four corners"
        loop.append(following[0])
    if not ends[loop[-1]] & ends[loop[0]]:
        return None, "its four edges do not make one loop of four corners"
    return loop, None


def _ring(topology, edges_of, f, first, feature_angle) -> _Segment | None:
    """The segment round face ``f`` with its rails its edges ``first`` and ``first + 2``, if the
    faces across them close a ring of four duct sides."""
    rail_in, rail = edges_of[f][first], edges_of[f][first + 2]
    sides, rails = [f], []
    here = f
    for _ in range(4):
        if not _is_rail(topology, rail, feature_angle):
            return None
        across = [g for g in topology.faces_of_edge(rail).tolist() if g != here]
        if len(across) != 1:
            return None
        rails.append(rail)
        here = across[0]
        if len(rails) == 4:
            break
        if here not in edges_of or here in sides:
            return None
        loop = edges_of[here]
        rail = loop[(loop.index(rail) + 2) % 4]
        sides.append(here)
    if here != f or rails[-1] != rail_in:
        return None
    cross = []
    for i, side in enumerate(sides):
        mine = {rails[i - 1], rails[i]}
        cross.append([e for e in edges_of[side] if e not in mine])
    # Each side's edge at the start meets the one before's at the start end of their rail.
    start_vertex = set(topology.edge_vertices[rails[-1]].tolist())
    start, end = [], []
    corner = min(start_vertex)
    for i in range(4):
        a, b = cross[i]
        at_start = a if corner in topology.edge_vertices[a] else b
        start.append(at_start)
        end.append(b if at_start == a else a)
        corner = int(next(v for v in topology.edge_vertices[at_start].tolist() if v != corner))
    return _Segment(tuple(sides), tuple(rails), tuple(start), tuple(end))


def _is_rail(topology: CadTopology, edge: int, feature_angle: float) -> bool:
    """Whether ``edge`` folds convexly by more than ``feature_angle`` between two faces."""
    return topology.faces_of_edge(edge).size == 2 and bool(
        topology.edge_angle[edge] > feature_angle
    )


def _shared(topology: CadTopology, a: int, b: int) -> int:
    """The one edge faces ``a`` and ``b`` share."""
    shared = np.intersect1d(topology.edges_of_face(a), topology.edges_of_face(b))
    return int(shared[0])


def _chains(topology, segments: list[_Segment]):
    """The segments joined where they share a rim: open chains, each ordered from one free rim to
    the other, and closed ones."""
    by_rim: dict[frozenset, list[int]] = {}
    for k, segment in enumerate(segments):
        for rim in (segment.start, segment.end):
            by_rim.setdefault(frozenset(rim), []).append(k)

    def across(k, rim):
        return next((j for j in by_rim[frozenset(rim)] if j != k), None)

    left, chains, closed = set(range(len(segments))), [], []
    while left:
        component, reach = set(), [min(left)]
        while reach:
            k = reach.pop()
            if k not in component:
                component.add(k)
                reach += [j for rim in (segments[k].start, segments[k].end)
                          if (j := across(k, rim)) is not None]  # fmt: skip
        left -= component
        free = sorted(
            (k, rim)
            for k in component
            for rim in (segments[k].start, segments[k].end)
            if across(k, rim) is None
        )
        if not free:
            closed.append([segments[k] for k in sorted(component)])
            continue
        k, rim = free[0]
        chain = [segments[k].starting_at(rim, topology)]
        while (j := across(k, chain[-1].end)) is not None:
            chain.append(segments[j].starting_at(chain[-1].end, topology))
            k = j
        chains.append(chain)
    return chains, closed


def _faces(chain: list[_Segment]) -> set[int]:
    return {side for segment in chain for side in segment.sides}


def _slenderness(topology: CadTopology, chain: list[_Segment]) -> float:
    """A chain's length over its mean width across."""
    across = float(np.mean(topology.edge_length[list(chain[0].start)]))
    return sum(segment.length(topology) for segment in chain) / across


def _duct(chain: list[_Segment]) -> Duct:
    return Duct(
        sides=np.array([s.sides for s in chain], dtype=np.int64),
        rails=np.array([s.rails for s in chain], dtype=np.int64),
        rims=np.array([chain[0].start] + [s.end for s in chain], dtype=np.int64),
    )


def _extent(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def _edge_lines(topology, points, triangles, face, faces, extent) -> dict[int, np.ndarray]:
    """The triangulation's points along each edge of the four-sided ``faces``, read from each
    face's loop between its corners: a polyline per edge, from its first vertex to its second."""
    lines: dict[int, np.ndarray] = {}
    for f in faces:
        loops = _loops(triangles[face == f])
        if len(loops) != 1:
            continue
        loop = loops[0]
        edges, _ = _four_edges(topology, f)
        corners = sorted({v for e in edges for v in topology.edge_vertices[e].tolist()})
        at = {}
        for v in corners:
            distance = np.linalg.norm(points[loop] - topology.vertices[v], axis=1)
            k = int(np.argmin(distance))
            if distance[k] <= CORNER_TOLERANCE * extent:
                at[k] = v
        if len(at) != 4:
            continue
        positions = sorted(at)
        for i, k in enumerate(positions):
            following = positions[(i + 1) % 4]
            run = loop[np.arange(k, k + (following - k) % loop.size + 1) % loop.size]
            pair = (at[k], at[following])
            edge = next(e for e in edges if set(topology.edge_vertices[e].tolist()) == set(pair))
            if pair[0] != topology.edge_vertices[edge][0]:
                run = run[::-1]
            lines.setdefault(edge, points[run])
    return lines


def _segments(lines: dict[int, np.ndarray], edge: int) -> int:
    """The segments the triangulation divides ``edge`` into; one where it was not read."""
    return lines[edge].shape[0] - 1 if edge in lines else 1


def _from(topology: CadTopology, line: np.ndarray, edge: int, vertex: int) -> np.ndarray:
    """An edge's polyline (from its first vertex) run from ``vertex``, one of its ends."""
    return line if topology.edge_vertices[edge][0] == vertex else line[::-1]


def _divided(topology, points, triangles, face, duct: Duct, extent) -> bool:
    """Whether the triangulation divides the duct's edges as its grids need: each segment's rails
    alike, each pair of opposite sides' edges across alike all along it."""
    lines = _edge_lines(topology, points, triangles, face, duct.faces, extent)
    groups = [rails.tolist() for rails in duct.rails]
    groups += [duct.rims[:, pair].reshape(-1).tolist() for pair in ((0, 2), (1, 3))]
    for group in groups:
        found = {_segments(lines, e) if e in lines else None for e in group}
        if len(found) != 1 or None in found:
            return False
    return True


def _hollow(topology, points, triangles, face, duct: Duct) -> bool:
    """Whether no point of the surface but the duct's own and its ends' lies inside it: the duct's
    sides closed at each rim by a fan of triangles from the rim's centroid.

    A face across one of the duct's two end rims -- a cap, or a wall the duct opens through -- is
    the duct's end, whose points lie on that fan or beyond its rim, so it is not looked at.
    """
    own_faces = duct.faces
    own = triangles[np.isin(face, own_faces)]
    loops = _loops(own)
    if len(loops) != 2:
        return False
    closing, centres = [], []
    for loop in loops:
        centre = points.shape[0] + len(centres)
        centres.append(points[loop].mean(axis=0))
        closing.append(np.stack([np.roll(loop, -1), loop, np.full(loop.size, centre)], axis=1))
    vertices = np.vstack([points, centres])[np.vstack([own, *closing])]
    ends = {
        g
        for edge in np.concatenate([duct.rims[0], duct.rims[-1]]).tolist()
        for g in topology.faces_of_edge(edge).tolist()
    } - set(own_faces)
    others = np.setdiff1d(np.unique(triangles[~np.isin(face, [*own_faces, *ends])]), own)
    low, high = vertices.reshape(-1, 3).min(axis=0), vertices.reshape(-1, 3).max(axis=0)
    near = others[np.all((points[others] >= low) & (points[others] <= high), axis=1)]
    if not near.size:
        return True
    votes = parity_votes(points[near], vertices)
    return not np.any(votes.sum(axis=1) * 2 > votes.shape[1])
