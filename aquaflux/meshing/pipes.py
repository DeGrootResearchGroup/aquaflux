"""Pipes in a region: found from its faces, and their cores filled with structured O-grids.

A pipe is a face of the region shaped as a tube with the fluid inside it -- a straight length, a
bend, a coil, a reducer, a pipe swept along any curve -- bounded by two loops that a structured grid
joins row by row (:class:`~aquaflux.meshing.MappedQuads`), with nothing else inside it. It is told
from the face's own triangulation, not from the kind of surface the drawing records: closed across
its two rims, the face must enclose a volume (so it is a tube, not a flat ring), with its triangles
facing out of it (the fluid is inside, not round it as round a bore), and no other point of the
region's surface inside it. Its wall is meshed as that grid and, where the walls carry layers, layered like any other wall;
the core the layers leave is filled with a :class:`~aquaflux.meshing.PipeCore` O-grid, which meets the
layers' top quadrilateral for quadrilateral.

The O-grid needs both loops of one point count, a multiple of four. The loops' points are the surface
triangulation's, whose division of the rims follows the mesh size, so :func:`rim_divisions` says how to
divide each candidate's rim edges for the triangulation to give an O-grid's counts.

Pipes that share a rim -- a straight length, a bend, another straight length -- are one **run**
(:func:`pipe_runs`), filled by one O-grid from end to end, so the cross-section where two of them
meet is shared rather than left to the fill. At each end of a run:

- **an open end** -- the pipe ends on a flat face without layers, such as an inlet, bounded only by
  the pipe's own rim -- the O-grid runs to the end, and its last cross-section takes the place of
  that face's own faces inside the layers' ring, in the same patch;
- **a junction** -- the pipe opens into the rest of the region -- the O-grid stops ``junction_rows``
  rows short, where the layers' columns still run straight, and the rows beyond are left to the
  tetrahedral fill as before: the O-grid's last cross-section is the interface the fill meets.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from aquaflux.io.cad import CadTopology

from .mapped import _loops
from .pipe_core import PipeCore
from .polygons import Polygons
from .regions import ClosedSurface

#: How much volume, as a share of its area times its rims' radius, a face closed across its two
#: rims must hold to be a tube: a tube holds half (its volume pi r^2 L over 2 pi r L times r), a
#: flat ring none; the sign says which side the fluid is on.
TUBE_VOLUME = 0.1

__all__ = [
    "PipeEnds",
    "fill_pipes",
    "find_pipes",
    "open_ends",
    "pipe_ends",
    "pipe_runs",
    "rim_divisions",
]


@dataclasses.dataclass(frozen=True)
class PipeEnds:
    """A run of pipes' faces and what each end of its grid meets.

    Attributes
    ----------
    faces : tuple of int
        The run's faces, in order along it.
    first, last : int
        For the grid's first and last row: the flat face it ends on (an open end), or ``-1`` where
        the pipe opens into the rest of the region (a junction).
    """

    faces: tuple[int, ...]
    first: int
    last: int


def find_pipes(
    topology: CadTopology, points: np.ndarray, triangles: np.ndarray, face: np.ndarray,
    candidates: np.ndarray, rim_counts: bool = True,
) -> tuple[list[int], dict[int, str]]:  # fmt: skip
    """Which of ``candidates`` are pipes, and why each other is not.

    Parameters
    ----------
    topology : CadTopology
    points, triangles, face : np.ndarray
        The region's surface triangulation, each triangle tagged by its face.
    candidates : np.ndarray of int
        The faces to look at.
    rim_counts : bool
        Whether a pipe's rims must already have the O-grid's counts: equal, a multiple of four.
        Off, a face is a pipe whatever its triangulation's rims, as one triangulated again with its
        rims divided alike (:func:`rim_divisions`) would be.

    Returns
    -------
    pipes : list of int
    refused : dict of {int: str}
        For each candidate that is not a pipe, the reason.
    """
    pipes, refused = [], {}
    for f in np.asarray(candidates, dtype=np.int64).tolist():
        own = triangles[face == f]
        tube = _TubeShape.of(points, own)
        if isinstance(tube, str):
            refused[f] = tube
            continue
        sizes = sorted(loop.size for loop in tube.rims)
        if rim_counts and (sizes[0] != sizes[1] or sizes[0] % 4):
            refused[f] = (
                f"its rims have {sizes[0]} and {sizes[1]} points; an O-grid needs equal counts, "
                "a multiple of four"
            )
            continue
        caps = [g for g in _neighbours(topology, f) if _is_cap(topology, g, [f])]
        others = np.setdiff1d(np.unique(triangles[~np.isin(face, [f, *caps])]), np.unique(own))
        if np.any(tube.holds(points[others])):
            refused[f] = "something else lies inside it"
            continue
        pipes.append(f)
    return pipes, refused


@dataclasses.dataclass(frozen=True)
class _TubeShape:
    """A face's triangles closed across its two rims, each by a fan from the rim's centroid.

    Attributes
    ----------
    rims : tuple of np.ndarray of int
        The face's two loops, in the triangles' winding.
    points : np.ndarray, shape ``(n, 3)``
        The surface's points, then the two rims' centroids.
    triangles : np.ndarray of int, shape ``(m, 3)``
        The closed surface: the face's triangles, then the fans across its rims.
    """

    rims: tuple[np.ndarray, np.ndarray]
    points: np.ndarray
    triangles: np.ndarray

    @classmethod
    def of(cls, points: np.ndarray, own: np.ndarray) -> _TubeShape | str:
        """The face of triangles ``own`` (shape ``(m, 3)``, wound out of the region) as a closed
        tube, or why it is not a pipe's: not two rims, the fluid round it rather than in it, or
        no volume between its rims."""
        try:
            loops = _loops(own)
        except ValueError as irregular:
            return f"its boundary is not simple loops: {irregular}"
        if len(loops) != 2:
            return f"it is bounded by {len(loops)} loops, not the two of a band"
        centres = np.stack([points[loop].mean(axis=0) for loop in loops])
        extended = np.vstack([points, centres])
        fans = [
            np.stack([np.roll(loop, -1), loop, np.full(loop.size, points.shape[0] + k)], axis=1)
            for k, loop in enumerate(loops)
        ]  # each rim's edges the other way round, so every edge is used once each way
        closed = np.vstack([own, *fans])
        tube = cls((loops[0], loops[1]), extended, closed)
        corner = extended[closed]
        volume = (
            float(np.einsum("ij,ij->i", corner[:, 0], np.cross(corner[:, 1], corner[:, 2])).sum())
            / 6.0
        )
        normal = np.cross(
            points[own[:, 1]] - points[own[:, 0]], points[own[:, 2]] - points[own[:, 0]]
        )
        area = 0.5 * float(np.linalg.norm(normal, axis=1).sum())
        radius = float(np.mean([np.linalg.norm(points[loop] - centres[k], axis=1).mean()
                                for k, loop in enumerate(loops)]))  # fmt: skip
        if volume < -TUBE_VOLUME * area * radius:
            return "the fluid is outside it, not inside"
        if volume <= TUBE_VOLUME * area * radius:
            return "closed across its rims it holds no tube's volume (a flat ring, or a short band)"
        return tube

    def holds(self, points: np.ndarray) -> np.ndarray:
        """Which of ``points`` (shape ``(k, 3)``) lie inside the closed tube, shape ``(k,)``."""
        inside = np.zeros(points.shape[0], dtype=bool)
        corner = self.points[self.triangles]
        low, high = corner.reshape(-1, 3).min(axis=0), corner.reshape(-1, 3).max(axis=0)
        near = np.all((points > low) & (points < high), axis=1)
        if not near.any():
            return inside
        edges = np.linalg.norm(corner - np.roll(corner, 1, axis=1), axis=2)
        region = ClosedSurface.of(self.points, self.triangles, float(edges.mean()))
        inside[near] = region.inside(points[near])
        return inside


def rim_divisions(
    topology: CadTopology, points: np.ndarray, triangles: np.ndarray, face: np.ndarray,
    candidates: np.ndarray,
) -> dict[int, int]:  # fmt: skip
    """The segments to ask of each rim edge so that every pipe candidate's rims can be joined by
    an O-grid: both rims of one count, a multiple of four.

    A candidate is a face that closes into a tube with the fluid inside it, bounded by two loops of
    the triangulation (the rest are left to :func:`find_pipes` to refuse). Candidates sharing a rim are
    one run, given one count, so that the rims each shares are divided as both need. Where a
    run's loops already agree and are a multiple of four nothing is asked. Otherwise every rim of
    the run gets its largest count rounded up to a multiple of four, so a rim is never divided
    more coarsely than the triangulation chose; a rim drawn as several edges shares its count
    among them by length, each edge at least one segment.

    Parameters
    ----------
    topology : CadTopology
    points, triangles, face : np.ndarray
        The region's surface triangulation: points ``(n_points, 3)``, triangles
        ``(n_triangles, 3)``, and each triangle's face ``(n_triangles,)``.
    candidates : np.ndarray of int
        The faces to look at.

    Returns
    -------
    dict of {int: int}
        Segments per edge, by the edge's index in ``topology``; empty when no candidate needs any.
    """
    sizes, rims = {}, {}
    for f in np.asarray(candidates, dtype=np.int64).tolist():
        tube = _TubeShape.of(points, triangles[face == f])
        found = _rims(topology, f)
        if isinstance(tube, str) or len(found) != 2:
            continue
        sizes[f], rims[f] = [loop.size for loop in tube.rims], found
    divisions: dict[int, int] = {}
    for run in _joined(rims):
        counts = [size for f in run for size in sizes[f]]
        if len(set(counts)) == 1 and counts[0] % 4 == 0:
            continue
        count = 4 * math.ceil(max(counts) / 4)
        for f in run:
            for rim in rims[f]:
                for edge, segments in _shared_by_length(topology.edge_length[rim], count, rim):
                    divisions[edge] = max(divisions.get(edge, 0), segments)
    return divisions


def _joined(rims: dict[int, list[np.ndarray]]) -> list[list[int]]:
    """The faces whose rims share an edge, gathered into groups (each face in exactly one)."""
    group = {f: f for f in rims}

    def root(f):
        while group[f] != f:
            f = group[f]
        return f

    by_edge: dict[int, int] = {}
    for f, found in rims.items():
        for edge in np.concatenate(found).tolist():
            if edge in by_edge:
                group[root(f)] = root(by_edge[edge])
            else:
                by_edge[edge] = f
    groups: dict[int, list[int]] = {}
    for f in rims:
        groups.setdefault(root(f), []).append(f)
    return list(groups.values())


def _rims(topology: CadTopology, f: int) -> list[np.ndarray]:
    """Face ``f``'s boundary edges, seams and degenerate edges left out, grouped into the loops
    their shared vertices join."""
    edges = [
        e
        for e in topology.edges_of_face(f).tolist()
        if not topology.seam[e] and topology.edge_length[e] > 0.0
    ]
    group = {e: e for e in edges}

    def root(e):
        while group[e] != e:
            e = group[e]
        return e

    for i, a in enumerate(edges):
        for b in edges[i + 1 :]:
            if set(topology.edge_vertices[a].tolist()) & set(topology.edge_vertices[b].tolist()):
                group[root(b)] = root(a)
    rims: dict[int, list[int]] = {}
    for e in edges:
        rims.setdefault(root(e), []).append(e)
    return [np.array(rim, dtype=np.int64) for rim in rims.values()]


def _shared_by_length(lengths: np.ndarray, count: int, edges: np.ndarray):
    """``count`` segments shared among ``edges`` in proportion to their ``lengths`` (the largest
    remainders rounded up), each at least one: pairs of (edge, segments)."""
    share = np.maximum(count * lengths / lengths.sum(), 1.0)
    segments = np.floor(share).astype(np.int64)
    by_remainder = np.argsort(segments - share)
    for k in range(count - int(segments.sum())):
        segments[by_remainder[k % segments.size]] += 1
    return zip(edges.tolist(), segments.tolist(), strict=True)


def _neighbours(topology: CadTopology, f: int) -> list[int]:
    """The faces sharing an edge with face ``f``."""
    found = set()
    for edge in topology.edges_of_face(f):
        found.update(topology.faces_of_edge(edge).tolist())
    found.discard(f)
    return sorted(found)


def _is_cap(topology: CadTopology, end: int, pipes) -> bool:
    """Whether face ``end`` is bounded by nothing but the edges of the faces ``pipes``: a pipe's
    (or a run of pipes') end."""
    edges = set()
    for pipe in pipes:
        edges.update(topology.edges_of_face(pipe).tolist())
    return set(topology.edges_of_face(end).tolist()) <= edges


def open_ends(topology: CadTopology, pipe: int, layered: np.ndarray) -> list[int]:
    """The faces a pipe opens through, joining nothing: flat ends bounded by its rim alone (an
    inlet, an outlet) whose faces carry no layers -- the ends :func:`pipe_ends` finds open.

    Parameters
    ----------
    topology : CadTopology
    pipe : int
    layered : np.ndarray of bool, shape ``(n_faces,)``
        Which of the drawing's faces carry layers.
    """
    return [
        g for g in _neighbours(topology, pipe) if _is_cap(topology, g, [pipe]) and not layered[g]
    ]


def pipe_runs(grids: dict[int, np.ndarray]) -> list[tuple[tuple[int, ...], np.ndarray]]:
    """The pipes joined end to end at shared rims, each run as one grid.

    Parameters
    ----------
    grids : dict of {int: np.ndarray of int}
        Each pipe's grid, shape ``(rows + 1, n)``, by its face; grids sharing a rim share its
        points.

    Returns
    -------
    list of (tuple of int, np.ndarray of int)
        Each run's faces in order along it, and its grid: theirs joined, the shared rows once,
        every row round it in the same direction from the same column.

    Raises
    ------
    ValueError
        If three or more pipes share a rim, or the pipes close on themselves in a loop.
    """
    ends: dict[frozenset, list[int]] = {}
    for f, grid in grids.items():
        for row in (grid[0], grid[-1]):
            ends.setdefault(frozenset(row.tolist()), []).append(f)
    for faces in ends.values():
        if len(faces) > 2:
            raise ValueError(
                f"pipes {sorted(faces)} all meet at one rim; an O-grid runs through two"
            )

    def beyond(f, row):
        """The pipe across rim ``row`` of pipe ``f``, if any."""
        others = [g for g in ends[frozenset(row.tolist())] if g != f]
        return others[0] if others else None

    runs, seen = [], set()
    starts = [f for f, g in grids.items() if beyond(f, g[0]) is None or beyond(f, g[-1]) is None]
    for start in starts:
        if start in seen:
            continue
        grid = grids[start]
        if beyond(start, grid[0]) is not None:
            grid = grid[::-1]
        faces = [start]
        seen.add(start)
        while (after := beyond(faces[-1], grid[-1])) is not None:
            grid = np.vstack([grid, _continued(grid[-1], grids[after])[1:]])
            faces.append(after)
            seen.add(after)
        runs.append((tuple(faces), grid))
    if len(seen) != len(grids):
        raise ValueError(
            f"pipes {sorted(set(grids) - seen)} join end to end in a loop, which has no end to "
            "start an O-grid from"
        )
    return runs


def _continued(rim: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """``grid`` turned so its first row is ``rim``, point for point."""
    if set(grid[0].tolist()) != set(rim.tolist()):
        grid = grid[::-1]
    row = grid[0].tolist()
    at = row.index(int(rim[0]))
    if row[(at + 1) % len(row)] != int(rim[1]):
        grid = grid[:, ::-1]
        at = len(row) - 1 - at
    return np.roll(grid, -at, axis=1)


def pipe_ends(topology, grid: np.ndarray, run, faces: Polygons, tag, layered) -> PipeEnds:
    """What each end of a run of pipes' grid meets: the face it opens through
    (:func:`open_ends`), if any.

    Parameters
    ----------
    topology : CadTopology
    grid : np.ndarray of int, shape ``(rows + 1, n)``
        The run's grid, in the surface's numbering.
    run : sequence of int
        The run's faces, in order along it.
    faces : Polygons
        The surface, in the same numbering.
    tag : np.ndarray of int
        Each of its faces' face of the drawing.
    layered : np.ndarray of bool, shape ``(topology.n_faces,)``
        Which of the drawing's faces carry layers.
    """
    run = tuple(int(f) for f in run)
    opening = {g for f in (run[0], run[-1]) for g in open_ends(topology, f, layered)}
    pairs, owner = faces.edges()
    keys = {
        tuple(sorted(pair)): k
        for k, pair in enumerate(pairs.tolist())
        if int(tag[owner[k]]) not in run
    }
    ends = []
    for rim in (grid[0], grid[-1]):
        rim_edges = zip(rim, np.roll(rim, -1), strict=True)
        beside = {int(tag[owner[keys[tuple(sorted((a, b)))]]]) for a, b in rim_edges}
        end = beside.pop() if len(beside) == 1 else -1
        ends.append(end if end in opening else -1)
    return PipeEnds(run, *ends)


def fill_pipes(
    core: PipeCore, points: np.ndarray, fill: Polygons, walls: Polygons, wall_tag: np.ndarray,
    pipes: list[tuple[PipeEnds, np.ndarray]],
):  # fmt: skip
    """The pipes' cores filled with O-grids, taken out of the fill's surface and the boundary.

    Parameters
    ----------
    core : PipeCore
    points : np.ndarray, shape ``(n_points, 3)``
    fill : Polygons
        The surface the tetrahedral fill would be given, wound out of the region it fills.
    walls : Polygons
        The region's boundary faces, ``wall_tag`` (shape ``(walls.n_faces,)``) the face of the
        drawing under each.
    pipes : list of (PipeEnds, np.ndarray)
        Each run of pipes' ends, and its rings: its grid's rows of points round the core (the layers' top
        over the grid, or the grid itself without layers), shape ``(rows + 1, n)``.

    Returns
    -------
    points : np.ndarray
        ``points`` with the O-grids' own after them.
    hexahedra : np.ndarray of int, shape ``(n, 8)``
    fill, walls : Polygons
    wall_tag : np.ndarray of int
    """
    hexahedra = [np.zeros((0, 8), np.int64)]
    drop_fill = []
    walls_kept = np.ones(walls.n_faces, dtype=bool)
    new_walls, new_tags, discs = [], [], []
    for ends, rings in pipes:
        last = rings.shape[0] - 1
        first_row = 0 if ends.first >= 0 else core.junction_rows
        last_row = last if ends.last >= 0 else last - core.junction_rows
        if last_row - first_row < 1:
            continue
        cored = rings[first_row : last_row + 1]
        filled = core.fill(points, cored)
        points = filled.points
        hexahedra.append(filled.hexahedra)
        # The fill no longer meets the rings' quadrilaterals along the cored rows.
        closed = np.concatenate([cored, cored[:, :1]], axis=1)
        drop_fill.append(
            np.stack(
                [closed[:-1, :-1], closed[:-1, 1:], closed[1:, 1:], closed[1:, :-1]], axis=2
            ).reshape(-1, 4)
        )
        for end, section, inward in ((ends.first, filled.first, +1), (ends.last, filled.last, -1)):
            ring = cored[0] if inward > 0 else cored[-1]
            if end >= 0:
                # An open end: the O-grid's cross-section replaces the end face's own faces.
                walls_kept &= wall_tag != end
                new_walls.append(section)
                new_tags.append(np.full(section.shape[0], end))
            else:
                # A junction: the fill meets the cross-section, wound out of the fill, into the pipe.
                towards = points[cored[1 if inward > 0 else -2]].mean(axis=0) - points[ring].mean(
                    axis=0
                )
                discs.append(_facing(points, section, towards))
    drop = np.concatenate(drop_fill) if drop_fill else np.zeros((0, 4), np.int64)
    end_faces = walls.where(~walls_kept)
    kept_fill = ~_matches(
        fill, Polygons(end_faces.triangles, np.concatenate([drop, end_faces.quads]))
    )
    fill = fill.where(kept_fill)
    disc = np.concatenate(discs) if discs else np.zeros((0, 4), np.int64)
    fill = Polygons(fill.triangles, np.concatenate([fill.quads, disc]))
    kept_triangles, kept_quads = walls.split(walls_kept)
    tag_triangles, tag_quads = walls.split(wall_tag)
    new_quads = np.concatenate(new_walls) if new_walls else np.zeros((0, 4), np.int64)
    new_tag = np.concatenate(new_tags) if new_tags else np.zeros(0, np.int64)
    walls_out = Polygons(
        walls.triangles[kept_triangles], np.concatenate([walls.quads[kept_quads], new_quads])
    )
    tags_out = np.concatenate([tag_triangles[kept_triangles], tag_quads[kept_quads], new_tag])
    return points, np.concatenate(hexahedra), fill, walls_out, tags_out


def _facing(points, quads, direction) -> np.ndarray:
    """``quads`` each wound so its right-hand normal points along ``direction``."""
    corner = points[quads]
    normal = np.cross(corner[:, 2] - corner[:, 0], corner[:, 3] - corner[:, 1])
    flip = normal @ direction < 0.0
    quads = quads.copy()
    quads[flip] = quads[flip][:, ::-1]
    return quads


def _matches(faces: Polygons, among: Polygons) -> np.ndarray:
    """Which of ``faces`` have the same corners as one of ``among``, shape ``(faces.n_faces,)``."""

    def keys(polygons):
        return {tuple(sorted(ring[ring >= 0].tolist())) for ring in polygons.padded()}

    wanted = keys(among)
    return np.array(
        [tuple(sorted(ring[ring >= 0].tolist())) in wanted for ring in faces.padded()], dtype=bool
    )
