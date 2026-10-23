"""Pipes in a region: found from its faces, and their cores filled with structured O-grids.

A pipe is a cylindrical or toroidal face of the region with the fluid inside it -- a straight
length of pipe, or a bend -- bounded by two loops that a structured grid joins row by row
(:class:`~aquaflux.meshing.MappedQuads`), with nothing else inside it. Its wall is meshed as that grid and, where the walls carry layers, layered like any other wall;
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

from aquaflux.io.cad import CadTopology, CylinderFace, TorusFace

from .mapped import _loops
from .pipe_core import PipeCore
from .polygons import Polygons

__all__ = ["PipeEnds", "fill_pipes", "find_pipes", "pipe_ends", "pipe_runs", "rim_divisions"]


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
        record = topology.faces[f]
        if not isinstance(record, (CylinderFace, TorusFace)):
            refused[f] = "it is neither a cylinder nor a torus"
            continue
        if not record.solid_inside:
            refused[f] = "the fluid is outside it, not inside"
            continue
        loops = _loops(triangles[face == f])
        sizes = sorted(loop.size for loop in loops)
        if len(loops) != 2:
            refused[f] = f"it is bounded by {len(loops)} loops, not the two of a band"
            continue
        if rim_counts and (sizes[0] != sizes[1] or sizes[0] % 4):
            refused[f] = (
                f"its rims have {sizes[0]} and {sizes[1]} points; an O-grid needs equal counts, "
                "a multiple of four"
            )
            continue
        if not _hollow(record, points, triangles, face, f, topology):
            refused[f] = "something else lies inside it"
            continue
        pipes.append(f)
    return pipes, refused


def rim_divisions(
    topology: CadTopology, triangles: np.ndarray, face: np.ndarray, candidates: np.ndarray
) -> dict[int, int]:
    """The segments to ask of each rim edge so that every pipe candidate's rims can be joined by
    an O-grid: both rims of one count, a multiple of four.

    A candidate is a cylinder or a torus with the fluid inside it, bounded by two loops of the
    triangulation (the rest are left to :func:`find_pipes` to refuse). Candidates sharing a rim are
    one run, given one count, so that the rims each shares are divided as both need. Where a
    run's loops already agree and are a multiple of four nothing is asked. Otherwise every rim of
    the run gets its largest count rounded up to a multiple of four, so a rim is never divided
    more coarsely than the triangulation chose; a rim drawn as several edges shares its count
    among them by length, each edge at least one segment.

    Parameters
    ----------
    topology : CadTopology
    triangles, face : np.ndarray
        The region's surface triangulation, shape ``(n_triangles, 3)``, and each triangle's face,
        shape ``(n_triangles,)``.
    candidates : np.ndarray of int
        The faces to look at.

    Returns
    -------
    dict of {int: int}
        Segments per edge, by the edge's index in ``topology``; empty when no candidate needs any.
    """
    sizes, rims = {}, {}
    for f in np.asarray(candidates, dtype=np.int64).tolist():
        record = topology.faces[f]
        if not (isinstance(record, (CylinderFace, TorusFace)) and record.solid_inside):
            continue
        loops = _loops(triangles[face == f])
        found = _rims(topology, f)
        if len(loops) != 2 or len(found) != 2:
            continue
        sizes[f], rims[f] = [loop.size for loop in loops], found
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


def _hollow(record, points, triangles, face, f, topology) -> bool:
    """Whether no point of the surface but the pipe's own and its end faces' lies inside it."""
    own = np.unique(triangles[face == f])
    caps = [g for g in _neighbours(topology, f) if _is_cap(topology, g, [f])]
    others = np.setdiff1d(np.unique(triangles[~np.isin(face, [f, *caps])]), own)
    tube = _Tube.of(record, points[own])
    return not np.any(tube.inside(points[others]))


@dataclasses.dataclass(frozen=True)
class _Tube:
    """The inside of a pipe's face, between its rims: within ``radius`` of its centreline, and
    within the range its own points span along it.

    The centreline is a straight line (a cylinder: ``bend`` is zero) or a circle of radius
    ``bend`` about ``axis`` through ``origin`` (a torus); the position along it is a height in
    the first case and an angle about the axis, from the direction ``reference``, in the second.
    """

    origin: np.ndarray
    axis: np.ndarray
    radius: float
    bend: float
    reference: np.ndarray
    span: tuple[float, float]

    @classmethod
    def of(cls, record, own: np.ndarray) -> _Tube:
        """The tube of a cylinder or torus record, spanning the points ``own`` (shape ``(n, 3)``)."""
        if isinstance(record, CylinderFace):
            origin, radius, bend = record.origin, record.radius, 0.0
        else:
            origin, radius, bend = record.centre, record.minor_radius, record.major_radius
        origin = np.asarray(origin, dtype=np.float64)
        axis = np.asarray(record.axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        mean = own.mean(axis=0) - origin
        reference = mean - (mean @ axis) * axis
        norm = np.linalg.norm(reference)
        reference = reference / norm if norm > 0.0 else reference
        tube = cls(origin, axis, float(radius), float(bend), reference, (0.0, 0.0))
        along, _ = tube._position(own)
        return dataclasses.replace(tube, span=(float(along.min()), float(along.max())))

    def _position(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each point's position along the centreline, and its distance from it."""
        offset = points - self.origin
        height = offset @ self.axis
        radial = offset - height[:, None] * self.axis
        if self.bend == 0.0:
            return height, np.linalg.norm(radial, axis=1)
        sideways = np.cross(self.axis, self.reference)
        angle = np.arctan2(radial @ sideways, radial @ self.reference)
        return angle, np.hypot(np.linalg.norm(radial, axis=1) - self.bend, height)

    def inside(self, points: np.ndarray) -> np.ndarray:
        """Which of ``points`` (shape ``(m, 3)``) lie inside, shape ``(m,)``."""
        along, across = self._position(points)
        low, high = self.span
        return (along > low) & (along < high) & (across < self.radius * (1 - 1e-6))


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
    """What each end of a run of pipes' grid meets: the flat face without layers it ends on, if
    any.

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
    layered : np.ndarray of bool
        Which of its faces carry layers.
    """
    run = tuple(int(f) for f in run)
    pairs, owner = faces.edges()
    keys = {
        tuple(sorted(pair)): k
        for k, pair in enumerate(pairs.tolist())
        if int(tag[owner[k]]) not in run
    }
    ends = []
    for rim in (grid[0], grid[-1]):
        rim_edges = zip(rim, np.roll(rim, -1), strict=True)
        across = {owner[keys[tuple(sorted((a, b)))]] for a, b in rim_edges}
        beside = {int(tag[k]) for k in across}
        end = beside.pop() if len(beside) == 1 else -1
        open_end = (
            end >= 0
            and not np.any(layered[np.asarray(sorted(across))])
            and _is_cap(topology, end, run)
        )
        ends.append(end if open_end else -1)
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
