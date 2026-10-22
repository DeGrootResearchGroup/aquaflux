"""Pipes in a region: found from its faces, and their cores filled with structured O-grids.

A pipe is a cylindrical face of the region with the fluid inside it, bounded by two loops that a
structured grid joins row by row (:class:`~aquaflux.meshing.MappedQuads`), with nothing else inside
it. Its wall is meshed as that grid and, where the walls carry layers, layered like any other wall;
the core the layers leave is filled with a :class:`~aquaflux.meshing.PipeCore` O-grid, which meets the
layers' top quadrilateral for quadrilateral.

The O-grid needs both loops of one point count, a multiple of four. The loops' points are the surface
triangulation's, whose division of the rims follows the mesh size, so :func:`rim_divisions` says how to
divide each candidate's rim edges for the triangulation to give an O-grid's counts.

At each end of a pipe:

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

from aquaflux.io.cad import CadTopology, CylinderFace

from .mapped import _loops
from .pipe_core import PipeCore
from .polygons import Polygons

__all__ = ["PipeEnds", "fill_pipes", "find_pipes", "pipe_ends", "rim_divisions"]


@dataclasses.dataclass(frozen=True)
class PipeEnds:
    """A pipe's face and what each end of its grid meets.

    Attributes
    ----------
    face : int
        The pipe's face.
    first, last : int
        For the grid's first and last row: the flat face it ends on (an open end), or ``-1`` where
        the pipe opens into the rest of the region (a junction).
    """

    face: int
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
        if not isinstance(record, CylinderFace):
            refused[f] = "it is not a cylinder"
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

    A candidate is a cylinder with the fluid inside it, bounded by two loops of the triangulation
    (the rest are left to :func:`find_pipes` to refuse). Where its loops' counts already agree and
    are a multiple of four nothing is asked. Otherwise both rims get the larger count rounded up to
    a multiple of four, so a rim is never divided more coarsely than the triangulation chose; a rim
    drawn as several edges shares its count among them by length, each edge at least one segment.
    An edge on the rims of two candidates takes the larger count asked of it.

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
    divisions: dict[int, int] = {}
    for f in np.asarray(candidates, dtype=np.int64).tolist():
        record = topology.faces[f]
        if not (isinstance(record, CylinderFace) and record.solid_inside):
            continue
        loops = _loops(triangles[face == f])
        if len(loops) != 2:
            continue
        sizes = [loop.size for loop in loops]
        if sizes[0] == sizes[1] and sizes[0] % 4 == 0:
            continue
        rims = _rims(topology, f)
        if len(rims) != 2:
            continue
        count = 4 * math.ceil(max(sizes) / 4)
        for rim in rims:
            for edge, segments in _shared_by_length(topology.edge_length[rim], count, rim):
                divisions[edge] = max(divisions.get(edge, 0), segments)
    return divisions


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


def _hollow(record: CylinderFace, points, triangles, face, f, topology) -> bool:
    """Whether no point of the surface but the pipe's own and its end faces' lies inside it."""
    axis = np.asarray(record.axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    origin = np.asarray(record.origin, dtype=np.float64)
    own = np.unique(triangles[face == f])
    height = (points[own] - origin) @ axis
    caps = [g for g in _neighbours(topology, f) if _is_cap(topology, g, f)]
    others = np.setdiff1d(np.unique(triangles[~np.isin(face, [f, *caps])]), own)
    offset = points[others] - origin
    along = offset @ axis
    across = np.linalg.norm(offset - along[:, None] * axis, axis=1)
    inside = (along > height.min()) & (along < height.max()) & (across < record.radius * (1 - 1e-6))
    return not np.any(inside)


def _neighbours(topology: CadTopology, f: int) -> list[int]:
    """The faces sharing an edge with face ``f``."""
    found = set()
    for edge in topology.edges_of_face(f):
        found.update(topology.faces_of_edge(edge).tolist())
    found.discard(f)
    return sorted(found)


def _is_cap(topology: CadTopology, end: int, pipe: int) -> bool:
    """Whether face ``end`` is bounded by nothing but face ``pipe``'s edges: a pipe's end."""
    return set(topology.edges_of_face(end).tolist()) <= set(topology.edges_of_face(pipe).tolist())


def pipe_ends(topology, grid: np.ndarray, pipe: int, faces: Polygons, tag, layered) -> PipeEnds:
    """What each end of a pipe's grid meets: the flat face without layers it ends on, if any.

    Parameters
    ----------
    topology : CadTopology
    grid : np.ndarray of int, shape ``(rows + 1, n)``
        The pipe's grid, in the surface's numbering.
    pipe : int
    faces : Polygons
        The surface, in the same numbering.
    tag : np.ndarray of int
        Each of its faces' face of the drawing.
    layered : np.ndarray of bool
        Which of its faces carry layers.
    """
    pairs, owner = faces.edges()
    keys = {
        tuple(sorted(pair)): k for k, pair in enumerate(pairs.tolist()) if tag[owner[k]] != pipe
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
            and _is_cap(topology, end, pipe)
        )
        ends.append(end if open_end else -1)
    return PipeEnds(pipe, *ends)


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
        Each pipe's ends, and its rings: its grid's rows of points round the core (the layers' top
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
