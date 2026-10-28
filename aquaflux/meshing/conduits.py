"""Conduits -- pipes and ducts -- whose cores are filled with structured hexahedra, taken out of the fill.

A conduit is a part of the region whose wall is meshed as a structured grid of quadrilaterals, one
row of the grid round it after another along it: a pipe's cylinder (:mod:`~aquaflux.meshing.pipes`)
or a duct's four sides (:mod:`~aquaflux.meshing.ducts`). The core the wall's layers leave is a stack
of rings of points round it, one per row, which a structured core (a pipe's O-grid, a duct's
cross-sections) fills with hexahedra meeting the rings' own quadrilaterals. The rest of the region
is left to the tetrahedral fill.

At each end of a conduit:

- **an open end** -- the conduit ends on a flat face without layers, such as an inlet, bounded only
  by the conduit's own rim -- the core runs to the end, and its last cross-section takes the place
  of that face's own faces inside the layers' ring, in the same patch;
- **a junction** -- the conduit opens into the rest of the region -- the core stops
  ``junction_rows`` rows short, where the layers' columns still run straight, and the rows beyond
  are left to the tetrahedral fill: the core's last cross-section is the interface the fill meets.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable

import numpy as np

from aquaflux.io.cad import CadTopology

from .polygons import Polygons

__all__ = ["Conduit", "ConduitEnds", "ConduitFill", "conduit_ends", "fill_conduits"]


@dataclasses.dataclass(frozen=True)
class ConduitFill:
    """The hexahedra filling a conduit's core, and its two end cross-sections.

    Attributes
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
        The points given, then the core's own.
    hexahedra : np.ndarray of int, shape ``(n_hexahedra, 8)``
        Each hexahedron's quadrilateral in one cross-section, then the same one in the next.
    first, last : np.ndarray of int, shape ``(n_quads, 4)``
        The quadrilaterals of the first and the last cross-section.
    """

    points: np.ndarray
    hexahedra: np.ndarray
    first: np.ndarray
    last: np.ndarray


@dataclasses.dataclass(frozen=True)
class ConduitEnds:
    """What each end of a conduit's grid meets.

    Attributes
    ----------
    first, last : int
        For the grid's first and last row: the flat face it ends on (an open end), or ``-1`` where
        the conduit opens into the rest of the region (a junction).
    """

    first: int
    last: int


@dataclasses.dataclass(frozen=True)
class Conduit:
    """A conduit to be filled: its ends, the rings round its core, and how to fill them.

    Attributes
    ----------
    ends : ConduitEnds
    rings : np.ndarray of int, shape ``(rows + 1, n)``
        Its grid's rows of points round the core -- the layers' top over the grid, or the grid
        itself without layers -- each in the same direction, point ``i`` of one joined to point
        ``i`` of the next.
    fill : callable
        ``fill(points, rings)`` fills consecutive rings with hexahedra, returning a
        :class:`ConduitFill`.
    junction_rows : int
        How many rows short of a junction the core stops.
    """

    ends: ConduitEnds
    rings: np.ndarray
    fill: Callable[[np.ndarray, np.ndarray], ConduitFill]
    junction_rows: int


def neighbours(topology: CadTopology, faces: Iterable[int]) -> list[int]:
    """The faces sharing an edge with any of ``faces``, other than those."""
    faces = set(faces)
    found = set()
    for f in faces:
        for edge in topology.edges_of_face(f):
            found.update(topology.faces_of_edge(edge).tolist())
    return sorted(found - faces)


def is_cap(topology: CadTopology, end: int, own: Iterable[int]) -> bool:
    """Whether face ``end`` is bounded by nothing but the edges of the faces ``own``: a conduit's
    end."""
    edges = set()
    for f in own:
        edges.update(topology.edges_of_face(f).tolist())
    return set(topology.edges_of_face(end).tolist()) <= edges


def conduit_ends(
    topology: CadTopology, rims: tuple[np.ndarray, np.ndarray], own: Iterable[int],
    faces: Polygons, tag: np.ndarray, layered: np.ndarray,
) -> ConduitEnds:  # fmt: skip
    """What each end of a conduit's grid meets: the flat face without layers it ends on, if any.

    Parameters
    ----------
    topology : CadTopology
    rims : tuple of np.ndarray of int
        The grid's first and last rows, closed loops in the surface's numbering.
    own : iterable of int
        The conduit's own faces of the drawing.
    faces : Polygons
        The surface, in the same numbering.
    tag : np.ndarray of int
        Each of its faces' face of the drawing.
    layered : np.ndarray of bool
        Which of its faces carry layers.
    """
    own = sorted(set(own))
    pairs, owner = faces.edges()
    outside = ~np.isin(tag[owner], own)
    keys = {tuple(sorted(pair)): k for k, pair in enumerate(pairs.tolist()) if outside[k]}
    ends = []
    for rim in rims:
        rim_edges = zip(rim, np.roll(rim, -1), strict=True)
        across = {owner[keys[tuple(sorted((a, b)))]] for a, b in rim_edges}
        beside = {int(tag[k]) for k in across}
        end = beside.pop() if len(beside) == 1 else -1
        open_end = (
            end >= 0
            and not np.any(layered[np.asarray(sorted(across))])
            and is_cap(topology, end, own)
        )
        ends.append(end if open_end else -1)
    return ConduitEnds(*ends)


def fill_conduits(
    points: np.ndarray, fill: Polygons, walls: Polygons, wall_tag: np.ndarray,
    conduits: list[Conduit],
):  # fmt: skip
    """The conduits' cores filled with hexahedra, taken out of the fill's surface and the boundary.

    Parameters
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
    fill : Polygons
        The surface the tetrahedral fill would be given, wound out of the region it fills.
    walls : Polygons
        The region's boundary faces, ``wall_tag`` (shape ``(walls.n_faces,)``) the face of the
        drawing under each.
    conduits : list of Conduit

    Returns
    -------
    points : np.ndarray
        ``points`` with the cores' own after them.
    hexahedra : np.ndarray of int, shape ``(n, 8)``
    fill, walls : Polygons
    wall_tag : np.ndarray of int
    """
    hexahedra = [np.zeros((0, 8), np.int64)]
    drop_fill = []
    walls_kept = np.ones(walls.n_faces, dtype=bool)
    new_walls, new_tags, discs = [], [], []
    for conduit in conduits:
        ends, rings = conduit.ends, conduit.rings
        last = rings.shape[0] - 1
        first_row = 0 if ends.first >= 0 else conduit.junction_rows
        last_row = last if ends.last >= 0 else last - conduit.junction_rows
        if last_row - first_row < 1:
            continue
        cored = rings[first_row : last_row + 1]
        filled = conduit.fill(points, cored)
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
                # An open end: the core's cross-section replaces the end face's own faces.
                walls_kept &= wall_tag != end
                new_walls.append(section)
                new_tags.append(np.full(section.shape[0], end))
            else:
                # A junction: the fill meets the cross-section, wound out of the fill, into the core.
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
