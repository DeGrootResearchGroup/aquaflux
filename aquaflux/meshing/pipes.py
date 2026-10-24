"""Pipes in a region: found from its faces, their rims divided for a structured O-grid.

A pipe is a cylindrical face of the region with the fluid inside it, bounded by two loops that a
structured grid joins row by row (:class:`~aquaflux.meshing.MappedQuads`), with nothing else inside
it. Its wall is meshed as that grid and, where the walls carry layers, layered like any other wall;
the core the layers leave is filled with a :class:`~aquaflux.meshing.PipeCore` O-grid, which meets the
layers' top quadrilateral for quadrilateral, and its ends are open or junctions as any conduit's
(:mod:`~aquaflux.meshing.conduits`).

The O-grid needs both loops of one point count, a multiple of four. The loops' points are the surface
triangulation's, whose division of the rims follows the mesh size, so :func:`rim_divisions` says how to
divide each candidate's rim edges for the triangulation to give an O-grid's counts.
"""

from __future__ import annotations

import math

import numpy as np

from aquaflux.io.cad import CadTopology, CylinderFace

from .conduits import is_cap, neighbours
from .mapped import _loops

__all__ = ["find_pipes", "rim_divisions"]


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
    caps = [g for g in neighbours(topology, [f]) if is_cap(topology, g, [f])]
    others = np.setdiff1d(np.unique(triangles[~np.isin(face, [f, *caps])]), own)
    offset = points[others] - origin
    along = offset @ axis
    across = np.linalg.norm(offset - along[:, None] * axis, axis=1)
    inside = (along > height.min()) & (along < height.max()) & (across < record.radius * (1 - 1e-6))
    return not np.any(inside)
