"""Structured quadrilaterals on chosen faces of a triangulated surface, by mapped meshing.

A wall meshed with a structured grid of quadrilaterals carries hexahedral layers, so the dual mesh's
wall cells are hexahedra rather than polygonal prisms. Here such a grid replaces the triangles of a
face of a surface triangulation, keeping every point on the face's edges: the faces beside it keep
their triangles, which meet the grid at exactly those points, so the surface stays conformal.

Two shapes of face are meshed:

- **a band** -- a face bounded by two closed loops, such as a cylinder or a cone round an axis, or a
  flat ring: the grid runs between the two loops, its rows interpolated from one loop to the other;
- **a four-sided face** -- one loop with four corners: the grid is the transfinite (Coons)
  interpolation of its four sides.

The grid's points are interpolated in space and then moved onto the face. It is wound as the
triangles were by construction: its rows run from a loop, in the loop's own winding (which has the
face on its left, seen from outside), into the face. Its sides take the edge points the
triangulation gave them, so opposite sides do not always have equal counts. A band whose loops
differ, or a four-sided face one pair of whose opposite sides differ, is joined to the shorter one
through a strip of triangles along it; a four-sided face whose two pairs both differ is refused.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping

import numpy as np
from scipy.spatial import cKDTree

from .polygons import Polygons

__all__ = ["MappedQuads", "MappedSurface"]

#: The rows over which a band's rows go from a rim's own shape to a flat ring.
RIM_ROWS = 2


@dataclasses.dataclass(frozen=True)
class MappedSurface:
    """A surface of triangles and quadrilaterals, each tagged by the face it lies on.

    Attributes
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
        Every one used by a face.
    faces : Polygons
        Wound out of the solid, as the triangulation was.
    face : np.ndarray of int, shape ``(faces.n_faces,)``
        The face each lies on, the triangles' first.
    grids : mapping of {int: np.ndarray of int}
        For each band whose loops have equal counts, its points as a grid ``(rows + 1, n)``: row 0
        one loop, the last the other, each row round the band in the same direction, column ``i``
        of every row joined to column ``i`` of the next by the grid's edges. Its quadrilaterals are
        ``[grid[j, i], grid[j, i + 1], grid[j + 1, i + 1], grid[j + 1, i]]``, round it. For each
        four-sided face whose opposite sides have equal counts, its points as a grid
        ``(rows + 1, columns + 1)``, its first and last rows two opposite sides and its first and
        last columns the other two, with the same quadrilaterals but not round it.
    """

    points: np.ndarray
    faces: Polygons
    face: np.ndarray
    grids: Mapping[int, np.ndarray] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class MappedQuads:
    """Replaces the triangles of chosen faces with structured quadrilaterals.

    Attributes
    ----------
    corner_tolerance : float
        How near a point of a face's edge must lie to one of the drawing's vertices to be one of
        the face's corners, relative to the triangulation's extent.
    """

    corner_tolerance: float = 1e-9

    def remesh(
        self,
        points: np.ndarray,
        triangles: np.ndarray,
        face: np.ndarray,
        chosen: np.ndarray,
        vertices: np.ndarray,
        onto: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> MappedSurface:
        """The surface with the triangles of each chosen face replaced by a structured grid.

        Parameters
        ----------
        points : np.ndarray, shape ``(n_points, 3)``
        triangles : np.ndarray of int, shape ``(n_triangles, 3)``
            A surface wound out of the solid.
        face : np.ndarray of int, shape ``(n_triangles,)``
            The face each triangle lies on.
        chosen : np.ndarray of int
            The faces to mesh with quadrilaterals.
        vertices : np.ndarray, shape ``(n_vertices, 3)``
            The drawing's vertices, which mark a four-sided face's corners.
        onto : callable
            ``onto(points, faces)`` moves points ``(m, 3)`` onto the faces ``(m,)`` given and
            returns them, shape ``(m, 3)``.

        Returns
        -------
        MappedSurface
            The points the surface still uses, the triangulation's first in their order, then the
            grids' new ones.

        Raises
        ------
        ValueError
            If a chosen face is neither a band nor four-sided, or is four-sided with both pairs of
            opposite sides of different counts.
        """
        points = np.asarray(points, dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int64)
        face = np.asarray(face, dtype=np.int64)
        extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        corners = cKDTree(np.asarray(vertices, dtype=np.float64).reshape(-1, 3))
        grid = _Builder(points)
        kept_triangles = [triangles[~np.isin(face, chosen)]]
        kept_face = [face[~np.isin(face, chosen)]]
        quads, quad_face = [], []
        grids: dict[int, np.ndarray] = {}
        for f in np.unique(np.asarray(chosen, dtype=np.int64)):
            mine = triangles[face == f]
            loops = _loops(mine)
            if len(loops) == 2:
                new_triangles, new_quads = grid.band(*loops)
            elif len(loops) == 1:
                loop = loops[0]
                distance, _ = corners.query(points[loop])
                at = np.flatnonzero(distance <= self.corner_tolerance * extent)
                if at.size != 4:
                    raise ValueError(
                        f"face {f} is bounded by one loop with {at.size} corner(s), not four, so is "
                        "neither a band nor four-sided"
                    )
                new_triangles, new_quads = grid.four_sided(loop, at, f)
            else:
                raise ValueError(
                    f"face {f} is bounded by {len(loops)} loops, so is neither a band nor four-sided"
                )
            if grid.last_grid is not None:
                grids[int(f)] = grid.last_grid
            grid.onto(onto, f)
            kept_triangles.append(new_triangles)
            kept_face.append(np.full(new_triangles.shape[0], f))
            quads.append(new_quads)
            quad_face.append(np.full(new_quads.shape[0], f))
        faces = Polygons(
            np.concatenate(kept_triangles), np.concatenate(quads or [np.zeros((0, 4))])
        )
        tags = np.concatenate([*kept_face, *(quad_face or [np.zeros(0, np.int64)])])
        return _compacted(grid.points, faces, tags, grids)


class _Builder:
    """The surface's points with each grid's new points appended, and the grids' faces."""

    def __init__(self, points: np.ndarray) -> None:
        self.points = points
        self.moved_from = points.shape[0]
        self.last_grid: np.ndarray | None = None  # the last face's grid, when it is one

    def _added(self, new: np.ndarray) -> np.ndarray:
        """Append points (still to be moved onto their face); their indices."""
        start = self.points.shape[0]
        self.points = np.vstack([self.points, new.reshape(-1, 3)])
        return start + np.arange(new.reshape(-1, 3).shape[0]).reshape(new.shape[:-1])

    def onto(self, onto, face: int) -> None:
        """Move the points added since the last call onto ``face``."""
        added = slice(self.moved_from, self.points.shape[0])
        new = self.points[added]
        if new.shape[0]:
            self.points = self.points.copy()
            self.points[added] = onto(new, np.full(new.shape[0], face, dtype=np.int64))
        self.moved_from = self.points.shape[0]

    def band(self, first: np.ndarray, second: np.ndarray):
        """The grid between two closed loops: rows from the one with more points to the other."""
        # Both loops are wound with the face on their left, so they run opposite ways round it.
        a, b = (first, second[::-1]) if first.size >= second.size else (second, first[::-1])
        p = self.points
        b = np.roll(b, -int(np.argmin(np.linalg.norm(p[b] - p[a[0]], axis=1))))
        self.last_grid = None
        if a.size == b.size:
            shifts = [np.roll(b, -s) for s in (-1, 0, 1)]
            b = min(shifts, key=lambda c: float(np.sum(np.linalg.norm(p[c] - p[a], axis=1) ** 2)))
            rows = _row_count(p[a], p[b], closed=True)
            self.last_grid = self._between(a, p[b], rows, b)
            return np.zeros((0, 3), np.int64), _grid_quads(self.last_grid, True)
        across = _resampled(p[b], _fractions(p[a], closed=True), closed=True)
        rows = _row_count(p[a], across, closed=True)
        # The loops differ: the grid stops one row short of ``b``, and a strip joins that row to it.
        grid = self._between(a, across, rows)
        return _zipped(grid[-1], b, self.points, closed=True), _grid_quads(grid, closed=True)

    def _between(self, a, across, rows, b=None):
        """Rows of points from loop ``a`` (row 0) toward the points ``across``, row ``rows``:
        ending with ``b`` when given, else one row short of ``across``.

        Each row is the linear interpolation of the two loops flattened -- each moved along the
        band's axis onto the plane through its centroid across it -- plus each loop's own bow out
        of that plane, fading over :data:`RIM_ROWS` rows from it. So away from a rim that is not
        flat (a pipe's saddle where it meets a vessel) the rows are flat rings, as a pipe's
        cross-sections are; a band between flat rims is the plain interpolation.
        """
        start = self.points[a]
        axis = across.mean(axis=0) - start.mean(axis=0)
        axis /= np.linalg.norm(axis)
        bow_start = ((start - start.mean(axis=0)) @ axis)[:, None] * axis
        bow_end = ((across - across.mean(axis=0)) @ axis)[:, None] * axis
        grid = [a]
        for j in range(1, rows):
            t = j / rows
            row = (1 - t) * (start - bow_start) + t * (across - bow_end)
            row += max(0.0, 1.0 - j / RIM_ROWS) * bow_start
            row += max(0.0, 1.0 - (rows - j) / RIM_ROWS) * bow_end
            grid.append(self._added(row))
        if b is not None:
            grid.append(b)
        return np.stack(grid)

    def four_sided(self, loop: np.ndarray, at: np.ndarray, face: int):
        """The Coons grid of a four-sided face, its corners at positions ``at`` of ``loop``."""
        loop = np.roll(loop, -int(at[0]))
        at = np.sort((at - at[0]) % loop.size)
        closed = np.append(loop, loop[0])
        sides = [closed[at[k] : (at[k + 1] if k < 3 else loop.size) + 1] for k in range(4)]
        counts = [side.size - 1 for side in sides]
        if counts[0] != counts[2] and counts[1] != counts[3]:
            raise ValueError(
                f"face {face} has sides of {counts} segments: neither pair of opposite sides is "
                "equal, which a structured grid with one strip of triangles cannot join"
            )
        if counts[0] == counts[2] and counts[1] != counts[3]:
            sides = sides[1:] + sides[:1]  # put the unequal pair as the first and third
            counts = counts[1:] + counts[:1]
        bottom, right, top, left = sides[0], sides[1], sides[2][::-1], sides[3][::-1]
        p = self.points
        self.last_grid = None
        if counts[0] == counts[2]:
            self.last_grid = self._coons(bottom, right, left, p[top], counts[1], top)
            return np.zeros((0, 3), np.int64), _grid_quads(self.last_grid, closed=False)
        # The top differs: the grid stops one row short of it, and a strip joins that row to it.
        resampled = _resampled(p[top], _fractions(p[bottom], closed=False), closed=False)
        grid = self._coons(bottom, right, left, resampled, counts[1])
        return _zipped(grid[-1], top, self.points, closed=False), _grid_quads(grid, closed=False)

    def _coons(self, bottom, right, left, top_points, rows, top=None):
        """The transfinite interpolation of four sides into a grid of point indices: bottom and top
        running the same way, left and right from bottom to top, ``rows`` rows of cells. The last
        row is ``top`` when given, else the grid stops one row short of ``top_points``."""
        p = self.points
        bottom_p, left_p, right_p = p[bottom], p[left], p[right]
        u = 0.5 * (_fractions(bottom_p, False) + _fractions(top_points, False))
        v = 0.5 * (_fractions(left_p, False) + _fractions(right_p, False))
        uu, vv = u[None, :, None], v[:, None, None]
        blended = (
            (1 - vv) * bottom_p[None] + vv * top_points[None]
            + (1 - uu) * left_p[:, None] + uu * right_p[:, None]
            - (1 - uu) * (1 - vv) * bottom_p[0] - uu * (1 - vv) * bottom_p[-1]
            - (1 - uu) * vv * top_points[0] - uu * vv * top_points[-1]
        )  # fmt: skip
        n_rows = rows + 1 if top is not None else rows
        grid = np.empty((n_rows, bottom.size), dtype=np.int64)
        grid[0], grid[:, 0], grid[:, -1] = bottom, left[:n_rows], right[:n_rows]
        grid[1:rows, 1:-1] = self._added(blended[1:rows, 1:-1])
        if top is not None:
            grid[-1] = top
        return grid


def _loops(triangles: np.ndarray) -> list[np.ndarray]:
    """The closed loops of edges each used once by ``triangles``, each in the triangles' winding.

    Raises
    ------
    ValueError
        If the triangles are not wound alike, so that their boundary edges do not chain into loops.
    """
    directed = np.stack([triangles, np.roll(triangles, -1, axis=1)], axis=2).reshape(-1, 2)
    reverse = {(int(b), int(a)) for a, b in directed}
    following: dict[int, int] = {}
    for a, b in directed:
        if (int(a), int(b)) not in reverse:
            following[int(a)] = int(b)
    loops, left = [], set(following)
    while left:
        start = min(left)
        loop = [start]
        here = following[start]
        while here != start:
            loop.append(here)
            if len(loop) > len(following) or here not in following:
                raise ValueError(
                    "the triangles' boundary edges do not chain into loops: they are not wound alike"
                )
            here = following[here]
        left -= set(loop)
        loops.append(np.array(loop, dtype=np.int64))
    return loops


def _fractions(polyline: np.ndarray, closed: bool) -> np.ndarray:
    """Each point's arc-length fraction along a polyline (round it, when closed, from its first)."""
    path = np.vstack([polyline, polyline[:1]]) if closed else polyline
    length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    fractions = length / length[-1]
    return fractions[:-1] if closed else fractions


def _resampled(polyline: np.ndarray, fractions: np.ndarray, closed: bool) -> np.ndarray:
    """The points at given arc-length fractions along a polyline."""
    path = np.vstack([polyline, polyline[:1]]) if closed else polyline
    length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    at = fractions * length[-1]
    return np.stack([np.interp(at, length, path[:, k]) for k in range(3)], axis=1)


def _row_count(start: np.ndarray, end: np.ndarray, closed: bool) -> int:
    """Rows between two matched polylines that keep the cells about as long as they are wide."""
    across = float(np.mean(np.linalg.norm(end - start, axis=1)))
    path = np.vstack([start, start[:1]]) if closed else start
    along = float(np.mean(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    return max(1, round(across / along))


def _grid_quads(grid: np.ndarray, closed: bool) -> np.ndarray:
    """The quadrilaterals of a grid of point indices ``(rows, columns)``, round it when closed."""
    if closed:
        grid = np.concatenate([grid, grid[:, :1]], axis=1)
    lower, upper = grid[:-1], grid[1:]
    return np.stack([lower[:, :-1], lower[:, 1:], upper[:, 1:], upper[:, :-1]], axis=2).reshape(
        -1, 4
    )


def _zipped(first: np.ndarray, second: np.ndarray, points: np.ndarray, closed: bool) -> np.ndarray:
    """Triangles joining two polylines that run the same way, each step taking the shorter diagonal.

    Closed polylines are joined round from their first points, which should lie across from each
    other.
    """
    if closed:
        first, second = np.append(first, first[0]), np.append(second, second[0])
    triangles = []
    i = j = 0
    while i < first.size - 1 or j < second.size - 1:
        along_first = j == second.size - 1 or (
            i < first.size - 1
            and np.linalg.norm(points[first[i + 1]] - points[second[j]])
            <= np.linalg.norm(points[second[j + 1]] - points[first[i]])
        )
        if along_first:
            triangles.append([first[i], first[i + 1], second[j]])
            i += 1
        else:
            triangles.append([first[i], second[j + 1], second[j]])
            j += 1
    return np.array(triangles, dtype=np.int64).reshape(-1, 3)


def _compacted(points, faces: Polygons, face, grids) -> MappedSurface:
    """The surface with the points no face uses dropped, the rest kept in order."""
    used = np.zeros(points.shape[0], dtype=bool)
    for kind in faces.kinds():
        used[kind.reshape(-1)] = True
    position = np.cumsum(used) - 1
    renumbered = {f: position[g] for f, g in grids.items()}
    return MappedSurface(points[used], faces.renumbered(position), face, renumbered)
