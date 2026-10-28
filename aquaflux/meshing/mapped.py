"""Structured quadrilaterals on chosen faces of a triangulated surface, by mapped meshing.

A wall meshed with a structured grid of quadrilaterals carries hexahedral layers, so the dual mesh's
wall cells are hexahedra rather than polygonal prisms. Here such a grid replaces the triangles of a
face of a surface triangulation, keeping every point on the face's edges: the faces beside it keep
their triangles, which meet the grid at exactly those points, so the surface stays conformal.

Two shapes of face are meshed:

- **a band** -- a face bounded by two closed loops, such as a pipe's wall (straight, bent, coiled or
  swept along any curve), a cone, or a flat ring: the grid runs between the two loops. On a tube its
  rows are the tube's cross-sections: the one rigid motion between its rims (a slide, a turn, a
  screw) where the wall is what that motion sweeps, else marched along the wall
  (:class:`~aquaflux.meshing.sections.TubeWall`); any other band's rows are that one motion's;
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
from typing import ClassVar

import numpy as np
from scipy.spatial import cKDTree

from .polygons import Polygons
from .rows import PipeGrading
from .sections import TubeWall

__all__ = ["MappedQuads", "MappedSurface", "RimMotion"]

#: The rows over which a band's rows go from a rim's own shape to a flat ring.
RIM_ROWS = 2

#: How far apart, as a share of the larger, a band's rims' radii may be for it to be a tube whose
#: rows are marched along it; further apart (a cone, a flat ring), its rows are the one motion
#: between its rims.
TUBE_RADII = 0.02

#: How far, as a share of its radius, a tube's wall may stray from the tube its rim sweeps along
#: the one rigid motion between its rims (a straight pipe's slide, a bend's turn, a coil's screw)
#: for its rows to be that motion's; further (an S-bend, a sweep along a free curve), they are
#: marched along it. The wall's triangles stand inside the true wall by their sag, so this is
#: more than that sag at any spacing a grid would use.
SWEPT_WALL = 0.05

#: How far round, as a share of its length, a rim's centroid may be carried for its path to be
#: taken as straight: a straight pipe's rims fit with a twist about its own axis, which carries
#: the centroid nowhere.
STRAIGHT_PATH = 1e-3

#: The wall points measured against the swept tube at a time, to bound the memory it takes.
_FOLLOW_CHUNK = 2048


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
class RimMotion:
    """The rigid motion that best carries one loop of points onto another, as a screw.

    A screw turns by :attr:`angle` about the line through :attr:`pivot` along :attr:`axis` while
    sliding :attr:`slide` along it; every rigid motion is one. Part of the way along it, at
    ``t`` in ``[0, 1]``, is the same turn and slide scaled by ``t``: a straight pipe's rims are
    one slide apart, a bend's one turn about the bend's axis, and a point carried along traces
    the pipe's own wall. The fit is the least-squares rotation (Kabsch's) between the loops'
    points taken in order, so it asks only that they be matched one for one.

    Attributes
    ----------
    axis : np.ndarray, shape ``(3,)``
        Unit vector along the screw's line (any unit vector when it does not turn).
    pivot : np.ndarray, shape ``(3,)``
        A point on that line.
    angle : float
        The turn, in radians, in ``[0, pi]``.
    slide : np.ndarray, shape ``(3,)``
        The translation along the line; when the motion does not turn, the whole translation.
    """

    axis: np.ndarray
    pivot: np.ndarray
    angle: float
    slide: np.ndarray

    #: Below this turn, in radians, the motion is taken to be a translation: the rims of a
    #: straight pipe fit with a turn of rounding error, about whose line a point is far away.
    STRAIGHT: ClassVar[float] = 1e-6

    @classmethod
    def fit(cls, start: np.ndarray, end: np.ndarray) -> RimMotion:
        """The rigid motion carrying ``start`` closest to ``end``, point ``i`` to point ``i``.

        Parameters
        ----------
        start, end : np.ndarray, shape ``(n, 3)``
        """
        a_centre, b_centre = start.mean(axis=0), end.mean(axis=0)
        u, _, vt = np.linalg.svd((start - a_centre).T @ (end - b_centre))
        flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
        rotation = vt.T @ flip @ u.T
        translation = b_centre - rotation @ a_centre
        angle = float(np.arccos(np.clip(0.5 * (np.trace(rotation) - 1.0), -1.0, 1.0)))
        if angle < cls.STRAIGHT:
            return cls(np.array([0.0, 0.0, 1.0]), np.zeros(3), 0.0, translation)
        skew = rotation - rotation.T
        axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        if np.linalg.norm(axis) < cls.STRAIGHT:  # a half turn: the axis from R + I
            axis = (rotation + np.eye(3))[np.argmax(np.diag(rotation))]
        axis /= np.linalg.norm(axis)
        slide = (translation @ axis) * axis
        # The point on the line nearest the origin: what the turn about it moves by the rest.
        across = translation - slide
        pivot = 0.5 * (across + np.cross(axis, across) / np.tan(0.5 * angle))
        return cls(axis, pivot, angle, slide)

    def bends(self, point: np.ndarray) -> bool:
        """Whether the path ``point`` (shape ``(3,)``) travels is curved, not a straight line: a
        turn about a line through it (a straight pipe's rims twisted against each other) leaves it
        straight; the same turn about a line away from it (a bend's) does not."""
        offset = point - self.pivot
        radial = offset - (offset @ self.axis) * self.axis
        across = self.angle * float(np.linalg.norm(radial))
        return across > STRAIGHT_PATH * float(self.path_length(point[None])[0])

    def scaled(self, factor: float) -> RimMotion:
        """The same screw, its turn and slide scaled by ``factor``."""
        return RimMotion(self.axis, self.pivot, factor * self.angle, factor * self.slide)

    def turned(self, t: float, vectors: np.ndarray) -> np.ndarray:
        """``vectors`` (shape ``(n, 3)``) turned by ``t`` of the motion's turn."""
        if self.angle == 0.0:
            return vectors
        k, theta = self.axis, t * self.angle
        return (
            vectors * np.cos(theta)
            + np.cross(k, vectors) * np.sin(theta)
            + np.outer(vectors @ k, k) * (1.0 - np.cos(theta))
        )

    def at(self, t: float, points: np.ndarray) -> np.ndarray:
        """``points`` (shape ``(n, 3)`` or ``(3,)``) carried ``t`` of the way along the motion."""
        points = np.asarray(points, dtype=np.float64)
        flat = points.reshape(-1, 3)
        moved = self.turned(t, flat - self.pivot) + self.pivot + t * self.slide
        return moved.reshape(points.shape)

    def direction(self, t: float, point: np.ndarray) -> np.ndarray:
        """The velocity, per unit of ``t``, of the point the motion has carried to ``point`` at
        ``t``; shape ``(3,)``."""
        return self.angle * np.cross(self.axis, point - self.pivot) + self.slide

    def path_length(self, points: np.ndarray) -> np.ndarray:
        """How far each point (shape ``(n, 3)``) travels along the whole motion, shape ``(n,)``:
        round a helix, which is an arc when nothing slides and a line when nothing turns."""
        offset = points - self.pivot
        radial = offset - np.outer(offset @ self.axis, self.axis)
        across = self.angle * np.linalg.norm(radial, axis=1)
        return np.hypot(across, np.linalg.norm(self.slide))


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
        graded: Mapping[int, tuple[PipeGrading, np.ndarray]] | None = None,
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
            returns them with the face's unit normal at each (either way round), each shape
            ``(m, 3)``.
        graded : mapping of {int: (PipeGrading, np.ndarray of int)}, optional
            For the faces given, how rows are graded along them where the face is a straight
            tube (its rims one slide apart), and the points of its rims that are open ends -- an
            inlet or an outlet, joining nothing -- which the rows grow toward
            (:class:`~aquaflux.meshing.PipeGrading`). Other faces' rows are evenly spread.

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
                new_triangles, new_quads = grid.band(
                    *loops, mine, _on_face(onto, int(f)), (graded or {}).get(int(f))
                )
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
            self.points[added] = onto(new, np.full(new.shape[0], face, dtype=np.int64))[0]
        self.moved_from = self.points.shape[0]

    def band(self, first: np.ndarray, second: np.ndarray, triangles: np.ndarray, onto, graded=None):
        """The grid between two closed loops: rows from the one with more points to the other.

        ``triangles`` are the face's own, and ``onto(points)`` moves points onto it, with its
        normals there. The rows follow the one rigid motion between the rims (a slide, a turn, a
        screw) -- for a band that is no tube (:func:`_tube`: a cone, a flat ring), and for a tube
        whose wall is what that motion sweeps its rim along (:func:`_follows`: a straight pipe, a
        bend, a coil). Any other tube's rows are its cross-sections marched along its triangles
        (:class:`~aquaflux.meshing.sections.TubeWall`), as many as keep the cells as long as they
        are wide, the tube's length its area over its rims' mean perimeter. A straight tube's rows
        are graded when ``graded`` -- a :class:`~aquaflux.meshing.PipeGrading` and the points of
        rims that are open ends -- is given: growing away from each rim that is not open. Either
        way the rows are then carried onto the second rim (:meth:`_rows`).
        """
        # Both loops are wound with the face on their left, so they run opposite ways round it.
        a, b = (first, second[::-1]) if first.size >= second.size else (second, first[::-1])
        p = self.points
        equal = a.size == b.size
        fractions = _fractions(p[a], closed=True)

        def across_from(start: int) -> np.ndarray:
            """The points of ``b``, from its point ``start``, matched to ``a``'s one for one."""
            rolled = np.roll(b, -start)
            return p[rolled] if equal else _resampled(p[rolled], fractions, closed=True)

        into = _into_face(p, a, triangles)
        bow_start = _bow(p[a], into)
        flat_start = p[a] - bow_start
        b_normal = _newell_normal(p[b])

        # Which point of ``b`` lies across from ``a``'s first: the start for which the rows' end
        # (``reached``) is carried onto the rim by the least turn. A circle maps onto itself
        # turned through any angle, so every start fits a round rim alike; the least turn is the
        # one whose columns run along the face rather than twisting round it.
        def nearest_start(reached: np.ndarray) -> int:
            def turn(k: int) -> float:
                end = across_from(k)
                return RimMotion.fit(reached, end - _bow(end, b_normal)).angle

            return min(range(b.size), key=turn)

        start = nearest_start(flat_start)
        across = across_from(start)
        motion = RimMotion.fit(flat_start, across - _bow(across, b_normal))
        fractions = None
        if _tube(p[a], p[b]) and not _follows(motion, flat_start, p[np.unique(triangles)]):
            # A pipe that turns other than by one screw: march its cross-sections along it.
            along = _mean_spacing(p[a])
            length = _surface_area(p, triangles) / (0.5 * (_perimeter(p[a]) + _perimeter(p[b])))
            rows = max(1, round(length / along))
            marched = TubeWall.of(p, triangles, onto).march(flat_start, into, length / rows, rows)
            start = nearest_start(marched[-1])
            across = across_from(start)
        elif graded is not None and not motion.bends(flat_start.mean(axis=0)):
            # A straight tube, its rows graded away from the rims that join something.
            grading, open_points = graded
            fine = tuple(not np.isin(loop, open_points).all() for loop in (a, b))
            length = float(np.mean(motion.path_length(flat_start)))
            fractions = grading.fractions(length, _mean_spacing(p[a]), fine)
            marched = [flat_start] * fractions.size
        else:
            marched = [flat_start] * (_row_count(p[a], across) + 1)
        if fractions is None:
            fractions = np.linspace(0.0, 1.0, len(marched))
        b = np.roll(b, -start)
        self.last_grid = None
        if equal:
            self.last_grid = self._rows(a, bow_start, marched, fractions, across, b_normal, b)
            return np.zeros((0, 3), np.int64), _grid_quads(self.last_grid, True)
        # The loops differ: the grid stops one row short of ``b``, and a strip joins that row to it.
        grid = self._rows(a, bow_start, marched, fractions, across, b_normal)
        return _zipped(grid[-1], b, self.points, closed=True), _grid_quads(grid, closed=True)

    def _rows(self, a, bow_start, marched, fractions, across, b_normal, b=None):
        """Rows of points from loop ``a`` (row 0) toward the points ``across``, row ``rows``:
        ending with ``b`` when given, else one row short of ``across``.

        Row ``j`` is ``marched[j]`` -- a tube's cross-section, its last where the march ends, or
        for any other band the first loop flattened -- carried ``t = fractions[j]`` of the way
        (``j / rows`` unless graded) along the rigid motion that
        best carries the last of them onto the second loop flattened (:class:`RimMotion`), plus
        that share of what the motion leaves over (the change of radius along a cone, across a
        flat ring), plus each loop's own bow out of its plane, fading over :data:`RIM_ROWS` rows
        from it. A loop is flattened by moving each point along the loop's normal onto the plane
        through its centroid. So away from a rim that is not flat (a pipe's saddle where it meets a
        vessel) the rows are flat rings, as a pipe's cross-sections are.
        """
        rows = len(marched) - 1
        bow_end = _bow(across, b_normal)
        flat_end = across - bow_end
        motion = RimMotion.fit(marched[-1], flat_end)
        left_over = flat_end - motion.at(1.0, marched[-1])
        flat_start = marched[0]
        grid = [a]
        for j in range(1, rows):
            t = float(fractions[j])
            row = motion.at(t, marched[j]) + t * left_over
            start_share = max(0.0, 1.0 - j / RIM_ROWS)
            end_share = max(0.0, 1.0 - (rows - j) / RIM_ROWS)
            if start_share:
                row += start_share * RimMotion.fit(flat_start, row).turned(1.0, bow_start)
            if end_share:
                row += end_share * RimMotion.fit(flat_end, row).turned(1.0, bow_end)
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
        If the boundary is not simple loops: a point where two loops touch, or an edge that does
        not lead on round a loop -- as when the triangles are not wound alike.
    """
    directed = np.stack([triangles, np.roll(triangles, -1, axis=1)], axis=2).reshape(-1, 2)
    reverse = {(int(b), int(a)) for a, b in directed}
    following: dict[int, int] = {}
    for a, b in directed:
        if (int(a), int(b)) not in reverse:
            if int(a) in following:
                raise ValueError(
                    f"the boundary passes through point {int(a)} more than once: two loops "
                    "touch there, or the triangles are not wound alike"
                )
            following[int(a)] = int(b)
    loops, left = [], set(following)
    while left:
        start = min(left)
        loop = [start]
        here = following.get(start)
        while here != start:
            if here is None or here not in left or here in loop:
                raise ValueError(
                    f"the boundary from point {start} does not close into a loop: the "
                    "triangles are not wound alike"
                )
            loop.append(here)
            here = following.get(here)
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


def _row_count(start: np.ndarray, end: np.ndarray) -> int:
    """Rows between two matched loops that keep the cells about as long as they are wide: the
    mean path their points travel along the rigid motion carrying one onto the other, over the
    first loop's spacing."""
    across = float(np.mean(RimMotion.fit(start, end).path_length(start)))
    return max(1, round(across / _mean_spacing(start)))


def _bow(loop: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Each point's offset along ``direction`` from the plane through the loop's centroid across
    it, shape ``(n, 3)``."""
    unit = direction / np.linalg.norm(direction)
    return ((loop - loop.mean(axis=0)) @ unit)[:, None] * unit


def _newell_normal(loop: np.ndarray) -> np.ndarray:
    """A closed loop's unit normal (Newell's), the way it winds round it, shape ``(3,)``."""
    offset = loop - loop.mean(axis=0)
    normal = np.cross(offset, np.roll(offset, -1, axis=0)).sum(axis=0)
    return normal / np.linalg.norm(normal)


def _into_face(points: np.ndarray, loop: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """A rim's normal, turned to point into the face it bounds: toward the far corners of the
    face's triangles standing on the rim's edges, shape ``(3,)``."""
    normal = _newell_normal(points[loop])
    on_rim = np.isin(triangles, loop)
    standing = triangles[on_rim.sum(axis=1) == 2]
    far = standing[~np.isin(standing, loop)]
    if float(np.sum((points[far] - points[loop].mean(axis=0)) @ normal)) < 0.0:
        normal = -normal
    return normal


def _radius(loop: np.ndarray) -> float:
    """A loop's mean distance from its centroid, across its own normal."""
    offset = loop - loop.mean(axis=0)
    normal = _newell_normal(loop)
    return float(np.linalg.norm(offset - np.outer(offset @ normal, normal), axis=1).mean())


def _tube(first: np.ndarray, second: np.ndarray) -> bool:
    """Whether a band's two loops are rims of one tube -- the same size within
    :data:`TUBE_RADII` -- rather than of a cone or a flat ring."""
    a, b = _radius(first), _radius(second)
    return abs(a - b) <= TUBE_RADII * max(a, b)


def _follows(motion: RimMotion, rim: np.ndarray, wall: np.ndarray) -> bool:
    """Whether a band's wall is the tube its first rim sweeps along ``motion``: every point of
    ``wall`` (shape ``(m, 3)``) within :data:`SWEPT_WALL` of the rim's radius from the path its
    centroid travels, carried on a little past either rim (a saddle-shaped rim reaches beyond the
    plane the motion starts from)."""
    radius = _radius(rim)
    length = float(motion.path_length(rim.mean(axis=0)[None])[0])
    beyond = 2.0 * radius / max(length, radius)
    samples = max(64, int(np.ceil(8.0 * length / radius)))
    t = np.linspace(-beyond, 1.0 + beyond, samples)
    path = np.stack([motion.at(s, rim.mean(axis=0)) for s in t])
    start, segment = path[:-1], np.diff(path, axis=0)
    lengths = np.einsum("kd,kd->k", segment, segment)
    for chunk in range(0, wall.shape[0], _FOLLOW_CHUNK):
        offset = wall[chunk : chunk + _FOLLOW_CHUNK, None, :] - start[None]
        along = np.clip(np.einsum("mkd,kd->mk", offset, segment) / lengths, 0.0, 1.0)
        distance = np.linalg.norm(offset - along[..., None] * segment[None], axis=2).min(axis=1)
        if np.any(np.abs(distance - radius) > SWEPT_WALL * radius):
            return False
    return True


def _perimeter(loop: np.ndarray) -> float:
    return float(np.linalg.norm(np.roll(loop, -1, axis=0) - loop, axis=1).sum())


def _mean_spacing(loop: np.ndarray) -> float:
    return _perimeter(loop) / loop.shape[0]


def _surface_area(points: np.ndarray, triangles: np.ndarray) -> float:
    corner = points[triangles]
    normal = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
    return 0.5 * float(np.linalg.norm(normal, axis=1).sum())


def _on_face(onto, face: int):
    """``onto(points, faces)`` for the points of one face, as ``onto(points)``."""

    def on_face(points: np.ndarray):
        return onto(points, np.full(points.shape[0], face, dtype=np.int64))

    return on_face


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
