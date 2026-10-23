"""Cross-sections of a tube's wall, marched along it from one rim.

A pipe's wall -- straight, bent, coiled, or swept along any curve -- is a tube, and the rows of a
structured grid on it should be its cross-sections, so that the O-grid filling its core is built
from flat rings square to the flow. A cross-section of a tube of round section is the cut of the
wall by a plane square to the tube's centreline, and it has a property that needs no centreline to
test: **the wall's normals along it all lie in its plane**. :class:`TubeWall` finds such planes and
the points where they cut the wall, one after another from a rim, a fixed step apart, each plane's
normal the direction least in line with the wall's normals round it.

A cut is found on a triangulation of the wall (a ray from the plane's centre to the triangles it
crosses first) and then moved onto the true wall, which also gives the wall's normal there: a
triangulation's own normals, even averaged at its vertices, lean toward the larger facets round a
vertex, and so mislead the plane wherever the wall's curvature changes along it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
from scipy.spatial import cKDTree

from .regions import crossing_distances

__all__ = ["TubeWall"]

#: How small a change of a cross-section's normal, in radians, ends the search for its plane: far
#: below any tilt a grid would notice, and above the jitter a projection onto the wall leaves
#: (a refinement takes about a tenth of the last one's change, so three or four are made).
SETTLED = 1e-6

#: The most refinements of one cross-section's plane before the search stops where it is.
MAX_REFINEMENTS = 20


@dataclasses.dataclass(frozen=True)
class TubeWall:
    """A tube's wall, as triangles and the surface they stand for, cut by the planes of its
    cross-sections.

    Attributes
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
    triangles : np.ndarray of int, shape ``(n_triangles, 3)``
        The wall's triangulation.
    onto : callable
        ``onto(points)`` moves points ``(m, 3)`` onto the wall and returns them with the wall's
        unit normal at each (either way round), each shape ``(m, 3)``.
    tree : scipy.spatial.cKDTree
        The triangles' centroids, for the triangles near a cut.
    """

    points: np.ndarray
    triangles: np.ndarray
    onto: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
    tree: cKDTree

    @classmethod
    def of(cls, points: np.ndarray, triangles: np.ndarray, onto) -> TubeWall:
        """The wall the ``triangles`` (shape ``(m, 3)``) over ``points`` (shape ``(n, 3)``) are
        drawn of, ``onto`` moving points onto it (see :attr:`onto`)."""
        points = np.asarray(points, dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int64)
        return cls(points, triangles, onto, cKDTree(points[triangles].mean(axis=1)))

    def march(
        self, start: np.ndarray, into: np.ndarray, step: float, sections: int
    ) -> list[np.ndarray]:
        """Cross-sections from a flat rim, ``step`` apart along the tube.

        Each is found from the last carried a step on -- along the last normal, turned on by the
        last step's turn -- by refining its plane (:meth:`section`). The columns are carried
        without twisting: each step turns the last section's points by the least rotation
        taking its normal to the next one's.

        Parameters
        ----------
        start : np.ndarray, shape ``(n, 3)``
            The first rim, flat.
        into : np.ndarray, shape ``(3,)``
            Its unit normal, pointing along the tube.
        step : float
            The distance between sections' centres.
        sections : int
            How many to march, the first rim not counted.

        Returns
        -------
        list of np.ndarray, each shape ``(n, 3)``
            The rim, then each section in turn; the last is where the march reaches past the
            wall's end, carried on without a cut (it is only told how far the march fell short of
            the far rim).
        """
        found = [np.asarray(start, dtype=np.float64)]
        centre = found[0].mean(axis=0)
        normal = np.asarray(into, dtype=np.float64)
        offsets = found[0] - centre
        turn = np.eye(3)
        for j in range(1, sections + 1):
            ahead = turn @ normal
            middle = normal + ahead
            centre_ahead = centre + step * middle / np.linalg.norm(middle)
            offsets_ahead = offsets @ _least_rotation(normal, ahead).T
            if j == sections:
                found.append(centre_ahead + offsets_ahead)
                break
            section, centre_ahead, cut_normal = self.section(centre_ahead, ahead, offsets_ahead)
            turn = _least_rotation(normal, cut_normal)
            centre, normal, offsets = centre_ahead, cut_normal, section - centre_ahead
            found.append(section)
        return found

    def section(self, centre: np.ndarray, normal: np.ndarray, offsets: np.ndarray):
        """The cross-section nearest a guess: a plane through ``centre`` turned until the wall's
        normals where it cuts the wall lie in it, re-centred on the cut each time.

        Parameters
        ----------
        centre, normal : np.ndarray, shape ``(3,)``
            The guessed plane: a point on it and its unit normal.
        offsets : np.ndarray, shape ``(n, 3)``
            From ``centre`` toward each point wanted round the section: the cut is looked for
            along each, turned into the plane.

        Returns
        -------
        points : np.ndarray, shape ``(n, 3)``
            Where the plane cuts the wall along each direction (the guess, ``centre + offset``,
            where it does not).
        centre, normal : np.ndarray, shape ``(3,)``
            The section's plane.
        """
        centre, normal = np.asarray(centre, dtype=np.float64), np.asarray(normal, dtype=np.float64)
        offsets = np.asarray(offsets, dtype=np.float64)
        reach = 2.0 * float(np.linalg.norm(offsets, axis=1).max())
        for _ in range(MAX_REFINEMENTS):
            points, normals, cut = self._cut(centre, normal, offsets, reach)
            if np.count_nonzero(cut) < 3:
                break
            moment = normals[cut].T @ normals[cut]
            turned = np.linalg.eigh(moment)[1][:, 0]
            turned = turned if turned @ normal >= 0.0 else -turned
            offsets = offsets @ _least_rotation(normal, turned).T
            settled = float(np.arccos(np.clip(turned @ normal, -1.0, 1.0))) < SETTLED
            normal = turned
            away = points[cut].mean(axis=0) - centre
            centre = centre + away - (away @ normal) * normal
            if settled:
                break
        points, _, cut = self._cut(centre, normal, offsets, reach)
        guess = centre + offsets - np.outer(offsets @ normal, normal)
        return np.where(cut[:, None], points, guess), centre, normal

    def _cut(self, centre, normal, offsets, reach):
        """Where rays from ``centre``, along ``offsets`` turned into the plane, meet the wall,
        moved onto it; the wall's normal there; and which rays meet it."""
        directions = offsets - np.outer(offsets @ normal, normal)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        near = np.asarray(self.tree.query_ball_point(centre, reach), dtype=np.int64)
        corners = self.points[self.triangles[near]]
        distance = crossing_distances(centre, directions, corners).min(axis=1, initial=np.inf)
        cut = np.isfinite(distance)
        points = centre + np.where(cut, distance, 0.0)[:, None] * directions
        normals = np.zeros_like(directions)
        if cut.any():
            points[cut], normals[cut] = self.onto(points[cut])
        return points, normals, cut


def _least_rotation(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """The rotation matrix turning unit vector ``start`` onto unit vector ``end`` about the axis
    square to both (the identity when they agree), shape ``(3, 3)``."""
    axis = np.cross(start, end)
    sine, cosine = float(np.linalg.norm(axis)), float(start @ end)
    if sine < 1e-15:
        return np.eye(3)
    k = axis / sine
    cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + sine * cross + (1.0 - cosine) * cross @ cross
