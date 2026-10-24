"""A region of space as an octree asks about it: which points lie inside, and which boxes its
boundary may pass through.

:class:`SampledRegion` answers from an inside test alone -- an analytic body, a
:class:`~aquaflux.solids.Body`'s ``contains`` -- so whether a boundary passes through a box is
judged on a lattice of sample points over it, and a boundary passing between samples is missed.
:class:`ClosedSurface` is the region a closed triangle surface bounds, and answers exactly: a box
the surface meets is always said to be met (a box is tested against the voxels of a
:class:`~aquaflux.triangle_grid.TriangleGrid` the triangles are registered in, so the answer errs
only towards *met*, by at most a voxel), and inside is decided by ray parity, once for each
connected stretch of voxels the surface does not reach.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Protocol

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from aquaflux.triangle_grid import MAX_VOXELS, TriangleGrid

from .near_surface import cube_samples
from .netgen_process import check_closed_surface

__all__ = ["ClosedSurface", "Region", "SampledRegion"]

#: The directions a parity ray is cast in: three far from any axis, from one another and from
#: each other's opposites, so a ray from a voxel's centre through a mesh built on axis-aligned
#: lattices does not run along a face or through an edge. Each gives the parity on its own; they
#: must agree.
_RAY_DIRECTIONS = np.array(
    [[0.5773, 0.6143, 0.5380], [-0.6411, 0.2903, -0.7104], [0.2197, -0.8641, -0.4527]]
)
_RAY_DIRECTIONS = _RAY_DIRECTIONS / np.linalg.norm(_RAY_DIRECTIONS, axis=1, keepdims=True)


class Region(Protocol):
    """A region of space: which points lie in it, and where its boundary may pass."""

    def inside(self, points: np.ndarray) -> np.ndarray:
        """Whether each point lies in the region.

        Parameters
        ----------
        points : np.ndarray, shape ``(m, 3)``

        Returns
        -------
        np.ndarray of bool, shape ``(m,)``
        """

    def meets(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Whether the region's boundary is taken to pass through each axis-aligned box.

        Parameters
        ----------
        lower, upper : np.ndarray, shape ``(m, 3)``
            Opposite corners of each box.

        Returns
        -------
        np.ndarray of bool, shape ``(m,)``
        """


@dataclasses.dataclass(frozen=True)
class SampledRegion:
    """The region an inside test holds, its boundary looked for on a lattice of sample points.

    Attributes
    ----------
    contains : callable
        ``contains(points)`` for ``points`` of shape ``(m, 3)`` returns a boolean array of shape
        ``(m,)``.
    samples : int
        Sample points along each edge of a box, corners included, at least two. A box is met where
        its samples are some inside and some outside; with an odd count the box's centre is one of
        them.
    """

    contains: Callable[[np.ndarray], np.ndarray]
    samples: int = 3

    def __post_init__(self) -> None:
        if self.samples < 2:
            raise ValueError(f"SampledRegion.samples must be at least two, got {self.samples!r}.")

    def inside(self, points: np.ndarray) -> np.ndarray:
        """See :meth:`Region.inside`: the inside test itself."""
        return np.asarray(self.contains(np.asarray(points, dtype=float)), dtype=bool)

    def meets(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """See :meth:`Region.meets`: whether a box's samples are some inside and some outside.

        Only cubes are asked about, so the boxes must be cubes.
        """
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        size = upper[:, 0] - lower[:, 0]
        points = cube_samples(lower, size, self.samples, 0.0)
        inside = self.inside(points.reshape(-1, 3)).reshape(points.shape[:2])
        return np.any(inside, axis=1) & ~np.all(inside, axis=1)


@dataclasses.dataclass(frozen=True)
class ClosedSurface:
    """The region a closed, consistently wound triangle surface bounds.

    Built by :meth:`of`. The surface may have several pieces -- a vessel's wall and the walls of a
    hole in it -- so long as together they bound the region.

    Attributes
    ----------
    grid : TriangleGrid
        The triangles, registered in the voxels their bounding boxes span.
    label : np.ndarray of int, shape ``grid.resolution``
        Each voxel no triangle is registered in, labelled by the face-connected stretch of such
        voxels it belongs to (from one); ``0`` for a voxel holding a triangle.
    within : np.ndarray of bool, shape ``(n_labels + 1,)``
        Whether each stretch lies inside the region (entry ``0`` unused). The surface does not
        reach into a stretch, so the whole of it lies on one side.
    """

    grid: TriangleGrid
    label: np.ndarray
    within: np.ndarray

    @classmethod
    def of(cls, points: np.ndarray, triangles: np.ndarray, spacing: float) -> ClosedSurface:
        """The region the surface ``triangles`` over ``points`` bounds, over voxels about
        ``spacing`` across.

        Parameters
        ----------
        points : np.ndarray, shape ``(n_points, 3)``
        triangles : np.ndarray of int, shape ``(n_triangles, 3)``
            The surface, closed and wound out of the region (as
            :func:`~aquaflux.meshing.check_closed_surface` requires); points no triangle uses are
            ignored.
        spacing : float
            The voxels' edge wanted, which bounds how far beyond the surface :meth:`meets` may see
            it. Coarsened where the box would need more than
            :data:`~aquaflux.triangle_grid.MAX_VOXELS`.

        Returns
        -------
        ClosedSurface

        Raises
        ------
        ValueError
            If the spacing is not positive, the surface is not closed and wound outward, or a
            stretch's side cannot be decided (its rays disagree, as one grazing the surface may).
        """
        if not (np.isfinite(spacing) and spacing > 0.0):
            raise ValueError(f"the voxel spacing must be positive and finite, got {spacing!r}")
        triangles = np.asarray(triangles, dtype=np.int64)
        used, local = np.unique(triangles, return_inverse=True)
        check_closed_surface(np.asarray(points, dtype=float)[used], local.reshape(-1, 3))
        vertices = np.asarray(points, dtype=float)[triangles]
        grid = triangle_grid(vertices, spacing)
        held = (np.diff(grid.starts) > 0).reshape(tuple(grid.resolution))
        label, n_labels = ndimage.label(~held)
        within = np.zeros(n_labels + 1, dtype=bool)
        if n_labels:
            first = ndimage.labeled_comprehension(
                np.arange(label.size).reshape(label.shape), label, np.arange(1, n_labels + 1),
                np.min, np.int64, -1,
            )  # fmt: skip
            voxel = np.stack(np.unravel_index(first, label.shape), axis=1)
            within[1:] = _parity(grid.low + (voxel + 0.5) * grid.spacing, vertices)
        return cls(grid=grid, label=label, within=within)

    def inside(self, points: np.ndarray) -> np.ndarray:
        """See :meth:`Region.inside`: the side of its stretch for a point in a voxel the surface
        does not reach, and its own rays' parity for one in a voxel it does. Outside the grid,
        which holds the surface, a point is outside."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        cell = np.floor((points - self.grid.low) / self.grid.spacing).astype(np.int64)
        in_grid = np.all((cell >= 0) & (cell < self.grid.resolution), axis=1)
        found = np.zeros(points.shape[0], dtype=bool)
        stretch = np.zeros(points.shape[0], dtype=np.int64)
        stretch[in_grid] = self.label[tuple(cell[in_grid].T)]
        found[in_grid] = self.within[stretch[in_grid]]
        near = in_grid & (stretch == 0)
        if np.any(near):
            found[near] = _parity(points[near], self.grid.vertices)
        return found

    def meets(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """See :meth:`Region.meets`: whether any voxel a box overlaps holds a triangle, so never
        False where the surface passes through the box. A box clear of the grid, which holds the
        surface, overlaps none (the grid's own test would read it against its edge voxels)."""
        return grid_meets(self.grid, lower, upper)


def triangle_grid(vertices: np.ndarray, spacing: float) -> TriangleGrid:
    """Triangles registered in a grid of voxels about ``spacing`` across over their box, coarsened
    where that would need more than :data:`~aquaflux.triangle_grid.MAX_VOXELS`.

    Parameters
    ----------
    vertices : np.ndarray, shape ``(n, 3, 3)``
        Each triangle's corners.
    spacing : float
        The voxels' edge wanted.

    Returns
    -------
    TriangleGrid
    """
    extent = vertices.reshape(-1, 3).max(axis=0) - vertices.reshape(-1, 3).min(axis=0)
    counts = np.maximum(np.ceil(extent / spacing), 1).astype(np.int64)
    while np.prod(counts, dtype=float) > MAX_VOXELS:
        counts = np.maximum(counts // 2, 1)
    return TriangleGrid.build(vertices, resolution=tuple(int(n) for n in counts))


def grid_meets(grid: TriangleGrid, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Whether any voxel each box overlaps holds a triangle of ``grid``, so never False where a
    triangle passes through the box; a box clear of the grid, which holds every triangle, overlaps
    none (the grid's own test would read it against its edge voxels).

    Parameters
    ----------
    grid : TriangleGrid
    lower, upper : np.ndarray, shape ``(m, 3)``
        Opposite corners of each box.

    Returns
    -------
    np.ndarray of bool, shape ``(m,)``
    """
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    high = grid.low + grid.spacing * grid.resolution
    overlaps = np.all((upper >= grid.low) & (lower <= high), axis=1)
    return overlaps & grid.holds_any(lower, upper)


def _parity(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Whether each point lies inside the closed surface ``vertices`` (shape ``(n, 3, 3)``), by
    the parity of the triangles a ray from it crosses, in each of :data:`_RAY_DIRECTIONS`.

    Raises
    ------
    ValueError
        If the directions disagree at a point, as a ray through a triangle's edge or along its
        plane may.
    """
    votes = np.array(
        [[ray_hits(point, direction, vertices).size % 2 == 1 for direction in _RAY_DIRECTIONS]
         for point in points],
        dtype=bool,
    ).reshape(-1, _RAY_DIRECTIONS.shape[0])  # fmt: skip
    torn = np.flatnonzero(np.any(votes != votes[:, :1], axis=1))
    if torn.size:
        raise ValueError(
            f"rays from {torn.size} point(s), the first at {points[torn[0]].tolist()}, disagree "
            "about which side of the surface it lies on: one grazes the surface"
        )
    return votes[:, 0]


def first_hits(points, faces, origins, directions, reach, skipped) -> np.ndarray:
    """How far each ray goes before it meets a surface again: its nearest crossing of a face.

    Only faces whose centroid lies within ``reach`` of a ray's origin are looked at, and none with a
    corner among the ray's ``skipped`` points (the faces it starts on); a ray crossing none goes
    ``inf``. A quadrilateral is crossed where either triangle of a diagonal cut is.

    Parameters
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
    faces : Polygons
        The surface.
    origins, directions : np.ndarray, shape ``(n_rays, 3)``
        Each ray's start and its direction, a unit vector.
    reach : np.ndarray, shape ``(n_rays,)``
    skipped : np.ndarray of int, shape ``(n_rays, k)``
        Points whose faces each ray ignores; ``-1`` pads a row.

    Returns
    -------
    np.ndarray, shape ``(n_rays,)``
    """
    points = np.asarray(points, dtype=np.float64)
    triangles = faces.as_triangles()
    rings = faces.padded()[faces.triangle_face()]
    tree = cKDTree(points[triangles].mean(axis=1))
    gaps = np.full(len(origins), np.inf)
    for ray in range(len(origins)):
        near = np.asarray(tree.query_ball_point(origins[ray], reach[ray]), dtype=np.int64)
        own = skipped[ray][skipped[ray] >= 0]
        near = near[~np.any(np.isin(rings[near], own), axis=1)]
        if not near.size:
            continue
        hit = ray_hits(origins[ray], directions[ray], points[triangles[near]])
        if hit.size:
            gaps[ray] = hit.min()
    return gaps


def ray_hits(origin, direction, corners) -> np.ndarray:
    """The distances along a ray at which it crosses each triangle it crosses (Moller-Trumbore).

    Parameters
    ----------
    origin, direction : np.ndarray, shape ``(3,)``
    corners : np.ndarray, shape ``(n, 3, 3)``

    Returns
    -------
    np.ndarray
        The distances, in units of ``direction``'s length, each positive.
    """
    edge1 = corners[:, 1] - corners[:, 0]
    edge2 = corners[:, 2] - corners[:, 0]
    pvec = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tvec = origin - corners[:, 0]
    u = np.einsum("ij,ij->i", tvec, pvec) * inv
    qvec = np.cross(tvec, edge1)
    v = (qvec @ direction) * inv
    t = np.einsum("ij,ij->i", edge2, qvec) * inv
    inside = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return t[inside]
