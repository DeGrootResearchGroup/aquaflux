"""A uniform grid over the blocking triangles, so a segment tests a few of them instead of all.

The shadow mask asks one question per (receiver, facet) pair -- does anything lie across this
segment -- and answers it by testing every triangle of the surface. That is
``n_receivers x n_facets x n_triangles``, which a reactor puts out of reach: 1.6M cells, 7,516
lamp facets and 53,500 wall triangles is 6.6e14 intersections.

This is the standard remedy. Each triangle is registered in the voxels its bounding box spans;
a segment walks the voxels it crosses (Amanatides & Woo's 3D-DDA) and tests only what those
voxels hold, stopping at the first triangle that blocks it. **Measured on that reactor**
(``validation/sozzi_radiation/ray_acceleration_probe.py``), a 128-cubed grid holds 9.7
triangles in the average occupied voxel and a segment enters 65 of them, so it tests **52
triangles instead of 53,500** -- and, stopping early, reaches a blocker after about **5**.

⚠️ **The walk is deliberately NOT a traced kernel, and that is the whole reason it works.** The
mask is frozen: it is built once from geometry alone, so none of the constraints that shape the
rest of this package apply to it. A traced walk would need a static trip count and a static
number of triangles per voxel, and would therefore pay the *worst* case on every ray -- 183
steps times 36 triangles, **6,588 tests a ray**, worse than the brute force it replaces.

⚠️ **Nor is it array code, which prices every step instead.** A walk written as whole-array
passes over the rays still in flight -- which this one was -- spends about 150-180 ns per ray per
voxel step on that bookkeeping (profiled on a 51,200-triangle cylindrical wall), most of its
time, so refining its grid made it *slower*: fewer triangles tested, more steps taken. Here each
ray is walked to its first hit by one compiled loop (:mod:`~aquaflux.radiation.grid_walk`), with
its state in registers and nothing formed per step, which measured 17-37x faster on the same
rays with identical answers.

**Where its time goes.** Not in empty space: a compiled step through an empty voxel costs about
as much as one or two triangle tests, and on that reactor a segment to a cell in an outlet pipe
spent its time testing the pipe wall's triangles in every voxel it crossed, the pipe being about
a voxel across. Which end a segment is walked from made no difference there either (0.95-1.06x).
What moved the cost was the voxels' *shape*, which is why the default grid takes the proportions
of the triangles' bounding box.

**What it does not do.** It does not reduce the number of *rays*, which at mesh scale is the
binding cost: 1.6M cells against 7,516 facets is 1.2e10 segments however cheaply each is
answered. Cutting the rays walked is what the clearance certificates of a
:class:`~aquaflux.radiation.TriangleBody` are for.
"""

from __future__ import annotations

import dataclasses

import numba
import numpy as np

from aquaflux.radiation.grid_walk import walk_to_first_hit

__all__ = ["TriangleGrid"]

#: Triangles per occupied voxel of the near-cubic grid a default grid's voxel count is scaled
#: from (``_VOXEL_BUDGET``), and that a body's occupancy grid is refined from. Refining a
#: near-cubic grid trades steps for tests: on the Sozzi reactor's wall, 1x to 4x per axis cut the
#: triangles tested per segment 444 to 139 and raised the steps 64 to 255, at the same rate.
_TARGET_PER_VOXEL = 10.0

#: How many voxels a default grid spends, as a multiple of the near-cubic grid at
#: ``_TARGET_PER_VOXEL``. On the two synthetic vessels of ``validation/radiation_grid_walk.py``,
#: halving the voxels per axis cost 1% (long vessel) and 19% (annular reactor) and doubling them
#: cost 18% and gained 1%; on the Sozzi wall a grid with about twice the voxels ran faster still.
#: So it is a compromise across scenes, not any one scene's optimum.
_VOXEL_BUDGET = 4.0

#: Longest voxel edge over the shortest, in a default grid. The Sozzi reactor's box is 19:1.
_MAX_ASPECT = 32.0

#: Never build more voxels than this, whatever the triangle count asks for. The grid's own
#: memory is one index per voxel plus one entry per (triangle, voxel it spans).
_MAX_VOXELS = 8_000_000

#: The thickness a flat axis of the grid's box is given, as a share of the box's widest extent
#: (``_box_and_area``). Thin enough that a segment crossing the plane steps through few extra
#: voxels, and far wider than the rounding of a crossing point, which must land inside it.
_FLAT_THICKNESS = 1e-6


@dataclasses.dataclass(frozen=True)
class TriangleGrid:
    """Triangles indexed by the voxels their bounding boxes span.

    Built by :meth:`build`. Plain host arrays rather than a pytree: nothing here is traced or
    differentiated, and the grid exists only to decide which triangles a segment is worth
    testing against.

    Attributes
    ----------
    vertices : np.ndarray, shape ``(n_triangles, 3, 3)``
        The triangles themselves, in the order the indices refer to.
    low, spacing : np.ndarray, shape ``(3,)``
        The grid's corner and its voxel size.
    resolution : np.ndarray of int, shape ``(3,)``
        Voxels along each axis.
    starts : np.ndarray of int, shape ``(n_voxels + 1,)``
        Where each voxel's triangles begin in :attr:`triangles` -- compressed-sparse-row form,
        a row-pointer array plus a flat index array.
    triangles : np.ndarray of int
        Triangle indices, grouped by voxel.
    occupied_below : np.ndarray of int, shape ``resolution + 1``
        A summed-volume table of the occupied voxels: entry ``[i, j, k]`` counts the occupied
        voxels with every index below ``(i, j, k)``, so how many a box of voxels holds is eight
        lookups whatever its size (:meth:`holds_any`).
    """

    vertices: np.ndarray
    low: np.ndarray
    spacing: np.ndarray
    resolution: np.ndarray
    starts: np.ndarray
    triangles: np.ndarray
    occupied_below: np.ndarray

    @property
    def n_voxels(self) -> int:
        """How many voxels the grid holds."""
        return int(np.prod(self.resolution))

    @classmethod
    def build(cls, vertices, *, resolution: int | tuple[int, int, int] | None = None):
        """Register every triangle in the voxels its bounding box, a rounding wider, spans.

        Parameters
        ----------
        vertices : array_like, shape ``(n_triangles, 3, 3)``
            The blocking triangles.
        resolution : int or tuple of int, optional
            Voxels per axis. The default gives every axis about the same number of voxels, so
            each voxel has the proportions of the surface's bounding box, no edge more than
            32 times another; and it holds four times the voxels of a near-cubic grid sized so
            an occupied voxel holds about ten triangles: a segment's steps along an axis go as its
            travel there over the voxel's edge, and a shadow mask's segments spread through the
            domain as its bounding box does.

        Returns
        -------
        TriangleGrid

        Raises
        ------
        ValueError
            If there are no triangles, or a resolution is not positive.
        """
        vertices = np.ascontiguousarray(vertices, dtype=float)
        if vertices.ndim != 3 or vertices.shape[1:] != (3, 3):
            msg = f"vertices must have shape (n_triangles, 3, 3); got {vertices.shape}"
            raise ValueError(msg)
        if len(vertices) == 0:
            msg = "a grid needs at least one triangle"
            raise ValueError(msg)
        low, extent, area = _box_and_area(vertices)
        counts = _resolution(extent, area, len(vertices), resolution)
        spacing = extent / counts

        # Each triangle's box is widened by the margin the box test widens by. A triangle lying on
        # a voxel boundary -- an axis-aligned face on a voxel plane, an edge along one -- then sits
        # in the voxels on both sides of it, so whichever of them the rounding of a walk visits as
        # it passes through that boundary holds it. Unwidened, a segment through such an edge
        # where the walk's tie-break steps around the one voxel holding the triangle the exact
        # test credits the hit to reads clear.
        margin = _rounding_margin(extent)
        span_low = np.clip(
            ((vertices.min(axis=1) - margin - low) / spacing).astype(int), 0, counts - 1
        )
        span_high = np.clip(
            ((vertices.max(axis=1) + margin - low) / spacing).astype(int), 0, counts - 1
        )
        voxel_of, triangle_of = _spans(span_low, span_high, counts)
        order = np.argsort(voxel_of, kind="stable")
        per_voxel = np.bincount(voxel_of, minlength=int(np.prod(counts)))
        occupied_below = np.zeros(tuple(counts + 1), dtype=np.int64)
        occupied_below[1:, 1:, 1:] = (
            (per_voxel > 0).reshape(tuple(counts)).cumsum(0).cumsum(1).cumsum(2)
        )
        return cls(
            vertices=vertices,
            low=low,
            spacing=spacing,
            resolution=counts,
            starts=np.concatenate([[0], np.cumsum(per_voxel)]),
            triangles=triangle_of[order],
            occupied_below=occupied_below,
        )

    def holds_any(self, low, high) -> np.ndarray:
        """Whether any triangle could meet each axis-aligned box: False only where none can.

        A triangle is registered in every voxel its bounding box, widened by a margin a
        billionth of the grid's extent, spans, so a triangle that meets a box does so at a point
        lying in some voxel it is registered in -- one the box overlaps. So a box overlapping no
        occupied voxel meets no triangle, exactly, and saying so costs eight lookups in
        :attr:`occupied_below` however large the box. The voxel range is found by the same
        truncation the registration uses, and the box is first widened by the same margin, so
        a triangle a rounding outside a box does not read as clear of a segment the exact test
        would call cut.

        Parameters
        ----------
        low, high : array_like, shape ``(..., 3)``
            Opposite corners of each box.

        Returns
        -------
        np.ndarray of bool, shape ``(...)``
        """
        low = np.asarray(low, dtype=float)
        shape = low.shape[:-1]
        low = np.ascontiguousarray(low.reshape(-1, 3))
        high = np.ascontiguousarray(np.broadcast_to(np.asarray(high, dtype=float), (*shape, 3)))
        every = np.arange(len(low))
        held = _unions_held(
            low, high.reshape(-1, 3), every, low, high.reshape(-1, 3), every, *self._lookup
        )
        return held.reshape(shape)

    def holds_any_in_unions(self, first_low, first_high, second_low, second_high, rows, cols):
        """:meth:`holds_any` for the bounding box of each pair of boxes, without forming it.

        Box ``t`` is the smallest box holding both ``first[rows[t]]`` and ``second[cols[t]]``: the
        smaller of their low corners and the larger of their high ones. One compiled loop forms
        each such box's corners and makes its eight lookups, so nothing the size of the pairs is
        written out, and the answers are those :meth:`holds_any` gives on the boxes themselves.

        Parameters
        ----------
        first_low, first_high : array_like, shape ``(n_first, 3)``
            Opposite corners of the first set's boxes.
        second_low, second_high : array_like, shape ``(n_second, 3)``
            Opposite corners of the second set's boxes.
        rows, cols : array_like of int, shape ``(n_pairs,)``
            Which box of each set each pair joins.

        Returns
        -------
        np.ndarray of bool, shape ``(n_pairs,)``
        """

        def corners(array):
            return np.ascontiguousarray(array, dtype=float)

        return _unions_held(
            corners(first_low),
            corners(first_high),
            np.ascontiguousarray(rows, dtype=np.int64),
            corners(second_low),
            corners(second_high),
            np.ascontiguousarray(cols, dtype=np.int64),
            *self._lookup,
        )

    @property
    def _lookup(self):
        """What the compiled box test reads of the grid: corner, voxel size, margin, top, table."""
        margin = _rounding_margin(self.spacing * self.resolution)
        top = (self.resolution - 1).astype(np.int64)
        return self.low, self.spacing, margin, top, self.occupied_below

    def blocks(self, origin, target, min_distance, *, exclude=None) -> np.ndarray:
        """Whether any triangle lies across each segment, testing only what the grid selects.

        The contract is :func:`~aquaflux.radiation.triangles.segment_is_cut`'s, and the answers
        are the same: the grid decides what is *worth* testing, never what counts as a hit. Each
        segment is walked to its first hit, or its end, by
        :func:`~aquaflux.radiation.grid_walk.walk_to_first_hit`, so nothing the size of the rays
        times the triangles a step visits is ever formed, and no work limit is needed.

        Parameters
        ----------
        origin, target : array_like, shape ``(n_rays, 3)``
            Segment endpoints; ``origin`` is the source.
        min_distance : array_like, shape ``(n_rays,)``
            How far from ``origin`` a hit must be before it counts, in length units.
        exclude : array_like of int, shape ``(n_rays,)`` or ``(n_rays, k)``, optional
            Triangles each ray ignores; ``-1`` excludes nothing.

        Returns
        -------
        np.ndarray of bool, shape ``(n_rays,)``
        """
        origin = np.ascontiguousarray(origin, dtype=float)
        target = np.ascontiguousarray(target, dtype=float)
        near = np.ascontiguousarray(np.broadcast_to(min_distance, origin.shape[:1]), dtype=float)
        exclude = _exclusions(exclude, len(origin))
        direction = target - origin
        # The segment is parametrized on [0, 1], so a hit distance is in those units too --
        # which is what makes the walk's own parameter and the intersection test's comparable.
        # `min_distance` is a LENGTH, though, as `segment_is_cut` takes it, so it converts here
        # rather than being compared against a parameter it does not share units with.
        length = np.sqrt(np.sum(direction * direction, axis=-1))
        near = near / np.where(length == 0.0, 1.0, length)
        blocked = np.zeros(len(origin), dtype=bool)
        alive, entry = _enters_grid(origin, direction, self.low, self.spacing, self.resolution)
        ray = np.flatnonzero(alive)
        if len(ray) == 0:
            return blocked
        voxel, until, step, delta = _walk_state(
            origin[ray], direction[ray], entry[ray], self.low, self.spacing, self.resolution
        )
        # A DDA visits at most nx + ny + nz - 2 voxels, so this bound is never what ends a walk.
        max_steps = int(self.resolution.sum()) + 3
        return walk_to_first_hit(
            ray, voxel, until, step, delta, origin, direction, near, exclude, self, max_steps
        )


@numba.njit(inline="always")
def _voxel_span(low, high, corner, spacing, margin, top):
    """The voxels a box's extent along one axis overlaps, as ``first`` and one past ``last``.

    The same truncation the registration uses, after widening by ``margin`` either side.
    """
    first = min(max(int((low - margin - corner) / spacing), 0), top)
    last = min(max(int((high + margin - corner) / spacing), 0), top) + 1
    return first, last


@numba.njit(parallel=True)
def _unions_held(
    first_low, first_high, rows, second_low, second_high, cols, corner, spacing, margin, top, table
):
    """Whether any occupied voxel lies in the bounding box of each pair of boxes, shape ``(n_pairs,)``."""
    held = np.empty(len(rows), dtype=np.bool_)
    for pair in numba.prange(len(rows)):
        row, col = rows[pair], cols[pair]
        fx, lx = _voxel_span(
            min(first_low[row, 0], second_low[col, 0]),
            max(first_high[row, 0], second_high[col, 0]),
            corner[0],
            spacing[0],
            margin[0],
            top[0],
        )
        fy, ly = _voxel_span(
            min(first_low[row, 1], second_low[col, 1]),
            max(first_high[row, 1], second_high[col, 1]),
            corner[1],
            spacing[1],
            margin[1],
            top[1],
        )
        fz, lz = _voxel_span(
            min(first_low[row, 2], second_low[col, 2]),
            max(first_high[row, 2], second_high[col, 2]),
            corner[2],
            spacing[2],
            margin[2],
            top[2],
        )
        count = (
            table[lx, ly, lz]
            - table[fx, ly, lz]
            - table[lx, fy, lz]
            - table[lx, ly, fz]
            + table[fx, fy, lz]
            + table[fx, ly, fz]
            + table[lx, fy, fz]
            - table[fx, fy, fz]
        )
        held[pair] = count > 0
    return held


def _rounding_margin(extent) -> np.ndarray:
    """A billionth of ``extent``: how far past a box a rounding may put what belongs in it.

    The grid's box is padded by it, and the registration and the box test widen by it, so the
    three agree on how far a rounding reaches.
    """
    return 1e-9 * np.asarray(extent, dtype=float)


def _box_and_area(vertices):
    """The grid's corner and extent -- the triangles' bounding box, a hair wider -- and their area."""
    corner = vertices.reshape(-1, 3)
    low, high = corner.min(axis=0), corner.max(axis=0)
    span = high - low
    # A flat axis -- every triangle in one axis-aligned plane, as a planar patch is -- is given a
    # thickness. With none its voxels are a few times 1e-308 thick, so a segment's crossing point
    # a rounding off the plane is a voxel index past the range of an integer, and with more than
    # one voxel through the thickness it lands in a voxel holding nothing and reads clear.
    extent = np.maximum(span, max(_FLAT_THICKNESS * float(span.max()), np.finfo(float).tiny))
    # A margin, so a triangle exactly on the far face still lands inside the grid.
    margin = _rounding_margin(extent)
    low = low - margin
    extent = extent + 2.0 * margin
    # Twice the area, since the cross product of two edges spans the parallelogram.
    edges = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    return low, extent, 0.5 * float(np.sqrt(np.sum(edges * edges, axis=-1)).sum())


def _resolution(extent, area: float, n_triangles: int, resolution) -> np.ndarray:
    """Voxels per axis: what the caller asked for, or voxels shaped like the box.

    **Why the box's shape, and not cubes.** A walk pays per voxel it steps through and per
    triangle it tests. Along each axis the steps go as how far the segment travels along it over
    the voxel's edge there, so for a given number of voxels the steps are fewest when each edge
    is in proportion to that travel. A shadow mask pairs every receiver with every emitting
    facet, so its segments spread through the domain as its bounding box does, and near-cubic
    voxels in a long vessel make every axial segment take many short steps. Measured on three
    long vessels -- a synthetic thin tube, a synthetic annular reactor, and the Sozzi reactor's
    53,500-triangle wall walked by its real cell-to-lamp segments -- box-shaped voxels were
    1.26-1.66x the near-cubic rate at the same voxel count, and this default was 1.33-1.96x the
    near-cubic one it replaced, segments that stay in the open chamber being the exception at 1.03x
    (``validation/radiation_grid_walk.py``,
    ``validation/sozzi_radiation/grid_walk_direction.py``). The box is the triangles', not the
    segments', which is where this can mislead: the edges are capped at ``_MAX_ASPECT`` to
    one another so a very flat box is not cut into sheets a segment crossing it would test whole.

    ⚠️ **The voxel COUNT still comes from the triangles' AREA, not from the box's volume.**
    Blocking triangles tile a surface, so the voxels they occupy are the ones their sheet passes
    through: about ``area / size**2`` of them, however large the box around it. Dividing the
    *volume* into one voxel per ten triangles assumes the box is filled, and a thin shell in a
    long box is the opposite of that -- on a reactor's wall (53,500 triangles over ~0.3 m^2 in
    a 0.9 m box) it sized 5,350 voxels of which **298** were occupied, holding 217 triangles
    each against the ten intended, and a walk through those cost more memory than testing every
    triangle would have. The near-cubic grid of that size (``_near_cubic``) sets the count, and
    ``_VOXEL_BUDGET`` times it is what the box-shaped grid spends.
    """
    if resolution is not None:
        counts = np.broadcast_to(np.asarray(resolution, dtype=int), (3,)).copy()
        if np.any(counts < 1):
            msg = f"a grid resolution must be at least 1 voxel per axis; got {tuple(counts)}"
            raise ValueError(msg)
        return counts
    voxels = _VOXEL_BUDGET * float(np.prod(_near_cubic(extent, area, n_triangles), dtype=float))
    counts = _shaped_like(extent, voxels)
    while np.prod(counts, dtype=float) > _MAX_VOXELS:
        counts = np.maximum(counts // 2, 1)
    return counts


def _near_cubic(extent, area: float, n_triangles: int) -> np.ndarray:
    """Voxels per axis of near-cubic voxels holding about ``_TARGET_PER_VOXEL`` triangles.

    Sized from the triangles' area (see ``_resolution``), and never more than
    ``_MAX_VOXELS`` voxels.
    """
    wanted = max(n_triangles / _TARGET_PER_VOXEL, 1.0)
    size = float(np.sqrt(max(area, np.finfo(float).tiny) / wanted))
    counts = np.maximum(np.round(extent / max(size, np.finfo(float).tiny)), 1).astype(int)
    while np.prod(counts, dtype=float) > _MAX_VOXELS:
        counts = np.maximum(counts // 2, 1)
    return counts


def _shaped_like(extent, voxels: float) -> np.ndarray:
    """About ``voxels`` voxels in all, their edges in the box's proportions within the cap.

    Each voxel edge is a common scale times its axis's extent, the shorter extents first raised
    to ``_MAX_ASPECT`` below the longest. An axis that would get less than one voxel gets
    exactly one -- a flat box is one voxel thick -- and the others share the count between them.
    """
    edge = np.maximum(extent, extent.max() / _MAX_ASPECT)
    free = np.ones(3, dtype=bool)
    while True:
        # The scale at which the free axes' counts multiply to `voxels`. Their geometric mean
        # is then at least one, so at least one of them is, and `free` never empties.
        scale = (np.prod(extent[free] / edge[free]) / max(voxels, 1.0)) ** (1.0 / free.sum())
        counts = np.where(free, extent / (scale * edge), 1.0)
        if np.all(counts[free] >= 1.0):
            return np.maximum(np.round(counts), 1).astype(int)
        free &= counts >= 1.0


def _near_cubic_resolution(vertices) -> np.ndarray:
    """Voxels per axis of a near-cubic grid over these triangles, about ten in an occupied voxel.

    What a grid read for which voxels are occupied, rather than walked, is sized from: a box is
    clear of the surface only if it overlaps no occupied voxel, which wants voxels small in
    every direction rather than long along the segments.

    Parameters
    ----------
    vertices : array_like, shape ``(n_triangles, 3, 3)``

    Returns
    -------
    np.ndarray of int, shape ``(3,)``
    """
    vertices = np.ascontiguousarray(vertices, dtype=float)
    _, extent, area = _box_and_area(vertices)
    return _near_cubic(extent, area, len(vertices))


def _spans(span_low, span_high, counts):
    """Every (voxel, triangle) pair a set of index boxes covers."""
    sizes = span_high - span_low + 1
    total = np.prod(sizes, axis=1)
    triangle_of = np.repeat(np.arange(len(sizes)), total)
    within = np.arange(total.sum()) - np.repeat(np.cumsum(total) - total, total)
    depth = np.repeat(sizes[:, 2], total)
    height = np.repeat(sizes[:, 1], total)
    k = within % depth
    j = (within // depth) % height
    i = within // (depth * height)
    index = np.stack([i, j, k], axis=1) + np.repeat(span_low, total, axis=0)
    flat = (index[:, 0] * counts[1] + index[:, 1]) * counts[2] + index[:, 2]
    return flat, triangle_of


def _exclusions(exclude, n_rays: int) -> np.ndarray:
    """The excluded triangle indices as one ``(n_rays, k)`` array, empty meaning none."""
    if exclude is None:
        return np.full((n_rays, 1), -1, dtype=int)
    exclude = np.asarray(exclude, dtype=int)
    return exclude[:, None] if exclude.ndim == 1 else exclude


def _enters_grid(origin, direction, low, spacing, resolution):
    """Where each segment first lies inside the grid, and whether it does at all.

    A receiver outside the surface's own bounding box is ordinary -- a probe beyond the vessel,
    a cell beside a lamp whose box does not reach it -- so the walk starts at the segment's
    entry into the box rather than at its origin, and a segment that misses the box entirely is
    answered without walking. Along an axis it does not move on, a segment is within the box's
    slab for its whole length or for none of it, as its origin is.
    """
    high = low + spacing * resolution
    with np.errstate(divide="ignore", invalid="ignore"):
        to_low = (low - origin) / direction
        to_high = (high - origin) / direction
    within = (origin >= low) & (origin <= high)
    unmoving = direction == 0.0
    near = np.where(unmoving, np.where(within, -np.inf, np.inf), np.minimum(to_low, to_high))
    far = np.where(unmoving, np.inf, np.maximum(to_low, to_high))
    inside = np.all(within, axis=1)
    entry = np.maximum(near.max(axis=1), 0.0)
    exit_at = np.minimum(far.min(axis=1), 1.0)
    return inside | (entry <= exit_at), np.where(inside, 0.0, entry)


def _walk_state(origin, direction, entry, low, spacing, resolution):
    """The starting voxel of each segment and the parameters its walk steps with."""
    start = origin + entry[:, None] * direction
    voxel = np.clip(((start - low) / spacing).astype(int), 0, resolution - 1)
    step = np.where(direction > 0, 1, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.abs(spacing / direction)
        boundary = low + (voxel + (step > 0)) * spacing
        until = (boundary - origin) / direction
    unmoving = direction == 0.0
    return voxel, np.where(unmoving, np.inf, until), step, np.where(unmoving, np.inf, delta)
