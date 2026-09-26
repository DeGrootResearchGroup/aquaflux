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

⚠️ **This is deliberately NOT a traced kernel, and that is the whole reason it works.** The
mask is frozen: it is built once from geometry alone, so none of the constraints that shape the
rest of this package apply to it. A traced walk would need a static trip count and a static
number of triangles per voxel, and would therefore pay the *worst* case on every ray -- 183
steps times 36 triangles, **6,588 tests a ray**, worse than the brute force it replaces. The
walk here is ordinary host code over the rays still in flight, which shrinks as they hit
something or run out; only the intersection test itself is traced, on the compacted
(ray, triangle) pairs, through the same predicate the unaccelerated path uses.

**Two walks, one set of answers.** That array walk pays for every step in whole-array operations
over every ray still in flight -- about 150-180 ns per ray per voxel step, profiled on a
51,200-triangle cylindrical wall, which is most of its time, and which is why a finer grid makes it
slower: fewer triangles tested, more steps taken. ``walk="compiled"``
(:mod:`~aquaflux.radiation.grid_walk`, which needs Numba) walks each ray to its end in one compiled
loop instead, visiting the same voxels and testing the same triangles in the same order.

**What it does not do.** It does not reduce the number of *rays*, which at mesh scale is the
binding cost: 1.6M cells against 7,516 facets is 1.2e10 segments however cheaply each is
answered. It makes scenes of up to a few times 1e8 rays practical; beyond that the ray count
has to come down instead.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aquaflux.radiation.triangles import _pair_is_cut, padded_length

__all__ = ["TriangleGrid"]

#: Triangles per occupied voxel a default grid aims for. Below about this the walk lengthens
#: faster than the per-voxel work falls: on the measured reactor, going from 64 to 128 cubed
#: halved the triangles tested per ray (121 to 52) while doubling the steps taken (33 to 65).
_TARGET_PER_VOXEL = 10.0

#: Never build more voxels than this, whatever the triangle count asks for. The grid's own
#: memory is one index per voxel plus one entry per (triangle, voxel it spans).
_MAX_VOXELS = 8_000_000


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
        """Register every triangle in the voxels its bounding box spans.

        Parameters
        ----------
        vertices : array_like, shape ``(n_triangles, 3, 3)``
            The blocking triangles.
        resolution : int or tuple of int, optional
            Voxels per axis. The default divides the surface's bounding box so that an occupied
            voxel holds about ten triangles, which is where the measured trade between walk
            length and per-voxel work sits, and keeps the axes' voxels near cubic so a long
            thin domain is not walked in tiny steps across its short axis.

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
        corner = vertices.reshape(-1, 3)
        low, high = corner.min(axis=0), corner.max(axis=0)
        extent = np.maximum(high - low, np.finfo(float).tiny)
        # A margin, so a triangle exactly on the far face still lands inside the grid.
        low = low - 1e-9 * extent
        extent = extent * (1.0 + 2e-9)
        # Twice the area, since the cross product of two edges spans the parallelogram.
        edges = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        area = 0.5 * float(np.sqrt(np.sum(edges * edges, axis=-1)).sum())
        counts = _resolution(extent, area, len(vertices), resolution)
        spacing = extent / counts

        span_low = np.clip(((vertices.min(axis=1) - low) / spacing).astype(int), 0, counts - 1)
        span_high = np.clip(((vertices.max(axis=1) - low) / spacing).astype(int), 0, counts - 1)
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

        A triangle is registered in every voxel its bounding box spans, so a triangle that
        meets a box does so at a point lying in some voxel it is registered in -- one the box
        overlaps. So a box overlapping no occupied voxel meets no triangle, exactly, and saying
        so costs eight lookups in :attr:`occupied_below` however large the box. The voxel range
        is found by the same truncation the registration uses, and the box is first widened by
        a margin a billionth of the grid's extent, so a triangle a rounding outside a box does
        not read as clear of a segment the exact test would call cut.

        Parameters
        ----------
        low, high : array_like, shape ``(..., 3)``
            Opposite corners of each box.

        Returns
        -------
        np.ndarray of bool, shape ``(...)``
        """
        margin = 1e-9 * self.spacing * self.resolution
        top = self.resolution - 1
        first = np.clip(
            ((np.asarray(low, dtype=float) - margin - self.low) / self.spacing).astype(int), 0, top
        )
        last = (
            np.clip(
                ((np.asarray(high, dtype=float) + margin - self.low) / self.spacing).astype(int),
                0,
                top,
            )
            + 1
        )
        table = self.occupied_below

        def at(x, y, z):
            return table[x[..., 0], y[..., 1], z[..., 2]]

        held = (
            at(last, last, last)
            - at(first, last, last)
            - at(last, first, last)
            - at(last, last, first)
            + at(first, first, last)
            + at(first, last, first)
            + at(last, first, first)
            - at(first, first, first)
        )
        return held > 0

    def blocks(
        self,
        origin,
        target,
        min_distance,
        *,
        exclude=None,
        work_limit: int = 4_000_000,
        walk: str = "array",
    ) -> np.ndarray:
        """Whether any triangle lies across each segment, testing only what the grid selects.

        The contract is :func:`~aquaflux.radiation.triangles.segment_is_cut`'s, and the answers
        are the same: the grid decides what is *worth* testing, never what counts as a hit.

        Parameters
        ----------
        origin, target : array_like, shape ``(n_rays, 3)``
            Segment endpoints; ``origin`` is the source.
        min_distance : array_like, shape ``(n_rays,)``
            How far from ``origin`` a hit must be before it counts, in length units.
        exclude : array_like of int, shape ``(n_rays,)`` or ``(n_rays, k)``, optional
            Triangles each ray ignores; ``-1`` excludes nothing.
        work_limit : int, optional
            Most (ray, triangle) pairs to hold at once, as
            :func:`~aquaflux.radiation.triangles.segment_is_cut` bounds its own block. A step of
            the walk tests every live ray against everything its voxel holds, so without a bound
            a coarse grid over many rays builds one array of every pair in that step -- which is
            how a grid runs a machine out of memory rather than saving it work. The compiled walk
            forms no pairs and ignores it.
        walk : {"array", "compiled"}, optional
            How the segments are walked. ``"array"`` (the default) steps every ray still in
            flight together, as whole-array operations, and tests each step's pairs in one traced
            call. ``"compiled"`` walks each ray to its end in one compiled loop
            (:mod:`~aquaflux.radiation.grid_walk`, which needs Numba), visiting the same voxels
            and testing the same triangles, so the answers agree.

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
        if walk not in ("array", "compiled"):
            msg = f"walk must be 'array' or 'compiled'; got {walk!r}"
            raise ValueError(msg)
        alive, entry = _enters_grid(origin, direction, self.low, self.spacing, self.resolution)
        if not np.any(alive):
            return blocked
        max_steps = int(self.resolution.sum()) + 3
        if walk == "compiled":
            from aquaflux.radiation.grid_walk import walk_to_first_hit

            ray = np.flatnonzero(alive)
            state = _walk_state(
                origin[ray], direction[ray], entry[ray], self.low, self.spacing, self.resolution
            )
            return walk_to_first_hit(ray, *state, origin, direction, near, exclude, self, max_steps)
        # Every ray's data and every triangle go to the kernel once per call; a step of the walk
        # then sends only which ray meets which triangle, as two indices, and the kernel gathers
        # the rest itself. Gathering them here instead copied about 150 bytes a candidate pair
        # on the host, three times over, which was most of a walk's time.
        # Padded to a power of two, like the pairs below, so calls with different ray counts
        # share their compiled programs; no pair ever names a padding ray.
        spare = padded_length(len(origin)) - len(origin)
        rays = _Rays(
            origin=jnp.asarray(_pad(origin, spare)),
            direction=jnp.asarray(_pad(direction, spare)),
            near=jnp.asarray(_pad(near, spare)),
            exclude=jnp.asarray(_pad(exclude, spare)),
            vertices=jnp.asarray(self.vertices),
        )

        ray = np.flatnonzero(alive)
        voxel, until, step, delta = _walk_state(
            origin[ray], direction[ray], entry[ray], self.low, self.spacing, self.resolution
        )
        for _ in range(max_steps):
            if len(ray) == 0:
                break
            flat = (voxel[:, 0] * self.resolution[1] + voxel[:, 1]) * self.resolution[2] + voxel[
                :, 2
            ]
            hit = self._test(ray, flat, rays, work_limit)
            blocked[ray[hit]] = True
            axis = np.argmin(until, axis=1)
            rows = np.arange(len(ray))
            leaving = until[rows, axis]
            # Done when it hit something, when the next crossing is past the far end, or when
            # the walk would leave the grid.
            moved = voxel[rows, axis] + step[rows, axis]
            keep = ~hit & (leaving <= 1.0) & (moved >= 0) & (moved < self.resolution[axis])
            voxel[rows, axis] = moved
            until[rows, axis] = leaving + delta[rows, axis]
            ray, voxel, until, step, delta = (
                ray[keep],
                voxel[keep],
                until[keep],
                step[keep],
                delta[keep],
            )
        return blocked

    def _test(self, ray, flat, rays, work_limit) -> np.ndarray:
        """Test each live ray against the triangles of the voxel it is in."""
        first, last = self.starts[flat], self.starts[flat + 1]
        held = last - first
        busy = np.flatnonzero(held)
        hit = np.zeros(len(ray), dtype=bool)
        if len(busy) == 0:
            return hit
        counts = held[busy]
        for start, stop in _work_groups(counts, work_limit):
            group, repeats = busy[start:stop], counts[start:stop]
            # Compressed-sparse-row expansion: one (ray, triangle) pair per triangle held by the
            # voxel that ray is in, without a Python loop over the rays or the voxels.
            opens = np.cumsum(repeats) - repeats
            offsets = np.arange(repeats.sum()) - np.repeat(opens, repeats)
            candidate = self.triangles[np.repeat(first[group], repeats) + offsets]
            of_ray = np.repeat(ray[group], repeats)
            # Padded to a power of two so a walk compiles a couple of dozen programs rather than
            # one per step; the padding repeats a real pair and its answer is dropped.
            pad = padded_length(len(candidate)) - len(candidate)
            struck = np.asarray(
                _indexed_pair_is_cut(
                    rays,
                    jnp.asarray(_pad(of_ray.astype(np.int32), pad)),
                    jnp.asarray(_pad(candidate.astype(np.int32), pad)),
                )
            )[: len(candidate)]
            # A ray's candidates are contiguous, in the order of its group, so each ray's answer
            # is one reduction over its own run.
            hit[group] |= np.logical_or.reduceat(struck, opens)
        return hit


class _Rays(eqx.Module):
    """One call's rays and triangles, held by the kernel for the whole walk."""

    origin: jnp.ndarray
    direction: jnp.ndarray
    near: jnp.ndarray
    exclude: jnp.ndarray
    vertices: jnp.ndarray


@jax.jit
def _indexed_pair_is_cut(rays: _Rays, ray, triangle):
    """Whether each ray ``ray[k]`` meets triangle ``triangle[k]``, gathering both here."""
    return _pair_is_cut(
        rays.origin[ray],
        rays.direction[ray],
        rays.near[ray],
        rays.vertices[triangle],
        triangle,
        rays.exclude[ray],
    )


def _work_groups(counts: np.ndarray, work_limit: int):
    """Slices of a per-voxel triangle count whose pairs each fit ``work_limit``.

    Yields ``(start, stop)`` index pairs. A single voxel holding more than the limit is its own
    group rather than being dropped -- the limit bounds what is held at once where it can, and
    a grid coarse enough to break it is the caller's to re-size.
    """
    total = int(counts.sum())
    if total <= work_limit or len(counts) == 1:
        yield 0, len(counts)
        return
    running = np.cumsum(counts)
    start = 0
    while start < len(counts):
        taken = int(running[start - 1]) if start else 0
        stop = max(int(np.searchsorted(running, taken + work_limit, side="right")), start + 1)
        yield start, stop
        start = stop


def _pad(array: np.ndarray, pad: int) -> np.ndarray:
    """Repeat the last entry ``pad`` times, so a traced call sees a shape it has compiled."""
    return array if pad == 0 else np.concatenate([array, np.repeat(array[-1:], pad, axis=0)])


def _resolution(extent, area: float, n_triangles: int, resolution) -> np.ndarray:
    """Voxels per axis: what the caller asked for, or near-cubic voxels of the target size.

    ⚠️ **The size comes from the triangles' AREA, not from the box's volume.** Blocking
    triangles tile a surface, so the voxels they occupy are the ones their sheet passes
    through: about ``area / size**2`` of them, however large the box around it. Dividing the
    *volume* into one voxel per ten triangles assumes the box is filled, and a thin shell in a
    long box is the opposite of that -- on a reactor's wall (53,500 triangles over ~0.3 m^2 in
    a 0.9 m box) it sized 5,350 voxels of which **298** were occupied, holding 217 triangles
    each against the ten intended, and a walk through those cost more memory than testing every
    triangle would have.
    """
    if resolution is not None:
        counts = np.broadcast_to(np.asarray(resolution, dtype=int), (3,)).copy()
        if np.any(counts < 1):
            msg = f"a grid resolution must be at least 1 voxel per axis; got {tuple(counts)}"
            raise ValueError(msg)
        return counts
    wanted = max(n_triangles / _TARGET_PER_VOXEL, 1.0)
    size = float(np.sqrt(max(area, np.finfo(float).tiny) / wanted))
    counts = np.maximum(np.round(extent / max(size, np.finfo(float).tiny)), 1).astype(int)
    while np.prod(counts, dtype=float) > _MAX_VOXELS:
        counts = np.maximum(counts // 2, 1)
    return counts


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
    answered without walking.
    """
    high = low + spacing * resolution
    with np.errstate(divide="ignore", invalid="ignore"):
        to_low = (low - origin) / direction
        to_high = (high - origin) / direction
    near = np.where(direction == 0.0, -np.inf, np.minimum(to_low, to_high))
    far = np.where(direction == 0.0, np.inf, np.maximum(to_low, to_high))
    inside = np.all((origin >= low) & (origin <= high), axis=1)
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
