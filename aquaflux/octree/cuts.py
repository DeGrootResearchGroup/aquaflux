"""Where a cell's edges cross the boundary of the region a field lives in.

A field sampled on an octree's nodes knows nothing between the last node in the region and the
region's boundary, and near a boundary that gap is where the field changes fastest: every node on
the inside of a cell that the boundary cuts is farther from the boundary than the points between
them. So each edge the boundary crosses gets a **cut point** of its own, on the edge, and the field
is sampled there too.

The region is described by a solid in the sense of :mod:`aquaflux.solids` -- anything whose inside,
along a line, is a union of intervals it can report, and which reports its signed distance. The
intervals find where an edge crosses the boundary exactly; the cut point is then placed on the edge,
between the crossing and the middle of the region the edge holds there, where the region's signed
distance is minus the **standoff**: a standoff from the boundary, measured along its normal, whatever
angle the edge meets it at. Kept on the edge, a cut point lies on a line of the lattice and is the
same point for every cell with that edge, at whatever level.

Nothing nearer the boundary than the standoff is sampled. Where the region an edge holds is never a
standoff deep -- an edge grazing the boundary, or running along it -- there is no cut point.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from aquaflux.ragged import group
from aquaflux.ragged import rows as rows_of

from .region import intervals, signed_distance
from .tree import EDGES, Octree


def merged_intervals(enter: np.ndarray, exit_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The union of each row's intervals, as disjoint intervals sorted along the row.

    Parameters
    ----------
    enter, exit_ : np.ndarray, shape ``(n, k)``
        Each row's intervals, unsorted and possibly overlapping; an interval with ``enter > exit``
        is empty.

    Returns
    -------
    enter, exit : np.ndarray, shape ``(n, k)``
        The merged intervals, leftmost first, padded with empty intervals ``(+inf, -inf)``.
    """
    enter = np.asarray(enter, dtype=float)
    exit_ = np.asarray(exit_, dtype=float)
    empty = ~(enter <= exit_)
    enter = np.where(empty, np.inf, enter)
    exit_ = np.where(empty, -np.inf, exit_)
    order = np.argsort(enter, axis=1)
    enter = np.take_along_axis(enter, order, axis=1)
    exit_ = np.take_along_axis(exit_, order, axis=1)
    reach = np.maximum.accumulate(exit_, axis=1)
    before = np.concatenate([np.full((len(enter), 1), -np.inf), reach[:, :-1]], axis=1)
    real = np.isfinite(enter)
    starts = real & (enter > before)
    # A merged interval ends at the reach of its last member: the member before the next start.
    after = np.concatenate([starts[:, 1:], np.ones((len(enter), 1), dtype=bool)], axis=1)
    ends = real & (after | ~np.concatenate([real[:, 1:], np.zeros((len(enter), 1), bool)], axis=1))
    # Compact both lists to the left, preserving order.
    rank_start = np.cumsum(starts, axis=1) - 1
    rank_end = np.cumsum(ends, axis=1) - 1
    out_enter = np.full(enter.shape, np.inf)
    out_exit = np.full(enter.shape, -np.inf)
    rows = np.broadcast_to(np.arange(len(enter))[:, None], enter.shape)
    out_enter[rows[starts], rank_start[starts]] = enter[starts]
    out_exit[rows[ends], rank_end[ends]] = reach[ends]
    return out_enter, out_exit


#: How near a segment's end, as a fraction of its length, a crossing is taken to be at that end -- so
#: an edge ending on a boundary that lies along a lattice plane is cut all the same.
_AT_THE_END = 1e-9

#: Halvings of the bracket in the search for a cut point: down to ``2**-40`` of the region the edge
#: holds, far below any standoff.
_BISECTIONS = 40


def _at_depth(region, start, direction, near, far, standoff) -> tuple[np.ndarray, np.ndarray]:
    """Where along each segment, between ``near`` (a crossing) and ``far`` (inside the region), the
    region's signed distance is ``-standoff``; and whether ``far`` is that deep at all.

    Bisection on the signed distance, which keeps ``far`` at least a standoff deep throughout, so the
    point returned is never nearer the boundary than the standoff.
    """

    def deep(t):
        return signed_distance(region, start + t[:, None] * direction) <= -standoff

    found = deep(far)
    near, far = near.copy(), far.copy()
    for _ in range(_BISECTIONS):
        middle = 0.5 * (near + far)
        below = deep(middle)
        far = np.where(below, middle, far)
        near = np.where(below, near, middle)
    return far, found


def _cut_parameters(region, start, direction, enter, exit_, standoff, ends=(True, True)):
    """The cut points of merged intervals along segments, as ``(segment, t)``.

    Each interval the segment ``[0, 1]`` cuts gets a point at each end it cuts (of those ``ends`` asks
    for: the low end, the high end) where the region is a standoff deep, searched between the
    crossing and the interval's middle; none where the middle is not that deep.
    """
    held = (exit_ > 0.0) & (enter < 1.0) & (exit_ > enter)
    low = np.where(held, np.maximum(enter, 0.0), 0.0)
    high = np.where(held, np.minimum(exit_, 1.0), 0.0)
    middle = 0.5 * (low + high)
    rows, ts = [], []
    for wanted, cut, crossing in (
        (ends[0], held & (enter > -_AT_THE_END), low),
        (ends[1], held & (exit_ < 1.0 + _AT_THE_END), high),
    ):
        if not wanted:
            continue
        row, column = np.nonzero(cut)
        if len(row) == 0:
            continue
        t, found = _at_depth(region, start[row], direction[row], crossing[row, column],
                             middle[row, column], standoff)  # fmt: skip
        rows.append(row[found])
        ts.append(t[found])
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    return np.concatenate(rows), np.concatenate(ts)


def _merged_along(region, start: np.ndarray, end: np.ndarray):
    direction = end - start
    enter, exit_ = merged_intervals(*intervals(region, start, direction))
    return direction, enter, exit_


def crossing_points(
    region, start: np.ndarray, end: np.ndarray, standoff: float
) -> tuple[np.ndarray, np.ndarray]:
    """The cut points of segments: one inside each end of the region on them that the segment cuts.

    Parameters
    ----------
    region : aquaflux.solids.Solid
        Anything with ``intervals(origin, direction)`` reporting where a line is inside it.
    start, end : np.ndarray, shape ``(n, 3)``
        The segments.
    standoff : float
        How far from the boundary, by the region's signed distance, each point is placed.

    Returns
    -------
    points : np.ndarray, shape ``(m, 3)``
    segment : np.ndarray of int, shape ``(m,)``
        The segment each point lies on.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if len(start) == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    direction, enter, exit_ = _merged_along(region, start, end)
    segment, t = _cut_parameters(region, start, direction, enter, exit_, standoff)
    return start[segment] + t[:, None] * direction[segment], segment


def exit_points(
    region, inside: np.ndarray, outside: np.ndarray, standoff: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For segments from a point in the region to one outside it, the cut point where each leaves.

    The same point :func:`crossing_points` places at that end of the interval holding the start -- a
    standoff from the boundary, between the exit and the interval's middle -- so the two find one
    point.

    Parameters
    ----------
    region : aquaflux.solids.Solid
    inside, outside : np.ndarray, shape ``(n, 3)``
        Each segment's start, in the region, and end.
    standoff : float

    Returns
    -------
    points : np.ndarray, shape ``(n, 3)``
    t : np.ndarray, shape ``(n,)``
        How far along its segment each point is, as a fraction of the segment.
    found : np.ndarray of bool, shape ``(n,)``
        Which segments have such a point: not those whose region from the start is nowhere a standoff
        deep, nor any that never leaves the region.
    """
    inside = np.asarray(inside, dtype=float)
    outside = np.asarray(outside, dtype=float)
    if len(inside) == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=bool)
    direction, enter, exit_ = _merged_along(region, inside, outside)
    # Only the interval holding the start, and only its far end.
    holding = (enter <= 0.0) & (exit_ >= 0.0)
    enter = np.where(holding, enter, np.inf)
    exit_ = np.where(holding, exit_, -np.inf)
    segment, where = _cut_parameters(
        region, inside, direction, enter, exit_, standoff, (False, True)
    )
    t = np.zeros(len(inside))
    found = np.zeros(len(inside), dtype=bool)
    t[segment] = where
    found[segment] = True
    points = inside + t[:, None] * direction
    return points, t, found


@dataclasses.dataclass(frozen=True)
class LeafCuts:
    """The cut points on some leaves' edges.

    Attributes
    ----------
    points : np.ndarray, shape ``(m, 3)``
        Distinct cut points, in space.
    offsets, items : np.ndarray of int
        Compressed-sparse-row (CSR) form of the points on each leaf: leaf ``r`` (in the order the
        leaves were given) has the points ``items[offsets[r]:offsets[r + 1]]``.
    """

    points: np.ndarray
    offsets: np.ndarray
    items: np.ndarray


def leaf_cuts(tree: Octree, rows: np.ndarray, region, standoff: float) -> LeafCuts:
    """The cut points on the edges of some of a tree's leaves.

    Edges shared by several of the leaves are cut once; the same crossing reached along a longer and a
    shorter edge of one lattice line comes out at the same point to rounding, and is kept once.

    Parameters
    ----------
    tree : Octree
    rows : np.ndarray of int
        The leaves.
    region : aquaflux.solids.Solid
        The region the field lives in (see :func:`crossing_points`).
    standoff : float
        See :func:`crossing_points`.

    Returns
    -------
    LeafCuts
    """
    rows = np.asarray(rows, dtype=np.int64)
    corners = tree.corners(rows)  # (r, 8, 3)
    first, second = corners[:, EDGES[:, 0], :], corners[:, EDGES[:, 1], :]  # (r, 12, 3)
    edge_codes = np.stack([tree.key(first), tree.key(second)], axis=-1).reshape(-1, 2)
    unique, edge_of = np.unique(edge_codes, axis=0, return_inverse=True)
    edge_of = edge_of.reshape(-1)
    ends = tree.unkey(unique)  # (e, 2, 3)
    points, on_edge = crossing_points(
        region, tree.position(ends[:, 0]), tree.position(ends[:, 1]), standoff
    )
    # The same point reached along different edges of one line: merge to a tolerance far below the
    # standoff, far above rounding.
    quantum = 1e-9 * tree.step
    rounded = np.round((points - tree.origin) / quantum).astype(np.int64)
    distinct, point_of = np.unique(rounded, axis=0, return_index=False, return_inverse=True)
    point_of = point_of.reshape(-1)
    first_seen = np.full(len(distinct), -1, dtype=np.int64)
    first_seen[point_of[::-1]] = np.arange(len(point_of))[::-1]
    distinct_points = points[first_seen]
    # Leaf -> its edges -> their points.
    edge_offsets, by_edge = group(on_edge, len(unique))
    found, slot, _ = rows_of(edge_offsets, point_of[by_edge], edge_of)
    pairs = np.unique(np.stack([slot // len(EDGES), found], axis=1), axis=0)
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(np.bincount(pairs[:, 0], minlength=len(rows)), out=offsets[1:])
    return LeafCuts(points=distinct_points, offsets=offsets, items=pairs[:, 1])
