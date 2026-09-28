"""Points ordered along a Morton (Z-order) curve, so a run of consecutive points is a local patch.

Several loops over points want them grouped into small neighbourhoods -- to bound a group once
rather than each member, or to cut a pass into chunks that each cover a small region -- and a
sort along a space-filling curve gives every such grouping at once: cut the order into runs of
any length and each run lies close together. :func:`morton_order` is that sort.

⚠️ **Each axis of the bounding box is scaled to its own extent, so the curve's cells have the
box's proportions, not a cube's -- deliberately.** On a long thin surface -- a lamp far longer than
it is wide -- cells that follow the box are long along it and narrow across it, so a run of
consecutive facets covers a narrow slice *around* the surface and a longer stretch *along* it: its
facets face nearly the same way. Cubic cells instead make a run a compact patch that wraps further
around, and its facets face many ways. Compactness is what a bounding-volume test wants; a shared
facing direction is what a test for "every facet of this group lies behind these points" wants,
and on such a lamp the second is worth more. Measured both ways on a whole reactor field, cubic
cells left more pairs undecided and made the facet-to-facet transfer build slower, with no gain
anywhere.

Build-time numpy only: nothing here is traced or differentiated. Imports nothing from the package,
so any layer may use it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["morton_order"]

#: Levels per axis of the grid a point's place on the curve is read from, as bits. Ten: finer
#: than any group is worth ordering within, and a key of thirty bits. Points sharing a cell keep
#: their input order.
_BITS = 10


def _spread(values: np.ndarray) -> np.ndarray:
    """Each value's low :data:`_BITS` bits, moved to every third bit position."""
    values = values.astype(np.uint64)
    spread = np.zeros_like(values)
    for bit in range(_BITS):
        spread |= ((values >> np.uint64(bit)) & np.uint64(1)) << np.uint64(3 * bit)
    return spread


def morton_order(points) -> np.ndarray:
    """The indices of ``points`` in their order along a Morton (Z-order) curve.

    Each axis of the points' bounding box is divided into ``2**10`` levels, each point's three
    level numbers are interleaved bit by bit into one key, x in the lowest bit, and the points are
    sorted by key. An axis with no extent puts every point at its first level. Ties keep their
    input order, so the ordering is reproducible.

    Parameters
    ----------
    points : array_like, shape ``(n, 3)``

    Returns
    -------
    np.ndarray of int, shape ``(n,)``
        The indices of ``points`` in curve order.
    """
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    low = points.min(axis=0)
    extent = points.max(axis=0) - low
    levels = (1 << _BITS) - 1
    scaled = np.where(extent > 0.0, (points - low) / np.where(extent > 0.0, extent, 1.0), 0.0)
    cell = np.minimum((scaled * levels).astype(np.int64), levels)
    key = _spread(cell[:, 0]) | (_spread(cell[:, 1]) << 1) | (_spread(cell[:, 2]) << 2)
    return np.argsort(key, kind="stable")
