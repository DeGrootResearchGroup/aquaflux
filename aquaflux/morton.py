"""Points ordered along a Morton (Z-order) curve, so a run of consecutive points is a compact patch.

Several loops over points want them grouped into small neighbourhoods -- to bound a group once
rather than each member, or to cut a pass into chunks that each cover a small region -- and a
sort along a space-filling curve gives every such grouping at once: cut the order into runs of
any length and each run lies close together. :func:`morton_order` is that sort.

⚠️ **The curve's cells are cubes, sized by the bounding box's LONGEST side.** Scaling each axis to
its own extent instead makes the cells as elongated as the box, and on a long thin surface -- a
lamp far longer than it is wide -- the curve then zig-zags from end to end within a handful of
points, so that on a lamp forty times longer than it is wide a run of two consecutive facets
could span half its length. With cubic cells a run is compact whatever the box's proportions.

Build-time numpy only: nothing here is traced or differentiated. Imports nothing from the package,
so any layer may use it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["morton_order"]

#: Bits of each coordinate interleaved into a key: 3 x 21 = 63, the most a 64-bit key holds. A
#: cell is then the longest side over two million, so points share a cell only where they
#: practically coincide, and the order among those is their input order.
_BITS = 21


def _spread(values: np.ndarray) -> np.ndarray:
    """Each value's low :data:`_BITS` bits, moved to every third bit position."""
    values = values.astype(np.uint64)
    spread = np.zeros_like(values)
    for bit in range(_BITS):
        spread |= ((values >> np.uint64(bit)) & np.uint64(1)) << np.uint64(3 * bit)
    return spread


def morton_order(points) -> np.ndarray:
    """The indices of ``points`` in their order along a Morton (Z-order) curve.

    The points' bounding box is covered by a grid of equal cubes, :data:`_BITS` bits of them along
    its longest side; each point's three cell numbers are interleaved bit by bit into one key, x
    in the lowest bit, and the points are sorted by key. Ties keep their input order, so the
    ordering is reproducible.

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
    side = float((points.max(axis=0) - low).max())
    levels = (1 << _BITS) - 1
    scaled = (points - low) / side if side > 0.0 else np.zeros_like(points)
    cell = np.minimum((scaled * levels).astype(np.int64), levels)
    key = _spread(cell[:, 0]) | (_spread(cell[:, 1]) << 1) | (_spread(cell[:, 2]) << 2)
    return np.argsort(key, kind="stable")
