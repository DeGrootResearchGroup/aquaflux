"""The region's geometry, asked in compiled batches of a few fixed sizes.

The refinement asks the region the same two questions -- how far is a point from its boundary, and
where does a line cross it -- of a different number of points on every pass. A solid of
:mod:`aquaflux.solids` answers in array operations, which run eagerly one operation at a time and are
compiled anew for every new array shape; at a few hundred operations per question and a new shape on
every pass, compiling dominates. So each question is compiled once per batch size, and every request
is cut into batches padded up to a power of two between :data:`SMALLEST_BATCH` and
:data:`LARGEST_BATCH`: a handful of compilations for the whole refinement.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

#: The smallest batch compiled, so small requests share one compilation.
SMALLEST_BATCH = 1 << 10
#: The largest batch compiled; a longer request is cut into batches this size.
LARGEST_BATCH = 1 << 16


@jax.jit
def _signed_distance(region, points):
    return region.signed_distance(points)


@jax.jit
def _intervals(region, origin, direction):
    return region.intervals(origin, direction)


def _batched(function, region, *arrays) -> tuple[np.ndarray, ...]:
    """``function(region, *arrays)`` over the leading axis, in padded batches of compiled sizes."""
    n = len(arrays[0])
    if n == 0:
        out = function(region, *(jnp.zeros((1, *a.shape[1:])) for a in arrays))
        out = out if isinstance(out, tuple) else (out,)
        return tuple(np.zeros((0, *np.shape(o)[1:])) for o in out)
    size = min(LARGEST_BATCH, max(SMALLEST_BATCH, 1 << int(np.ceil(np.log2(n)))))
    pieces = []
    for start in range(0, n, size):
        chunk = [np.asarray(a[start : start + size], dtype=float) for a in arrays]
        short = size - len(chunk[0])
        # Pad by repeating the first row, so the padding asks a question with a well-defined answer.
        padded = [
            np.concatenate([c, np.repeat(c[:1], short, axis=0)]) if short else c for c in chunk
        ]
        out = function(region, *(jnp.asarray(p) for p in padded))
        out = out if isinstance(out, tuple) else (out,)
        pieces.append(tuple(np.asarray(o)[: size - short] for o in out))
    return tuple(np.concatenate(parts) for parts in zip(*pieces, strict=True))


def signed_distance(region, points) -> np.ndarray:
    """The region's signed distance at ``points`` (``(n, 3)``): negative inside, ``(n,)``."""
    (distance,) = _batched(_signed_distance, region, np.asarray(points, dtype=float))
    return distance


def intervals(region, origin, direction) -> tuple[np.ndarray, np.ndarray]:
    """Where each line ``origin + t direction`` (``(n, 3)`` each) is inside the region.

    Returns ``enter, exit``, each ``(n, k)``, as :meth:`aquaflux.solids.Solid.intervals` does.
    """
    enter, exit_ = _batched(_intervals, region, np.asarray(origin), np.asarray(direction))
    return enter, exit_
