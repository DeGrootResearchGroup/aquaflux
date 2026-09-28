"""How much one pass over receivers may form, stated in the unit that actually costs memory.

Every loop in this package that visits receivers — the gather, the shadow-mask build, the ray test,
the winding-number check — forms arrays of one entry per **receiver-by-facet pair**, so a pass's
memory is its receivers times the emitter's facets. Bounding a pass by a number of *receivers*
therefore leaves its size to a number the caller did not choose: the same receiver count that is
a few hundred megabytes against a coarse lamp is gigabytes against a finely refined one, and
refining the emitter is exactly what a user does to make an answer more accurate. The failure
then is not an error but an out-of-memory kill, with no traceback naming the setting.

So a pass is bounded by pairs, and the receiver count follows from it and from the facet count
here, in one place. :func:`in_passes` is the traced loop itself, shared by the volume gather and
by the facet-to-facet walk through a graded medium, whose "receivers" are the receiving facets.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

__all__ = ["DEFAULT_PAIR_LIMIT", "in_passes", "receivers_per_pass", "receivers_per_step"]

#: Receiver-by-facet pairs one pass may form, when a caller does not say.
DEFAULT_PAIR_LIMIT = 4_000_000

#: Receiver-by-facet pairs one step of a traced loop forms, however high the pair limit: the
#: traced loop's own bound, set by speed rather than by memory. See :func:`receivers_per_step`.
PASS_PAIRS = 1 << 16


def receivers_per_pass(pair_limit: int, per_receiver: int) -> int:
    """How many receivers a pass takes so that it forms at most ``pair_limit`` pairs.

    Parameters
    ----------
    pair_limit : int
        The pairs one pass may form. Must be at least one.
    per_receiver : int
        The pairs each receiver brings — the emitter's facet count, usually.

    Returns
    -------
    int
        At least one, so a single receiver against more facets than the limit still gets a pass
        of its own rather than none.

    Raises
    ------
    ValueError
        If ``pair_limit`` is less than one.
    """
    if pair_limit < 1:
        msg = f"pair_limit must be at least 1 receiver-by-facet pair; got {pair_limit}"
        raise ValueError(msg)
    return max(1, pair_limit // max(1, per_receiver))


def receivers_per_step(pair_limit: int, per_receiver: int) -> int:
    """How many receivers one compiled step takes: the pair limit's count, and at most
    :data:`PASS_PAIRS` pairs' worth however high the limit is.

    A step's intermediates are arrays of one entry per pair, and the compiled body writes each
    out and reads it back. While they fit in a core's cache that traffic is cheap; past it the
    same pairs cost more -- about twice as long in the volume gather, a fifth more in the transfer
    build. So the limit bounds how much a step *may* form, for memory, and this bounds how much it
    *does*. The count is a cut of the work, not of the answer: each receiver's row is formed from
    its own inputs either way.

    Parameters
    ----------
    pair_limit : int
        The pairs one pass may form, for memory. Must be at least one.
    per_receiver : int
        The pairs each receiver brings.

    Returns
    -------
    int
        At least one.
    """
    return receivers_per_pass(min(pair_limit, PASS_PAIRS), per_receiver)


def in_passes(arrays, pair_limit: int, per_receiver: int, body):
    """Apply ``body`` to the receivers in fixed-size chunks and join the results along axis 0.

    Each entry of ``arrays`` is an ``(array, axis)`` pair: the array is cut along ``axis``, its
    receiver axis, the same way as every other, so a per-receiver quantity computed outside — a
    layer of the visibility mask, a receiving surface's normal — stays lined up with its point
    without the body having to index anything itself. The first array's receiver count is the
    count, and ``body`` returns an array whose leading axis is the chunk's receivers — one value
    each, or a row each.

    The receiver-by-source product is the whole cost of such a loop and would be the whole of its
    memory too if it were formed at once: a hundred thousand cells against a thousand facets is
    a hundred million entries per intermediate. A chunk takes as many receivers as keep it within
    ``pair_limit`` pairs (and within :data:`PASS_PAIRS`, below), so the working set is set by the
    limit and not by how finely the emitter happens to be divided.

    **Chunks are sliced out of the arrays where they lie**, not cut from a padded copy: a shadow
    mask is the size of the whole problem, and padding it to a whole number of chunks would copy
    it. The full chunks run as one scan, compiled once; a shorter remainder, if there is one, runs
    after it as a second scan of one step, so it is compiled too rather than run one operation at
    a time.

    ⚠️ **The body is checkpointed, and that is what makes the limit hold for a gradient too.** A
    scan's reverse pass otherwise keeps every chunk's intermediates for the backward sweep, so a
    gradient's memory grows with the receiver count -- about 25 bytes a pair -- whatever the
    limit says, and at a mesh's cells against a finely divided lamp that is terabytes.
    Checkpointed, each chunk is recomputed on the way back instead, for roughly two thirds more
    time on the gradient and nothing on a forward evaluation, whose values it does not change.

    ⚠️ **A step forms at most :data:`PASS_PAIRS` pairs, even under a higher ``pair_limit``**
    (:func:`receivers_per_step`).
    """
    arrays = [(jnp.asarray(array), axis) for array, axis in arrays]
    first, axis = arrays[0]
    n_points = first.shape[axis]
    if n_points == 0:
        return jnp.zeros(0)
    per_chunk = min(receivers_per_step(pair_limit, per_receiver), n_points)
    n_full, remainder = divmod(n_points, per_chunk)

    # The slicing is inside the checkpoint, so what a gradient keeps per chunk is the chunk's
    # starting index and not the chunk it cut -- which, summed over the chunks, is every array
    # handed in. prevent_cse=False is the setting for a checkpoint inside a scan, which already
    # stops the recomputation from being merged back into the forward pass.
    def run(size):
        return jax.checkpoint(
            lambda start: body(*[_slice(array, axis, start, size) for array, axis in arrays]),
            prevent_cse=False,
        )

    def scanned(size, count, first):
        chunk = run(size)
        _, out = lax.scan(
            lambda carry, index: (carry, chunk(first + index * size)), None, jnp.arange(count)
        )
        return out.reshape(-1, *out.shape[2:])

    pieces = []
    if n_full:
        pieces.append(scanned(per_chunk, n_full, 0))
    if remainder:
        # A scan of one step rather than a bare call: a scan is compiled as a whole even when
        # nothing around it is, whereas a bare call runs the body one operation at a time -- and
        # a remainder can be nearly a full chunk.
        pieces.append(scanned(remainder, 1, n_full * per_chunk))
    return pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces)


def _slice(array, axis: int, start, size: int):
    """``size`` receivers of ``array`` along its receiver axis, from ``start``.

    ⚠️ **A gather of those rows, not a dynamic slice, and with the indices clipped** -- the same
    rows either way, bit for bit. On the CPU backend a chunk cut by ``lax.dynamic_slice`` made
    the whole gather about a fifth slower, and ``jnp.take``'s default out-of-bounds mode, which
    fills with a placeholder and so tests every index, cost about as much again; a clipped
    gather costs neither. The rows are always in bounds here, so clipping never changes one.
    """
    return jnp.take(array, start + jnp.arange(size), axis=axis, mode="clip")
