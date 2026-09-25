"""Whether a segment is cut by any triangle of a surface.

This is the test that lets a reactor shadow itself: a facet on one leg of a bent duct does not
illuminate a cell in the other leg, because the wall between them is in the way. The analytic
bodies of :mod:`aquaflux.solids` cannot express that — the geometry doing the
blocking *is* the emitting surface.

The intersection is Woop, Benthin and Wald's **watertight** ray-triangle test (*Journal of
Computer Graphics Techniques* 2(1), 2013) rather than the more familiar Möller-Trumbore form.
The two agree almost everywhere; they differ on rays that pass through an edge shared by two
triangles, where an ordinary test can report a hit on both or on neither depending on rounding.
Neither is right, and "neither" is the one that matters: it is a pinhole through a closed
surface. Woop's construction transforms the ray to the ``+z`` axis and evaluates the three edge
functions in a way that makes the two triangles sharing an edge give exactly opposite values, so
one of them always claims the ray.

⚠️ **A facet must not shadow itself**, and the exclusion is by index rather than by tolerance:
the ray starts at the facet's own centroid, so the facet is always hit, at zero distance.
Excluding its whole *solid* would be wrong — a non-convex body such as a bent duct is precisely
the case this module exists for, and there the blocking wall belongs to the same body as the
emitter. Neighbouring facets that share an edge with the source are handled by the same
near-origin exclusion the analytic bodies use.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["segment_is_cut"]


def _edge_function(first, second, third, fourth):
    """``first * second - third * fourth``, with a value that does not depend on which pair leads.

    This is the 2D cross product Woop's inside test is built from, and the watertight guarantee
    rests on one property of it: two triangles sharing an edge evaluate it with the two operand
    pairs swapped, and the results must be **exact negatives**, so that a ray passing through the
    shared edge is claimed by exactly one of them. Written as the plain difference, that holds
    only while each product is rounded before the subtraction — and a compiler is free to fuse a
    multiply and an add into one instruction that rounds once, which keeps one product exact and
    rounds the other. The two triangles then keep *different* products exact and stop being
    negatives, and the ray escapes through the pinhole between them.

    ⚠️ **This is not hypothetical and it is not visible in the source.** Evaluated one operation
    at a time the plain difference is antisymmetric everywhere; compiled, it is not, on a third of
    random operand quadruples — and that is enough to reopen every leak the watertight form exists
    to close (six of 268 rays aimed at the vertices and edge midpoints of a closed hull, which is
    exactly what the older Möller-Trumbore test leaks on the same fixture).

    Averaging the expression with the negation of its own swap restores it unconditionally.
    Whatever the compiler does to ``a - b``, it does the same to ``b - a`` up to an exact sign,
    because subtraction is antisymmetric in IEEE arithmetic however its operands were formed. The
    result is both exactly antisymmetric and identical to the one-operation-at-a-time value, so
    it costs two multiplies and a subtraction per edge and changes no answer.
    """
    forward = first * second - third * fourth
    return 0.5 * (forward - (third * fourth - first * second))


def _watertight_hit(origin, direction, vertices):
    """Woop's edge test, for one ray against many triangles.

    ``origin`` and ``direction`` broadcast as ``(..., 3)``; ``vertices`` is ``(..., 3, 3)``.
    Returns the hit distance along ``direction`` in units of its length, and whether the ray's
    line meets the triangle at all — the caller decides which distances count.
    """
    # Put the ray's largest component on the z axis, so the shear below never divides by a
    # small number.
    kz = jnp.argmax(jnp.abs(direction), axis=-1)
    kx = (kz + 1) % 3
    ky = (kz + 2) % 3
    # The published algorithm also swaps kx and ky when the chosen component is negative, to
    # keep the coordinate system right-handed. That is deliberately omitted: it is inert here.
    # Flipping handedness negates u, v, w and the determinant together, and both places they
    # are used are invariant to that — the inside test accepts all-non-negative *or*
    # all-non-positive, and the distance divides by the same determinant. Measured over 20000
    # rays against 300 triangles, adding the swap changes no hit and moves no distance by more
    # than 0.0. The swap is needed when the determinant's sign is used for back-face culling,
    # which this module does with the facet normal instead.

    def component(array, axis):
        return jnp.take_along_axis(array, axis[..., None], axis=-1)[..., 0]

    dx, dy, dz = (component(direction, axis) for axis in (kx, ky, kz))
    shear_x, shear_y, shear_z = dx / dz, dy / dz, 1.0 / dz

    relative = vertices - origin[..., None, :]
    corners = []
    for corner in range(3):
        offset = relative[..., corner, :]
        ox = component(offset, kx)
        oy = component(offset, ky)
        oz = component(offset, kz)
        corners.append((ox - shear_x * oz, oy - shear_y * oz, oz * shear_z))
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = corners

    u = _edge_function(cx, by, cy, bx)
    v = _edge_function(ax, cy, ay, cx)
    w = _edge_function(bx, ay, by, ax)

    determinant = u + v + w
    same_side = ((u >= 0.0) & (v >= 0.0) & (w >= 0.0)) | ((u <= 0.0) & (v <= 0.0) & (w <= 0.0))
    edge_on = determinant == 0.0
    safe = jnp.where(edge_on, 1.0, determinant)
    distance = (u * az + v * bz + w * cz) / safe
    return distance, same_side & ~edge_on


def padded_length(count: int) -> int:
    """The padded length a chunk of ``count`` items is clipped at: the next power of two.

    Trades a handful of compiled programs for not clipping thousands of padding entries. Both
    ends of that trade are real -- one shape for everything makes a small body absurdly slow,
    and one shape per chunk makes any body absurdly slow. Shared by everything here that hands
    a variable number of items to a traced kernel.
    """
    return 1 << max(0, int(count - 1)).bit_length() if count else 0


def _counts_as_hit(distance, meets, near, index, exclude):
    """Which of the hits a ray-triangle test found actually block the segment.

    The geometry is in :func:`_watertight_hit`; this is the rest of the rule, shared by the
    block kernel and the grid-culled one so the two cannot drift: a hit counts when it lies
    **past** the origin's margin, **at or before** the far end (a segment ending exactly in a
    facet's plane is blocked by it, which is why the far end is inclusive), and on a triangle
    the ray was not told to ignore. Shapes broadcast, so ``index`` may be a block's column of
    triangle indices or one index per pair.
    """
    hit = meets & (distance > near) & (distance <= 1.0)
    return hit & jnp.all(index[..., None] != exclude, axis=-1)


@jax.jit
def _pair_is_cut(origin, direction, near, vertices, index, exclude):
    """Whether each ray meets the ONE triangle paired with it.

    The grid-culled path tests a compacted list of (ray, triangle) pairs rather than every ray
    against every triangle of a block, because with a grid each ray has its own candidates. The
    predicate is the same one: :func:`_watertight_hit` for the geometry and
    :func:`_counts_as_hit` for the window and the exclusions.
    """
    distance, meets = _watertight_hit(origin, direction, vertices)
    return _counts_as_hit(distance, meets, near, index, exclude)


@jax.jit
def _block_is_cut(origin, direction, near, block, first, exclude):
    """Whether each ray meets any triangle of one block, as a single traced kernel.

    ``first`` is the index the block starts at, so the exclusion can be tested against global
    triangle indices without the caller rebuilding them. Everything is shaped
    ``(n_rays, n_block)`` inside and reduced away before it is returned.

    ⚠️ **Tracing this is not an optimization detail, it is most of the performance of the
    visibility build.** Evaluated eagerly, every line here materializes a full
    ``n_rays x n_block`` array, so the test is bound by the memory of those intermediates
    rather than by the arithmetic. Traced, the compiler fuses the edge test, the distance
    window and the exclusion straight into the reduction and never forms one: on an
    eleven-core machine with 19 GB, double precision, 100000 rays against 200 triangles, the
    same computation runs at **332-432** million ray-by-triangle tests per second traced
    against **50-58** eager, for identical output on every ray.
    """
    distance, meets = _watertight_hit(origin[:, None, :], direction[:, None, :], block[None, ...])
    index = first + jnp.arange(block.shape[0])
    counts = _counts_as_hit(distance, meets, near[:, None], index[None, :], exclude[:, None, :])
    return jnp.any(counts, axis=-1)


def _call_shape(n_rays: int, n_triangles: int, work_limit: int) -> tuple[int, int]:
    """How many rays and how many triangles one compiled call covers: ``(ray_chunk, block)``.

    **Triangles take the budget first, and rays take what is left.** The product never exceeds
    ``work_limit`` (unless one axis alone is larger than it, when that axis is split to single
    items), so the working set of a call stays bounded either way. What the order decides is how
    often each ray's data is reused: a call streams its rays' origins, directions, margins and
    exclusions once and tests each against every triangle of the block, so the block is the
    reuse factor.

    ⚠️ **Rays first was the old order, and it made the pass slower the more rays it had.** The
    ray count is receivers times facets, so it reaches the millions and took the whole budget,
    leaving a block of **one** triangle: each call then streamed millions of rays to test a single
    triangle. Measured with the identical kernel, 1532 triangles, 3.2 million rays, double
    precision, eleven cores: 120.5 million tests per second rays-first against 465.4 with the
    whole triangle set per call, bit-identical output -- and at 200,000 rays, 350.5 against
    464.1. Triangles-first makes the throughput flat in the ray count, which is what a pass whose
    cost is its tests should be.
    """
    block = max(1, min(n_triangles, work_limit))
    ray_chunk = max(1, min(n_rays, work_limit // block))
    return ray_chunk, block


def segment_is_cut(origin, target, vertices, min_distance, *, exclude=None, work_limit=4_000_000):
    """Whether any triangle lies across the segment from ``origin`` to ``target``.

    Parameters
    ----------
    origin, target : jnp.ndarray, shape ``(n_rays, 3)``
        Segment endpoints. By convention ``origin`` is the source.
    vertices : jnp.ndarray, shape ``(n_triangles, 3, 3)``
        The blocking triangles.
    min_distance : jnp.ndarray, shape ``(n_rays,)``
        How far from ``origin`` a hit must be before it counts, in length units.
    exclude : jnp.ndarray of int, shape ``(n_rays,)`` or ``(n_rays, k)``, optional
        Triangles each ray must ignore. Pass ``-1`` in a slot to exclude nothing there.

        There are two of them whenever a ray runs between two facets, and leaving either out
        blocks the pair outright rather than approximately:

        - the **source** facet, which every ray leaving its centroid hits at zero distance —
          guarded from the origin end by ``min_distance`` as well;
        - the **target** facet, when the ray is aimed at a point lying on one. There is no
          margin at the far end to lean on: the segment ends exactly in that facet's plane, and
          a hit at ``distance == 1`` counts.
    work_limit : int, optional
        How many ray-by-triangle entries one block may cover. It bounds the working set of a
        single pass, and the default keeps one pass near 32 MB of entries.

        It used to be the one knob that mattered, because each pass materialized that many
        entries and throughput fell off a cliff when they stopped fitting — 42-57 Mtest/s from
        0.5 MB through 134 MB of intermediate, against **2.4** at 537 MB, twenty times slower
        (2048 triangles, double precision, an eleven-core machine with 19 GB). Now that the
        block is a traced kernel the compiler fuses the intermediate away, so the cliff is not
        reached. **The bound is kept as a bound**, not as a tuning knob: it is what guarantees a
        working set whatever the compiler decides to do with a given shape.

        Both axes are cut to honour it, **triangles first**: each call takes as many triangles
        as the bound allows -- the whole set, for any mesh this pass can afford -- and as many
        rays as then fit. The order matters because the triangle block is how often each ray's
        data is reused, and cutting the rays first left a block of one triangle at the ray
        counts a transfer build reaches, which made the pass about four times slower there.

        Each distinct block shape is compiled once, so a ray count or triangle count that does
        not divide evenly costs one extra compilation for its remainder — a few small programs,
        not one per block.

    Returns
    -------
    jnp.ndarray of bool, shape ``(n_rays,)``
    """
    origin = jnp.asarray(origin, dtype=float)
    direction = jnp.asarray(target, dtype=float) - origin
    length = jnp.sqrt(jnp.sum(direction * direction, axis=-1))
    near = jnp.asarray(min_distance) / jnp.where(length == 0.0, 1.0, length)

    vertices = jnp.asarray(vertices, dtype=float)
    if exclude is None:
        # One sentinel row standing for every ray, rather than a second kernel without the
        # exclusion in it: -1 is already this function's "exclude nothing here" slot, and
        # traced, the comparison it costs does not show up against the edge test.
        exclude = jnp.full((1, 1), -1)
    else:
        exclude = jnp.asarray(exclude)
        exclude = exclude[:, None] if exclude.ndim == 1 else exclude
    n_rays, n_triangles = origin.shape[0], vertices.shape[0]
    if n_rays == 0 or n_triangles == 0:
        return jnp.zeros(n_rays, dtype=bool)

    ray_chunk, block_size = _call_shape(n_rays, n_triangles, work_limit)
    full_chunk, _ = _call_shape(work_limit, n_triangles, work_limit)

    pieces = []
    for first in range(0, n_rays, ray_chunk):
        rays = slice(first, first + ray_chunk)
        count = min(ray_chunk, n_rays - first)
        # A chunk shorter than the full one -- the last, or the only one when there are few rays
        # -- is padded to a power of two by repeating its last ray, so a caller whose ray counts
        # vary from pass to pass compiles a few shapes rather than one per pass. The padding's
        # answers are dropped.
        pad = min(padded_length(count), full_chunk) - count if count < full_chunk else 0
        # The sentinel has one row for all of them and so is not sliced alongside the rays.
        excluded = exclude if exclude.shape[0] == 1 else _repeat_last(exclude[rays], pad)
        chunk_origin = _repeat_last(origin[rays], pad)
        chunk_direction = _repeat_last(direction[rays], pad)
        chunk_near = _repeat_last(near[rays], pad)
        cut = jnp.zeros(count + pad, dtype=bool)
        for start in range(0, n_triangles, block_size):
            cut = cut | _block_is_cut(
                chunk_origin,
                chunk_direction,
                chunk_near,
                vertices[start : start + block_size],
                start,
                excluded,
            )
        pieces.append(cut[:count])
    return jnp.concatenate(pieces)


def _repeat_last(array, pad: int):
    """``array`` with its last row repeated ``pad`` more times."""
    return array if pad == 0 else jnp.concatenate([array, jnp.repeat(array[-1:], pad, axis=0)])
