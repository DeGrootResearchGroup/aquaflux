"""Convex clipping, and the filter that makes its sign tests decidable.

The clip itself is a few lines of compaction; everything hard about it is that a meshed
enclosure asks it to decide signs that are exactly zero. Neighbouring triangles share vertices,
two triangles of one wall are coplanar, and a receiver sits in the plane of every triangle on
its own facet -- so the tests below are mostly about degenerate configurations, because those
are the common case and not the edge case.

Each degeneracy test names the wrong answer it catches: without the filter the quantity is
last-bit noise whose *sign* depends on how the compiler scheduled the batch, so the clip keeps
a different polygon at a different batch shape and the same geometry gives two answers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation.clipping import (
    clip_to_halfspace,
    decidable_heights,
    spanning_plane,
)
from aquaflux.vectors import dot

SQUARE = jnp.asarray([[1.0, 0.0, 3.0], [0.0, 1.0, 3.0], [-1.0, 0.0, 3.0], [0.0, -1.0, 3.0]])


def test_a_zero_clipping_plane_leaves_the_loop_alone():
    """Relied on by every repeated slot in a fixed-width loop, so it is pinned not assumed."""
    clipped = clip_to_halfspace(SQUARE, decidable_heights(SQUARE, jnp.zeros(3)), 4)
    assert np.allclose(np.asarray(clipped), np.asarray(SQUARE))


def test_an_offset_plane_is_not_the_same_as_one_through_the_receiver():
    """The bug that broke the old occlusion cull, pinned on the heights that replaced it.

    Omitting ``through`` answers for a parallel plane through the origin, which for a source's
    supporting plane rejects everything in *front* of the source rather than behind it.
    """
    loop = jnp.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [0.0, -1.0, 1.0]])
    plane = jnp.asarray([0.0, 0.0, -1.0])
    through = jnp.asarray([0.0, 0.0, 3.0])
    assert not np.allclose(
        np.asarray(clip_to_halfspace(loop, decidable_heights(loop, plane), 5)),
        np.asarray(clip_to_halfspace(loop, decidable_heights(loop, plane, through=through), 5)),
    )


def test_a_vertex_lying_in_the_clipping_plane_gets_a_height_of_exactly_zero():
    """The whole point of the filter, on the configuration a shared mesh vertex produces.

    A vertex used to build the plane lies in it by construction, so its height is zero in exact
    arithmetic. Computed as a dot product it is not: the naive value below is a last-bit
    positive here and is a last-bit negative at another batch shape.
    """
    first = jnp.asarray([0.3, -0.7, 1.1])
    second = jnp.asarray([0.9, 0.2, -0.4])
    plane = jnp.cross(first, second)
    loop = jnp.stack([first, second, first + second])

    naive = np.asarray(dot(loop, plane[None, :]))
    assert np.any(naive != 0.0), "the naive height is already exact, so this pins nothing"
    assert np.all(np.asarray(decidable_heights(loop, plane)) == 0.0)


def test_a_height_that_is_genuinely_small_is_still_reported():
    """The other side of the filter, and what pins how wide its band may be.

    A filter that answered zero for anything small would make the clip ignore a real plane, and
    every other test here would still pass. What matters is the band's width *relative to the
    cancellation*, so the fixture cancels: the three terms of the dot product are each of order
    one and sum to a millionth, which is where a bound stated as a share of the terms summed
    can go wrong without a nearly-zero height anywhere in sight.
    """
    plane = jnp.asarray([1.0, 1.0, 1.0])
    loop = jnp.asarray([[1.0, 1.0, -2.0 + 1e-6], [1.0, 1.0, -2.0 - 1e-6], [0.0, 0.0, 3.0]])
    height = np.asarray(decidable_heights(loop, plane))
    assert height[0] == pytest.approx(1e-6, rel=1e-9)
    assert height[1] == pytest.approx(-1e-6, rel=1e-9)


def test_a_plane_spanned_by_a_direction_and_itself_is_exactly_zero():
    """``cross(v, v)`` is not zero in floating point, and the difference is not cosmetic.

    The compiler fuses one of the two products into the subtraction and rounds the other, so
    the result is a vector of last-bit noise. Clipping by that noise cuts a loop the real,
    zero plane would have left whole -- which can empty it, reporting a blocker that covers
    most of a source as covering none of it.
    """
    direction = jnp.asarray([0.41, -0.62, 0.17])
    assert np.any(np.asarray(jnp.cross(direction, direction)) != 0.0), "no noise to filter"
    assert np.all(np.asarray(spanning_plane(direction, direction)) == 0.0)


def test_a_plane_spanned_by_two_directions_one_rounding_apart_is_exactly_zero():
    """The near-repeat, which an exact-equality test on the two corners cannot catch.

    A clip emits a crossing point at the very end of an edge as ``a + 1.0 * (b - a)``, which is
    ``b`` to within a rounding but not bit for bit. The slot is a repeat in every sense that
    matters and its edge has no plane, yet the two corners compare unequal -- so a guard written
    as ``corner[k] == corner[k + 1]`` passes the noise straight through.
    """
    direction = jnp.asarray([0.41, -0.62, 0.17])
    nudged = direction.at[1].set(np.nextafter(np.float64(-0.62), 0.0))
    assert not bool(jnp.all(direction == nudged)), "the two are bit-identical, so this pins nothing"
    assert np.all(np.asarray(spanning_plane(direction, nudged)) == 0.0)


def test_a_plane_spanned_by_two_real_directions_survives():
    """The control for the two above: the filter must leave an ordinary edge plane alone."""
    plane = np.asarray(spanning_plane(jnp.asarray([0.3, -0.7, 1.1]), jnp.asarray([0.9, 0.2, -0.4])))
    assert np.allclose(plane, np.cross([0.3, -0.7, 1.1], [0.9, 0.2, -0.4]))


def test_a_plane_spanned_by_two_nearly_parallel_directions_still_survives():
    """What pins how wide the plane filter's band may be, from the side the control cannot reach.

    An ordinary edge plane clears the band by thirty orders of magnitude, so the control above
    would pass however wide the band were set. The two directions here are a hundredth of a
    microradian apart -- a real, if thin, silhouette edge, and still ten million times above the
    rounding a cross product of parallel inputs leaves behind.
    """
    plane = np.asarray(spanning_plane(jnp.asarray([1.0, 0.0, 0.0]), jnp.asarray([1.0, 1e-8, 0.0])))
    assert plane[2] == pytest.approx(1e-8, rel=1e-9)


def test_a_vertex_on_the_plane_is_kept_without_a_crossing_being_emitted_as_well():
    """The two sign tests must agree, or the clip builds a polygon with a doubled vertex.

    A vertex at height zero is inside the closed half-space, and the edges meeting it do not
    cross -- their crossing *is* that vertex, already emitted. Were the crossing emitted too,
    the loop would carry it twice and the stage would not be the identity it should be.
    """
    plane = jnp.asarray([0.0, 1.0, 0.0])
    loop = jnp.asarray([[1.0, 0.0, 3.0], [0.0, 2.0, 3.0], [-1.0, 0.0, 3.0], [0.0, 1.0, 3.0]])
    clipped = np.asarray(clip_to_halfspace(loop, decidable_heights(loop, plane), 5))
    assert np.allclose(clipped[:4], np.asarray(loop))
    assert np.allclose(clipped[4], clipped[3])


def test_the_clip_gives_the_same_answer_however_the_batch_is_shaped():
    """The defect all of this exists to remove, on the configuration that produced it.

    A compiler fuses multiplies and adds differently at different batch shapes, so a height
    that is mathematically zero comes out a last-bit positive in one shape and a last-bit
    negative in another, and the clip keeps a different polygon in each. Here the plane is
    built from two of the loop's own vertices, which is what a shared mesh vertex amounts to.
    """
    first = jnp.asarray([0.3, -0.7, 1.1])
    second = jnp.asarray([0.9, 0.2, -0.4])
    loop = jnp.stack([first, second, first + second, first - second])
    plane = jnp.cross(first, second)

    @jax.jit
    def clipped(loop, plane):
        return clip_to_halfspace(loop, decidable_heights(loop, plane), 5)

    one = np.asarray(clipped(loop, plane))
    many = np.asarray(
        clipped(jnp.broadcast_to(loop, (4096, 4, 3)), jnp.broadcast_to(plane, (4096, 3)))
    )
    assert np.array_equal(one, many[0])
