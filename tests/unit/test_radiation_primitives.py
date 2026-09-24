"""The analytic primitives, the CSG that composes them, and the fluid-region occluder.

Each primitive is pinned against a closed form, because that is the only reference that says a
body is the shape it claims to be rather than merely a consistent shape. On top of that sits one
check that applies to everything here at once: **the intervals a body reports must agree with
dense sampling of its own ``contains``**. Those are two independent readings — one solves the
geometry along a line, the other asks a point at a time — so a body whose interval arithmetic is
wrong cannot pass by being self-consistent, which is the failure a closed form aimed at one
shape would not catch in a combinator.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation import NoOcclusion, Surfaces, build_visibility
from aquaflux.radiation.occluders import (
    Box,
    Cone,
    Cylinder,
    Difference,
    HalfSpace,
    Intersection,
    Occluder,
    Outside,
    Solid,
    Sphere,
    Union,
)

NO_OFFSET = jnp.zeros(())


def blocks(body, start, finish, min_distance=0.0):
    """``body.blocks`` on one segment, as a plain bool."""
    return bool(body.blocks(jnp.asarray(start), jnp.asarray(finish), jnp.asarray(min_distance)))


def interval(body, start, finish):
    """The one interval of a convex body along a segment, as plain floats."""
    enter, exit_ = body.intervals(
        jnp.asarray(start, dtype=float),
        jnp.asarray(finish, dtype=float) - jnp.asarray(start, dtype=float),
    )
    return float(enter[0]), float(exit_[0])


# ---------------------------------------------------------------------------------------
# Each primitive against a closed form
# ---------------------------------------------------------------------------------------


def test_a_sphere_cuts_the_chord_the_pythagorean_theorem_says_it_does():
    """A line missing the centre by ``h`` meets a ball of radius ``R`` over ``2 sqrt(R^2 - h^2)``.

    Catches a radius used where a squared radius belongs, a centre offset dropped, and an
    interval that is merely the right width in the wrong place: both ends are asserted, not the
    length.
    """
    radius, miss = 2.0, 1.2
    ball = Sphere(centre=[1.0, 0.0, 0.0], radius=radius)
    start, finish = (-9.0, miss, 0.0), (11.0, miss, 0.0)
    half_chord = np.sqrt(radius**2 - miss**2)
    enter, exit_ = interval(ball, start, finish)
    # The segment runs 20 long from x = -9, so x = 1 +/- half_chord is t = (10 +/- h) / 20.
    assert enter == pytest.approx((10.0 - half_chord) / 20.0, abs=1e-12)
    assert exit_ == pytest.approx((10.0 + half_chord) / 20.0, abs=1e-12)
    assert blocks(ball, start, finish)
    assert not blocks(ball, (-9.0, radius + 1e-9, 0.0), (11.0, radius + 1e-9, 0.0))


def test_a_cone_has_the_radius_its_two_end_radii_interpolate():
    """Sliced across the axis, a taper is a circle whose radius the end radii interpolate.

    The slice is exact: at a fixed axial station the surface is a circle of radius ``R(u)``, so
    the chord is the ball's chord with that radius. Catches a slope taken from the wrong pair of
    ends, a taper measured from the centre rather than from the base, and the whole body being
    silently a cylinder.
    """
    # Apex at x = 0, widening to 0.3 at x = 1.
    cone = Cone(
        centre=[0.5, 0, 0], axis=[1, 0, 0], half_length=0.5, base_radius=0.0, tip_radius=0.3
    )
    station, miss = 0.6, 0.1
    here = 0.3 * station
    half_chord = np.sqrt(here**2 - miss**2)
    enter, exit_ = interval(cone, (station, -1.0, miss), (station, 1.0, miss))
    assert enter == pytest.approx((1.0 - half_chord) / 2.0, abs=1e-12)
    assert exit_ == pytest.approx((1.0 + half_chord) / 2.0, abs=1e-12)
    assert blocks(cone, (station, -1.0, here - 1e-9), (station, 1.0, here - 1e-9))
    assert not blocks(cone, (station, -1.0, here + 1e-9), (station, 1.0, here + 1e-9))


def test_a_cone_is_hit_on_its_own_nappe_and_not_on_the_mirror_one_beyond_the_apex():
    """A ray steeper than the taper meets BOTH nappes of the cone the equation describes.

    Only one of them is the body. A line parallel to the axis at radius 0.05 is inside this cone
    from ``x = 1/6`` onward — and inside the mirror cone from ``x = -1/6`` backward, which is not
    there. Picking the mirror branch does not give a wrong hit; the end caps then discard it and
    the body reads as **clear**, so the failure is light passing through a solid cone.
    """
    cone = Cone(
        centre=[0.5, 0, 0], axis=[1, 0, 0], half_length=0.5, base_radius=0.0, tip_radius=0.3
    )
    start, finish = (-1.0, 0.05, 0.0), (2.0, 0.05, 0.0)
    enter, _ = interval(cone, start, finish)
    assert (-1.0 + 3.0 * enter) == pytest.approx(0.05 / 0.3, abs=1e-12)
    assert blocks(cone, start, finish)
    # The mirror nappe alone, with the caps moved off it, is still not solid.
    assert not blocks(cone, (-1.0, 0.05, 0.0), (-0.01, 0.05, 0.0))


def test_a_cone_with_equal_end_radii_is_refused_as_a_cylinder():
    """The taper's quadratic has no grazing-safe form, so the body with one is the right spelling."""
    with pytest.raises(ValueError, match="Cylinder"):
        Cone(centre=[0, 0, 0], axis=[1, 0, 0], half_length=1.0, base_radius=0.5, tip_radius=0.5)


def test_a_box_is_bounded_by_its_own_axes_and_not_by_the_coordinate_ones():
    """Turned 45 degrees, a unit box reaches ``sqrt(2)`` along ``x`` — its corner, not its face.

    The whole content of an oriented box is that the half-sizes are measured along ``axes``.
    A test on an axis-aligned box cannot tell whether they are, which is why this one is turned.
    """
    turn = np.sqrt(0.5)
    turned = Box(
        centre=[0, 0, 0],
        half_sizes=[1.0, 1.0, 1.0],
        axes=[[turn, turn, 0.0], [-turn, turn, 0.0], [0.0, 0.0, 1.0]],
    )
    aligned = Box(centre=[0, 0, 0], half_sizes=[1.0, 1.0, 1.0])
    assert bool(turned.contains(jnp.asarray([1.40, 0.0, 0.0])))
    assert not bool(aligned.contains(jnp.asarray([1.40, 0.0, 0.0])))
    assert not bool(turned.contains(jnp.asarray([1.45, 0.0, 0.0])))
    # Along its own diagonal the reach is the corner distance, sqrt(2) in the turned frame.
    enter, exit_ = interval(turned, (-4.0, 0.0, 0.0), (4.0, 0.0, 0.0))
    assert (-4.0 + 8.0 * enter) == pytest.approx(-np.sqrt(2.0), abs=1e-12)
    assert (-4.0 + 8.0 * exit_) == pytest.approx(np.sqrt(2.0), abs=1e-12)


def test_a_cylinder_is_the_tube_and_its_two_ends_and_the_ends_are_where_they_say():
    """The axial clip is asserted by value: a ray down the axis stops at the flat end."""
    body = Cylinder(centre=[0, 0, 1.0], axis=[0, 0, 1], radius=0.5, half_length=2.0)
    enter, exit_ = interval(body, (0.0, 0.0, -10.0), (0.0, 0.0, 10.0))
    assert (-10.0 + 20.0 * enter) == pytest.approx(-1.0, abs=1e-12)
    assert (-10.0 + 20.0 * exit_) == pytest.approx(3.0, abs=1e-12)


# ---------------------------------------------------------------------------------------
# One oracle for all of them: the intervals against dense sampling of `contains`
# ---------------------------------------------------------------------------------------


def _bodies() -> dict[str, Solid]:
    """One of everything, including bodies whose interval count is more than one."""
    return {
        "half space": HalfSpace([0.2, 0, 0], [1.0, 0.5, 0.0]),
        "sphere": Sphere([0.1, -0.2, 0.3], 0.8),
        "cylinder": Cylinder([0, 0, 0], [0.3, 1.0, 0.2], 0.5, 0.9),
        "cone": Cone([0, 0, 0], [1.0, 0.2, 0.0], 0.7, 0.6),
        "frustum": Cone([0.1, 0, 0], [0, 0, 1], 0.5, 0.6, 0.2),
        "turned box": Box([0, 0, 0], [0.4, 0.6, 0.3], [[1, 1, 0], [-1, 1, 0], [0, 0, 1]]),
        "union": Union(Sphere([0.6, 0, 0], 0.4), Cylinder([-0.4, 0, 0], [1, 0, 0], 0.25, 0.5)),
        "intersection": Intersection(Sphere([0, 0, 0], 0.7), Box([0.2, 0, 0], [0.5, 0.5, 0.5])),
        "difference": Difference(
            Cylinder([0, 0, 0], [0, 0, 1], 0.6, 0.8), Cylinder([0, 0, 0], [0, 0, 1], 0.3, 2.0)
        ),
        "two holes in two lobes": Difference(
            Union(Sphere([0.5, 0, 0], 0.45), Sphere([-0.5, 0, 0], 0.45)), Sphere([0, 0, 0], 0.35)
        ),
    }


@pytest.mark.parametrize("name", list(_bodies()))
def test_the_intervals_agree_with_sampling_the_body_a_point_at_a_time(name):
    """Sampling ``contains`` along each segment is an independent answer to ``blocks``.

    A body may be wrong in either direction, and the two show up differently: an interval that is
    too wide reports a hit where no sampled point is inside, and one that is too narrow, or in
    the wrong place, misses a stretch of points that are. **A disagreement is only forgiven when
    the interval the body claims is narrower than the sampling step**, which is a genuine sliver
    the sampler stepped over rather than a defect — and the forgiveness is bounded by asserting
    that width, not by a tolerance on the count.
    """
    body = _bodies()[name]
    rng = np.random.default_rng(0)
    n, steps = 2000, 1201
    start = rng.uniform(-2.0, 2.0, (n, 3))
    finish = rng.uniform(-2.0, 2.0, (n, 3))
    step = np.linspace(0.0, 1.0, steps)
    walk = start[:, None, :] + step[None, :, None] * (finish - start)[:, None, :]

    sampled = np.asarray(body.contains(jnp.asarray(walk))).any(axis=1)
    told = np.asarray(body.blocks(jnp.asarray(start), jnp.asarray(finish), jnp.zeros(n)))
    assert told.any() and not told.all(), "the fixture must exercise both answers"

    disputed = np.flatnonzero(sampled != told)
    if len(disputed):
        enter, exit_ = body.intervals(
            jnp.asarray(start[disputed]), jnp.asarray((finish - start)[disputed])
        )
        enter, exit_ = np.asarray(enter), np.asarray(exit_)
        widest = np.where(exit_ >= enter, exit_ - enter, 0.0).max(axis=-1)
        assert (widest < 1.0 / (steps - 1)).all(), (
            f"{name}: {len(disputed)} segments disagree with sampling over a stretch the sampler "
            f"could not have stepped over (widest {widest.max():.3e})"
        )


def test_the_interval_count_is_what_composition_says_it_is():
    """It is a static property, so a wrong one is a silently mis-shaped array, not an error."""
    ball = Sphere([0, 0, 0], 1.0)
    assert ball.interval_count == 1
    assert Union(ball, ball, ball).interval_count == 3
    assert Intersection(ball, Union(ball, ball)).interval_count == 2
    assert Difference(Union(ball, ball), ball).interval_count == 4
    for body in (Union(ball, ball), Difference(ball, ball)):
        enter, _ = body.intervals(jnp.zeros((5, 3)), jnp.ones((5, 3)))
        assert enter.shape == (5, body.interval_count)


# ---------------------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------------------


def test_six_half_spaces_intersected_are_the_box_built_directly():
    """A composed body must reproduce the primitive it is a composition of, bit for bit.

    The box builds its six planes and intersects their intervals *inside* one convex body; the
    intersection builds six convex bodies and multiplies their intervals together. Those are
    different code paths to the same shape, so this pins the CSG product against the convex
    shortcut rather than against another tolerance.
    """
    centre, half = np.array([0.3, -0.2, 0.1]), np.array([0.4, 0.7, 0.25])
    direct = Box(centre=centre, half_sizes=half)
    composed = Intersection(
        *[
            HalfSpace(
                point=centre + sign * half[axis] * np.eye(3)[axis], normal=sign * np.eye(3)[axis]
            )
            for axis in range(3)
            for sign in (1.0, -1.0)
        ]
    )
    assert composed.interval_count == 1
    rng = np.random.default_rng(3)
    start = jnp.asarray(rng.uniform(-1.5, 1.5, (4000, 3)))
    finish = jnp.asarray(rng.uniform(-1.5, 1.5, (4000, 3)))
    near = jnp.zeros(4000)
    assert np.array_equal(
        np.asarray(direct.blocks(start, finish, near)),
        np.asarray(composed.blocks(start, finish, near)),
    )
    points = jnp.asarray(rng.uniform(-1.5, 1.5, (4000, 3)))
    assert np.array_equal(
        np.asarray(direct.contains(points)), np.asarray(composed.contains(points))
    )


def test_a_difference_lets_a_ray_down_the_bore_it_cut():
    """The point of a difference: a sleeve is not a rod, and a ray down its axis passes."""
    rod = Cylinder([0, 0, 0], [0, 0, 1], 0.6, 0.8)
    sleeve = Difference(rod, Cylinder([0, 0, 0], [0, 0, 1], 0.3, 2.0))
    assert blocks(rod, (0.0, 0.0, -3.0), (0.0, 0.0, 3.0))
    assert not blocks(sleeve, (0.0, 0.0, -3.0), (0.0, 0.0, 3.0))
    # Off the bore it still blocks, twice: once through each wall.
    assert blocks(sleeve, (0.0, 0.45, -3.0), (0.0, 0.45, 3.0))
    assert not bool(sleeve.contains(jnp.asarray([0.0, 0.0, 0.0])))
    assert bool(sleeve.contains(jnp.asarray([0.0, 0.45, 0.0])))


def test_a_hole_in_several_pieces_is_refused_with_the_rewrite_that_works():
    """Subtracting a union would need its complement, which is not one interval."""
    ball = Sphere([0, 0, 0], 1.0)
    with pytest.raises(ValueError, match=r"Difference\(Difference"):
        Difference(ball, Union(ball, ball))


def test_a_combinator_refuses_a_body_that_cannot_report_intervals():
    """``Outside`` answers only the two occluder questions, so it cannot be composed."""
    ball = Sphere([0, 0, 0], 1.0)
    with pytest.raises(TypeError, match="intervals"):
        Union(ball, Outside(ball))
    with pytest.raises(ValueError, match="at least one"):
        Union()


# ---------------------------------------------------------------------------------------
# The fluid as regions
# ---------------------------------------------------------------------------------------


def _reactor(pipe_bottom: float = 0.05) -> Outside:
    """A chamber with a pipe standing on it, the pipe's foot at ``z = pipe_bottom``.

    The chamber's own top is ``z = 0.1``, so the default reaches *inside* it — which is how a
    branch has to be described for the two regions to have no gap between them.
    """
    top = 1.0
    chamber = Cylinder(centre=[0.5, 0, 0], axis=[1, 0, 0], radius=0.1, half_length=0.5)
    pipe = Cylinder(
        centre=[0.2, 0, (pipe_bottom + top) / 2.0],
        axis=[0, 0, 1],
        radius=0.02,
        half_length=(top - pipe_bottom) / 2.0,
    )
    return Outside(chamber, pipe)


def test_a_segment_between_two_points_of_one_convex_region_is_never_blocked():
    """The shortcut the whole formulation rests on, asserted rather than assumed.

    Convexity means the straight line between two interior points stays interior, so a chamber
    cell and a lamp point in the same chamber need no test at all. If this ever failed, every
    pair in the bulk of a reactor would be answered wrongly and the field would be dark.
    """
    fluid = _reactor()
    rng = np.random.default_rng(5)
    angle = rng.uniform(0.0, 2.0 * np.pi, 3000)
    radius = 0.1 * np.sqrt(rng.uniform(0.0, 1.0, 3000))
    inside = np.stack(
        [rng.uniform(0.0, 1.0, 3000), radius * np.cos(angle), radius * np.sin(angle)], axis=1
    )
    assert not np.asarray(fluid.contains(jnp.asarray(inside))).any()
    assert not np.asarray(
        fluid.blocks(jnp.asarray(inside[:1500]), jnp.asarray(inside[1500:]), jnp.zeros(1500))
    ).any()


def test_a_point_up_the_pipe_sees_only_what_the_opening_lets_through():
    """The whole reason the formulation is worth having, checked against dense sampling.

    Nothing identifies the opening; the covering test finds it. Sampling the fluid a point at a
    time along each segment is the independent answer.
    """
    fluid = _reactor()
    rng = np.random.default_rng(6)
    sources = np.stack([rng.uniform(0.0, 1.0, 900), np.zeros(900), np.zeros(900)], axis=1)
    up_the_pipe = np.stack(
        [
            0.2 + rng.uniform(-0.01, 0.01, 900),
            rng.uniform(-0.01, 0.01, 900),
            rng.uniform(0.1, 0.9, 900),
        ],
        axis=1,
    )
    assert not np.asarray(fluid.contains(jnp.asarray(up_the_pipe))).any()
    told = np.asarray(fluid.blocks(jnp.asarray(sources), jnp.asarray(up_the_pipe), jnp.zeros(900)))
    step = np.linspace(0.0, 1.0, 4001)
    walk = sources[:, None, :] + step[None, :, None] * (up_the_pipe - sources)[:, None, :]
    left_the_fluid = np.asarray(fluid.contains(jnp.asarray(walk))).any(axis=1)
    assert told.any() and not told.all(), "the fixture must exercise both answers"
    assert np.array_equal(told, left_the_fluid)


def test_a_gap_between_two_regions_reads_as_solid():
    """The documented trap, and the reason the covering test looks past ``lower``.

    A pipe stopped short of the chamber leaves a ring of neither region. Nothing raises: the
    reactor simply goes dark up that pipe. Examining only the near end of each segment — the
    obvious cheap covering test — would call every one of these clear.
    """
    source = np.tile([0.2, 0.0, 0.0], (200, 1))
    target = np.stack([np.full(200, 0.2), np.zeros(200), np.linspace(0.15, 0.9, 200)], axis=1)
    assert not np.asarray(
        _reactor(pipe_bottom=0.05).blocks(jnp.asarray(source), jnp.asarray(target), jnp.zeros(200))
    ).any()
    # Lifting the pipe's foot to z = 0.12 clears the chamber's top at z = 0.1, so a ring of
    # neither region is left between them.
    assert np.asarray(
        _reactor(pipe_bottom=0.12).blocks(jnp.asarray(source), jnp.asarray(target), jnp.zeros(200))
    ).all()


def test_a_receiver_sitting_on_the_wall_needs_the_far_end_excluded_too():
    """A facet centroid is *on* the boundary, and a rounding outside it is not a shadow.

    ``min_distance`` guards the near end of a segment against a facet shadowing itself. The far
    end has no margin of its own, so a receiver a rounding outside the fluid leaves an
    infinitesimal uncovered sliver and the pair reads as blocked — which over an enclosure is not
    a small error but a shadow everywhere. The margin is applied at both ends for that reason.
    """
    fluid = Outside(Cylinder(centre=[0, 0, 0], axis=[0, 0, 1], radius=1.0, half_length=1.0))
    start = jnp.asarray([0.0, 0.0, 0.0])
    on_the_wall = jnp.asarray([1.0 + 4e-16, 0.0, 0.0])
    assert blocks(fluid, start, on_the_wall, 0.0)
    assert not blocks(fluid, start, on_the_wall, 1e-6)
    # The margin must not reach so far that a real blocker is missed.
    assert blocks(fluid, start, jnp.asarray([2.0, 0.0, 0.0]), 1e-6)


def test_a_cell_outside_every_region_is_embedded_in_the_wall_unless_it_is_a_rounding():
    """``contains`` is what refuses a scene, and a mesh never lands exactly on its own surface."""
    fluid = Outside(Cylinder(centre=[0, 0, 0], axis=[0, 0, 1], radius=1.0, half_length=1.0))
    just_out = jnp.asarray([[1.0 + 1e-9, 0.0, 0.0]])
    assert bool(fluid.contains(just_out)[0])
    assert not bool(Outside(*fluid.regions, tolerance=1e-6).contains(just_out)[0])
    # A tolerance is a rounding, not a licence: a cell a millimetre into the metal is still in it.
    assert bool(
        Outside(*fluid.regions, tolerance=1e-6).contains(jnp.asarray([[1.001, 0.0, 0.0]]))[0]
    )


def test_the_regions_can_be_any_solid_not_only_a_convex_one():
    """Correctness needs the intervals, not convexity; convexity is only what makes it cheap."""
    lobes = Union(Sphere([-0.4, 0, 0], 0.5), Sphere([0.4, 0, 0], 0.5))
    fluid = Outside(lobes)
    assert fluid.fluid.interval_count == 2
    # Along the axis the lobes overlap from x = -0.1 to x = 0.1, so the line stays in the fluid.
    assert not blocks(fluid, (-0.7, 0.0, 0.0), (0.7, 0.0, 0.0))
    # At y = 0.35 each lobe reaches only sqrt(0.5^2 - 0.35^2) = 0.357 from its own centre, so
    # they stop at x = -0.043 and x = 0.043 and the segment crosses the gap between them.
    assert blocks(fluid, (-0.7, 0.35, 0.0), (0.7, 0.35, 0.0))


# ---------------------------------------------------------------------------------------
# What the rest of the module expects of an occluder
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        Sphere([0, 0, 0], 0.5),
        Cone([0, 0, 0], [1, 0, 0], 0.5, 0.4),
        Box([0, 0, 0], [0.3, 0.3, 0.3]),
        Union(Sphere([0, 0, 0], 0.4), Sphere([0.5, 0, 0], 0.4)),
        Outside(Cylinder([0, 0, 0], [1, 0, 0], 1.0, 2.0)),
    ],
    ids=["sphere", "cone", "box", "union", "outside"],
)
def test_the_intersection_tests_broadcast(body):
    """The mask is built by broadcasting facets against receivers, so a body must take that."""
    rng = np.random.default_rng(0)
    starts = jnp.asarray(rng.uniform(-1.0, 1.0, (5, 1, 3)))
    finishes = jnp.asarray(rng.uniform(3.0, 5.0, (1, 4, 3)))
    assert body.blocks(starts, finishes, jnp.zeros((5, 4))).shape == (5, 4)
    assert body.contains(starts).shape == (5, 1)


def test_a_body_survives_being_traced_with_its_geometry_as_arguments():
    """The mask build compiles the whole per-chunk expression, so nothing here may need a value."""
    fluid = _reactor()
    # Straight up the pipe from under it is clear; from along the chamber the line leaves through
    # the chamber's curved wall before it reaches the pipe's footprint.
    start = jnp.asarray([[0.2, 0.0, 0.0], [0.8, 0.0, 0.0], [0.2, 0.0, 0.0], [0.6, 0.0, 0.0]])
    finish = jnp.asarray([[0.2, 0.0, 0.5], [0.2, 0.0, 0.5], [0.2, 0.0, 0.9], [0.2, 0.0, 0.9]])
    eager = np.asarray(fluid.blocks(start, finish, jnp.zeros(4)))
    compiled = np.asarray(
        eqx.filter_jit(lambda body, a, b, n: body.blocks(a, b, n))(
            fluid, start, finish, jnp.zeros(4)
        )
    )
    assert eager.any() and not eager.all(), "the fixture must exercise both answers"
    assert np.array_equal(eager, compiled)


class _HostBody(Occluder):
    """A blocker answered on the host, as a triangulated one is: numpy, and index bookkeeping.

    Stands in for the triangle path, whose grid walk drops rays as they are settled and so
    cannot be traced. What matters here is only that it touches ``numpy`` on its arguments,
    which is what raises under a trace.
    """

    def contains(self, position) -> jnp.ndarray:
        """A zero-thickness sheet has no interior, so nothing is inside it."""
        return jnp.zeros(jnp.asarray(position).shape[:-1], dtype=bool)

    def blocks(self, origin, target, min_distance) -> jnp.ndarray:
        """Blocks whatever starts on the far side of ``x = 0``."""
        del min_distance
        source, _ = np.broadcast_arrays(
            np.asarray(origin, dtype=float), np.asarray(target, dtype=float)
        )
        live = np.flatnonzero(np.ones(source.shape[:-1]).ravel())
        out = np.zeros(source.shape[:-1], dtype=bool).ravel()
        out[live] = source.reshape(-1, 3)[live, 0] < 0.0
        return jnp.asarray(out.reshape(source.shape[:-1]))


def test_a_host_side_blocker_and_a_primitive_stand_in_one_scene():
    """The hybrid: a vessel described as primitives beside whatever really is a triangle soup.

    ⚠️ **Compiling the mask build unconditionally breaks this, and breaks it by raising.** A
    primitive wants compiling — several inequalities across a receivers-by-facets array
    materialize every intermediate otherwise — and a host-side blocker cannot be traced at all.
    Whoever builds the mask therefore reads each body's own declaration. Without that, the only
    bodies that work are the ones this module happens to ship.
    """
    vertices = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    emitters = Surfaces.from_triangles(vertices, emission=1.0)
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    primitive = Sphere(centre=[0.0, 0.0, 5.0], radius=0.1)
    assert primitive.traceable and not _HostBody().traceable

    mask = build_visibility(
        [primitive, _HostBody()], emitters, points, self_occlusion=NoOcclusion()
    )
    assert mask.blocked.shape == (2, 2, 1)
    assert not np.asarray(mask.blocked[0]).any(), "the sphere is nowhere near these segments"
    assert not np.asarray(mask.blocked[1]).any(), "the facet centroid is not at negative x"
