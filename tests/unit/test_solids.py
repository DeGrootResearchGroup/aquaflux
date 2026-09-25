"""The analytic primitives, the CSG that composes them, and the fluid-region body.

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
from aquaflux.solids import (
    Body,
    Box,
    Cone,
    Cylinder,
    Difference,
    HalfSpace,
    Intersection,
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
    """``Outside`` answers only the two ``Body`` questions, so it cannot be composed."""
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
# What a consumer building a mask expects of a body
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


# ---------------------------------------------------------------------------------------
# Clearance: a body vouching for a whole convex hull at once
# ---------------------------------------------------------------------------------------


def _directions(count: int, seed: int = 0) -> np.ndarray:
    """Random directions of random lengths, so a support that ignores the length is caught."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(count, 3)) * rng.uniform(0.2, 3.0, size=(count, 1))


def _inside_samples(body, count: int = 200_000, seed: int = 1) -> np.ndarray:
    """Points of ``body`` found by rejection from a box around every fixture."""
    rng = np.random.default_rng(seed)
    trial = rng.uniform(-2.0, 2.0, size=(count, 3))
    return trial[np.asarray(body.contains(jnp.asarray(trial)))]


@pytest.mark.parametrize("name", [name for name in _bodies() if name != "half space"])
def test_the_support_is_never_exceeded_by_any_point_of_the_body(name):
    """An upper bound, for primitives and compositions alike: no inside point reaches further.

    Sampling the inside is independent of the support formula, so a formula that under-reaches
    along any direction -- a union taking the nearer member, a box read along the wrong edges --
    shows here as a sampled point beyond it.
    """
    body = _bodies()[name]
    inside = _inside_samples(body)
    assert len(inside) > 1000
    directions = _directions(64)
    support = np.asarray(body.support(jnp.asarray(directions)))
    reached = (inside @ directions.T).max(axis=0)
    assert np.all(reached <= support + 1e-12)


def _rim(centre, axis, radius, count=20_000) -> np.ndarray:
    """Dense points on the circle of a disc."""
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    first = np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0])
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return (
        np.asarray(centre)
        + radius * np.cos(angle)[:, None] * first
        + radius * np.sin(angle)[:, None] * second
    )


def test_a_round_body_s_support_is_reached_on_the_rims_of_its_two_ends():
    """Exact, not merely a bound: the furthest point of a cylinder or a frustum is on an end rim.

    Both are the convex hull of their two end discs, so dense rim points reach the support to
    within the sampling's angular resolution. A formula that bounded loosely -- the radius added
    at full length whatever the direction, say -- would pass the bound test above and fail here.
    """
    directions = _directions(64, seed=3)
    cylinder = Cylinder([0.1, 0.0, -0.2], [0.3, 1.0, 0.2], 0.5, 0.9)
    frustum = Cone([0.1, 0.0, 0.0], [0.0, 0.3, 1.0], 0.5, 0.6, 0.2)
    for body, rims in (
        (cylinder, [(end, 0.5) for end in (-0.9, 0.9)]),
        (frustum, [(-0.5, 0.6), (0.5, 0.2)]),
    ):
        axis = np.asarray(body.axis)
        points = np.concatenate(
            [_rim(np.asarray(body.centre) + along * axis, axis, r) for along, r in rims]
        )
        support = np.asarray(body.support(jnp.asarray(directions)))
        reached = (points @ directions.T).max(axis=0)
        np.testing.assert_allclose(reached, support, rtol=0, atol=1e-6)
    sphere = Sphere([0.3, -0.1, 0.2], 0.7)
    surface = np.asarray(sphere.centre) + 0.7 * _directions(200_000, seed=5) / np.linalg.norm(
        _directions(200_000, seed=5), axis=1, keepdims=True
    )
    length = np.linalg.norm(directions, axis=1)
    np.testing.assert_allclose(
        (surface @ directions.T).max(axis=0) / length,
        np.asarray(sphere.support(jnp.asarray(directions))) / length,
        rtol=0,
        atol=1e-3,
    )


def test_a_parallelepiped_s_support_is_its_furthest_corner():
    """The eight corners solved for directly, on a box whose axes are not perpendicular.

    A rectangular box has its edges along its own axes, so reading the support along the axes
    rather than along the edges -- the rows of ``axes`` rather than the columns of its inverse --
    is right there and wrong here; only a skewed box tells the two apart.
    """
    axes = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.3, 1.0]])
    box = Box([0.2, -0.1, 0.4], [0.3, 0.2, 0.1], axes=axes)
    unit_axes = np.asarray(box.axes)
    corners = np.array(
        [
            np.asarray(box.centre)
            + np.linalg.solve(unit_axes, (2 * np.array(signs) - 1) * np.asarray(box.half_sizes))
            for signs in np.ndindex(2, 2, 2)
        ]
    )
    directions = _directions(64, seed=4)
    np.testing.assert_allclose(
        np.asarray(box.support(jnp.asarray(directions))),
        (corners @ directions.T).max(axis=0),
        rtol=0,
        atol=1e-12,
    )


def _certified(body, cloud) -> bool:
    """Whether ``body`` vouches for the hull of ``cloud``: some witness negative at every point."""
    witnesses = np.asarray(body.clearance(jnp.asarray(cloud, dtype=float)))
    return bool(witnesses.shape[-1]) and bool(np.any(witnesses.max(axis=0) < 0.0))


@pytest.mark.parametrize("name", list(_bodies()))
def test_a_certified_hull_holds_no_point_of_the_body(name):
    """The guarantee itself, swept over random clouds: a certified hull never touches the body.

    Each cloud is a few points around a random centre; where the body certifies it, points drawn
    throughout its hull -- random convex combinations, not only the cloud itself, since a hull
    can pass through a body its corners all miss -- are checked with ``contains``, and segments
    between cloud points with ``blocks``. Both halves are required to be exercised: some clouds
    certified, and some that touch the body, which a certificate must never cover.
    """
    body = _bodies()[name]
    rng = np.random.default_rng(11)
    n_clouds, size, n_hull = 400, 5, 400
    centre = rng.uniform(-1.2, 1.2, (n_clouds, 1, 3))
    clouds = centre + rng.normal(size=(n_clouds, size, 3)) * rng.uniform(
        0.02, 0.8, (n_clouds, 1, 1)
    )
    hull = np.einsum("chk,ckd->chd", rng.dirichlet(np.ones(size), (n_clouds, n_hull)), clouds)
    touching = np.asarray(body.contains(jnp.asarray(hull))).any(axis=1)
    witnesses = np.asarray(body.clearance(jnp.asarray(clouds)))
    certified = np.any(witnesses.max(axis=1) < 0.0, axis=-1)
    assert not np.any(certified & touching)
    start, finish = clouds[:, [0, 1, 2, 3]], clouds[:, [4, 3, 1, 0]]
    crossed = np.asarray(
        body.blocks(jnp.asarray(start), jnp.asarray(finish), jnp.zeros(start.shape[:-1]))
    )
    assert not np.any(crossed[certified])
    assert certified.sum() > 20, certified.sum()
    assert touching.sum() > 20, touching.sum()


def test_a_union_is_cleared_only_when_every_one_of_its_bodies_is():
    """A cloud clear of one lobe but inside the other is not certified.

    A union reaches as far as its furthest member; bounding it by the nearer one would vouch for
    the cloud here, which sits inside the second sphere.
    """
    union = Union(Sphere([-1.0, 0.0, 0.0], 0.3), Sphere([1.0, 0.0, 0.0], 0.3))
    cloud = np.array([[1.0, 0.0, 0.0], [1.1, 0.05, 0.0], [0.95, 0.0, 0.05]])
    assert Sphere([-1.0, 0.0, 0.0], 0.3).clearance(jnp.asarray(cloud)).max(axis=0).min() < 0.0
    assert not _certified(union, cloud)
    assert _certified(union, cloud + np.array([0.0, 0.0, 1.0]))


def test_an_intersection_is_cleared_by_any_one_of_its_bodies():
    """A cloud inside the sphere but beyond the box misses the intersection, and is certified."""
    sphere, box = Sphere([0.0, 0.0, 0.0], 1.0), Box([2.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    cloud = np.array([[-0.5, 0.0, 0.0], [-0.4, 0.2, 0.1], [-0.6, -0.1, 0.2]])
    assert not _certified(sphere, cloud)
    assert _certified(Intersection(sphere, box), cloud)


def test_a_face_is_a_witness_but_a_cloud_touching_it_is_not_vouched_for():
    """Beyond a half-space's plane is certified; on the plane is left to the exact test.

    A half-space reaches infinitely along every fixed direction but its normal, so its face is
    its only witness -- and the margin keeps a cloud that merely touches the solid from being
    vouched for, where the segment test would call a touching segment blocked.
    """
    wall = HalfSpace([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    beyond = np.array([[0.0, 0.0, 1.001], [5.0, -3.0, 3.0], [-4.0, 7.0, 1.001]])
    assert _certified(wall, beyond)
    # On the plane, and a rounding beyond it: neither is a gap worth vouching for. The plane is
    # off the origin so that its projection, and with it the rounding, is not zero.
    on_plane = beyond * np.array([1.0, 1.0, 0.0]) + np.array([0.0, 0.0, 1.0])
    assert not _certified(wall, on_plane)
    assert not _certified(wall, on_plane + np.array([0.0, 0.0, 1e-15]))


def test_an_outside_vouches_for_points_in_one_convex_region_and_not_across_two():
    """The chamber-and-pipe fluid: points all in the chamber are certified; chamber plus pipe is not.

    Every point of the second cloud is in the water, so a certificate asking only that -- each
    point in *some* region -- would vouch for it; but the hull cuts through the chamber's roof
    beside the pipe, and a segment between two of its points is blocked.
    """
    fluid = _reactor()
    chamber = np.array([[0.1, 0.0, 0.0], [0.9, 0.05, -0.02], [0.5, -0.06, 0.07]])
    assert _certified(fluid, chamber)
    across = np.array([[0.8, 0.0, 0.0], [0.2, 0.0, 0.9], [0.5, 0.05, 0.0]])
    assert not np.asarray(fluid.contains(jnp.asarray(across))).any()
    assert blocks(fluid, across[0], across[1], 1e-9)
    assert not _certified(fluid, across)


def test_a_body_that_cannot_vouch_for_anything_has_no_witnesses():
    """The default, for a body answered some other way: no columns, so nothing is ever certified."""

    class Bespoke(Body):
        def blocks(self, origin, target, min_distance):
            return jnp.zeros(jnp.broadcast_shapes(origin.shape, target.shape)[:-1], dtype=bool)

        def contains(self, position):
            return jnp.zeros(position.shape[:-1], dtype=bool)

    witnesses = Bespoke().clearance(jnp.zeros((7, 3)))
    assert witnesses.shape == (7, 0)
    assert not _certified(Bespoke(), np.zeros((3, 3)))
