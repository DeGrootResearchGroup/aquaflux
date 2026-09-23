"""The emitting surface shadowing itself — the case no analytic body can express."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation.gather import direct_fluence_rate
from aquaflux.radiation.occluders import Cylinder
from aquaflux.radiation.self_occlusion import NoOcclusion, RayCastOcclusion
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.triangles import _call_shape, _edge_function, segment_is_cut
from aquaflux.radiation.visibility import build_visibility
from scipy.spatial import ConvexHull

from tests.unit.radiation_references import (
    L_OUTLINE,
    closed_drum,
    closed_prism,
    cylinder_triangles,
    rectangle_triangles,
)

ONE_TRIANGLE = jnp.asarray([[[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 1.0, 1.0]]])
NO_OFFSET = jnp.zeros(1)


def _emitter_and_panel():
    """A small emitter at the origin and, in the same surface set, a panel in its way."""
    emitter = rectangle_triangles([0.0, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.02])
    panel = rectangle_triangles([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    # The emitter faces +x; the panel's own emission is zero, it is only in the way.
    vertices = np.concatenate([emitter, panel])
    return Surfaces.from_triangles(vertices, emission=[1000.0, 1000.0, 0.0, 0.0])


# ---------------------------------------------------------------------------------------
# The intersection test
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "finish", "expected", "what"),
    [
        ((0.0, 0.0, 0.0), (0.0, 0.0, 3.0), True, "straight through"),
        ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), False, "parallel to it"),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.5), False, "stops short"),
        ((0.0, 0.0, 2.0), (0.0, 0.0, 3.0), False, "starts past it"),
        ((0.0, 0.0, 0.0), (0.0, 5.0, 3.0), False, "outside the edge"),
        ((0.0, 0.0, 3.0), (0.0, 0.0, 0.0), True, "the other way round"),
    ],
)
def test_a_triangle_cuts_what_passes_through_it(start, finish, expected, what):
    cut = segment_is_cut(jnp.asarray([start]), jnp.asarray([finish]), ONE_TRIANGLE, NO_OFFSET)
    assert bool(cut[0]) is expected, what


@pytest.mark.parametrize("work_limit", [1, 17, 97, 4_000_000])
def test_the_answer_does_not_depend_on_how_the_work_is_split(work_limit):
    """The split is a memory strategy and must not be a numerical one.

    Both axes are cut to honour the limit, so a limit of one puts a single ray against a single
    triangle per pass, and the accumulated result has to be identical to forming the whole thing
    at once. Seventeen splits the forty triangles unevenly, leaving a remainder block, which is
    the one shape the others do not reach.
    """
    rng = np.random.default_rng(2)
    triangles = jnp.asarray(rng.normal(size=(40, 3, 3)))
    origins = jnp.asarray(rng.normal(size=(23, 3)))
    targets = jnp.asarray(rng.normal(size=(23, 3)) * 2.0)
    reference = segment_is_cut(origins, targets, triangles, jnp.zeros(23))
    split = segment_is_cut(origins, targets, triangles, jnp.zeros(23), work_limit=work_limit)
    np.testing.assert_array_equal(np.asarray(split), np.asarray(reference))
    assert int(np.count_nonzero(np.asarray(reference))) > 0, "the fixture blocks nothing"


def test_the_triangle_block_does_not_shrink_as_the_rays_grow():
    """Triangles take the call's budget first; rays take what is left.

    The order is invisible to every correctness test -- the answer is bit-identical either way,
    which the test above checks -- and it is worth a factor of four. With rays first, the ray
    count of a transfer build (receivers times facets, millions) took the whole budget and left a
    block of ONE triangle, so each call streamed millions of rays to test a single triangle.
    Measured with the identical kernel on 1532 triangles and 3.2 million rays: 120.5 million
    tests per second rays-first against 465.4 with the whole set per call.
    """
    limit = 4_000_000
    for rays in (1_000, 200_000, 3_200_000, 10_137_856):
        ray_chunk, block = _call_shape(rays, 1532, limit)
        assert block == 1532, f"{rays:,} rays shrank the triangle block to {block}"
        assert ray_chunk * block <= limit, "the call exceeds the bound it exists to keep"
        assert ray_chunk >= 1
    # A triangle set larger than the bound on its own is split, one ray per call.
    assert _call_shape(10, 5_000_000, limit) == (1, limit)


def test_a_facet_is_excluded_from_cutting_its_own_rays_by_index():
    """Every ray leaves its facet's centroid, so the facet is always hit, at zero distance.

    By index rather than by tolerance: a tolerance large enough to cover this would also
    swallow a genuine blocker a short way off, and there is no need to guess when the identity
    of the facet is known.
    """
    start, finish = jnp.asarray([[0.0, 0.0, 0.0]]), jnp.asarray([[0.0, 0.0, 3.0]])
    assert bool(segment_is_cut(start, finish, ONE_TRIANGLE, NO_OFFSET)[0]) is True
    excluded = segment_is_cut(start, finish, ONE_TRIANGLE, NO_OFFSET, exclude=jnp.asarray([0]))
    assert bool(excluded[0]) is False


def test_the_intersection_is_watertight_where_the_usual_test_leaks():
    """Rays aimed at the vertices and edges of a **closed** mesh must not escape it.

    This is what the watertight formulation buys, and it is measured rather than asserted. From
    a point inside a closed hull, every ray must cross the boundary; a ray aimed exactly at a
    vertex or an edge midpoint is where an ordinary test can report a hit on neither of the two
    triangles sharing that feature, which is a pinhole through a closed surface.

    Möller-Trumbore, implemented here for the comparison, leaks on several of these. The test
    used by the module leaks on none.
    """
    rng = np.random.default_rng(11)
    points = rng.normal(size=(40, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    hull = ConvexHull(points)
    triangles = points[hull.simplices]

    aims = [points[i] for i in range(len(points))]
    for simplex in hull.simplices:
        for corner in range(3):
            aims.append(0.5 * (points[simplex[corner]] + points[simplex[(corner + 1) % 3]]))
    aims = np.array(aims)
    origins = np.zeros_like(aims)

    def moller_trumbore_leaks():
        first = triangles[:, 0]
        edge_a, edge_b = triangles[:, 1] - first, triangles[:, 2] - first
        direction = aims * 3.0
        perpendicular = np.cross(direction[:, None, :], edge_b[None, :, :])
        determinant = np.sum(edge_a[None, :, :] * perpendicular, axis=-1)
        inverse = 1.0 / np.where(determinant != 0, determinant, np.inf)
        offset = origins[:, None, :] - first[None, :, :]
        u = np.sum(offset * perpendicular, axis=-1) * inverse
        cross = np.cross(offset, edge_a[None, :, :])
        v = np.sum(direction[:, None, :] * cross, axis=-1) * inverse
        distance = np.sum(edge_b[None, :, :] * cross, axis=-1) * inverse
        hit = (determinant != 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & (distance > 0)
        return int((~np.any(hit & (distance <= 1), axis=-1)).sum())

    arguments = (
        jnp.asarray(origins),
        jnp.asarray(aims * 3.0),
        jnp.asarray(triangles),
        jnp.zeros(len(aims)),
    )
    assert int(np.count_nonzero(~np.asarray(segment_is_cut(*arguments)))) == 0
    # And under an outer trace as well. The guarantee rests on two triangles sharing an edge
    # computing exactly opposite edge functions, which a compiler that fuses a multiply into a
    # subtraction destroys -- this fixture leaks six of its rays if the edge function is written
    # as a plain difference of products.
    compiled = jax.jit(lambda *a: segment_is_cut(*a))
    assert int(np.count_nonzero(~np.asarray(compiled(*arguments)))) == 0
    assert moller_trumbore_leaks() > 0, "the fixture no longer separates the two formulations"


def _features(triangles):
    """Every vertex, edge midpoint and face centroid of a triangulation — exactly.

    ⚠️ Not rounded and not deduplicated. These aims are useful only because they land on a
    feature *exactly*; snapping them to a tolerance moves them off it, and a sweep built that way
    reports no leaks whatever the intersection test does.
    """
    midpoints = [0.5 * (triangles[:, k] + triangles[:, (k + 1) % 3]) for k in range(3)]
    return np.concatenate([triangles.reshape(-1, 3), *midpoints, triangles.mean(axis=1)])


def _escaping_rays(body, interior, *, nested):
    """How many rays from inside ``body`` at its own features are not stopped by it.

    ``nested`` wraps the call in a further trace. Note that the unnested arm is **not** an eager
    one: the per-block kernel is traced either way, so what this varies is whether there is an
    outer trace around it, not whether the arithmetic is compiled.
    """
    cut = jax.jit(lambda *a: segment_is_cut(*a)) if nested else segment_is_cut
    aims = _features(body)
    escaped = 0
    for point in interior:
        target = point + (aims - point) * 3.0
        origin = np.broadcast_to(point, target.shape)
        blocked = cut(
            jnp.asarray(origin), jnp.asarray(target), jnp.asarray(body), jnp.zeros(len(aims))
        )
        escaped += int(np.count_nonzero(~np.asarray(blocked)))
    return escaped


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "inside an outer trace"])
@pytest.mark.parametrize(
    "body, interior",
    [
        (closed_prism(L_OUTLINE, 1.0), [[0.4, 0.4, 0.0], [1.5, 0.4, 0.3], [0.4, 1.5, -0.4]]),
        (closed_drum(48), [[0.0, 0.0, 0.0], [0.4, -0.2, 0.5], [-0.3, 0.35, -0.7]]),
    ],
    ids=["L-prism", "drum"],
)
def test_no_ray_escapes_a_closed_body(body, interior, nested):
    """The watertight guarantee on bodies the convex-hull fixture does not reach.

    Two features it adds. The L-prism has a **reflex** edge, where an interior ray leaves
    through a corner the surface turns inward at; the drum has a **curved seam that actually
    meets itself**, built by index from one vertex table rather than from trigonometry evaluated
    twice — a seam assembled the other way is short of closing by a few last bits, and then the
    rays that escape through the slit get blamed on the intersection test.

    Written as a plain difference of products the edge function loses its exact antisymmetry
    once compiled, and these two bodies then leak 6 and 76 rays. Both arms here go through the
    traced block kernel; the second only adds an outer trace around it, which is the shape a
    caller who wraps the whole build in ``jit`` produces.
    """
    assert _escaping_rays(np.asarray(body), np.asarray(interior), nested=nested) == 0


def test_the_edge_function_survives_being_compiled():
    """Woop's inside test needs neighbouring triangles to disagree by an exact sign.

    Two triangles sharing an edge evaluate the same edge function with the two operand pairs
    swapped. Every ray through that edge is claimed by exactly one of them only while the two
    results are exact negatives — and written as ``a * b - c * d`` that stops being true the
    moment a compiler fuses one of the multiplies into the subtraction, because the two
    triangles then keep different products at full precision. The second assertion below is
    what keeps this test honest: it fails if the plain difference has stopped being able to
    break, which would mean this fixture no longer exercises the thing being pinned.
    """
    rng = np.random.default_rng(5)
    first, second, third, fourth = (jnp.asarray(rng.normal(size=20_000)) for _ in range(4))

    compiled = jax.jit(_edge_function)
    forward = np.asarray(compiled(first, second, third, fourth))
    # The operand order the triangle on the other side of the edge sees.
    backward = np.asarray(compiled(third, fourth, first, second))
    assert np.array_equal(forward, -backward)
    # Compiling it must not move the value either, or every distance shifts under tracing.
    assert np.array_equal(forward, np.asarray(_edge_function(first, second, third, fourth)))

    plain = jax.jit(lambda a, b, c, d: a * b - c * d)
    assert not np.array_equal(
        np.asarray(plain(first, second, third, fourth)),
        -np.asarray(plain(third, fourth, first, second)),
    ), "the compiler no longer fuses the difference of products, so this fixture proves nothing"


# ---------------------------------------------------------------------------------------
# Through the gather
# ---------------------------------------------------------------------------------------


def test_a_panel_of_the_same_surface_shadows_what_is_behind_it():
    """The capability. An analytic body cannot express this, because the geometry doing the
    blocking *is* the emitting surface."""
    surfaces = _emitter_and_panel()
    behind = np.array([[2.0, 0.0, 0.0]])
    past_the_edge = np.array([[2.0, 5.0, 0.0]])

    shadowed = build_visibility([], surfaces, behind)
    clear = build_visibility([], surfaces, past_the_edge)
    assert float(direct_fluence_rate(surfaces, behind, visibility=shadowed)[0]) == 0.0
    assert float(direct_fluence_rate(surfaces, past_the_edge, visibility=clear)[0]) > 0.0
    assert float(direct_fluence_rate(surfaces, behind)[0]) > 0.0, "unoccluded, it is lit"


def test_turning_self_occlusion_off_puts_the_light_back():
    surfaces = _emitter_and_panel()
    behind = np.array([[2.0, 0.0, 0.0]])
    ignored = build_visibility([], surfaces, behind, self_occlusion=NoOcclusion())
    assert float(direct_fluence_rate(surfaces, behind, visibility=ignored)[0]) == pytest.approx(
        float(direct_fluence_rate(surfaces, behind)[0]), rel=1e-15
    )


def test_the_surface_s_own_geometry_is_opaque_whatever_transmittance_is_given():
    """Walls and bodies of the emitting set carry no transmittance. A partly transmitting body
    is an analytic primitive instead, and this keeps the two from being confused."""
    surfaces = _emitter_and_panel()
    behind = np.array([[2.0, 0.0, 0.0]])
    mask = build_visibility([], surfaces, behind)
    assert float(direct_fluence_rate(surfaces, behind, visibility=mask, transmittance=[])[0]) == 0.0


def test_a_convex_body_is_completely_unaffected_by_tracing_its_own_triangles():
    """The consistency check that validates both halves at once.

    For a convex emitter the source-side cosine clamp *is* the exact visibility test, so tracing
    the body's own triangles must change nothing at all. It does not: the two answers are
    **bit-identical**, and both reproduce the closed form ``G = (4B/pi) arcsin(R/d)`` to the
    discretization of the fixture. Had the tracer produced spurious self-hits — acne on facets
    adjacent to the source — this is where they would show.
    """
    exitance, radius, distance = 3.0, 1.0, 2.0
    surfaces = Surfaces.from_triangles(
        cylinder_triangles(radius, half_length=20.0, sectors=48, slices=48), emission=exitance
    )
    probe = np.array([[distance, 0.0, 0.0]])
    mask = build_visibility([], surfaces, probe)

    clamp_only = float(direct_fluence_rate(surfaces, probe)[0])
    with_tracing = float(direct_fluence_rate(surfaces, probe, visibility=mask)[0])
    closed_form = (4.0 * exitance / np.pi) * np.arcsin(radius / distance)

    assert with_tracing == clamp_only
    assert clamp_only == pytest.approx(closed_form, rel=3e-3)


def test_a_flat_plate_does_not_shadow_itself():
    """Acne: neighbouring facets sharing an edge with the source must not block it.

    A plate seen from in front is entirely visible. If the near-origin exclusion were missing,
    or the self-exclusion were by tolerance rather than by index, a fraction of the facets would
    drop out and the plate would simply be dimmer, with nothing to say so.
    """
    plate = np.concatenate(
        [
            rectangle_triangles([x, y, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0])
            for x in np.linspace(-0.4, 0.4, 9)
            for y in np.linspace(-0.4, 0.4, 9)
        ]
    )
    surfaces = Surfaces.from_triangles(plate, emission=100.0)
    probe = np.array([[0.0, 0.0, 1.0]])
    mask = build_visibility([], surfaces, probe)
    assert float(direct_fluence_rate(surfaces, probe, visibility=mask)[0]) == pytest.approx(
        float(direct_fluence_rate(surfaces, probe)[0]), rel=1e-15
    )


def test_a_bent_duct_does_not_light_its_own_far_leg():
    """The shape the headline reactor has, and the reason self-occlusion is not optional.

    Two panels meeting at a right angle, with an emitter on the inside of one. A receiver
    tucked behind the second panel is out of sight of the emitter even though nothing but the
    duct's own wall is between them — exactly the shadowing the summation and view-factor models
    are unable to represent.
    """
    # The floor of the bend, with a small emitter sitting on it facing up, and the outer wall
    # rising from the corner. Floor and wall are one body; so is the emitter.
    emitter = rectangle_triangles([0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0])
    floor = rectangle_triangles([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    wall = rectangle_triangles([1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    surfaces = Surfaces.from_triangles(
        np.concatenate([emitter, floor, wall]),
        emission=[500.0, 500.0] + [0.0] * 4,
    )
    hidden = np.array([[2.0, 0.0, 1.0]])
    visible = np.array([[0.5, 0.0, 1.0]])

    assert (
        float(
            direct_fluence_rate(
                surfaces, hidden, visibility=build_visibility([], surfaces, hidden)
            )[0]
        )
        == 0.0
    )
    assert (
        float(
            direct_fluence_rate(
                surfaces, visible, visibility=build_visibility([], surfaces, visible)
            )[0]
        )
        > 0.0
    )


# ---------------------------------------------------------------------------------------
# Receivers that are themselves facets
# ---------------------------------------------------------------------------------------


def _facing_plates(*heights, half=0.5):
    """Square plates stacked along z, two triangles each, all in their own plane."""
    return Surfaces.from_triangles(
        np.concatenate(
            [
                rectangle_triangles([0.0, 0.0, z], [half, 0.0, 0.0], [0.0, half, 0.0])
                for z in heights
            ]
        )
    )


def test_a_ray_aimed_at_a_facet_is_not_blocked_by_that_facet():
    """The whole point of ``receiver_facet``, and the defect it closes.

    A segment between two facet centroids ends *exactly* in the target facet's plane, and a hit
    at the far endpoint counts. ``offset_scale`` guards the near end and there is nothing
    guarding the far one, so without the target's index every pair of facets that can see each
    other reads as blocked. Two plates facing one another across empty space: there is nothing
    between them, and the answer must be no shadow anywhere.
    """
    surfaces = _facing_plates(-1.0, 1.0)
    centroids = np.asarray(surfaces.centroid)
    mask = build_visibility(
        [],
        surfaces,
        centroids,
        receiver_facet=np.arange(surfaces.n_facets),
        self_occlusion=RayCastOcclusion(),
    )
    assert not bool(jnp.any(mask.hidden_by_geometry)), np.asarray(mask.hidden_by_geometry)


def test_without_the_target_index_the_same_scene_is_entirely_shadowed():
    """The failure this guards against, pinned so the parameter cannot be quietly dropped.

    Recorded as a *property of omitting it* rather than as a bug: omitting ``receiver_facet``
    is right for receivers out in the volume, and this is what it does when the receivers are
    facets instead.
    """
    surfaces = _facing_plates(-1.0, 1.0)
    mask = build_visibility(
        [], surfaces, np.asarray(surfaces.centroid), self_occlusion=RayCastOcclusion()
    ).hidden_by_geometry
    cross = np.asarray(mask)[2:, :2]
    assert cross.all(), "the endpoint hit should shadow every cross pair"


def test_a_facet_genuinely_behind_another_is_still_blocked():
    """Excluding the target must not disable self-occlusion, only stop it misfiring.

    Three parallel plates, the middle one four times the width of the others: the outer two
    cannot see each other through it, while each of them can see the middle one directly.
    """
    surfaces = Surfaces.from_triangles(
        np.concatenate(
            [
                rectangle_triangles([0.0, 0.0, -1.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]),
                rectangle_triangles([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]),
                rectangle_triangles([0.0, 0.0, 1.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]),
            ]
        )
    )
    mask = np.asarray(
        build_visibility(
            [],
            surfaces,
            np.asarray(surfaces.centroid),
            receiver_facet=np.arange(6),
            self_occlusion=RayCastOcclusion(),
        ).hidden_by_geometry
    )
    assert mask[4:, :2].all(), "the middle plate must hide the outer two from each other"
    assert mask[:2, 4:].all(), "and symmetrically"
    assert not mask[2:4, :2].any(), "the middle plate is directly visible from the lower one"
    assert not mask[:2, 2:4].any()


def test_a_two_column_exclusion_ignores_both_triangles():
    """``segment_is_cut`` takes one index per ray or several; a ray between two facets needs
    two, and a one-column exclusion is the same thing it always was."""
    surfaces = _facing_plates(-1.0, 1.0)
    centroids = np.asarray(surfaces.centroid)
    origin = np.repeat(centroids[:2], 2, axis=0)
    target = np.tile(centroids[2:], (2, 1))
    source = np.repeat(np.arange(2), 2)
    receiver = np.tile(np.arange(2, 4), 2)
    near = np.full(4, 1e-9)

    both = segment_is_cut(
        origin, target, surfaces.vertices, near, exclude=np.stack([source, receiver], axis=-1)
    )
    only_source = segment_is_cut(origin, target, surfaces.vertices, near, exclude=source)
    assert not bool(jnp.any(both)), "nothing stands between the plates"
    assert bool(jnp.all(only_source)), "the target facet is hit at the far endpoint"


@pytest.mark.parametrize("occluders", [[], [Cylinder([0, 0, 0], [1, 0, 0], 0.1, 4.0)]])
def test_a_visibility_with_no_receivers_at_all_builds(occluders):
    """A surface-only study asks for no receivers, and the chunk loop then runs no passes —
    which used to leave nothing to concatenate and raise rather than return an empty mask.

    Parametrized over having a body and not because the two arrays are guarded separately, and
    a fixture with no occluder leaves the analytic one's guard unexercised: that gap survived
    the first version of this test and was found by mutation, not by reading it.
    """
    surfaces = _facing_plates(-1.0, 1.0)
    mask = build_visibility(
        occluders, surfaces, np.zeros((0, 3)), self_occlusion=RayCastOcclusion()
    )
    assert mask.hidden_by_geometry.shape == (0, 4)
    assert mask.blocked.shape == (len(occluders), 0, 4)


# ---------------------------------------------------------------------------------------
# Culling the candidates with a grid
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("grid", [True, 4, (3, 5, 7)])
def test_the_grid_changes_what_the_ray_test_COSTS_and_not_what_it_ANSWERS(grid):
    """The contract of the acceleration, on a body that genuinely shadows itself.

    A grid decides which triangles a segment is worth testing; it must never decide whether one
    blocks. So the two paths are compared bit for bit, with the body's own facet centroids as
    receivers, at three resolutions including one deliberately mismatched to the shape.
    """
    # ⚠️ A CONVEX body does not shadow itself: every sight line between two of its interior
    # facets stays inside it and meets nothing, so a drum alone compares 0 against 0. The
    # partition across the middle is what puts geometry between facets -- and the
    # one-sidedness check below is what caught the drum-only version of this test.
    drum = closed_drum(48, radius=1.0, half_height=1.0)
    partition = rectangle_triangles([0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [0.0, 0.0, 0.95])
    surfaces = Surfaces.from_triangles(np.concatenate([drum, partition]), emission=1.0)
    receivers = np.asarray(surfaces.centroid)
    facet_of = np.arange(len(receivers))
    near = 1e-6 * np.sqrt(np.asarray(surfaces.area))

    plain = RayCastOcclusion().field(surfaces, receivers, near, facet_of)
    culled = RayCastOcclusion(grid=grid).field(surfaces, receivers, near, facet_of)
    np.testing.assert_array_equal(np.asarray(culled.fraction), np.asarray(plain.fraction))
    blocked = np.asarray(plain.fraction) > 0.0
    assert 0.2 < blocked.mean() < 0.9, f"fixture is one-sided: {blocked.mean()}"
