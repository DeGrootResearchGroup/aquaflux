"""Bodies in the way: the intersection tests, the frozen mask, and what stays differentiable."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation.gather import direct_fluence_rate
from aquaflux.radiation.occluders import Cylinder, HalfSpace
from aquaflux.radiation.profiles import Isotropic
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.visibility import Visibility, build_visibility

POWER = 100.0
NO_OFFSET = jnp.zeros(())


def point_source(position=(0.0, 0.0, 0.0), power=POWER):
    positions = np.atleast_2d(np.asarray(position, dtype=float))
    return Surfaces.from_triangles(
        np.repeat(positions[:, None, :], 3, axis=1), power=power, profiles=(Isotropic(),)
    )


def sleeve(radius=0.5, centre=(2.0, 0.0, 0.0), half_length=3.0):
    return Cylinder(centre=centre, axis=[0.0, 0.0, 1.0], radius=radius, half_length=half_length)


# ---------------------------------------------------------------------------------------
# The primitives
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "finish", "expected", "what"),
    [
        ((-3.0, 0.0, 0.0), (3.0, 0.0, 0.0), True, "straight through"),
        ((-3.0, 2.0, 0.0), (3.0, 2.0, 0.0), False, "wide of it"),
        ((-3.0, 0.0, 5.0), (3.0, 0.0, 5.0), False, "over the end cap"),
        ((0.0, 0.0, 9.0), (0.0, 0.0, -9.0), True, "down the axis"),
        ((-3.0, 0.999, 0.0), (3.0, 0.999, 0.0), True, "grazing, just inside"),
        ((-3.0, 1.001, 0.0), (3.0, 1.001, 0.0), False, "grazing, just outside"),
        ((-3.0, 0.0, 0.0), (-2.0, 0.0, 0.0), False, "stops short of it"),
        ((5.0, 0.0, 0.0), (9.0, 0.0, 0.0), False, "starts past it"),
    ],
)
def test_a_cylinder_blocks_what_passes_through_it(start, finish, expected, what):
    body = Cylinder(centre=[0, 0, 0], axis=[0, 0, 1], radius=1.0, half_length=2.0)
    assert bool(body.blocks(jnp.asarray(start), jnp.asarray(finish), NO_OFFSET)) is expected, what


def test_the_grazing_discriminant_is_formed_the_way_that_survives_it():
    """The reformulation earns its place, measured rather than asserted.

    Taking the discriminant as ``b^2 - a c`` subtracts two nearly equal numbers exactly when a
    ray almost touches the surface. Across eighteen near-tangential cases — three source
    distances, offsets a few parts in ``1e9`` to ``1e13`` either side of the radius — the naive
    form misclassifies **eight**, and every one of them in the dangerous direction: it reports a
    discriminant of exactly zero for a ray that does pass inside, so light leaks through a body
    that should stop it. The form this module uses gets all eighteen right.

    This is not a corner case. A sleeve sits at essentially the radius of the lamp facets it
    surrounds, so nearly tangential rays are the ordinary geometry here.
    """
    radius = 1.0
    body = Cylinder(centre=[0, 0, 0], axis=[0, 0, 1], radius=radius, half_length=10.0)

    def naive_says_blocked(height, distance):
        offset = np.array([-distance, height, 0.0])
        direction = np.array([2.0 * distance, 0.0, 0.0])
        a = direction @ direction
        b = offset @ direction
        c = offset @ offset - radius**2
        return b * b - a * c > 0.0

    wrong_naive = wrong_robust = 0
    for distance in (1e3, 1e6, 1e8):
        for delta in (1e-9, 1e-11, 1e-13, -1e-9, -1e-11, -1e-13):
            height = radius * (1.0 + delta)
            truth = delta < 0.0
            start = jnp.asarray([-distance, height, 0.0])
            finish = jnp.asarray([distance, height, 0.0])
            wrong_robust += bool(body.blocks(start, finish, NO_OFFSET)) is not truth
            wrong_naive += naive_says_blocked(height, distance) is not truth
    assert wrong_robust == 0
    assert wrong_naive >= 6, "the fixture no longer separates the two formulations"


def test_a_facet_lying_on_a_body_does_not_shadow_itself():
    """A fixed epsilon fails at both ends; the exclusion is a fraction of the facet's own size."""
    body = Cylinder(centre=[0, 0, 0], axis=[0, 0, 1], radius=1.0, half_length=2.0)
    on_the_surface = jnp.asarray([1.0, 0.0, 0.0])
    outward = jnp.asarray([3.0, 0.0, 0.0])
    assert bool(body.blocks(on_the_surface, outward, jnp.asarray(1e-9))) is False


def test_a_half_space_swallows_whatever_is_on_its_solid_side():
    """A cell on the far side of a wall is inside metal, and reporting light there would be
    reporting light inside metal."""
    wall = HalfSpace(point=[0, 0, 0], normal=[0, 0, 1])
    above, below = jnp.asarray([0.0, 0.0, 1.0]), jnp.asarray([1.0, 0.0, -2.0])
    assert bool(wall.blocks(above, jnp.asarray([1.0, 0.0, 2.0]), NO_OFFSET)) is False
    assert bool(wall.blocks(above, below, NO_OFFSET)) is True
    assert bool(wall.contains(below)) is True
    assert bool(wall.contains(above)) is False


def test_the_intersection_tests_broadcast():
    body = sleeve()
    rng = np.random.default_rng(0)
    starts = jnp.asarray(rng.uniform(-1.0, 1.0, (5, 1, 3)))
    finishes = jnp.asarray(rng.uniform(3.0, 5.0, (1, 4, 3)))
    assert body.blocks(starts, finishes, jnp.zeros((5, 4))).shape == (5, 4)
    assert body.contains(starts).shape == (5, 1)


# ---------------------------------------------------------------------------------------
# The mask
# ---------------------------------------------------------------------------------------


def test_the_shadow_boundary_lands_where_the_tangent_says_it_should():
    """Case 8b: a point source, an offset cylinder, and a probe line across the edge.

    The shadow of a cylinder of radius ``R`` seen from a point source at distance ``d`` is
    bounded by the tangent lines, at a half-angle of ``arcsin(R/d)``. The probe line crosses
    that edge, and the boundary is asserted to within the probe spacing — the only tolerance
    a discrete probe line can support.

    The unoccluded side must also still reproduce the inverse-square law *exactly*: a mask that
    dimmed the lit region as well would place the edge correctly and still be wrong.
    """
    radius, distance = 0.5, 2.0
    source = point_source()
    body = sleeve(radius=radius, centre=(distance, 0.0, 0.0), half_length=9.0)

    probe_x = 4.0
    edge = probe_x * np.tan(np.arcsin(radius / distance))
    offsets = np.linspace(0.0, 2.0 * edge, 201)
    spacing = offsets[1] - offsets[0]
    probes = np.stack([np.full_like(offsets, probe_x), offsets, np.zeros_like(offsets)], axis=1)

    mask = build_visibility([body], source, probes)
    field = np.asarray(direct_fluence_rate(source, probes, visibility=mask, transmittance=[0.0]))

    lit = np.flatnonzero(field > 0.0)
    assert len(lit) > 0
    measured_edge = offsets[lit[0]]
    assert measured_edge == pytest.approx(edge, abs=spacing)

    radii = np.linalg.norm(probes[lit], axis=1)
    np.testing.assert_allclose(field[lit], POWER / (4.0 * np.pi * radii**2), rtol=1e-14)


def test_an_opaque_body_removes_the_source_entirely():
    source = point_source()
    probes = np.array([[4.0, 0.0, 0.0]])
    mask = build_visibility([sleeve()], source, probes)
    assert (
        float(direct_fluence_rate(source, probes, visibility=mask, transmittance=[0.0])[0]) == 0.0
    )


def test_a_perfectly_transmitting_body_changes_nothing():
    source = point_source()
    probes = np.array([[4.0, 0.0, 0.0]])
    mask = build_visibility([sleeve()], source, probes)
    assert float(direct_fluence_rate(source, probes, visibility=mask, transmittance=[1.0])[0]) == (
        pytest.approx(float(direct_fluence_rate(source, probes)[0]), rel=1e-15)
    )


def test_overlapping_bodies_multiply():
    """Two sleeves in a row transmit the product, not the smaller of the two."""
    source = point_source()
    probes = np.array([[6.0, 0.0, 0.0]])
    bodies = [sleeve(centre=(2.0, 0.0, 0.0)), sleeve(centre=(4.0, 0.0, 0.0))]
    mask = build_visibility(bodies, source, probes)
    clear = float(direct_fluence_rate(source, probes)[0])
    through = float(
        direct_fluence_rate(source, probes, visibility=mask, transmittance=[0.5, 0.25])[0]
    )
    assert through == pytest.approx(clear * 0.5 * 0.25, rel=1e-14)


def test_a_mask_with_no_bodies_blocks_nothing():
    source = point_source()
    probes = np.array([[4.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    mask = build_visibility([], source, probes)
    np.testing.assert_allclose(
        np.asarray(direct_fluence_rate(source, probes, visibility=mask)),
        np.asarray(direct_fluence_rate(source, probes)),
        rtol=1e-15,
    )


def test_a_mask_defaults_to_opaque_rather_than_to_clear():
    """A mask supplied without transmittances must block. Defaulting the other way would make
    a forgotten argument look like a working occlusion model that happens to do nothing."""
    source = point_source()
    probes = np.array([[4.0, 0.0, 0.0]])
    mask = build_visibility([sleeve()], source, probes)
    assert float(direct_fluence_rate(source, probes, visibility=mask)[0]) == 0.0


def test_a_mask_built_for_other_receivers_is_refused():
    """The mask is indexed by receiver, so the wrong one puts every shadow in the wrong place
    and raises nothing of its own."""
    source = point_source()
    mask = build_visibility([sleeve()], source, np.array([[4.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="built for different receivers"):
        direct_fluence_rate(
            source, np.array([[4.0, 1.0, 0.0]]), visibility=mask, transmittance=[0.0]
        )


def test_transmittance_without_a_mask_is_refused():
    with pytest.raises(ValueError, match="without a visibility mask"):
        direct_fluence_rate(point_source(), np.array([[4.0, 0.0, 0.0]]), transmittance=[0.5])


@pytest.mark.parametrize("what", ["facet", "receiver"])
def test_a_point_inside_a_body_is_refused(what):
    """Embedded in the solid, not shadowed by it: nothing computed there means anything."""
    inside = (2.0, 0.0, 0.0)
    source = point_source(inside if what == "facet" else (0.0, 0.0, 0.0))
    probes = np.array([list(inside)]) if what == "receiver" else np.array([[4.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match=f"{what}.*lie inside occluder"):
        build_visibility([sleeve()], source, probes)


# ---------------------------------------------------------------------------------------
# The differentiability contract
# ---------------------------------------------------------------------------------------


def test_the_gradient_reaches_a_body_s_transmittance():
    """The live half of the split, and the one a design study varies."""
    source = point_source()
    probes = np.array([[4.0, 0.0, 0.0], [5.0, 0.2, 0.0]])
    mask = build_visibility([sleeve(half_length=9.0)], source, probes)

    def total(transmittance):
        return jnp.sum(
            direct_fluence_rate(source, probes, visibility=mask, transmittance=transmittance)
        )

    step = 1e-6
    base = jnp.asarray([0.4])
    finite_difference = (float(total(base + step)) - float(total(base - step))) / (2.0 * step)
    gradient = float(jax.grad(total)(base)[0])
    assert gradient == pytest.approx(finite_difference, rel=1e-8)
    assert gradient > 0.0


def test_the_gradient_with_respect_to_a_body_s_GEOMETRY_is_exactly_zero():
    """Stated as a contract rather than discovered as a surprise.

    Whether a body lies across a segment is a step function of where that body is: nothing
    changes as it moves until a shadow edge sweeps past a receiver, and then the answer jumps.
    Automatic differentiation of the function actually implemented therefore returns zero, and
    that is the correct derivative *of that function* — the continuum quantity it approximates
    has a boundary term that collapsing the emitter to a point destroys.

    Zero, not small: the mask is frozen, so the radius never enters the traced computation at
    all.
    """
    source = point_source()
    probes = np.array([[4.0, 0.0, 0.0], [4.0, 1.05, 0.0]])

    def total(radius):
        body = Cylinder(centre=[2.0, 0.0, 0.0], axis=[0, 0, 1], radius=radius, half_length=9.0)
        mask = build_visibility([body], source, probes)
        return jnp.sum(direct_fluence_rate(source, probes, visibility=mask, transmittance=[0.2]))

    assert float(jax.grad(total)(jnp.asarray(0.5))) == 0.0
    # And it is a staircase, not a constant: the value does move, in steps, as the body grows.
    assert float(total(jnp.asarray(0.4))) != float(total(jnp.asarray(0.6)))


def test_the_mask_is_a_frozen_array_and_not_recomputed_per_call():
    source = point_source()
    probes = np.array([[4.0, 0.0, 0.0]])
    mask = build_visibility([sleeve()], source, probes)
    assert isinstance(mask, Visibility)
    assert mask.blocked.dtype == jnp.bool_
    assert mask.blocked.shape == (1, 1, 1)
    assert mask.n_occluders == 1


# ---------------------------------------------------------------------------------------
# Streaming the mask instead of holding it
# ---------------------------------------------------------------------------------------


def _probe_line(n: int) -> np.ndarray:
    """Receivers strung past the sleeve, so most of them are shadowed and some are not."""
    return np.stack([np.full(n, 4.0), np.linspace(-1.5, 1.5, n), np.zeros(n)], axis=1)


@pytest.mark.parametrize("transmittance", [None, [0.25]])
def test_streaming_the_bodies_gives_what_a_built_mask_gives(transmittance):
    """The same arithmetic either way: streaming changes when the mask exists, not what it says."""
    source, probes = point_source(), _probe_line(37)
    built = build_visibility([sleeve()], source, probes)
    held = direct_fluence_rate(source, probes, visibility=built, transmittance=transmittance)
    streamed = direct_fluence_rate(
        source, probes, occluders=[sleeve()], transmittance=transmittance, chunk_size=8
    )
    np.testing.assert_array_equal(np.asarray(streamed), np.asarray(held))
    assert float(np.asarray(held).min()) < float(np.asarray(held).max()), "nothing is shadowed"


def test_streaming_never_builds_a_mask_wider_than_a_chunk(monkeypatch):
    """The memory claim, pinned mechanically rather than by timing or by peak RSS.

    A mask is ``receivers x facets`` per body, so what bounds it is the number of receivers each
    build is handed. At mesh scale that is the difference between tens of gigabytes and a few
    hundred megabytes, and it is invisible in the answer -- which is why this checks the calls
    rather than the field.
    """
    from aquaflux.radiation import gather

    handed = []
    real = gather.build_visibility

    def watched(occluders, surfaces, points, **options):
        handed.append(np.asarray(points).shape[0])
        return real(occluders, surfaces, points, **options)

    monkeypatch.setattr(gather, "build_visibility", watched)
    source, probes = point_source(), _probe_line(37)
    direct_fluence_rate(source, probes, occluders=[sleeve()], chunk_size=8)
    assert handed == [8, 8, 8, 8, 5], handed


def test_streaming_with_no_bodies_still_streams_the_surface_s_own_shadowing():
    """An empty sequence is a scene with nothing but the emitting surface in it, which shadows
    itself -- not a scene with the mask switched off."""
    from aquaflux.radiation.self_occlusion import NoOcclusion

    panel = Surfaces.from_triangles(
        np.concatenate(
            [
                np.array([[[0.0, -0.2, -0.2], [0.0, 0.2, -0.2], [0.0, 0.0, 0.2]]]),
                np.array([[[1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 0.0, 1.0]]]),
            ]
        ),
        emission=[1000.0, 0.0],
    )
    probes = np.array([[3.0, 0.0, 0.0], [3.0, 2.0, 0.0]])
    shadowed = direct_fluence_rate(panel, probes, occluders=[], chunk_size=1)
    clear = direct_fluence_rate(panel, probes, occluders=[], self_occlusion=NoOcclusion())
    assert float(shadowed[0]) == 0.0, "the panel in the way should hide the emitter"
    assert float(clear[0]) > 0.0, "with self-occlusion off the emitter is visible again"
    np.testing.assert_allclose(shadowed[1], clear[1], rtol=1e-12)


def test_a_built_mask_and_the_bodies_together_are_refused():
    source, probes = point_source(), _probe_line(4)
    with pytest.raises(ValueError, match="not both"):
        direct_fluence_rate(
            source, probes, visibility=build_visibility([sleeve()], source, probes),
            occluders=[sleeve()],
        )  # fmt: skip
