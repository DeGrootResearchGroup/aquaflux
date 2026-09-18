"""The vacuum gather, against the closed forms the ultraviolet literature validates against.

Each case is chosen so a specific wrong answer turns it red. Several are deliberately paired:
a case at one distance cannot separate an inverse-square error from a constant, a case on axis
cannot see a missing receiver cosine, and a case with one source cannot see the difference
between fluence rate and irradiance at all.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation.gather import fluence_rate, irradiance
from aquaflux.radiation.profiles import CosinePower, Isotropic, Lambertian
from aquaflux.radiation.surfaces import Surfaces

from tests.unit.radiation_references import (
    cylinder_triangles,
    disc_triangles,
    finite_line_fluence_rate,
    rectangle_triangles,
)

POWER = 100.0


def point_source(positions, power=POWER):
    """Zero-area facets carrying radiant power, at the given positions."""
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    vertices = np.repeat(positions[:, None, :], 3, axis=1)
    return Surfaces.from_triangles(vertices, power=power, profiles=(Isotropic(),))


# ---------------------------------------------------------------------------------------
# Point sources
# ---------------------------------------------------------------------------------------


def test_a_point_source_falls_off_as_the_inverse_square_of_distance():
    """Case 1. Swept over distance, so an ``r`` instead of ``r^2`` and a ``pi`` instead of a
    ``4 pi`` must each turn it red — one radius alone could not separate them."""
    radii = np.array([0.5, 1.0, 2.0, 7.0])
    points = np.stack([radii, np.zeros_like(radii), np.zeros_like(radii)], axis=1)
    measured = np.asarray(fluence_rate(point_source([0.0, 0.0, 0.0]), points))
    np.testing.assert_allclose(measured, POWER / (4.0 * np.pi * radii**2), rtol=1e-14)


def test_point_sources_add():
    """Nothing in the model couples sources, so the gather must be linear in them."""
    left, right = point_source([-1.0, 0.0, 0.0]), point_source([1.0, 0.0, 0.0])
    both = point_source([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    probe = np.array([[0.3, 0.4, 0.0]])
    assert float(fluence_rate(both, probe)[0]) == pytest.approx(
        float(fluence_rate(left, probe)[0]) + float(fluence_rate(right, probe)[0]), rel=1e-14
    )


@pytest.mark.parametrize("angle", [0.0, 0.3, 0.9, 1.4])
def test_irradiance_is_the_fluence_rate_times_the_receiver_cosine_for_one_source(angle):
    """Case 2, swept over angle. At zero a missing receiver cosine passes, so it is not enough
    to test facing the source.

    The geometry is a **fixed radius with a tilted normal**, not a plane at fixed perpendicular
    distance -- on a plane the published result is ``cos^3``, because the slant range grows too.
    """
    source = point_source([0.0, 0.0, 0.0])
    probe = np.array([[1.0, 0.0, 0.0]])
    normal = np.array([[-np.cos(angle), -np.sin(angle), 0.0]])
    assert float(irradiance(source, probe, normal)[0]) == pytest.approx(
        float(fluence_rate(source, probe)[0]) * np.cos(angle), rel=1e-13
    )


def test_a_receiver_facing_away_from_a_point_source_is_not_illuminated():
    source = point_source([0.0, 0.0, 0.0])
    probe = np.array([[1.0, 0.0, 0.0]])
    assert float(irradiance(source, probe, np.array([[1.0, 0.0, 0.0]]))[0]) == 0.0


def test_the_cosine_identity_fails_once_there_is_more_than_one_source():
    """The warning in the module, made a test so nobody re-derives ``E = G cos`` as general.

    Two sources on opposite sides of a receiver: their fluence rates add while their
    irradiances oppose, so the ratio is not a cosine of anything.
    """
    both = point_source([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    probe, normal = np.array([[0.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])
    total_fluence = float(fluence_rate(both, probe)[0])
    total_irradiance = float(irradiance(both, probe, normal)[0])
    assert total_fluence == pytest.approx(2.0 * POWER / (4.0 * np.pi), rel=1e-14)
    assert total_irradiance == pytest.approx(POWER / (4.0 * np.pi), rel=1e-14)


# ---------------------------------------------------------------------------------------
# Line sources, by multiple point source summation
# ---------------------------------------------------------------------------------------


def _line_source(half_length, power_per_length, count):
    """A line source as the field builds one: a row of equally spaced isotropic points."""
    edges = np.linspace(-half_length, half_length, count + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    positions = np.stack([np.zeros(count), np.zeros(count), centres], axis=1)
    return point_source(positions, power=power_per_length * 2.0 * half_length / count)


def test_a_summed_line_source_converges_on_the_closed_form_at_second_order():
    """Case 7. The closed form is exact; the *sum* is a midpoint rule, so the thing to assert
    is the rate, not agreement at one count.

    Asserting agreement at a single count either passes trivially at a large one or has to be
    given a tolerance nobody can justify. The rate is the real property, and it is what tells
    you how many segments a lamp needs.
    """
    radius, half_length = 0.05, 0.4
    reference = finite_line_fluence_rate(1.0, half_length, radius)
    errors = []
    for count in (40, 80, 160, 320):
        measured = float(
            fluence_rate(_line_source(half_length, 1.0, count), np.array([[radius, 0.0, 0.0]]))[0]
        )
        errors.append(abs(measured - reference) / reference)
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(3.5 < ratio < 4.5 for ratio in ratios), f"not second order: {ratios}"


def test_a_summed_line_source_is_not_second_order_until_its_segments_resolve_the_distance():
    """Where the rate above comes from, and the rule for choosing a segment count.

    The sum only behaves like a quadrature once the spacing between point sources is smaller
    than the receiver's distance from the line; nearer than that, the discrete sources are
    individually resolved and the error is governed by the nearest one rather than by the rule.
    At a segment count of 10 the spacing is 0.08 against a distance of 0.05 and the error is
    4%; doubling it to 20 improves matters 54-fold, which is not a rate and must not be read as
    one. This is why a lamp needs hundreds of segments to be accurate *at its own sleeve*, and
    far fewer to be accurate across the reactor.
    """
    radius, half_length = 0.05, 0.4
    reference = finite_line_fluence_rate(1.0, half_length, radius)

    def error(count):
        measured = float(
            fluence_rate(_line_source(half_length, 1.0, count), np.array([[radius, 0.0, 0.0]]))[0]
        )
        return abs(measured - reference) / reference

    assert 2.0 * half_length / 10 > radius, "the coarse fixture no longer under-resolves"
    assert error(10) / error(20) > 10.0


def test_a_long_summed_line_source_approaches_the_infinite_limit():
    """Case 6 in vacuum: ``G -> P' / (4 r)``, the limit the radial and MPSS lamp models differ
    on by a factor of ``pi / 2``."""
    radius = 0.05
    measured = float(
        fluence_rate(_line_source(50.0, 1.0, 20000), np.array([[radius, 0.0, 0.0]]))[0]
    )
    assert measured == pytest.approx(1.0 / (4.0 * radius), rel=2e-3)


# ---------------------------------------------------------------------------------------
# Areal facets
# ---------------------------------------------------------------------------------------


def test_a_small_lambertian_facet_approaches_the_inverse_square_law_at_second_order():
    """Case 3. The far-field limit is ``G = M A / (pi r^2)``, and the approach to it is the
    error of the point approximation this module exists to avoid -- second order in the ratio
    of facet width to distance, so it is 331% at a quarter of a width and 1% at five."""
    half = 0.005
    exitance = 1000.0
    facet = rectangle_triangles([0.0, 0.0, 0.0], [half, 0.0, 0.0], [0.0, half, 0.0])
    surfaces = Surfaces.from_triangles(facet, emission=exitance)
    area = (2.0 * half) ** 2
    errors = []
    for distance in (0.05, 0.5, 5.0):
        far_field = exitance * area / (np.pi * distance**2)
        measured = float(fluence_rate(surfaces, np.array([[0.0, 0.0, distance]]))[0])
        errors.append(abs(measured - far_field) / far_field)
    # The relative error is second order in facet width over distance, so a tenfold increase
    # in distance divides it by a hundred. Measured against the *absolute* error the rate is
    # fourth order, which is the same statement and an easy one to quote by mistake.
    assert errors[0] / errors[1] == pytest.approx(100.0, rel=0.1)
    assert errors[1] / errors[2] == pytest.approx(100.0, rel=0.1)
    assert errors[0] == pytest.approx(0.25 * (2.0 * half / 0.05) ** 2, rel=0.05)


def test_a_facet_cannot_illuminate_what_is_behind_it():
    """The clamp, at its simplest. Without it a facet emits from its back as brightly as its
    front and every enclosure is twice as bright as it should be."""
    facet = rectangle_triangles([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0])
    surfaces = Surfaces.from_triangles(facet, emission=1.0)
    in_front = float(fluence_rate(surfaces, np.array([[0.0, 0.0, 1.0]]))[0])
    behind = float(fluence_rate(surfaces, np.array([[0.0, 0.0, -1.0]]))[0])
    assert in_front > 0.0
    assert behind == 0.0


@pytest.mark.parametrize(("radius", "height"), [(1.0, 1.0), (1.0, 0.25), (4.0, 0.5)])
def test_a_lambertian_disc_reproduces_its_closed_form_on_axis(radius, height):
    """Case 4, exact at every height rather than only far away: ``G = 2B[1 - h/sqrt(a^2+h^2)]``
    and ``E = B a^2 / (a^2 + h^2)``.

    The two together are the strongest pair here, because they separate the two kernels on one
    geometry: the same disc, the same receiver, and answers that differ by the receiver cosine.
    The tolerance is set by the polygon's area deficit, not by the gather.
    """
    exitance = 3.0
    surfaces = Surfaces.from_triangles(disc_triangles(radius), emission=exitance)
    probe = np.array([[0.0, 0.0, height]])
    facing = np.array([[0.0, 0.0, -1.0]])
    expected_fluence = 2.0 * exitance * (1.0 - height / np.sqrt(radius**2 + height**2))
    expected_irradiance = exitance * radius**2 / (radius**2 + height**2)
    assert float(fluence_rate(surfaces, probe)[0]) == pytest.approx(expected_fluence, rel=2e-3)
    assert float(irradiance(surfaces, probe, facing)[0]) == pytest.approx(
        expected_irradiance, rel=2e-3
    )


def test_refining_the_disc_reduces_the_discretization_error():
    """Separates a discretization error from a modelling error: only the first one refines away."""
    exitance, radius, height = 3.0, 1.0, 0.25
    expected = 2.0 * exitance * (1.0 - height / np.sqrt(radius**2 + height**2))
    errors = []
    for sectors in (16, 32, 64):
        surfaces = Surfaces.from_triangles(
            disc_triangles(radius, rings=12, sectors=sectors), emission=exitance
        )
        measured = float(fluence_rate(surfaces, np.array([[0.0, 0.0, height]]))[0])
        errors.append(abs(measured - expected))
    assert errors[0] > errors[1] > errors[2]


def test_a_large_disc_reaches_the_infinite_plane_limits():
    """``G -> 2B`` and ``E -> B``, so ``G / E -> 2`` becomes a measurable limit rather than a
    number needing slack -- and it is the cleanest statement of what separates the two."""
    exitance = 5.0
    surfaces = Surfaces.from_triangles(
        disc_triangles(400.0, rings=200, sectors=256), emission=exitance
    )
    probe, facing = np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, -1.0]])
    measured_fluence = float(fluence_rate(surfaces, probe)[0])
    measured_irradiance = float(irradiance(surfaces, probe, facing)[0])
    assert measured_fluence == pytest.approx(2.0 * exitance, rel=5e-3)
    assert measured_irradiance == pytest.approx(exitance, rel=5e-3)
    assert measured_fluence / measured_irradiance == pytest.approx(2.0, rel=5e-3)


@pytest.mark.parametrize("ratio", [1.2, 2.0, 5.0])
def test_an_emitting_cylinder_reproduces_its_closed_form(ratio):
    """Case 8, the clamp test. ``G(d) = (4B/pi) arcsin(R/d)`` for a uniform Lambertian tube.

    A cylinder is convex, so the source-side clamp *is* the visibility condition here and the
    closed form is reproduced with no occluder present -- no occluder is needed to test it, and
    none is used.

    The failure modes are exact, and were measured rather than assumed. At ``d/R = 2`` the
    clamped answer is 1.99978 against a reference of 2. Because the clamp is applied as a
    **gate** -- the radiance is a constant in front and zero behind -- both taking the absolute
    value of the cosine and dropping the clamp altogether give 3.99941, exactly double: the far
    side of the tube contributes as brightly as the near side. A third formulation, keeping the
    signed cosine as a *multiplier* rather than a gate, instead gives 1.5e-4, which is zero to
    the discretization: the two sides cancel. That is the reason the sign is a gate here and
    not a factor, and it is worth knowing because the near-zero answer looks like an empty
    scene rather than like a physics error.
    """
    exitance, radius = 3.0, 1.0
    surfaces = Surfaces.from_triangles(
        cylinder_triangles(radius, half_length=200.0), emission=exitance
    )
    measured = float(fluence_rate(surfaces, np.array([[ratio * radius, 0.0, 0.0]]))[0])
    expected = (4.0 * exitance / np.pi) * np.arcsin(1.0 / ratio)
    assert measured == pytest.approx(expected, rel=2e-3)


def test_a_narrower_beam_puts_more_on_axis_and_less_to_the_side_at_equal_exitance():
    """Otherwise the profile reaches the gather normalized but unused."""
    facet = rectangle_triangles([0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0])
    on_axis, off_axis = np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 1.0]])
    diffuse = Surfaces.from_triangles(facet, emission=1.0, profiles=(Lambertian(),))
    narrow = Surfaces.from_triangles(facet, emission=1.0, profiles=(CosinePower(8.0),))
    assert float(fluence_rate(narrow, on_axis)[0]) > float(fluence_rate(diffuse, on_axis)[0])
    assert float(fluence_rate(narrow, off_axis)[0]) < float(fluence_rate(diffuse, off_axis)[0])


def test_sources_of_different_kinds_are_all_summed():
    """The host-side grouping by distribution must not drop a group."""
    facet = rectangle_triangles([0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0])
    vertices = np.concatenate([facet, np.zeros((1, 3, 3))])
    mixed = Surfaces.from_triangles(
        vertices,
        emission=[1.0, 1.0, 0.0],
        power=[0.0, 0.0, POWER],
        profiles=(CosinePower(3.0), Isotropic()),
        profile_index=[0, 0, 1],
    )
    areal = Surfaces.from_triangles(facet, emission=1.0, profiles=(CosinePower(3.0),))
    probe = np.array([[0.0, 0.0, 1.0]])
    assert float(fluence_rate(mixed, probe)[0]) == pytest.approx(
        float(fluence_rate(areal, probe)[0]) + POWER / (4.0 * np.pi), rel=1e-13
    )


# ---------------------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [1, 7, 64, 10_000])
def test_chunking_changes_nothing_about_the_answer(chunk_size):
    """Chunking is a memory strategy; it must not be a numerical one.

    The last chunk is padded rather than shortened so the traced body compiles once, and the
    padding must not leak into the result -- a chunk size that does not divide the receiver
    count is the case that catches it, which is why 37 receivers and a chunk of 7.
    """
    rng = np.random.default_rng(0)
    probes = rng.uniform(0.5, 2.0, (37, 3))
    surfaces = point_source([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
    reference = np.asarray(fluence_rate(surfaces, probes, chunk_size=1_000_000))
    np.testing.assert_allclose(
        np.asarray(fluence_rate(surfaces, probes, chunk_size=chunk_size)), reference, rtol=1e-15
    )


def test_no_receivers_gives_no_answers():
    assert fluence_rate(point_source([0.0, 0.0, 0.0]), np.zeros((0, 3))).shape == (0,)


def test_the_gather_is_differentiable_in_emission_and_in_power():
    """The reason the module exists. A finite difference, not a finiteness check: a severed
    adjoint returns zero, which is finite."""
    facet = rectangle_triangles([0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0])
    vertices = np.concatenate([facet, np.zeros((1, 3, 3))])
    surfaces = Surfaces.from_triangles(
        vertices,
        emission=[1.0, 1.0, 0.0],
        power=[0.0, 0.0, POWER],
        profiles=(Lambertian(), Isotropic()),
        profile_index=[0, 0, 1],
    )
    probe = np.array([[0.05, 0.0, 0.4]])

    def total(scale):
        scaled = surfaces.with_optics(
            emission=jnp.asarray(surfaces.emission) * scale,
            power=jnp.asarray(surfaces.power) * scale,
        )
        return jnp.sum(fluence_rate(scaled, probe))

    step = 1e-6
    finite_difference = (float(total(1.0 + step)) - float(total(1.0 - step))) / (2.0 * step)
    assert float(jax.grad(total)(jnp.asarray(1.0))) == pytest.approx(finite_difference, rel=1e-7)


def test_the_gather_compiles_with_the_geometry_closed_over():
    """The contract: geometry is built, values are passed. Every consumer will be traced.

    Which facets are point sources and which distribution each uses decides the *shape* of the
    traced program, so those cannot be traced themselves. Closing over the surface set and
    substituting only the values is what the built-model boundary will formalize.
    """
    surfaces = point_source([0.0, 0.0, 0.0])
    probes = jnp.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    compiled = jax.jit(lambda power: fluence_rate(surfaces.with_optics(power=power), probes))(
        jnp.asarray(POWER)
    )
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(fluence_rate(surfaces, probes)))


def test_passing_the_whole_surface_set_as_a_traced_argument_says_why_it_cannot_work():
    """A bare tracer error from inside the partition would send a reader looking in the wrong
    place; the message names the fix instead."""
    surfaces = point_source([0.0, 0.0, 0.0])
    probes = jnp.asarray([[1.0, 0.0, 0.0]])
    with pytest.raises(TypeError, match="profile index must be concrete"):
        jax.jit(lambda s, p: fluence_rate(s, p))(surfaces, probes)


def test_a_source_can_be_moved_under_a_gradient():
    """Where a lamp should go is the question a design study asks, and the answer is a
    derivative with respect to its position.

    This is what separating the point-source label from the area buys. With the kind of a
    source inferred from its area, a traced vertex made the area a tracer and the partition
    unavailable, so the gradient could not be taken at all -- not wrong, unbuildable.
    """
    facet = rectangle_triangles([0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0])
    surfaces = Surfaces.from_triangles(facet, emission=100.0)
    probe = np.array([[0.02, 0.01, 0.3]])

    def total(lift):
        moved = surfaces.with_geometry(jnp.asarray(facet) + jnp.asarray([0.0, 0.0, 1.0]) * lift)
        return jnp.sum(fluence_rate(moved, probe))

    step = 1e-7
    finite_difference = (float(total(jnp.asarray(step))) - float(total(jnp.asarray(-step)))) / (
        2.0 * step
    )
    assert float(jax.grad(total)(jnp.asarray(0.0))) == pytest.approx(finite_difference, rel=1e-7)


def test_a_point_source_can_be_moved_under_a_gradient():
    """The same for a lamp modelled as a point, where the position is all there is to move."""
    source = point_source([0.0, 0.0, 0.0], power=50.0)
    probe = np.array([[0.0, 0.0, 2.0]])

    def total(lift):
        moved = source.with_geometry(jnp.zeros((1, 3, 3)) + jnp.asarray([0.0, 0.0, 1.0]) * lift)
        return jnp.sum(fluence_rate(moved, probe))

    step = 1e-7
    finite_difference = (float(total(jnp.asarray(step))) - float(total(jnp.asarray(-step)))) / (
        2.0 * step
    )
    assert float(jax.grad(total)(jnp.asarray(0.0))) == pytest.approx(finite_difference, rel=1e-7)


def test_the_gradient_reaches_every_vertex_of_every_facet():
    """A per-vertex jacobian, not one scalar knob -- a shape optimizer moves each one."""
    facet = rectangle_triangles([0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0])
    surfaces = Surfaces.from_triangles(facet, emission=100.0)
    probe = np.array([[0.02, 0.01, 0.3]])
    jacobian = jax.grad(
        lambda vertices: jnp.sum(fluence_rate(surfaces.with_geometry(vertices), probe))
    )(jnp.asarray(facet))
    assert jacobian.shape == facet.shape
    assert bool(jnp.all(jnp.isfinite(jacobian)))
    assert float(jnp.min(jnp.abs(jacobian).sum(axis=(1, 2)))) > 0.0
