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
from aquaflux.radiation.absorption import UniformAbsorption, VoxelAbsorption
from aquaflux.radiation.gather import (
    direct_fluence_rate,
    direct_irradiance,
    summed_fluence_rate,
)
from aquaflux.radiation.profiles import CosinePower, Isotropic, Lambertian
from aquaflux.radiation.subdivide import refine_for_receivers
from aquaflux.radiation.surfaces import Surfaces
from scipy.integrate import quad
from scipy.special import expn

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
    measured = np.asarray(direct_fluence_rate(point_source([0.0, 0.0, 0.0]), points))
    np.testing.assert_allclose(measured, POWER / (4.0 * np.pi * radii**2), rtol=1e-14)


def test_point_sources_add():
    """Nothing in the model couples sources, so the gather must be linear in them."""
    left, right = point_source([-1.0, 0.0, 0.0]), point_source([1.0, 0.0, 0.0])
    both = point_source([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    probe = np.array([[0.3, 0.4, 0.0]])
    assert float(direct_fluence_rate(both, probe)[0]) == pytest.approx(
        float(direct_fluence_rate(left, probe)[0]) + float(direct_fluence_rate(right, probe)[0]),
        rel=1e-14,
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
    assert float(direct_irradiance(source, probe, normal)[0]) == pytest.approx(
        float(direct_fluence_rate(source, probe)[0]) * np.cos(angle), rel=1e-13
    )


def test_a_receiver_facing_away_from_a_point_source_is_not_illuminated():
    source = point_source([0.0, 0.0, 0.0])
    probe = np.array([[1.0, 0.0, 0.0]])
    assert float(direct_irradiance(source, probe, np.array([[1.0, 0.0, 0.0]]))[0]) == 0.0


def test_the_cosine_identity_fails_once_there_is_more_than_one_source():
    """The warning in the module, made a test so nobody re-derives ``E = G cos`` as general.

    Two sources on opposite sides of a receiver: their fluence rates add while their
    irradiances oppose, so the ratio is not a cosine of anything.
    """
    both = point_source([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    probe, normal = np.array([[0.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])
    total_fluence = float(direct_fluence_rate(both, probe)[0])
    total_irradiance = float(direct_irradiance(both, probe, normal)[0])
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
            direct_fluence_rate(
                _line_source(half_length, 1.0, count), np.array([[radius, 0.0, 0.0]])
            )[0]
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
            direct_fluence_rate(
                _line_source(half_length, 1.0, count), np.array([[radius, 0.0, 0.0]])
            )[0]
        )
        return abs(measured - reference) / reference

    assert 2.0 * half_length / 10 > radius, "the coarse fixture no longer under-resolves"
    assert error(10) / error(20) > 10.0


def test_a_long_summed_line_source_approaches_the_infinite_limit():
    """Case 6 in vacuum: ``G -> P' / (4 r)``, the limit the radial and MPSS lamp models differ
    on by a factor of ``pi / 2``."""
    radius = 0.05
    measured = float(
        direct_fluence_rate(_line_source(50.0, 1.0, 20000), np.array([[radius, 0.0, 0.0]]))[0]
    )
    assert measured == pytest.approx(1.0 / (4.0 * radius), rel=2e-3)


def _bickley(order, x):
    """The Bickley-Naylor function ``Ki_n(x) = int_0^(pi/2) exp(-x / cos t) cos^(n-1) t dt``.

    What the exponential integral is to a plane, this is to a line: the attenuation integrated
    over every slant path from an infinite line to a point beside it.
    """
    value, _ = quad(
        lambda t: np.exp(-x / np.cos(t)) * np.cos(t) ** (order - 1),
        0.0,
        np.pi / 2,
        epsabs=0.0,
        epsrel=1e-13,
    )
    return value


@pytest.mark.parametrize("optical_radius", [0.5, 2.0])
def test_an_infinite_line_in_an_absorbing_medium_gives_the_bickley_functions(optical_radius):
    """Case 6 in absorbing water, where the lamp models are usually compared.

    An infinite line of isotropic sources, ``P'`` per length, gives at distance ``r``::

        G = P' / (2 pi r) Ki_1(a r)          E_radial = P' / (2 pi r) Ki_2(a r)

    Both are gated, because each catches what the other cannot. At ``a = 0`` the irradiance is
    ``P' / (2 pi r)``, which is **numerically identical to the radial lamp model** -- so a code
    comparing its irradiance against that model agrees to machine precision while its fluence
    rate is ``pi / 2`` off. And attenuating every slant path by the perpendicular one,
    ``P' / (4 r) exp(-a r)``, overstates the fluence rate by 1.48 at ``a r = 0.5`` and 2.19 at
    ``a r = 2``, which is why one probe sits that deep.

    The line runs to an optical half-length of 35, so what it leaves out is below ``e^-35``,
    with points a fiftieth of the distance apart. Measured, that reproduces both functions to
    better than 1e-11: a midpoint sum of a smooth integrand decaying along an effectively
    infinite line converges far faster than its nominal second order, so the tolerance here is
    set by the quadrature of the reference, not by the sum.
    """
    radius = 0.05
    coefficient = optical_radius / radius
    half_length = 35.0 / coefficient
    count = int(np.ceil(2.0 * half_length / (radius / 50.0)))
    line = _line_source(half_length, 1.0, count)
    probe, facing_the_line = np.array([[radius, 0.0, 0.0]]), np.array([[-1.0, 0.0, 0.0]])
    medium = UniformAbsorption(coefficient)

    fluence = float(direct_fluence_rate(line, probe, absorption=medium)[0])
    irradiance = float(direct_irradiance(line, probe, facing_the_line, absorption=medium)[0])
    scale = 1.0 / (2.0 * np.pi * radius)
    assert fluence == pytest.approx(scale * _bickley(1, optical_radius), rel=1e-9)
    assert irradiance == pytest.approx(scale * _bickley(2, optical_radius), rel=1e-9)


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
        measured = float(direct_fluence_rate(surfaces, np.array([[0.0, 0.0, distance]]))[0])
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
    in_front = float(direct_fluence_rate(surfaces, np.array([[0.0, 0.0, 1.0]]))[0])
    behind = float(direct_fluence_rate(surfaces, np.array([[0.0, 0.0, -1.0]]))[0])
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
    assert float(direct_fluence_rate(surfaces, probe)[0]) == pytest.approx(
        expected_fluence, rel=2e-3
    )
    assert float(direct_irradiance(surfaces, probe, facing)[0]) == pytest.approx(
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
        measured = float(direct_fluence_rate(surfaces, np.array([[0.0, 0.0, height]]))[0])
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
    measured_fluence = float(direct_fluence_rate(surfaces, probe)[0])
    measured_irradiance = float(direct_irradiance(surfaces, probe, facing)[0])
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
    measured = float(direct_fluence_rate(surfaces, np.array([[ratio * radius, 0.0, 0.0]]))[0])
    expected = (4.0 * exitance / np.pi) * np.arcsin(1.0 / ratio)
    assert measured == pytest.approx(expected, rel=2e-3)


def test_a_narrower_beam_puts_more_on_axis_and_less_to_the_side_at_equal_exitance():
    """Otherwise the profile reaches the gather normalized but unused."""
    facet = rectangle_triangles([0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0])
    on_axis, off_axis = np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 1.0]])
    diffuse = Surfaces.from_triangles(facet, emission=1.0, profiles=(Lambertian(),))
    narrow = Surfaces.from_triangles(facet, emission=1.0, profiles=(CosinePower(8.0),))
    assert float(direct_fluence_rate(narrow, on_axis)[0]) > float(
        direct_fluence_rate(diffuse, on_axis)[0]
    )
    assert float(direct_fluence_rate(narrow, off_axis)[0]) < float(
        direct_fluence_rate(diffuse, off_axis)[0]
    )


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
    assert float(direct_fluence_rate(mixed, probe)[0]) == pytest.approx(
        float(direct_fluence_rate(areal, probe)[0]) + POWER / (4.0 * np.pi), rel=1e-13
    )


# ---------------------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("pair_limit", [1, 7, 64, 10_000])
def test_chunking_changes_nothing_about_the_answer(pair_limit):
    """Chunking is a memory strategy; it must not be a numerical one.

    The last chunk is padded rather than shortened so the traced body compiles once, and the
    padding must not leak into the result -- a chunk that does not divide the receiver count is
    the case that catches it, which is why 37 receivers against two facets, and limits giving
    chunks of one, three and thirty-two.
    """
    rng = np.random.default_rng(0)
    probes = rng.uniform(0.5, 2.0, (37, 3))
    surfaces = point_source([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
    reference = np.asarray(direct_fluence_rate(surfaces, probes, pair_limit=1_000_000))
    np.testing.assert_allclose(
        np.asarray(direct_fluence_rate(surfaces, probes, pair_limit=pair_limit)),
        reference,
        rtol=1e-15,
    )


@pytest.mark.parametrize("pair_limit", [0, -1])
def test_a_limit_of_no_pairs_is_refused(pair_limit):
    """Zero divides the receiver count into an infinite number of chunks, so the unguarded
    version raises a ``ZeroDivisionError`` from inside the padding arithmetic — an error that
    names nothing the caller passed."""
    probes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    surfaces = point_source([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="pair_limit must be at least 1"):
        direct_fluence_rate(surfaces, probes, pair_limit=pair_limit)
    with pytest.raises(ValueError, match="pair_limit must be at least 1"):
        direct_irradiance(surfaces, probes, probes, pair_limit=pair_limit)


def _chunk_slices(monkeypatch):
    """Record every chunk the gather cuts: the whole array's shape, the axis cut and the size.

    Watches the slicing rather than replacing it, so the gather still computes its answer.
    """
    from aquaflux.radiation import gather

    seen = []
    real = gather._slice

    def watched(array, axis, start, size):
        seen.append((tuple(array.shape), axis, size))
        return real(array, axis, start, size)

    monkeypatch.setattr(gather, "_slice", watched)
    return seen


@pytest.mark.parametrize("n_facets", [1, 5])
def test_a_chunk_holds_as_many_receivers_as_fit_the_pair_limit(monkeypatch, n_facets):
    """The bound is on PAIRS, so a finer emitter gets fewer receivers per chunk, not a larger one.

    This is the whole of the defect it replaces: a receiver count left each chunk's size to the
    facet count, and against a finely divided lamp the default formed a chunk of gigabytes. Same
    limit, same receivers, two facet counts -- the chunk's receiver count must fall by the ratio.
    """
    seen = _chunk_slices(monkeypatch)
    rng = np.random.default_rng(1)
    surfaces = point_source(rng.uniform(-0.1, 0.1, (n_facets, 3)))
    direct_fluence_rate(surfaces, rng.uniform(1.0, 2.0, (40, 3)), pair_limit=10)
    sizes = [size for _, _, size in seen]
    per_chunk = sizes[0]
    assert per_chunk == 10 // n_facets
    assert all(size <= per_chunk for size in sizes)


def test_a_scene_with_nothing_in_the_way_forms_no_array_the_size_of_the_problem(monkeypatch):
    """With no mask, nothing per receiver-and-facet may be formed before chunking.

    A row of ones per receiver used to stand in for "nothing occludes", formed whole: at a mesh's
    cells against a finely divided lamp that is hundreds of gigabytes, which the chunking that
    follows could not bound. So the only arrays cut into chunks are the points (and normals).
    """
    seen = _chunk_slices(monkeypatch)
    probes = np.random.default_rng(2).uniform(1.0, 2.0, (40, 3))
    direct_fluence_rate(point_source([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]), probes, pair_limit=8)
    direct_irradiance(point_source([[0.0, 0.0, 0.0]]), probes, probes, pair_limit=8)
    assert {shape for shape, _, _ in seen} == {(40, 3)}, seen


def test_a_mask_is_cut_into_chunks_rather_than_turned_into_a_fraction_first(monkeypatch):
    """The surviving fraction is formed per chunk, from the mask's own layers.

    Formed before chunking it would be a floating-point array the size of the whole problem --
    eight bytes a pair on top of a one-byte mask -- and padding the mask to whole chunks would
    copy it again. So what is cut must be the mask itself, along its receiver axis, and nothing
    of the problem's size may be formed from it first.
    """
    from aquaflux.radiation.visibility import Visibility, build_visibility
    from aquaflux.solids import Cylinder

    source = point_source([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    probes = np.random.default_rng(4).uniform([3.0, -1.0, -1.0], [4.0, 1.0, 1.0], (40, 3))
    sleeve = Cylinder(centre=[2.0, 0.0, 0.0], axis=[0, 0, 1], radius=0.5, half_length=3.0)
    mask = build_visibility([sleeve], source, probes)
    whole = []
    monkeypatch.setattr(Visibility, "surviving", lambda self, t: whole.append(t))
    seen = _chunk_slices(monkeypatch)
    field = direct_fluence_rate(source, probes, visibility=mask, transmittance=[0.3], pair_limit=8)
    assert not whole, "the surviving fraction was formed for the whole mask"
    assert {(shape, axis) for shape, axis, _ in seen} == {
        ((40, 3), 0),
        ((1, 40, 2), 1),
        ((40, 2), 0),
    }, seen
    # And the answer is the one the whole-mask fraction gives.
    monkeypatch.undo()
    reference = direct_fluence_rate(
        source, probes, visibility=mask, transmittance=[0.3], pair_limit=10_000
    )
    np.testing.assert_allclose(np.asarray(field), np.asarray(reference), rtol=1e-15)
    assert float(np.asarray(field).min()) < float(np.asarray(field).max()), "nothing is shadowed"


def _mixed_set():
    """Two areal profiles and a point source, so every branch of the gather is exercised."""
    panel = rectangle_triangles([0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0])
    vertices = np.concatenate([panel, np.full((1, 3, 3), 0.05)])
    return Surfaces.from_triangles(
        vertices,
        emission=[3.0, 5.0, 0.0],
        power=[0.0, 0.0, POWER],
        profiles=(CosinePower(4.0), Lambertian(), Isotropic()),
        profile_index=[0, 1, 2],
    )


def test_summing_sets_in_one_pass_is_summing_their_gathers():
    """One geometric pass for several sets must give what gathering each and adding does.

    The fixture's sets differ in profiles as well as values -- the second is re-read as
    Lambertian, as the reflected field is -- so a set's weights cannot be read off another's
    grouping, and there are enough receivers for several chunks and a shorter remainder.
    """
    first = _mixed_set()
    second = first.with_optics(emission=[1.0, 2.0, 0.0], power=0.0, profiles=(Lambertian(),))
    probes = np.random.default_rng(5).uniform([-0.1, -0.1, 0.2], [0.3, 0.3, 0.6], (23, 3))
    medium = UniformAbsorption(3.0)
    apart = direct_fluence_rate(first, probes, absorption=medium) + direct_fluence_rate(
        second, probes, absorption=medium
    )
    together = summed_fluence_rate((first, second), probes, absorption=medium, pair_limit=12)
    np.testing.assert_allclose(np.asarray(together), np.asarray(apart), rtol=1e-14)


def test_sets_that_do_not_share_their_geometry_are_refused():
    first = _mixed_set()
    moved = first.with_geometry(np.asarray(first.vertices) + 0.01)
    with pytest.raises(ValueError, match="must share their geometry"):
        summed_fluence_rate((first, moved), np.array([[0.1, 0.1, 0.5]]))


def test_the_point_sources_alone_are_gathered_without_visiting_a_facet(monkeypatch):
    """What a surface solve needs from outside its transfer: the point sources' arrivals only.

    Weighting the areal facets by zero gave the same number and paid for every one of them -- a
    clipped projected solid angle per pair -- so the areal branch must not run at all.
    """
    from aquaflux.radiation import gather

    surfaces = _mixed_set()
    probes = np.array([[0.1, 0.1, 0.5], [0.3, -0.1, 0.4]])
    normals = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    dark = surfaces.with_optics(emission=0.0)
    expected = direct_irradiance(dark, probes, normals)
    visited = []
    real = gather.projected_solid_angle
    monkeypatch.setattr(gather, "projected_solid_angle", lambda *a: visited.append(1) or real(*a))
    alone = direct_irradiance(surfaces, probes, normals, point_sources_only=True)
    assert not visited, "the areal facets were visited"
    np.testing.assert_allclose(np.asarray(alone), np.asarray(expected), rtol=1e-15)
    assert float(np.asarray(alone).min()) > 0.0


def test_no_receivers_gives_no_answers():
    assert direct_fluence_rate(point_source([0.0, 0.0, 0.0]), np.zeros((0, 3))).shape == (0,)


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
        return jnp.sum(direct_fluence_rate(scaled, probe))

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
    compiled = jax.jit(
        lambda power: direct_fluence_rate(surfaces.with_optics(power=power), probes)
    )(jnp.asarray(POWER))
    np.testing.assert_allclose(
        np.asarray(compiled), np.asarray(direct_fluence_rate(surfaces, probes))
    )


def _narrow_panel():
    panel = rectangle_triangles([0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0])
    return Surfaces.from_triangles(panel, emission=5.0, profiles=(CosinePower(4.0),))


@pytest.mark.parametrize(
    "rebuild",
    [
        pytest.param(
            lambda surfaces, value: surfaces.with_optics(
                emission=value * surfaces.emission, profiles=(Lambertian(),)
            ),
            id="profiles replaced",
        ),
        pytest.param(
            lambda surfaces, value: surfaces.with_geometry(surfaces.vertices + value - 1.0),
            id="geometry moved",
        ),
    ],
)
def test_a_surface_set_rebuilt_inside_the_trace_still_compiles(rebuild):
    """Which profile each facet uses must survive being rebuilt inside a traced function.

    Re-reading a set as Lambertian is what the volume field does for its reflected part, and
    moving a set is how a lamp's position is differentiated. Both rebuild the record inside the
    trace, and a ``jnp`` array made there is staged even from concrete input -- so the grouping
    the gather partitions on used to arrive as a tracer and be refused.
    """
    surfaces = _narrow_panel()
    probes = jnp.asarray([[0.1, 0.1, 0.5], [0.3, -0.1, 0.4]])

    def field(value):
        return direct_fluence_rate(rebuild(surfaces, value), probes)

    value = jnp.asarray(1.0)
    np.testing.assert_allclose(
        np.asarray(jax.jit(field)(value)), np.asarray(field(value)), rtol=1e-13
    )


def test_a_gradient_s_memory_is_bounded_by_the_pair_limit_not_the_receiver_count():
    """The pair limit bounds a reverse pass as well as a forward one.

    A scan's reverse pass keeps every chunk's intermediates unless its body is checkpointed, so
    without the checkpoint a gradient's working memory grows in proportion to the receivers --
    at a mesh's cells against a finely divided lamp, terabytes -- however small the limit.
    Pinned on the compiled program's own working-memory figure, which is exact and immune to
    what else the machine is doing, at sixteen times the receivers and the same limit.
    """
    surfaces = _narrow_panel()
    pair_limit = 16 * surfaces.n_facets
    medium = UniformAbsorption(2.0)

    def working_memory(n_receivers):
        rng = np.random.default_rng(3)
        probes = jnp.asarray(rng.uniform([-0.2, -0.2, 0.3], [0.4, 0.4, 0.9], (n_receivers, 3)))

        def total(emission):
            lit = surfaces.with_optics(emission=emission)
            return jnp.sum(
                direct_fluence_rate(lit, probes, absorption=medium, pair_limit=pair_limit)
            )

        compiled = jax.jit(jax.grad(total)).lower(jnp.asarray(surfaces.emission)).compile()
        return compiled.memory_analysis().temp_size_in_bytes

    few, many = working_memory(64), working_memory(1024)
    assert many < 1.5 * few, (few, many)


def test_passing_the_whole_surface_set_as_a_traced_argument_says_why_it_cannot_work():
    """A bare tracer error from inside the partition would send a reader looking in the wrong
    place; the message names the fix instead."""
    surfaces = point_source([0.0, 0.0, 0.0])
    probes = jnp.asarray([[1.0, 0.0, 0.0]])
    with pytest.raises(TypeError, match="profile index must be concrete"):
        jax.jit(lambda s, p: direct_fluence_rate(s, p))(surfaces, probes)


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
        return jnp.sum(direct_fluence_rate(moved, probe))

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
        return jnp.sum(direct_fluence_rate(moved, probe))

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
        lambda vertices: jnp.sum(direct_fluence_rate(surfaces.with_geometry(vertices), probe))
    )(jnp.asarray(facet))
    assert jacobian.shape == facet.shape
    assert bool(jnp.all(jnp.isfinite(jacobian)))
    assert float(jnp.min(jnp.abs(jacobian).sum(axis=(1, 2)))) > 0.0


# ---------------------------------------------------------------------------------------
# Absorbing media
# ---------------------------------------------------------------------------------------


def test_a_uniform_medium_attenuates_a_point_source_exactly():
    radii = np.array([0.5, 1.0, 3.0])
    points = np.stack([radii, np.zeros_like(radii), np.zeros_like(radii)], axis=1)
    coefficient = 0.7
    measured = np.asarray(
        direct_fluence_rate(
            point_source([0.0, 0.0, 0.0]), points, absorption=UniformAbsorption(coefficient)
        )
    )
    expected = POWER / (4.0 * np.pi * radii**2) * np.exp(-coefficient * radii)
    np.testing.assert_allclose(measured, expected, rtol=1e-14)


def test_a_transparent_medium_is_the_vacuum_field():
    """A zero coefficient must reproduce the vacuum answer exactly, not merely closely."""
    probes = np.array([[1.0, 0.5, 0.25], [2.0, 0.0, 0.0]])
    source = point_source([0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        np.asarray(direct_fluence_rate(source, probes, absorption=UniformAbsorption(0.0))),
        np.asarray(direct_fluence_rate(source, probes)),
        rtol=1e-15,
    )


def test_a_constant_graded_medium_agrees_with_the_closed_form_one_through_the_gather():
    """The two strategies are interchangeable where they describe the same medium, so a
    bookkeeping error in the expensive one shows up against the cheap one."""
    probes = np.array([[1.0, 0.5, 0.25], [2.0, 0.0, 0.0]])
    source = point_source([0.0, 0.0, 0.0])
    coefficient = 0.9
    graded = VoxelAbsorption(
        np.full((5, 5, 5), coefficient), origin=[-3.0, -3.0, -3.0], spacing=[1.5, 1.5, 1.5]
    )
    np.testing.assert_allclose(
        np.asarray(direct_fluence_rate(source, probes, absorption=graded)),
        np.asarray(direct_fluence_rate(source, probes, absorption=UniformAbsorption(coefficient))),
        rtol=1e-12,
    )


@pytest.mark.parametrize("depth", [0.2, 0.5, 2.0])
def test_a_diffuse_wall_in_an_absorbing_medium_gives_the_exponential_integrals(depth):
    """Case 5, and the case whose two answers are most often swapped.

    A Lambertian wall of exitance ``M`` seen through an absorbing medium gives
    ``G = 2 M E_2(kappa x)`` and ``E = 2 M E_3(kappa x)``. ⚠️ **It is not**
    ``exp(-kappa x)``: that is the collimated result, and quoting it for a diffuse wall is the
    standard error, because every ray but the axial one travels a longer slant path.

    One probe is at ``kappa x = 2`` deliberately. At small optical depth
    ``exp(-t) ~ 1 - t`` and the exponential integrals are close to it, so a shallow sweep
    cannot tell the two apart.

    The disc is refined against the probes before the gather rather than built fine: near the
    axis a uniform disc has facets wider than their distance to the receiver, and no amount of
    extra radius fixes that. Refining brings the error from 2% to below 2e-3, which is also a
    demonstration that the criterion does what it claims.
    """
    exitance, coefficient = 3.0, 2.0
    probe = np.array([[0.0, 0.0, depth / coefficient]])
    coarse = Surfaces.from_triangles(disc_triangles(6.0, rings=60, sectors=96), emission=exitance)
    wall, _ = refine_for_receivers(coarse, probe, max_ratio=0.25, max_levels=7)
    medium = UniformAbsorption(coefficient)

    measured_fluence = float(direct_fluence_rate(wall, probe, absorption=medium)[0])
    measured_irradiance = float(
        direct_irradiance(wall, probe, np.array([[0.0, 0.0, -1.0]]), absorption=medium)[0]
    )
    assert measured_fluence == pytest.approx(2.0 * exitance * expn(2, depth), rel=3e-3)
    assert measured_irradiance == pytest.approx(2.0 * exitance * expn(3, depth), rel=3e-3)


@pytest.mark.parametrize(
    ("depth", "fluence_ratio", "irradiance_ratio"), [(0.1, 0.80, 0.920), (2.0, 0.28, 0.445)]
)
def test_the_diffuse_and_collimated_answers_differ_by_a_published_ratio(
    depth, fluence_ratio, irradiance_ratio
):
    """Pins which quantity each published ratio belongs to.

    Against the collimated ``exp(-kappa x)``, the diffuse wall's fluence rate is 0.80 of it at
    an optical depth of 0.1 and 0.28 at 2 — while the *irradiance* ratios at the same depths
    are 0.920 and 0.445. Quoting one set for the other is a factor of one and a half at depth,
    and both numbers look plausible.
    """
    assert expn(2, depth) / np.exp(-depth) == pytest.approx(fluence_ratio, abs=5e-3)
    assert 2.0 * expn(3, depth) / np.exp(-depth) == pytest.approx(irradiance_ratio, abs=5e-3)


def test_the_gather_is_differentiable_in_a_uniform_absorption_coefficient():
    """The sensitivity of a dose to water quality, which is a question a plant operator asks."""
    source = point_source([0.0, 0.0, 0.0])
    probes = np.array([[1.0, 0.0, 0.0], [2.0, 0.5, 0.0]])

    def total(coefficient):
        return jnp.sum(
            direct_fluence_rate(source, probes, absorption=UniformAbsorption(coefficient))
        )

    step = 1e-6
    finite_difference = (float(total(0.5 + step)) - float(total(0.5 - step))) / (2.0 * step)
    assert float(jax.grad(total)(jnp.asarray(0.5))) == pytest.approx(finite_difference, rel=1e-7)


def test_the_gather_is_differentiable_in_a_graded_absorbance_field():
    """Cell by cell, which is what an inverse problem for water quality needs."""
    source = point_source([0.0, 0.0, 0.0])
    probes = np.array([[1.0, 0.0, 0.0]])
    origin, spacing = [-3.0, -3.0, -3.0], [1.5, 1.5, 1.5]
    base = np.full((5, 5, 5), 0.6)

    def total(coefficient):
        medium = VoxelAbsorption(coefficient, origin, spacing)
        return jnp.sum(direct_fluence_rate(source, probes, absorption=medium))

    jacobian = jax.grad(total)(jnp.asarray(base))
    assert jacobian.shape == (5, 5, 5)
    assert bool(jnp.all(jnp.isfinite(jacobian)))
    # Every cell the ray crosses must show a negative sensitivity: more absorbance, less light.
    assert float(jnp.min(jacobian)) < 0.0
    assert float(jnp.max(jacobian)) <= 0.0
