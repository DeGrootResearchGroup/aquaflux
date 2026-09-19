"""The assembled model: the interreflection solve, the volume field, and what stays live."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from aquaflux.radiation.absorption import UniformAbsorption
from aquaflux.radiation.gather import direct_fluence_rate
from aquaflux.radiation.model import (
    RadiationSettings,
    build_radiation_model,
    fluence_rate,
    radiosity,
    surface_irradiance,
)
from aquaflux.radiation.occluders import Cylinder
from aquaflux.radiation.profiles import CosinePower, Isotropic, Lambertian
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.solve import relative_residual_gmres

from tests.unit.radiation_references import box, inward_box, rectangle_triangles


def surface_model(surfaces, *, occluders=(), **settings):
    """A model with no receivers, for the tests that are about the surface system alone.

    ``self_occlusion`` defaults to ``False`` here: these fixtures are closed boxes whose facets
    are all mutually visible, and tracing rays between every pair of them to rediscover that
    costs the whole suite for nothing. The tests that are *about* occlusion pass it back in.
    """
    settings.setdefault("self_occlusion", False)
    return build_radiation_model(
        np.zeros((0, 3)), surfaces, occluders=occluders, settings=RadiationSettings(**settings)
    )


# ---------------------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("reflectance", [0.0, 0.5, 0.9])
@pytest.mark.parametrize("divisions", [1, 2, 4])
def test_a_uniform_closed_box_reaches_its_closed_form(reflectance, divisions):
    """The case that catches a truncated bounce count. ``B = M / (1 - rho)``.

    At a reflectance of 0.9 a single bounce gives 1.9 against the true 10, so nothing that stops
    early can pass. The answer here is exact to machine precision at every refinement, which it
    is only because the row sums are.
    """
    exitance = 3.0
    surfaces = box(divisions, emission=exitance, reflectance=reflectance)
    outgoing, _ = radiosity(surface_model(surfaces), surfaces)
    np.testing.assert_allclose(np.asarray(outgoing), exitance / (1.0 - reflectance), rtol=1e-12)


def test_a_closed_box_reaches_its_closed_form_AT_THE_DEFAULT_SETTINGS():
    """Every other test in this file switches self-occlusion off, and that hid a bug.

    The transfer build's receivers are the facet centroids themselves, so each self-occlusion
    ray ends on the facet it is aimed at — which counted as a hit. The default therefore
    reported every mutually visible pair as blocked, and a closed box came back with ``B = M``:
    at a reflectance of 0.9 that is ten times too dark, and it looks like a field rather than an
    error. No test could see it, because every fixture passed ``self_occlusion=False``, and the
    self-occlusion tests all use receivers out in the volume where the ray ends on nothing.

    ⚠️ **``row_sum_error`` is blind to this by construction** and cannot be made to catch it:
    the mask is applied live in ``TransferMatrix.assemble``, not baked into ``geometric``, so
    the gate reads 1e-15 while the matrix it is reporting on is being zeroed downstream.

    So this test is deliberately the one that takes :class:`RadiationSettings` as it comes.
    """
    exitance, reflectance = 3.0, 0.9
    surfaces = box(2, emission=exitance, reflectance=reflectance)
    model = build_radiation_model(np.zeros((0, 3)), surfaces)
    assert model.settings.self_occlusion is None, "the point of this test is the default"
    outgoing, _ = radiosity(model, surfaces)
    np.testing.assert_allclose(np.asarray(outgoing), exitance / (1.0 - reflectance), rtol=1e-12)


def test_the_radiosity_of_a_convex_enclosure_ignores_the_shadow_mask():
    """The other side of the same fact, one layer up: a box shadows nothing, so switching the
    mask on must move the solved radiosity by exactly zero rather than merely by a little."""
    surfaces = box(2, emission=3.0, reflectance=0.9)
    shadowed, _ = radiosity(surface_model(surfaces, self_occlusion=True), surfaces)
    clear, _ = radiosity(surface_model(surfaces), surfaces)
    np.testing.assert_array_equal(np.asarray(shadowed), np.asarray(clear))


def test_the_irradiance_matches_what_the_radiosity_implies():
    """``B = M + rho H`` must hold facet by facet, or the two are computing different systems."""
    exitance, reflectance = 3.0, 0.7
    surfaces = box(2, emission=exitance, reflectance=reflectance)
    model = surface_model(surfaces)
    outgoing, _ = radiosity(model, surfaces)
    landing, _ = surface_irradiance(model, surfaces)
    np.testing.assert_allclose(
        np.asarray(outgoing), exitance + reflectance * np.asarray(landing), rtol=1e-12
    )


def test_the_bounce_count_is_not_a_parameter():
    """The inverse *is* the infinite bounce sum, so the solve must match a long Neumann series
    and must not match a short one."""
    reflectance = 0.8
    surfaces = box(2, emission=1.0, reflectance=reflectance)
    model = surface_model(surfaces)
    exact, _ = radiosity(model, surfaces)

    matrix = np.asarray(model.transfer.geometric) * reflectance
    emission = np.ones(surfaces.n_facets)

    def bounces(count):
        total, term = emission.copy(), emission.copy()
        for _ in range(count):
            term = matrix @ term
            total = total + term
        return total

    assert np.max(np.abs(bounces(200) - np.asarray(exact))) < 1e-10
    assert np.max(np.abs(bounces(1) - np.asarray(exact))) > 1.0


def test_an_external_irradiance_is_reflected_like_any_other_arrival():
    surfaces = box(1, emission=0.0, reflectance=0.5)
    model = surface_model(surfaces)
    lit = np.full(surfaces.n_facets, 2.0)
    outgoing, _ = radiosity(model, surfaces, external_irradiance=lit)
    # Each facet re-emits half of what arrives, and what it re-emits comes back round the box.
    assert float(jnp.min(outgoing)) > 0.5 * 2.0
    dark, _ = radiosity(model, surfaces)
    np.testing.assert_allclose(np.asarray(dark), 0.0, atol=1e-14)


def test_the_irradiance_is_not_a_number_on_a_point_source():
    """A point source has no surface for light to land on; zero there would read as shadow."""
    vertices = np.concatenate([inward_box(1), np.zeros((1, 3, 3))])
    surfaces = Surfaces.from_triangles(
        vertices,
        emission=[1.0] * 12 + [0.0],
        profiles=(Lambertian(), Isotropic()),
        profile_index=[0] * 12 + [1],
    )
    landing, _ = surface_irradiance(surface_model(surfaces), surfaces)
    assert bool(jnp.isnan(landing[-1]))
    assert bool(jnp.all(jnp.isfinite(landing[:-1])))


def test_a_body_in_the_way_removes_the_transfer_across_it():
    """Occlusion multiplies the same matrix, so a blocked pair simply stops exchanging."""
    facing = np.concatenate(
        [
            rectangle_triangles([0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            rectangle_triangles([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ]
    )
    surfaces = Surfaces.from_triangles(facing, emission=1.0, reflectance=0.5)
    clear = surface_model(surfaces)
    blocked = surface_model(
        surfaces,
        occluders=[Cylinder(centre=[0, 0, 0], axis=[1, 0, 0], radius=0.4, half_length=4.0)],
    )
    assert float(jnp.sum(clear.transfer.geometric)) > 0.0
    assert float(jnp.sum(blocked.transfer.visibility.blocked)) > 0.0
    outgoing, _ = radiosity(blocked, surfaces, transmittance=[0.0])
    np.testing.assert_allclose(np.asarray(outgoing), 1.0, rtol=1e-14)


# ---------------------------------------------------------------------------------------
# Non-Lambertian sources
# ---------------------------------------------------------------------------------------


def test_a_lambertian_source_transfers_its_emission_exactly_as_it_transfers_a_reflection():
    """The reduction that pins the profile constants: for a Lambertian emitter the emitted and
    reflected model matrices are the same matrix, so the whole default path collapses to one."""
    surfaces = box(2, emission=1.0, reflectance=0.6)
    model = surface_model(surfaces)
    reflected, emitted = model.transfer.assemble(surfaces)
    assert reflected is emitted


def test_a_narrow_source_sends_its_emission_somewhere_different():
    """Otherwise the profile reaches the surface system normalized and unused.

    A cosine-power source concentrates its emission along its own normal, so the facet it faces
    across the box receives more than it would from a Lambertian source of the same exitance,
    and the facets off to the side receive less.
    """
    exitance = 1.0
    diffuse = box(2, emission=exitance, reflectance=0.0, profiles=(Lambertian(),))
    narrow = box(2, emission=exitance, reflectance=0.0, profiles=(CosinePower(8.0),))
    model = surface_model(diffuse)

    facing, _ = surface_irradiance(model, diffuse)
    beamed, _ = surface_irradiance(model, narrow)
    assert not np.allclose(np.asarray(facing), np.asarray(beamed))


@pytest.mark.parametrize("exponent", [1.0, 2.0, 8.0])
def test_only_a_lambertian_source_balances_its_energy_exactly(exponent):
    """A limitation with a number on it, rather than an assumption left implicit.

    The source's own distribution is still evaluated at **one** direction — from the source's
    centroid to the receiver's — because it has to stay outside the frozen build to keep its
    parameters differentiable. Quadrature over the receiver therefore does not fix this one, and
    measurably does not: at the six-point default the balance is 1.086 / 0.978 / 0.982 / 0.987 at
    12, 48, 192 and 432 facets for an exponent of 8, against 1.145 / 0.970 / 0.970 / 0.977 at one
    point, and the two agree to three figures from three points upward. For a Lambertian source
    the distribution cancels against the projected solid angle and the balance is exact at every
    refinement and every rule. What does shrink this error is refining the mesh, which samples
    more directions.
    """
    surfaces = box(1, emission=1.0, reflectance=0.0, profiles=(CosinePower(exponent),))
    fine = box(4, emission=1.0, reflectance=0.0, profiles=(CosinePower(exponent),))

    def balance(facets):
        landing, _ = surface_irradiance(surface_model(facets), facets)
        area = np.asarray(facets.area)
        return float(np.sum(area * np.asarray(landing)) / np.sum(area))

    coarse_error = abs(balance(surfaces) - 1.0)
    fine_error = abs(balance(fine) - 1.0)
    if exponent == 1.0:
        assert coarse_error < 1e-12 and fine_error < 1e-12, "Lambertian must be exact"
    else:
        assert coarse_error > 5e-2, "the fixture no longer shows the coarse error"
        assert fine_error < coarse_error / 4.0
        assert fine_error < 5e-2


# ---------------------------------------------------------------------------------------
# What stays differentiable, and the adjoint
# ---------------------------------------------------------------------------------------


def _with_profile(surfaces, profile):
    """A copy of the surface set emitting with a different angular distribution."""
    return eqx.tree_at(lambda s: s.profiles, surfaces, (profile,), is_leaf=lambda x: x is None)


def _central_difference(function, value, step=1e-6):
    return (float(function(value + step)) - float(function(value - step))) / (2.0 * step)


def _lit_box(**optics):
    """A box whose facets differ, so a gradient has somewhere to point."""
    rng = np.random.default_rng(0)
    vertices = inward_box(2)
    return Surfaces.from_triangles(
        vertices, emission=rng.uniform(0.0, 2.0, len(vertices)), **optics
    )


def test_the_gradient_in_reflectance_is_exact():
    surfaces = _lit_box(reflectance=0.6)
    model = surface_model(surfaces)

    def total(reflectance):
        outgoing, _ = radiosity(model, surfaces.with_optics(reflectance=reflectance))
        return jnp.sum(outgoing)

    gradient = float(jax.grad(total)(jnp.asarray(0.6)))
    assert gradient == pytest.approx(_central_difference(total, 0.6), rel=1e-6)
    assert gradient > 0.0


def test_the_gradient_in_emission_is_exact():
    surfaces = _lit_box(reflectance=0.6)
    model = surface_model(surfaces)
    base = jnp.asarray(surfaces.emission)

    def total(scale):
        outgoing, _ = radiosity(model, surfaces.with_optics(emission=base * scale))
        return jnp.sum(outgoing)

    assert float(jax.grad(total)(jnp.asarray(1.0))) == pytest.approx(
        _central_difference(total, 1.0), rel=1e-6
    )


def test_the_gradient_reaches_an_occluder_s_transmittance_through_the_solve():
    """One of the two ways this split has been got wrong. Freezing the visibility inside the
    geometry term leaves a finite, plausible number here that is short by about two thirds."""
    surfaces = _lit_box(reflectance=0.6)
    model = surface_model(
        surfaces,
        occluders=[Cylinder(centre=[0.5, 0.5, 0.5], axis=[0, 0, 1], radius=0.2, half_length=0.3)],
    )
    assert int(jnp.sum(model.transfer.visibility.blocked)) > 0, "the body blocks nothing"

    def total(value):
        outgoing, _ = radiosity(model, surfaces, transmittance=jnp.asarray([value]))
        return jnp.sum(outgoing)

    gradient = float(jax.grad(total)(jnp.asarray(0.4)))
    assert gradient == pytest.approx(_central_difference(total, 0.4), rel=1e-6)
    assert abs(gradient) > 0.0


def test_the_gradient_reaches_the_absorption_coefficient_through_the_solve():
    """The other one. Freezing the whole model matrix costs a few percent of this and leaves
    the rest looking healthy."""
    surfaces = _lit_box(reflectance=0.6)
    model = surface_model(surfaces)

    def total(coefficient):
        outgoing, _ = radiosity(model, surfaces, absorption=UniformAbsorption(coefficient))
        return jnp.sum(outgoing)

    gradient = float(jax.grad(total)(jnp.asarray(0.5)))
    assert gradient == pytest.approx(_central_difference(total, 0.5), rel=1e-6)
    assert gradient < 0.0, "more absorbance, less light"


def test_the_gradient_reaches_a_source_s_profile_parameter():
    surfaces = _lit_box(reflectance=0.6, profiles=(CosinePower(3.0),))
    model = surface_model(surfaces)

    def total(exponent):
        narrowed = _with_profile(surfaces, CosinePower(exponent))
        outgoing, _ = radiosity(model, narrowed)
        return jnp.sum(outgoing)

    gradient = float(jax.grad(total)(jnp.asarray(3.0)))
    assert gradient == pytest.approx(_central_difference(total, 3.0, step=1e-5), rel=1e-4)
    assert abs(gradient) > 0.0


def test_the_gradient_does_not_depend_on_how_hard_the_solve_worked():
    """The adjoint must be a solve on the converged system, not the iteration replayed backwards.

    Restarting the solver more often changes the forward path completely — 166 cycles against 3
    for the same answer — so a gradient taped through the iteration would move with it. The step
    counts are asserted to differ, because a test whose two arms take the same path compares a
    configuration against itself.
    """
    surfaces = _lit_box(reflectance=0.9)
    model = surface_model(surfaces)

    def make(restart):
        def total(reflectance):
            outgoing, _ = radiosity(
                model,
                surfaces.with_optics(reflectance=reflectance),
                solver=relative_residual_gmres(1e-12, restart=restart),
            )
            return jnp.sum(outgoing)

        return total

    counts = []
    for restart in (2, 120):
        _, steps = radiosity(
            model, surfaces, solver=relative_residual_gmres(1e-12, restart=restart)
        )
        counts.append(int(steps))
    assert counts[0] != counts[1], f"both arms took the same path: {counts}"

    gradients = [float(jax.grad(make(restart))(jnp.asarray(0.9))) for restart in (2, 120)]
    assert gradients[0] == pytest.approx(gradients[1], rel=1e-8)


def test_the_expensive_geometry_is_frozen():
    """The solid angles are the ``n^2`` build, and nothing that varies per evaluation may force
    it again — so they carry no derivative with respect to where the facets are."""
    surfaces = _lit_box(reflectance=0.6)
    model = surface_model(surfaces)
    gradient = jax.grad(lambda matrix: jnp.sum(matrix))(model.transfer.geometric)
    assert gradient.shape == model.transfer.geometric.shape
    # stop_gradient is applied at the build, so differentiating the build gives nothing back.
    moved = jax.grad(
        lambda scale: jnp.sum(
            surface_model(
                surfaces.with_geometry(jnp.asarray(inward_box(2)) * scale)
            ).transfer.geometric
        )
    )(jnp.asarray(1.0))
    assert float(moved) == 0.0


# ---------------------------------------------------------------------------------------
# The structure a real scene actually has
# ---------------------------------------------------------------------------------------


def _lamp_in_a_dark_box():
    """A few emitting facets and a great many dark ones — what a reactor looks like.

    Every fixture above emits everywhere, which is convenient and is *not* the shape of the
    problem: a lamp is a handful of facets among walls that only reflect. That makes most rows
    of the right-hand side exactly zero, which is the structure the stopping rule below turns on.
    """
    vertices = inward_box(2)
    emission = np.zeros(len(vertices))
    emission[:4] = 100.0
    return Surfaces.from_triangles(vertices, emission=emission, reflectance=0.9)


def test_a_lamp_among_dark_walls_lights_all_of_them():
    surfaces = _lamp_in_a_dark_box()
    outgoing, _ = radiosity(surface_model(surfaces), surfaces)
    assert float(jnp.min(outgoing)) > 0.0, "reflection reaches every facet"
    assert float(jnp.max(outgoing)) > float(jnp.min(outgoing))
    # What the lamp emits is what the walls swallow. Global conservation needs reciprocity, so
    # it is not exact, but at the default six-point rule it is close: absorbed over emitted
    # measures 1.000000 / 1.000212 / 1.000171 / 1.000120 / 1.000102 on boxes of 12, 48, 192, 432
    # and 768 facets, with this scene's four emitting facets and reflectance 0.9 throughout.
    area = np.asarray(surfaces.area)
    landing, _ = surface_irradiance(surface_model(surfaces), surfaces)
    emitted = float(np.sum(area * np.asarray(surfaces.emission)))
    absorbed = float(np.sum(area * (1.0 - 0.9) * np.asarray(landing)))
    assert absorbed == pytest.approx(emitted, rel=1e-3)


def _absorbed_over_emitted(surfaces, n_points):
    area = np.asarray(surfaces.area)
    model = surface_model(surfaces, receiver_quadrature=n_points)
    landing, _ = surface_irradiance(model, surfaces)
    absorbed = float(np.sum(area * (1.0 - 0.9) * np.asarray(landing)))
    return absorbed / float(np.sum(area * np.asarray(surfaces.emission)))


def test_integrating_the_receiver_is_what_keeps_a_fine_mesh_conservative():
    """The defect the quadrature was added for, in the form a user would meet it.

    Sampling the receiver at its centroid does not merely cost a fixed amount of conservation —
    on a small lamp in a large box it gets **worse** as the mesh is refined, because the facets
    nearest the lamp close in on it while staying the same size relative to their separation.
    Measured on this scene at one point per receiver: 1.000000, 0.999868, 0.982769, 0.975625 and
    0.973077 at 12, 48, 192, 432 and 768 facets — 2.7% of the lamp's output unaccounted for, and
    still growing. At the six-point default the same series is 1.000000, 1.000212, 1.000171,
    1.000120 and 1.000102, improving rather than degrading.

    Note which fixture cannot see any of this: a box where *every* facet emits balances to
    1.000000 at every mesh and every rule, because then each facet's error is its neighbour's and
    they cancel identically. That is the same blindness the lamp-in-a-dark-box fixture exists for.
    """
    vertices = inward_box(4)
    emission = np.zeros(len(vertices))
    emission[:4] = 100.0
    surfaces = Surfaces.from_triangles(vertices, emission=emission, reflectance=0.9)

    assert _absorbed_over_emitted(surfaces, 6) == pytest.approx(1.0, rel=1e-3)
    assert abs(_absorbed_over_emitted(surfaces, 1) - 1.0) > 1e-2, "the one-point rule was fine?"

    everywhere = Surfaces.from_triangles(vertices, emission=np.ones(len(vertices)), reflectance=0.9)
    assert _absorbed_over_emitted(everywhere, 1) == pytest.approx(1.0, rel=1e-9)


def test_a_componentwise_stopping_rule_cannot_solve_this_and_a_global_one_can():
    """Why the solver is chosen rather than left to a default.

    A componentwise relative test asks every entry of the residual to fall below ``rtol`` times
    *its own* right-hand side. Where that side is exactly zero — every facet that does not emit,
    which is most of them — the demand becomes absolute and unsatisfiable, and the solve fails
    outright rather than merely running long. The global relative test used here asks the same
    of the residual as a whole and converges in three restart cycles.
    """
    surfaces = _lamp_in_a_dark_box()
    model = surface_model(surfaces)

    reference, cycles = radiosity(model, surfaces)
    assert int(cycles) <= 5
    assert bool(jnp.all(jnp.isfinite(reference)))

    with pytest.raises(Exception, match=r"(?i)stagnation|diverge|singular|not converge"):
        radiosity(model, surfaces, solver=lx.GMRES(rtol=1e-10, atol=0.0))


# ---------------------------------------------------------------------------------------
# Assembling the model
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(3,), (4, 2), (2, 3, 1)])
def test_the_receivers_must_be_a_list_of_points(shape):
    """A single point passed as ``[x, y, z]`` is the easy mistake, and it broadcasts silently
    against a surface set of three facets rather than raising anywhere downstream."""
    with pytest.raises(ValueError, match=r"receivers must be \(n_receivers, 3\)"):
        build_radiation_model(np.zeros(shape), box(1))


def test_an_unset_setting_is_not_passed_on_at_all():
    """Each default is written down once, beside its own reasoning.

    An unset field has to reach the build as an *absent* keyword rather than as a copy of the
    value that function would have chosen: a second copy drifts the day the first one moves, and
    nothing fails when it does — the build simply keeps using the stale number.
    """
    for options in ("transfer_options", "gather_options", "visibility_options"):
        assert getattr(RadiationSettings(), options)() == {}, options
    assert RadiationSettings(receiver_quadrature=3).transfer_options() == {"receiver_quadrature": 3}
    assert RadiationSettings(gather_chunk_size=8).transfer_options() == {}
    assert RadiationSettings(gather_chunk_size=8).gather_options() == {"chunk_size": 8}
    # The two shadow masks are built from one mapping, so self-occlusion reaches both of them.
    shadowed = RadiationSettings(self_occlusion=True)
    assert shadowed.visibility_options() == {"self_occlusion": True}
    assert shadowed.transfer_options() == {"self_occlusion": True}


def test_a_setting_reaches_the_transfer_build():
    """Otherwise the object is accepted, ignored, and the study it configures measures nothing."""
    surfaces = box(2)
    coarse = surface_model(surfaces, receiver_quadrature=1)
    default = surface_model(surfaces)
    assert not np.allclose(
        np.asarray(coarse.transfer.geometric), np.asarray(default.transfer.geometric)
    )


def test_the_gather_chunk_size_reaches_the_gather():
    """A setting that is accepted and then dropped is worse than one that is not offered.

    Bounding peak memory has no effect on the answer by design, so invariance alone cannot tell
    a delivered setting from a discarded one — every value passes. The second half asks the
    gather for a chunk it refuses, which only raises if the number got there.
    """
    surfaces = box(2, emission=3.0, reflectance=0.5)
    points = np.array([[0.5, 0.5, 0.5], [0.2, 0.3, 0.7], [0.9, 0.1, 0.5], [0.4, 0.8, 0.2]])
    whole, _ = fluence_rate(_volume_model(surfaces, points), surfaces)
    in_threes, _ = fluence_rate(_volume_model(surfaces, points, gather_chunk_size=3), surfaces)
    np.testing.assert_allclose(np.asarray(whole), np.asarray(in_threes), rtol=1e-14)

    with pytest.raises(ValueError, match="chunk_size must be at least 1 receiver"):
        fluence_rate(_volume_model(surfaces, points, gather_chunk_size=0), surfaces)


def test_the_two_masks_are_built_against_the_same_bodies():
    """The trap the single build exists to close.

    A body that shadows a facet has to shadow the cells behind it as well. Built separately —
    the surface solve from one list of occluders and the volume gather from another, or from
    none — the field comes out lit through a lamp sleeve the surface solve correctly treated as
    opaque, and nothing in either half looks wrong on its own.
    """
    surfaces = box(2, emission=1.0, reflectance=0.5)
    sleeve = Cylinder(centre=[0.5, 0.5, 0.5], axis=[0, 0, 1], radius=0.2, half_length=0.4)
    behind = np.array([[0.9, 0.5, 0.5]])
    model = _volume_model(surfaces, behind, occluders=[sleeve])
    assert int(jnp.sum(model.transfer.visibility.blocked)) > 0, "the body shadows no facet"
    assert int(jnp.sum(model.receiver_visibility.blocked)) > 0, "the body shadows no receiver"


# ---------------------------------------------------------------------------------------
# The volume field
# ---------------------------------------------------------------------------------------


def _volume_model(surfaces, points, *, occluders=(), **settings):
    settings.setdefault("self_occlusion", False)
    return build_radiation_model(
        points, surfaces, occluders=occluders, settings=RadiationSettings(**settings)
    )


#: Three interior points of the unit box, none of them on a symmetry plane of the facet mesh.
INSIDE_THE_BOX = np.array([[0.5, 0.5, 0.5], [0.2, 0.3, 0.7], [0.9, 0.1, 0.5]])

#: The same, minus the centre, for the fixtures that put a point source there — a receiver
#: coincident with a lamp is at zero range from it and reads as infinite, correctly.
AROUND_THE_LAMP = np.array([[0.2, 0.3, 0.7], [0.9, 0.1, 0.5], [0.5, 0.5, 0.8]])


@pytest.mark.parametrize("reflectance", [0.0, 0.5, 0.9])
def test_the_field_inside_a_uniform_closed_box_is_four_times_its_radiosity(reflectance):
    """A closed form that pins the whole assembly at once, and is exact.

    A closed Lambertian enclosure at uniform radiosity ``B`` has radiance ``B / pi`` in every
    direction, so the fluence rate at *any* interior point is ``4 pi`` times that: ``G = 4 B``,
    with no dependence on where the point is. With uniform emission and reflectance the solve
    gives ``B = M / (1 - rho)``, so ``G = 4 M / (1 - rho)``.

    It is the reflected half that makes this a test rather than a formality: at a reflectance of
    0.9 a gather that drops it returns ``4 M``, a tenth of the answer, and a plausible-looking
    field rather than an error. The direct gather is evaluated here to show exactly that.
    """
    exitance = 3.0
    surfaces = box(2, emission=exitance, reflectance=reflectance)
    model = _volume_model(surfaces, INSIDE_THE_BOX)
    field, _ = fluence_rate(model, surfaces)
    np.testing.assert_allclose(np.asarray(field), 4.0 * exitance / (1.0 - reflectance), rtol=1e-12)

    emitted_only = direct_fluence_rate(
        surfaces, INSIDE_THE_BOX, visibility=model.receiver_visibility
    )
    np.testing.assert_allclose(np.asarray(emitted_only), 4.0 * exitance, rtol=1e-12)


def test_what_a_facet_reflects_leaves_lambertian_whatever_it_emitted_like():
    """The assembly step that is easiest to get wrong by leaving it out.

    Light a box of **narrow-beam** facets purely from outside, with no emission of their own, so
    that every watt in the enclosure has been reflected once. Diffuse reflection is Lambertian,
    so the box is again a uniform-radiosity enclosure and ``G = 4 B`` holds exactly, with
    ``B = rho E / (1 - rho)``.

    Re-gathering the reflected part with the *source's* distribution instead returns 92.7 and
    131.0 at the first two points here against the correct 72.0 — wrong by a quarter to nearly a
    factor of two, and no longer even uniform across the enclosure.
    """
    reflectance, arriving = 0.9, 2.0
    surfaces = box(2, emission=0.0, reflectance=reflectance, profiles=(CosinePower(8.0),))
    model = _volume_model(surfaces, INSIDE_THE_BOX)
    lit = np.full(surfaces.n_facets, arriving)

    outgoing, _ = radiosity(model, surfaces, external_irradiance=lit)
    uniform = reflectance * arriving / (1.0 - reflectance)
    np.testing.assert_allclose(np.asarray(outgoing), uniform, rtol=1e-12)

    field, _ = fluence_rate(model, surfaces, external_irradiance=lit)
    np.testing.assert_allclose(np.asarray(field), 4.0 * uniform, rtol=1e-12)


def _lamp_in_a_box(reflectance, divisions=2):
    """A box of reflecting walls with an isotropic point source at its centre.

    The shape of a real reactor rather than of a convenient fixture: the light all comes from
    something with no area, which is exactly the part the facet-to-facet transfer cannot carry.
    """
    walls = inward_box(divisions)
    vertices = np.concatenate([walls, np.full((1, 3, 3), 0.5)])
    n_facets = len(vertices)
    return Surfaces.from_triangles(
        vertices,
        power=[0.0] * (n_facets - 1) + [10.0],
        reflectance=[reflectance] * (n_facets - 1) + [0.0],
        profiles=(Lambertian(), Isotropic()),
        profile_index=[0] * (n_facets - 1) + [1],
    )


def test_a_lone_point_source_gives_the_inverse_square_law_exactly():
    """With nothing to reflect off, the whole field is one lamp: ``G = P / (4 pi r^2)``."""
    surfaces = _lamp_in_a_box(reflectance=0.0)
    field, _ = fluence_rate(_volume_model(surfaces, AROUND_THE_LAMP), surfaces)
    radius = np.linalg.norm(AROUND_THE_LAMP - 0.5, axis=1)
    np.testing.assert_allclose(np.asarray(field), 10.0 / (4.0 * np.pi * radius**2), rtol=1e-12)


def test_a_point_source_lights_the_walls_it_is_not_in_the_transfer_matrix_with():
    """A point source has no area, so it is absent from the facet-to-facet transfer entirely and
    has to be fed in as an arrival. Leaving that out gives a dark enclosure rather than an error.

    Checked by conservation: every watt the lamp radiates is absorbed by a wall, so
    ``sum(A (1 - rho) H)`` must come back to the lamp's power. It is not exact, because the
    lamp's direction to each wall facet is still evaluated at that facet's centroid — measured
    here at reflectance 0.9 with the lamp at the centre, absorbed over emitted is 1.413436,
    1.030131, 1.010384 and 1.004639 at 12, 48, 192 and 432 wall facets. What the assertion below
    catches is the wiring, which is off by the whole quantity rather than by 3%.
    """
    reflectance = 0.9
    surfaces = _lamp_in_a_box(reflectance)
    landing, _ = surface_irradiance(surface_model(surfaces), surfaces)
    area = np.asarray(surfaces.area)[:-1]
    absorbed = float(np.sum(area * (1.0 - reflectance) * np.asarray(landing)[:-1]))
    assert absorbed == pytest.approx(10.0, rel=5e-2)

    finer = _lamp_in_a_box(reflectance, divisions=4)
    landing, _ = surface_irradiance(surface_model(finer), finer)
    area = np.asarray(finer.area)[:-1]
    refined = float(np.sum(area * (1.0 - reflectance) * np.asarray(landing)[:-1]))
    assert abs(refined - 10.0) < abs(absorbed - 10.0) / 2.0


def test_the_reflected_walls_add_to_the_lamp_rather_than_replacing_it():
    """Both halves of the field are present: the lamp's own light and what the walls send back.

    Every receiver must read strictly brighter than the bare inverse-square law, and the excess
    must grow with the reflectance — a field that merely scales would pass neither.
    """
    fields = []
    for reflectance in (0.0, 0.5, 0.9):
        surfaces = _lamp_in_a_box(reflectance)
        field, _ = fluence_rate(_volume_model(surfaces, AROUND_THE_LAMP), surfaces)
        fields.append(np.asarray(field))
    assert np.all(fields[1] > fields[0]) and np.all(fields[2] > fields[1])
    # The walls contribute far more than the lamp once they reflect well.
    assert np.all(fields[2] - fields[0] > 2.0 * fields[0])


def test_the_field_comes_back_in_the_receivers_own_order():
    """``G`` is a bare array, so its order is the only thing that says which cell each entry is.

    The gather runs in chunks and pads the last one rather than shortening it, so the case that
    can go wrong is a chunk size that does not divide the receiver count — five receivers in
    chunks of two here. A field that is right everywhere and attached to the wrong cells reads
    as a plausible result.
    """
    surfaces = _lamp_in_a_box(reflectance=0.5)
    points = np.concatenate([AROUND_THE_LAMP, [[0.1, 0.9, 0.3], [0.7, 0.2, 0.8]]])
    shuffle = np.array([3, 0, 4, 2, 1])
    in_twos = {"gather_chunk_size": 2}
    straight, _ = fluence_rate(_volume_model(surfaces, points, **in_twos), surfaces)
    shuffled, _ = fluence_rate(_volume_model(surfaces, points[shuffle], **in_twos), surfaces)
    np.testing.assert_allclose(np.asarray(straight)[shuffle], np.asarray(shuffled), rtol=1e-14)
    assert len(np.unique(np.round(np.asarray(straight), 9))) == len(points), (
        "the fixture's receivers must not all read the same, or any order would pass"
    )


def test_a_body_in_the_way_darkens_the_cells_behind_it():
    """The receiver mask is live in the transmittance, so a sleeve can be opened and closed
    without rebuilding the ``n^2`` geometry — and closing it must actually remove light."""
    surfaces = box(2, emission=3.0, reflectance=0.0)
    sleeve = Cylinder(centre=[0.5, 0.5, 0.5], axis=[0, 0, 1], radius=0.2, half_length=0.4)
    model = _volume_model(surfaces, np.array([[0.9, 0.5, 0.5]]), occluders=[sleeve])
    opaque, _ = fluence_rate(model, surfaces, transmittance=[0.0])
    clear, _ = fluence_rate(model, surfaces, transmittance=[1.0])
    assert float(opaque[0]) < float(clear[0])
    np.testing.assert_allclose(np.asarray(clear), 4.0 * 3.0, rtol=1e-12)


def test_the_gradient_of_the_field_in_reflectance_is_exact():
    """The whole point of the package: a derivative that reaches through the interreflection
    solve *and* the volume gather, not merely through one of them."""
    surfaces = _lit_box(reflectance=0.6)
    model = _volume_model(surfaces, INSIDE_THE_BOX)

    def total(reflectance):
        field, _ = fluence_rate(model, surfaces.with_optics(reflectance=reflectance))
        return jnp.sum(field)

    gradient = float(jax.grad(total)(jnp.asarray(0.6)))
    assert gradient == pytest.approx(_central_difference(total, 0.6), rel=1e-6)
    assert gradient > 0.0
