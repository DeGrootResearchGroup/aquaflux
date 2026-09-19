"""Interreflection between surfaces: the transfer matrix, the solve, and what stays live."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from aquaflux.radiation.absorption import UniformAbsorption
from aquaflux.radiation.occluders import Cylinder
from aquaflux.radiation.profiles import CosinePower, Isotropic, Lambertian
from aquaflux.radiation.radiosity import (
    build_transfer,
    radiosity,
    reciprocity_residual,
    row_sum_error,
    surface_irradiance,
)
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.solve import relative_residual_gmres

from tests.unit.radiation_references import inward_box, rectangle_triangles


def box(divisions=2, **optics):
    return Surfaces.from_triangles(inward_box(divisions), **optics)


def stretched_box(divisions=2, **optics):
    """A closed box whose facets do **not** all have the same area.

    Stretching the unit box along one axis is affine and positive, so it stays closed and stays
    consistently wound, but its long walls carry triangles three times the area of its ends.
    Every equal-area fixture is blind to which index of the transfer matrix an area belongs on,
    because both choices are then the same expression — see the reciprocity tests below.
    """
    return Surfaces.from_triangles(inward_box(divisions) * np.array([1.0, 1.0, 3.0]), **optics)


# ---------------------------------------------------------------------------------------
# The transfer matrix
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("divisions", [1, 2, 4])
def test_every_row_of_the_transfer_matrix_sums_to_one_in_a_closed_box(divisions):
    """The identity the whole solve rests on, and the one that is exact.

    All the light leaving a facet inside a closed enclosure lands somewhere. That bounds the
    spectral radius of ``diag(rho) F`` by the largest reflectance, which is what makes the system
    well conditioned at any reflectance below one. Substituting the *plain* solid angle for the
    projected one doubles every row sum, and an elementary ``A cos(theta) / r^2`` kernel in place
    of the closed form pins the maximum near 1.2 under every refinement; both are visible here
    immediately.
    """
    transfer = build_transfer(box(divisions), self_occlusion=False)
    assert row_sum_error(transfer) < 1e-12


@pytest.mark.parametrize("n_points", [1, 3, 6, 12])
def test_the_row_sums_stay_exact_however_finely_the_receiver_is_integrated(n_points):
    """The reason the *source* is left in closed form rather than also being sampled.

    Each quadrature point of a receiver sees a whole closed enclosure, so its own row sums to
    one; the weights sum to one, so their average does too — at every rule, on both a cube and
    an enclosure whose facets differ in area. Integrating both facets by quadrature would buy
    exact reciprocity and give this up, which is the worse trade: this is what bounds the
    conditioning, and what catches a wrong kernel.
    """
    for surfaces in (box(2), stretched_box(2)):
        transfer = build_transfer(surfaces, self_occlusion=False, receiver_quadrature=n_points)
        assert row_sum_error(transfer) < 1e-12


def test_the_default_build_integrates_the_receiver_over_six_points():
    """A default worth pinning: it is the point at which the measured cost/accuracy frontier
    turns over, and it is what every other test in this file is measured under."""
    surfaces = box(2)
    default = build_transfer(surfaces, self_occlusion=False)
    six = build_transfer(surfaces, self_occlusion=False, receiver_quadrature=6)
    one = build_transfer(surfaces, self_occlusion=False, receiver_quadrature=1)
    np.testing.assert_array_equal(np.asarray(default.geometric), np.asarray(six.geometric))
    assert not np.allclose(np.asarray(default.geometric), np.asarray(one.geometric))


def test_a_facet_does_not_transfer_to_itself():
    transfer = build_transfer(box(1), self_occlusion=False)
    np.testing.assert_allclose(np.diag(np.asarray(transfer.geometric)), 0.0, atol=0.0)


def test_reciprocity_converges_in_the_number_of_points_on_the_receiver():
    """The knob that fixes it, and the only one that does.

    Reciprocity is exact for the double-area-integral form factor. Integrating the source exactly
    and sampling the receiver at one point breaks it; integrating the receiver too restores it, in
    the point count. Measured on a cube: 0.2421, 0.0281, 0.0078, 0.0045 at 1, 3, 6 and 12 points.
    """
    surfaces = box(2)
    residuals = [
        reciprocity_residual(
            build_transfer(surfaces, self_occlusion=False, receiver_quadrature=n), surfaces.area
        )
        for n in (1, 3, 6, 12)
    ]
    assert residuals == sorted(residuals, reverse=True), residuals
    assert residuals[0] > 0.2, "the one-point rule should still be the bad one"
    assert residuals[-1] < 0.01, residuals
    assert residuals[0] / residuals[-1] > 20.0, residuals


@pytest.mark.parametrize("n_points", [1, 3, 6, 12])
def test_refining_the_mesh_does_not_improve_reciprocity_at_any_rule(n_points):
    """The counterintuitive half, and the reason the quadrature is the fix rather than a finer
    mesh: refining a closed box brings each facet's neighbours proportionally closer, so the
    ratio the quadrature error depends on never changes and the residual sits flat."""
    residuals = [
        reciprocity_residual(
            build_transfer(box(n), self_occlusion=False, receiver_quadrature=n_points), box(n).area
        )
        for n in (1, 2, 4)
    ]
    assert max(residuals) - min(residuals) < 1e-6, residuals


def test_the_area_weighting_of_reciprocity_multiplies_the_row_not_the_column():
    """``geometric[i, j]`` is the form factor *from* ``i`` *to* ``j``, so reciprocity pairs
    ``A_i F_ij`` with ``A_j F_ji``.

    Putting the area on the column index is the identical expression on any fixture whose facets
    share one area — which every box built by subdividing a cube does, so the cube tests above
    cannot see the difference at all. On an enclosure stretched to three times its length the
    correct pairing converges with the quadrature while the transposed one sits near 0.9 at every
    rule; that gap is what this test is for.
    """
    surfaces = stretched_box(2)
    area = np.asarray(surfaces.area)
    assert len(np.unique(np.round(area, 12))) > 1, "the fixture must have unequal areas"

    residuals = [
        reciprocity_residual(
            build_transfer(surfaces, self_occlusion=False, receiver_quadrature=n), area
        )
        for n in (1, 3, 6, 12)
    ]
    assert residuals == sorted(residuals, reverse=True), residuals
    assert residuals[-1] < 0.02, residuals

    # What the transposed weighting would report on the same matrices: near 0.9 throughout, and
    # flat, so it is distinguishable both by size and by its refusal to converge.
    matrix = np.asarray(build_transfer(surfaces, self_occlusion=False).geometric)
    transposed = area[None, :] * matrix
    mismatch = np.max(np.abs(transposed - transposed.T)) / np.max(np.abs(transposed))
    assert mismatch > 0.5, mismatch


def test_reciprocity_is_near_exact_between_two_facets_far_apart():
    """Separating the two error sources: with the facets a hundred widths apart the transfer
    integrand barely varies over either of them, so the quadrature is not the limitation and what
    is left is the weighting. Unequal areas — 1 against 100 — make a wrong index unmissable.
    """
    corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])

    def square(half, height, facing_up):
        plane = np.concatenate([corners * 2.0 * half, np.full((4, 1), height)], axis=1)
        triangles = np.array([plane[[0, 1, 2]], plane[[0, 2, 3]]])
        return triangles if facing_up else triangles[:, ::-1, :]

    surfaces = Surfaces.from_triangles(
        np.concatenate([square(0.5, 0.0, True), square(5.0, 100.0, False)])
    )
    area = np.asarray(surfaces.area)
    np.testing.assert_allclose(np.sort(area), [0.5, 0.5, 50.0, 50.0])
    assert reciprocity_residual(build_transfer(surfaces, self_occlusion=False), area) < 1e-6


def test_point_sources_take_no_part_in_the_transfer():
    """They have no area to emit from and no surface to receive on; they enter as an external
    irradiance instead."""
    vertices = np.concatenate([inward_box(1), np.zeros((1, 3, 3))])
    surfaces = Surfaces.from_triangles(
        vertices, profiles=(Lambertian(), Isotropic()), profile_index=[0] * 12 + [1]
    )
    transfer = build_transfer(surfaces, self_occlusion=False)
    matrix = np.asarray(transfer.geometric)
    np.testing.assert_allclose(matrix[-1, :], 0.0, atol=0.0)
    np.testing.assert_allclose(matrix[:, -1], 0.0, atol=0.0)


def test_a_body_in_the_way_removes_the_transfer_across_it():
    """Occlusion multiplies the same matrix, so a blocked pair simply stops exchanging."""
    facing = np.concatenate(
        [
            rectangle_triangles([0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            rectangle_triangles([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ]
    )
    surfaces = Surfaces.from_triangles(facing, emission=1.0, reflectance=0.5)
    clear = build_transfer(surfaces, self_occlusion=False)
    blocked = build_transfer(
        surfaces,
        occluders=[Cylinder(centre=[0, 0, 0], axis=[1, 0, 0], radius=0.4, half_length=4.0)],
        self_occlusion=False,
    )
    assert float(jnp.sum(clear.geometric)) > 0.0
    assert float(jnp.sum(blocked.visibility.blocked)) > 0.0
    outgoing, _ = radiosity(blocked, surfaces, transmittance=[0.0])
    np.testing.assert_allclose(np.asarray(outgoing), 1.0, rtol=1e-14)


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
    outgoing, _ = radiosity(build_transfer(surfaces, self_occlusion=False), surfaces)
    np.testing.assert_allclose(np.asarray(outgoing), exitance / (1.0 - reflectance), rtol=1e-12)


def test_the_irradiance_matches_what_the_radiosity_implies():
    """``B = M + rho H`` must hold facet by facet, or the two are computing different systems."""
    exitance, reflectance = 3.0, 0.7
    surfaces = box(2, emission=exitance, reflectance=reflectance)
    transfer = build_transfer(surfaces, self_occlusion=False)
    outgoing, _ = radiosity(transfer, surfaces)
    landing, _ = surface_irradiance(transfer, surfaces)
    np.testing.assert_allclose(
        np.asarray(outgoing), exitance + reflectance * np.asarray(landing), rtol=1e-12
    )


def test_the_bounce_count_is_not_a_parameter():
    """The inverse *is* the infinite bounce sum, so the solve must match a long Neumann series
    and must not match a short one."""
    reflectance = 0.8
    surfaces = box(2, emission=1.0, reflectance=reflectance)
    transfer = build_transfer(surfaces, self_occlusion=False)
    exact, _ = radiosity(transfer, surfaces)

    matrix = np.asarray(transfer.geometric) * reflectance
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
    transfer = build_transfer(surfaces, self_occlusion=False)
    lit = np.full(surfaces.n_facets, 2.0)
    outgoing, _ = radiosity(transfer, surfaces, external_irradiance=lit)
    # Each facet re-emits half of what arrives, and what it re-emits comes back round the box.
    assert float(jnp.min(outgoing)) > 0.5 * 2.0
    dark, _ = radiosity(transfer, surfaces)
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
    landing, _ = surface_irradiance(build_transfer(surfaces, self_occlusion=False), surfaces)
    assert bool(jnp.isnan(landing[-1]))
    assert bool(jnp.all(jnp.isfinite(landing[:-1])))


# ---------------------------------------------------------------------------------------
# Non-Lambertian sources
# ---------------------------------------------------------------------------------------


def test_a_lambertian_source_transfers_its_emission_exactly_as_it_transfers_a_reflection():
    """The reduction that pins the profile constants: for a Lambertian emitter the emitted and
    reflected transfer matrices are the same matrix, so the whole default path collapses to one."""
    surfaces = box(2, emission=1.0, reflectance=0.6)
    transfer = build_transfer(surfaces, self_occlusion=False)
    from aquaflux.radiation.radiosity import _live_transfer

    reflected, emitted = _live_transfer(transfer, surfaces, None, None)
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
    transfer = build_transfer(diffuse, self_occlusion=False)

    facing, _ = surface_irradiance(transfer, diffuse)
    beamed, _ = surface_irradiance(transfer, narrow)
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
        landing, _ = surface_irradiance(build_transfer(facets, self_occlusion=False), facets)
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
    transfer = build_transfer(surfaces, self_occlusion=False)

    def total(reflectance):
        outgoing, _ = radiosity(transfer, surfaces.with_optics(reflectance=reflectance))
        return jnp.sum(outgoing)

    gradient = float(jax.grad(total)(jnp.asarray(0.6)))
    assert gradient == pytest.approx(_central_difference(total, 0.6), rel=1e-6)
    assert gradient > 0.0


def test_the_gradient_in_emission_is_exact():
    surfaces = _lit_box(reflectance=0.6)
    transfer = build_transfer(surfaces, self_occlusion=False)
    base = jnp.asarray(surfaces.emission)

    def total(scale):
        outgoing, _ = radiosity(transfer, surfaces.with_optics(emission=base * scale))
        return jnp.sum(outgoing)

    assert float(jax.grad(total)(jnp.asarray(1.0))) == pytest.approx(
        _central_difference(total, 1.0), rel=1e-6
    )


def test_the_gradient_reaches_an_occluder_s_transmittance_through_the_solve():
    """One of the two ways this split has been got wrong. Freezing the visibility inside the
    geometry term leaves a finite, plausible number here that is short by about two thirds."""
    surfaces = _lit_box(reflectance=0.6)
    transfer = build_transfer(
        surfaces,
        occluders=[Cylinder(centre=[0.5, 0.5, 0.5], axis=[0, 0, 1], radius=0.2, half_length=0.3)],
        self_occlusion=False,
    )
    assert int(jnp.sum(transfer.visibility.blocked)) > 0, "the body blocks nothing"

    def total(value):
        outgoing, _ = radiosity(transfer, surfaces, transmittance=jnp.asarray([value]))
        return jnp.sum(outgoing)

    gradient = float(jax.grad(total)(jnp.asarray(0.4)))
    assert gradient == pytest.approx(_central_difference(total, 0.4), rel=1e-6)
    assert abs(gradient) > 0.0


def test_the_gradient_reaches_the_absorption_coefficient_through_the_solve():
    """The other one. Freezing the whole transfer matrix costs a few percent of this and leaves
    the rest looking healthy."""
    surfaces = _lit_box(reflectance=0.6)
    transfer = build_transfer(surfaces, self_occlusion=False)

    def total(coefficient):
        outgoing, _ = radiosity(transfer, surfaces, absorption=UniformAbsorption(coefficient))
        return jnp.sum(outgoing)

    gradient = float(jax.grad(total)(jnp.asarray(0.5)))
    assert gradient == pytest.approx(_central_difference(total, 0.5), rel=1e-6)
    assert gradient < 0.0, "more absorbance, less light"


def test_the_gradient_reaches_a_source_s_profile_parameter():
    surfaces = _lit_box(reflectance=0.6, profiles=(CosinePower(3.0),))
    transfer = build_transfer(surfaces, self_occlusion=False)

    def total(exponent):
        narrowed = _with_profile(surfaces, CosinePower(exponent))
        outgoing, _ = radiosity(transfer, narrowed)
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
    transfer = build_transfer(surfaces, self_occlusion=False)

    def make(restart):
        def total(reflectance):
            outgoing, _ = radiosity(
                transfer,
                surfaces.with_optics(reflectance=reflectance),
                solver=relative_residual_gmres(1e-12, restart=restart),
            )
            return jnp.sum(outgoing)

        return total

    counts = []
    for restart in (2, 120):
        _, steps = radiosity(
            transfer, surfaces, solver=relative_residual_gmres(1e-12, restart=restart)
        )
        counts.append(int(steps))
    assert counts[0] != counts[1], f"both arms took the same path: {counts}"

    gradients = [float(jax.grad(make(restart))(jnp.asarray(0.9))) for restart in (2, 120)]
    assert gradients[0] == pytest.approx(gradients[1], rel=1e-8)


def test_the_expensive_geometry_is_frozen():
    """The solid angles are the ``n^2`` build, and nothing that varies per evaluation may force
    it again — so they carry no derivative with respect to where the facets are."""
    surfaces = _lit_box(reflectance=0.6)
    transfer = build_transfer(surfaces, self_occlusion=False)
    gradient = jax.grad(lambda matrix: jnp.sum(matrix))(transfer.geometric)
    assert gradient.shape == transfer.geometric.shape
    # stop_gradient is applied at the build, so differentiating the build gives nothing back.
    moved = jax.grad(
        lambda scale: jnp.sum(
            build_transfer(
                surfaces.with_geometry(jnp.asarray(inward_box(2)) * scale), self_occlusion=False
            ).geometric
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
    outgoing, _ = radiosity(build_transfer(surfaces, self_occlusion=False), surfaces)
    assert float(jnp.min(outgoing)) > 0.0, "reflection reaches every facet"
    assert float(jnp.max(outgoing)) > float(jnp.min(outgoing))
    # What the lamp emits is what the walls swallow. Global conservation needs reciprocity, so
    # it is not exact, but at the default six-point rule it is close: absorbed over emitted
    # measures 1.000000 / 1.000212 / 1.000171 / 1.000120 / 1.000102 on boxes of 12, 48, 192, 432
    # and 768 facets, with this scene's four emitting facets and reflectance 0.9 throughout.
    area = np.asarray(surfaces.area)
    landing, _ = surface_irradiance(build_transfer(surfaces, self_occlusion=False), surfaces)
    emitted = float(np.sum(area * np.asarray(surfaces.emission)))
    absorbed = float(np.sum(area * (1.0 - 0.9) * np.asarray(landing)))
    assert absorbed == pytest.approx(emitted, rel=1e-3)


def _absorbed_over_emitted(surfaces, n_points):
    area = np.asarray(surfaces.area)
    transfer = build_transfer(surfaces, self_occlusion=False, receiver_quadrature=n_points)
    landing, _ = surface_irradiance(transfer, surfaces)
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
    transfer = build_transfer(surfaces, self_occlusion=False)

    reference, cycles = radiosity(transfer, surfaces)
    assert int(cycles) <= 5
    assert bool(jnp.all(jnp.isfinite(reference)))

    with pytest.raises(Exception, match=r"(?i)stagnation|diverge|singular|not converge"):
        radiosity(transfer, surfaces, solver=lx.GMRES(rtol=1e-10, atol=0.0))
