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


# ---------------------------------------------------------------------------------------
# The transfer matrix
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("divisions", [1, 2, 4])
def test_every_row_of_the_transfer_matrix_sums_to_one_in_a_closed_box(divisions):
    """The identity the whole solve rests on, and the one that is exact.

    All the light leaving a facet inside a closed enclosure lands somewhere. That bounds the
    spectral radius of ``diag(rho) F`` by the largest reflectance, which is what makes the system
    well conditioned at any reflectance below one. Substituting the *plain* solid angle for the
    projected one doubles every row sum, and the centroid approximation pins the maximum near
    1.2 under every refinement; both are visible here immediately.
    """
    transfer = build_transfer(box(divisions), self_occlusion=False)
    assert row_sum_error(transfer) < 1e-12


def test_a_facet_does_not_transfer_to_itself():
    transfer = build_transfer(box(1), self_occlusion=False)
    np.testing.assert_allclose(np.diag(np.asarray(transfer.geometric)), 0.0, atol=0.0)


def test_reciprocity_is_violated_by_a_fixed_amount_that_refinement_does_not_reduce():
    """A documented limitation, pinned so it is not mistaken for a gate.

    Reciprocity is exact for the double-area-integral form factor. This matrix integrates the
    *source* exactly and evaluates the *receiver* at its centroid, and that one-point rule breaks
    reciprocity by an amount refinement leaves alone — shrinking the facets brings their
    neighbours proportionally closer, so the geometry stays self-similar. The row sums are exact
    regardless, which is what the conditioning actually needs.
    """
    residuals = [
        reciprocity_residual(build_transfer(box(n), self_occlusion=False), box(n).area)
        for n in (1, 2, 4)
    ]
    assert all(0.2 < value < 0.3 for value in residuals), residuals
    assert max(residuals) - min(residuals) < 1e-6, "it converged; the docstring is now wrong"


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


def test_a_closed_box_reaches_its_closed_form_AT_THE_DEFAULT_SETTINGS():
    """Every other test in this file switches self-occlusion off, and that hid a bug.

    ``build_transfer``'s receivers are the facet centroids themselves, so each self-occlusion
    ray ends on the facet it is aimed at — which counted as a hit. The default therefore
    reported every mutually visible pair as blocked, and a closed box came back with ``B = M``:
    at a reflectance of 0.9 that is ten times too dark, and it looks like a field rather than an
    error. No test could see it, because every fixture passed ``self_occlusion=False``, and the
    self-occlusion tests all use receivers out in the volume where the ray ends on nothing.

    ⚠️ **``row_sum_error`` is blind to this by construction** and cannot be made to catch it:
    the mask is applied live in ``_live_transfer``, not baked into ``geometric``, so the gate
    reads 1e-15 while the matrix it is reporting on is being zeroed downstream.
    """
    exitance, reflectance = 3.0, 0.9
    surfaces = box(2, emission=exitance, reflectance=reflectance)
    outgoing, _ = radiosity(build_transfer(surfaces), surfaces)
    np.testing.assert_allclose(np.asarray(outgoing), exitance / (1.0 - reflectance), rtol=1e-12)


def test_a_convex_enclosure_measures_the_same_with_self_occlusion_on_and_off():
    """A box is convex, so its facets shadow nothing and the mask must change no number at all.

    Bit-identical rather than merely close: the mask multiplies the transfer elementwise, so an
    all-clear mask multiplies by exactly one. Anything less than exact equality means it is not
    all-clear. This is also what licenses every other measurement in this file, all of which are
    taken with the mask off.
    """
    surfaces = box(2, emission=3.0, reflectance=0.9)
    area = np.asarray(surfaces.area)
    with_mask = build_transfer(surfaces, self_occlusion=True)
    without = build_transfer(surfaces, self_occlusion=False)
    assert not bool(np.any(np.asarray(with_mask.visibility.blocked_by_geometry)))
    assert row_sum_error(with_mask) == row_sum_error(without)
    assert reciprocity_residual(with_mask, area) == reciprocity_residual(without, area)
    np.testing.assert_array_equal(
        np.asarray(radiosity(with_mask, surfaces)[0]),
        np.asarray(radiosity(without, surfaces)[0]),
    )


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

    The source's own distribution is evaluated at the centroid direction, the same one-point rule
    the receiver gets. For a Lambertian source it cancels against the projected solid angle and
    the total landing on a closed box equals the total leaving, exactly, at every refinement. For
    a cosine-power source it does not: measured balances are 1.145 / 0.970 / 0.970 / 0.981 at 12,
    48, 192 and 768 facets for an exponent of 8, and 1.069 / 1.007 / 0.995 / 0.995 for an
    exponent of 2. It shrinks as the receiving facets shrink and more directions are sampled.
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
    # What the lamp emits is what the walls swallow, and the closeness of that balance is
    # worth a number rather than a tolerance pulled from the air. It is NOT exact, because
    # global conservation needs reciprocity and this matrix evaluates the receiver at a point:
    # per column the identity ``sum_i A_i F_ij = A_j`` is violated by up to 8.9%. Summed over
    # a whole enclosure those column errors cancel almost entirely, leaving a balance measured
    # at 1.000000 / 0.999868 / 1.000112 / 1.000083 / 1.000062 on boxes of 12 to 768 facets.
    area = np.asarray(surfaces.area)
    landing, _ = surface_irradiance(build_transfer(surfaces, self_occlusion=False), surfaces)
    emitted = float(np.sum(area * np.asarray(surfaces.emission)))
    absorbed = float(np.sum(area * (1.0 - 0.9) * np.asarray(landing)))
    assert absorbed == pytest.approx(emitted, rel=1e-3)


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
