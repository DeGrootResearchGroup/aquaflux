"""The transfer matrix: the geometry of what reaches what, before any optics."""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.radiation.profiles import Isotropic, Lambertian
from aquaflux.radiation.self_occlusion import NoOcclusion, RayCastOcclusion
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.transfer import build_transfer, reciprocity_residual, row_sum_error

from tests.unit.radiation_references import box, inward_box, stretched_box

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
    transfer = build_transfer(box(divisions), self_occlusion=NoOcclusion())
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
        transfer = build_transfer(
            surfaces, self_occlusion=NoOcclusion(), receiver_quadrature=n_points
        )
        assert row_sum_error(transfer) < 1e-12


def test_the_default_build_integrates_the_receiver_over_six_points():
    """A default worth pinning: it is the point at which the measured cost/accuracy frontier
    turns over, and it is what every other test in this file is measured under."""
    surfaces = box(2)
    default = build_transfer(surfaces, self_occlusion=NoOcclusion())
    six = build_transfer(surfaces, self_occlusion=NoOcclusion(), receiver_quadrature=6)
    one = build_transfer(surfaces, self_occlusion=NoOcclusion(), receiver_quadrature=1)
    np.testing.assert_array_equal(np.asarray(default.geometric), np.asarray(six.geometric))
    assert not np.allclose(np.asarray(default.geometric), np.asarray(one.geometric))


def test_a_convex_enclosure_builds_the_same_matrix_with_self_occlusion_on_and_off():
    """A box is convex, so its facets shadow nothing and the mask must change no number at all.

    Bit-identical rather than merely close: the mask multiplies the transfer elementwise, so an
    all-clear mask multiplies by exactly one. Anything less than exact equality means it is not
    all-clear — which is how a ray that was blocked by the very facet it was aimed at went
    unnoticed. This is also what licenses every other measurement in this file, all of which are
    taken with the mask off because tracing rays between every pair of a few hundred facets to
    rediscover that a box is convex costs the whole suite for nothing.
    """
    surfaces = box(2)
    area = np.asarray(surfaces.area)
    with_mask = build_transfer(surfaces, self_occlusion=RayCastOcclusion())
    without = build_transfer(surfaces, self_occlusion=NoOcclusion())
    assert not bool(np.any(np.asarray(with_mask.visibility.hidden_by_geometry)))
    assert row_sum_error(with_mask) == row_sum_error(without)
    assert reciprocity_residual(with_mask, area) == reciprocity_residual(without, area)


def test_a_facet_does_not_transfer_to_itself():
    transfer = build_transfer(box(1), self_occlusion=NoOcclusion())
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
            build_transfer(surfaces, self_occlusion=NoOcclusion(), receiver_quadrature=n),
            surfaces.area,
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
            build_transfer(box(n), self_occlusion=NoOcclusion(), receiver_quadrature=n_points),
            box(n).area,
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
            build_transfer(surfaces, self_occlusion=NoOcclusion(), receiver_quadrature=n), area
        )
        for n in (1, 3, 6, 12)
    ]
    assert residuals == sorted(residuals, reverse=True), residuals
    assert residuals[-1] < 0.02, residuals

    # What the transposed weighting would report on the same matrices: near 0.9 throughout, and
    # flat, so it is distinguishable both by size and by its refusal to converge.
    matrix = np.asarray(build_transfer(surfaces, self_occlusion=NoOcclusion()).geometric)
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
    assert reciprocity_residual(build_transfer(surfaces, self_occlusion=NoOcclusion()), area) < 1e-6


def test_point_sources_take_no_part_in_the_transfer():
    """They have no area to emit from and no surface to receive on; they enter as an external
    irradiance instead."""
    vertices = np.concatenate([inward_box(1), np.zeros((1, 3, 3))])
    surfaces = Surfaces.from_triangles(
        vertices, profiles=(Lambertian(), Isotropic()), profile_index=[0] * 12 + [1]
    )
    transfer = build_transfer(surfaces, self_occlusion=NoOcclusion())
    matrix = np.asarray(transfer.geometric)
    np.testing.assert_allclose(matrix[-1, :], 0.0, atol=0.0)
    np.testing.assert_allclose(matrix[:, -1], 0.0, atol=0.0)
