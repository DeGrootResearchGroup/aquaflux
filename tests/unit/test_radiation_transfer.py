"""The transfer matrix: the geometry of what reaches what, before any optics."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from aquaflux.radiation.profiles import Isotropic, Lambertian
from aquaflux.radiation.self_occlusion import (
    NoOcclusion,
    RayCastOcclusion,
    SilhouetteOcclusion,
)
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.transfer import build_transfer, reciprocity_residual, row_sum_error

from tests.unit.radiation_references import (
    box,
    facing_plates,
    inward_box,
    rectangle_triangles,
    stretched_box,
)

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


def _box_with_a_lamp():
    """48 wall facets and a point source in the middle: 49 rows, which few block sizes divide.

    The point source sits LAST and a wall facet first, so a block that masks the point source or
    the diagonal by its own local row index rather than the global one gets the wrong rows.
    """
    vertices = np.concatenate([inward_box(2), np.full((1, 3, 3), 0.5)])
    return Surfaces.from_triangles(
        vertices, profiles=(Lambertian(), Isotropic()), profile_index=[0] * 48 + [1]
    )


@pytest.mark.parametrize("chunk_size", [5, 7, 48])
def test_the_matrix_does_not_depend_on_how_its_rows_are_blocked(chunk_size):
    """Blocking is a memory strategy; it must not be a numerical one.

    None of these sizes divides 49, so each build ends with a block that starts early and
    recomputes rows its predecessor already wrote -- the one piece of index arithmetic the
    blocked build has. Agreement is to a rounding of the O(1) row sums and not bit for bit: a
    different block shape compiles to a differently fused program, and a pair of facets on one
    wall, whose transfer is zero, comes back as dust of order 1e-17 that moves with it. A wrong
    row, or a mask applied at the wrong index, is off by the size of a transfer factor.
    """
    surfaces = _box_with_a_lamp()
    whole = build_transfer(surfaces, self_occlusion=NoOcclusion(), chunk_size=1000)
    blocked = build_transfer(surfaces, self_occlusion=NoOcclusion(), chunk_size=chunk_size)
    for name in ("geometric", "source_cosine", "separation"):
        np.testing.assert_allclose(
            np.asarray(getattr(blocked, name)), np.asarray(getattr(whole, name)), rtol=0, atol=1e-15
        )
    matrix = np.asarray(blocked.geometric)
    assert np.all(np.diag(matrix) == 0.0), "a facet transfers nothing to itself"
    assert np.count_nonzero(matrix > 1e-6) > 0.5 * 48 * 47, "the walls do see one another"
    assert np.all(matrix[-1] == 0.0) and np.all(matrix[:, -1] == 0.0), "nor does a point source"


def test_no_block_handed_to_the_compiled_pass_exceeds_the_chunk(monkeypatch):
    """What bounds the build's working set, pinned by the blocks it computes rather than by timing."""
    from aquaflux.radiation import transfer

    rows_seen = []
    real = transfer._row_block

    def watched(geometry, sample, weight, start, *, rows):
        rows_seen.append((int(start), rows))
        return real(geometry, sample, weight, start, rows=rows)

    monkeypatch.setattr(transfer, "_row_block", watched)
    build_transfer(_box_with_a_lamp(), self_occlusion=NoOcclusion(), chunk_size=7)
    assert all(rows == 7 for _, rows in rows_seen), rows_seen
    covered = set()
    for start, rows in rows_seen:
        covered.update(range(start, start + rows))
    assert covered == set(range(49)), "every row is computed, and none beyond the matrix"


@pytest.mark.parametrize("chunk_size", [7, 1000])
def test_a_point_source_is_left_out_by_its_label_not_by_its_area(chunk_size):
    """A facet labelled a point source takes no part in the transfer even when it has an area.

    The kernel already returns nothing for a zero-area triangle, so a point source built the usual
    way cannot show whether the label is honoured -- the mask could vanish and every fixture of
    that kind would still pass. A traced lamp keeps its labels given explicitly, so a labelled
    facet may well have an area. Put one late in the set so that, blocked by seven, it lies in a
    block that does not start at zero.
    """
    labelled = 45
    surfaces = Surfaces.from_triangles(
        inward_box(2),
        profiles=(Lambertian(), Isotropic()),
        profile_index=[1 if i == labelled else 0 for i in range(48)],
        point_sources=[labelled],
    )
    assert float(np.asarray(surfaces.area)[labelled]) > 0.0, "the fixture must have an area"
    matrix = np.asarray(
        build_transfer(surfaces, self_occlusion=NoOcclusion(), chunk_size=chunk_size).geometric
    )
    np.testing.assert_array_equal(matrix[labelled, :], 0.0)
    np.testing.assert_array_equal(matrix[:, labelled], 0.0)
    assert np.count_nonzero(matrix[labelled - 1] > 1e-6) > 10, "its neighbour still transfers"


#: Walton's obstructed view factor: two directly opposed unit squares one unit apart, with a
#: centred 0.5 x 0.5 square blocker parallel to them, three quarters of the way from the first
#: to the second (Walton, NISTIR 6925, the "Shapiro" test). Reproduced independently to
#: 0.1156206021 by integrating point-to-rectangle view factors over the first square with
#: Gauss-Legendre quadrature, where the blocker's shadow is a square wholly inside the second.
WALTON_OBSTRUCTED_SQUARES = 0.11562061

#: The two bodies of the obstructed pair, both zero-thickness sheets.
SHEETS = ("plates", "blocker")


def _obstructed_squares(n: int, strategy) -> tuple[float, float]:
    """``F_12`` and ``F_21`` for Walton's obstructed pair, each plate meshed ``n x n``.

    The blocker is a single one-sided sheet, as a baffle usually is in a surface file. The
    plates and the blocker are all zero-thickness sheets, and named so, which the silhouette
    clip needs in order to count a sheet seen from behind.
    """
    blocker = rectangle_triangles([0.0, 0.0, 0.25], [0.25, 0.0, 0.0], [0.0, 0.25, 0.0])
    surfaces = Surfaces.from_triangles(
        np.concatenate([facing_plates(n, half=0.5, gap=0.5), blocker]),
        reflectance=0.0,
        solid_id=[0] * (4 * n * n) + [1] * len(blocker),
        solid_names=SHEETS,
    )
    matrix, _ = build_transfer(surfaces, self_occlusion=strategy).assemble(surfaces)
    matrix, area = np.asarray(matrix), np.asarray(surfaces.area)
    lower, upper = slice(0, 2 * n * n), slice(2 * n * n, 4 * n * n)

    def plate_to_plate(receiver, source):
        return float(area[receiver] @ matrix[receiver, source].sum(axis=1) / area[receiver].sum())

    return plate_to_plate(lower, upper), plate_to_plate(upper, lower)


@pytest.mark.parametrize(
    "strategy",
    [RayCastOcclusion(), SilhouetteOcclusion(two_sided=SHEETS)],
    ids=["ray", "silhouette"],
)
def test_an_obstructed_pair_converges_on_the_published_view_factor(strategy):
    """Case 8c: an exact answer for occlusion between areas, in closed form.

    The analytic-body tests pin *where* a shadow falls; this pins how much of a view factor it
    removes once both areas are integrated, and so exercises the mask, the receiver quadrature
    and the area weighting together. Both directions are checked, since the pair is not
    symmetric -- the blocker is three times nearer one plate -- while the view factor is,
    the two plates having equal areas.

    Unobstructed, the pair's view factor is 0.1998; the blocker takes away 42% of it. Both
    strategies converge on the published value at second order in the plate spacing. With the
    blocker near the source (the lower plate receiving) the silhouette is six times more
    accurate than one ray per pair, at 1.6e-4 and 4.0e-5 at 6 and 12 plates a side; with it
    near the receiver every shadow edge of this fixture lands on a facet edge, so there the two
    strategies agree exactly.
    """
    errors = np.array(
        [np.subtract(_obstructed_squares(n, strategy), WALTON_OBSTRUCTED_SQUARES) for n in (6, 12)]
    )
    rates = np.abs(errors[0] / errors[1])
    assert np.all((rates > 3.5) & (rates < 4.8)), f"not second order: {rates}"
    assert np.all(np.abs(errors[1]) < 3e-4), f"errors at 12 plates a side: {errors[1]}"


def test_an_undeclared_sheet_is_warned_about_and_hides_nothing_from_behind():
    """The failure the declaration exists for, pinned so it cannot come back quietly.

    Counted only from the side it faces, the one-sided blocker hides nothing from the plate
    behind it -- that plate sees the unobstructed 0.1998 -- while still shadowing the plate it
    faces. So the build must say so, naming the bodies whose pieces have a free edge.
    """
    with pytest.warns(UserWarning, match=r"free edge.*two_sided") as caught:
        behind, facing = _obstructed_squares(6, SilhouetteOcclusion())
    assert "'blocker'" in str(caught[0].message) and "'plates'" in str(caught[0].message)
    clear, _ = _obstructed_squares(6, NoOcclusion())
    assert behind == pytest.approx(clear, rel=1e-12)
    assert facing == pytest.approx(WALTON_OBSTRUCTED_SQUARES, abs=2e-3)


def test_declaring_every_sheet_silences_the_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _obstructed_squares(2, SilhouetteOcclusion(two_sided=SHEETS))


def test_a_misspelt_sheet_is_refused_rather_than_left_one_sided():
    with pytest.raises(ValueError, match=r"two_sided names no body.*\['blokcer'\]"):
        _obstructed_squares(2, SilhouetteOcclusion(two_sided=("plates", "blokcer")))
