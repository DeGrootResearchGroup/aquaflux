"""The binary per-pair occlusion mask: the instrument that measures it, and that it is real.

The full measurement lives in ``validation/radiation_partial_occlusion.py`` and takes a couple
of minutes; what is here is the cheap half that has to stay true for those numbers to mean
anything, so a change to the transfer build cannot quietly invalidate them without a failure.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.radiation.occluders import Cylinder
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.transfer import build_transfer

from tests.unit.radiation_references import area_average_onto, facing_plates, quad_of_facet

ROD = Cylinder(centre=[0, 0, 0], axis=[0, 1, 0], radius=0.3, half_length=4.0)


def _effective(n: int, occluders) -> tuple[np.ndarray, np.ndarray]:
    """The transfer a solve uses: frozen geometry times the live mask, bodies fully opaque."""
    surfaces = Surfaces.from_triangles(facing_plates(n), emission=1.0, reflectance=0.0)
    transfer = build_transfer(surfaces, occluders=occluders, self_occlusion=False)
    reflected, _ = transfer.assemble(
        surfaces, transmittance=np.zeros(len(occluders)) if occluders else None
    )
    return np.asarray(reflected), np.asarray(surfaces.area)


def _coarse_against_reference(n_coarse: int, n_reference: int, occluders):
    coarse, coarse_area = _effective(n_coarse, occluders)
    fine, fine_area = _effective(n_reference, occluders)
    got = area_average_onto(coarse, coarse_area, n_coarse, n_coarse)
    want = area_average_onto(fine, fine_area, n_reference, n_coarse)
    return np.abs(got - want) / np.max(want)


def test_with_nothing_in_the_way_the_refined_transfer_aggregates_to_the_coarse_one():
    """The control, and the reason the measured mask error is a measurement at all.

    Area-averaging a refined transfer onto the coarse patches *is* the coarse form factor, so
    with no occluder the two must agree — and they do, to about 2e-05 here, which is the
    difference between six quadrature points on one large receiving triangle and six on each of
    many small ones. Everything above that floor in the validation harness is the mask.

    This is also a real statement about the transfer build: if the receiver quadrature or the
    live assembly stopped being consistent under refinement, the floor would rise and every
    number recorded against it would silently become unreadable.
    """
    assert float(np.max(_coarse_against_reference(2, 8, ()))) < 1e-4


def test_the_binary_mask_is_orders_above_that_floor_on_a_partly_shadowed_pair():
    """The defect itself, at the smallest size that shows it.

    One ray per facet pair records a half-shadowed pair as wholly blocked or wholly clear. The
    error is ~7% of the largest transfer entry in the mean here, four orders above the control
    — so it is the mask and not the discretization. The validation harness measures how it
    behaves under refinement and against the size of the body; this only pins that it is there.
    """
    error = _coarse_against_reference(2, 8, (ROD,))
    assert float(np.mean(error)) > 1e-2
    assert float(np.max(error)) > 1e-1


def test_the_two_plates_actually_face_each_other():
    """A fixture guard, because half of it can fail silently.

    The second plate is built from the same template as the first and must be reversed, or its
    normal points away and the source-side clamp deletes every transfer *into* it. The matrix is
    then half zeros — and every measurement above still passes, because both the coarse and the
    refined answer are zero there and the comparison between them is perfectly consistent about
    a quantity that no longer exists. Measured: the un-reversed fixture reads 1.5986 one way and
    exactly 0.0 the other.
    """
    surfaces = Surfaces.from_triangles(facing_plates(2))
    transfer = np.asarray(build_transfer(surfaces, self_occlusion=False).geometric)
    half = surfaces.n_facets // 2
    assert float(transfer[:half, half:].sum()) > 0.1
    assert float(transfer[half:, :half].sum()) > 0.1


def test_the_aggregation_pairs_each_coarse_patch_with_its_own_two_triangles():
    """The map that was wrong once, and the check that would not have caught it.

    The two triangles of a quad are adjacent in the facet order, not in two blocks. Getting that
    wrong still gives every coarse patch exactly two facets on the correct plate — so a test
    that only counted them passed, and the control above was what actually caught it, reading
    0.398 instead of 5.7e-07. This asserts the stronger property: the pair a patch owns must
    average to that patch's own centre.
    """
    for n in (2, 4):
        surfaces = Surfaces.from_triangles(facing_plates(n))
        centroid = np.asarray(surfaces.centroid)
        owner = quad_of_facet(n, n)
        edges = np.linspace(-1.0, 1.0, n + 1)
        middle = (edges[:-1] + edges[1:]) / 2.0
        for quad in range(owner.max() + 1):
            pair = centroid[owner == quad]
            assert len(pair) == 2, f"quad {quad} owns {len(pair)} facets"
            row, column = divmod(quad % (n * n), n)
            np.testing.assert_allclose(pair.mean(axis=0)[:2], [middle[row], middle[column]])


@pytest.mark.parametrize(("fine", "coarse"), [(18, 4), (8, 3), (4, 0), (4, -1)])
def test_a_coarse_grid_that_does_not_divide_the_fine_one_is_refused(fine, coarse):
    """A trap the aggregation walked into: 18 does not divide by 4.

    Integer division then runs the owner index off the end of the coarse grid, and the failure
    surfaces as an out-of-range scatter deep inside ``area_average_onto`` — a message about
    numpy's internals rather than about the two meshes being incompatible.
    """
    with pytest.raises(ValueError, match="must divide"):
        quad_of_facet(fine, coarse)


@pytest.mark.parametrize("radius", [0.15, 0.6])
def test_a_body_of_any_size_leaves_the_mask_error_above_the_floor(radius):
    """Both regimes the harness sweeps: a rod thinner than a facet and one wider than several.

    A thin body's shadow can fall entirely between two ray endpoints or entirely inside one
    pair; a wide one blocks whole pairs correctly and errs only at its edges. Neither is small.
    """
    body = Cylinder(centre=[0, 0, 0], axis=[0, 1, 0], radius=radius, half_length=4.0)
    assert float(np.mean(_coarse_against_reference(2, 8, (body,)))) > 1e-2
