"""Build-time geometry checks, against the defects real surface files carry."""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.radiation.checks import (
    check_profiles,
    check_winding,
    open_facets,
    stored_normal_disagreement,
    winding_report,
)

from tests.unit.radiation_references import closed_prism, inward_box, mid_box_sheet

#: Two triangles covering the unit square, consistently wound counter-clockwise.
SQUARE = np.array(
    [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    ]
)


def tetrahedron() -> np.ndarray:
    """A closed, consistently wound tetrahedron — no boundary edges anywhere."""
    a, b, c, d = (
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    return np.array([[a, c, b], [a, b, d], [a, d, c], [b, c, d]])


def test_a_consistently_wound_surface_passes():
    report = winding_report(SQUARE)
    assert report.consistent
    assert report.boundary_edges == 4
    assert report.nonmanifold_edges == 0
    assert report.merged_vertices == 4


def test_a_closed_body_has_no_boundary_edges():
    """Distinguishes a watertight body from one with a crack, which is a different defect."""
    report = winding_report(tetrahedron())
    assert report.consistent
    assert report.boundary_edges == 0
    assert report.merged_vertices == 4


def test_one_reversed_triangle_is_found_and_both_of_its_neighbours_named():
    """The defect this module exists for: a reversed facet emits nothing and says nothing.

    Its normal points into the solid, the source-side visibility clamp discards it, and the
    surface is simply dimmer over that patch with no error raised anywhere.
    """
    reversed_one = SQUARE.copy()
    reversed_one[1] = reversed_one[1][::-1]
    report = winding_report(reversed_one)
    assert not report.consistent
    assert len(report.conflicting_edges) == 1
    np.testing.assert_array_equal(report.conflicting_facets, [0, 1])


def test_a_reversed_triangle_in_a_closed_body_is_found():
    broken = tetrahedron()
    broken[2] = broken[2][::-1]
    assert not winding_report(broken).consistent


def test_check_winding_raises_and_says_what_to_do():
    reversed_one = SQUARE.copy()
    reversed_one[1] = reversed_one[1][::-1]
    with pytest.raises(ValueError, match="inconsistent triangle winding"):
        check_winding(reversed_one)


def test_check_winding_returns_the_report_when_it_passes():
    """So a caller wanting the boundary count does not run the analysis a second time."""
    assert check_winding(SQUARE).boundary_edges == 4


def test_vertices_that_differ_only_by_rounding_are_treated_as_one_point():
    """Triangles meeting at an edge repeat its endpoints as separate, inexact coordinates.

    Matching them exactly would report every shared edge as two boundary edges, and no
    conflict could ever be detected because no edge would be shared.
    """
    jittered = SQUARE.copy()
    jittered[1, 1] += 1e-13
    report = winding_report(jittered)
    assert report.merged_vertices == 4
    assert report.boundary_edges == 4


def test_a_genuine_gap_is_not_merged_away():
    """The tolerance must not be so generous that it welds a cracked surface shut."""
    cracked = SQUARE.copy()
    cracked[1, 1] += 1e-3
    report = winding_report(cracked, tolerance=1e-9)
    assert report.merged_vertices == 5
    assert report.boundary_edges == 6


def test_an_edge_shared_by_three_triangles_is_counted_as_non_manifold():
    """Not fatal for a gather, which never has to decide which side of a surface it is on."""
    third = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, 0.5, 1.0]]])
    report = winding_report(np.concatenate([SQUARE, third]))
    assert report.nonmanifold_edges == 1


def test_a_degenerate_triangle_does_not_register_as_a_winding_conflict():
    """A point source is a zero-area facet; its edges have no direction to disagree about."""
    with_point_source = np.concatenate([SQUARE, np.zeros((1, 3, 3))])
    assert winding_report(with_point_source).consistent


def test_a_zero_tolerance_is_refused():
    with pytest.raises(ValueError, match="tolerance must be positive"):
        winding_report(SQUARE, tolerance=0.0)


def test_normals_recorded_as_zero_are_not_a_disagreement():
    """Many exporters write zeros, and the winding is authoritative regardless."""
    assert len(stored_normal_disagreement(SQUARE, np.zeros((2, 3)))) == 0


def test_a_recorded_normal_pointing_the_other_way_is_reported():
    stored = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    np.testing.assert_array_equal(stored_normal_disagreement(SQUARE, stored), [1])


def test_a_recorded_normal_that_agrees_but_is_unnormalized_is_not_reported():
    """The format does not require a unit normal, so length must not be read as direction."""
    stored = np.array([[0.0, 0.0, 17.0], [0.0, 0.0, 0.004]])
    assert len(stored_normal_disagreement(SQUARE, stored)) == 0


def _surfaces(vertices, **kwargs):
    from aquaflux.radiation.surfaces import Surfaces

    return Surfaces.from_triangles(vertices, **kwargs)


def test_an_isotropic_profile_on_an_emitting_surface_is_refused():
    """Isotropic describes a point source. On a surface it has no radiance to give."""
    from aquaflux.radiation.profiles import Isotropic

    with pytest.raises(ValueError, match="Isotropic profile"):
        check_profiles(_surfaces(SQUARE, profiles=(Isotropic(),)))


def test_a_directional_profile_on_a_point_source_is_refused():
    """The silent one: a zero normal reads as a right angle, so the source contributes nothing
    and simply never appears in the field."""
    from aquaflux.radiation.profiles import CosinePower

    with pytest.raises(ValueError, match="point source"):
        check_profiles(_surfaces(np.zeros((1, 3, 3)), profiles=(CosinePower(2.0),)))


def test_a_set_pairing_each_kind_with_its_own_profile_passes():
    from aquaflux.radiation.profiles import Isotropic, Lambertian

    mixed = _surfaces(
        np.concatenate([SQUARE, np.zeros((1, 3, 3))]),
        profiles=(Lambertian(), Isotropic()),
        profile_index=[0, 0, 1],
    )
    check_profiles(mixed)


# ---------------------------------------------------------------------------------------
# Open pieces of surface
# ---------------------------------------------------------------------------------------


def test_a_single_sheet_is_open_and_a_closed_body_is_not():
    assert open_facets(SQUARE).all()
    assert not open_facets(tetrahedron()).any()
    assert not open_facets(inward_box(3)).any()


def test_a_free_sheet_inside_a_closed_body_is_told_apart_from_it():
    """Openness is a property of each connected piece, not of the whole file: a baffle floating
    inside a closed box is open, and the box around it stays closed."""
    box = inward_box(2)
    sheet = mid_box_sheet(2, span=0.6)
    found = open_facets(np.concatenate([box, sheet]))
    assert not found[: len(box)].any()
    assert found[len(box) :].all()


def test_every_triangle_of_an_open_piece_is_open_not_only_those_on_its_rim():
    """The interior triangles of a sheet share every edge they have, and are just as one-sided
    as its rim -- which is why openness is decided per piece and not per edge."""
    sheet = mid_box_sheet(4, span=0.6)
    report = winding_report(sheet)
    assert report.boundary_edges == 16  # the rim alone, 4 a side
    assert open_facets(sheet).all()


def test_a_duct_whose_end_caps_were_not_exported_reads_as_open():
    """Documented, because it is where the reading is wrong in the other direction: the wall
    of a solid, left open by an export, looks exactly like a sheet."""
    prism = closed_prism(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), 2.0)
    assert not open_facets(prism).any()
    walls = prism[np.arange(len(prism)) % 4 < 2]
    assert open_facets(walls).all()


def test_a_sheet_welded_in_all_the_way_round_reads_as_closed():
    """Documented, because it is where topology cannot see a sheet at all: each rim edge is used
    by the sheet and by the two wall triangles either side of it, three uses, which neither
    joins the sheet to the wall nor marks it open. This is why a sheet has to be named rather
    than inferred."""
    box = inward_box(2)
    sheet = mid_box_sheet(2, span=1.0)
    assert winding_report(np.concatenate([box, sheet])).nonmanifold_edges == 8
    assert not open_facets(np.concatenate([box, sheet])).any()


def test_no_facets_have_no_open_pieces():
    assert open_facets(np.zeros((0, 3, 3))).shape == (0,)
