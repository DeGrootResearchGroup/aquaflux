"""Build-time geometry checks, against the defects real surface files carry."""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.radiation.checks import check_winding, stored_normal_disagreement, winding_report

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
