"""Whether a point is inside the solid — the check that a cell in the metal is not a dark cell."""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.radiation.checks import check_points_outside, enclosure_winding
from aquaflux.radiation.solid_angle import signed_solid_angle, solid_angle

from tests.unit.radiation_references import (
    cylinder_triangles,
    disc_triangles,
    inward_box,
    rectangle_triangles,
)


def _capped_cylinder(*, consistent: bool = True):
    """A geometrically closed tube with both end caps, and points in and out of it.

    A second closed shape whose facets are not axis-aligned, so nothing measured on it can be
    passing on the box's symmetry. ``disc_triangles`` winds both caps the same way, which for
    the lower one faces *into* the tube — so ``consistent`` reverses it, and leaving it False
    reproduces exactly the defect :func:`check_winding` exists to catch on a real file.
    """
    radius, half_length = 0.3, 0.5
    tube = cylinder_triangles(radius, half_length, sectors=32, slices=4)
    top = disc_triangles(radius, rings=4, sectors=32, height=half_length)
    bottom = disc_triangles(radius, rings=4, sectors=32, height=-half_length)
    return (
        np.concatenate([tube, top, bottom[:, ::-1, :] if consistent else bottom]),
        np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.4]]),
        np.array([[0.0, 0.0, 0.6], [0.5, 0.0, 0.0]]),
    )


INSIDE = np.array([[0.5, 0.5, 0.5], [0.1, 0.9, 0.3], [0.999, 0.5, 0.5]])
OUTSIDE = np.array([[1.5, 0.5, 0.5], [-0.2, 0.5, 0.5], [0.5, 0.5, 2.0], [1.001, 0.5, 0.5]])


# ---------------------------------------------------------------------------------------
# The signed kernel
# ---------------------------------------------------------------------------------------


def test_the_signed_kernel_agrees_with_the_unsigned_one_in_magnitude():
    """The public kernel is defined as the magnitude of this one, so nothing about the
    established behaviour may move — every measurement in the subsystem rests on it."""
    rng = np.random.default_rng(0)
    vertices = rng.uniform(-1.0, 1.0, (40, 3, 3))
    points = rng.uniform(-2.0, 2.0, (25, 3))
    signed = np.asarray(signed_solid_angle(points[:, None, :], vertices[None, ...]))
    plain = np.asarray(solid_angle(points[:, None, :], vertices[None, ...]))
    np.testing.assert_array_equal(np.abs(signed), plain)


def test_reversing_a_triangle_flips_the_sign_and_not_the_magnitude():
    """What the sign records on a single triangle, and why the public kernel discards it: the
    order the vertices happen to be stored in, which for an imported file is arbitrary."""
    triangle = np.array([[[1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 1.0]]])
    point = np.zeros(3)
    forward = float(signed_solid_angle(point, triangle)[0])
    backward = float(signed_solid_angle(point, triangle[:, ::-1, :])[0])
    assert forward == pytest.approx(-backward, rel=1e-15)
    assert abs(forward) > 0.1, "the fixture must subtend something"


# ---------------------------------------------------------------------------------------
# The winding number
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("divisions", [1, 2, 4])
def test_a_closed_box_winds_once_inside_and_not_at_all_outside(divisions):
    """Exact rather than asymptotic, and at every refinement — including at a point a
    thousandth of a box-width from a wall, where a distance-based test would need a tolerance
    and would get it wrong."""
    vertices = inward_box(divisions)
    np.testing.assert_allclose(np.abs(enclosure_winding(vertices, INSIDE)), 1.0, atol=1e-12)
    np.testing.assert_allclose(enclosure_winding(vertices, OUTSIDE), 0.0, atol=1e-12)


def test_the_sign_follows_the_winding_and_the_test_is_on_the_magnitude():
    """A surface file may be wound either way, so an enclosure test that looked at the sign
    would answer correctly for one convention and backwards for the other."""
    inward = inward_box(2)
    outward = inward[:, ::-1, :]
    one_way = enclosure_winding(inward, INSIDE)
    other = enclosure_winding(outward, INSIDE)
    np.testing.assert_allclose(one_way, -other, atol=1e-12)
    np.testing.assert_allclose(np.abs(one_way), np.abs(other), atol=1e-12)


def test_an_open_surface_reads_between_the_two_answers_rather_than_guessing():
    """The property that makes this the right test for surfaces that arrive from a file.

    Open surfaces are legal here, and a bare disc has no inside. The winding number says so, by
    landing nowhere near either 0 or 1 — a ray-parity test would instead return a clean bit
    that happens to be meaningless.
    """
    disc = disc_triangles(1.0, rings=6, sectors=24)
    just_off = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]])
    winding = enclosure_winding(disc, just_off)
    np.testing.assert_allclose(np.abs(winding), 0.45, atol=0.01)
    assert np.all(np.abs(winding) < 0.5), "an open surface must not read as enclosing"
    # Far away on the axis it falls off towards zero, as a small object should.
    assert abs(float(enclosure_winding(disc, np.array([[0.0, 0.0, 3.0]]))[0])) < 0.03


def test_a_capped_cylinder_encloses_its_axis_and_not_the_space_beside_it():
    """A second closed shape, and one whose facets are not axis-aligned, so nothing here can
    be passing on the box's symmetry."""
    closed, inside, outside = _capped_cylinder()
    np.testing.assert_allclose(np.abs(enclosure_winding(closed, inside)), 1.0, atol=1e-9)
    np.testing.assert_allclose(enclosure_winding(closed, outside), 0.0, atol=1e-9)


def test_a_surface_wound_inconsistently_reads_as_neither_inside_nor_outside():
    """Why :func:`check_winding` comes first, and what it looks like when it has not.

    Flipping one cap of the cylinder leaves a surface that is still geometrically closed and no
    longer *consistently* wound, so the facets' signs no longer agree and their sum is not a
    winding number at all. It reads 0.858 and 0.951 at two interior points — not 1, not 0, and
    **different at each point**, which is the tell: a real winding number is constant over a
    region. The check reports this as an open surface, which is the closest honest description
    it has, and it is why the message says to look rather than telling you the points are clear.
    """
    flipped, inside, _ = _capped_cylinder(consistent=False)
    confused = enclosure_winding(flipped, inside)
    np.testing.assert_allclose(np.abs(confused), [0.858, 0.951], atol=1e-3)
    assert abs(confused[0] - confused[1]) > 0.05, "a real winding number would not vary here"


def test_chunking_the_points_changes_nothing():
    """``work_limit`` bounds the point-by-facet intermediate, which is the whole memory cost.
    A limit below the facet count still has to make progress rather than divide to a chunk of
    zero points and loop forever.
    """
    vertices = inward_box(2)
    points = np.concatenate([INSIDE, OUTSIDE])
    whole = enclosure_winding(vertices, points)
    for work_limit in (1, 97, 10_000_000):
        np.testing.assert_allclose(
            enclosure_winding(vertices, points, work_limit=work_limit), whole, rtol=1e-14
        )


def test_no_points_and_no_facets_are_both_answered_rather_than_raising():
    assert enclosure_winding(inward_box(1), np.zeros((0, 3))).shape == (0,)
    np.testing.assert_array_equal(enclosure_winding(np.zeros((0, 3, 3)), INSIDE), 0.0)


def test_points_must_be_a_list_of_points():
    with pytest.raises(ValueError, match=r"points must be \(n_points, 3\)"):
        enclosure_winding(inward_box(1), np.array([0.5, 0.5, 0.5]))


# ---------------------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------------------


def test_a_point_in_the_metal_is_refused_and_named():
    """The failure this exists to catch is silent: the gather returns a plausible dim number at
    a point that is inside the solid, and nothing about it says the cell is not water."""
    vertices = inward_box(2)
    points = np.concatenate([OUTSIDE, INSIDE])
    with pytest.raises(ValueError, match="lie inside the surface") as raised:
        check_points_outside(vertices, points)
    message = str(raised.value)
    assert "3 of 7" in message
    assert "[4, 5, 6]" in message, "the offenders are named, not just counted"


def test_a_clear_scene_passes_and_hands_back_the_numbers():
    winding = check_points_outside(inward_box(2), OUTSIDE)
    np.testing.assert_allclose(winding, 0.0, atol=1e-12)


def test_an_open_surface_warns_rather_than_raising():
    """Open surfaces are legal, so refusing one would reject a geometry the rest of the package
    accepts — but a clean pass from a surface with no inside establishes less than it reads as,
    and silence is how that goes unnoticed."""
    plate = rectangle_triangles([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    with pytest.warns(UserWarning, match="does not appear to be closed"):
        winding = check_points_outside(plate, np.array([[0.0, 0.0, 0.2]]))
    assert 0.0 < abs(float(winding[0])) < 0.5


def test_a_closed_surface_does_not_warn():
    """The other half: the warning must distinguish an open surface from a closed one, or it is
    noise that gets filtered and then never seen when it matters."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_points_outside(inward_box(2), OUTSIDE)
