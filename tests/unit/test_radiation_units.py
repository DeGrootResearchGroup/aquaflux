"""Converting how ultraviolet equipment is specified into what the model wants."""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.units import absorption_from_uvt, lamp_exitance

from tests.unit.radiation_references import cylinder_triangles, rectangle_triangles

# ---------------------------------------------------------------------------------------
# Ultraviolet transmittance
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("uvt", [99.0, 95.0, 80.0, 65.0, 40.0])
def test_the_coefficient_reproduces_the_transmittance_it_came_from(uvt):
    """The definition, inverted: send light back through the same one-centimetre cell and the
    quoted fraction must survive. An independent statement of the conversion rather than the
    same expression rearranged, because it goes through ``exp`` rather than through ``log``."""
    coefficient = float(absorption_from_uvt(uvt))
    assert float(np.exp(-coefficient * 0.01)) == pytest.approx(uvt / 100.0, rel=1e-14)


def test_the_result_is_napierian_per_metre_not_the_two_neighbouring_conventions():
    """The two ways this is silently wrong, each pinned to its own number.

    A *decadic* coefficient pairs with ``10^(-A r)`` and is smaller by ``ln 10`` — a factor of
    2.3 that reads as clearer water. A coefficient per *centimetre* is smaller by a hundred.
    Neither would be caught by anything downstream: both give a plausible, dimmer or brighter
    field.
    """
    napierian = float(absorption_from_uvt(95.0))
    assert napierian == pytest.approx(5.129329438755057, rel=1e-14)
    decadic = -np.log10(0.95) / 0.01
    assert napierian == pytest.approx(decadic * np.log(10.0), rel=1e-14)
    assert napierian == pytest.approx(100.0 * (-np.log(0.95)), rel=1e-14)


def test_perfectly_clear_water_absorbs_nothing_and_says_so_with_a_positive_zero():
    """``-log(1)`` is ``-0.0``, which is the right number and reads as a wrong one."""
    clear = float(absorption_from_uvt(100.0))
    assert clear == 0.0
    assert not np.signbit(clear), "an absorption coefficient should not print as -0.0"


def test_a_longer_cell_means_a_smaller_coefficient_for_the_same_reading():
    """Transmittance is not a property of the water alone; the path it was measured through is
    half of the number. The same 50% reading through 2 cm is half the absorbance of 1 cm."""
    through_one = float(absorption_from_uvt(50.0))
    through_two = float(absorption_from_uvt(50.0, path_length=0.02))
    assert through_two == pytest.approx(through_one / 2.0, rel=1e-14)


def test_it_converts_a_whole_array_of_readings():
    readings = np.array([99.0, 95.0, 80.0])
    together = np.asarray(absorption_from_uvt(readings))
    apart = [float(absorption_from_uvt(one)) for one in readings]
    np.testing.assert_allclose(together, apart, rtol=0.0)
    assert np.all(np.diff(together) > 0.0), "murkier water absorbs more"


@pytest.mark.parametrize("uvt", [0.0, -5.0, 101.0])
def test_a_transmittance_outside_the_percentage_range_is_refused(uvt):
    with pytest.raises(ValueError, match=r"percentage in \(0, 100\]"):
        absorption_from_uvt(uvt)


@pytest.mark.parametrize("uvt", [0.95, 1.0, 0.65])
def test_a_fraction_passed_as_a_percentage_is_refused(uvt):
    """The one mistake the range check alone cannot catch, because 0.95 is a legal percentage.

    It is also the worst one: read as written, 0.95% UVT is 466 per metre against 95% UVT's
    5.13 — ninety times apart — and the field it produces is dark rather than erroneous. The
    guard costs the ability to express a genuinely sub-1% water, which is outside the range
    ultraviolet reactors are built for.
    """
    assert (-np.log(0.0095) / 0.01) / (-np.log(0.95) / 0.01) == pytest.approx(90.8, abs=0.1)
    with pytest.raises(ValueError, match="percentage, not a fraction"):
        absorption_from_uvt(uvt)


def test_the_fraction_guard_has_no_escape_hatch_through_the_path_length():
    """A longer cell does not make 0.95 a plausible percentage — the confusion is between two
    readings of one number, and has nothing to do with the path it was measured over."""
    with pytest.raises(ValueError, match="percentage, not a fraction"):
        absorption_from_uvt(0.95, path_length=1.0)


def test_one_bad_reading_in_an_array_is_still_refused():
    """The guard is over the whole array, not its first entry — a single fraction hidden among
    percentages is exactly how this arrives."""
    with pytest.raises(ValueError, match="percentage, not a fraction"):
        absorption_from_uvt(np.array([95.0, 0.9, 80.0]))


@pytest.mark.parametrize("path_length", [0.0, -0.01])
def test_a_path_length_that_is_not_a_length_is_refused(path_length):
    with pytest.raises(ValueError, match="positive length in metres"):
        absorption_from_uvt(95.0, path_length=path_length)


# ---------------------------------------------------------------------------------------
# Lamp rating to exitance
# ---------------------------------------------------------------------------------------


def _lamp_and_wall(sectors: int = 16):
    """A faceted cylindrical lamp and a flat wall, as two named bodies of one set."""
    lamp = cylinder_triangles(0.0115, 0.2, sectors=sectors, slices=4)
    wall = rectangle_triangles([0.2, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3])
    vertices = np.concatenate([lamp, wall])
    solid_id = [0] * len(lamp) + [1] * len(wall)
    return Surfaces.from_triangles(vertices, solid_id=solid_id, solid_names=("lamp", "wall"))


@pytest.mark.parametrize("sectors", [8, 16, 64])
def test_the_model_radiates_exactly_the_rating_at_every_refinement(sectors):
    """The property the whole helper exists for, and it is exact rather than close.

    ``sum(M A)`` over the lamp comes back to its watts because the same facet areas appear in
    the division and in the sum. Dividing by the analytic ``pi d L`` instead does not: an
    inscribed triangulation undershoots it by 2.55% at eight sectors, 0.64% at sixteen and
    0.04% at sixty-four, so the model quietly radiates that much less than the lamp — always in
    the same direction, and invisible in any later result.
    """
    surfaces = _lamp_and_wall(sectors)
    rating = 25.0
    exitance = np.asarray(lamp_exitance(surfaces, {"lamp": rating}))
    area = np.asarray(surfaces.area)
    assert float(np.sum(exitance * area)) == pytest.approx(rating, rel=1e-12)

    # What the hand-computed version would radiate instead, on this same triangulation.
    analytic = 2.0 * np.pi * 0.0115 * 0.4
    by_hand = float(np.sum(np.where(exitance > 0.0, rating / analytic, 0.0) * area))
    assert by_hand < rating
    if sectors == 8:
        assert by_hand / rating == pytest.approx(0.9745, abs=5e-4), "the 2.55% shortfall"


def test_a_body_with_no_rating_does_not_emit():
    surfaces = _lamp_and_wall()
    exitance = np.asarray(lamp_exitance(surfaces, {"lamp": 25.0}))
    wall = np.asarray(surfaces.solid_id) == 1
    np.testing.assert_array_equal(exitance[wall], 0.0)
    assert np.all(exitance[~wall] > 0.0)


def test_every_facet_of_one_body_gets_the_same_exitance():
    """Exitance is intensive: the rating is spread over the body's area, not over its facets,
    so a body whose triangles differ in size must still be uniformly bright."""
    surfaces = _lamp_and_wall()
    exitance = np.asarray(lamp_exitance(surfaces, {"lamp": 25.0, "wall": 3.0}))
    for body in (0, 1):
        of_body = exitance[np.asarray(surfaces.solid_id) == body]
        assert len(np.unique(np.round(of_body, 12))) == 1


def test_two_bodies_are_rated_independently():
    surfaces = _lamp_and_wall()
    exitance = np.asarray(lamp_exitance(surfaces, {"lamp": 25.0, "wall": 3.0}))
    area = np.asarray(surfaces.area)
    body = np.asarray(surfaces.solid_id)
    assert float(np.sum(exitance[body == 0] * area[body == 0])) == pytest.approx(25.0, rel=1e-12)
    assert float(np.sum(exitance[body == 1] * area[body == 1])) == pytest.approx(3.0, rel=1e-12)


def test_a_misspelled_body_name_is_refused():
    """Otherwise the lamp is simply dark, which looks like a shadowing result."""
    with pytest.raises(KeyError, match="no such body"):
        lamp_exitance(_lamp_and_wall(), {"lmap": 25.0})


def test_a_point_source_body_has_no_exitance_to_give():
    """A zero-area body would divide by zero and hand back an infinity that propagates. It is a
    category error rather than a numerical one: a point source carries watts, not W/m²."""
    wall = rectangle_triangles([0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0])
    vertices = np.concatenate([wall, np.zeros((1, 3, 3))])
    surfaces = Surfaces.from_triangles(vertices, solid_id=[0, 0, 1], solid_names=("wall", "lamp"))
    with pytest.raises(ValueError, match="has no area"):
        lamp_exitance(surfaces, {"lamp": 25.0})
    # The same set is fine as long as nothing rates the point source.
    assert float(np.sum(np.asarray(lamp_exitance(surfaces, {"wall": 1.0})))) > 0.0


def test_the_areas_it_divides_by_are_the_set_s_own():
    """``area_by_solid`` is the one place the per-body total is formed, so the helper and any
    report a user writes cannot disagree about how big the lamp is."""
    surfaces = _lamp_and_wall()
    totals = surfaces.area_by_solid()
    assert set(totals) == {"lamp", "wall"}
    area, body = np.asarray(surfaces.area), np.asarray(surfaces.solid_id)
    assert totals["lamp"] == pytest.approx(float(np.sum(area[body == 0])), rel=1e-15)
    assert totals["wall"] == pytest.approx(float(np.sum(area[body == 1])), rel=1e-15)
    assert float(np.sum(list(totals.values()))) == pytest.approx(float(np.sum(area)), rel=1e-15)
