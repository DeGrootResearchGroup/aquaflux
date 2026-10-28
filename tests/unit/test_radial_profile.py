"""Values given against distance from an axis: ``RadialProfile``, ``AxialVector``, ``radial_average``,
and the table they are kept in (``write_profile_table`` / ``read_profile_table``)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import AxialVector, RadialProfile, radial_average
from aquaflux.io import read_profile_table, write_profile_table

CENTRE = np.array([1.0, -2.0, 0.5])
AXIS = np.array([1.0, 2.0, 2.0]) / 3.0  # a unit axis leaning off every coordinate axis


def _points_at(radii: np.ndarray, along: np.ndarray) -> np.ndarray:
    """Points at the given distances from the axis, at the given positions along it, each turned a
    different way round it."""
    across = np.cross(AXIS, [0.0, 0.0, 1.0])
    across /= np.linalg.norm(across)
    other = np.cross(AXIS, across)
    turns = np.linspace(0.0, 5.0, radii.size)
    return (
        CENTRE
        + along[:, None] * AXIS
        + radii[:, None] * (np.cos(turns)[:, None] * across + np.sin(turns)[:, None] * other)
    )


def test_a_profile_is_read_by_each_points_distance_from_the_axis_whatever_its_place_along_it():
    """Interpolated linearly between the radii given, held at the ends beyond them."""
    profile = RadialProfile(CENTRE, 3.0 * AXIS, [0.1, 0.3, 0.6], [4.0, 3.0, 1.0])
    radii = np.array([0.0, 0.1, 0.2, 0.45, 0.6, 2.0])
    points = _points_at(radii, along=np.linspace(-3.0, 7.0, radii.size))
    np.testing.assert_allclose(profile.radius(points), radii, atol=1e-14)
    np.testing.assert_allclose(profile(points), [4.0, 4.0, 3.5, 2.0, 1.0, 1.0], atol=1e-13)


def test_an_axial_vector_points_along_the_unit_axis_as_long_as_the_profile_says():
    profile = RadialProfile(CENTRE, -2.0 * AXIS, [0.0, 1.0], [2.0, 0.0])
    points = _points_at(np.array([0.25, 0.5]), along=np.array([0.0, 1.0]))
    np.testing.assert_allclose(
        AxialVector(profile)(points), np.outer([1.5, 1.0], -AXIS), atol=1e-13
    )


def test_a_profiles_values_are_leaves_a_gradient_reaches():
    """The values are differentiable, so a sensitivity to the inflow's profile can be taken."""
    points = jnp.asarray(_points_at(np.array([0.25]), along=np.array([0.0])))

    def value(values):
        return RadialProfile(CENTRE, AXIS, jnp.array([0.0, 1.0]), values)(points)[0]

    np.testing.assert_allclose(jax.grad(value)(jnp.array([2.0, 6.0])), [0.75, 0.25])


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ((CENTRE, AXIS, [0.0, 0.5, 0.5], [1.0, 2.0, 3.0]), "radii must increase"),
        ((CENTRE, AXIS, [0.0, 1.0], [1.0]), "one value at each radius"),
        ((CENTRE, [0.0, 0.0, 0.0], [0.0, 1.0], [1.0, 2.0]), "axis must not be zero"),
        ((CENTRE, AXIS[:2], [0.0, 1.0], [1.0, 2.0]), "two vectors of one dimension"),
    ],
)
def test_a_profile_that_is_not_one_is_refused(arguments, match):
    with pytest.raises(ValueError, match=match):
        RadialProfile(*arguments)


def test_a_radial_average_weights_each_point_and_reports_the_bands_own_mean_radius():
    """Two bands of the four hold points; each row is its points' weighted means, radius included."""
    radii = np.array([0.1, 0.2, 0.9, 0.95, 1.0])
    weights = np.array([1.0, 3.0, 2.0, 1.0, 1.0])
    value = np.array([10.0, 20.0, 1.0, 2.0, 3.0])
    points = _points_at(radii, along=np.array([0.0, 4.0, -1.0, 2.0, 3.0]))
    bands, means = radial_average(points, weights, {"value": value}, CENTRE, 5.0 * AXIS, bands=4)
    np.testing.assert_allclose(bands, [(0.1 + 3 * 0.2) / 4, (2 * 0.9 + 0.95 + 1.0) / 4])
    np.testing.assert_allclose(means["value"], [(10 + 60) / 4, (2 + 2 + 3) / 4])


def test_a_radial_average_refuses_no_band_and_a_weight_that_is_not_positive():
    points = _points_at(np.array([0.1, 0.2]), along=np.zeros(2))
    with pytest.raises(ValueError, match="at least one band"):
        radial_average(points, np.ones(2), {}, CENTRE, AXIS, bands=0)
    with pytest.raises(ValueError, match="weights must be positive"):
        radial_average(points, np.array([1.0, 0.0]), {}, CENTRE, AXIS, bands=2)


def test_a_profile_table_reads_back_exactly_with_its_columns_in_order(tmp_path):
    radii = np.array([0.0012345678901234567, 0.25, 1.0 / 3.0])
    columns = {"U": np.array([8.1, 7.0 / 3.0, -1e-300]), "k": np.array([0.1, 0.2, np.pi])}
    path = write_profile_table(tmp_path / "p.csv", radii, columns, comment="a pipe\nat Re 6e4")
    assert path.read_text().splitlines()[:3] == ["# a pipe", "# at Re 6e4", "radius,U,k"]
    table = read_profile_table(path)
    np.testing.assert_array_equal(table.radii, radii)
    assert list(table.columns) == ["U", "k"]
    for name, values in columns.items():
        np.testing.assert_array_equal(table.columns[name], values)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("r,U\n0.0,1.0\n", "first column is 'radius'"),
        ("radius,U\n0.0,1.0\n0.1\n", "row 3 holds 1 values under 2 columns"),
        ("radius,U\n0.1,1.0\n0.0,2.0\n", "radii must increase"),
        ("# nothing\n", "no profile table"),
    ],
)
def test_a_file_that_is_not_a_profile_table_is_refused(tmp_path, text, match):
    path = tmp_path / "p.csv"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        read_profile_table(path)


@pytest.mark.parametrize(
    ("radii", "columns", "match"),
    [
        ([0.0, 0.0], {"U": [1.0, 2.0]}, "must be one increasing row"),
        ([0.0, 1.0], {"U": [1.0]}, "holds \\(1,\\) values for \\(2,\\) radii"),
        ([0.0, 1.0], {"radius": [1.0, 2.0]}, "cannot name a column 'radius'"),
        ([0.0, 1.0], {"a,b": [1.0, 2.0]}, "cannot name a column 'a,b'"),
    ],
)
def test_a_table_that_cannot_be_read_back_is_not_written(tmp_path, radii, columns, match):
    with pytest.raises(ValueError, match=match):
        write_profile_table(tmp_path / "p.csv", np.array(radii), columns)
    assert not (tmp_path / "p.csv").exists()
