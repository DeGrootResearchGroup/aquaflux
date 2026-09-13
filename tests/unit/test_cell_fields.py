"""Unit tests for the cell-field validation both writers share.

The reason this is one helper rather than two copies is that the message has to be the same
whichever writer was being used, so that is what these pin -- alongside the format-specific tests
that go through each writer's own entry point.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.io.cell_fields import as_cell_values


def test_values_are_converted_to_a_float_array():
    values = as_cell_values("p", [1, 2, 3], 3)
    assert isinstance(values, np.ndarray)
    assert values.dtype == np.float64
    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])


def test_a_jax_array_is_accepted():
    # Fields arrive off a solve, so this is the ordinary case rather than a convenience.
    values = as_cell_values("p", jnp.arange(4.0), 4)
    assert isinstance(values, np.ndarray)
    np.testing.assert_array_equal(values, np.arange(4.0))


def test_trailing_axes_are_left_alone():
    # What components mean is the caller's question; this one only checks the per-cell axis.
    values = as_cell_values("tau", np.zeros((3, 3, 3)), 3)
    assert values.shape == (3, 3, 3)


def test_a_field_of_the_wrong_length_names_itself():
    with pytest.raises(ValueError, match=r"field 'U' has 5 values but the mesh has 4 cells"):
        as_cell_values("U", np.zeros((5, 3)), 4)


def test_a_field_with_no_per_cell_axis_is_refused():
    with pytest.raises(ValueError, match="single value, not one value per cell"):
        as_cell_values("p", 1.0, 4)
