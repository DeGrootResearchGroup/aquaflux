"""The placement of a CAD model in a case's frame: rigid maps only, reflections included."""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.io.cad import Placement


def test_the_identity_is_the_default():
    placement = Placement()
    np.testing.assert_array_equal(placement.matrix, np.eye(3))
    np.testing.assert_array_equal(placement.offset, np.zeros(3))


def test_swapping_two_axes_is_allowed_although_it_is_a_reflection():
    swap = Placement(matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]], offset=[1.0, 0.0, 0.0])
    assert np.linalg.det(swap.matrix) == pytest.approx(-1.0)


def test_a_stretch_is_refused_because_it_would_make_a_cylinder_elliptic():
    with pytest.raises(ValueError, match="orthogonal"):
        Placement(matrix=np.diag([1.0, 1.0, 1.001]))


def test_a_uniform_scale_is_refused_too_since_units_are_the_readers_business():
    with pytest.raises(ValueError, match="orthogonal"):
        Placement(matrix=1e-3 * np.eye(3))


def test_the_wrong_shapes_are_refused():
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        Placement(matrix=np.eye(2))
    with pytest.raises(ValueError, match=r"\(3,\)"):
        Placement(offset=[0.0, 0.0])
