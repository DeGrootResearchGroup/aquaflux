"""Axis-aligned hexahedra written with no mesh: the corner order VTK needs, and point data."""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.io import write_hexahedra
from aquaflux.octree.tree import CORNERS

from tests.unit.test_vtk_xml import _decode

VTK_HEXAHEDRON = 12


def _two_cells():
    """A unit cube and a half-size cube beside it, its corners numbered by binary index."""
    first = CORNERS.astype(float)
    second = np.array([1.0, 0.0, 0.0]) + 0.5 * CORNERS
    points = np.concatenate([first, second])
    corners = np.stack([np.arange(8), 8 + np.arange(8)])
    return points, corners


@pytest.mark.parametrize("binary", [True, False])
def test_each_cell_is_a_vtk_hexahedron_in_vtk_s_own_vertex_order(tmp_path, binary):
    points, corners = _two_cells()
    path = write_hexahedra(points, corners, tmp_path / "cells.vtu", binary=binary,
                           point_fields={"G": np.arange(16.0)},
                           cell_fields={"level": np.array([0.0, 1.0])})  # fmt: skip
    root, values = _decode(path.read_bytes())
    np.testing.assert_array_equal(values["types"], [VTK_HEXAHEDRON] * 2)
    np.testing.assert_array_equal(values["offsets"], [8, 16])
    np.testing.assert_allclose(values["Points"], points)
    cells = values["connectivity"].reshape(2, 8)
    for cell in cells:
        at = points[cell]
        bottom, top = at[:4], at[4:]
        # The bottom face is a ring (consecutive vertices joined by an edge of the cube) ...
        edge = np.roll(bottom, -1, axis=0) - bottom
        assert np.allclose(np.sort(np.abs(edge), axis=1)[:, :2], 0.0)
        # ... whose right-hand normal points into the cell, toward the top face ...
        normal = np.sum(np.cross(bottom, np.roll(bottom, -1, axis=0)), axis=0)
        assert np.dot(normal, top.mean(axis=0) - bottom.mean(axis=0)) > 0.0
        # ... and vertex i + 4 is joined to vertex i by an edge.
        rise = top - bottom
        assert np.allclose(rise, rise[0]) and np.count_nonzero(rise[0]) == 1
    point_data = root.find(".//PointData")
    assert point_data is not None and point_data.find("DataArray").attrib["Name"] == "G"
    np.testing.assert_allclose(values["G"], np.arange(16.0))
    np.testing.assert_allclose(values["level"], [0.0, 1.0])


def test_bad_corners_and_lengths_are_refused(tmp_path):
    points, corners = _two_cells()
    with pytest.raises(ValueError, match=r"\(n_cells, 8\)"):
        write_hexahedra(points, corners[:, :7], tmp_path / "a.vtu")
    with pytest.raises(ValueError, match="does not exist"):
        write_hexahedra(points, corners + 1, tmp_path / "b.vtu")
    with pytest.raises(ValueError, match="'G'"):
        write_hexahedra(points, corners, tmp_path / "c.vtu", point_fields={"G": np.zeros(3)})
