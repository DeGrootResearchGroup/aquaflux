"""The surface check that guards Netgen's fill: each defect refused by its own message.

Runs without Netgen. Each case breaks one property of a valid surface (a unit cube's twelve
triangles, wound outward), so the message it is matched against can come from only one branch.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.meshing import check_closed_surface

#: A unit cube's corners, and its twelve triangles wound out of the cube.
CUBE_POINTS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=float,
)
CUBE = np.array(
    [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
     [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]]
)  # fmt: skip


def test_the_cube_passes():
    check_closed_surface(CUBE_POINTS, CUBE)


def test_a_wrong_shape_is_refused():
    with pytest.raises(ValueError, match=r"points must have shape \(n, 3\)"):
        check_closed_surface(CUBE_POINTS[:, :2], CUBE)
    with pytest.raises(ValueError, match=r"triangles must have shape \(m, 3\)"):
        check_closed_surface(CUBE_POINTS, CUBE[:, :2])


def test_a_non_finite_point_is_refused():
    points = CUBE_POINTS.copy()
    points[3, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite coordinate"):
        check_closed_surface(points, CUBE)


def test_an_index_out_of_range_is_refused():
    triangles = CUBE.copy()
    triangles[0, 0] = 8
    with pytest.raises(ValueError, match=r"outside \[0, 8\)"):
        check_closed_surface(CUBE_POINTS, triangles)


def test_an_unused_point_is_refused():
    with pytest.raises(ValueError, match="1 point"):
        check_closed_surface(np.vstack([CUBE_POINTS, [[0.5, 0.5, 0.5]]]), CUBE)


def test_one_triangle_wound_the_other_way_is_refused():
    triangles = CUBE.copy()
    triangles[0] = triangles[0][::-1]
    with pytest.raises(ValueError, match="traversed twice in the same direction"):
        check_closed_surface(CUBE_POINTS, triangles)


def test_an_open_surface_is_refused():
    with pytest.raises(ValueError, match="no triangle on their other side"):
        check_closed_surface(CUBE_POINTS, CUBE[:-2])


def test_a_surface_wound_into_the_region_is_refused():
    with pytest.raises(ValueError, match="encloses a volume of -1"):
        check_closed_surface(CUBE_POINTS, CUBE[:, ::-1])
