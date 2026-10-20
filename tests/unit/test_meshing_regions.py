"""A region as the octree core asks about it, from a closed triangle surface.

The surface is a unit box with a cubic hole in it, so which points lie inside and which boxes its
boundary passes through are both known in closed form here, not from the region's own grid.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.meshing import ClosedSurface

from tests.support.meshes import box_surface

HOLE = (0.3, 0.7)
SPACING = 0.05


def _box_with_a_hole():
    outer_points, outer = box_surface(0.0, 1.0, 6, into_box=False)
    inner_points, inner = box_surface(*HOLE, 2, into_box=True, offset=len(outer_points))
    return np.vstack([outer_points, inner_points]), np.vstack([outer, inner])


def _within(points, lo, hi):
    return np.all((points > lo) & (points < hi), axis=1)


def test_a_point_is_inside_where_the_box_is_and_its_hole_is_not():
    region = ClosedSurface.of(*_box_with_a_hole(), SPACING)
    points = np.random.default_rng(0).uniform(-0.1, 1.1, size=(4000, 3))
    expected = _within(points, 0.0, 1.0) & ~_within(points, *HOLE)
    np.testing.assert_array_equal(region.inside(points), expected)
    # Both ways of answering are asked: a voxel's stretch, and a ray where the surface reaches.
    cell = np.floor((points - region.grid.low) / region.grid.spacing).astype(int)
    in_grid = np.all((cell >= 0) & (cell < region.grid.resolution), axis=1)
    stretch = region.label[tuple(cell[in_grid].T)]
    assert np.count_nonzero(stretch == 0) > 100 and np.count_nonzero(stretch > 0) > 1000
    # The hole is a stretch of its own, outside, though the surface surrounds it.
    assert np.count_nonzero(~region.within[1:]) >= 1 and np.any(region.within[1:])


def _meets_exactly(lower, upper):
    """Whether the boundary of the box less its hole passes through each closed box: neither
    wholly inside the region's interior nor wholly outside its closure."""
    interior = _within(lower, 0.0, 1.0) & _within(upper, 0.0, 1.0)
    interior &= np.any((upper < HOLE[0]) | (lower > HOLE[1]), axis=1)
    exterior = np.any((upper < 0.0) | (lower > 1.0), axis=1)
    exterior |= _within(lower, *HOLE) & _within(upper, *HOLE)
    return ~interior & ~exterior


def test_a_box_the_surface_passes_through_is_met_and_one_two_voxels_clear_is_not():
    region = ClosedSurface.of(*_box_with_a_hole(), SPACING)
    rng = np.random.default_rng(1)
    lower = rng.uniform(-0.2, 1.1, size=(6000, 3))
    upper = lower + rng.uniform(0.0, 0.3, size=(6000, 1))
    met = region.meets(lower, upper)
    passes = _meets_exactly(lower, upper)
    assert np.count_nonzero(passes) > 1000 and np.count_nonzero(~passes) > 1000
    assert np.all(met[passes])
    clear = ~_meets_exactly(lower - 2 * SPACING, upper + 2 * SPACING)
    assert np.count_nonzero(clear) > 300
    assert not np.any(met[clear])


def test_a_surface_with_a_face_missing_is_refused():
    points, triangles = box_surface(0.0, 1.0, 2, into_box=False)
    top = np.all(np.isclose(points[triangles][:, :, 2], 1.0), axis=1)
    with pytest.raises(ValueError, match="have no triangle on their other side"):
        ClosedSurface.of(points, triangles[~top], SPACING)
