"""The caps' apexes moved after the fill: to the middle of their neighbours, never past a worse cell.

Each fixture is a square pyramid of height one over the unit square, cut into a low cap -- the
square and an apex above its centre -- and the four tetrahedra between the cap's sides and the
pyramid's top. Where the apex should go is worked out here by hand.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.meshing import ApexRelaxation

SQUARE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
CAP = np.array([[0, 1, 2, 3, 4]])
FILL = np.array([[i, (i + 1) % 4, 4, 5] for i in range(4)])


def _signed(points, corners):
    c = points[corners]
    e = c[:, 1:] - c[:, :1]
    return np.einsum("ij,ij->i", np.cross(e[:, 0], e[:, 1]), e[:, 2]) / 6.0


def _volumes(points, tetrahedra, pyramids):
    """The tetrahedra's signed volumes, and each pyramid's as two tetrahedra."""
    pyramid = _signed(points, pyramids[:, [0, 1, 2, 4]]) + _signed(
        points, pyramids[:, [0, 2, 3, 4]]
    )
    return _signed(points, tetrahedra), pyramid


def _corners(points, pyramids):
    """The signed volumes of the tetrahedra at each pyramid's four corners -- each corner, the
    corners either side and the apex -- every one of which a valid pyramid keeps positive."""
    rings = [[3, 0, 1, 4], [0, 1, 2, 4], [1, 2, 3, 4], [2, 3, 0, 4]]
    return np.concatenate([_signed(points, pyramids[:, ring]) for ring in rings])


def test_an_apex_goes_to_the_mean_of_the_points_it_shares_a_cell_with():
    # The square's corners and the top: the mean is a fifth of the way up. Once there, it stays.
    points = np.vstack([SQUARE, [[0.5, 0.5, 0.1], [0.5, 0.5, 1.0]]])
    relaxed = ApexRelaxation().relaxed(points, FILL, CAP, np.array([4]))
    np.testing.assert_allclose(relaxed.points[4], [0.5, 0.5, 0.2], atol=1e-15)
    np.testing.assert_array_equal(relaxed.points[[0, 1, 2, 3, 5]], points[[0, 1, 2, 3, 5]])
    assert relaxed.converged and relaxed.sweeps == 2
    tetrahedra, pyramid = _volumes(relaxed.points, FILL, CAP)
    np.testing.assert_allclose(tetrahedra.sum() + pyramid.sum(), 1.0 / 3.0, rtol=1e-14)


def test_apexes_sharing_cells_sweep_together_to_where_each_is_its_neighbours_mean():
    # Two caps side by side, under one roof: each apex's mean holds the other apex, so it takes
    # several sweeps, and where they stop each is the mean of the points it shares a cell with.
    left, right = SQUARE, SQUARE + np.array([1.0, 0.0, 0.0])
    points = np.vstack([left, right[[1, 2]], [[0.5, 0.5, 0.1], [1.5, 0.5, 0.1], [1.0, 0.5, 1.0]]])
    a, b, top = 6, 7, 8
    caps = np.array([[0, 1, 2, 3, a], [1, 4, 5, 2, b]])
    left_sides = [[0, 1, a, top], [2, 3, a, top], [3, 0, a, top]]
    right_sides = [[4, 5, b, top], [5, 2, b, top], [1, 4, b, top]]
    between = [[1, 2, a, b], [1, 2, b, top], [1, 2, top, a]]
    fill = np.array(left_sides + right_sides + between)
    relaxed = ApexRelaxation(max_sweeps=200, tolerance=1e-9).relaxed(
        points, fill, caps, np.array([a, b])
    )
    assert relaxed.converged and relaxed.sweeps > 3
    moved = relaxed.points
    for apex in (a, b):
        sharing = np.unique(np.concatenate([fill[np.any(fill == apex, 1)].ravel(),
                                            caps[np.any(caps == apex, 1)].ravel()]))  # fmt: skip
        sharing = sharing[sharing != apex]
        np.testing.assert_allclose(moved[apex], moved[sharing].mean(axis=0), atol=1e-7)


def test_a_move_that_would_invert_a_cell_is_cut_short():
    # The top leans out over the square's edge x = 1, so the mean of the apex's neighbours lies
    # so low and so far over that the full move would turn a tetrahedron of the fill inside out.
    # The apex goes only part of the way, and every cell and every corner of the cap keeps its
    # orientation.
    points = np.vstack([SQUARE, [[0.5, 0.5, 0.2], [1.5, 0.5, 0.3]]])
    full = np.vstack([SQUARE, points[[5]]]).mean(axis=0)
    inverted = points.copy()
    inverted[4] = full

    def oriented(at):
        return np.concatenate([_corners(at, CAP), _signed(at, FILL)]) * np.sign(start)

    start = np.concatenate([_corners(points, CAP), _signed(points, FILL)])
    assert np.any(oriented(inverted) < 0.0)
    relaxed = ApexRelaxation().relaxed(points, FILL, CAP, np.array([4]))
    assert np.all(oriented(relaxed.points) > 1e-4)
    moved = np.linalg.norm(relaxed.points[4] - points[4])
    assert 0.05 * np.linalg.norm(full - points[4]) < moved < np.linalg.norm(full - points[4])


def _shape(points, corners):
    """A tetrahedron's volume over a regular one's of the same root-mean-square edge."""
    c = points[corners]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    rms = np.sqrt(np.mean([np.sum((c[:, i] - c[:, j]) ** 2, axis=1) for i, j in pairs], axis=0))
    return np.abs(_signed(points, corners)) / (rms**3 / (6.0 * np.sqrt(2.0)))


def test_a_move_that_would_flatten_the_cap_is_cut_short_at_the_floor():
    # A top just above the apex pulls it down towards the square: the mean is at a height of
    # 0.06, where the cap's corner tetrahedra would be flatter than the floor of 0.1, though the
    # fill's tetrahedra above would be fine. The apex comes down only as far as the floor allows.
    points = np.vstack([SQUARE, [[0.5, 0.5, 0.2], [0.5, 0.5, 0.3]]])
    rings = np.array([[3, 0, 1, 4], [0, 1, 2, 4], [1, 2, 3, 4], [2, 3, 0, 4]])
    assert _shape(points, CAP[0][rings]).min() > 0.1
    relaxed = ApexRelaxation(min_quality=0.1).relaxed(points, FILL, CAP, np.array([4]))
    assert 0.06 < relaxed.points[4, 2] < 0.2
    assert _shape(relaxed.points, CAP[0][rings]).min() >= 0.1


@pytest.mark.parametrize(
    "settings", [{"max_sweeps": 0}, {"tolerance": 0.0}, {"min_quality": 1.0}, {"max_halvings": -1}]
)
def test_settings_out_of_range_are_refused(settings):
    with pytest.raises(ValueError, match="ApexRelaxation"):
        ApexRelaxation(**settings)
