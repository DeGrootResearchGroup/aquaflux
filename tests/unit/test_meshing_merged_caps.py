"""Pyramids made from the fill's own tetrahedra on a cut quadrilateral (``MergedCaps``).

Each case is a unit square cut along its 0-2 diagonal with a hand-built fill of tetrahedra above it:
the pyramid must stand on the square with an apex the fill already had (or, opened, one point of
its own), and the cells left must fill exactly what the tetrahedra filled -- every one right side
out, every face shared by two cells or on the boundary.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.mesh import closed_cell_residual
from aquaflux.meshing import MergedCaps, element_mesh
from aquaflux.meshing.merged_caps import _Tetrahedra

SQUARE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
QUAD = np.array([[0, 1, 2, 3]])  # wound so its normal, +z, points into the fill


def _filled(points, tetrahedra, pyramids):
    mesh = element_mesh(points, tetrahedra, pyramids=pyramids)
    np.testing.assert_allclose(np.asarray(closed_cell_residual(mesh)), 0.0, atol=1e-12)
    volume = np.asarray(mesh.geometry().cell.volume)
    assert np.all(volume > 0.0)
    return float(volume.sum())


def _tet_volume(points, tetrahedra):
    corner = points[tetrahedra]
    return float(np.sum(np.einsum(
        "ij,ij->i", np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0]),
        corner[:, 3] - corner[:, 0],
    )) / 6.0)  # fmt: skip


def _oriented(points, tetrahedra):
    """Each tetrahedron turned right side out (the fourth corner on its first three's normal side)."""
    tetrahedra = np.asarray(tetrahedra)
    return np.array(
        [t if _tet_volume(points, t[None]) > 0 else t[[0, 1, 3, 2]] for t in tetrahedra]
    )


def test_two_tetrahedra_on_the_cut_are_one_pyramid_on_their_shared_point():
    points = np.vstack([SQUARE, [[0.4, 0.6, 0.9]]])
    tetrahedra = _oriented(points, [[0, 1, 2, 4], [0, 2, 3, 4]])
    before = _tet_volume(points, tetrahedra)
    merged, left, pyramids, counts = MergedCaps().merged(points, tetrahedra, QUAD)
    assert (counts.direct, counts.swapped, counts.opened, counts.failed) == (1, 0, 0, 0)
    np.testing.assert_array_equal(pyramids, [[0, 1, 2, 3, 4]])
    assert left.shape == (0, 4) and merged.shape == points.shape
    assert _filled(merged, np.zeros((0, 4), int), pyramids) == pytest.approx(before, rel=1e-12)


def _three_round_the_cut():
    """Three tetrahedra round the cut 0-2, from corner 1 by points 4 and 5 to corner 3."""
    points = np.vstack([SQUARE, [[0.7, 0.3, 0.8], [0.3, 0.7, 0.8]]])
    return points, _oriented(points, [[0, 2, 1, 4], [0, 2, 4, 5], [0, 2, 5, 3]])


def test_more_tetrahedra_round_the_cut_are_swapped_down_to_two_first():
    points, tetrahedra = _three_round_the_cut()
    before = _tet_volume(points, tetrahedra)
    merged, left, pyramids, counts = MergedCaps().merged(points, tetrahedra, QUAD)
    assert (counts.direct, counts.swapped, counts.opened, counts.failed) == (0, 1, 0, 0)
    assert pyramids[0, 4] in (4, 5)  # an apex the fill had
    assert left.shape[0] == 2  # the swap's two tetrahedra beside the pyramid
    assert _filled(merged, left, pyramids) == pytest.approx(before, rel=1e-12)


def test_an_opened_pyramid_gets_a_point_of_its_own_and_fills_the_same_space():
    points, tetrahedra = _three_round_the_cut()
    before = _tet_volume(points, tetrahedra)
    mesh = _Tetrahedra(points, tetrahedra, {(0, 2)})
    apex = mesh.opened(0, 2, QUAD[0], lift=0.5)
    assert apex == points.shape[0]
    left = mesh.array()
    pyramids = np.array([[0, 1, 2, 3, apex]])
    assert mesh.points[apex, 2] > 0.0
    assert _filled(mesh.points, left, pyramids) == pytest.approx(before, rel=1e-12)


def test_a_bent_quadrilateral_is_cut_along_its_ridge():
    """Corner 2 raised into the fill lifts the 0-2 diagonal over corners 1 and 3: a ridge, cut
    there; lowered, 0-2 is a valley (a tetrahedron of the fill could hold all four corners), and
    the cut is 1-3."""
    points = SQUARE.copy()
    points[2, 2] = 0.3
    assert MergedCaps.diagonals(points, QUAD)[0]
    points[2, 2] = -0.3
    assert not MergedCaps.diagonals(points, QUAD)[0]


def test_lift_must_be_positive():
    with pytest.raises(ValueError, match=r"MergedCaps\.lift"):
        MergedCaps(lift=0.0)
