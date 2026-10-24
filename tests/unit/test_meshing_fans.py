"""Layers fanned round a sheet's free edge: the surface split at the edge, the cells grown on it.

The fixture is the standing sheet of ``test_meshing_sheets`` -- the unit cube's surface on a
lattice of spacing 1/4, a sheet in the plane ``x = 1/2`` standing on the wall ``y = 0``, ``y`` in
``[0, 1/2]``, ``z`` in ``[1/4, 3/4]`` -- opened along the sheet, every face layered. Its free edge
runs up from the wall at ``z = 1/4``, along ``y = 1/2`` and back down at ``z = 3/4``: two corners,
and two points where it lands on the wall.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.meshing import Polygons, PrismLayers, element_mesh
from aquaflux.meshing.fans import fanned
from aquaflux.meshing.sheets import opened

from tests.unit.test_meshing_sheets import _enclosed, _onto_the_plane, _with_sheet

LAYERS = PrismLayers(first_height=0.02, growth=1.3, count=3)


@pytest.fixture(scope="module")
def sheet():
    points, faces, face = _with_sheet([0.0, 0.25, 0.5], [0.25, 0.5, 0.75])
    return opened(points, faces, face, np.array([1]), _onto_the_plane)


def _free_edge(points):
    """Whether each point lies on the sheet's free edge, the two points it lands at included."""
    y, z = points[:, 1], points[:, 2]
    on_sheet = np.isclose(points[:, 0], 0.5) & (y < 0.5 + 1e-12) & (np.abs(z - 0.5) < 0.25 + 1e-12)
    return on_sheet & (np.isclose(y, 0.5) | np.isclose(np.abs(z - 0.5), 0.25))


def _triangles(faces: Polygons) -> np.ndarray:
    return np.vstack([faces.triangles, faces.quads[:, [0, 1, 2]], faces.quads[:, [0, 2, 3]]])


def test_the_fanned_surface_is_closed_and_encloses_what_the_surface_did(sheet):
    layered = np.ones(sheet.faces.n_faces, dtype=bool)
    fan = fanned(sheet.points, sheet.faces, layered)
    pairs, _ = fan.faces.edges()
    directed = {tuple(pair) for pair in pairs.tolist()}
    assert len(directed) == len(pairs)
    assert all((b, a) in directed for a, b in directed)
    np.testing.assert_allclose(
        _enclosed(fan.points, _triangles(fan.faces)),
        _enclosed(sheet.points, _triangles(sheet.faces)),
        rtol=1e-12,
    )
    # Every point of the free edge is split, and only those; each copy stands where its point is.
    free = np.flatnonzero(_free_edge(sheet.points))
    np.testing.assert_array_equal(np.unique(fan.split), free)
    np.testing.assert_array_equal(fan.points, sheet.points[fan.point])


def test_each_side_of_a_split_point_keeps_its_own_faces(sheet):
    layered = np.ones(sheet.faces.n_faces, dtype=bool)
    fan = fanned(sheet.points, sheet.faces, layered)
    own = fan.faces.where(~fan.added)  # the faces of no area have no normal
    rings = own.padded()
    normals = own.normals(fan.points)
    for copy in np.unique(np.concatenate([fan.split, fan.seconds])):
        holding = np.flatnonzero(np.any(rings == copy, axis=1))
        on_sheet = holding[np.isclose(normals[holding, 0] ** 2, 1.0)]
        assert on_sheet.size
        assert np.unique(np.sign(normals[on_sheet, 0])).size == 1


def test_a_corner_has_a_middle_copy_for_each_edge_and_a_landing_one_for_both(sheet):
    layered = np.ones(sheet.faces.n_faces, dtype=bool)
    fan = fanned(sheet.points, sheet.faces, layered)
    at = sheet.points[fan.split]
    corner = np.isclose(at[:, 1], 0.5) & (np.isclose(at[:, 2], 0.25) | np.isclose(at[:, 2], 0.75))
    # Two corners, each with a middle copy for each of its two edges; one copy elsewhere.
    assert np.count_nonzero(corner) == 4
    assert np.unique(fan.split[corner]).size == 2
    assert np.all(fan.ahead_of[corner] >= 0) and np.all(fan.ahead_of[~corner] < 0)
    for point in np.unique(fan.split[corner]):
        ends = sheet.points[fan.ahead_of[fan.split == point]]
        assert np.unique(np.round(ends, 12), axis=0).shape[0] == 2
    # The two points the edge lands at close the fan along an edge on the wall y = 0.
    lands = fan.landing >= 0
    np.testing.assert_allclose(at[lands, 1], 0.0)
    np.testing.assert_allclose(sheet.points[fan.landing[lands], 1], 0.0)
    assert np.count_nonzero(lands) == 2


def test_layers_fanned_round_the_free_edge_fill_the_room_between_the_walls_and_their_top(sheet):
    layered = np.ones(sheet.faces.n_faces, dtype=bool)
    grown = LAYERS.extrude(sheet.points, sheet.faces, layered)
    # A tetrahedron in each corner per side of its split, a tent where the edge lands on a wall.
    assert grown.tetrahedra.shape == (4, 4)
    assert grown.tents.shape == (2, 6)
    mesh = element_mesh(
        grown.points, grown.tetrahedra, grown.prisms, grown.hexahedra, tents=grown.tents
    )
    volume = np.asarray(mesh.geometry().cell.volume)
    assert np.all(volume > 0.0)
    # The cells fill exactly the box less what the layers' top encloses: none missing, none twice.
    room = 1.0 - _enclosed(grown.points, _triangles(grown.inner))
    np.testing.assert_allclose(volume.sum(), room, rtol=1e-12)
    # Each point of the free edge heads a column for each of its copies: two sides, one middle
    # (two at a corner); those columns are the fanned ones.
    free = _free_edge(grown.points[grown.columns[:, 0]])
    np.testing.assert_array_equal(grown.fanned, free)
    assert np.count_nonzero(free) == 3 * 7 + 2


def test_a_corners_middle_columns_stand_square_off_their_own_edges(sheet):
    layered = np.ones(sheet.faces.n_faces, dtype=bool)
    fan = fanned(sheet.points, sheet.faces, layered)
    grown = LAYERS.extrude(sheet.points, sheet.faces, layered)
    corners = np.flatnonzero(fan.ahead_of >= 0)
    # The merged surface keeps the surface's points, then each layer's: a middle copy's column is
    # the one whose base is the corner and whose first point lies in the sheet's plane.
    for k in corners:
        point, end = fan.split[k], fan.ahead_of[k]
        along = sheet.points[end] - sheet.points[point]
        rows = grown.columns[grown.columns[:, 0] == point]
        up = grown.points[rows[:, 1]] - grown.points[point]
        middle = up[np.isclose(up[:, 0], 0.0, atol=1e-15)]
        # Two middle columns at the corner, one square off each of its edges.
        assert middle.shape[0] == 2
        square = np.abs(middle @ along) < 1e-12 * np.linalg.norm(along)
        assert np.count_nonzero(square) == 1
        # Away from the sheet: off its extent, past the corner.
        tip = grown.points[point] + middle[square][0]
        assert tip[1] > 0.5 or tip[2] < 0.25 or tip[2] > 0.75


def test_a_tents_faces_are_its_two_quadrilaterals_on_its_edge_and_four_triangles():
    # A tent standing on the edge p q, its top the corners r over q and a, m, b over p, with a
    # prism stacked on each of its top triangles. A tent's face listed wrongly would not be met
    # by the prism above it, or would leave the cell open.
    points = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.3],  # p, q, r
        [0.0, -0.4, 0.3], [0.0, 0.0, 0.4], [0.0, 0.4, 0.3],  # a, m, b
        [1.0, 0.0, 1.0], [0.0, -0.4, 1.0], [0.0, 0.0, 1.0], [0.0, 0.4, 1.0],  # above r, a, m, b
    ])  # fmt: skip
    tent = np.array([[0, 1, 2, 3, 4, 5]])
    prisms = np.array([[2, 4, 3, 6, 8, 7], [2, 5, 4, 6, 9, 8]])
    mesh = element_mesh(points, prisms=prisms, tents=tent)
    neighbour = np.asarray(mesh.face_cells.neighbour)
    # 6 + 5 + 5 faces, the two top triangles and the face between the prisms shared.
    assert mesh.face_nodes.n_faces == 13
    assert np.count_nonzero(neighbour >= 0) == 3
    counts = np.diff(np.asarray(mesh.face_nodes.offsets))
    owner = np.asarray(mesh.face_cells.owner)
    of_tent = (owner == 2) | (neighbour == 2)  # the tent is numbered after the prisms
    np.testing.assert_array_equal(np.sort(counts[of_tent]), [3, 3, 3, 3, 4, 4])
    from aquaflux.mesh import closed_cell_residual

    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-14
    assert np.all(np.asarray(mesh.geometry().cell.volume) > 0.0)
