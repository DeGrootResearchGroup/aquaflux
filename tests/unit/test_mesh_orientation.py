"""Faces turned out of their owner by how each cell's faces join, not by where its centroid is.

The cells here are chosen so a centroid cannot decide: a thin chevron (a "V") has its mean point
outside the faces of its inner corner, so turning each face away from that point turns those faces
the wrong way, and the cell no longer closes. Orienting the faces consistently over the edges they
share, then outward by the cell's volume, closes it whatever its node order. A cell whose faces do
not join at shared edges (a face carrying a node on each of its edges, which its neighbours along
them do not) falls back to the centroid.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.mesh import Mesh, closed_cell_residual
from aquaflux.mesh.orientation import ring_signs

# A chevron in the x-y plane: an arm along x and an arm along y meeting at the origin, each 1 long
# and 0.1 thick. Its corners anticlockwise; the inner corner is (0.1, 0.1).
CHEVRON = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.1], [0.1, 0.1], [0.1, 1.0], [0.0, 1.0]])
CHEVRON_AREA = 1.0 * 0.1 + 0.9 * 0.1


def _chevron_2d(reverse: bool) -> Mesh:
    edges = [[i, (i + 1) % 6] for i in range(6)]
    if reverse:
        edges = [edge[::-1] for edge in edges]
    return Mesh.from_faces(CHEVRON, edges, np.zeros(6, int), -np.ones(6, int), 1)


def _chevron_3d(reverse: bool) -> Mesh:
    """The chevron extruded 0.1 in z: two hexagonal caps and six rectangular sides."""
    points = np.vstack([np.c_[CHEVRON, np.zeros(6)], np.c_[CHEVRON, np.full(6, 0.1)]])
    faces = [list(range(6))[::-1], list(range(6, 12))]
    faces += [[i, (i + 1) % 6, (i + 1) % 6 + 6, i + 6] for i in range(6)]
    if reverse:
        faces = [face[::-1] for face in faces]
    return Mesh.from_faces(points, faces, np.zeros(8, int), -np.ones(8, int), 1)


def _naive_flips(mesh: Mesh) -> int:
    """How many faces turning away from the mean of the face centroids would point inward."""
    geometry = mesh.geometry()
    centroid = np.asarray(geometry.face.centroid)
    outward = np.einsum(
        "ij,ij->i", centroid - centroid.mean(axis=0), np.asarray(geometry.face.normal)
    )
    return int(np.count_nonzero(outward < 0))


@pytest.mark.parametrize("build", [_chevron_2d, _chevron_3d])
@pytest.mark.parametrize("reverse", [False, True])
def test_a_chevron_cell_closes_with_every_face_turned_out(build, reverse):
    mesh = build(reverse)
    # The case that needs it: the mean of the face centroids lies beyond the inner corner's faces.
    assert _naive_flips(mesh) > 0
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-15
    volume = float(np.asarray(mesh.geometry().cell.volume)[0])
    assert volume == pytest.approx(CHEVRON_AREA * (1.0 if mesh.dim == 2 else 0.1), rel=1e-12)
    assert bool(np.asarray(ring_signs(mesh.face_nodes, mesh.face_cells).determined)[0])


def test_a_cube_whose_top_face_joins_no_side_falls_back_to_its_centroid():
    # The top carries a node at the middle of each of its edges, which the sides do not: no edge
    # of the top is an edge of a side, so the top's direction is fixed by nothing but the centroid.
    corners = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float)
    middles = np.array([[0.5, 0, 1], [1, 0.5, 1], [0.5, 1, 1], [0, 0.5, 1]], float)
    points = np.vstack([corners, middles])
    faces = [
        [0, 2, 3, 1],  # bottom
        [4, 8, 5, 9, 7, 10, 6, 11],  # top, with the middles of its edges
        [0, 1, 5, 4], [1, 3, 7, 5], [3, 2, 6, 7], [2, 0, 4, 6],
    ]  # fmt: skip
    mesh = Mesh.from_faces(points, faces, np.zeros(6, int), -np.ones(6, int), 1)
    assert not bool(np.asarray(ring_signs(mesh.face_nodes, mesh.face_cells).determined)[0])
    assert float(np.asarray(mesh.geometry().cell.volume)[0]) == pytest.approx(1.0, rel=1e-12)
    normal = np.asarray(mesh.geometry().face.normal)
    np.testing.assert_allclose(normal[0], [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(normal[1], [0, 0, 1], atol=1e-12)


def test_two_cells_sharing_a_face_turn_it_out_of_its_owner():
    # Two chevrons stacked in z, the lower one owning the face between them.
    lower = _chevron_3d(reverse=False)
    points = np.vstack([np.asarray(lower.node_coords), np.c_[CHEVRON, np.full(6, 0.2)]])
    faces = [list(range(6))[::-1], list(range(6, 12)), list(range(12, 18))]
    faces += [[i, (i + 1) % 6, (i + 1) % 6 + 6, i + 6] for i in range(6)]
    faces += [[i + 6, (i + 1) % 6 + 6, (i + 1) % 6 + 12, i + 12][::-1] for i in range(6)]
    owner = [0, 0, 1] + [0] * 6 + [1] * 6
    neighbour = [-1, 1, -1] + [-1] * 12
    mesh = Mesh.from_faces(points, faces, np.array(owner), np.array(neighbour), 2)
    np.testing.assert_allclose(np.asarray(mesh.geometry().face.normal)[1], [0, 0, 1], atol=1e-12)
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-15
