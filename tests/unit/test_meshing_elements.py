"""A mesh of tetrahedra from their corners, checked against what the corners imply.

Five tetrahedra tile a cube (one in the middle, four at alternate corners), so the mesh must have
the cube's volume, sixteen faces -- four inside, twelve on the walls, two per wall -- and walls
whose patches cover exactly the triangles given for them.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.mesh import closed_cell_residual
from aquaflux.meshing import element_mesh

CUBE = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
# Corner index = 4x + 2y + z. The middle tetrahedron joins the four corners with x + y + z even.
FIVE = np.array([[0, 3, 5, 6], [1, 0, 3, 5], [2, 0, 3, 6], [4, 0, 5, 6], [7, 3, 5, 6]])


def _walls(mesh_points, triangles):
    """The cube's boundary triangles by wall, from where their corners lie."""
    walls = {}
    for axis, name in enumerate("xyz"):
        for value, side in ((0.0, "-"), (1.0, "+")):
            on = np.all(np.isclose(mesh_points[triangles][:, :, axis], value), axis=1)
            walls[f"{name}{side}"] = triangles[on]
    return walls


def _boundary_triangles():
    faces = FIVE[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]].reshape(-1, 3)
    keys, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    return keys[counts == 1]


def test_five_tetrahedra_make_a_closed_cube_with_two_triangles_per_wall():
    walls = _walls(CUBE, _boundary_triangles())
    mesh = element_mesh(CUBE, FIVE, patches=walls)
    assert (mesh.n_cells, mesh.n_faces) == (5, 16)
    volume = np.asarray(mesh.geometry().cell.volume)
    np.testing.assert_allclose(volume, [1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6], rtol=1e-12)
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-14
    centroid = np.asarray(mesh.geometry().face.centroid)
    for name in walls:
        faces = np.asarray(mesh.face_patches.indices(name))
        assert faces.size == 2
        axis, value = "xyz".index(name[0]), 1.0 if name[1] == "+" else 0.0
        np.testing.assert_allclose(centroid[faces, axis], value)


def test_faces_are_numbered_as_first_seen_and_owned_by_the_first_to_see_them():
    mesh = element_mesh(CUBE, FIVE)
    owner, neighbour = np.asarray(mesh.face_cells.owner), np.asarray(mesh.face_cells.neighbour)
    # The middle tetrahedron is listed first, so it owns its four faces, which come first, each
    # shared with one corner tetrahedron, in the order of the corners opposite them.
    np.testing.assert_array_equal(owner[:4], 0)
    np.testing.assert_array_equal(neighbour[:4], [4, 3, 2, 1])
    assert np.all(neighbour[4:] == -1)
    rings = np.asarray(mesh.face_nodes.face_node_indices).reshape(-1, 3)
    np.testing.assert_array_equal(rings[:4], [[3, 5, 6], [0, 5, 6], [0, 3, 6], [0, 3, 5]])


def test_a_boundary_face_in_no_patch_is_refused():
    walls = _walls(CUBE, _boundary_triangles())
    walls["x-"] = walls["x-"][:1]
    with pytest.raises(ValueError, match="1 boundary face\\(s\\) in none, 0 in more than one"):
        element_mesh(CUBE, FIVE, patches=walls)


def test_a_boundary_face_in_two_patches_is_refused():
    walls = _walls(CUBE, _boundary_triangles())
    walls["again"] = walls["x-"][::-1, ::-1]  # the same triangles, listed otherwise
    with pytest.raises(ValueError, match="0 boundary face\\(s\\) in none, 2 in more than one"):
        element_mesh(CUBE, FIVE, patches=walls)


def test_a_patch_triangle_inside_the_domain_is_refused():
    walls = _walls(CUBE, _boundary_triangles())
    walls["x-"] = np.vstack([walls["x-"], [[0, 3, 5]]])
    with pytest.raises(ValueError, match="'x-' holds a face that is not a boundary face"):
        element_mesh(CUBE, FIVE, patches=walls)


def test_a_reserved_patch_name_is_refused():
    with pytest.raises(ValueError, match=r"\['boundary'\] are reserved"):
        element_mesh(CUBE, FIVE, patches={"boundary": _boundary_triangles()})


def test_a_triangle_on_three_tetrahedra_is_refused():
    with pytest.raises(ValueError, match="1 face\\(s\\) belong to more than two cells"):
        # The corner tetrahedron at (1, 1, 1) twice: its face with the middle one is on three.
        element_mesh(CUBE, np.vstack([FIVE, [[3, 5, 6, 7]]]))
