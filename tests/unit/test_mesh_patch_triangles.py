"""A mesh's boundary patches as triangles facing into the domain."""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.mesh import Mesh, patch_triangles, structured_grid_2d, structured_grid_3d

LX, LY, LZ = 1.0, 2.0, 3.0

#: The inward unit normal of each side of the box ``structured_grid_3d`` builds.
INWARD = {
    "left": (1, 0, 0),
    "right": (-1, 0, 0),
    "bottom": (0, 1, 0),
    "top": (0, -1, 0),
    "back": (0, 0, 1),
    "front": (0, 0, -1),
}
SIDE_AREA = {
    "left": LY * LZ,
    "right": LY * LZ,
    "bottom": LX * LZ,
    "top": LX * LZ,
    "back": LX * LY,
    "front": LX * LY,
}


def box(nx=3, ny=2, nz=4) -> Mesh:
    return structured_grid_3d(nx, ny, nz, LX, LY, LZ, named_boundaries=True)


def rebuilt_with_reversed_rings(mesh: Mesh, faces) -> Mesh:
    """The same mesh, with the node rings of ``faces`` listed the other way round."""
    offsets = np.asarray(mesh.face_nodes.offsets)
    indices = np.asarray(mesh.face_nodes.face_node_indices).copy()
    for face in faces:
        indices[offsets[face] : offsets[face + 1]] = indices[offsets[face] : offsets[face + 1]][
            ::-1
        ]
    names = [n for n in mesh.face_patches.names if n not in ("interior", "boundary")]
    return Mesh.from_csr(
        mesh.node_coords,
        offsets,
        indices,
        mesh.face_cells.owner,
        mesh.face_cells.neighbour,
        mesh.face_cells.n_cells,
        face_patches={n: np.asarray(mesh.face_patches.indices(n)) for n in names},
    )


def normals(vertices: np.ndarray) -> np.ndarray:
    return np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])


def test_every_triangle_faces_into_the_domain_whatever_the_node_order():
    """The side is decided by the owner-outward normal, so a reversed ring changes nothing.

    Half of every patch's rings are reversed, so a winding taken from the node order would put
    half the triangles on each side and fail here.
    """
    mesh = box()
    boundary = np.flatnonzero(np.asarray(mesh.face_cells.neighbour) < 0)
    mesh = rebuilt_with_reversed_rings(mesh, boundary[::2])
    triangles = patch_triangles(mesh, mesh.geometry(), list(INWARD))
    n = normals(triangles.vertices)
    for k, name in enumerate(triangles.patch_names):
        mine = n[triangles.patch_id == k]
        unit = mine / np.linalg.norm(mine, axis=1)[:, None]
        np.testing.assert_allclose(unit, np.broadcast_to(INWARD[name], unit.shape), atol=1e-12)


def test_the_triangles_have_the_patch_area_and_four_per_quad():
    mesh = box()
    geometry = mesh.geometry()
    triangles = patch_triangles(mesh, geometry, ["top", "front"])
    area = 0.5 * np.linalg.norm(normals(triangles.vertices), axis=1)
    for k, name in enumerate(triangles.patch_names):
        faces = np.asarray(mesh.face_patches.indices(name))
        assert np.sum(triangles.patch_id == k) == 4 * len(faces)
        assert area[triangles.patch_id == k].sum() == pytest.approx(SIDE_AREA[name], rel=1e-14)
        # Each face's triangles sum to that face's own area, so the map to faces is right too.
        per_face = np.bincount(triangles.face, weights=area, minlength=mesh.n_faces)[faces]
        np.testing.assert_allclose(per_face, np.asarray(geometry.face.area)[faces], rtol=1e-14)


def test_patches_come_back_in_the_order_asked_for():
    mesh = box()
    triangles = patch_triangles(mesh, mesh.geometry(), ["front", "left"])
    assert triangles.patch_names == ("front", "left")
    front = np.asarray(mesh.face_patches.indices("front"))
    assert set(triangles.face[triangles.patch_id == 0]) == set(front)


def test_a_face_whose_fan_folds_over_itself_is_refused():
    """A C-shaped face's vertex mean lies in its mouth, outside the face, so the fan overlaps."""
    c_shape = [(0, 0), (3, 0), (3, 0.5), (0.5, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0, 3)]
    ring = [(x, y, 0.0) for x, y in c_shape]
    apex = (1.0, 1.5, -1.0)
    nodes = np.array([*ring, apex])
    base = list(range(len(ring)))
    sides = [[i, (i + 1) % len(ring), len(ring)] for i in range(len(ring))]
    mesh = Mesh.from_faces(
        nodes,
        [base, *sides],
        owner=[0] * (1 + len(sides)),
        neighbour=[-1] * (1 + len(sides)),
        n_cells=1,
        face_patches={"base": [0], "sides": list(range(1, 1 + len(sides)))},
    )
    with pytest.raises(ValueError, match="not star-shaped"):
        patch_triangles(mesh, mesh.geometry(), ["base"])
    # The triangular sides are convex, so the same mesh is otherwise fine.
    assert patch_triangles(mesh, mesh.geometry(), ["sides"]).n_triangles == 3 * len(sides)


def test_a_patch_of_interior_faces_is_refused():
    mesh = box(nx=2, ny=1, nz=1)
    interior = np.flatnonzero(np.asarray(mesh.face_cells.neighbour) >= 0)
    names = [n for n in mesh.face_patches.names if n not in ("interior", "boundary")]
    patches = {n: np.asarray(mesh.face_patches.indices(n)) for n in names}
    baffled = Mesh.from_csr(
        mesh.node_coords,
        mesh.face_nodes.offsets,
        mesh.face_nodes.face_node_indices,
        mesh.face_cells.owner,
        mesh.face_cells.neighbour,
        mesh.face_cells.n_cells,
        face_patches={**patches, "baffle": interior},
    )
    with pytest.raises(ValueError, match="interior faces"):
        patch_triangles(baffled, baffled.geometry(), ["baffle"])


@pytest.mark.parametrize(
    ("names", "message"),
    [([], "no patch names"), (["top", "top"], "more than once"), (["nowhere"], "no group named")],
)
def test_bad_patch_lists_are_refused(names, message):
    mesh = box()
    with pytest.raises(ValueError, match=message):
        patch_triangles(mesh, mesh.geometry(), names)


def test_a_2d_mesh_is_refused():
    mesh = structured_grid_2d(2, 2, named_boundaries=True)
    with pytest.raises(ValueError, match="3D mesh"):
        patch_triangles(mesh, mesh.geometry(), ["top"])


def test_selecting_faces_keeps_their_rings_in_the_order_given():
    mesh = box()
    faces = np.array([7, 2, 40])
    selected = mesh.face_nodes.select(faces)
    assert selected.n_faces == 3
    np.testing.assert_array_equal(
        np.asarray(selected.vertex_mean(mesh.node_coords)),
        np.asarray(mesh.face_nodes.vertex_mean(mesh.node_coords))[faces],
    )
    offsets = np.asarray(mesh.face_nodes.offsets)
    indices = np.asarray(mesh.face_nodes.face_node_indices)
    expected = np.concatenate([indices[offsets[f] : offsets[f + 1]] for f in faces])
    np.testing.assert_array_equal(np.asarray(selected.face_node_indices), expected)
    with pytest.raises(ValueError, match="outside"):
        mesh.face_nodes.select([mesh.n_faces])
