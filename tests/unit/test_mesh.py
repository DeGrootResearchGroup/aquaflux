"""Unit tests for mesh geometry (`aquaflux.mesh`).

Meshes are built by hand (no reader yet) from ragged per-face node lists via
`Mesh.from_faces`, and asserted against analytically-known geometry. 2D: unit squares, a
triangle, a parallelogram. 3D: a unit cube, a tetrahedron, two adjacent cubes
(interior-face scatter), a triangular prism (mixed triangle/quad faces), and a non-unity
box with distinct side lengths (catches scale-factor and axis errors a unit cube would
hide). No solver is involved.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64; geometry needs float64)
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.mesh import (
    CellGeometry,
    FaceGeometry,
    FaceNodeConnectivity,
    Mesh,
    MeshGeometry,
    PolygonFaceGeometry,
    centroid_iteration_shift,
    closed_cell_residual,
    face_geometry_scheme,
    face_planarity,
)

# --------------------------------------------------------------------------- 2D meshes


def _two_unit_squares() -> Mesh:
    """Two adjacent unit squares sharing a vertical face.

    Cell 0 = [0,1]x[0,1], cell 1 = [1,2]x[0,1]. Seven faces: one shared interior face
    (owner 0, neighbour 1) plus three boundary faces per cell.
    """
    nodes = jnp.array(
        [
            [0.0, 0.0],  # 0
            [1.0, 0.0],  # 1
            [2.0, 0.0],  # 2
            [0.0, 1.0],  # 3
            [1.0, 1.0],  # 4
            [2.0, 1.0],  # 5
        ]
    )
    # Intentionally mixed node orderings, to prove orientation is robust to how each
    # edge is listed.
    faces = [
        [1, 4],  # shared, x=1 (interior)
        [0, 1],  # cell0 bottom
        [4, 3],  # cell0 top
        [3, 0],  # cell0 left
        [1, 2],  # cell1 bottom
        [2, 5],  # cell1 right
        [5, 4],  # cell1 top
    ]
    owner = [0, 0, 0, 0, 1, 1, 1]
    neighbour = [1, -1, -1, -1, -1, -1, -1]
    return Mesh.from_faces(nodes, faces, owner, neighbour, n_cells=2)


def _triangle() -> Mesh:
    """A single right triangle with vertices (0,0), (1,0), (0,1). Area 1/2."""
    nodes = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    faces = [[0, 1], [1, 2], [2, 0]]
    return Mesh.from_faces(nodes, faces, [0, 0, 0], [-1, -1, -1], n_cells=1)


def _parallelogram() -> Mesh:
    """A single parallelogram, vertices (0,0),(2,0),(3,1),(1,1). Area 2, centroid (1.5,0.5)."""
    nodes = jnp.array([[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [1.0, 1.0]])
    faces = [[0, 1], [1, 2], [2, 3], [3, 0]]
    return Mesh.from_faces(nodes, faces, [0, 0, 0, 0], [-1, -1, -1, -1], n_cells=1)


# --------------------------------------------------------------------------- 3D meshes


def _unit_cube() -> Mesh:
    """A single unit cube [0,1]^3 with six quad faces. Volume 1, centroid (0.5,0.5,0.5)."""
    nodes = jnp.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
        ]
    )
    faces = [
        [0, 1, 2, 3],  # z=0 bottom
        [4, 5, 6, 7],  # z=1 top
        [0, 1, 5, 4],  # y=0 front
        [3, 2, 6, 7],  # y=1 back
        [0, 3, 7, 4],  # x=0 left
        [1, 2, 6, 5],  # x=1 right
    ]
    owner = [0, 0, 0, 0, 0, 0]
    neighbour = [-1, -1, -1, -1, -1, -1]
    return Mesh.from_faces(nodes, faces, owner, neighbour, n_cells=1)


def _tetrahedron() -> Mesh:
    """A tetrahedron with vertices at the origin and unit axes. Volume 1/6."""
    nodes = jnp.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 1.0],  # 3
        ]
    )
    faces = [
        [0, 1, 2],  # base z=0
        [0, 1, 3],  # y=0
        [0, 2, 3],  # x=0
        [1, 2, 3],  # slanted
    ]
    return Mesh.from_faces(nodes, faces, [0, 0, 0, 0], [-1, -1, -1, -1], n_cells=1)


def _two_unit_cubes() -> Mesh:
    """Two adjacent unit cubes sharing the x=1 face. Interior-face scatter in 3D."""
    nodes = jnp.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [2.0, 0.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [1.0, 1.0, 0.0],  # 4
            [2.0, 1.0, 0.0],  # 5
            [0.0, 0.0, 1.0],  # 6
            [1.0, 0.0, 1.0],  # 7
            [2.0, 0.0, 1.0],  # 8
            [0.0, 1.0, 1.0],  # 9
            [1.0, 1.0, 1.0],  # 10
            [2.0, 1.0, 1.0],  # 11
        ]
    )
    faces = [
        [1, 4, 10, 7],  # shared x=1 (interior): owner 0, neighbour 1
        [0, 3, 9, 6],  # cell0 x=0
        [0, 1, 7, 6],  # cell0 y=0
        [3, 4, 10, 9],  # cell0 y=1
        [0, 1, 4, 3],  # cell0 z=0
        [6, 7, 10, 9],  # cell0 z=1
        [2, 5, 11, 8],  # cell1 x=2
        [1, 2, 8, 7],  # cell1 y=0
        [4, 5, 11, 10],  # cell1 y=1
        [1, 2, 5, 4],  # cell1 z=0
        [7, 8, 11, 10],  # cell1 z=1
    ]
    owner = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    neighbour = [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    return Mesh.from_faces(nodes, faces, owner, neighbour, n_cells=2)


def _triangular_prism() -> Mesh:
    """A right triangular prism (triangle base extruded in z by 1). Volume 1/2.

    Mixes two triangular faces (3 nodes) with three quad faces (4 nodes) with no
    padding — the ragged CSR storage holds each face at its own node count.
    """
    nodes = jnp.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 1.0],  # 3
            [1.0, 0.0, 1.0],  # 4
            [0.0, 1.0, 1.0],  # 5
        ]
    )
    faces = [
        [0, 1, 2],  # z=0 triangle
        [3, 4, 5],  # z=1 triangle
        [0, 1, 4, 3],  # quad, y=0
        [1, 2, 5, 4],  # quad, slanted hypotenuse
        [2, 0, 3, 5],  # quad, x=0
    ]
    owner = [0, 0, 0, 0, 0]
    neighbour = [-1, -1, -1, -1, -1]
    return Mesh.from_faces(nodes, faces, owner, neighbour, n_cells=1)


def _box_2_3_4() -> Mesh:
    """A rectangular box [0,2]x[0,3]x[0,4]. Volume 24, centroid (1, 1.5, 2).

    Deliberately non-unity with three *distinct* side lengths: a wrong overall factor
    (e.g. a missing ``/dim``) or an axis transposition survives a unit cube but not this.
    Distinct face areas {6, 8, 12} also catch per-face area errors.
    """
    a, b, c = 2.0, 3.0, 4.0
    nodes = jnp.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [a, 0.0, 0.0],  # 1
            [a, b, 0.0],  # 2
            [0.0, b, 0.0],  # 3
            [0.0, 0.0, c],  # 4
            [a, 0.0, c],  # 5
            [a, b, c],  # 6
            [0.0, b, c],  # 7
        ]
    )
    faces = [
        [0, 1, 2, 3],  # z=0   area a*b = 6
        [4, 5, 6, 7],  # z=c   area a*b = 6
        [0, 1, 5, 4],  # y=0   area a*c = 8
        [3, 2, 6, 7],  # y=b   area a*c = 8
        [0, 3, 7, 4],  # x=0   area b*c = 12
        [1, 2, 6, 5],  # x=a   area b*c = 12
    ]
    owner = [0, 0, 0, 0, 0, 0]
    neighbour = [-1, -1, -1, -1, -1, -1]
    return Mesh.from_faces(nodes, faces, owner, neighbour, n_cells=1)


def _warped_hex() -> Mesh:
    """A unit cube with node 6 pulled out to (1.3, 1.3, 1.3) → three non-planar faces.

    Exercises the centre-fan on warped faces inside a closed cell: normals must stay unit,
    the volume positive, and (planarity < 1) some faces are genuinely non-planar.
    """
    nodes = jnp.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.3, 1.3, 1.3],  # 6  <- pulled out of plane
            [0.0, 1.0, 1.0],  # 7
        ]
    )
    faces = [
        [0, 1, 2, 3],  # z=0 (planar)
        [4, 5, 6, 7],  # z=1 (warped by node 6)
        [0, 1, 5, 4],  # y=0 (planar)
        [3, 2, 6, 7],  # y=1 (warped)
        [0, 3, 7, 4],  # x=0 (planar)
        [1, 2, 6, 5],  # x=1 (warped)
    ]
    owner = [0, 0, 0, 0, 0, 0]
    neighbour = [-1, -1, -1, -1, -1, -1]
    return Mesh.from_faces(nodes, faces, owner, neighbour, n_cells=1)


ALL_BUILDERS = [
    _two_unit_squares,
    _triangle,
    _parallelogram,
    _unit_cube,
    _tetrahedron,
    _two_unit_cubes,
    _triangular_prism,
    _box_2_3_4,
    _warped_hex,
]

# A single warped quad, node 2 lifted a full edge-length off the plane of the other three.
_WARPED_QUAD = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])


# --------------------------------------------------------------------------- 2D tests


def test_face_areas_unit_edges() -> None:
    mesh = _two_unit_squares()
    area, _, _ = face_geometry_scheme(mesh.dim).unoriented_geometry(
        mesh.node_coords, mesh.face_nodes
    )
    assert jnp.allclose(area, 1.0)


def test_face_centroids_are_edge_midpoints() -> None:
    mesh = _two_unit_squares()
    _, centroid, _ = face_geometry_scheme(mesh.dim).unoriented_geometry(
        mesh.node_coords, mesh.face_nodes
    )
    assert jnp.allclose(centroid[0], jnp.array([1.0, 0.5]))  # shared face
    assert jnp.allclose(centroid[1], jnp.array([0.5, 0.0]))  # cell0 bottom


def test_shared_normal_points_from_owner_to_neighbour_2d() -> None:
    mesh = _two_unit_squares()
    face_geom = mesh.geometry().face
    # Shared face (index 0) separates cell 0 (left) from cell 1 (right): +x, any ordering.
    assert jnp.allclose(face_geom.normal[0], jnp.array([1.0, 0.0]))


def test_unit_square_volumes_and_centroids() -> None:
    mesh = _two_unit_squares()
    cell_geom = mesh.geometry().cell
    assert jnp.allclose(cell_geom.volume, jnp.array([1.0, 1.0]))
    assert jnp.allclose(cell_geom.centroid, jnp.array([[0.5, 0.5], [1.5, 0.5]]))


def test_triangle_area_and_centroid() -> None:
    mesh = _triangle()
    cell_geom = mesh.geometry().cell
    assert jnp.allclose(cell_geom.volume, jnp.array([0.5]))
    assert jnp.allclose(cell_geom.centroid, jnp.array([[1.0 / 3.0, 1.0 / 3.0]]))


def test_parallelogram_area_and_centroid() -> None:
    mesh = _parallelogram()
    cell_geom = mesh.geometry().cell
    assert jnp.allclose(cell_geom.volume, jnp.array([2.0]))
    assert jnp.allclose(cell_geom.centroid, jnp.array([[1.5, 0.5]]))


# --------------------------------------------------------------------------- 3D tests


def test_cube_face_areas_are_unit() -> None:
    mesh = _unit_cube()
    area, _, _ = face_geometry_scheme(mesh.dim).unoriented_geometry(
        mesh.node_coords, mesh.face_nodes
    )
    assert jnp.allclose(area, 1.0)


def test_cube_volume_and_centroid() -> None:
    mesh = _unit_cube()
    cell_geom = mesh.geometry().cell
    assert jnp.allclose(cell_geom.volume, jnp.array([1.0]))
    assert jnp.allclose(cell_geom.centroid, jnp.array([[0.5, 0.5, 0.5]]))


def test_cube_normals_are_axis_aligned_and_outward() -> None:
    mesh = _unit_cube()
    geom = mesh.geometry()
    face_geom, cell_geom = geom.face, geom.cell
    assert jnp.allclose(jnp.linalg.norm(face_geom.normal, axis=1), 1.0)
    outward = face_geom.centroid - cell_geom.centroid[mesh.face_cells.owner]
    assert bool(jnp.all(jnp.sum(outward * face_geom.normal, axis=1) > 0.0))


def test_tetrahedron_volume_and_centroid() -> None:
    mesh = _tetrahedron()
    cell_geom = mesh.geometry().cell
    assert jnp.allclose(cell_geom.volume, jnp.array([1.0 / 6.0]))
    assert jnp.allclose(cell_geom.centroid, jnp.array([[0.25, 0.25, 0.25]]))


def test_box_nonunity_volume_centroid_and_face_areas() -> None:
    """Distinct side lengths + non-unity volume: catches scale-factor and axis errors."""
    mesh = _box_2_3_4()
    geom = mesh.geometry()
    face_geom, cell_geom = geom.face, geom.cell
    assert jnp.allclose(cell_geom.volume, jnp.array([24.0]))
    assert jnp.allclose(cell_geom.centroid, jnp.array([[1.0, 1.5, 2.0]]))
    assert jnp.allclose(jnp.sort(face_geom.area), jnp.array([6.0, 6.0, 8.0, 8.0, 12.0, 12.0]))


def test_two_cubes_interior_face_and_volumes() -> None:
    mesh = _two_unit_cubes()
    geom = mesh.geometry()
    face_geom, cell_geom = geom.face, geom.cell
    assert jnp.allclose(face_geom.area[0], 1.0)
    assert jnp.allclose(face_geom.normal[0], jnp.array([1.0, 0.0, 0.0]))
    assert jnp.allclose(cell_geom.volume, jnp.array([1.0, 1.0]))
    assert jnp.allclose(cell_geom.centroid, jnp.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]]))


def test_prism_mixed_faces_volume_and_centroid() -> None:
    mesh = _triangular_prism()
    geom = mesh.geometry()
    face_geom, cell_geom = geom.face, geom.cell
    # The two triangular faces (stored with 3 nodes) still have area 1/2.
    assert jnp.allclose(face_geom.area[0], 0.5)
    assert jnp.allclose(face_geom.area[1], 0.5)
    assert jnp.allclose(cell_geom.volume, jnp.array([0.5]))
    assert jnp.allclose(cell_geom.centroid, jnp.array([[1.0 / 3.0, 1.0 / 3.0, 0.5]]))


# ------------------------------------------------------- non-planar faces (centre fan)


def _quad_face_geometry(coords, order):
    """Raw geometry of a single polygon face given its node coords and node ordering."""
    offsets = jnp.array([0, len(order)])
    indices = jnp.array(order)
    face_nodes = FaceNodeConnectivity.from_csr(offsets, indices)
    return PolygonFaceGeometry().unoriented_geometry(coords, face_nodes)


def test_warped_quad_has_unit_normal_and_vector_area() -> None:
    """A non-planar quad still yields a unit normal and area = |S| (the vector area)."""
    area, _, normal = _quad_face_geometry(_WARPED_QUAD, [0, 1, 2, 3])
    assert jnp.allclose(jnp.linalg.norm(normal, axis=1), 1.0)
    # Vector area S = area * normal is the polygon boundary integral: (-0.5, -0.5, 1.0).
    S = area[:, None] * normal
    assert jnp.allclose(S[0], jnp.array([-0.5, -0.5, 1.0]))
    assert jnp.allclose(area[0], jnp.linalg.norm(jnp.array([-0.5, -0.5, 1.0])))


def test_warped_face_is_start_vertex_independent() -> None:
    """Cyclic rotations of a warped face's node list give identical area/normal/centroid."""
    base_area, base_centroid, base_normal = _quad_face_geometry(_WARPED_QUAD, [0, 1, 2, 3])
    for rotation in ([1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]):
        area, centroid, normal = _quad_face_geometry(_WARPED_QUAD, rotation)
        assert jnp.allclose(area, base_area)
        assert jnp.allclose(normal, base_normal)
        assert jnp.allclose(centroid, base_centroid)


def test_planarity_metric() -> None:
    """Planarity is 1 for planar faces and < 1 for the warped hex's non-planar faces."""
    assert jnp.allclose(face_planarity(_unit_cube()), 1.0)
    assert jnp.allclose(face_planarity(_box_2_3_4()), 1.0)
    warped = face_planarity(_warped_hex())
    assert bool(jnp.all(warped <= 1.0 + 1e-12))
    assert bool(jnp.any(warped < 0.999))  # some faces are genuinely non-planar


def test_centroid_iteration_shift_is_negligible() -> None:
    """One centre-fan pass is effectively exact: the iteration shift is tiny even warped."""
    assert bool(jnp.all(centroid_iteration_shift(_unit_cube()) < 1e-12))
    # The warped hex has strong warp, yet a second pass barely moves the centroid.
    assert bool(jnp.all(centroid_iteration_shift(_warped_hex()) < 1e-3))


# ------------------------------------------------------------- validation & validity


@pytest.mark.parametrize("builder", ALL_BUILDERS)
def test_validate_accepts_good_meshes(builder) -> None:
    """Every well-formed builder passes validation (from_faces already runs it)."""
    mesh = builder()
    assert mesh.validate() is mesh


def test_from_faces_rejects_out_of_range_owner() -> None:
    nodes = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    faces = [[0, 1], [1, 2], [2, 3], [3, 0]]
    with pytest.raises(ValueError, match="owner index out of range"):
        Mesh.from_faces(nodes, faces, [0, 0, 0, 9], [-1, -1, -1, -1], n_cells=1)


def test_from_faces_rejects_out_of_range_node_index() -> None:
    nodes = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="face_node_indices out of range"):
        Mesh.from_faces(nodes, [[0, 1], [1, 99], [2, 0]], [0, 0, 0], [-1, -1, -1], n_cells=1)


def test_from_faces_rejects_degenerate_3d_face() -> None:
    """A 3D face with only two nodes would give a zero vector area / NaN normal."""
    nodes = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    faces = [[0, 1], [0, 1, 3], [0, 2, 3], [1, 2, 3]]  # first face has 2 nodes
    with pytest.raises(ValueError, match="needs >= 3"):
        Mesh.from_faces(nodes, faces, [0, 0, 0, 0], [-1, -1, -1, -1], n_cells=1)


def test_from_faces_rejects_self_neighbour() -> None:
    nodes = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    faces = [[0, 1], [1, 2], [2, 3], [3, 0]]
    with pytest.raises(ValueError, match="owner and neighbour"):
        Mesh.from_faces(nodes, faces, [0, 0, 0, 0], [0, -1, -1, -1], n_cells=1)


def test_from_csr_rejects_bad_offsets() -> None:
    """Malformed CSR row pointers are rejected at connectivity construction (FaceNodeConnectivity)."""
    with pytest.raises(ValueError, match="face_node_offsets"):
        Mesh.from_csr(
            jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            [0, 2, 3],  # last (3) != len(face_node_indices) (4)
            [0, 1, 1, 2],
            [0, 0],
            [-1, -1],
            n_cells=1,
        )


@pytest.mark.parametrize("builder", ALL_BUILDERS)
def test_closed_cell_residual_is_zero_for_valid_meshes(builder) -> None:
    mesh = builder()
    assert bool(jnp.all(closed_cell_residual(mesh) < 1e-10))


def test_closed_cell_residual_flags_a_missing_face() -> None:
    """Dropping a face from the cube leaves the cell open — the residual jumps to ~face area."""
    nodes = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    faces = [  # the z=1 top face is omitted -> cell is not closed
        [0, 1, 2, 3],
        [0, 1, 5, 4],
        [3, 2, 6, 7],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
    ]
    open_cube = Mesh.from_faces(
        faces=faces, node_coords=nodes, owner=[0] * 5, neighbour=[-1] * 5, n_cells=1
    )
    assert float(closed_cell_residual(open_cube)[0]) > 0.5  # ~ the missing unit face's area


# ---------------------------------------------------------- cell zones & face patches


def _square_pair_args():
    """(nodes, faces, owner, neighbour) for the two-unit-square mesh (face 0 = interior)."""
    nodes = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    faces = [[1, 4], [0, 1], [4, 3], [3, 0], [1, 2], [2, 5], [5, 4]]
    owner = [0, 0, 0, 0, 1, 1, 1]
    neighbour = [1, -1, -1, -1, -1, -1, -1]
    return nodes, faces, owner, neighbour


def test_default_zones_and_patches() -> None:
    """A plain mesh gets one 'default' zone and interior/boundary patches."""
    mesh = _two_unit_squares()  # 1 interior face + 6 boundary faces, 2 cells
    assert mesh.cell_zones.names == ("default",)
    assert mesh.cell_zones.size("default") == 2
    assert set(mesh.face_patches.names) == {"interior", "boundary"}
    assert mesh.face_patches.size("interior") == 1
    assert mesh.face_patches.size("boundary") == 6


def test_named_boundary_patches() -> None:
    """Named patches move boundary faces off the default 'boundary' patch."""
    nodes, faces, owner, neighbour = _square_pair_args()
    mesh = Mesh.from_faces(
        nodes, faces, owner, neighbour, n_cells=2, face_patches={"inlet": [3], "outlet": [5]}
    )
    assert mesh.face_patches.size("inlet") == 1
    assert mesh.face_patches.size("outlet") == 1
    assert mesh.face_patches.size("boundary") == 4  # remaining boundary faces
    assert mesh.face_patches.is_boundary_patch("inlet", mesh.face_cells)
    assert bool(mesh.face_patches.mask("inlet")[3])


def test_baffle_patch_on_interior_face() -> None:
    """An interior face can be named as a patch (a baffle)."""
    nodes, faces, owner, neighbour = _square_pair_args()
    mesh = Mesh.from_faces(nodes, faces, owner, neighbour, n_cells=2, face_patches={"baffle": [0]})
    assert mesh.face_patches.size("baffle") == 1
    assert not mesh.face_patches.is_boundary_patch("baffle", mesh.face_cells)
    assert mesh.face_patches.size("interior") == 0  # the only interior face is now the baffle


def test_cell_zones_and_derived_interface() -> None:
    """Two-zone mesh: the shared face is a derived zone interface."""
    nodes, faces, owner, neighbour = _square_pair_args()
    mesh = Mesh.from_faces(
        nodes, faces, owner, neighbour, n_cells=2, cell_zones={"fluid": [0], "solid": [1]}
    )
    assert set(mesh.cell_zones.names) == {"default", "fluid", "solid"}
    interface = mesh.cell_zones.interface_mask(mesh.face_cells)
    assert bool(interface[0])  # the shared face separates fluid and solid
    assert int(jnp.sum(interface)) == 1  # only that face
    between = mesh.cell_zones.interface_mask_between(mesh.face_cells, "fluid", "solid")
    assert bool(jnp.all(between == interface))


def test_partition_violation_raises() -> None:
    """Two named patches claiming the same face is a partition error."""
    with pytest.raises(ValueError, match="overlaps"):
        Mesh.from_faces(
            jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            [[0, 1], [1, 2], [2, 3], [3, 0]],
            [0, 0, 0, 0],
            [-1, -1, -1, -1],
            n_cells=1,
            face_patches={"a": [0, 1], "b": [1, 2]},  # face 1 in both
        )


# --------------------------------------------------------------------------- shared


@pytest.mark.parametrize("builder", ALL_BUILDERS)
def test_positive_volumes_and_unit_normals(builder) -> None:
    """Every mesh: positive volumes and unit owner-outward normals, order-independent."""
    mesh = builder()
    geom = mesh.geometry()
    face_geom, cell_geom = geom.face, geom.cell
    assert bool(jnp.all(cell_geom.volume > 0.0))
    assert jnp.allclose(jnp.linalg.norm(face_geom.normal, axis=1), 1.0)


@pytest.mark.parametrize("builder", ALL_BUILDERS)
def test_boundary_normals_point_out_of_domain(builder) -> None:
    """Boundary-face normals point away from their owner cell centroid."""
    mesh = builder()
    geom = mesh.geometry()
    face_geom, cell_geom = geom.face, geom.cell
    outward = face_geom.centroid - cell_geom.centroid[mesh.face_cells.owner]
    projection = jnp.sum(outward * face_geom.normal, axis=1)
    boundary = mesh.face_cells.neighbour < 0
    assert bool(jnp.all(projection[boundary] > 0.0))


def test_cell_geometry_matches_direct_call() -> None:
    """`Mesh.geometry()` composes the face/cell strategies consistently."""
    mesh = _two_unit_cubes()
    geom = mesh.geometry()
    face_geom, cell_geom = geom.face, geom.cell
    direct = CellGeometry.from_faces(face_geom, mesh.face_cells, mesh.dim)
    assert jnp.allclose(direct.volume, cell_geom.volume)
    assert jnp.allclose(direct.centroid, cell_geom.centroid)


def test_geometry_returns_bundled_mesh_geometry() -> None:
    """`Mesh.geometry()` returns a `MeshGeometry` bundle of the face and cell metrics."""
    geom = _two_unit_cubes().geometry()
    assert isinstance(geom, MeshGeometry)
    assert isinstance(geom.face, FaceGeometry)
    assert isinstance(geom.cell, CellGeometry)


# -------------------------------------------------- non-convex face centroid (signed weights)


def test_concave_face_centroid_is_area_correct() -> None:
    """A non-convex planar face: the signed-area centroid matches the true polygon centroid.

    Unsigned ``|d_i|`` weighting would misplace it — a reflex vertex makes one fan triangle wind
    against the face normal. This is a planar "dart" (node 2 reflex), true centroid ``(1, 1, 0)``.
    """
    coords = jnp.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 4.0, 0.0]])
    area, centroid, _ = _quad_face_geometry(coords, [0, 1, 2, 3])
    assert jnp.allclose(area[0], 4.0)  # shoelace area of the dart
    assert jnp.allclose(centroid[0], jnp.array([1.0, 1.0, 0.0]))


# ------------------------------------------------------------------ differentiable geometry


def _sum_volume(node_coords: jnp.ndarray, mesh: Mesh) -> jnp.ndarray:
    """Total cell volume of ``mesh`` rebuilt with the given node coordinates (a scalar of it)."""
    rebuilt = eqx.tree_at(lambda m: m.node_coords, mesh, node_coords)
    return rebuilt.geometry().cell.volume.sum()


@pytest.mark.parametrize("builder", [_two_unit_squares, _box_2_3_4, _two_unit_cubes, _warped_hex])
def test_geometry_is_differentiable_wrt_node_coords(builder) -> None:
    """``grad`` of a scalar of the geometry w.r.t. node coordinates is finite — no norm/area NaN.

    The project is a differentiable solver, and the centre-fan/divergence geometry is full of
    ``norm`` and ``/area`` operations that a plain implementation would NaN under ``grad``.
    """
    mesh = builder()
    grad = jax.grad(_sum_volume)(mesh.node_coords, mesh)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_volume_gradient_matches_finite_difference() -> None:
    """d(total volume)/d(a node coordinate) matches a central finite difference (AD is correct)."""
    mesh = _box_2_3_4()
    nc = mesh.node_coords
    grad = jax.grad(_sum_volume)(nc, mesh)
    i, ax, h = 6, 0, 1e-6  # bump a corner node in x
    fd = (_sum_volume(nc.at[i, ax].add(h), mesh) - _sum_volume(nc.at[i, ax].add(-h), mesh)) / (
        2 * h
    )
    assert jnp.allclose(grad[i, ax], fd, atol=1e-5)


# --------------------------------------------------- validate(): silent-NaN robustness guards


def _unit_square_2d():
    """(nodes, faces, owner, neighbour) for a single 2D unit square (four boundary edges)."""
    nodes = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return nodes, [[0, 1], [1, 2], [2, 3], [3, 0]], [0, 0, 0, 0], [-1, -1, -1, -1]


def test_rejects_non_finite_node_coords() -> None:
    nodes = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, jnp.inf]])
    with pytest.raises(ValueError, match="non-finite"):
        Mesh.from_faces(nodes, [[0, 1], [1, 2], [2, 0]], [0, 0, 0], [-1, -1, -1], n_cells=1)


def test_rejects_non_integer_indices() -> None:
    nodes, faces, _, neighbour = _unit_square_2d()
    with pytest.raises(ValueError, match="integer dtype"):
        Mesh.from_faces(nodes, faces, [0.0, 0.0, 0.0, 0.0], neighbour, n_cells=1)


def test_rejects_2d_face_that_is_not_an_edge() -> None:
    """A 2D face must be a 2-node edge; a 3-node face would break the edge scheme's reshape."""
    nodes = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="2D face must be an edge"):
        Mesh.from_faces(nodes, [[0, 1, 2], [2, 3], [3, 0]], [0, 0, 0], [-1, -1, -1], n_cells=1)


def test_rejects_repeated_node_in_face() -> None:
    """A face that lists the same node twice is degenerate (zero area → NaN normal)."""
    nodes = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    faces = [[0, 1, 1], [0, 1, 3], [0, 2, 3], [1, 2, 3]]  # face 0 repeats node 1
    with pytest.raises(ValueError, match="same node more than once"):
        Mesh.from_faces(nodes, faces, [0, 0, 0, 0], [-1, -1, -1, -1], n_cells=1)


def test_rejects_unreferenced_cell() -> None:
    """A cell touched by no face would get 0/0 = NaN geometry; validation catches it."""
    nodes, faces, owner, neighbour = _unit_square_2d()
    with pytest.raises(ValueError, match="not referenced"):
        Mesh.from_faces(nodes, faces, owner, neighbour, n_cells=2)  # cell 1 unreferenced


def test_rejects_neighbour_below_minus_one() -> None:
    """-1 is the only accepted boundary sentinel (docs and validation now agree)."""
    nodes, faces, owner, _ = _unit_square_2d()
    with pytest.raises(ValueError, match="neighbour index out of range"):
        Mesh.from_faces(nodes, faces, owner, [-2, -1, -1, -1], n_cells=1)


def test_quality_flags_collinear_zero_area_face_without_nan() -> None:
    """A geometric-only degeneracy (distinct but collinear nodes → zero area) is out of scope for
    topology-only ``validate()``, so it passes construction — but the quality diagnostics flag it
    as maximally bad (planarity 0, iteration shift inf) instead of a silent, clean-reading NaN."""
    nodes = jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = [[0, 1, 2], [0, 1, 4], [0, 3, 4], [1, 3, 4], [2, 3, 4]]  # face 0: collinear (zero area)
    mesh = Mesh.from_faces(nodes, faces, [0, 0, 0, 0, 0], [-1, -1, -1, -1, -1], n_cells=1)
    planarity = face_planarity(mesh)
    assert bool(jnp.all(jnp.isfinite(planarity)))  # no NaN
    assert jnp.allclose(planarity[0], 0.0)  # the degenerate face reads as maximally non-planar
    assert not bool(jnp.isnan(centroid_iteration_shift(mesh)[0]))  # inf, not a silent NaN
    assert bool(jnp.all(jnp.isfinite(mesh.geometry().face.normal[1:])))  # other normals stay finite


# ---------------------------------------------------------------- groups.py footgun guards


def test_empty_patch_is_not_a_boundary_patch() -> None:
    """An empty patch reports ``False`` (not a vacuously-true 'all faces are boundary faces')."""
    nodes, faces, owner, neighbour = _square_pair_args()
    mesh = Mesh.from_faces(nodes, faces, owner, neighbour, n_cells=2, face_patches={"empty": []})
    assert mesh.face_patches.size("empty") == 0
    assert not mesh.face_patches.is_boundary_patch("empty", mesh.face_cells)


def test_reserved_patch_name_rejected() -> None:
    """The auto-assigned names 'interior'/'boundary' may not be reused for a named patch."""
    nodes, faces, owner, neighbour = _square_pair_args()
    with pytest.raises(ValueError, match="reserved"):
        Mesh.from_faces(nodes, faces, owner, neighbour, n_cells=2, face_patches={"boundary": [3]})


def test_interface_mask_between_same_zone_raises() -> None:
    """An interface is between two *distinct* zones; equal names are a caller mistake."""
    nodes, faces, owner, neighbour = _square_pair_args()
    mesh = Mesh.from_faces(
        nodes, faces, owner, neighbour, n_cells=2, cell_zones={"fluid": [0], "solid": [1]}
    )
    with pytest.raises(ValueError, match="distinct zones"):
        mesh.cell_zones.interface_mask_between(mesh.face_cells, "fluid", "fluid")


def test_interface_mask_between_is_orientation_symmetric() -> None:
    """``interface_mask_between(a, b)`` equals ``(b, a)`` — both windings of the interface."""
    nodes, faces, owner, neighbour = _square_pair_args()
    mesh = Mesh.from_faces(
        nodes, faces, owner, neighbour, n_cells=2, cell_zones={"fluid": [0], "solid": [1]}
    )
    forward = mesh.cell_zones.interface_mask_between(mesh.face_cells, "fluid", "solid")
    reverse = mesh.cell_zones.interface_mask_between(mesh.face_cells, "solid", "fluid")
    assert bool(jnp.all(forward == reverse))
    assert int(jnp.sum(forward)) == 1
