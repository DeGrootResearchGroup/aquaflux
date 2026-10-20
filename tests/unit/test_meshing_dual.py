"""The feature-preserving polyhedral dual, checked against what a dual must be.

Every check is a property with an independent answer: the dual of a block of hexahedra is hexahedra
of known size centred on the primal points; the dual of any mesh of a domain with flat walls fills
exactly that domain, its cells closed, its boundary faces on the walls and in the walls' patches;
a corner kept as a feature keeps the domain's volume, and one not kept loses a predictable sliver;
a cell at a sharp concave edge is cut into convex pieces, meeting on the plane bisecting the edge.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from aquaflux.io.openfoam.assembler import assemble
from aquaflux.mesh import Mesh, closed_cell_residual, structured_grid_2d, structured_grid_3d
from aquaflux.meshing import PolyhedralDual
from aquaflux.meshing.dual import _check_connected

from tests.support.meshes import (
    hexahedron_beside_a_prism,
    restated,
    tetrahedral_grid_3d,
    without_cells,
    worst_excursion,
)
from tests.support.polymesh import cyclic_two_cube_polymesh_data

SHARP = PolyhedralDual(feature_angle=math.radians(45))
LX, LY, LZ = 3.0, 1.0, 2.0
NX, NY, NZ = 3, 2, 4


def _box():
    return structured_grid_3d(NX, NY, NZ, LX, LY, LZ, named_boundaries=True)


def _cube_walls(mesh: Mesh) -> Mesh:
    """``mesh`` of an axis-aligned box with its boundary faces in six patches, one per wall."""
    geometry = mesh.geometry()
    boundary = np.flatnonzero(np.asarray(mesh.face_cells.neighbour) < 0)
    normal = np.asarray(geometry.face.normal)[boundary]
    axis = np.argmax(np.abs(normal), axis=1)
    side = normal[np.arange(boundary.size), axis] > 0
    names = {(a, s): f"{'xyz'[a]}{'+' if s else '-'}" for a in range(3) for s in (False, True)}
    patches = {name: boundary[(axis == a) & (side == s)] for (a, s), name in names.items()}
    return Mesh.from_csr(
        mesh.node_coords,
        mesh.face_nodes.offsets,
        mesh.face_nodes.face_node_indices,
        mesh.face_cells.owner,
        mesh.face_cells.neighbour,
        mesh.n_cells,
        face_patches=patches,
        patch_types={"x-": "patch", "x+": "patch"},
        patch_groups={"walls": ["y-", "y+", "z-", "z+"]},
    )


def _single_patch(mesh: Mesh) -> Mesh:
    """``mesh`` with every boundary face in one patch, so no patch border is a feature."""
    boundary = np.flatnonzero(np.asarray(mesh.face_cells.neighbour) < 0)
    return Mesh.from_csr(
        mesh.node_coords,
        mesh.face_nodes.offsets,
        mesh.face_nodes.face_node_indices,
        mesh.face_cells.owner,
        mesh.face_cells.neighbour,
        mesh.n_cells,
        face_patches={"walls": boundary},
    )


def _reversed(mesh: Mesh) -> Mesh:
    """``mesh`` with every face ring stored backwards."""
    return restated(mesh, reverse=np.ones(mesh.n_faces, dtype=bool))


def _patch_areas(mesh: Mesh) -> dict[str, float]:
    area = np.asarray(mesh.geometry().face.area)
    return {
        name: float(area[np.asarray(mesh.face_patches.indices(name))].sum())
        for name in mesh.face_patches.names
        if name not in ("interior", "boundary")
    }


def _assert_closed(dual: Mesh) -> None:
    scale = float(np.max(np.asarray(dual.geometry().face.area)))
    assert float(np.max(np.asarray(closed_cell_residual(dual)))) < 1e-12 * scale


def test_the_dual_of_hexahedra_is_hexahedra_of_half_spacing_at_the_walls():
    primal = _box()
    dual = SHARP.build(primal)
    assert dual.n_cells == primal.n_nodes
    h = np.array([LX / NX, LY / NY, LZ / NZ])
    points = np.asarray(primal.node_coords)
    # Each dual cell spans half a spacing either side of its point, clipped at the walls.
    on_wall = np.isclose(points, 0.0) | np.isclose(points, [LX, LY, LZ])
    expected = np.prod(np.where(on_wall, h / 2, h), axis=1)
    geometry = dual.geometry()
    np.testing.assert_allclose(geometry.cell.volume, expected, rtol=1e-12)
    interior = ~on_wall.any(axis=1)
    np.testing.assert_allclose(
        np.asarray(geometry.cell.centroid)[interior], points[interior], atol=1e-12
    )
    # An interior dual cell of a hexahedral grid is a hexahedron: six faces of four nodes.
    owner, neighbour = np.asarray(dual.face_cells.owner), np.asarray(dual.face_cells.neighbour)
    counts = np.asarray(dual.face_nodes.counts)
    for cell in np.flatnonzero(interior):
        mine = (owner == cell) | (neighbour == cell)
        assert np.count_nonzero(mine) == 6 and np.all(counts[mine] == 4)


@pytest.mark.parametrize(
    "build",
    [
        lambda: _cube_walls(tetrahedral_grid_3d(3, perturb=0.25, seed=2)),
        lambda: _cube_walls(_reversed(tetrahedral_grid_3d(2))),
        lambda: _box(),
    ],
)
def test_the_dual_fills_the_domain_with_closed_cells_and_keeps_each_walls_area(build):
    primal = build()
    dual = SHARP.build(primal)
    volume = np.asarray(dual.geometry().cell.volume)
    assert np.all(volume > 0.0)
    np.testing.assert_allclose(
        volume.sum(), np.asarray(primal.geometry().cell.volume).sum(), rtol=1e-12
    )
    _assert_closed(dual)
    primal_areas, dual_areas = _patch_areas(primal), _patch_areas(dual)
    assert dual_areas.keys() == primal_areas.keys()
    for name, area in primal_areas.items():
        assert dual_areas[name] == pytest.approx(area, rel=1e-12)


def test_every_dual_boundary_face_lies_on_the_wall_of_its_patch():
    primal = _cube_walls(tetrahedral_grid_3d(3, perturb=0.25, seed=3))
    dual = SHARP.build(primal)
    coords = np.asarray(dual.node_coords)
    offsets = np.asarray(dual.face_nodes.offsets)
    nodes = np.asarray(dual.face_nodes.face_node_indices)
    for name in ("x-", "x+", "y-", "y+", "z-", "z+"):
        axis, value = "xyz".index(name[0]), 1.0 if name[1] == "+" else 0.0
        for face in np.asarray(dual.face_patches.indices(name)):
            np.testing.assert_allclose(
                coords[nodes[offsets[face] : offsets[face + 1]], axis], value, atol=1e-12
            )


def test_a_patch_border_on_a_flat_wall_is_kept():
    # The box's left wall split into two patches along y = 0.5: nothing folds there, so only the
    # rule that a patch border is a feature keeps each dual boundary face in one patch.
    box = _box()
    patches = box.face_patches
    left = np.asarray(patches.indices("left"))
    low = np.asarray(box.geometry().face.centroid)[left, 1] < 0.5 * LY
    named = {
        name: np.asarray(patches.indices(name)) for name in patches.names[2:] if name != "left"
    }
    named |= {"left_low": left[low], "left_high": left[~low]}
    split = Mesh.from_csr(
        box.node_coords, box.face_nodes.offsets, box.face_nodes.face_node_indices,
        box.face_cells.owner, box.face_cells.neighbour, box.n_cells, face_patches=named,
    )  # fmt: skip
    dual = SHARP.build(split)
    areas = _patch_areas(dual)
    assert areas["left_low"] == pytest.approx(0.5 * LY * LZ, rel=1e-12)
    assert areas["left_high"] == pytest.approx(0.5 * LY * LZ, rel=1e-12)
    _assert_closed(dual)


def test_patch_types_and_groups_carry_over():
    dual = SHARP.build(_cube_walls(tetrahedral_grid_3d(2)))
    assert dict(dual.face_patches.patch_types) == {"x-": "patch", "x+": "patch"}
    assert dict(dual.face_patches.patch_groups) == {"walls": ("y-", "y+", "z-", "z+")}


def test_a_corner_not_kept_is_cut_off_and_naming_its_edges_keeps_it():
    # One patch over the whole cube, so only the fold can make the cube's edges features. Below
    # their 90 degree fold they are; above it they are not, and each dual boundary cell then cuts
    # the corner through the centroids of the faces around it.
    primal = _single_patch(tetrahedral_grid_3d(2))
    kept = PolyhedralDual(feature_angle=math.radians(80)).build(primal)
    rounded = PolyhedralDual(feature_angle=math.radians(100)).build(primal)
    assert float(np.asarray(kept.geometry().cell.volume).sum()) == pytest.approx(1.0, rel=1e-12)
    lost = 1.0 - float(np.asarray(rounded.geometry().cell.volume).sum())
    assert lost > 1e-3
    _assert_closed(rounded)
    # Naming the cube's edges explicitly keeps them whatever the angle.
    points = np.asarray(primal.node_coords)
    edge_nodes, _ = primal.face_nodes.edges()
    on_two_walls = (
        (np.isclose(points, 0.0) | np.isclose(points, 1.0)).sum(axis=1)[edge_nodes] >= 2
    ).all(axis=1)
    along = (
        np.count_nonzero(np.isclose(points[edge_nodes[:, 0]], points[edge_nodes[:, 1]]), axis=1)
        == 2
    )
    named = edge_nodes[on_two_walls & along]
    explicit = PolyhedralDual(feature_angle=math.radians(100), feature_edges=named[:, ::-1]).build(
        primal
    )
    assert float(np.asarray(explicit.geometry().cell.volume).sum()) == pytest.approx(1.0, rel=1e-12)


def _without_corner(mesh: Mesh, corner, axes) -> Mesh:
    """``mesh`` with the cells beyond ``corner`` along every one of ``axes`` taken out."""
    centroid = np.asarray(mesh.geometry().cell.centroid)
    beyond = np.all(centroid[:, axes] > np.asarray(corner), axis=1)
    return without_cells(mesh, np.flatnonzero(beyond), patch="cut")


def _assert_conformal(dual: Mesh) -> None:
    """Each cell's faces use every edge between two of its vertices exactly twice -- so no face
    passes a vertex that the face beside it along that edge does not."""
    owner, neighbour = np.asarray(dual.face_cells.owner), np.asarray(dual.face_cells.neighbour)
    offsets = np.asarray(dual.face_nodes.offsets)
    nodes = np.asarray(dual.face_nodes.face_node_indices)
    uses = [dict() for _ in range(dual.n_cells)]
    for face in range(dual.n_faces):
        ring = nodes[offsets[face] : offsets[face + 1]]
        for a, b in zip(ring, np.roll(ring, -1), strict=True):
            for cell in (owner[face], neighbour[face]):
                if cell >= 0:
                    key = (min(a, b), max(a, b))
                    uses[cell][key] = uses[cell].get(key, 0) + 1
    assert all(count == 2 for cell in uses for count in cell.values())


# An L of hexahedra, longer in z than x so that no cell centroid lies on the plane bisecting it:
# the concave edge runs along y at x = 1/3, z = 3/4.
L_EDGE = (1 / 3, 0.75)


def _hexahedral_l() -> Mesh:
    return _without_corner(
        structured_grid_3d(3, 2, 2, 1.0, 1.0, 1.5, named_boundaries=True), L_EDGE, [0, 2]
    )


def test_a_cell_on_a_sharp_concave_edge_is_cut_into_two_convex_pieces():
    # The dual cell of a point on the L's inside edge would wrap round three quarters of a turn.
    # Cut along the plane bisecting the edge, it becomes two cells, each a convex half of it, and
    # the face between them lies in that plane, (x - 1/3) = (z - 3/4).
    primal = _hexahedral_l()
    dual = SHARP.build(primal)
    points = np.asarray(primal.node_coords)
    on_edge = np.flatnonzero(
        np.isclose(points[:, 0], L_EDGE[0]) & np.isclose(points[:, 2], L_EDGE[1])
    )
    assert on_edge.size == 3 and dual.n_cells == primal.n_nodes + on_edge.size
    assert float(worst_excursion(dual).max()) < 1e-12
    np.testing.assert_allclose(np.asarray(dual.geometry().cell.volume).sum(), 1.0, rtol=1e-12)
    _assert_closed(dual)
    _assert_conformal(dual)
    # A point's further piece follows the primal points, in the order of the points.
    owner, neighbour = np.asarray(dual.face_cells.owner), np.asarray(dual.face_cells.neighbour)
    coords = np.asarray(dual.node_coords)
    offsets = np.asarray(dual.face_nodes.offsets)
    nodes = np.asarray(dual.face_nodes.face_node_indices)
    for point, piece in zip(on_edge, range(primal.n_nodes, dual.n_cells), strict=True):
        between = np.flatnonzero(
            ((owner == point) & (neighbour == piece)) | ((owner == piece) & (neighbour == point))
        )
        assert between.size > 0
        for face in between:
            ring = coords[nodes[offsets[face] : offsets[face + 1]]]
            np.testing.assert_allclose(ring[:, 0] - L_EDGE[0], ring[:, 2] - L_EDGE[1], atol=1e-12)


def test_the_cut_dual_of_tetrahedra_fills_an_l_and_keeps_its_walls():
    # Kuhn tetrahedra of a block 1 x 1 x 1.5 with a corner taken out. Pieces or not, the dual must
    # close, fill the L and keep each wall's area.
    block = tetrahedral_grid_3d(3)
    stretched = Mesh.from_csr(
        np.asarray(block.node_coords) * [1.0, 1.0, 1.5], block.face_nodes.offsets,
        block.face_nodes.face_node_indices, block.face_cells.owner, block.face_cells.neighbour,
        block.n_cells,
    )  # fmt: skip
    primal = _without_corner(_cube_walls(stretched), (1 / 3, 0.5), [0, 2])
    dual = SHARP.build(primal)
    assert dual.n_cells > primal.n_nodes
    np.testing.assert_allclose(
        np.asarray(dual.geometry().cell.volume).sum(),
        np.asarray(primal.geometry().cell.volume).sum(),
        rtol=1e-12,
    )
    _assert_closed(dual)
    _assert_conformal(dual)
    for name, area in _patch_areas(primal).items():
        assert _patch_areas(dual)[name] == pytest.approx(area, rel=1e-12)


def test_a_concave_corner_of_three_walls_is_cut_into_three_pieces():
    # A block with one corner taken out: three concave edges meet at the inside corner, whose dual
    # cell is cut into three, one behind each wall; each other point on those edges into two.
    # Spacings 0.5, 0.4 and 0.6, so that no cell centroid is equally deep behind two walls.
    corner = np.array([0.5, 0.8, 1.2])
    primal = _without_corner(structured_grid_3d(2, 3, 4, 1.0, 1.2, 2.4), corner, [0, 1, 2])
    dual = SHARP.build(primal)
    points = np.asarray(primal.node_coords)
    inside = np.isclose(points, corner)
    on_an_edge = (np.count_nonzero(inside, axis=1) == 2) & np.all(points >= corner - 1e-12, axis=1)
    assert np.count_nonzero(np.all(inside, axis=1)) == 1
    assert dual.n_cells == primal.n_nodes + 2 + np.count_nonzero(on_an_edge)
    # Nearly convex: a face cut around two points of the corner is crossed by two of the planes
    # between pieces, and its one vertex on that dual edge lies between them. An uncut cell there
    # would wrap round the corner (an excursion near 1).
    assert float(worst_excursion(dual).max()) < 0.05
    removed = np.prod([1.0, 1.2, 2.4] - corner)
    np.testing.assert_allclose(
        np.asarray(dual.geometry().cell.volume).sum(), 1.0 * 1.2 * 2.4 - removed, rtol=1e-12
    )
    _assert_closed(dual)


def test_a_cell_centroid_on_the_cutting_plane_gives_no_coincident_vertices():
    # Cubic cells: the cell in the L's inside corner has its centroid on the bisecting plane, so the
    # vertex where its dual edges cross between pieces would sit on that centroid.
    primal = _without_corner(structured_grid_3d(2, 1, 2, named_boundaries=True), (0.5, 0.5), [0, 2])
    dual = SHARP.build(primal)
    assert dual.n_cells > primal.n_nodes
    coords = np.asarray(dual.node_coords)
    offsets = np.asarray(dual.face_nodes.offsets)
    nodes = np.asarray(dual.face_nodes.face_node_indices)
    shortest = min(
        float(np.min(np.linalg.norm(ring - np.roll(ring, -1, axis=0), axis=1)))
        for ring in (coords[nodes[offsets[f] : offsets[f + 1]]] for f in range(dual.n_faces))
    )
    assert shortest > 1e-3
    _assert_closed(dual)


def test_a_gently_concave_fold_is_not_cut():
    # The same L, its inside edge folding by 90 degrees, is below a feature angle of 100.
    primal = _hexahedral_l()
    assert PolyhedralDual(feature_angle=math.radians(100)).build(primal).n_cells == primal.n_nodes


def test_a_piece_in_two_parts_is_refused():
    # Four cells around a point in a ring 0-1-2-3-0, the first wall's cells 0 and 2 opposite.
    cells = np.arange(4)
    first, second = np.array([0, 1, 2, 3]), np.array([1, 2, 3, 0])
    group = np.array([0, 1, 0, 1])
    with pytest.raises(NotImplementedError, match="would make a dual cell in two parts"):
        _check_connected(cells, 2, first, second, group[first] == group[second])
    group = np.array([0, 0, 1, 1])
    _check_connected(cells, 2, first, second, group[first] == group[second])


def test_mixed_cells_have_a_closed_dual():
    primal = _single_patch(hexahedron_beside_a_prism())
    dual = SHARP.build(primal)
    np.testing.assert_allclose(np.asarray(dual.geometry().cell.volume).sum(), 1.5, rtol=1e-12)
    _assert_closed(dual)


def test_a_two_dimensional_mesh_is_refused():
    with pytest.raises(ValueError, match="three-dimensional mesh; this one has dim 2"):
        SHARP.build(structured_grid_2d(2, 2))


def test_a_periodic_mesh_is_refused():
    with pytest.raises(ValueError, match="periodic"):
        SHARP.build(assemble(cyclic_two_cube_polymesh_data()))


def test_several_cell_zones_are_refused():
    box = structured_grid_3d(2, 1, 1)
    zoned = Mesh.from_csr(
        box.node_coords, box.face_nodes.offsets, box.face_nodes.face_node_indices,
        box.face_cells.owner, box.face_cells.neighbour, box.n_cells, cell_zones={"left": [0]},
    )  # fmt: skip
    with pytest.raises(ValueError, match="several cell zones"):
        SHARP.build(zoned)


def test_a_patch_on_interior_faces_is_refused():
    box = structured_grid_3d(2, 1, 1)
    interior = np.flatnonzero(np.asarray(box.face_cells.neighbour) >= 0)
    baffled = Mesh.from_csr(
        box.node_coords, box.face_nodes.offsets, box.face_nodes.face_node_indices,
        box.face_cells.owner, box.face_cells.neighbour, box.n_cells,
        face_patches={"baffle": interior},
    )  # fmt: skip
    with pytest.raises(ValueError, match=r"\['baffle'\] hold interior faces"):
        SHARP.build(baffled)


def test_a_point_on_no_face_is_refused():
    box = structured_grid_3d(1, 1, 1)
    stray = Mesh.from_csr(
        np.vstack([np.asarray(box.node_coords), [[5.0, 5.0, 5.0]]]),
        box.face_nodes.offsets, box.face_nodes.face_node_indices,
        box.face_cells.owner, box.face_cells.neighbour, box.n_cells,
    )  # fmt: skip
    with pytest.raises(ValueError, match="1 primal point"):
        SHARP.build(stray)


def test_two_cubes_touching_along_an_edge_are_refused():
    # Diagonal neighbours sharing only the edge x = y = 1: that edge has four boundary faces.
    cube = structured_grid_3d(2, 2, 1, 2.0, 2.0, 1.0)
    centroid = np.asarray(cube.geometry().cell.centroid)
    off_diagonal = np.flatnonzero(np.abs(centroid[:, 0] - centroid[:, 1]) > 0.5)
    pinched = without_cells(cube, off_diagonal, patch="cut")
    with pytest.raises(ValueError, match="other than two boundary faces"):
        SHARP.build(pinched)


def test_a_named_feature_edge_off_the_boundary_is_refused():
    primal = _box()
    edge_nodes, _ = primal.face_nodes.edges()
    points = np.asarray(primal.node_coords)
    inside = ~(np.isclose(points, 0.0) | np.isclose(points, [LX, LY, LZ])).any(axis=1)
    interior_edge = edge_nodes[inside[edge_nodes].all(axis=1)][:1]
    with pytest.raises(ValueError, match="not an edge of the primal boundary"):
        PolyhedralDual(feature_angle=1.0, feature_edges=interior_edge).build(primal)
