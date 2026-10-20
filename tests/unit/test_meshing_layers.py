"""Prism layers grown inward from a closed surface, checked against the volumes they must sweep.

On a cube every point's layers stretch to clear each face it lies on by the full thickness, so a
cube layered all round leaves exactly the cube shrunk by that thickness on every side, and one with
its top left bare (the layers sliding along it) a box shrunk on five sides; the prisms between fill
the difference exactly. Where walls face each other closely the layers are thinned, and where a
triangle drawn in to meet the layers would turn over, thinned until it does not.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.mesh import Mesh
from aquaflux.meshing import Polygons, PrismLayers, element_mesh

from tests.support.meshes import tetrahedral_grid_3d, voxel_surface


def _outward_surface(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """A mesh's boundary as points and triangles wound out of it, every point used."""
    boundary = np.asarray(mesh.face_cells.neighbour) < 0
    rings = np.asarray(mesh.face_nodes.face_node_indices).reshape(-1, 3)[boundary]
    normal = np.asarray(mesh.geometry().face.normal)[boundary]
    points = np.asarray(mesh.node_coords)
    corner = points[rings]
    ring_normal = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
    rings = np.where(
        (np.einsum("ij,ij->i", ring_normal, normal) < 0)[:, None], rings[:, ::-1], rings
    )
    used, rings = np.unique(rings, return_inverse=True)
    return points[used], rings.reshape(-1, 3)


def _box(n: int, scale=(1.0, 1.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    points, triangles = _outward_surface(tetrahedral_grid_3d(n))
    return points * np.asarray(scale), triangles


def _shell_volume(extrusion) -> float:
    return float(
        np.asarray(
            element_mesh(extrusion.points, prisms=extrusion.prisms).geometry().cell.volume
        ).sum()
    )


LAYERS = PrismLayers(first_height=0.01, growth=1.25, count=4)


def test_the_duals_layer_boundaries_grow_geometrically_from_the_first_height():
    # A dual layer's boundary is the centroid of the prism under it: halfway up that prism.
    tops = LAYERS.offsets()
    centroids = 0.5 * (np.concatenate([[0.0], tops[:-1]]) + tops)
    np.testing.assert_allclose(centroids, 0.01 * np.cumsum(1.25 ** np.arange(4)), rtol=1e-12)
    assert tops[0] == pytest.approx(0.02)
    assert np.all(np.diff(np.concatenate([[0.0], tops])) > 0)


def test_a_cube_layered_all_round_leaves_the_cube_shrunk_by_the_thickness():
    points, triangles = _box(3)
    grown = LAYERS.extrude(points, Polygons(triangles), np.ones(len(triangles), dtype=bool))
    thickness = LAYERS.offsets()[-1]
    assert grown.prisms.shape == (len(triangles) * 4, 6)
    assert _shell_volume(grown) == pytest.approx(1.0 - (1.0 - 2 * thickness) ** 3, rel=1e-12)
    inner = grown.points[np.unique(grown.inner.triangles)]
    np.testing.assert_allclose(inner.min(axis=0), thickness, rtol=1e-12)
    np.testing.assert_allclose(inner.max(axis=0), 1.0 - thickness, rtol=1e-12)
    assert grown.sides.shape == (0, 4)


def test_layers_meeting_a_bare_face_slide_along_it():
    # The top (z = 1) bare: the layers' sides lie on it, and the box left inside is shrunk on the
    # other five sides only, its top drawn in to meet them.
    points, triangles = _box(3)
    top = np.all(np.isclose(points[triangles][:, :, 2], 1.0), axis=1)
    grown = LAYERS.extrude(points, Polygons(triangles), ~top)
    thickness = LAYERS.offsets()[-1]
    side_points = grown.points[grown.sides]
    np.testing.assert_allclose(side_points[:, :, 2], 1.0, atol=1e-12)
    assert len(grown.sides) == 4 * 3 * 4  # four edges of the top, three segments each, four layers
    assert _shell_volume(grown) == pytest.approx(
        1.0 - (1.0 - 2 * thickness) ** 2 * (1.0 - thickness), rel=1e-12
    )
    drawn = grown.points[grown.walls.triangles[top]]
    np.testing.assert_allclose(drawn[:, :, 2], 1.0, atol=1e-12)
    assert drawn[:, :, :2].min() == pytest.approx(thickness)


def test_walls_close_together_share_the_gap():
    # A slab 0.05 thick: four layers would be 0.104 thick from each face, so each side takes at
    # most gap_share of the 0.05 between them. Its middle is far from the slab's thin edges, where
    # the layers are thinned anyway to keep the prisms right side out.
    points, triangles = _box(6, scale=(1.0, 1.0, 0.05))
    grown = LAYERS.extrude(points, Polygons(triangles), np.ones(len(triangles), dtype=bool))
    n = len(points)
    lifted = grown.points[n:].reshape(4, n, 3)  # each layer's points, in the surface's order
    middle = np.isclose(points[:, 2], 0.0) & np.all(
        (points[:, :2] > 0.3) & (points[:, :2] < 0.7), axis=1
    )
    rise = lifted[-1, middle, 2]
    assert middle.sum() >= 4
    assert np.all(rise <= LAYERS.gap_share * 0.05 + 1e-12)
    assert np.all(rise > 0.5 * LAYERS.gap_share * 0.05)


def test_layers_meeting_a_tilted_bare_face_turn_to_slide_along_its_plane():
    # The top tilted to z = 1 + 0.3 x: an inward wall normal at its rim (+x, +y, -x, -y) leaves the
    # top's plane, so it must be turned into it for the prisms' sides to lie on the top.
    points, triangles = _box(3)
    points = points.copy()
    points[:, 2] *= 1.0 + 0.3 * points[:, 0]
    top = np.all(
        np.isclose(points[triangles][:, :, 2], 1.0 + 0.3 * points[triangles][:, :, 0]), axis=1
    )
    top &= np.all(points[triangles][:, :, 2] > 0.5, axis=1)
    grown = LAYERS.extrude(points, Polygons(triangles), ~top)
    sides = grown.points[grown.sides]
    np.testing.assert_allclose(sides[:, :, 2], 1.0 + 0.3 * sides[:, :, 0], atol=1e-12)
    drawn = grown.points[grown.walls.triangles[top]]
    np.testing.assert_allclose(drawn[:, :, 2], 1.0 + 0.3 * drawn[:, :, 0], atol=1e-12)


def test_a_drawn_in_triangle_that_would_turn_over_is_saved_by_thinning():
    # The top of a finer cube left bare, layers thicker than its triangles: drawn in by the full
    # thickness, the triangles at its rim would turn over; with no halving allowed it is refused,
    # with halving the layers there are thinned until none does.
    points, triangles = _box(4)
    top = np.all(np.isclose(points[triangles][:, :, 2], 1.0), axis=1)
    thick = PrismLayers(first_height=0.12, growth=1.2, count=2, max_halvings=0)
    with pytest.raises(ValueError, match="inside out after their layers were thinned 0 times"):
        thick.extrude(points, Polygons(triangles), ~top)
    grown = PrismLayers(first_height=0.12, growth=1.2, count=2).extrude(
        points, Polygons(triangles), ~top
    )
    c = grown.points[grown.walls.triangles[top]]
    normal = np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0])
    assert np.all(normal[:, 2] > 0.0)
    assert _shell_volume(grown) > 0.0
    # Around the thinned rim the layers taper: no point is more than half again as thick as a
    # neighbour, and the bottom's middle, far from the rim, is thicker than the rim.
    # Each layer adds a point for every point that moves, in index order.
    n, movers = len(points), np.unique(triangles[~top])
    reach = np.zeros(n)
    reach[movers] = np.linalg.norm(grown.points[n + movers.size :] - points[movers], axis=1)
    edges = np.unique(
        np.sort(
            np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]),
            axis=1,
        ),
        axis=0,
    )
    # (on the bottom's interior, where nothing is stretched to clear a second face)
    flat = np.isclose(points[:, 2], 0.0) & np.all(
        (points[:, :2] > 1e-9) & (points[:, :2] < 1 - 1e-9), axis=1
    )
    a, b = reach[edges[:, 0]], reach[edges[:, 1]]
    both = flat[edges[:, 0]] & flat[edges[:, 1]]
    assert np.all(np.maximum(a, b)[both] <= 1.5 * np.minimum(a, b)[both] * (1 + 1e-9))
    bottom_middle = np.isclose(points[:, 2], 0.0) & np.all(
        np.abs(points[:, :2] - 0.5) < 0.2, axis=1
    )
    rim = np.isclose(points[:, 2], 1.0) & (reach > 0)
    assert np.all(reach[bottom_middle] <= 0.288 + 1e-12)
    assert reach[bottom_middle].min() > 1.4 * reach[rim].min()


def test_a_prism_that_would_turn_over_at_a_concave_edge_is_saved_by_thinning():
    # An L, layered all round, its layers thicker than its cells: at the inside edge the layers of
    # the two walls run into each other and prisms turn over -- refused with no halving allowed,
    # every prism right side out with it.
    from tests.support.meshes import without_cells

    block = tetrahedral_grid_3d(4)
    centroid = np.asarray(block.geometry().cell.centroid)
    l_shape = without_cells(
        block, np.flatnonzero((centroid[:, 0] > 0.5) & (centroid[:, 2] > 0.5)), patch="cut"
    )
    points, triangles = _outward_surface(l_shape)
    every = np.ones(len(triangles), dtype=bool)
    with pytest.raises(ValueError, match="inside out"):
        PrismLayers(first_height=0.1, growth=1.2, count=2, max_halvings=0).extrude(
            points, Polygons(triangles), every
        )
    grown = PrismLayers(first_height=0.1, growth=1.2, count=2).extrude(
        points, Polygons(triangles), every
    )
    volume = np.asarray(element_mesh(grown.points, prisms=grown.prisms).geometry().cell.volume)
    assert np.all(volume > 0.0)


def test_a_point_with_nowhere_to_slide_is_refused():
    # A square pyramid: its apex lies on four sides; with three of them bare and the fourth
    # layered, the apex would have to slide along three planes at once.
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1.0]], float)
    triangles = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    layered = np.array([True, True, True, False, False, False])
    with pytest.raises(ValueError, match="three or more faces without layers, so cannot slide"):
        LAYERS.extrude(points, Polygons(triangles), layered)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"growth": 1.0}, "growth must be greater than 1"),
        ({"first_height": 0.0}, "first_height must be a positive"),
        ({"count": 0}, "count must be a whole number >= 1"),
        ({"gap_share": 0.6}, r"gap_share must be in \(0, 0.5\]"),
        ({"max_stretch": 0.5}, "max_stretch must be >= 1"),
    ],
)
def test_settings_out_of_range_are_refused(settings, message):
    with pytest.raises(ValueError, match=message):
        PrismLayers(**{"first_height": 0.01, "growth": 1.2, "count": 3, **settings})


def _cells_volume(extrusion) -> np.ndarray:
    return np.asarray(
        element_mesh(extrusion.points, prisms=extrusion.prisms, hexahedra=extrusion.hexahedra)
        .geometry()
        .cell.volume
    )


def test_a_cube_of_quadrilaterals_layered_all_round_grows_hexahedra():
    points, quads = voxel_surface([(0, 0, 0)], 3)
    faces = Polygons(quads=quads)
    assert np.all(np.einsum("ij,ij->i", faces.normals(points), faces.centroids(points) - 0.5) > 0)
    grown = LAYERS.extrude(points, faces, np.ones(faces.n_faces, dtype=bool))
    thickness = LAYERS.offsets()[-1]
    assert grown.prisms.shape == (0, 6)
    assert grown.hexahedra.shape == (len(quads) * 4, 8)
    volume = _cells_volume(grown)
    assert np.all(volume > 0.0)
    assert volume.sum() == pytest.approx(1.0 - (1.0 - 2 * thickness) ** 3, rel=1e-12)
    top = grown.points[np.unique(grown.inner.quads)]
    np.testing.assert_allclose(top.min(axis=0), thickness, rtol=1e-12)
    np.testing.assert_allclose(top.max(axis=0), 1.0 - thickness, rtol=1e-12)
    assert grown.inner.triangles.shape == (0, 3)
    # Each moved point's column: the point, then its layers' points straight above it.
    assert grown.columns.shape == (len(points), 5)
    np.testing.assert_array_equal(grown.columns[:, 0], np.arange(len(points)))


def test_triangles_and_quadrilaterals_layered_together_join_conformally():
    # The cube's top cut into triangles, its other faces quadrilaterals: prisms over the top,
    # hexahedra elsewhere, sharing their sides where the top meets the walls.
    points, quads = voxel_surface([(0, 0, 0)], 3)
    on_top = np.all(np.isclose(points[quads][:, :, 2], 1.0), axis=1)
    triangles = np.concatenate([quads[on_top][:, [0, 1, 2]], quads[on_top][:, [0, 2, 3]]])
    faces = Polygons(triangles, quads[~on_top])
    grown = LAYERS.extrude(points, faces, np.ones(faces.n_faces, dtype=bool))
    thickness = LAYERS.offsets()[-1]
    assert grown.prisms.shape[0] == 4 * len(triangles)
    assert grown.hexahedra.shape[0] == 4 * np.count_nonzero(~on_top)
    volume = _cells_volume(grown)  # element_mesh refuses a face of three cells, or a gap
    assert volume.sum() == pytest.approx(1.0 - (1.0 - 2 * thickness) ** 3, rel=1e-12)


def test_quadrilateral_layers_meeting_a_bare_quadrilateral_face_slide_along_it():
    points, quads = voxel_surface([(0, 0, 0)], 3)
    top = np.all(np.isclose(points[quads][:, :, 2], 1.0), axis=1)
    grown = LAYERS.extrude(points, Polygons(quads=quads), ~top)
    thickness = LAYERS.offsets()[-1]
    np.testing.assert_allclose(grown.points[grown.sides][:, :, 2], 1.0, atol=1e-12)
    assert len(grown.sides) == 4 * 3 * 4
    assert _cells_volume(grown).sum() == pytest.approx(
        1.0 - (1.0 - 2 * thickness) ** 2 * (1.0 - thickness), rel=1e-12
    )
    drawn = grown.points[grown.walls.quads[top]]
    np.testing.assert_allclose(drawn[:, :, 2], 1.0, atol=1e-12)
    assert drawn[:, :, :2].min() == pytest.approx(thickness)
    # The drawn-in quadrilaterals are left for the fill, as the layers' top is.
    assert grown.inner.quads.shape[0] == len(quads)


def test_a_hexahedron_that_would_turn_over_at_a_concave_edge_is_saved_by_thinning():
    # An L of three cubes of quadrilaterals, its layers thicker than its cells: at the inside edge
    # the two walls' hexahedra run into each other -- refused with no halving allowed, every one
    # right side out with it.
    points, quads = voxel_surface([(0, 0, 0), (1, 0, 0), (0, 0, 1)], 2, size=0.5)
    every = np.ones(len(quads), dtype=bool)
    with pytest.raises(ValueError, match="inside out"):
        PrismLayers(first_height=0.1, growth=1.2, count=2, max_halvings=0).extrude(
            points, Polygons(quads=quads), every
        )
    grown = PrismLayers(first_height=0.1, growth=1.2, count=2).extrude(
        points, Polygons(quads=quads), every
    )
    assert np.all(_cells_volume(grown) > 0.0)
