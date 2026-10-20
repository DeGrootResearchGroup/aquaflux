"""Hexahedra and pyramids made into a mesh, and the octree hexahedral core built from them.

Each core is checked against what its region implies, worked out here without the core's own
machinery: a box keeps exactly the cubes of its uniform grid; a refined core fills exactly the union
of its cubes, with no point at the middle of any cell's edge (where a hanging point would sit) and a
boundary of exactly the union's area (a face matched on one side only would add its area twice); the
capped surface is closed, manifold, and encloses the core and its caps.
"""

from __future__ import annotations

import itertools

import numpy as np
from aquaflux.mesh import closed_cell_residual
from aquaflux.meshing import NearSurface, OctreeCore, check_closed_surface, element_mesh

CUBE = np.array(list(itertools.product((0.0, 1.0), repeat=3)))[:, ::-1]  # index = x + 2y + 4z
HEX = [0, 1, 3, 2, 4, 5, 7, 6]


def _closed_and_volumes(mesh):
    volume = np.asarray(mesh.geometry().cell.volume)
    scale = float(np.max(np.asarray(mesh.geometry().face.area)))
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-12 * scale
    return volume


#: Two unit cubes side by side along x: point index = x + 3y + 6z.
PAIR = np.array([[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0, 2.0)])


def test_a_cube_is_one_hexahedron_or_six_pyramids_and_they_join_where_they_meet():
    hexahedron = [0, 1, 4, 3, 6, 7, 10, 9]
    # The second cube's faces, each a pyramid to its centre (point 12): x = 1, x = 2, y = 0, y = 1,
    # z = 0, z = 1.
    sides = [
        [1, 4, 10, 7],
        [2, 5, 11, 8],
        [1, 2, 8, 7],
        [4, 5, 11, 10],
        [1, 2, 5, 4],
        [7, 8, 11, 10],
    ]
    points = np.vstack([PAIR, [[1.5, 0.5, 0.5]]])
    pyramids = np.array([[*side, 12] for side in sides])
    mesh = element_mesh(points, hexahedra=np.array([hexahedron]), pyramids=pyramids)
    assert (mesh.n_cells, mesh.n_faces) == (7, 6 + 6 * 5 - 12 - 1)
    np.testing.assert_allclose(_closed_and_volumes(mesh), [1.0] + [1 / 6] * 6, rtol=1e-12)
    owner, neighbour = np.asarray(mesh.face_cells.owner), np.asarray(mesh.face_cells.neighbour)
    # The hexahedron's faces come first, in its own order; its side at x = 1 is a pyramid's base.
    rings = np.asarray(mesh.face_nodes.face_node_indices)[:24].reshape(6, 4)
    np.testing.assert_array_equal(rings[:2], [[0, 1, 4, 3], [6, 7, 10, 9]])
    np.testing.assert_array_equal(owner[:6], 0)
    np.testing.assert_array_equal(neighbour[:6], [-1, -1, -1, 1, -1, -1])
    assert np.count_nonzero(neighbour >= 0) == 1 + 12


def test_cells_are_numbered_tetrahedra_prisms_hexahedra_then_pyramids():
    # Four unit cubes apart along z, one cell in each, given in another order than they come out.
    points = np.vstack([*(CUBE + np.array([0.0, 0.0, 2.0 * k]) for k in range(4)), [0.5, 0.5, 6.5]])
    mesh = element_mesh(
        points,
        pyramids=np.array([[24, 25, 27, 26, 32]]),  # height one half: volume 1/6
        hexahedra=np.array([[16 + i for i in HEX]]),
        prisms=np.array([[8, 9, 10, 12, 13, 14]]),  # half a cube
        tetrahedra=np.array([[0, 1, 2, 4]]),  # a corner: volume 1/6
    )
    np.testing.assert_allclose(
        np.asarray(mesh.geometry().cell.volume), [1 / 6, 1 / 2, 1.0, 1 / 6], rtol=1e-12
    )


def _box(points):
    return np.all((points >= 0.0) & (points <= [2.0, 1.5, 1.0]), axis=1)


def test_a_box_keeps_exactly_the_cubes_of_its_uniform_grid_clear_of_its_walls():
    # Cubes of 0.25 from the origin, kept where they stay 0.2 clear of the walls: lower corners
    # 0.25 to 1.5 along x, 0.25 to 1.0 along y, 0.25 and 0.5 along z.
    core = OctreeCore(size=0.25, clearance=0.2).mesh(_box, [0, 0, 0], [2.0, 1.5, 1.0])
    expected = (
        np.array(list(itertools.product(np.arange(1, 7), np.arange(1, 5), np.arange(1, 3)))) * 0.25
    )
    lower = core.points[core.hexahedra[:, 0]]
    np.testing.assert_allclose(
        lower[np.lexsort(lower.T[::-1])], expected[np.lexsort(expected.T[::-1])], atol=1e-12
    )
    assert core.pyramids.size == core.tetrahedra.size == 0
    mesh = element_mesh(core.points, hexahedra=core.hexahedra)
    np.testing.assert_allclose(_closed_and_volumes(mesh), 0.25**3, rtol=1e-12)
    # Every boundary quadrilateral is capped: 2 (6 x 4 + 6 x 2 + 4 x 2) of them.
    assert core.caps.shape[0] == 88
    assert core.points.shape[0] == 7 * 5 * 3 + 88


def _centred_ball(radius):
    return lambda points: np.linalg.norm(points - 1.0, axis=1) < radius


def _union_of_cubes(core):
    """The kept cubes as occupied cells of the tree's finest grid, padded by a layer of empties."""
    tree = core.octree
    occupied = np.zeros(tuple(tree.shape << tree.depth) + np.array(2), dtype=bool)
    for corner, span in zip(tree.corner[core.kept], tree.span[core.kept], strict=True):
        x, y, z = corner + 1
        occupied[x : x + span, y : y + span, z : z + span] = True
    return occupied


def _union_area(core):
    occupied = _union_of_cubes(core)
    faces = sum(np.count_nonzero(np.diff(occupied, axis=axis)) for axis in range(3))
    return faces * core.octree.unit**2


def _refined_core():
    return OctreeCore(
        size=0.2, clearance=0.1, refinements=(NearSurface(_centred_ball(0.3), 0.05),)
    ).mesh(_centred_ball(0.9), [0, 0, 0], [2, 2, 2])


def test_a_refined_core_is_conformal_with_no_point_hanging_on_an_edge():
    core = _refined_core()
    assert set(np.unique(core.octree.level[core.kept]).tolist()) == {0, 1, 2}
    assert core.pyramids.shape[0] > 0 and core.tetrahedra.shape[0] > 0
    mesh = element_mesh(
        core.points, tetrahedra=core.tetrahedra, hexahedra=core.hexahedra, pyramids=core.pyramids
    )
    volume = _closed_and_volumes(mesh)
    assert np.all(volume > 0.0)
    np.testing.assert_allclose(volume.sum(), np.sum(core.octree.size[core.kept] ** 3), rtol=1e-12)
    boundary = np.asarray(mesh.face_cells.neighbour) < 0
    area = np.asarray(mesh.geometry().face.area)[boundary].sum()
    np.testing.assert_allclose(area, _union_area(core), rtol=1e-12)
    # A hanging point would sit at the middle of an edge of a larger cell.
    edges = {
        tuple(sorted(pair))
        for cells, rings in (
            (core.hexahedra, [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                              [0, 4], [1, 5], [2, 6], [3, 7]]),
            (core.pyramids, [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]]),
            (core.tetrahedra, list(itertools.combinations(range(4), 2))),
        )
        for cell in cells.tolist()
        for pair in ((cell[a], cell[b]) for a, b in rings)
    }  # fmt: skip
    pairs = np.array(sorted(edges))
    middles = 0.5 * (core.points[pairs[:, 0]] + core.points[pairs[:, 1]])
    points = {tuple(p) for p in np.round(core.points / core.octree.unit * 4).astype(int).tolist()}
    assert not points & {
        tuple(p) for p in np.round(middles / core.octree.unit * 4).astype(int).tolist()
    }


def _fans_per_point(triangles):
    """How many fans of triangles, joined across shared edges, meet at each point."""
    fans = {}
    for point in np.unique(triangles):
        around = [t for t in triangles.tolist() if point in t]
        label = list(range(len(around)))

        def root(i, label=label):
            while label[i] != i:
                i = label[i]
            return i

        for i, j in itertools.combinations(range(len(around)), 2):
            if len(set(around[i]) & set(around[j])) == 2:
                label[root(i)] = root(j)
        fans[point] = len({root(i) for i in range(len(around))})
    return fans


def _enclosed_volume(points, triangles):
    corner = points[triangles]
    return np.einsum("ij,ij->i", corner[:, 0], np.cross(corner[:, 1], corner[:, 2])).sum() / 6.0


def _surface_alone(core):
    used = np.unique(core.surface)
    return core.points[used], np.searchsorted(used, core.surface)


def test_the_capped_surface_is_closed_manifold_and_wound_into_the_core():
    core = _refined_core()
    points, triangles = _surface_alone(core)
    # Wound into the core: reversed, it is an outward closed surface.
    check_closed_surface(points, triangles[:, ::-1])
    assert set(_fans_per_point(triangles).values()) == {1}
    # Both kinds of boundary piece are present: caps' sides and transition cubes' triangles.
    assert core.surface.shape[0] > 4 * core.caps.shape[0] > 0
    edge = np.linalg.norm(core.points[core.caps[:, 0]] - core.points[core.caps[:, 1]], axis=1)
    caps = np.sum(edge**2 * (0.25 * edge) / 3.0)
    cubes = np.sum(core.octree.size[core.kept] ** 3)
    np.testing.assert_allclose(_enclosed_volume(points, triangles), -(cubes + caps), rtol=1e-12)
    # Each apex stands outside every cube, a quarter of its base's edge from it.
    apex = core.points[core.caps[:, 4]]
    occupied = _union_of_cubes(core)
    cell = np.floor((apex - core.octree.origin) / core.octree.unit).astype(int) + 1
    assert not np.any(occupied[tuple(cell.T)])


def _touching_along_an_edge(points):
    first = np.all((points >= 0.0) & (points <= 1.0), axis=1)
    return first | np.all((points >= [1.0, 1.0, 0.0]) & (points <= [2.0, 2.0, 1.0]), axis=1)


def test_cubes_meeting_only_along_an_edge_are_parted_there_and_nowhere_else():
    core = OctreeCore(size=0.25, clearance=0.0).mesh(_touching_along_an_edge, [0, 0, 0], [2, 2, 1])
    points, triangles = _surface_alone(core)
    check_closed_surface(points, triangles[:, ::-1])
    assert set(_fans_per_point(triangles).values()) == {1}
    # Only cubes on the shared edge (x = y = 1) are dropped, and fewer than the two columns there.
    lower = core.octree.lower
    on_edge = np.isclose(lower[:, 0], 0.75) & np.isclose(lower[:, 1], 0.75)
    on_edge |= np.isclose(lower[:, 0], 1.0) & np.isclose(lower[:, 1], 1.0)
    dropped = np.setdiff1d(np.flatnonzero(_touching_along_an_edge(core.octree.lower + 0.125)),
                           core.kept)  # fmt: skip
    assert 4 <= dropped.size < 8
    assert np.all(on_edge[dropped])
