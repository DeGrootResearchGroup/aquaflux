"""A closed surface with a sheet standing in it, opened along the sheet into one closed surface.

The fixtures are the unit cube's surface on a lattice of spacing 1/4 with a sheet in the plane
``x = 1/2`` on the same lattice, so the walls' triangles already meet the sheet along its edge on a
wall. Each check is a property the opened surface must have whatever the numbering: closed and
wound out, one fan of faces round every point, each copy of a point holding the faces of one side
of the sheet, and the volume enclosed unchanged.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from aquaflux.meshing import Polygons, check_closed_surface, element_mesh
from aquaflux.meshing.sheets import SlitSurface, opened, parted
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from tests.support.meshes import box_surface, tetrahedral_grid_3d

SPACING = 0.25


def _with_sheet(ys, zs, x=0.5):
    """The cube's surface and a sheet in the plane ``x`` (default 1/2) over ``ys x zs`` (lattice
    ticks), its squares cut along their first diagonal; the sheet's triangles are the drawing's
    face 1, the walls' 0."""
    points, triangles = box_surface(0.0, 1.0, 4, into_box=False)
    points = [tuple(np.round(p, 12)) for p in points]
    index = {p: k for k, p in enumerate(points)}

    def point(y, z):
        key = (x, round(y, 12), round(z, 12))
        if key not in index:
            index[key] = len(points)
            points.append(key)
        return index[key]

    sheet = []
    for y0, y1 in pairwise(ys):
        for z0, z1 in pairwise(zs):
            ring = [point(y0, z0), point(y1, z0), point(y1, z1), point(y0, z1)]
            sheet += [ring[:3], [ring[0], ring[2], ring[3]]]
    faces = Polygons(np.vstack([triangles, sheet]))
    face = np.r_[np.zeros(len(triangles), np.int64), np.ones(len(sheet), np.int64)]
    return np.array(points, dtype=float), faces, face


def _onto_the_plane(points, faces):
    """Points moved onto the sheet's plane: the drawing every test uses."""
    moved = np.array(points, dtype=float)
    moved[:, 0] = 0.5
    return moved


@pytest.fixture
def standing():
    """A sheet standing on the wall ``y = 0``, ``y`` in ``[0, 1/2]``, ``z`` in ``[1/4, 3/4]``:
    free along its other three sides."""
    return _with_sheet([0.0, 0.25, 0.5], [0.25, 0.5, 0.75])


def _fans(triangles, n_points) -> np.ndarray:
    """How many fans of triangles each point has: faces round it joined across its edges."""
    fans = np.zeros(n_points, dtype=np.int64)
    for p in range(n_points):
        mine = np.flatnonzero(np.any(triangles == p, axis=1))
        if not mine.size:
            continue
        # Two faces at the point are joined when they share one of its edges: two corners.
        joined = [
            (i, j)
            for i in range(mine.size)
            for j in range(i + 1, mine.size)
            if len(set(triangles[mine[i]]) & set(triangles[mine[j]])) == 2
        ]
        rows, cols = np.array(joined, dtype=np.int64).reshape(-1, 2).T
        graph = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(mine.size, mine.size))
        fans[p] = connected_components(graph, directed=False)[0]
    return fans


def _enclosed(points, triangles) -> float:
    corner = points[triangles]
    return float(
        np.einsum("ij,ij->i", corner[:, 0], np.cross(corner[:, 1], corner[:, 2])).sum() / 6
    )


def test_the_opened_surface_is_closed_wound_out_and_one_fan_round_every_point(standing):
    points, faces, face = standing
    result = opened(points, faces, face, np.array([1]), _onto_the_plane)
    triangles = result.faces.triangles
    check_closed_surface(result.points, triangles)
    assert np.all(_fans(triangles, result.points.shape[0]) == 1)
    # The two sides of the sheet cancel: the volume is the cube's.
    np.testing.assert_allclose(_enclosed(result.points, triangles), 1.0, rtol=1e-12)
    # No two faces on the same three points (a triangle with every corner on the free edge would be).
    keys = np.unique(np.sort(triangles, axis=1), axis=0)
    assert keys.shape[0] == triangles.shape[0]


def test_a_point_is_copied_where_the_sides_part_and_kept_along_the_free_edge(standing):
    points, faces, face = standing
    result = opened(points, faces, face, np.array([1]), _onto_the_plane)
    copies = np.bincount(result.origin, minlength=result.origin.size)

    def copies_at(y, z):
        at = np.flatnonzero(np.all(np.isclose(result.points, [0.5, y, z]), axis=1))
        return np.unique(result.origin[at]).size, at.size

    # Inside the sheet and along its foot on the wall: two points each, one per side.
    assert copies_at(0.25, 0.5) == (1, 2)
    assert copies_at(0.0, 0.5) == (1, 2)
    # Along the free edge, and where it reaches the wall: one point, the fluid running round it.
    for y, z in [(0.5, 0.25), (0.5, 0.5), (0.5, 0.75), (0.25, 0.25), (0.0, 0.25), (0.0, 0.75)]:
        assert copies_at(y, z) == (1, 1), (y, z)
    assert copies.max() <= 2


def test_each_copy_of_a_point_holds_the_faces_of_one_side(standing):
    # At the sheet's foot on the wall, one copy must hold the wall's triangles on x < 1/2 and the
    # sheet's side facing x < 1/2's fluid (wound to point out of it, along +x); the other copy the
    # rest. A pairing that joined faces across the sheet would mix them.
    points, faces, face = standing
    result = opened(points, faces, face, np.array([1]), _onto_the_plane)
    triangles, points = result.faces.triangles, result.points
    centroid = points[triangles].mean(axis=1)
    normal = np.cross(*(points[triangles[:, 1:]] - points[triangles[:, :1]]).transpose(1, 0, 2))
    on_sheet = result.face[: triangles.shape[0]] == 1
    for copy in np.flatnonzero(np.all(np.isclose(points, [0.5, 0.0, 0.5]), axis=1)):
        mine = np.any(triangles == copy, axis=1)
        walls = mine & ~on_sheet
        sides = mine & on_sheet
        assert walls.any() and sides.any()
        x_side = np.sign(centroid[walls, 0] - 0.5)
        assert np.all(x_side == x_side[0])
        # The fluid on x < 1/2 is bounded by the sheet's side whose outward normal is +x.
        assert np.all(np.sign(normal[sides, 0]) == -x_side[0])


def test_a_sheet_free_on_every_side_is_opened_round_its_whole_edge():
    points, faces, face = _with_sheet([0.25, 0.5, 0.75], [0.25, 0.5, 0.75])
    result = opened(points, faces, face, np.array([1]), _onto_the_plane)
    triangles = result.faces.triangles
    check_closed_surface(result.points, triangles)
    assert np.all(_fans(triangles, result.points.shape[0]) == 1)
    np.testing.assert_allclose(_enclosed(result.points, triangles), 1.0, rtol=1e-12)
    # A point parts exactly when it is strictly inside the sheet (the middle, and the midpoints of
    # the two diagonals run between points of the edge); the fluid runs round the edge's points.
    copied = np.bincount(result.origin) == 2
    sheet_point = np.isclose(result.points[: copied.size, 0], 0.5)
    inside = np.all(
        (result.points[: copied.size, 1:] > 0.25 + 1e-12)
        & (result.points[: copied.size, 1:] < 0.75 - 1e-12),
        axis=1,
    )
    np.testing.assert_array_equal(copied, sheet_point & inside)
    assert np.count_nonzero(copied) == 3


def test_a_sheet_in_a_corner_parts_the_corner_point_too():
    # Standing on the walls y = 0 and z = 0: the cube's edge point (1/2, 0, 0) is on both feet.
    points, faces, face = _with_sheet([0.0, 0.25, 0.5], [0.0, 0.25, 0.5])
    result = opened(points, faces, face, np.array([1]), _onto_the_plane)
    triangles = result.faces.triangles
    check_closed_surface(result.points, triangles)
    assert np.all(_fans(triangles, result.points.shape[0]) == 1)
    corner = np.all(np.isclose(result.points, [0.5, 0.0, 0.0]), axis=1)
    assert np.count_nonzero(corner) == 2


def test_an_edge_between_two_points_of_the_free_edge_is_cut_at_its_middle():
    # The single square (1/4..1/2)^2 is cut along its diagonal (1/4, 1/4)-(1/2, 1/2), whose ends
    # are both on the free edge; it is cut at (1/2, 3/8, 3/8), moved onto the sheet.
    points, faces, face = _with_sheet([0.25, 0.5], [0.25, 0.5])
    moved = []

    def onto(at, under):
        moved.append((np.array(at), np.array(under)))
        return _onto_the_plane(at, under)

    result = opened(points, faces, face, np.array([1]), onto)
    assert len(moved) == 1
    np.testing.assert_allclose(moved[0][0], [[0.5, 0.375, 0.375]])
    np.testing.assert_array_equal(moved[0][1], [1])
    check_closed_surface(result.points, result.faces.triangles)
    assert np.all(_fans(result.faces.triangles, result.points.shape[0]) == 1)
    # The midpoint is the one point inside the sheet, so it alone is copied.
    middle = np.all(np.isclose(result.points, [0.5, 0.375, 0.375]), axis=1)
    assert np.count_nonzero(middle) == 2


def test_a_surface_without_sheets_comes_back_unchanged(standing):
    points, faces, face = standing
    walls = face == 0
    result = opened(points, faces.where(walls), face[walls], np.array([1]), _onto_the_plane)
    np.testing.assert_array_equal(result.faces.triangles, faces.triangles[walls])
    np.testing.assert_array_equal(result.origin, np.arange(points.shape[0]))


def test_a_sheet_of_quadrilaterals_is_refused(standing):
    points, faces, face = standing
    sheet = faces.triangles[face == 1][:2]
    quads = Polygons(faces.triangles[face == 0], np.array([[*sheet[0], sheet[1][2]]]))
    with pytest.raises(ValueError, match="sheet is meshed with quadrilaterals"):
        opened(points, quads, np.r_[face[face == 0], 1], np.array([1]), _onto_the_plane)


def test_two_faces_meeting_an_edge_at_one_angle_are_refused_unless_they_are_twins(standing):
    # A second copy of one sheet triangle, as if two sheets lay on top of each other.
    points, faces, face = standing
    extra = faces.triangles[face == 1][3]
    doubled = Polygons(np.vstack([faces.triangles, extra]))
    with pytest.raises(ValueError, match="at one angle, and they are not"):
        opened(points, doubled, np.r_[face, 1], np.array([1]), _onto_the_plane)


# ---------------------------------------------------------------------------------------------
# A fill about a sheet: the tetrahedra on each side given that side's points
# ---------------------------------------------------------------------------------------------


def _kuhn_cube(n: int):
    """The unit cube's Kuhn tetrahedra on an ``n`` lattice: points, tetrahedra and the triangles of
    every face, with each face's owner and neighbour (``-1`` on the boundary)."""
    mesh = tetrahedral_grid_3d(n)
    points = np.asarray(mesh.node_coords)
    rings = np.asarray(mesh.face_nodes.face_node_indices).reshape(-1, 3)
    owner, neighbour = np.asarray(mesh.face_cells.owner), np.asarray(mesh.face_cells.neighbour)
    corners = [set() for _ in range(mesh.n_cells)]
    for face, ring in enumerate(rings):
        for cell in (owner[face], neighbour[face]):
            if cell >= 0:
                corners[cell].update(ring.tolist())
    tetrahedra = np.array([sorted(c) for c in corners])
    # Each tetrahedron positively oriented, and each boundary triangle wound out of the cube.
    corner = points[tetrahedra]
    volume = np.einsum(
        "ij,ij->i", np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0]),
        corner[:, 3] - corner[:, 0],
    )  # fmt: skip
    tetrahedra[volume < 0] = tetrahedra[volume < 0][:, [0, 2, 1, 3]]
    return points, tetrahedra, rings, owner, neighbour


def _outward(points, rings, cells_centroid):
    """Each ring wound so its normal points away from the given centroid."""
    corner = points[rings]
    normal = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
    away = np.einsum("ij,ij->i", normal, corner.mean(axis=1) - cells_centroid) > 0
    return np.where(away[:, None], rings, rings[:, ::-1])


@pytest.fixture
def slit_cube():
    """Kuhn tetrahedra of the unit cube (lattice 1/4) with a sheet on their faces in ``x = 1/2``,
    ``y <= 1/2``: standing on the walls ``y = 0``, ``z = 0`` and ``z = 1``, free along ``y = 1/2``.
    The surface (walls, then the sheet) is opened along it."""
    points, tetrahedra, rings, owner, neighbour = _kuhn_cube(4)
    centroid = points[tetrahedra].mean(axis=1)
    corner = points[rings]
    walls = neighbour < 0
    sheet = (neighbour >= 0) & np.all(np.isclose(corner[..., 0], 0.5), axis=1)
    sheet &= np.all(corner[..., 1] <= 0.5 + 1e-12, axis=1)
    triangles = np.vstack([_outward(points, rings[walls], centroid[owner[walls]]), rings[sheet]])
    face = np.r_[np.zeros(walls.sum(), np.int64), np.ones(sheet.sum(), np.int64)]
    surface = opened(points, Polygons(triangles), face, np.array([1]), _onto_the_plane)
    assert surface.points.shape[0] == surface.origin.size  # no edge was cut: the lattice has none
    assert np.array_equal(surface.points[: points.shape[0]], points)
    return surface, tetrahedra, int(sheet.sum())


def test_a_slit_surface_gives_a_fill_the_sheet_once_and_the_rest_as_it_is(slit_cube):
    surface, _, n_sheet = slit_cube
    slit = SlitSurface.of(surface.faces.triangles, surface.origin)
    assert slit.internal.shape[0] == n_sheet
    assert slit.triangles.shape[0] == surface.faces.n_faces - 2 * n_sheet
    # Every corner at the place it shares with its copies: the original point.
    assert slit.internal.max() < surface.origin.size and np.all(
        surface.origin[slit.internal] == slit.internal
    )
    on_sheet = np.isclose(surface.points[slit.internal][..., 0], 0.5).all()
    assert on_sheet


def test_the_tetrahedra_on_each_side_of_a_sheet_take_that_sides_points(slit_cube):
    # The fill's tetrahedra use the original points; given back their own side's copies, their
    # boundary faces must be the opened surface's exactly -- the walls and both sides of the sheet
    # -- and no face between two of them may lie in the sheet.
    surface, tetrahedra, n_sheet = slit_cube
    result = parted(tetrahedra, surface.points, surface.faces.triangles, surface.origin)
    mesh = element_mesh(surface.points, result)
    neighbour = np.asarray(mesh.face_cells.neighbour)
    rings = np.asarray(mesh.face_nodes.face_node_indices).reshape(-1, 3)
    boundary = np.unique(np.sort(rings[neighbour < 0], axis=1), axis=0)
    expected = np.unique(np.sort(surface.faces.triangles, axis=1), axis=0)
    np.testing.assert_array_equal(boundary, expected)
    volume = np.asarray(mesh.geometry().cell.volume)
    assert np.all(volume > 0.0)
    np.testing.assert_allclose(volume.sum(), 1.0, rtol=1e-12)
    # Each tetrahedron beside the sheet holds the copies of its own side: its corners there are
    # those of the sheet faces wound to point away from it.
    corner = surface.points[result]
    side = np.sign(corner.mean(axis=1)[:, 0] - 0.5)
    sheet_faces = surface.faces.triangles[surface.face == 1]
    normal_x = np.cross(
        *(surface.points[sheet_faces[:, 1:]] - surface.points[sheet_faces[:, :1]]).transpose(
            1, 0, 2
        )
    )[:, 0]
    for s in (-1.0, 1.0):
        # The fluid on the side x < 1/2 is bounded by the sheet's faces whose normal is +x.
        own = np.unique(sheet_faces[np.sign(normal_x) == -s])
        theirs = np.setdiff1d(np.unique(sheet_faces[np.sign(normal_x) == s]), own)
        assert not np.isin(result[side == s], theirs).any()
    assert n_sheet > 0


def test_the_dual_of_a_fill_about_a_sheet_closes_and_cuts_the_cells_at_its_free_edge(slit_cube):
    # The free edge folds the cube's boundary right back on itself: the most concave fold there
    # is. Each point on it is cut in two, one piece per side; read as a convex fold it would not be.
    from aquaflux.mesh import closed_cell_residual
    from aquaflux.meshing import PolyhedralDual

    surface, tetrahedra, _ = slit_cube
    result = parted(tetrahedra, surface.points, surface.faces.triangles, surface.origin)
    primal = element_mesh(surface.points, result)
    dual = PolyhedralDual(feature_angle=np.radians(45)).build(primal)
    free = np.isclose(surface.points[:, 0], 0.5) & np.isclose(surface.points[:, 1], 0.5)
    assert dual.n_cells == primal.n_nodes + np.count_nonzero(free)
    volume = np.asarray(dual.geometry().cell.volume)
    assert np.all(volume > 0.0)
    np.testing.assert_allclose(volume.sum(), 1.0, rtol=1e-12)
    scale = float(np.max(np.asarray(dual.geometry().face.area)))
    assert float(np.max(np.asarray(closed_cell_residual(dual)))) < 1e-12 * scale


def test_layers_grow_alike_on_both_sides_of_a_sheets_foot():
    # A sheet a quarter of the way across the cube, standing on the wall y = 0. The walls' layers
    # slide up it from its foot on both sides; on the narrow side they are thinned against the wall
    # x = 0 and taper towards the foot, on the wide side not. Each copy of a foot point must still
    # grow the same column, or the two sides' layer tops would not meet at the sheet.
    from aquaflux.meshing import PrismLayers

    points, faces, face = _with_sheet([0.0, 0.25, 0.5], [0.25, 0.5, 0.75], x=0.25)

    def onto(at, under):
        moved = np.array(at, dtype=float)
        moved[:, 0] = 0.25
        return moved

    surface = opened(points, faces, face, np.array([1]), onto)
    layers = PrismLayers(first_height=0.08, growth=1.2, count=2)
    layered = surface.face == 0
    copies = np.flatnonzero(surface.origin != np.arange(surface.origin.size))
    originals = surface.origin[copies]

    def columns_of(grown, wanted):
        row = {int(base): k for k, base in enumerate(grown.columns[:, 0])}
        return grown.points[grown.columns[[row[int(p)] for p in wanted]]]

    tied = layers.extrude(surface.points, surface.faces, layered, twins=surface.origin)
    foot = np.isclose(surface.points[copies, 1], 0.0)  # the copies on the wall y = 0
    assert foot.any()
    np.testing.assert_array_equal(columns_of(tied, copies[foot]), columns_of(tied, originals[foot]))
    # Untied, the two sides grow differently here: the fixture exercises the tie.
    free = layers.extrude(surface.points, surface.faces, layered)
    assert not np.allclose(columns_of(free, copies[foot]), columns_of(free, originals[foot]))
