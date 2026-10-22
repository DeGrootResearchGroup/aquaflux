"""Hexahedral blocks, their assembly and the round layouts, checked against independent answers.

A flat block must reproduce the tensor-product grid of its node fractions (and so the structured
grid's points); hexahedra and pyramids must make the mesh their corners imply; a projected face must
lie on its surface; joined blocks must share exactly the points of their common faces, and a join
cut differently on its two sides must be refused; each round layout's cells must be right-handed,
its boundary on its cylinders or spheres, and its volume -- and its dual's -- must approach the
exact solid's at second order.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from aquaflux.mesh import closed_cell_residual, structured_grid_3d
from aquaflux.meshing import (
    Annulus,
    Arc,
    BlockMesh,
    CubedSphereCap,
    HexBlock,
    PipeOGrid,
    PolyhedralDual,
    SampledCurve,
    SphereSurface,
    element_mesh,
)

DUAL = PolyhedralDual(feature_angle=math.radians(45))
UNIT = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=float,
)


def _box(extent, divisions, origin=(0.0, 0.0, 0.0), **options):
    return HexBlock(np.asarray(origin) + UNIT * np.asarray(extent), divisions, **options)


def _corner_jacobians(points, hexahedra):
    """The determinant of each hexahedron's edge frame at each of its eight corners, ``(n, 8)``."""
    # For each corner: the corners along u, v and w from it, and the sign turning them to +u, +v, +w.
    frames = [
        (1, 3, 4, 1),
        (0, 2, 5, -1),
        (3, 1, 6, 1),
        (2, 0, 7, -1),
        (5, 7, 0, -1),
        (4, 6, 1, 1),
        (7, 5, 2, -1),
        (6, 4, 3, 1),
    ]
    c = points[hexahedra]
    out = []
    for corner, (a, b, d, sign) in enumerate(frames):
        eu, ev, ew = (c[:, k] - c[:, corner] for k in (a, b, d))
        out.append(sign * np.einsum("ij,ij->i", np.cross(eu, ev), ew))
    return np.stack(out, axis=1)


def _second_order(errors):
    """Each refinement's ratio of successive errors; second order gives about 4."""
    errors = np.abs(np.asarray(errors))
    return errors[:-1] / errors[1:]


# --- a single block ------------------------------------------------------------------------------


def test_a_flat_block_reproduces_the_tensor_grid_of_its_fractions():
    fu = np.array([0.0, 0.1, 0.35, 1.0])
    fw = np.array([0.0, 0.5, 0.8, 0.95, 1.0])
    points = _box((3.0, 1.0, 2.0), (fu, 2, fw), origin=(1.0, -1.0, 0.5)).points()
    grid = np.meshgrid(1.0 + 3.0 * fu, [-1.0, -0.5, 0.0], 0.5 + 2.0 * fw, indexing="ij")
    np.testing.assert_allclose(points, np.stack(grid, axis=-1), rtol=0, atol=1e-15)


def test_a_graded_annular_sector_is_the_exact_polar_grid():
    # Between radii 1 and 2 over a third of a quarter turn, extruded along z: the interpolation
    # weighted by the node fractions gives r(fraction) along each ray, exactly.
    fr, theta = np.array([0.0, 0.05, 0.15, 0.4, 1.0]), math.pi / 6
    ray = np.array([math.cos(theta), math.sin(theta), 0.0])
    ring = [[1, 0, 0], [2, 0, 0], 2 * ray, ray]
    corners = np.array([[x, y, z] for z in (0.0, 0.5) for x, y, _ in ring])
    edges = {
        e: Arc(np.array([0, 0, corners[e[0]][2]]), corners[e[0]], corners[e[1]])
        for e in [(0, 3), (1, 2), (4, 7), (5, 6)]
    }
    points = HexBlock(corners, (fr, 3, 2), edges=edges).points()
    angle = np.linspace(0.0, theta, 4)
    radius = 1.0 + fr
    expected = np.stack(
        np.broadcast_arrays(
            (radius[:, None] * np.cos(angle))[:, :, None],
            (radius[:, None] * np.sin(angle))[:, :, None],
            np.array([0.0, 0.25, 0.5])[None, None, :],
        ),
        axis=-1,
    )
    np.testing.assert_allclose(points, expected, rtol=0, atol=1e-14)


def test_a_uniform_box_block_gives_the_structured_grids_points():
    nx, ny, nz = 3, 2, 4
    points = _box((3.0, 1.0, 2.0), (nx, ny, nz)).points()
    grid = structured_grid_3d(nx, ny, nz, 3.0, 1.0, 2.0)
    # The structured grid numbers its nodes with x fastest; the block indexes them (i, j, k).
    np.testing.assert_allclose(
        points.transpose(2, 1, 0, 3).reshape(-1, 3), np.asarray(grid.node_coords), atol=1e-15
    )


def test_a_curved_edge_bends_the_block_and_its_opposite_edge_stays_straight():
    # A box whose u-edge from corner 3 to 2 bows up into a circular arc of the unit circle.
    corners = UNIT * [2.0, 1.0, 1.0] + [-1.0, 0.0, 0.0]
    corners[[2, 3, 6, 7], 1] = 1.0
    # (-1, 1, 0) and (1, 1, 0) are sqrt(2) from the origin, so the arc bows out to y = sqrt(2).
    edges = {
        (3, 2): Arc(np.zeros(3), corners[3], corners[2]),
        (7, 6): Arc(np.array([0.0, 0.0, 1.0]), corners[7], corners[6]),
    }
    block = HexBlock(corners, (4, 2, 1), edges=edges)
    points = block.points()
    np.testing.assert_allclose(np.hypot(points[:, -1, 0, 0], points[:, -1, 0, 1]), math.sqrt(2))
    assert points[2, -1, 0, 1] == pytest.approx(math.sqrt(2))  # the arc's middle, on the y axis
    np.testing.assert_allclose(points[:, 0, 0, 1], 0.0, atol=1e-15)  # the opposite edge, straight


def test_an_edge_given_from_its_far_corner_is_the_same_edge():
    corners = UNIT * [2.0, 1.0, 1.0]
    bow = SampledCurve(np.array([[0.0, 1.0, 0.0], [0.5, 1.5, 0.0], [2.0, 1.0, 0.0]]))
    reverse = SampledCurve(bow.points[::-1])
    one = HexBlock(corners, (np.array([0.0, 0.2, 1.0]), 2, 1), edges={(3, 2): bow}).points()
    other = HexBlock(corners, (np.array([0.0, 0.2, 1.0]), 2, 1), edges={(2, 3): reverse}).points()
    np.testing.assert_allclose(one, other, atol=1e-15)
    # A fifth of the way along the polyline's length falls on its shorter first leg.
    first, second = np.array([0.5, 0.5, 0.0]), np.array([1.5, -0.5, 0.0])
    along = 0.2 * (np.linalg.norm(first) + np.linalg.norm(second)) / np.linalg.norm(first)
    np.testing.assert_allclose(one[1, -1, 0], [0.0, 1.0, 0.0] + along * first, atol=1e-15)


def _bump(s, t):
    """A surface over the unit square at height 1, bulging up inside its edges."""
    return np.stack(np.broadcast_arrays(s, t, 1.0 + s * (1 - s) * t * (1 - t)), axis=-1)


@pytest.mark.parametrize("given", ["surface", "nodes"])
def test_a_face_given_as_a_surface_is_the_blocks_face_and_stretches_its_inside(given):
    fu, fv, fw = np.array([0.0, 0.3, 1.0]), np.linspace(0.0, 1.0, 4), np.array([0.0, 0.4, 1.0])
    s, t = np.meshgrid(fu, fv, indexing="ij")
    face = _bump if given == "surface" else _bump(s, t)
    points = HexBlock(UNIT, (fu, fv, fw), faces={"w+": face}).points()
    np.testing.assert_allclose(points[:, :, -1], _bump(s, t), atol=1e-15)
    # With flat sides, each column of nodes is the top's height scaled by the w fraction.
    np.testing.assert_allclose(points[..., 2], fw * _bump(s, t)[..., 2][..., None], atol=1e-15)


def test_a_face_disagreeing_with_a_given_edge_is_refused():
    lifted = SampledCurve(np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 1.2], [1.0, 0.0, 1.0]]))
    with pytest.raises(ValueError, match="edge \\(4, 5\\) and face 'w\\+''s edges disagree"):
        HexBlock(UNIT, (2, 2, 2), edges={(4, 5): lifted}, faces={"w+": _bump}).points()


def test_a_projected_face_lies_on_its_sphere_and_the_unprojected_one_does_not():
    # A block between spheres of radius 1 and 2 over a cube face's patch, its edges great circles.
    cube = np.array([[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]], dtype=float) / math.sqrt(3)
    corners = np.array(
        [cube[0], 2 * cube[0], 2 * cube[1], cube[1], cube[3], 2 * cube[3], 2 * cube[2], cube[2]]
    )
    on_spheres = [(0, 3), (4, 7), (0, 4), (3, 7), (1, 2), (5, 6), (1, 5), (2, 6)]
    edges = {e: Arc(np.zeros(3), corners[e[0]], corners[e[1]]) for e in on_spheres}
    points = HexBlock(
        corners, (2, 4, 4), edges=edges, projections={"u-": SphereSurface(1.0)}
    ).points()
    np.testing.assert_allclose(np.linalg.norm(points[0], axis=-1), 1.0, rtol=1e-14)
    # The outer face is the interpolation of its edges alone, which sags inside the sphere.
    assert np.max(2.0 - np.linalg.norm(points[-1], axis=-1)) > 0.1


def test_a_left_handed_block_is_refused():
    with pytest.raises(ValueError, match="are not right-handed"):
        HexBlock(UNIT[[0, 3, 2, 1, 4, 7, 6, 5]], (1, 1, 1)).points()


def test_an_edge_not_joining_its_corners_is_refused():
    off = Arc(np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match=r"edge \(0, 1\) and its corners disagree"):
        HexBlock(UNIT, (2, 2, 2), edges={(0, 1): off}).points()


# --- cell kinds ----------------------------------------------------------------------------------


def test_hexahedra_make_the_structured_grids_mesh():
    blocks = BlockMesh.assemble([_box((3.0, 1.0, 2.0), (3, 2, 4))])
    mesh = element_mesh(blocks.points, hexahedra=blocks.hexahedra)
    grid = structured_grid_3d(3, 2, 4, 3.0, 1.0, 2.0)
    assert (mesh.n_cells, mesh.n_faces) == (grid.n_cells, grid.n_faces)
    assert int(np.count_nonzero(np.asarray(mesh.face_cells.neighbour) < 0)) == 2 * (6 + 12 + 8)
    np.testing.assert_allclose(np.asarray(mesh.geometry().cell.volume), 0.25, rtol=1e-13)


def test_six_pyramids_on_a_hexahedron_make_a_closed_mesh():
    # A unit cube cut into six pyramids from its centre, then a hexahedron set on its top face.
    centre = len(UNIT)
    top = UNIT + np.array([0.0, 0.0, 1.0])
    points = np.vstack([UNIT, [[0.5, 0.5, 0.5]], top[4:]])
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    pyramids = np.array([[*face, centre] for face in faces])
    hexahedra = np.array([[4, 5, 6, 7, 9, 10, 11, 12]])
    mesh = element_mesh(points, hexahedra=hexahedra, pyramids=pyramids)
    # Six bases (one shared with the hexahedron), twelve triangles inside, five hexahedron faces.
    assert (mesh.n_cells, mesh.n_faces) == (7, 6 + 12 + 5)
    neighbour = np.asarray(mesh.face_cells.neighbour)
    assert int(np.count_nonzero(neighbour >= 0)) == 12 + 1
    np.testing.assert_allclose(
        np.asarray(mesh.geometry().cell.volume), [1.0] + [1 / 6] * 6, rtol=1e-13
    )
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-14


# --- joining blocks ------------------------------------------------------------------------------


def test_two_blocks_share_their_common_face_once():
    left = _box(
        (1.0, 1.0, 1.0), (2, 3, 4), patches={f: "walls" for f in ("u-", "v-", "v+", "w-", "w+")}
    )
    right = _box(
        (1.0, 1.0, 1.0),
        (5, 3, 4),
        origin=(1.0, 0.0, 0.0),
        patches={f: "walls" for f in ("u+", "v-", "v+", "w-", "w+")},
    )
    joined = BlockMesh.assemble([left, right])
    assert joined.points.shape[0] == 3 * 4 * 5 + 6 * 4 * 5 - 4 * 5
    assert joined.hexahedra.shape[0] == 2 * 3 * 4 + 5 * 3 * 4
    # No quadrilateral of the common face is on the boundary.
    assert not np.any((joined.boundary_block == 0) & (joined.boundary_face == 1))
    mesh = joined.mesh()
    np.testing.assert_allclose(np.sum(np.asarray(mesh.geometry().cell.volume)), 2.0, rtol=1e-13)
    assert mesh.n_faces - int(np.count_nonzero(np.asarray(mesh.face_cells.neighbour) < 0)) == (
        # Interior faces: the two blocks' own, and the 3 x 4 of the common face.
        (1 * 3 * 4 + 2 * 2 * 4 + 2 * 3 * 3) + (4 * 3 * 4 + 5 * 2 * 4 + 5 * 3 * 3) + 3 * 4
    )


def test_a_face_cut_differently_on_its_two_sides_is_refused():
    left = _box((1.0, 1.0, 1.0), (2, 2, 2))
    right = _box((1.0, 1.0, 1.0), (2, 4, 2), origin=(1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match=r"\[\(0, 'u\+'\), \(1, 'u-'\)\] .* only part"):
        BlockMesh.assemble([left, right])


def test_a_boundary_face_in_no_patch_is_refused():
    joined = BlockMesh.assemble([_box((1.0, 1.0, 1.0), (1, 1, 1), patches={"u-": "inlet"})])
    with pytest.raises(ValueError, match=r"\(0, 'u\+'\).* in no patch"):
        joined.mesh()


# --- the round layouts ---------------------------------------------------------------------------


def _layout_checks(blocks, exact_volume, surfaces):
    """Right-handed cells, boundary nodes on their surfaces; the mesh's and its dual's volumes."""
    joined = BlockMesh.assemble(blocks)
    assert np.all(_corner_jacobians(joined.points, joined.hexahedra) > 0.0)
    patches = joined.patches()
    for name, distance in surfaces.items():
        np.testing.assert_allclose(
            distance(joined.points[np.unique(patches[name])]), 0.0, atol=1e-14
        )
    mesh = joined.mesh()
    dual = DUAL.build(mesh)
    assert dual.n_cells == joined.points.shape[0]
    assert float(np.max(np.asarray(closed_cell_residual(dual)))) < 1e-12
    assert np.all(np.asarray(dual.geometry().cell.volume) > 0.0)
    return (
        float(np.sum(np.asarray(mesh.geometry().cell.volume))) - exact_volume,
        float(np.sum(np.asarray(dual.geometry().cell.volume))) - exact_volume,
    )


def _off_cylinder(radius):
    return lambda p: np.hypot(p[:, 0], p[:, 1]) - radius


def _off_sphere(radius, centre=(0.0, 0.0, 0.0)):
    return lambda p: np.linalg.norm(p - np.asarray(centre), axis=1) - radius


def test_a_pipe_o_grid_fills_its_cylinder_at_second_order_and_so_does_its_dual():
    radius, length = 0.3, 1.2
    errors = [
        _layout_checks(
            PipeOGrid(radius, [-0.4, length - 0.4], n, n // 2, n // 2).blocks(),
            math.pi * radius**2 * length,
            {"wall": _off_cylinder(radius)},
        )
        for n in (4, 8, 16)
    ]
    for ratios in (_second_order([e[0] for e in errors]), _second_order([e[1] for e in errors])):
        assert np.all((ratios > 3.6) & (ratios < 4.4)), ratios


def test_a_pipe_o_grids_patches_are_its_wall_and_ends():
    n, radial, axial = 4, np.array([0.0, 0.6, 1.0]), 3
    grid = PipeOGrid(1.0, [1.0, 3.0], n, radial, axial)
    patches = BlockMesh.assemble(grid.blocks()).patches()
    assert {k: v.shape[0] for k, v in patches.items()} == {
        "inlet": n * n + 4 * n * 2,
        "outlet": n * n + 4 * n * 2,
        "wall": 4 * n * axial,
    }
    points = BlockMesh.assemble(grid.blocks()).points
    np.testing.assert_allclose(points[np.unique(patches["inlet"])][:, 2], 1.0)
    np.testing.assert_allclose(points[np.unique(patches["outlet"])][:, 2], 3.0)


def test_an_annulus_fills_the_space_between_its_cylinders_at_second_order():
    inner, outer, length = 0.2, 0.5, 0.8
    errors = [
        _layout_checks(
            Annulus([inner, outer], [0.0, length], n, n // 2, 2, azimuth=0.3).blocks(),
            math.pi * (outer**2 - inner**2) * length,
            {"inner": _off_cylinder(inner), "outer": _off_cylinder(outer)},
        )
        for n in (4, 8, 16)
    ]
    for ratios in (_second_order([e[0] for e in errors]), _second_order([e[1] for e in errors])):
        assert np.all((ratios > 3.6) & (ratios < 4.4)), ratios


def test_a_cubed_sphere_cap_fills_its_hemispherical_shell_at_second_order():
    inner, outer, base = 0.2, 0.5, 0.7
    errors = [
        _layout_checks(
            CubedSphereCap(inner, outer, n, n // 2, 2, base=base).blocks(),
            2 * math.pi / 3 * (outer**3 - inner**3),
            {"inner": _off_sphere(inner, (0, 0, base)), "outer": _off_sphere(outer, (0, 0, base))},
        )
        for n in (4, 8, 16)
    ]
    for ratios in (_second_order([e[0] for e in errors]), _second_order([e[1] for e in errors])):
        assert np.all((ratios > 3.6) & (ratios < 4.4)), ratios


def test_a_cap_joins_an_annulus_turned_a_quarter_of_a_quarter_turn():
    n, radial, inner, outer = 4, 3, 0.2, 0.5
    cap = CubedSphereCap(inner, outer, n, 2, radial).blocks()
    sleeve = Annulus([inner, outer], [-1.0, 0.0], n, radial, 2, azimuth=math.pi / 4).blocks()
    alone = [BlockMesh.assemble(blocks).points.shape[0] for blocks in (cap, sleeve)]
    joined = BlockMesh.assemble(cap + sleeve)
    # The rim's points are given once: 4 n around, radial + 1 out.
    assert joined.points.shape[0] == sum(alone) - 4 * n * (radial + 1)
    assert set(joined.patches()) == {"inner", "outer", "start"}
    volume = 2 * math.pi / 3 * (outer**3 - inner**3) + math.pi * (outer**2 - inner**2)
    dual = DUAL.build(joined.mesh())
    assert float(np.max(np.asarray(closed_cell_residual(dual)))) < 1e-12
    assert np.sum(np.asarray(dual.geometry().cell.volume)) == pytest.approx(volume, rel=0.1)
