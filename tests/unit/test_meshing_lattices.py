"""Block lattices, divisions, projections, placement and the general round layouts.

Each check has an answer worked out independently: a graded division's cell sizes form the
geometric progression asked for; a projection's image lies on its surface with the displacement
normal to it; a placed block is the original's points moved; a lattice's nodes are the tensor product
of its per-interval fractions, its masked blocks are absent, and its curved faces lie on the
cylinders and cones its corners name; an O-grid's shells, bow and azimuth put nodes where their
definitions say; and the volumes converge at second order to the exact solid's.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from aquaflux.io.cad import Placement
from aquaflux.mesh import closed_cell_residual
from aquaflux.meshing import (
    Annulus,
    Arc,
    BlockLattice,
    BlockMesh,
    CartesianFrame,
    ConeSurface,
    CylindricalFrame,
    Graded,
    HexBlock,
    PipeOGrid,
    PlaneSurface,
    PolyhedralDual,
    SphereSurface,
    Straight,
    cell_counts,
)

DUAL = PolyhedralDual(feature_angle=math.radians(45))
RNG = np.random.default_rng(7)


def _ratios(errors):
    errors = np.abs(np.asarray(errors))
    return errors[:-1] / errors[1:]


# --- divisions -----------------------------------------------------------------------------------


def test_a_graded_division_grows_geometrically_from_the_end_it_names():
    sizes = np.diff(Graded(4, 1.5).fractions())
    np.testing.assert_allclose(sizes[1:] / sizes[:-1], 1.5)
    np.testing.assert_allclose(sizes, np.array([1, 1.5, 2.25, 3.375]) / 8.125)
    np.testing.assert_allclose(np.diff(Graded(4, 1.5, "end").fractions()), sizes[::-1])
    both = np.diff(Graded(6, 2.0, "both").fractions())
    np.testing.assert_allclose(both, np.array([1, 2, 4, 4, 2, 1]) / 14)


def test_scaling_a_graded_division_keeps_the_ratio_of_its_end_cells():
    fine = Graded(4, 1.5).scaled(2.5)
    assert fine.cells == 10
    sizes = np.diff(fine.fractions())
    assert sizes[-1] / sizes[0] == pytest.approx(1.5**3)


def test_explicit_fractions_scale_only_by_a_whole_factor_each_cell_split_evenly():
    block = HexBlock(_UNIT, (np.array([0.0, 0.2, 1.0]), 1, 1))
    np.testing.assert_allclose(block.scaled_counts(2).fractions[0], [0.0, 0.1, 0.2, 0.6, 1.0])
    with pytest.raises(ValueError, match="whole factor"):
        block.scaled_counts(1.5)


def test_cell_counts_follow_a_target_size_at_least_one_each():
    assert cell_counts([0.0, 1.0, 3.6, 3.65], 0.5) == [2, 5, 1]


# --- projections ---------------------------------------------------------------------------------


def test_a_plane_projection_moves_points_along_its_normal_onto_it():
    plane = PlaneSurface(np.array([1.0, 2.0, 0.5]), np.array([1.0, -2.0, 2.0]))
    points = RNG.normal(size=(20, 3))
    moved = plane(points)
    normal = np.array([1.0, -2.0, 2.0]) / 3.0
    np.testing.assert_allclose((moved - [1.0, 2.0, 0.5]) @ normal, 0.0, atol=1e-14)
    np.testing.assert_allclose(np.cross(points - moved, normal), 0.0, atol=1e-14)


def test_a_cone_projection_lands_on_the_cone_at_the_nearest_point():
    first, second = np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 2.0])
    cone = ConeSurface(first, second, 1.0, 0.5)
    points = RNG.normal(size=(20, 3)) + np.array([0.0, 1.0, 1.0])
    moved = cone(points)
    height = moved[:, 2]
    radius = np.hypot(moved[:, 0], moved[:, 1] - 1.0)
    np.testing.assert_allclose(radius, 1.0 - 0.25 * height, atol=1e-14)
    # Nearest: the displacement is normal to the cone's generator through the image.
    outward = np.stack([moved[:, 0], moved[:, 1] - 1.0, 0 * height], axis=1) / radius[:, None]
    generator = np.array([0.0, 0.0, 2.0]) + (-0.5) * outward
    np.testing.assert_allclose(np.einsum("ij,ij->i", points - moved, generator), 0.0, atol=1e-13)


# --- placement -----------------------------------------------------------------------------------

_UNIT = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=float,
)


def _sphere_block():
    """A block between spheres of radius 1 and 2 over a cube face, its edges great circles."""
    cube = np.array([[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]], dtype=float) / math.sqrt(3)
    corners = np.array(
        [cube[0], 2 * cube[0], 2 * cube[1], cube[1], cube[3], 2 * cube[3], 2 * cube[2], cube[2]]
    )
    on_spheres = [(0, 3), (4, 7), (0, 4), (3, 7), (1, 2), (5, 6), (1, 5), (2, 6)]
    return HexBlock(
        corners,
        (2, 3, Graded(4, 1.4)),
        edges={e: Arc(np.zeros(3), corners[e[0]], corners[e[1]]) for e in on_spheres},
        projections={"u-": SphereSurface(1.0), "u+": SphereSurface(2.0)},
        patches={"w-": "bottom", "u+": "outer"},
    )


def test_a_placed_block_is_the_original_moved():
    turn = math.radians(30)
    matrix = [[math.cos(turn), -math.sin(turn), 0], [math.sin(turn), math.cos(turn), 0], [0, 0, 1]]
    placement = Placement(matrix=matrix, offset=[1.0, -2.0, 3.0])
    block = _sphere_block()
    expected = block.points() @ np.asarray(matrix).T + [1.0, -2.0, 3.0]
    np.testing.assert_allclose(block.placed(placement).points(), expected, atol=1e-14)


def test_a_reflected_block_reads_w_the_other_way_and_keeps_its_faces_on_their_spheres():
    swap = Placement(matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]], offset=[0.0, 0.0, 5.0])
    block = _sphere_block()
    placed = block.placed(swap)  # refused if it were left-handed
    moved = block.points() @ swap.matrix.T + swap.offset
    np.testing.assert_allclose(placed.points(), moved[:, :, ::-1], atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(placed.points()[0] - [0, 0, 5], axis=-1), 1.0)
    assert placed.patches == {"w+": "bottom", "u+": "outer"}


# --- lattices ------------------------------------------------------------------------------------


def test_a_cartesian_lattice_is_the_tensor_product_of_its_intervals_fractions():
    lattice = BlockLattice(
        CartesianFrame(), ([0.0, 1.0, 3.0], [0.0, 1.0], [0.0, 2.0]), ([2, Graded(3, 2.0)], 1, 1)
    )
    joined = BlockMesh.assemble(lattice.blocks())
    xs = np.unique(np.round(joined.points[:, 0], 12))
    # Sizes 1, 2, 4 over 7 of a length 2 from x = 1.
    np.testing.assert_allclose(xs, [0.0, 0.5, 1.0, 1 + 2 / 7, 1 + 6 / 7, 3.0], atol=1e-12)
    assert joined.hexahedra.shape[0] == 5


def test_a_masked_block_is_left_out_and_its_faces_become_boundary():
    lattice = BlockLattice(CartesianFrame(), ([0, 1, 2], [0, 1, 2], [0, 1]), (2, 2, 1))
    lattice.mask[1, 1, 0] = True
    lattice.patches[1, 1, 0, 0] = "notch"  # the face x = 1 beside the missing block
    lattice.patches[1, 1, 0, 1] = "notch"
    joined = BlockMesh.assemble(lattice.blocks())
    mesh = joined.mesh(default_patch="walls")
    assert mesh.n_cells == 3 * 2 * 2
    np.testing.assert_allclose(np.sum(np.asarray(mesh.geometry().cell.volume)), 3.0, rtol=1e-13)
    assert joined.patches("walls")["notch"].shape[0] == 2 * 2
    centroids = np.asarray(mesh.geometry().cell.centroid)
    assert not np.any((centroids[:, 0] > 1) & (centroids[:, 1] > 1))


def _twisted_shell(n, twist, top_radii):
    lattice = BlockLattice(
        CylindricalFrame(),
        ([0.5, 1.0], np.linspace(0.0, 2.0 * math.pi, 5), [0.0, 1.0]),
        (n // 2, n, n),
    )
    lattice.corners[:, :, 1, 1] += twist
    lattice.corners[:, :, 1, 0] = np.asarray(top_radii)[:, None]
    lattice.patches[0, :, :, 0], lattice.patches[-1, :, :, 0] = "inner", "outer"
    return BlockMesh.assemble(lattice.blocks())


def test_a_twisted_cylindrical_lattice_keeps_its_faces_on_their_cylinders():
    joined = _twisted_shell(6, 0.3, [0.5, 1.0])
    patches = joined.patches("ends")
    for name, radius in (("inner", 0.5), ("outer", 1.0)):
        on = joined.points[np.unique(patches[name])]
        np.testing.assert_allclose(np.hypot(on[:, 0], on[:, 1]), radius, rtol=1e-14)


def test_a_cylindrical_lattice_narrowing_along_z_puts_its_faces_on_cones():
    joined = _twisted_shell(6, 0.3, [0.4, 0.7])
    patches = joined.patches("ends")
    for name, (bottom, top) in (("inner", (0.5, 0.4)), ("outer", (1.0, 0.7))):
        on = joined.points[np.unique(patches[name])]
        expected = bottom + (top - bottom) * on[:, 2]
        np.testing.assert_allclose(np.hypot(on[:, 0], on[:, 1]), expected, rtol=1e-13)


def test_a_frustum_shell_fills_its_volume_at_second_order_and_so_does_its_dual():
    exact = math.pi / 3 * (1.0 + 0.7 + 0.49) - math.pi / 3 * (0.25 + 0.2 + 0.16)
    errors = []
    for n in (4, 8, 16):
        mesh = _twisted_shell(n, 0.0, [0.4, 0.7]).mesh(default_patch="ends")
        dual = DUAL.build(mesh)
        assert float(np.max(np.asarray(closed_cell_residual(dual)))) < 1e-12
        errors.append([np.sum(np.asarray(m.geometry().cell.volume)) - exact for m in (mesh, dual)])
    for ratios in _ratios(np.asarray(errors)).T:
        assert np.all((ratios > 3.6) & (ratios < 4.4)), ratios


def test_a_cylindrical_lattice_refuses_half_turns_and_the_axis():
    half = BlockLattice(CylindricalFrame(), ([0.5, 1.0], [0.0, math.pi], [0.0, 1.0]), (1, 1, 1))
    with pytest.raises(ValueError, match=r"turns 180\.0 degrees"):
        half.blocks()
    axis = BlockLattice(CylindricalFrame(), ([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]), (1, 1, 1))
    with pytest.raises(ValueError, match="radii must be positive"):
        axis.blocks()


def test_a_list_of_divisions_must_have_one_per_interval():
    with pytest.raises(ValueError, match="2 divisions given along axis 0, which has 1"):
        BlockLattice(CartesianFrame(), ([0, 1], [0, 1], [0, 1]), ([1, 2], 1, 1))


def test_a_straight_override_replaces_a_cylindrical_lattices_arc():
    lattice = BlockLattice(
        CylindricalFrame(), ([0.5, 1.0], [0.0, 0.5 * math.pi], [0.0, 1.0]), (1, 4, 1)
    )
    lattice.edges[0, 0, :, 1] = Straight()
    lattice.projections[0, 0, 0, 0] = False
    points = lattice.blocks()[0].points()
    inner = points[0, :, 0]
    np.testing.assert_allclose(inner[:, 0] + inner[:, 1], 0.5, atol=1e-15)  # the chord x + y = 0.5


# --- the general round layouts ------------------------------------------------------------------


def test_a_pipe_with_shells_side_blocks_and_segments_puts_each_shell_on_its_cylinder():
    grid = PipeOGrid(
        1.0, [0.0, 0.5, 2.0], 3, [2, Graded(3, 1.3, "end")], [2, 4], shells=[0.8], side_blocks=3
    )
    blocks = grid.blocks()
    joined = BlockMesh.assemble(blocks)
    radius = np.hypot(joined.points[:, 0], joined.points[:, 1])
    # On the middle shell and the wall: 4 sides x 3 blocks x 3 cells around, at 2 + 4 + 1 heights.
    assert np.count_nonzero(np.isclose(radius, 0.8, rtol=0, atol=1e-13)) == 36 * 7
    assert np.count_nonzero(np.isclose(radius, 1.0, rtol=0, atol=1e-13)) == 36 * 7
    patches = joined.patches()
    assert patches["wall"].shape[0] == 36 * 6
    assert patches["inlet"].shape[0] == 9 * 9 + 36 * 5
    # The first shell's lines out from the core are straight, though their ends are at different
    # angles (the core's side nodes are evenly spaced along it, the shell's around it).
    first_shell = blocks[9 * 2 : 9 * 2 + 12 * 2]
    for block in first_shell:
        line = block.points()[:, 0, 0]
        np.testing.assert_allclose(np.cross(line - line[0], line[-1] - line[0]), 0.0, atol=1e-14)


def test_a_bowed_core_moves_each_sides_middle_out_by_its_fraction():
    a, bow = 0.5, 0.6
    joined = BlockMesh.assemble(PipeOGrid(1.0, [0.0, 1.0], 4, 2, 1, core_bow=bow).blocks())
    points = joined.points[joined.points[:, 2] == 0.0]
    middle = a + bow * (math.sqrt(2) - 1) * a
    for x, y in ((middle, 0.0), (0.0, middle), (-middle, 0.0), (0.0, -middle)):
        assert np.min(np.hypot(points[:, 0] - x, points[:, 1] - y)) < 1e-14
    assert np.min(np.hypot(points[:, 0] - a, points[:, 1] - a)) < 1e-14  # corners stay


def test_a_bowed_pipe_fills_its_cylinder_at_second_order_and_so_does_its_dual():
    errors = []
    for n in (4, 8, 16):
        mesh = BlockMesh.assemble(
            PipeOGrid(0.3, [0.0, 1.0], n, n // 2, n // 2, core_bow=0.5, side_blocks=2).blocks()
        ).mesh()
        dual = DUAL.build(mesh)
        exact = math.pi * 0.09
        errors.append([np.sum(np.asarray(m.geometry().cell.volume)) - exact for m in (mesh, dual)])
    for ratios in _ratios(np.asarray(errors)).T:
        assert np.all((ratios > 3.6) & (ratios < 4.4)), ratios


def test_a_pipes_azimuth_turns_its_core_corners_and_wall_nodes():
    joined = BlockMesh.assemble(PipeOGrid(1.0, [0.0, 1.0], 2, 1, 1, azimuth=0.0).blocks())
    at = joined.points[:, :2]
    assert np.min(np.hypot(at[:, 0] - 0.5 * math.sqrt(2), at[:, 1])) < 1e-14  # a core corner on x
    assert np.min(np.hypot(at[:, 0] - 1.0, at[:, 1])) < 1e-14  # a wall corner on x


def test_an_annulus_of_six_blocks_and_two_shells_fills_its_volume_at_second_order():
    errors = []
    for n in (2, 4, 8):
        mesh = BlockMesh.assemble(
            Annulus(
                [0.3, 0.6, 1.0], [0.0, 1.0], n, [n // 2, Graded(n, 1.2, "both")], 2, around=6
            ).blocks()
        ).mesh(patch_types={"inner": "wall", "outer": "wall"})
        errors.append(np.sum(np.asarray(mesh.geometry().cell.volume)) - math.pi * 0.91)
        assert mesh.n_cells == 6 * n * (n // 2 + n) * 2
    assert dict(mesh.face_patches.patch_types) == {"inner": "wall", "outer": "wall"}
    ratios = _ratios(errors)
    assert np.all((ratios > 3.6) & (ratios < 4.4)), ratios


def test_a_count_factor_scales_every_blocks_cells():
    grid = PipeOGrid(1.0, [0.0, 1.0], 2, Graded(2, 1.5, "end"), 3)
    coarse, fine = (BlockMesh.assemble(grid.blocks(), count_factor=f) for f in (1, 2))
    assert fine.hexahedra.shape[0] == 8 * coarse.hexahedra.shape[0]
