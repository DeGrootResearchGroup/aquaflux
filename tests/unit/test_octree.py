"""The octree, its hierarchical interpolant, the boundary's cut points and the adaptive refinement."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import numpy as np
import pytest
from aquaflux.octree import (
    Hierarchy,
    Linear,
    Logarithm,
    Octree,
    crossing_points,
    exit_points,
    ghost_values,
    leaf_cuts,
    merged_intervals,
    sample_adaptively,
)
from aquaflux.octree.tree import CORNERS, EDGES
from aquaflux.solids import Box, Cylinder, Difference, Sphere


def _split_twice() -> Octree:
    """Two root cells, the first split, one of its children split again: three levels side by side."""
    tree = Octree.covering([0.0, 0.0, 0.0], [2.0, 1.0, 1.0], 1.0, 3)
    tree, children = tree.split(np.array([True, False]))
    mask = np.zeros(tree.n_leaves, dtype=bool)
    mask[children[0, 7]] = True  # the child touching the unsplit root cell's face and the centre
    tree, _ = tree.split(mask)
    return tree


def _all_nodes(tree: Octree) -> np.ndarray:
    return np.unique(tree.key(tree.corners()))


# -- the tree --------------------------------------------------------------------------------------


def test_edges_join_corners_that_differ_along_one_axis():
    assert EDGES.shape == (12, 2)
    difference = CORNERS[EDGES[:, 1]] - CORNERS[EDGES[:, 0]]
    assert np.all(np.abs(difference).sum(axis=1) == 1)
    assert np.all(difference >= 0)
    assert len({tuple(e) for e in EDGES}) == 12


def test_root_cells_cover_the_box_with_whole_cells():
    tree = Octree.covering([0.0, 0.0, 0.0], [0.25, 0.1, 0.1], 0.1, 2)
    np.testing.assert_array_equal(tree.shape, [3, 1, 1])
    assert tree.n_leaves == 3
    assert tree.step == pytest.approx(0.025)
    np.testing.assert_array_equal(tree.low[:, 0], [0, 4, 8])


def test_a_key_names_one_lattice_point_and_back():
    tree = Octree.covering([0.0, 0.0, 0.0], [3.0, 2.0, 5.0], 1.0, 2)
    lattice = np.stack(np.meshgrid(*(np.arange(n) for n in tree.lattice_shape), indexing="ij"), -1)
    lattice = lattice.reshape(-1, 3)
    keys = tree.key(lattice)
    assert len(np.unique(keys)) == len(keys)
    np.testing.assert_array_equal(tree.unkey(keys), lattice)


def test_a_node_level_is_the_coarsest_lattice_it_lies_on():
    tree = Octree.covering([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], 1.0, 3)
    lattice = np.array(
        [[0, 0, 0], [8, 0, 8], [4, 0, 0], [8, 4, 0], [2, 6, 0], [1, 0, 0], [8, 8, 3]]
    )
    np.testing.assert_array_equal(tree.node_level(lattice), [0, 0, 1, 1, 2, 3, 3])


def test_splitting_makes_eight_children_filling_the_parent():
    tree = Octree.covering([0.0, 0.0, 0.0], [2.0, 1.0, 1.0], 1.0, 2)
    split, children = tree.split(np.array([False, True]))
    assert split.n_leaves == 9
    np.testing.assert_array_equal(children, [np.arange(1, 9)])
    np.testing.assert_array_equal(split.levels, [0] + [1] * 8)
    np.testing.assert_array_equal(split.low[children[0]], tree.low[1] + 2 * CORNERS)
    with pytest.raises(ValueError, match="already at max_level"):
        Octree.covering([0.0] * 3, [1.0] * 3, 1.0, 0).split(np.array([True]))


def test_every_point_is_located_in_the_one_leaf_holding_it():
    tree = _split_twice()
    rng = np.random.default_rng(0)
    points = rng.uniform([0.0, 0.0, 0.0], [2.0, 1.0, 1.0], (4000, 3))
    leaf = tree.locate(points)
    assert np.all(leaf >= 0)
    u = tree.lattice(points)
    size = tree.size(tree.levels[leaf])[:, None]
    assert np.all((u >= tree.low[leaf]) & (u <= tree.low[leaf] + size))
    np.testing.assert_array_equal(
        tree.locate(np.array([[2.5, 0.5, 0.5], [-0.1, 0.5, 0.5]])), [-1, -1]
    )
    # Every level was reached, so no level's lookup can be skipped unnoticed.
    assert set(tree.levels[leaf]) == {0, 1, 2}


# -- the hierarchical interpolant ------------------------------------------------------------------


def _hierarchy_of(tree: Octree, function) -> Hierarchy:
    keys = _all_nodes(tree)
    return Hierarchy.empty(tree).with_values(tree, keys, function(tree.position(tree.unkey(keys))))


def test_the_interpolant_passes_through_every_node():
    tree = _split_twice()
    keys = _all_nodes(tree)
    values = np.random.default_rng(1).normal(size=len(keys))
    hierarchy = Hierarchy.empty(tree).with_values(tree, keys, values)
    np.testing.assert_allclose(hierarchy.at_lattice(tree, tree.unkey(keys)), values, atol=1e-12)


def test_nodes_added_in_any_order_or_batch_give_the_same_interpolant():
    tree = _split_twice()
    keys = _all_nodes(tree)
    values = np.random.default_rng(2).normal(size=len(keys))
    whole = Hierarchy.empty(tree).with_values(tree, keys, values)
    level = tree.node_level(tree.unkey(keys))
    staged = Hierarchy.empty(tree)
    for depth in (0, 1, 2):  # coarse first, as a refinement adds them
        rows = level == depth
        staged = staged.with_values(tree, keys[rows], values[rows])
    points = np.random.default_rng(3).uniform([0, 0, 0], [2, 1, 1], (500, 3))
    np.testing.assert_allclose(staged(tree, points), whole(tree, points), atol=1e-12)


def test_a_surplus_is_how_far_the_coarser_nodes_missed():
    tree = Octree.covering([0.0] * 3, [1.0] * 3, 1.0, 1)
    tree, _ = tree.split(np.array([True]))
    corner_keys = tree.key(2 * CORNERS)  # the root cell's corners
    coarse = Hierarchy.empty(tree).with_values(tree, corner_keys, np.arange(8.0))
    centre = tree.key(np.array([1, 1, 1]))
    fine = coarse.with_values(tree, [centre], [10.0])
    assert fine.surpluses[1][0] == pytest.approx(10.0 - np.mean(np.arange(8.0)))


def test_the_interpolant_is_continuous_where_a_coarse_cell_meets_finer_ones():
    """Across the face x = 1 between the unsplit root cell and the split one, at points on finer
    nodes' hats -- where a trilinear interpolant of each cell's own corners would jump."""
    tree = _split_twice()
    hierarchy = _hierarchy_of(
        tree, lambda p: np.sin(3 * p[:, 0]) * np.cos(2 * p[:, 1]) + p[:, 2] ** 2
    )
    rng = np.random.default_rng(4)
    on_face = np.column_stack([np.ones(300), rng.uniform(0, 1, (300, 2))])
    across = np.array([1e-9, 0.0, 0.0])
    left = hierarchy(tree, on_face - across)
    right = hierarchy(tree, on_face + across)
    np.testing.assert_allclose(left, right, atol=1e-7)
    # The finer nodes on the face do reach the coarse side: a coarse-corners-only interpolant differs.
    coarse = tree.locate(on_face + across)
    assert np.all(tree.levels[coarse] == 0)
    root = Hierarchy.empty(tree).with_values(
        tree,
        tree.key(tree.corners([coarse[0]])[0]),
        hierarchy.at_lattice(tree, tree.corners([coarse[0]])[0]),
    )
    assert np.max(np.abs(root(tree, on_face + across) - right)) > 1e-3


def test_a_linear_field_is_reproduced_everywhere_by_any_tree():
    tree = _split_twice()
    linear = lambda p: 2.0 + p @ np.array([1.5, -3.0, 0.25])  # noqa: E731
    hierarchy = _hierarchy_of(tree, linear)
    points = np.random.default_rng(5).uniform([0, 0, 0], [2, 1, 1], (1000, 3))
    np.testing.assert_allclose(hierarchy(tree, points), linear(points), atol=1e-12)
    # Every surplus but the root nodes' is zero: the coarse tree already predicts the finer nodes.
    assert all(np.allclose(s, 0.0, atol=1e-12) for s in hierarchy.surpluses[1:])


def test_a_node_given_twice_is_refused():
    tree = _split_twice()
    keys = _all_nodes(tree)
    hierarchy = Hierarchy.empty(tree).with_values(tree, keys[:3], np.zeros(3))
    with pytest.raises(ValueError, match="given twice"):
        hierarchy.with_values(tree, keys[[5, 5]], np.zeros(2))
    with pytest.raises(ValueError, match="already in the hierarchy"):
        hierarchy.with_values(tree, keys[:1], np.zeros(1))


# -- the boundary's cut points ---------------------------------------------------------------------


def test_intervals_are_merged_sorted_and_compacted():
    enter = np.array([[0.5, 0.1, np.inf, 0.2], [0.0, 2.0, 0.6, 3.0]])
    exit_ = np.array([[0.9, 0.3, -np.inf, 0.4], [0.5, 1.0, 0.7, 4.0]])
    merged_enter, merged_exit = merged_intervals(enter, exit_)
    np.testing.assert_allclose(merged_enter[0, :2], [0.1, 0.5])
    np.testing.assert_allclose(merged_exit[0, :2], [0.4, 0.9])
    assert np.all(merged_enter[0, 2:] == np.inf)
    np.testing.assert_allclose(merged_enter[1, :3], [0.0, 0.6, 3.0])
    np.testing.assert_allclose(merged_exit[1, :3], [0.5, 0.7, 4.0])  # (2, 1) is empty


def test_a_segment_cut_by_a_ball_gets_a_point_a_standoff_inside_each_crossing():
    ball = Sphere(centre=[0.0, 0.0, 0.0], radius=1.0)
    points, segment = crossing_points(
        ball, np.array([[-2.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-2.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 5.0, 0.0]]), standoff=0.01,
    )  # fmt: skip
    np.testing.assert_allclose(
        points[np.argsort(segment)], [[-0.99, 0, 0], [0.99, 0, 0]], atol=1e-12
    )
    np.testing.assert_array_equal(np.sort(segment), [0, 1])


def test_a_cut_point_is_a_standoff_deep_whatever_angle_the_edge_meets_the_boundary_at():
    ball = Sphere(centre=[0.0, 0.0, 0.0], radius=1.0)
    # Edges along x at heights 0, 0.6 and 0.95: the last meets the sphere at a grazing angle.
    heights = np.array([0.0, 0.6, 0.95])
    start = np.column_stack([np.full(3, -2.0), heights, np.zeros(3)])
    end = np.column_stack([np.zeros(3), heights, np.zeros(3)])
    points, segment = crossing_points(ball, start, end, standoff=0.01)
    np.testing.assert_array_equal(np.sort(segment), [0, 1, 2])
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 0.99, rtol=1e-9)


def test_a_region_never_a_standoff_deep_gets_no_cut_point():
    slab = Box(centre=[0.0, 0.0, 0.0], half_sizes=[0.008, 1.0, 1.0])
    points, _ = crossing_points(slab, np.array([[-1.0, 0, 0]]), np.array([[1.0, 0, 0]]), 0.01)
    assert len(points) == 0
    # Deep enough: one point a standoff inside each face.
    points, _ = crossing_points(slab, np.array([[-1.0, 0, 0]]), np.array([[1.0, 0, 0]]), 0.001)
    np.testing.assert_allclose(np.sort(points[:, 0]), [-0.007, 0.007], atol=1e-12)
    # An edge grazing a ball -- along a line 0.995 from its centre -- holds no point a 0.01 deep.
    ball = Sphere(centre=[0.0, 0.0, 0.0], radius=1.0)
    points, _ = crossing_points(
        ball, np.array([[-2.0, 0.995, 0]]), np.array([[2.0, 0.995, 0]]), 0.01
    )
    assert len(points) == 0


def test_a_segment_wholly_inside_or_outside_has_no_cut_point():
    ball = Sphere(centre=[0.0, 0.0, 0.0], radius=1.0)
    points, _ = crossing_points(
        ball, np.array([[-0.5, 0, 0], [2.0, 2.0, 0]]), np.array([[0.5, 0, 0], [3.0, 2.0, 0]]), 0.01
    )
    assert len(points) == 0


def test_a_cut_on_an_edge_two_leaves_share_is_one_point_listed_on_both():
    tree = Octree.covering([0.0, 0.0, 0.0], [2.0, 1.0, 1.0], 1.0, 1)
    # A ball centred on the shared face's top edge x = 1, z = 1, cutting it at y = 0.5 +- 0.3.
    ball = Sphere(centre=[1.0, 0.5, 1.0], radius=0.3)
    cuts = leaf_cuts(tree, np.array([0, 1]), ball, standoff=0.01)
    on_edge = np.flatnonzero(np.all(np.abs(cuts.points[:, [0, 2]] - 1.0) < 1e-12, axis=1))
    assert len(on_edge) == 2
    first = cuts.items[cuts.offsets[0] : cuts.offsets[1]]
    second = cuts.items[cuts.offsets[1] : cuts.offsets[2]]
    assert set(on_edge) <= set(first) and set(on_edge) <= set(second)
    np.testing.assert_allclose(np.sort(cuts.points[on_edge, 1]), [0.21, 0.79], atol=1e-12)


def test_a_crossing_reached_along_a_long_and_a_short_edge_is_one_point():
    tree = Octree.covering([0.0, 0.0, 0.0], [2.0, 1.0, 1.0], 1.0, 1)
    tree, _ = tree.split(np.array([True, False]))  # left cell split: its edge on x = 1 is halved
    ball = Sphere(
        centre=[1.0, 0.0, 0.0], radius=0.3
    )  # cuts the shared edge x = 1, y = 0 at z = 0.3
    cuts = leaf_cuts(tree, np.arange(tree.n_leaves), ball, standoff=0.01)
    on_line = np.all(np.abs(cuts.points[:, :2] - [1.0, 0.0]) < 1e-12, axis=1)
    assert on_line.sum() == 1


def test_an_exit_point_is_the_crossing_points_own_point_at_that_end():
    ball = Sphere(centre=[0.0, 0.0, 0.0], radius=1.0)
    inside = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.995, 0.0, 0.0], [0.0, 0.0, 0.0]])
    outside = np.array([[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [2.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    points, t, found = exit_points(ball, inside, outside, 0.01)
    np.testing.assert_array_equal(found, [True, True, False, False])  # a sliver; never leaves
    np.testing.assert_allclose(points[:2], [[0.99, 0, 0], [0, 0.99, 0]], atol=1e-12)
    np.testing.assert_allclose(t[:2], [0.495, 0.66], atol=1e-12)
    same, _ = crossing_points(ball, inside[:2], outside[:2], 0.01)
    np.testing.assert_allclose(np.sort(same, axis=0), np.sort(points[:2], axis=0), atol=1e-12)


# -- the ghost values ------------------------------------------------------------------------------


def _linear(offsets):
    return 2.0 + offsets @ np.array([0.5, -1.0, 0.25])


def test_samples_of_a_linear_field_give_its_value_as_the_ghost():
    """Two edges and a few neighbours, all from one linear field: the ghost is that field's value at
    the node, whichever equations decide it."""
    t = np.array([0.4, 0.7])
    axes = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # where each edge's sampled neighbour is
    cut = axes * (1.0 - t[:, None])
    neighbours = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 1.0]])
    data = np.concatenate([neighbours, cut])
    ghosts, informed = ghost_values(
        1,
        np.zeros(2, int),
        t,
        _linear(axes),
        _linear(cut),
        np.zeros(len(data), int),
        data,
        _linear(data),
    )
    assert informed[0]
    assert ghosts[0] == pytest.approx(_linear(np.zeros((1, 3)))[0], abs=1e-12)


def test_an_edge_bends_the_ghost_away_from_the_straight_line_guess():
    """A cut point above the line through the neighbours: the ghost follows the edge."""
    neighbours = np.array([[-1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [-1.0, 0.0, 1.0]])
    t = np.array([0.5])
    ghosts, _ = ghost_values(1, np.zeros(1, int), t, np.array([0.0]), np.array([1.0]),
                             np.zeros(4, int), neighbours, np.zeros(4))  # fmt: skip
    # Alone, the edge asks for 2 (0 at the neighbour, 1 halfway); the flat extrapolation asks for 0,
    # weighted 0.1**2 against the edge's 0.5**2.
    assert ghosts[0] == pytest.approx(0.5 * 1.0 / (0.5**2 + 0.1**2), rel=1e-12)
    assert ghosts[0] > 1.9


def test_a_ghost_cannot_run_away_from_the_samples_it_was_fitted_to():
    """Two samples a hair apart along one axis with a small difference: their slope, carried a whole
    cell, would be enormous; the ghost stays within their range widened by that range."""
    offsets = np.array([[-1.0, 0.0, 0.0], [-1.0 + 1e-6, 0.0, 0.0]])
    values = np.array([5.0, 5.001])
    ghosts, _ = ghost_values(1, np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0),
                             np.zeros(2, int), offsets, values)  # fmt: skip
    assert 5.0 - 0.001 <= ghosts[0] <= 5.001 + 0.001


def test_one_diagonal_sample_extrapolates_as_a_constant():
    ghosts, informed = ghost_values(1, np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0),
                                    np.zeros(1, int), np.array([[1.0, 1.0, 0.0]]), np.array([7.0]))  # fmt: skip
    assert informed[0]
    assert ghosts[0] == pytest.approx(7.0)


def test_a_node_with_nothing_around_it_has_no_ghost():
    ghosts, informed = ghost_values(2, np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0),
                                    np.zeros(1, int), np.array([[1.0, 0.0, 0.0]]), np.array([3.0]))  # fmt: skip
    np.testing.assert_array_equal(informed, [True, False])
    assert ghosts[1] == 0.0


def test_a_cut_point_beside_its_neighbour_leaves_the_ghost_to_the_extrapolation():
    """t tiny: the exact edge fit would be enormous; the ghost stays near the extrapolation."""
    neighbours = np.array(
        [[-1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [-1.0, -1.0, -1.0]]
    )
    ghosts, _ = ghost_values(1, np.zeros(1, int), np.array([1e-4]), np.array([0.0]), np.array([0.01]),
                             np.zeros(4, int), neighbours, np.zeros(4))  # fmt: skip
    assert abs(ghosts[0]) < 0.02


# -- the refinement --------------------------------------------------------------------------------


class _Recorder:
    """A sampler that records every point it is asked for."""

    def __init__(self, function):
        self.function = function
        self.asked: list[np.ndarray] = []

    def __call__(self, points):
        self.asked.append(np.array(points))
        return self.function(points)


def _water():
    """A cylindrical vessel of water with a lamp-like rod along its axis."""
    vessel = Cylinder(centre=[0.0, 0.0, 0.0], axis=[1.0, 0.0, 0.0], radius=0.02, half_length=0.02)
    rod = Cylinder(centre=[0.0, 0.0, 0.0], axis=[1.0, 0.0, 0.0], radius=0.005, half_length=0.05)
    return Difference(vessel, rod)


def _decay(points):
    radius = np.hypot(points[:, 1], points[:, 2])
    return 100.0 * np.exp(-200.0 * (radius - 0.005)) * 0.005 / radius


BOX = dict(low=[-0.0225, -0.0225, -0.0225], high=[0.0225, 0.0225, 0.0225])


def test_a_field_linear_in_what_is_interpolated_is_not_refined_and_is_exact():
    water = _water()
    linear = lambda p: 5.0 + p @ np.array([10.0, -20.0, 30.0])  # noqa: E731
    field = sample_adaptively(linear, water, **BOX, spacing=0.005, max_level=3, tolerance=1e-8,
                              standoff=1e-4, min_level=1)  # fmt: skip
    assert set(field.tree.levels) <= {0, 1}
    assert field.rounds[-1].marked == 0
    points = np.random.default_rng(7).uniform(-0.02, 0.02, (2000, 3))
    points = points[np.asarray(water.contains(points))]
    np.testing.assert_allclose(field(points), linear(points), atol=1e-9)


def test_refinement_goes_deep_only_beside_the_steep_part_and_meets_the_tolerance():
    water = _water()
    sampler = _Recorder(_decay)
    field = sample_adaptively(sampler, water, **BOX, spacing=0.005, max_level=4, tolerance=0.01,
                              standoff=1e-5, transform=Logarithm(1e-6))  # fmt: skip
    centres = field.tree.centres()
    distance = np.hypot(centres[:, 1], centres[:, 2]) - 0.005
    deepest = field.tree.levels == field.tree.max_level
    # The field's curvature is steepest beside the rod, so most of the finest cells are there.
    assert np.median(distance[deepest]) < 0.001
    assert np.median(distance[field.tree.levels == field.tree.max_level - 1]) > 0.004
    rng = np.random.default_rng(8)
    points = rng.uniform(-0.02, 0.02, (20000, 3))
    points = points[np.asarray(water.contains(points))]
    error = np.abs(field(points) / _decay(points) - 1.0)
    assert np.median(error) < 1e-3
    assert np.percentile(error, 99) < 0.02
    # A uniform grid at the root spacing does far worse: the refinement is doing the work.
    uniform = sample_adaptively(_decay, water, **BOX, spacing=0.005, max_level=0, tolerance=np.inf,
                                standoff=1e-5, transform=Logarithm(1e-6), min_level=0)  # fmt: skip
    assert np.median(np.abs(uniform(points) / _decay(points) - 1.0)) > 10 * np.median(error)


def test_each_point_is_sampled_once_and_inside_the_region():
    water = _water()
    sampler = _Recorder(_decay)
    field = sample_adaptively(sampler, water, **BOX, spacing=0.005, max_level=3, tolerance=0.01,
                              standoff=1e-5, transform=Logarithm(1e-6))  # fmt: skip
    asked = np.concatenate(sampler.asked)
    assert len(sampler.asked) == len(field.rounds)
    assert len(np.unique(np.round(asked / 1e-12), axis=0)) == len(asked)
    assert np.all(np.asarray(water.contains(asked)))
    np.testing.assert_array_equal(asked, field.points)


def test_the_interpolant_passes_through_every_sampled_node_and_near_every_cut_point():
    water = _water()
    field = sample_adaptively(_decay, water, **BOX, spacing=0.005, max_level=3, tolerance=0.01,
                              standoff=1e-5, transform=Logarithm(1e-6))  # fmt: skip
    nodes = field.points[~field.is_cut]
    np.testing.assert_allclose(field(nodes), field.values[~field.is_cut], rtol=1e-10)
    cut = field.points[field.is_cut]
    np.testing.assert_allclose(field(cut), field.values[field.is_cut], rtol=0.01)


def test_a_cut_point_beside_the_rod_is_what_makes_the_layer_next_to_it_right():
    """Points within one root cell of the rod's surface, on a tree refined nowhere: with the cut
    points the layer is interpolated between values on both sides; the cut points themselves sit a
    standoff from the surface."""
    water = _water()
    field = sample_adaptively(_decay, water, **BOX, spacing=0.0025, max_level=0, tolerance=np.inf,
                              standoff=1e-5, transform=Logarithm(1e-6), min_level=0)  # fmt: skip
    # A cut point lies a standoff from the boundary by signed distance, whatever its edge's angle.
    cut = field.points[field.is_cut]
    rod = np.hypot(cut[:, 1], cut[:, 2]) - 0.005
    near_rod = (rod < 0.001) & (np.abs(cut[:, 0]) < 0.019)  # not those beside the vessel's ends
    assert near_rod.sum() > 100
    np.testing.assert_allclose(rod[near_rod], 1e-5, rtol=1e-6)
    rng = np.random.default_rng(9)
    angle, gap = rng.uniform(0, 2 * np.pi, 2000), rng.uniform(1e-5, 5e-4, 2000)
    radius = 0.005 + gap
    points = np.column_stack(
        [rng.uniform(-0.015, 0.015, 2000), radius * np.cos(angle), radius * np.sin(angle)]
    )
    error = np.abs(field(points) / _decay(points) - 1.0)
    assert np.median(error) < 0.05


def test_logarithm_floors_and_refuses_a_floor_of_zero():
    transform = Logarithm(1e-3)
    np.testing.assert_allclose(
        transform.forward(np.array([0.0, 1e-5, 1.0])), [np.log(1e-3)] * 2 + [0.0]
    )
    np.testing.assert_allclose(transform.inverse(transform.forward(np.array([2.0]))), [2.0])
    with pytest.raises(ValueError, match="positive"):
        Logarithm(0.0)
    np.testing.assert_array_equal(Linear().forward(np.array([-1.0, 2.0])), [-1.0, 2.0])


def test_a_point_outside_the_box_is_refused():
    water = _water()
    field = sample_adaptively(_decay, water, **BOX, spacing=0.01, max_level=0, tolerance=np.inf,
                              standoff=1e-5, min_level=0)  # fmt: skip
    with pytest.raises(ValueError, match="in no leaf of the octree"):
        field(np.array([[1.0, 0.0, 0.0]]))


def test_hexahedra_are_the_leaves_holding_the_region_with_its_samples_at_their_corners():
    water = _water()
    field = sample_adaptively(_decay, water, **BOX, spacing=0.005, max_level=2, tolerance=0.01,
                              standoff=1e-5, transform=Logarithm(1e-6))  # fmt: skip
    points, corners, rows, values = field.hexahedra()
    np.testing.assert_allclose(points[corners], field.tree.position(field.tree.corners(rows)))
    inside = np.asarray(water.signed_distance(points)) <= -1e-5  # sampled: a standoff inside
    np.testing.assert_array_equal(np.isfinite(values), inside)
    np.testing.assert_allclose(values[inside], _decay(points[inside]), rtol=1e-10)
    # Every leaf left out has no corner in the water.
    left_out = np.setdiff1d(np.arange(field.tree.n_leaves), rows)
    outside = ~np.asarray(
        water.contains(field.tree.position(field.tree.corners(left_out)).reshape(-1, 3))
    )
    assert outside.all()
    # And the leaves kept include the ones the rod cuts, all of whose corners are inside the rod.
    assert np.any(~inside[corners].any(axis=1))


def test_the_cut_points_misses_take_the_cells_beside_the_rod_to_the_deepest_level():
    """Beside the rod the steepest change is between the last node and the surface, where only cut
    points sample it: their misses are what split the cells there the last time."""
    field = sample_adaptively(_decay, _water(), **BOX, spacing=0.005, max_level=4, tolerance=0.05,
                              standoff=1e-5, transform=Logarithm(1e-6))  # fmt: skip
    deepest = field.tree.levels == field.tree.max_level
    assert deepest.any()
    centres = field.tree.centres(np.flatnonzero(deepest))
    assert np.median(np.hypot(centres[:, 1], centres[:, 2]) - 0.005) < 0.001


def test_a_cell_with_a_corner_not_sampled_is_tested_at_its_centre_and_split_where_that_misses():
    """A cell the rod cuts has corners that are ghosts, fitted to agree with its cut points, so its
    own samples cannot show how far it misses between them. Its centre is sampled as a probe; no cell
    is left unsplit above the deepest level with a probe at its centre that missed by more than the
    tolerance. At this standoff the cut points lie far enough from the rod that several cells' corners
    and cut points all agree while their centres do not."""
    tolerance = 0.05
    field = sample_adaptively(_decay, _water(), **BOX, spacing=0.005, max_level=4,
                              tolerance=tolerance, standoff=5e-4, transform=Logarithm(1e-6))  # fmt: skip
    tree = field.tree
    # The last pass probes too: its probes join the interpolant then, with no later pass to add them.
    assert field.rounds[-1].probes > 0
    rows = np.flatnonzero(tree.levels < tree.max_level)
    centres = tree.key(tree.low[rows] + tree.size(tree.levels[rows])[:, None] // 2)
    nodes = np.flatnonzero(~field.is_cut)
    sampled = tree.key(np.round(tree.lattice(field.points[nodes])).astype(np.int64))
    probed = np.isin(centres, sampled)
    assert probed.sum() > 100  # cells beside the rod, left whole, with a probe at the centre
    miss = dict(zip(sampled.tolist(), field.misses[nodes].tolist(), strict=True))
    assert max(miss[key] for key in centres[probed].tolist()) <= tolerance
    # Each probe joined the interpolant as a node: it passes through the value sampled there.
    at = nodes[np.isin(sampled, centres[probed])]
    np.testing.assert_allclose(field(field.points[at]), field.values[at], rtol=1e-10)


@pytest.mark.parametrize("scale", [1.0, 1e-3])
def test_the_default_standoff_is_a_thousandth_of_the_finest_cell_at_any_scale(scale):
    """The same vessel at two sizes a thousand apart: every sample lies inside the region by at least
    a thousandth of the finest cell, and the cut points lie exactly that deep."""
    vessel = Cylinder(centre=[0.0, 0.0, 0.0], axis=[1.0, 0.0, 0.0], radius=0.02 * scale,
                      half_length=0.02 * scale)  # fmt: skip
    rod = Cylinder(centre=[0.0, 0.0, 0.0], axis=[1.0, 0.0, 0.0], radius=0.005 * scale,
                   half_length=0.05 * scale)  # fmt: skip
    water = Difference(vessel, rod)
    box = {key: np.asarray(value) * scale for key, value in BOX.items()}
    field = sample_adaptively(lambda p: _decay(p / scale), water, **box, spacing=0.005 * scale,
                              max_level=2, tolerance=0.05, transform=Logarithm(1e-6))  # fmt: skip
    standoff = 1e-3 * 0.005 * scale / 4
    depth = -np.asarray(water.signed_distance(field.points))
    assert depth.min() >= standoff * (1 - 1e-6)
    np.testing.assert_allclose(depth[field.is_cut], standoff, rtol=1e-6)


def test_a_standoff_that_is_not_positive_is_refused():
    with pytest.raises(ValueError, match="standoff must be positive"):
        sample_adaptively(_decay, _water(), **BOX, spacing=0.01, max_level=0, tolerance=np.inf,
                          standoff=0.0, min_level=0)  # fmt: skip
