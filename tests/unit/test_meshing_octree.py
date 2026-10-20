"""An octree refined near a sphere, checked by brute force over every pair of its leaves.

Which leaves touch, and how, is worked out here from their boxes alone -- two closed boxes share a
face when they overlap with positive extent along two axes, part of an edge when along one -- not
from the tree's own neighbour search, so a fault in that search cannot hide a fault in the balance.
"""

from __future__ import annotations

import numpy as np
from aquaflux.meshing import NearSurface, Octree

CENTRE, RADIUS = np.array([0.4, 0.4, 0.4]), 0.2
BOX = 1.5
ROOT, FINEST = 0.25, 0.25 / 8


def _ball(points):
    return np.linalg.norm(points - CENTRE, axis=1) < RADIUS


def _largest_jumps(tree, rows=256):
    """The largest level difference between two leaves sharing a face or part of an edge, and the
    most axes two distinct leaves overlap along with positive extent (``3``: they overlap)."""
    lo, hi = tree.corner, tree.corner + tree.span[:, None]
    jump, most = 0, 0
    for start in range(0, lo.shape[0], rows):
        part = slice(start, start + rows)
        overlap = np.minimum(hi[part, None], hi[None]) - np.maximum(lo[part, None], lo[None])
        axes = np.where(np.all(overlap >= 0, axis=2), np.count_nonzero(overlap > 0, axis=2), -1)
        axes[np.arange(axes.shape[0]), np.arange(start, start + axes.shape[0])] = -1
        touching = axes >= 1  # a face (two axes) or part of an edge (one)
        levels = np.abs(tree.level[part, None] - tree.level[None])
        jump = max(jump, int(np.max(levels[touching], initial=0)))
        most = max(most, int(axes.max()))
    return jump, most


def _refined():
    return Octree.lattice([0, 0, 0], [BOX] * 3, ROOT).refined_by([NearSurface(_ball, FINEST)])


def test_refinement_near_a_sphere_reaches_its_size_and_balance_leaves_neighbours_a_level_apart():
    tree = _refined().balanced()
    jump, most = _largest_jumps(tree)
    assert most == 2, "leaves overlap"
    np.testing.assert_allclose(np.sum(tree.size**3), BOX**3, rtol=1e-12)
    assert jump == 1
    # Balance did not get there by refining everything: every level remains.
    assert set(np.unique(tree.level).tolist()) == {0, 1, 2, 3}
    # Every leaf the sphere passes through is of the size asked for.
    rng = np.random.default_rng(0)
    direction = rng.normal(size=(2000, 3))
    on_sphere = CENTRE + RADIUS * direction / np.linalg.norm(direction, axis=1, keepdims=True)
    lower, size = tree.lower, tree.size
    holder = np.all(
        (on_sphere[:, None] >= lower[None])
        & (on_sphere[:, None] < lower[None] + size[None, :, None]),
        axis=2,
    )
    assert np.all(holder.sum(axis=1) == 1)
    np.testing.assert_allclose(size[np.argmax(holder, axis=1)], FINEST)


def test_without_balancing_the_same_refinement_has_neighbours_three_levels_apart():
    # The case above needs the balance: the refinement alone leaves the root cubes the sphere
    # misses beside cubes of the finest level.
    assert _largest_jumps(_refined())[0] == 3


def test_a_box_of_three_by_two_cubes_is_covered_by_whole_root_cubes():
    tree = Octree.lattice([1.0, 2.0, 3.0], [1.7, 2.4, 3.2], 0.25)
    np.testing.assert_array_equal(tree.shape, [3, 2, 1])
    np.testing.assert_allclose(np.sort(np.unique(tree.lower[:, 0])), [1.0, 1.25, 1.5])
    assert tree.level.tolist() == [0] * 6
