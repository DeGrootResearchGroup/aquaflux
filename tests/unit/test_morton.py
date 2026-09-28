"""Tests for the Morton (Z-order) ordering of points shared by every grouping of them."""

from __future__ import annotations

import numpy as np
from aquaflux.morton import morton_order


def test_the_curve_visits_a_cube_s_corners_in_z_order():
    """The eight corners of a cube, shuffled, come back ordered by ``x + 2y + 4z``.

    That is the interleaving -- x in the lowest bit -- stated at the one resolution where it can
    be read off by hand.
    """
    corners = np.array(list(np.ndindex(2, 2, 2)), dtype=float)[:, ::-1]  # rows as (x, y, z)
    shuffled = corners[np.random.default_rng(2).permutation(8)]
    ordered = shuffled[morton_order(shuffled)]
    keys = ordered @ np.array([1.0, 2.0, 4.0])
    assert keys.tolist() == list(range(8))


def test_the_curve_keeps_neighbours_together():
    """Consecutive points along the curve are far closer than consecutive points of a shuffle."""
    rng = np.random.default_rng(4)
    points = rng.uniform(0.0, 1.0, (4096, 3))
    order = morton_order(points)
    assert sorted(order.tolist()) == list(range(4096))
    step = np.linalg.norm(np.diff(points[order], axis=0), axis=1).mean()
    shuffled = np.linalg.norm(np.diff(points, axis=0), axis=1).mean()
    assert step < 0.2 * shuffled, (step, shuffled)


def test_a_run_around_a_long_thin_tube_faces_one_way():
    """On a tube forty times longer than it is wide, a run of 32 spans a narrow slice of angle.

    That is what the per-axis cells are for: a group whose facets face nearly the same way is one
    a "wholly behind these points" test can drop at once. A tube of radius 0.01 along x and 0.8
    long -- a lamp's proportions -- sampled at random, each point's outward normal radial. The
    median, over runs of 32, of the largest angle between a member's normal and the run's mean is
    about 8 degrees with cells following the box, and about 50 with cubic cells, which make a run
    compact but wrap it further around the tube. The bound sits between the two.
    """
    rng = np.random.default_rng(3)
    angle, along = rng.uniform(0.0, 2 * np.pi, 4096), rng.uniform(0.0, 0.8, 4096)
    normal = np.stack([np.zeros(4096), np.cos(angle), np.sin(angle)], axis=1)
    points = np.stack([along, 0.01 * normal[:, 1], 0.01 * normal[:, 2]], axis=1)
    runs = normal[morton_order(points)].reshape(-1, 32, 3)
    mean = runs.sum(axis=1)
    mean /= np.linalg.norm(mean, axis=1, keepdims=True)
    spread = np.degrees(np.arccos(np.clip(np.einsum("rkd,rd->rk", runs, mean), -1.0, 1.0)))
    assert np.median(spread.max(axis=1)) < 25.0, np.median(spread.max(axis=1))


def test_stretching_one_axis_does_not_change_the_order():
    """Each axis is scaled to its own extent, so stretching one leaves the curve where it was."""
    points = np.random.default_rng(7).uniform(0.0, 1.0, (500, 3))
    assert np.array_equal(morton_order(points), morton_order(points * [40.0, 1.0, 0.05]))


def test_the_order_does_not_depend_on_where_the_points_are_or_their_scale():
    """A translated and uniformly scaled copy is visited in the same order."""
    points = np.random.default_rng(5).uniform(0.0, 1.0, (500, 3))
    moved = 0.001 * points + [10.0, -4.0, 2.5]
    assert np.array_equal(morton_order(points), morton_order(moved))


def test_degenerate_point_sets_are_ordered_without_dividing_by_zero():
    """No points, one point, coincident points, and points in a plane."""
    assert morton_order(np.zeros((0, 3))).tolist() == []
    assert morton_order(np.ones((1, 3))).tolist() == [0]
    assert morton_order(np.ones((5, 3))).tolist() == [0, 1, 2, 3, 4]  # ties keep input order
    flat = np.random.default_rng(6).uniform(0.0, 1.0, (64, 3)) * [1.0, 1.0, 0.0]
    assert sorted(morton_order(flat).tolist()) == list(range(64))
