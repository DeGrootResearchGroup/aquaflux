"""Refinement regions: which cubes they split, the points they hand Netgen, and their refusals.

Each region is checked against an independent description of the part of space it covers -- a
ball's or a box's distance in closed form, a square patch's slab -- not against the test the code
itself makes.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from aquaflux.mesh import closed_cell_residual
from aquaflux.meshing import (
    LocalSizes,
    OctreeCore,
    RefinedBox,
    RefinedCylinder,
    RefinedNear,
    RefinedSphere,
    SampledRegion,
    element_mesh,
)
from aquaflux.octree import Octree
from scipy.spatial import cKDTree

CENTRE = np.array([0.5, 0.5, 0.5])


def _box_distance(lower, upper, point):
    """Each box's distance from ``point``, zero where the box holds it."""
    nearest = np.clip(point, lower, upper)
    return np.linalg.norm(nearest - point, axis=1)


def test_a_ball_splits_every_cube_it_reaches_and_no_cube_far_from_it():
    sphere = RefinedSphere(centre=tuple(CENTRE), radius=0.2, size=0.05).located({})
    tree = Octree.lattice([0, 0, 0], [1, 1, 1], 0.4).refined_by([sphere])
    lower, size = tree.lower, tree.size
    met = _box_distance(lower, lower + size[:, None], CENTRE) < 0.2
    # Every cube the ball reaches is no larger than the size: 0.4 halved three times.
    np.testing.assert_allclose(size[met], 0.05, rtol=1e-12)
    # A cube is split only where its parent's centre lies within the radius and half its diagonal,
    # so a fine cube's centre lies within the radius and one and a half of its parent's.
    fine = size < 0.06
    centre = lower[fine] + 0.5 * size[fine, None]
    assert np.all(np.linalg.norm(centre - CENTRE, axis=1) <= 0.2 + 1.5 * np.sqrt(3) * 0.1)
    assert np.any(~fine)


def test_a_balls_points_cover_it_at_its_size_and_stay_beside_it():
    sphere = RefinedSphere(centre=tuple(CENTRE), radius=0.2, size=0.05).located({})
    sizes = sphere.local_sizes([0, 0, 0], [1, 1, 1])
    np.testing.assert_allclose(sizes.sizes, 0.05)
    rng = np.random.default_rng(3)
    direction = rng.normal(size=(2000, 3))
    inside = CENTRE + (
        direction / np.linalg.norm(direction, axis=1, keepdims=True) * 0.2 * rng.random((2000, 1))
    )
    # Every point of the ball has a point of the lattice within half a cube's diagonal.
    gap, _ = cKDTree(sizes.points).query(inside)
    assert gap.max() <= 0.5 * np.sqrt(3) * 0.05 + 1e-12
    # Each point is a centre of a cube of the size whose centre is within the radius and half
    # its diagonal: the points are the ball's, not the box's.
    np.testing.assert_allclose(
        (sizes.points - 0.025) / 0.05, np.round((sizes.points - 0.025) / 0.05), atol=1e-9
    )
    assert np.all(
        np.linalg.norm(sizes.points - CENTRE, axis=1) <= 0.2 + 0.5 * np.sqrt(3) * 0.05 + 1e-12
    )
    assert len(sizes.points) < 0.1 * 20**3


def test_a_box_and_a_cylinder_cover_exactly_the_cubes_they_hold():
    box = RefinedBox(lower=(0.2, 0.2, 0.2), upper=(0.6, 0.4, 0.8), size=0.1).located({})
    points = box.local_sizes([0, 0, 0], [1, 1, 1]).points
    # Every cube of the lattice inside the box, and none whose centre is beyond its half-diagonal.
    held = np.array(list(itertools.product(np.arange(0.25, 0.6, 0.1), [0.25, 0.35],
                                           np.arange(0.25, 0.8, 0.1))))  # fmt: skip
    assert len(cKDTree(points).query_ball_point(held, 1e-9, return_length=True).nonzero()[0]) == (
        len(held)
    )
    beyond = np.max(np.maximum(np.abs(points - [0.4, 0.3, 0.5]) - [0.2, 0.1, 0.3], 0), axis=1)
    assert np.all(beyond <= 0.5 * np.sqrt(3) * 0.1 + 1e-12)

    cylinder = RefinedCylinder(start=(0.5, 0.5, 0.1), end=(0.5, 0.5, 0.9), radius=0.1, size=0.05)
    points = cylinder.located({}).local_sizes([0, 0, 0], [1, 1, 1]).points
    radial = np.linalg.norm(points[:, :2] - 0.5, axis=1)
    reach = 0.5 * np.sqrt(3) * 0.05 + 1e-12
    assert np.all(radial <= 0.1 + reach)
    assert np.all((points[:, 2] >= 0.1 - reach) & (points[:, 2] <= 0.9 + reach))
    # The axis is covered end to end.
    axis = np.column_stack([np.full(9, 0.5), np.full(9, 0.5), np.linspace(0.1, 0.9, 9)])
    assert cKDTree(points).query(axis)[0].max() <= reach


def _square():
    """The square z = 0, 0 <= x, y <= 1, as two triangles' corners."""
    corner = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    return corner[[[0, 1, 2], [0, 2, 3]]]


def test_the_fluid_near_a_patch_is_refined_to_its_distance():
    near = RefinedNear(patches=("lid",), distance=0.15, size=0.05)
    located = near.located({"lid": _square(), "walls": _square() + np.array([0, 0, 1])})
    points = located.local_sizes([0, 0, -1], [1, 1, 1]).points
    # The slab over the square, up to the distance, is covered ...
    slab = np.array(list(itertools.product(np.linspace(0, 1, 11), np.linspace(0, 1, 11),
                                           np.linspace(-0.15, 0.15, 7))))  # fmt: skip
    assert cKDTree(points).query(slab)[0].max() <= 0.5 * np.sqrt(3) * 0.05 + 1e-12
    # ... and nothing much beyond it: a cube is reached within the distance of the voxels the
    # triangles are registered in, along each axis -- not the other patch, a metre away.
    assert np.abs(points[:, 2]).max() <= 0.15 + 0.05 + 0.025 + 1e-9
    assert points[:, 2].max() < 0.5


def test_a_refinement_near_a_patch_the_region_lacks_is_refused_naming_the_patches():
    with pytest.raises(ValueError, match=r"\['inlet'\], which are not patches.*\['walls'\]"):
        RefinedNear(patches=("inlet",), distance=0.1, size=0.05).located({"walls": _square()})


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: RefinedSphere(centre=(0, 0, 0), radius=1.0, size=0.0), "size must be a positive"),
        (lambda: RefinedSphere(centre=(0, 0), radius=1.0, size=0.1), "centre must be three"),
        (lambda: RefinedSphere(centre=(0, 0, 0), radius=-1.0, size=0.1), "radius must be"),
        (lambda: RefinedBox(lower=(0, 0, 0), upper=(1, 0, 1), size=0.1), "above lower"),
        (lambda: RefinedCylinder(start=(0, 0, 0), end=(0, 0, 0), radius=1.0, size=0.1), "apart"),
        (lambda: RefinedNear(patches=(), distance=0.1, size=0.1), "names the patches"),
        (lambda: RefinedNear(patches=("a",), distance=0.0, size=0.1), "distance must be"),
    ],
)
def test_a_refinement_out_of_range_is_refused(make, message):
    with pytest.raises(ValueError, match=message):
        make()


def test_local_sizes_refuse_a_point_without_a_positive_size():
    with pytest.raises(ValueError, match="one size each"):
        LocalSizes(np.zeros((2, 3)), np.ones(3))
    with pytest.raises(ValueError, match="positive"):
        LocalSizes(np.zeros((2, 3)), np.array([1.0, 0.0]))
    joined = LocalSizes.joined(
        [LocalSizes(np.zeros((1, 3)), [1.0]), LocalSizes(np.ones((2, 3)), [2.0, 3.0])]
    )
    assert joined.sizes.tolist() == [1.0, 2.0, 3.0]


def _box(points):
    return np.all((points >= 0.0) & (points <= 2.0), axis=1)


def test_the_hex_core_is_refined_inside_a_region_reaching_its_boundary_and_stays_conformal():
    # Cubes of 0.4 kept 0.2 clear of a 2 m box; a ball of radius 0.5 at a corner of the core
    # refines them to 0.1 there, out to the core's boundary, so transition cubes reach it.
    ball = RefinedSphere(centre=(0.6, 0.6, 0.6), radius=0.5, size=0.1).located({})
    core = OctreeCore(size=0.4, wall_size=0.2, clearance=0.2).mesh(
        SampledRegion(_box), [0, 0, 0], [2, 2, 2], cap_height=0.25, boundary_size=0.2,
        refinements=[ball],
    )  # fmt: skip
    tree, kept = core.octree, core.kept
    lower, size = tree.lower[kept], tree.size[kept]
    reached = _box_distance(lower, lower + size[:, None], np.array([0.6, 0.6, 0.6])) < 0.5
    assert np.any(reached)
    np.testing.assert_allclose(size[reached], 0.1, rtol=1e-12)
    assert size.max() == pytest.approx(0.4)
    mesh = element_mesh(
        core.points, core.tetrahedra, None, core.hexahedra, np.vstack([core.pyramids, core.caps])
    )
    volume = np.asarray(mesh.geometry().cell.volume)
    scale = float(np.max(np.asarray(mesh.geometry().face.area)))
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-12 * scale
    assert np.all(volume > 0.0)
    # The caps are the last cells; the rest fill exactly the cubes kept.
    assert volume[: -len(core.caps)].sum() == pytest.approx(np.sum(size**3), rel=1e-12)
