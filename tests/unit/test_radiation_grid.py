"""The uniform grid over blocking triangles: it must change the cost and nothing else."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation.grid import TriangleGrid
from aquaflux.radiation.triangles import segment_is_cut

from tests.unit.radiation_references import closed_drum


def scattered_triangles(count: int, rng, *, scale: float = 0.25) -> np.ndarray:
    """Small triangles at random positions and orientations through the unit cube."""
    centre = rng.uniform(0.0, 1.0, (count, 1, 3))
    return centre + scale * rng.normal(size=(count, 3, 3)) / np.sqrt(3.0)


def rays_through(count: int, rng, *, spread: float = 1.6):
    """Segments crossing the cube from random points outside it, with a few degenerate ones."""
    origin = rng.uniform(-spread, 1.0 + spread, (count, 3))
    target = rng.uniform(-spread, 1.0 + spread, (count, 3))
    # Axis-parallel segments exercise the walk's zero-direction guards, which is where a
    # division by a zero component would otherwise put an infinity into the stepping.
    origin[: count // 8, 1:] = 0.5
    target[: count // 8, 1:] = 0.5
    return origin, target


def brute(origin, target, vertices, near, exclude=None) -> np.ndarray:
    return np.asarray(
        segment_is_cut(
            jnp.asarray(origin),
            jnp.asarray(target),
            jnp.asarray(vertices),
            jnp.asarray(near),
            exclude=None if exclude is None else jnp.asarray(exclude),
        )
    )


@pytest.mark.parametrize("resolution", [None, 1, 3, 16, (2, 7, 5)])
def test_the_grid_answers_exactly_what_testing_every_triangle_answers(resolution):
    """The whole contract. A grid decides what is worth testing, never what counts as a hit, so
    any disagreement with the unaccelerated test is a defect however small -- and the ones a
    walk gets wrong are systematic: a triangle in a voxel the walk skips is missed on every ray
    that would have hit it. Swept over resolutions because that is what changes which voxels a
    segment enters; one voxel is the brute-force test, and a rectangular grid catches an axis
    mixed up in the flattening.
    """
    rng = np.random.default_rng(11)
    vertices = scattered_triangles(400, rng)
    origin, target = rays_through(3000, rng)
    near = np.zeros(len(origin))
    grid = TriangleGrid.build(vertices, resolution=resolution)
    found = grid.blocks(origin, target, near)
    expected = brute(origin, target, vertices, near)
    assert 0.2 < expected.mean() < 0.9, f"fixture is one-sided: {expected.mean()}"
    np.testing.assert_array_equal(found, expected)


def test_the_grid_honours_the_exclusions_and_the_near_margin():
    """Both ends of the distance window, since the grid re-tests them on its own candidates.

    A ray leaving a triangle's own plane must ignore that triangle, and one aimed at a point on
    another must ignore that one too, or a pair is blocked by its own endpoints.
    """
    rng = np.random.default_rng(3)
    vertices = scattered_triangles(120, rng)
    centroid = vertices.mean(axis=1)
    origin = np.repeat(centroid[:1], len(centroid), axis=0)
    target = centroid
    near = np.full(len(origin), 1e-6)
    exclude = np.stack([np.zeros(len(origin), dtype=int), np.arange(len(origin))], axis=1)
    grid = TriangleGrid.build(vertices)
    np.testing.assert_array_equal(
        grid.blocks(origin, target, near, exclude=exclude),
        brute(origin, target, vertices, near, exclude),
    )


def test_a_segment_that_never_enters_the_grid_is_not_blocked():
    """Cheap and common: a probe out beyond the vessel, or a lamp whose box the cell misses.

    Answered without walking, so it must be answered correctly without walking.
    """
    rng = np.random.default_rng(5)
    grid = TriangleGrid.build(scattered_triangles(50, rng))
    origin = np.array([[-5.0, -5.0, -5.0], [3.0, 3.0, 3.0]])
    target = np.array([[-4.0, -5.0, -5.0], [4.0, 3.0, 3.0]])
    assert not grid.blocks(origin, target, np.zeros(2)).any()


def test_no_ray_escapes_a_closed_body_through_the_grid():
    """The property the shadow mask exists for, on geometry whose facets meet edge to edge:
    every segment from inside a closed drum to a point outside it is blocked. A grid that
    registered a triangle in too few voxels leaks exactly here, and a leak is a bright spot in
    a field rather than an error.
    """
    rng = np.random.default_rng(7)
    drum = closed_drum(64, radius=1.0, half_height=1.0)
    inside = rng.uniform(-0.4, 0.4, (400, 3))
    outside = 4.0 * rng.normal(size=(400, 3))
    outside /= np.linalg.norm(outside, axis=1, keepdims=True) / 4.0
    grid = TriangleGrid.build(drum)
    assert grid.blocks(inside, outside, np.zeros(len(inside))).all()


def test_the_default_resolution_follows_the_triangle_count_and_the_shape_of_the_box():
    """Voxels near cubic, and about ten triangles in an occupied one -- the measured trade.

    A grid sized by axis rather than by extent walks a long thin domain in tiny steps across its
    short axis, which is how a walk ends up costing more than the test it replaces.
    """
    rng = np.random.default_rng(2)
    stretched = scattered_triangles(4000, rng, scale=0.05) * np.array([8.0, 1.0, 1.0])
    grid = TriangleGrid.build(stretched)
    spacing = grid.spacing
    assert spacing.max() / spacing.min() < 1.5, f"voxels are not near cubic: {spacing}"
    assert grid.resolution[0] > grid.resolution[1], "the long axis needs more voxels"
    occupied = np.diff(grid.starts)
    assert 2.0 < occupied[occupied > 0].mean() < 40.0, occupied[occupied > 0].mean()


def test_a_grid_needs_triangles_and_a_positive_resolution():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="at least one triangle"):
        TriangleGrid.build(np.zeros((0, 3, 3)))
    with pytest.raises(ValueError, match="at least 1 voxel"):
        TriangleGrid.build(scattered_triangles(4, rng), resolution=0)


def test_the_near_margin_is_read_in_LENGTH_units_not_as_a_share_of_the_segment():
    """``min_distance`` is a length, as :func:`segment_is_cut` takes it, and the grid must read
    it the same way -- it divides by the segment's length before comparing.

    Every other fixture in this file runs segments about one unit long with a hair of a margin,
    where a length and a fraction of a length are indistinguishable, so none of them can tell
    the two readings apart. This one puts a blocker halfway along a segment **ten** units long:
    a margin of 3 lengths reaches 0.3 of the way, so the blocker at 0.5 counts, while the same
    number read as a share of the segment sits past the far end and would hide it. A margin of 7
    genuinely does hide it, which is the other half of the check -- a grid that ignored the
    margin entirely would pass the first assertion and fail this one.
    """
    vertices = np.array(
        [
            [[5.0, -1.0, -1.0], [5.0, 3.0, -1.0], [5.0, 0.0, 3.0]],  # across the segment at 0.5
            [[8.0, 4.0, 4.0], [8.0, 6.0, 4.0], [8.0, 5.0, 6.0]],  # off to the side, for extent
        ]
    )
    origin = np.zeros((2, 3))
    target = np.tile([10.0, 0.0, 0.0], (2, 1))
    near = np.array([3.0, 7.0])
    found = TriangleGrid.build(vertices).blocks(origin, target, near)
    # Pinned against the geometry itself, not only against the unaccelerated path, so the two
    # cannot be wrong together.
    assert found.tolist() == [True, False]
    np.testing.assert_array_equal(found, brute(origin, target, vertices, near))


def test_the_work_limit_bounds_the_pairs_it_holds_without_changing_the_answers():
    """A step tests every live ray against everything its voxel holds, so the pairs of one step
    are unbounded unless something bounds them -- and a coarse grid over many rays is exactly
    where that bites: the first attempt to walk a reactor's wall was killed for memory, not
    slow. Splitting a step into groups must be invisible in the answers, so the same walk is run
    at limits far below one step's pairs and compared with the unsplit one.
    """
    rng = np.random.default_rng(13)
    vertices = scattered_triangles(300, rng)
    origin, target = rays_through(400, rng)
    near = np.zeros(len(origin))
    grid = TriangleGrid.build(vertices, resolution=4)
    whole = grid.blocks(origin, target, near)
    assert 0.2 < whole.mean() < 0.9, f"fixture is one-sided: {whole.mean()}"
    for limit in (1, 17, 500):
        np.testing.assert_array_equal(grid.blocks(origin, target, near, work_limit=limit), whole)


def test_the_default_resolution_is_sized_by_the_TRIANGLES_AREA_not_the_boxs_volume():
    """The occupancy target is about a surface, because blocking triangles are a surface.

    A shell in a large box is the case that separates the two rules: its triangles occupy the
    voxels its sheet passes through, of order ``area / size**2``, while the box holds
    ``volume / size**3`` of them. Sizing by the volume therefore lands a shell in far too few
    voxels -- measured on a reactor wall, 217 triangles in an occupied voxel against the ten
    intended. The tolerance here is loose on the high side on purpose: a triangle registers in
    every voxel its bounding box spans, so entries per voxel run above the sheet's own
    occupancy.
    """
    rng = np.random.default_rng(4)
    angle = rng.uniform(0.0, 2.0 * np.pi, 3000)
    height = rng.uniform(-4.0, 4.0, 3000)
    # A thin cylindrical shell of small triangles, inside a box eight times as long as it is wide.
    centre = np.stack([np.cos(angle), np.sin(angle), height], axis=1)
    shell = centre[:, None, :] + 0.02 * rng.normal(size=(3000, 3, 3))
    grid = TriangleGrid.build(shell)
    occupied = np.diff(grid.starts)
    assert (occupied > 0).sum() > 300, f"the shell landed in {(occupied > 0).sum()} voxels"
    assert occupied[occupied > 0].mean() < 40.0, occupied[occupied > 0].mean()
