"""The uniform grid over blocking triangles: it must change the cost and nothing else."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.mesh import structured_grid_3d
from aquaflux.mesh.surface import patch_triangles
from aquaflux.radiation import grid as grid_module
from aquaflux.radiation.grid import TriangleGrid, _near_cubic_resolution
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
    # And some along y, so a walk or an intersection test that assumed the leading component is the
    # largest -- dividing by a zero x -- is caught too.
    origin[count // 8 : count // 4, [0, 2]] = 0.5
    target[count // 8 : count // 4, [0, 2]] = 0.5
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


def test_the_default_voxels_take_the_boxs_proportions_and_its_triangle_count():
    """Voxels shaped like the box, and several times a near-cubic ten-a-voxel grid's in number.

    A segment's steps along an axis go as its travel there over the voxel's edge, and segments
    through a long box travel mostly along it, so near-cubic voxels walk them in many short steps.
    A near-cubic rule -- the one this replaced -- gives this 8:1 box eight times the voxels along
    its long axis and fails the first assertion.
    """
    rng = np.random.default_rng(2)
    stretched = scattered_triangles(4000, rng, scale=0.05) * np.array([8.0, 1.0, 1.0])
    grid = TriangleGrid.build(stretched)
    assert grid.resolution.max() - grid.resolution.min() <= 1, grid.resolution
    box = np.ptp(stretched.reshape(-1, 3), axis=0)
    np.testing.assert_allclose(grid.spacing / grid.spacing[0], box / box[0], rtol=0.05)
    cubic = _near_cubic_resolution(stretched)
    assert 3.0 < np.prod(grid.resolution) / np.prod(cubic) < 5.0, (grid.resolution, cubic)
    occupied = np.diff(grid.starts)
    assert 1.0 < occupied[occupied > 0].mean() < 40.0, occupied[occupied > 0].mean()


def test_no_default_voxel_is_more_than_the_cap_longer_than_it_is_wide():
    """A very flat or very long box is not cut into sheets: the edges stay within the cap.

    Without the cap a 200:1 box gets 200:1 voxels, and a segment crossing the short way passes the
    long edge of every voxel it enters. A flat surface gets one voxel through its thickness rather
    than a count split over an axis it does not extend along.
    """
    rng = np.random.default_rng(6)
    long = scattered_triangles(3000, rng, scale=0.01) * np.array([200.0, 1.0, 1.0])
    spacing = TriangleGrid.build(long).spacing
    assert spacing.max() / spacing.min() <= 1.1 * grid_module._MAX_ASPECT, spacing
    assert spacing.max() / spacing.min() > 0.5 * grid_module._MAX_ASPECT, spacing
    flat = scattered_triangles(3000, rng, scale=0.01)
    flat[..., 2] = 0.5
    resolution = TriangleGrid.build(flat).resolution
    assert resolution[2] == 1 and resolution[0] > 10 and resolution[1] > 10, resolution
    # And the count is still the budget's, spent over the two axes the surface extends along;
    # solving for it over all three would divide by the flat axis's near-zero extent.
    cubic = _near_cubic_resolution(flat)
    assert 3.0 < np.prod(resolution) / np.prod(cubic) < 5.0, (resolution, cubic)


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


def test_no_ray_aimed_through_an_edge_or_a_vertex_escapes_a_closed_body():
    """The watertight test's own fixture, walked: rays from inside a closed drum aimed exactly at
    every vertex and every edge midpoint, carried on well past the wall.

    A ray through a feature two or more triangles share is claimed by exactly one of them only while
    the edge function is exactly antisymmetric, and an ordinary test leaks on precisely these rays --
    a pinhole in a closed surface, which a field shows as a bright spot rather than an error. Random
    directions, as in the test above, essentially never hit a shared edge, so they cannot see it.
    """
    drum = closed_drum(48, radius=1.0, half_height=1.0)
    corners = drum.reshape(-1, 3)
    edges = 0.5 * (drum + np.roll(drum, -1, axis=1)).reshape(-1, 3)
    aims = np.unique(np.concatenate([corners, edges]), axis=0)
    origin = np.tile([0.1, -0.2, 0.05], (len(aims), 1))
    target = origin + 3.0 * (aims - origin)
    grid = TriangleGrid.build(drum)
    assert grid.blocks(origin, target, np.zeros(len(aims))).all()


def test_a_segment_ending_in_a_triangle_s_plane_is_blocked_by_it():
    """The far end of the window is inclusive: a segment aimed at a point on a triangle it was not
    told to ignore ends in that triangle, and is blocked by it. This is why a ray between two facet
    centroids must exclude its target facet -- without the exclusion every such pair reads blocked --
    and it is the rule the exclusion is written against.
    """
    vertices = np.array(
        [
            [[5.0, -1.0, -1.0], [5.0, 3.0, -1.0], [5.0, 0.0, 3.0]],
            [[8.0, 4.0, 4.0], [8.0, 6.0, 4.0], [8.0, 5.0, 6.0]],  # off to the side, for extent
        ]
    )
    origin = np.zeros((2, 3))
    target = np.array([[5.0, 0.0, 0.0], [4.0, 0.0, 0.0]])  # on the first triangle, and short of it
    found = TriangleGrid.build(vertices).blocks(origin, target, np.zeros(2))
    assert found.tolist() == [True, False]
    np.testing.assert_array_equal(found, brute(origin, target, vertices, np.zeros(2)))


def test_a_hit_exactly_at_the_near_margin_does_not_count():
    """The near end of the window is exclusive. Axis-aligned, so the hit distance is exact: a
    triangle across the segment at 5 of its 10 units is at exactly half of it, and a margin of
    exactly 5 must pass it by, while a margin a hair short must not.
    """
    vertices = np.array(
        [
            [[5.0, -1.0, -1.0], [5.0, 3.0, -1.0], [5.0, 0.0, 3.0]],
            [[8.0, 4.0, 4.0], [8.0, 6.0, 4.0], [8.0, 5.0, 6.0]],
        ]
    )
    origin = np.zeros((2, 3))
    target = np.tile([10.0, 0.0, 0.0], (2, 1))
    near = np.array([5.0, np.nextafter(5.0, 0.0)])
    found = TriangleGrid.build(vertices).blocks(origin, target, near)
    assert found.tolist() == [False, True]
    np.testing.assert_array_equal(found, brute(origin, target, vertices, near))


def flat_sheet() -> np.ndarray:
    """A box mesh's ``front`` patch: 24 triangles exactly in the plane ``z = 2``, 3 x 4 faces."""
    mesh = structured_grid_3d(3, 4, 5, 1.0, 1.5, 2.0, named_boundaries=True)
    return np.asarray(patch_triangles(mesh, mesh.geometry(), ["front"]).vertices)


def through_the_sheet_s_seams(sheet, count: int, rng):
    """Segments crossing a sheet through its interior nodes and edge midpoints, plus a few
    random crossings and segments parallel to it, on its plane and off it.

    Aimed at the seams because that is where a walk that visits the wrong voxel misses: a ray
    through an edge two triangles share is claimed by one of them, and a voxel boundary along
    that edge decides which voxel the walk is in when it gets there. Each segment is carried on
    past the sheet and none touches its rim, so the two intersection kernels -- which may round
    differently where a segment ends exactly on a triangle or grazes the sheet's outline -- have
    no knife edge to disagree on, and every disagreement is the grid's.
    """
    low = sheet.reshape(-1, 3).min(axis=0)
    high = sheet.reshape(-1, 3).max(axis=0)
    flat = int(np.argmin(high - low))
    seams = np.concatenate(
        [sheet.reshape(-1, 3), 0.5 * (sheet + np.roll(sheet, -1, axis=1)).reshape(-1, 3)]
    )
    in_plane = [axis for axis in range(3) if axis != flat]
    interior = np.all(
        (seams[:, in_plane] > low[in_plane]) & (seams[:, in_plane] < high[in_plane]), axis=1
    )
    seams = np.unique(seams[interior], axis=0)
    origin = rng.uniform(low - 1.0, high + 1.0, (count, 3))
    # Carried on past the seam by a random factor: aimed at twice the distance, every segment
    # would cross at exactly half its length, where the crossing point never rounds.
    reach = rng.uniform(1.5, 3.0, (count, 1))
    target = origin + reach * (seams[rng.integers(0, len(seams), count)] - origin)
    random = slice(0, count // 8)
    target[random] = rng.uniform(low - 1.0, high + 1.0, (count // 8, 3))
    parallel = slice(count // 8, count // 4)
    target[parallel] = rng.uniform(low - 1.0, high + 1.0, (count // 8, 3))
    target[parallel, flat] = origin[parallel, flat]
    on_plane = slice(count // 4, count // 4 + count // 16)
    origin[on_plane, flat] = target[on_plane, flat] = low[flat]
    return origin, target


#: Where the sheet is put, as an offset applied before its axes are permuted. A sheet whose
#: plane coordinate is large next to the segments' travel across it is crossed at a point that
#: rounds exactly onto the plane, which hides the flat box's defect; near ``0.1`` and ``0.03`` it
#: does not. The second turns the flat axis onto x, so it is not always the last one. The third,
#: far from the origin, is where the stacked sheets' seams round below a voxel boundary often
#: enough to see a triangle registered short of its last voxel.
SHEET_PLACEMENTS = [
    ((0.1, 0.1, -1.9), (0, 1, 2)),
    ((-7.3, 0.37, -1.97), (2, 0, 1)),
    ((1e3, -3.1, 1e3), (0, 1, 2)),
]


def placed(sheet, offset, axes) -> np.ndarray:
    return (sheet + np.asarray(offset))[..., list(axes)]


@pytest.mark.parametrize("resolution", [None, (3, 4, 1), (6, 8, 1), (6, 8, 2)])
@pytest.mark.parametrize(("offset", "axes"), SHEET_PLACEMENTS)
def test_a_flat_sheet_answers_exactly_what_testing_every_triangle_answers(resolution, offset, axes):
    """Triangles all in one axis-aligned plane -- a planar mesh patch -- give a box with no
    extent along that axis. With none, the grid's voxels there were ``2.2e-308`` thick: a
    crossing point a rounding off the plane was a voxel index past the range of an integer -- an
    overflow cast numpy warns about, and warnings are errors here -- and, with more than one
    voxel through the thickness, landed in a voxel holding nothing, so the segment read clear
    (110 of 48,000 such segments, measured). Voxel boundaries on the sheet's own face edges
    (3 x 4, 6 x 8) put seams on boundaries, and two voxels through the thickness put a voxel
    plane in the sheet.
    """
    sheet = placed(flat_sheet(), offset, axes)
    resolution = None if resolution is None else tuple(np.asarray(resolution)[list(axes)])
    origin, target = through_the_sheet_s_seams(sheet, 6000, np.random.default_rng(13))
    near = np.zeros(len(origin))
    grid = TriangleGrid.build(sheet, resolution=resolution)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        found = grid.blocks(origin, target, near)
    expected = brute(origin, target, sheet, near)
    assert 0.3 < expected.mean() < 0.9, f"fixture is one-sided: {expected.mean()}"
    np.testing.assert_array_equal(found, expected)


@pytest.mark.parametrize("resolution", [(3, 4, 2), (6, 8, 2), (6, 8, 4)])
@pytest.mark.parametrize(("offset", "axes"), SHEET_PLACEMENTS)
def test_a_sheet_lying_on_a_voxel_plane_inside_a_thick_box_is_found_from_either_side(
    resolution, offset, axes
):
    """The same kind of miss without a flat box: three parallel sheets a unit apart, and an even
    voxel count through the box's height, so the middle sheet lies exactly on a voxel plane. A
    segment through one of its seams crosses that plane and an in-plane voxel boundary at the
    same point, the walk's tie-break steps around one of the voxels meeting there, and a
    triangle registered only on the side its rounding put it is missed. Registering each
    triangle in the voxels its box spans widened by a rounding margin puts it on both sides.
    """
    sheet = flat_sheet()
    stack = placed(
        np.concatenate([sheet, sheet - [0.0, 0.0, 1.0], sheet - [0.0, 0.0, 2.0]]), offset, axes
    )
    middle = placed(sheet - [0.0, 0.0, 1.0], offset, axes)
    origin, target = through_the_sheet_s_seams(middle, 6000, np.random.default_rng(17))
    near = np.zeros(len(origin))
    resolution = tuple(np.asarray(resolution)[list(axes)])
    found = TriangleGrid.build(stack, resolution=resolution).blocks(origin, target, near)
    np.testing.assert_array_equal(found, brute(origin, target, stack, near))


def test_a_segment_parallel_to_an_axis_beside_the_grid_is_not_walked(monkeypatch):
    """Along an axis it does not move on, a segment lies in the grid's slab for its whole length
    or none of it. One running beside the grid's box was walked anyway -- across the voxels its
    shadow on the box falls on, testing triangles it cannot reach -- because a zero direction
    component was read as never leaving the slab rather than as never being in it.
    """
    walked = []
    real_walk = grid_module.walk_to_first_hit

    def recording_walk(ray, *args):
        walked.append(np.asarray(ray))
        return real_walk(ray, *args)

    monkeypatch.setattr(grid_module, "walk_to_first_hit", recording_walk)
    sheet = flat_sheet()
    origin = np.array([[-0.5, 0.7, 2.5], [-0.5, 0.7, 1.5], [-0.5, 0.7, 2.0], [-0.5, 0.7, 2.5]])
    target = np.array([[1.5, 0.8, 2.5], [1.5, 0.8, 1.5], [1.5, 0.8, 2.0], [0.5, 0.7, 1.5]])
    found = TriangleGrid.build(sheet).blocks(origin, target, np.zeros(4))
    assert found.tolist() == [False, False, False, True]
    # Above, below and in the sheet's plane; the last crosses it. Only the last two are in the slab.
    assert np.concatenate(walked).tolist() == [2, 3]
