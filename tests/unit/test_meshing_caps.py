"""Caps sized to the room they have (``ClearedCaps``), and the rays that find that room.

Checked inside unions of unit cubes whose faces are cut into 4 x 4 quadrilaterals, the fill inside
(the surface wound into it): each quadrilateral's cap against the height its own rule gives by hand
-- its share of the edge, a side's lean over a concave or a convex edge, or the gap across.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.meshing import ClearedCaps, OctreeCore, Polygons, PyramidCaps, SampledRegion
from aquaflux.meshing.caps import capped
from aquaflux.meshing.regions import first_hits

from tests.support.meshes import voxel_surface

N = 4  # quadrilaterals across each cube face, each a quarter wide
HALF = 0.5 / N  # a quadrilateral's centre to its edge


def _into(voxels):
    """The union's surface, wound into it, and its points."""
    points, quads = voxel_surface(voxels, N)
    return points, Polygons(quads=np.asarray(quads)[:, ::-1])


def _touching_edges(points, quads, axis_lines):
    """Which quadrilaterals have a corner on one of ``axis_lines`` (``(point, direction)``)."""
    corner = points[quads]
    on = np.zeros(quads.shape[0], dtype=bool)
    for origin, direction in axis_lines:
        offset = corner - np.asarray(origin, float)
        across = offset - (offset @ np.asarray(direction, float))[..., None] * direction
        on |= np.any(np.linalg.norm(across, axis=2) < 1e-9, axis=1)
    return on


def test_a_cubes_caps_lean_a_share_of_half_the_fills_angle_at_each_fold():
    """Inside a cube every edge folds concavely at a right angle: a quadrilateral beside one may
    lean ``fold`` times 45 degrees over it, one in the middle of a face ``fold`` times 90."""
    points, surface = _into([(0, 0, 0)])
    rises = ClearedCaps(target=0.7, fold=0.6, gap=0.4).rises(points, surface)
    corner = points[surface.quads]
    at_edge = (np.isclose(corner, 0.0) | np.isclose(corner, 1.0)).sum(axis=(1, 2)) > 4
    np.testing.assert_allclose(rises[at_edge], HALF * np.tan(0.6 * np.pi / 4), rtol=1e-12)
    np.testing.assert_allclose(rises[~at_edge], HALF * np.tan(0.6 * np.pi / 2), rtol=1e-12)
    assert at_edge.sum() == 6 * 12 and (~at_edge).sum() == 6 * 4


def test_a_convex_fold_does_not_hold_a_cap_down():
    """In an L of three cubes the fill turns round the inner edge (270 degrees there): the caps
    beside it reach their target, as on a flat face, where a concave reading would hold them to
    the right-angle lean."""
    voxels = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    points, surface = _into(voxels)
    sizing = ClearedCaps(target=0.7, fold=0.9, gap=0.4)
    rises = sizing.rises(points, surface)
    inner = _touching_edges(points, surface.quads, [((1, 1, 0), np.array([0.0, 0.0, 1.0]))])
    # Away from the box's own edges: the quadrilaterals on the inner edge's faces, mid-height.
    corner = points[surface.quads]
    mid = np.all((corner[..., 2] > 0.2) & (corner[..., 2] < 0.8), axis=1)
    chosen = inner & mid
    assert chosen.sum() == 4
    np.testing.assert_allclose(rises[chosen], 0.7 * 2 * HALF, rtol=1e-12)
    assert np.all(HALF * np.tan(0.9 * np.pi / 4) < 0.7 * 2 * HALF)  # the concave bound would bind


def test_a_cap_takes_its_share_of_the_room_before_the_surface_opposite():
    """Across a cube the opposite face is one away; ``gap`` 0.1 of it binds every middle cap, and
    a ``clearance`` closer still binds instead."""
    points, surface = _into([(0, 0, 0)])
    corner = points[surface.quads]
    middle = ~((np.isclose(corner, 0.0) | np.isclose(corner, 1.0)).sum(axis=(1, 2)) > 4)
    np.testing.assert_allclose(ClearedCaps(gap=0.1).rises(points, surface)[middle], 0.1, rtol=1e-12)
    np.testing.assert_allclose(
        ClearedCaps(gap=0.1).rises(points, surface, clearance=0.5)[middle], 0.05, rtol=1e-12
    )


def test_caps_sized_to_their_room_stand_where_their_rises_say():
    points, surface = _into([(0, 0, 0)])
    sizing = ClearedCaps()
    caps = capped(points, surface, sizing)
    corner = points[surface.quads]
    normal = surface.normals(points)
    height = np.einsum("ij,ij->i", caps.apexes - corner.mean(axis=1), normal)
    np.testing.assert_allclose(height, sizing.rises(points, surface), rtol=1e-12)
    np.testing.assert_array_equal(caps.pyramids[:, :4], surface.quads)


def test_caps_given_a_share_are_the_caps_on_the_first_edge():
    points, surface = _into([(0, 0, 0)])
    np.testing.assert_array_equal(
        capped(points, surface, 0.3).apexes, PyramidCaps.on(points, surface.quads, 0.3).apexes
    )


def test_a_ray_meets_the_surface_again_where_the_face_across_is():
    """From each face's middle quadrilaterals inward: one across a unit cube; inf beyond reach; and
    a ray told to skip the opposite face's points goes on to nothing."""
    points, surface = _into([(0, 0, 0)])
    centre = points[surface.quads].mean(axis=1)
    normal = surface.normals(points)
    reach = np.full(centre.shape[0], 2.0)
    np.testing.assert_allclose(
        first_hits(points, surface, centre, normal, reach, surface.quads), 1.0, rtol=1e-12
    )
    assert np.all(np.isinf(first_hits(points, surface, centre, normal, 0.5 * reach / 2,
                                      surface.quads)))  # fmt: skip
    everything = np.tile(np.arange(points.shape[0]), (centre.shape[0], 1))
    assert np.all(np.isinf(first_hits(points, surface, centre, normal, reach, everything)))


def test_an_octree_cores_caps_sized_to_their_room_keep_inside_the_clearance():
    """The core's caps take at most ``gap`` of its clearance, and stand higher than the shipped
    quarter of an edge on the core's flat faces."""

    def box(points):
        return np.all((points > 0.0) & (points < [2.0, 1.5, 1.0]), axis=1)

    sizing = ClearedCaps(target=0.7, fold=0.6, gap=0.4)
    core = OctreeCore(size=0.25, clearance=0.2).mesh(
        SampledRegion(box), [0, 0, 0], [2.0, 1.5, 1.0], cap_height=sizing, boundary_size=0.25
    )
    base = core.points[core.caps[:, :4]]
    normal = Polygons(quads=core.caps[:, :4]).normals(core.points)
    height = np.einsum("ij,ij->i", core.points[core.caps[:, 4]] - base.mean(axis=1), normal)
    assert np.all(height <= 0.4 * 0.2 + 1e-12)
    assert np.max(height) > 0.25 * 0.25
    assert np.min(height) > 0.0


@pytest.mark.parametrize("field, value", [("fold", 1.0), ("gap", 0.0), ("target", 0.0)])
def test_sizing_out_of_range_is_refused(field, value):
    with pytest.raises(ValueError, match=f"ClearedCaps.{field}"):
        ClearedCaps(**{field: value})
