"""A body given as triangles: what it blocks, what is inside it, and what it can vouch for.

Every answer here is checked against something computed another way -- the brute-force segment
test over every triangle, points sampled on the triangles themselves, a closed form for which
side of a wound surface is solid -- because the body's whole job is to give those answers
faster, never differently.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation import (
    EveryPair,
    NoOcclusion,
    ShaftCulling,
    Surfaces,
    TriangleBody,
    build_visibility,
)
from aquaflux.radiation.grid import TriangleGrid
from aquaflux.radiation.triangles import segment_is_cut
from aquaflux.solids import Box

from tests.unit.radiation_references import closed_drum, inward_box


def _vessel() -> np.ndarray:
    """A closed box from -1 to 1, wound to face in: a vessel, solid everywhere outside it."""
    return inward_box(4) * 2.0 - 1.0


def _sleeve() -> np.ndarray:
    """A closed drum round the z axis, wound to face out: a solid sleeve inside the vessel."""
    return closed_drum(24, radius=0.2, half_height=0.5)


def _water(rng, count: int, body: TriangleBody) -> np.ndarray:
    """Points in the vessel and out of the sleeve."""
    points = rng.uniform(-0.95, 0.95, (4 * count, 3))
    return points[~np.asarray(body.contains(points))][:count]


def test_a_segment_is_blocked_exactly_where_some_triangle_lies_across_it():
    """The grid walk against every triangle tested, over the broadcast pair shapes a mask uses."""
    triangles = np.concatenate([_vessel(), _sleeve()])
    body = TriangleBody.build(triangles)
    rng = np.random.default_rng(1)
    sources, receivers = _water(rng, 40, body), _water(rng, 30, body)
    blocked = np.asarray(body.blocks(sources[None, :, :], receivers[:, None, :], 1e-9))
    origin = np.broadcast_to(sources[None, :, :], (30, 40, 3)).reshape(-1, 3)
    target = np.broadcast_to(receivers[:, None, :], (30, 40, 3)).reshape(-1, 3)
    expected = np.asarray(
        segment_is_cut(
            jnp.asarray(origin), jnp.asarray(target), jnp.asarray(triangles), jnp.full(1200, 1e-9)
        )
    ).reshape(30, 40)
    assert 0.05 < expected.mean() < 0.95
    np.testing.assert_array_equal(blocked, expected)


def test_a_closed_piece_is_solid_on_the_side_its_normals_point_away_from():
    """Vessel wound in and sleeve wound out: the water is clear, the sleeve and the metal are solid.

    Three points, one in each region, and one more outside the vessel's bounding box -- where the
    winding is not even computed, so that shortcut must give the same answer the winding would.
    """
    body = TriangleBody.build(np.concatenate([_vessel(), _sleeve()]))
    assert body.inward_pieces == 1
    points = np.array([[0.6, 0.6, 0.0], [0.0, 0.05, 0.3], [1.0 - 1e-3, 0.0, 0.9], [3.0, 0.0, 0.0]])
    # Water, sleeve, water a millimetre inside the vessel's wall, metal beyond it.
    np.testing.assert_array_equal(np.asarray(body.contains(points)), [False, True, False, True])
    # And inside-out, from the winding alone: the same drum wound inward is a cavity.
    cavity = TriangleBody.build(_sleeve()[:, ::-1, :])
    np.testing.assert_array_equal(
        np.asarray(cavity.contains(np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))), [False, True]
    )


def test_an_open_piece_is_a_sheet_with_no_inside():
    """A vessel missing one face has a free edge: nothing is embedded in it, anywhere."""
    vessel = _vessel()
    open_top = vessel[~np.all(vessel[:, :, 2] > 1.0 - 1e-9, axis=1)]
    assert len(open_top) < len(vessel)
    body = TriangleBody.build(open_top)
    assert body.inward_pieces == 0 and len(body.enclosing) == 0
    points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    assert not np.asarray(body.contains(points)).any()


def test_the_caller_can_overrule_topology_either_way():
    """``sheet=True`` empties a closed body; ``sheet=False`` refuses an open one."""
    closed = TriangleBody.build(_vessel(), sheet=True)
    assert not np.asarray(closed.contains(np.array([[3.0, 0.0, 0.0]]))).any()
    open_top = _vessel()[~np.all(_vessel()[:, :, 2] > 1.0 - 1e-9, axis=1)]
    with pytest.raises(ValueError, match="declared closed"):
        TriangleBody.build(open_top, sheet=False)


def test_a_closed_piece_enclosing_nothing_is_refused():
    """A triangle and its reverse close each other's edges and enclose no volume."""
    triangle = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    flat = np.concatenate([triangle, triangle[:, ::-1, :]])
    with pytest.raises(ValueError, match="enclose no volume"):
        TriangleBody.build(flat)


def _on_triangles(triangles: np.ndarray, per: int, rng) -> np.ndarray:
    """Points spread over every triangle, including its edges and corners."""
    weights = rng.dirichlet(np.ones(3), (len(triangles), per))
    weights[:, :3] = np.eye(3)
    return np.einsum("tpk,tkd->tpd", weights, triangles).reshape(-1, 3)


def test_a_box_the_grid_calls_empty_holds_no_point_of_any_triangle():
    """Sound against points sampled on the triangles, and not vacuous: some boxes are empty.

    The boxes are small and large, inside the grid and straddling its edge; a box any sampled
    point lies in must be reported as holding something.
    """
    rng = np.random.default_rng(3)
    triangles = np.concatenate([_vessel(), _sleeve()])
    grid = TriangleGrid.build(triangles, resolution=24)
    samples = _on_triangles(triangles, 40, rng)
    centre = rng.uniform(-1.3, 1.3, (3000, 3))
    half = rng.uniform(0.005, 0.4, (3000, 3))
    low, high = centre - half, centre + half
    held = grid.holds_any(low, high)
    touched = np.array(
        [
            np.any(np.all((samples >= lo) & (samples <= hi), axis=1))
            for lo, hi in zip(low, high, strict=True)
        ]
    )
    assert not np.any(touched & ~held)
    assert np.count_nonzero(~held) > 300, np.count_nonzero(~held)


def test_a_body_vouches_only_for_sets_whose_segments_are_all_clear():
    """Random compact clouds in the water: where the body vouches, no segment among them is cut."""
    triangles = np.concatenate([_vessel(), _sleeve()])
    body = TriangleBody.build(triangles)
    rng = np.random.default_rng(4)
    centre = _water(rng, 400, body)
    clouds = centre[:, None, :] + rng.normal(scale=0.05, size=(len(centre), 6, 3))
    summary = np.asarray(body.clearance(clouds)).max(axis=1)
    vouched = np.asarray(body.vouches(summary))
    start, finish = clouds[:, [0, 1, 2]], clouds[:, [3, 4, 5]]
    cut = np.asarray(
        segment_is_cut(
            jnp.asarray(start.reshape(-1, 3)),
            jnp.asarray(finish.reshape(-1, 3)),
            jnp.asarray(triangles),
            jnp.zeros(start.size // 3),
        )
    ).reshape(len(clouds), 3)
    assert not np.any(cut[vouched])
    assert 0.2 * len(clouds) < np.count_nonzero(vouched) < len(clouds)


def test_a_primitive_and_a_triangle_body_shadow_one_scene_together():
    """Both kinds in one occluder list, each answering its own layer, culled or not.

    The primitive is compiled and the triangle body is not; a mask build that compiled both, or
    culled one with the other's certificate, would raise or differ here.
    """
    vessel = TriangleBody.build(_vessel())
    baffle = Box(centre=[0.4, 0.0, 0.0], half_sizes=[0.02, 0.5, 0.5])
    lamp = Surfaces.from_triangles(_sleeve(), emission=1.0)
    rng = np.random.default_rng(5)
    points = _water(rng, 300, vessel)
    points = points[~np.asarray(baffle.contains(jnp.asarray(points)))]
    points = points[np.linalg.norm(points[:, :2], axis=1) > 0.25]
    plain = build_visibility([baffle, vessel], lamp, points, self_occlusion=NoOcclusion())
    culled = build_visibility(
        [baffle, vessel], lamp, points, self_occlusion=NoOcclusion(), body_culling=ShaftCulling()
    )
    blocked = np.asarray(plain.blocked)
    assert blocked[0].any() and not blocked[0].all()  # the baffle shadows part of the scene
    np.testing.assert_array_equal(np.asarray(culled.blocked), blocked)


def test_refinement_vouches_for_tiles_the_coarse_level_could_not():
    """Splitting refused tiles certifies more of a triangulated vessel -- and changes no answer."""
    vessel = TriangleBody.build(_vessel())
    lamp = Surfaces.from_triangles(_sleeve(), emission=1.0)
    rng = np.random.default_rng(6)
    points = _water(rng, 600, vessel)
    points = points[np.linalg.norm(points[:, :2], axis=1) > 0.25]
    sources, near = np.asarray(lamp.centroid), np.full(lamp.n_facets, 1e-9)
    coarse = ShaftCulling(receiver_blocks=(32,), source_clusters=(32,))
    refined = ShaftCulling(receiver_blocks=(32, 8, 2), source_clusters=(32, 8, 2))
    once = coarse.certified_pairs([vessel], sources, points)[0]
    more = refined.certified_pairs([vessel], sources, points)[0]
    assert more > once, (more, once)
    reference = np.asarray(EveryPair().blocked([vessel], sources, near, points))
    np.testing.assert_array_equal(
        np.asarray(refined.blocked([vessel], sources, near, points)), reference
    )
    assert more <= np.count_nonzero(~reference)
