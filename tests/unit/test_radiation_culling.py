"""Shaft culling of the analytic-body mask: the same mask as testing every pair, for fewer tests.

The whole claim is an equality -- :class:`ShaftCulling` returns, bit for bit, what
:class:`EveryPair` returns -- so every test here builds a scene where that equality has
something to lose: real shadows from every body kind, receivers on both sides of a shadow edge,
groups that do not divide the point counts, and batches cut small enough to be padded. Each also
checks that the culling *did* decide something, since an equality reached by testing every pair
anyway would pass while saving nothing.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation import (
    EveryPair,
    NoOcclusion,
    RadiationSettings,
    ShaftCulling,
    Surfaces,
    build_visibility,
)
from aquaflux.radiation.culling import _Curve, spatial_order
from aquaflux.solids import Body, Box, Cylinder, Difference, Outside, Sphere

#: The chamber is a cylinder along x, radius 0.1, from x = 0 to 1; the pipe stands on it at x = 0.2.
CHAMBER = Cylinder(centre=[0.5, 0.0, 0.0], axis=[1.0, 0.0, 0.0], radius=0.1, half_length=0.5)
PIPE = Cylinder(centre=[0.2, 0.0, 0.5], axis=[0.0, 0.0, 1.0], radius=0.02, half_length=0.45)


def _lamp(n_along: int = 24, n_around: int = 12) -> Surfaces:
    """A thin tube of facets on the chamber's axis, from x = 0.3 to 0.9, wound outward."""
    x = np.linspace(0.3, 0.9, n_along + 1)
    angle = np.linspace(0.0, 2.0 * np.pi, n_around, endpoint=False)
    ring = 0.01 * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    faces = []
    for i in range(n_along):
        for j in range(n_around):
            k = (j + 1) % n_around
            a, b = [x[i], *ring[j]], [x[i], *ring[k]]
            c, d = [x[i + 1], *ring[k]], [x[i + 1], *ring[j]]
            faces += [[a, b, c], [a, c, d]]
    return Surfaces.from_triangles(np.asarray(faces), emission=1.0)


def _bodies() -> list[Body]:
    """One of each kind a scene holds, each placed to shadow part of the receivers."""
    return [
        Outside(CHAMBER, PIPE),
        Box(centre=[0.6, 0.0, 0.06], half_sizes=[0.01, 0.1, 0.02]),
        Sphere(centre=[0.15, 0.05, -0.05], radius=0.02),
        Difference(
            Cylinder([0.95, 0.0, 0.0], [1.0, 0.0, 0.0], 0.06, 0.02),
            Cylinder([0.95, 0.0, 0.0], [1.0, 0.0, 0.0], 0.04, 0.05),
        ),
    ]


def _receivers(count: int, seed: int = 3) -> np.ndarray:
    """Points in the water -- chamber and pipe -- outside every solid body."""
    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    while sum(len(block) for block in kept) < count:
        trial = rng.uniform([0.0, -0.1, -0.1], [1.0, 0.1, 0.95], (8 * count, 3))
        inside = np.zeros(len(trial), dtype=bool)
        for body in _bodies():
            inside |= np.asarray(body.contains(jnp.asarray(trial)))
        kept.append(trial[~inside])
    return np.concatenate(kept)[:count]


def _arms(sources, near, receivers, culling, bodies=None, pair_limit=4_000_000):
    """The reference mask and the culled one, for the same scene."""
    bodies = _bodies() if bodies is None else bodies
    reference = np.asarray(EveryPair().blocked(bodies, sources, near, receivers, pair_limit))
    culled = np.asarray(culling.blocked(bodies, sources, near, receivers, pair_limit))
    return reference, culled


@pytest.mark.parametrize(
    ("blocks", "clusters"),
    [
        ((32,), (32,)),
        ((7,), (5,)),
        ((1,), (1,)),
        ((64,), (3,)),
        ((32, 8), (32, 8)),
        ((24, 6, 1), (8, 4, 2)),
    ],
)
def test_the_culled_mask_is_the_mask_every_pair_gives(blocks, clusters):
    """Bit for bit, with group sizes that divide nothing and one that makes every pair a tile.

    Every body blocks some pairs and leaves others clear, and at the default sizes each has tiles
    certified -- so a certificate that vouched for a shadowed tile, or a tile written back to the
    wrong pairs, shows as a differing entry.
    """
    lamp = _lamp()
    near = 1e-6 * np.sqrt(np.asarray(lamp.area))
    receivers = _receivers(500)
    culling = ShaftCulling(receiver_blocks=blocks, source_clusters=clusters)
    reference, culled = _arms(lamp.centroid, near, receivers, culling)
    assert reference.any(axis=(1, 2)).all() and not reference.all(axis=(1, 2)).any()
    np.testing.assert_array_equal(culled, reference)
    certified = culling.certified_pairs(_bodies(), lamp.centroid, receivers)
    assert np.all(certified > 0), certified


def test_the_pairs_certified_are_all_clear_and_are_most_of_the_open_ones():
    """What is counted as saved is a subset of what is clear, and on this scene a large one.

    Every pair the certificate vouches for must be clear in the reference, so the count can
    never exceed the clear count; and the chamber is most of the water, so for the fluid the
    bulk of its clear pairs should be vouched for rather than tested.
    """
    lamp = _lamp()
    near = 1e-6 * np.sqrt(np.asarray(lamp.area))
    receivers = _receivers(500)
    reference, _ = _arms(lamp.centroid, near, receivers, ShaftCulling())
    certified = ShaftCulling().certified_pairs(_bodies(), lamp.centroid, receivers)
    clear = (~reference).sum(axis=(1, 2))
    assert np.all(certified <= clear), (certified, clear)
    assert certified[0] > 0.5 * clear[0], (certified[0], clear[0])


def test_an_open_chamber_is_certified_whole_and_counted_without_its_padding():
    """Receivers and sources all in one convex region: every pair vouched for, and only those.

    Neither count divides by the group size, so the last group of each is padded; a count that
    included the padding would exceed the number of pairs there are.
    """
    rng = np.random.default_rng(0)
    receivers = rng.uniform([0.1, -0.05, -0.05], [0.9, 0.05, 0.05], (101, 3))
    sources = rng.uniform([0.1, -0.05, -0.05], [0.9, 0.05, 0.05], (37, 3))
    culling = ShaftCulling(receiver_blocks=(16, 4), source_clusters=(8, 2))
    assert culling.certified_pairs([Outside(CHAMBER)], sources, receivers).tolist() == [101 * 37]
    reference, culled = _arms(sources, np.zeros(37), receivers, culling, bodies=[Outside(CHAMBER)])
    assert not reference.any()
    np.testing.assert_array_equal(culled, reference)


def test_batches_cut_to_one_tile_and_padded_still_give_the_same_mask():
    """A pair limit smaller than one tile: every batch is one tile, the last ones padded."""
    lamp = _lamp(12, 8)
    near = 1e-6 * np.sqrt(np.asarray(lamp.area))
    receivers = _receivers(200)
    culling = ShaftCulling(receiver_blocks=(8,), source_clusters=(8,))
    reference, culled = _arms(lamp.centroid, near, receivers, culling, pair_limit=50)
    assert reference.any()
    np.testing.assert_array_equal(culled, reference)


class _Opaque(Body):
    """A sphere answered eagerly with no witnesses: a body that cannot vouch for anything."""

    sphere: Sphere

    def blocks(self, origin, target, min_distance):
        return self.sphere.blocks(origin, target, min_distance)

    def contains(self, position):
        return self.sphere.contains(position)


def test_a_body_with_no_witnesses_is_tested_everywhere_and_certified_nowhere():
    """Not traceable and no clearance: every tile falls back, and the mask is still right."""
    lamp = _lamp(12, 8)
    near = 1e-6 * np.sqrt(np.asarray(lamp.area))
    receivers = _receivers(150)
    body = _Opaque(Sphere(centre=[0.15, 0.05, -0.05], radius=0.02))
    culling = ShaftCulling(receiver_blocks=(8,), source_clusters=(8,))
    reference, culled = _arms(lamp.centroid, near, receivers, culling, bodies=[body])
    assert reference.any()
    np.testing.assert_array_equal(culled, reference)
    assert culling.certified_pairs([body], lamp.centroid, receivers).tolist() == [0]


def test_a_built_mask_and_a_model_setting_reach_the_culling():
    """``build_visibility`` takes the strategy, and ``RadiationSettings`` hands it to both masks."""
    lamp = _lamp(12, 8)
    receivers = _receivers(150)
    plain = build_visibility(_bodies(), lamp, receivers, self_occlusion=NoOcclusion())
    culled = build_visibility(
        _bodies(), lamp, receivers, self_occlusion=NoOcclusion(), body_culling=ShaftCulling()
    )
    np.testing.assert_array_equal(np.asarray(culled.blocked), np.asarray(plain.blocked))
    settings = RadiationSettings(body_culling=ShaftCulling(), receiver_occlusion=NoOcclusion())
    assert settings.visibility_options() == {"body_culling": ShaftCulling()}
    assert settings.receiver_visibility_options() == {
        "body_culling": ShaftCulling(),
        "self_occlusion": NoOcclusion(),
    }


def test_a_group_size_ladder_that_cannot_be_refined_is_refused():
    """Sizes must be positive, nest, and come one of each per level."""
    with pytest.raises(ValueError, match="receiver_blocks sizes must be at least 1"):
        ShaftCulling(receiver_blocks=(0,), source_clusters=(8,))
    with pytest.raises(ValueError, match="source_clusters sizes must each divide"):
        ShaftCulling(receiver_blocks=(32, 8), source_clusters=(32, 12))
    with pytest.raises(ValueError, match="as many source cluster sizes"):
        ShaftCulling(receiver_blocks=(32, 8), source_clusters=(32,))
    with pytest.raises(ValueError, match="needs at least one group size"):
        ShaftCulling(receiver_blocks=(), source_clusters=())


def test_the_curve_visits_a_cube_s_corners_in_z_order():
    """The eight corners of a box, shuffled, come back ordered by ``x + 2y + 4z``.

    That is the interleaving -- x in the lowest bit -- stated at the one resolution where it can
    be read off by hand.
    """
    corners = np.array(list(np.ndindex(2, 2, 2)), dtype=float)[:, ::-1]  # rows as (x, y, z)
    shuffled = corners[np.random.default_rng(2).permutation(8)]
    ordered = shuffled[spatial_order(shuffled)]
    keys = ordered @ np.array([1.0, 2.0, 4.0])
    assert keys.tolist() == list(range(8))


def test_the_curve_keeps_neighbours_together():
    """Consecutive points along the curve are far closer than consecutive points of a shuffle."""
    rng = np.random.default_rng(4)
    points = rng.uniform(0.0, 1.0, (4096, 3))
    order = spatial_order(points)
    assert sorted(order.tolist()) == list(range(4096))
    step = np.linalg.norm(np.diff(points[order], axis=0), axis=1).mean()
    shuffled = np.linalg.norm(np.diff(points, axis=0), axis=1).mean()
    assert step < 0.2 * shuffled, (step, shuffled)


def test_the_last_group_is_padded_with_its_own_last_member_at_every_size():
    """Ten points padded to groups of four, then read in twos: the padding is a real point.

    At the coarse size the counts are four, four, two; at the fine size the sixth group is all
    padding and holds no real point, which is what lets refinement drop it rather than ask it.
    """
    points = np.random.default_rng(6).uniform(0.0, 1.0, (10, 3))
    curve = _Curve.of(points, 4)
    assert curve.counts(4).tolist() == [4, 4, 2]
    assert curve.counts(2).tolist() == [2, 2, 2, 2, 2, 0]
    members = curve.members(4)
    assert sorted(members.ravel()[:10].tolist()) == list(range(10))
    assert members[2, 2] == members[2, 1] == members[2, 3]
