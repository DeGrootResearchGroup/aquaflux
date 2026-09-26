"""Facet clusters: a partition of the surface into compact groups, bounded by spheres."""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.radiation.clusters import FacetClusters

from tests.unit.radiation_references import closed_drum, inward_box


def _scene():
    return np.concatenate([inward_box(4), closed_drum(12, 0.15, 0.3) + 0.5])


@pytest.mark.parametrize("size", [1, 7, 32, 1000])
def test_every_facet_is_in_exactly_one_cluster(size):
    """A cull that rejects cluster pairs whole must see every facet once: twice would count a
    pair twice, never would drop it. Sizes that divide the count and sizes that do not."""
    vertices = _scene()
    clusters = FacetClusters.build(vertices, size)
    members = clusters.members[clusters.members >= 0]
    assert np.array_equal(np.sort(members), np.arange(len(vertices)))
    assert clusters.members.shape[1] == size
    # Only the last cluster may be short.
    assert np.all(clusters.members[:-1] >= 0)


def test_each_sphere_contains_every_corner_of_its_cluster():
    """The sphere is what the source-plane bound rejects a whole cluster on, so a corner outside
    it is an occluder the cull could drop."""
    vertices = _scene()
    clusters = FacetClusters.build(vertices, 7)
    for c, row in enumerate(clusters.members):
        corners = vertices[row[row >= 0]].reshape(-1, 3)
        distance = np.linalg.norm(corners - clusters.centre[c], axis=-1)
        assert np.all(distance <= clusters.radius[c])


def test_clusters_are_compact_rather_than_arbitrary():
    """Grouping along a space-filling curve keeps neighbours together, which is the whole reason
    a cluster's bound is tight enough to reject anything. A random grouping of the same facets
    has spheres spanning the scene."""
    vertices = inward_box(16)
    clusters = FacetClusters.build(vertices, 16)
    rng = np.random.default_rng(3)
    arbitrary = rng.permutation(len(vertices))[: len(vertices) // 16 * 16].reshape(-1, 16)
    corners = vertices[arbitrary].reshape(len(arbitrary), -1, 3)
    arbitrary_radius = np.linalg.norm(corners - corners.mean(axis=1, keepdims=True), axis=-1)
    assert np.median(clusters.radius) < 0.25 * np.median(arbitrary_radius.max(axis=1))


def test_a_cluster_must_hold_something():
    with pytest.raises(ValueError, match="at least one facet"):
        FacetClusters.build(_scene(), 0)
