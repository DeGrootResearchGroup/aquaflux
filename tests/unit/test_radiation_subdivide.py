"""Refining facets against the width-over-distance criterion, and what must survive it."""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.radiation.checks import winding_report
from aquaflux.radiation.subdivide import refine_for_receivers, subdivide_to_width
from aquaflux.radiation.surfaces import Surfaces

SQUARE = np.array(
    [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    ]
)
NEAR_RECEIVER = np.array([[0.5, 0.5, 0.2]])
FAR_RECEIVER = np.array([[0.5, 0.5, 40.0]])


def test_a_facet_far_from_every_receiver_is_left_alone():
    """Refinement costs memory in the gather, so it must not fire where it buys nothing."""
    division = subdivide_to_width(SQUARE, FAR_RECEIVER, max_ratio=0.25)
    assert division.n_facets == 2
    np.testing.assert_array_equal(division.level, [0, 0])


def test_a_facet_close_to_a_receiver_is_refined_until_the_criterion_is_met():
    division = subdivide_to_width(SQUARE, NEAR_RECEIVER, max_ratio=0.25)
    assert division.n_facets > 2
    assert division.realized_ratio.max() <= 0.25
    assert len(division.unmet) == 0


@pytest.mark.parametrize("max_ratio", [0.25, 0.15, 0.05])
def test_a_tighter_ratio_refines_further(max_ratio):
    """The criterion has to be an actual dial, not a threshold that happens to fire once."""
    division = subdivide_to_width(SQUARE, NEAR_RECEIVER, max_ratio=max_ratio, max_levels=8)
    assert division.realized_ratio.max() <= max_ratio


def test_refinement_conserves_area_exactly():
    """Splitting at edge midpoints partitions a triangle; it must not lose or gain surface."""
    division = subdivide_to_width(SQUARE, NEAR_RECEIVER, max_ratio=0.25)
    before = Surfaces.from_triangles(SQUARE)
    after = Surfaces.from_triangles(division.vertices)
    assert float(np.sum(np.asarray(after.area))) == pytest.approx(
        float(np.sum(np.asarray(before.area))), rel=1e-14
    )


def test_every_child_keeps_its_parent_s_winding():
    """The middle child of a four-way split is the one whose winding is easy to get backwards.

    A reversed middle child points its normal into the solid, so a quarter of every refined
    facet would stop emitting — and the winding check would then blame the input file.
    """
    division = subdivide_to_width(SQUARE, NEAR_RECEIVER, max_ratio=0.25)
    normals = np.asarray(Surfaces.from_triangles(division.vertices).normal)
    np.testing.assert_allclose(normals, np.tile([0.0, 0.0, 1.0], (len(normals), 1)), atol=1e-14)
    assert winding_report(division.vertices).consistent


def test_each_split_produces_four_similar_children():
    division = subdivide_to_width(SQUARE, NEAR_RECEIVER, max_ratio=0.25, max_levels=1)
    assert division.n_facets == 8
    np.testing.assert_array_equal(np.unique(division.level), [1])


def test_the_level_cap_is_respected_and_the_shortfall_is_reported():
    """A receiver *on* a facet can never satisfy the criterion, so the cap must be honest.

    Silently iterating to exhaustion would multiply the facet count by four every round; the
    realized ratio is reported instead, so a build states the accuracy it actually reached.
    """
    on_the_surface = np.array([[0.25, 0.25, 0.0]])
    division = subdivide_to_width(SQUARE, on_the_surface, max_ratio=0.25, max_levels=3)
    assert division.level.max() == 3
    assert division.n_facets <= 2 * 4**3
    assert len(division.unmet) > 0


def test_properties_are_carried_onto_every_child():
    surfaces = Surfaces.from_triangles(
        SQUARE, solid_id=[0, 1], solid_names=("wall", "lamp"), emission=[1.0, 7.0], reflectance=0.4
    )
    refined, division = refine_for_receivers(surfaces, NEAR_RECEIVER, max_ratio=0.25)
    assert refined.n_facets == division.n_facets
    np.testing.assert_allclose(
        np.asarray(refined.emission), np.asarray(surfaces.emission)[division.origin]
    )
    np.testing.assert_allclose(np.asarray(refined.reflectance), 0.4)
    np.testing.assert_array_equal(
        np.asarray(refined.solid_id), np.asarray(surfaces.solid_id)[division.origin]
    )
    assert refined.solid_names == ("wall", "lamp")


def test_refining_a_set_carrying_radiant_power_is_refused():
    """Power is extensive: inherited unchanged it would be multiplied by the child count."""
    with_point_source = Surfaces.from_triangles(
        np.concatenate([SQUARE, np.zeros((1, 3, 3))]), power=[0.0, 0.0, 35.0]
    )
    with pytest.raises(ValueError, match="cannot carry radiant power"):
        refine_for_receivers(with_point_source, NEAR_RECEIVER)


def test_the_distance_is_measured_from_the_nearest_corner_and_not_the_centroid():
    """The case that separates the two measures, which most geometries do not.

    One triangle of longest edge ``sqrt(2)``, with a receiver placed beyond one corner at a
    distance chosen so the two measures straddle the threshold: from the centroid the ratio is
    0.237 and nothing is refined, from the nearest corner it is 0.267 and one split happens.
    A centroid distance is never smaller, so it can only ever refine less — and it under-refines
    exactly where the receiver is closest to the facet, which is where the error is worst.
    """
    triangle = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    beyond_a_corner = np.array([[6.3, 0.0, 0.0]])

    width = np.sqrt(2.0)
    from_corner = width / 5.3
    from_centroid = width / np.linalg.norm(beyond_a_corner[0] - triangle[0].mean(axis=0))
    assert from_centroid < 0.25 < from_corner, "the fixture no longer straddles the threshold"

    division = subdivide_to_width(triangle, beyond_a_corner, max_ratio=0.25, max_levels=4)
    assert division.n_facets == 4
    assert division.level.max() == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_ratio": 0.0}, "max_ratio must be positive"),
        ({"receivers": np.zeros((0, 3))}, "receivers is empty"),
        ({"receivers": np.zeros((2, 2))}, "receivers must have shape"),
    ],
)
def test_a_meaningless_request_is_refused(kwargs, message):
    receivers = kwargs.pop("receivers", NEAR_RECEIVER)
    with pytest.raises(ValueError, match=message):
        subdivide_to_width(SQUARE, receivers, **kwargs)
