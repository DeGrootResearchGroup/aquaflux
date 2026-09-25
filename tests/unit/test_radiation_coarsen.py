"""Coarsening a dense surface under size, chord and angle bounds, and what must survive it."""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.radiation.checks import winding_report
from aquaflux.radiation.coarsen import (
    _point_triangle_distance,
    coarsen_surfaces,
    coarsen_to_size,
)
from aquaflux.radiation.surfaces import Surfaces


def grid(n: int, width: float = 1.0, height: float = 1.0) -> np.ndarray:
    """A flat ``width x height`` rectangle in ``z = 0`` as ``2 n^2`` triangles facing ``+z``."""
    x = np.linspace(0.0, width, n + 1)
    y = np.linspace(0.0, height, n + 1)
    triangles = []
    for i in range(n):
        for j in range(n):
            a, b = (x[i], y[j], 0.0), (x[i + 1], y[j], 0.0)
            c, d = (x[i + 1], y[j + 1], 0.0), (x[i], y[j + 1], 0.0)
            triangles += [[a, b, c], [a, c, d]]
    return np.array(triangles)


RADIUS, LENGTH = 0.01, 0.03


def open_cylinder(sectors: int = 60, slices: int = 30, length: float = LENGTH) -> np.ndarray:
    """A tube of radius 10 mm along ``z``, open at both ends, wound outward."""
    angle = np.linspace(0.0, 2.0 * np.pi, sectors, endpoint=False)
    z = np.linspace(0.0, length, slices + 1)
    triangles = []
    for k in range(slices):
        for i in range(sectors):
            j = (i + 1) % sectors
            a = (RADIUS * np.cos(angle[i]), RADIUS * np.sin(angle[i]), z[k])
            b = (RADIUS * np.cos(angle[j]), RADIUS * np.sin(angle[j]), z[k])
            c = (RADIUS * np.cos(angle[j]), RADIUS * np.sin(angle[j]), z[k + 1])
            d = (RADIUS * np.cos(angle[i]), RADIUS * np.sin(angle[i]), z[k + 1])
            triangles += [[a, b, c], [a, c, d]]
    return np.array(triangles)


def area(vertices: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(
        np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=1
    )


@pytest.fixture(scope="module")
def cylinder():
    tube = open_cylinder()
    return tube, coarsen_to_size(tube, max_edge=4e-3, chord=5e-5)


def test_the_distance_to_a_triangle_is_to_its_face_its_edge_or_its_corner():
    triangle = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    points = np.array([[0.2, 0.2, 0.5], [0.5, -1.0, 0.0], [-3.0, -4.0, 0.0], [1.0, 1.0, 0.0]])
    expected = [0.5, 1.0, 5.0, np.sqrt(0.5)]
    np.testing.assert_allclose(
        _point_triangle_distance(points[:, None], triangle[None])[:, 0], expected
    )


def test_a_flat_surface_coarsens_far_and_keeps_its_area_exactly():
    fine = grid(20)
    coarse = coarsen_to_size(fine, max_edge=0.3, chord=1e-6)
    assert coarse.n_facets < len(fine) / 8
    assert area(coarse.vertices).sum() == pytest.approx(1.0, rel=1e-12)
    assert coarse.longest_edge.max() <= 0.3
    unit = np.cross(
        coarse.vertices[:, 1] - coarse.vertices[:, 0], coarse.vertices[:, 2] - coarse.vertices[:, 0]
    )
    assert np.all(unit[:, 2] > 0.0)  # every facet still faces +z: none turned over
    assert len(winding_report(coarse.vertices).conflicting_edges) == 0


def test_every_coarse_vertex_is_an_input_vertex(cylinder):
    tube, coarse = cylinder
    inputs = {tuple(np.round(p, 12)) for p in tube.reshape(-1, 3)}
    assert all(tuple(np.round(p, 12)) in inputs for p in coarse.vertices.reshape(-1, 3))


def test_the_bounds_hold_measured_independently_of_the_bookkeeping(cylinder):
    """The chord is re-measured from scratch: every input vertex against the whole coarse surface."""
    tube, coarse = cylinder
    distance = _point_triangle_distance(
        np.unique(tube.reshape(-1, 3), axis=0)[:, None], coarse.vertices[None]
    )
    assert distance.min(axis=1).max() <= 5e-5
    assert coarse.chord.max() <= 5e-5
    assert coarse.longest_edge.max() <= 4e-3 * (1 + 1e-12)
    assert coarse.n_facets < len(tube) / 2
    assert len(winding_report(coarse.vertices).conflicting_edges) == 0


def test_a_tighter_chord_keeps_more_facets():
    tube = open_cylinder(sectors=40, slices=8, length=0.01)
    loose = coarsen_to_size(tube, max_edge=4e-3, chord=2e-4)
    tight = coarsen_to_size(tube, max_edge=4e-3, chord=2e-5)
    assert loose.chord.max() <= 2e-4
    assert tight.chord.max() <= 2e-5
    assert tight.n_facets > loose.n_facets


def test_an_open_rim_stays_where_it_was(cylinder):
    """A rim vertex may slide along the rim but never leave it, so both ends stay flat."""
    _, coarse = cylinder
    report = winding_report(coarse.vertices)
    assert report.boundary_edges > 0
    z = coarse.vertices[..., 2]
    radial = np.hypot(coarse.vertices[..., 0], coarse.vertices[..., 1])
    np.testing.assert_allclose(radial, RADIUS, rtol=1e-12)  # on the tube, a corollary of subsets
    # Each end is still a closed loop of edges in its own plane: every rim vertex keeps z = 0 or
    # z = L, and the ends' extent is the full circle.
    for end in (0.0, LENGTH):
        ring = coarse.vertices[np.isclose(z, end)]
        assert np.ptp(ring[:, 0]) == pytest.approx(2 * RADIUS, rel=0.02)
    assert coarse.area_after[0] == pytest.approx(coarse.area_before[0], rel=2e-3)


def test_with_a_loose_chord_the_outline_is_held_by_the_feature_rule_alone():
    """A chord of a tenth of the square would let its rim drift inward, and its corners be cut.

    Only the feature rule holds them: a rim vertex slides along the rim, and a corner, where the
    rim turns, stays put. So the area stays exactly one and all four corners survive.
    """
    coarse = coarsen_to_size(grid(16), max_edge=0.6, chord=0.1)
    assert area(coarse.vertices).sum() == pytest.approx(1.0, rel=1e-12)
    corners = {tuple(p) for p in np.round(coarse.vertices.reshape(-1, 3), 12)}
    assert {(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)} <= corners
    report = winding_report(coarse.vertices)
    assert report.boundary_edges > 4  # the rim was coarsened along itself, not left alone


def test_the_angle_bounds_a_curved_surface_where_the_chord_would_not():
    """A loose chord on a tube: only the angle stops a facet spanning a wide arc.

    Measured independently of the coarsener's bookkeeping: each input facet against the coarse
    triangle nearest its centroid.
    """
    tube = open_cylinder(sectors=40, slices=8, length=0.01)
    coarse = coarsen_to_size(tube, max_edge=0.02, chord=2e-3, angle=0.2)
    centroid = tube.mean(axis=1)
    nearest = _point_triangle_distance(centroid[:, None], coarse.vertices[None]).argmin(axis=1)

    def unit(v):
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        return n / np.linalg.norm(n, axis=1)[:, None]

    cosine = np.sum(unit(tube) * unit(coarse.vertices)[nearest], axis=1)
    assert np.arccos(np.clip(cosine, -1.0, 1.0)).max() <= 0.2 + 1e-9
    assert coarse.angle.max() <= 0.2 + 1e-9


def test_the_line_between_two_bodies_is_kept():
    """Two halves of one plane, as two bodies: no coarse facet may straddle x = 0.5."""
    fine = grid(20)
    body = (fine.mean(axis=1)[:, 0] > 0.5).astype(int)
    coarse = coarsen_to_size(fine, max_edge=0.3, chord=1e-6, solid_id=body)
    x = coarse.vertices[..., 0]
    left = coarse.solid_id == 0
    assert np.all(x[left] <= 0.5 + 1e-12)
    assert np.all(x[~left] >= 0.5 - 1e-12)
    np.testing.assert_allclose(coarse.area_after, [0.5, 0.5], rtol=1e-12)
    np.testing.assert_allclose(coarse.area_before, [0.5, 0.5], rtol=1e-12)


def test_a_crease_sharper_than_the_angle_is_kept():
    """A strip folded at 90 degrees along x = 0.5: every coarse facet lies in one of its planes."""
    flat = grid(16, width=1.0, height=0.5)
    folded = flat.copy()
    beyond = flat[..., 0] > 0.5
    folded[..., 2] = np.where(beyond, flat[..., 0] - 0.5, 0.0)
    folded[..., 0] = np.where(beyond, 0.5, flat[..., 0])
    coarse = coarsen_to_size(folded, max_edge=0.3, chord=1e-6)
    normal = np.cross(
        coarse.vertices[:, 1] - coarse.vertices[:, 0], coarse.vertices[:, 2] - coarse.vertices[:, 0]
    )
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    in_a_plane = np.isclose(np.abs(normal[:, 2]), 1.0) | np.isclose(np.abs(normal[:, 0]), 1.0)
    assert np.all(in_a_plane)
    assert coarse.angle.max() < 1e-9
    assert coarse.n_facets < len(folded) / 4


def test_the_surface_set_keeps_each_bodys_power_and_carries_reflectance():
    tube = open_cylinder(sectors=40, slices=8, length=0.01)
    half = (tube.mean(axis=1)[:, 2] > 0.005).astype(int)
    fine = Surfaces.from_triangles(tube, solid_id=half, solid_names=("lamp", "sleeve"))
    fine = fine.with_optics(
        emission=fine.per_facet({"lamp": 700.0}, default=0.0),
        reflectance=fine.per_facet({"sleeve": 0.3}, default=0.0),
    )
    coarse, record = coarsen_surfaces(fine, max_edge=4e-3, chord=5e-5)

    def power(surfaces):
        return np.bincount(
            np.asarray(surfaces.solid_id),
            weights=np.asarray(surfaces.emission) * np.asarray(surfaces.area),
            minlength=2,
        )

    np.testing.assert_allclose(power(coarse), power(fine), rtol=1e-12)
    # The area did fall, so the exitance must have been raised to make the power come out right.
    assert record.area_after[0] < record.area_before[0]
    assert float(np.asarray(coarse.emission)[np.asarray(coarse.solid_id) == 0].max()) > 700.0
    sleeve = np.asarray(coarse.solid_id) == 1
    np.testing.assert_array_equal(np.asarray(coarse.reflectance)[sleeve], 0.3)
    assert coarse.solid_names == ("lamp", "sleeve")


def test_a_body_whose_optics_vary_is_refused():
    fine = Surfaces.from_triangles(grid(4), emission=np.linspace(1.0, 2.0, 32))
    with pytest.raises(ValueError, match="emission varies"):
        coarsen_surfaces(fine, max_edge=0.5, chord=1e-6)


def test_point_sources_are_refused():
    point = np.zeros((1, 3, 3))
    fine = Surfaces.from_triangles(np.concatenate([grid(2), point]), power=0.0)
    with pytest.raises(ValueError, match="point sources"):
        coarsen_surfaces(fine, max_edge=0.5, chord=1e-6)


@pytest.mark.parametrize("bound", ["max_edge", "chord", "angle"])
def test_a_bound_must_be_positive(bound):
    settings = {"max_edge": 0.3, "chord": 1e-6, "angle": 0.5} | {bound: 0.0}
    with pytest.raises(ValueError, match=f"{bound} must be positive"):
        coarsen_to_size(grid(2), **settings)
