"""Quadrature rules on a triangle.

Everything here is a property of the tables themselves. What they buy on the integral they
exist for — the transfer between two facets — is measured in ``test_radiation_radiosity.py``,
because that needs the kernel and not just the points.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from aquaflux.radiation import TriangleQuadrature, triangle_quadrature
from aquaflux.radiation.quadrature import _RULES

SIZES = sorted(_RULES)


@pytest.mark.parametrize("n_points", SIZES)
def test_the_weights_sum_to_one(n_points):
    """Not a convention — it is what keeps the transfer matrix's row sums exact.

    Each quadrature point of a receiver inside a closed enclosure sees a full hemisphere of
    facets, so its own row sums to one. The average of those rows is one only if the weights
    are. A rule normalized to the triangle's area instead would scale every row by that area.
    """
    rule = triangle_quadrature(n_points)
    assert sum(rule.weight) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("n_points", SIZES)
def test_every_point_is_strictly_inside_the_triangle(n_points):
    """A point on an edge of one facet sits on the surface of its neighbour in a closed
    enclosure, where the transfer integrand diverges."""
    barycentric = np.asarray(triangle_quadrature(n_points).barycentric)
    np.testing.assert_allclose(barycentric.sum(axis=1), 1.0, atol=1e-14)
    assert barycentric.min() > 0.0


@pytest.mark.parametrize("n_points", SIZES)
def test_every_weight_is_strictly_positive(n_points):
    """A negative weight can drive a transfer factor below zero, which is outside the range the
    radiosity system's conditioning argument assumes. It is why the standard four-point
    degree-three rule is not catalogued."""
    assert min(triangle_quadrature(n_points).weight) > 0.0


def _average(rule, exponents):
    """Average of ``x**a * y**b`` over the unit triangle, by the rule."""
    a, b = exponents
    barycentric = np.asarray(rule.barycentric)
    # Barycentric coordinates against vertices (0, 0), (1, 0), (0, 1) give x, y directly.
    return float(np.sum(np.asarray(rule.weight) * barycentric[:, 1] ** a * barycentric[:, 2] ** b))


def _exact(exponents):
    """The same average in closed form: ``2 a! b! / (a + b + 2)!`` over a triangle of area 1/2."""
    a, b = exponents
    return 2.0 * math.factorial(a) * math.factorial(b) / math.factorial(a + b + 2)


@pytest.mark.parametrize("n_points", SIZES)
def test_each_rule_integrates_polynomials_up_to_its_stated_degree(n_points):
    rule = triangle_quadrature(n_points)
    for total in range(rule.degree + 1):
        for a in range(total + 1):
            exponents = (a, total - a)
            assert _average(rule, exponents) == pytest.approx(_exact(exponents), abs=1e-13), (
                exponents
            )


@pytest.mark.parametrize("n_points", SIZES)
def test_no_rule_reaches_one_degree_higher_than_it_claims(n_points):
    """Without this the degree field could say anything and every test above would still pass.

    Only *some* monomial of the next degree need fail: a symmetric rule integrates several of
    them exactly by cancellation, which is why this asks for one rather than for all.
    """
    rule = triangle_quadrature(n_points)
    total = rule.degree + 1
    errors = [
        abs(_average(rule, (a, total - a)) - _exact((a, total - a))) for a in range(total + 1)
    ]
    assert max(errors) > 1e-10, f"{n_points}-point rule is better than degree {rule.degree}"


@pytest.mark.parametrize("n_points", SIZES)
def test_a_rule_does_not_depend_on_which_vertex_a_triangle_is_written_from(n_points):
    """The rules are symmetric, so two facets meeting at an edge are sampled the same way
    whichever order their vertices happen to be stored in. An asymmetric rule would make the
    transfer matrix depend on the triangulation's bookkeeping."""
    rule = triangle_quadrature(n_points)
    triangle = np.array([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [0.4, 0.9, 0.0]])
    weight = np.asarray(rule.weight)

    def integral(order):
        points = np.asarray(rule.points(triangle[None, order]))[0]
        # A deliberately lopsided integrand: a symmetric one would agree for the wrong reason.
        return float(np.sum(weight * np.exp(points[:, 0]) * (1.0 + points[:, 1]) ** 3))

    reference = integral([0, 1, 2])
    for order in ([1, 2, 0], [2, 0, 1], [0, 2, 1], [1, 0, 2], [2, 1, 0]):
        assert integral(order) == pytest.approx(reference, rel=1e-12)


def test_the_points_of_a_triangle_land_on_it():
    rule = triangle_quadrature(6)
    triangle = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]])
    points = np.asarray(rule.points(triangle[None]))[0]
    assert points.shape == (6, 3)
    np.testing.assert_allclose(points[:, 2], 2.0, atol=1e-14, err_msg="left the triangle's plane")
    assert np.all(points[:, 0] > 0.0) and np.all(points[:, 1] > 0.0)
    assert np.all(points[:, 0] + points[:, 1] < 1.0)


def test_the_weighted_mean_of_the_points_is_the_centroid():
    """True of any rule that integrates a linear function, which all of them claim to."""
    rule = triangle_quadrature(12)
    triangle = np.array([[0.2, -1.0, 0.0], [3.0, 0.5, 1.0], [-0.4, 2.0, -2.0]])
    points = np.asarray(rule.points(triangle[None]))[0]
    mean = np.sum(np.asarray(rule.weight)[:, None] * points, axis=0)
    np.testing.assert_allclose(mean, triangle.mean(axis=0), atol=1e-14)


def test_each_barycentric_coordinate_belongs_to_the_vertex_of_the_same_index():
    """A gap the catalogue cannot cover, found by mutation.

    Every rule offered here is symmetric, so permuting the three coordinates maps its point set
    onto itself — reversing them inside :meth:`TriangleQuadrature.points` changes nothing about
    any catalogued rule, and every other test in this file stayed green under exactly that
    mutation. Only a deliberately lopsided rule, which the class accepts, can tell whether
    coordinate ``k`` is paired with vertex ``k``.
    """
    rule = TriangleQuadrature(barycentric=((0.7, 0.2, 0.1),), weight=(1.0,), degree=0)
    triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    point = np.asarray(rule.points(triangle[None]))[0, 0]
    np.testing.assert_allclose(point, [0.2, 0.1, 0.0], atol=1e-15)


def test_the_one_point_rule_is_the_centroid():
    triangle = np.array([[0.2, -1.0, 0.0], [3.0, 0.5, 1.0], [-0.4, 2.0, -2.0]])
    points = np.asarray(triangle_quadrature(1).points(triangle[None]))[0]
    np.testing.assert_allclose(points[0], triangle.mean(axis=0), atol=1e-15)


def test_the_rule_maps_a_whole_set_of_triangles_at_once():
    rule = triangle_quadrature(3)
    triangles = np.random.default_rng(0).normal(size=(7, 3, 3))
    points = np.asarray(rule.points(triangles))
    assert points.shape == (7, 3, 3)
    for index in range(7):
        one = np.asarray(rule.points(triangles[index][None]))[0]
        np.testing.assert_allclose(points[index], one, atol=0.0)


def test_an_uncatalogued_size_is_refused_and_the_message_says_what_there_is():
    with pytest.raises(ValueError, match=r"no 5-point triangle rule.*\[1, 3, 6, 12\]"):
        triangle_quadrature(5)


def test_the_seven_point_rule_is_absent_because_it_was_dominated():
    """Pinned so that re-adding it is a decision rather than an oversight: on the transfer
    integral it measured 0.0169 against the six-point rule's 0.0078, at more cost."""
    assert 7 not in _RULES


def test_a_rule_is_hashable_so_it_can_be_a_compile_time_constant():
    assert len({triangle_quadrature(6), triangle_quadrature(6), triangle_quadrature(3)}) == 2


def test_a_rule_built_by_hand_is_still_a_rule():
    """The class is a value object, so nothing stops a caller supplying its own table; the
    catalogue is a convenience, not a gate."""
    rule = TriangleQuadrature(
        barycentric=((0.5, 0.25, 0.25), (0.25, 0.5, 0.25), (0.25, 0.25, 0.5)),
        weight=(1.0 / 3.0,) * 3,
        degree=1,
    )
    assert rule.n_points == 3
    assert sum(rule.weight) == pytest.approx(1.0, abs=1e-15)
