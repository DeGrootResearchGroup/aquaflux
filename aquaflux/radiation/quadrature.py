"""Quadrature points on a triangle, for integrating over a facet rather than sampling it once.

A surface-to-surface transfer factor is a *double* area integral — over the sending facet and
over the receiving one. :func:`~aquaflux.radiation.solid_angle.projected_solid_angle` does the
sending half exactly and in closed form, which leaves the receiving half to quadrature. Sampling
the receiver at one point is the cheapest such rule and the one that costs the most accuracy; the
rules here are the alternatives.

Every rule is stated in **barycentric coordinates** — three weights summing to one, giving a
point as that combination of the triangle's vertices — so a single table serves every triangle
whatever its size, shape or orientation. The quadrature weights likewise sum to one, so the rule
returns an **area average** rather than an area integral, and a quantity already equal to one
over the whole triangle stays exactly one.

Only symmetric rules with **strictly positive weights and interior points** are kept. A negative
weight can drive a transfer factor below zero, which is not merely inaccurate but outside the
range the radiosity system's conditioning argument assumes, and a point on the boundary of a
facet sits on the surface of its neighbour in a closed enclosure, where the integrand diverges.
That rules out the otherwise standard four-point degree-three rule, whose centroid weight is
negative.

⚠️ **Polynomial degree does not order these rules by accuracy on the integral they are used
for.** The transfer integrand goes like ``1/r^2`` and is nearly singular between facets that
share an edge, which is exactly where the error concentrates and exactly what a polynomial rule
is poor at. Measured on the reciprocity of a closed box, the seven-point degree-five rule is
**worse than the six-point degree-four rule** — 0.0169 against 0.0078 — because it spends nearly
a quarter of its weight at the centroid, far from where the integrand varies. The rules offered
here are the ones that were measured to sit on the cost/accuracy frontier for this integrand;
they are not simply the standard table up to some degree.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

__all__ = ["TriangleQuadrature", "triangle_quadrature"]


def _three(a: float, b: float, c: float, weight: float):
    """The three cyclic permutations of one barycentric triple, at a shared weight."""
    return [((a, b, c), weight), ((c, a, b), weight), ((b, c, a), weight)]


def _six(a: float, b: float, c: float, weight: float):
    """All six permutations of one barycentric triple, at a shared weight."""
    return [
        ((a, b, c), weight),
        ((a, c, b), weight),
        ((b, a, c), weight),
        ((b, c, a), weight),
        ((c, a, b), weight),
        ((c, b, a), weight),
    ]


# Symmetric Gaussian rules on the triangle (Dunavant 1985), keyed by point count. The degree
# each integrates exactly is recorded beside it, as a property of the rule rather than as a
# prediction about the transfer integral — see the module docstring on why the two differ.
_RULES: dict[int, tuple[int, list]] = {
    1: (1, [((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), 1.0)]),
    3: (2, _three(2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0)),
    6: (
        4,
        _three(0.108103018168070, 0.445948490915965, 0.445948490915965, 0.223381589678011)
        + _three(0.816847572980459, 0.091576213509771, 0.091576213509771, 0.109951743655322),
    ),
    12: (
        6,
        _three(0.873821971016996, 0.063089014491502, 0.063089014491502, 0.050844906370207)
        + _three(0.501426509658179, 0.249286745170910, 0.249286745170910, 0.116786275726379)
        + _six(0.636502499121399, 0.310352451033785, 0.053145049844816, 0.082851075618374),
    ),
}


class TriangleQuadrature(eqx.Module):
    """A rule for averaging a function over a triangle.

    Built by :func:`triangle_quadrature` rather than directly; the tables are a fixed catalogue,
    not something a caller composes. Both fields are static, so an instance is hashable and may
    be passed as a compile-time constant.

    Attributes
    ----------
    barycentric : tuple of (float, float, float)
        One triple per point, each summing to one, giving that point as a combination of the
        triangle's three vertices.
    weight : tuple of float
        One weight per point, together summing to one.
    degree : int
        Highest polynomial degree the rule integrates exactly over a triangle. A property of the
        rule; **not** a ranking of accuracy on the transfer integral, which is not polynomial.
    """

    barycentric: tuple[tuple[float, float, float], ...] = eqx.field(static=True)
    weight: tuple[float, ...] = eqx.field(static=True)
    degree: int = eqx.field(static=True)

    @property
    def n_points(self) -> int:
        """Number of points the rule evaluates per triangle.

        ⚠️ **Not the factor it multiplies a transfer build by** — that is much smaller, because
        the extra points reuse geometry the build has already loaded. See
        :func:`~aquaflux.radiation.radiosity.build_transfer` for the measured figures.
        """
        return len(self.weight)

    def points(self, vertices) -> jnp.ndarray:
        """Place the rule on each of a set of triangles.

        Parameters
        ----------
        vertices : array_like, shape ``(n_triangles, 3, 3)``
            Triangle vertex positions.

        Returns
        -------
        jnp.ndarray, shape ``(n_triangles, n_points, 3)``
            The quadrature points of each triangle, in the order of :attr:`weight`.
        """
        vertices = jnp.asarray(vertices, dtype=float)
        return jnp.einsum("qk,tkd->tqd", jnp.asarray(self.barycentric), vertices)


def triangle_quadrature(n_points: int = 1) -> TriangleQuadrature:
    """Look up a symmetric quadrature rule by its number of points.

    Parameters
    ----------
    n_points : int, optional
        One of 1, 3, 6 or 12. These are the measured cost/accuracy frontier for the transfer
        integral; a seven-point rule of higher polynomial degree was dominated by the six-point
        rule and is deliberately absent.

    Returns
    -------
    TriangleQuadrature

    Raises
    ------
    ValueError
        If no rule of that size is catalogued.

    Examples
    --------
    >>> rule = triangle_quadrature(3)
    >>> rule.n_points, rule.degree
    (3, 2)
    """
    if n_points not in _RULES:
        raise ValueError(
            f"no {n_points}-point triangle rule; available sizes are "
            f"{sorted(_RULES)} (see triangle_quadrature)"
        )
    degree, table = _RULES[n_points]
    return TriangleQuadrature(
        barycentric=tuple(point for point, _ in table),
        weight=tuple(weight for _, weight in table),
        degree=degree,
    )
