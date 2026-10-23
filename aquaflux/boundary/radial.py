"""Boundary values given against distance from an axis: a fully developed pipe flow's profiles.

The flow far down a straight pipe no longer changes along it, and in a round pipe it depends on
nothing but the distance from the axis. Its profiles -- the axial velocity, the turbulence -- are
then each one function of the radius, and an inlet that admits that flow prescribes those functions
on its faces by each face's distance from the inlet's centre. :class:`RadialProfile` is one such
function, a table of values at increasing radii interpolated linearly between them and held at its
ends beyond them; :class:`AxialVector` turns one into a vector along the axis, for a velocity.
:func:`radial_average` makes the table from a computed developed flow, by averaging its cell values
over bands of radius.
"""

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from aquaflux.vectors import dot, norm, scale

from .conditions import as_float_leaf

__all__ = ["AxialVector", "RadialProfile", "radial_average"]


def _distance_from_axis(points, centre, axis):
    """Each point's distance from the line through ``centre`` along the unit ``axis``."""
    offset = points - centre
    return norm(offset - scale(jnp.broadcast_to(axis, offset.shape), dot(offset, axis)))


class RadialProfile(eqx.Module):
    """A value given at radii from an axis, interpolated linearly between them.

    Beyond the first and last radius the value is held at its end, so a face nearer the axis or the
    wall than any radius tabulated takes the nearest value tabulated.

    Attributes
    ----------
    centre : jnp.ndarray, shape ``(dim,)``
        A point on the axis.
    axis : jnp.ndarray, shape ``(dim,)``
        The axis's direction, scaled to unit length on construction.
    radii : jnp.ndarray, shape ``(n,)``
        The radii the values are given at, increasing.
    values : jnp.ndarray, shape ``(n,)``
        The value at each radius.

    Raises
    ------
    ValueError
        If the radii are not increasing, the values do not match them, the axis is zero, or the centre
        and the axis are of different dimensions.
    """

    centre: jnp.ndarray
    axis: jnp.ndarray
    radii: jnp.ndarray
    values: jnp.ndarray

    def __init__(self, centre, axis, radii, values):
        centre, axis = as_float_leaf(centre), as_float_leaf(axis)
        radii, values = as_float_leaf(radii), as_float_leaf(values)
        if centre.shape != axis.shape or centre.ndim != 1:
            raise ValueError(
                f"a radial profile's centre and axis are two vectors of one dimension, got shapes "
                f"{centre.shape} and {axis.shape}."
            )
        length = float(np.linalg.norm(np.asarray(axis)))
        if not length > 0.0:
            raise ValueError("a radial profile's axis must not be zero.")
        if radii.ndim != 1 or values.shape != radii.shape or radii.size == 0:
            raise ValueError(
                f"a radial profile gives one value at each radius, got {values.shape} values at "
                f"{radii.shape} radii."
            )
        if np.any(np.diff(np.asarray(radii)) <= 0.0):
            raise ValueError("a radial profile's radii must increase.")
        self.centre, self.axis = centre, axis / length
        self.radii, self.values = radii, values

    def radius(self, points: jnp.ndarray) -> jnp.ndarray:
        """Each point's distance from the axis.

        Parameters
        ----------
        points : jnp.ndarray, shape ``(m, dim)``

        Returns
        -------
        jnp.ndarray, shape ``(m,)``
        """
        return _distance_from_axis(points, self.centre, self.axis)

    def __call__(self, points: jnp.ndarray) -> jnp.ndarray:
        """The value at each point, by its distance from the axis.

        Parameters
        ----------
        points : jnp.ndarray, shape ``(m, dim)``
            Face centroids, say.

        Returns
        -------
        jnp.ndarray, shape ``(m,)``
        """
        return jnp.interp(self.radius(points), self.radii, self.values)


class AxialVector(eqx.Module):
    """A vector along a radial profile's axis, as long as the profile's value: a developed velocity.

    Attributes
    ----------
    magnitude : RadialProfile
        The component along the axis, by radius; its axis is the vector's direction.
    """

    magnitude: RadialProfile

    def __call__(self, points: jnp.ndarray) -> jnp.ndarray:
        """The vector at each point.

        Parameters
        ----------
        points : jnp.ndarray, shape ``(m, dim)``

        Returns
        -------
        jnp.ndarray, shape ``(m, dim)``
        """
        return scale(jnp.broadcast_to(self.magnitude.axis, points.shape), self.magnitude(points))


def radial_average(
    points: np.ndarray,
    weights: np.ndarray,
    fields: Mapping[str, np.ndarray],
    centre: np.ndarray,
    axis: np.ndarray,
    bands: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fields averaged over bands of distance from an axis, weighted (by cell volume, say).

    The bands divide the radii from zero to the largest point's into ``bands`` of equal width; each
    band that holds a point gives one row, at the weighted mean radius of its points, so a radius
    and its values are averaged over the same points.

    Parameters
    ----------
    points : np.ndarray, shape ``(m, dim)``
        Where the values are (cell centroids).
    weights : np.ndarray, shape ``(m,)``
        Each point's weight, positive (cell volumes).
    fields : mapping of {str: np.ndarray}
        Each field's value at the points, shape ``(m,)``.
    centre, axis : np.ndarray, shape ``(dim,)``
        The axis: a point on it, and its direction (any length).
    bands : int
        How many bands of radius, at least one.

    Returns
    -------
    radii : np.ndarray, shape ``(n,)``
        The bands' mean radii, increasing, ``n <= bands`` (empty bands give none).
    means : dict of {str: np.ndarray}
        Each field's mean over each band, shape ``(n,)``.

    Raises
    ------
    ValueError
        If ``bands`` is below one or a weight is not positive.
    """
    if bands < 1:
        raise ValueError(f"a radial average needs at least one band, got {bands}.")
    points, weights = np.asarray(points, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    if np.any(weights <= 0.0):
        raise ValueError("a radial average's weights must be positive.")
    direction = np.asarray(axis, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    radius = np.asarray(_distance_from_axis(points, np.asarray(centre, np.float64), direction))
    band = np.minimum((radius / radius.max() * bands).astype(np.int64), bands - 1)
    total = np.bincount(band, weights, minlength=bands)
    held = total > 0.0

    def mean(values: np.ndarray) -> np.ndarray:
        return (np.bincount(band, weights * values, minlength=bands) / np.where(held, total, 1.0))[
            held
        ]

    return mean(radius), {name: mean(np.asarray(values)) for name, values in fields.items()}
