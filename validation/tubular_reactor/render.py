"""Draw a sampled fluence-rate field on a plane up to the reactor's true boundary.

The field is known at grid points in the water (a 3D background grid, or a 2D grid on a plane). A
picture of it drawn cell by cell shows the grid's staircase wherever a wall, a sleeve or a hole cuts
through a cell. Here the plane is cut into fine pixels instead, each kept only if the reactor's exact
water test (``Reactor.in_water``, the same test that chose the grid's points) says it is water, and
each given a value interpolated from the grid points around it. So every boundary is drawn where it
is, and the colour between the last grid point and a wall is an interpolation -- an estimate, not a
gathered value. The stored field is not changed.

Interpolation is multilinear in log G over the corners of the pixel's grid cell that are in the water,
their weights rescaled (a dry corner carries no value). A corner where G is exactly zero (the plate
hides every lamp) is left out of the logarithm and fades the value towards zero by its weight; a pixel
whose wet corners are all zero is zero. A pixel in the water whose cell has no wet corner takes the
nearest grid point's value.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

#: Pixel size of a rendered plane, in metres.
PIXEL = 5e-4


def wet_interpolate(field: np.ndarray, origin, spacing: float, points: np.ndarray) -> np.ndarray:
    """``field`` (any number of axes, NaN where dry) at ``points`` (n, field.ndim).

    Returns the values (n,), NaN where no corner of a point's cell is wet.
    """
    field = np.asarray(field, dtype=float)
    shape = np.asarray(field.shape)
    scaled = (points - np.asarray(origin)) / spacing
    low = np.clip(np.floor(scaled).astype(int), 0, shape - 2)
    frac = np.clip(scaled - low, 0.0, 1.0)
    log_sum = np.zeros(len(points))
    positive = np.zeros(len(points))
    wet = np.zeros(len(points))
    for corner in np.ndindex(*(2,) * field.ndim):
        c = np.asarray(corner)
        weight = np.prod(np.where(c == 1, frac, 1.0 - frac), axis=1)
        value = field[tuple((low + c).T)]
        is_wet = np.isfinite(value)
        is_positive = is_wet & (value > 0.0)
        wet += np.where(is_wet, weight, 0.0)
        positive += np.where(is_positive, weight, 0.0)
        log_sum += np.where(is_positive, weight * np.log(np.where(is_positive, value, 1.0)), 0.0)
    out = np.full(len(points), np.nan)
    some = wet > 0.0
    lit = positive > 0.0
    out[some] = 0.0
    out[lit] = np.exp(log_sum[lit] / positive[lit]) * positive[lit] / wet[lit]
    return out


def fill_nearest(values, points, field, origin, spacing) -> np.ndarray:
    """Where ``values`` is NaN, the value of the nearest wet point of ``field``."""
    missing = np.isnan(values)
    if not missing.any():
        return values
    wet = np.argwhere(np.isfinite(field))
    tree = cKDTree(np.asarray(origin) + spacing * wet)
    _, nearest = tree.query(points[missing])
    values = values.copy()
    values[missing] = field[tuple(wet[nearest].T)]
    return values


def plane(reactor, field, origin, spacing, normal_axis: int, offset: float, extent, pixel=PIXEL):
    """The field on the plane ``x[normal_axis] = offset``, in pixels over ``extent``.

    Parameters
    ----------
    reactor : Reactor
        Supplies the exact water test.
    field : np.ndarray
        3D grid values (NaN where dry), or 2D values already on this plane.
    origin, spacing
        The field's first point and its spacing (2D: the two in-plane coordinates).
    normal_axis : int
        0, 1 or 2: which coordinate is fixed.
    offset : float
        Its value.
    extent : (low_u, high_u, low_v, high_v)
        The in-plane coordinates drawn, in the order the two remaining axes come.

    Returns
    -------
    u, v : np.ndarray
        Pixel centres along each in-plane axis.
    image : np.ndarray, shape (len(u), len(v))
        The field, NaN outside the water.
    """
    axes = [a for a in range(3) if a != normal_axis]
    u = np.arange(extent[0] + pixel / 2, extent[1], pixel)
    v = np.arange(extent[2] + pixel / 2, extent[3], pixel)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    points = np.zeros((uu.size, 3))
    points[:, normal_axis] = offset
    points[:, axes[0]] = uu.ravel()
    points[:, axes[1]] = vv.ravel()
    water = reactor.in_water(points, margin=0.0)
    image = np.full(uu.size, np.nan)
    if field.ndim == 3:
        at = points[water]
        values = wet_interpolate(field, origin, spacing, at)
        values = fill_nearest(values, at, field, origin, spacing)
    else:
        at = points[water][:, axes]
        values = wet_interpolate(field, origin, spacing, at)
        values = fill_nearest(values, at, field, origin, spacing)
    image[water] = values
    return u, v, image.reshape(uu.shape)
