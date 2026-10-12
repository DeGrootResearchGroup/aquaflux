"""Paths that skim the rim of a cylindrical shell, found by shooting rather than by descent.

Light from a source to a receiver can pass through the wall of a quartz sleeve without entering the
air gap it holds: it skims the sleeve's rim, bent round it. Every such path is a saddle point of its
optical length -- the shell bends it as a converging lens does -- so a search that only ever
shortens the path cannot find one. This module finds them the other way: a ray is fired backwards
from the receiver, traced through every surface exactly, and its aim is corrected until it reaches
the source.

**The two numbers that make that cheap.** Every surface crossed is the side of a cylinder parallel
to one axis, so the normal at every crossing is perpendicular to it, and two things are conserved
along the whole path: ``beta = n u_z``, the index times the direction's component along the axis,
and, about each cylinder's own axis, the index times the impact parameter of the path's projection.
The projection onto the plane across the axis then obeys Snell's law with the effective index
``sqrt(n^2 - beta^2)`` in each medium, and the distance travelled along the axis follows from the
projected length of each leg. So a path is fixed by two unknowns, ``beta`` and the impact parameter
``b`` of the receiver's leg about the shell's axis, and both have known ranges: ``b`` lies between
the radius at which the path inside the shell would just miss its core and the shell's own radius.

**How a path is found.** Over a grid of ``beta`` and ``b`` the ray is traced and its miss of the
source measured, across the projected ray and along the axis. Along each row of fixed ``beta`` the
projected miss changes sign where the ray passes the source; along the rows that crossing's axial
miss changes sign where the path reaches it. The cell where both do is the start of a Newton
iteration on the two misses, with their derivatives taken exactly. On each side of the shell there
is at most one such path between two points in every case measured, and the first cell is taken.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from aquaflux.vectors import dot, norm

#: Values of ``beta`` scanned, across the range over which the shell's band of paths exists.
_ROWS = 32

#: Values of the impact parameter scanned across that band, at each ``beta``.
_COLUMNS = 32

#: Halvings that find each end of the run of the band over which the traced ray exists.
_EDGE_HALVINGS = 30

#: Newton steps from the bracketing cell; the iteration converges quadratically in a handful.
_POLISH = 12

#: A path is found when both misses are below this share of the separation of its ends.
_TOLERANCE = 1e-10


def _across(vector):
    """``vector`` turned a quarter turn anticlockwise in the plane, ``(..., 2)``."""
    return jnp.stack([-vector[..., 1], vector[..., 0]], axis=-1)


def _cross(a, b):
    """The plane cross product, ``a_x b_y - a_y b_x``."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _trace(b, beta, geometry):
    """The ray from the receiver with impact parameter ``b`` about the shell, traced to the source.

    Returns the source's distance across the final projected leg, the axial miss at the source,
    whether every crossing and the final leg exist, and the crossings as ``(m, 2)`` projected
    points and ``(m,)`` axial positions, in the chain's order from the source.
    """
    centres, radii, outs, indices = (
        geometry["centres"],
        geometry["radii"],
        geometry["outs"],
        geometry["indices"],
    )
    effective = jnp.sqrt(jnp.maximum(indices**2 - beta**2, 1e-300))
    towards = geometry["shell"] - geometry["receiver"]
    distance = norm(towards)
    towards = towards / distance
    ratio = b / distance
    ok = jnp.abs(ratio) < 1.0
    direction = jnp.sqrt(jnp.maximum(1.0 - ratio**2, 0.0)) * towards - ratio * _across(towards)
    position, axial = geometry["receiver"], geometry["receiver_z"]
    points, heights = [None] * len(outs), [None] * len(outs)
    for k in reversed(range(len(outs))):
        offset = position - centres[k]
        along = dot(direction, offset)
        clearance = dot(offset, offset) - radii[k] ** 2
        discriminant = along**2 - clearance
        root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
        ok = ok & (discriminant > 0.0)
        # Forward the path leaves this body here, so backward it arrives from outside and enters
        # at the near intersection; forward it enters, so backward it leaves at the far one.
        if outs[k]:
            step = -along - root
            ok = ok & (clearance > 0.0)
        else:
            step = -along + root
        ok = ok & (step > 0.0)
        position = position + step * direction
        axial = axial - step * beta / effective[k + 1]
        points[k], heights[k] = position, axial
        normal = (position - centres[k]) / radii[k]
        normal = jnp.where(dot(direction, normal) > 0.0, -normal, normal)
        cosine = -dot(direction, normal)
        ratio_k = effective[k + 1] / effective[k]
        sine_squared = ratio_k**2 * (1.0 - cosine**2)
        ok = ok & (sine_squared < 1.0)
        direction = (
            ratio_k * direction
            + (ratio_k * cosine - jnp.sqrt(jnp.maximum(1.0 - sine_squared, 0.0))) * normal
        )
        direction = direction / norm(direction)
    remaining = geometry["source"] - position
    reach = dot(direction, remaining)
    ok = ok & (reach > 0.0)
    across = _cross(direction, remaining)
    miss = axial - reach * beta / effective[0] - geometry["source_z"]
    return across, miss, ok, jnp.stack(points), jnp.stack(heights)


def rim_crossings(geometry, side, start=None):
    """Where the path round one side of the shell crosses each surface, and whether it exists.

    Found from the grid of :data:`_ROWS` by :data:`_COLUMNS` and a Newton polish, or, given
    ``start``, by the polish alone from there: the unknowns of the path between a neighbouring source
    point and the same receiver, round the same side. Paths move smoothly with their source, so a
    neighbour's is a start inside the polish's basin, and it finds a path whose run of rays is too
    narrow for the grid to sample.

    Parameters
    ----------
    geometry : dict
        In the plane across the common axis: ``receiver`` and ``source`` ``(2,)`` and their axial
        positions ``receiver_z`` and ``source_z``; ``centres`` ``(m, 2)``, ``radii`` ``(m,)`` and
        ``outs`` (a tuple of ``m`` bools, a static value) for the cylinders crossed, in the chain's
        order; ``indices`` ``(m + 1,)``, each leg's refractive index; ``shell`` ``(2,)`` the shell's
        axis, ``shell_radius``, ``core_radius`` (the largest core the path must miss),
        ``shell_index`` and ``outside_index``. The receiver's leg must be the one leaving the shell.
    side : float
        ``+1`` or ``-1``: which side of the shell's axis the receiver's leg passes.
    start : jnp.ndarray, shape ``(2,)``, optional
        Unknowns to polish from instead of scanning the grid, as ``unknowns`` returns them.

    Returns
    -------
    points : jnp.ndarray, shape ``(m, 2)``
    heights : jnp.ndarray, shape ``(m,)``
        The crossings, projected and along the axis.
    found : jnp.ndarray
        Whether a path was found.
    unknowns : jnp.ndarray, shape ``(2,)``
        The path's share of the way across the band of impact parameters at its ``beta``, and its
        ``beta``: a start for a neighbouring source's path.
    """
    shell, core = geometry["shell_radius"], geometry["core_radius"]
    inner, outer = geometry["shell_index"], geometry["outside_index"]
    # The band is empty past the beta at which the core's edge, seen through the shell, reaches
    # the shell's own radius; and no beta can reach an index of any leg.
    edge = (shell**2 * outer**2 - core**2 * inner**2) / (shell**2 - core**2)
    reach = jnp.minimum(
        jnp.sqrt(jnp.maximum(edge, 0.0)), jnp.min(geometry["indices"]) * (1.0 - 1e-9)
    )
    exists = edge > 0.0
    scale = jnp.maximum(norm(geometry["receiver"] - geometry["source"]), 1e-300)

    def impact(fraction, beta):
        low = core * jnp.sqrt(inner**2 - beta**2) / jnp.sqrt(outer**2 - beta**2)
        return side * (low + (shell - low) * fraction)

    def misses(fraction, beta):
        across, miss, ok, _, _ = _trace(impact(fraction, beta), beta, geometry)
        return jnp.stack([across, miss]) / scale, ok

    def polish(_, unknowns):
        values, _ = misses(unknowns[0], unknowns[1])
        jacobian = jax.jacfwd(lambda u: misses(u[0], u[1])[0])(unknowns)
        moved = unknowns - jnp.linalg.solve(jacobian, values)
        moved = jnp.stack([jnp.clip(moved[0], 0.0, 1.0), jnp.clip(moved[1], -reach, reach)])
        return jnp.where(jnp.all(jnp.isfinite(moved)), moved, unknowns)

    if start is None:
        start, bracketed = _scanned(misses, reach)
    else:
        bracketed = jnp.asarray(True)
    unknowns = jax.lax.fori_loop(0, _POLISH, polish, start)
    values, ok = misses(unknowns[0], unknowns[1])
    _, _, _, points, heights = _trace(impact(unknowns[0], unknowns[1]), unknowns[1], geometry)
    found = (
        exists
        & bracketed
        & ok
        & (unknowns[0] > 0.0)
        & (unknowns[0] < 1.0)
        & jnp.all(jnp.abs(values) <= _TOLERANCE)
    )
    return points, heights, found, unknowns


def _scanned(misses, reach):
    """A start for the polish from the grid, and whether the grid bracketed a path at all."""
    fractions = (jnp.arange(_COLUMNS) + 0.5) / _COLUMNS
    rows = reach * ((jnp.arange(_ROWS) + 0.5) / _ROWS * 2.0 - 1.0)

    def row(beta):
        """Where along a row the projected ray passes the source, and the axial miss there.

        The coarse samples find where along the band the traced ray exists at all; the ends of that
        run are then found by bisection and the run sampled again between them, ends included. A
        path near an end -- grazing the sleeve, or just reaching the source's own sleeve -- lies
        between the last coarse sample that exists and the edge, where the coarse samples alone
        cannot see the projected miss change sign.
        """

        def exists(fraction):
            return misses(fraction, beta)[1]

        coarse = jax.vmap(exists)(fractions)
        first_ok = jnp.argmax(coarse)
        last_ok = _COLUMNS - 1 - jnp.argmax(coarse[::-1])

        def edge(good, bad):
            """The end of the run between a fraction where the ray exists and one where it may not."""
            good, bad = jnp.where(exists(bad), bad, good), bad

            def halve(_, ends):
                inside, outside = ends
                middle = 0.5 * (inside + outside)
                there = exists(middle)
                return jnp.where(there, middle, inside), jnp.where(there, outside, middle)

            return jax.lax.fori_loop(0, _EDGE_HALVINGS, halve, (good, bad))[0]

        low = edge(fractions[first_ok], jnp.where(first_ok > 0, fractions[first_ok - 1], 0.0))
        high = edge(
            fractions[last_ok], jnp.where(last_ok < _COLUMNS - 1, fractions[last_ok + 1], 1.0)
        )
        samples = low + (high - low) * jnp.linspace(0.0, 1.0, _COLUMNS)
        values, ok = jax.vmap(lambda fraction: misses(fraction, beta))(samples)
        across, miss = values[:, 0], values[:, 1]
        changes = ok[:-1] & ok[1:] & (jnp.sign(across[:-1]) != jnp.sign(across[1:]))
        first = jnp.argmax(changes)
        share = across[first] / (across[first] - across[first + 1])
        fraction = samples[first] + share * (samples[first + 1] - samples[first])
        found = jnp.any(changes) & jnp.any(coarse)
        return found, fraction, miss[first] + share * (miss[first + 1] - miss[first])

    # One row at a time: a whole grid per pair, over a batch of pairs, is too much to hold at once.
    has, fraction, miss = jax.lax.map(row, rows)
    changes = has[:-1] & has[1:] & (jnp.sign(miss[:-1]) != jnp.sign(miss[1:]))
    first = jnp.argmax(changes)
    share = miss[first] / (miss[first] - miss[first + 1])
    start = jnp.stack(
        [
            fraction[first] + share * (fraction[first + 1] - fraction[first]),
            rows[first] + share * (rows[first + 1] - rows[first]),
        ]
    )
    return start, jnp.any(changes)
