"""Which sources each receiver can actually see, and how much of their light gets through.

Occlusion enters the gather as one more factor on each source-receiver term, and it is built in
two halves because the halves behave completely differently.

**Whether a body lies across a segment is a hard yes or no, fixed by geometry.** It is computed
once, when the model is built, and stored. It has no derivative worth having: move an occluder
by a hair and nothing changes until a shadow edge sweeps past a receiver, at which point the
answer jumps. Freezing it is therefore not a compromise — the frozen thing is a staircase, and
nothing is lost by not differentiating a staircase.

**How much light a body lets through is a number, and it is live.** A quartz sleeve transmits
most of it, a baffle none, and the figure is exactly the sort of thing a design study varies.
It stays an ordinary argument, differentiable, supplied at every call:

    surviving = product over bodies of  [ 1 - blocked * (1 - transmittance) ]

which is one where nothing blocks, the transmittance where one opaque-or-not body does, and the
product where several overlap.

⚠️ **The mask is indexed by receiver, so it belongs to a particular set of receivers.** A mask
built for one set of points and used with another is silently wrong — every shadow lands in the
wrong place — so :class:`Visibility` carries the points it was built for and the gather checks
them.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

__all__ = ["Visibility", "build_visibility"]


class Visibility(eqx.Module):
    """A frozen record of which bodies lie between which sources and which receivers.

    Attributes
    ----------
    blocked : jnp.ndarray of bool, shape ``(n_occluders, n_receivers, n_facets)``
        Whether each body lies across the segment from each facet to each receiver.
    receivers : jnp.ndarray, shape ``(n_receivers, 3)``
        The receiver positions this mask was built for.

    Notes
    -----
    The mask is the module's largest array: a hundred thousand receivers against a thousand
    facets is a hundred million entries **per body**. It is stored as booleans, one byte each,
    which is eight times larger than it needs to be; packing to bits is the obvious saving if it
    ever matters, and costs an unpack in the gather.
    """

    blocked: jnp.ndarray
    receivers: jnp.ndarray

    @property
    def n_occluders(self) -> int:
        """How many bodies the mask covers."""
        return int(self.blocked.shape[0])

    def surviving(self, transmittance) -> jnp.ndarray:
        """Fraction of light getting through, per source-receiver pair.

        Parameters
        ----------
        transmittance : array_like, shape ``(n_occluders,)``
            What fraction each body transmits, in ``[0, 1]``. Differentiable.

        Returns
        -------
        jnp.ndarray, shape ``(n_receivers, n_facets)``
        """
        transmittance = jnp.broadcast_to(
            jnp.asarray(transmittance, dtype=float), (self.n_occluders,)
        )
        attenuation = 1.0 - self.blocked * (1.0 - transmittance[:, None, None])
        return jnp.prod(attenuation, axis=0)

    def for_receivers(self, points) -> jnp.ndarray:
        """Check that ``points`` are the receivers this mask was built for, and return it.

        Raises
        ------
        ValueError
            If the points differ, in count or in position. A mask used with the wrong receivers
            puts every shadow in the wrong place and raises no error of its own.
        """
        points = jnp.asarray(points, dtype=float)
        if points.shape != self.receivers.shape or not bool(jnp.all(points == self.receivers)):
            msg = (
                "this visibility mask was built for different receivers "
                f"(mask {tuple(self.receivers.shape)}, given {tuple(points.shape)}). The mask is "
                "indexed by receiver, so using it with another set silently moves every shadow."
            )
            raise ValueError(msg)
        return self.blocked


def build_visibility(
    occluders,
    surfaces,
    points,
    *,
    offset_scale: float = 1e-6,
    chunk_size: int = 4096,
) -> Visibility:
    """Work out, once, which bodies lie between which sources and which receivers.

    Brute force over the bodies, which is correct practice at this count: a bounding-volume
    hierarchy over a handful of primitives is a single leaf node, and the traversal would cost
    more than the tests it saves.

    Parameters
    ----------
    occluders : sequence of Occluder
        The bodies. An empty sequence gives a mask that blocks nothing.
    surfaces : Surfaces
        The emitting set; segments start at facet centroids.
    points : array_like, shape ``(n_receivers, 3)``
        Receiver positions.
    offset_scale : float, optional
        How far along each segment to start looking for hits, as a fraction of the facet's own
        size -- specifically of the square root of its area. **Relative rather than absolute**,
        so the module has one length scale rather than three, and so the same setting means the
        same thing on a reactor in metres and a lamp in millimetres. A fixed epsilon fails at
        both ends: too small and a facet shadows itself, too large and light leaks past a body
        that should stop it.
    chunk_size : int, optional
        Receivers per pass, bounding the peak memory of the build.

    Returns
    -------
    Visibility

    Raises
    ------
    ValueError
        If any facet centroid or receiver lies inside one of the bodies. Such a point is
        embedded in the solid, and every answer computed there would be meaningless rather than
        merely small.
    """
    points = jnp.asarray(points, dtype=float)
    occluders = tuple(occluders)
    n_receivers, n_facets = points.shape[0], surfaces.n_facets

    if not occluders:
        return Visibility(
            blocked=jnp.zeros((0, n_receivers, n_facets), dtype=bool), receivers=points
        )

    for index, body in enumerate(occluders):
        for name, position in (("facet", surfaces.centroid), ("receiver", points)):
            inside = np.flatnonzero(np.asarray(body.contains(position)))
            if len(inside):
                msg = (
                    f"{len(inside)} {name}(s) lie inside occluder {index} "
                    f"({type(body).__name__}; first few: {inside[:8].tolist()}). A point inside "
                    "a solid body is embedded in it, not shadowed by it, and nothing computed "
                    "there means anything. Move the body, or remove the points."
                )
                raise ValueError(msg)

    # Relative to the facet's own size, per the `offset_scale` note above. A point source has no
    # area and no surface to shadow itself with, so it needs no exclusion.
    near = offset_scale * jnp.sqrt(surfaces.area)

    rows = []
    for start in range(0, n_receivers, chunk_size):
        receivers = points[start : start + chunk_size]
        origin = surfaces.centroid[None, :, :]
        target = receivers[:, None, :]
        rows.append(
            jnp.stack([body.blocks(origin, target, near[None, :]) for body in occluders], axis=0)
        )
    return Visibility(blocked=jnp.concatenate(rows, axis=1), receivers=points)
