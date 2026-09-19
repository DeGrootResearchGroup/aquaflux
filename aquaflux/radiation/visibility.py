"""Which sources each receiver can actually see, and how much of their light gets through.

Occlusion enters the gather as one more factor on each source-receiver term, and it is built in
two halves because the halves behave completely differently.

The bodies are of two kinds, stored separately. **Analytic primitives** — a sleeve, a baffle —
each carry their own transmittance, so each needs its own layer of the mask. **The emitting
surface's own triangles** are the reactor's walls and are opaque, so they collapse into one layer
with no transmittance to carry. That second kind is what lets a bent duct shadow itself, which no
primitive can express because the geometry doing the blocking *is* the emitting surface.

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

from aquaflux.radiation.triangles import segment_is_cut

__all__ = ["Visibility", "build_visibility"]


class Visibility(eqx.Module):
    """A frozen record of which bodies lie between which sources and which receivers.

    Attributes
    ----------
    blocked : jnp.ndarray of bool, shape ``(n_occluders, n_receivers, n_facets)``
        Whether each body lies across the segment from each facet to each receiver.
    receivers : jnp.ndarray, shape ``(n_receivers, 3)``
        The receiver positions this mask was built for.
    blocked_by_geometry : jnp.ndarray of bool, shape ``(n_receivers, n_facets)``
        Whether the emitting surface's **own triangles** stand between each facet and each
        receiver. Kept apart from :attr:`blocked` because it carries no transmittance: the
        surface set is the reactor's walls and bodies, and those are opaque. A partly
        transmitting body belongs in :attr:`blocked`, as an analytic primitive.

    Notes
    -----
    The mask is the module's largest array: a hundred thousand receivers against a thousand
    facets is a hundred million entries **per body**. It is stored as booleans, one byte each,
    which is eight times larger than it needs to be; packing to bits is the obvious saving if it
    ever matters, and costs an unpack in the gather.
    """

    blocked: jnp.ndarray
    receivers: jnp.ndarray
    blocked_by_geometry: jnp.ndarray

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
        return jnp.prod(attenuation, axis=0) * ~self.blocked_by_geometry

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
    self_occlusion: bool = True,
    offset_scale: float = 1e-6,
    chunk_size: int = 4096,
    work_limit: int = 4_000_000,
) -> Visibility:
    """Work out, once, which bodies lie between which sources and which receivers.

    Brute force over the bodies, which is correct practice at this count: a bounding-volume
    hierarchy over a handful of primitives is a single leaf node, and the traversal would cost
    more than the tests it saves.

    Parameters
    ----------
    occluders : sequence of Occluder
        The analytic bodies. An empty sequence is fine; the surface's own triangles are handled
        separately.
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
    self_occlusion : bool, optional
        Whether the emitting surface's own triangles block light. **On by default**: a surface
        that does not shadow itself is the defect this module exists to fix, and a mask silently
        missing it looks exactly like one that includes it. Turn it off only for a scene known
        to be convex, where the source-side cosine clamp is already the exact visibility test
        and is cheaper.
    chunk_size : int, optional
        Receivers per pass, bounding the peak memory of the build.
    work_limit : int, optional
        Ray-by-triangle entries per pass of the self-occlusion test, which is what bounds its
        memory and, through that, its speed.

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

    facet_index = jnp.arange(n_facets)

    primitive_rows, geometry_rows = [], []
    for start in range(0, n_receivers, chunk_size):
        receivers = points[start : start + chunk_size]
        rays = receivers.shape[0]
        origin = surfaces.centroid[None, :, :]
        target = receivers[:, None, :]
        if occluders:
            primitive_rows.append(
                jnp.stack(
                    [body.blocks(origin, target, near[None, :]) for body in occluders], axis=0
                )
            )
        if self_occlusion:
            flat = (rays * n_facets, 3)
            geometry_rows.append(
                segment_is_cut(
                    jnp.broadcast_to(origin, (rays, n_facets, 3)).reshape(flat),
                    jnp.broadcast_to(target, (rays, n_facets, 3)).reshape(flat),
                    surfaces.vertices,
                    jnp.broadcast_to(near, (rays, n_facets)).reshape(-1),
                    exclude=jnp.broadcast_to(facet_index, (rays, n_facets)).reshape(-1),
                    work_limit=work_limit,
                ).reshape(rays, n_facets)
            )

    blocked = (
        jnp.concatenate(primitive_rows, axis=1)
        if occluders
        else jnp.zeros((0, n_receivers, n_facets), dtype=bool)
    )
    by_geometry = (
        jnp.concatenate(geometry_rows, axis=0)
        if self_occlusion
        else jnp.zeros((n_receivers, n_facets), dtype=bool)
    )
    return Visibility(blocked=blocked, receivers=points, blocked_by_geometry=by_geometry)
