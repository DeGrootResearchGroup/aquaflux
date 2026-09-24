"""Which sources each receiver can actually see, and how much of their light gets through.

Occlusion enters the gather as one more factor on each source-receiver term, and it is built in
two halves because the halves behave completely differently.

The bodies are of two kinds, stored separately. **Analytic bodies** — a sleeve, a baffle, a
vessel composed from primitives, or the water a vessel holds — each carry their own
transmittance, so each needs its own layer of the mask. **The emitting surface's own triangles**
are the reactor's walls and are opaque, so they collapse into one layer with no transmittance to
carry. That second kind is what answers for a shape nobody has described analytically: the
geometry doing the blocking is the emitting surface itself, and a triangle soup is all there is
to go on.

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

from aquaflux.radiation.self_occlusion import (
    RayCastOcclusion,
    SelfOcclusion,
)
from aquaflux.radiation.work import DEFAULT_PAIR_LIMIT, receivers_per_pass

__all__ = ["Visibility", "build_visibility"]


@eqx.filter_jit
def _compiled_blocks(body, origin, target, near):
    """One body's layer of one chunk, as a single compiled expression.

    ⚠️ **Compiling an analytic body is worth a great deal, and compiling a triangulated one
    raises.** An analytic test is a few dozen arithmetic operations over a receivers-by-facets
    array, and a body assembled from several inequalities — or a fluid described by several
    regions — is several such arrays. Evaluated eagerly, every one of them is materialized in
    turn, hundreds of megabytes each at a production chunk, and the mask's cost becomes the cost
    of writing those intermediates rather than of the arithmetic. Compiled, they fuse into the
    reduction at the end and none is ever formed. A body that answers on the host instead —
    walking a grid of triangles, dropping rays as they are settled — cannot be traced at all,
    and deliberately so.

    So this is applied only where :attr:`~aquaflux.solids.Body.traceable` says
    it may be, which is a declaration on the body rather than a guess from its type. The two
    kinds are meant to compose in one scene: a vessel described as primitives, with whatever
    genuinely is a triangle soup standing beside it.
    """
    return body.blocks(origin, target, near)


def _blocked_by(bodies, origin, target, near):
    """One chunk's layer of the mask, per body, compiling each where it says it can be."""
    return jnp.stack(
        [
            _compiled_blocks(body, origin, target, near)
            if body.traceable
            else body.blocks(origin, target, near)
            for body in bodies
        ],
        axis=0,
    )


class Visibility(eqx.Module):
    """A frozen record of which bodies lie between which sources and which receivers.

    Attributes
    ----------
    blocked : jnp.ndarray of bool, shape ``(n_occluders, n_receivers, n_facets)``
        Whether each body lies across the segment from each facet to each receiver.
    receivers : jnp.ndarray, shape ``(n_receivers, 3)``
        The receiver positions this mask was built for.
    hidden_by_geometry : jnp.ndarray, shape ``(n_receivers, n_facets)``
        What fraction of each source the emitting surface's **own triangles** hide from each
        receiver. Kept apart from :attr:`blocked` because it carries no transmittance: the
        surface set is the reactor's walls and bodies, and those are opaque. A partly
        transmitting body belongs in :attr:`blocked`, as an analytic primitive.

        A **fraction** rather than a flag, so the two self-occlusion strategies share one field
        and nothing downstream branches on which ran: a ray test returns only zeros and ones,
        while the silhouette clip returns what it measures. It costs eight bytes a pair where a
        flag cost one.
    overlapping : jnp.ndarray of bool, shape ``(n_receivers, n_facets)``
        Whether more than one blocker contributed to :attr:`hidden_by_geometry` here, so their
        fractions were **added**. Exactness is proven for a pair where exactly one did; where
        several did they may or may not overlap in angle -- a tiling of one flat wall does not,
        and is the common benign case -- so this reports "not proven", not "wrong". Always
        ``False`` from a ray test, whose ``or`` is idempotent.

    Notes
    -----
    The mask is the module's largest array: a hundred thousand receivers against a thousand
    facets is a hundred million entries **per body**. It is stored as booleans, one byte each,
    which is eight times larger than it needs to be; packing to bits is the obvious saving if it
    ever matters, and costs an unpack in the gather.
    """

    blocked: jnp.ndarray
    receivers: jnp.ndarray
    hidden_by_geometry: jnp.ndarray
    overlapping: jnp.ndarray

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
        return jnp.prod(attenuation, axis=0) * (1.0 - self.hidden_by_geometry)

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
    receiver_facet=None,
    self_occlusion: SelfOcclusion | None = None,
    offset_scale: float = 1e-6,
    pair_limit: int = DEFAULT_PAIR_LIMIT,
) -> Visibility:
    """Work out, once, which bodies lie between which sources and which receivers.

    Brute force over the bodies, which is correct practice at this count: a bounding-volume
    hierarchy over a handful of primitives is a single leaf node, and the traversal would cost
    more than the tests it saves.

    Parameters
    ----------
    occluders : sequence of aquaflux.solids.Body
        The analytic bodies. An empty sequence is fine; the surface's own triangles are handled
        separately.
    surfaces : Surfaces
        The emitting set; segments start at facet centroids.
    points : array_like, shape ``(n_receivers, 3)``
        Receiver positions.
    receiver_facet : array_like of int, shape ``(n_receivers,)``, optional
        When each receiver point is itself the centroid of one of ``surfaces``' facets, that
        facet's index. ⚠️ **Omitting it where it applies blocks every pair of facets that can
        see each other**, because the segment then ends exactly in the target facet's plane and
        the self-occlusion test reads that as a hit. There is no distance margin at the far end
        of a segment to absorb it, the way ``offset_scale`` absorbs the same thing at the near
        end, and the result is not a near miss: a closed enclosure comes back fully shadowed,
        its interreflection silently switched off. Use ``-1`` for a receiver that is not on a
        facet, and omit the argument entirely for receivers in the volume, which is the case it
        does not apply to.
    offset_scale : float, optional
        How far along each segment to start looking for hits, as a fraction of the facet's own
        size -- specifically of the square root of its area. **Relative rather than absolute**,
        so the module has one length scale rather than three, and so the same setting means the
        same thing on a reactor in metres and a lamp in millimetres. A fixed epsilon fails at
        both ends: too small and a facet shadows itself, too large and light leaks past a body
        that should stop it.
    self_occlusion : SelfOcclusion or None, optional
        How the emitting surface's own triangles are tested for standing in the light.
        Defaults to :class:`~aquaflux.radiation.self_occlusion.RayCastOcclusion`, one ray per
        pair. Pass :class:`~aquaflux.radiation.self_occlusion.SilhouetteOcclusion` for an exact
        fraction instead of a bit, at a cost that depends strongly on facet count.

        ⚠️ **To switch self-occlusion off, pass
        :class:`~aquaflux.radiation.self_occlusion.NoOcclusion`, not ``None``** -- ``None`` means
        "use the default", which is on. Switching it off is rarely right: a surface that does not
        shadow itself is the defect this module exists to fix, and a mask silently missing it
        looks exactly like one that includes it.
    pair_limit : int, optional
        Receiver-by-facet pairs per pass of the analytic-body test, bounding its peak memory
        whatever the facet count. Each self-occlusion strategy carries its own bound, because
        what has to be bounded differs between them.

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
    strategy = RayCastOcclusion() if self_occlusion is None else self_occlusion
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

    per_pass = receivers_per_pass(pair_limit, n_facets)
    primitive_rows = []
    for start in range(0, n_receivers, per_pass):
        receivers = points[start : start + per_pass]
        origin = surfaces.centroid[None, :, :]
        target = receivers[:, None, :]
        if occluders:
            primitive_rows.append(_blocked_by(occluders, origin, target, near[None, :]))
    # The fallback turns on whether any row was produced, not on whether one was asked for: a
    # set with no receivers at all -- a surface-only study, which is a legal thing to build --
    # runs no chunks, so the list is empty however the flags are set, and concatenating nothing
    # raises rather than giving back an empty array.
    blocked = (
        jnp.concatenate(primitive_rows, axis=1)
        if primitive_rows
        else jnp.zeros((len(occluders), n_receivers, n_facets), dtype=bool)
    )
    geometry = strategy.field(surfaces, points, near, receiver_facet)
    return Visibility(
        blocked=blocked,
        receivers=points,
        hidden_by_geometry=geometry.fraction,
        overlapping=geometry.overlapping,
    )
