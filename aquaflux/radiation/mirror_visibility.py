"""What stands in the way of light reflected once in a mirror: a two-leg mask per mirror.

Light that reflects once in a planar mirror travels two straight legs: from the source to the
point where it meets the mirror, and from there to the receiver. Seen through the mirror it comes
straight from the source's mirror image, and that straight line crosses the mirror's plane at the
same meeting point -- so the point is found from the image, and each leg is then tested where the
light actually goes, through the real scene, against everything a direct path is tested against.

**One path per pair, as a ray test casts one segment per pair.** The path is the one through the
source's centroid: from the receiver towards the centroid's image. A source seen only partly
through the mirror is still recorded by that one path, blocked or clear, in the same sense that
:class:`~aquaflux.radiation.self_occlusion.RayCastOcclusion` records a source half in shadow
wholly one or the other. Which share of the image the mirror's facets show is the gather's
business, and exact; this is only what stands in front of it.

**A body counts the legs it lies across, because light crossing it twice is filtered twice.** A
quartz sleeve between a lamp and a wall mirror is crossed on the way to the mirror and again on
the way back, and transmits its fraction both times. So each body records 0, 1 or 2, and

    surviving = product over bodies of transmittance ** legs crossed

times nothing at all where the surface's own triangles stand across either leg -- those are the
reactor's walls, and opaque. With every count at most one this is exactly the direct mask's
expression, and it is evaluated through it (:func:`reflected_surviving`).

**The meeting point is lifted off the mirror by a margin**, a millionth of its facets' size by
default, towards the side it faces. Both legs end there, so without the lift each would end
exactly in the mirror's plane, and a ray test reads a segment ending on a triangle as cut by it:
the mirror would shadow every path reflected in it. Lifting the point clears every triangle lying
in that plane -- the mirror's own and any other -- without naming them, and moves a path by far
less than any shadow edge it could cross.

A source whose centroid is not strictly in front of the mirror is recorded clear rather than
tested: it straddles the plane, and its path through the centroid does not meet the mirror at all.
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from aquaflux.radiation.culling import _body_blocks
from aquaflux.radiation.mirrors import Mirror
from aquaflux.radiation.self_occlusion import NoOcclusion, RayCastOcclusion, SelfOcclusion
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.triangles import padded_length
from aquaflux.radiation.visibility import same_receivers, surviving_fraction
from aquaflux.radiation.work import DEFAULT_PAIR_LIMIT, receivers_per_pass

__all__ = [
    "MirrorVisibility",
    "build_mirror_masks",
    "build_mirror_visibility",
    "reflected_surviving",
    "shadows_reflected_paths",
]

#: The most legs of one path a body can lie across.
_LEGS = 2


def reflected_surviving(crossings, hidden, transmittance) -> jnp.ndarray:
    """Fraction of reflected light getting through, from how many legs each body crosses.

    ``transmittance ** legs`` for each body, written as one factor per leg so that it is the
    direct mask's own expression (:func:`~aquaflux.radiation.visibility.surviving_fraction`)
    applied once for the bodies crossing at least one leg and once for those crossing both -- and
    so that its derivative stays finite at an opaque body, where a power with a zero exponent
    would not.

    Parameters
    ----------
    crossings : jnp.ndarray of int, shape ``(n_occluders, *pairs)``
        Legs of each pair's path each body lies across: 0, 1 or 2.
    hidden : jnp.ndarray of bool, shape ``pairs``, or None
        Whether the surface's own triangles stand across either leg. ``None`` hides nothing.
    transmittance : array_like, shape ``(n_occluders,)``
        What fraction each body transmits, in ``[0, 1]``. Differentiable.

    Returns
    -------
    jnp.ndarray, shape ``pairs``
    """
    crossings = jnp.asarray(crossings)
    once = surviving_fraction(crossings >= 1, hidden, transmittance)
    return once * surviving_fraction(crossings >= _LEGS, None, transmittance)


class MirrorVisibility(eqx.Module):
    """A frozen record of what stands across each path reflected in one mirror.

    Built by :func:`build_mirror_visibility`, for one mirror, one set of receivers and the
    sources a path through that mirror can come from.

    Attributes
    ----------
    crossings : jnp.ndarray of uint8, shape ``(n_occluders, n_receivers, n_sources)``
        How many of the two legs of each path each body lies across.
    hidden : jnp.ndarray of bool, shape ``(n_receivers, n_sources)``, or None
        Whether the surface's own triangles stand across either leg; ``None`` where the surface
        does not shadow itself.
    receivers : jnp.ndarray, shape ``(n_receivers, 3)``
        The receiver positions the mask was built for. Receivers on or behind the mirror see
        nothing in it and are recorded clear.
    sources : numpy.ndarray of int, shape ``(n_sources,)``
        The facets the columns stand for, in ascending order. A **numpy** array: it is a label
        saying which facets the mask covers, as a mirror's facets are.
    """

    crossings: jnp.ndarray
    hidden: jnp.ndarray | None
    receivers: jnp.ndarray
    sources: np.ndarray

    @property
    def n_occluders(self) -> int:
        """How many bodies the mask covers."""
        return int(self.crossings.shape[0])

    def surviving(self, transmittance) -> jnp.ndarray:
        """Fraction of light getting through, per receiver and source.

        Parameters
        ----------
        transmittance : array_like, shape ``(n_occluders,)``
            What fraction each body transmits, in ``[0, 1]``. Differentiable.

        Returns
        -------
        jnp.ndarray, shape ``(n_receivers, n_sources)``
        """
        return reflected_surviving(self.crossings, self.hidden, transmittance)

    def for_receivers(self, points) -> MirrorVisibility:
        """Check that ``points`` are the receivers this mask was built for, and return it.

        The check is :func:`~aquaflux.radiation.visibility.same_receivers`.

        Raises
        ------
        ValueError
            If the points differ, in count or in position.
        """
        same_receivers(self.receivers, points, "mirror visibility mask")
        return self

    def for_sources(self, facets) -> MirrorVisibility:
        """This mask cut down to the columns of ``facets``, which become its sources.

        Parameters
        ----------
        facets : array_like of int, shape ``(k,)``
            Ascending, and every one covered by this mask.

        Returns
        -------
        MirrorVisibility
        """
        facets = np.asarray(facets, dtype=int)
        position = jnp.asarray(self.columns(facets))
        return MirrorVisibility(
            crossings=jnp.take(self.crossings, position, axis=2),
            hidden=None if self.hidden is None else jnp.take(self.hidden, position, axis=1),
            receivers=self.receivers,
            sources=facets,
        )

    def columns(self, facets) -> np.ndarray:
        """The columns standing for ``facets``.

        Parameters
        ----------
        facets : array_like of int, shape ``(k,)``

        Returns
        -------
        numpy.ndarray of int, shape ``(k,)``

        Raises
        ------
        ValueError
            If the mask has no column for one of them: it was built for other sources, and
            reading another facet's column in its place would shadow the wrong path.
        """
        facets = np.asarray(facets, dtype=int)
        position = np.searchsorted(self.sources, facets)
        found = position < len(self.sources)
        found[found] = self.sources[position[found]] == facets[found]
        if not found.all():
            msg = (
                f"this mirror visibility mask has no column for facets "
                f"{facets[~found][:8].tolist()}; it was built for other sources"
            )
            raise ValueError(msg)
        return position


def build_mirror_visibility(
    mirror: Mirror,
    occluders: Sequence,
    surfaces: Surfaces,
    points,
    *,
    receiver_facet=None,
    sources=None,
    self_occlusion: SelfOcclusion | None = None,
    offset_scale: float = 1e-6,
    pair_limit: int = DEFAULT_PAIR_LIMIT,
) -> MirrorVisibility:
    """Work out, once, what stands across each path reflected in ``mirror``.

    Parameters
    ----------
    mirror : Mirror
        Whose facets index ``surfaces``.
    occluders : sequence of aquaflux.solids.Body
        The analytic bodies, each tested against both legs.
    surfaces : Surfaces
        The emitting set, concrete: the paths start at its centroids, and its triangles are the
        blockers the self-occlusion strategy tests.
    points : array_like, shape ``(n_receivers, 3)``
        Receiver positions, concrete.
    receiver_facet : array_like of int, shape ``(n_receivers,)``, optional
        When each receiver is the centroid of one of ``surfaces``' facets, that facet's index,
        which the second leg must ignore -- it ends on it. ``-1`` for a receiver on no facet;
        omitted for receivers in the volume.
    sources : array_like of int, optional
        The facets to build columns for, ascending. Defaults to every facet with some part in
        front of the mirror (:meth:`~aquaflux.radiation.mirrors.Mirror.sources_in_front`), which
        are the only ones with an image a receiver could see.
    self_occlusion : SelfOcclusion or None, optional
        How the surface's own triangles are tested; ``None`` means the default,
        :class:`~aquaflux.radiation.self_occlusion.RayCastOcclusion`, as for the direct mask. Every
        strategy tests a leg with one ray
        (:meth:`~aquaflux.radiation.self_occlusion.SelfOcclusion.segments_hidden`), and
        :class:`~aquaflux.radiation.self_occlusion.NoOcclusion` tests nothing.
    offset_scale : float, optional
        The margins, as a fraction of a facet's size (the square root of its area): at the
        source, of the source facet's, as for the direct mask; at the mirror, of its facets' mean.
    pair_limit : int, optional
        Receiver-by-source pairs per pass, bounding the build's peak memory.

    Returns
    -------
    MirrorVisibility
    """
    points = np.asarray(points, dtype=float)
    occluders = tuple(occluders)
    strategy = (RayCastOcclusion() if self_occlusion is None else self_occlusion).prepared(surfaces)
    sources = (
        mirror.sources_in_front(surfaces) if sources is None else np.asarray(sources, dtype=int)
    )
    on_facet = None if receiver_facet is None else np.asarray(receiver_facet, dtype=int)
    centroid = np.asarray(surfaces.centroid, dtype=float)
    size = np.sqrt(np.asarray(surfaces.area, dtype=float))
    normal = np.asarray(mirror.normal, dtype=float)
    lift = offset_scale * float(np.mean(size[np.asarray(mirror.facets)]))

    n_receivers, n_sources = len(points), len(sources)
    crossings = np.zeros((len(occluders), n_receivers, n_sources), dtype=np.uint8)
    hidden = np.zeros((n_receivers, n_sources), dtype=bool)
    receiver_height = mirror.heights(points)
    source_height = mirror.heights(centroid[sources])
    rows = np.flatnonzero(receiver_height > 0.0)
    tested = source_height > 0.0
    hides = True
    per_pass = receivers_per_pass(pair_limit, max(1, n_sources))
    for start in range(0, len(rows), per_pass):
        local, column = np.nonzero(
            np.broadcast_to(tested, (len(rows[start : start + per_pass]), n_sources))
        )
        if not len(local):
            continue
        row = rows[start : start + per_pass][local]
        facet = sources[column]
        legs = _Legs.of(
            points[row],
            centroid[facet],
            receiver_height[row],
            source_height[column],
            normal,
            lift,
            size[facet] * offset_scale,
        )
        for index, body in enumerate(occluders):
            crossings[index, row, column] = legs.crossed_by(body)
        if hides:
            blocked = legs.hidden_by(
                strategy, surfaces, facet, None if on_facet is None else on_facet[row]
            )
            if blocked is None:
                hides = False
            else:
                hidden[row, column] = blocked
    return MirrorVisibility(
        crossings=jnp.asarray(crossings),
        hidden=jnp.asarray(hidden) if hides else None,
        receivers=jnp.asarray(points),
        sources=sources,
    )


def shadows_reflected_paths(occluders, self_occlusion: SelfOcclusion | None) -> bool:
    """Whether anything can stand across a reflected path: a body, or the surface itself.

    Only when there are no bodies and the surface is told not to shadow itself -- ``None`` means
    the default, which does -- is every path clear, and then no mask need be built at all.
    """
    return bool(tuple(occluders)) or not isinstance(self_occlusion, NoOcclusion)


def build_mirror_masks(
    mirrors: Sequence[Mirror],
    occluders: Sequence,
    surfaces: Surfaces,
    points,
    *,
    receiver_facet=None,
    self_occlusion: SelfOcclusion | None = None,
    **visibility_options,
) -> tuple[MirrorVisibility, ...] | None:
    """One :class:`MirrorVisibility` per mirror, or ``None`` where nothing can shadow a path.

    What a direct mask is built against, a reflected path is built against too: the same bodies
    and the same self-occlusion strategy, so a sleeve or a baffle shadows light on its way to and
    from a mirror exactly as it shadows light coming straight.

    Parameters
    ----------
    mirrors : sequence of Mirror
    occluders, surfaces, points, receiver_facet, self_occlusion
        As for :func:`build_mirror_visibility`.
    **visibility_options
        The options a direct mask is built with. ``offset_scale`` and ``pair_limit`` are read;
        the rest -- how the bodies' layer is culled -- concern the direct mask's tiles of pairs,
        which a reflected path does not form, and are ignored.

    Returns
    -------
    tuple of MirrorVisibility, or None
        ``None`` when :func:`shadows_reflected_paths` says nothing can stand in the way.
    """
    if not shadows_reflected_paths(occluders, self_occlusion):
        return None
    options = {
        key: visibility_options[key]
        for key in ("offset_scale", "pair_limit")
        if key in visibility_options
    }
    if self_occlusion is not None:
        # Whatever the strategy can work out from the surface alone -- a grid over its
        # triangles -- is done once for every mirror rather than once each.
        self_occlusion = self_occlusion.prepared(surfaces)
    return tuple(
        build_mirror_visibility(
            mirror,
            occluders,
            surfaces,
            points,
            receiver_facet=receiver_facet,
            self_occlusion=self_occlusion,
            **options,
        )
        for mirror in mirrors
    )


class _Legs:
    """The two legs of one pass's paths: source to the lifted meeting point, and on to the receiver.

    Every array is padded to the next power of two, so a compiled body test sees a handful of
    shapes across all the passes of all the mirrors rather than one per pass.
    """

    def __init__(self, source, meeting, receiver, near_source, lift, count):
        self.source = source
        self.meeting = meeting
        self.receiver = receiver
        self.near_source = near_source
        self.lift = lift
        self.count = count

    @classmethod
    def of(cls, receivers, sources, receiver_height, source_height, normal, lift, near_source):
        """The legs from each source to each receiver, through the mirror with this ``normal``.

        The source's image lies ``source_height`` behind the plane, so the line from the receiver
        to it crosses the plane the share ``receiver_height / (receiver_height + source_height)``
        of the way along.
        """
        count = len(receivers)
        share = receiver_height / (receiver_height + source_height)
        image = sources - 2.0 * source_height[:, None] * normal
        meeting = receivers + share[:, None] * (image - receivers) + lift * normal
        pad = padded_length(count) - count

        def padded(array):
            return np.concatenate([array, np.repeat(array[-1:], pad, axis=0)]) if pad else array

        return cls(
            padded(sources),
            padded(meeting),
            padded(receivers),
            padded(near_source),
            np.full(count + pad, lift),
            count,
        )

    def crossed_by(self, body) -> np.ndarray:
        """How many of the two legs ``body`` lies across, per path."""
        out = _body_blocks(
            body, jnp.asarray(self.source), jnp.asarray(self.meeting), jnp.asarray(self.near_source)
        )
        back = _body_blocks(
            body, jnp.asarray(self.meeting), jnp.asarray(self.receiver), jnp.asarray(self.lift)
        )
        both = np.asarray(out, dtype=np.uint8) + np.asarray(back, dtype=np.uint8)
        return both[: self.count]

    def hidden_by(self, strategy: SelfOcclusion, surfaces, facet, receiver_facet):
        """Whether the surface's own triangles stand across either leg, or ``None`` if never.

        The first leg ignores the source facet it leaves; the second the facet its receiver sits
        on, if any. Neither needs to ignore the mirror: the lift keeps both off its plane.
        """
        count = self.count
        out = strategy.segments_hidden(
            surfaces,
            self.source[:count],
            self.meeting[:count],
            self.near_source[:count],
            facet[:, None],
        )
        if out is None:
            return None
        landing = np.full(count, -1) if receiver_facet is None else receiver_facet
        back = strategy.segments_hidden(
            surfaces,
            self.meeting[:count],
            self.receiver[:count],
            self.lift[:count],
            landing[:, None],
        )
        return out | back
