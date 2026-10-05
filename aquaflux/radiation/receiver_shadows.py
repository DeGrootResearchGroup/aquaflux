"""How a radiation model shadows its receivers: a mask held whole, or one built chunk by chunk.

A model's receivers — cell centres, usually — are shadowed by the same bodies as its facets, and
the shadow is a receiver-by-facet mask per body. Held whole it is built once and every later call
reads it, which is what a design study sweeping lamp power or the medium's absorbance over one
scene wants.
But its size is the receiver count times the facet count per body: at a reactor mesh's 1.6 million
cells against a lamp's 7,516 facets that is 12 GB per body, which cannot be held at all.

So there are two strategies, and **which one is the caller's choice, made at build**, because it
trades memory against repeated work and only the caller knows which it can afford:

* :class:`FrozenShadows` — the whole mask, built once. The default.
* :class:`StreamedShadows` — nothing held; each chunk's mask is built when a field is asked for
  and dropped with the chunk, in a gradient as well as in the forward pass. Memory is set by the
  chunk; every call pays the mask build again.

Both give the same field and the same gradients. The choice is about memory, not about
differentiability.

**Each strategy also gathers what reaches the receivers by one bounce in a mirror**, shadowed by a
mask per mirror of the same bodies and the same self-occlusion
(:mod:`~aquaflux.radiation.mirror_visibility`), held whole or built per chunk alongside the direct
mask -- so the choice governs the reflected paths' masks too.
"""

from __future__ import annotations

import abc
import dataclasses

import equinox as eqx
import jax.numpy as jnp

from aquaflux.radiation.absorption import Absorption
from aquaflux.radiation.gather import streamed_fluence_rate, summed_fluence_rate
from aquaflux.radiation.images import summed_mirrored_fluence_rate
from aquaflux.radiation.mirror_visibility import MirrorVisibility, build_mirror_masks
from aquaflux.radiation.mirrors import Mirror
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.visibility import Visibility, build_visibility, refuse_points_inside
from aquaflux.radiation.work import DEFAULT_PAIR_LIMIT

__all__ = ["FrozenShadows", "ReceiverShadows", "StreamedShadows"]


class ReceiverShadows(eqx.Module):
    """Strategy interface: the summed fluence rate of surface sets at shadowed receivers."""

    @abc.abstractmethod
    def fluence_rate(
        self,
        sets,
        receivers,
        *,
        absorption: Absorption | None = None,
        transmittance=None,
        pair_limit: int | None = None,
    ) -> jnp.ndarray:
        """The fluence rate the sets deliver at the receivers, summed, shape ``(n_receivers,)``.

        Straight from each source, and by one bounce in each of the mirrors the strategy was
        built with. Every set must have the geometry the strategy was built for: they cast its
        shadows.
        """


def _pair_limit(pair_limit: int | None) -> dict:
    return {} if pair_limit is None else {"pair_limit": pair_limit}


def _mirrored(sets, mirrors, receivers, masks, absorption, transmittance, pair_limit):
    """What the mirrors send the receivers, through ``masks`` -- or unshadowed when ``None``."""
    if not mirrors:
        return 0.0
    return summed_mirrored_fluence_rate(
        sets,
        mirrors,
        receivers,
        absorption=absorption,
        shadows=masks,
        transmittance=None if masks is None else transmittance,
        **_pair_limit(pair_limit),
    )


class FrozenShadows(ReceiverShadows):
    """The whole receiver mask, built once and read by every call.

    Attributes
    ----------
    visibility : Visibility
        The mask, for the model's receivers.
    mirrors : tuple of Mirror
        The planes light reaches the receivers by one bounce in.
    mirror_visibility : tuple of MirrorVisibility, or None
        One mask per mirror, of what stands across each reflected path; ``None`` where nothing
        can.
    """

    visibility: Visibility
    mirrors: tuple[Mirror, ...] = ()
    mirror_visibility: tuple[MirrorVisibility, ...] | None = None

    @classmethod
    def build(
        cls, occluders, surfaces, receivers, *, mirrors=(), **visibility_options
    ) -> FrozenShadows:
        """Build the masks for these receivers: the direct one, and one per mirror."""
        mirrors = tuple(mirrors)
        return cls(
            visibility=build_visibility(occluders, surfaces, receivers, **visibility_options),
            mirrors=mirrors,
            mirror_visibility=build_mirror_masks(
                mirrors, occluders, surfaces, receivers, **visibility_options
            )
            if mirrors
            else None,
        )

    def fluence_rate(
        self, sets, receivers, *, absorption=None, transmittance=None, pair_limit=None
    ):
        """The sets gathered through the held masks, each in one shared pass.

        See :meth:`ReceiverShadows.fluence_rate`.
        """
        direct = summed_fluence_rate(
            sets,
            receivers,
            absorption=absorption,
            visibility=self.visibility,
            transmittance=transmittance,
            **_pair_limit(pair_limit),
        )
        return direct + _mirrored(
            sets,
            self.mirrors,
            receivers,
            self.mirror_visibility,
            absorption,
            transmittance,
            pair_limit,
        )


class StreamedShadows(ReceiverShadows):
    """No mask held: each chunk's is built when a field is asked for, and dropped with it.

    Attributes
    ----------
    geometry : Surfaces
        The surface set the model was built for. Masks are built from it rather than from the
        sets a call supplies, so every call is shadowed by the build's geometry, as it is with a
        held mask.
    occluders : tuple of aquaflux.solids.Body
        The bodies in the way.
    visibility_options : dict
        What the mask builds are told — the self-occlusion strategy among them.
    mirrors : tuple of Mirror
        The planes light reaches the receivers by one bounce in. Each chunk builds a mask per
        mirror beside its direct one, and drops them with it.
    """

    geometry: Surfaces
    occluders: tuple
    visibility_options: dict
    mirrors: tuple[Mirror, ...] = ()

    @classmethod
    def build(
        cls, occluders, surfaces, receivers, *, mirrors=(), **visibility_options
    ) -> StreamedShadows:
        """Keep what the masks are built from, and refuse now a receiver inside a body.

        Checked here rather than left to the first call, which is where a held mask would have
        refused it: a scene with a point embedded in metal is wrong at build, whichever strategy
        holds its shadows.
        """
        occluders = tuple(occluders)
        refuse_points_inside(occluders, surfaces, receivers)
        return cls(
            geometry=surfaces,
            occluders=occluders,
            visibility_options=visibility_options,
            mirrors=tuple(mirrors),
        )

    def fluence_rate(
        self, sets, receivers, *, absorption=None, transmittance=None, pair_limit=None
    ):
        """The sets gathered chunk by chunk, each chunk's masks shared by all of them."""
        limit = DEFAULT_PAIR_LIMIT if pair_limit is None else pair_limit
        options = dict(self.visibility_options)
        if options.get("self_occlusion") is not None and self.mirrors:
            # Prepared once for every chunk's mirror masks, as the stream prepares it once for
            # every chunk's direct mask.
            options["self_occlusion"] = options["self_occlusion"].prepared(self.geometry)
        return streamed_fluence_rate(
            sets,
            receivers,
            shadow_geometry=self.geometry,
            occluders=self.occluders,
            visibility_options=self.visibility_options,
            absorption=absorption,
            transmittance=transmittance,
            extra=_ChunkMirrors(self.mirrors, self.occluders, self.geometry, options, limit)
            if self.mirrors
            else None,
            **_pair_limit(pair_limit),
        )


@dataclasses.dataclass(frozen=True, eq=False)
class _ChunkMirrors:
    """What one streamed chunk receives by the mirrors, through masks built for the chunk alone.

    Called by the stream inside each chunk's own gradient rule, so its masks are built and dropped
    with the chunk, forward and on the way back. They are built from the stream's geometry, which
    is concrete, never from the live sets, which are traced on the way back.
    """

    mirrors: tuple
    occluders: tuple
    geometry: Surfaces
    options: dict
    pair_limit: int

    def field(self, live, points):
        """The chunk's share, ``(n_chunk,)``, from ``live = (sets, absorption, transmittance)``."""
        sets, absorption, transmittance = live
        masks = build_mirror_masks(
            self.mirrors, self.occluders, self.geometry, points, **self.options
        )
        return _mirrored(
            sets, self.mirrors, points, masks, absorption, transmittance, self.pair_limit
        )
