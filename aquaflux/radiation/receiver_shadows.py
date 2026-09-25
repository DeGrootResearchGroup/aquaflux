"""How a radiation model shadows its receivers: a mask held whole, or one built chunk by chunk.

A model's receivers — cell centres, usually — are shadowed by the same bodies as its facets, and
the shadow is a receiver-by-facet mask per body. Held whole it is built once and every later call
reads it, which is what a design study sweeping lamp power or water quality over one scene wants.
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
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp

from aquaflux.radiation.absorption import Absorption
from aquaflux.radiation.gather import direct_fluence_rate, streamed_fluence_rate
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.visibility import Visibility, build_visibility, refuse_points_inside

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

        Every set must have the geometry the strategy was built for: they cast its shadows.
        """


def _pair_limit(pair_limit: int | None) -> dict:
    return {} if pair_limit is None else {"pair_limit": pair_limit}


class FrozenShadows(ReceiverShadows):
    """The whole receiver mask, built once and read by every call.

    Attributes
    ----------
    visibility : Visibility
        The mask, for the model's receivers.
    """

    visibility: Visibility

    @classmethod
    def build(cls, occluders, surfaces, receivers, **visibility_options) -> FrozenShadows:
        """Build the mask for these receivers."""
        return cls(build_visibility(occluders, surfaces, receivers, **visibility_options))

    def fluence_rate(
        self, sets, receivers, *, absorption=None, transmittance=None, pair_limit=None
    ):
        """Each set gathered through the held mask. See :meth:`ReceiverShadows.fluence_rate`."""
        return sum(
            direct_fluence_rate(
                surfaces,
                receivers,
                absorption=absorption,
                visibility=self.visibility,
                transmittance=transmittance,
                **_pair_limit(pair_limit),
            )
            for surfaces in sets
        )


class StreamedShadows(ReceiverShadows):
    """No mask held: each chunk's is built when a field is asked for, and dropped with it.

    Attributes
    ----------
    geometry : Surfaces
        The surface set the model was built for. Masks are built from it rather than from the
        sets a call supplies, so a gradient with respect to a lamp's position is taken with the
        shadows frozen, as it is with a held mask.
    occluders : tuple of aquaflux.solids.Body
        The bodies in the way.
    visibility_options : dict
        What the mask builds are told — the self-occlusion strategy among them.
    """

    geometry: Surfaces
    occluders: tuple
    visibility_options: dict

    @classmethod
    def build(cls, occluders, surfaces, receivers, **visibility_options) -> StreamedShadows:
        """Keep what the masks are built from, and refuse now a receiver inside a body.

        Checked here rather than left to the first call, which is where a held mask would have
        refused it: a scene with a point embedded in metal is wrong at build, whichever strategy
        holds its shadows.
        """
        occluders = tuple(occluders)
        refuse_points_inside(occluders, surfaces, receivers)
        return cls(geometry=surfaces, occluders=occluders, visibility_options=visibility_options)

    def fluence_rate(
        self, sets, receivers, *, absorption=None, transmittance=None, pair_limit=None
    ):
        """The sets gathered chunk by chunk, one mask per chunk shared by all of them."""
        return streamed_fluence_rate(
            sets,
            receivers,
            shadow_geometry=self.geometry,
            occluders=self.occluders,
            visibility_options=self.visibility_options,
            absorption=absorption,
            transmittance=transmittance,
            **_pair_limit(pair_limit),
        )
