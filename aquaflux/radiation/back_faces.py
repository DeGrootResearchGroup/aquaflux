"""Which receivers lie certainly behind which facets, pair by pair or a tile of pairs at a time.

A facet whose outward normal points away from a receiver sends that receiver nothing when its
profile is dark behind itself (:attr:`~aquaflux.radiation.profiles.Profile.dark_behind`): the
gather weights the pair by the profile's radiance towards the receiver, which is then exactly
zero. So whether anything stands between the two cannot change the field, and a shadow mask can
record such a pair clear without testing it. This is the test of which pairs those are.

**The sign is the receiver's height above the facet's plane, and only a sign that rounding
cannot have set is believed.** A height too close to zero to trust is resolved to zero
(:func:`~aquaflux.radiation.clipping.decidable_heights`), and zero keeps the pair, so a pair is
recorded as behind only where the gather's own cosine is certainly not positive.

**A tile of pairs is decided from the receivers' bounding box.** The largest height any point of
a box reaches above a plane is the height of the box corner the normal points towards, so a
facet is behind every receiver of a block when that corner is behind it -- by a margin wide
enough that the per-pair test, run on any point of the box, also finds it behind. A tile decided
this way therefore never contradicts the pair test on any of its pairs; it is the same answer
reached without forming the pairs.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numba
import numpy as np

from aquaflux.radiation.clipping import _SLACK, decidable_heights

__all__ = ["BackFaces"]

#: How many times the pair test's own rounding allowance a box must clear to be decided whole.
#: The box's corner height and a pair's height are each wrong by a few roundings of the same
#: magnitudes, and a pair is only behind past one allowance; three leave room for both errors.
_BOX_SLACK = 3.0 * _SLACK


def _kept(array):
    """A traced array as it is, a concrete one on the host.

    Not through ``jnp``: inside a trace that stages even a concrete array, and the box tests
    read these positions on the host.
    """
    return array if isinstance(array, jax.core.Tracer) else np.asarray(array, dtype=float)


@jax.jit
def _certainly_behind(receivers, normal, centroid):
    """Whether each receiver lies certainly behind each plane, all arrays broadcast together."""
    height = decidable_heights(receivers[..., None, :], normal, through=centroid)
    return height[..., 0] < 0.0


class BackFaces(eqx.Module):
    """The facets' planes, for asking which receivers lie certainly behind which facets.

    Attributes
    ----------
    centroid : np.ndarray or jnp.ndarray, shape ``(n_facets, 3)``
        A point on each facet's plane: held on the host unless it is traced.
    normal : np.ndarray or jnp.ndarray, shape ``(n_facets, 3)``
        Each facet's outward unit normal.
    areal : np.ndarray of bool, shape ``(n_facets,)``
        Which facets have a plane at all. A point source has none, so no receiver is ever behind
        it; it is excluded by its label rather than by its zero normal.
    """

    centroid: jnp.ndarray
    normal: jnp.ndarray
    areal: np.ndarray

    @classmethod
    def of(cls, surfaces) -> BackFaces:
        """The planes of a surface set's facets.

        Parameters
        ----------
        surfaces : Surfaces

        Returns
        -------
        BackFaces
        """
        return cls(
            centroid=_kept(surfaces.centroid),
            normal=_kept(surfaces.normal),
            areal=~np.asarray(surfaces.is_point_source),
        )

    @property
    def n_facets(self) -> int:
        """How many facets there are."""
        return int(self.areal.shape[0])

    def behind(self, receivers, facets) -> jnp.ndarray:
        """Whether each receiver lies certainly behind the facet paired with it.

        Parameters
        ----------
        receivers : array_like, shape ``(..., 3)``
            Receiver positions. May be traced.
        facets : array_like of int, shape broadcastable to ``receivers.shape[:-1]``
            The facet each receiver is paired with.

        Returns
        -------
        jnp.ndarray of bool, broadcast shape of ``receivers.shape[:-1]`` and ``facets``
        """
        facets = np.asarray(facets)
        away = _certainly_behind(
            jnp.asarray(receivers, dtype=float),
            jnp.asarray(self.normal)[facets],
            jnp.asarray(self.centroid)[facets],
        )
        return away & jnp.asarray(self.areal[facets])

    def every_pair(self, receivers) -> jnp.ndarray:
        """:meth:`behind` for every receiver against every facet.

        Parameters
        ----------
        receivers : array_like, shape ``(n_receivers, 3)``

        Returns
        -------
        jnp.ndarray of bool, shape ``(n_receivers, n_facets)``
        """
        receivers = jnp.asarray(receivers, dtype=float)
        return self.behind(receivers[:, None, :], np.arange(self.n_facets)[None, :])

    def tiles_behind(self, receivers, receiver_groups, facet_groups) -> np.ndarray:
        """Which tiles have every receiver certainly behind every facet, from the receivers' box.

        A tile this says is behind is behind by :meth:`behind` on every one of its pairs; a
        tile it does not may still be, and is left for the pairs to decide. One compiled loop
        over the tiles, each stopping at the first facet it cannot prove, so nothing the size of
        the tiles times their facets is ever formed.

        Parameters
        ----------
        receivers : array_like, shape ``(n_receivers, 3)``
            Receiver positions; concrete.
        receiver_groups : np.ndarray of int, shape ``(n_tiles, block)``
            Each tile's receiver indices. Repeats are harmless.
        facet_groups : np.ndarray of int, shape ``(n_tiles, cluster)``
            Each tile's facet indices. Repeats are harmless.

        Returns
        -------
        np.ndarray of bool, shape ``(n_tiles,)``
        """
        return _boxes_behind(
            np.ascontiguousarray(receivers, dtype=float),
            np.ascontiguousarray(receiver_groups, dtype=np.int64),
            np.ascontiguousarray(facet_groups, dtype=np.int64),
            np.ascontiguousarray(self.centroid, dtype=float),
            np.ascontiguousarray(self.normal, dtype=float),
            np.ascontiguousarray(self.areal, dtype=np.bool_),
            _BOX_SLACK,
        )

    def lit_facets(self, receivers, receiver_groups, facets) -> tuple[np.ndarray, np.ndarray]:
        """Per group of receivers, the facets its box cannot be proven to lie wholly behind.

        The complement, per group, of what :meth:`tiles_behind` proves for a tile of that group
        and one facet: a facet left out is behind every receiver of the group by :meth:`behind`,
        so for a source dark behind itself it sends that group nothing. Two compiled passes, one
        counting and one filling, so what is formed is the lists themselves and not the groups
        times the facets.

        Parameters
        ----------
        receivers : array_like, shape ``(n_receivers, 3)``
            Receiver positions; concrete.
        receiver_groups : np.ndarray of int, shape ``(n_groups, block)``
            Each group's receiver indices. Repeats are harmless.
        facets : np.ndarray of int, shape ``(n_candidates,)``
            The facets to test, in the order the lists keep them.

        Returns
        -------
        offsets : np.ndarray of int, shape ``(n_groups + 1,)``
            Where each group's list starts in ``lit``: compressed-sparse-row (CSR) form.
        lit : np.ndarray of int, shape ``(offsets[-1],)``
            The facets left in, group after group, each group's in the order of ``facets``.
        """
        arguments = (
            np.ascontiguousarray(receivers, dtype=float),
            np.ascontiguousarray(receiver_groups, dtype=np.int64),
            np.ascontiguousarray(facets, dtype=np.int64),
            np.ascontiguousarray(self.centroid, dtype=float),
            np.ascontiguousarray(self.normal, dtype=float),
            np.ascontiguousarray(self.areal, dtype=np.bool_),
            _BOX_SLACK,
        )
        counts = _count_lit(*arguments)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        return offsets, _fill_lit(*arguments, offsets)


@numba.njit(inline="always")
def _box(receivers, members):
    """The bounding box of the receivers ``members`` indexes, as its low and high corners."""
    low = receivers[members[0]].copy()
    high = low.copy()
    for member in range(1, len(members)):
        point = receivers[members[member]]
        for axis in range(3):
            low[axis] = min(low[axis], point[axis])
            high[axis] = max(high[axis], point[axis])
    return low, high


@numba.njit(inline="always")
def _box_behind(low, high, facet, centroid, normal, areal, slack):
    """Whether a box lies certainly behind one facet's plane.

    The highest point of a box above a plane is the corner the normal points towards. It must
    be below the plane by ``slack`` times the magnitudes its height is formed from, taken at the
    point of the box farthest from the facet along each axis -- so that any receiver in the box,
    tested pair by pair, is certainly behind too. A point source has no plane and nothing is
    behind it.
    """
    if not areal[facet]:
        return False
    highest = 0.0
    reach = 0.0
    for axis in range(3):
        n, c = normal[facet, axis], centroid[facet, axis]
        highest += ((high[axis] if n > 0.0 else low[axis]) - c) * n
        reach += max(abs(low[axis] - c), abs(high[axis] - c)) * abs(n)
    return highest < -slack * reach


@numba.njit(parallel=True)
def _boxes_behind(receivers, receiver_groups, facet_groups, centroid, normal, areal, slack):
    """Whether each tile's receivers' box lies certainly behind every facet of its cluster."""
    n_tiles = receiver_groups.shape[0]
    cluster = facet_groups.shape[1]
    behind = np.zeros(n_tiles, dtype=np.bool_)
    for tile in numba.prange(n_tiles):
        low, high = _box(receivers, receiver_groups[tile])
        proven = True
        for entry in range(cluster):
            if not _box_behind(
                low, high, facet_groups[tile, entry], centroid, normal, areal, slack
            ):
                proven = False
                break
        behind[tile] = proven
    return behind


@numba.njit(parallel=True)
def _count_lit(receivers, receiver_groups, facets, centroid, normal, areal, slack):
    """How many of ``facets`` each group's box cannot be proven behind."""
    n_groups = receiver_groups.shape[0]
    counts = np.zeros(n_groups, dtype=np.int64)
    for group in numba.prange(n_groups):
        low, high = _box(receivers, receiver_groups[group])
        count = 0
        for facet in facets:
            if not _box_behind(low, high, facet, centroid, normal, areal, slack):
                count += 1
        counts[group] = count
    return counts


@numba.njit(parallel=True)
def _fill_lit(receivers, receiver_groups, facets, centroid, normal, areal, slack, offsets):
    """The facets :func:`_count_lit` counted, written group by group at ``offsets``."""
    lit = np.empty(offsets[-1], dtype=np.int64)
    for group in numba.prange(receiver_groups.shape[0]):
        low, high = _box(receivers, receiver_groups[group])
        at = offsets[group]
        for facet in facets:
            if not _box_behind(low, high, facet, centroid, normal, areal, slack):
                lit[at] = facet
                at += 1
    return lit
