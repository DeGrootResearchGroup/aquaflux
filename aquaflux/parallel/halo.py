"""Halo exchange: refresh a partition's ghost-cell values from the partitions that own them.

The one new operator the distributed residual needs. Under `shard_map` each device holds only its
*owned* cell values; before the residual gathers owner/neighbour state, every partition's ghost
(halo) cells must be filled with the current values of the remote owned cells they mirror. This is
the `MPI_Sendrecv` analogue — expressed here as a JAX collective so its reverse-mode adjoint is
derived automatically (the transpose of an all-gather is a reduce-scatter; the transpose of a
gather-by-index is a scatter-add), giving a differentiable distributed residual with no hand-written
adjoint communication.

The strategy is split from the exchange *mechanism* so the same per-partition fill logic is exercised
both in a plain-Python emulation (stack every partition's owned state) and under `shard_map` (an
`all_gather` over the device axis).

The `AllGatherHaloExchange` here gathers *all* owned values onto every partition: simple and exactly
correct, but O(n_partitions × owned) per device — right while correctness is the priority and partition
counts are modest, to be superseded by a neighbour-only `ppermute` variant behind this same interface.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp


class HaloExchange(eqx.Module):
    """Strategy interface for filling a partition's ghost cells from remote owned values."""

    @abc.abstractmethod
    def fill(
        self,
        owned_local: jnp.ndarray,
        all_owned: jnp.ndarray,
        ghost_src_partition: jnp.ndarray,
        ghost_src_owned_index: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return this partition's full local field (owned rows, then filled ghost rows).

        Parameters
        ----------
        owned_local : jnp.ndarray
            This partition's owned-cell values, shape ``(n_owned, ...)``.
        all_owned : jnp.ndarray
            Every partition's owned values, shape ``(n_partitions, n_owned_max, ...)`` — the
            all-gathered stack (under ``shard_map``, the result of ``all_gather``; in emulation, a
            plain stack). Padded partitions use ``n_owned_max``.
        ghost_src_partition : jnp.ndarray
            Owning partition of each ghost cell, shape ``(n_ghost,)`` (the halo plan).
        ghost_src_owned_index : jnp.ndarray
            Owned-index within its owning partition of each ghost cell, shape ``(n_ghost,)``.

        Returns
        -------
        jnp.ndarray
            The local field on owned + ghost cells, shape ``(n_owned + n_ghost, ...)``.
        """


class AllGatherHaloExchange(HaloExchange):
    """Fill ghosts by indexing into the all-gathered stack of every partition's owned values.

    Correct for any partition graph (each ghost reads its exact remote owner). Communication is the
    full all-gather, so memory is O(n_partitions × owned_max) per device — fine at small partition
    counts; a `ppermute` neighbour-exchange variant is the scaling upgrade.
    """

    def fill(
        self,
        owned_local: jnp.ndarray,
        all_owned: jnp.ndarray,
        ghost_src_partition: jnp.ndarray,
        ghost_src_owned_index: jnp.ndarray,
    ) -> jnp.ndarray:
        ghost = all_owned[ghost_src_partition, ghost_src_owned_index]
        return jnp.concatenate([owned_local, ghost], axis=0)
