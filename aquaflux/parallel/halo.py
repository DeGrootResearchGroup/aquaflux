"""Halo exchange: refresh a partition's ghost-cell values from the partitions that own them.

The one new operator the distributed residual needs. Under ``shard_map`` each device holds only its
*owned* cell values; before the residual gathers owner/neighbour state, every partition's ghost
(halo) cells must be filled with the current values of the remote owned cells they mirror. This is
the analogue of a paired send/receive between processes — expressed here as a JAX collective, so
its reverse-mode adjoint is derived automatically (the transpose of an all-gather is a
reduce-scatter; the transpose of a gather-by-index is a scatter-add), giving a differentiable
distributed residual with no hand-written adjoint communication.

A strategy owns **both halves** of the exchange:

- :meth:`HaloExchange.plan` runs once at setup, off the jit path, and returns the per-partition
  index bookkeeping that strategy needs, stacked along a leading partition axis so ``shard_map``
  can shard it;
- :meth:`HaloExchange.fill` runs inside the sharded residual and issues **its own** collectives.

Keeping the collective inside ``fill`` is what makes the strategies genuinely interchangeable: an
all-gather and a neighbour-only permutation need different communication *and* different plans, so
an interface that named either one in its signature could only ever describe the strategy it was
written for.

:class:`AllGatherHaloExchange` gathers every partition's owned values onto every partition: simple,
exactly correct for any partition graph, and O(n_partitions × owned) per device — right at modest
partition counts. A neighbour-only variant, exchanging with just the partitions a given one
actually borders, is the scaling upgrade behind this same interface.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp

from .padding import PaddedLayout


class HaloPlan(eqx.Module):
    """Per-partition index bookkeeping for a halo exchange, stacked over the partition axis.

    Built by :meth:`HaloExchange.plan` and consumed by :meth:`HaloExchange.fill`. Each concrete
    exchange defines its own plan contents; the type exists so the sharded residual can carry and
    shard a plan without knowing which strategy produced it.
    """


class HaloExchange(eqx.Module):
    """Strategy interface: fill a partition's ghost cells from remote owned values."""

    @abc.abstractmethod
    def plan(self, layout: PaddedLayout) -> HaloPlan:
        """Build the exchange's index bookkeeping from the padded layout (setup, off the jit path).

        Parameters
        ----------
        layout : PaddedLayout
            The padded decomposition: the per-partition sizes and the ghost→(owning partition,
            owned index) map.

        Returns
        -------
        HaloPlan
            The plan, with every array carrying a leading partition axis so the sharded residual
            can pass it through ``shard_map`` as a sharded input.
        """

    @abc.abstractmethod
    def fill(self, owned_local: jnp.ndarray, plan: HaloPlan, axis_name: str) -> jnp.ndarray:
        """Return this partition's local field: its owned rows, then its filled ghost rows.

        Runs inside ``shard_map`` and issues whatever collectives the strategy needs over
        ``axis_name``. Generic in the trailing axes, so it carries scalar ``(n,)`` fields and
        vector or tensor ``(n, dim, ...)`` fields alike — which is what exchanging a reconstructed
        gradient or a limiter needs.

        Parameters
        ----------
        owned_local : jnp.ndarray
            This partition's owned-cell values, shape ``(n_owned_max, ...)``.
        plan : HaloPlan
            This partition's slice of the plan from :meth:`plan`, with the leading partition axis
            already removed.
        axis_name : str
            Name of the device axis to communicate over.

        Returns
        -------
        jnp.ndarray
            The local field on owned + ghost cells, shape ``(n_owned_max + n_ghost_max, ...)``.
        """


class AllGatherPlan(HaloPlan):
    """Where each ghost cell's value lives: its owning partition, and its index within it.

    Attributes
    ----------
    src_partition : jnp.ndarray
        Owning partition of each ghost cell, shape ``(n_partitions, n_ghost_max)``.
    src_owned_index : jnp.ndarray
        That ghost's owned-index within its owning partition, same shape.
    """

    src_partition: jnp.ndarray
    src_owned_index: jnp.ndarray


class AllGatherHaloExchange(HaloExchange):
    """Fill ghosts by indexing into an all-gathered stack of every partition's owned values.

    Correct for any partition graph, however irregular: each ghost reads its exact remote owner,
    with no assumption that partitions form a structured neighbour set. The cost is that every
    device receives every partition's owned values, so communication and memory are
    O(n_partitions × owned_max) per device — fine at modest partition counts, and the reason a
    neighbour-only variant is the scaling upgrade.
    """

    def plan(self, layout: PaddedLayout) -> AllGatherPlan:
        """Take the ghost→(owning partition, owned index) map straight from the padded layout."""
        return AllGatherPlan(
            src_partition=layout.ghost_src_partition,
            src_owned_index=layout.ghost_src_owned_index,
        )

    def fill(self, owned_local: jnp.ndarray, plan: AllGatherPlan, axis_name: str) -> jnp.ndarray:
        """All-gather every partition's owned values, then index each ghost's own source."""
        # `tiled=False` stacks the shards on a new leading axis, giving
        # `(n_partitions, n_owned_max, ...)` — the layout the plan's index pair addresses.
        all_owned = jax.lax.all_gather(owned_local, axis_name, axis=0, tiled=False)
        ghost = all_owned[plan.src_partition, plan.src_owned_index]
        return jnp.concatenate([owned_local, ghost], axis=0)
