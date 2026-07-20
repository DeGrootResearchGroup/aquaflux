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
all-gather and a neighbour ``all_to_all`` need different communication *and* different plans, so an
interface that named either one in its signature could only ever describe the strategy it was
written for.

Two implementations, both correct for an arbitrary (irregular) partition graph:

- :class:`AllGatherHaloExchange` gathers *every* partition's owned values onto every partition.
  Dead simple, but communication and memory are O(n_partitions × owned_max) per device — it does
  not distribute memory, so it is a correctness reference and a small-scale convenience, not the
  scaling path.
- :class:`AllToAllHaloExchange` sends each partition only the boundary cells its neighbours actually
  ghost, via a single ``all_to_all``. Communication is the real halo volume and per-device memory is
  O(n_partitions × max halo *per pair*), so it scales. This is the default. It uses plain
  ``all_to_all`` (uniform per-pair buffers, padded to a common size) rather than a neighbour-only
  ``ppermute``, because a min-cut partitioner produces an *irregular* neighbour set — the number of
  partitions a given one borders varies — which a one-to-one permutation primitive cannot express in
  a fixed schedule, but a general all-to-all handles directly.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

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


class AllToAllPlan(HaloPlan):
    """The pack/unpack bookkeeping for a neighbour exchange over ``all_to_all``.

    Attributes
    ----------
    send_gather : jnp.ndarray
        For each sending partition, which owned cells to pack into the buffer for each destination:
        shape ``(n_partitions_send, n_partitions_dest, n_send_max)``. ``send_gather[s, d, k]`` is the
        owned-index within partition ``s`` of the ``k``-th cell it sends to partition ``d`` (padding
        slots read owned cell 0 — the value lands in a buffer slot no ghost reads).
    src_partition : jnp.ndarray
        Owning partition of each ghost cell, shape ``(n_partitions, n_ghost_max)`` — which received
        buffer to read it from.
    recv_slot : jnp.ndarray
        Position of each ghost cell within its source's buffer, same shape — which slot to read.
    """

    send_gather: jnp.ndarray
    src_partition: jnp.ndarray
    recv_slot: jnp.ndarray


class AllToAllHaloExchange(HaloExchange):
    """Fill ghosts by exchanging only boundary cells, via a single ``all_to_all``.

    Each partition packs, for every other partition, exactly the owned cells that partition ghosts
    from it, and one ``all_to_all`` delivers every packed buffer to its destination; each ghost then
    reads its value out of the buffer from its source. Communication is the true halo volume and
    per-device memory is O(n_partitions × per-pair halo), so unlike the all-gather this actually
    distributes memory.

    Correct for an arbitrary partition graph: the pairwise buffers are padded to one common size, so
    an irregular neighbour set (a partition bordering a variable number of others — what a min-cut
    partitioner produces) needs no special handling. A partition that borders another only lightly
    still exchanges a full (mostly-padding) buffer with it; that padding is the price of using a
    uniform ``all_to_all`` instead of a ragged one, and is small when the per-pair halos are.
    """

    def plan(self, layout: PaddedLayout) -> AllToAllPlan:
        """Build the pack/unpack indices from the ghost→(owning partition, owned index) map.

        Pure build-time numpy (off the jit path). For each destination partition, its ghost cells
        are grouped by source partition, and each is assigned the next free slot in that source's
        buffer; the send side is the transpose of the same enumeration, so the slot a cell is packed
        into on the sender equals the slot the ghost reads on the receiver.
        """
        n_parts = layout.n_partitions
        src = np.asarray(layout.ghost_src_partition)  # (n_partitions, n_ghost_max)
        owned_index = np.asarray(layout.ghost_src_owned_index)
        recv_slot = np.zeros_like(src)
        # (sender, dest) -> owned indices to pack, in the order the destination will read them.
        packed: dict[tuple[int, int], list[int]] = {}
        for dest in range(n_parts):
            next_slot: dict[int, int] = {}
            for g in range(layout.n_ghost[dest]):
                s = int(src[dest, g])
                slot = next_slot.get(s, 0)
                recv_slot[dest, g] = slot
                next_slot[s] = slot + 1
                packed.setdefault((s, dest), []).append(int(owned_index[dest, g]))

        n_send_max = max((len(v) for v in packed.values()), default=1) or 1
        send_gather = np.zeros((n_parts, n_parts, n_send_max), dtype=np.int64)
        for (s, dest), indices in packed.items():
            send_gather[s, dest, : len(indices)] = indices

        return AllToAllPlan(
            send_gather=jnp.asarray(send_gather),
            src_partition=layout.ghost_src_partition,
            recv_slot=jnp.asarray(recv_slot),
        )

    def fill(self, owned_local: jnp.ndarray, plan: AllToAllPlan, axis_name: str) -> jnp.ndarray:
        """Pack per-destination buffers, exchange them, then read each ghost from its source."""
        # Pack: buffer for destination d is this partition's owned cells that d ghosts from it.
        send = owned_local[plan.send_gather]  # (n_partitions_dest, n_send_max, ...)
        # Exchange: send[d] goes to device d; recv[s] is the buffer partition s sent to this one.
        recv = jax.lax.all_to_all(send, axis_name, split_axis=0, concat_axis=0, tiled=True)
        ghost = recv[plan.src_partition, plan.recv_slot]  # (n_ghost_max, ...)
        return jnp.concatenate([owned_local, ghost], axis=0)
