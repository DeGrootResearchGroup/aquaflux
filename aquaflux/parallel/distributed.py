"""Distributed (sharded) residual: run a partition's own residual assembler under ``shard_map``.

Combines the decomposition (:mod:`~aquaflux.parallel.partition`), the uniform-shape padding
(:mod:`~aquaflux.parallel.padding`), and the halo exchange (:mod:`~aquaflux.parallel.halo`) to
evaluate a cell residual across devices.

The per-device program is **not** a distributed re-implementation of the residual: each device
holds a real residual assembler built on its own padded local mesh, and the sharded body just
refreshes the halo and calls it. So the distributed residual runs exactly the operators, schemes,
properties, and boundary closures the serial residual runs, and matches it bit-for-bit — value and
gradient — with nothing to keep in step by hand.

Nothing here names a physical operator. The caller injects a builder,
``assemble(mesh, geometry) -> assembler``, which is applied per partition; this module only needs
what every assembler already exposes:

- ``assembler.residual(field)`` — the cell residual on that local mesh;
- ``assembler.boundary`` — the mesh-bound :class:`~aquaflux.boundary.BoundaryConditions`, whose
  per-patch face-index arrays are padded to a common length here (a patch holds a different number
  of faces on each partition, and ``shard_map`` needs uniform shapes).

The P per-partition assemblers are then stacked into a single pytree with a leading partition axis
and passed to ``shard_map`` as a sharded input, so each device receives its own. Everything that is
the *same* on every partition — the operators, the property model, the boundary closures themselves
— is carried along in that stack; the only genuinely per-partition data is the mesh, the geometry,
and the patch bindings.

The halo exchange is an injected :class:`~aquaflux.parallel.halo.HaloExchange` collective, whose
adjoint JAX derives automatically; it defaults to the neighbour-only
:class:`~aquaflux.parallel.halo.AllToAllHaloExchange` (exchanging only boundary cells), with the
all-gather variant available as a simple reference.

On a non-orthogonal mesh the residual reads a reconstructed cell gradient at each face, and a
ghost cell's gradient cannot be reconstructed locally (its stencil reaches past the one-cell halo).
So the field is exchanged in **one layer, twice**: exchange ``phi``, let the assembler reconstruct
the gradient (correct on owned cells, whose stencil is within owned + the ``phi`` halo), then
exchange the gradient so each ghost carries the value its owning partition computed — the assembler's
``gradient_hook`` seam. Both exchanges reuse the same :class:`~aquaflux.parallel.halo.HaloExchange`,
which is generic in the trailing axes, so a per-cell vector gradient rides the identical plan.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh as DeviceMesh
from jax.sharding import PartitionSpec as Pspec

from aquaflux.boundary import BoundaryConditions
from aquaflux.mesh import Mesh, MeshGeometry

from .halo import AllToAllHaloExchange, HaloExchange, HaloPlan
from .padding import PaddedLayout, pad_partition
from .partition import PartitionedMesh

# Name of the device axis the partitions are mapped over.
_DEVICE_AXIS = "p"


def _uniform_boundary_faces(assemblers: list, fill_face: int) -> list:
    """Pad every assembler's per-patch boundary-face indices to one length across partitions.

    A named patch holds a different number of faces on each partition (and none at all on a
    partition that does not touch it), but ``shard_map`` requires uniform shapes. Each patch's
    index array is therefore padded to the largest count over all partitions, filling with the
    reserved null face.

    This is safe because the boundary fold writes values with ``.at[faces].set(...)``: the padded
    entries write a boundary value at the null face, which has zero area and is owned by the null
    cell, so it contributes no flux to any real row. The closure evaluated for those entries reads
    the null cell's benign, finite geometry, so it cannot introduce a ``NaN``.

    Parameters
    ----------
    assemblers : list
        One assembler per partition, each exposing a mesh-bound ``boundary``.
    fill_face : int
        The reserved null-face index to pad with.

    Returns
    -------
    list
        The assemblers, with every patch binding padded to a common length.
    """
    targets = {
        name: max(int(a.boundary.faces[name].shape[0]) for a in assemblers)
        for name in assemblers[0].boundary.conditions
    }
    padded = []
    for a in assemblers:
        faces = {}
        for name, target in targets.items():
            real = np.asarray(a.boundary.faces[name])
            fill = np.full(target - real.shape[0], fill_face, dtype=real.dtype)
            faces[name] = jnp.asarray(np.concatenate([real, fill]))
        padded.append(
            eqx.tree_at(
                lambda t: t.boundary, a, BoundaryConditions(a.boundary.conditions, _faces=faces)
            )
        )
    return padded


class DistributedResidual(eqx.Module):
    """A cell residual evaluated across devices, one partition per device.

    Built by :func:`build_distributed_residual`. Holds the padded layout, the stacked
    per-partition assemblers, and the halo strategy; :meth:`residual` is the sharded evaluation,
    which takes and returns ordinary *global* per-cell vectors, so it is a drop-in for the serial
    ``assembler.residual`` and is differentiable in the same way.

    Attributes
    ----------
    layout : PaddedLayout
        The uniform per-partition sizes, owned→global map, and halo plan.
    assemblers : object
        The per-partition assemblers stacked into one pytree with a leading partition axis.
    halo : HaloExchange
        The ghost-cell refresh strategy.
    halo_plan : HaloPlan
        That strategy's index bookkeeping, stacked over the partition axis and sharded alongside
        the assemblers.
    exchange_gradient : bool
        Whether to exchange ghost-cell gradients (static). ``True`` when the assembler reconstructs
        a gradient (a non-orthogonal correction reads it across partition boundaries); ``False`` for
        an orthogonal grid with no gradient scheme, where the gradient is identically zero and the
        exchange would move only zeros.
    """

    layout: PaddedLayout
    assemblers: object
    halo: HaloExchange
    halo_plan: HaloPlan
    exchange_gradient: bool = eqx.field(static=True)

    def residual(self, global_field: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the residual across devices and reassemble it into a global vector.

        Requires at least ``layout.n_partitions`` devices (use simulated CPU devices for testing).

        Parameters
        ----------
        global_field : jnp.ndarray
            The global cell field, shape ``(n_global_cells,)``.

        Returns
        -------
        jnp.ndarray
            The global residual, shape ``(n_global_cells,)``.
        """
        layout = self.layout
        n_owned_max = layout.n_owned_max
        exchange_gradient = self.exchange_gradient
        device_mesh = DeviceMesh(
            np.array(jax.devices()[: layout.n_partitions]), axis_names=(_DEVICE_AXIS,)
        )
        owned_states = layout.owned_states_from_global(global_field)

        def per_device(owned_shard, plan_shard, assembler_shard):
            # Drop the size-1 partition axis each shard carries, recovering this device's own data.
            owned = owned_shard[0]
            plan = jax.tree.map(lambda a: a[0], plan_shard)
            assembler = jax.tree.map(lambda a: a[0], assembler_shard)

            def fill_local(owned_rows):
                # Exchange owned rows into owned + ghost (the exchange issues its own collectives,
                # whose adjoint JAX derives), then append the null cell: it mirrors no remote cell,
                # nothing reads it, and every padding face scatters into it.
                filled = self.halo.fill(owned_rows, plan, _DEVICE_AXIS)
                null_row = jnp.zeros((1, *filled.shape[1:]), filled.dtype)
                return jnp.concatenate([filled, null_row])

            local_field = fill_local(owned)
            # A ghost cell's reconstructed gradient is wrong (its stencil is incomplete locally), so
            # overwrite the ghost rows with the value each ghost's owning partition computed — the
            # same one-layer exchange, now carrying a per-cell vector. Owner-cell gradients (all a
            # ghost gradient could corrupt is a partition-boundary face's non-orthogonal correction)
            # are untouched, so the reconstruction and boundary values stay serial-exact.
            hook = (
                (lambda gradient: fill_local(gradient[:n_owned_max])) if exchange_gradient else None
            )
            residual = assembler.residual(local_field, gradient_hook=hook)
            return residual[:n_owned_max].reshape(1, n_owned_max)

        sharded = jax.shard_map(
            per_device,
            mesh=device_mesh,
            in_specs=(Pspec(_DEVICE_AXIS),) * 3,
            out_specs=Pspec(_DEVICE_AXIS),
        )
        owned_out = sharded(owned_states, self.halo_plan, self.assemblers)
        return layout.scatter_owned_to_global(owned_out)


def build_distributed_residual(
    pmesh: PartitionedMesh,
    global_geometry: MeshGeometry,
    assemble: Callable[[Mesh, MeshGeometry], object],
    *,
    halo: HaloExchange | None = None,
) -> DistributedResidual:
    """Build a sharded residual from a decomposition and a per-partition assembler builder.

    Pads each partition to uniform shapes, applies ``assemble`` to the padded local mesh and its
    gathered geometry, uniformizes the boundary-patch bindings, and stacks the result along a
    leading partition axis ready for ``shard_map``.

    Parameters
    ----------
    pmesh : PartitionedMesh
        The decomposition, from :func:`~aquaflux.parallel.partition.partition_mesh`.
    global_geometry : MeshGeometry
        The global geometry (``mesh.geometry()``); gathered per partition, never recomputed, so the
        local rows agree with the serial residual exactly.
    assemble : callable
        ``assemble(mesh, geometry) -> assembler``, applied once per partition. The assembler must
        expose ``residual(field)`` and a mesh-bound ``boundary``. Typically a closure over the
        problem's operators, properties, and boundary closures, e.g.
        ``lambda m, g: ResidualAssembler.build(m, g, properties, operators, boundary)``.
    halo : HaloExchange, optional
        Ghost-cell refresh strategy (default
        :class:`~aquaflux.parallel.halo.AllToAllHaloExchange`, which exchanges only boundary cells).

    Returns
    -------
    DistributedResidual

    Raises
    ------
    ValueError
        If the per-partition assemblers do not share one pytree structure once padded — which means
        something varies per partition that the padding does not cover.
    """
    layout = PaddedLayout.from_partitioned(pmesh)
    assemblers = [
        assemble(*pad_partition(layout, p, part.mesh, part.local_geometry(global_geometry)))
        for p, part in enumerate(pmesh.partitions)
    ]
    assemblers = _uniform_boundary_faces(assemblers, layout.null_face)

    if len({jax.tree.structure(a) for a in assemblers}) != 1:
        raise ValueError(
            "the per-partition assemblers do not share one pytree structure, so they cannot be "
            "stacked for shard_map; something varies per partition beyond the padded sizes"
        )
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *assemblers)
    halo = halo or AllToAllHaloExchange()
    # Exchange ghost gradients only when a gradient is actually reconstructed; the schemes are the
    # same on every partition, so one assembler settles it.
    exchange_gradient = assemblers[0].gradient_scheme is not None
    return DistributedResidual(
        layout=layout,
        assemblers=stacked,
        halo=halo,
        halo_plan=halo.plan(layout),
        exchange_gradient=exchange_gradient,
    )
