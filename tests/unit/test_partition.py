"""Gate 1a — domain decomposition reproduces the serial residual (and its gradient).

Before any `shard_map` or collective, this proves the *decomposition itself* is correct: split the
mesh into partitions with halo rings, gather each partition's local field (owned + ghost) from the
true global field, run the existing serial residual on each local mesh, and scatter the owned rows
back — the result must equal the serial residual cell-for-cell, and differentiating a parameter
through it must match the serial gradient. The halo is filled here directly from the global field
(no collective yet); replacing that fill with a `shard_map` all-gather is Gate 1b.

Uses an **orthogonal** grid with no gradient scheme, so the non-orthogonal correction vanishes and a
single ghost layer is sufficient (the two-layer / gradient-halo case is a documented follow-on).
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet
from aquaflux.discretization import DiffusionFlux, ResidualAssembler
from aquaflux.mesh import structured_grid_3d
from aquaflux.parallel import AllGatherHaloExchange, partition_mesh
from aquaflux.properties import Constant, PropertyModel

SIDES = ("left", "right", "bottom", "top", "back", "front")
BC_VALUES = {"left": 1.0, "right": 0.0, "bottom": 0.3, "top": -0.2, "back": 0.5, "front": 0.1}


def _boundary():
    return {name: Dirichlet(BC_VALUES[name]) for name in SIDES}


def _serial_assembler(mesh, gamma_val):
    geom = mesh.geometry()
    properties = PropertyModel({"diffusivity": Constant(gamma_val)})
    return ResidualAssembler.build(
        mesh,
        geom,
        properties,
        (DiffusionFlux(),),
        BoundaryConditions(_boundary()),
    )


def _local_assembler(part, global_geom, gamma_val):
    geom = part.local_geometry(global_geom)
    properties = PropertyModel({"diffusivity": Constant(gamma_val)})
    return ResidualAssembler.build(
        part.mesh,
        geom,
        properties,
        (DiffusionFlux(),),
        BoundaryConditions(_boundary()),
    )


def _distributed_residual(pmesh, global_geom, gamma_val, global_phi):
    """Emulated distributed residual: gather ghosts from the global field, assemble, scatter owned."""
    owned = []
    for part in pmesh.partitions:
        asm = _local_assembler(part, global_geom, gamma_val)
        local_res = asm.residual(part.gather_cells(global_phi))
        owned.append(local_res[: part.n_owned])
    return pmesh.scatter_owned(owned)


def _slab_labels(n_cells, n_partitions):
    """Contiguous index blocks — planar z-slabs for the structured numbering (small interfaces)."""
    return (np.arange(n_cells) * n_partitions // n_cells).astype(np.int64)


def _all_owned_stack(pmesh, global_field):
    """Emulate the `shard_map` all-gather: stack every partition's owned values, padded uniform."""
    n_owned_max = max(p.n_owned for p in pmesh.partitions)
    rows = []
    for part in pmesh.partitions:
        owned = global_field[part.owned_global]
        pad = jnp.zeros(n_owned_max - part.n_owned, dtype=global_field.dtype)
        rows.append(jnp.concatenate([owned, pad]))
    return jnp.stack(rows)  # (n_partitions, n_owned_max)


def _distributed_residual_via_halo(pmesh, global_geom, gamma_val, global_phi):
    """Distributed residual with ghosts filled by the halo plan (not a direct global gather)."""
    all_owned = _all_owned_stack(pmesh, global_phi)
    halo = AllGatherHaloExchange()
    owned = []
    for part in pmesh.partitions:
        local_full = halo.fill(
            global_phi[part.owned_global],
            all_owned,
            part.ghost_src_partition,
            part.ghost_src_owned_index,
        )
        asm = _local_assembler(part, global_geom, gamma_val)
        owned.append(asm.residual(local_full)[: part.n_owned])
    return pmesh.scatter_owned(owned)


def test_partition_covers_every_cell_once() -> None:
    mesh = structured_grid_3d(5, 5, 5, named_boundaries=True)
    pmesh = partition_mesh(mesh, _slab_labels(mesh.n_cells, 3))
    owned = np.concatenate([np.asarray(p.owned_global) for p in pmesh.partitions])
    assert np.array_equal(np.sort(owned), np.arange(mesh.n_cells))  # disjoint + complete
    assert sum(p.n_owned for p in pmesh.partitions) == mesh.n_cells
    assert all(p.n_ghost > 0 for p in pmesh.partitions)  # interfaces produce halos


@pytest.mark.parametrize("n_partitions", [2, 3, 4])
def test_distributed_residual_matches_serial(n_partitions) -> None:
    mesh = structured_grid_3d(6, 6, 6, named_boundaries=True)
    gamma_val = 1.3
    geom = mesh.geometry()
    pmesh = partition_mesh(mesh, _slab_labels(mesh.n_cells, n_partitions))

    phi = jnp.asarray(np.random.default_rng(0).standard_normal(mesh.n_cells))
    serial = _serial_assembler(mesh, gamma_val).residual(phi)
    distributed = _distributed_residual(pmesh, geom, gamma_val, phi)

    assert jnp.allclose(serial, distributed, atol=1e-12)


def test_halo_plan_reproduces_the_global_gather() -> None:
    """The halo plan (ghost -> src partition + owned index) fills ghosts to exactly the values a
    direct global gather would — validating the metadata the `shard_map` all-gather will consume."""
    mesh = structured_grid_3d(5, 5, 5, named_boundaries=True)
    pmesh = partition_mesh(mesh, _slab_labels(mesh.n_cells, 3))
    phi = jnp.asarray(np.random.default_rng(2).standard_normal(mesh.n_cells))
    all_owned = _all_owned_stack(pmesh, phi)
    halo = AllGatherHaloExchange()

    for part in pmesh.partitions:
        local_full = halo.fill(
            phi[part.owned_global], all_owned, part.ghost_src_partition, part.ghost_src_owned_index
        )
        assert jnp.allclose(local_full, part.gather_cells(phi), atol=1e-14)


def test_distributed_residual_via_halo_matches_serial() -> None:
    """The full distributed residual with ghosts filled by the halo exchange equals serial — the
    only remaining Gate-1b step is wiring this per-partition function into `shard_map` with padding."""
    mesh = structured_grid_3d(6, 6, 6, named_boundaries=True)
    geom = mesh.geometry()
    pmesh = partition_mesh(mesh, _slab_labels(mesh.n_cells, 3))
    phi = jnp.asarray(np.random.default_rng(3).standard_normal(mesh.n_cells))

    serial = _serial_assembler(mesh, 1.3).residual(phi)
    distributed = _distributed_residual_via_halo(pmesh, geom, 1.3, phi)
    assert jnp.allclose(serial, distributed, atol=1e-12)


def test_distributed_parameter_gradient_matches_serial() -> None:
    """Differentiating a physical parameter (gamma) through the decomposition matches serial —
    the adjoint flows correctly across the partition gather/scatter (the Gate-1b precondition)."""
    mesh = structured_grid_3d(5, 5, 5, named_boundaries=True)
    geom = mesh.geometry()
    pmesh = partition_mesh(mesh, _slab_labels(mesh.n_cells, 3))

    rng = np.random.default_rng(1)
    phi = jnp.asarray(rng.standard_normal(mesh.n_cells))
    weight = jnp.asarray(rng.standard_normal(mesh.n_cells))

    def serial_obj(gamma_val):
        return jnp.sum(weight * _serial_assembler(mesh, gamma_val).residual(phi))

    def distributed_obj(gamma_val):
        return jnp.sum(weight * _distributed_residual(pmesh, geom, gamma_val, phi))

    g_serial = float(jax.grad(serial_obj)(1.3))
    g_distributed = float(jax.grad(distributed_obj)(1.3))
    assert np.isfinite(g_serial)
    assert abs(g_serial - g_distributed) < 1e-9
