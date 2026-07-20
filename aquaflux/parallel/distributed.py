"""Distributed (sharded) residual: run the per-partition residual under `shard_map`.

Combines the decomposition (`partition.py`) with the halo exchange (`halo.py`) to evaluate the
cell residual across devices. Each device owns one partition; the per-device program all-gathers
owned values, fills its ghost ring, and runs the **existing** diffusion physics
(`DiffusionFlux` over a per-partition `FaceContext` + `face_cells.scatter_conservative`) on its local
mesh — so the sharded residual reuses the serial operators verbatim and matches them bit-for-bit,
gradient included.

`shard_map` requires uniform per-shard shapes, so partitions are padded to common sizes with
**inert padding faces**: a padding face has zero area (⇒ zero flux) and a benign geometry chosen so
the diffusion denominator is 1 (⇒ no ``0/0`` and no NaN in the gradient). Padding cells receive no
real flux and are sliced off the owned output. This is the first-cut scaling scaffold (node
remapping, a ``ppermute`` halo, and per-partition RCM sharpen it); correctness is the priority,
established against the serial residual.

Currently wires the scalar diffusion operator (the Layer-0 proof); generalizing the per-device body
to arbitrary injected operators/BCs is the follow-on, on the same padding + `shard_map` scaffold.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh as DeviceMesh
from jax.sharding import PartitionSpec as Pspec

from aquaflux.discretization import DiffusionFlux, FaceContext
from aquaflux.mesh import CellGeometry, FaceGeometry, MeshGeometry
from aquaflux.mesh.connectivity import FaceCellConnectivity, interior_mask

from .halo import AllGatherHaloExchange, HaloExchange
from .partition import PartitionedMesh, scatter_owned_partitions


class _LocalArrays(NamedTuple):
    """The padded per-partition arrays a device needs to evaluate its local diffusion residual."""

    owner: jnp.ndarray
    neighbour: jnp.ndarray
    area: jnp.ndarray
    face_centroid: jnp.ndarray
    normal: jnp.ndarray
    cell_centroid: jnp.ndarray
    gamma: jnp.ndarray
    boundary_value: jnp.ndarray
    ghost_src_partition: jnp.ndarray
    ghost_src_owned_index: jnp.ndarray


def local_diffusion_residual(arrays: _LocalArrays, phi_full: jnp.ndarray) -> jnp.ndarray:
    """Owner-outward diffusion residual on one (padded) local mesh, reusing the serial physics.

    Orthogonal grid: the cell gradient is zero (the non-orthogonal correction vanishes), so a
    single ghost layer suffices. Reuses the serial ``DiffusionFlux`` and
    ``FaceCellConnectivity.scatter_conservative`` via a per-partition ``FaceContext``.

    Parameters
    ----------
    arrays : _LocalArrays
        Padded per-partition connectivity, geometry, coefficient and boundary values.
    phi_full : jnp.ndarray
        Local field on owned + ghost (+ padding) cells, shape ``(n_local,)``.

    Returns
    -------
    jnp.ndarray
        Residual on every local cell, shape ``(n_local,)`` (owned rows are the meaningful ones).
    """
    n_cells = phi_full.shape[0]
    dim = arrays.cell_centroid.shape[1]
    face_cells = FaceCellConnectivity(arrays.owner, arrays.neighbour, n_cells)
    geometry = MeshGeometry(
        face=FaceGeometry(arrays.area, arrays.face_centroid, arrays.normal),
        # cell volume is unused by the diffusion flux; supply a placeholder.
        cell=CellGeometry(jnp.ones(n_cells, dtype=phi_full.dtype), arrays.cell_centroid),
    )
    context = FaceContext(
        face_cells=face_cells,
        geometry=geometry,
        boundary_values=arrays.boundary_value,
        # orthogonal grid: the cell gradient is zero (the non-orthogonal correction vanishes).
        gradient=jnp.zeros((n_cells, dim), dtype=phi_full.dtype),
        properties={"diffusivity": arrays.gamma},
    )
    flux = DiffusionFlux().face_flux(phi_full, context)
    return face_cells.scatter_conservative(flux)


class PaddedDiffusion(eqx.Module):
    """Padded, stacked per-partition arrays for a sharded scalar-diffusion residual.

    Built by :func:`build_padded_diffusion`. Every array has a leading partition axis and is padded
    to common sizes so ``shard_map`` can map over the partition axis. The static sizes and the
    (padded) owned→global index map let a global field be scattered/gathered around the sharded core.

    Attributes
    ----------
    arrays : _LocalArrays
        Stacked per-partition arrays, each shape ``(n_partitions, pad, ...)``.
    owned_global : jnp.ndarray
        Padded owned→global cell map, shape ``(n_partitions, n_owned_max)``.
    n_owned : tuple of int
        Real owned-cell count per partition (static) — the valid prefix of each owned row.
    n_owned_max, n_ghost_max, n_local_max, n_faces_max : int
        Padded sizes (static).
    n_global_cells : int
        Global cell count (static).
    """

    arrays: _LocalArrays
    owned_global: jnp.ndarray
    n_owned: tuple[int, ...] = eqx.field(static=True)
    n_owned_max: int = eqx.field(static=True)
    n_ghost_max: int = eqx.field(static=True)
    n_local_max: int = eqx.field(static=True)
    n_faces_max: int = eqx.field(static=True)
    n_global_cells: int = eqx.field(static=True)

    @property
    def n_partitions(self) -> int:
        """Number of partitions."""
        return len(self.n_owned)

    def owned_states_from_global(self, global_field: jnp.ndarray) -> jnp.ndarray:
        """Gather a global field into the padded owned layout, shape ``(n_partitions, n_owned_max)``.

        Padding owned entries read cell 0 (harmless: they are never referenced as ghosts and are
        sliced off the output).
        """
        return global_field[self.owned_global]

    def scatter_owned_to_global(self, owned_out: jnp.ndarray) -> jnp.ndarray:
        """Scatter padded owned residual rows back to a global vector, shape ``(n_global_cells,)``."""
        return scatter_owned_partitions(
            [self.owned_global[p, :n] for p, n in enumerate(self.n_owned)],
            [owned_out[p, :n] for p, n in enumerate(self.n_owned)],
            self.n_global_cells,
        )


def _unit_axis0(dim: int) -> np.ndarray:
    """Unit vector along the first axis, ``(dim,)`` — the benign normal for inert padding faces."""
    e = np.zeros(dim)
    e[0] = 1.0
    return e


def build_padded_diffusion(
    pmesh: PartitionedMesh,
    global_geometry,
    gamma_global: jnp.ndarray,
    boundary_value_global: jnp.ndarray,
) -> PaddedDiffusion:
    """Pad and stack a partitioned mesh into arrays for a sharded scalar-diffusion residual.

    Parameters
    ----------
    pmesh : PartitionedMesh
        The decomposition (from :func:`~aquaflux.parallel.partition.partition_mesh`).
    global_geometry : MeshGeometry
        Global geometry (``mesh.geometry()``); gathered per partition, never recomputed.
    gamma_global : jnp.ndarray
        Per-global-cell diffusion coefficient, shape ``(n_global_cells,)``.
    boundary_value_global : jnp.ndarray
        Per-global-face weak boundary value (Dirichlet constant on boundary faces, 0 on interior),
        shape ``(n_global_faces,)``.

    Returns
    -------
    PaddedDiffusion
    """
    global_face_geometry, global_cell_geometry = global_geometry.face, global_geometry.cell
    parts = pmesh.partitions
    dim = int(np.asarray(global_cell_geometry.centroid).shape[1])
    e0 = _unit_axis0(dim)

    n_owned = tuple(p.n_owned for p in parts)
    n_ghost = tuple(p.n_ghost for p in parts)
    n_faces = tuple(int(p.mesh.n_faces) for p in parts)
    NO = max(n_owned)
    NG = max(n_ghost)
    NL = NO + NG
    NF = max(n_faces)

    face_area = np.asarray(global_face_geometry.area)
    face_centroid = np.asarray(global_face_geometry.centroid)
    face_normal = np.asarray(global_face_geometry.normal)
    cell_centroid = np.asarray(global_cell_geometry.centroid)
    gamma = np.asarray(gamma_global)
    bval = np.asarray(boundary_value_global)

    owner_s, nb_s, area_s, fcent_s, normal_s = [], [], [], [], []
    ccent_s, gamma_s, bval_s, gsp_s, gsi_s, ownedg_s = [], [], [], [], [], []

    for p, part in enumerate(parts):
        no, ng, nf = n_owned[p], n_ghost[p], n_faces[p]
        owned_global = np.asarray(part.owned_global)
        ghost_global = np.asarray(part.local_global)[no:]  # global ids of this partition's ghosts
        owner0_global = int(owned_global[0])

        # Remap original local cell indices (ghosts at [no, no+ng)) to the padded layout
        # (owned block [0, NO), ghost block [NO, NO+ng)).
        def remap(idx, no=no):
            return np.where(idx < no, idx, NO + (idx - no))

        owner_o = np.asarray(part.mesh.face_cells.owner)
        nb_o = np.asarray(part.mesh.face_cells.neighbour)
        owner_local = remap(owner_o)
        nb_interior = interior_mask(nb_o)
        nb_local = np.where(nb_interior, remap(np.where(nb_interior, nb_o, 0)), -1)

        # Padded cell centroid / gamma: real owned, real ghost, then benign padding (copy cell 0).
        ccent = np.tile(cell_centroid[owner0_global], (NL, 1))
        ccent[:no] = cell_centroid[owned_global]
        ccent[NO : NO + ng] = cell_centroid[ghost_global]
        gam = np.full(NL, gamma[owner0_global])
        gam[:no] = gamma[owned_global]
        gam[NO : NO + ng] = gamma[ghost_global]

        # Padded faces: real local faces, then inert padding faces (area 0, dpn = 1, boundary).
        fg_faces = np.asarray(part.faces_global)
        owner_pad = np.zeros(NF, dtype=np.int64)
        nb_pad = np.full(NF, -1, dtype=np.int64)
        area_pad = np.zeros(NF)
        fcent_pad = np.tile(ccent[0] + e0, (NF, 1))  # dpn = (fcent - x_owner0).e0 = 1
        normal_pad = np.tile(e0, (NF, 1))
        bval_pad = np.zeros(NF)
        owner_pad[:nf] = owner_local
        nb_pad[:nf] = nb_local
        area_pad[:nf] = face_area[fg_faces]
        fcent_pad[:nf] = face_centroid[fg_faces]
        normal_pad[:nf] = face_normal[fg_faces]
        bval_pad[:nf] = bval[fg_faces]

        # Padded halo plan and owned→global map; padding entries read cell (0, 0) harmlessly.
        gsp = np.zeros(NG, dtype=np.int64)
        gsi = np.zeros(NG, dtype=np.int64)
        gsp[:ng] = np.asarray(part.ghost_src_partition)
        gsi[:ng] = np.asarray(part.ghost_src_owned_index)
        ownedg = np.zeros(NO, dtype=np.int64)
        ownedg[:no] = owned_global

        owner_s.append(owner_pad)
        nb_s.append(nb_pad)
        area_s.append(area_pad)
        fcent_s.append(fcent_pad)
        normal_s.append(normal_pad)
        ccent_s.append(ccent)
        gamma_s.append(gam)
        bval_s.append(bval_pad)
        gsp_s.append(gsp)
        gsi_s.append(gsi)
        ownedg_s.append(ownedg)

    arrays = _LocalArrays(
        owner=jnp.asarray(np.stack(owner_s)),
        neighbour=jnp.asarray(np.stack(nb_s)),
        area=jnp.asarray(np.stack(area_s)),
        face_centroid=jnp.asarray(np.stack(fcent_s)),
        normal=jnp.asarray(np.stack(normal_s)),
        cell_centroid=jnp.asarray(np.stack(ccent_s)),
        gamma=jnp.asarray(np.stack(gamma_s)),
        boundary_value=jnp.asarray(np.stack(bval_s)),
        ghost_src_partition=jnp.asarray(np.stack(gsp_s)),
        ghost_src_owned_index=jnp.asarray(np.stack(gsi_s)),
    )
    return PaddedDiffusion(
        arrays=arrays,
        owned_global=jnp.asarray(np.stack(ownedg_s)),
        n_owned=n_owned,
        n_owned_max=NO,
        n_ghost_max=NG,
        n_local_max=NL,
        n_faces_max=NF,
        n_global_cells=pmesh.n_global_cells,
    )


def distributed_diffusion_residual(
    padded: PaddedDiffusion,
    global_phi: jnp.ndarray,
    *,
    halo: HaloExchange | None = None,
) -> jnp.ndarray:
    """Sharded scalar-diffusion residual over the partitions, reassembled to a global vector.

    Maps the per-partition local residual over the device axis with ``shard_map``; the halo is an
    ``all_gather`` over that axis, whose adjoint JAX derives automatically — so the whole function
    is reverse-mode differentiable in ``global_phi``. Requires at least ``n_partitions`` devices
    (use simulated CPU devices for testing).

    Parameters
    ----------
    padded : PaddedDiffusion
        The padded, stacked decomposition.
    global_phi : jnp.ndarray
        Global cell field, shape ``(n_global_cells,)``.
    halo : HaloExchange, optional
        Halo-fill strategy (default :class:`AllGatherHaloExchange`).

    Returns
    -------
    jnp.ndarray
        Global residual, shape ``(n_global_cells,)``.
    """
    halo = halo or AllGatherHaloExchange()
    n_part = padded.n_partitions
    NO = padded.n_owned_max
    devices = np.array(jax.devices()[:n_part])
    device_mesh = DeviceMesh(devices, axis_names=("p",))

    owned_states = padded.owned_states_from_global(global_phi)  # (n_part, NO)

    def per_device(owned_shard, arrays_shard):
        all_owned = jax.lax.all_gather(owned_shard, "p", axis=0, tiled=True)  # (n_part, NO)
        arrays = jax.tree.map(lambda a: a[0], arrays_shard)  # drop the size-1 partition axis
        owned = owned_shard[0]
        phi_full = halo.fill(
            owned, all_owned, arrays.ghost_src_partition, arrays.ghost_src_owned_index
        )
        res = local_diffusion_residual(arrays, phi_full)
        return res[:NO].reshape(1, NO)

    sharded = jax.shard_map(
        per_device,
        mesh=device_mesh,
        in_specs=(Pspec("p"), Pspec("p")),
        out_specs=Pspec("p"),
    )
    owned_out = sharded(owned_states, padded.arrays)  # (n_part, NO)
    return padded.scatter_owned_to_global(owned_out)
