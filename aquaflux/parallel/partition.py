"""Domain decomposition: split a global :class:`~aquaflux.mesh.Mesh` into local partitions.

This is the **wrapper** that separates the parallelization concern from mesh storage: a `Mesh`
stays a pure contiguous block of cells, and a partition's *local* mesh — its owned cells plus
an appended **halo ring** of ghost
cells copied from neighbouring partitions — is itself an ordinary `Mesh`, so every operator,
scheme, and boundary closure runs on it verbatim. What lives here is only the decomposition
metadata: which local cells are ghosts, which remote owned cell each mirrors, and the index maps
to gather fields onto a partition and scatter owned results back to the global vector.

Key correctness facts for a cell-centred FVM decomposition:

- **Domain-boundary faces never cross a partition.** A boundary face has ``neighbour = -1`` and a
  single owner, so it and its boundary condition live entirely within one partition. Only
  *interior* faces become partition-boundary faces (owned cell ↔ ghost cell).
- **Geometry is gathered, not recomputed.** A ghost cell's face stencil is incomplete on the
  local partition, so recomputing its centroid/volume locally would be wrong. Instead the local
  face/cell geometry is *gathered* from the global geometry (:meth:`local_face_geometry` /
  :meth:`local_cell_geometry`), guaranteeing bit-for-bit agreement with the serial residual.
- **Global owner/neighbour roles are preserved.** Remapping keeps each face's global owner as the
  local owner, so the gathered owner-outward normals stay consistent — no re-orientation.

This module produces a correct decomposition and nothing more; extending each partition to the
uniform shapes ``shard_map`` needs is a separate concern, in
:mod:`~aquaflux.parallel.padding`. The current builder keeps the full global node array on every
partition (correct, simple); remapping to a partition-local node set is a scaling follow-on.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aquaflux.mesh import CellGeometry, FaceGeometry, Mesh, MeshGeometry
from aquaflux.mesh.connectivity import (
    FaceCellConnectivity,
    FaceNodeConnectivity,
    interior_mask,
)
from aquaflux.mesh.groups import CellZones, FacePatches


def scatter_owned_partitions(
    owned_global: list[jnp.ndarray],
    owned_values: list[jnp.ndarray],
    n_global_cells: int,
) -> jnp.ndarray:
    """Reassemble a global per-cell vector from each partition's owned rows.

    Each ``(owned_global[p], owned_values[p])`` pair writes that partition's owned-cell values into
    the global vector at its global cell ids — the single home for the owned→global placement, used
    by both :meth:`PartitionedMesh.scatter_owned` and the padded sharded residual.

    Parameters
    ----------
    owned_global : list of jnp.ndarray
        Per-partition global cell ids of the owned cells, shape ``(n_owned_p,)`` each.
    owned_values : list of jnp.ndarray
        Per-partition owned-cell values, shape ``(n_owned_p, ...)`` each (matching trailing shape).
    n_global_cells : int
        Global cell count.

    Returns
    -------
    jnp.ndarray
        The global vector, shape ``(n_global_cells, ...)``.
    """
    out = jnp.zeros((n_global_cells, *owned_values[0].shape[1:]), dtype=owned_values[0].dtype)
    for og, vals in zip(owned_global, owned_values, strict=True):
        out = out.at[og].set(vals)
    return out


class LocalPartition(eqx.Module):
    """One partition's local mesh (owned + ghost cells) plus its decomposition metadata.

    Attributes
    ----------
    mesh : Mesh
        The local mesh: cells ``[0, n_owned)`` are owned, ``[n_owned, n_local)`` are ghosts;
        connectivity uses local indices. Domain-boundary faces and their patches are preserved.
    n_owned : int
        Number of owned cells (static); the local residual is meaningful only on these rows.
    owned_global : jnp.ndarray
        Global cell id of each owned cell, shape ``(n_owned,)`` — used to scatter owned results
        back to the global vector.
    local_global : jnp.ndarray
        Global cell id of every local cell (owned then ghost), shape ``(n_local,)`` — used to
        gather a global per-cell field (and geometry) onto this partition.
    faces_global : jnp.ndarray
        Global face id of every local face, shape ``(n_faces_local,)`` — gathers per-face fields.
    ghost_src_partition : jnp.ndarray
        For each ghost cell, the partition that owns it, shape ``(n_ghost,)`` — the halo plan.
    ghost_src_owned_index : jnp.ndarray
        For each ghost cell, its owned-index within its source partition, shape ``(n_ghost,)``.
    """

    mesh: Mesh
    n_owned: int = eqx.field(static=True)
    owned_global: jnp.ndarray
    local_global: jnp.ndarray
    faces_global: jnp.ndarray
    ghost_src_partition: jnp.ndarray
    ghost_src_owned_index: jnp.ndarray

    @property
    def n_local(self) -> int:
        """Total local cell count (owned + ghost)."""
        return self.mesh.n_cells

    @property
    def n_ghost(self) -> int:
        """Number of ghost (halo) cells."""
        return self.mesh.n_cells - self.n_owned

    def gather_cells(self, global_field: jnp.ndarray) -> jnp.ndarray:
        """Gather a global per-cell field onto this partition's local cells (owned + ghost).

        Parameters
        ----------
        global_field : jnp.ndarray
            A field indexed by global cell, shape ``(n_global_cells, ...)``.

        Returns
        -------
        jnp.ndarray
            The field on local cells, shape ``(n_local, ...)``.
        """
        return global_field[self.local_global]

    def local_cell_geometry(self, global_cg: CellGeometry) -> CellGeometry:
        """Gather global cell geometry onto local cells (correct ghost centroids/volumes)."""
        return jax.tree.map(lambda a: a[self.local_global], global_cg)

    def local_face_geometry(self, global_fg: FaceGeometry) -> FaceGeometry:
        """Gather global face geometry onto local faces (owner-outward normals preserved)."""
        return jax.tree.map(lambda a: a[self.faces_global], global_fg)

    def local_geometry(self, global_geometry: MeshGeometry) -> MeshGeometry:
        """Gather the global mesh geometry onto this partition's local faces and cells.

        Composes :meth:`local_face_geometry` and :meth:`local_cell_geometry` so the local
        residual is assembled from the *gathered* global geometry (bit-for-bit agreement with the
        serial residual), never a locally recomputed one.
        """
        return MeshGeometry(
            face=self.local_face_geometry(global_geometry.face),
            cell=self.local_cell_geometry(global_geometry.cell),
        )


class PartitionedMesh(eqx.Module):
    """A global mesh decomposed into ``n_partitions`` local partitions with halo rings.

    Built by :func:`partition_mesh` from a global mesh and a per-cell partition label. Holds the
    local partitions and the machinery to scatter owned per-partition results back to a single
    global vector; gathering onto a partition is :meth:`LocalPartition.gather_cells`.

    Attributes
    ----------
    partitions : tuple of LocalPartition
        One entry per partition.
    n_global_cells : int
        Global cell count (static).
    labels : jnp.ndarray
        Partition id per global cell, shape ``(n_global_cells,)``.
    """

    partitions: tuple[LocalPartition, ...]
    n_global_cells: int = eqx.field(static=True)
    labels: jnp.ndarray

    @property
    def n_partitions(self) -> int:
        """Number of partitions."""
        return len(self.partitions)

    def scatter_owned(self, owned_results: list[jnp.ndarray]) -> jnp.ndarray:
        """Reassemble a global per-cell vector from each partition's owned-cell result.

        Parameters
        ----------
        owned_results : list of jnp.ndarray
            One array per partition, each shape ``(n_owned_p, ...)`` aligned with
            ``partitions[p].owned_global``.

        Returns
        -------
        jnp.ndarray
            The global vector, shape ``(n_global_cells, ...)``.
        """
        return scatter_owned_partitions(
            [part.owned_global for part in self.partitions], owned_results, self.n_global_cells
        )


def partition_mesh(mesh: Mesh, labels) -> PartitionedMesh:
    """Decompose ``mesh`` into local partitions with halo rings from a per-cell partition label.

    Parameters
    ----------
    mesh : Mesh
        The global mesh.
    labels : array-like of int, shape ``(n_cells,)``
        Partition id in ``[0, n_partitions)`` for each global cell.

    Returns
    -------
    PartitionedMesh

    Notes
    -----
    Pure build-time topology work in numpy (not part of the differentiable solve). The local mesh
    keeps the full global node array; node remapping is a scaling follow-on. Padding to the
    uniform shapes ``shard_map`` needs is applied separately, by
    :func:`~aquaflux.parallel.padding.pad_partition`.
    """
    labels = np.asarray(labels)
    owner = np.asarray(mesh.face_cells.owner)
    nb = np.asarray(mesh.face_cells.neighbour)
    n_global = mesh.n_cells
    n_partitions = int(labels.max()) + 1

    node_coords = mesh.node_coords
    offsets = np.asarray(mesh.face_nodes.offsets)
    indices = np.asarray(mesh.face_nodes.face_node_indices)
    face_patch_label = np.asarray(mesh.face_patches.label)
    cell_zone_label = np.asarray(mesh.cell_zones.label)

    # Owned-index of each global cell within its own partition (for the halo plan).
    owned_index = np.full(n_global, -1, dtype=np.int64)
    for p in range(n_partitions):
        owned_p = np.where(labels == p)[0]
        owned_index[owned_p] = np.arange(len(owned_p))

    # partition label of each face's neighbour; boundary faces (no neighbour) get -1
    nb_interior = interior_mask(nb)
    nb_label = np.where(nb_interior, labels[np.where(nb_interior, nb, 0)], -1)

    partitions: list[LocalPartition] = []
    for p in range(n_partitions):
        owned_p = np.where(labels == p)[0]
        n_owned = len(owned_p)

        # Faces incident to an owned cell (owner or neighbour owned).
        face_in_p = (labels[owner] == p) | (nb_label == p)
        local_faces = np.where(face_in_p)[0]

        # Cells referenced by those faces; the ghosts are the non-owned ones.
        nb_local = nb[local_faces]
        referenced = np.unique(
            np.concatenate([owner[local_faces], nb_local[interior_mask(nb_local)]])
        )
        owned_set = set(owned_p.tolist())
        ghost_global = np.array(
            sorted(c for c in referenced.tolist() if c not in owned_set), dtype=np.int64
        )
        n_ghost = len(ghost_global)

        # Global -> local cell map: owned first, then ghosts.
        g2l = np.full(n_global, -1, dtype=np.int64)
        g2l[owned_p] = np.arange(n_owned)
        g2l[ghost_global] = n_owned + np.arange(n_ghost)
        local_global = np.concatenate([owned_p, ghost_global])

        # Remap face connectivity to local indices (boundary neighbour stays -1).
        lf_owner = g2l[owner[local_faces]]
        lf_nb_global = nb[local_faces]
        lf_interior = interior_mask(lf_nb_global)
        lf_nb = np.where(lf_interior, g2l[np.where(lf_interior, lf_nb_global, 0)], -1)

        # Local ragged face-node arrays (compressed-sparse-row: offsets + flat indices)
        # from the selected global faces.
        local_face_nodes = [indices[offsets[f] : offsets[f + 1]] for f in local_faces]
        local_offsets = np.concatenate(
            [[0], np.cumsum([len(x) for x in local_face_nodes], dtype=np.int64)]
        )
        local_indices = (
            np.concatenate(local_face_nodes) if local_face_nodes else np.zeros(0, dtype=np.int64)
        )

        local_patches = FacePatches(
            label=jnp.asarray(face_patch_label[local_faces]), names=mesh.face_patches.names
        )
        local_zones = CellZones(
            label=jnp.asarray(cell_zone_label[local_global]), names=mesh.cell_zones.names
        )
        local_mesh = Mesh(
            node_coords=node_coords,
            face_cells=FaceCellConnectivity(
                jnp.asarray(lf_owner), jnp.asarray(lf_nb), n_owned + n_ghost
            ),
            face_nodes=FaceNodeConnectivity.from_csr(local_offsets, local_indices),
            cell_zones=local_zones,
            face_patches=local_patches,
        ).validate()

        partitions.append(
            LocalPartition(
                mesh=local_mesh,
                n_owned=n_owned,
                owned_global=jnp.asarray(owned_p),
                local_global=jnp.asarray(local_global),
                faces_global=jnp.asarray(local_faces),
                ghost_src_partition=jnp.asarray(labels[ghost_global]),
                ghost_src_owned_index=jnp.asarray(owned_index[ghost_global]),
            )
        )

    return PartitionedMesh(
        partitions=tuple(partitions), n_global_cells=n_global, labels=jnp.asarray(labels)
    )
