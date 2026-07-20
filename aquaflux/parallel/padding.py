"""Uniform per-partition shapes: pad a decomposition so ``shard_map`` can map over it.

``shard_map`` maps one program over a leading device axis and requires **identical local shapes
on every shard**, but partitions differ in cell, ghost, and face counts. This module is the one
home for reconciling those two facts, and it is deliberately **operator-independent**: it knows
about cells, faces, and halos, and nothing about diffusion, advection, or flow. Whatever residual
runs per device, it runs on the layout built here.

Padding works by extending every partition to common sizes with **inert** cells and faces, plus
two reserved slots that make the padding safe for *any* operator:

- **The null cell** (local index ``n_owned_max + n_ghost_max``) is past every partition's real
  cells. Every padding face names it as owner, so a padding face can only ever scatter into the
  null cell — never into a real owned row. Its residual row is discarded with the rest of the
  padding.
- **The null face** (local index ``n_faces_max``) is past every partition's real faces. It is the
  fill value for padded boundary-patch index lists, which must also be uniform in length.

Padding is inert through three independent mechanisms, so no single one has to carry the
correctness argument alone:

1. **Zero area.** A padding face has zero area, so any flux through it is zero.
2. **Null-cell ownership.** Even a non-zero flux would scatter only into the null cell.
3. **Benign geometry.** The padding normal is the first unit axis and the padding face centroid
   sits one unit along it from the null-cell centroid, so the owner-to-face normal distance is
   exactly ``1``. Nothing divides by zero, so no ``NaN`` reaches the value *or* the gradient —
   a ``NaN`` in a discarded row would still poison the cotangent of the rows that share the
   reduction.

Padding cells carry unit volume and a real cell centroid (not zeros), so a transient term or a
volume source integrating over a padding cell stays finite rather than dividing by zero.

The padded local mesh is deliberately **not** passed through ``Mesh.validate()``: padding faces
list no nodes and padding cells are touched by no face, both of which that check rejects — and
rightly so, for a mesh describing real geometry. A padded mesh is bookkeeping for the device
axis, and its geometry is *gathered* from the global mesh rather than computed from nodes, so
the node-level invariants do not apply to it.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from aquaflux.mesh import (
    CellGeometry,
    CellZones,
    FaceCellConnectivity,
    FaceGeometry,
    FaceNodeConnectivity,
    FacePatches,
    Mesh,
    MeshGeometry,
    interior_mask,
)

from .partition import PartitionedMesh, scatter_owned_partitions


class PaddedLayout(eqx.Module):
    """Uniform per-partition sizes, the owned→global map, and the halo plan.

    The operator-independent half of the padded decomposition: it fixes how many cells and faces
    every shard has, where each partition's real data stops and its padding begins, and which
    remote owned value each ghost mirrors. It carries no physics, so the same layout serves any
    residual.

    Attributes
    ----------
    n_owned, n_ghost, n_faces : tuple of int
        Real per-partition counts (static); the valid prefix of each padded row.
    n_owned_max, n_ghost_max, n_faces_max : int
        The padded sizes (static) — the maxima of the corresponding counts.
    n_incidences_max : int
        Padded length of the face-node index array (static) — the largest face-node incidence
        count over the partitions.
    n_nodes : tuple of int
        Real per-partition node counts (static); the valid prefix of each padded node array.
    n_nodes_max : int
        Padded node count (static) — the largest per-partition node count.
    n_global_cells : int
        Global cell count (static).
    owned_global : jnp.ndarray
        Padded owned→global cell map, shape ``(n_partitions, n_owned_max)``. Padding entries read
        global cell 0; the rows they produce are discarded.
    ghost_src_partition, ghost_src_owned_index : jnp.ndarray
        The halo plan, shape ``(n_partitions, n_ghost_max)`` each: the partition owning each ghost
        cell, and that ghost's owned-index within it. Padding entries read partition 0, index 0.
    """

    n_owned: tuple[int, ...] = eqx.field(static=True)
    n_ghost: tuple[int, ...] = eqx.field(static=True)
    n_faces: tuple[int, ...] = eqx.field(static=True)
    n_owned_max: int = eqx.field(static=True)
    n_ghost_max: int = eqx.field(static=True)
    n_faces_max: int = eqx.field(static=True)
    n_incidences_max: int = eqx.field(static=True)
    n_nodes: tuple[int, ...] = eqx.field(static=True)
    n_nodes_max: int = eqx.field(static=True)
    n_global_cells: int = eqx.field(static=True)
    owned_global: jnp.ndarray
    ghost_src_partition: jnp.ndarray
    ghost_src_owned_index: jnp.ndarray

    @property
    def n_partitions(self) -> int:
        """Number of partitions."""
        return len(self.n_owned)

    @property
    def null_cell(self) -> int:
        """Local index of the reserved null cell — past every partition's real cells."""
        return self.n_owned_max + self.n_ghost_max

    @property
    def null_face(self) -> int:
        """Local index of the reserved null face — past every partition's real faces."""
        return self.n_faces_max

    @property
    def n_local(self) -> int:
        """Padded local cell count per shard: owned block, ghost block, then the null cell."""
        return self.null_cell + 1

    @property
    def n_faces_local(self) -> int:
        """Padded local face count per shard: the real faces, padding faces, then the null face."""
        return self.null_face + 1

    def remap_cells(self, indices: np.ndarray, partition: int) -> np.ndarray:
        """Map a partition's original local cell indices into the padded layout.

        A local mesh numbers owned cells ``[0, n_owned)`` then ghosts ``[n_owned, n_local)``. The
        padded layout fixes the owned block at ``[0, n_owned_max)`` and the ghost block at
        ``[n_owned_max, ...)``, so ghost indices shift by the owned padding.

        Parameters
        ----------
        indices : numpy.ndarray of int
            Original local cell indices for partition ``partition``.
        partition : int
            Which partition ``indices`` belongs to.

        Returns
        -------
        numpy.ndarray of int
            The same cells in padded-layout indices.
        """
        no = self.n_owned[partition]
        return np.where(indices < no, indices, self.n_owned_max + (indices - no))

    def owned_states_from_global(self, global_field: jnp.ndarray) -> jnp.ndarray:
        """Gather a global per-cell field into the padded owned layout.

        Parameters
        ----------
        global_field : jnp.ndarray
            Field indexed by global cell, shape ``(n_global_cells, ...)``.

        Returns
        -------
        jnp.ndarray
            Shape ``(n_partitions, n_owned_max, ...)``. Padding entries read global cell 0, which
            is harmless: they are never referenced as a ghost source and are sliced off the output.
        """
        return global_field[self.owned_global]

    def scatter_owned_to_global(self, owned_out: jnp.ndarray) -> jnp.ndarray:
        """Scatter padded owned rows back to a global per-cell vector.

        Parameters
        ----------
        owned_out : jnp.ndarray
            Padded owned rows, shape ``(n_partitions, n_owned_max, ...)``; only the first
            ``n_owned[p]`` rows of each partition are real.

        Returns
        -------
        jnp.ndarray
            The global vector, shape ``(n_global_cells, ...)``.
        """
        return scatter_owned_partitions(
            [self.owned_global[p, :n] for p, n in enumerate(self.n_owned)],
            [owned_out[p, :n] for p, n in enumerate(self.n_owned)],
            self.n_global_cells,
        )

    @classmethod
    def from_partitioned(cls, pmesh: PartitionedMesh) -> PaddedLayout:
        """Derive the uniform layout and padded halo plan from a decomposition.

        Parameters
        ----------
        pmesh : PartitionedMesh
            The decomposition to pad.

        Returns
        -------
        PaddedLayout
        """
        parts = pmesh.partitions
        n_owned = tuple(p.n_owned for p in parts)
        n_ghost = tuple(p.n_ghost for p in parts)
        n_faces = tuple(int(p.mesh.n_faces) for p in parts)
        n_nodes = tuple(int(p.mesh.n_nodes) for p in parts)
        no_max, ng_max = max(n_owned), max(n_ghost)

        owned_global = np.zeros((len(parts), no_max), dtype=np.int64)
        gsp = np.zeros((len(parts), ng_max), dtype=np.int64)
        gsi = np.zeros((len(parts), ng_max), dtype=np.int64)
        for p, part in enumerate(parts):
            owned_global[p, : n_owned[p]] = np.asarray(part.owned_global)
            gsp[p, : n_ghost[p]] = np.asarray(part.ghost_src_partition)
            gsi[p, : n_ghost[p]] = np.asarray(part.ghost_src_owned_index)

        return cls(
            n_owned=n_owned,
            n_ghost=n_ghost,
            n_faces=n_faces,
            n_owned_max=no_max,
            n_ghost_max=ng_max,
            n_faces_max=max(n_faces),
            n_incidences_max=max(int(p.mesh.face_nodes.face_node_indices.shape[0]) for p in parts),
            n_nodes=n_nodes,
            n_nodes_max=max(n_nodes),
            n_global_cells=pmesh.n_global_cells,
            owned_global=jnp.asarray(owned_global),
            ghost_src_partition=jnp.asarray(gsp),
            ghost_src_owned_index=jnp.asarray(gsi),
        )


def _first_unit_axis(dim: int) -> np.ndarray:
    """Unit vector along the first axis, shape ``(dim,)`` — the padding face normal."""
    e = np.zeros(dim)
    e[0] = 1.0
    return e


def pad_partition(
    layout: PaddedLayout,
    partition: int,
    local_mesh: Mesh,
    local_geometry: MeshGeometry,
) -> tuple[Mesh, MeshGeometry]:
    """Extend one partition's local mesh and geometry to the layout's uniform sizes.

    The returned mesh is an ordinary :class:`~aquaflux.mesh.Mesh` — every operator, scheme, and
    boundary closure runs on it unchanged — whose trailing cells and faces are the inert padding
    described in the module docstring. The geometry is the partition's *gathered* global geometry
    extended the same way, so the real rows stay bit-for-bit identical to the serial values.

    Parameters
    ----------
    layout : PaddedLayout
        The uniform sizes to pad to.
    partition : int
        Which partition ``local_mesh`` is.
    local_mesh : Mesh
        The partition's local mesh (owned cells, then the halo ring).
    local_geometry : MeshGeometry
        That mesh's geometry, gathered from the global geometry.

    Returns
    -------
    (Mesh, MeshGeometry)
        The padded local mesh and its padded geometry.
    """
    nf = layout.n_faces[partition]
    n_local, n_faces_local = layout.n_local, layout.n_faces_local
    null_cell = layout.null_cell
    dim = local_mesh.dim
    e0 = _first_unit_axis(dim)

    # --- connectivity -------------------------------------------------------------------
    # Real faces keep their cells (shifted into the padded blocks); padding faces are boundary
    # faces owned by the null cell, so they can only ever scatter into a discarded row.
    owner_real = layout.remap_cells(np.asarray(local_mesh.face_cells.owner), partition)
    nb_real_raw = np.asarray(local_mesh.face_cells.neighbour)
    nb_interior = interior_mask(nb_real_raw)
    nb_real = np.where(
        nb_interior, layout.remap_cells(np.where(nb_interior, nb_real_raw, 0), partition), -1
    )

    owner = np.full(n_faces_local, null_cell, dtype=np.int64)
    neighbour = np.full(n_faces_local, -1, dtype=np.int64)
    owner[:nf] = owner_real
    neighbour[:nf] = nb_real

    # --- geometry -----------------------------------------------------------------------
    # The null cell's centroid anchors the padding geometry; take the first owned cell's centroid
    # so it is a real, finite point inside the domain.
    cell_centroid_real = np.asarray(local_geometry.cell.centroid)
    cell_volume_real = np.asarray(local_geometry.cell.volume)
    anchor = cell_centroid_real[0]

    cell_centroid = np.tile(anchor, (n_local, 1))
    cell_volume = np.ones(n_local)  # unit volume: a source/transient term over padding stays finite
    padded_cells = layout.remap_cells(np.arange(local_mesh.n_cells), partition)
    cell_centroid[padded_cells] = cell_centroid_real
    cell_volume[padded_cells] = cell_volume_real  # real cells keep their real volumes

    face_area = np.zeros(n_faces_local)
    face_normal = np.tile(e0, (n_faces_local, 1))
    # Owner-to-face normal distance is exactly 1 for padding faces, so nothing divides by zero.
    face_centroid = np.tile(anchor + e0, (n_faces_local, 1))
    face_area[:nf] = np.asarray(local_geometry.face.area)
    face_normal[:nf] = np.asarray(local_geometry.face.normal)
    face_centroid[:nf] = np.asarray(local_geometry.face.centroid)

    # --- groups -------------------------------------------------------------------------
    # Padding faces take the unnamed "boundary" patch, so no named boundary closure claims them.
    patch_names = local_mesh.face_patches.names
    patch_label = np.full(n_faces_local, patch_names.index("boundary"), dtype=np.int64)
    patch_label[:nf] = np.asarray(local_mesh.face_patches.label)

    zone_label_real = np.asarray(local_mesh.cell_zones.label)
    zone_label = np.full(n_local, zone_label_real[0], dtype=np.int64)
    zone_label[padded_cells] = zone_label_real

    # --- nodes --------------------------------------------------------------------------
    # Node counts are ragged across partitions (each carries only its own nodes), so the node
    # coordinates are padded to a common count with copies of node 0 — a real, finite point. The
    # padding nodes are referenced by nothing (padded faces list no nodes), so their value is inert.
    node_coords_real = np.asarray(local_mesh.node_coords)
    node_coords = np.tile(node_coords_real[0], (layout.n_nodes_max, 1))
    node_coords[: node_coords_real.shape[0]] = node_coords_real

    # --- face nodes ---------------------------------------------------------------------
    # The face-node array is ragged across partitions, so it is padded to a common length too.
    # Row pointers must still end at that length, so the first padding face absorbs the whole
    # padded tail and the remaining padding faces list no nodes. Those incidences all point at
    # node 0: nothing reads them, because a padded face's geometry is supplied directly rather
    # than derived from its nodes.
    indices_real = np.asarray(local_mesh.face_nodes.face_node_indices)
    indices = np.zeros(layout.n_incidences_max, dtype=indices_real.dtype)
    indices[: indices_real.shape[0]] = indices_real

    offsets_real = np.asarray(local_mesh.face_nodes.offsets)
    offsets = np.full(n_faces_local + 1, layout.n_incidences_max, dtype=np.int64)
    offsets[: nf + 1] = offsets_real

    padded_mesh = Mesh(
        node_coords=jnp.asarray(node_coords),
        face_cells=FaceCellConnectivity(jnp.asarray(owner), jnp.asarray(neighbour), n_local),
        face_nodes=FaceNodeConnectivity.from_csr(offsets, indices),
        cell_zones=CellZones(label=jnp.asarray(zone_label), names=local_mesh.cell_zones.names),
        face_patches=FacePatches(label=jnp.asarray(patch_label), names=patch_names),
    )
    padded_geometry = MeshGeometry(
        face=FaceGeometry(
            jnp.asarray(face_area), jnp.asarray(face_centroid), jnp.asarray(face_normal)
        ),
        cell=CellGeometry(jnp.asarray(cell_volume), jnp.asarray(cell_centroid)),
    )
    return padded_mesh, padded_geometry
