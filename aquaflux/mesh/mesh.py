"""The static mesh container and its geometry-assembly method.

``Mesh`` is an ``equinox.Module`` holding topology and node coordinates as struct-of-
arrays. Face node-lists are stored **ragged**
in compressed-sparse-row (CSR) form — a row-pointer array of offsets plus a flat index array,
the standard layout for variable-length rows — so faces are arbitrary polygons with no padding.
``Mesh.geometry()`` derives face and cell geometry by wiring the face-geometry strategy and
:class:`~aquaflux.mesh.cell.CellGeometry` in dependency order, returning a bundled
:class:`~aquaflux.mesh.geometry.MeshGeometry`.

``geometry()`` is a **build-time, eager** call (run once per mesh, not inside the jitted
solve): the 3D face triangulation enumerates a data-dependent number of triangles, so it is
not itself jittable. Its outputs (static geometry) are what flow into the differentiable
residual. Geometry is a *derived product*, not stored on the mesh — it is a pure function of
the (differentiable) node coordinates, and recomputing it keeps node-position gradients correct
and avoids a stale cache under cell renumbering / partitioning (see
:class:`~aquaflux.mesh.geometry.MeshGeometry`).

The bulky per-face displacement vectors used by the diffusion flux (``x_ip - x_owner``,
``x_ip - x_neighbour``) are deliberately *not* stored — they are cheap gathers from the cell
centroids, formed where needed.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from . import cell, face, groups
from .connectivity import FaceCellConnectivity, FaceNodeConnectivity, interior_mask
from .geometry import MeshGeometry


class Mesh(eqx.Module):
    """Static unstructured mesh: node coordinates plus the face→cell and face→node connectivity.

    The raw struct-of-arrays index plumbing — owner/neighbour and the ragged CSR node lists — is
    encapsulated inside two connectivity objects, so callers compose gather/scatter/traversal
    operators rather than hand-indexing arrays. The public surface is ``node_coords`` +
    :attr:`face_cells` + :attr:`face_nodes` (plus the size properties). The raw arrays remain
    reachable through those objects (e.g. ``mesh.face_cells.owner``) for the build-time transforms
    that genuinely need the values.

    Attributes
    ----------
    node_coords : jnp.ndarray
        Node coordinates, shape ``(n_nodes, dim)``.
    face_cells : FaceCellConnectivity
        The face→cell incidence (owner / neighbour with the boundary convention) and its
        gather/scatter operators — the substrate every residual term composes.
    face_nodes : FaceNodeConnectivity
        The face→node incidence (ragged CSR) and its perimeter-traversal operators — what the
        face-geometry schemes consume.
    cell_zones : CellZones
        Partition of cells into named zones (default: a single ``"default"`` zone).
    face_patches : FacePatches
        Partition of faces into named patches (default: ``"interior"`` + ``"boundary"``).

    Notes
    -----
    Because this is an ``equinox.Module`` pytree, ``node_coords`` is a differentiable leaf, so
    gradients with respect to node positions (mesh-sensitivity diagnostics) flow through it.
    """

    node_coords: jnp.ndarray
    face_cells: FaceCellConnectivity
    face_nodes: FaceNodeConnectivity
    cell_zones: groups.CellZones
    face_patches: groups.FacePatches

    @classmethod
    def from_faces(
        cls,
        node_coords,
        faces,
        owner,
        neighbour,
        n_cells: int,
        *,
        cell_zones: dict[str, object] | None = None,
        face_patches: dict[str, object] | None = None,
    ) -> Mesh:
        """Build a mesh from a ragged list of per-face node-index lists.

        Parameters
        ----------
        node_coords : array-like
            Node coordinates, shape ``(n_nodes, dim)``.
        faces : sequence of sequence of int
            One node-index list per face — the natural output of a mesh reader. In 3D a face
            is an arbitrary polygon (any node count ``>= 3``); in 2D a face is an edge with
            **exactly two** nodes. **Each face's nodes must be listed in order around the face
            perimeter** (consecutive nodes share an edge); the winding *direction* is free (the
            stored normal is oriented owner-outward at build time), but a non-perimeter order —
            e.g. listing a quad across its diagonal — produces silently wrong geometry and is
            *not* detected (it is not a topological property). See
            :class:`~aquaflux.mesh.PolygonFaceGeometry`.
        owner, neighbour : array-like of int
            Owner / neighbour cell index per face. A boundary face is marked by neighbour
            ``== -1`` (the only accepted boundary sentinel).
        n_cells : int
            Number of cells. Every cell in ``[0, n_cells)`` must be referenced by at least one
            face (as an owner or an interior neighbour); an unreferenced cell is rejected.
        cell_zones : dict of {str: indices}, optional
            Named cell zones; cells not listed fall in the ``"default"`` zone. Omit for a
            single-zone mesh.
        face_patches : dict of {str: indices}, optional
            Named face patches overlaying the default ``"interior"`` / ``"boundary"`` split
            (boundary patches for BCs; interior patches for baffles / named interfaces).
        """
        offsets = [0]
        flat: list[int] = []
        for face_nodes in faces:
            flat.extend(int(i) for i in face_nodes)
            offsets.append(len(flat))
        return cls.from_csr(
            node_coords,
            offsets,
            flat,
            owner,
            neighbour,
            n_cells,
            cell_zones=cell_zones,
            face_patches=face_patches,
        )

    @classmethod
    def from_csr(
        cls,
        node_coords,
        face_node_offsets,
        face_node_indices,
        owner,
        neighbour,
        n_cells: int,
        *,
        cell_zones: dict[str, object] | None = None,
        face_patches: dict[str, object] | None = None,
    ) -> Mesh:
        """Build a mesh from already-assembled CSR face connectivity.

        The vectorized counterpart of :meth:`from_faces`: it takes the CSR arrays directly rather
        than a ragged Python list, so a generator that already knows the connectivity as arrays
        (e.g. a structured all-hex grid) avoids the per-face Python loop, which is the bottleneck
        for large meshes. :meth:`from_faces` delegates here after flattening its ragged input, so
        the zone/patch construction and validation live in one place.

        Parameters
        ----------
        node_coords : array-like
            Node coordinates, shape ``(n_nodes, dim)``.
        face_node_offsets : array-like of int
            CSR row pointers, shape ``(n_faces + 1,)``, starting at 0 and ending at
            ``len(face_node_indices)``.
        face_node_indices : array-like of int
            Flat concatenation of every face's node indices, in per-face perimeter order (see
            the node-ordering contract on :meth:`from_faces`).
        owner, neighbour : array-like of int
            Owner / neighbour cell index per face (neighbour ``== -1`` = boundary).
        n_cells : int
            Number of cells (every cell must be referenced by at least one face).
        cell_zones, face_patches : dict of {str: indices}, optional
            As in :meth:`from_faces`.
        """
        neighbour = jnp.asarray(neighbour)
        zones = (
            groups.CellZones.from_dict(n_cells, cell_zones)
            if cell_zones
            else groups.CellZones.default(n_cells)
        )
        patches = groups.FacePatches.from_dict(neighbour, face_patches or {})
        mesh = cls(
            node_coords=jnp.asarray(node_coords),
            face_cells=FaceCellConnectivity(jnp.asarray(owner), neighbour, n_cells),
            face_nodes=FaceNodeConnectivity.from_csr(face_node_offsets, face_node_indices),
            cell_zones=zones,
            face_patches=patches,
        )
        return mesh.validate()

    def validate(self) -> Mesh:
        """Check topological consistency; raise ``ValueError`` on the first problem found.

        Cheap (``O(n)``, no geometry) — catches the silent failures that otherwise surface as
        wrong volumes or NaNs rather than errors: non-finite coordinates, non-integer or
        out-of-range indices (``segment_sum`` silently *drops* an out-of-range owner), wrong
        per-face node counts (a 3D face with ``< 3`` nodes, or a 2D face that is not an edge), a
        face that repeats a node (a degenerate, zero-area face → NaN normal), shape mismatches, and
        any cell left unreferenced by every face (its geometry would be ``0/0``). Called by
        :meth:`from_faces`; call it directly after building a ``Mesh`` another way (e.g. a reader).
        Returns ``self`` for chaining. (The CSR *structure* — well-formed row pointers — is validated
        earlier, at :meth:`FaceNodeConnectivity.from_csr`, which owns that invariant.)

        Two defects are **out of scope** here because they are *geometric*, not topological, and
        are surfaced by the geometry-level diagnostics in :mod:`aquaflux.mesh.quality` instead: a
        face whose (distinct) nodes are collinear/coplanar and so has zero area — flagged by
        :func:`~aquaflux.mesh.quality.face_planarity` (returns ``0``) — and duplicate faces or
        unclosed cells — flagged by :func:`~aquaflux.mesh.quality.closed_cell_residual`.
        """
        nc = np.asarray(self.node_coords)
        off = np.asarray(self.face_nodes.offsets)
        idx = np.asarray(self.face_nodes.face_node_indices)
        own = np.asarray(self.face_cells.owner)
        nb = np.asarray(self.face_cells.neighbour)
        n_cells = self.face_cells.n_cells

        if nc.ndim != 2 or nc.shape[1] not in (2, 3):
            raise ValueError(f"node_coords must be (n_nodes, 2 or 3); got shape {nc.shape}")
        if not np.isfinite(nc).all():
            raise ValueError("node_coords contains a non-finite value (NaN or inf)")
        n_nodes, dim = nc.shape

        for name, arr in (("face_node_indices", idx), ("owner", own), ("neighbour", nb)):
            if not np.issubdtype(arr.dtype, np.integer):
                raise ValueError(f"{name} must have an integer dtype; got {arr.dtype}")

        n_faces = self.face_nodes.n_faces  # CSR structure already validated at from_csr

        if idx.size and (idx.min() < 0 or idx.max() >= n_nodes):
            raise ValueError(f"face_node_indices out of range [0, {n_nodes})")

        # Per-face node count: a 2D face is an edge (exactly 2 nodes); a 3D face is a polygon
        # (>= 3 nodes). The 2D edge scheme reshapes to (n_faces, 2), so exactly-2 is required.
        counts = np.diff(off)
        if dim == 2 and np.any(counts != 2):
            bad = int(np.argmax(counts != 2))
            raise ValueError(
                f"face {bad} has {int(counts[bad])} node(s); a 2D face must be an edge (exactly 2)"
            )
        if dim == 3 and np.any(counts < 3):
            bad = int(np.argmin(counts))
            raise ValueError(f"face {bad} has {int(counts[bad])} node(s); a 3D face needs >= 3")

        # A face that lists the same node twice is degenerate (zero area → NaN normal). Detect a
        # repeated (face, node) incidence: unique pairs fewer than incidences means some face
        # repeats a node. O(n log n), build-time only.
        inc_face = np.repeat(np.arange(n_faces), counts)
        if np.unique(np.stack([inc_face, idx], axis=1), axis=0).shape[0] != idx.shape[0]:
            raise ValueError(
                "a face lists the same node more than once (degenerate, zero-area face)"
            )

        if own.shape != (n_faces,) or nb.shape != (n_faces,):
            raise ValueError(
                f"owner and neighbour must both have shape ({n_faces},); "
                f"got {own.shape} and {nb.shape}"
            )
        if n_cells <= 0:
            raise ValueError(f"n_cells must be positive; got {n_cells}")
        if own.size and (own.min() < 0 or own.max() >= n_cells):
            raise ValueError(f"owner index out of range [0, {n_cells})")
        if nb.size and (nb.min() < -1 or nb.max() >= n_cells):
            raise ValueError(f"neighbour index out of range [-1, {n_cells}) (only -1 = boundary)")
        interior = interior_mask(nb)
        if np.any(own[interior] == nb[interior]):
            raise ValueError("an interior face has the same cell as both owner and neighbour")

        # Every cell must be touched by at least one face; an unreferenced cell gets zero faces,
        # so its volume/centroid would be 0/0 = NaN. (Indices are in range by the checks above.)
        referenced = np.zeros(n_cells, dtype=bool)
        referenced[own] = True
        referenced[nb[interior]] = True
        if not referenced.all():
            raise ValueError(f"cell {int(np.argmin(referenced))} is not referenced by any face")

        # Zone / patch partitions must cover the right element count with in-range labels.
        zl = np.asarray(self.cell_zones.label)
        if zl.shape != (n_cells,):
            raise ValueError(f"cell_zones.label must have shape ({n_cells},); got {zl.shape}")
        if zl.size and (zl.min() < 0 or zl.max() >= self.cell_zones.n_groups):
            raise ValueError("cell_zones.label references an undefined zone id")
        pl = np.asarray(self.face_patches.label)
        if pl.shape != (n_faces,):
            raise ValueError(f"face_patches.label must have shape ({n_faces},); got {pl.shape}")
        if pl.size and (pl.min() < 0 or pl.max() >= self.face_patches.n_groups):
            raise ValueError("face_patches.label references an undefined patch id")
        return self

    @property
    def dim(self) -> int:
        """Spatial dimension."""
        return self.node_coords.shape[1]

    @property
    def n_faces(self) -> int:
        """Number of faces."""
        return self.face_nodes.n_faces

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return self.node_coords.shape[0]

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return self.face_cells.n_cells

    def geometry(self) -> MeshGeometry:
        """Compute the derived face and cell geometry (build-time, eager).

        Wires the computation in dependency order: the dimension's face-geometry strategy
        gives area/centroid and a node-order normal; the approximate cell centroids orient the
        normals owner-outward; then the exact cell volume/centroid follow.

        Geometry is returned as a fresh :class:`~aquaflux.mesh.geometry.MeshGeometry` product
        rather than stored on the mesh: it is a pure function of the (differentiable) node
        coordinates and the topology, and recomputing it keeps gradients with respect to node
        positions correct and avoids a cache going stale under cell renumbering / partitioning
        (see :class:`~aquaflux.mesh.geometry.MeshGeometry`).

        Returns
        -------
        MeshGeometry
            The bundled face metrics (areas, centroids, owner-outward normals) and cell metrics
            (volumes, centroids).
        """
        scheme = face.face_geometry_scheme(self.dim)
        face_nodes = self.face_nodes
        face_cells = self.face_cells
        area, centroid, node_order_normal = scheme.unoriented_geometry(self.node_coords, face_nodes)
        approx = cell.CellGeometry.approx_centroids(centroid, face_cells)
        normal = scheme.orient_owner_outward(node_order_normal, centroid, approx[face_cells.owner])

        face_geometry = face.FaceGeometry(area=area, centroid=centroid, normal=normal)
        cell_geometry = cell.CellGeometry.from_faces(face_geometry, face_cells, self.dim)
        return MeshGeometry(face=face_geometry, cell=cell_geometry)
