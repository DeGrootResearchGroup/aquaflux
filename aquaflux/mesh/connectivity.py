"""Mesh connectivity: the storage-and-movement layer between nodes, faces, and cells.

The struct-of-arrays substrate is a set of **incidence relations** between entity kinds, plus
the gather/scatter/reduce operators over them. This module owns that layer so that geometry
(and the residual substrate) can *compose* it and read as math rather than as index plumbing:

- :class:`FaceCellConnectivity` — the face→cell relation (owner / neighbour). Gather cell
  values onto faces; scatter face contributions back to cells with the boundary convention
  applied once. This is the operator behind every ``gather → compute → scatter`` residual term.
- :class:`FaceNodeConnectivity` — the face→node relation, stored ragged in compressed-sparse-row
  (CSR) form (a row-pointer array plus a flat index array). Gather node values onto a face's
  perimeter; reduce a per-node-incidence quantity to per-face. This is what lets the face-geometry
  schemes traverse a polygon without open-coding the CSR arithmetic.

The single convention every scatter/gather depends on lives here too: a **neighbour index
``< 0`` (by convention ``-1``) marks a boundary face**, which couples only its owner. The
module-level function :func:`interior_mask` is the boundary-convention primitive the classes are
built on, and the form the numpy build-time paths (validation, renumbering, partitioning) use
directly on raw owner/neighbour arrays where no connectivity object is in hand.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.ops import segment_max, segment_min, segment_sum

from aquaflux.vectors import scale


def interior_mask(neighbour):
    """Boolean per-face mask, ``True`` on interior faces.

    A face is interior when it has a real neighbour cell; by convention a neighbour index
    ``< 0`` (``-1``) marks a boundary face, which couples only its owner.

    Parameters
    ----------
    neighbour : array_like of int, shape ``(n_faces,)``
        Neighbour cell index per face (``< 0`` on boundary faces). Accepts a NumPy or JAX
        array; the result uses the same array library as the input (the body is a plain
        comparison, so a NumPy input stays NumPy).

    Returns
    -------
    ndarray of bool, shape ``(n_faces,)``
        ``True`` where ``neighbour >= 0``.
    """
    return neighbour >= 0


def _broadcast_face_mask(mask: jnp.ndarray, ndim: int) -> jnp.ndarray:
    """Reshape a per-face mask ``(n_faces,)`` to broadcast against a rank-``ndim`` per-face array."""
    return mask.reshape(mask.shape + (1,) * (ndim - 1))


class FaceCellConnectivity(eqx.Module):
    """The face→cell incidence (owner / neighbour) and the gather/scatter operators over it.

    Every finite-volume residual term is a ``gather → compute → scatter`` over this relation:
    gather owner/neighbour cell state onto faces, compute a face quantity, scatter it back to
    cells. This object owns the storage-layout mechanics — the boundary convention, the
    owner-outward sign, and the ``segment_sum`` scatter — so operators express only the physics.

    Attributes
    ----------
    owner : jnp.ndarray of int, shape ``(n_faces,)``
        Owner cell index per face.
    neighbour : jnp.ndarray of int, shape ``(n_faces,)``
        Neighbour cell index per face (``< 0`` marks a boundary face).
    n_cells : int
        Number of cells to scatter into (static).
    neighbour_offset : jnp.ndarray or None, shape ``(n_faces, dim)``
        Per-face translation added to the neighbour cell's centroid to give its **periodic image**
        as seen from the owner — nonzero only on the wrap faces of a periodic seam, ``+L`` along the
        periodic axis (see :meth:`neighbour_centroid`). ``None`` (the default) means a zero offset
        everywhere: an ordinary non-periodic mesh, stored without allocating the array.
    """

    owner: jnp.ndarray
    neighbour: jnp.ndarray
    n_cells: int = eqx.field(static=True)
    neighbour_offset: jnp.ndarray | None = None

    @property
    def interior(self) -> jnp.ndarray:
        """Boolean per-face mask, ``True`` on interior faces, shape ``(n_faces,)``."""
        return interior_mask(self.neighbour)

    @property
    def safe_neighbour(self) -> jnp.ndarray:
        """Neighbour index with boundary faces substituted by their owner, shape ``(n_faces,)``.

        On a boundary face the neighbour index (``< 0``) is not a valid cell, so it cannot index a
        per-cell array. Substituting the *owner* gives an always-in-range index: a gather
        ``field[safe_neighbour]`` then reads the owner's own value (a boundary face reads as
        ``owner == neighbour``), and a scatter of a boundary contribution — which callers mask to
        zero — lands harmlessly on the owner. Interior entries are unchanged.

        The substitution is only *correct* because callers mask the boundary face's contribution to
        zero (interior faces alone couple two cells): the substituted index makes the operation
        **valid** (in range), the caller's mask makes it **correct**. Substituting the owner rather
        than an arbitrary in-range index (e.g. ``0``) is deliberate — it makes the degenerate
        boundary face read as ``owner == neighbour``, the meaningful value where the neighbour is
        used unmasked (a boundary face is not a zone interface, its neighbour value equals the
        owner's, etc.). Gather several fields off it without recomputing the substitution each time
        (``field[conn.safe_neighbour]``).
        """
        return jnp.where(self.interior, self.neighbour, self.owner)

    def neighbour_centroid(self, cell_centroid: jnp.ndarray) -> jnp.ndarray:
        """Neighbour cell centroids gathered per face, shifted to their **periodic image**.

        Every displacement-forming operator (the owner→neighbour vector a diffusion, gradient, or
        Rhie--Chow term needs) must gather the neighbour centroid through *this* accessor rather than
        indexing ``cell_centroid[safe_neighbour]`` directly. On an ordinary interior or boundary face
        the two agree; across a periodic seam the raw neighbour centroid sits a full period away, so
        the owner→neighbour vector would be wrong — adding :attr:`neighbour_offset` (``+L`` on the
        wrap faces) returns the neighbour's periodic image, making the seam delta identical to an
        ordinary interior face. Only geometric *position* is shifted; field *values* are periodic and
        gather unchanged off :attr:`safe_neighbour`.

        Parameters
        ----------
        cell_centroid : jnp.ndarray, shape ``(n_cells, dim)``
            Per-cell centroids.

        Returns
        -------
        jnp.ndarray, shape ``(n_faces, dim)``
            The (periodic-image) neighbour centroid per face.
        """
        neighbour_centroid = cell_centroid[self.safe_neighbour]
        if self.neighbour_offset is None:
            return neighbour_centroid
        return neighbour_centroid + self.neighbour_offset

    def combine_face_values(
        self, interior_values: jnp.ndarray, boundary_values: jnp.ndarray
    ) -> jnp.ndarray:
        """Assemble a full per-face field from its interior and boundary parts.

        Returns ``interior_values`` on interior faces and ``boundary_values`` on boundary faces —
        the complete per-face array a scheme forms once it has computed a value for the interior
        faces (an interpolation, a reconstruction, a flux branch) and holds the boundary-face
        values separately. The per-face ``interior`` mask broadcasts over any trailing component
        axes, so scalar ``(n_faces,)`` and vector ``(n_faces, dim)`` face fields both work.

        Parameters
        ----------
        interior_values : jnp.ndarray
            The value on interior faces, shape ``(n_faces, ...)``; its boundary-face entries are
            ignored (typically a harmless placeholder left by the interior formula).
        boundary_values : jnp.ndarray
            The value on boundary faces, shape broadcastable to ``interior_values``; its
            interior-face entries are ignored.

        Returns
        -------
        jnp.ndarray
            The combined per-face field, shape ``(n_faces, ...)`` — ``interior_values`` where
            interior, ``boundary_values`` elsewhere.
        """
        mask = _broadcast_face_mask(self.interior, interior_values.ndim)
        return jnp.where(mask, interior_values, boundary_values)

    def scatter(self, owner_contrib: jnp.ndarray, neighbour_contrib: jnp.ndarray) -> jnp.ndarray:
        """Scatter per-face contributions to cells: owner gets ``owner_contrib``, its interior
        neighbour gets ``neighbour_contrib``.

        The neighbour contribution is masked to zero on boundary faces (which have no neighbour)
        — the one place that masking is written. The owner contribution is scattered for every
        face, boundary faces included.

        Parameters
        ----------
        owner_contrib, neighbour_contrib : jnp.ndarray
            Per-face contributions, shape ``(n_faces, ...)`` (same shape; any trailing rank).

        Returns
        -------
        jnp.ndarray
            Per-cell sum, shape ``(n_cells, ...)``.
        """
        mask = _broadcast_face_mask(self.interior, neighbour_contrib.ndim)
        neigh = jnp.where(mask, neighbour_contrib, 0.0)
        return segment_sum(owner_contrib, self.owner, self.n_cells) + segment_sum(
            neigh, self.safe_neighbour, self.n_cells
        )

    def scatter_conservative(self, face_flux: jnp.ndarray) -> jnp.ndarray:
        """Scatter an owner-outward face flux conservatively: owner ``+flux``, neighbour ``−flux``.

        The finite-volume conservation scatter — what a flux crossing a face adds to one cell it
        must remove from the other. Boundary faces add ``+flux`` to their owner only.

        Parameters
        ----------
        face_flux : jnp.ndarray
            Owner-outward flux per face, shape ``(n_faces, ...)``.

        Returns
        -------
        jnp.ndarray
            Net per-cell flux, shape ``(n_cells, ...)``.
        """
        return self.scatter(face_flux, -face_flux)

    def scatter_symmetric(self, face_contrib: jnp.ndarray) -> jnp.ndarray:
        """Scatter a face contribution to *both* its cells equally (owner and interior neighbour).

        For symmetric per-face quantities — a face's contribution to a cell-averaged mean, or a
        symmetric coupling coefficient's diagonal — where both incident cells receive the same
        value.

        Parameters
        ----------
        face_contrib : jnp.ndarray
            Per-face contribution, shape ``(n_faces, ...)``.

        Returns
        -------
        jnp.ndarray
            Per-cell sum, shape ``(n_cells, ...)``.
        """
        return self.scatter(face_contrib, face_contrib)

    def scatter_max(
        self, owner_contrib: jnp.ndarray, neighbour_contrib: jnp.ndarray
    ) -> jnp.ndarray:
        """Per-cell **maximum** of per-face contributions (owner always; interior neighbour only).

        The extremum counterpart of :meth:`scatter`: the boundary neighbour side is masked to the
        max identity ``-inf`` (as :meth:`scatter` masks it to the sum identity ``0``), so the
        boundary convention still lives in one place. Used to gather stencil maxima (e.g. a slope
        limiter's neighbourhood range).

        Parameters
        ----------
        owner_contrib, neighbour_contrib : jnp.ndarray
            Per-face contributions, shape ``(n_faces, ...)``.

        Returns
        -------
        jnp.ndarray
            Per-cell maximum, shape ``(n_cells, ...)``.
        """
        return self._scatter_extremum(
            owner_contrib, neighbour_contrib, -jnp.inf, segment_max, jnp.maximum
        )

    def scatter_min(
        self, owner_contrib: jnp.ndarray, neighbour_contrib: jnp.ndarray
    ) -> jnp.ndarray:
        """Per-cell **minimum** of per-face contributions (neighbour side masked to ``+inf``).

        The min counterpart of :meth:`scatter_max`; see it for the convention.
        """
        return self._scatter_extremum(
            owner_contrib, neighbour_contrib, jnp.inf, segment_min, jnp.minimum
        )

    def _scatter_extremum(self, owner_contrib, neighbour_contrib, identity, segment, combine):
        """Shared core of :meth:`scatter_max` / :meth:`scatter_min` (reducer + boundary identity)."""
        mask = _broadcast_face_mask(self.interior, neighbour_contrib.ndim)
        neigh = jnp.where(mask, neighbour_contrib, identity)
        return combine(
            segment(owner_contrib, self.owner, self.n_cells),
            segment(neigh, self.safe_neighbour, self.n_cells),
        )

    def interior_edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interior faces as a numpy edge list ``(owner, neighbour, face_index)``.

        A build-time helper (numpy) that gives the cell↔cell adjacency an aggregation multigrid or
        a graph partitioner needs: only interior faces (a real owner↔neighbour pair) appear, so the
        boundary convention is applied once here rather than open-coded at each graph builder.

        Returns
        -------
        owner, neighbour, face_index : np.ndarray of int
            Owner / neighbour cell index and the global face index of each interior face.
        """
        owner = np.asarray(self.owner)
        neighbour = np.asarray(self.neighbour)
        faces = np.nonzero(interior_mask(neighbour))[0]
        return owner[faces], neighbour[faces], faces


class FaceNodeConnectivity(eqx.Module):
    """The face→node incidence (ragged CSR) and the gather/reduce operators over a face's nodes.

    A face is an ordered ring of nodes stored ragged (each face keeps exactly its own node
    count, no padding). This object owns the CSR traversal — which node follows which around the
    perimeter, and how to sum a per-node-incidence quantity into per-face — so the face-geometry
    schemes express only the polygon math (edge normal, centre-fan triangles).

    The perimeter maps are enumerated once at build time (numpy, because the per-face node count
    is data-dependent), matching how face geometry is a once-per-mesh eager computation.

    Attributes
    ----------
    offsets : jnp.ndarray of int, shape ``(n_faces + 1,)``
        CSR row pointers: face ``f``'s nodes are ``face_node_indices[offsets[f] : offsets[f+1]]``.
    face_node_indices : jnp.ndarray of int, shape ``(n_incidences,)``
        Flat concatenation of every face's node indices (CSR order).
    face_of_incidence : jnp.ndarray of int, shape ``(n_incidences,)``
        The face each incidence belongs to — the ``segment_sum`` segment ids.
    next_incidence : jnp.ndarray of int, shape ``(n_incidences,)``
        For each incidence, the flat incidence index of the *next* node around the same face's
        perimeter (wrapping the last node back to the first).
    counts : jnp.ndarray of int, shape ``(n_faces,)``
        Number of nodes per face.
    n_faces : int
        Number of faces (static).
    """

    offsets: jnp.ndarray
    face_node_indices: jnp.ndarray
    face_of_incidence: jnp.ndarray
    next_incidence: jnp.ndarray
    counts: jnp.ndarray
    n_faces: int = eqx.field(static=True)

    @classmethod
    def from_csr(cls, face_node_offsets, face_node_indices) -> FaceNodeConnectivity:
        """Build from CSR face-node arrays, enumerating the perimeter maps once (numpy).

        Validates the CSR structure it owns (well-formed row pointers) so a malformed mesh fails
        with a clear message here rather than crashing deeper in the traversal build. Semantic
        checks that need more context (node-index range, per-face node counts vs. dimension) live
        in :meth:`aquaflux.mesh.Mesh.validate`.

        Parameters
        ----------
        face_node_offsets : array_like of int, shape ``(n_faces + 1,)``
            CSR row pointers.
        face_node_indices : array_like of int, shape ``(n_incidences,)``
            Flat node indices.

        Raises
        ------
        ValueError
            If the offsets are not a 1-D, non-decreasing pointer array starting at 0 and ending at
            ``len(face_node_indices)``.
        """
        offsets = np.asarray(face_node_offsets)
        indices = np.asarray(face_node_indices)
        if offsets.ndim != 1 or offsets.shape[0] < 1 or int(offsets[0]) != 0:
            raise ValueError("face_node_offsets must be a 1-D CSR pointer array starting at 0")
        if np.any(np.diff(offsets) < 0):
            raise ValueError("face_node_offsets must be non-decreasing (CSR row pointers)")
        if int(offsets[-1]) != indices.shape[0]:
            raise ValueError(
                f"face_node_offsets[-1] ({int(offsets[-1])}) must equal "
                f"len(face_node_indices) ({indices.shape[0]})"
            )
        counts = offsets[1:] - offsets[:-1]  # nodes per face
        n_faces = counts.shape[0]
        total = indices.shape[0]

        # One perimeter edge (hence one incidence) per face-vertex, wrapping the last back to the
        # first: the incidence's own node is the edge start, next_incidence's node the edge end.
        inc_face = np.repeat(np.arange(n_faces), counts)  # face of each incidence
        starts = offsets[:-1][inc_face]
        local = np.arange(total) - starts  # position of the incidence within its face
        next_pos = starts + (local + 1) % counts[inc_face]  # wrap to the face's first node
        return cls(
            offsets=jnp.asarray(offsets),
            face_node_indices=jnp.asarray(indices),
            face_of_incidence=jnp.asarray(inc_face),
            next_incidence=jnp.asarray(next_pos),
            counts=jnp.asarray(counts),
            n_faces=n_faces,
        )

    def gather_node_coords(self, node_coords: jnp.ndarray) -> jnp.ndarray:
        """Node coordinates for every incidence, in CSR order, shape ``(n_incidences, dim)``."""
        return node_coords[self.face_node_indices]

    def perimeter_next(self, per_incidence: jnp.ndarray) -> jnp.ndarray:
        """Reorder a per-incidence quantity to the next node around each face's perimeter."""
        return per_incidence[self.next_incidence]

    def reduce_to_faces(self, per_incidence: jnp.ndarray) -> jnp.ndarray:
        """Sum a per-incidence quantity into per-face totals, shape ``(n_faces, ...)``."""
        return segment_sum(per_incidence, self.face_of_incidence, self.n_faces)

    def vertex_mean(self, node_coords: jnp.ndarray) -> jnp.ndarray:
        """Mean of each face's node coordinates, shape ``(n_faces, dim)``."""
        return scale(self.reduce_to_faces(self.gather_node_coords(node_coords)), 1.0 / self.counts)
