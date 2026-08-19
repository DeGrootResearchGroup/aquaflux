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


def index_dtype(bound: int) -> type[np.signedinteger]:
    """The narrowest signed integer type holding every value up to ``bound``.

    Connectivity carries several arrays sized per face or per face-node incidence, so on a
    mesh of a few million cells the width of an index is the difference between a few hundred
    megabytes and a gigabyte or more, and nothing about the values themselves needs the extra
    range. The bound is always known from a shape or a value already in hand before the array
    is formed, which is what lets the width be chosen up front rather than inherited from
    whatever the platform default happens to be.

    Parameters
    ----------
    bound : int
        An upper bound (exclusive) on every value the array will hold.

    Returns
    -------
    type
        ``numpy.int32`` if it can represent every value up to ``bound``, else ``numpy.int64``.
    """
    return np.int32 if bound < np.iinfo(np.int32).max else np.int64


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

    :attr:`interior`, :attr:`safe_neighbour` and the internal scatter index are pure functions of
    :attr:`owner` / :attr:`neighbour`, computed once in :meth:`__post_init__` rather than as
    properties recomputed on every access. Outside a loop XLA folds a recomputed property away, so
    the two are equivalent there; **inside** a traced ``while_loop`` body (a Krylov matvec, say)
    they do not fold, and the compare-plus-select would run on every iteration instead of once.

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
    interior : jnp.ndarray of bool, shape ``(n_faces,)``
        ``True`` on interior faces. Computed once at construction; never pass this explicitly —
        it is derived from :attr:`neighbour` in :meth:`__post_init__`.
    safe_neighbour : jnp.ndarray of int, shape ``(n_faces,)``
        Neighbour index with boundary faces substituted by their owner (an always-in-range gather
        index: ``field[fc.owner]`` / ``field[fc.safe_neighbour]``, the boundary-safe idiom used
        throughout). A boundary face reads as ``owner == neighbour``. Computed once at
        construction; never pass this explicitly.
    """

    owner: jnp.ndarray
    neighbour: jnp.ndarray
    n_cells: int = eqx.field(static=True)
    neighbour_offset: jnp.ndarray | None = None
    interior: jnp.ndarray = None
    safe_neighbour: jnp.ndarray = None
    _neighbour_scatter_index: jnp.ndarray = None

    def __post_init__(self) -> None:
        interior = interior_mask(self.neighbour)
        object.__setattr__(self, "interior", interior)
        object.__setattr__(self, "safe_neighbour", jnp.where(interior, self.neighbour, self.owner))
        # A boundary face's contribution must never land on a real cell (it has no neighbour to
        # scatter to), so redirect it to a trash row one past the last real cell -- `scatter`
        # slices that row away, which excludes the boundary contribution structurally, with no
        # select and no extra `(n_faces, ...)` masked array.
        object.__setattr__(
            self, "_neighbour_scatter_index", jnp.where(interior, self.neighbour, self.n_cells)
        )

    @property
    def n_faces(self) -> int:
        """Number of faces — the length of every per-face array over this relation.

        Derived from :attr:`owner` rather than stored, so it cannot disagree with the arrays it
        describes. (:attr:`n_cells` is stored because it is *not* recoverable this way: no face need
        reference the last cell.) The size a consumer allocates a per-face accumulator at — a
        residual's face-flux sum, for instance — without reaching back to the whole ``Mesh``.
        """
        return self.owner.shape[0]

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

    def gather_neighbour_offset(self, face_index) -> jnp.ndarray | None:
        """The periodic :attr:`neighbour_offset` carried onto a **new face numbering**.

        A mesh transform that renumbers faces — keeping a subset of them (a partition's local
        faces) or appending new ones (a shard's inert trailing padding faces) — must carry this
        per-face translation across with them. Dropping it turns a periodic seam back into a pair
        of cells a full period apart, and nothing raises:
        an absent offset is exactly what an ordinary mesh has, so the transformed mesh simply
        reports wrong owner→neighbour displacements everywhere the seam is read (a boundary-column
        cell accrues a spurious ``L * A`` of volume, for one).

        Parameters
        ----------
        face_index : array_like of int, shape ``(n_new_faces,)``
            For each face of the new numbering, which face of *this* relation it came from. A
            negative entry marks a face that came from none of them — a newly introduced face — and
            takes a zero offset.

        Returns
        -------
        jnp.ndarray, shape ``(n_new_faces, dim)``, or None
            The offsets in the new face numbering, or ``None`` when this relation carries no offset
            — so a non-periodic mesh stays offset-free rather than gaining an array of zeros.
        """
        if self.neighbour_offset is None:
            return None
        index = jnp.asarray(face_index)
        carried = index >= 0
        gathered = self.neighbour_offset[jnp.where(carried, index, 0)]
        return jnp.where(_broadcast_face_mask(carried, gathered.ndim), gathered, 0.0)

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

        A boundary face has no neighbour, so its ``neighbour_contrib`` must not land on a real
        cell; it is excluded structurally rather than masked — every boundary face's scatter index
        points one row past the last real cell, and that trash row is sliced away after the
        reduction, so this never materializes a masked ``(n_faces, ...)`` array or runs a select.
        The owner contribution is scattered for every face, boundary faces included.

        Parameters
        ----------
        owner_contrib, neighbour_contrib : jnp.ndarray
            Per-face contributions, shape ``(n_faces, ...)`` (same shape; any trailing rank).

        Returns
        -------
        jnp.ndarray
            Per-cell sum, shape ``(n_cells, ...)``.
        """
        owner_sum = segment_sum(owner_contrib, self.owner, self.n_cells)
        neighbour_sum = segment_sum(
            neighbour_contrib, self._neighbour_scatter_index, self.n_cells + 1
        )[: self.n_cells]
        return owner_sum + neighbour_sum

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

        The extremum counterpart of :meth:`scatter`: the boundary neighbour side is excluded the
        same structural way (its own trash-row scatter target, sliced away), so the boundary
        convention still lives in one place and needs no identity element or select. Used to
        gather stencil maxima (e.g. a slope limiter's neighbourhood range).

        Parameters
        ----------
        owner_contrib, neighbour_contrib : jnp.ndarray
            Per-face contributions, shape ``(n_faces, ...)``.

        Returns
        -------
        jnp.ndarray
            Per-cell maximum, shape ``(n_cells, ...)``.
        """
        return self._scatter_extremum(owner_contrib, neighbour_contrib, segment_max, jnp.maximum)

    def scatter_min(
        self, owner_contrib: jnp.ndarray, neighbour_contrib: jnp.ndarray
    ) -> jnp.ndarray:
        """Per-cell **minimum** of per-face contributions (interior neighbour only).

        The min counterpart of :meth:`scatter_max`; see it for the convention.
        """
        return self._scatter_extremum(owner_contrib, neighbour_contrib, segment_min, jnp.minimum)

    def _scatter_extremum(self, owner_contrib, neighbour_contrib, segment, combine):
        """Shared core of :meth:`scatter_max` / :meth:`scatter_min` (reducer + boundary exclusion).

        Whatever value a boundary face's ``neighbour_contrib`` carries, its scatter index routes it
        to the trash row (see :meth:`scatter`), which this slices away before combining with the
        owner side — so, unlike the old masked form, no identity element (``±inf``) is needed here:
        the boundary face's contribution to the *real* cells is excluded structurally, not by value.
        """
        owner_result = segment(owner_contrib, self.owner, self.n_cells)
        neighbour_result = segment(
            neighbour_contrib, self._neighbour_scatter_index, self.n_cells + 1
        )[: self.n_cells]
        return combine(owner_result, neighbour_result)

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
    is data-dependent) — a build-time-only cost, like the compiled geometry arithmetic that
    consumes them.

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
        n_faces = offsets.shape[0] - 1
        total = indices.shape[0]
        # Every array formed below holds a face index (< n_faces), an incidence index (< total),
        # or a node index (the largest value already in `indices`) -- so this single bound covers
        # all of them, and one dtype keeps every arithmetic mix between the arrays same-width.
        node_bound = int(indices.max()) + 1 if indices.size else 1
        dtype = index_dtype(max(total, n_faces, node_bound))
        # Narrowed only when already integer-typed: `Mesh.validate` rejects a non-integer
        # `face_node_indices`, and a `.astype` here would silently round it into a passing one.
        if np.issubdtype(offsets.dtype, np.integer):
            offsets = offsets.astype(dtype)
        if np.issubdtype(indices.dtype, np.integer):
            indices = indices.astype(dtype)
        counts = offsets[1:] - offsets[:-1]  # nodes per face

        # One perimeter edge (hence one incidence) per face-vertex, wrapping the last back to the
        # first: the incidence's own node is the edge start, next_incidence's node the edge end.
        inc_face = np.repeat(np.arange(n_faces, dtype=dtype), counts)  # face of each incidence
        starts = offsets[:-1][inc_face]
        local = np.arange(total, dtype=dtype) - starts  # position of the incidence within its face
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
