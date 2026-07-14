"""The cell--cell connectivity graph (sparsity pattern of the cell operator).

Two cells are adjacent when an interior face separates them; the resulting graph is the
sparsity pattern of the assembled cell operator. It is what bandwidth-reducing renumbering acts
on, what a graph partitioner decomposes, and what aggregation multigrid coarsens -- a single
mesh-level product shared by all three, rather than a graph rebuilt per consumer.

**Build-time only (eager numpy/scipy, never traced).** These helpers run once per mesh, before
the JAX trace begins -- they take a :class:`~aquaflux.mesh.Mesh` and return a SciPy/numpy graph
built from numpy, so they must be called during mesh preprocessing, *not* inside ``jit``/``grad``
(passing tracers to SciPy would fail). The graph is pure topology (interior owner/neighbour
pairs); it carries no differentiable data, so keeping it off the trace costs nothing.

The boundary convention (a ``neighbour < 0`` face couples no cells) is applied once, upstream, by
:meth:`~aquaflux.mesh.connectivity.FaceCellConnectivity.interior_edges`; this module only chooses
the sparse representation each consumer's library wants -- :func:`cell_adjacency_coo` (coordinate
``(row, col)`` form, for SciPy's ``csgraph`` reordering) and :func:`cell_adjacency_csr`
(compressed-sparse-row arrays, for a graph partitioner).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix

from .mesh import Mesh


def cell_adjacency_coo(mesh: Mesh) -> coo_matrix:
    """Symmetric cell--cell adjacency graph from interior faces (COO, unit weights).

    Two cells are adjacent when an interior face separates them; the graph is the sparsity
    pattern of the assembled cell operator. It is what bandwidth-reducing reordering acts on and
    what a graph partitioner decomposes -- the single source of the cell graph for both.

    Build-time only: returns a SciPy coordinate-format (``coo_matrix``) graph built from numpy, so
    call it during mesh preprocessing, not inside a ``jit``/``grad`` trace.

    Parameters
    ----------
    mesh : Mesh
        The mesh whose interior faces define the adjacency.

    Returns
    -------
    scipy.sparse.coo_matrix, shape ``(n_cells, n_cells)``
        Symmetric adjacency with unit weights (one entry per interior-face endpoint pair).
    """
    o, m, _ = mesh.face_cells.interior_edges()
    rows = np.concatenate([o, m])
    cols = np.concatenate([m, o])
    data = np.ones(rows.shape[0])
    return coo_matrix((data, (rows, cols)), shape=(mesh.n_cells, mesh.n_cells))


def cell_adjacency_csr(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Cell--cell adjacency graph as compressed-sparse-row (CSR) arrays ``(adj_offsets, adj_neighbours)``.

    The standard vertex-based graph format a partitioner consumes (Scotch ``verttab``/``edgetab``,
    METIS ``xadj``/``adjncy``): cell ``c``'s neighbours are
    ``adj_neighbours[adj_offsets[c] : adj_offsets[c + 1]]``. The CSR view of the same
    :func:`cell_adjacency_coo` graph (interior faces only; boundary faces couple no cells).

    Build-time only: returns numpy CSR arrays built via SciPy, so call it during mesh
    preprocessing, not inside a ``jit``/``grad`` trace.

    Parameters
    ----------
    mesh : Mesh
        The mesh to build the cell graph for.

    Returns
    -------
    adj_offsets : numpy.ndarray of int, shape ``(n_cells + 1,)``
        CSR row pointers (base 0).
    adj_neighbours : numpy.ndarray of int, shape ``(2 * n_interior_faces,)``
        Neighbour cell index per graph edge.
    """
    graph = cell_adjacency_coo(mesh).tocsr()
    graph.sort_indices()
    return graph.indptr.astype(np.int64), graph.indices.astype(np.int64)
