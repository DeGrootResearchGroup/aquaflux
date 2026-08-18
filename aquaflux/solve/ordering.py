"""Elimination orderings for a zero-fill incomplete factorization of a coupled block operator.

**Why an ordering is a first-class choice here rather than a detail.** A zero-fill incomplete
factorization keeps exactly the operator's own stored entries and discards every fill the elimination
would otherwise create. *Which* entries that discards is decided entirely by the order the unknowns are
eliminated in — so for zero fill the ordering is part of the factorization, far more so than for a
factorization with fill, which can recover some of what a poor order costs it.

Measured on a coupled velocity--pressure saddle, the difference is not marginal. Holding everything else
fixed and changing only the order the same matrix is eliminated in took one stationary sweep — the
count this smoother actually runs — from **amplifying** the true residual by 5.5x to **contracting** it
to 0.07x, and took the Krylov solve that factorization preconditions from stalling to converging in
about 113 applications. On the same case it is the difference between a continuation march that goes
non-finite at its third step and one that completes a Reynolds rung at full Newton steps. That is why
this is an injected strategy and not a constant.

**The unit of ordering is the CELL, not the degree of freedom.** Every ordering here permutes whole
cells and keeps each cell's fields adjacent within the result, because interleaving is what stops the
elimination dividing by a lone continuity diagonal: in a collocated pressure--velocity discretization
that entry carries only the Rhie--Chow damping, and it is the entry most likely to come out near zero.
Orderings that break the interleave — all of one field, then the next — were measured and fail at five
of the six shift/state combinations tried, including every small-shift one, where the interleaved
orderings converge. (They are not uniformly worse: at the largest shift, field-major converges where the
mesh's own cell order does not. But nothing that fails at zero shift can serve an adjoint.) Putting the
pressures *first* is catastrophic everywhere — a single sweep growing by 1e+59. So the choice this
module exposes is *which order to visit the cells in*, and the within-cell grouping is not a parameter.

The strategies are plain host classes (``numpy``/``scipy``), matching the host V-cycle they serve;
nothing here is traced, and this module holds no mesh, no field and no ``jax``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

__all__ = [
    "AscendingRowLengthCells",
    "CellMajor",
    "CellOrder",
    "EliminationOrdering",
    "NaturalCells",
    "ReverseCuthillMcKeeCells",
    "cell_graph",
    "cell_major_permutation",
]


def cell_major_permutation(n_cells: int, n_fields: int) -> np.ndarray:
    """Permutation from cell-major to field-major degree-of-freedom ordering.

    The state is stored **field-major** — degree of freedom ``(cell i, field f)`` at ``f * n + i``.
    An incomplete factorization of the indefinite saddle is well conditioned in **cell-major** order —
    ``(cell i, field f)`` at ``i * n_fields + f`` — which interleaves the pressure among the velocity
    unknowns. This returns ``perm`` with ``perm[i * n_fields + f] = f * n + i``, so ``A[perm][:, perm]``
    reorders a field-major matrix into cell-major, and ``x[perm]`` / scatter-by-``perm`` map vectors
    across the two orderings.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    n_fields : int
        Degrees of freedom per cell.

    Returns
    -------
    np.ndarray
        The permutation, shape ``(n_fields * n_cells,)``.
    """
    perm = np.empty(n_fields * n_cells, dtype=np.int64)
    for f in range(n_fields):
        perm[f::n_fields] = f * n_cells + np.arange(n_cells)
    return perm


def cell_graph(matrix: sp.spmatrix, n_fields: int) -> sp.csr_matrix:
    """The cell-to-cell adjacency underlying a field-major block operator, as a binary pattern.

    Every field couples cell ``i`` to cell ``j`` over the same mesh connectivity, so the block pattern
    collapses by mapping each degree of freedom to its cell (``index % n_cells``, the field-major
    layout) and deduplicating. Built from the sparsity pattern alone — values play no part in an
    adjacency — and symmetrized, because the orderings below want an undirected graph while this
    operator is not symmetric.

    Parameters
    ----------
    matrix : scipy.sparse matrix
        The field-major block, shape ``(n_fields * n_cells,) * 2``.
    n_fields : int
        Degrees of freedom per cell.

    Returns
    -------
    scipy.sparse.csr_matrix
        The symmetrized cell adjacency, shape ``(n_cells, n_cells)``.
    """
    n_cells = matrix.shape[0] // n_fields
    coo = matrix.tocoo()
    collapsed = sp.coo_matrix(
        (np.ones(coo.nnz, dtype=np.int8), (coo.row % n_cells, coo.col % n_cells)),
        shape=(n_cells, n_cells),
    ).tocsr()
    collapsed.data[:] = 1
    return (collapsed + collapsed.T).tocsr()


@runtime_checkable
class CellOrder(Protocol):
    """The order an elimination visits the cells in.

    Takes the operator rather than a prepared cell graph, so each strategy derives only what it needs:
    the default order needs nothing but the cell count, and making it pay for a graph collapse it never
    reads would put that cost on every level of every mid-march refresh. The strategies that do want an
    adjacency share :func:`cell_graph` rather than each collapsing the block themselves.
    """

    def order(self, matrix: sp.spmatrix, n_fields: int) -> np.ndarray:
        """Cell indices in visiting order, shape ``(n_cells,)``, for a field-major block."""


class NaturalCells:
    """Cells in the order the mesh stores them — the identity, and the shipped default.

    On a mesh whose numbering already follows the geometry this is a reasonable order and it costs
    nothing to form. On one whose numbering does not, it is measurably the worst of the orderings here:
    an elimination in storage order discards the couplings that a geometric order would have kept
    adjacent, and on a coupled saddle that is the difference between a contracting sweep and an
    amplifying one.
    """

    def order(self, matrix: sp.spmatrix, n_fields: int) -> np.ndarray:
        """Cells in their stored order — no adjacency needed, so none is built."""
        return np.arange(matrix.shape[0] // n_fields)


class ReverseCuthillMcKeeCells:
    """Cells in reverse Cuthill--McKee order of the cell adjacency — bandwidth-reducing.

    A breadth-first level-set order, reversed. It keeps each cell adjacent to the cells it couples to,
    which for a zero-fill elimination means the entries it discards are the ones furthest from the
    diagonal — the ones a complete factorization would have filled least usefully anyway.

    Measured on a coupled velocity--pressure saddle at zero fill, this is the strongest ordering tried
    wherever it works — two to six times fewer Krylov applications than the mesh's own order at every
    small-shift state, including at **zero** shift, which is the one an implicit-function-theorem
    adjoint solves. It is also the ordering under which a full Reynolds-continuation rung marched at
    full Newton steps where the default order went non-finite by the third step.

    ⚠️ **It is not uniformly best: it fails at a large shift on an already-converged state**, where
    :class:`AscendingRowLengthCells` still converges. That combination is off the path a march or an
    adjoint takes — the shift is small by the time the state is converged — but it means this is the
    faster order rather than the safer one.
    """

    def order(self, matrix: sp.spmatrix, n_fields: int) -> np.ndarray:
        """Cells in reverse Cuthill--McKee order of the collapsed adjacency."""
        graph = cell_graph(matrix, n_fields)
        return np.asarray(csgraph.reverse_cuthill_mckee(graph, symmetric_mode=True))


class AscendingRowLengthCells:
    """Cells in ascending order of how many cells they couple to — the classic minimal-fill heuristic.

    With zero fill there is no fill to minimize, but the reasoning survives in a weaker form: a cell
    with few couplings is one whose elimination discards least. Eliminating those first leaves the
    densely-coupled cells to be handled against a matrix the sparse ones have already conditioned.

    Slightly cheaper to form than :class:`ReverseCuthillMcKeeCells` — both collapse the cell graph, but
    this sorts its row counts where that one traverses it.

    Measured on a coupled velocity--pressure saddle at zero fill, this is the **most robust** ordering
    tried: the only one that converged at every shift and state probed, including the large-shift/
    converged-state combination where reverse Cuthill--McKee fails. It pays for that by being two to
    six times slower than reverse Cuthill--McKee at the small shifts a march actually spends its steps
    at. Prefer this one when coverage matters more than speed.
    """

    def order(self, matrix: sp.spmatrix, n_fields: int) -> np.ndarray:
        """Cells sorted by ascending adjacency count, ties in stored order."""
        return np.argsort(np.diff(cell_graph(matrix, n_fields).indptr), kind="stable")


@runtime_checkable
class EliminationOrdering(Protocol):
    """The order an incomplete factorization eliminates a block's degrees of freedom in."""

    def permutation(self, matrix: sp.spmatrix, n_fields: int) -> np.ndarray:
        """Field-major degree-of-freedom indices in elimination order, shape ``(n_dofs,)``."""


class CellMajor:
    """Every field of a cell together, cells in an injected order.

    The interleave is fixed and the cell order is the choice — see this module's docstring for why that
    is the right seam. ``CellMajor()`` with the default :class:`NaturalCells` reproduces
    :func:`cell_major_permutation` exactly, so it is the shipped behaviour written as a strategy.

    Parameters
    ----------
    cells : CellOrder
        How to order the cells. Defaults to :class:`NaturalCells`, the mesh's own storage order.

    Examples
    --------
    >>> import numpy as np, scipy.sparse as sp
    >>> from aquaflux.solve.ordering import CellMajor, cell_major_permutation
    >>> a = sp.eye(6, format="csr")
    >>> np.array_equal(CellMajor().permutation(a, 3), cell_major_permutation(2, 3))
    True
    """

    def __init__(self, cells: CellOrder | None = None) -> None:
        self.cells = NaturalCells() if cells is None else cells

    def permutation(self, matrix: sp.spmatrix, n_fields: int) -> np.ndarray:
        """The elimination order over ``matrix``, whose layout is field-major.

        Raises
        ------
        ValueError
            If the matrix is not square or its size is not a multiple of ``n_fields`` — a partition
            that does not divide the operator would silently produce a permutation of the wrong length.
        """
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"an elimination ordering needs a square operator, got {matrix.shape}."
            )
        n_dofs = matrix.shape[0]
        if n_dofs % n_fields != 0:
            raise ValueError(
                f"operator size {n_dofs} is not a multiple of n_fields={n_fields}; the cell partition "
                "does not divide the operator."
            )
        n_cells = n_dofs // n_fields
        cells = np.asarray(self.cells.order(matrix, n_fields), dtype=np.int64)
        # `(field, cell) -> f * n_cells + cell`, cell by cell: the field-major index of each of a
        # cell's degrees of freedom, laid out so one cell's fields are contiguous in the result.
        return (np.arange(n_fields)[None, :] * n_cells + cells[:, None]).ravel()
