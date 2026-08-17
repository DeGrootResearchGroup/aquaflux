"""The elimination orderings: valid permutations, the right structure, and the default unchanged."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.solve.ordering import (
    AscendingRowLengthCells,
    CellMajor,
    NaturalCells,
    ReverseCuthillMcKeeCells,
    cell_graph,
    cell_major_permutation,
)

CELL_ORDERS = (NaturalCells(), ReverseCuthillMcKeeCells(), AscendingRowLengthCells())


def line_graph_operator(n_cells: int, n_fields: int) -> sp.csr_matrix:
    """A field-major block over a 1-D chain of cells: each cell couples to its neighbours, all fields.

    Small enough to reason about by hand and structured enough that a bandwidth-reducing order has
    something to do — which a random pattern would not give.
    """
    rows, cols = [], []
    for cell in range(n_cells):
        for other in {cell, max(cell - 1, 0), min(cell + 1, n_cells - 1)}:
            for f in range(n_fields):
                for g in range(n_fields):
                    rows.append(f * n_cells + cell)
                    cols.append(g * n_cells + other)
    size = n_cells * n_fields
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(size, size)).tocsr()


@pytest.mark.parametrize("cells", CELL_ORDERS, ids=lambda c: type(c).__name__)
def test_every_ordering_is_a_valid_permutation(cells) -> None:
    """Each degree of freedom is eliminated exactly once — the property every arm silently assumes.

    A "permutation" that repeated or dropped an index would factorize a different matrix and still
    run, which is the failure this cannot be allowed to have.
    """
    n_cells, n_fields = 12, 3
    permutation = CellMajor(cells).permutation(line_graph_operator(n_cells, n_fields), n_fields)
    np.testing.assert_array_equal(np.sort(permutation), np.arange(n_cells * n_fields))


@pytest.mark.parametrize("cells", CELL_ORDERS, ids=lambda c: type(c).__name__)
def test_every_ordering_keeps_a_cell_s_fields_adjacent(cells) -> None:
    """Whatever order the cells come in, one cell's degrees of freedom stay contiguous.

    This is the invariant the module is built on rather than an incidental property: the interleave is
    what stops the elimination reaching a pressure unknown only after every velocity unknown, and a
    cell order is allowed to permute cells but never to break their grouping.
    """
    n_cells, n_fields = 12, 3
    permutation = CellMajor(cells).permutation(line_graph_operator(n_cells, n_fields), n_fields)
    for block in permutation.reshape(n_cells, n_fields):
        assert len({index % n_cells for index in block}) == 1


def test_cell_major_over_the_natural_order_is_the_shipped_permutation() -> None:
    """``CellMajor()`` must reproduce ``cell_major_permutation`` exactly — the default cannot move.

    Every existing measurement of the incomplete factorizations was taken under that permutation, so a
    default that drifted would silently invalidate them while every test still passed.
    """
    n_cells, n_fields = 9, 4
    np.testing.assert_array_equal(
        CellMajor().permutation(line_graph_operator(n_cells, n_fields), n_fields),
        cell_major_permutation(n_cells, n_fields),
    )


def test_reverse_cuthill_mckee_reduces_the_bandwidth_of_a_shuffled_chain() -> None:
    """The bandwidth-reducing order must actually reduce bandwidth, on a chain whose cells are shuffled.

    Built by relabelling a 1-D chain at random, so the natural order is bad by construction and a
    genuine improvement is separable from a lucky one.
    """
    n_cells, n_fields = 60, 3
    chain = line_graph_operator(n_cells, n_fields)
    shuffle = np.random.default_rng(0).permutation(n_cells)
    scatter = (np.arange(n_fields)[:, None] * n_cells + shuffle[None, :]).ravel()
    shuffled = chain[scatter][:, scatter].tocsr()

    def bandwidth(matrix, permutation):
        reordered = matrix[permutation][:, permutation].tocoo()
        return int(np.abs(reordered.row - reordered.col).max())

    natural = CellMajor(NaturalCells()).permutation(shuffled, n_fields)
    rcm = CellMajor(ReverseCuthillMcKeeCells()).permutation(shuffled, n_fields)
    assert bandwidth(shuffled, rcm) < bandwidth(shuffled, natural)


def test_ascending_row_length_visits_the_least_coupled_cells_first() -> None:
    """The row-length order is non-increasing in adjacency count, which is the whole of its claim."""
    n_cells, n_fields = 20, 2
    operator = line_graph_operator(n_cells, n_fields)
    graph = cell_graph(operator, n_fields)
    order = AscendingRowLengthCells().order(operator, n_fields)
    counts = np.diff(graph.indptr)[order]
    assert np.all(np.diff(counts) >= 0)


def test_the_cell_graph_collapses_fields_onto_the_mesh_connectivity() -> None:
    """A block operator's cell graph is the chain it was built from, independent of the field count."""
    n_cells = 8
    for n_fields in (1, 3, 5):
        graph = cell_graph(line_graph_operator(n_cells, n_fields), n_fields)
        assert graph.shape == (n_cells, n_cells)
        dense = (graph.toarray() != 0).astype(int)
        expected = np.abs(np.subtract.outer(np.arange(n_cells), np.arange(n_cells))) <= 1
        np.testing.assert_array_equal(dense, expected.astype(int))


@pytest.mark.parametrize(
    ("shape", "n_fields", "message"),
    [((6, 5), 3, "square"), ((7, 7), 3, "not a multiple")],
)
def test_a_partition_that_does_not_divide_the_operator_raises(shape, n_fields, message) -> None:
    """A mismatched partition must raise, not return a permutation of the wrong length.

    Silently producing one would index the matrix out of range at best, and at worst reorder it into a
    different operator that factorizes perfectly well and preconditions nothing.
    """
    with pytest.raises(ValueError, match=message):
        CellMajor().permutation(sp.eye(shape[0], shape[1], format="csr"), n_fields)
