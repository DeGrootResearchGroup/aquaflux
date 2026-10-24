"""Unit tests for the cell pattern over a full block of field pairs (``KroneckerPattern``)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.precondition.entry_map import SparsePattern
from aquaflux.precondition.kronecker_pattern import KroneckerPattern, entry_position


def _cells(shape, density, seed, *, diagonal=False):
    """A random cell pattern, rows of uneven length, some empty, columns ascending."""
    matrix = sp.random(*shape, density=density, random_state=seed, format="lil")
    matrix[1, :] = 0.0  # one empty row
    matrix = sp.csr_matrix(matrix)
    if diagonal:
        matrix = matrix + sp.identity(shape[0], format="csr")
    matrix.sort_indices()
    return SparsePattern(matrix)


@pytest.mark.parametrize(
    ("fields", "shape"), [((1, 1), (7, 7)), ((2, 3), (6, 9)), ((4, 4), (8, 8))]
)
def test_the_dof_pattern_is_the_field_major_kronecker_product_of_the_cell_pattern(fields, shape):
    """Expanded, the pattern is ``kron(ones(row_fields, column_fields), C)``, CSR array for array.

    ``scipy.sparse.kron`` builds the same field-major product by a different route, and its CSR form
    (rows ascending, each row's columns ascending) is the order the stored values keep.
    """
    cells = _cells(shape, 0.3, 4)
    lengths = np.diff(cells.indptr)
    assert (
        lengths[1] == 0 and lengths.max() > lengths[lengths > 0].min()
    )  # an empty row, uneven rows
    pattern = KroneckerPattern(cells, *fields)
    expected = sp.kron(np.ones(fields), cells.csr(np.ones(cells.nnz)), format="csr")
    expected.sort_indices()

    got = pattern.dof_pattern()
    assert got.shape == pattern.shape == expected.shape
    assert pattern.nnz == expected.nnz
    assert np.array_equal(got.indptr, expected.indptr)
    assert np.array_equal(got.indices, expected.indices)


def test_entry_position_places_every_field_pair_of_every_cell_entry_where_the_dof_pattern_stores_it():
    """Each (row field, column field, cell entry) names the stored entry at ``(a n + i, b m + j)``."""
    cells = _cells((6, 5), 0.4, 9)
    fields = (3, 2)
    pattern = KroneckerPattern(cells, *fields)
    dof = pattern.dof_pattern()
    dof_rows = np.repeat(np.arange(dof.shape[0]), np.diff(dof.indptr))
    rows = pattern.cell_rows()
    entries = np.arange(cells.nnz)
    for a in range(fields[0]):
        for b in range(fields[1]):
            place = entry_position(cells.indptr, rows, entries, a, b, fields[1], cells.nnz)
            assert np.array_equal(dof_rows[place], a * 6 + rows)
            assert np.array_equal(dof.indices[place], b * 5 + cells.indices)


def test_the_diagonal_positions_are_where_each_dof_stores_its_diagonal():
    cells = _cells((9, 9), 0.25, 2, diagonal=True)
    pattern = KroneckerPattern(cells, 3, 3)
    dof = pattern.dof_pattern()
    dof_rows = np.repeat(np.arange(dof.shape[0]), np.diff(dof.indptr))
    expected = np.flatnonzero(dof.indices == dof_rows)
    assert np.array_equal(pattern.diagonal_positions(), expected)


def test_a_cell_row_without_its_diagonal_is_refused():
    cells = SparsePattern(sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 0.0]])))
    with pytest.raises(ValueError, match="stores no diagonal"):
        KroneckerPattern(cells, 2, 2).diagonal_positions()


def test_a_cell_pattern_with_unsorted_columns_is_refused():
    """An unsorted row would describe a degree-of-freedom order that is not CSR order."""
    indptr, indices = np.array([0, 2, 3]), np.array([1, 0, 1])
    cells = SparsePattern.of_arrays((2, 2), indptr, indices)
    with pytest.raises(ValueError, match="ascending"):
        KroneckerPattern(cells, 2, 2)


def test_of_coordinates_sorts_the_coordinates_into_the_cell_pattern_and_says_how():
    rows, columns = np.array([2, 0, 1, 0, 2]), np.array([0, 2, 1, 0, 2])
    pattern, order = KroneckerPattern.of_coordinates(rows, columns, (3, 3), (2, 2))
    assert np.array_equal(pattern.cells.indptr, [0, 2, 3, 5])
    assert np.array_equal(pattern.cells.indices, [0, 2, 1, 0, 2])
    assert np.array_equal(rows[order], [0, 0, 1, 2, 2])
    assert np.array_equal(columns[order], pattern.cells.indices)
