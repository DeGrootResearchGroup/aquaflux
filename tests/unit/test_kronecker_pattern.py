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


@pytest.mark.parametrize(("fields", "weighted"), [(1, False), (2, False), (4, False), (3, True)])
def test_the_cell_level_galerkin_map_is_the_degree_of_freedom_map_exactly(fields, weighted):
    """``Pᵀ A P`` at cell level: the landing the term-by-term map lands on, and its values bit for bit.

    The product of a cell pattern over every field pair through ``I ⊗ T`` sums, for each field pair, the
    same cell entries in the same order as the degree-of-freedom map's segment sum does, so the two
    agree exactly. A transfer with a fine cell in no aggregate (an empty row, as a dropped singleton
    gives), unit and non-unit weights.
    """
    import jax.numpy as jnp
    from aquaflux.precondition.entry_map import EntryMap
    from aquaflux.precondition.kronecker_pattern import KroneckerGalerkin, nodal_cell_transfer

    n, n_coarse = 12, 4
    cells = _cells((n, n), 0.3, 7, diagonal=True)
    aggregate = np.arange(n) % n_coarse
    kept = np.arange(n) != 5  # cell 5 restricts nowhere
    weights = np.random.default_rng(1).uniform(0.5, 2.0, n) if weighted else np.ones(n)
    cell_transfer = sp.csr_matrix(
        (weights[kept], (np.flatnonzero(kept), aggregate[kept])), shape=(n, n_coarse)
    )
    transfer = sp.csr_matrix(sp.kron(sp.identity(fields), cell_transfer))
    pattern = KroneckerPattern(cells, fields, fields)
    dof = pattern.dof_pattern()

    assert nodal_cell_transfer(transfer, fields) is not None
    galerkin, coarse = KroneckerGalerkin.of(pattern, nodal_cell_transfer(transfer, fields))
    product = EntryMap.product(sp.csr_matrix(transfer.T), dof, transfer)
    landing = coarse.dof_pattern()
    assert np.array_equal(landing.indptr, product.landing.indptr)
    assert np.array_equal(landing.indices, product.landing.indices)

    values = jnp.asarray(np.random.default_rng(2).standard_normal(dof.nnz))
    got = np.asarray(galerkin.apply(values))
    want = np.asarray(product.on_device().apply(values))
    assert np.array_equal(got, want)
    reference = (transfer.T @ dof.csr(np.asarray(values)) @ transfer).toarray()
    assert np.allclose(landing.csr(got).toarray(), reference, rtol=1e-12, atol=1e-12)


def test_a_transfer_that_mixes_fields_is_no_nodal_cell_transfer():
    from aquaflux.precondition.kronecker_pattern import nodal_cell_transfer

    cell = sp.csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    transfer = sp.kron(sp.identity(2), cell, format="lil")
    transfer[0, 3] = 1.0  # field 0 of cell 0 also interpolates field 1's coarse unknown
    assert nodal_cell_transfer(sp.csr_matrix(transfer), 2) is None
    assert nodal_cell_transfer(sp.kron(sp.identity(2), cell, format="csr"), 2) is not None


@pytest.mark.parametrize("from_slots", [False, True])
@pytest.mark.parametrize(
    ("rows", "columns"), [((0, 3), (0, 3)), ((0, 3), (3, 4)), ((3, 4), (0, 3)), ((1, 4), (2, 4))]
)
def test_a_field_block_gathered_at_cell_level_is_the_sliced_block(rows, columns, from_slots):
    """A block of field pairs read at cell level: the CSR slice of the dof pattern, its values exact.

    From the stored values in CSR order, and from the operator's padded-row slots padded past the
    widest row (as a budgeted level is), against slicing the assembled matrix.
    """
    import jax.numpy as jnp
    from aquaflux.precondition.kronecker_pattern import KroneckerGather
    from aquaflux.precondition.padded_rows import _padded_rows

    cells = _cells((10, 10), 0.3, 3, diagonal=True)
    pattern = KroneckerPattern(cells, 4, 4)
    dof = pattern.dof_pattern()
    values = np.random.default_rng(8).standard_normal(dof.nnz)
    matrix = dof.csr(values)
    width = 4 * int(np.diff(cells.indptr).max()) + 3
    gather, block = KroneckerGather.of(
        pattern, rows, columns, slot_width=width if from_slots else 0
    )
    operand = _padded_rows(matrix.copy(), width, "row")[1].ravel() if from_slots else matrix.data
    got = np.asarray(gather.apply(jnp.asarray(operand)))

    n = cells.shape[0]
    want = matrix[rows[0] * n : rows[1] * n, columns[0] * n : columns[1] * n].sorted_indices()
    assert np.array_equal(block.indptr, want.indptr)
    assert np.array_equal(block.indices, want.indices)
    assert np.array_equal(got, want.data)
