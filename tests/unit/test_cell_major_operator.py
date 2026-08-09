"""Unit tests for the fixed-pattern shifted cell-major assembler.

:class:`~aquaflux.solve.amg_preconditioner.ShiftedCellMajorOperator` hoists the pattern-dependent part
of a preconditioner refresh out of the refresh. It is pure NumPy/SciPy — it holds no PETSc — so these
run without the optional multigrid dependency, unlike ``test_amg_preconditioner.py``.

The contract that matters is **bit-identity** with the generic sparse composition it replaces: only how
much work is repeated per refresh may differ, never the operator handed to the factorization.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.solve.amg_preconditioner import ShiftedCellMajorOperator, _row_chunks
from aquaflux.solve.ilut_preconditioner import equilibrate_cell_major


def _fixed_pattern_jacobian(n_cells: int, n_fields: int, seed: int) -> sp.csr_matrix:
    """A block-stencil-shaped coupled Jacobian: every cell's own block plus a few neighbour blocks.

    Built the way the coloured probe builds one — a full pattern that **keeps explicit zeros**, since
    that fixed structure is what the assembler is allowed to assume.
    """
    rng = np.random.default_rng(seed)
    n_dofs = n_cells * n_fields
    blocks = {(i, i) for i in range(n_cells)}
    for i in range(n_cells):  # a couple of neighbour couplings per cell, both directions
        for j in {(i + 1) % n_cells, (i + 3) % n_cells}:
            blocks |= {(i, j), (j, i)}
    rows, cols = [], []
    for i, j in sorted(blocks):
        for a in range(n_fields):
            for b in range(n_fields):
                rows.append(a * n_cells + i)  # field-major: (cell i, field a) -> a * n_cells + i
                cols.append(b * n_cells + j)
    data = rng.normal(size=len(rows))
    data[rng.random(len(rows)) < 0.15] = 0.0  # explicit zeros, as the full pattern keeps them
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs))
    matrix.sort_indices()
    # A saddle block has zero-diagonal pressure rows; keep some so the zero-diagonal rule is exercised.
    diagonal = matrix.diagonal()
    diagonal[::7] = 0.0
    matrix.setdiag(diagonal)
    matrix.sort_indices()
    return matrix


@pytest.mark.parametrize("n_fields", [3, 5])
def test_assemble_is_bit_identical_to_the_generic_sparse_composition(n_fields: int) -> None:
    """The precomputed gather reproduces ``equilibrate_cell_major(J + diag(shift))`` exactly.

    This is the whole licence for the fast path: it is a data-movement change, not a numerical one, so
    the preconditioner built from it must be the same operator to the last bit.
    """
    n_cells = 11
    jacobian = _fixed_pattern_jacobian(n_cells, n_fields, seed=0)
    shift = np.linspace(0.5, 3.0, n_cells * n_fields)

    expected, expected_scale, expected_perm = equilibrate_cell_major(
        (jacobian + sp.diags(shift)).tocsr(), n_fields
    )
    operator = ShiftedCellMajorOperator(jacobian.indptr, jacobian.indices, n_fields)
    actual, scale, perm = operator.assemble(jacobian.data, shift)

    np.testing.assert_array_equal(perm, expected_perm)
    np.testing.assert_array_equal(scale, expected_scale)
    # The fast path keeps the full pattern's explicit zeros, so compare as dense rather than by nnz.
    np.testing.assert_array_equal(actual.toarray(), expected.toarray())


def test_a_second_assemble_reflects_the_new_shift_and_values() -> None:
    """The reused buffer is fully overwritten -- a later call cannot inherit an earlier one's values.

    The matrix aliases a preallocated buffer, so a stale carry-over would be a silent wrong operator
    rather than an error.
    """
    n_cells, n_fields = 9, 4
    first = _fixed_pattern_jacobian(n_cells, n_fields, seed=1)
    second = _fixed_pattern_jacobian(n_cells, n_fields, seed=2)
    shift_a = np.full(n_cells * n_fields, 0.25)
    shift_b = np.linspace(1.0, 2.0, n_cells * n_fields)

    operator = ShiftedCellMajorOperator(first.indptr, first.indices, n_fields)
    operator.assemble(first.data, shift_a)
    actual, scale, _ = operator.assemble(second.data, shift_b)

    expected, expected_scale, _ = equilibrate_cell_major(
        (second + sp.diags(shift_b)).tocsr(), n_fields
    )
    np.testing.assert_array_equal(actual.toarray(), expected.toarray())
    np.testing.assert_array_equal(scale, expected_scale)


def test_a_pattern_without_a_full_diagonal_is_rejected() -> None:
    """A missing diagonal entry means the shift has nowhere to go -- fail at build, not silently."""
    n_cells, n_fields = 6, 2
    n_dofs = n_cells * n_fields
    # Strictly upper-triangular: structurally valid CSR, no diagonal anywhere.
    rows = np.arange(n_dofs - 1)
    matrix = sp.csr_matrix((np.ones(n_dofs - 1), (rows, rows + 1)), shape=(n_dofs, n_dofs))
    with pytest.raises(ValueError, match="missing a diagonal entry"):
        ShiftedCellMajorOperator(matrix.indptr, matrix.indices, n_fields)


def test_a_size_that_is_not_a_multiple_of_the_field_count_is_rejected() -> None:
    """The cell-major reorder is undefined unless the degrees of freedom divide into whole cells."""
    matrix = sp.eye(7, format="csr")
    with pytest.raises(ValueError, match="not a multiple of"):
        ShiftedCellMajorOperator(matrix.indptr, matrix.indices, 2)


@pytest.mark.parametrize("target", [1, 3, 10, 10_000])
def test_row_chunks_cover_every_row_exactly_once(target: int) -> None:
    """The chunked symmetric scaling must partition the rows -- a gap silently leaves a row unscaled."""
    indptr = np.array([0, 2, 2, 7, 9, 14, 20])  # includes an empty row
    chunks = _row_chunks(indptr, target)

    assert chunks[0][0] == 0
    assert chunks[-1][1] == indptr.shape[0] - 1
    covered = [row for start, stop in chunks for row in range(start, stop)]
    assert covered == list(range(indptr.shape[0] - 1))


def test_row_chunks_of_an_empty_matrix_is_empty() -> None:
    """No rows means no work, not a one-element chunk of nothing."""
    assert _row_chunks(np.array([0]), 4) == ()


def test_equilibrate_cell_major_returns_canonical_csr():
    """The cell-major reorder must leave each row's column indices ASCENDING.

    The permutation that produces cell-major order does not preserve column order, and a consumer that
    assumes ascending indices then reads the wrong entries. PETSc's AIJ format is exactly such a
    consumer, and it fails ASYMMETRICALLY: handed this matrix unsorted, a point-block-Jacobi
    preconditioner returns NaN in most entries while point Jacobi and incomplete-LU are unaffected,
    because a diagonal scan does not care about column order and a block extraction does. That looks
    precisely like a broken block method, which is how it was first (mis)diagnosed -- so the ordering is
    established at the source and pinned here.
    """
    import numpy as np
    import scipy.sparse as sp
    from aquaflux.solve.ilut_preconditioner import equilibrate_cell_major

    # Dense-ish so the permutation genuinely scrambles the column order within each row.
    rng = np.random.default_rng(5)
    n_cells, n_fields = 6, 3
    n = n_cells * n_fields
    dense = rng.standard_normal((n, n))
    dense += np.diag(np.abs(dense).sum(axis=1) + 1.0)  # keep the diagonal safely nonzero
    reordered, _, _ = equilibrate_cell_major(sp.csr_matrix(dense), n_fields)

    assert reordered.has_sorted_indices
    for row in range(n):
        cols = reordered.indices[reordered.indptr[row] : reordered.indptr[row + 1]]
        assert np.all(np.diff(cols) > 0), f"row {row} column indices are not ascending: {cols}"
    # Sorting must not have changed the matrix, only its storage order.
    check = reordered.copy()
    check.sort_indices()
    assert abs(reordered - check).nnz == 0
