"""Unit tests for the symmetric equilibration's **pattern preservation**.

:func:`~aquaflux.solve.frozen_operator.symmetrically_equilibrate` is applied to every operator that
reaches a factorization or a coarsening. Written as the sparse product ``diags(s) @ a @ diags(s)`` it
silently drops every explicit zero, because a sparse product stores only entries whose result is
nonzero. That is invisible in the values and decisive for a consumer that reads the structure: an
incomplete factorization with zero fill takes its pattern from the stored entries, so it would receive a
structurally weaker factorization of a numerically identical matrix.

These pin both halves — the values are what the product gives, and the pattern is what the caller
assembled — because a test on values alone cannot tell the two implementations apart.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.solve.frozen_operator import (
    apply_symmetric_scale,
    equilibration_scale,
    row_chunks,
    symmetrically_equilibrate,
)
from aquaflux.solve.ordering import cell_major_permutation


def _matrix_with_explicit_zeros() -> sp.csr_matrix:
    """A CSR carrying stored zeros off the diagonal, built directly so they really are stored.

    Assigning zero through ``lil``/``__setitem__`` would not store one, and a matrix built from a dense
    array has none at all — so a fixture that does either cannot exercise the property under test.
    """
    data = np.array([4.0, 0.0, 1.0, 9.0, 0.0, 2.0, 16.0])
    indices = np.array([0, 1, 2, 1, 2, 0, 2])
    indptr = np.array([0, 3, 5, 7])
    return sp.csr_matrix((data, indices, indptr), shape=(3, 3))


def test_equilibration_keeps_every_stored_entry_including_explicit_zeros() -> None:
    """The pattern out is the pattern in -- the property the sparse triple product does not have."""
    a = _matrix_with_explicit_zeros()
    assert (a.data == 0).sum() == 2  # the fixture is doing its job

    scaled, _ = symmetrically_equilibrate(a)

    assert scaled.nnz == a.nnz
    np.testing.assert_array_equal(scaled.indptr, a.indptr)
    np.testing.assert_array_equal(scaled.indices, a.indices)
    assert (scaled.data == 0).sum() == 2


def test_the_sparse_product_would_have_dropped_them() -> None:
    """The failure mode is real, not hypothetical -- pin it so the old form cannot quietly return."""
    a = _matrix_with_explicit_zeros()
    scale = equilibration_scale(a.diagonal())

    product = (sp.diags(scale) @ a @ sp.diags(scale)).tocsr()

    assert product.nnz < a.nnz


def test_values_match_the_sparse_product_exactly_where_it_stores_them() -> None:
    """Only the pattern changes: every entry the product does keep is bit-identical."""
    a = _matrix_with_explicit_zeros()
    scale = equilibration_scale(a.diagonal())

    scaled, returned_scale = symmetrically_equilibrate(a)
    product = (sp.diags(scale) @ a @ sp.diags(scale)).tocsr()

    np.testing.assert_array_equal(returned_scale, scale)
    np.testing.assert_array_equal(scaled.toarray(), product.toarray())


def test_equilibration_does_not_mutate_its_argument() -> None:
    """Scaling in place is an implementation detail; the caller's matrix must be untouched."""
    a = _matrix_with_explicit_zeros()
    before = a.data.copy()

    symmetrically_equilibrate(a)

    np.testing.assert_array_equal(a.data, before)


def test_the_scaled_diagonal_has_unit_magnitude() -> None:
    """What the rescaling is *for*: a diagonal of magnitude one, whatever the operator's own scale."""
    a = _matrix_with_explicit_zeros()

    scaled, _ = symmetrically_equilibrate(a)

    np.testing.assert_allclose(np.abs(scaled.diagonal()), 1.0)


@pytest.mark.parametrize("target", [1, 2, 5, 1000])
def test_chunking_does_not_change_the_result(target: int) -> None:
    """Every chunk size must give one answer -- a row split across chunks is scaled once, not twice."""
    a = _matrix_with_explicit_zeros()
    scale = equilibration_scale(a.diagonal())

    reference = a.copy()
    apply_symmetric_scale(reference.data, reference.indptr, reference.indices, scale)
    chunked = a.copy()
    apply_symmetric_scale(
        chunked.data,
        chunked.indptr,
        chunked.indices,
        scale,
        chunks=row_chunks(chunked.indptr, target),
    )

    np.testing.assert_array_equal(chunked.data, reference.data)


def test_an_empty_row_is_left_alone() -> None:
    """A structurally empty row has no entries to scale and must not upset the chunk arithmetic."""
    data = np.array([4.0, 9.0])
    indices = np.array([0, 2])
    indptr = np.array([0, 1, 1, 2])  # row 1 is empty
    a = sp.csr_matrix((data, indices, indptr), shape=(3, 3))

    scaled, scale = symmetrically_equilibrate(a)

    assert scaled.nnz == 2
    assert scale[1] == 1.0  # a zero diagonal scales by one rather than dividing by zero


def test_cell_major_permutation_is_a_valid_involution_free_permutation():
    """Lives here with the function: the reorder and the rescale are one transform.

    It sat in the threshold ILU's tests only because that preconditioner needed the reorder first --
    while the multigrid V-cycle, the field split and three study harnesses all use it too."""
    perm = cell_major_permutation(4, 3)
    assert sorted(perm.tolist()) == list(range(12))
    # (cell i, field f) at cell-major i*3+f maps to field-major f*4+i
    assert perm[1 * 3 + 2] == 2 * 4 + 1
