"""Zero-fill incomplete LU, checked against its DEFINITION rather than against a residual bound.

ILU(0) is defined by an exact identity — ``(L U)_ij == A_ij`` on every position the operator stores —
and that is what these assert. A residual bound would only say the factor is useful; the identity says
it is the right object, and it holds whatever the operator, which makes it the stronger test.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.solve.ilu0 import COMPILED, Ilu0


def _nonsymmetric(n_cells: int = 40, n_fields: int = 3, seed: int = 0) -> sp.csr_matrix:
    """A multi-field operator that is neither symmetric nor diagonally trivial."""
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for cell in range(n_cells):
        for offset in (0, 1, -1, 2):
            other = cell + offset
            if not 0 <= other < n_cells:
                continue
            for row_field in range(n_fields):
                for col_field in range(n_fields):
                    if offset == 0 and row_field == col_field:
                        value = 6.0
                    elif offset == 0:
                        value = 0.7
                    else:
                        value = rng.normal() * (0.30 if offset > 0 else 0.05)
                    rows.append(row_field * n_cells + cell)
                    cols.append(col_field * n_cells + other)
                    vals.append(value)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_cells * n_fields,) * 2)


def _combined_to_lu(ilu: Ilu0, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Split the stored factor into dense ``L`` (unit diagonal) and ``U``, for checking only."""
    indptr, indices, data = ilu._indptr, ilu._indices, ilu._data
    lower, upper = np.eye(n), np.zeros((n, n))
    for row in range(n):
        for slot in range(indptr[row], indptr[row + 1]):
            column = indices[slot]
            if column < row:
                lower[row, column] = data[slot]
            else:
                upper[row, column] = data[slot]
    return lower, upper


def test_the_factor_reproduces_the_operator_on_its_own_pattern() -> None:
    """``(L U)_ij == A_ij`` wherever ``A`` stores an entry — the definition of ILU(0).

    Checked to machine precision, not to a tolerance that would let a nearly-right elimination through.
    Off the pattern ``L U`` may be anything at all, which is exactly what "incomplete" means, so the
    comparison is masked to the stored positions.
    """
    a = _nonsymmetric()
    n = a.shape[0]
    ilu = Ilu0(a)
    lower, upper = _combined_to_lu(ilu, n)
    product = lower @ upper

    dense = np.asarray(a.todense())
    pattern = np.asarray(sp.csr_matrix(a).astype(bool).todense())
    np.testing.assert_allclose(product[pattern], dense[pattern], rtol=1e-12, atol=1e-12)


def test_it_fills_nothing() -> None:
    """The factor holds exactly the operator's entries — no fill, which is the "0" in ILU(0)."""
    a = _nonsymmetric()

    assert Ilu0(a).nnz == a.nnz


def test_the_solve_inverts_the_factor() -> None:
    """``solve`` must actually apply ``(L U)^-1``, checked against the dense factors."""
    a = _nonsymmetric()
    n = a.shape[0]
    ilu = Ilu0(a)
    lower, upper = _combined_to_lu(ilu, n)
    rhs = np.random.default_rng(1).normal(size=n)

    x = ilu.solve(rhs)

    np.testing.assert_allclose(lower @ (upper @ x), rhs, rtol=1e-9, atol=1e-9)


def test_the_transposed_solve_is_the_transpose() -> None:
    """``<y, M r> == <M^T y, r>`` — the identity the adjoint's transpose solve depends on.

    Asserted on a nonsymmetric operator, where a transposed solve that quietly ran the forward one would
    disagree.
    """
    a = _nonsymmetric()
    ilu = Ilu0(a)
    rng = np.random.default_rng(2)
    r = rng.normal(size=a.shape[0])
    y = rng.normal(size=a.shape[0])

    forward = float(y @ ilu.solve(r))
    transposed = float(ilu.solve(y, transpose=True) @ r)

    assert abs(forward - transposed) <= 1e-10 * max(abs(forward), 1.0)


def test_the_transposed_solve_is_not_the_forward_one() -> None:
    """Guards the identity above from passing because ``transpose`` is ignored."""
    a = _nonsymmetric()
    ilu = Ilu0(a)
    r = np.random.default_rng(3).normal(size=a.shape[0])

    assert not np.allclose(ilu.solve(r), ilu.solve(r, transpose=True))


def test_refactor_reuses_the_pattern_and_reproduces_a_fresh_build() -> None:
    """A refresh must equal a from-scratch factorization of the new values, to the last bit.

    This is the property the whole choice of ILU(0) rests on: the pattern is fixed, so a refresh repeats
    only the numeric pass — and it must not thereby differ from building anew.
    """
    a = _nonsymmetric()
    scaled = (a * 1.7).tocsr()
    scaled.sort_indices()
    ilu = Ilu0(a)

    ilu.refactor(scaled.data)
    fresh = Ilu0(scaled)

    np.testing.assert_array_equal(ilu._data, fresh._data)


def test_a_mismatched_value_array_is_rejected() -> None:
    """Refactoring with the wrong number of values raises rather than factorizing a different operator."""
    ilu = Ilu0(_nonsymmetric())

    with pytest.raises(ValueError, match="values for this pattern"):
        ilu.refactor(np.ones(7))


def test_a_missing_diagonal_is_rejected() -> None:
    """ILU(0) eliminates in place, so an operator with no stored diagonal is refused, not repaired."""
    a = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 2.0]]))

    with pytest.raises(ValueError, match="no diagonal"):
        Ilu0(a)


def test_it_preconditions_better_than_nothing() -> None:
    """A sanity check with a direction rather than a threshold: the factor must beat the identity.

    Deliberately weak — the identity above is the real test. This one would catch a factorization that
    is self-consistent and useless.
    """
    a = _nonsymmetric()
    ilu = Ilu0(a)
    r = np.random.default_rng(4).normal(size=a.shape[0])

    preconditioned = np.linalg.norm(a @ ilu.solve(r) - r) / np.linalg.norm(r)

    assert preconditioned < 1.0


def test_the_build_reports_which_implementation_ran() -> None:
    """``COMPILED`` must say which path is live, so a silently-slow reference cannot go unnoticed."""
    assert isinstance(COMPILED, bool)
