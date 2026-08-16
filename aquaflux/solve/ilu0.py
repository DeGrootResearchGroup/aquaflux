"""Zero-fill incomplete LU: the reference implementation, and the compiled one when it is available.

**What ILU(0) is, stated precisely, because the definition is what the tests check.** Factorize ``A`` as
``L U`` restricted to ``A``'s own sparsity pattern: the elimination runs normally but any update landing
on a position ``A`` does not have is discarded. The defining property is therefore

    ``(L U)_ij == A_ij``  for every ``(i, j)`` the operator stores,

with no constraint at all off the pattern. That identity is exact, holds whatever the operator, and is
what :func:`factor` is tested against — rather than a residual bound, which would only say the result is
*useful*.

Three properties follow, and each is a reason this is written rather than taken from a library:

* **Nothing is dropped by value.** A drop-tolerance factorization decides what to keep by magnitude, and
  on a velocity--pressure saddle that removes what a pivot needed: measured on this project's flow block
  it returns either an exactly singular factor or one whose entries reach 1e+23.
* **The symbolic phase is free.** The pattern is the operator's, so a refresh re-runs only the numeric
  pass over the same index arrays. A march refreshes tens of times.
* **No pivoting, no reordering.** Rows are eliminated in place in the order given, which makes the
  caller responsible for supplying an ordering the factorization survives.

The loops are irreducibly sequential -- row ``i`` depends on every row above it -- so this belongs on a
CPU, and the accelerator path uses a different relaxation entirely rather than a parallel imitation of
this one.

**The pure-Python reference here is correct but slow, and that is deliberate.** It defines the behaviour
the compiled extension must reproduce, and it keeps the package importable and testable where no
compiler is available; :data:`COMPILED` says which one is in use. Do not use the reference on a real
operator -- some tens of millions of nonzeros through a Python loop is not a slow factorization, it is
an unusable one.
"""

from __future__ import annotations

import numpy as np

__all__ = ["COMPILED", "Ilu0"]

try:  # pragma: no cover - which branch runs depends on the build, and both are tested
    from . import _ilu0 as _compiled

    COMPILED = True
except ImportError:  # pragma: no cover
    _compiled = None
    COMPILED = False


def _diagonal_positions(indptr: np.ndarray, indices: np.ndarray, n: int) -> np.ndarray:
    """Index into the value array of each row's diagonal entry.

    Computed once per sparsity pattern and reused by every refresh, which is the whole economy of a
    zero-fill factorization: only the values change.

    Raises
    ------
    ValueError
        If a row stores no diagonal. ILU(0) eliminates in place and cannot introduce one, so this is an
        operator the factorization does not apply to rather than a case to work around.
    """
    positions = np.full(n, -1, dtype=np.int32)
    for row in range(n):
        span = slice(indptr[row], indptr[row + 1])
        hit = np.flatnonzero(indices[span] == row)
        if hit.size:
            positions[row] = indptr[row] + hit[0]
    missing = np.flatnonzero(positions < 0)
    if missing.size:
        raise ValueError(
            f"rows {missing[:5].tolist()}{' …' if missing.size > 5 else ''} store no diagonal entry; "
            f"ILU(0) eliminates in place and cannot introduce one."
        )
    return positions


def _reference_factor(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, diag: np.ndarray, n: int
) -> None:
    """The IKJ elimination, in place, over the operator's own pattern.

    ``position`` maps a column to its slot in the current row, so the update ``a_ij -= a_ik a_kj`` finds
    ``(i, j)`` without searching. A column of row ``k`` that row ``i`` does not store is exactly the fill
    ILU(0) discards, and is skipped.
    """
    position = np.full(n, -1, dtype=np.int64)
    for row in range(n):
        start, stop = indptr[row], indptr[row + 1]
        position[indices[start:stop]] = np.arange(start, stop)
        for slot in range(start, stop):
            column = indices[slot]
            if column >= row:
                break
            pivot = data[diag[column]]
            if pivot == 0.0:
                raise ZeroDivisionError(
                    f"ILU(0) met an exactly zero pivot at row {column}; the operator needs a different "
                    f"ordering or an explicit shift, neither of which this routine applies silently."
                )
            data[slot] /= pivot
            multiplier = data[slot]
            upper = slice(diag[column] + 1, indptr[column + 1])
            targets = position[indices[upper]]
            present = targets >= 0
            data[targets[present]] -= multiplier * data[upper][present]
        position[indices[start:stop]] = -1


def _reference_solve(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    diag: np.ndarray,
    x: np.ndarray,
    n: int,
) -> None:
    """``x <- U^-1 L^-1 x`` in place. ``L``'s unit diagonal is implicit, so the forward sweep divides by
    nothing."""
    for row in range(n):
        lower = slice(indptr[row], diag[row])
        x[row] -= np.dot(data[lower], x[indices[lower]])
    for row in range(n - 1, -1, -1):
        upper = slice(diag[row] + 1, indptr[row + 1])
        x[row] = (x[row] - np.dot(data[upper], x[indices[upper]])) / data[diag[row]]


def _reference_solve_transpose(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    diag: np.ndarray,
    x: np.ndarray,
    n: int,
) -> None:
    """``x <- L^-T U^-T x`` in place -- the transposed solve the adjoint needs.

    ``(L U)^T = U^T L^T``, so this is a forward sweep with ``U^T`` (lower, carrying the diagonal) then a
    backward sweep with ``L^T`` (upper, unit diagonal). Both are written as **scatters**: a compressed-row
    layout gives a row cheaply and a column expensively, so each solved unknown pushes its contribution
    into the remaining right-hand side rather than gathering from a column it cannot see. That keeps the
    transpose on the same storage as the forward solve, with no second copy of the factor.
    """
    for row in range(n):
        value = x[row] / data[diag[row]]
        x[row] = value
        upper = slice(diag[row] + 1, indptr[row + 1])
        np.subtract.at(x, indices[upper], data[upper] * value)
    for row in range(n - 1, -1, -1):
        value = x[row]
        lower = slice(indptr[row], diag[row])
        np.subtract.at(x, indices[lower], data[lower] * value)


class Ilu0:
    """A zero-fill incomplete factorization of a sparse operator, refreshable in place.

    Parameters
    ----------
    matrix : scipy.sparse matrix
        The operator to factorize. Converted to compressed-sparse-row form with sorted column indices,
        which the elimination requires in order to walk a row's lower part by stopping at the diagonal.

    Attributes
    ----------
    nnz : int
        Entries in the factor — equal to the operator's, by construction.

    Raises
    ------
    ValueError
        If a row stores no diagonal entry.
    ZeroDivisionError
        If a pivot is exactly zero. Deliberately not shifted: a shift changes the operator being
        factorized, and applying one silently would give a preconditioner for a system nothing solves.
    """

    def __init__(self, matrix) -> None:
        import scipy.sparse as sp

        csr = sp.csr_matrix(matrix)
        csr.sort_indices()
        if csr.shape[0] != csr.shape[1]:
            raise ValueError(f"ILU(0) needs a square operator, got {csr.shape}.")
        self._n = csr.shape[0]
        self._indptr = csr.indptr.astype(np.int32)
        self._indices = csr.indices.astype(np.int32)
        if COMPILED:
            self._diag = _compiled.diagonal_positions(self._indptr, self._indices, self._n)
        else:
            self._diag = _diagonal_positions(self._indptr, self._indices, self._n)
        self.refactor(csr.data)

    @property
    def nnz(self) -> int:
        """Entries in the factor — the operator's own, since nothing is filled and nothing dropped."""
        return int(self._data.size)

    def refactor(self, values: np.ndarray) -> None:
        """Re-run the numeric phase on new values over the SAME pattern.

        This is the operation a mid-march refresh wants and the reason ILU(0) is the right factorization
        for one: the pattern and the diagonal map are already known, so nothing symbolic is repeated.

        Parameters
        ----------
        values : np.ndarray
            The operator's values in the stored order, shape ``(nnz,)``.

        Raises
        ------
        ValueError
            If the value array does not match the pattern this was built on — a silent mismatch would
            factorize one operator and apply it to another.
        """
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self._indices.size,):
            raise ValueError(
                f"expected {self._indices.size} values for this pattern, got {values.shape[0]}."
            )
        self._data = values.copy()
        if COMPILED:
            _compiled.factor(self._indptr, self._indices, self._data, self._diag, self._n)
        else:
            _reference_factor(self._indptr, self._indices, self._data, self._diag, self._n)

    def solve(self, rhs: np.ndarray, transpose: bool = False) -> np.ndarray:
        """``(L U)^-1 r``, or ``(L U)^-T r`` when ``transpose``."""
        x = np.array(rhs, dtype=np.float64, copy=True)
        run = (
            (_compiled.solve_transpose if transpose else _compiled.solve)
            if COMPILED
            else (_reference_solve_transpose if transpose else _reference_solve)
        )
        run(self._indptr, self._indices, self._data, self._diag, x, self._n)
        return x
