# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Zero-fill incomplete LU: the factorization, and its triangular solves, in compiled loops.

**ILU(0) keeps the operator's own sparsity pattern and drops nothing.** That single property is what
makes it the right factorization here and what no drop-tolerance routine can imitate: a drop-tolerance
factorization decides what to keep by *value*, and on a velocity--pressure saddle it removes what a pivot
needed, returning either an exactly singular factor or one whose entries run to 1e+23. Keeping the
pattern cannot do that -- the elimination touches only positions the operator already has.

Two consequences follow, and both are why this is written rather than imported:

* **The symbolic phase is free.** The pattern *is* the operator's pattern, so there is nothing to search
  for and nothing to reuse -- a refresh re-runs only the numeric pass over the same arrays. A march
  refreshes its preconditioner tens of times, and a routine that re-derives a fill pattern each time pays
  that cost every time.
* **There is no pivoting and no reordering.** Rows are eliminated in place, in the order given. That is
  not a simplification but the specification: the caller is responsible for handing over an ordering the
  factorization can survive (here, an equilibrated cell-major one).

The loops are sequential by nature -- row ``i`` depends on every row above it -- which is exactly why
this smoother belongs on a CPU and why the accelerator path uses a different relaxation entirely.

The combined factor is stored in ONE array over the operator's pattern: strictly-lower entries are ``L``
(whose unit diagonal is implicit and not stored), the diagonal and strictly-upper entries are ``U``.
"""

import numpy as np

cimport numpy as cnp
from libc.stdlib cimport free, malloc

cnp.import_array()


def diagonal_positions(cnp.int32_t[::1] indptr, cnp.int32_t[::1] indices, int n):
    """Index into the value array of each row's diagonal entry.

    Computed once per sparsity pattern and reused across every refresh, which is the whole economy of a
    zero-fill factorization: only the values change.

    Raises
    ------
    ValueError
        If a row has no stored diagonal. ILU(0) eliminates in place with no pivoting, so a missing
        diagonal is not a degenerate case to work around -- it is an operator this factorization cannot
        be applied to, and saying so is better than dividing by a zero that was never there.
    """
    cdef cnp.int32_t[::1] diag = np.empty(n, dtype=np.int32)
    cdef int i, idx, found
    for i in range(n):
        found = -1
        for idx in range(indptr[i], indptr[i + 1]):
            if indices[idx] == i:
                found = idx
                break
        if found < 0:
            # Worded to match the pure-Python reference: the two implementations must be
            # interchangeable in their failures as well as their results, and a test that pins only
            # one of them will pass on whichever build happens to be live.
            raise ValueError(
                f"row {i} stores no diagonal entry; ILU(0) eliminates in place and cannot "
                f"introduce one."
            )
        diag[i] = found
    return np.asarray(diag)


def factor(
    cnp.int32_t[::1] indptr,
    cnp.int32_t[::1] indices,
    cnp.float64_t[::1] data,
    cnp.int32_t[::1] diag,
    int n,
):
    """Factorize in place over the operator's own pattern (the IKJ formulation).

    ``data`` is overwritten with the combined factor. Column indices must be sorted within each row,
    which is what lets the lower part of a row be walked by breaking at the diagonal.

    The ``position`` scratch array is what makes the inner update O(1): it maps a column to that column's
    slot in the current row, so the update ``a_ij -= a_ik a_kj`` can find ``(i, j)`` without searching.
    Entries of row ``k`` whose column is absent from row ``i`` are exactly the fill ILU(0) discards, and
    they are skipped rather than stored.

    Raises
    ------
    ZeroDivisionError
        If a pivot is exactly zero. Raised rather than shifted: a shift changes the operator being
        factorized, and doing that silently would leave a preconditioner for a system nothing is solving.
    """
    cdef int *position = <int *> malloc(n * sizeof(int))
    if position == NULL:
        raise MemoryError("could not allocate the ILU(0) column-position scratch array")
    cdef int i, idx, idx2, j, slot
    cdef double factor_ij, pivot
    try:
        for i in range(n):
            position[i] = -1
        for i in range(n):
            for idx in range(indptr[i], indptr[i + 1]):
                position[indices[idx]] = idx
            for idx in range(indptr[i], indptr[i + 1]):
                j = indices[idx]
                if j >= i:
                    break
                pivot = data[diag[j]]
                if pivot == 0.0:
                    raise ZeroDivisionError(
                        f"ILU(0) met an exactly zero pivot at row {j}; the operator needs a different "
                        f"ordering or an explicit shift, neither of which this routine applies silently."
                    )
                data[idx] = data[idx] / pivot
                factor_ij = data[idx]
                for idx2 in range(diag[j] + 1, indptr[j + 1]):
                    slot = position[indices[idx2]]
                    if slot != -1:
                        data[slot] = data[slot] - factor_ij * data[idx2]
            for idx in range(indptr[i], indptr[i + 1]):
                position[indices[idx]] = -1
    finally:
        free(position)


def solve(
    cnp.int32_t[::1] indptr,
    cnp.int32_t[::1] indices,
    cnp.float64_t[::1] data,
    cnp.int32_t[::1] diag,
    cnp.float64_t[::1] x,
    int n,
):
    """``x <- U^-1 L^-1 x`` in place: forward substitution, then backward.

    ``L`` has an implicit unit diagonal, so the forward sweep divides by nothing.
    """
    cdef int i, idx
    cdef double acc
    for i in range(n):
        acc = x[i]
        for idx in range(indptr[i], diag[i]):
            acc = acc - data[idx] * x[indices[idx]]
        x[i] = acc
    for i in range(n - 1, -1, -1):
        acc = x[i]
        for idx in range(diag[i] + 1, indptr[i + 1]):
            acc = acc - data[idx] * x[indices[idx]]
        x[i] = acc / data[diag[i]]


def solve_transpose(
    cnp.int32_t[::1] indptr,
    cnp.int32_t[::1] indices,
    cnp.float64_t[::1] data,
    cnp.int32_t[::1] diag,
    cnp.float64_t[::1] x,
    int n,
):
    """``x <- L^-T U^-T x`` in place -- the transposed solve the adjoint needs.

    ``(L U)^T = U^T L^T``, so this is a forward sweep with ``U^T`` (lower triangular, carrying the
    diagonal) followed by a backward sweep with ``L^T`` (upper triangular, unit diagonal).

    Both are written as **scatters** rather than gathers: a compressed-row layout gives a row's entries
    cheaply and a column's expensively, so each solved unknown pushes its contribution forward into the
    remaining right-hand side instead of each unknown gathering from a column it cannot see. That keeps
    the transposed solve on the same storage as the forward one, with no second copy of the factor.
    """
    cdef int i, idx
    cdef double value
    for i in range(n):
        value = x[i] / data[diag[i]]
        x[i] = value
        for idx in range(diag[i] + 1, indptr[i + 1]):
            x[indices[idx]] = x[indices[idx]] - data[idx] * value
    for i in range(n - 1, -1, -1):
        value = x[i]
        for idx in range(indptr[i], diag[i]):
            x[indices[idx]] = x[indices[idx]] - data[idx] * value
