"""Assemble and rescale a frozen transport operator, for the algebraic-multigrid setup.

The preconditioners freeze an approximate linearization of a transport equation — a symmetric
diffusive coupling on the interior faces, optionally plus first-order-upwind convection at a
reference flux — and coarsen it once, off the jit path. It sits beside the coarsening it feeds, and
deliberately outside :mod:`aquaflux.solve.multigrid`, which stays a pure operator-coarsening library.
The first-order-upwind stencil is the *preconditioner's* choice, not the model's: whatever scheme the
residual discretizes advection with, the frozen operator always upwinds first-order, because that is
what makes it a diagonally dominant M-matrix an aggregation hierarchy can coarsen.

Every consumer of a frozen operator builds it through this one assembler: the pressure Schur and the
viscous velocity block (symmetric, ``flux=None``), the convection-aware velocity block, and the k/omega
scalar-transport preconditioner. The multigrid builders in :mod:`aquaflux.solve.multigrid` then take
the assembled matrix, so they stay a pure operator-coarsening library.

It also holds the **symmetric square-root-diagonal equilibration rule**
(:func:`equilibration_scale`), because a frozen operator is rescaled before it is factored *or*
coarsened and both must apply the identical rule. They did not: the incomplete factorizations
rescaled and the aggregation did not, so the two coarsened different matrices and no comparison
between them meant anything. One home is what makes that failure impossible rather than merely
unlikely.

This module imports only ``numpy`` and ``scipy.sparse`` — it holds no mesh, no field, and no
``jax`` — so it stays testable on a bare graph and adds no dependency to any subsystem that already
builds a hierarchy.
"""

from __future__ import annotations

import itertools

import numpy as np
import scipy.sparse as sp

#: Target nonzeros per row-chunk in :func:`apply_symmetric_scale`. Bounds the transient allocation
#: there to a few megabytes rather than the size of the matrix's values; small enough to stay in
#: cache, large enough that the per-chunk NumPy overhead is negligible.
_SCALE_CHUNK_NNZ = 1 << 20


def row_chunks(
    indptr: np.ndarray, target_nnz: int = _SCALE_CHUNK_NNZ
) -> tuple[tuple[int, int], ...]:
    """Split ``[0, n_rows)`` into ``(start, stop)`` row ranges of roughly ``target_nnz`` nonzeros each.

    Ranges are cut on row boundaries so each chunk is a contiguous slice of the compressed-sparse-row
    (CSR) values, and every row lands in exactly one chunk. A row wider than ``target_nnz`` simply forms
    its own oversized chunk.

    Parameters
    ----------
    indptr : np.ndarray
        The CSR row-pointer array, shape ``(n_rows + 1,)``.
    target_nnz : int
        Approximate nonzeros per chunk.

    Returns
    -------
    tuple of tuple of int
        The ``(start, stop)`` row ranges, covering every row exactly once.
    """
    n_rows = int(indptr.shape[0]) - 1
    if n_rows == 0:
        return ()
    edges = np.searchsorted(indptr, np.arange(0, int(indptr[-1]), target_nnz), side="right") - 1
    bounds = np.unique(np.concatenate(([0], np.maximum(edges, 0), [n_rows])))
    return tuple((int(a), int(b)) for a, b in itertools.pairwise(bounds) if b > a)


def apply_symmetric_scale(
    data: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    scale: np.ndarray,
    chunks: tuple[tuple[int, int], ...] | None = None,
) -> None:
    """Scale stored entries **in place** by ``scale[row] * scale[column]`` — ``D A D``, pattern-preserving.

    **This is why it is not written as the sparse product ``diags(scale) @ a @ diags(scale)``, and the
    distinction is load-bearing rather than a performance detail.** A sparse product stores only the
    entries whose *result* is nonzero, so it silently deletes every explicit zero the operator carried.
    Those positions are not decoration: an incomplete factorization with zero fill takes its pattern
    from exactly the stored entries, so dropping them gives it a structurally weaker factorization of
    the same matrix. Scaling the values in place cannot change which entries exist, so the operator a
    factorization or a coarsening receives has the pattern its assembler chose.

    Scaling per stored entry is also strictly cheaper than two sparse products over the whole matrix,
    and it is applied in row chunks so the row factor's per-nonzero expansion never allocates an array
    the size of the values.

    Parameters
    ----------
    data : np.ndarray
        The CSR values, shape ``(nnz,)``. **Modified in place.**
    indptr, indices : np.ndarray
        The CSR structure, shapes ``(n_rows + 1,)`` and ``(nnz,)``.
    scale : np.ndarray
        The per-row/column factor, shape ``(n_rows,)`` (see :func:`equilibration_scale`).
    chunks : tuple of tuple of int, optional
        Precomputed row ranges from :func:`row_chunks`, for a caller that holds a fixed structure and
        would otherwise re-derive them every call. ``None`` (default) derives them here.
    """
    counts = np.diff(indptr)
    for start, stop in row_chunks(indptr) if chunks is None else chunks:
        lo, hi = int(indptr[start]), int(indptr[stop])
        block = data[lo:hi]
        # The row factor needs one entry per nonzero; the column factor is a gather on the indices.
        block *= np.repeat(scale[start:stop], counts[start:stop])
        block *= scale[indices[lo:hi]]


def equilibration_scale(diagonal: np.ndarray) -> np.ndarray:
    """The symmetric square-root-diagonal equilibration factor ``diag(D) = 1/sqrt(|diag A|)``.

    The rule on its own, so that a caller which already holds the diagonal — or which moves the matrix
    data itself by a precomputed gather rather than by generic sparse products — applies the identical
    one rather than restating it. A zero diagonal entry is treated as one, so a structurally empty row
    scales by one instead of producing ``inf``.

    Parameters
    ----------
    diagonal : np.ndarray
        The matrix diagonal, shape ``(n_dofs,)``.

    Returns
    -------
    np.ndarray
        The equilibration factor, shape ``(n_dofs,)``.
    """
    magnitude = np.abs(np.asarray(diagonal, dtype=np.float64))
    magnitude[magnitude == 0.0] = 1.0
    return 1.0 / np.sqrt(magnitude)


def symmetrically_equilibrate(a: sp.spmatrix) -> tuple[sp.csr_matrix, np.ndarray]:
    """Rescale ``a`` to a unit-magnitude diagonal: return ``(D A D, diag(D))``.

    The similarity transform behind every rescaled frozen operator, with ``D`` from
    :func:`equilibration_scale`. It is symmetric, so it preserves symmetry, sparsity pattern and
    definiteness, and it is exactly invertible: a solve of ``(D A D) y = D b`` recovers ``A x = b`` as
    ``x = D y``. Nothing is approximated — only the scale in which the operator is presented to a
    factorization or a coarsening.

    **The sparsity pattern is preserved entry for entry** (:func:`apply_symmetric_scale`), including any
    explicit zeros. Written as the sparse product ``diags(scale) @ a @ diags(scale)`` it would not be:
    a sparse product stores only entries whose result is nonzero, so every explicit zero would be
    dropped. That is invisible in the values and decisive for a consumer that reads the structure — an
    incomplete factorization with zero fill takes its pattern from the stored entries, so it would be
    handed a structurally weaker factorization of a numerically identical matrix, and a caller that
    assembled a deliberately fixed pattern would silently not get one.

    Why it matters to the coarsening in particular: every step of a multigrid setup reads the diagonal
    — the smoother's damping, the prolongation smoothing, and the spectral estimates that scale both.
    On an operator whose diagonal spans several orders of magnitude those steps are calibrated against
    a scale that has no physical meaning, and the resulting hierarchy is not the one the same algorithm
    would build on the rescaled matrix.

    Parameters
    ----------
    a : scipy.sparse matrix
        The operator to rescale, shape ``(n, n)``.

    Returns
    -------
    scaled : scipy.sparse.csr_matrix
        ``D A D``, shape ``(n, n)``.
    scale : np.ndarray
        ``diag(D)``, shape ``(n,)`` — apply to the right-hand side before, and to the result after, a
        solve with ``scaled``.
    """
    scaled = a.tocsr().copy()
    scale = equilibration_scale(scaled.diagonal())
    apply_symmetric_scale(scaled.data, scaled.indptr, scaled.indices, scale)
    return scaled, scale


def require_valid_graph(n: int, owner: np.ndarray, nb: np.ndarray, where: str) -> None:
    """Reject a malformed interior-face graph before it is assembled into an operator.

    The frozen operators are assembled and coarsened once, off the jit path, and then held fixed; a
    bad graph would otherwise bake ``inf``/``NaN`` into the frozen preconditioner (via a later
    zero-diagonal inversion) and only show up as a silently stalling runtime V-cycle. Checks the
    invariants that must hold for *any* mesh: at least one cell, matched edge arrays, and every edge
    index in range.

    Parameters
    ----------
    n : int
        Number of cells.
    owner, nb : np.ndarray
        Interior-face edge endpoints, shape ``(n_edges,)`` each.
    where : str
        Caller name, included in the error message -- so one validator can serve callers that report
        their own name, which is why this is shared rather than re-inlined per caller.

    Raises
    ------
    ValueError
        If ``n < 1``, ``owner`` and ``nb`` differ in length, or any endpoint is outside ``[0, n)``.
    """
    if n < 1:
        raise ValueError(f"{where}: need at least one cell, got n={n}.")
    owner, nb = np.asarray(owner), np.asarray(nb)
    if owner.shape != nb.shape:
        raise ValueError(
            f"{where}: owner and nb must have the same shape, got {owner.shape} and {nb.shape}."
        )
    if owner.size and (owner.min() < 0 or owner.max() >= n or nb.min() < 0 or nb.max() >= n):
        raise ValueError(f"{where}: edge endpoints out of range for n={n} cells.")


def convection_diffusion_operator(
    owner: np.ndarray,
    nb: np.ndarray,
    coefficient: np.ndarray,
    n: int,
    *,
    flux: np.ndarray | None = None,
    boundary_diagonal: np.ndarray | None = None,
) -> sp.csr_matrix:
    """Frozen convection-diffusion operator ``A`` on an interior-face graph, as a scipy CSR matrix.

    Each interior edge ``(owner P, neighbour N)`` carries a symmetric diffusive coupling
    ``coefficient`` (e.g. ``Gamma_face A / (d.n)``). With a ``flux`` it also carries a
    first-order-upwind convective coupling from the owner-outward face flux: an outflow
    (``flux > 0``) advects the owner value, an inflow the neighbour value. The entries are

        A[P, N] = -(coefficient + max(-flux, 0)),   A[N, P] = -(coefficient + max(flux, 0)),

    with the matching diagonal contributions ``coefficient + max(flux, 0)`` at P and
    ``coefficient + max(-flux, 0)`` at N — so ``A`` is a diagonally dominant M-matrix, nonsymmetric
    exactly where the flux is non-zero. Without a ``flux`` the convective terms vanish and ``A`` is
    the symmetric graph Laplacian of the edge coefficients (the pressure-Schur and viscous-momentum
    case). ``boundary_diagonal`` adds the per-cell boundary-face contributions the interior edges do
    not carry (Dirichlet wall/inlet stiffness, outflow convection, a reaction linearization).

    Parameters
    ----------
    owner, nb : np.ndarray
        Interior-face edge endpoints, shape ``(n_edges,)`` each.
    coefficient : np.ndarray
        Per-edge symmetric diffusive coefficient, shape ``(n_edges,)``.
    n : int
        Number of cells.
    flux : np.ndarray, optional
        Per-edge owner-outward convective face flux (the frozen convective linearization), shape
        ``(n_edges,)``. Omit for a symmetric (pure diffusion) operator.
    boundary_diagonal : np.ndarray, optional
        Per-cell boundary-face diagonal contribution, shape ``(n_cells,)``.

    Returns
    -------
    scipy.sparse.csr_matrix
        The assembled operator, shape ``(n, n)``.
    """
    require_valid_graph(n, owner, nb, "convection_diffusion_operator")
    o, m = np.asarray(owner), np.asarray(nb)
    c = np.asarray(coefficient)
    zero = np.zeros_like(c)
    f = zero if flux is None else np.asarray(flux)
    up_out = np.maximum(f, 0.0)  # outflow leaves the owner: owner value is upwind
    up_in = np.maximum(-f, 0.0)  # inflow enters the owner: neighbour value is upwind
    rows = np.concatenate([o, m, o, m])
    cols = np.concatenate([m, o, o, m])
    vals = np.concatenate([-(c + up_in), -(c + up_out), c + up_out, c + up_in])
    a = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    if boundary_diagonal is not None:
        a = a + sp.diags(np.asarray(boundary_diagonal))
    return a.tocsr()


def decouple_dof(a: sp.csr_matrix, index: int) -> sp.csr_matrix:
    """Decouple one degree of freedom from ``a``: zero its row and column, unit diagonal.

    The regularization for a closed-domain pressure system, whose operator is otherwise singular (a
    pure-Neumann Laplacian defines pressure only up to a constant). Decoupling the pinned cell — as
    opposed to handling the pin after the fact — leaves the operator nonsingular and makes the pinned
    cell a singleton in any subsequent aggregation, so the coarse space's null space matches the
    pinned outer Jacobian and no post-hoc pin handling is needed in the V-cycle.

    Parameters
    ----------
    a : scipy.sparse matrix
        The assembled operator, shape ``(n, n)``.
    index : int
        The pinned cell index.

    Returns
    -------
    scipy.sparse.csr_matrix
        The operator with row/column ``index`` decoupled.
    """
    a = a.tolil()
    a[index, :] = 0
    a[:, index] = 0
    a[index, index] = 1.0
    return a.tocsr()


# --- cell-major reordering -------------------------------------------------------------
#
# The other half of the transform above. `symmetrically_equilibrate` rescales; these reorder, and every
# consumer applies the two together -- a factorization or a coarsening wants the matrix both
# unit-diagonal and grouped by cell, and `equilibrate_cell_major` below is exactly that pair.
#
# They lived in `ilut_preconditioner.py` because the threshold ILU needed them first, but three of the
# four consumers are elsewhere and the multigrid V-cycle uses them more than the ILUT does. That made
# the ILUT -- the family member most likely to be deleted, being dominated by the complete LU at 2D and
# by the AMG at 3D -- load-bearing for two preconditioners with nothing to do with it.


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


def equilibrate_cell_major(
    matrix: sp.spmatrix, n_fields: int
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Symmetrically equilibrate an assembled coupled block matrix and reorder it to cell-major.

    The two conditioning transforms the indefinite Rhie--Chow saddle needs before *any* incomplete
    factorization or multigrid smoother acts on it, shared by :func:`factorize_ilut` and the multigrid
    preconditioner so both precondition the identical operator:

    * **Symmetric square-root-diagonal equilibration** ``D A D`` with ``D = diag(1/sqrt(|diag A|))`` — the
      momentum and continuity rows differ in scale by more than an order of magnitude, and this balances
      them so the incomplete pivots (or the smoother's) stay well conditioned.
    * **Cell-major reordering** — interleave the per-cell fields ``[u, v, (w,) p, k, omega]`` (rather than
      all of one field then the next), so each cell's degrees of freedom occupy a contiguous block and a
      pressure unknown is eliminated among the velocity unknowns of its own cell rather than after all of
      them.

    ⚠️ **On why the interleaving helps, be careful what is claimed.** An earlier version of this docstring
    said it "keeps the pressure among the velocity unknowns so the saddle does not present a zero pivot".
    That is stronger than anything demonstrated, and the literature does not support it as stated: the
    published saddle-point incomplete factorizations that report *stable* factorizations number velocity
    first and pressure last — the opposite grouping — because eliminating the velocities first fills the
    pressure diagonal before it is reached (Konshin, Olshanskii & Vassilevski, *SIAM J. Sci. Comput.*
    37(5), 2015). What *is* proven is narrower and is about pairing rather than grouping: for F-matrices,
    an ordering in which each pressure node is eliminated together with a connected velocity node is
    numerically stable (de Niet & Wubs, *IMA J. Numer. Anal.* 29(1), 2009), and cell-major is a coarse
    approximation of that. Neither result covers a Rhie–Chow (p,p) block, which is nonzero here, so this
    ordering is a reasonable default rather than a guaranteed one.

    Parameters
    ----------
    matrix : scipy.sparse matrix
        The assembled **field-major** coupled Jacobian (already shifted), shape ``(n_fields * n, ...)``.
    n_fields : int
        Degrees of freedom per cell.

    Returns
    -------
    cell_major : scipy.sparse.csr_matrix
        The equilibrated, cell-major matrix ``(D A D)`` reordered by ``perm``.
    scale : np.ndarray
        The equilibration ``diag(D)``, shape ``(n_dofs,)`` — applied to a field-major vector before, and
        after, the reordered solve/apply.
    perm : np.ndarray
        The cell-major permutation, shape ``(n_dofs,)`` (see :func:`cell_major_permutation`).

    Raises
    ------
    ValueError
        If ``matrix`` is not square or its size is not a multiple of ``n_fields``.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"equilibrate_cell_major: matrix must be square, got {matrix.shape}.")
    n_dofs = matrix.shape[0]
    if n_dofs % n_fields != 0:
        raise ValueError(
            f"equilibrate_cell_major: matrix size {n_dofs} is not a multiple of n_fields={n_fields}."
        )
    equilibrated, scale = symmetrically_equilibrate(matrix)
    perm = cell_major_permutation(n_dofs // n_fields, n_fields)
    reordered = equilibrated[perm][:, perm].tocsr()
    # Canonical form, because the permutation above leaves each row's column indices OUT OF ORDER and a
    # consumer that assumes ascending indices then reads the wrong entries. PETSc's AIJ format is exactly
    # such a consumer: handed this matrix unsorted, a point-block-Jacobi preconditioner returns NaN in
    # most entries while a point-Jacobi one is unaffected -- a diagonal scan does not care about column
    # order, a block extraction does. That asymmetry looks precisely like a broken block method and is
    # not, so the ordering is established here rather than left to each caller to remember. `sort_indices`
    # is a no-op on an already-canonical matrix, so callers that sort defensively cost nothing.
    reordered.sort_indices()
    return reordered, scale, perm
