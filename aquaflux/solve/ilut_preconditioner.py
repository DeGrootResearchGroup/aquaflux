"""A monolithic incomplete-LU (ILUT) preconditioner for the coupled saddle-point Newton solve.

The block-triangular SIMPLE preconditioner approximates the pressure Schur complement; on a
convection-dominated collocated Rhie--Chow saddle that approximation, not its inversion, sets the
Krylov cost. This preconditioner takes the opposite tack: it factors the **assembled coupled
Jacobian** incompletely, so the factorization forms the true Schur coupling ``B F^{-1} G`` through its
fill rather than approximating it. Measured on the coupled RANS saddle it reaches the forward
tolerance in a handful of GMRES cycles where the block-triangular preconditioner needs hundreds.

Three ingredients are load-bearing and each is here for a measured reason:

* **Enough fill.** A zero-fill factorization (ILU(0)) drops exactly the fill that forms the pressure
  Schur, leaving the weak Rhie--Chow pressure diagonal and a singular factor. A threshold ILU that
  keeps that fill (small ``drop_tol``) is what makes it a strong preconditioner.
* **Equilibration.** The momentum rows and the continuity row differ in scale by more than an order of
  magnitude; a symmetric square-root-diagonal scaling ``D A D`` balances them so the incomplete
  pivots stay well conditioned.
* **Cell-major ordering.** Interleaving the per-cell fields ``[u, v, p, k, omega]`` (rather than
  storing all of one field then the next) keeps the indefinite saddle's pivots away from the zero the
  pressure block would otherwise put on the diagonal.

The factorization is a host ``scipy`` object, built **once off the jit path** at a reference state and
shift (like the frozen algebraic-multigrid blocks), and applied inside the jitted Krylov solve through
``jax.pure_callback``. Because the coefficients are frozen, the preconditioner only accelerates the
iteration — it never changes the converged solution or its adjoint — and the adjoint's transpose solve
reuses the *same* factorization with a transposed triangular solve (:meth:`IlutFactors.apply` with
``transpose=True``), so no separate adjoint preconditioner is assembled.

The heavy fill (several times the operator's own nonzero count) is affordable at moderate problem
sizes but is the method's weak point at very large three-dimensional scale, where a fill-limited or
multigrid-smoothed variant would be needed; that is deliberately out of scope here.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


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


class IlutFactors:
    """A frozen equilibrated, cell-major threshold-ILU factorization and its forward/transpose apply.

    A pure host (NumPy/SciPy) object — no JAX — so it is testable directly and is the piece
    :class:`MonolithicIlutPreconditioner` wraps in a callback. The apply reproduces the factorization's
    change of variables: scale by the equilibration, reorder to cell-major, triangular-solve, reorder
    back, scale again. ``M ~= A^{-1}``; with ``transpose=True`` it applies ``M^T``, which the adjoint's
    transpose linear solve needs.

    Attributes
    ----------
    lu : scipy.sparse.linalg.SuperLU
        The incomplete factorization of the equilibrated, cell-major matrix.
    scale : np.ndarray
        The symmetric equilibration ``diag(D)`` with ``D A D`` factored, shape ``(n_dofs,)``.
    perm : np.ndarray
        The cell-major permutation, shape ``(n_dofs,)``.
    """

    def __init__(self, lu: spla.SuperLU, scale: np.ndarray, perm: np.ndarray) -> None:
        self.lu = lu
        self.scale = scale
        self.perm = perm

    @property
    def n_dofs(self) -> int:
        """Number of degrees of freedom the factorization acts on."""
        return self.scale.shape[0]

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        """Apply ``M ~= A^{-1}`` (or ``M^T``) to a field-major residual vector.

        Parameters
        ----------
        residual : np.ndarray
            The field-major right-hand side, shape ``(n_dofs,)``.
        transpose : bool
            Apply ``M^T`` (for the adjoint transpose solve) instead of ``M``.

        Returns
        -------
        np.ndarray
            The preconditioned vector, shape ``(n_dofs,)``.
        """
        residual = np.asarray(residual, dtype=np.float64)
        scaled = (self.scale * residual)[self.perm]
        solved = self.lu.solve(scaled, trans="T" if transpose else "N")
        out = np.empty_like(residual)
        out[self.perm] = solved
        return self.scale * out


def factorize_ilut(
    matrix: sp.spmatrix,
    n_fields: int,
    *,
    fill_factor: float = 30.0,
    drop_tol: float = 1e-6,
    diag_pivot_thresh: float = 0.1,
) -> IlutFactors:
    """Equilibrate, reorder to cell-major, and threshold-ILU-factor a coupled block matrix.

    Parameters
    ----------
    matrix : scipy.sparse matrix
        The assembled field-major coupled Jacobian (already shifted for the pseudo-transient step),
        shape ``(n_fields * n_cells, n_fields * n_cells)``.
    n_fields : int
        Degrees of freedom per cell.
    fill_factor : float
        Upper bound on the factor's fill relative to ``matrix`` (``scipy.spilu``). Large enough not to
        bind — the fill is governed by ``drop_tol``.
    drop_tol : float
        Threshold below which fill entries are dropped. ``1e-6`` keeps the small-magnitude fill that
        forms the Schur coupling; a larger value drops it and weakens the preconditioner.
    diag_pivot_thresh : float
        SuperLU partial-pivoting threshold; a small positive value pivots enough to keep the indefinite
        saddle's factorization non-singular.

    Returns
    -------
    IlutFactors
        The frozen factorization.

    Raises
    ------
    ValueError
        If ``matrix`` is not square or its size is not a multiple of ``n_fields``.
    RuntimeError
        Propagated from ``scipy.spilu`` if the incomplete factor is singular (too little fill /
        insufficient pivoting).
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"factorize_ilut: matrix must be square, got {matrix.shape}.")
    n_dofs = matrix.shape[0]
    if n_dofs % n_fields != 0:
        raise ValueError(
            f"factorize_ilut: matrix size {n_dofs} is not a multiple of n_fields={n_fields}."
        )
    n_cells = n_dofs // n_fields
    matrix = matrix.tocsr()
    diagonal = np.abs(matrix.diagonal())
    diagonal[diagonal == 0.0] = 1.0
    scale = 1.0 / np.sqrt(diagonal)
    equilibrated = (sp.diags(scale) @ matrix @ sp.diags(scale)).tocsr()
    perm = cell_major_permutation(n_cells, n_fields)
    cell_major = sp.csc_matrix(equilibrated[perm][:, perm])
    lu = spla.spilu(
        cell_major, fill_factor=fill_factor, drop_tol=drop_tol, diag_pivot_thresh=diag_pivot_thresh
    )
    return IlutFactors(lu, scale, perm)


class MonolithicIlutPreconditioner:
    """The coupled ILUT preconditioner as JAX matvecs, wrapping a frozen :class:`IlutFactors`.

    Not an :class:`equinox.Module`: the factorization is a host ``scipy`` object, so this is held by a
    caller and captured in the ``jax.pure_callback`` closure rather than threaded through the jit as a
    traced argument. :meth:`matvec` returns the ``M`` (or ``M^T``) applied inside the jitted Krylov
    solve — a plain ``residual -> preconditioned`` callable, the shape :func:`~aquaflux.solve.solve_linear`
    expects for its ``preconditioner`` argument. Because the factors are frozen, the callback is never
    differentiated: the forward solve calls ``M`` and the adjoint's transpose solve calls ``M^T``, both
    only in forward evaluations.

    Build it off the jit path with :meth:`build` from the residual, a reference state, and the
    (already-formed) shift diagonal.
    """

    def __init__(self, factors: IlutFactors) -> None:
        self.factors = factors

    @classmethod
    def build(
        cls,
        matvec: Callable[[jnp.ndarray], jnp.ndarray],
        colouring,
        n_fields: int,
        shift_diagonal: np.ndarray,
        *,
        fill_factor: float = 30.0,
        drop_tol: float = 1e-6,
        diag_pivot_thresh: float = 0.1,
    ) -> MonolithicIlutPreconditioner:
        """Materialize the shifted coupled Jacobian and factor it, off the jit path.

        Parameters
        ----------
        matvec : callable
            The frozen Jacobian-vector product ``v -> J v`` at the state it is frozen at.
        colouring : BlockColouring
            The stencil colouring for the materialization
            (:func:`~aquaflux.solve.sparse_jacobian.block_stencil_colouring`).
        n_fields : int
            Degrees of freedom per cell.
        shift_diagonal : np.ndarray
            The pseudo-transient shift added to the Jacobian's diagonal, shape ``(n_fields * n,)`` —
            the same block-diagonal shift the step solves against (velocity/scalar shifts, pressure
            zero).
        fill_factor, drop_tol, diag_pivot_thresh : float
            Passed to :func:`factorize_ilut`.

        Returns
        -------
        MonolithicIlutPreconditioner
            The built preconditioner.
        """
        from .sparse_jacobian import materialize_block_jacobian

        jacobian = materialize_block_jacobian(matvec, colouring, n_fields)
        shifted = (jacobian + sp.diags(np.asarray(shift_diagonal))).tocsr()
        factors = factorize_ilut(
            shifted,
            n_fields,
            fill_factor=fill_factor,
            drop_tol=drop_tol,
            diag_pivot_thresh=diag_pivot_thresh,
        )
        return cls(factors)

    def matvec(self, *, transpose: bool = False) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """The preconditioner as a JAX callable ``residual -> M residual`` (or ``M^T``).

        Parameters
        ----------
        transpose : bool
            Return ``M^T`` (for the adjoint transpose solve) instead of ``M``.

        Returns
        -------
        callable
            A ``jax.pure_callback`` matvec applying the frozen factorization on the host.
        """
        factors = self.factors
        shape = jax.ShapeDtypeStruct((factors.n_dofs,), jnp.float64)

        def apply(residual: jnp.ndarray) -> jnp.ndarray:
            return jax.pure_callback(
                lambda r: factors.apply(r, transpose=transpose), shape, residual
            )

        return apply
