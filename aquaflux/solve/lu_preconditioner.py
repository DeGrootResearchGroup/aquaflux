"""A monolithic *complete* sparse-LU preconditioner for the coupled saddle-point Newton solve.

This factors the assembled coupled Jacobian *completely*, so the preconditioner is the operator's exact
inverse and a Krylov solve converges in a single iteration.

The complete factorization needs **no equilibration and no cell-major reordering**: the
solver's own pivoting and fill-reducing ordering handle the indefinite saddle directly on the raw
field-major matrix. The apply is therefore a plain triangular solve (``M = A^{-1}``); the adjoint's
transpose solve reuses the same factorization with a transposed solve.

**Scope — a two-dimensional / moderate-mesh tool.** A complete LU's fill grows as ``O(n log n)`` in 2D
but ``O(n^{4/3})`` in 3D, so its memory becomes the wall on large three-dimensional meshes (a few times
``10^4`` cells in 3D on a workstation). There it must give way to the multigrid-smoothed path (or a
rank-structured direct solver); this class is the exact preconditioner where the mesh is
two-dimensional or moderate.

**Factorization (host, off the jit path).** The factorization is SciPy's SuperLU
(``scipy.sparse.linalg.splu``), built once at a reference state and applied inside the jitted Krylov
solve through ``jax.pure_callback``. A mid-march refresh factors afresh, because SuperLU exposes no
separate symbolic and numeric phases -- and because the coupled Jacobian's sparsity grows as the flow
develops, so a fixed-pattern numeric-only refactor would be wrong in any case.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .host_preconditioner import HostPreconditioner
from .refresh_timing import PhaseTimer


class LuFactors:
    """A frozen complete-LU factorization of the coupled Jacobian and its forward/transpose apply.

    A pure host object (no JAX). It applies ``M = A^{-1}`` (or ``M^T``) by a triangular solve on the raw
    field-major vector -- no equilibration or reordering, because the complete factorization handles the
    saddle directly.

    Parameters
    ----------
    matrix : scipy.sparse matrix
        The assembled field-major matrix to factor, square, shape ``(n_dofs, n_dofs)``.
    """

    def __init__(self, matrix: sp.spmatrix) -> None:
        self._n_dofs = matrix.shape[0]
        self._factor(matrix)

    def _factor(self, matrix: sp.spmatrix) -> None:
        # A small diagonal-pivot threshold keeps the indefinite saddle's factorization non-singular
        # without full partial pivoting.
        self._lu = spla.splu(matrix.tocsc(), diag_pivot_thresh=0.1)

    @property
    def n_dofs(self) -> int:
        """Number of degrees of freedom the factorization acts on."""
        return self._n_dofs

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        """Apply ``M = A^{-1}`` (or ``M^T``) to a field-major residual vector.

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
        return self._lu.solve(
            np.asarray(residual, dtype=np.float64), trans="T" if transpose else "N"
        )

    def refactor_block(self, block: sp.spmatrix) -> None:
        """Factor ``block`` afresh, IN PLACE, keeping this object's identity.

        Parameters
        ----------
        block : scipy.sparse matrix
            The new assembled field-major matrix, already shifted, shape ``(n_dofs, n_dofs)``.
        """
        self._factor(block)


def factorize_lu(matrix: sp.spmatrix) -> LuFactors:
    """Completely LU-factor an assembled coupled block matrix, with SciPy's SuperLU.

    Parameters
    ----------
    matrix : scipy.sparse matrix
        The assembled field-major coupled Jacobian (already shifted for the pseudo-transient step),
        shape ``(n_fields * n_cells, n_fields * n_cells)``.

    Returns
    -------
    LuFactors
        The frozen factorization.

    Raises
    ------
    ValueError
        If ``matrix`` is not square.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"factorize_lu: matrix must be square, got {matrix.shape}.")
    return LuFactors(matrix)


class MonolithicLuPreconditioner(HostPreconditioner):
    """The coupled complete-LU preconditioner as JAX matvecs, wrapping a frozen :class:`LuFactors`.

    Shares its interface (:meth:`build`, :meth:`refresh_in_place`, :meth:`matvec`) with the
    materialized-Jacobian preconditioners (:class:`~aquaflux.solve.MaterializedJacobianPreconditioner`),
    so it is a drop-in for the coupled continuation. Not an :class:`equinox.Module`: the factorization is a host object, held by a caller and
    captured in the ``jax.pure_callback`` closure rather than threaded through the jit as a traced
    argument. Because the factors are frozen (their coefficients ``stop_gradient``-ed by the solver), the
    callback is never differentiated: the forward solve calls ``M`` and the adjoint's transpose solve
    calls ``M^T``, both only in forward evaluations.
    """

    @staticmethod
    def _materialize(matvec: Callable[[jnp.ndarray], jnp.ndarray], plan) -> sp.csr_matrix:
        """The coupled Jacobian **without** the shift, by the coloured jvp probe."""
        from .sparse_jacobian import materialize_block_jacobian

        return materialize_block_jacobian(matvec, plan).tocsr()

    @staticmethod
    def _shifted(jacobian: sp.csr_matrix, shift_diagonal: np.ndarray) -> sp.csr_matrix:
        """The Jacobian with the pseudo-transient shift on its diagonal."""
        from .sparse_jacobian import shifted_jacobian

        return shifted_jacobian(jacobian, shift_diagonal)

    @classmethod
    def build(
        cls,
        matvec: Callable[[jnp.ndarray], jnp.ndarray],
        plan,
        shift_diagonal: np.ndarray,
    ) -> MonolithicLuPreconditioner:
        """Materialize the shifted coupled Jacobian and completely factor it, off the jit path.

        Parameters
        ----------
        matvec : callable
            The frozen Jacobian-vector product ``v -> J v`` at the state it is frozen at.
        plan : ColumnProbePlan
            The probing plan for the materialization
            (:class:`~aquaflux.solve.sparse_jacobian.ColumnProbePlan`).
        shift_diagonal : np.ndarray
            The pseudo-transient shift added to the Jacobian's diagonal, shape ``(n_fields * n,)`` — the
            same block-diagonal shift the step solves against (velocity/scalar shifts, pressure zero).

        Returns
        -------
        MonolithicLuPreconditioner
            The built preconditioner.
        """
        matrix = cls._shifted(cls._materialize(matvec, plan), shift_diagonal)
        return cls(factorize_lu(matrix))

    def refresh_in_place(
        self,
        matvec: Callable[[jnp.ndarray], jnp.ndarray],
        plan,
        shift_diagonal: np.ndarray,
    ) -> tuple[tuple[str, float], ...]:
        """Re-factor at a developed state and swap the factorization IN PLACE (no new object).

        The arguments are :meth:`build`'s, evaluated at the developed state. The matrix is factored
        afresh (the coupled Jacobian's sparsity grows as the flow develops, so the pattern is not
        fixed). Because
        this preconditioner is held as a **static field** of the shift policy and :meth:`matvec` reads
        ``self.factors`` at call time, mutating the factorization here re-preconditions the **same
        compiled** Krylov solve (a compilation cache hit -- no recompile).

        **Forward-march use ONLY — the mutation is impure and must never touch a differentiated path.**
        The adjoint's transpose solve reads the same factorization and would be corrupted by a change
        between its calls; only the eager, non-differentiated march may refresh. The refresh never moves
        the converged root (the shift vanishes there), so it changes only the forward Krylov path.

        Returns
        -------
        tuple of (str, float)
            ``("probe", s), ("assemble", s), ("refactor", s)`` -- the same breakdown every
            materialized-Jacobian refresh reports, so a march log reads identically whichever
            preconditioner is installed. Here "assemble" is only the diagonal shift.
        """
        timer = PhaseTimer()
        jacobian = self._materialize(matvec, plan)
        timer.lap("probe")
        matrix = self._shifted(jacobian, shift_diagonal)
        timer.lap("assemble")
        self.factors.refactor_block(matrix)
        timer.lap("refactor")
        return timer.phases()
