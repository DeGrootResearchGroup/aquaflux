"""A monolithic *complete* sparse-LU preconditioner for the coupled saddle-point Newton solve.

The sibling of :mod:`~aquaflux.solve.ilut_preconditioner`. Where the ILUT factors the coupled Jacobian
*incompletely* (threshold dropping) and reaches the Krylov tolerance in a handful of cycles, this factors
it *completely*, so the preconditioner is the operator's exact inverse and a Krylov solve converges in a
single iteration. On a moderate two-dimensional mesh a good multifrontal LU (UMFPACK) factors the coupled
Jacobian roughly an order of magnitude faster than the threshold-ILU costs — because it uses a
fill-reducing ordering and a dense-block (BLAS-3) numeric kernel rather than paying the ILU's
drop-tolerance and pivoting-search overhead — while being exact rather than approximate.

Unlike the ILUT, the complete factorization needs **no equilibration and no cell-major reordering**: the
solver's own pivoting and fill-reducing ordering handle the indefinite saddle directly on the raw
field-major matrix. The apply is therefore a plain triangular solve (``M = A^{-1}``); the adjoint's
transpose solve reuses the same factorization with a transposed solve.

**Scope — a two-dimensional / moderate-mesh tool.** A complete LU's fill grows as ``O(n log n)`` in 2D
but ``O(n^{4/3})`` in 3D, so its memory becomes the wall on large three-dimensional meshes (a few times
``10^4`` cells in 3D on a workstation). There it must give way to the incomplete / multigrid-smoothed
paths (or a rank-structured direct solver); this class is the fast, exact preconditioner where the mesh
is two-dimensional or moderate.

**Factorization backend (host, off the jit path).** The factorization is a host object, built once at a
reference state and applied inside the jitted Krylov solve through ``jax.pure_callback`` — exactly like
the ILUT. Two backends implement the same small interface:

* ``"umfpack"`` — UMFPACK (SuiteSparse) via ``petsc4py``, the fast path. A mid-march refresh rebuilds the
  factorization from scratch (the coupled Jacobian's sparsity grows as the flow develops, so a
  fixed-pattern numeric-only refactor would be wrong) -- cheap because the full factorization is fast.
  Requires the optional ``petsc`` dependency.
* ``"scipy"`` — ``scipy.sparse.linalg.splu`` (SuperLU), always available. Exact and correct but without a
  fill-reducing nested-dissection ordering it is not faster to factor than the ILUT; it is the fallback
  so the class works with no optional dependency, and it is what the tests run under.

``"auto"`` (the default) uses UMFPACK when ``petsc4py`` with a working UMFPACK is importable, else SciPy.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class _LuBackend:
    """A host complete-LU backend: factor a matrix, solve with it (or its transpose), refactor in place.

    Concrete backends (:class:`_ScipyLuBackend`, :class:`_PetscUmfpackBackend`) implement the same three
    operations so :class:`LuFactors` is backend-agnostic. All operate on the **raw field-major** matrix —
    no equilibration or reordering, unlike the ILUT.
    """

    n_dofs: int

    def solve(self, rhs: np.ndarray, *, transpose: bool) -> np.ndarray:
        raise NotImplementedError

    def refactor(self, matrix: sp.spmatrix) -> None:
        """Refactor at the given matrix (a fresh factorization; the coupled Jacobian's pattern may grow)."""
        raise NotImplementedError


class _ScipyLuBackend(_LuBackend):
    """SuperLU complete LU via ``scipy.sparse.linalg.splu`` — always available, no symbolic reuse."""

    def __init__(self, matrix: sp.spmatrix) -> None:
        self.n_dofs = matrix.shape[0]
        self._factor(matrix)

    def _factor(self, matrix: sp.spmatrix) -> None:
        # A small diagonal-pivot threshold keeps the indefinite saddle's factorization non-singular
        # without full partial pivoting, matching the ILUT's guard.
        self._lu = spla.splu(matrix.tocsc(), diag_pivot_thresh=0.1)

    def solve(self, rhs: np.ndarray, *, transpose: bool) -> np.ndarray:
        return self._lu.solve(np.asarray(rhs, dtype=np.float64), trans="T" if transpose else "N")

    def refactor(self, matrix: sp.spmatrix) -> None:
        # SuperLU via scipy exposes no symbolic/numeric split, so a refactor is a fresh factorization.
        self._factor(matrix)


class _PetscUmfpackBackend(_LuBackend):
    """UMFPACK (SuiteSparse) complete LU via ``petsc4py`` — the fast path.

    Holds a PETSc ``Mat`` and a ``KSP`` configured as a direct solve (``preonly`` + ``lu`` +
    ``umfpack``). :meth:`refactor` rebuilds them at the new matrix and re-runs the analysis + numeric
    factorization; :meth:`solve` runs the (transpose) triangular solve.

    A refresh rebuilds from scratch rather than reusing UMFPACK's symbolic analysis on a frozen pattern:
    the coupled Jacobian's sparsity **grows as the flow develops** (cross-coupling entries that are
    exactly zero at the cold reference become nonzero), so a fixed-pattern numeric refactor would be both
    wrong (missing the new entries) and a shape error. The full factorization is fast enough (~1 s at
    moderate 2D size) that re-analysing each refresh is cheap; symbolic reuse would save only a small
    fraction of that and is not worth the fixed-pattern assumption.
    """

    def __init__(self, matrix: sp.spmatrix) -> None:
        from petsc4py import PETSc

        self._PETSc = PETSc
        self.n_dofs = matrix.shape[0]
        self._factor(matrix)

    def _factor(self, matrix: sp.spmatrix) -> None:
        PETSc = self._PETSc
        matrix = matrix.tocsr()
        matrix.sort_indices()
        self._mat = PETSc.Mat().createAIJ(
            size=matrix.shape,
            csr=(
                matrix.indptr.astype(np.int32),
                matrix.indices.astype(np.int32),
                matrix.data.copy(),
            ),
        )
        self._mat.assemble()
        self._ksp = PETSc.KSP().create()
        self._ksp.setType("preonly")
        self._ksp.setOperators(self._mat)
        pc = self._ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("umfpack")
        pc.setUp()  # symbolic analysis + numeric factorization
        self._x = self._mat.createVecLeft()
        self._b = self._mat.createVecRight()

    def solve(self, rhs: np.ndarray, *, transpose: bool) -> np.ndarray:
        self._b.setArray(np.asarray(rhs, dtype=np.float64))
        if transpose:
            self._ksp.solveTranspose(self._b, self._x)
        else:
            self._ksp.solve(self._b, self._x)
        return self._x.getArray().copy()

    def refactor(self, matrix: sp.spmatrix) -> None:
        # Rebuild at the new matrix (the pattern may have grown as the flow developed), destroying the old
        # objects so their PETSc memory is released rather than leaked across a long refreshing march.
        self._ksp.destroy()
        self._mat.destroy()
        self._factor(matrix)


def _umfpack_available() -> bool:
    """Whether ``petsc4py`` with a working UMFPACK factor solver can be imported."""
    try:
        from petsc4py import PETSc
    except Exception:
        return False
    try:
        m = PETSc.Mat().createAIJ(
            [1, 1], csr=(np.array([0, 1], np.int32), np.array([0], np.int32), np.array([1.0]))
        )
        m.assemble()
        ksp = PETSc.KSP().create()
        ksp.setOperators(m)
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("umfpack")
        pc.setUp()
        return True
    except Exception:
        return False


def _make_backend(matrix: sp.spmatrix, backend: str) -> _LuBackend:
    if backend == "scipy":
        return _ScipyLuBackend(matrix)
    if backend == "umfpack":
        return _PetscUmfpackBackend(matrix)
    if backend == "auto":
        return _PetscUmfpackBackend(matrix) if _umfpack_available() else _ScipyLuBackend(matrix)
    raise ValueError(
        f"factorize_lu: unknown backend {backend!r} (want 'auto', 'umfpack', or 'scipy')."
    )


class LuFactors:
    """A frozen complete-LU factorization of the coupled Jacobian and its forward/transpose apply.

    A pure host object (no JAX), the complete-LU counterpart of
    :class:`~aquaflux.solve.ilut_preconditioner.IlutFactors`. It applies ``M = A^{-1}`` (or ``M^T``) by a
    triangular solve on the raw field-major vector — no equilibration or reordering, because the complete
    factorization handles the saddle directly. Wraps a pluggable :class:`_LuBackend`.

    Attributes
    ----------
    backend : _LuBackend
        The complete-LU backend (UMFPACK or SciPy SuperLU) doing the factor/solve.
    """

    def __init__(self, backend: _LuBackend) -> None:
        self.backend = backend

    @property
    def n_dofs(self) -> int:
        """Number of degrees of freedom the factorization acts on."""
        return self.backend.n_dofs

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
        return self.backend.solve(residual, transpose=transpose)


def factorize_lu(matrix: sp.spmatrix, *, backend: str = "auto") -> LuFactors:
    """Completely LU-factor an assembled coupled block matrix.

    Parameters
    ----------
    matrix : scipy.sparse matrix
        The assembled field-major coupled Jacobian (already shifted for the pseudo-transient step),
        shape ``(n_fields * n_cells, n_fields * n_cells)``.
    backend : {'auto', 'umfpack', 'scipy'}
        The factorization backend. ``'auto'`` uses UMFPACK (via ``petsc4py``) when available and falls
        back to ``scipy.sparse.linalg.splu`` (SuperLU) otherwise.

    Returns
    -------
    LuFactors
        The frozen factorization.

    Raises
    ------
    ValueError
        If ``matrix`` is not square, or ``backend`` is unknown.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"factorize_lu: matrix must be square, got {matrix.shape}.")
    return LuFactors(_make_backend(matrix, backend))


class MonolithicLuPreconditioner:
    """The coupled complete-LU preconditioner as JAX matvecs, wrapping a frozen :class:`LuFactors`.

    The complete-LU counterpart of
    :class:`~aquaflux.solve.ilut_preconditioner.MonolithicIlutPreconditioner`, with the identical
    interface (:meth:`build`, :meth:`refresh_in_place`, :meth:`matvec`) so it is a drop-in for the coupled
    continuation. Not an :class:`equinox.Module`: the factorization is a host object, held by a caller and
    captured in the ``jax.pure_callback`` closure rather than threaded through the jit as a traced
    argument. Because the factors are frozen (their coefficients ``stop_gradient``-ed by the solver), the
    callback is never differentiated: the forward solve calls ``M`` and the adjoint's transpose solve
    calls ``M^T``, both only in forward evaluations.
    """

    def __init__(self, factors: LuFactors) -> None:
        self.factors = factors

    @staticmethod
    def _materialize(
        matvec: Callable[[jnp.ndarray], jnp.ndarray],
        plan,
        shift_diagonal: np.ndarray,
    ) -> sp.spmatrix:
        from .sparse_jacobian import materialize_block_jacobian

        jacobian = materialize_block_jacobian(matvec, plan)
        return (jacobian + sp.diags(np.asarray(shift_diagonal))).tocsr()

    @classmethod
    def build(
        cls,
        matvec: Callable[[jnp.ndarray], jnp.ndarray],
        plan,
        shift_diagonal: np.ndarray,
        *,
        backend: str = "auto",
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
        backend : {'auto', 'umfpack', 'scipy'}
            The factorization backend (see :func:`factorize_lu`).

        Returns
        -------
        MonolithicLuPreconditioner
            The built preconditioner.
        """
        matrix = cls._materialize(matvec, plan, shift_diagonal)
        return cls(factorize_lu(matrix, backend=backend))

    def refresh_in_place(
        self,
        matvec: Callable[[jnp.ndarray], jnp.ndarray],
        plan,
        shift_diagonal: np.ndarray,
    ) -> None:
        """Re-factor at a developed state and swap the factorization IN PLACE (no new object).

        The arguments are :meth:`build`'s, evaluated at the developed state. The backend re-factors at the
        new matrix (a fresh factorization -- the coupled Jacobian's sparsity grows as the flow develops,
        so the pattern is not fixed; the full factorization is fast enough that this is cheap). Because
        this preconditioner is held as a **static field** of the shift policy and :meth:`matvec` reads
        ``self.factors`` at call time, mutating the factorization here re-preconditions the **same
        compiled** Krylov solve (a compilation cache hit -- no recompile).

        **Forward-march use ONLY — the mutation is impure and must never touch a differentiated path.**
        The adjoint's transpose solve reads the same factorization and would be corrupted by a change
        between its calls; only the eager, non-differentiated march may refresh. The refresh never moves
        the converged root (the shift vanishes there), so it changes only the forward Krylov path.
        """
        matrix = self._materialize(matvec, plan, shift_diagonal)
        self.factors.backend.refactor(matrix)

    def matvec(self, *, transpose: bool = False) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """The preconditioner as a JAX callable ``residual -> M residual`` (or ``M^T``).

        The callback reads ``self.factors`` at call time rather than capturing it, so a
        :meth:`refresh_in_place` between two calls is picked up without rebuilding the callback.

        Parameters
        ----------
        transpose : bool
            Return ``M^T`` (for the adjoint transpose solve) instead of ``M``.

        Returns
        -------
        callable
            A ``jax.pure_callback`` matvec applying the current factorization on the host.
        """
        shape = jax.ShapeDtypeStruct((self.factors.n_dofs,), jnp.float64)

        def apply(residual: jnp.ndarray) -> jnp.ndarray:
            return jax.pure_callback(
                lambda r: self.factors.apply(r, transpose=transpose), shape, residual
            )

        return apply
