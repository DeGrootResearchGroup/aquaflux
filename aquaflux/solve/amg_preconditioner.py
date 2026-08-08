"""A monolithic algebraic-multigrid preconditioner for the coupled saddle-point Newton solve.

The third member of the monolithic-factorization family, beside
:mod:`~aquaflux.solve.ilut_preconditioner` (incomplete-LU) and :mod:`~aquaflux.solve.lu_preconditioner`
(complete LU). Both of those factor the assembled coupled Jacobian directly; their fill is the wall in
three dimensions — a complete LU's is ``O(n^{4/3})`` (out of memory past a few ``10^4`` 3D cells) and even
the threshold-ILU's is several times the operator's own, which for a distance-3 three-dimensional stencil
(hundreds of nonzeros per row) makes the incomplete factorization itself run for minutes. This
preconditioner keeps the heavy fill off the fine grid entirely: it is one **algebraic-multigrid V-cycle**
whose only exact (direct-LU) solve is on the small coarsest grid, so the memory stays bounded and the setup
is a matter of seconds where the incomplete factorization took minutes.

The V-cycle is a **fixed linear operator** — a single application, not an inner Krylov solve — so it is a
drop-in for the same callback-matvec interface the ILUT and LU preconditioners expose (and, being linear
and transposable, it serves the adjoint's transpose solve through the multigrid's own transpose, with no
flexible outer Krylov needed). It preconditions the **equilibrated, cell-major** coupled matrix (the same
conditioning transform the ILUT uses, :func:`~aquaflux.solve.ilut_preconditioner.equilibrate_cell_major`),
which balances the momentum/continuity row scales and interleaves the pressure among the velocity unknowns
so the aggregation and the level smoother see a well-scaled block operator.

The V-cycle is built with PETSc's aggregation multigrid (``PCGAMG``): a **direct LU coarse solve** and a
**stationary zero-fill incomplete-LU level smoother**, over **plain (unsmoothed) aggregation**. Each of
those three is a measured choice against this indefinite saddle rather than a default — fill in the
smoother produces negative pivots as the pseudo-transient shift falls, smoothing the prolongator degrades
the coarse correction (both measured on the coupled backward-facing step), and a Krylov-accelerated
smoother would make the operator nonlinear, which the outer Krylov solve and the adjoint transpose cannot
use. It is a host object, built once off the jit path at a reference state and shift and
applied inside the jitted Krylov solve through ``jax.pure_callback`` — exactly like the ILUT and LU. Because
PETSc supplies the multigrid it is the one member of the family that requires the optional ``petsc``
dependency; there is no pure-SciPy algebraic-multigrid fallback.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .ilut_preconditioner import (
    cell_major_permutation,
    equilibrate_cell_major,
    equilibration_scale,
)
from .refresh_timing import PhaseTimer

# A process-unique options prefix per V-cycle, so several preconditioners' PETSc options never collide.
_prefix_counter = itertools.count()


class _ShellContext:
    """PETSc Python-matrix context delegating the shell matvec to an :class:`AmgVCycle` (see
    :meth:`AmgVCycle._shell_mult`) -- a thin object because petsc4py looks up ``mult`` on the context."""

    def __init__(self, vcycle: AmgVCycle) -> None:
        self._vcycle = vcycle

    def mult(self, mat, x, y) -> None:
        self._vcycle._shell_mult(mat, x, y)


def _petsc():
    """Import ``petsc4py.PETSc`` or raise a clear error naming the optional dependency."""
    try:
        from petsc4py import PETSc
    except Exception as exc:  # pragma: no cover - exercised only without the optional dep
        raise ImportError(
            "The monolithic AMG preconditioner needs PETSc (petsc4py); install the optional "
            "dependency with `pip install aquaflux[petsc]`."
        ) from exc
    return PETSc


class AmgVCycle:
    """A frozen algebraic-multigrid V-cycle on the equilibrated, cell-major coupled matrix.

    A pure host object (PETSc, no JAX) — the algebraic-multigrid counterpart of
    :class:`~aquaflux.solve.ilut_preconditioner.IlutFactors` /
    :class:`~aquaflux.solve.lu_preconditioner.LuFactors`. :meth:`apply` runs **one** V-cycle
    (``M ~= A^{-1}``, a fixed linear operator, not an inner solve); with ``transpose=True`` it runs the
    multigrid's transpose V-cycle (``M^T``), which the adjoint's transpose linear solve needs. The
    equilibration and cell-major reordering are applied around the cycle so the caller works in the raw
    field-major ordering.

    Attributes
    ----------
    scale : np.ndarray
        The symmetric equilibration ``diag(D)``, shape ``(n_dofs,)``.
    perm : np.ndarray
        The cell-major permutation, shape ``(n_dofs,)``.
    """

    def __init__(
        self,
        cell_major: sp.csr_matrix,
        scale: np.ndarray,
        perm: np.ndarray,
        n_fields: int,
        *,
        smoother_fill_levels: int,
        smoother_sweeps: int,
        coarse_eq_limit: int | None = None,
        native: bool = False,
        solve_rtol: float = 1e-8,
        solve_restart: int = 30,
        extra_options: dict | None = None,
    ) -> None:
        self._PETSc = _petsc()
        self._extra_options = extra_options
        self.scale = scale
        self.perm = perm
        self._n_fields = n_fields
        self._smoother_fill_levels = smoother_fill_levels
        self._smoother_sweeps = smoother_sweeps
        self._coarse_eq_limit = coarse_eq_limit
        # When ``native``, an extra PETSc KSP drives the same GAMG V-cycle as a full host solve whose
        # operator is a *shell* over the EXACT Jacobian (:meth:`solve_exact`) -- true Newton at native
        # speed, no per-matvec JAX round-trip. The GAMG hierarchy is still coarsened from the frozen
        # materialized matrix (the preconditioner matrix), which a strong preconditioner tolerates.
        self._native = native
        self._solve_rtol = solve_rtol
        self._solve_restart = solve_restart
        self._cur_matvec = None  # set per solve: the field-major linearized operator ``v -> J v``
        self._cur_shift = None  # set per solve: the field-major pseudo-time shift ``beta d``
        self._prefix = f"aqamg{next(_prefix_counter)}_"
        self._build(cell_major)

    @property
    def n_dofs(self) -> int:
        """Number of degrees of freedom the V-cycle acts on."""
        return self.scale.shape[0]

    @property
    def has_native_solve(self) -> bool:
        """Whether the native host exact-Jacobian forward solve (:meth:`solve_exact`) is available."""
        return self._native

    @property
    def levels(self) -> int:
        """Grids in the multigrid hierarchy, counting the fine grid and the direct coarse solve."""
        return int(self._pc.getMGLevels())

    @property
    def coarse_size(self) -> int:
        """Equations on the coarsest grid — the size of the direct solve at the bottom of the cycle.

        Worth reporting alongside :attr:`levels` whenever an aggregation option is being varied,
        because it says whether the option changed the hierarchy at all. Raising
        ``pc_gamg_coarse_eq_limit`` above a coarse grid that is already smaller than the old limit,
        for instance, produces an identical hierarchy — and without this number that no-op is
        indistinguishable from a real setting that happened to make no difference.
        """
        return int(self._pc.getMGCoarseSolve().getOperators()[0].getSize()[0])

    def destroy(self) -> None:
        """Release the PETSc objects (hierarchy, factors, matrix) this V-cycle holds.

        Garbage collection reaches them eventually, but a caller that builds several V-cycles in a
        loop — a preconditioner study, say — cannot afford to wait: each holds a copy of a
        three-dimensional coupled operator and its factors, and two live at once is enough to exhaust
        a workstation. Calling this makes the release the caller's decision rather than the
        collector's. The object must not be used afterwards.
        """
        if self._native:
            self._ksp.destroy()
            self._shell.destroy()
        self._pc.destroy()
        self._mat.destroy()

    def _build(self, cell_major: sp.csr_matrix) -> None:
        """Assemble the PETSc ``Mat`` and set up the ``PCGAMG`` V-cycle at ``cell_major``."""
        PETSc = self._PETSc
        cell_major = cell_major.tocsr()
        cell_major.sort_indices()
        # The Mat wraps a PERSISTENT copy of the CSR arrays: a refresh (:meth:`refactor`) overwrites
        # ``self._data`` in place (O(nnz) numpy) and re-sets-up the PC, so the aggregation/prolongation
        # and the smoother's ordering are kept -- only the coarse operators and factor values recompute.
        self._indptr = cell_major.indptr.astype(PETSc.IntType)
        self._indices = cell_major.indices.astype(PETSc.IntType)
        self._data = cell_major.data.astype(PETSc.ScalarType).copy()
        # The index arrays of the last matrix whose pattern matched, so a repeat refresh from the same
        # fixed-pattern assembler is settled by identity rather than an O(nnz) comparison.
        self._pattern_seen: tuple[np.ndarray, np.ndarray] | None = None
        self._mat = PETSc.Mat().createAIJWithArrays(
            size=cell_major.shape, csr=(self._indptr, self._indices, self._data)
        )
        self._mat.setBlockSize(
            self._n_fields
        )  # each cell's fields are one block, for GAMG aggregation
        self._mat.assemble()
        self._configure()
        self._x = self._mat.createVecRight()
        self._b = self._mat.createVecLeft()
        # A KSP driving the same GAMG V-cycle as a *native* full solve (GMRES, 1% stop): the whole Krylov
        # loop and the V-cycle applies run on the host, so a march step pays one JAX round-trip rather than
        # one per matvec (JAX-side GMRES with the V-cycle as a per-matvec callback is far slower). The
        # operator is a *shell* over the EXACT Jacobian-vector product (:meth:`_shell_mult`, calling the
        # current linearized ``matvec`` set by :meth:`solve_exact`), so the solve stays true-Newton; the
        # GAMG preconditioner is built from the frozen materialized matrix ``self._mat`` (the Pmat).
        if self._native:
            shell = PETSc.Mat().createPython(self._mat.getSizes(), comm=self._mat.getComm())
            shell.setPythonContext(_ShellContext(self))
            shell.setUp()
            self._shell = shell
            ksp = PETSc.KSP().create()
            ksp.setOptionsPrefix(self._prefix)
            ksp.setOperators(shell, self._mat)  # operator = exact-jvp shell; PC coarsened from Pmat
            ksp.setType("gmres")
            ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
            ksp.setGMRESRestart(self._solve_restart)
            ksp.setTolerances(rtol=self._solve_rtol, atol=1e-50, max_it=self._solve_restart * 20)
            ksp.setPC(self._pc)
            ksp.setUp()
            self._ksp = ksp

    def _shell_mult(self, mat, x, y) -> None:
        """The exact-Jacobian matvec for the native solve's shell operator, in equilibrated cell-major
        coordinates: ``y_cm = D P (J + beta d) P^T D x_cm``, with ``J`` the exact jvp set per solve."""
        xc = np.array(x.array_r)  # the input Vec is locked read-only during MatMult
        w = np.empty(self.n_dofs)
        w[self.perm] = self.scale[self.perm] * xc  # P^T D x -> field-major
        jw = np.asarray(self._cur_matvec(w)) + self._cur_shift * w  # (J + beta d) w, field-major
        y.setArray(self.scale[self.perm] * jw[self.perm])  # D P (J + beta d) w -> cell-major

    def _configure(self) -> None:
        """A smoothed-aggregation V-cycle: direct-LU coarse solve, stationary ILU level smoother."""
        PETSc = self._PETSc
        opts = PETSc.Options()
        p = self._prefix
        # A stationary (Richardson) incomplete-LU smoother -- one level of fill reaches the tolerance on
        # the indefinite saddle where zero-fill stalls; a Krylov-accelerated smoother would make the
        # V-cycle a nonlinear operator, which the outer GMRES and the adjoint transpose cannot use.
        for key, value in {
            "pc_type": "gamg",
            "pc_gamg_type": "agg",
            # PLAIN aggregation, not smoothed. Smoothing the tentative prolongator with a Jacobi step
            # improves interpolation for an M-matrix-like operator and degrades it for a strongly
            # indefinite one, which is what this saddle is. Measured on the coupled backward-facing step
            # with the march's own states, right-hand sides and shift pairing:
            #
            #   state                          smoothed   plain
            #   below the shift floor           22 cyc     9 cyc     (2.4x, and 66x lower true residual)
            #   a step whose line search died    4 cyc     3 cyc
            #   the converged tail               6 cyc     6 cyc     <- ties, which is why it hid
            #
            # The tie at the converged state is the reason this went unnoticed: an easy operator does not
            # discriminate between preconditioners, so a probe taken there reports no difference. Plain
            # aggregation is also marginally cheaper to set up, and it is what makes a DEEP hierarchy
            # usable at all -- with smoothing on, adding levels (via a strength threshold) produces dense
            # coarse operators and a V-cycle that returns NaN.
            "pc_gamg_agg_nsmooths": 0,
            # Keep the aggregation + prolongation and the level-smoother ordering across a refresh
            # (:meth:`refactor`): the operator's sparsity graph is fixed (the graph-coloured Jacobian
            # probe uses a fixed stencil reach; the equilibration and cell-major reorder are value-only),
            # so the coarse space stays valid as the state and shift drift, and only the coarse operators
            # and the incomplete-LU factor values are recomputed -- the bulk of the setup cost is skipped.
            "pc_gamg_reuse_interpolation": True,
            "mg_levels_pc_factor_reuse_ordering": True,
            "mg_levels_pc_factor_reuse_fill": True,
            "mg_coarse_ksp_type": "preonly",
            "mg_coarse_pc_type": "lu",
            "mg_levels_ksp_type": "richardson",
            "mg_levels_ksp_max_it": self._smoother_sweeps,
            "mg_levels_pc_type": "ilu",
            "mg_levels_pc_factor_levels": self._smoother_fill_levels,
        }.items():
            opts[p + key] = value
        # The number of equations at which aggregation stops and the coarsest grid is solved directly (by
        # the ``mg_coarse`` LU above). PETSc's default coarsens to a tiny (~50-equation) coarse grid, whose
        # direct solve captures only the crudest global mode; raising it grows the coarse-level LU so it
        # inverts more of the global coupling exactly -- a stronger V-cycle on the saddle's global pressure
        # mode, at a bounded coarse-solve cost that grows far sub-linearly with the mesh. ``None`` leaves the
        # PETSc default in place.
        if self._coarse_eq_limit is not None:
            opts[p + "pc_gamg_coarse_eq_limit"] = self._coarse_eq_limit
        # Caller overrides last, so a study can vary an aggregation or smoother option without editing
        # the defaults. The defaults above are the measured bundle; this is the seam for the ones that
        # have never been swept on this operator (strength-of-connection, prolongator smoothing).
        for key, value in (self._extra_options or {}).items():
            opts[p + key] = value
        pc = PETSc.PC().create()
        pc.setOptionsPrefix(p)
        pc.setOperators(self._mat)
        pc.setFromOptions()
        pc.setUp()
        self._pc = pc

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        """Apply one V-cycle ``M ~= A^{-1}`` (or its transpose ``M^T``) to a field-major residual.

        Parameters
        ----------
        residual : np.ndarray
            The field-major right-hand side, shape ``(n_dofs,)``.
        transpose : bool
            Apply the transpose V-cycle ``M^T`` (for the adjoint transpose solve) instead of ``M``.

        Returns
        -------
        np.ndarray
            The preconditioned vector, shape ``(n_dofs,)``.
        """
        residual = np.asarray(residual, dtype=np.float64)
        self._b.setArray((self.scale * residual)[self.perm])
        if transpose:
            self._pc.applyTranspose(self._b, self._x)
        else:
            self._pc.apply(self._b, self._x)
        out = np.empty_like(residual)
        out[self.perm] = self._x.getArray()
        return self.scale * out

    def solve_exact(
        self, matvec: Callable[[np.ndarray], np.ndarray], rhs: np.ndarray, shift: np.ndarray
    ) -> np.ndarray:
        """Native host forward solve ``(J + beta d) delta = rhs`` to the 1% stop, with the EXACT ``J``.

        Runs PETSc's own GMRES + GAMG entirely on the host, its operator a shell over the exact
        Jacobian-vector product ``matvec`` (the jvp at the current iterate) plus the pseudo-time shift --
        so the solve is true-Newton and reaches the inexact-Newton tolerance in ~1 iteration (measured),
        without the per-matvec JAX round-trip a JAX-side Krylov with the V-cycle as a callback would pay.
        The GAMG hierarchy is coarsened from the frozen materialized matrix (a strong preconditioner
        tolerates the state/shift drift). Impure (drives host PETSc state), so it is a **forward-only**
        path -- never on a differentiated solve; the adjoint uses the differentiable single-V-cycle
        :meth:`apply`.

        Parameters
        ----------
        matvec : callable
            The field-major exact linearized operator ``v -> J v`` at the current iterate (a jvp), taking
            and returning length-``n_dofs`` arrays.
        rhs : np.ndarray
            The field-major right-hand side (the step solves ``(J + beta d) delta = rhs``), shape ``(n_dofs,)``.
        shift : np.ndarray
            The field-major pseudo-time shift ``beta d``, shape ``(n_dofs,)``.

        Returns
        -------
        np.ndarray
            The correction ``delta``, shape ``(n_dofs,)``.
        """
        if not self._native:
            raise RuntimeError(
                "AmgVCycle.solve_exact needs the native KSP (build with native=True)."
            )
        self._cur_matvec = matvec
        self._cur_shift = np.asarray(shift, dtype=np.float64)
        rhs = np.asarray(rhs, dtype=np.float64)
        self._b.setArray((self.scale * rhs)[self.perm])  # D P rhs -> equilibrated cell-major
        self._x.set(0.0)
        self._ksp.solve(self._b, self._x)
        out = np.empty_like(rhs)
        out[self.perm] = self._x.getArray()
        return self.scale * out  # P^T D solution -> field-major delta

    def refactor(self, cell_major: sp.csr_matrix, scale: np.ndarray, perm: np.ndarray) -> None:
        """Refresh the V-cycle at a new (developed-state, new-shift) matrix, reusing the coarse space.

        A β-tracking march re-factors every step. Because the graph-coloured Jacobian probe uses a
        **fixed** stencil reach and the equilibration + cell-major reorder are value-only, the operator's
        sparsity graph is **identical** across refreshes -- only its values change. So the refresh
        overwrites the persistent CSR values in place (O(nnz) numpy) and re-sets-up the *same* PC with the
        ``pc_gamg_reuse_interpolation`` / smoother-``reuse_ordering`` flags (:meth:`_configure`): the
        aggregation, prolongation and factor orderings are kept, and only the Galerkin coarse operators and
        the incomplete-LU factor values are recomputed. That is markedly cheaper than rebuilding the whole
        hierarchy, which dominates the refresh cost.

        If the sparsity pattern ever differs (it should not, given the fixed stencil), it falls back to a
        full rebuild. The native exact-solve KSP (:attr:`_native`) also takes the full rebuild -- it is the
        deferred experimental path and shares the ``Mat`` with its shell operator.
        """
        self.scale = scale
        self.perm = perm
        cell_major = cell_major.tocsr()
        cell_major.sort_indices()
        if not self._matches_pattern(cell_major):
            self.destroy()
            self._build(cell_major)
            return
        # In-place value refresh: the Mat wraps ``self._data``, so overwriting it updates the operator
        # without re-validating the pattern; re-setting-up reuses the aggregation/ordering.
        self._data[:] = cell_major.data.astype(self._PETSc.ScalarType)
        self._mat.assemble()
        self._pc.setOperators(self._mat)
        self._pc.setUp()

    def _matches_pattern(self, cell_major: sp.csr_matrix) -> bool:
        """Whether ``cell_major`` has the sparsity the persistent ``Mat`` was built on.

        Comparing the index arrays is ``O(nnz)`` and runs on every refresh, which on a three-dimensional
        coupled Jacobian is tens of millions of elements compared twice to re-confirm something that is
        fixed by construction. A caller that assembles through a precomputed fixed-pattern structure hands
        back the **same** index arrays every time, so their identity settles it; the element-wise
        comparison stays as the fallback for a caller that does not.
        """
        if self._native:
            return False
        if (
            self._pattern_seen is not None
            and cell_major.indptr is self._pattern_seen[0]
            and cell_major.indices is self._pattern_seen[1]
        ):
            return True
        matches = (
            cell_major.data.shape[0] == self._data.shape[0]
            and np.array_equal(cell_major.indptr, self._indptr)
            and np.array_equal(cell_major.indices, self._indices)
        )
        if matches:
            self._pattern_seen = (cell_major.indptr, cell_major.indices)
        return matches


def build_amg_vcycle(
    matrix: sp.spmatrix,
    n_fields: int,
    *,
    smoother_fill_levels: int = 1,
    smoother_sweeps: int = 2,
    coarse_eq_limit: int | None = None,
    native: bool = False,
    extra_options: dict | None = None,
) -> AmgVCycle:
    """Equilibrate + reorder a coupled block matrix and build a multigrid V-cycle preconditioner for it.

    Parameters
    ----------
    matrix : scipy.sparse matrix
        The assembled field-major coupled Jacobian (already shifted for the pseudo-transient step),
        shape ``(n_fields * n_cells, n_fields * n_cells)``.
    n_fields : int
        Degrees of freedom per cell.
    smoother_fill_levels : int
        Incomplete-LU fill levels of the stationary level smoother (``1`` = ILU(1); the indefinite saddle
        stalls at ``0``).
    smoother_sweeps : int
        Richardson sweeps of the level smoother per V-cycle visit. Two is the default: on the
        low-shift operator the pseudo-transient march spends its tail in, a second smoother sweep
        roughly quarters the outer Krylov iteration count (measured ~4× on the `bfs3d` coupled
        Jacobian at a low shift, ~2× the whole solve there), for one extra cheap incomplete-LU
        back-solve per visit — a large net saving where each outer iteration pays a full
        Jacobian-vector product.
    coarse_eq_limit : int or None
        The equation count at which aggregation stops and the coarsest grid is solved directly. ``None``
        (default) keeps PETSc's default (~50), a tiny coarse grid whose direct LU captures only the crudest
        global mode; a larger value grows the coarse-level LU so it inverts more of the saddle's global
        pressure coupling exactly — a stronger V-cycle at a bounded coarse-solve cost.
    native : bool
        Also assemble the native host exact-Jacobian forward solve (:meth:`AmgVCycle.solve_exact`), whose
        operator is a shell over the exact jvp supplied per solve. ``False`` builds the single-V-cycle
        apply only (the frozen preconditioner and adjoint path).

    Returns
    -------
    AmgVCycle
        The frozen V-cycle.
    """
    cell_major, scale, perm = equilibrate_cell_major(matrix, n_fields)
    return AmgVCycle(
        cell_major,
        scale,
        perm,
        n_fields,
        smoother_fill_levels=smoother_fill_levels,
        smoother_sweeps=smoother_sweeps,
        coarse_eq_limit=coarse_eq_limit,
        native=native,
        extra_options=extra_options,
    )


class ShiftedCellMajorOperator:
    """Assemble the equilibrated cell-major operator ``D P (J + diag(shift)) Pᵀ D`` for a FIXED pattern.

    Every refresh of the V-cycle re-forms the same three transforms of the materialized Jacobian: add the
    pseudo-transient shift to the diagonal, symmetrically equilibrate by the square-root diagonal, and
    reorder to cell-major. Written with generic sparse operations that is a diagonal add, two sparse
    products and two fancy-index permutations, each allocating and re-sorting a matrix the size of the
    coupled Jacobian — work that is repeated identically every refresh, because for a fixed stencil reach
    the sparsity **pattern never changes** and only the values do.

    This precomputes the pattern-dependent part once — which base nonzero feeds each output nonzero, where
    the diagonal entries sit, and the cell-major CSR structure — so a refresh is one gather plus an
    ``O(n_dofs)`` diagonal add plus a symmetric scale, written into a **preallocated** buffer. The scaling
    rule itself is :func:`~aquaflux.solve.ilut_preconditioner.equilibration_scale`, shared with the generic
    :func:`~aquaflux.solve.ilut_preconditioner.equilibrate_cell_major` that serves an arbitrary matrix;
    only the data movement differs, and the two agree to the last bit (pinned by a unit test).

    The scale and permutation are applied to vectors by the V-cycle, so :meth:`assemble` returns them
    alongside the matrix exactly as ``equilibrate_cell_major`` does.

    .. warning::
       The returned matrix **aliases a reused buffer** and is overwritten by the next :meth:`assemble`.
       It is meant to be consumed immediately (handed to a factorization, which takes its own copy), not
       retained.

    Parameters
    ----------
    indptr, indices : np.ndarray
        The fixed **field-major** CSR structure of the materialized Jacobian, shapes ``(n_dofs + 1,)`` and
        ``(nnz,)``. Degree of freedom ``(cell i, field f)`` is at ``f * n_cells + i``.
    n_fields : int
        Degrees of freedom per cell.

    Raises
    ------
    ValueError
        If the size is not a multiple of ``n_fields``, or the pattern is missing a diagonal entry (the
        shift has nowhere to go, so the pattern is not the full block-stencil one this needs).
    """

    def __init__(self, indptr: np.ndarray, indices: np.ndarray, n_fields: int) -> None:
        n_dofs = int(indptr.shape[0]) - 1
        if n_dofs % n_fields != 0:
            raise ValueError(
                f"ShiftedCellMajorOperator: {n_dofs} degrees of freedom is not a multiple of "
                f"n_fields={n_fields}."
            )
        self._shape = (n_dofs, n_dofs)
        self._perm = cell_major_permutation(n_dofs // n_fields, n_fields)
        # perm maps cell-major -> field-major; its inverse relabels a field-major index to cell-major.
        inverse = np.empty(n_dofs, dtype=np.int64)
        inverse[self._perm] = np.arange(n_dofs)
        base_rows = np.repeat(np.arange(n_dofs, dtype=np.int64), np.diff(indptr))
        rows = inverse[base_rows]
        cols = inverse[np.asarray(indices, dtype=np.int64)]
        # Sorting the relabelled coordinates by (row, col) IS the permuted matrix's CSR order, and the
        # sort permutation is the gather: output position t reads base value `source[t]`.
        order = np.lexsort((cols, rows))
        self._source = _smallest_index(order)
        self._indices = _smallest_index(cols[order])
        self._indptr = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=n_dofs)))).astype(
            self._indices.dtype
        )
        # Row counts drive the chunked row scaling below; the full per-nonzero row array would be another
        # array the size of the Jacobian's values, which is what this class exists to avoid allocating.
        self._counts = np.diff(self._indptr)
        self._diagonal = np.flatnonzero(rows[order] == self._indices)
        if self._diagonal.shape[0] != n_dofs:
            raise ValueError(
                "ShiftedCellMajorOperator: the pattern is missing a diagonal entry "
                f"({self._diagonal.shape[0]} of {n_dofs}); the pseudo-transient shift has nowhere to go."
            )
        self._data = np.zeros(order.shape[0], dtype=np.float64)
        self._matrix = sp.csr_matrix(
            (self._data, self._indices, self._indptr), shape=self._shape, copy=False
        )
        # scipy may re-type the index arrays on construction; keep writing into whatever it actually holds.
        self._data = self._matrix.data
        self._matrix.has_sorted_indices = True
        self._chunks = _row_chunks(self._indptr, _SCALE_CHUNK_NNZ)

    def assemble(
        self, values: np.ndarray, shift_diagonal: np.ndarray
    ) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        """The equilibrated cell-major matrix, its equilibration factor and its permutation.

        Parameters
        ----------
        values : np.ndarray
            The materialized Jacobian's CSR values in the fixed field-major pattern, shape ``(nnz,)``.
        shift_diagonal : np.ndarray
            The pseudo-transient shift ``β d`` in field-major order, shape ``(n_dofs,)``.

        Returns
        -------
        matrix : scipy.sparse.csr_matrix
            The equilibrated cell-major operator. **Reused and overwritten by the next call.**
        scale : np.ndarray
            The equilibration ``diag(D)`` in **field-major** order, shape ``(n_dofs,)``.
        perm : np.ndarray
            The cell-major permutation, shape ``(n_dofs,)``.
        """
        data = self._data
        np.take(np.asarray(values, dtype=np.float64), self._source, out=data)
        shift = np.asarray(shift_diagonal, dtype=np.float64)[self._perm]
        data[self._diagonal] += shift
        cell_major_scale = equilibration_scale(data[self._diagonal])
        # D A D, chunked over rows: the row factor needs one entry per nonzero and the column factor a
        # gather of the same length, so doing it whole would allocate two more Jacobian-sized temporaries.
        for start, stop in self._chunks:
            lo, hi = int(self._indptr[start]), int(self._indptr[stop])
            block = data[lo:hi]
            block *= np.repeat(cell_major_scale[start:stop], self._counts[start:stop])
            block *= cell_major_scale[self._indices[lo:hi]]
        scale = np.empty_like(cell_major_scale)
        scale[self._perm] = cell_major_scale
        return self._matrix, scale, self._perm


#: Target nonzeros per row-chunk in the symmetric scaling of :meth:`ShiftedCellMajorOperator.assemble`.
#: Bounds the transient allocation there to a few megabytes rather than the size of the Jacobian's
#: values; small enough to stay in cache, large enough that the per-chunk NumPy overhead is negligible.
_SCALE_CHUNK_NNZ = 1 << 20


def _smallest_index(values: np.ndarray) -> np.ndarray:
    """``values`` as the narrowest signed integer type that holds them (32-bit where it fits).

    These index arrays are as long as the Jacobian's nonzeros -- tens of millions on a three-dimensional
    coupled mesh -- and are held for the life of the preconditioner, so halving their width is worth the
    check.
    """
    values = np.asarray(values)
    return values.astype(np.int32) if values.size == 0 or values.max() < 2**31 else values


def _row_chunks(indptr: np.ndarray, target_nnz: int) -> tuple[tuple[int, int], ...]:
    """Split ``[0, n_rows)`` into ``(start, stop)`` row ranges of roughly ``target_nnz`` nonzeros each.

    Ranges are cut on row boundaries so each chunk is a contiguous slice of the CSR values, and every row
    lands in exactly one chunk. A row wider than ``target_nnz`` simply forms its own oversized chunk.
    """
    n_rows = int(indptr.shape[0]) - 1
    if n_rows == 0:
        return ()
    edges = np.searchsorted(indptr, np.arange(0, int(indptr[-1]), target_nnz), side="right") - 1
    bounds = np.unique(np.concatenate(([0], np.maximum(edges, 0), [n_rows])))
    return tuple((int(a), int(b)) for a, b in itertools.pairwise(bounds) if b > a)


class MonolithicAmgPreconditioner:
    """The coupled algebraic-multigrid preconditioner as JAX matvecs, wrapping a frozen :class:`AmgVCycle`.

    The algebraic-multigrid counterpart of
    :class:`~aquaflux.solve.ilut_preconditioner.MonolithicIlutPreconditioner` /
    :class:`~aquaflux.solve.lu_preconditioner.MonolithicLuPreconditioner`, with the identical interface
    (:meth:`build`, :meth:`refresh_in_place`, :meth:`matvec`) so it is a drop-in for the coupled
    continuation's :class:`~aquaflux.turbulence.MonolithicFactorShiftPolicy`. Not an
    :class:`equinox.Module`: the V-cycle is a host PETSc object, held by a caller and captured in the
    ``jax.pure_callback`` closure rather than threaded through the jit as a traced argument. Because the
    V-cycle is frozen (its coefficients ``stop_gradient``-ed by the solver), the callback is never
    differentiated: the forward solve calls ``M`` and the adjoint's transpose solve calls ``M^T``, both only
    in forward evaluations.
    """

    def __init__(
        self,
        vcycle: AmgVCycle,
        residual_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        jacobian_no_shift: sp.csr_matrix | None = None,
        n_fields: int | None = None,
        assembler: ShiftedCellMajorOperator | None = None,
    ) -> None:
        self.factors = vcycle
        self._residual_fn = residual_fn
        # The fixed-pattern shift/equilibrate/reorder assembler, present exactly when the materialize ran
        # on a precomputed ``structure`` (which is what guarantees the pattern is the same every refresh).
        # ``None`` falls back to the generic sparse path, which works for any pattern.
        self._assembler = assembler
        # The materialized jvp Jacobian *without* the pseudo-transient shift, cached so a β-only refresh
        # (:meth:`refresh_shift_in_place`) can re-add a new ``β d`` diagonal without re-running the coloured jvp probe.
        self._jacobian_no_shift = jacobian_no_shift
        self._n_fields = n_fields
        # A jitted jvp ``(phi, w) -> J(phi) w`` for the native exact solve's shell operator, called eagerly
        # on the host inside the solve's pure_callback (linearizing at the current iterate ``phi``).
        self._jvp = (
            jax.jit(lambda phi, w: jax.jvp(residual_fn, (phi,), (w,))[1])
            if residual_fn is not None
            else None
        )

    @staticmethod
    def _materialize_jacobian(
        matvec: Callable[[jnp.ndarray], jnp.ndarray],
        colouring,
        n_fields: int,
        batched_matvec: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        probe_batch_size: int | None = None,
        structure: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> sp.csr_matrix:
        """The coupled Jacobian **without** the shift, from the graph-coloured jvp probe (one jvp per
        (colour, field) -- the expensive part of a refresh; e.g. ~670 probes on a 23k-cell reach-3 bfs3d
        mesh). ``batched_matvec`` (built once, reused) runs the probes as a few batched passes rather than a
        per-probe loop (~1.6x on that mesh); ``probe_batch_size`` chunks the batch for memory. ``structure``
        (from ``block_stencil_gather_map``, built once) de-compresses by a single gather into the fixed
        full-pattern CSR instead of a scatter loop + re-sort (multigrid-only -- it keeps explicit zeros)."""
        from .sparse_jacobian import materialize_block_jacobian

        return materialize_block_jacobian(
            matvec,
            colouring,
            n_fields,
            batched_matvec=batched_matvec,
            probe_batch_size=probe_batch_size,
            structure=structure,
        ).tocsr()

    @staticmethod
    def _shifted(jacobian_no_shift: sp.csr_matrix, shift_diagonal: np.ndarray) -> sp.csr_matrix:
        """Add the pseudo-transient shift ``β d`` to the Jacobian's diagonal (cheap -- ``O(nnz)`` numpy)."""
        return (jacobian_no_shift + sp.diags(np.asarray(shift_diagonal))).tocsr()

    @staticmethod
    def _assembler_for(
        structure: tuple[np.ndarray, np.ndarray, np.ndarray] | None, n_fields: int
    ) -> ShiftedCellMajorOperator | None:
        """The fixed-pattern assembler for this materialize, or ``None`` for the generic sparse path.

        A precomputed ``structure`` is precisely the guarantee the pattern is identical every refresh
        (that is what :func:`~aquaflux.solve.sparse_jacobian.block_stencil_gather_map` provides, explicit
        zeros and all), so it is also the condition under which the pattern-dependent work can be hoisted
        out of the refresh. Without it the materialize eliminates numerical zeros and the pattern may move
        with the state, so each refresh must re-derive it.
        """
        if structure is None:
            return None
        indptr, indices, _ = structure
        return ShiftedCellMajorOperator(indptr, indices, n_fields)

    def _cell_major(
        self, jacobian: sp.csr_matrix, shift_diagonal: np.ndarray, n_fields: int
    ) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        """The equilibrated cell-major operator ``D P (J + diag(shift)) Pᵀ D`` the V-cycle is built on.

        Routes to the precomputed :class:`ShiftedCellMajorOperator` when the pattern is fixed, and to the
        generic diagonal-add + :func:`~aquaflux.solve.ilut_preconditioner.equilibrate_cell_major` otherwise.
        The two paths agree to the last bit; only how much work is repeated per refresh differs.
        """
        if self._assembler is not None:
            return self._assembler.assemble(jacobian.data, shift_diagonal)
        return equilibrate_cell_major(self._shifted(jacobian, shift_diagonal), n_fields)

    @classmethod
    def build(
        cls,
        matvec: Callable[[jnp.ndarray], jnp.ndarray],
        colouring,
        n_fields: int,
        shift_diagonal: np.ndarray,
        *,
        native: bool = False,
        residual_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        smoother_fill_levels: int = 1,
        smoother_sweeps: int = 2,
        coarse_eq_limit: int | None = None,
        batched_matvec: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        probe_batch_size: int | None = None,
        structure: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        extra_options: dict | None = None,
    ) -> MonolithicAmgPreconditioner:
        """Materialize the shifted coupled Jacobian and build a V-cycle preconditioner for it, off the jit path.

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
            The pseudo-transient shift added to the Jacobian's diagonal, shape ``(n_fields * n,)`` — the
            same block-diagonal shift the step solves against (velocity/scalar shifts, pressure zero).
        native : bool
            Enable the native host exact-Jacobian forward solve (:meth:`exact_solve`); ``residual_fn`` is
            then required (the shell operator linearizes it at each iterate).
        residual_fn : callable, optional
            The steady residual ``phi -> R(phi)`` the native solve linearizes for its exact-Jacobian shell.
        smoother_fill_levels, smoother_sweeps : int
            The level-smoother controls (see :func:`build_amg_vcycle`).
        coarse_eq_limit : int or None
            The coarse-grid direct-solve size (see :func:`build_amg_vcycle`). ``None`` keeps PETSc's default.
        batched_matvec : callable, optional
            A batched form of ``matvec`` (``(k, nf) -> (k, nf)``), built once and reused, so the coloured
            probes run as a few batched passes instead of a per-probe loop (a pure materialization speedup).
        probe_batch_size : int or None
            The batched-probe chunk size (simultaneous tangents), to bound peak memory; ``None`` runs all
            probes in one batch.
        structure : tuple of np.ndarray or None
            A precomputed ``(indptr, indices, gather_map)`` (:func:`~aquaflux.solve.sparse_jacobian.block_stencil_gather_map`)
            so the materialize de-compresses by one gather into the fixed full-pattern CSR instead of a
            scatter loop + re-sort. Built once and reused across refreshes.

        Returns
        -------
        MonolithicAmgPreconditioner
            The built preconditioner.
        """
        jacobian = cls._materialize_jacobian(
            matvec, colouring, n_fields, batched_matvec, probe_batch_size, structure
        )
        assembler = cls._assembler_for(structure, n_fields)
        # `build_amg_vcycle` equilibrates and reorders internally, so the build takes the generic path
        # regardless; the assembler is constructed here so every later refresh has it.
        matrix = cls._shifted(jacobian, shift_diagonal)
        return cls(
            build_amg_vcycle(
                matrix,
                n_fields,
                smoother_fill_levels=smoother_fill_levels,
                smoother_sweeps=smoother_sweeps,
                coarse_eq_limit=coarse_eq_limit,
                native=native,
                extra_options=extra_options,
            ),
            residual_fn=residual_fn,
            jacobian_no_shift=jacobian,
            n_fields=n_fields,
            assembler=assembler,
        )

    def refresh_in_place(
        self,
        matvec: Callable[[jnp.ndarray], jnp.ndarray],
        colouring,
        n_fields: int,
        shift_diagonal: np.ndarray,
        *,
        smoother_fill_levels: int = 1,
        smoother_sweeps: int = 2,
        batched_matvec: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        probe_batch_size: int | None = None,
        structure: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[tuple[str, float], ...]:
        """Rebuild the V-cycle at a developed state and swap it IN PLACE (no new object).

        The arguments are :meth:`build`'s, evaluated at the developed state. Because this preconditioner is
        held as a **static field** of the shift policy and :meth:`matvec` reads ``self.factors`` at call
        time, mutating the V-cycle here re-preconditions the **same compiled** Krylov solve (a compilation
        cache hit -- no recompile).

        **Forward-march use ONLY — the mutation is impure and must never touch a differentiated path.** The
        adjoint's transpose solve reads the same V-cycle and would be corrupted by a change between its
        calls; only the eager, non-differentiated march may refresh. The refresh never moves the converged
        root (the shift vanishes there), so it changes only the forward Krylov path.

        Returns
        -------
        tuple of (str, float)
            ``("probe", s), ("assemble", s), ("refactor", s)`` — the coloured jvp probe, the shift +
            equilibration + cell-major reorder, and the multigrid re-setup. Reported so a march log can
            say *where* a refresh spent its time; the three call for different fixes and are
            indistinguishable in an aggregate.
        """
        del smoother_fill_levels, smoother_sweeps  # the smoother config is fixed at build
        timer = PhaseTimer()
        self._jacobian_no_shift = self._materialize_jacobian(
            matvec, colouring, n_fields, batched_matvec, probe_batch_size, structure
        )
        timer.lap("probe")
        self._n_fields = n_fields
        if self._assembler is None:
            self._assembler = self._assembler_for(structure, n_fields)
        cell_major, scale, perm = self._cell_major(
            self._jacobian_no_shift, shift_diagonal, n_fields
        )
        timer.lap("assemble")
        self.factors.refactor(cell_major, scale, perm)
        timer.lap("refactor")
        return timer.phases()

    def refresh_shift_in_place(self, shift_diagonal: np.ndarray) -> tuple[tuple[str, float], ...]:
        """Re-preconditioner at a new shift ``β d`` REUSING the frozen Jacobian — no re-materialization.

        The operator is ``J(φ) + β d``, and the pseudo-transient shift ``β d`` touches only the **diagonal**.
        So tracking a moving ``β`` (and the cheap per-cell shift ``d``) needs only to re-add the new diagonal
        to the **cached** Jacobian and re-factor — it does **not** need the coloured-probe materialization of ``J``
        that :meth:`refresh_in_place` pays. The probe is the dominant refresh cost — measured on a
        three-dimensional coupled march the two branches differ by roughly a factor of three, and that
        difference *is* the probe — so a shift-only refresh is several times cheaper. The Jacobian is held frozen at the last full
        :meth:`build` / :meth:`refresh_in_place`, so ``J``'s *state* drift is not tracked here — pair frequent
        shift-only refreshes with an occasional full refresh (a state-staleness trigger) to catch that.

        **Forward-march use ONLY**, exactly as :meth:`refresh_in_place`: the mutation is impure and must never
        touch a differentiated path. Raises if no Jacobian has been materialized yet (call :meth:`build` or
        :meth:`refresh_in_place` first).

        Parameters
        ----------
        shift_diagonal : np.ndarray
            The new pseudo-transient shift ``β d``, shape ``(n_fields * n,)`` — added to the cached Jacobian's
            diagonal.

        Returns
        -------
        tuple of (str, float)
            ``("assemble", s), ("refactor", s)`` — this branch runs no probe, which is exactly what makes
            it the cheap one.
        """
        if self._jacobian_no_shift is None or self._n_fields is None:
            raise RuntimeError(
                "refresh_shift_in_place needs a cached Jacobian; call build() or refresh_in_place() first."
            )
        timer = PhaseTimer()
        cell_major, scale, perm = self._cell_major(
            self._jacobian_no_shift, shift_diagonal, self._n_fields
        )
        timer.lap("assemble")
        self.factors.refactor(cell_major, scale, perm)
        timer.lap("refactor")
        return timer.phases()

    @property
    def has_native_solve(self) -> bool:
        """Whether the native host exact-Jacobian forward solve is available (built with ``native=True``)."""
        return self.factors.has_native_solve and self._jvp is not None

    def destroy(self) -> None:
        """Release the V-cycle's PETSc objects and the cached Jacobian (see :meth:`AmgVCycle.destroy`)."""
        self.factors.destroy()
        self._jacobian_no_shift = None

    @property
    def is_exact_native(self) -> bool:
        """Marks this preconditioner so the pseudo-transient step applies the native full solve directly
        (see :func:`aquaflux.solve.continuation._shifted_solve`) instead of a JAX-side Krylov iteration."""
        return self.has_native_solve

    def exact_solve(self, phi: jnp.ndarray, rhs: jnp.ndarray, shift: jnp.ndarray) -> jnp.ndarray:
        """The full inexact-Newton correction ``delta`` solving ``(J(phi) + shift) delta = rhs`` to ~1%.

        Runs PETSc's GMRES + native GAMG V-cycle entirely on the host, its operator a shell over the EXACT
        jvp linearized at ``phi`` (so the solve is true-Newton) -- reaching the tolerance in ~1 iteration
        (measured) without the per-matvec JAX round-trip a JAX-side Krylov with the V-cycle as a callback
        would pay. One JAX ``pure_callback`` per step, carrying ``phi``/``rhs``/``shift`` in and ``delta``
        out; the exact jvp is evaluated eagerly inside the callback. Forward-only (drives host PETSc state);
        the adjoint uses the differentiable single-V-cycle :meth:`matvec` transpose.
        """
        shape = jax.ShapeDtypeStruct((self.factors.n_dofs,), jnp.float64)

        def host(rhs_np, phi_np, shift_np):
            phi_j = jnp.asarray(phi_np)
            matvec = lambda w: np.asarray(self._jvp(phi_j, jnp.asarray(w)))  # noqa: E731
            return self.factors.solve_exact(matvec, rhs_np, shift_np)

        return jax.pure_callback(host, shape, rhs, phi, shift)

    def matvec(self, *, transpose: bool = False) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """The preconditioner as a JAX callable ``residual -> M residual`` (or ``M^T``).

        The callback reads ``self.factors`` at call time rather than capturing it, so a
        :meth:`refresh_in_place` between two calls is picked up without rebuilding the callback.

        Parameters
        ----------
        transpose : bool
            Return the transpose V-cycle ``M^T`` (for the adjoint transpose solve) instead of ``M``.

        Returns
        -------
        callable
            A ``jax.pure_callback`` matvec applying the current V-cycle on the host.
        """
        shape = jax.ShapeDtypeStruct((self.factors.n_dofs,), jnp.float64)

        def apply(residual: jnp.ndarray) -> jnp.ndarray:
            return jax.pure_callback(
                lambda r: self.factors.apply(r, transpose=transpose), shape, residual
            )

        return apply
