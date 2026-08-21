"""A host V-cycle over the traced hierarchy, smoothed by an incomplete factorization.

**One aggregation, two applies.** The coarsening in :mod:`~aquaflux.solve.multigrid` is already a host
computation in ``scipy``; only the *apply* is traced. That leaves room for a second apply over the very
same hierarchy: this one runs on the host in ``numpy`` and relaxes each level with an incomplete-LU
factorization, where :mod:`~aquaflux.solve.saddle_multigrid` and
:mod:`~aquaflux.solve.field_split` run a traced cycle relaxed by SIMPLE or by Jacobi sweeps.

The two are for different machines, and the reason is the smoother rather than the hierarchy:

* an **incomplete-LU** sweep is a sequential triangular solve. It is the stronger smoother on this
  operator class and the right choice on a CPU, and it is the one piece of the coupled solver that does
  not move to an accelerator;
* a **SIMPLE or Jacobi** sweep is diagonal scalings and sparse matrix--vector products, which is what an
  accelerator wants, and is what the traced cycle exists for.

Before this, choosing the incomplete-LU smoother meant taking a *second, independent* multigrid: a host
library's own aggregation, its own coarse space, its own refresh path, tuned by its own options. Two
hierarchies over one operator is two things to keep true at once, and they drifted -- the traced one grew
a strength threshold, a level cap and a refit path that the other expressed differently or not at all.
Here the hierarchy is the traced one in both cases, so a coarsening improvement reaches the CPU path and
the accelerator path together, and only the relaxation differs.

**The transpose is built, not borrowed, and that is the delicate part.** A V-cycle is not symmetric
unless its smoother is, and an incomplete factorization is not: ``M^T`` is the same recursion with the
operator, the coarse solve and the smoother each transposed, and the pre- and post-smoothing exchanged.
:meth:`IluSmoothedInverse.apply` implements exactly that, and the adjoint identity
``<y, M x> == <M^T y, x>`` is asserted in the unit tests rather than argued for here -- it is the one
property whose failure would leave the forward solve healthy and every gradient wrong.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from .frozen_operator import equilibrate_ordered
from .ilu0 import Ilu0
from .multigrid import SmoothedHierarchy, build_convection_hierarchy
from .ordering import CellMajor, EliminationOrdering

__all__ = ["IluSmoothedInverse", "ilu_smoothed_inverse"]


class _HostLevel:
    """One level of the hierarchy in host form: its operator, its prolongation, its smoother.

    Built by converting the traced level back to ``scipy`` rather than by coarsening again, so the two
    applies are guaranteed to be running over the *same* hierarchy — a second coarsening would be a
    second thing to keep true.
    """

    def __init__(
        self,
        operator: sp.csr_matrix,
        prolongation: sp.csr_matrix | None,
        smoother: _LevelSmoother | None,
        coarse_inverse: np.ndarray | None,
    ) -> None:
        self.operator = operator
        self.operator_t = operator.T.tocsr() if operator is not None else None
        self.prolongation = prolongation
        self.smoother = smoother
        self.coarse_inverse = coarse_inverse


class _LevelSmoother:
    """An incomplete-LU relaxation for one level, applied as a fixed number of stationary sweeps.

    Stationary (Richardson) rather than Krylov-accelerated, and that is a correctness requirement rather
    than a tuning choice: an inner Krylov method makes the cycle a **nonlinear** operator, which the
    non-flexible outer solve and the transposed adjoint both forbid.

    The factorization is :class:`~aquaflux.solve.ilu0.Ilu0` -- zero fill, the operator's own pattern.
    A refresh re-runs only its numeric pass over the same index arrays, which is what makes a smoother
    affordable on a path that re-fits tens of times per march, and its transposed solve reads the same
    stored factor so the adjoint needs no second one.
    """

    def __init__(
        self, operator: sp.csr_matrix, n_fields: int, sweeps: int, ordering: EliminationOrdering
    ):
        self.sweeps = sweeps
        # ⚠️ EQUILIBRATE AND REORDER BEFORE FACTORIZING -- this is not tidying, it is what makes an
        # incomplete factorization possible on this operator at all. Factorized raw and field-major, the
        # saddle's weak continuity diagonal produces an exactly singular factor and the smoother raises
        # outright (observed on the `bfs3d` flow block). Two effects, both needed: symmetric
        # equilibration balances momentum rows against continuity rows, which differ by more than an
        # order of magnitude; the interleave puts each cell's own fields adjacent, so the factorization's
        # fill stays local to a cell instead of spanning a whole field block.
        #
        # The host AMG this path replaces was handed exactly the same preprocessing, so it was never a
        # property of that library -- it was a step performed on the way in, and doing it here is what
        # moves the hierarchy without moving the requirement.
        #
        # The ORDER the cells are visited in is injected rather than fixed, because at zero fill it
        # decides which entries the elimination discards and is worth as much as any other single choice
        # on this operator (see `aquaflux.solve.ordering`).
        self.operator, self.scale, self.perm = equilibrate_ordered(operator, n_fields, ordering)
        # ZERO-FILL, and that is the whole point. A drop-tolerance factorization is a different
        # algorithm: it chooses which entries to keep by magnitude within a memory budget, so it
        # discards pattern entries and keeps fill ones. Measured on this project's flow block that
        # leaves a factor whose entries reach 1e+23 and an applied residual of 1e+38 -- at the SAME
        # nonzero count. ILU(0) keeps the operator's own pattern and cannot do that.
        self.factors = Ilu0(self.operator)

    def _apply_inverse(self, residual: np.ndarray, transpose: bool) -> np.ndarray:
        """``M^-1 r`` for a FIELD-MAJOR vector, through the equilibrated cell-major factorization.

        The scaling and the permutation are undone around the solve, so the caller never sees the
        reordered space and the smoother composes with the rest of the cycle unchanged.
        """
        out = np.empty_like(residual)
        out[self.perm] = self.factors.solve((self.scale * residual)[self.perm], transpose)
        return self.scale * out

    def sweep(self, operator, b: np.ndarray, x: np.ndarray, transpose: bool) -> np.ndarray:
        """``sweeps`` stationary corrections ``x <- x + M^-1 (b - A x)``, transposed on request."""
        for _ in range(self.sweeps):
            x = x + self._apply_inverse(b - operator @ x, transpose)
        return x

    def sweep_from_zero(self, operator, b: np.ndarray, transpose: bool) -> np.ndarray:
        """The same relaxation from a zero iterate, with the first sweep's matvec peeled off.

        At ``x = 0`` the residual ``b - A x`` is exactly ``b``, so that application of the level
        operator computes a known answer at full price; the pre-smooth always starts here, so it is
        charged at every level of every cycle. Exact, not approximate.
        """
        if self.sweeps <= 0:
            return np.zeros_like(b)
        x = self._apply_inverse(b, transpose)
        return self.sweep(operator, b, x, transpose) if self.sweeps > 1 else x


def _to_scipy(level) -> sp.csr_matrix:
    """That level's operator as a host CSR matrix, read off the traced record."""
    return sp.csr_matrix(
        (
            np.asarray(level.operator.data),
            np.asarray(level.operator.indices),
            np.asarray(level.operator.indptr),
        ),
        shape=level.operator.shape,
    )


def _prolongation(level) -> sp.csr_matrix | None:
    """That level's prolongation as a host CSR matrix, or ``None`` on the coarsest level."""
    if level.p_frow is None:
        return None
    return sp.coo_matrix(
        (np.asarray(level.p_val), (np.asarray(level.p_frow), np.asarray(level.p_ccol))),
        shape=(level.n, level.n_coarse),
    ).tocsr()


class IluSmoothedInverse:
    """A frozen V-cycle over the traced hierarchy, relaxed on the host by an incomplete factorization.

    Satisfies the same ``n_dofs`` + ``apply(residual, transpose=…)`` contract as every other frozen
    inverse in this package, so it drops into a field split or a monolithic preconditioner wherever one
    is wanted, and it offers ``refactor_block`` so a mid-march refresh can re-fit it in place.

    Parameters
    ----------
    block : scipy.sparse matrix
        The block to precondition, field-major.
    n_fields : int
        Fields per cell, the aggregation's block size.
    cycles : int
        V-cycles per application. Fixed, so ``b -> x`` stays a linear map — required by the
        non-flexible outer Krylov solve and by the transposed adjoint.
    ordering : EliminationOrdering or None
        The order each level's smoother eliminates its unknowns in
        (:mod:`~aquaflux.solve.ordering`). ``None`` (default) is
        :class:`~aquaflux.solve.ordering.CellMajor` over the mesh's own cell order — the historical
        behaviour.

        ⚠️ **This is not a tuning knob of the usual kind.** A zero-fill factorization keeps only the
        operator's own entries, so the elimination order decides which couplings it discards. Measured
        on a coupled velocity--pressure saddle, changing nothing but this took a stationary sweep from
        growing the residual 5.5× in one application to shrinking it, and took the Krylov solve it
        preconditions from stalling to converging. The default is the *cheapest* order, not the best
        one.
    sweeps : int
        Incomplete-LU sweeps per level, per pre- and post-smooth.

        ⚠️ More is not reliably better, and this is measured rather than expected: on a coupled
        velocity--pressure saddle at zero shift the outer restart-cycle count came out **non-monotone**
        in this argument — 4, 6, 4 at 1, 2 and 4 sweeps — with four sweeps costing 2.8× the solve for
        the cycle count one sweep already reached. Nothing explains the non-monotonicity, so treat a
        result taken at a single sweep count as a result about that sweep count only.
    **coarsening
        Forwarded to :func:`~aquaflux.solve.multigrid.build_convection_hierarchy` — ``max_levels``,
        ``max_coarse``, ``strength_threshold``, ``avoid_singletons`` and the rest of the same surface
        the traced inverses take, so a coarsening choice means the same thing on both paths.
    """

    def __init__(
        self,
        block: sp.spmatrix,
        n_fields: int,
        *,
        cycles: int = 1,
        sweeps: int = 2,
        ordering: EliminationOrdering | None = None,
        **coarsening,
    ) -> None:
        matrix = sp.csr_matrix(block)
        self._n_dofs = matrix.shape[0]
        self._cycles = cycles
        self._sweeps = sweeps
        self._ordering = CellMajor() if ordering is None else ordering
        self._build_settings = dict(
            block_size=n_fields,
            mis_aggregation=True,
            **coarsening,
        )
        self._rebuild(matrix)

    def _rebuild(self, matrix: sp.csr_matrix) -> None:
        """Coarsen, then convert every level to host form and factorize its smoother."""
        self._hierarchy: SmoothedHierarchy = build_convection_hierarchy(
            matrix, **self._build_settings
        )
        self._levels = []
        for level in self._hierarchy.levels:
            coarse = None if level.coarse_inv is None else np.asarray(level.coarse_inv)
            if coarse is not None:
                self._levels.append(_HostLevel(_to_scipy(level), None, None, coarse))
                continue
            operator = _to_scipy(level)
            self._levels.append(
                _HostLevel(
                    operator,
                    _prolongation(level),
                    _LevelSmoother(operator, level.block_size, self._sweeps, self._ordering),
                    None,
                )
            )
        equilibration = self._hierarchy.equilibration
        self._equilibration = None if equilibration is None else np.asarray(equilibration)

    @property
    def n_dofs(self) -> int:
        """Degrees of freedom this inverse spans."""
        return self._n_dofs

    def _v_cycle(self, index: int, b: np.ndarray, transpose: bool) -> np.ndarray:
        """One V-cycle at ``index``, or its transpose.

        **The transposed cycle is the same recursion with every piece transposed and the pre- and
        post-smoothing exchanged.** With restriction ``P^T`` and prolongation ``P``, the coarse-grid
        correction ``P A_c^-1 P^T`` transposes to ``P A_c^-T P^T`` — the transfers keep their roles and
        only the coarse solve flips — while the smoother and the level operator each transpose. Getting
        this wrong leaves the forward solve healthy and every gradient silently wrong, which is why the
        adjoint identity is a test rather than a comment.
        """
        level = self._levels[index]
        if level.coarse_inverse is not None:
            return (level.coarse_inverse.T if transpose else level.coarse_inverse) @ b

        operator = level.operator_t if transpose else level.operator
        # Pre-smooth from a zero iterate; on the transposed cycle this is the post-smooth's transpose,
        # which is the same relaxation because pre and post share one smoother.
        x = level.smoother.sweep_from_zero(operator, b, transpose)
        residual = b - operator @ x
        coarse = level.prolongation.T @ residual
        x = x + level.prolongation @ self._v_cycle(index + 1, coarse, transpose)
        return level.smoother.sweep(operator, b, x, transpose)

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        """Approximate ``A^-1 r`` (or ``A^-T r``) with a fixed number of host V-cycles."""
        b = np.asarray(residual, dtype=np.float64)
        if self._equilibration is not None:
            b = self._equilibration * b
        if self._cycles <= 0:
            x = np.zeros_like(b)
        else:
            # The first pass starts from a zero iterate, so its residual is `b` itself and the matvec
            # that would form it is against a zero vector. Peeling it is exact.
            x = self._v_cycle(0, b, transpose)
            operator = self._levels[0].operator_t if transpose else self._levels[0].operator
            for _ in range(self._cycles - 1):
                x = x + self._v_cycle(0, b - operator @ x, transpose)
        return self._equilibration * x if self._equilibration is not None else x

    def refactor_block(self, block: sp.spmatrix) -> None:
        """Re-fit to a new operator on the same graph, in place, as a march refresh requires.

        Raises
        ------
        ValueError
            If the new block's shape differs from the built one — silently re-fitting to a different
            operator would give a preconditioner for a system nothing is solving.
        """
        matrix = sp.csr_matrix(block)
        if matrix.shape != (self._n_dofs, self._n_dofs):
            raise ValueError(
                f"cannot refactor a {self._n_dofs}-dof inverse onto a {matrix.shape[0]}-dof block."
            )
        self._rebuild(matrix)

    def destroy(self) -> None:
        """Nothing to release — ``scipy`` factors, not a host solver's handles."""


def ilu_smoothed_inverse(**settings) -> Callable[[sp.spmatrix, int], IluSmoothedInverse]:
    """A block-inverse factory pairing the traced hierarchy with a host incomplete-LU smoother.

    The CPU counterpart of :func:`~aquaflux.solve.saddle_multigrid.simple_smoothed_inverse` and
    :func:`~aquaflux.solve.field_split.jacobi_smoothed_inverse`: same coarsening surface, same contract,
    a sequential smoother instead of a vectorized one.

    Because it smooths a leading flow block the way a host algebraic-multigrid library does — a
    zero-fill incomplete factorization over an equilibrated cell-major operator — swapping this in for
    such a library isolates the **coarsening**, which is otherwise the one thing a comparison between
    them cannot hold fixed. Measured that way on a coupled backward-facing step, the two carry a full
    three-rung continuation to the same root with the aggregation here taking about a tenth fewer
    outer restart cycles and paying it back in a dearer application: parity, not a win. Read that as
    licence to drop the second hierarchy, not as a reason to prefer this one.

    Returns
    -------
    callable
        ``(block, n_group_fields) -> IluSmoothedInverse``, the shape
        :func:`~aquaflux.solve.field_split.build_block_triangular_field_split` expects.
    """

    def build(block: sp.spmatrix, n_group_fields: int) -> IluSmoothedInverse:
        return IluSmoothedInverse(block, n_group_fields, **settings)

    return build
