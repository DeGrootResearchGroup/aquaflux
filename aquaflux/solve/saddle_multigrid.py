"""A JAX-native multigrid for a velocity--pressure saddle block, smoothed by SIMPLE relaxation.

The coupled preconditioner's leading ``[u, v, w, p]`` block is a *generalized* saddle point: collocated
Rhie--Chow interpolation leaves a nonzero pressure--pressure block and makes the divergence operator
differ from the transposed gradient. The incumbent preconditioner for it is an incomplete-LU-smoothed
algebraic multigrid, which is strong but whose smoother is a **sequential triangular solve** -- the one
piece of the coupled solver that does not move to an accelerator.

This module replaces that smoother, not the hierarchy. Each level is relaxed by a SIMPLE sweep: a
velocity predictor built from an approximate inverse of the velocity block, a few damped-Jacobi sweeps on
an algebraic Schur complement, and a correction. Every operation in it is a diagonal scaling or a sparse
matrix--vector product, so the whole cycle is the shape an accelerator wants; the coarse grid, not the
smoother, carries the smooth global pressure mode a SIMPLE-type Schur approximates worst.

**The velocity splitting is the Frobenius-optimal one, and that choice is load-bearing.** Minimizing
``||I - F~^-1 F||_F`` over diagonal approximate inverses gives ``F~^-1_ii = F_ii / ||F_i||^2`` (Jemcov and
Maruszewski, 2008) -- Jacobi with a derived per-row under-relaxation rather than a tuned constant. Under a
plain Jacobi diagonal the sweep *amplifies*, so more sweeps make it worse; under this one it contracts.
:func:`block_approximate_inverse` generalizes it to a per-cell block, where the transpose in
``M_i = F_ii^T (R_i R_i^T)^-1`` is easy to drop and reduces to the scalar form at one field per cell, so a
scalar check cannot catch its absence.

The cycle is a **fixed** linear operator -- a fixed number of cycles, sweeps and inner sweeps, with no
inner Krylov method -- which is what a non-flexible outer GMRES requires and what lets the adjoint
transpose it with :func:`jax.linear_transpose`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .multigrid import (
    _AGGREGATE_STATS,
    SmoothedHierarchy,
    _CsrOperator,
    _operator_matvec,
    _smoothed_ops,
    build_convection_hierarchy,
)

__all__ = ["NativeSimpleInverse", "block_approximate_inverse", "native_saddle_inverse"]


class _SimplePieces(eqx.Module):
    """One level's SIMPLE relaxation pieces, all frozen and all traced.

    ``n_velocity`` is the split point in the level's field-major vector; everything else is either an
    elementwise reciprocal or a sparse operator, so a sweep is diagonal scalings and matrix-vector
    products and nothing else -- no factorization, no triangular solve, no sequential dependency.

    **An ``equinox.Module`` with only the split point static, on the same rule as
    :class:`~aquaflux.solve.multigrid._SparseLevel`, and for the same reason.** That makes the record a
    pytree whose every array is a traced leaf, so it can be passed as an *argument* to the jitted cycle
    rather than captured by it. Re-deriving the pieces at a new operator on unchanged shapes then reuses
    the compiled cycle instead of tracing a new one -- which is what makes a mid-march refresh cheap, and
    is defeated outright by a closure, since a closed-over array is a compile-time constant and every
    refresh mints a new one.

    The formed Schur is deliberately **not** carried here. It has no traced form a diagnostic can
    transpose, and a host sparse matrix is neither a leaf nor a hashable static field, so
    :func:`_simple_pieces` returns it alongside instead.
    """

    f_diagonal_inverse: jnp.ndarray  # (n_velocity,) 1 / diag(F)
    dg: _CsrOperator  # diag(F)^-1 G, the velocity correction operator
    divergence: _CsrOperator  # D
    schur: _CsrOperator  # S = C - D diag(F)^-1 G
    schur_diagonal_inverse: jnp.ndarray  # (n_pressure,) 1 / diag(S)
    #: Set when the splitting is a per-cell BLOCK inverse instead of a scalar diagonal; the velocity
    #: predictor then applies this rather than an elementwise multiply. ``None`` keeps the diagonal path,
    #: which stays an elementwise multiply rather than a nine-nonzero-per-row matvec.
    f_block_inverse: _CsrOperator | None
    #: Static: it slices the level vector, so it must be concrete.
    n_velocity: int = eqx.field(static=True)


def block_approximate_inverse(f_block, n_cells, n_fields, frobenius):
    """A per-cell block approximate inverse of the velocity block, as a sparse operator.

    The scalar diagonal this replaces throws away the coupling between a cell's own velocity components.
    A block inverse keeps it at a precomputed ``n_fields x n_fields`` solve per cell -- nine multiplies
    instead of three, negligible beside the matrix-vector products -- so it is the cheapest strengthening
    of the splitting available, and the splitting error is what the eigenvalue clustering of a
    block-diagonal saddle preconditioner is governed by.

    ``frobenius`` selects the block generalization of the Frobenius-optimal diagonal. Minimizing
    ``||I - M F||_F`` over block-diagonal ``M`` decouples by cell: with ``R_i`` the cell's row block of
    ``F``, the minimizer is ``M_i = F_ii^T (R_i R_i^T)^-1``. At one field per cell this reduces exactly to
    ``F_ii / ||F_i||^2``, which is the diagonal form measured to be worth four orders as a velocity
    predictor -- so the block version should not be built without it.

    Parameters
    ----------
    f_block : scipy.sparse matrix
        The velocity block, shape ``(n_fields * n_cells,) * 2``, field-major.
    n_cells, n_fields : int
        Cells, and velocity components per cell.
    frobenius : bool
        Use the Frobenius-optimal block rather than the exact inverse of the diagonal block.

    Returns
    -------
    scipy.sparse.csr_matrix
        The block-diagonal approximate inverse, shape ``(n_fields * n_cells,) * 2``.
    """
    diagonal_blocks = np.zeros((n_cells, n_fields, n_fields))
    for f in range(n_fields):
        rows_f = f_block[f * n_cells : (f + 1) * n_cells]
        for g in range(n_fields):
            diagonal_blocks[:, f, g] = rows_f[:, g * n_cells : (g + 1) * n_cells].diagonal()
    if frobenius:
        # Gram matrix of each cell's row block: (R_i R_i^T)_{fg} = <row f, row g> over the whole row.
        gram = np.zeros((n_cells, n_fields, n_fields))
        row_blocks = [f_block[f * n_cells : (f + 1) * n_cells] for f in range(n_fields)]
        for f in range(n_fields):
            for g in range(n_fields):
                gram[:, f, g] = np.asarray(
                    row_blocks[f].multiply(row_blocks[g]).sum(axis=1)
                ).ravel()
        # M_i = F_ii^T G^-1 with G = R_i R_i^T symmetric, computed as solve(G, F_ii) transposed:
        # (G^-1 F_ii)^T = F_ii^T G^-T = F_ii^T G^-1. Passing F_ii^T here instead would give F_ii G^-1,
        # which differs whenever the cell's own velocity-component block is nonsymmetric -- and it
        # coincides at one field per cell, so a scalar reduction check cannot catch the difference.
        inverse = np.transpose(np.linalg.solve(gram, diagonal_blocks), (0, 2, 1))
    else:
        inverse = np.linalg.inv(diagonal_blocks)
    cells = np.arange(n_cells)
    rows, cols, vals = [], [], []
    for f in range(n_fields):
        for g in range(n_fields):
            rows.append(cells + f * n_cells)
            cols.append(cells + g * n_cells)
            vals.append(inverse[:, f, g])
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_fields * n_cells,) * 2,
    )


def _simple_pieces(
    a: sp.csr_matrix,
    block_size: int,
    frobenius: bool = True,
    schur_frobenius: bool = False,
    block_splitting: bool = False,
    simplec: bool = False,
    report: Callable[[str], None] = lambda _message: None,
) -> tuple[_SimplePieces, sp.csr_matrix]:
    """Form one level's SIMPLE pieces from that level's assembled operator.

    Works on **any** level, including a Galerkin coarse operator, because it reads nothing but the
    matrix -- which is the property that decides whether a SIMPLE relaxation can be a smoother at all.
    The assembler-built Schur cannot: it needs the mesh, the Rhie--Chow coefficients and the boundary
    closures, none of which a coarse level has.

    Returns
    -------
    pieces : _SimplePieces
        The traced record the smoother applies.
    schur : scipy.sparse.csr_matrix
        The same Schur complement in host sparse form, returned separately because it cannot ride on a
        traced pytree. Forming it is the expensive part of this function, so a diagnostic that needs a
        transposable sparse operator takes it from here rather than rebuilding it -- which would be a
        second spelling of ``C - D diag(F)^-1 G``.
    """
    n_cells = a.shape[0] // block_size
    nv = (block_size - 1) * n_cells
    # The approximate velocity inverse the whole relaxation is built on. Jacobi (`1 / F_ii`) is what
    # this smoother used when it amplified; the Frobenius-optimal diagonal is the same object with a
    # derived per-row relaxation, which is measured to be worth 25x in a flat setting on this operator
    # -- and an under-relaxed sweep is exactly what an amplifying one is missing.
    f_inverse = _diagonal_approximate_inverse(
        a[:nv, :nv].tocsr(), frobenius, row_sum=simplec, label=" (velocity)"
    )
    g_block = a[:nv, nv:].tocsr()
    d_block = a[nv:, :nv].tocsr()
    if block_splitting and simplec:
        # The block inverse overrides the diagonal entirely, so a SIMPLEC diagonal built alongside it is
        # computed, reported, and never applied -- an arm whose name claims a splitting it does not run.
        # Caught because the pair returned a residual bit-identical to block splitting alone.
        raise ValueError(
            "block splitting overrides the velocity diagonal, so `simplec` would have no effect; "
            "choose one."
        )
    if block_splitting:
        f_inverse_matrix = block_approximate_inverse(
            a[:nv, :nv].tocsr(), n_cells, block_size - 1, frobenius
        )
        block_inverse = _CsrOperator.from_scipy(f_inverse_matrix)
    else:
        f_inverse_matrix = sp.diags(f_inverse)
        block_inverse = None
    dg = (f_inverse_matrix @ g_block).tocsr()
    schur = (a[nv:, nv:] - d_block @ dg).tocsr()
    # The formed Schur against the pieces it is built from. Applying `S` matrix-free -- as C.p minus
    # D.(diag(F)^-1.(G.p)) -- is algebraically identical, so which is cheaper is purely a question of
    # which side carries more nonzeros. Printed rather than assumed.
    report(
        f"      Schur {schur.data.shape[0] / 1e6:.1f}M nnz formed vs "
        f"{(a[nv:, nv:].nnz + d_block.nnz + g_block.nnz) / 1e6:.1f}M as pieces "
        f"(C {a[nv:, nv:].nnz / 1e6:.1f}M + D {d_block.nnz / 1e6:.1f}M + G {g_block.nnz / 1e6:.1f}M)",
    )
    schur_diagonal = schur.diagonal()
    # A zero Schur diagonal would make the pressure relaxation undefined. It does not arise on this
    # discretization -- Rhie-Chow damping gives the continuity row a genuine diagonal, and the
    # correction only strengthens it -- so this is a guard against a coarse level degenerating, not a
    # routine case to smooth over.
    if np.any(schur_diagonal == 0.0):
        raise ValueError("the Schur complement has a zero diagonal on some level.")
    schur_inverse = _diagonal_approximate_inverse(schur, schur_frobenius)
    if schur_frobenius:
        ratio = (
            schur_inverse * schur_diagonal
        )  # vs Jacobi's 1/S_ii, so this is the relaxation factor
        report(
            f"      Schur relaxation (Eq. 39): min {ratio.min():.3e} median {np.median(ratio):.3e} "
            f"max {ratio.max():.3e}  ({schur.data.shape[0] / schur.shape[0]:.0f} nnz/row)",
        )
    return (
        _SimplePieces(
            f_block_inverse=block_inverse,
            n_velocity=nv,
            f_diagonal_inverse=jnp.asarray(f_inverse),
            dg=_CsrOperator.from_scipy(dg),
            divergence=_CsrOperator.from_scipy(d_block),
            schur=_CsrOperator.from_scipy(schur),
            schur_diagonal_inverse=jnp.asarray(schur_inverse),
        ),
        schur,
    )


def _simple_correction(
    pieces: _SimplePieces, residual, pressure_sweeps: int, pressure_omega: float
):
    """One SIMPLE correction for a level residual: the classical predictor / Schur / correct sequence.

    ``du* = diag(F)^-1 r_u``; ``dp`` from a few damped-Jacobi sweeps on ``S`` against
    ``r_p - D du*``; ``du = du* - diag(F)^-1 G dp``. Every operation is a diagonal scaling or a sparse
    matrix-vector product, which is the whole point: this is what an incomplete-LU sweep is *not*, and
    it is why it can run on an accelerator.

    The pressure solve is deliberately a fixed handful of sweeps rather than anything converged. As a
    smoother it only has to damp high-frequency error -- the coarse grid carries the smooth pressure
    mode, which is exactly the mode a SIMPLE Schur approximates worst. A fixed count also keeps the
    whole correction a constant **linear** map, which the non-flexible outer Krylov and the transposed
    adjoint both require.
    """
    nv = pieces.n_velocity
    velocity_residual, pressure_residual = residual[:nv], residual[nv:]
    predictor = (
        pieces.f_block_inverse.apply(velocity_residual)
        if pieces.f_block_inverse is not None
        else pieces.f_diagonal_inverse * velocity_residual
    )
    rhs = pressure_residual - pieces.divergence.apply(predictor)
    # The first sweep is peeled because it would otherwise multiply the Schur complement by a zero
    # vector: starting from p = 0, the update collapses to omega * S_diag^-1 * rhs. That is one of every
    # `pressure_sweeps` applications of the densest operator in the smoother, and it is not folded away
    # -- the sparse product runs at full cost against the zeros. Peeling it is bit-identical.
    if pressure_sweeps <= 0:
        pressure = jnp.zeros_like(rhs)
    else:
        pressure = pressure_omega * pieces.schur_diagonal_inverse * rhs

    # A LOOP, not a Python `for`, so the remaining sweeps enter the graph ONCE instead of being
    # unrolled into `pressure_sweeps - 1` copies of the densest operator in the smoother. Every sweep
    # has identical shapes, so there is nothing for unrolling to specialize on, and the traced graph is
    # what a refresh has to recompile whenever the coarsening moves. `fori_loop` at a static trip count
    # stays linearly transposable -- verified to give the same value AND the same transpose as the
    # unrolled form -- which the adjoint requires.
    def sweep(_, p):
        return p + pressure_omega * pieces.schur_diagonal_inverse * (rhs - pieces.schur.apply(p))

    pressure = jax.lax.fori_loop(0, max(pressure_sweeps - 1, 0), sweep, pressure)
    return jnp.concatenate([predictor - pieces.dg.apply(pressure), pressure])


class _SmootherSettings(NamedTuple):
    """The cycle's counts and relaxations -- everything about it that must be concrete.

    A plain tuple of Python numbers, so it is hashable and compares by value: under
    :func:`equinox.filter_jit` it lands wholly on the static side and two builds at the same settings
    share one compiled cycle. That is the point of separating it from the pieces, which land wholly on
    the traced side.
    """

    cycles: int
    sweeps: int
    pressure_sweeps: int
    pressure_omega: float
    omega: float
    mu: int
    pre_smooth: bool


@eqx.filter_jit
def _native_saddle_cycle(
    hierarchy: SmoothedHierarchy,
    pieces: dict[int, _SimplePieces],
    residual: jnp.ndarray,
    settings: _SmootherSettings,
) -> jnp.ndarray:
    """``settings.cycles`` V-cycles over ``hierarchy``, smoothed by SIMPLE relaxation.

    **Module-level, and taking the hierarchy and the pieces as ARGUMENTS rather than closing over them,
    so a refresh at unchanged shapes is a compilation-cache hit.** A locally-defined ``jax.jit`` closure
    is a fresh cache entry per closure, so re-deriving the preconditioner would recompile the whole
    cycle every time -- which is what the level records' static/traced split, the shape ladder and
    :meth:`~aquaflux.solve.multigrid.SmoothedHierarchy.refit` all exist to avoid. Closing over the
    arrays is also expensive on the *first* build: they become compile-time constants, so a hierarchy's
    worth of them is embedded in the compiled program rather than passed as buffers.

    ``pieces`` is keyed by level size because the V-cycle recursion is unrolled at trace time, so the
    smoother is handed the concrete level object and looks its pieces up by a static attribute; the
    coarsest level solves directly and has none.

    Parameters
    ----------
    hierarchy : SmoothedHierarchy
        The coarsened levels, finest first.
    pieces : dict
        Each smoothed level's SIMPLE pieces, keyed by that level's degree-of-freedom count.
    residual : jnp.ndarray
        The right-hand side, shape ``(n_dofs,)``.
    settings : _SmootherSettings
        The cycle's static counts and relaxations.

    Returns
    -------
    jnp.ndarray
        The approximate solution, shape ``(n_dofs,)``.
    """

    def smooth(level, rhs, guess):
        piece = pieces[level.n]

        # Looped rather than unrolled, for the same reason as the inner pressure sweeps: the graph is
        # what a refresh recompiles, and it is otherwise unrolled over levels x sweeps x inner sweeps.
        # The level dimension has to stay unrolled (each level has its own shapes), so the two sweep
        # dimensions are where the size actually comes from.
        def outer(_, g):
            correction = _simple_correction(
                piece,
                rhs - _operator_matvec(level, g),
                settings.pressure_sweeps,
                settings.pressure_omega,
            )
            return g + settings.omega * correction

        return jax.lax.fori_loop(0, settings.sweeps, outer, guess)

    def smooth_zero(level, rhs):
        """``smooth`` from a zero iterate, with the first sweep's residual matvec peeled off.

        At ``g = 0`` the residual ``rhs - A g`` is exactly ``rhs``, so that application of the level
        operator -- the densest thing in the smoother -- computes a known answer at full price. The
        pre-smooth always starts here, so it is charged at every level of every cycle. This is the same
        peel :func:`_simple_correction` already does for the first pressure sweep, one loop out.
        """
        if settings.sweeps <= 0:
            return jnp.zeros_like(rhs)
        piece = pieces[level.n]
        guess = settings.omega * _simple_correction(
            piece, rhs, settings.pressure_sweeps, settings.pressure_omega
        )

        def outer(_, g):
            correction = _simple_correction(
                piece,
                rhs - _operator_matvec(level, g),
                settings.pressure_sweeps,
                settings.pressure_omega,
            )
            return g + settings.omega * correction

        return jax.lax.fori_loop(0, settings.sweeps - 1, outer, guess)

    ops = _smoothed_ops(
        smooth, mu=settings.mu, pre_smooth=settings.pre_smooth, smooth_zero=smooth_zero
    )
    return hierarchy.fixed_cycle_solve(residual, settings.cycles, ops)


def _diagonal_approximate_inverse(
    f_block: sp.csr_matrix,
    frobenius: bool,
    row_sum: bool = False,
    label: str = "",
    report: Callable[[str], None] = lambda _message: None,
) -> np.ndarray:
    """The diagonal approximate inverse of a block, Jacobi or Frobenius-optimal.

    Written for the velocity block but specific to nothing about it -- the derivation reads only the
    rows of whatever matrix it is handed, so it applies equally to the Schur complement, whose own
    relaxation faces the same choice and whose diagonal is a **worse** approximation to its inverse
    (the Schur here carries some 300 nonzeros per row against the flow block's 227).

    Jacobi takes ``1 / F_ii``. The Frobenius-optimal diagonal instead minimizes ``||I - F~^-1 F||_F``,
    which for a diagonal unknown decouples row by row and has the closed form ``F_ii / ||F_i||^2``.

    **Written as a ratio the two differ by ``F_ii^2 / ||F_i||^2``, which is at most one and is the
    fraction of row i's energy sitting on its diagonal — so the optimal choice is Jacobi with an
    automatic, per-row under-relaxation.** That is worth stating plainly because the hand-tuned version
    of the same quantity appears throughout this solver: a velocity-row under-relaxation, a
    preconditioner-only shift floor on the velocity rows, and the relative velocity-row relaxation the
    closest published work on this discretization never manages to drop. Here it is derived rather than
    chosen, which is the paper's own argument for why its formulation needs no under-relaxation at all.
    """
    diagonal = f_block.diagonal()
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal == 0.0):
        raise ValueError("the velocity block has a zero or non-finite diagonal.")
    if row_sum:
        # The SIMPLEC coefficient. Where SIMPLE drops the neighbour corrections entirely and divides by
        # a_P, SIMPLEC approximates each neighbour correction by the cell's own, which collapses the
        # neighbour sum onto the diagonal and divides by a_P - sum(a_nb) -- exactly this matrix's ROW
        # SUM. Its appeal as a smoother is a property the other two choices lack: `I - F~^-1 F`
        # annihilates the constant vector exactly, so the smoother does not fight the coarse grid over
        # the smoothest mode, which is the one the coarse grid exists to carry.
        #
        # It has a failure mode the others do not. A row sum can approach zero where a diagonal cannot,
        # and this operator is measurably NOT diagonally dominant, so rows whose sum is small relative
        # to their own diagonal would produce an enormous coefficient. Those fall back to the
        # Frobenius-optimal value, and the count is reported: if many rows fall back, SIMPLEC is not
        # meaningfully in force and any result under it is really a result about the fallback.
        sums = np.asarray(f_block.sum(axis=1)).ravel()
        usable = sums > 0.1 * np.abs(diagonal)
        squared = np.asarray(f_block.multiply(f_block).sum(axis=1)).ravel()
        fallback = diagonal / squared
        inverse = np.where(usable, 1.0 / np.where(usable, sums, 1.0), fallback)
        report(
            f"      SIMPLEC splitting{label}: {int((~usable).sum())} of {usable.size} rows fell back "
            f"to Frobenius (row sum below a tenth of the diagonal)",
        )
        return inverse
    if not frobenius:
        return 1.0 / diagonal
    # Row 2-norms squared, straight off the CSR values.
    squared = np.asarray(f_block.multiply(f_block).sum(axis=1)).ravel()
    if np.any(squared == 0.0):
        raise ValueError("the velocity block has an empty row; the optimal inverse is undefined.")
    return diagonal / squared


class NativeSimpleInverse:
    """A JAX-native multigrid over the flow saddle whose LEVEL SMOOTHER is a SIMPLE relaxation.

    This is the arm every earlier one was not. Those replaced the flow block's preconditioner outright
    -- a flat block inverse with no hierarchy, no levels and no coarse-grid correction -- so their
    accuracy was capped by how well a single application approximates ``A^-1``, and for a SIMPLE-type
    method that cap is set by the Schur approximation's worst mode, the smooth global pressure mode.
    That is visible in the measurements as a residual floor almost insensitive to the pseudo-transient
    shift, which is the signature of an approximation ceiling rather than a conditioning one.

    Here the hierarchy is kept and SIMPLE replaces the incomplete-LU **smoother**. The division of
    labour is the point: the smoother only has to damp high-frequency error, where a local algebraic
    Schur is accurate, and the coarse solve carries the global mode, where it is not.

    Nothing in it is a host solver and nothing in it is sequential -- the setup is sparse products in
    ``scipy`` and the apply is diagonal scalings and sparse matrix-vector products in JAX.

    Parameters
    ----------
    block : scipy.sparse matrix
        The flow block, field-major, ``[u, v, w]`` then ``p``.
    n_fields : int
        Fields per cell, the aggregation's block size.
    cycles, sweeps : int
        V-cycles per application, and SIMPLE sweeps per level. Both fixed, so ``b -> x`` stays linear.
    pressure_sweeps, pressure_omega : int, float
        The damped-Jacobi sweeps used for the Schur solve inside one SIMPLE sweep, and their damping.
    omega : float
        Relaxation applied to the whole SIMPLE correction.
    """

    def __init__(
        self,
        block: sp.spmatrix,
        n_fields: int,
        *,
        cycles: int = 1,
        sweeps: int = 2,
        pressure_sweeps: int = 4,
        # Undamped, and the Frobenius diagonals below are why. Measured as a 2x2 at four sweeps: with
        # the Jacobi diagonal an undamped pressure sweep blows up (3.4e-01 against 6.8e-05), while with
        # the Frobenius one removing the damping HELPS (6.8e-05 -> 4.2e-05). The hand-set 0.7 and the
        # derived per-row relaxation were doing the same job, so stacking them over-damped -- but the
        # constant cannot simply be dropped, it can only be replaced.
        pressure_omega: float = 1.0,
        omega: float = 0.7,
        max_coarse: int = 2000,
        frobenius: bool = True,
        schur_frobenius: bool = True,
        levels: int = 2,
        aggressive: int = 1,
        strength_threshold: float = 0.0,
        orthonormal: bool = False,
        avoid_singletons: bool = False,
        # A W-cycle visits each coarse level twice per visit of its parent. It buys convergence with
        # COARSE work, where every other lever here buys it with fine-level relaxation -- and the fine
        # level is ~60% of the smoothing cost, so the two are priced very differently.
        # Replace the velocity predictor's scalar diagonal with a per-cell block inverse. The splitting
        # error is the larger half of what governs this preconditioner's eigenvalue clustering at the fine
        # level, and a diagonal cannot represent a cell's own velocity-component coupling at all.
        block_splitting: bool = False,
        # SIMPLEC's velocity coefficient: divide by the row sum rather than the diagonal. The appeal as
        # a smoother is that the splitting error then annihilates constants exactly, so it does not
        # fight the coarse grid over the smoothest mode.
        simplec: bool = False,
        mu: int = 1,
        # Dropping the pre-relaxation also drops the residual matvec that follows it, which at two inner
        # pressure sweeps is the single largest term in a sweep.
        pre_smooth: bool = True,
        # Every arm here has run UNSMOOTHED aggregation, which interpolates a coarse correction by
        # injecting it piecewise-constant over each aggregate. That is the standard explanation for a
        # hierarchy that works at two levels and gains nothing deeper: the interpolation error does not
        # fall as the grids coarsen. Smoothing the prolongator once with the operator is what makes
        # aggregation multigrid depth-independent.
        prolongation_smoothing: str = "none",
        equilibrate: bool = False,
        # Refresh by refitting values onto the coarsening derived at the first build, rather than
        # coarsening again. At a nonzero strength threshold the aggregation reads |A_ij|, so a rebuild
        # at a developed state returns a DIFFERENT partition and the jitted cycle retraces; frozen, the
        # shapes cannot move. What it trades is coarse-space quality: the partition then describes the
        # operator at the state it was first built at, for the whole march.
        frozen_coarsening: bool = False,
        # Coarsen into a FIXED ladder of array sizes, discovered on the first build and padded by this
        # factor. Unlike freezing, the partition is still re-derived from the current operator at every
        # refresh -- only the sizes it is poured into are held -- so the coarse space tracks the flow
        # while the compiled cycle stays a cache hit. `None` disables it and is byte-identical.
        shape_headroom: float | None = None,
        # Where the build record goes. The hierarchy's level sizes, aggregate statistics and parsed
        # smoother settings are worth having in a run log -- a spec-token collision once made an arm run
        # forty sweeps while every other line of output looked correct -- but a library object must not
        # write to stdout on its own. `None` is silent; a case passes `print`.
        report: Callable[[str], None] | None = None,
    ) -> None:
        self._report = report if report is not None else lambda _message: None
        matrix = sp.csr_matrix(block)
        self._n_dofs = matrix.shape[0]
        # Held so a refresh re-derives at exactly the settings this was built at, rather than at
        # whatever the builder defaults happen to be -- the same reason the sibling nodal inverse
        # keeps its own build settings.
        self._settings = dict(
            n_fields=n_fields,
            cycles=cycles,
            sweeps=sweeps,
            pressure_sweeps=pressure_sweeps,
            pressure_omega=pressure_omega,
            omega=omega,
            max_coarse=max_coarse,
            frobenius=frobenius,
            schur_frobenius=schur_frobenius,
            levels=levels,
            aggressive=aggressive,
            strength_threshold=strength_threshold,
            orthonormal=orthonormal,
            avoid_singletons=avoid_singletons,
            block_splitting=block_splitting,
            simplec=simplec,
            mu=mu,
            pre_smooth=pre_smooth,
            prolongation_smoothing=prolongation_smoothing,
            equilibrate=equilibrate,
            frozen_coarsening=frozen_coarsening,
            shape_headroom=shape_headroom,
        )
        # The cycle's static half, built once: it never changes over this inverse's life, so a refresh
        # cannot move the compilation key through it.
        self._smoother = _SmootherSettings(
            cycles=cycles,
            sweeps=sweeps,
            pressure_sweeps=pressure_sweeps,
            pressure_omega=pressure_omega,
            omega=omega,
            mu=mu,
            pre_smooth=pre_smooth,
        )
        # Discovered on the first build and held for the life of the inverse, so every later rebuild
        # coarsens into the same ladder. `None` until then, and forever if no headroom was asked for.
        self._budget = None
        self._rebuild(matrix)

    def _rebuild(self, matrix: sp.csr_matrix) -> None:
        """Coarsen ``matrix`` from scratch, then derive the smoother and the jitted cycle over it.

        With ``shape_headroom`` set, the FIRST build runs twice: once to discover what this operator
        naturally coarsens into, then again into a budget derived from it. Every later rebuild reuses
        that budget, so it re-derives the partition from the current operator -- unlike a frozen
        coarsening -- while keeping the array shapes, and therefore the compiled cycle, fixed.
        """
        settings = self._settings
        self._hierarchy = self._coarsen(matrix, self._budget)
        if self._budget is None and settings["shape_headroom"] is not None:
            self._budget = self._hierarchy.shape_budget(settings["shape_headroom"])
            self._report(
                f"      shape ladder: cells {self._budget.coarse_cells}, "
                f"nnz {self._budget.operator_nnz} (headroom {settings['shape_headroom']})",
            )
            self._hierarchy = self._coarsen(matrix, self._budget)
        self._after_coarsening()

    def _coarsen(self, matrix: sp.csr_matrix, budget):
        """Build the hierarchy at this inverse's settings, optionally into a fixed shape ladder."""
        settings = self._settings
        return build_convection_hierarchy(
            matrix,
            block_size=settings["n_fields"],
            max_coarse=settings["max_coarse"],
            mis_aggregation=True,
            # Depth and coarsening RATE are two halves of one choice. One aggressive level gives a
            # roughly hundredfold jump in a single step, so one prolongation from a ~860-equation
            # coarse space carries the whole error; more levels at a gentler rate ask less of each
            # interpolation.
            aggressive_levels=settings["aggressive"],
            max_levels=settings["levels"],
            # The coarsening RATE. Aggregating only along strong connections makes the aggregates
            # smaller, landing a coarse grid between the squared graph's ~106x and plain aggregation's
            # ~21x -- and the coarse grid size is the binding cost here, since the coarsest level is
            # inverted DENSELY (cubic to build, quadratic to store), which is affordable at ~1000
            # equations and is not at 4300 and above.
            strength_threshold=settings["strength_threshold"],
            orthonormal_prolongation=settings["orthonormal"],
            avoid_singletons=settings["avoid_singletons"],
            prolongation_smoothing=settings["prolongation_smoothing"],
            equilibrate=settings["equilibrate"],
            shape_budget=budget,
        )

    def _after_coarsening(self) -> None:
        """Capture this build's aggregate statistics, then derive the smoother over the hierarchy."""
        # Snapshot NOW, while the accumulator still describes this build: it is module-level and every
        # later hierarchy overwrites it, so reading it at report time is reading someone else's.
        self._aggregate_stats = list(_AGGREGATE_STATS[-(len(self._hierarchy.levels) - 1) :])
        self._derive_cycle()

    def _derive_cycle(self) -> None:
        """Build the per-level SIMPLE pieces over the current hierarchy, and report the build.

        Split out of :meth:`_rebuild` because a refresh has two ways to reach the same place -- coarsen
        again, or refit the values onto the coarsening already held -- and only the hierarchy differs
        between them. Everything below reads ``self._hierarchy`` and is common to both.

        **It builds no jitted function.** The cycle is the module-level :func:`_native_saddle_cycle`,
        which takes the hierarchy and these pieces as arguments, so re-deriving them on unchanged shapes
        reuses the compiled cycle. Building one here instead would put them in a closure, where every
        refresh is a fresh cache key and a full recompile.
        """
        settings = self._settings
        sweeps = settings["sweeps"]
        pressure_sweeps = settings["pressure_sweeps"]
        pressure_omega = settings["pressure_omega"]
        omega = settings["omega"]
        frobenius = settings["frobenius"]
        schur_frobenius = settings["schur_frobenius"]
        strength_threshold = settings["strength_threshold"]
        block_splitting = settings["block_splitting"]
        simplec = settings["simplec"]
        # Keyed by level size: the V-cycle recursion is unrolled at trace time, so the smoother is
        # handed the concrete level object and can look its pieces up by a static attribute. The
        # coarsest level solves directly and needs none.
        pieces = {}
        for level in self._hierarchy.levels:
            if level.coarse_inv is not None:
                continue
            operator = level.operator
            level_matrix = sp.csr_matrix(
                (
                    np.asarray(operator.data),
                    np.asarray(operator.indices),
                    np.asarray(operator.indptr),
                ),
                shape=operator.shape,
            )
            # The formed Schur is discarded: nothing here applies it, and the smoother's traced copy
            # is the one the cycle uses. A diagnostic that wants it in host form calls `_simple_pieces`
            # itself and takes it from the pair.
            pieces[level.n], _ = _simple_pieces(
                level_matrix,
                level.block_size,
                frobenius,
                schur_frobenius,
                block_splitting,
                simplec,
                report=self._report,
            )
        sizes = ", ".join(
            f"level {n}: S {p.schur.shape[0]} dofs / {p.schur.data.shape[0] / 1e6:.1f}M nnz"
            for n, p in pieces.items()
        )
        # Report the COARSE GRID SIZE and the coarsening ratio beside the level count. A level count
        # alone cannot distinguish a hierarchy that stopped because it hit its cap from one that
        # stopped because it reached its coarse limit, and it says nothing about whether a single
        # aggregation had to represent the error across a hundredfold jump.
        #
        # These come from THIS inverse's own last coarsening, captured when it ran. Reading the module
        # accumulator here instead would print whichever hierarchy aggregated most recently -- and on
        # the refit path nothing aggregates at all, so it printed another block's aggregates as if they
        # were this one's. A frozen coarsening still HAS aggregates; they are the ones below.
        for depth, stat in enumerate(self._aggregate_stats):
            self._report(
                f"      aggregates level {depth}: {stat['aggregates']} of size "
                f"{stat['min']}/{stat['median']:.0f}/{stat['max']} (min/med/max), "
                f"spread {stat['spread']:.0f}x, {stat['singletons']} singletons",
            )
        # Print what was actually PARSED, not just what was built. A spec-token collision once made an
        # arm run forty sweeps instead of four while every other line of output looked correct.
        self._report(
            f"      smoother: {sweeps} sweeps x {pressure_sweeps} inner, omega {omega}, "
            f"simplec {simplec}, "
            f"pressure_omega {pressure_omega}, strength {strength_threshold}, "
            f"block splitting {block_splitting}",
        )
        coarse = self._hierarchy.levels[-1]
        self._report(
            f"      native SIMPLE smoother: {len(self._hierarchy.levels)} levels, "
            f"fine {self._n_dofs} -> coarse {coarse.n} dofs "
            f"({self._n_dofs / max(coarse.n, 1):.0f}x, direct solve), {sizes}",
        )

        self._pieces = pieces
        # LAZY. `jax.linear_transpose` traces eagerly, and a forward march never applies the transpose --
        # only the adjoint does. Building it at construction cost a measured 0.27 GB and 0.27 s per arm
        # for something most callers never touch, on a machine where the memory is the binding constraint.
        # Dropped here rather than kept, because it describes the pieces this call has just replaced.
        self._transpose_fn = None

    @property
    def n_dofs(self) -> int:
        return self._n_dofs

    def _solve(self, residual: jnp.ndarray) -> jnp.ndarray:
        """The cycle over the pieces this inverse currently holds, traced in and not captured."""
        return _native_saddle_cycle(self._hierarchy, self._pieces, residual, self._smoother)

    def refactor_block(self, block: sp.spmatrix) -> None:
        """Re-fit to a new operator on the same graph, in place. Required to survive a march refresh.

        The field split refuses to refresh an inverse that offers neither this nor ``refactor``, so
        without it this preconditioner cannot be used in a march at all -- a single-state probe never
        reaches the code path.

        **Whether a rebuild is structure-preserving depends on a setting.** The sibling nodal inverse
        can rebuild freely because its aggregation reads only the sparsity pattern, so every array keeps
        its shape and the jitted cycle stays a compilation-cache hit. That argument does NOT hold at a
        nonzero strength threshold, where the aggregation reads ``|A_ij|`` and the coarsening moves as
        the flow develops. Two answers are available and the setting chooses between them: coarsen again
        and report any move (the level sizes are compared against the previous build, so a retrace shows
        up in the refresh timings with a reason attached), or hold the coarsening and refit only the
        values onto it (``frozen_coarsening``), which cannot move the shapes at all.

        Parameters
        ----------
        block : scipy.sparse matrix
            The group's new diagonal block, field-major, of the shape this was built at.

        Raises
        ------
        ValueError
            If the new block's shape differs from the built one.
        """
        matrix = sp.csr_matrix(block)
        if matrix.shape != (self._n_dofs, self._n_dofs):
            raise ValueError(
                f"cannot refactor a {self._n_dofs}-dof inverse onto a {matrix.shape[0]}-dof block."
            )
        if self._settings["frozen_coarsening"]:
            self._hierarchy = self._hierarchy.refit(matrix)
            self._derive_cycle()
            return
        before = tuple(level.n for level in self._hierarchy.levels)
        self._rebuild(matrix)
        after = tuple(level.n for level in self._hierarchy.levels)
        if after != before:
            self._report(
                f"      refresh moved the coarsening: levels {before} -> {after} "
                f"(the jitted cycle retraces; the strength threshold reads values)",
            )

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        vector = jnp.asarray(residual, dtype=jnp.float64)
        if not transpose:
            return np.asarray(self._solve(vector), dtype=np.float64)
        if self._transpose_fn is None:
            self._transpose_fn = jax.linear_transpose(
                self._solve, jnp.zeros(self._n_dofs, dtype=jnp.float64)
            )
        return np.asarray(self._transpose_fn(vector)[0], dtype=np.float64)

    def destroy(self) -> None:
        """Nothing to release -- plain arrays, no host solver handles."""


def native_saddle_inverse(**settings) -> Callable[[sp.spmatrix, int], object]:
    """A ``leading_inverse`` factory using :class:`NativeSimpleInverse`.

    Every keyword is forwarded, so the defaults — and the reasoning behind them — live on the class
    rather than being restated here. The settings worth knowing about are ``strength_threshold`` (at
    zero the aggregation reads no operator values at all and coarsens across the stiff direction, which
    on a wall-graded mesh is the difference between a working hierarchy and one that stalls) and the
    pair ``levels``/``max_coarse``, which have to move together: a strength threshold makes aggregates
    *smaller*, so it enlarges the coarse grid, and the coarsest level is inverted densely.

    Parameters
    ----------
    **settings
        Forwarded to :class:`NativeSimpleInverse`.

    Returns
    -------
    callable
        ``(block, n_group_fields) -> NativeSimpleInverse``, the shape
        :func:`build_block_triangular_field_split` expects.
    """

    def build(block: sp.spmatrix, n_group_fields: int) -> object:
        return NativeSimpleInverse(block, n_group_fields, **settings)

    return build
