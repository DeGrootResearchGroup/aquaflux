"""A traced multigrid for a velocity--pressure saddle block, smoothed by SIMPLE relaxation.

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
Block splitting generalizes it to a per-cell block, where the transpose in
``M_i = F_ii^T (R_i R_i^T)^-1`` is easy to drop and reduces to the scalar form at one field per cell, so a
scalar check cannot catch its absence. Every piece is formed on the device from the level operator's
values (:mod:`~aquaflux.precondition.simple_fit`), so a refit moves no level through the host.

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
import scipy.sparse as sp

from .entry_map import SparsePattern
from .hierarchy_inverse import HierarchyBlockInverse
from .multigrid import SmoothedHierarchy, _operator_matvec, _smoothed_ops
from .padded_rows import _PaddedRowOperator
from .simple_fit import (
    SimpleChecks,
    SimplePlan,
    SimpleValues,
    fit_simple_level,
    refuse_unusable,
    simple_plan,
)
from .traced_cycle import in_precision

__all__ = ["SimpleSmoothedInverse"]


class _SimplePieces(eqx.Module):
    """One level's SIMPLE relaxation pieces, all frozen and all traced.

    ``n_velocity`` is the split point in the level's field-major vector; everything else is either an
    elementwise reciprocal or a sparse operator, so a sweep is diagonal scalings and matrix-vector
    products and nothing else -- no factorization, no triangular solve, no sequential dependency.

    **An ``equinox.Module`` with only the split point static, on the same rule as
    :class:`~aquaflux.precondition.multigrid._SparseLevel`, and for the same reason.** That makes the
    record a pytree whose every array is a traced leaf, so it can be passed as an *argument* to the
    jitted cycle rather than captured by it. Re-deriving the pieces at a new operator on unchanged shapes
    then reuses the compiled cycle instead of tracing a new one -- which is what makes a mid-march
    refresh cheap, and is defeated outright by a closure, since a closed-over array is a compile-time
    constant and every refresh mints a new one.

    The Schur complement ``S = C - D M G`` is not held: :meth:`schur_apply` applies it from ``C``, ``D``
    and ``M G``. Formed, it reaches twice as far as the level operator and its rows are wider than its
    pieces' together, so holding it, with its transpose, cost more memory than the one piece it adds,
    ``C``; the product from the pieces reads about a third more entries than a formed one would.
    """

    f_diagonal_inverse: jnp.ndarray  # (n_velocity,) 1 / diag(F)
    dg: _PaddedRowOperator  # M G, the velocity correction operator (M approximates F^-1)
    divergence: _PaddedRowOperator  # D
    pressure: _PaddedRowOperator  # C, the pressure-pressure block
    schur_diagonal_inverse: jnp.ndarray  # (n_pressure,) the approximate inverse of diag(S)
    #: Set when the splitting is a per-cell BLOCK inverse instead of a scalar diagonal; the velocity
    #: predictor then applies this rather than an elementwise multiply. ``None`` keeps the diagonal path,
    #: which stays an elementwise multiply rather than a nine-nonzero-per-row matvec.
    f_block_inverse: _PaddedRowOperator | None
    #: Static: it slices the level vector, so it must be concrete.
    n_velocity: int = eqx.field(static=True)

    def schur_apply(self, p: jnp.ndarray) -> jnp.ndarray:
        """``S p = C p - D (M G p)``, shape ``(n_pressure,)``.

        Linear, and transposed through each piece's stored transpose. It sums in another order than a
        product with ``S`` formed, so the two differ in the last bits of the working precision.
        """
        return self.pressure.apply(p) - self.divergence.apply(self.dg.apply(p))


def _simple_pieces(
    a: sp.spmatrix,
    block_size: int,
    frobenius: bool = True,
    schur_frobenius: bool = False,
    block_splitting: bool = False,
    simplec: bool = False,
    report: Callable[[str], None] = lambda _message: None,
    precision: str = "float64",
) -> _SimplePieces:
    """Form one level's SIMPLE pieces from an assembled operator, through a plan made for it.

    Works on **any** level, including a Galerkin coarse operator, because it reads nothing but the
    matrix -- which is the property that decides whether a SIMPLE relaxation can be a smoother at all.
    The assembler-built Schur cannot: it needs the mesh, the Rhie--Chow coefficients and the boundary
    closures, none of which a coarse level has. :class:`SimpleSmoothedInverse` holds each level's plan
    and fits it at every refit; this makes one for a single operator.

    Returns
    -------
    _SimplePieces
        The traced record the smoother applies, in ``precision``.
    """
    a = sp.csr_matrix(a).sorted_indices()
    plan = simple_plan(
        SparsePattern.of(a),
        block_size,
        frobenius=frobenius,
        schur_frobenius=schur_frobenius,
        block_splitting=block_splitting,
        simplec=simplec,
    )
    values, checks = fit_simple_level(plan, _PaddedRowOperator.from_scipy(a).values)
    return _held_pieces(plan, values, checks, report, precision)


def _held_pieces(
    plan: SimplePlan,
    values: SimpleValues,
    checks: SimpleChecks,
    report: Callable[[str], None],
    precision: str,
) -> _SimplePieces:
    """A fitted level's pieces on its plan's layouts, in ``precision``, once its checks are clear.

    Rounded as soon as they are held, so a level's pieces are never all held in double beside their
    rounded copies.
    """
    refuse_unusable(checks)
    c_entries, d_entries, g_entries, schur_entries = plan.entry_counts
    # What the Schur would store formed, against the pieces it is applied from.
    report(
        f"      Schur {schur_entries / 1e6:.1f}M nnz if formed, applied from "
        f"{(c_entries + d_entries + g_entries) / 1e6:.1f}M as pieces "
        f"(C {c_entries / 1e6:.1f}M + D {d_entries / 1e6:.1f}M + G {g_entries / 1e6:.1f}M)",
    )
    if plan.schur_frobenius:
        low, median, high = (float(v) for v in checks.schur_relaxation)
        n_pressure = plan.schur_rows.shape[0]
        report(
            f"      Schur relaxation (Eq. 39): min {low:.3e} median {median:.3e} "
            f"max {high:.3e}  ({schur_entries / n_pressure:.0f} nnz/row)",
        )
    block_inverse = None
    if values.block_inverse is not None:
        block_inverse = plan.block_layout.with_values(*values.block_inverse)
    return in_precision(
        _SimplePieces(
            f_block_inverse=block_inverse,
            n_velocity=plan.n_velocity,
            f_diagonal_inverse=values.velocity_inverse,
            dg=plan.correction_layout.with_values(*values.correction),
            divergence=plan.divergence_layout.with_values(*values.divergence),
            pressure=plan.pressure_layout.with_values(*values.pressure),
            schur_diagonal_inverse=values.schur_inverse,
        ),
        precision,
    )


def _simple_correction(
    pieces: _SimplePieces, residual, pressure_sweeps: int, pressure_smoother_weight: float
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
    # vector: starting from p = 0, the update collapses to pressure_smoother_weight * S_diag^-1 * rhs. That is one of every
    # `pressure_sweeps` applications of the Schur complement (three sparse products), and it is not
    # folded away -- the products run at full cost against the zeros. Peeling it is bit-identical.
    if pressure_sweeps <= 0:
        pressure = jnp.zeros_like(rhs)
    else:
        pressure = pressure_smoother_weight * pieces.schur_diagonal_inverse * rhs

    # A LOOP, not a Python `for`, so the remaining sweeps enter the graph ONCE instead of being
    # unrolled into `pressure_sweeps - 1` copies of the Schur complement's products. Every sweep
    # has identical shapes, so there is nothing for unrolling to specialize on, and the traced graph is
    # what a refresh has to recompile whenever the coarsening moves. `fori_loop` at a static trip count
    # stays linearly transposable -- verified to give the same value AND the same transpose as the
    # unrolled form -- which the adjoint requires.
    def sweep(_, p):
        return p + pressure_smoother_weight * pieces.schur_diagonal_inverse * (
            rhs - pieces.schur_apply(p)
        )

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
    pressure_smoother_weight: float
    smoother_weight: float
    cycle_index: int
    pre_smooth: bool


@eqx.filter_jit
def _simple_smoothed_cycle(
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
    :meth:`~aquaflux.precondition.multigrid.FrozenCoarsening.fit` all exist to avoid. Closing over the
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

    def sweeps_from(level, rhs, guess, count):
        """``count`` relaxation sweeps from ``guess``, each correcting the residual it is left with.

        The two entry points below differ only in where they start and how many sweeps that leaves, so
        the sweep itself is written once here -- it was written twice, identically, with the trip count
        the only thing outside the closure that differed.

        Looped rather than unrolled, for the same reason as the inner pressure sweeps: the graph is what
        a refresh recompiles, and it is otherwise unrolled over levels x sweeps x inner sweeps. The level
        dimension has to stay unrolled (each level has its own shapes), so the two sweep dimensions are
        where the size actually comes from.
        """
        piece = pieces[level.n]

        def outer(_, g):
            correction = _simple_correction(
                piece,
                rhs - _operator_matvec(level, g),
                settings.pressure_sweeps,
                settings.pressure_smoother_weight,
            )
            return g + settings.smoother_weight * correction

        return jax.lax.fori_loop(0, count, outer, guess)

    def smooth(level, rhs, guess):
        return sweeps_from(level, rhs, guess, settings.sweeps)

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
        peeled = settings.smoother_weight * _simple_correction(
            piece, rhs, settings.pressure_sweeps, settings.pressure_smoother_weight
        )
        return sweeps_from(level, rhs, peeled, settings.sweeps - 1)

    ops = _smoothed_ops(
        smooth,
        cycle_index=settings.cycle_index,
        pre_smooth=settings.pre_smooth,
        smooth_zero=smooth_zero,
    )
    return hierarchy.fixed_cycle_solve(residual, settings.cycles, ops)


class SimpleSmoothedInverse(HierarchyBlockInverse):
    """A traced multigrid over the flow saddle whose LEVEL SMOOTHER is a SIMPLE relaxation.

    This is the arm every earlier one was not. Those replaced the flow block's preconditioner outright
    -- a flat block inverse with no hierarchy, no levels and no coarse-grid correction -- so their
    accuracy was capped by how well a single application approximates ``A^-1``, and for a SIMPLE-type
    method that cap is set by the Schur approximation's worst mode, the smooth global pressure mode.
    That is visible in the measurements as a residual floor almost insensitive to the pseudo-transient
    shift, which is the signature of an approximation ceiling rather than a conditioning one.

    Here the hierarchy is kept and SIMPLE replaces the incomplete-LU **smoother**. The division of
    labour is the point: the smoother only has to damp high-frequency error, where a local algebraic
    Schur is accurate, and the coarse solve carries the global mode, where it is not.

    Nothing in it is a host solver and nothing in it is sequential -- each level's pieces are formed on
    the device from the level operator's values (:func:`~aquaflux.precondition.simple_fit.fit_simple_level`,
    on a plan made once per coarsening) and the apply is diagonal scalings and sparse matrix-vector
    products in JAX.

    Parameters
    ----------
    block, n_fields, strength_threshold, max_levels, max_coarse, frozen_coarsening, shape_headroom, report
        See :class:`~aquaflux.precondition.hierarchy_inverse.HierarchyBlockInverse`, which owns the
        hierarchy, the in-place refresh and the host boundary. Only the smoother below belongs to this
        class.
    cycles, sweeps : int
        V-cycles per application, and SIMPLE sweeps per level. Both fixed, so ``b -> x`` stays linear.
    pressure_sweeps, pressure_smoother_weight : int, float
        The damped-Jacobi sweeps used for the Schur solve inside one SIMPLE sweep, and their damping.
    smoother_weight : float
        Relaxation applied to the whole SIMPLE correction.
    frobenius : bool
        Approximate the velocity block's inverse by the Frobenius-optimal diagonal
        ``F_ii / ||F_i||^2`` rather than the Jacobi diagonal ``1 / F_ii``. Default ``True``.

        The Frobenius-optimal diagonal is Jacobi with a derived per-row under-relaxation, which is
        what an amplifying sweep lacks. The change is qualitative rather than one of degree: under
        Jacobi more sweeps make the residual worse, under this one they make it better.
    schur_frobenius : bool
        Use the Frobenius-optimal diagonal, rather than the Jacobi one, in the pressure-Schur
        relaxation as well. Default ``True``.

        It is why ``pressure_smoother_weight`` defaults to undamped. Measured as a 2x2 at four sweeps: with
        the Jacobi diagonal an undamped pressure sweep blows up (3.4e-01 against 6.8e-05), while
        with the Frobenius one removing the damping helps (6.8e-05 to 4.2e-05). The hand-set damping
        and the derived per-row relaxation do the same job, so stacking them over-damps.
    aggressive_levels : int
        How many leading levels aggregate over the squared graph, which coarsens much faster than
        ordinary aggregation. Default ``1``.

        Depth and coarsening rate are two halves of one choice. One aggressive level gives a roughly
        hundredfold jump in a single step, so one prolongation from a ~860-equation coarse space
        carries the whole error; more levels at a gentler rate ask less of each interpolation.
        Aggressive coarsening exists to bound the level count as the mesh grows, not to improve the
        coarse space. See :func:`~aquaflux.precondition.multigrid.build_convection_hierarchy`.
    orthonormal : bool
        Scale each aggregate's prolongation column to unit length, by ``1 / sqrt(|aggregate|)``.
        Default ``False``.

        This is inert on a two-level cycle with an exact coarse solve, but not deeper. With 0/1
        columns the Galerkin coarse operator picks up a scaling of order ``|agg_i| * |agg_j|`` per
        entry, which the next level's smoother and spectral estimate read, so the distortion
        compounds once per level in proportion to how uneven the aggregate sizes are.
    avoid_singletons : bool
        Attach a vertex whose neighbours are all already claimed to one of their aggregates, instead
        of letting it form an aggregate of its own. Default ``False``.

        Such singletons are an artifact of arrival order in the random sweep, not of the graph, and
        they interact badly with ``orthonormal``, which promotes a singleton several-fold against a
        real aggregate. See :func:`~aquaflux.precondition.multigrid.build_convection_hierarchy`.
    block_splitting : bool
        Replace the velocity predictor's scalar diagonal with a per-cell block inverse. Default
        ``False``. It cannot be combined with ``simplec``.

        The splitting error is the larger half of what governs this preconditioner's eigenvalue
        clustering at the fine level, and a diagonal cannot represent a cell's own coupling between
        velocity components at all.
    simplec : bool
        Use SIMPLEC's velocity coefficient, dividing by the row sum rather than the diagonal.
        Default ``False``. It cannot be combined with ``block_splitting``.

        The appeal as a smoother is that the splitting error then annihilates constants exactly, so
        it does not fight the coarse grid over the smoothest mode.
    cycle_index : int
        How many times each coarse level is visited per visit of its parent: ``1`` is a V-cycle,
        ``2`` a W-cycle. Default ``1``.

        A W-cycle buys convergence with coarse work, where every other setting here buys it with
        fine-level relaxation, and the fine level is ~60% of the smoothing cost, so the two are
        priced very differently.
    pre_smooth : bool
        Relax before the coarse-grid correction as well as after it. Default ``True``.

        Dropping the pre-relaxation also drops the residual matrix-vector product that follows it,
        which at two inner pressure sweeps is the single largest term in a sweep.
    prolongation_smoothing : {"none", "symmetric-part", "standard"}
        How the coarse correction is interpolated back. ``"none"`` (default) injects it
        piecewise-constant over each aggregate; the other two smooth the prolongator once with the
        operator.

        Unsmoothed aggregation is the standard explanation for a hierarchy that works at two levels
        and gains nothing deeper, since the interpolation error does not fall as the grids coarsen.
        Smoothing the prolongator is what makes aggregation multigrid depth-independent.
        ``"standard"`` presumes a unit-magnitude diagonal, so pair it with ``equilibrate``. Passed
        to :func:`~aquaflux.precondition.multigrid.build_convection_hierarchy`.
    equilibrate : bool
        Coarsen the operator rescaled to a unit-magnitude diagonal, rather than as given. The answer
        is unchanged. Default ``False``.

        Every step of the setup reads the diagonal, so on an operator whose diagonal spans orders of
        magnitude it is otherwise calibrated against a scale with no meaning. Passed to
        :func:`~aquaflux.precondition.multigrid.build_convection_hierarchy`.
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
        pressure_smoother_weight: float = 1.0,
        smoother_weight: float = 0.7,
        frobenius: bool = True,
        schur_frobenius: bool = True,
        aggressive_levels: int = 1,
        orthonormal: bool = False,
        avoid_singletons: bool = False,
        # Replace the velocity predictor's scalar diagonal with a per-cell block inverse. The splitting
        # error is the larger half of what governs this preconditioner's eigenvalue clustering at the
        # fine level, and a diagonal cannot represent a cell's own velocity-component coupling at all.
        block_splitting: bool = False,
        # SIMPLEC's velocity coefficient: divide by the row sum rather than the diagonal. The appeal as
        # a smoother is that the splitting error then annihilates constants exactly, so it does not
        # fight the coarse grid over the smoothest mode.
        simplec: bool = False,
        # A W-cycle visits each coarse level twice per visit of its parent. It buys convergence with
        # COARSE work, where every other lever here buys it with fine-level relaxation -- and the fine
        # level is ~60% of the smoothing cost, so the two are priced very differently.
        cycle_index: int = 1,
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
        **hierarchy_settings,
    ) -> None:
        self._frobenius = frobenius
        self._schur_frobenius = schur_frobenius
        self._block_splitting = block_splitting
        self._simplec = simplec
        self._aggressive_levels = aggressive_levels
        self._orthonormal = orthonormal
        self._avoid_singletons = avoid_singletons
        self._prolongation_smoothing = prolongation_smoothing
        self._equilibrate = equilibrate
        # Each smoothed level's plan, and the level operators' index arrays they were made for: a
        # frozen refit keeps those very arrays, a rebuild makes new ones (see `_simple_plans`).
        self._plans: tuple[SimplePlan | None, ...] = ()
        self._plans_for: tuple = ()
        # The cycle's static half, built once: it never changes over this inverse's life, so a refresh
        # cannot move the compilation key through it.
        self._smoother = _SmootherSettings(
            cycles=cycles,
            sweeps=sweeps,
            pressure_sweeps=pressure_sweeps,
            pressure_smoother_weight=pressure_smoother_weight,
            smoother_weight=smoother_weight,
            cycle_index=cycle_index,
            pre_smooth=pre_smooth,
        )
        super().__init__(block, n_fields, **hierarchy_settings)

    def build_settings(self) -> dict:
        """This family coarsens by a randomized maximal independent set, on the settings above."""
        return {
            "mis_aggregation": True,
            # Depth and coarsening RATE are two halves of one choice. One aggressive level gives a
            # roughly hundredfold jump in a single step, so one prolongation from a ~860-equation
            # coarse space carries the whole error; more levels at a gentler rate ask less of each
            # interpolation.
            "aggressive_levels": self._aggressive_levels,
            "orthonormal_prolongation": self._orthonormal,
            "avoid_singletons": self._avoid_singletons,
            "prolongation_smoothing": self._prolongation_smoothing,
            "equilibrate": self._equilibrate,
        }

    def smoother(self) -> _SmootherSettings:
        return self._smoother

    def cycle(self):
        return _simple_smoothed_cycle

    def derive_extras(self, hierarchy) -> dict:
        """The per-level SIMPLE pieces over ``hierarchy``, and the build record.

        Returned as a plain dict keyed by level size, which is a pytree: the cycle takes it as an
        argument, so re-deriving it on unchanged shapes leaves the compiled cycle a cache hit.
        """
        smoother = self._smoother
        # Keyed by level size: the V-cycle recursion is unrolled at trace time, so the smoother is
        # handed the concrete level object and can look its pieces up by a static attribute. The
        # coarsest level solves directly and needs none.
        pieces = {}
        for level, plan in zip(hierarchy.levels, self._simple_plans(hierarchy), strict=True):
            if plan is None:
                continue
            values, checks = fit_simple_level(plan, level.operator.values)
            pieces[level.n] = _held_pieces(
                plan, values, checks, self._report, self._stored_precision
            )
            del values
        sizes = ", ".join(
            f"level {n}: S {p.pressure.shape[0]} dofs / "
            f"{(p.pressure.nnz + p.divergence.nnz + p.dg.nnz) / 1e6:.1f}M nnz as C, D, M G"
            for n, p in pieces.items()
        )
        # Report the COARSE GRID SIZE and the coarsening ratio beside the level count. A level count
        # alone cannot distinguish a hierarchy that stopped because it hit its cap from one that
        # stopped because it reached its coarse limit, and it says nothing about whether a single
        # aggregation had to represent the error across a hundredfold jump.
        #
        # These come from THIS inverse's own last coarsening, into a list `_coarsen` hands the builder
        # for exactly this build -- never a shared accumulator another hierarchy could have written to
        # since. A frozen-coarsening refresh fits the held coarsening rather than calling `_coarsen`, so
        # it aggregates nothing new; `self._aggregate_stats` then still holds this same inverse's own
        # last real aggregation, which is the honest thing to report (a frozen coarsening still HAS
        # aggregates, just not new ones).
        for depth, stat in enumerate(self._aggregate_stats):
            self._report(
                f"      aggregates level {depth}: {stat['aggregates']} of size "
                f"{stat['min']}/{stat['median']:.0f}/{stat['max']} (min/med/max), "
                f"spread {stat['spread']:.0f}x, {stat['singletons']} singletons",
            )
        # Print what was actually PARSED, not just what was built. A spec-token collision once made an
        # arm run forty sweeps instead of four while every other line of output looked correct.
        self._report(
            f"      smoother: {smoother.sweeps} sweeps x {smoother.pressure_sweeps} inner, "
            f"smoother_weight {smoother.smoother_weight}, simplec {self._simplec}, "
            f"pressure_smoother_weight {smoother.pressure_smoother_weight}, "
            f"strength {self._build_settings['strength_threshold']}, "
            f"block splitting {self._block_splitting}",
        )
        coarse = hierarchy.levels[-1]
        self._report(
            f"      SIMPLE-smoothed hierarchy: {len(hierarchy.levels)} levels, "
            f"fine {self._n_dofs} -> coarse {coarse.n} dofs "
            f"({self._n_dofs / max(coarse.n, 1):.0f}x, direct solve), {sizes}",
        )

        return pieces

    def _simple_plans(self, hierarchy: SmoothedHierarchy) -> tuple[SimplePlan | None, ...]:
        """Each level's SIMPLE plan (``None`` on the coarsest, which solves directly), made once.

        A plan is the level's structure, so it stays valid while the level operator keeps its pattern.
        A frozen coarsening's refit assembles every level operator on its layout's own index arrays,
        so those arrays -- compared by identity -- say whether the held plans still apply; a rebuild
        makes new ones, and the plans are made again.
        """
        columns = tuple(
            array for level in hierarchy.levels for array in jax.tree.leaves(level.operator.rows)
        )
        held = len(columns) == len(self._plans_for) and all(
            a is b for a, b in zip(columns, self._plans_for, strict=True)
        )
        if not held:
            self._plans = tuple(
                None
                if level.coarse_inv is not None
                else simple_plan(
                    SparsePattern.of(level.operator.to_scipy()),
                    level.block_size,
                    level.operator.width,
                    frobenius=self._frobenius,
                    schur_frobenius=self._schur_frobenius,
                    block_splitting=self._block_splitting,
                    simplec=self._simplec,
                )
                for level in hierarchy.levels
            )
            self._plans_for = columns
        return self._plans
