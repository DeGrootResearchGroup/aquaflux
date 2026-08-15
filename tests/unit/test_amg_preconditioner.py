"""Operator-level unit tests for the monolithic AMG V-cycle preconditioner.

The V-cycle machinery -- build, one-cycle apply, transpose apply, in-place refactor -- checked on a
well-understood symmetric-positive-definite model operator (a 2D five-point Laplacian), independent of the
coupled RANS saddle (whose convergence and adjoint are the integration test's job). Skipped where
``petsc4py`` is unavailable, since the V-cycle is built with PETSc.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("petsc4py")

from aquaflux.solve import (
    AmgVCycle,
    MonolithicAmgPreconditioner,
    build_amg_vcycle,
    equilibrate_cell_major,
)


def _laplacian_2d(n: int) -> sp.csr_matrix:
    """The 2D five-point Laplacian on an ``n x n`` grid (symmetric positive definite)."""
    ident = sp.identity(n)
    tri = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    return (sp.kron(ident, tri) + sp.kron(tri, ident)).tocsr()


def test_equilibrate_cell_major_balances_the_diagonal_and_reorders() -> None:
    """The equilibration makes every diagonal unit-magnitude; the permutation is cell-major."""
    # Two fields whose diagonals differ by ~100x, plus off-diagonal coupling, field-major on 3 cells.
    n_cells, n_fields = 3, 2
    field_scales = np.array([1.0, 100.0])
    diag = np.concatenate([np.full(n_cells, s) for s in field_scales])
    matrix = sp.diags(diag).tolil()
    matrix[0, 3] = matrix[3, 0] = 5.0  # cross-field coupling on cell 0
    matrix = matrix.tocsr()

    cell_major, scale, perm = equilibrate_cell_major(matrix, n_fields)
    assert np.allclose(np.abs(cell_major.diagonal()), 1.0)  # balanced
    # perm[i * n_fields + f] = f * n_cells + i -- interleaves the two fields per cell.
    assert np.array_equal(perm, np.array([0, 3, 1, 4, 2, 5]))
    # scale = 1/sqrt(|diag|), one per field-major dof.
    assert np.allclose(scale, 1.0 / np.sqrt(diag))


def test_amg_vcycle_reduces_the_residual() -> None:
    """One V-cycle is a genuine approximate inverse: it contracts the residual on the model Poisson."""
    a = _laplacian_2d(40)
    vcycle = build_amg_vcycle(a, n_fields=1)
    rng = np.random.default_rng(0)
    b = rng.standard_normal(a.shape[0])
    x = vcycle.apply(b)
    ratio = np.linalg.norm(a @ x - b) / np.linalg.norm(b)
    assert ratio < 0.7  # a single multigrid cycle removes most of the residual


def _with_stored_zeros(a: sp.csr_matrix, per_row: int = 3) -> sp.csr_matrix:
    """``a`` padded with exactly-zero entries at positions it does not already store.

    Imitates what the coloured probe hands the preconditioner: the *full* block-stencil pattern, whose
    out-of-reach and genuinely-uncoupled positions are stored as exact zeros. Built from coordinates so the
    zeros survive -- assembling through a sparse product or addition would drop them, which is the very
    behaviour the tests below exist to be independent of.
    """
    coo = a.tocoo()
    n = a.shape[0]
    extra_rows = np.repeat(np.arange(n), per_row)
    extra_cols = (extra_rows + np.tile(np.arange(2, 2 + per_row), n) * 7) % n
    rows = np.concatenate([coo.row, extra_rows])
    cols = np.concatenate([coo.col, extra_cols])
    data = np.concatenate([coo.data, np.zeros(extra_rows.shape[0])])
    padded = sp.csr_matrix((data, (rows, cols)), shape=a.shape)
    padded.sort_indices()
    return padded


def test_live_drops_stored_zeros_without_changing_the_operator() -> None:
    """Pruning the stored zeros changes the pattern and nothing else -- same matrix, fewer stored entries."""
    a = _laplacian_2d(12)
    padded = _with_stored_zeros(a)
    assert (padded.data == 0.0).sum() > 0, "fixture stores no zeros, so it tests nothing"

    live = AmgVCycle._live(padded)

    assert live.nnz < padded.nnz  # positions really were removed
    assert (live.data == 0.0).sum() == 0
    np.testing.assert_array_equal(live.toarray(), padded.toarray())  # identical operator


def test_the_vcycle_is_built_on_the_live_pattern_not_the_stored_one() -> None:
    """The incomplete-LU smoother must not be handed stored zeros -- they are fill slots for it.

    A stored exactly-zero entry changes the smoother's factorization without changing the operator, and on
    the three-dimensional coupled case at **zero** pseudo-transient shift that takes the V-cycle from
    converging in 11 restart cycles to stalling at a true relative residual of 2.3e-02. Zero shift is the
    operator the implicit-function-theorem adjoint solves, so no forward march ever visits it (the march
    floors its preconditioner at a positive shift) and no march can reveal the regression -- which is
    exactly why it is pinned here rather than left to the case.

    Neither of the two natural diagnoses detects it, so neither can serve as the gate: the fine-level
    pivots are identical with and without the stored zeros, and so is the coarse space's size.
    """
    a = _laplacian_2d(12)
    padded = _with_stored_zeros(a)
    stored_zeros = int((padded.data == 0.0).sum())
    assert stored_zeros > 0, "fixture stores no zeros, so it tests nothing"

    # Handed to the V-cycle DIRECTLY, with an identity equilibration and permutation, rather than through
    # `equilibrate_cell_major`. That keeps the test independent of which spelling the assemblers use: a
    # pruning one would strip the zeros before the V-cycle ever saw them and the test would silently
    # assert nothing, while a pattern-preserving one hands them straight through. This is the boundary
    # that has to stop them either way.
    identity_scale, identity_perm = np.ones(padded.shape[0]), np.arange(padded.shape[0])
    vcycle = AmgVCycle(
        padded, identity_scale, identity_perm, 1, smoother_fill_levels=0, smoother_sweeps=2
    )

    assert vcycle._data.shape[0] == padded.nnz - stored_zeros
    assert vcycle._data.shape[0] < padded.nnz  # the pruning is not vacuous
    assert not np.any(vcycle._data == 0.0)
    # and it still preconditions the operator it was built on
    rng = np.random.default_rng(5)
    b = rng.standard_normal(a.shape[0])
    assert np.linalg.norm(padded @ vcycle.apply(b) - b) / np.linalg.norm(b) < 0.7


def test_a_refactor_does_not_reintroduce_stored_zeros() -> None:
    """The refresh prunes on the same terms as the build, or the first refresh undoes the guard."""
    a = _laplacian_2d(12)
    vcycle = build_amg_vcycle(a, n_fields=1)
    padded = _with_stored_zeros((2.5 * a).tocsr())
    assert (padded.data == 0.0).sum() > 0, "fixture stores no zeros, so it tests nothing"
    identity_scale, identity_perm = np.ones(padded.shape[0]), np.arange(padded.shape[0])

    vcycle.refactor(padded, identity_scale, identity_perm)

    assert not np.any(vcycle._data == 0.0)
    rng = np.random.default_rng(6)
    b = rng.standard_normal(a.shape[0])
    assert np.linalg.norm(padded @ vcycle.apply(b) - b) / np.linalg.norm(b) < 0.7


def test_amg_vcycle_transpose_is_consistent() -> None:
    """``M^T`` matches ``M`` acting on the other side: ``<y, M x> == <M^T y, x>``."""
    a = _laplacian_2d(30)
    vcycle = build_amg_vcycle(a, n_fields=1)
    rng = np.random.default_rng(1)
    x = rng.standard_normal(a.shape[0])
    y = rng.standard_normal(a.shape[0])
    left = float(y @ vcycle.apply(x))
    right = float(vcycle.apply(y, transpose=True) @ x)
    assert abs(left - right) <= 1e-9 * (abs(left) + abs(right) + 1e-30)


def test_amg_vcycle_refactor_rebuilds_at_a_new_matrix() -> None:
    """``refactor`` rebuilds the V-cycle at a new (here rescaled) matrix and still preconditions it."""
    a = _laplacian_2d(30)
    vcycle = build_amg_vcycle(a, n_fields=1)
    scaled = (2.5 * a).tocsr()
    cell_major, scale, perm = equilibrate_cell_major(scaled, 1)
    vcycle.refactor(cell_major, scale, perm)
    rng = np.random.default_rng(2)
    b = rng.standard_normal(a.shape[0])
    x = vcycle.apply(b)
    assert np.linalg.norm(scaled @ x - b) / np.linalg.norm(b) < 0.7


def test_amg_refresh_shift_in_place_tracks_the_shift_reusing_the_jacobian() -> None:
    """A shift-only refresh re-preconditions ``J + new_shift`` by reusing the frozen ``J`` -- it takes only
    the shift (no matvec/colouring), so it cannot re-materialize, and it must actually re-precondition the
    NEW operator (the reused-Jacobian trade is real, not a no-op)."""
    jac = _laplacian_2d(30)  # the frozen SPD Jacobian, no shift
    n_dof = jac.shape[0]
    shift0 = np.full(n_dof, 1.0)
    shift1 = np.full(
        n_dof, 40.0
    )  # a very different shift -> the shift0 V-cycle mis-scales the coarse solve
    vcycle = build_amg_vcycle((jac + sp.diags(shift0)).tocsr(), n_fields=1)
    pc = MonolithicAmgPreconditioner(vcycle, jacobian_no_shift=jac, n_fields=1)

    rng = np.random.default_rng(0)
    b = rng.standard_normal(n_dof)
    a1 = (jac + sp.diags(shift1)).tocsr()
    x_stale = pc.factors.apply(b)  # V-cycle still built for shift0

    pc.refresh_shift_in_place(shift1)
    x_fresh = pc.factors.apply(b)

    assert not np.allclose(x_fresh, x_stale)  # the shift refresh changed the preconditioner
    assert (
        np.linalg.norm(a1 @ x_fresh - b) / np.linalg.norm(b) < 0.7
    )  # approx-inverse of J + shift1


def test_amg_refresh_shift_in_place_requires_a_cached_jacobian() -> None:
    """Without a materialized Jacobian (no build/refresh_in_place yet) the shift-only refresh raises."""
    vcycle = build_amg_vcycle(_laplacian_2d(10), n_fields=1)
    pc = MonolithicAmgPreconditioner(vcycle)  # no jacobian_no_shift cached
    with pytest.raises(RuntimeError, match="cached Jacobian"):
        pc.refresh_shift_in_place(np.ones(100))


def test_extra_options_reach_petsc_under_the_v_cycle_own_prefix() -> None:
    """The tuning seam delivers caller options to GAMG, and they win over the shipped defaults.

    The shipped bundle (zero-fill smoother, 4 sweeps, a 2000-equation direct coarse solve) is measured,
    but several standard aggregation options have never been swept on the coupled saddle. This is the
    seam a study varies them through, rather than editing the defaults or writing into the global PETSc
    options database behind the preconditioner's back -- which would leak across every other V-cycle in
    the process, since each carries its own prefix precisely to stay isolated.
    """
    from petsc4py import PETSc

    matrix = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(240, 240), format="csr")
    tuned = build_amg_vcycle(
        matrix,
        1,
        smoother_fill_levels=0,
        smoother_sweeps=2,
        coarse_eq_limit=20,
        extra_options={"pc_gamg_threshold": 0.5, "pc_gamg_agg_nsmooths": 0},
    )
    prefix = tuned._pc.getOptionsPrefix()
    opts = PETSc.Options()
    assert opts[prefix + "pc_gamg_threshold"] == "0.5"
    assert opts[prefix + "pc_gamg_agg_nsmooths"] == "0"
    # ...and an option the caller did NOT set keeps the shipped value.
    assert opts[prefix + "mg_levels_ksp_type"] == "richardson"


def test_no_extra_options_is_the_shipped_configuration() -> None:
    """``None`` (the default) must leave the measured bundle exactly as it is."""
    matrix = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(120, 120), format="csr")
    plain = build_amg_vcycle(
        matrix, 1, smoother_fill_levels=0, smoother_sweeps=2, coarse_eq_limit=20
    )
    explicit = build_amg_vcycle(
        matrix, 1, smoother_fill_levels=0, smoother_sweeps=2, coarse_eq_limit=20, extra_options=None
    )
    rhs = np.ones(120)
    np.testing.assert_allclose(plain.apply(rhs), explicit.apply(rhs), rtol=0, atol=0)
