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

from aquaflux.solve import MonolithicAmgPreconditioner, build_amg_vcycle
from aquaflux.solve.ilut_preconditioner import equilibrate_cell_major


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
