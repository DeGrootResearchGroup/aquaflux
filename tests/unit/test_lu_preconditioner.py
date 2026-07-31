"""Unit tests for the monolithic complete-LU preconditioner (host factorization + apply + refactor).

The complete-LU counterpart of the ILUT preconditioner: a *complete* factorization, so its apply is the
operator's exact inverse (a Krylov solve converges in one iteration). These exercise the host factorization
directly on small matrices with the always-available SciPy (SuperLU) backend -- no coupled solve, no
optional dependency -- checking the exact forward/transpose solves and the in-place numeric refactor.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import numpy as np
import scipy.sparse as sp
from aquaflux.solve.lu_preconditioner import _umfpack_available, factorize_lu


def _nonsymmetric_system(n=120, seed=0):
    rng = np.random.default_rng(seed)
    a = sp.random(n, n, density=0.06, random_state=seed).tocsr() + sp.eye(n) * 6.0
    b = rng.standard_normal(n)
    return a.tocsr(), b


def test_complete_lu_apply_is_the_exact_inverse() -> None:
    """A complete LU applies ``A^{-1}`` exactly: ``A (M b) = b`` to machine precision (forward solve)."""
    a, b = _nonsymmetric_system()
    factors = factorize_lu(a, backend="scipy")
    x = factors.apply(b)
    assert np.linalg.norm(a @ x - b) / np.linalg.norm(b) < 1e-12


def test_complete_lu_transpose_apply_solves_the_transposed_system() -> None:
    """``apply(transpose=True)`` applies ``M^T = A^{-T}`` -- the adjoint transpose solve."""
    a, b = _nonsymmetric_system(seed=1)
    factors = factorize_lu(a, backend="scipy")
    xt = factors.apply(b, transpose=True)
    assert np.linalg.norm(a.T @ xt - b) / np.linalg.norm(b) < 1e-12


def test_refactor_tracks_new_values_on_the_same_pattern() -> None:
    """Refactoring at new values (same sparsity) makes the apply solve the NEW system, not the old one."""
    a, b = _nonsymmetric_system(seed=2)
    factors = factorize_lu(a, backend="scipy")
    a2 = a.copy()
    a2.data = a2.data * np.repeat(
        np.random.default_rng(3).uniform(0.5, 2.0, a.shape[0]), np.diff(a.indptr)
    )
    factors.backend.refactor(a2)
    x = factors.apply(b)
    assert np.linalg.norm(a2 @ x - b) / np.linalg.norm(b) < 1e-12  # solves the refactored system
    assert np.linalg.norm(a @ x - b) / np.linalg.norm(b) > 1e-3  # ... and NOT the original


def test_monolithic_lu_preconditioner_matvec_and_refresh() -> None:
    """The JAX wrapper's matvec applies the exact inverse through a callback, and refreshes in place.

    Built from a small ``matvec`` + colouring like the coupled builder does; the block structure is a
    2-cell 2-field system so the stencil colouring machinery is exercised without a mesh.
    """
    import jax
    import jax.numpy as jnp
    from aquaflux.solve import MonolithicLuPreconditioner
    from aquaflux.solve.sparse_jacobian import block_stencil_colouring

    n_cells, n_fields = 40, 2
    owner = np.arange(n_cells - 1)
    nb = np.arange(1, n_cells)  # a 1D chain of cells
    colouring = block_stencil_colouring(owner, nb, n_cells, 2)
    dof = n_cells * n_fields
    # A block-TRIDIAGONAL operator matching the chain stencil (so the coloured probe materializes it
    # exactly): each cell a dense n_fields x n_fields block, coupled to its chain neighbours, diagonally
    # dominant. Field-major layout, dof (cell i, field f) = f * n_cells + i.
    rng = np.random.default_rng(5)
    a = sp.lil_matrix((dof, dof))
    for i in range(n_cells):
        for j in (i - 1, i, i + 1):
            if 0 <= j < n_cells:
                blk = rng.standard_normal((n_fields, n_fields))
                if i == j:
                    blk += np.eye(n_fields) * 8.0
                for fi in range(n_fields):
                    for fj in range(n_fields):
                        a[fi * n_cells + i, fj * n_cells + j] = blk[fi, fj]
    a = a.tocsr()

    def matvec(v):
        return jnp.asarray(a @ np.asarray(v))

    shift = np.zeros(dof)
    pc = MonolithicLuPreconditioner.build(matvec, colouring, n_fields, shift, backend="scipy")
    b = jnp.asarray(np.random.default_rng(6).standard_normal(dof))
    x = jax.jit(pc.matvec())(b)
    assert float(jnp.linalg.norm(jnp.asarray(a @ np.asarray(x)) - b) / jnp.linalg.norm(b)) < 1e-10
    # refresh in place at a scaled operator: the same object now inverts the new matvec
    factors_before = pc.factors.backend
    pc.refresh_in_place(lambda v: 2.0 * matvec(v), colouring, n_fields, shift)
    assert pc.factors.backend is factors_before  # same backend object, refactored in place
    x2 = jax.jit(pc.matvec())(b)
    assert (
        float(jnp.linalg.norm(jnp.asarray(2.0 * (a @ np.asarray(x2))) - b) / jnp.linalg.norm(b))
        < 1e-10
    )


def test_auto_backend_selects_a_working_factorization() -> None:
    """``backend='auto'`` yields a working exact factorization whether or not UMFPACK is present."""
    a, b = _nonsymmetric_system(seed=7)
    factors = factorize_lu(a, backend="auto")
    x = factors.apply(b)
    assert np.linalg.norm(a @ x - b) / np.linalg.norm(b) < 1e-10
    # a smoke check that the availability probe runs without raising
    assert isinstance(_umfpack_available(), bool)
