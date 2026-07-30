"""Unit tests for the monolithic ILUT preconditioner (host core + JAX callback wrapper)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.solve.ilut_preconditioner import (
    MonolithicIlutPreconditioner,
    cell_major_permutation,
    factorize_ilut,
)
from aquaflux.solve.sparse_jacobian import block_stencil_colouring


def _diagonally_dominant_block_matrix(n, n_fields, rng):
    """A field-major block-tridiagonal matrix, made diagonally dominant so ILU is well posed."""
    rows, cols, vals = [], [], []
    for i in range(n):
        neighbours = [i] + ([i - 1] if i > 0 else []) + ([i + 1] if i < n - 1 else [])
        for j in neighbours:
            block = 0.2 * rng.standard_normal((n_fields, n_fields))
            if i == j:
                block += (n_fields + 3.0) * np.eye(n_fields)  # dominant diagonal block
            for a in range(n_fields):
                for b in range(n_fields):
                    rows.append(a * n + i)
                    cols.append(b * n + j)
                    vals.append(block[a, b])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_fields * n, n_fields * n)).tocsr()


def test_cell_major_permutation_is_a_valid_involution_free_permutation():
    perm = cell_major_permutation(4, 3)
    assert sorted(perm.tolist()) == list(range(12))
    # (cell i, field f) at cell-major i*3+f maps to field-major f*4+i
    assert perm[1 * 3 + 2] == 2 * 4 + 1


def test_ilut_inverts_the_matrix_with_ample_fill():
    rng = np.random.default_rng(0)
    n, nf = 15, 3
    a = _diagonally_dominant_block_matrix(n, nf, rng)
    factors = factorize_ilut(a, nf, fill_factor=100.0, drop_tol=1e-12)
    x = rng.standard_normal(nf * n)
    # M A x ~= x  (near-complete factorization)
    assert np.linalg.norm(factors.apply(a @ x) - x) / np.linalg.norm(x) < 1e-8


def test_transpose_apply_satisfies_the_adjoint_identity():
    rng = np.random.default_rng(1)
    n, nf = 12, 4
    a = _diagonally_dominant_block_matrix(n, nf, rng)
    factors = factorize_ilut(a, nf, fill_factor=100.0, drop_tol=1e-12)
    x = rng.standard_normal(nf * n)
    y = rng.standard_normal(nf * n)
    # <M^T x, y> == <x, M y>
    lhs = float(factors.apply(x, transpose=True) @ y)
    rhs = float(x @ factors.apply(y))
    assert abs(lhs - rhs) / (abs(rhs) + 1e-30) < 1e-10


def test_transpose_solves_the_transposed_system():
    rng = np.random.default_rng(2)
    n, nf = 12, 3
    a = _diagonally_dominant_block_matrix(n, nf, rng)
    factors = factorize_ilut(a, nf, fill_factor=100.0, drop_tol=1e-12)
    x = rng.standard_normal(nf * n)
    # M^T (A^T x) ~= x
    assert np.linalg.norm(factors.apply(a.T @ x, transpose=True) - x) / np.linalg.norm(x) < 1e-8


def test_rejects_non_square_or_mismatched_size():
    with pytest.raises(ValueError):
        factorize_ilut(sp.csr_matrix((6, 5)), 3)
    with pytest.raises(ValueError):
        factorize_ilut(sp.identity(7).tocsr(), 3)  # 7 not a multiple of 3


def test_jax_wrapper_matches_the_host_core_and_jits():
    rng = np.random.default_rng(3)
    n, nf = 10, 3
    a = _diagonally_dominant_block_matrix(n, nf, rng)
    pc = MonolithicIlutPreconditioner(factorize_ilut(a, nf, fill_factor=100.0, drop_tol=1e-12))
    r = rng.standard_normal(nf * n)
    m = pc.matvec()
    mt = pc.matvec(transpose=True)
    # the callback matches the host apply, and runs under jit
    np.testing.assert_allclose(
        np.asarray(jax.jit(m)(jnp.asarray(r))), pc.factors.apply(r), rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(jax.jit(mt)(jnp.asarray(r))), pc.factors.apply(r, transpose=True), rtol=1e-12
    )


def test_build_materializes_and_factors_from_a_matvec():
    """`build` should materialize J via colouring, shift, and factor — recovering a known matrix."""
    rng = np.random.default_rng(4)
    n, nf = 12, 3
    a = _diagonally_dominant_block_matrix(n, nf, rng)  # block-tridiagonal => reach 1
    colouring = block_stencil_colouring(np.arange(n - 1), np.arange(1, n), n, reach=1)
    shift = np.zeros(nf * n)
    pc = MonolithicIlutPreconditioner.build(
        lambda v: jnp.asarray(a @ np.asarray(v)),
        colouring,
        nf,
        shift,
        fill_factor=100.0,
        drop_tol=1e-12,
    )
    x = rng.standard_normal(nf * n)
    assert np.linalg.norm(pc.factors.apply(a @ x) - x) / np.linalg.norm(x) < 1e-8


def test_refresh_in_place_repreconditions_the_same_compiled_matvec():
    """`refresh_in_place` re-factors at a new operator and the SAME already-jitted matvec picks it up.

    This is the forward-march cheap refresh: the compiled Krylov solve holds the preconditioner by
    identity and the callback reads ``self.factors`` at call time, so mutating the factorization
    re-preconditions the existing compiled matvec with no recompile.
    """
    rng = np.random.default_rng(5)
    n, nf = 12, 3
    a = _diagonally_dominant_block_matrix(n, nf, rng)
    b = _diagonally_dominant_block_matrix(
        n, nf, rng
    )  # a different operator, same sparsity structure
    colouring = block_stencil_colouring(np.arange(n - 1), np.arange(1, n), n, reach=1)
    shift = np.zeros(nf * n)
    pc = MonolithicIlutPreconditioner.build(
        lambda v: jnp.asarray(a @ np.asarray(v)),
        colouring,
        nf,
        shift,
        fill_factor=100.0,
        drop_tol=1e-12,
    )
    jm = jax.jit(pc.matvec())  # compile the matvec ONCE, against A's factorization
    x = rng.standard_normal(nf * n)
    # M_A (A x) ~= x
    assert np.linalg.norm(np.asarray(jm(jnp.asarray(a @ x))) - x) / np.linalg.norm(x) < 1e-8

    pc.refresh_in_place(
        lambda v: jnp.asarray(b @ np.asarray(v)),
        colouring,
        nf,
        shift,
        fill_factor=100.0,
        drop_tol=1e-12,
    )
    y = rng.standard_normal(nf * n)
    # The SAME jitted callable now preconditions B: M_B (B y) ~= y (the refresh is seen without a rebuild)
    assert np.linalg.norm(np.asarray(jm(jnp.asarray(b @ y))) - y) / np.linalg.norm(y) < 1e-8
    # ...and the factorization genuinely changed: applying M_B to (A x) does NOT recover x
    assert np.linalg.norm(np.asarray(jm(jnp.asarray(a @ x))) - x) / np.linalg.norm(x) > 1e-2
