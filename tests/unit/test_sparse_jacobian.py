"""Unit tests for the compressed (graph-coloured) sparse-Jacobian materialization."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.solve.sparse_jacobian import (
    block_stencil_colouring,
    block_stencil_gather_map,
    jacobian_relative_error,
    materialize_block_jacobian,
)


def _path_graph(n):
    """A 1-D chain of cells: edges ``(i, i+1)``."""
    owner = np.arange(n - 1)
    nb = np.arange(1, n)
    return owner, nb


def test_pattern_reach_one_is_the_adjacency():
    owner, nb = _path_graph(5)
    c = block_stencil_colouring(owner, nb, 5, reach=1)
    pattern = {(int(i), int(j)) for i, j in zip(c.pattern_rows, c.pattern_cols, strict=True)}
    # each cell couples to itself and its chain neighbours only
    expected = (
        {(i, i) for i in range(5)} | {(i, i + 1) for i in range(4)} | {(i + 1, i) for i in range(4)}
    )
    assert pattern == expected


def test_pattern_reach_two_adds_next_nearest_neighbours():
    owner, nb = _path_graph(5)
    c1 = block_stencil_colouring(owner, nb, 5, reach=1)
    c2 = block_stencil_colouring(owner, nb, 5, reach=2)
    p1 = {(int(i), int(j)) for i, j in zip(c1.pattern_rows, c1.pattern_cols, strict=True)}
    p2 = {(int(i), int(j)) for i, j in zip(c2.pattern_rows, c2.pattern_cols, strict=True)}
    assert p1 < p2  # strict superset
    assert (0, 2) in p2 and (0, 2) not in p1  # distance-2 coupling appears


def test_colouring_is_collision_free():
    """Two cells sharing a colour must not both be reachable from any common row cell."""
    rng = np.random.default_rng(0)
    n = 40
    # a random-ish connected graph (a chain plus a few chords)
    owner = np.concatenate([np.arange(n - 1), rng.integers(0, n, 6)])
    nb = np.concatenate([np.arange(1, n), rng.integers(0, n, 6)])
    keep = owner != nb
    c = block_stencil_colouring(owner[keep], nb[keep], n, reach=2)
    pattern = sp.csr_matrix(
        (np.ones(len(c.pattern_rows)), (c.pattern_rows, c.pattern_cols)), shape=(n, n)
    )
    # rows coupled to each column
    for colour in range(c.n_colours):
        cols = np.where(c.colour == colour)[0]
        for row in range(n):
            coupled = set(pattern.indices[pattern.indptr[row] : pattern.indptr[row + 1]].tolist())
            assert len(coupled & set(cols.tolist())) <= 1, "colour collision on a row"


def _random_block_matrix(n, nvar, reach, rng):
    """A field-major block-sparse matrix with the given stencil reach; DOF (i,f) -> f*n + i."""
    owner, nb = _path_graph(n)
    c = block_stencil_colouring(owner, nb, n, reach=reach)
    rows, cols, vals = [], [], []
    for i, j in zip(c.pattern_rows, c.pattern_cols, strict=True):
        block = rng.standard_normal((nvar, nvar))
        for a in range(nvar):
            for b in range(nvar):
                rows.append(a * n + i)
                cols.append(b * n + j)
                vals.append(block[a, b])
    return sp.csr_matrix((vals, (rows, cols)), shape=(nvar * n, nvar * n)).tocsr(), c


def _matvec_of(k):
    def matvec(v):
        return jnp.asarray(k @ np.asarray(v))

    return matvec


def test_materialize_recovers_a_known_block_matrix_exactly():
    rng = np.random.default_rng(1)
    n, nvar, reach = 12, 3, 2
    k, colouring = _random_block_matrix(n, nvar, reach, rng)
    materialized = materialize_block_jacobian(_matvec_of(k), colouring, nvar)
    assert (materialized - k).nnz == 0 or abs(materialized - k).max() < 1e-12


def test_batched_probing_matches_the_per_probe_loop():
    """Batched probing (the coloured probes share one linearization, applied to stacked seeds) recovers
    exactly the per-probe loop's Jacobian -- it is a pure speedup, not an approximation. Checked both in
    one batch and chunked (so the final-chunk padding is exercised)."""
    import jax

    rng = np.random.default_rng(3)
    n, nvar, reach = 12, 3, 2
    k, colouring = _random_block_matrix(n, nvar, reach, rng)
    dense = jnp.asarray(k.toarray())

    def matvec(v):  # pure-jax, so vmap-able (unlike the NumPy `_matvec_of`)
        return dense @ v

    loop = materialize_block_jacobian(matvec, colouring, nvar)
    one_batch = materialize_block_jacobian(matvec, colouring, nvar, batched_matvec=jax.vmap(matvec))
    chunked = materialize_block_jacobian(
        matvec, colouring, nvar, batched_matvec=jax.vmap(matvec), probe_batch_size=4
    )
    assert abs(loop - one_batch).max() < 1e-14  # bit-identical to the loop
    assert abs(loop - chunked).max() < 1e-14  # chunking + final-chunk padding changes nothing
    assert abs(loop - k).max() < 1e-12  # and both recover the true matrix


def test_gather_de_compression_matches_the_scatter_loop():
    """The cached-structure gather (one vectorized ``responses.ravel()[gather_map]`` into the fixed
    full-pattern CSR) recovers the same matrix as the scatter loop, and the map is state-independent -- it
    depends only on the colouring, so a *different* operator on the same pattern materializes correctly with
    the same precomputed structure (what makes it reusable across every refresh)."""
    import jax

    n, nvar, reach = 12, 3, 2
    k, colouring = _random_block_matrix(n, nvar, reach, np.random.default_rng(4))
    structure = block_stencil_gather_map(colouring, nvar)

    def matvec_of_dense(dense):
        d = jnp.asarray(dense)
        return lambda v: d @ v

    mv = matvec_of_dense(k.toarray())
    loop = materialize_block_jacobian(mv, colouring, nvar)
    gathered = materialize_block_jacobian(
        mv, colouring, nvar, batched_matvec=jax.vmap(mv), structure=structure
    )
    assert (
        abs(loop - gathered).max() < 1e-14
    )  # same matrix (gather keeps explicit zeros; values identical)

    # State-independent: a DIFFERENT operator on the SAME pattern materializes with the SAME structure/map.
    k2, _ = _random_block_matrix(n, nvar, reach, np.random.default_rng(5))
    mv2 = matvec_of_dense(k2.toarray())
    loop2 = materialize_block_jacobian(mv2, colouring, nvar)
    gathered2 = materialize_block_jacobian(
        mv2, colouring, nvar, batched_matvec=jax.vmap(mv2), structure=structure
    )
    assert abs(loop2 - gathered2).max() < 1e-14


def test_jacobian_relative_error_flags_too_small_a_reach():
    rng = np.random.default_rng(2)
    n, nvar = 12, 2
    k, _ = _random_block_matrix(n, nvar, reach=2, rng=rng)  # a distance-2 operator
    matvec = _matvec_of(k)
    # colouring at the true reach -> exact
    good = block_stencil_colouring(*_path_graph(n), n, reach=2)
    assert jacobian_relative_error(materialize_block_jacobian(matvec, good, nvar), matvec) < 1e-12
    # too-small reach -> misses the distance-2 couplings, large error
    short = block_stencil_colouring(*_path_graph(n), n, reach=1)
    assert jacobian_relative_error(materialize_block_jacobian(matvec, short, nvar), matvec) > 1e-3


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        block_stencil_colouring(np.array([0]), np.array([1]), 0, reach=1)
    with pytest.raises(ValueError):
        block_stencil_colouring(np.array([0]), np.array([1]), 2, reach=0)
    with pytest.raises(ValueError):
        block_stencil_colouring(np.array([0]), np.array([5]), 2, reach=1)  # endpoint out of range
