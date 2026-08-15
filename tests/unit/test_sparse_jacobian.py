"""Unit tests for the compressed (graph-coloured) sparse-Jacobian materialization."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.solve.sparse_jacobian import (
    ColumnProbePlan,
    _index_dtype,
    block_stencil_colouring,
    block_stencil_gather_map,
    column_probe_plan,
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


def _lattice_graph(nx, ny, nz):
    """A 3-D hexahedral cell graph: face-neighbour edges on an ``nx * ny * nz`` lattice."""
    index = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    owner = np.concatenate([index[:-1].ravel(), index[:, :-1].ravel(), index[:, :, :-1].ravel()])
    nb = np.concatenate([index[1:].ravel(), index[:, 1:].ravel(), index[:, :, 1:].ravel()])
    return owner, nb, nx * ny * nz


def _degree_ordered_colouring(conflict, n):
    """The plain highest-degree-first greedy, as a baseline for the saturation ordering."""
    colour = np.full(n, -1, dtype=np.int64)
    for i in np.argsort(-np.diff(conflict.indptr)):
        used = set(colour[conflict.indices[conflict.indptr[i] : conflict.indptr[i + 1]]].tolist())
        c = 0
        while c in used:
            c += 1
        colour[i] = c
    return colour


def test_saturation_colouring_uses_fewer_colours_than_degree_ordering():
    """The colour count is the probe count, so the ordering is a cost decision, not a detail.

    On a three-dimensional cell graph at the reach a coupled RANS Jacobian needs, ordering by
    saturation beats ordering by degree — and every colour saved is one directional derivative per
    field off every materialize.
    """
    owner, nb, n = _lattice_graph(8, 8, 8)
    c = block_stencil_colouring(owner, nb, n, reach=3)

    pattern = sp.csr_matrix(
        (np.ones(len(c.pattern_rows)), (c.pattern_rows, c.pattern_cols)), shape=(n, n)
    )
    conflict = (pattern.T @ pattern).tocsr()
    baseline = int(_degree_ordered_colouring(conflict, n).max()) + 1

    assert c.n_colours < baseline, f"saturation {c.n_colours} did not beat degree {baseline}"
    # ...and it is still a valid colouring: no conflict edge joins two same-colour cells.
    coo = conflict.tocoo()
    off_diagonal = coo.row != coo.col
    assert not np.any(c.colour[coo.row[off_diagonal]] == c.colour[coo.col[off_diagonal]])


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


def _mixed_reach_matrix(n, column_reach, rng):
    """A chain-graph matrix in which column field ``b`` couples only within ``column_reach[b]``.

    The shape a per-column probing plan exists for: a coupled residual whose scalar columns close
    inside a shorter stencil than its velocity columns.
    """
    nvar = len(column_reach)
    rows, cols, vals = [], [], []
    for b, reach in enumerate(column_reach):
        for i in range(n):
            for j in range(max(0, i - reach), min(n, i + reach + 1)):
                for a in range(nvar):
                    rows.append(a * n + i)
                    cols.append(b * n + j)
                    vals.append(rng.standard_normal())
    return sp.csr_matrix((vals, (rows, cols)), shape=(nvar * n, nvar * n)).tocsr()


def test_per_column_plan_recovers_a_mixed_reach_matrix_exactly_and_more_cheaply():
    """Probing each column at its own reach is exact when the column really does close inside it."""
    n, column_reach = 14, (3, 1)
    k = _mixed_reach_matrix(n, column_reach, np.random.default_rng(7))
    owner, nb = _path_graph(n)

    uniform = ColumnProbePlan.uniform(block_stencil_colouring(owner, nb, n, 3), len(column_reach))
    tuned = column_probe_plan(owner, nb, n, column_reach, pattern_reach=3)

    full = materialize_block_jacobian(_matvec_of(k), uniform)
    short = materialize_block_jacobian(_matvec_of(k), tuned)

    assert tuned.n_probes < uniform.n_probes  # the point: fewer directional derivatives
    assert abs(short - full).max() < 1e-13  # ...for the same matrix
    assert abs(short - k).max() < 1e-13


def test_a_short_probed_column_zeroes_its_out_of_reach_entries():
    """An entry beyond its column's reach must be zeroed, NOT read from that column's response.

    The colouring is collision-free only for the reach it was built at, so a response can hold another
    cell's near coupling at that row. Gathering it into the far position would write a real value where
    the matrix has none — silently, and only on meshes where the two cells collide.
    """
    n = 12
    owner, nb = _path_graph(n)
    # Column 0 is probed at reach 1 but assembled into the reach-2 pattern; on a chain the reach-1
    # colouring puts cells 3 apart in one colour, so row i+1 sees both cell i (distance 1) and cell
    # i+3 (distance 2) — the collision this guards against.
    plan = column_probe_plan(owner, nb, n, (1, 2), pattern_reach=2)
    dense = np.zeros((2 * n, 2 * n))
    for i in range(n):  # give column 0 a strong distance-1 coupling to make a leak obvious
        for j in range(max(0, i - 1), min(n, i + 2)):
            dense[i, j] = 1.0 + i
    k = sp.csr_matrix(dense)

    materialized = materialize_block_jacobian(_matvec_of(k), plan).toarray()
    for i in range(n):  # every distance-2 entry of column 0 is outside its reach
        for j in (i - 2, i + 2):
            if 0 <= j < n:
                assert materialized[i, j] == 0.0, f"leaked into ({i}, {j})"


def test_the_gather_and_scatter_paths_agree_on_a_per_column_plan():
    """The cached-structure gather must zero the same entries the scatter path drops."""
    import jax

    n, column_reach = 12, (2, 1)
    k = _mixed_reach_matrix(n, column_reach, np.random.default_rng(11))
    owner, nb = _path_graph(n)
    plan = column_probe_plan(owner, nb, n, column_reach, pattern_reach=2)
    dense = jnp.asarray(k.toarray())

    def matvec(v):
        return dense @ v

    scatter = materialize_block_jacobian(matvec, plan)
    gather = materialize_block_jacobian(
        matvec,
        plan,
        batched_matvec=jax.vmap(matvec),
        structure=block_stencil_gather_map(plan),
    )
    assert abs(scatter - gather).max() < 1e-14
    assert abs(gather - k).max() < 1e-13


def test_materialize_recovers_a_known_block_matrix_exactly():
    rng = np.random.default_rng(1)
    n, nvar, reach = 12, 3, 2
    k, colouring = _random_block_matrix(n, nvar, reach, rng)
    materialized = materialize_block_jacobian(
        _matvec_of(k), ColumnProbePlan.uniform(colouring, nvar)
    )
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

    loop = materialize_block_jacobian(matvec, ColumnProbePlan.uniform(colouring, nvar))
    one_batch = materialize_block_jacobian(
        matvec, ColumnProbePlan.uniform(colouring, nvar), batched_matvec=jax.vmap(matvec)
    )
    chunked = materialize_block_jacobian(
        matvec,
        ColumnProbePlan.uniform(colouring, nvar),
        batched_matvec=jax.vmap(matvec),
        probe_batch_size=4,
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
    structure = block_stencil_gather_map(ColumnProbePlan.uniform(colouring, nvar))

    def matvec_of_dense(dense):
        d = jnp.asarray(dense)
        return lambda v: d @ v

    mv = matvec_of_dense(k.toarray())
    loop = materialize_block_jacobian(mv, ColumnProbePlan.uniform(colouring, nvar))
    gathered = materialize_block_jacobian(
        mv,
        ColumnProbePlan.uniform(colouring, nvar),
        batched_matvec=jax.vmap(mv),
        structure=structure,
    )
    assert (
        abs(loop - gathered).max() < 1e-14
    )  # same matrix (gather keeps explicit zeros; values identical)

    # State-independent: a DIFFERENT operator on the SAME pattern materializes with the SAME structure/map.
    k2, _ = _random_block_matrix(n, nvar, reach, np.random.default_rng(5))
    mv2 = matvec_of_dense(k2.toarray())
    loop2 = materialize_block_jacobian(mv2, ColumnProbePlan.uniform(colouring, nvar))
    gathered2 = materialize_block_jacobian(
        mv2,
        ColumnProbePlan.uniform(colouring, nvar),
        batched_matvec=jax.vmap(mv2),
        structure=structure,
    )
    assert abs(loop2 - gathered2).max() < 1e-14


def test_jacobian_relative_error_flags_too_small_a_reach():
    rng = np.random.default_rng(2)
    n, nvar = 12, 2
    k, _ = _random_block_matrix(n, nvar, reach=2, rng=rng)  # a distance-2 operator
    matvec = _matvec_of(k)
    # colouring at the true reach -> exact
    good = block_stencil_colouring(*_path_graph(n), n, reach=2)
    assert (
        jacobian_relative_error(
            materialize_block_jacobian(matvec, ColumnProbePlan.uniform(good, nvar)), matvec
        )
        < 1e-12
    )
    # too-small reach -> misses the distance-2 couplings, large error
    short = block_stencil_colouring(*_path_graph(n), n, reach=1)
    assert (
        jacobian_relative_error(
            materialize_block_jacobian(matvec, ColumnProbePlan.uniform(short, nvar)), matvec
        )
        > 1e-3
    )


def test_index_width_is_chosen_from_the_bound_and_widens_when_it_must():
    """The de-compression's index arrays narrow to 32-bit, but only where the bound allows it.

    They carry one entry per Jacobian nonzero and several are live at once while the map is assembled,
    so the width sets that build's peak — but a bound that does not fit must widen rather than wrap.
    Every mesh reached so far is far inside 32-bit, which is exactly why this needs pinning: the wide
    branch never fires in practice, and a wide branch that never fires is indistinguishable from one
    that is broken until the day something needs it.
    """
    limit = np.iinfo(np.int32).max
    assert _index_dtype(0) is np.int32
    assert _index_dtype(limit - 1) is np.int32
    # At and above the limit the narrow type can no longer hold every value up to the bound.
    assert _index_dtype(limit) is np.int64
    assert _index_dtype(4 * limit) is np.int64

    # A bound just inside the limit must still round-trip its largest value, which is the property the
    # width exists to protect: `np.int32(limit - 1)` is representable, one more is not.
    assert int(np.array(limit - 1, dtype=_index_dtype(limit - 1))) == limit - 1
    assert int(np.array(4 * limit, dtype=_index_dtype(4 * limit))) == 4 * limit


def test_gather_map_index_arrays_are_narrow_on_a_realistic_pattern():
    """A real pattern's map is assembled in 32-bit, and its scatter still lands the right values.

    The narrowing is only sound because every index is bounded before it is formed, so this pins the
    outcome (the dtype) together with the behaviour it must not disturb (what `scatter` writes).
    """
    owner, nb = _path_graph(24)
    plan = ColumnProbePlan.uniform(block_stencil_colouring(owner, nb, 24, 2), 3)
    gather = block_stencil_gather_map(plan)

    assert gather._source.dtype == np.int32
    assert gather._position.dtype == np.int32

    # The scatter must place each response element where the map says, unaffected by the width.
    nf = plan.n_fields * plan.n_cells
    responses = np.arange(plan.n_probes * nf, dtype=np.float64).reshape(plan.n_probes, nf)
    data = np.zeros(gather.nnz)
    for probe in range(plan.n_probes):
        gather.scatter(data, responses[probe : probe + 1], probe, 1)

    rows = np.repeat(np.arange(gather.indptr.shape[0] - 1), np.diff(gather.indptr))
    field, cell = np.divmod(gather.indices.astype(np.int64), plan.n_cells)
    expected = (np.asarray(plan.probe_base)[field] + plan.colour[field, cell]) * nf + rows
    written = data != 0.0
    assert np.array_equal(data[written], expected[written].astype(np.float64))


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        block_stencil_colouring(np.array([0]), np.array([1]), 0, reach=1)
    with pytest.raises(ValueError):
        block_stencil_colouring(np.array([0]), np.array([1]), 2, reach=0)
    with pytest.raises(ValueError):
        block_stencil_colouring(np.array([0]), np.array([5]), 2, reach=1)  # endpoint out of range
