"""The SIMPLE smoother's level pieces fitted on the device (``precondition/simple_fit.py``).

Each test fits one level's pieces through a plan and checks them against the same quantities formed
independently with scipy and numpy on the host: the approximate velocity inverse (Jacobi, Frobenius,
SIMPLEC, and the per-cell block), the correction ``M G``, the pressure block ``C``, and the diagonal
approximate inverse of the Schur complement ``C - D M G``, which the fit forms from ``S``'s rows without
holding ``S``.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.precondition import simple_fit
from aquaflux.precondition.entry_map import SparsePattern
from aquaflux.precondition.padded_rows import _PaddedRowOperator
from aquaflux.precondition.simple_fit import fit_simple_level, refuse_unusable, simple_plan

N_CELLS, FIELDS = 30, 3


def _saddle(seed: int = 0, shape: str = "mixed") -> sp.csr_matrix:
    """A field-major four-field saddle on a ring of cells: a nonsymmetric velocity block whose cells
    couple their own components, a gradient, a divergence that is not its transpose, and a nonzero
    pressure block. Stored entries are shuffled within each row, so nothing relies on sorted input.

    ``shape`` decides which shortcuts the fit can take. ``"mixed"``: a cell's velocity rows store
    different columns (the Gram matrix is searched), while the divergence and the correction line up
    by cell (the Schur product sums each cell's fields first). ``"aligned"``: every velocity coupling
    reaches the same cells, so both shortcuts apply. ``"ragged"``: as ``"mixed"`` with one divergence
    entry left out, so the Schur product takes every entry on its own. ``"decoupled"``: no
    velocity-pressure coupling is stored at all.
    """
    rng = np.random.default_rng(seed)
    n = N_CELLS
    ring = np.arange(n)
    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(r)
        cols.append(c)
        vals.append(v)

    for f in range(FIELDS + 1):
        for g in range(FIELDS + 1):
            if f == g:
                add(f * n + ring, g * n + ring, 4.0 + rng.random(n))
                add(f * n + ring, g * n + (ring + 1) % n, -0.5 - 0.3 * rng.random(n))
                add(f * n + ring, g * n + (ring - 1) % n, -0.7 - 0.3 * rng.random(n))
            elif max(f, g) < FIELDS and shape == "aligned":
                add(f * n + ring, g * n + ring, 0.3 * rng.standard_normal(n))
                add(f * n + ring, g * n + (ring + 1) % n, 0.1 * rng.standard_normal(n))
                add(f * n + ring, g * n + (ring - 1) % n, 0.1 * rng.standard_normal(n))
            elif max(f, g) < FIELDS:
                add(f * n + ring, g * n + ring, 0.3 * rng.standard_normal(n))
                add(f * n + ring, g * n + (ring + 2) % n, 0.1 * rng.standard_normal(n))
            elif shape == "decoupled":
                continue
            else:
                add(f * n + ring, g * n + ring, 0.2 * rng.standard_normal(n))
                neighbour = g * n + (ring + 1) % n
                if shape == "ragged" and f == FIELDS and g == 1:
                    keep = ring != 3  # cell 3's divergence misses one field's neighbour
                    add((f * n + ring)[keep], neighbour[keep], 0.2 * rng.standard_normal(n - 1))
                else:
                    add(f * n + ring, neighbour, 0.2 * rng.standard_normal(n))
    a = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=((FIELDS + 1) * n,) * 2,
    )
    a.sum_duplicates()
    order = np.concatenate(
        [a.indptr[r] + rng.permutation(a.indptr[r + 1] - a.indptr[r]) for r in range(a.shape[0])]
    )
    return sp.csr_matrix((a.data[order], a.indices[order], a.indptr), shape=a.shape)


def _fit(a: sp.csr_matrix, **splitting):
    plan = simple_plan(SparsePattern.of(a.sorted_indices()), FIELDS + 1, **splitting)
    values, checks = fit_simple_level(plan, _PaddedRowOperator.from_scipy(a).values)
    return plan, values, checks


def _blocks(a: sp.csr_matrix):
    nv = FIELDS * N_CELLS
    return a[:nv, :nv], a[:nv, nv:], a[nv:, :nv], a[nv:, nv:]


def _host_velocity_inverse(f: sp.csr_matrix, frobenius: bool, block: bool) -> sp.csr_matrix:
    n = N_CELLS
    if not block:
        diagonal = f.diagonal()
        if not frobenius:
            return sp.diags(1.0 / diagonal)
        return sp.diags(diagonal / np.asarray(f.multiply(f).sum(axis=1)).ravel())
    cells = np.zeros((n, FIELDS, FIELDS))
    gram = np.zeros((n, FIELDS, FIELDS))
    dense = f.toarray()
    for i in range(n):
        rows = i + n * np.arange(FIELDS)
        cells[i] = dense[np.ix_(rows, rows)]
        gram[i] = dense[rows] @ dense[rows].T
    inverse = (
        np.array([cells[i].T @ np.linalg.inv(gram[i]) for i in range(n)])
        if frobenius
        else np.linalg.inv(cells)
    )
    m = np.zeros((FIELDS * n,) * 2)
    for i in range(n):
        rows = i + n * np.arange(FIELDS)
        m[np.ix_(rows, rows)] = inverse[i]
    return sp.csr_matrix(m)


@pytest.mark.parametrize(
    "splitting",
    [
        {"frobenius": False},
        {"frobenius": True, "schur_frobenius": True},
        {"frobenius": True, "block_splitting": True, "schur_frobenius": True},
        {"frobenius": False, "block_splitting": True},
    ],
    ids=["jacobi", "frobenius", "frobenius-block", "exact-block"],
)
@pytest.mark.parametrize("terms_per_chunk", [simple_fit.TERMS_PER_CHUNK, 7])
@pytest.mark.parametrize(
    ("shape", "aligned_rows", "aligned_divergence"),
    [("mixed", False, True), ("aligned", True, True), ("ragged", False, False)],
)
def test_the_fitted_pieces_are_the_host_formed_ones(
    monkeypatch, splitting, terms_per_chunk, shape, aligned_rows, aligned_divergence
) -> None:
    """Every piece equals its host formula, on each path the fit can take; a chunk of a few terms
    gives the same pieces.

    The small chunk takes the products and the Gram matrices through many chunks and a ragged last one,
    which the default size never reaches on a fixture this small. The chunk size is read when the fit is
    traced, so the compiled fit is dropped first: a cached one would run the other size's chunks.
    """
    monkeypatch.setattr(simple_fit, "TERMS_PER_CHUNK", terms_per_chunk)
    jax.clear_caches()
    a = _saddle(shape=shape)
    plan, values, checks = _fit(a, **splitting)
    assert plan.aligned_velocity_rows == aligned_rows
    assert bool(plan.divergence_run_width) == aligned_divergence
    refuse_unusable(checks)
    f, g, d, c = _blocks(a)
    block = splitting.get("block_splitting", False)
    m = _host_velocity_inverse(f, splitting["frobenius"], block)
    correction = (m @ g).toarray()
    schur = (c - d @ (m @ g)).toarray()

    def dense(layout, pair):
        return layout.with_values(*pair).to_scipy().toarray()

    np.testing.assert_allclose(
        dense(plan.correction_layout, values.correction), correction, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_array_equal(dense(plan.pressure_layout, values.pressure), c.toarray())
    if block:
        np.testing.assert_allclose(
            dense(plan.block_layout, values.block_inverse), m.toarray(), rtol=1e-11, atol=1e-14
        )
    schur_diagonal = np.diag(schur)
    expected = (
        schur_diagonal / np.sum(schur * schur, axis=1)
        if splitting.get("schur_frobenius")
        else 1.0 / schur_diagonal
    )
    np.testing.assert_allclose(np.asarray(values.schur_inverse), expected, rtol=1e-12)
    # Both halves of each transpose are the same operator's.
    x = np.random.default_rng(3).standard_normal(N_CELLS)
    for layout, pair, matrix in (
        (plan.pressure_layout, values.pressure, c),
        (plan.divergence_layout, values.divergence, d),
    ):
        operator = layout.with_values(*pair)
        y = np.random.default_rng(5).standard_normal(matrix.shape[0])
        (transposed,) = jax.linear_transpose(operator.apply, np.zeros(matrix.shape[1]))(y)
        np.testing.assert_allclose(np.asarray(transposed), matrix.T @ y, rtol=1e-12, atol=1e-13)
    assert x.shape == (N_CELLS,)


@pytest.mark.parametrize("shape", ["mixed", "aligned", "ragged"])
def test_the_schur_rows_reduce_alike_in_any_chunking(monkeypatch, shape) -> None:
    """``S``'s diagonal and row norms come out bit for bit the same however its rows are chunked.

    Each chunk places its rows' terms in a buffer of its own and reads the diagonal at its rows' global
    indices, so a chunk of several rows with a ragged last one -- padding rows past the level's end --
    is where an off-by-one in either would show. Every size here is checked to give at least one such
    chunking, against the fit in one chunk.
    """
    a = _saddle(shape=shape)
    splitting = {"frobenius": True, "block_splitting": True, "schur_frobenius": True}
    chunkings = []
    in_chunks = simple_fit._in_chunks

    def recording(n_rows, per_row):
        rows, chunks = in_chunks(n_rows, per_row)
        chunkings.append((n_rows, rows, chunks))
        return rows, chunks

    monkeypatch.setattr(simple_fit, "_in_chunks", recording)
    monkeypatch.setattr(simple_fit, "TERMS_PER_CHUNK", 1 << 30)
    jax.clear_caches()
    whole = np.asarray(_fit(a, **splitting)[1].schur_inverse)
    for terms in (40, 76, 133, 400):
        monkeypatch.setattr(simple_fit, "TERMS_PER_CHUNK", terms)
        jax.clear_caches()
        np.testing.assert_array_equal(np.asarray(_fit(a, **splitting)[1].schur_inverse), whole)
    assert any(n == N_CELLS and 1 < rows and rows * chunks > n for n, rows, chunks in chunkings), (
        chunkings
    )


def test_the_simplec_coefficient_divides_by_the_row_sum_and_falls_back_where_it_is_small() -> None:
    """SIMPLEC's velocity coefficient is ``1 / sum_j F_ij``, the Frobenius value where that sum is
    below a tenth of the diagonal."""
    a = _saddle(seed=2)
    f = _blocks(a)[0]
    nv = FIELDS * N_CELLS
    # Make two rows' sums small beside their diagonal, so the fallback is exercised.
    lil = a.tolil()
    for row in (4, 50):
        lil[row, (row + 1) % nv] = -lil[row, row] * 1.02
    a = lil.tocsr()
    f = _blocks(a)[0]
    _, values, checks = _fit(a, simplec=True)
    refuse_unusable(checks)
    sums = np.asarray(f.sum(axis=1)).ravel()
    diagonal = f.diagonal()
    usable = sums > 0.1 * np.abs(diagonal)
    assert (~usable).sum() >= 2
    expected = np.where(
        usable, 1.0 / sums, diagonal / np.asarray(f.multiply(f).sum(axis=1)).ravel()
    )
    np.testing.assert_allclose(np.asarray(values.velocity_inverse), expected, rtol=1e-13)


def test_an_unusable_level_is_refused_by_name() -> None:
    """A zero velocity diagonal, and a singular velocity cell block, are refused on the host."""
    a = _saddle().tolil()
    a[7, 7] = 0.0
    with pytest.raises(ValueError, match="velocity block has a zero or non-finite diagonal"):
        refuse_unusable(_fit(a.tocsr(), frobenius=True)[2])

    a = _saddle().tolil()
    rows = 5 + N_CELLS * np.arange(FIELDS)
    for f in rows:
        for g in rows:
            a[f, g] = 1.0  # cell 5's velocity block is all ones: rank one
    with pytest.raises(ValueError, match="velocity cell block is singular"):
        refuse_unusable(_fit(a.tocsr(), frobenius=False, block_splitting=True)[2])


def test_block_splitting_with_simplec_is_refused() -> None:
    """The block inverse overrides the diagonal, so asking for both names a splitting that never runs."""
    a = _saddle()
    with pytest.raises(ValueError, match="choose one"):
        simple_plan(SparsePattern.of(a), FIELDS + 1, block_splitting=True, simplec=True)


def test_a_level_storing_no_velocity_pressure_coupling_has_the_pressure_block_as_its_schur() -> (
    None
):
    """With ``G`` and ``D`` storing nothing the products have no terms at all: ``M G`` is empty and
    ``S = C``. A product over no entries must not reach its chunk loop, which slices a row even when it
    runs no iteration (it failed so on a field-split hierarchy)."""
    a = _saddle(shape="decoupled")
    for splitting in ({"frobenius": True}, {"frobenius": True, "block_splitting": True}):
        plan, values, checks = _fit(a, **splitting)
        refuse_unusable(checks)
        c = _blocks(a)[3].toarray()
        np.testing.assert_array_equal(
            plan.pressure_layout.with_values(*values.pressure).to_scipy().toarray(), c
        )
        np.testing.assert_allclose(np.asarray(values.schur_inverse), 1.0 / np.diag(c), rtol=1e-14)
        assert plan.correction_layout.with_values(*values.correction).to_scipy().nnz == 0
