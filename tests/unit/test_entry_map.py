"""The fixed linear maps a refit applies to an operand's stored values (``precondition/entry_map.py``).

A refit re-derives a hierarchy at an operator whose sparsity pattern it was built at, so every sparse
operation it performs -- a sub-block, a transpose, a padded-row packing, a Galerkin product with frozen
transfers -- is a fixed linear function of the stored values. These tests pin that each map computes
the operation it stands for (exactly for an entry-moving operation, to the rounding of the magnitudes
summed for a product), that a changed operand entry reaches only the outputs it enters, and that a map
refuses an operand of another structure rather than misplacing its values. Mesh-free throughout.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.precondition import JacobiSmoothedInverse, hierarchy_inverse
from aquaflux.precondition.entry_map import EntryMap, EntryMaps, SparsePattern
from aquaflux.precondition.multigrid import (
    StructuralPatterns,
    _block_tentative,
    _cell_blocks,
    _galerkin_coarse,
    _ones,
    _onto_pattern,
    build_convection_hierarchy,
)
from aquaflux.precondition.padded_rows import _padded_operator, _padded_rows, _PaddedRowOperator

EPS = np.finfo(np.float64).eps


def _random_pattern(n_rows: int, n_columns: int, density: float, seed: int) -> sp.csr_matrix:
    """A random pattern with a full diagonal, rows of unequal length, and some stored exact zeros."""
    rng = np.random.default_rng(seed)
    a = sp.random(n_rows, n_columns, density=density, random_state=rng, format="csr")
    a = (a + sp.eye(n_rows, n_columns)).tocsr()
    a.sort_indices()
    a.data = rng.standard_normal(a.nnz)
    a.data[:: max(a.nnz // 9, 1)] = 0.0  # stored zeros keep their slot
    return a


def _revalued(pattern: sp.csr_matrix, seed: int) -> sp.csr_matrix:
    """The same pattern, new values: what a refit hands each map."""
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(pattern.nnz) * 10.0 ** rng.uniform(-3.0, 3.0, pattern.nnz)
    return sp.csr_matrix((values, pattern.indices, pattern.indptr), shape=pattern.shape)


def _field_major_block(n_cells: int, block_size: int, seed: int) -> sp.csr_matrix:
    """A field-major nodal operator on a ring of cells: every field couples within a cell and to the
    same field of the two neighbouring cells, diagonally dominant so each cell block is invertible."""
    rng = np.random.default_rng(seed)
    cells = np.arange(n_cells)
    rows, columns, values = [], [], []
    for f in range(block_size):
        for g in range(block_size):
            rows.append(f * n_cells + cells)
            columns.append(g * n_cells + cells)
            values.append(
                (8.0 + rng.random(n_cells)) if f == g else 0.5 * rng.standard_normal(n_cells)
            )
        for shift in (-1, 1):
            rows.append(f * n_cells + cells)
            columns.append(f * n_cells + (cells + shift) % n_cells)
            values.append(-1.0 - rng.random(n_cells))
    n = block_size * n_cells
    return sp.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(n, n)
    )


# --- gathers: an operation that only moves entries is reproduced exactly -----------------------------


@pytest.mark.parametrize(
    "operation",
    [
        lambda m: m[3:17, 5:],
        lambda m: m[np.arange(0, 20, 3), :],
        lambda m: sp.csr_matrix(m.T),
    ],
    ids=["sub-block", "row selection", "transpose"],
)
def test_a_gathering_map_reproduces_an_entry_moving_operation_exactly(operation) -> None:
    """The map of a cut or a transpose lands on the operation's own pattern with its exact values.

    A gather copies values without arithmetic, so the comparison is bit for bit, at values other than
    the ones the map was built from.
    """
    pattern = _random_pattern(20, 24, 0.2, seed=0)
    entry_map = EntryMap.gathering(SparsePattern(pattern), operation)
    for seed in (1, 2):
        operand = _revalued(pattern, seed)
        want = sp.csr_matrix(operation(operand))
        got = entry_map.csr(operand)
        assert np.array_equal(got.indptr, want.indptr)
        assert np.array_equal(got.indices, want.indices)
        assert np.array_equal(got.data, want.data)


def test_a_gathering_map_into_a_dense_layout_leaves_an_unreached_slot_zero() -> None:
    """Padded rows and cell blocks land in dense arrays; a slot no stored entry reaches holds zero."""
    pattern = _random_pattern(30, 30, 0.15, seed=3)
    operand = _revalued(pattern, 4)
    width = int(np.diff(pattern.indptr).max()) + 2  # padding slots that no entry fills
    padded = EntryMap.gathering(SparsePattern(pattern), lambda m: _padded_rows(m, width, "row")[1])
    want = _padded_rows(operand.copy(), width, "row")[1]
    assert np.array_equal(padded.apply(operand).reshape(want.shape), want)

    blocks = EntryMap.gathering(SparsePattern(pattern), lambda m: _cell_blocks(m, 3))
    want = _cell_blocks(operand, 3)
    assert np.any(want == 0.0)  # the fixture has cell-block entries the operator does not store
    assert np.array_equal(blocks.apply(operand).reshape(want.shape), want)


def test_a_padded_gather_holds_its_filled_slots_as_a_mask_and_still_names_each_term_s_slot() -> (
    None
):
    """A gather into padded rows fills its slots in ascending order, so it holds a byte a slot rather
    than a slot index a term; each term's slot, read back, is still where its entry lands.

    The slots are read independently, from where each stored entry's number lands in the numbered
    padded rows, both for the rows and for the transpose (whose terms read the operand out of order).
    """
    pattern = _random_pattern(30, 30, 0.15, seed=33)
    width = int(np.diff(pattern.indptr).max()) + 1
    for transposed in (False, True):
        numbered = SparsePattern(pattern).numbered()
        operand = sp.csr_matrix(numbered.T) if transposed else numbered
        numbers = _padded_rows(operand, width, "column" if transposed else "row")[1].ravel()
        packing = EntryMap.placing(SparsePattern(pattern), numbers)
        assert packing._filled is not None and packing._target is None
        assert packing._filled.nbytes < 4 * packing.source.size
        landed = np.flatnonzero(numbers)
        assert np.array_equal(packing.target, landed)
        assert np.array_equal(packing.source, numbers[landed].astype(np.int64) - 1)


def test_a_held_padded_layout_packs_exactly_what_a_fresh_packing_does() -> None:
    """``_padded_operator`` through a held layout equals ``_PaddedRowOperator.from_scipy``, both ways."""
    pattern = _random_pattern(25, 25, 0.2, seed=5)
    maps = EntryMaps()
    for seed in (6, 7):
        operand = _revalued(pattern, seed)
        got = _padded_operator(operand, maps=maps, slot="level")
        want = _PaddedRowOperator.from_scipy(operand)
        for leaf_got, leaf_want in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
            assert np.array_equal(np.asarray(leaf_got), np.asarray(leaf_want))
        assert got.shape == want.shape


def test_an_operation_that_combines_entries_has_no_gathering_map() -> None:
    """Summing two entries into one slot is not a gather; building one as if it were is refused."""
    pattern = SparsePattern(_random_pattern(10, 10, 0.3, seed=8))
    with pytest.raises(ValueError, match="combined or rescaled"):
        EntryMap.gathering(pattern, lambda m: m @ m)


# --- products: L M R with L and R fixed --------------------------------------------------------------


def _terms_bound(entry_map: EntryMap, operand: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Per output slot, ``sum |w_j a_j|`` over its terms and the number of terms: what rounding scales with."""
    weight = 1.0 if entry_map.weight is None else entry_map.weight
    magnitude = np.bincount(
        entry_map.target,
        weights=np.abs(weight * operand.data[entry_map.source]),
        minlength=entry_map.n_out,
    )
    count = np.bincount(entry_map.target, minlength=entry_map.n_out)
    return magnitude, count


@pytest.mark.parametrize("smoothed", [False, True], ids=["tentative", "general"])
def test_the_product_map_agrees_with_the_sparse_product_to_the_rounding_of_its_terms(
    smoothed: bool,
) -> None:
    """``Pᵀ M P`` by the map against ``scipy``'s product stored on the same structural pattern.

    The two sum the same terms in different orders, so they differ by rounding: at most a few ulps of
    the slot's ``sum |w_j a_j|`` per term (a sum of ``n`` terms rounds by at most ``(n - 1) eps`` of its
    magnitude in each order). Judged against that magnitude rather than the slot's value, because a
    slot whose terms cancel has a value far below the size of its rounding. Both a 0/1 tentative
    prolongation and a general one (several entries per row, each a weight) are covered.
    """
    n_cells, block_size = 40, 2
    aggregate = np.arange(n_cells) // 3
    prolongation = _block_tentative(aggregate, int(aggregate.max()) + 1, block_size, False).tocsr()
    if smoothed:
        prolongation = (
            prolongation + 0.3 * _random_pattern(80, 80, 0.05, 9) @ prolongation
        ).tocsr()
    restriction = prolongation.T.tocsr()
    fine = _field_major_block(n_cells, block_size, seed=10)
    entry_map = EntryMap.product(restriction, SparsePattern(fine), prolongation)
    for seed in (11, 12):
        operand = _revalued(fine, seed)
        want = _onto_pattern(
            restriction @ operand @ prolongation,
            _ones(restriction) @ _ones(operand) @ _ones(prolongation),
        )
        got = entry_map.csr(operand)
        assert np.array_equal(got.indptr, want.indptr)
        assert np.array_equal(got.indices, want.indices)
        magnitude, count = _terms_bound(entry_map, operand)
        assert np.all(np.abs(got.data - want.data) <= 2.0 * count * EPS * magnitude)
        assert not np.array_equal(got.data, np.zeros_like(got.data))


def test_a_changed_operand_entry_reaches_exactly_the_products_it_enters() -> None:
    """Moving one fine entry ``M_ij`` changes exactly the slots ``(k, l)`` with ``L_ki R_jl`` stored,
    each by ``delta L_ki R_jl`` -- no other slot moves.

    Linear in the operand, so the difference of two applications is the map of the perturbation alone,
    up to the rounding of each slot's other terms (bounded by their summed magnitudes, as above); a
    misplaced or mis-weighted term would move a slot outside that set, or by the wrong amount.
    """
    left = _random_pattern(12, 30, 0.15, seed=14)
    right = _random_pattern(30, 9, 0.15, seed=15)
    middle = _random_pattern(30, 30, 0.12, seed=16)
    entry_map = EntryMap.product(left, SparsePattern(middle), right)
    base = _revalued(middle, 17)
    for entry in range(middle.nnz):  # every entry, so every term of the map is exercised
        moved = base.copy()
        delta = 0.37
        moved.data[entry] += delta
        change = (entry_map.csr(moved) - entry_map.csr(base)).toarray()
        i = int(np.searchsorted(middle.indptr, entry, side="right") - 1)
        j = int(middle.indices[entry])
        want = delta * np.outer(left[:, i].toarray().ravel(), right[j, :].toarray().ravel())
        assert np.array_equal(change != 0.0, want != 0.0)
        rounding = np.zeros(change.shape)
        for operand in (base, moved):
            magnitude, count = _terms_bound(entry_map, operand)
            rounding += entry_map.landing.csr(2.0 * count * EPS * magnitude).toarray()
        assert np.all(np.abs(change - want) <= rounding)
        assert np.all(np.abs(change[want != 0.0] - want[want != 0.0]) < 1e-6 * np.abs(delta))


def _product_by_unique(left, pattern: SparsePattern, right):
    """The product map's terms formed the plain way -- every list kept, the slots by ``np.unique`` --
    as ``(source, target, weight, landing keys)``: the reference the lean builder must reproduce."""
    n_rows_m, n_columns_m = pattern.shape
    row = np.repeat(np.arange(n_rows_m), np.diff(pattern.indptr))
    source, column = np.arange(pattern.nnz), pattern.indices.astype(np.int64)
    weight = np.ones(pattern.nnz)
    n_columns = n_columns_m
    if left is not None:
        left = sp.coo_matrix(left)
        pairs = [(t, e) for t in range(row.size) for e in range(left.nnz) if left.col[e] == row[t]]
        index_t, index_l = (np.array(p, dtype=np.int64) for p in zip(*pairs, strict=True))
        source, column = source[index_t], column[index_t]
        row, weight = left.row[index_l].astype(np.int64), weight[index_t] * left.data[index_l]
    if right is not None:
        right = sp.coo_matrix(right)
        pairs = [
            (t, r)
            for t in range(column.size)
            for r in range(right.nnz)
            if right.row[r] == column[t]
        ]
        index_t, index_r = (np.array(p, dtype=np.int64) for p in zip(*pairs, strict=True))
        source, row = source[index_t], row[index_t]
        column, weight = right.col[index_r].astype(np.int64), weight[index_t] * right.data[index_r]
        n_columns = right.shape[1]
    keys, target = np.unique(row * n_columns + column, return_inverse=True)
    return source, target, weight, keys


@pytest.mark.parametrize(
    "sides", ["both, 0/1", "both, weighted", "left only", "right only"], ids=str
)
def test_the_product_map_holds_the_terms_of_the_plain_construction_in_its_order(sides) -> None:
    """The product map is built to hold one term list at a time; its terms -- which operand entry,
    which slot, what weight, in what order -- must be exactly those of the plain construction, so a
    refit's sums (taken in term order) keep their bits.

    Each term is enumerated here by brute force over every pair rather than by the builder's pairing,
    and the cases cover a 0/1 transfer on both sides (the shipped refit: no weight list is formed),
    weighted transfers with several entries a row (the general pairing), and each side alone.
    """
    fine = _field_major_block(12, 2, seed=30)
    tentative = _block_tentative(np.arange(12) // 3, 4, 2, False).tocsr()
    weighted = (tentative + 0.3 * _random_pattern(24, 24, 0.1, 31) @ tentative).tocsr()
    left, right = {
        "both, 0/1": (tentative.T.tocsr(), tentative),
        "both, weighted": (weighted.T.tocsr(), weighted),
        "left only": (weighted.T.tocsr(), None),
        "right only": (None, weighted),
    }[sides]
    pattern = SparsePattern(fine)
    entry_map = EntryMap.product(left, pattern, right)
    source, target, weight, keys = _product_by_unique(left, pattern, right)
    assert np.array_equal(entry_map.source, source)
    assert np.array_equal(entry_map.target, target)
    if entry_map.weight is None:
        assert np.all(weight == 1.0) and sides == "both, 0/1"
    else:
        assert np.array_equal(entry_map.weight, weight)
    n_columns = entry_map.landing.shape[1]
    landing_rows = np.repeat(
        np.arange(entry_map.landing.shape[0]), np.diff(entry_map.landing.indptr)
    )
    assert np.array_equal(landing_rows * n_columns + entry_map.landing.indices, keys)
    operand = _revalued(fine, 32)
    want = np.bincount(target, weights=operand.data[source] * weight, minlength=keys.size)
    assert np.array_equal(entry_map.apply(operand), want)


def test_two_applications_of_one_map_agree_bit_for_bit() -> None:
    """The map sums its terms in one fixed order, so it is deterministic: a refit by a held map and one
    by a map built afresh from the same pattern give the same bits."""
    fine = _field_major_block(30, 2, seed=18)
    prolongation = _block_tentative(np.arange(30) // 2, 15, 2, False).tocsr()
    operand = _revalued(fine, 19)
    first = EntryMap.product(prolongation.T, SparsePattern(fine), prolongation).apply(operand)
    again = EntryMap.product(prolongation.T, SparsePattern(fine), prolongation).apply(operand)
    assert np.array_equal(first, again)


# --- refusal: a map applies only to the structure it was built for ---------------------------------


def test_two_owners_reading_one_pattern_hold_one_copy_and_another_pattern_is_never_given_it() -> (
    None
):
    """Owners reading the same block -- a map's output, or a copy of it -- share one held pattern.

    The second owner's operand is a separate copy (as a scaling or a shift makes), so only a match by
    content can find the first owner's pattern. A pattern of the same shape and entry count but other
    columns must not be given it: its values would land in the wrong slots.
    """
    block = _random_pattern(40, 40, 0.1, seed=40)
    first, second = EntryMaps(), EntryMaps()
    held = first.pattern_of(block)
    copy = sp.csr_matrix((block.data * 2.0, block.indices.copy(), block.indptr.copy()), block.shape)
    assert second.pattern_of(copy) is held
    landing = EntryMap.gathering(
        SparsePattern(block.T.tocsr()), lambda m: sp.csr_matrix(m.T)
    ).landing
    assert landing is held  # a map landing on a held pattern lands on that pattern
    moved = block.copy()
    row = int(np.flatnonzero(np.diff(moved.indptr) > 1)[0])
    start, stop = moved.indptr[row], moved.indptr[row + 1]
    taken = set(moved.indices[start:stop].tolist())
    moved.indices[stop - 1] = next(
        c for c in range(40) if c not in taken
    )  # one entry, another column
    moved.has_sorted_indices = False
    other = second.pattern_of(moved)
    assert other is not held and np.array_equal(other.indices, moved.indices)


def test_a_map_refuses_an_operand_whose_structure_has_moved() -> None:
    """A different shape, or the same entry count in other places, is refused rather than gathered from.

    Applied anyway, the values would land in the slots of entries the operand does not hold.
    """
    pattern = _random_pattern(15, 15, 0.2, seed=20)
    entry_map = EntryMap.gathering(SparsePattern(pattern), lambda m: m[2:9, 2:9])
    with pytest.raises(ValueError, match="wrong slots"):
        entry_map.apply(_random_pattern(16, 16, 0.2, seed=20))
    shifted = pattern.copy()
    first_row = slice(shifted.indptr[0], shifted.indptr[1])
    # Same shape and entry count, one entry moved to an unused column of its row.
    used = set(shifted.indices[first_row])
    free = next(c for c in range(15) if c not in used)
    shifted.indices[shifted.indptr[0]] = free
    assert shifted.shape == pattern.shape and shifted.nnz == pattern.nnz
    with pytest.raises(ValueError, match="wrong slots"):
        entry_map.apply(shifted)


def test_a_held_product_is_rebuilt_when_a_fixed_operand_changes_its_values() -> None:
    """``EntryMaps`` bakes ``L`` and ``R`` into the weights, so a changed ``R`` must give a new map --
    held, the old weights would silently compute the old product."""
    fine = _field_major_block(20, 1, seed=21)
    prolongation = _block_tentative(np.arange(20) // 4, 5, 1, False).tocsr()
    maps = EntryMaps()
    operand = _revalued(fine, 22)
    maps.product("galerkin", prolongation.T, operand, prolongation)
    scaled = (2.0 * prolongation).tocsr()
    got = maps.product("galerkin", scaled.T, operand, scaled)
    np.testing.assert_allclose(got.toarray(), (scaled.T @ operand @ scaled).toarray(), rtol=1e-13)


# --- the maps inside a refit -------------------------------------------------------------------------


def _hierarchy(block: sp.csr_matrix):
    return build_convection_hierarchy(
        block,
        block_size=2,
        max_coarse=20,
        max_levels=3,
        mis_aggregation=True,
        prolongation_smoothing="none",
        aggressive_levels=0,
    )


def test_a_mapped_refit_has_the_built_shapes_and_the_product_refit_s_values() -> None:
    """A refit through the held maps keeps every array shape of the hierarchy it refits (so a compiled
    cycle stays a cache hit) and equals the refit by sparse products to rounding.

    Each level operator is a Galerkin product of the one above, so each is checked against the product
    of the level above it in the same refit, to the rounding of its terms' magnitudes as above (checking
    against the product refit's own chain would compound the levels' rounding).
    """
    built = _hierarchy(_field_major_block(120, 2, seed=23))
    patterns = StructuralPatterns()
    for seed in (24, 25):
        operand = _revalued(_field_major_block(120, 2, seed=23), seed) + sp.eye(240) * 30.0
        operand = sp.csr_matrix(operand)
        mapped = built.refit(operand, patterns=patterns)
        plain = built.refit(operand)
        for got, want, original in zip(mapped.levels, plain.levels, built.levels, strict=True):
            for leaf_got, leaf_want, leaf_built in zip(
                jax.tree.leaves(got), jax.tree.leaves(want), jax.tree.leaves(original), strict=True
            ):
                assert np.shape(leaf_got) == np.shape(leaf_want) == np.shape(leaf_built)
        for depth, level in enumerate(mapped.levels[:-1]):
            fine = level.operator.to_scipy()
            prolongation = level.prolongation_scipy().tocsr()
            restriction = prolongation.T.tocsr()
            entry_map = EntryMap.product(restriction, SparsePattern(fine), prolongation)
            coarse = _galerkin_coarse(restriction, fine, prolongation)
            magnitude, count = _terms_bound(entry_map, fine)
            got = mapped.levels[depth + 1].operator.to_scipy()
            assert np.array_equal(got.indices, coarse.indices)
            assert np.all(np.abs(got.data - coarse.data) <= 2.0 * count * EPS * magnitude)


def test_a_coarsening_that_reads_no_values_is_refitted_rather_than_aggregated_again(
    monkeypatch,
) -> None:
    """At no strength threshold and an unsmoothed prolongation, a refresh refits instead of rebuilding.

    The aggregation then reads only the pattern, so a rebuild would reproduce the coarsening it holds;
    the refit returns the same hierarchy as a fresh build at the new operator (to the rounding of the
    Galerkin sums, judged here at a relative 1e-12) without aggregating. A strength threshold reads
    values, so there the refresh must still aggregate.
    """
    block = _field_major_block(150, 2, seed=26)
    moved = sp.csr_matrix(_revalued(block, 27) + sp.eye(300) * 30.0)
    builds = []
    build = hierarchy_inverse.build_convection_hierarchy

    def counted(*args, **kwargs):
        builds.append(1)
        return build(*args, **kwargs)

    settings = dict(max_coarse=20, max_levels=3, aggressive_levels=0, equilibrate=False)
    inverse = JacobiSmoothedInverse(block, 2, **settings)
    monkeypatch.setattr(hierarchy_inverse, "build_convection_hierarchy", counted)
    inverse.refactor_block(moved)
    assert not builds
    monkeypatch.undo()
    fresh = JacobiSmoothedInverse(moved, 2, **settings)
    got = jax.tree.leaves(inverse.traced_cycle().state)
    want = jax.tree.leaves(fresh.traced_cycle().state)
    for leaf_got, leaf_want in zip(got, want, strict=True):
        leaf_got, leaf_want = np.asarray(leaf_got), np.asarray(leaf_want)
        assert leaf_got.shape == leaf_want.shape
        np.testing.assert_allclose(
            leaf_got, leaf_want, rtol=1e-12, atol=1e-12 * np.max(np.abs(leaf_want))
        )

    reads_values = JacobiSmoothedInverse(block, 2, **settings, strength_threshold=0.25)
    monkeypatch.setattr(hierarchy_inverse, "build_convection_hierarchy", counted)
    reads_values.refactor_block(moved)
    assert builds
