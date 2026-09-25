"""Gathering, grouping and pairing ragged lists, each against the plain loop it replaces."""

from __future__ import annotations

import numpy as np
from aquaflux import ragged

# Rows of lengths 2, 0, 3 and 1: an empty row is the case a stride-based shortcut gets wrong.
OFFSETS = np.array([0, 2, 2, 5, 6])
ITEMS = np.array([10, 11, 20, 21, 22, 30])


def test_rows_gathers_the_chosen_rows_in_the_order_chosen_with_repeats():
    selected = np.array([2, 0, 1, 2, 3])
    values, row, counts = ragged.rows(OFFSETS, ITEMS, selected)
    expected = [ITEMS[OFFSETS[r] : OFFSETS[r + 1]] for r in selected]
    np.testing.assert_array_equal(values, np.concatenate(expected))
    np.testing.assert_array_equal(row, np.repeat(np.arange(5), [len(e) for e in expected]))
    np.testing.assert_array_equal(counts, [3, 2, 0, 3, 1])


def test_group_is_the_csr_of_items_by_key_keeping_their_order():
    keys = np.array([2, 0, 2, 3, 0, 2])
    offsets, order = ragged.group(keys, 5)
    np.testing.assert_array_equal(offsets, [0, 2, 2, 5, 6, 6])
    for key in range(5):
        members = order[offsets[key] : offsets[key + 1]]
        np.testing.assert_array_equal(members, np.flatnonzero(keys == key))


def test_pairs_within_groups_is_the_per_group_cartesian_product():
    rng = np.random.default_rng(0)
    group_a = rng.integers(0, 6, 40)
    group_b = rng.integers(0, 6, 25)
    left, right = ragged.pairs_within_groups(group_a, group_b, 6)
    got = sorted(zip(left.tolist(), right.tolist(), strict=True))
    expected = sorted((i, j) for i in range(40) for j in range(25) if group_a[i] == group_b[j])
    assert got == expected


def test_pairs_can_be_formed_a_range_of_groups_at_a_time():
    rng = np.random.default_rng(1)
    group_a = rng.integers(0, 7, 30)
    group_b = rng.integers(0, 7, 30)
    whole = set(zip(*ragged.pairs_within_groups(group_a, group_b, 7), strict=True))
    pieces = [
        set(zip(*ragged.pairs_within_groups(group_a, group_b, 7, first, stop), strict=True))
        for first, stop in ((0, 2), (2, 3), (3, 7))
    ]
    assert set().union(*pieces) == whole
    assert sum(len(p) for p in pieces) == len(whole)
    assert all(group_a[i] in (2,) for i, _ in pieces[1])


def test_group_keeps_each_groups_items_in_their_original_order_at_scale():
    """At six items any sort is stable; a quick sort reorders equal keys only on larger inputs."""
    keys = np.random.default_rng(2).integers(0, 7, 5000)
    offsets, order = ragged.group(keys, 7)
    for key in range(7):
        np.testing.assert_array_equal(
            order[offsets[key] : offsets[key + 1]], np.flatnonzero(keys == key)
        )
