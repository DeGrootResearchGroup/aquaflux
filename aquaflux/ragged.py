"""Ragged lists as flat arrays: gathering rows, grouping items and pairing them within groups.

A ragged list -- rows of different lengths, such as the nodes of each face or the triangles
around each vertex -- is stored in compressed-sparse-row (CSR) form: one flat array of items and
a row-pointer array of offsets, row ``r`` being ``items[offsets[r]:offsets[r + 1]]``. Working on
many rows at once without a per-row Python loop comes down to three operations, defined once
here:

- :func:`rows` -- the items of a chosen set of rows, concatenated, with the row each came from;
- :func:`group` -- the CSR of items keyed by an integer label, i.e. building a ragged list;
- :func:`pairs_within_groups` -- every pairing of an item of one list with an item of another that
  carries the same group label, the flat form of a per-group Cartesian product.

Build-time numpy only: nothing here is traced or differentiated. Imports nothing from the package,
so any layer may use it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["group", "pairs_within_groups", "rows"]


def rows(
    offsets: np.ndarray, items: np.ndarray, selected: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The items of some rows of a CSR list, concatenated in the order the rows are selected.

    Parameters
    ----------
    offsets : np.ndarray of int, shape ``(n_rows + 1,)``
        Row pointers.
    items : np.ndarray, shape ``(n_items, ...)``
        The flat items.
    selected : np.ndarray of int, shape ``(n_selected,)``
        Rows to gather; need not be sorted, contiguous or distinct.

    Returns
    -------
    (values, row, counts)
        ``values`` the gathered items; ``row`` for each the position in ``selected`` of the row
        it came from (``0`` to ``n_selected - 1``); ``counts`` the length of each selected row.
    """
    selected = np.asarray(selected, dtype=np.int64)
    counts = offsets[selected + 1] - offsets[selected]
    total = int(counts.sum())
    row = np.repeat(np.arange(selected.size), counts)
    position = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    return items[np.repeat(offsets[selected], counts) + position], row, counts


def group(keys: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Group item indices by an integer key, as a CSR list.

    Parameters
    ----------
    keys : np.ndarray of int, shape ``(n_items,)``
        Each item's group, in ``[0, n_groups)``.
    n_groups : int

    Returns
    -------
    (offsets, order)
        Row pointers of shape ``(n_groups + 1,)``, and the item indices sorted by key -- stable,
        so items of one group keep their original order.
    """
    keys = np.asarray(keys, dtype=np.int64)
    order = np.argsort(keys, kind="stable")
    offsets = np.zeros(n_groups + 1, dtype=np.int64)
    np.cumsum(np.bincount(keys, minlength=n_groups), out=offsets[1:])
    return offsets, order


def pairs_within_groups(
    group_a: np.ndarray, group_b: np.ndarray, n_groups: int, first: int = 0, stop: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Every pairing of an ``a`` item with a ``b`` item of the same group.

    Parameters
    ----------
    group_a, group_b : np.ndarray of int
        Group label of each item of the two lists, in ``[0, n_groups)``.
    n_groups : int
    first, stop : int, optional
        Pair only the groups in ``[first, stop)``, so a large product can be formed in pieces.

    Returns
    -------
    (index_a, index_b)
        Indices into the two lists, one entry per pair, grouped by group label.
    """
    stop = n_groups if stop is None else stop
    group_a = np.asarray(group_a, dtype=np.int64)
    group_b = np.asarray(group_b, dtype=np.int64)
    in_a = np.flatnonzero((group_a >= first) & (group_a < stop))
    in_b = np.flatnonzero((group_b >= first) & (group_b < stop))
    offsets_b, order_b = group(group_b[in_b] - first, stop - first)
    labels = group_a[in_a] - first
    counts = offsets_b[labels + 1] - offsets_b[labels]
    index_a = np.repeat(in_a, counts)
    position = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    index_b = in_b[order_b[np.repeat(offsets_b[labels], counts) + position]]
    return index_a, index_b
