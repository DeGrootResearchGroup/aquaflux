"""Fixed linear maps between the stored values of sparse operators whose patterns do not move.

A preconditioner refit re-derives a hierarchy at a new operator whose sparsity pattern is the one it
was built at: the probe fills a fixed pattern, and a frozen coarsening keeps every prolongation. Every
sparse operation the refit performs -- cutting a block out of a matrix, transposing it, packing it into
padded rows, forming a Galerkin product ``R A P`` with ``R`` and ``P`` held fixed -- is then a fixed
linear function of the operand's stored values, ``out_t = sum_j w_tj a_j``, and only the values change.

:class:`EntryMap` stores that function once as three flat arrays (source entry, target slot, weight), so
each refit applies it as one gather, multiply and sum in place of the sparse operation's own index work
(the search, the sort, the symbolic product). Where the result lands -- a compressed-sparse-row (CSR)
matrix, padded rows, a dense per-cell block -- is the caller's; the map is the same.

:class:`EntryMaps` keeps the maps one owner needs between refits, keyed by call site, rebuilding a map
only when an operand's pattern (or a fixed operand's values) has moved.

A map is built on the host, from patterns alone, with numpy and scipy. :meth:`EntryMap.on_device` gives
the same map as a :class:`DeviceEntryMap`, whose arrays live on the device and whose :meth:`~DeviceEntryMap.
apply` is traced JAX: a preconditioner refit then moves its operator's values through the map without
leaving the device.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Hashable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from aquaflux.ragged import pairs_within_groups

__all__ = ["DeviceEntryMap", "EntryMap", "EntryMaps", "SparsePattern"]


class SparsePattern:
    """A CSR sparsity pattern -- shape, row pointers, column indices -- that operands are checked against.

    The check is an exact comparison of the stored index arrays, so it accepts only an operand storing
    the same entries in the same order: anything else would put each value in the wrong slot. It
    remembers the last index arrays it accepted, so an operand built on those very arrays (the output of
    another map, which shares its pattern's arrays) passes by identity without comparing again.
    """

    def __init__(self, matrix: sp.spmatrix) -> None:
        matrix = sp.csr_matrix(matrix)
        self.shape: tuple[int, int] = matrix.shape
        # Copies: an operand sorted in place later must not change the pattern held here.
        self.indptr = matrix.indptr.copy()
        self.indices = matrix.indices.copy()
        # Weak references: remembering an operand's arrays must not keep them alive.
        self._accepted = (weakref.ref(self.indptr), weakref.ref(self.indices))

    @property
    def nnz(self) -> int:
        """Stored entries."""
        return int(self.indices.size)

    def matches(self, matrix: sp.csr_matrix) -> bool:
        """Whether ``matrix`` stores exactly this pattern's entries, in this pattern's order."""
        if matrix.shape != self.shape or matrix.nnz != self.nnz:
            return False
        if matrix.indptr is self._accepted[0]() and matrix.indices is self._accepted[1]():
            return True
        if np.array_equal(matrix.indptr, self.indptr) and np.array_equal(
            matrix.indices, self.indices
        ):
            self._accepted = (weakref.ref(matrix.indptr), weakref.ref(matrix.indices))
            return True
        return False

    def csr(self, values: np.ndarray) -> sp.csr_matrix:
        """``values`` on this pattern, sharing its index arrays (so a later check passes by identity)."""
        return sp.csr_matrix((values, self.indices, self.indptr), shape=self.shape, copy=False)

    def numbered(self) -> sp.csr_matrix:
        """This pattern with entry ``j`` holding the value ``j + 1``: a sparse operation over it reports,
        in each output slot, which entry landed there (``0`` where none did).

        A fresh matrix on its own index arrays each call, so an operation that sorts it in place leaves
        the pattern held here untouched. Not kept: it is needed only while a map is built, and held it
        would cost more than the map.
        """
        return sp.csr_matrix(
            (np.arange(1, self.nnz + 1, dtype=np.float64), self.indices.copy(), self.indptr.copy()),
            shape=self.shape,
        )


class EntryMap:
    """``out = bincount(target, weight * values[source], n_out)``: a fixed linear map of stored values.

    Built once from the pattern of the operand it reads (:class:`SparsePattern`); :meth:`apply` refuses
    an operand whose pattern differs, rather than putting its values in the wrong slots. ``landing`` is
    the CSR pattern of the output when the output is a sparse matrix (:meth:`csr`); a caller packing the
    output into another layout reads the flat values from :meth:`apply`.

    Two builders cover what a refit does. :meth:`gathering` turns any sparse operation that only moves
    entries -- a sub-block, a transpose, a padded-row packing -- into its map by running the operation
    once on a numbered copy of the pattern. :meth:`product` gives the map ``M -> L M R`` for fixed ``L``
    and ``R``, landing on the structural pattern of the product (every entry the operands' patterns can
    produce, so none is dropped for coming out exactly zero).

    Attributes
    ----------
    pattern : SparsePattern
        The pattern of the operand the map reads.
    source, target : np.ndarray of int, shape ``(n_terms,)``
        For each term, the operand entry it reads and the output slot it adds into.
    weight : np.ndarray or None, shape ``(n_terms,)``
        Each term's factor; ``None`` when every factor is one (a gather, or a product through 0/1
        transfers).
    distinct : bool
        The builder's statement that no two terms share a target, so an unweighted map is applied as an
        indexed copy rather than a sum. Not checked: a map stating it falsely keeps only one term per slot.
    n_out : int
        Output slots; a slot no term reaches is zero.
    landing : SparsePattern or None
        The output's CSR pattern, ``n_out`` entries, when the output is a sparse matrix.
    """

    def __init__(
        self,
        pattern: SparsePattern,
        source: np.ndarray,
        target: np.ndarray,
        n_out: int,
        *,
        weight: np.ndarray | None = None,
        landing: SparsePattern | None = None,
        distinct: bool = False,
    ) -> None:
        if landing is not None and landing.nnz != n_out:
            raise ValueError(
                f"a landing pattern of {landing.nnz} entries cannot hold {n_out} slots."
            )
        # The smallest index type that holds every entry and slot: a map is held for a session, and its
        # index arrays are most of its memory.
        index = np.int32 if max(pattern.nnz, int(n_out)) < np.iinfo(np.int32).max else np.int64
        self.pattern = pattern
        self._source = np.asarray(source, dtype=index)
        self._target = np.asarray(target, dtype=index)
        if weight is not None and np.all(np.asarray(weight) == 1.0):
            weight = None  # a product through 0/1 transfers sums the entries unweighted
        self.weight = None if weight is None else np.asarray(weight, dtype=np.float64)
        self.n_out = int(n_out)
        self.landing = landing
        self.distinct = distinct
        # A gather whose terms reach distinct slots is a plain indexed copy, and one that fills every
        # slot is a single gather in slot order; anything else sums.
        self._scatter = distinct and self.weight is None
        self._slot_ordered = self._scatter and self._target.size == self.n_out
        if self._slot_ordered:
            # Every slot filled once: held as the source of each slot in order, the targets dropped.
            self._source = self._source[np.argsort(self._target, kind="stable")]
            self._target = None
        # A map reading every entry once, in storage order, holds no source either.
        if self._source.size == pattern.nnz and np.array_equal(
            self._source, np.arange(pattern.nnz, dtype=index)
        ):
            self._source = None

    @classmethod
    def gathering(
        cls,
        pattern: SparsePattern,
        operation: Callable[[sp.csr_matrix], sp.spmatrix | np.ndarray],
    ) -> EntryMap:
        """The map of a sparse operation that only moves entries (never combines two into one slot).

        Parameters
        ----------
        pattern : SparsePattern
            The pattern of the operand the operation will be applied to.
        operation : callable
            ``matrix -> result``, a sparse matrix (the map then lands on its pattern) or a dense array
            (the map fills it flat, in C order). It must keep every value it is given unchanged and add
            none together -- slicing, transposing, sampling and padding all do. A slot holding no entry
            stays zero.

        Returns
        -------
        EntryMap
            The map, built from one run of ``operation`` on the numbered pattern.
        """
        result = operation(pattern.numbered())
        if sp.issparse(result):
            result = sp.csr_matrix(result)
            return cls.placing(pattern, result.data, landing=SparsePattern(result))
        return cls.placing(pattern, result)

    @classmethod
    def placing(
        cls, pattern: SparsePattern, numbers: np.ndarray, landing: SparsePattern | None = None
    ) -> EntryMap:
        """The gather that fills each output slot with the operand entry ``numbers`` names there.

        ``numbers`` is the output of an entry-moving operation run over :meth:`SparsePattern.numbered`:
        ``j + 1`` where entry ``j`` landed, ``0`` where none did. :meth:`gathering` runs the operation;
        a caller that runs it anyway for another output of the same pass passes its result here.

        Parameters
        ----------
        pattern : SparsePattern
            The operand's pattern.
        numbers : np.ndarray
            The numbered output, any shape; read flat, in C order.
        landing : SparsePattern, optional
            The output's CSR pattern, when the output is a sparse matrix.

        Returns
        -------
        EntryMap
            The gather.

        Raises
        ------
        ValueError
            If ``numbers`` is not a set of entry numbers -- the operation combined or rescaled entries.
        """
        numbers = np.asarray(numbers, dtype=np.float64).ravel()
        if np.any(numbers != np.round(numbers)) or np.any((numbers < 0) | (numbers > pattern.nnz)):
            raise ValueError(
                "the operation combined or rescaled entries; only an operation that moves entries "
                "unchanged has a gathering map."
            )
        filled = np.flatnonzero(numbers)
        return cls(
            pattern,
            numbers[filled].astype(np.int64) - 1,
            filled,
            numbers.size,
            landing=landing,
            distinct=True,
        )

    @classmethod
    def product(
        cls, left: sp.spmatrix | None, pattern: SparsePattern, right: sp.spmatrix | None
    ) -> EntryMap:
        """The map ``M -> L M R`` for fixed ``L`` and ``R``, onto the product's structural pattern.

        A term per triple of stored entries ``(L_ki, M_ij, R_jl)``, weighted ``L_ki R_jl`` and summed into
        ``(k, l)``. The landing pattern is every ``(k, l)`` such a triple reaches, sorted -- the pattern of
        the same product taken over all-ones operands, so it depends on the three patterns alone and an
        exactly-zero result keeps its slot.

        Parameters
        ----------
        left : scipy.sparse matrix or None
            ``L``, shape ``(n_rows, M.shape[0])``; ``None`` for the identity.
        pattern : SparsePattern
            The pattern of ``M``, the operand whose values vary.
        right : scipy.sparse matrix or None
            ``R``, shape ``(M.shape[1], n_columns)``; ``None`` for the identity.

        Returns
        -------
        EntryMap
            The map, landing on the product's structural pattern.
        """
        # The term lists are as long as the operand (and longer, if a fixed operand has rows of more
        # than one entry), so they are built to be held one at a time: no term list is kept once the
        # next is made, a weight list exists only once a fixed operand's entries are not all one, and
        # the landing slots are found by sorting a copy of the keys rather than by `np.unique`, whose
        # inverse costs several more lists of the same length.
        n_rows_m, n_columns_m = pattern.shape
        row = np.repeat(np.arange(n_rows_m), np.diff(pattern.indptr))
        column = pattern.indices.astype(np.int64)
        source = weight = None  # every operand entry, in storage order; every weight one
        n_rows, n_columns = n_rows_m, n_columns_m
        if left is not None:
            left = sp.coo_matrix(left)
            if left.shape[1] != n_rows_m:
                raise ValueError(f"L is {left.shape}, but M has {n_rows_m} rows.")
            # Pair each term with every L entry in the column its row names. The terms are the long
            # list, so they go first: the pairing sorts only its second list.
            index_t, index_l = _pairs(row, left.col, n_rows_m)
            del row
            source, column = index_t, column[index_t]
            row = left.row[index_l].astype(np.int64)
            weight = _times(None, left.data, index_l)
            del index_l
            n_rows = left.shape[0]
        if right is not None:
            right = sp.coo_matrix(right)
            if right.shape[0] != n_columns_m:
                raise ValueError(f"R is {right.shape}, but M has {n_columns_m} columns.")
            index_t, index_r = _pairs(column, right.row, n_columns_m)
            del column
            source = index_t if source is None else source[index_t]
            row = row[index_t]
            weight = _times(None if weight is None else weight[index_t], right.data, index_r)
            del index_t
            column = right.col[index_r].astype(np.int64)
            del index_r
            n_columns = right.shape[1]
        if source is None:
            source = np.arange(pattern.nnz)
        # `row * n_columns + column`, formed in place over the row list.
        np.multiply(row, n_columns, out=row)
        np.add(row, column, out=row)
        del column
        keys = np.sort(row)
        distinct = np.empty(keys.size, dtype=bool)
        distinct[:1] = True
        np.not_equal(keys[1:], keys[:-1], out=distinct[1:])
        keys = keys[distinct]
        del distinct
        target = np.searchsorted(keys, row)
        del row
        counts = np.bincount(keys // n_columns, minlength=n_rows)
        indptr = np.concatenate([[0], np.cumsum(counts)])
        landing = SparsePattern(
            sp.csr_matrix(
                (np.zeros(keys.size), (keys % n_columns).astype(np.int32), indptr),
                shape=(n_rows, n_columns),
            )
        )
        return cls(pattern, source, target, keys.size, weight=weight, landing=landing)

    @property
    def source(self) -> np.ndarray:
        """Each term's operand entry, shape ``(n_terms,)``."""
        return np.arange(self.pattern.nnz) if self._source is None else self._source

    @property
    def target(self) -> np.ndarray:
        """Each term's output slot, shape ``(n_terms,)``."""
        return np.arange(self.n_out) if self._target is None else self._target

    def on_device(self) -> DeviceEntryMap:
        """This map with its arrays on the device, applied by traced JAX (:class:`DeviceEntryMap`).

        A map whose terms fill distinct slots becomes one gather per output slot; any other is held with
        its terms sorted by slot, so the sum into each slot is a sorted segment sum.
        """
        if self._scatter:
            slots = np.full(self.n_out, self.pattern.nnz, dtype=self._index_dtype())
            slots[self.target] = self.source
            return DeviceEntryMap(
                slots=jnp.asarray(slots),
                source=None,
                target=None,
                weight=None,
                n_in=self.pattern.nnz,
                n_out=self.n_out,
            )
        order = np.argsort(self.target, kind="stable")
        return DeviceEntryMap(
            slots=None,
            source=jnp.asarray(self.source[order]),
            target=jnp.asarray(self.target[order]),
            weight=None if self.weight is None else jnp.asarray(self.weight[order]),
            n_in=self.pattern.nnz,
            n_out=self.n_out,
        )

    def _index_dtype(self) -> type[np.signedinteger]:
        """The smallest index type holding every entry, the sentinel past them, and every slot."""
        bound = max(self.pattern.nnz + 1, self.n_out)
        return np.int32 if bound < np.iinfo(np.int32).max else np.int64

    def apply(self, matrix: sp.spmatrix) -> np.ndarray:
        """The output values for ``matrix``, flat, shape ``(n_out,)``.

        Raises
        ------
        ValueError
            If ``matrix`` does not store exactly the pattern this map was built for.
        """
        matrix = sp.csr_matrix(matrix)
        if not self.pattern.matches(matrix):
            raise ValueError(
                f"this map was built for a {self.pattern.shape} operator storing {self.pattern.nnz} "
                f"entries; a {matrix.shape} operator storing {matrix.nnz} entries in another pattern "
                "would put its values in the wrong slots. Build a map for the new structure."
            )
        values = matrix.data.copy() if self._source is None else matrix.data[self._source]
        if self._slot_ordered:
            return values
        if self._scatter:
            out = np.zeros(self.n_out, dtype=np.float64)
            out[self._target] = values
            return out
        if self.weight is not None:
            values = values * self.weight
        return np.bincount(self._target, weights=values, minlength=self.n_out)

    def csr(self, matrix: sp.spmatrix) -> sp.csr_matrix:
        """The output as a CSR matrix on :attr:`landing`, sharing its index arrays."""
        if self.landing is None:
            raise ValueError("this map's output is not a sparse matrix; read it with `apply`.")
        return self.landing.csr(self.apply(matrix))


class DeviceEntryMap(eqx.Module):
    """An :class:`EntryMap` on the device: ``out = sum over terms of weight * values[source]``, traced.

    Built by :meth:`EntryMap.on_device`, never directly. One of two forms, decided by the map:

    - a **gather** (``slots`` set): each output slot reads at most one operand entry, ``slots`` holding
      its index or ``n_in`` for a slot no entry reaches, which reads a zero appended to the values;
    - a **sum** (``source``, ``target`` set): one term per pair, sorted by ``target``, summed into each
      slot by a sorted segment sum, each term scaled by ``weight`` when the map has weights.

    The index arrays are leaves and only the sizes are static, so a map rides into a jitted refit as an
    argument and every refit at the same pattern reuses one compiled program.

    Attributes
    ----------
    slots : jnp.ndarray or None
        For a gather, the operand entry each output slot reads, shape ``(n_out,)``.
    source, target : jnp.ndarray or None
        For a sum, each term's operand entry and output slot, shape ``(n_terms,)``, sorted by slot.
    weight : jnp.ndarray or None
        For a weighted sum, each term's factor, shape ``(n_terms,)``.
    n_in, n_out : int
        Operand entries and output slots. Static.
    """

    slots: jnp.ndarray | None
    source: jnp.ndarray | None
    target: jnp.ndarray | None
    weight: jnp.ndarray | None
    n_in: int = eqx.field(static=True)
    n_out: int = eqx.field(static=True)

    def apply(self, values: jnp.ndarray) -> jnp.ndarray:
        """The output values for an operand's stored ``values``, shape ``(n_in,)`` -> ``(n_out,)``.

        In ``values``'s own floating-point type: a gather moves entries without arithmetic, and a sum
        adds in the operand's type as the host map does.
        """
        if self.slots is not None:
            return jnp.concatenate([values, jnp.zeros(1, dtype=values.dtype)])[self.slots]
        terms = values[self.source]
        if self.weight is not None:
            terms = terms * self.weight.astype(values.dtype)
        return jax.ops.segment_sum(
            terms, self.target, num_segments=self.n_out, indices_are_sorted=True
        )


class EntryMaps:
    """The entry maps one owner reuses across refits, one per call site, rebuilt when an operand moves.

    Each ``slot`` names a call site (``("leading", 0)``). A map is reused while its operand still stores
    the pattern it was built for; otherwise it is rebuilt and replaced. Patterns are shared between the maps reading one
    operand, so a pattern compared once in a refit passes by identity at every other map that reads it.
    A host object, never a leaf of a traced hierarchy.
    """

    def __init__(self) -> None:
        self._maps: dict[Hashable, tuple[SparsePattern, object]] = {}
        self._patterns: list[SparsePattern] = []

    def pattern_of(self, matrix: sp.csr_matrix) -> SparsePattern:
        """The held pattern ``matrix`` stores, or a new one held from now on."""
        for pattern in self._patterns:
            if pattern.matches(matrix):
                return pattern
        pattern = SparsePattern(matrix)
        # Bounded: a long-lived owner whose operands keep moving holds only the recent patterns.
        self._patterns = [*self._patterns[-15:], pattern]
        return pattern

    def held(
        self,
        slot: Hashable,
        matrix: sp.csr_matrix,
        build: Callable[[SparsePattern], object],
    ):
        """What ``build`` made for this slot, reused while ``matrix``'s pattern holds.

        Parameters
        ----------
        slot : hashable
            The call site.
        matrix : scipy.sparse.csr_matrix
            The operand whose values vary.
        build : callable
            ``pattern -> object`` holding one or more maps over ``matrix``'s pattern.

        Returns
        -------
        object
            The held or newly built object.
        """
        entry = self._maps.get(slot)
        if entry is not None and entry[0].matches(matrix):
            return entry[1]
        pattern = self.pattern_of(matrix)
        built = build(pattern)
        self._maps[slot] = (pattern, built)
        return built

    def csr(
        self,
        slot: Hashable,
        matrix: sp.csr_matrix,
        operation: Callable[[sp.csr_matrix], sp.spmatrix],
    ) -> sp.csr_matrix:
        """``operation(matrix)`` for an operation that only moves entries, by its held map."""
        return self.held(slot, matrix, lambda p: EntryMap.gathering(p, operation)).csr(matrix)


def _pairs(terms: np.ndarray, fixed: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Every pairing of a term with a fixed-operand entry of the same group (``pairs_within_groups``).

    A prolongation built from aggregates has at most one entry in each fine row, so most products pair
    each term with at most one entry; then the pairing is a lookup of that entry, done here without the
    general expansion.
    """
    counts = np.bincount(fixed, minlength=n_groups)
    if counts.max(initial=0) > 1:
        return pairs_within_groups(terms, fixed, n_groups)
    entry_of_group = np.full(n_groups, -1, dtype=np.int64)
    entry_of_group[fixed] = np.arange(fixed.size)
    index_fixed = entry_of_group[terms]
    index_terms = np.flatnonzero(index_fixed >= 0)
    return index_terms, index_fixed[index_terms]


def _times(weight: np.ndarray | None, data: np.ndarray, index: np.ndarray) -> np.ndarray | None:
    """Each term's weight times the fixed-operand entry it pairs with; ``None`` (every weight one)
    while both are all ones. ``weight`` ``None`` is all ones, and one times an entry is that entry
    exactly, so the weights are those a list of ones would have given."""
    if np.all(data == 1.0):
        return weight
    factor = np.asarray(data, dtype=np.float64)[index]
    return factor if weight is None else weight * factor
