"""Where a probed Jacobian's entries are stored, and how each probe chunk's responses are placed there.

A coloured probe (:mod:`~aquaflux.precondition.sparse_jacobian`) fills a fixed sparsity pattern, so two
things about it are fixed by the mesh graph and built once: the layout of the stored values
(:class:`BlockLayout`, one block-major array over a two-group field partition), and which probe response
element feeds each stored entry (:class:`ProbeGather`). Only the values change from one materialize to
the next, and they stay on the device (:class:`ProbedBlocks`).

The second can be held two ways, the placement the same either way: entry by entry, or -- wherever each
block of the partition stores one cell pattern over all of its field pairs -- at cell level, the field
pair implicit (:class:`~aquaflux.precondition.kronecker_pattern.KroneckerPattern`).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .entry_map import DeviceMatrix, SparsePattern
from .kronecker_pattern import CellLevelPattern, KroneckerPattern, entry_position

if TYPE_CHECKING:
    from .sparse_jacobian import ColumnProbePlan, _ChunkProbe


def _index_dtype(bound: int) -> type[np.signedinteger]:
    """The narrowest signed integer type holding every value up to ``bound``.

    The arrays this module builds carry one entry per Jacobian nonzero — tens of millions on a
    three-dimensional coupled mesh — and several are live at once while the de-compression is assembled,
    so the width of an index is the difference between a peak of a few hundred megabytes and several
    gigabytes. Every such array here has a bound known before it is formed, which is what lets the type
    be chosen up front and the wide form never materialize.
    """
    return np.int32 if bound < np.iinfo(np.int32).max else np.int64


class BlockLayout:
    """Where a two-group partition's four blocks lie in one block-major value array.

    A partition whose leading group is every field is the one-group case: its leading block is the whole
    matrix in CSR order (:attr:`whole`) and the other three are empty.

    The stored entries of a field-major matrix are ordered leading-by-leading, leading-by-trailing,
    trailing-by-leading, trailing-by-trailing, each block's entries in that block's own compressed-sparse-
    row (CSR) order, so each block is a contiguous slice of the values. The structure of each block is
    held here, on the host, as the pattern every inverse fitted to that block checks against; only the
    values change from one materialize to the next, and those stay on the device.

    Attributes
    ----------
    leading_dofs : int
        The degrees of freedom of the leading group (the partition point).
    n_dofs : int
        The degrees of freedom of the whole matrix.
    """

    def __init__(
        self,
        leading_dofs: int,
        n_dofs: int,
        blocks: tuple[tuple[int, int, SparsePattern], ...],
        diagonal: jnp.ndarray,
    ) -> None:
        self.leading_dofs = leading_dofs
        self.n_dofs = n_dofs
        # Per block: (start, stop) of its slice of the values, and its pattern.
        self._blocks = blocks
        # Position of each degree of freedom's diagonal entry in the values, shape (n_dofs,), on device.
        self._diagonal = diagonal

    @classmethod
    def of(cls, csr: sp.csr_matrix, leading_dofs: int) -> tuple[BlockLayout, np.ndarray]:
        """The layout of ``csr``'s stored entries, and the order taking CSR positions to block-major ones.

        Raises
        ------
        ValueError
            If a diagonal entry is not stored: the shift is added in place on the stored diagonal, so a
            missing one would have nowhere to go.
        """
        n_dofs = csr.shape[0]
        index_dtype = _index_dtype(max(n_dofs, csr.nnz))
        rows = np.repeat(np.arange(n_dofs, dtype=index_dtype), np.diff(csr.indptr))
        cols = csr.indices
        key = (rows >= leading_dofs).astype(np.int8) * 2 + (cols >= leading_dofs)
        # A stable order keeps each block's entries in the CSR order they had in the whole matrix.
        order = np.argsort(key, kind="stable")
        bounds = np.concatenate([[0], np.cumsum(np.bincount(key, minlength=4))])
        offsets = (0, leading_dofs)
        sizes = (leading_dofs, n_dofs - leading_dofs)
        blocks = []
        diagonal = np.full(n_dofs, -1, dtype=index_dtype)
        for block in range(4):
            start, stop = int(bounds[block]), int(bounds[block + 1])
            taken = order[start:stop]
            row0, col0 = offsets[block // 2], offsets[block % 2]
            block_rows = rows[taken] - index_dtype(row0)
            indptr = np.concatenate(
                [[0], np.cumsum(np.bincount(block_rows, minlength=sizes[block // 2]))]
            ).astype(index_dtype)
            indices = (cols[taken] - index_dtype(col0)).astype(index_dtype, copy=False)
            shape = (sizes[block // 2], sizes[block % 2])
            blocks.append((start, stop, SparsePattern.of_arrays(shape, indptr, indices)))
            if block in (0, 3):
                on_diagonal = np.flatnonzero(block_rows == indices)
                diagonal[block_rows[on_diagonal] + row0] = start + on_diagonal
        if (diagonal < 0).any():
            raise ValueError(
                "BlockLayout: the pattern stores no diagonal entry for some degree of freedom, so the "
                "shift has nowhere to be added."
            )
        return cls(leading_dofs, n_dofs, tuple(blocks), jnp.asarray(diagonal)), order

    @classmethod
    def of_cells(
        cls,
        cells: SparsePattern,
        n_fields: int,
        leading_fields: int,
        stored: tuple[bool, bool, bool, bool],
    ) -> BlockLayout:
        """The layout :meth:`of` gives a matrix storing ``cells`` over each stored block's field pairs.

        Built from the cell pattern alone: each stored block is a :class:`KroneckerPattern`, whose
        degree-of-freedom CSR arrays and diagonal positions follow from it without a sort, and a block
        not stored is empty. Equal, array for array, to :meth:`of` over the whole matrix.

        Raises
        ------
        ValueError
            As :meth:`of`, if a diagonal entry is not stored.
        """
        n = cells.shape[0]
        counts = (leading_fields, n_fields - leading_fields)
        offsets = (0, leading_fields * n)
        n_dofs = n_fields * n
        nnz = sum(
            counts[block // 2] * counts[block % 2] * cells.nnz
            for block in range(4)
            if stored[block]
        )
        index_dtype = _index_dtype(max(n_dofs, nnz))
        blocks, start = [], 0
        diagonal = np.full(n_dofs, -1, dtype=index_dtype)
        for block in range(4):
            fields = (counts[block // 2], counts[block % 2])
            shape = (fields[0] * n, fields[1] * n)
            if stored[block]:
                # Held as its cell pattern: every refit checks its blocks against these by identity, so
                # their dof arrays are needed only when a block is read back to the host.
                kronecker = KroneckerPattern(cells, *fields)
                pattern = CellLevelPattern(kronecker)
                if block in (0, 3) and fields[0]:
                    row0 = offsets[block // 2]
                    diagonal[row0 : row0 + shape[0]] = start + kronecker.diagonal_positions()
            else:
                pattern = SparsePattern.of_arrays(
                    shape,
                    np.zeros(shape[0] + 1, dtype=index_dtype),
                    np.zeros(0, dtype=index_dtype),
                )
            blocks.append((start, start + pattern.nnz, pattern))
            start += pattern.nnz
        if (diagonal < 0).any():
            raise ValueError(
                "BlockLayout: the pattern stores no diagonal entry for some degree of freedom, so the "
                "shift has nowhere to be added."
            )
        return cls(offsets[1], n_dofs, tuple(blocks), jnp.asarray(diagonal))

    @property
    def nnz(self) -> int:
        """Stored entries, over every block."""
        return self._blocks[-1][1]

    def start(self, block: int) -> int:
        """Where block ``block`` (in :meth:`blocks`'s order) starts among the stored values."""
        return self._blocks[block][0]

    @property
    def whole(self) -> SparsePattern | None:
        """The whole matrix's pattern, when this layout keeps it as one group; ``None`` when it splits it."""
        return self._blocks[0][2] if self.leading_dofs == self.n_dofs else None

    def blocks(self, data: jnp.ndarray) -> tuple[DeviceMatrix, ...]:
        """The four blocks over ``data`` (shape ``(nnz,)``), each its slice of the values, on the device.

        Returns
        -------
        tuple of DeviceMatrix
            ``(A_ll, A_lt, A_tl, A_tt)``, as :meth:`~aquaflux.precondition.FieldGroups.blocks` orders them.
        """
        return tuple(
            DeviceMatrix(pattern, data[start:stop]) for start, stop, pattern in self._blocks
        )

    def diagonal(self, data: jnp.ndarray) -> jnp.ndarray:
        """The stored diagonal of ``data`` in degree-of-freedom order, shape ``(n_dofs,)``."""
        return _take(data, self._diagonal)

    def add_to_diagonal(self, data: jnp.ndarray, shift_diagonal: jnp.ndarray) -> jnp.ndarray:
        """``data`` with ``shift_diagonal`` (shape ``(n_dofs,)``) added to its stored diagonal.

        ``data`` is consumed: its buffer is reused for the result, so the shift costs no second copy.
        """
        return _add_at(data, self._diagonal, jnp.asarray(shift_diagonal, dtype=jnp.float64))


@jax.jit
def _take(values: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
    """``values[positions]``, compiled (run eagerly, a gather is a dispatch of its own)."""
    return values[positions]


@functools.partial(jax.jit, donate_argnums=0)
def _add_at(values: jnp.ndarray, positions: jnp.ndarray, amounts: jnp.ndarray) -> jnp.ndarray:
    """``values`` with ``amounts`` added at ``positions`` (distinct), reusing ``values``'s buffer."""
    return values.at[positions].add(amounts, unique_indices=True)


class ProbeGather:
    """The fixed CSR structure of a materialize, and how to fill it from the probe responses.

    Both halves are fixed by the mesh graph, so they are built once and reused by every refresh: which
    probe supplies which Jacobian entry does not change, only the response *values* do.

    **The de-compression runs one probe-chunk at a time, which is what keeps the peak bounded.** The
    obvious form — hold every response, then gather the whole matrix in one indexing expression — needs
    ``n_probes x n_fields x n_cells`` floats live at once, several hundred megabytes on a
    three-dimensional coupled case, and a naive sentinel for the out-of-reach entries doubles that again
    by copying the lot. Grouping the entries by the probe that feeds them instead lets each chunk's
    responses be consumed as soon as they are computed and then dropped, so the only large array left is
    the matrix being built.

    **The de-compression runs on the device**, where the probes are computed: the entry order is held
    there, and each chunk is placed by one compiled program, so no response crosses to the host.
    :func:`materialize_blocks` keeps the values there (:class:`ProbedBlocks`); the public
    :func:`materialize_block_jacobian` reads them back once, at its end, to return a host matrix.

    Which response feeds which entry is held by one of two orders, the same placement either way:
    :class:`_CellEntries`, at cell level, wherever each block keeps all of its field pairs or none (every
    pattern a field split or a single block inverse probes), and :class:`_DofEntries`, entry by entry,
    for any other.

    Entries lying outside their column's reach have no source probe at all (see
    :class:`ColumnProbePlan`); they simply never appear in either order, so an output initialized to
    zero leaves them zero and no sentinel is needed.

    Attributes
    ----------
    layout : BlockLayout
        Where the entries lie: block by block over a two-group partition, or, over one group, the whole
        matrix in its own CSR order (:attr:`BlockLayout.whole`).
    indptr, indices : np.ndarray or None
        The whole ``(n_fields n, n_fields n)`` Jacobian's CSR structure, read off the layout's whole
        pattern (formed on each read when that pattern is held at cell level); ``None`` when the layout
        splits the matrix into blocks.
    nnz : int
        Stored entries, including the explicit zeros the full pattern keeps.
    """

    def __init__(self, entries: _CellEntries | _DofEntries, nf: int, layout: BlockLayout) -> None:
        self.layout = layout
        self.nnz = layout.nnz
        self._entries = entries
        self._nf = nf

    @property
    def indptr(self) -> np.ndarray | None:
        """The whole matrix's CSR row pointers, read off the layout's whole pattern."""
        whole = self.layout.whole
        return None if whole is None else whole.indptr

    @property
    def indices(self) -> np.ndarray | None:
        """The whole matrix's CSR column indices, read off the layout's whole pattern."""
        whole = self.layout.whole
        return None if whole is None else whole.indices

    def width(self, chunks: Sequence[tuple[int, int, int]]) -> int:
        """The most lanes any one of ``chunks`` (``(field, start, count)`` each) needs.

        :meth:`scatter` places every chunk's entries through this many lanes, so one compiled program
        serves every chunk of a materialize.
        """
        return max((self._entries.lanes(start, count) for _, start, count in chunks), default=0)

    def scatter(
        self, data: jnp.ndarray, responses: jnp.ndarray, start: int, count: int, width: int
    ) -> jnp.ndarray:
        """``data`` with every entry fed by probes ``[start, start + count)`` filled from their responses.

        Parameters
        ----------
        data : jnp.ndarray
            The values being filled, shape ``(nnz,)``, starting from zero -- entries outside their
            column's reach are never written. Consumed: its buffer is reused for the result.
        responses : jnp.ndarray
            This chunk's responses, shape ``(rows >= count, n_fields * n_cells)``; rows beyond ``count``
            are ignored, so a padded final chunk needs no special case.
        start, count : int
            The chunk's half-open probe range, inside one column field.
        width : int
            At least the chunk's lane count: :meth:`width` of the materialize's chunks.

        Returns
        -------
        jnp.ndarray
            The values, shape ``(nnz,)``.
        """
        return self._entries.scatter(data, responses, start, count, width)

    def values(self, probing: _ChunkProbe) -> jnp.ndarray:
        """Every stored entry's value, in this structure's order, by probing one chunk at a time.

        De-compressed AS THE PROBES COME BACK: each chunk's responses are scattered into the fixed values
        and then dropped, so the only large array live at once is the values being built. Holding every
        response instead costs ``n_probes * nf`` floats -- comparable to the matrix itself.
        """
        chunks = tuple(probing.chunks())
        width = self.width(chunks)
        data = jnp.zeros(self.nnz, dtype=jnp.float64)
        for field, start, count in chunks:
            data = self.scatter(data, probing.responses(field, start, count), start, count, width)
        return data


class _DofEntries:
    """Which response element feeds each stored entry, one pair of indices per entry, grouped by probe.

    ``_source`` is the flat index of the response element, ``_position`` the entry it fills. The
    general order, for a pattern whose blocks keep only some of their field pairs; it holds two indices
    for every stored entry, so :class:`_CellEntries` replaces it wherever the pattern allows.
    """

    def __init__(self, gather_map: np.ndarray, zero_slot: int, nf: int) -> None:
        self._nf = nf
        # Narrowed as each array is formed rather than at the end, so the wide form is never one of the
        # several live at once. A source index is bounded by `zero_slot`, a position by the entry count.
        # Choosing from the bound also removes a latent wrap: these were narrowed to 32-bit
        # unconditionally, which is right for every mesh reached so far and silently truncates on one
        # whose entry count or probe-response index does not fit.
        index_dtype = _index_dtype(zero_slot)
        position_dtype = _index_dtype(gather_map.shape[0])
        live = gather_map != zero_slot
        position = np.flatnonzero(live).astype(position_dtype, copy=False)
        source = gather_map[live].astype(index_dtype, copy=False)
        # Group the entries by the probe that supplies them, so a chunk's entries are one contiguous
        # slice. Grouped by a `scipy.sparse` COO -> CSR conversion rather than `np.argsort`: treating
        # `probe_of` as the row index of a throwaway sparse matrix makes the grouping a counting sort
        # over `probe_of`'s own range -- a few hundred values -- so it never forms the full-entry-count
        # index array `np.argsort` is forced to return (numpy has no narrower form for a sort
        # permutation). Order within a probe is free -- `scatter` below writes each entry to a unique
        # position regardless of it -- so the unstable grouping a counting sort gives is exact, not an
        # approximation; there are no duplicate (probe, position) pairs (`position` values are already
        # unique), so the conversion only reorders and never sums.
        n_probes = zero_slot // nf
        probe_of = source // index_dtype(nf)
        grouped = sp.coo_matrix(
            (source, (probe_of, position)), shape=(n_probes, gather_map.shape[0])
        ).tocsr()
        # On the device, where the responses are computed, so a materialize never moves one to the host.
        # The host copies are dropped as soon as each is moved: only the probe boundaries stay here.
        self._source = jnp.asarray(grouped.data)
        self._position = jnp.asarray(grouped.indices.astype(position_dtype, copy=False))
        # One boundary per probe plus a closing one, so a chunk ending at the last probe still has a
        # `_probe_start[start + count]` to read -- exactly a CSR row-pointer array, which the conversion
        # above already produces.
        self._probe_start = grouped.indptr.astype(index_dtype, copy=False)

    def lanes(self, start: int, count: int) -> int:
        """The entries probes ``[start, start + count)`` fill."""
        return int(self._probe_start[start + count] - self._probe_start[start])

    def scatter(
        self, data: jnp.ndarray, responses: jnp.ndarray, start: int, count: int, width: int
    ) -> jnp.ndarray:
        """:meth:`ProbeGather.scatter` by this order."""
        lo, hi = int(self._probe_start[start]), int(self._probe_start[start + count])
        if hi == lo:
            return data
        return _scatter_chunk(
            data, responses, self._source, self._position, lo, hi, start * self._nf, width=width
        )


@functools.partial(jax.jit, static_argnames="width", donate_argnums=0)
def _scatter_chunk(
    data: jnp.ndarray,
    responses: jnp.ndarray,
    source: jnp.ndarray,
    position: jnp.ndarray,
    lo: int,
    hi: int,
    offset: int,
    *,
    width: int,
) -> jnp.ndarray:
    """Entries ``[lo, hi)`` of the probe-grouped order placed from one chunk's responses.

    Through ``width`` lanes whatever the chunk's entry count, so one program serves every chunk: a lane
    past ``hi`` is sent past the end of ``data`` and dropped. Each entry's position is distinct, so the
    write is a plain placement.
    """
    lane = lo + jnp.arange(width)
    live = lane < hi
    taken = jnp.where(live, source.at[lane].get(mode="fill", fill_value=0) - offset, 0)
    placed = jnp.where(live, position.at[lane].get(mode="fill", fill_value=0), data.shape[0])
    return data.at[placed].set(responses.reshape(-1)[taken], mode="drop", unique_indices=True)


class _ColumnColouring(NamedTuple):
    """The cell entries one colouring's probes read, grouped by the colour of their column cell.

    ``entries`` (on the device) lists every cell entry inside the colouring's reach, those of colour
    ``c`` at ``[colour_start[c], colour_start[c + 1])``; ``colour_start`` is held on the host, to cut a
    chunk, and on the device (``device_start``), to tell each lane which probe of the chunk feeds it.
    """

    entries: jnp.ndarray
    colour_start: np.ndarray
    device_start: jnp.ndarray


class _CellEntries:
    """Which response element feeds each stored entry, described at cell level.

    Each block of the layout stores one cell pattern over every one of its field pairs
    (:class:`~aquaflux.precondition.kronecker_pattern.KroneckerPattern`), so the response element that
    feeds an entry follows from the cell entry, the field pair and the plan: column field ``b``'s probe
    ``(b, colour of the column cell)`` holds it at row ``a n + i``. And the entry's place among the stored
    values follows from the same three by :func:`~aquaflux.precondition.kronecker_pattern.entry_position`.
    What is held is therefore one list of cell entries per distinct colouring, grouped by colour, and each
    cell entry's row: indices over the cell pattern rather than two over every stored entry, which on a
    six-field coupled Jacobian is a seventy-second of the index storage.

    A chunk's probes all read one column field ``b`` (:meth:`ColumnProbePlan.field_chunks`), so a chunk is
    one slice of ``b``'s colouring's entries, and each lane fills that entry for every row field of a
    stored block in ``b``'s column.
    """

    def __init__(
        self,
        plan: ColumnProbePlan,
        cells: SparsePattern,
        in_reach: np.ndarray,
        layout: BlockLayout,
        stored: tuple[bool, bool, bool, bool],
    ) -> None:
        n = plan.n_cells
        self._n_cells = n
        self._nf = plan.n_fields * n
        self._probe_base = plan.probe_base
        self._probe_field = np.repeat(np.arange(plan.n_fields), np.asarray(plan.n_colours))
        nnz = cells.nnz
        index_dtype = _index_dtype(max(layout.n_dofs, layout.nnz + 1))
        self._indptr = jnp.asarray(cells.indptr.astype(index_dtype))
        rows = np.repeat(np.arange(n, dtype=index_dtype), np.diff(cells.indptr))
        self._rows = jnp.asarray(rows)
        del rows
        # One entry list per distinct colouring: fields probed at one reach share it.
        self._colourings: list[_ColumnColouring] = []
        held: list[tuple[np.ndarray, np.ndarray]] = []
        self._colouring_of: list[int] = []
        for field in range(plan.n_fields):
            colour, reach = plan.colour[field], in_reach[field]
            for index, (seen_colour, seen_reach) in enumerate(held):
                if np.array_equal(colour, seen_colour) and np.array_equal(reach, seen_reach):
                    self._colouring_of.append(index)
                    break
            else:
                live = np.flatnonzero(reach)
                key = np.asarray(colour)[cells.indices[live]]
                grouped = live[np.argsort(key, kind="stable")].astype(index_dtype)
                start = np.concatenate(
                    [[0], np.cumsum(np.bincount(key, minlength=plan.n_colours[field]))]
                ).astype(index_dtype)
                self._colourings.append(
                    _ColumnColouring(jnp.asarray(grouped), start, jnp.asarray(start))
                )
                held.append((colour, reach))
                self._colouring_of.append(len(held) - 1)
        # Per column field: where each row field's block of entries starts (-1 where the block is not
        # stored), the column field's place in its block, and its block's column-field count.
        leading = layout.leading_dofs // n
        counts = (leading, plan.n_fields - leading)
        self._bases, self._column = [], []
        for b in range(plan.n_fields):
            column_group = int(b >= leading)
            bases = np.full(plan.n_fields, -1, dtype=np.int64)
            for a in range(plan.n_fields):
                row_group = int(a >= leading)
                block = 2 * row_group + column_group
                if stored[block]:
                    row_field = a - row_group * leading
                    bases[a] = layout.start(block) + row_field * counts[column_group] * nnz
            self._bases.append(jnp.asarray(bases.astype(index_dtype)))
            self._column.append((b - column_group * leading, counts[column_group]))
        self._nnz = nnz

    def _range(self, start: int, count: int) -> tuple[int, _ColumnColouring, int]:
        """The column field of probes ``[start, start + count)``, its colouring, and the first colour."""
        field = int(self._probe_field[start])
        return field, self._colourings[self._colouring_of[field]], start - self._probe_base[field]

    def lanes(self, start: int, count: int) -> int:
        """The cell entries probes ``[start, start + count)`` read."""
        _, colouring, first = self._range(start, count)
        return int(colouring.colour_start[first + count] - colouring.colour_start[first])

    def scatter(
        self, data: jnp.ndarray, responses: jnp.ndarray, start: int, count: int, width: int
    ) -> jnp.ndarray:
        """:meth:`ProbeGather.scatter` by this order."""
        field, colouring, first = self._range(start, count)
        lo = int(colouring.colour_start[first])
        hi = int(colouring.colour_start[first + count])
        if hi == lo:
            return data
        column_field, column_fields = self._column[field]
        return _scatter_cell_chunk(
            data,
            responses,
            colouring.entries,
            colouring.device_start,
            self._rows,
            self._indptr,
            self._bases[field],
            lo,
            hi,
            first,
            column_field,
            column_fields=column_fields,
            n_cells=self._n_cells,
            width=width,
        )


@functools.partial(jax.jit, static_argnames=("column_fields", "n_cells", "width"), donate_argnums=0)
def _scatter_cell_chunk(
    data: jnp.ndarray,
    responses: jnp.ndarray,
    entries: jnp.ndarray,
    colour_start: jnp.ndarray,
    rows: jnp.ndarray,
    indptr: jnp.ndarray,
    bases: jnp.ndarray,
    lo: int,
    hi: int,
    first: int,
    column_field: int,
    *,
    column_fields: int,
    n_cells: int,
    width: int,
) -> jnp.ndarray:
    """Cell entries ``[lo, hi)`` of one colouring placed, for every stored row field, from a chunk's responses.

    Lane ``l`` reads cell entry ``entries[lo + l]`` in row ``i``; its probe within the chunk is its
    colour's offset from the chunk's first colour; and for row field ``a`` it takes response element
    ``a n + i`` of that probe into the entry's place in row field ``a``'s block (``bases[a]``, ``-1``
    where none is stored, which drops the write). Through ``width`` lanes whatever the chunk's size, so
    one program serves every chunk of one colouring; a lane past ``hi`` is dropped. Every placement is
    distinct.
    """
    lane = lo + jnp.arange(width)
    live = lane < hi
    entry = entries.at[lane].get(mode="fill", fill_value=0)
    row = rows[entry]
    probe = jnp.searchsorted(colour_start, lane, side="right") - 1 - first
    within = entry_position(indptr, row, entry, 0, column_field, column_fields, 0)
    row_fields = jnp.arange(bases.shape[0])
    kept = live[:, None] & (bases >= 0)[None, :]
    placed = jnp.where(kept, bases[None, :] + within[:, None], data.shape[0])
    taken = jnp.where(kept, probe[:, None] * (row_fields.shape[0] * n_cells), 0)
    taken = taken + jnp.where(kept, row_fields[None, :] * n_cells + row[:, None], 0)
    return data.at[placed.ravel()].set(
        responses.reshape(-1)[taken.ravel()], mode="drop", unique_indices=True
    )


class ProbedBlocks:
    """One materialize's values over a :class:`BlockLayout`, on the device: the blocks, and the shift.

    Attributes
    ----------
    data : jnp.ndarray
        The values, block-major, shape ``(nnz,)``.
    layout : BlockLayout
        Where each block lies in them.
    """

    def __init__(self, data: jnp.ndarray, layout: BlockLayout) -> None:
        self.data = data
        self.layout = layout

    def blocks(self) -> tuple[DeviceMatrix, ...]:
        """``(A_ll, A_lt, A_tl, A_tt)`` over :attr:`data`, each its own slice of the values."""
        return self.layout.blocks(self.data)

    def diagonal(self) -> jnp.ndarray:
        """The Jacobian's diagonal in degree-of-freedom order, shape ``(n_dofs,)``."""
        return self.layout.diagonal(self.data)

    def add_shift(self, shift_diagonal: jnp.ndarray) -> None:
        """Add the pseudo-transient shift to the diagonal, in place.

        The in-place counterpart of :func:`shifted_jacobian`, for values this materialize owns: the
        whole matrix is never formed, so there is no input to keep unmodified, and every diagonal entry
        is stored (:meth:`BlockLayout.of` refuses a pattern without one), so no entry is created.
        """
        self.data = self.layout.add_to_diagonal(self.data, shift_diagonal)
