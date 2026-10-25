"""The padded-row (ELLPACK) level operator, and the fixed gathers that pack one on a held pattern.

Every multigrid level, every SIMPLE-smoother piece and the field split's coupling store their frozen
operator this way: ``(n_rows, width)`` column indices and entries, rows shorter than ``width`` filled
with zero-valued slots. :class:`_PaddedRowOperator` is the operator and its matvec, and packs its
transpose only when asked (:meth:`_PaddedRowOperator.with_transpose`, and
:func:`~aquaflux.solve.with_transposes` over a whole map);
:func:`_padded_rows` is the one slot rule; :class:`PaddedColumns` is each slot's column and the stored
entry that fills it, held as arrays or, for an operator storing a cell pattern over full field-pair
blocks, computed from the cell pattern; :class:`_PaddedLayout` holds the columns of an operator whose
pattern does not move, so a refit moves only values, from a host matrix or from values already on the
device; :func:`_padded_operator` picks between packing afresh and the held layout.
"""

from __future__ import annotations

from collections.abc import Hashable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.custom_derivatives import linear_call

from . import fused_product
from .entry_map import DeviceMatrix, EntryMap, EntryMaps, SparsePattern
from .kronecker_pattern import KroneckerPattern


class PaddedColumns(eqx.Module):
    """Which column each slot of a padded-row (ELLPACK) operator reads, and which stored entry fills it.

    The index half of a padded-row operator, the half that does not change while its pattern does not.
    Two forms, the same arrays either way: :class:`_HeldColumns` holds them, and :class:`_KroneckerColumns`
    computes them from a cell pattern when the operator stores one over a full block of field pairs.
    """

    def columns(self) -> jnp.ndarray:
        """The column of every slot, shape ``(n_rows, width)``, int32; a padding slot reads column 0."""
        raise NotImplementedError

    def row_lengths(self) -> jnp.ndarray:
        """The stored entries of every row, shape ``(n_rows,)``, int32; the slots after them pad it."""
        raise NotImplementedError

    def entries(self) -> jnp.ndarray:
        """The stored entry each slot holds, shape ``(n_rows, width)``; ``n_in`` for a padding slot."""
        raise NotImplementedError

    def product(self, values: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """``A x`` for the operator storing ``values`` in these slots.

        On the CPU, where :func:`~aquaflux.precondition.fused_product.fused_products` allows it, one
        fused, threaded loop over the stored slots; otherwise a gather of ``x`` at every slot's column,
        a multiply and a sum along each row.

        Parameters
        ----------
        values : jnp.ndarray
            The entry of every slot, shape ``(n_rows, width)``; a padding slot holds zero.
        x : jnp.ndarray
            The vector, shape ``(n_columns,)``.

        Returns
        -------
        jnp.ndarray
            The product, shape ``(n_rows,)``.
        """
        return fused_product.routed(self._fused_product, self._gathered_product, values, x)

    def _gathered_product(self, values: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(values * x[self.columns()], axis=1)

    def _fused_product(self, values: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


class _HeldColumns(PaddedColumns):
    """:class:`PaddedColumns` held as arrays, for any pattern.

    ``held_entries`` is ``None`` for an operator packed directly from a matrix, which no gather fills.
    """

    held_columns: jnp.ndarray
    lengths: jnp.ndarray
    held_entries: jnp.ndarray | None

    def columns(self) -> jnp.ndarray:
        return self.held_columns

    def row_lengths(self) -> jnp.ndarray:
        return self.lengths

    def entries(self) -> jnp.ndarray:
        return self.held_entries

    def _fused_product(self, values: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return fused_product.held_product(self.held_columns, self.lengths, values, x)


class _KroneckerColumns(PaddedColumns):
    """:class:`PaddedColumns` of a cell pattern over full field-pair blocks, computed at cell level.

    The rows are field-major over this direction's cells: row ``(f, d)`` of row field ``f`` and cell ``d``
    stores ``other_fields * length(d)`` entries, its columns ascending -- the other side's field ``g`` the
    run of slots ``[g length(d), (g + 1) length(d))``, each run the cell row's columns. Slot ``s`` is then
    ``g = s // length(d)``, ``k = s % length(d)``, column ``g n_other + cells[d, k]``, and a slot past the
    stored entries pads at column 0: exactly the slots :func:`_padded_rows` gives the degree-of-freedom
    pattern, decoded rather than held.

    The stored entries are consecutive along a row: row ``(f, d)``'s first is its field's offset plus
    ``other_fields`` times its cell row's first cell entry.

    Attributes
    ----------
    cells : jnp.ndarray
        Each cell row's columns, shape ``(n_cells, cell_width)``, padding unspecified.
    lengths : jnp.ndarray
        Each cell row's stored entries, shape ``(n_cells,)``.
    indptr : jnp.ndarray
        The row pointers of the cell pattern, shape ``(n_cells + 1,)``.
    fields, other_fields : int
        Row fields, and column fields. Static.
    n_other : int
        Column cells. Static.
    width : int
        Slots per row. Static.
    cell_nnz : jnp.ndarray
        Stored entries of the cell pattern, 0-d.
    n_in : jnp.ndarray
        Stored entries of the operator, the sentinel of a padding slot, 0-d.

    The two counts are leaves, not static: a rebuild into one shape budget changes them, and must still
    reuse the compiled cycle.
    """

    cells: jnp.ndarray
    lengths: jnp.ndarray
    indptr: jnp.ndarray
    fields: int = eqx.field(static=True)
    other_fields: int = eqx.field(static=True)
    n_other: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    cell_nnz: jnp.ndarray
    n_in: jnp.ndarray

    @classmethod
    def of(cls, pattern: KroneckerPattern, width: int) -> _KroneckerColumns:
        """The rows of ``pattern``, each padded to ``width`` slots."""
        cells = pattern.cells
        lengths = np.diff(cells.indptr).astype(np.int32)
        # As wide as the budget allows rather than as the widest row, so a rebuild into one budget keeps
        # every shape, as the held form's does.
        cell_width = max(int(width) // pattern.column_fields, 1)
        slot = np.arange(cell_width)[None, :]
        live = slot < lengths[:, None]
        flat = np.minimum(cells.indptr[:-1, None] + slot, max(cells.nnz - 1, 0))
        held = np.where(live, cells.indices[flat] if cells.nnz else 0, 0).astype(np.int32)
        return cls(
            cells=jnp.asarray(held),
            lengths=jnp.asarray(lengths),
            indptr=jnp.asarray(cells.indptr.astype(np.int32)),
            fields=pattern.row_fields,
            other_fields=pattern.column_fields,
            n_other=cells.shape[1],
            width=int(width),
            cell_nnz=jnp.asarray(pattern.cells.nnz, dtype=jnp.int32),
            n_in=jnp.asarray(pattern.nnz, dtype=jnp.int32),
        )

    def _slots(self):
        """Per cell row and slot: the other side's field, the place in the cell row, and liveness."""
        slot = jnp.arange(self.width, dtype=jnp.int32)[None, :]
        length = self.lengths[:, None]
        # The other side's field, ``slot // length``, counted rather than divided: the field count is a
        # handful, and a vector integer division is not (on a CPU it is several instructions a lane).
        field = sum(
            (slot >= step * length).astype(jnp.int32) for step in range(1, self.other_fields)
        )
        field = jnp.zeros_like(slot) if isinstance(field, int) else field
        place = jnp.clip(slot - field * length, 0, self.cells.shape[1] - 1)
        return field, place, slot < self.other_fields * length

    def _cell_columns(self) -> jnp.ndarray:
        """The column of every slot of one row field's rows, shape ``(n_cells, width)``.

        Every row field's rows read the same columns, so this is :meth:`columns` before the repeat.
        """
        field, place, live = self._slots()
        cell = jnp.take_along_axis(self.cells, place, axis=1)
        return jnp.where(live, field * self.n_other + cell, 0)

    def _by_row_field(self, per_cell: jnp.ndarray) -> jnp.ndarray:
        """A per-cell-row array repeated for every row field: shape ``(fields * n_cells, ...)``."""
        return jnp.broadcast_to(per_cell[None], (self.fields, *per_cell.shape)).reshape(
            self.fields * per_cell.shape[0], *per_cell.shape[1:]
        )

    def columns(self) -> jnp.ndarray:
        return self._by_row_field(self._cell_columns())

    def _gathered_product(self, values: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """XLA's form of the product, with ``x`` gathered once per cell row and read by every row field.

        The same slots, entries and row sums as the general gathered form, without decoding and
        gathering ``fields`` identical copies of each cell row's columns: inside a loop the compiler
        keeps the decoded columns live for the whole loop, so the repeat costs memory as well as work.
        """
        n_cells = self.cells.shape[0]
        gathered = x[self._cell_columns()]
        per_field = values.reshape(self.fields, n_cells, self.width)
        return jnp.sum(per_field * gathered[None], axis=2).reshape(self.fields * n_cells)

    def row_lengths(self) -> jnp.ndarray:
        return self._by_row_field(self.other_fields * self.lengths)

    def _fused_product(self, values: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        # The loop decodes each slot's column from the cell pattern itself, so ``columns()`` is not formed.
        return fused_product.cell_product(
            self.cells, self.lengths, values, x, self.other_fields, self.n_other
        )

    def entries(self) -> jnp.ndarray:
        _, _, live = self._slots()
        row_field = jnp.arange(self.fields, dtype=jnp.int32)[:, None, None]
        # Row (f, d): its stored entries are consecutive, from the row's first.
        cell = jnp.arange(self.cells.shape[0], dtype=jnp.int32)[None, :, None]
        slot = jnp.arange(self.width, dtype=jnp.int32)[None, None, :]
        start = (
            row_field * (self.other_fields * self.cell_nnz) + self.other_fields * self.indptr[cell]
        )
        entry = jnp.where(live[None], start + slot, self.n_in)
        return entry.reshape(self.fields * self.cells.shape[0], self.width)


class _PaddedRowOperator(eqx.Module):
    """A frozen level operator with every row padded to one width, owning its own matvec.

    Stored in padded-row form (the ELLPACK layout): each row's column indices and entries in
    ``(n_rows, width)`` slots, rows shorter than ``width`` filled with zero-valued slots. The product is
    a sum along each row of ``values * x[columns]``, which the index half computes
    (:meth:`PaddedColumns.product`): on the CPU, one fused loop over the stored slots threaded over rows
    (:mod:`~aquaflux.precondition.fused_product`); elsewhere a gather, a multiply and a row sum, whose
    gathered products are written out to memory before the row sum reads them back.

    The column indices are a :class:`PaddedColumns`: held as an array, or -- for an operator storing a
    cell pattern over full field-pair blocks -- computed inside the product from the cell pattern, so the
    operator holds its index arrays at a fraction of the size. The slots are the same either way, so the
    product is too; the cell-level form also gathers ``x`` once per cell row for all its row fields
    (:meth:`PaddedColumns.product`), which in single precision may sum a row in another order.

    **The transpose is stored only when asked for.** :meth:`with_transpose` packs ``A^T`` in the same
    padded form (``transposed``, ``t_values``), and :meth:`apply` tells ``jax.linear_transpose`` to apply
    it as a forward product. Derived automatically, the transpose of a gather is a scatter-add into every
    slot, which runs on one thread and collides on its outputs; the adjoint's transpose solve applies
    every level operator transposed, so it would pay a scatter per product. A forward solve never applies
    the transpose, so an operator is fitted without one -- a second copy of its storage the forward solve
    would never read -- and whoever transposes a map built on it asks first
    (:func:`~aquaflux.solve.with_transposes`; an adjoint's transpose solve asks by itself).
    Transposing an operator holding no transpose raises rather than falling back to the scatter.

    **Why not compressed-sparse-row (CSR).** A CSR product in JAX lowers to one dedicated sparse kernel
    call, which runs on one thread and does not fuse with the arithmetic around it. Measured on a
    coupled flow block's saddle hierarchy (pitzDaily, 12225 cells, the shipped field split at its hybrid
    start, 4-core Linux, jax 0.11.2): the finest level operator (36675 rows, 2.68M entries) took 6.5 ms
    as CSR against 4.4–5.2 ms padded, the Schur complement (12225 rows, 0.99M entries) 6.3 ms against
    1.8–2.0 ms, and one whole application of the field split 174 ms against 128 ms, equal to 1.2e-15.

    **What the padding costs.** A row is as wide as the widest row of its operator, so the storage is
    ``n_rows * width`` rather than the entry count. On a cell-graph operator every row of the fine
    level has nearly the same width (3-5 % padding on that hierarchy, 11 % on a 3D hexahedral one);
    aggregation's coarse levels vary more (up to 1.95x on the 2D hierarchy and 6.6x on the deepest
    levels of the 3D one) but are small, so a whole 3D field split stores 1.3x its entries. An operator
    with a few very wide rows -- a dense constraint row -- would pay that width on every row, and this
    form is the wrong one for it.

    Kept as its own object rather than loose arrays on each level because both level kinds carry one
    and both apply it the same way; the arrays never travel without each other.

    The arrays are all **traced** leaves and only ``shape`` is static, so re-deriving a hierarchy at a
    new operator on the same graph leaves every shape untouched -- the compiled V-cycle is a cache hit
    rather than a retrace, which is what makes a mid-march preconditioner refresh affordable.
    """

    rows: PaddedColumns
    values: jnp.ndarray  # (n_rows, width) the entries; a padding slot holds zero
    shape: tuple[int, int] = eqx.field(static=True)
    transposed: PaddedColumns | None = None  # the same, for A^T, once asked for
    t_values: jnp.ndarray | None = None  # (n_cols, the widest column of A)

    @classmethod
    def from_scipy(cls, a: sp.spmatrix, width: int | None = None) -> _PaddedRowOperator:
        """Freeze an assembled ``scipy`` matrix, each row's entries in ascending column order.

        Every stored entry is kept, an explicit zero included: :meth:`to_scipy` returns the same
        sparsity, which the structural patterns of a refit are computed from.

        Parameters
        ----------
        a : scipy.sparse matrix
            The operator to freeze.
        width : int, optional
            Pad every row to exactly this many slots, so an operator whose sparsity depends on a
            coarsening keeps a **fixed shape** across rebuilds and the compiled cycle stays a cache hit.
            ``None`` pads to the widest row of ``a``.

        Returns
        -------
        _PaddedRowOperator
            The frozen operator, holding no transpose (:meth:`with_transpose`).

        Raises
        ------
        ValueError
            If a row of ``a`` has more entries than ``width``. This is a **checked overflow, never a
            silent truncation** -- dropping entries would corrupt the Galerkin operator with no bound on
            the error, so the budget must be raised instead.
        """
        a = sp.csr_matrix(a)
        columns, values, lengths = _padded_rows(a, width, "row")
        return cls(
            rows=_HeldColumns(jnp.asarray(columns), jnp.asarray(lengths.astype(np.int32)), None),
            values=jnp.asarray(values),
            shape=a.shape,
        )

    def with_transpose(self) -> _PaddedRowOperator:
        """This operator holding its transpose too, so ``jax.linear_transpose`` applies it by gathering.

        ``A^T`` is packed by the slot rule the rows are (:func:`_padded_rows`), each row padded to the
        widest column of ``A``, in the values' own precision: the transpose's slots hold this operator's
        entries, moved, not recomputed. Read back through the host (:meth:`to_scipy`), so it needs
        concrete arrays, not traced ones -- build it before tracing the map that transposes it. An
        operator already holding its transpose is returned as it is.

        Raises
        ------
        RuntimeError
            If the operator's values are traced.
        """
        if self.transposed is not None:
            return self
        if isinstance(self.values, jax.core.Tracer):
            raise RuntimeError(
                "a padded-row operator's transpose is packed on the host, from concrete values: call "
                "aquaflux.solve.with_transposes() on the preconditioner before tracing the map that "
                "transposes it."
            )
        t_columns, t_values, t_lengths = _padded_rows(
            sp.csr_matrix(self.to_scipy().T), None, "column"
        )
        return _PaddedRowOperator(
            rows=self.rows,
            values=self.values,
            shape=self.shape,
            transposed=_HeldColumns(
                jnp.asarray(t_columns), jnp.asarray(t_lengths.astype(np.int32)), None
            ),
            t_values=jnp.asarray(t_values),
        )

    @property
    def columns(self) -> jnp.ndarray:
        """``(n_rows, width)`` column of each slot; a padding slot holds column 0."""
        return self.rows.columns()

    @property
    def row_lengths(self) -> jnp.ndarray:
        """``(n_rows,)`` stored entries in each row; the slots after them are padding."""
        return self.rows.row_lengths()

    @property
    def width(self) -> int:
        """Slots per row: the widest row, or the budget the operator was padded to."""
        return int(self.values.shape[1])

    @property
    def nnz(self) -> int:
        """Stored entries, padding slots excluded."""
        return int(np.sum(np.asarray(self.row_lengths)))

    def apply(self, x: jnp.ndarray) -> jnp.ndarray:
        """``A x``. Linear; transposed, it applies the stored ``A^T`` as a forward product.

        Works under ``jax.jit``, forward- and reverse-mode differentiation and
        ``jax.linear_transpose`` -- the last only once the operator holds its transpose
        (:meth:`with_transpose`); without it, transposing raises. Not under ``jax.vmap``: the rule that
        hands the transpose to ``A^T`` has no batching rule.
        """
        return linear_call(
            _padded_product,
            _padded_product_transposed,
            (self.rows, self.values, self.transposed, self.t_values),
            x,
        )

    def to_scipy(self) -> sp.csr_matrix:
        """The exact inverse of :meth:`from_scipy`: this operator as a host ``scipy`` CSR matrix.

        The one place a traced level's operator is read back onto the host -- a march refresh
        re-deriving a hierarchy at a new operator, a SIMPLE-smoothed level forming its Schur pieces --
        so every such consumer shares one conversion. Padding slots are dropped; stored zeros are not.
        """
        lengths = np.asarray(self.row_lengths)
        stored = np.arange(self.width)[None, :] < lengths[:, None]
        indptr = np.concatenate([[0], np.cumsum(lengths)])
        return sp.csr_matrix(
            (np.asarray(self.values)[stored], np.asarray(self.columns)[stored], indptr),
            shape=self.shape,
        )

    @property
    def diagonal(self) -> jnp.ndarray:
        """The operator's diagonal, read off the stored entries (a padding slot holds zero)."""
        rows = jnp.arange(self.shape[0])[:, None]
        return jnp.sum(jnp.where(self.columns == rows, self.values, 0.0), axis=1)


def _padded_rows(a: sp.csr_matrix, width: int | None, what: str):
    """``a``'s rows as ``(columns, values, lengths)``, each row padded to ``width`` slots."""
    a.sort_indices()
    lengths = np.diff(a.indptr)
    width = _checked_width(lengths, width, what)
    rows = np.repeat(np.arange(a.shape[0]), lengths)
    # Each entry's flat slot: its row's first slot plus its place within the row.
    flat = np.arange(a.nnz) + rows * width - a.indptr[rows]
    columns = np.zeros((a.shape[0], width), dtype=np.int32)
    values = np.zeros((a.shape[0], width), dtype=a.dtype)
    columns.ravel()[flat] = a.indices
    values.ravel()[flat] = a.data
    return columns, values, lengths


class _PaddedLayout(eqx.Module):
    """Where each stored entry of a fixed-pattern operator lands in its padded-row form.

    :meth:`_PaddedRowOperator.from_scipy` derives every slot from the operator's index arrays; on a
    pattern that does not move, the packing is the same gather every time. This holds the rows'
    :class:`PaddedColumns` -- each slot's column and the stored entry it is filled from, read off one run
    of :func:`_padded_rows` over the numbered pattern (so the slot rule has one home), or computed from
    the cell pattern when the operator stores one over full field-pair blocks -- on the device, so
    packing moves only values and runs either from a host matrix (:meth:`operator`) or, traced, from
    values already on the device (:meth:`packed`). The transpose is not packed here: an operator packs
    its own when asked (:meth:`_PaddedRowOperator.with_transpose`).

    An ``equinox.Module`` whose arrays are leaves and whose shape is static, so a layout rides into a
    jitted refit as an argument. An operand read row by row and never applied uses :meth:`row_values`.
    """

    rows: PaddedColumns
    shape: tuple[int, int] = eqx.field(static=True)
    #: Stored entries of the pattern, the length of the values a layout packs.
    nnz: int = eqx.field(static=True)
    width: int = eqx.field(static=True)

    @classmethod
    def of(
        cls,
        pattern: SparsePattern,
        width: int | None = None,
        *,
        fields: tuple[int, int] | None = None,
    ) -> _PaddedLayout:
        """The layout of ``pattern`` padded to ``width`` slots.

        ``None`` pads to the widest row, as :meth:`_PaddedRowOperator.from_scipy` does. ``fields`` -- the
        operator's row and column fields per cell, when the caller knows them -- lets a pattern that is
        one cell pattern over every field pair be held at cell level (:class:`_KroneckerColumns`); any
        other is held as arrays.
        """
        kronecker = None if fields is None else KroneckerPattern.of_dof(pattern, *fields)
        if kronecker is not None:
            width = _checked_width(
                kronecker.column_fields * np.diff(kronecker.cells.indptr), width, "row"
            )
            rows = _KroneckerColumns.of(kronecker, width)
        else:
            numbered = pattern.numbered()
            columns, numbers, lengths = _padded_rows(numbered, width, "row")
            rows = _held(columns, lengths, EntryMap.placing(pattern, numbers))
            width = columns.shape[1]
        return cls(rows=rows, shape=pattern.shape, nnz=pattern.nnz, width=width)

    @property
    def columns(self) -> jnp.ndarray:
        """``(n_rows, width)`` the slot columns of every operator packed by this layout."""
        return self.rows.columns()

    @property
    def row_lengths(self) -> jnp.ndarray:
        """``(n_rows,)`` stored entries in each row."""
        return self.rows.row_lengths()

    def row_values(self, values: jnp.ndarray) -> jnp.ndarray:
        """The slot values of ``values``, shape ``(nnz,)``, on this layout's rows: ``(n_rows, width)``."""
        return _gather_slots(values, self.rows.entries())

    def packed(self, values: jnp.ndarray) -> _PaddedRowOperator:
        """The operator storing ``values`` on this layout's pattern, shape ``(nnz,)``, traced.

        In ``values``'s own floating-point type: a gather moves entries without arithmetic, so it is
        exact in any type, and a single-precision operator is packed into single-precision slots.
        """
        return self.with_values(self.row_values(values))

    def with_values(self, values: jnp.ndarray) -> _PaddedRowOperator:
        """The operator holding this layout's index arrays and the given slot values.

        Parameters
        ----------
        values : jnp.ndarray
            The slot values, shape ``(n_rows, width)``.
        """
        return _PaddedRowOperator(rows=self.rows, values=values, shape=self.shape)

    def operator(self, a: sp.csr_matrix | DeviceMatrix) -> _PaddedRowOperator:
        """``a`` in padded-row form: equal, slot for slot, to ``_PaddedRowOperator.from_scipy(a, ...)``.

        ``a`` must store this layout's pattern (the caller holds the layout keyed by it); a
        :class:`~aquaflux.precondition.entry_map.DeviceMatrix` is packed from its values where they are.
        The gather is compiled as one program per layout shape (run eagerly, it would be a dispatch of
        its own), and the operator holds this layout's own index arrays: a compiled function's outputs
        are new buffers, so only the slot values come out of it.
        """
        values = a.values if isinstance(a, DeviceMatrix) else jnp.asarray(a.data)
        return self.with_values(_slot_values(self, values))


def _held(columns: np.ndarray, lengths: np.ndarray, placing: EntryMap) -> _HeldColumns:
    """Padded rows held as arrays, their entries the gather ``placing`` describes, on the device."""
    gather = placing.on_device()
    return _HeldColumns(
        jnp.asarray(columns),
        jnp.asarray(lengths.astype(np.int32)),
        gather.slots.reshape(columns.shape),
    )


def _checked_width(lengths: np.ndarray, width: int | None, what: str) -> int:
    """The padded width of rows of these lengths: ``width``, refused if a row overflows it, else the widest."""
    widest = int(lengths.max()) if lengths.size else 0
    if width is None:
        return max(widest, 1)
    if widest > width:
        raise ValueError(
            f"operator has a {what} of {widest} entries, wider than the {width} budgeted for this "
            f"level. Raise the budget (see SmoothedHierarchy.shape_budget's headroom); entries are "
            f"never dropped to fit."
        )
    return width


def _gather_slots(values: jnp.ndarray, entries: jnp.ndarray) -> jnp.ndarray:
    """``values`` (shape ``(nnz,)``) in slots: the entry each slot names, zero where it names ``nnz``."""
    return jnp.concatenate([values, jnp.zeros(1, dtype=values.dtype)])[entries]


@eqx.filter_jit
def _slot_values(layout: _PaddedLayout, values: jnp.ndarray) -> jnp.ndarray:
    """The padded-row slot values of ``values`` on ``layout``, compiled."""
    return layout.row_values(values)


def _padded_operator(
    a: sp.spmatrix | DeviceMatrix,
    width: int | None = None,
    maps: EntryMaps | None = None,
    slot: Hashable = "padded",
    fields: tuple[int, int] | None = None,
) -> _PaddedRowOperator:
    """``a`` in padded-row form: by the layout ``maps`` holds under ``slot`` if given, else packed afresh.

    A :class:`~aquaflux.precondition.entry_map.DeviceMatrix` needs ``maps``: its layout is keyed by its
    pattern, and it is packed on the device. ``fields`` is :meth:`_PaddedLayout.of`'s, for a held layout.
    """
    if maps is None:
        return _PaddedRowOperator.from_scipy(a, width)
    on_device = isinstance(a, DeviceMatrix)
    if not on_device:
        a = sp.csr_matrix(a)
    return maps.held(
        (slot, width, fields),
        a.pattern if on_device else a,
        lambda pattern: _PaddedLayout.of(pattern, width, fields=fields),
    ).operator(a)


def _padded_product(arrays, x: jnp.ndarray) -> jnp.ndarray:
    rows, values, _, _ = arrays
    return rows.product(values, x)


def _padded_product_transposed(arrays, y: jnp.ndarray) -> jnp.ndarray:
    _, _, transposed, t_values = arrays
    if transposed is None:
        raise RuntimeError(
            "a padded-row operator was transposed without its stored transpose, which a forward "
            "solve does not hold. Pack it first: aquaflux.solve.with_transposes(tree) over the "
            "preconditioner (root_adjoint does this for an adjoint's preconditioner factory; "
            "reverse-mode differentiation through a preconditioned linear solve, such as "
            "newton_step's, needs the factory passed through it by its caller; a map that closes "
            "over its operators instead of holding them as leaves cannot be reached that way), or "
            "_PaddedRowOperator.with_transpose() for one operator."
        )
    return transposed.product(t_values, y)
