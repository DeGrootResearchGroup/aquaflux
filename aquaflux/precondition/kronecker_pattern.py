"""A field-major sparsity pattern that is one cell pattern repeated over a full block of field pairs.

A coupled Jacobian probed on a mesh couples cell ``i`` to cell ``j`` through every pair of fields, or
through none: the stencil is a property of the cell graph, and every field of a cell sees it. Laid out
field-major -- degree of freedom ``(cell i, field f)`` at ``f * n_cells + i`` -- such a matrix stores the
Kronecker product of a field-pair block that is full and a **cell pattern** ``C``: entry ``(a n + i, b n
+ j)`` is stored exactly when ``(i, j)`` is stored in ``C``. Nodal aggregation keeps the form (its
prolongation is an aggregate indicator times the identity over the fields), so every coarse level of a
nodal hierarchy built on such a matrix has it too.

Every index array a degree-of-freedom description of such a pattern would hold -- column indices, row
pointers, where each entry sits -- is then a function of the cell pattern and two field counts, at a
fraction of the size: the field pair is implicit. :class:`KroneckerPattern` holds that description, and
:func:`entry_position` is the one formula placing an entry in the degree-of-freedom compressed-sparse-row
(CSR) order -- rows ascending, each row's columns ascending -- so that the stored values keep exactly the
order a degree-of-freedom pattern gives them.
"""

from __future__ import annotations

import weakref

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .entry_map import EntryMap, SparsePattern, ValueMap

__all__ = [
    "CellLevelPattern",
    "KroneckerGalerkin",
    "KroneckerGather",
    "KroneckerPattern",
    "entry_position",
    "nodal_cell_transfer",
]


def entry_position(indptr, row, entry, row_field, column_field, column_fields: int, cell_nnz: int):
    """Where the entry for field pair ``(row_field, column_field)`` at cell entry ``entry`` sits.

    The degree-of-freedom row ``(row_field, row)`` stores ``column_fields * length(row)`` entries, its
    columns ascending, so the field pair's block of the row is the ``column_field``-th run of the cell
    row's length. Rows are stored field by field, each field's rows in cell order. Works elementwise on
    NumPy and JAX arrays alike, so a traced kernel and a host build place an entry by the same formula.

    Parameters
    ----------
    indptr : array of int
        The cell pattern's row pointers, shape ``(n_row_cells + 1,)``.
    row, entry : array of int
        The cell row and the cell entry (its position in the cell pattern's storage), broadcastable.
    row_field, column_field : array of int
        The field pair, each counted from the first field of the pattern, broadcastable.
    column_fields : int
        Column fields in the pattern.
    cell_nnz : int
        Stored entries of the cell pattern.

    Returns
    -------
    array of int
        The entry's position among the degree-of-freedom pattern's stored entries.
    """
    start = indptr[row]
    length = indptr[row + 1] - start
    return (
        row_field * (column_fields * cell_nnz)
        + column_fields * start
        + column_field * length
        + (entry - start)
    )


class KroneckerPattern:
    """A cell pattern over a full block of field pairs, field-major on both sides.

    Attributes
    ----------
    cells : SparsePattern
        The cell pattern, each row's columns ascending, shape ``(n_row_cells, n_column_cells)``.
    row_fields, column_fields : int
        Fields per row cell and per column cell.
    """

    def __init__(self, cells: SparsePattern, row_fields: int, column_fields: int) -> None:
        indptr, indices = cells.indptr, cells.indices
        rows = np.repeat(np.arange(cells.shape[0]), np.diff(indptr))
        if np.any((indices[1:] <= indices[:-1]) & (rows[1:] == rows[:-1])):
            raise ValueError(
                "KroneckerPattern: the cell pattern's rows must store their columns ascending and "
                "once each, or the degree-of-freedom order it describes is not a CSR order."
            )
        self.cells = cells
        self.row_fields = int(row_fields)
        self.column_fields = int(column_fields)

    @classmethod
    def of_coordinates(
        cls, rows: np.ndarray, columns: np.ndarray, shape: tuple[int, int], fields: tuple[int, int]
    ) -> tuple[KroneckerPattern, np.ndarray]:
        """The pattern storing the cell coordinates ``(rows, columns)``, and the order that sorts them.

        Parameters
        ----------
        rows, columns : np.ndarray
            Distinct cell coordinates, any order, shape ``(n_entries,)`` each.
        shape : tuple of int
            The cell pattern's shape.
        fields : tuple of int
            ``(row_fields, column_fields)``.

        Returns
        -------
        pattern : KroneckerPattern
            The pattern.
        order : np.ndarray
            ``order[e]`` is the coordinate stored as cell entry ``e``.
        """
        order = np.lexsort((columns, rows))
        index = np.int32 if max(shape[1], rows.size) < np.iinfo(np.int32).max else np.int64
        counts = np.bincount(rows, minlength=shape[0])
        indptr = np.concatenate([[0], np.cumsum(counts)]).astype(index)
        cells = SparsePattern.of_arrays(shape, indptr, columns[order].astype(index))
        return cls(cells, *fields), order

    @classmethod
    def of_dof(
        cls, pattern: SparsePattern, row_fields: int, column_fields: int
    ) -> KroneckerPattern | None:
        """``pattern`` as a cell pattern over ``row_fields x column_fields`` field pairs, if it is one.

        The candidate cell pattern is the first row field's entries in the first column field; the
        pattern is that cell pattern over full field-pair blocks exactly when expanding the candidate
        gives back its arrays. ``None`` when it does not -- a pattern missing some field pair of some
        cell entry, or storing a row's columns out of order.
        """
        rows, columns = pattern.shape
        if row_fields < 1 or column_fields < 1 or rows % row_fields or columns % column_fields:
            return None
        n_rows, n_columns = rows // row_fields, columns // column_fields
        indptr, indices = pattern.indptr, pattern.indices
        first = np.repeat(np.arange(n_rows), np.diff(indptr[: n_rows + 1]))
        kept = np.flatnonzero(indices[: indptr[n_rows]] < n_columns)
        counts = np.bincount(first[kept], minlength=n_rows)
        cell_indices = indices[kept]
        if cell_indices.size * row_fields * column_fields != pattern.nnz:
            return None
        cells = SparsePattern.of_arrays(
            (n_rows, n_columns),
            np.concatenate([[0], np.cumsum(counts)]).astype(indptr.dtype),
            cell_indices.astype(indices.dtype),
        )
        try:
            candidate = cls(cells, row_fields, column_fields)
        except ValueError:
            return None
        dof_indptr, dof_indices = candidate.dof_arrays()
        if np.array_equal(dof_indptr, indptr) and np.array_equal(dof_indices, indices):
            return candidate
        return None

    @property
    def shape(self) -> tuple[int, int]:
        """The degree-of-freedom shape."""
        return (
            self.row_fields * self.cells.shape[0],
            self.column_fields * self.cells.shape[1],
        )

    @property
    def nnz(self) -> int:
        """Stored degree-of-freedom entries."""
        return self.row_fields * self.column_fields * self.cells.nnz

    def cell_rows(self) -> np.ndarray:
        """The cell row of each cell entry, shape ``(cells.nnz,)``."""
        return np.repeat(
            np.arange(self.cells.shape[0], dtype=self.cells.indices.dtype),
            np.diff(self.cells.indptr),
        )

    def dof_pattern(self) -> SparsePattern:
        """The same pattern at degree-of-freedom level, as the compressed-sparse-row arrays spell it."""
        return SparsePattern.of_arrays(self.shape, *self.dof_arrays())

    def dof_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """The degree-of-freedom pattern's row pointers and column indices, formed afresh each call.

        Every row of one field repeats the column template of the rows of the first, so the template is
        built once, for one row field, and repeated.
        """
        n_columns = self.cells.shape[1]
        indptr, indices = self.cells.indptr, self.cells.indices
        index = np.int32 if max(self.nnz, self.shape[1]) < np.iinfo(np.int32).max else np.int64
        lengths = np.diff(indptr).astype(index)
        rows = self.cell_rows()
        entries = np.arange(self.cells.nnz, dtype=index)
        template = np.empty(self.column_fields * self.cells.nnz, dtype=index)
        for column_field in range(self.column_fields):
            place = entry_position(
                indptr.astype(index), rows, entries, 0, column_field, self.column_fields, 0
            )
            template[place] = column_field * index(n_columns) + indices
        row_pointers = np.concatenate(
            [
                [0],
                np.cumsum(np.tile(self.column_fields * lengths, self.row_fields)),
            ]
        ).astype(index)
        return row_pointers, np.tile(template, self.row_fields)

    def same_as(self, other: KroneckerPattern) -> bool:
        """Whether ``other`` stores this very pattern: the same fields over the same cell pattern."""
        return (
            self.row_fields == other.row_fields
            and self.column_fields == other.column_fields
            and (other.cells is self.cells or self.cells.matches(other.cells))
        )

    def diagonal_positions(self) -> np.ndarray:
        """Where each degree of freedom's diagonal entry is stored, shape ``(shape[0],)``.

        Raises
        ------
        ValueError
            If the pattern is not square in its fields and cells, or a cell row stores no diagonal.
        """
        if self.row_fields != self.column_fields or self.cells.shape[0] != self.cells.shape[1]:
            raise ValueError("KroneckerPattern: only a square pattern has a diagonal.")
        rows = self.cell_rows()
        diagonal = np.flatnonzero(self.cells.indices == rows)
        if diagonal.size != self.cells.shape[0]:
            raise ValueError("KroneckerPattern: some cell row stores no diagonal entry.")
        index = np.int32 if self.nnz < np.iinfo(np.int32).max else np.int64
        fields = np.arange(self.row_fields, dtype=index)[:, None]
        cells = np.arange(self.cells.shape[0], dtype=index)[None, :]
        return entry_position(
            self.cells.indptr.astype(index),
            cells,
            diagonal.astype(index)[None, :],
            fields,
            fields,
            self.column_fields,
            self.cells.nnz,
        ).ravel()


class CellLevelPattern(SparsePattern):
    """A :class:`~aquaflux.precondition.entry_map.SparsePattern` held as its cell pattern over every field pair.

    For a pattern held for a whole session but read as arrays only now and then -- a probe's block
    patterns, against which every refit's operands are checked by identity -- the degree-of-freedom row
    pointers and column indices are formed when asked and not kept: :attr:`indptr` and :attr:`indices`
    each expand the cell pattern afresh (:meth:`KroneckerPattern.dof_arrays`), so a caller reading both
    should read them once. :meth:`matches` compares cell patterns where the other side is held the same
    way, and expands only against a pattern held as arrays.

    Attributes
    ----------
    kronecker : KroneckerPattern
        The cell pattern and the field counts.
    """

    def __init__(self, kronecker: KroneckerPattern) -> None:
        self.kronecker = kronecker
        self.shape = kronecker.shape
        SparsePattern._live.setdefault((self.shape, self.nnz), weakref.WeakSet()).add(self)

    @property
    def indptr(self) -> np.ndarray:
        """The degree-of-freedom row pointers, formed on this call."""
        return self.kronecker.dof_arrays()[0]

    @property
    def indices(self) -> np.ndarray:
        """The degree-of-freedom column indices, formed on this call."""
        return self.kronecker.dof_arrays()[1]

    @property
    def nnz(self) -> int:
        """Stored entries."""
        return self.kronecker.nnz

    def matches(self, matrix: sp.csr_matrix | SparsePattern) -> bool:
        """Whether ``matrix`` stores exactly this pattern's entries, in this pattern's order."""
        if matrix is self:
            return True
        if matrix.shape != self.shape or matrix.nnz != self.nnz:
            return False
        if isinstance(matrix, CellLevelPattern):
            return self.kronecker.same_as(matrix.kronecker)
        indptr, indices = self.kronecker.dof_arrays()
        return np.array_equal(matrix.indptr, indptr) and np.array_equal(matrix.indices, indices)

    def csr(self, values: np.ndarray) -> sp.csr_matrix:
        """``values`` on this pattern, on index arrays formed for it."""
        indptr, indices = self.kronecker.dof_arrays()
        return sp.csr_matrix((values, indices, indptr), shape=self.shape, copy=False)

    def numbered(self) -> sp.csr_matrix:
        """As :meth:`SparsePattern.numbered`, on index arrays formed for it."""
        return self.csr(np.arange(1, self.nnz + 1, dtype=np.float64))


def nodal_cell_transfer(transfer: sp.csr_matrix, fields: int) -> sp.csr_matrix | None:
    """The cell transfer ``T`` of a field-major transfer that is ``I_fields ⊗ T``, else ``None``.

    A nodal aggregation's prolongation carries each field of a cell on that field's own coarse unknown,
    so it is the identity over the fields times one cell-to-aggregate transfer; that transfer is read
    off the first field's rows and accepted only if repeating it over the fields gives ``transfer`` back,
    entry for entry.
    """
    rows, columns = transfer.shape
    if fields < 1 or rows % fields or columns % fields:
        return None
    transfer = sp.csr_matrix(transfer)
    transfer.sort_indices()
    n_rows, n_columns = rows // fields, columns // fields
    cell = sp.csr_matrix(transfer[:n_rows, :n_columns])
    cell.sort_indices()
    expected = sp.csr_matrix(sp.kron(sp.identity(fields, format="csr"), cell, format="csr"))
    expected.sort_indices()
    same = (
        expected.nnz == transfer.nnz
        and np.array_equal(expected.indptr, transfer.indptr)
        and np.array_equal(expected.indices, transfer.indices)
        and np.array_equal(expected.data, transfer.data)
    )
    return cell if same else None


class KroneckerGalerkin(ValueMap):
    """``Pᵀ A P`` for ``P = I ⊗ T`` and ``A`` a cell pattern over full field-pair blocks, at cell level.

    The product of a cell pattern over every field pair is the coarse cell pattern over every field
    pair, and each field pair's coarse entry sums the same fine cell entries, with the same weights, as
    every other pair's: the map ``Tᵀ C T`` over the cell pattern says which. So the map holds that cell
    map's terms -- source cell entry, its row, target coarse cell entry, sorted by target, and their
    weights where not all one -- and places each field pair's terms by
    :func:`entry_position` on both sides. Each target sums its terms in source order, as the
    degree-of-freedom map's segment sum does.

    Attributes
    ----------
    source, source_rows, target : jnp.ndarray
        Each term's fine cell entry, its fine cell row, and its coarse cell entry; shape ``(n_terms,)``,
        sorted by ``target``.
    weight : jnp.ndarray or None
        Each term's factor; ``None`` when every factor is one.
    fine_indptr, coarse_indptr : jnp.ndarray
        The two cell patterns' row pointers.
    coarse_rows : jnp.ndarray
        The coarse cell row of each coarse cell entry, shape ``(coarse_nnz,)``.
    fields : int
        Fields per cell, on both levels. Static.
    fine_nnz, coarse_nnz : int
        Stored entries of the two cell patterns. Static.
    """

    source: jnp.ndarray
    source_rows: jnp.ndarray
    target: jnp.ndarray
    weight: jnp.ndarray | None
    fine_indptr: jnp.ndarray
    coarse_indptr: jnp.ndarray
    coarse_rows: jnp.ndarray
    fields: int = eqx.field(static=True)
    fine_nnz: int = eqx.field(static=True)
    coarse_nnz: int = eqx.field(static=True)

    @classmethod
    def of(
        cls, pattern: KroneckerPattern, transfer: sp.csr_matrix
    ) -> tuple[KroneckerGalerkin, KroneckerPattern]:
        """The map for ``pattern`` (square, ``fields x fields``) and the cell transfer ``T``.

        Returns
        -------
        galerkin : KroneckerGalerkin
            The map.
        coarse : KroneckerPattern
            The coarse operator's pattern: the cell product's structural pattern over every field pair.
        """
        cells = EntryMap.product(sp.csr_matrix(transfer.T), pattern.cells, transfer)
        source, target = cells.source, cells.target
        order = np.argsort(target, kind="stable")
        source, target = source[order], target[order]
        rows = pattern.cell_rows()
        coarse = KroneckerPattern(cells.landing, pattern.row_fields, pattern.column_fields)
        index = np.int32
        return (
            cls(
                source=jnp.asarray(source.astype(index)),
                source_rows=jnp.asarray(rows[source].astype(index)),
                target=jnp.asarray(target.astype(index)),
                weight=None if cells.weight is None else jnp.asarray(cells.weight[order]),
                fine_indptr=jnp.asarray(pattern.cells.indptr.astype(index)),
                coarse_indptr=jnp.asarray(coarse.cells.indptr.astype(index)),
                coarse_rows=jnp.asarray(coarse.cell_rows().astype(index)),
                fields=pattern.row_fields,
                fine_nnz=pattern.cells.nnz,
                coarse_nnz=coarse.cells.nnz,
            ),
            coarse,
        )

    def apply(self, values: jnp.ndarray) -> jnp.ndarray:
        """The coarse operator's stored values from the fine operator's, both in their CSR order."""
        b = self.fields
        pair = jnp.arange(b * b, dtype=jnp.int32)[:, None]
        row_field, column_field = pair // b, pair % b
        fine = entry_position(
            self.fine_indptr,
            self.source_rows[None],
            self.source[None],
            row_field,
            column_field,
            b,
            self.fine_nnz,
        )
        terms = values[fine]
        if self.weight is not None:
            terms = terms * self.weight.astype(values.dtype)[None]
        segment = self.target[None] + pair * self.coarse_nnz
        summed = jax.ops.segment_sum(
            terms.ravel(),
            segment.ravel(),
            num_segments=b * b * self.coarse_nnz,
            indices_are_sorted=True,
        )
        entry = jnp.arange(self.coarse_nnz, dtype=jnp.int32)[None]
        coarse = entry_position(
            self.coarse_indptr,
            self.coarse_rows[entry],
            entry,
            row_field,
            column_field,
            b,
            self.coarse_nnz,
        )
        return (
            jnp.zeros(b * b * self.coarse_nnz, dtype=values.dtype)
            .at[coarse.ravel()]
            .set(summed, unique_indices=True)
        )


class KroneckerGather(ValueMap):
    """A block of field pairs of a cell pattern over full field-pair blocks, read into its own CSR order.

    The output stores the cell pattern over the row fields ``[row_start, row_start + rows)`` and the
    column fields ``[column_start, column_start + columns)`` of the operand, each entry the operand's
    entry for the same field pair and cell entry. The operand is the pattern's stored values in CSR order
    or, with ``slot_width``, the same operator's padded-row slots of that width (the slot of an entry is
    its place in its row), flattened. A placement only, so exact in any floating-point type.

    Attributes
    ----------
    indptr, rows : jnp.ndarray
        The cell pattern's row pointers and each cell entry's row.
    fields : int
        Column fields of the operand. Static.
    row_start, rows_out, column_start, columns_out : int
        The block's field ranges. Static.
    cell_nnz : int
        Stored entries of the cell pattern. Static.
    slot_width : int
        The operand's padded-row width, or ``0`` for an operand in CSR order. Static.
    """

    indptr: jnp.ndarray
    rows: jnp.ndarray
    fields: int = eqx.field(static=True)
    row_start: int = eqx.field(static=True)
    rows_out: int = eqx.field(static=True)
    column_start: int = eqx.field(static=True)
    columns_out: int = eqx.field(static=True)
    cell_nnz: int = eqx.field(static=True)
    slot_width: int = eqx.field(static=True)

    @classmethod
    def of(
        cls,
        pattern: KroneckerPattern,
        rows: tuple[int, int],
        columns: tuple[int, int],
        *,
        slot_width: int = 0,
    ) -> tuple[KroneckerGather, SparsePattern]:
        """The gather of the field block ``rows x columns`` (half-open ranges), and its pattern."""
        index = np.int32
        block = KroneckerPattern(pattern.cells, rows[1] - rows[0], columns[1] - columns[0])
        return (
            cls(
                indptr=jnp.asarray(pattern.cells.indptr.astype(index)),
                rows=jnp.asarray(pattern.cell_rows().astype(index)),
                fields=pattern.column_fields,
                row_start=rows[0],
                rows_out=rows[1] - rows[0],
                column_start=columns[0],
                columns_out=columns[1] - columns[0],
                cell_nnz=pattern.cells.nnz,
                slot_width=int(slot_width),
            ),
            block.dof_pattern(),
        )

    def apply(self, values: jnp.ndarray) -> jnp.ndarray:
        """The block's stored values, in its CSR order, from the operand's."""
        row_field = jnp.arange(self.rows_out, dtype=jnp.int32)[:, None, None]
        column_field = jnp.arange(self.columns_out, dtype=jnp.int32)[None, :, None]
        entry = jnp.arange(self.cell_nnz, dtype=jnp.int32)[None, None, :]
        row = self.rows[entry]
        if self.slot_width:
            start = self.indptr[row]
            length = self.indptr[row + 1] - start
            n_cells = self.indptr.shape[0] - 1
            source = (
                ((row_field + self.row_start) * n_cells + row) * self.slot_width
                + (column_field + self.column_start) * length
                + (entry - start)
            )
        else:
            source = entry_position(
                self.indptr,
                row,
                entry,
                row_field + self.row_start,
                column_field + self.column_start,
                self.fields,
                self.cell_nnz,
            )
        target = entry_position(
            self.indptr, row, entry, row_field, column_field, self.columns_out, self.cell_nnz
        )
        out = self.rows_out * self.columns_out * self.cell_nnz
        return (
            jnp.zeros(out, dtype=values.dtype)
            .at[target.ravel()]
            .set(values[source.ravel()], unique_indices=True)
        )
