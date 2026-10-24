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

import numpy as np

from .entry_map import SparsePattern

__all__ = ["KroneckerPattern", "entry_position"]


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
        """The same pattern at degree-of-freedom level, as the compressed-sparse-row arrays spell it.

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
        return SparsePattern.of_arrays(self.shape, row_pointers, np.tile(template, self.row_fields))

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
