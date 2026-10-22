"""The padded-row (ELLPACK) level operator, and the fixed gathers that pack one on a held pattern.

Every multigrid level, every SIMPLE-smoother piece and the field split's coupling store their frozen
operator this way: two ``(n_rows, width)`` arrays of column indices and entries, rows shorter than
``width`` filled with zero-valued slots, and the same for the transpose. :class:`_PaddedRowOperator` is
the operator and its matvec; :func:`_padded_rows` is the one slot rule; :class:`_PaddedLayout` holds the
packing of an operator whose pattern does not move as two fixed gathers (entry maps), so a refit moves
only values; :func:`_padded_operator` picks between packing afresh and the held layout.
"""

from __future__ import annotations

from collections.abc import Hashable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.custom_derivatives import linear_call

from .entry_map import EntryMap, EntryMaps, SparsePattern


class _PaddedRowOperator(eqx.Module):
    """A frozen level operator with every row padded to one width, owning its own matvec.

    Stored in padded-row form (the ELLPACK layout): two ``(n_rows, width)`` arrays holding each row's
    column indices and entries, rows shorter than ``width`` filled with zero-valued slots. The product
    is then a gather of ``x`` at the stored columns, a multiply, and a sum along each row, each step
    split across threads by the compiler. (The gather is written out to memory before the row sum
    reads it back; the two are not fused into one loop.)

    **The transpose is stored too.** ``A^T`` is kept in the same padded form (``t_columns``,
    ``t_values``), and :meth:`apply` tells ``jax.linear_transpose`` to apply it as a forward product.
    Derived automatically, the transpose of a gather is a scatter-add into every slot, which runs on
    one thread and collides on its outputs; the adjoint's transpose solve applies every level operator
    that way, so it paid a scatter per product. The cost is a second copy of the operator's storage.

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

    columns: jnp.ndarray  # (n_rows, width) column of each slot; a padding slot holds column 0
    values: jnp.ndarray  # (n_rows, width) the entries; a padding slot holds zero
    row_lengths: (
        jnp.ndarray
    )  # (n_rows,) stored entries in each row; the slots after them are padding
    shape: tuple[int, int] = eqx.field(static=True)
    t_columns: jnp.ndarray  # (n_cols, transpose_width) the same, for A^T
    t_values: jnp.ndarray  # (n_cols, transpose_width)

    @classmethod
    def from_scipy(
        cls, a: sp.spmatrix, width: int | None = None, transpose_width: int | None = None
    ) -> _PaddedRowOperator:
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
        transpose_width : int, optional
            The same for the stored ``A^T``: slots per column of ``a``. ``None`` pads to the widest
            column.

        Returns
        -------
        _PaddedRowOperator
            The frozen operator.

        Raises
        ------
        ValueError
            If a row of ``a`` has more entries than ``width``, or a column more than
            ``transpose_width``. This is a **checked overflow, never a silent truncation** -- dropping entries would corrupt the Galerkin operator with no bound on
            the error, so the budget must be raised instead.
        """
        a = sp.csr_matrix(a)
        columns, values, lengths = _padded_rows(a, width, "row")
        t_columns, t_values, _ = _padded_rows(sp.csr_matrix(a.T), transpose_width, "column")
        return cls(
            columns=jnp.asarray(columns),
            values=jnp.asarray(values),
            row_lengths=jnp.asarray(lengths.astype(np.int32)),
            shape=a.shape,
            t_columns=jnp.asarray(t_columns),
            t_values=jnp.asarray(t_values),
        )

    @property
    def width(self) -> int:
        """Slots per row: the widest row, or the budget the operator was padded to."""
        return int(self.values.shape[1])

    @property
    def transpose_width(self) -> int:
        """Slots per row of the stored transpose: the widest column, or its budget."""
        return int(self.t_values.shape[1])

    @property
    def nnz(self) -> int:
        """Stored entries, padding slots excluded."""
        return int(np.sum(np.asarray(self.row_lengths)))

    def apply(self, x: jnp.ndarray) -> jnp.ndarray:
        """``A x``. Linear; transposed, it applies the stored ``A^T`` as a forward product.

        Works under ``jax.jit``, forward- and reverse-mode differentiation and
        ``jax.linear_transpose``, but not under ``jax.vmap``: the rule that hands the transpose to
        ``A^T`` has no batching rule.
        """
        return linear_call(
            _padded_product,
            _padded_product_transposed,
            (self.columns, self.values, self.t_columns, self.t_values),
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
    widest = int(lengths.max()) if lengths.size else 0
    if width is None:
        width = max(widest, 1)
    elif widest > width:
        raise ValueError(
            f"operator has a {what} of {widest} entries, wider than the {width} budgeted for this "
            f"level. Raise the budget (see SmoothedHierarchy.shape_budget's headroom); entries are "
            f"never dropped to fit."
        )
    rows = np.repeat(np.arange(a.shape[0]), lengths)
    # Each entry's flat slot: its row's first slot plus its place within the row.
    flat = np.arange(a.nnz) + rows * width - a.indptr[rows]
    columns = np.zeros((a.shape[0], width), dtype=np.int32)
    values = np.zeros((a.shape[0], width), dtype=a.dtype)
    columns.ravel()[flat] = a.indices
    values.ravel()[flat] = a.data
    return columns, values, lengths


class _PaddedLayout:
    """Where each stored entry of a fixed-pattern operator lands in its padded-row form, both ways.

    :meth:`_PaddedRowOperator.from_scipy` derives every slot from the operator's index arrays and
    transposes it to pack ``Aᵀ``; on a pattern that does not move, both packings are the same gather
    every time. This holds the two gathers (:class:`EntryMap`, read off one run of :func:`_padded_rows`
    over the numbered pattern each way, so the slot rule has one home) and the index arrays they fill,
    already on the device, so :meth:`operator` moves only values.
    """

    def __init__(
        self, pattern: SparsePattern, width: int | None, transpose_width: int | None
    ) -> None:
        numbered = pattern.numbered()
        columns, numbers, lengths = _padded_rows(numbered, width, "row")
        t_columns, t_numbers, _ = _padded_rows(sp.csr_matrix(numbered.T), transpose_width, "column")
        self._rows = EntryMap.placing(pattern, numbers)
        self._transposed = EntryMap.placing(pattern, t_numbers)
        self._shape = pattern.shape
        self._columns = jnp.asarray(columns)
        self._row_lengths = jnp.asarray(lengths.astype(np.int32))
        self._t_columns = jnp.asarray(t_columns)

    def operator(self, a: sp.csr_matrix) -> _PaddedRowOperator:
        """``a`` in padded-row form: equal, slot for slot, to ``_PaddedRowOperator.from_scipy(a, ...)``."""
        return _PaddedRowOperator(
            columns=self._columns,
            values=jnp.asarray(self._rows.apply(a).reshape(self._columns.shape)),
            row_lengths=self._row_lengths,
            shape=self._shape,
            t_columns=self._t_columns,
            t_values=jnp.asarray(self._transposed.apply(a).reshape(self._t_columns.shape)),
        )


def _padded_operator(
    a: sp.spmatrix,
    width: int | None = None,
    transpose_width: int | None = None,
    maps: EntryMaps | None = None,
    slot: Hashable = "padded",
) -> _PaddedRowOperator:
    """``a`` in padded-row form: by the layout ``maps`` holds under ``slot`` if given, else packed afresh."""
    if maps is None:
        return _PaddedRowOperator.from_scipy(a, width, transpose_width)
    a = sp.csr_matrix(a)
    return maps.held(
        (slot, width, transpose_width),
        a,
        lambda pattern: _PaddedLayout(pattern, width, transpose_width),
    ).operator(a)


def _padded_product(arrays, x: jnp.ndarray) -> jnp.ndarray:
    columns, values, _, _ = arrays
    return jnp.sum(values * x[columns], axis=1)


def _padded_product_transposed(arrays, y: jnp.ndarray) -> jnp.ndarray:
    _, _, t_columns, t_values = arrays
    return jnp.sum(t_values * y[t_columns], axis=1)
