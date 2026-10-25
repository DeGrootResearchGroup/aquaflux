"""The SIMPLE smoother's level pieces, fitted on the device from a level operator's values.

A SIMPLE-smoothed level relaxes the saddle ``[[F, G], [D, C]]`` with an approximate velocity inverse
``M`` (a diagonal, or a per-cell block), the correction operator ``M G`` and the Schur complement
``S = C - D M G``. On a frozen coarsening every level's pattern is fixed, so the patterns of all of these
are fixed too; what changes at a refit is their values, and every value is a fixed function of the level
operator's values.

``S`` itself is never held. The smoother applies it as ``C x - D (M G x)`` from pieces it holds anyway
(``D`` and ``M G``) plus ``C``, and needs of ``S`` only its diagonal and its row norms, for the pressure
relaxation. ``S`` reaches about twice as far as the level operator, so its rows are wider than all of
its pieces' together on a fine level, and holding it (with its transpose) cost more memory than ``C``,
the one piece the smoother would not hold otherwise.

This module splits the two as :mod:`~aquaflux.precondition.level_fit` splits a hierarchy's levels. The
host decides structure once per level -- the four block cuts, the structural patterns of ``M G`` and
``S``, the padded-row layouts of ``M G``, ``D`` and ``C``, and the columns of ``S``'s rows -- and holds
it as a :class:`SimplePlan`. One compiled function, :func:`fit_simple_level`, then forms every value from
the level operator's slot values, on the device:

- the diagonal approximate inverses (Jacobi, the Frobenius-optimal diagonal ``F_ii / ||F_i||^2``, or the
  SIMPLEC row sum) and, for block splitting, each cell's Frobenius-optimal block
  ``M_i = F_ii^T (R_i R_i^T)^-1``, whose Gram matrix ``R_i R_i^T`` is read by looking each of a row's
  columns up in its sibling rows (sorted, so a binary search);
- the product ``M G``, formed onto its structural pattern a bounded chunk of its terms at a time, and
  ``S``'s diagonal and squared row norms, formed a bounded chunk of ``S``'s rows at a time and reduced
  before the next: each term is placed by a binary search of its column in the output row, so no list
  of product terms is ever held -- it would outweigh the operands several times over -- and of ``S``
  only one chunk of rows is.

Every product sums its terms in another order than a host sparse product, so its entries differ from
one in the last bits.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .entry_map import DeviceEntryMap, EntryMap, SparsePattern, ValueMap
from .kronecker_pattern import KroneckerGather, KroneckerPattern
from .level_fit import cell_blocks
from .padded_rows import _padded_rows, _PaddedLayout

#: Product terms (or Gram lookups) formed at once. Bounds the transient of a level's products -- each
#: term holds a value, its column, its place in the output row and a mask, about 20 bytes, so ~5 MB a
#: chunk -- and sets the cache footprint: on a 4-core CPU 2**18 fitted pitzDaily's SIMPLE levels in
#: 0.25 s against 0.41 s at 2**21 (2026-10-22). A device with more parallelism may want larger chunks.
TERMS_PER_CHUNK = 1 << 18


class SimplePlan(eqx.Module):
    """One SIMPLE level's structure: everything :func:`fit_simple_level` needs besides the values.

    Built by :func:`simple_plan` from the level operator's pattern; an ``equinox.Module`` whose arrays
    are leaves, so a plan rides into the compiled fit as an argument and every refit on the same
    coarsening reuses one compiled program.

    Attributes
    ----------
    unpack : ValueMap
        The level operator's stored values, in CSR order, read from its flattened padded-row slots.
    velocity, gradient, divergence, pressure : ValueMap
        The cuts of the level's values into ``F``, ``G``, ``D`` and ``C``, each onto its block's pattern
        with every row's columns ascending. Computed from the cell pattern when the level stores one over
        every field pair (:class:`~aquaflux.precondition.kronecker_pattern.KroneckerGather`), held as
        gathers otherwise.
    velocity_layout, gradient_layout : _PaddedLayout
        ``F``'s and ``G``'s rows, read but never applied.
    divergence_layout, correction_layout, pressure_layout : _PaddedLayout
        The pieces the smoother applies: ``D``, ``M G`` and ``C``, which give ``S x = C x - D (M G x)``.
    schur_rows : _PaddedLayout
        The columns of ``S``'s rows on its structural pattern, without a transpose: where the fit places
        each term of ``S`` while it reduces a chunk of rows to their diagonal and norms.
    cells : DeviceEntryMap or None
        ``F``'s cell blocks, ``(n_cells, b, b)`` flattened, with block splitting.
    block_layout : _PaddedLayout or None
        The block inverse ``M``'s layout, with block splitting.
    block_order : jnp.ndarray or None
        Each of ``M``'s stored entries as an index into the flattened cell blocks.
    """

    unpack: ValueMap
    velocity: ValueMap
    gradient: ValueMap
    divergence: ValueMap
    pressure: ValueMap
    velocity_layout: _PaddedLayout
    gradient_layout: _PaddedLayout
    divergence_layout: _PaddedLayout
    correction_layout: _PaddedLayout
    pressure_layout: _PaddedLayout
    schur_rows: _PaddedLayout
    cells: DeviceEntryMap | None
    block_layout: _PaddedLayout | None
    block_order: jnp.ndarray | None
    n_cells: int = eqx.field(static=True)
    n_velocity_fields: int = eqx.field(static=True)
    frobenius: bool = eqx.field(static=True)
    schur_frobenius: bool = eqx.field(static=True)
    simplec: bool = eqx.field(static=True)
    #: Whether every cell's velocity rows store the same columns, so their Gram matrix needs no search.
    aligned_velocity_rows: bool = eqx.field(static=True)
    #: When every row of ``D`` stores its columns as one run per velocity field and every cell's rows of
    #: ``M G`` share their columns, the widest run: the Schur rows then sum a cell's fields before
    #: placing a term. ``0`` otherwise.
    divergence_run_width: int = eqx.field(static=True)
    #: The stored entries of ``C``, ``D``, ``G`` and ``S`` (structural, never held), for the build record.
    entry_counts: tuple[int, int, int, int] = eqx.field(static=True)

    @property
    def n_velocity(self) -> int:
        """The velocity unknowns, the split point of the level vector."""
        return self.n_cells * self.n_velocity_fields

    @property
    def block_splitting(self) -> bool:
        """Whether the velocity inverse is a per-cell block rather than a diagonal."""
        return self.block_layout is not None


class SimpleValues(NamedTuple):
    """One level's SIMPLE pieces as values, double precision, as :func:`fit_simple_level` returns them.

    Each operator comes as its padded-row slot values and those of its transpose (``(values,
    t_values)``), to be held on its plan's layout: an array a compiled function returns is a new buffer,
    so returning whole operators would copy the layouts' index arrays every refit.
    """

    velocity_inverse: jnp.ndarray  # (n_velocity,) the diagonal approximate inverse of F
    block_inverse: tuple[jnp.ndarray, jnp.ndarray] | None
    correction: tuple[jnp.ndarray, jnp.ndarray]
    divergence: tuple[jnp.ndarray, jnp.ndarray]
    pressure: tuple[jnp.ndarray, jnp.ndarray]
    schur_inverse: jnp.ndarray  # (n_pressure,) the approximate inverse of S's diagonal


class SimpleChecks(NamedTuple):
    """What a SIMPLE level fit must refuse, as counts the host reads back once, and the build record."""

    velocity_bad_diagonal: jnp.ndarray  # rows of F with a zero or non-finite diagonal
    velocity_empty_rows: (
        jnp.ndarray
    )  # rows of F with no nonzero entry, where the Frobenius form divides
    schur_zero_diagonal: jnp.ndarray
    schur_bad_diagonal: jnp.ndarray  # rows of S with a non-finite diagonal
    schur_empty_rows: jnp.ndarray
    block_not_finite: (
        jnp.ndarray
    )  # entries of the block inverse that are not finite (a singular block)
    #: Min, median and max of the Schur relaxation factor ``S_ii * S~^-1_ii``.
    schur_relaxation: jnp.ndarray


def simple_plan(
    pattern: SparsePattern,
    block_size: int,
    width: int | None = None,
    *,
    frobenius: bool = True,
    schur_frobenius: bool = False,
    block_splitting: bool = False,
    simplec: bool = False,
) -> SimplePlan:
    """The plan of one SIMPLE level whose operator stores ``pattern`` in ``width`` slots per row.

    Parameters
    ----------
    pattern : SparsePattern
        The level operator's pattern, field-major with the pressure last.
    block_size : int
        Fields per cell, the pressure included.
    width : int, optional
        The level operator's padded-row width, as its layout pads it; ``None`` pads to the widest row.
    frobenius, schur_frobenius, block_splitting, simplec : bool
        The splitting (:class:`~aquaflux.precondition.saddle_multigrid.SimpleSmoothedInverse`).

    Returns
    -------
    SimplePlan
        The level's structure.

    Raises
    ------
    ValueError
        If ``block_splitting`` and ``simplec`` are both set.
    """
    if block_splitting and simplec:
        # The block inverse overrides the diagonal entirely, so a SIMPLEC diagonal built alongside it is
        # computed, reported, and never applied -- an arm whose name claims a splitting it does not run.
        raise ValueError(
            "block splitting overrides the velocity diagonal, so `simplec` would have no effect; "
            "choose one."
        )
    n_cells = pattern.shape[0] // block_size
    fields = block_size - 1
    nv = fields * n_cells

    # The field ranges of the four blocks: (velocity, velocity), (velocity, pressure), ...
    by_fields = {"velocity": (0, fields), "pressure": (fields, block_size)}
    kronecker = KroneckerPattern.of_dof(pattern, block_size, block_size)
    if kronecker is not None:
        # One cell pattern over every field pair: each cut, and reading the level's stored values out
        # of its padded rows, is a placement computed from the cell pattern.
        def cut(rows: str, columns: str) -> tuple[ValueMap, SparsePattern]:
            return KroneckerGather.of(kronecker, by_fields[rows], by_fields[columns])

        widest = block_size * int(np.diff(kronecker.cells.indptr).max(initial=0))
        unpack, _ = KroneckerGather.of(
            kronecker,
            (0, block_size),
            (0, block_size),
            slot_width=max(widest, 1) if width is None else width,
        )
    else:
        by_dofs = {"velocity": slice(None, nv), "pressure": slice(nv, None)}

        def cut(rows: str, columns: str) -> tuple[ValueMap, SparsePattern]:
            gather = EntryMap.gathering(
                pattern, lambda m: m[by_dofs[rows], by_dofs[columns]].sorted_indices()
            )
            return gather.on_device(), gather.landing

        # A gather from the padded slots, each entry reading the slot it is stored in.
        widest = int(np.diff(pattern.indptr).max(initial=0))
        unpack = DeviceEntryMap(
            slots=jnp.asarray(_slot_of_entry(pattern, width)),
            source=None,
            target=None,
            weight=None,
            n_in=pattern.shape[0] * (max(widest, 1) if width is None else width),
            n_out=pattern.nnz,
        )

    (f_map, f_pattern), (g_map, g_pattern) = (
        cut("velocity", "velocity"),
        cut("velocity", "pressure"),
    )
    (d_map, d_pattern), (c_map, c_pattern) = (
        cut("pressure", "velocity"),
        cut("pressure", "pressure"),
    )
    cells = block_layout = block_order = None
    if block_splitting:
        cells = EntryMap.gathering(f_pattern, lambda m: cell_blocks(m, fields)).on_device()
        block_pattern, order = _block_pattern(n_cells, fields)
        block_layout = _PaddedLayout.of(block_pattern, fields=(fields, fields))
        block_order = jnp.asarray(order)
        inverse_pattern = block_pattern
    else:
        inverse_pattern = SparsePattern(sp.identity(nv, format="csr"))
    # The structural patterns: every entry the operands' patterns can produce, whatever their values.
    correction_pattern = _structural(_ones(inverse_pattern) @ _ones(g_pattern))
    schur_pattern = _structural(_ones(c_pattern) + _ones(d_pattern) @ _ones(correction_pattern))
    correction_layout = _PaddedLayout.of(
        g_pattern
        if correction_pattern.matches(g_pattern.csr(np.zeros(g_pattern.nnz)))
        else correction_pattern,
        fields=(fields, 1),
    )
    return SimplePlan(
        unpack=unpack,
        velocity=f_map,
        gradient=g_map,
        divergence=d_map,
        pressure=c_map,
        velocity_layout=_PaddedLayout.of(f_pattern, transposed=False, fields=(fields, fields)),
        gradient_layout=_PaddedLayout.of(g_pattern, transposed=False, fields=(fields, 1)),
        divergence_layout=_PaddedLayout.of(d_pattern, fields=(1, fields)),
        correction_layout=correction_layout,
        pressure_layout=_PaddedLayout.of(c_pattern, fields=(1, 1)),
        schur_rows=_PaddedLayout.of(schur_pattern, transposed=False, fields=(1, 1)),
        cells=cells,
        block_layout=block_layout,
        block_order=block_order,
        n_cells=n_cells,
        n_velocity_fields=fields,
        frobenius=frobenius,
        schur_frobenius=schur_frobenius,
        simplec=simplec,
        aligned_velocity_rows=_aligned_rows(f_pattern, n_cells, fields),
        divergence_run_width=(
            _run_width(d_pattern, n_cells, fields)
            if _aligned_rows(correction_pattern, n_cells, fields)
            else 0
        ),
        entry_counts=(c_pattern.nnz, d_pattern.nnz, g_pattern.nnz, schur_pattern.nnz),
    )


@eqx.filter_jit
def fit_simple_level(
    plan: SimplePlan, slot_values: jnp.ndarray
) -> tuple[SimpleValues, SimpleChecks]:
    """One level's SIMPLE pieces from its operator's padded-row slot values, traced.

    Parameters
    ----------
    plan : SimplePlan
        The level's structure.
    slot_values : jnp.ndarray
        The level operator's padded-row values, shape ``(n_rows, width)``.

    Returns
    -------
    values : SimpleValues
        The pieces, in ``slot_values``'s precision.
    checks : SimpleChecks
        What the host must refuse, and the Schur relaxation's spread.
    """
    level = plan.unpack.apply(slot_values.ravel())
    f = plan.velocity.apply(level)
    f_rows = plan.velocity_layout.row_values(f)
    f_columns, f_lengths = plan.velocity_layout.columns, plan.velocity_layout.row_lengths
    diagonal = _row_diagonal(f_columns, f_rows)
    squared = jnp.sum(f_rows * f_rows, axis=1)
    bad_diagonal = jnp.sum(~jnp.isfinite(diagonal) | (diagonal == 0.0))
    empty_rows = jnp.zeros((), dtype=jnp.int32)
    if plan.simplec:
        # The SIMPLEC coefficient divides by the row sum, which can approach zero where a diagonal
        # cannot; rows whose sum is below a tenth of their diagonal fall back to the Frobenius value.
        sums = jnp.sum(f_rows, axis=1)
        usable = sums > 0.1 * jnp.abs(diagonal)
        velocity_inverse = jnp.where(usable, 1.0 / jnp.where(usable, sums, 1.0), diagonal / squared)
    elif plan.frobenius:
        empty_rows = jnp.sum(squared == 0.0)
        velocity_inverse = diagonal / jnp.where(squared == 0.0, 1.0, squared)
    else:
        velocity_inverse = 1.0 / diagonal

    block_not_finite = jnp.zeros((), dtype=jnp.int32)
    block_inverse = None
    if plan.block_splitting:
        fields = plan.n_velocity_fields
        blocks = plan.cells.apply(f).reshape(plan.n_cells, fields, fields)
        if plan.frobenius:
            gram = _cell_gram(
                f_columns, f_rows, f_lengths, plan.n_cells, fields, plan.aligned_velocity_rows
            )
            # M_i = F_ii^T G^-1, computed as solve(G, F_ii) transposed: (G^-1 F_ii)^T = F_ii^T G^-1
            # for the symmetric Gram matrix G. F_ii G^-1 instead differs whenever the cell's own
            # velocity block is nonsymmetric, and coincides at one field per cell.
            inverse = jnp.transpose(jnp.linalg.solve(gram, blocks), (0, 2, 1))
        else:
            inverse = jnp.linalg.inv(blocks)
        block_not_finite = jnp.sum(~jnp.isfinite(inverse))
        block_values = inverse.ravel()[plan.block_order]
        block = plan.block_layout.packed(block_values)
        block_inverse = (block.values, block.t_values)
        left = _single_rows(block.columns, block.row_lengths, block_values)
    else:
        rows = jnp.arange(plan.n_velocity, dtype=jnp.int32)
        left = (rows, rows[:, None], velocity_inverse[:, None])

    g_layout = plan.gradient_layout
    gradient = (
        g_layout.columns,
        g_layout.row_values(plan.gradient.apply(level)),
        g_layout.row_lengths,
    )
    correction = plan.correction_layout.packed(
        _product_onto(left, gradient, plan.correction_layout)
    )
    divergence = plan.divergence_layout.packed(plan.divergence.apply(level))
    if plan.divergence_run_width:
        divergence_rows = _rows_by_cell(
            divergence.columns,
            divergence.values,
            divergence.row_lengths,
            plan.n_velocity_fields,
            plan.divergence_run_width,
        )
    else:
        divergence_rows = _rows_by_entry(
            divergence.columns, divergence.values, divergence.row_lengths
        )
    pressure = plan.pressure_layout.packed(plan.pressure.apply(level))
    schur_diagonal, schur_squared = _schur_rows(
        (pressure.columns, pressure.values, pressure.row_lengths),
        divergence_rows,
        (correction.columns, correction.values, correction.row_lengths),
        plan.schur_rows,
    )
    if plan.schur_frobenius:
        schur_inverse = schur_diagonal / jnp.where(schur_squared == 0.0, 1.0, schur_squared)
        relaxation = schur_inverse * schur_diagonal
        spread = jnp.stack([jnp.min(relaxation), jnp.median(relaxation), jnp.max(relaxation)])
        schur_empty = jnp.sum(schur_squared == 0.0)
    else:
        schur_inverse = 1.0 / schur_diagonal
        spread = jnp.ones(3, dtype=schur_diagonal.dtype)
        schur_empty = jnp.zeros((), dtype=jnp.int32)
    values = SimpleValues(
        velocity_inverse=velocity_inverse,
        block_inverse=block_inverse,
        correction=(correction.values, correction.t_values),
        divergence=(divergence.values, divergence.t_values),
        pressure=(pressure.values, pressure.t_values),
        schur_inverse=schur_inverse,
    )
    checks = SimpleChecks(
        velocity_bad_diagonal=bad_diagonal,
        velocity_empty_rows=empty_rows,
        schur_zero_diagonal=jnp.sum(schur_diagonal == 0.0),
        schur_bad_diagonal=jnp.sum(~jnp.isfinite(schur_diagonal)),
        schur_empty_rows=schur_empty,
        block_not_finite=block_not_finite,
        schur_relaxation=spread,
    )
    return values, checks


def refuse_unusable(checks: SimpleChecks) -> None:
    """Raise if a fitted level's pieces cannot relax anything, reading ``checks`` on the host.

    Raises
    ------
    ValueError
        If ``F`` has a zero or non-finite diagonal or (for the Frobenius diagonal) an empty row, a
        velocity cell block is singular, or the Schur complement has a zero or non-finite diagonal or
        (for its Frobenius diagonal) an empty row.
    """
    if int(checks.velocity_bad_diagonal):
        raise ValueError("the velocity block has a zero or non-finite diagonal.")
    if int(checks.velocity_empty_rows):
        raise ValueError("the velocity block has an empty row; the optimal inverse is undefined.")
    if int(checks.block_not_finite):
        raise ValueError(
            "a velocity cell block is singular, so the block-split velocity inverse is not finite."
        )
    # A zero Schur diagonal would make the pressure relaxation undefined. It does not arise on this
    # discretization -- Rhie-Chow damping gives the continuity row a genuine diagonal, and the
    # correction only strengthens it -- so this guards against a coarse level degenerating.
    if int(checks.schur_zero_diagonal):
        raise ValueError("the Schur complement has a zero diagonal on some level.")
    if int(checks.schur_bad_diagonal):
        raise ValueError("the Schur complement has a non-finite diagonal on some level.")
    if int(checks.schur_empty_rows):
        raise ValueError("the Schur complement has an empty row; the optimal inverse is undefined.")


def _row_diagonal(columns: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    """A square operator's diagonal from its padded rows (a padding slot holds zero)."""
    rows = jnp.arange(columns.shape[0])[:, None]
    return jnp.sum(jnp.where(columns == rows, values, 0.0), axis=1)


def _sentinel_columns(columns: jnp.ndarray, lengths: jnp.ndarray, sentinel: int) -> jnp.ndarray:
    """Padded rows' columns with every padding slot set to ``sentinel``, so each row stays ascending."""
    slots = jnp.arange(columns.shape[1])[None, :]
    return jnp.where(slots < lengths[:, None], columns, sentinel)


def _in_chunks(n_rows: int, per_row: int) -> tuple[int, int]:
    """Rows per chunk, and chunks, for ``per_row`` terms a row within :data:`TERMS_PER_CHUNK`."""
    rows = max(1, min(n_rows, TERMS_PER_CHUNK // max(per_row, 1)))
    return rows, -(-n_rows // rows)


def _padded_to(array: jnp.ndarray, n_rows: int, fill) -> jnp.ndarray:
    """``array`` with rows of ``fill`` appended up to ``n_rows``."""
    extra = n_rows - array.shape[0]
    return jnp.concatenate([array, jnp.full((extra, *array.shape[1:]), fill, array.dtype)])


def _search(rows: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    """Where each of ``values`` sits in its own ascending ``rows`` row: ``(n, k)`` places in ``(n, w)``.

    The binary search unrolled: on a row of a few dozen columns it is about twice as fast as the looped
    one, and it is the bulk of a level's fit.
    """
    return jax.vmap(lambda row, value: jnp.searchsorted(row, value, method="scan_unrolled"))(
        rows, values
    )


def _entries(
    columns: jnp.ndarray, lengths: jnp.ndarray, nnz: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The stored entries of padded rows, in storage order: each one's row and column, ``(nnz,)`` each.

    A padded row holds its entries first and in storage order, so the stored slots read row-major are
    the entries in storage order, which is the order of the values a layout packs.
    """
    width = columns.shape[1]
    stored = jnp.arange(width)[None, :] < lengths[:, None]
    (slots,) = jnp.nonzero(stored.ravel(), size=nnz)
    return (slots // width).astype(jnp.int32), columns.ravel()[slots]


def _chunked(arrays: tuple[jnp.ndarray, ...], per_entry: int):
    """``arrays`` (entry-aligned) padded to whole chunks of :func:`_in_chunks`, with the chunk size,
    the chunk count and a mask of the real entries."""
    n = arrays[0].shape[0]
    chunk, n_chunks = _in_chunks(n, per_entry)
    total = chunk * n_chunks
    real = jnp.arange(total) < n
    return tuple(_padded_to(array, total, 0) for array in arrays), chunk, n_chunks, real


def _placed_terms(row_columns, columns, values, stored, right):
    """Left entries' product terms against ``R``'s rows, and where each term lands in its output row.

    ``columns`` and ``values`` are ``(e, k)``: per left entry the ``k`` rows of ``R`` it combines and
    their weights -- one row for a plain sparse product, or a cell's rows of every field when those share
    their columns, so the fields are summed before any term is placed; the first combined row gives the
    columns. ``stored`` is ``(e,)``, false for a padding entry. ``row_columns`` is ``(e, w)``: each
    entry's output row's columns, ascending and padded with a sentinel past every column. ``right`` is
    padded rows ``(columns, values, lengths)``.

    Returns ``(terms, places, kept)``, each ``(e, width(R))``: a term, its slot in its output row (by a
    binary search), and whether it is a stored one.
    """
    r_columns, r_values, r_lengths = right
    r_width = r_columns.shape[1]
    terms = jnp.einsum("ek,ekb->eb", values, r_values[columns])
    lead = columns[:, 0]
    kept = stored[:, None] & (jnp.arange(r_width)[None, :] < r_lengths[lead][:, None])
    return terms, _search(row_columns, r_columns[lead]), kept


def _product_onto(left, right, out: _PaddedLayout) -> jnp.ndarray:
    """``L R`` summed into the stored entries of ``out``'s pattern, shape ``(nnz(out),)``.

    ``left`` is ``(rows, columns, values)``: per left entry its output row and, as
    :func:`_placed_terms` takes them, the rows of ``R`` it combines with their weights. ``right`` is
    padded rows ``(columns, values, lengths)``. ``out``'s pattern must hold every entry the product can
    produce (its structural pattern).

    The left entries are taken a chunk at a time and each term is added in where it lands, so the terms
    of one chunk are all that is ever held.
    """
    rows, columns, values = left
    n_out = out.nnz
    if rows.shape[0] == 0:
        # No stored entry on the left: no terms, and no chunk to slice -- a loop over none still traces
        # its body once.
        return jnp.zeros(n_out, right[1].dtype)
    out_columns = _sentinel_columns(out.columns, out.row_lengths, out.shape[1])
    out_starts = jnp.cumsum(out.row_lengths) - out.row_lengths
    (rows, columns, values), chunk, n_chunks, real = _chunked(
        (rows, columns, values), right[0].shape[1] * columns.shape[1]
    )

    def add_chunk(index, result):
        def of_chunk(array):
            return jax.lax.dynamic_slice_in_dim(array, index * chunk, chunk)

        row = of_chunk(rows)
        terms, places, kept = _placed_terms(
            out_columns[row], of_chunk(columns), of_chunk(values), of_chunk(real), right
        )
        targets = jnp.where(kept, out_starts[row][:, None] + places, n_out)
        return result.at[targets.ravel()].add(terms.ravel(), mode="drop")

    return jax.lax.fori_loop(0, n_chunks, add_chunk, jnp.zeros(n_out, right[1].dtype))


def _schur_rows(pressure, divergence, correction, rows: _PaddedLayout):
    """The diagonal and squared row norms of ``S = C - D (M G)``, shape ``(n,)`` each, ``S`` never held.

    ``pressure`` and ``correction`` are padded rows ``(columns, values, lengths)`` of ``C`` and ``M G``;
    ``divergence`` is ``D``'s rows as :func:`_rows_by_entry` or :func:`_rows_by_cell` give them; ``rows``
    holds the columns of ``S``'s rows on its structural pattern. A chunk of rows at a time, every term
    of ``D (M G)`` is placed in its row of ``S`` (as :func:`_product_onto` places them, in the same
    order), negated, ``C``'s entries are added, and the chunk is reduced to its diagonal and norms
    before the next is formed.
    """
    d_columns, d_values, d_stored = divergence
    c_columns, c_values, c_lengths = pressure
    n, width = rows.shape[0], rows.width
    dtype = correction[1].dtype
    if n == 0:
        return jnp.zeros(0, dtype), jnp.zeros(0, dtype)
    entries, fields = d_columns.shape[1], d_columns.shape[2]
    per_row = entries * correction[0].shape[1] * fields + c_columns.shape[1] + width
    chunk, n_chunks = _in_chunks(n, per_row)
    total = chunk * n_chunks
    s_columns = _padded_to(_sentinel_columns(rows.columns, rows.row_lengths, n), total, n)
    d_columns, d_values, c_columns, c_values = (
        _padded_to(array, total, 0) for array in (d_columns, d_values, c_columns, c_values)
    )
    d_stored = _padded_to(d_stored, total, False)
    c_stored = _padded_to(
        jnp.arange(c_columns.shape[1])[None, :] < c_lengths[:, None], total, False
    )
    local = jnp.arange(chunk, dtype=jnp.int32)

    def reduced(index):
        def of_chunk(array):
            return jax.lax.dynamic_slice_in_dim(array, index * chunk, chunk)

        own = of_chunk(s_columns)  # (chunk, width)
        row = jnp.repeat(local, entries)
        terms, places, kept = _placed_terms(
            own[row],
            of_chunk(d_columns).reshape(-1, fields),
            of_chunk(d_values).reshape(-1, fields),
            of_chunk(d_stored).ravel(),
            correction,
        )
        out = chunk * width
        targets = jnp.where(kept, row[:, None] * width + places, out)
        values = jnp.zeros(out, dtype).at[targets.ravel()].add(terms.ravel(), mode="drop")
        c_targets = jnp.where(
            of_chunk(c_stored), local[:, None] * width + _search(own, of_chunk(c_columns)), out
        )
        values = (-values).at[c_targets.ravel()].add(of_chunk(c_values).ravel(), mode="drop")
        values = values.reshape(chunk, width)
        on_diagonal = own == (index * chunk + local)[:, None]
        return (
            jnp.sum(jnp.where(on_diagonal, values, 0.0), axis=1),
            jnp.sum(values * values, axis=1),
        )

    diagonal, squared = jax.lax.map(reduced, jnp.arange(n_chunks))
    return diagonal.ravel()[:n], squared.ravel()[:n]


def _single_rows(columns: jnp.ndarray, lengths: jnp.ndarray, values: jnp.ndarray):
    """A padded operator's stored entries as :func:`_product_onto`'s left side, one row each."""
    rows, entry_columns = _entries(columns, lengths, values.size)
    return rows, entry_columns[:, None], values[:, None]


def _rows_by_entry(columns: jnp.ndarray, values: jnp.ndarray, lengths: jnp.ndarray):
    """A padded operator's rows as :func:`_schur_rows`'s left side: per slot one combined row.

    Returns ``(columns, values, stored)``, shapes ``(n, width, 1)``, ``(n, width, 1)`` and ``(n, width)``.
    """
    stored = jnp.arange(columns.shape[1])[None, :] < lengths[:, None]
    return columns[..., None], values[..., None], stored


def _rows_by_cell(columns, values, lengths, fields: int, width: int):
    """A padded operator storing each row's columns as ``fields`` equal runs, one per field (cell
    ``c``'s field-``f`` column is ``c + f n``), as :func:`_schur_rows`'s left side: per slot of a run one
    cell, carrying that cell's ``fields`` columns and values.

    ``width`` is the widest run. Returns ``(columns, values, stored)``, shapes ``(n, width, fields)``
    twice and ``(n, width)``.
    """
    run = lengths // fields
    slots = jnp.arange(width)
    within = jnp.minimum(
        jnp.arange(fields)[:, None, None] * run[None, :, None] + slots[None, None, :],
        columns.shape[1] - 1,
    )  # (fields, n_rows, width): field f's slot a of each row

    def by_field(array):
        picked = jnp.take_along_axis(
            jnp.broadcast_to(array, (fields, *array.shape)), within, axis=2
        )
        return jnp.transpose(picked, (1, 2, 0))

    return by_field(columns), by_field(values), slots[None, :] < run[:, None]


def _cell_gram(columns, values, lengths, n_cells: int, fields: int, aligned: bool) -> jnp.ndarray:
    """Each cell's Gram matrix of its velocity rows, ``(R_i R_i^T)_fg``, shape ``(n_cells, b, b)``.

    Row ``f * n_cells + i`` of ``F`` is cell ``i``'s field-``f`` row, given as padded rows. Where every
    cell's rows store the same columns (``aligned``, which a probed Jacobian and its nodal Galerkin
    levels do), the dot product of two rows is that of their slot values. Otherwise each row's columns
    are looked up in the cell's other rows by a binary search (the padded rows are ascending), a chunk
    of rows at a time.
    """
    n_rows, width = columns.shape
    if aligned or n_rows == 0:
        by_field = values.reshape(fields, n_cells, width)
        return jnp.einsum("fca,gca->cfg", by_field, by_field)
    searchable = _sentinel_columns(columns, lengths, n_rows)
    stored = jnp.arange(width)[None, :] < lengths[:, None]
    (own_columns, own_values, own_stored, row_index), chunk, n_chunks, _ = _chunked(
        (columns, values, stored, jnp.arange(n_rows, dtype=jnp.int32)), fields * width
    )

    def dots(index):
        start = index * chunk

        def of_chunk(array):
            return jax.lax.dynamic_slice_in_dim(array, start, chunk)

        rows, own = of_chunk(row_index), of_chunk(own_columns)
        # The cell's rows, one per field: (chunk, fields).
        siblings = (rows % n_cells)[:, None] + n_cells * jnp.arange(fields)[None, :]
        partner_columns = searchable[siblings].reshape(-1, width)  # (chunk * fields, width)
        places = jnp.minimum(_search(partner_columns, jnp.repeat(own, fields, axis=0)), width - 1)
        places = places.reshape(chunk, fields, width)
        found = jnp.take_along_axis(searchable[siblings], places, axis=2) == own[:, None, :]
        partner = jnp.take_along_axis(values[siblings], places, axis=2)
        kept = found & of_chunk(own_stored)[:, None, :]
        return jnp.sum(jnp.where(kept, partner * of_chunk(own_values)[:, None, :], 0.0), axis=2)

    by_row = jax.lax.map(dots, jnp.arange(n_chunks)).reshape(-1, fields)[:n_rows]
    # Row f * n_cells + i holds cell i's (f, g) entries: (fields, n_cells, fields) -> (n_cells, f, g).
    return jnp.transpose(by_row.reshape(fields, n_cells, fields), (1, 0, 2))


def _ones(pattern: SparsePattern) -> sp.csr_matrix:
    """``pattern`` holding ones, so a product over it stores every entry it can produce."""
    return pattern.csr(np.ones(pattern.nnz))


def _structural(matrix: sp.spmatrix) -> SparsePattern:
    """The pattern of a product over ones, each row's columns ascending."""
    return SparsePattern.of(sp.csr_matrix(matrix).sorted_indices())


def _block_pattern(n_cells: int, fields: int) -> tuple[SparsePattern, np.ndarray]:
    """The block-diagonal pattern of a per-cell block, and each stored entry's place in the blocks.

    Cell ``i``'s ``(f, g)`` entry is row ``f * n_cells + i``, column ``g * n_cells + i``; flattened,
    the blocks hold it at ``(i * fields + f) * fields + g``.
    """
    cell, f, g = np.meshgrid(
        np.arange(n_cells), np.arange(fields), np.arange(fields), indexing="ij"
    )
    index = ((cell * fields + f) * fields + g).ravel()
    matrix = sp.csr_matrix(
        (
            (index + 1).astype(np.float64),
            ((f * n_cells + cell).ravel(), (g * n_cells + cell).ravel()),
        ),
        shape=(fields * n_cells,) * 2,
    ).sorted_indices()
    order = (matrix.data - 1).astype(np.int32)
    return SparsePattern.of(matrix), order


def _slot_of_entry(pattern: SparsePattern, width: int | None) -> np.ndarray:
    """Each stored entry's slot in ``pattern``'s padded rows of ``width``, shape ``(nnz,)``."""
    _, numbers, _ = _padded_rows(pattern.numbered(), width, "row")
    numbers = numbers.ravel()
    slots = np.flatnonzero(numbers)
    slot_of_entry = np.empty(pattern.nnz, dtype=np.int32)
    slot_of_entry[numbers[slots].astype(np.int64) - 1] = slots
    return slot_of_entry


def _aligned_rows(pattern: SparsePattern, n_cells: int, fields: int) -> bool:
    """Whether every cell's ``fields`` rows of a field-major ``pattern`` store the same columns."""
    lengths = np.diff(pattern.indptr).reshape(fields, n_cells)
    if not np.all(lengths == lengths[0]):
        return False
    by_field = [
        pattern.indices[pattern.indptr[f * n_cells] : pattern.indptr[(f + 1) * n_cells]]
        for f in range(fields)
    ]
    return all(np.array_equal(columns, by_field[0]) for columns in by_field[1:])


def _run_width(pattern: SparsePattern, n_cells: int, fields: int) -> int:
    """The widest run if every row of ``pattern`` stores its columns as ``fields`` runs, the field-``f``
    run the field-0 run plus ``f n_cells``; ``0`` if any row does not."""
    lengths = np.diff(pattern.indptr)
    if np.any(lengths % fields):
        return 0
    run = lengths // fields
    rows = np.repeat(np.arange(pattern.shape[0]), lengths)
    place = np.arange(pattern.nnz) - pattern.indptr[rows]
    field, slot = place // run[rows], place % run[rows]
    first = pattern.indices[pattern.indptr[rows] + slot]
    if not np.array_equal(pattern.indices, first + field * n_cells):
        return 0
    return max(int(run.max(initial=0)), 1)
