"""Probe one column at a short reach plus the couplings beyond it that the Jacobian shows to be strong.

A column probed at a shorter reach than the assembled pattern drops its couplings beyond that reach and
folds them onto near entries (:class:`~aquaflux.precondition.sparse_jacobian.ColumnProbePlan`). Where
the cells are stretched, graph distance is not physical distance: across a thin cell, the cell three
faces away can couple as strongly as a face neighbour along the cell's long side, so a short probe of
such a column drops couplings as large as some it keeps. Probing the whole column at the full reach
costs every colour of that reach.

:class:`StrongCouplings` takes the middle way. Once, at the first fit, it probes the column at the full
reach and keeps, beyond the column's own reach, only the couplings at least ``fraction`` times as large
as the weakest face-neighbour coupling of their row. Every later fit probes the column for that pattern
alone (:meth:`~aquaflux.precondition.sparse_jacobian.ColumnProbePlan.with_column_pattern`), which takes
few more colours than the short reach. The selection reads the Jacobian itself, so it needs no model of
which terms carry the far couplings, and it is held fixed for the session, as a frozen coarsening is.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping

import numpy as np
import scipy.sparse as sp

from .sparse_jacobian import _members, _stencil_pattern

__all__ = ["StrongCouplings"]


@dataclasses.dataclass(frozen=True)
class StrongCouplings:
    """Probe one column at its own reach plus its strong couplings beyond it, chosen from the Jacobian.

    The column's own reach is its entry in the probe's ``column_reach``, and must be shorter than the
    probe's ``stencil_reach``. At the first fit the session probes the column at ``stencil_reach`` once
    and keeps every coupling within the column's reach and, beyond it, each coupling ``(i, j)`` whose
    magnitude in some row field of ``rows`` is at least ``fraction`` times the smallest nonzero magnitude
    of row ``i``'s face-neighbour couplings to the column in that row field. Every fit from then on,
    including the first, probes the column for that pattern.

    Attributes
    ----------
    column : int
        The column field, in the flat layout's order (``3`` is ``p`` in ``[u, v, w, p, k, omega]``).
    fraction : float
        How strong, relative to its row's weakest face coupling, a coupling beyond the column's reach
        must be to be kept; positive.
    rows : tuple of int or None
        The row fields whose couplings to the column are judged, in the same order. ``None`` (default)
        judges the column field's own rows -- for pressure, the continuity rows. A row field whose face
        couplings to the column can vanish (a momentum row's pressure gradient across a face parallel to
        that component) makes a poor yardstick, since its weakest face coupling is near zero.

    Raises
    ------
    ValueError
        If ``column`` or an entry of ``rows`` is negative, ``rows`` is empty, or ``fraction`` is not
        positive.
    """

    column: int
    fraction: float
    rows: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.column < 0:
            raise ValueError(f"StrongCouplings.column must be a field index, got {self.column}.")
        if not self.fraction > 0.0:
            raise ValueError(f"StrongCouplings.fraction must be positive, got {self.fraction!r}.")
        if self.rows is not None:
            rows = tuple(int(r) for r in self.rows)
            if not rows or min(rows) < 0:
                raise ValueError(
                    f"StrongCouplings.rows must name at least one field index, got {self.rows!r}."
                )
            object.__setattr__(self, "rows", rows)

    @property
    def row_fields(self) -> tuple[int, ...]:
        """The row fields judged: :attr:`rows`, or the column's own field when it is unset."""
        return (self.column,) if self.rows is None else self.rows

    def pattern(
        self,
        products: Callable[[np.ndarray], np.ndarray],
        row_starts: Mapping[int, int],
        owner: np.ndarray,
        nb: np.ndarray,
        n_cells: int,
        reach: int,
        colouring,
        batch: int = 8,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """The cell pattern the column is probed for: within ``reach``, plus the strong pairs beyond.

        Parameters
        ----------
        products : callable
            ``seeds -> responses``: the Jacobian applied to tangents nonzero in the column field alone,
            ``seeds`` of shape ``(batch, n_cells)`` (the column field's values) and responses of shape
            ``(batch, n_dofs)`` over the whole state.
        row_starts : mapping of int to int
            Where each of :attr:`row_fields` starts in a response (cell ``i`` of row field ``f`` is
            entry ``row_starts[f] + i``).
        owner, nb : np.ndarray
            Interior-face cell-graph edge endpoints, shape ``(n_edges,)`` each.
        n_cells : int
            Number of cells.
        reach : int
            The column's own reach.
        colouring : BlockColouring
            A collision-free colouring at the reach the assembled pattern covers, at which the column is
            probed for the selection.
        batch : int
            Colours probed per call of ``products``.

        Returns
        -------
        rows, cols : np.ndarray
            The pattern's cell coordinates, shape ``(n_entries,)`` each.
        kept : int
            How many pairs beyond ``reach`` were kept.

        Raises
        ------
        ValueError
            If ``reach`` is not shorter than the colouring's, or ``row_starts`` misses a row field.
        """
        if not 1 <= reach < colouring.reach:
            raise ValueError(
                f"StrongCouplings needs the column probed at a reach shorter than the pattern's "
                f"{colouring.reach}, and column {self.column} is probed at {reach}: there is nothing "
                "beyond it to select."
            )
        missing = set(self.row_fields) - set(row_starts)
        if missing:
            raise ValueError(f"no response offset for row field(s) {sorted(missing)}.")
        cell_rows = np.asarray(colouring.pattern_rows, dtype=np.int64)
        cell_cols = np.asarray(colouring.pattern_cols, dtype=np.int64)
        starts = [row_starts[field] for field in self.row_fields]
        magnitude = _column_magnitudes(products, colouring, cell_rows, cell_cols, starts, batch)
        flat = cell_rows * n_cells + cell_cols
        near = _members(flat, _flat_pairs(_stencil_pattern(owner, nb, n_cells, reach), n_cells))
        face = _members(flat, _flat_pairs(_stencil_pattern(owner, nb, n_cells, 1), n_cells)) & (
            cell_rows != cell_cols
        )
        strong = np.zeros(flat.size, dtype=bool)
        for row_field in magnitude:
            judged = np.where(face & (row_field > 0.0), row_field, np.inf)
            weakest = np.full(n_cells, np.inf)
            np.minimum.at(weakest, cell_rows, judged)
            strong |= row_field >= self.fraction * weakest[cell_rows]
        keep = near | strong
        return cell_rows[keep], cell_cols[keep], int((strong & ~near).sum())


def _column_magnitudes(products, colouring, cell_rows, cell_cols, starts, batch):
    """``|J|`` at each stored pair ``(i, j)``, per judged row field, from the column probed by colour."""
    colour = np.asarray(colouring.colour)
    pair_colour = colour[cell_cols]
    magnitude = np.zeros((len(starts), cell_rows.size))
    for first in range(0, colouring.n_colours, batch):
        colours = np.arange(first, min(first + batch, colouring.n_colours))
        seeds = (colour[None, :] == colours[:, None]).astype(np.float64)
        responses = np.asarray(products(seeds))
        hit = (pair_colour >= first) & (pair_colour < first + colours.size)
        which = pair_colour[hit] - first
        for k, start in enumerate(starts):
            magnitude[k, hit] = np.abs(responses[which, start + cell_rows[hit]])
    return magnitude


def _flat_pairs(pattern: sp.csr_matrix, n_cells: int) -> np.ndarray:
    """A 0/1 cell pattern's pairs as sorted flat indices ``i * n_cells + j``."""
    coo = pattern.tocoo()
    return np.sort(coo.row.astype(np.int64) * n_cells + coo.col)
