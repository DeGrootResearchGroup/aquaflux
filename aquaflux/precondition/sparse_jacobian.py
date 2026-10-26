"""Materialize the sparse Jacobian of a block residual, by compressed (graph-coloured) probing.

Some preconditioners need the assembled coupled Jacobian as a sparse matrix rather than as a
matrix-vector product — an incomplete-factorization (ILU) preconditioner factors the matrix, which a
matrix-free operator cannot supply. This module recovers that matrix from the *same* residual the
solver uses, so there is no second, hand-derived assembly to drift from it: it probes the residual's
directional derivative (one ``jax.jvp`` per colour and column field) and de-compresses the responses
into the sparse matrix, exploiting the mesh's finite stencil so a handful of probes recover the whole
Jacobian instead of one probe per degree of freedom.

The state is laid out **field-major** — degree of freedom ``(cell i, field f)`` lives at flat index
``f * n_cells + i`` — matching the coupled-state layout the solver packs. The Jacobian's sparsity is
the mesh cell graph raised to the stencil's reach (distance one per differencing/interpolation step),
blocked up to ``n_fields`` per cell.

Two stages, split so the graph work is testable without JAX:

* :func:`block_stencil_colouring` (pure NumPy/SciPy) — from the interior-face cell graph, the
  cell-block sparsity pattern at a given reach and a **compatible colouring**: two cell-columns share
  a colour only if no cell-row couples to both, so one probe seeded on a whole colour reads that
  colour's blocks with no collisions.
* :func:`materialize_block_jacobian` — run one ``jvp`` per (colour, field), scatter the responses into
  a ``scipy.sparse`` matrix.

:func:`jacobian_relative_error` checks a materialized matrix against the operator on a random vector —
the cheap guard that the chosen ``reach`` actually covers the stencil (too small a reach silently
drops the far couplings).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .frozen_operator import require_valid_graph
from .kronecker_pattern import KroneckerPattern
from .probe_layout import (
    BlockLayout,
    ProbedBlocks,
    ProbeGather,
    _CellEntries,
    _DofEntries,
    _index_dtype,
)


class BlockColouring:
    """A cell-block sparsity pattern and a probing colouring for a field-major block Jacobian.

    Attributes
    ----------
    n_cells : int
        Number of cells.
    reach : int
        The stencil reach (graph distance) the pattern covers.
    colour : np.ndarray
        Per-cell colour index, shape ``(n_cells,)``; ``n_colours`` distinct values.
    n_colours : int
        Number of colours.
    pattern_rows, pattern_cols : np.ndarray
        The nonzero cell-block coordinates ``(i, j)`` of the Jacobian (row cell ``i`` couples to
        column cell ``j``), each shape ``(n_blocks,)``.
    """

    def __init__(
        self,
        n_cells: int,
        reach: int,
        colour: np.ndarray,
        n_colours: int,
        pattern_rows: np.ndarray,
        pattern_cols: np.ndarray,
    ) -> None:
        self.n_cells = n_cells
        self.reach = reach
        self.colour = colour
        self.n_colours = n_colours
        self.pattern_rows = pattern_rows
        self.pattern_cols = pattern_cols


class ColumnProbePlan:
    """Which probe reads each column field, over one shared assembly pattern.

    A probe costs one directional derivative per (colour, **column** field), and the colour count falls
    steeply as the stencil reach drops, so the reach is worth choosing per column field rather than once
    for the whole block. A column whose couplings all lie inside a shorter reach can be probed with that
    reach's colouring and still assembled into the wide pattern: its outer positions are the explicit
    zeros they already were.

    That reduction is **exact**, not an approximation, and the reason is worth stating because the
    opposite case is easy to reach by accident. A colouring is collision-free only for the pattern it was
    built from, so two cells sharing a short-reach colour may still both couple to a common row further
    out. The probe response for that row is then the sum of both couplings, and the de-compression
    charges the whole sum to the one column inside the short pattern — the far coupling is *folded onto*
    a near entry rather than dropped. Where the column genuinely has nothing that far out, there is
    nothing to fold, and the recovered matrix is the same one a full-reach probe returns.

    So a shorter reach recovers the matrix exactly only for a column measured to carry no mass beyond
    it. Choosing one where the far couplings are merely small perturbs the near entries instead, by the
    folded mass -- a trade some preconditioners take with no change to the solve and others do not
    (an incomplete factorization of the perturbed pattern can fail outright), so it is measured per
    case, never assumed.

    Attributes
    ----------
    n_cells : int
        Number of cells.
    n_fields : int
        Degrees of freedom per cell.
    pattern_rows, pattern_cols : np.ndarray
        The assembly pattern's cell-block coordinates ``(i, j)``, from the widest reach in the plan.
    colour : np.ndarray
        Per column field and cell, the probe colour; shape ``(n_fields, n_cells)``.
    n_colours : tuple of int
        Colours used by each column field.
    probe_base : tuple of int
        The flat probe index each column field's colours start at, so probe ``(field b, colour c)`` is
        ``probe_base[b] + c``.
    n_probes : int
        Total directional derivatives one materialize costs.
    reach : tuple of int
        The stencil reach each column field is probed at.
    pattern_reach : int
        The stencil reach the assembly pattern covers, at least the widest of ``reach``.
    edges : tuple of np.ndarray or None
        The interior-face cell graph ``(owner, neighbour)`` the plan was coloured on, shape
        ``(n_edges,)`` each, when its builder had it (:func:`column_probe_plan` does), else ``None``.
    in_reach : np.ndarray
        Per column field and pattern entry, whether that entry lies within the column's own reach;
        shape ``(n_fields, n_blocks)``. Entries outside it are **forced to zero** rather than read from
        a response — see the note below.
    active_rows : np.ndarray
        Per row field and column field, whether that field-pair block is wanted at all;
        shape ``(n_fields, n_fields)``, indexed ``[row_field, column_field]``. Unlike ``in_reach`` (a
        graph-distance question, decided per pattern entry) this is a **static** declaration from the
        consumer -- some structural sub-block of the assembled matrix that consumer will never read,
        whatever the mesh or the state. An excluded pair is not merely zeroed: :func:`block_stencil_gather_map`
        removes it from the stored pattern entirely, so it costs neither the materialize's de-compression
        nor the assembled matrix's memory. ``True`` everywhere by default (every block wanted, which is
        the only sound default for a plan a consumer's identity is unknown to).

    Notes
    -----
    A pattern entry lying outside its column's own reach must not be read from that column's probe. The
    colouring guarantees at most one same-colour cell within the *probed* reach of a given row, but says
    nothing beyond it, so a response can legitimately hold another cell's near coupling — and gathering
    it into the far position would write a real value where the matrix has none. Those positions are set
    to zero explicitly, which is what they already were in a full-reach build.
    """

    def __init__(
        self,
        n_cells: int,
        n_fields: int,
        pattern_rows: np.ndarray,
        pattern_cols: np.ndarray,
        colour: np.ndarray,
        n_colours: tuple[int, ...],
        reach: tuple[int, ...],
        in_reach: np.ndarray | None = None,
        active_rows: np.ndarray | None = None,
        pattern_reach: int | None = None,
        edges: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.n_cells = n_cells
        self.n_fields = n_fields
        self.pattern_rows = pattern_rows
        self.pattern_cols = pattern_cols
        self.colour = colour
        self.n_colours = n_colours
        self.reach = reach
        self.pattern_reach = max(reach) if pattern_reach is None else int(pattern_reach)
        self.edges = edges
        self.in_reach = (
            np.ones((n_fields, pattern_rows.shape[0]), dtype=bool) if in_reach is None else in_reach
        )
        self.active_rows = (
            np.ones((n_fields, n_fields), dtype=bool) if active_rows is None else active_rows
        )
        base, total = [], 0
        for count in n_colours:
            base.append(total)
            total += count
        self.probe_base = tuple(base)
        self.n_probes = total
        # The (field, colour) each flat probe index stands for, so a seed can be built from its index
        # alone rather than by walking the nesting.
        self._probe_field = np.repeat(np.arange(n_fields), np.asarray(n_colours))
        self._probe_colour = np.concatenate(
            [np.arange(count) for count in n_colours] if total else [np.zeros(0, dtype=np.int64)]
        )
        # The three arrays the seeds are formed from, on the device; moved there on first use.
        self._device: tuple[jnp.ndarray, ...] | None = None

    @classmethod
    def uniform(
        cls, colouring: BlockColouring, n_fields: int, active_rows: np.ndarray | None = None
    ) -> ColumnProbePlan:
        """Probe every column field at one reach — the plan a single :class:`BlockColouring` describes."""
        return cls(
            colouring.n_cells,
            n_fields,
            colouring.pattern_rows,
            colouring.pattern_cols,
            np.broadcast_to(colouring.colour, (n_fields, colouring.n_cells)),
            (colouring.n_colours,) * n_fields,
            (colouring.reach,) * n_fields,
            active_rows=active_rows,
            pattern_reach=colouring.reach,
        )

    def with_column_pattern(
        self, field: int, rows: np.ndarray, cols: np.ndarray
    ) -> ColumnProbePlan:
        """This plan with column field ``field`` probed for the cell pattern ``(rows, cols)`` alone.

        The column is given a colouring of its own, collision-free for that pattern
        (:func:`_pattern_colouring`), and reads only the stored entries the pattern holds; the rest of
        its stored entries are written as the explicit zeros a short-probed column's out-of-reach
        entries are. The stored pattern, every other column and the active blocks are unchanged, so a
        consumer of the materialized matrix sees the same sparsity. It is how a column is probed at a
        short reach plus a chosen few couplings beyond it, which no single reach describes.

        The column is recovered exactly only where its couplings lie inside the pattern; a coupling
        outside it is folded onto a pattern entry sharing its colour, as for any short probe.

        Parameters
        ----------
        field : int
            The column field, counted over the per-cell fields.
        rows, cols : np.ndarray
            The pattern's cell coordinates ``(i, j)`` (row cell ``i`` couples to column cell ``j``),
            shape ``(n_entries,)`` each; every pair must be in the stored pattern. Repeats are allowed.

        Returns
        -------
        ColumnProbePlan
            The plan.

        Raises
        ------
        ValueError
            If ``field`` is out of range or a pair lies outside the stored pattern.
        """
        if not 0 <= field < self.n_fields:
            raise ValueError(f"column field {field} is not one of this plan's {self.n_fields}.")
        n = self.n_cells
        stored = self.pattern_rows.astype(np.int64) * n + self.pattern_cols
        wanted = np.unique(np.asarray(rows, dtype=np.int64) * n + np.asarray(cols, dtype=np.int64))
        in_stored = _members(wanted, np.sort(stored))
        if not in_stored.all():
            first = int(wanted[~in_stored][0])
            raise ValueError(
                f"the pattern couples cell {first // n} to cell {first % n}, which the stored pattern "
                "does not hold; a column cannot be probed beyond the pattern it is assembled into."
            )
        pattern = sp.csr_matrix((np.ones(wanted.size), (wanted // n, wanted % n)), shape=(n, n))
        colour, count = _pattern_colouring(pattern, n)
        colours = np.array(self.colour, copy=True)
        colours[field] = colour
        n_colours = list(self.n_colours)
        n_colours[field] = count
        in_reach = np.array(self.in_reach, copy=True)
        in_reach[field] = _members(stored, wanted)
        return ColumnProbePlan(
            n,
            self.n_fields,
            self.pattern_rows,
            self.pattern_cols,
            colours,
            tuple(n_colours),
            self.reach,
            in_reach,
            active_rows=self.active_rows,
            pattern_reach=self.pattern_reach,
            edges=self.edges,
        )

    def probe(self, index: int) -> tuple[int, int]:
        """The ``(column field, colour)`` that flat probe ``index`` stands for."""
        return int(self._probe_field[index]), int(self._probe_colour[index])

    def seed_block(self, start: int, stop: int, rows: int | None = None) -> jnp.ndarray:
        """Seed vectors for probes ``[start, stop)``, shape ``(rows, n_fields * n_cells)``, on the device.

        Probe ``probe_base[b] + c`` carries a one in field ``b`` of every cell of colour ``c`` — the
        directional derivative that reads column ``b`` of every block in that colour's share of the
        pattern.

        **Built a chunk at a time, on the device, because the full set is large and needed only one
        chunk at a time.** All the seeds together are ``n_probes * n_fields * n_cells`` floats —
        hundreds of megabytes on a three-dimensional coupled case, held for the whole materialize
        against a few megabytes for one chunk — and each is an indicator of the plan's colours, which
        one compiled program forms where the product reads it, so no seed crosses from the host.

        Parameters
        ----------
        start, stop : int
            Half-open range of flat probe indices.
        rows : int, optional
            The number of rows to return, at least ``stop - start``; the surplus rows are zero, which is
            exactly the padding a batched map needs to keep one compiled shape on a final short chunk.
            ``None`` returns ``stop - start`` rows.

        Returns
        -------
        jnp.ndarray
            The seeds, float64.
        """
        return self._seeds(start, stop, rows, whole_state=True, dtype=np.dtype(np.float64))

    def field_seeds(
        self, start: int, stop: int, rows: int | None = None, dtype: np.dtype | str = "float64"
    ) -> jnp.ndarray:
        """One column field's seeds for probes ``[start, stop)``, shape ``(rows, n_cells)``, on the device.

        The same indicators as :meth:`seed_block`, restricted to the one field every probe in the range
        reads: what a product differentiating the residual in that field alone takes. ``rows`` pads as
        :meth:`seed_block`'s does.

        Parameters
        ----------
        start, stop : int
            Half-open range of flat probe indices, all of one column field.
        rows : int, optional
            The number of rows to return, at least ``stop - start``. ``None`` returns ``stop - start``.
        dtype : np.dtype or str
            The seeds' floating type.

        Returns
        -------
        jnp.ndarray
            The seeds.

        Raises
        ------
        ValueError
            If the range spans more than one column field.
        """
        fields = np.unique(self._probe_field[start:stop])
        if fields.size > 1:
            raise ValueError(
                f"probes [{start}, {stop}) read column fields {fields.tolist()}; one field's seeds "
                "cannot carry another's."
            )
        return self._seeds(start, stop, rows, whole_state=False, dtype=np.dtype(dtype))

    def _seeds(
        self, start: int, stop: int, rows: int | None, *, whole_state: bool, dtype: np.dtype
    ) -> jnp.ndarray:
        rows = stop - start if rows is None else rows
        if rows < stop - start:
            raise ValueError(f"{rows} rows cannot hold the seeds of probes [{start}, {stop}).")
        if self._device is None:
            # The colours and each probe's (field, colour), moved once: every seed is formed from them.
            narrow = _index_dtype(max(self.n_fields, max(self.n_colours, default=0)))
            self._device = tuple(
                jnp.asarray(np.asarray(array), dtype=narrow)
                for array in (self.colour, self._probe_field, self._probe_colour)
            )
        return _indicator_seeds(
            *self._device, start, stop - start, rows=rows, whole_state=whole_state, dtype=dtype
        )

    def field_chunks(self, chunk: int):
        """The probes in ranges of at most ``chunk``, never spanning two column fields.

        Probes are numbered field by field (:attr:`probe_base`), so each field's run is cut into chunks
        of its own and a field's last chunk may be short. Yields ``(field, start, count)``.
        """
        for field, (base, count) in enumerate(zip(self.probe_base, self.n_colours, strict=True)):
            for start in range(base, base + count, chunk):
                yield field, start, min(chunk, base + count - start)


def column_probe_plan(
    owner: np.ndarray,
    nb: np.ndarray,
    n: int,
    column_reach: Sequence[int],
    pattern_reach: int | None = None,
    active_rows: np.ndarray | None = None,
) -> ColumnProbePlan:
    """A probing plan giving each column field its own stencil reach.

    The **assembly pattern** is separate from what the columns are probed at, and keeping it at the full
    ``pattern_reach`` is what makes the saving free downstream: every consumer sees the same sparsity as
    a uniform-reach build, so a coarsening hierarchy and any factorization over the pattern are
    unchanged. A short-probed column simply leaves the outer positions at the explicit zeros they held
    anyway.

    One colouring is built per *distinct* reach rather than per field, since fields sharing a reach share
    a colouring.

    **A column with mass beyond its reach is perturbed, not truncated** — see :class:`ColumnProbePlan`.
    Shorten one only where that has been measured to leave the solve unchanged.
    Which columns qualify depends on the discretization: a first-order-upwind scalar with a
    non-orthogonal diffusion correction reaches two, while a second-order upwind reconstruction on the
    same field reaches three. It is therefore a property of the assembled case, not a library constant.

    Parameters
    ----------
    owner, nb : np.ndarray
        Interior-face cell-graph edge endpoints, shape ``(n_edges,)`` each.
    n : int
        Number of cells.
    column_reach : sequence of int
        The stencil reach for each column field, in field-major order.
    pattern_reach : int, optional
        The reach the assembly pattern covers. ``None`` (default) uses the widest column reach. Must be
        at least that, since a column cannot be probed beyond the pattern it is assembled into.
    active_rows : np.ndarray, optional
        Forwarded to :class:`ColumnProbePlan` — see its own docstring. ``None`` (default) wants every
        row/column field pair, byte-identical to a plan built without it.

    Returns
    -------
    ColumnProbePlan
        The pattern and the per-column colourings.

    Raises
    ------
    ValueError
        If ``column_reach`` is empty, or ``pattern_reach`` is narrower than the widest column reach.
    """
    reaches = tuple(int(r) for r in column_reach)
    if not reaches:
        raise ValueError("column_probe_plan: need a reach for at least one column field.")
    widest = max(reaches)
    pattern_reach = widest if pattern_reach is None else int(pattern_reach)
    if pattern_reach < widest:
        raise ValueError(
            f"column_probe_plan: pattern_reach={pattern_reach} is narrower than the widest column "
            f"reach {widest}; a column cannot be probed beyond the pattern it is assembled into."
        )
    by_reach = {
        r: block_stencil_colouring(owner, nb, n, r) for r in sorted({*reaches, pattern_reach})
    }
    pattern = by_reach[pattern_reach]
    # Which pattern entries each column may actually read from its own probe: membership of the
    # column's own pattern, as a lookup on the flattened (i, j) index. The coordinate lists are NOT
    # guaranteed sorted, so the array being searched is sorted explicitly.
    flat_pattern = pattern.pattern_rows.astype(np.int64) * n + pattern.pattern_cols
    in_reach = np.empty((len(reaches), flat_pattern.size), dtype=bool)
    for field, r in enumerate(reaches):
        inner = by_reach[r]
        in_reach[field] = _members(
            flat_pattern, np.sort(inner.pattern_rows.astype(np.int64) * n + inner.pattern_cols)
        )
    return ColumnProbePlan(
        n,
        len(reaches),
        pattern.pattern_rows,
        pattern.pattern_cols,
        np.stack([by_reach[r].colour for r in reaches]),
        tuple(by_reach[r].n_colours for r in reaches),
        reaches,
        in_reach,
        active_rows=active_rows,
        pattern_reach=pattern_reach,
        edges=(np.asarray(owner), np.asarray(nb)),
    )


def _members(values: np.ndarray, sorted_set: np.ndarray) -> np.ndarray:
    """Whether each of ``values`` is in ``sorted_set`` (ascending), as a boolean array like ``values``."""
    if sorted_set.size == 0:
        return np.zeros(values.shape, dtype=bool)
    where = np.searchsorted(sorted_set, values)
    return sorted_set[np.minimum(where, sorted_set.size - 1)] == values


def block_stencil_colouring(
    owner: np.ndarray, nb: np.ndarray, n: int, reach: int
) -> BlockColouring:
    """The cell-block sparsity pattern at ``reach`` and a compatible probing colouring.

    The Jacobian couples cell ``i`` to cell ``j`` only when they are within ``reach`` steps on the
    interior-face cell graph (each differencing/interpolation step reaches one ring of neighbours).
    Two column-cells can share a probe colour only if no row-cell couples to both — i.e. they are not
    both within ``reach`` of a common cell — so the conflict graph is the pattern squared, and a
    colouring of it (by saturation degree, :func:`_saturation_colouring`) gives collision-free probes.
    The colour count *is* what a materialize costs — one directional derivative per colour and field —
    so the colouring is chosen to keep it low.

    Parameters
    ----------
    owner, nb : np.ndarray
        Interior-face cell-graph edge endpoints, shape ``(n_edges,)`` each.
    n : int
        Number of cells.
    reach : int
        Stencil reach (``1`` for a compact first-neighbour operator; a corrected-gradient / Rhie--Chow
        coupled RANS Jacobian reaches ``2``--``3``).

    Returns
    -------
    BlockColouring
        The pattern and colouring.

    Raises
    ------
    ValueError
        If ``n < 1`` or ``reach < 1``, or the edge endpoints are out of range.
    """
    # The cell-count and edge-endpoint checks are the same three every consumer of an interior-face
    # graph needs, and they have a home that takes the caller's name for its messages. Only the reach
    # is this function's own.
    require_valid_graph(n, owner, nb, "block_stencil_colouring")
    if reach < 1:
        raise ValueError(f"block_stencil_colouring: reach must be >= 1, got {reach}.")
    pattern = _stencil_pattern(owner, nb, n, reach)
    colour, n_colours = _pattern_colouring(pattern, n)

    coo = pattern.tocoo()
    return BlockColouring(n, reach, colour, n_colours, coo.row, coo.col)


def _stencil_pattern(owner: np.ndarray, nb: np.ndarray, n: int, reach: int) -> sp.csr_matrix:
    """Every cell pair within graph distance ``reach`` on the interior-face graph, as a 0/1 CSR matrix."""
    owner, nb = np.asarray(owner), np.asarray(nb)
    ones = np.ones(2 * owner.size + n)
    adjacency = sp.coo_matrix(
        (
            ones,
            (
                np.concatenate([owner, nb, np.arange(n)]),
                np.concatenate([nb, owner, np.arange(n)]),
            ),
        ),
        shape=(n, n),
    ).tocsr()
    adjacency.data[:] = 1.0
    pattern = adjacency
    for _ in range(reach - 1):
        pattern = pattern @ adjacency
        pattern.data[:] = 1.0
    return pattern


def _pattern_colouring(pattern: sp.csr_matrix, n: int) -> tuple[np.ndarray, int]:
    """A collision-free probe colouring of a 0/1 cell pattern (row cells couple to column cells).

    Two column cells may share a colour only if no row cell couples to both, so the conflict graph is
    ``pattern^T pattern``, coloured by saturation degree (:func:`_saturation_colouring`). Returns the
    per-cell colour, shape ``(n,)``, and the number of colours.
    """
    conflict = (pattern.T @ pattern).tocsr()
    conflict.data[:] = 1.0
    colour = _saturation_colouring(conflict, n)
    return colour, int(colour.max()) + 1 if n else 0


_WORD = 64
_FULL_WORD = (1 << _WORD) - 1


def _saturation_colouring(conflict: sp.csr_matrix, n: int) -> np.ndarray:
    """Colour a conflict graph by saturation degree, ties broken by degree.

    Every probe costs one directional derivative per field, so the colour count *is* the cost of a
    materialize and a colouring worth its build time is one that uses few colours. This picks, at each
    step, the uncoloured vertex whose neighbours already show the most **distinct** colours — the
    vertex most constrained, and so the one most likely to need a new colour if left until later —
    and gives it the smallest colour none of its neighbours holds. Ordering by plain degree instead
    (colouring the highest-degree vertex first) ignores how constrained a vertex has actually become
    and uses measurably more colours: on a three-dimensional hexahedral cell graph at stencil reach
    three it needs 112 colours where this needs 94, i.e. 108 more directional derivatives per
    materialize for a six-field block.

    Each vertex carries a bitset of the colours its neighbours hold, so colouring a vertex updates its
    whole neighbourhood with one vectorized or-assign rather than one set operation per incident edge
    — the difference between seconds and minutes on a graph whose vertices have hundreds of
    neighbours.

    Picking the next vertex is a **bucket queue** over `rank`, not a full scan for the maximum: a
    scan costs ``O(n)`` on every one of the ``n`` rounds, quadratic overall, even though the rest of
    the round (updating the coloured vertex's neighbours) is already degree-bounded and vectorized —
    it is the scan alone that made the whole colouring quadratic. `rank` is bounded by
    ``degree.max() * (degree.max() + 2)``, a property of the mesh's *local* connectivity and not of
    ``n``, so a bucket per possible rank value, each a doubly linked list of the vertices currently at
    that rank, turns "extract the uncoloured vertex of maximum rank" into an ``O(1)``-amortized pop: a
    pointer at the highest occupied bucket only ever moves down, except when a vertex's rank grows
    past it, and both the total downward drift and the total number of such jumps are bounded by the
    (``n``-independent) bucket range plus the total number of rank increases — so extracting all ``n``
    vertices costs ``O(n + m)`` for ``m`` conflict-graph edges, matching the already-vectorized half of
    the round instead of dominating it. Moving a vertex to a new bucket when its rank grows is an
    ``O(1)`` unlink-and-relink using the same linked-list pointers. Ties within a bucket are broken by
    which vertex became eligible for it most recently, not by vertex index — an arbitrary but
    deterministic choice among genuine ties, which does not change the colouring's validity or its
    quality relative to plain degree ordering (see the module's tests).

    Any collision-free colouring de-compresses the probe responses to the same matrix, so this choice
    changes what a materialize *costs* and not what it returns.

    Parameters
    ----------
    conflict : scipy.sparse.csr_matrix
        The conflict graph, shape ``(n, n)``: an edge wherever two cells may not share a colour.
    n : int
        Number of vertices.

    Returns
    -------
    np.ndarray
        Per-vertex colour index, shape ``(n,)``.
    """
    indptr, indices = conflict.indptr, conflict.indices
    degree = np.diff(indptr).astype(np.int64)
    colour = np.full(n, -1, dtype=np.int64)
    max_degree = int(degree.max()) if n else 0
    # Greedy never needs more than max-degree + 1 colours, so this bitset can never overflow.
    n_words = max_degree // _WORD + 2 if n else 1
    taken = np.zeros((n, n_words), dtype=np.uint64)  # colours held by each vertex's neighbours
    saturation = np.zeros(n, dtype=np.int64)  # popcount of `taken`, maintained incrementally
    # One scalar ranking both keys: saturation dominates, degree breaks ties.
    tie = max_degree + 1 if n else 1
    degree_list = degree.tolist()

    # One bucket per reachable rank value: `bucket_head[r]` is the vertex at the head of rank r's
    # list (-1 if empty), `bucket_next`/`bucket_prev` its doubly linked neighbours within that list
    # (plain Python lists, not numpy arrays -- this loop is scalar-access-bound, and list indexing
    # avoids numpy's per-element scalar-wrapping overhead). `top` is the highest bucket known to
    # possibly be non-empty; it only moves down (in the pop loop below) or up (when a push lands
    # above it), never re-scanning the whole range.
    bucket_head = [-1] * (tie * max_degree + max_degree + 1)
    bucket_next = [-1] * n
    bucket_prev = [-1] * n
    top = 0
    for vertex, rank in enumerate(degree_list):  # initial rank = degree (saturation is 0)
        head = bucket_head[rank]
        bucket_next[vertex] = head
        if head >= 0:
            bucket_prev[head] = vertex
        bucket_head[rank] = vertex
        if rank > top:
            top = rank

    for _ in range(n):
        while bucket_head[top] < 0:
            top -= 1
        vertex = bucket_head[top]
        following = bucket_next[vertex]
        bucket_head[top] = following
        if following >= 0:
            bucket_prev[following] = -1

        for word in range(n_words):
            free = ~int(taken[vertex, word]) & _FULL_WORD
            if free:
                chosen = word * _WORD + (free & -free).bit_length() - 1
                break
        colour[vertex] = chosen

        neighbours = indices[indptr[vertex] : indptr[vertex + 1]]
        word, bit = chosen // _WORD, np.uint64(1) << np.uint64(chosen % _WORD)
        previous = taken[neighbours, word]
        taken[neighbours, word] = previous | bit
        # Only a neighbour that did not already see this colour becomes more saturated -- and a
        # neighbour that is already coloured must never be moved between buckets.
        gained = neighbours[(previous & bit) == 0]
        gained = gained[colour[gained] < 0]
        old_ranks = (saturation[gained] * tie + degree[gained]).tolist()
        saturation[gained] += 1
        new_ranks = (saturation[gained] * tie + degree[gained]).tolist()

        for v, r_old, r_new in zip(gained.tolist(), old_ranks, new_ranks, strict=True):
            p, following = bucket_prev[v], bucket_next[v]
            if p >= 0:
                bucket_next[p] = following
            else:
                bucket_head[r_old] = following
            if following >= 0:
                bucket_prev[following] = p
            head = bucket_head[r_new]
            bucket_next[v] = head
            bucket_prev[v] = -1
            if head >= 0:
                bucket_prev[head] = v
            bucket_head[r_new] = v
            if r_new > top:
                top = r_new
    return colour


def block_stencil_gather_map(
    plan: ColumnProbePlan, leading_fields: int | None = None
) -> ProbeGather:
    """Precompute the fixed field-major CSR structure and a gather map from the probe responses to it.

    The sparsity pattern and the coloured probes are fixed (they depend only on the mesh graph), so the
    materialize's de-compression -- which probe response each Jacobian entry reads -- is the same every
    time; only the response *values* change. This precomputes, once, the **full-pattern** CSR structure
    (``indptr``, ``indices``) and the de-compression that fills it, sorted so that each probe chunk's
    entries are one contiguous slice (:meth:`ProbeGather.scatter`) -- so a materialize needs no CSR
    re-sort and never holds the full response array. The full
    pattern (no ``eliminate_zeros``) makes the structure truly fixed, which is exactly what an in-place
    preconditioner refactor needs; an aggregation multigrid tolerates the explicit zeros (a
    strength-of-connection coarsening ignores a zero coupling). An incomplete/complete factorization does
    **not** -- an explicit zero is fill -- so this path is for the multigrid preconditioner only.

    Degree of freedom ``(cell i, field f)`` is the flat index ``f * n_cells + i``; the probe for
    ``(colour c, field b)`` is index ``probe_base[b] + c`` (the order
    :meth:`ColumnProbePlan.seed_block` builds them), so the response element feeding entry
    ``(row_dof = a n + i, col_dof = b n + j)`` is
    ``responses[probe_base[b] + colour[b][j]][a n + i]``, i.e. flat index
    ``(probe_base[b] + colour[b][j]) * (n_fields n) + row_dof``.

    With ``leading_fields`` the matrix is never laid out whole: the entries are ordered block by block
    over the partition of the fields into the first ``leading_fields`` and the rest (each block in its
    own CSR order), so the four blocks a field split consumes are slices of one value array
    (:func:`materialize_blocks`) rather than cut out of a full matrix and its shifted copy. Without it
    the layout is the one group, the whole matrix in CSR order, so a single block inverse is fitted from
    the same device values by the same path.

    Parameters
    ----------
    plan : ColumnProbePlan
        The assembly pattern and the colouring probing each column field.
    leading_fields : int, optional
        Partition the stored entries into blocks at this many leading fields (see above). ``None``
        (default) keeps the whole matrix's CSR order, as a layout of one group.

    Returns
    -------
    ProbeGather
        The fixed CSR structure (or block layout) and the de-compression, chunk by chunk.
    """
    leading = plan.n_fields if leading_fields is None else leading_fields
    stored = _stored_blocks(plan.active_rows, leading)
    if stored is None:
        return _dof_probe_gather(plan, leading)
    return _cell_probe_gather(plan, leading, stored)


def _stored_blocks(active: np.ndarray, leading: int) -> tuple[bool, bool, bool, bool] | None:
    """Which of a two-group partition's four blocks are stored, when ``active`` keeps or drops each whole.

    ``None`` when some block keeps part of its field pairs: its pattern is then no single cell pattern
    over a full block of field pairs, and the probe is laid out entry by entry instead.
    """
    groups = (slice(0, leading), slice(leading, active.shape[0]))
    stored = []
    for rows, columns in ((g, h) for g in groups for h in groups):
        block = active[rows, columns]
        if block.size and not (block.all() or not block.any()):
            return None
        stored.append(bool(block.size) and bool(block.all()))
    return tuple(stored)


def _cell_probe_gather(
    plan: ColumnProbePlan, leading: int, stored: tuple[bool, bool, bool, bool]
) -> ProbeGather:
    """The structure of :func:`block_stencil_gather_map` described at cell level (see :class:`_CellEntries`)."""
    n = plan.n_cells
    pattern, order = KroneckerPattern.of_coordinates(
        plan.pattern_rows, plan.pattern_cols, (n, n), (plan.n_fields, plan.n_fields)
    )
    layout = BlockLayout.of_cells(pattern.cells, plan.n_fields, leading, stored)
    entries = _CellEntries(plan, pattern.cells, plan.in_reach[:, order], layout, stored)
    return ProbeGather(entries, plan.n_fields * n, layout)


def _dof_probe_gather(plan: ColumnProbePlan, leading: int) -> ProbeGather:
    """The structure of :func:`block_stencil_gather_map` described entry by entry (see :class:`_DofEntries`).

    Every pattern takes this path when its field pairs are not kept or dropped a whole block at a time.
    """
    n = plan.n_cells
    n_fields = plan.n_fields
    nf = n_fields * n
    rows_i, cols_i = plan.pattern_rows, plan.pattern_cols  # cell-block (i, j) coordinates
    shape = (
        rows_i.shape[0],
        n_fields,
        n_fields,
    )  # (block, a-field, b-field), the entry grid per block
    zero_slot = plan.n_probes * nf
    # Every index formed here is bounded by `zero_slot`, so the whole build runs in 32-bit wherever
    # that fits -- which is every mesh this path is used on.
    index_dtype = _index_dtype(zero_slot)
    # Expand each cell-block (i, j) to its n_fields x n_fields DOF entries and, for each, the flat index of
    # the probe response that supplies it. Shapes broadcast over (block, a-field, b-field).
    ri = rows_i.astype(index_dtype)[:, None, None]
    cj = cols_i.astype(index_dtype)[:, None, None]
    a = np.arange(n_fields, dtype=index_dtype)[None, :, None]
    b = np.arange(n_fields, dtype=index_dtype)[None, None, :]
    # The probe that carries column field b of column cell j, per (block, b) — each column field has its
    # own colouring, so this is a lookup per field rather than one shared colour array.
    probe = (
        np.asarray(plan.probe_base, dtype=index_dtype)[:, None]
        + plan.colour[:, cols_i].astype(index_dtype)
    ).T[:, None, :]
    row_dof_2d = a * index_dtype(n) + ri  # (block, a-field, 1): the row degree of freedom a*n + i
    # (probe index for (b, colour_b[j])) * nf + row_dof -> the flat index into responses.ravel()
    source = probe * index_dtype(nf) + row_dof_2d  # materializes the full (block, a, b) grid
    # An entry outside its column's own reach reads the zero row the materialize appends, not that
    # column's response -- the response there holds a different cell's near coupling. Written into
    # `source` rather than into a copy of it, which at this size is a whole second grid.
    # Inverted before it is broadcast, not after: negating the broadcast view would materialize the
    # full grid again, where negating the per-(block, column-field) mask costs one entry per block.
    np.copyto(
        source,
        index_dtype(zero_slot),
        where=np.broadcast_to((~plan.in_reach).T[:, None, :], shape),
    )
    row_dof = np.broadcast_to(row_dof_2d, shape).ravel()  # a*n + i
    col_dof = np.broadcast_to(b * index_dtype(n) + cj, shape).ravel()  # b*n + j
    source = source.ravel()
    # A row/column field pair a consumer declared it will never read (`active_rows`) is not a reach
    # question -- it is true at every cell-block, for every state, for the life of the plan -- so it is
    # dropped from the pattern outright rather than routed to the zero-fill sentinel `in_reach` uses.
    # That difference matters: an `in_reach` zero is KEPT (a fixed pattern an in-place refactor needs),
    # while an `active_rows`-excluded entry never had anywhere to be refactored INTO, because no consumer
    # of this plan ever reads it. Filtering here, before the sparsity is built, is what keeps the saving
    # out of the assembled matrix's own memory rather than only out of its values.
    active = plan.active_rows
    if not active.all():
        keep = np.broadcast_to(active[None, :, :], shape).ravel()
        source, row_dof, col_dof = source[keep], row_dof[keep], col_dof[keep]
    # Sort into CSR order by carrying the source index as the data; there are no duplicate
    # (row_dof, col_dof), so `tocsr` only reorders (never sums), and its data is the gather map in CSR
    # order. The payload stays an integer throughout: routing it through float64 and back, as this once
    # did, costs four full-length arrays at double the width for no gain.
    csr = sp.coo_matrix((source, (row_dof, col_dof)), shape=(nf, nf)).tocsr()
    if csr.nnz != source.size:  # a collision would corrupt the map by summing two sources
        raise ValueError(
            "block_stencil_gather_map: duplicate (row, col) in the pattern (bad colouring)."
        )
    layout, order = BlockLayout.of(csr, leading * n)
    return ProbeGather(_DofEntries(csr.data[order], zero_slot, nf), nf, layout)


@functools.partial(jax.jit, static_argnames=("rows", "whole_state", "dtype"))
def _indicator_seeds(
    colour: jnp.ndarray,
    probe_field: jnp.ndarray,
    probe_colour: jnp.ndarray,
    start: int,
    count: int,
    *,
    rows: int,
    whole_state: bool,
    dtype: np.dtype,
) -> jnp.ndarray:
    """The seeds of probes ``[start, start + count)`` padded to ``rows`` rows: :meth:`ColumnProbePlan.seed_block`.

    Row ``r`` is the indicator of the cells probe ``start + r`` seeds, in its field (``whole_state``) or
    on its own; a row past ``count`` asks for colour ``-1``, which no cell has, so it is zero.
    """
    lane = jnp.arange(rows)
    probe = jnp.minimum(start + lane, probe_field.shape[0] - 1)
    field = probe_field[probe]
    wanted = jnp.where(lane < count, probe_colour[probe], -1)
    if not whole_state:
        return (colour[field] == wanted[:, None]).astype(dtype)
    n_fields, n_cells = colour.shape
    in_field = jnp.arange(n_fields)[None, :, None] == field[:, None, None]
    seeds = in_field & (colour[None] == wanted[:, None, None])
    return seeds.reshape(rows, n_fields * n_cells).astype(dtype)


def materialize_block_jacobian(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    plan: ColumnProbePlan,
    *,
    batched_matvec: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    probe_batch_size: int | None = None,
    structure: ProbeGather | None = None,
) -> sp.csr_matrix:
    """Assemble the field-major block Jacobian of ``matvec`` by compressed probing.

    Runs one directional derivative per (colour, field): a seed with a one in field ``b`` of every
    cell of the colour, whose response holds column ``b`` of the Jacobian's block for every
    ``(row cell, column cell)`` in that colour's share of the sparsity pattern (collision-free by the
    colouring). Degree of freedom ``(cell i, field f)`` is the flat index ``f * n_cells + i``.

    The plan may probe different column fields at different stencil reaches, in which case they cost
    different numbers of probes; the assembled pattern is the plan's, whatever the individual columns
    were probed at.

    The plan's probes share the operator's linearization (one fixed state), so they can
    be evaluated as one batched directional derivative instead of a Python loop of separate calls. Pass
    ``batched_matvec`` -- a ``(k, nf) -> (k, nf)`` map applying ``matvec`` to ``k`` stacked seeds at once
    (e.g. ``jax.vmap`` of the jvp, **built once and reused** so it compiles a single time) -- to take that
    path; the responses are the per-probe loop's to rounding (the same directional derivative per seed,
    in a program compiled differently), so it is a speedup, not an approximation. A
    :class:`FieldwiseMatvec` instead takes one column field's seeds at a time and differentiates the
    residual in that field alone. ``probe_batch_size`` chunks the batch to bound peak memory (``k``
    simultaneous tangents); a chunk never spans two column fields, and ``None`` evaluates each column
    field's probes in one batch. Without ``batched_matvec`` the per-probe loop is used, so any ``matvec``
    (including a non-``vmap``-able NumPy one) still works.

    Parameters
    ----------
    matvec : callable
        The Jacobian-vector product ``v -> J v`` (e.g. ``lambda v: jax.jvp(residual, (state,), (v,))[1]``
        at a frozen state), mapping and returning a flat vector of length ``n_fields * n_cells``. Used for
        the per-probe loop when ``batched_matvec`` is not given.
    plan : ColumnProbePlan
        The assembly pattern and the colouring probing each column field
        (:meth:`ColumnProbePlan.uniform` for one reach throughout, :func:`column_probe_plan` to give
        each column its own).
    batched_matvec : callable or FieldwiseMatvec, optional
        A batched form of ``matvec``: ``(k, nf) -> (k, nf)`` applying the same directional derivative to
        ``k`` stacked seeds at once, or a :class:`FieldwiseMatvec` taking ``(field, (k, n_cells))``. When
        given, probing runs batched (a few fused passes) instead of the per-probe loop. Must be built once
        and reused by the caller so it compiles a single time.
    probe_batch_size : int, optional
        The batched-path chunk size (number of simultaneous tangents), to bound peak memory. ``None``
        (default) runs each column field's probes in one batch. Ignored on the per-probe loop.
    structure : ProbeGather, optional
        A precomputed :func:`block_stencil_gather_map` for this plan. When given, each chunk's responses
        are scattered into the **fixed full-pattern** CSR as they are computed and then dropped -- so the
        materialize never holds more than one chunk, and ``eliminate_zeros`` is **not** applied, so the
        structure stays fixed -- which an in-place multigrid refactor needs.

        **A stored exactly-zero entry is a fill slot for any incomplete factorization, so it must not reach
        one.** On this path the stored zeros are a large share of the pattern (8.0M of 47.2M on a 23k-cell
        three-dimensional coupled mesh), and handing them to the multigrid's incomplete-LU level smoother
        stops the zero-shift operator converging. They are left in place here, so this path stays
        fixed-pattern; a consumer that hands the matrix to an incomplete factorization must prune them
        first.

    Returns
    -------
    scipy.sparse.csr_matrix
        The assembled Jacobian, shape ``(n_fields * n, n_fields * n)``. With ``structure`` the pattern is the
        full colouring pattern (explicit zeros kept); otherwise numerically-zero entries are eliminated.
    """
    n = plan.n_cells
    n_fields = plan.n_fields
    nf = n_fields * n
    rows_i, cols_i = plan.pattern_rows, plan.pattern_cols

    # One seed per (field, colour): a one in field `b` of every cell of that field's colour. These probes
    # are independent directional derivatives sharing the fixed linearization, so they can be batched.
    # The seeds are built one chunk at a time into a reused buffer rather than all at once: the full set
    # is `n_probes * nf` floats -- hundreds of megabytes on a three-dimensional coupled case, and held
    # for the whole materialize -- while a chunk is a few, and rebuilding an indicator vector is cheap.
    n_probes = plan.n_probes
    probing = _ChunkProbe(plan, probe_batch_size, matvec, batched_matvec)

    if structure is not None:
        if structure.indptr is None:
            raise ValueError(
                "this structure lays its entries out block by block; materialize_blocks recovers it."
            )
        # Read back once, whole: this path hands the matrix to a host consumer.
        data = np.array(structure.values(probing))
        return sp.csr_matrix((data, structure.indices, structure.indptr), shape=(nf, nf))

    probes = [plan.probe(index) for index in range(n_probes)]
    responses = np.empty((n_probes, nf), dtype=np.float64)
    for field, start, m in probing.chunks():
        responses[start : start + m] = np.asarray(probing.responses(field, start, m))[:m]

    # De-compress by scatter: each probe's response holds column `b` of every block in its colour's share.
    # The share is per (field, colour), since each column field carries its own colouring -- and only
    # the entries inside that column's reach, the rest being zero (see ColumnProbePlan).
    group_blocks: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for b, c in probes:
        in_group = np.isin(cols_i, np.where(plan.colour[b] == c)[0]) & plan.in_reach[b]
        group_blocks[(b, c)] = (rows_i[in_group], cols_i[in_group])
    rows_out: list[np.ndarray] = []
    cols_out: list[np.ndarray] = []
    vals_out: list[np.ndarray] = []
    for (b, c), response in zip(probes, responses, strict=True):
        block_rows, block_cols = group_blocks[(b, c)]
        for a in range(n_fields):
            if not plan.active_rows[a, b]:  # a block this plan's consumer will never read
                continue
            rows_out.append(a * n + block_rows)
            cols_out.append(b * n + block_cols)
            vals_out.append(response[a * n + block_rows])
    jacobian = sp.csr_matrix(
        (np.concatenate(vals_out), (np.concatenate(rows_out), np.concatenate(cols_out))),
        shape=(nf, nf),
    )
    jacobian.eliminate_zeros()
    return jacobian


def materialize_blocks(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    plan: ColumnProbePlan,
    *,
    structure: ProbeGather,
    batched_matvec: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    probe_batch_size: int | None = None,
) -> ProbedBlocks:
    """The block Jacobian of ``matvec`` over ``structure``'s partition, by the same coloured probing.

    The counterpart of :func:`materialize_block_jacobian` that keeps the values on the device: the
    probes fill one block-major value array (:func:`block_stencil_gather_map`), and the four blocks are
    matrices over slices of it. Entry for entry, the blocks are those
    :meth:`~aquaflux.precondition.FieldGroups.blocks` cuts from the whole matrix, explicit zeros included;
    over one group, the leading block is the whole matrix.

    Parameters
    ----------
    matvec, plan, batched_matvec, probe_batch_size
        As :func:`materialize_block_jacobian` takes them.
    structure : ProbeGather
        The structure, with its :class:`BlockLayout`.

    Returns
    -------
    ProbedBlocks
        The values and their blocks.
    """
    return ProbedBlocks(
        structure.values(_ChunkProbe(plan, probe_batch_size, matvec, batched_matvec)),
        structure.layout,
    )


class FieldwiseMatvec:
    """A batched Jacobian product that takes one column field's seeds at a time.

    ``apply(field, seeds)`` returns ``J @ e`` for each row of ``seeds`` embedded in column field
    ``field`` of an otherwise zero direction: shape ``(k, n_cells) -> (k, n_fields * n_cells)``. Passed
    as ``batched_matvec``, it makes the probe seed each chunk with that one field
    (:meth:`ColumnProbePlan.field_seeds`) rather than with the whole state.

    **Why a product would rather be told the field.** A direction given as a whole state, zero outside
    one field, still carries a tangent into every field: the zeros are values, and every kernel
    downstream of them runs on them. A product that differentiates the residual in the one field alone
    leaves the others constants of the derivative, so the work they would have carried is never
    traced; for a coupled flow residual, a pressure or turbulence column never forms the velocity
    gradient's tangents at all.

    Attributes
    ----------
    apply : callable
        ``(field, seeds) -> responses`` as above. Must be built once and reused, so each field's
        program compiles a single time.
    """

    def __init__(self, apply: Callable[[int, jnp.ndarray], jnp.ndarray]) -> None:
        self.apply = apply

    def __call__(self, field: int, seeds: jnp.ndarray) -> jnp.ndarray:
        return self.apply(field, seeds)

    def seed_dtype(self, field: int) -> np.dtype:
        """The floating type ``field``'s seeds are built in: every field's program runs in double."""
        return np.dtype(np.float64)


class _ChunkProbe:
    """One materialize's probes, a chunk at a time, through whichever product the caller supplied.

    Chunks never span two column fields (:meth:`ColumnProbePlan.field_chunks`), so a
    :class:`FieldwiseMatvec` can be handed one field's seeds; a whole-state product is seeded with the
    same chunks laid out over the whole state. Either way the seeds keep one shape across every chunk
    of a field, so a batched product compiles once per program and a padded final chunk needs no special
    case; without a batched product each chunk is a plain loop of ``matvec`` over its rows, which any
    ``matvec`` supports, including a non-vectorizable NumPy one.
    """

    def __init__(self, plan: ColumnProbePlan, probe_batch_size, matvec, batched_matvec) -> None:
        self._plan = plan
        # Unset: each column field's probes in one batch.
        self._chunk = max(plan.n_colours) if probe_batch_size is None else max(1, probe_batch_size)
        self._matvec = matvec
        self._batched = batched_matvec
        self._fieldwise = isinstance(batched_matvec, FieldwiseMatvec)

    def chunks(self):
        """``(field, start, count)`` for every chunk, as :meth:`ColumnProbePlan.field_chunks`."""
        return self._plan.field_chunks(self._chunk)

    def responses(self, field: int, start: int, count: int) -> jnp.ndarray:
        """Responses of probes ``[start, start + count)``, shape ``(rows >= count, n_fields * n_cells)``.

        Seeded and left on the device, where :meth:`ProbeGather.scatter` reads them.
        """
        plan, end = self._plan, start + count
        if self._fieldwise:
            seeds = plan.field_seeds(start, end, self._chunk, self._batched.seed_dtype(field))
            return jnp.asarray(self._batched(field, seeds), dtype=jnp.float64)
        seeds = plan.seed_block(start, end, self._chunk)
        if self._batched is not None:
            return jnp.asarray(self._batched(seeds), dtype=jnp.float64)
        return jnp.stack(
            [jnp.asarray(self._matvec(seeds[row]), dtype=jnp.float64) for row in range(count)]
        )


def jacobian_relative_error(
    jacobian: sp.csr_matrix, matvec: Callable[[jnp.ndarray], jnp.ndarray], seed: int = 0
) -> float:
    """Relative error of a materialized Jacobian against the operator on a random vector.

    The cheap guard that the colouring's ``reach`` covers the true stencil: too small a reach silently
    drops the far couplings, which shows up here as a large error. Both sides are exact forward-mode
    derivatives (the probe is a ``jax.jvp``, not a difference quotient), so a faithful materialization
    returns the floating-point round-off floor.

    Parameters
    ----------
    jacobian : scipy.sparse.csr_matrix
        The materialized Jacobian.
    matvec : callable
        The true Jacobian-vector product ``v -> J v``.
    seed : int
        Seed for the random probe vector.

    Returns
    -------
    float
        ``||J v - matvec(v)|| / ||matvec(v)||`` for a random ``v``.
    """
    v = np.random.default_rng(seed).standard_normal(jacobian.shape[0])
    reference = np.asarray(matvec(jnp.asarray(v)), dtype=np.float64)
    return float(np.linalg.norm(jacobian @ v - reference) / (np.linalg.norm(reference) + 1e-300))


def shifted_jacobian(jacobian: sp.csr_matrix, shift_diagonal: np.ndarray) -> sp.csr_matrix:
    """Add the pseudo-transient shift ``beta d`` to a materialized Jacobian's diagonal.

    Cheap -- ``O(nnz)`` numpy, no sparse arithmetic. The single home for the shift every host
    preconditioner adds before factoring or coarsening, so the three cannot disagree about what
    "the shifted Jacobian" is.

    Written as a **diagonal assignment** rather than ``jacobian + sp.diags(shift)``, for the same reason
    the equilibration is (:func:`~aquaflux.precondition.frozen_operator.apply_symmetric_scale`): a sparse
    *addition* stores only entries whose result is nonzero, so it silently drops every explicit zero the
    coloured probe deliberately kept when it assembled against a fixed pattern. Those stored zeros are
    the pattern, and a zero-fill incomplete factorization takes its own pattern from exactly that.

    The two spellings are otherwise **identical, including the sparsity pattern** -- verified over a
    matrix with a full diagonal, one with diagonal entries missing (both create them), and one carrying
    explicit zeros, which is the only case where they part. So this is a no-op for a caller that
    materialized without a precomputed ``structure`` (nothing is storing zeros there) and a correctness
    fix for one that did.

    Parameters
    ----------
    jacobian : sp.csr_matrix
        The materialized Jacobian **without** the shift, shape ``(n_dofs, n_dofs)``.
    shift_diagonal : np.ndarray
        The shift to add to the diagonal, shape ``(n_dofs,)`` -- the same block-diagonal shift the step
        solves against (velocity/scalar shifts, pressure zero).

    Returns
    -------
    sp.csr_matrix
        A copy of ``jacobian`` with the shift on its diagonal. The input is not modified.
    """
    shifted = jacobian.tocsr().copy()
    shifted.setdiag(shifted.diagonal() + np.asarray(shift_diagonal, dtype=np.float64))
    return shifted
