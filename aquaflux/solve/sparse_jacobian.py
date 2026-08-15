"""Materialize the sparse Jacobian of a block residual, by compressed (graph-coloured) probing.

Some preconditioners need the assembled coupled Jacobian as a sparse matrix rather than as a
matrix-vector product — an incomplete-factorization (ILU) preconditioner factors the matrix, which a
matrix-free operator cannot supply. This module recovers that matrix from the *same* residual the
solver uses, so there is no second, hand-derived assembly to drift from it: it probes the residual's
directional derivative (one ``jax.jvp`` per colour) and de-compresses the responses into the sparse
matrix, exploiting the mesh's finite stencil so a handful of probes recover the whole Jacobian instead
of one probe per degree of freedom.

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

from collections.abc import Callable, Sequence

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp


def _index_dtype(bound: int) -> type[np.signedinteger]:
    """The narrowest signed integer type holding every value up to ``bound``.

    The arrays this module builds carry one entry per Jacobian nonzero — tens of millions on a
    three-dimensional coupled mesh — and several are live at once while the de-compression is assembled,
    so the width of an index is the difference between a peak of a few hundred megabytes and several
    gigabytes. Every such array here has a bound known before it is formed, which is what lets the type
    be chosen up front and the wide form never materialize.
    """
    return np.int32 if bound < np.iinfo(np.int32).max else np.int64


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

    So a shorter reach is only admissible for a column measured to carry no mass beyond it. Choosing one
    on the assumption that the far couplings are "small" silently perturbs the near entries instead.

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
    in_reach : np.ndarray
        Per column field and pattern entry, whether that entry lies within the column's own reach;
        shape ``(n_fields, n_blocks)``. Entries outside it are **forced to zero** rather than read from
        a response — see the note below.

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
    ) -> None:
        self.n_cells = n_cells
        self.n_fields = n_fields
        self.pattern_rows = pattern_rows
        self.pattern_cols = pattern_cols
        self.colour = colour
        self.n_colours = n_colours
        self.reach = reach
        self.in_reach = (
            np.ones((n_fields, pattern_rows.shape[0]), dtype=bool) if in_reach is None else in_reach
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

    @classmethod
    def uniform(cls, colouring: BlockColouring, n_fields: int) -> ColumnProbePlan:
        """Probe every column field at one reach — the plan a single :class:`BlockColouring` describes."""
        return cls(
            colouring.n_cells,
            n_fields,
            colouring.pattern_rows,
            colouring.pattern_cols,
            np.broadcast_to(colouring.colour, (n_fields, colouring.n_cells)),
            (colouring.n_colours,) * n_fields,
            (colouring.reach,) * n_fields,
        )

    def probe(self, index: int) -> tuple[int, int]:
        """The ``(column field, colour)`` that flat probe ``index`` stands for."""
        return int(self._probe_field[index]), int(self._probe_colour[index])

    def seed_block(self, start: int, stop: int, out: np.ndarray | None = None) -> np.ndarray:
        """Seed vectors for probes ``[start, stop)``, shape ``(rows, n_fields * n_cells)``.

        Probe ``probe_base[b] + c`` carries a one in field ``b`` of every cell of colour ``c`` — the
        directional derivative that reads column ``b`` of every block in that colour's share of the
        pattern.

        **Built a block at a time, and reusing ``out``, because the full set is large and needed only
        one chunk at a time.** All the seeds together are ``n_probes * n_fields * n_cells`` floats —
        hundreds of megabytes on a three-dimensional coupled case, held for the whole materialize
        against a few megabytes for one chunk — and each is a cheap indicator vector, so keeping them
        costs far more than rebuilding them.

        ``out`` may be **longer** than ``stop - start``; the surplus rows are left zero, which is
        exactly the padding a batched map needs to keep one compiled shape on a final short chunk.

        Parameters
        ----------
        start, stop : int
            Half-open range of flat probe indices.
        out : np.ndarray, optional
            A buffer of shape ``(rows >= stop - start, n_fields * n_cells)`` to fill. It is zeroed
            first, so a caller may reuse one buffer across every chunk. ``None`` allocates one.

        Returns
        -------
        np.ndarray
            The buffer, filled.
        """
        n = self.n_cells
        rows = stop - start
        if out is None:
            out = np.zeros((rows, self.n_fields * n))
        else:
            out[:] = 0.0
        for row, index in enumerate(range(start, stop)):
            field, colour = self.probe(index)
            out[row, field * n + np.where(self.colour[field] == colour)[0]] = 1.0
        return out


def column_probe_plan(
    owner: np.ndarray,
    nb: np.ndarray,
    n: int,
    column_reach: Sequence[int],
    pattern_reach: int | None = None,
) -> ColumnProbePlan:
    """A probing plan giving each column field its own stencil reach.

    The **assembly pattern** is separate from what the columns are probed at, and keeping it at the full
    ``pattern_reach`` is what makes the saving free downstream: every consumer sees the same sparsity as
    a uniform-reach build, so a coarsening hierarchy and any factorization over the pattern are
    unchanged. A short-probed column simply leaves the outer positions at the explicit zeros they held
    anyway.

    One colouring is built per *distinct* reach rather than per field, since fields sharing a reach share
    a colouring.

    **Only shorten a column measured to carry no mass beyond its reach** — see
    :class:`ColumnProbePlan` for why a column with far couplings is corrupted rather than truncated.
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
        flat_inner = np.sort(inner.pattern_rows.astype(np.int64) * n + inner.pattern_cols)
        where = np.searchsorted(flat_inner, flat_pattern)
        in_reach[field] = (where < flat_inner.size) & (
            flat_inner[np.minimum(where, flat_inner.size - 1)] == flat_pattern
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
    )


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
    if n < 1:
        raise ValueError(f"block_stencil_colouring: need at least one cell, got n={n}.")
    if reach < 1:
        raise ValueError(f"block_stencil_colouring: reach must be >= 1, got {reach}.")
    owner, nb = np.asarray(owner), np.asarray(nb)
    if owner.shape != nb.shape:
        raise ValueError("block_stencil_colouring: owner and nb must have the same shape.")
    if owner.size and (owner.min() < 0 or owner.max() >= n or nb.min() < 0 or nb.max() >= n):
        raise ValueError(f"block_stencil_colouring: edge endpoints out of range for n={n} cells.")

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
    conflict = (pattern.T @ pattern).tocsr()
    conflict.data[:] = 1.0

    colour = _saturation_colouring(conflict, n)
    n_colours = int(colour.max()) + 1 if n else 0

    coo = pattern.tocoo()
    return BlockColouring(n, reach, colour, n_colours, coo.row, coo.col)


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
    # Greedy never needs more than max-degree + 1 colours, so this bitset can never overflow.
    n_words = int(degree.max()) // _WORD + 2 if n else 1
    taken = np.zeros((n, n_words), dtype=np.uint64)  # colours held by each vertex's neighbours
    saturation = np.zeros(n, dtype=np.int64)  # popcount of `taken`, maintained incrementally
    # One scalar ranking both keys: saturation dominates, degree breaks ties, -1 marks "already done".
    tie = int(degree.max()) + 1 if n else 1
    rank = saturation * tie + degree

    for _ in range(n):
        vertex = int(np.argmax(rank))
        for word in range(n_words):
            free = ~int(taken[vertex, word]) & _FULL_WORD
            if free:
                chosen = word * _WORD + (free & -free).bit_length() - 1
                break
        colour[vertex] = chosen
        rank[vertex] = -1

        neighbours = indices[indptr[vertex] : indptr[vertex + 1]]
        word, bit = chosen // _WORD, np.uint64(1) << np.uint64(chosen % _WORD)
        previous = taken[neighbours, word]
        taken[neighbours, word] = previous | bit
        # Only a neighbour that did not already see this colour becomes more saturated -- and a
        # neighbour that is already coloured must keep rank -1 so it is never selected again.
        gained = neighbours[(previous & bit) == 0]
        gained = gained[colour[gained] < 0]
        saturation[gained] += 1
        rank[gained] = saturation[gained] * tie + degree[gained]
    return colour


def block_stencil_gather_map(plan: ColumnProbePlan) -> ProbeGather:
    """Precompute the fixed field-major CSR structure and a gather map from the probe responses to it.

    The sparsity pattern and the coloured probes are fixed (they depend only on the mesh graph), so the
    materialize's de-compression -- which probe response each Jacobian entry reads -- is the same every
    time; only the response *values* change. This precomputes, once, the **full-pattern** CSR structure
    (``indptr``, ``indices``) and a flat index ``gather_map`` such that a materialize is a single gather
    ``data = responses.ravel()[gather_map]`` (no per-materialize scatter loop, no CSR re-sort). The full
    pattern (no ``eliminate_zeros``) makes the structure truly fixed, which is exactly what an in-place
    preconditioner refactor needs; an aggregation multigrid tolerates the explicit zeros (a
    strength-of-connection coarsening ignores a zero coupling). An incomplete/complete factorization does
    **not** -- an explicit zero is fill -- so this path is for the multigrid preconditioner only.

    Degree of freedom ``(cell i, field f)`` is the flat index ``f * n_cells + i``; the probe for
    ``(colour c, field b)`` is index ``probe_base[b] + c`` (the order :meth:`ColumnProbePlan.seeds`
    builds them), so the response element feeding entry ``(row_dof = a n + i, col_dof = b n + j)`` is
    ``responses[probe_base[b] + colour[b][j]][a n + i]``, i.e. flat index
    ``(probe_base[b] + colour[b][j]) * (n_fields n) + row_dof``.

    Parameters
    ----------
    plan : ColumnProbePlan
        The assembly pattern and the colouring probing each column field.

    Returns
    -------
    ProbeGather
        The fixed CSR structure and the de-compression, chunk by chunk.
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
    # Sort into CSR order by carrying the source index as the data; there are no duplicate
    # (row_dof, col_dof), so `tocsr` only reorders (never sums), and its data is the gather map in CSR
    # order. The payload stays an integer throughout: routing it through float64 and back, as this once
    # did, costs four full-length arrays at double the width for no gain.
    csr = sp.coo_matrix((source.ravel(), (row_dof, col_dof)), shape=(nf, nf)).tocsr()
    if csr.nnz != source.size:  # a collision would corrupt the map by summing two sources
        raise ValueError(
            "block_stencil_gather_map: duplicate (row, col) in the pattern (bad colouring)."
        )
    return ProbeGather(csr.indptr, csr.indices, csr.data, zero_slot, nf)


class ProbeGather:
    """The fixed CSR structure of a materialize, and how to fill it from the probe responses.

    Both halves are fixed by the mesh graph, so they are built once and reused by every refresh: which
    probe supplies which Jacobian entry does not change, only the response *values* do.

    **The de-compression runs one probe-chunk at a time, which is what keeps the peak bounded.** The
    obvious form — hold every response, then gather the whole matrix in one indexing expression — needs
    ``n_probes x n_fields x n_cells`` floats live at once, several hundred megabytes on a
    three-dimensional coupled case, and a naive sentinel for the out-of-reach entries doubles that again
    by copying the lot. Sorting the entries by the probe that feeds them instead lets each chunk's
    responses be consumed as soon as they are computed and then dropped, so the only large array left is
    the matrix being built.

    Entries lying outside their column's reach have no source probe at all (see
    :class:`ColumnProbePlan`); they simply never appear in the sorted order, so an output initialized to
    zero leaves them zero and no sentinel is needed.

    Attributes
    ----------
    indptr, indices : np.ndarray
        The fixed CSR structure of the ``(n_fields n, n_fields n)`` Jacobian.
    nnz : int
        Stored entries, including the explicit zeros the full pattern keeps.
    """

    def __init__(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        gather_map: np.ndarray,
        zero_slot: int,
        nf: int,
    ) -> None:
        self.indptr = indptr
        self.indices = indices
        self.nnz = int(indices.shape[0])
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
        # slice. The sort permutation is the one remaining array here that must be full width, since
        # numpy returns an index array; a counting sort over the probe index -- which ranges over a few
        # hundred values -- would avoid it, at the cost of reordering entries within a probe.
        probe_of = source // index_dtype(nf)
        by_probe = np.argsort(probe_of, kind="stable")
        self._position = position[by_probe]
        self._source = source[by_probe]
        # One boundary per probe plus a closing one, so a chunk ending at the last probe still has a
        # `_probe_start[start + count]` to read.
        n_probes = zero_slot // nf
        self._probe_start = np.searchsorted(probe_of[by_probe], np.arange(n_probes + 1))

    def scatter(self, data: np.ndarray, responses: np.ndarray, start: int, count: int) -> None:
        """Fill every entry fed by probes ``[start, start + count)`` from that chunk's responses.

        Parameters
        ----------
        data : np.ndarray
            The CSR data array being filled, length :attr:`nnz`. Must start zeroed — entries outside
            their column's reach are never written.
        responses : np.ndarray
            This chunk's responses, shape ``(rows >= count, n_fields * n_cells)``; rows beyond ``count``
            are ignored, so a padded final chunk needs no special case.
        start, count : int
            The chunk's half-open probe range.
        """
        lo, hi = self._probe_start[start], self._probe_start[start + count]
        if hi > lo:
            data[self._position[lo:hi]] = responses.reshape(-1)[
                self._source[lo:hi].astype(np.int64) - start * self._nf
            ]


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
    path; the responses are bit-identical to the per-probe loop (the same directional derivative per seed),
    so it is a pure speedup. ``probe_batch_size`` chunks the batch to bound peak memory (``k`` simultaneous
    tangents); ``None`` evaluates all probes in one batch. Without ``batched_matvec`` the per-probe loop is
    used, so any ``matvec`` (including a non-``vmap``-able NumPy one) still works.

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
    batched_matvec : callable, optional
        A batched form of ``matvec``: ``(k, nf) -> (k, nf)`` applying the same directional derivative to
        ``k`` stacked seeds at once. When given, probing runs batched (a few fused passes) instead of the
        per-probe loop. Must be built once and reused by the caller so it compiles a single time.
    probe_batch_size : int, optional
        The batched-path chunk size (number of simultaneous tangents), to bound peak memory. ``None``
        (default) runs all probes in one batch. Ignored on the per-probe loop.
    structure : ProbeGather, optional
        A precomputed :func:`block_stencil_gather_map` for this plan. When given, each chunk's responses
        are scattered into the **fixed full-pattern** CSR as they are computed and then dropped -- so the
        materialize never holds more than one chunk, and ``eliminate_zeros`` is **not** applied, so the
        structure stays fixed -- which an in-place multigrid refactor needs.

        **A stored exactly-zero entry is a fill slot for any incomplete factorization, so it must not reach
        one.** On this path the stored zeros are a large share of the pattern (8.0M of 47.2M on a 23k-cell
        three-dimensional coupled mesh), and handing them to the multigrid's incomplete-LU level smoother
        stops the zero-shift operator converging. They are pruned at the boundary where the operator
        reaches the factorization
        (:meth:`~aquaflux.solve.amg_preconditioner.AmgVCycle._live`) rather than here, so this path stays
        fixed-pattern; a consumer that factors the matrix *without* going through that boundary must prune
        it first.

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
    chunk = n_probes if probe_batch_size is None else max(1, probe_batch_size)

    if structure is not None:
        # De-compress AS THE PROBES COME BACK: each chunk's responses are scattered into the fixed CSR
        # and then dropped, so the only large array live at once is the matrix being built. Holding
        # every response instead costs `n_probes * nf` floats -- comparable to the matrix itself.
        data = np.zeros(
            structure.nnz
        )  # zero: entries outside their column's reach are never written
        block = np.zeros((chunk, nf))
        for start in range(0, n_probes, chunk):
            m = min(chunk, n_probes - start)
            plan.seed_block(start, start + m, out=block)
            responses = _probe_chunk(matvec, batched_matvec, block, m)
            structure.scatter(data, responses, start, m)
        return sp.csr_matrix((data, structure.indices, structure.indptr), shape=(nf, nf))

    probes = [plan.probe(index) for index in range(n_probes)]
    responses = np.empty((n_probes, nf), dtype=np.float64)
    block = np.zeros((chunk, nf))
    for start in range(0, n_probes, chunk):
        m = min(chunk, n_probes - start)
        plan.seed_block(start, start + m, out=block)
        responses[start : start + m] = _probe_chunk(matvec, batched_matvec, block, m)[:m]

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
            rows_out.append(a * n + block_rows)
            cols_out.append(b * n + block_cols)
            vals_out.append(response[a * n + block_rows])
    jacobian = sp.csr_matrix(
        (np.concatenate(vals_out), (np.concatenate(rows_out), np.concatenate(cols_out))),
        shape=(nf, nf),
    )
    jacobian.eliminate_zeros()
    return jacobian


def _probe_chunk(matvec, batched_matvec, block, count):
    """This chunk's probe responses, shape ``(rows, n_fields * n_cells)``.

    One batched call when ``batched_matvec`` is given -- the buffer keeps a single shape across every
    chunk, so it compiles once and a padded final chunk needs no special case -- otherwise a plain loop
    over the chunk's rows, which any ``matvec`` supports including a non-vectorizable NumPy one.
    """
    if batched_matvec is not None:
        return np.asarray(batched_matvec(jnp.asarray(block)), dtype=np.float64)
    return np.stack(
        [np.asarray(matvec(jnp.asarray(block[row])), dtype=np.float64) for row in range(count)]
    )


def jacobian_relative_error(
    jacobian: sp.csr_matrix, matvec: Callable[[jnp.ndarray], jnp.ndarray], seed: int = 0
) -> float:
    """Relative error of a materialized Jacobian against the operator on a random vector.

    The cheap guard that the colouring's ``reach`` covers the true stencil: too small a reach silently
    drops the far couplings, which shows up here as a large error. A faithful materialization returns
    the finite-difference / floating-point floor (``~1e-7`` or below for a smooth residual).

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
