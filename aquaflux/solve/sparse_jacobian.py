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

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp


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


def block_stencil_colouring(
    owner: np.ndarray, nb: np.ndarray, n: int, reach: int
) -> BlockColouring:
    """The cell-block sparsity pattern at ``reach`` and a compatible probing colouring.

    The Jacobian couples cell ``i`` to cell ``j`` only when they are within ``reach`` steps on the
    interior-face cell graph (each differencing/interpolation step reaches one ring of neighbours).
    Two column-cells can share a probe colour only if no row-cell couples to both — i.e. they are not
    both within ``reach`` of a common cell — so the conflict graph is the pattern squared, and a
    greedy colouring of it (highest degree first) gives collision-free probes.

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

    colour = np.full(n, -1, dtype=np.int64)
    degree = np.diff(conflict.indptr)
    for i in np.argsort(-degree):
        neighbours = conflict.indices[conflict.indptr[i] : conflict.indptr[i + 1]]
        used = set(colour[neighbours].tolist())
        c = 0
        while c in used:
            c += 1
        colour[i] = c
    n_colours = int(colour.max()) + 1 if n else 0

    coo = pattern.tocoo()
    return BlockColouring(n, reach, colour, n_colours, coo.row, coo.col)


def block_stencil_gather_map(
    colouring: BlockColouring, n_fields: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    ``(colour c, field b)`` is index ``c * n_fields + b`` (the order :func:`materialize_block_jacobian`
    seeds them), so the response element feeding entry ``(row_dof = a n + i, col_dof = b n + j)`` is
    ``responses[colour[j] * n_fields + b][a n + i]``, i.e. flat index
    ``(colour[j] * n_fields + b) * (n_fields n) + row_dof``.

    Parameters
    ----------
    colouring : BlockColouring
        The pattern and colouring from :func:`block_stencil_colouring`.
    n_fields : int
        Degrees of freedom per cell.

    Returns
    -------
    indptr, indices : np.ndarray
        The fixed CSR structure of the ``(n_fields n, n_fields n)`` Jacobian.
    gather_map : np.ndarray
        For each CSR data position, the flat index into ``responses.ravel()`` (shape
        ``(n_colours * n_fields, n_fields * n_cells)``) that supplies its value.
    """
    n = colouring.n_cells
    nf = n_fields * n
    rows_i, cols_i = colouring.pattern_rows, colouring.pattern_cols  # cell-block (i, j) coordinates
    colour = colouring.colour
    # Expand each cell-block (i, j) to its n_fields x n_fields DOF entries and, for each, the flat index of
    # the probe response that supplies it. Shapes broadcast over (block, a-field, b-field).
    ri = rows_i[:, None, None]
    cj = cols_i[:, None, None]
    a = np.arange(n_fields)[None, :, None]
    b = np.arange(n_fields)[None, None, :]
    cb = colour[cols_i][:, None, None]  # colour of column cell j
    shape = (
        rows_i.shape[0],
        n_fields,
        n_fields,
    )  # (block, a-field, b-field), the entry grid per block
    row_dof = np.broadcast_to(a * n + ri, shape).ravel()  # a*n + i
    col_dof = np.broadcast_to(b * n + cj, shape).ravel()  # b*n + j
    # (probe index for (colour[j], b)) * nf + row_dof -> the flat index into responses.ravel()
    flat_src = np.broadcast_to((cb * n_fields + b) * nf + (a * n + ri), shape).ravel()
    # Sort into CSR order by carrying flat_src as the data; there are no duplicate (row_dof, col_dof), so
    # `tocsr` only reorders (never sums), and its data is the gather map in CSR order.
    csr = sp.coo_matrix((flat_src.astype(np.float64), (row_dof, col_dof)), shape=(nf, nf)).tocsr()
    if csr.nnz != flat_src.size:  # a collision would corrupt the map by summing two sources
        raise ValueError(
            "block_stencil_gather_map: duplicate (row, col) in the pattern (bad colouring)."
        )
    return csr.indptr, csr.indices, csr.data.astype(np.int64)


def materialize_block_jacobian(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    colouring: BlockColouring,
    n_fields: int,
    *,
    batched_matvec: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    probe_batch_size: int | None = None,
    structure: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> sp.csr_matrix:
    """Assemble the field-major block Jacobian of ``matvec`` by compressed probing.

    Runs one directional derivative per (colour, field): a seed with a one in field ``b`` of every
    cell of the colour, whose response holds column ``b`` of the Jacobian's block for every
    ``(row cell, column cell)`` in that colour's share of the sparsity pattern (collision-free by the
    colouring). Degree of freedom ``(cell i, field f)`` is the flat index ``f * n_cells + i``.

    The ``n_colours * n_fields`` probes share the operator's linearization (one fixed state), so they can
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
    colouring : BlockColouring
        The pattern and colouring from :func:`block_stencil_colouring`.
    n_fields : int
        Degrees of freedom per cell (e.g. ``dim + 1`` for a flow saddle, ``dim + 3`` for coupled RANS).
    batched_matvec : callable, optional
        A batched form of ``matvec``: ``(k, nf) -> (k, nf)`` applying the same directional derivative to
        ``k`` stacked seeds at once. When given, probing runs batched (a few fused passes) instead of the
        per-probe loop. Must be built once and reused by the caller so it compiles a single time.
    probe_batch_size : int, optional
        The batched-path chunk size (number of simultaneous tangents), to bound peak memory. ``None``
        (default) runs all probes in one batch. Ignored on the per-probe loop.
    structure : tuple of np.ndarray, optional
        A precomputed ``(indptr, indices, gather_map)`` from :func:`block_stencil_gather_map`. When given,
        the de-compression is a single vectorized gather into the **fixed full-pattern** CSR
        (``data = responses.ravel()[gather_map]``) rather than the scatter loop + CSR re-sort, and
        ``eliminate_zeros`` is **not** applied (the structure stays fixed, which an in-place multigrid
        refactor needs; the explicit zeros are harmless to aggregation but would be fill for an incomplete
        factorization, so this path is multigrid-only). Requires ``batched_matvec`` (the gather consumes the
        stacked responses).

    Returns
    -------
    scipy.sparse.csr_matrix
        The assembled Jacobian, shape ``(n_fields * n, n_fields * n)``. With ``structure`` the pattern is the
        full colouring pattern (explicit zeros kept); otherwise numerically-zero entries are eliminated.
    """
    n = colouring.n_cells
    nf = n_fields * n
    rows_i, cols_i = colouring.pattern_rows, colouring.pattern_cols
    groups = [np.where(colouring.colour == c)[0] for c in range(colouring.n_colours)]

    # One seed per (colour, field): a one in field `b` of every cell of the colour. These probes are
    # independent directional derivatives sharing the fixed linearization, so they can be batched.
    probes: list[tuple[int, int]] = []  # (group index, field b)
    seeds: list[np.ndarray] = []
    for gi, group in enumerate(groups):
        for b in range(n_fields):
            seed = np.zeros(nf)
            seed[b * n + group] = 1.0
            probes.append((gi, b))
            seeds.append(seed)

    if batched_matvec is None:
        responses = [np.asarray(matvec(jnp.asarray(s)), dtype=np.float64) for s in seeds]
    else:
        seed_matrix = np.stack(seeds)
        n_probes = seed_matrix.shape[0]
        chunk = n_probes if probe_batch_size is None else max(1, probe_batch_size)
        responses = np.empty((n_probes, nf), dtype=np.float64)
        for start in range(0, n_probes, chunk):
            block = seed_matrix[start : start + chunk]
            m = block.shape[0]
            if m < chunk:  # pad the final chunk to a uniform shape so the batched map compiles once
                block = np.vstack([block, np.zeros((chunk - m, nf))])
            responses[start : start + m] = np.asarray(
                batched_matvec(jnp.asarray(block)), dtype=np.float64
            )[:m]

    if structure is not None:
        # De-compress by one vectorized gather into the fixed full-pattern CSR (no scatter loop, no re-sort).
        indptr, indices, gather_map = structure
        response_matrix = responses if isinstance(responses, np.ndarray) else np.stack(responses)
        data = response_matrix.reshape(-1)[gather_map]
        return sp.csr_matrix((data, indices, indptr), shape=(nf, nf))

    # De-compress by scatter: each probe's response holds column `b` of every block in its colour's share.
    group_blocks = [
        (rows_i[np.isin(cols_i, group)], cols_i[np.isin(cols_i, group)]) for group in groups
    ]
    rows_out: list[np.ndarray] = []
    cols_out: list[np.ndarray] = []
    vals_out: list[np.ndarray] = []
    for (gi, b), response in zip(probes, responses, strict=True):
        block_rows, block_cols = group_blocks[gi]
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
