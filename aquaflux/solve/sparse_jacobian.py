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


def materialize_block_jacobian(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    colouring: BlockColouring,
    n_fields: int,
    *,
    batched_matvec: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    probe_batch_size: int | None = None,
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

    Returns
    -------
    scipy.sparse.csr_matrix
        The assembled Jacobian, shape ``(n_fields * n, n_fields * n)``.
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

    # De-compress: each probe's response holds column `b` of every block in its colour's share.
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
