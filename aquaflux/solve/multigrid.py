"""Matrix-free aggregation multigrid for a symmetric graph-Laplacian operator.

A pressure-Poisson-like operator ``A`` on an unstructured mesh is a **graph Laplacian**: it is
defined by a set of edges ``(owner, nb)`` each carrying a coefficient ``c_e`` (here ``A p`` is
``sum_e c_e (p_owner - p_nb)`` scattered to cells, diagonal ``sum_e c_e``). Unpreconditioned
Krylov on it needs ``O(h^-2)`` iterations; a multigrid V-cycle makes the iteration count
**mesh-independent**, which is what a scalable inner solve for the SIMPLE pressure Schur needs at
large mesh sizes.

Design for a differentiable JAX/GPU pipeline:

* **The aggregation *structure* is integer graph work — built once, off the jit path, per fixed
  mesh** (:func:`build_hierarchy`). Greedy aggregation coarsens the graph; piecewise-constant
  (unsmoothed) prolongation keeps every coarse level a graph Laplacian, so the Galerkin coarse
  operator is again ``(owner, nb, coeff)`` with the coarse coefficient the sum of the fine
  coefficients crossing between aggregates.
* **The coefficients flow through per call, under jit** (:func:`level_coefficients`): the fine
  coefficients change every Newton iterate (they depend on the lagged ``a_P``), but propagating
  them up the fixed hierarchy is just ``segment_sum`` — matvec-only, and ``stop_gradient``-ed by
  the caller, so the whole V-cycle is a frozen linear operator (adjoint-transparent).
* **The V-cycle** (:func:`v_cycle`) is a fixed number of damped-Jacobi pre/post smooths + a coarse
  correction, recursion unrolled at trace time (the number of levels is static). A *fixed* cycle is
  a constant linear operator, so plain left-preconditioned GMRES suffices.

This is unsmoothed aggregation (a first, robust cut); smoothed aggregation is the accuracy upgrade,
at the cost of denser coarse operators. A closed-domain pressure system is regularized by a pin (one
cell per level held to the right-hand side); the pin only affects preconditioner quality, never the
converged solution (the outer solve terminates on the true residual).
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.ops import segment_sum
from scipy.sparse.csgraph import reverse_cuthill_mckee


class _Level(NamedTuple):
    """One multigrid level's static (integer, mesh-fixed) structure."""

    n: int  # cells at this level (static)
    owner: jnp.ndarray  # (n_edges,) edge owner index
    nb: jnp.ndarray  # (n_edges,) edge neighbour index
    pin: int  # pinned cell index at this level, or -1 (static)
    agg: jnp.ndarray | None  # (n,) map cell -> next-coarser cell; None on the coarsest level
    edge_map: (
        jnp.ndarray | None
    )  # (n_edges,) map edge -> next-coarser edge (n_coarse_edges = collapsed)
    n_coarse_edges: int  # number of edges on the next-coarser level (static; 0 on coarsest)


class MultigridHierarchy(NamedTuple):
    """A built aggregation hierarchy: a tuple of :class:`_Level` from finest to coarsest."""

    levels: tuple[_Level, ...]


def _rcm_order(owner: np.ndarray, nb: np.ndarray, n: int) -> np.ndarray:
    """A locality-preserving cell visit order (reverse Cuthill--McKee) for the greedy aggregation.

    The two-pass aggregation is greedy in the order it visits cells, so a spatially-local order gives
    compact, well-shaped aggregates and a near-optimal coarse space, whereas an arbitrary cell
    numbering gives irregular aggregates and a measurably worse V-cycle contraction. Reverse
    Cuthill--McKee supplies that order from the level's own adjacency graph. The graph is undirected,
    so the ``(owner, nb)`` edges are symmetrized before the ordering.

    This is applied per level to the level's *own* operator graph — the fine graph and every
    Galerkin-coarse graph alike — so the coarsening is ordering-robust throughout the hierarchy
    without renumbering the mesh: only the aggregation's visit sequence changes, not any cell label
    the caller sees.
    """
    symmetric = sp.coo_matrix(
        (
            np.ones(2 * len(owner)),
            (np.concatenate([owner, nb]), np.concatenate([nb, owner])),
        ),
        shape=(n, n),
    ).tocsr()
    return reverse_cuthill_mckee(symmetric, symmetric_mode=True)


def _aggregate(owner: np.ndarray, nb: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    """Two-pass aggregation (Vaněk et al.): seed clean aggregates, then attach leftovers.

    Pass 1 forms an aggregate ``{i} ∪ neighbours(i)`` only from a cell ``i`` whose neighbours are all
    still free — giving well-shaped, ~stencil-sized aggregates. Pass 2 attaches each remaining cell to
    an adjacent existing aggregate (rare orphans seed their own). This yields a healthy coarsening
    ratio (~4× in 2D) with no singletons, which a naive one-pass greedy does not.

    Both passes visit cells in a locality-preserving order (:func:`_rcm_order`) so the greedy seeding
    is robust to the incoming cell numbering — an arbitrary order otherwise degrades the coarse space.
    """
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for o, m in zip(owner.tolist(), nb.tolist(), strict=True):
        adjacency[o].append(m)
        adjacency[m].append(o)
    order = _rcm_order(owner, nb, n).tolist()
    aggregate = np.full(n, -1, dtype=np.int64)
    count = 0
    for i in order:  # pass 1: seed from cells in a fully-free neighbourhood
        if aggregate[i] != -1 or any(aggregate[j] != -1 for j in adjacency[i]):
            continue
        aggregate[i] = count
        for j in adjacency[i]:
            aggregate[j] = count
        count += 1
    for i in order:  # pass 2: attach leftovers to an adjacent aggregate (else seed their own)
        if aggregate[i] != -1:
            continue
        neighbour_aggregates = [aggregate[j] for j in adjacency[i] if aggregate[j] != -1]
        if neighbour_aggregates:
            aggregate[i] = neighbour_aggregates[0]
        else:
            aggregate[i] = count
            count += 1
    return aggregate, count


def _coarsen_edges(
    owner: np.ndarray, nb: np.ndarray, aggregate: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Coarse graph from the aggregation: unique inter-aggregate edges and the fine->coarse edge map.

    Returns ``(coarse_owner, coarse_nb, edge_map, n_coarse_edges)`` where ``edge_map[e]`` is the
    coarse-edge index of fine edge ``e`` (or ``n_coarse_edges`` for intra-aggregate edges, which
    collapse and contribute nothing to the coarse operator).
    """
    coarse_o = aggregate[owner]
    coarse_n = aggregate[nb]
    inter = coarse_o != coarse_n
    lo = np.minimum(coarse_o, coarse_n)  # canonical (undirected) pair
    hi = np.maximum(coarse_o, coarse_n)
    pairs = np.stack([lo[inter], hi[inter]], axis=1)
    unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
    n_coarse_edges = len(unique_pairs)
    edge_map = np.full(len(owner), n_coarse_edges, dtype=np.int64)  # intra edges -> collapse index
    edge_map[inter] = inverse.astype(np.int64)
    return unique_pairs[:, 0], unique_pairs[:, 1], edge_map, n_coarse_edges


def build_hierarchy(
    owner: np.ndarray,
    nb: np.ndarray,
    n: int,
    pin: int | None = None,
    *,
    max_coarse: int = 16,
    max_levels: int = 20,
) -> MultigridHierarchy:
    """Build the aggregation hierarchy structure (integer, mesh-fixed) — call once, off the jit path.

    Parameters
    ----------
    owner, nb : np.ndarray
        Fine-level graph edges (interior faces), shape ``(n_edges,)`` each.
    n : int
        Number of fine cells.
    pin : int, optional
        Pinned cell index (closed-domain regularization); propagated to each coarse level.
    max_coarse : int
        Stop coarsening once a level has at most this many cells.
    max_levels : int
        Hard cap on the number of levels.

    Returns
    -------
    MultigridHierarchy
        The finest-to-coarsest level structure, with JAX integer arrays ready for the jit-ed apply.
    """
    owner = np.asarray(owner, dtype=np.int64)
    nb = np.asarray(nb, dtype=np.int64)
    levels: list[_Level] = []
    current_n, current_pin = n, (-1 if pin is None else int(pin))
    while True:
        coarsest = current_n <= max_coarse or len(levels) + 1 >= max_levels or len(owner) == 0
        if coarsest:
            levels.append(
                _Level(current_n, jnp.asarray(owner), jnp.asarray(nb), current_pin, None, None, 0)
            )
            break
        aggregate, n_coarse = _aggregate(owner, nb, current_n)
        c_owner, c_nb, edge_map, n_coarse_edges = _coarsen_edges(owner, nb, aggregate)
        levels.append(
            _Level(
                current_n,
                jnp.asarray(owner),
                jnp.asarray(nb),
                current_pin,
                jnp.asarray(aggregate),
                jnp.asarray(edge_map),
                n_coarse_edges,
            )
        )
        owner, nb = c_owner, c_nb
        current_pin = -1 if current_pin < 0 else int(aggregate[current_pin])
        current_n = n_coarse
    return MultigridHierarchy(tuple(levels))


def _laplacian_diagonal(level: _Level, coeff: jnp.ndarray) -> jnp.ndarray:
    """Diagonal of the level's graph Laplacian (``sum_e c_e`` per cell); pinned row set to 1."""
    diagonal = segment_sum(coeff, level.owner, level.n) + segment_sum(coeff, level.nb, level.n)
    if level.pin >= 0:
        diagonal = diagonal.at[level.pin].set(1.0)
    return diagonal


def level_coefficients(
    hierarchy: MultigridHierarchy, fine_coeff: jnp.ndarray
) -> tuple[tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...]]:
    """Propagate the fine edge coefficients up every level (Galerkin) and form each diagonal.

    Pure ``segment_sum`` — jit/GPU-friendly and differentiable — so the (frozen) fine coefficients
    of the current Newton iterate produce the whole hierarchy's coefficients on the fly.

    Returns ``(coeffs, diagonals)`` aligned with ``hierarchy.levels``.
    """
    coeffs: list[jnp.ndarray] = []
    diagonals: list[jnp.ndarray] = []
    coeff = fine_coeff
    for level in hierarchy.levels:
        coeffs.append(coeff)
        diagonals.append(_laplacian_diagonal(level, coeff))
        if (
            level.edge_map is not None
        ):  # Galerkin coarsen: sum fine coeffs crossing each coarse edge
            coeff = segment_sum(coeff, level.edge_map, level.n_coarse_edges + 1)[
                : level.n_coarse_edges
            ]
    return tuple(coeffs), tuple(diagonals)


def _matvec(
    level: _Level, coeff: jnp.ndarray, p: jnp.ndarray, diag_extra: jnp.ndarray | None = None
) -> jnp.ndarray:
    """Graph-Laplacian matvec ``A p`` at a level; pinned row returns ``p[pin]`` (identity).

    ``diag_extra`` is an optional per-cell diagonal added to the operator (a Dirichlet boundary
    stiffness, e.g. a pressure outlet), making ``A = Laplacian + diag(diag_extra)``.
    """
    flux = coeff * (p[level.owner] - p[level.nb])
    result = segment_sum(flux, level.owner, level.n) - segment_sum(flux, level.nb, level.n)
    if diag_extra is not None:
        result = result + diag_extra * p
    if level.pin >= 0:
        result = result.at[level.pin].set(p[level.pin])
    return result


def _smooth(
    level: _Level,
    coeff: jnp.ndarray,
    diagonal: jnp.ndarray,
    b: jnp.ndarray,
    x: jnp.ndarray,
    sweeps: int,
    omega: float,
    diag_extra: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """A few damped-Jacobi sweeps ``x <- x + omega D^-1 (b - A x)``, holding the pinned cell."""
    inv_diagonal = 1.0 / diagonal
    for _ in range(sweeps):
        x = x + omega * inv_diagonal * (b - _matvec(level, coeff, x, diag_extra))
        if level.pin >= 0:
            x = x.at[level.pin].set(b[level.pin])
    return x


def v_cycle(
    hierarchy: MultigridHierarchy,
    coeffs: tuple[jnp.ndarray, ...],
    diagonals: tuple[jnp.ndarray, ...],
    b: jnp.ndarray,
    *,
    pre_sweeps: int = 2,
    post_sweeps: int = 2,
    coarse_sweeps: int = 30,
    omega: float = 0.7,
    level_index: int = 0,
    diag_extras: tuple[jnp.ndarray, ...] | None = None,
) -> jnp.ndarray:
    """One multigrid V-cycle for ``A x = b`` at ``level_index`` (recursion unrolled at trace time).

    A single fixed V-cycle is a constant linear operator in ``b`` (the smoother sweeps and the coarse
    recursion are all fixed-length), so it is a valid frozen left preconditioner. ``diag_extras``, when
    given, is the per-level Dirichlet boundary diagonal added to each level's operator.
    """
    level = hierarchy.levels[level_index]
    coeff, diagonal = coeffs[level_index], diagonals[level_index]
    extra = None if diag_extras is None else diag_extras[level_index]
    if level.agg is None:  # coarsest level: smooth to (near-)solve the small system
        return _smooth(level, coeff, diagonal, b, jnp.zeros_like(b), coarse_sweeps, omega, extra)

    next_level = hierarchy.levels[level_index + 1]
    x = _smooth(level, coeff, diagonal, b, jnp.zeros_like(b), pre_sweeps, omega, extra)
    residual = b - _matvec(level, coeff, x, extra)
    coarse_residual = segment_sum(residual, level.agg, next_level.n)
    if next_level.pin >= 0:  # the pinned row is an identity Dirichlet condition: zero error there
        coarse_residual = coarse_residual.at[next_level.pin].set(0.0)
    coarse_error = v_cycle(
        hierarchy,
        coeffs,
        diagonals,
        coarse_residual,
        pre_sweeps=pre_sweeps,
        post_sweeps=post_sweeps,
        coarse_sweeps=coarse_sweeps,
        omega=omega,
        level_index=level_index + 1,
        diag_extras=diag_extras,
    )
    x = x + coarse_error[level.agg]  # prolong (piecewise-constant) and correct
    if level.pin >= 0:
        x = x.at[level.pin].set(b[level.pin])
    return _smooth(level, coeff, diagonal, b, x, post_sweeps, omega, extra)


def multigrid_solve(
    hierarchy: MultigridHierarchy,
    coeffs: tuple[jnp.ndarray, ...],
    diagonals: tuple[jnp.ndarray, ...],
    b: jnp.ndarray,
    *,
    cycles: int = 1,
    pre_sweeps: int = 2,
    post_sweeps: int = 2,
    coarse_sweeps: int = 30,
    omega: float = 0.7,
    diag_extras: tuple[jnp.ndarray, ...] | None = None,
) -> jnp.ndarray:
    """A **fixed** number of V-cycles for ``A x = b`` — the mesh-independent, constant-linear inner
    solve for the SIMPLE pressure Schur.

    A fixed cycle count makes ``b -> x`` a constant linear operator, so it is a valid frozen left
    preconditioner under plain GMRES. One cycle (the default) is the usual preconditioner choice.

    Parameters
    ----------
    hierarchy : MultigridHierarchy
        The aggregation structure from :func:`build_hierarchy`.
    coeffs, diagonals : tuple of jnp.ndarray
        Per-level coefficients/diagonals from :func:`level_coefficients` (current iterate).
    b : jnp.ndarray
        Right-hand side, shape ``(n_cells,)``.
    cycles : int
        Number of V-cycles (static).
    pre_sweeps, post_sweeps, coarse_sweeps : int
        Damped-Jacobi sweep counts (static).
    omega : float
        Jacobi damping factor.
    diag_extras : tuple of jnp.ndarray, optional
        Per-level Dirichlet boundary diagonal added to each level's operator (e.g. a pressure-outlet
        stiffness that de-singularises an otherwise pure-Neumann Schur). The ``diagonals`` passed in
        must already include it, so the Jacobi smoother scales by the true diagonal.

    Returns
    -------
    jnp.ndarray
        The approximate solution ``x``, shape ``(n_cells,)``.
    """
    fine = hierarchy.levels[0]
    fine_extra = None if diag_extras is None else diag_extras[0]
    x = jnp.zeros_like(b)
    for _ in range(cycles):
        residual = b - _matvec(fine, coeffs[0], x, fine_extra)
        x = x + v_cycle(
            hierarchy,
            coeffs,
            diagonals,
            residual,
            pre_sweeps=pre_sweeps,
            post_sweeps=post_sweeps,
            coarse_sweeps=coarse_sweeps,
            omega=omega,
            diag_extras=diag_extras,
        )
    return x


# --- smoothed aggregation ---------------------------------------------------------------
#
# Unsmoothed (piecewise-constant) aggregation above is correct but weak (V-cycle contraction ~0.97):
# the coarse space cannot represent smooth error. Smoothed aggregation fixes this by smoothing the
# tentative prolongation, ``P = (I - omega D^-1 A) P_tent`` -- which makes the coarse operator denser
# (no longer a graph Laplacian), so each level is a **general sparse operator** ``(row, col, val)`` and
# the Galerkin coarse operator ``A_c = P^T A P`` is a genuine sparse triple product. That product is
# nonlinear in the coefficients, so the hierarchy is built **once, off the jit path** with
# ``scipy.sparse`` from a reference coefficient field (the standard "AMG setup once, reuse across
# nonlinear iterates" practice) and then applied as a frozen matrix-free V-cycle under jit.
#
# Three ingredients make the V-cycle **mesh-independent** (~0.25 contraction, flat over 256->9216
# cells), where a naive version degrades toward 1: (i) a **direct coarse solve** (dense pseudo-inverse,
# the dominant fix -- an inexact bottom solve leaves the smoothest error in and compounds with depth);
# (ii) a **Chebyshev polynomial smoother** (a far stronger, still matrix-free/linear smoother than
# damped Jacobi); (iii) **pin decoupling** -- the closed-domain pressure pin is zeroed out of the
# operator (SPD singleton) so the AMG null space matches the pinned outer Jacobian, rather than being
# patched post-hoc (which fights the constant-preserving smoothed prolongation).


class _SparseLevel(NamedTuple):
    """One smoothed-aggregation level: a general sparse operator + its prolongation, all frozen."""

    n: int  # cells at this level (static)
    row: jnp.ndarray  # (nnz,) COO row of the level operator A
    col: jnp.ndarray  # (nnz,) COO col
    val: jnp.ndarray  # (nnz,) COO value
    diagonal: jnp.ndarray  # (n,) diagonal of A
    lam_max: float  # largest eigenvalue of D^-1 A, for the Chebyshev smoother (static)
    coarse_inv: jnp.ndarray | None  # dense pseudo-inverse (coarsest level only); None otherwise
    p_frow: jnp.ndarray | None  # (pnnz,) prolongation fine row (this level); None on coarsest
    p_ccol: jnp.ndarray | None  # (pnnz,) prolongation coarse col (next level)
    p_val: jnp.ndarray | None  # (pnnz,) prolongation value
    n_coarse: int  # next-coarser cell count (static; 0 on coarsest)


class SmoothedHierarchy(NamedTuple):
    """A built smoothed-aggregation hierarchy: general-sparse levels, finest to coarsest."""

    levels: tuple[_SparseLevel, ...]


def _laplacian_csr(owner: np.ndarray, nb: np.ndarray, coeff: np.ndarray, n: int) -> sp.csr_matrix:
    """Symmetric graph Laplacian ``(row, col)`` with edge coefficients, as a scipy compressed-sparse-row (CSR) matrix."""
    o, m, c = np.asarray(owner), np.asarray(nb), np.asarray(coeff)
    rows = np.concatenate([o, m, o, m])
    cols = np.concatenate([m, o, o, m])
    vals = np.concatenate([-c, -c, c, c])  # off-diagonal -c; diagonal accumulates +c
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def _sparse_level(
    a: sp.csr_matrix,
    lam_max: float,
    coarse_inv: np.ndarray | None,
    prolongation: sp.coo_matrix | None,
    n_coarse: int,
) -> _SparseLevel:
    """Freeze a scipy sparse operator (+ optional prolongation / coarse inverse) into JAX arrays."""
    a_coo = a.tocoo()
    p_frow = p_ccol = p_val = None
    if prolongation is not None:
        p_frow = jnp.asarray(prolongation.row)
        p_ccol = jnp.asarray(prolongation.col)
        p_val = jnp.asarray(prolongation.data)
    return _SparseLevel(
        n=a.shape[0],
        row=jnp.asarray(a_coo.row),
        col=jnp.asarray(a_coo.col),
        val=jnp.asarray(a_coo.data),
        diagonal=jnp.asarray(a.diagonal()),
        lam_max=float(lam_max),
        coarse_inv=None if coarse_inv is None else jnp.asarray(coarse_inv),
        p_frow=p_frow,
        p_ccol=p_ccol,
        p_val=p_val,
        n_coarse=n_coarse,
    )


def _spectral_radius(matrix: sp.spmatrix, iterations: int = 20) -> float:
    """Estimate the largest eigenvalue magnitude of a sparse matrix by power iteration (off-jit)."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal(matrix.shape[0])
    v /= np.linalg.norm(v)
    lam = 1.0
    for _ in range(iterations):
        w = matrix @ v
        lam = float(np.linalg.norm(w))
        if lam == 0.0:
            return 1.0
        v = w / lam
    return lam


def build_smoothed_hierarchy(
    owner: np.ndarray,
    nb: np.ndarray,
    coeff: np.ndarray,
    n: int,
    pin: int | None = None,
    boundary_diagonal: np.ndarray | None = None,
    *,
    omega_smooth: float = 2.0 / 3.0,
    max_coarse: int = 16,
    max_levels: int = 20,
) -> SmoothedHierarchy:
    """Build the smoothed-aggregation hierarchy — call once, off the jit path (uses ``scipy.sparse``).

    Parameters
    ----------
    owner, nb : np.ndarray
        Fine-level graph edges (interior faces), shape ``(n_edges,)`` each.
    coeff : np.ndarray
        Reference fine edge coefficients ``c_f``, shape ``(n_edges,)`` (e.g. from the viscous ``a_P``).
    n : int
        Number of fine cells.
    pin : int, optional
        Pinned cell index (closed-domain regularization). The pinned DOF is **decoupled** (its row and
        column are zeroed, unit diagonal) so the operator is SPD-nonsingular and the pin becomes a
        singleton aggregate — the null space matches the pinned outer Jacobian, and no post-hoc pin
        handling is needed in the V-cycle.
    boundary_diagonal : np.ndarray, optional
        Extra diagonal ``(n_cells,)`` added to the fine operator (e.g. Dirichlet boundary-face viscous
        coefficients for a velocity block), making an otherwise-singular Laplacian SPD-nonsingular. Not
        combined with ``pin`` (a Dirichlet-boundary operator needs no pin).
    omega_smooth : float
        Prolongation-smoothing damping factor; the applied damping is ``omega_smooth * 2 / lambda_max``
        (i.e. ``4/(3 lambda_max)`` at the default ``2/3``), with ``lambda_max`` estimated per level.
    max_coarse : int
        Stop coarsening once a level has at most this many cells (solved directly there).
    max_levels : int
        Hard cap on the number of levels.

    Returns
    -------
    SmoothedHierarchy
        Frozen finest-to-coarsest general-sparse levels for :func:`smoothed_multigrid_solve`.
    """
    a = _laplacian_csr(owner, nb, coeff, n)
    if boundary_diagonal is not None:  # Dirichlet boundary stiffness -> SPD-nonsingular (no pin)
        a = a + sp.diags(np.asarray(boundary_diagonal))
    if pin is not None:  # decouple the pinned DOF: zero its row/col, unit diagonal -> SPD singleton
        a = a.tolil()
        a[pin, :] = 0
        a[:, pin] = 0
        a[pin, pin] = 1.0
        a = a.tocsr()
    levels: list[_SparseLevel] = []
    while True:
        d_inv = sp.diags(1.0 / a.diagonal())
        lam_max = _spectral_radius(d_inv @ a)  # for the Chebyshev smoother and prolongation damping
        if a.shape[0] <= max_coarse or len(levels) + 1 >= max_levels:
            # Coarsest level: a direct (dense pseudo-inverse) solve — an inexact coarse solve is the
            # dominant cause of mesh-dependent V-cycle degradation, so it must be an actual solve.
            levels.append(_sparse_level(a, lam_max, np.linalg.pinv(a.toarray()), None, 0))
            break
        upper = sp.triu(a, k=1).tocoo()  # aggregate on this level's graph
        aggregate, n_coarse = _aggregate(upper.row, upper.col, a.shape[0])
        tentative = sp.csr_matrix(
            (np.ones(a.shape[0]), (np.arange(a.shape[0]), aggregate)), shape=(a.shape[0], n_coarse)
        )
        prolongation = (
            tentative - (omega_smooth * 2.0 / lam_max) * (d_inv @ (a @ tentative))
        ).tocsr()
        levels.append(_sparse_level(a, lam_max, None, prolongation.tocoo(), n_coarse))
        a = (prolongation.T @ a @ prolongation).tocsr()  # Galerkin coarse operator
    return SmoothedHierarchy(tuple(levels))


def _sparse_apply(level: _SparseLevel, x: jnp.ndarray) -> jnp.ndarray:
    """General sparse matvec ``A x`` (``segment_sum`` over the frozen COO operator)."""
    return segment_sum(level.val * x[level.col], level.row, level.n)


def _chebyshev_smooth(
    level: _SparseLevel, b: jnp.ndarray, x: jnp.ndarray, degree: int, lo_frac: float
) -> jnp.ndarray:
    """Chebyshev polynomial smoother of ``degree`` on ``[lo_frac, 1.05] * lambda_max`` (of ``D^-1 A``).

    Matrix-free (only ``A``-matvecs and the diagonal), a fixed *linear* operator, and a far stronger
    smoother than the same number of damped-Jacobi sweeps — the fix for the weak-smoother half of the
    V-cycle degradation. Reuses the per-level ``lambda_max`` estimated at build time.
    """
    lo, hi = level.lam_max * lo_frac, level.lam_max * 1.05
    centre, half_width = 0.5 * (hi + lo), 0.5 * (hi - lo)
    inv_diagonal = 1.0 / level.diagonal
    residual = b - _sparse_apply(level, x)
    direction = jnp.zeros_like(x)
    alpha = 0.0
    for i in range(degree):
        preconditioned = inv_diagonal * residual
        if i == 0:
            direction, alpha = preconditioned, 2.0 / centre
        else:
            beta = (half_width * alpha / 2.0) ** 2
            alpha = 1.0 / (centre - beta / alpha)
            direction = preconditioned + beta * direction
        x = x + alpha * direction
        residual = b - _sparse_apply(level, x)
    return x


_Smoother = Callable[["_SparseLevel", jnp.ndarray, jnp.ndarray], jnp.ndarray]


def _smoothed_v_cycle(
    hierarchy: SmoothedHierarchy, b: jnp.ndarray, level_index: int, smoother: _Smoother
) -> jnp.ndarray:
    """One V-cycle on a frozen sparse-operator hierarchy (recursion unrolled at trace time).

    The pre/post ``smoother`` is injected — a symmetric-part-agnostic seam so the symmetric
    (Chebyshev) and nonsymmetric convection-diffusion (damped-Jacobi) paths share this one recursion,
    the prolongation/restriction, and the direct coarse solve. ``smoother(level, b, x) -> x`` applies a
    fixed, matrix-free relaxation for ``A x = b`` starting from ``x`` on that level.
    """
    level = hierarchy.levels[level_index]
    if level.coarse_inv is not None:  # coarsest: an actual (dense pseudo-inverse) solve
        return level.coarse_inv @ b

    x = smoother(level, b, jnp.zeros_like(b))  # pre-smooth
    residual = b - _sparse_apply(level, x)
    coarse_residual = segment_sum(
        level.p_val * residual[level.p_frow], level.p_ccol, hierarchy.levels[level_index + 1].n
    )
    coarse_error = _smoothed_v_cycle(hierarchy, coarse_residual, level_index + 1, smoother)
    x = x + segment_sum(level.p_val * coarse_error[level.p_ccol], level.p_frow, level.n)  # prolong
    return smoother(level, b, x)  # post-smooth


def smoothed_multigrid_solve(
    hierarchy: SmoothedHierarchy,
    b: jnp.ndarray,
    *,
    cycles: int = 1,
    degree: int = 3,
    lo_frac: float = 0.25,
) -> jnp.ndarray:
    """A **fixed** number of smoothed-aggregation V-cycles for ``A x = b`` — the mesh-independent,
    constant-linear inner solve for the SIMPLE pressure Schur.

    The hierarchy is frozen (built once off-jit); a fixed cycle count with fixed Chebyshev smoothing
    and a direct coarse solve makes ``b -> x`` a constant linear operator, so it is a valid frozen left
    preconditioner under plain GMRES. On a model Poisson the V-cycle contraction is ~0.25 and roughly
    mesh-independent (256 → 9216 cells).

    Parameters
    ----------
    hierarchy : SmoothedHierarchy
        From :func:`build_smoothed_hierarchy`.
    b : jnp.ndarray
        Right-hand side, shape ``(n_cells,)``.
    cycles : int
        Number of V-cycles (static).
    degree : int
        Chebyshev smoother degree (static; 3 is a good default).
    lo_frac : float
        Lower end of the Chebyshev smoothing interval as a fraction of ``lambda_max`` (static).

    Returns
    -------
    jnp.ndarray
        The approximate solution ``x``, shape ``(n_cells,)``.
    """

    def smoother(level: _SparseLevel, rhs: jnp.ndarray, guess: jnp.ndarray) -> jnp.ndarray:
        return _chebyshev_smooth(level, rhs, guess, degree, lo_frac)

    x = jnp.zeros_like(b)
    for _ in range(cycles):
        residual = b - _sparse_apply(hierarchy.levels[0], x)
        x = x + _smoothed_v_cycle(hierarchy, residual, 0, smoother)
    return x


# --- nonsymmetric (convection-diffusion) smoothed aggregation ---------------------------
#
# The symmetric path above builds its hierarchy on a graph Laplacian, so a point smoother (Chebyshev)
# and constant-preserving aggregation resolve the smooth error. A momentum block with strong convection
# is a different operator: first-order upwind adds a **nonsymmetric** off-diagonal ``max(±mdot, 0)`` to
# the viscous coupling, and its error modes are advected along the flow, not smooth in the Laplacian
# sense — so a Laplacian-only AMG (even rescaled by the convective diagonal) is not Peclet-robust and
# stalls once the cell Peclet number grows.
#
# The convection-aware hierarchy carries the true nonsymmetric operator ``A = viscous + upwind`` up the
# coarse levels: aggregation and the smoothed prolongation use the **symmetric part** ``(A + Aᵀ)/2`` (so
# the coarse space is well-shaped and the prolongation smoothing is stable), while the Galerkin coarse
# operator ``A_c = Pᵀ A P`` is formed from the **true** ``A`` — the standard way to keep the coarse
# operators convection-aware. The upwind convection-diffusion operator is a diagonally dominant M-matrix
# (positive diagonal, non-positive off-diagonals), so a **damped-Jacobi** smoother — matrix-free, a
# fixed linear operator, and safe on the operator's positive-real-part spectrum where a Chebyshev
# interval smoother is not — is the robust choice. Built once off-jit at a frozen reference mass flux
# and applied as a frozen matrix-free V-cycle, exactly like the symmetric path.


def _convection_diffusion_csr(
    owner: np.ndarray,
    nb: np.ndarray,
    visc: np.ndarray,
    mdot: np.ndarray,
    n: int,
    boundary_diagonal: np.ndarray | None = None,
) -> sp.csr_matrix:
    """First-order-upwind convection-diffusion operator ``A`` as a scipy CSR matrix (nonsymmetric).

    Each interior edge ``(owner P, neighbour N)`` carries a symmetric viscous coupling ``visc`` and a
    first-order-upwind convective coupling from the owner-outward face mass flux ``mdot``: an outflow
    (``mdot > 0``) advects the owner value, an inflow the neighbour value. The resulting entries are

        A[P, N] = -(visc + max(-mdot, 0)),   A[N, P] = -(visc + max(mdot, 0)),

    with the matching diagonal contributions ``visc + max(mdot, 0)`` at P and ``visc + max(-mdot, 0)``
    at N — so ``A`` is a diagonally dominant M-matrix, nonsymmetric wherever ``mdot != 0``.
    ``boundary_diagonal`` (per cell) adds the boundary-face stiffness to the diagonal.
    """
    o, m = np.asarray(owner), np.asarray(nb)
    v, f = np.asarray(visc), np.asarray(mdot)
    up_out = np.maximum(f, 0.0)  # outflow leaves the owner: owner value is upwind
    up_in = np.maximum(-f, 0.0)  # inflow enters the owner: neighbour value is upwind
    rows = np.concatenate([o, m, o, m])
    cols = np.concatenate([m, o, o, m])
    vals = np.concatenate([-(v + up_in), -(v + up_out), v + up_out, v + up_in])
    a = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    if boundary_diagonal is not None:
        a = a + sp.diags(np.asarray(boundary_diagonal))
    return a.tocsr()


def build_convection_hierarchy(
    owner: np.ndarray,
    nb: np.ndarray,
    visc: np.ndarray,
    mdot: np.ndarray,
    n: int,
    boundary_diagonal: np.ndarray | None = None,
    *,
    omega_smooth: float = 2.0 / 3.0,
    max_coarse: int = 16,
    max_levels: int = 20,
) -> SmoothedHierarchy:
    """Build the nonsymmetric convection-diffusion aggregation hierarchy — call once, off the jit path.

    The frozen reference operator is ``A = viscous + first-order-upwind convection`` (see
    :func:`_convection_diffusion_csr`); the symmetric part drives aggregation and prolongation smoothing
    while the Galerkin coarse operators keep the true nonsymmetric ``A``.

    Parameters
    ----------
    owner, nb : np.ndarray
        Fine-level interior-face edges, shape ``(n_edges,)`` each.
    visc : np.ndarray
        Per-edge viscous coefficient ``mu_f A_f / (d.n)_f``, shape ``(n_edges,)``.
    mdot : np.ndarray
        Reference owner-outward face mass flux on the interior edges, shape ``(n_edges,)`` — the frozen
        convective linearisation the hierarchy is built at.
    n : int
        Number of fine cells.
    boundary_diagonal : np.ndarray, optional
        Extra diagonal ``(n_cells,)`` from boundary faces (Dirichlet velocity walls/inlet + outlet
        convection), making the operator diagonally dominant and nonsingular.
    omega_smooth : float
        Prolongation-smoothing damping factor; the applied damping is ``omega_smooth * 2 / lambda_max``
        (``lambda_max`` of the symmetric part per level).
    max_coarse : int
        Stop coarsening once a level has at most this many cells (solved directly there).
    max_levels : int
        Hard cap on the number of levels.

    Returns
    -------
    SmoothedHierarchy
        Frozen finest-to-coarsest general-sparse levels for :func:`convection_multigrid_solve`.
    """
    a = _convection_diffusion_csr(owner, nb, visc, mdot, n, boundary_diagonal)
    levels: list[_SparseLevel] = []
    while True:
        a_sym = (0.5 * (a + a.T)).tocsr()  # aggregation + prolongation use the symmetric part
        d_inv = sp.diags(1.0 / a.diagonal())
        # Two spectral estimates with different roles: the symmetric part's real ``lambda_max`` sets the
        # constant-preserving prolongation smoothing; the **true** (nonsymmetric) operator's spectral
        # radius sets the damped-Jacobi relaxation. The Galerkin coarse operators pick up large complex
        # eigenvalues the symmetric part misses, so damping by the symmetric estimate alone makes the
        # coarse-level smoother diverge — the smoother must see the true spectral radius.
        lam_sym = _spectral_radius(d_inv @ a_sym)
        lam_true = _spectral_radius(d_inv @ a)
        if a.shape[0] <= max_coarse or len(levels) + 1 >= max_levels:
            # Coarsest: a direct (dense pseudo-inverse) solve — pinv handles the nonsymmetric operator.
            levels.append(_sparse_level(a, lam_true, np.linalg.pinv(a.toarray()), None, 0))
            break
        upper = sp.triu(a_sym, k=1).tocoo()  # aggregate on the symmetric graph
        aggregate, n_coarse = _aggregate(upper.row, upper.col, a.shape[0])
        tentative = sp.csr_matrix(
            (np.ones(a.shape[0]), (np.arange(a.shape[0]), aggregate)), shape=(a.shape[0], n_coarse)
        )
        prolongation = (
            tentative - (omega_smooth * 2.0 / lam_sym) * (d_inv @ (a_sym @ tentative))
        ).tocsr()
        levels.append(_sparse_level(a, lam_true, None, prolongation.tocoo(), n_coarse))
        a = (prolongation.T @ a @ prolongation).tocsr()  # Galerkin coarse operator from the true A
    return SmoothedHierarchy(tuple(levels))


def _jacobi_smooth(
    level: _SparseLevel, b: jnp.ndarray, x: jnp.ndarray, sweeps: int, omega: float
) -> jnp.ndarray:
    """Damped-Jacobi smoother ``x <- x + (omega / lambda_max) D^-1 (b - A x)`` (``sweeps`` times).

    Matrix-free and a fixed linear operator. The relaxation is scaled by the per-level ``lambda_max``
    (of the symmetric part) so ``omega`` in ``(0, 1]`` is a mesh- and scale-independent damping — the
    high-frequency-smoothing choice for the M-matrix convection-diffusion operator, where a Chebyshev
    interval smoother (assuming a real spectrum) is not safe.
    """
    alpha = omega / level.lam_max
    inv_diagonal = 1.0 / level.diagonal
    for _ in range(sweeps):
        x = x + alpha * inv_diagonal * (b - _sparse_apply(level, x))
    return x


def convection_multigrid_solve(
    hierarchy: SmoothedHierarchy,
    b: jnp.ndarray,
    *,
    cycles: int = 1,
    sweeps: int = 2,
    omega: float = 0.8,
) -> jnp.ndarray:
    """A **fixed** number of convection-diffusion V-cycles for ``A x = b`` — the Peclet-robust,
    constant-linear inner solve for the momentum (velocity) block.

    The hierarchy is frozen (built once off-jit at a reference mass flux); a fixed cycle count with a
    fixed damped-Jacobi smoother and a direct coarse solve makes ``b -> x`` a constant linear operator,
    so it is a valid frozen left preconditioner under plain GMRES and transposes cleanly for the adjoint.

    Parameters
    ----------
    hierarchy : SmoothedHierarchy
        From :func:`build_convection_hierarchy`.
    b : jnp.ndarray
        Right-hand side, shape ``(n_cells,)``.
    cycles : int
        Number of V-cycles (static).
    sweeps : int
        Damped-Jacobi pre/post sweeps per level (static).
    omega : float
        Jacobi damping factor in ``(0, 1]`` (static).

    Returns
    -------
    jnp.ndarray
        The approximate solution ``x``, shape ``(n_cells,)``.
    """

    def smoother(level: _SparseLevel, rhs: jnp.ndarray, guess: jnp.ndarray) -> jnp.ndarray:
        return _jacobi_smooth(level, rhs, guess, sweeps, omega)

    x = jnp.zeros_like(b)
    for _ in range(cycles):
        residual = b - _sparse_apply(hierarchy.levels[0], x)
        x = x + _smoothed_v_cycle(hierarchy, residual, 0, smoother)
    return x


# --- local approximate ideal restriction (lAIR) -----------------------------------------
#
# Aggregation multigrid (symmetric or convection-diffusion above) coarsens by grouping cells and, for
# strong convection, its deep Galerkin recursion is not stable: the coarse operators lose the flow
# structure and the coarse correction amplifies error. Reduction-based AMG takes the opposite view. A
# coarse/fine (C/F) splitting partitions the unknowns; with ``A = [[A_ff, A_fc], [A_cf, A_cc]]`` the
# *ideal* restriction ``R = [-A_cf A_ff⁻¹, I]`` makes the coarse operator the exact Schur complement, so
# eliminating the F-points reproduces the fine operator's coarse action. For a convection-dominated
# operator (nearly triangular in the flow ordering) that elimination is nearly exact, so a few V-cycles
# behave almost like a direct solve — and the recursion is Peclet-robust and mesh-independent where
# aggregation is not (Manteuffel, Ruge & Southworth, SISC 2018; Southworth et al.).
#
# lAIR (local AIR) approximates ``A_cf A_ff⁻¹`` by a **local** solve per C-point: over the F-neighbours
# within a few steps, solve ``A_ff[N,N]^T z = -A[g, N]^T`` for the restriction weights. Interpolation is
# the cheap ``one-point`` rule (each F-point takes its strongest C-neighbour); the smoother is FC-Jacobi
# (a few F-point sweeps then a C-point sweep) — the F-relaxation is what makes it work for advection. The
# whole setup is integer/sparse graph work done once off the jit path in scipy/numpy; the apply is
# frozen ``segment_sum`` matvecs over ``R`` / ``P`` / ``A_c`` and a masked FC-Jacobi, and transposes for
# the adjoint (``R != Pᵀ`` is handled by the transpose of the linear apply).


class _AirLevel(NamedTuple):
    """One lAIR level: the operator, its restriction and prolongation, and the C/F masks — all frozen.

    Unlike :class:`_SparseLevel` (which stores one prolongation and takes ``R = Pᵀ``), a reduction-based
    level carries an **independent** restriction ``R`` (fine → coarse) and prolongation ``P`` (coarse →
    fine), plus the fine/coarse masks the FC-Jacobi smoother relaxes over.
    """

    n: int  # cells at this level (static)
    row: jnp.ndarray  # (nnz,) COO row of the level operator A
    col: jnp.ndarray  # (nnz,) COO col
    val: jnp.ndarray  # (nnz,) COO value
    diagonal: jnp.ndarray  # (n,) diagonal of A
    f_mask: jnp.ndarray  # (n,) 1.0 on fine points, else 0.0
    c_mask: jnp.ndarray  # (n,) 1.0 on coarse points, else 0.0
    r_row: jnp.ndarray | None  # (rnnz,) restriction COO coarse row; None on coarsest
    r_col: jnp.ndarray | None  # (rnnz,) restriction COO fine col
    r_val: jnp.ndarray | None  # (rnnz,) restriction value
    p_row: jnp.ndarray | None  # (pnnz,) prolongation COO fine row; None on coarsest
    p_col: jnp.ndarray | None  # (pnnz,) prolongation COO coarse col
    p_val: jnp.ndarray | None  # (pnnz,) prolongation value
    coarse_inv: jnp.ndarray | None  # dense pseudo-inverse (coarsest level only); None otherwise
    n_coarse: int  # next-coarser cell count (static; 0 on coarsest)


class AirHierarchy(NamedTuple):
    """A built lAIR hierarchy: reduction-based levels, finest to coarsest."""

    levels: tuple[_AirLevel, ...]


def _strength_classical(a: sp.csr_matrix, theta: float) -> sp.csr_matrix:
    """Classical strength graph ``S``: ``S[i,j]=1`` iff ``|A_ij| >= theta · max_{k!=i}|A_ik|``.

    Row ``i`` marks the connections cell ``i`` *depends on strongly* — for an upwind operator these are
    the flow-aligned couplings that must be honoured by the coarsening and the restriction.
    """
    a = a.tocsr()
    n = a.shape[0]
    abs_a = a.copy()
    abs_a.data = np.abs(abs_a.data)
    rows: list[int] = []
    cols: list[int] = []
    indptr, indices, data = abs_a.indptr, abs_a.indices, abs_a.data
    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        ci, vi = indices[s:e], data[s:e]
        off = ci != i
        if not off.any():
            continue
        m = vi[off].max()
        if m == 0.0:
            continue
        strong = ci[off][vi[off] >= theta * m]
        rows.extend([i] * len(strong))
        cols.extend(strong.tolist())
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def _rs_split(strength: sp.csr_matrix) -> np.ndarray:
    """Ruge--Stueben first-pass C/F splitting (greedy, influence-weighted). Returns 1 = C, 0 = F.

    Repeatedly makes the highest-influence undecided point coarse (a point's influence is how many
    others depend strongly on it), marks its dependents fine, and boosts the influence of what a new
    fine point depends on — so coarse points cover the strong connections. A max-heap keeps it
    ``O(nnz log n)``.
    """
    strength = strength.tocsr()
    n = strength.shape[0]
    dependents = strength.T.tocsr()  # dependents[i] = points that depend on i (its influence set)
    influence = np.asarray(strength.sum(axis=0)).ravel().astype(float)
    split = np.full(n, -1, dtype=np.int64)
    heap = [(-influence[i], i) for i in range(n)]
    heapq.heapify(heap)
    while heap:
        neg, i = heapq.heappop(heap)
        if split[i] != -1 or -neg != influence[i]:
            continue  # stale heap entry (influence was bumped since this was pushed)
        split[i] = 1  # coarse
        for j in dependents.indices[dependents.indptr[i] : dependents.indptr[i + 1]]:
            if split[j] == -1:
                split[j] = 0  # a dependent of a coarse point becomes fine
                row = strength.indices[strength.indptr[j] : strength.indptr[j + 1]]
                for k in row:  # boost the influence of what this fine point depends on
                    if split[k] == -1:
                        influence[k] += 1.0
                        heapq.heappush(heap, (-influence[k], k))
    split[split == -1] = 1  # any leftovers -> coarse (a safe singleton)
    return split


def _one_point_interpolation(a: sp.csr_matrix, split: np.ndarray) -> sp.csr_matrix:
    """One-point interpolation ``P``: each F-point takes its strongest C-neighbour; C-points injected."""
    a = a.tocsr()
    n = a.shape[0]
    coarse = np.where(split == 1)[0]
    coarse_index = -np.ones(n, dtype=np.int64)
    coarse_index[coarse] = np.arange(len(coarse))
    abs_a = a.copy()
    abs_a.data = np.abs(abs_a.data)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i in range(n):
        if split[i] == 1:
            rows.append(i)
            cols.append(int(coarse_index[i]))
            vals.append(1.0)
            continue
        s, e = abs_a.indptr[i], abs_a.indptr[i + 1]
        ci, vi = abs_a.indices[s:e], abs_a.data[s:e]
        c_neighbour = (split[ci] == 1) & (ci != i)
        if c_neighbour.any():
            j = ci[c_neighbour][np.argmax(vi[c_neighbour])]
            rows.append(i)
            cols.append(int(coarse_index[j]))
            vals.append(1.0)  # an F-point with no C-neighbour interpolates nothing (zero row)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, len(coarse)))


def _lair_restriction(a: sp.csr_matrix, split: np.ndarray, degree: int) -> sp.csr_matrix:
    """lAIR restriction ``R``: per C-point, a local approximate-ideal solve over its F-neighbourhood.

    The ideal restriction row for coarse point ``g`` solves ``R_g A_ff = -A[g, F]``; localised to the
    F-points ``N`` within ``degree`` steps of ``g`` this is the small dense solve ``A_ff[N,N]^T z =
    -A[g, N]^T``, with the identity entry ``R[g, g] = 1``.
    """
    a = a.tocsr()
    n = a.shape[0]
    coarse = np.where(split == 1)[0]
    coarse_index = -np.ones(n, dtype=np.int64)
    coarse_index[coarse] = np.arange(len(coarse))
    fine = split == 0
    indptr, indices = a.indptr, a.indices
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for g in coarse:
        ci = int(coarse_index[g])
        rows.append(ci)
        cols.append(int(g))
        vals.append(1.0)  # identity on the C-point itself
        neighbourhood: set[int] = set()
        frontier = {int(g)}
        for _ in range(degree):  # F-points within `degree` steps of g
            nxt: set[int] = set()
            for u in frontier:
                for v in indices[indptr[u] : indptr[u + 1]]:
                    v = int(v)
                    if fine[v] and v not in neighbourhood:
                        neighbourhood.add(v)
                        nxt.add(v)
            frontier = nxt
        if not neighbourhood:
            continue
        nb = np.array(sorted(neighbourhood))
        a_ff = a[np.ix_(nb, nb)].toarray()
        rhs = np.asarray(a[g, nb].todense()).ravel()
        try:
            z = np.linalg.solve(a_ff.T, -rhs)
        except np.linalg.LinAlgError:
            z = np.linalg.lstsq(a_ff.T, -rhs, rcond=None)[0]
        rows.extend([ci] * len(nb))
        cols.extend(nb.tolist())
        vals.extend(z.tolist())
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(coarse), n))


def _air_level(a: sp.csr_matrix, split: np.ndarray, restriction, prolongation) -> _AirLevel:
    """Freeze a scipy operator, its C/F masks, and (optional) restriction/prolongation into JAX arrays."""
    a_coo = a.tocoo()
    coarsest = restriction is None
    r = None if coarsest else restriction.tocoo()
    p = None if coarsest else prolongation.tocoo()
    return _AirLevel(
        n=a.shape[0],
        row=jnp.asarray(a_coo.row),
        col=jnp.asarray(a_coo.col),
        val=jnp.asarray(a_coo.data),
        diagonal=jnp.asarray(a.diagonal()),
        f_mask=jnp.asarray((split == 0).astype(np.float64)),
        c_mask=jnp.asarray((split == 1).astype(np.float64)),
        r_row=None if coarsest else jnp.asarray(r.row),
        r_col=None if coarsest else jnp.asarray(r.col),
        r_val=None if coarsest else jnp.asarray(r.data),
        p_row=None if coarsest else jnp.asarray(p.row),
        p_col=None if coarsest else jnp.asarray(p.col),
        p_val=None if coarsest else jnp.asarray(p.data),
        coarse_inv=jnp.asarray(np.linalg.pinv(a.toarray())) if coarsest else None,
        n_coarse=0 if coarsest else prolongation.shape[1],
    )


def build_air_hierarchy(
    a: sp.csr_matrix,
    *,
    theta: float = 0.25,
    degree: int = 2,
    max_coarse: int = 20,
    max_levels: int = 20,
) -> AirHierarchy:
    """Build the lAIR hierarchy — call once, off the jit path (uses ``scipy.sparse`` / ``numpy``).

    Parameters
    ----------
    a : scipy.sparse matrix
        The (nonsymmetric) fine operator, e.g. a frozen convection-diffusion momentum block.
    theta : float
        Classical strength-of-connection threshold in ``(0, 1)`` for the C/F splitting.
    degree : int
        The F-neighbourhood radius (in graph steps) of the local approximate-ideal restriction solves.
    max_coarse : int
        Stop coarsening once a level has at most this many cells (solved directly there).
    max_levels : int
        Hard cap on the number of levels.

    Returns
    -------
    AirHierarchy
        Frozen finest-to-coarsest reduction-based levels for :func:`air_multigrid_solve`.
    """
    a = a.tocsr()
    levels: list[_AirLevel] = []
    while True:
        n = a.shape[0]
        if n <= max_coarse or len(levels) + 1 >= max_levels:
            levels.append(_air_level(a, np.ones(n, dtype=np.int64), None, None))
            break
        split = _rs_split(_strength_classical(a, theta))
        n_coarse = int((split == 1).sum())
        if n_coarse == 0 or n_coarse == n:  # degenerate coarsening -> solve here
            levels.append(_air_level(a, np.ones(n, dtype=np.int64), None, None))
            break
        prolongation = _one_point_interpolation(a, split)
        restriction = _lair_restriction(a, split, degree)
        levels.append(_air_level(a, split, restriction, prolongation))
        a = (restriction @ a @ prolongation).tocsr()  # Galerkin coarse operator R A P
    return AirHierarchy(tuple(levels))


def build_convection_air_hierarchy(
    owner: np.ndarray,
    nb: np.ndarray,
    visc: np.ndarray,
    mdot: np.ndarray,
    n: int,
    boundary_diagonal: np.ndarray | None = None,
    *,
    theta: float = 0.25,
    degree: int = 2,
    max_coarse: int = 20,
    max_levels: int = 20,
) -> AirHierarchy:
    """Build the lAIR hierarchy for a first-order-upwind convection-diffusion operator — off-jit.

    Assembles the frozen ``viscous + upwind`` momentum operator (:func:`_convection_diffusion_csr`, the
    same one :func:`build_convection_hierarchy` uses) and coarsens it by reduction (:func:`build_air_hierarchy`).
    The arguments match :func:`build_convection_hierarchy`; see :func:`build_air_hierarchy` for the lAIR
    parameters.
    """
    a = _convection_diffusion_csr(owner, nb, visc, mdot, n, boundary_diagonal)
    return build_air_hierarchy(
        a, theta=theta, degree=degree, max_coarse=max_coarse, max_levels=max_levels
    )


def _coo_apply(row, col, val, x: jnp.ndarray, n_out: int) -> jnp.ndarray:
    """General sparse matvec ``M x`` for a COO operator with ``n_out`` output rows."""
    return segment_sum(val * x[col], row, n_out)


def _fc_jacobi(
    level: _AirLevel, b: jnp.ndarray, x: jnp.ndarray, f_iters: int, c_iters: int, omega: float
) -> jnp.ndarray:
    """FC-Jacobi smoother: ``f_iters`` F-point damped-Jacobi sweeps then ``c_iters`` C-point sweeps.

    Each sweep relaxes only the fine (or coarse) block via the mask, matrix-free and a fixed linear
    operator. The F-relaxation is the reduction-based smoother that suppresses the F-point error the
    ideal restriction is built to eliminate.
    """
    inv_diagonal = 1.0 / level.diagonal
    for _ in range(f_iters):
        x = x + omega * level.f_mask * inv_diagonal * (b - _sparse_apply(level, x))
    for _ in range(c_iters):
        x = x + omega * level.c_mask * inv_diagonal * (b - _sparse_apply(level, x))
    return x


def _air_v_cycle(
    hierarchy: AirHierarchy,
    b: jnp.ndarray,
    level_index: int,
    f_iters: int,
    c_iters: int,
    omega: float,
) -> jnp.ndarray:
    """One reduction-based V-cycle (recursion unrolled at trace time), pre/post FC-Jacobi smoothed."""
    level = hierarchy.levels[level_index]
    if level.coarse_inv is not None:  # coarsest: a direct (dense pseudo-inverse) solve
        return level.coarse_inv @ b

    x = _fc_jacobi(level, b, jnp.zeros_like(b), f_iters, c_iters, omega)  # pre-smooth
    residual = b - _sparse_apply(level, x)
    coarse_residual = _coo_apply(level.r_row, level.r_col, level.r_val, residual, level.n_coarse)
    coarse_error = _air_v_cycle(
        hierarchy, coarse_residual, level_index + 1, f_iters, c_iters, omega
    )
    x = x + _coo_apply(level.p_row, level.p_col, level.p_val, coarse_error, level.n)  # prolong
    return _fc_jacobi(level, b, x, f_iters, c_iters, omega)  # post-smooth


def air_multigrid_solve(
    hierarchy: AirHierarchy,
    b: jnp.ndarray,
    *,
    cycles: int = 1,
    f_iters: int = 2,
    c_iters: int = 1,
    omega: float = 1.0,
) -> jnp.ndarray:
    """A **fixed** number of lAIR V-cycles for ``A x = b`` — the Peclet-robust, mesh-independent inner
    solve for a convection-dominated (velocity) block.

    The hierarchy is frozen (built once off-jit); a fixed cycle count with fixed FC-Jacobi smoothing
    and a direct coarse solve makes ``b -> x`` a constant linear operator, so it is a valid frozen left
    preconditioner under plain GMRES and transposes cleanly for the adjoint.

    Parameters
    ----------
    hierarchy : AirHierarchy
        From :func:`build_air_hierarchy`.
    b : jnp.ndarray
        Right-hand side, shape ``(n_cells,)``.
    cycles : int
        Number of V-cycles (static).
    f_iters, c_iters : int
        Fine- and coarse-point Jacobi sweeps per smoother application (static).
    omega : float
        Jacobi damping factor (static).

    Returns
    -------
    jnp.ndarray
        The approximate solution ``x``, shape ``(n_cells,)``.
    """
    x = jnp.zeros_like(b)
    for _ in range(cycles):
        residual = b - _sparse_apply(hierarchy.levels[0], x)
        x = x + _air_v_cycle(hierarchy, residual, 0, f_iters, c_iters, omega)
    return x
