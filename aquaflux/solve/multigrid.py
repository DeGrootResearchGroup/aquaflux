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

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.ops import segment_sum


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


def _aggregate(owner: np.ndarray, nb: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    """Two-pass aggregation (Vaněk et al.): seed clean aggregates, then attach leftovers.

    Pass 1 forms an aggregate ``{i} ∪ neighbours(i)`` only from a cell ``i`` whose neighbours are all
    still free — giving well-shaped, ~stencil-sized aggregates. Pass 2 attaches each remaining cell to
    an adjacent existing aggregate (rare orphans seed their own). This yields a healthy coarsening
    ratio (~4× in 2D) with no singletons, which a naive one-pass greedy does not.
    """
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for o, m in zip(owner.tolist(), nb.tolist(), strict=True):
        adjacency[o].append(m)
        adjacency[m].append(o)
    aggregate = np.full(n, -1, dtype=np.int64)
    count = 0
    for i in range(n):  # pass 1: seed from cells in a fully-free neighbourhood
        if aggregate[i] != -1 or any(aggregate[j] != -1 for j in adjacency[i]):
            continue
        aggregate[i] = count
        for j in adjacency[i]:
            aggregate[j] = count
        count += 1
    for i in range(n):  # pass 2: attach leftovers to an adjacent aggregate (else seed their own)
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


def _matvec(level: _Level, coeff: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    """Graph-Laplacian matvec ``A p`` at a level; pinned row returns ``p[pin]`` (identity)."""
    flux = coeff * (p[level.owner] - p[level.nb])
    result = segment_sum(flux, level.owner, level.n) - segment_sum(flux, level.nb, level.n)
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
) -> jnp.ndarray:
    """A few damped-Jacobi sweeps ``x <- x + omega D^-1 (b - A x)``, holding the pinned cell."""
    inv_diagonal = 1.0 / diagonal
    for _ in range(sweeps):
        x = x + omega * inv_diagonal * (b - _matvec(level, coeff, x))
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
) -> jnp.ndarray:
    """One multigrid V-cycle for ``A x = b`` at ``level_index`` (recursion unrolled at trace time).

    A single fixed V-cycle is a constant linear operator in ``b`` (the smoother sweeps and the coarse
    recursion are all fixed-length), so it is a valid frozen left preconditioner.
    """
    level = hierarchy.levels[level_index]
    coeff, diagonal = coeffs[level_index], diagonals[level_index]
    if level.agg is None:  # coarsest level: smooth to (near-)solve the small system
        return _smooth(level, coeff, diagonal, b, jnp.zeros_like(b), coarse_sweeps, omega)

    next_level = hierarchy.levels[level_index + 1]
    x = _smooth(level, coeff, diagonal, b, jnp.zeros_like(b), pre_sweeps, omega)
    residual = b - _matvec(level, coeff, x)
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
    )
    x = x + coarse_error[level.agg]  # prolong (piecewise-constant) and correct
    if level.pin >= 0:
        x = x.at[level.pin].set(b[level.pin])
    return _smooth(level, coeff, diagonal, b, x, post_sweeps, omega)


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

    Returns
    -------
    jnp.ndarray
        The approximate solution ``x``, shape ``(n_cells,)``.
    """
    fine = hierarchy.levels[0]
    x = jnp.zeros_like(b)
    for _ in range(cycles):
        residual = b - _matvec(fine, coeffs[0], x)
        x = x + v_cycle(
            hierarchy,
            coeffs,
            diagonals,
            residual,
            pre_sweeps=pre_sweeps,
            post_sweeps=post_sweeps,
            coarse_sweeps=coarse_sweeps,
            omega=omega,
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


def _smoothed_v_cycle(
    hierarchy: SmoothedHierarchy, b: jnp.ndarray, level_index: int, degree: int, lo_frac: float
) -> jnp.ndarray:
    """One V-cycle on the frozen smoothed-aggregation hierarchy (recursion unrolled at trace time)."""
    level = hierarchy.levels[level_index]
    if level.coarse_inv is not None:  # coarsest: an actual (dense pseudo-inverse) solve
        return level.coarse_inv @ b

    x = _chebyshev_smooth(level, b, jnp.zeros_like(b), degree, lo_frac)  # pre-smooth
    residual = b - _sparse_apply(level, x)
    coarse_residual = segment_sum(
        level.p_val * residual[level.p_frow], level.p_ccol, hierarchy.levels[level_index + 1].n
    )
    coarse_error = _smoothed_v_cycle(hierarchy, coarse_residual, level_index + 1, degree, lo_frac)
    x = x + segment_sum(level.p_val * coarse_error[level.p_ccol], level.p_frow, level.n)  # prolong
    return _chebyshev_smooth(level, b, x, degree, lo_frac)  # post-smooth


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
    x = jnp.zeros_like(b)
    for _ in range(cycles):
        residual = b - _sparse_apply(hierarchy.levels[0], x)
        x = x + _smoothed_v_cycle(hierarchy, residual, 0, degree, lo_frac)
    return x
