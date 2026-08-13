"""Matrix-free algebraic multigrid for the preconditioner's inner solves.

A pressure-Poisson-like operator ``A`` on an unstructured mesh needs ``O(h^-2)`` unpreconditioned
Krylov iterations; a multigrid V-cycle makes the iteration count **mesh-independent**, which is what
a scalable inner solve for the SIMPLE pressure Schur (and the convection-dominated velocity block)
needs at large mesh sizes.

Design for a differentiable JAX/GPU pipeline — every hierarchy is **built once, off the jit path,
and frozen**, then applied under jit as a fixed matrix-free V-cycle (a constant linear operator in
``b``, so plain left-preconditioned GMRES suffices and the adjoint transposes cleanly). Each level
carries its operator as a general sparse ``(row, col, val)`` triple and its intergrid transfers as
sparse operators; the one recursion (:func:`_frozen_v_cycle`) applies the shared operator matvec
(:func:`_operator_matvec`) and direct coarse solve, and is specialized per family by the injected
:class:`_VCycleOps` (restriction, prolongation, smoother). The outer fixed-cycle driver every
``*_multigrid_solve`` entry point runs is likewise one function (:func:`_fixed_cycle_solve`), so a
family contributes only its ops:

* **Smoothed aggregation** (:func:`build_smoothed_hierarchy`, :func:`smoothed_multigrid_solve`): the
  symmetric pressure Schur. The tentative piecewise-constant prolongation is smoothed
  ``P = (I - omega D^-1 A) P_tent``, restriction is ``Pᵀ``, the smoother is a Chebyshev polynomial,
  and the coarse level is a direct (dense pseudo-inverse) solve — ~0.25 mesh-independent contraction.
  Its two-level convection variant (:func:`build_convection_hierarchy`,
  :func:`convection_multigrid_solve`) uses a damped-Jacobi smoother for the nonsymmetric momentum
  operator.
* **Reduction — local approximate ideal restriction, lAIR** (:func:`build_air_hierarchy`,
  :func:`air_multigrid_solve`): an **independent** restriction ``R != Pᵀ`` and an FC-Jacobi smoother
  for the strongly convection-dominated velocity block — Peclet-robust and mesh-independent to large
  meshes.

Every builder takes an **assembled operator** ``a`` (a ``scipy.sparse`` matrix) and returns a frozen
hierarchy: this module coarsens operators and knows nothing about meshes, fluxes, or which face value
a scheme upwinds. Callers assemble with :func:`aquaflux.solve.frozen_operator.convection_diffusion_operator`
(and regularize a closed-domain pressure system with
:func:`aquaflux.solve.frozen_operator.decouple_dof` before building, so the AMG null space matches the
pinned outer Jacobian; the pin only affects preconditioner quality, never the converged solution).

The coefficients are frozen at a reference field at build time (the standard "AMG setup once, reuse
across nonlinear iterates" practice), with the per-iterate operator scale restored by a symmetric
diagonal rescaling in the apply.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.experimental.sparse import BCSR
from jax.ops import segment_sum
from scipy.sparse.csgraph import reverse_cuthill_mckee

from .frozen_operator import symmetrically_equilibrate


def _require_positive_diagonal(diagonal: np.ndarray, where: str) -> None:
    """Reject a non-positive or non-finite operator diagonal before it is inverted or frozen.

    A frozen level's diagonal is inverted by the smoother (``D^-1``) and, at build time, by the
    prolongation-smoothing damping. A zero entry — from a disconnected component, an isolated or
    zero-volume cell, or a degenerate nonsymmetric Galerkin (``R A P``) row — makes ``1/0 = inf`` and
    poisons the frozen preconditioner. A symmetric graph Laplacian with positive coefficients (and a
    diagonally dominant convection-diffusion operator) has a strictly positive diagonal, so a
    violation means the operator is degenerate; fail here rather than silently stall the V-cycle. The
    diagonal is checked *after* any boundary stiffness has been folded in, so a cell that is closed
    off from the interior but carries a boundary coefficient is correctly allowed.

    Parameters
    ----------
    diagonal : np.ndarray
        The operator diagonal at one level, shape ``(n_cells,)``.
    where : str
        Caller name (with level index), included in the error message.

    Raises
    ------
    ValueError
        If any entry is non-finite or ``<= 0``.
    """
    diagonal = np.asarray(diagonal)
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise ValueError(
            f"{where}: operator diagonal must be finite and strictly positive, but its minimum is "
            f"{np.nanmin(diagonal):.3e}. A zero/negative diagonal indicates a degenerate operator "
            "(disconnected component, isolated or zero-volume cell, or a degenerate coarse-grid row) "
            "that would bake inf/NaN into the frozen preconditioner."
        )


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


def _cell_graph(a: sp.csr_matrix, block_size: int) -> sp.csr_matrix:
    """Collapse a block operator's degree-of-freedom graph onto its **cell** connectivity.

    Aggregation on the degree-of-freedom graph is field-blind: with several fields per cell it can put
    one field of cell ``i`` and a *different* field of cell ``j`` into the same aggregate. On a strongly
    nonsymmetric multi-field operator that produces a degenerate Galerkin (``Rᵀ A P``) row, so the build
    is then refused for a non-positive coarse diagonal — a failure that reads as though the *fine*
    operator were at fault when the fine operator is clean. Coarsening whole cells removes the cause,
    which is why a nodal aggregation is what a multi-field hierarchy needs.

    The collapse is exact rather than approximate because the layout is **field-major**: degree of
    freedom ``(cell i, field f)`` sits at ``f * n_cells + i``, so ``index % n_cells`` *is* the cell. The
    cell edge weight is the sum of ``|A_ij|`` over the corresponding ``block_size × block_size`` block —
    the standard nodal strength measure, and the reason a cell pair couples if **any** of their fields do.

    Parameters
    ----------
    a : scipy.sparse matrix
        The operator to coarsen, shape ``(block_size * n_cells,) * 2``, field-major. Pass the true
        operator, not its symmetric part: the magnitude is taken per entry here, so symmetrizing
        first would let an antisymmetric coupling cancel itself out of the graph.
    block_size : int
        Fields per cell.

    Returns
    -------
    scipy.sparse.csr_matrix
        The cell graph, shape ``(n_cells, n_cells)``, with non-negative weights.
    """
    n_cells = a.shape[0] // block_size
    entries = sp.coo_matrix(abs(a))
    # `coo_matrix` sums duplicate (row, col) pairs on conversion, which is exactly the block sum.
    return sp.coo_matrix(
        (entries.data, (entries.row % n_cells, entries.col % n_cells)),
        shape=(n_cells, n_cells),
    ).tocsr()


def _block_tentative(
    aggregate: np.ndarray, n_coarse_cells: int, block_size: int, orthonormal: bool = False
) -> sp.csr_matrix:
    """The piecewise-constant prolongation for a **nodal** aggregation, one column per coarse field.

    Degree of freedom ``(cell i, field f)`` interpolates from coarse degree of freedom
    ``(aggregate[i], field f)`` — each field is carried by its own coarse unknown, so the coarse
    operator is a block operator of the *same* block size and the recursion can coarsen it again. A
    single shared column per aggregate would instead average the fields together, which is the
    field-blind failure in a different guise.

    Parameters
    ----------
    aggregate : np.ndarray
        Aggregate index per fine cell, shape ``(n_cells,)``.
    n_coarse_cells : int
        Number of aggregates.
    block_size : int
        Fields per cell, preserved onto the coarse level.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape ``(block_size * n_cells, block_size * n_coarse_cells)``, field-major on both sides.
    """
    n_cells = aggregate.shape[0]
    rows = np.arange(block_size * n_cells)
    fields, cells = np.divmod(rows, n_cells)
    cols = fields * n_coarse_cells + aggregate[cells]
    values = np.ones(rows.shape[0])
    if orthonormal:
        # Orthonormalize each aggregate's column, which for a piecewise-constant prolongation is
        # exactly its QR: divide by ``sqrt(|aggregate|)`` so every column has unit 2-norm.
        #
        # **This is provably inert on a two-level cycle with an exact coarse solve** -- ``P A_c^-1
        # P^T`` is invariant under a column rescaling -- and that is precisely why it went unnoticed:
        # every hierarchy here has been two-level. It is NOT inert deeper. With 0/1 columns the
        # Galerkin operator ``P^T A P`` picks up a scaling of order ``|agg_i| * |agg_j|`` per entry,
        # and the next level's smoother and spectral estimate read ``D^-1 A_c`` and ``lambda_max`` off
        # that mis-scaled operator -- neither of which is invariant to it. The distortion therefore
        # compounds once per level, in proportion to how UNEVEN the aggregate sizes are, which is the
        # standing explanation for depth being unhelpful and sometimes harmful here.
        counts = np.bincount(aggregate, minlength=n_coarse_cells).astype(np.float64)
        values = values / np.sqrt(np.maximum(counts, 1.0))[aggregate[cells]]
    return sp.csr_matrix(
        (values, (rows, cols)),
        shape=(block_size * n_cells, block_size * n_coarse_cells),
    )


def aggregate_size_histogram(aggregate: np.ndarray, n_coarse_cells: int) -> dict:
    """Aggregate-size statistics for one coarsening step — the spread is what makes scaling bite.

    A piecewise-constant prolongation with 0/1 columns distorts the Galerkin coarse operator in
    proportion to the *variation* in aggregate size, so a hierarchy whose aggregates are all the same
    size loses nothing by not orthonormalizing and one whose sizes span an order of magnitude loses a
    great deal. Reported rather than assumed, because the two cases call for different work.
    """
    counts = np.bincount(aggregate, minlength=n_coarse_cells)
    counts = counts[counts > 0]
    return {
        "aggregates": int(counts.size),
        "min": int(counts.min()),
        "median": float(np.median(counts)),
        "max": int(counts.max()),
        "singletons": int((counts == 1).sum()),
        "spread": float(counts.max() / max(counts.min(), 1)),
    }


def _cell_block_inverse(a: sp.csr_matrix, block_size: int, where: str = "operator") -> np.ndarray:
    """The inverse of each cell's own dense ``block_size × block_size`` block, shape ``(n_cells, b, b)``.

    What a **block** smoother inverts, in place of the scalar diagonal a point smoother uses. The
    distinction is the whole reason a nodal level needs its own smoother: on a multi-field operator the
    within-cell coupling between fields can dwarf the diagonal itself, and a point method throws all of
    it away, so it neither smooths nor even reliably contracts.

    The requirement is correspondingly weaker than the point smoother's, which is the useful part: point
    Jacobi needs every ``a_ii`` positive, while this needs only each block **invertible** — a block can
    be perfectly well conditioned with a negative entry on its diagonal.

    Field-major, so a cell's degrees of freedom are strided by ``n_cells`` rather than contiguous.

    **Singularity is judged against the product of the block's ROW NORMS, not against its Frobenius
    norm — and the difference is not pedantry.** Hadamard's inequality bounds ``|det| <= prod ||row_i||``
    with equality when the rows are orthogonal, so that ratio is how close a block is to rank-deficient
    *given the size of its rows*, and it is invariant under rescaling either a row or a column. A
    ``||B||_F ** b`` denominator is only equivalent when the rows have comparable magnitude, and it is
    dominated by the largest of them when they do not.

    On the coupled turbulence pair they do not. A degenerate-looking cell there reads

        [[ 8.8e-06,  1.7e-12],        row norms 8.8e-06 and 1.4e+03,
         [-1.3e+03,  1.5e-01]]        a ratio of 1.5e+08

    whose determinant is 1.35e-06 against a Frobenius-squared norm of 1.6e+06 — so the older test put
    the bar at 1.6e-06, just above the determinant, and **refused a block that is not singular at all**:
    the same block scores 1.2e-04 on the row-norm bound, eight orders clear. That false refusal aborted
    a march at a mid-march refresh, and rescaling does not avoid it (equilibration moves the imbalance
    from the rows into the subdiagonal and the Frobenius test misfires identically).

    Note the block is genuinely **ill conditioned** — around 1e12 here, so its inverse is large — and
    that is a real question about whether a cell-local method should invert it exactly. It is a
    different question from whether the block is invertible, which it is, and it is not this guard's to
    answer.

    Parameters
    ----------
    a : scipy.sparse matrix
        The operator whose cell blocks to invert, shape ``(block_size * n_cells,) * 2``, field-major.
    block_size : int
        Degrees of freedom per cell.
    where : str
        What to name in the refusal below. **A count alone is not diagnosable**: the caller coarsens,
        so this runs on the fine operator and on every Galerkin coarse one, and "4 of 23040 are
        singular" sent three separate attempts hunting a fine-grid state that was never degenerate.
        Say which level.

    Raises
    ------
    ValueError
        If any cell block is singular, naming where and how many — a genuinely degenerate operator, as
        distinct from a merely indefinite one, and not something a different smoother would rescue.
    """
    n_cells = a.shape[0] // block_size
    rows = np.arange(n_cells)
    blocks = np.empty((n_cells, block_size, block_size))
    for f in range(block_size):
        for g in range(block_size):
            blocks[:, f, g] = np.asarray(a[f * n_cells + rows, g * n_cells + rows]).ravel()
    # Hadamard: |det| <= prod ||row_i||, so the ratio is how near rank-deficiency the block is given
    # its own row sizes -- invariant under rescaling any row or column, which the Frobenius form is not.
    # A structurally empty row makes the bound zero, and the comparison below can then never fire, so
    # that case is named rather than left to arithmetic.
    bound = np.prod(np.linalg.norm(blocks, axis=2), axis=1)
    singular = (bound == 0.0) | (np.abs(np.linalg.det(blocks)) < 1e-12 * bound)
    if singular.any():
        raise ValueError(
            f"{where}: block smoothing needs every cell block invertible, but {int(singular.sum())} "
            f"of {n_cells} are singular. Unlike a non-positive scalar diagonal — which a block smoother "
            "tolerates — this is a genuinely degenerate operator that no smoother choice repairs."
        )
    return np.linalg.inv(blocks)


def _block_diagonal_inverse_operator(
    a: sp.csr_matrix, block_size: int, where: str = "operator"
) -> sp.csr_matrix:
    """The block-diagonal inverse as a sparse operator, for the build-time spectral estimate.

    The runtime smoother applies :func:`_cell_block_inverse` as a batched contraction; the *build* needs
    the same operator in sparse form, because the damping factor comes from a power iteration on
    ``D_blk⁻¹ A``. Estimating it from the scalar ``D⁻¹ A`` instead would scale the sweep by the wrong
    spectrum entirely — on this operator by orders of magnitude, since the two differ by exactly the
    within-cell coupling the block inverse exists to capture.

    Field-major, so cell ``i``'s degrees of freedom are ``{f * n_cells + i}`` and the assembled operator
    is block-diagonal in the *cell* sense while being scattered in index space.
    """
    inverse = _cell_block_inverse(a, block_size, where)
    n_cells = inverse.shape[0]
    cells = np.arange(n_cells)
    rows = np.concatenate(
        [f * n_cells + cells for f in range(block_size) for _ in range(block_size)]
    )
    cols = np.concatenate(
        [g * n_cells + cells for _ in range(block_size) for g in range(block_size)]
    )
    values = np.concatenate(
        [inverse[:, f, g] for f in range(block_size) for g in range(block_size)]
    )
    return sp.coo_matrix((values, (rows, cols)), shape=a.shape).tocsr()


def _square_graph(graph: sp.csr_matrix) -> sp.csr_matrix:
    """``G·G`` — the graph of distance-2 connectivity, for an aggressive first coarsening.

    Aggregating on the squared graph makes each aggregate span a cell's neighbours-of-neighbours, which
    coarsens far faster than the plain graph and is what keeps the number of levels low as the mesh
    grows. It is the standard aggressive-coarsening device and is applied to the **first** level only:
    deeper levels aggregate on their own plain graph, because by then the operator is dense enough that
    squaring it again would produce very large, badly-shaped aggregates.
    """
    squared = (graph @ graph).tocsr()
    squared.setdiag(0.0)
    squared.eliminate_zeros()
    return squared


def _mis_aggregate(
    graph: sp.csr_matrix, seed: int = 0, avoid_singletons: bool = False
) -> tuple[np.ndarray, int]:
    """Greedy maximal-independent-set aggregation over a **randomized** visit order.

    One sweep over the vertices in a random permutation. Any vertex still unclaimed when it is visited
    becomes a **selector**: it opens an aggregate and claims every unclaimed neighbour. Because a
    selector is chosen on its *own* state rather than its neighbourhood's, a single sweep leaves nothing
    behind — every vertex is either a selector or claimed by one — so there is no leftover pass and no
    aggregate built from whatever happened to be adjacent to a straggler.

    That is the difference from a stricter two-pass scheme that only seeds from vertices whose whole
    neighbourhood is free: such a scheme seeds few aggregates and consigns most vertices to the
    cleanup pass, which attaches them to the first adjacent aggregate it finds. The aggregates it
    produces are correspondingly ragged, and a ragged coarse space is measurably worse at the same
    aggregate count.

    **The order is random on purpose.** A locality-preserving order (reverse Cuthill--McKee, say) is the
    intuitive choice and is the wrong one here: sweeping along a spatial ordering makes each selector
    claim the vertices just ahead of it, producing long thin aggregates aligned with the numbering
    rather than compact ones. Randomizing removes that bias.

    Parameters
    ----------
    graph : scipy.sparse matrix
        Symmetric connectivity, shape ``(n, n)``. Only its sparsity is read.
    seed : int
        Seed for the permutation, so a hierarchy is reproducible.
    avoid_singletons : bool
        Attach a vertex whose neighbours are **all already claimed** to one of their aggregates instead
        of letting it open an aggregate containing only itself.

        Those singletons are an artifact of arrival order, not of the graph: a vertex reached late in
        the random sweep can find every neighbour taken, and the one-sweep selector rule then opens a
        new aggregate for it alone. Measured on a coupled flow block, the second level of a three-level
        hierarchy came out **49 singletons of 161 aggregates, median aggregate size 3** -- a coarse
        space that is a slightly smaller copy of the fine grid with a third of its unknowns standing for
        one cell each and coupling to almost nothing. They also interact badly with an orthonormalized
        prolongation, which scales a column by ``1 / sqrt(|agg|)`` and so *promotes* a singleton by
        several-fold against a real aggregate.

    Returns
    -------
    tuple
        ``(aggregate, roots, n_aggregates)`` — the aggregate index of every vertex, and the vertex
        that seeded each aggregate. The roots are returned because a caller coarsening a *squared*
        graph needs them to repair the result (:func:`_reattach_to_adjacent_root`).
    """
    n = graph.shape[0]
    indptr, indices = graph.indptr, graph.indices
    aggregate = np.full(n, -1, dtype=np.int64)
    roots: list[int] = []
    for i in np.random.default_rng(seed).permutation(n):
        if aggregate[i] != -1:
            continue
        neighbours = indices[indptr[i] : indptr[i + 1]]
        if avoid_singletons and neighbours.size:
            claimed = neighbours[aggregate[neighbours] != -1]
            if claimed.size == neighbours.size:
                # Every neighbour is taken, so opening an aggregate here would produce a singleton.
                # Join an adjacent one instead. A vertex with NO neighbours is a true isolate and still
                # gets its own aggregate -- it has nothing to attach to.
                aggregate[i] = aggregate[claimed[0]]
                continue
        # A true singleton (no neighbour but itself) is left for the sweep to pick up as its own
        # aggregate rather than being attached to something it does not touch.
        aggregate[i] = len(roots)
        for j in neighbours:
            if aggregate[j] == -1:
                aggregate[j] = len(roots)
        roots.append(int(i))
    return aggregate, np.asarray(roots, dtype=np.int64), len(roots)


def _reattach_to_adjacent_root(
    aggregate: np.ndarray, roots: np.ndarray, graph: sp.csr_matrix
) -> np.ndarray:
    """Repair a squared-graph aggregation: give every member a root it is genuinely adjacent to.

    Aggregating the squared graph is what makes a hierarchy coarsen fast enough to stay shallow, but
    it buys that by letting an aggregate reach two hops: a member can be assigned to a root it has no
    direct coupling to at all, which is a poor thing for a piecewise-constant coarse basis function to
    be supported on. This walks each root in ascending index order and claims every **distance-1**
    neighbour — in the *unsquared* graph — that currently belongs to some other aggregate.

    Only members move; a root is never stolen from, so no aggregate is emptied and the count is
    unchanged. What changes is aggregate *shape*. Later roots override earlier ones, so a member
    adjacent to several roots ends up with the highest-indexed of them — arbitrary, but the tie has to
    break somehow and matching the reference's order keeps the two comparable.

    Parameters
    ----------
    aggregate : np.ndarray
        Aggregate index per vertex, shape ``(n,)``, from coarsening the squared graph.
    roots : np.ndarray
        The seeding vertex of each aggregate, shape ``(n_aggregates,)``.
    graph : scipy.sparse matrix
        The **unsquared** symmetric connectivity, shape ``(n, n)``. Only its sparsity is read.

    Returns
    -------
    np.ndarray
        The repaired aggregate index per vertex, shape ``(n,)``.
    """
    aggregate = aggregate.copy()
    indptr, indices = graph.indptr, graph.indices
    is_root = np.zeros(aggregate.shape[0], dtype=bool)
    is_root[roots] = True
    for root in np.sort(roots):
        target = aggregate[root]
        for j in indices[indptr[root] : indptr[root + 1]]:
            if not is_root[j] and aggregate[j] != target:
                aggregate[j] = target
    return aggregate


def _absorb_singleton_aggregates(
    aggregate: np.ndarray, n_aggregates: int, graph: sp.csr_matrix
) -> tuple[np.ndarray, int]:
    """Dissolve any aggregate left holding only its own root, and renumber what remains.

    The repair inside the aggregation sweep cannot catch these. It refuses to *open* an aggregate that
    would be a singleton, but the reattachment pass that repairs a squared graph's reach runs
    afterwards and moves members away from aggregates it does not own — so an aggregate can be reduced
    to its root alone after the fact. A coarse unknown standing for one cell, coupled to almost
    nothing, is what makes a coarse level a slightly smaller copy of the fine one.

    Each lone vertex joins the largest aggregate it is genuinely adjacent to, ties broken by lowest
    index so the result does not depend on iteration order. A vertex with no neighbour outside its own
    aggregate is a true isolate and keeps its aggregate — there is nothing to join. Aggregate labels
    are then made contiguous again, because the caller uses the count to size the prolongation.

    Parameters
    ----------
    aggregate : np.ndarray
        Aggregate index per vertex, shape ``(n,)``.
    n_aggregates : int
        Number of aggregates before absorption.
    graph : scipy.sparse matrix
        The symmetric connectivity the aggregates were formed over, shape ``(n, n)``. Only its
        sparsity is read.

    Returns
    -------
    tuple
        ``(aggregate, n_aggregates)`` — relabelled contiguously from zero.
    """
    counts = np.bincount(aggregate, minlength=n_aggregates)
    if not (counts == 1).any():
        return aggregate, n_aggregates
    aggregate = aggregate.copy()
    indptr, indices = graph.indptr, graph.indices
    for vertex in np.flatnonzero(counts[aggregate] == 1):
        current = aggregate[vertex]
        if counts[current] != 1:  # already absorbed one of its neighbours
            continue
        neighbours = indices[indptr[vertex] : indptr[vertex + 1]]
        candidates = aggregate[neighbours[aggregate[neighbours] != current]]
        if candidates.size == 0:
            continue
        target = int(candidates[np.lexsort((candidates, -counts[candidates]))[0]])
        counts[current] -= 1
        counts[target] += 1
        aggregate[vertex] = target
    kept = np.flatnonzero(counts > 0)
    relabel = np.zeros(n_aggregates, dtype=np.int64)
    relabel[kept] = np.arange(kept.size, dtype=np.int64)
    return relabel[aggregate], int(kept.size)


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


# --- smoothed aggregation ---------------------------------------------------------------
#
# Piecewise-constant (unsmoothed) aggregation is correct but weak (V-cycle contraction ~0.97): the
# coarse space cannot represent smooth error. Smoothed aggregation fixes this by smoothing the
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


class _SparseLevel(eqx.Module):
    """One smoothed-aggregation level: a general sparse operator + its prolongation, all frozen.

    **The level is split into a static index structure and dynamic values (binding).** Only ``n`` and
    ``n_coarse`` are static: they size the sparse matvec's output (:func:`_coo_apply`'s ``n_out``), so
    they must be concrete. Everything else — including ``lam_max``, which is pure arithmetic in the
    smoothers — rides as a **traced** leaf. That split is what makes a hierarchy *refreshable*: because
    the coarsening is a pure function of the graph (see :func:`_aggregate`, which reads only the
    sparsity pattern), re-deriving a hierarchy at a new operator on the same mesh yields the identical
    structure and changes only these values, so a refreshed hierarchy passed as a **jit argument** has
    unchanged static metadata and array shapes — a compilation-cache hit rather than a rebuild-and-
    recompile. Keeping ``lam_max`` a Python ``float`` would defeat exactly that (a changed static field
    is a changed cache key), which is why it is stored as a 0-d array.
    """

    n: int = eqx.field(static=True)  # cells at this level (sizes the matvec output)
    operator: _CsrOperator  # the level operator A, in CSR form and owning its matvec
    diagonal: jnp.ndarray  # (n,) diagonal of A
    lam_max: jnp.ndarray  # 0-d: largest eigenvalue of D^-1 A, for the smoother damping
    coarse_inv: jnp.ndarray | None  # dense pseudo-inverse (coarsest level only); None otherwise
    p_frow: jnp.ndarray | None  # (pnnz,) prolongation fine row (this level); None on coarsest
    p_ccol: jnp.ndarray | None  # (pnnz,) prolongation coarse col (next level)
    p_val: jnp.ndarray | None  # (pnnz,) prolongation value
    n_coarse: int = eqx.field(static=True)  # next-coarser cell count (0 on coarsest)
    # A nodal level additionally carries the inverse of each cell's own dense block. The scalar
    # diagonal cannot smooth a multi-field operator whose within-cell coupling dwarfs it -- a point
    # method discards that coupling entirely -- so the smoother inverts the block instead. Traced,
    # like `diagonal`, so a refreshed hierarchy stays a compilation-cache hit; `block_size` is static
    # because it sizes the reshape.
    block_inverse: jnp.ndarray | None = None  # (n_cells, b, b), or None for a scalar level
    block_size: int = eqx.field(static=True, default=1)


class SmoothedHierarchy(eqx.Module):
    """A built smoothed-aggregation hierarchy: general-sparse levels, finest to coarsest.

    The levels may have been coarsened on a **rescaled** operator ``D A D`` rather than on ``A``
    itself (:func:`~aquaflux.solve.frozen_operator.symmetrically_equilibrate`), in which case
    ``equilibration`` holds ``diag(D)`` and :meth:`fixed_cycle_solve` moves the right-hand side into
    that scale and the answer back out. Keeping the factor **on the hierarchy** rather than asking
    each caller to remember it is what makes the two spellings interchangeable at every call site;
    ``None`` means the levels were built on ``a`` as given, and the solve is then unscaled.
    """

    levels: tuple[_SparseLevel, ...]
    equilibration: jnp.ndarray | None = None  # (n,) diag(D), or None for an unscaled hierarchy

    def fixed_cycle_solve(self, b: jnp.ndarray, cycles: int, ops: _VCycleOps) -> jnp.ndarray:
        """``cycles`` V-cycles for ``A x = b``, undoing any equilibration around the solve.

        With ``D A D`` coarsened into the levels, ``A^-1 = D (D A D)^-1 D``, so the scaled solve is
        ``x = D M(D b)`` — exact, because the rescaling is a similarity transform and not an
        approximation. It composes two fixed linear maps with a third, so the result is still a fixed
        linear operator: valid as a frozen preconditioner under a non-flexible Krylov solve, and
        transposable for the adjoint.

        Parameters
        ----------
        b : jnp.ndarray
            Right-hand side, shape ``(n,)``.
        cycles : int
            Number of V-cycles (static).
        ops : _VCycleOps
            The family's restriction, prolongation and smoother.

        Returns
        -------
        jnp.ndarray
            The approximate solution ``x``, shape ``(n,)``.
        """
        if self.equilibration is None:
            return _fixed_cycle_solve(self.levels, b, cycles, ops)
        return self.equilibration * _fixed_cycle_solve(
            self.levels, self.equilibration * b, cycles, ops
        )


def _sparse_level(
    a: sp.csr_matrix,
    lam_max: float,
    coarse_inv: np.ndarray | None,
    prolongation: sp.coo_matrix | None,
    n_coarse: int,
    block_size: int = 1,
) -> _SparseLevel:
    """Freeze a scipy sparse operator (+ optional prolongation / coarse inverse) into JAX arrays."""
    p_frow = p_ccol = p_val = None
    if prolongation is not None:
        p_frow = jnp.asarray(prolongation.row)
        p_ccol = jnp.asarray(prolongation.col)
        p_val = jnp.asarray(prolongation.data)
    return _SparseLevel(
        n=a.shape[0],
        operator=_CsrOperator.from_scipy(a),
        diagonal=jnp.asarray(a.diagonal()),
        lam_max=jnp.asarray(float(lam_max)),
        coarse_inv=None if coarse_inv is None else jnp.asarray(coarse_inv),
        p_frow=p_frow,
        p_ccol=p_ccol,
        p_val=p_val,
        n_coarse=n_coarse,
        block_inverse=(
            None if block_size == 1 else jnp.asarray(_cell_block_inverse(a, block_size))
        ),
        block_size=block_size,
    )


def _largest_singular_value(matrix: sp.spmatrix, iterations: int = 20) -> float:
    """Largest singular value of a sparse matrix, by power iteration on ``MᵀM`` (off-jit).

    Distinct from :func:`_spectral_radius`, and the distinction is load-bearing for a nonsymmetric
    operator: ``sigma_max >= |lambda|_max``, so using an eigenvalue where the method calls for a
    singular value produces too large a smoothing step. They coincide only for a normal matrix.
    """
    rng = np.random.default_rng(0)
    v = rng.standard_normal(matrix.shape[1])
    v /= np.linalg.norm(v)
    sigma = 1.0
    for _ in range(iterations):
        w = matrix.T @ (matrix @ v)
        norm = np.linalg.norm(w)
        if norm == 0.0:
            return 0.0
        v = w / norm
        sigma = np.sqrt(norm)
    return float(sigma)


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


def _aggregation_edges(a_agg: sp.csr_matrix, strength_threshold: float) -> sp.coo_matrix:
    """The upper-triangular edge set the greedy aggregation pairs cells across.

    ``strength_threshold == 0`` returns the aggregation operator's **full** graph — isotropic
    aggregation, which pairs a cell with any neighbour regardless of coupling strength. On an
    anisotropic operator (a high-aspect-ratio cell couples far more strongly across the thin direction
    than along it) that coarsens across the stiff direction, and the resulting V-cycle stalls
    (contraction → 1 as the aspect ratio grows).

    ``strength_threshold > 0`` keeps only **strong** connections — edge ``(i, j)`` survives iff
    ``|A_ij| >= threshold · max_{k!=i}|A_ik|`` from *either* endpoint (:func:`_strength_classical`,
    symmetrized) — so aggregates form along the strong-coupling directions and the coarse space
    resolves the stiff modes. This is the strength-of-connection fix for anisotropic / high-skewness
    operators; ``0.25`` is a standard threshold. It makes the coarsening **value-dependent** (it reads
    ``|A_ij|``), so a hierarchy re-derived at a new operator no longer has an invariant structure — see
    the note on :func:`build_smoothed_hierarchy`.
    """
    if strength_threshold <= 0.0:
        return sp.triu(a_agg, k=1).tocoo()
    strength = _strength_classical(a_agg, strength_threshold)
    return sp.triu((strength + strength.T).tocsr(), k=1).tocoo()


# How the tentative piecewise-constant prolongation is smoothed before it is frozen into a level.
#
# `"none"` keeps the tentative injection (plain aggregation). It is not a degenerate case: on a
# strongly indefinite or badly scaled operator the smoothing step degrades the coarse correction it
# exists to improve, and the smoothing damping is only meaningful on a unit-magnitude diagonal.
# `"standard"` is the textbook `P <- (I - 1.4/sigma_max(D^-1 A)) P_tent` on the true operator with the
# scalar diagonal. `"symmetric-part"` damps by an eigenvalue of the aggregation operator instead, and
# on a nodal level uses the block diagonal; the two coincide on a symmetric operator up to the damping
# constant.
_PROLONGATION_SMOOTHING = frozenset({"none", "standard", "symmetric-part"})


#: Aggregate-size statistics from the most recent hierarchy build, newest level last. A diagnostic
#: only -- read it after a build and clear it before the next one.
_AGGREGATE_STATS: list[dict] = []


#: Largest coarsest-level size, in degrees of freedom, that may be inverted densely. The coarse solve
#: is a dense pseudo-inverse: quadratic to store (8 bytes per entry, so ~512 MB here) and cubic to
#: build. Exceeding it is never intentional -- it means the level cap stopped the coarsening before the
#: coarse-size limit could, which makes the coarse grid grow with the mesh instead of staying fixed.
_MAX_DENSE_COARSE_DOFS = 8192


def _build_aggregation_hierarchy(
    a: sp.csr_matrix,
    *,
    aggregation_operator: Callable[[sp.csr_matrix], sp.csr_matrix],
    omega_smooth: float,
    max_coarse: int,
    max_levels: int,
    strength_threshold: float = 0.0,
    block_size: int = 1,
    orthonormal_prolongation: bool = False,
    avoid_singletons: bool = False,
    mis_aggregation: bool = False,
    aggressive_levels: int = 0,
    equilibrate: bool = False,
    prolongation_smoothing: str = "symmetric-part",
) -> SmoothedHierarchy:
    """Coarsen ``a`` into a frozen smoothed-aggregation hierarchy — the loop shared by the symmetric
    and convection-diffusion builders.

    ``aggregation_operator`` maps each level's true operator to the operator that drives aggregation
    and prolongation smoothing: identity for a symmetric graph Laplacian, and the symmetric part
    ``(A + Aᵀ)/2`` for a nonsymmetric convection-diffusion operator (whose advected error modes need a
    well-shaped, stable coarse space). The Galerkin coarse operator ``Pᵀ A P`` always carries the true
    ``a`` up the levels, so a nonsymmetric operator stays convection-aware.

    Two spectral estimates play different roles. ``lam_smooth`` — the largest eigenvalue magnitude of
    the aggregation operator's ``D⁻¹ A_agg`` — sets the constant-preserving prolongation smoothing.
    ``lam_store`` — of the **true** ``D⁻¹ A`` — is frozen into each level for the runtime smoother: the
    Galerkin coarse operators of a convection-diffusion problem pick up large complex eigenvalues the
    symmetric part misses, so a coarse-level smoother damped by ``lam_smooth`` alone would diverge. When
    the aggregation operator *is* the true operator (the symmetric path) the two coincide and the
    estimate is computed once.

    ``equilibrate`` rescales the fine operator to a unit-magnitude diagonal before any of that runs,
    and the hierarchy then carries the factor so the solve is unchanged. It matters because every
    quantity above is derived from ``D``: both spectral estimates, the smoother damping they scale,
    and the prolongation smoothing. On an operator whose diagonal spans orders of magnitude those are
    calibrated against a scale with no meaning — the same algorithm on the rescaled matrix builds a
    different hierarchy. Only the *fine* operator is rescaled; the Galerkin coarse operators inherit
    whatever scale the prolongation gives them, which is what a coarsening on a rescaled input does.
    """
    if prolongation_smoothing not in _PROLONGATION_SMOOTHING:
        raise ValueError(
            f"prolongation_smoothing must be one of {sorted(_PROLONGATION_SMOOTHING)}, "
            f"got {prolongation_smoothing!r}."
        )
    scale: np.ndarray | None = None
    if equilibrate:
        a, scale = symmetrically_equilibrate(a)
    _AGGREGATE_STATS.clear()  # this build's statistics only; the consumer reads the whole list
    levels: list[_SparseLevel] = []
    while True:
        a_agg = aggregation_operator(a)
        # A nodal level inverts cell blocks, not the scalar diagonal, so the scalar-positivity
        # precondition does not apply to it -- `_cell_block_inverse` enforces the weaker and correct
        # one (every block invertible) when it builds them.
        if block_size == 1:
            _require_positive_diagonal(
                a.diagonal(), f"_build_aggregation_hierarchy (level {len(levels)})"
            )
            d_inv = sp.diags(1.0 / a.diagonal())
        else:
            d_inv = _block_diagonal_inverse_operator(
                a, block_size, f"_build_aggregation_hierarchy (level {len(levels)})"
            )
        lam_smooth = _spectral_radius(d_inv @ a_agg)  # prolongation-smoothing damping
        lam_store = (
            lam_smooth if a_agg is a else _spectral_radius(d_inv @ a)
        )  # runtime smoother scale
        if a.shape[0] <= max_coarse or len(levels) + 1 >= max_levels:
            if a.shape[0] > _MAX_DENSE_COARSE_DOFS:
                raise ValueError(
                    f"coarsest level has {a.shape[0]} degrees of freedom, above the "
                    f"{_MAX_DENSE_COARSE_DOFS} that may be inverted densely "
                    f"(~{8 * a.shape[0] ** 2 / 1e9:.1f} GB, and cubic to build). The level cap "
                    f"(max_levels={max_levels}) stopped the coarsening before max_coarse="
                    f"{max_coarse} could: raise max_levels so the size limit binds, or lower "
                    f"max_coarse."
                )
            # Coarsest level: a direct (dense pseudo-inverse) solve — an inexact coarse solve is the
            # dominant cause of mesh-dependent V-cycle degradation, so it must be an actual solve; pinv
            # also handles a nonsymmetric coarse operator.
            levels.append(
                _sparse_level(a, lam_store, np.linalg.pinv(a.toarray()), None, 0, block_size)
            )
            break
        # Coarsen CELLS, not degrees of freedom, when the operator has several fields per cell: the
        # aggregation graph is collapsed onto cell connectivity and the prolongation carries each field
        # on its own coarse unknown, so the coarse operator keeps the same block size and the recursion
        # stays nodal all the way down. At `block_size == 1` both reduce to the scalar path exactly.
        # Take the magnitude BEFORE symmetrizing, never after. On a nonsymmetric operator an edge
        # with `A_ij ~ -A_ji` cancels in the symmetric part and vanishes from the graph entirely, so
        # two strongly coupled cells can end up with no edge between them to aggregate across. On an
        # M-matrix (every frozen upwind transport operator) the off-diagonals share a sign and the two
        # orders coincide exactly, which is why this costs the shipped hierarchies nothing.
        graph = _cell_graph(a, block_size) if block_size > 1 else abs(a).tocsr()
        if mis_aggregation:
            connectivity = sp.csr_matrix(abs(_aggregation_edges(graph, strength_threshold)))
            connectivity = (connectivity + connectivity.T).tocsr()
            if len(levels) < aggressive_levels:
                # Aggressive coarsening: aggregate over distance-2 connectivity so the hierarchy
                # coarsens fast enough to stay shallow as the mesh grows, then repair the reach it
                # buys by re-attaching each member to a root it actually touches.
                aggregate, roots, n_coarse_cells = _mis_aggregate(
                    _square_graph(connectivity), seed=len(levels), avoid_singletons=avoid_singletons
                )
                aggregate = _reattach_to_adjacent_root(aggregate, roots, connectivity)
                if avoid_singletons:
                    # Reattachment moves members between aggregates, so it can strand a root that the
                    # sweep's own repair had no way to foresee. Dissolve those here, where the final
                    # assignment is known.
                    aggregate, n_coarse_cells = _absorb_singleton_aggregates(
                        aggregate, n_coarse_cells, connectivity
                    )
            else:
                aggregate, _, n_coarse_cells = _mis_aggregate(
                    connectivity, seed=len(levels), avoid_singletons=avoid_singletons
                )
        else:
            upper = _aggregation_edges(
                graph, strength_threshold
            )  # full graph, or strong edges only
            aggregate, n_coarse_cells = _aggregate(upper.row, upper.col, graph.shape[0])
        _AGGREGATE_STATS.append(aggregate_size_histogram(aggregate, n_coarse_cells))
        tentative = _block_tentative(
            aggregate, n_coarse_cells, block_size, orthonormal_prolongation
        )
        n_coarse = tentative.shape[1]
        if prolongation_smoothing == "none":
            prolongation = tentative.tocsr()
        elif prolongation_smoothing == "standard":
            # The prolongator smoothing as smoothed aggregation actually specifies it:
            #   P <- (I - 1.4/sigma_max * D^-1 A) P_tent
            # with four details that each matter and that a from-memory version gets wrong.
            # `A` is the TRUE operator, not its symmetric part -- the advected error modes the
            # smoothing is meant to capture live in the nonsymmetric part. `D` is the SCALAR
            # diagonal even when the operator is a block one. The scale is the largest
            # SINGULAR value, not the largest eigenvalue: for a nonsymmetric operator
            # sigma_max >= |lambda|_max, so an eigenvalue estimate gives too large a step and
            # over-smooths the prolongator, degrading the very coarse space it is improving.
            # And the whole step presumes a unit-magnitude diagonal (`equilibrate=True`): the
            # step length is set by sigma_max(D^-1 A), which on a raw operator whose diagonal
            # spans orders of magnitude measures the spread of the diagonal rather than
            # anything about the coupling, so the resulting smoothing is arbitrary.
            scalar_d_inv = sp.diags(1.0 / a.diagonal())
            prolongation = (
                tentative
                - (1.4 / _largest_singular_value(scalar_d_inv @ a))
                * (scalar_d_inv @ (a @ tentative))
            ).tocsr()
        else:
            # The symmetric-part variant: damp by an EIGENVALUE of the aggregation operator's
            # `D^-1 A_agg`, and use the block diagonal on a nodal level. On the symmetric path
            # `A_agg is A` and the two forms coincide up to the damping constant.
            prolongation = (
                tentative - (omega_smooth * 2.0 / lam_smooth) * (d_inv @ (a_agg @ tentative))
            ).tocsr()
        levels.append(_sparse_level(a, lam_store, None, prolongation.tocoo(), n_coarse, block_size))
        a = (prolongation.T @ a @ prolongation).tocsr()  # Galerkin coarse operator from the true A
    return SmoothedHierarchy(
        tuple(levels), None if scale is None else jnp.asarray(scale, dtype=jnp.float64)
    )


def build_smoothed_hierarchy(
    a: sp.csr_matrix,
    *,
    omega_smooth: float = 2.0 / 3.0,
    max_coarse: int = 16,
    max_levels: int = 20,
    strength_threshold: float = 0.0,
) -> SmoothedHierarchy:
    """Build the smoothed-aggregation hierarchy for operator ``a`` — off the jit path.

    ``a`` is an assembled **symmetric** operator (a graph Laplacian of the edge coefficients, plus any
    boundary stiffness, with a closed-domain pressure system's pinned degree of freedom already
    decoupled). Aggregation and prolongation smoothing run on ``a`` itself, and its ``lambda_max`` also
    feeds the runtime Chebyshev smoother.

    Parameters
    ----------
    a : scipy.sparse matrix
        The frozen symmetric fine operator, shape ``(n_cells, n_cells)``.
    omega_smooth : float
        Prolongation-smoothing damping factor; the applied damping is ``omega_smooth * 2 / lambda_max``
        (i.e. ``4/(3 lambda_max)`` at the default ``2/3``), with ``lambda_max`` estimated per level.
    max_coarse : int
        Stop coarsening once a level has at most this many **degrees of freedom** (solved directly
        there). Dofs, not cells: at ``block_size`` fields per cell the two differ by that factor, and
        the limit exists to bound a dense pseudo-inverse whose cost is cubic in the dof count.
    max_levels : int
        Hard cap on the number of levels.
    strength_threshold : float
        Strength-of-connection threshold for the aggregation (default ``0`` = the historical isotropic
        aggregation on the full graph). A value like ``0.25`` aggregates only along **strong**
        connections (:func:`_aggregation_edges`), which is what keeps the V-cycle contracting on an
        **anisotropic / high-aspect-ratio** operator — where isotropic aggregation coarsens across the
        stiff direction and stalls (measured: on a uniformly anisotropic Poisson the plain V-cycle
        contraction climbs past ``0.9`` and fails to reach a 1% residual, while ``0.25`` holds ``~0.5``
        and reaches 1% in ~3 cycles, mesh-independently). **It makes the coarsening value-dependent**,
        so unlike the ``0`` path a re-derivation at a new operator changes the aggregate structure and
        shapes; use it only where the hierarchy is frozen (never refreshed), or refresh by rebuilding.

    Returns
    -------
    SmoothedHierarchy
        Frozen finest-to-coarsest general-sparse levels for :func:`smoothed_multigrid_solve`.
    """
    return _build_aggregation_hierarchy(
        a.tocsr(),
        aggregation_operator=lambda m: m,
        omega_smooth=omega_smooth,
        max_coarse=max_coarse,
        max_levels=max_levels,
        strength_threshold=strength_threshold,
    )


class _CsrOperator(eqx.Module):
    """A frozen level operator in compressed-sparse-row (CSR) form, owning its own matvec.

    **Why CSR and not the coordinate (COO) form the rest of this module uses.** A COO matvec is a
    scatter-add: it computes every ``val * x[col]`` and then reduces them onto output rows that several
    entries share. A CSR matvec instead walks one row at a time and accumulates into a single output
    element, so nothing collides and the reduction is a contiguous scan. Measured on the coupled
    turbulence block (46080 rows, 4.2M nonzeros), that is worth **9.5x** — 13.3 ms for the scatter-add
    against 1.4 ms here, which also beats a host ``scipy`` CSR matvec at 2.6 ms. The level operator is
    applied about ten times per V-cycle, so this is most of what the cycle costs.

    Kept as its own object rather than three loose arrays on each level because both level kinds carry
    one and both apply it the same way; the arrays never travel without each other.

    The three arrays are all **traced** leaves and only ``shape`` is static, so re-deriving a hierarchy
    at a new operator on the same graph leaves every shape untouched — the compiled V-cycle is a cache
    hit rather than a retrace, which is what makes a mid-march preconditioner refresh affordable.
    """

    indptr: jnp.ndarray  # (n_rows + 1,) row start offsets
    indices: jnp.ndarray  # (nnz,) column index of each entry
    data: jnp.ndarray  # (nnz,) the entries themselves
    shape: tuple[int, int] = eqx.field(static=True)

    @classmethod
    def from_scipy(cls, a: sp.csr_matrix) -> _CsrOperator:
        """Freeze an assembled ``scipy`` matrix, in canonical (sorted-column) CSR form."""
        a = a.tocsr()
        a.sort_indices()
        return cls(
            indptr=jnp.asarray(a.indptr),
            indices=jnp.asarray(a.indices),
            data=jnp.asarray(a.data),
            shape=a.shape,
        )

    def apply(self, x: jnp.ndarray) -> jnp.ndarray:
        """``A x``. Linear and transposable, so the adjoint's transpose solve goes through it."""
        return BCSR((self.data, self.indices, self.indptr), shape=self.shape) @ x

    @property
    def diagonal(self) -> jnp.ndarray:
        """The operator's diagonal, read off the CSR structure."""
        rows = jnp.repeat(
            jnp.arange(self.shape[0]), jnp.diff(self.indptr), total_repeat_length=self.data.shape[0]
        )
        return segment_sum(
            jnp.where(rows == self.indices, self.data, 0.0),
            rows,
            self.shape[0],
            indices_are_sorted=True,
        )


def _coo_apply(row, col, val, x: jnp.ndarray, n_out: int) -> jnp.ndarray:
    """General sparse matvec ``M x`` for a COO operator: ``segment_sum(val * x[col], row, n_out)``.

    The one sparse-matvec kernel, shared by every frozen operator, prolongation, and restriction.

    ``indices_are_sorted`` is asserted rather than hoped for: every operator here is frozen from a
    ``scipy.sparse`` CSR matrix, and ``csr.tocoo()`` emits entries in row-major order, so the segment
    identifiers are non-decreasing by construction. Saying so lets the reduction run as a contiguous
    segmented scan instead of an unordered scatter-add, which is the difference between reading the
    output once and colliding on it — the same reason a CSR matvec beats a COO one.
    """
    return segment_sum(val * x[col], row, n_out, indices_are_sorted=True)


def _operator_matvec(level: _SparseLevel | _AirLevel, x: jnp.ndarray) -> jnp.ndarray:
    """Apply a frozen level's operator ``A x``. Works for either level kind — ``_SparseLevel`` and
    ``_AirLevel`` both carry the operator as a :class:`_CsrOperator`."""
    return level.operator.apply(x)


def _chebyshev_smooth(
    level: _SparseLevel, b: jnp.ndarray, x: jnp.ndarray, degree: int, lo_frac: float
) -> jnp.ndarray:
    """Chebyshev polynomial smoother of ``degree`` on ``[lo_frac, 1.05] * lambda_max`` (of ``D^-1 A``).

    Matrix-free (only ``A``-matvecs and the diagonal), a fixed *linear* operator, and a far stronger
    smoother than the same number of damped-Jacobi sweeps — the fix for the weak-smoother half of the
    V-cycle degradation. Reuses the per-level ``lambda_max`` estimated at build time.

    The error-propagation polynomial is the scaled Chebyshev polynomial
    ``P_k(z) = T_k((theta - z) / delta) / T_k(theta / delta)`` on the interval ``[lo, hi]`` with
    centre ``theta = (lo + hi) / 2`` and half-width ``delta = (hi - lo) / 2``. Since ``theta / delta
    > 1``, ``|P_k| <= 1 / T_k(theta / delta) < 1`` across ``[lo, hi]`` — every mode in the band is
    damped, and the damping is optimal (min-max) over the band. Realized by the standard three-term
    recurrence (Saad, *Iterative Methods for Sparse Linear Systems*, Alg. 12.1): the first step is
    the scaled-Richardson ``(1 / theta) D^-1 r``, and each subsequent increment mixes the previous
    increment with the current preconditioned residual through the ``rho`` recurrence.
    """
    lo, hi = level.lam_max * lo_frac, level.lam_max * 1.05
    centre, half_width = 0.5 * (hi + lo), 0.5 * (hi - lo)
    sigma = centre / half_width  # theta / delta > 1
    inv_diagonal = 1.0 / level.diagonal

    residual = b - _operator_matvec(level, x)
    increment = (inv_diagonal * residual) / centre  # first step: (1 / theta) D^-1 r
    x = x + increment
    rho = 1.0 / sigma
    for _ in range(1, degree):
        residual = b - _operator_matvec(level, x)
        rho_next = 1.0 / (2.0 * sigma - rho)
        increment = rho_next * rho * increment + (2.0 * rho_next / half_width) * (
            inv_diagonal * residual
        )
        x = x + increment
        rho = rho_next
    return x


_Smoother = Callable[[object, jnp.ndarray, jnp.ndarray], jnp.ndarray]


class _VCycleOps(NamedTuple):
    """The three level-local operations that specialize the shared frozen V-cycle recursion.

    ``restrict(level, r) -> coarse_r`` moves a fine residual to the next-coarser level; ``prolong(level,
    coarse_e) -> fine_e`` moves a coarse error back; ``smooth(level, b, x) -> x`` applies a fixed,
    matrix-free relaxation for ``A x = b``. The frozen-operator matvec and the direct coarse solve are
    identical across every frozen path, so only these three vary: smoothed aggregation restricts with
    ``Pᵀ`` (the ``R = Pᵀ`` special case, :func:`_smoothed_ops`) and lAIR with an independent restriction
    ``R`` (:func:`_air_ops`); the smoother is Chebyshev / damped-Jacobi (symmetric / convection
    two-level) or FC-Jacobi (reduction).

    ``mu`` is the number of times each coarse level is visited per visit of its parent: 1 is a V-cycle,
    2 a W-cycle, which moves work onto the cheaper coarse levels instead of buying convergence with more
    fine-level relaxation. ``pre_smooth`` may be turned off, which removes both the pre-relaxation and
    the residual evaluation that follows it — with a zero initial guess the fine residual *is* the
    right-hand side, so the matvec is pure waste. Both are static, so the cycle stays a fixed linear
    operator and transposes as one.
    """

    restrict: Callable[[object, jnp.ndarray], jnp.ndarray]
    prolong: Callable[[object, jnp.ndarray], jnp.ndarray]
    smooth: _Smoother
    mu: int = 1
    pre_smooth: bool = True


def _frozen_v_cycle(
    levels: tuple, b: jnp.ndarray, level_index: int, ops: _VCycleOps
) -> jnp.ndarray:
    """One V-cycle on a frozen COO-operator hierarchy (recursion unrolled at trace time).

    Shared by every frozen path — smoothed aggregation, its convection two-level variant, and lAIR:
    the operator matvec (:func:`_operator_matvec`) and the direct coarse solve are common, and the
    restriction, prolongation, and pre/post smoother come from the injected ``ops`` (:class:`_VCycleOps`).
    """
    level = levels[level_index]
    if level.coarse_inv is not None:  # coarsest: a direct (dense pseudo-inverse) solve
        return level.coarse_inv @ b

    if ops.pre_smooth:
        x = ops.smooth(level, b, jnp.zeros_like(b))
        residual = b - _operator_matvec(level, x)
    else:
        # No pre-relaxation, so the iterate is still zero and the residual is the right-hand side
        # itself: skipping the matvec here is exact, not an approximation.
        x = jnp.zeros_like(b)
        residual = b
    coarse_residual = ops.restrict(level, residual)
    coarse_error = _frozen_v_cycle(levels, coarse_residual, level_index + 1, ops)
    coarse = levels[level_index + 1]
    for _ in range(ops.mu - 1):
        if coarse.coarse_inv is not None:
            break  # the child solves exactly; visiting it again corrects nothing
        defect = coarse_residual - _operator_matvec(coarse, coarse_error)
        coarse_error = coarse_error + _frozen_v_cycle(levels, defect, level_index + 1, ops)
    x = x + ops.prolong(level, coarse_error)  # prolong and correct
    return ops.smooth(level, b, x)  # post-smooth


def _fixed_cycle_solve(levels: tuple, b: jnp.ndarray, cycles: int, ops: _VCycleOps) -> jnp.ndarray:
    """The outer driver shared by every frozen path: ``cycles`` V-cycles from a zero initial guess.

    Each pass corrects the current iterate by a V-cycle on the current residual, so with a frozen
    hierarchy and a fixed ``cycles`` the map ``b -> x`` is a constant linear operator — what makes it
    a valid frozen left preconditioner under plain GMRES, and what lets the adjoint transpose it.
    Only the level-local ``ops`` differ between the families.
    """
    x = jnp.zeros_like(b)
    for _ in range(cycles):
        residual = b - _operator_matvec(levels[0], x)
        x = x + _frozen_v_cycle(levels, residual, 0, ops)
    return x


def _smoothed_ops(smoother: _Smoother, mu: int = 1, pre_smooth: bool = True) -> _VCycleOps:
    """V-cycle ops for a smoothed-aggregation level: restrict by ``Pᵀ``, prolong by ``P`` (``R = Pᵀ``)."""
    return _VCycleOps(
        mu=mu,
        pre_smooth=pre_smooth,
        restrict=lambda level, r: _coo_apply(
            level.p_ccol, level.p_frow, level.p_val, r, level.n_coarse
        ),
        prolong=lambda level, e: _coo_apply(level.p_frow, level.p_ccol, level.p_val, e, level.n),
        smooth=smoother,
    )


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

    return hierarchy.fixed_cycle_solve(b, cycles, _smoothed_ops(smoother))


# --- nonsymmetric (convection-diffusion) smoothed aggregation ---------------------------
#
# The symmetric path above builds its hierarchy on a graph Laplacian, so a point smoother (Chebyshev)
# and constant-preserving aggregation resolve the smooth error. A momentum block with strong convection
# is a different operator: first-order upwind adds a **nonsymmetric** off-diagonal ``max(±mdot, 0)`` to
# the viscous coupling, and its error modes are advected along the flow, not smooth in the Laplacian
# sense — so a Laplacian-only AMG (even rescaled by the convective diagonal) is not Peclet-robust and
# stalls once the cell Peclet number grows.
#
# The convection-aware hierarchy is **two-level**: it aggregates the fine cells once and forms a single
# Galerkin coarse operator ``A_c = Pᵀ A P`` from the **true** ``A`` (aggregation and the smoothed
# prolongation use the **symmetric part** ``(A + Aᵀ)/2`` so the coarse space is well-shaped), then
# solves that coarse operator *directly*. The upwind convection-diffusion operator is a diagonally
# dominant M-matrix (positive diagonal, non-positive off-diagonals); on the **fine** level a single
# damping factor makes a **damped-Jacobi** smoother — matrix-free, a fixed linear operator, and safe on
# the operator's positive-real-part spectrum where a Chebyshev interval smoother is not — contract, so
# the two-level cycle is robust at high cell Peclet. It stays two-level on purpose: a coarser-still
# Galerkin operator acquires near-imaginary-axis eigenvalues that no single-factor damped-Jacobi
# smoother can damp, so the coarse level is an exact solve, and deeper coarsening is the job of the
# reduction-based lAIR hierarchy (:func:`build_air_hierarchy`) instead. The caller assembles the frozen
# operator (the upwind stencil is transport discretization, not coarsening); here it is coarsened and
# applied as a frozen matrix-free V-cycle, exactly like the symmetric path.


# The two levels this convection hierarchy builds: a single aggregation of the fine cells, then a
# direct (dense pseudo-inverse) solve on that one coarse level.
_CONVECTION_LEVELS = 2


def build_convection_hierarchy(
    a: sp.csr_matrix,
    *,
    omega_smooth: float = 2.0 / 3.0,
    max_coarse: int = 16,
    strength_threshold: float = 0.0,
    block_size: int = 1,
    max_levels: int = _CONVECTION_LEVELS,
    mis_aggregation: bool = False,
    aggressive_levels: int = 0,
    equilibrate: bool = False,
    prolongation_smoothing: str = "symmetric-part",
    orthonormal_prolongation: bool = False,
    avoid_singletons: bool = False,
) -> SmoothedHierarchy:
    """Build the convection-diffusion hierarchy for operator ``a`` — off the jit path.

    ``a`` is an assembled first-order-upwind convection-diffusion operator (viscous coupling plus the
    upwind convective off-diagonals); the symmetric part ``(A + Aᵀ)/2`` drives aggregation and
    prolongation smoothing while the single Galerkin coarse operator keeps the true nonsymmetric ``A``.

    This hierarchy is deliberately **two-level**: the fine cells are aggregated once and the resulting
    coarse operator is solved *directly* (a dense pseudo-inverse), with the damped-Jacobi smoother
    applied only on the fine level. On the fine level the operator is a diagonally dominant M-matrix,
    where a single damping factor contracts, so the two-level cycle is robust at high cell Peclet. A
    *deeper* Galerkin recursion is not built here: the coarse-of-coarse operators of a strongly
    convection-dominated problem acquire near-imaginary-axis eigenvalues that no single-factor
    damped-Jacobi smoother can damp (it becomes non-contractive), so the correct coarse level is an
    exact solve. For a hierarchy that coarsens all the way down and stays Peclet-robust, use the
    reduction-based :func:`build_air_hierarchy` (local approximate ideal restriction).

    Parameters
    ----------
    a : scipy.sparse matrix
        The frozen (nonsymmetric) convection-diffusion operator, shape ``(n_cells, n_cells)``.
    omega_smooth : float
        Prolongation-smoothing damping factor; the applied damping is ``omega_smooth * 2 / lambda_max``
        (``lambda_max`` of the symmetric part).
    max_coarse : int
        Stop coarsening once a level has at most this many **degrees of freedom**, and solve it
        directly there. Dofs, not cells: at ``block_size`` fields per cell the two differ by that
        factor, and the limit exists to bound a dense pseudo-inverse whose cost is cubic in the dof
        count.
    strength_threshold : float
        Strength-of-connection threshold for the aggregation (default ``0`` = isotropic aggregation on
        the full cell-adjacency graph, which reads no values from the operator at all). ``> 0`` aggregates only along strong connections
        (:func:`_aggregation_edges`) — the fix for an anisotropic / high-aspect-ratio operator; see
        :func:`build_smoothed_hierarchy` for the effect and the value-dependence caveat.
    block_size : int
        Degrees of freedom per cell. Above one the aggregation coarsens **cells** — the graph is
        collapsed onto cell connectivity and each field rides its own coarse unknown — and the level
        smoother inverts each cell's dense block instead of the scalar diagonal. Both are required
        together on a multi-field operator whose within-cell coupling exceeds its diagonal: a
        field-blind aggregation can merge one field's degree of freedom with another field's from a
        different cell, which manufactures a degenerate coarse row, and a point smoother discards the
        dominant coupling outright.
    max_levels : int
        Stop coarsening at this many levels and solve the last one directly.
    mis_aggregation : bool
        Aggregate by a randomly-ordered maximal independent set rather than the two-pass
        reverse-Cuthill--McKee pairing. The pairing only seeded an aggregate from a fully-free
        neighbourhood, so it seeded few and left most cells to a ragged cleanup pass.
    prolongation_smoothing : {"symmetric-part", "standard", "none"}
        How the tentative prolongation is smoothed. ``"standard"`` is the textbook
        ``P <- (I - 1.4/sigma_max(D^-1 A)) P_tent`` on the true operator and the scalar diagonal, and
        presumes a unit-magnitude diagonal, so pair it with ``equilibrate=True``. ``"none"`` keeps the
        tentative injection; on a badly scaled or strongly indefinite operator that is a real choice
        rather than a degenerate one, since the smoothing can degrade the coarse correction it exists
        to improve.
    aggressive_levels : int
        Aggregate over the squared graph ``G·G`` on this many leading levels. It exists to bound the
        level count as the mesh grows, not to improve the coarse space, and it over-coarsens a mesh
        that already reaches the coarse limit in one step.
    equilibrate : bool
        Coarsen ``D A D`` — the operator rescaled to a unit-magnitude diagonal — instead of ``a`` as
        given, carrying the factor on the hierarchy so ``b -> x`` is unchanged. Every step of the
        setup reads the diagonal (both spectral estimates, the smoother damping they scale, and the
        prolongation smoothing), so on an operator whose diagonal spans orders of magnitude they are
        calibrated against a scale with no meaning. Default ``False`` builds bit-identically to an
        unscaled build.

    Returns
    -------
    SmoothedHierarchy
        The frozen fine + direct-coarse levels for :func:`convection_multigrid_solve`.
    """
    # Nonsymmetric operator: aggregate and smooth the prolongation on the symmetric part ``(A + Aᵀ)/2``,
    # while the level stores the true operator's spectral radius for the damped-Jacobi smoother.
    return _build_aggregation_hierarchy(
        a.tocsr(),
        aggregation_operator=lambda m: (0.5 * (m + m.T)).tocsr(),
        omega_smooth=omega_smooth,
        max_coarse=max_coarse,
        max_levels=max_levels,
        strength_threshold=strength_threshold,
        block_size=block_size,
        mis_aggregation=mis_aggregation,
        aggressive_levels=aggressive_levels,
        equilibrate=equilibrate,
        prolongation_smoothing=prolongation_smoothing,
        orthonormal_prolongation=orthonormal_prolongation,
        avoid_singletons=avoid_singletons,
    )


def _apply_block_inverse(level: _SparseLevel, vector: jnp.ndarray) -> jnp.ndarray:
    """Apply the per-cell block inverse to a field-major vector: reshape, contract per cell, flatten."""
    per_cell = vector.reshape(level.block_size, -1).T  # (n_cells, block_size)
    return jnp.einsum("cij,cj->ci", level.block_inverse, per_cell).T.ravel()


def _jacobi_smooth(
    level: _SparseLevel,
    b: jnp.ndarray,
    x: jnp.ndarray,
    sweeps: int,
    omega: float,
    spectral_damping: bool = True,
) -> jnp.ndarray:
    """Damped-Jacobi smoother ``x <- x + alpha D^-1 (b - A x)`` (``sweeps`` times).

    Matrix-free and a fixed linear operator. With ``spectral_damping`` the relaxation is
    ``alpha = omega / lambda_max``, scaled by the per-level ``lambda_max`` (of the symmetric part) so
    ``omega`` in ``(0, 1]`` is a mesh- and scale-independent damping — the high-frequency-smoothing
    choice for the M-matrix convection-diffusion operator, where a Chebyshev interval smoother
    (assuming a real spectrum) is not safe.

    **Without it, ``alpha = omega`` is an absolute Richardson scale, and ``omega = 1`` is the
    undamped sweep ``x <- x + D^-1 (b - A x)``.** That is a real and sometimes much better choice, not
    a degenerate one: ``D^-1 A`` has unit diagonal blocks, so its eigenvalues average one and
    ``lambda_max >= 1`` always — spectral damping therefore *never* relaxes by more than ``omega``,
    and on an operator with an eigenvalue tail it relaxes by far less, under-smoothing every mode that
    is not the extreme one. Measured on the coupled turbulence block, the undamped sweep is worth 10
    cycles against 2 at four sweeps, closing the whole gap to an equivalently-configured PETSc GAMG.
    An undamped sweep is not a contraction on its own, which is why it is not the default; under a
    coarse correction and an outer Krylov it does not need to be.

    On a **nodal** level ``D`` is the per-cell dense block rather than the scalar diagonal, which is not
    a refinement but a requirement: where the within-cell coupling between fields exceeds the diagonal,
    a point method discards the dominant term and the sweep stops contracting. The per-cell solves are
    independent — a batched tiny contraction, no sequential dependency — so unlike the incomplete-LU
    sweep this is the same shape on an accelerator as on a host.
    """
    alpha = omega / level.lam_max if spectral_damping else omega
    if level.block_inverse is None:
        inv_diagonal = 1.0 / level.diagonal
        for _ in range(sweeps):
            x = x + alpha * inv_diagonal * (b - _operator_matvec(level, x))
        return x
    for _ in range(sweeps):
        x = x + alpha * _apply_block_inverse(level, b - _operator_matvec(level, x))
    return x


def convection_multigrid_solve(
    hierarchy: SmoothedHierarchy,
    b: jnp.ndarray,
    *,
    cycles: int = 1,
    sweeps: int = 2,
    omega: float = 0.8,
    spectral_damping: bool = True,
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
        The smoother's relaxation factor (static). Its meaning depends on ``spectral_damping``: a
        damping in ``(0, 1]`` relative to the level's ``lambda_max`` when that is set, or the absolute
        relaxation itself when it is not.
    spectral_damping : bool
        Scale the relaxation by the level's ``lambda_max`` (default). Setting it ``False`` makes
        ``omega`` an absolute relaxation, so ``omega = 1`` is the plain undamped sweep — which on a
        strongly nonsymmetric multi-field block is much the stronger smoother, because
        ``lambda_max >= 1`` always and dividing by it under-relaxes every mode but the extreme one.
        See :func:`_jacobi_smooth`.

    Returns
    -------
    jnp.ndarray
        The approximate solution ``x``, shape ``(n_cells,)``.
    """

    def smoother(level: _SparseLevel, rhs: jnp.ndarray, guess: jnp.ndarray) -> jnp.ndarray:
        return _jacobi_smooth(level, rhs, guess, sweeps, omega, spectral_damping)

    return hierarchy.fixed_cycle_solve(b, cycles, _smoothed_ops(smoother))


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


class _AirLevel(eqx.Module):
    """One lAIR level: the operator, its restriction and prolongation, and the C/F masks — all frozen.

    Unlike :class:`_SparseLevel` (which stores one prolongation and takes ``R = Pᵀ``), a reduction-based
    level carries an **independent** restriction ``R`` (fine → coarse) and prolongation ``P`` (coarse →
    fine), plus the fine/coarse masks the FC-Jacobi smoother relaxes over.

    Split static-index / dynamic-value on the same rule as :class:`_SparseLevel` (only the two counts
    that size a matvec are static). Note the **refreshability caveat**: unlike the aggregation path,
    lAIR's coarsening reads operator *values* (:func:`_strength_classical`), so re-deriving it at a new
    operator can legitimately change the C/F split and every shape — a reduction hierarchy is therefore
    **not** guaranteed refreshable on a fixed structure the way an aggregation one is.
    """

    n: int = eqx.field(static=True)  # cells at this level (sizes the matvec output)
    operator: _CsrOperator  # the level operator A, in CSR form and owning its matvec
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
    n_coarse: int = eqx.field(static=True)  # next-coarser cell count (0 on coarsest)


class AirHierarchy(eqx.Module):
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


def _coarse_index(split: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The coarse-point ids and the inverse map (global index -> coarse index, ``-1`` for F-points).

    The ``-1`` sentinel marks the F-points; an off-by-one in this map silently corrupts the
    interpolation / restriction sparsity, so it lives in exactly one place.
    """
    coarse = np.where(split == 1)[0]
    index = -np.ones(len(split), dtype=np.int64)
    index[coarse] = np.arange(len(coarse))
    return coarse, index


def _one_point_interpolation(a: sp.csr_matrix, split: np.ndarray) -> sp.csr_matrix:
    """One-point interpolation ``P``: each F-point takes its strongest C-neighbour; C-points injected."""
    a = a.tocsr()
    n = a.shape[0]
    coarse, coarse_index = _coarse_index(split)
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
    coarse, coarse_index = _coarse_index(split)
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
        f_nbrs = np.array(sorted(neighbourhood))
        a_ff = a[np.ix_(f_nbrs, f_nbrs)].toarray()
        rhs = np.asarray(a[g, f_nbrs].todense()).ravel()
        try:
            z = np.linalg.solve(a_ff.T, -rhs)
        except np.linalg.LinAlgError:
            z = np.linalg.lstsq(a_ff.T, -rhs, rcond=None)[0]
        rows.extend([ci] * len(f_nbrs))
        cols.extend(f_nbrs.tolist())
        vals.extend(z.tolist())
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(coarse), n))


def _air_level(a: sp.csr_matrix, split: np.ndarray, restriction, prolongation) -> _AirLevel:
    """Freeze a scipy operator, its C/F masks, and (optional) restriction/prolongation into JAX arrays."""
    coarsest = restriction is None
    r = None if coarsest else restriction.tocoo()
    p = None if coarsest else prolongation.tocoo()
    return _AirLevel(
        n=a.shape[0],
        operator=_CsrOperator.from_scipy(a),
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


def refresh_air_hierarchy(
    hierarchy: AirHierarchy, a: sp.csr_matrix, *, degree: int = 2
) -> AirHierarchy:
    """Re-derive an lAIR hierarchy's **values** at a new operator, reusing its frozen coarsening.

    A frozen preconditioner goes stale as the flow develops, and re-freezing it at the current state is
    a large win on the scalar transport blocks. Simply calling :func:`build_air_hierarchy` again would
    work numerically but is a *different* hierarchy: lAIR's coarsening reads operator **values** (the
    strength graph in :func:`_strength_classical`, and the strongest-C-neighbour choice in
    :func:`_one_point_interpolation`), so a rebuild generally changes the C/F split and every shape
    below the first level or two — a new compilation signature, so the refreshed preconditioner would
    force a recompile of the solve it accelerates.

    This instead holds the **structural** decisions fixed and recomputes only the numbers: each level
    reuses its stored C/F split and its stored prolongation, and re-solves the local approximate-ideal
    restriction against ``a``. That is legitimate because *any* valid C/F split gives a valid
    preconditioner — freezing the reference's split trades a possibly better-adapted coarse space for a
    refresh that costs no recompile. The restriction's sparsity depends only on the split and on ``a``'s
    *pattern* (the degree-``d`` neighbourhood walk), both unchanged, so every level's shapes — and hence
    the Galerkin ``R A P`` patterns below it — are invariant by construction; this is checked before
    returning.

    Parameters
    ----------
    hierarchy : AirHierarchy
        The hierarchy whose coarsening is reused (built by :func:`build_air_hierarchy`).
    a : scipy.sparse matrix
        The new fine operator. Must have the same sparsity pattern as the one ``hierarchy`` was built
        from (same mesh graph), which is what makes the structure invariant.
    degree : int
        The restriction neighbourhood degree; must match the one used to build ``hierarchy``.

    Returns
    -------
    AirHierarchy
        A hierarchy with identical static metadata and array shapes, carrying values from ``a``.

    Raises
    ------
    ValueError
        If the refreshed structure does not match ``hierarchy`` — which means the assumption above was
        violated (a different mesh graph, or a mismatched ``degree``).
    """
    a = a.tocsr()
    if a.shape[0] != hierarchy.levels[0].n:
        raise ValueError(
            f"refresh_air_hierarchy: operator has {a.shape[0]} rows but the hierarchy's finest level "
            f"has {hierarchy.levels[0].n}. A refresh reuses the frozen coarsening, so the operator "
            "must come from the same mesh graph."
        )
    levels: list[_AirLevel] = []
    for index, level in enumerate(hierarchy.levels):
        _require_positive_diagonal(a.diagonal(), f"refresh_air_hierarchy (level {index})")
        if level.r_row is None:  # coarsest: a direct solve, no transfers to rebuild
            levels.append(_air_level(a, np.ones(a.shape[0], dtype=np.int64), None, None))
            break
        # Reuse the frozen coarsening: the C/F split (from the stored masks) and the prolongation
        # (whose column choice is value-dependent, so it must be carried over rather than re-derived).
        split = np.asarray(level.c_mask).astype(np.int64)
        prolongation = sp.csr_matrix(
            (
                np.asarray(level.p_val),
                (np.asarray(level.p_row), np.asarray(level.p_col)),
            ),
            shape=(level.n, level.n_coarse),
        )
        restriction = _lair_restriction(a, split, degree)  # same pattern, values from `a`
        levels.append(_air_level(a, split, restriction, prolongation))
        a = (restriction @ a @ prolongation).tocsr()

    refreshed = AirHierarchy(tuple(levels))
    _require_matching_structure(hierarchy, refreshed, "refresh_air_hierarchy")
    return refreshed


def _require_matching_structure(original, refreshed, where: str) -> None:
    """Reject a refresh that changed any shape — the property the no-recompile refresh depends on."""
    if len(original.levels) != len(refreshed.levels):
        raise ValueError(
            f"{where}: refreshed hierarchy has {len(refreshed.levels)} levels, not "
            f"{len(original.levels)}. The operator's sparsity pattern must match the one the "
            "hierarchy was built from (same mesh graph), and `degree` must match the build."
        )
    for i, (old, new) in enumerate(zip(original.levels, refreshed.levels, strict=True)):
        if (old.n, old.n_coarse) != (
            new.n,
            new.n_coarse,
        ) or old.operator.data.shape != new.operator.data.shape:
            raise ValueError(
                f"{where}: level {i} changed shape — (n, n_coarse, nnz) "
                f"{(old.n, old.n_coarse, old.operator.data.shape[0])} -> "
                f"{(new.n, new.n_coarse, new.operator.data.shape[0])}. The refreshed values would be a new "
                "compilation signature, defeating the purpose; check that `a` has the same sparsity "
                "pattern and that `degree` matches the build."
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
        _require_positive_diagonal(a.diagonal(), f"build_air_hierarchy (level {len(levels)})")
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
        x = x + omega * level.f_mask * inv_diagonal * (b - _operator_matvec(level, x))
    for _ in range(c_iters):
        x = x + omega * level.c_mask * inv_diagonal * (b - _operator_matvec(level, x))
    return x


def _air_ops(f_iters: int, c_iters: int, omega: float) -> _VCycleOps:
    """V-cycle ops for a reduction (lAIR) level: an independent restriction ``R != Pᵀ`` and an
    FC-Jacobi smoother (the reduction analogue of :func:`_smoothed_ops`)."""
    return _VCycleOps(
        restrict=lambda level, r: _coo_apply(
            level.r_row, level.r_col, level.r_val, r, level.n_coarse
        ),
        prolong=lambda level, e: _coo_apply(level.p_row, level.p_col, level.p_val, e, level.n),
        smooth=lambda level, b, x: _fc_jacobi(level, b, x, f_iters, c_iters, omega),
    )


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
    return _fixed_cycle_solve(hierarchy.levels, b, cycles, _air_ops(f_iters, c_iters, omega))
