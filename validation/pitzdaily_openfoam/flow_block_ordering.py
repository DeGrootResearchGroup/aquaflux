"""Can an elimination ORDERING make a zero-fill incomplete-LU work on the SPLIT flow block?

**This is the correctly-scoped version of the question ``ilu0_remedy_sweep.py`` asks monolithically.**
That sweep builds one V-cycle over all five fields interleaved cell-major, and the case does not do
that: it splits, sending the ``[u, v, p]`` saddle to the algebraic-multigrid V-cycle whose level
smoother is the incomplete factorization, and the ``[k, omega]`` pair to a nodal hierarchy that is not
an incomplete factorization at all. **No ``k`` or ``omega`` row is ever eliminated by an incomplete
factorization in the shipped solver**, so every ordering result taken monolithically is a property of a
five-field factorization that nothing uses. Everything here is measured on ``[u, v, p]`` alone, taken
from the assembled operator by the very :class:`~aquaflux.solve.FieldGroups` the split uses.

**Why ordering, and why it is the lever worth asking about.** A zero-fill incomplete factorization keeps
exactly the operator's own entries and discards every fill the elimination would create. *Which* entries
that discards is decided entirely by the order the unknowns are eliminated in, so the ordering is part
of the factorization rather than a detail of it — far more so than for a factorization with fill, which
can recover some of what a bad order costs it. On this project's other case an ordering change took a
zero-fill smoother from outright failure to a single cycle.

**The two things measured, and why both.** The failure recorded for this block is *stationary-sweep
amplification* — the factor's entries are unremarkable (max magnitude in the hundreds, not the 1e+23 of
a broken threshold factorization) and yet applying it as a smoother grows the residual. So:

* **the stationary sweep** ``x <- x + M^-1 (b - A x)`` from a zero iterate, reported as the TRUE
  relative residual after each of several sweeps. This is exactly what the V-cycle does with the
  factorization, and a smoother that amplifies is visible here in one sweep and nowhere cheaper;
* **restarted GMRES right-preconditioned by the factorization alone**, judged on the TRUE residual
  recomputed from the returned solution. A factorization can be a serviceable Krylov preconditioner and
  a useless stationary smoother, so this measures preconditioner quality proper, and the pair
  distinguishes "a weak factorization" from "an amplifying one".

A pivot census is printed beside both and is a **proxy**: this project records a census that came back
identical across arms whose cycle counts differed five-fold, so it never decides anything here.

**States.** Both are measured, because they are different questions and the failure lives at one of them:

* the case's own **self-start**, which is where the ``hostilu`` march fails at step 1 — the operating
  point the failure under investigation actually occupies;
* the **converged root**, which is the operator a ``jax.grad`` meets: the shift has vanished by
  construction there, so zero shift is the adjoint's operating point and no march measurement speaks
  to it. It is the **latest** ``checkpoints/state-*.npz`` — a run artifact written by a full march, not
  a checked-in fixture, and ``checkpoints/`` is untracked. Regenerate it by running the case
  (``PITZ_FLOW_INVERSE=native validation/run_case.sh validation/pitzdaily_openfoam/compare.py``).
  Absent, this reports the self-start half only and says so, rather than quietly measuring something
  else.

⚠️ **Whichever march wrote that checkpoint chose its own stencil reach, and this re-probes at reach 5
regardless.** A SIMPLE-smoothed block preconditioner, for instance, is reach-insensitive -- it relaxes through
diagonal and Schur approximations and inherits no sparsity pattern, so a short reach is merely a
slightly wrong operator to it -- which is why the run that produced this state could afford reach 3. An
incomplete factorization is the opposite: it takes its pattern from the stored entries, so a corrupted
pattern gives a corrupted factor. The **state** is unaffected either way, being a root of the exact
residual (the Krylov operator is the exact Jacobian-vector product; only the preconditioner is built
from the materialized matrix), but the **matrix** every arm below is ranked on must be the exact one.

Run from the repo root::

    validation/run_case.sh validation/pitzdaily_openfoam/flow_block_ordering.py
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
import ilu0_remedy_sweep as monolithic  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.csgraph as csgraph  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    COMPILED,
    FieldGroups,
    Ilu0,
    shifted_jacobian,
    symmetrically_equilibrate,
)
from aquaflux.turbulence.coupled import (  # noqa: E402
    _DEFAULT_SHIFT_BASIS,
    _frozen_shift_diagonal,
    _monolithic_shift_source,
)

#: Pseudo-transient shifts. ``0.5`` is the march's own start, ``0.05`` the preconditioner's shipped
#: floor, ``0.0`` the adjoint's operator. The march's failure and the adjoint's operator are different
#: shifts and an arm is allowed to win at one and lose at another -- that is the recorded behaviour of
#: every arm tried so far, and reporting a single shift is how that gets missed.
BETAS = (0.5, 0.05, 0.0)

#: Stencil reach of the materialized Jacobian. This case needs 5 for a matrix matching the
#: Jacobian-vector product to the float64 floor; reach 3 leaves a 2e-07 relative error carried by the
#: pressure column. An incomplete factorization inherits the stored pattern, so it is the consumer that
#: a corrupted pattern hurts most -- which is exactly why the arms are compared at the exact reach.
REACH = 5

#: Stationary sweeps to report the amplification over. Four is the shipped smoother count on this block.
SWEEP_COUNTS = (1, 2, 4)

#: Report each arm's factorization and pivot census and SKIP the Krylov solve
#: (``FLOW_BLOCK_CENSUS_ONLY=1``). The census is cheap and the solve is not, so a run that only needs
#: to re-ask the pivot question should not pay for verdicts that are already recorded.
CENSUS_ONLY = os.environ.get("FLOW_BLOCK_CENSUS_ONLY", "") not in ("", "0")

#: Krylov budget for the preconditioner-quality arm. Generous enough that arms separate rather than tie,
#: capped so a failing arm costs a bounded amount -- a failure is identified by its true residual long
#: before it would converge.
RTOL, RESTART, MAX_RESTARTS = 1e-8, 30, 20


# --------------------------------------------------------------------------------------------------
# Elimination orderings. Each returns the field-major degree-of-freedom indices in ELIMINATION ORDER,
# so `A[perm][:, perm]` is the matrix the factorization walks top to bottom.
#
# The block is stored FIELD-MAJOR: degree of freedom `(cell i, field f)` sits at `f * n_cells + i`, so
# a dof's cell is `index % n_cells` and its field is `index // n_cells`. Every ordering below is built
# from that one fact rather than re-deriving the layout.
# --------------------------------------------------------------------------------------------------


def _cell_of(index: np.ndarray, n_cells: int) -> np.ndarray:
    """The cell each field-major degree of freedom belongs to."""
    return index % n_cells


def _interleave(cells: np.ndarray, fields: tuple[int, ...], n_cells: int) -> np.ndarray:
    """``fields`` of each cell in ``cells``, cell by cell — the cell-major layout over a cell order."""
    return (np.asarray(fields)[None, :] * n_cells + cells[:, None]).ravel()


def cell_major(matrix, n_fields, n_cells):
    """The SHIPPED ordering: every field of a cell together, cells in their natural order.

    The control every other arm is read against. Its stated rationale is that interleaving keeps each
    pressure unknown among the velocity unknowns of its own cell, so the elimination never divides by a
    lone continuity diagonal carrying only the Rhie--Chow damping.
    """
    return _interleave(np.arange(n_cells), tuple(range(n_fields)), n_cells)


def field_major(matrix, n_fields, n_cells):
    """The RAW storage order — all of ``u``, then all of ``v``, then all of ``p``.

    Worth an arm despite being expected to fail, because it is the crudest reading of "eliminate the
    velocities before the pressure" and it fixes the scale the deliberate pressure-last arm is read on.
    """
    return np.arange(n_fields * n_cells)


def pressure_last(matrix, n_fields, n_cells):
    """Velocities cell-major first, then every pressure unknown — the saddle-point literature's order.

    Konshin, Olshanskii and Vassilevski (SISC 37(5), 2015) eliminate the velocity unknowns before the
    pressure ones, which is the opposite of the shipped interleave. ⚠️ **The prediction here is that it
    fails at ZERO fill and the reason is structural**: what makes pressure-last work is that eliminating
    the velocities fills the pressure block with the Schur complement, and zero fill discards precisely
    that fill. The pressure block is then eliminated against its own bare, near-singular diagonal. The
    arm is measured rather than argued, but read a failure as a confirmation of that mechanism.
    """
    velocity = _interleave(np.arange(n_cells), tuple(range(n_fields - 1)), n_cells)
    return np.concatenate([velocity, (n_fields - 1) * n_cells + np.arange(n_cells)])


def pressure_first(matrix, n_fields, n_cells):
    """Every pressure unknown first, then the velocities cell-major — pressure-last reversed.

    The companion control: if the ordering's effect ran through the pressure block's position alone,
    these two would bracket the shipped interleave from either side.
    """
    velocity = _interleave(np.arange(n_cells), tuple(range(n_fields - 1)), n_cells)
    return np.concatenate([(n_fields - 1) * n_cells + np.arange(n_cells), velocity])


def _cell_graph(matrix: sp.csr_matrix, n_cells: int) -> sp.csr_matrix:
    """The cell-to-cell adjacency underlying a field-major block, as a binary pattern.

    Every field couples cell ``i`` to cell ``j`` in the same graph, so the block pattern is collapsed by
    mapping each degree of freedom to its cell and deduplicating. Built from the pattern alone — the
    values play no part in an adjacency — and symmetrized, since the orderings below want an undirected
    graph and this operator is not symmetric.
    """
    coo = matrix.tocoo()
    collapsed = sp.coo_matrix(
        (
            np.ones(coo.nnz, dtype=np.int8),
            (_cell_of(coo.row, n_cells), _cell_of(coo.col, n_cells)),
        ),
        shape=(n_cells, n_cells),
    ).tocsr()
    collapsed.data[:] = 1
    return collapsed + collapsed.T


def cell_major_rcm(matrix, n_fields, n_cells):
    """Cell-major over cells in reverse Cuthill--McKee order of the CELL graph.

    Bandwidth-reducing, and applied at cell granularity so each cell's fields stay adjacent — which is
    what keeps it a re-ordering of the shipped arm rather than a different family. The pointwise variant
    below is the one that does not preserve the cell blocks, and the pair separates "the cell order
    matters" from "the within-cell grouping matters".
    """
    order = csgraph.reverse_cuthill_mckee(_cell_graph(matrix, n_cells).tocsr(), symmetric_mode=True)
    return _interleave(np.asarray(order), tuple(range(n_fields)), n_cells)


def cell_major_rowlength(matrix, n_fields, n_cells):
    """Cell-major over cells in ascending block-row-length order.

    The cell-granular form of the ordering that, on this project's monolithic sweep at one shift, took a
    zero-fill factorization from outright failure to a single cycle. Eliminating the sparsest rows first
    is the classic minimal-fill heuristic; with zero fill there is no fill to minimize, but the sparse
    rows are also the ones whose elimination discards least.
    """
    lengths = np.diff(_cell_graph(matrix, n_cells).tocsr().indptr)
    return _interleave(np.argsort(lengths, kind="stable"), tuple(range(n_fields)), n_cells)


def pointwise_rcm(matrix, n_fields, n_cells):
    """Reverse Cuthill--McKee on the CELL-MAJOR matrix pointwise, cell blocks not preserved."""
    base = cell_major(matrix, n_fields, n_cells)
    reordered = matrix[base][:, base].tocsr()
    pattern = reordered + reordered.T
    return base[np.asarray(csgraph.reverse_cuthill_mckee(pattern.tocsr(), symmetric_mode=True))]


def pointwise_rowlength(matrix, n_fields, n_cells):
    """Ascending row length on the CELL-MAJOR matrix pointwise — the closest analogue of the arm that
    won the monolithic sweep, which PETSc applies per row rather than per cell block."""
    base = cell_major(matrix, n_fields, n_cells)
    reordered = matrix[base][:, base].tocsr()
    return base[np.argsort(np.diff(reordered.indptr), kind="stable")]


def _defer_small_diagonal(matrix, n_fields, n_cells, quantile):
    """Cell-major, with the rows carrying the smallest RAW diagonal magnitudes moved to the end.

    The static half of the deferring in HILUCSI (Chen, Ghai and Jiao, arXiv:1911.10139): symmetrically
    permute the rows an elimination is most likely to break on into the lower-right corner, where they
    are handled last and against a matrix the rest of the elimination has already conditioned.

    ⚠️ **The criterion is taken PRE-equilibration and that is not optional.** The symmetric square-root
    equilibration divides every row by ``sqrt(|diag|)``, which forces every nonzero diagonal to magnitude
    exactly one — so "the rows with a small diagonal" is not a question the equilibrated matrix can
    answer at all, and a criterion read off it would select on floating-point noise.
    """
    base = cell_major(matrix, n_fields, n_cells)
    magnitude = np.abs(matrix.diagonal())[base]
    cut = np.quantile(magnitude, quantile)
    deferred = magnitude <= cut
    return np.concatenate([base[~deferred], base[deferred]])


def defer_small_diagonal_1pct(matrix, n_fields, n_cells):
    """Deferring the smallest 1 % of raw diagonals."""
    return _defer_small_diagonal(matrix, n_fields, n_cells, 0.01)


def defer_small_diagonal_5pct(matrix, n_fields, n_cells):
    """Deferring the smallest 5 % of raw diagonals."""
    return _defer_small_diagonal(matrix, n_fields, n_cells, 0.05)


def mc64_symmetric(matrix, n_fields, n_cells):
    """A maximum-product bipartite matching used as a SYMMETRIC permutation.

    The permutation half of MC64: match rows to columns maximizing the product of the matched entries
    (equivalently, a minimum-weight full matching on ``-log|a_ij|``), which puts large entries on the
    diagonal. Applied symmetrically (``P_r = P_c``) as HILUCSI does, so it permutes the operator rather
    than only its rows and the cell-block structure survives as a relabelling.

    ⚠️ **PERMUTATION ONLY — this is not MC64.** The method's other half is the pair of dual potentials
    that rescale the matched matrix to have unit diagonal and off-diagonals bounded by one, and ``scipy``
    does not expose them. Read this arm as "does putting large entries on the diagonal help", not as a
    measurement of MC64.
    """
    base = cell_major(matrix, n_fields, n_cells)
    reordered = matrix[base][:, base].tocoo()
    keep = reordered.data != 0.0
    weights = sp.coo_matrix(
        (-np.log(np.abs(reordered.data[keep])), (reordered.row[keep], reordered.col[keep])),
        shape=reordered.shape,
    ).tocsr()
    _, columns = csgraph.min_weight_full_bipartite_matching(weights)
    return base[np.asarray(columns)]


def cell_major_reversed(matrix, n_fields, n_cells):
    """Cell-major over cells in reverse natural order — the control for "does ANY relabelling matter".

    A convection-dominated operator is directional, so simply walking the mesh the other way is a real
    change to a zero-fill elimination even though it reduces no bandwidth and defers no pivot. If this
    moves the result as much as the considered arms do, the considered arms are not measuring what their
    names claim.
    """
    return _interleave(np.arange(n_cells)[::-1], tuple(range(n_fields)), n_cells)


#: The sweep. `cell_major` is the shipped control and must stay first: every other arm is read against
#: it, and a subset selected on the command line that drops it has nothing to be read against.
ORDERINGS = (
    cell_major,
    field_major,
    pressure_last,
    pressure_first,
    cell_major_rcm,
    cell_major_rowlength,
    pointwise_rcm,
    pointwise_rowlength,
    defer_small_diagonal_1pct,
    defer_small_diagonal_5pct,
    mc64_symmetric,
    cell_major_reversed,
)


def stationary_sweeps(factors, operator, permutation, scale, b, counts):
    """TRUE relative residuals after each of ``counts`` stationary sweeps from a zero iterate.

    ``x <- x + M^-1 (b - A x)`` in the caller's own field-major, unequilibrated space: the scaling and
    the permutation are undone around each application exactly as the shipped smoother does, so what is
    measured is the smoother as the V-cycle would apply it and not a quantity in some interior space.

    This is the V-cycle's actual dependency. A residual that GROWS here is the recorded failure mode on
    this block, and it is visible in a single sweep.
    """
    reported, norm_b = {}, float(np.linalg.norm(b))
    x = np.zeros_like(b)
    for sweep in range(1, max(counts) + 1):
        residual = b - operator @ x
        correction = np.empty_like(residual)
        correction[permutation] = factors.solve((scale * residual)[permutation])
        x = x + scale * correction
        if sweep in counts:
            reported[sweep] = float(np.linalg.norm(b - operator @ x)) / norm_b
        if not np.all(np.isfinite(x)):  # an amplifying smoother reaches this in a few sweeps
            for remaining in counts:
                reported.setdefault(remaining, float("inf"))
            break
    return reported


def krylov(factors, operator, permutation, scale, b):
    """Restarted GMRES right-preconditioned by the factorization alone: applies, and the TRUE residual.

    Right-preconditioned, so the Krylov residual is already the true one; the true relative residual is
    recomputed from the returned solution regardless, because a preconditioned-norm "win" has been
    recorded on this project and was an artifact.
    """
    applies = [0]

    def apply(v):
        applies[0] += 1
        out = np.empty_like(v)
        out[permutation] = factors.solve((scale * np.asarray(v, dtype=np.float64))[permutation])
        return scale * out

    m = spla.LinearOperator(operator.shape, matvec=apply, dtype=np.float64)
    x, _ = spla.gmres(operator, b, M=m, rtol=RTOL, restart=RESTART, maxiter=MAX_RESTARTS)
    true = float(np.linalg.norm(operator @ x - b) / np.linalg.norm(b))
    return applies[0], true


def run_ordering(ordering, block, n_fields, n_cells, b):
    """Factorize the flow block under one ordering and report both measurements plus the census."""
    began = time.perf_counter()
    try:
        permutation = np.asarray(ordering(block, n_fields, n_cells), dtype=np.int64)
        equilibrated, scale = symmetrically_equilibrate(block)
        reordered = equilibrated[permutation][:, permutation].tocsr()
        reordered.sort_indices()
        factors = Ilu0(reordered)
    except Exception as failure:
        print(
            f"  {ordering.__name__:<26} FACTORIZATION FAILED  {type(failure).__name__}: "
            f"{str(failure)[:70]}",
            flush=True,
        )
        return
    build = time.perf_counter() - began

    sweeps = stationary_sweeps(factors, block, permutation, scale, b, SWEEP_COUNTS)
    began = time.perf_counter()
    applies, true = (0, float("nan")) if CENSUS_ONLY else krylov(factors, block, permutation, scale, b)
    solve = time.perf_counter() - began

    # ⚠️ The FACTOR's pivots, not the operator's diagonal. Reading `reordered.diagonal()` here reported
    # `min|piv| 1.00` for every arm at every shift -- which is not a finding but an artifact: the
    # symmetric square-root equilibration forces exactly that, so the census was measuring the
    # conditioning transform rather than the factorization. Still a PROXY beside the verdict, never
    # instead of it.
    pivots = factors.pivots
    swept = "  ".join(f"x{n} {sweeps[n]:.2e}" for n in SWEEP_COUNTS)
    verdict = "census" if CENSUS_ONLY else ("converged" if true <= 1e-6 else "FAILED")
    print(
        f"  {ordering.__name__:<26} build {build:5.1f}s  sweep {swept}  |  "
        f"gmres {applies:4d} TRUE {true:.2e} {verdict:9} {solve:5.1f}s  |  "
        f"neg {int((pivots < 0).sum()):5d} min|piv| {np.abs(pivots).min():.2e}",
        flush=True,
    )


def states(coupled):
    """The states to measure at: the case's self-start, and the converged root if one is checked in.

    Both are gated on a finite residual before anything is measured. A state packed the wrong way — a
    physical ``omega`` where this case solves ``log(omega)`` — reads finite in every field while the
    residual is silently NaN, after which every factorization fails in its own idiom and invites a
    confident, completely wrong story about the method.
    """
    found = [("self-start", monolithic.load_state(coupled, None))]
    # The LATEST checkpoint, never a hard-coded filename. The case checkpoints on a rolling keep-N, so
    # naming one pins this harness to a file the next march deletes -- which happened, and cost the
    # converged-root half of a sweep. Highest step number wins; the directory is untracked, so a fresh
    # clone simply has none.
    saved = sorted((CASE / "checkpoints").glob("state-*.npz"))
    root = saved[-1] if saved else CASE / "checkpoints" / "(none)"
    if saved:
        state = jnp.asarray(np.load(root)["state"])
        if not bool(jnp.all(jnp.isfinite(coupled.residual(state)))):
            raise SystemExit(
                f"the residual at {root} is not finite; nothing here would mean anything."
            )
        found.append((f"converged root ({root.name})", state))
    else:
        print(f"no converged root at {root}; measuring the self-start only.", flush=True)
    return found


def main() -> None:
    """Every ordering, at every shift, at every available state.

    A subset of orderings may be named on the command line or in ``FLOW_BLOCK_ORDERINGS`` (the case
    runner launches a script with no arguments, so without the environment form a two-arm follow-up
    silently reruns the whole sweep).
    """
    wanted = sys.argv[1:] or os.environ.get("FLOW_BLOCK_ORDERINGS", "").split()
    orderings = [o for o in ORDERINGS if not wanted or any(w in o.__name__ for w in wanted)]

    coupled = compare.build_case()["coupled"]
    n_cells = coupled.layout.n_cells
    n_flow = coupled.layout.dim + 1
    groups = FieldGroups(n_cells, n_flow, coupled.layout.dim + 3 - n_flow)
    print(
        f"pitzDaily flow block [u, v, p]: {groups.n_leading_dofs} of {groups.n_dofs} dofs over "
        f"{n_cells} cells, reach {REACH}, Ilu0 COMPILED={COMPILED}\n"
        f"the split's OWN partition ({n_flow} leading fields); [k, omega] goes to the nodal hierarchy "
        f"and is NOT factorized, so it is absent here by design.",
        flush=True,
    )

    for name, state in states(coupled):
        source = _monolithic_shift_source(coupled, state, _DEFAULT_SHIFT_BASIS)
        full_rhs = -np.asarray(coupled.residual(state), dtype=np.float64)
        # One cache per STATE, dropped when this state is done: the Jacobian depends on the state as
        # much as on the reach, and one materialized matrix at a time is what this machine has room for.
        matrix, _ = monolithic.jacobian(coupled, state, REACH, {})
        for beta in BETAS:
            shift = np.asarray(_frozen_shift_diagonal(source, beta, state), dtype=np.float64)
            block = groups.blocks(shifted_jacobian(matrix, shift))[0]
            b = full_rhs[groups.leading]
            print(
                f"\n{'=' * 118}\n{name} | beta {beta} (operator AND factorization matched, no "
                f"preconditioner-only floor) | block nnz {block.nnz / 1e6:.2f} M | "
                f"|rhs_flow| {np.linalg.norm(b):.4e}",
                flush=True,
            )
            for ordering in orderings:
                run_ordering(ordering, block, n_flow, n_cells, b)
            del block
            gc.collect()
        del matrix
        gc.collect()


if __name__ == "__main__":
    main()
