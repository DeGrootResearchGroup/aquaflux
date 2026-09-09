"""Does a point-implicit step descend at a cold start where no pseudo-transient shift does?

A Reynolds-number ladder exists here for one measured reason: a cold solve at the target Reynolds
number **fails at its first step for every shift strength tried** -- no ``beta`` makes step 1 descend.
That is a statement about one family of directions, the shifted Newton step
``-(J + beta D)^-1 R``, searched over ``beta`` and over step length. It is not a statement about every
direction.

Mavriplis (Computers and Fluids 220:104859, 2021) observes that where a pseudo-transient Newton solver
stagnates it is effectively taking explicit local time steps -- among the weakest ways to advance a
nonlinear problem -- while ordinary *local nonlinear* solvers (point-implicit, Gauss-Seidel, line)
converge on the same states without difficulty. His residual-smoothing scheme is built so the
small-pseudo-timestep limit becomes ``dw = -D^-1 R`` for a local operator ``D`` instead of an explicit
step, while the large limit still recovers Newton.

So the question this asks is the cheapest possible version of "could that remove the ladder": at the
cold initialization, at the true target viscosity, **is there a descent direction that the shifted
Newton family does not contain?** It compares, at one state and with no march:

``explicit``
    ``-R/V``, the small-pseudo-timestep limit of the shipped continuation -- the direction the solver
    actually takes when its shift is driven up. The baseline to beat.
``point-implicit``
    ``-D^-1 R`` with ``D`` the per-cell diagonal block of the true Jacobian, which is Mavriplis's
    small-timestep limit and the whole hypothesis.
``row-scaled``
    ``-R/d`` on the pseudo-transient shift diagonal -- a scalar stand-in for the block, included to
    separate "a local *block* solve is what matters" from "any local rescaling would do".
``strong-block``
    ``-D^-1 R`` with ``D`` the Jacobian restricted to **aggregates of strongly connected cells**. This
    is the arm that matters, because Mavriplis's smoother is *line*-preconditioned and a line through an
    anisotropic near-wall layer is exactly a chain of strong couplings -- so an algebraic
    strength-of-connection grouping is the same object, not an analogy. A per-cell block cannot see the
    wall-normal coupling that a line is built to invert, so a negative result from the cell-block arm
    alone would not settle the question.

Each direction is line-searched over the same ladder of step lengths and scored in the row-equilibrated
measure the march itself stops on. **A direction that cannot descend here kills the hypothesis
cheaply.** One that does descend does *not* establish that a march reaches the root -- the ladder may
be buying basin membership rather than descent -- but it is the necessary condition, and it costs one
Jacobian materialization instead of a re-march.

Usage -- a case directory, and optionally the viscosity scale to pose the question at::

    python3 validation/point_implicit_step.py pitzdaily_openfoam
    python3 validation/point_implicit_step.py pitzdaily_openfoam 10
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

VALIDATION = Path(__file__).resolve().parent
sys.path.insert(0, str(VALIDATION.parent))

#: Step lengths tried along every direction, logarithmic over many decades.
#:
#: ⚠️ **The range has to be this wide, and a linear ladder is useless here.** The directions carry wildly
#: different magnitudes -- ``-R/V`` divides by a cell volume of order 1e-8 on this mesh, so even a step
#: length of 1e-3 is an enormous move and every trial state is non-finite. Scoring them on a common
#: ladder would then measure the scaling rather than the direction, which is not the question. Sweeping
#: decades and reporting each direction's own best makes the comparison about direction quality; the
#: step size that achieved it is reported beside it so a direction that only works when annihilated is
#: still visible as such.

ALPHAS = tuple(10.0**-e for e in range(15))

#: The probe reach the Jacobian is materialized at. Only the per-cell diagonal block is read out of it,
#: which every reach recovers exactly, so this trades materialization cost for nothing here.
REACH = 3

#: Strength-of-connection threshold for the line-like arm: a connection counts as strong when it is at
#: least this fraction of the row's largest off-diagonal. In an anisotropic near-wall layer the
#: wall-normal coupling dominates, so a high threshold selects chains along it -- which is what a line
#: solver inverts.
STRENGTH = 0.25


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: point_implicit_step.py <case> [viscosity-scale]")
    case, scale = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    directory = VALIDATION / case
    sys.path.insert(0, str(directory))
    compare = importlib.import_module("compare")

    import jax.numpy as jnp
    from aquaflux.solve import MonolithicAmgPreconditioner, block_stencil_gather_map
    from aquaflux.turbulence import hybrid_initialize
    from aquaflux.turbulence.coupled import (
        _PROBE_BATCH_SIZE,
        _batched_jacobian_matvec,
        _coupled_jacobian_plan,
        _coupled_shift_policy,
        _jacobian_matvec,
        coupled_scaled_norm,
    )

    coupled = compare.build_case()["coupled"].with_scaled_molecular_viscosity(scale)
    flow, k, omega = hybrid_initialize(coupled.momentum, coupled.turbulence)
    state = coupled.state_from_physical(flow, k, omega)

    policy = _coupled_shift_policy(coupled, state, None, build_flow_block=False)
    measure = coupled_scaled_norm(coupled, policy, state)
    residual = coupled.residual(state)
    base = float(measure(residual))

    n, dim = coupled.layout.n_cells, coupled.layout.dim
    fields = dim + 3
    print(f"[{case}] cold hybrid initialization at viscosity scale {scale:g}")
    print(f"  {n} cells x {fields} fields, |R| = {base:.4e} in the march's own measure\n")

    # The per-cell diagonal block of the true Jacobian, via the same coloured probe the preconditioner
    # is built from -- so `D` is the operator the solver actually meets, not a model of it.
    plan = _coupled_jacobian_plan(coupled, REACH)
    structure = block_stencil_gather_map(plan)
    jacobian = MonolithicAmgPreconditioner._materialize_jacobian(
        lambda v: _jacobian_matvec(coupled, state, v),
        plan,
        lambda seeds: _batched_jacobian_matvec(coupled, state, seeds),
        _PROBE_BATCH_SIZE,
        structure,
    )
    csr, r = jacobian.tocsr(), np.asarray(residual)

    def block_solve_at(groups, rhs):
        """``-D^-1 R`` with ``D`` the Jacobian restricted to each group of cells, all fields.

        ⚠️ **The coupled state is FIELD-major** -- field ``f`` of cell ``c`` is index ``f*n + c``, which
        is what the ``k`` block's ``((dim+1)n, (dim+2)n)`` slice says -- and ``_materialize_jacobian``
        returns the Jacobian in that same raw ordering, because the cell-major permutation is applied
        *around* the V-cycle rather than baked into the matrix. Reading per-cell blocks with
        ``tobsr(blocksize=(fields, fields))`` therefore groups ``fields`` consecutive **cells of one
        field**, not the fields of one cell. Doing that here produced a step 600x the state norm and an
        apparent refutation of the whole idea. Index the degrees of freedom explicitly instead.
        """
        step, singular = np.zeros(n * fields), 0
        for cells in groups:
            dof = np.concatenate([np.asarray(cells) + f * n for f in range(fields)])
            sub = csr[dof][:, dof].toarray()
            try:
                step[dof] = -np.linalg.solve(sub, rhs[dof])
            except np.linalg.LinAlgError:
                step[dof], singular = -rhs[dof], singular + 1
        return jnp.asarray(step), singular

    cells_each = [[c] for c in range(n)]
    point_implicit, singular = block_solve_at(cells_each, r)

    # The line-like arm: amalgamate the Jacobian to one scalar per cell pair (the block's Frobenius
    # norm), keep only strong connections, and aggregate along them -- then invert the Jacobian
    # restricted to each aggregate. Both the strength filter and the aggregation are the library's own,
    # so this is the coarsening the preconditioner already uses, repurposed as a smoother's blocks.
    from aquaflux.solve.multigrid import _aggregate, _strength_classical

    coo = jacobian.tocoo()
    cell_row, cell_col = coo.row // fields, coo.col // fields
    amalgam = sp.coo_matrix((coo.data**2, (cell_row, cell_col)), shape=(n, n)).tocsr()
    amalgam.data = np.sqrt(amalgam.data)
    strong = _strength_classical(amalgam, STRENGTH).tocoo()
    keep = strong.row != strong.col
    labels, n_aggregates = _aggregate(strong.row[keep], strong.col[keep], n)
    members: dict[int, list[int]] = {}
    for cell, label in enumerate(labels.tolist()):
        members.setdefault(int(label), []).append(cell)
    sizes = np.array([len(v) for v in members.values()])
    groups = list(members.values())
    strong_block, singular_groups = block_solve_at(groups, r)

    volume = np.asarray(coupled.momentum.geometry.cell.volume)
    explicit = -residual / jnp.asarray(np.tile(volume, fields))
    shift = jnp.asarray(policy.shift_term(state).diagonal)
    row_scaled = -residual / jnp.where(shift > 0, shift, 1.0)

    print(
        f"  strength-of-connection aggregates: {n_aggregates} over {n} cells, "
        f"size min/median/max {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}"
        f" ({singular_groups} singular)\n"
    )
    if singular:
        print(
            f"  ⚠️ {singular} of {n} cell blocks were singular and fell back to the explicit step\n"
        )

    scale_of = float(jnp.linalg.norm(state))

    # A single application is not the method: Mavriplis's smoothing term is FIVE cycles of a nonlinear
    # smoother, so the local operator is iterated with the residual re-evaluated each sweep. `D` is held
    # frozen at the initial state, as his RK cycle holds its own local operator frozen.
    def sweep(groups, omega, cycles=5):
        """Residual after ``cycles`` damped nonlinear sweeps of the frozen local solve."""
        u = state
        for _ in range(cycles):
            direction, _ = block_solve_at(groups, np.asarray(coupled.residual(u)))
            u = u + omega * direction
            if not np.isfinite(float(measure(coupled.residual(u)))):
                return np.inf
        return float(measure(coupled.residual(u))) / base

    print(f"{'direction':>16} {'|step|/|u| at a=1':>18} {'best |R|/|R0|':>15} {'at a':>9}")
    print("-" * 62)
    for label, step in (
        ("explicit", explicit),
        ("row-scaled", row_scaled),
        ("point-implicit", point_implicit),
        ("strong-block", strong_block),
    ):
        ratios = []
        for alpha in ALPHAS:
            moved = float(measure(coupled.residual(state + alpha * step)))
            ratios.append(moved / base if np.isfinite(moved) else np.inf)
        best = int(np.argmin(ratios))
        mark = "  <- DESCENDS" if ratios[best] < 1.0 else "  (no descent)"
        print(
            f"{label:>16} {float(jnp.linalg.norm(step)) / scale_of:>18.3e} "
            f"{ratios[best]:>15.4f} {ALPHAS[best]:>9.0e}{mark}"
        )
    print("\nratios are |R(u + a*step)| / |R(u)| in the march's own measure; below 1.0 descends.\n")
    print(f"{'iterated 5 sweeps':>20} " + " ".join(f"w={w:<7g}" for w in (1.0, 0.5, 0.2, 0.05)))
    print("-" * 60)
    for label, groups_ in (("point-implicit", cells_each), ("strong-block", groups)):
        row = [sweep(groups_, w) for w in (1.0, 0.5, 0.2, 0.05)]
        print(f"{label:>20} " + " ".join(f"{v:<9.4f}" for v in row))
    print("\nfive damped nonlinear sweeps of the frozen local solve -- Mavriplis's smoothing term.")


if __name__ == "__main__":
    main()
