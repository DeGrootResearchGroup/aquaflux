"""Which level of the native hierarchy defeats an incomplete factorization, and why?

A host V-cycle over the native hierarchy raises ``Factor is exactly singular`` on the ``bfs3d`` flow
block, and it does so with the equilibration and cell-major reorder the host AMG path applies -- so the
preprocessing is not the whole story. Two candidates, which want different fixes:

1. **The dropping.** ``scipy``'s ``spilu`` is a drop-tolerance factorization, not a level-based one:
   there is no ILU(0) in ``scipy`` at all. Dropping small entries can remove precisely what a pivot
   needed, where a level-0 factorization keeps the operator's own pattern and cannot.
2. **The level.** The coarse operators come from Galerkin products, and this module already carries a
   pseudo-inverse fallback for a singular *coarsest* operator -- so a near-singular intermediate level
   is not hypothetical, and a factorization of one fails whatever the dropping.

This reports, per level, whether the factorization survives across a grid of drop tolerances and fill
allowances, beside the level's own diagonal statistics. A failure that clears as the dropping relaxes is
(1); one that persists to a near-complete factorization is (2).

Usage::

    BFS3D_PROBE_STATE=state-00067 validation/run_case.sh validation/bfs3d_openfoam/host_ilu_levels.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

CASE = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE))
sys.path.insert(0, str(CASE.parents[1]))

import compare  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    build_convection_hierarchy,
    equilibrate_cell_major,
)
from field_split_probe import STATES, load_state, materialize  # noqa: E402

#: The flow block's own coarsening, matching what the host V-cycle asks for.
COARSENING = dict(
    block_size=4,
    mis_aggregation=True,
    max_coarse=500,
    max_levels=5,
    strength_threshold=0.25,
    avoid_singletons=True,
    aggressive_levels=0,
    prolongation_smoothing="none",
)


def _to_scipy(level) -> sp.csr_matrix:
    return sp.csr_matrix(
        (
            np.asarray(level.operator.data),
            np.asarray(level.operator.indices),
            np.asarray(level.operator.indptr),
        ),
        shape=level.operator.shape,
    )


def main() -> None:
    name = os.environ.get("BFS3D_PROBE_STATE", "state-00067")
    if name not in STATES:
        raise SystemExit(f"BFS3D_PROBE_STATE={name!r} is not one of {list(STATES)}")
    coupled = compare.build_case()["coupled"]
    state = load_state(name)
    print(f"\n{'=' * 78}\nhost-ILU level survey on {name}\n{'=' * 78}", flush=True)

    from aquaflux.solve import FieldGroups, block_stencil_gather_map
    from aquaflux.turbulence.coupled import _coupled_jacobian_plan

    n_fields = coupled.layout.dim + 3
    plan = _coupled_jacobian_plan(coupled, 3)
    jacobian = materialize(coupled, state, plan, block_stencil_gather_map(plan), n_fields)
    groups = FieldGroups(
        n_cells=coupled.layout.n_cells,
        n_leading_fields=coupled.layout.dim + 1,
        n_trailing_fields=2,
    )
    flow = sp.csr_matrix(jacobian[groups.leading, :][:, groups.leading])
    print(f"  flow block: {flow.shape[0]} dofs, {flow.nnz} nnz", flush=True)

    hierarchy = build_convection_hierarchy(flow, **COARSENING)
    print(f"  hierarchy: {len(hierarchy.levels)} levels\n", flush=True)

    # ⚠️ THE ORDERING, not just the dropping. `spilu` applies a COLAMD **column permutation** and
    # partial pivoting by default, which discards the cell-major interleave the equilibration step just
    # imposed -- the very ordering that keeps an incomplete factorization's fill local to a cell on this
    # saddle. `permc_spec="NATURAL"` keeps it; `diag_pivot_thresh=0` forces diagonal pivoting.
    # A near-complete row is deliberately absent: a complete LU of a 21M-nonzero 3D block is the fill
    # wall this whole programme exists to avoid, and running one risks the machine for little.
    # `drop_tol=0` with NATURAL ordering and no pivoting is the CLOSEST scipy gets to ILU(0): no
    # value-based dropping, no reordering, no row exchanges. It is still not ILU(0) -- SuperLU bounds
    # fill by the fill_factor rather than by the operator's own pattern, so it may fill positions ILU(0)
    # would never touch -- but if it produces a usable factor the dependency question is moot, and it
    # costs one run to find out. This is the arm the first survey never reached.
    grid = [
        (1e-4, 1.0, "NATURAL", 0.0),
        (0.0, 1.0, "NATURAL", 0.0),
        (0.0, 2.0, "NATURAL", 0.0),
        (0.0, 4.0, "NATURAL", 0.0),
    ]
    for depth, level in enumerate(hierarchy.levels):
        operator = _to_scipy(level)
        if level.coarse_inv is not None:
            print(
                f"  level {depth}: {operator.shape[0]:>6} dofs -- COARSEST (direct solve, skipped)"
            )
            continue
        scaled, _, _ = equilibrate_cell_major(operator, level.block_size)
        diagonal = np.abs(scaled.diagonal())
        zeros = int((diagonal == 0.0).sum())
        print(
            f"  level {depth}: {operator.shape[0]:>6} dofs, {operator.nnz / 1e6:>5.2f}M nnz  "
            f"|diag| min {diagonal.min():.2e} median {np.median(diagonal):.2e}  "
            f"exact zeros {zeros}",
            flush=True,
        )
        for drop_tol, fill, permc, pivot in grid:
            try:
                factors = spla.spilu(
                    scaled.tocsc(),
                    drop_tol=drop_tol,
                    fill_factor=fill,
                    permc_spec=permc,
                    diag_pivot_thresh=pivot,
                )
                # ⚠️ "it returned" is NOT "it is usable", and reading the first as the second cost a
                # run: forcing diagonal pivoting avoids the exactly-singular error by pivoting on a
                # tiny diagonal instead, which returns a factor whose entries are enormous and whose
                # application is NaN. Judge it the way the solver will -- by what one application does
                # to a real residual.
                rng = np.random.default_rng(0)
                probe = rng.normal(size=scaled.shape[0])
                applied = factors.solve(probe)
                residual = np.linalg.norm(scaled @ applied - probe) / np.linalg.norm(probe)
                verdict = (
                    f"{factors.nnz / max(scaled.nnz, 1):.2f}x nnz  "
                    f"max|U| {np.abs(factors.U.data).max():.2e}  "
                    f"||A M^-1 r - r||/||r|| {residual:.3e}"
                )
            except RuntimeError as error:
                verdict = f"FAILED ({error})"
            print(
                f"      drop {drop_tol:<7g} fill {fill:<4g} perm {permc:<8} pivot {pivot:<4g} "
                f"-> {verdict}",
                flush=True,
            )

    print(
        "\n  READ: if NATURAL ordering clears what COLAMD failed, the cell-major interleave is what makes\n"
        "  an incomplete factorization viable here and `spilu`'s default reordering was discarding it.\n"
        "  If nothing clears, scipy cannot express this smoother (it has no level-based ILU(0)) and a\n"
        "  hand-written or third-party ILU(0) is the route.",
        flush=True,
    )


if __name__ == "__main__":
    main()
