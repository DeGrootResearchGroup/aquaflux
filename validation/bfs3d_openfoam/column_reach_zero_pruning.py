"""Do a shortened column's explicit zeros survive into the operator the smoother factorizes?

A per-column probing reach holds the assembly pattern at the full reach and writes the entries outside
a column's own reach as **exact zeros**, on the reasoning that a consumer then sees the same sparsity as
a uniform-reach build. A uniform build puts the *true* values there instead, which on this case are tiny
but not zero.

That difference is invisible in any comparison of values, and it is not invisible to the sparsity:
symmetric equilibration is a matrix product ``D A D``, and a sparse product stores only the entries
whose result is nonzero. An exact zero is dropped; a value of 1e-26 is kept. So the two arms can reach
the multigrid with **different patterns** despite carrying the same numbers -- and an incomplete
factorization with zero fill takes its pattern from exactly that, which is why a shortened column can
break a smoother while being numerically exact.

This measures it end to end on the real operator: how many entries each arm stores, how many of those
are exact zeros, and how many positions survive equilibration.

Read the ``lost`` column. If the per-column arm loses positions the uniform arm keeps, the pattern
handed to the smoother genuinely differs between them and the shortened reach is not the no-op it is
documented to be.

One arm at a time, with the Jacobian released between them -- each is some gigabytes.

Usage::

    python3 -u validation/bfs3d_openfoam/column_reach_zero_pruning.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    ColumnProbePlan,
    FieldGroups,
    block_stencil_colouring,
    block_stencil_gather_map,
    column_probe_plan,
)
from aquaflux.solve.amg_preconditioner import MonolithicAmgPreconditioner  # noqa: E402
from aquaflux.solve import equilibrate_cell_major  # noqa: E402
from aquaflux.turbulence import hybrid_initialize  # noqa: E402
from aquaflux.turbulence.coupled import (  # noqa: E402
    _PROBE_BATCH_SIZE,
    _batched_jacobian_matvec,
    _jacobian_matvec,
)

REACH = 3
#: The arm under test against the uniform control.
COLUMN_REACH = (3, 3, 3, 2, 2, 2)


def assembled(coupled, state, plan, n_fields):
    """Materialize on the fixed-pattern path the multigrid preconditioner uses."""
    structure = block_stencil_gather_map(plan)

    def matvec(v):
        return _jacobian_matvec(coupled, state, v)

    def batched(seeds):
        return _batched_jacobian_matvec(coupled, state, seeds)

    return MonolithicAmgPreconditioner._materialize_jacobian(
        matvec, plan, batched, _PROBE_BATCH_SIZE, structure
    )


def census(label, jacobian, n_fields):
    """Stored entries and exact zeros, before and after the equilibration the V-cycle applies."""
    before, zeros = jacobian.nnz, int((jacobian.data == 0).sum())
    equilibrated, _, _ = equilibrate_cell_major(jacobian, n_fields)
    after = equilibrated.nnz
    print(
        f"  {label:>18s} {before:12d} {zeros:14d} {after:12d} {before - after:10d}"
        f" {(before - after) / before:8.1%}",
        flush=True,
    )
    del equilibrated
    gc.collect()
    return before - after


def main():
    print("[case] building", flush=True)
    case = compare.build_case()
    coupled = case["coupled"]
    mesh = coupled.momentum.mesh
    n = mesh.n_cells
    n_fields = coupled.layout.dim + 3
    owner, nb, _ = mesh.face_cells.interior_edges()
    owner, nb = np.asarray(owner), np.asarray(nb)

    flow, k, omega = hybrid_initialize(coupled.momentum, coupled.turbulence)
    state = np.asarray(coupled.state_from_physical(flow, k, omega))
    print("[state] the target-Re cold initial field -- where the divergence occurs", flush=True)

    arms = (
        (
            "uniform reach 3",
            ColumnProbePlan.uniform(block_stencil_colouring(owner, nb, n, REACH), n_fields),
        ),
        (
            f"column {COLUMN_REACH}",
            column_probe_plan(owner, nb, n, COLUMN_REACH, pattern_reach=REACH),
        ),
    )

    print(
        f"\n  {'arm':>18s} {'stored':>12s} {'exact zeros':>14s} {'after equil':>12s}"
        f" {'lost':>10s} {'':>8s}",
        flush=True,
    )
    lost = {}
    blocks = None
    for label, plan in arms:
        jacobian = assembled(coupled, state, plan, n_fields)
        lost[label] = census(label, jacobian, n_fields)
        if blocks is None:
            # The field split's own blocks, from the UNIFORM arm -- the shipped configuration. The
            # question here is separate from the column reach: an aggregation at zero strength
            # threshold reads only the graph, so if equilibration prunes a block's explicit zeros it
            # coarsens a DIFFERENT graph, which a similarity-transform argument does not cover.
            groups = FieldGroups(n_cells=n, n_leading_fields=n_fields - 2, n_trailing_fields=2)
            a_ll, _, _, a_tt = groups.blocks(jacobian)
            blocks = {"leading [u,v,w,p]": a_ll, "trailing [k,omega]": a_tt}
        del jacobian
        gc.collect()

    print(
        "\n  the field split's own blocks (uniform arm), through the same equilibration:",
        flush=True,
    )
    print(
        f"  {'block':>18s} {'stored':>12s} {'exact zeros':>14s} {'after equil':>12s}"
        f" {'lost':>10s} {'':>8s}",
        flush=True,
    )
    for label, block in blocks.items():
        fields = n_fields - 2 if label.startswith("leading") else 2
        census(label, block.tocsr(), fields)

    print(
        "\n  If the per-column arm loses positions the uniform arm keeps, the two reach the smoother\n"
        "  with different patterns although their values agree -- and a zero-fill incomplete\n"
        "  factorization takes its pattern from precisely that.",
        flush=True,
    )


if __name__ == "__main__":
    main()
