"""Does equilibration change the graph the aggregation coarsens, and by how much?

Symmetric equilibration is a similarity transform, so every *spectral* quantity in a multigrid setup
is invariant under it. That argument says nothing about the **graph**, and at the default strength
threshold of zero the aggregation reads the graph and nothing else.

The connection is that ``symmetrically_equilibrate`` is a sparse triple product ``D A D``, and a
sparse product stores only entries whose result is nonzero -- so it **drops every explicit zero**,
while the rest of the pipeline (``abs``, the coordinate regroup in ``_cell_graph``, ``triu``) preserves
them. A stored zero is a weight-zero edge of the cell graph, and at threshold zero a weight-zero edge
is a full-status connection: two cells joined only by zeros can land in one aggregate.

So the two settings of ``equilibrate`` may coarsen genuinely different graphs. This measures whether
they do, on the operator the native trailing hierarchy is actually built from, by running the real
``_cell_graph`` / ``_aggregation_edges`` / ``_mis_aggregate`` rather than a restatement of them.

**A cell edge only dies if EVERY one of its ``block_size**2`` entries was exactly zero**, so the count
of dropped degrees of freedom is an upper bound on lost edges and can overstate it by that factor.
That is the quantity in question, and it is why this is measured rather than argued.

The leading ``[u,v,w,p]`` block is carried as a control: it holds almost no explicit zeros, so it
should show no change, and if it does the effect is not the one described here.

Usage::

    python3 -u validation/bfs3d_openfoam/equilibration_graph_effect.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

CASE = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    ColumnProbePlan,
    FieldGroups,
    block_stencil_colouring,
    block_stencil_gather_map,
)
from aquaflux.solve import MonolithicAmgPreconditioner  # noqa: E402
from aquaflux.solve import symmetrically_equilibrate  # noqa: E402
from aquaflux.solve.multigrid import (  # noqa: E402
    _aggregation_edges,
    _cell_graph,
    _mis_aggregate,
)
from aquaflux.turbulence import hybrid_initialize  # noqa: E402
from aquaflux.turbulence.coupled import (  # noqa: E402
    _PROBE_BATCH_SIZE,
    _batched_jacobian_matvec,
    _jacobian_matvec,
)

REACH = 3
#: The aggregation's default: keep the full graph, so a weight-zero edge counts as a connection.
STRENGTH_THRESHOLD = 0.0


def coarsen(a, block_size, seed=0):
    """Run the real coarsening path and report what it produced.

    Mirrors ``_build_aggregation_hierarchy``'s first level: the cell graph, the edge set at the
    strength threshold, then the maximal-independent-set aggregation.
    """
    graph = _cell_graph(a, block_size) if block_size > 1 else abs(a).tocsr()
    edges = _aggregation_edges(graph, STRENGTH_THRESHOLD)
    connectivity = sp.csr_matrix(abs(edges))
    result = _mis_aggregate(connectivity, seed=seed)
    aggregate, n_coarse = (result[0], result[-1])
    sizes = np.bincount(aggregate, minlength=n_coarse)
    return {
        "graph_edges": int(graph.nnz),
        "upper_edges": int(connectivity.nnz),
        "aggregates": int(n_coarse),
        "ratio": a.shape[0] // block_size / max(n_coarse, 1),
        "max_size": int(sizes.max()),
        "aggregate": aggregate,
    }


def compare_block(label, block, block_size):
    """Coarsen a block raw and equilibrated, and report the difference."""
    raw = sp.csr_matrix(block)
    equilibrated, _ = symmetrically_equilibrate(raw)
    zeros = int((raw.data == 0).sum())
    print(
        f"\n  {label}  ({raw.shape[0]} dofs, block size {block_size}, "
        f"{raw.nnz} stored, {zeros} exact zeros)",
        flush=True,
    )
    print(
        f"    {'arm':>14s} {'cell edges':>12s} {'upper edges':>12s} {'aggregates':>11s}"
        f" {'ratio':>7s} {'max size':>9s}",
        flush=True,
    )
    out = {}
    for arm, matrix in (("raw", raw), ("equilibrated", equilibrated)):
        out[arm] = coarsen(matrix, block_size)
        r = out[arm]
        print(
            f"    {arm:>14s} {r['graph_edges']:12d} {r['upper_edges']:12d} {r['aggregates']:11d}"
            f" {r['ratio']:7.2f} {r['max_size']:9d}",
            flush=True,
        )
    lost = out["raw"]["upper_edges"] - out["equilibrated"]["upper_edges"]
    moved = int((out["raw"]["aggregate"] != out["equilibrated"]["aggregate"]).sum())
    print(
        f"    -> {lost} of {out['raw']['upper_edges']} cell edges lost"
        f" ({lost / max(out['raw']['upper_edges'], 1):.1%});"
        f" coarse size {out['raw']['aggregates']} vs {out['equilibrated']['aggregates']}",
        flush=True,
    )
    # Aggregate LABELS are not comparable between runs, so this counts label disagreement only as a
    # coarse signal that the partitions are not identical -- a zero here is meaningful, a nonzero is not
    # a magnitude.
    print(f"    -> partitions identical: {moved == 0}", flush=True)
    del equilibrated
    gc.collect()


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
    print("[state] target-Re cold initial field; uniform reach 3 (the shipped arm)", flush=True)

    plan = ColumnProbePlan.uniform(block_stencil_colouring(owner, nb, n, REACH), n_fields)
    structure = block_stencil_gather_map(plan)

    def matvec(v):
        return _jacobian_matvec(coupled, state, v)

    def batched(seeds):
        return _batched_jacobian_matvec(coupled, state, seeds)

    jacobian = MonolithicAmgPreconditioner._materialize_jacobian(
        matvec, plan, batched, _PROBE_BATCH_SIZE, structure
    )
    groups = FieldGroups(n_cells=n, n_leading_fields=n_fields - 2, n_trailing_fields=2)
    a_ll, _, _, a_tt = groups.blocks(jacobian)
    del jacobian
    gc.collect()

    compare_block("trailing [k, omega]  (the block the traced hierarchy coarsens)", a_tt, 2)
    compare_block("leading  [u,v,w,p]   (control: almost no explicit zeros)", a_ll, n_fields - 2)

    print(
        "\n  A large edge loss with a changed coarse size supports the graph explanation for the\n"
        "  equilibrate-on/off cycle-count gap. A negligible loss refutes it, and the gap is then\n"
        "  something the similarity-transform argument does cover after all.",
        flush=True,
    )


if __name__ == "__main__":
    main()
