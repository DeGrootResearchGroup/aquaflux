"""What exact coarse operators for one column would cost in probe colours, on bfs3d or pitzDaily.

A short probe of a column drops its far couplings, and on bfs3d the reach-2 pressure fit fails through
its COARSE levels (``option4_diagnostic.md``): the leading hierarchy's Galerkin operators ``Pᵀ A P``
sum each aggregate's couplings, and the dropped ones move those sums. The leading block's prolongation
is a frozen, unsmoothed nodal aggregation, so the first coarse operator of the column is exactly
``Pᵀ (J P)`` restricted to that column, and every deeper level follows from it. Two ways to read it from
the Jacobian, each costed here by its colour count (one batched directional derivative per colour):

  aggregate  probe ``J P`` directly: a seed is the sum of the column's aggregate indicators of one
             colour. Two aggregates may share a colour only if no fine row couples (within the stored
             reach) to both. Costs these colours ON TOP of the column's fine reach-2 probe.
  aware      one fine colouring of the column that serves both levels: two cells may share a colour
             only if they are not both within the short reach of a common row (the short probe's own
             rule), and, when both are within the stored reach of a common row, they lie in the same
             aggregate -- so every far coupling a row receives lands on the aggregate it came from and
             the coarse sums are exact, while the fine level keeps the short probe's drop and fold.

The session is built as the viscosity-ramp march builds it (anchor companion, anchor seed) under the probe
``COARSE_COLOURS_BUILD`` (a ``reach_folding`` probe token, default ``2``: every column at reach 2), which
freezes the aggregates. Nothing is solved.

Usage::

    REACH_FOLDING_STEP=5 COARSE_COLOURS_COLUMN=p \\
        validation/run_case.sh validation/reach_research/coarse_probe_colours.py --wait
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import sys
from pathlib import Path

VALIDATION = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VALIDATION.parent))
sys.path.insert(0, str(VALIDATION))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import reach_folding  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from aquaflux.precondition.sparse_jacobian import (  # noqa: E402
    _pattern_colouring,
    _saturation_colouring,
    _stencil_pattern,
)
from aquaflux.turbulence import coupled_step, open_session  # noqa: E402

#: The reach of the stored pattern the column's couplings lie in.
STORED_REACH = 3


def aggregates_of(level, n_cells, field):
    """Each cell's aggregate on the next level, for one field of a nodal prolongation, ``(n_cells,)``."""
    rows, cols = np.asarray(level.p_frow), np.asarray(level.p_ccol)
    mine = (rows >= field * n_cells) & (rows < (field + 1) * n_cells)
    label = np.full(n_cells, -1)
    label[rows[mine] - field * n_cells] = cols[mine]
    if (label < 0).any():
        raise SystemExit(f"{int((label < 0).sum())} cells of field {field} have no aggregate")
    _, label = np.unique(label, return_inverse=True)
    return label


def hop_diameters(label, adjacency, n_aggregates):
    """Largest graph distance inside each aggregate, over the aggregate's own cells (BFS within it)."""
    diameters = np.zeros(n_aggregates, dtype=int)
    members = sp.csr_matrix(
        (np.ones(label.size), (label, np.arange(label.size))), shape=(n_aggregates, label.size)
    )
    for a in range(n_aggregates):
        cells = members.indices[members.indptr[a] : members.indptr[a + 1]]
        sub = adjacency[cells][:, cells]
        dist = sp.csgraph.shortest_path(sub, unweighted=True, directed=False)
        finite = dist[np.isfinite(dist)]
        diameters[a] = int(finite.max()) if finite.size else 0
    return diameters


def main():
    step_number = int(os.environ.get("REACH_FOLDING_STEP", "5"))
    checkpoints = Path(
        os.environ.get("REACH_FOLDING_CHECKPOINTS", reach_folding.CASE / "checkpoints")
    )
    with np.load(checkpoints / f"state-{step_number - 1:05d}.npz") as data:
        phi = jnp.asarray(data["state"])
    with np.load(checkpoints / f"state-{step_number:05d}.npz") as data:
        beta = float(data["shift"])
    coupled = reach_folding.compare.build_case()["coupled"]
    n, n_fields = coupled.layout.n_cells, coupled.layout.n_fields
    names = reach_folding.field_names(n_fields)
    column_name = os.environ.get("COARSE_COLOURS_COLUMN", "p")
    column = names.index(column_name)
    build_label, build_reach = reach_folding.probe_reach(
        os.environ.get("COARSE_COLOURS_BUILD", "2"), n_fields, names
    )
    rebuild = reach_folding.StepRebuild(coupled, phi, beta, step_number)
    settings = rebuild.spec.settings()
    file_pc = settings["preconditioner"]
    spec = dataclasses.replace(
        file_pc, probe=dataclasses.replace(file_pc.probe, column_reach=build_reach)
    )
    allowed = inspect.signature(coupled_step).parameters
    march = {k: v for k, v in settings.items() if k in allowed and k != "preconditioner"}
    print(
        f"[configuration] case {reach_folding.CASE_NAME}, {n} cells, built at step {step_number}'s "
        f"anchor seed under {build_label}, column {column_name}, stored reach {STORED_REACH}",
        flush=True,
    )
    session = open_session(spec, coupled)
    session.rebind(rebuild.anchor)
    session.build(rebuild.seed, **march)
    levels = session.preconditioner.inverse.leading._hierarchy.levels
    print(
        "[hierarchy] leading levels (degrees of freedom): " + ", ".join(str(lv.n) for lv in levels),
        flush=True,
    )
    owner, nb, _ = coupled.momentum.mesh.face_cells.interior_edges()
    label = aggregates_of(levels[0], n, column)
    n_agg = int(label.max()) + 1
    sizes = np.bincount(label)
    adjacency = _stencil_pattern(owner, nb, n, 1)
    diameters = hop_diameters(label, adjacency, n_agg)
    print(
        f"[aggregates] {n_agg} for {n} cells; size min/median/mean/max {sizes.min()} / "
        f"{int(np.median(sizes))} / {sizes.mean():.2f} / {sizes.max()}; hop diameter median / max "
        f"{int(np.median(diameters))} / {diameters.max()}",
        flush=True,
    )

    short = _stencil_pattern(owner, nb, n, 2)
    stored = _stencil_pattern(owner, nb, n, STORED_REACH)
    _, short_colours = _pattern_colouring(short, n)
    _, stored_colours = _pattern_colouring(stored, n)

    indicator = sp.csr_matrix((np.ones(n), (np.arange(n), label)), shape=(n, n_agg))
    _, aggregate_colours = _pattern_colouring((stored @ indicator).tocsr(), n_agg)

    same = (indicator @ indicator.T).tocsr()
    same.data[:] = 1.0
    near_conflict = (short.T @ short).tocsr()
    far_conflict = (stored.T @ stored).tocsr()
    far_conflict.data[:] = 1.0
    cross = far_conflict - far_conflict.multiply(same)
    cross.eliminate_zeros()
    conflict = (near_conflict + cross).tocsr()
    conflict.data[:] = 1.0
    aware = _saturation_colouring(conflict, n)
    aware_colours = int(aware.max()) + 1

    composite, deeper = label, []
    for depth in range(1, len(levels) - 1):
        n_level = levels[depth].n // levels[depth].block_size
        composite = aggregates_of(levels[depth], n_level, column)[composite]
        n_comp = int(composite.max()) + 1
        lifted = sp.csr_matrix((np.ones(n), (np.arange(n), composite)), shape=(n, n_comp))
        _, colours = _pattern_colouring((stored @ lifted).tocsr(), n_comp)
        deeper.append(
            f"level {depth + 1}: {n_comp} aggregates of {n / n_comp:.1f} cells, {colours}"
        )
    print(
        "[colours] exact coarse from a deeper level down, by composite aggregate seeds: "
        + "; ".join(deeper),
        flush=True,
    )
    print(
        f"[colours] column {column_name}: short reach 2 {short_colours}, stored reach "
        f"{STORED_REACH} {stored_colours}; exact coarse by aggregate seeds {aggregate_colours} on top "
        f"of the short probe ({short_colours + aggregate_colours} in all); aggregate-aware fine "
        f"colouring {aware_colours}",
        flush=True,
    )


if __name__ == "__main__":
    main()
