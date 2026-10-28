"""What a short probe of one column drops, per row field, at one checkpoint of bfs3d or pitzDaily.

Materializes the Jacobian at ``state-<REACH_FOLDING_STEP - 1>`` (a step's start) twice in the reach-3
pattern, once at a uniform reach of 3 and once with column ``REACH_FOLDING_COLUMN`` (default ``p``) at
reach 2, and prints, for each row field of that column: the block's stored entries and norm, how many
entries the short probe leaves exactly zero (the dropped ring, as ``reach_folding.ScaledFold`` reads it)
with their norm against the block's, and the whole difference ``|J_short - J3|`` against the block's.

Usage::

    REACH_FOLDING_STEP=5 REACH_FOLDING_CHECKPOINTS=<dir> \\
        validation/run_case.sh validation/reach_research/dropped_census.py --wait
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

VALIDATION = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VALIDATION.parent))
sys.path.insert(0, str(VALIDATION))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import reach_folding  # noqa: E402


def main():
    step = int(os.environ["REACH_FOLDING_STEP"])
    checkpoints = Path(
        os.environ.get("REACH_FOLDING_CHECKPOINTS", reach_folding.CASE / "checkpoints")
    )
    column = os.environ.get("REACH_FOLDING_COLUMN", "p")
    coupled = reach_folding.compare.build_case()["coupled"]
    n, n_fields = coupled.layout.n_cells, coupled.layout.n_fields
    names = reach_folding.field_names(n_fields)
    with np.load(checkpoints / f"state-{step - 1:05d}.npz") as data:
        state = jnp.asarray(data["state"])
    owner, nb, _ = coupled.momentum.mesh.face_cells.interior_edges()
    owner, nb = np.asarray(owner), np.asarray(nb)
    _label, reach = reach_folding.probe_reach(column, n_fields, names)
    print(
        f"[configuration] case {reach_folding.CASE_NAME}, {n} cells, state-{step - 1:05d}, "
        f"column {column} at reach 2",
        flush=True,
    )
    uniform = reach_folding.materialize(coupled, state, owner, nb, n, n_fields, None)
    short = reach_folding.materialize(coupled, state, owner, nb, n, n_fields, reach)
    dropped = (uniform - uniform.multiply(short != 0)).tocsr()
    dropped.eliminate_zeros()
    difference = (short - uniform).tocsr()
    f = names.index(column)
    columns = slice(f * n, (f + 1) * n)
    for a in range(n_fields):
        rows = slice(a * n, (a + 1) * n)
        block = uniform[rows, columns]
        size = np.linalg.norm(block.data)
        print(
            f"  {names[a]:>5s} rows x {column} column: {block.nnz:9d} entries, |J3| {size:.3e}; "
            f"dropped {dropped[rows, columns].nnz:8d}, |dropped| / |J3| "
            f"{np.linalg.norm(dropped[rows, columns].data) / size:.2e}; |J_short - J3| / |J3| "
            f"{np.linalg.norm(difference[rows, columns].data) / size:.2e}",
            flush=True,
        )
    print(f"  total dropped {dropped.nnz}", flush=True)


if __name__ == "__main__":
    main()
