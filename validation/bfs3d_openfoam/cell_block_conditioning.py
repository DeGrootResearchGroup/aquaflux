"""How near-singular are the coupled operator's per-cell blocks, and in which field?

Any preconditioner that relaxes a **cell at a time** -- point-block Jacobi, a Vanka patch, a block
incomplete-LU pivoting on the cell block -- has to invert the diagonal ``(n_fields, n_fields)`` block of
the assembled operator. If that block is near-singular the local solve produces an enormous correction
in the offending direction, and the method fails for a reason that has nothing to do with the
multigrid's coarse space or its sweep count. A *global* smoother such as a cell-major incomplete-LU
never inverts a cell block in isolation and is untroubled by the same operator, which is why the two
classes of smoother can behave completely differently on one matrix.

This reports the distribution of the smallest singular value over every cell block, and for the worst
blocks the **field composition of the near-null direction** -- which turns "some patches are badly
conditioned" into "this field is the problem". It also prints ``k`` alongside, because a degenerate
turbulence row is usually a story about the state, not the discretization.

Measured on the three-dimensional backward-facing step at a mid-march state (2026-08-08, entering the
step at beta 0.0293, V-cycle shift at the 0.05 floor): median ``sigma_min`` 2.9e-2, **353 of 23040 cells
below 1e-3** with condition numbers of 5e6-9e6, and in every one of the twenty worst blocks the
near-null direction is **pure omega** to three decimals. Those cells sit in low-``k`` regions (median
``k`` 0.122 against 0.655 over the mesh) and come in spanwise-symmetric pairs.

**A pure-omega near-null direction does NOT mean the omega column is empty**, which is how that result was
first read. A block that is the identity plus one large entry in the omega ROW -- the omega residual
depending strongly on the cell's own pressure or velocity -- has a ``sigma_min`` of about one over that
entry and a near-null right singular vector along omega, while its omega diagonal is ordinary and an exact
local solve is perfectly stable (the inverse grows by the size of the entry, not by a small pivot). The two
readings are told apart by removing couplings rather than by the singular vector: this script also reports
the near-singular count with the omega row's couplings to the other fields set to zero, and, as the
control, with the omega column's set to zero instead, plus the largest omega-row coupling in each worst
block. On pitzDaily (``state-00017``, shift 1e-4, 2026-10-14) removing the omega-row couplings took the
count below 1e-3 from 6 to **0** and below 1e-2 from 273 to **0**, removing the omega-column couplings
changed neither, and over the 200 smallest blocks ``sigma_min`` tracked one over the largest omega-row
coupling with a log-log correlation of 0.99 -- the large-coupling reading, not the empty-column one.

**Sweeping the shift is cheap.** The Jacobian does not depend on beta -- only the diagonal added to it
does -- so one materialization serves every shift, and each extra value costs an assembly and a batched
decomposition rather than another coloured probe. On the step the count barely moved between shift 0 and
0.05, which the large-coupling reading predicts: the shift adds to a diagonal that was never the small part.

**Usage** -- the checkpoint index, then one or more shifts to report at::

    python3 validation/bfs3d_openfoam/cell_block_conditioning.py 49 0.0293
    python3 validation/bfs3d_openfoam/cell_block_conditioning.py 49 0.5 0.1 0.05 0.02 0.005 0.0
    python3 validation/bfs3d_openfoam/cell_block_conditioning.py --case pitzdaily_openfoam 17 0.0001

The first shift is the one whose worst blocks get the per-field breakdown; the rest are reported as
summary rows. ``--case`` names another case directory under ``validation/`` with a ``compare.build_case``
and a ``checkpoints/`` directory (the step's own by default). Pass ``0`` for the unshifted Jacobian. Materializes the coupled Jacobian, so run it on its
own: it is a couple of gigabytes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _case_directory(argv: list[str]) -> tuple[Path, list[str]]:
    """The case directory ``--case`` names (this script's own by default), and the remaining arguments."""
    if argv[:1] == ["--case"]:
        return ROOT / "validation" / argv[1], argv[2:]
    return Path(__file__).resolve().parent, argv


CASE, ARGUMENTS = _case_directory(sys.argv[1:])
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from aquaflux.precondition import MaterializedJacobianPreconditioner
from aquaflux.precondition.frozen_operator import (equilibrate_cell_major,)
from aquaflux.precondition.sparse_jacobian import (block_stencil_gather_map,)
from aquaflux.turbulence import ScalarTwoLevel  # noqa: E402
from aquaflux.turbulence.coupled import _coupled_jacobian_plan, _coupled_shift_policy
from aquaflux.precondition.materialized_session import (PROBE_BATCH_SIZE, batched_jacobian_matvec, frozen_shift_diagonal, jacobian_matvec,)

#: Field order of the coupled state, for labelling the null direction, by field count.
FIELDS = {5: ("u", "v", "p", "k", "omega"), 6: ("u", "v", "w", "p", "k", "omega")}

#: Below this smallest singular value a cell block counts as near-singular. The operator is
#: symmetrically equilibrated to unit diagonals, so this is three orders down from the block's own scale
#: -- comfortably past anything a well-posed local solve produces.
SINGULAR_BELOW = 1e-3


def diagonal_blocks(matrix, n_fields: int, n_cells: int) -> np.ndarray:
    """The per-cell diagonal blocks of a cell-major operator, shape ``(n_cells, n_fields, n_fields)``."""
    block = matrix.tobsr(blocksize=(n_fields, n_fields))
    block.sort_indices()
    out = np.empty((n_cells, n_fields, n_fields))
    for cell in range(n_cells):
        lo, hi = block.indptr[cell], block.indptr[cell + 1]
        where = np.flatnonzero(block.indices[lo:hi] == cell)
        out[cell] = block.data[lo + where[0]] if where.size else 0.0
    return out


def conditioning(blocks):
    """``(sigma_min, condition number)`` per block, and the near-null right singular vectors."""
    singular = np.linalg.svd(blocks, compute_uv=False)
    return singular[:, -1], singular[:, 0] / np.maximum(singular[:, -1], 1e-300)


def without_couplings(blocks: np.ndarray, field: int, *, row: bool) -> np.ndarray:
    """The blocks with one field's row (or column) couplings to the other fields set to zero.

    Its own diagonal entry is kept. Comparing ``sigma_min`` before and after tells a field whose own
    column is nearly empty (removing its ROW couplings leaves the block near-singular) from one whose row
    carries a large coupling to the other fields (removing them makes the block well conditioned).
    """
    cut = blocks.copy()
    others = [f for f in range(blocks.shape[1]) if f != field]
    if row:
        cut[:, field, others] = 0.0
    else:
        cut[:, others, field] = 0.0
    return cut


def main():
    index = int(ARGUMENTS[0])
    shifts = [float(argument) for argument in ARGUMENTS[1:]] or [0.0]
    data = np.load(CASE / f"checkpoints/state-{index:05d}.npz")
    state = jnp.asarray(data["state"])
    coupled = compare.build_case()["coupled"]
    n_fields = coupled.layout.n_fields
    n_cells = coupled.layout.n_cells
    fields = FIELDS[n_fields]
    omega = fields.index("omega")
    plan = _coupled_jacobian_plan(coupled, 3)
    structure = block_stencil_gather_map(plan)
    policy = _coupled_shift_policy(coupled, state, ScalarTwoLevel())
    jacobian = MaterializedJacobianPreconditioner._materialize_jacobian(
        lambda v: jacobian_matvec(coupled, state, v),
        plan,
        lambda seeds: batched_jacobian_matvec(coupled, state, seeds),
        PROBE_BATCH_SIZE,
        structure,
    )
    k = np.asarray(coupled.layout.unpack(state)[1])  # [flow, k, omega]
    print(
        f"state-{index:05d}: {n_cells} cells, block size {n_fields}, equilibrated to unit diagonals\n"
        f"{'shift':>10} {'median':>11} {'p1':>11} {'min':>11} {'below ' + format(SINGULAR_BELOW, 'g'):>11}"
        f" {'no ω-row':>9} {'no ω-col':>9}   (counts below with omega's row / column couplings removed)",
        flush=True,
    )
    first = None
    for beta in shifts:
        diagonal = (
            frozen_shift_diagonal(policy, beta, state) if beta else np.zeros(n_cells * n_fields)
        )
        cell_major, _, _ = equilibrate_cell_major(
            MaterializedJacobianPreconditioner._shifted(jacobian, diagonal), n_fields
        )
        blocks = diagonal_blocks(cell_major, n_fields, n_cells)
        smallest, condition = conditioning(blocks)
        no_row, _ = conditioning(without_couplings(blocks, omega, row=True))
        no_column, _ = conditioning(without_couplings(blocks, omega, row=False))
        print(
            f"{beta:>10.4g} {np.median(smallest):>11.3e} {np.quantile(smallest, 0.01):>11.3e} "
            f"{smallest.min():>11.3e} {int((smallest < SINGULAR_BELOW).sum()):>11}"
            f" {int((no_row < SINGULAR_BELOW).sum()):>9} {int((no_column < SINGULAR_BELOW).sum()):>9}",
            flush=True,
        )
        if first is None:
            first = (beta, blocks, smallest, condition)

    beta, blocks, smallest, condition = first
    order = np.argsort(smallest)
    near_singular = int((smallest < SINGULAR_BELOW).sum())
    worst20 = order[:20]
    _, _, right = np.linalg.svd(blocks[worst20])
    no_row, _ = conditioning(without_couplings(blocks[worst20], omega, row=True))
    coupling = np.abs(np.delete(blocks[worst20, omega, :], omega, axis=1)).max(axis=1)
    print(f"\nat shift {beta:g}, the 20 most singular blocks -- near-null direction by field:")
    print(
        f"  {'cell':>7} {'sigma_min':>10} {'cond':>9} {'k':>10}  "
        + "".join(f"{f:>8}" for f in fields)
        + f"  {'max|ω row|':>11} {'1/that':>9} {'no ω-row':>9}"
    )
    for row, cell in enumerate(worst20):
        weights = np.abs(right[row, -1])
        print(
            f"  {cell:>7} {smallest[cell]:>10.2e} {condition[cell]:>9.1e} "
            f"{k[cell]:>10.2e}  " + "".join(f"{w:>8.3f}" for w in weights)
            + f"  {coupling[row]:>11.3e} {1 / coupling[row]:>9.2e} {no_row[row]:>9.2e}"
        )
    if near_singular:
        worst = order[:near_singular]
        print(
            f"\nk over the {near_singular} near-singular cells: median {np.median(k[worst]):.3e}; "
            f"over all cells: median {np.median(k):.3e}; minimum anywhere: {k.min():.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
