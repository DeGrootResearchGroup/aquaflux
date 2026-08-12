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
near-null direction is **pure omega** to three decimals. So it is the omega column that is nearly empty:
changing omega in such a cell barely moves any of that cell's own equations, since omega's influence
there travels through neighbour transport rather than locally. Those cells sit in low-``k`` regions
(median ``k`` 0.122 against 0.655 over the mesh) and come in spanwise-symmetric pairs.

**Sweeping the shift is the interesting use, and it is cheap.** The Jacobian does not depend on beta --
only the diagonal added to it does -- so one materialization serves every shift, and each extra value
costs an assembly and a batched decomposition rather than another coloured probe. That matters because
of what it can settle: the pseudo-transient shift is what keeps a degenerate row solvable, so if
``sigma_min`` in those blocks tracks beta downwards, the march's low-shift conditioning wall **is** this
local degeneracy, and the lever is a field-specific shift rather than anything global.

**Usage** -- the checkpoint index, then one or more shifts to report at::

    python3 validation/bfs3d_openfoam/cell_block_conditioning.py 49 0.0293
    python3 validation/bfs3d_openfoam/cell_block_conditioning.py 49 0.5 0.1 0.05 0.02 0.005 0.0

The first shift is the one whose worst blocks get the per-field breakdown; the rest are reported as
summary rows. Pass ``0`` for the unshifted Jacobian. Materializes the coupled Jacobian, so run it on its
own: it is a couple of gigabytes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    MonolithicAmgPreconditioner,
    block_stencil_gather_map,
)
from aquaflux.solve.amg_preconditioner import ShiftedCellMajorOperator  # noqa: E402
from aquaflux.turbulence.coupled import (  # noqa: E402
    _PROBE_BATCH_SIZE,
    _batched_jacobian_matvec,
    _coupled_jacobian_plan,
    _coupled_shift_policy,
    _frozen_shift_diagonal,
    _jacobian_matvec,
)

#: Field order of the coupled state, for labelling the null direction.
FIELDS = ("u", "v", "w", "p", "k", "omega")

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


def main():
    index = int(sys.argv[1])
    shifts = [float(argument) for argument in sys.argv[2:]] or [0.0]
    data = np.load(CASE / f"checkpoints/state-{index:05d}.npz")
    state = jnp.asarray(data["state"])
    coupled = compare.build_case()["coupled"]
    n_fields = coupled.layout.dim + 3
    n_cells = coupled.layout.n_cells
    plan = _coupled_jacobian_plan(coupled, 3)
    structure = block_stencil_gather_map(plan)
    policy = _coupled_shift_policy(coupled, state, "twolevel")
    jacobian = MonolithicAmgPreconditioner._materialize_jacobian(
        lambda v: _jacobian_matvec(coupled, state, v),
        plan,
        lambda seeds: _batched_jacobian_matvec(coupled, state, seeds),
        _PROBE_BATCH_SIZE,
        structure,
    )
    indptr, indices, _ = structure
    assembler = ShiftedCellMajorOperator(indptr, indices, n_fields)
    k = np.asarray(coupled.layout.unpack(state)[1])  # [flow, k, omega]
    print(
        f"state-{index:05d}: {n_cells} cells, block size {n_fields}, equilibrated to unit diagonals\n"
        f"{'shift':>10} {'median':>11} {'p1':>11} {'min':>11} {'below ' + format(SINGULAR_BELOW, 'g'):>11}",
        flush=True,
    )
    first = None
    for beta in shifts:
        diagonal = (
            _frozen_shift_diagonal(policy, beta, state) if beta else np.zeros(n_cells * n_fields)
        )
        cell_major, _, _ = assembler.assemble(jacobian.data, diagonal)
        blocks = diagonal_blocks(cell_major, n_fields, n_cells)
        smallest, condition = conditioning(blocks)
        print(
            f"{beta:>10.4g} {np.median(smallest):>11.3e} {np.quantile(smallest, 0.01):>11.3e} "
            f"{smallest.min():>11.3e} {int((smallest < SINGULAR_BELOW).sum()):>11}",
            flush=True,
        )
        if first is None:
            first = (beta, blocks, smallest, condition)

    beta, blocks, smallest, condition = first
    order = np.argsort(smallest)
    near_singular = int((smallest < SINGULAR_BELOW).sum())
    _, _, right = np.linalg.svd(blocks[order[:20]])
    print(f"\nat shift {beta:g}, the 20 most singular blocks -- near-null direction by field:")
    print(
        f"  {'cell':>7} {'sigma_min':>10} {'cond':>9} {'k':>10}  "
        + "".join(f"{f:>8}" for f in FIELDS)
    )
    for row, cell in enumerate(order[:20]):
        weights = np.abs(right[row, -1])
        print(
            f"  {cell:>7} {smallest[cell]:>10.2e} {condition[cell]:>9.1e} "
            f"{k[cell]:>10.2e}  " + "".join(f"{w:>8.3f}" for w in weights)
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
