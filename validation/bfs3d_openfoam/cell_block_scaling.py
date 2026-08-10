"""Does equilibration help or harm the per-cell blocks a block smoother inverts?

A symmetric rescaling ``D A D`` balances the operator's *global* diagonal, which is what an incomplete
factorization and a coarsening want. A block-Jacobi smoother wants something different and possibly
opposed: every cell's own small dense block well conditioned **in isolation**, because it inverts each
one on its own. Those two are not the same requirement, and on the coupled turbulence pair they may
point opposite ways -- rescaling drives each 2x2 toward a unit diagonal with a large subdiagonal, whose
determinant is ``1 - a_kw a_wk``, so it can manufacture near-singularity while improving the global
scaling.

That matters because it decides how to unblock a march. Raw, the framework-native block-Jacobi
hierarchy reproduced the host V-cycle step for step on the middle Reynolds rung and was stopped only by
the build's singular-block guard; equilibrated, it built but its very first differing step dropped the
line-search factor from 1.000 to 0.579. If the numbers below show rescaling worsening the per-cell
conditioning, that is the mechanism, and equilibration is the wrong fix for the guard.

Reports, for both scalings: how many blocks are singular, the distribution of per-cell condition
numbers, and what the inverse's magnitude does -- the last being what a smoother actually applies.

Usage::

    python3 -u validation/bfs3d_openfoam/cell_block_scaling.py state-00057
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    FieldGroups,
    MonolithicAmgPreconditioner,
    block_stencil_gather_map,
    symmetrically_equilibrate,
)
from aquaflux.turbulence.coupled import (  # noqa: E402
    _coupled_jacobian_colouring,
    _coupled_shift_policy,
    _frozen_shift_diagonal,
)
from field_split_probe import FLOOR, STATES, load_state, materialize  # noqa: E402

N_TRAILING = 2


def cell_blocks(a: sp.csr_matrix, block_size: int) -> np.ndarray:
    """Every cell's own dense ``block_size x block_size`` block, shape ``(n_cells, b, b)``.

    Field-major, mirroring what the block smoother extracts: cell ``i``'s degrees of freedom are
    ``{f * n_cells + i}``, so a cell's entries are strided rather than contiguous.
    """
    n_cells = a.shape[0] // block_size
    rows = np.arange(n_cells)
    blocks = np.empty((n_cells, block_size, block_size))
    for f in range(block_size):
        for g in range(block_size):
            blocks[:, f, g] = np.asarray(a[f * n_cells + rows, g * n_cells + rows]).ravel()
    return blocks


def report(label: str, blocks: np.ndarray) -> None:
    """Per-cell conditioning of a set of blocks, and the size of the inverse they produce."""
    singular_values = np.linalg.svd(blocks, compute_uv=False)
    largest, smallest = singular_values[:, 0], singular_values[:, -1]
    # The build's own scale-free singularity test, so the counts here are the ones it acts on.
    scale = np.linalg.norm(blocks, axis=(1, 2)) ** blocks.shape[1]
    singular = np.abs(np.linalg.det(blocks)) < 1e-12 * np.maximum(scale, np.finfo(float).tiny)
    condition = np.where(smallest > 0, largest / np.maximum(smallest, np.finfo(float).tiny), np.inf)
    # What the smoother applies is the INVERSE, so its norm is the quantity that actually blows up.
    healthy = ~singular
    inverse_norm = np.full(blocks.shape[0], np.inf)
    inverse_norm[healthy] = 1.0 / smallest[healthy]

    print(f"  {label}")
    print(f"    singular blocks                {int(singular.sum())} of {blocks.shape[0]}")
    print(
        f"    condition number  p50 {np.median(condition[healthy]):.3e}  "
        f"p99 {np.percentile(condition[healthy], 99):.3e}  max {condition[healthy].max():.3e}"
    )
    print(
        f"    |A_cell^-1|       p50 {np.median(inverse_norm[healthy]):.3e}  "
        f"p99 {np.percentile(inverse_norm[healthy], 99):.3e}  "
        f"max {inverse_norm[healthy].max():.3e}"
    )
    for bar in (1e3, 1e6, 1e9):
        print(
            f"    cells with cond > {bar:.0e}       {int((condition > bar).sum())} "
            f"({100 * (condition > bar).mean():.2f}%)"
        )
    print(flush=True)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in STATES:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <{' | '.join(STATES)}>")
    name = sys.argv[1]
    march_beta, _, description = STATES[name]
    pc_beta = max(march_beta, FLOOR) if march_beta > 0 else 0.0

    coupled = compare.build_case()["coupled"]
    n_fields = coupled.layout.dim + 3
    groups = FieldGroups(
        n_cells=coupled.layout.n_cells,
        n_leading_fields=coupled.layout.dim + 1,
        n_trailing_fields=N_TRAILING,
    )
    print(f"{'=' * 96}\ncell-block conditioning, [k, omega] block, {name} -- {description}")
    print(f"operator beta {march_beta}, preconditioner beta {pc_beta}\n{'=' * 96}", flush=True)

    state = load_state(name)
    colouring = _coupled_jacobian_colouring(coupled, 3)
    structure = block_stencil_gather_map(colouring, n_fields)
    base = _coupled_shift_policy(coupled, state, "twolevel")
    jacobian = materialize(coupled, state, colouring, structure, n_fields)
    shift = _frozen_shift_diagonal(base, pc_beta, state) if pc_beta > 0 else np.zeros(groups.n_dofs)
    block = sp.csr_matrix(
        MonolithicAmgPreconditioner._shifted(jacobian, shift)[groups.trailing, :][
            :, groups.trailing
        ]
    )
    del jacobian
    gc.collect()

    raw = cell_blocks(block, N_TRAILING)
    scaled, _ = symmetrically_equilibrate(block)
    equilibrated = cell_blocks(sp.csr_matrix(scaled), N_TRAILING)

    print()
    report("RAW (what the block smoother sees today)", raw)
    report("EQUILIBRATED (D A D)", equilibrated)

    # The mechanism, stated as a number: rescaling normalizes the diagonal, so whatever coupling the
    # block has ends up expressed relative to it. If the off-diagonal grows in those terms, the block
    # becomes harder to invert on its own even though the operator as a whole became better scaled.
    for label, blocks in (("raw", raw), ("equilibrated", equilibrated)):
        diagonal = np.abs(blocks[:, 0, 0] * blocks[:, 1, 1])
        off = np.abs(blocks[:, 0, 1] * blocks[:, 1, 0])
        ratio = off / np.maximum(diagonal, np.finfo(float).tiny)
        print(
            f"  {label:<13} |a_kw a_wk| / |a_kk a_ww|:  p50 {np.median(ratio):.3e}  "
            f"p99 {np.percentile(ratio, 99):.3e}  "
            f"cells within 1% of 1.0: {int((np.abs(ratio - 1.0) < 0.01).sum())}",
            flush=True,
        )


if __name__ == "__main__":
    main()
