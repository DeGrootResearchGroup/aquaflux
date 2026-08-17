"""Does a maximum-product diagonal permutation have anything to fix on this matrix?

A maximum-product matching (MC64 and relatives) permutes rows so that the product of the diagonal
magnitudes is maximal, and pairs that with a two-sided scaling. Its whole mechanism is to put large
entries on the diagonal, so that an elimination without pivoting does not divide by something tiny.
That mechanism can only help a matrix whose diagonal is **not** already the largest entry in its row.

This asks that question directly, and costs no solve: assemble the shifted, equilibrated, cell-major
**monolithic** matrix, then for every row report whether the diagonal carries the row's largest
magnitude, and by how much it is beaten when it does not. Reported **per field**, because a
whole-matrix count averages any one field into four times as many rows of the others and hides it.

⚠️ **The monolithic matrix is NOT what the case's smoother factorizes, and the per-field rows must be
read with that in mind.** This case runs with a field split: the ``[u, v, p]`` saddle goes to the
algebraic-multigrid V-cycle whose level smoother is the incomplete factorization, and the
``[k, omega]`` pair goes to a nodal hierarchy that is **not** an incomplete factorization at all. So
the ``k`` and ``omega`` rows here are rows **no ILU ever eliminates**, and their (large) off-diagonal
ratios say nothing about the shipped preconditioner. Only ``u``, ``v`` and ``p`` are on the path that
an incomplete factorization actually sees -- and even those are measured here with their cross-field
couplings to ``k`` and ``omega`` still present, which the split removes from the factorized block.
Read this as a survey of the coupled Jacobian, not as a measurement of the smoother's input.

Read it as a gate, not as a verdict: a diagonal that already dominates means a maximum-product
permutation has no work to do here, and that is a reason not to spend a day building one. A diagonal
that does not dominate does not by itself promise the permutation will help.

Run from the repo root::

    validation/run_case.sh validation/pitzdaily_openfoam/mc64_precheck.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "pitzdaily_openfoam"))

import compare  # noqa: E402
import ilu0_remedy_sweep as sweep  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from aquaflux.turbulence.coupled import (  # noqa: E402
    _DEFAULT_SHIFT_BASIS,
    _frozen_shift_diagonal,
    _monolithic_shift_source,
)
from ilu_fill_probe import FIELDS2, FIELDS3  # noqa: E402


def diagonal_dominance(matrix, n_fields: int, names: tuple[str, ...]) -> None:
    """Per field: how often the diagonal is NOT the largest magnitude in its row, and by how much.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        The equilibrated, cell-major matrix, shape ``(n_fields * n_cells, n_fields * n_cells)``.
    n_fields : int
        Fields per cell; in cell-major order a row's field is ``row % n_fields``.
    names : tuple of str
        Field names, in the order they are interleaved.
    """
    diagonal = np.abs(matrix.diagonal())
    indptr, indices, data = matrix.indptr, matrix.indices, np.abs(matrix.data)
    rows = matrix.shape[0]
    largest_off = np.zeros(rows)
    for row in range(rows):
        start, end = indptr[row], indptr[row + 1]
        columns, values = indices[start:end], data[start:end]
        off = values[columns != row]
        largest_off[row] = off.max() if off.size else 0.0
    # A ratio above one is a row whose diagonal is beaten by one of its own off-diagonal entries,
    # which is the only condition a maximum-product permutation can improve.
    ratio = np.where(diagonal > 0.0, largest_off / np.where(diagonal > 0.0, diagonal, 1.0), np.inf)
    field_of_row = np.arange(rows) % n_fields
    print(f"    {'field':<8}{'rows':>8}{'beaten':>9}{'%':>7}{'median':>11}{'p99':>11}{'max':>11}")
    for field, name in enumerate(names):
        mask = field_of_row == field
        r = ratio[mask]
        beaten = int((r > 1.0).sum())
        print(
            f"    {name:<8}{r.size:>8}{beaten:>9}{100.0 * beaten / r.size:>6.1f}%"
            f"{np.median(r):>11.3e}{np.quantile(r, 0.99):>11.3e}{r.max():>11.3e}",
            flush=True,
        )


def main():
    coupled = compare.build_case()["coupled"]
    n_fields = coupled.layout.dim + 3
    names = FIELDS2 if coupled.layout.dim == 2 else FIELDS3
    state = sweep.load_state(coupled, None)
    base = _monolithic_shift_source(coupled, state, _DEFAULT_SHIFT_BASIS)
    print(
        f"pitzDaily, {coupled.layout.n_cells} cells, {n_fields} fields, "
        f"|R| {float(jnp.linalg.norm(coupled.residual(state))):.4e}",
        flush=True,
    )
    print(
        "ratio = (largest off-diagonal magnitude) / |diagonal|, per row; > 1 means the diagonal "
        "is NOT the row's largest entry",
        flush=True,
    )
    for beta in sweep.BETAS:
        shift = _frozen_shift_diagonal(base, beta, state)
        arm = sweep.Arm("precheck")
        cell_major, _scale, _perm = sweep.assemble(coupled, state, arm, shift, n_fields)
        print(
            f"\n  beta {beta} (equilibrated, cell-major -- what the smoother factorizes)",
            flush=True,
        )
        diagonal_dominance(cell_major.tocsr(), n_fields, names)


if __name__ == "__main__":
    main()
