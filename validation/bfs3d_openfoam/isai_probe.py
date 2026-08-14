"""Can a sparse pattern represent the inverses of this operator's incomplete factors?

An incomplete factorization is the incumbent preconditioner here and it wins on CPU, but its application
is a forward and a backward triangular solve -- sequential, and the one property that does not move to an
accelerator. The incomplete sparse approximate inverse replaces those solves with two sparse matrix-vector
products: approximate ``L^-1`` and ``U^-1`` on a prescribed pattern, then apply them. The factorization,
and therefore its numerical quality, is kept; only the *application* changes.

That matters for a specific reason measured on this case. A shift makes an incomplete factorization nearly
exact -- its error is second order in diagonal dominance where a relaxation's is first order -- which is
why the incumbent collapses to two restart cycles as soon as the pseudo-transient shift appears, and why a
multigrid cannot follow it there. An approximate inverse OF the factorization inherits that behaviour; a
smoother-based hierarchy structurally cannot.

**What this script measures, and why it is a sample.** For row ``i`` with the sparsity pattern ``S`` of
``L``'s row ``i``, minimizing the residual of ``M_L L = I`` restricted to that pattern gives
``M_L[i, S] L[S, S] = e_i[S]``. Since ``L`` is lower triangular and every column of ``S`` is at most ``i``,
``L[S, S]`` is itself lower triangular, so each row is a small triangular solve and every row is
independent. Building all of them is minutes of Python dominated by sparse fancy-indexing; the question
that decides whether the direction is worth pursuing does not need all of them. A few thousand sampled
rows give the distribution of ``||e_i - M_L[i,S] L[S,:]||``, and if a sparse pattern cannot represent these
inverses that shows up immediately and cheaply.

Read the result against the SIMPLE smoother's splitting error on the same operator, which is **1.449** --
a 145 % error. An approximate inverse worth building should be far below that.
"""

from __future__ import annotations

import os
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular

#: Rows sampled per factor. The statistic wanted is a distribution, not a norm, so a few thousand rows out
#: of ninety thousand is ample -- and the whole point is to answer the question without the full build.
SAMPLE = int(os.environ.get("BFS3D_ISAI_SAMPLE", "2000"))


def row_residuals(factor: sp.csr_matrix, lower: bool, sample: int, seed: int = 0) -> np.ndarray:
    """``||e_i - M[i,S] factor[S,:]||`` for a sample of rows, with ``M[i,S]`` the pattern-restricted fit.

    Parameters
    ----------
    factor : scipy.sparse matrix
        A triangular factor, compressed sparse row.
    lower : bool
        Whether it is lower triangular. Only the orientation of the small solve depends on it.
    sample : int
        How many rows to draw.
    seed : int
        Draw seed, so a repeat is comparable.

    Returns
    -------
    np.ndarray
        One residual per sampled row.
    """
    factor = factor.tocsr()
    n = factor.shape[0]
    rows = np.random.default_rng(seed).choice(n, size=min(sample, n), replace=False)
    out = np.empty(rows.size)
    for slot, i in enumerate(rows):
        pattern = np.sort(factor.indices[factor.indptr[i] : factor.indptr[i + 1]])
        block = np.asarray(factor[np.ix_(pattern, pattern)].todense())
        rhs = (pattern == i).astype(float)
        # M[i, S] F[S, S] = e  =>  F[S, S]^T x = e; transposing flips the orientation.
        x = solve_triangular(block.T, rhs, lower=not lower)
        # The residual against the FULL row, not just the pattern -- the entries outside the pattern are
        # exactly what a sparse approximate inverse gives up, so excluding them would flatter it.
        full = np.zeros(n)
        full[pattern] = x
        out[slot] = np.linalg.norm((full @ factor) - np.eye(1, n, i).ravel())
    return out


def report(name: str, residuals: np.ndarray) -> None:
    quantiles = np.quantile(residuals, [0.5, 0.9, 0.99])
    print(
        f"  {name:22s} median {quantiles[0]:.3e}  p90 {quantiles[1]:.3e}  "
        f"p99 {quantiles[2]:.3e}  max {residuals.max():.3e}",
        flush=True,
    )


def main() -> None:
    # ⚠️ NOT YET RUNNABLE. `_leading_block_at` does not exist: the probe builds the shifted leading block
    # inline inside its `main()` (search for `groups.blocks(shifted)`), and factoring that out is a real
    # refactor rather than an import. Doing it is the first step of using this script.
    from field_split_probe import _leading_block_at

    state = os.environ.get("BFS3D_PROBE_STATE", "state-00067")
    print(
        f"incomplete sparse approximate inverse, {state}, {SAMPLE} sampled rows per factor",
        flush=True,
    )
    block = _leading_block_at(state)
    print(f"  flow block: {block.shape[0]} dofs, {block.nnz / 1e6:.1f}M nnz", flush=True)

    started = time.time()
    # `spilu` at a zero drop tolerance and unit fill factor is the closest available stand-in for a
    # fixed-pattern incomplete factorization; it is what the recorded refresh cost was measured on.
    factorization = sp.linalg.spilu(block.tocsc(), drop_tol=0.0, fill_factor=1.0)
    print(f"  factorized in {time.time() - started:.0f}s", flush=True)

    report("L^-1 row residual", row_residuals(factorization.L.tocsr(), True, SAMPLE))
    report("U^-1 row residual", row_residuals(factorization.U.tocsr(), False, SAMPLE))
    print("  (the SIMPLE smoother's splitting error on this operator is 1.449)", flush=True)


if __name__ == "__main__":
    main()
