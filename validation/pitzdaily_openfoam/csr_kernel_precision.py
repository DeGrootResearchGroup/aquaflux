"""Where a float32 multigrid cycle spends its time on CPU: each kernel of the V-cycle, in each precision.

A float32 copy of the field split's flow hierarchy converges the march's systems in exactly the same
cycles as float64 (``precision_replay_probe.py``) but applies about twice as slowly on CPU. This times the
cycle's kernels separately on the real hierarchy -- each level's CSR operator product, its per-cell
block solves, its prolongation, and the coarse dense solve -- and then compares the CSR product against
the alternative a level could use, a gather and a sorted segment sum, on a synthetic matrix of the same
size and density, reporting which compiled kernel each lowers to.

Usage
-----
    validation/run_case.sh validation/pitzdaily_openfoam/csr_kernel_precision.py

It reads the capture ``replay.py`` reads (``PITZ_CHECKPOINT_KEEP=500 PITZ_INNER_DUMP_ABOVE=1``) for the
hierarchy at the target station's first system.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from jax.experimental.sparse import BCSR  # noqa: E402
from precision_replay_probe import _narrowed  # noqa: E402
from replay import MarchReplay  # noqa: E402

PRECISIONS = (jnp.float64, jnp.float32)


def _best(f, *args, repeats=30):
    """Best wall time of ``f(*args)`` in ms, after one warm call; the slower runs measure other load."""
    jax.block_until_ready(f(*args))
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(f(*args))
        best = min(best, time.perf_counter() - started)
    return best * 1e3


def _levels():
    replay = MarchReplay()
    next(replay.systems())  # the first target-station system, with the preconditioner fitted for it
    return replay.preconditioner.inverse._leading._hierarchy.levels


def _kernels(levels):
    rng = np.random.default_rng(0)
    csr = jax.jit(lambda op, x: op.apply(x))
    dense = jax.jit(lambda m, x: m @ x)
    blocks = jax.jit(lambda inverse, y: jnp.einsum("cij,cj->ci", inverse, y))
    for dtype in PRECISIONS:
        cells = []
        for i, level in enumerate(_narrowed(levels, dtype)):
            x = jnp.asarray(rng.standard_normal(level.n), dtype)
            cells.append(f"L{i} csr {_best(csr, level.operator, x):.2f}")
            if level.coarse_inv is not None:
                cells.append(f"L{i} dense {_best(dense, level.coarse_inv, x):.2f}")
            if level.block_inverse is not None:
                y = x.reshape(level.block_size, -1).T
                cells.append(f"L{i} block {_best(blocks, level.block_inverse, y):.2f}")
            if level.p_val is not None:
                coarse = jnp.asarray(rng.standard_normal(level.n_coarse), dtype)
                prolong = jax.jit(
                    lambda r, c, v, y, n=level.n: jax.ops.segment_sum(v * y[c], r, num_segments=n)
                )
                cells.append(
                    f"L{i} prolong {_best(prolong, level.p_frow, level.p_ccol, level.p_val, coarse):.2f}"
                )
        print(f"{np.dtype(dtype).name} (ms): " + " | ".join(cells), flush=True)


def _csr_against_segment_sum(n, per_row):
    """The CSR product's kernel against a gather and sorted segment sum, at the finest level's shape."""
    matrix = sp.random(n, n, density=per_row / n, format="csr", random_state=0) + sp.eye(n)
    matrix = matrix.tocsr()
    matrix.sort_indices()
    rows = jnp.asarray(np.repeat(np.arange(n), np.diff(matrix.indptr)))
    product = jax.jit(lambda d, i, p, x: BCSR((d, i, p), shape=(n, n)) @ x)
    segments = jax.jit(
        lambda d, i, r, x: jax.ops.segment_sum(d * x[i], r, num_segments=n, indices_are_sorted=True)
    )
    for dtype in PRECISIONS:
        data = jnp.asarray(matrix.data, dtype)
        indices, indptr = jnp.asarray(matrix.indices), jnp.asarray(matrix.indptr)
        x = jnp.asarray(np.random.default_rng(0).standard_normal(n), dtype)
        compiled = product.lower(data, indices, indptr, x).compile().as_text()
        targets = sorted(
            {
                line.split("custom_call_target=")[1].split(",")[0]
                for line in compiled.splitlines()
                if "custom_call_target=" in line
            }
        )
        print(
            f"synthetic {n} x {n}, {matrix.nnz} nnz, {np.dtype(dtype).name}: CSR product "
            f"{_best(product, data, indices, indptr, x):.2f} ms (kernel {', '.join(targets)}), "
            f"gather + segment sum {_best(segments, data, indices, rows, x):.2f} ms",
            flush=True,
        )


def main():
    print(f"[configuration] jax {jax.__version__}, {jax.default_backend()}", flush=True)
    levels = _levels()
    print(
        "levels (n, block, nnz): "
        + ", ".join(f"({lv.n}, {lv.block_size}, {lv.operator.data.size})" for lv in levels),
        flush=True,
    )
    _kernels(levels)
    finest = levels[0]
    _csr_against_segment_sum(finest.n, round(finest.operator.data.size / finest.n))


if __name__ == "__main__":
    main()
