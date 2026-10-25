"""What one product with a step case's widest preconditioner operator costs, compiled and as host kernels.

A field-split application on bfs3d spends most of its time in its hierarchies' level products ``A x``,
padded-row (ELLPACK) products. Written in JAX, XLA compiles one as two kernels: a gather fusion that
writes ``values * x[columns]`` out as a whole ``(n_rows * width)`` array, then a row-sum kernel that reads
it back (on Linux a single-threaded custom fusion); on bfs3d's leading fine level that intermediate is
177 MiB. On the CPU the library now computes it as a fused Numba loop called from inside the compiled
program (``aquaflux.precondition.fused_product``). This times, on the same operator and vector,

* ``operator apply`` -- the operator's own ``apply``, compiled alone: the fused loop where the library
  chooses it, XLA's two kernels under ``AQUAFLUX_DISABLE_FUSED_PRODUCT=1`` (the banner says which);
* ``scipy csr``   -- ``scipy``'s CSR product (one thread), as a reference point;
* ``numba csr``   -- a fused CSR product (one pass over the stored entries, no intermediate), threaded
  over rows with ``numba.prange`` on every core, in the operator's own precision;
* ``numba ell``   -- the same fused, threaded loop over the padded-row arrays the traced operator holds.

Each is the mean of ``SPMV_REPEATS`` back-to-back calls after a warm one, with the cores it kept busy.
The host results are checked against the compiled product (largest difference over the largest entry of
the result). The operators are the case's leading and trailing fine levels, fitted at the case's own
settings at its hybrid start (``SPMV_CASE``: ``bfs3d`` or ``pitzdaily``).

The host kernels run outside any compiled program; they bound what a fused loop can do, which the
``operator apply`` row with the library's fused product shows from inside one.

Usage
-----
    SPMV_CASE=bfs3d validation/run_case.sh validation/spmv_kernel_probe.py --wait
"""

from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from step_cases import load_case  # noqa: E402

CASE = os.environ.get("SPMV_CASE", "bfs3d")
REPEATS = int(os.environ.get("SPMV_REPEATS", "30"))


def _timed(f, *args):
    """``(mean ms, cores)`` of ``REPEATS`` back-to-back calls of ``f(*args)`` after one warm call."""
    f(*args)
    before = resource.getrusage(resource.RUSAGE_SELF)
    started = time.perf_counter()
    for _ in range(REPEATS):
        f(*args)
    wall = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    return 1e3 * wall / REPEATS, cpu / wall


def _kernels():
    import numba
    import numpy as np

    @numba.njit(parallel=True, fastmath=False, cache=False)
    def csr(indptr, indices, data, x, out):
        for i in numba.prange(out.shape[0]):
            total = data.dtype.type(0.0)
            for k in range(indptr[i], indptr[i + 1]):
                total += data[k] * x[indices[k]]
            out[i] = total

    @numba.njit(parallel=True, fastmath=False, cache=False)
    def ell(columns, values, x, out):
        for i in numba.prange(out.shape[0]):
            total = values.dtype.type(0.0)
            for k in range(values.shape[1]):
                total += values[i, k] * x[columns[i, k]]
            out[i] = total

    return csr, ell, np


def _compare(name, operator):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    csr_kernel, ell_kernel, _ = _kernels()
    matrix = operator.to_scipy().tocsr()
    dtype = np.asarray(operator.values).dtype
    matrix = matrix.astype(dtype)
    matrix.sort_indices()
    columns = np.ascontiguousarray(np.asarray(operator.columns), dtype=np.int32)
    values = np.ascontiguousarray(np.asarray(operator.values))
    x = np.random.default_rng(0).standard_normal(matrix.shape[1]).astype(dtype)
    x_device = jnp.asarray(x)
    out = np.empty(matrix.shape[0], dtype=dtype)
    print(
        f"\n{name}: {matrix.shape[0]} rows, {matrix.nnz} stored entries, padded width {values.shape[1]} "
        f"({values.size} slots), {dtype}",
        flush=True,
    )

    traced = eqx.filter_jit(lambda op, v: op.apply(v))
    reference = np.asarray(traced(operator, x_device))
    scale = float(np.max(np.abs(reference)))
    rows = [
        ("operator apply", lambda: jax.block_until_ready(traced(operator, x_device)), None),
        ("scipy csr", lambda: matrix @ x, None),
        (
            "numba csr",
            lambda: csr_kernel(matrix.indptr, matrix.indices, matrix.data, x, out),
            lambda: out.copy(),
        ),
        ("numba ell", lambda: ell_kernel(columns, values, x, out), lambda: out.copy()),
    ]
    print("| product | mean ms | cores | max diff / max |result| |\n|---|---|---|---|")
    for label, call, result in rows:
        mean, cores = _timed(call)
        if result is None:
            got = np.asarray(call())
        else:
            call()
            got = result()
        diff = float(np.max(np.abs(got.astype(np.float64) - reference))) / scale
        print(f"| {label} | {mean:.2f} | {cores:.2f} | {diff:.1e} |", flush=True)


def main():
    case = load_case(CASE)

    import aquaflux  # noqa: F401  (enables x64)
    import jax
    import numba
    from aquaflux.initialization import hybrid_initialize
    from aquaflux.precondition.fused_product import fused_products
    from aquaflux.turbulence import open_session

    print(
        f"[configuration] case {CASE}, the hybrid start, operator apply "
        f"{'fused' if fused_products() else 'gathered (XLA)'}, jax {jax.__version__}, numba {numba.__version__} "
        f"on {numba.config.NUMBA_NUM_THREADS} threads, {os.cpu_count()} cpus, {REPEATS} calls per row",
        flush=True,
    )
    coupled = case.build_case()["coupled"]
    state = coupled.state_from_physical(*hybrid_initialize(coupled))
    session = open_session(case.SOLVER.settings()["preconditioner"], coupled)
    session.build(state, dual_time=case.SOLVER.dual_time)
    inverse = session.preconditioner.inverse
    for name, block in (
        ("leading fine level", inverse._leading),
        ("trailing fine level", inverse._trailing),
    ):
        _compare(name, block._hierarchy.levels[0].operator)


if __name__ == "__main__":
    main()
