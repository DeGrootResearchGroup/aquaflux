"""What the coloured probe's batch size buys, in time, and costs, in memory.

``probe_batch_size`` chunks the coloured probes into batched directional derivatives. A larger chunk
amortizes dispatch over more probes; it also holds more simultaneous forward-mode tangents. The shipped
default was chosen against a peak that has since been rebuilt -- the seeds and the response array used
to dominate it and neither scaled with the batch -- so the trade needs re-measuring rather than
inheriting.

**Both axes are measured here, because either alone picks the wrong number.** Time alone says "bigger is
better" until the machine swaps; memory alone says "smaller is better" and gives up free throughput. The
timing follows the discipline three earlier attempts at this needed: a discarded warm-up (each chunk
size is its own compiled shape, so the first call at each size pays a trace), repetitions, and the
minimum reported, since a shared machine only ever adds time.

Peak is measured two ways because they answer different questions: ``tracemalloc`` sees the Python-side
arrays this module allocates, and the process's own maximum resident set sees everything including the
JAX runtime's device buffers. The second is the one that decides whether a machine swaps.

Usage::

    python3 -u validation/bfs3d_openfoam/probe_batch_sweep.py [state-00077]
"""

from __future__ import annotations

import gc
import os
import resource
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
from aquaflux.solve import block_stencil_gather_map  # noqa: E402
from aquaflux.solve.amg_preconditioner import MonolithicAmgPreconditioner  # noqa: E402
from aquaflux.turbulence.coupled import (  # noqa: E402
    _batched_jacobian_matvec,
    _coupled_jacobian_plan,
    _jacobian_matvec,
)

#: Chunk sizes to sweep. Kept modest at the top end deliberately: a materialize of a three-dimensional
#: coupled Jacobian is already the largest allocation in the process, and the point of the sweep is to
#: find where the return flattens, not to find the size that exhausts the machine.
BATCH_SIZES = (1, 2, 4, 8, 16, 32)
REPEATS = 3

#: Stop the sweep once the process is this large. Not a tuning parameter -- a guard. Driving a
#: three-dimensional coupled materialize into swap suspends the whole machine, which is a failure you
#: cannot debug from inside the run that caused it.
RSS_CEILING_MB = float(os.environ.get("BFS3D_SWEEP_RSS_CEILING_MB", "6000"))


def peak_rss_mb():
    """The process's maximum resident set so far, in MB (a high-water mark, never decreasing)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1e3  # bytes on macOS, kB on Linux


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "state-00077"
    case = compare.build_case()
    coupled = case["coupled"]
    plan = _coupled_jacobian_plan(coupled, 3, compare.COLUMN_REACH)
    structure = block_stencil_gather_map(plan)
    with np.load(CASE / "checkpoints" / f"{name}.npz") as data:
        state = np.asarray(data["state"])
        print(
            f"[state] {name}: step {int(data['step'])}, shift {float(data['shift']):.6g}",
            flush=True,
        )
    print(
        f"[plan] reach {plan.reach}, {plan.n_probes} probes, {structure.nnz / 1e6:.1f}M nnz"
        f"  (matrix data alone {structure.nnz * 8 / 1e6:.0f} MB)",
        flush=True,
    )

    def materialize(batch):
        return MonolithicAmgPreconditioner._materialize_jacobian(
            lambda v: _jacobian_matvec(coupled, state, v),
            plan,
            lambda seeds: _batched_jacobian_matvec(coupled, state, seeds),
            batch,
            structure,
        )

    print(f"\n  {'batch':>6s} {'wall (s)':>10s} {'cpu (s)':>9s} {'python peak':>13s} {'rss':>9s}")
    baseline = None
    for batch in BATCH_SIZES:
        # Stop climbing before the machine starts swapping rather than after. A materialize is already
        # the largest allocation in the process, and this sweep deliberately makes it larger; running a
        # 3D case into swap does not merely slow the sweep, it takes the whole machine down with it.
        if peak_rss_mb() > RSS_CEILING_MB:
            print(
                f"  stopping before batch {batch}: resident set is {peak_rss_mb():.0f} MB,"
                f" over the {RSS_CEILING_MB:.0f} MB ceiling",
                flush=True,
            )
            break
        warm = materialize(batch)  # this chunk shape compiles on its first call; discard it
        del warm
        gc.collect()
        walls, cpus = [], []
        tracemalloc.start()
        for _ in range(REPEATS):
            w, c = time.perf_counter(), time.process_time()
            jacobian = materialize(batch)
            walls.append(time.perf_counter() - w)
            cpus.append(time.process_time() - c)
            del jacobian
            gc.collect()
        _, python_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        wall, cpu = min(walls), min(cpus)
        baseline = wall if baseline is None else baseline
        print(
            f"  {batch:6d} {wall:10.2f} {cpu:9.2f} {python_peak / 1e6:11.0f} MB"
            f" {peak_rss_mb():7.0f} MB     ({baseline / wall:.2f}x vs batch 1)",
            flush=True,
        )

    print(
        "\n  Read the rss column as a high-water mark for the WHOLE process, so it only ever rises"
        "\n  across the sweep; the batch-to-batch difference is what it costs, not the absolute.",
        flush=True,
    )


if __name__ == "__main__":
    main()
