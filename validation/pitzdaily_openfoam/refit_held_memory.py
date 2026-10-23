"""What a pitzDaily march holds after each refit, with the allocator's kept memory given back.

``march_cost.py``'s high-water mark mixes two things: memory the march holds, and memory it freed that
glibc keeps in the arenas XLA's worker threads allocate from. This runs the case file's march with the
re-fit hook wrapped, and after every refit prints the resident size, the resident size after
``malloc_trim(0)`` returns the kept memory, and the bytes held by live JAX arrays (on a CPU backend
those are in this process too). The median after the trim is the held memory two arms are compared by;
trimming inside the march also lowers its own high-water mark, so read that from ``march_cost.py``.

Usage
-----
    validation/run_case.sh validation/pitzdaily_openfoam/refit_held_memory.py --wait
"""

from __future__ import annotations

import ctypes
import gc
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import jax  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.precondition import materialized_session  # noqa: E402

LIBC = ctypes.CDLL("libc.so.6")


def resident() -> dict[str, float]:
    """``VmRSS`` and ``VmHWM`` of this process, in GiB."""
    fields = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in ("VmHWM", "VmRSS"):
            fields[name] = int(value.split()[0]) / 2**20
    return fields


def main() -> None:
    import compare

    rows = []
    refit = materialized_session.BetaTrackingRefresh._materialize_at

    def measured(self, *args, **kwargs):
        phases = refit(self, *args, **kwargs)
        after = resident()
        gc.collect()
        LIBC.malloc_trim(0)
        trimmed = resident()["VmRSS"]
        live = sum(array.nbytes for array in jax.live_arrays()) / 2**30
        rows.append((after["VmRSS"], trimmed, live))
        print(
            f"[held] refit {len(rows)}: resident {after['VmRSS']:.3f} GiB, after the trim "
            f"{trimmed:.3f} GiB, live arrays {live:.3f} GiB, high-water {after['VmHWM']:.3f} GiB",
            flush=True,
        )
        return phases

    materialized_session.BetaTrackingRefresh._materialize_at = measured
    compare.solve_aquaflux()
    table = np.array(rows)
    print(
        f"[held] {len(rows)} refits: after the trim median {np.median(table[:, 1]):.3f} GiB, max "
        f"{table[:, 1].max():.3f}; live arrays median {np.median(table[:, 2]):.3f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
