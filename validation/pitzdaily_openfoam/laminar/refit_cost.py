"""What the laminar pitzDaily march's refits cost: the single-block preconditioner's refit, timed.

Solves ``case.yaml`` (beside this file) from its file, in this process, with the session's refit hook
wrapped, and prints one record: build and solve seconds, the process's peak resident memory
(``ru_maxrss``; on a CPU backend the device buffers are in this process), the refit count and their
seconds, summed by phase as the refit reports them (probe, assemble, refactor). The march's step count
is the case's own record; two arms must share it before their seconds compare.

Usage
-----
    validation/run_case.sh validation/pitzdaily_openfoam/laminar/refit_cost.py --wait
"""

from __future__ import annotations

import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np  # noqa: E402
from aquaflux.case import read_case  # noqa: E402
from aquaflux.precondition import materialized_session  # noqa: E402


def main() -> None:
    timings = []
    refit = materialized_session.BetaTrackingRefresh._materialize_at

    def timed(self, *args, **kwargs):
        started = time.perf_counter()
        phases = refit(self, *args, **kwargs)
        timings.append((time.perf_counter() - started, dict(phases)))
        return phases

    materialized_session.BetaTrackingRefresh._materialize_at = timed
    checked = read_case(HERE / "case.yaml").check()
    started = time.perf_counter()
    checked.solve(checked.build())
    seconds = time.perf_counter() - started
    commit = subprocess.run(
        ["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    total = np.array([seconds for seconds, _ in timings])
    parts = {
        name: sum(phases.get(name, 0.0) for _, phases in timings)
        for name in ("probe", "assemble", "refactor")
    }
    print(
        f"[refit_cost] commit {commit}: build and solve {seconds:.1f} s, peak resident "
        f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.3f} GiB, {len(timings)} refits "
        f"{total.sum():.1f} s (median {np.median(total):.3f} s; probe {parts['probe']:.1f}, assemble "
        f"{parts['assemble']:.1f}, refactor {parts['refactor']:.1f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
