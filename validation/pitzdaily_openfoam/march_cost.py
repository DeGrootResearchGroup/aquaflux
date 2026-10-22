"""What one shipped pitzDaily march costs: wall clock, refits, and the process's peak memory.

The before-and-after instrument for a change to how the preconditioner is fitted or held: it runs the
case file's march (``compare.py``'s ``solve_aquaflux``, no study overrides unless the environment sets
them) and prints one record of

- the march: steps, cumulative restart cycles, final stopping residual, ``x_r/h`` and the peak eddy
  viscosity ratio -- the counts two arms must share before their seconds can be compared;
- the clock: the whole solve, and the march clock at its first and last step (the first step carries
  the compilation);
- the refits: their count and the summed seconds of each phase, read from the march log's ``pc`` lines
  (probe, assemble, refactor);
- memory: the process's resident high-water mark (``VmHWM``) and resident size at the end
  (``VmRSS``), from ``/proc/self/status``. On a CPU backend JAX's device buffers live in this process,
  so the high-water mark is host and device together.

Unlike ``compare.py``'s ``main`` it needs no OpenFOAM reference fields. Run it with
``validation/run_case.sh``, and run an arm twice before reading its seconds: the first run on a new
checkout partly measures compilation.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import jax  # noqa: E402
import numpy as np  # noqa: E402

#: A step row of the march log's summary table: ``| step | t(s) | beta | in | cyc | R | ...``.
STEP_ROW = re.compile(
    r"^\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+[0-9.]+\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+(\S+)\s+\|"
)
#: A refit line: ``| pc full 1.7s (probe 1.0 assemble 0.0 refactor 0.7)``.
REFIT_ROW = re.compile(
    r"^\|\s+pc\s+(\w+)\s+([0-9.]+)s\s+\(probe\s+([0-9.]+)\s+assemble\s+([0-9.]+)\s+refactor\s+([0-9.]+)\)"
)
CUMULATIVE_ROW = re.compile(r"^\|\s+cum\s+(\d+)")


def memory_kib() -> dict[str, int]:
    """``VmHWM`` and ``VmRSS`` of this process, in KiB."""
    fields = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in ("VmHWM", "VmRSS"):
            fields[name] = int(value.split()[0])
    return fields


def read_log(path: Path) -> dict:
    """The step table's last row, the cycle count, and the refits' phase sums."""
    steps, refits, cycles = [], [], 0
    for line in path.read_text().splitlines():
        if match := STEP_ROW.match(line):
            steps.append((int(match[1]), int(match[2]), match[3]))
        elif match := REFIT_ROW.match(line):
            refits.append((match[1], *(float(match[i]) for i in range(2, 6))))
        elif match := CUMULATIVE_ROW.match(line):
            cycles = int(match[1])
    if not steps:
        raise SystemExit(f"[march_cost] no step rows in {path}: has the log's table changed shape?")
    totals = np.sum([r[1:] for r in refits], axis=0) if refits else np.zeros(4)
    return {
        "steps": steps[-1][0],
        "first_step_clock": steps[0][1],
        "march_clock": steps[-1][1],
        "final_R": steps[-1][2],
        "cycles": cycles,
        "refits": len(refits),
        "refit_kinds": {kind: sum(r[0] == kind for r in refits) for kind in {r[0] for r in refits}},
        "refit_seconds": dict(zip(("total", "probe", "assemble", "refactor"), totals, strict=True)),
    }


def main() -> None:
    import compare

    commit = subprocess.run(
        ["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print(
        f"[march_cost] commit {commit}, jax {jax.__version__}, devices {jax.devices()}", flush=True
    )
    log_path = HERE / "march_cost.log"
    started = time.perf_counter()
    solution = compare.solve_aquaflux(log_path=log_path)
    solve_seconds = time.perf_counter() - started
    memory = memory_kib()
    march = read_log(log_path)
    x_r = compare.reattachment_length(solution["centroid"], solution["U"][:, 0])
    nut_peak = float(solution["nut"].max() / compare.NU)
    seconds = march["refit_seconds"]
    print(
        f"[march_cost] march: {march['steps']} steps, {march['cycles']} cycles, final R "
        f"{march['final_R']}, x_r/h {x_r:.4f}, nut/nu peak {nut_peak:.2f}\n"
        f"[march_cost] clock: solve {solve_seconds:.1f} s; march clock {march['march_clock']} s, "
        f"{march['first_step_clock']} s of it to the end of step 1\n"
        f"[march_cost] refits: {march['refits']} {march['refit_kinds']}, "
        f"{seconds['total']:.1f} s (probe {seconds['probe']:.1f}, assemble "
        f"{seconds['assemble']:.1f}, refactor {seconds['refactor']:.1f})\n"
        f"[march_cost] memory: VmHWM {memory['VmHWM'] / 2**20:.3f} GiB, VmRSS at the end "
        f"{memory['VmRSS'] / 2**20:.3f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
