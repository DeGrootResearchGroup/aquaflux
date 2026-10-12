"""Where a march spends its XLA compilation: per step, from a log written with ``JAX_LOG_COMPILES=1``.

A march run with JAX's compile logging on prints one ``Finished XLA compilation of <program> in <s> sec``
line per compiled program, interleaved with its own step table. This reads such a log and totals the
compilation before step 1, between consecutive steps and after the march, with the largest programs.
It answers two questions without re-instrumenting anything: what a first run pays to compile, and
whether a station change recompiles (it should be a compilation-cache hit).

Usage
-----
    AQUAFLUX_COMPILATION_CACHE_DIR=<empty dir> JAX_LOG_COMPILES=1 \\
        validation/run_case.sh validation/pitzdaily_openfoam/compare.py
    python3 validation/pitzdaily_openfoam/compile_timeline.py <its run-*.log>

Leave ``AQUAFLUX_COMPILATION_CACHE_DIR`` unset for a warm run; an empty directory makes it cold. A run
after the library changed is partly cold however warm the cache, since every changed program has a new
cache key.
"""

from __future__ import annotations

import re
import sys
from collections import Counter

_STEP_ROW = re.compile(r"^\|\s+(\d+) \|\s+(\d+) \|\s+[0-9.]+ \|")
_COMPILED = re.compile(r"Finished XLA compilation of (\S+) in ([0-9.e-]+) sec")


def timeline(lines):
    """``(phases, seconds, counts, compiles)`` from a march log's lines, in order of phase."""
    phase = "before step 1"
    phases, seconds, counts, compiles = [phase], Counter(), Counter(), []
    for line in lines:
        if row := _STEP_ROW.match(line):
            phase = f"after step {row.group(1)}"
            phases.append(phase)
            continue
        if "aquaflux coupled solve:" in line:
            phase = "after the march"
            phases.append(phase)
        if compiled := _COMPILED.search(line):
            seconds[phase] += float(compiled.group(2))
            counts[phase] += 1
            compiles.append((float(compiled.group(2)), phase, compiled.group(1)))
    return [p for p in dict.fromkeys(phases) if counts[p]], seconds, counts, compiles


def main(path):
    with open(path) as log:
        phases, seconds, counts, compiles = timeline(log.read().splitlines())
    for phase in phases:
        print(f"{phase:16s} {counts[phase]:5d} compiles {seconds[phase]:8.1f} s")
    print(f"total {sum(counts.values())} compiles, {sum(seconds.values()):.1f} s")
    for time, phase, name in sorted(compiles, reverse=True)[:12]:
        print(f"  {time:7.1f} s  {phase:16s} {name[:90]}")


if __name__ == "__main__":
    main(sys.argv[1])
