"""Where a march spends its XLA compilation: per step, from a log written with ``JAX_LOG_COMPILES=1``.

A march run with JAX's compile logging on prints one ``Finished XLA compilation of <program> in <s> sec``
line per compiled program, interleaved with its own step table. This reads such a log and totals the
compilation before step 1, between consecutive steps and after the march, with the largest programs.
It answers two questions without re-instrumenting anything: what a first run pays to compile, and
whether a station change recompiles (it should be a compilation-cache hit).

A program read from the persistent compilation cache logs the same ``Finished XLA compilation`` line,
carrying the time to read it, immediately after a ``Persistent compilation cache hit`` line. Those are
counted as cache hits, apart from the compilations; counting them as compilations made a warm run read
as if it had compiled every program it loaded.

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
_CACHE_HIT = re.compile(r"Persistent compilation cache hit for ")


def timeline(lines):
    """``(phases, seconds, counts, compiles, hits)`` from a march log's lines, in order of phase.

    ``seconds`` and ``counts`` are the XLA compilations per phase and ``compiles`` lists them as
    ``(seconds, phase, program)``; ``hits`` counts per phase the programs read from the persistent
    cache instead, which none of the other three include.
    """
    phase = "before step 1"
    phases, seconds, counts, compiles, hits = [phase], Counter(), Counter(), [], Counter()
    hit_pending = False
    for line in lines:
        if row := _STEP_ROW.match(line):
            phase = f"after step {row.group(1)}"
            phases.append(phase)
            continue
        if "aquaflux coupled solve:" in line:
            phase = "after the march"
            phases.append(phase)
        if _CACHE_HIT.search(line):
            hit_pending = True
            continue
        if compiled := _COMPILED.search(line):
            if hit_pending:
                hits[phase] += 1
                hit_pending = False
                continue
            seconds[phase] += float(compiled.group(2))
            counts[phase] += 1
            compiles.append((float(compiled.group(2)), phase, compiled.group(1)))
    active = [p for p in dict.fromkeys(phases) if counts[p] or hits[p]]
    return active, seconds, counts, compiles, hits


def main(path):
    with open(path) as log:
        phases, seconds, counts, compiles, hits = timeline(log.read().splitlines())
    for phase in phases:
        print(
            f"{phase:16s} {counts[phase]:5d} compiles {seconds[phase]:8.1f} s"
            f"  {hits[phase]:4d} cache hits"
        )
    print(
        f"total {sum(counts.values())} compiles, {sum(seconds.values()):.1f} s; "
        f"{sum(hits.values())} cache hits"
    )
    for time, phase, name in sorted(compiles, reverse=True)[:12]:
        print(f"  {time:7.1f} s  {phase:16s} {name[:90]}")


if __name__ == "__main__":
    main(sys.argv[1])
