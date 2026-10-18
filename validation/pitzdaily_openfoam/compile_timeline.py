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

:func:`breakdown` answers the finer question -- does a refit, a station change or a redo recompile -- by
splitting each step where its own work begins and at each redo, and by summing the tracing and lowering
lines apart from the compilations: the persistent cache skips only the XLA compilation, so a program it
serves is still traced and lowered every run, and on a warm run that is most of what is left.

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
from dataclasses import dataclass, field

_STEP_ROW = re.compile(r"^\|\s+(\d+) \|\s+(\d+) \|\s+[0-9.]+ \|")
_COMPILED = re.compile(r"Finished XLA compilation of (\S+) in ([0-9.e-]+) sec")
_CACHE_HIT = re.compile(r"Persistent compilation cache hit for ")
_TRACED = re.compile(r"Finished tracing (\S+) for jit in ([0-9.e-]+) sec")
_LOWERED = re.compile(r"Finished jaxpr to MLIR module conversion (\S+) in ([0-9.e-]+) sec")
_INNER_ROW = re.compile(r"^    \|\s+(\d+) \|\s+\d+ \|")
_REDO = re.compile(r"^\s+-> redo step (\d+) \(attempt (\d+)\): (\w+)")
_REFRESH = re.compile(r"^\| (pc [^|(]*?)\s+[0-9.]+s\b")


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


@dataclass
class Phase:
    """What one phase of a march log paid to compile, trace and lower, programs over a floor named.

    ``compiled`` and ``hits`` are ``(count, seconds)`` of XLA compilations and of programs read from
    the persistent cache (the seconds of a hit are its read). ``traced`` and ``lowered`` sum the
    tracing and the jaxpr-to-MLIR lowering of every program, hit or not: the persistent cache skips
    only the XLA compilation, so a hit still pays both.
    """

    name: str
    compiled: list = field(default_factory=lambda: [0, 0.0])
    hits: list = field(default_factory=lambda: [0, 0.0])
    traced: list = field(default_factory=lambda: [0, 0.0])
    lowered: list = field(default_factory=lambda: [0, 0.0])
    large: list = field(default_factory=list)
    refresh: str = ""

    @property
    def seconds(self) -> float:
        """Compilation, cache reads, tracing and lowering together."""
        return self.compiled[1] + self.hits[1] + self.traced[1] + self.lowered[1]


def breakdown(lines, floor: float = 0.1):
    """Every phase of a march log, finer than :func:`timeline`, as a list of :class:`Phase`.

    A step ``N`` is split where its own work begins: ``step N set-up`` runs from the previous step's
    row to step ``N``'s first inner-iteration row -- the station change, the preconditioner refit due
    at its start, the measure rebuilt at its state and the compiled step's dispatch -- and ``step N
    inner`` from there to its row; a redone attempt opens ``step N redo K (reason)``, which holds the
    re-fit the redo asks for. Inner rows are written only by a march logged with ``"inner"`` detail;
    without them a step is one ``step N set-up`` phase. The refresh a step's row reports (its ``pc``
    line, which follows the row) is attached to all of that step's phases. Each program over
    ``floor`` seconds -- compiled, read from the cache, traced or lowered -- is named in ``large``.
    """
    phases = [Phase("before step 1")]
    step = 1
    by_step: dict[int, list[Phase]] = {1: phases}
    hit_pending = False

    def open_phase(name: str) -> Phase:
        phases.append(Phase(name))
        by_step.setdefault(step, []).append(phases[-1])
        return phases[-1]

    for line in lines:
        current = phases[-1]
        if row := _STEP_ROW.match(line):
            step = int(row.group(1)) + 1
            open_phase(f"step {step} set-up")
            continue
        if inner := _INNER_ROW.match(line):
            if int(inner.group(1)) == 0 and current.name == f"step {step} set-up":
                open_phase(f"step {step} inner")
            continue
        if redo := _REDO.match(line):
            open_phase(f"step {step} redo {redo.group(2)} ({redo.group(3)})")
            continue
        if refresh := _REFRESH.match(line):
            for phase in by_step.get(step - 1, []):
                phase.refresh = refresh.group(1)
            continue
        if "aquaflux coupled solve:" in line:
            open_phase("after the march")
            step = -1
            continue
        if _CACHE_HIT.search(line):
            hit_pending = True
            continue
        for pattern, bucket, kind in (
            (_COMPILED, None, "compile"),
            (_TRACED, "traced", "trace"),
            (_LOWERED, "lowered", "lower"),
        ):
            if found := pattern.search(line):
                seconds = float(found.group(2))
                if bucket is None:
                    bucket = "hits" if hit_pending else "compiled"
                    kind = "hit" if hit_pending else kind
                    hit_pending = False
                tally = getattr(current, bucket)
                tally[0] += 1
                tally[1] += seconds
                if seconds >= floor:
                    current.large.append((seconds, kind, found.group(1)))
                break
    return [phase for phase in phases if phase.seconds > 0.0 or phase.large]


def print_breakdown(phases) -> None:
    """The :func:`breakdown` table, one line per phase and one per program over its floor."""
    print(
        f"{'phase':34s} {'refresh':10s} {'compiled':>13s} {'cache hits':>13s} "
        f"{'traced':>13s} {'lowered':>13s}"
    )
    for phase in phases:
        cells = " ".join(
            f"{tally[0]:5d} {tally[1]:6.2f}s"
            for tally in (phase.compiled, phase.hits, phase.traced, phase.lowered)
        )
        print(f"{phase.name:34s} {phase.refresh[3:]:10s} {cells}")
        for seconds, kind, name in sorted(phase.large, reverse=True):
            print(f"{'':36s}{seconds:6.2f} s {kind:7s} {name[:80]}")


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
    print()
    with open(path) as log:
        print_breakdown(breakdown(log.read().splitlines()))


if __name__ == "__main__":
    main(sys.argv[1])
