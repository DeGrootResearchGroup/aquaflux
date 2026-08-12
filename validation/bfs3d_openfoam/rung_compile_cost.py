"""What each continuation rung pays for its FIRST step, over and above a comparable step of its own.

A Reynolds-continuation march's most expensive step in every rung is that rung's first, and the giveaway
that the cost is not linear algebra is that its Krylov cycle count is no higher -- often lower -- than
the cheap steps around it. The excess is the rung boundary itself: whatever the driver rebuilds there
that the compiled coupled solve is keyed on has to be compiled again.

That is a number worth being able to compute rather than eyeball, because it is the thing a fix has to
move, and because reading it wrongly is easy in two specific ways this script exists to avoid:

* **Subtract the step's own preconditioner cost.** A rung's first step legitimately pays a
  preconditioner refresh, which the log reports (``pc full 15.9s``). Leaving it in attributes solver
  work to compilation.
* **Compare against SAME-CYCLE-COUNT peers.** A step's wall time is dominated by how many Krylov cycles
  it ran, so comparing a first step against its rung's mean step conflates the two. The comparison here
  is against the median of the rung's other steps that ran the same cycle count; if there are none, the
  rung is reported as not comparable rather than compared anyway.

Read the per-rung numbers, not the total: the first rung's excess is the process's unavoidable first
compilation, and only the later rungs' is removable.

Usage::

    python3 validation/bfs3d_openfoam/rung_compile_cost.py <march.log> [<march.log> ...]
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

#: ``[point 2/3 (Re/10)]`` -- the rung marker the driver writes before configuring each point.
_POINT = re.compile(r"^\[point (\d+)/(\d+) \(([^)]*)\)\]")
#: A summary row: ``|    15 |    479 | 0.5000 |  3 |   4 | 1.267e-01 | 1.000 |     |``. The wall-clock
#: column is CUMULATIVE over the whole march, so a step's own wall is the difference from the previous.
_STEP = re.compile(
    r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*[\d.eE+-]+\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.eE+-]+)\s*\|"
)
#: The preconditioner aside: ``| pc full 15.9s (probe 13.4 assemble 0.2 refactor 2.3) |``.
_PC = re.compile(r"^\|\s*pc\s+(\w+)\s+([\d.]+)s")
#: The case-metric aside: ``| xr/h 8.361  xr/h_full 16.14 |``.
_XR = re.compile(r"^\|\s*xr/h\s+([\d.eE+-]+)")


def parse(path: Path) -> tuple[list[dict], str | None]:
    """Every step of a march log, as records carrying their rung, own wall time and own ``pc`` cost.

    Parsed **block by block** rather than with independent scans for each field. The ``pc`` aside is one
    line per step block and the asides are optional, so zipping separate lists of matches silently
    misaligns them by however many a step happened not to write -- a shift that is invisible afterwards,
    because the misaligned table is just as smooth as the correct one.
    """
    rung, previous_wall, records, reattachment = None, 0, [], None
    pending: dict | None = None

    def flush() -> None:
        if pending is not None:
            records.append(pending)

    for line in path.read_text().splitlines():
        if (point := _POINT.match(line)) is not None:
            rung = f"{point.group(1)}/{point.group(2)} {point.group(3)}"
        elif (step := _STEP.match(line)) is not None:
            flush()
            wall = int(step.group(2))
            pending = {
                "rung": rung,
                "step": int(step.group(1)),
                "wall": wall - previous_wall,
                "cycles": int(step.group(4)),
                "residual": float(step.group(5)),
                "pc": 0.0,
            }
            previous_wall = wall
        elif pending is not None and (pc := _PC.match(line)) is not None:
            pending["pc"] += float(pc.group(2))
        elif (xr := _XR.match(line)) is not None:
            reattachment = xr.group(1)
    flush()
    return records, reattachment


def excess(records: list[dict]) -> list[tuple[str, dict]]:
    """Per rung, the first step's non-preconditioner wall against its same-cycle-count peers' median."""
    rungs: dict[str, list[dict]] = {}
    for record in records:
        rungs.setdefault(record["rung"] or "(no rung marker)", []).append(record)

    out = []
    for name, steps in rungs.items():
        first, rest = steps[0], steps[1:]
        peers = [s["wall"] - s["pc"] for s in rest if s["cycles"] == first["cycles"]]
        summary = {
            "steps": len(steps),
            "wall": sum(s["wall"] for s in steps),
            "first_wall": first["wall"],
            "first_cycles": first["cycles"],
            "first_net": first["wall"] - first["pc"],
            "peers": len(peers),
            "median": statistics.median(peers) if peers else None,
        }
        summary["excess"] = (
            None if summary["median"] is None else summary["first_net"] - summary["median"]
        )
        out.append((name, summary))
    return out


def report(path: Path) -> None:
    records, reattachment = parse(path)
    if not records:
        print(f"{path.name}: no step rows found -- is this a march log?")
        return
    rungs = excess(records)
    total = sum(s["wall"] for _, s in rungs)
    print(f"\n{path.name}: {len(records)} steps, {total} s, mid-span x_r/h {reattachment or '--'}")
    print(
        f"  {'rung':<18} {'steps':>5} {'wall':>7} {'first':>7} {'cyc':>4} "
        f"{'first-pc':>9} {'peers':>6} {'median':>7} {'excess':>7}"
    )
    removable = 0.0
    for index, (name, s) in enumerate(rungs):
        median = "--" if s["median"] is None else f"{s['median']:.0f}"
        gap = "n/a" if s["excess"] is None else f"{s['excess']:.0f}"
        print(
            f"  {name:<18} {s['steps']:>5} {s['wall']:>7} {s['first_wall']:>7} "
            f"{s['first_cycles']:>4} {s['first_net']:>9.0f} {s['peers']:>6} {median:>7} {gap:>7}"
        )
        if index > 0 and s["excess"] is not None:
            removable += s["excess"]
    print(
        f"  {'':<18} {'':>5} {'':>7} {'':>7} {'':>4} {'':>9} {'':>6} {'rungs 2+':>7} {removable:>7.0f}"
    )


def main() -> None:
    paths = [Path(arg) for arg in sys.argv[1:]]
    if not paths:
        raise SystemExit(__doc__.strip().splitlines()[-1].strip())
    for path in paths:
        report(path)


if __name__ == "__main__":
    main()
