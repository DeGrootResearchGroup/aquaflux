"""Replay the march's stall bailout over every march log on disk, and count what it would do.

The bailout ends a segment once a constraint-bound step stops changing anything
(``forward_march(stop_on_limit_stall=...)``). Its danger is not missing a lock-up -- those are
unmistakable once seen -- but ending a rung that would have recovered, and **a march that recovers looks
exactly like one that does not, right up until it does.** So a candidate rule cannot be judged on the
run that motivated it. It has to be replayed over the healthy marches too.

That is what this does: parse every ``march*.log`` in this directory into per-rung step histories, run
the real predicate over each, and report the longest stalling run and the step the bailout would fire
at. A rule is acceptable when it fires on the locked-up rungs and on nothing that went on to converge.

Two candidate rules were rejected here, each having looked right on the failing run alone:

* "the residual did not fall" -- never fires at all, because a locked-up step still moves the state by
  ``alpha`` times the correction and so does reduce the residual, by ~1e-6 relative.
* "the residual did not fall by 0.1%" -- fires on a **converging** rung whose residual was climbing
  through a pseudo-transient excursion (caps 0.983, 0.928, 0.253; residual 1.293e-01 -> 1.489e-01, then
  9.241e-02 and on to 4.994e-06).

Reads the logs rather than re-running anything, so it costs seconds and can be re-asked whenever the
rule or its threshold changes.

Usage::

    python3 -u validation/bfs3d_openfoam/march_stall_replay.py
    python3 -u validation/bfs3d_openfoam/march_stall_replay.py 5   # a different stall count
"""

from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))

from aquaflux.solve import StepReport  # noqa: E402
from aquaflux.solve.march import _limit_collapsing  # noqa: E402

#: One summary row: step, wall, beta, inners, cycles, residual, a_min, flags.
_ROW = re.compile(
    r"\|\s*(\d+) \|\s*(\d+) \| ([\d.]+) \|\s*(\d+) \|\s*(\d+) \| ([\d.e+-]+) \| ([\d.]+) \|"
)


def rung_histories(text: str) -> list[list[StepReport]]:
    """Every continuation rung in one march log, as the step histories the march itself reported.

    Split on the summary-block delimiter and parse **within** each block. Parsing the whole log with one
    pattern per field does not work: the ``limit`` aside is written only where the cap was the binding
    constraint, so there are fewer of those than of step rows (105 against 108 on one log here), and
    zipping two independently-collected lists shifts the caps against the steps they belong to. The
    shifted table is just as smooth as the real one, so nothing looks wrong.

    Parameters
    ----------
    text : str
        The full text of a march log.

    Returns
    -------
    list of list of StepReport
        One history per rung, in order. Only the fields this replay reads are filled.
    """
    histories = []
    for rung in re.split(r"\[point \d/\d[^\]]*\]", text)[1:]:
        history = []
        for block in rung.split("+= summary stats")[1:]:
            row = _ROW.search(block)
            if row is None:
                continue
            cap = re.search(r"\| limit ([\d.e+-]+)", block)
            history.append(
                StepReport(
                    step=int(row.group(1)),
                    cycles=int(row.group(5)),
                    residual_norm=float(row.group(6)),
                    residual_ratio=0.0,
                    alpha=float(row.group(7)),
                    # No aside means the cap was not the binding constraint, which the march reports as 1.
                    binding_limit=float(cap.group(1)) if cap else 1.0,
                )
            )
        if history:
            histories.append(history)
    return histories


def replay(history: list[StepReport], stall_limit: int) -> tuple[int, int | None]:
    """The longest run of stalling steps in a history, and the step the bailout would fire at.

    Parameters
    ----------
    history : list of StepReport
        One rung's steps, in order.
    stall_limit : int
        Consecutive stalling steps that end the segment.

    Returns
    -------
    tuple(int, int or None)
        ``(longest_run, fired_at)``; ``fired_at`` is ``None`` if the rung would run to completion.
    """
    run, longest, fired = 0, 0, None
    for previous, report in itertools.pairwise(history):
        run = run + 1 if _limit_collapsing(previous, report) else 0
        longest = max(longest, run)
        if fired is None and run >= stall_limit:
            fired = report.step
    return longest, fired


def main() -> None:
    stall_limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    logs = sorted(CASE.glob("march*.log"))
    if not logs:
        raise SystemExit(f"no march*.log in {CASE}")
    print(f"{'=' * 100}\nstall bailout replay, stop_on_limit_stall = {stall_limit}\n{'=' * 100}")
    for path in logs:
        text = path.read_text()
        for number, history in enumerate(rung_histories(text), 1):
            longest, fired = replay(history, stall_limit)
            print(
                f"  {path.name:<30} rung {number}  steps {len(history):>4}  "
                f"final |R| {history[-1].residual_norm:.3e}  longest stall {longest:>3}  "
                f"{'FIRES at step ' + str(fired) if fired is not None else 'runs to completion'}",
                flush=True,
            )


if __name__ == "__main__":
    main()
