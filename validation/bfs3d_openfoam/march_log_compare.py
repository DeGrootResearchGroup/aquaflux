"""Read archived march logs back as data, and compare two arms rung by rung.

Every question this case asks about solver cost -- why one preconditioner arm takes nine more steps
than another, where in the ladder the extra steps land, what the controller did in response to a
clipped step -- is a question about the per-step record, and that record exists only in the logs. They
are written for a human to read down; this reads them back into rows so the same run can be summed,
grouped by continuation rung, and set beside another run.

The parse is deliberately anchored on the **summary stats** grid rather than on the inner-iteration
block, because the summary row is the one line per step that is always emitted, carries the
controller's shift and the accepted step length, and states the flags. The inner block is optional
detail and its width varies with the configuration.

Flags, as the logger writes them:

``L``
    The step length was set by an injected positivity limit rather than by the descent test.
``eN``
    The step was redone ``N`` times with the shift escalated before it was accepted.
``R``
    The step was redone by the tight-Krylov divergence retry.

Usage
-----
Summarize one log, or compare two::

    python3 validation/bfs3d_openfoam/march_log_compare.py march-A.log
    python3 validation/bfs3d_openfoam/march_log_compare.py march-A.log march-B.log

The comparison prints a per-rung table and then the first step at which the two runs diverge, which
is the question that matters when two arms are meant to differ in one setting only: a pair that
separates at step one differs in the operator itself, while a pair that tracks for a whole rung and
then splits differs in how the controller reacted to something.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

#: One row of the ``summary stats`` grid: ``| step | t(s) | beta | in | cyc | R | a_min | flg |``.
#: The flag cell is optional in the sense of being blank, never absent, so it is matched as a
#: possibly-empty run of non-``|`` characters and stripped.
_STEP_ROW = re.compile(
    r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.eE+-]+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
    r"\s*([\d.eE+-]+)\s*\|\s*([\d.eE+-]+)\s*\|\s*([^|]*)\|\s*$"
)
#: The continuation-rung banner, e.g. ``[point 3/3 (target Re)]``. The label is kept verbatim; it
#: names the physical rung ("Re/100", "target Re") which is what a reader wants beside the counts.
_RUNG = re.compile(r"^\[point (\d+)/(\d+) \(([^)]*)\)\]")
#: The preconditioner-cost aside. It comes in three forms -- ``pc full`` (the hierarchy was rebuilt),
#: ``pc inner`` (an existing hierarchy was re-fitted mid-step), and ``pc none`` (nothing was rebuilt) --
#: and all three carry the seconds this step spent in the preconditioner. Matching only ``pc full``
#: undercounts the total by an order of magnitude, because the mid-step re-fits are the bulk of it.
_PC = re.compile(r"\|\s*pc (full|inner|none) ([\d.]+)s")
#: The breakdown inside a rebuilding aside, e.g. ``(probe 11.5 assemble 0.2 refactor 1.9)``. Absent on
#: a ``pc none`` step, and the ``other`` term is present only when the build did something unusual.
_PC_PARTS = re.compile(r"probe ([\d.]+) assemble ([\d.]+) refactor ([\d.]+)")


@dataclass
class Step:
    """One accepted march step, as the summary grid recorded it."""

    step: int
    elapsed: float
    beta: float
    inners: int
    cycles: int
    residual: float
    alpha: float
    flags: str
    rung: int
    rung_label: str

    @property
    def escalations(self) -> int:
        """How many shift escalations this step was redone with, from the ``eN`` flag."""
        found = re.search(r"e(\d+)", self.flags)
        return int(found.group(1)) if found else 0

    @property
    def limited(self) -> bool:
        """Whether an injected positivity limit, not the descent test, set the step length."""
        return "L" in self.flags

    @property
    def retried(self) -> bool:
        """Whether the tight-Krylov divergence retry redid this step."""
        return "R" in self.flags


@dataclass
class Run:
    """One march log, parsed."""

    path: Path
    steps: list[Step]
    config: dict[str, str]
    pc_seconds: float
    probe_seconds: float

    @property
    def wall(self) -> float:
        """Elapsed seconds at the last recorded step."""
        return self.steps[-1].elapsed if self.steps else 0.0

    @property
    def cycles(self) -> int:
        """Total Krylov restart cycles over the whole march."""
        return sum(s.cycles for s in self.steps)

    def rungs(self) -> dict[int, list[Step]]:
        """The steps grouped by continuation rung, in rung order."""
        grouped: dict[int, list[Step]] = {}
        for step in self.steps:
            grouped.setdefault(step.rung, []).append(step)
        return grouped


def parse(path: Path) -> Run:
    """Read one march log into a :class:`Run`.

    Parameters
    ----------
    path : Path
        The log to read.

    Returns
    -------
    Run
        The parsed steps, the configuration banner as a mapping, and the summed preconditioner cost.
    """
    steps: list[Step] = []
    config: dict[str, str] = {}
    pc_seconds = 0.0
    probe_seconds = 0.0
    rung, rung_label = 0, ""
    in_banner = False

    for line in path.read_text().splitlines():
        if line.startswith("[configuration]"):
            in_banner = True
            continue
        if in_banner:
            if line.startswith("["):
                in_banner = False
            elif ":" in line:
                key, _, value = line.strip().partition(":")
                config[key.strip()] = value.strip()

        banner = _RUNG.match(line)
        if banner:
            rung, rung_label = int(banner.group(1)), banner.group(3)
            continue

        cost = _PC.search(line)
        if cost:
            pc_seconds += float(cost.group(2))
            parts = _PC_PARTS.search(line)
            if parts:
                probe_seconds += float(parts.group(1))
            continue

        row = _STEP_ROW.match(line)
        # The heading row matches the shape of a data row in column count but not in type; requiring
        # the first cell to be an integer is what separates them, and the regex already does that.
        if row:
            steps.append(
                Step(
                    step=int(row.group(1)),
                    elapsed=float(row.group(2)),
                    beta=float(row.group(3)),
                    inners=int(row.group(4)),
                    cycles=int(row.group(5)),
                    residual=float(row.group(6)),
                    alpha=float(row.group(7)),
                    flags=row.group(8).strip(),
                    rung=rung,
                    rung_label=rung_label,
                )
            )
    return Run(
        path=path,
        steps=steps,
        config=config,
        pc_seconds=pc_seconds,
        probe_seconds=probe_seconds,
    )


def summarize(run: Run) -> None:
    """Print one run's per-rung counts and its whole-march totals."""
    print(f"\n=== {run.path.name} ===")
    for key in (
        "turbulence inverse",
        "k wall BC",
        "k positivity floor",
        "native trailing settings",
    ):
        if key in run.config:
            print(f"  {key}: {run.config[key]}")
    print(
        f"  {'rung':<12} {'steps':>6} {'cycles':>7} {'wall(s)':>8} "
        f"{'clips':>6} {'esc':>4} {'lim':>4} {'beta_end':>10}"
    )
    previous_elapsed = 0.0
    for _rung, group in sorted(run.rungs().items()):
        wall = group[-1].elapsed - previous_elapsed
        previous_elapsed = group[-1].elapsed
        print(
            f"  {group[0].rung_label:<12} {len(group):>6} {sum(s.cycles for s in group):>7} "
            f"{wall:>8.0f} {sum(1 for s in group if s.alpha < 1.0):>6} "
            f"{sum(s.escalations for s in group):>4} "
            f"{sum(1 for s in group if s.limited):>4} {group[-1].beta:>10.4f}"
        )
    print(
        f"  {'TOTAL':<12} {len(run.steps):>6} {run.cycles:>7} {run.wall:>8.0f} "
        f"{sum(1 for s in run.steps if s.alpha < 1.0):>6} "
        f"{sum(s.escalations for s in run.steps):>4} "
        f"{sum(1 for s in run.steps if s.limited):>4}"
    )
    print(
        f"  preconditioner: {run.pc_seconds:.0f}s "
        f"({100 * run.pc_seconds / max(run.wall, 1):.0f}% of wall), "
        f"of which the coloured Jacobian probe is {run.probe_seconds:.0f}s"
    )


def diverged_at(a: Run, b: Run, tolerance: float = 1e-3) -> int | None:
    """The first step index at which two runs' residuals differ by more than ``tolerance`` relative.

    Returns ``None`` when the shorter run is a prefix of the longer to within the tolerance, which is
    the signature of two arms that took the *same* path at different cost.
    """
    for left, right in zip(a.steps, b.steps, strict=False):
        if abs(left.residual - right.residual) > tolerance * abs(right.residual):
            return left.step
    return None


def compare(a: Run, b: Run) -> None:
    """Print both runs' summaries, then where and how they part company."""
    summarize(a)
    summarize(b)
    split = diverged_at(a, b)
    print(f"\n=== {a.path.name} vs {b.path.name} ===")
    if split is None:
        print("  the two runs follow the same trajectory for as far as both go")
    else:
        print(f"  trajectories part at step {split}")
        print(
            f"  {'step':>5} {'beta_A':>9} {'beta_B':>9} {'R_A':>11} {'R_B':>11} {'aA':>6} {'aB':>6}"
        )
        for left, right in list(zip(a.steps, b.steps, strict=False))[max(0, split - 3) : split + 6]:
            print(
                f"  {left.step:>5} {left.beta:>9.4f} {right.beta:>9.4f} "
                f"{left.residual:>11.3e} {right.residual:>11.3e} "
                f"{left.alpha:>6.3f} {right.alpha:>6.3f}"
            )


def main() -> None:
    paths = [Path(arg) for arg in sys.argv[1:]]
    if not paths:
        raise SystemExit(__doc__)
    if len(paths) == 1:
        summarize(parse(paths[0]))
    else:
        compare(parse(paths[0]), parse(paths[1]))


if __name__ == "__main__":
    main()
