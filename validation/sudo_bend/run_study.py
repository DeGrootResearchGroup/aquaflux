"""Run the bend's grid study end to end: each grid's meshes, its precursor and its bend, then the GCI.

For each grid (``grids/round36``, ``round48``, ``round64``, written by ``write_grids.py``), in order:

1. ``aquaflux mesh mesh.yaml`` and ``aquaflux mesh precursor_mesh.yaml``;
2. ``aquaflux run precursor.yaml`` -- the developed inflow, ``work/precursor/profiles.csv``;
3. ``aquaflux run case.yaml`` -- the bend, ``work/results``;

then ``gci.py`` once all three bends have converged. A step whose output is already there is
skipped -- a mesh that exists, a run whose ``run.yaml`` says it converged -- so the study resumes
where it stopped; ``SUDO_REDO=1`` redoes every step. Each step is its own process (so a grid's
memory is returned before the next grid starts), stopped with this one by ``kill <pid>``; its output
goes to the grid's ``work/<step>.log``, and one line per step is printed here.

Run it through the case runner, which logs it, keeps the machine awake and refuses to start beside
another case::

    validation/run_case.sh validation/sudo_bend/run_study.py           # returns; tail -f the log
    validation/run_case.sh --status                                    # what is running
    SUDO_GRIDS=round48 validation/run_case.sh validation/sudo_bend/run_study.py --wait

Settings (environment):

* ``SUDO_GRIDS`` -- the grids, comma-separated (default all three, coarse to fine);
* ``SUDO_REDO`` -- ``1`` to redo steps whose output is already there (default ``0``);
* ``SUDO_FORCE`` -- ``1`` to start a bend whose memory estimate exceeds what is available
  (default ``0``);
* ``SUDO_GCI`` -- ``0`` to leave the GCI to someone else, as a batch job per grid does (default
  ``1``: run it once all three bends have converged).

On a cluster, ``slurm/submit.sh`` submits one job per grid and a GCI job after them.

The bend's march holds its whole materialized Jacobian: about 10.5 GB on ``round36``, and an
estimated 20 GB on ``round48`` and 45 GB on ``round64`` (about 200 KB a cell). Before each bend the
script prints that estimate beside the memory available to it (a batch job's or a container's
limit where one applies), and refuses a bend whose estimate
exceeds it -- one swapping or killed half-way costs more than not starting -- unless ``SUDO_FORCE=1``
(the estimate is scaled from the coarse grid, not measured on the others).
"""

from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # validation/, for peak_footprint

from peak_footprint import run_forwarded  # noqa: E402

GRIDS = HERE / "grids"
DEFAULT_GRIDS = ("round36", "round48", "round64")
#: Bytes a bend cell holds at the march's peak, measured on round36 (about 10.5 GB on 42,660 cells
#: with the interpreter and caches), and the bend's cells on each grid (README, "The grid study").
BYTES_PER_CELL = 2.0e5
CELLS = {"round36": 42660, "round48": 98987, "round64": 221102}


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _available_gib() -> float:
    """The memory this process may still take: the machine's available memory, or less where a
    control group (a Slurm job's ``--mem``, a container's limit) caps the process's group.

    The cap is the tightest limit from the process's own group up to the root, less what that
    group already uses; cgroup v2 (``memory.max``) and v1 (``memory.limit_in_bytes``) alike.
    """
    with open("/proc/meminfo") as meminfo:
        available = next(
            int(line.split()[1]) * 1024 for line in meminfo if line.startswith("MemAvailable:")
        )
    for limit, usage in _cgroup_limits():
        available = min(available, max(limit - usage, 0))
    return available / 2**30


def _cgroup_limits() -> list[tuple[int, int]]:
    """``(limit, usage)`` in bytes of each memory control group holding this process, up to the root."""
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return []
    found = []
    for line in lines:
        _, controllers, path = line.split(":", 2)
        if controllers == "":  # cgroup v2
            root, limit_file, usage_file = Path("/sys/fs/cgroup"), "memory.max", "memory.current"
        elif "memory" in controllers.split(","):  # cgroup v1
            root = Path("/sys/fs/cgroup/memory")
            limit_file, usage_file = "memory.limit_in_bytes", "memory.usage_in_bytes"
        else:
            continue
        group = root / path.lstrip("/")
        while True:
            try:
                limit = (group / limit_file).read_text().strip()
                usage = int((group / usage_file).read_text())
            except (OSError, ValueError):
                limit = None
            if limit not in (None, "max") and int(limit) < 2**60:
                found.append((int(limit), usage))
            if group == root or root not in group.parents:
                break
            group = group.parent
    return found


def _converged(run_yaml: Path) -> bool:
    """Whether a run's record says it converged (read as text: the record is flat YAML)."""
    return run_yaml.exists() and "converged: true" in run_yaml.read_text()


def _step(grid: Path, name: str, arguments: list[str]) -> bool:
    """Run ``python -m aquaflux <arguments>`` for ``grid``, its output to its ``work/<name>.log``."""
    log = grid / "work" / f"{name}.log"
    log.parent.mkdir(exist_ok=True)
    _say(f"{grid.name}: {name} -- aquaflux {' '.join(arguments)} (log {log})")
    start = time.perf_counter()
    saved = os.dup(1), os.dup(2)
    with open(log, "w") as out:
        # The child inherits this process's stdout and stderr: point both at the log for its run.
        os.dup2(out.fileno(), 1)
        os.dup2(out.fileno(), 2)
        try:
            status = run_forwarded([sys.executable, "-u", "-m", "aquaflux", *arguments]).returncode
        finally:
            for fd, copy in zip((1, 2), saved, strict=True):
                os.dup2(copy, fd)
                os.close(copy)
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
    _say(
        f"{grid.name}: {name} exit {status} after {time.perf_counter() - start:.0f} s "
        f"(largest step so far {peak:.1f} GiB resident)"
    )
    return status == 0


def run_grid(name: str, redo: bool, force: bool) -> bool:
    """Every step of one grid; whether its bend converged."""
    grid = GRIDS / name
    work = grid / "work"
    for mesh_file, aqmesh in (("mesh.yaml", "bend"), ("precursor_mesh.yaml", "precursor")):
        if redo or not (work / f"{aqmesh}.aqmesh").exists():
            step = mesh_file.removesuffix(".yaml")
            if not _step(grid, step, ["mesh", str(grid / mesh_file), "--overwrite"]):
                return False
        else:
            _say(f"{name}: {aqmesh}.aqmesh there, not remeshed")
    for case, record in (("precursor.yaml", "precursor"), ("case.yaml", "results")):
        if not redo and _converged(work / record / "run.yaml"):
            _say(f"{name}: {case} converged already ({work / record / 'run.yaml'})")
            continue
        if case == "case.yaml":
            need, have = BYTES_PER_CELL * CELLS[name] / 2**30, _available_gib()
            _say(f"{name}: the bend needs about {need:.0f} GiB; {have:.1f} GiB available")
            if need > have and not force:
                _say(f"{name}: not started -- too little memory (SUDO_FORCE=1 starts it anyway)")
                return False
        # --overwrite: a run that stopped short left its output directory, which a run refuses.
        if not _step(grid, case.removesuffix(".yaml"), ["run", str(grid / case), "--overwrite"]):
            return False
        if not _converged(work / record / "run.yaml"):
            _say(f"{name}: {case} stopped without converging; see {work / record}")
            return False
    return True


def main() -> int:
    grids = [g.strip() for g in os.environ.get("SUDO_GRIDS", ",".join(DEFAULT_GRIDS)).split(",")]
    unknown = [g for g in grids if g not in CELLS]
    if unknown:
        _say(f"unknown grids {unknown}; the grids are {list(CELLS)}")
        return 2
    redo = os.environ.get("SUDO_REDO", "0") == "1"
    force = os.environ.get("SUDO_FORCE", "0") == "1"
    _say(
        f"grids {grids}; redo {redo}, force {force}; {len(os.sched_getaffinity(0))} cores, "
        f"{_available_gib():.1f} GiB available"
    )
    done = {grid: run_grid(grid, redo, force) for grid in grids}
    _say("bends converged: " + ", ".join(f"{g} {'yes' if ok else 'NO'}" for g, ok in done.items()))
    gci = os.environ.get("SUDO_GCI", "1") == "1"
    if gci and all(_converged(GRIDS / g / "work" / "results" / "run.yaml") for g in DEFAULT_GRIDS):
        _say("all three bends converged: the GCI")
        return run_forwarded([sys.executable, "-u", str(HERE / "gci.py")]).returncode
    return 0 if all(done.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
