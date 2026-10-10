"""Does seeding pitzDaily's second-order march from a first-order solution make the whole solve cheaper?

Defect-correction seeding: march the residual with first-order upwind momentum advection -- a smoother,
more diagonally dominant problem -- to a loose tolerance, and start the shipped second-order march from
that state instead of from the viscosity ramp. Whether it pays is a whole-solve question, so this runs
three arms through ``aquaflux run``, each in its own process, from case files derived from the shipped
``case.yaml`` (so they cannot drift from it):

``first_order``   the shipped case with ``momentum_advection: FirstOrderUpwind`` and the stop loosened to
                  ``atol = PITZ_SEED_ATOL`` (default 1e-3), keeping the ramp, writing checkpoints.
``from_seed``     the shipped case with no ramp, starting from ``first_order``'s last checkpoint. Its
                  problem differs from the seed's, so it opens as a new march.
``baseline``      the shipped case.

Each run's own record gives the wall time, steps and convergence; JAX's compile log, kept beside it,
gives the compilation (``compile_timeline.py``), since a first-order program is compiled cold on its
first run where the second-order ones may already be cached. The comparison is ``first_order +
from_seed`` against ``baseline``, both with and without compilation. Both second-order arms stop at the
shipped tolerance on the same problem, so they reach the same root.

Usage
-----
    validation/run_case.sh validation/pitzdaily_openfoam/seed_study.py

``PITZ_SEED_WORK`` (default ``seed_study`` beside this file, gitignored) holds the case files and
results; ``PITZ_SEED_ARMS`` picks arms (default ``first_order,from_seed,baseline``).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from compile_timeline import timeline  # noqa: E402

WORK = Path(os.environ.get("PITZ_SEED_WORK", HERE / "seed_study")).resolve()
SEED_ATOL = os.environ.get("PITZ_SEED_ATOL", "1.0e-03")
ARMS = tuple(os.environ.get("PITZ_SEED_ARMS", "first_order,from_seed,baseline").split(","))
_SHIPPED_ATOL = "atol: 1.0e-05"
_SHIPPED_ADVECTION = re.compile(
    r"  momentum_advection:\n    kind: LimitedUpwind\n    limiter: .*\n"
)
_SHIPPED_RAMP = re.compile(r"  continuation: \{kind: ViscosityRamp.*\}\n")


def _replace_once(text, old, new):
    """``text`` with ``old`` (a string or compiled pattern) replaced exactly once, or refuse."""
    pattern = old if isinstance(old, re.Pattern) else re.compile(re.escape(old))
    result, count = pattern.subn(new, text)
    if count != 1:
        raise SystemExit(f"case.yaml no longer holds {pattern.pattern!r} exactly once ({count})")
    return result


def _case_text(arm):
    """The arm's case file, as edits of the shipped ``case.yaml`` text (a YAML 1.1 round trip would
    misread ``1.0e-05`` as a string)."""
    text = (HERE / "case.yaml").read_text()
    text = _replace_once(text, "path: runs/kwsst/polyMesh", f"path: {HERE / 'runs/kwsst/polyMesh'}")
    if arm == "first_order":
        text = _replace_once(
            text, _SHIPPED_ADVECTION, "  momentum_advection: {kind: FirstOrderUpwind}\n"
        )
        text = _replace_once(text, _SHIPPED_ATOL, f"atol: {SEED_ATOL}")
    if arm == "from_seed":
        text = _replace_once(text, _SHIPPED_RAMP, "")
        text += f"\ninitial: {{kind: Checkpoint, path: {WORK / 'first_order' / 'results' / 'checkpoints'}}}\n"
    text += (
        "\noutputs:\n  directory: results\n  log: march.log\n  history: history.csv\n"
        "  checkpoints: {kind: Checkpoints, every: 1, keep: 2}\n  fields: []\n"
    )
    return text


def _run(arm):
    directory = WORK / arm
    directory.mkdir(parents=True, exist_ok=True)
    case = directory / "case.yaml"
    case.write_text(_case_text(arm))
    log = directory / "stdout.log"
    environment = dict(os.environ, JAX_LOG_COMPILES="1")
    print(f"[{arm}] running {case}", flush=True)
    with log.open("w") as out:
        status = subprocess.run(
            [sys.executable, "-m", "aquaflux", "run", "--overwrite", str(case)],
            stdout=out,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        ).returncode
    record = yaml.safe_load((directory / "results" / "run.yaml").read_text())
    _, seconds, _, _ = timeline(log.read_text().splitlines())
    compiled = sum(seconds.values())
    print(
        f"[{arm}] exit {status}, converged {record['converged']}, {record['steps']} steps, "
        f"{record['seconds']} s, of which {compiled:.0f} s compiling; residual {record['residual']}",
        flush=True,
    )
    return record["seconds"], compiled


def main():
    print(f"[configuration] work {WORK}; seed atol {SEED_ATOL}; arms {ARMS}", flush=True)
    results = {arm: _run(arm) for arm in ARMS}
    if {"first_order", "from_seed", "baseline"} <= results.keys():
        seeded = [a + b for a, b in zip(results["first_order"], results["from_seed"], strict=True)]
        base = results["baseline"]
        print(
            f"\nseeded {seeded[0]:.0f} s ({seeded[0] - seeded[1]:.0f} s without compiling) against "
            f"baseline {base[0]:.0f} s ({base[0] - base[1]:.0f} s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
