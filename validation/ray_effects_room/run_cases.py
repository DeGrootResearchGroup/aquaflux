"""Run the room's case files, one after another in one process, as ``aquaflux run`` would.

Each file in ``cases/`` is a whole configuration -- the room, the lamp, what is gathered and where it is
written -- so this script holds no setting: it only lets ``validation/run_case.sh`` run several of them
in sequence, unbuffered, with the machine held awake, which a bare ``aquaflux run`` on the command line
would not. Each run writes into the directory its file names (``work/cases/<name>/``), and its
``run.yaml`` records what ran, when, on which commit and where the lamp's power went.

Environment: ``RAY_CASES`` (default every file, the floor cases first), names without ``.yaml``;
``RAY_OVERWRITE=1`` replaces an earlier run's results instead of refusing.

    RAY_CASES="bunny_floor empty_floor" validation/run_case.sh validation/ray_effects_room/run_cases.py --wait
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from aquaflux.case import prepare_run  # noqa: E402

ORDER = (
    "bunny_floor",
    "empty_floor",
    "bunny_volume",
    "empty_volume",
    "bunny_reflecting",
    "bunny_reflecting_coarse",
)


def main() -> int:
    names = os.environ.get("RAY_CASES", " ".join(ORDER)).split()
    overwrite = os.environ.get("RAY_OVERWRITE") == "1"
    failed = []
    for name in names:
        started = time.perf_counter()
        record = prepare_run(HERE / "cases" / f"{name}.yaml", overwrite=overwrite).run()
        print(
            f"[{time.strftime('%H:%M:%S')}] {name}: "
            f"{'done' if record.converged else 'NOT CONVERGED'} in "
            f"{time.perf_counter() - started:.1f} s",
            flush=True,
        )
        if not record.converged:
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
