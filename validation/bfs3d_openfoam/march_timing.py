"""Time the full continuation march, without the OpenFOAM comparison.

``compare.main`` refuses to run without the transient reference fields, because its job is to *validate*
against them. A timing question does not need them: what is wanted is the wall clock and the per-step
record of the march itself, which ``solve_aquaflux`` produces on its own. Keeping this as its own entry
point means the timing question can be re-asked later without either generating a reference or editing
the comparison.

Reads the same environment the comparison does, so an arm is selected the same way::

    BFS3D_FLOW_INVERSE=simplesmooth BFS3D_REFRESH_ON_CYCLES=8 \\
        validation/run_case.sh validation/bfs3d_openfoam/march_timing.py

``BFS3D_FLOW_INVERSE`` selects the leading (flow saddle) block's inverse. **Raise
``BFS3D_REFRESH_ON_CYCLES`` alongside a slower-converging one**: the refresh fires when a solve *reaches*
the threshold, and the shipped 3 is calibrated to an incomplete-LU that runs two cycles per solve, so a
preconditioner that healthily takes six or seven trips it on essentially every step and the march then
measures the trigger rather than the preconditioner.
"""

from __future__ import annotations

import time
from pathlib import Path

import compare

HERE = Path(__file__).parent


def main() -> None:
    if not (compare.RUNS / "polyMesh").exists():
        raise SystemExit(f"mesh not found in {compare.RUNS}; run of_case/run_of.sh first.")
    print(
        f"flow-block inverse: {compare.FLOW_INVERSE}  |  turbulence: {compare.TURBULENCE_INVERSE}  |  "
        f"field split: {compare.FIELD_SPLIT}  |  refresh at {compare.REFRESH_ON_CYCLES} cycles",
        flush=True,
    )
    # The host arm's own settings, every one of them, because two runs that differ only in a sweep
    # count or in whether the coarsening is frozen would otherwise produce identical headers -- and a
    # timing comparison between them is then unattributable after the fact.
    if compare.FLOW_INVERSE == "simplesmooth":
        print(f"native flow block: {compare._SIMPLE_FLOW}", flush=True)
    log_path = compare._fresh_log(HERE / "march_timing.log")
    started = time.time()
    result = compare.solve_aquaflux(log_path=log_path, checkpoint_dir=HERE / "checkpoints")
    elapsed = time.time() - started
    # The velocity range is not a validation check -- it is a cheap tell that the march produced a
    # physical field rather than converging to something degenerate, which a wall-clock number alone
    # would not reveal.
    print(
        f"march complete in {elapsed:.0f}s  |  Ux in "
        f"[{result['U'][:, 0].min():.3f}, {result['U'][:, 0].max():.3f}]",
        flush=True,
    )
    print(f"per-step record: {log_path}", flush=True)


if __name__ == "__main__":
    main()
