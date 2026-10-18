"""The march's compiled step program, timed call by call, with its page faults and Krylov work.

``krylov_overhead.py`` decomposes one Krylov iteration on replayed systems; this reads the figure that
decomposition has to account for, from the march itself. It runs ``compare.py``'s march unchanged, with
the compiled step (``aquaflux.solve.march._march_step``) wrapped by a timer that blocks on its result,
and prints one line per call: wall, the compilation seconds JAX reported during it (a call that compiled
or loaded a program from the persistent cache is not a measurement of the program), the operator
applications the step's solves made, and the minor page faults the call took. The summary divides the
compile-free calls' wall by their applications: an upper bound on one application inside the step, since
the step also runs the line search, the inner-loop bookkeeping and its per-step residuals.

The applications are counted only under ``PITZ_KRYLOV_STOP=residual`` (``compare.py``'s counting
instrument, whose step table is the shipped march's); without it the column reads 0.

Usage
-----
    PITZ_KRYLOV_STOP=residual validation/run_case.sh validation/pitzdaily_openfoam/step_program_timing.py --wait
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import compare  # noqa: E402  (imports aquaflux, which enables x64)
import jax  # noqa: E402
from aquaflux.solve import march  # noqa: E402


class _CompileSeconds:
    """The seconds JAX's monitoring reports for compilation events, accumulated."""

    seconds = 0.0

    def __call__(self, event, duration, **_):
        if "compil" in event or "cache" in event:
            self.seconds += duration


_COMPILE = _CompileSeconds()
jax.monitoring.register_event_duration_secs_listener(_COMPILE)
_STEP = march._march_step
_CALLS = []


def _timed_step(*args, **kwargs):
    applications = compare.KRYLOV_WORK.applications
    faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    compiled = _COMPILE.seconds
    started = time.perf_counter()
    result = jax.block_until_ready(_STEP(*args, **kwargs))
    wall = time.perf_counter() - started
    record = (
        wall,
        _COMPILE.seconds - compiled,
        compare.KRYLOV_WORK.applications - applications,
        resource.getrusage(resource.RUSAGE_SELF).ru_minflt - faults,
    )
    _CALLS.append(record)
    per = f"{wall * 1e3 / record[2]:.1f} ms/app" if record[2] else "-"
    print(
        f"[step program] call {len(_CALLS)}: {wall:.2f} s, compile {record[1]:.2f} s, "
        f"{record[2]} applications ({per}), {record[3]} minor faults",
        flush=True,
    )
    return result


def _summary():
    clean = [c for c in _CALLS if c[1] < 1e-3 and c[2] > 0]
    wall = sum(c[0] for c in clean)
    applications = sum(c[2] for c in clean)
    faults = sum(c[3] for c in clean)
    print(
        f"[step program] {len(_CALLS)} calls, {len(clean)} without compilation: {wall:.1f} s over "
        f"{applications} applications = {wall * 1e3 / max(applications, 1):.1f} ms per application, "
        f"{faults / max(applications, 1):.0f} minor faults per application; all calls "
        f"{sum(c[0] for c in _CALLS):.1f} s, {sum(c[2] for c in _CALLS)} applications",
        flush=True,
    )


if __name__ == "__main__":
    march._march_step = _timed_step
    try:
        compare.main()
    finally:
        _summary()
