"""Resident memory of the bfs3d case, stage by stage, in one process (Linux).

Whether the 3D case fits a machine is a question about its peak, and a peak read end to end cannot say
which stage set it. This harness runs the case's own pieces in order and prints, after each stage, the
resident set (``VmRSS``), the process's high-water mark (``VmHWM``) and the stage's own peak, sampled
every 20 ms by a thread, so a stage whose peak sits under an earlier one is still seen:

1. the imports (``compare`` imports ``aquaflux`` and reads ``case.yaml``);
2. the mesh read, and 3. its geometry;
4. the case build (``compare.build_case``: its own mesh read and geometry again, the wall-distance
   search, both assemblers);
5. ``hybrid_initialize``;
6. the first residual, compiled and run; 7. the first Jacobian-vector product, compiled and run
   (``jacobian_matvec``, the probe's own function);

then frees those and runs the case's own march (``compare.solve_aquaflux``, the shipped solver section
of ``case.yaml``), whose stages are read from inside it without restating any of it:

8. the march's own case build and initialization, up to the session's first fit;
9. the probe plan, colouring and gather map (the session's first ``_probe_for``);
10. the first fit -- the coloured probe's responses and the block inverses' factorization;
11. each later re-fit (``BetaTrackingRefresh._materialize_at``);
12. each step, the first including the march step's compilation; stopped after ``BFS3D_MEMORY_STEPS``
    steps (default 3).

Run it through the case runner, alone::

    validation/run_case.sh validation/bfs3d_openfoam/memory_by_stage.py --wait

On Linux the resident set is the footprint (there are no compressed pages to leave out). It is not
macOS's footprint, so a figure from here and one from ``peak_footprint.py`` are not comparable.
"""

from __future__ import annotations

import gc
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

#: The steps marched before the measurement stops.
STEPS = int(os.environ.get("BFS3D_MEMORY_STEPS", "3"))


def _status_kib(field: str) -> int:
    """A ``/proc/self/status`` field, in KiB."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1])
    raise RuntimeError(f"/proc/self/status has no {field}")


def _rss_kib() -> int:
    """The resident set alone, read from ``statm`` (cheaper than ``status`` at the sampler's rate)."""
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") // 1024


class _Sampler(threading.Thread):
    """The largest resident set seen since the last :meth:`window`, sampled every 20 ms."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._peak = 0
        self._lock = threading.Lock()

    def run(self) -> None:
        while True:
            rss = _rss_kib()
            with self._lock:
                self._peak = max(self._peak, rss)
            time.sleep(0.02)

    def window(self) -> int:
        """The peak since the previous call (or start), in KiB; starts a new window."""
        with self._lock:
            peak, self._peak = max(self._peak, _rss_kib()), _rss_kib()
        return peak


SAMPLER = _Sampler()
_last = {"rss": 0, "t": time.perf_counter()}


def stage(label: str) -> None:
    """Print one row: the stage's own peak, the resident set after it and its change, the high-water mark."""
    rss, hwm, peak = _status_kib("VmRSS"), _status_kib("VmHWM"), SAMPLER.window()
    now = time.perf_counter()
    print(
        f"| {label:<52} | {peak / 2**20:6.2f} | {rss / 2**20:6.2f} | "
        f"{(rss - _last['rss']) / 2**20:+6.2f} | {hwm / 2**20:6.2f} | {now - _last['t']:7.1f} |",
        flush=True,
    )
    _last.update(rss=rss, t=now)


def _header() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    meminfo = {
        line.split(":")[0]: int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
    }
    cache = Path(
        os.environ.get("AQUAFLUX_COMPILATION_CACHE_DIR", Path.home() / ".cache/aquaflux/jax")
    )
    entries = len(list(cache.iterdir())) if cache.is_dir() else 0
    print(
        f"commit {commit}{' (uncommitted changes)' if dirty else ''}, {platform.platform()}, "
        f"{os.cpu_count()} CPUs, MemTotal {meminfo['MemTotal'] / 2**20:.1f} GiB, "
        f"MemAvailable {meminfo['MemAvailable'] / 2**20:.1f} GiB, "
        f"SwapTotal {meminfo['SwapTotal'] / 2**20:.1f} GiB",
        flush=True,
    )
    print(
        f"compilation cache {cache}: {entries} entries before the run "
        f"({'warm' if entries else 'cold'}; disabled: "
        f"{os.environ.get('AQUAFLUX_DISABLE_COMPILATION_CACHE', '0')})",
        flush=True,
    )


def _instrument_the_march(on_step_count: list[int]) -> None:
    """Wrap the session's stages so each prints a row as it ends; nothing they compute is changed."""
    from aquaflux.precondition import MaterializedSession
    from aquaflux.precondition.materialized_session import BetaTrackingRefresh

    probe_for, fit, materialize_at = (
        MaterializedSession._probe_for,
        MaterializedSession._fit,
        BetaTrackingRefresh._materialize_at,
    )
    seen = {"probe": False, "fit": 0, "refit": 0}

    def _probe_for(self):
        if self._probe is None and not seen["probe"]:
            stage("march: case build + initialization, to the first fit")
            built = probe_for(self)
            seen["probe"] = True
            stage("probe plan, colouring and gather map")
            return built
        return probe_for(self)

    def _fit(self, state, base):
        fitted = fit(self, state, base)
        seen["fit"] += 1
        stage(f"fit {seen['fit']}: probe responses + block inverses")
        return fitted

    def _materialize_at(self, step, state, beta):
        if seen["refit"] == 0:
            stage("from the first fit to the first re-fit")
        phases = materialize_at(self, step, state, beta)
        seen["refit"] += 1
        stage(f"re-fit {seen['refit']} (β {beta:.3g})")
        return phases

    MaterializedSession._probe_for = _probe_for
    MaterializedSession._fit = _fit
    BetaTrackingRefresh._materialize_at = _materialize_at


class _Stop(Exception):
    """Raised by the step observer once the measured steps are done."""


def main() -> None:
    SAMPLER.start()
    _header()
    print(
        "| stage | stage peak GiB | RSS GiB | ΔRSS GiB | HWM GiB | s |\n|---|---|---|---|---|---|",
        flush=True,
    )
    stage("start (interpreter)")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(HERE))
    import compare  # imports aquaflux (x64) and reads case.yaml
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from aquaflux.initialization import hybrid_initialize
    from aquaflux.io import read_openfoam
    from aquaflux.precondition.materialized_session import jacobian_matvec

    stage("imports")
    print(f"jax {jax.__version__}, mesh {compare.MESH.relative_to(ROOT)}", flush=True)

    mesh = read_openfoam(compare.MESH)
    stage(f"mesh read ({mesh.n_cells} cells)")
    geometry = mesh.geometry()
    jax.block_until_ready(geometry)
    stage("geometry")
    del mesh, geometry
    gc.collect()

    case = compare.build_case()
    coupled = case["coupled"]
    jax.block_until_ready(coupled)
    stage("case build (its own mesh + geometry, wall distance)")

    flow, k, omega = hybrid_initialize(coupled)
    state = coupled.state_from_physical(flow, k, omega)
    jax.block_until_ready(state)
    stage("hybrid_initialize")

    residual = eqx.filter_jit(lambda c, s: c.residual(s))(coupled, state)
    jax.block_until_ready(residual)
    stage("first residual, compiled and run")
    if not bool(jnp.all(jnp.isfinite(residual))):
        raise SystemExit("the hybrid start's residual is not finite")

    tangent = jnp.ones_like(state)
    jvp = jacobian_matvec(coupled, state, tangent)
    jax.block_until_ready(jvp)
    stage("first JVP, compiled and run")

    del case, coupled, flow, k, omega, state, residual, tangent, jvp
    gc.collect()
    jax.clear_caches()
    stage("freed (before the march)")

    steps: list[int] = []
    _instrument_the_march(steps)

    def on_step(report):
        steps.append(1)
        stage(f"step {len(steps)}" + (" (march step compiled)" if len(steps) == 1 else ""))
        if len(steps) >= STEPS:
            raise _Stop

    try:
        compare.solve_aquaflux(log_path=HERE / "memory_by_stage.march.log", on_step=on_step)
    except _Stop:
        pass
    stage("end")
    print(
        f"peak resident set (VmHWM) {_status_kib('VmHWM') / 2**20:.2f} GiB after {len(steps)} steps",
        flush=True,
    )


if __name__ == "__main__":
    main()
