"""Where a step case's preconditioner re-fit spends its time, at one fixed state, piece by piece.

On a march with a one-step-per-station ramp the re-fit runs before nearly every step, so its cost is
most of the march's. ``RefreshTiming`` splits it three ways (probe, assemble, refactor); this splits the
refactor further, into the field split's own pieces, and reports each piece's wall time, the processor
time it used (so the cores it kept busy: CPU seconds over wall seconds) and the minor page faults it took
(on Linux an allocation over glibc's mmap threshold is faulted in afresh on every call, which shows as
system time). Run it on both step cases on one machine to compare the two cases there, and set
``REFIT_STATE`` to a march checkpoint for a state that is not the hybrid start.

Nothing is restated: the case builds its own problem and session, the re-fit is the session's own
``BetaTrackingRefresh._materialize_at``, and the pieces are timed by wrapping, on the one fitted object,
the methods ``refresh_in_place`` already calls.

Usage::

    validation/run_case.sh validation/refit_profile.py --wait          # REFIT_CASE=bfs3d (default)
    REFIT_CASE=pitzdaily validation/run_case.sh validation/refit_profile.py --wait

``REFIT_CASE`` is ``bfs3d`` or ``pitzdaily``; ``REFIT_REPEATS`` re-fits are timed after one untimed warm
re-fit (default 3); ``REFIT_BETA`` is the shift strength re-fitted at (default 0.005, both cases'
``beta_min``); ``REFIT_STATE`` is a ``.npz`` holding ``state``; ``REFIT_JVP_BATCHES`` (say ``1,4,8,16``)
also times the probe's batched Jacobian-vector product alone at each batch size, best of five, per
tangent, which separates the cost of one directional derivative from the number of probes.
"""

from __future__ import annotations

import importlib.util
import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CASES = {"bfs3d": "bfs3d_openfoam", "pitzdaily": "pitzdaily_openfoam"}

CASE = os.environ.get("REFIT_CASE", "bfs3d")
REPEATS = int(os.environ.get("REFIT_REPEATS", "3"))
BETA = float(os.environ.get("REFIT_BETA", "0.005"))
STATE = os.environ.get("REFIT_STATE")
#: Batch sizes at which to time the probe's batched JVP alone, per tangent; empty skips it.
JVP_BATCHES = [int(b) for b in os.environ.get("REFIT_JVP_BATCHES", "").split(",") if b]


def _clock() -> tuple[float, float, int]:
    """Wall seconds, this process's processor seconds (user + system), and its minor faults."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return time.perf_counter(), usage.ru_utime + usage.ru_stime, usage.ru_minflt


class _Pieces:
    """Accumulated wall, processor time and faults per named piece, over one re-fit."""

    def __init__(self) -> None:
        self.rows: dict[str, list[float]] = {}

    def wrap(self, owner: object, method: str, label: str) -> None:
        """Time every call of ``owner.method`` under ``label``, on this one object only."""
        inner = getattr(owner, method)

        def timed(*args, **kwargs):
            wall, cpu, faults = _clock()
            out = inner(*args, **kwargs)
            wall2, cpu2, faults2 = _clock()
            row = self.rows.setdefault(label, [0.0, 0.0, 0.0])
            row[0] += wall2 - wall
            row[1] += cpu2 - cpu
            row[2] += faults2 - faults
            return out

        # ``object.__setattr__`` also reaches a frozen dataclass; the wrapper lives on this instance only.
        object.__setattr__(owner, method, timed)


def _case_module():
    directory = HERE / CASES[CASE]
    for path in (ROOT, directory):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(f"{CASE}_case", directory / "compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jvp_sweep(session, state) -> None:
    """The batched JVP the probe runs, timed alone per batch size: best of five, per tangent."""
    import jax
    import jax.numpy as jnp
    from aquaflux.precondition.materialized_session import batched_jacobian_matvec

    probe = session._probe_for()
    probed = probe.narrow(session.problem.assembler)
    frozen = jax.lax.stop_gradient(state)
    print(
        "\n| JVP batch | ms per call | ms per tangent | cores | faults per call |\n|---|---|---|---|---|"
    )
    for batch in JVP_BATCHES:
        # Any seed of the probe's shape costs the same to differentiate along; this one is sparse like a colour.
        block = jnp.zeros((batch, state.shape[0])).at[:, ::7].set(1.0)
        jax.block_until_ready(batched_jacobian_matvec(probed, frozen, block))  # compile
        best = None
        for _ in range(5):
            wall, cpu, faults = _clock()
            jax.block_until_ready(batched_jacobian_matvec(probed, frozen, block))
            wall2, cpu2, faults2 = _clock()
            sample = (wall2 - wall, (cpu2 - cpu) / (wall2 - wall), faults2 - faults)
            best = sample if best is None or sample[0] < best[0] else best
        print(
            f"| {batch} | {best[0] * 1e3:.1f} | {best[0] * 1e3 / batch:.1f} | {best[1]:.2f} | "
            f"{best[2] / 1e3:.0f}k |",
            flush=True,
        )


def main() -> None:
    case = _case_module()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from aquaflux.initialization import hybrid_initialize
    from aquaflux.turbulence import open_session

    coupled = case.build_case()["coupled"]
    if STATE:
        with np.load(STATE) as data:
            state = jnp.asarray(data["state"])
    else:
        state = coupled.state_from_physical(*hybrid_initialize(coupled))
    print(
        f"[configuration] case {CASE}, state "
        f"{STATE or 'the hybrid start'}, beta {BETA:g}, jax {jax.__version__}, "
        f"{os.cpu_count()} cpus, allocator settings "
        + (
            " ".join(
                f"{name}={os.environ[name]}"
                for name in sorted(os.environ)
                if name.startswith("MALLOC_") or name == "LD_PRELOAD"
            )
            or "none (glibc defaults)"
        ),
        flush=True,
    )
    session = open_session(case.SOLVER.settings()["preconditioner"], coupled)
    wall, cpu, faults = _clock()
    step = session.build(state, dual_time=case.SOLVER.dual_time)
    wall2, cpu2, faults2 = _clock()
    print(
        f"first build (plan, gather map, fit, compiles): {wall2 - wall:.1f} s wall, "
        f"{(cpu2 - cpu) / (wall2 - wall):.2f} cores, {(faults2 - faults) / 1e6:.2f}M faults",
        flush=True,
    )
    hook = session._refresh_hook()
    host = session.preconditioner
    split = host.inverse
    plan = session._probe_for().plan
    print(f"{plan.n_cells} cells, {plan.n_fields} fields, {plan.n_probes} probes", flush=True)

    pieces = _Pieces()
    pieces.wrap(host, "_materialize_jacobian", "probe (coloured JVPs + gather)")
    pieces.wrap(host, "_shifted", "assemble the shift")
    pieces.wrap(split._groups, "blocks", "refactor: slice the blocks")
    pieces.wrap(split._leading, "refactor_block", "refactor: leading [u,v,w,p]")
    pieces.wrap(split._trailing, "refactor_block", "refactor: trailing [k,omega]")
    pieces.wrap(split, "_set_coupling", "refactor: retained coupling")

    if JVP_BATCHES:
        _jvp_sweep(session, state)
    hook._materialize_at(step, state, BETA)  # warm: any compile a re-fit triggers is paid here
    pieces.rows.clear()
    totals = [0.0, 0.0, 0.0]
    for _ in range(REPEATS):
        wall, cpu, faults = _clock()
        hook._materialize_at(step, state, BETA)
        jax.block_until_ready(state)
        wall2, cpu2, faults2 = _clock()
        for i, change in enumerate((wall2 - wall, cpu2 - cpu, faults2 - faults)):
            totals[i] += change

    print(
        f"\n| piece (mean of {REPEATS} re-fits) | wall s | cores | minor faults |\n|---|---|---|---|"
    )
    for label, (w, c, f) in pieces.rows.items():
        print(
            f"| {label} | {w / REPEATS:.2f} | {c / max(w, 1e-9):.2f} | {f / REPEATS / 1e3:.0f}k |"
        )
    w, c, f = totals
    print(
        f"| **whole re-fit** | {w / REPEATS:.2f} | {c / w:.2f} | {f / REPEATS / 1e3:.0f}k |",
        flush=True,
    )


if __name__ == "__main__":
    main()
