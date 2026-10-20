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
``beta_min``); ``REFIT_STATE`` is a ``.npz`` holding ``state``; ``REFIT_PROBE_BATCH`` re-fits at that probe batch instead of
the library's ``PROBE_BATCH_SIZE``; ``REFIT_JVP_BATCHES`` (say ``1,4,8,16``)
also times the probe's batched Jacobian-vector product alone at each batch size, best of five, per
tangent, which separates the cost of one directional derivative from the number of probes. That sweep
times the whole-state product; where the case's problem is probed one column field at a time (both step
cases are) the re-fits run the per-field programs instead, which ``fusion_profile.py``'s
``probe-fields`` profiles.

The report also splits the probe's host work out of it (filling each chunk's seeds and scattering its
responses into the matrix), sums the processor time per thread name (which pool kept the cores busy),
and prints the process's resident high-water mark after the warm re-fit and after the timed ones: an
allocator setting that retains freed memory shows there as well as in the time.
"""

from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from step_cases import load_case  # noqa: E402

CASE = os.environ.get("REFIT_CASE", "bfs3d")
REPEATS = int(os.environ.get("REFIT_REPEATS", "3"))
BETA = float(os.environ.get("REFIT_BETA", "0.005"))
STATE = os.environ.get("REFIT_STATE")
#: The probe's batch for the re-fits themselves; unset keeps the library's ``PROBE_BATCH_SIZE``.
PROBE_BATCH = os.environ.get("REFIT_PROBE_BATCH")
#: Batch sizes at which to time the probe's batched JVP alone, per tangent; empty skips it.
JVP_BATCHES = [int(b) for b in os.environ.get("REFIT_JVP_BATCHES", "").split(",") if b]


def _thread_cpu() -> dict[str, float]:
    """Processor seconds used so far by this process's threads, summed per thread name (Linux only).

    The thread name says which pool a thread serves (``tf_foreach`` runs the compiled program's thunks,
    ``tf_XLAEigen`` its intra-op work, the Python-named rest the interpreter, BLAS and the host
    refactor), so a difference of two readings says which pool a piece kept busy. Empty where
    ``/proc/self/task`` does not exist.
    """
    tick = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    out: dict[str, float] = {}
    try:
        tasks = os.listdir("/proc/self/task")
    except OSError:
        return out
    for task in tasks:
        try:
            with open(f"/proc/self/task/{task}/stat") as handle:
                stat = handle.read()
        except OSError:
            continue  # a thread that exited between the listing and the read
        name = stat[stat.index("(") + 1 : stat.rindex(")")]
        fields = stat[stat.rindex(")") + 2 :].split()
        # utime and stime are fields 14 and 15 of the stat line, so 11 and 12 after the name.
        name = name.rstrip("0123456789-")
        out[name] = out.get(name, 0.0) + (int(fields[11]) + int(fields[12])) / tick
    return out


def _high_water_mark() -> str:
    """This process's resident high-water mark and current resident size, from ``/proc/self/status``."""
    try:
        with open("/proc/self/status") as handle:
            lines = dict(line.split(":", 1) for line in handle if ":" in line)
    except OSError:
        return "unavailable"
    return (
        f"VmHWM {int(lines['VmHWM'].split()[0]) / 2**20:.2f} GiB, "
        f"VmRSS {int(lines['VmRSS'].split()[0]) / 2**20:.2f} GiB"
    )


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
    case = load_case(CASE)
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
    from aquaflux.precondition import materialized_session

    if PROBE_BATCH:
        # Read by the session at each fit and re-fit, so setting it before the first build reaches both.
        materialized_session.PROBE_BATCH_SIZE = int(PROBE_BATCH)
    print(f"probe batch {materialized_session.PROBE_BATCH_SIZE}", flush=True)
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

    probe = session._probe_for()
    pieces = _Pieces()
    pieces.wrap(host, "_materialize_jacobian", "probe (coloured JVPs + gather)")
    pieces.wrap(probe.plan, "seed_block", "probe: fill the seeds (host)")
    pieces.wrap(
        probe.plan, "field_seeds", "probe: fill the seeds (host)"
    )  # the per-field programs' seeds
    pieces.wrap(probe.structure, "scatter", "probe: scatter the responses (host)")
    pieces.wrap(host, "_shifted", "assemble the shift")
    pieces.wrap(split._groups, "blocks", "refactor: slice the blocks")
    pieces.wrap(split._leading, "refactor_block", "refactor: leading [u,v,w,p]")
    pieces.wrap(split._trailing, "refactor_block", "refactor: trailing [k,omega]")
    pieces.wrap(split, "_set_coupling", "refactor: retained coupling")

    if JVP_BATCHES:
        _jvp_sweep(session, state)
    hook._materialize_at(step, state, BETA)  # warm: any compile a re-fit triggers is paid here
    print(f"after the warm re-fit: {_high_water_mark()}", flush=True)
    pieces.rows.clear()
    totals = [0.0, 0.0, 0.0]
    threads = _thread_cpu()
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
    used = _thread_cpu()
    by_pool = {name: used[name] - threads.get(name, 0.0) for name in used}
    print(
        "processor seconds per re-fit by thread name: "
        + ", ".join(
            f"{name} {seconds / REPEATS:.1f}"
            for name, seconds in sorted(by_pool.items(), key=lambda kv: -kv[1])
            if seconds > 0
        )
    )
    print(f"after the timed re-fits: {_high_water_mark()}", flush=True)


if __name__ == "__main__":
    main()
