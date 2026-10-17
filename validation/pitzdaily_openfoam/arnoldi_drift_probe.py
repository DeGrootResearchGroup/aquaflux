"""How far does the restarted GMRES's Arnoldi residual drift from the true ``b - A M t``? Replayed.

``residual_stop_gmres`` tests its stop on the residual the Arnoldi relation gives,
``r = V_{j+1} (beta e_1 - H_j y_j)``, which costs no application of the operator. Whether the next cycle
may also START from that vector, and whether a final verdict may read it, depends on how far it departs
from the true residual as the basis loses orthogonality. This probe measures that departure on the
march's own systems, in the march's own measure.

**The replay** is :class:`replay.MarchReplay` (see its docstring for the capture it needs).

**The solve** is restarted GMRES(``RESTART``), right-preconditioned, orthogonalized by classical
Gram--Schmidt applied twice and started at each restart from the RECURRENCE residual, as the library
does after any cycle that falls short (it reads the true residual only for its verdict, which this probe
forms at every stop anyway). Each system is solved toward the
tightest target, at most ``MAX_CYCLES`` cycles, and the true residual is formed (uncounted) at:

* every cycle end, where the next cycle starts from the recurrence vector; and
* the first iteration at which the recurrence meets each target ``tau * measure(b)``, which is where a
  solve run to that target would stop and give its verdict.

Both are compared in the step's own row-scaled measure. ``departure`` is
``measure(r_true - r_recurrence) / measure(r_true)``; ``verdict`` is ``measure(r_true) / (tau
measure(b))`` at the stopping iteration, so a verdict above one is a solve the recurrence declared
converged that is not. A solve run to a looser target follows the same iterates up to its own stop, so
one run per system answers every target.

Usage
-----
    PITZ_CHECKPOINT_KEEP=500 PITZ_INNER_DUMP_ABOVE=1 \\
        validation/run_case.sh validation/pitzdaily_openfoam/compare.py
    validation/run_case.sh validation/pitzdaily_openfoam/arnoldi_drift_probe.py

``PITZ_REPLAY_FROM`` / ``PITZ_REPLAY_TO`` bound the replayed steps (default: the target station to the
end); ``PITZ_DRIFT_TARGETS`` the targets (default ``0.3,1e-3,1e-6,1e-10``); ``PITZ_DRIFT_MAX_CYCLES``
the cycle cap (default 20).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from replay import MarchReplay  # noqa: E402

TARGETS = tuple(
    float(t) for t in os.environ.get("PITZ_DRIFT_TARGETS", "0.3,1e-3,1e-6,1e-10").split(",")
)
RESTART = 15
MAX_CYCLES = int(os.environ.get("PITZ_DRIFT_MAX_CYCLES", "20"))


def drift(apply, b, measure):
    """Restarted GMRES from the recurrence residual; the true residual sampled where it matters.

    Returns ``(cycle_ends, stops, applications)``: ``cycle_ends`` a list of ``(departure, level)`` at
    each cycle end (``level`` the true residual over ``measure(b)``), ``stops`` a mapping from each
    target reached to ``(departure, verdict, applications)`` at the iteration that first met it.
    """
    reference = measure(b)
    tightest = min(TARGETS) * reference
    t = np.zeros_like(b)
    r = b.copy()
    cycle_ends, stops, applications = [], {}, 0
    for _ in range(MAX_CYCLES):
        beta = np.linalg.norm(r)
        V = np.zeros((b.size, RESTART + 1))
        V[:, 0] = r / beta
        H = np.zeros((RESTART + 1, RESTART))
        steps, finished = RESTART, False
        for j in range(RESTART):
            w = apply(V[:, j])
            applications += 1
            for _ in range(2):  # classical Gram--Schmidt, twice
                h = V[:, : j + 1].T @ w
                w -= V[:, : j + 1] @ h
                H[: j + 1, j] += h
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] > 0:
                V[:, j + 1] = w / H[j + 1, j]
            e = np.zeros(j + 2)
            e[0] = beta
            y = np.linalg.lstsq(H[: j + 2, : j + 1], e, rcond=None)[0]
            residual = V[:, : j + 2] @ (e - H[: j + 2, : j + 1] @ y)
            level = measure(residual)
            for tau in TARGETS:
                if tau not in stops and level <= tau * reference:
                    true = b - apply(t + V[:, : j + 1] @ y)
                    stops[tau] = (
                        measure(true - residual) / measure(true),
                        measure(true) / (tau * reference),
                        applications,
                    )
            if level <= tightest or H[j + 1, j] <= np.finfo(float).eps * beta:
                steps, finished = j + 1, True
                break
        t += V[:, :steps] @ y
        true = b - apply(t)
        cycle_ends.append((measure(true - residual) / measure(true), measure(true) / reference))
        r = residual
        if finished:
            break
    return cycle_ends, stops, applications


def main():
    replay = MarchReplay()
    print(
        f"{replay.describe()}; probe restart {RESTART}, cycle cap {MAX_CYCLES}, targets {TARGETS}",
        flush=True,
    )

    @eqx.filter_jit
    def measured(measure, r):
        return measure(r)

    worst_cycle = {tau: 0.0 for tau in TARGETS}
    worst_stop = {tau: (0.0, 0.0) for tau in TARGETS}
    reached = dict.fromkeys(TARGETS, 0)
    solves = 0
    header = " ".join(f"{f'{tau:g}':>19}" for tau in TARGETS)
    print(f"\n{'step':>4} {'in':>2} {'cyc':>3} {'worst cycle-end dep':>19} | {header}", flush=True)
    for system in replay.systems():
        measure = system.measure

        def apply(v, system=system):
            return np.array(replay.preconditioned(system, jnp.asarray(v)))

        def norm(v, measure=measure):
            return float(measured(measure, jnp.asarray(v)))

        cycle_ends, stops, _ = drift(apply, np.asarray(system.b), norm)
        solves += 1
        # A cycle end counts against a target only while that target is still unmet there: past it,
        # a solve run to the target would have stopped.
        for tau in TARGETS:
            live = [d for d, level in cycle_ends if level > tau]
            worst_cycle[tau] = max([worst_cycle[tau], *live])
            if tau in stops:
                reached[tau] += 1
                departure, verdict, _ = stops[tau]
                worst_stop[tau] = (
                    max(worst_stop[tau][0], departure),
                    max(worst_stop[tau][1], verdict),
                )
        cells = " ".join(
            f"{stops[tau][0]:9.1e} {stops[tau][1]:9.2e}" if tau in stops else f"{'unmet':>19}"
            for tau in TARGETS
        )
        print(
            f"{system.step:4d} {system.inner:2d} {len(cycle_ends):3d} "
            f"{max(d for d, _ in cycle_ends):19.2e} | {cells}",
            flush=True,
        )
    print(f"\n{solves} systems; per target (departure, verdict at the stop):", flush=True)
    for tau in TARGETS:
        departure, verdict = worst_stop[tau]
        print(
            f"  target {tau:g}: reached by {reached[tau]}; worst departure at a cycle end still above "
            f"it {worst_cycle[tau]:.2e}, at its stop {departure:.2e}; worst verdict {verdict:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"[done in {time.perf_counter() - t0:.0f} s]")
