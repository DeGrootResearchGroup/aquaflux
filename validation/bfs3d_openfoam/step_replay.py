"""Re-run one outer step of this case's march from its checkpoint, at chosen shifts and solve settings.

A retried step is only explained by re-running it: the march log records the attempt it kept and a
table of the ones it threw away, but not why an inner Newton iteration's direction failed. This module
rebuilds the march's own step at the state a capture checkpointed before it (``compare.py`` run with
``BFS3D_CHECKPOINT_KEEP=200``), re-fits the preconditioner there in full -- which is exactly what the
march does on the first step of a viscosity station, and on the first step at the target -- and takes
the step, printing every inner iteration as the march log does. So a step that opens a station replays
faithfully; one in the middle of a station does not (its inverse dates from an earlier re-fit), and is
refused.

A full march overwrites ``checkpoints/``: its rolling checkpointer prunes older states, so copy a
capture's checkpoints elsewhere before another march runs, or the states this module replays are lost.

The first run of a step at its recorded shift is the check: it must reproduce the attempt the log
recorded (inner iterations, cycles, step lengths). Every further arm changes one thing.

Usage, from the repository root::

    python3 -u validation/bfs3d_openfoam/step_replay.py <step> <beta> [<beta> ...]

``<step>`` is one-based, as the march log numbers it; each ``<beta>`` is run in turn. Environment, all
optional:

- ``REPLAY_RTOL``, ``REPLAY_MAX_RESTARTS``: the inner solve's tolerance and restart budget.
- ``REPLAY_REFIT_FLOOR``: the shift the inverse is re-fitted at, at least (the preconditioner's
  ``refit_beta_floor``).
- ``REPLAY_STATION``: run the step on another viscosity station's residual, to separate a station
  change from the state it meets.
- ``REPLAY_NO_ALPHA_ABORT=1``: let the inner loop run on past a collapsed step length.
- ``REPLAY_OMEGA_STEP_RATIO``: bound each cell's ``omega`` to change by at most this factor per
  correction (after the per-cell ``k`` clip every step applies); ``none`` leaves it unbounded. Unset,
  the case's own setting is used.
- ``REPLAY_PREVIOUS_REFIT=1``: first re-fit where the march re-fitted before the previous step, at that
  step's state and recorded shift, and keep the step it returns. The march builds each step's measure
  from the step as last re-fitted, so a step whose predecessor re-fitted (a station change, or a solve
  that reached the re-fit threshold) needs it.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import os
import sys
import time
from pathlib import Path

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.initialization import hybrid_initialize  # noqa: E402
from aquaflux.solve import ConstantRelaxation  # noqa: E402
from aquaflux.solve.march import _march_step  # noqa: E402
from aquaflux.turbulence import coupled_step, open_session, scale_both_blocks, scale_momentum_only  # noqa: E402
from aquaflux.turbulence.coupled import coupled_scaled_norm  # noqa: E402

CHECKPOINTS = CASE / "checkpoints"
_COMPANIONS = {"flow": scale_momentum_only, "both": scale_both_blocks, None: scale_both_blocks}


def _inner_row(inner, g_in, g_out, cycles, alpha, candidate):
    """Print one inner iteration the way the march log does."""
    print(
        f"    | {int(inner):5d} | {int(cycles):4d} | {float(g_in):10.3e} | {float(g_out):10.3e} | "
        f"{float(g_out) / float(g_in):6.3f} | {float(alpha):5.3f} |",
        flush=True,
    )


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    k = int(sys.argv[1])
    betas = [float(b) for b in sys.argv[2:]]

    solver = compare.SOLVER
    if os.environ.get("REPLAY_RTOL") or os.environ.get("REPLAY_MAX_RESTARTS"):
        linear = solver.linear_solve
        linear = dataclasses.replace(
            linear,
            **({"rtol": float(os.environ["REPLAY_RTOL"])} if os.environ.get("REPLAY_RTOL") else {}),
            **(
                {"max_restarts": int(os.environ["REPLAY_MAX_RESTARTS"])}
                if os.environ.get("REPLAY_MAX_RESTARTS")
                else {}
            ),
        )
        solver = dataclasses.replace(solver, linear_solve=linear)
    if os.environ.get("REPLAY_REFIT_FLOOR"):
        preconditioner = dataclasses.replace(
            solver.preconditioner, refit_beta_floor=float(os.environ["REPLAY_REFIT_FLOOR"])
        )
        solver = dataclasses.replace(solver, preconditioner=preconditioner)
    ramp = solver.continuation
    stations = ramp.stations * ramp.steps_per_station
    station = min((k - 1) // ramp.steps_per_station, ramp.stations)
    if (k - 1) % ramp.steps_per_station:
        raise SystemExit(f"step {k} does not open a station, so its inverse is not re-fitted there")

    coupled = compare.build_case()["coupled"]
    companion = _COMPANIONS[ramp.scale]

    def assembler_for(s):
        return coupled if s == ramp.stations else companion(coupled, ramp.anchor ** (1.0 - s / ramp.stations))

    anchor = assembler_for(0)
    seed = anchor.state_from_physical(*hybrid_initialize(anchor))
    allowed = inspect.signature(coupled_step).parameters
    settings = solver.settings()
    march = {k_: v for k_, v in settings.items() if k_ in allowed and k_ != "preconditioner"}
    if os.environ.get("REPLAY_OMEGA_STEP_RATIO"):
        ratio = os.environ["REPLAY_OMEGA_STEP_RATIO"]
        march["omega_step_ratio"] = math.inf if ratio == "none" else float(ratio)
    session = open_session(settings["preconditioner"], coupled)
    session.rebind(anchor)
    base = session.build(seed, **march)
    print(
        f"[guards] per-cell k clip, omega step ratio {march.get('omega_step_ratio') or 'default'}",
        flush=True,
    )
    policy = base.shift_policy
    retry = solver.retry
    if os.environ.get("REPLAY_NO_ALPHA_ABORT"):
        retry = dataclasses.replace(retry, on_alpha=None)
    base = retry.with_inner_abort(base)
    base = dataclasses.replace(base, inner_observer=_inner_row)

    if os.environ.get("REPLAY_PREVIOUS_REFIT") and k > 1:
        with np.load(CHECKPOINTS / f"state-{k - 2:05d}.npz") as data:
            phi_previous = jnp.asarray(data["state"])
        with np.load(CHECKPOINTS / f"state-{k - 1:05d}.npz") as data:
            beta_previous = float(data["shift"])
        session.rebind(assembler_for(min((k - 2) // ramp.steps_per_station, ramp.stations)))
        previous = eqx.tree_at(lambda s: s.relaxation_schedule, base, ConstantRelaxation(beta_previous))
        update = session.refresh_preconditioner(previous, phi_previous, due=True)
        if update is not None:
            base = update(base)
        policy = base.shift_policy
        print(f"[previous re-fit] at state-{k - 2:05d}, beta {beta_previous:g}", flush=True)
    with np.load(CHECKPOINTS / f"state-{k - 1:05d}.npz") as data:
        phi_n = jnp.asarray(data["state"])
    if os.environ.get("REPLAY_STATION"):
        station = int(os.environ["REPLAY_STATION"])
    assembler = assembler_for(station)
    reference = float(coupled_scaled_norm(coupled, policy, seed)(anchor.residual(seed)))
    measure = coupled_scaled_norm(coupled, policy, phi_n)
    print(
        f"[configuration] step {k} (station {station} of {ramp.stations}, target from {stations + 1}); "
        f"state-{k - 1:05d}; linear {solver.linear_solve}; reference |R0| {reference:.4e}",
        flush=True,
    )

    for beta in betas:
        session.rebind(assembler)
        active = eqx.tree_at(lambda s: s.relaxation_schedule, base, ConstantRelaxation(beta))
        update = session.refresh_preconditioner(active, phi_n, due=True)
        if update is not None:
            active = update(active)
        active = active.with_norm(measure)
        started = time.perf_counter()
        print(f"\n  beta {beta:g}", flush=True)
        print("    | inner |  cyc |       G in |      G out |   rate | alpha |", flush=True)
        outcome, residual_norm = _march_step(
            active,
            assembler.residual,
            phi_n,
            assembler.residual(phi_n),
            jnp.asarray(reference),
            active.linear_solver(),
        )
        print(
            f"  -> |R| {float(residual_norm):.3e}, min alpha {float(outcome.alpha):.3f}, "
            f"{int(outcome.inner_iterations)} inner, {int(outcome.raw_cycles)} raw cycles, "
            f"{time.perf_counter() - started:.0f} s",
            flush=True,
        )
        k_slice, w_slice = coupled.layout.slice_of("k"), coupled.layout.slice_of("omega")
        moved = outcome.phi - phi_n
        print(
            f"  -> min k {float(outcome.phi[k_slice].min()):.3e}, max |d log omega| "
            f"{float(jnp.abs(moved[w_slice]).max()):.3g}, max |dk| {float(jnp.abs(moved[k_slice]).max()):.3g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
