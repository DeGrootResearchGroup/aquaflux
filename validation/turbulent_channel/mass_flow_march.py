"""A channel held at its bulk velocity, marched monolithically and solved by the segregated loop, side by side.

The law-of-the-wall channels (``cases/*.yaml``) hold the bulk velocity with a solved body force. This
harness solves one channel file two ways and reduces each to wall units, so the coupled march can be
judged against the segregated reference on the quantities the study reports (``Re_tau``, the plateau
``kappa``) and on what each solve cost:

* ``segregated``   -- the file's own ``Segregated`` solve, unchanged; one line per sweep;
* ``coupled``      -- :func:`~aquaflux.turbulence.solve_coupled` on the same problem, a single shifted
  step per outer iteration; one line per outer step;
* ``coupled-dual`` -- the same with a three-iteration dual-time loop (and so the Courant step control a
  dual-time march defaults to);
* ``coupled-ramp`` -- the dual-time march along a viscosity ramp inside the one march
  (:func:`~aquaflux.turbulence.solve_reynolds_ramp`: from 100 times the case's viscosity in 16
  geometric stations of one outer step each, both blocks scaled).

Each arm prints its configuration first, then one flushed line per step (and, for a coupled arm, the
measure's per-block residuals), then the reduced result.

Settings (environment):

* ``CHANNEL_CASE``  -- the case file under ``cases/`` (default ``re20000.yaml``);
* ``CHANNEL_ARM``   -- ``segregated``, ``coupled``, ``coupled-dual`` or ``coupled-ramp`` (default
  ``segregated``);
* ``CHANNEL_MAX_STEPS`` -- the coupled arms' outer-step cap (default 200);
* ``CHANNEL_PRECONDITIONER`` -- the coupled arms' preconditioner: ``default`` (``BlockDiagonal()``) or
  ``convection`` (``BlockDiagonal(scalar=ScalarAir(), velocity=ConvectionTwoLevel())``, the bundle
  ``tests/integration/test_coupled_mass_flow.py`` converges this channel with);
* ``CHANNEL_MEASURE`` -- the coupled arms' stopping measure: ``rowscaled`` (the solve's default) or
  ``euclidean``, each at ``rtol 1e-10, atol 1e-12``.

Run from the repository root through the case runner, one arm at a time::

    CHANNEL_ARM=coupled validation/run_case.sh validation/turbulent_channel/mass_flow_march.py --wait
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Running a script puts the script's directory on `sys.path`, not the working directory, so without
# this the case runner cannot import `aquaflux` from a plain checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aquaflux  # noqa: F401  (enables x64)
import aquaflux.case.solver as case_solver
import jax
from aquaflux.case import read_case
from aquaflux.flow import ConvectionTwoLevel
from aquaflux.solve import Convergence, DualTimeLoop, Euclidean, RowScaled
from aquaflux.turbulence import (
    BlockDiagonal,
    ScalarAir,
    held_force,
    solve_coupled,
    solve_reynolds_ramp,
)
from compare import reduce_to_wall_units

HERE = Path(__file__).resolve().parent
CASE = os.environ.get("CHANNEL_CASE", "re20000.yaml")
ARM = os.environ.get("CHANNEL_ARM", "segregated")
MAX_STEPS = int(os.environ.get("CHANNEL_MAX_STEPS", "200"))
PRECONDITIONERS = {
    "default": BlockDiagonal(),
    "convection": BlockDiagonal(scalar=ScalarAir(), velocity=ConvectionTwoLevel()),
}
PRECONDITIONER = os.environ.get("CHANNEL_PRECONDITIONER", "default")
MEASURES = {"rowscaled": RowScaled(), "euclidean": Euclidean()}
MEASURE = os.environ.get("CHANNEL_MEASURE", "rowscaled")
ARMS = ("segregated", "coupled", "coupled-dual", "coupled-ramp")
#: The ramp arm's schedule: the anchor viscosity factor, the stations and the outer steps per station.
RAMP = {"anchor": 100.0, "stations": 16, "steps_per_station": 1}


class _Clock:
    """Seconds since the solve started, for the per-step lines."""

    def __init__(self) -> None:
        self.start = time.monotonic()

    def __call__(self) -> float:
        return time.monotonic() - self.start


def _count_flow_solves(clock: _Clock) -> list[int]:
    """Count (and print) each sweep of the segregated loop, by wrapping the bordered flow solve it builds.

    The loop takes no observer, so the sweep count is read off the flow solve it calls once per sweep.
    """
    calls = [0]
    build = case_solver.bulk_velocity_flow_solve

    def counted(momentum, **settings):
        solve = build(momentum, **settings)

        def sweep(m, state):
            calls[0] += 1
            out = solve(m, state)
            print(f"sweep {calls[0]:4d}  t {clock():8.1f}s", flush=True)
            return out

        return sweep

    case_solver.bulk_velocity_flow_solve = counted
    return calls


def _printing(clock: _Clock, reports: list):
    """An ``on_step`` observer printing each outer step and keeping its report."""

    def on_step(report) -> None:
        reports.append(report)
        print(
            f"step {report.step:4d}  |R| {report.residual_norm:.3e}  ratio "
            f"{report.residual_ratio:.3e}  cycles {report.cycles:3d}  inner "
            f"{report.inner_iterations:2d}  alpha {report.alpha:.3f}  shift {report.shift:.4f}  esc "
            f"{report.escalations}  t {clock():8.1f}s",
            flush=True,
        )

    return on_step


def _printing_blocks(blocks) -> None:
    """An ``on_residuals`` observer printing the measure's per-block residuals of each step."""
    print("  blocks " + "  ".join(f"{name} {float(v):.3e}" for name, v in blocks.items()), flush=True)


def main() -> None:
    if ARM not in ARMS:
        raise SystemExit(f"CHANNEL_ARM must be one of {ARMS}, got {ARM!r}")
    case_file = read_case(HERE / "cases" / CASE)
    checked = case_file.check()
    coupled = checked.build()
    spec = case_file.spec
    print(
        f"case {CASE}  arm {ARM}  cells {coupled.momentum.mesh.n_cells}  "
        f"nu {spec.fluid.kinematic_viscosity:.6e}  jax {jax.__version__}",
        flush=True,
    )
    clock = _Clock()
    if ARM == "segregated":
        print(f"solver: the file's {spec.solver!r}", flush=True)
        calls = _count_flow_solves(clock)
        flow, _, _ = checked.solve(coupled)
        cost = f"{calls[0]} sweeps"
    else:
        dual_time = DualTimeLoop(inner_steps=3) if ARM != "coupled" else None
        convergence = Convergence(measure=MEASURES[MEASURE], rtol=1e-10, atol=1e-12)
        preconditioner = PRECONDITIONERS[PRECONDITIONER]
        print(
            f"solver: solve_coupled, {preconditioner!r}, {convergence!r}, max_steps {MAX_STEPS}, "
            f"dual_time {dual_time!r}, ramp {RAMP if ARM == 'coupled-ramp' else None}",
            flush=True,
        )
        reports: list = []
        settings = {} if dual_time is None else {"dual_time": dual_time}
        settings |= {
            "preconditioner": preconditioner,
            "convergence": convergence,
            "max_steps": MAX_STEPS,
            "on_step": _printing(clock, reports),
            "on_residuals": _printing_blocks,
        }
        if ARM == "coupled-ramp":
            flow, k, omega = solve_reynolds_ramp(
                coupled, point_setup=lambda companion, seed, point: {}, **RAMP, **settings
            )
        else:
            flow, k, omega = solve_coupled(coupled, **settings)
        cost = (
            f"{len(reports)} steps, {sum(r.cycles for r in reports)} restart cycles, "
            f"{sum(r.escalations for r in reports)} escalations, "
            f"beta {float(held_force(coupled, flow, k, omega)):.6e}"
        )
    r = reduce_to_wall_units(spec, coupled.momentum, flow, clock())
    print(
        f"RESULT case {CASE} arm {ARM}: Re_tau {r['Re_tau']:.1f}  u_tau {r['u_tau']:.5f}  "
        f"kappa {r['kappa_plateau']:.4f}  U_bulk {r['u_bulk']:.6f}  first y+ "
        f"{r['first_yplus']:.3f}  {cost}  {r['dt']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
