"""pitzDaily backward-facing step by Reynolds-number continuation with a dual-time pseudo-timestep ramp.

The companion to :mod:`compare`: the **same** benchmark (mesh, boundary conditions, model constants and
scheme choices come from :func:`compare.build_case`, imported here rather than restated), solved by a
different march strategy and judged the same way -- reattachment length ``x_r/h`` against the
time-accurate OpenFOAM reference (``x_r/h`` ~ 7.74). Where :mod:`compare` drives one coupled solve at the
target viscosity, this case walks a **homotopy in Reynolds number** and marches each rung in dual time:

* **Reynolds-number continuation** (:func:`aquaflux.turbulence.solve_reynolds_continuation`): a sequence
  of lower-Reynolds solves from an easy anchor up to the true target, each seeded by the previous
  converged field. With the default geometric schedule (one decade per rung) and ``n_points = 2`` the
  ramp is **Re = target/100, target/10, target** -- i.e. the molecular viscosity is scaled by
  ``(100, 10, 1)``, giving companion Reynolds numbers of roughly 250 and 2500 before the true ~25000.
  Raising the viscosity weakens the convective nonlinearity, so the anchor develops the recirculation
  cheaply and each rung is a small jump from a converged neighbour; the continuation dissolves at the
  target, so the root and its adjoint are the target problem's, unchanged.

* **Dual-time (backward-Euler) march** (``inner_steps > 1`` + :class:`aquaflux.solve.DualTimeControl`):
  each outer pseudo-timestep holds a reference field and runs an inner Newton loop on the transient
  residual ``R + (1/dtau)(phi - phi_ref)``, so the pseudo-time term sits in the residual (not only the
  Jacobian) and the measured steady residual is the honest discrete time-derivative. The inner loop keeps
  a **large pseudo-timestep** (small pseudo-transient shift beta) stable, and a Courant-style control
  ramps the timestep up while the inner loop stays comfortable -- the lever on how many steps it takes to
  develop the bubble.

**The conservative Courant control is load-bearing (binding -- do not restore the defaults).** The shipped
:class:`~aquaflux.solve.DualTimeControl` defaults (``grow_above = 0.5``, ``backoff_below = 0.25``,
``grow = 1.5``) grow the pseudo-timestep even when an inner step has *clipped* (line-search factor 0.5),
which on the stiff target-Reynolds rung drives the timestep past the point where the low-shift coupled
solve loses diagonal dominance and the step goes non-finite. This case therefore grows the timestep
**only on a fully comfortable inner step** (``grow_above = 1.0``), backs off the moment one clips
(``backoff_below = 0.5``), and grows gently (``grow = 1.3``) -- which develops the recirculation
vigorously and stably. It still **stalls short of the OpenFOAM reattachment** on the target rung, as the
pseudo-time step meets the low-shift conditioning wall of the coupled saddle at this Reynolds number; that stall is
the phenomenon this case exists to reproduce and study, not a failure to run.

Per-step progress (reattachment length, the control's pseudo-timestep shift, restart-cycle cost, residual)
is streamed as the march runs, because it is a long, deliberately non-terminating study.

Run (after ``of_case/run_of.sh``, from the repo root)::

    python3 -u validation/pitzdaily_openfoam/compare_reynolds_continuation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Reuse the benchmark definition and the reattachment metric from the base case rather than restate them.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aquaflux  # noqa: F401  (enables x64 at import)
import compare
import numpy as np
from aquaflux.solve import CoefficientDriftTrigger, DualTimeControl
from aquaflux.turbulence import GeometricReynoldsSchedule, solve_reynolds_continuation

# The Reynolds ramp: 2 lower-Re rungs before the target -> viscosity scales (100, 10, 1) -> Re ~ 250,
# 2500, 25000. The rest of the march budget is the working configuration for this case.
N_POINTS = 2
MAX_STEPS = 80  # per rung and per refreshed segment
REFRESH_LIMIT = 20  # preconditioner re-freezes allowed per rung (drift-triggered)
INNER_STEPS = 5  # dual-time inner Newton iterations per outer pseudo-timestep
INNER_TOL = 0.05  # inner loop stops at this fraction of the anchor residual
INTERMEDIATE_RTOL = 1e-2  # lower-Re rungs are only seeds -> converge them loosely
RTOL = 1e-6  # target-rung tolerance
DRIFT_THRESHOLD = 0.1  # eddy-viscosity drift that re-freezes the frozen preconditioner
RESTART = 120  # forward-solve GMRES restart; matvecs ~= restart-cycles x RESTART

# Conservative Courant control (see the module docstring -- the shipped defaults diverge on the target
# rung): grow the pseudo-timestep only on a fully comfortable inner step, back off on any clip.
CONTROL = DualTimeControl(
    beta_start=0.5,
    grow=1.3,
    backoff=2.0,
    grow_above=1.0,
    backoff_below=0.5,
    beta_min=0.05,
    beta_max=4.0,
)


class _ShiftLoggingControl:
    """Wrap a step control to record the pseudo-transient shift (beta) it selects each outer step.

    A :class:`~aquaflux.solve.StepReport` carries the inner line-search factor, not the shift the control
    chose, so the streamed log cannot otherwise show the pseudo-timestep ramping. This delegates to the
    wrapped control and stashes the shift it returns (which is the control's carried state), so the
    observer can print it. It resets to the wrapped control's ``beta_start`` at each new Reynolds rung on
    its own, because the continuation restarts the control state per rung.

    Parameters
    ----------
    inner : StepControl
        The control to delegate to (here a :class:`~aquaflux.solve.DualTimeControl`).
    """

    def __init__(self, inner: DualTimeControl) -> None:
        self.inner = inner
        self.beta_start = inner.beta_start
        self.last_beta = float(inner.beta_start)

    def next_step(
        self, base_step: object, previous: object, state: object
    ) -> tuple[object, object]:
        """Delegate to the wrapped control and record the selected shift; signature per ``StepControl``."""
        step, beta = self.inner.next_step(base_step, previous, state)
        self.last_beta = float(beta)
        return step, beta


def solve_aquaflux_continuation(**solve_kwargs: object) -> dict:
    """Solve the pitzDaily case by Reynolds continuation + dual time; stream progress; return the fields.

    Parameters
    ----------
    **solve_kwargs
        Overrides forwarded to :func:`aquaflux.turbulence.solve_reynolds_continuation` (and thence to each
        per-Reynolds :func:`~aquaflux.turbulence.solve_coupled`), on top of the defaults set here.

    Returns
    -------
    dict
        ``centroid`` ``(n_cells, 2)``, ``U`` ``(n_cells, 2)``, and ``p``, ``k``, ``omega``, ``nut`` each
        ``(n_cells,)``, in the mesh's own cell order -- the same shape :func:`compare.solve_aquaflux`
        returns, so the two cases compare cell-for-cell.
    """
    case = compare.build_case()
    coupled, momentum, geom = case["coupled"], case["momentum"], case["geom"]
    centroid = np.asarray(geom.cell.centroid)

    scales = GeometricReynoldsSchedule().scales(N_POINTS)
    print(
        f"[cfg] Reynolds ramp (viscosity scales) = {scales}  ->  Re ~ "
        f"{', '.join(f'{1.0 / s:g}x' for s in scales)} target; "
        f"inner_steps={INNER_STEPS} beta_start={CONTROL.beta_start} beta_min={CONTROL.beta_min} "
        f"rtol={RTOL} drift={DRIFT_THRESHOLD}",
        flush=True,
    )

    control = _ShiftLoggingControl(CONTROL)
    t0 = time.time()
    counters = {"obs": 0, "cum_cycles": 0}

    def on_checkpoint(report, state):
        counters["obs"] += 1
        counters["cum_cycles"] += int(report.cycles)
        flow, _k, _omega = coupled.physical_fields(state)
        velocity, _p = momentum.unpack(flow)
        xr = compare.reattachment_length(centroid, np.asarray(velocity)[:, 0])
        print(
            f"  t={time.time() - t0:6.0f}s step={counters['obs']:4d} beta={control.last_beta:6.3f} "
            f"cyc={report.cycles:3d} cumcyc={counters['cum_cycles']:5d} "
            f"matvecs~{counters['cum_cycles'] * RESTART:7d} |R|={report.residual_norm:.4e} "
            f"ratio={report.residual_ratio:.3e} alpha_inner={report.alpha:.3f} "
            f"drift={report.drift:.3f} x_r/h={xr:.3f}",
            flush=True,
        )

    options = (
        dict(
            intermediate_rtol=INTERMEDIATE_RTOL,
            method="twolevel",
            max_steps=MAX_STEPS,
            rtol=RTOL,
            inner_steps=INNER_STEPS,
            inner_tol=INNER_TOL,
            step_control=control,
            refresh_trigger=CoefficientDriftTrigger(threshold=DRIFT_THRESHOLD),
            refresh_limit=REFRESH_LIMIT,
            on_checkpoint=on_checkpoint,
        )
        | solve_kwargs
    )

    print("[run] starting Reynolds-continuation dual-time march...", flush=True)
    flow, k, omega = solve_reynolds_continuation(coupled, N_POINTS, **options)
    velocity, pressure = momentum.unpack(flow)
    nu_t = coupled.turbulence.closure_fields(momentum.velocity_fields(flow), k, omega).nu_t
    print(
        f"[done] wall={time.time() - t0:.0f}s obs_steps={counters['obs']} "
        f"cum_cycles={counters['cum_cycles']} matvecs~{counters['cum_cycles'] * RESTART}",
        flush=True,
    )
    return dict(
        centroid=centroid,
        U=np.asarray(velocity),
        p=np.asarray(pressure),
        k=np.asarray(k),
        omega=np.asarray(omega),
        nut=np.asarray(nu_t),
    )


def main():
    if not (compare.RUNS / "U").exists():
        raise SystemExit(
            f"OpenFOAM results not found in {compare.RUNS}; run of_case/run_of.sh first."
        )
    of = compare.read_openfoam_reference()
    xr_of = compare.reattachment_length(of["centroid"], of["U"][:, 0])
    print(f"[ref] OpenFOAM transient x_r/h = {xr_of:.3f} (target)", flush=True)

    aq = solve_aquaflux_continuation()

    xr_aq = compare.reattachment_length(aq["centroid"], aq["U"][:, 0])
    print(
        f"[result] reattachment x_r/h: aquaflux {xr_aq:.3f} vs OpenFOAM {xr_of:.3f}; "
        f"peak nu_t/nu: aquaflux {aq['nut'].max() / compare.NU:.0f} vs OpenFOAM "
        f"{of['nut'].max() / compare.NU:.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
