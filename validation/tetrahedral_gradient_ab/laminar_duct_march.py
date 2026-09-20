"""The laminar tetrahedral duct, marched four ways: is the failure the flow-only path's, or the case's?

Issue #448 found that a hand-rolled ``newton_march`` over ``momentum_continuation`` did not converge a
resolved, near-quadratic laminar duct (Re_Dh 50) on the tetrahedral mesh of this directory, even with
corrected Green-Gauss (the scheme that marches the turbulent case): ``|R|`` rose from 1.7e-4 to ~1.3e-3
and did not fall below 8.9e-4 in 14 steps, and the reported shift stayed at 0.0000. The open question
was whether that is the flow-only path's *configuration* or a genuine gap, because the flow-only path
offered none of the robustness machinery the turbulent march runs on. It does now (``solve_flow_march``),
so the same case can be marched by the same machinery, and this script does that:

* ``bare``   -- ``newton_march`` over ``momentum_continuation``, the configuration the issue reported;
* ``staged`` -- ``solve_flow_march``, single shifted step, row-scaled measure, default globalization;
* ``lu``     -- ``solve_flow_march`` with a dual-time loop and a complete-LU ``MaterializedJacobian``;
* ``simple`` -- the same with ``SimpleSmoothed`` (the traced saddle hierarchy, no PETSc).

**Operating point.** The duct of ``make_mesh.py`` (0.25 x 0.025 x 0.025 m, 2462 tetrahedra), unit density,
``mu`` = 5e-4, inlet speed 1 m/s, so ``Re_Dh = 1 * 0.025 / 5e-4 = 50``; ``FirstOrderUpwind`` advection.
Started, by default, from a uniform plug at the inlet speed with zero pressure (``LAM_START=rest``
starts from zero velocity instead). ``potential_flow`` is not used: its scalar Laplace solve is
reported to stagnate on this mesh (see ``compare.py``), and it has not been tried for this laminar
case.

**Read a row as one measurement.** Every arm is one run from one start, and the arms measure the
residual in different norms (``bare`` in the plain 2-norm, the staged ones in the row-scaled measure
they steer by), so the final-residual columns are not comparable across those two groups; the step
count and whether the arm converged are.

Run:
    validation/run_case.sh validation/tetrahedral_gradient_ab/laminar_duct_march.py --wait

Settings (environment):

* ``LAM_START`` = ``plug`` (default) or ``rest`` (zero velocity and pressure);
* ``LAM_MEASURE`` = ``rowscaled`` (default) or ``euclid``. Use ``euclid`` from rest: the row-scaled
  measure divides by the mean speed, which is zero there, and reports NaN at step 0;
* ``LAM_SCHEME`` = ``corrected`` (default, ``CorrectedGreenGauss``) or ``multiple``
  (``MultipleCorrectionGradient`` with the corner-cell fallback);
* ``LAM_ARMS`` = a comma list of ``bare,staged,lu,simple``;
* ``LAM_MAX_STEPS`` (default 60).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Running a script puts the SCRIPT's directory on `sys.path`, not the working directory, so
# `import aquaflux` needs the repo root added explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax.numpy as jnp
from aquaflux.boundary import BoundaryConditions
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import (
    MomentumContinuity,
    NoSlipWall,
    PressureOutlet,
    VelocityInlet,
    momentum_continuation,
    solve_flow_march,
)
from aquaflux.io import read_openfoam
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CorrectedGreenGauss, MultipleCorrectionGradient, SkewCorrectedGradient
from aquaflux.solve import (
    CompleteLu,
    Convergence,
    DualTimeLoop,
    Euclidean,
    MaterializedJacobian,
    RowScaled,
    SimpleSmoothed,
    newton_march,
)

HERE = Path(__file__).resolve().parent
POLYMESH = HERE / "of_case" / "constant" / "polyMesh"

RHO, MU, U_IN = 1.0, 5e-4, 1.0
MAX_STEPS = int(os.environ.get("LAM_MAX_STEPS", "60"))
SCHEME = os.environ.get("LAM_SCHEME", "corrected")
START = os.environ.get("LAM_START", "plug")
MEASURE = os.environ.get("LAM_MEASURE", "rowscaled")
ARMS = os.environ.get("LAM_ARMS", "bare,staged,lu,simple").split(",")
ATOL = 1e-8  # the staged arms' stop, in the chosen measure
BARE_TARGET = 1.7e-10  # the Euclidean target the issue quoted for the bare arm


def build_case() -> MomentumContinuity:
    """The duct's laminar flow assembler, with the chosen gradient reconstruction."""
    mesh = read_openfoam(POLYMESH)
    scheme = (
        CorrectedGreenGauss()
        if SCHEME == "corrected"
        else MultipleCorrectionGradient(fallback=SkewCorrectedGradient())
    )
    return MomentumContinuity.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"viscosity": Constant(jnp.asarray(MU)), "density": Constant(RHO)}),
        BoundaryConditions(
            {
                "inlet": VelocityInlet(velocity=(U_IN, 0.0, 0.0)),
                "outlet": PressureOutlet(pressure=0.0),
                "walls": NoSlipWall(),
            }
        ),
        gradient_scheme=scheme,
        advection_scheme=FirstOrderUpwind(),
    )


def initial_state(momentum: MomentumContinuity) -> jnp.ndarray:
    """A uniform inlet-speed plug with zero pressure, or rest (``LAM_START=rest``)."""
    if START == "rest":
        return momentum.initial_state()
    n, dim = momentum.mesh.n_cells, momentum.mesh.dim
    velocity = jnp.zeros((n, dim)).at[:, 0].set(U_IN)
    return momentum.pack(velocity, jnp.zeros(n))


def _log(arm: str):
    started = time.perf_counter()

    def on_step(report) -> None:
        print(
            f"[{arm}] step {report.step:3d}  cycles {report.cycles:3d}  "
            f"|R| {report.residual_norm:.3e}  ratio {report.residual_ratio:.3e}  "
            f"t {time.perf_counter() - started:6.1f}s",
            flush=True,
        )

    return on_step


def run_bare(momentum: MomentumContinuity, state: jnp.ndarray) -> tuple[bool, int, float]:
    """The configuration the issue reported: ``newton_march`` over ``momentum_continuation``, Euclidean."""
    result = newton_march(
        momentum_continuation(momentum),
        momentum.residual,
        state,
        max_steps=MAX_STEPS,
        rtol=0.0,
        atol=BARE_TARGET,
        observer=_log("bare"),
    )
    final = float(jnp.linalg.norm(momentum.residual(result.state)))
    return bool(result.converged), len(result.reports), final


def run_staged(
    arm: str, momentum: MomentumContinuity, state: jnp.ndarray, **march
) -> tuple[bool, int, float]:
    """``solve_flow_march``; reports whether it converged, its step count and its last measured residual."""
    reports = []
    log = _log(arm)

    def observe(report) -> None:
        reports.append(report)
        log(report)

    try:
        solve_flow_march(
            momentum,
            state,
            convergence=Convergence(
                measure=Euclidean() if MEASURE == "euclid" else RowScaled(), rtol=0.0, atol=ATOL
            ),
            max_steps=MAX_STEPS,
            on_step=observe,
            **march,
        )
        converged = True
    except eqx.EquinoxRuntimeError as error:
        print(f"[{arm}] FAILED: {str(error)[:160]}", flush=True)
        converged = False
    last = reports[-1].residual_norm if reports else float("nan")
    return converged, len(reports), float(last)


def main() -> None:
    momentum = build_case()
    state = initial_state(momentum)
    print(
        f"laminar duct: {momentum.mesh.n_cells} cells, Re_Dh {RHO * U_IN * 0.025 / MU:.0f}, "
        f"start {START}, measure {MEASURE}, gradient scheme {SCHEME}, max steps {MAX_STEPS}, |R0| "
        f"{float(jnp.linalg.norm(momentum.residual(state))):.3e}",
        flush=True,
    )
    runners = {
        "bare": lambda: run_bare(momentum, state),
        "staged": lambda: run_staged("staged", momentum, state),
        "lu": lambda: run_staged(
            "lu",
            momentum,
            state,
            preconditioner=MaterializedJacobian(CompleteLu(backend="scipy")),
            dual_time=DualTimeLoop(inner_steps=3),
        ),
        "simple": lambda: run_staged(
            "simple",
            momentum,
            state,
            preconditioner=MaterializedJacobian(SimpleSmoothed()),
            dual_time=DualTimeLoop(inner_steps=3),
        ),
    }
    rows = []
    for arm in ARMS:
        started = time.perf_counter()
        converged, steps, residual = runners[arm]()
        rows.append((arm, converged, steps, residual, time.perf_counter() - started))
    print("\narm      converged  steps  last |R|      wall", flush=True)
    for arm, converged, steps, residual, wall in rows:
        print(f"{arm:8s} {converged!s:9s} {steps:5d}  {residual:.3e}  {wall:7.1f}s", flush=True)


if __name__ == "__main__":
    main()
