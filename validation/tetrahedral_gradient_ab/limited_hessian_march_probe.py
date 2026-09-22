"""Issue #435: does the coupled march start once the Hessian correction is dropped on ill-conditioned cells?

A uniform-plug start at the Re/10 anchor, exact complete-LU preconditioner, 10 outer steps -- the same
configuration under which the full multiple-correction scheme stalls and corrected Green--Gauss marches.
The gradient is the full multiple-correction reconstruction except the first pass on cells whose
``max|M2^-1|`` exceeds ``TET_M2_LIMIT`` (default 10).

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/limited_hessian_march_probe.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax.numpy as jnp
from aquaflux.io import read_openfoam
from aquaflux.schemes import MultipleCorrectionGradient, OwnerGradient, SkewCorrectedGradient
from aquaflux.solve import CompleteLu, DualTimeLoop, MaterializedJacobian
from aquaflux.turbulence import (
    SSTModel,
    inlet_k,
    inlet_omega,
    solve_coupled,
)
from compare import (
    BACKEND,
    INNER_STEPS,
    INNER_TOL,
    INTENSITY,
    LENGTH_SCALE,
    POLYMESH,
    RETRY,
    U_IN,
    build_case,
)
from diffusion_operator_probe import FirstPassWhereIllConditioned

LIMIT = float(os.environ.get("TET_M2_LIMIT", "10"))
STEPS = int(os.environ.get("TET_STEPS", "10"))
CLOSURES = {
    "owner": lambda: MultipleCorrectionGradient(boundary_closure=OwnerGradient(), fallback=None),
    "repaired": lambda: MultipleCorrectionGradient(
        boundary_closure=OwnerGradient(), fallback=SkewCorrectedGradient()
    ),
}


def main() -> None:
    mesh = read_openfoam(POLYMESH)
    geometry = mesh.geometry()
    for name in os.environ.get("TET_ARMS", "repaired,owner").split(","):
        bound = CLOSURES[name]().bind(mesh, geometry)
        coupled = build_case(FirstPassWhereIllConditioned(bound, LIMIT))
        momentum = coupled.momentum
        n = momentum.mesh.n_cells
        flow = momentum.pack(jnp.zeros((n, 3)).at[:, 0].set(U_IN), jnp.zeros(n))
        k_in = float(inlet_k(jnp.array(U_IN), INTENSITY))
        omega_in = float(inlet_omega(jnp.array(k_in), LENGTH_SCALE, SSTModel()))
        print(
            f"=== {name}, first pass where max|M2^-1| > {LIMIT:g}, plug IC, Re/10 ===", flush=True
        )
        started = time.perf_counter()

        def on_step(report, started=started):
            print(
                f"  step {report.step:3d} |R| {report.residual_norm:.4e} "
                f"ratio {report.residual_ratio:.3e} alpha {report.alpha:.4g} cyc {report.cycles} "
                f"esc {report.escalations}  {time.perf_counter() - started:.0f}s",
                flush=True,
            )

        try:
            solve_coupled(
                coupled.with_scaled_molecular_viscosity(10.0),
                flow,
                jnp.full(n, k_in),
                jnp.full(n, omega_in),
                preconditioner=MaterializedJacobian(CompleteLu(backend=BACKEND)),
                dual_time=DualTimeLoop(inner_steps=INNER_STEPS, inner_tol=INNER_TOL),
                max_steps=STEPS,
                rtol=1e-3,
                atol=0.0,
                positivity_projection=True,
                retry=RETRY,
                on_step=on_step,
            )
            print("  CONVERGED", flush=True)
        except Exception as exc:  # the march's own guard raises on non-convergence -- report it
            print(f"  ended: {type(exc).__name__}: {str(exc).splitlines()[0][:140]}", flush=True)


if __name__ == "__main__":
    main()
