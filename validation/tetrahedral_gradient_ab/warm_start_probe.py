"""Issue #435: does the multiple-correction scheme march from a smooth state, where it stalls from a plug?

A uniform plug beside no-slip walls is a jump at the wall scale, which the scheme's Hessian correction
spreads into interior cells as a large spurious strain rate. This converges the same Re/10 anchor with
corrected Green--Gauss from the plug, then hands that state to the multiple-correction scheme (with its
boundary-condition first pass) and marches again.

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/warm_start_probe.py
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
from aquaflux.schemes import (
    CorrectedGreenGauss,
    MultipleCorrectionGradient,
    OwnerGradient,
    SkewCorrectedGradient,
)
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

STEPS = int(os.environ.get("TET_STEPS", "25"))
SEED_STEPS = 25
#: Re-fit the complete LU mid-step once an inner solve takes this many restart cycles (0: never, the
#: first factorization is kept for the whole march -- exact only at the state and shift it was built at).
REFRESH_ON_CYCLES = int(os.environ.get("TET_REFRESH_ON_CYCLES", "0")) or None
ARMS = os.environ.get("TET_ARMS", "owner,repaired").split(",")
RATIO = 10.0


def march(label, coupled, flow, k, omega, rtol, max_steps=STEPS):
    started = time.perf_counter()

    def on_step(report):
        print(
            f"  [{label}] step {report.step:3d} |R| {report.residual_norm:.4e} "
            f"ratio {report.residual_ratio:.3e} alpha {report.alpha:.4g} cyc {report.cycles} "
            f"esc {report.escalations}  {time.perf_counter() - started:.0f}s",
            flush=True,
        )

    try:
        result = solve_coupled(
            coupled,
            flow,
            k,
            omega,
            preconditioner=MaterializedJacobian(CompleteLu(backend=BACKEND)),
            dual_time=DualTimeLoop(
                inner_steps=INNER_STEPS, inner_tol=INNER_TOL, refresh_on_cycles=REFRESH_ON_CYCLES
            ),
            max_steps=max_steps,
            rtol=rtol,
            atol=0.0,
            positivity_projection=True,
            retry=RETRY,
            on_step=on_step,
        )
        print(f"  [{label}] CONVERGED", flush=True)
        return result
    except Exception as exc:  # the march's own guard raises on non-convergence -- report it
        print(
            f"  [{label}] ended: {type(exc).__name__}: {str(exc).splitlines()[0][:140]}", flush=True
        )
        return None


def main() -> None:
    mesh = read_openfoam(POLYMESH)
    geometry = mesh.geometry()
    seed_case = build_case(CorrectedGreenGauss()).with_scaled_molecular_viscosity(RATIO)
    momentum = seed_case.momentum
    n = momentum.mesh.n_cells
    flow = momentum.pack(jnp.zeros((n, 3)).at[:, 0].set(U_IN), jnp.zeros(n))
    k_in = float(inlet_k(jnp.array(U_IN), INTENSITY))
    omega_in = float(inlet_omega(jnp.array(k_in), LENGTH_SCALE, SSTModel()))
    print("=== seed: corrected Green--Gauss from the plug, Re/10 ===", flush=True)
    seed = march(
        "seed",
        seed_case,
        flow,
        jnp.full(n, k_in),
        jnp.full(n, omega_in),
        rtol=1e-4,
        max_steps=SEED_STEPS,
    )
    if seed is None:
        return
    for name, scheme in (
        ("owner", MultipleCorrectionGradient(boundary_closure=OwnerGradient(), fallback=None)),
        (
            "repaired",
            MultipleCorrectionGradient(
                boundary_closure=OwnerGradient(), fallback=SkewCorrectedGradient()
            ),
        ),
    ):
        if name not in ARMS:
            continue
        coupled = build_case(scheme.bind(mesh, geometry)).with_scaled_molecular_viscosity(RATIO)
        print(f"=== multiple correction ({name}) from the converged seed, Re/10 ===", flush=True)
        march(name, coupled, *seed, rtol=1e-3)


if __name__ == "__main__":
    main()
