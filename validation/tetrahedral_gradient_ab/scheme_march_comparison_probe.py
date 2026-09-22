"""Issue #435: does the divergence trace to the gradient SCHEME, not the wall/production treatment?

Every fix tried so far (the wall-model velocity gradient, ring-1's Hessian exclusion, the two
KProduction Jacobian freezes) targeted a specific symptom and only partly helped. But the SEED itself
marches this exact problem -- same mesh, same Reynolds number, same turbulence model, same boundary
conditions -- to convergence, using ``CorrectedGreenGauss``. If several OTHER gradient schemes also hold
at the same conditions while ``MultipleCorrectionGradient`` does not (even patched), the failure is a
property of that scheme's two-pass Hessian-correction machinery itself, not of the turbulence closure or
the wall treatment every arm shares.

This marches the SAME converged seed under four arms, all at the seed's own Reynolds number (no
continuation to a higher target -- this is exactly the same problem the seed just solved, only
re-discretized). ``SkewCorrectedGradient`` is not itself a standalone scheme -- it is a
``GradientBoundaryClosure``, the strategy ``MultipleCorrectionGradient`` uses to read a boundary value,
not something ``build_case`` can bind on its own -- so it appears only inside the ``multcorr`` arms
below, exactly as every other probe in this case uses it:

* ``corrected GG`` -- the control. This is the scheme the seed itself used, so it should re-converge in
  essentially no steps; a failure here would mean the comparison itself is broken.
* ``compact GG`` -- a single-pass, non-Hessian scheme with no boundary-condition weight at all.
* ``multcorr repaired`` -- ``MultipleCorrectionGradient`` with the corner-cell first-pass fix
  (``fallback=SkewCorrectedGradient``), no wall-model velocity treatment.
* ``option 1`` -- ``multcorr repaired`` plus the wall-model imposed velocity gradient and ring-1
  Hessian exclusion (``wall_velocity_gradient_probe.with_wall_model``).

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/scheme_march_comparison_probe.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax.numpy as jnp
from aquaflux.io import read_openfoam
from aquaflux.schemes import (
    CompactGreenGauss,
    CorrectedGreenGauss,
    MultipleCorrectionGradient,
    OwnerGradient,
    SkewCorrectedGradient,
)
from aquaflux.turbulence import SSTModel, inlet_k, inlet_omega
from compare import INTENSITY, LENGTH_SCALE, POLYMESH, U_IN, build_case
from wall_velocity_gradient_probe import with_wall_model
from warm_start_probe import RATIO, SEED_STEPS, march

STEPS = int(os.environ.get("TET_STEPS", "15"))
ARMS = os.environ.get("TET_ARMS", "corrected GG,compact GG,multcorr repaired,option 1")


def main() -> None:
    mesh = read_openfoam(POLYMESH)
    seed_case = build_case(CorrectedGreenGauss()).with_scaled_molecular_viscosity(RATIO)
    n = mesh.n_cells
    flow0 = seed_case.momentum.pack(jnp.zeros((n, 3)).at[:, 0].set(U_IN), jnp.zeros(n))
    k0 = jnp.full(n, float(inlet_k(jnp.array(U_IN), INTENSITY)))
    omega0 = jnp.full(n, float(inlet_omega(jnp.array(k0[0]), LENGTH_SCALE, SSTModel())))
    seed = march("seed", seed_case, flow0, k0, omega0, 1e-4, SEED_STEPS)
    if seed is None:
        return

    multcorr_repaired = MultipleCorrectionGradient(
        boundary_closure=OwnerGradient(), fallback=SkewCorrectedGradient()
    )
    arms = {
        "corrected GG": build_case(CorrectedGreenGauss()).with_scaled_molecular_viscosity(RATIO),
        "compact GG": build_case(CompactGreenGauss()).with_scaled_molecular_viscosity(RATIO),
        "multcorr repaired": build_case(multcorr_repaired).with_scaled_molecular_viscosity(RATIO),
        "option 1": with_wall_model(
            build_case(multcorr_repaired).with_scaled_molecular_viscosity(RATIO), first_ring=True
        ),
    }
    for label in ARMS.split(","):
        if label not in arms:
            print(f"skipping unknown arm {label!r}", flush=True)
            continue
        print(f"=== {label}, from the seed, same Re/{RATIO:g} ===", flush=True)
        march(label, arms[label], *seed, 1e-3, STEPS)
        del arms[
            label
        ]  # release the coupled build (and any factorization it triggers) between arms


if __name__ == "__main__":
    main()
