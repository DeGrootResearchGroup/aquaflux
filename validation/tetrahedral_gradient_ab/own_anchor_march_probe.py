"""Issue #435: does each gradient scheme converge its OWN low-Re anchor from the plug, or only when
warm-started from CorrectedGreenGauss's root?

``scheme_march_comparison_probe.py`` found that every scheme except ``CorrectedGreenGauss`` fails when
marched from ``CorrectedGreenGauss``'s converged seed -- including ``CompactGreenGauss``, which has no
relationship to ``MultipleCorrectionGradient`` at all. That leaves two explanations open: either
warm-starting FROM a different scheme's root produces too large an initial mismatch for these schemes to
recover from, or these schemes cannot reach this Reynolds number's anchor at all, independent of how
they are started.

This probe settles it directly: it builds each candidate scheme's OWN coupled case at the anchor
viscosity (Re/RATIO, the same low-Re problem ``CorrectedGreenGauss``'s own seed solves) and marches it
from the SAME uniform plug initial condition ``CorrectedGreenGauss``'s seed starts from -- never touching
another scheme's converged state. If a scheme converges here, its earlier failure was a warm-start
artifact; if it fails here too, the failure is intrinsic to the scheme on this mesh at this Reynolds
number.

The ``corrected GG sweeps=N`` / ``corrected GG gmres`` arms resolve the default's under-resolved
(4-sweep) correction -- the warning every run on this mesh prints -- to test whether that partial
correction was masking a divergence a fully-resolved ``CorrectedGreenGauss`` would also show.

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/own_anchor_march_probe.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax.numpy as jnp
import numpy as np
from aquaflux.io import read_openfoam
from aquaflux.schemes import (
    CompactGreenGauss,
    CorrectedGreenGauss,
    GmresGradientSolve,
    MultipleCorrectionGradient,
    OwnerGradient,
    SkewCorrectedGradient,
    SweptGradientSolve,
)
from aquaflux.turbulence import SSTModel, inlet_k, inlet_omega
from compare import INTENSITY, LENGTH_SCALE, POLYMESH, U_IN, build_case
from diffusion_operator_probe import FirstPassOnCells, FirstPassOnly, FirstPassWhereIllConditioned
from seed_state_probe import rings_from_wall
from wall_velocity_gradient_probe import with_wall_model
from warm_start_probe import RATIO, march

STEPS = int(os.environ.get("TET_STEPS", "25"))
ARMS = os.environ.get("TET_ARMS", "corrected GG,compact GG,multcorr repaired,option 1")


def main() -> None:
    k_in = float(inlet_k(jnp.array(U_IN), INTENSITY))
    omega_in = float(inlet_omega(jnp.array(k_in), LENGTH_SCALE, SSTModel()))

    multcorr_repaired = MultipleCorrectionGradient(
        boundary_closure=OwnerGradient(), fallback=SkewCorrectedGradient()
    )
    mesh = read_openfoam(POLYMESH)
    bound_repaired = multcorr_repaired.bind(mesh, mesh.geometry())
    # Graph distance (face hops) of each cell from the wall cells, for the ring bisection arms below.
    rings = rings_from_wall(mesh, build_case(CorrectedGreenGauss()).turbulence.wall_cells)

    def hessian_off_through(ring: int):
        cells = jnp.asarray(np.flatnonzero(rings <= ring))
        return lambda: build_case(FirstPassOnCells(bound_repaired, cells))

    builders = {
        "corrected GG": lambda: build_case(CorrectedGreenGauss()),
        # The default (4 sweeps) leaves the correction under-resolved on this mesh -- the warning every
        # run prints. These arms resolve it: 12 sweeps, then well past the floating-point floor, then an
        # exact Krylov solve, to test whether the default's partial correction was masking a divergence.
        "corrected GG sweeps=12": lambda: build_case(
            CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=12))
        ),
        "corrected GG sweeps=40": lambda: build_case(
            CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=40))
        ),
        "corrected GG gmres": lambda: build_case(CorrectedGreenGauss(solver=GmresGradientSolve())),
        "compact GG": lambda: build_case(CompactGreenGauss()),
        "multcorr repaired": lambda: build_case(multcorr_repaired),
        # The full scheme everywhere except the ~184 cells whose Hessian is ill-conditioned
        # (max|M2^-1| > 10), where it falls back to its linear-exact first pass -- for EVERY field.
        "multcorr M2<=10": lambda: build_case(FirstPassWhereIllConditioned(bound_repaired, 10.0)),
        # The multiple-correction first pass alone for every field: linear-exact, no Hessian, no M2.
        "multcorr first pass": lambda: build_case(FirstPassOnly(bound_repaired)),
        # Ring bisection: the full scheme, except its Hessian is withheld (first pass only) in the wall
        # cells (ring 0) and every ring out to N, for EVERY field.
        "hessian off rings<=0": hessian_off_through(0),
        "hessian off rings<=1": hessian_off_through(1),
        "hessian off rings<=2": hessian_off_through(2),
        "hessian off rings<=3": hessian_off_through(3),
        "option 1": lambda: with_wall_model(build_case(multcorr_repaired), first_ring=True),
    }

    for label in ARMS.split(","):
        if label not in builders:
            print(f"skipping unknown arm {label!r}", flush=True)
            continue
        case = builders[label]().with_scaled_molecular_viscosity(RATIO)
        n = case.momentum.mesh.n_cells
        flow0 = case.momentum.pack(jnp.zeros((n, 3)).at[:, 0].set(U_IN), jnp.zeros(n))
        k0 = jnp.full(n, k_in)
        omega0 = jnp.full(n, omega_in)
        print(f"=== {label}, own anchor from the plug, Re/{RATIO:g} ===", flush=True)
        march(label, case, flow0, k0, omega0, 1e-4, STEPS)


if __name__ == "__main__":
    main()
