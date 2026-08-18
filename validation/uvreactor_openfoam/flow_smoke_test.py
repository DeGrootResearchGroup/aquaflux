"""Does the coupled RANS solve even start on a real snappyHexMesh polyhedral mesh?

Exploratory, not a validated case: there is no k-omega SST OpenFOAM reference here (the source case
runs realizable k-epsilon), so this checks only that the pipeline -- import, case assembly, hybrid
initialization, and a handful of coupled Newton steps -- runs cleanly on a real general-polyhedral
mesh, at a scale (~1.6M cells at ``of_case``'s own settings) meant to give the GPU enough parallel
work per kernel to matter. It exists to de-risk before spending real wall time on a cluster GPU.

Mesh: ``of_case/`` is the untouched uvReactorSozzi2006 geometry
(https://github.com/DeGrootResearchGroup/of-optical-radiation), at its own snappyHexMesh refinement
settings; see this directory's README for how to build it. OpenFOAM's own checkMesh reports it clean.
Every interior and boundary face's normal-distance denominator in the diffusion flux is now strictly
positive on this mesh (verified directly, not just inferred from checkMesh) -- a cell whose size is
small relative to its distance from the coordinate origin needs the divergence-theorem centroid
accumulation to be done in local coordinates to keep that true (`aquaflux/mesh/cell.py`); without it,
`hybrid_initialize` failed outright on this same mesh.

Operating point, from the source case's ``0/`` and ``constant/physicalProperties`` (25 GPM through a
19.1 mm pipe, water at 20 C, realizable k-epsilon converted to k-omega via omega = epsilon/(Cmu k)):
Re ~= 1.05e5. At that Reynolds number a direct march (no continuation) is not expected to reach the
stopping tolerance in a handful of steps -- this script's own march intentionally does not attempt
that; it only checks the residual stays finite and the march runs some real steps before whatever it
eventually hits.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import aquaflux  # noqa: E402,F401  (enables x64)
import jax.numpy as jnp  # noqa: E402
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient  # noqa: E402
from aquaflux.discretization import FirstOrderUpwind, LimitedUpwind  # noqa: E402
from aquaflux.flow import (  # noqa: E402
    MomentumContinuity,
    NoSlipWall,
    PressureOutlet,
    VelocityInlet,
)
from aquaflux.io import read_openfoam  # noqa: E402
from aquaflux.properties import Constant, PropertyModel  # noqa: E402
from aquaflux.schemes import CorrectedGreenGauss, VenkatakrishnanLimiter  # noqa: E402
from aquaflux.solve import MarchLogger  # noqa: E402
from aquaflux.turbulence import (  # noqa: E402
    CoupledRANS,
    LogScalars,
    SSTModel,
    SSTTurbulence,
    coupled_fields,
    hybrid_initialize,
    solve_coupled,
)

MESH = HERE / "of_case" / "constant" / "polyMesh"

# Water at 20 C (constant/physicalProperties).
RHO, NU = 998.0, 1e-6
# 25 GPM through the 19.1 mm inlet pipe; enters at x=1.739 travelling in -x (0/U).
U_IN = (-5.51, 0.0, 0.0)
# 0/k: k = 1.5*(TI*U)^2, TI=0.10. 0/epsilon: eps = Cmu^0.75 k^1.5 / (0.07*D_inlet).
K_IN = 0.4555
EPSILON_IN = 37.83
# aquaflux transports omega, not epsilon: the standard k-epsilon/k-omega relation, Cmu = 0.09.
OMEGA_IN = EPSILON_IN / (0.09 * K_IN)

WALLS = ("bodyWall", "lampWall")

#: A smoke test needs only enough steps to see whether the residual is finite and moving in the
#: right direction, not convergence -- Re ~1e5 on a mesh this stiff is not expected to reach the
#: root in a handful of direct steps (the sibling cases all need Reynolds continuation for that).
MAX_STEPS = 8


def build_case():
    mesh = read_openfoam(MESH)
    geom = mesh.geometry()

    grad = CorrectedGreenGauss()
    momentum_upwind = LimitedUpwind(limiter=VenkatakrishnanLimiter())
    scalar_upwind = FirstOrderUpwind()

    momentum = MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(jnp.asarray(RHO * NU)), "density": Constant(RHO)}),
        grad,
        BoundaryConditions(
            {
                "inlet": VelocityInlet(velocity=U_IN),
                "outlet": PressureOutlet(pressure=0.0),
                "bodyWall": NoSlipWall(),
                "lampWall": NoSlipWall(),
            }
        ),
        advection_scheme=momentum_upwind,
    )
    turbulence = SSTTurbulence.build(
        SSTModel(),
        mesh,
        geom,
        grad,
        scalar_upwind,
        density=RHO,
        molecular_viscosity=jnp.full(mesh.n_cells, NU),
        wall_patches=WALLS,
        explicit_production_limiter=True,
        k_boundary=BoundaryConditions(
            {
                "inlet": Dirichlet(K_IN),
                "outlet": ZeroGradient(),
                "bodyWall": ZeroGradient(),
                "lampWall": ZeroGradient(),
            }
        ),
        omega_boundary=BoundaryConditions(
            {
                "inlet": Dirichlet(OMEGA_IN),
                "outlet": ZeroGradient(),
                "bodyWall": ZeroGradient(),
                "lampWall": ZeroGradient(),
            }
        ),
    )
    # Log-omega: every case this package has run at a comparable Reynolds number needs it -- a
    # direct-omega Newton step drives omega negative once the flow develops, and the residual stays
    # finite so the divergence guard never trips.
    coupled = CoupledRANS.build(momentum, turbulence, omega_transform=LogScalars())
    return dict(coupled=coupled, momentum=momentum, turbulence=turbulence, geom=geom)


def main() -> None:
    print(f"mesh: {MESH}", flush=True)
    t0 = time.time()
    case = build_case()
    coupled, momentum, turbulence = case["coupled"], case["momentum"], case["turbulence"]
    print(
        f"case built in {time.time() - t0:.1f}s, {coupled.layout.dim}D, "
        f"{momentum.mesh.n_cells} cells",
        flush=True,
    )

    flow, k, omega = hybrid_initialize(momentum, turbulence)
    state = coupled.state_from_physical(flow, k, omega)

    r0 = coupled.residual(state)
    print(
        f"initial residual finite: {bool(jnp.all(jnp.isfinite(r0)))}, "
        f"||R0|| = {float(jnp.linalg.norm(r0)):.3e}",
        flush=True,
    )

    logger = MarchLogger(fields=coupled_fields(coupled), detail=("fields",), rtol=0.0, atol=1e-8)
    t0 = time.time()
    # A smoke test does not expect MAX_STEPS to reach the stopping tolerance at this Reynolds number
    # (Re ~1e5, no continuation) -- catch the convergence guard's raise so the per-step log already
    # printed by `logger.on_checkpoint` still tells us whether the march was ever going to converge.
    try:
        solve_coupled(
            coupled,
            flow,
            k,
            omega,
            max_steps=MAX_STEPS,
            rtol=0.0,
            atol=1e-8,
            on_checkpoint=logger.on_checkpoint,
        )
    except Exception as exc:
        print(f"\ndid not converge in {MAX_STEPS} steps (expected): {type(exc).__name__}: {exc}")
    print(f"\n{MAX_STEPS}-step smoke march: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
