"""Does the shipped self-start work on this mesh, now that a reconstruction marches it? (issue #435)

``hybrid_initialize`` seeds a coupled RANS problem with potential flow: one scalar Laplace solve
whose operator carries the same gradient reconstruction the flow's residual uses, preconditioned by
a smoothed-aggregation V-cycle. On this mesh that solve raised a stagnation under **both**
multiple-correction closures, and the exact Jacobian of the same scalar problem came back at a
condition number of 3.3e19 / 1.7e19 -- past what float64 can solve. That was read at the time as a
property of the mesh; the marches have since shown the weights to be what fails, so it is worth
asking directly whether the reconstruction was the cause here too.

Per arm this reports, in order:

1. whether ``hybrid_initialize`` returns at all, and what it returns (velocity and ``k``/``omega``
   ranges, and the mean inlet-normal speed against the case's own ``U_IN``);
2. the condition number of the **exact** Jacobian of that same Laplace operator, materialized by
   ``jacfwd`` -- the quantity the earlier diagnosis turned on, measured the same way;
3. whether the coupled march then runs **from that seed** rather than from the hand-built one, which
   is the question that decides whether the self-start is usable here (``TET_SEED_MARCH=0`` skips it).

Settings: ``TET_SEED_ARMS`` (comma list of ``compare.MARCH_ARMS`` names, default all three),
``TET_BLEND`` as elsewhere, ``TET_SEED_MARCH``.

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/potential_flow_probe.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.boundary import BoundaryConditions, Dirichlet, Neumann, ZeroGradient
from aquaflux.flow import PressureOutlet, VelocityInlet
from aquaflux.flow.initialization import laplace_field
from aquaflux.initialization import hybrid_initialize
from compare import MARCH_ARMS, U_IN, build_case, run_march_ab

ARMS = os.environ.get("TET_SEED_ARMS", "projected,owner,repaired").split(",")
MARCH = os.environ.get("TET_SEED_MARCH", "1") != "0"


def potential_conditions(momentum) -> BoundaryConditions:
    """The potential's boundary data, as ``potential_flow`` derives it.

    A prescribed potential at the pressure outlet, the inlet's normal speed as a Neumann datum
    (``Gamma = 1``, so ``q = -u_in . n``), and no penetration at the walls. Read here only to
    materialize that operator's Jacobian -- the solve itself goes through ``hybrid_initialize``, so
    this probe never substitutes its own operator for the shipped one.

    This duct has a pressure outlet, so there is a potential datum and no pinned cell; a case without
    one is refused rather than silently measured against a different (singular) operator.
    """
    mesh, geometry = momentum.mesh, momentum.geometry
    conditions: dict[str, object] = {}
    has_reference = False
    for name, closure in momentum.boundary.conditions.items():
        if isinstance(closure, PressureOutlet):
            conditions[name] = Dirichlet(0.0)
            has_reference = True
        elif isinstance(closure, VelocityInlet):
            faces = mesh.face_patches.indices(name)
            normal = geometry.face.normal[faces]
            velocity = closure.reference_velocity(normal, geometry.face.centroid[faces])
            conditions[name] = Neumann(flux=-float(jnp.mean(jnp.sum(velocity * normal, axis=1))))
        else:
            conditions[name] = ZeroGradient()
    if not has_reference:
        raise ValueError("this probe expects a pressure outlet to carry the potential's datum")
    return BoundaryConditions(conditions)


def laplace_conditioning(coupled, label: str) -> None:
    """The condition number of the exact potential-flow Jacobian under this arm's reconstruction.

    Measured from the assembler ``laplace_field`` returns, so the operator is the one the solve
    actually forms rather than a re-derivation of it. That costs a second Laplace solve on top of
    ``hybrid_initialize``'s, which on this mesh is seconds; where that solve stagnates there is no
    assembler to read and the stagnation is itself the reported result.
    """
    momentum = coupled.momentum
    mesh, geometry = momentum.mesh, momentum.geometry
    try:
        _, assembler = laplace_field(
            mesh,
            geometry,
            potential_conditions(momentum),
            gradient_scheme=momentum.gradient_scheme,
        )
    except eqx.EquinoxRuntimeError as exc:
        print(f"  [{label}] the Laplace solve stagnated, no operator to read: {exc}", flush=True)
        return
    jacobian = np.asarray(jax.jacfwd(assembler.residual)(jnp.zeros(mesh.n_cells)))
    singular = np.linalg.svd(jacobian, compute_uv=False)
    condition = singular[0] / singular[-1] if singular[-1] > 0 else np.inf
    print(
        f"  [{label}] exact Laplace Jacobian: condition {condition:.2e} "
        f"(largest {singular[0]:.3e}, smallest {singular[-1]:.3e})",
        flush=True,
    )


def report_seed(coupled, label: str) -> bool:
    """Run ``hybrid_initialize`` on this arm and describe what it returns. True if it returned."""
    momentum = coupled.momentum
    started = time.perf_counter()
    try:
        flow, k, omega = hybrid_initialize(coupled)
    except eqx.EquinoxRuntimeError as exc:
        elapsed = time.perf_counter() - started
        print(f"  [{label}] hybrid_initialize FAILED after {elapsed:.1f}s: {exc}", flush=True)
        return False
    elapsed = time.perf_counter() - started
    velocity, _ = momentum.unpack(flow)
    speed = np.linalg.norm(np.asarray(velocity), axis=-1)
    inlet = np.asarray(momentum.mesh.face_patches.indices("inlet"))
    normal = np.asarray(momentum.geometry.face.normal[inlet])
    owner = np.asarray(momentum.mesh.face_cells.owner)[inlet]
    through = float(np.mean(np.sum(np.asarray(velocity)[owner] * normal, axis=1)))
    finite = bool(np.all(np.isfinite(speed)))
    print(
        f"  [{label}] hybrid_initialize returned in {elapsed:.1f}s: finite {finite}, "
        f"|u| max {speed.max():.3f} mean {speed.mean():.3f} (U_IN {U_IN}), "
        f"mean inlet-normal {through:+.3f}, k [{float(jnp.min(k)):.3g}, {float(jnp.max(k)):.3g}], "
        f"omega [{float(jnp.min(omega)):.3g}, {float(jnp.max(omega)):.3g}]",
        flush=True,
    )
    return finite


def main() -> None:
    seeded = {}
    for name in ARMS:
        print(f"\n=== {name} ===", flush=True)
        coupled = build_case(MARCH_ARMS[name]())
        seeded[name] = report_seed(coupled, name)
        laplace_conditioning(coupled, name)

    if not MARCH:
        return
    for name in ARMS:
        if not seeded[name]:
            print(f"\n[{name}] no usable seed, so no march from it", flush=True)
            continue
        print(f"\n=== {name}: the march, started from hybrid_initialize ===", flush=True)
        run_march_ab(name, MARCH_ARMS[name](), start=hybrid_initialize)


if __name__ == "__main__":
    main()
