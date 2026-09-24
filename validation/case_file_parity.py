"""Does a case file build the same problem as the case assembled by hand?

For each case with a ``case.yaml``, the problem :meth:`aquaflux.case.CheckedCase.build` assembles is
compared against the same case written out by hand, two ways:

1. **the same pytree** -- one tree structure (which includes every static field, and so every compile
   cache key) and every array leaf bit-for-bit equal;
2. **the same residual** -- bit-for-bit, at the reference problem's own hybrid initial condition.

The first is the strong check: two problems that are one pytree evaluate every residual, Jacobian and
adjoint identically. The second is the one a reader recognizes, and it guards against a leaf the tree
comparison could see as equal but a residual reads differently (there should be none).

The references are deliberately independent of the case files:

* **pitzDaily** -- :func:`_hand_built_pitzdaily` below, a frozen copy of the assembly
  ``pitzdaily_openfoam/compare.py`` wrote out by hand before it was switched to read its case file (at
  that driver's defaults: the multiple-correction gradient, ``k`` zero-gradient at the walls). The
  driver itself cannot be the reference any more, since it now builds from the very file under test.
* **bfs3d** -- ``bfs3d_openfoam/compare.py``'s own ``build_case()``, which still assembles by hand, at
  its defaults (corrected Green--Gauss, ``k`` zero-gradient at the walls, the production limiter on).

Run from the repository root::

    python -u validation/case_file_parity.py [pitzdaily] [bfs3d]

It prints one line per check and exits non-zero if any fails.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient  # noqa: E402
from aquaflux.case import read_case  # noqa: E402
from aquaflux.discretization import FirstOrderUpwind, LimitedUpwind  # noqa: E402
from aquaflux.flow import (  # noqa: E402
    MomentumContinuity,
    NoSlipWall,
    PressureOutlet,
    VelocityInlet,
)
from aquaflux.initialization import hybrid_initialize  # noqa: E402
from aquaflux.io import read_openfoam  # noqa: E402
from aquaflux.properties import Constant, PropertyModel  # noqa: E402
from aquaflux.schemes import MultipleCorrectionGradient, VenkatakrishnanLimiter  # noqa: E402
from aquaflux.turbulence import CoupledRANS, LogScalars, SSTModel, SSTTurbulence  # noqa: E402


def _hand_built_pitzdaily() -> CoupledRANS:
    """pitzDaily as its driver assembled it by hand, at the driver's defaults -- the reference."""
    rho, nu = 1.0, 1e-5
    u_in, k_in, omega_in = 10.0, 0.375, 440.15
    walls = ["upperWall", "lowerWall"]
    mesh = read_openfoam(HERE / "pitzdaily_openfoam" / "runs" / "kwsst" / "polyMesh")
    geom = mesh.geometry()
    grad = MultipleCorrectionGradient()
    properties = PropertyModel(
        {"viscosity": Constant(jnp.asarray(rho * nu)), "density": Constant(rho)}
    )
    momentum = MomentumContinuity.build(
        mesh,
        geom,
        properties,
        BoundaryConditions(
            {
                "inlet": VelocityInlet(velocity=(u_in, 0.0)),
                "outlet": PressureOutlet(pressure=0.0),
                "upperWall": NoSlipWall(),
                "lowerWall": NoSlipWall(),
            }
        ),
        gradient_scheme=grad,
        advection_scheme=LimitedUpwind(limiter=VenkatakrishnanLimiter()),
    )
    turbulence = SSTTurbulence.build(
        SSTModel(),
        mesh,
        geom,
        FirstOrderUpwind(),
        properties,
        gradient_scheme=grad,
        wall_patches=walls,
        explicit_production_limiter=True,
        k_boundary=BoundaryConditions(
            {
                "inlet": Dirichlet(k_in),
                "outlet": ZeroGradient(),
                "upperWall": ZeroGradient(),
                "lowerWall": ZeroGradient(),
            }
        ),
        omega_boundary=BoundaryConditions(
            {
                "inlet": Dirichlet(omega_in),
                "outlet": ZeroGradient(),
                "upperWall": ZeroGradient(),
                "lowerWall": ZeroGradient(),
            }
        ),
    )
    return CoupledRANS.build(momentum, turbulence, omega_transform=LogScalars())


def _bfs3d_driver_build() -> CoupledRANS:
    """bfs3d as its driver assembles it, at the driver's defaults -- the reference."""
    path = HERE / "bfs3d_openfoam" / "compare.py"
    spec = importlib.util.spec_from_file_location("bfs3d_case_parity", path)
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs: the driver's dataclasses resolve their annotations through it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_case()["coupled"]


REFERENCES = {
    "pitzdaily": (_hand_built_pitzdaily, HERE / "pitzdaily_openfoam" / "case.yaml"),
    "bfs3d": (_bfs3d_driver_build, HERE / "bfs3d_openfoam" / "case.yaml"),
}


def _tree_mismatches(built: object, reference: object) -> list[str]:
    """Every way the two pytrees differ: their structure, or an array leaf's shape, type or bits."""
    built_leaves, built_def = jax.tree.flatten(built)
    reference_leaves, reference_def = jax.tree.flatten(reference)
    if built_def != reference_def:
        return ["tree structure (static fields included) differs"]
    problems = []
    for i, (a, b) in enumerate(zip(built_leaves, reference_leaves, strict=True)):
        if eqx.is_array(a) != eqx.is_array(b):
            # A number beside an equal array: equal values, but a different compiled program.
            problems.append(f"leaf {i}: {type(a).__name__} vs {type(b).__name__}")
        elif eqx.is_array(a):
            a, b = np.asarray(a), np.asarray(b)
            if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
                problems.append(f"leaf {i}: {a.dtype}{a.shape} vs {b.dtype}{b.shape}")
        elif a != b:
            problems.append(f"leaf {i}: {a!r} vs {b!r}")
    return problems


def compare(name: str) -> bool:
    """Build ``name`` both ways and report whether they are one problem; ``True`` if they are."""
    reference_build, case_file = REFERENCES[name]
    start = time.perf_counter()
    reference = reference_build()
    built = read_case(case_file).check().build()
    print(f"[{name}] built both in {time.perf_counter() - start:.1f} s", flush=True)

    mismatches = _tree_mismatches(built, reference)
    print(f"[{name}] same pytree: {'yes' if not mismatches else 'NO'}", flush=True)
    for mismatch in mismatches[:10]:
        print(f"[{name}]   {mismatch}", flush=True)

    flow, k, omega = hybrid_initialize(reference)
    state = reference.state_from_physical(flow, k, omega)
    built_residual = np.asarray(built.residual(state))
    reference_residual = np.asarray(reference.residual(state))
    same = np.array_equal(built_residual, reference_residual)
    print(
        f"[{name}] residual at the hybrid initial condition: |R| = "
        f"{np.linalg.norm(reference_residual):.6e}; bit-identical: {'yes' if same else 'NO'}"
        + (
            ""
            if same
            else f" (max difference {np.max(np.abs(built_residual - reference_residual)):.3e})"
        ),
        flush=True,
    )
    return not mismatches and same


if __name__ == "__main__":
    names = sys.argv[1:] or list(REFERENCES)
    unknown = sorted(set(names) - set(REFERENCES))
    if unknown:
        raise SystemExit(f"unknown case {unknown}; the cases are {sorted(REFERENCES)}")
    results = {name: compare(name) for name in names}
    print(
        "PASS" if all(results.values()) else f"FAIL: {[n for n, ok in results.items() if not ok]}"
    )
    sys.exit(0 if all(results.values()) else 1)
