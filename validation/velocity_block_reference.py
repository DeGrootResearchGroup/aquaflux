"""Which state should the block-diagonal family's velocity block be frozen at? (issue #389)

The block-diagonal coupled preconditioner (`BlockDiagonal`, built by `_coupled_shift_policy`) freezes
its convection-aware velocity block's convective linearization at a mass flux. Until #389 it called
`BlockPreconditioner.build` without a `reference_state`, so that mass flux was the boundary-derived
**uniform plug** at the fastest prescribed speed -- while the eddy viscosity the same block is
assembled with was taken at the policy's own reference state. It now passes that reference flow. This
harness compares the two choices with everything else held fixed, each arm setting the frozen state
itself so the comparison does not depend on which one the library ships:

- ``plug``: the boundary-derived uniform flow (the build before #389);
- ``reference``: the flow part of the policy's reference state, falling back to the plug when it
  carries no mass flux (a closed domain or a start from rest), where the block would otherwise
  collapse onto the viscous operator (the build since #389).

Only the first build differs: a refresh carries the flow block over unchanged.

Three cases. Two **bump channels** (a Gaussian bump on the lower wall), on which the hybrid start's
potential flow accelerates over the crest and so differs materially from a plug: ``bump`` (0.10 H)
and ``bump-high`` (0.30 H). And the straight **channel** of `tests/integration/test_coupled_rans.py`,
where potential flow is the plug to round-off: a control whose arms must come out identical.

On a bump the plug is far from divergence-free -- it pushes the inlet flux into a channel that
narrows -- and from about 0.15 H the frozen upwind operator it gives loses positivity on the first
coarse level, so the plug arm's hierarchy refuses to build. A build refusal is recorded as the arm's
result rather than ending the run.

Run with ``validation/run_case.sh validation/velocity_block_reference.py``. Environment:
``VBR_CASES`` (comma list of ``bump``, ``bump-high``, ``channel``; default all three), ``VBR_ARMS``
(default ``plug,reference``), ``VBR_MAX_STEPS`` (default 300).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Before the `aquaflux` import: run as a script, only this file's directory is on `sys.path`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import (
    MomentumContinuity,
    NoSlipWall,
    PressureOutlet,
    VelocityInlet,
)
from aquaflux.mesh import graded_nodes, structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.turbulence import (
    BlockDiagonal,
    SSTModel,
    SSTTurbulence,
    inlet_k,
    inlet_omega,
)
from aquaflux.turbulence import coupled as coupled_module
from aquaflux.turbulence.coupled import CoupledRANS, solve_coupled

RHO, U_IN, H = 1.0, 1.0, 1.0
INTENSITY = 0.05
MAX_STEPS = int(os.environ.get("VBR_MAX_STEPS", "300"))

# --- the arm switch ---------------------------------------------------------------------------
#
# `_coupled_shift_policy` is looked up as a module global by the block session, so wrapping it records
# the state each first build is frozen at; the block builder it calls is likewise a module global, and
# each arm substitutes one that sets the frozen state its own way.

_ORIGINAL_POLICY = coupled_module._coupled_shift_policy
_ORIGINAL_BUILDER = coupled_module.BlockPreconditioner
_FROZEN = {}


def _recording_policy(coupled, reference_state, scalar, reuse=None, *args, **kwargs):
    if reuse is None:
        _FROZEN["flow"] = coupled.physical_fields(reference_state)[0]
    return _ORIGINAL_POLICY(coupled, reference_state, scalar, reuse, *args, **kwargs)


class _PlugFrozenBuilder:
    """`BlockPreconditioner` with `build` given no reference state: the boundary-derived plug."""

    @staticmethod
    def build(momentum, **options):
        options.pop("reference_state", None)
        return _ORIGINAL_BUILDER.build(momentum, **options)


class _ReferenceFrozenBuilder:
    """`BlockPreconditioner` with `build` handed the policy's reference flow, plug fallback."""

    @staticmethod
    def build(momentum, **options):
        options.pop("reference_state", None)
        flow = _FROZEN["flow"]
        if float(jnp.max(jnp.abs(momentum.mass_flux(flow)))) > 0.0:
            options["reference_state"] = flow
            _FROZEN["used"] = "reference"
        else:
            _FROZEN["used"] = "plug (reference carries no mass flux)"
        return _ORIGINAL_BUILDER.build(momentum, **options)


def _select_arm(arm: str) -> None:
    coupled_module._coupled_shift_policy = _recording_policy
    coupled_module.BlockPreconditioner = (
        _ReferenceFrozenBuilder if arm == "reference" else _PlugFrozenBuilder
    )
    _FROZEN.clear()
    _FROZEN["used"] = "plug" if arm == "plug" else "?"


# --- the cases ----------------------------------------------------------------------------------


def _bump_mesh(height, nx=72, ny=32, length=6.0, crest=1.5, width=0.35):
    """A channel whose lower wall carries a Gaussian bump; nodes sheared wall-normal onto it."""
    mesh = structured_grid_2d(
        nx, ny, lx=length, ly=H, named_boundaries=True, y_nodes=graded_nodes(ny, H, 1.15)
    )
    coords = np.asarray(mesh.node_coords)
    x, y = coords[:, 0], coords[:, 1]
    wall = height * np.exp(-(((x - crest) / width) ** 2))
    warped = np.stack([x, wall + y * (H - wall) / H], axis=1)
    return eqx.tree_at(lambda m: m.node_coords, mesh, jnp.asarray(warped))


def _straight_mesh():
    return structured_grid_2d(
        28, 20, lx=4.0, ly=H, named_boundaries=True, y_nodes=graded_nodes(20, H, 1.2)
    )


CASES = {
    # name: (mesh builder, kinematic viscosity)
    "bump": (lambda: _bump_mesh(0.10), 1e-4),  # Re_H = 1e4
    "bump-high": (lambda: _bump_mesh(0.30), 1e-4),
    "channel": (_straight_mesh, 4e-4),  # the integration test's Re_H = 2500
}


def _coupled(name: str) -> CoupledRANS:
    mesh_builder, nu = CASES[name]
    mesh = mesh_builder()
    geometry = mesh.geometry()
    model = SSTModel()
    k_in = float(inlet_k(jnp.array(U_IN), INTENSITY))
    omega_in = float(inlet_omega(jnp.array(k_in), 0.07 * H, model))
    properties = PropertyModel(
        {"viscosity": Constant(jnp.asarray(RHO * nu)), "density": Constant(jnp.asarray(RHO))}
    )
    walls = {"bottom": NoSlipWall(), "top": NoSlipWall()}
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        properties,
        BoundaryConditions(
            {"left": VelocityInlet(velocity=(U_IN, 0.0)), "right": PressureOutlet(pressure=0.0)}
            | walls
        ),
        gradient_scheme=CompactGreenGauss(),
        advection_scheme=FirstOrderUpwind(),
    )
    turbulence = SSTTurbulence.build(
        model,
        mesh,
        geometry,
        FirstOrderUpwind(),
        properties,
        gradient_scheme=CompactGreenGauss(),
        wall_patches=["bottom", "top"],
        k_boundary=BoundaryConditions(
            {
                "left": Dirichlet(k_in),
                "right": ZeroGradient(),
                "bottom": Dirichlet(0.0),
                "top": Dirichlet(0.0),
            }
        ),
        omega_boundary=BoundaryConditions(
            {
                "left": Dirichlet(omega_in),
                "right": ZeroGradient(),
                "bottom": ZeroGradient(),
                "top": ZeroGradient(),
            }
        ),
    )
    return CoupledRANS.build(momentum, turbulence)


def _separated_length(coupled: CoupledRANS, flow: jnp.ndarray) -> float:
    """Streamwise extent of reversed flow in the lower-wall-adjacent cells (0 when attached)."""
    momentum = coupled.momentum
    mesh = momentum.mesh
    velocity, _ = momentum.unpack(flow)
    faces = np.asarray(mesh.face_patches.indices("bottom"))
    cells = np.unique(np.asarray(mesh.face_cells.owner)[faces])
    centroids = np.asarray(momentum.geometry.cell.centroid)[cells]
    order = np.argsort(centroids[:, 0])
    x = centroids[order, 0]
    u = np.asarray(velocity)[cells][order, 0]
    widths = np.gradient(x)
    return float(np.sum(widths[u < 0.0]))


def _plug_versus_reference_flux(coupled: CoupledRANS) -> float:
    """Relative L2 difference between the two frozen mass fluxes at the hybrid start."""
    from aquaflux.flow.block_preconditioner import _characteristic_reference_state
    from aquaflux.initialization import hybrid_initialize

    momentum = coupled.momentum
    flow, _, _ = hybrid_initialize(coupled)
    plug = momentum.mass_flux(_characteristic_reference_state(momentum))
    reference = momentum.mass_flux(flow)
    return float(jnp.linalg.norm(reference - plug) / jnp.linalg.norm(plug))


def run(name: str, arm: str) -> dict:
    _select_arm(arm)
    coupled = _coupled(name)
    reports = []
    started = time.monotonic()

    def on_step(report):
        reports.append(report)
        print(
            f"  {name}/{arm} step {len(reports):3d}  |R| {float(report.residual_norm):.4e}"
            f"  cycles {int(report.restart_cycles):3d}  alpha {float(report.alpha):.3f}"
            f"  t {time.monotonic() - started:7.1f}s",
            flush=True,
        )

    try:
        flow, k, omega = solve_coupled(
            coupled, preconditioner=BlockDiagonal(), max_steps=MAX_STEPS, on_step=on_step
        )
    except Exception as error:  # a refused build or an unconverged march is this arm's result
        print(f"  {name}/{arm} FAILED: {type(error).__name__}: {str(error)[:200]}", flush=True)
        return {
            "frozen at": _FROZEN.get("used"),
            "failed": type(error).__name__,
            "steps": len(reports),
            "cycles": sum(int(r.restart_cycles) for r in reports),
            "wall": time.monotonic() - started,
        }
    wall = time.monotonic() - started
    return {
        "frozen at": _FROZEN.get("used"),
        "steps": len(reports),
        "cycles": sum(int(r.restart_cycles) for r in reports),
        "wall": wall,
        "final |R|": float(reports[-1].residual_norm),
        "separated length": _separated_length(coupled, flow),
        "fields": np.concatenate([np.asarray(flow), np.asarray(k), np.asarray(omega)]),
    }


def main() -> None:
    cases = os.environ.get("VBR_CASES", "bump,bump-high,channel").split(",")
    arms = os.environ.get("VBR_ARMS", "plug,reference").split(",")
    for name in cases:
        coupled = _coupled(name)
        print(
            f"== {name}: {coupled.momentum.mesh.n_cells} cells, frozen mass flux plug vs hybrid "
            f"start: rel L2 {_plug_versus_reference_flux(coupled):.3f}",
            flush=True,
        )
        results = {arm: run(name, arm) for arm in arms}
        print(f"== {name} summary", flush=True)
        for arm, r in results.items():
            if "failed" in r:
                print(
                    f"  {arm:10s} frozen at {r['frozen at']}  FAILED ({r['failed']}) after "
                    f"{r['steps']} steps, {r['cycles']} cycles, {r['wall']:.1f}s",
                    flush=True,
                )
                continue
            print(
                f"  {arm:10s} frozen at {r['frozen at']:>10s}  steps {r['steps']:3d}  "
                f"cycles {r['cycles']:4d}  wall {r['wall']:7.1f}s  final |R| {r['final |R|']:.3e}  "
                f"separated length {r['separated length']:.4f}",
                flush=True,
            )
        if len(results) == 2 and all("fields" in r for r in results.values()):
            a, b = (r["fields"] for r in results.values())
            print(
                f"  roots agree to rel {np.linalg.norm(a - b) / np.linalg.norm(a):.2e}", flush=True
            )


if __name__ == "__main__":
    main()
