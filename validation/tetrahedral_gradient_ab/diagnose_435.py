"""Issue #435 diagnostics: why the coupled march cannot start on the tetrahedral duct.

Probe 1 -- reconstruction exactness on this mesh: gradient of linear and quadratic fields, exact
boundary values, both closures. Probe 2 -- where the residual at a uniform-plug state lives, per block,
and the cell that dominates it.

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/diagnose_435.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax.numpy as jnp
import numpy as np
from aquaflux.schemes import MultipleCorrectionGradient, OwnerGradient, SkewCorrectedGradient
from aquaflux.turbulence import SSTModel, inlet_k, inlet_omega
from compare import INTENSITY, LENGTH_SCALE, U_IN, build_case

ARMS = {
    "owner": lambda: MultipleCorrectionGradient(boundary_closure=OwnerGradient(), fallback=None),
    "repaired": lambda: MultipleCorrectionGradient(
        boundary_closure=OwnerGradient(), fallback=SkewCorrectedGradient()
    ),
}


def boundary_face_count(mesh) -> np.ndarray:
    owner = np.asarray(mesh.face_cells.owner)
    interior = np.asarray(mesh.face_cells.interior)
    return np.bincount(owner[~interior], minlength=mesh.n_cells)


def probe_exactness(coupled, label: str) -> None:
    momentum = coupled.momentum
    mesh, geometry, scheme = momentum.mesh, momentum.geometry, momentum.gradient_scheme
    x = np.asarray(geometry.cell.centroid)
    xf = np.asarray(geometry.face.centroid)
    origin = x.mean(axis=0)
    xc, xfc = x - origin, xf - origin
    nb = boundary_face_count(mesh)
    fields = {
        "linear": (
            lambda p: 3.0 * p[:, 0] - 2.0 * p[:, 1] + 5.0 * p[:, 2],
            lambda p: np.tile([3.0, -2.0, 5.0], (p.shape[0], 1)),
        ),
        "quadratic": (
            lambda p: p[:, 0] ** 2 + 2 * p[:, 1] * p[:, 2] - p[:, 2] ** 2,
            lambda p: np.stack([2 * p[:, 0], 2 * p[:, 2], 2 * p[:, 1] - 2 * p[:, 2]], 1),
        ),
    }
    for name, (f, g) in fields.items():
        grad = np.asarray(scheme.gradients(jnp.asarray(f(xc)), mesh, geometry, jnp.asarray(f(xfc))))
        exact = g(xc)
        scale = max(np.abs(exact).max(), 1e-300)
        err = np.linalg.norm(grad - exact, axis=1) / scale
        worst = int(np.nanargmax(err))
        print(
            f"  [{label}] {name:<9} max err {np.nanmax(err):.3e} (cell {worst}, "
            f"{nb[worst]} bnd faces)  by bnd faces: "
            + "  ".join(f"{c}:{np.nanmax(err[nb == c]):.2e}" for c in np.unique(nb)),
            flush=True,
        )


def probe_plug_residual(coupled, label: str) -> None:
    momentum = coupled.momentum
    mesh = momentum.mesh
    n, dim = mesh.n_cells, mesh.dim
    velocity = jnp.zeros((n, dim)).at[:, 0].set(U_IN)
    flow = momentum.pack(velocity, jnp.zeros(n))
    k_in = float(inlet_k(jnp.array(U_IN), INTENSITY))
    omega_in = float(inlet_omega(jnp.array(k_in), LENGTH_SCALE, SSTModel()))
    k = jnp.full(n, k_in)
    omega = jnp.full(n, omega_in)
    state = coupled.state_from_physical(flow, k, omega)
    residual = np.asarray(coupled.residual(state))
    nb = boundary_face_count(mesh)
    names = [f"u{i}" for i in range(dim)] + ["p", "k", "omega"]
    blocks = residual.reshape(dim + 3, n)
    print(f"  [{label}] plug |R| = {np.linalg.norm(residual):.4e}", flush=True)
    for name, block in zip(names, blocks, strict=True):
        worst = int(np.nanargmax(np.abs(block)))
        finite = np.isfinite(block).all()
        print(
            f"    {name:<6} |R| {np.linalg.norm(block):.3e}  max {np.abs(block).max():.3e} at cell "
            f"{worst} ({nb[worst]} bnd faces)  finite={finite}",
            flush=True,
        )

    closure, _ = coupled.effective_momentum(flow, k, omega)
    for attr in ("nu_t", "strain_rate", "production"):
        value = getattr(closure, attr, None)
        if value is not None:
            arr = np.asarray(value)
            print(
                f"    closure.{attr}: max {np.nanmax(np.abs(arr)):.3e} at cell "
                f"{int(np.nanargmax(np.abs(arr)))}",
                flush=True,
            )
    grad_u = (
        np.asarray(momentum.velocity_fields(flow).gradient)
        if hasattr(momentum.velocity_fields(flow), "gradient")
        else None
    )
    if grad_u is not None:
        mag = np.linalg.norm(grad_u.reshape(n, -1), axis=1)
        w = int(np.nanargmax(mag))
        print(f"    |grad u| max {mag[w]:.3e} at cell {w} ({nb[w]} bnd faces)", flush=True)


def main() -> None:
    for label, make in ARMS.items():
        coupled = build_case(make())
        print(f"=== {label} ===", flush=True)
        probe_exactness(coupled, label)
        probe_plug_residual(coupled, label)


if __name__ == "__main__":
    main()
