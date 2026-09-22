"""Issue #435: at a SMOOTH developed state, does the multiple-correction scheme still invent strain?

Converges the Re/10 anchor with corrected Green--Gauss (as ``warm_start_probe.py`` does), then evaluates
the closure and the coupled ``omega`` Jacobian diagonal at that state under each gradient scheme: the
reconstructed strain rate over omega, how many non-fixation cells the omega-production cap binds in, how
many have a non-positive ``omega`` diagonal, and how far those sit from a wall (in wall-cell rings).

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/seed_state_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.io import read_openfoam
from aquaflux.schemes import (
    CorrectedGreenGauss,
    MultipleCorrectionGradient,
    OwnerGradient,
    SkewCorrectedGradient,
)
from aquaflux.turbulence import SSTModel, inlet_k, inlet_omega
from compare import INTENSITY, LENGTH_SCALE, POLYMESH, U_IN, build_case
from diffusion_operator_probe import FirstPassOnly
from warm_start_probe import RATIO, SEED_STEPS, march

#: sqrt(10 beta*): the omega-production cap binds where S / omega exceeds it.
CAP = float(np.sqrt(10.0 * 0.09))


def rings_from_wall(mesh, wall_cells) -> np.ndarray:
    """Graph distance (face hops) of each cell from the nearest wall-owning cell."""
    face_cells = mesh.face_cells
    inner = np.asarray(face_cells.interior)
    owner, neighbour = np.asarray(face_cells.owner)[inner], np.asarray(face_cells.neighbour)[inner]
    dist = np.full(mesh.n_cells, -1)
    dist[np.asarray(wall_cells)] = 0
    front = 0
    while (dist < 0).any():
        now = dist == front
        grow = np.zeros(mesh.n_cells, dtype=bool)
        grow[neighbour[now[owner]]] = True
        grow[owner[now[neighbour]]] = True
        new = grow & (dist < 0)
        if not new.any():
            break
        dist[new] = front + 1
        front += 1
    return dist


def main() -> None:
    mesh = read_openfoam(POLYMESH)
    geometry = mesh.geometry()
    seed_case = build_case(CorrectedGreenGauss()).with_scaled_molecular_viscosity(RATIO)
    n = mesh.n_cells
    flow = seed_case.momentum.pack(jnp.zeros((n, 3)).at[:, 0].set(U_IN), jnp.zeros(n))
    k_in = float(inlet_k(jnp.array(U_IN), INTENSITY))
    omega_in = float(inlet_omega(jnp.array(k_in), LENGTH_SCALE, SSTModel()))
    seed = march(
        "seed", seed_case, flow, jnp.full(n, k_in), jnp.full(n, omega_in), 1e-4, SEED_STEPS
    )
    if seed is None:
        return
    flow, k, omega = seed
    owner = MultipleCorrectionGradient(boundary_closure=OwnerGradient(), fallback=None).bind(
        mesh, geometry
    )
    arms = {
        "corrected GG": CorrectedGreenGauss(),
        "multcorr first pass": FirstPassOnly(owner),
        "multcorr owner": owner,
        "multcorr repaired": MultipleCorrectionGradient(
            boundary_closure=OwnerGradient(), fallback=SkewCorrectedGradient()
        ).bind(mesh, geometry),
    }
    rings = None
    for label, scheme in arms.items():
        coupled = build_case(scheme).with_scaled_molecular_viscosity(RATIO)
        turbulence = coupled.turbulence
        if rings is None:
            rings = rings_from_wall(mesh, turbulence.wall_cells)
        closure = turbulence.closure_fields(coupled.momentum.velocity_fields(flow), k, omega)
        ratio = np.asarray(closure.strain_rate / omega)
        state = coupled.state_from_physical(flow, k, omega)

        def r_omega(w, coupled=coupled, state=state):
            return coupled.residual(state.at[5 * n :].set(w))[5 * n :]

        diag = np.diag(np.asarray(jax.jacfwd(r_omega)(state[5 * n :])))
        free = rings > 0
        bad = free & (diag <= 0)
        residual = float(jnp.linalg.norm(coupled.residual(state)))
        print(
            f"[{label:20s}] |R| at the seed {residual:.3e}; non-fixed cells: S/omega median "
            f"{np.median(ratio[free]):.2e} max {ratio[free].max():.2e}; cap binds in "
            f"{(free & (ratio > CAP)).sum()}; omega diag<=0 in {bad.sum()}"
            + (
                f" at wall rings {np.bincount(rings[bad], minlength=4)[1:].tolist()}"
                if bad.any()
                else ""
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
