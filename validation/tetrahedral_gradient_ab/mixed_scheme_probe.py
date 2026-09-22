"""Issue #435: is the multiple-correction failure in the VELOCITY reconstruction or in k/omega's?

Converges the Re/10 anchor with corrected Green--Gauss, then, at that smooth state, assembles the coupled
case with the multiple-correction scheme on one block and corrected Green--Gauss on the other. For each
arm: the coupled residual there, the reconstructed strain rate over omega off the wall-fixation cells, how
many non-fixation ``omega`` rows have a non-positive diagonal (and which closure field makes them so),
and, optionally, a short march from that state.

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/mixed_scheme_probe.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import equinox as eqx
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
from aquaflux.turbulence import CoupledRANS, SSTModel, inlet_k, inlet_omega
from compare import INTENSITY, LENGTH_SCALE, POLYMESH, U_IN, build_case
from diffusion_operator_probe import FirstPassOnCells
from seed_state_probe import CAP, rings_from_wall
from warm_start_probe import RATIO, SEED_STEPS, march

MARCH_STEPS = int(os.environ.get("TET_STEPS", "0"))
CLOSURE_FIELDS = ("nu_t", "f1", "strain_rate", "grad_k", "grad_omega", "omega", "k")


def mixed(velocity_scheme, turbulence_scheme) -> CoupledRANS:
    return CoupledRANS.build(
        build_case(velocity_scheme).momentum, build_case(turbulence_scheme).turbulence
    ).with_scaled_molecular_viscosity(RATIO)


def main() -> None:
    mesh = read_openfoam(POLYMESH)
    geometry = mesh.geometry()
    n = mesh.n_cells
    seed_case = build_case(CorrectedGreenGauss()).with_scaled_molecular_viscosity(RATIO)
    flow0 = seed_case.momentum.pack(jnp.zeros((n, 3)).at[:, 0].set(U_IN), jnp.zeros(n))
    k_in = float(inlet_k(jnp.array(U_IN), INTENSITY))
    omega_in = float(inlet_omega(jnp.array(k_in), LENGTH_SCALE, SSTModel()))
    seed = march(
        "seed", seed_case, flow0, jnp.full(n, k_in), jnp.full(n, omega_in), 1e-4, SEED_STEPS
    )
    if seed is None:
        return
    flow, k, omega = seed
    multcorr = MultipleCorrectionGradient(
        boundary_closure=OwnerGradient(), fallback=SkewCorrectedGradient()
    ).bind(mesh, geometry)
    corrected = CorrectedGreenGauss()
    wall_rings = rings_from_wall(mesh, build_case(corrected).turbulence.wall_cells)
    ring0 = jnp.asarray(np.flatnonzero(wall_rings == 0))
    ring01 = jnp.asarray(np.flatnonzero(wall_rings <= 1))
    arms = {
        "all corrected GG": (corrected, corrected),
        "multcorr on velocity only": (multcorr, corrected),
        "multcorr on k/omega only": (corrected, multcorr),
        "multcorr on both": (multcorr, multcorr),
        "both, 1st pass ring 0": (
            FirstPassOnCells(multcorr, ring0),
            FirstPassOnCells(multcorr, ring0),
        ),
        "both, 1st pass rings 0-1": (
            FirstPassOnCells(multcorr, ring01),
            FirstPassOnCells(multcorr, ring01),
        ),
    }
    selected = os.environ.get("TET_ARMS")
    if selected:
        arms = {name: arms[name] for name in selected.split(";")}
    rings = None
    for label, (velocity_scheme, turbulence_scheme) in arms.items():
        coupled = mixed(velocity_scheme, turbulence_scheme)
        momentum, turbulence = coupled.momentum, coupled.turbulence
        if rings is None:
            rings = rings_from_wall(mesh, turbulence.wall_cells)
        free = rings > 0
        velocity_fields = momentum.velocity_fields(flow)
        closure = turbulence.closure_fields(velocity_fields, k, omega)
        ratio = np.asarray(closure.strain_rate / omega)
        state = coupled.state_from_physical(flow, k, omega)

        def r_omega(w, coupled=coupled, state=state):
            return coupled.residual(state.at[5 * n :].set(w))[5 * n :]

        diag = np.diag(np.asarray(jax.jacfwd(r_omega)(state[5 * n :])))
        bad = free & (diag <= 0)
        residual = float(jnp.linalg.norm(coupled.residual(state)))
        print(
            f"[{label:26s}] |R| {residual:.3e}; S/omega max {ratio[free].max():.2e}, cap binds "
            f"{(free & (ratio > CAP)).sum()}; omega diag<=0 {bad.sum()}"
            + (
                f" at rings {np.bincount(rings[bad], minlength=4)[1:].tolist()}"
                if bad.any()
                else ""
            ),
            flush=True,
        )
        if bad.any():
            mdot = coupled.effective_momentum(flow, k, omega)[1].flow_fields(flow).mdot
            row = coupled.omega_transform.fixation_row()

            def diag_with(
                live_names,
                turbulence=turbulence,
                mdot=mdot,
                row=row,
                vf=velocity_fields,
                frozen=closure,
            ):
                def r(w):
                    live = turbulence.closure_fields(vf, k, w)
                    mix = frozen
                    for name in live_names:
                        mix = eqx.tree_at(
                            lambda c, name=name: getattr(c, name), mix, getattr(live, name)
                        )
                    return turbulence.omega_residual(mdot, mix, row)(w)

                return np.diag(np.asarray(jax.jacfwd(r)(omega)))

            base = diag_with(())
            parts = []
            for name in CLOSURE_FIELDS:
                delta = (diag_with((name,)) - base)[bad]
                parts.append(f"{name} {np.median(delta):+.1e}")
            print(
                f"    frozen-closure diag median on those cells {np.median(base[bad]):+.2e}; "
                f"live-field deltas (median): " + ", ".join(parts),
                flush=True,
            )
        if MARCH_STEPS:
            march(label, coupled, flow, k, omega, 1e-3, MARCH_STEPS)


if __name__ == "__main__":
    main()
