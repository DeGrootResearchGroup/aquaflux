"""Issue #435: the coupled RANS Jacobian at a uniform-plug state, across gradient schemes.

For each scheme and each probe reach in ``TET_REACHES`` (default 3, 5, 7): whether the coloured probe reproduces the true Jacobian (per column field),
how well an exact LU of the probed matrix solves the Newton system against the true operator, how many
non-fixation ``omega`` rows have a non-positive diagonal, and where the smallest-singular direction lives.

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/coupled_operator_probe.py
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
import scipy.sparse.linalg as spla
from aquaflux.io import read_openfoam
from aquaflux.schemes import (
    CorrectedGreenGauss,
    MultipleCorrectionGradient,
    OwnerGradient,
    SkewCorrectedGradient,
)
from aquaflux.solve import materialize_block_jacobian
from aquaflux.turbulence import SSTModel, inlet_k, inlet_omega
from aquaflux.turbulence.coupled import _coupled_jacobian_plan
from compare import INTENSITY, LENGTH_SCALE, POLYMESH, U_IN, build_case
from diffusion_operator_probe import FirstPassOnCells, FirstPassOnly, FirstPassWhereIllConditioned
from seed_state_probe import rings_from_wall

LIMIT = float(os.environ.get("TET_M2_LIMIT", "10"))
REACHES = tuple(int(r) for r in os.environ.get("TET_REACHES", "3,5,7").split(","))
NAMES = ("u", "v", "w", "p", "k", "omega")


def schemes(mesh, geometry):
    repaired = MultipleCorrectionGradient(
        boundary_closure=OwnerGradient(), fallback=SkewCorrectedGradient()
    ).bind(mesh, geometry)
    rings = rings_from_wall(mesh, build_case(CorrectedGreenGauss()).turbulence.wall_cells)
    through_ring_2 = jnp.asarray(np.flatnonzero(rings <= 2))
    return {
        "corrected GG": CorrectedGreenGauss(),
        "multcorr first pass": FirstPassOnly(repaired),
        "multcorr repaired": repaired,
        f"repaired, 1st pass M2>{LIMIT:g}": FirstPassWhereIllConditioned(repaired, LIMIT),
        "hessian off rings<=2": FirstPassOnCells(repaired, through_ring_2),
    }


def main() -> None:
    mesh = read_openfoam(POLYMESH)
    geometry = mesh.geometry()
    for label, scheme in schemes(mesh, geometry).items():
        coupled = build_case(scheme).with_scaled_molecular_viscosity(10.0)
        momentum, turbulence = coupled.momentum, coupled.turbulence
        n = momentum.mesh.n_cells
        flow = momentum.pack(jnp.zeros((n, 3)).at[:, 0].set(U_IN), jnp.zeros(n))
        k_in = float(inlet_k(jnp.array(U_IN), INTENSITY))
        omega_in = float(inlet_omega(jnp.array(k_in), LENGTH_SCALE, SSTModel()))
        state = coupled.state_from_physical(flow, jnp.full(n, k_in), jnp.full(n, omega_in))
        residual = np.asarray(eqx.filter_jit(lambda c, s: c.residual(s))(coupled, state))
        _, linear = jax.linearize(coupled.residual, state)
        matvec = eqx.filter_jit(linear)
        for reach in REACHES:
            jac = materialize_block_jacobian(
                matvec,
                _coupled_jacobian_plan(coupled, reach),
                batched_matvec=eqx.filter_jit(jax.vmap(linear)),
                probe_batch_size=32,
            ).tocsc()
            rng = np.random.default_rng(0)
            errors = []
            for f in range(len(NAMES)):
                v = np.zeros(residual.size)
                v[f * n : (f + 1) * n] = rng.standard_normal(n)
                reference = np.asarray(matvec(jnp.asarray(v)))
                errors.append(np.linalg.norm(jac @ v - reference) / np.linalg.norm(reference))
            lu = spla.splu(jac)
            x = lu.solve(-residual)
            true = np.linalg.norm(np.asarray(matvec(jnp.asarray(x))) + residual) / np.linalg.norm(
                residual
            )
            vec = np.random.default_rng(1).standard_normal(residual.size)
            for _ in range(30):
                vec = lu.solve(vec)
                vec /= np.linalg.norm(vec)
            share = (vec.reshape(len(NAMES), n) ** 2).sum(axis=1)
            fixed = np.zeros(n, dtype=bool)
            fixed[np.asarray(turbulence.wall_cells)] = True
            omega_diag = jac.diagonal()[5 * n :]
            print(
                f"[{label}, reach {reach}] |R| {np.linalg.norm(residual):.3e}  probe col err max {max(errors):.1e}  "
                f"LU Newton solve: true rel resid {true:.2e}, |x| {np.linalg.norm(x):.2e}  "
                f"omega diag<=0 {(omega_diag[~fixed] <= 0).sum()}/{(~fixed).sum()}  "
                f"smallest-singular |Jv| {np.linalg.norm(jac @ vec):.2e} (|J|max {abs(jac).max():.2e}) shares "
                + " ".join(f"{a}:{b:.0e}" for a, b in zip(NAMES, share, strict=True)),
                flush=True,
            )


if __name__ == "__main__":
    main()
