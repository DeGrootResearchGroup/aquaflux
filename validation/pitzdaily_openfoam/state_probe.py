"""Is the ILU fill ranking a property of the CASE, or of the STATE the operator is taken at?

Every recorded pitzDaily fill measurement is at a COLD start (steps 3-4 of a march from the hybrid
initial condition); every recorded bfs3d one is at a DEVELOPED, converged state at the target
Reynolds number after a continuation.  Those are two different operators, and nothing so far
separates "the two cases differ" from "cold and developed states differ".

This runs the identical measurement on pitzDaily at BOTH: its own hybrid cold seed, and the
OpenFOAM time-accurate converged field on the same mesh, cell for cell.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ilu_fill_probe as P  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from aquaflux.turbulence.coupled import (  # noqa: E402
    _coupled_shift_policy,
    _frozen_shift_diagonal,
)
from ilu_fill_probe import assemble, ilu_pivots, ksp_solve, materialize  # noqa: E402


def openfoam_state(coupled):
    """The time-accurate OpenFOAM field on the same mesh, mapped into the solved variables.

    Not a root of aquaflux's own equations -- a different discretization and wall treatment -- but
    a genuinely DEVELOPED field with the recirculation formed, which is what the bfs3d checkpoints
    are and the hybrid seed is not.
    """
    import compare

    of = compare.read_openfoam_reference()
    momentum = coupled.momentum
    centroid = np.asarray(momentum.geometry.cell.centroid)
    from scipy.spatial import cKDTree

    dist, idx = cKDTree(of["centroid"]).query(centroid)
    assert float(dist.max()) < 1e-6, f"mesh mismatch, max centroid distance {dist.max()}"
    velocity = jnp.asarray(of["U"][idx])
    pressure = jnp.asarray(of["p"][idx])
    k = jnp.asarray(np.maximum(of["k"][idx], 1e-10))
    omega = jnp.asarray(np.maximum(of["omega"][idx], 1e-10))
    flow = momentum.pack(velocity, pressure)
    return coupled.state_from_physical(flow, k, omega)


def run(label, coupled, state, betas, reach=3):
    n_fields = coupled.layout.dim + 3
    n = coupled.layout.n_cells
    residual = coupled.residual(state)
    assert bool(jnp.all(jnp.isfinite(residual))), f"{label}: residual is NOT finite"
    rhs = -np.asarray(residual, dtype=np.float64)
    print(f"\n{'=' * 90}\n{label}: |R| {np.linalg.norm(rhs):.4e}", flush=True)
    jacobian = materialize(coupled, state, reach)
    base = _coupled_shift_policy(coupled, state, "twolevel")
    for beta in betas:
        shift = (_frozen_shift_diagonal(base, beta, state) if beta > 0
                 else np.zeros(n_fields * n))
        cell_major, scaling, perm = assemble(jacobian, np.asarray(shift), n_fields)
        rhs_eq = (np.asarray(scaling) * rhs)[perm]
        out = []
        for levels, arm in ((0, "ilu0"), (1, "ilu1")):
            census = ilu_pivots(cell_major, n_fields, levels)
            r = ksp_solve(cell_major, rhs_eq, n_fields, arm, max_it=600)
            true = np.linalg.norm(cell_major @ r["x"] - rhs_eq) / np.linalg.norm(rhs_eq)
            out.append(f"{arm} its {r['its']:>4} rel {true:.1e} neg {census.get('negative', -1):>4} "
                       f"min|p| {census.get('min', float('nan')):.1e}")
        print(f"  beta {beta:<6} | " + " | ".join(out), flush=True)
        del cell_major
        gc.collect()
    del jacobian
    gc.collect()


BETAS = (0.0, 0.02, 0.05, 0.1, 0.5, 2.0)

if __name__ == "__main__":
    coupled = P.build_pitz("corrected")
    cold, _ = P.seed_state(coupled)
    run("pitzDaily COLD (hybrid seed)", coupled, cold, BETAS)
    developed = openfoam_state(coupled)
    run("pitzDaily DEVELOPED (OpenFOAM transient field)", coupled, developed, BETAS)
