"""Does the coloured probe at the case's reach recover the coupled Jacobian exactly, block by block?

The case probes its Jacobian at ``stencil_reach: 3``, on the strength of the multiple-correction
reconstruction's own residual reaching exactly distance three. That was measured on a scalar Laplace
residual. The coupled residual also feeds the reconstructed velocity gradient into the eddy viscosity and
the face viscosity, which can spend further rings, and a probe shorter than the residual's stencil folds
the far coupling onto near entries rather than dropping it.

This materializes the residual's Jacobian at each reach in ``PROBE_CHECK_REACHES`` (default ``3,4``) and
reports, per (row field, column field) block, the relative error of the materialized matrix against the
exact Jacobian-vector product on a random vector supported on that column field. A faithful probe returns
the round-off floor in every block.

Usage
-----
    validation/run_case.sh validation/pitzdaily_openfoam/probe_reach_check.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import compare  # noqa: E402
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    PROBE_BATCH_SIZE,
    JacobianProbe,
    batched_jacobian_matvec,
    jacobian_matvec,
)
from aquaflux.solve.sparse_jacobian import materialize_block_jacobian  # noqa: E402
from local_jacobian_probe import reference_state  # noqa: E402

REACHES = tuple(int(r) for r in os.environ.get("PROBE_CHECK_REACHES", "3,4").split(","))
FIELDS = ("u", "v", "p", "k", "omega")


def main():
    print(f"[configuration] reaches {REACHES}, jax {jax.__version__}, {jax.default_backend()}")
    coupled = compare.build_case()["coupled"]
    mesh = coupled.momentum.mesh
    n, nf = coupled.layout.n_cells, coupled.layout.n_fields
    print(f"  cells {n}, fields {nf}, gradient {type(coupled.momentum.gradient_scheme).__name__}")
    state = reference_state(coupled)
    r = np.asarray(coupled.residual(state))
    if not np.all(np.isfinite(r)):
        raise SystemExit("non-finite residual at the reference state")
    print(f"  state: OpenFOAM time-accurate field, |R| {np.linalg.norm(r):.4e}", flush=True)

    matvec = eqx.filter_jit(lambda v: jacobian_matvec(coupled, state, v))
    batched = eqx.filter_jit(lambda vs: batched_jacobian_matvec(coupled, state, vs))
    rng = np.random.default_rng(0)
    for reach in REACHES:
        probe = JacobianProbe.build(mesh.face_cells, n, nf, reach)
        jac = materialize_block_jacobian(
            matvec,
            probe.plan,
            batched_matvec=batched,
            probe_batch_size=PROBE_BATCH_SIZE,
            structure=probe.structure,
        )
        print(
            f"\n[reach {reach}] {probe.plan.n_probes} probes; block error (rows down, columns across)"
        )
        print("        " + "".join(f"{f:>10s}" for f in FIELDS))
        errors = np.zeros((nf, nf))
        for b in range(nf):
            v = np.zeros(nf * n)
            v[b * n : (b + 1) * n] = rng.standard_normal(n)
            exact = np.asarray(matvec(jnp.asarray(v)))
            got = jac @ v
            for a in range(nf):
                rows = slice(a * n, (a + 1) * n)
                scale = np.linalg.norm(exact[rows])
                errors[a, b] = np.linalg.norm(got[rows] - exact[rows]) / scale if scale else 0.0
        for a in range(nf):
            print(f"  {FIELDS[a]:>6s}" + "".join(f"{errors[a, b]:10.1e}" for b in range(nf)))
        print(f"  worst block {errors.max():.1e}", flush=True)


if __name__ == "__main__":
    main()
