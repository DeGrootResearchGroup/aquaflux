"""Does the ILU(0)-vs-ILU(1) gap on pitzDaily need pitzDaily, or only pitzDaily's SIZE?

An incomplete factorization is not a scalable preconditioner: its iteration count grows with the
mesh, and the growth is steeper the less fill it carries.  So a gap between ILU(0) and ILU(1) that
is 1.6x on a few hundred cells and 4x on twelve thousand is the ordinary behaviour of the method,
not a property of the case.  This sweeps a PERFECTLY ORTHOGONAL uniform channel from a few hundred
cells to pitzDaily's twelve thousand with every other variable pinned, so the size axis is read
alone.

Re is swept in a second arm for the same reason: pitzDaily runs at Re 25000 and the channel at
2500, and convection dominance is the other thing that makes an incomplete factorization weak.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ilu_fill_probe as P  # noqa: E402
from aquaflux.turbulence.coupled import (  # noqa: E402
    _coupled_shift_policy,
    _frozen_shift_diagonal,
)
from ilu_fill_probe import (  # noqa: E402
    assemble,
    ilu_pivots,
    ksp_solve,
    materialize,
    seed_state,
    skew_metrics,
)


def run(label, coupled, betas, nu):
    n_fields = coupled.layout.dim + 3
    n = coupled.layout.n_cells
    ratio, interior = skew_metrics(coupled)
    ar = P.cell_aspect_ratio(coupled)
    print(f"\n-- {label}: {n} cells, {n_fields * n} dofs, nu {nu:.1e}, "
          f"Re {P.U_IN * P.H / nu:.0f}, skew max {ratio[interior].max():.1e}, "
          f"AR median {np.median(ar):.2f} max {ar.max():.1f}", flush=True)
    state, residual = seed_state(coupled)
    rhs = -np.asarray(residual, dtype=np.float64)
    jacobian = materialize(coupled, state, 3)
    base = _coupled_shift_policy(coupled, state, "twolevel")
    for beta in betas:
        shift = (_frozen_shift_diagonal(base, beta, state) if beta > 0
                 else np.zeros(n_fields * n))
        cell_major, scaling, perm = assemble(jacobian, np.asarray(shift), n_fields)
        rhs_eq = (np.asarray(scaling) * rhs)[perm]
        out = []
        for levels, arm in ((0, "ilu0"), (1, "ilu1")):
            census = ilu_pivots(cell_major, n_fields, levels)
            r = ksp_solve(cell_major, rhs_eq, n_fields, arm, max_it=2000)
            true = np.linalg.norm(cell_major @ r["x"] - rhs_eq) / np.linalg.norm(rhs_eq)
            out.append(f"{arm} its {r['its']:>5} rel {true:.1e} neg {census.get('negative', -1):>4}")
        print(f"   beta {beta:<5} | " + " | ".join(out), flush=True)
        del cell_major
        gc.collect()
    del jacobian
    gc.collect()


BETAS = (0.05, 0.5)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "size"
    if which == "size":
        print("SIZE SWEEP -- orthogonal uniform channel, Re 2500, skew 0, everything else pinned")
        for nx, ny in ((24, 16), (48, 32), (72, 48), (96, 64), (144, 96)):
            coupled = P.build_grid(0.0, dim=2, nx=nx, ny=ny)
            run(f"{nx}x{ny}", coupled, BETAS, P.NU)
            del coupled
            gc.collect()
    else:
        print("REYNOLDS SWEEP -- orthogonal uniform 96x64 channel, size pinned")
        for nu in (4e-3, 4e-4, 4e-5):
            original = P.NU
            P.NU = nu
            try:
                coupled = P.build_grid(0.0, dim=2, nx=96, ny=64)
                run(f"nu {nu:.0e}", coupled, BETAS, nu)
            finally:
                P.NU = original
            del coupled
            gc.collect()
