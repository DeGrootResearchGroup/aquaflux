"""WHERE does the zero-fill solve stall -- which field, which cells -- and how much fill closes it.

At the developed state both factorizations have healthy pivots, so the pivot census is not the
mechanism.  This reports the fill ladder (levels 0..3) and, for each arm, the TRUE residual
resolved per field block and the spatial distribution of the worst rows.
"""
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ilu_fill_probe as P
import numpy as np
from aquaflux.turbulence.coupled import _coupled_shift_policy, _frozen_shift_diagonal
from ilu_fill_probe import assemble, ilu_pivots, ksp_solve, materialize
from state_probe import openfoam_state

FIELDS = ("u", "v", "p", "k", "omega")
coupled = P.build_pitz("corrected")
nf, n = coupled.layout.dim + 3, coupled.layout.n_cells
skew = P.per_cell_skew(coupled); walls = P.boundary_faces_per_cell(coupled)
ar = P.cell_aspect_ratio(coupled)

for label, state in (("COLD", P.seed_state(coupled)[0]), ("DEVELOPED", openfoam_state(coupled))):
    rhs = -np.asarray(coupled.residual(state), dtype=np.float64)
    J = materialize(coupled, state, 3)
    base = _coupled_shift_policy(coupled, state, "twolevel")
    for beta in (0.05,):
        shift = _frozen_shift_diagonal(base, beta, state)
        A, s, perm = assemble(J, np.asarray(shift), nf)
        b = (np.asarray(s) * rhs)[perm]
        print(f"\n=== {label}, beta {beta} ===", flush=True)
        for levels in (0, 1, 2, 3):
            census = ilu_pivots(A, nf, levels)
            o = ksp_solve(A, b, nf, f"ilu{levels}", max_it=300)
            r = A @ o["x"] - b
            rn, bn = np.linalg.norm(r), np.linalg.norm(b)
            # cell-major: row = cell*nf + field
            field = np.arange(A.shape[0]) % nf
            per = [np.linalg.norm(r[field == f]) / np.linalg.norm(b[field == f]) for f in range(nf)]
            print(f"  ILU({levels}) nnz {census.get('nnz',0)/1e6:5.2f}M neg {census.get('negative',-1):>4} "
                  f"min|p| {census.get('min',float('nan')):.2e} | its {o['its']:>4} TRUE {rn/bn:.2e} | "
                  + " ".join(f"{FIELDS[f]} {per[f]:.1e}" for f in range(nf)), flush=True)
            if levels == 0:
                mag = np.abs(r).reshape(n, nf)
                worst = np.argsort(-mag.max(axis=1))[:200]
                print(f"       worst-200 rows by |r|: skew med {np.median(skew[worst]):.2e} "
                      f"(mesh {np.median(skew):.2e}) | wall faces med {np.median(walls[worst]):.1f} "
                      f"(mesh {np.median(walls):.1f}) | AR med {np.median(ar[worst]):.2f} "
                      f"(mesh {np.median(ar):.2f}) | share AR>mesh p90 "
                      f"{float((ar[worst] > np.quantile(ar, 0.9)).mean()):.2f} (base 0.10)", flush=True)
        del A; gc.collect()
    del J; gc.collect()
