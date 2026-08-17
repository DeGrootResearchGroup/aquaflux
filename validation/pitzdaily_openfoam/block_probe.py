"""Is the zero-fill deficit in the SADDLE, or does it come from k/omega being in the block?

bfs3d ships field_split=True, so its ILU(0) factorizes the 4-field [u,v,w,p] saddle alone.  If
ILU(0) is healthy on pitzDaily's [u,v,p] leading block and only fails on the full 5-field block,
the case split is a configuration difference rather than a case difference.
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

coupled = P.build_pitz("corrected")
nf, n = coupled.layout.dim + 3, coupled.layout.n_cells

def sub(A, b, keep):
    """Symmetric slice of the cell-major operator down to the fields in `keep`."""
    rows = np.concatenate([np.arange(n) * nf + f for f in keep])
    rows.sort()
    S = A[rows][:, rows].tocsr(); S.sort_indices()
    return S, b[rows]

for label, state in (("COLD", P.seed_state(coupled)[0]), ("DEVELOPED", openfoam_state(coupled))):
    rhs = -np.asarray(coupled.residual(state), dtype=np.float64)
    J = materialize(coupled, state, 3)
    base = _coupled_shift_policy(coupled, state, "twolevel")
    for beta in (0.05,):
        shift = _frozen_shift_diagonal(base, beta, state)
        A, s, perm = assemble(J, np.asarray(shift), nf)
        b = (np.asarray(s) * rhs)[perm]
        print(f"\n=== {label}, beta {beta} ===", flush=True)
        for name, keep, bs in (("full [u,v,p,k,w]", (0,1,2,3,4), nf),
                               ("leading [u,v,p]", (0,1,2), 3),
                               ("trailing [k,w]", (3,4), 2)):
            S, bb = sub(A, b, keep) if len(keep) != nf else (A, b)
            out = []
            for lv in (0, 1):
                c = ilu_pivots(S, bs, lv)
                o = ksp_solve(S, bb, bs, f"ilu{lv}", max_it=400)
                t = np.linalg.norm(S @ o["x"] - bb) / np.linalg.norm(bb)
                out.append(f"ilu{lv} its {o['its']:>4} rel {t:.1e} neg {c.get('negative',-1):>4}")
            print(f"  {name:<18} nnz {S.nnz/1e6:5.2f}M | " + " | ".join(out), flush=True)
            if len(keep) != nf:
                del S
            gc.collect()
        del A; gc.collect()
    del J; gc.collect()
