"""Does the CELL ORDERING or the ASPECT RATIO decide whether zero fill is enough?

ILU(k) here does no pivoting and no reordering -- `solve/ilu0.py` says so explicitly and PETSc's
ilu0 path takes the identity permutation -- so the elimination order is the caller's choice and a
zero-fill factorization is far more exposed to a bad one than a level-1 factorization, which
carries a second ring of couplings.

Two arms, on the SAME assembled operator, so nothing but the permutation / the mesh grading moves:

  ordering   permute the cells symmetrically before factorizing: natural (the mesh's own order),
             reverse-Cuthill-McKee, streamwise (sorted by x), and random as the pathological control.
  grading    the perturbed-grid channel with wall-normal grading, sweeping the growth ratio, so
             aspect ratio moves with skewness held at zero.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scipy.sparse as sp  # noqa: E402
from aquaflux.turbulence.coupled import (  # noqa: E402
    _coupled_shift_policy,
    _frozen_shift_diagonal,
)
from ilu_fill_probe import (  # noqa: E402
    FIELDS2,
    FIELDS3,
    assemble,
    build_grid,
    build_pitz,
    cell_aspect_ratio,
    ilu_pivots,
    ksp_solve,
    materialize,
    seed_state,
    skew_metrics,
)


def cell_graph(coupled):
    mesh = coupled.momentum.mesh
    fc = mesh.face_cells
    owner, nb = np.asarray(fc.owner), np.asarray(fc.neighbour)
    interior = np.asarray(fc.interior)
    o, n = owner[interior], nb[interior]
    data = np.ones(o.size)
    graph = sp.coo_matrix((data, (o, n)), shape=(mesh.n_cells, mesh.n_cells)).tocsr()
    return graph + graph.T


def orderings(coupled):
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    mesh = coupled.momentum.mesh
    n = mesh.n_cells
    centroid = np.asarray(mesh.geometry().cell.centroid)
    rng = np.random.default_rng(0)
    return {
        "natural": np.arange(n),
        "rcm": np.asarray(reverse_cuthill_mckee(cell_graph(coupled).tocsr(), symmetric_mode=True)),
        "streamwise": np.argsort(centroid[:, 0], kind="stable"),
        "random": rng.permutation(n),
    }


def bandwidth(a):
    rows = np.repeat(np.arange(a.shape[0]), np.diff(a.indptr))
    return int(np.abs(rows - a.indices).max())


def permute_cells(cell_major, order, n_fields):
    dof = (order[:, None] * n_fields + np.arange(n_fields)[None, :]).ravel()
    out = cell_major[dof][:, dof].tocsr()
    out.sort_indices()
    return out, dof


def run(name, coupled, betas, arms=("ilu0", "ilu1"), reach=3):
    dim = coupled.layout.dim
    n_fields = dim + 3
    names = FIELDS2 if dim == 2 else FIELDS3
    n = coupled.layout.n_cells
    ratio, interior = skew_metrics(coupled)
    live = ratio[interior]
    ar = cell_aspect_ratio(coupled)
    print(f"\n{'=' * 96}\nCASE {name}: {n} cells, {dim}D")
    print(f"  skew  median {np.median(live):.3e}  max {live.max():.3e}")
    print(f"  aspect ratio  median {np.median(ar):.2f}  p99 {np.quantile(ar, 0.99):.2f}  "
          f"max {ar.max():.2f}", flush=True)

    state, residual = seed_state(coupled)
    rhs = -np.asarray(residual, dtype=np.float64)
    jacobian = materialize(coupled, state, reach)
    base = _coupled_shift_policy(coupled, state, "twolevel")
    order_set = orderings(coupled)

    for beta in betas:
        shift = (_frozen_shift_diagonal(base, beta, state) if beta > 0
                 else np.zeros(n_fields * n))
        cell_major, scaling, perm = assemble(jacobian, np.asarray(shift), n_fields)
        rhs_eq = (np.asarray(scaling) * rhs)[perm]
        print(f"\n  -- beta {beta}", flush=True)
        for label, order in order_set.items():
            permuted, dof = permute_cells(cell_major, order, n_fields)
            b = rhs_eq[dof]
            bw = bandwidth(permuted)
            line = [f"     {label:<11} bw {bw:>7}"]
            for levels in (0, 1):
                census = ilu_pivots(permuted, n_fields, levels)
                line.append(f"| ILU({levels}) neg {census.get('negative', -1):>5} "
                            f"min|p| {census.get('min', float('nan')):.2e}")
            for arm in arms:
                out = ksp_solve(permuted, b, n_fields, arm)
                if "failed" in out:
                    line.append(f"| {arm} FAILED")
                    continue
                true = np.linalg.norm(permuted @ out["x"] - b) / np.linalg.norm(b)
                line.append(f"| {arm} its {out['its']:>4} rel {true:.2e}")
            print("  ".join(line), flush=True)
            del permuted
            gc.collect()
        del cell_major
        gc.collect()
    del jacobian
    gc.collect()


BETAS = tuple(float(b) for b in
              __import__("os").environ.get("PROBE_BETAS", "0.05,0.5").split(","))

if __name__ == "__main__":
    for spec in sys.argv[1:]:
        if spec == "pitz":
            coupled = build_pitz("corrected")
        elif spec == "pitz-compact":
            coupled = build_pitz("compact")
        elif spec.startswith("grid2d-"):
            _, p, growth = spec.split("-")
            coupled = build_grid(float(p), dim=2, growth=float(growth))
        else:
            raise SystemExit(f"unknown {spec!r}")
        run(spec, coupled, BETAS)
        del coupled
        gc.collect()
