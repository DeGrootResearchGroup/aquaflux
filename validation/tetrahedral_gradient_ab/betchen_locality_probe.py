"""How local is Betchen's coupled reconstruction on this tetrahedral mesh, and how fast does its sweep converge?

``HessianCorrectedGradient`` -- the coupled gradient and Hessian reconstruction of Betchen and Straatman
(2010) -- defines each cell's gradient through a system coupling every cell to its neighbours, solved by
a fixed number of coupled block sweeps. Two questions decide whether a variant converging in two or
three sweeps is possible at all:

1. **Reach of the converged operator.** Its gradient is linear in the field, ``g_P = sum_j w_Pj phi_j``.
   If the weight falls off fast with graph distance from ``P``, a local iteration can reproduce it; if
   it does not, no preconditioner can, because a preconditioner changes how fast the fixed point is
   approached, never how far it reaches. Reported: per cell, the fraction of ``sum_j |w_Pj|`` within
   ``d`` face hops, over ``BL_SAMPLE`` (default 150) sampled interior cells and as many owning a
   boundary face.
2. **Rate of the sweep.** The error after ``k`` sweeps against a far-converged reference, on a smooth
   and on a rough (random) field, with the per-sweep rate between successive counts, and where the
   slow error sits: its share in the cells owning a boundary face against their share of the mesh.

Settings: ``BL_CLOSURE`` = ``neighbour`` (default) or ``interior`` (the Hessian boundary closure);
``BL_WEIGHTS`` (comma-separated, default ``0.2``) the neighbour closure's ``weight``, one rate table per
value; ``BL_REACH`` = ``1`` (default) or ``0`` to skip part 1; ``BL_SWEEPS`` the sweep counts;
``BL_REFERENCE`` (default 300) the reference sweep count. The rate table splits the wall share into
the cells owning two or more boundary faces (``corner``) and the rest. Boundary values are Dirichlet
zero, so the operator is the field-to-gradient map with the boundary data held fixed.

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/betchen_locality_probe.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.io import read_openfoam
from aquaflux.schemes import (
    AveragedInteriorHessian,
    AveragedNeighbourHessian,
    CoupledBlockSweep,
    HessianCorrectedGradient,
)
from compare import POLYMESH
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

CLOSURES = {"neighbour": AveragedNeighbourHessian, "interior": AveragedInteriorHessian}
CLOSURE = os.environ.get("BL_CLOSURE", "neighbour")
REFERENCE = int(os.environ.get("BL_REFERENCE", "300"))
WEIGHTS = [float(w) for w in os.environ.get("BL_WEIGHTS", "0.2").split(",")]
REACH = os.environ.get("BL_REACH", "1") == "1"
SWEEPS = tuple(
    int(k)
    for k in os.environ.get("BL_SWEEPS", "1,2,3,4,5,6,8,10,15,20,30,40,60,80,120,160,200").split(
        ","
    )
)
MAX_HOPS = 8
SAMPLE = int(os.environ.get("BL_SAMPLE", "150"))  # cells per class for the reach rows
CHUNK = 16  # sampled cells per reverse-mode batch


def main() -> None:
    mesh = read_openfoam(POLYMESH)
    geometry = mesh.geometry()
    n = mesh.n_cells
    face_cells = mesh.face_cells
    owner, neighbour = np.asarray(face_cells.owner), np.asarray(face_cells.neighbour)
    interior = neighbour >= 0
    at_wall = np.isin(np.arange(n), owner[~interior])
    corner = np.bincount(owner[~interior], minlength=n) >= 2
    boundary_values = jnp.zeros(face_cells.n_faces)
    print(
        f"closure {CLOSURE}, reference {REFERENCE} sweeps, {n} cells, "
        f"{int(at_wall.sum())} owning a boundary face",
        flush=True,
    )

    def scheme(sweeps: int, weight: float = WEIGHTS[0]):
        closure = (
            AveragedNeighbourHessian(weight=weight)
            if CLOSURE == "neighbour"
            else CLOSURES[CLOSURE]()
        )
        return HessianCorrectedGradient(
            boundary_closure=closure, hessian_solve=CoupledBlockSweep(sweeps=sweeps)
        ).bind(mesh, geometry)

    def gradient(bound, field):
        return bound.gradients(field, mesh, geometry, boundary_values)

    # -- 2. the sweep's rate ------------------------------------------------------------------------
    rng = np.random.default_rng(7)
    centroid = np.asarray(geometry.cell.centroid)
    span = centroid.max(axis=0) - centroid.min(axis=0)
    unit = (centroid - centroid.min(axis=0)) / span
    fields = {
        "smooth": jnp.asarray(
            np.sin(2.0 * unit[:, 0]) * np.cos(1.5 * unit[:, 1]) + unit[:, 2] ** 2
        ),
        "rough": jnp.asarray(rng.standard_normal(n)),
    }
    for weight in WEIGHTS:
        reference = scheme(REFERENCE, weight)
        for label, field in fields.items():
            exact = np.asarray(gradient(reference, field))
            scale = np.abs(exact).max()
            print(
                f"\n[{label}, weight {weight}] error after k sweeps vs {REFERENCE}, "
                "max over cells / max|g|; share of squared error:"
            )
            print(f"{'k':>5s} {'error':>11s} {'rate/sweep':>11s} {'wall':>8s} {'corner':>8s}")
            previous = None
            for k in SWEEPS:
                error = np.linalg.norm(
                    np.asarray(gradient(scheme(k, weight), field)) - exact, axis=-1
                )
                worst = error.max() / scale
                rate = (
                    (worst / previous[1]) ** (1.0 / (k - previous[0])) if previous else float("nan")
                )
                squared = max((error**2).sum(), 1e-300)
                print(
                    f"{k:5d} {worst:11.3e} {rate:11.3f} {(error[at_wall] ** 2).sum() / squared:8.3f} "
                    f"{(error[corner] ** 2).sum() / squared:8.3f}",
                    flush=True,
                )
                previous = (k, worst)
    print(
        f"  (cells owning a boundary face are {at_wall.mean():.3f} of the mesh, "
        f"corner cells {corner.mean():.3f})"
    )
    if not REACH:
        return
    reference = scheme(REFERENCE)

    # -- 1. reach of the converged operator ---------------------------------------------------------
    # One row per sampled cell and gradient component, by reverse mode: the operator is linear, so a
    # vjp at zero returns the row's weights. Building the whole operator forward-mode instead carries
    # every cell's tangent through every sweep at once and exhausted this machine's memory.
    sample = np.concatenate(
        [
            rng.choice(np.flatnonzero(~at_wall), SAMPLE, replace=False),
            rng.choice(np.flatnonzero(at_wall), SAMPLE, replace=False),
        ]
    )
    print(
        f"\nrows of the converged operator ({REFERENCE} sweeps) for {SAMPLE} interior and "
        f"{SAMPLE} wall cells...",
        flush=True,
    )
    _, pullback = jax.vjp(lambda phi: gradient(reference, phi), jnp.zeros(n))
    dim = mesh.dim
    magnitude = np.zeros((sample.size, n))
    for start in range(0, sample.size, CHUNK):
        cells = sample[start : start + CHUNK]
        seeds = np.zeros((cells.size * dim, n, dim))
        for i, cell in enumerate(cells):
            for c in range(dim):
                seeds[i * dim + c, cell, c] = 1.0
        rows = np.asarray(jax.vmap(lambda seed: pullback(seed)[0])(jnp.asarray(seeds)))
        magnitude[start : start + cells.size] = np.abs(rows).reshape(cells.size, dim, n).sum(axis=1)
    adjacency = coo_matrix(
        (np.ones(interior.sum()), (owner[interior], neighbour[interior])), shape=(n, n)
    )
    hops = shortest_path(adjacency, unweighted=True, directed=False, indices=sample)
    total = magnitude.sum(axis=1)
    wall_row = at_wall[sample]
    print("\nfraction of sum_j |w_Pj| within d face hops (median / 10th percentile / worst cell):")
    print(f"{'d':>3s} {'interior cells':>28s} {'cells at a wall':>28s}")
    for d in range(MAX_HOPS + 1):
        within = np.where(hops <= d, magnitude, 0.0).sum(axis=1) / total
        columns = []
        for rows_here in (~wall_row, wall_row):
            part = within[rows_here]
            columns.append(
                f"{np.median(part):8.4f} {np.percentile(part, 10):8.4f} {part.min():8.4f}"
            )
        print(f"{d:3d} {columns[0]:>28s} {columns[1]:>28s}", flush=True)
    stencil = np.array([(hops <= d).sum(axis=1) for d in (1, 2, 3)])
    print(
        "\nmedian stencil size within 1 / 2 / 3 hops: "
        + " / ".join(f"{int(np.median(s))}" for s in stencil)
    )


if __name__ == "__main__":
    main()
