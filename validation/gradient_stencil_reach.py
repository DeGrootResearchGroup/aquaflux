"""How far does the corrected-gradient sweep count push a residual's Jacobian across the cell graph?

A preconditioner assembled by coloured probing recovers the Jacobian out to a fixed graph distance
and no further. Where the residual reaches past that distance the far couplings are **folded onto
near entries** rather than dropped, because a colouring is collision-free only for the pattern it was
built at -- so an under-reaching probe corrupts entries a factorization turns into pivots instead of
merely omitting small terms. That makes "how far does the residual actually reach" a load-bearing
question, and the corrected gradient is the term most able to move the answer: each of its Richardson
sweeps applies an operator that couples a cell to its face neighbours.

This measures both halves of the trade the sweep count controls, so a cap can be chosen rather than
guessed:

1. **reach** -- the largest cell-graph distance carrying a nonzero of ``dR/dphi``, and the share of
   ``|dR/dphi|`` lying beyond distances 2 and 3 (what a probe at those reaches would have to fold).
2. **accuracy** -- the reconstructed gradient's departure from the exact (Krylov) solve of the same
   system, which is what the sweeps are being truncated out of.

Read the two together. Truncating the sweeps does **not** remove far coupling -- the mass beyond a
given distance is set by the mesh skewness, and a shorter sweep relocates it inward rather than
deleting it. What a cap buys is that the probe's recovered matrix is *exact for the residual it was
taken from*, which is a stated approximation of the operator instead of a corrupted one.

The case is scalar Laplace on a randomly perturbed grid with the exact field imposed on every
boundary, so nothing but the reconstruction is in play, and the whole sweep runs in about a minute.

Run: ``python3 validation/gradient_stencil_reach.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run directly (`python3 validation/gradient_stencil_reach.py`): Python puts THIS file's directory on
# the path, not the repository root, so put the root there ourselves -- as the case harnesses do.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.boundary import BoundaryConditions, DirichletField
from aquaflux.discretization import DiffusionFlux, ResidualAssembler
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import (
    CorrectedGreenGauss,
    GmresGradientSolve,
    SweptGradientSolve,
)
from tests.support.meshes import perturbed_grid_2d

CELLS = 12
SKEWS = (0.05, 0.15, 0.25, 0.40)
SWEEPS = (1, 2, 3, 4, 6, 8)


def _linear(x):
    """A harmonic linear field -- the exact discrete solution of the Laplace problem on any mesh."""
    return 2.0 * x[..., 0] - 3.0 * x[..., 1] + 1.0


def _smooth(x):
    """A field with genuine curvature, so the reconstruction has something to get wrong."""
    return jnp.sin(3.0 * x[..., 0]) * jnp.cos(2.0 * x[..., 1])


def _assembler(mesh, scheme):
    boundary = DirichletField(field_fn=_linear)
    return ResidualAssembler.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"diffusivity": Constant(1.0)}),
        (DiffusionFlux(),),
        BoundaryConditions({side: boundary for side in ("left", "right", "bottom", "top")}),
        gradient_scheme=scheme,
    )


def _cell_graph_distance(mesh) -> np.ndarray:
    """All-pairs graph distance over the interior-face cell graph (breadth-first per source)."""
    owner, nb, _ = mesh.face_cells.interior_edges()
    owner, nb = np.asarray(owner), np.asarray(nb)
    adjacency: list[list[int]] = [[] for _ in range(mesh.n_cells)]
    for a, b in zip(owner, nb, strict=True):
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    distance = np.full((mesh.n_cells, mesh.n_cells), mesh.n_cells, dtype=int)
    for source in range(mesh.n_cells):
        distance[source, source] = 0
        frontier, step = [source], 0
        while frontier:
            step += 1
            reached = []
            for u in frontier:
                for v in adjacency[u]:
                    if distance[source, v] > step:
                        distance[source, v] = step
                        reached.append(v)
            frontier = reached
    return distance


def main() -> None:
    print(f"\nscalar Laplace, {CELLS}x{CELLS} randomly perturbed grid, all-Dirichlet\n")
    header = f"{'skew':>6} {'sweeps':>7} {'reach':>6} {'beyond d=2':>12} {'beyond d=3':>12} {'grad err':>10}"
    for perturb in SKEWS:
        mesh = perturbed_grid_2d(CELLS, CELLS, perturb=perturb, seed=1, named_boundaries=True)
        distance = _cell_graph_distance(mesh)
        zero = jnp.zeros(mesh.n_cells)
        curved = _smooth(mesh.geometry().cell.centroid)

        exact = _assembler(mesh, CorrectedGreenGauss(solver=GmresGradientSolve()))
        exact_gradient = exact.gradient(curved)
        total = np.abs(np.asarray(jax.jacfwd(exact.residual)(zero))).sum()

        print(header)
        for sweeps in (*SWEEPS, None):
            solver = GmresGradientSolve() if sweeps is None else SweptGradientSolve(sweeps=sweeps)
            case = _assembler(mesh, CorrectedGreenGauss(solver=solver))
            jacobian = np.abs(np.asarray(jax.jacfwd(case.residual)(zero)))
            live = jacobian > 1e-13 * jacobian.max()
            error = float(
                jnp.linalg.norm(case.gradient(curved) - exact_gradient)
                / jnp.linalg.norm(exact_gradient)
            )
            print(
                f"{perturb:>6.2f} {'exact' if sweeps is None else sweeps:>7} "
                f"{int(distance[live].max()):>6} "
                f"{jacobian[distance > 2].sum() / total:>12.3e} "
                f"{jacobian[distance > 3].sum() / total:>12.3e} {error:>10.2e}"
            )
        print()


if __name__ == "__main__":
    main()
