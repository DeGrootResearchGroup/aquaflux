"""How large are `ProjectedStencilGradient`'s weights on this mesh, and does the blend move them?

The scheme takes, per cell, the stencil weights exact for quadratics nearest ``blend`` x a reference
reconstruction; ``blend = 0`` is the minimum-norm end. On this graded mesh its conditioning guard
fires -- the near-wall cells are thin, so a two-hop stencil through them is nearly flat and the
quadratic fit is ill-conditioned -- and the question is whether a lower blend keeps the weights small
enough there without paying for a wider stencil.

⚠️ **The conditioning number is a property of the STENCIL, not of the blend**: the same monomials are
fitted whatever the weights aim at, so every blend reports the same guard. What the blend moves is the
size of the weights the fit lands on, and through them the sign of the Rhie--Chow damping.

Reported per blend: the largest and the 99th-percentile weight, over the whole mesh and over the cells
the guard names; and the damping operator's diagonal -- how many cells it flips, and the worst
retention of the compact two-point difference (1 = kept whole, negative = sign flipped). The diagonal
is assembled from the stored weights directly rather than from a materialized operator, which at this
mesh size would be 4 GB.

Run: validation/run_case.sh validation/bfs3d_openfoam/gradient_weight_probe.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
from aquaflux.io import read_openfoam
from aquaflux.schemes import BoundaryLinearization, ProjectedStencilGradient
from aquaflux.schemes.interpolation import interpolation_factor
from aquaflux.schemes.projected_stencil import _ILL_CONDITIONED, build_stencil, polynomial_basis
from compare import RUNS

BLENDS = [float(b) for b in os.environ.get("BFS3D_BLENDS", "0,0.25,0.5,0.75,1").split(",")]


def main() -> None:
    mesh = read_openfoam(RUNS / "polyMesh")
    geometry = mesh.geometry()
    n, dim = mesh.n_cells, mesh.dim
    face_cells = mesh.face_cells
    owner = np.asarray(face_cells.owner)
    neighbour = np.asarray(face_cells.neighbour)
    interior = neighbour >= 0
    print(f"{n} cells, {int((~interior).sum())} boundary faces", flush=True)

    # The pressure's conditions on this case: a prescribed value at the outlet, everything else
    # extrapolating -- the binding whose damping the Rhie-Chow coupling reads.
    outlet = np.zeros(mesh.n_faces, dtype=bool)
    outlet[np.asarray(mesh.face_patches.indices("outlet"))] = True
    linearization = BoundaryLinearization(
        value_weight=jnp.asarray(np.where(~interior & ~outlet, 1.0, 0.0)),
        gradient_weight=jnp.zeros((mesh.n_faces, dim)),
    )

    # Which cells the guard names -- the same set for every blend, so it is measured once.
    stencil = build_stencil(mesh, 2)
    centroid = geometry.cell.centroid
    size = geometry.cell.volume ** (1.0 / dim)
    offsets = (centroid[stencil.cells] - centroid[:, None, :]) / size[:, None, None]
    basis = jnp.where(stencil.cell_used[:, :, None], polynomial_basis(offsets, dim), 0.0)
    eigenvalues = np.linalg.eigvalsh(np.asarray(jnp.einsum("nkt,nks->nts", basis, basis)))
    condition = np.where(
        eigenvalues[:, 0] > 0,
        eigenvalues[:, -1] / np.where(eigenvalues[:, 0] > 0, eigenvalues[:, 0], 1.0),
        np.inf,
    )
    flagged = condition > _ILL_CONDITIONED
    print(
        f"cells the guard names: {int(flagged.sum())} (worst condition {condition.max():.2e}); "
        "the same set at every blend -- the fit is the stencil's, not the aim's",
        flush=True,
    )

    # The compact two-point difference's own diagonal, and the pieces the gradient contributes.
    factor = np.asarray(interpolation_factor(face_cells, geometry))
    separation = np.asarray(face_cells.neighbour_centroid(centroid) - centroid[face_cells.owner])
    normal = np.asarray(geometry.face.normal)
    area = np.asarray(geometry.face.area)
    along = np.sum(separation * normal, axis=-1)
    weight = np.where(interior, area / np.where(np.abs(along) > 0, along, 1.0), 0.0)
    compact = np.zeros(n)
    np.add.at(compact, owner[interior], -weight[interior])
    np.add.at(compact, neighbour[interior], -weight[interior])

    print(
        f"\n{'blend':>6s} {'max|w|':>9s} {'p99|w|':>9s} {'max|w| flagged':>15s} "
        f"{'flipped':>8s} {'worst retention':>16s}"
    )
    for blend in BLENDS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scheme = ProjectedStencilGradient(blend=blend).bind(mesh, geometry, linearization)
        cells, cell_weights = scheme.prepared[0].cells, scheme.prepared[1]
        magnitude = np.abs(np.asarray(cell_weights))

        # The damping diagonal: for each interior face, the gradient of the owner and of the
        # neighbour each contribute their own column-P entry, which is the stencil weight on P.
        index = np.asarray(cells)
        values = np.asarray(cell_weights)
        own_column = np.zeros((n, dim))
        for k in range(index.shape[1]):
            hit = index[:, k] == np.arange(n)
            own_column[hit] += values[hit, :, k]

        # column P of the neighbour's row: the neighbour's weight on P, where P is in its stencil
        def column_of(rows, target, index=index, values=values):
            out = np.zeros((rows.size, dim))
            for k in range(index.shape[1]):
                hit = index[rows, k] == target
                out[hit] += values[rows[hit], :, k]
            return out

        o, nb, f = owner[interior], neighbour[interior], factor[interior]
        face_gradient_own = (1.0 - f)[:, None] * own_column[o] + f[:, None] * column_of(nb, o)
        face_gradient_nb = (1.0 - f)[:, None] * column_of(o, nb) + f[:, None] * own_column[nb]
        through_own = np.sum(separation[interior] * face_gradient_own, axis=-1) * weight[interior]
        through_nb = np.sum(separation[interior] * face_gradient_nb, axis=-1) * weight[interior]
        diagonal = compact.copy()
        np.add.at(diagonal, o, -through_own)
        np.add.at(diagonal, nb, through_nb)
        retention = diagonal / compact
        print(
            f"{blend:6.2f} {magnitude.max():9.2f} {np.percentile(magnitude, 99):9.3f} "
            f"{magnitude[flagged].max():15.2f} {int((diagonal > 0).sum()):8d} "
            f"{retention.min():16.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
