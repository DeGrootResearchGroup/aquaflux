"""Issue #435: does the second pass break the SIGN of the Rhie-Chow damping, from geometry alone?

The pressure block of the coupled flow residual contains, per interior face, a deliberately cancelling
difference ``T_f = (p_N - p_P) - interp(grad p) . d`` weighted by ``A / (d . n)``: the compact two-point
difference minus the reconstructed gradient's contribution, which vanishes on a linear pressure. Its
diagonal is negative (damping) when the gradient reconstruction perturbs the compact difference by less
than the difference itself, and positive (anti-damping) when the correction overshoots it.

This assembles that operator ``L = d/dp sum_f signed (A / (d . n)) T_f`` with no flow state at all (unit
weights, all-Dirichlet-zero boundary values) for the multiple-correction gradient with its second pass
withheld on chosen cells, and reports, per variant: the number of positive diagonals, the number of
positive eigenvalues of the symmetric part, the extreme eigenvalues, the median per-cell retention of the
compact diagonal (1 = kept, negative = sign flipped) and how many cells have flipped. It then prints the
localization of the largest anti-damping eigenvector against the rows' ``max|M2^-1|``.

Variants: first pass only, the full scheme, the second pass withheld in the wall-owning cells and each
ring out to k (graph distance from a boundary-owning cell), and the interventions the laminar march
arms tried (``max|M2^-1| > 10`` and ``> 3``, the cells owning two or more boundary faces, and exactly the
cells whose diagonal flips). The laminar march converges only with the second pass withheld out to ring 3,
and this operator's anti-damping eigenvalue falls to 1.3x the first pass's exactly there.

Measured under: the case mesh (``TET_POLYMESH``), ``OwnerGradient`` with ``SkewCorrectedGradient`` fallback,
unit weights and Dirichlet-zero boundary values (so counts corroborate, not reproduce, the flow residual's
own diagonals).

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/rhie_chow_sign_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.io import read_openfoam
from aquaflux.schemes import MultipleCorrectionGradient, OwnerGradient, SkewCorrectedGradient
from aquaflux.schemes.interpolation import interpolation_factor
from aquaflux.schemes.multiple_correction import Corrections
from aquaflux.vectors import dot
from compare import POLYMESH


def rings_from_boundary(mesh) -> np.ndarray:
    """Graph distance (face hops) of each cell from the nearest boundary-owning cell."""
    face_cells = mesh.face_cells
    owner = np.asarray(face_cells.owner)
    neighbour = np.asarray(face_cells.neighbour)
    interior = neighbour >= 0
    adjacent = [[] for _ in range(mesh.n_cells)]
    for o, v in zip(owner[interior], neighbour[interior], strict=True):
        adjacent[o].append(v)
        adjacent[v].append(o)
    ring = np.full(mesh.n_cells, -1)
    front = np.unique(owner[~interior])
    ring[front] = 0
    current, distance = list(front), 0
    while current:
        following = []
        for cell in current:
            for other in adjacent[cell]:
                if ring[other] < 0:
                    ring[other] = distance + 1
                    following.append(other)
        current, distance = following, distance + 1
    return ring


def main() -> None:
    mesh = read_openfoam(POLYMESH)
    geometry = mesh.geometry()
    n = mesh.n_cells
    face_cells = mesh.face_cells
    owner, neighbour = np.asarray(face_cells.owner), np.asarray(face_cells.neighbour)
    interior = neighbour >= 0

    base = MultipleCorrectionGradient(
        boundary_closure=OwnerGradient(), fallback=SkewCorrectedGradient()
    ).bind(mesh, geometry)
    prepared = base.prepared
    ring = rings_from_boundary(mesh)
    print("ring populations:", np.bincount(ring[ring >= 0]).tolist(), flush=True)

    factor = interpolation_factor(face_cells, geometry)
    d_vector = (
        face_cells.neighbour_centroid(geometry.cell.centroid)
        - geometry.cell.centroid[face_cells.owner]
    )
    normal_distance = dot(d_vector, geometry.face.normal)
    weight = jnp.where(
        face_cells.interior,
        geometry.face.area / jnp.where(jnp.abs(normal_distance) > 0, normal_distance, 1.0),
        0.0,
    )
    boundary_values = jnp.zeros(face_cells.n_faces)

    def with_second_pass_on(cells_full: np.ndarray):
        """The scheme with its second pass active only where ``cells_full`` is True."""
        defect = jnp.asarray(np.asarray(prepared.gradient_defect) * cells_full[:, None, None])
        return MultipleCorrectionGradient(
            boundary_closure=base.boundary_closure,
            fallback=base.fallback,
            prepared=Corrections(
                prepared.m1, prepared.m1_inverse, prepared.m2_inverse, defect, prepared.closure
            ),
        )

    def damping_matrix(scheme) -> np.ndarray:
        def cell_sum(pressure):
            gradient = scheme.reconstruct(pressure, mesh, geometry, boundary_values)[0]
            face_gradient = (1.0 - factor)[:, None] * gradient[face_cells.owner] + factor[
                :, None
            ] * gradient[face_cells.safe_neighbour]
            difference = (pressure[face_cells.safe_neighbour] - pressure[face_cells.owner]) - dot(
                face_gradient, d_vector
            )
            return face_cells.scatter_conservative(
                jnp.where(face_cells.interior, weight * difference, 0.0)
            )

        return np.asarray(jax.jacfwd(cell_sum)(jnp.zeros(n)))

    # The compact-only diagonal (gradient == 0): T_f = p_N - p_P.
    compact_diagonal = np.zeros(n)
    face_weight = np.asarray(weight)
    np.add.at(compact_diagonal, owner[interior], -face_weight[interior])
    np.add.at(compact_diagonal, neighbour[interior], -face_weight[interior])

    def report(label: str, cells_full: np.ndarray):
        matrix = damping_matrix(with_second_pass_on(cells_full.astype(float)))
        diagonal = np.diag(matrix)
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        tolerance = 1e-10 * np.abs(eigenvalues).max()
        retention = diagonal / compact_diagonal
        print(
            f"{label:34s} {int((diagonal > 0).sum()):7d} {int((eigenvalues > tolerance).sum()):6d} "
            f"{eigenvalues.max():+11.3e} {eigenvalues.min():+11.3e} {np.median(retention):14.4f} "
            f"{int((retention < 0).sum()):12d} {int(cells_full.sum()):11d}",
            flush=True,
        )
        return matrix

    print(
        f"\n{'variant':34s} {'diag>0':>7s} {'eig>0':>6s} {'max eig':>11s} {'min eig':>11s} "
        f"{'retention med':>14s} {'ret<0 cells':>12s} {'cells full':>11s}"
    )
    report("first pass only", np.zeros(n, bool))
    full_matrix = report("FULL everywhere", np.ones(n, bool))
    for k in range(5):
        report(f"withhold ring<={k}", ring > k)

    m2_worst = np.max(np.abs(np.asarray(prepared.m2_inverse)), axis=(1, 2))
    boundary_faces = np.zeros(n, int)
    np.add.at(boundary_faces, owner[~interior], 1)
    print("\ninterventions the laminar march arms tried:")
    for label, cells_full in {
        "withhold max|M2^-1|>10": m2_worst <= 10,
        "withhold max|M2^-1|>3": m2_worst <= 3,
        "withhold the 176 corner cells": boundary_faces < 2,
        "withhold the sign-flipped cells": ~(np.diag(full_matrix) > 0),
    }.items():
        report(label, cells_full)

    eigenvalues, vectors = np.linalg.eigh(0.5 * (full_matrix + full_matrix.T))
    energy = vectors[:, -1] ** 2
    order = np.argsort(-energy)
    print(
        f"\nFULL anti-damping mode (eig {eigenvalues[-1]:+.3e}): top cell {order[0]} holds "
        f"{energy[order[0]] * 100:.1f}%, top5 {energy[order[:5]].sum() * 100:.1f}%, ring "
        f"{ring[order[0]]}, max|M2^-1| {m2_worst[order[0]]:.1f}, boundary faces {boundary_faces[order[0]]}"
    )
    spearman = np.corrcoef(np.argsort(np.argsort(energy)), np.argsort(np.argsort(m2_worst)))[0, 1]
    print(f"Spearman(mode energy, max|M2^-1|) = {spearman:.3f}")
    print(
        f"mesh max|M2^-1|: median {np.median(m2_worst):.3f}  p99 {np.percentile(m2_worst, 99):.1f}  "
        f"max {m2_worst.max():.3g}"
    )


if __name__ == "__main__":
    main()
