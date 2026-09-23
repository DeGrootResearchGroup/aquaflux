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

``RC_BINDING`` picks what the gradient sees:

* ``geometry`` (default) -- the scheme bound with no boundary linearization, reconstructing with unit
  weights and Dirichlet-zero boundary values (so counts corroborate, not reproduce, the flow residual's
  own diagonals);
* ``pressure`` -- the laminar duct's own pressure gradient (``laminar_duct_march.build_case`` with the
  multiple-correction scheme): bound against the pressure conditions' boundary linearization, and
  evaluated through the assembler with those conditions' run-time boundary values.

``RC_SCHEMES`` (comma-separated, default ``multiple``) picks the schemes. ``multiple`` runs the variant
ladder above; every other entry adds one row for a whole scheme under the same binding: ``corrected``
(``CorrectedGreenGauss``, the scheme the laminar march converges with), ``projected[-<blend>]``
(``ProjectedStencilGradient`` at that blend, default 0.75), ``hessian-owner``,
``hessian-neighbour`` and ``hessian-interior`` (``HessianCorrectedGradient`` -- the coupled gradient and
Hessian reconstruction of Betchen and Straatman (2010) -- under each Hessian boundary closure, at its
default 20 coupled sweeps) and ``hessian-<closure>-<k>`` (the same at ``k`` coupled sweeps: 100
shows whether 20 converged, and a small ``k`` is a local reconstruction whose reach grows by one face
per sweep). The ``nnz/row``
column is the operator's mean nonzeros per row, the reach a flow Jacobian would inherit. A
quadratic-exactness line per scheme follows, since a fixed sweep count carries no convergence test.

Measured under: the case mesh; the multiple-correction scheme with ``OwnerGradient`` and the
``SkewCorrectedGradient`` fallback.

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/rhie_chow_sign_probe.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.io import read_openfoam
from aquaflux.schemes import (
    AveragedInteriorHessian,
    AveragedNeighbourHessian,
    CorrectedGreenGauss,
    CoupledBlockSweep,
    HessianCorrectedGradient,
    MultipleCorrectionGradient,
    OwnerGradient,
    OwnerHessian,
    ProjectedStencilGradient,
    SkewCorrectedGradient,
)
from aquaflux.schemes.interpolation import interpolation_factor
from aquaflux.vectors import dot
from compare import POLYMESH

BINDING = os.environ.get("RC_BINDING", "geometry")
SCHEMES = os.environ.get("RC_SCHEMES", "multiple").split(",")
HESSIAN_CLOSURES = {
    "owner": OwnerHessian,
    "neighbour": AveragedNeighbourHessian,
    "interior": AveragedInteriorHessian,
}


def other_scheme(name: str):
    """``corrected``, ``projected[-<blend>]``, or ``hessian-<closure>[-<sweeps>]`` (default 20)."""
    if name == "corrected":
        return CorrectedGreenGauss()
    parts = name.split("-")
    if parts[0] == "projected":
        blend = float(parts[1]) if len(parts) == 2 else 0.75
        return ProjectedStencilGradient(blend=blend)
    if parts[0] != "hessian" or len(parts) not in (2, 3) or parts[1] not in HESSIAN_CLOSURES:
        raise ValueError(f"unknown RC_SCHEMES entry {name!r}")
    closure = HESSIAN_CLOSURES[parts[1]]()
    if len(parts) == 2:
        return HessianCorrectedGradient(boundary_closure=closure)
    return HessianCorrectedGradient(
        boundary_closure=closure, hessian_solve=CoupledBlockSweep(sweeps=int(parts[2]))
    )


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


def damping_face_weight(mesh, geometry):
    """``A / (d . n)`` per interior face, zero on a boundary face -- the damping term's own weight."""
    face_cells = mesh.face_cells
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
    return d_vector, weight


def damping_matrix(mesh, geometry, gradient_of) -> np.ndarray:
    """The Rhie--Chow damping operator ``L = d/dp sum_f signed (A / (d . n)) T_f``.

    ``gradient_of(pressure)`` returns the cell gradient the reconstruction under test produces for
    that pressure, so a caller supplies whatever binding and boundary data it is measuring under.
    Materialized by ``jacfwd``, which is affordable on this mesh and exact.
    """
    face_cells = mesh.face_cells
    factor = interpolation_factor(face_cells, geometry)
    d_vector, weight = damping_face_weight(mesh, geometry)

    def cell_sum(pressure):
        gradient = gradient_of(pressure)
        face_gradient = (1.0 - factor)[:, None] * gradient[face_cells.owner] + factor[:, None] * (
            gradient[face_cells.safe_neighbour]
        )
        difference = (pressure[face_cells.safe_neighbour] - pressure[face_cells.owner]) - dot(
            face_gradient, d_vector
        )
        return face_cells.scatter_conservative(
            jnp.where(face_cells.interior, weight * difference, 0.0)
        )

    return np.asarray(jax.jacfwd(cell_sum)(jnp.zeros(mesh.n_cells)))


def compact_damping_diagonal(mesh, geometry) -> np.ndarray:
    """The same operator's diagonal with the gradient set to zero: ``T_f = p_N - p_P``.

    The denominator of the retention column -- what the two-point difference damps by on its own,
    before any reconstruction is subtracted from it.
    """
    owner = np.asarray(mesh.face_cells.owner)
    neighbour = np.asarray(mesh.face_cells.neighbour)
    interior = neighbour >= 0
    face_weight = np.asarray(damping_face_weight(mesh, geometry)[1])
    diagonal = np.zeros(mesh.n_cells)
    np.add.at(diagonal, owner[interior], -face_weight[interior])
    np.add.at(diagonal, neighbour[interior], -face_weight[interior])
    return diagonal


def main() -> None:
    def cell_gradient(scheme, field, boundary_values):
        """The scheme's cell gradient; the multiple-correction ladder keeps its own entry point."""
        if isinstance(scheme, MultipleCorrectionGradient):
            return scheme.reconstruct(field, mesh, geometry, boundary_values)[0]
        return scheme.gradients(field, mesh, geometry, boundary_values)

    if BINDING == "pressure":
        os.environ["LAM_SCHEME"] = "multiple"
        from laminar_duct_march import build_case

        momentum = build_case()
        mesh, geometry = momentum.mesh, momentum.geometry
        base = momentum.pressure_gradient_scheme
        linearization = momentum._build_time_pressure_linearization()

        def bound(scheme):
            return scheme.bind(mesh, geometry, linearization)

        def gradient_of(scheme, pressure):
            with_scheme = eqx.tree_at(lambda m: m.pressure_gradient_scheme, momentum, scheme)
            return with_scheme._pressure_gradient(pressure)[0]

    elif BINDING == "geometry":
        mesh = read_openfoam(POLYMESH)
        geometry = mesh.geometry()
        base = MultipleCorrectionGradient(
            boundary_closure=OwnerGradient(), fallback=SkewCorrectedGradient()
        ).bind(mesh, geometry)
        boundary_values = jnp.zeros(mesh.face_cells.n_faces)

        def bound(scheme):
            return scheme.bind(mesh, geometry)

        def gradient_of(scheme, pressure):
            return cell_gradient(scheme, pressure, boundary_values)

    else:
        raise ValueError(f"RC_BINDING must be 'geometry' or 'pressure', not {BINDING!r}")
    for name in SCHEMES:
        if name != "multiple":
            other_scheme(name)  # reject a mistyped entry before any work
    print(f"binding: {BINDING}   schemes: {','.join(SCHEMES)}", flush=True)
    n = mesh.n_cells
    face_cells = mesh.face_cells
    owner, neighbour = np.asarray(face_cells.owner), np.asarray(face_cells.neighbour)
    interior = neighbour >= 0
    prepared = base.prepared
    ring = rings_from_boundary(mesh)
    print("ring populations:", np.bincount(ring[ring >= 0]).tolist(), flush=True)

    def with_second_pass_on(cells_full: np.ndarray):
        """The scheme with its second pass active only where ``cells_full`` is True."""
        defect = jnp.asarray(np.asarray(prepared.gradient_defect) * cells_full[:, None, None])
        return eqx.tree_at(lambda scheme: scheme.prepared.gradient_defect, base, defect)

    def matrix_of(scheme) -> np.ndarray:
        return damping_matrix(mesh, geometry, lambda pressure: gradient_of(scheme, pressure))

    compact_diagonal = compact_damping_diagonal(mesh, geometry)

    def report(label: str, cells_full: np.ndarray):
        matrix = matrix_of(with_second_pass_on(cells_full.astype(float)))
        report_matrix(label, matrix, int(cells_full.sum()))
        return matrix

    def report_matrix(label: str, matrix: np.ndarray, cells_full: int) -> None:
        if not np.all(np.isfinite(matrix)):
            print(f"{label:34s} non-finite operator (the reconstruction diverged)", flush=True)
            return
        diagonal = np.diag(matrix)
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        tolerance = 1e-10 * np.abs(eigenvalues).max()
        retention = diagonal / compact_diagonal
        print(
            f"{label:34s} {int((diagonal > 0).sum()):7d} {int((eigenvalues > tolerance).sum()):6d} "
            f"{eigenvalues.max():+11.3e} {eigenvalues.min():+11.3e} {np.median(retention):14.4f} "
            f"{int((retention < 0).sum()):12d} {cells_full:11d} "
            f"{np.count_nonzero(np.abs(matrix) > 1e-14 * np.abs(matrix).max()) / n:8.1f}",
            flush=True,
        )

    def quadratic_error(scheme) -> tuple[float, float]:
        """Worst gradient error on a quadratic with exact boundary values: all cells, then cells
        owning no boundary face -- relative to the largest exact gradient."""
        geometric = scheme.bind(mesh, geometry)
        hessian = jnp.array([[1.3, 0.4, -0.2], [0.4, -0.7, 0.3], [-0.2, 0.3, 0.9]])
        slope = jnp.array([0.5, -1.1, 0.8])

        def value(x):
            return 0.5 * jnp.einsum("...i,ij,...j->...", x, hessian, x) + x @ slope

        exact = geometry.cell.centroid @ hessian + slope
        computed = cell_gradient(
            geometric, value(geometry.cell.centroid), value(geometry.face.centroid)
        )
        error = np.linalg.norm(np.asarray(computed - exact), axis=-1)
        scale = float(np.linalg.norm(np.asarray(exact), axis=-1).max())
        clear = ~np.isin(np.arange(n), owner[~interior])
        return error.max() / scale, error[clear].max() / scale

    if "multiple" not in SCHEMES:
        print(
            f"\n{'scheme':34s} {'diag>0':>7s} {'eig>0':>6s} {'max eig':>11s} {'min eig':>11s} "
            f"{'retention med':>14s} {'ret<0 cells':>12s} {'cells full':>11s} {'nnz/row':>8s}"
        )
        for name in SCHEMES:
            report_matrix(name, matrix_of(bound(other_scheme(name))), n)
        print(
            "\nquadratic exactness (worst |g - g_exact| / max|g_exact|): all cells, interior cells"
        )
        for name in SCHEMES:
            everywhere, inside = quadratic_error(other_scheme(name))
            print(f"{name:34s} {everywhere:10.3e} {inside:10.3e}", flush=True)
        return

    print(
        f"\n{'variant':34s} {'diag>0':>7s} {'eig>0':>6s} {'max eig':>11s} {'min eig':>11s} "
        f"{'retention med':>14s} {'ret<0 cells':>12s} {'cells full':>11s} {'nnz/row':>8s}"
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
