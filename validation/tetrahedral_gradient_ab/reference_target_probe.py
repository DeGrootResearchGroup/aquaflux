"""Does ``ProjectedStencilGradient`` need the gradient--Hessian system to aim at, or will anything
bounded do?

The scheme takes, of the weights exact for quadratics on its stencil, the ones nearest ``blend`` x a
reference. The reference is a *target*, never part of the answer: the exact weights form an affine
set, the projection lands in it whatever it aims at, and what the target decides is only which point
of that set -- so its own accuracy is irrelevant and its **magnitude** is the whole contribution. It
is currently one sweep of ``HessianCorrectedGradient``'s system, which means binding builds that
scheme's systems (its Hessian block, its boundary closure, its local Schur correction) purely to be
thrown away.

If a cheaper bounded target does the same job, that construction is dead weight: the module stops
importing the package's most complicated scheme to prepare its simplest, ``boundary_weight`` (which
exists only to configure the target's Hessian closure) goes with it, and binding gets cheaper.

Three targets, all ``P^-1`` times the SAME Green--Gauss face sum, differing only in the per-cell
block ``P``:

* ``hessian`` -- the gradient equation's diagonal block with its local Schur correction (shipped);
* ``block`` -- the same diagonal block, no Schur correction (no Hessian system, no closure);
* ``compact`` -- ``V I``, i.e. the target IS compact Green--Gauss, the cheapest bounded choice.

Per target and blend it reports: the time to build the target, the resulting weight magnitudes, the
Rhie--Chow damping the flow would see (flipped diagonals and worst retention -- the quantity that
decides whether the coupled march can start at all), exactness on a quadratic, and the error on a
smooth non-quadratic field, which is where the blend earns its keep. A target is a candidate
replacement only if its damping column matches and its accuracy does not fall off.

Settings: ``REF_TARGETS`` (comma list, default all three), ``REF_BLENDS`` (default ``0,0.75``).

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/reference_target_probe.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
from aquaflux.io import read_openfoam
from aquaflux.schemes import AveragedNeighbourHessian, ProjectedStencilGradient
from aquaflux.schemes.gradient import HessianCorrectedGradient
from aquaflux.schemes.projected_stencil import (
    _constrained_weights,
    _reference_inverse,
    _reference_weights,
    build_stencil,
)
from compare import POLYMESH
from rhie_chow_sign_probe import compact_damping_diagonal, damping_matrix

TARGETS = os.environ.get("REF_TARGETS", "hessian,block,compact").split(",")
BLENDS = [float(b) for b in os.environ.get("REF_BLENDS", "0,0.75").split(",")]

HESSIAN = np.array([[1.3, 0.4, -0.2], [0.4, -0.7, 0.3], [-0.2, 0.3, 0.9]])
SLOPE = np.array([0.5, -1.1, 0.8])


def quadratic(points: np.ndarray) -> np.ndarray:
    return 0.5 * np.einsum("...i,ij,...j->...", points, HESSIAN, points) + points @ SLOPE


def quadratic_gradient(points: np.ndarray) -> np.ndarray:
    return points @ HESSIAN + SLOPE


def smooth(points: np.ndarray) -> np.ndarray:
    """A field with every derivative order in it, so the blend's effect on truncation error shows."""
    return np.sin(3.1 * points[:, 0]) * np.exp(0.4 * points[:, 1]) + np.cos(2.3 * points[:, 2])


def smooth_gradient(points: np.ndarray) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return np.stack(
        [
            3.1 * np.cos(3.1 * x) * np.exp(0.4 * y),
            0.4 * np.sin(3.1 * x) * np.exp(0.4 * y),
            -2.3 * np.sin(2.3 * z),
        ],
        axis=1,
    )


def block_inverse(mesh, geometry, name: str) -> jnp.ndarray:
    """The per-cell block each target divides the Green--Gauss sum by, ``(n_cells, dim, dim)``."""
    if name == "hessian":
        return _reference_inverse(mesh, geometry, AveragedNeighbourHessian(weight=0.5))
    if name == "block":
        # The same diagonal block with no Schur correction, which is what skipping the Hessian
        # system amounts to: `_systems` still assembles it, so this arm isolates the correction's
        # effect rather than its cost.
        systems = HessianCorrectedGradient._systems(mesh, geometry, AveragedNeighbourHessian())
        return systems.outer_preconditioner(systems.inner(), False).inverse
    if name == "compact":
        # Compact Green--Gauss: the sum divided by the cell volume, and nothing else.
        volume = np.asarray(geometry.cell.volume)
        return jnp.asarray(np.eye(mesh.dim)[None, :, :] / volume[:, None, None])
    raise ValueError(f"unknown target {name!r}; expected one of hessian, block, compact")


def main() -> None:
    mesh = read_openfoam(POLYMESH)
    geometry = mesh.geometry()
    n = mesh.n_cells
    centroid = np.asarray(geometry.cell.centroid)
    face_centroid = np.asarray(geometry.face.centroid)
    interior = np.asarray(mesh.face_cells.interior)
    print(f"{n} cells, {int((~interior).sum())} boundary faces", flush=True)

    stencil = build_stencil(mesh, 2)
    extrapolates = np.zeros(mesh.n_faces, dtype=bool)  # geometry binding: every face prescribed
    compact = compact_damping_diagonal(mesh, geometry)
    volume = np.asarray(geometry.cell.volume)

    print(
        f"\n{'target':9s} {'blend':>5s} {'build s':>8s} {'max|w|':>9s} {'p99|w|':>9s} "
        f"{'flipped':>8s} {'worst ret':>10s} {'max eig':>11s} {'quadratic':>10s} {'smooth':>10s}"
    )
    for name in TARGETS:
        started = time.perf_counter()
        inverse = block_inverse(mesh, geometry, name)
        reference = _reference_weights(mesh, geometry, stencil, inverse)
        jnp.asarray(reference[0]).block_until_ready()
        build = time.perf_counter() - started
        size = float(jnp.abs(reference[0]).max())
        print(
            f"  [{name}] target built in {build:.2f}s, largest target weight {size:.4g}", flush=True
        )

        for blend in BLENDS:
            cell_weights, face_weights = _constrained_weights(
                geometry, stencil, extrapolates, blend, reference
            )
            scheme = ProjectedStencilGradient(
                blend=blend,
                prepared=(
                    stencil,
                    cell_weights,
                    face_weights,
                    jnp.asarray(~interior),
                    jnp.asarray(extrapolates),
                ),
            )
            magnitude = np.abs(np.asarray(cell_weights))

            matrix = damping_matrix(
                mesh,
                geometry,
                lambda pressure, s=scheme: s.gradients(
                    pressure, mesh, geometry, jnp.zeros(mesh.n_faces)
                ),
            )
            diagonal = np.diag(matrix)
            eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
            retention = diagonal / compact

            def error(field, exact, s=scheme) -> float:
                computed = np.asarray(
                    s.gradients(
                        jnp.asarray(field(centroid)),
                        mesh,
                        geometry,
                        jnp.asarray(field(face_centroid)),
                    )
                )
                difference = np.linalg.norm(computed - exact(centroid), axis=-1)
                scale = np.linalg.norm(exact(centroid), axis=-1).max()
                # Volume-weighted mean, so a few small cells do not set the number.
                return float((difference * volume).sum() / volume.sum() / scale)

            print(
                f"{name:9s} {blend:5.2f} {build:8.2f} {magnitude.max():9.3f} "
                f"{np.percentile(magnitude, 99):9.3f} {int((diagonal > 0).sum()):8d} "
                f"{retention.min():10.4f} {eigenvalues.max():+11.3e} "
                f"{error(quadratic, quadratic_gradient):10.2e} "
                f"{error(smooth, smooth_gradient):10.2e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
