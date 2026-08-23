"""Why does the Schur-block preconditioner diverge on a handful of cells of a real mesh?

``HessianCorrectedGradient(local_schur_block=True)`` builds its outer preconditioner from the Schur
complement's own per-cell block rather than from ``A_gg``'s. On every synthetic mesh tested that is
better everywhere; on a 1.6M-cell snappyHexMesh reactor it is better on the median cell and
catastrophically wrong on a few -- the reconstruction of a quadratic reaches ``4.4e+18`` at its worst
cell, where the ``A_gg`` block reaches ``6.2e-02`` and an exact Krylov solve reaches ``2.4e-02``.

The operator is not at fault: an exact solve of the same system is fine. What fails is the fixed-count
Richardson sweep under this preconditioner, which is what a preconditioner that is catastrophically
wrong on one row does to a stationary iteration -- a Krylov method would merely spend iterations.

This script finds those cells, describes their geometry, and compares three blocks on them:

* the **true** Schur block, obtained by lighting one cell's degree of freedom and reading back that
  cell's own row. No neighbour can contribute to it, so it is the true block whatever preconditioner
  is in force -- which makes it the one measurement here that cannot be circular;
* the cell-local **approximation** the preconditioner actually builds;
* ``A_gg``'s block, which does not diverge.

That comparison separates the candidate mechanisms. If the approximation is near-singular where the
true block is not, the dropped neighbour paths are the cause and a per-cell conditioning fallback is
the fix. If the true block is itself near-singular, then ``A_gg``'s block survives only by being
wrong in a harmless direction, and the answer is a Krylov outer solve on such meshes rather than a
patched preconditioner.

Usage
-----
    UV_MESH=<path-to-polyMesh> validation/run_case.sh validation/uvreactor_openfoam/schur_block_diagnosis.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import aquaflux  # noqa: E402,F401  (enables x64)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.io import read_openfoam  # noqa: E402
from aquaflux.mesh.quality import closed_cell_residual, face_planarity  # noqa: E402
from aquaflux.schemes import HessianCorrectedGradient, SweptGradientSolve  # noqa: E402
from aquaflux.schemes.gradient import cell_diagonal_block  # noqa: E402,F401

INNER = int(os.environ.get("UV_INNER", "12"))
WORST = int(os.environ.get("UV_WORST", "12"))


def quadratic(points, centre, extent):
    u = (points - centre) / extent
    return 0.6 + 1.7 * u[:, 0] - 1.1 * u[:, 1] + 0.8 * u[:, 2] + 0.4 * u[:, 0] ** 2


def quadratic_gradient(points, centre, extent):
    u = (points - centre) / extent
    return (
        np.stack(
            [1.7 + 0.8 * u[:, 0], -1.1 * np.ones_like(u[:, 0]), 0.8 * np.ones_like(u[:, 0])],
            axis=-1,
        )
        / extent
    )


def main() -> None:
    mesh_path = os.environ.get("UV_MESH")
    if mesh_path is None:
        raise SystemExit("set UV_MESH to a polyMesh directory")
    mesh = read_openfoam(Path(mesh_path))
    geom = mesh.geometry()
    print(f"mesh: {mesh.n_cells} cells, {mesh.n_faces} faces", flush=True)

    x = np.asarray(geom.cell.centroid)
    centre, extent = x.mean(axis=0), max(float(np.abs(x - x.mean(axis=0)).max()), 1e-300)
    field = jnp.asarray(quadratic(x, centre, extent))
    bvals = jnp.asarray(quadratic(np.asarray(geom.face.centroid), centre, extent))
    exact = quadratic_gradient(x, centre, extent)

    inner = SweptGradientSolve(sweeps=INNER, warn_tol=None)
    errors = {}
    for label, flag in (("schur", True), ("a_gg", False)):
        grad = np.asarray(
            HessianCorrectedGradient(hessian_solver=inner, local_schur_block=flag).gradients(
                field, mesh, geom, bvals
            )
        )
        errors[label] = np.linalg.norm(grad - exact, axis=-1) / np.linalg.norm(exact, axis=-1)
        print(
            f"  {label:5s} median {np.median(errors[label]):.3e}  "
            f"p99 {np.percentile(errors[label], 99):.3e}  max {errors[label].max():.3e}",
            flush=True,
        )

    # HOW MANY cells are bad, not just how bad the worst is -- one pathological cell is a guard, a
    # population is a design fault, and the max alone cannot tell those apart.
    bad = errors["schur"] > 1.0
    print(f"\n  cells above 100% error under the Schur block: {int(bad.sum())} of {mesh.n_cells}")
    print(f"  the same cells under the A_gg block: max {errors['a_gg'][bad].max():.3e}", flush=True)

    worst = np.argsort(errors["schur"])[::-1][:WORST]
    volume = np.asarray(geom.cell.volume)
    planarity = np.asarray(face_planarity(mesh))
    closure = np.asarray(closed_cell_residual(mesh))
    owner, neighbour = np.asarray(mesh.face_cells.owner), np.asarray(mesh.face_cells.neighbour)
    faces_per_cell = np.bincount(owner, minlength=mesh.n_cells) + np.bincount(
        neighbour[neighbour >= 0], minlength=mesh.n_cells
    )
    # worst planarity among each cell's own faces -- a cell-level view of a face-level metric
    worst_planarity = np.ones(mesh.n_cells)
    np.minimum.at(worst_planarity, owner, planarity)
    np.minimum.at(worst_planarity, neighbour[neighbour >= 0], planarity[neighbour >= 0])

    print("\n  the worst cells, and how they differ from the mesh as a whole")
    print(
        f"    {'cell':>9} {'err schur':>10} {'err a_gg':>10} {'volume':>10} "
        f"{'vol/med':>9} {'faces':>6} {'planarity':>10} {'closure':>10}"
    )
    for c in worst:
        print(
            f"    {c:9d} {errors['schur'][c]:10.2e} {errors['a_gg'][c]:10.2e} {volume[c]:10.2e} "
            f"{volume[c] / np.median(volume):9.2e} {faces_per_cell[c]:6d} "
            f"{worst_planarity[c]:10.4f} {closure[c]:10.2e}",
            flush=True,
        )
    # THE DECISIVE MEASUREMENT. `C` -- the Hessian equation's per-cell block -- is the only new
    # ingredient the Schur correction brings: `A_gH C^-1 A_Hg`. `A_gg`'s block never forms `C^-1`,
    # which is why it survives where this does not. A 3x3 Hessian has six independent components, so
    # a cell with only four faces cannot determine one, and `C` should be near-singular exactly there.
    systems = HessianCorrectedGradient._systems(mesh, geom)
    inner_system = systems.inner()
    hessian_block = np.linalg.inv(np.asarray(inner_system.preconditioner.inverse))
    plain_block = np.linalg.inv(
        np.asarray(systems.outer(inner, inner_system, False).preconditioner.inverse)
    )
    schur_block = np.linalg.inv(
        np.asarray(systems.outer(inner, inner_system, True).preconditioner.inverse)
    )
    cond_c = np.linalg.cond(hessian_block)
    cond_plain = np.linalg.cond(plain_block)
    cond_schur = np.linalg.cond(schur_block)
    print("\n  conditioning: is `C` the culprit?")
    print(f"    {'cell':>9} {'faces':>6} {'cond C':>11} {'cond A_gg':>11} {'cond Schur':>11}")
    for c in worst[:8]:
        print(
            f"    {c:9d} {faces_per_cell[c]:6d} {cond_c[c]:11.3e} "
            f"{cond_plain[c]:11.3e} {cond_schur[c]:11.3e}",
            flush=True,
        )
    print(
        f"    mesh-wide medians: cond C {np.median(cond_c):.3e} | "
        f"cond A_gg {np.median(cond_plain):.3e} | cond Schur {np.median(cond_schur):.3e}"
    )
    for n_faces in (4, 5, 6):
        pick = faces_per_cell == n_faces
        if pick.sum():
            print(
                f"    cells with {n_faces} faces (n={int(pick.sum())}): "
                f"cond C median {np.median(cond_c[pick]):.3e} max {cond_c[pick].max():.3e}",
                flush=True,
            )
    # ⚠️ THE PRECONDITIONER-INDEPENDENT CHECK, and the one that discriminates. Conditioning does not
    # explain the failures: the three worst cells carry a badly conditioned Schur block, but others
    # fail just as hard with a perfectly conditioned one. A block can be well conditioned and simply
    # WRONG -- a poor approximation of the true local Schur complement makes `I - P^-1 S` expansive on
    # that row, and twenty sweeps of a modest amplification is an enormous number.
    #
    # Lighting ONE cell's degree of freedom and reading back THAT cell's own row gives the true block:
    # no neighbour can contribute to it, so it is the truth whatever preconditioner is in force. Done
    # for a handful of named cells it costs `dim` operator applies each, rather than the `n * dim` a
    # whole-mesh extraction would.
    outer_system = systems.outer(inner, inner_system, True)
    print(
        "\n  is the block WRONG rather than ill-conditioned? (relative error vs the true S block)"
    )
    print(f"    {'cell':>9} {'faces':>6} {'A_gg block':>12} {'Schur block':>12} {'cond Schur':>11}")
    basis = np.eye(dim := mesh.dim)
    for c in worst[:8]:
        columns = []
        for k in range(dim):
            probe = jnp.zeros((mesh.n_cells, dim)).at[c, k].set(1.0)
            columns.append(np.asarray(outer_system.operator(probe))[c])
        true_block = np.stack(columns, axis=-1)
        scale = np.linalg.norm(true_block)
        err_plain = np.linalg.norm(plain_block[c] - true_block) / scale
        err_schur = np.linalg.norm(schur_block[c] - true_block) / scale
        print(
            f"    {c:9d} {faces_per_cell[c]:6d} {err_plain:12.3e} {err_schur:12.3e} "
            f"{cond_schur[c]:11.3e}",
            flush=True,
        )
    _ = basis

    print(
        f"\n  mesh-wide: volume median {np.median(volume):.2e} min {volume.min():.2e} | "
        f"faces/cell median {int(np.median(faces_per_cell))} max {faces_per_cell.max()} | "
        f"planarity min {planarity.min():.4f} | closure max {closure.max():.2e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
