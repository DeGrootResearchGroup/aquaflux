"""How much moving the dual's interior vertices flattens its faces, on duals of Netgen tetrahedra.

The solids of ``dual_scale_convexity.py`` (cylinder, sleeve box, L, Sozzi water) are meshed by
Netgen in its child process and dualized with the same feature angle. The vertices moved are the
primal cell centroids -- every interior vertex except those the concave cut placed on the surface
between pieces; the boundary, its features and the cut stay. For each case and ``maxh`` the
harness prints, before and after ``FacePlanarizer``: the total warp it minimizes; face warp (the
largest out-of-plane vertex height over the square root of the face's area) on interior and on
boundary faces; cell non-convexity (largest vertex excursion beyond a face plane over the cell
size); the cells ``checkMesh`` would count concave by face planes; the largest change in a cell's
volume that re-triangulating its faces could make; non-orthogonality (maximum, and the average as
``checkMesh`` takes it, the angle of the mean cosine) and skewness; and the worst closure.

Run in an environment with ``cadquery-ocp-novtk`` and ``netgen-mesher``::

    python validation/meshing_dual/planarization.py L cylinder 0.01 0.005

Names and sizes as in ``dual_scale_convexity.py``; ``--sweeps N`` caps the sweeps (default 300).
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from aquaflux.mesh import (
    CONCAVE_COSINE,
    Mesh,
    cell_concavity,
    cell_non_convexity,
    closed_cell_residual,
    face_non_orthogonality,
    face_skewness,
    face_warp,
    triangulation_volume_change,
)
from aquaflux.meshing import FacePlanarizer, NetgenProcess, element_mesh
from dual_scale_convexity import DUAL, _solids


def describe(mesh: Mesh) -> str:
    interior = np.asarray(mesh.face_cells.interior)
    warp = np.asarray(face_warp(mesh))
    convexity = cell_non_convexity(mesh)
    volume = np.asarray(triangulation_volume_change(mesh))
    angle = np.asarray(face_non_orthogonality(mesh))[interior]
    average = np.degrees(np.arccos(np.cos(np.radians(angle)).mean()))
    return (
        f"warp int p99 {np.quantile(warp[interior], 0.99):.3f} max {warp[interior].max():.3f} "
        f"bnd p99 {np.quantile(warp[~interior], 0.99):.3f} max {warp[~interior].max():.3f} | "
        f"non-convexity p99 {np.quantile(convexity, 0.99):.3f} max {convexity.max():.3f} | "
        f"concave {int((cell_concavity(mesh) > CONCAVE_COSINE).sum())} | "
        f"tri-volume p99 {np.quantile(volume, 0.99):.2e} max {volume.max():.2e} | "
        f"non-orth max {angle.max():.1f} avg {average:.1f} skew {np.asarray(face_skewness(mesh)).max():.2f} | "
        f"closure {float(np.max(np.asarray(closed_cell_residual(mesh)))):.1e}"
    )


def main(names: list[str], sizes: list[float], sweeps: int) -> None:
    netgen = NetgenProcess()
    planarizer = FacePlanarizer(max_sweeps=sweeps)
    with tempfile.TemporaryDirectory() as directory:
        for name, path in _solids(pathlib.Path(directory)).items():
            if names and name not in names:
                continue
            for maxh in sizes:
                surface = netgen.triangulate(path, max_size=maxh)
                filled = netgen.fill(surface.points, surface.triangles, max_size=maxh)
                primal = element_mesh(
                    filled.points, filled.tetrahedra, patches={"walls": surface.triangles}
                )
                dual = DUAL.build(primal)
                # The dual numbers the primal cell centroids first among its vertices.
                movable = np.arange(dual.n_nodes) < primal.n_cells
                print(f"{name} maxh={maxh:g} cells={dual.n_cells} moved={int(movable.sum())}")
                print(f"  before: {describe(dual)}", flush=True)
                start = time.perf_counter()
                result = planarizer.planarize(dual, movable)
                spent = time.perf_counter() - start
                print(
                    f"  sweeps {result.sweeps} converged {result.converged} "
                    f"total warp {result.warp[0]:.3g} -> {result.warp[-1]:.3g} in {spent:.1f} s"
                )
                print(f"  after:  {describe(result.mesh)}", flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    cap = 300
    if "--sweeps" in arguments:
        at = arguments.index("--sweeps")
        cap = int(arguments[at + 1])
        del arguments[at : at + 2]
    main(
        [a for a in arguments if not a[0].isdigit()],
        [float(a) for a in arguments if a[0].isdigit()],
        cap,
    )
