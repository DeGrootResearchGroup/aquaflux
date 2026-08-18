"""Per-face ``D_P.n`` census for an imported OpenFOAM mesh: is the diffusion flux's normal-distance
denominator strictly positive everywhere, and if not, where?

``D_P.n`` (``aquaflux/discretization/diffusion.py``'s ``dpn``) is the projection of the
owner-centroid-to-face-centroid displacement onto the face's own outward normal. A well-formed mesh
has it strictly positive on every face; a negative value flips the sign of the flux's non-orthogonal
correction and can drive the AMG preconditioner's diagonal-positivity guard to raise on setup. This
script computes the exact quantity directly from the mesh's own geometry (not a proxy for it) and
reports where any bad faces sit -- by patch, and by the interior faces' location -- so a mesh problem
can be diagnosed before it ever reaches a solve.

Usage
-----
    python3 validation/uvreactor_openfoam/dpn_diagnostic.py <path-to-polyMesh-dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import aquaflux  # noqa: E402,F401  (enables x64)
import numpy as np  # noqa: E402
from aquaflux.io import read_openfoam  # noqa: E402
from aquaflux.vectors import dot  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <path-to-polyMesh-dir>")
    mesh_path = Path(sys.argv[1]).resolve()

    mesh = read_openfoam(mesh_path)
    geom = mesh.geometry()
    fc = mesh.face_cells

    fg = geom.face
    x_cell = np.asarray(geom.cell.centroid)
    n = np.asarray(fg.normal)
    x_ip = np.asarray(fg.centroid)

    d_p = x_ip - x_cell[np.asarray(fc.owner)]
    dpn = np.asarray(dot(d_p, n))

    interior = np.asarray(fc.interior)
    bad_interior = interior & (dpn <= 0)
    print(f"mesh: {mesh_path}")
    print(f"n_cells={mesh.n_cells}  n_interior_faces={int(interior.sum())}")
    print(
        f"D_P.n <= 0 (interior): {int(bad_interior.sum())} / {int(interior.sum())} "
        f"({100 * bad_interior.sum() / max(interior.sum(), 1):.3f}%)"
    )
    if bad_interior.sum():
        pts = x_ip[bad_interior]
        print("  bad-interior-face centroid bbox:")
        for axis, label in enumerate("xyz"):
            print(f"    {label}: [{pts[:, axis].min():.4f}, {pts[:, axis].max():.4f}]")

    boundary = ~interior
    bad_boundary = boundary & (dpn <= 0)
    print(f"D_P.n <= 0 (boundary): {int(bad_boundary.sum())} / {int(boundary.sum())}")
    labels = np.asarray(mesh.face_patches.label)
    for i, name in enumerate(mesh.face_patches.names):
        patch_mask = labels == i
        n_patch = int((patch_mask & boundary).sum())
        n_bad = int((patch_mask & bad_boundary).sum())
        if n_patch:
            print(f"  patch '{name}': {n_bad}/{n_patch} bad ({100 * n_bad / n_patch:.3f}%)")


if __name__ == "__main__":
    main()
