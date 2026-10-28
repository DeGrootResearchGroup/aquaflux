"""Cell anisotropy of the two validated meshes, read from geometry alone -- no case, no solve.

The aspect-ratio sweep says a zero-fill factorization degrades with cell anisotropy while a
level-1 one does not.  That mechanism only explains the pitzDaily/bfs3d split if the two meshes
differ on this axis, so measure it directly.  Reading the mesh costs a fraction of building the
coupled case, which is what makes this affordable on bfs3d.

Three measures, because the crude one can mislead:
  face-centroid ratio   max/min distance from the cell centroid to its own face centroids
  spacing ratio         max/min of the normal distance |d_f . n_f| across a cell's faces -- an aspect
                        ratio read the way a face flux sees it
  conductance ratio     max/min of |S_f| / |d_f . n_f| over a cell's faces -- the ratio of the
                        diffusive couplings a transport operator actually assembles, which is
                        what an incomplete factorization has to reproduce (roughly the square of
                        the spacing ratio on a stretched hexahedron)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Run directly from anywhere: resolve the repository root from THIS file rather than pinning a
# checkout, so the harness survives being run from another worktree.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import aquaflux  # noqa: F401,E402
import jax.numpy as jnp  # noqa: E402
from aquaflux.io import read_openfoam  # noqa: E402
from aquaflux.schemes.interpolation import interpolation_factor  # noqa: E402
from aquaflux.vectors import dot, scale  # noqa: E402

MESHES = {
    "pitzDaily": ROOT / "validation/pitzdaily_openfoam/runs/kwsst/polyMesh",
    "bfs3d": ROOT / "validation/bfs3d_openfoam/of_case/constant/polyMesh",
}


def _spread_over_faces(mesh, per_face):
    """Per cell, max/min of a positive per-face quantity over the cell's faces, shape ``(n_cells,)``."""
    fc = mesh.face_cells
    owner, neighbour = np.asarray(fc.owner), np.asarray(fc.neighbour)
    interior = np.asarray(fc.interior)
    lo, hi = np.full(mesh.n_cells, np.inf), np.zeros(mesh.n_cells)
    for cells, faces in (
        (owner, np.arange(owner.size)),
        (neighbour[interior], np.flatnonzero(interior)),
    ):
        np.minimum.at(lo, cells, per_face[faces])
        np.maximum.at(hi, cells, per_face[faces])
    return hi / np.maximum(lo, 1e-300)


def _normal_distance(mesh, geom):
    """``|d_f . n_f|`` per face: owner to neighbour centroid, or owner to face centroid on a boundary."""
    fc = mesh.face_cells
    x_p = geom.cell.centroid[fc.owner]
    across = np.asarray(dot(fc.neighbour_centroid(geom.cell.centroid) - x_p, geom.face.normal))
    to_face = np.asarray(dot(geom.face.centroid - x_p, geom.face.normal))
    return np.maximum(np.abs(np.where(np.asarray(fc.interior), across, to_face)), 1e-300)


def spacing_ratio(mesh, geom):
    """Per cell, max/min of the normal distance ``|d_f . n_f|`` over its faces, shape ``(n,)``."""
    return _spread_over_faces(mesh, _normal_distance(mesh, geom))


def conductance_ratio(mesh, geom):
    """Per cell, max/min of the diffusive conductance ``|S_f| / |d_f . n_f|`` over its faces, shape ``(n,)``."""
    area = np.asarray(geom.face.area)  # one scalar per face
    return _spread_over_faces(mesh, area / _normal_distance(mesh, geom))


def report(name, path):
    mesh = read_openfoam(path)
    geom = mesh.geometry()
    fc = mesh.face_cells
    owner, nb = np.asarray(fc.owner), np.asarray(fc.neighbour)
    interior = np.asarray(fc.interior)
    centroid = np.asarray(geom.cell.centroid)
    fcentroid = np.asarray(geom.face.centroid)
    n = mesh.n_cells

    # skewness, for the record
    g = interpolation_factor(fc, geom)
    x_p = geom.cell.centroid[fc.owner]
    d = fc.neighbour_centroid(geom.cell.centroid) - x_p
    skew = np.asarray(
        jnp.linalg.norm(geom.face.centroid - (x_p + scale(d, g)), axis=-1)
        / jnp.linalg.norm(d, axis=-1)
    )[interior]

    # face-centroid distance ratio
    lo, hi = np.full(n, np.inf), np.zeros(n)
    for cells, faces in ((owner, np.arange(owner.size)), (nb[interior], np.flatnonzero(interior))):
        dist = np.linalg.norm(fcentroid[faces] - centroid[cells], axis=-1)
        np.minimum.at(lo, cells, dist)
        np.maximum.at(hi, cells, dist)
    ratio = hi / np.maximum(lo, 1e-300)

    print(f"\n{name}: {n} cells, {mesh.dim}D, {int(interior.sum())} interior faces")
    for label, q in (
        ("face-centroid ratio", ratio),
        ("spacing ratio", spacing_ratio(mesh, geom)),
        ("conductance ratio", conductance_ratio(mesh, geom)),
    ):
        print(
            f"  {label:<22} median {np.median(q):8.2f}  p90 {np.quantile(q, 0.90):9.2f}  "
            f"p99 {np.quantile(q, 0.99):9.2f}  max {q.max():10.2f}  "
            f"share > 10: {float((q > 10).mean()):.3f}  > 100: {float((q > 100).mean()):.3f}"
        )
    print(
        f"  {'skewness |s|/|d|':<22} median {np.median(skew):.3e}  max {skew.max():.3e}  "
        f"share > 1e-6: {float((skew > 1e-6).mean()):.3f}"
    )
    del mesh, geom


if __name__ == "__main__":
    for name, path in MESHES.items():
        if not path.exists():
            print(f"{name}: {path} missing")
            continue
        report(name, path)
