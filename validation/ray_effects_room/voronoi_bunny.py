#!/usr/bin/env python3
"""
Generate a Voronoi-perforated shell ("lattice bunny") STL for radiation
shadow demos.

Method (no mesh booleans, so it is robust to non-watertight inputs such as
the raw Stanford bunny):
  1. Voxelize the input surface and build a smoothed unsigned distance
     field with an EDT (no inside/outside test needed).
  2. Sample seed points evenly on the surface. For every voxel near the
     surface, find the two nearest seeds; (d2 - d1)/2 approximates the
     distance to the Voronoi cell boundary.
  3. Keep voxels that are within the shell (thickness t) AND near a cell
     boundary (strut width w). Contour with marching cubes -> STL.

Units: metres. The mesh is scaled so its largest extent equals --size.

Examples:
  python voronoi_bunny.py                           # downloads bunny, 0.6 m tall
  python voronoi_bunny.py --cell 0.07 --strut 0.015 # bigger holes (ears get sparse)
  python voronoi_bunny.py --input my_animal.stl --size 0.8
"""
import argparse, io, tarfile, urllib.request
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure

BUNNY_URL = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
BUNNY_MEMBER = "bunny/reconstruction/bun_zipper.ply"


def load_bunny(cache=Path("bun_zipper.ply")):
    if not cache.exists():
        print(f"Downloading Stanford bunny from {BUNNY_URL} ...")
        data = urllib.request.urlopen(BUNNY_URL).read()
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            cache.write_bytes(tf.extractfile(BUNNY_MEMBER).read())
    return trimesh.load(cache, force="mesh")


def surface_distance_grid(mesh, pitch, pad):
    """Unsigned distance to the surface on a voxel grid.

    Using an unsigned field means the input need not be watertight: the shell
    is simply a band of thickness t centred on the surface (open edges, like
    the holes in the raw bunny's base, stay open).
    """
    vox = mesh.voxelized(pitch)            # surface voxels only
    surf = np.pad(vox.matrix, pad)
    udf = ndimage.distance_transform_edt(~surf) * pitch
    udf = ndimage.gaussian_filter(udf, sigma=1.0)
    origin = vox.transform[:3, 3] - pad * pitch  # world coords of index (0,0,0)
    return udf, origin


def poisson_disk_surface(mesh, spacing, seed):
    """Greedy Poisson-disk seeds on the surface with minimum spacing ~ spacing."""
    rng = np.random.default_rng(seed)
    n_cand = int(40 * mesh.area / spacing**2)
    cand, _ = trimesh.sample.sample_surface(mesh, n_cand, seed=seed)
    cand = cand[rng.permutation(len(cand))]
    tree = cKDTree(cand)
    alive = np.ones(len(cand), bool)
    keep = []
    for i in range(len(cand)):
        if alive[i]:
            keep.append(i)
            alive[tree.query_ball_point(cand[i], spacing)] = False
    return cand[keep]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, help="input mesh (default: Stanford bunny)")
    p.add_argument("--output", type=Path, default=Path("voronoi_bunny.stl"))
    p.add_argument("--size", type=float, default=0.6, help="largest extent [m]")
    p.add_argument("--cell", type=float, default=0.045,
                   help="target Voronoi cell (hole) spacing [m]")
    p.add_argument("--strut", type=float, default=0.012, help="strut width [m]")
    p.add_argument("--shell", type=float, default=0.010, help="shell thickness [m]")
    p.add_argument("--pitch", type=float, default=None,
                   help="voxel size [m] (default: strut/4)")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    pitch = a.pitch or a.strut / 4
    mesh = load_bunny() if a.input is None else trimesh.load(a.input, force="mesh")

    # Normalize: centre, scale, stand upright with z up and base at z = 0
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale(a.size / mesh.extents.max())
    if a.input is None:  # Stanford bunny is y-up
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    mesh.apply_translation([0, 0, -mesh.bounds[0, 2]])

    print(f"Building distance field (pitch {pitch*1e3:.1f} mm) ...")
    pad = int(np.ceil(a.shell / pitch)) + 4
    udf, origin = surface_distance_grid(mesh, pitch, pad)

    # Seeds evenly spread over the surface (Poisson-disk-like)
    seeds = poisson_disk_surface(mesh, a.cell, a.seed)
    print(f"{len(seeds)} Voronoi cells")

    # Only evaluate the Voronoi field near the shell to save time/memory
    band = udf < a.shell
    idx = np.argwhere(band)
    pts = origin + idx * pitch
    d, _ = cKDTree(seeds).query(pts, k=2)
    edge = 0.5 * (d[:, 1] - d[:, 0]) - 0.5 * a.strut  # <0 on struts

    # Implicit field: intersection of shell and strut regions (negative = solid)
    shell = udf - 0.5 * a.shell
    field = np.full(udf.shape, a.strut)  # empty
    field[band] = np.maximum(shell[band], edge)
    field = ndimage.gaussian_filter(field, sigma=0.6)

    verts, faces, _, _ = measure.marching_cubes(field, level=0.0, spacing=(pitch,) * 3)
    out = trimesh.Trimesh(verts + origin, faces, process=True)
    # Keep the largest connected piece (drop floating strut fragments)
    parts = out.split(only_watertight=False)
    out = max(parts, key=lambda m: len(m.faces))
    trimesh.repair.fix_normals(out)

    out.export(a.output)
    ext = out.extents
    print(f"Wrote {a.output}: {len(out.faces)} faces, "
          f"{ext[0]:.3f} x {ext[1]:.3f} x {ext[2]:.3f} m, "
          f"watertight={out.is_watertight}, dropped {len(parts)-1} fragments")


if __name__ == "__main__":
    main()
