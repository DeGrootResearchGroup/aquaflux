"""Build a structured hexahedral mesh of the SAME duct as ``make_mesh.py``, for a like-for-like comparison.

The tetrahedral duct's meshes are non-orthogonal and skewed; this meshes the identical geometry (0.25 x
0.025 x 0.025 m, patches ``inlet`` / ``outlet`` / ``walls``) with an orthogonal transfinite hexahedral
mesh at a matching cell size, so any difference in how the gradient schemes march is the mesh's cell
shape and nothing else. Cells at the duct's edges own two boundary faces, as the tetrahedral mesh's
corner cells do.

Usage::

    python3 make_hex_mesh.py OUTDIR [NX NY NZ]      # default 60 6 6: 4.2 mm cubes, 2160 cells

writes ``OUTDIR/duct.msh`` (Gmsh MSH 2.2, the format ``gmshToFoam`` reads). Convert with the same Docker
step as the tetrahedral mesh, after copying ``system/controlDict`` next to it.
"""

import os
import sys

import gmsh

LX, LY, LZ = 0.25, 0.025, 0.025
TOL = 1e-6 * max(LX, LY, LZ)


def classify(tag):
    xmin, _y0, _z0, xmax, _y1, _z1 = gmsh.model.getBoundingBox(2, tag)
    if abs(xmin) < TOL and abs(xmax) < TOL:
        return "inlet"
    if abs(xmin - LX) < TOL and abs(xmax - LX) < TOL:
        return "outlet"
    return "walls"


def main(out_dir: str, nx: int, ny: int, nz: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("hexahedral_duct")
    box = gmsh.model.occ.addBox(0.0, 0.0, 0.0, LX, LY, LZ)
    gmsh.model.occ.synchronize()

    divisions = {0: nx, 1: ny, 2: nz}
    for _dim, tag in gmsh.model.getEntities(1):
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(1, tag)
        extent = [x1 - x0, y1 - y0, z1 - z0]
        axis = max(range(3), key=lambda a: extent[a])
        gmsh.model.mesh.setTransfiniteCurve(tag, divisions[axis] + 1)
    for _dim, tag in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(tag)
        gmsh.model.mesh.setRecombine(2, tag)
    gmsh.model.mesh.setTransfiniteVolume(box)

    groups = {"inlet": [], "outlet": [], "walls": []}
    for _dim, tag in gmsh.model.getBoundary([(3, box)], oriented=False):
        groups[classify(tag)].append(tag)
    for name, tags in groups.items():
        gmsh.model.addPhysicalGroup(2, tags, name=name)
    gmsh.model.addPhysicalGroup(3, [box], name="fluid")

    gmsh.model.mesh.generate(3)
    n_hex = len(gmsh.model.mesh.getElementsByType(5)[0])
    print(f"  {n_hex} hexahedra ({nx} x {ny} x {nz} requested)")
    if n_hex != nx * ny * nz:
        raise RuntimeError("the transfinite mesh did not come out fully hexahedral")
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    path = os.path.join(out_dir, "duct.msh")
    gmsh.write(path)
    print(f"  wrote {path}")
    gmsh.finalize()


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    counts = [int(a) for a in args[1:4]] or [60, 6, 6]
    main(args[0], *counts)
