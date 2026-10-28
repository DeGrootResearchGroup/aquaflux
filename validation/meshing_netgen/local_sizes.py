"""Do the local sizes handed to Netgen make its tetrahedra and triangles that small, and how far does
the refinement spread?

A unit box wall of triangles a quarter across, filled to ``max_size`` 0.25 with a ball of radius
0.15 at its centre refined to ``size`` (a :class:`~aquaflux.meshing.RefinedSphere`'s local sizes);
and a ``[0, 1] x [0, 2] x [0, 3]`` box solid triangulated to 0.5 with the same kind of ball, of
radius 0.2, centred on its face ``x = 0``. The harness prints the median tetrahedron edge in shells
by distance from the ball's centre, without and with the local sizes, and the median triangle edge
near the spot and away from it -- the measurement behind the mesher's refinement regions: the size
holds inside the part, and Netgen's grading spreads it outward from there.

Run in an environment with ``netgen-mesher``::

    python validation/meshing_netgen/local_sizes.py 0.04
"""

from __future__ import annotations

import itertools
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from aquaflux.meshing import NetgenProcess, RefinedSphere
from tests.support.meshes import box_surface

SHELLS = (0.0, 0.15, 0.25, 0.35, 0.5, 0.87)


def _edges(points, cells):
    k = cells.shape[1]
    pairs = np.concatenate([cells[:, [i, j]] for i in range(k) for j in range(i + 1, k)])
    middle = points[pairs].mean(axis=1)
    return middle, np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)


def fill(size: float) -> None:
    points, triangles = box_surface(0.0, 1.0, 4, into_box=False)
    ball = RefinedSphere(centre=(0.5, 0.5, 0.5), radius=0.15, size=size).located({})
    sizes = ball.local_sizes([0, 0, 0], [1, 1, 1])
    print(f"fill: {len(sizes.sizes)} local sizes of {size}")
    for name, local in (("plain", None), ("refined", sizes)):
        filled = NetgenProcess().fill(points, triangles, max_size=0.25, local_sizes=local)
        middle, length = _edges(filled.points, filled.tetrahedra)
        radius = np.linalg.norm(middle - 0.5, axis=1)
        shells = [
            f"{np.median(length[(radius >= a) & (radius < b)]):.3f}"
            for a, b in itertools.pairwise(SHELLS)
        ]
        print(f"  {name:8s} tets {len(filled.tetrahedra):6d}  median edge by radius {shells}")


def triangulate(size: float) -> None:
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepTools import BRepTools
    from OCP.gp import gp_Pnt

    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "box.brep"
        BRepTools.Write_s(
            BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), 1.0, 2.0, 3.0).Shape(), str(path)
        )
        spot = RefinedSphere(centre=(0.0, 1.0, 1.5), radius=0.2, size=size).located({})
        sizes = spot.local_sizes([0, 0, 0], [1, 2, 3])
        for name, local in (("plain", None), ("refined", sizes)):
            surface = NetgenProcess().triangulate(path, max_size=0.5, local_sizes=local)
            middle, length = _edges(surface.points, surface.triangles)
            near = np.linalg.norm(middle - [0.0, 1.0, 1.5], axis=1) < 0.2
            close = f"{np.median(length[near]):.3f}" if near.any() else "none there"
            print(
                f"triangles {name:8s} {len(surface.triangles):5d}  median edge near {close}, "
                f"away {np.median(length[~near]):.3f}"
            )


if __name__ == "__main__":
    size = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04
    fill(size)
    triangulate(size)
