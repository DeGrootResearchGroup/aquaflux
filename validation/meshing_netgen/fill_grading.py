"""Does Netgen's fill grade its tetrahedra from the sizes of the triangles it is given?

A unit box wall of fine triangles (edge ``1 / n_wall``) round a cubic hole ``[0.3, 0.7]^3`` of
coarse ones (edge ``0.4 / n_hole``), filled with ``max_size`` the hole's edge, at two gradings. The
harness prints the median tetrahedron edge in shells by distance from the outer wall: if the fill
sizes itself from the surface it is given, the edge grows from about the wall's to about the hole's
across the gap, which is what lets the mesher leave the grading between a wall and the octree core's
cubes to Netgen, with only its largest tetrahedron set.

Run in an environment with ``netgen-mesher``::

    python validation/meshing_netgen/fill_grading.py 40 4
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from aquaflux.meshing import NetgenProcess
from tests.support.meshes import box_surface


def main(n_wall: int, n_hole: int) -> None:
    outer, outer_triangles = box_surface(0.0, 1.0, n_wall, into_box=False)
    hole, hole_triangles = box_surface(0.3, 0.7, n_hole, into_box=True, offset=len(outer))
    hole_size = 0.4 / n_hole
    for grading in (0.3, 0.5):
        filled = NetgenProcess().fill(
            np.vstack([outer, hole]),
            np.vstack([outer_triangles, hole_triangles]),
            max_size=hole_size,
            grading=grading,
        )
        corner = filled.points[filled.tetrahedra]
        centre = corner.mean(axis=1)
        depth = np.min(np.minimum(centre, 1.0 - centre), axis=1)
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        edge = np.mean([np.linalg.norm(corner[:, a] - corner[:, b], axis=1) for a, b in pairs], 0)
        print(
            f"wall edge {1 / n_wall:.4f}  hole edge {hole_size:.3f}  grading {grading}  "
            f"tetrahedra {len(filled.tetrahedra)}",
            flush=True,
        )
        for low in np.arange(0.0, 0.3, 0.025):
            shell = (depth >= low) & (depth < low + 0.025)
            if shell.any():
                print(
                    f"  depth {low:.3f}-{low + 0.025:.3f}: median edge {np.median(edge[shell]):.4f}"
                    f"  n {int(shell.sum())}",
                    flush=True,
                )


if __name__ == "__main__":
    arguments = [int(argument) for argument in sys.argv[1:]]
    main(*(arguments or [40, 4]))
