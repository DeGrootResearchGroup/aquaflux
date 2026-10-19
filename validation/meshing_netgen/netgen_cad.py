"""What Netgen does with a fluid OCP wrote: read it, triangulate its faces, mesh its volume.

Runs in a process that never imports OCP. Prints one line per step: the solids and faces Netgen
reads from each file, the surface triangulation (and that every triangle names its CAD face), and
the tet mesh's size and wall time at each ``maxh`` (metres).

Usage::

    python validation/meshing_netgen/netgen_cad.py <stem written by write_fluid.py> 0.01 0.006 0.004
"""

import sys
import time

import numpy as np
from netgen.meshing import MeshingStep
from netgen.occ import OCCGeometry


def main(stem: str, sizes: list[float]) -> None:
    for extension in (".brep", ".step"):
        shape = OCCGeometry(stem + extension).shape
        print(f"{extension}: {len(shape.solids)} solids, {len(shape.faces)} faces", flush=True)

    start = time.perf_counter()
    surface = OCCGeometry(stem + ".brep").GenerateMesh(
        maxh=sizes[0], perfstepsend=MeshingStep.MESHSURFACE
    )
    elements = surface.Elements2D()
    faces = np.array([element.index for element in elements])
    print(
        f"surface at maxh={sizes[0]}: {len(elements)} triangles naming {len(np.unique(faces))} "
        f"faces, {time.perf_counter() - start:.1f}s",
        flush=True,
    )

    for maxh in sizes:
        start = time.perf_counter()
        mesh = OCCGeometry(stem + ".brep").GenerateMesh(maxh=maxh)
        print(
            f"volume at maxh={maxh}: {mesh.ne} tets, {len(mesh.Points())} points, "
            f"{time.perf_counter() - start:.1f}s",
            flush=True,
        )
        del mesh


if __name__ == "__main__":
    main(sys.argv[1], [float(size) for size in sys.argv[2:]])
