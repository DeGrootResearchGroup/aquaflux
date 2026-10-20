"""How the polyhedral dual of Netgen tetrahedra scales, and how non-convex its cells get.

Four solids, written by OCP and meshed by Netgen in its child process: a cylinder (curved walls,
convex edges); a box with a lamp-like cylindrical hole through it (a gently concave curved wall);
an L-shaped block (one straight sharp concave edge, whose dual cells are cut into pieces); and the
Sozzi reactor's water (two pipe junctions: curved sharp concave edges, and the lamp). For
each ``maxh`` the harness prints the point count, the time to build the dual, its worst cell
closure, its volume against the tetrahedra's, and a convexity measure: for every dual cell, the
largest distance by which one of its own vertices lies outside the plane of one of its faces
(planes through each face's centroid, normal to its area vector), relative to the cell's size
(the cube root of its volume), and separately over the pieces of cut cells. Zero for a convex cell with planar faces; the faces of a dual of
tetrahedra are not planar, so a small positive value is the face warp, not a fold. Reported
separately for the cells whose point lies on the hole's wall.

Run in an environment with ``cadquery-ocp-novtk`` and ``netgen-mesher``::

    python validation/meshing_dual/dual_scale_convexity.py 0.02 0.01 0.006

``sozzi`` alone, or any subset of the names, may be given first: ``... sozzi 0.02 0.01``.
"""

from __future__ import annotations

import math
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from aquaflux.io.cad import CadRegion, Placement, read_step
from aquaflux.mesh import cell_non_convexity, closed_cell_residual
from aquaflux.meshing import NetgenProcess, PolyhedralDual, element_mesh
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.BRepTools import BRepTools
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

DUAL = PolyhedralDual(feature_angle=math.radians(45))


def _solids(directory: pathlib.Path) -> dict[str, pathlib.Path]:
    shapes = {
        "cylinder": BRepPrimAPI_MakeCylinder(0.05, 0.3).Shape(),
        "sleeve": BRepAlgoAPI_Cut(
            BRepPrimAPI_MakeBox(gp_Pnt(-0.05, -0.05, 0), 0.1, 0.1, 0.2).Shape(),
            BRepPrimAPI_MakeCylinder(
                gp_Ax2(gp_Pnt(0, 0, -0.01), gp_Dir(0, 0, 1)), 0.01, 0.22
            ).Shape(),
        ).Shape(),
        "L": BRepAlgoAPI_Fuse(
            BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 0.2, 0.1, 0.1).Shape(),
            BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0.1), 0.1, 0.1, 0.1).Shape(),
        ).Shape(),
    }
    paths = {}
    for name, shape in shapes.items():
        paths[name] = directory / f"{name}.brep"
        BRepTools.Write_s(shape, str(paths[name]))
    drawing = (
        pathlib.Path(__file__).resolve().parents[1]
        / "uvreactor_openfoam/of_case/SozziTaghipour.step"
    )
    model = read_step(drawing, Placement(matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
    water = CadRegion(("reactor_body", "inlet_pipe", "outlet_pipe"), without=("lamp",))
    paths["sozzi"] = model.write_brep(water, directory / "sozzi.brep")
    return paths


def main(names: list[str], sizes: list[float]) -> None:
    netgen = NetgenProcess()
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
                start = time.perf_counter()
                try:
                    dual = DUAL.build(primal)
                except (NotImplementedError, ValueError) as refusal:
                    print(
                        f"{name:8s} maxh={maxh:<6g} points={primal.n_nodes:8d} refused: {refusal}"
                    )
                    continue
                built = time.perf_counter() - start
                volume = np.asarray(dual.geometry().cell.volume)
                tets = np.asarray(primal.geometry().cell.volume).sum()
                closure = float(np.max(np.asarray(closed_cell_residual(dual))))
                line = (
                    f"{name:8s} maxh={maxh:<6g} points={primal.n_nodes:8d} "
                    f"tets={primal.n_cells:9d} dual faces={dual.n_faces:9d} build={built:7.2f}s "
                    f"closure={closure:.1e} volume err={abs(volume.sum() / tets - 1):.1e}"
                )
                if primal.n_nodes <= 60000:
                    bad = cell_non_convexity(dual)
                    line += f" non-convexity p50={np.median(bad):.3f} max={bad.max():.3f}"
                    # A cut point's first piece keeps its index; the further pieces follow.
                    cut = dual.n_cells - primal.n_nodes
                    if cut:
                        line += f" | {cut} extra piece(s), max={bad[primal.n_nodes :].max():.3f}"
                    if name == "sleeve":
                        points = np.asarray(primal.node_coords)
                        near = np.hypot(points[:, 0], points[:, 1]) < 0.0105
                        line += f" | on the hole ({near.sum()} cells) max={bad[: primal.n_nodes][near].max():.3f}"
                print(line, flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    main(
        [a for a in arguments if not a[0].isdigit()],
        [float(a) for a in arguments if a[0].isdigit()],
    )
