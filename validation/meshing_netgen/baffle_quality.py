"""What a baffle costs the dual's quality, against the same region without it and against the L.

The unit cube with a plate standing on its wall ``y = 0`` (the plane ``x = 1/2``, ``y`` in
``[0, 0.6]``, ``z`` in ``[0.2, 0.8]``), meshed by ``CadMesher`` at ``max_size`` with a feature angle
of 45 degrees, with and without the plate and with and without three layers on the walls
(``first_height = max_size / 20``, growth 1.3). The L of two unit boxes (the first concave edge the
dual was built for) is meshed the same way, for comparison: whatever the cut at a concave edge costs
there, it costs at a baffle's free edge too, which is the most concave fold there is. Each row
prints the cells, the worst and 99th-percentile cell non-convexity, and the interior faces'
non-orthogonality (worst, and how many exceed 50 and 90 degrees), with the count of faces smaller
than a hundredth of the median.

``--opened-fill`` instead hands Netgen the surface opened along the plate -- its two sides as two
coincident surfaces on their own points, the region bounded by them -- and reports whether the fill
finishes within ``--timeout`` seconds. It did not (2026-10-24): the mesher gives the fill the
plate's sides once, as internal triangles, instead.

Run in an environment with ``netgen-mesher`` and the CAD kernel::

    python validation/meshing_netgen/baffle_quality.py 0.1
    python validation/meshing_netgen/baffle_quality.py 0.1 --opened-fill --timeout 120
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from aquaflux.io.cad import CadModel, CadRegion
from aquaflux.io.cad.kernel import OpenCascade
from aquaflux.mesh import cell_non_convexity, face_non_orthogonality
from aquaflux.meshing import CadMesher, NetgenProcess, Polygons, PrismLayers
from aquaflux.meshing.netgen_process import CHILD_SCRIPT
from aquaflux.meshing.sheets import opened
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Ax3, gp_Dir, gp_Pln, gp_Pnt
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain


def plate_in_a_box() -> CadModel:
    plane = gp_Pln(gp_Ax3(gp_Pnt(0.5, 0.0, 0.2), gp_Dir(1, 0, 0), gp_Dir(0, 1, 0)))
    return CadModel(
        {"box": BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0).Shape()},
        OpenCascade(),
        sheets={"plate": BRepBuilderAPI_MakeFace(plane, 0.0, 0.6, 0.0, 0.6).Face()},
    )


def an_l() -> CadModel:
    fused = BRepAlgoAPI_Fuse(
        BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 2.0, 1.0, 1.0).Shape(),
        BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 1), 1.0, 1.0, 1.0).Shape(),
    ).Shape()
    unified = ShapeUpgrade_UnifySameDomain(fused)
    unified.Build()
    return CadModel({"l": unified.Shape()}, OpenCascade())


def quality(label: str, mesh) -> None:
    inner = np.asarray(mesh.face_cells.neighbour) >= 0
    angle = np.asarray(face_non_orthogonality(mesh))[inner]
    area = np.asarray(mesh.geometry().face.area)[inner]
    convexity = np.asarray(cell_non_convexity(mesh))
    print(
        f"{label:28s} cells {mesh.n_cells:6d}  non-convexity max {convexity.max():.3f} "
        f"p99 {np.percentile(convexity, 99):.3f}  non-orthogonality max {angle.max():6.1f} "
        f"> 50: {np.count_nonzero(angle > 50):3d}  > 90: {np.count_nonzero(angle > 90):3d}  "
        f"small faces {np.count_nonzero(area < 1e-2 * np.median(area)):3d}",
        flush=True,
    )


def table(size: float) -> None:
    model, l_model = plate_in_a_box(), an_l()
    for layered in (False, True):
        layers = PrismLayers(first_height=size / 20, growth=1.3, count=3) if layered else None
        mesher = CadMesher(max_size=size, feature_angle=np.radians(45), layers=layers)
        walls = ("walls",) if layered else ()
        suffix = " + layers" if layered else ""
        quality("box" + suffix, mesher.mesh(model, CadRegion(("box",), baffles=()), layered=walls))
        quality("box and plate" + suffix, mesher.mesh(model, CadRegion(("box",)), layered=walls))
        quality("L" + suffix, mesher.mesh(l_model, CadRegion(("l",)), layered=walls))


def opened_fill(size: float, timeout: float) -> None:
    model, region = plate_in_a_box(), CadRegion(("box",))
    topology = model.topology(region)
    with tempfile.TemporaryDirectory() as directory:
        work = pathlib.Path(directory)
        surface = NetgenProcess().triangulate(model.write_brep(region, work / "r.brep"), max_size=size)
        result = opened(
            surface.points, Polygons(surface.triangles), surface.face,
            np.flatnonzero(topology.sheet),
            lambda points, faces: model.project(region, points, faces=faces).points,
        )  # fmt: skip
        print(
            f"opened surface: {result.points.shape[0]} points ({surface.points.shape[0]} before), "
            f"{result.faces.n_faces} triangles",
            flush=True,
        )
        request = {"operation": "fill", "max_size": size, "grading": 0.3}
        (work / "request.json").write_text(json.dumps(request))
        np.savez(
            work / "input.npz", points=result.points, triangles=result.faces.triangles,
            internal=np.zeros((0, 3), np.int64),
        )  # fmt: skip
        try:
            ran = subprocess.run(
                [sys.executable, "-P", str(CHILD_SCRIPT), str(work)],
                capture_output=True, text=True, check=False, timeout=timeout,
            )  # fmt: skip
            print(f"fill finished, exit status {ran.returncode}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"fill did not finish within {timeout:g} s (stopped)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("size", type=float)
    parser.add_argument("--opened-fill", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    arguments = parser.parse_args()
    if arguments.opened_fill:
        opened_fill(arguments.size, arguments.timeout)
    else:
        table(arguments.size)
