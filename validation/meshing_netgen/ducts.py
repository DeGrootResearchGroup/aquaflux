"""Rectangular ducts through the CAD mesher: a straight one, a bend, a sweep along a spline, a tank.

Each drawing is made by OCP and meshed by ``CadMesher`` with its defaults (ducts found by
themselves); each line reports the cells, the cells' closure (the largest residual over the
largest face area), the share of six-faced cells, the mesh's quality summary, and, for a duct,
how far its rows' cross-sections tilt from square to its centreline (the plane through each ring
of the core against the line between its neighbours' centroids, in degrees). The cases:

- ``straight``: a 0.2 x 0.1 x 1 m box, open at both ends (inlet and outlet).
- ``bent``: 0.2 x 0.1 m, a 0.5 m leg, a quarter bend of centreline radius 0.4 m, another leg.
- ``sweep``: 0.1 x 0.06 m swept along a spline through five points by ``BRepOffsetAPI_MakePipeShell``;
  the profile is drawn in ``z = 0`` while the spline leaves the origin leaning, so its ends are cut
  obliquely and its corners fold by 73 to 125 degrees, not square.
- ``tank``: a 1 x 1 x 0.6 m box with a 0.2 x 0.2 x 0.5 m inlet duct on one wall and a lamp
  (radius 0.05 m) standing in it, so the tank is no duct; ``tank-noducts`` meshes it with
  ``ducts=()``, the mesher as it was before ducts.

``layers`` grows prism layers on the walls (first height 2 or 3 mm, growth 1.2). Run with
``netgen-mesher`` and the CAD kernel installed::

    python validation/meshing_netgen/ducts.py straight bent bent-layers sweep sweep-layers \
        tank-layers tank-noducts-layers
"""

from __future__ import annotations

import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from aquaflux.io.cad import CadRegion, read_step
from aquaflux.mesh import QualitySummary, closed_cell_residual
from aquaflux.meshing import CadMesher, FacesNear, PrismLayers, cad_mesher


def _written(shape, path: Path):
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    writer.Write(str(path))
    model = read_step(path)
    return model, CadRegion((model.names[0],))


def straight(directory):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    shape = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(200, 100, 1000)).Shape()  # millimetres
    patches = {"inlet": FacesNear(((0.1, 0.05, 0.0),)), "outlet": FacesNear(((0.1, 0.05, 1.0),))}
    return (*_written(shape, directory / "straight.step"), patches, 0.025, 0.002)


def bent(directory):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeRevol
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt

    inner, outer, height, leg = 300.0, 500.0, 100.0, 500.0
    rectangle = BRepBuilderAPI_MakePolygon(
        gp_Pnt(inner, 0, 0), gp_Pnt(outer, 0, 0), gp_Pnt(outer, 0, height),
        gp_Pnt(inner, 0, height), True,
    ).Wire()  # fmt: skip
    bend = BRepPrimAPI_MakeRevol(
        BRepBuilderAPI_MakeFace(rectangle).Face(), gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)),
        math.pi / 2,
    ).Shape()  # fmt: skip
    first = BRepPrimAPI_MakeBox(gp_Pnt(inner, -leg, 0), gp_Pnt(outer, 0, height)).Shape()
    second = BRepPrimAPI_MakeBox(gp_Pnt(-leg, inner, 0), gp_Pnt(0, outer, height)).Shape()
    shape = BRepAlgoAPI_Fuse(BRepAlgoAPI_Fuse(first, bend).Shape(), second).Shape()
    patches = {"inlet": FacesNear(((0.4, -0.5, 0.05),)), "outlet": FacesNear(((-0.5, 0.4, 0.05),))}
    return (*_written(shape, directory / "bent.step"), patches, 0.025, 0.002)


def sweep(directory):
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_MakeWire,
    )
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
    from OCP.collections import Array1_gp_Pnt
    from OCP.GeomAPI import GeomAPI_PointsToBSpline
    from OCP.gp import gp_Pnt

    through = [(0, 0, 0), (0, 0, 300), (150, 0, 600), (150, 100, 900), (0, 100, 1200)]
    array = Array1_gp_Pnt(1, len(through))
    for k, point in enumerate(through):
        array.SetValue(k + 1, gp_Pnt(*point))
    spine = BRepBuilderAPI_MakeWire(
        BRepBuilderAPI_MakeEdge(GeomAPI_PointsToBSpline(array).Curve()).Edge()
    ).Wire()
    profile = BRepBuilderAPI_MakePolygon(
        gp_Pnt(-50, -30, 0), gp_Pnt(50, -30, 0), gp_Pnt(50, 30, 0), gp_Pnt(-50, 30, 0), True
    ).Wire()
    maker = BRepOffsetAPI_MakePipeShell(spine)
    maker.Add(profile)
    maker.Build()
    maker.MakeSolid()
    patches = {"inlet": FacesNear(((0, 0, 0),)), "outlet": FacesNear(((0, 0.1, 1.2),))}
    return (*_written(maker.Shape(), directory / "sweep.step"), patches, 0.015, 0.002)


def tank(directory):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    shape = BRepAlgoAPI_Fuse(
        BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(1000, 1000, 600)).Shape(),
        BRepPrimAPI_MakeBox(gp_Pnt(1000, 400, 200), gp_Pnt(1500, 600, 400)).Shape(),
    ).Shape()
    lamp = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(500, 500, 100), gp_Dir(0, 0, 1)), 50, 400)
    shape = BRepAlgoAPI_Cut(shape, lamp.Shape()).Shape()
    patches = {"inlet": FacesNear(((1.5, 0.5, 0.3),)), "outlet": FacesNear(((0.5, 0.5, 0.6),))}
    return (*_written(shape, directory / "tank.step"), patches, 0.05, 0.003)


def _tilt(points, conduits) -> str:
    """The worst and mean tilt of each duct's rings from square to its centreline."""
    found = []
    for conduit in conduits:
        rings = conduit.rings
        centre = points[rings].mean(axis=1)
        tilt = []
        for j in range(1, len(rings) - 1):
            normal = np.linalg.svd(points[rings[j]] - centre[j])[2][-1]
            along = centre[j + 1] - centre[j - 1]
            along /= np.linalg.norm(along)
            tilt.append(math.degrees(math.acos(min(1.0, abs(normal @ along)))))
        found.append(f"tilt max {max(tilt):.1f} mean {np.mean(tilt):.1f}")
    return "; ".join(found)


def main(arms: list[str]) -> None:
    drawings = {"straight": straight, "bent": bent, "sweep": sweep, "tank": tank}
    captured = {}
    real = cad_mesher.fill_conduits

    def spy(points, fill, walls, tag, conduits):
        captured["points"], captured["conduits"] = points, conduits
        return real(points, fill, walls, tag, conduits)

    cad_mesher.fill_conduits = spy
    for arm in arms:
        words = arm.split("-")
        with tempfile.TemporaryDirectory() as directory:
            model, region, patches, size, first = drawings[words[0]](Path(directory))
            layers = PrismLayers(first_height=first, growth=1.2, count=3) if "layers" in words else None
            options = {"ducts": ()} if "noducts" in words else {}
            captured.clear()
            start = time.perf_counter()
            mesh = CadMesher(max_size=size, feature_angle=math.radians(45), layers=layers).mesh(
                model, region, patches, layered=("walls",) if layers else (), **options
            )
            seconds = time.perf_counter() - start
        geometry = mesh.geometry()
        owner, neighbour = np.asarray(mesh.face_cells.owner), np.asarray(mesh.face_cells.neighbour)
        faces = np.bincount(owner, minlength=mesh.n_cells)
        faces += np.bincount(neighbour[neighbour >= 0], minlength=mesh.n_cells)
        closure = float(np.max(np.asarray(closed_cell_residual(mesh)))) / float(
            np.max(np.asarray(geometry.face.area))
        )
        tilt = _tilt(captured["points"], captured["conduits"]) if captured else "no conduit"
        print(
            f"{arm}: {seconds:.1f} s, {mesh.n_cells} cells, closure {closure:.1e}, "
            f"six-faced {np.mean(faces == 6):.3f}, {tilt}",
            flush=True,
        )
        print(f"  {QualitySummary.of(mesh).to_mapping()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["straight", "bent", "sweep"])
