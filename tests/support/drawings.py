"""Drawings written by the CAD kernel for the tests that read a STEP file: OCP draws, the file is read
back as any drawing would be. Each helper needs the ``cad`` extra; a caller skips without it."""

from __future__ import annotations

import math
from pathlib import Path

#: The bent duct's sides: 0.2 m wide (radially), 0.1 m high; its legs 0.3 m; its bend's centreline
#: radius 0.4 m about ``z``.
DUCT_INNER, DUCT_OUTER, DUCT_HEIGHT, DUCT_LEG = 0.3, 0.5, 0.1, 0.3


def write_bent_duct(path: Path) -> Path:
    """A rectangular duct with a quarter bend, as a STEP file: a leg along ``y``, the bend about
    ``z``, a leg along ``-x``, drawn as two boxes and a revolved rectangle fused, so its sides are
    planes and, in the bend, two cylinders and two flat sectors -- three segments of four sides.
    Drawn in millimetres, which is what a STEP file without units is read in."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeRevol
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    inner, outer, height, leg = (
        1000.0 * x for x in (DUCT_INNER, DUCT_OUTER, DUCT_HEIGHT, DUCT_LEG)
    )
    rectangle = BRepBuilderAPI_MakePolygon(
        gp_Pnt(inner, 0, 0), gp_Pnt(outer, 0, 0), gp_Pnt(outer, 0, height),
        gp_Pnt(inner, 0, height), True,
    ).Wire()  # fmt: skip
    face = BRepBuilderAPI_MakeFace(rectangle).Face()
    bend = BRepPrimAPI_MakeRevol(face, gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), math.pi / 2)
    first = BRepPrimAPI_MakeBox(gp_Pnt(inner, -leg, 0), gp_Pnt(outer, 0, height)).Shape()
    second = BRepPrimAPI_MakeBox(gp_Pnt(-leg, inner, 0), gp_Pnt(0, outer, height)).Shape()
    shape = BRepAlgoAPI_Fuse(BRepAlgoAPI_Fuse(first, bend.Shape()).Shape(), second).Shape()
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    writer.Write(str(path))
    return Path(path)
