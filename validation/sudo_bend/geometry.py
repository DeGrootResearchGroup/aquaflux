"""The 90-degree circular pipe bend of Sudo, Sumida & Hibara (1998), written as a STEP drawing.

Sudo, K., Sumida, M. & Hibara, H. (1998). Experimental investigation on turbulent flow in a
circular-sectioned 90-degree bend. *Experiments in Fluids* 25, 42-49. Air at a bulk Reynolds number
of 6.0e4 through a pipe of diameter ``D`` = 104 mm, bent through 90 degrees on a centreline radius
``R_c = 2 D``, with a long straight pipe upstream (the flow enters the bend fully developed) and a
straight pipe downstream. Mean and fluctuating velocities are reported upstream of the bend,
through it, and downstream to 10 diameters.

Every length is in metres. The bend lies in the plane ``z = 0``, which is a plane of symmetry of the
mean flow: the fluid enters along ``+x`` through an inlet at ``x = -upstream``, turns about the axis
through ``(0, R_c, 0)`` along ``z``, and leaves along ``+y`` through an outlet at
``y = R_c + downstream``. The three pieces -- the upstream pipe, the bend and the downstream pipe --
are written as one solid, ``fluid``, with five faces: the inlet and outlet discs, the two pipe
cylinders and the toroidal bend wall between them.

Usage::

    python validation/sudo_bend/geometry.py [<out.step>]
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
)
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeRevol
from OCP.gp import gp_Ax1, gp_Ax2, gp_Circ, gp_Dir, gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.STEPCAFControl import STEPCAFControl_Writer
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFDoc import XCAFDoc_DocumentTool

HERE = Path(__file__).parent
STEP = HERE / "bend.step"
SOLID = "fluid"


@dataclasses.dataclass(frozen=True)
class Bend:
    """The bend's dimensions, in metres.

    Attributes
    ----------
    diameter : float
        The pipe's inner diameter ``D``.
    curvature_ratio : float
        The bend's centreline radius over ``D``.
    upstream : float
        The straight pipe before the bend, in diameters.
    downstream : float
        The straight pipe after the bend, in diameters.
    """

    diameter: float = 0.104
    curvature_ratio: float = 2.0
    upstream: float = 3.0
    downstream: float = 15.0

    @property
    def radius(self) -> float:
        """The pipe's inner radius."""
        return 0.5 * self.diameter

    @property
    def bend_radius(self) -> float:
        """The bend's centreline radius ``R_c``."""
        return self.curvature_ratio * self.diameter

    @property
    def inlet_centre(self) -> tuple[float, float, float]:
        """The centre of the inlet disc."""
        return (-self.upstream * self.diameter, 0.0, 0.0)

    @property
    def outlet_centre(self) -> tuple[float, float, float]:
        """The centre of the outlet disc."""
        rc = self.bend_radius
        return (rc, rc + self.downstream * self.diameter, 0.0)

    def wall_point(self, angle: float) -> tuple[float, float, float]:
        """A point on the bend's wall at a bend angle (radians), on its top (``z = r``)."""
        rc = self.bend_radius
        return (rc * math.sin(angle), rc * (1.0 - math.cos(angle)), self.radius)

    def shape(self):
        """The fluid as one kernel solid: the two pipes and the bend fused, faces merged."""
        r, rc, d = self.radius, self.bend_radius, self.diameter
        # Every seam on the top of the pipe (+z), off the bend's inner and outer walls.
        top = gp_Dir(0.0, 0.0, 1.0)
        upstream = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(*self.inlet_centre), gp_Dir(1.0, 0.0, 0.0), top), r, self.upstream * d
        ).Shape()
        section = gp_Circ(gp_Ax2(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0), top), r)
        disc = BRepBuilderAPI_MakeFace(
            BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(section).Edge()).Wire()
        ).Face()
        bend = BRepPrimAPI_MakeRevol(
            disc, gp_Ax1(gp_Pnt(0.0, rc, 0.0), gp_Dir(0.0, 0.0, 1.0)), 0.5 * math.pi
        ).Shape()
        downstream = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(rc, rc, 0.0), gp_Dir(0.0, 1.0, 0.0), top), r, self.downstream * d
        ).Shape()
        fused = BRepAlgoAPI_Fuse(BRepAlgoAPI_Fuse(upstream, bend).Shape(), downstream).Shape()
        unify = ShapeUpgrade_UnifySameDomain(fused, True, True, True)
        unify.Build()
        return unify.Shape()

    def write_step(self, path: Path) -> None:
        """Write the fluid as a STEP file holding one solid named ``fluid``."""
        document = TDocStd_Document(TCollection_ExtendedString("bend"))
        # Lengths are written in metres, as drawn; the kernel's default unit is the millimetre.
        XCAFDoc_DocumentTool.SetLengthUnit_s(document, 1.0)
        Interface_Static.SetCVal_s("write.step.unit", "M")
        tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
        label = tool.AddShape(self.shape(), False)
        TDataStd_Name.Set_s(label, TCollection_ExtendedString(SOLID))
        writer = STEPCAFControl_Writer()
        writer.SetNameMode(True)
        writer.Transfer(document)
        if writer.Write(str(path)) != IFSelect_RetDone:
            raise OSError(f"could not write {path}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else STEP
    Bend().write_step(out)
    print(f"wrote {out}", flush=True)
