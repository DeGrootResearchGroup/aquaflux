"""STEP files written for a test by the CAD kernel: named parts, placed as an assembly if asked.

Needs the optional kernel (``pip install aquaflux[cad]``); a test importing this skips without it
first (``pytest.importorskip("OCP")``).
"""

from __future__ import annotations

from OCP.gp import gp_Trsf, gp_Vec
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPCAFControl import STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_AsIs
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.TopLoc import TopLoc_Location
from OCP.XCAFDoc import XCAFDoc_DocumentTool


def write_step(path, parts, instances=None):
    """Write named parts to STEP in millimetres; ``instances`` places a part several times.

    ``parts`` maps a name to a kernel solid, or sheet, given in metres. ``instances``, if given, maps a part's
    name to translations (in metres); the part is then written once and placed at each, as an
    assembly, which is how a CAD tool writes a repeated component.
    """
    document = TDocStd_Document(TCollection_ExtendedString("test"))
    XCAFDoc_DocumentTool.SetLengthUnit_s(document, 1.0)
    tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    assembly = tool.NewShape() if instances else None
    for name, shape in parts.items():
        label = tool.AddShape(shape, False)
        TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))
        for offset in (instances or {}).get(name, ()):
            move = gp_Trsf()
            move.SetTranslation(gp_Vec(*offset))
            tool.AddComponent(assembly, label, TopLoc_Location(move))
    if assembly is not None:
        tool.UpdateAssemblies()
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    writer = STEPCAFControl_Writer()
    writer.Transfer(document, STEPControl_AsIs)
    assert writer.Write(str(path)) == IFSelect_RetDone
    return path
