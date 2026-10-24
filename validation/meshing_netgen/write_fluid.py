"""Write the Sozzi fluid (reactor body and pipes, less the lamp) as BREP and STEP files.

The OCP half of the file hand-off to Netgen: it runs in a process that never imports Netgen.

Usage::

    python validation/meshing_netgen/write_fluid.py <out stem>
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from aquaflux.io.cad import Placement
from aquaflux.io.cad.kernel import OpenCascade
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepTools import BRepTools
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

STEP = pathlib.Path(__file__).parents[1] / "uvreactor_openfoam" / "of_case" / "SozziTaghipour.step"


def main(stem: str) -> None:
    kernel = OpenCascade()
    solids, _ = kernel.read_step(STEP, Placement())  # the drawing has no sheets
    body = kernel.fuse([solids["reactor_body"], solids["inlet_pipe"], solids["outlet_pipe"]])
    fluid = BRepAlgoAPI_Cut(body, solids["lamp"]).Shape()
    BRepTools.Write_s(fluid, f"{stem}.brep")
    writer = STEPControl_Writer()
    writer.Transfer(fluid, STEPControl_AsIs)
    if writer.Write(f"{stem}.step") != IFSelect_RetDone:
        raise OSError(f"could not write {stem}.step")
    print(f"wrote {stem}.brep and {stem}.step", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
