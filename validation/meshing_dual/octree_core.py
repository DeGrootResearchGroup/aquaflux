"""Two vessels meshed about an octree hex core, against the same meshes with a polyhedral core.

Each drawing is meshed at ``maxh`` with five layers on every wall (first dual height 0.5 mm,
growth 1.2), once with a polyhedral core and once with an octree core of cubes ``size``, its wall
size and clearance left unset (the mesher's ``maxh``, and twice that) or the clearance given:

* **Sozzi**: the Sozzi water (the vessel, its two pipes, the lamp taken out), the lamp's and the
  pipes' cylinders mapped, as in ``mapped_layers_sozzi.py``. Its chamber is an annulus about
  34 mm across round the lamp, so at ``maxh`` 0.01 little or no core fits;
* **open vessel**: a cylinder 0.3 m across and 0.6 m long with a lamp sleeve 25 mm across along
  its axis, its ends patches of their own, drawn by OCP here.

For each it prints the cell count, the time to mesh, the closure, the smallest cell, how many cells
are cubes (six quadrilaterals, the volume of a cube of ``maxh`` or a power of two above it), the
non-orthogonality and skewness of the faces (maximum and 99th percentile), and the ratio of the
larger to the smaller of the two cells on each interior face (99th percentile, maximum, and how
many faces exceed 20) -- the measure of the small dual cell each pyramid cap's apex becomes.

Run in an environment with ``cadquery-ocp-novtk`` and ``netgen-mesher``::

    python validation/meshing_dual/octree_core.py [maxh [size [clearance]]]
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from aquaflux.io.cad import CadRegion, Placement, read_step
from aquaflux.mesh import closed_cell_residual, face_non_orthogonality, face_skewness
from aquaflux.meshing import CadMesher, FacesNear, OctreeCore

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mapped_layers_sozzi import DRAWING, LAYERED, LAYERS, PATCHES, WATER, _hexahedral


def _open_vessel(directory: pathlib.Path, placement: Placement | None = None):
    """A cylinder 0.3 m across and 0.6 m long about z, less a sleeve 25 mm across along its
    axis: the model (read with ``placement``, if given), the region, its patches and the patches
    layered."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    # Drawn in millimetres, which is what a STEP file without units is read in.
    axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    vessel = BRepPrimAPI_MakeCylinder(axis, 150.0, 600.0).Shape()
    sleeve = BRepPrimAPI_MakeCylinder(axis, 12.5, 600.0).Shape()
    water = BRepAlgoAPI_Cut(vessel, sleeve).Shape()
    writer = STEPControl_Writer()
    writer.Transfer(water, STEPControl_AsIs)
    writer.Write(str(directory / "vessel.step"))
    model = read_step(directory / "vessel.step", placement)
    patches = {
        "inlet": FacesNear(((0.08, 0.0, 0.0),)),
        "outlet": FacesNear(((0.08, 0.0, 0.6),)),
        "sleeve": FacesNear(((0.0125, 0.0, 0.3),)),
    }
    return model, CadRegion((model.names[0],)), patches, ("walls", "sleeve")


def _report(mesh, seconds: float, label: str, wall: float) -> None:
    geometry = mesh.geometry()
    volume = np.asarray(geometry.cell.volume)
    scale = float(np.max(np.asarray(geometry.face.area)))
    closure = float(np.max(np.asarray(closed_cell_residual(mesh)))) / scale
    hexahedra, _ = _hexahedral(mesh)
    levels = np.log2(volume[hexahedra] / wall**3) / 3.0
    cubes = int(np.count_nonzero(np.isclose(levels, np.round(levels), atol=1e-6)))
    owner = np.asarray(mesh.face_cells.owner)
    neighbour = np.asarray(mesh.face_cells.neighbour)
    inside = neighbour >= 0
    pair = np.stack([volume[owner[inside]], volume[neighbour[inside]]])
    ratio = pair.max(axis=0) / pair.min(axis=0)
    angle = np.asarray(face_non_orthogonality(mesh))[inside]
    skew = np.asarray(face_skewness(mesh))[inside]
    small = int(np.count_nonzero(volume < 0.1 * wall**3))
    print(
        f"{label:>11}  cells {mesh.n_cells:7d}  {seconds:6.1f} s  closure {closure:.1e}  "
        f"min volume {volume.min():.2e}  cubes {cubes:6d}  cells under a tenth of maxh^3 {small}",
        flush=True,
    )
    print(
        f"{'':>11}  non-orthogonality max {angle.max():.1f} p99 {np.percentile(angle, 99):.1f} deg  "
        f"skewness max {skew.max():.2f} p99 {np.percentile(skew, 99):.2f}  "
        f"volume ratio p99 {np.percentile(ratio, 99):.1f} max {ratio.max():.1f} "
        f"over 20: {int(np.count_nonzero(ratio > 20))}",
        flush=True,
    )


def _arms(model, region, patches, layered, mapped, max_size, cores) -> None:
    for label, chosen in (("polyhedral", None), *cores):
        mesher = CadMesher(max_size=max_size, feature_angle=0.785, layers=LAYERS, core=chosen)
        start = time.perf_counter()
        try:
            mesh = mesher.mesh(model, region, patches, layered=layered, mapped=mapped)
        except ValueError as error:
            print(f"{label:>11}  refused: {error}", flush=True)
            continue
        _report(mesh, time.perf_counter() - start, label, max_size)
        del mesh


def main(max_size: float, size: float, clearance: float | None) -> None:
    cores = []
    for wall_size in (None, 2.0 * max_size):
        core = OctreeCore(size=size, wall_size=wall_size, clearance=clearance)
        wall, gap = core.resolved(max_size)
        cores.append((f"wall {wall:g}", core))
        print(f"maxh {max_size}: core cubes {size}, wall {wall}, clearance {gap}", flush=True)
    print("Sozzi", flush=True)
    model = read_step(DRAWING, Placement(matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
    _arms(model, WATER, PATCHES, LAYERED, ("lamp", "pipes"), max_size, cores)
    print("open vessel", flush=True)
    with tempfile.TemporaryDirectory() as directory:
        model, region, patches, layered = _open_vessel(pathlib.Path(directory))
        _arms(model, region, patches, layered, (), max_size, cores)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    main(
        float(arguments[0]) if arguments else 0.01,
        float(arguments[1]) if len(arguments) > 1 else 0.04,
        float(arguments[2]) if len(arguments) > 2 else None,
    )
