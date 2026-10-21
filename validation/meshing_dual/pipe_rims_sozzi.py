"""Whether the Sozzi water's two pipes get their O-grid cores, size by size, and whether the mesh closes.

The water is meshed as in ``mapped_layers_sozzi.py``'s mapped arm (layers 0.5 mm x 1.2 x 5 on every
wall, the lamp's cylinder mapped, every pipe the mesher finds O-gridded). For each ``maxh`` the
harness prints, from Netgen's own triangulation of the region, the rims' point counts of each pipe
candidate; then, from the mesh, whether each pipe was cored -- every cell within 2 mm of its axis,
away from its junction, a hexahedron -- and the mesh's cell count, smallest volume and worst closure
residual over the largest face area. A pipe the mesher refuses falls back to tetrahedra, whose dual
cells on the axis are not hexahedra, so the core test reads the mesh, not the mesher's bookkeeping.

``NETGEN_PYTHON`` names an interpreter with ``netgen-mesher`` when this one has none::

    NETGEN_PYTHON=/path/to/venv/bin/python python validation/meshing_dual/pipe_rims_sozzi.py 0.02 0.01
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from aquaflux.io.cad import CylinderFace, Placement, read_step
from aquaflux.mesh import closed_cell_residual
from aquaflux.meshing import CadMesher, MeshingError, NetgenProcess
from aquaflux.meshing.mapped import _loops
from mapped_layers_sozzi import DRAWING, LAYERED, LAYERS, PATCHES, WATER


def _rims(netgen, model, topology, size: float) -> dict[int, list[int]]:
    """Each fluid-filled cylinder's rims' point counts in Netgen's own triangulation."""
    with tempfile.TemporaryDirectory() as directory:
        surface = netgen.triangulate(
            model.write_brep(WATER, pathlib.Path(directory) / "water.brep"), max_size=size
        )
    return {
        f: sorted(loop.size for loop in _loops(surface.triangles[surface.face == f]))
        for f, record in enumerate(topology.faces)
        if isinstance(record, CylinderFace) and record.solid_inside and f != 0
    }


def _cored(mesh) -> tuple[bool, bool]:
    """Whether the riser's and the outlet pipe's cells along their axes are all hexahedra."""
    geometry = mesh.geometry()
    owner = np.asarray(mesh.face_cells.owner)
    neighbour = np.asarray(mesh.face_cells.neighbour)
    inside = neighbour >= 0
    faces = np.bincount(owner, minlength=mesh.n_cells) + np.bincount(
        neighbour[inside], minlength=mesh.n_cells
    )
    sizes = np.diff(np.asarray(mesh.face_nodes.offsets))
    quads = np.bincount(owner, weights=sizes == 4, minlength=mesh.n_cells) + np.bincount(
        neighbour[inside], weights=(sizes == 4)[inside], minlength=mesh.n_cells
    )
    hexahedral = (faces == 6) & (quads == 6)
    centroid = np.asarray(geometry.cell.centroid)
    riser = (np.hypot(centroid[:, 0] - 0.0476, centroid[:, 1]) < 0.002) & (centroid[:, 2] > 0.1)
    outlet = (np.hypot(centroid[:, 1], centroid[:, 2]) < 0.002) & (centroid[:, 0] > 0.95)
    return tuple(bool(np.any(on) and np.all(hexahedral[on])) for on in (riser, outlet))


def main(sizes: list[float]) -> None:
    model = read_step(DRAWING, Placement(matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
    topology = model.topology(WATER)
    netgen = NetgenProcess(python=os.environ.get("NETGEN_PYTHON", sys.executable))
    for size in sizes:
        print(f"maxh {size}: Netgen's rims {_rims(netgen, model, topology, size)}", flush=True)
        mesher = CadMesher(max_size=size, feature_angle=0.785, layers=LAYERS, netgen=netgen)
        start = time.perf_counter()
        try:
            mesh = mesher.mesh(model, WATER, PATCHES, layered=LAYERED, mapped=("lamp",))
        except MeshingError as error:
            print(f"  meshing failed: {str(error).splitlines()[0]}", flush=True)
            continue
        seconds = time.perf_counter() - start
        geometry = mesh.geometry()
        volume = np.asarray(geometry.cell.volume)
        scale = float(np.max(np.asarray(geometry.face.area)))
        closure = float(np.max(np.asarray(closed_cell_residual(mesh)))) / scale
        riser, outlet = _cored(mesh)
        print(
            f"  cored: riser {riser} outlet {outlet}  cells {mesh.n_cells}  {seconds:.1f} s  "
            f"min volume {volume.min():.2e}  closure {closure:.1e}",
            flush=True,
        )
        del mesh, geometry


if __name__ == "__main__":
    main([float(argument) for argument in sys.argv[1:]] or [0.02, 0.01, 0.005, 0.0035, 0.0025])
