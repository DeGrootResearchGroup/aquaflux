"""The pyramid caps' apexes moved after the fill (``ApexRelaxation``), against left where they stand.

Two drawings, each meshed from five copies rotated 1-5 µrad about x (one change anywhere
moves Netgen's whole fill, so a single draw is not enough), with the relaxation off and on:

* **Sozzi**, at ``maxh`` 0.02 as #785 measured it: the water of ``mapped_layers_sozzi.py`` (five layers on every wall, the lamp mapped, the
  pipes found and filled with O-grids), no octree core -- the wall caps above the lamp's layers;
* **open vessel**, at ``maxh``: the cylinder of ``octree_core.py`` (0.3 m across, 0.6 m long, a sleeve 25 mm across
  on its axis), an octree core of cubes ``size`` at its unset wall size and clearance -- the core's caps.

Each row prints the cell count, the time to mesh, the closure, the smallest cell, the faces over a
volume ratio of 20 (larger cell over smaller) and its 99th percentile, the non-orthogonality's 99th
percentile, maximum and count over 50 degrees, and the small cells away from the walls: cells under
2 mm across (cube root of volume) 13.5-22 mm from Sozzi's lamp axis at 300 < x < 500 mm, or under
``0.4 maxh`` across more than 12 mm inside the open vessel's walls.

Run in an environment with ``cadquery-ocp-novtk`` and ``netgen-mesher``::

    python validation/meshing_dual/apex_relaxation.py [maxh [size]]
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from aquaflux.io.cad import Placement, read_step
from aquaflux.mesh import closed_cell_residual, face_non_orthogonality
from aquaflux.meshing import ApexRelaxation, CadMesher, OctreeCore

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mapped_layers_sozzi import DRAWING, LAYERED, LAYERS, PATCHES, WATER
from octree_core import _open_vessel

#: The draws' rotations about x, in radians.
ANGLES = (1e-6, 2e-6, 3e-6, 4e-6, 5e-6)


def _about_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _small_sozzi(centroid, size):
    radius = np.hypot(centroid[:, 1], centroid[:, 2])
    near = (radius > 0.0135) & (radius < 0.022) & (centroid[:, 0] > 0.3) & (centroid[:, 0] < 0.5)
    return int(np.count_nonzero(near & (size < 0.002)))


def _small_vessel(max_size):
    def count(centroid, size):
        radius = np.hypot(centroid[:, 0], centroid[:, 1])
        inside = (radius > 0.0125 + 0.012) & (radius < 0.15 - 0.012)
        inside &= (centroid[:, 2] > 0.012) & (centroid[:, 2] < 0.6 - 0.012)
        return int(np.count_nonzero(inside & (size < 0.4 * max_size)))

    return count


def _row(label, mesh, seconds, small) -> None:
    geometry = mesh.geometry()
    volume = np.asarray(geometry.cell.volume)
    centroid = np.asarray(geometry.cell.centroid)
    scale = float(np.max(np.asarray(geometry.face.area)))
    closure = float(np.max(np.asarray(closed_cell_residual(mesh)))) / scale
    owner = np.asarray(mesh.face_cells.owner)
    neighbour = np.asarray(mesh.face_cells.neighbour)
    inside = neighbour >= 0
    pair = np.stack([volume[owner[inside]], volume[neighbour[inside]]])
    ratio = pair.max(axis=0) / pair.min(axis=0)
    angle = np.asarray(face_non_orthogonality(mesh))[inside]
    print(
        f"{label:>22}  cells {mesh.n_cells:6d}  {seconds:5.1f} s  closure {closure:.0e}  "
        f"min vol {volume.min():.1e}  ratio>20 {int(np.count_nonzero(ratio > 20)):6d} "
        f"p99 {np.percentile(ratio, 99):6.1f}  non-orth p99 {np.percentile(angle, 99):4.1f} "
        f"max {angle.max():5.1f} >50 {int(np.count_nonzero(angle > 50)):5d}  "
        f"small {small(centroid, np.cbrt(volume)):5d}",
        flush=True,
    )


def main(max_size: float, size: float) -> None:
    arms = (("left", None), ("relaxed", ApexRelaxation()))
    print(
        f"Sozzi maxh 0.02; open vessel maxh {max_size}, core cubes {size}, wall and clearance unset",
        flush=True,
    )
    swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    for angle in ANGLES:
        model = read_step(DRAWING, Placement(matrix=_about_x(angle) @ swap))
        for name, relaxation in arms:
            mesher = CadMesher(
                max_size=0.02, feature_angle=0.785, layers=LAYERS, apex_relaxation=relaxation
            )
            start = time.perf_counter()
            mesh = mesher.mesh(model, WATER, PATCHES, layered=LAYERED, mapped=("lamp",))
            _row(f"Sozzi {angle * 1e6:.0f} µrad {name}", mesh, time.perf_counter() - start,
                 _small_sozzi)  # fmt: skip
            del mesh
    for angle in ANGLES:
        with tempfile.TemporaryDirectory() as directory:
            model, region, patches, layered = _open_vessel(
                pathlib.Path(directory), Placement(matrix=_about_x(angle))
            )
            for name, relaxation in arms:
                mesher = CadMesher(
                    max_size=max_size, feature_angle=0.785, layers=LAYERS,
                    core=OctreeCore(size=size), apex_relaxation=relaxation,
                )  # fmt: skip
                start = time.perf_counter()
                mesh = mesher.mesh(model, region, patches, layered=layered)
                _row(f"vessel {angle * 1e6:.0f} µrad {name}", mesh, time.perf_counter() - start,
                     _small_vessel(max_size))  # fmt: skip
                del mesh


if __name__ == "__main__":
    arguments = [float(argument) for argument in sys.argv[1:]]
    main(*(arguments or [0.01, 0.04]))
