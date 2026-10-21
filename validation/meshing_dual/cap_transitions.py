"""Ways of joining hexahedra to the tetrahedral fill, measured on the dual over rotated draws.

Every quadrilateral the fill would meet -- the top of a hexahedral layer column on a mapped wall, or
the boundary of an octree hex core -- has to be joined to Netgen's tetrahedra, which meet triangles
only. Each arm here is one way of doing it (a mesher configuration), and each is meshed from five
copies of each drawing rotated 1-5 µrad about x, since one change anywhere moves Netgen's whole fill
and a single draw is not enough:

* **Sozzi**, at ``maxh`` 0.02: the water of ``mapped_layers_sozzi.py`` (five layers on every wall,
  the lamp mapped, the pipes found and filled with O-grids), no octree core -- the wall caps above
  the lamp's layers and at the pipes' junctions;
* **open vessel**, at ``maxh`` 0.01: the cylinder of ``octree_core.py`` (0.3 m across, 0.6 m long, a
  sleeve 25 mm across on its axis), an octree core of 0.04 m cubes at its unset wall size and
  clearance -- the core's caps.

Each row prints the cell count, the time to mesh, the closure, the smallest cell, the faces over a
volume ratio of 20 (larger cell over smaller) and its 99th percentile, the non-orthogonality's 99th
percentile, maximum and count over 50 degrees (and, on Sozzi, the count inside the pipes) and over
90 (a cell turned inside out), and the
small cells away from the walls: cells under 2 mm across (cube root of volume) 13.5-22 mm from
Sozzi's lamp axis at 300 < x < 500 mm, or under ``0.4 maxh`` across more than 12 mm inside the open
vessel's walls. A draw Netgen fails on is printed as failed.

Run in an environment with ``cadquery-ocp-novtk`` and ``netgen-mesher``, naming the arms (or in
``MESHING_CAP_ARMS``, comma-separated; all of them by default)::

    python validation/meshing_dual/cap_transitions.py [arm ...]
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from aquaflux.io.cad import CylinderFace, Placement, read_step
from aquaflux.mesh import closed_cell_residual, face_non_orthogonality
from aquaflux.meshing import CadMesher, ClearedCaps, MeshingError, OctreeCore

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mapped_layers_sozzi import DRAWING, LAYERED, LAYERS, PATCHES, WATER
from octree_core import _open_vessel

#: The draws' rotations about x, in radians.
ANGLES = (1e-6, 2e-6, 3e-6, 4e-6, 5e-6)
SOZZI_MAXH, VESSEL_MAXH, CORE_SIZE = 0.02, 0.01, 0.04

#: Each arm's settings for the mesher, beyond the size, the feature angle and the layers.
ARMS = {
    "caps 0.25": {"cap_height": 0.25},
    "caps 0.45": {"cap_height": 0.45},
    "cleared 0.7": {"cap_height": ClearedCaps()},
    "cleared 1.0": {"cap_height": ClearedCaps(target=1.0, fold=0.8)},
}


def _about_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _small_sozzi(centroid, size):
    radius = np.hypot(centroid[:, 1], centroid[:, 2])
    near = (radius > 0.0135) & (radius < 0.022) & (centroid[:, 0] > 0.3) & (centroid[:, 0] < 0.5)
    return int(np.count_nonzero(near & (size < 0.002)))


def _small_vessel(centroid, size):
    radius = np.hypot(centroid[:, 0], centroid[:, 1])
    inside = (radius > 0.0125 + 0.012) & (radius < 0.15 - 0.012)
    inside &= (centroid[:, 2] > 0.012) & (centroid[:, 2] < 0.6 - 0.012)
    return int(np.count_nonzero(inside & (size < 0.4 * VESSEL_MAXH)))


def _in_pipes(model, region):
    """Which points lie inside one of the region's pipes: the cylinders the fluid fills, under
    10 mm across."""
    pipes = [
        face
        for face in model.topology(region).faces
        if isinstance(face, CylinderFace) and face.solid_inside and face.radius < 0.01
    ]

    def inside(points):
        found = np.zeros(points.shape[0], dtype=bool)
        for pipe in pipes:
            axis = np.asarray(pipe.axis, dtype=float)
            offset = points - np.asarray(pipe.origin, dtype=float)
            along = offset @ axis
            across = np.linalg.norm(offset - along[:, None] * axis, axis=1)
            low, high = pipe.axial_range
            found |= (across < pipe.radius) & (along > low) & (along < high)
        return found

    return inside


def _row(label, mesh, seconds, small, in_pipes=None) -> None:
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
    pipes = ""
    if in_pipes is not None:
        where = in_pipes(np.asarray(geometry.face.centroid)[inside])
        pipes = f" (pipes {int(np.count_nonzero(where & (angle > 50))):4d})"
    print(
        f"{label:>28}  cells {mesh.n_cells:6d}  {seconds:5.1f} s  closure {closure:.0e}  "
        f"min vol {volume.min():.1e}  ratio>20 {int(np.count_nonzero(ratio > 20)):6d} "
        f"p99 {np.percentile(ratio, 99):6.1f}  non-orth p99 {np.percentile(angle, 99):4.1f} "
        f"max {angle.max():5.1f} >50 {int(np.count_nonzero(angle > 50)):5d}{pipes} "
        f">90 {int(np.count_nonzero(angle > 90)):3d}  "
        f"small {small(centroid, np.cbrt(volume)):5d}",
        flush=True,
    )


def _attempt(label, mesher, *args, **kwargs):
    """``mesher``'s mesh of ``args`` and its time, or ``None`` (printed) where Netgen fails."""
    start = time.perf_counter()
    try:
        mesh = mesher.mesh(*args, **kwargs)
    except MeshingError as error:
        print(f"{label:>28}  failed: {str(error).splitlines()[-1][:100]}", flush=True)
        return None, 0.0
    return mesh, time.perf_counter() - start


def main(arms: list[str]) -> None:
    print(
        f"Sozzi maxh {SOZZI_MAXH}; open vessel maxh {VESSEL_MAXH}, core cubes {CORE_SIZE}, wall and "
        f"clearance unset; arms {arms}",
        flush=True,
    )
    swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    for angle in ANGLES:
        model = read_step(DRAWING, Placement(matrix=_about_x(angle) @ swap))
        in_pipes = _in_pipes(model, WATER)
        for name in arms:
            label = f"Sozzi {angle * 1e6:.0f} µrad {name}"
            mesher = CadMesher(
                max_size=SOZZI_MAXH, feature_angle=0.785, layers=LAYERS, **ARMS[name]
            )
            mesh, seconds = _attempt(
                label, mesher, model, WATER, PATCHES, layered=LAYERED, mapped=("lamp",)
            )
            if mesh is not None:
                _row(label, mesh, seconds, _small_sozzi, in_pipes)
            del mesh
    for angle in ANGLES:
        with tempfile.TemporaryDirectory() as directory:
            model, region, patches, layered = _open_vessel(
                pathlib.Path(directory), Placement(matrix=_about_x(angle))
            )
            for name in arms:
                label = f"vessel {angle * 1e6:.0f} µrad {name}"
                mesher = CadMesher(
                    max_size=VESSEL_MAXH, feature_angle=0.785, layers=LAYERS,
                    core=OctreeCore(size=CORE_SIZE), **ARMS[name],
                )  # fmt: skip
                mesh, seconds = _attempt(label, mesher, model, region, patches, layered=layered)
                if mesh is not None:
                    _row(label, mesh, seconds, _small_vessel)
                del mesh


if __name__ == "__main__":
    chosen = os.environ.get("MESHING_CAP_ARMS")
    main(sys.argv[1:] or (chosen.split(",") if chosen else list(ARMS)))
