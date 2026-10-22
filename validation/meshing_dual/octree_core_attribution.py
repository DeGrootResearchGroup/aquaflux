"""Which cells of a mesh about an octree core carry its bad faces: the core caps' apexes, or others?

The open vessel of ``octree_core.py`` (a cylinder 0.3 m across and 0.6 m long, less a sleeve 25 mm
across on its axis; five layers on the wall and the sleeve) is meshed at ``maxh`` about an octree
core of cubes ``size``, its wall size and clearance unset or given. A dual cell is a primal point, so
each cell is classed by the point it stands for, recorded as the mesher builds the primal:

* **cap apex**: the apex of a pyramid capping one of the core's boundary quadrilaterals;
* **transition centre**: the centre of a transition cube, cut into pyramids and tetrahedra about it;
* **cube corner**: any other point of the core;
* **fill**: a point Netgen's fill added;
* **layers and walls**: the walls' points and the layers' points above them.

For each class the harness prints its cell count and median volume over a cube of the wall size,
then, for the interior faces whose larger cell is more than 20 times the smaller, how many have
their smaller cell in each class; and, for the faces over 45 degrees non-orthogonal, how many touch
each class (a face between two classes is counted for both).

Run in an environment with ``cadquery-ocp-novtk`` and ``netgen-mesher``::

    python validation/meshing_dual/octree_core_attribution.py [maxh [size [wall_size [clearance]]]]
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
from typing import ClassVar

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from aquaflux.mesh import face_non_orthogonality
from aquaflux.meshing import CadMesher, OctreeCore

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mapped_layers_sozzi import LAYERS
from octree_core import _open_vessel

CLASSES = ("cap apex", "transition centre", "cube corner", "fill", "layers and walls")


class _Recording(CadMesher):
    """The mesher, keeping where the core's points went and how many points the fill was given."""

    record: ClassVar[dict] = {}

    def _cored(self, points, surface):
        joined, holed, cells = super()._cored(points, surface)
        core = self.core.within(
            points, surface, cap_height=self.cap_height, boundary_size=self.max_size
        )
        _Recording.record.update(start=points.shape[0], core=core)
        return joined, holed, cells

    def _filled(self, points, triangles, max_size):
        _Recording.record["given"] = points.shape[0]
        return super()._filled(points, triangles, max_size)


def _classes(n_cells: int) -> np.ndarray:
    """Each dual cell's class, as an index into :data:`CLASSES`."""
    record = _Recording.record
    start, core = record["start"], record["core"]
    label = np.full(n_cells, CLASSES.index("layers and walls"))
    label[record["given"] :] = CLASSES.index("fill")
    label[start : start + core.points.shape[0]] = CLASSES.index("cube corner")
    centres = np.unique(np.concatenate([core.pyramids[:, 4], core.tetrahedra[:, 0]]))
    label[start + centres] = CLASSES.index("transition centre")
    label[start + core.caps[:, 4]] = CLASSES.index("cap apex")
    return label


def main(max_size: float, size: float, wall_size: float | None, clearance: float | None) -> None:
    core = OctreeCore(size=size, wall_size=wall_size, clearance=clearance)
    wall, gap = core.resolved(max_size)
    print(f"open vessel, maxh {max_size}: cubes {size}, wall {wall}, clearance {gap}", flush=True)
    with tempfile.TemporaryDirectory() as directory:
        model, region, patches, layered = _open_vessel(pathlib.Path(directory))
        mesher = _Recording(max_size=max_size, feature_angle=0.785, layers=LAYERS, core=core)
        mesh = mesher.mesh(model, region, patches, layered=layered)
    record = _Recording.record
    if mesh.n_cells < record["given"]:
        raise RuntimeError("the dual has fewer cells than the fill was given points")
    label = _classes(mesh.n_cells)
    volume = np.asarray(mesh.geometry().cell.volume)
    owner = np.asarray(mesh.face_cells.owner)
    neighbour = np.asarray(mesh.face_cells.neighbour)
    inside = neighbour >= 0
    a, b = owner[inside], neighbour[inside]
    ratio = np.maximum(volume[a], volume[b]) / np.minimum(volume[a], volume[b])
    smaller = np.where(volume[a] <= volume[b], a, b)
    angle = np.asarray(face_non_orthogonality(mesh))[inside]
    lopsided, skewed = ratio > 20.0, angle > 45.0
    print(
        f"cells {mesh.n_cells}; faces with volume ratio over 20: {int(lopsided.sum())}; "
        f"faces over 45 deg: {int(skewed.sum())}",
        flush=True,
    )
    print(f"{'class':>18} {'cells':>7} {'median vol/wall^3':>18} {'ratio>20 (smaller)':>19} "
          f"{'>45 deg (touching)':>19}", flush=True)  # fmt: skip
    for index, name in enumerate(CLASSES):
        mine = label == index
        touching = skewed & ((label[a] == index) | (label[b] == index))
        print(
            f"{name:>18} {int(mine.sum()):7d} {np.median(volume[mine]) / wall**3 if mine.any() else 0:18.3f} "
            f"{int(np.count_nonzero(lopsided & (label[smaller] == index))):19d} "
            f"{int(touching.sum()):19d}",
            flush=True,
        )
    # The worst faces' pairs of classes, to see what an apex cell is lopsided against.
    pairs = np.sort(np.stack([label[a], label[b]], axis=1)[lopsided], axis=1)
    found, counts = np.unique(pairs, axis=0, return_counts=True)
    for (first, second), count in sorted(
        zip(found.tolist(), counts.tolist(), strict=True), key=lambda x: -x[1]
    )[:6]:
        print(f"  ratio>20 between {CLASSES[first]} and {CLASSES[second]}: {count}", flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]

    def value(k, default):
        return float(arguments[k]) if len(arguments) > k else default

    main(value(0, 0.01), value(1, 0.04), value(2, None), value(3, None))
