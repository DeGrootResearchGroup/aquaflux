"""Which cells carry the bad faces of a mesh joined to its tetrahedra by pyramid caps?

One draw each of the two drawings of ``cap_transitions.py`` (Sozzi at ``maxh`` 0.02 with its lamp
mapped and its pipes O-gridded; the open vessel at ``maxh`` 0.01 about an octree core of 0.04 m
cubes), meshed with the arms named (``cap_transitions.ARMS``). A dual cell is a primal point, so
each cell is classed by the point it stands for: a **wall cap's apex** (over a layer's top or a pipe
core's end), a **core cap's apex** (over the octree core's boundary), or **other**. The apexes are
recorded as the caps are built and found among the primal's points by position.

For each arm it prints each class's cell count and median size (cube root of volume), then for the
interior faces over 50 degrees non-orthogonal, and those over a volume ratio of 20, how many touch
each class (a face between two classes counts for both), the faces over 50 degrees by the pair of
classes they join, and every face over 90 degrees: where it is, its two cells' classes and sizes.
A **fill** cell is a point Netgen's fill added; **other** is the walls', the layers' and the cores'.

Run in an environment with ``cadquery-ocp-novtk`` and ``netgen-mesher``, naming the arms (or in
``MESHING_CAP_ARMS``, comma-separated)::

    python validation/meshing_dual/cap_attribution.py [arm ...]
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from aquaflux.io.cad import Placement, read_step
from aquaflux.mesh import face_non_orthogonality
from aquaflux.meshing import CadMesher, OctreeCore, cad_mesher, hex_core
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cap_transitions import ARMS, CORE_SIZE, SOZZI_MAXH, VESSEL_MAXH
from mapped_layers_sozzi import DRAWING, LAYERED, LAYERS, PATCHES, WATER
from octree_core import _open_vessel

CLASSES = ("wall cap apex", "core cap apex", "fill", "other")
_RECORD: dict = {}


def _recording(module, name, key):
    """Wrap ``module.name`` so each call's result (or first argument) is kept under ``key``."""
    original = getattr(module, name)

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        if key == "primal":
            _RECORD["primal"] = np.asarray(args[0], dtype=float)
        else:
            _RECORD.setdefault(key, []).append(result.apexes)
        return result

    setattr(module, name, wrapped)


_recording(cad_mesher, "capped", "wall")
_recording(hex_core, "capped", "core")
_recording(cad_mesher, "element_mesh", "primal")
_filled = CadMesher._filled


def _recorded_fill(self, points, triangles, max_size):
    joined, tetrahedra = _filled(self, points, triangles, max_size)
    _RECORD["fill"] = [joined[points.shape[0] :]]
    return joined, tetrahedra


CadMesher._filled = _recorded_fill


def _classes(n_cells: int) -> np.ndarray:
    """Each dual cell's class, by the primal point it stands for."""
    found = np.full(n_cells, CLASSES.index("other"))
    tree = cKDTree(_RECORD["primal"])
    for key, label in (("fill", "fill"), ("wall", "wall cap apex"), ("core", "core cap apex")):
        for apexes in _RECORD.get(key, []):
            if not len(apexes):
                continue
            # Recorded before the core is placed among the mesh's points, or before compaction,
            # at the same coordinates. A point of the fill is Netgen's own.
            distance, point = tree.query(apexes)
            point = point[distance < 1e-12]
            found[point[point < n_cells]] = CLASSES.index(label)
    return found


def _report(name, mesh) -> None:
    geometry = mesh.geometry()
    volume = np.asarray(geometry.cell.volume)
    size = np.cbrt(volume)
    kind = _classes(mesh.n_cells)
    owner = np.asarray(mesh.face_cells.owner)
    neighbour = np.asarray(mesh.face_cells.neighbour)
    inside = np.flatnonzero(neighbour >= 0)
    a, b = owner[inside], neighbour[inside]
    angle = np.asarray(face_non_orthogonality(mesh))[inside]
    ratio = np.maximum(volume[a], volume[b]) / np.minimum(volume[a], volume[b])
    print(f"  {name}: cells {mesh.n_cells}", flush=True)
    for index, label in enumerate(CLASSES):
        mine = kind == index
        touching = (kind[a] == index) | (kind[b] == index)
        print(
            f"    {label:<14} cells {int(mine.sum()):6d}  median size "
            f"{np.median(size[mine]) * 1000 if mine.any() else float('nan'):6.2f} mm  "
            f"faces >50 touching {int(np.count_nonzero(touching & (angle > 50))):5d} of "
            f"{int(np.count_nonzero(angle > 50))}  ratio>20 touching "
            f"{int(np.count_nonzero(touching & (ratio > 20))):5d} of {int(np.count_nonzero(ratio > 20))}",
            flush=True,
        )
    bad = angle > 50
    pairs = np.sort(np.stack([kind[a[bad]], kind[b[bad]]], axis=1), axis=1)
    found, count = np.unique(pairs, axis=0, return_counts=True)
    print(
        "    faces >50 by the classes they join: "
        + ", ".join(
            f"{CLASSES[i]} | {CLASSES[j]} {c}" for (i, j), c in zip(found, count, strict=True)
        ),
        flush=True,
    )
    centroid = np.asarray(geometry.face.centroid)[inside]
    for k in np.flatnonzero(angle > 90):
        print(
            f"    face over 90: {angle[k]:5.1f} deg at {np.round(centroid[k] * 1000, 1)} mm, cells "
            f"{CLASSES[kind[a[k]]]} {size[a[k]] * 1000:.2f} mm / {CLASSES[kind[b[k]]]} "
            f"{size[b[k]] * 1000:.2f} mm",
            flush=True,
        )


def main(arms: list[str]) -> None:
    swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    model = read_step(DRAWING, Placement(matrix=swap))
    print(f"Sozzi maxh {SOZZI_MAXH}, one draw", flush=True)
    for name in arms:
        _RECORD.clear()
        mesher = CadMesher(max_size=SOZZI_MAXH, feature_angle=0.785, layers=LAYERS, **ARMS[name])
        _report(name, mesher.mesh(model, WATER, PATCHES, layered=LAYERED, mapped=("lamp",)))
    print(f"open vessel maxh {VESSEL_MAXH}, core cubes {CORE_SIZE}, one draw", flush=True)
    with tempfile.TemporaryDirectory() as directory:
        model, region, patches, layered = _open_vessel(pathlib.Path(directory))
        for name in arms:
            _RECORD.clear()
            mesher = CadMesher(
                max_size=VESSEL_MAXH, feature_angle=0.785, layers=LAYERS,
                core=OctreeCore(size=CORE_SIZE), **ARMS[name],
            )  # fmt: skip
            _report(name, mesher.mesh(model, region, patches, layered=layered))


if __name__ == "__main__":
    chosen = os.environ.get("MESHING_CAP_ARMS")
    main(sys.argv[1:] or (chosen.split(",") if chosen else list(ARMS)))
