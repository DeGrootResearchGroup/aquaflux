"""Hexahedral layers on the Sozzi reactor's lamp and pipes, against prism layers on triangles.

The Sozzi water (the vessel, its two pipes, the lamp taken out) is meshed twice at each ``maxh``,
with the same layers on every wall (first dual height 0.5 mm, growth 1.2, five layers): once with
every face triangulated, so the layers are prisms, and once with the lamp's cylinder and the two
pipes' cylinders meshed as structured quadrilaterals (``mapped``), so their layers are hexahedra.
For each the harness prints the cell count, the time to mesh, how many dual cells are hexahedra,
the share of the lamp's and the pipes' wall faces that are quadrilaterals of hexahedral cells, the
median and 5-95 % range of the dual's first layer boundary from the true lamp wall (the dual
vertices 0.2-0.8 mm from it, 0.1 < x < 0.7), and the non-orthogonality of the faces near the lamp
(within 3 mm) and everywhere: the angle between a face's normal and the line joining its two cells'
centroids, its maximum and 99th percentile.

Run in an environment with ``cadquery-ocp-novtk`` and ``netgen-mesher``::

    python validation/meshing_dual/mapped_layers_sozzi.py 0.02 0.01
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from aquaflux.io.cad import CadRegion, Placement, read_step
from aquaflux.mesh import closed_cell_residual
from aquaflux.meshing import CadMesher, FacesNear, PrismLayers

DRAWING = (
    pathlib.Path(__file__).resolve().parents[1] / "uvreactor_openfoam/of_case/SozziTaghipour.step"
)
WATER = CadRegion(("reactor_body", "inlet_pipe", "outlet_pipe"), without=("lamp",))
PATCHES = {
    "inlet": FacesNear(((0.04765, 0.0, 0.8945),)),
    "outlet": FacesNear(((1.739, 0.0, 0.0),)),
    "lamp": FacesNear(((0.4, 0.01, 0.0),)),
    "lamp_tip": FacesNear(((0.81, 0.0, 0.0),)),
    "pipes": FacesNear(((0.0572, 0.0, 0.5), (1.3, 0.0096, 0.0))),
}
LAYERS = PrismLayers(first_height=0.0005, growth=1.2, count=5)
LAYERED = ("walls", "lamp", "lamp_tip", "pipes")


def _hexahedral(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Which cells are hexahedra (six quadrilaterals), and each face's corner count."""
    size = np.diff(np.asarray(mesh.face_nodes.offsets))
    faces = np.zeros(mesh.n_cells, dtype=np.int64)
    quads = np.zeros(mesh.n_cells, dtype=np.int64)
    for side in (mesh.face_cells.owner, mesh.face_cells.neighbour):
        side = np.asarray(side)
        np.add.at(faces, side[side >= 0], 1)
        np.add.at(quads, side[side >= 0], size[side >= 0] == 4)
    return (faces == 6) & (quads == 6), size


def _non_orthogonality(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Each interior face's non-orthogonality in degrees, and its centroid."""
    geometry = mesh.geometry()
    owner = np.asarray(mesh.face_cells.owner)
    neighbour = np.asarray(mesh.face_cells.neighbour)
    inside = neighbour >= 0
    centroid = np.asarray(geometry.cell.centroid)
    joining = centroid[neighbour[inside]] - centroid[owner[inside]]
    normal = np.asarray(geometry.face.normal)[inside]
    cosine = np.einsum("ij,ij->i", joining, normal) / np.linalg.norm(joining, axis=1)
    return np.degrees(np.arccos(np.clip(np.abs(cosine), 0.0, 1.0))), np.asarray(
        geometry.face.centroid
    )[inside]


def _report(model, mesh, seconds: float, label: str) -> None:
    geometry = mesh.geometry()
    volume = np.asarray(geometry.cell.volume)
    scale = float(np.max(np.asarray(geometry.face.area)))
    closure = float(np.max(np.asarray(closed_cell_residual(mesh)))) / scale
    hexahedra, size = _hexahedral(mesh)
    owner = np.asarray(mesh.face_cells.owner)
    shares = []
    for patch in ("lamp", "pipes"):
        mine = np.asarray(mesh.face_patches.indices(patch))
        shares.append(float(np.mean((size[mine] == 4) & hexahedra[owner[mine]])))
    coords = np.asarray(mesh.node_coords)
    radius = np.hypot(coords[:, 1], coords[:, 2])
    near = (coords[:, 0] > 0.1) & (coords[:, 0] < 0.7) & (np.abs(radius - 0.0105) < 0.001)
    away = model.project(WATER, coords[near]).distance
    first = away[(away > 0.0002) & (away < 0.0008)] * 1000
    angle, where = _non_orthogonality(mesh)
    by_lamp = (np.hypot(where[:, 1], where[:, 2]) < 0.013) & (where[:, 0] < 0.79)
    print(
        f"{label:>9}  cells {mesh.n_cells:7d}  {seconds:6.1f} s  min volume {volume.min():.2e}  "
        f"closure {closure:.1e}  hexahedra {int(hexahedra.sum()):6d}  "
        f"hexahedral walls lamp {shares[0]:.3f} pipes {shares[1]:.3f}",
        flush=True,
    )
    print(
        f"{'':>9}  first layer on the lamp: median {np.median(first):.4f} mm "
        f"(5-95 % {np.percentile(first, 5):.4f}-{np.percentile(first, 95):.4f}, n {first.size})  "
        f"non-orthogonality by the lamp max {angle[by_lamp].max():.1f} p99 "
        f"{np.percentile(angle[by_lamp], 99):.1f} deg; everywhere max {angle.max():.1f} p99 "
        f"{np.percentile(angle, 99):.1f} deg",
        flush=True,
    )


def main(sizes: list[float]) -> None:
    model = read_step(DRAWING, Placement(matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
    for size in sizes:
        print(f"maxh {size}", flush=True)
        mesher = CadMesher(max_size=size, feature_angle=0.785, layers=LAYERS)
        for label, mapped in (("triangles", ()), ("mapped", ("lamp", "pipes"))):
            start = time.perf_counter()
            mesh = mesher.mesh(model, WATER, PATCHES, layered=LAYERED, mapped=mapped)
            _report(model, mesh, time.perf_counter() - start, label)
            del mesh


if __name__ == "__main__":
    main([float(argument) for argument in sys.argv[1:]] or [0.02, 0.01])
