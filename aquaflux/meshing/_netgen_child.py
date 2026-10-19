"""Run one Netgen operation in a process of its own: the only module that imports Netgen.

Netgen's Python package carries its own build of the OpenCASCADE geometry kernel, and the CAD reader
uses another, through OCP. Loading both into one interpreter aborts it at exit (each library's static
destructors free memory through the other's allocator), so Netgen never runs in a process that might
hold OCP. :class:`aquaflux.meshing.NetgenProcess` starts this file as a script, by path, in a fresh
interpreter -- so it imports neither ``aquaflux`` nor anything that could pull the CAD kernel in --
and exchanges plain arrays with it through files in a work directory:

* ``request.json``: ``{"operation": <name>, ...}`` with the operation's settings;
* ``input.npz``: the operation's arrays, when it takes any;
* ``result.npz``: written by this script on success. On failure it raises, so the process exits
  non-zero with the reason on its standard error.

Indices in every file are zero-based; Netgen's own are one-based and are converted here. Netgen's
arrays are views of its own memory, freed with its mesh, so every one is copied before it is kept.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _triangulate(work: Path, request: dict) -> dict[str, np.ndarray]:
    """Triangulate the faces of the solid in a BREP or STEP file, each triangle tagged by its face.

    Netgen numbers faces in the order a topology explorer visits them in the shape it reads, which is
    the order OCP's explorer visits them in the same file; the returned ``face`` is that order.
    Triangles are wound outward from the solid.
    """
    from netgen.meshing import MeshingStep
    from netgen.occ import OCCGeometry

    geometry = OCCGeometry(request["shape"])
    if len(geometry.shape.solids) != 1:
        raise ValueError(f"{request['shape']} holds {len(geometry.shape.solids)} solids, not one")
    mesh = geometry.GenerateMesh(
        perfstepsend=MeshingStep.MESHSURFACE,
        maxh=request["max_size"],
        minh=request["min_size"],
        grading=request["grading"],
    )
    elements = mesh.Elements2D().NumPy()
    if elements.size == 0:
        raise RuntimeError("Netgen produced no surface triangles")
    return {
        "points": np.array(mesh.Coordinates(), dtype=np.float64),
        "triangles": elements["nodes"].astype(np.int64) - 1,
        "face": elements["index"].astype(np.int64) - 1,
    }


def _fill(work: Path, request: dict) -> dict[str, np.ndarray]:
    """Fill the region a closed, outward-wound triangle surface bounds with tetrahedra.

    The supplied surface is kept exactly: its points come back first, at their indices and
    coordinates, and no boundary triangle is split or added. Tetrahedra come back positively oriented
    (the fourth vertex on the side the first three's right-hand normal points to).
    """
    from netgen.meshing import FaceDescriptor
    from netgen.meshing import Mesh as NetgenMesh

    with np.load(work / "input.npz") as arrays:
        points = np.ascontiguousarray(arrays["points"], dtype=np.float64)
        triangles = np.ascontiguousarray(arrays["triangles"], dtype=np.int32)
    mesh = NetgenMesh(dim=3)
    # The face descriptor says the region lies on the side opposite each triangle's normal.
    descriptor = mesh.Add(FaceDescriptor(surfnr=1, domin=1, domout=0, bc=1))
    mesh.AddPoints(points)
    mesh.AddElements(dim=2, index=descriptor, data=triangles, base=0)
    mesh.GenerateVolumeMesh(maxh=request["max_size"], grading=request["grading"])
    if mesh.ne == 0:
        raise RuntimeError("Netgen produced no tetrahedra (its own message is in the output above)")
    coordinates = np.array(mesh.Coordinates(), dtype=np.float64)
    if not np.array_equal(coordinates[: len(points)], points):
        raise RuntimeError("Netgen moved or renumbered a supplied point")
    if len(mesh.Elements2D()) != len(triangles):
        raise RuntimeError("Netgen changed the supplied surface")
    tetrahedra = mesh.Elements3D().NumPy()["nodes"].astype(np.int64) - 1
    corner = coordinates[tetrahedra]
    edges = corner[:, 1:] - corner[:, :1]
    volume = np.einsum("ij,ij->i", np.cross(edges[:, 0], edges[:, 1]), edges[:, 2]) / 6.0
    tetrahedra[volume < 0.0] = tetrahedra[volume < 0.0][:, [0, 2, 1, 3]]
    _check_fills(points, triangles, tetrahedra, np.abs(volume).sum())
    return {"points": coordinates, "tetrahedra": tetrahedra}


def _check_fills(points, triangles, tetrahedra, volume) -> None:
    """Raise unless the tetrahedra fill exactly the region the surface bounds.

    Netgen can return a mesh that keeps every supplied point and boundary element and still fills
    the wrong region: given a hole whose walls are wound the wrong way it ignores the hole and fills
    straight through it. So the result is checked directly: the faces used by only one tetrahedron
    must be the supplied triangles, and the tetrahedra's volume the volume the surface encloses.
    """
    faces = np.concatenate([tetrahedra[:, [1, 2, 3]], tetrahedra[:, [0, 3, 2]],
                            tetrahedra[:, [0, 1, 3]], tetrahedra[:, [0, 2, 1]]])  # fmt: skip
    keys, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    boundary = keys[counts == 1]
    supplied = np.unique(np.sort(triangles, axis=1), axis=0)
    if boundary.shape != supplied.shape or not np.array_equal(boundary, supplied):
        raise RuntimeError(
            f"Netgen's tetrahedra are bounded by {len(boundary)} triangles, not the {len(supplied)} "
            "supplied: the surface does not bound the region it was meant to (is a hole's wall wound "
            "out of the hole?)"
        )
    corner = points[triangles]
    enclosed = np.einsum("ij,ij->i", corner[:, 0], np.cross(corner[:, 1], corner[:, 2])).sum() / 6.0
    if not np.isclose(volume, enclosed, rtol=1e-9, atol=0.0):
        raise RuntimeError(
            f"Netgen's tetrahedra hold a volume of {volume:.12g}, the surface encloses {enclosed:.12g}"
        )


OPERATIONS = {"triangulate": _triangulate, "fill": _fill}


def main(directory: str) -> None:
    """Read the request in ``directory``, run it, and write ``result.npz`` there."""
    work = Path(directory)
    request = json.loads((work / "request.json").read_text())
    result = OPERATIONS[request["operation"]](work, request)
    np.savez(work / "result.npz", **result)


if __name__ == "__main__":
    main(sys.argv[1])
