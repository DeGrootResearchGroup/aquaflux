"""The boundary patches of a meshed room, as triangles and face records, for every script here.

Both solvers must see one geometry, so the aquaflux gather and the reference read the lamp, the
floor and the bunny from the **same mesh** the discrete-ordinates method ran on, not from the
STL it was snapped to. This module reads them once, with aquaflux's own OpenFOAM parser, and
writes ``work/<mesh>/patches.npz``; the reference then loads that file without importing aquaflux.

Per patch it stores the faces' centres, outward (out of the air) unit normals and areas, computed
the way OpenFOAM computes them -- a fan of triangles about the mean of the face's points, the
centre their area-weighted centroid -- and those fan triangles themselves, wound as the face is.

Run as a script to (re)write the file for one mesh: ``python patches.py bunny``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
PATCHES = ("floor", "lamp", "bunny", "ceiling", "walls")


def npz_path(mesh: str) -> Path:
    return WORK / mesh / "patches.npz"


def face_records(points, offsets, indices, faces) -> dict[str, np.ndarray]:
    """Centres, unit normals, areas and fan triangles of the listed faces.

    Parameters
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
    offsets, indices : np.ndarray
        The mesh's faces in compressed-sparse-row (CSR) form: face ``f``'s points are
        ``indices[offsets[f]:offsets[f + 1]]``, in the order that gives its normal.
    faces : np.ndarray of int, shape ``(n,)``
        The faces wanted.

    Returns
    -------
    dict
        ``centre`` ``(n, 3)``, ``normal`` ``(n, 3)``, ``area`` ``(n,)``, and ``triangles``
        ``(n_triangles, 3, 3)`` with ``triangle_face`` ``(n_triangles,)`` naming each one's face.
    """
    start, stop = offsets[faces], offsets[faces + 1]
    count = stop - start
    owner = np.repeat(np.arange(len(faces)), count)
    rank = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    corner = points[indices[np.repeat(start, count) + rank]]
    following = points[indices[np.repeat(start, count) + (rank + 1) % np.repeat(count, count)]]
    mean = np.zeros((len(faces), 3))
    np.add.at(mean, owner, corner)
    mean /= count[:, None]
    # One triangle per edge, from the edge to the mean point.
    area_vector = 0.5 * np.cross(corner - mean[owner], following - mean[owner])
    centroid = (corner + following + mean[owner]) / 3.0
    vector = np.zeros((len(faces), 3))
    np.add.at(vector, owner, area_vector)
    area = np.linalg.norm(vector, axis=1)
    normal = vector / area[:, None]
    # Weight each triangle's centroid by its area projected on the face normal, as OpenFOAM does,
    # so a warped face's centre is not pulled by triangles folded against it.
    weight = np.einsum("ij,ij->i", area_vector, normal[owner])
    centre = np.zeros((len(faces), 3))
    np.add.at(centre, owner, weight[:, None] * centroid)
    total = np.zeros(len(faces))
    np.add.at(total, owner, weight)
    centre /= total[:, None]
    triangles = np.stack([corner, following, mean[owner]], axis=1)
    return {
        "centre": centre,
        "normal": normal,
        "area": area,
        "triangles": triangles,
        "triangle_face": owner,
    }


def extract(mesh: str) -> Path:
    """Read the named mesh's patches and write them to ``patches.npz``."""
    sys.path.insert(0, str(HERE.parents[1]))
    from aquaflux.io import OpenFOAMReader

    data = OpenFOAMReader(WORK / mesh / "case").read_polymesh()
    # A cell may appear only as a neighbour (the highest-numbered ones often do), so the count is
    # taken over both label lists, not the owners alone.
    last = max(int(data.owner.max()), int(data.neighbour_internal.max(initial=-1)))
    arrays = {"n_cells": np.array(last + 1)}
    for patch in data.patches:
        if patch.name not in PATCHES:
            continue
        faces = np.arange(patch.start_face, patch.start_face + patch.n_faces)
        for key, value in face_records(
            data.points, data.face_node_offsets, data.face_node_indices, faces
        ).items():
            arrays[f"{patch.name}/{key}"] = value
    path = npz_path(mesh)
    np.savez_compressed(path, **arrays)
    return path


def load(mesh: str) -> dict[str, dict[str, np.ndarray]]:
    """``{patch: {centre, normal, area, triangles, triangle_face}}`` from ``patches.npz``."""
    stored = np.load(npz_path(mesh))
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in stored.files:
        if "/" in key:
            patch, name = key.split("/")
            out.setdefault(patch, {})[name] = stored[key]
    out["_mesh"] = {"n_cells": int(stored["n_cells"])}
    return out


def patch_values(field: Path, patch: str, n_cells: int, n_faces: int) -> np.ndarray:
    """The values of one patch of a ``volScalarField`` file, shape ``(n_faces,)``."""
    sys.path.insert(0, str(HERE.parents[1]))
    from aquaflux.io.openfoam.fields import parse_scalar_field
    from aquaflux.io.openfoam.foamfile import read_foam_body

    return parse_scalar_field(read_foam_body(field), n_cells, {patch: n_faces})[n_cells:]


if __name__ == "__main__":
    for name in sys.argv[1:] or ["bunny", "empty"]:
        print(extract(name), flush=True)
