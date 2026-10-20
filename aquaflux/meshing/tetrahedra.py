"""A face-based mesh of tetrahedra given by their corners, with its boundary in named patches."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from aquaflux.mesh import RESERVED_PATCH_NAMES, Mesh

__all__ = ["tetrahedral_mesh"]

#: The three corners of a tetrahedron's face opposite each of its four corners, in that order.
_OPPOSITE = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


def tetrahedral_mesh(
    points: np.ndarray, tetrahedra: np.ndarray, patches: Mapping[str, np.ndarray] | None = None
) -> Mesh:
    """The face-based mesh of tetrahedra, each triangle once, its boundary in named patches.

    The faces are numbered in the order they are first seen, tetrahedron by tetrahedron and, within
    one, opposite its corners in turn; the tetrahedron seeing a face first owns it, and its ring is
    the three corners as that tetrahedron lists them (a mesh orients its faces itself).

    Parameters
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
    tetrahedra : np.ndarray of int, shape ``(n_cells, 4)``
        Each tetrahedron's corners, in any order.
    patches : mapping of {str: np.ndarray of int, shape ``(n, 3)``}, optional
        The boundary triangles in each patch, as corner triples in any order. Given, every boundary
        face must be in exactly one; unset, the boundary is left unnamed.

    Returns
    -------
    Mesh

    Raises
    ------
    ValueError
        If a triangle is a face of more than two tetrahedra, or the patches do not cover the boundary
        once: a boundary face in no patch or two, or a patch triangle that is not a boundary face.
    """
    points = np.asarray(points, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    n_cells = tetrahedra.shape[0]
    corners = tetrahedra[:, _OPPOSITE].reshape(-1, 3)  # each tetrahedron's four faces, in turn
    keys = np.sort(corners, axis=1)
    # Stable, so the first of equal triangles is the one seen first.
    order = np.lexsort(keys.T[::-1])
    keys = keys[order]
    starts = np.flatnonzero(np.concatenate([[True], np.any(keys[1:] != keys[:-1], axis=1)]))
    counts = np.diff(np.concatenate([starts, [order.size]]))
    if np.any(counts > 2):
        raise ValueError(
            f"{int(np.count_nonzero(counts > 2))} triangle(s) are faces of more than two "
            "tetrahedra: the tetrahedra do not form a mesh"
        )
    first = order[starts]
    second = np.where(counts == 2, order[np.minimum(starts + 1, order.size - 1)], -1)
    seen = np.argsort(first)
    first, second, unique = first[seen], second[seen], keys[starts][seen]
    owner = first // 4
    neighbour = np.where(second >= 0, second // 4, -1)
    rings = corners[first]
    face_patches = None
    if patches is not None:
        face_patches = _patched(unique, neighbour < 0, patches)
    return Mesh.from_csr(
        points,
        np.arange(rings.shape[0] + 1) * 3,
        rings.reshape(-1),
        owner,
        neighbour,
        n_cells,
        face_patches=face_patches,
    )


def _patched(unique: np.ndarray, boundary: np.ndarray, patches) -> dict[str, np.ndarray]:
    """The faces of each patch, matched by sorted corner triple; refuses a gap or an overlap."""
    reserved = sorted(set(patches) & set(RESERVED_PATCH_NAMES))
    if reserved:
        raise ValueError(f"patch name(s) {reserved} are reserved for the mesh's own groups")
    bound = int(unique.max()) + 1 if unique.size else 1
    face_key = (unique[:, 0] * bound + unique[:, 1]) * bound + unique[:, 2]
    order = np.argsort(face_key)
    claimed = np.zeros(unique.shape[0], dtype=np.int64)
    found = {}
    for name, triangles in patches.items():
        triangles = np.sort(np.asarray(triangles, dtype=np.int64).reshape(-1, 3), axis=1)
        key = (triangles[:, 0] * bound + triangles[:, 1]) * bound + triangles[:, 2]
        position = np.minimum(np.searchsorted(face_key[order], key), order.size - 1)
        face = order[position]
        if np.any(face_key[face] != key) or not np.all(boundary[face]):
            raise ValueError(f"patch {name!r} holds a triangle that is not a boundary face")
        np.add.at(claimed, face, 1)
        found[name] = np.sort(face)
    if np.any(claimed[boundary] != 1):
        missing = int(np.count_nonzero(claimed[boundary] == 0))
        doubled = int(np.count_nonzero(claimed[boundary] > 1))
        raise ValueError(
            f"the patches do not cover the boundary once: {missing} boundary face(s) in none, "
            f"{doubled} in more than one"
        )
    return found
