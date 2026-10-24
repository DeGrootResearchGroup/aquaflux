"""A face-based mesh of tetrahedra, triangular prisms, hexahedra, pyramids and tents given by their
corners, its boundary in named patches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from aquaflux.mesh import RESERVED_PATCH_NAMES, Mesh

__all__ = ["element_mesh"]

#: A tetrahedron's faces, face ``k`` opposite corner ``k``.
TETRAHEDRON_FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
#: A prism's faces: its two triangles, then its three quadrilateral sides, each ring in order.
_PRISM = [[0, 1, 2], [3, 4, 5], [0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]]
#: A hexahedron's corners as the corners of the unit cube, in the order its corners are given: one
#: quadrilateral ``(0,0,0), (1,0,0), (1,1,0), (0,1,0)``, then the opposite one corner over corner.
HEXAHEDRON_CORNERS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
)
#: A hexahedron's faces: its two given quadrilaterals, then its four sides, each ring in order.
_HEXAHEDRON = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
#: A pyramid's faces: its quadrilateral, then its four triangles, each ring in order.
_PYRAMID = [[0, 1, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
#: A tent's faces: its two quadrilaterals standing on its edge, then its four triangles, each ring
#: in order.
_TENT = [[0, 1, 2, 3], [1, 0, 5, 2], [0, 3, 4], [0, 4, 5], [2, 4, 3], [2, 5, 4]]


def element_mesh(
    points: np.ndarray,
    tetrahedra: np.ndarray | None = None,
    prisms: np.ndarray | None = None,
    hexahedra: np.ndarray | None = None,
    pyramids: np.ndarray | None = None,
    tents: np.ndarray | None = None,
    patches: Mapping[str, np.ndarray | Sequence[np.ndarray]] | None = None,
    patch_types: Mapping[str, str] | None = None,
) -> Mesh:
    """The face-based mesh of tetrahedra, prisms, hexahedra, pyramids and tents, each face once,
    its boundary in named patches.

    The cells are the tetrahedra, then the prisms, the hexahedra, the pyramids and the tents. The
    faces are
    numbered in the order they are first seen, cell by cell and, within one, in the order its faces
    are listed (a tetrahedron's opposite its corners in turn; a prism's two triangles, then its
    sides; a hexahedron's two given quadrilaterals, then its sides; a pyramid's quadrilateral, then
    its triangles; a tent's quadrilaterals, then its triangles); the cell seeing a face first owns it, and its ring is that cell's (a mesh
    orients its faces itself).

    Parameters
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
    tetrahedra : np.ndarray of int, shape ``(n, 4)``, optional
        Each tetrahedron's corners, in any order.
    prisms : np.ndarray of int, shape ``(n, 6)``, optional
        Each prism's corners: one triangle ``a, b, c``, then the other ``d, e, f`` with ``d`` joined to
        ``a``, ``e`` to ``b`` and ``f`` to ``c`` by the prism's edges.
    hexahedra : np.ndarray of int, shape ``(n, 8)``, optional
        Each hexahedron's corners: one quadrilateral ``a, b, c, d`` in order round it, then the
        opposite one ``e, f, g, h`` with ``e`` joined to ``a``, ``f`` to ``b``, ``g`` to ``c`` and
        ``h`` to ``d`` by the hexahedron's edges.
    pyramids : np.ndarray of int, shape ``(n, 5)``, optional
        Each pyramid's corners: its quadrilateral ``a, b, c, d`` in order round it, then its apex.
    tents : np.ndarray of int, shape ``(n, 6)``, optional
        Each tent's corners ``p, q, r, a, m, b``: a cell standing on the edge ``p q``, with two
        quadrilaterals ``p, q, r, a`` and ``q, p, b, r`` through it and four triangles,
        ``p, a, m`` and ``p, m, b`` at ``p``'s end, ``r, m, a`` and ``r, b, m`` on top. It is
        what is left of two triangular prisms over triangles ``q, a', m'`` and ``q, m', b'`` of
        no area once their three corners ``a', m', b'`` at one place are made the one point
        ``p``: the two prisms would then share two faces. The corner ``q`` has two faces only.
    patches : mapping of {str: np.ndarray of int, or a sequence of them}, optional
        The boundary faces in each patch, as corners -- triangles ``(n, 3)``, quadrilaterals
        ``(n, 4)``, or a sequence of such arrays -- in any order. Given, every boundary face must be
        in exactly one; unset, the boundary is left unnamed.
    patch_types : mapping of {str: str}, optional
        A type for named patches (``wall``, ``patch``, ``symmetry``, ...), carried on the mesh's
        patches uninterpreted, as a mesh file's declared types are.

    Returns
    -------
    Mesh

    Raises
    ------
    ValueError
        If a face is a face of more than two cells, or the patches do not cover the boundary once: a
        boundary face in no patch or two, or a patch face that is not a boundary face.
    """
    points = np.asarray(points, dtype=np.float64)
    kinds = [
        (np.zeros((0, 4), np.int64) if tetrahedra is None else tetrahedra, TETRAHEDRON_FACES),
        (np.zeros((0, 6), np.int64) if prisms is None else prisms, _PRISM),
        (np.zeros((0, 8), np.int64) if hexahedra is None else hexahedra, _HEXAHEDRON),
        (np.zeros((0, 5), np.int64) if pyramids is None else pyramids, _PYRAMID),
        (np.zeros((0, 6), np.int64) if tents is None else tents, _TENT),
    ]
    rings, base = [], 0
    for corners, faces in kinds:
        corners = np.asarray(corners, dtype=np.int64)
        for local, face in enumerate(faces):
            ring = np.full((corners.shape[0], 4), -1, dtype=np.int64)
            ring[:, : len(face)] = corners[:, face]
            rings.append((ring, base + np.arange(corners.shape[0]), local, len(faces)))
        base += corners.shape[0]
    n_cells = base
    # Every face of every cell, in the order a walk through the cells, then their faces, sees them.
    corners = np.concatenate([ring for ring, *_ in rings])
    owner_of = np.concatenate([cells for _, cells, *_ in rings])
    sighting = np.concatenate([cells * 8 + local for _, cells, local, _ in rings])
    order_seen = np.argsort(sighting, kind="stable")
    corners, owner_of = corners[order_seen], owner_of[order_seen]
    keys = face_keys(corners, points.shape[0])
    order = np.lexsort(keys.T[::-1])  # stable: the first of equal faces is the one seen first
    keys = keys[order]
    starts = np.flatnonzero(np.concatenate([[True], np.any(keys[1:] != keys[:-1], axis=1)]))
    counts = np.diff(np.concatenate([starts, [order.size]]))
    if np.any(counts > 2):
        raise ValueError(
            f"{int(np.count_nonzero(counts > 2))} face(s) belong to more than two cells: the "
            "cells do not form a mesh"
        )
    first = order[starts]
    second = np.where(counts == 2, order[np.minimum(starts + 1, order.size - 1)], -1)
    seen = np.argsort(first)
    first, second, unique = first[seen], second[seen], keys[starts][seen]
    owner = owner_of[first]
    neighbour = np.where(second >= 0, owner_of[np.maximum(second, 0)], -1)
    ring = corners[first]
    size = np.count_nonzero(ring >= 0, axis=1)
    face_patches = None
    if patches is not None:
        face_patches = _patched(unique, neighbour < 0, patches, points.shape[0])
    return Mesh.from_csr(
        points,
        np.concatenate([[0], np.cumsum(size)]),
        ring[ring >= 0],
        owner,
        neighbour,
        n_cells,
        face_patches=face_patches,
        patch_types=patch_types,
    )


def face_keys(corners: np.ndarray, n_points: int) -> np.ndarray:
    """A face's identity: its corners sorted, a triangle's fourth slot past every point index."""
    padded = np.where(corners < 0, n_points, corners)
    return np.sort(padded, axis=1)


def _patched(unique, boundary, patches, n_points) -> dict[str, np.ndarray]:
    """The faces of each patch, matched by sorted corners; refuses a gap or an overlap."""
    reserved = sorted(set(patches) & set(RESERVED_PATCH_NAMES))
    if reserved:
        raise ValueError(f"patch name(s) {reserved} are reserved for the mesh's own groups")
    order = np.lexsort(unique.T[::-1])
    ordered = unique[order]
    claimed = np.zeros(unique.shape[0], dtype=np.int64)
    found = {}
    for name, given in patches.items():
        arrays = [given] if isinstance(given, np.ndarray) else list(given)
        wanted = []
        for faces in arrays:
            faces = np.asarray(faces, dtype=np.int64)
            padded = np.full((faces.shape[0], 4), -1, dtype=np.int64)
            padded[:, : faces.shape[1]] = faces
            wanted.append(face_keys(padded, n_points))
        wanted = np.concatenate(wanted) if wanted else np.zeros((0, 4), np.int64)
        position = search_rows(ordered, wanted)
        face = order[np.minimum(position, order.size - 1)]
        if np.any(np.any(unique[face] != wanted, axis=1)) or not np.all(boundary[face]):
            raise ValueError(f"patch {name!r} holds a face that is not a boundary face")
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


def search_rows(ordered: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Where each row of ``wanted`` sits among the lexicographically sorted rows of ``ordered``."""
    both = np.concatenate([ordered, wanted])
    rank = np.empty(both.shape[0], dtype=np.int64)
    # Rows of ``ordered`` first among equals, so an equal wanted row lands on it.
    is_wanted = np.concatenate([np.zeros(len(ordered), bool), np.ones(len(wanted), bool)])
    sorter = np.lexsort(np.vstack([is_wanted, both.T[::-1]]))
    rank[sorter] = np.arange(both.shape[0])
    before = np.cumsum(~is_wanted[sorter])  # rows of ``ordered`` up to and including each position
    return before[rank[len(ordered) :]] - 1
