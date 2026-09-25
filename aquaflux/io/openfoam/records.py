"""Plain value records for a parsed OpenFOAM polyMesh.

These are build-time carriers between the text parsers and the mesh assembler — ordinary
``NamedTuple``\\s holding numpy arrays, not JAX pytrees. Bundling the parsed arrays into
:class:`PolyMeshData` lets the assembler take one cohesive object rather than a fistful of loose
arrays.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class FoamPatch(NamedTuple):
    """One boundary patch from the polyMesh ``boundary`` file.

    Attributes
    ----------
    name : str
        Patch name (becomes an aquaflux face-patch name).
    type_ : str
        Patch type as declared in the file (``wall`` / ``patch`` / ``empty`` / ``symmetry`` /
        ``cyclic`` / …). Carried through, not interpreted, except that ``empty`` marks a face plane
        to collapse away for a two-dimensional case, and ``cyclic`` marks a pair of patches to fuse
        into interior periodic seam faces (see :mod:`.cyclic`).
    start_face : int
        Index of the patch's first face. OpenFOAM orders faces so a patch owns the contiguous block
        ``[start_face, start_face + n_faces)``.
    n_faces : int
        Number of faces in the patch.
    neighbour_patch : str
        For a ``cyclic`` patch, the name of the patch it is paired with. Empty for every other
        patch type, and for a ``cyclic`` patch whose ``boundary`` entry omits ``neighbourPatch``
        (which :func:`.cyclic.fuse_cyclic_patches` rejects).
    in_groups : tuple of str
        The patch groups the ``inGroups`` entry puts this patch in, in the order written. Empty when the
        entry is absent.
    """

    name: str
    type_: str
    start_face: int
    n_faces: int
    neighbour_patch: str = ""
    in_groups: tuple[str, ...] = ()


def patch_face_range(patch: FoamPatch) -> np.ndarray:
    """Global face indices covered by one patch's contiguous face block.

    A boundary patch owns ``[start_face, start_face + n_faces)`` — the one arange this shape
    implies, shared by the assembler's patch-naming step and the cyclic-patch fusion, which both
    need "the face indices this patch owns" as a plain array.

    Parameters
    ----------
    patch : FoamPatch

    Returns
    -------
    np.ndarray of int, shape ``(patch.n_faces,)``
    """
    return np.arange(patch.start_face, patch.start_face + patch.n_faces)


class CellZone(NamedTuple):
    """One cell zone from the polyMesh ``cellZones`` file.

    Attributes
    ----------
    name : str
        Zone name (becomes an aquaflux cell-zone name).
    cell_labels : np.ndarray
        Cell indices in the zone, shape ``(n_zone_cells,)``.
    """

    name: str
    cell_labels: np.ndarray


class PolyMeshData(NamedTuple):
    """The raw arrays and records parsed from a polyMesh directory, before assembly into a ``Mesh``.

    Attributes
    ----------
    points : np.ndarray
        Node coordinates, shape ``(n_nodes, 3)``.
    face_node_offsets : np.ndarray
        Compressed-sparse-row (CSR) row pointers for the ragged face node lists, shape
        ``(n_faces + 1,)``.
    face_node_indices : np.ndarray
        Flat concatenation of every face's node indices, in perimeter order.
    owner : np.ndarray
        Owner cell index per face, shape ``(n_faces,)``.
    neighbour_internal : np.ndarray
        Neighbour cell index for the interior faces only, shape ``(n_internal_faces,)`` — the raw
        ``neighbour`` file length. The assembler pads it to full length with the boundary sentinel.
    patches : tuple of FoamPatch
        Boundary patches in file order.
    cell_zones : tuple of CellZone
        Cell zones (empty if the mesh has no ``cellZones`` file).
    """

    points: np.ndarray
    face_node_offsets: np.ndarray
    face_node_indices: np.ndarray
    owner: np.ndarray
    neighbour_internal: np.ndarray
    patches: tuple[FoamPatch, ...]
    cell_zones: tuple[CellZone, ...]
