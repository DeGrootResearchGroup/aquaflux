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
        Patch type as declared in the file (``wall`` / ``patch`` / ``empty`` / ``symmetry`` / …).
        Carried through, not interpreted, except that ``empty`` marks a face plane to collapse away
        for a two-dimensional case.
    start_face : int
        Index of the patch's first face. OpenFOAM orders faces so a patch owns the contiguous block
        ``[start_face, start_face + n_faces)``.
    n_faces : int
        Number of faces in the patch.
    """

    name: str
    type_: str
    start_face: int
    n_faces: int


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
