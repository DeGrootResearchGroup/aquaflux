"""Join two boundary patches of a mesh into one periodic seam.

A periodic domain -- a length of pipe whose outflow is its own inflow, a channel repeating along its
length -- is meshed as an ordinary domain with two boundary patches that are one surface seen from
each side: every face of one has a twin on the other, displaced by the domain's period. A join
matches the twins, keeps one face of each pair as an interior face between the two cells beside it,
and drops the other. The kept face carries the period as its
:attr:`~aquaflux.mesh.FaceCellConnectivity.neighbour_offset`, the translation of the neighbour's
centroid to its image beside the face, which is how every operator measures a distance across the
seam (the same representation a structured grid's ``periodic=`` builds directly).

The match is geometric: each face is placed by the mean of its vertices, the period is estimated
from the two patches' mean places, and each face of the second patch is paired with the nearest face
of the first once shifted by it. A pure translation moves a vertex mean exactly as it moves the
face, so a genuinely periodic pair matches to round-off; a pair that is not one -- different face
counts, a rotation, faces placed differently on the two sides -- fails the tolerance or the
one-to-one test and is refused.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial import cKDTree

from .carried import carried_cell_zones, carried_face_patches
from .mesh import Mesh

__all__ = [
    "PERIODIC_MATCH_TOLERANCE",
    "PeriodicJoin",
    "face_vertex_means",
    "matched_by_translation",
    "without_faces",
]

PERIODIC_MATCH_TOLERANCE = 1e-4
"""How far a face may land from its twin, as a fraction of the extent of the patch kept.

Relative rather than absolute, so it scales with the mesh: twins displaced by an exact translation
land to round-off, far inside it, and a neighbouring face lies a face's width away, far outside it.
"""


@dataclasses.dataclass(frozen=True)
class PeriodicJoin:
    """Two boundary patches of a mesh joined into one periodic seam.

    The faces of :attr:`first` become interior faces, each between its own cell and the cell beside
    its twin on :attr:`second`, whose faces are dropped. Both patches cease to be patches.

    Attributes
    ----------
    first, second : str
        The two patches, each the other displaced by the period.
    tolerance : float
        How far a face may land from its twin (see :data:`PERIODIC_MATCH_TOLERANCE`).

    Raises
    ------
    ValueError
        If the two patches are the same, or the tolerance is not positive.
    """

    first: str
    second: str
    tolerance: float = PERIODIC_MATCH_TOLERANCE

    def __post_init__(self) -> None:
        if self.first == self.second:
            raise ValueError(
                f"a periodic join needs two different patches, got {self.first!r} twice."
            )
        if not self.tolerance > 0.0:
            raise ValueError(f"PeriodicJoin.tolerance must be positive, got {self.tolerance!r}.")

    def join(self, mesh: Mesh) -> Mesh:
        """The mesh with the two patches joined.

        Parameters
        ----------
        mesh : Mesh
            A three-dimensional mesh holding both patches as boundary patches, not yet periodic.

        Returns
        -------
        Mesh
            The same nodes and cells, the second patch's faces dropped and the first's made
            interior; every other patch, type, group and zone carried over.

        Raises
        ------
        ValueError
            If either patch is missing, holds an interior face or holds no face, if the mesh is
            periodic already, or if the faces do not match one-to-one by a translation.
        """
        if mesh.face_cells.neighbour_offset is not None:
            raise ValueError(
                "this mesh is periodic already; join all of its periodic patches at once."
            )
        names = mesh.face_patches.names
        missing = [name for name in (self.first, self.second) if name not in names]
        if missing:
            raise ValueError(f"the mesh has no patch {', '.join(map(repr, missing))} to join.")
        neighbour = np.asarray(mesh.face_cells.neighbour)
        kept = np.nonzero(np.asarray(mesh.face_patches.mask(self.first)))[0]
        donor = np.nonzero(np.asarray(mesh.face_patches.mask(self.second)))[0]
        for name, faces in ((self.first, kept), (self.second, donor)):
            if faces.size == 0:
                raise ValueError(f"patch {name!r} holds no face to join.")
            if np.any(neighbour[faces] >= 0):
                raise ValueError(
                    f"patch {name!r} holds interior faces; only boundary patches join."
                )
        offsets = np.asarray(mesh.face_nodes.offsets)
        indices = np.asarray(mesh.face_nodes.face_node_indices)
        centroids = face_vertex_means(np.asarray(mesh.node_coords), offsets, indices)
        matched = matched_by_translation(
            centroids[kept], centroids[donor], (self.first, self.second), self.tolerance
        )
        twin = kept[matched]
        owner = np.asarray(mesh.face_cells.owner)
        neighbour = neighbour.copy()
        neighbour[twin] = owner[donor]
        offset = np.zeros((mesh.n_faces, mesh.dim))
        offset[twin] = centroids[twin] - centroids[donor]
        keep = np.ones(mesh.n_faces, dtype=bool)
        keep[donor] = False
        kept_faces = np.nonzero(keep)[0]
        new_offsets, new_indices = without_faces(offsets, indices, keep)
        removed = (self.first, self.second)
        patch_types, patch_groups = mesh.face_patches.without(removed)
        return Mesh.from_csr(
            np.asarray(mesh.node_coords),
            new_offsets,
            new_indices,
            owner[keep],
            neighbour[keep],
            n_cells=mesh.n_cells,
            cell_zones=carried_cell_zones(mesh),
            face_patches=carried_face_patches(mesh, removed, kept_faces),
            patch_types=patch_types,
            patch_groups=patch_groups,
            neighbour_offset=offset[keep],
        )


def face_vertex_means(points: np.ndarray, offsets: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """The mean of each face's vertices, over compressed-sparse-row (CSR) face-node lists.

    A translation moves a vertex mean exactly as it moves the face's true (area-weighted) centroid,
    since both are linear in the vertices, so it places a face for a periodic match at a fraction
    of the centroid's cost.

    Parameters
    ----------
    points : np.ndarray, shape ``(n_nodes, dim)``
    offsets : np.ndarray of int, shape ``(n_faces + 1,)``
        Each face's first entry in ``indices`` (the CSR row pointers).
    indices : np.ndarray of int, shape ``(offsets[-1],)``
        The faces' nodes, face after face.

    Returns
    -------
    np.ndarray, shape ``(n_faces, dim)``
    """
    counts = np.diff(offsets)
    face_of_node = np.repeat(np.arange(counts.size), counts)
    sums = np.zeros((counts.size, points.shape[1]))
    np.add.at(sums, face_of_node, points[indices])
    return sums / counts[:, None]


def matched_by_translation(
    kept: np.ndarray, donor: np.ndarray, names: tuple[str, str], tolerance: float
) -> np.ndarray:
    """Pair each face of one patch with its twin on another, displaced by one translation.

    Independent of the order of either patch's faces: the translation is estimated from the two
    patches' mean places, the donor faces are shifted by it and each is paired with the nearest kept
    face.

    Parameters
    ----------
    kept, donor : np.ndarray, shape ``(n, dim)``
        Where each face of the two patches is (:func:`face_vertex_means`).
    names : tuple of str
        The two patches' names, for the message.
    tolerance : float
        How far a face may land from its twin, as a fraction of the kept patch's extent.

    Returns
    -------
    np.ndarray of int, shape ``(n,)``
        For donor face ``i``, the index into ``kept`` of its twin.

    Raises
    ------
    ValueError
        If the patches hold different numbers of faces, or a face lands farther than the tolerance
        from its nearest, or two land on one face.
    """
    first, second = names
    if kept.shape[0] != donor.shape[0]:
        raise ValueError(
            f"patches '{first}'/'{second}' have different face counts ({kept.shape[0]} vs "
            f"{donor.shape[0]}), so they are not one periodic surface"
        )
    extent = float(np.linalg.norm(kept.max(axis=0) - kept.min(axis=0)))
    reach = tolerance * extent if extent > 0.0 else tolerance
    translation = kept.mean(axis=0) - donor.mean(axis=0)
    distance, matched = cKDTree(kept).query(donor + translation)
    if distance.max() > reach or len(np.unique(matched)) != matched.size:
        raise ValueError(
            f"patches '{first}'/'{second}' could not be matched face-for-face within tolerance "
            f"{reach:.3g} (worst mismatch {distance.max():.3g}); they may not be a pure "
            "translational pair"
        )
    return matched


def without_faces(
    offsets: np.ndarray, indices: np.ndarray, keep: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compressed-sparse-row (CSR) face-node lists with the faces ``keep`` excludes dropped.

    Parameters
    ----------
    offsets, indices : np.ndarray of int
        The CSR lists (see :func:`face_vertex_means`).
    keep : np.ndarray of bool, shape ``(n_faces,)``

    Returns
    -------
    tuple of np.ndarray
        The kept faces' row pointers and nodes, in their order.
    """
    counts = np.diff(offsets)
    new_offsets = np.concatenate([[0], np.cumsum(counts[keep])]).astype(offsets.dtype)
    face_of_node = np.repeat(np.arange(counts.size), counts)
    return new_offsets, indices[keep[face_of_node]]
