"""Fuse OpenFOAM ``cyclic`` boundary-patch pairs into interior periodic seam faces.

A ``cyclic`` patch pair describes two geometrically-separate boundary planes that are really one
periodic seam: each patch's ``neighbourPatch`` entry names the other. The reader's own assembly
step (:mod:`.assembler`) turns every boundary patch into faces with ``neighbour = -1``, which is
correct for an ordinary boundary but wrong for a cyclic pair — it loses the periodicity. This module
is the fix: it matches the two patches' faces one-for-one by translated centroid, drops the
duplicate (donor) side, and turns the kept side's faces into interior faces carrying a
:attr:`~aquaflux.mesh.connectivity.FaceCellConnectivity.neighbour_offset` — the same periodic-image
mechanism :func:`~aquaflux.mesh.structured.structured_grid_2d`'s ``periodic=`` builds directly.

Everything here is a pure function of already-parsed arrays (no file I/O), so it is testable on a
hand-built pair of patches with no polyMesh files.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.spatial import cKDTree

from .records import FoamPatch, patch_face_range

DEFAULT_MATCH_TOLERANCE = 1e-4
"""Default cyclic-face match tolerance, as a fraction of the kept patch's own bounding extent.

Mirrors OpenFOAM's own ``matchTolerance`` cyclic-patch entry in spirit (a relative, not absolute,
tolerance — so it scales with mesh size), but is not read from the file: the geometric match below
verifies the pair *is* a pure translation rather than trusting a declared transform type, so no
``transform``/``matchTolerance`` parsing is needed to reach the same guarantee.
"""


class _PatchFace(NamedTuple):
    """One cyclic patch, with its face block resolved to global face indices."""

    patch: FoamPatch
    faces: np.ndarray


class CyclicFusion(NamedTuple):
    """The face arrays after fusing every matched cyclic pair, ready for ``Mesh.from_csr``.

    Attributes
    ----------
    face_node_offsets, face_node_indices : np.ndarray
        The CSR face-node connectivity with every donor-side face dropped.
    owner, neighbour : np.ndarray
        Owner / neighbour cell index per surviving face — a kept cyclic face now has a real
        neighbour instead of ``-1``.
    neighbour_offset : np.ndarray or None, shape ``(n_faces, dim)``
        The periodic-image translation on every surviving face (nonzero only on the fused seam
        faces), or ``None`` when there were no cyclic pairs to fuse — an ordinary mesh stays
        offset-free rather than gaining an array of zeros.
    patches : tuple of FoamPatch
        The non-cyclic patches, renumbered onto the surviving faces; every cyclic patch (both the
        kept and donor sides) is gone — its faces are interior now, not a named patch.
    """

    face_node_offsets: np.ndarray
    face_node_indices: np.ndarray
    owner: np.ndarray
    neighbour: np.ndarray
    neighbour_offset: np.ndarray | None
    patches: tuple[FoamPatch, ...]


def fuse_cyclic_patches(
    points: np.ndarray,
    face_node_offsets: np.ndarray,
    face_node_indices: np.ndarray,
    owner: np.ndarray,
    neighbour: np.ndarray,
    patches: tuple[FoamPatch, ...],
    *,
    match_tolerance: float = DEFAULT_MATCH_TOLERANCE,
) -> CyclicFusion:
    """Fuse every matched ``cyclic`` patch pair in ``patches`` into interior seam faces.

    Parameters
    ----------
    points : np.ndarray
        Node coordinates, shape ``(n_nodes, dim)``.
    face_node_offsets, face_node_indices : np.ndarray
        The CSR face-node connectivity for all ``n_faces`` faces.
    owner, neighbour : np.ndarray
        Owner / neighbour cell index per face, already padded to full length (``neighbour == -1``
        on every boundary face, cyclic patches included).
    patches : tuple of FoamPatch
        Every boundary patch, in file order.
    match_tolerance : float
        Passed to the face-matching search (see :data:`DEFAULT_MATCH_TOLERANCE`).

    Returns
    -------
    CyclicFusion

    Raises
    ------
    ValueError
        If a ``cyclic`` patch has no ``neighbourPatch``, names a patch that does not exist or is
        not itself cyclic, does not name it back, names itself, or if the two patches' faces
        cannot be matched one-for-one within tolerance (different face counts, or not a pure
        translation).
    """
    pairs = _cyclic_pairs(patches)
    if not pairs:
        return CyclicFusion(face_node_offsets, face_node_indices, owner, neighbour, None, patches)

    n_faces = owner.shape[0]
    dim = points.shape[1]
    centroids = _face_centroids(points, face_node_offsets, face_node_indices)

    new_neighbour = neighbour.copy()
    full_offset = np.zeros((n_faces, dim))
    donor_blocks = []
    fused_names: set[str] = set()

    for kept, donor in pairs:
        if kept.faces.size != donor.faces.size:
            raise ValueError(
                f"cyclic patches '{kept.patch.name}'/'{donor.patch.name}' have different face "
                f"counts ({kept.faces.size} vs {donor.faces.size})"
            )
        if kept.faces.size == 0:
            raise ValueError(
                f"cyclic patches '{kept.patch.name}'/'{donor.patch.name}' have no faces to fuse"
            )
        matched = _match_faces(
            centroids[kept.faces],
            centroids[donor.faces],
            kept.patch.name,
            donor.patch.name,
            match_tolerance,
        )
        kept_global = kept.faces[matched]
        new_neighbour[kept_global] = owner[donor.faces]
        full_offset[kept_global] = centroids[kept_global] - centroids[donor.faces]
        donor_blocks.append(donor.faces)
        fused_names.add(kept.patch.name)
        fused_names.add(donor.patch.name)

    keep_mask = np.ones(n_faces, dtype=bool)
    keep_mask[np.concatenate(donor_blocks)] = False
    new_index = np.cumsum(keep_mask) - 1

    new_offsets, new_indices = _compress_face_node_csr(
        face_node_offsets, face_node_indices, keep_mask
    )
    surviving_patches = tuple(
        patch._replace(start_face=int(new_index[patch.start_face]))
        for patch in patches
        if patch.name not in fused_names
    )
    return CyclicFusion(
        new_offsets,
        new_indices,
        owner[keep_mask],
        new_neighbour[keep_mask],
        full_offset[keep_mask],
        surviving_patches,
    )


def _cyclic_pairs(patches: tuple[FoamPatch, ...]) -> list[tuple[_PatchFace, _PatchFace]]:
    """Every matched cyclic patch pair, kept side first.

    The patch declared earlier in the boundary file is the "kept" side — its faces stay where they
    are (now interior); the other's faces are dropped as duplicates. This is the only tie-break
    needed: fusion is symmetric under which side is "kept" (see the module-level offset derivation
    in :func:`fuse_cyclic_patches`), so declaration order just needs to be *deterministic*.
    """
    by_name = {patch.name: (index, patch) for index, patch in enumerate(patches)}
    paired: set[str] = set()
    pairs: list[tuple[_PatchFace, _PatchFace]] = []
    for index, patch in enumerate(patches):
        if patch.type_ != "cyclic" or patch.name in paired:
            continue
        if not patch.neighbour_patch:
            raise ValueError(f"cyclic patch '{patch.name}' has no neighbourPatch entry")
        if patch.neighbour_patch == patch.name:
            raise ValueError(f"cyclic patch '{patch.name}' names itself as its own neighbourPatch")
        partner_entry = by_name.get(patch.neighbour_patch)
        if partner_entry is None:
            raise ValueError(
                f"cyclic patch '{patch.name}' names neighbourPatch '{patch.neighbour_patch}', "
                "which does not exist"
            )
        partner_index, partner = partner_entry
        if partner.type_ != "cyclic":
            raise ValueError(
                f"cyclic patch '{patch.name}' pairs with '{partner.name}', which is not itself a "
                f"cyclic patch (type '{partner.type_}')"
            )
        if partner.neighbour_patch != patch.name:
            raise ValueError(
                f"cyclic patch '{patch.name}' names neighbourPatch '{patch.neighbour_patch}', but "
                f"'{partner.name}' names '{partner.neighbour_patch}' instead of '{patch.name}'"
            )
        paired.add(patch.name)
        paired.add(partner.name)
        kept, donor = (patch, partner) if index < partner_index else (partner, patch)
        pairs.append(
            (_PatchFace(kept, patch_face_range(kept)), _PatchFace(donor, patch_face_range(donor)))
        )
    return pairs


def _face_centroids(points: np.ndarray, offsets: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Vertex-mean centroid of every face, vectorized over the ragged CSR node lists.

    A pure translation shifts a vertex mean by exactly the same vector it shifts the true
    (area-weighted) centroid, since both are linear in the vertex positions — so this cheap
    estimate gives the *exact* seam translation for a genuinely periodic pair, with no per-face
    Python loop.
    """
    counts = np.diff(offsets)
    face_of_node = np.repeat(np.arange(counts.size), counts)
    sums = np.zeros((counts.size, points.shape[1]))
    np.add.at(sums, face_of_node, points[indices])
    return sums / counts[:, None]


def _match_faces(
    kept_centroids: np.ndarray,
    donor_centroids: np.ndarray,
    kept_name: str,
    donor_name: str,
    match_tolerance: float,
) -> np.ndarray:
    """Pair each donor face to its kept-side counterpart by translated nearest centroid.

    Robust to face ordering within each patch (OpenFOAM does not guarantee the two patches'
    faces are listed in corresponding order): estimate the seam translation from the two patches'
    centroid means, shift the donor centroids by it, and match each to its nearest kept centroid.
    A genuinely periodic pair lands every match within a tight tolerance of its true counterpart
    and uses each kept face exactly once; anything else — a face-count mismatch, a rotational or
    non-uniform cyclic transform, mismatched patches — fails one of those checks.

    Returns
    -------
    np.ndarray of int, shape ``(n_donor,)``
        For donor face ``i``, the index into ``kept_centroids`` of its matched pair.

    Raises
    ------
    ValueError
        If the nearest-centroid match is not within tolerance, or is not one-to-one.
    """
    extent = float(np.linalg.norm(kept_centroids.max(axis=0) - kept_centroids.min(axis=0)))
    tolerance = match_tolerance * extent if extent > 0.0 else match_tolerance
    translation = kept_centroids.mean(axis=0) - donor_centroids.mean(axis=0)
    tree = cKDTree(kept_centroids)
    distance, matched = tree.query(donor_centroids + translation)
    if distance.max() > tolerance or len(np.unique(matched)) != matched.size:
        raise ValueError(
            f"cyclic patches '{kept_name}'/'{donor_name}' could not be matched face-for-face "
            f"within tolerance {tolerance:.3g} (worst mismatch {distance.max():.3g}); they may not "
            "be a pure translational pair"
        )
    return matched


def _compress_face_node_csr(
    offsets: np.ndarray, indices: np.ndarray, keep_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop the CSR rows (faces) ``keep_mask`` excludes, preserving every other row's order."""
    counts = np.diff(offsets)
    new_offsets = np.concatenate([[0], np.cumsum(counts[keep_mask])]).astype(offsets.dtype)
    face_of_node = np.repeat(np.arange(counts.size), counts)
    return new_offsets, indices[keep_mask[face_of_node]]
