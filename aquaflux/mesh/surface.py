"""A mesh's boundary patches as triangles, wound to face into the domain.

A mesh already describes the surfaces that bound its domain: every boundary face is a polygon on
one of them, and the owner-outward normal says which side the domain is on. Handing those faces
out as triangles lets a model that works on a surface (the radiation gather, for one) use the
mesh's own boundary instead of a second description of the same geometry that can disagree with
it.

The faces are cut into the **centre fan** the mesh's own face geometry uses (triangles from each
face's vertex mean to its perimeter edges), so a planar face's triangles have exactly the face's
area and a warped face's differ from it only by the warp. Each triangle is wound so that its
right-hand normal points *into* the domain -- the reverse of the stored owner-outward normal of a
boundary face -- which is decided per face from that normal and so needs no guess from the node
order.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from aquaflux.vectors import dot

from .face import PolygonFaceGeometry
from .geometry import MeshGeometry
from .mesh import Mesh

__all__ = ["PatchTriangles", "patch_triangles"]


@dataclasses.dataclass(frozen=True)
class PatchTriangles:
    """The triangles of some boundary patches, and which face and patch each came from.

    Plain arrays: this is a build-time product, formed once, outside anything compiled.

    Attributes
    ----------
    vertices : np.ndarray, shape ``(n_triangles, 3, 3)``
        Triangle corners, wound so that ``(v1 - v0) x (v2 - v0)`` points into the domain.
    patch_id : np.ndarray of int, shape ``(n_triangles,)``
        Index into :attr:`patch_names` of the patch each triangle lies on.
    patch_names : tuple of str
        The patches, in the order they were asked for.
    face : np.ndarray of int, shape ``(n_triangles,)``
        The mesh face each triangle was cut from, for carrying a per-face quantity across.
    """

    vertices: np.ndarray
    patch_id: np.ndarray
    patch_names: tuple[str, ...]
    face: np.ndarray

    @property
    def n_triangles(self) -> int:
        """Number of triangles."""
        return int(self.vertices.shape[0])


def patch_triangles(
    mesh: Mesh, geometry: MeshGeometry, patch_names: Sequence[str]
) -> PatchTriangles:
    """Cut the faces of named boundary patches into triangles facing into the domain.

    Parameters
    ----------
    mesh : Mesh
        A 3D mesh. Supplies the nodes, the face→node rings and the patch lookup.
    geometry : MeshGeometry
        The mesh's geometry; only the owner-outward face normals are read, to orient each face.
    patch_names : sequence of str
        The boundary patches to triangulate, each at most once.

    Returns
    -------
    PatchTriangles

    Raises
    ------
    ValueError
        If the mesh is not 3D, no patch or a repeated patch is named, a patch is empty or holds
        interior faces (a baffle has two sides, and which one faces the domain is not a property of
        the face), or a face is not star-shaped from its vertex mean, so that its fan would overlap
        itself.
    """
    if mesh.dim != 3:
        raise ValueError(f"patch_triangles needs a 3D mesh; this one is {mesh.dim}D")
    names = tuple(patch_names)
    if not names:
        raise ValueError("patch_triangles: no patch names given")
    if len(set(names)) != len(names):
        raise ValueError(f"patch_triangles: a patch is named more than once in {list(names)}")

    faces = []
    for name in names:
        index = np.asarray(mesh.face_patches.indices(name))
        if index.size == 0:
            raise ValueError(f"patch '{name}' has no faces")
        if not mesh.face_patches.is_boundary_patch(name, mesh.face_cells):
            raise ValueError(
                f"patch '{name}' holds interior faces; only boundary patches have a single side "
                "facing the domain"
            )
        faces.append(index)
    face_index = np.concatenate(faces)
    patch_of_face = np.repeat(np.arange(len(names)), [len(f) for f in faces])

    selected = mesh.face_nodes.select(face_index)
    apex, start, end = (
        np.asarray(corner)
        for corner in PolygonFaceGeometry().fan_triangles(mesh.node_coords, selected)
    )
    owner = np.asarray(selected.face_of_incidence)
    # Twice each triangle's vector area in node order, against its face's owner-outward normal.
    # A boundary face's owner is the cell inside the domain, so a triangle facing into the domain
    # has a NEGATIVE projection; those that project positive are rewound.
    winding = np.cross(start - apex, end - apex)
    outward = np.asarray(geometry.face.normal)[face_index][owner]
    projection = np.asarray(dot(winding, outward))

    # Every triangle of a star-shaped face projects with one sign. One of the other sign means
    # the fan from the vertex mean folds back over the face, and the triangles would cover part
    # of it twice.
    face_sign = np.sign(np.bincount(owner, weights=projection, minlength=len(face_index)))
    folded = projection * face_sign[owner] <= 0.0
    if np.any(folded):
        bad = int(face_index[owner[np.argmax(folded)]])
        count = len(np.unique(owner[folded]))
        raise ValueError(
            f"{count} face(s) are not star-shaped from their vertex mean (face {bad} first), so "
            "their centre fan overlaps itself"
        )

    rewind = projection > 0.0
    first = np.where(rewind[:, None], end, start)
    second = np.where(rewind[:, None], start, end)
    return PatchTriangles(
        vertices=np.stack([apex, first, second], axis=1),
        patch_id=patch_of_face[owner],
        patch_names=names,
        face=face_index[owner],
    )
