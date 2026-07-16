"""Distance from cells to named boundary patches — the geometric field wall models need.

A near-wall model needs each cell's distance to the wall. :func:`distance_to_patches` returns, per
cell, the distance from the cell centroid to the nearest face centroid among a named set of boundary
patches (the wall patches, in that use).

This is the nearest-face-*centroid* distance, an approximation to the true distance to the wall
surface: for a wall-adjacent cell it is essentially the centroid's normal distance to its own wall
face — what near-wall models depend on — and it loosens on coarser cells farther from the wall. It
is a function of the static mesh geometry, so it is computed once at build time and reused as a
frozen field. (The patch-to-face lookup is data-dependent, so it runs eagerly, not under ``jit``.)

The nearest face is found by materializing the full cell-by-target-face offset array, so the working
memory scales with the cell count times the number of target faces; a spatial index would replace
this for very large meshes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquaflux.vectors import norm_squared

if TYPE_CHECKING:
    from .geometry import MeshGeometry
    from .mesh import Mesh


def distance_to_patches(
    mesh: Mesh, geometry: MeshGeometry, patch_names: Sequence[str]
) -> jnp.ndarray:
    """Per-cell distance to the nearest boundary face in the named patches, shape ``(n_cells,)``.

    Parameters
    ----------
    mesh : Mesh
        Supplies the boundary patch -> face-index lookup (``mesh.face_patches``).
    geometry : MeshGeometry
        Supplies the cell and face centroids.
    patch_names : sequence of str
        The boundary patches whose faces the distance is measured to (e.g. the wall patches).

    Returns
    -------
    jnp.ndarray
        The distance from each cell centroid to the nearest face centroid among the named patches,
        shape ``(n_cells,)``.

    Raises
    ------
    ValueError
        If ``patch_names`` is empty, names a patch the mesh does not have, or the named patches
        contain no faces.
    """
    if not patch_names:
        raise ValueError("distance_to_patches: no patch names given")
    face_index = jnp.concatenate([mesh.face_patches.indices(name) for name in patch_names])
    if face_index.shape[0] == 0:
        raise ValueError(f"distance_to_patches: patches {list(patch_names)} contain no faces")
    target = geometry.face.centroid[face_index]  # (n_target, dim)
    offset = geometry.cell.centroid[:, None, :] - target[None, :, :]  # (n_cells, n_target, dim)
    # Compare squared distances (cheaper, same argmin), take the sqrt of the nearest.
    return jnp.sqrt(jnp.min(norm_squared(offset), axis=1))
