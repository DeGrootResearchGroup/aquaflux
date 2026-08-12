"""Distance from cells to named boundary patches — the geometric field wall models need.

A near-wall model needs each cell's distance to the wall. :func:`distance_to_patches` returns, per
cell, the distance from the cell centroid to the nearest face centroid among a named set of boundary
patches (the wall patches, in that use).

This is the nearest-face-*centroid* distance, an approximation to the true distance to the wall
surface: for a wall-adjacent cell it is essentially the centroid's normal distance to its own wall
face — what near-wall models depend on — and it loosens on coarser cells farther from the wall. It
is a function of the static mesh geometry, so it is computed once at build time and reused as a
frozen field. (The patch-to-face lookup is data-dependent, so it runs eagerly, not under ``jit``.)

The nearest face is found by direct search over the target faces, **a block of cells at a time**. The
whole-array form of that search is one expression, but it is a dense cell-by-target-face structure and
it does not survive contact with a three-dimensional case: on a 23040-cell mesh with 4736 wall faces
it needs 109 million entries, and because the squared magnitude materializes its own product, a little
over 6 GB is live at once — far more than the mesh, the geometry and the assembled case put together.
Blocking the search bounds that by construction while computing exactly the same numbers in the same
order, since each cell's nearest face is independent of every other cell's.
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
    return jnp.sqrt(_nearest_squared_distance(geometry.cell.centroid, target))


#: Working set the blocked search is allowed per block, in bytes. It buys nothing but a bound: every
#: block computes the same numbers, so this trades the number of dispatches against how much is live at
#: once, and never the answer. Sized to sit far below the assembled case rather than to be tuned.
_SEARCH_WORKING_BYTES = 64 << 20


def _nearest_squared_distance(centroid: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Squared distance from each cell centroid to the nearest target, shape ``(n_cells,)``.

    Searched a block of cells at a time. Each cell's nearest target is independent of every other
    cell's, so blocking reorders nothing and drops nothing — the result is identical entry for entry to
    the whole-array form, which is what lets the block size be a memory decision alone.

    Kept as ``jnp`` throughout rather than handing the search to a spatial index. The index would be
    asymptotically better, but it would need concrete coordinates, and these may be **tracers**:
    geometry is derived from ``node_coords`` on demand precisely so that gradients with respect to node
    positions chain through it, and a build that is differentiated would reach here with traced
    centroids. Memory was the problem; the search cost was not.
    """
    n_cells = centroid.shape[0]
    n_target, dim = target.shape
    # Per cell: the offset, the square the magnitude forms, and the reduced row -- the three that are
    # live together at the peak.
    per_cell = max(1, 3 * n_target * dim * centroid.dtype.itemsize)
    block = max(1, min(n_cells, _SEARCH_WORKING_BYTES // per_cell))
    return jnp.concatenate(
        [
            jnp.min(
                norm_squared(centroid[start : start + block, None, :] - target[None, :, :]), axis=1
            )
            for start in range(0, n_cells, block)
        ]
    )
