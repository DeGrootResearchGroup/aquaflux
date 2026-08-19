"""Distance from cells to named boundary patches — the geometric field wall models need.

A near-wall model needs each cell's distance to the wall. :func:`distance_to_patches` returns, per
cell, the distance from the cell centroid to the nearest face centroid among a named set of boundary
patches (the wall patches, in that use).

This is the nearest-face-*centroid* distance, an approximation to the true distance to the wall
surface: for a wall-adjacent cell it is essentially the centroid's normal distance to its own wall
face — what near-wall models depend on — and it loosens on coarser cells farther from the wall. It
is a function of the static mesh geometry, so it is computed once at build time and reused as a
frozen field. (The patch-to-face lookup is data-dependent, so it runs eagerly, not under ``jit``.)

The nearest face is found two different ways, chosen by whether the centroids are **concrete or
traced**:

- **Concrete** (the ordinary case: an eager build from a static mesh) — a ``scipy.spatial.cKDTree``
  built on the target face centroids, queried once for every cell centroid. This is the common path
  and the one case assembly actually pays for, so it is the one worth making fast.
- **Traced** — geometry is derived from ``node_coords`` on demand precisely so that a gradient with
  respect to node positions chains through it, and a build that is differentiated reaches here with
  traced centroids. A spatial index needs concrete coordinates, so a differentiated call instead runs
  a direct, blocked ``jnp`` search: dense enough over cells and targets that the whole-array form does
  not survive contact with a three-dimensional case (on a 23040-cell mesh with 4736 wall faces it
  needs 109 million entries, and because the squared magnitude materializes its own product, a little
  over 6 GB would be live at once), so it is searched a block of cells at a time instead, each cell's
  nearest target independent of every other cell's. The blocks run through a single compiled loop
  (``jax.lax.map``) rather than a Python loop tracing and dispatching one XLA program per block and
  feeding a `jnp.concatenate` over as many operands as there are blocks — a block count that grows
  with the product of cell count and target count as the memory bound shrinks the block, and that
  operand count is what turns compiling this path into a multi-minute affair when it is embedded in a
  larger jitted computation. This path is reached rarely (a differentiated build), so it is written
  for boundedness under tracing rather than for raw throughput.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

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


def _is_traced(*arrays: jnp.ndarray) -> bool:
    """``True`` if any array is a :class:`jax.core.Tracer` (under ``jit`` or ``grad``, not eager)."""
    return any(isinstance(a, jax.core.Tracer) for a in arrays)


def _nearest_squared_distance(centroid: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Squared distance from each cell centroid to the nearest target, shape ``(n_cells,)``.

    Dispatches on whether the inputs are concrete (build a spatial index) or traced (search the
    ``jnp`` array directly) -- see the module docstring for why the two need different algorithms.
    """
    if _is_traced(centroid, target):
        return _nearest_squared_distance_traced(centroid, target)
    return _nearest_squared_distance_concrete(centroid, target)


def _nearest_squared_distance_concrete(centroid: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Squared nearest-target distance via a ``scipy.spatial.cKDTree`` over the target centroids.

    Asymptotically far better than the brute-force search below (a tree query is
    ``O(log n_target)`` per cell against the brute force's ``O(n_target)``), and run in parallel
    across cells (``workers=-1``). Exact up to floating-point rounding, not approximate -- a k-d tree
    nearest-neighbour query returns the true nearest point, just found by a smarter search.
    """
    tree = cKDTree(np.asarray(target))
    distance, _ = tree.query(np.asarray(centroid), workers=-1)
    return jnp.asarray(distance, dtype=centroid.dtype) ** 2


#: Working set the traced fallback's blocked search is allowed per block, in bytes. It buys nothing
#: but a bound: every block computes the same numbers, so this trades the number of loop steps
#: against how much is live at once, and never the answer. Sized to sit far below the assembled case
#: rather than to be tuned.
_SEARCH_WORKING_BYTES = 64 << 20


def _nearest_squared_distance_traced(centroid: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Squared nearest-target distance via a direct, blocked ``jnp`` search (the traced fallback).

    Searched a block of cells at a time, the blocks run through ``jax.lax.map`` rather than a Python
    loop. Each cell's nearest target is independent of every other cell's, so blocking reorders nothing
    and drops nothing — the result is identical entry for entry to the whole-array form, which is what
    lets the block size be a memory decision alone. ``jax.lax.map`` compiles the per-block computation
    once and loops it on-device (it is built on ``scan``), so the block count — which grows with the
    product of cell count and target count as the memory bound shrinks the block — drives loop
    iterations rather than separately-traced-and-dispatched XLA programs feeding one large
    ``jnp.concatenate``.
    """
    n_cells = centroid.shape[0]
    n_target, dim = target.shape
    # Per cell: the offset, the square the magnitude forms, and the reduced row -- the three that are
    # live together at the peak.
    per_cell = max(1, 3 * n_target * dim * centroid.dtype.itemsize)
    block = max(1, min(n_cells, _SEARCH_WORKING_BYTES // per_cell))
    n_blocks = -(-n_cells // block)  # ceiling division

    def block_nearest(block_centroid: jnp.ndarray) -> jnp.ndarray:
        return jnp.min(norm_squared(block_centroid[:, None, :] - target[None, :, :]), axis=1)

    # Pad to a whole number of blocks by repeating the last cell -- its recomputed distance is
    # discarded by the final slice, so the padding value only has to be finite, and repeating a real
    # cell keeps it so under a differentiated (traced) centroid without introducing a sentinel.
    padded = jnp.pad(centroid, ((0, n_blocks * block - n_cells), (0, 0)), mode="edge")
    return jax.lax.map(block_nearest, padded.reshape(n_blocks, block, -1)).reshape(-1)[:n_cells]
