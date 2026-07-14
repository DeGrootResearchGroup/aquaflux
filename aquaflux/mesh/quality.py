"""Mesh-quality diagnostics — the CFD sense of "mesh metrics" (planarity here; skewness /
non-orthogonality later).

These are run **once** at build time to check a mesh, not inside the solve. They exist to
answer questions like "are my faces warped enough that the centre-fan geometry needs
iterating?" — cheaply and quantitatively, so robustness is a measured property rather than
an assumption.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from .face import PolygonFaceGeometry

if TYPE_CHECKING:
    from .mesh import Mesh


def closed_cell_residual(mesh: Mesh) -> jnp.ndarray:
    """Per-cell ``||Σ outward face area-vectors||`` — the standard FVM mesh-validity check.

    For a properly closed cell the outward face area-vectors sum to zero (a discrete
    divergence theorem on the constant field). A value well above rounding error means the
    cell is not closed — a missing or mis-connected face — or, as a useful side effect, that
    the owner-outward orientation is wrong. This is the *geometry*-level complement to
    :meth:`aquaflux.mesh.Mesh.validate` (which is topology-only): a mesh can pass every
    topological check yet still have unclosed cells if faces are absent. Run once at build.

    Returns
    -------
    jnp.ndarray
        Residual magnitude per cell, shape ``(n_cells,)``. ``~0`` for a valid mesh; compare
        against a small multiple of a representative ``face_area`` to flag bad cells.
    """
    face_geometry = mesh.geometry().face
    area_vector = face_geometry.area[:, None] * face_geometry.normal  # owner-outward S_f
    # a closed cell has Σ outward face area-vectors = 0: the conservative scatter of S_f.
    per_cell = mesh.face_cells.scatter_conservative(area_vector)
    return jnp.linalg.norm(per_cell, axis=1)


def face_planarity(mesh: Mesh) -> jnp.ndarray:
    """Per-face planarity ratio ``|S| / sum_i|triangle_i|`` in ``(0, 1]``.

    ``1.0`` means perfectly planar; smaller means more warped (``1 - ratio`` is the warp
    fraction). This is a near-free byproduct of the centre-fan geometry — a cheap first
    screen for whether any faces are non-planar at all. Trivially ``1`` in 2D (edges), and
    ``0`` for a degenerate (zero-area) face — the clearest possible "bad face" signal.

    Returns
    -------
    jnp.ndarray
        Planarity per face, shape ``(n_faces,)``.
    """
    if mesh.dim == 2:
        return jnp.ones(mesh.n_faces)
    # Planarity is inherently 3D (a byproduct of the polygon centre fan), so the concrete 3D
    # scheme is used directly — the 2D case returned above.
    area, _, _, total_tri_area = PolygonFaceGeometry().centre_fan(mesh.node_coords, mesh.face_nodes)
    return jnp.where(
        total_tri_area > 0.0, area / jnp.where(total_tri_area > 0.0, total_tri_area, 1.0), 0.0
    )


def centroid_iteration_shift(mesh: Mesh) -> jnp.ndarray:
    """Per-face ``|c2 - c1| / sqrt(area)``: how far a second centre-fan pass moves the centroid.

    ``c1`` is the one-pass centre-fan centroid (apex = vertex mean); ``c2`` repeats the fan
    with ``c1`` as the apex. The normalized shift is the **direct** measure of whether centre
    iteration would refine the face centroid: ``~0`` means one pass is already exact (planar
    or mild warp), an appreciable value means the faces are warped enough that iterating
    would help. For realistic meshes this is machine-negligible, so one pass is used; run
    this on a representative mesh to confirm before deciding to add iteration. Zeros in 2D,
    and ``inf`` for a degenerate (zero-area) face — a clear "bad face" signal, not a silent NaN.

    Returns
    -------
    jnp.ndarray
        Normalized centroid shift per face, shape ``(n_faces,)``.
    """
    if mesh.dim == 2:
        return jnp.zeros(mesh.n_faces)
    # Inherently 3D (needs the polygon centre fan); the 2D case returned above.
    scheme = PolygonFaceGeometry()
    face_nodes = mesh.face_nodes
    area, c1, _, _ = scheme.centre_fan(mesh.node_coords, face_nodes)
    _, c2, _, _ = scheme.centre_fan(mesh.node_coords, face_nodes, apex=c1)
    shift = jnp.linalg.norm(c2 - c1, axis=1)
    return jnp.where(area > 0.0, shift / jnp.sqrt(jnp.where(area > 0.0, area, 1.0)), jnp.inf)
