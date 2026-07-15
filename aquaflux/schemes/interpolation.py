"""The linear owner/neighbour blend and its projection factor, in one place.

The elementary blend ``(1 - g)·f_owner + g·f_nb`` and its projection factor
``g = ((x_ip - x_P)·d) / (d·d)`` (of the face centroid onto the owner→neighbour line) recur across
every gradient scheme and the Rhie–Chow / SIMPLE-Schur flow terms. The blend gives the field value
at ``x_g = x_P + g·d`` — the *foot* of the face centroid's projection onto the owner→neighbour line
— which equals the true face-centroid value **only on an orthogonal grid**; on a skewed grid the
two differ by ``∇f·(x_ip - x_g)``, the skewness correction that each consuming scheme adds itself
(or deliberately omits). The name is therefore ``interpolate_owner_neighbour``, not "to the face":
it interpolates *between the two cells*, and hitting the face centroid is the orthogonal-grid
special case. Both the blend and the factor belong to the schemes layer, so they are defined
**once** here and imported everywhere rather than re-derived per operator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquaflux.vectors import dot, norm_squared

if TYPE_CHECKING:
    from aquaflux.mesh import FaceCellConnectivity, MeshGeometry


def interpolation_factor(
    face_cells: FaceCellConnectivity,
    geometry: MeshGeometry,
) -> jnp.ndarray:
    """Projection factor ``g`` of each face centroid onto its owner→neighbour line.

    ``g = ((x_ip - x_P)·d) / (d·d)`` with ``d = x_N - x_P`` and ``x_ip`` the face centroid — the
    weight such that ``x_ip`` projects to ``x_P + g·d`` on the P→N line. Zero on boundary faces
    (where the neighbour is the owner and ``d = 0``), so callers blend interior faces and supply
    boundary values separately.

    Parameters
    ----------
    face_cells : FaceCellConnectivity
        The face→cell incidence (``mesh.face_cells``).
    geometry : MeshGeometry
        The mesh metrics; reads the face and cell centroids.

    Returns
    -------
    jnp.ndarray
        ``g`` per face, shape ``(n_faces,)`` (``0`` on boundary faces).
    """
    cell_centroid = geometry.cell.centroid
    interior = face_cells.interior
    x_p = cell_centroid[face_cells.owner]
    d = cell_centroid[face_cells.safe_neighbour] - x_p
    dd = jnp.where(interior, norm_squared(d), 1.0)  # avoid 0/0 on boundary faces
    return jnp.where(interior, dot(geometry.face.centroid - x_p, d) / dd, 0.0)


def interpolate_owner_neighbour(
    cell_field: jnp.ndarray,
    factor: jnp.ndarray,
    face_cells: FaceCellConnectivity,
) -> jnp.ndarray:
    """Linear owner/neighbour blend ``(1 - g)·f_owner + g·f_nb`` per face.

    Gives the field value at ``x_g = x_P + g·d`` — the projection of the face centroid onto the
    owner→neighbour line — which is the true face-centroid value only on an orthogonal grid. The
    skewness correction ``∇f·(x_ip - x_g)`` that closes the gap on a skewed grid is layered on by
    the scheme that wants it, not here. On a boundary face ``g = 0`` gives ``f_owner`` (callers
    override interior/boundary as needed).

    Parameters
    ----------
    cell_field : jnp.ndarray
        Per-cell values, shape ``(n_cells, ...)`` — scalar, vector, or tensor per cell.
    factor : jnp.ndarray
        The projection factor ``g`` per face, shape ``(n_faces,)`` (from :func:`interpolation_factor`).
    face_cells : FaceCellConnectivity
        The face→cell incidence (``mesh.face_cells``).

    Returns
    -------
    jnp.ndarray
        The blended value per face, shape ``(n_faces, ...)`` matching ``cell_field``'s trailing shape.
    """
    g = factor.reshape(factor.shape + (1,) * (cell_field.ndim - 1))  # broadcast over trailing dims
    return (1.0 - g) * cell_field[face_cells.owner] + g * cell_field[face_cells.safe_neighbour]
