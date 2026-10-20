"""Mesh-quality diagnostics — the CFD sense of "mesh metrics": closure, face planarity and
warp, non-orthogonality, skewness, cell convexity, and how much a cell's volume depends on how its
faces are triangulated.

These are run at build time to check a mesh, not inside the solve. They exist to
answer questions like "are my faces warped enough that the centre-fan geometry needs
iterating?" — cheaply and quantitatively, so robustness is a measured property rather than
an assumption. A mesh smoother checks every trial position of its vertices; :class:`MeshQuality`
finds what the checks need of the connectivity once, so each check costs only its arithmetic.

Non-orthogonality, skewness and the face-centroid concavity test follow the definitions of
OpenFOAM's ``checkMesh``, so a mesh's numbers can be compared with what that tool reports for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numba
import numpy as np

from aquaflux.vectors import dot, norm, scale

from .face import PolygonFaceGeometry
from .geometry import MeshGeometry
from .orientation import ring_signs

if TYPE_CHECKING:
    from .mesh import Mesh

#: ``checkMesh`` counts a cell concave when its :func:`cell_concavity` cosine exceeds this.
CONCAVE_COSINE = -1e-6


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
    area_vector = scale(face_geometry.normal, face_geometry.area)  # owner-outward S_f
    # a closed cell has Σ outward face area-vectors = 0: the conservative scatter of S_f.
    per_cell = mesh.face_cells.scatter_conservative(area_vector)
    return jnp.linalg.norm(per_cell, axis=1)


def face_planarity(mesh: Mesh) -> jnp.ndarray:
    """Per-face planarity ratio ``|S| / sum_i|triangle_i|`` in ``[0, 1]``.

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


def face_warp(mesh: Mesh) -> jnp.ndarray:
    """Per-face warp: how far the face's vertices lie off its plane, over the face's size.

    The plane passes through the face's vertex mean ``c``, normal to its area vector; the warp is
    ``max_k |n . (v_k - c)| / sqrt(area)`` over the face's vertices ``v_k``. Unlike
    :func:`face_planarity`, which departs from one only at second order in the out-of-plane
    distance, this is first order in it, so it resolves a mild warp. ``0`` for a planar face and
    in 2D, and ``inf`` for a degenerate (zero-area) face.

    Returns
    -------
    jnp.ndarray
        Normalized warp per face, shape ``(n_faces,)``.
    """
    if mesh.dim == 2:
        return jnp.zeros(mesh.n_faces)
    face_geometry = mesh.geometry().face
    lift = _vertex_lift(mesh.face_nodes, mesh.node_coords, face_geometry.normal)
    area = face_geometry.area
    largest = mesh.face_nodes.max_to_faces(jnp.abs(lift))
    return jnp.where(area > 0.0, largest / jnp.sqrt(jnp.where(area > 0.0, area, 1.0)), jnp.inf)


def triangulation_volume_change(mesh: Mesh) -> jnp.ndarray:
    """Per cell: the largest relative change of volume that re-triangulating its faces can make.

    A non-planar face bounds no definite volume: the polyhedron's volume depends on how each face
    is cut into triangles. Fanning a face from one of its own vertices ``v_k`` instead of from its
    vertex mean ``c`` (the centre fan the mesh uses) encloses the extra volume
    ``S . (v_k - c) / 3`` beyond the face, ``S`` its outward area vector — the two fans share the
    face's perimeter, and every centre-fan triangle has zero volume seen from ``c``. Choosing, face
    by face, the vertex that adds the most (or removes the most) gives the largest change any such
    choice can make; this returns the larger of the two, over the cell's volume. ``0`` when every
    face is planar, and in 2D.

    Returns
    -------
    jnp.ndarray
        Relative volume change per cell, shape ``(n_cells,)``.
    """
    if mesh.dim == 2:
        return jnp.zeros(mesh.n_cells)
    geometry = mesh.geometry()
    face_geometry = geometry.face
    lift = _vertex_lift(
        mesh.face_nodes, mesh.node_coords, face_geometry.normal
    )  # along the owner-outward normal
    third_area = face_geometry.area / 3.0
    rise = third_area * mesh.face_nodes.max_to_faces(lift)  # the most a fan can add, owner side
    fall = third_area * mesh.face_nodes.max_to_faces(-lift)  # the most it can remove
    face_cells = mesh.face_cells
    # Seen from the neighbour the outward normal is reversed, so what adds to the owner removes.
    gain = face_cells.scatter(rise, fall)
    loss = face_cells.scatter(fall, rise)
    return jnp.maximum(gain, loss) / geometry.cell.volume


@eqx.filter_jit
def _vertex_lift(face_nodes, node_coords, normal: jnp.ndarray) -> jnp.ndarray:
    """Each face incidence's height ``n . (v - c)`` above its face's vertex-mean plane."""
    mean = face_nodes.vertex_mean(node_coords)
    of_face = face_nodes.face_of_incidence
    return dot(face_nodes.gather_node_coords(node_coords) - mean[of_face], normal[of_face])


def _centre_line(geometry: MeshGeometry, face_cells) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per face, the owner centroid to the face centroid, and the owner to the neighbour centroid.

    On a boundary face the second is the first projected onto the face normal: the line from the
    owner's centroid perpendicular to the face, which is what both checks measure against there.
    """
    cell_centroid = geometry.cell.centroid
    owner_centroid = cell_centroid[face_cells.owner]
    to_face = geometry.face.centroid - owner_centroid
    normal = geometry.face.normal
    across = face_cells.neighbour_centroid(cell_centroid) - owner_centroid
    perpendicular = scale(normal, dot(normal, to_face))
    interior = face_cells.interior[:, None]
    return to_face, jnp.where(interior, across, perpendicular)


def face_non_orthogonality(mesh: Mesh) -> jnp.ndarray:
    """Per-face non-orthogonality: the angle, in degrees, between the centre line and the normal.

    On an interior face the centre line runs from the owner's centroid to the neighbour's (its
    periodic image across a seam); on a boundary face, from the owner's centroid to the face's
    centroid. ``0`` on an orthogonal mesh. ``checkMesh`` reports the maximum and the average over
    the **interior** faces (``mesh.face_cells.interior``).

    Returns
    -------
    jnp.ndarray
        Angle per face in degrees, shape ``(n_faces,)``.
    """
    return _non_orthogonality(mesh.geometry(), mesh.face_cells)


@eqx.filter_jit
def _non_orthogonality(geometry: MeshGeometry, face_cells) -> jnp.ndarray:
    """:func:`face_non_orthogonality` from a geometry already computed."""
    to_face, across = _centre_line(geometry, face_cells)
    line = jnp.where(face_cells.interior[:, None], across, to_face)
    normal = geometry.face.normal
    cosine = dot(line, normal) / jnp.maximum(norm(line), jnp.finfo(line.dtype).tiny)
    return jnp.degrees(jnp.arccos(jnp.clip(cosine, -1.0, 1.0)))


def face_skewness(mesh: Mesh) -> jnp.ndarray:
    """Per-face skewness, as ``checkMesh`` defines it.

    The skewness vector runs from where the centre line ``d`` crosses the face's plane to the
    face's centroid, ``s = (c_f - c_P) - (S . (c_f - c_P)) / (S . d) d``. Its length is divided by
    the face's extent in that direction — the largest ``|s_hat . (v_k - c_f)|`` over the face's
    vertices — floored at ``0.2 |d|`` on an interior face. On a boundary face ``d`` is the
    owner centroid's perpendicular to the face, so ``s`` is the tangential offset of the face
    centroid from the owner's, and the floor is ``0.4 |d|`` (``d`` spans half the distance an
    interior face's line does). ``0`` when the centre line passes through the face centroid.

    Returns
    -------
    jnp.ndarray
        Skewness per face, shape ``(n_faces,)``.
    """
    return _skewness(mesh.node_coords, mesh.geometry(), mesh.face_cells, mesh.face_nodes)


@eqx.filter_jit
def _skewness(node_coords, geometry: MeshGeometry, face_cells, face_nodes) -> jnp.ndarray:
    """:func:`face_skewness` from a geometry already computed at ``node_coords``."""
    to_face, line = _centre_line(geometry, face_cells)
    face_geometry = geometry.face
    normal = face_geometry.normal
    tiny = jnp.finfo(line.dtype).tiny
    along = dot(normal, to_face) / jnp.where(dot(normal, line) == 0.0, tiny, dot(normal, line))
    skew = to_face - scale(line, along)
    length = norm(skew)
    direction = scale(skew, 1.0 / jnp.maximum(length, tiny))
    of_face = face_nodes.face_of_incidence
    offset = face_nodes.gather_node_coords(node_coords) - face_geometry.centroid[of_face]
    extent = face_nodes.max_to_faces(jnp.abs(dot(offset, direction[of_face])))
    floor = jnp.where(face_cells.interior, 0.2, 0.4) * norm(line)
    return length / jnp.maximum(jnp.maximum(extent, floor), tiny)


def cell_non_convexity(mesh: Mesh) -> np.ndarray:
    """Per cell: how far its farthest vertex lies outside one of its faces' planes, over its size.

    Each face's plane passes through its centroid, normal to its area vector (both as the mesh
    computes them, the normal out of the cell). The value is the largest signed distance of any of
    the cell's vertices beyond any of its faces' planes, divided by the cell's size (the
    ``dim``-th root of its volume). ``0`` for a convex cell with planar faces; a warped face makes
    its own vertices stand off its plane, so a small positive value is face warp, and a large one
    is a fold. Never negative: a face's own vertices are never all behind its plane.

    Returns
    -------
    np.ndarray
        Non-convexity per cell, shape ``(n_cells,)``.
    """
    quality = MeshQuality(mesh)
    return quality.cell_non_convexity(mesh.node_coords, quality.geometry(mesh.node_coords))


def cell_concavity(mesh: Mesh) -> np.ndarray:
    """Per cell: ``checkMesh``'s concave-cell test by face planes, as the cosine it thresholds.

    For each pair of a cell's faces ``i`` and ``j``, the cosine between face ``i``'s outward
    normal and the direction from its centroid to face ``j``'s centroid; this returns the largest.
    In a convex cell every other face's centroid lies behind each face's plane, so the value is
    negative; ``checkMesh`` counts a cell concave when it exceeds :data:`CONCAVE_COSINE`. Face warp alone does
    not raise it, since only face centroids are tested.

    Returns
    -------
    np.ndarray
        Largest cosine per cell, shape ``(n_cells,)``.
    """
    quality = MeshQuality(mesh)
    return quality.cell_concavity(quality.geometry(mesh.node_coords))


class MeshQuality:
    """The quality checks of one mesh's connectivity, at whatever node coordinates are given.

    What the checks need of the connectivity -- each cell's faces with the side the cell sees each
    from, each cell's distinct vertices, and the faces' ring directions the geometry is oriented
    by -- depends on the topology alone. It is found once, here, so a caller that moves the
    vertices many times and checks the mesh after each move (a smoother) pays only for the
    arithmetic. The module's functions of a :class:`~aquaflux.mesh.Mesh` evaluate through one.

    Parameters
    ----------
    mesh : Mesh
        The topology; its node coordinates are not used.

    Examples
    --------
    >>> from aquaflux.mesh import MeshQuality, structured_grid_3d
    >>> mesh = structured_grid_3d(2, 1, 1)
    >>> quality = MeshQuality(mesh)
    >>> geometry = quality.geometry(mesh.node_coords)
    >>> float(quality.face_non_orthogonality(geometry).max())
    0.0
    """

    def __init__(self, mesh: Mesh) -> None:
        self._mesh = mesh
        self._signs = ring_signs(mesh.face_nodes, mesh.face_cells)
        owner = np.asarray(mesh.face_cells.owner, dtype=np.int64)
        neighbour = np.asarray(mesh.face_cells.neighbour, dtype=np.int64)
        interior = np.flatnonzero(neighbour >= 0)
        cell = np.concatenate([owner, neighbour[interior]])
        order = np.argsort(cell, kind="stable")
        # Each cell's sides -- (face, outward sign) -- in compressed-sparse-row (CSR) form.
        self._side_offsets = np.zeros(mesh.n_cells + 1, dtype=np.int64)
        np.cumsum(np.bincount(cell, minlength=mesh.n_cells), out=self._side_offsets[1:])
        self._side_face = np.concatenate([np.arange(owner.size), interior])[order]
        self._side_sign = np.concatenate([np.ones(owner.size), -np.ones(interior.size)])[order]
        self._node_offsets, self._nodes = mesh.cell_nodes()

    def geometry(self, node_coords: jnp.ndarray) -> MeshGeometry:
        """The mesh's geometry with its vertices at ``node_coords``, shape ``(n_nodes, dim)``."""
        moved = eqx.tree_at(lambda m: m.node_coords, self._mesh, jnp.asarray(node_coords))
        return moved.geometry(self._signs)

    def vertex_heights(self, node_coords: jnp.ndarray, geometry: MeshGeometry) -> jnp.ndarray:
        """Each face incidence's height above its face's vertex-mean plane, along the face's normal.

        The heights :func:`face_warp` takes the largest of, at ``node_coords``, ``geometry`` being
        the geometry there; shape ``(n_incidences,)``, in ``face_nodes.face_node_indices`` order.
        """
        return _vertex_lift(self._mesh.face_nodes, jnp.asarray(node_coords), geometry.face.normal)

    def cell_non_convexity(self, node_coords: jnp.ndarray, geometry: MeshGeometry) -> np.ndarray:
        """:func:`cell_non_convexity` at ``node_coords``, ``geometry`` being the geometry there."""
        worst = _largest_excursion(
            self._side_offsets,
            self._side_face,
            self._side_sign,
            self._node_offsets,
            self._nodes,
            np.asarray(node_coords, dtype=np.float64),
            np.asarray(geometry.face.centroid),
            np.asarray(geometry.face.normal),
        )
        return worst / np.asarray(geometry.cell.volume) ** (1.0 / self._mesh.dim)

    def cell_concavity(self, geometry: MeshGeometry) -> np.ndarray:
        """:func:`cell_concavity`, ``geometry`` being the mesh's geometry."""
        return _largest_cosine(
            self._side_offsets,
            self._side_face,
            self._side_sign,
            np.asarray(geometry.face.centroid),
            np.asarray(geometry.face.normal),
        )

    def face_non_orthogonality(self, geometry: MeshGeometry) -> jnp.ndarray:
        """:func:`face_non_orthogonality`, ``geometry`` being the mesh's geometry."""
        return _non_orthogonality(geometry, self._mesh.face_cells)

    def face_skewness(self, node_coords: jnp.ndarray, geometry: MeshGeometry) -> jnp.ndarray:
        """:func:`face_skewness` at ``node_coords``, ``geometry`` being the geometry there."""
        mesh = self._mesh
        return _skewness(jnp.asarray(node_coords), geometry, mesh.face_cells, mesh.face_nodes)


@numba.njit(parallel=True, cache=True)
def _largest_excursion(
    side_offsets, side_face, side_sign, node_offsets, nodes, coords, centre, normal
):
    """Per cell, the largest signed height of one of its vertices above one of its faces' planes.

    A NaN height makes its cell's value NaN, as a maximum taken by numpy would.
    """
    n_cells, dim = side_offsets.size - 1, coords.shape[1]
    worst = np.full(n_cells, -np.inf)
    for c in numba.prange(n_cells):
        for s in range(side_offsets[c], side_offsets[c + 1]):
            f = side_face[s]
            for k in range(node_offsets[c], node_offsets[c + 1]):
                rise = 0.0
                for d in range(dim):
                    rise += (coords[nodes[k], d] - centre[f, d]) * normal[f, d]
                value = side_sign[s] * rise
                if value > worst[c] or value != value:
                    worst[c] = value
    return worst


@numba.njit(parallel=True, cache=True)
def _largest_cosine(side_offsets, side_face, side_sign, centre, normal):
    """Per cell, the largest cosine between one face's outward normal and the direction from its
    centroid to another face's centroid; ``-1`` for a face paired with itself.

    A NaN cosine makes its cell's value NaN, as a maximum taken by numpy would.
    """
    n_cells, dim = side_offsets.size - 1, centre.shape[1]
    worst = np.full(n_cells, -np.inf)
    for c in numba.prange(n_cells):
        for s in range(side_offsets[c], side_offsets[c + 1]):
            face = side_face[s]
            for t in range(side_offsets[c], side_offsets[c + 1]):
                towards = side_face[t]
                if towards == face:
                    value = -1.0
                else:
                    along, length = 0.0, 0.0
                    for d in range(dim):
                        offset = centre[towards, d] - centre[face, d]
                        along += offset * normal[face, d]
                        length += offset * offset
                    length = np.sqrt(length)
                    value = side_sign[s] * along / (length if length > 0.0 else 1.0)
                if value > worst[c] or value != value:
                    worst[c] = value
    return worst
