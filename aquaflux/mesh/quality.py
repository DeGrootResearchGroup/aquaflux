"""Mesh-quality diagnostics — the CFD sense of "mesh metrics": closure, face planarity and
warp, non-orthogonality, skewness, cell convexity, and how much a cell's volume depends on how its
faces are triangulated.

These are run **once** at build time to check a mesh, not inside the solve. They exist to
answer questions like "are my faces warped enough that the centre-fan geometry needs
iterating?" — cheaply and quantitatively, so robustness is a measured property rather than
an assumption.

Non-orthogonality, skewness and the face-centroid concavity test follow the definitions of
OpenFOAM's ``checkMesh``, so a mesh's numbers can be compared with what that tool reports for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from aquaflux import ragged
from aquaflux.vectors import dot, norm, scale

from .face import PolygonFaceGeometry

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
    lift = _vertex_lift(mesh, face_geometry.normal)
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
    lift = _vertex_lift(mesh, face_geometry.normal)  # along the owner-outward normal
    third_area = face_geometry.area / 3.0
    rise = third_area * mesh.face_nodes.max_to_faces(lift)  # the most a fan can add, owner side
    fall = third_area * mesh.face_nodes.max_to_faces(-lift)  # the most it can remove
    face_cells = mesh.face_cells
    # Seen from the neighbour the outward normal is reversed, so what adds to the owner removes.
    gain = face_cells.scatter(rise, fall)
    loss = face_cells.scatter(fall, rise)
    return jnp.maximum(gain, loss) / geometry.cell.volume


def _vertex_lift(mesh: Mesh, normal: jnp.ndarray) -> jnp.ndarray:
    """Each face incidence's height ``n . (v - c)`` above its face's vertex-mean plane."""
    face_nodes = mesh.face_nodes
    mean = face_nodes.vertex_mean(mesh.node_coords)
    of_face = face_nodes.face_of_incidence
    return dot(face_nodes.gather_node_coords(mesh.node_coords) - mean[of_face], normal[of_face])


def _centre_line(mesh: Mesh) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per face, the owner centroid to the face centroid, and the owner to the neighbour centroid.

    On a boundary face the second is the first projected onto the face normal: the line from the
    owner's centroid perpendicular to the face, which is what both checks measure against there.
    """
    geometry = mesh.geometry()
    face_cells = mesh.face_cells
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
    to_face, across = _centre_line(mesh)
    interior = mesh.face_cells.interior[:, None]
    line = jnp.where(interior, across, to_face)
    normal = mesh.geometry().face.normal
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
    to_face, line = _centre_line(mesh)
    face_geometry = mesh.geometry().face
    normal = face_geometry.normal
    tiny = jnp.finfo(line.dtype).tiny
    along = dot(normal, to_face) / jnp.where(dot(normal, line) == 0.0, tiny, dot(normal, line))
    skew = to_face - scale(line, along)
    length = norm(skew)
    direction = scale(skew, 1.0 / jnp.maximum(length, tiny))
    face_nodes = mesh.face_nodes
    of_face = face_nodes.face_of_incidence
    offset = face_nodes.gather_node_coords(mesh.node_coords) - face_geometry.centroid[of_face]
    extent = face_nodes.max_to_faces(jnp.abs(dot(offset, direction[of_face])))
    floor = jnp.where(mesh.face_cells.interior, 0.2, 0.4) * norm(line)
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
    geometry = mesh.geometry()
    coords = np.asarray(mesh.node_coords)
    normal = np.asarray(geometry.face.normal)
    centroid = np.asarray(geometry.face.centroid)
    sides = _CellSides.of(mesh)
    offsets, vertex = mesh.cell_nodes()
    vertex_cell = np.repeat(np.arange(mesh.n_cells), np.diff(offsets))

    def excursion(side, other):
        face = sides.face[side]
        rise = np.einsum("ij,ij->i", coords[vertex[other]] - centroid[face], normal[face])
        return sides.sign[side] * rise

    worst = sides.largest(vertex_cell, excursion, mesh.n_cells)
    return worst / np.asarray(geometry.cell.volume) ** (1.0 / mesh.dim)


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
    geometry = mesh.geometry()
    normal = np.asarray(geometry.face.normal)
    centroid = np.asarray(geometry.face.centroid)
    sides = _CellSides.of(mesh)

    def cosine(side, other):
        face, towards = sides.face[side], sides.face[other]
        offset = centroid[towards] - centroid[face]
        length = np.linalg.norm(offset, axis=1)
        value = sides.sign[side] * np.einsum("ij,ij->i", offset, normal[face])
        return np.where(towards == face, -1.0, value / np.where(length > 0.0, length, 1.0))

    return sides.largest(sides.cell, cosine, mesh.n_cells)


class _CellSides:
    """Each cell's faces, as (cell, face, outward sign) triples, and maxima over pairs within cells.

    A cell-level check pairs every face of a cell with every vertex (or other face) of the same
    cell. The pairs are formed a block of cells at a time, so the working set stays bounded on a
    large mesh.
    """

    _PAIRS_PER_BLOCK = 4_000_000

    def __init__(self, cell: np.ndarray, face: np.ndarray, sign: np.ndarray) -> None:
        self.cell, self.face, self.sign = cell, face, sign

    @classmethod
    def of(cls, mesh: Mesh) -> _CellSides:
        owner = np.asarray(mesh.face_cells.owner, dtype=np.int64)
        neighbour = np.asarray(mesh.face_cells.neighbour, dtype=np.int64)
        interior = np.flatnonzero(neighbour >= 0)
        cell = np.concatenate([owner, neighbour[interior]])
        face = np.concatenate([np.arange(owner.size), interior])
        sign = np.concatenate([np.ones(owner.size), -np.ones(interior.size)])
        return cls(cell, face, sign)

    def largest(self, other_cell: np.ndarray, value, n_cells: int) -> np.ndarray:
        """Per cell, the largest ``value(side, other)`` over its sides and the items of ``other_cell``.

        ``other_cell`` labels each of a second list of items with its cell; ``value`` maps arrays of
        side indices and item indices, paired within cells, to one value per pair.
        """
        worst = np.full(n_cells, -np.inf)
        sides_per_cell = np.bincount(self.cell, minlength=n_cells)
        pairs = np.cumsum(sides_per_cell * np.bincount(other_cell, minlength=n_cells))
        first = 0
        while first < n_cells:
            done = pairs[first - 1] if first else 0
            stop = int(np.searchsorted(pairs, done + self._PAIRS_PER_BLOCK, side="right"))
            stop = max(stop, first + 1)
            side, other = ragged.pairs_within_groups(self.cell, other_cell, n_cells, first, stop)
            if side.size:
                # The pairs come in runs of one side each: reduce each run, then scatter the runs.
                runs = np.flatnonzero(np.diff(side, prepend=-1))
                largest = np.maximum.reduceat(value(side, other), runs)
                np.maximum.at(worst, self.cell[side[runs]], largest)
            first = stop
        return worst
