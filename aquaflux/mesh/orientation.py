"""Which way each face's ring runs within each of its cells, from how the cell's faces join.

A cell's boundary is a closed surface, and its faces are consistently oriented when every edge two
of them share is traversed once each way. That fixes each face's direction relative to the others up
to one choice for the whole cell -- inward or outward -- and the sign of the cell's volume, taken
with that orientation, makes the choice. Unlike asking which side of a face the cell's centroid lies
on, this holds however far the cell is from convex: a thin cell bent around a curved wall, whose
centroid can lie beyond one of its own faces, is oriented as surely as a cube.

This module finds the relative directions, which depend only on the connectivity; the volume's sign
depends on the coordinates and is taken where the geometry is computed. In two dimensions a face is
an edge, and two faces of a cell join at a node: consistently oriented, one ends where the other
starts.

A cell whose faces do not all join this way -- a face linked to the rest of its cell through no shared
edge, as happens when one face's edge carries a node the face beside it does not, or a cell whose
faces cannot be oriented consistently -- is reported undetermined, and its faces are oriented by its
centroid instead.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numba
import numpy as np

from aquaflux.vectors import dot, scale

__all__ = ["RingSigns", "owner_outward_normals", "ring_signs"]


class RingSigns(eqx.Module):
    """Each face's ring direction within its owner and its neighbour, relative to the cell's others.

    Attributes
    ----------
    owner, neighbour : jnp.ndarray of float, shape ``(n_faces,)``
        ``+1`` or ``-1``: the factor that makes the face's stored ring consistently oriented with the
        other faces of its owner (its neighbour). ``0`` on a boundary face's neighbour side, and on
        both sides of every face of an undetermined cell.
    determined : jnp.ndarray of bool, shape ``(n_cells,)``
        Whether each cell's faces are consistently oriented by the signs.
    """

    owner: jnp.ndarray
    neighbour: jnp.ndarray
    determined: jnp.ndarray


def ring_signs(face_nodes, face_cells) -> RingSigns:
    """The consistent ring directions of every cell's faces.

    Parameters
    ----------
    face_nodes : FaceNodeConnectivity
    face_cells : FaceCellConnectivity

    Returns
    -------
    RingSigns
    """
    offsets = np.asarray(face_nodes.offsets, dtype=np.int64)
    nodes = np.asarray(face_nodes.face_node_indices, dtype=np.int64)
    owner = np.asarray(face_cells.owner, dtype=np.int64)
    neighbour = np.asarray(face_cells.neighbour, dtype=np.int64)
    counts = np.diff(offsets)
    two_dimensional = bool(counts.size) and bool(np.all(counts == 2))
    sign, determined = _signs(
        offsets, nodes, owner, neighbour, int(face_cells.n_cells), two_dimensional
    )
    return RingSigns(
        owner=jnp.asarray(sign[0::2], dtype=jnp.float64),
        neighbour=jnp.asarray(sign[1::2], dtype=jnp.float64),
        determined=jnp.asarray(determined),
    )


@numba.njit(cache=True)
def _signs(offsets, nodes, owner, neighbour, n_cells, two_dimensional):
    """Each incidence's sign (face ``f``'s owner side ``2f``, neighbour side ``2f + 1``) and which
    cells are determined, solved cell by cell.

    A cell's joints are its faces' ring edges (in two dimensions, their end nodes), each with the
    direction the face runs along it; a joint met by exactly two of the cell's faces asks that their
    signs times those directions be opposite. Each cell's joints are sorted on their own, then the
    constraints are solved by a breadth-first search from the cell's first face.
    """
    n_faces = owner.shape[0]
    # The incidences of each cell, in compressed-sparse-row (CSR) form.
    start = np.zeros(n_cells + 1, dtype=np.int64)
    for f in range(n_faces):
        start[owner[f] + 1] += 1
        if neighbour[f] >= 0:
            start[neighbour[f] + 1] += 1
    for c in range(n_cells):
        start[c + 1] += start[c]
    members = np.empty(start[-1], dtype=np.int64)
    place = start[:-1].copy()
    for f in range(n_faces):
        members[place[owner[f]]] = 2 * f
        place[owner[f]] += 1
        if neighbour[f] >= 0:
            members[place[neighbour[f]]] = 2 * f + 1
            place[neighbour[f]] += 1
    sign = np.zeros(2 * n_faces, dtype=np.int64)
    determined = np.zeros(n_cells, dtype=np.bool_)
    for c in range(n_cells):
        first, last = start[c], start[c + 1]
        n_members = last - first
        if n_members == 0:
            continue
        n_joints = 0
        for m in range(first, last):
            f = members[m] // 2
            n_joints += offsets[f + 1] - offsets[f]
        key = np.empty(n_joints, dtype=np.int64)
        direction = np.empty(n_joints, dtype=np.int64)
        local = np.empty(n_joints, dtype=np.int64)  # which member each joint belongs to
        j = 0
        for m in range(n_members):
            f = members[first + m] // 2
            begin, end = offsets[f], offsets[f + 1]
            size = end - begin
            for k in range(size):
                a = nodes[begin + k]
                if two_dimensional:
                    key[j] = a
                    direction[j] = 1 if k == 1 else -1
                else:
                    b = nodes[begin + (k + 1) % size]
                    if a < b:
                        key[j] = a * (2**31) + b
                        direction[j] = 1
                    else:
                        key[j] = b * (2**31) + a
                        direction[j] = -1
                local[j] = m
                j += 1
        order = np.argsort(key)
        # Constraints between members: neighbours in a small adjacency list.
        link_to = np.empty(n_joints, dtype=np.int64)
        link_from = np.empty(n_joints, dtype=np.int64)
        link_parity = np.empty(n_joints, dtype=np.int64)
        n_links = 0
        i = 0
        while i < n_joints:
            k = i + 1
            while k < n_joints and key[order[k]] == key[order[i]]:
                k += 1
            if k - i == 2:
                p, q = order[i], order[i + 1]
                link_from[n_links] = local[p]
                link_to[n_links] = local[q]
                link_parity[n_links] = -direction[p] * direction[q]
                n_links += 1
            i = k
        local_sign = np.zeros(n_members, dtype=np.int64)
        local_sign[0] = 1
        queue = np.empty(n_members, dtype=np.int64)
        queue[0] = 0
        head, tail = 0, 1
        consistent = True
        while head < tail:
            here = queue[head]
            head += 1
            for k in range(n_links):
                if link_from[k] == here:
                    there = link_to[k]
                elif link_to[k] == here:
                    there = link_from[k]
                else:
                    continue
                wanted = local_sign[here] * link_parity[k]
                if local_sign[there] == 0:
                    local_sign[there] = wanted
                    queue[tail] = there
                    tail += 1
                elif local_sign[there] != wanted:
                    consistent = False
        if consistent and tail == n_members:
            determined[c] = True
            for m in range(n_members):
                sign[members[first + m]] = local_sign[m]
    return sign, determined


def owner_outward_normals(
    scheme, node_order_normal, area, centroid, approx_centroid, face_cells, signs: RingSigns
):
    """Each face's unit normal turned to point out of its owner cell.

    Within a determined cell the faces' rings, times their :class:`RingSigns`, are consistently
    oriented; the cell's volume with that orientation is positive when they face outward, so its
    sign turns them all outward together. A face whose owner is undetermined is turned by the
    owner's approximate centroid instead (:meth:`FaceGeometryScheme.orient_owner_outward`).

    Parameters
    ----------
    scheme : FaceGeometryScheme
    node_order_normal : jnp.ndarray, shape ``(n_faces, dim)``
        The unit normals the stored rings imply.
    area : jnp.ndarray, shape ``(n_faces,)``
    centroid : jnp.ndarray, shape ``(n_faces, dim)``
    approx_centroid : jnp.ndarray, shape ``(n_cells, dim)``
        Each cell's approximate centroid, the reference its volume is accumulated about.
    face_cells : FaceCellConnectivity
    signs : RingSigns

    Returns
    -------
    jnp.ndarray, shape ``(n_faces, dim)``
    """
    from .cell import CellGeometry

    owner, neighbour = face_cells.owner, face_cells.safe_neighbour
    owner_sign = signs.owner.astype(node_order_normal.dtype)
    neighbour_sign = signs.neighbour.astype(node_order_normal.dtype)
    flux = scale(node_order_normal, area)
    owner_moment = dot(flux, centroid - approx_centroid[owner]) * owner_sign
    neighbour_centroid = CellGeometry._neighbour_face_centroid(centroid, face_cells)
    neighbour_moment = dot(flux, neighbour_centroid - approx_centroid[neighbour]) * neighbour_sign
    volume = face_cells.scatter(owner_moment, neighbour_moment)
    turn = jnp.where(volume < 0.0, -1.0, 1.0)
    consistent = scale(node_order_normal, owner_sign * turn[owner])
    by_centroid = scheme.orient_owner_outward(node_order_normal, centroid, approx_centroid[owner])
    determined = signs.determined[owner]
    return jnp.where(determined[:, None], consistent, by_centroid)
