"""Facets grouped into small spatial clusters, so a pass over pairs can reject many at once.

A cull that tests every (source, blocker) pair of a surface costs the product of their counts,
however cheap each test is. Grouping the facets into clusters of a few dozen neighbours and
bounding each cluster lets one test answer for every pair between two clusters: most cluster
pairs are rejected whole, and only the members of the rest are tested one by one.

The grouping is by the **Morton order** of the facet centroids -- the order a Z-shaped
space-filling curve visits them in -- cut into runs of equal length. Consecutive facets along
that curve are close in space, so each run is a compact patch, and building it is one sort. The
clusters are the same for every receiver: what depends on the receiver (the directions a cluster
occupies) is bounded afresh from its members, per receiver, by whoever uses them.
"""

from __future__ import annotations

import equinox as eqx
import numpy as np

__all__ = ["FacetClusters"]

#: Bits of each coordinate interleaved into a Morton key: 3 x 21 = 63, the most a signed 64-bit
#: key holds. Finer than any mesh this package builds, so ties are between coincident centroids.
_MORTON_BITS = 21


class FacetClusters(eqx.Module):
    """A partition of a surface's facets into spatially compact groups of at most ``size``.

    Attributes
    ----------
    members : np.ndarray of int, shape ``(n_clusters, size)``
        Facet indices of each cluster, ``-1`` in the slots of a last cluster left short.
        Every facet appears exactly once.
    centre : np.ndarray, shape ``(n_clusters, 3)``
        Centre of a sphere containing every vertex of the cluster's facets.
    radius : np.ndarray, shape ``(n_clusters,)``
        Radius of that sphere, enlarged by a rounding's worth so that a vertex exactly on it is
        still inside when the distance is recomputed elsewhere.
    """

    members: np.ndarray
    centre: np.ndarray
    radius: np.ndarray

    @classmethod
    def build(cls, vertices, size: int = 32) -> FacetClusters:
        """Group the triangles ``vertices`` into runs of ``size`` along their Morton order.

        Parameters
        ----------
        vertices : array_like, shape ``(n_facets, 3, 3)``
            Triangle corners.
        size : int
            Facets per cluster. Must be at least one.

        Returns
        -------
        FacetClusters

        Raises
        ------
        ValueError
            If ``size`` is less than one.
        """
        if size < 1:
            msg = f"a cluster must hold at least one facet; got size={size}"
            raise ValueError(msg)
        vertices = np.asarray(vertices, dtype=float)
        n = len(vertices)
        order = np.argsort(_morton_keys(vertices.mean(axis=1)), kind="stable")
        n_clusters = -(-n // size)
        members = np.full(n_clusters * size, -1, dtype=np.int64)
        members[:n] = order
        members = members.reshape(n_clusters, size)

        corners = vertices[np.maximum(members, 0)].reshape(n_clusters, 3 * size, 3)
        present = np.repeat(members >= 0, 3, axis=1)
        lo = np.where(present[..., None], corners, np.inf).min(axis=1)
        hi = np.where(present[..., None], corners, -np.inf).max(axis=1)
        centre = 0.5 * (lo + hi)
        distance = np.linalg.norm(corners - centre[:, None, :], axis=-1)
        radius = np.where(present, distance, 0.0).max(axis=1)
        # Slack for recomputing the same distance in another order, relative to the coordinates'
        # own size so that it means the same on a lamp in millimetres and a tank in metres.
        scale = np.abs(corners).max(axis=(1, 2)) + radius
        return cls(members=members, centre=centre, radius=radius + 1e-12 * scale)

    @property
    def size(self) -> int:
        """Facet slots per cluster."""
        return int(self.members.shape[1])


def _morton_keys(points: np.ndarray) -> np.ndarray:
    """Each point's position along a Z-order curve through the points' bounding box."""
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    lo = points.min(axis=0)
    extent = np.maximum(points.max(axis=0) - lo, np.finfo(float).tiny)
    cells = (1 << _MORTON_BITS) - 1
    grid = np.clip(((points - lo) / extent * cells).astype(np.int64), 0, cells)
    key = np.zeros(len(points), dtype=np.int64)
    for bit in range(_MORTON_BITS):
        for axis in range(3):
            key |= ((grid[:, axis] >> bit) & 1) << (3 * bit + axis)
    return key
