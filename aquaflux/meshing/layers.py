"""Layers of triangular prisms grown inward from a closed triangulated wall.

A wall-resolved mesh needs thin cells against the wall, growing away from it. Here they are grown on
the triangulated boundary of the region, before the rest is filled with tetrahedra: each point of a
layered wall moves inward along a smoothed normal, a step per layer, and each triangle sweeps out a
column of prisms. The top of the last layer, joined to the faces that carry no layers, is a closed
surface that a tetrahedral fill keeps exactly, so the layers and the core meet conformally.

**Sized for the dual mesh.** The mesh a solver sees is the polyhedral dual of these prisms (and of
the tetrahedra beyond), whose cells next to the wall run between the centroids of successive prisms.
So the layer heights are chosen for the dual: its wall cells are ``first_height`` tall and each
layer above is ``growth`` times the one below. The dual cell of the ``k``-th row of points spans from
the centroid of the prism below it to that of the prism above, so its height is the mean of those two
prisms' heights, and the prisms themselves alternate thicker and thinner about a geometric
progression (the alternation shrinks as ``growth ** k``). The first prism is ``2 first_height`` tall.

**Where a layered wall meets a face without layers** (a pipe's end, an inlet), the points on their
shared edge slide along that face, so the sides of the prisms lie on it, and the face's own triangles
are drawn in to meet the layers' top.

**How thick, where.** Along its direction each point's layers are stretched so that every face it
lies on is cleared by the full thickness (at most ``max_stretch`` times it), and thinned where the
wall faces another closely -- to at most ``gap_share`` of the distance across -- and then wherever a
prism, or a triangle drawn in, would turn inside out, halving the thickness there until none does.
Around a thinned point the layers taper back to full thickness, by at most half again from one
point to the next.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy.spatial import cKDTree

from aquaflux import ragged

__all__ = ["LayerExtrusion", "PrismLayers"]


@dataclasses.dataclass(frozen=True)
class LayerExtrusion:
    """Prism layers grown from a surface, and what is left for a fill.

    Attributes
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
        The surface's points first, at their indices (those that slid along a face without layers
        moved), then each layer's new points.
    prisms : np.ndarray of int, shape ``(n_prisms, 6)``
        Each prism's lower triangle, then its upper one, corner over corner.
    inner : np.ndarray of int, shape ``(n, 3)``
        The closed surface left for a fill: the top of the last layer and the triangles without
        layers, wound out of the region still to be filled.
    walls : np.ndarray of int, shape ``(n_triangles, 3)``
        The boundary triangles of the layered region with the faces without layers: the surface's
        triangles, those without layers drawn in to the layers' top. In the surface's order.
    sides : np.ndarray of int, shape ``(n_sides, 4)``
        The prisms' sides lying on a face without layers, each a boundary quadrilateral.
    side_triangle : np.ndarray of int, shape ``(n_sides,)``
        The surface triangle without layers each side lies beside.
    """

    points: np.ndarray
    prisms: np.ndarray
    inner: np.ndarray
    walls: np.ndarray
    sides: np.ndarray
    side_triangle: np.ndarray


@dataclasses.dataclass(frozen=True)
class PrismLayers:
    """Grow layers of prisms inward from the layered triangles of a closed surface.

    Attributes
    ----------
    first_height : float
        The height of the dual mesh's cells against the wall.
    growth : float
        The ratio of each dual layer's height to the one below, ``> 1``: equal dual layers would
        need prisms of zero height between them.
    count : int
        The number of prisms in each column.
    gap_share : float
        The largest share of the distance across to another wall the layers may take, in
        ``(0, 0.5]``, so that two walls facing each other leave room between their layers.
    max_stretch : float
        The most a point's layers are lengthened, at a fold, to clear the faces it lies on.
    max_halvings : int
        How many times a point's thickness may be halved to keep its prisms right side out.
    """

    first_height: float
    growth: float
    count: int
    gap_share: float = 0.4
    max_stretch: float = 2.0
    max_halvings: int = 8

    def __post_init__(self) -> None:
        if not (math.isfinite(self.first_height) and self.first_height > 0):
            raise ValueError(
                f"PrismLayers.first_height must be a positive, finite number, got {self.first_height!r}."
            )
        if not (math.isfinite(self.growth) and self.growth > 1):
            raise ValueError(
                f"PrismLayers.growth must be greater than 1, got {self.growth!r}: a dual layer is the "
                "mean of two prisms' heights, so equal dual layers would need prisms of zero height."
            )
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise ValueError(f"PrismLayers.count must be a whole number >= 1, got {self.count!r}.")
        if not (0 < self.gap_share <= 0.5):
            raise ValueError(f"PrismLayers.gap_share must be in (0, 0.5], got {self.gap_share!r}.")
        if not (math.isfinite(self.max_stretch) and self.max_stretch >= 1):
            raise ValueError(f"PrismLayers.max_stretch must be >= 1, got {self.max_stretch!r}.")

    def offsets(self) -> np.ndarray:
        """Each layer's top above the wall, before any stretch or thinning, shape ``(count,)``.

        The dual's layer boundaries -- the prisms' centroids -- sit at ``first_height`` times the
        partial sums of ``growth ** k``; each prism's top is placed so that its centroid is there.
        """
        boundaries = self.first_height * np.cumsum(self.growth ** np.arange(self.count))
        tops = np.empty(self.count)
        below = 0.0
        for k, boundary in enumerate(boundaries):
            tops[k] = 2.0 * boundary - below
            below = tops[k]
        return tops

    def extrude(
        self, points: np.ndarray, triangles: np.ndarray, layered: np.ndarray
    ) -> LayerExtrusion:
        """Grow the layers.

        Parameters
        ----------
        points : np.ndarray, shape ``(n_points, 3)``
        triangles : np.ndarray of int, shape ``(n_triangles, 3)``
            A closed surface wound out of the region, every point used.
        layered : np.ndarray of bool, shape ``(n_triangles,)``
            Which triangles carry layers.

        Returns
        -------
        LayerExtrusion

        Raises
        ------
        ValueError
            If a point of a layered triangle lies on three or more faces without layers that meet
            at an angle (it has nowhere to slide), or a prism stays inside out after the thickness
            there has been halved ``max_halvings`` times.
        """
        points = np.asarray(points, dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int64)
        layered = np.asarray(layered, dtype=bool)
        surface = _Surface.of(points, triangles)
        moving = np.zeros(points.shape[0], dtype=bool)
        moving[triangles[layered].reshape(-1)] = True
        direction, stretch = _directions(surface, layered, moving, self.max_stretch)
        offsets = self.offsets()
        thickness = stretch * offsets[-1]
        share = np.ones(points.shape[0])
        gap = _gaps(surface, direction, moving, 2.5 * thickness)
        share[moving] = np.minimum(1.0, self.gap_share * gap[moving] / thickness[moving])
        share = _smoothed_down(surface, share, moving)
        for _ in range(self.max_halvings + 1):
            extrusion = _built(surface, layered, moving, direction, offsets, share * stretch)
            bad = _inverted(extrusion, surface, layered, moving)
            if not bad.size:
                return extrusion
            share[bad] *= 0.5
            share = _smoothed_down(surface, share, moving)
        raise ValueError(
            f"{bad.size} point(s) still have a prism or a drawn-in triangle inside out after their "
            f"layers were thinned {self.max_halvings} times"
        )


@dataclasses.dataclass(frozen=True)
class _Surface:
    """A closed triangulated surface: its points, triangles, unit normals and point neighbours."""

    points: np.ndarray
    triangles: np.ndarray
    normals: np.ndarray  # (n_triangles, 3), outward
    angles: np.ndarray  # (n_triangles, 3): the interior angle at each corner
    point_offsets: np.ndarray  # CSR, point -> incident triangles
    point_triangles: np.ndarray
    edge_points: np.ndarray  # (n_edges, 2), ascending
    edge_triangles: np.ndarray  # (n_edges, 2)

    @classmethod
    def of(cls, points: np.ndarray, triangles: np.ndarray) -> _Surface:
        corner = points[triangles]
        normal = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        angles = np.empty(triangles.shape)
        for k in range(3):
            a = corner[:, (k + 1) % 3] - corner[:, k]
            b = corner[:, (k + 2) % 3] - corner[:, k]
            cosine = np.einsum("ij,ij->i", a, b) / (
                np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            )
            angles[:, k] = np.arccos(np.clip(cosine, -1.0, 1.0))
        point_offsets, order = ragged.group(triangles.reshape(-1), points.shape[0])
        point_triangles = order // 3
        edges = np.sort(np.stack([triangles, np.roll(triangles, -1, axis=1)], axis=2), axis=2)
        edges = edges.reshape(-1, 2)
        owner = np.repeat(np.arange(triangles.shape[0]), 3)
        sorter = np.lexsort(edges.T[::-1])
        edges, owner = edges[sorter], owner[sorter]
        return cls(
            points, triangles, normal, angles, point_offsets, point_triangles,
            edges[0::2], np.stack([owner[0::2], owner[1::2]], axis=1),
        )  # fmt: skip

    def around(self, point: int) -> np.ndarray:
        """The triangles at ``point``."""
        return self.point_triangles[self.point_offsets[point] : self.point_offsets[point + 1]]


def _directions(surface: _Surface, layered, moving, max_stretch):
    """Each moving point's unit direction into the region, and how much its layers are stretched.

    The direction is the angle-weighted mean of the inward normals of the layered triangles at the
    point, turned into the plane of any face without layers it lies on (so it slides along it). The
    stretch makes every layered face at the point cleared by the full thickness: ``1 / cos`` of the
    largest angle between the direction and those faces' inward normals, at most ``max_stretch``.
    """
    n_points = surface.points.shape[0]
    direction = np.zeros((n_points, 3))
    stretch = np.ones(n_points)
    for point in np.flatnonzero(moving):
        around = surface.around(point)
        position = np.argmax(surface.triangles[around] == point, axis=1)
        weight = surface.angles[around, position]
        on = layered[around]
        inward = -(weight[on, None] * surface.normals[around[on]]).sum(axis=0)
        planes = _distinct_normals(surface.normals[around[~on]])
        if len(planes) >= 3:
            raise ValueError(
                f"point {point} lies on three or more faces without layers, so cannot slide"
            )
        if len(planes) == 2:
            along = np.cross(planes[0], planes[1])
            inward = along if along @ inward >= 0.0 else -along
        elif len(planes) == 1:
            inward = inward - (inward @ planes[0]) * planes[0]
        inward /= np.linalg.norm(inward)
        direction[point] = inward
        cosine = np.min(-(surface.normals[around[on]] @ inward))
        stretch[point] = min(max_stretch, 1.0 / max(cosine, 1.0 / max_stretch))
    return direction, stretch


def _distinct_normals(normals: np.ndarray, tolerance: float = 1e-3) -> list[np.ndarray]:
    """The distinct planes among some triangles' unit normals (one per flat face)."""
    distinct: list[np.ndarray] = []
    for normal in normals:
        if all(abs(normal @ other) < 1 - tolerance for other in distinct):
            distinct.append(normal)
    if len(distinct) > 1:
        # Faces that are not flat (a cylinder's triangles) differ a little; merge near-parallel ones.
        merged: list[np.ndarray] = []
        for normal in distinct:
            if all(abs(normal @ other) < 0.9 for other in merged):
                merged.append(normal)
        return [m / np.linalg.norm(m) for m in merged]
    return distinct


def _gaps(surface: _Surface, direction, moving, reach) -> np.ndarray:
    """How far each moving point can go along its direction before meeting the surface again.

    Only triangles within ``reach`` of the point are looked at; beyond, the gap is ``inf``.
    """
    centroids = surface.points[surface.triangles].mean(axis=1)
    tree = cKDTree(centroids)
    gap = np.full(surface.points.shape[0], np.inf)
    for point in np.flatnonzero(moving):
        near = np.asarray(
            tree.query_ball_point(surface.points[point], reach[point]), dtype=np.int64
        )
        if not near.size:
            continue
        near = near[~np.any(surface.triangles[near] == point, axis=1)]
        hit = _ray_hits(
            surface.points[point], direction[point], surface.points[surface.triangles[near]]
        )
        if hit.size:
            gap[point] = hit.min()
    return gap


def _ray_hits(origin, direction, corners) -> np.ndarray:
    """The distances along a ray at which it crosses each triangle it crosses (Moller-Trumbore)."""
    edge1 = corners[:, 1] - corners[:, 0]
    edge2 = corners[:, 2] - corners[:, 0]
    pvec = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tvec = origin - corners[:, 0]
    u = np.einsum("ij,ij->i", tvec, pvec) * inv
    qvec = np.cross(tvec, edge1)
    v = (qvec @ direction) * inv
    t = np.einsum("ij,ij->i", edge2, qvec) * inv
    inside = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return t[inside]


def _smoothed_down(surface: _Surface, share, moving, ratio: float = 1.5) -> np.ndarray:
    """``share`` lowered until no moving point's exceeds ``ratio`` times its thinnest neighbour's.

    Never raised anywhere: a point thinned to keep its prisms right side out, or by a close wall,
    thins the points around it only as far as needed for the layers to change gently -- by at most
    ``ratio`` from one point to the next -- not the whole wall.
    """
    first, second = surface.edge_points[:, 0], surface.edge_points[:, 1]
    share = share.copy()
    while True:
        thinnest = np.full_like(share, np.inf)
        np.minimum.at(thinnest, first, share[second])
        np.minimum.at(thinnest, second, share[first])
        limited = np.where(moving, np.minimum(share, ratio * thinnest), share)
        if np.array_equal(limited, share):
            return share
        share = limited


def _built(surface, layered, moving, direction, offsets, reach) -> LayerExtrusion:
    """The extrusion with each moving point's layers along ``direction``, ``reach`` times
    ``offsets`` from the wall."""
    points, triangles = surface.points, surface.triangles
    n_points, n_layers = points.shape[0], offsets.size
    rank = np.full(n_points, -1, dtype=np.int64)
    movers = np.flatnonzero(moving)
    rank[movers] = np.arange(movers.size)
    # Row k of the layers: row 0 is the wall itself, row n the layers' top.
    rows = [np.arange(n_points)]
    added = [points]
    base = n_points
    for k in range(n_layers):
        row = np.arange(n_points)
        row[movers] = base + rank[movers]
        rows.append(row)
        added.append(points[movers] + (reach[movers] * offsets[k])[:, None] * direction[movers])
        base += movers.size
    all_points = np.concatenate(added)
    on = np.flatnonzero(layered)
    prisms = np.concatenate([
        np.concatenate([rows[k][triangles[on]], rows[k + 1][triangles[on]]], axis=1)
        for k in range(n_layers)
    ])  # fmt: skip
    top = rows[n_layers]
    walls = triangles.copy()
    walls[~layered] = top[triangles[~layered]]
    # A prism's side on a face without layers: along each edge between a layered triangle and one
    # without.
    first, second = surface.edge_triangles[:, 0], surface.edge_triangles[:, 1]
    rim = layered[first] != layered[second]
    beside = np.where(layered[first[rim]], second[rim], first[rim])
    a, b = surface.edge_points[rim, 0], surface.edge_points[rim, 1]
    sides = np.concatenate([
        np.stack([rows[k][a], rows[k][b], rows[k + 1][b], rows[k + 1][a]], axis=1)
        for k in range(n_layers)
    ])  # fmt: skip
    inner = np.concatenate([top[triangles[on]], walls[~layered]])
    return LayerExtrusion(all_points, prisms, inner, walls, sides, np.tile(beside, n_layers))


def _inverted(extrusion: LayerExtrusion, surface: _Surface, layered, moving) -> np.ndarray:
    """The moving points of any prism that is inside out, or of any drawn-in triangle that turned."""
    points = extrusion.points
    corners = points[extrusion.prisms]
    bad = np.zeros(extrusion.prisms.shape[0], dtype=bool)
    # The three tetrahedra of a prism's standard split must each keep the orientation of the first
    # one of a right prism: lower triangle wound out of the region, the upper one inward of it.
    for tet in ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)):
        p = corners[:, tet]
        volume = np.einsum(
            "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
        )
        bad |= volume >= 0.0  # a lower triangle wound outward puts the inside at negative volume
    drawn = extrusion.walls[~layered]
    c = points[drawn]
    normal = np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0])
    turned = np.einsum("ij,ij->i", normal, surface.normals[~layered]) <= 0.0
    # The surface's points at the bad cells, among the moving ones.
    n_surface = surface.points.shape[0]
    rows = np.concatenate([extrusion.prisms[bad].reshape(-1), drawn[turned].reshape(-1)])
    wall_points = _wall_point(rows, n_surface, np.flatnonzero(moving))
    return np.unique(wall_points[moving[wall_points]])


def _wall_point(indices: np.ndarray, n_surface: int, movers: np.ndarray) -> np.ndarray:
    """The surface point each layer point was grown from (a surface point is its own)."""
    out = indices.copy()
    above = indices >= n_surface
    out[above] = movers[(indices[above] - n_surface) % movers.size]
    return out
