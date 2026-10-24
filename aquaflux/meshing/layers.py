"""Layers of prisms grown inward from a closed wall of triangles and quadrilaterals.

A wall-resolved mesh needs thin cells against the wall, growing away from it. Here they are grown on
the boundary of the region, before the rest is filled: each point of a layered wall moves inward
along a smoothed normal, a step per layer, and each face sweeps out a column of cells -- triangular
prisms over a triangle, hexahedra over a quadrilateral. The top of the last layer, joined to the
faces that carry no layers, is a closed surface that a fill keeps exactly, so the layers and the core
meet conformally. A wall of structured quadrilaterals gives hexahedral layers, and so a dual mesh
whose wall cells are hexahedra too.

**Sized for the dual mesh.** The mesh a solver sees is the polyhedral dual of these cells (and of
the cells beyond), whose cells next to the wall run between the centroids of successive layers.
So the layer heights are chosen for the dual: its wall cells are ``first_height`` tall and each
layer above is ``growth`` times the one below. The dual cell of the ``k``-th row of points spans from
the centroid of the cell below it to that of the cell above, so its height is the mean of those two
cells' heights, and the cells themselves alternate thicker and thinner about a geometric
progression (the alternation shrinks as ``growth ** k``). The first is ``2 first_height`` tall.

**Where a layered wall meets a face without layers** (a pipe's end, an inlet), the points on their
shared edge slide along that face, so the sides of the layers' cells lie on it, and the face's own
faces are drawn in to meet the layers' top.

**How thick, where.** Along its direction each point's layers are stretched so that every face it
lies on is cleared by the full thickness (at most ``max_stretch`` times it), and thinned where the
wall faces another closely -- to at most ``gap_share`` of the distance across -- and then wherever a
cell, or a face drawn in, would turn inside out, halving the thickness there until none does.
Where the region's walls meet at less than a straight angle, the columns there lean together and
the faces of the layers' top beside the edge come out narrower than the wall faces under them -- by
about the thickness, so a face whose wall is about as wide as the layers are thick collapses to a
sliver. A sliver is no face for a fill to meet (a pyramid on it stands over its neighbours' faces),
so an edge of the top shorter than ``min_top_share`` of the wall edge under it is halved the same
way. Around a thinned point the layers taper back to full thickness, by at most half again from one
point to the next.

**Where layered faces fold right back over each other** -- the two sides of a sheet standing in the
region, along its free edge -- no one direction clears both, so the layers fan round the edge: each
point there grows a column for each side and one between them, ahead of the edge (at a corner of
the sheet, one ahead of each of its two edges), and the cells between the columns wrap round it
(:mod:`~aquaflux.meshing.fans`). Where the edge ends on a wall, the fan closes down onto the wall.

**Points at one place grow alike.** A surface opened along a sheet standing in the region has a
point for each side of the sheet where the sides part, at one place. Where such a point slides along
the sheet beside a layered wall, each copy grows the same column -- the same direction, the
thinnest thickness of any -- so the layers' tops on the two sides meet at one place again, and the
sheet's two sides above them stay one surface a fill can keep.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from aquaflux import ragged

from .dual import FOLDED_BACK
from .fans import FannedSurface, fanned, merged
from .polygons import Polygons
from .regions import first_hits

__all__ = ["LayerExtrusion", "PrismLayers"]

#: The tetrahedra whose volumes say whether a layer cell is right side out, for a cell whose lower
#: face is wound out of the region and its upper face is the same ring one layer in, each ordered so
#: a right cell gives a negative volume: a prism's three of its standard split; a hexahedron's eight
#: at its corners, each a corner and its three neighbours along the cell's edges.
_PRISM_TETRAHEDRA = ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5))
#: The share of a layer cell's tetrahedra's volumes within which one of them counts as flat.
_FLAT_SHARE = 1e-9

_HEXAHEDRON_TETRAHEDRA = tuple(
    [(i, (i + 1) % 4, (i + 3) % 4, i + 4) for i in range(4)]
    + [(4 + i, 4 + (i + 3) % 4, 4 + (i + 1) % 4, i) for i in range(4)]
)


@dataclasses.dataclass(frozen=True)
class LayerExtrusion:
    """Layers grown from a surface, and what is left for a fill.

    Attributes
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
        The surface's points first, at their indices (those that slid along a face without layers
        moved), then each layer's new points.
    prisms : np.ndarray of int, shape ``(n_prisms, 6)``
        The cells over the layered triangles: each prism's lower triangle, then its upper one,
        corner over corner. Layer by layer, each layer in the surface's order.
    hexahedra : np.ndarray of int, shape ``(n_hexahedra, 8)``
        The cells over the layered quadrilaterals, likewise. Where the layers fan round an edge,
        the fan's cells follow each kind's: its wedges and hexahedra.
    tetrahedra : np.ndarray of int, shape ``(n_tetrahedra, 4)``
        The fan's first cells in the corners of a folded-back edge; none without a fan.
    tents : np.ndarray of int, shape ``(n_tents, 6)``
        The fan's first cells where it closes down onto a wall, as
        :func:`~aquaflux.meshing.element_mesh` takes them; none without a fan.
    inner : Polygons
        The closed surface left for a fill: the top of the last layer and the faces without layers,
        wound out of the region still to be filled.
    walls : Polygons
        The boundary faces of the layered region with the faces without layers: the surface's
        faces, those without layers drawn in to the layers' top. In the surface's order.
    sides : Polygons
        The layer cells' sides lying on a face without layers, each a boundary face: a
        quadrilateral, or where a fan closes down onto the face, a triangle.
    side_face : np.ndarray of int, shape ``(sides.n_faces,)``
        The surface face without layers each side lies on or beside.
    columns : np.ndarray of int, shape ``(n_moved, count + 1)``
        Each moved surface point, then the point each layer grew above it. Where the layers fan
        round an edge, a point heads one column for each copy it was split into.
    fanned : np.ndarray of bool, shape ``(n_moved,)``
        Whether each column stands on a point the layers fan round. Its cells wrap round the
        edge, so a dual cutting the point's own cell there need not cut theirs (cut with it, the
        cells beside the edge come out far less convex).
    """

    points: np.ndarray
    prisms: np.ndarray
    hexahedra: np.ndarray
    tetrahedra: np.ndarray
    tents: np.ndarray
    inner: Polygons
    walls: Polygons
    sides: Polygons
    side_face: np.ndarray
    columns: np.ndarray
    fanned: np.ndarray


@dataclasses.dataclass(frozen=True)
class PrismLayers:
    """Grow layers of prisms inward from the layered faces of a closed surface.

    Attributes
    ----------
    first_height : float
        The height of the dual mesh's cells against the wall.
    growth : float
        The ratio of each dual layer's height to the one below, ``> 1``: equal dual layers would
        need cells of zero height between them.
    count : int
        The number of cells in each column.
    gap_share : float
        The largest share of the distance across to another wall the layers may take, in
        ``(0, 0.5]``, so that two walls facing each other leave room between their layers.
    max_stretch : float
        The most a point's layers are lengthened, at a fold, to clear the faces it lies on.
    max_halvings : int
        How many times a point's thickness may be halved to keep its cells right side out.
    min_top_share : float
        The shortest an edge of the layers' top may be, as a share of the wall edge under it, in
        ``[0, 1)``; a shorter one is thinned. The default leaves untouched the uniform narrowing
        inside a pipe whose layers take ``gap_share`` of it (``1 - 2 gap_share``, 0.2 at the
        default), and catches a top face collapsed where walls meet.
    """

    first_height: float
    growth: float
    count: int
    gap_share: float = 0.4
    max_stretch: float = 2.0
    max_halvings: int = 8
    min_top_share: float = 0.1

    def __post_init__(self) -> None:
        if not (math.isfinite(self.first_height) and self.first_height > 0):
            raise ValueError(
                f"PrismLayers.first_height must be a positive, finite number, got {self.first_height!r}."
            )
        if not (math.isfinite(self.growth) and self.growth > 1):
            raise ValueError(
                f"PrismLayers.growth must be greater than 1, got {self.growth!r}: a dual layer is the "
                "mean of two cells' heights, so equal dual layers would need cells of zero height."
            )
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise ValueError(f"PrismLayers.count must be a whole number >= 1, got {self.count!r}.")
        if not (0 < self.gap_share <= 0.5):
            raise ValueError(f"PrismLayers.gap_share must be in (0, 0.5], got {self.gap_share!r}.")
        if not (math.isfinite(self.max_stretch) and self.max_stretch >= 1):
            raise ValueError(f"PrismLayers.max_stretch must be >= 1, got {self.max_stretch!r}.")
        if not (0 <= self.min_top_share < 1):
            raise ValueError(
                f"PrismLayers.min_top_share must be in [0, 1), got {self.min_top_share!r}."
            )

    def offsets(self) -> np.ndarray:
        """Each layer's top above the wall, before any stretch or thinning, shape ``(count,)``.

        The dual's layer boundaries -- the cells' centroids -- sit at ``first_height`` times the
        partial sums of ``growth ** k``; each cell's top is placed so that its centroid is there.
        """
        boundaries = self.first_height * np.cumsum(self.growth ** np.arange(self.count))
        tops = np.empty(self.count)
        below = 0.0
        for k, boundary in enumerate(boundaries):
            tops[k] = 2.0 * boundary - below
            below = tops[k]
        return tops

    def extrude(
        self,
        points: np.ndarray,
        faces: Polygons,
        layered: np.ndarray,
        twins: np.ndarray | None = None,
    ) -> LayerExtrusion:
        """Grow the layers.

        Parameters
        ----------
        points : np.ndarray, shape ``(n_points, 3)``
        faces : Polygons
            A closed surface wound out of the region, every point used.
        layered : np.ndarray of bool, shape ``(faces.n_faces,)``
            Which faces carry layers, the triangles' first.
        twins : np.ndarray of int, shape ``(n_points,)``, optional
            For each point, the point it is a copy of, at the same place (itself if none): points
            sharing one grow the same column. Unset, every point is its own.

        Returns
        -------
        LayerExtrusion

        Raises
        ------
        ValueError
            If a point of a layered face lies on three or more faces without layers that meet at an
            angle (it has nowhere to slide), or a cell stays inside out (or an edge of the top too
            short) after the thickness there has been halved ``max_halvings`` times.
        """
        points = np.asarray(points, dtype=np.float64)
        layered = np.asarray(layered, dtype=bool)
        fan = fanned(points, faces, layered)
        if fan is None:
            return self._grown(_Surface.of(points, faces), layered, twins)
        n_points = points.shape[0]
        own = np.arange(n_points) if twins is None else np.asarray(twins, dtype=np.int64)
        twins = np.concatenate([own, np.arange(n_points, fan.points.shape[0])])
        middles = _Middles.of(_Surface.of(points, faces), layered, fan)
        surface = _Surface.of(fan.points, fan.faces, added=fan.added)
        grown = self._grown(surface, layered[fan.face], twins, middles)
        return _merged(grown, fan, layered[fan.face])

    def _grown(self, surface: _Surface, layered, twins, middles=None) -> LayerExtrusion:
        """The layers grown on ``surface``, every point with its own column, the fan's
        ``middles``' columns set between their two sides'."""
        n_points = surface.points.shape[0]
        moving = np.zeros(n_points, dtype=bool)
        rings = surface.rings[layered]
        moving[rings[rings >= 0]] = True
        alike = _Alike.of(twins, n_points)
        direction, stretch = _directions(surface, layered, moving, self.max_stretch)
        if middles is not None:
            middles.set(direction, stretch, surface.points, self.max_stretch)
        direction, stretch = alike.direction(direction), alike.largest(stretch)
        offsets = self.offsets()
        thickness = stretch * offsets[-1]
        share = np.ones(n_points)
        gap = _gaps(surface, direction, moving, 2.5 * thickness)
        share[moving] = np.minimum(1.0, self.gap_share * gap[moving] / thickness[moving])
        share = alike.smallest(_smoothed_down(surface, share, moving))
        for _ in range(self.max_halvings + 1):
            extrusion = _built(surface, layered, moving, direction, offsets, share * stretch)
            bad = _to_be_thinned(extrusion, surface, layered, moving, self.min_top_share)
            if not bad.size:
                return extrusion
            share[bad] *= 0.5
            share = alike.smallest(_smoothed_down(surface, alike.smallest(share), moving))
        raise ValueError(
            f"{bad.size} point(s) still have a cell or a drawn-in face inside out, or an edge of "
            f"the layers' top too short, after their layers were thinned {self.max_halvings} times"
        )


@dataclasses.dataclass(frozen=True)
class _Alike:
    """Points at one place, given one value: each point's group (unset: each its own, and every
    value is returned as it is)."""

    group: np.ndarray | None
    n_groups: int

    @classmethod
    def of(cls, twins: np.ndarray | None, n_points: int) -> _Alike:
        if twins is None:
            return cls(None, n_points)
        _, group = np.unique(np.asarray(twins, dtype=np.int64), return_inverse=True)
        return cls(group.reshape(-1), int(group.max()) + 1)

    def smallest(self, values: np.ndarray) -> np.ndarray:
        """Each point's value replaced by the smallest of its group's."""
        if self.group is None:
            return values
        least = np.full(self.n_groups, np.inf)
        np.minimum.at(least, self.group, values)
        return least[self.group]

    def largest(self, values: np.ndarray) -> np.ndarray:
        """Each point's value replaced by the largest of its group's."""
        if self.group is None:
            return values
        most = np.full(self.n_groups, -np.inf)
        np.maximum.at(most, self.group, values)
        return most[self.group]

    def direction(self, directions: np.ndarray) -> np.ndarray:
        """Each point's unit direction replaced by its group's mean, normalized; a point with no
        direction (zero) adds nothing to it, and keeps none if its group has none."""
        if self.group is None:
            return directions
        total = np.zeros((self.n_groups, 3))
        np.add.at(total, self.group, directions)
        length = np.linalg.norm(total, axis=1, keepdims=True)
        mean = np.divide(total, length, out=np.zeros_like(total), where=length > 0.0)
        return mean[self.group]


@dataclasses.dataclass(frozen=True)
class _Surface:
    """A closed surface: its points, faces, unit normals and point neighbours, and which faces have
    no area (joining copies of a point, :mod:`~aquaflux.meshing.fans`): theirs normals and angles
    are zero, so they weigh nothing in a point's direction."""

    points: np.ndarray
    faces: Polygons
    rings: np.ndarray  # (n_faces, 4), a triangle's fourth corner -1
    normals: np.ndarray  # (n_faces, 3), outward
    angles: np.ndarray  # (n_faces, 4): the interior angle at each corner, 0 past a triangle's third
    point_offsets: np.ndarray  # CSR, point -> incident faces
    point_faces: np.ndarray
    edge_points: np.ndarray  # (n_edges, 2), ascending
    edge_faces: np.ndarray  # (n_edges, 2)
    added: np.ndarray  # (n_faces,) bool: a face of no area

    @classmethod
    def of(cls, points: np.ndarray, faces: Polygons, added: np.ndarray | None = None) -> _Surface:
        added = np.zeros(faces.n_faces, dtype=bool) if added is None else np.asarray(added, bool)
        rings = faces.padded()
        angles = np.zeros(rings.shape)
        base = 0
        for kind in faces.kinds():
            corner = points[kind]
            size = kind.shape[1]
            for k in range(size):
                a = corner[:, (k + 1) % size] - corner[:, k]
                b = corner[:, (k - 1) % size] - corner[:, k]
                with np.errstate(divide="ignore", invalid="ignore"):
                    cosine = np.einsum("ij,ij->i", a, b) / (
                        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
                    )
                angles[base : base + kind.shape[0], k] = np.arccos(np.clip(cosine, -1.0, 1.0))
            base += kind.shape[0]
        used = rings >= 0
        point_offsets, order = ragged.group(rings[used], points.shape[0])
        point_faces = np.nonzero(used)[0][order]
        pairs, owner = faces.edges()
        edges = np.sort(pairs, axis=1)
        sorter = np.lexsort(edges.T[::-1])
        edges, owner = edges[sorter], owner[sorter]
        normals = np.zeros((faces.n_faces, 3))
        real = faces.where(~added)
        normals[~added] = real.normals(points)
        angles[added] = 0.0
        return cls(
            points, faces, rings, normals, angles, point_offsets, point_faces,
            edges[0::2], np.stack([owner[0::2], owner[1::2]], axis=1), added,
        )  # fmt: skip

    def around(self, point: int) -> np.ndarray:
        """The faces at ``point``."""
        return self.point_faces[self.point_offsets[point] : self.point_offsets[point + 1]]


def _directions(surface: _Surface, layered, moving, max_stretch):
    """Each moving point's unit direction into the region, and how much its layers are stretched.

    The direction is the angle-weighted mean of the inward normals of the layered faces at the
    point, turned into the plane of any face without layers it lies on (so it slides along it). The
    stretch makes every layered face at the point cleared by the full thickness: ``1 / cos`` of the
    largest angle between the direction and those faces' inward normals, at most ``max_stretch``.
    Faces of no area take no part; a point on nothing else (the middle copy of a fanned point)
    gets no direction here.
    """
    n_points = surface.points.shape[0]
    direction = np.zeros((n_points, 3))
    stretch = np.ones(n_points)
    for point in np.flatnonzero(moving):
        around = surface.around(point)
        around = around[~surface.added[around]]
        if not around.size:
            continue
        position = np.argmax(surface.rings[around] == point, axis=1)
        direction[point], stretch[point] = _lean(
            surface.normals[around], surface.angles[around, position], layered[around],
            np.zeros((around.size, 3)), max_stretch, point,
        )  # fmt: skip
    return direction, stretch


@dataclasses.dataclass(frozen=True)
class _Middles:
    """The middle columns of a fan (:mod:`~aquaflux.meshing.fans`): each between its point's two
    sides' columns.

    Between two folded-back edges it goes ahead of them (``ahead``, leaning off nothing else); where
    the edge turns, each of its two middle columns ahead of its own edge.
    Where the fan closes down onto a wall along an edge, it bisects the two sides' columns, tipped
    along that edge: so round the edge it lies between them, and the two cells closing the fan
    there are right side out whatever way the edge runs. Each is stretched to clear the point's
    other layered faces (``upright``, their unit mean inward normal; zero if none).
    """

    middles: np.ndarray
    sides: np.ndarray  # (n, 2): the copies of the two sides
    landing: np.ndarray
    ahead: np.ndarray
    upright: np.ndarray

    @classmethod
    def of(cls, surface: _Surface, layered, fan: FannedSurface) -> _Middles:
        """The middle columns of ``fan``, read off the surface it split."""
        centroids = surface.faces.centroids(surface.points)
        ahead, upright = np.zeros((fan.split.size, 3)), np.zeros((fan.split.size, 3))
        for k, point in enumerate(fan.split.tolist()):
            around = surface.around(point)
            position = np.argmax(surface.rings[around] == point, axis=1)
            normals, weights = surface.normals[around], surface.angles[around, position]
            back = centroids[around] - surface.points[point]
            if fan.ahead_of[k] >= 0:
                ahead[k] = _square_off(surface, point, fan.ahead_of[k], around, back)
            else:
                ahead[k], _ = _lean(normals, weights, layered[around], back, 1.0, point)
            folded = _folded(normals, layered[around])
            mean = -(weights[~folded & layered[around], None] * normals[~folded & layered[around]])
            length = np.linalg.norm(mean.sum(axis=0))
            upright[k] = mean.sum(axis=0) / length if length > 0.0 else 0.0
        return cls(
            fan.middles, np.stack([fan.split, fan.seconds], axis=1), fan.landing, ahead, upright
        )

    def set(self, direction, stretch, points, max_stretch) -> None:
        """Set the middle copies' ``direction`` and ``stretch``, in place, from their sides'."""
        for k, middle in enumerate(self.middles.tolist()):
            if self.landing[k] >= 0:
                along = points[self.landing[k]] - points[self.sides[k, 0]]
                between = direction[self.sides[k]].sum(axis=0)
                lean = between / np.linalg.norm(between) + along / np.linalg.norm(along)
                direction[middle] = lean / np.linalg.norm(lean)
            else:
                direction[middle] = self.ahead[k]
            cosine = direction[middle] @ self.upright[k]
            stretch[middle] = (
                1.0
                if not self.upright[k].any()
                else min(max_stretch, 1.0 / max(cosine, 1.0 / max_stretch))
            )


def _square_off(surface: _Surface, point: int, end: int, around, back) -> np.ndarray:
    """The unit direction square off a folded-back edge from ``point`` to ``end``, in its faces'
    plane, away from them: where the edge turns, the faces at the corner lie diagonally behind
    it, so the way out of them is not square off either edge."""
    along = surface.points[end] - surface.points[point]
    along /= np.linalg.norm(along)
    on_edge = np.any(surface.rings[around] == end, axis=1)
    normal = surface.normals[around[on_edge][0]]
    away = -back[on_edge].mean(axis=0)
    away -= (away @ along) * along + (away @ normal) * normal
    return away / np.linalg.norm(away)


def _folded(normals: np.ndarray, layered: np.ndarray) -> np.ndarray:
    """Which of the layered faces at a point fold right back over another layered one there."""
    return layered & np.any((normals @ normals[layered].T) <= FOLDED_BACK - 1.0, axis=1)


def _lean(normals, weights, layered, back, max_stretch, point):
    """A point's unit direction into the region and its stretch, from the faces round it: their
    outward ``normals``, angle ``weights`` at the point, which are ``layered``, and the way ``back``
    from the point to each one's centroid.

    The direction is the weighted mean of the layered faces' inward normals, turned into the plane
    of any face without layers (so it slides along it). Layered faces folded right back over each
    other give no direction: they are cleared by going ahead of the edge they fold at, along them
    -- the way from their centroids back to the point, in their planes -- bisected with the other
    layered faces' mean. The stretch clears every other layered face by the full thickness.
    """
    folded = _folded(normals, layered)
    upright = layered & ~folded
    inward = -(weights[upright, None] * normals[upright]).sum(axis=0)
    if folded.any():
        along = (
            back[folded]
            - np.einsum("ij,ij->i", back[folded], normals[folded])[:, None] * (normals[folded])
        )
        ahead = -(weights[folded, None] * along).sum(axis=0)
        ahead /= np.linalg.norm(ahead)
        length = np.linalg.norm(inward)
        inward = ahead if length == 0.0 else ahead + inward / length
    planes = _distinct_normals(normals[~layered])
    if len(planes) >= 3:
        raise ValueError(
            f"point {point} lies on three or more faces without layers, so cannot slide"
        )
    if len(planes) == 2:
        along_both = np.cross(planes[0], planes[1])
        inward = along_both if along_both @ inward >= 0.0 else -along_both
    elif len(planes) == 1:
        inward = inward - (inward @ planes[0]) * planes[0]
    inward /= np.linalg.norm(inward)
    if not upright.any():
        return inward, 1.0
    cosine = np.min(-(normals[upright] @ inward))
    return inward, min(max_stretch, 1.0 / max(cosine, 1.0 / max_stretch))


def _distinct_normals(normals: np.ndarray, tolerance: float = 1e-3) -> list[np.ndarray]:
    """The distinct planes among some faces' unit normals (one per flat face)."""
    distinct: list[np.ndarray] = []
    for normal in normals:
        if all(abs(normal @ other) < 1 - tolerance for other in distinct):
            distinct.append(normal)
    if len(distinct) > 1:
        # Faces that are not flat (a cylinder's facets) differ a little; merge near-parallel ones.
        merged: list[np.ndarray] = []
        for normal in distinct:
            if all(abs(normal @ other) < 0.9 for other in merged):
                merged.append(normal)
        return [m / np.linalg.norm(m) for m in merged]
    return distinct


def _gaps(surface: _Surface, direction, moving, reach) -> np.ndarray:
    """How far each moving point can go along its direction before meeting the surface again.

    Only faces within ``reach`` of the point are looked at; beyond, the gap is ``inf``.
    """
    gap = np.full(surface.points.shape[0], np.inf)
    moved = np.flatnonzero(moving)
    gap[moved] = first_hits(
        surface.points, surface.faces.where(~surface.added), surface.points[moved],
        direction[moved], reach[moved], _at_one_place(surface.points)[moved],
    )  # fmt: skip
    return gap


def _at_one_place(points: np.ndarray) -> np.ndarray:
    """For each point, every point at its place (itself and any copies of it), padded with ``-1``:
    a ray from it starts on all their faces. Shape ``(n_points, k)``."""
    _, group, counts = np.unique(points, axis=0, return_inverse=True, return_counts=True)
    group = group.reshape(-1)
    offsets, order = ragged.group(group, counts.size)
    rows = np.full((points.shape[0], int(counts.max())), -1, dtype=np.int64)
    for k in range(int(counts.max())):
        has = offsets[group] + k < offsets[group + 1]
        rows[has, k] = order[offsets[group[has]] + k]
    return rows


def _smoothed_down(surface: _Surface, share, moving, ratio: float = 1.5) -> np.ndarray:
    """``share`` lowered until no moving point's exceeds ``ratio`` times its thinnest neighbour's.

    Never raised anywhere: a point thinned to keep its cells right side out, or by a close wall,
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
    points, faces = surface.points, surface.faces
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
    on = faces.where(layered)
    cells = [
        np.concatenate(
            [np.concatenate([rows[k][kind], rows[k + 1][kind]], axis=1) for k in range(n_layers)]
        )
        for kind in on.kinds()
    ]
    top = rows[n_layers]
    bare = faces.where(~layered)
    drawn = bare.renumbered(top)
    triangles, quads = faces.kinds()
    tri_layered, quad_layered = faces.split(layered)
    walls = Polygons(
        np.where(tri_layered[:, None], triangles, top[triangles]),
        np.where(quad_layered[:, None], quads, top[quads]),
    )
    # A cell's side on a face without layers: along each edge between a layered face and one
    # without.
    first, second = surface.edge_faces[:, 0], surface.edge_faces[:, 1]
    rim = layered[first] != layered[second]
    beside = np.where(layered[first[rim]], second[rim], first[rim])
    a, b = surface.edge_points[rim, 0], surface.edge_points[rim, 1]
    sides = np.concatenate([
        np.stack([rows[k][a], rows[k][b], rows[k + 1][b], rows[k + 1][a]], axis=1)
        for k in range(n_layers)
    ])  # fmt: skip
    lifted = on.renumbered(top)
    inner = Polygons(
        np.concatenate([lifted.triangles, drawn.triangles]),
        np.concatenate([lifted.quads, drawn.quads]),
    )
    columns = np.stack([row[movers] for row in rows], axis=1)
    return LayerExtrusion(
        all_points, cells[0], cells[1], np.zeros((0, 4), np.int64), np.zeros((0, 6), np.int64),
        inner, walls, Polygons(quads=sides), np.tile(beside, n_layers), columns,
        np.zeros(columns.shape[0], dtype=bool),
    )  # fmt: skip


def _to_be_thinned(extrusion: LayerExtrusion, surface: _Surface, layered, moving, min_top_share):
    """The moving points of any cell that is inside out, of any drawn-in face that turned, and at
    either end of any edge of the layers' top shorter than ``min_top_share`` of the wall edge
    under it."""
    points = extrusion.points
    rows = []
    for cells, tetrahedra in ((extrusion.prisms, _PRISM_TETRAHEDRA),
                              (extrusion.hexahedra, _HEXAHEDRON_TETRAHEDRA)):  # fmt: skip
        corner = points[cells]
        # Each tetrahedron keeps the orientation it has in a right cell: the lower face wound out of
        # the region puts the inside at negative volume. One may be flat -- a cell at a sheet's free
        # edge, its column grown along the sheet, has its front face in its lower face's plane --
        # but none may turn, and the cell must hold volume.
        volume = np.stack([
            np.einsum(
                "ij,ij->i",
                np.cross(corner[:, b] - corner[:, a], corner[:, c] - corner[:, a]),
                corner[:, d] - corner[:, a],
            )
            for a, b, c, d in tetrahedra
        ], axis=1)  # fmt: skip
        flat = _FLAT_SHARE * np.abs(volume).sum(axis=1)
        bad = np.any(volume > flat[:, None], axis=1) | (volume.sum(axis=1) >= -flat)
        rows.append(cells[bad].reshape(-1))
    bare = ~layered & ~surface.added
    drawn = extrusion.walls.where(bare)
    turned = np.einsum("ij,ij->i", drawn.normals(points), surface.normals[bare]) <= 0.0
    turned_triangles, turned_quads = drawn.split(turned)
    rows += [drawn.triangles[turned_triangles].reshape(-1), drawn.quads[turned_quads].reshape(-1)]
    top = np.arange(points.shape[0])
    top[extrusion.columns[:, 0]] = extrusion.columns[:, -1]
    pairs, _ = surface.faces.where(layered).edges()
    wall = np.linalg.norm(points[pairs[:, 1]] - points[pairs[:, 0]], axis=1)
    above = np.linalg.norm(points[top[pairs[:, 1]]] - points[top[pairs[:, 0]]], axis=1)
    rows.append(pairs[above < min_top_share * wall].reshape(-1))
    # The surface's points at the bad cells, among the moving ones.
    n_surface = surface.points.shape[0]
    wall_points = _wall_point(np.concatenate(rows), n_surface, np.flatnonzero(moving))
    return np.unique(wall_points[moving[wall_points]])


def _wall_point(indices: np.ndarray, n_surface: int, movers: np.ndarray) -> np.ndarray:
    """The surface point each layer point was grown from (a surface point is its own)."""
    out = indices.copy()
    above = indices >= n_surface
    out[above] = movers[(indices[above] - n_surface) % movers.size]
    return out


def _merged(grown: LayerExtrusion, fan: FannedSurface, layered: np.ndarray) -> LayerExtrusion:
    """Layers grown on a fanned surface, its copies merged back into their points: the surface's
    own faces as walls, in its order; the fan's faces of no area left out, but those without
    layers, drawn in to the layers' top, kept as sides on the face they part."""
    cells = merged(fan, grown.points.shape[0], layered, grown.prisms, grown.hexahedra)
    renumber = cells.renumber
    n_fanned, n_points = fan.points.shape[0], int(fan.point.max()) + 1
    tri_added, quad_added = fan.faces.split(fan.added)
    walls = Polygons(grown.walls.triangles[~tri_added], grown.walls.quads[~quad_added])
    # Sides, with the surface face each lies on or beside: a face of no area without layers drawn
    # in, then the cells' sides; each after merging, a corner repeated at a copy dropped.
    drawn = fan.added & ~layered
    rings = np.vstack([grown.walls.padded()[drawn], grown.sides.padded()])
    lies_on = np.concatenate([fan.face[drawn], fan.face[grown.side_face]])
    sides, side_face = _without_repeats(np.where(rings >= 0, renumber[rings], -1), lies_on)
    return LayerExtrusion(
        points=np.vstack([grown.points[:n_points], grown.points[n_fanned:]]),
        prisms=cells.prisms,
        hexahedra=cells.hexahedra,
        tetrahedra=cells.tetrahedra,
        tents=cells.tents,
        inner=grown.inner.renumbered(renumber),
        walls=walls.renumbered(renumber),
        sides=sides,
        side_face=side_face,
        columns=renumber[grown.columns],
        fanned=np.isin(fan.point[grown.columns[:, 0]], fan.split),
    )


def _without_repeats(rings: np.ndarray, face: np.ndarray) -> tuple[Polygons, np.ndarray]:
    """Faces given as padded rings, ``(n, 4)``, each with a corner repeated round it dropped: a
    quadrilateral with two corners at one point a triangle, a face left with fewer than three
    corners gone. Returns the faces and ``face`` in their order (the triangles first)."""
    kept = []
    for ring in rings.tolist():
        ring = [c for c in ring if c >= 0]
        ring = [c for i, c in enumerate(ring) if c != ring[i - 1]]
        kept.append(ring)
    size = np.array([len(r) for r in kept])
    triangles = np.reshape([r for r in kept if len(r) == 3], (-1, 3)).astype(np.int64)
    quads = np.reshape([r for r in kept if len(r) == 4], (-1, 4)).astype(np.int64)
    return Polygons(triangles, quads), np.concatenate([face[size == 3], face[size == 4]])
