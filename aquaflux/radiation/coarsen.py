"""Coarsening a dense triangulated surface to the resolution the gather needs.

The gather's cost is receivers times facets, so a surface's facet count is its price. A surface
taken from a mesh's boundary carries the mesh's resolution rather than one chosen for radiation:
a mesher refines the cells beside a lamp for the flow there, and every one of those cells puts a
face on the lamp. Refinement (``subdivide.py``) goes the other way and cannot help.

**Edge collapse, with every surviving vertex an input vertex.** An edge ``(a, b)`` is removed by
moving ``a`` onto ``b``, so vertices are never placed anywhere new: every vertex of the result lies
on the input surface exactly, and no projection onto it is needed. Collapses are taken shortest
edge first and each is refused unless all of the following still hold:

- **size** -- no edge longer than ``max_edge``. A bound on the chord alone is not enough: along a
  straight cylinder's axis the surface does not curve, so a chord-only simplification runs slivers
  down the whole tube, and a facet carries one emission value and one absorption path along its
  entire length;
- **chord** -- every input vertex lies within ``chord`` of the facet it has been assigned to;
- **angle** -- every input facet's normal is within ``angle`` of the normal of the facet it has
  been assigned to, which is also what stops a collapse from folding a facet over;
- **shape** -- no facet falls below a minimum shape quality;
- **topology** -- the link condition, so the surface stays a manifold with the same holes, and
  **features** stay put: an edge on an open rim, between two bodies, or across a crease sharper
  than ``angle`` may only shorten along itself, and a vertex where features meet, or where a
  feature line turns by more than ``angle``, never moves.

Collapses alternate with edge flips that improve shape under the same bounds, until neither
changes anything.

**What the chord bound measures.** The distance is taken at the *input vertices*, not over the
continuous input surface, so it is exact to within the input's own resolution. On a snapped patch
of millimetre faces that difference is far below any chord worth asking for.

**Area, and power.** A coarse facet spans a chord of the input, so each body's area falls a
little. :func:`coarsen_surfaces` rescales each body's exitance by the area ratio so the body
radiates exactly the power it radiated before -- the same reasoning that makes
:func:`~aquaflux.radiation.units.lamp_exitance` divide by the triangulation's own area.
"""

from __future__ import annotations

import dataclasses
import heapq

import numpy as np

from aquaflux.radiation.checks import _merge_vertices
from aquaflux.radiation.surfaces import Surfaces

__all__ = ["Coarsening", "coarsen_surfaces", "coarsen_to_size"]

#: Facets whose shape quality ``4 sqrt(3) A / sum(l^2)`` (1 for equilateral) falls below this are
#: refused, unless they were already that poor. 0.3 admits a right isosceles triangle (0.87) and a
#: 1:4 rectangle's half (0.41), and refuses the slivers a greedy collapse otherwise leaves.
MIN_QUALITY = 0.3

#: Rounds of collapse-then-flip, each run to completion, before giving up on a fixed point.
MAX_ROUNDS = 8


@dataclasses.dataclass(frozen=True)
class Coarsening:
    """The coarsened triangles, and what the coarsening achieved.

    Attributes
    ----------
    vertices : np.ndarray, shape ``(n_facets, 3, 3)``
        The coarse triangles, wound as the input was.
    solid_id : np.ndarray of int, shape ``(n_facets,)``
        Body of each coarse triangle; a coarse triangle never spans two bodies.
    longest_edge : np.ndarray, shape ``(n_facets,)``
        Realized longest edge of each coarse triangle.
    chord : np.ndarray, shape ``(n_facets,)``
        Realized chord of each coarse triangle: the farthest any vertex of the input facets it
        replaces lies from the coarse surface, measured when those facets were last reassigned.
    angle : np.ndarray, shape ``(n_facets,)``
        Realized angle, in radians, between each coarse triangle's normal and the farthest-turned
        normal among the input facets it replaces.
    area_before, area_after : np.ndarray, shape ``(n_bodies,)``
        Each body's area, indexed by ``solid_id``, before and after.
    n_input : int
        Number of input triangles.
    """

    vertices: np.ndarray
    solid_id: np.ndarray
    longest_edge: np.ndarray
    chord: np.ndarray
    angle: np.ndarray
    area_before: np.ndarray
    area_after: np.ndarray
    n_input: int

    @property
    def n_facets(self) -> int:
        """Number of coarse triangles."""
        return int(self.vertices.shape[0])


# The arithmetic below runs on arrays of a few dozen triangles, many thousands of times over, so it
# is written out by component: at that size `np.cross` and `np.linalg.norm` spend most of their time
# normalizing axes rather than computing.


def _cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1],
            u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2],
            u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0],
        ),
        axis=-1,
    )


def _dot(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u[..., 0] * v[..., 0] + u[..., 1] * v[..., 1] + u[..., 2] * v[..., 2]


def _edges(corners: np.ndarray) -> np.ndarray:
    """The three edge vectors of triangles ``(..., 3, 3)``, each from a corner to the next."""
    return corners[..., (1, 2, 0), :] - corners


def _normal(corners: np.ndarray) -> np.ndarray:
    """Twice the vector area of triangles ``(..., 3, 3)``, by their winding."""
    return _cross(corners[..., 1, :] - corners[..., 0, :], corners[..., 2, :] - corners[..., 0, :])


def _quality(corners: np.ndarray) -> np.ndarray:
    """Shape quality ``4 sqrt(3) A / sum(l^2)`` of triangles ``(..., 3, 3)``; 1 is equilateral."""
    edges = _edges(corners)
    normal = _normal(corners)
    twice_area = np.sqrt(_dot(normal, normal))
    squares = _dot(edges, edges).sum(axis=-1)
    return 2.0 * np.sqrt(3.0) * twice_area / np.where(squares > 0.0, squares, 1.0)


def _point_triangle_distance(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Distance from each point to each triangle, shape ``(n_points, n_triangles)``.

    The plane distance where the point projects inside the triangle, and otherwise the nearest of
    its three edges.
    """
    p = points[:, None, :]
    a, b, c = (corners[None, :, k, :] for k in range(3))
    normal = _cross(b - a, c - a)
    length = np.sqrt(_dot(normal, normal))
    unit = normal / np.where(length > 0.0, length, 1.0)[..., None]
    height = _dot(p - a, unit)
    foot = p - height[..., None] * unit
    inside = np.broadcast_to(length > 0.0, height.shape).copy()
    nearest_edge = np.full(height.shape, np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        along = end - start
        inside &= _dot(_cross(along, foot - start), normal) >= 0.0
        span = _dot(along, along)
        t = np.clip(_dot(p - start, along) / np.where(span > 0, span, 1.0), 0.0, 1.0)
        gap = p - (start + t[..., None] * along)
        nearest_edge = np.minimum(nearest_edge, _dot(gap, gap))
    return np.where(inside, np.abs(height), np.sqrt(nearest_edge))


class _Decimator:
    """Greedy half-edge collapse and edge flips over an indexed triangle surface.

    Holds the evolving surface as vertex-indexed triangles with their incidence, the feature
    lines, and, for every live triangle, the input facets it currently stands for.
    """

    def __init__(self, vertices, solid_id, *, max_edge, chord, angle, tolerance):
        labels, n_vertices = _merge_vertices(vertices, tolerance)
        if np.any(
            (labels[:, 0] == labels[:, 1])
            | (labels[:, 1] == labels[:, 2])
            | (labels[:, 2] == labels[:, 0])
        ):
            msg = "a triangle has two coincident corners; coarsening needs areal facets only"
            raise ValueError(msg)
        self.position = np.zeros((n_vertices, 3))
        self.position[labels.reshape(-1)] = vertices.reshape(-1, 3)
        self.max_edge, self.chord, self.cos_angle = max_edge, chord, np.cos(angle)

        # The input, fixed: what every coarse facet is judged against.
        normal = _normal(vertices)
        self.input_normal = normal / np.sqrt(_dot(normal, normal))[:, None]
        self.input_centroid = vertices.mean(axis=1)
        self.input_labels = labels
        self.input_solid = solid_id

        self.triangles = labels.copy()
        self.solid = solid_id.copy()
        self.alive = np.ones(len(labels), dtype=bool)
        self.covers: list[np.ndarray] = [np.array([k]) for k in range(len(labels))]
        self.deviation = np.zeros(len(labels))
        self.incident: list[set[int]] = [set() for _ in range(n_vertices)]
        for index, corners in enumerate(labels):
            for vertex in corners:
                self.incident[vertex].add(index)
        self._find_features(angle)

    # -- topology ------------------------------------------------------------------------------

    def _neighbours(self, vertex: int) -> set[int]:
        found = set(self.triangles[list(self.incident[vertex])].ravel().tolist())
        found.discard(vertex)
        return found

    def _sharing(self, a: int, b: int) -> list[int]:
        return sorted(self.incident[a] & self.incident[b])

    def _find_features(self, angle: float) -> None:
        """Mark rims, body interfaces, non-manifold edges and creases sharper than ``angle``."""
        uses: dict[tuple[int, int], list[int]] = {}
        for index, corners in enumerate(self.triangles):
            for k in range(3):
                a, b = int(corners[k]), int(corners[(k + 1) % 3])
                uses.setdefault((min(a, b), max(a, b)), []).append(index)
        self.feature_neighbours: list[set[int]] = [set() for _ in range(len(self.position))]
        self.pinned = np.zeros(len(self.position), dtype=bool)
        cos_angle = np.cos(angle)
        for (a, b), users in uses.items():
            if len(users) == 2:
                first, second = users
                if (
                    self.solid[first] == self.solid[second]
                    and self.input_normal[first] @ self.input_normal[second] >= cos_angle
                ):
                    continue
            elif len(users) > 2:
                self.pinned[[a, b]] = True
            self.feature_neighbours[a].add(b)
            self.feature_neighbours[b].add(a)
        # A vertex on a feature line may only slide along it, and only where the line runs
        # straight through it: at a corner of the line -- a rectangle's corner, the end of a
        # crease -- sliding would cut the corner off.
        for vertex, line in enumerate(self.feature_neighbours):
            if len(line) == 2:
                before, after = (self.position[v] for v in line)
                incoming = self.position[vertex] - before
                outgoing = after - self.position[vertex]
                turn = _dot(incoming, outgoing) / np.sqrt(
                    _dot(incoming, incoming) * _dot(outgoing, outgoing)
                )
                if turn < cos_angle:
                    self.pinned[vertex] = True

    # -- the quality bounds --------------------------------------------------------------------

    def _reassign(self, changing: list[int], corners: np.ndarray, removed: list[int]):
        """Redistribute the input facets of a changing patch of triangles, or fail.

        ``changing`` are the triangles an operation rewrites, with their new ``corners``, and
        ``removed`` those it deletes; their input facets are handed out again among the changing
        triangles and the unchanged **ring** around them. The ring matters: an input facet near
        the patch's edge can overhang it, and measured against the patch alone it would read as
        lying off the surface when it lies flat on the triangle next door.

        The chord is judged per input **vertex**, against the nearest candidate: the coarse
        surface is the union of the triangles, and a small input facet straddling two of them lies
        on that surface even though it lies on neither alone. (Any body's candidate will do for a
        distance, since two bodies meet along a kept line.) The angle is judged per input facet,
        against the candidate of its own body its centroid is nearest, which is also the triangle
        the facet is then assigned to.

        Returns
        -------
        dict or None
            The new cover of every candidate that gains or changes one, or ``None`` if a vertex
            would lie beyond the chord or a facet's nearest candidate turns further than the
            angle. The deviations are recorded only on success.
        """
        # The ring is found through the vertices of the patch as it is now, which include every
        # vertex it will have: an operation here only ever drops a vertex or rewires the patch.
        vertices = {
            int(v) for index in list(changing) + list(removed) for v in self.triangles[index]
        }
        excluded = set(changing) | set(removed)
        ring = sorted({t for v in vertices for t in self.incident[v]} - excluded)
        candidates = list(changing) + ring
        all_corners = np.concatenate([corners, self.position[self.triangles[ring]]], axis=0)

        normal = _normal(all_corners)
        length = np.sqrt(_dot(normal, normal))
        if np.any(length[: len(changing)] <= 0.0):
            return None
        covered = np.concatenate([self.covers[index] for index in list(changing) + list(removed)])
        # Each input vertex once, however many covered facets share it.
        vertex, of_corner = np.unique(self.input_labels[covered], return_inverse=True)
        near = _point_triangle_distance(self.position[vertex], all_corners).min(axis=1)
        deviation = near[of_corner.reshape(-1, 3)].max(axis=1)
        if np.any(deviation > self.chord):
            return None
        own_body = self.input_solid[covered][:, None] == self.solid[candidates][None, :]
        centroid = np.where(
            own_body, _point_triangle_distance(self.input_centroid[covered], all_corners), np.inf
        )
        best = np.argmin(centroid, axis=1)
        unit = normal[best] / np.where(length[best] > 0.0, length[best], 1.0)[:, None]
        if np.any(_dot(self.input_normal[covered], unit) < self.cos_angle):
            return None
        self.deviation[covered] = deviation
        covers = {index: covered[best == k] for k, index in enumerate(changing)}
        for k, index in enumerate(ring, start=len(changing)):
            gained = covered[best == k]
            if gained.size:
                covers[index] = np.concatenate([self.covers[index], gained])
        return covers

    def _acceptable_shape(self, new: np.ndarray, old: np.ndarray) -> bool:
        edges = _edges(new)
        if np.any(_dot(edges, edges) > self.max_edge**2):
            return False
        floor = min(MIN_QUALITY, float(_quality(old).min()))
        return bool(np.all(_quality(new) >= floor))

    # -- collapse ------------------------------------------------------------------------------

    def _may_move(self, a: int, b: int) -> bool:
        if self.pinned[a]:
            return False
        degree = len(self.feature_neighbours[a])
        if degree == 0:
            return True
        return degree == 2 and b in self.feature_neighbours[a]

    def _collapse(self, a: int, b: int) -> bool:
        """Move vertex ``a`` onto ``b`` if every bound survives; report whether it did."""
        if not self._may_move(a, b):
            return False
        shared = self._sharing(a, b)
        if not shared:
            return False
        opposite = {int(v) for index in shared for v in self.triangles[index]} - {a, b}
        if self._neighbours(a) & self._neighbours(b) != opposite:
            return False  # the link condition: the collapse would pinch the surface

        moved = [index for index in self.incident[a] if index not in shared]
        region = moved + [index for index in self.incident[b] if index not in shared]
        old = self.position[self.triangles[region]]
        rewritten = self.triangles[region].copy()
        rewritten[rewritten == a] = b
        new = self.position[rewritten]
        if not region:
            return False  # nothing would remain to stand for the facets this removes
        if moved and not self._acceptable_shape(new[: len(moved)], old[: len(moved)]):
            return False
        # A facet may not turn over: its new normal stays on the side of its old one.
        if np.any(np.sum(_normal(new) * _normal(old), axis=-1) <= 0.0):
            return False
        covers = self._reassign(region, new, shared)
        if covers is None:
            return False

        for index in shared:
            self.alive[index] = False
            for vertex in self.triangles[index]:
                self.incident[vertex].discard(index)
        for index, corners in zip(region, rewritten, strict=True):
            self.triangles[index] = corners
            self.incident[b].add(index)
        for index, cover in covers.items():
            self.covers[index] = cover
        self.incident[a] = set()
        if b in self.feature_neighbours[a]:
            for c in self.feature_neighbours[a] - {b}:
                self.feature_neighbours[c].discard(a)
                self.feature_neighbours[c].add(b)
                self.feature_neighbours[b].add(c)
            self.feature_neighbours[b].discard(a)
            self.feature_neighbours[a] = set()
        return True

    def collapse_all(self) -> int:
        """Collapse edges shortest first until none can be; return how many were."""
        heap = []
        for a in range(len(self.position)):
            for b in self._neighbours(a):
                if a < b:
                    length = float(np.linalg.norm(self.position[a] - self.position[b]))
                    if length < self.max_edge:
                        heap.append((length, a, b))
        heapq.heapify(heap)
        done = 0
        while heap:
            _, a, b = heapq.heappop(heap)
            if not self.incident[a] or not self.incident[b] or not self._sharing(a, b):
                continue
            # The vertex with fewer features is the one to move; either way round is tried.
            order = (
                (a, b)
                if len(self.feature_neighbours[a]) <= len(self.feature_neighbours[b])
                else (b, a)
            )
            for source, target in (order, order[::-1]):
                if self._collapse(source, target):
                    done += 1
                    for c in self._neighbours(target):
                        length = float(np.linalg.norm(self.position[target] - self.position[c]))
                        if length < self.max_edge:
                            heapq.heappush(heap, (length, min(target, c), max(target, c)))
                    break
        return done

    # -- flips ---------------------------------------------------------------------------------

    def _flip(self, first: int, second: int, a: int, b: int) -> bool:
        """Swap the diagonal ``(a, b)`` of two triangles for the other one, if it helps."""
        if self.solid[first] != self.solid[second] or b in self.feature_neighbours[a]:
            return False
        p = next(int(v) for v in self.triangles[first] if v not in (a, b))
        q = next(int(v) for v in self.triangles[second] if v not in (a, b))
        if p == q or q in self._neighbours(p):
            return False
        # Keep the winding: in `first` the edge runs a -> b, so the new pair runs p -> q.
        corners = list(self.triangles[first])
        k = corners.index(a)
        if corners[(k + 1) % 3] != b:
            a, b = b, a
        new_labels = np.array([[a, q, p], [q, b, p]])
        old = self.position[self.triangles[[first, second]]]
        new = self.position[new_labels]
        if _quality(new).min() <= _quality(old).min() + 1e-9:
            return False
        if not self._acceptable_shape(new, old):
            return False
        if np.any(_normal(new) @ _normal(old).sum(axis=0) <= 0.0):
            return False
        covers = self._reassign([first, second], new, [])
        if covers is None:
            return False
        for index, before, after in zip(
            (first, second), self.triangles[[first, second]], new_labels, strict=True
        ):
            for vertex in before:
                self.incident[vertex].discard(index)
            for vertex in after:
                self.incident[vertex].add(index)
            self.triangles[index] = after
        for index, cover in covers.items():
            self.covers[index] = cover
        return True

    def flip_all(self) -> int:
        """One sweep of improving flips over every interior edge; return how many were made."""
        done = 0
        for index in np.flatnonzero(self.alive):
            for k in range(3):
                a, b = int(self.triangles[index][k]), int(self.triangles[index][(k + 1) % 3])
                if a > b:
                    continue
                shared = self._sharing(a, b)
                if len(shared) == 2 and index in shared and self._flip(*shared, a, b):
                    done += 1
                    break
        return done

    # -- the result ----------------------------------------------------------------------------

    def result(self, n_bodies: int, area_before: np.ndarray) -> Coarsening:
        live = np.flatnonzero(self.alive)
        corners = self.position[self.triangles[live]]
        normal = _normal(corners)
        unit = normal / np.linalg.norm(normal, axis=1)[:, None]
        chord = np.zeros(len(live))
        angle = np.zeros(len(live))
        for row, index in enumerate(live):
            covered = self.covers[index]
            if covered.size:
                chord[row] = self.deviation[covered].max()
                cosine = np.clip(self.input_normal[covered] @ unit[row], -1.0, 1.0)
                angle[row] = float(np.arccos(cosine.min()))
        area = 0.5 * np.linalg.norm(normal, axis=1)
        return Coarsening(
            vertices=corners,
            solid_id=self.solid[live],
            longest_edge=np.linalg.norm(np.roll(corners, -1, axis=1) - corners, axis=2).max(axis=1),
            chord=chord,
            angle=angle,
            area_before=area_before,
            area_after=np.bincount(self.solid[live], weights=area, minlength=n_bodies),
            n_input=len(self.alive),
        )


def coarsen_to_size(
    vertices,
    *,
    max_edge: float,
    chord: float,
    angle: float = 0.5,
    solid_id=None,
    tolerance: float | None = None,
) -> Coarsening:
    """Coarsen a triangulated surface as far as the size, chord and angle bounds allow.

    Parameters
    ----------
    vertices : array_like, shape ``(n_facets, 3, 3)``
        The input triangles, consistently wound. Shared corners are matched by position.
    max_edge : float
        The longest edge a coarse triangle may have, in metres.
    chord : float
        The farthest an input vertex may lie from the coarse triangle that replaces it, in metres.
    angle : float, optional
        The largest angle, in radians, between a coarse triangle's normal and the normal of any
        input facet it replaces; also the dihedral angle above which an edge is a crease that is
        kept.
    solid_id : array_like of int, shape ``(n_facets,)``, optional
        Body of each triangle. The line between two bodies is kept, and no coarse triangle spans
        two. Defaults to one body.
    tolerance : float, optional
        Distance within which two corners are the same vertex; defaults to a billionth of the
        surface's extent.

    Returns
    -------
    Coarsening

    Raises
    ------
    ValueError
        If a bound is not positive, the input is not ``(n_facets, 3, 3)``, or a triangle has two
        coincident corners.
    """
    for name, value in (("max_edge", max_edge), ("chord", chord), ("angle", angle)):
        if not value > 0.0:
            raise ValueError(f"{name} must be positive; got {value}")
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 3 or vertices.shape[1:] != (3, 3):
        raise ValueError(f"vertices must have shape (n_facets, 3, 3); got {vertices.shape}")
    solid_id = (
        np.zeros(len(vertices), dtype=np.int64)
        if solid_id is None
        else np.asarray(solid_id, dtype=np.int64)
    )
    if solid_id.shape != (len(vertices),):
        raise ValueError(f"solid_id must have shape ({len(vertices)},); got {solid_id.shape}")
    n_bodies = int(solid_id.max(initial=-1)) + 1
    area = 0.5 * np.linalg.norm(
        _cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=1
    )
    decimator = _Decimator(
        vertices, solid_id, max_edge=max_edge, chord=chord, angle=angle, tolerance=tolerance
    )
    for _ in range(MAX_ROUNDS):
        if decimator.collapse_all() + decimator.flip_all() == 0:
            break
    return decimator.result(n_bodies, np.bincount(solid_id, weights=area, minlength=n_bodies))


def _per_body(values: np.ndarray, solid_id: np.ndarray, n_bodies: int, name: str) -> np.ndarray:
    """One value per body, refusing a body whose facets disagree."""
    per_body = np.zeros(n_bodies, dtype=values.dtype)
    for body in range(n_bodies):
        mine = values[solid_id == body]
        if mine.size and np.any(mine != mine[0]):
            raise ValueError(
                f"{name} varies across body {body}; a coarse facet replaces many input facets, so "
                "only a value uniform over each body can be carried across"
            )
        if mine.size:
            per_body[body] = mine[0]
    return per_body


def coarsen_surfaces(
    surfaces: Surfaces,
    *,
    max_edge: float,
    chord: float,
    angle: float = 0.5,
) -> tuple[Surfaces, Coarsening]:
    """Coarsen a surface set, carrying its optics and keeping each body's emitted power.

    Reflectance and angular profile are carried unchanged, being intensive. Exitance is scaled,
    per body, by the ratio of the body's area before to after, so each body emits the same power
    over its coarse area as over its original one.

    Parameters
    ----------
    surfaces : Surfaces
        The set to coarsen. Its optics must be uniform over each body.
    max_edge, chord, angle : float
        As in :func:`coarsen_to_size`.

    Returns
    -------
    tuple of (Surfaces, Coarsening)

    Raises
    ------
    ValueError
        If the set holds point sources, or an optical property varies within a body.
    """
    if surfaces.point_source_index:
        raise ValueError(
            "coarsen_surfaces cannot carry point sources: they have no area to coarsen. "
            "Coarsen the areal facets, then add the point sources."
        )
    solid_id = np.asarray(surfaces.solid_id)
    n_bodies = len(surfaces.solid_names)
    emission = _per_body(np.asarray(surfaces.emission), solid_id, n_bodies, "emission")
    reflectance = _per_body(np.asarray(surfaces.reflectance), solid_id, n_bodies, "reflectance")
    profile = _per_body(np.asarray(surfaces.profile_index), solid_id, n_bodies, "profile")

    coarsening = coarsen_to_size(
        np.asarray(surfaces.vertices),
        max_edge=max_edge,
        chord=chord,
        angle=angle,
        solid_id=solid_id,
    )
    before = np.zeros(n_bodies)
    after = np.ones(n_bodies)
    present = len(coarsening.area_before)
    before[:present] = coarsening.area_before
    after[:present] = np.where(coarsening.area_after > 0.0, coarsening.area_after, 1.0)
    exitance = emission * before / after
    body = coarsening.solid_id
    coarse = Surfaces.from_triangles(
        coarsening.vertices,
        solid_id=body,
        solid_names=surfaces.solid_names,
        emission=exitance[body],
        reflectance=reflectance[body],
        profiles=surfaces.profiles,
        profile_index=profile[body],
    )
    return coarse, coarsening
