"""Coarsening a dense triangulated surface to the resolution the gather needs.

The gather's cost is receivers times facets, so a surface's facet count is its price. A surface
taken from a mesh's boundary carries the mesh's resolution rather than one chosen for radiation:
a mesher refines the cells beside a lamp for the flow there, and every one of those cells puts a
face on the lamp. Refinement (``subdivide.py``) goes the other way and cannot help.

**Edge collapse, with every surviving vertex an input vertex.** An edge ``(a, b)`` is removed by
moving ``a`` onto ``b``, so vertices are never placed anywhere new: every vertex of the result lies
on the input surface exactly, and no projection onto it is needed. A collapse is refused unless all
of the following still hold:

- **size** -- no edge longer than ``max_edge``. A bound on the chord alone is not enough: along a
  straight cylinder's axis the surface does not curve, so a chord-only simplification runs slivers
  down the whole tube, and a facet carries one emission value and one absorption path along its
  entire length;
- **chord** -- every input vertex lies within ``chord`` of the coarse surface;
- **angle** -- every input facet's normal is within ``angle`` of the normal of the facet it has
  been assigned to, which is also what stops a collapse from folding a facet over;
- **shape** -- no facet falls below a minimum shape quality;
- **topology** -- the link condition, so the surface stays a manifold with the same holes, and
  **features** stay put: an edge on an open rim, between two bodies, or across a crease sharper
  than ``angle`` may only shorten along itself, and a vertex where features meet, or where a
  feature line turns by more than ``angle``, never moves.

Collapses alternate with edge flips that improve shape under the same bounds, until neither
changes anything.

**In batches, not one at a time.** Every check reads only the triangles around an operation and
the ring of triangles around those, so two operations three edges apart or more cannot affect each
other. Each sweep takes every candidate that is best within two edges of it -- the shortest edge,
to within a band of lengths, with ties broken at random so that a sweep takes candidates
everywhere at once rather than along a front -- checks all of them in one vectorized pass, and
applies all that pass. The point-to-triangle distance at the heart of the chord and angle checks
runs compiled, since a sweep measures millions of pairs. The random tie-break is seeded, so a
coarsening is reproducible.

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

import jax
import jax.numpy as jnp
import numpy as np

from aquaflux import ragged
from aquaflux.radiation.checks import _merge_vertices
from aquaflux.radiation.surfaces import Surfaces

__all__ = ["Coarsening", "coarsen_surfaces", "coarsen_to_size"]

#: Facets whose shape quality ``4 sqrt(3) A / sum(l^2)`` (1 for equilateral) falls below this are
#: refused, unless they were already that poor. 0.3 admits a right isosceles triangle (0.87) and a
#: 1:4 rectangle's half (0.41), and refuses the slivers a greedy collapse otherwise leaves.
MIN_QUALITY = 0.3

#: Rounds of collapse-then-flip, each run to completion, before giving up on a fixed point.
MAX_ROUNDS = 8

#: Point-to-triangle pairs measured at once when a batch is checked. It bounds the memory of one
#: check (a few hundred bytes a pair across the temporaries), not its answer.
PAIR_LIMIT = 2_000_000

#: Width of the priority bands a sweep orders its candidates by, as a fraction of ``max_edge`` for
#: a collapse's length and of quality for a flip's gain; within a band the order is random.
PRIORITY_BAND = 0.05

#: A collapse sweep applying fewer than this fraction of the candidates it chose makes the next
#: sweep check every remaining candidate at once (see ``_Decimator._collapse_sweep``).
SCREEN_BELOW = 0.25


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


def _cross(u: np.ndarray, v: np.ndarray, xp=np) -> np.ndarray:
    return xp.stack(
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


def _point_triangle_distance(points: np.ndarray, corners: np.ndarray, xp=np) -> np.ndarray:
    """Distance from points ``(..., 3)`` to triangles ``(..., 3, 3)``, the two broadcast together.

    The plane distance where the point projects inside the triangle, and otherwise the nearest of
    its three edges. Pass ``points[:, None]`` and ``corners[None]`` for every pairing, or equal
    leading shapes for one triangle per point. ``xp`` is the array namespace, so that the same
    formula also runs compiled (:func:`_pair_distance`).
    """
    a, b, c = corners[..., 0, :], corners[..., 1, :], corners[..., 2, :]
    normal = _cross(b - a, c - a, xp)
    length = xp.sqrt(_dot(normal, normal))
    unit = normal / xp.where(length > 0.0, length, 1.0)[..., None]
    height = _dot(points - a, unit)
    foot = points - height[..., None] * unit
    inside = xp.broadcast_to(length > 0.0, height.shape)
    nearest = xp.full(height.shape, xp.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        along = end - start
        inside = inside & (_dot(_cross(along, foot - start, xp), normal) >= 0.0)
        span = _dot(along, along)
        t = xp.clip(_dot(points - start, along) / xp.where(span > 0.0, span, 1.0), 0.0, 1.0)
        gap = points - (start + t[..., None] * along)
        nearest = xp.minimum(nearest, _dot(gap, gap))
    return xp.where(inside, xp.abs(height), xp.sqrt(nearest))


@jax.jit
def _compiled_pair_distance(points, position, labels, point, candidate, keep):
    """Distance from ``points[point]`` to the triangle ``position[labels[candidate]]``, per pair,
    and ``inf`` where ``keep`` is false; one fused, compiled pass."""
    distance = _point_triangle_distance(points[point], position[labels[candidate]], jnp)
    return jnp.where(keep, distance, jnp.inf)


def _bucket(n: int) -> int:
    """The power of two a length is padded to, so a compiled program is reused across sizes."""
    return max(1024, 1 << max(n - 1, 0).bit_length())


def _padded(values: np.ndarray, size: int, fill=0) -> np.ndarray:
    out = np.full((size, *values.shape[1:]), fill, dtype=values.dtype)
    out[: len(values)] = values
    return out


def _unique_pairs(first: np.ndarray, second: np.ndarray, n_second: int):
    """The distinct ``(first, second)`` pairs, sorted by ``first`` then ``second``."""
    key = np.unique(first.astype(np.int64) * n_second + second)
    return key // n_second, key % n_second


def _count(groups: np.ndarray, n: int, where=None) -> np.ndarray:
    """How many items of each group there are (or satisfy ``where``), shape ``(n,)``."""
    weights = None if where is None else np.asarray(where, dtype=float)
    return np.bincount(groups, weights=weights, minlength=n).astype(np.int64)


def _group_ranges(weight: np.ndarray, limit: int):
    """Consecutive ranges of groups, each carrying at most ``limit`` total weight (or one group)."""
    cumulative = np.concatenate([[0], np.cumsum(weight)])
    first = 0
    while first < len(weight):
        stop = int(np.searchsorted(cumulative, cumulative[first] + limit, side="right")) - 1
        stop = max(stop, first + 1)
        yield first, stop
        first = stop


@dataclasses.dataclass(frozen=True)
class _Topology:
    """The live surface's incidence, rebuilt at the start of every sweep.

    Attributes
    ----------
    vertex_offsets, vertex_triangles : np.ndarray
        Compressed-sparse-row (CSR) list of the live triangles around each vertex.
    edge_key : np.ndarray, shape ``(n_edges,)``
        Sorted keys ``low * n_vertices + high`` of the live edges.
    edge_ends : np.ndarray, shape ``(n_edges, 2)``
        Each edge's two vertices, lower first.
    edge_uses : np.ndarray, shape ``(n_edges,)``
        How many live triangles use each edge.
    edge_first, edge_second : np.ndarray, shape ``(n_edges,)``
        The first two of them, ``-1`` where there is no second.
    """

    vertex_offsets: np.ndarray
    vertex_triangles: np.ndarray
    edge_key: np.ndarray
    edge_ends: np.ndarray
    edge_uses: np.ndarray
    edge_first: np.ndarray
    edge_second: np.ndarray


@dataclasses.dataclass(frozen=True)
class _Patches:
    """What a batch of checked operations would change, one group label per operation.

    Attributes
    ----------
    rewritten_group, rewritten, rewritten_labels : np.ndarray
        Triangles that take new corners, and the corners.
    removed_group, removed : np.ndarray
        Triangles that go.
    input_group, inputs, new_owner, deviation : np.ndarray
        Input facets handed to a new triangle, and their farthest vertex from the new surface.
    """

    rewritten_group: np.ndarray
    rewritten: np.ndarray
    rewritten_labels: np.ndarray
    removed_group: np.ndarray
    removed: np.ndarray
    input_group: np.ndarray
    inputs: np.ndarray
    new_owner: np.ndarray
    deviation: np.ndarray

    def accepted(self, accept: np.ndarray) -> _Patches:
        """Only the operations ``accept`` marks."""
        return _Patches(
            **{
                name: getattr(self, name)[accept[getattr(self, group)]]
                for name, group in (
                    ("rewritten_group", "rewritten_group"),
                    ("rewritten", "rewritten_group"),
                    ("rewritten_labels", "rewritten_group"),
                    ("removed_group", "removed_group"),
                    ("removed", "removed_group"),
                    ("input_group", "input_group"),
                    ("inputs", "input_group"),
                    ("new_owner", "input_group"),
                    ("deviation", "input_group"),
                )
            }
        )


class _Decimator:
    """Half-edge collapses and edge flips over an indexed triangle surface, in batches.

    The surface is vertex-indexed triangles over the fixed input vertex positions, a record of which
    coarse triangle stands for each input facet, and the feature lines. Work is done in sweeps: each
    sweep takes every candidate operation that has the best priority within two edges of it, checks
    all of them at once, and applies the ones that pass. Two operations chosen together are then
    three edges apart or more, so neither touches a triangle the other reads -- the triangles
    around an operation and the ring of triangles around those, against which its chord is
    measured -- and they can be checked and applied independently.
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
        # The points the compiled distance measures from -- every vertex, then every input
        # facet's centroid -- held on the device in one array, so they are not copied per call
        # and both uses share one compiled program.
        self.device_points = jnp.asarray(np.concatenate([self.position, self.input_centroid]))
        self.device_position = jnp.asarray(self.position)
        self.input_labels = labels
        self.input_solid = solid_id

        self.triangles = labels.copy()
        self.solid = solid_id.copy()
        self.alive = np.ones(len(labels), dtype=bool)
        self.owner = np.arange(len(labels))
        self.deviation = np.zeros(len(labels))
        # Edges an operation was refused on in this round.
        self.new_round()
        # Seeded, so a coarsening is reproducible.
        self.random = np.random.default_rng(0)
        self._find_features(angle)

    @property
    def n_vertices(self) -> int:
        return len(self.position)

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)

    # -- topology ------------------------------------------------------------------------------

    def _topology(self) -> _Topology:
        live = np.flatnonzero(self.alive)
        corners = self.triangles[live]
        vertex_offsets, order = ragged.group(corners.ravel(), self.n_vertices)
        start, end = corners.ravel(), corners[:, (1, 2, 0)].ravel()
        low, high = np.minimum(start, end), np.maximum(start, end)
        key, of_use, uses = np.unique(
            low * self.n_vertices + high, return_inverse=True, return_counts=True
        )
        by_edge = np.argsort(of_use, kind="stable")
        user = np.repeat(live, 3)[by_edge]
        first_use = np.cumsum(uses) - uses
        second = np.where(uses >= 2, user[np.minimum(first_use + 1, len(user) - 1)], -1)
        return _Topology(
            vertex_offsets=vertex_offsets,
            vertex_triangles=live[order // 3],
            edge_key=key,
            edge_ends=np.stack([key // self.n_vertices, key % self.n_vertices], axis=1),
            edge_uses=uses,
            edge_first=user[first_use],
            edge_second=second,
        )

    def _pair_distance(self, point, labels, candidate, keep) -> np.ndarray:
        """Point-to-triangle distance per pair, compiled; ``inf`` where ``keep`` is false.

        A batched check measures millions of pairs at a time, where numpy's pass per operation
        over memory costs several times the arithmetic. Compiled, the formula is one fused pass,
        so every pair is measured exactly and none needs pruning first. Pairs and candidates are
        padded to powers of two, so the program is compiled for a few sizes only.
        """
        pairs, triangles = _bucket(len(point)), _bucket(len(labels))
        distance = _compiled_pair_distance(
            self.device_points,
            self.device_position,
            _padded(labels, triangles),
            _padded(point, pairs),
            _padded(candidate, pairs),
            _padded(keep, pairs, fill=False),
        )
        return np.asarray(distance)[: len(point)]

    def _inputs_of(self, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The input facets each of ``triangles`` stands for, and which entry each belongs to.

        Found by a mask over the owners rather than by grouping all of them, which would sort every
        input facet on every sweep to read the few a batch touches.
        """
        wanted, compact = np.unique(triangles, return_inverse=True)
        slot = np.full(self.n_triangles, -1)
        slot[wanted] = np.arange(len(wanted))
        mine = np.flatnonzero(slot[self.owner] >= 0)
        offsets, order = ragged.group(slot[self.owner[mine]], len(wanted))
        inputs, which, _ = ragged.rows(offsets, mine[order], compact)
        return inputs, which

    def _find_features(self, angle: float) -> None:
        """Mark rims, body interfaces, non-manifold edges and creases sharper than ``angle``."""
        topology = self._topology()
        first, second = topology.edge_first, topology.edge_second
        pair = np.maximum(second, 0)
        crease = (self.solid[first] != self.solid[pair]) | (
            _dot(self.input_normal[first], self.input_normal[pair]) < np.cos(angle)
        )
        feature = (topology.edge_uses != 2) | crease
        self.features = topology.edge_ends[feature]
        self.pinned = np.zeros(self.n_vertices, dtype=bool)
        self.pinned[topology.edge_ends[topology.edge_uses > 2].ravel()] = True
        # A vertex on a feature line may only slide along it, and only where the line runs
        # straight through it: at a corner of the line -- a rectangle's corner, the end of a
        # crease -- sliding would cut the corner off.
        ends = np.concatenate([self.features, self.features[:, ::-1]])
        offsets, order = ragged.group(ends[:, 0], self.n_vertices)
        on_a_line = np.flatnonzero(np.diff(offsets) == 2)
        neighbours = ends[order[offsets[on_a_line][:, None] + np.arange(2)], 1]
        incoming = self.position[on_a_line] - self.position[neighbours[:, 0]]
        outgoing = self.position[neighbours[:, 1]] - self.position[on_a_line]
        turn = _dot(incoming, outgoing) / np.sqrt(
            _dot(incoming, incoming) * _dot(outgoing, outgoing)
        )
        self.pinned[on_a_line[turn < np.cos(angle)]] = True

    def _feature_state(self, topology: _Topology):
        """Each vertex's number of feature edges, and which live edges are features."""
        degree = np.bincount(self.features.ravel(), minlength=self.n_vertices)
        key = self.features[:, 0] * self.n_vertices + self.features[:, 1]
        return degree, np.isin(topology.edge_key, key)

    def _priority(self, band: np.ndarray) -> np.ndarray:
        """Distinct priorities: ``band`` first, then a random order within each band.

        A strict order by length is not what a batch wants. Most edges of a snapped patch are the
        same length, ties fall to index order, and a candidate is then a local minimum only at
        the leading edge of that order -- a sweep advances as a thin front, a percent of the
        surface at a time. Ordering by band and at random within one keeps the pass shortest first
        to within a band while letting a sweep take candidates everywhere at once, which is the
        randomized choice of Luby's independent-set algorithm.
        """
        return band + self.random.random(len(band))

    def _local_minima(self, ends: np.ndarray, priority: np.ndarray, topology: _Topology):
        """Which candidates have the smallest priority within two edges of their vertices."""
        best = np.full(self.n_vertices, np.inf)
        np.minimum.at(best, ends.ravel(), np.repeat(priority, ends.shape[1]))
        low, high = topology.edge_ends.T
        for _ in range(2):
            spread = best.copy()
            np.minimum.at(spread, low, best[high])
            np.minimum.at(spread, high, best[low])
            best = spread
        return np.all(best[ends] >= priority[:, None], axis=1)

    # -- the quality bounds, for a batch of patches --------------------------------------------

    def _check_patches(self, group, changing, labels, removed_group, removed, n, topology):
        """Check a batch of patch replacements against the chord and the angle.

        Operation ``g`` rewrites the triangles ``changing[group == g]`` to have corners
        ``labels[group == g]`` and deletes ``removed[removed_group == g]``. The input facets
        those triangles stood for are handed out again among the rewritten triangles and the
        unchanged **ring** around them: an input facet near a patch's edge can overhang it, and
        measured against the patch alone it would read as lying off the surface when it lies flat
        on the triangle next door.

        The chord is judged per input **vertex**, against the nearest candidate: the coarse
        surface is the union of the triangles, and a small input facet straddling two of them
        lies on that surface even though it lies on neither alone. (Any body's candidate will do
        for a distance, since two bodies meet along a kept line.) The angle is judged per input
        facet, against the candidate of its own body its centroid is nearest, which is also the
        triangle the facet is then assigned to.

        Returns
        -------
        (ok, patches)
            Whether each of the ``n`` operations passes, and what each would change.
        """
        nv, nt = self.n_vertices, self.n_triangles
        ok = np.ones(n, dtype=bool)

        # The ring, found through the vertices of the patch as it is now, which include every
        # vertex it will have: an operation here only ever drops a vertex or rewires the patch.
        foot_group, foot = _unique_pairs(
            np.concatenate([np.repeat(group, 3), np.repeat(removed_group, 3)]),
            np.concatenate([self.triangles[changing].ravel(), self.triangles[removed].ravel()]),
            nv,
        )
        around, which, _ = ragged.rows(topology.vertex_offsets, topology.vertex_triangles, foot)
        ring_group, ring = _unique_pairs(foot_group[which], around, nt)
        inside = np.concatenate([group * nt + changing, removed_group * nt + removed])
        outside = ~np.isin(ring_group * nt + ring, inside)
        ring_group, ring = ring_group[outside], ring[outside]

        candidate_group = np.concatenate([group, ring_group])
        candidate = np.concatenate([changing, ring])
        corners = self.position[np.concatenate([labels, self.triangles[ring]])]
        normal = _normal(corners)
        length = np.sqrt(_dot(normal, normal))
        ok &= _count(group, n, length[: len(group)] <= 0.0) == 0

        inputs, which = self._inputs_of(np.concatenate([changing, removed]))
        input_group = np.concatenate([group, removed_group])[which]

        # The chord, at each input vertex once however many covered facets share it.
        key, of_corner = np.unique(
            np.repeat(input_group, 3) * nv + self.input_labels[inputs].ravel(),
            return_inverse=True,
        )
        vertex_group, vertex = key // nv, key % nv
        candidate_labels = np.concatenate([labels, self.triangles[ring]])

        def nearest(point, groups, allowed=None):
            """Each point's nearest candidate in its group: (distance, index), inf if none allowed.

            ``point`` indexes the decimator's device points (vertices, then input centroids).
            """
            distance = np.full(len(point), np.inf)
            index = np.zeros(len(point), dtype=np.int64)
            weight = _count(groups, n) * _count(candidate_group, n)
            for first, stop in _group_ranges(weight, PAIR_LIMIT):
                left, right = ragged.pairs_within_groups(groups, candidate_group, n, first, stop)
                if not left.size:
                    continue
                keep = np.ones(len(left), dtype=bool) if allowed is None else allowed(left, right)
                exact = self._pair_distance(point[left], candidate_labels, right, keep)
                # A group lies in one range, so each point's minimum is final within it.
                np.minimum.at(distance, left, exact)
                hit = exact <= distance[left]
                index[left[hit]] = right[hit]
            return distance, index

        # The chord, at each input vertex once however many covered facets share it.
        near, _ = nearest(vertex, vertex_group)
        deviation = near[of_corner.reshape(-1, 3)].max(axis=1)
        ok &= _count(input_group, n, deviation > self.chord) == 0

        # The angle, against the nearest candidate of the facet's own body.
        def own_body(left, right):
            return self.input_solid[inputs[left]] == self.solid[candidate[right]]

        found, best = nearest(self.n_vertices + inputs, input_group, own_body)
        found = np.isfinite(found)
        unit = normal[best] / np.where(length[best] > 0.0, length[best], 1.0)[:, None]
        turned = ~found | (_dot(self.input_normal[inputs], unit) < self.cos_angle)
        ok &= _count(input_group, n, turned) == 0

        return ok, _Patches(
            rewritten_group=group,
            rewritten=changing,
            rewritten_labels=labels,
            removed_group=removed_group,
            removed=removed,
            input_group=input_group,
            inputs=inputs,
            new_owner=candidate[best],
            deviation=deviation,
        )

    def _shape_refused(self, old: np.ndarray, new: np.ndarray, floor: np.ndarray) -> np.ndarray:
        """Rewritten triangles over the edge bound, below the quality floor, or turned over."""
        edges = _edges(new)
        too_long = np.any(_dot(edges, edges) > self.max_edge**2, axis=-1)
        turned_over = _dot(_normal(new), _normal(old)) <= 0.0
        return too_long | (_quality(new) < floor) | turned_over

    def _apply(self, patches: _Patches) -> None:
        self.alive[patches.removed] = False
        self.triangles[patches.rewritten] = patches.rewritten_labels
        self.owner[patches.inputs] = patches.new_owner
        self.deviation[patches.inputs] = patches.deviation

    def new_round(self) -> None:
        """Forget every refusal, so that a round tries each operation again.

        Within a round a refused operation stays refused even when something beside it changes.
        Clearing the refusals around every change instead re-checks the same doomed candidates
        after each of their neighbours' collapses, which cost a surface of 194,636 triangles most
        of its time in sweeps that applied nothing; a round is repeated until it changes nothing,
        so an operation that a later change makes valid is still found, one round later.
        """
        self.refused_collapse = np.zeros(0, dtype=np.int64)
        self.refused_flip = np.zeros(0, dtype=np.int64)
        self.screening = False

    # -- collapse ------------------------------------------------------------------------------

    def _check_collapses(self, a: np.ndarray, b: np.ndarray, topology: _Topology):
        """Check moving each ``a[g]`` onto ``b[g]``; return which pass and what they change."""
        n, nt = len(a), self.n_triangles
        at_a, group_a, _ = ragged.rows(topology.vertex_offsets, topology.vertex_triangles, a)
        at_b, group_b, _ = ragged.rows(topology.vertex_offsets, topology.vertex_triangles, b)
        shared_a = np.isin(group_a * nt + at_a, group_b * nt + at_b)
        shared_b = np.isin(group_b * nt + at_b, group_a * nt + at_a)
        shared_group, shared = group_a[shared_a], at_a[shared_a]
        moved_group, moved = group_a[~shared_a], at_a[~shared_a]
        kept_group, kept = group_b[~shared_b], at_b[~shared_b]

        ok = _count(shared_group, n) > 0  # the edge still exists
        # Nothing would remain to stand for the facets the collapse removes.
        ok &= _count(np.concatenate([moved_group, kept_group]), n) > 0

        # The link condition: the two vertices' common neighbours are exactly the corners opposite
        # the edge, or the collapse would pinch the surface.
        def neighbours(groups, triangles, excluded):
            g = np.repeat(groups, 3)
            v = self.triangles[triangles].ravel()
            keep = ~np.any(v[:, None] == excluded[g], axis=1)
            return _unique_pairs(g[keep], v[keep], self.n_vertices)

        of_a = neighbours(group_a, at_a, a[:, None])
        of_b = neighbours(group_b, at_b, b[:, None])
        common = np.intersect1d(
            of_a[0] * self.n_vertices + of_a[1], of_b[0] * self.n_vertices + of_b[1]
        )
        opposite = neighbours(shared_group, shared, np.stack([a, b], axis=1))[0]
        ok &= _count(common // self.n_vertices, n) == _count(opposite, n)

        labels = self.triangles[moved]
        labels = np.where(labels == a[moved_group][:, None], b[moved_group][:, None], labels)
        old = self.position[self.triangles[moved]]
        floor = np.full(n, MIN_QUALITY)
        np.minimum.at(floor, moved_group, _quality(old))
        refused = self._shape_refused(old, self.position[labels], floor[moved_group])
        ok &= _count(moved_group, n, refused) == 0

        # The chord and angle, only for what survived the cheap checks.
        on_m, on_k, on_s = ok[moved_group], ok[kept_group], ok[shared_group]
        passed, patches = self._check_patches(
            np.concatenate([moved_group[on_m], kept_group[on_k]]),
            np.concatenate([moved[on_m], kept[on_k]]),
            np.concatenate([labels[on_m], self.triangles[kept[on_k]]]),
            shared_group[on_s],
            shared[on_s],
            n,
            topology,
        )
        return ok & passed, patches

    def _collapse_sweep(self) -> tuple[int, bool]:
        """One batch of collapses; return how many were made and whether any was tried."""
        topology = self._topology()
        degree, is_feature = self._feature_state(topology)
        low, high = topology.edge_ends.T
        span = self.position[low] - self.position[high]
        short = _dot(span, span) < self.max_edge**2

        def movable(a, b):
            return ~self.pinned[a] & ((degree[a] == 0) | ((degree[a] == 2) & is_feature))

        low_moves, high_moves = movable(low, high), movable(high, low)
        refused = np.isin(topology.edge_key, self.refused_collapse)
        tried = np.flatnonzero(short & ~refused & (low_moves | high_moves))
        if not tried.size:
            return 0, False
        length = np.sqrt(_dot(span[tried], span[tried]))
        priority = self._priority(np.floor(length / (PRIORITY_BAND * self.max_edge)))

        def check(edges):
            """Check each edge's collapse, the vertex with fewer features moving first."""
            lo, hi = low[edges], high[edges]
            lo_ok, hi_ok = low_moves[edges], high_moves[edges]
            lo_first = lo_ok & (~hi_ok | (degree[lo] <= degree[hi]))
            first_a, first_b = np.where(lo_first, lo, hi), np.where(lo_first, hi, lo)
            both = np.flatnonzero(lo_ok & hi_ok)
            a = np.concatenate([first_a, first_b[both]])
            b = np.concatenate([first_b, first_a[both]])
            passed, patches = self._check_collapses(a, b, topology)
            # The second way round only where the first failed.
            take = passed.copy()
            take[len(edges) :] &= ~passed[both]
            of_edge = np.concatenate([np.arange(len(edges)), both])
            return a, b, take, of_edge, patches

        if self.screening:
            # Checking reads and never writes, so every candidate can be checked at once; only
            # applying needs them independent. Refuse all that fail, then choose among the rest.
            a, b, take, of_edge, patches = check(tried)
            passing = np.zeros(len(tried), dtype=bool)
            passing[of_edge[take]] = True
            refuse = tried[~passing]
            keep = np.flatnonzero(passing)
            chosen = np.zeros(len(tried), dtype=bool)
            chosen[
                keep[self._local_minima(topology.edge_ends[tried[keep]], priority[keep], topology)]
            ] = True
            take &= chosen[of_edge]
            selected = int(chosen.sum())
        else:
            # Shortest first, within each neighbourhood.
            chosen = tried[self._local_minima(topology.edge_ends[tried], priority, topology)]
            a, b, take, of_edge, patches = check(chosen)
            succeeded = np.zeros(len(chosen), dtype=bool)
            succeeded[of_edge[take]] = True
            refuse = chosen[~succeeded]
            selected = len(chosen)
        self.refused_collapse = np.union1d(self.refused_collapse, topology.edge_key[refuse])
        # Once most chosen candidates fail, the rest are mostly refusals waiting their turn at
        # one percent a sweep: check them all in the next sweep instead. Its survivors all pass,
        # so ordinary sweeps follow until the rate falls again.
        self.screening = int(take.sum()) < SCREEN_BELOW * selected
        self._apply(patches.accepted(take))
        # The removed vertex's feature edges become its target's.
        remap = np.arange(self.n_vertices)
        remap[a[take]] = b[take]
        features = np.sort(remap[self.features], axis=1)
        self.features = np.unique(features[features[:, 0] != features[:, 1]], axis=0)
        return int(take.sum()), True

    # -- flips ---------------------------------------------------------------------------------

    def _flip_sweep(self) -> tuple[int, bool]:
        """One batch of flips; return how many were made and whether any was tried."""
        topology = self._topology()
        _, is_feature = self._feature_state(topology)
        first, second = topology.edge_first, topology.edge_second
        pair = np.maximum(second, 0)
        tried = np.flatnonzero(
            (topology.edge_uses == 2)
            & ~is_feature
            & (self.solid[first] == self.solid[pair])
            & ~np.isin(topology.edge_key, self.refused_flip)
        )
        if not tried.size:
            return 0, False
        one, two = first[tried], second[tried]
        a, b = topology.edge_ends[tried].T
        p = self.triangles[one].sum(axis=1) - a - b
        q = self.triangles[two].sum(axis=1) - a - b
        # Keep the winding: in `one` the edge must run a -> b, so the new pair runs p -> q.
        rows = self.triangles[one]
        forward = np.any((rows == a[:, None]) & (np.roll(rows, -1, axis=1) == b[:, None]), axis=1)
        a, b = np.where(forward, a, b), np.where(forward, b, a)
        new = np.stack([np.stack([a, q, p], axis=1), np.stack([q, b, p], axis=1)], axis=1)
        old = self.position[np.stack([rows, self.triangles[two]], axis=1)]
        corners = self.position[new]
        gain = _quality(corners).min(axis=1) - _quality(old).min(axis=1)
        diagonal = np.minimum(p, q) * self.n_vertices + np.maximum(p, q)
        edges = _edges(corners)
        too_long = np.any(_dot(edges, edges) > self.max_edge**2, axis=(1, 2))
        floor = np.minimum(MIN_QUALITY, _quality(old).min(axis=1))
        too_poor = _quality(corners).min(axis=1) < floor
        # Neither new triangle may face away from the pair it replaces.
        facing = _normal(old).sum(axis=1)[:, None, :]
        turned_over = np.any(_dot(_normal(corners), facing) <= 0.0, axis=1)
        refused = too_long | too_poor | turned_over
        viable = (p != q) & ~np.isin(diagonal, topology.edge_key) & (gain > 1e-9) & ~refused
        self.refused_flip = np.union1d(self.refused_flip, topology.edge_key[tried[~viable]])
        if not viable.any():
            return 0, True
        keep = np.flatnonzero(viable)
        ends = np.stack([a, b, p, q], axis=1)[keep]
        priority = self._priority(np.floor(-gain[keep] / PRIORITY_BAND))
        chosen = keep[self._local_minima(ends, priority, topology)]

        n = len(chosen)
        passed, patches = self._check_patches(
            np.repeat(np.arange(n), 2),
            np.stack([one[chosen], two[chosen]], axis=1).ravel(),
            new[chosen].reshape(-1, 3),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            n,
            topology,
        )
        self.refused_flip = np.union1d(self.refused_flip, topology.edge_key[tried[chosen[~passed]]])
        self._apply(patches.accepted(passed))
        return int(passed.sum()), True

    def run(self, sweep) -> int:
        """Sweep until nothing is left to try; return how many operations were made."""
        done = 0
        while True:
            made, tried = sweep()
            done += made
            if not tried:
                return done

    # -- the result ----------------------------------------------------------------------------

    def result(self, n_bodies: int, area_before: np.ndarray) -> Coarsening:
        live = np.flatnonzero(self.alive)
        corners = self.position[self.triangles[live]]
        normal = _normal(corners)
        length = np.sqrt(_dot(normal, normal))
        row = np.full(self.n_triangles, -1)
        row[live] = np.arange(len(live))
        owner = row[self.owner]
        chord = np.zeros(len(live))
        np.maximum.at(chord, owner, self.deviation)
        cosine = _dot(self.input_normal, normal[owner] / length[owner][:, None])
        angle = np.zeros(len(live))
        np.maximum.at(angle, owner, np.arccos(np.clip(cosine, -1.0, 1.0)))
        edges = _edges(corners)
        return Coarsening(
            vertices=corners,
            solid_id=self.solid[live],
            longest_edge=np.sqrt(_dot(edges, edges).max(axis=1)),
            chord=chord,
            angle=angle,
            area_before=area_before,
            area_after=np.bincount(self.solid[live], weights=0.5 * length, minlength=n_bodies),
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
        decimator.new_round()
        if decimator.run(decimator._collapse_sweep) + decimator.run(decimator._flip_sweep) == 0:
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
