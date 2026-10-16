"""Sample a field on an octree, refining where the coarser tree failed to predict it.

The loop, from a uniform tree of root cells over a region:

1. **Sample** the field at every new node in the region, and at every new cut point -- where a new
   cell's edge crosses the region's boundary, a standoff inside (:mod:`.cuts`). Nothing sampled
   before is sampled again: a sampled value is kept as it is, so the loop suits a field whose value
   at a point does not depend on any other point, such as one computed by an independent sum.
2. **Measure each new sample's miss**: its value against what the tree *before* this pass predicted
   there. That is the interpolation error the previous tree made at that point, measured rather than
   estimated.
3. **Split** each new cell where a sample on it (a new corner, or a new cut point on one of its
   edges) missed by more than the tolerance; or where the cell is coarser than ``min_level``.
4. Repeat until no cell is split, or every cell that would be is at ``max_level``.

**The interpolant** is the hierarchical one of :mod:`.hierarchy`: continuous everywhere, and through
every node sampled. A node outside the region -- or inside it by less than the standoff -- is not
sampled but is given a ghost value (:mod:`.ghosts`) fitted to the cut points on its edges and the
samples around it, so that the cells the boundary cuts follow the field to the boundary instead of
extrapolating it from one side. A ghost is fitted again whenever finer cells appear around it, at
their scale, and the surpluses are then rebuilt from every node's value: a sampled value never
changes, but a ghost is an estimate, and the first one, made from coarse cells, is the poorest.

Only the root cells the region can reach are kept, so a region filling little of its box -- an
L-shaped reactor in the box around it -- costs nothing for the empty part; a point there lies in no
leaf and is refused when interpolated.

**What is interpolated** can be a transform of the field (:class:`Logarithm` for a positive field
that varies over orders of magnitude, where a tolerance is then relative); misses and the tolerance
are in the transformed values.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable

import numpy as np
from scipy.spatial import cKDTree

from .cuts import exit_points, leaf_cuts
from .ghosts import ghost_values
from .hierarchy import Hierarchy
from .region import signed_distance as region_distance
from .tree import Octree

#: The six neighbours along the axes, as offsets.
_AXES = np.concatenate([np.eye(3, dtype=np.int64), -np.eye(3, dtype=np.int64)])
#: How many cells of its own level, either way along each axis, a ghost's extrapolation reaches for
#: data. One cell -- the node's own hat -- leaves a node at a convex edge of the region with data all
#: in one plane, and so no gradient across it; two reach past the edge's other side.
_EXTRAPOLATION_REACH = 2
#: The lattice points around a node within that reach, as offsets in cells.
_AROUND = np.array(
    [
        (i, j, k)
        for i in range(-_EXTRAPOLATION_REACH, _EXTRAPOLATION_REACH + 1)
        for j in range(-_EXTRAPOLATION_REACH, _EXTRAPOLATION_REACH + 1)
        for k in range(-_EXTRAPOLATION_REACH, _EXTRAPOLATION_REACH + 1)
        if (i, j, k) != (0, 0, 0)
    ],
    dtype=np.int64,
)
#: Two cut points closer than this many lattice steps are one: the same crossing reached along edges
#: of different lengths lands within rounding of itself, far below any standoff.
_SAME_POINT = 1e-9


@dataclasses.dataclass(frozen=True)
class Linear:
    """Interpolate the field as it is."""

    def forward(self, values: np.ndarray) -> np.ndarray:
        """The field, as interpolated."""
        return np.asarray(values, dtype=float)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """The interpolated values, as the field."""
        return np.asarray(values, dtype=float)


@dataclasses.dataclass(frozen=True)
class Logarithm:
    """Interpolate the logarithm of a field that is positive, or zero below a floor.

    A field falling off exponentially -- light through an absorbing medium -- is close to linear in
    its logarithm, and a tolerance on the logarithm is a tolerance on the relative error.

    Attributes
    ----------
    floor : float
        Values below it are taken as it, so a zero (a shadow) has a finite logarithm and a miss among
        values all below the floor is zero: the tolerance is relative above the floor and does not
        apply below it.
    """

    floor: float

    def __post_init__(self) -> None:
        if not self.floor > 0.0:
            raise ValueError(f"Logarithm.floor must be positive, got {self.floor!r}.")

    def forward(self, values: np.ndarray) -> np.ndarray:
        """``log(max(values, floor))``."""
        return np.log(np.maximum(np.asarray(values, dtype=float), self.floor))

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """``exp(values)``."""
        return np.exp(np.asarray(values, dtype=float))


@dataclasses.dataclass(frozen=True)
class Round:
    """What one pass of the refinement did.

    Attributes
    ----------
    leaves : int
        Leaves in the tree after the pass.
    split : int
        Leaves split to make the pass's new cells (zero on the first pass, which made the root cells).
    nodes : int
        New nodes sampled (inside the region).
    cut_points : int
        New cut points sampled.
    ghosts : int
        Nodes outside the region (or nearer its boundary than the standoff) given a ghost value: new
        ones, and old ones fitted again at the scale of the pass's new cells.
    largest_miss : float
        The largest miss among the pass's samples, in the interpolated (transformed) values; ``nan``
        on the first pass, which had nothing to predict them.
    marked : int
        New cells marked to be split in the next pass.
    seconds : float
        Wall time of the pass's sampling.
    """

    leaves: int
    split: int
    nodes: int
    cut_points: int
    ghosts: int
    largest_miss: float
    marked: int
    seconds: float


@dataclasses.dataclass(frozen=True, eq=False)
class SampledField:
    """A field sampled adaptively on an octree, and its interpolant.

    Call it with points to interpolate the field there.

    Attributes
    ----------
    tree : Octree
        The final tree.
    hierarchy : Hierarchy
        The nodes' surpluses, of the transformed field: the sampled nodes and the ghosts.
    transform : Linear or Logarithm
        What is interpolated.
    points : np.ndarray, shape ``(m, 3)``
        Every point sampled, in the order sampled.
    values : np.ndarray, shape ``(m,)``
        The field there, as sampled.
    is_cut : np.ndarray of bool, shape ``(m,)``
        Which of them are cut points rather than nodes.
    misses : np.ndarray, shape ``(m,)``
        How far the tree before each sample's pass missed it, in the transformed values; ``nan`` for
        the first pass's samples, which nothing predicted.
    sampled_nodes, ghost_nodes : np.ndarray of int
        The sorted keys (:meth:`Octree.key`) of the nodes sampled -- those at least the standoff inside
        the region -- and of the others given a ghost value.
    rounds : tuple of Round
        What each pass did.
    """

    tree: Octree
    hierarchy: Hierarchy
    transform: Linear | Logarithm
    points: np.ndarray
    values: np.ndarray
    is_cut: np.ndarray
    misses: np.ndarray
    sampled_nodes: np.ndarray
    ghost_nodes: np.ndarray
    rounds: tuple

    def interpolated(self, points) -> np.ndarray:
        """The transformed field's interpolant at ``points`` (``(n, 3)``).

        Raises
        ------
        ValueError
            If a point lies in no leaf: outside the tree's box, or in a root cell the region does not
            reach, which is not kept.
        """
        points = np.asarray(points, dtype=float)
        outside = self.tree.locate(points) < 0
        if outside.any():
            raise ValueError(
                f"{int(outside.sum())} points lie in no leaf of the octree: outside its box, or in a "
                "part of the box the region does not reach."
            )
        return self.hierarchy(self.tree, points)

    def __call__(self, points) -> np.ndarray:
        """The field interpolated to ``points`` (``(n, 3)``), shape ``(n,)``."""
        return self.transform.inverse(self.interpolated(points))

    def hexahedra(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The leaves the field reaches -- a corner sampled or given a ghost -- as hexahedra.

        Returns
        -------
        points : np.ndarray, shape ``(k, 3)``
            The corners of those leaves, each once.
        corners : np.ndarray of int, shape ``(m, 8)``
            Each leaf's corners as rows of ``points``, in :data:`~aquaflux.octree.tree.CORNERS` order.
        rows : np.ndarray of int, shape ``(m,)``
            The leaves, as rows of the tree.
        values : np.ndarray, shape ``(k,)``
            The field sampled at each point in the region, ``nan`` at one outside it.
        """
        tree = self.tree
        keys = tree.key(tree.corners())
        reached = np.isin(keys, self.sampled_nodes) | np.isin(keys, self.ghost_nodes)
        rows = np.flatnonzero(reached.any(axis=1))
        nodes, corners = np.unique(keys[rows], return_inverse=True)
        values = np.full(len(nodes), np.nan)
        inside = np.isin(nodes, self.sampled_nodes)
        values[inside] = self.transform.inverse(
            self.hierarchy.at_lattice(tree, tree.unkey(nodes[inside]))
        )
        return tree.position(tree.unkey(nodes)), corners.reshape(-1, 8), rows, values


def sample_adaptively(
    sample: Callable[[np.ndarray], np.ndarray],
    region,
    *,
    low,
    high,
    spacing: float,
    max_level: int,
    tolerance: float,
    standoff: float,
    transform: Linear | Logarithm | None = None,
    min_level: int = 1,
    report: Callable[[str], None] | None = None,
) -> SampledField:
    """Sample a field over a region on an octree refined until it predicts its own new samples.

    Parameters
    ----------
    sample : callable
        ``sample(points)`` with points ``(n, 3)`` inside the region returns the field there, ``(n,)``.
        Called once per pass, with only that pass's new points.
    region : aquaflux.solids.Solid
        Where the field lives: anything answering ``intervals(origin, direction)`` and
        ``signed_distance(points)`` as :class:`aquaflux.solids.Solid` does. A node
        is sampled only where it lies at least ``standoff`` inside; one nearer the boundary, or outside,
        is given a ghost value instead.
    low, high : array_like, shape ``(3,)``
        A box holding the region, with room around it: a node outside the region needs neighbours.
    spacing : float
        The root cells' edge length.
    max_level : int
        How many times a root cell may be halved.
    tolerance : float
        The largest miss, in the transformed field, that leaves a cell unsplit.
    standoff : float
        How far from the boundary, by the region's signed distance, every sample lies: a node nearer
        it is not sampled, and a cut point is placed that deep. Set it beyond any layer in which the
        field is not to be trusted -- a lamp drawn as flat facets lies up to its chord tolerance
        inside the true surface, and the field a few chords from it still carries the facets'
        pattern, which a refinement asked to resolve it would chase.
    transform : Linear or Logarithm, optional
        What is interpolated; the field itself if unset.
    min_level : int, optional
        Every cell holding part of the region is split at least to this level, whatever its misses.
        The first pass has nothing to measure a miss against, so at zero nothing is refined.
    report : callable, optional
        Called with a line of text after each pass.

    Returns
    -------
    SampledField

    Raises
    ------
    ValueError
        If no node or cut point of the root cells lies in the region, ``min_level`` is not in
        ``[0, max_level]``, or the sampler returns the wrong shape.
    """
    if not 0 <= min_level <= max_level:
        raise ValueError(
            f"sample_adaptively: min_level must lie in [0, max_level], got {min_level} and {max_level}."
        )
    transform = Linear() if transform is None else transform
    say = report or (lambda _: None)
    build = _Refinement(sample, region, transform, standoff)
    tree = Octree.covering(low, high, spacing, max_level)
    tree = tree.subset(np.flatnonzero(_within_reach(region, tree, np.arange(tree.n_leaves))))
    new = np.arange(tree.n_leaves)
    split = 0
    while True:
        record, misses = build.absorb(tree, new)
        marked = build.reached(tree, new) & (tree.levels[new] < max_level)
        marked &= (tree.levels[new] < min_level) | (misses > tolerance)
        record = dataclasses.replace(record, split=split, marked=int(marked.sum()))
        build.rounds.append(record)
        say(
            f"pass {len(build.rounds) - 1}: {record.leaves} leaves; sampled {record.nodes} nodes and "
            f"{record.cut_points} cut points in {record.seconds:.1f} s, {record.ghosts} ghosts; "
            f"largest miss {record.largest_miss:.3g}; {record.marked} cells to split"
        )
        if not marked.any():
            break
        mask = np.zeros(tree.n_leaves, dtype=bool)
        mask[new[marked]] = True
        tree, children = tree.split(mask)
        new, split = children.reshape(-1), int(marked.sum())
    return build.field(tree)


def _within_reach(region, tree: Octree, rows: np.ndarray, reach: float = 0.0) -> np.ndarray:
    """Which leaves the region's boundary, or the region, may reach within ``reach`` of: those whose
    centre is within half a diagonal of it, and ``reach``. The signed distance of a solid is at most
    the true distance in magnitude, so this keeps every leaf it should."""
    half_diagonal = 0.5 * np.sqrt(3.0) * tree.cell_spacing(rows)
    return region_distance(region, tree.centres(rows)) <= half_diagonal + reach


class _Refinement:
    """The samples taken so far and the interpolant they make, grown pass by pass."""

    def __init__(self, sample, region, transform, standoff):
        self.sample = sample
        self.region = region
        self.transform = transform
        self.standoff = float(standoff)
        self.hierarchy: Hierarchy | None = None
        self.tree: Octree | None = None
        # Every node met, sorted by key: whether it is in the region, and its row of the samples
        # (-1 outside the region).
        self.node_keys = np.zeros(0, dtype=np.int64)
        self.node_inside = np.zeros(0, dtype=bool)
        self.node_sample = np.zeros(0, dtype=np.int64)
        self.ghost_keys = np.zeros(0, dtype=np.int64)  # sorted: every node given a ghost
        self.ghost_values = np.zeros(0)
        self.cut_index: dict[tuple, int] = {}  # rounded cut point -> its row of the samples
        self.points: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        self.is_cut: list[np.ndarray] = []
        self.misses: list[np.ndarray] = []
        self.transformed = np.zeros(0)  # every sample's transformed value, in sample order
        self.rounds: list[Round] = []

    # -- bookkeeping ------------------------------------------------------------------------------

    def _rounded(self, tree: Octree, points: np.ndarray) -> list[tuple]:
        quantum = _SAME_POINT * tree.step
        return list(map(tuple, np.round((points - tree.origin) / quantum).astype(np.int64)))

    def _lookup(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Which keys are known nodes, and the row of each in the node table."""
        keys = np.asarray(keys, dtype=np.int64)
        if len(self.node_keys) == 0:
            return np.zeros(keys.shape, dtype=bool), np.zeros(keys.shape, dtype=np.int64)
        slot = np.minimum(np.searchsorted(self.node_keys, keys), len(self.node_keys) - 1)
        return self.node_keys[slot] == keys, slot

    def _register_nodes(self, keys: np.ndarray, inside: np.ndarray, sample: np.ndarray) -> None:
        merged = np.concatenate([self.node_keys, keys])
        order = np.argsort(merged)
        self.node_keys = merged[order]
        self.node_inside = np.concatenate([self.node_inside, inside])[order]
        self.node_sample = np.concatenate([self.node_sample, sample])[order]

    def _sampled_value(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For lattice keys: whether each is a sampled node, and its transformed value (else 0)."""
        known, slot = self._lookup(keys)
        row = np.where(known, self.node_sample[slot], -1)
        found = row >= 0
        return found, np.where(found, self.transformed[np.maximum(row, 0)], 0.0)

    # -- one pass ---------------------------------------------------------------------------------

    def absorb(self, tree: Octree, new: np.ndarray) -> tuple[Round, np.ndarray]:
        """Sample the new leaves' new nodes and cut points, fit ghosts at their outside corners, and
        rebuild the interpolant from every node's value.

        Returns the pass's record and each new leaf's largest miss: ``-inf`` where it had no new
        sample, and everywhere on the first pass, which had nothing to predict its samples.
        """
        first = self.hierarchy is None
        corner_keys = tree.key(tree.corners(new))  # (r, 8)
        candidates = np.unique(corner_keys)
        fresh = candidates[~self._lookup(candidates)[0]]
        fresh_lattice = tree.unkey(fresh)
        # A node is sampled only at least a standoff inside the region; one nearer the boundary is given a
        # ghost, as one outside is. On a lamp's true surface a node can lie in the plane of the facets that
        # stand in for it, where the field is not defined -- a gather there returns nothing.
        fresh_inside = region_distance(self.region, tree.position(fresh_lattice)) <= -self.standoff
        sampled_nodes = fresh[fresh_inside]
        base = len(self.transformed)
        node_rows = np.full(len(fresh), -1, dtype=np.int64)
        node_rows[fresh_inside] = base + np.arange(len(sampled_nodes))
        self._register_nodes(fresh, fresh_inside, node_rows)

        # The cut points: on the new leaves' edges, and where an edge from a new outside node to a
        # neighbour in the region leaves it -- usually the same points.
        half_diagonal = 0.5 * np.sqrt(3.0) * tree.cell_spacing(new)
        boundary_near = np.abs(region_distance(self.region, tree.centres(new))) <= half_diagonal
        cuts = leaf_cuts(tree, new[boundary_near], self.region, self.standoff)
        cut_leaf = np.flatnonzero(boundary_near)
        # Ghosts: for every outside corner of the new leaves -- a new node, or an old one whose ghost was
        # fitted at a coarser scale and is fitted again at the finest new leaf around it -- whose hat
        # can reach into the region.
        leaf_size = np.broadcast_to(tree.size(tree.levels[new])[:, None], corner_keys.shape)
        corner_known, corner_slot = self._lookup(corner_keys)
        outside_corner = corner_known & ~self.node_inside[corner_slot]
        outside, at = np.unique(corner_keys[outside_corner], return_inverse=True)
        size = np.full(len(outside), np.iinfo(np.int64).max)
        np.minimum.at(size, at.reshape(-1), leaf_size[outside_corner])
        reachable = region_distance(self.region, tree.position(tree.unkey(outside))) <= (
            np.sqrt(3.0) * tree.step * size
        )
        outside, size = outside[reachable], size[reachable]
        edges = self._ghost_edges(tree, outside, size)
        cut_like = np.concatenate([cuts.points, edges["points"]])
        rounded = self._rounded(tree, cut_like)
        pending: dict[tuple, int] = {}
        for key in rounded:
            if key not in self.cut_index and key not in pending:
                pending[key] = len(pending)
        cut_rows = np.array(list(pending.values()), dtype=np.int64)
        first_of = {key: index for index, key in reversed(list(enumerate(rounded)))}
        cut_points = cut_like[[first_of[key] for key in pending]] if pending else np.zeros((0, 3))

        points = np.concatenate([tree.position(tree.unkey(sampled_nodes)), cut_points])
        started = time.perf_counter()
        values = np.asarray(self.sample(points), dtype=float) if len(points) else np.zeros(0)
        seconds = time.perf_counter() - started
        if values.shape != (len(points),):
            raise ValueError(
                f"sample_adaptively: the sampler returned shape {values.shape} for {len(points)} points."
            )
        if first and len(points) == 0:
            raise ValueError(
                "sample_adaptively: no node or cut point of the root cells lies in the region; "
                "check the box and the region, or use a finer root spacing."
            )
        transformed = self.transform.forward(values)
        if first:
            miss = np.full(len(points), -np.inf)
        else:
            miss = np.abs(transformed - self.hierarchy(self.tree, points))

        self.points.append(points)
        self.values.append(values)
        self.is_cut.append(np.arange(len(points)) >= len(sampled_nodes))
        self.misses.append(np.full(len(points), np.nan) if first else miss)
        self.transformed = np.concatenate([self.transformed, transformed])
        for key, row in zip(pending, cut_rows, strict=True):
            self.cut_index[key] = base + len(sampled_nodes) + row

        ghost_keys, ghosts = self._ghosts(tree, edges, rounded[len(cuts.points) :])
        self._set_ghosts(ghost_keys, ghosts)
        # The surpluses, from every node's value afresh: a ghost fitted again changes what every finer
        # node's surplus is measured against.
        sampled = self.node_keys[self.node_inside]
        sampled_values = self.transformed[self.node_sample[self.node_inside]]
        self.hierarchy = Hierarchy.empty(tree).with_values(
            tree,
            np.concatenate([sampled, self.ghost_keys]),
            np.concatenate([sampled_values, self.ghost_values]),
        )
        self.tree = tree

        # Each new leaf's largest miss over the new samples on it: its new corners in the region and
        # the new cut points on its edges.
        miss_of = {key: miss[len(sampled_nodes) + row] for key, row in pending.items()}
        corner_miss = np.full(corner_keys.shape, -np.inf)
        if len(sampled_nodes):
            slot = np.minimum(np.searchsorted(sampled_nodes, corner_keys), len(sampled_nodes) - 1)
            on = sampled_nodes[slot] == corner_keys
            corner_miss[on] = miss[slot[on]]
        leaf_miss = corner_miss.max(axis=1)
        cut_miss = np.array([miss_of.get(key, -np.inf) for key in rounded[: len(cuts.points)]])
        owner = cut_leaf[np.repeat(np.arange(len(cut_leaf)), np.diff(cuts.offsets))]
        if len(owner):
            np.maximum.at(leaf_miss, owner, cut_miss[cuts.items])
        record = Round(
            leaves=tree.n_leaves,
            split=0,
            nodes=len(sampled_nodes),
            cut_points=len(cut_points),
            ghosts=len(ghost_keys),
            largest_miss=float(np.max(miss)) if (len(miss) and not first) else float("nan"),
            marked=0,
            seconds=seconds,
        )
        return record, leaf_miss

    def _set_ghosts(self, keys: np.ndarray, values: np.ndarray) -> None:
        """Add these ghosts, or replace the values of those already given one."""
        kept = ~np.isin(self.ghost_keys, keys)
        merged = np.concatenate([self.ghost_keys[kept], keys])
        order = np.argsort(merged)
        self.ghost_keys = merged[order]
        self.ghost_values = np.concatenate([self.ghost_values[kept], values])[order]

    def _ghost_edges(self, tree: Octree, outside: np.ndarray, size: np.ndarray) -> dict:
        """For nodes outside the region, the edges of length ``size`` (lattice steps, one per node) to
        their neighbours in it along the axes, and where each edge leaves the region."""
        lattice = tree.unkey(outside)
        neighbour = lattice[:, None, :] + size[:, None, None] * _AXES[None, :, :]  # (n, 6, 3)
        in_box = np.all((neighbour >= 0) & (neighbour < tree.lattice_shape), axis=-1)
        keys = tree.key(np.where(in_box[..., None], neighbour, 0))
        known, slot = self._lookup(keys)
        usable = in_box & known & self.node_inside[slot]
        owner, axis = np.nonzero(usable)
        start = tree.position(neighbour[owner, axis])
        points, t, found = exit_points(
            self.region, start, tree.position(lattice[owner]), self.standoff
        )
        return {
            "nodes": outside,
            "lattice": lattice,
            "size": size,
            "owner": owner[found],
            "neighbour": keys[owner, axis][found],
            "t": t[found],
            "points": points[found],
        }

    def _ghosts(
        self, tree: Octree, edges: dict, rounded: list[tuple]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ghost values for nodes outside the region, from the samples around each within the scale of
        its edges; only the nodes with any sample around them get one."""
        n = len(edges["nodes"])
        if n == 0:
            return edges["nodes"], np.zeros(0)
        lattice, size = edges["lattice"], edges["size"].astype(float)
        # Along each edge: the neighbour's value and the cut point's.
        _, neighbour_value = self._sampled_value(edges["neighbour"])
        cut_value = (
            self.transformed[[self.cut_index[key] for key in rounded]] if rounded else np.zeros(0)
        )
        # For the extrapolation: the sampled nodes around each at its own level, and every cut point, in
        # a box reaching _EXTRAPOLATION_REACH cells of its own level either way, edges included.
        around = lattice[:, None, :] + edges["size"][:, None, None] * _AROUND[None, :, :]
        in_box = np.all((around >= 0) & (around < tree.lattice_shape), axis=-1)
        found, value = self._sampled_value(tree.key(np.where(in_box[..., None], around, 0)))
        owner, which = np.nonzero(in_box & found)
        data_owner, data_offset, data_value = (
            [owner],
            [_AROUND[which].astype(float)],
            [value[owner, which]],
        )
        rows = np.fromiter(self.cut_index.values(), dtype=np.int64, count=len(self.cut_index))
        if len(rows):
            position = tree.lattice(np.concatenate(self.points)[rows])
            reach = _EXTRAPOLATION_REACH * size * (1.0 + 1e-9)
            near = cKDTree(position).query_ball_point(lattice.astype(float), r=reach, p=np.inf)
            counts = np.array([len(group) for group in near], dtype=np.int64)
            hit = np.concatenate([np.asarray(group, dtype=np.int64) for group in near])
            cut_owner = np.repeat(np.arange(n), counts)
            data_owner.append(cut_owner)
            data_offset.append((position[hit] - lattice[cut_owner]) / size[cut_owner, None])
            data_value.append(self.transformed[rows[hit]])
        ghosts, informed = ghost_values(
            n,
            edges["owner"],
            edges["t"],
            neighbour_value,
            cut_value,
            np.concatenate(data_owner),
            np.concatenate(data_offset),
            np.concatenate(data_value),
        )
        return edges["nodes"][informed], ghosts[informed]

    def reached(self, tree: Octree, rows: np.ndarray) -> np.ndarray:
        """Which leaves may hold part of the region: a corner sampled in it, or the region within reach
        of the centre."""
        known, slot = self._lookup(tree.key(tree.corners(rows)))
        inside = (known & self.node_inside[slot]).any(axis=1)
        return inside | _within_reach(self.region, tree, rows)

    def field(self, tree: Octree) -> SampledField:
        return SampledField(
            tree=tree,
            hierarchy=self.hierarchy,
            transform=self.transform,
            points=np.concatenate(self.points),
            values=np.concatenate(self.values),
            is_cut=np.concatenate(self.is_cut),
            misses=np.concatenate(self.misses),
            sampled_nodes=self.node_keys[self.node_inside],
            ghost_nodes=self.ghost_keys,
            rounds=tuple(self.rounds),
        )
