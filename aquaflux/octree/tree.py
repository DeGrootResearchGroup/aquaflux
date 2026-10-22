"""An octree of cubes over a box: a lattice of equal root cubes, each split into eight halves as often
as it is asked.

Cubes are held in integer **units**, the edge of the smallest cube the tree has: a cube of level ``l``
in a tree of depth ``d`` spans ``2**(d - l)`` units, so its corners, its edges' ends and the point at
its centre all have integer coordinates, and two cubes share a corner exactly when they name the same
integer triple. Nothing is compared as a float. Leaves never overlap, so a leaf is named by its lower
corner and its level.

A tree's depth is its finest level. Splitting a leaf at that level halves the unit and doubles every
coordinate, so a tree can grow as deep as it is asked (:meth:`Octree.refined`); a tree built at a depth
it will not exceed (:meth:`Octree.lattice` with ``depth``) keeps one unit throughout, which is what a
caller holding coordinates from one pass to the next needs.

A lattice point's **level** is the coarsest level whose cubes have corners there -- the level at which
it first appears as a corner -- and is read off its coordinates alone (:meth:`Octree.point_level`).

Two ways of refining share the tree. :meth:`Octree.refined_by` splits where any of some criteria asks,
and :meth:`Octree.balanced` then splits until neighbours differ by at most a level, the bound a cube
cut into fixed shapes to match smaller neighbours needs. A caller that decides its own splits, leaf by
leaf, calls :meth:`Octree.refined` directly and need not balance.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
from collections.abc import Sequence
from typing import Protocol

import numpy as np

#: Bits per axis in a point's key (:meth:`Octree.keys`). Coordinates may be held at half a unit, so a
#: tree may span at most ``2**(BITS - 1)`` units along an axis.
BITS = 21

#: The eighteen directions to a cube's face and edge neighbours: each offset with one or two of its
#: components non-zero.
FACE_AND_EDGE_DIRECTIONS = np.array(
    [d for d in itertools.product((-1, 0, 1), repeat=3) if 1 <= np.count_nonzero(d) <= 2]
)

#: A cube's eight corners, as offsets from its lower corner in units of its edge, ordered so that
#: corner ``c`` has offset ``((c >> 2) & 1, (c >> 1) & 1, c & 1)``. In units of half its edge they are
#: its eight children's lower corners, in the order :meth:`Octree.refined` puts the children.
CORNERS = np.array(list(itertools.product((0, 1), repeat=3)), dtype=np.int64)

#: The twelve edges of a cube, as pairs of corners in :data:`CORNERS` order, the lower corner first.
EDGES = np.array(
    [
        (a, b)
        for a in range(8)
        for b in range(8)
        if a < b and np.abs(CORNERS[a] - CORNERS[b]).sum() == 1
    ],
    dtype=np.int64,
)


class RefinementCriterion(Protocol):
    """Which cubes to split."""

    def refines(self, lower: np.ndarray, size: np.ndarray) -> np.ndarray:
        """Whether to split each cube.

        Parameters
        ----------
        lower : np.ndarray, shape ``(n, 3)``
            Each cube's lower corner.
        size : np.ndarray, shape ``(n,)``
            Each cube's edge length.

        Returns
        -------
        np.ndarray of bool, shape ``(n,)``
        """


@dataclasses.dataclass(frozen=True)
class Octree:
    """The leaves of a lattice of octrees over a box.

    Attributes
    ----------
    origin : np.ndarray, shape ``(3,)``
        The lattice's lower corner.
    unit : float
        The edge of the smallest cube the tree has: the root cubes' edge over ``2**depth``.
    depth : int
        The finest level; the root cubes are level ``0``.
    shape : np.ndarray of int, shape ``(3,)``
        Root cubes along each axis.
    level : np.ndarray of int, shape ``(n_leaves,)``
        Each leaf's level.
    corner : np.ndarray of int, shape ``(n_leaves, 3)``
        Each leaf's lower corner, in units from the origin.
    """

    origin: np.ndarray
    unit: float
    depth: int
    shape: np.ndarray
    level: np.ndarray
    corner: np.ndarray

    @classmethod
    def lattice(
        cls, lower: Sequence[float], upper: Sequence[float], size: float, depth: int = 0
    ) -> Octree:
        """Root cubes of edge ``size`` from ``lower``, as many along each axis as cover ``upper``.

        Parameters
        ----------
        lower, upper : array_like, shape ``(3,)``
            Opposite corners of the box to cover.
        size : float
            The root cubes' edge.
        depth : int, optional
            The tree's depth from the start: its unit is ``size / 2**depth``. A tree is split no
            deeper than this without its unit changing.

        Returns
        -------
        Octree
            Every root cube a leaf, in order of its integer coordinates.

        Raises
        ------
        ValueError
            If the box is empty, the size is not positive, the depth is negative, or the tree would
            span more units along an axis than a key can hold.
        """
        lower = np.asarray(lower, dtype=np.float64)
        extent = np.asarray(upper, dtype=np.float64) - lower
        if size <= 0.0 or np.any(extent <= 0.0):
            raise ValueError("the box must have positive extent and the cubes a positive size")
        if depth < 0:
            raise ValueError(f"the depth must be at least zero, got {depth!r}")
        shape = np.maximum(np.ceil(extent / size - 1e-9), 1).astype(np.int64)
        corner = np.array(list(itertools.product(*(range(n) for n in shape))), dtype=np.int64)
        tree = cls(
            lower,
            float(size) / 2**depth,
            int(depth),
            shape,
            np.zeros(len(corner), np.int64),
            corner << depth,
        )
        tree._check_span()
        return tree

    # -- the leaves -------------------------------------------------------------------------------

    @property
    def n_leaves(self) -> int:
        """How many leaves the tree has."""
        return len(self.level)

    @property
    def extent(self) -> np.ndarray:
        """The box's edge along each axis, in units; shape ``(3,)``. Its lattice points run from
        ``0`` to ``extent`` inclusive."""
        return self.shape << self.depth

    @property
    def span(self) -> np.ndarray:
        """Each leaf's edge, in units; shape ``(n_leaves,)``."""
        return self.span_at(self.level)

    @property
    def lower(self) -> np.ndarray:
        """Each leaf's lower corner; shape ``(n_leaves, 3)``."""
        return self.position(self.corner)

    @property
    def size(self) -> np.ndarray:
        """Each leaf's edge length; shape ``(n_leaves,)``."""
        return self.unit * self.span

    @property
    def centre(self) -> np.ndarray:
        """Each leaf's centre; shape ``(n_leaves, 3)``."""
        return self.lower + 0.5 * self.size[:, None]

    def cube_corners(self, rows=None) -> np.ndarray:
        """The eight corners of each of these leaves (all by default), in units, ``(n, 8, 3)`` in
        :data:`CORNERS` order."""
        rows = slice(None) if rows is None else rows
        return self.corner[rows][:, None, :] + self.span[rows][:, None, None] * CORNERS[None]

    def refined(self, split: np.ndarray) -> Octree:
        """This tree with each leaf where ``split`` holds replaced by its eight children.

        The children follow the leaves kept, eight per split leaf in the order of their parents and in
        :data:`CORNERS` order within each, so the children of the ``k``-th split leaf are rows
        ``n_kept + 8 k`` to ``n_kept + 8 k + 7``.
        """
        split = np.asarray(split, dtype=bool)
        if not np.any(split):
            return self
        tree = self
        if np.any(self.level[split] == self.depth):  # a new finest level halves the unit
            tree = dataclasses.replace(
                self, unit=self.unit / 2.0, depth=self.depth + 1, corner=2 * self.corner
            )
            tree._check_span()
        half = (tree.span[split] // 2)[:, None, None]
        children = tree.corner[split][:, None, :] + half * CORNERS[None, :, :]
        level = np.concatenate([tree.level[~split], np.repeat(tree.level[split] + 1, 8)])
        corner = np.concatenate([tree.corner[~split], children.reshape(-1, 3)])
        return dataclasses.replace(tree, level=level, corner=corner)

    def refined_by(self, criteria: Sequence[RefinementCriterion]) -> Octree:
        """Split the leaves any criterion asks for, again and again until none asks."""
        tree = self
        while criteria:
            split = np.zeros(tree.level.shape, dtype=bool)
            for criterion in criteria:
                split |= np.asarray(criterion.refines(tree.lower, tree.size), dtype=bool)
            if not np.any(split):
                return tree
            tree = tree.refined(split)
        return tree

    def balanced(self) -> Octree:
        """Split leaves until any two sharing a face or part of an edge are at most a level apart.

        A leaf two or more levels coarser than a face or edge neighbour covers the whole of the
        neighbour's own-size region across that face or edge, so one finest cell just across it
        finds that leaf.
        """
        tree = self
        while True:
            split = np.zeros(tree.level.shape, dtype=bool)
            span = tree.span[:, None]
            for direction in FACE_AND_EDGE_DIRECTIONS:
                across = tree.corner + np.where(direction < 0, -1, np.where(direction > 0, span, 0))
                found = tree.leaf_at(across)
                coarse = (found >= 0) & (tree.level[np.maximum(found, 0)] < tree.level - 1)
                split[found[coarse]] = True
            if not np.any(split):
                return tree
            tree = tree.refined(split)

    def subset(self, rows) -> Octree:
        """The tree with only these leaves, in this order -- the cubes of a box that some region can
        reach, say. Points elsewhere in the box then lie in no leaf."""
        rows = np.asarray(rows, dtype=np.int64)
        return dataclasses.replace(self, level=self.level[rows], corner=self.corner[rows])

    # -- finding leaves ---------------------------------------------------------------------------

    @functools.cached_property
    def _sorted_corners(self) -> tuple[np.ndarray, np.ndarray]:
        """The leaves' lower corners' keys, sorted, and the leaf each belongs to."""
        keys = self.keys(self.corner)
        order = np.argsort(keys)
        return keys[order], order

    def leaf_at(self, cells: np.ndarray) -> np.ndarray:
        """The leaf holding each finest cell, named by its lower corner in units; ``-1`` outside.

        Parameters
        ----------
        cells : np.ndarray of int, shape ``(m, 3)``

        Returns
        -------
        np.ndarray of int, shape ``(m,)``
        """
        cells = np.asarray(cells, dtype=np.int64)
        within = np.all((cells >= 0) & (cells < self.extent), axis=1)
        ordered, order = self._sorted_corners
        found = np.full(cells.shape[0], -1, dtype=np.int64)
        for level in np.unique(self.level):
            shift = int(self.depth - level)
            candidate = self.keys(np.where(within[:, None], (cells >> shift) << shift, 0))
            position = np.minimum(np.searchsorted(ordered, candidate), ordered.size - 1)
            leaf = order[position]
            hit = within & (ordered[position] == candidate) & (self.level[leaf] == level)
            found[hit] = leaf[hit]
        return found

    def locate(self, points) -> np.ndarray:
        """The leaf holding each point in space, ``(n,)``; ``-1`` for a point in no leaf.

        A point on a face two leaves share is given to the leaf on its upper side, except on the
        box's own upper faces, where it is given to the leaf below.
        """
        units = self.units(np.asarray(points, dtype=float).reshape(-1, 3))
        extent = self.extent
        outside = np.any((units < 0.0) | (units > extent), axis=-1)
        cells = np.minimum(np.floor(np.clip(units, 0.0, extent)).astype(np.int64), extent - 1)
        return np.where(outside, -1, self.leaf_at(cells))

    # -- lattice points ---------------------------------------------------------------------------

    def span_at(self, level) -> np.ndarray:
        """The edge, in units, of a cube at each level."""
        return np.left_shift(np.int64(1), self.depth - np.asarray(level, dtype=np.int64))

    def position(self, units) -> np.ndarray:
        """Where each point given in units (``(..., 3)``, integer or not) lies in space."""
        return self.origin + self.unit * np.asarray(units, dtype=float)

    def units(self, points) -> np.ndarray:
        """Each point's coordinates in units from the origin, unrounded, ``(..., 3)``."""
        return (np.asarray(points, dtype=float) - self.origin) / self.unit

    def point_level(self, units) -> np.ndarray:
        """The coarsest level whose cubes have corners at each lattice point, ``(...)``.

        A point is a corner of level-``l`` cubes exactly when each of its coordinates is a multiple of
        their edge, ``2**(depth - l)``.
        """
        units = np.asarray(units, dtype=np.int64)
        level = np.full(units.shape[:-1], self.depth, dtype=np.int64)
        for coarser in range(self.depth - 1, -1, -1):
            on = np.all(units % self.span_at(coarser) == 0, axis=-1)
            level = np.where(on, coarser, level)
        return level

    @staticmethod
    def keys(coordinates) -> np.ndarray:
        """One integer per point of non-negative integer coordinates, ordered as the points are by
        ``(x, y, z)``; shape ``(...)`` from ``(..., 3)``."""
        coordinates = np.asarray(coordinates, dtype=np.int64)
        return (
            (coordinates[..., 0] << (2 * BITS))
            | (coordinates[..., 1] << BITS)
            | coordinates[..., 2]
        )

    @staticmethod
    def coordinates(keys) -> np.ndarray:
        """The integer coordinates :meth:`keys` encoded; shape ``(..., 3)``."""
        keys = np.asarray(keys, dtype=np.int64)
        mask = (np.int64(1) << BITS) - 1
        return np.stack([keys >> (2 * BITS), (keys >> BITS) & mask, keys & mask], axis=-1)

    def _check_span(self) -> None:
        if np.any(self.shape << self.depth >= 1 << (BITS - 1)):
            raise ValueError(
                f"the tree spans {int(np.max(self.shape << self.depth))} of its smallest cubes "
                f"along an axis; at most {(1 << (BITS - 1)) - 1} are supported"
            )
