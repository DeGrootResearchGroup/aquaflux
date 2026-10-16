"""The octree: a box of cubic root cells, each split into eight halves as often as it is asked.

Every position the tree can name lies on one integer **lattice**, the corners of the finest cells it
could ever hold. A cell at level ``l`` spans ``2**(max_level - l)`` lattice steps along each axis, so
its corners, its edges and the lattice point at its centre are all integer coordinates, and two
cells share a corner exactly when they name the same integer triple. Nothing is compared as a float.

A **node** is a lattice point that is a corner of some cell the tree holds or has held. Its *level*
is the coarsest level whose cells have corners there -- the level at which it first appears -- and
is read off its coordinates alone: the coarsest lattice spacing all three divide.

The leaves are stored as two arrays, a level and a low corner per leaf. Splitting replaces a leaf by
its eight children; nothing keeps the interior cells, because nothing asks for them.
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np

#: The eight corners of a cell, as offsets from its low corner in units of its size. Ordered so that
#: corner ``c`` has offset ``((c >> 2) & 1, (c >> 1) & 1, c & 1)``.
CORNERS = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.int64)

#: The twelve edges of a cell, as pairs of corners in :data:`CORNERS` order, the lower corner first.
EDGES = np.array(
    [
        (a, b)
        for a in range(8)
        for b in range(8)
        if a < b and np.abs(CORNERS[a] - CORNERS[b]).sum() == 1
    ],
    dtype=np.int64,
)


@dataclasses.dataclass(frozen=True, eq=False)
class Octree:
    """Leaves of an octree over a box of cubic root cells.

    Build one with :meth:`covering`, then :meth:`split` it.

    Attributes
    ----------
    origin : np.ndarray, shape ``(3,)``
        The low corner of the box: lattice point ``(0, 0, 0)``.
    spacing : float
        The root cells' edge length.
    shape : np.ndarray of int, shape ``(3,)``
        Root cells along each axis.
    max_level : int
        The deepest level a cell may reach; the finest cells have edge ``spacing / 2**max_level``,
        which is also the lattice step.
    levels : np.ndarray of int, shape ``(n_leaves,)``
        Each leaf's level, ``0`` for a root cell.
    low : np.ndarray of int, shape ``(n_leaves, 3)``
        Each leaf's low corner, in lattice steps.
    """

    origin: np.ndarray
    spacing: float
    shape: np.ndarray
    max_level: int
    levels: np.ndarray
    low: np.ndarray

    @classmethod
    def covering(cls, low, high, spacing: float, max_level: int) -> Octree:
        """The root cells of edge ``spacing`` covering the box from ``low`` to ``high``.

        Parameters
        ----------
        low, high : array_like, shape ``(3,)``
            Opposite corners of the box to cover.
        spacing : float
            The root cells' edge length.
        max_level : int
            The deepest level a cell may be split to; at least zero.

        Returns
        -------
        Octree
            Every root cell a leaf, ordered by its integer coordinates.

        Raises
        ------
        ValueError
            If the box is empty, the spacing is not positive, or ``max_level`` is negative or so deep
            that the lattice cannot be numbered in 64 bits.
        """
        low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
        if spacing <= 0.0:
            raise ValueError(f"Octree spacing must be positive, got {spacing!r}.")
        if max_level < 0:
            raise ValueError(f"Octree max_level must be at least zero, got {max_level!r}.")
        if np.any(high <= low):
            raise ValueError(f"Octree box is empty: low {low.tolist()}, high {high.tolist()}.")
        shape = np.maximum(np.ceil((high - low) / spacing - 1e-9).astype(np.int64), 1)
        nodes_per_axis = shape * 2**max_level + 1
        if np.log2(nodes_per_axis.astype(float)).sum() + np.log2(max_level + 1) > 62:
            raise ValueError(
                f"Octree: {shape.tolist()} root cells at max_level {max_level} have too many lattice "
                "points to number in 64 bits; use a coarser root spacing or fewer levels."
            )
        cells = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"), axis=-1)
        step = 2**max_level
        return cls(
            origin=low,
            spacing=float(spacing),
            shape=shape,
            max_level=int(max_level),
            levels=np.zeros(int(np.prod(shape)), dtype=np.int64),
            low=cells.reshape(-1, 3) * step,
        )

    # -- the lattice ------------------------------------------------------------------------------

    @property
    def n_leaves(self) -> int:
        """How many leaves the tree has."""
        return len(self.levels)

    @property
    def step(self) -> float:
        """The lattice step: the finest cell's edge length."""
        return self.spacing / 2**self.max_level

    @property
    def lattice_shape(self) -> np.ndarray:
        """Lattice points along each axis."""
        return self.shape * 2**self.max_level + 1

    def size(self, levels) -> np.ndarray:
        """Edge length, in lattice steps, of a cell at each level."""
        return np.left_shift(np.int64(1), self.max_level - np.asarray(levels, dtype=np.int64))

    def key(self, lattice) -> np.ndarray:
        """The integer key of each lattice point, shape ``(...)`` from ``(..., 3)``."""
        lattice = np.asarray(lattice, dtype=np.int64)
        n = self.lattice_shape
        return (lattice[..., 0] * n[1] + lattice[..., 1]) * n[2] + lattice[..., 2]

    def unkey(self, keys) -> np.ndarray:
        """The lattice point, shape ``(..., 3)``, of each key."""
        keys = np.asarray(keys, dtype=np.int64)
        n = self.lattice_shape
        return np.stack([keys // (n[1] * n[2]), (keys // n[2]) % n[1], keys % n[2]], axis=-1)

    def position(self, lattice) -> np.ndarray:
        """Where each lattice point (``(..., 3)`` integers, or fractions of a step) is, in space."""
        return self.origin + self.step * np.asarray(lattice, dtype=float)

    def lattice(self, points) -> np.ndarray:
        """Each point's position in lattice steps from :attr:`origin`, unrounded, ``(..., 3)``."""
        return (np.asarray(points, dtype=float) - self.origin) / self.step

    def node_level(self, lattice) -> np.ndarray:
        """The coarsest level whose cell corners include each lattice point, shape ``(...)``.

        A point is a corner of level-``l`` cells exactly when each of its coordinates is a multiple of
        their size, ``2**(max_level - l)``; the root lattice (level zero) is the coarsest.
        """
        lattice = np.asarray(lattice, dtype=np.int64)
        level = np.full(lattice.shape[:-1], self.max_level, dtype=np.int64)
        for depth in range(self.max_level - 1, -1, -1):
            size = np.int64(1) << (self.max_level - depth)
            on = np.all(lattice % size == 0, axis=-1)
            level = np.where(on, depth, level)
        return level

    # -- the leaves -------------------------------------------------------------------------------

    def corners(self, rows=None) -> np.ndarray:
        """The lattice points of each leaf's eight corners, ``(n, 8, 3)`` in :data:`CORNERS` order."""
        rows = slice(None) if rows is None else rows
        size = self.size(self.levels[rows])
        return self.low[rows][:, None, :] + size[:, None, None] * CORNERS[None, :, :]

    def centres(self, rows=None) -> np.ndarray:
        """Each leaf's centre, in space, ``(n, 3)``."""
        rows = slice(None) if rows is None else rows
        size = self.size(self.levels[rows])
        return self.position(self.low[rows] + 0.5 * size[:, None])

    def cell_spacing(self, rows=None) -> np.ndarray:
        """Each leaf's edge length, in space, ``(n,)``."""
        rows = slice(None) if rows is None else rows
        return self.step * self.size(self.levels[rows])

    def split(self, marked) -> tuple[Octree, np.ndarray]:
        """The tree with the marked leaves each replaced by their eight children.

        Parameters
        ----------
        marked : np.ndarray of bool, shape ``(n_leaves,)``
            Leaves to split. A leaf already at :attr:`max_level` must not be marked.

        Returns
        -------
        tree : Octree
            The unmarked leaves in their order, then the children, eight per split leaf in the order
            the leaves were marked and in :data:`CORNERS` order within each.
        children : np.ndarray of int, shape ``(n_split, 8)``
            The rows of each split leaf's children in the new tree.

        Raises
        ------
        ValueError
            If a marked leaf is already at :attr:`max_level`.
        """
        marked = np.asarray(marked, dtype=bool)
        if np.any(self.levels[marked] >= self.max_level):
            raise ValueError(
                f"Octree.split: {int(np.sum(self.levels[marked] >= self.max_level))} marked leaves are "
                f"already at max_level {self.max_level}."
            )
        kept = ~marked
        parents = np.flatnonzero(marked)
        half = self.size(self.levels[parents] + 1)
        child_low = self.low[parents][:, None, :] + half[:, None, None] * CORNERS[None, :, :]
        child_level = np.repeat(self.levels[parents] + 1, 8)
        tree = dataclasses.replace(
            self,
            levels=np.concatenate([self.levels[kept], child_level]),
            low=np.concatenate([self.low[kept], child_low.reshape(-1, 3)]),
        )
        children = int(kept.sum()) + np.arange(8 * len(parents)).reshape(-1, 8)
        return tree, children

    def subset(self, rows) -> Octree:
        """The tree with only these leaves, in this order -- the cells of a box that the region a
        field lives in can reach, say. Points elsewhere in the box then lie in no leaf."""
        rows = np.asarray(rows, dtype=np.int64)
        return dataclasses.replace(self, levels=self.levels[rows], low=self.low[rows])

    @functools.cached_property
    def _codes(self) -> tuple[np.ndarray, np.ndarray]:
        """Each leaf's code (its low corner's key and its level, as one integer), sorted, and the
        leaf each sorted code belongs to."""
        codes = self.key(self.low) * (self.max_level + 1) + self.levels
        order = np.argsort(codes)
        return codes[order], order

    def locate(self, points) -> np.ndarray:
        """The leaf holding each point, ``(n,)``; ``-1`` for a point in no leaf.

        A point on a face shared by two leaves is given to the one on its high side, except on the
        box's own high faces, where it is given to the leaf below.
        """
        u = self.lattice(points)
        outside = np.any((u < 0.0) | (u > self.lattice_shape - 1), axis=-1)
        u = np.clip(u, 0.0, self.lattice_shape - 1)
        codes, order = self._codes
        found = np.full(len(u), -1, dtype=np.int64)
        for level in range(self.max_level + 1):
            size = self.size(level)
            low = np.minimum(
                np.floor(u / size).astype(np.int64) * size, self.lattice_shape - 1 - size
            )
            code = self.key(low) * (self.max_level + 1) + level
            slot = np.minimum(np.searchsorted(codes, code), len(codes) - 1)
            hit = (codes[slot] == code) & (found < 0)
            found[hit] = order[slot[hit]]
        found[outside] = -1
        return found
