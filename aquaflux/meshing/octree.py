"""An octree of cubes over a box, refined where asked and balanced so neighbours differ by one level.

The cubes start as a lattice of equal root cubes covering the box. A refinement criterion says which
cubes to split into eight, and the splitting repeats until none asks; then the tree is balanced:
any two cubes that share a face or a stretch of an edge are at most one level apart, so a cube meets
at most four smaller ones across a face and two along an edge. That bound is what lets a cube next to
smaller ones be cut into a few fixed shapes that match them exactly (see the hexahedral core).

Cubes are held in integer units of the smallest cube the tree has: a cube of level ``l`` in a tree of
depth ``d`` spans ``2**(d - l)`` units. Leaves never overlap, so a leaf is named by its lower corner.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from .regions import Region

__all__ = ["NearSurface", "Octree", "RefinementCriterion"]

#: Bits per axis in a point's key. Coordinates are held at half a unit (see the hexahedral core), so a
#: tree may span at most ``2**(BITS - 1)`` units along an axis.
BITS = 21

#: The eighteen directions to a cube's face and edge neighbours: each offset with one or two of its
#: components non-zero.
FACE_AND_EDGE_DIRECTIONS = np.array(
    [d for d in itertools.product((-1, 0, 1), repeat=3) if 1 <= np.count_nonzero(d) <= 2]
)

#: A cube's eight children's lower corners, in units of half the cube.
CHILD_OFFSETS = np.array(list(itertools.product((0, 1), repeat=3)))


def point_keys(coordinates: np.ndarray) -> np.ndarray:
    """One integer per point of non-negative integer coordinates, ordered as the points are by
    ``(x, y, z)``."""
    coordinates = np.asarray(coordinates, dtype=np.int64)
    return (coordinates[..., 0] << (2 * BITS)) | (coordinates[..., 1] << BITS) | coordinates[..., 2]


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


def cube_samples(lower: np.ndarray, size: np.ndarray, per_edge: int, margin: float) -> np.ndarray:
    """A lattice of ``per_edge`` points along each axis of each cube grown by ``margin`` on every
    side, corners included; shape ``(n, per_edge**3, 3)``."""
    ticks = np.linspace(0.0, 1.0, per_edge)
    unit = np.array(list(itertools.product(ticks, repeat=3)))
    grown = size[:, None, None] + 2.0 * margin
    return lower[:, None, :] - margin + grown * unit[None, :, :]


@dataclasses.dataclass(frozen=True)
class NearSurface:
    """Split every cube a region's boundary passes near until it is no larger than ``size``.

    Attributes
    ----------
    region : Region
        The region (see :class:`~aquaflux.meshing.ClosedSurface`,
        :class:`~aquaflux.meshing.SampledRegion`); a cube is split where its boundary is taken to
        pass through the cube grown by ``margin`` on each side.
    size : float
        The largest cube kept where the boundary passes.
    margin : float
        How far beyond each cube to look for the boundary.
    """

    region: Region
    size: float
    margin: float = 0.0

    def refines(self, lower: np.ndarray, size: np.ndarray) -> np.ndarray:
        large = size > self.size * (1.0 + 1e-9)
        refines = np.zeros(size.shape, dtype=bool)
        if np.any(large):
            low = lower[large] - self.margin
            refines[large] = self.region.meets(low, low + size[large, None] + 2.0 * self.margin)
        return refines


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
    def lattice(cls, lower: Sequence[float], upper: Sequence[float], size: float) -> Octree:
        """Root cubes of edge ``size`` from ``lower``, as many along each axis as cover ``upper``."""
        lower = np.asarray(lower, dtype=np.float64)
        extent = np.asarray(upper, dtype=np.float64) - lower
        if size <= 0.0 or np.any(extent <= 0.0):
            raise ValueError("the box must have positive extent and the cubes a positive size")
        shape = np.maximum(np.ceil(extent / size - 1e-9), 1).astype(np.int64)
        corner = np.array(list(itertools.product(*(range(n) for n in shape))), dtype=np.int64)
        tree = cls(lower, float(size), 0, shape, np.zeros(len(corner), np.int64), corner)
        tree._check_span()
        return tree

    @property
    def span(self) -> np.ndarray:
        """Each leaf's edge, in units; shape ``(n_leaves,)``."""
        return np.int64(1) << (self.depth - self.level)

    @property
    def lower(self) -> np.ndarray:
        """Each leaf's lower corner; shape ``(n_leaves, 3)``."""
        return self.origin + self.unit * self.corner

    @property
    def size(self) -> np.ndarray:
        """Each leaf's edge length; shape ``(n_leaves,)``."""
        return self.unit * self.span

    def refined(self, split: np.ndarray) -> Octree:
        """This tree with each leaf where ``split`` holds replaced by its eight children.

        The children follow the leaves kept, in the order of their parents.
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
        children = tree.corner[split][:, None, :] + half * CHILD_OFFSETS[None, :, :]
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
        within = np.all((cells >= 0) & (cells < self.shape << self.depth), axis=1)
        keys = point_keys(self.corner)
        order = np.argsort(keys)
        ordered = keys[order]
        found = np.full(cells.shape[0], -1, dtype=np.int64)
        for level in np.unique(self.level):
            shift = int(self.depth - level)
            candidate = point_keys(np.where(within[:, None], (cells >> shift) << shift, 0))
            position = np.minimum(np.searchsorted(ordered, candidate), ordered.size - 1)
            leaf = order[position]
            hit = within & (ordered[position] == candidate) & (self.level[leaf] == level)
            found[hit] = leaf[hit]
        return found

    def _check_span(self) -> None:
        if np.any(self.shape << self.depth >= 1 << (BITS - 1)):
            raise ValueError(
                f"the tree spans {int(np.max(self.shape << self.depth))} of its smallest cubes "
                f"along an axis; at most {(1 << (BITS - 1)) - 1} are supported"
            )
