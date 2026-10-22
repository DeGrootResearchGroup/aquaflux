"""A continuous interpolant on an octree's nodes, as a sum of hat functions level by level.

Each node carries a hat: the trilinear tent that is one at the node and falls to zero one cell of
the node's own level away along each axis. The interpolant is ``sum over nodes of surplus * hat``,
and a node's **surplus** is what its value adds to everything coarser:

    surplus(n) = value(n) - (sum of the coarser hats at n)

Three properties follow, and they are why the octree needs no special treatment where a fine cell
meets a coarse one:

- **It passes through every node's value.** At a node of level ``l``, the hats of finer nodes are
  zero (each vanishes on the lattice one level coarser than its own, which holds the node), the hats
  of other level-``l`` nodes are zero (they vanish on each other), and the coarser ones sum to what
  the surplus was measured against.
- **It is continuous everywhere**, being a sum of continuous functions. A node in the middle of a
  coarse cell's face -- one a finer neighbour introduced -- reaches into the coarse cell through its
  hat, so both sides of the face see its value.
- **The surplus is the interpolation error the node corrected.** It is exactly how far the tree
  without the node missed the value there, which is the measure an adaptive refinement asks for.

Adding a node changes no other node's surplus: a node's surplus depends only on coarser hats, and the
new node's hat is zero at every node at its own level or coarser. Nodes added together are taken in
order of level, so a node's surplus sees every coarser node, old or new.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .tree import CORNERS, Octree


@dataclasses.dataclass(frozen=True, eq=False)
class Hierarchy:
    """Surpluses of an octree's nodes, kept per level.

    A node absent from the hierarchy has a surplus of zero: the interpolant there is whatever the
    coarser nodes make it. Build one empty with :meth:`empty` and add nodes with :meth:`with_values`.

    Attributes
    ----------
    keys : tuple of np.ndarray of int
        Per level, the sorted keys (:meth:`Octree.keys`) of the nodes of that level.
    surpluses : tuple of np.ndarray
        Per level, each node's surplus, in the order of ``keys``.
    """

    keys: tuple
    surpluses: tuple

    @classmethod
    def empty(cls, tree: Octree) -> Hierarchy:
        """A hierarchy with no nodes, whose interpolant is zero everywhere."""
        none = tuple(np.zeros(0, dtype=np.int64) for _ in range(tree.depth + 1))
        return cls(keys=none, surpluses=tuple(np.zeros(0) for _ in none))

    @property
    def n_nodes(self) -> int:
        """How many nodes carry a surplus."""
        return sum(len(k) for k in self.keys)

    def at_units(self, tree: Octree, u) -> np.ndarray:
        """The interpolant at points given in units of the tree (:meth:`Octree.units`), ``(n, 3)`` -> ``(n,)``."""
        u = np.asarray(u, dtype=float)
        top = tree.extent
        total = np.zeros(len(u))
        for level, (keys, surplus) in enumerate(zip(self.keys, self.surpluses, strict=True)):
            if len(keys) == 0:
                continue
            size = tree.span_at(level)
            low = np.clip(np.floor(u / size).astype(np.int64) * size, 0, top - size)
            fraction = (u - low) / size
            for offset in CORNERS:
                corner = low + size * offset
                weight = np.prod(np.where(offset == 1, fraction, 1.0 - fraction), axis=-1)
                key = tree.keys(corner)
                slot = np.minimum(np.searchsorted(keys, key), len(keys) - 1)
                hit = keys[slot] == key
                total += np.where(hit, surplus[slot] * weight, 0.0)
        return total

    def __call__(self, tree: Octree, points) -> np.ndarray:
        """The interpolant at points in space, ``(n, 3)`` -> ``(n,)``."""
        return self.at_units(tree, tree.units(points))

    def with_values(self, tree: Octree, keys, values) -> Hierarchy:
        """The hierarchy with these nodes added at these values.

        Parameters
        ----------
        tree : Octree
            The tree whose lattice the keys name.
        keys : np.ndarray of int, shape ``(m,)``
            Nodes not already in the hierarchy, distinct.
        values : np.ndarray, shape ``(m,)``
            What the interpolant must equal at each.

        Returns
        -------
        Hierarchy

        Raises
        ------
        ValueError
            If a key is already in the hierarchy or repeated.
        """
        keys = np.asarray(keys, dtype=np.int64)
        values = np.asarray(values, dtype=float)
        if len(np.unique(keys)) != len(keys):
            raise ValueError("Hierarchy.with_values: a node is given twice.")
        lattice = tree.coordinates(keys)
        point_level = tree.point_level(lattice)
        new_keys, new_surpluses = list(self.keys), list(self.surpluses)
        current = self
        for level in np.unique(point_level):
            rows = np.flatnonzero(point_level == level)
            existing = new_keys[level]
            if np.isin(keys[rows], existing).any():
                raise ValueError(
                    "Hierarchy.with_values: a node already in the hierarchy is given again."
                )
            surplus = values[rows] - current.at_units(tree, lattice[rows])
            merged = np.concatenate([existing, keys[rows]])
            order = np.argsort(merged)
            new_keys[level] = merged[order]
            new_surpluses[level] = np.concatenate([new_surpluses[level], surplus])[order]
            current = Hierarchy(keys=tuple(new_keys), surpluses=tuple(new_surpluses))
        return current
