"""A structured grid of hexahedra filling a rectangular duct's core, inside rings of points round it.

A duct whose four sides are meshed as structured grids (and its layers, when it has them) leaves a
core: a ring of points round it at each row of the grids, one ring after another along the duct, each
ring four-sided -- ``width`` segments along one side, ``height`` along the next, then the same two
again. Each ring's cross-section is filled with ``width x height`` quadrilaterals by the transfinite
interpolation of its four sides, and each cross-section is joined to the next, so the duct's core is
hexahedra throughout, and its faces round it are the rings' own quadrilaterals, point for point. The
cross-sections follow the rings wherever they go, so a duct bending along a curve is filled as a
straight one is.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .blocks import transfinite
from .conduits import ConduitFill
from .mapped import _fractions

__all__ = ["DuctCore"]


@dataclasses.dataclass(frozen=True)
class DuctCore:
    """Fills the core inside four-sided rings of points with hexahedra, cross-section by section.

    Attributes
    ----------
    junction_rows : int
        At an end where a duct opens into the rest of the region, how many rows of its walls'
        grids the core stops short of the opening, leaving them to the tetrahedral fill.
    """

    junction_rows: int = 2

    def __post_init__(self) -> None:
        if (
            isinstance(self.junction_rows, bool)
            or not isinstance(self.junction_rows, int)
            or (self.junction_rows < 1)
        ):
            raise ValueError(
                f"DuctCore.junction_rows must be a whole number >= 1, got {self.junction_rows!r}"
            )

    def fill(self, points: np.ndarray, rings: np.ndarray, width: int) -> ConduitFill:
        """The hexahedra inside ``rings``.

        Parameters
        ----------
        points : np.ndarray, shape ``(n_points, 3)``
        rings : np.ndarray of int, shape ``(n_rows, 2 (width + height))``
            The points round the core at each cross-section, each ring starting at a corner and
            running ``width`` segments to the next, ``height`` to the next, and the same two again;
            each ring in the same direction, point ``i`` of every ring joined to point ``i`` of
            the next. At least two rows.
        width : int
            The segments along a ring's first side.

        Returns
        -------
        ConduitFill
            Its cross-sections' quadrilaterals ``width x height`` each, and a hexahedron for each
            between one cross-section and the next.

        Raises
        ------
        ValueError
            If the rings are fewer than two, or do not have four sides of at least one segment.
        """
        points = np.asarray(points, dtype=np.float64)
        rings = np.asarray(rings, dtype=np.int64)
        n_rows, n = rings.shape
        height = n // 2 - width
        if n_rows < 2 or n % 2 or width < 1 or height < 1:
            raise ValueError(
                f"a duct's core needs at least two rings of four sides, got {n_rows} of {n} "
                f"points with a first side of {width}"
            )
        sections, new = [], [points]
        base = points.shape[0]
        for ring in rings:
            lattice, added = _section(points, ring, width, height, base)
            sections.append(lattice)
            new.append(added)
            base += added.shape[0]
        quads = np.stack([_section_quads(lattice) for lattice in sections])
        hexahedra = np.concatenate([quads[:-1], quads[1:]], axis=2).reshape(-1, 8)
        return ConduitFill(np.vstack(new), hexahedra, quads[0], quads[-1])


def _rim(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """The lattice positions ``(i, j)`` round a ``(width + 1, height + 1)`` lattice, from its
    corner ``(0, 0)`` along ``i``, then ``j``, then back: the order of a ring's points."""
    i = np.concatenate([
        np.arange(width), np.full(height, width), np.arange(width, 0, -1), np.zeros(height, int),
    ])  # fmt: skip
    j = np.concatenate([
        np.zeros(width, int), np.arange(height), np.full(width, height), np.arange(height, 0, -1),
    ])  # fmt: skip
    return i, j


def _section(points, ring, width, height, base):
    """One cross-section as a lattice of point indices ``(width + 1, height + 1)``, its boundary the
    ring's points and its interior new ones, and those new points."""
    i, j = _rim(width, height)
    nodes = np.zeros((width + 1, height + 1, 3))
    nodes[i, j] = points[ring]
    u = 0.5 * (_fractions(nodes[:, 0], False) + _fractions(nodes[:, -1], False))
    v = 0.5 * (_fractions(nodes[0, :], False) + _fractions(nodes[-1, :], False))
    filled = transfinite(nodes, [u, v])
    lattice = np.empty((width + 1, height + 1), dtype=np.int64)
    lattice[i, j] = ring
    interior = filled[1:-1, 1:-1].reshape(-1, 3)
    lattice[1:-1, 1:-1] = base + np.arange(interior.shape[0]).reshape(width - 1, height - 1)
    return lattice, interior


def _section_quads(lattice: np.ndarray) -> np.ndarray:
    """A cross-section's quadrilaterals, each wound the same way."""
    lower, upper = lattice[:-1], lattice[1:]
    return np.stack([lower[:, :-1], upper[:, :-1], upper[:, 1:], lower[:, 1:]], axis=2).reshape(
        -1, 4
    )
