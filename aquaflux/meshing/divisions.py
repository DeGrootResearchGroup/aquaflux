"""How a block's cells are spread along one of its directions.

A direction's division is one of three things: a cell count (cells of equal size), a
:class:`Graded` progression (cells growing geometrically away from one end or both), or the node
fractions themselves (any increasing sequence from 0 to 1). Each gives the ``n + 1`` node
fractions the transfinite interpolation places its nodes at. A count and a :class:`Graded` can also
be scaled to more or fewer cells; explicit fractions only by a whole factor, each cell split evenly.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from aquaflux.mesh import graded_nodes

__all__ = ["Graded", "cell_counts"]

#: Which end a :class:`Graded` division's smallest cells sit at.
_ENDS = ("start", "end", "both")


@dataclasses.dataclass(frozen=True)
class Graded:
    """Cells growing geometrically away from the start, the end, or both ends of a direction.

    Attributes
    ----------
    cells : int
    growth : float
        The size ratio of adjacent cells, ``> 0``: ``> 1`` puts the smallest cells at ``toward``.
    toward : str
        ``"start"``, ``"end"`` or ``"both"``. Unset, ``"start"``.

    Raises
    ------
    ValueError
        If ``cells`` is not a positive integer, ``growth`` is not positive and finite, or
        ``toward`` is not one of the three ends.
    """

    cells: int
    growth: float
    toward: str = "start"

    def __post_init__(self) -> None:
        if int(self.cells) != self.cells or self.cells < 1:
            raise ValueError(f"Graded.cells must be a positive integer, got {self.cells!r}")
        if not (math.isfinite(self.growth) and self.growth > 0.0):
            raise ValueError(f"Graded.growth must be positive and finite, got {self.growth!r}")
        if self.toward not in _ENDS:
            raise ValueError(f"Graded.toward must be one of {_ENDS}, got {self.toward!r}")

    def fractions(self) -> np.ndarray:
        """The ``cells + 1`` node fractions, increasing from 0 to 1."""
        nodes = graded_nodes(self.cells, 1.0, self.growth, both_sides=self.toward == "both")
        if self.toward == "end":
            nodes = 1.0 - nodes[::-1]
        nodes[0], nodes[-1] = 0.0, 1.0
        return nodes

    def scaled(self, factor: float) -> Graded:
        """The progression with ``factor`` times the cells, the same ratio of its end cells.

        One-sided, the largest cell over the smallest, ``growth ** (cells - 1)``, is kept exactly.
        """
        cells = max(1, round(self.cells * factor))
        growth = self.growth ** ((self.cells - 1) / (cells - 1)) if cells > 1 else 1.0
        return Graded(cells, growth, self.toward)

    def reversed(self) -> Graded:
        """The same progression read from the other end."""
        toward = {"start": "end", "end": "start", "both": "both"}[self.toward]
        return Graded(self.cells, self.growth, toward)


def fractions_of(division) -> np.ndarray:
    """The node fractions a division gives: a count, a :class:`Graded`, or the fractions.

    Raises
    ------
    ValueError
        If a count is not a positive integer, or fractions do not increase from exactly 0 to 1.
    """
    if isinstance(division, Graded):
        return division.fractions()
    if np.ndim(division) == 0:
        if int(division) != division or division < 1:
            raise ValueError(f"a cell count must be a positive integer, got {division!r}")
        return np.linspace(0.0, 1.0, int(division) + 1)
    fractions = np.asarray(division, dtype=np.float64)
    if (
        fractions.ndim != 1
        or fractions.size < 2
        or fractions[0] != 0.0
        or fractions[-1] != 1.0
        or np.any(np.diff(fractions) <= 0.0)
    ):
        raise ValueError("node fractions must increase from exactly 0 to exactly 1")
    return fractions


def scaled_division(division, factor: float):
    """A division with ``factor`` times its cells (at least one).

    Raises
    ------
    ValueError
        If the division is explicit fractions and ``factor`` is not a whole number.
    """
    if isinstance(division, Graded):
        return division.scaled(factor)
    if np.ndim(division) == 0:
        return max(1, round(int(division) * factor))
    if factor != round(factor) or factor < 1:
        raise ValueError(
            f"explicit node fractions can only be scaled by a whole factor, got {factor!r}"
        )
    fractions = fractions_of(division)
    split = np.linspace(0.0, 1.0, int(factor) + 1)[:-1]
    inside = (fractions[:-1, None] + np.diff(fractions)[:, None] * split).ravel()
    return np.append(inside, 1.0)


def reversed_division(division):
    """The division read from its other end."""
    if isinstance(division, Graded):
        return division.reversed()
    if np.ndim(division) == 0:
        return division
    return 1.0 - fractions_of(division)[::-1]


def cell_counts(coordinates, size: float) -> list[int]:
    """Cell counts giving cells about ``size`` long between successive coordinates.

    Parameters
    ----------
    coordinates : array-like, shape ``(n + 1,)``
        Increasing positions of the block boundaries along one direction.
    size : float
        The target cell length, ``> 0``.

    Returns
    -------
    list of int, length ``n``
        Each interval's length over ``size``, rounded, at least one: a list, so a lattice reads it
        as one division per interval.
    """
    if not size > 0.0:
        raise ValueError(f"the target cell size must be positive, got {size!r}")
    lengths = np.diff(np.asarray(coordinates, dtype=np.float64))
    return [int(n) for n in np.maximum(1, np.round(lengths / size))]
