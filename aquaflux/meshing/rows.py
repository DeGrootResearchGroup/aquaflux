"""How a straight pipe's rows are spread along it: graded away from what it joins.

The rows of a pipe's wall grid (and so the O-grid's cross-sections) are by default about as long as
the cells round the pipe are wide, all along it. In a long straight length that is more rows than
the flow needs: the flow develops slowly there, and changes fast only near what the pipe joins -- a
bend, another pipe, a vessel. :class:`PipeGrading` keeps the rows that short at each end of a
straight length that joins something and lets them grow geometrically away from it, toward an open
end (an inlet or an outlet, where nothing joins) or toward the length's middle, up to a longest
row.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

__all__ = ["PipeGrading"]


@dataclasses.dataclass(frozen=True)
class PipeGrading:
    """Rows along a straight pipe, growing away from each end that joins something.

    Attributes
    ----------
    growth : float
        The length ratio of neighbouring rows, ``>= 1``: 1 keeps every row the rims' spacing.
    max_length : float or None
        The longest a row may grow, in metres; unset, no limit.

    Raises
    ------
    ValueError
        If ``growth`` is below 1 or not finite, or ``max_length`` is not positive.
    """

    growth: float
    max_length: float | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.growth) and self.growth >= 1.0):
            raise ValueError(f"PipeGrading.growth must be at least 1, got {self.growth!r}")
        if self.max_length is not None and not self.max_length > 0.0:
            raise ValueError(f"PipeGrading.max_length must be positive, got {self.max_length!r}")

    def fractions(self, length: float, spacing: float, fine: tuple[bool, bool]) -> np.ndarray:
        """The rows' boundaries along a length, as fractions of it from its start.

        Rows start ``spacing`` long at each end marked ``fine`` and grow by :attr:`growth` per row
        away from it, never past :attr:`max_length` (nor shorter than ``spacing``); with both ends
        fine they grow toward the middle, and with neither every row is ``spacing`` long. They are
        then scaled together to fill the length exactly, by less than one row's share.

        Parameters
        ----------
        length : float
            The length to divide.
        spacing : float
            The shortest row: the rims' own spacing.
        fine : tuple of bool
            Whether the start and the end each join something.

        Returns
        -------
        np.ndarray, shape ``(rows + 1,)``
            Increasing from exactly 0 to exactly 1.
        """
        if not any(fine) or self.growth == 1.0:
            rows = max(1, round(length / spacing))
            return np.linspace(0.0, 1.0, rows + 1)
        longest = max(spacing, self.max_length if self.max_length is not None else math.inf)
        if all(fine):
            half = self._grown(0.5 * length, spacing, longest)
            sizes = np.concatenate([half, half[::-1]])
        else:
            sizes = self._grown(length, spacing, longest)
            if fine[1]:
                sizes = sizes[::-1]
        fractions = np.concatenate([[0.0], np.cumsum(sizes)]) / sizes.sum()
        fractions[-1] = 1.0
        return fractions

    def _grown(self, length: float, spacing: float, longest: float) -> np.ndarray:
        """Row lengths from ``spacing``, each ``growth`` times the last up to ``longest``, as many
        as come nearest to filling ``length`` (at least one)."""
        sizes, total = [], 0.0
        size = spacing
        while total < length:
            sizes.append(size)
            total += size
            size = min(size * self.growth, longest)
        # Drop the last row when the length is nearer without it.
        if len(sizes) > 1 and total - length > 0.5 * sizes[-1]:
            sizes.pop()
        return np.asarray(sizes)
