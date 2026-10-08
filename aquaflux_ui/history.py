"""A run's convergence history, read from the comma-separated-values (CSV) file a march writes.

The solver writes one row per step of its march (``history.csv`` in a run's output directory), every
number at full precision; this reads it back into one array per column. It reads with the standard
library alone and does not need the solver installed, so a history can be inspected anywhere the
file can be copied.
"""

from __future__ import annotations

import csv
import dataclasses
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

__all__ = ["ConvergenceHistory"]

#: The column a history is indexed by: the step's 1-based count over the whole march.
STEP = "step"

#: The columns a convergence plot offers, in order of preference, with their axis labels. The
#: residual ratio comes first because it is on the same scale for every segment of a march.
RESIDUAL_COLUMNS: Mapping[str, str] = {
    "residual_ratio": "Residual / initial residual",
    "residual_norm": "Residual",
}


@dataclasses.dataclass(frozen=True)
class ConvergenceHistory:
    """The per-step record of one march, one array per column.

    Attributes
    ----------
    columns : mapping of {str: np.ndarray}
        Each column by its header name, every array of shape ``(n_steps,)``. A column whose every
        value reads as a number is ``float64``; any other is kept as strings.
    source : pathlib.Path or None
        The file it was read from.

    Raises
    ------
    ValueError
        If the columns are not all the same length, or there is no ``step`` column.
    """

    columns: Mapping[str, np.ndarray]
    source: Path | None = None

    def __post_init__(self) -> None:
        if STEP not in self.columns:
            raise ValueError(
                f"a convergence history needs a {STEP!r} column; got {list(self.columns)}."
            )
        lengths = {name: len(values) for name, values in self.columns.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"a convergence history's columns differ in length: {lengths}.")

    @classmethod
    def read(cls, path: str | os.PathLike[str]) -> ConvergenceHistory:
        """Read a history file.

        Parameters
        ----------
        path : path-like
            The CSV file: a header row, then one row per step.

        Returns
        -------
        ConvergenceHistory
            Its columns. An empty file, or one holding only its header, has zero steps.

        Raises
        ------
        ValueError
            If a row has a different number of cells from the header, or there is no ``step``
            column.
        """
        source = Path(path)
        with source.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        # The solver writes the header with the first step, so an empty file is a march that has not
        # finished one yet -- or a solve that takes no march steps at all.
        if not rows:
            return cls({STEP: np.zeros(0)}, source)
        header, body = rows[0], rows[1:]
        # A run still being written can end in a partly written line; anything shorter than the header
        # is that, and is left for the next read rather than refused.
        if body and len(body[-1]) < len(header):
            body = body[:-1]
        for number, row in enumerate(body, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"{source}, line {number}: {len(row)} cells where the header names {len(header)}."
                )
        columns = {name: _column([row[index] for row in body]) for index, name in enumerate(header)}
        return cls(columns, source)

    @property
    def n_steps(self) -> int:
        """How many steps it records."""
        return len(self.columns[STEP])

    @property
    def steps(self) -> np.ndarray:
        """The step counts, shape ``(n_steps,)``."""
        return self.columns[STEP]

    def residual_columns(self) -> dict[str, str]:
        """The residual columns this history holds, by name, with their axis labels.

        Returns
        -------
        dict of {str: str}
            In order of preference: the residual relative to where the march began, then the
            residual itself. Empty if it holds neither.
        """
        return {name: label for name, label in RESIDUAL_COLUMNS.items() if name in self.columns}


def _column(cells: list[str]) -> np.ndarray:
    """One column's cells as numbers when every one reads as a number, else as the strings."""
    try:
        return np.asarray([float(cell) for cell in cells], dtype=float)
    except ValueError:
        return np.asarray(cells, dtype=str)
