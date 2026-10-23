"""Profiles against radius as a comma-separated-values (CSV) file: a developed pipe flow, kept.

A fully developed pipe flow is a few functions of the distance from the pipe's axis, and is computed
once and then admitted through the inlets of other cases. :func:`write_profile_table` keeps it as a
small text file -- a ``radius`` column, then one column per field, every number written to its full
precision -- that a reader and a program alike can open, and :func:`read_profile_table` reads it
back exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np

__all__ = ["ProfileTable", "read_profile_table", "write_profile_table"]

#: The first column's name.
RADIUS = "radius"


class ProfileTable(NamedTuple):
    """Fields tabulated against radius.

    Attributes
    ----------
    radii : np.ndarray, shape ``(n,)``
        Increasing.
    columns : dict of {str: np.ndarray}
        Each field's value at each radius, shape ``(n,)``, in the file's order.
    """

    radii: np.ndarray
    columns: dict[str, np.ndarray]


def write_profile_table(
    path: str | Path, radii: np.ndarray, columns: Mapping[str, np.ndarray], *, comment: str = ""
) -> Path:
    """Write fields tabulated against radius as a CSV file.

    Parameters
    ----------
    path : str or path-like
        The file.
    radii : np.ndarray, shape ``(n,)``
        Increasing.
    columns : mapping of {str: np.ndarray}
        Each field's values, shape ``(n,)``, by name.
    comment : str, optional
        Lines written first, each behind ``#``: what the table is of.

    Returns
    -------
    pathlib.Path
        The file written.

    Raises
    ------
    ValueError
        If the radii do not increase, a column does not match them, or a name is ``radius``, holds a
        comma or is given no column.
    """
    radii = np.asarray(radii, dtype=np.float64)
    if radii.ndim != 1 or np.any(np.diff(radii) <= 0.0):
        raise ValueError("a profile table's radii must be one increasing row of numbers.")
    names = list(columns)
    bad = [name for name in names if name == RADIUS or "," in name or not name]
    if bad:
        raise ValueError(f"a profile table cannot name a column {bad[0]!r}.")
    values = [np.asarray(columns[name], dtype=np.float64) for name in names]
    for name, column in zip(names, values, strict=True):
        if column.shape != radii.shape:
            raise ValueError(
                f"column {name!r} holds {column.shape} values for {radii.shape} radii."
            )
    path = Path(path)
    lines = [f"# {line}" for line in comment.splitlines()]
    lines.append(",".join([RADIUS, *names]))
    for row in np.column_stack([radii, *values]):
        lines.append(",".join(repr(float(value)) for value in row))
    path.write_text("\n".join(lines) + "\n")
    return path


def read_profile_table(path: str | Path) -> ProfileTable:
    """Read a table :func:`write_profile_table` wrote.

    Parameters
    ----------
    path : str or path-like

    Returns
    -------
    ProfileTable

    Raises
    ------
    ValueError
        If the file's first column is not ``radius``, a row holds the wrong number of values, or the
        radii do not increase.
    """
    rows = [
        line for line in Path(path).read_text().splitlines() if line and not line.startswith("#")
    ]
    if not rows:
        raise ValueError(f"{path}: no profile table in the file.")
    header = rows[0].split(",")
    if header[0] != RADIUS:
        raise ValueError(
            f"{path}: a profile table's first column is {RADIUS!r}, got {header[0]!r}."
        )
    values = []
    for number, line in enumerate(rows[1:], start=2):
        row = line.split(",")
        if len(row) != len(header):
            raise ValueError(
                f"{path}: row {number} holds {len(row)} values under {len(header)} columns."
            )
        values.append([float(value) for value in row])
    table = np.asarray(values, dtype=np.float64).reshape(-1, len(header))
    radii = table[:, 0]
    if radii.size == 0 or np.any(np.diff(radii) <= 0.0):
        raise ValueError(f"{path}: a profile table's radii must increase.")
    return ProfileTable(radii, {name: table[:, i + 1] for i, name in enumerate(header[1:])})
