"""Fixed-width ASCII tables for solver reports.

A long solve reports repeating, columnar data -- one row per inner iteration, per equation, per patch.
Printed as free-form ``key=value`` text it is readable one line at a time but not *scannable*: nothing
lines up, so a reader cannot follow one quantity down the run, which is exactly how a trend (a rate
creeping toward 1, a cost column climbing) is spotted.

:class:`TextTable` renders such data as a ruled, fixed-width table. It is built for **streaming**: the
rule, the heading row and each data row are separate methods returning one line each, so a caller emits
rows as results arrive rather than collecting them and formatting at the end. That is what lets a table
appear in a log file that is being tailed while the solve runs.

This module holds no numerics and imports nothing from the rest of the package, so any subsystem may
format a report with it.

Examples
--------
>>> table = TextTable([Column("iter", 4, "d"), Column("residual", 10, ".3e")])
>>> print(table.rule("sweep 1"))
+- sweep 1 ---------+
>>> print(table.headings())
| iter |   residual |
>>> print(table.rule())
+------+------------+
>>> print(table.row((3, 1.25e-4)))
|    3 |  1.250e-04 |
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple

__all__ = ["Column", "TextTable"]


class Column(NamedTuple):
    """One column of a :class:`TextTable`.

    Attributes
    ----------
    heading : str
        The column's label in the heading row.
    width : int
        The field width in characters, excluding the single space of padding either side. Choose it
        wide enough for the formatted values: an over-wide value is **not** truncated (see
        :meth:`TextTable.row`).
    spec : str
        A :func:`format` specification applied to each value, e.g. ``"d"``, ``".3e"``, ``".2f"``.
        Empty (the default) formats with ``str``, for a text column.
    align : str
        The fill alignment character -- ``">"`` (right, the default, correct for numbers), ``"<"``
        (left, for text) or ``"^"`` (centred). Applies to both the values and the heading.
    """

    heading: str
    width: int
    spec: str = ""
    align: str = ">"


class TextTable:
    """A ruled, fixed-width ASCII table rendered one line at a time.

    Each method returns a single line as a string rather than writing it, so the caller owns the
    stream and can interleave table lines with other output.

    Parameters
    ----------
    columns : sequence of Column
        The column definitions, in order. Must be non-empty.

    Raises
    ------
    ValueError
        If ``columns`` is empty.

    Examples
    --------
    >>> table = TextTable([Column("name", 6, align="<"), Column("value", 7, ".2f")])
    >>> print(table.row(("alpha", 0.5)))
    | alpha  |    0.50 |
    """

    def __init__(self, columns: Sequence[Column]) -> None:
        if not columns:
            raise ValueError("a TextTable needs at least one column")
        self._columns = tuple(columns)

    @property
    def width(self) -> int:
        """The rendered line length in characters (every line this table returns has it)."""
        # Each column contributes its width plus the two padding spaces; the separators are one more
        # than the number of columns (the leading and trailing `|`, and one between each pair).
        return sum(column.width + 2 for column in self._columns) + len(self._columns) + 1

    def rule(self, title: str | None = None) -> str:
        """A horizontal rule, optionally carrying a title.

        The untitled form is segmented at the column boundaries (``+----+------+``), so it reads as
        part of the grid. The titled form is a single unsegmented span (``+- title ------+``) used to
        open a block: the title is free text, so aligning it to the columns would be meaningless.

        Parameters
        ----------
        title : str or None
            Text to embed in the rule. ``None`` (default) gives the plain segmented rule.

        Returns
        -------
        str
            One line of :attr:`width` characters -- longer only if ``title`` does not fit, which
            widens the rule rather than cutting the title.
        """
        if title is None:
            return "+" + "+".join("-" * (column.width + 2) for column in self._columns) + "+"
        return "+" + f"- {title} ".ljust(self.width - 2, "-") + "+"

    def spanning(self, text: str) -> str:
        """One row of free text spanning every column, for a note that belongs inside the grid.

        Used for values that are *not* per-column -- a set of secondary readings, a status line -- so
        they sit within the table's borders instead of interrupting it. Over-long text widens the row
        rather than being cut, for the same reason as :meth:`row`.
        """
        return "| " + text.ljust(self.width - 4) + " |"

    def headings(self) -> str:
        """The heading row, each label formatted in its own column's width and alignment."""
        return self._render(column.heading for column in self._columns)

    def row(self, values: Iterable[Any]) -> str:
        """One data row: each value formatted by its column's ``spec``, ``width`` and ``align``.

        Parameters
        ----------
        values : iterable
            One value per column, in column order. Extra values raise; missing ones raise.

        Returns
        -------
        str
            The rendered row. A value whose formatted text exceeds its column ``width`` **widens that
            row** rather than being truncated: a cut-off number is a wrong number, so a misaligned row
            is the safer failure -- and it is visible, which prompts a wider column.

        Raises
        ------
        ValueError
            If the number of values does not match the number of columns.
        """
        values = tuple(values)
        if len(values) != len(self._columns):
            raise ValueError(
                f"expected {len(self._columns)} values for columns "
                f"{[column.heading for column in self._columns]}, got {len(values)}"
            )
        return self._render(
            format(value, column.spec) for value, column in zip(values, self._columns, strict=True)
        )

    def _render(self, cells: Iterable[str]) -> str:
        """Pad each already-formatted cell into its column and join them with the grid separators."""
        padded = [
            format(cell, f"{column.align}{column.width}")
            for cell, column in zip(cells, self._columns, strict=True)
        ]
        return "| " + " | ".join(padded) + " |"
