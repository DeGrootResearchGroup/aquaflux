"""Unit tests for :class:`~aquaflux.text_table.TextTable`.

Pure string formatting, so every test is a direct comparison against the expected line. What is pinned
here is the contract a streaming report depends on: every line is the same width (so the grid holds when
rows are emitted one at a time over minutes), values are never silently truncated, and a row that does
not match the columns fails loudly rather than rendering a misleading table.
"""

from __future__ import annotations

import pytest
from aquaflux.text_table import Column, TextTable


@pytest.fixture
def table() -> TextTable:
    return TextTable(
        [Column("iter", 5, "d"), Column("residual", 9, ".2e"), Column("note", 6, "", "<")]
    )


def test_every_line_has_the_same_width(table: TextTable) -> None:
    """A streaming table is emitted line by line, so the grid only holds if the widths agree."""
    lines = [table.rule(), table.headings(), table.row((1, 1.0e-3, "ok")), table.rule("title")]

    assert {len(line) for line in lines} == {table.width}


def test_values_are_formatted_and_aligned_by_their_column(table: TextTable) -> None:
    assert table.row((7, 1.25e-4, "ok")) == "|     7 |  1.25e-04 | ok     |"


def test_the_plain_rule_is_segmented_and_the_titled_rule_spans(table: TextTable) -> None:
    """The plain rule reads as part of the grid; a title is free text, so aligning it would mislead."""
    assert table.rule() == "+-------+-----------+--------+"
    assert table.rule("step 4") == "+- step 4 -------------------+"


def test_an_over_wide_value_widens_its_row_rather_than_being_truncated() -> None:
    """A cut-off number is a wrong number; a misaligned row is the visible, safer failure."""
    narrow = TextTable([Column("value", 3, ".1f")])

    assert narrow.row((12345.6,)) == "| 12345.6 |"


def test_a_row_that_does_not_match_the_columns_raises(table: TextTable) -> None:
    """Silently padding or dropping a value would render a table whose columns lie about the data."""
    with pytest.raises(ValueError, match="expected 3 values"):
        table.row((1, 2.0))


def test_a_table_needs_at_least_one_column() -> None:
    with pytest.raises(ValueError, match="at least one column"):
        TextTable([])
