"""How a straight pipe's rows are spread along it: graded away from the ends that join something.

Each case is checked against the progression it should be -- the first row the spacing given, each
next one ``growth`` times the last up to the longest allowed -- and against the length it must fill.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.meshing import PipeGrading


def _rows(fractions, length):
    return np.diff(fractions) * length


def test_rows_grow_from_the_end_that_joins_something_toward_the_open_end():
    rows = _rows(PipeGrading(growth=1.1).fractions(2.0, 0.01, (True, False)), 2.0)
    np.testing.assert_allclose(rows[1:] / rows[:-1], 1.1, rtol=1e-12)
    # Scaled to fill the length by less than one row's share.
    assert rows[0] == pytest.approx(0.01, rel=0.1)
    assert rows.sum() == pytest.approx(2.0)


def test_rows_grow_toward_the_start_when_only_the_end_joins_something():
    rows = _rows(PipeGrading(growth=1.1).fractions(2.0, 0.01, (False, True)), 2.0)
    np.testing.assert_allclose(rows[:-1] / rows[1:], 1.1, rtol=1e-12)


def test_rows_stop_growing_at_the_longest_allowed():
    rows = _rows(PipeGrading(growth=1.2, max_length=0.05).fractions(2.0, 0.01, (True, False)), 2.0)
    assert rows.max() == pytest.approx(0.05, rel=0.05)
    assert np.count_nonzero(np.isclose(rows, rows.max())) > 20
    assert np.all(np.diff(rows) >= -1e-12 * rows.max())


def test_a_length_joined_at_both_ends_grows_toward_its_middle_alike_from_each():
    rows = _rows(PipeGrading(growth=1.1, max_length=0.04).fractions(1.0, 0.01, (True, True)), 1.0)
    np.testing.assert_allclose(rows, rows[::-1], rtol=1e-12)
    assert rows[0] == pytest.approx(0.01, rel=0.1)
    assert rows[len(rows) // 2] == pytest.approx(0.04, rel=0.1)


@pytest.mark.parametrize(
    ("grading", "fine"),
    [(PipeGrading(growth=1.1), (False, False)), (PipeGrading(growth=1.0), (True, False))],
)
def test_rows_are_even_with_no_end_to_grade_from_or_no_growth(grading, fine):
    np.testing.assert_allclose(grading.fractions(1.0, 0.01, fine), np.linspace(0.0, 1.0, 101))


@pytest.mark.parametrize(("growth", "longest"), [(0.9, None), (float("nan"), None), (1.1, 0.0)])
def test_a_shrinking_or_unbounded_grading_is_refused(growth, longest):
    with pytest.raises(ValueError, match="PipeGrading"):
        PipeGrading(growth=growth, max_length=longest)
