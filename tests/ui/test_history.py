"""Tests for :class:`~aquaflux_ui.ConvergenceHistory`, reading the file the solver's history writes.

The file is written here by the solver's own :class:`~aquaflux.solve.StepHistory`, so these pin the
contract between the two packages: whatever the writer puts down, the reader gets back exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista")

from aquaflux.solve import StepHistory, StepReport
from aquaflux_ui import ConvergenceHistory


def _write(path, residuals):
    with StepHistory(path, clock=lambda: 0.0) as history:
        for step, residual in enumerate(residuals):
            history.on_checkpoint(StepReport(step, 4, residual, residual / 7.0, 0.5))
    return path


def test_it_reads_back_every_value_the_solver_wrote(tmp_path):
    residuals = [1.0 / 3.0, 1e-3 / 7.0, 2.5e-9]
    history = ConvergenceHistory.read(_write(tmp_path / "history.csv", residuals))

    assert history.n_steps == 3
    np.testing.assert_array_equal(history.steps, [1, 2, 3])
    # Exact: the writer keeps every digit, so a plot of the read values is a plot of the march's own.
    np.testing.assert_array_equal(history.columns["residual_norm"], residuals)
    np.testing.assert_array_equal(history.columns["residual_ratio"], np.asarray(residuals) / 7.0)
    assert set(history.columns) == set(StepHistory.COLUMNS)


def test_the_residual_columns_are_offered_relative_first(tmp_path):
    history = ConvergenceHistory.read(_write(tmp_path / "history.csv", [1.0, 0.1]))
    assert list(history.residual_columns()) == ["residual_ratio", "residual_norm"]


def test_a_history_with_no_steps_yet_has_none(tmp_path):
    # What a run writes before its first step, and all a segregated solve, which takes none, writes.
    history = ConvergenceHistory.read(_write(tmp_path / "history.csv", []))
    assert history.n_steps == 0
    assert history.steps.shape == (0,)


def test_a_history_with_only_its_header_has_no_steps(tmp_path):
    (tmp_path / "history.csv").write_text(",".join(StepHistory.COLUMNS) + "\n")
    assert ConvergenceHistory.read(tmp_path / "history.csv").n_steps == 0


def test_a_partly_written_last_line_is_left_for_the_next_read(tmp_path):
    path = _write(tmp_path / "history.csv", [1.0, 0.5])
    with path.open("a") as stream:
        stream.write("3,0.0,2")  # a run still writing its third row
    assert ConvergenceHistory.read(path).n_steps == 2


def test_a_row_of_the_wrong_width_in_the_middle_is_refused(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text("step,residual_norm\n1,0.5\n2\n3,0.1\n")
    with pytest.raises(ValueError, match="line 3: 1 cells where the header names 2"):
        ConvergenceHistory.read(path)


def test_a_file_with_no_step_column_is_refused(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text("iteration,residual\n1,0.5\n")
    with pytest.raises(ValueError, match="needs a 'step' column"):
        ConvergenceHistory.read(path)


def test_a_column_that_is_not_numeric_is_kept_as_text(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text("step,phase\n1,ramp\n2,target\n")
    history = ConvergenceHistory.read(path)
    assert list(history.columns["phase"]) == ["ramp", "target"]
    assert history.residual_columns() == {}
