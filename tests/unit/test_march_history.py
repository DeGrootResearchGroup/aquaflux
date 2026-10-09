"""Unit tests for :class:`~aquaflux.solve.StepHistory`, the per-step comma-separated-values record.

Driven by synthetic :class:`~aquaflux.solve.StepReport`s, read back with the standard library's own
CSV reader -- the file is for a program to read, so a program reading it is the test.
"""

from __future__ import annotations

import csv

import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.solve import RefreshTiming, StepHistory, StepReport


def _rows(path):
    return list(csv.DictReader(path.open(newline="")))


def test_the_file_is_empty_until_the_first_step_and_the_header_comes_with_it(tmp_path):
    # The equation columns are known only once the march reports them, so the header waits for them.
    with StepHistory(tmp_path / "history.csv") as history:
        assert (tmp_path / "history.csv").read_text() == ""
        history.on_checkpoint(StepReport(0, 3, 1.0, 1.0, 1.0))
        header = (tmp_path / "history.csv").read_text().splitlines()[0].split(",")
    assert tuple(header) == StepHistory.COLUMNS


def test_every_report_field_is_a_column_and_the_segment_index_is_renamed():
    columns = StepHistory.COLUMNS
    assert columns[:3] == ("step", "seconds", "segment_step")
    assert columns[-3:] == ("refits", "refit_seconds", "retry_reasons")
    assert set(columns) == {
        "step",
        "seconds",
        "restart_cycles",
        "refits",
        "refit_seconds",
        "retry_reasons",
    } | {"segment_step" if name == "step" else name for name in StepReport._fields}


def test_a_steps_refits_are_counted_and_timed_and_a_reused_preconditioner_is_not_one(tmp_path):
    with StepHistory(tmp_path / "h.csv") as history:
        history.on_refresh(RefreshTiming("full", 1.5))
        history.on_refresh(RefreshTiming("none", 0.25))
        history.on_refresh(RefreshTiming("inner", 2.0))
        history.on_checkpoint(StepReport(0, 3, 1.0, 1.0, 1.0))
        history.on_checkpoint(StepReport(1, 3, 1.0, 1.0, 1.0))
    first, second = _rows(tmp_path / "h.csv")
    assert (first["refits"], float(first["refit_seconds"])) == ("2", 3.5)
    # What was heard for one step is not carried onto the next.
    assert (second["refits"], float(second["refit_seconds"])) == ("0", 0.0)


def test_a_redone_step_records_each_reason_in_order(tmp_path):
    with StepHistory(tmp_path / "h.csv") as history:
        history.on_retry("alpha", 1, 0.2)
        history.on_retry("solver", 2, 0.2)
        history.on_checkpoint(StepReport(0, 3, 1.0, 1.0, 1.0, escalations=1))
        history.on_checkpoint(StepReport(1, 3, 1.0, 1.0, 1.0))
    first, second = _rows(tmp_path / "h.csv")
    assert first["retry_reasons"] == "alpha;solver"
    assert second["retry_reasons"] == ""


def test_each_equations_residual_is_a_column_after_the_fixed_ones(tmp_path):
    with StepHistory(tmp_path / "h.csv") as history:
        history.on_residuals({"u": 1.0 / 3.0, "p": 2.0e-7})
        history.on_checkpoint(StepReport(0, 3, 1.0, 1.0, 1.0))
        # A step whose measure named nothing leaves its cells empty rather than repeating the last.
        history.on_checkpoint(StepReport(1, 3, 1.0, 1.0, 1.0))
    header = (tmp_path / "h.csv").read_text().splitlines()[0].split(",")
    assert tuple(header[-2:]) == ("residual_of_u", "residual_of_p")
    first, second = _rows(tmp_path / "h.csv")
    assert float(first["residual_of_u"]) == 1.0 / 3.0 and float(first["residual_of_p"]) == 2.0e-7
    assert (second["residual_of_u"], second["residual_of_p"]) == ("", "")


def test_a_step_naming_other_equations_than_the_header_is_refused(tmp_path):
    with StepHistory(tmp_path / "h.csv") as history:
        history.on_residuals({"u": 1.0, "p": 1.0})
        history.on_checkpoint(StepReport(0, 3, 1.0, 1.0, 1.0))
        with pytest.raises(ValueError, match=r"records the equations \('u', 'p'\)"):
            history.on_residuals({"u": 1.0, "k": 1.0})


def test_a_row_carries_every_value_exactly(tmp_path):
    report = StepReport(
        step=4,
        cycles=9,
        residual_norm=1.0 / 3.0,
        residual_ratio=2.0e-7 / 3.0,
        alpha=0.25,
        drift=0.125,
        inner_iterations=2,
        max_inner_cycles=3,
        binding_limit=0.5,
        shift=1.5,
        escalations=1,
        diverged_retry=True,
    )
    with StepHistory(tmp_path / "h.csv", clock=iter([10.0, 12.5]).__next__) as history:
        history.on_checkpoint(report, state=np.zeros(3))

    (row,) = _rows(tmp_path / "h.csv")
    assert row["step"] == "1" and row["segment_step"] == "4"
    assert float(row["seconds"]) == 2.5
    # Exact, not close: the record is worth keeping only if it is the march's own doubles.
    assert float(row["residual_norm"]) == 1.0 / 3.0
    assert float(row["residual_ratio"]) == 2.0e-7 / 3.0
    assert row["diverged_retry"] == "1" and row["escalations"] == "1"
    assert int(row["restart_cycles"]) == report.restart_cycles
    assert {name: row[name] for name in ("alpha", "drift", "binding_limit", "shift")} == {
        "alpha": "0.25",
        "drift": "0.125",
        "binding_limit": "0.5",
        "shift": "1.5",
    }


def test_the_step_count_runs_on_across_segments(tmp_path):
    # A continuation restarts each report's index at every segment; the history's own count does not.
    with StepHistory(tmp_path / "h.csv", clock=lambda: 0.0) as history:
        for segment_step in (0, 1, 0, 1, 2):
            history.on_checkpoint(StepReport(segment_step, 3, 1.0, 1.0, 1.0), state=None)
    rows = _rows(tmp_path / "h.csv")
    assert [row["step"] for row in rows] == ["1", "2", "3", "4", "5"]
    assert [row["segment_step"] for row in rows] == ["0", "1", "0", "1", "2"]


def test_array_scalars_are_written_as_plain_numbers(tmp_path):
    # A march's report can carry NumPy or JAX scalars, whose own repr names their type.
    report = StepReport(
        step=np.int64(0),
        cycles=jnp.asarray(5),
        residual_norm=np.float64(0.1),
        residual_ratio=jnp.asarray(0.2),
        alpha=np.float32(0.5),
        diverged_retry=np.bool_(False),
    )
    with StepHistory(tmp_path / "h.csv") as history:
        history.on_checkpoint(report)
    (row,) = _rows(tmp_path / "h.csv")
    assert (row["cycles"], row["residual_norm"], row["alpha"], row["diverged_retry"]) == (
        "5",
        "0.1",
        "0.5",
        "0",
    )
    assert float(row["residual_ratio"]) == 0.2


def test_each_row_is_on_disk_as_soon_as_it_is_written(tmp_path):
    with StepHistory(tmp_path / "h.csv") as history:
        history.on_checkpoint(StepReport(0, 3, 1.0, 1.0, 1.0))
        # Read through a separate handle while the writer is still open.
        assert len(_rows(tmp_path / "h.csv")) == 1
