"""Tests for what the Run section draws of a march: its limits, events, headline numbers and plots.

The histories are written by the solver's own :class:`~aquaflux.solve.StepHistory`, so what is pinned
is the contract between the two packages: whatever the march records, the page shows from it. The
limits are read from the committed pitzDaily case through the real schema, so a renamed setting
breaks a test rather than silently dropping a line from a plot.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("plotly")

from aquaflux.case import case_schema, read_case_document
from aquaflux.solve import RefreshTiming, StepHistory, StepReport
from aquaflux.solve.retry import ESCALATING_REASONS
from aquaflux_ui.case_form import CaseSchema, setting
from aquaflux_ui.history import ConvergenceHistory
from aquaflux_ui.run_monitor import (
    RETRY_REASONS,
    RunLimits,
    cost_figures,
    equation_columns,
    residual_figure,
    run_events,
    run_tiles,
)

REPO = Path(__file__).resolve().parents[2]
PITZDAILY = REPO / "validation" / "pitzdaily_openfoam" / "case.yaml"


@pytest.fixture(scope="module")
def schema():
    return CaseSchema(case_schema())


def _history(tmp_path, steps, *, refits=(), retries=(), equations=None, ramp=0):
    """A march of ``len(steps)`` steps with residuals ``steps``, written as the solver writes one.

    ``refits`` and ``retries`` name the 0-based steps that had them; the first ``ramp`` steps drove a
    continuation's stations rather than the case's own problem.
    """
    path = tmp_path / "history.csv"
    with StepHistory(path, clock=iter(np.arange(0.0, 100.0, 2.5)).__next__) as history:
        for index, residual in enumerate(steps):
            if index in refits:
                history.on_refresh(RefreshTiming("inner", 1.5))
            if index in retries:
                history.on_retry("alpha", 1, 0.2)
            if equations is not None:
                history.on_residuals({name: residual * share for name, share in equations.items()})
            history.on_checkpoint(
                StepReport(
                    index, 3 + index, residual, residual / steps[0], 1.0 if index not in retries else 0.005,
                    inner_iterations=2, max_inner_cycles=4, shift=0.5 / (index + 1),
                    escalations=int(index in retries), station=min(index, ramp), arrived=index >= ramp,
                )
            )  # fmt: skip
    return ConvergenceHistory.read(path)


# --- the limits, from the case file --------------------------------------------------------------


def test_the_limits_are_the_ones_the_case_file_sets(schema):
    limits = RunLimits.from_case(schema, read_case_document(PITZDAILY))
    assert limits == RunLimits(
        rtol=0.0,
        atol=1.0e-5,
        max_steps=150,
        inner_steps=5,
        abort_above_cycles=10,
        beta_min=0.005,
        retry_alpha=0.01,
    )


def test_a_setting_inside_a_section_whose_unset_means_off_does_not_apply(schema):
    document = read_case_document(PITZDAILY)
    del document["solver"]["dual_time"], document["solver"]["retry"]
    # The inner loop's iterations have a default of their own -- which does not apply when there is
    # no inner loop to iterate.
    loop = schema.field("DualTimeLoop", "inner_steps")
    assert (loop.get("default") or loop.get("resolved_default")) is not None
    assert setting(schema, document, ("solver", "dual_time", "inner_steps")) is None
    limits = RunLimits.from_case(schema, document)
    assert (limits.inner_steps, limits.retry_alpha) == (None, None)


def test_an_unset_setting_takes_its_stated_default(schema):
    document = read_case_document(PITZDAILY)
    del document["solver"]["max_steps"]
    field = schema.field("CoupledMarch", "max_steps")
    # Not stated on the setting itself (it is unset, the solve's own), but resolved from the solve.
    assert field["default"] is None and field["resolved_default"] is not None
    assert setting(schema, document, ("solver", "max_steps")) == field["resolved_default"]


def test_the_target_is_the_stopping_test_at_the_first_residual():
    assert RunLimits(atol=1e-5, rtol=1e-3).target(2.0) == pytest.approx(1e-5 + 2e-3)
    assert RunLimits(atol=1e-5).target(None) == 1e-5
    assert RunLimits().target(2.0) is None


# --- what happened ---------------------------------------------------------------------------------


def test_events_are_newest_first_and_say_what_happened(tmp_path):
    history = _history(tmp_path, [1.0, 0.5, 0.2, 0.1, 0.05], refits=(1, 3), retries=(3,), ramp=2)
    events = run_events(history)
    assert [(event.step, event.kind) for event in events] == [
        (4, "retry"),
        (4, "refit"),
        (3, "arrived"),
        (2, "refit"),
    ]
    assert events[0].text == f"Step redone: {RETRY_REASONS['alpha']}"
    assert events[1].seconds == 1.5
    # What the page is sent is plain JSON: a NumPy number in it is dropped by the page's state.
    assert json.loads(json.dumps([event.to_state() for event in events]))[1] == {
        "step": 4, "kind": "refit", "text": "Preconditioner refitted", "seconds": 1.5,
    }  # fmt: skip


def test_the_end_of_a_continuation_is_reported_once():
    # Only the first step on the case's own problem is the event, not every one after it.
    history = ConvergenceHistory(
        {"step": np.arange(1.0, 6.0), "arrived": np.array([0.0, 0.0, 1.0, 1.0, 1.0])}
    )
    assert [(e.step, e.kind) for e in run_events(history)] == [(3, "arrived")]


def test_every_reason_the_march_redoes_a_step_for_is_said_in_words():
    # The march names four; the escalating ones are the solver's own constant.
    assert set(RETRY_REASONS) == {"cycles", "alpha", "diverged", "solver"}
    assert ESCALATING_REASONS <= set(RETRY_REASONS)


# --- the headline numbers ---------------------------------------------------------------------------


def test_the_tiles_measure_the_way_to_the_target_in_decades(tmp_path):
    history = _history(tmp_path, [1.0, 1e-2, 1e-3], refits=(1,), retries=(1,))
    residual, step, cycles, retries, refits = run_tiles(history, RunLimits(atol=1e-5))
    # Three of five decades travelled, two to go.
    assert residual["progress"] == pytest.approx(0.6)
    assert "2.0 decades to go" in residual["detail"]
    assert step["value"] == "3"
    assert cycles["value"] == str(int(history.columns["restart_cycles"].sum()))
    assert (retries["value"], retries["detail"]) == ("1", "last at step 2")
    assert (refits["value"], refits["detail"]) == ("1", "last at step 2")


def test_a_march_below_its_target_reads_as_reached(tmp_path):
    (residual, *_) = run_tiles(_history(tmp_path, [1.0, 1e-6]), RunLimits(atol=1e-5))
    assert residual["progress"] == 1.0 and residual["detail"].endswith("reached")


# --- the plots -------------------------------------------------------------------------------------


def test_the_residual_plot_draws_the_target_the_retries_refits_and_continuation(tmp_path):
    history = _history(tmp_path, [1.0, 0.5, 0.2, 0.1], refits=(2,), retries=(3,), ramp=2)
    figure = residual_figure(history, RunLimits(atol=1e-5))
    names = [trace.name for trace in figure.data]
    assert names == ["Residual", "Redone"]
    assert list(figure.data[1].x) == [4]
    lines = [shape for shape in figure.layout.shapes if shape.type == "line"]
    assert any(shape.y0 == 1e-5 and shape.y1 == 1e-5 for shape in lines)  # the target
    assert any(shape.x0 == 3 and shape.yref == "paper" for shape in lines)  # the refit tick
    (band,) = [shape for shape in figure.layout.shapes if shape.type == "rect"]
    assert (band.x0, band.x1) == (0.5, 2.5)  # the two continuation steps
    assert "continuation" in [annotation.text for annotation in figure.layout.annotations]
    # The small plots shade it unlabelled -- and an empty label would show plotly's placeholder.
    for small in cost_figures(history, RunLimits()).values():
        assert [shape.type for shape in small.layout.shapes] == ["rect"]
        assert not small.layout.annotations


def test_the_residual_columns_of_the_total_are_not_taken_for_equations(tmp_path):
    history = _history(tmp_path, [1.0, 0.5])
    assert {"residual_norm", "residual_ratio"} <= set(history.columns)
    assert equation_columns(history) == {}


def test_per_equation_the_residual_plot_draws_one_labelled_line_per_equation(tmp_path):
    shares = {"u": 0.6, "v": 0.3, "p": 0.1}
    history = _history(tmp_path, [1.0, 0.5, 0.2], equations=shares)
    assert equation_columns(history) == {
        "residual_of_u": "u",
        "residual_of_v": "v",
        "residual_of_p": "p",
    }
    figure = residual_figure(history, RunLimits(atol=1e-5), by_equation=True)
    assert [trace.name for trace in figure.data] == ["u", "v", "p"]
    np.testing.assert_allclose(figure.data[0].y, np.array([1.0, 0.5, 0.2]) * 0.6)
    # Each is named at its end too, since three colours alone do not identify three lines.
    assert [annotation.text for annotation in figure.layout.annotations] == ["u", "v", "p"]
    assert math.isclose(figure.layout.annotations[0].y, math.log10(0.2 * 0.6))
    # The target applies to the total, which is not drawn here.
    assert not any(shape.type == "line" and shape.yref != "paper" for shape in figure.layout.shapes)


def test_a_history_without_equation_columns_draws_the_total_even_when_asked_for_equations(tmp_path):
    figure = residual_figure(_history(tmp_path, [1.0, 0.5]), RunLimits(), by_equation=True)
    assert [trace.name for trace in figure.data] == ["Residual"]


def test_each_small_plot_draws_its_limit_where_the_case_sets_one(tmp_path):
    history = _history(tmp_path, [1.0, 0.5, 0.2], retries=(1,))
    limits = RunLimits(abort_above_cycles=10, inner_steps=5, beta_min=0.005, retry_alpha=0.01)
    figures = cost_figures(history, limits)
    for name, value in {"cycles": 10, "inner": 5, "shift": 0.005, "alpha": 0.01}.items():
        limits_drawn = [s.y0 for s in figures[name].layout.shapes if s.type == "line"]
        assert limits_drawn == [value], name
    # A redone step is marked in the cost plot, and its collapsed step length stays on the axis.
    assert figures["cycles"].data[0].marker.color[1] != figures["cycles"].data[0].marker.color[0]
    assert figures["alpha"].data[0].y[1] == 0.005
    unset = cost_figures(history, RunLimits())
    assert not any(s.type == "line" for f in unset.values() for s in f.layout.shapes)


def test_every_plot_waits_for_the_first_step(tmp_path):
    empty = ConvergenceHistory({"step": np.zeros(0)})
    figures = [residual_figure(empty, RunLimits()), *cost_figures(empty, RunLimits()).values()]
    assert all(f.layout.annotations[0].text == "Waiting for the first step" for f in figures)
    assert run_tiles(empty, RunLimits()) == [] and run_events(empty) == []
