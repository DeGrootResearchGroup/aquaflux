"""Unit tests for :class:`~aquaflux.solve.MarchLogger`.

Formatting only -- driven by synthetic :class:`~aquaflux.solve.StepReport`s, so no solve is needed and
the tests stay fast. The point of the class is that a study does not re-derive the formatter, so what
is pinned here is the contract a study relies on: the columns it can scan down, the constants it does
*not* have to re-read on every row, the opt-in diagnostics, and the visibility of a redone step.

Assertions read **cells**, not substrings, so a column reordering is not a false failure while a wrong
value still is.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
from aquaflux.solve import (
    MarchLogger,
    StepReport,
    combine_metrics,
    field_change_metrics,
)


def _report(**kwargs) -> StepReport:
    fields = dict(step=0, cycles=10, residual_norm=2.0e-2, residual_ratio=4.0e-2, alpha=1.0)
    return StepReport(**(fields | kwargs))


def _log(**kwargs) -> tuple[MarchLogger, io.StringIO]:
    buffer = io.StringIO()
    return MarchLogger(buffer, clock=lambda: 0.0, **kwargs), buffer


def _cells(line: str) -> list[str]:
    """The cells of one grid line, stripped.

    Column headings deliberately avoid ``|`` (``R``, not ``|R|``) precisely so this split works: a
    heading carrying the delimiter makes the log unparseable by any column-splitting tool.
    """
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _asides(buffer: io.StringIO) -> list[str]:
    """The full-width lines beneath each step row -- where everything that is not a column lives."""
    return [
        line.strip("| ").strip()
        for line in buffer.getvalue().splitlines()
        if line.startswith("| ") and "|" not in line.strip("| ")
    ]


def _step_rows(buffer: io.StringIO) -> list[dict[str, str]]:
    """Every step row in the log, as ``{heading: cell}`` -- the table's headings carry the meaning."""
    headings, rows = None, []
    for line in buffer.getvalue().splitlines():
        if not line.startswith("|"):
            continue
        cells = _cells(line)
        if cells[:1] == ["step"]:
            headings = cells
        elif headings is not None and len(cells) == len(headings):
            rows.append(dict(zip(headings, cells, strict=True)))
    return rows


def test_the_stopping_test_is_stated_once_rather_than_on_every_row() -> None:
    """``|R0|`` and the target are constants within a rung, so repeating them per row is pure width.

    The residual alone cannot say how close the solve is to stopping -- the test is
    ``|R| <= atol + rtol*|R0|`` -- so the reference has to be somewhere; it just does not belong in
    every line.
    """
    logger, buffer = _log(rtol=1e-3)
    logger.on_step(_report(residual_norm=2.0e-2, residual_ratio=4.0e-2))
    logger.on_step(_report(residual_norm=1.0e-2, residual_ratio=2.0e-2))

    banners = [line for line in buffer.getvalue().splitlines() if line.startswith("reference")]
    assert len(banners) == 1  # both steps share one |R0|, so it is announced once
    assert "|R0| = 5.0000e-01" in banners[0]  # 2.0e-2 / 4.0e-2
    assert "stopping at |R| <= 5.000e-04" in banners[0]  # rtol * |R0|


def test_a_new_reference_is_announced_again() -> None:
    """A continuation rung re-bases ``|R0|``; a stale banner misstates the bar for every row under it."""
    logger, buffer = _log(rtol=1e-3)
    logger.on_step(_report(residual_norm=2.0e-2, residual_ratio=4.0e-2))
    logger.on_step(_report(residual_norm=1.0e-2, residual_ratio=1.0e-1))

    banners = [line for line in buffer.getvalue().splitlines() if line.startswith("reference")]
    assert len(banners) == 2
    assert "1.0000e-01" in banners[1]


def test_a_diverged_step_reports_no_reference_rather_than_nan() -> None:
    """A NaN ratio must not print ``|R0| = nan``.

    Every comparison against NaN is False, so a ``ratio <= 0`` guard would let a diverged step through
    into the division -- and the diverged step is exactly the line a reader is studying.
    """
    logger, buffer = _log(rtol=1e-3)
    logger.on_step(_report(residual_norm=float("nan"), residual_ratio=float("nan")))

    assert "reference" not in buffer.getvalue()  # not faked
    assert _step_rows(buffer)[0]["R"] == "nan"  # but the measured value is still reported


def test_injected_metrics_are_reported_beneath_the_row() -> None:
    """Case-specific quantities the solver cannot know ride on an injected callable.

    They sit in the aside rather than in columns: their names and count vary by case, and widening
    the grid per case is what stops it reading as a table.
    """
    logger, buffer = _log(metrics=lambda state: {"xr/h": state * 2.0})
    logger.on_checkpoint(_report(), 3.0)

    assert "xr/h 6" in _asides(buffer)[0]


def test_on_step_omits_metrics_because_it_has_no_state() -> None:
    """``on_step`` carries no state, so it cannot evaluate the metrics -- and must not try."""
    logger, buffer = _log(metrics=lambda state: {"xr/h": state * 2.0})
    logger.on_step(_report())  # would raise if the metrics were called with no state

    assert "xr/h" not in buffer.getvalue()


def test_a_redone_step_is_flagged() -> None:
    """``cycles`` counts only the accepted attempt, so a redone step needs its own marker.

    Without this a step that escalated twice reads exactly like a cheap one, and a retry mechanism
    left unconfigured never announces its absence.
    """
    logger, buffer = _log()
    logger.on_step(_report(escalations=2))
    logger.on_step(_report(diverged_retry=True))
    logger.on_step(_report())

    escalated, retried, plain = _step_rows(buffer)
    assert escalated["flg"] == "e2"
    assert retried["flg"] == "R"
    assert plain["flg"] == ""


def test_cumulative_cycles_and_steps_run_across_phases() -> None:
    """A phase (a continuation rung) relabels the log; it does not restart the running totals."""
    logger, buffer = _log()
    logger.on_step(_report(cycles=10))
    logger.phase("rung", total=3)
    logger.on_step(_report(cycles=5))

    assert any("rung 1/3" in line for line in buffer.getvalue().splitlines())
    assert _step_rows(buffer)[1]["step"] == "2"
    assert "cum 11" in _asides(buffer)[1]  # (10-2) + (5-2), offsets stripped


def test_solve_cost_is_split_into_inner_count_and_corrected_cycles() -> None:
    """One summed count conflates *how many* solves a step needed with how hard each one was.

    It also overstates the work: the raw count carries lineax's +2 per solve, so this step's ``21``
    is really 15 cycles over 3 inner iterations, not 21.
    """
    logger, buffer = _log()
    logger.on_step(_report(cycles=21, inner_iterations=3))

    row = _step_rows(buffer)[0]
    assert row["in"] == "3"
    assert row["cyc"] == "15"  # 21 - 2*3


def test_a_step_whose_count_is_entirely_offset_reads_as_its_real_cost() -> None:
    """Two ideal single-cycle solves report a raw ``6``; reading that as six cycles triples the cost.

    This is the regime the correction matters in -- the cheap steps near the root, where the raw
    number is dominated by the offset rather than by any real work.
    """
    logger, buffer = _log()
    logger.on_step(_report(cycles=6, inner_iterations=2))

    assert _step_rows(buffer)[0]["cyc"] == "2"


def test_the_step_row_reports_the_minimum_alpha_not_an_inner_one() -> None:
    """``a_min`` is named apart from the inner table's ``alpha`` because they differ, confusingly.

    An inner iteration reports ``alpha = 1`` even when its line search failed to descend, while the
    step folds any such iteration in as ``0`` -- so ``a_min=0.000`` beside ``alpha=1.000`` is a real
    state (a full step that did not reduce ``|G|``), not a contradiction.
    """
    logger, buffer = _log(detail=("inner",))
    logger.on_inner(0, 4.0e-2, 5.0e-2, 5, 1.0)  # grew: the non-descent fallback, still alpha 1
    logger.on_step(_report(alpha=0.0))

    inner = next(
        line for line in buffer.getvalue().splitlines() if line.lstrip().startswith("|     0")
    )
    assert inner.rstrip().endswith("| 1.000 |")  # the inner iteration's own factor
    assert _step_rows(buffer)[0]["a_min"] == "0.000"  # what the step reports
    assert float(_cells(inner)[4]) >= 1.0  # rate >= 1 identifies the non-descent exactly


def test_on_inner_tabulates_each_iteration_with_its_own_cost_and_rate() -> None:
    """The per-inner hook resolves the step summary into per-solve cost and the ``|G|`` trajectory."""
    logger, buffer = _log(detail=("inner",))
    logger.on_inner(0, 4.0e-2, 1.0e-2, 5, 1.0)
    logger.on_inner(1, 1.0e-2, 5.0e-3, 9, 0.5)

    rows = [
        line.strip() for line in buffer.getvalue().splitlines() if line.lstrip().startswith("| ")
    ]
    headings, first, second = rows
    for heading in ("inner", "cyc", "G in", "G out", "rate", "alpha"):
        assert heading in headings
    # cyc is 5 - 2, this single solve's offset; rate is the inner contraction 1.0e-2 / 4.0e-2.
    assert first == "|     0 |    3 |  4.000e-02 |  1.000e-02 |  0.250 | 1.000 |"
    assert second == "|     1 |    7 |  1.000e-02 |  5.000e-03 |  0.500 | 0.500 |"


def test_the_inner_block_opens_on_each_attempt_and_is_nested_under_its_step() -> None:
    """One step is one block: a retry re-opens at ``inner=0``, and the block is indented as detail.

    Without the re-open, a redone step's rows run into the abandoned attempt's and the block stops
    being a record of one step's work -- which is exactly what makes a retry's cost legible.
    """
    logger, buffer = _log(detail=("inner",))
    logger.on_inner(0, 4.0e-2, 3.0e-2, 5, 1.0)
    logger.on_inner(0, 4.0e-2, 1.0e-2, 5, 1.0)  # the step redone at an escalated beta
    logger.on_step(_report(escalations=1))

    titles = [line for line in buffer.getvalue().splitlines() if "+- step" in line]
    assert len(titles) == 2  # the retry gets its own block
    assert all(line.startswith("    ") for line in titles)  # indented as detail of the step row
    assert _step_rows(buffer)[0]["flg"] == "e1"


def test_a_block_title_does_not_repeat_the_previous_steps_residual() -> None:
    """Under a self-rescaling measure the previous step's ``R`` is NOT this step's entering ``G in``.

    The march re-derives the row scales at the state each outer iteration begins from, so the two
    measure the same state in different scales -- over one march they differed on every step. Printing
    both invites a comparison that never holds; ``inner 0``'s ``G in`` is the entering residual.
    """
    logger, buffer = _log(detail=("inner",))
    logger.on_step(_report(residual_norm=2.0e-2))
    logger.on_inner(0, 1.9e-2, 1.0e-2, 5, 1.0)

    title = next(line for line in buffer.getvalue().splitlines() if "+- step" in line)
    assert "step 2" in title
    assert "2.000e-02" not in title  # the previous step's R, in the previous step's scales


def test_headings_are_re_emitted_so_a_long_run_stays_readable() -> None:
    """Scrolling a thousand-line tail back to a single heading row is not reading."""
    logger, buffer = _log()
    for _ in range(MarchLogger.HEADINGS_EVERY + 1):
        logger.on_step(_report())

    headings = [line for line in buffer.getvalue().splitlines() if line.startswith("| step")]
    assert len(headings) == 2


def test_diagnostics_are_off_by_default_even_when_the_hooks_are_wired() -> None:
    """A routine run should pay no log volume for instruments it did not ask for.

    The hooks stay safe to call regardless, so a driver wires them once and flips ``detail`` -- rather
    than rewiring, which is how a driver ends up hand-rolling its own conditional plumbing.
    """
    logger, buffer = _log(fields=lambda s: {"u": np.asarray(s)})
    logger.on_inner(0, 4.0e-2, 1.0e-2, 5, 1.0)
    logger.on_refresh("full", 27.0)
    logger.on_checkpoint(_report(), [1.0, 2.0])

    text = buffer.getvalue()
    assert "+- step" not in text  # no inner block
    assert "pc " not in text  # no preconditioner reporting
    assert "rel u" not in text  # no field-change row


def test_each_detail_switches_on_only_its_own_output() -> None:
    logger, buffer = _log(fields=lambda s: {"u": np.asarray(s)}, detail=("pc", "fields"))
    logger.on_inner(0, 4.0e-2, 1.0e-2, 5, 1.0)  # not requested -- must stay silent
    logger.on_refresh("shift", 0.4)
    logger.on_checkpoint(_report(), [1.0, 2.0])

    assert "pc shift 0.4s" in _asides(buffer)[0]
    assert "rel u" in buffer.getvalue()
    assert "+- step" not in buffer.getvalue()


def test_the_preconditioner_column_is_reported_once_for_the_step_it_preceded() -> None:
    """The refresh happens before the step; it belongs to that step, not to every later one."""
    logger, buffer = _log(detail=("pc",))
    logger.on_refresh("full", 27.0)
    logger.on_step(_report())
    logger.on_step(_report())

    first, second = _asides(buffer)
    assert "pc full 27.0s" in first
    assert "pc -" in second


def test_an_unknown_detail_name_raises_rather_than_being_ignored() -> None:
    """A silently-dropped typo means losing a diagnostic you believed you had switched on."""
    with pytest.raises(ValueError, match="unknown march-log detail"):
        MarchLogger(detail=("inner", "preconditioner"))


def test_combine_metrics_merges_several_metric_callables() -> None:
    """A run wants a case quantity AND a solver diagnostic, but the logger takes one callable."""
    combined = combine_metrics(lambda s: {"xr/h": 7.0}, lambda s: {"du/u": 1e-3})

    assert dict(combined(None)) == {"xr/h": 7.0, "du/u": 1e-3}


def test_field_change_reports_each_fields_relative_movement() -> None:
    """The residual says the equations are unsatisfied; this says whether the SOLUTION still moves."""
    metric = field_change_metrics(lambda s: {"u": np.asarray(s["u"]), "k": np.asarray(s["k"])})
    metric({"u": [3.0, 4.0], "k": [1.0, 0.0]})  # first call primes the previous iterate

    out = metric({"u": [3.0, 4.5], "k": [1.0, 0.0]})
    assert out["du/u"] == pytest.approx(0.5 / 5.0)  # |(0, 0.5)| / |(3, 4)|
    assert out["dk/k"] == pytest.approx(0.0)  # k did not move -- reported separately from u


def test_the_first_call_has_nothing_to_compare_against() -> None:
    """`nan`, not 0: a zero would read as "converged" on the very first step of every march."""
    metric = field_change_metrics(lambda s: {"u": np.asarray(s)})

    assert math.isnan(metric([1.0, 2.0])["du/u"])


def test_a_field_that_was_identically_zero_reports_nan_rather_than_dividing() -> None:
    """No scale to be relative to -- reporting a number here would be inventing one."""
    metric = field_change_metrics(lambda s: {"k": np.asarray(s)})
    metric([0.0, 0.0])

    assert math.isnan(metric([1.0, 1.0])["dk/k"])


def test_the_grid_stays_narrow_whatever_is_switched_on() -> None:
    """Width is the whole point: a table wide enough for every diagnostic stops reading as one.

    The step grid must stay comparable to the nested inner table, so the two read as one document and
    a row is never so far from its heading that the numbers are unpairable.
    """
    logger, buffer = _log(
        metrics=lambda s: {"xr/h": 7.243, "xr/h_full": 15.17},
        fields=lambda s: {name: np.asarray([s]) for name in ("u", "p", "k", "w", "nut")},
        detail=("inner", "fields", "pc"),
    )
    logger.on_refresh("full", 21.1)
    logger.on_inner(0, 4.0e-2, 1.0e-2, 5, 1.0)
    logger.on_checkpoint(_report(), 1.0)
    logger.on_checkpoint(_report(), 2.0)

    grid = [
        line for line in buffer.getvalue().splitlines() if line.lstrip().startswith(("|", "+-"))
    ]
    assert max(len(line) for line in grid) <= 70


def test_a_row_following_an_inner_block_is_still_labelled() -> None:
    """An inner block separates a step row from the last heading, leaving bare numbers.

    The compact re-heading is one line rather than three, because labelling one row must not cost
    more than the row.
    """
    logger, buffer = _log(detail=("inner",))
    logger.on_inner(0, 4.0e-2, 1.0e-2, 5, 1.0)
    logger.on_step(_report())

    lines = buffer.getvalue().splitlines()
    heading = next(i for i, line in enumerate(lines) if line.startswith("| step"))
    row = next(i for i, line in enumerate(lines) if line.startswith("|    1 "))
    assert row - heading == 2  # heading, rule, row -- nothing between them


def test_on_retry_explains_why_a_step_is_repeated() -> None:
    """Three triggers call for different responses, so "it happened again" is not enough.

    The abandoned attempt's block is closed first, so the explanation sits between the two blocks
    rather than inside the one it is ending.
    """
    logger, buffer = _log(detail=("inner",))
    logger.on_inner(0, 1.0e-2, 3.3e-3, 5, 1.0)
    logger.on_retry("cycles", 1, 0.0347)
    logger.on_inner(0, 1.0e-2, 3.3e-3, 3, 1.0)

    lines = buffer.getvalue().splitlines()
    explain = next(i for i, line in enumerate(lines) if "redo step" in line)
    assert lines[explain - 1].lstrip().startswith("+--")  # the abandoned block closed first
    assert (
        "cycles" in lines[explain] and "0.0694" in lines[explain]
    )  # reason and the escalated beta
    assert "attempt 2" in next(line for line in lines[explain:] if "+- step" in line)


def test_the_attempt_counter_resets_between_steps() -> None:
    """Otherwise every later block reads as a retry of a step that was taken cleanly."""
    logger, buffer = _log(detail=("inner",))
    logger.on_retry("cycles", 1, 0.03)
    logger.on_inner(0, 1.0e-2, 3.3e-3, 3, 1.0)
    logger.on_step(_report(escalations=1))
    logger.on_inner(0, 1.0e-2, 3.3e-3, 3, 1.0)

    titles = [line for line in buffer.getvalue().splitlines() if "+- step" in line]
    assert "attempt" in titles[0]
    assert "attempt" not in titles[-1]


def test_a_legend_explains_that_G_and_R_are_different_residuals() -> None:
    """Nothing in the table can say that the inner ``G`` and the step's ``R`` are different quantities.

    ``G`` is the implicit timestep's residual and the inner loop drives it to zero; ``R`` is the steady
    residual and is not driven to zero -- at ``G = 0`` it equals minus the shift term. So the step's
    ``R`` is never the last ``G out``, which is exactly what a reader expects it to be.
    """
    logger, buffer = _log(detail=("inner",))
    logger.on_inner(0, 1.422e-5, 2.613e-12, 3, 1.0)
    logger.on_step(_report(residual_norm=4.064e-6))
    logger.on_inner(0, 4.06e-6, 1.0e-9, 3, 1.0)

    text = buffer.getvalue()
    assert text.count("R + beta*d*(phi - phi_n)") == 1  # written once, not per block
    assert text.index("R + beta*d") < text.index("+- step")  # before the first block it explains
