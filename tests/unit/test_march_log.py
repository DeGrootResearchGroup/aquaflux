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
    RefreshTiming,
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
    logger.on_refresh(RefreshTiming("full", 27.0))
    logger.on_checkpoint(_report(), [1.0, 2.0])

    text = buffer.getvalue()
    assert "+- step" not in text  # no inner block
    assert "pc " not in text  # no preconditioner reporting
    assert "rel. change" not in text  # no per-equation block


def test_each_detail_switches_on_only_its_own_output() -> None:
    logger, buffer = _log(fields=lambda s: {"u": np.asarray(s)}, detail=("pc", "fields"))
    logger.on_inner(0, 4.0e-2, 1.0e-2, 5, 1.0)  # not requested -- must stay silent
    logger.on_refresh(RefreshTiming("shift", 0.4))
    logger.on_checkpoint(_report(), [1.0, 2.0])

    assert "pc shift 0.4s" in _asides(buffer)[0]
    assert "rel. change" in buffer.getvalue()
    assert "+- step" not in buffer.getvalue()


def test_the_preconditioner_column_is_reported_once_for_the_step_it_preceded() -> None:
    """The refresh happens before the step; it belongs to that step, not to every later one."""
    logger, buffer = _log(detail=("pc",))
    logger.on_refresh(RefreshTiming("full", 27.0))
    logger.on_step(_report())
    logger.on_step(_report())

    first, second = [line for line in _asides(buffer) if line.startswith("pc")]
    assert first == "pc full 27.0s"
    assert second == "pc -"


def test_the_refresh_line_breaks_the_cost_down_by_phase() -> None:
    """Two refreshes of equal length can have opposite causes; the total alone cannot say which.

    A re-probe of the Jacobian and a rebuild of the multigrid call for entirely different fixes, so the
    parts are reported in the order they ran rather than left to be inferred by differencing two runs.
    """
    logger, buffer = _log(detail=("pc",))
    logger.on_refresh(
        RefreshTiming("full", 23.0, (("probe", 14.6), ("assemble", 3.2), ("refactor", 5.2)))
    )
    logger.on_step(_report())

    line = next(line for line in _asides(buffer) if line.startswith("pc"))
    assert line == "pc full 23.0s (probe 14.6 assemble 3.2 refactor 5.2)"


def test_wall_time_the_phases_do_not_account_for_is_shown() -> None:
    """A breakdown that silently fails to add up reads as complete, which is worse than none at all."""
    logger, buffer = _log(detail=("pc",))
    logger.on_refresh(RefreshTiming("full", 23.0, (("probe", 14.6),)))
    logger.on_step(_report())

    assert "other 8.4" in next(line for line in _asides(buffer) if line.startswith("pc"))


def test_a_refresh_reporting_no_phases_still_reports_its_total() -> None:
    """The factorization preconditioners do not instrument themselves; the branch and total still log."""
    logger, buffer = _log(detail=("pc",))
    logger.on_refresh(RefreshTiming("shift", 8.4))
    logger.on_step(_report())

    assert next(line for line in _asides(buffer) if line.startswith("pc")) == "pc shift 8.4s"


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
    assert out["u"] == pytest.approx(0.5 / 5.0)  # |(0, 0.5)| / |(3, 4)|
    assert out["k"] == pytest.approx(0.0)  # k did not move -- reported separately from u


def test_the_first_call_has_nothing_to_compare_against() -> None:
    """`nan`, not 0: a zero would read as "converged" on the very first step of every march."""
    metric = field_change_metrics(lambda s: {"u": np.asarray(s)})

    assert math.isnan(metric([1.0, 2.0])["u"])


def test_a_field_that_was_identically_zero_reports_nan_rather_than_dividing() -> None:
    """No scale to be relative to -- reporting a number here would be inventing one."""
    metric = field_change_metrics(lambda s: {"k": np.asarray(s)})
    metric([0.0, 0.0])

    assert math.isnan(metric([1.0, 1.0])["k"])


def test_the_grid_stays_narrow_whatever_is_switched_on() -> None:
    """Width is the whole point: a table wide enough for every diagnostic stops reading as one.

    The step grid must stay comparable to the nested inner table, so the two read as one document and
    a row is never so far from its heading that the numbers are unpairable.
    """
    logger, buffer = _log(
        metrics=lambda s: {"xr/h": 7.243, "xr/h_full": 15.17},
        fields=lambda s: {
            name: np.asarray([s]) for name in ("u", "v", "w", "p", "k", "omega", "nut")
        },
        residuals=lambda s: dict.fromkeys(("u", "v", "w", "p", "k", "omega"), 1.0e-3 * s),
        detail=("inner", "fields", "residuals", "pc"),
    )
    logger.on_refresh(RefreshTiming("full", 21.1))
    logger.on_inner(0, 4.0e-2, 1.0e-2, 5, 1.0)
    logger.on_checkpoint(_report(), 1.0)
    logger.on_checkpoint(_report(), 2.0)

    grid = [line for line in buffer.getvalue().splitlines() if line.lstrip().startswith(("|", "+"))]
    assert max(len(line) for line in grid) <= 70


def test_a_step_table_following_an_inner_block_is_framed_and_separated() -> None:
    """An unruled heading butted against the block above reads as debris hanging off it.

    The step grid needs a visible top edge and a blank line, or it does not read as a table of its
    own -- which is exactly how it looked when the heading was emitted bare.
    """
    logger, buffer = _log(detail=("inner",))
    logger.on_inner(0, 4.0e-2, 1.0e-2, 5, 1.0)
    logger.on_step(_report())

    lines = buffer.getvalue().splitlines()
    heading = next(i for i, line in enumerate(lines) if line.startswith("| step"))
    assert lines[heading - 1].startswith(
        "+= summary stats"
    )  # a titled top border above the heading
    assert lines[heading - 2] == ""  # and air between it and the inner block
    assert lines[heading + 1].startswith("+--")  # closed underneath as usual
    # The title must not share a prefix with the inner block's "step N": a grep could not tell the
    # two grids apart, even though a reader can.
    assert not lines[heading - 1].startswith("+- step")


def test_on_retry_explains_why_a_step_is_repeated() -> None:
    """Three triggers call for different responses, so "it happened again" is not enough.

    The abandoned attempt's block is closed first, so the explanation sits between the two blocks
    rather than inside the one it is ending.
    """
    logger, buffer = _log(detail=("inner",))
    logger.on_inner(0, 1.0e-2, 3.3e-3, 5, 1.0)
    logger.on_retry("cycles", 1, 0.0694)  # the march hands over the ESCALATED beta
    logger.on_inner(0, 1.0e-2, 3.3e-3, 3, 1.0)

    lines = buffer.getvalue().splitlines()
    explain = next(i for i, line in enumerate(lines) if "redo step" in line)
    assert lines[explain - 1].lstrip().startswith("+--")  # the abandoned block closed first
    assert "cycles" in lines[explain] and "0.0694" in lines[explain]  # reason and the beta as given
    assert "attempt 2" in next(line for line in lines[explain:] if "+- step" in line)


def test_the_retry_line_reports_the_beta_it_was_given_rather_than_recomputing_it() -> None:
    """The logger must not re-derive the escalated shift, at any factor.

    It used to print ``beta * 2`` -- the *default* ``retry.beta_factor``, applied whatever the march was
    actually configured with -- so a march escalating by any other factor logged a shift it never ran at.
    A factor of 2 cannot catch that, so this asserts on a value that is not twice anything plausible.
    """
    logger, buffer = _log()
    logger.on_retry("cycles", 1, 0.15)

    line = next(line for line in buffer.getvalue().splitlines() if "redo step" in line)
    assert "0.1500" in line
    assert "0.3000" not in line


def test_the_solver_retry_does_not_claim_an_escalation_it_did_not_perform() -> None:
    """``"solver"`` retries at a tighter Krylov tolerance and leaves the shift alone.

    The same line served both retries, so it announced ``beta -> x`` on a path where the march never
    touched beta -- and, before the beta was reported honestly, an x that was twice the real one.
    """
    logger, buffer = _log()
    logger.on_retry("solver", 1, 0.25)

    line = next(line for line in buffer.getvalue().splitlines() if "redo step" in line)
    assert "solver" in line and "0.2500" in line
    assert "unchanged" in line
    assert "->" not in line.split("solver", 1)[1]  # no escalation arrow after the reason


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


def test_the_free_form_lines_are_ruled_off_from_the_columned_row() -> None:
    """The asides share the grid's width but not its columns, so they need a separator.

    Without one the eye carries the column structure down into lines that do not have it.
    """
    logger, buffer = _log(metrics=lambda state: {"xr/h": 7.24}, detail=("pc",))
    logger.on_refresh(RefreshTiming("full", 21.2))
    logger.on_checkpoint(_report(), None)

    lines = buffer.getvalue().splitlines()
    row = next(i for i, line in enumerate(lines) if line.startswith("|    1 "))
    # The heavy rule, not the light one: it closes the columned grid rather than dividing it.
    assert lines[row + 1].startswith("+==")  # ruled off before the free-form lines
    assert lines[row + 2].startswith("| pc ")


def test_a_step_clipped_by_a_constraint_is_distinguished_from_one_that_overshot() -> None:
    """A small ``a_min`` has two opposite causes, so reporting it alone is not enough.

    Overshoot says shorten the step and escalate the shift; a binding constraint says the direction
    is fine and simply cannot be followed that far. The flag and the reported limit separate them.
    """
    logger, buffer = _log()
    logger.on_step(_report(alpha=0.013, binding_limit=0.013))  # stopped by the limit
    logger.on_step(_report(alpha=0.5))  # stopped by the descent test

    limited, overshot = _step_rows(buffer)
    assert limited["flg"] == "L"
    assert overshot["flg"] == ""
    assert "limit 1.30e-02" in _asides(buffer)[0]
    assert "limit" not in _asides(buffer)[1]


def _field_rows(buffer: io.StringIO) -> dict[str, dict[str, str]]:
    """The per-equation block's rows, as ``{field: {heading: cell}}`` (the last block logged)."""
    headings, rows = None, {}
    for line in buffer.getvalue().splitlines():
        if not line.startswith("|"):
            continue
        cells = _cells(line)
        if cells[:1] == ["field"]:
            headings = cells
        elif headings is not None and len(cells) == len(headings):
            rows[cells[0]] = dict(zip(headings, cells, strict=True))
    return rows


def test_the_per_equation_block_says_which_equation_is_limiting() -> None:
    """The scalar residual says the solve stopped improving; only this says what stopped it."""
    logger, buffer = _log(
        fields=lambda s: {"u": np.asarray([s]), "k": np.asarray([s])},
        residuals=lambda s: {"u": 1.0e-6, "k": 4.0e-2 * s},
        detail=("fields", "residuals"),
    )
    logger.on_checkpoint(_report(), 1.0)
    logger.on_checkpoint(_report(), 0.5)

    rows = _field_rows(buffer)
    assert rows["k"]["resid"] == "2.000e-02"  # 4e-2 * 0.5
    assert rows["k"]["rate"] == "5.000e-01"  # halved this step -- converging
    assert rows["u"]["resid"] == "1.000e-06"
    assert rows["u"]["rate"] == "1.000e+00"  # unmoved: small, but not contracting


def test_the_first_step_has_no_previous_residual_to_contract_against() -> None:
    """A rate of 1 would read as "not converging"; there is simply nothing to divide by yet."""
    logger, buffer = _log(residuals=lambda s: {"u": 1.0e-3}, detail=("residuals",))
    logger.on_checkpoint(_report(), 1.0)

    assert _field_rows(buffer)["u"]["rate"] == "--"


def test_a_derived_field_with_no_equation_still_gets_a_row() -> None:
    """``nu_t`` is what the momentum equations actually see, so its drift explains a stalling solve --
    but it is not solved, so filling in a residual for it would be inventing a number."""
    logger, buffer = _log(
        fields=lambda s: {"k": np.asarray([s]), "nut": np.asarray([s])},
        residuals=lambda s: {"k": 1.0e-3},
        detail=("fields", "residuals"),
    )
    logger.on_checkpoint(_report(), 1.0)
    logger.on_checkpoint(_report(), 2.0)

    row = _field_rows(buffer)["nut"]
    assert row["rel. change"] == "1.000e+00"  # it moved, and that is reportable
    assert (row["resid"], row["rate"]) == ("--", "--")  # it has no equation, and that is not


def test_an_equation_with_no_matching_field_still_gets_a_row() -> None:
    """The two mappings are joined, not required to agree: neither side may silently drop a row."""
    logger, buffer = _log(
        fields=lambda s: {"k": np.asarray([s])},
        residuals=lambda s: {"k": 1.0e-3, "omega": 5.0e-4},
        detail=("fields", "residuals"),
    )
    logger.on_checkpoint(_report(), 1.0)

    assert _field_rows(buffer)["omega"]["resid"] == "5.000e-04"


def test_a_diverged_residual_is_shown_rather_than_hidden_as_absent() -> None:
    """A non-finite residual is the single most important thing a row can say."""
    logger, buffer = _log(residuals=lambda s: {"u": float("nan")}, detail=("residuals",))
    logger.on_checkpoint(_report(), 1.0)

    assert _field_rows(buffer)["u"]["resid"] == "nan"


def test_every_line_of_a_step_block_is_the_same_width() -> None:
    """The step row, the per-equation grid and the asides stack inside one frame: a width mismatch
    renders as a broken box, which reads as a corrupted log rather than as a layout slip."""
    logger, buffer = _log(
        metrics=lambda s: {"xr/h": 7.243},
        fields=lambda s: {"u": np.asarray([s]), "nut": np.asarray([s])},
        residuals=lambda s: {"u": 1.0e-3},
        detail=("fields", "residuals", "pc"),
    )
    logger.on_refresh(RefreshTiming("full", 21.8))
    logger.on_checkpoint(_report(binding_limit=0.243), 1.0)

    grid = [line for line in buffer.getvalue().splitlines() if line.startswith(("|", "+"))]
    assert len({len(line) for line in grid}) == 1


def test_the_per_equation_block_is_opt_in_like_every_other_diagnostic() -> None:
    """It costs an extra residual evaluation per step, so a routine run must not pay for it."""
    logger, buffer = _log(residuals=lambda s: {"u": 1.0e-3})  # `detail` omits "residuals"
    logger.on_checkpoint(_report(), 1.0)

    assert "resid" not in buffer.getvalue()


def test_the_aside_puts_one_concern_on_each_line() -> None:
    """Run together on one line, the preconditioner, the case metrics and the solver's own counters
    all had to be read to find any one of them."""
    logger, buffer = _log(metrics=lambda s: {"xr/h": 6.728}, detail=("pc",))
    logger.on_refresh(RefreshTiming("full", 21.8))
    logger.on_checkpoint(_report(binding_limit=0.243), 1.0)

    assert _asides(buffer)[:3] == ["pc full 21.8s", "xr/h 6.728", "limit 2.43e-01"]
