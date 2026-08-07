"""Unit tests for :class:`~aquaflux.solve.MarchLogger`.

Formatting only -- driven by synthetic :class:`~aquaflux.solve.StepReport`s, so no solve is needed and
the tests stay fast. The point of the class is that a study does not re-derive the formatter, so what
is pinned here is the contract a study relies on: the derived columns (the reference norm and the
stopping target), the injected case metrics, and the visibility of a redone step.
"""

from __future__ import annotations

import io

from aquaflux.solve import MarchLogger, StepReport


def _report(**kwargs) -> StepReport:
    fields = dict(step=0, cycles=10, residual_norm=2.0e-2, residual_ratio=4.0e-2, alpha=1.0)
    return StepReport(**(fields | kwargs))


def _log(**kwargs) -> tuple[MarchLogger, io.StringIO]:
    buffer = io.StringIO()
    return MarchLogger(buffer, clock=lambda: 0.0, **kwargs), buffer


def test_reports_the_reference_norm_and_the_stopping_target() -> None:
    """The residual alone cannot say how close the solve is to stopping; ``|R0|`` and the target can.

    The march reports ``|R|`` and ``|R|/|R0|`` but not ``|R0|``, so the tolerance test
    ``|R| <= atol + rtol*|R0|`` is invisible without recovering the reference.
    """
    logger, buffer = _log(rtol=1e-3)
    logger.on_step(_report(residual_norm=2.0e-2, residual_ratio=4.0e-2))

    line = buffer.getvalue()
    assert "|R0|=5.0000e-01" in line  # 2.0e-2 / 4.0e-2
    assert "target=5.000e-04" in line  # rtol * |R0|


def test_a_diverged_step_reports_no_reference_rather_than_nan() -> None:
    """A NaN ratio must not print ``|R0|=nan target=nan``.

    Every comparison against NaN is False, so a ``ratio <= 0`` guard would let a diverged step through
    into the division -- and the diverged step is exactly the line a reader is studying.
    """
    logger, buffer = _log(rtol=1e-3)
    logger.on_step(_report(residual_norm=float("nan"), residual_ratio=float("nan")))

    line = buffer.getvalue()
    assert "|R|=nan" in line and "ratio=nan" in line  # the measured values are reported as they are
    assert "|R0|" not in line and "target" not in line  # the derived ones are omitted, not faked


def test_injected_metrics_become_columns() -> None:
    """Case-specific quantities the solver cannot know ride on an injected callable."""
    logger, buffer = _log(metrics=lambda state: {"xr/h": state * 2.0})
    logger.on_checkpoint(_report(), 3.0)

    assert "xr/h=6" in buffer.getvalue()


def test_on_step_omits_metrics_because_it_has_no_state() -> None:
    """``on_step`` carries no state, so it cannot evaluate the metrics -- and must not try."""
    logger, buffer = _log(metrics=lambda state: {"xr/h": state * 2.0})
    logger.on_step(_report())  # would raise if the metrics were called with no state

    assert "xr/h" not in buffer.getvalue()


def test_a_redone_step_is_visible() -> None:
    """``cycles`` counts only the accepted attempt, so a redone step needs its own marker.

    Without this a step that escalated twice reads exactly like a cheap one, and a retry mechanism
    left unconfigured never announces its absence.
    """
    logger, buffer = _log()
    logger.on_step(_report(escalations=2))
    logger.on_step(_report(diverged_retry=True))
    logger.on_step(_report())

    escalated, retried, plain = buffer.getvalue().splitlines()
    assert "<esc=2>" in escalated
    assert "<RETRY>" in retried
    assert "<" not in plain


def test_cumulative_cycles_and_steps_run_across_phases() -> None:
    """A phase (a continuation rung) relabels the log; it does not restart the running totals."""
    logger, buffer = _log()
    logger.on_step(_report(cycles=10))
    logger.phase("rung", total=3)
    logger.on_step(_report(cycles=5))

    lines = buffer.getvalue().splitlines()
    assert "[rung 1/3" in lines[1]  # numbered automatically; the caller keeps no counter
    assert "step=   2" in lines[2] and "cum=   11" in lines[2]  # (10-2) + (5-2), offsets stripped


def test_solve_cost_is_split_into_inner_count_and_per_inner_cycles() -> None:
    """One summed count conflates *how many* solves a step needed with how hard each one was.

    It also overstates the work: the raw count carries lineax's +2 per solve, so this step's ``21``
    is really 15 cycles over 3 inner iterations -- 5 apiece, not 21.
    """
    logger, buffer = _log()
    logger.on_step(_report(cycles=21, inner_iterations=3))

    line = buffer.getvalue()
    assert "in= 3" in line
    assert "cyc= 15" in line  # 21 - 2*3
    assert "c/in=  5.0" in line


def test_a_step_whose_count_is_entirely_offset_reads_as_its_real_cost() -> None:
    """Two ideal single-cycle solves report a raw ``6``; reading that as six cycles triples the cost.

    This is the regime the correction matters in -- the cheap steps near the root, where the raw
    number is dominated by the offset rather than by any real work.
    """
    logger, buffer = _log()
    logger.on_step(_report(cycles=6, inner_iterations=2))

    assert "cyc=  2" in buffer.getvalue()


def test_on_inner_tabulates_each_inner_iteration_with_its_own_cost_and_rate() -> None:
    """The per-inner hook resolves the step summary into per-solve cost and the ``|G|`` trajectory."""
    logger, buffer = _log()
    logger.on_inner(0, 4.0e-2, 1.0e-2, 5, 1.0)
    logger.on_inner(1, 1.0e-2, 5.0e-3, 9, 0.5)

    headings, first, second = (
        line for line in buffer.getvalue().splitlines() if line.startswith("| ")
    )
    for heading in ("inner", "cyc", "|G| in", "|G| out", "rate", "alpha"):
        assert heading in headings
    # cyc is 5 - 2, this single solve's offset; rate is the inner contraction 1.0e-2 / 4.0e-2.
    assert first == "|     0 |    3 |  4.000e-02 |  1.000e-02 |  0.250 | 1.000 |"
    assert second == "|     1 |    7 |  1.000e-02 |  5.000e-03 |  0.500 | 0.500 |"


def test_the_inner_table_opens_on_each_attempt_and_the_step_line_closes_it() -> None:
    """One step is one block: a retry re-opens at ``inner=0``, and the step line ends the record.

    Without the re-open, a redone step's rows run into the abandoned attempt's and the block stops
    being a record of one step's work -- which is exactly what makes a retry's cost legible.
    """
    logger, buffer = _log()
    logger.on_inner(0, 4.0e-2, 3.0e-2, 5, 1.0)
    logger.on_inner(0, 4.0e-2, 1.0e-2, 5, 1.0)  # the step redone at an escalated beta
    logger.on_step(_report(escalations=1))

    lines = buffer.getvalue().splitlines()
    assert lines[0].startswith("+- step 1")  # numbered for the step it precedes, not the last one
    assert lines[5].startswith("+- step 1")  # the retry gets its own block
    assert lines[-2].startswith("+--")  # the step line closed the table
    assert lines[-1].lstrip().startswith("t=") and "<esc=1>" in lines[-1]


def test_a_block_heads_with_the_residual_the_step_inherits() -> None:
    """The step's outcome cannot head its own block, so the title carries where the step starts from.

    Buffering the block to lead with the outcome would cost the live progress the rows exist to give.
    """
    logger, buffer = _log()
    logger.on_step(_report(residual_norm=2.0e-2))
    logger.on_inner(0, 2.0e-2, 1.0e-2, 5, 1.0)

    assert "step 2" in buffer.getvalue().splitlines()[1]
    assert "from |R|=2.0000e-02" in buffer.getvalue().splitlines()[1]
