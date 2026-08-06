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
    logger.phase("rung 2/3")
    logger.on_step(_report(cycles=5))

    lines = buffer.getvalue().splitlines()
    assert "[rung 2/3" in lines[1]
    assert "step=   2" in lines[2] and "cum=   15" in lines[2]
