"""Unit: the preconditioner-refresh cost record and the timer that accumulates it.

Pure bookkeeping, so it is tested with an injected clock rather than by sleeping -- the assertions are
about which interval lands under which name, which a wall-clock test could only check loosely.
"""

from __future__ import annotations

import pytest
from aquaflux.solve import PhaseTimer, RefreshTiming


class _FakeClock:
    """A clock returning each supplied reading in turn, so phase durations are exact."""

    def __init__(self, *readings: float) -> None:
        self._readings = list(readings)

    def __call__(self) -> float:
        return self._readings.pop(0)


def test_each_lap_closes_the_interval_since_the_previous_one() -> None:
    """Phases are consecutive and non-overlapping -- which is what makes them add up to the total."""
    timer = PhaseTimer(clock=_FakeClock(10.0, 12.5, 13.0, 20.0))

    timer.lap("probe")
    timer.lap("assemble")
    timer.lap("refactor")

    assert timer.phases() == (("probe", 2.5), ("assemble", 0.5), ("refactor", 7.0))


def test_a_timer_with_no_laps_reports_no_phases() -> None:
    """A branch that does no attributable work reports an empty breakdown, not a zero-length phase."""
    assert PhaseTimer(clock=_FakeClock(4.0)).phases() == ()


def test_unattributed_is_the_total_the_phases_do_not_account_for() -> None:
    """A breakdown that silently fails to add up reads as complete, so the remainder is exposed."""
    timing = RefreshTiming("full", 10.0, (("probe", 6.0), ("refactor", 3.0)))
    assert timing.unattributed == pytest.approx(1.0)


def test_a_timing_with_no_phases_attributes_nothing() -> None:
    """The whole total is unattributed when the preconditioner does not instrument itself."""
    assert RefreshTiming("none", 2.0).unattributed == pytest.approx(2.0)
