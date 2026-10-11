"""A geometric ramp's schedule and one march's walk along it: stations, progress, and the early end."""

from __future__ import annotations

import pytest
from aquaflux.solve import RampPosition, RampSchedule, StepReport


def _report(step: int, *, arrived: bool = False, alpha: float = 0.5) -> StepReport:
    """A step report carrying what a ramp's end test may read, and nothing else of note."""
    return StepReport(
        step=step, cycles=3, residual_norm=1.0, residual_ratio=1.0, alpha=alpha, arrived=arrived
    )


def _walk(schedule: RampSchedule, reports: list[StepReport]) -> list[RampPosition]:
    """Enter one step per report plus the first, handing each step the report before it."""
    walk = schedule.walk()
    positions = [walk.enter(0, None)]
    for step, previous in enumerate(reports, start=1):
        positions.append(walk.enter(step, previous))
    return positions


def _full_length(report: StepReport) -> bool:
    return report.alpha >= 1.0


def test_a_fixed_ramp_holds_each_station_for_its_budget_and_then_arrives() -> None:
    """Station ``s`` at ``progress s / stations``, held ``steps_per_station`` steps; the target after."""
    positions = _walk(RampSchedule(3, 2), [_report(i) for i in range(7)])

    assert [p.station for p in positions] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [p.progress for p in positions] == [0.0, 0.0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1.0, 1.0]
    assert [p.arrived for p in positions] == [False] * 6 + [True, True]


def test_the_end_test_moves_the_next_step_straight_to_the_target() -> None:
    """With ``finish = 1`` the step after the one the test fires on runs the target, whatever is left."""
    reports = [_report(0), _report(1), _report(2, alpha=1.0), _report(3), _report(4)]
    positions = _walk(RampSchedule(8, end_when=_full_length), reports)

    assert [p.progress for p in positions[:3]] == [0.0, 1 / 8, 2 / 8]
    assert positions[3].arrived  # the report of step 2 ended it
    assert positions[3].station != positions[2].station  # a change the march can see
    assert all(p.arrived for p in positions[3:])


def test_a_finish_walks_the_rest_in_equal_steps_of_progress() -> None:
    """Equal steps of progress are equal geometric steps of the parameter, the last landing on 1."""
    reports = [_report(0), _report(1, alpha=1.0)] + [_report(i) for i in range(2, 7)]
    positions = _walk(RampSchedule(8, end_when=_full_length, finish=3), reports)

    start = positions[1].progress  # where the end fired: station 1 of 8
    assert start == pytest.approx(1 / 8)
    expected = [start + (1 - start) * k / 3 for k in (1, 2)] + [1.0]
    assert [p.progress for p in positions[2:5]] == pytest.approx(expected, abs=0.0)
    assert positions[4].progress == 1.0 and positions[4].arrived
    assert not positions[3].arrived
    stations = [p.station for p in positions[1:5]]
    assert stations == sorted(set(stations))  # a new station on every finishing step
    assert positions[5] == positions[4] and positions[6] == positions[4]


def test_the_end_test_is_read_off_the_ramp_only_and_once() -> None:
    """A report from the target cannot end anything, and a second firing does not restart the finish."""
    seen: list[int] = []

    def records(report: StepReport) -> bool:
        seen.append(report.step)
        return True

    positions = _walk(RampSchedule(2, end_when=records, finish=2), [_report(i) for i in range(5)])

    assert seen == [0]  # asked once: after it fired, nothing asks again
    assert [p.progress for p in positions] == [0.0, 0.5, 1.0, 1.0, 1.0, 1.0]


def test_a_ramp_that_arrives_before_its_end_fires_is_not_ended_by_a_report_from_the_target() -> (
    None
):
    """The target is reached by the stations; a report taken there is not one the end test may read."""
    seen: list[int] = []

    def records(report: StepReport) -> bool:
        seen.append(report.step)
        return False

    reports = [_report(0), _report(1, arrived=True), _report(2, arrived=True)]
    positions = _walk(RampSchedule(1, end_when=records), reports)

    assert seen == [0]
    assert [p.arrived for p in positions] == [False, True, True, True]


def test_a_step_is_entered_in_order_and_a_repeat_returns_what_it_was_given() -> None:
    """The march measures its anchor at step 0 before taking it, so step 0 is entered twice."""
    walk = RampSchedule(4, end_when=_full_length).walk()
    first = walk.enter(0, None)

    assert (
        walk.enter(0, _report(0, alpha=1.0)) == first
    )  # not re-decided, though the test would fire
    assert walk.position(0) == first
    with pytest.raises(ValueError, match="entered once, in order"):
        walk.enter(2, _report(1))
    with pytest.raises(IndexError, match="has not been entered"):
        walk.position(1)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (dict(stations=0), "stations must be >= 1, got 0"),
        (dict(stations=2, steps_per_station=0), "steps_per_station must be >= 1, got 0"),
        (dict(stations=2, end_when=_full_length, finish=0), "finish must be >= 1, got 0"),
        (dict(stations=2, finish=3), "has nothing to end it"),
        (dict(stations=2, steps_per_station=2, end_when=_full_length), "cut a station short"),
    ],
)
def test_the_schedule_refuses_settings_it_cannot_walk(settings, message) -> None:
    with pytest.raises(ValueError, match=message):
        RampSchedule(**settings)
