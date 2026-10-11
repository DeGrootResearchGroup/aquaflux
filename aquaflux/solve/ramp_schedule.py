"""How far along a geometric parameter ramp each outer step of a march runs, and when the ramp ends.

A within-march homotopy walks a physical parameter from an easy value down to the case's own in
*stations*, one or more outer steps each, and the last station is the target problem itself. Where the
parameter sits at each step is a question about the march, not about the physics: it needs a station
count, how long each is held, and -- optionally -- a signal from the march that ends the walk early.
:class:`RampSchedule` holds those settings; :meth:`RampSchedule.walk` hands each march its own
:class:`RampWalk`, which records where each step ran. A homotopy turns a walk's
:attr:`RampPosition.progress` into its parameter (a geometric ramp from an anchor ``a`` runs at
``a ** (1 - progress)``).

**Why a ramp may end early.** A fixed station count is a guess at how long the march needs before the
target problem can be taken on, and measured on a Reynolds-averaged backward-facing step its best value
is not a property of the parameter's step size at all: it is the number of steps the march's
pseudo-time shift takes to reach its floor, plus the few the state then needs there to *settle* -- the
first step at the floor taken at full length. A ramp that arrives before the settle hands the target an
unsettled state, and one that arrives well after it spends steps on stations it no longer needs. Ending
the walk when an injected test of the previous step's report says the march has settled
(:attr:`RampSchedule.end_when`, typically :meth:`~aquaflux.solve.ShiftStrengthControl.settled`) removes
the second waste whatever the count, which then only sets the pace. The rest of the span is walked in
:attr:`RampSchedule.finish` equal geometric steps rather than one jump: a large jump at the smallest
shift is tolerated but undoes the settle, so the target has to settle again.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import NamedTuple

from .strategy import StepReport


class RampPosition(NamedTuple):
    """Where one outer step of a ramp runs.

    Attributes
    ----------
    station : int
        Which station the step runs, as an index that only ever increases; only equality between
        adjacent steps means anything (a change marks the step that entered a new station).
    progress : float
        How far along the ramp the station is, from ``0.0`` (the anchor) to ``1.0`` (the target).
    """

    station: int
    progress: float

    @property
    def arrived(self) -> bool:
        """Whether the step runs the target problem."""
        return self.progress >= 1.0


@dataclasses.dataclass(frozen=True)
class RampSchedule:
    """A geometric ramp's stations, how long each is held, and what may end it early.

    Attributes
    ----------
    stations : int
        How many stations the span is divided into before the target, ``>= 1``. Station ``s`` sits at
        ``progress = s / stations``. With :attr:`end_when` set this is the ramp's **pace**, the length it
        would have if nothing ended it.
    steps_per_station : int
        Outer steps each ramp station is held for, ``>= 1``. The target is held for as long as the march
        needs.
    end_when : callable or None
        ``StepReport -> bool``, read on each step's report before the next step enters its station. The
        first report off the target for which it is true ends the walk: the remaining span is walked in
        :attr:`finish` steps from the next one on. ``None`` (default) walks every station.
    finish : int
        How many steps the span left when :attr:`end_when` fires is walked in, in equal geometric steps,
        the last being the target, ``>= 1``. ``1`` (default) jumps straight to the target.

    Raises
    ------
    ValueError
        If a count is below one, if ``finish`` is set without ``end_when``, or if ``end_when`` is set with
        ``steps_per_station`` other than one -- the early end was measured on a ramp of one step per
        station, and on a longer one it would cut a station short.

    Examples
    --------
    Sixteen stations, ended on the march's settle and finished in three steps::

        RampSchedule(16, end_when=control.settled, finish=3)
    """

    stations: int
    steps_per_station: int = 1
    end_when: Callable[[StepReport], bool] | None = None
    finish: int = 1

    def __post_init__(self) -> None:
        for name in ("stations", "steps_per_station", "finish"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)!r}.")
        if self.end_when is None and self.finish != 1:
            raise ValueError(
                "finish says how the span left when the ramp is ended early is walked, and this ramp "
                "has nothing to end it."
            )
        if self.end_when is not None and self.steps_per_station != 1:
            raise ValueError(
                "a ramp ended early is one of one step per station; with steps_per_station = "
                f"{self.steps_per_station} the end would cut a station short."
            )

    @property
    def ramp_steps(self) -> int:
        """Outer steps the ramp occupies before the target when nothing ends it early."""
        return self.stations * self.steps_per_station

    def walk(self) -> RampWalk:
        """A fresh walk along this schedule, for one march."""
        return RampWalk(self)


class RampWalk:
    """One march's walk along a :class:`RampSchedule`: where each outer step ran.

    Mutable, because whether the ramp has ended depends on the reports of steps already taken, and a
    march asks about the step it is entering. Each step is entered once, in order; entering a step again
    returns what it was given the first time, so a march may ask before and during its first step.

    Parameters
    ----------
    schedule : RampSchedule
        The settings walked.
    """

    def __init__(self, schedule: RampSchedule) -> None:
        self.schedule = schedule
        self._positions: list[RampPosition] = []
        # The step the early end fired before, and the progress it fired at; None until it fires.
        self._ended: tuple[int, float] | None = None

    def position(self, step: int) -> RampPosition:
        """Where an already entered step ran.

        Raises
        ------
        IndexError
            If ``step`` has not been entered.
        """
        if not 0 <= step < len(self._positions):
            raise IndexError(f"step {step} has not been entered ({len(self._positions)} so far).")
        return self._positions[step]

    @property
    def ended_before(self) -> int | None:
        """The first step after the one whose report ended the ramp, or ``None`` if it has not ended."""
        return None if self._ended is None else self._ended[0]

    def enter(self, step: int, previous: StepReport | None) -> RampPosition:
        """Where outer step ``step`` runs, given the report of the step before it.

        Parameters
        ----------
        step : int
            The step being entered, ``0`` first; each is entered after the one before it.
        previous : StepReport or None
            The report of step ``step - 1``; ``None`` for the first.

        Returns
        -------
        RampPosition
            The station and progress the step runs.

        Raises
        ------
        ValueError
            If a step is entered out of order.
        """
        if step < len(self._positions):
            return self._positions[step]
        if step != len(self._positions):
            raise ValueError(
                f"RampWalk.enter got step {step} after {len(self._positions)} steps; each step is "
                "entered once, in order."
            )
        schedule = self.schedule
        if (
            self._ended is None
            and schedule.end_when is not None
            and previous is not None
            and not previous.arrived
            and schedule.end_when(previous)
        ):
            self._ended = (step, self._positions[-1].progress)
        if self._ended is None:
            station = min(step // schedule.steps_per_station, schedule.stations)
            position = RampPosition(station, station / schedule.stations)
        else:
            start, from_progress = self._ended
            taken = min(step - start + 1, schedule.finish)
            progress = (
                1.0
                if taken == schedule.finish
                else from_progress + (1.0 - from_progress) * taken / schedule.finish
            )
            # A new station on every finishing step, numbered on from the one the end fired at.
            station = self._positions[-1].station + (progress != self._positions[-1].progress)
            position = RampPosition(station, progress)
        self._positions.append(position)
        return position
