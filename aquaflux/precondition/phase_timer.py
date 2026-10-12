"""A stopwatch that splits one preconditioner refresh into named, consecutive phases."""

from __future__ import annotations

import time

__all__ = ["PhaseTimer"]


class PhaseTimer:
    """Accumulate ``(name, seconds)`` for the successive parts of one refresh.

    Each :meth:`lap` closes the part that started when the timer was created or the previous ``lap``
    returned, so a refresh names its phases inline rather than carrying start times around. Phases are
    consecutive and non-overlapping by construction, which is what makes them add up to the total.

    Examples
    --------
    >>> timer = PhaseTimer()
    >>> timer.lap("probe")          # doctest: +SKIP
    >>> timer.lap("factor")         # doctest: +SKIP
    >>> timer.phases()              # doctest: +SKIP
    (('probe', 12.4), ('factor', 6.1))
    """

    def __init__(self, clock: object = None) -> None:
        self._clock = time.perf_counter if clock is None else clock
        self._phases: list[tuple[str, float]] = []
        self._mark = self._clock()

    def lap(self, name: str) -> None:
        """Close the current phase under ``name`` and start the next."""
        now = self._clock()
        self._phases.append((str(name), now - self._mark))
        self._mark = now

    def phases(self) -> tuple[tuple[str, float], ...]:
        """The phases closed so far, in order."""
        return tuple(self._phases)
