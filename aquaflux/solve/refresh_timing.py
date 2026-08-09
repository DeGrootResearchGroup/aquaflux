"""What a preconditioner refresh did, and what each part of it cost.

A β-tracking march re-forms its frozen preconditioner as the state and the pseudo-transient shift move,
and that refresh is a large share of the march's wall time — so which branch it took and where the time
went inside it are load-bearing facts, not decoration. Reported as one aggregate they are not actionable:
a refresh that is dominated by the graph-coloured Jacobian probe and one dominated by the multigrid setup
cost the same on the clock and call for opposite fixes.

:class:`RefreshTiming` is the record an observer receives, and :class:`PhaseTimer` is how a refresh
accumulates it without threading start times through its own body.
"""

from __future__ import annotations

import dataclasses
import time


@dataclasses.dataclass(frozen=True)
class RefreshTiming:
    """One preconditioner refresh: which branch ran, what it cost in total, and its parts.

    Attributes
    ----------
    kind : str
        What the refresh actually did — ``"full"`` (re-probed the Jacobian and re-factored), ``"shift"``
        (re-added the pseudo-transient diagonal to the frozen Jacobian and re-factored) or ``"none"``
        (the gate declined; the standing factorization was reused).
    seconds : float
        Wall time for the whole refresh, including any part not attributed to a phase.
    phases : tuple of (str, float)
        The named parts in the order they ran, each with its own wall time. Empty for a refresh that
        reports no breakdown (a ``"none"``, or a preconditioner that does not instrument itself).
    """

    kind: str
    seconds: float
    phases: tuple[tuple[str, float], ...] = ()

    @property
    def unattributed(self) -> float:
        """Wall time not covered by any phase — nonzero means the breakdown is missing something."""
        return self.seconds - sum(seconds for _, seconds in self.phases)


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
