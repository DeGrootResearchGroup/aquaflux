"""What a preconditioner refresh did, and what each part of it cost.

A β-tracking march re-forms its frozen preconditioner as the state and the pseudo-transient shift move,
and that refresh is a large share of the march's wall time — so which branch it took and where the time
went inside it are load-bearing facts, not decoration. Reported as one aggregate they are not actionable:
a refresh that is dominated by the graph-coloured Jacobian probe and one dominated by the multigrid setup
cost the same on the clock and call for opposite fixes.

:class:`RefreshTiming` is the record an observer receives; the preconditioner that refreshes fills it
in.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class RefreshTiming:
    """One preconditioner refresh: which branch ran, what it cost in total, and its parts.

    Attributes
    ----------
    kind : str
        What the refresh actually did — ``"full"`` (re-probed the Jacobian and re-factored), ``"inner"``
        (the same, mid-step, at the inner loop's iterate) or ``"none"`` (the standing factorization was
        reused).
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
