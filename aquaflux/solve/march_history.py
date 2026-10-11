"""Record a march's per-step reports as rows of a comma-separated-values (CSV) file.

:class:`~aquaflux.solve.MarchLogger` writes a table for a person to read as the run goes; this writes
the same steps for a program to read -- a convergence plot, a comparison of two runs, a monitor
following a run that has not finished. One row per observed step, holding every field of its
:class:`~aquaflux.solve.StepReport` at full precision together with what happened around the step
that the report does not carry -- the preconditioner refits it paid for, why it was redone, and its
residual split by equation -- flushed as it is written, so the file is a complete record of the march
up to its last line however the run ends.

:class:`MarchRecorder` declares what such a recorder offers -- every march hook it can be handed -- so
a runner fanning the hooks out to several recorders asks for the capability with ``isinstance`` rather
than looking each hook up by name, which would skip a misspelled one without a word.
"""

from __future__ import annotations

import csv
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .refresh_timing import RefreshTiming
from .strategy import StepReport

__all__ = ["MarchRecorder", "StepHistory"]


@runtime_checkable
class MarchRecorder(Protocol):
    """A recorder of a whole march: every step, every retry, every preconditioner refit and, when it
    wants them, the per-equation residuals.

    A recorder offering only some of these -- a checkpointer, which keeps states and nothing else -- is
    handed ``on_checkpoint`` alone. ``on_residuals`` may be ``None``: each per-equation report costs a
    residual evaluation per step, so a recorder that does not want them says so rather than being asked.
    """

    on_residuals: Callable[[Mapping[str, float]], None] | None

    def on_checkpoint(self, report: StepReport, state: Any) -> None:
        """Record one accepted step and the state it reached."""
        ...

    def on_retry(self, reason: str, attempt: int, beta: float) -> None:
        """Record that the step under way is being redone, why, and at what shift."""
        ...

    def on_refresh(self, timing: RefreshTiming) -> None:
        """Record a preconditioner refit and what it cost."""
        ...


#: The prefix of a per-equation residual column: ``residual_of_u``, ``residual_of_omega``, ...
EQUATION_PREFIX = "residual_of_"

#: A :class:`RefreshTiming` kind that reused the standing preconditioner rather than refitting it.
_REUSED = "none"


class StepHistory:
    """Write one CSV row per observed march step, flushed as it is written.

    Not an :class:`equinox.Module`: it holds an open file and what it has heard since the last row, so
    it is a host-side observer like :class:`~aquaflux.solve.MarchLogger`. Each method is the march hook
    of the same name: ``on_checkpoint`` writes the row, and ``on_refresh``, ``on_retry`` and
    ``on_residuals``, called while the step is under way, fill in the row it then writes.

    The columns are, in order: ``step``, the 1-based count of steps observed so far, which keeps
    counting across the segments of a continuation; ``seconds``, the time since this object was made;
    every field of :class:`~aquaflux.solve.StepReport` under its own name, except that the report's
    ``step`` -- its index within its segment, restarting at each -- is written as ``segment_step``;
    ``restart_cycles``, the report's offset-corrected cycle count; ``refits`` and ``refit_seconds``,
    how many times the preconditioner was refitted for the step and the wall time that took (a refresh
    that reused the standing preconditioner is not a refit); ``retry_reasons``, why the step was redone,
    one reason per redo joined by ``;`` and empty for a step taken as-is; and then one
    ``residual_of_<equation>`` column per equation the march's measure names, when it names any. Booleans
    are written as ``0`` / ``1`` and floats with every digit that distinguishes them.

    **The header is written with the first row**, since the equation columns are known only once the
    march reports them; until then the file exists and is empty.

    Parameters
    ----------
    path : path-like
        The file, replaced if it exists. Its parent directory must exist.
    clock : callable, optional
        ``() -> float`` seconds, for the ``seconds`` column; injected so a test can supply a
        deterministic one. Defaults to :func:`time.monotonic`.

    Examples
    --------
    >>> with StepHistory("results/history.csv") as history:  # doctest: +SKIP
    ...     solve_coupled(
    ...         coupled,
    ...         on_checkpoint=history.on_checkpoint,
    ...         on_retry=history.on_retry,
    ...         on_residuals=history.on_residuals,
    ...     )
    """

    #: The columns every history holds, in the order they are written; the per-equation residual
    #: columns follow them.
    COLUMNS: tuple[str, ...] = (
        "step",
        "seconds",
        "segment_step",
        *(name for name in StepReport._fields if name != "step"),
        "restart_cycles",
        "refits",
        "refit_seconds",
        "retry_reasons",
    )

    def __init__(
        self, path: str | os.PathLike[str], *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.path = Path(path)
        self._clock = clock
        self._start = clock()
        self._count = 0
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._equations: tuple[str, ...] | None = None
        self._refits: list[RefreshTiming] = []
        self._retries: list[str] = []
        self._residuals: Mapping[str, float] = {}

    def on_refresh(self, timing: RefreshTiming) -> None:
        """Record a preconditioner refresh for the step under way -- a session's ``observer`` hook.

        Parameters
        ----------
        timing : RefreshTiming
            What the refresh did and cost.
        """
        if timing.kind != _REUSED:
            self._refits.append(timing)

    def on_retry(self, reason: str, attempt: int, beta: float) -> None:
        """Record that the step under way is being redone, and why -- the march's ``on_retry`` hook.

        Parameters
        ----------
        reason : str
            The march's reason (``"cycles"``, ``"alpha"``, ``"diverged"`` or ``"solver"``).
        attempt, beta : int, float
            The attempt and the shift it runs at; already in the step's report as ``escalations``
            and ``shift``, so not recorded again.
        """
        del attempt, beta
        self._retries.append(reason)

    def on_residuals(self, residuals: Mapping[str, float]) -> None:
        """Record the step's residual by equation -- the march's ``on_residuals`` hook.

        Parameters
        ----------
        residuals : mapping of {str: float}
            Each equation's term of the march's measure, in block order.

        Raises
        ------
        ValueError
            If it names different equations from those the header already holds.
        """
        if self._equations is not None and tuple(residuals) != self._equations:
            raise ValueError(
                f"this history records the equations {self._equations}, and a step reported "
                f"{tuple(residuals)}."
            )
        self._residuals = dict(residuals)

    def on_checkpoint(self, report: StepReport, state: Any = None) -> None:
        """Write ``report`` as the next row -- the ``on_checkpoint(report, state)`` observer.

        Parameters
        ----------
        report : StepReport
            The step just taken.
        state : object, optional
            The state it produced; not recorded.
        """
        del state
        if self._equations is None:
            self._equations = tuple(self._residuals)
            self._writer.writerow(
                (*self.COLUMNS, *(EQUATION_PREFIX + name for name in self._equations))
            )
        self._count += 1
        fields = report._asdict()
        segment_step = fields.pop("step")
        row = [
            self._count,
            self._clock() - self._start,
            segment_step,
            *fields.values(),
            report.restart_cycles,
            len(self._refits),
            sum(timing.seconds for timing in self._refits),
            ";".join(self._retries),
        ]
        # A step that reported no residuals (a march whose measure named none) leaves its cells empty.
        row += [self._residuals.get(name, "") for name in self._equations]
        self._writer.writerow([_cell(value) for value in row])
        self._file.flush()
        self._refits, self._retries, self._residuals = [], [], {}

    def close(self) -> None:
        """Close the file; further steps cannot be written."""
        self._file.close()

    def __enter__(self) -> StepHistory:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _cell(value: object) -> str:
    """One value as CSV text: a boolean as ``0``/``1``, a float by its shortest exact repr."""
    # A report's numbers may be NumPy or JAX scalars, whose own repr names their type. Asked by the
    # one method both share rather than by type, so this module needs neither library: it is how an
    # array scalar is recognized, not a capability any class of this package declares.
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return repr(value)
    return str(value)
