"""Record a march's per-step reports as rows of a comma-separated-values (CSV) file.

:class:`~aquaflux.solve.MarchLogger` writes a table for a person to read as the run goes; this writes
the same steps for a program to read -- a convergence plot, a comparison of two runs, a monitor
following a run that has not finished. One row per observed step, holding every field of its
:class:`~aquaflux.solve.StepReport` at full precision, flushed as it is written, so the file is a
complete record of the march up to its last line however the run ends.
"""

from __future__ import annotations

import csv
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .strategy import StepReport

__all__ = ["StepHistory"]


class StepHistory:
    """Write one CSV row per observed march step, flushed as it is written.

    Not an :class:`equinox.Module`: it holds an open file and a step counter, so it is a host-side
    observer like :class:`~aquaflux.solve.MarchLogger`, attached through ``on_checkpoint``.

    The columns are, in order: ``step``, the 1-based count of steps observed so far, which keeps
    counting across the segments of a continuation; ``seconds``, the time since this object was made;
    every field of :class:`~aquaflux.solve.StepReport` under its own name, except that the report's
    ``step`` -- its index within its segment, restarting at each -- is written as ``segment_step``; and
    ``restart_cycles``, the report's offset-corrected cycle count. Booleans are written as ``0`` /
    ``1`` and floats with every digit that distinguishes them.

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
    ...     solve_coupled(coupled, on_checkpoint=history.on_checkpoint)
    """

    #: The columns, in the order they are written.
    COLUMNS: tuple[str, ...] = (
        "step",
        "seconds",
        "segment_step",
        *(name for name in StepReport._fields if name != "step"),
        "restart_cycles",
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
        self._writer.writerow(self.COLUMNS)
        self._file.flush()

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
        self._count += 1
        fields = report._asdict()
        segment_step = fields.pop("step")
        row = [self._count, self._clock() - self._start, segment_step, *fields.values()]
        row.append(report.restart_cycles)
        self._writer.writerow([_cell(value) for value in row])
        self._file.flush()

    def close(self) -> None:
        """Close the file; further steps cannot be written."""
        self._file.close()

    def __enter__(self) -> StepHistory:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _cell(value: object) -> str:
    """One value as CSV text: a boolean as ``0``/``1``, a float by its shortest exact repr."""
    # A report's numbers may be NumPy or JAX scalars, whose own repr names their type.
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return repr(value)
    return str(value)
