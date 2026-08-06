"""Streaming per-step log for an observed march -- the reporting half of the ``on_step`` seam.

:func:`~aquaflux.solve.forward_march` (and the solvers that drive it) hand every observed step to an
``on_step`` / ``on_checkpoint`` callback, but a caller that wants to *watch* a long march then has to
write the formatter itself. Each study writing its own means the same field extraction and format
string are re-derived per script, they drift, and a reporting gap is fixed in one and not the others.

:class:`MarchLogger` owns everything derivable from a :class:`~aquaflux.solve.StepReport`. What it can
*not* know is the case: a reattachment length, a drag coefficient, a peak temperature. Those come from
an injected ``metrics`` callable mapping the checkpoint state to named columns, so the solver stays
free of case specifics while the log still carries the quantity a study is actually steering by.

It reports the **reference norm and the stopping target** alongside the residual. A log that prints
only the residual leaves the reader unable to tell how close the solve is to stopping -- the tolerance
test is ``‖R‖ <= atol + rtol·‖R₀‖``, so without ``‖R₀‖`` the target is invisible.

Output is one flushed line per step, so a multi-hour march can be tailed while it runs rather than
read after it ends.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping
from typing import IO, Any

from .march import StepReport

__all__ = ["MarchLogger"]


class MarchLogger:
    """Format and stream one line per observed march step.

    Not an :class:`equinox.Module`: it carries mutable counters (cumulative cycles, the phase label)
    and writes to a stream, so it is a host-side observer rather than a pytree the solve carries.

    Parameters
    ----------
    stream : file-like, optional
        Where to write. Defaults to ``sys.stdout``. Each line is flushed.
    metrics : callable, optional
        ``state -> mapping of name to float``, appended as extra columns by :meth:`on_checkpoint`.
        The seam for case-specific quantities (e.g. a reattachment length) the solver cannot know.
        Ignored by :meth:`on_step`, which has no state.
    rtol, atol : float, optional
        The stopping tolerances the march is running under, used to report the target
        ``atol + rtol·‖R₀‖`` next to the residual. Omit both to suppress the target column.
    clock : callable, optional
        ``() -> float`` seconds source for the elapsed-time column; injected so a test can supply a
        deterministic clock. Defaults to :func:`time.monotonic`.

    Examples
    --------
    >>> log = MarchLogger(metrics=lambda s: {"xr/h": reattachment(s)}, rtol=1e-3)  # doctest: +SKIP
    >>> solve_coupled(coupled, on_checkpoint=log.on_checkpoint)  # doctest: +SKIP
    """

    def __init__(
        self,
        stream: IO[str] | None = None,
        *,
        metrics: Callable[[Any], Mapping[str, float]] | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._stream = sys.stdout if stream is None else stream
        self._metrics = metrics
        self._rtol = rtol
        self._atol = atol
        self._clock = time.monotonic if clock is None else clock
        self._start = self._clock()
        self._cumulative_cycles = 0
        self._steps = 0
        self._phase = ""

    def phase(self, label: str) -> None:
        """Start a labelled phase (a continuation rung, a refreshed segment) and write a header.

        The step counter and cumulative cycles keep running across phases, so the log reads as one
        march; only the label changes.
        """
        self._phase = label
        self._write(f"[{label}  t={self._clock() - self._start:.0f}s]")

    def on_step(self, report: StepReport) -> None:
        """``on_step`` callback: log a step with no case metrics (no state is available here)."""
        self._log(report, None)

    def on_checkpoint(self, report: StepReport, state: Any) -> None:
        """``on_checkpoint`` callback: log a step and append the injected case metrics."""
        self._log(report, state)

    def _log(self, report: StepReport, state: Any | None) -> None:
        self._steps += 1
        self._cumulative_cycles += int(report.cycles)
        fields = [
            f"t={self._clock() - self._start:6.0f}s",
            f"step={self._steps:4d}",
            f"beta={report.shift:7.4f}",
            f"cyc={report.cycles:3d}",
            f"cum={self._cumulative_cycles:5d}",
            f"|R|={report.residual_norm:.4e}",
            f"ratio={report.residual_ratio:.3e}",
        ]
        reference = self._reference_norm(report)
        if reference is not None:
            fields.append(f"|R0|={reference:.4e}")
            target = self._target(reference)
            if target is not None:
                fields.append(f"target={target:.3e}")
        fields.append(f"a={report.alpha:.3f}")
        # `cycles` counts only the ACCEPTED attempt, so a redone step is otherwise indistinguishable
        # from a cheap one -- and a retry mechanism left unconfigured never announces its absence.
        if report.escalations or report.diverged_retry:
            marks = []
            if report.escalations:
                marks.append(f"esc={report.escalations}")
            if report.diverged_retry:
                marks.append("RETRY")
            fields.append("<" + ",".join(marks) + ">")
        if self._phase:
            fields.insert(1, f"[{self._phase}]")
        if state is not None and self._metrics is not None:
            fields += [f"{name}={value:.4g}" for name, value in self._metrics(state).items()]
        self._write("  " + " ".join(fields))

    @staticmethod
    def _reference_norm(report: StepReport) -> float | None:
        """Recover ``‖R₀‖`` from the report's own two residual fields (``‖R‖`` and ``‖R‖/‖R₀‖``)."""
        # `not > 0` rather than `<= 0`: a diverged step reports a NaN ratio, and every NaN comparison
        # is False, so a `<= 0` guard would let it through and print `|R0|=nan target=nan`.
        if not report.residual_ratio > 0.0:
            return None
        return report.residual_norm / report.residual_ratio

    def _target(self, reference: float) -> float | None:
        """The stopping target ``atol + rtol·‖R₀‖``, or ``None`` when no tolerance was supplied."""
        if self._rtol is None and self._atol is None:
            return None
        return (self._atol or 0.0) + (self._rtol or 0.0) * reference

    def _write(self, line: str) -> None:
        self._stream.write(line + "\n")
        self._stream.flush()
