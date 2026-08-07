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

Solve cost is reported the same way. An implicit timestep spreads its linear-solve count over several
inner Newton iterations, so a single summed count conflates *how many* solves the step needed with how
hard each one was -- and, because the raw count carries a fixed per-solve offset, a cheap step reads as
several times its real cost. This module reports the offset-corrected total, the inner count, and the
per-inner average together, and :meth:`MarchLogger.on_inner` streams each inner iteration's own solve
cost and ``‖G‖`` trajectory for the step being studied.

Output is one flushed line per step, so a multi-hour march can be tailed while it runs rather than
read after it ends.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping
from typing import IO, Any

from ..text_table import Column, TextTable
from .linear import restart_cycles
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

    #: The inner-iteration table. ``|G|`` is the implicit timestep's own residual (the one the inner
    #: Newton loop drives to zero), and ``rate`` its per-iteration contraction ``‖G‖out / ‖G‖in`` --
    #: the number that says whether the inner loop is converging or merely grinding.
    _INNER_TABLE = TextTable(
        [
            Column("inner", 5, "d"),
            Column("cyc", 4, "d"),
            Column("|G| in", 10, ".3e"),
            Column("|G| out", 10, ".3e"),
            Column("rate", 6, ".3f"),
            Column("alpha", 5, ".3f"),
        ]
    )

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
        self._phases = 0
        self._phase = ""
        self._open = False
        self._previous_norm: float | None = None

    def note(self, text: str) -> None:
        """Write an arbitrary line (a run header, a configuration echo) to the log's own stream.

        Exists so a driver never reaches for ``print`` alongside the logger: that splits the run across
        two destinations, and a log written to a file then silently loses whichever lines went to
        stdout -- including the configuration echo needed to interpret the run.
        """
        self._write(text)

    def phase(self, label: str, total: int | None = None) -> None:
        """Start a labelled phase (a continuation rung, a refreshed segment) and write a header.

        Phases are **numbered automatically** in call order, so a caller does not keep its own counter:
        ``phase("rung", total=3)`` yields ``rung 1/3``, ``rung 2/3``, ... The step counter and cumulative
        cycles keep running across phases, so the log still reads as one march.
        """
        self._phases += 1
        self._phase = f"{label} {self._phases}" + (f"/{total}" if total is not None else "")
        self._write(f"[{self._phase}  t={self._clock() - self._start:.0f}s]")

    def on_step(self, report: StepReport) -> None:
        """``on_step`` callback: log a step with no case metrics (no state is available here)."""
        self._log(report, None)

    def on_checkpoint(self, report: StepReport, state: Any) -> None:
        """``on_checkpoint`` callback: log a step and append the injected case metrics."""
        self._log(report, state)

    def on_inner(
        self,
        index: int,
        g_before: float,
        g_after: float,
        cycles: int,
        alpha: float,
    ) -> None:
        """``inner_observer`` callback: tabulate ONE inner Newton iteration of an implicit timestep.

        Matches the hook signature :class:`~aquaflux.solve.DualTimeStep` calls once per inner iteration,
        so a driver wires it as ``inner_observer=logger.on_inner``. The step line reports only the inner
        *count* and the *summed* cost; these rows resolve that summary into each solve's own cycle count
        and the ``‖G‖`` trajectory, which is what distinguishes a step that took five easy solves from one
        that took five increasingly hard ones.

        Rows open a table on the step's first inner iteration and the step line closes it, so the block
        reads as one record. Two things follow from that ordering. The rows are written **as the step
        runs** while the step line is written once it returns, so the summary is a footer rather than a
        header: the outcome it reports is not known any earlier, and buffering the block to lead with it
        would cost exactly the live progress these rows exist to give. And a step that is redone (a
        ``β`` escalation, a divergence retry) opens a fresh table per attempt while the step line reports
        only the accepted one -- so the extra blocks are the record of what the retries cost.

        This is a diagnostic for a march being studied, not always-on instrumentation: it writes several
        lines per step, and the hook it attaches to must not be set on a differentiated solve.
        """
        before, after = float(g_before), float(g_after)
        if int(index) == 0:
            # A step redone by a retry restarts its inner loop from 0, so this both opens the first
            # block and closes off the abandoned attempt's block ahead of the retry's.
            self._open_block()
        elif not self._open:
            self._open_block()  # observer attached mid-step: still give the rows a grid to sit in
        self._write(
            self._INNER_TABLE.row(
                (
                    int(index),
                    restart_cycles(cycles),
                    before,
                    after,
                    after / before if before > 0.0 else float("nan"),
                    float(alpha),
                )
            )
        )

    def _open_block(self) -> None:
        """Start an inner-iteration table for the step about to be reported.

        The title repeats the step number and elapsed time that the closing summary also carries, and
        deliberately: the two timestamps bracket the step, so their difference is how long it took, and
        a summary line that stands alone is what makes the log greppable when the table is not enabled.
        """
        self._close_block()
        entering = "" if self._previous_norm is None else f"  from |R|={self._previous_norm:.4e}"
        title = f"step {self._steps + 1}  t={self._clock() - self._start:.0f}s{entering}"
        self._write(self._INNER_TABLE.rule(title))
        self._write(self._INNER_TABLE.headings())
        self._write(self._INNER_TABLE.rule())
        self._open = True

    def _close_block(self) -> None:
        """Close an open inner-iteration table, if any. A no-op when the observer is not wired."""
        if self._open:
            self._write(self._INNER_TABLE.rule())
            self._open = False

    def _log(self, report: StepReport, state: Any | None) -> None:
        self._close_block()
        self._steps += 1
        # The offset-corrected count, not the raw one: a two-inner step reporting `cycles = 6` did two
        # single-cycle solves, so the raw number is entirely offset and overstates the work threefold.
        cycles = report.restart_cycles
        inner = max(int(report.inner_iterations), 1)
        self._cumulative_cycles += cycles
        fields = [
            f"t={self._clock() - self._start:6.0f}s",
            f"step={self._steps:4d}",
            f"beta={report.shift:7.4f}",
            f"in={inner:2d}",
            f"cyc={cycles:3d}",
            f"c/in={cycles / inner:5.1f}",
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
        # Where the NEXT step starts from, so its table can head with the residual it inherits.
        self._previous_norm = float(report.residual_norm)

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
