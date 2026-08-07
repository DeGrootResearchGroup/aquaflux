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
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import IO, Any

import numpy as np

from ..text_table import Column, TextTable
from .linear import restart_cycles
from .march import StepReport

__all__ = ["MarchLogger", "combine_metrics", "field_change_metrics"]


def combine_metrics(
    *metrics: Callable[[Any], Mapping[str, float]],
) -> Callable[[Any], Mapping[str, float]]:
    """Merge several ``state -> mapping`` metrics callables into one.

    :class:`MarchLogger` takes a single ``metrics`` callable, but a run usually wants more than one
    kind of column -- a case quantity and a solver diagnostic, say. Composing them here keeps a driver
    from writing its own merge, which is where the two would drift apart.

    Later callables win on a name collision, matching ``dict`` update order.

    Parameters
    ----------
    *metrics : callable
        Each ``state -> mapping of name to float``.

    Returns
    -------
    callable
        One ``state -> mapping`` evaluating each in turn and merging the results.

    Examples
    --------
    >>> combined = combine_metrics(lambda s: {"a": 1.0}, lambda s: {"b": 2.0})
    >>> sorted(combined(None).items())
    [('a', 1.0), ('b', 2.0)]
    """

    def combined(state: Any) -> Mapping[str, float]:
        merged: dict[str, float] = {}
        for metric in metrics:
            merged.update(metric(state))
        return merged

    return combined


def field_change_metrics(
    fields: Callable[[Any], Mapping[str, Any]],
) -> Callable[[Any], Mapping[str, float]]:
    """Report each named field's **relative change since the previous call**, as ``d<name>/<name>``.

    A residual norm says how far the equations are from being satisfied; it does not say whether the
    *solution* has stopped changing, nor which field is still moving. This does: for each field it
    reports ``‖φ - φ_prev‖ / ‖φ_prev‖`` in the 2-norm over all of the field's entries.

    Reading it: a residual that keeps falling while every field's relative change sits at ~1e-8 means
    the iterates have converged and the remaining residual reduction is buying nothing; a small
    *scalar* case metric holding still while these columns are at 1e-2 means the opposite -- the
    scalar is too coarse (often mesh-quantized) to see movement that is really there.

    **Stateful by construction** -- it holds the previous fields, so it must be called once per step,
    in order (i.e. from :meth:`MarchLogger.on_checkpoint`, not from an arbitrary probe). The first
    call has nothing to compare against and reports ``nan``, as does any field whose previous value is
    identically zero. Across a continuation rung boundary the first comparison spans the parameter
    jump, which is a real (large) change, not an artifact.

    Parameters
    ----------
    fields : callable
        ``state -> mapping of name to array`` (e.g.
        :func:`~aquaflux.turbulence.coupled_fields`). Any array shape; the norm is over all entries.

    Returns
    -------
    callable
        ``state -> {"d<name>/<name>": relative change}``, for :class:`MarchLogger`'s ``metrics``.

    Examples
    --------
    >>> import numpy as np
    >>> metric = field_change_metrics(lambda s: {"u": np.asarray(s)})
    >>> metric([1.0, 0.0])["du/u"]  # first call has no previous iterate
    nan
    >>> round(metric([1.1, 0.0])["du/u"], 3)  # |0.1| / |1.0|
    0.1
    """
    previous: dict[str, Any] = {}

    def metrics(state: Any) -> Mapping[str, float]:
        current = dict(fields(state))
        out: dict[str, float] = {}
        for name, value in current.items():
            before = previous.get(name)
            if before is None:
                out[f"d{name}/{name}"] = float("nan")
                continue
            before, value = np.asarray(before), np.asarray(value)
            scale = float(np.linalg.norm(before.ravel()))
            change = float(np.linalg.norm((value - before).ravel()))
            out[f"d{name}/{name}"] = change / scale if scale > 0.0 else float("nan")
        previous.update(current)
        return out

    return metrics


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
    fields : callable, optional
        ``state -> mapping of name to array`` (e.g. :func:`~aquaflux.turbulence.coupled_fields`),
        reported as each field's relative change per step. Requires ``"fields"`` in ``detail``.
    detail : collection of str, optional
        Which **diagnostic** outputs to emit. Empty (the default) gives the plain one-line-per-step
        log; the rest are opt-in because they are debugging and profiling instruments, not something a
        routine run should pay for in log volume:

        - ``"inner"`` -- the per-inner-iteration table from :meth:`on_inner`.
        - ``"fields"`` -- per-field relative-change columns, from ``fields``.
        - ``"pc"`` -- what the preconditioner did each step, from :meth:`on_refresh`.

        Every hook stays safe to wire regardless, and each no-ops when its name is absent, so a driver
        connects the instrumentation once and switches verbosity with this one argument rather than by
        rewiring. An unknown name raises: a silently-ignored typo would mean losing the diagnostic you
        believed you had enabled, which is worse than not asking for it.
    rtol, atol : float, optional
        The stopping tolerances the march is running under, used to report the target
        ``atol + rtol·‖R₀‖`` next to the residual. Omit both to suppress the target column.
    clock : callable, optional
        ``() -> float`` seconds source for the elapsed-time column; injected so a test can supply a
        deterministic clock. Defaults to :func:`time.monotonic`.

    Raises
    ------
    ValueError
        If ``detail`` names an unknown diagnostic.

    Examples
    --------
    >>> log = MarchLogger(metrics=lambda s: {"xr/h": reattachment(s)}, rtol=1e-3)  # doctest: +SKIP
    >>> solve_coupled(coupled, on_checkpoint=log.on_checkpoint)  # doctest: +SKIP
    """

    #: The diagnostics ``detail`` may name.
    DETAIL = frozenset({"inner", "fields", "pc"})

    #: Re-emit the step table's headings after this many rows, so a long run stays readable in a tail.
    #: Tighter when the inner table is on, since its blocks push the last headings further up-screen.
    HEADINGS_EVERY = 25
    HEADINGS_EVERY_NESTED = 8

    #: Indent for the inner-iteration block, marking it as detail belonging to the step row below it.
    _NEST = "    "

    #: The inner-iteration table. ``G`` is the implicit timestep's own residual (the one the inner
    #: Newton loop drives to zero), and ``rate`` its per-iteration contraction ``‖G‖out / ‖G‖in`` --
    #: the number that says whether the inner loop is converging or merely grinding.
    _INNER_TABLE = TextTable(
        [
            Column("inner", 5, "d"),
            Column("cyc", 4, "d"),
            Column("G in", 10, ".3e"),
            Column("G out", 10, ".3e"),
            Column("rate", 6, ".3f"),
            Column("alpha", 5, ".3f"),
        ]
    )

    def __init__(
        self,
        stream: IO[str] | None = None,
        *,
        metrics: Callable[[Any], Mapping[str, float]] | None = None,
        fields: Callable[[Any], Mapping[str, Any]] | None = None,
        detail: Collection[str] = (),
        rtol: float | None = None,
        atol: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        unknown = sorted(set(detail) - self.DETAIL)
        if unknown:
            raise ValueError(
                f"unknown march-log detail {unknown}; known: {sorted(self.DETAIL)}. "
                "A silently-ignored name would lose a diagnostic you believed was on."
            )
        self._detail = frozenset(detail)
        self._stream = sys.stdout if stream is None else stream
        self._metrics = metrics
        # Built here rather than by the caller so `detail` alone decides whether the (stateful) change
        # measure runs at all -- it must be called once per step in order, or not at all.
        self._fields = (
            field_change_metrics(fields)
            if fields is not None and "fields" in self._detail
            else None
        )
        self._refresh: tuple[str, float] | None = None
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
        self._step_table: TextTable | None = None
        self._extra: tuple[str, ...] = ()
        self._needs_headings = True
        self._rows = 0
        self._reference: float | None = None
        self._headings_every = (
            self.HEADINGS_EVERY_NESTED if "inner" in self._detail else self.HEADINGS_EVERY
        )

    def note(self, text: str) -> None:
        """Write an arbitrary line (a run header, a configuration echo) to the log's own stream.

        Exists so a driver never reaches for ``print`` alongside the logger: that splits the run across
        two destinations, and a log written to a file then silently loses whichever lines went to
        stdout -- including the configuration echo needed to interpret the run.
        """
        self._close_block()
        self._write(text)
        self._needs_headings = True

    def phase(self, label: str, total: int | None = None) -> None:
        """Start a labelled phase (a continuation rung, a refreshed segment) and write a header.

        Phases are **numbered automatically** in call order, so a caller does not keep its own counter:
        ``phase("rung", total=3)`` yields ``rung 1/3``, ``rung 2/3``, ... The step counter and cumulative
        cycles keep running across phases, so the log still reads as one march.
        """
        self._close_block()
        self._phases += 1
        self._phase = f"{label} {self._phases}" + (f"/{total}" if total is not None else "")
        self._write("")
        self._write(f"=== {self._phase}   t={self._clock() - self._start:.0f}s ".ljust(72, "="))
        self._needs_headings = True

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
        so a driver wires it as ``inner_observer=logger.on_inner``. The step row reports only the inner
        *count* and the *summed* cost; these rows resolve that into each solve's own cycle count and the
        ``‖G‖`` trajectory, which is what distinguishes a step that took five easy solves from one that
        took five increasingly hard ones.

        **``alpha`` here is not the step row's ``a_min``.** This is *this* iteration's backtracking
        factor, and it reads ``1.000`` even when the line search failed to descend at all -- the
        non-descent fallback returns the longest finite trial step. The step row instead reports the
        minimum over the step's iterations with any non-descending iteration folded in as ``0``. So a
        step row of ``a_min=0.000`` beside inner rows of ``alpha=1.000`` is not a contradiction: it
        means an iteration took its full step and still did not reduce ``‖G‖``. **``rate`` identifies
        those exactly** -- ``rate >= 1`` is the non-descent case, since the search is monotone.

        Rows open a block on the step's first inner iteration and the step row closes it. The block is
        written **as the step runs** while the step row is written once it returns, so the summary
        follows rather than heads it: the outcome is not known any earlier, and buffering the block to
        lead with it would cost exactly the live progress these rows exist to give. A step that is
        redone (a ``β`` escalation, a divergence retry) opens a fresh block per attempt while the step
        row reports only the accepted one -- so the extra blocks are the record of what the retries cost.

        No-ops unless ``"inner"`` is in ``detail``: it writes several lines per step, and the hook it
        attaches to must not be set on a differentiated solve.
        """
        if "inner" not in self._detail:
            return
        before, after = float(g_before), float(g_after)
        if int(index) == 0 or not self._open:
            # A step redone by a retry restarts its inner loop from 0, so this both opens the first
            # block and closes off the abandoned attempt's block ahead of the retry's.
            self._open_block()
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
            ),
            indent=self._NEST,
        )

    def on_refresh(self, kind: str, seconds: float) -> None:
        """``observer`` callback for a β-tracking preconditioner refresh: record what it did.

        Matches the hook :func:`~aquaflux.turbulence.amg_beta_tracking_refresh` calls once per step
        (``observer=logger.on_refresh``). ``kind`` is ``"full"``, ``"shift"`` or ``"none"``; the value
        rides on the *next* step row, since that is the step it was built for.

        Without this the log records only how long a step took, and which branch ran has to be guessed
        from wall-clock -- which is how a cost that is really an occasional expensive re-materialize
        gets mistaken for a fixed per-step overhead. No-ops unless ``"pc"`` is in ``detail``.
        """
        if "pc" in self._detail:
            self._refresh = (str(kind), float(seconds))

    def _open_block(self) -> None:
        """Start an inner-iteration table for the step about to be reported."""
        self._close_block()
        entering = "" if self._previous_norm is None else f"  from |R|={self._previous_norm:.3e}"
        title = f"step {self._steps + 1}{entering}"
        self._write("")
        self._write(self._INNER_TABLE.rule(title), indent=self._NEST)
        self._write(self._INNER_TABLE.headings(), indent=self._NEST)
        self._write(self._INNER_TABLE.rule(), indent=self._NEST)
        self._open = True

    def _close_block(self) -> None:
        """Close an open inner-iteration table.

        This does **not** force the step headings to be re-emitted. The step grid is fixed-width, so its
        rows still line up across an interruption, and re-heading around every block would cost three
        lines per step for no added legibility.
        """
        if self._open:
            self._write(self._INNER_TABLE.rule(), indent=self._NEST)
            self._open = False

    def _step_columns(self, extra: Sequence[str]) -> list[Column]:
        """The step table's columns: the fixed solver ones, then whatever the metrics named."""
        columns = [
            Column("step", 4, "d"),
            Column("t(s)", 6, ".0f"),
            Column("beta", 7, ".4f"),
            Column("in", 2, "d"),
            Column("cyc", 4, "d"),
            Column("cum", 6, "d"),
            Column("R", 9, ".3e"),
            Column("a_min", 5, ".3f"),
        ]
        if "pc" in self._detail:
            columns.append(Column("pc", 11, "", "<"))
        columns += [Column(name, max(len(name), 8), "", ">") for name in extra]
        columns.append(Column("flag", 4, "", "<"))
        return columns

    def _headings(self, extra: Sequence[str]) -> None:
        """(Re)build the step table when its columns change, and emit its heading rows."""
        if self._step_table is None or tuple(extra) != self._extra:
            self._extra = tuple(extra)
            self._step_table = TextTable(self._step_columns(extra))
        self._write(self._step_table.rule())
        self._write(self._step_table.headings())
        self._write(self._step_table.rule())
        self._rows = 0
        self._needs_headings = False

    def _log(self, report: StepReport, state: Any | None) -> None:
        self._close_block()
        self._steps += 1
        # The offset-corrected count, not the raw one: a two-inner step reporting `cycles = 6` did two
        # single-cycle solves, so the raw number is entirely offset and overstates the work threefold.
        cycles = report.restart_cycles
        inner = max(int(report.inner_iterations), 1)
        self._cumulative_cycles += cycles

        columns: dict[str, float] = {}
        if state is not None and self._metrics is not None:
            columns.update(self._metrics(state))
        deltas: dict[str, float] = {}
        if state is not None and self._fields is not None:
            deltas.update(self._fields(state))

        # The stopping test is a constant within a rung, so it belongs in a banner rather than in every
        # row -- repeating `|R0|` and `target` on each line was most of the old line's width.
        reference = self._reference_norm(report)
        if reference is not None and (
            self._reference is None or abs(reference - self._reference) > 1e-12 * self._reference
        ):
            self._reference = reference
            target = self._target(reference)
            stop = "" if target is None else f", stopping at |R| <= {target:.3e}"
            self._close_block()
            self._write("")
            self._write(f"reference |R0| = {reference:.4e}{stop}")
            self._needs_headings = True

        if self._needs_headings or self._rows >= self._headings_every:
            self._headings(list(columns))
        assert self._step_table is not None

        values: list[Any] = [
            self._steps,
            self._clock() - self._start,
            report.shift,
            inner,
            cycles,
            self._cumulative_cycles,
            report.residual_norm,
            report.alpha,
        ]
        if "pc" in self._detail:
            kind, seconds = self._refresh or ("-", 0.0)
            values.append("-" if kind == "-" else f"{kind} {seconds:.1f}s")
            self._refresh = None
        values += [f"{columns[name]:.4g}" for name in self._extra]
        # `cycles` counts only the ACCEPTED attempt, so a redone step is otherwise indistinguishable
        # from a cheap one -- and a retry mechanism left unconfigured never announces its absence.
        marks = ""
        if report.escalations:
            marks += f"e{int(report.escalations)}"
        if report.diverged_retry:
            marks += "R"
        values.append(marks)

        self._write(self._step_table.row(values))
        self._rows += 1
        if deltas:
            self._write(
                self._step_table.spanning(
                    "  " + "  ".join(f"{name}={value:.2e}" for name, value in deltas.items())
                )
            )
        # Where the NEXT step starts from, so its inner block can head with the residual it inherits.
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

    def _write(self, line: str, indent: str = "") -> None:
        self._stream.write(indent + line + "\n")
        self._stream.flush()
