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

Every line is flushed as it is written, so a multi-hour march can be tailed while it runs rather than
read after it ends. A step reports as one framed block: the columned step row, then -- when the
per-field diagnostics are on -- a per-equation grid of relative change, residual and contraction, then
the readings that belong to no column, one per line.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Collection, Mapping
from typing import IO, Any

import numpy as np

from ..text_table import Column, TextTable
from .forward_step import StepReport
from .linear import restart_cycles
from .refresh_timing import RefreshTiming

#: Below this many seconds an unattributed remainder is measurement noise, not a missing phase, and
#: reporting it would only add a column of zeros to every refresh line.
_UNATTRIBUTED_FLOOR = 0.05

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
    """Report each named field's **relative change since the previous call**, under its own name.

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
        ``state -> {name: relative change}``, keyed by the **field's own name** so a caller can join
        it against another per-field measure (a per-equation residual, say) without unpicking a
        decorated key. How the quantity is labelled is the report's business, not the measure's.

    Examples
    --------
    >>> import numpy as np
    >>> metric = field_change_metrics(lambda s: {"u": np.asarray(s)})
    >>> metric([1.0, 0.0])["u"]  # first call has no previous iterate
    nan
    >>> round(metric([1.1, 0.0])["u"], 3)  # |0.1| / |1.0|
    0.1
    """
    previous: dict[str, Any] = {}

    def metrics(state: Any) -> Mapping[str, float]:
        current = dict(fields(state))
        out: dict[str, float] = {}
        for name, value in current.items():
            before = previous.get(name)
            if before is None:
                out[name] = float("nan")
                continue
            before, value = np.asarray(before), np.asarray(value)
            scale = float(np.linalg.norm(before.ravel()))
            change = float(np.linalg.norm((value - before).ravel()))
            out[name] = change / scale if scale > 0.0 else float("nan")
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
    residuals : callable, optional
        ``state -> mapping of equation name to float`` (e.g.
        :func:`~aquaflux.turbulence.coupled_residuals`), reported as the per-equation residual and its
        step-on-step contraction. Requires ``"residuals"`` in ``detail``.

        This is what turns a stalling march from a mystery into a diagnosis: the scalar residual says
        the solve has stopped improving, and this says *which equation* stopped it. Names are joined
        against ``fields`` on the same key, so a field with no equation (a derived ``nu_t``) and an
        equation with no field both still get a row.
    detail : collection of str, optional
        Which **diagnostic** outputs to emit. Empty (the default) gives the plain one-line-per-step
        log; the rest are opt-in because they are debugging and profiling instruments, not something a
        routine run should pay for in log volume:

        - ``"inner"`` -- the per-inner-iteration table from :meth:`on_inner`.
        - ``"fields"`` -- per-field relative-change columns, from ``fields``.
        - ``"residuals"`` -- per-equation residual columns, from ``residuals``. Costs one extra
          residual evaluation per logged step, which is small beside the step's linear solves but is
          not free.
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
    DETAIL = frozenset({"inner", "fields", "residuals", "pc"})

    #: The per-equation block under each step row. Its width matches the step table's, so the two
    #: stack as one framed block -- pinned by a test, since a mismatch renders as a broken frame.
    _FIELD_TABLE = TextTable(
        [
            Column("field", 11, "", "<"),
            Column("rel. change", 13),
            Column("resid", 13),
            Column("rate", 13),
        ]
    )

    #: Printed where a per-field cell has no value: a field with no equation (``nu_t``) has no
    #: residual, and the first step of a run has nothing to form a change or a rate against. An
    #: explicit mark rather than a blank, so "not applicable" cannot be misread as "zero".
    _ABSENT = "--"

    #: Re-emit the step table's headings after this many rows, so a long run stays readable in a tail.
    #: Tighter when the inner table is on, since its blocks push the last headings further up-screen.
    HEADINGS_EVERY = 25

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
        residuals: Callable[[Any], Mapping[str, float]] | None = None,
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
        self._residuals = residuals if "residuals" in self._detail else None
        # The previous step's per-equation residuals, for the step-on-step contraction rate. Held
        # here rather than by the caller for the same reason `field_change_metrics` is built here:
        # the rate is only meaningful over consecutive logged steps, so its state belongs to the
        # thing that is called once per step, in order.
        self._previous_residuals: dict[str, float] = {}
        self._refresh: RefreshTiming | None = None
        self._rtol = rtol
        self._atol = atol
        self._clock = time.monotonic if clock is None else clock
        self._start = self._clock()
        self._cumulative_cycles = 0
        self._steps = 0
        self._phases = 0
        self._phase = ""
        self._open = False
        self._attempt = 1
        self._legend_written = False
        self._step_table: TextTable | None = None
        self._nested = False
        self._needs_headings = True
        self._rows = 0
        self._reference: float | None = None

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
        """``on_step`` callback: log a step with no case metrics (no state is available here).

        ⚠️ **Wire this OR** :meth:`on_checkpoint`, **never both.** They are two renderings of the same
        event, not two events -- this one without the injected case metrics, that one with -- and a
        march calls its ``on_step`` and ``on_checkpoint`` seams **unconditionally on every step**. Give
        it the same logger for both and every step is logged twice, with this object's step counter and
        cumulative-cycle total double-counted along with it (measured: 2 steps produce 4 rows and a
        doubled cumulative). Prefer :meth:`on_checkpoint` wherever a state is available, since it is
        the strictly more informative of the two.
        """
        self._log(report, None)

    def on_checkpoint(self, report: StepReport, state: Any) -> None:
        """``on_checkpoint`` callback: log a step and append the injected case metrics.

        ⚠️ **Wire this OR** :meth:`on_step`, **never both** -- see that method for why. This is the one
        to prefer: it renders the same row plus the case metrics.
        """
        self._log(report, state)

    def on_inner(
        self,
        index: int,
        g_before: float,
        g_after: float,
        cycles: int,
        alpha: float,
        iterate: object = None,
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

        The ``iterate`` the hook also carries is the inner state itself, which a log has no use for;
        it is accepted and ignored so this stays a drop-in for the hook a probe uses to capture it.

        No-ops unless ``"inner"`` is in ``detail``: it writes several lines per step, and the hook it
        attaches to must not be set on a differentiated solve.
        """
        del iterate
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

    def on_refresh(self, timing: RefreshTiming) -> None:
        """``observer`` callback for a β-tracking preconditioner refresh: record what it did.

        Matches the hook :func:`~aquaflux.turbulence.amg_beta_tracking_refresh` calls once per step
        (``observer=logger.on_refresh``). The record names the branch — ``"full"``, ``"shift"`` or
        ``"none"`` — its total, and each part's own cost; it rides on the *next* step row, since that is
        the step it was built for.

        Without this the log records only how long a step took, and which branch ran has to be guessed
        from wall-clock -- which is how a cost that is really an occasional expensive re-materialize
        gets mistaken for a fixed per-step overhead. **The phase breakdown is the same argument one level
        down:** two refreshes of equal length, one spent re-probing the Jacobian and one spent rebuilding
        the multigrid, call for entirely different fixes and are indistinguishable in the total.
        No-ops unless ``"pc"`` is in ``detail``.
        """
        if "pc" in self._detail:
            self._refresh = timing

    def on_retry(self, reason: str, attempt: int, beta: float) -> None:
        """``on_retry`` callback: announce that the step about to be repeated is being redone, and why.

        Matches the hook :func:`~aquaflux.solve.forward_march` calls just before a redo
        (``on_retry=logger.on_retry``). Without it a log shows the same step's work two or three times
        with nothing between the blocks, leaving a reader to infer the trigger from the numbers -- and
        the four triggers (a step cut short on cost, a collapsed step length, a diverged step, a
        too-loose linear solve) call for completely different responses.

        Written whatever ``detail`` says: a repeated step is not a diagnostic, it is the log failing to
        explain itself. The attempt number also titles the block that follows.

        ``beta`` is reported as the march hands it over -- the shift the retried attempt will run at --
        and never recomputed here. The two retries differ in whether that shift moved at all: the three
        escalation reasons raise it (``beta -> x``), while ``"solver"`` retries at a tighter Krylov
        tolerance and leaves it alone (``beta x unchanged``). Saying "->" on the solver line would claim
        an escalation the march did not perform.
        """
        self._close_block()  # the abandoned attempt's block ends here, before the explanation
        self._attempt = int(attempt) + 1
        shift = f"beta {beta:.4f} unchanged" if reason == "solver" else f"beta -> {beta:.4f}"
        self._write(
            f"  -> redo step {self._steps + 1} (attempt {self._attempt}): {reason}, {shift}"
        )

    #: Written once, before the first inner block. `G` and `R` are different residuals and the table
    #: cannot say so in a column heading; without this a reader reasonably expects the step's `R` to be
    #: the last `G out`, and it never is.
    _LEGEND = (
        "  G is the implicit timestep's own residual, R + beta*d*(phi - phi_n); the inner loop drives",
        "  it to zero. R is the STEADY residual at the stepped state -- the march's convergence measure.",
        "  Driving G to zero does not drive R to zero: the two terms cancel there,",
        "  leaving R = -beta*d*(phi - phi_n). So a step's R is NOT its last 'G out'; it measures the",
        "  state the next step starts from, which that step's inner-0 'G in' measures in the NEXT",
        "  iteration's row scales -- the same state, not the same number.",
    )

    def _open_block(self) -> None:
        """Start an inner-iteration table for the step about to be reported.

        The title carries the step number only. It deliberately does **not** repeat the previous step's
        ``R``, even though that is the residual this step starts from: under a self-rescaling measure
        the two are not the same number. The march re-derives the row scales at the state each outer
        iteration begins from, so the previous step's ``R`` measures that state in the *previous*
        iteration's scales while this block's ``inner 0`` ``G in`` measures it in the current ones.
        Measured over one march the two differed on **every** step -- by up to 2x early on, converging
        to 1 as the state settled and the scales stopped moving. Printing both side by side invited a
        comparison that never holds; ``G in`` at ``inner 0`` is the entering residual, correctly scaled.
        """
        self._close_block()
        if not self._legend_written:
            self._legend_written = True
            self._write("")
            for line in self._LEGEND:
                self._write(line)
        title = f"step {self._steps + 1}" + (
            f"  attempt {self._attempt}" if self._attempt > 1 else ""
        )
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
            self._nested = True

    def _step_columns(self) -> list[Column]:
        """The step table's columns.

        Deliberately **narrow and fixed**: only the quantities worth scanning *down* the run live in
        columns, and the set does not vary with what the metrics happen to be called. Everything else
        (the case metrics, the preconditioner branch, the field changes) rides in spanning rows below
        the row it belongs to. A table wide enough to hold all of it stops reading as a table -- it
        becomes a line of numbers whose heading is too far away to pair them with.
        """
        return [
            Column("step", 4, "d"),
            Column("t(s)", 6, ".0f"),
            Column("beta", 6, ".4f"),
            Column("in", 2, "d"),
            Column("cyc", 3, "d"),
            Column("R", 9, ".3e"),
            Column("a_min", 5, ".3f"),
            Column("flg", 3, "", "<"),
        ]

    def _headings(self) -> None:
        """Open the step table: a blank line, then a fully ruled heading.

        Always fully ruled, and always preceded by a blank line. An unruled heading butted straight
        against the inner block above it reads as debris hanging off that block rather than as the top
        of a new table -- the grid needs a visible top edge and some air to be legible as a unit.
        """
        if self._step_table is None:
            self._step_table = TextTable(self._step_columns())
        self._write("")
        # Titled, like the inner block's own opening rule: with two grids interleaved down the log, an
        # untitled one leaves the reader working out which table they have landed in. The wording is
        # deliberately NOT "step ...", which the inner block already uses -- a shared prefix makes the
        # two indistinguishable to a grep even though a human can tell them apart.
        # Heavy fill on the outer boundary: a step's report is several stacked grids, so the rule that
        # opens the whole block has to be distinguishable from the light rules dividing them.
        self._write(self._step_table.rule("summary stats", fill="="))
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
        residuals: dict[str, float] = {}
        if state is not None and self._residuals is not None:
            residuals.update(self._residuals(state))

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

        # An inner block just closed, so this row would otherwise be unlabelled: re-head compactly.
        if self._needs_headings or self._nested or self._rows >= self.HEADINGS_EVERY:
            self._headings()
        self._nested = False
        assert self._step_table is not None

        marks = ""
        # `L`: the step length was set by an injected limit (positivity), not by the descent test --
        # a different diagnosis from a clipped alpha, and one a reader would otherwise have to guess.
        if report.binding_limit < 1.0:
            marks += "L"
        if report.escalations:
            marks += f"e{int(report.escalations)}"
        if report.diverged_retry:
            marks += "R"
        self._write(
            self._step_table.row(
                [
                    self._steps,
                    self._clock() - self._start,
                    report.shift,
                    inner,
                    cycles,
                    report.residual_norm,
                    report.alpha,
                    marks,
                ]
            )
        )
        self._rows += 1
        self._write_field_table(deltas, residuals)
        self._write_aside(report, columns)
        # This step is done, so the next block is a fresh step, not another attempt at this one.
        self._attempt = 1

    @classmethod
    def _cell(cls, value: float | None) -> str:
        """One per-field cell: the value in exponential form, or :attr:`_ABSENT` when there is none.

        A non-finite *value* is rendered as-is rather than as absent -- on a residual it means the step
        diverged, which is the single most important thing the row can say.
        """
        return cls._ABSENT if value is None else f"{value:.3e}"

    def _write_field_table(
        self, changes: Mapping[str, float], residuals: Mapping[str, float]
    ) -> None:
        """The per-equation grid beneath the step row: relative change, residual, and its contraction rate.

        The three columns answer three different questions, and it is reading them *across* that
        diagnoses a march: a large residual with a tiny relative change means the step is no longer
        moving that equation, while a rate at or above one means the equation is not converging at
        all -- neither of which the scalar residual above can distinguish.

        Rows are the union of the two mappings, ``changes`` first, so a field with no equation (a
        derived ``nu_t``) and an equation with no field both still appear.
        """
        assert self._step_table is not None
        names = list(changes) + [name for name in residuals if name not in changes]
        if not names:
            return
        self._write(self._step_table.rule(fill="=", segmented=False))
        self._write(self._FIELD_TABLE.headings())
        self._write(self._FIELD_TABLE.rule())
        for name in names:
            change = changes.get(name)
            # `field_change_metrics` reports a NaN for "nothing to compare against" (the first step,
            # or a field that was identically zero), which is an absent cell rather than a value.
            if change is not None and not np.isfinite(change):
                change = None
            resid = residuals.get(name)
            before = self._previous_residuals.get(name)
            rate = None if resid is None or not before else resid / before
            self._write(
                self._FIELD_TABLE.row(
                    [name, self._cell(change), self._cell(resid), self._cell(rate)]
                )
            )
        self._previous_residuals.update(residuals)

    def _write_aside(self, report: StepReport, columns: Mapping[str, float]) -> None:
        """The readings that are not per-column, **one concern per line**, closing the step's block.

        Run together on one line these were unreadable: the preconditioner's activity, the case
        metrics and the solver's own counters have nothing to do with each other, so a reader looking
        for one of them had to parse all three. One line each costs vertical space and buys the
        ability to find a quantity by position.
        """
        assert self._step_table is not None
        self._write(self._step_table.rule(fill="=", segmented=False))
        lines = []
        if "pc" in self._detail:
            lines.append(self._refresh_line(self._refresh))
            self._refresh = None
        if columns:
            lines.append("  ".join(f"{name} {value:.4g}" for name, value in columns.items()))
        if report.binding_limit < 1.0:
            lines.append(f"limit {report.binding_limit:.2e}")
        lines.append(f"cum {self._cumulative_cycles}")
        for line in lines:
            self._write(self._step_table.spanning(line))
        self._write(self._step_table.rule(fill="=", segmented=False))

    @staticmethod
    def _refresh_line(timing: RefreshTiming | None) -> str:
        """The preconditioner aside: which branch ran, its total, and where the total went.

        A refresh that reports phases renders them in the order they ran, so the expensive part is read
        off directly rather than inferred by differencing two runs. Any wall time the phases do not
        account for is shown as ``other``, since a breakdown that silently fails to add up is worse than
        no breakdown -- it reads as complete.
        """
        if timing is None:
            return "pc -"
        line = f"pc {timing.kind} {timing.seconds:.1f}s"
        if not timing.phases:
            return line
        parts = [f"{name} {seconds:.1f}" for name, seconds in timing.phases]
        if timing.unattributed >= _UNATTRIBUTED_FLOOR:
            parts.append(f"other {timing.unattributed:.1f}")
        return f"{line} ({' '.join(parts)})"

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
