"""The Run section: solve the open case, and follow its march step by step as it goes.

Everything a run does is set in its case file, so this section has no settings of its own: a button
to start the solve and one to stop it, a summary of what the case file says the run will do, and the
march drawn as it goes -- its residual against the stopping target, what each step cost, how the
step control moved, and what happened along the way. The solve is an ``aquaflux run`` process
(:class:`~aquaflux_ui.run_process.CaseRun`); the plots are read from the per-step history it writes
(:mod:`aquaflux_ui.run_monitor`), re-read every second while it runs.

A run replaces an earlier one's results only when told to: if the case's output directory holds
results, the page asks before Run will start.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from trame.widgets import html, plotly
from trame.widgets import vuetify3 as v3

from .case_form import KIND, setting
from .history import ConvergenceHistory
from .run_monitor import (
    RunLimits,
    cost_figures,
    equation_columns,
    residual_figure,
    run_events,
    run_tiles,
)
from .run_process import CONVERGED, NOT_CONVERGED, REFUSED, CaseRun
from .solver_commands import RunPlan
from .widgets import empty_state

__all__ = ["BY_EQUATION", "OVERALL", "POLL_SECONDS", "RunSection"]

#: Seconds between reads of a running march's history.
POLL_SECONDS = 1.0

#: The residual plot's two views, as the page's toggle holds them. Strings, not booleans: a button
#: given ``value=True`` renders it as an HTML boolean attribute, and the page then holds ``""``.
OVERALL, BY_EQUATION = "overall", "equations"

#: The four plots beneath the residual: their key, title and what their dashed line is.
_SMALL_PLOTS = (
    ("cycles", "Linear solve cost per step", "redo above"),
    ("inner", "Inner Newton iterations", "limit"),
    ("shift", "Pseudo-time shift β", "β min"),
    ("alpha", "Line-search step α", "retry below"),  # noqa: RUF001  (the step's own symbol)
)

#: The status chip's colour by run state.
_STATE_COLOURS = {
    "idle": "secondary",
    "running": "primary",
    "converged": "success",
    "short": "warning",
    "stopped": "secondary",
    "failed": "error",
    "earlier": "secondary",
}


class RunSection:
    """The Run section.

    Parameters
    ----------
    server : trame server
        The workspace's server.
    setup : SetupSection
        The Setup section, whose open case this section runs: its document, schema and solver
        commands are read when they are needed, so an edit there is seen here.
    """

    key, title, icon = "run", "Run", "mdi-play-circle-outline"

    def __init__(self, server, setup) -> None:
        self.server = server
        self.setup = setup
        self.run: CaseRun | None = None
        self.plan: RunPlan | None = None
        self.limits = RunLimits()
        self._started_at = 0.0
        self._steps_drawn = -1
        self._console = Path(tempfile.mkdtemp(prefix="aquaflux-run-")) / "console.txt"
        self._tasks: set[asyncio.Task] = set()
        self._figures: dict[str, object] = {}
        server.state.update(
            {
                "run_state": "idle",
                "run_state_text": "",
                "run_ready": False,
                "run_block": "",
                "run_running": False,
                "run_occupied": [],
                "run_overwrite": False,
                "run_directory": "",
                "run_directory_shown": "",
                "run_summary": [],
                "run_status": [],
                "run_tiles": [],
                "run_events": [],
                "run_log": "",
                "run_feed": "events",
                "run_residual_view": OVERALL,
                "run_has_equations": False,
                "run_has_steps": False,
            }
        )
        server.state.change("case_path", "case_modified", "case_loaded")(self._on_case)
        server.state.change("run_overwrite")(lambda **_: self._update_ready())
        server.state.change("run_residual_view", "theme")(lambda **_: self._redraw(force=True))
        # A solve left running when the page ends would run on, unwatched, holding the machine.
        atexit.register(self._stop_quietly)

    # --- what will run ----------------------------------------------------------------------------

    def _on_case(self, **_) -> None:
        self._spawn(self.replan())

    async def replan(self) -> None:
        """Ask where a run of the open case would write, and what it would replace."""
        state = self.server.state
        path = state.case_path if state.case_loaded else ""
        plan = await asyncio.to_thread(self.setup.commands.plan, path) if path else None
        self.plan = plan
        with self.server.state as state:
            state.run_occupied = [] if plan is None else [str(p) for p in plan.occupied]
            directory = None if plan is None else plan.directory
            state.run_directory = "" if directory is None else str(directory)
            state.run_directory_shown = (
                "" if directory is None else _relative(directory, self.setup.path)
            )
            if not state.run_occupied:
                state.run_overwrite = False
            state.run_summary = self._summary()
            self._update_ready()
        # A run started here keeps its own ending on show; only one found on disk is "the earlier run".
        if self.run is None:
            self._show_earlier_run()

    def _summary(self) -> list[dict]:
        """What the case file says the run will do, as label and value rows."""
        document, schema = self.setup.document, self.setup.schema
        if document is None or schema is None:
            return []

        def kind(*path: str) -> str | None:
            value = setting(schema, document, path)
            return value.get(KIND) if isinstance(value, dict) else None

        rows = [("Physics", kind("physics")), ("Solver", kind("solver"))]
        continuation = setting(schema, document, ("solver", "continuation"))
        if isinstance(continuation, dict):
            stations = continuation.get("stations")
            rows.append(
                ("Continuation", continuation.get(KIND, "")
                 + (f", {stations} stations" if stations else ""))
            )  # fmt: skip
        inner = setting(schema, document, ("solver", "dual_time", "inner_steps"))
        rows.append(("Inner loop", f"Dual time, ≤ {inner} iterations" if inner else "None"))
        limits = RunLimits.from_case(schema, document)
        if limits.max_steps is not None:
            rows.append(("Max steps", f"{limits.max_steps} per segment"))
        if limits.atol is not None or limits.rtol is not None:
            rows.append(("Stops below", _stop_text(limits)))
        if self.plan is not None and self.plan.directory is not None:
            rows.append(("Output", _relative(self.plan.directory, self.setup.path)))
        return [{"label": label, "value": value} for label, value in rows if value]

    def _update_ready(self) -> None:
        """Whether Run can start, and if not, why not."""
        state = self.server.state
        running = self.run is not None and self.run.running
        if running:
            block = ""
        elif not state.case_loaded:
            block = "Open a case in Setup to run it."
        elif state.case_modified:
            block = "Save the case first: a run reads the saved file."
        elif self.plan is not None and self.plan.error:
            block = self.plan.error
        elif state.run_occupied and not state.run_overwrite:
            block = "The output directory holds results. Switch on replacing them to run."
        else:
            block = ""
        state.run_block = block
        state.run_ready = not running and not block
        state.run_running = running

    # --- starting and stopping --------------------------------------------------------------------

    def start(self) -> None:
        """Start the open case's solve, if it can start."""
        state = self.server.state
        if not state.run_ready or self.setup.path is None or self.setup.schema is None:
            return
        self.limits = RunLimits.from_case(self.setup.schema, self.setup.document)
        self.run = CaseRun(self.setup.path, self._console, overwrite=bool(state.run_overwrite))
        self._started_at = time.time()
        self.run.start()
        self._steps_drawn = -1
        state.update(
            {
                "run_state": "running",
                "run_state_text": "Starting",
                "run_events": [],
                "run_tiles": [],
                "run_status": [],
                "run_log": "",
                "run_has_steps": False,
                # Permission to replace results is for this run; the next one is asked again.
                "run_overwrite": False,
            }
        )
        self._update_ready()
        self._redraw(force=True)
        self._spawn(self._follow())

    def stop(self) -> None:
        """Stop the solve under way; its log and run record are still written."""
        if self.run is not None and self.run.running:
            self.server.state.run_state_text = "Stopping: the solve ends after its current step"
            self._spawn(asyncio.to_thread(self.run.stop))

    def _stop_quietly(self) -> None:
        if self.run is not None:
            with contextlib.suppress(Exception):
                self.run.stop(grace=3.0)

    async def _follow(self) -> None:
        """Re-read the history while the solve runs, then report how it ended."""
        run = self.run
        while run.running:
            with self.server.state:
                self._refresh()
            await asyncio.sleep(POLL_SECONDS)
        with self.server.state as state:
            self._refresh()
            state.run_state, state.run_state_text = _ending(run)
            state.run_log = run.console_tail()
            self._update_ready()
        await self.replan()

    # --- reading the march ------------------------------------------------------------------------

    def _history(self, since: float | None) -> ConvergenceHistory | None:
        """The history file, read, if it exists and (``since`` given) was written since then."""
        path = None if self.plan is None else self.plan.history
        if path is None or not path.is_file():
            return None
        if since is not None and path.stat().st_mtime < since:
            return None  # an earlier run's, about to be replaced
        try:
            return ConvergenceHistory.read(path)
        except (OSError, ValueError):
            return None

    def _refresh(self) -> None:
        """Re-read the running march's history and redraw what it shows."""
        state = self.server.state
        history = self._history(since=self._started_at - 1.0)
        state.run_log = self.run.console_tail() if self.run is not None else ""
        if history is None:
            state.run_state_text = "Starting: building the case"
            return
        self._show(history)
        if history.n_steps:
            state.run_state_text = f"Running · step {history.n_steps}"

    def _show_earlier_run(self) -> None:
        """Draw the history an earlier run left in the output directory, if there is one."""
        history = self._history(since=None)
        with self.server.state as state:
            if history is None or history.n_steps == 0:
                state.run_state, state.run_state_text = "idle", ""
                state.run_tiles, state.run_events, state.run_status = [], [], []
                state.run_has_steps = False
                self._steps_drawn = -1
                self._redraw(force=True, history=history)
                return
            self.limits = RunLimits.from_case(self.setup.schema, self.setup.document)
            state.run_state, state.run_state_text = "earlier", "Showing the earlier run"
            self._steps_drawn = -1
            self._show(history)

    def _show(self, history: ConvergenceHistory) -> None:
        state = self.server.state
        state.run_tiles = run_tiles(history, self.limits)
        state.run_events = [event.to_state() for event in run_events(history)]
        state.run_status = _status_rows(history, self.limits)
        state.run_has_steps = history.n_steps > 0
        state.run_has_equations = bool(equation_columns(history))
        self._redraw(history=history)

    def _redraw(self, force: bool = False, history: ConvergenceHistory | None = None) -> None:
        """Redraw the plots from ``history`` (re-read if not given), when it has new steps or ``force``."""
        if not self._figures:
            return
        if history is None:
            history = self._history(since=None) or _NO_STEPS
        if not force and history.n_steps == self._steps_drawn:
            return
        self._steps_drawn = history.n_steps
        theme = self.server.state.theme
        by_equation = self.server.state.run_residual_view == BY_EQUATION
        self._figures["residual"](residual_figure(history, self.limits, theme, by_equation))
        for name, figure in cost_figures(history, self.limits, theme).items():
            self._figures[name](figure)

    def _spawn(self, coroutine) -> None:
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # --- the page ---------------------------------------------------------------------------------

    def drawer(self) -> None:
        """Run and Stop, how the run is going, and what the case file says it will do."""
        with html.Div(classes="pa-4 d-flex flex-column", style="gap: 14px;"):
            html.Div("Run", classes="text-overline text-medium-emphasis")
            with html.Div(classes="d-flex", style="gap: 8px;"):
                v3.VBtn(
                    "Run case", prepend_icon="mdi-play", color="primary", flat=True,
                    disabled=("!run_ready",), click=self.start, classes="flex-grow-1",
                )  # fmt: skip
                v3.VBtn(
                    "Stop", prepend_icon="mdi-stop", color="error", variant="tonal",
                    disabled=("!run_running",), click=self.stop, classes="flex-grow-1",
                )  # fmt: skip
            with html.Div(v_if="run_state_text"):
                v3.VChip(
                    "{{ run_state_text }}",
                    color=(f"{_STATE_COLOURS}[run_state]",),
                    size="small",
                    variant="tonal",
                    prepend_icon=("run_running ? 'mdi-circle-medium' : ''",),
                )
            html.Div(
                "{{ run_block }}", v_if="run_block", classes="text-body-2 text-medium-emphasis"
            )
            with v3.VCard(v_if="run_status.length", variant="outlined", classes="pa-3"):
                _key_values("run_status")
            with html.Div(
                v_if="run_summary.length", classes="d-flex flex-column", style="gap: 8px;"
            ):
                html.Div("From the case file", classes="text-overline text-medium-emphasis")
                _key_values("run_summary")
                v3.VBtn(
                    "Change these in Setup", append_icon="mdi-arrow-right", variant="text",
                    size="small", color="primary", click="section = 'setup'",
                    classes="align-self-start px-0",
                )  # fmt: skip
            with html.Div(
                v_if="run_occupied.length", classes="d-flex flex-column", style="gap: 4px;"
            ):
                v3.VSwitch(
                    v_model=("run_overwrite",), label="Replace the earlier results",
                    hide_details=True, disabled=("run_running",),
                )  # fmt: skip
                html.Div(
                    "{{ run_directory_shown }} holds results from an earlier run. With this off, Run "
                    "will not start rather than replace them.",
                    classes="text-caption text-medium-emphasis",
                )

    def main(self) -> None:
        """The march: headline numbers, the residual, what each step cost, and what happened."""
        with html.Div(v_if="!case_loaded", style="height: 100%;"):
            empty_state(
                self.icon,
                "No case open",
                "A run solves the case open in Setup, with everything its case file sets.",
                steps=("Open a case in Setup, then come back here and press Run case.",),
            )
        with html.Div(v_show="case_loaded", classes="pa-3 d-flex flex-column", style="gap: 12px;"):
            with html.Div(classes="af-tiles", v_if="run_tiles.length"):
                with v3.VCard(
                    v_for="tile in run_tiles", key="tile.label", elevation=1, classes="pa-3"
                ):
                    html.Div("{{ tile.label }}", classes="text-caption text-medium-emphasis")
                    html.Div("{{ tile.value }}", classes="text-h6 af-number")
                    v3.VProgressLinear(
                        v_if="tile.progress !== null", model_value=("tile.progress * 100",),
                        color="primary", height=6, rounded=True, classes="my-1",
                    )  # fmt: skip
                    html.Div("{{ tile.detail }}", classes="text-caption")
            with v3.VCard(elevation=1):
                with v3.VCardItem(classes="py-2"):
                    v3.VCardTitle("Residual", classes="text-subtitle-1")
                    with v3.Template(v_slot_append=True):
                        _legend(
                            ("dash", "stopping target"),
                            ("retry", "redone step"),
                            ("refit", "preconditioner refit"),
                        )
                        with v3.VBtnToggle(
                            v_model=("run_residual_view",), mandatory=True, density="compact",
                            variant="outlined", divided=True, classes="ml-3",
                        ):  # fmt: skip
                            v3.VBtn("Overall", value=OVERALL, size="small")
                            v3.VBtn(
                                "Per equation", value=BY_EQUATION, size="small",
                                disabled=("!run_has_equations",),
                            )  # fmt: skip
                self._figure("residual", "height: 300px;")
            with html.Div(classes="af-plot-grid"):
                for key, title, limit in _SMALL_PLOTS:
                    with v3.VCard(elevation=1):
                        with v3.VCardItem(classes="py-1"):
                            v3.VCardTitle(title, classes="text-subtitle-2")
                            with v3.Template(v_slot_append=True):
                                _legend(("dash", limit))
                        self._figure(key, "height: 170px;")
            with v3.VCard(elevation=1, classes="pb-2"):
                with v3.VCardItem(classes="py-2"):
                    v3.VCardTitle("What happened", classes="text-subtitle-1")
                    with v3.Template(v_slot_append=True):
                        with v3.VBtnToggle(
                            v_model=("run_feed",), mandatory=True, density="compact",
                            variant="outlined", divided=True,
                        ):  # fmt: skip
                            v3.VBtn("Events", value="events", size="small")
                            v3.VBtn("Log", value="log", size="small")
                with html.Div(v_if="run_feed === 'events'", classes="px-4"):
                    html.Div(
                        "Nothing yet: retries, preconditioner refits and the end of a "
                        "continuation are listed here as they happen.",
                        v_if="!run_events.length",
                        classes="text-body-2 text-medium-emphasis py-2",
                    )
                    with html.Div(
                        v_for="(event, i) in run_events", key="i", classes="af-event"
                    ):  # fmt: skip
                        html.Span("step {{ event.step }}", classes="text-medium-emphasis af-number")
                        html.Span(classes=("'af-event-dot af-event-' + event.kind",))
                        html.Span("{{ event.text }}")
                        html.Span(
                            "{{ event.seconds === null ? '' : event.seconds.toFixed(1) + ' s' }}",
                            classes="text-caption text-medium-emphasis",
                        )
                html.Pre(
                    "{{ run_log || 'Nothing printed yet.' }}",
                    v_if="run_feed === 'log'",
                    classes="af-log mx-4",
                )

    def _figure(self, key: str, style: str) -> None:
        figure = plotly.Figure(display_logo=False, display_mode_bar=False, style=style)
        self._figures[key] = figure.update

    def toolbar(self) -> None:
        """No buttons of its own."""

    def shown(self) -> None:
        """Bring what will run up to date: the case may have been saved or changed in Setup."""
        self._spawn(self.replan())


#: A history with no steps, for drawing the plots' waiting state.
_NO_STEPS = ConvergenceHistory({"step": np.zeros(0)})


def _key_values(rows: str) -> None:
    """A two-column list of ``{label, value}`` rows held in the state variable ``rows``."""
    with html.Div(classes="af-kv"):
        with html.Template(v_for=f"row in {rows}", key="row.label"):
            html.Span("{{ row.label }}", classes="text-medium-emphasis")
            html.Span("{{ row.value }}", classes="af-number")


def _legend(*entries: tuple[str, str]) -> None:
    """The marks a plot uses, named: ``("dash" | "retry" | "refit", label)`` each."""
    with html.Div(classes="af-legend"):
        for mark, label in entries:
            with html.Span():
                html.Span(classes=f"af-mark af-mark-{mark}")
                html.Span(label)


def _status_rows(history: ConvergenceHistory, limits: RunLimits) -> list[dict]:
    """How far the march has come: steps, wall time, the last step's time, the residual and target."""
    if history.n_steps == 0:
        return []
    seconds = history.columns.get("seconds")
    rows = [{"label": "Step", "value": str(history.n_steps)}]
    if seconds is not None:
        rows.append({"label": "Elapsed", "value": _clock(float(seconds[-1]))})
        last = float(seconds[-1] - (seconds[-2] if history.n_steps > 1 else 0.0))
        rows.append({"label": "Last step", "value": f"{last:.1f} s"})
    arrived = history.columns.get("arrived")
    if arrived is not None:
        on_case = bool(arrived[-1])
        rows.append({"label": "Problem", "value": "the case's own" if on_case else "continuation"})
    rows.append({"label": "Residual", "value": f"{history.columns['residual_norm'][-1]:.2e}"})
    if limits.atol is not None or limits.rtol is not None:
        rows.append({"label": "Stops below", "value": _stop_text(limits)})
    return rows


def _stop_text(limits: RunLimits) -> str:
    parts = []
    if limits.atol:
        parts.append(f"{limits.atol:.1e}")
    if limits.rtol:
        parts.append(f"{limits.rtol:.1e} times the first residual")
    return " + ".join(parts) or "0"


def _clock(seconds: float) -> str:
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def _relative(directory: Path, case: Path | None) -> str:
    """``directory`` as seen from the case file's own directory, where that is shorter."""
    if case is None:
        return str(directory)
    with contextlib.suppress(ValueError):
        return os.path.relpath(directory, case.parent) + "/"
    return str(directory)


def _ending(run: CaseRun) -> tuple[str, str]:
    """The run state and the status line for a run that has ended."""
    status = run.status
    if run.stopped:
        return "stopped", "Stopped"
    if status == CONVERGED:
        return "converged", "Converged"
    if status == NOT_CONVERGED:
        return "short", "Ended short of its stopping test"
    if status == REFUSED:
        return "failed", "Refused: see the log"
    return "failed", f"Ended with status {status}: see the log"
