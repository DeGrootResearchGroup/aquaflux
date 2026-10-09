"""What the Run section shows of a march: its plots, its headline numbers and what happened along it.

Everything here is a pure function of the march's per-step history (:class:`ConvergenceHistory`) and
the limits its case file sets (:class:`RunLimits`), so it can be drawn for a run that is still going --
the history is re-read as it grows -- and tested without a page. The history's columns are the
solver's own names (its ``StepReport`` fields, ``refits``, ``retry_reasons``, and one
``residual_of_<equation>`` per equation the march's measure names); a history written before a column
existed simply lacks what that column shows.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go

from .case_form import CaseSchema, setting
from .history import ConvergenceHistory
from .plots import base_figure
from .theme import LIGHT, PlotStyle

__all__ = [
    "EQUATION_PREFIX",
    "RETRY_REASONS",
    "RunEvent",
    "RunLimits",
    "cost_figures",
    "residual_figure",
    "run_events",
    "run_tiles",
]

#: The prefix of a per-equation residual column in the history.
EQUATION_PREFIX = "residual_of_"

#: What each of the march's reasons for redoing a step means, said for a reader of the run.
RETRY_REASONS: Mapping[str, str] = {
    "cycles": "a solve took more restart cycles than the retry allows",
    "alpha": "the line search collapsed, so it was redone at a larger shift",
    "diverged": "the residual diverged, so it was redone at a larger shift",
    "solver": "it was redone with the tighter linear solver",
}

#: Margins of the four small plots, which sit two by two beneath the residual.
_SMALL_MARGIN = {"l": 48, "r": 12, "t": 8, "b": 28}


@dataclasses.dataclass(frozen=True)
class RunLimits:
    """The limits a case's march runs under, as its case file sets them (or defaults them).

    Any of them is ``None`` when the case does not say and no default is stated, and then it is not
    drawn.

    Attributes
    ----------
    rtol, atol : float or None
        The stopping test: the march stops below ``atol + rtol * R0``, ``R0`` its first residual.
    max_steps : int or None
        The step cap, per segment.
    inner_steps : int or None
        The inner Newton iterations a step may take; ``None`` without an inner loop.
    abort_above_cycles : int or None
        The restart cycles one solve may take before the retry redoes the step.
    beta_min : float or None
        The smallest pseudo-time shift the step control may take.
    retry_alpha : float or None
        The line-search step below which the retry redoes the step.
    """

    rtol: float | None = None
    atol: float | None = None
    max_steps: int | None = None
    inner_steps: int | None = None
    abort_above_cycles: int | None = None
    beta_min: float | None = None
    retry_alpha: float | None = None

    @classmethod
    def from_case(cls, schema: CaseSchema, document: Mapping) -> RunLimits:
        """The limits ``document`` runs under, its unset settings taken at their defaults."""

        def number(*path: str) -> float | None:
            value = setting(schema, document, ("solver", *path))
            return value if isinstance(value, int | float) and not isinstance(value, bool) else None

        return cls(
            rtol=number("convergence", "rtol"),
            atol=number("convergence", "atol"),
            max_steps=number("max_steps"),
            inner_steps=number("dual_time", "inner_steps"),
            abort_above_cycles=number("retry", "abort_above_cycles"),
            beta_min=number("step_control", "beta_min"),
            retry_alpha=number("retry", "on_alpha"),
        )

    def target(self, reference: float | None) -> float | None:
        """The residual the march stops below, given its first residual ``reference``."""
        if self.atol is None and self.rtol is None:
            return None
        relative = 0.0 if self.rtol is None or reference is None else self.rtol * reference
        return (self.atol or 0.0) + relative


@dataclasses.dataclass(frozen=True)
class RunEvent:
    """Something that happened at one step of a march.

    Attributes
    ----------
    step : int
        The step's 1-based count over the whole march.
    kind : {"retry", "refit", "arrived"}
        What happened: the step was redone, the preconditioner was refitted for it, or it was the
        first step on the case's own problem after a continuation.
    text : str
        What happened, for a reader.
    seconds : float or None
        What it cost in wall time, where the history records it.
    """

    step: int
    kind: str
    text: str
    seconds: float | None = None

    def to_state(self) -> dict:
        """As plain data for the page: Python numbers, which the page's state can carry."""
        seconds = None if self.seconds is None else float(self.seconds)
        return {"step": int(self.step), "kind": self.kind, "text": self.text, "seconds": seconds}


def run_events(history: ConvergenceHistory) -> list[RunEvent]:
    """What happened along the march, newest first."""
    events: list[RunEvent] = []
    steps = _ints(history, "step")
    reasons = history.columns.get("retry_reasons")
    refits = history.columns.get("refits")
    refit_seconds = history.columns.get("refit_seconds")
    arrived = history.columns.get("arrived")
    for index, step in enumerate(steps):
        if arrived is not None and index > 0 and arrived[index] and not arrived[index - 1]:
            events.append(
                RunEvent(step, "arrived", "Continuation finished: the case's own problem")
            )
        if refits is not None and refits[index] > 0:
            count = int(refits[index])
            seconds = None if refit_seconds is None else float(refit_seconds[index])
            times = "" if count == 1 else f" {count} times"
            events.append(RunEvent(step, "refit", f"Preconditioner refitted{times}", seconds))
        for reason in _reasons(reasons, index):
            text = RETRY_REASONS.get(reason, f"it was redone ({reason})")
            events.append(RunEvent(step, "retry", f"Step redone: {text}"))
    return events[::-1]


def run_tiles(history: ConvergenceHistory, limits: RunLimits) -> list[dict]:
    """The headline numbers: residual against target, steps, linear cost, retries and refits.

    Returns
    -------
    list of dict
        ``{"label", "value", "detail", "progress"}`` each, ``progress`` in ``[0, 1]`` or ``None``.
    """
    n = history.n_steps
    if n == 0:
        return []
    residual = history.columns["residual_norm"]
    target = limits.target(_reference(history))
    last = float(residual[-1])
    progress, detail = None, ""
    if target is not None and target > 0:
        detail = f"target {target:.1e}"
        start = float(residual[0])
        if last <= target:
            progress, detail = 1.0, f"target {target:.1e} · reached"
        elif start > target:
            progress = max(0.0, min(1.0, math.log(start / last) / math.log(start / target)))
            detail += f" · {math.log10(last / target):.1f} decades to go"
    cycles = history.columns.get("restart_cycles", np.zeros(n))
    retried = _ints(history, "step")[_retried(history)]
    refits = history.columns.get("refits", np.zeros(n))
    refitted = _ints(history, "step")[refits > 0]
    cap = "" if limits.max_steps is None else f"≤ {limits.max_steps} per segment"
    return [
        {"label": "Residual", "value": f"{last:.2e}", "detail": detail, "progress": progress},
        {"label": "Step", "value": str(n), "detail": cap, "progress": None},
        {
            "label": "Linear cycles",
            "value": str(int(cycles.sum())),
            "detail": f"{cycles.sum() / n:.1f} per step",
            "progress": None,
        },
        {
            "label": "Retries",
            "value": str(len(retried)),
            "detail": f"last at step {retried[-1]}" if len(retried) else "none",
            "progress": None,
        },
        {
            "label": "Preconditioner refits",
            "value": str(int(refits.sum())),
            "detail": f"last at step {refitted[-1]}" if len(refitted) else "none",
            "progress": None,
        },
    ]


def residual_figure(
    history: ConvergenceHistory, limits: RunLimits, theme: str = LIGHT, by_equation: bool = False
) -> go.Figure:
    """The residual against the step, with the stopping target, retries, refits and the continuation.

    Parameters
    ----------
    history : ConvergenceHistory
        The march so far.
    limits : RunLimits
        The case's limits; the target is drawn when it is known.
    theme : {"light", "dark"}
        The page's theme.
    by_equation : bool
        Draw each equation's term of the residual (when the history holds them) rather than the total.
        Their Euclidean combination is the total, so the line nearest it is the equation holding the
        march up; the target applies to the total only, so it is not drawn with them.
    """
    figure, style = base_figure(theme, margin={"l": 60, "r": 16, "t": 12, "b": 40}, x_title="Step")
    if history.n_steps == 0:
        return _waiting(figure)
    steps = history.steps
    equations = equation_columns(history)
    if by_equation and equations:
        for (column, name), colour in zip(
            equations.items(), _cycled(style.equations), strict=False
        ):
            values = history.columns[column]
            figure.add_trace(
                go.Scatter(
                    x=steps, y=values, mode="lines", name=name, line={"color": colour, "width": 2}
                )
            )
            # Labelled at its end as well as in the legend: identity is never colour alone.
            figure.add_annotation(
                x=steps[-1], y=math.log10(max(float(values[-1]), 1e-300)), text=name,
                showarrow=False, xanchor="left", xshift=4, font={"color": style.text, "size": 11},
            )  # fmt: skip
    else:
        residual = history.columns["residual_norm"]
        figure.add_trace(
            go.Scatter(
                x=steps, y=residual, mode="lines+markers", name="Residual",
                line={"color": style.lines[0], "width": 2}, marker={"size": 5},
            )
        )  # fmt: skip
        retried = _retried(history)
        if retried.any():
            figure.add_trace(
                go.Scatter(
                    x=steps[retried], y=residual[retried], mode="markers", name="Redone",
                    marker={"size": 11, "color": style.retry, "line": {"width": 2, "color": "white"}},
                )
            )  # fmt: skip
        target = limits.target(_reference(history))
        if target is not None and target > 0:
            figure.add_hline(
                y=target, line={"color": style.muted, "width": 1.5, "dash": "dash"},
                annotation_text=f"target {target:.0e}", annotation_position="bottom right",
                annotation_font={"color": style.muted, "size": 11},
            )  # fmt: skip
    _refit_ticks(figure, history, style)
    _continuation_band(figure, history, style, labelled=True)
    figure.update_yaxes(type="log", exponentformat="e", title="Residual")
    # Top right, clear of the continuation's label at the top left of its shading.
    figure.update_layout(
        showlegend=by_equation and bool(equations),
        legend={"orientation": "h", "x": 1.0, "xanchor": "right", "y": 1.0, "yanchor": "bottom"},
    )
    figure.update_xaxes(range=_x_range(history))
    return figure


def cost_figures(
    history: ConvergenceHistory, limits: RunLimits, theme: str = LIGHT
) -> dict[str, go.Figure]:
    """The four small plots beneath the residual, by name, each on the residual's step axis.

    - ``cycles``: each step's restart cycles summed over its solves, and its hardest single solve
      against the cycles above which the retry redoes the step;
    - ``inner``: inner Newton iterations against the inner loop's limit;
    - ``shift``: the pseudo-time shift against its smallest allowed value;
    - ``alpha``: the line-search step against the retry's threshold.
    """
    figures = {}
    names = ("cycles", "inner", "shift", "alpha")
    for name in names:
        figure, style = base_figure(theme, margin=_SMALL_MARGIN, x_title="")
        figure.update_layout(showlegend=False)
        figures[name] = figure
    if history.n_steps == 0:
        return {name: _waiting(figure) for name, figure in figures.items()}
    steps, columns, retried = history.steps, history.columns, _retried(history)
    bar_colours = [style.retry if redone else style.lines[0] for redone in retried]

    cycles = figures["cycles"]
    if "restart_cycles" in columns:
        cycles.add_trace(
            go.Bar(x=steps, y=columns["restart_cycles"], marker={"color": bar_colours},
                   name="Restart cycles, all solves")
        )  # fmt: skip
    if "max_inner_cycles" in columns:
        cycles.add_trace(
            go.Scatter(x=steps, y=columns["max_inner_cycles"], mode="markers", name="Hardest solve",
                       marker={"size": 6, "color": style.text, "symbol": "diamond"})
        )  # fmt: skip
    _limit(cycles, limits.abort_above_cycles, style)

    inner = figures["inner"]
    if "inner_iterations" in columns:
        inner.add_trace(
            go.Bar(x=steps, y=columns["inner_iterations"], marker={"color": style.muted},
                   name="Inner iterations")
        )  # fmt: skip
    _limit(inner, limits.inner_steps, style)

    shift = figures["shift"]
    if "shift" in columns:
        shift.add_trace(
            go.Scatter(x=steps, y=columns["shift"], mode="lines", name="Shift",
                       line={"color": style.lines[0], "width": 2, "shape": "hv"})
        )  # fmt: skip
        shift.update_yaxes(type="log", exponentformat="e")
    _limit(shift, limits.beta_min, style)

    alpha = figures["alpha"]
    if "alpha" in columns:
        # A step whose line search collapsed to nothing has no place on a logarithmic axis; drawn at
        # the axis floor rather than dropped, so the collapse stays visible.
        values = np.maximum(columns["alpha"], 1e-4)
        alpha.add_trace(
            go.Scatter(x=steps, y=values, mode="markers", name="Line-search step",
                       marker={"size": 7, "color": bar_colours})
        )  # fmt: skip
        alpha.update_yaxes(type="log", exponentformat="e")
    _limit(alpha, limits.retry_alpha, style)

    for figure in figures.values():
        _continuation_band(figure, history, style, labelled=False)
        figure.update_xaxes(range=_x_range(history))
    return figures


def equation_columns(history: ConvergenceHistory) -> dict[str, str]:
    """The history's per-equation residual columns, each with its equation's name, in file order."""
    return {
        name: name[len(EQUATION_PREFIX) :]
        for name in history.columns
        if name.startswith(EQUATION_PREFIX) and history.columns[name].dtype.kind == "f"
    }


# --- helpers ---------------------------------------------------------------------------------------


def _ints(history: ConvergenceHistory, name: str) -> np.ndarray:
    return np.asarray(history.columns[name], dtype=int)


def _reference(history: ConvergenceHistory) -> float | None:
    """The first residual the stopping test is relative to: ``residual / residual_ratio``."""
    ratio = history.columns.get("residual_ratio")
    if ratio is None or history.n_steps == 0 or not ratio[0] > 0:
        return None
    return float(history.columns["residual_norm"][0] / ratio[0])


def _retried(history: ConvergenceHistory) -> np.ndarray:
    """Which steps were redone, by any of the records that say so."""
    n = history.n_steps
    redone = np.zeros(n, dtype=bool)
    for name in ("escalations", "diverged_retry"):
        if name in history.columns:
            redone |= np.asarray(history.columns[name], dtype=float) > 0
    if "retry_reasons" in history.columns:
        redone |= np.asarray(
            [bool(_reasons(history.columns["retry_reasons"], i)) for i in range(n)]
        )
    return redone


def _reasons(column: np.ndarray | None, index: int) -> list[str]:
    """The reasons recorded for one step, which the history joins with ``;``."""
    if column is None:
        return []
    cell = column[index]
    if not isinstance(cell, str):  # a column with no reason anywhere reads as numbers
        return []
    return [reason for reason in cell.split(";") if reason]


def _refit_ticks(figure: go.Figure, history: ConvergenceHistory, style: PlotStyle) -> None:
    """A short mark at the foot of the plot under each step the preconditioner was refitted for."""
    refits = history.columns.get("refits")
    if refits is None:
        return
    for step in history.steps[np.asarray(refits) > 0]:
        figure.add_shape(
            type="line", x0=step, x1=step, yref="paper", y0=0.0, y1=0.06,
            line={"color": style.refit, "width": 2},
        )  # fmt: skip


def _continuation_band(
    figure: go.Figure, history: ConvergenceHistory, style: PlotStyle, *, labelled: bool
) -> None:
    """Shade the steps a continuation spent on the way to the case's own problem."""
    arrived = history.columns.get("arrived")
    if arrived is None:
        return
    before = history.steps[np.asarray(arrived) == 0]
    if len(before) == 0:
        return
    # A shape given an empty label shows plotly's placeholder text, so a label is passed only when wanted.
    label = (
        {
            "annotation_text": "continuation",
            "annotation_position": "top left",
            "annotation_font": {"color": style.muted, "size": 11},
        }
        if labelled
        else {}
    )
    figure.add_vrect(
        x0=before.min() - 0.5, x1=before.max() + 0.5, fillcolor=style.band, line_width=0,
        layer="below", **label,
    )  # fmt: skip


def _limit(figure: go.Figure, value: float | None, style: PlotStyle) -> None:
    """A dashed line at a limit the case sets, when it sets one."""
    if value is not None:
        figure.add_hline(y=value, line={"color": style.muted, "width": 1.5, "dash": "dash"})


def _x_range(history: ConvergenceHistory) -> list[float]:
    """The step axis every plot shares: the steps so far, with room to grow."""
    last = float(history.steps[-1]) if history.n_steps else 1.0
    return [0.5, max(10.0, last * 1.15) + 0.5]


def _waiting(figure: go.Figure) -> go.Figure:
    figure.add_annotation(
        text="Waiting for the first step", showarrow=False, xref="paper", yref="paper"
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


def _cycled(colours: Sequence[str]):
    """The colours in order, then again: more equations than colours repeat them."""
    while True:
        yield from colours
