"""The Results section: a run's datasets drawn in a 3D view, and its convergence history.

The scene is rendered here, by VTK, and the browser is sent images of it, so a mesh of millions of
cells is never sent to the page. :class:`ResultsSection` holds the one source, the plotter and the
scene, and routes every control change through :mod:`aquaflux_ui.controls`, which holds the rules
for what an edit does and turns the controls' values into a :class:`~aquaflux_ui.scene.View`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import plotly.graph_objects as go
import pyvista as pv
from trame.widgets import html, plotly
from trame.widgets import vtk as vtk_widgets
from trame.widgets import vuetify3 as v3

from . import controls
from .controls import COLORMAPS, MAGNITUDE
from .history import ConvergenceHistory
from .scene import AXES, Scene, automatic_range, field_components, field_values
from .sources import ResultSource
from .theme import LIGHT, PLOT_STYLES, THEMES, colormap_gradient
from .widgets import icon_button, panel

__all__ = ["ResultsSection", "convergence_figure"]

#: The opacity an opaque surface is faded to when the first slice or threshold is added, so what is
#: inside it shows.
FOCUS_CONTEXT_OPACITY = 0.2

#: Events a number box commits on: leaving it, or pressing Enter -- not every keystroke, so a value
#: part-way through being typed ("-", "0.") is never sent. A box shows its value to six significant
#: figures; the entry keeps it to full precision.
_COMMIT_EVENTS = [("keyup_enter", "keyup.enter"), "blur"]


def convergence_figure(history: ConvergenceHistory | None, theme: str = LIGHT) -> go.Figure:
    """The residual against the step, on a logarithmic axis; an empty figure saying why if none.

    Parameters
    ----------
    history : ConvergenceHistory or None
        The march's record.
    theme : {"light", "dark"}
        The page's theme, whose plot colours it is drawn in.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    style = PLOT_STYLES[theme if theme in PLOT_STYLES else LIGHT]
    axis = {"gridcolor": style.grid, "zerolinecolor": style.grid, "linecolor": style.grid}
    figure = go.Figure()
    figure.update_layout(
        margin={"l": 60, "r": 20, "t": 30, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "system-ui, sans-serif", "color": style.text, "size": 12},
        xaxis={"title": "Step", **axis},
        yaxis=axis,
        legend={"orientation": "h", "y": 1.15, "x": 0},
    )
    columns = {} if history is None else history.residual_columns()
    if not columns or history.n_steps == 0:
        figure.add_annotation(
            text="No convergence history: this run took no march steps"
            if history is not None
            else "No convergence history saved for these results",
            showarrow=False,
            xref="paper",
            yref="paper",
        )
        return figure
    for (name, label), colour in zip(columns.items(), style.lines, strict=False):
        figure.add_trace(
            go.Scatter(
                x=history.steps,
                y=history.columns[name],
                mode="lines+markers",
                name=label,
                line={"color": colour, "width": 2},
                marker={"size": 5},
            )
        )
    figure.update_yaxes(type="log", exponentformat="e")
    return figure


class ResultsSection:
    """The Results section: one source's datasets drawn in a 3D view, and its convergence history.

    Parameters
    ----------
    source : ResultSource
        What is shown.
    server : trame server
        The workspace's server, whose state this section's controls live in. Its ``theme`` state
        must already be set.
    """

    key, title, icon = "results", "Results", "mdi-chart-box-outline"

    def __init__(self, source: ResultSource, server) -> None:
        self.source = source
        self.server = server
        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.enable_anti_aliasing("fxaa")
        self.scene = Scene(self.plotter, THEMES[LIGHT])
        self._frame = source.frame(0)
        self._ranges: dict[tuple[int, str], tuple[float, float] | None] = {}
        self._next_id = 0
        self._build_state()
        state = self.server.state
        state.change("dataset")(self._on_dataset)
        state.change("field", "component", "log_scale")(self._on_field)
        state.change(
            "colormap", "auto_range", "range_min", "range_max", "surface", "surface_opacity",
            "edges", "slices", "thresholds",
        )(self._on_view)  # fmt: skip
        state.change("snapshot")(self._on_snapshot)
        state.change("theme")(self._on_theme)
        self.server.controller.on_server_ready.add(lambda **_: self.redraw())
        self.server.controller.on_server_ready.add(lambda **_: self.refresh_history())

    # --- state ----------------------------------------------------------------------------------

    def _build_state(self) -> None:
        state = self.server.state
        names = list(self._frame)
        times = self.source.times()
        state.update(
            {
                "info": [{"label": k, "value": v} for k, v in self.source.info().items()],
                "datasets": names,
                "dataset": _first_volume(self._frame) or names[0],
                "snapshot": 0,
                "n_snapshots": len(times),
                "snapshot_time": times[0],
                "results_panels": ["record", "colour", "slices", "thresholds"],
                "colormaps": [
                    {"title": name, "value": name, "gradient": colormap_gradient(name)}
                    for name in COLORMAPS
                ],
                "colormap": COLORMAPS[0],
                "log_scale": False,
                "auto_range": True,
                "surface": True,
                "surface_opacity": 1.0,
                "edges": False,
                "axes": list(AXES),
                "slices": [],
                "thresholds": [],
                "has_history": self.source.history() is not None,
            }
        )
        self._choose_dataset(state.dataset)

    @property
    def _mesh(self) -> pv.DataSet:
        return self._frame[self.server.state.dataset]

    def _field_ranges(self) -> dict[str, tuple[float, float] | None]:
        """Each field's finite extent on the current dataset (a vector's magnitude), computed once."""
        mesh = self._mesh
        ranges = {}
        for name in field_components(mesh):
            key = (id(mesh), name)
            if key not in self._ranges:
                self._ranges[key] = automatic_range(field_values(mesh, name))
            ranges[name] = self._ranges[key]
        return ranges

    def _choose_dataset(self, name: str) -> None:
        """Offer the dataset's fields, keeping the current one if it has it, and refit the entries."""
        state = self.server.state
        fields = field_components(self._frame[name])
        state.fields = [{"title": "(None)", "value": ""}] + [
            {"title": f, "value": f} for f in fields
        ]
        state.threshold_fields = list(fields)
        if state.field not in fields:
            state.field = next(iter(fields), "")
        bounds = self._mesh.bounds
        state.slices = [controls.fit_slice(entry, bounds) for entry in state.slices]
        ranges = self._field_ranges()
        state.thresholds = [
            controls.fit_threshold(entry, ranges, state.field or next(iter(fields), ""))
            for entry in state.thresholds
            if fields
        ]
        self._choose_field()

    def _choose_field(self) -> None:
        """Offer the field's components and reset the colour range to its own."""
        state = self.server.state
        n = field_components(self._mesh).get(state.field, 0) if state.field else 0
        state.components = (
            [{"title": "Magnitude", "value": MAGNITUDE}]
            + [{"title": str(i), "value": i} for i in range(n)]
            if n > 1
            else []
        )
        if n <= 1 or state.component not in range(-1, n):
            state.component = MAGNITUDE
        values = (
            field_values(
                self._mesh, state.field, None if state.component == MAGNITUDE else state.component
            )
            if state.field
            else np.empty(0)
        )
        state.range_min, state.range_max = automatic_range(values, bool(state.log_scale)) or (
            0.0,
            1.0,
        )

    # --- entries --------------------------------------------------------------------------------

    def _ident(self) -> int:
        self._next_id += 1
        return self._next_id

    def _fade_surface(self) -> None:
        """Before the first slice or threshold, fade an opaque surface so what is inside it shows."""
        state = self.server.state
        if (
            not (state.slices or state.thresholds)
            and state.surface
            and state.surface_opacity == 1.0
        ):
            state.surface_opacity = FOCUS_CONTEXT_OPACITY

    def add_slice(self) -> None:
        """Add a slice through the middle of the dataset."""
        self._fade_surface()
        state = self.server.state
        state.slices = [*state.slices, controls.new_slice(self._ident(), self._mesh.bounds)]

    def edit_slice(self, index: int, key: str, value: object) -> None:
        """Change one control of the ``index``-th slice."""
        entries = list(self.server.state.slices)
        entries[index] = controls.edit_slice(entries[index], key, value, self._mesh.bounds)
        self.server.state.slices = entries

    def remove_slice(self, index: int) -> None:
        """Remove the ``index``-th slice."""
        state = self.server.state
        state.slices = [entry for i, entry in enumerate(state.slices) if i != index]

    def add_threshold(self) -> None:
        """Add a threshold on the colour field (or the first field), keeping its upper half."""
        state = self.server.state
        fields = state.threshold_fields
        if not fields:
            return
        field = state.field if state.field in fields else fields[0]
        entry = controls.new_threshold(self._ident(), field, self._field_ranges()[field])
        self._fade_surface()
        state.thresholds = [*state.thresholds, entry]

    def edit_threshold(self, index: int, key: str, value: object) -> None:
        """Change one control of the ``index``-th threshold."""
        entries = list(self.server.state.thresholds)
        entries[index] = controls.edit_threshold(entries[index], key, value, self._field_ranges())
        self.server.state.thresholds = entries

    def remove_threshold(self, index: int) -> None:
        """Remove the ``index``-th threshold."""
        state = self.server.state
        state.thresholds = [entry for i, entry in enumerate(state.thresholds) if i != index]

    # --- reactions ------------------------------------------------------------------------------

    def _on_dataset(self, dataset, **_) -> None:
        self._choose_dataset(dataset)
        self.redraw()

    def _on_field(self, **_) -> None:
        self._choose_field()
        self.redraw()

    def _on_view(self, **_) -> None:
        self.redraw()

    def shown(self) -> None:
        """The section was brought into view: send the view a fresh frame of its current size."""
        self.server.controller.view_update()

    def _on_theme(self, theme, **_) -> None:
        self.scene.style = THEMES[theme if theme in THEMES else LIGHT]
        self.redraw()
        self.refresh_history()

    def _on_snapshot(self, snapshot, **_) -> None:
        index = int(snapshot)
        self._frame = self.source.frame(index)
        self.server.state.snapshot_time = self.source.times()[index]
        if self.server.state.dataset not in self._frame:
            self.server.state.dataset = next(iter(self._frame))
        self.redraw()

    def refresh_history(self) -> None:
        """Re-read the history, which a run still in progress is adding to, and redraw the plot."""
        history = self.source.history()
        self.server.state.has_history = history is not None
        self.server.controller.figure_update(convergence_figure(history, self.server.state.theme))

    def redraw(self) -> None:
        """Draw the view the controls describe."""
        self.scene.show(self._frame, controls.view_from_state(self.server.state.to_dict()))
        self.server.controller.view_update()

    # --- page -----------------------------------------------------------------------------------

    def drawer(self) -> None:
        """The side panel: the run's record, then what is drawn of it."""
        with v3.VExpansionPanels(
            v_model=("results_panels",), multiple=True, variant="accordion", flat=True
        ):
            panel("record", "Run record", "mdi-information-outline", _record_panel)
            panel("colour", "Colour", "mdi-palette-outline", _colour_panel)
            panel("surface", "Surface", "mdi-cube-outline", _surface_panel)
            panel(
                "slices", "Slices", "mdi-layers-outline", self._slice_panel,
                count="slices.length",
            )  # fmt: skip
            panel(
                "thresholds", "Thresholds", "mdi-filter-variant", self._threshold_panel,
                count="thresholds.length",
            )  # fmt: skip

    def main(self) -> None:
        """The rendered view, and the convergence plot beneath it."""
        with html.Div(classes="d-flex flex-column pa-3", style="height: 100%; gap: 12px;"):
            with v3.VCard(classes="af-view flex-grow-1", elevation=1, style="min-height: 0;"):
                view = vtk_widgets.VtkRemoteView(self.plotter.render_window, interactive_ratio=1)
                self.server.controller.view_update = view.update
            with v3.VCard(v_show="has_history", elevation=1, style="flex: none;"):
                with v3.VCardItem(classes="py-2"):
                    with v3.Template(v_slot_prepend=True):
                        v3.VIcon("mdi-chart-line", color="primary", size="small")
                    v3.VCardTitle("Convergence", classes="text-subtitle-1")
                    with v3.Template(v_slot_append=True):
                        v3.VBtn(
                            "Reload",
                            prepend_icon="mdi-refresh",
                            size="small",
                            variant="text",
                            click=self.refresh_history,
                        )
                figure = plotly.Figure(
                    display_logo=False, display_mode_bar=False, style="height: 210px;"
                )
                self.server.controller.figure_update = figure.update

    def toolbar(self) -> None:
        """The section's buttons in the app bar."""
        icon_button("mdi-fit-to-screen-outline", "Reset camera", self.reset_camera)

    def _slice_panel(self) -> None:
        with v3.VCard(
            v_for="(entry, i) in slices", key="entry.id", variant="tonal", classes="mb-3 pa-3"
        ):
            with html.Div(classes="d-flex align-center mb-1"):
                html.Span("Slice {{ i + 1 }}", classes="text-subtitle-2")
                v3.VSpacer()
                with v3.VBtnToggle(
                    model_value=("entry.axis",),
                    update_modelValue=(self.edit_slice, "[i, 'axis', $event]"),
                    mandatory=True,
                    density="compact",
                    variant="outlined",
                    color="primary",
                    divided=True,
                    rounded="lg",
                ):
                    for axis in AXES:
                        v3.VBtn(axis.upper(), value=axis, size="small", min_width=36)
                icon_button("mdi-close", "Remove slice", (self.remove_slice, "[i]"))
            v3.VSlider(
                model_value=("entry.coordinate",),
                update_modelValue=(self.edit_slice, "[i, 'coordinate', $event]"),
                min=("entry.min",),
                max=("entry.max",),
                step=("(entry.max - entry.min) / 500",),
                hide_details=True,
            )
            v3.VTextField(
                label=("`${entry.axis.toUpperCase()} coordinate`",),
                model_value=("Number(entry.coordinate.toPrecision(6))",),
                type="number",
                hide_details=True,
                keyup_enter=(self.edit_slice, "[i, 'coordinate', $event.target.value]"),
                blur=(self.edit_slice, "[i, 'coordinate', $event.target.value]"),
                __events=_COMMIT_EVENTS,
            )
        v3.VBtn(
            "Add slice", prepend_icon="mdi-plus", variant="tonal", color="primary", block=True,
            click=self.add_slice,
        )  # fmt: skip

    def _threshold_panel(self) -> None:
        with v3.VCard(
            v_for="(entry, i) in thresholds", key="entry.id", variant="tonal", classes="mb-3 pa-3"
        ):
            with html.Div(classes="d-flex align-center mb-2"):
                html.Span("Threshold {{ i + 1 }}", classes="text-subtitle-2")
                v3.VSpacer()
                icon_button("mdi-close", "Remove threshold", (self.remove_threshold, "[i]"))
            v3.VSelect(
                label="Field",
                model_value=("entry.field",),
                update_modelValue=(self.edit_threshold, "[i, 'field', $event]"),
                items=("threshold_fields",),
                hide_details=True,
            )
            v3.VRangeSlider(
                model_value=("[entry.low, entry.high]",),
                update_modelValue=(self.edit_threshold, "[i, 'range', $event]"),
                min=("entry.min",),
                max=("entry.max",),
                step=("(entry.max - entry.min) / 500",),
                hide_details=True,
                classes="mt-2",
            )
            with html.Div(classes="d-flex", style="gap: 8px;"):
                for end, label in (("low", "Minimum"), ("high", "Maximum")):
                    v3.VTextField(
                        label=label,
                        model_value=(f"Number(entry.{end}.toPrecision(6))",),
                        type="number",
                        hide_details=True,
                        keyup_enter=(self.edit_threshold, f"[i, '{end}', $event.target.value]"),
                        blur=(self.edit_threshold, f"[i, '{end}', $event.target.value]"),
                        __events=_COMMIT_EVENTS,
                    )
        v3.VBtn(
            "Add threshold", prepend_icon="mdi-plus", variant="tonal", color="primary",
            block=True, click=self.add_threshold, disabled=("!threshold_fields.length",),
        )  # fmt: skip

    def reset_camera(self) -> None:
        """Frame the whole dataset again."""
        self.plotter.reset_camera()
        self.server.controller.view_update()


def _record_panel() -> None:
    """What the results are: the run's record, and the snapshot shown."""
    with html.Div(classes="d-flex flex-column", style="gap: 6px;"):
        with html.Div(v_for="row in info", key="row.label", classes="d-flex justify-space-between"):
            html.Span("{{ row.label }}", classes="af-label")
            html.Span("{{ row.value }}", classes="text-body-2 text-right ml-4 text-truncate")
    v3.VSlider(
        v_if="n_snapshots > 1",
        v_model=("snapshot", 0),
        min=0,
        max=("n_snapshots - 1",),
        step=1,
        label="Snapshot",
        hint=("`t = ${snapshot_time}`",),
        persistent_hint=True,
        classes="mt-3",
    )


def _colour_panel() -> None:
    """What the dataset is coloured by, and how."""
    v3.VSelect(label="Dataset", v_model=("dataset",), items=("datasets",), classes="mb-1")
    v3.VSelect(label="Colour by", v_model=("field",), items=("fields",), classes="mb-1")
    v3.VSelect(
        v_if="components.length",
        label="Component",
        v_model=("component",),
        items=("components",),
        classes="mb-1",
    )
    with v3.VSelect(label="Colormap", v_model=("colormap",), items=("colormaps",), classes="mb-1"):
        with v3.Template(v_slot_selection=("{ item }",)):
            html.Div(classes="af-swatch mr-3", style=("{ background: item.raw.gradient }",))
            html.Span("{{ item.title }}")
        with v3.Template(v_slot_item=("{ props, item }",)):
            with v3.VListItem(v_bind="props"):
                with v3.Template(v_slot_prepend=True):
                    html.Div(classes="af-swatch mr-3", style=("{ background: item.raw.gradient }",))
    with html.Div(classes="d-flex", style="gap: 16px;"):
        v3.VSwitch(label="Log scale", v_model=("log_scale",), hide_details=True)
        v3.VSwitch(label="Auto range", v_model=("auto_range",), hide_details=True)
    with html.Div(v_if="!auto_range", classes="d-flex mt-2", style="gap: 8px;"):
        v3.VTextField(label="Min", v_model_number=("range_min",), type="number", hide_details=True)
        v3.VTextField(label="Max", v_model_number=("range_max",), type="number", hide_details=True)


def _surface_panel() -> None:
    """The dataset's outer surface."""
    with html.Div(classes="d-flex", style="gap: 16px;"):
        v3.VSwitch(label="Surface", v_model=("surface",), hide_details=True)
        v3.VSwitch(label="Edges", v_model=("edges",), hide_details=True)
    v3.VSlider(
        v_if="surface",
        label="Opacity",
        v_model=("surface_opacity",),
        min=0.0,
        max=1.0,
        step=0.05,
        thumb_label=True,
        hide_details=True,
    )


def _first_volume(frame: Mapping[str, pv.DataSet]) -> str | None:
    """The first dataset that is not a surface: the one a run's results are mostly about."""
    return next((name for name, mesh in frame.items() if not isinstance(mesh, pv.PolyData)), None)
