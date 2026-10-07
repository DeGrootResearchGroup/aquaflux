"""The Setup section: open a case file, change its settings, save it, and check it against its mesh.

The form is built from the case-file schema the installed solver prints (:mod:`aquaflux_ui.case_form`),
so every choice it offers is one the solver reads. Opening, saving and checking go through the
solver's own commands (:mod:`aquaflux_ui.solver_commands`): the file is read exactly as the solver
reads it, and written by the solver's own writer, which checks the case first and writes nothing it
would refuse.

Saving rewrites the file from its settings, so comments and hand formatting in it are not kept, and a
setting written out at its default value is dropped. The page says so before the first save over a
file it opened.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import tempfile
from pathlib import Path

import pyvista as pv
from trame.widgets import html
from trame.widgets import vtk as vtk_widgets
from trame.widgets import vuetify3 as v3

from . import case_form
from .case_form import CaseSchema, form_sections
from .file_browser import FileBrowser
from .mesh_view import MeshView, patches_addressed
from .solver_commands import CaseDocument, SolverCommands
from .theme import LIGHT, THEMES
from .widgets import empty_state, icon_button

__all__ = ["CASE_SUFFIXES", "SECTION_ICONS", "SetupSection"]

#: The icon a case-file section is headed by, by the section's name; one not named here gets a cog.
#: Only presentation: which sections there are comes from the schema.
SECTION_ICONS = {
    "mesh": "mdi-vector-triangle",
    "fluid": "mdi-water",
    "physics": "mdi-atom-variant",
    "boundaries": "mdi-border-outside",
    "numerics": "mdi-function-variant",
    "drive": "mdi-arrow-right-bold-outline",
    "pressure_datum": "mdi-map-marker-outline",
    "sources": "mdi-plus-box-outline",
    "solver": "mdi-cog-outline",
    "outputs": "mdi-folder-outline",
}

#: The files the file browser offers as case files.
CASE_SUFFIXES = (".yaml", ".yml")

#: Where in the case a refusal says the trouble is: ``... at 'boundaries.inlet': ...``.
_AT = re.compile(r" at '([^']+)'")

#: Widgets whose rows are edited by typing.
_TYPED = ("integer", "number", "string", "numbers", "strings", "raw")

#: The boolean rows' choices: the default, or either value stated.

#: Pixels each level of nesting is indented by; its guide line sits half-way into that margin.
GROUP_INDENT = 14

#: A row's indentation and guide lines, as a page expression: one 2-pixel line per enclosing group, each
#: half-way into that group's indent. Position and size are separate properties, since
#: ``background-image`` takes images only.
_RAIL_STYLE = (
    f"{{ paddingLeft: (row.depth * {GROUP_INDENT}) + 'px', "
    "backgroundImage: Array(row.depth).fill('linear-gradient(var(--af-rail), var(--af-rail))')"
    ".join(', '), "
    f"backgroundPosition: Array.from({{ length: row.depth }}, (_, i) => "
    f"({GROUP_INDENT // 2} + i * {GROUP_INDENT}) + 'px 0').join(', '), "
    "backgroundSize: '2px 100%', backgroundRepeat: 'no-repeat' }"
)

#: A group heading's subtitle, as a page expression: what is set inside it while it is folded, else
#: the one-line description of the kind it holds.
_GROUP_SUBTITLE = (
    "(case_collapsed.includes(row.path) && row.contents) ? row.contents : row.kind_summary"
)


class SetupSection:
    """The Setup section.

    Parameters
    ----------
    server : trame server
        The workspace's server.
    commands : SolverCommands, optional
        How the solver is asked; unset, the installed one.
    case : path-like, optional
        A case file to open when the page is first shown.
    """

    key, title, icon = "setup", "Setup", "mdi-file-cog-outline"

    def __init__(self, server, commands: SolverCommands | None = None, case=None) -> None:
        self.server = server
        self.commands = commands if commands is not None else SolverCommands()
        self.schema: CaseSchema | None = None
        self.document: dict | None = None
        self.path: Path | None = None
        self._confirmed_overwrite: Path | None = None
        self._rows: dict[str, case_form.Row] = {}
        self._tasks: set[asyncio.Task] = set()
        self._schema_task: asyncio.Future | None = None
        self.browser = FileBrowser(server, "case_browser", CASE_SUFFIXES, self._chosen)
        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.enable_anti_aliasing("fxaa")
        self.mesh_view = MeshView(self.plotter, THEMES[LIGHT])
        self._mesh_files = tempfile.TemporaryDirectory(prefix="aquaflux-ui-mesh-")
        self._mesh_loads = 0
        self._groups: dict[str, list[str]] = {}
        server.state.update(
            {
                "case_loaded": False,
                "case_path": "",
                "case_name": "",
                "case_modified": False,
                "case_error": "",
                "case_error_path": "",
                "case_sections": [],
                "case_open_sections": [],
                "case_collapsed": [],
                "case_busy": False,
                "case_opening": "",
                "case_message": "",
                "case_message_ok": True,
                "case_show_message": False,
                "case_confirm": False,
                "case_revision": 0,
                "mesh_loaded": False,
                "mesh_busy": False,
                "mesh_error": "",
                "mesh_stale": False,
                "mesh_cells": 0,
                "mesh_dim": 0,
                "mesh_patches": [],
                "mesh_names": [],
                "mesh_selected": [],
                "mesh_selected_key": "",
                "section_icons": dict(SECTION_ICONS),
            }
        )
        # Asking for the schema starts the solver in a process of its own, which takes seconds: ask
        # as soon as the page is served, so it is ready by the time a case is chosen.
        server.controller.on_server_ready.add(lambda **_: self._spawn(self._prefetch_schema()))
        if case is not None:
            server.controller.on_server_ready.add(lambda **_: self._spawn(self.open(case)))

    # --- opening, saving, checking ---------------------------------------------------------------

    async def _ask(self, function, *arguments):
        """Run a solver command off the page's event loop, showing the page as busy meanwhile."""
        with self.server.state as state:
            state.case_busy = True
        try:
            return await asyncio.to_thread(function, *arguments)
        finally:
            with self.server.state as state:
                state.case_busy = False

    async def _load_schema(self) -> CaseSchema:
        """The case-file schema, asked for once; a request already under way is waited for."""
        if self.schema is None:
            if self._schema_task is None:
                self._schema_task = asyncio.ensure_future(asyncio.to_thread(self.commands.schema))
            try:
                self.schema = await self._schema_task
            except RuntimeError:
                self._schema_task = None  # asked again by the next case opened
                raise
        return self.schema

    async def _prefetch_schema(self) -> None:
        """Ask for the schema before it is needed; a failure is reported when a case is opened."""
        with contextlib.suppress(RuntimeError):
            await self._load_schema()

    async def open(self, path) -> str | None:
        """Open a case file; returns why it could not be read, or ``None``."""
        with self.server.state as state:
            state.case_opening = Path(path).name
        try:
            await self._load_schema()
            document: CaseDocument = await self._ask(self.commands.show, path)
        except RuntimeError as error:
            self._say(str(error), ok=False)
            return str(error)
        finally:
            with self.server.state as state:
                state.case_opening = ""
        if document.case is None or not isinstance(document.case, dict):
            return document.error or f"{path} holds no case."
        self.document, self.path = document.case, Path(path).resolve()
        self._confirmed_overwrite = None
        with self.server.state as state:
            state.case_loaded = True
            state.case_path = str(self.path)
            state.case_name = self.path.name
            state.case_modified = False
            state.case_collapsed = []
            self._show_error(document.error)
            self._lay_out()
            state.case_collapsed = [row["path"] for row in self._default_collapsed()]
            state.case_open_sections = [section["key"] for section in state.case_sections[:1]]
        await self.load_mesh()
        return None

    async def load_mesh(self) -> None:
        """Read or generate the case's mesh, as its mesh section stands, and draw it."""
        if self.document is None or self.path is None:
            return
        section = self.document.get("mesh")
        if not isinstance(section, dict):
            with self.server.state as state:
                state.mesh_error, state.mesh_loaded = "The case names no mesh.", False
            return
        self._mesh_loads += 1
        directory = Path(self._mesh_files.name) / str(self._mesh_loads)
        with self.server.state as state:
            state.mesh_busy, state.mesh_error = True, ""
        try:
            export = await asyncio.to_thread(
                self.commands.mesh, section, self.path.parent, directory
            )
            if export.error is None:
                await asyncio.to_thread(self.mesh_view.load, directory, export.dim)
        finally:
            with self.server.state as state:
                state.mesh_busy = False
        with self.server.state as state:
            if export.error is not None:
                state.mesh_error, state.mesh_loaded = export.error, False
                return
            self._groups = export.groups
            names = [patch["name"] for patch in export.patches]
            state.mesh_loaded, state.mesh_stale = True, False
            state.mesh_cells, state.mesh_dim = export.cells, export.dim
            state.mesh_patches = export.patches
            state.mesh_names = names + list(export.groups)
            state.mesh_selected, state.mesh_selected_key = [], ""
        self.reset_camera()

    def select_boundary(self, name: str) -> None:
        """Highlight the patches a boundary key names (a group's members, or one patch); again to clear."""
        state = self.server.state
        if state.mesh_selected_key == name:
            state.mesh_selected_key, state.mesh_selected = "", []
        else:
            state.mesh_selected_key = name
            state.mesh_selected = list(patches_addressed(name, self._groups))
        self.mesh_view.show(state.mesh_selected)
        self.server.controller.mesh_view_update()

    def _label_clicked(self, widget: str, label: str) -> None:
        """A row's name was clicked: a table entry named after a patch or a group highlights it."""
        if widget == "entry" and label in self.server.state.mesh_names:
            self.select_boundary(label)

    def _on_theme(self, theme, **_) -> None:
        self.mesh_view.style = THEMES[theme if theme in THEMES else LIGHT]
        self.mesh_view.show(self.server.state.mesh_selected)
        self.server.controller.mesh_view_update()

    async def save(self, path=None) -> str | None:
        """Save the case to ``path`` (its own file, unset); returns why it was refused, or ``None``."""
        if self.document is None:
            return "No case is open."
        target = Path(path).resolve() if path is not None else self.path
        if target is None:
            self.browser.show("save", None, "case.yaml")
            return None
        origin = self.path.parent if self.path is not None else None
        error = await self._ask(self.commands.write, target, self.document, origin)
        with self.server.state as state:
            if error:
                self._show_error(error)
                self._lay_out()
                self._say(
                    "Not saved: the case is refused. See the message above the form.", ok=False
                )
                return error
            if origin is not None and origin != target.parent:
                # The relative paths were re-based on writing; show the file as it now reads.
                saved = await self._ask(self.commands.show, target)
                if saved.case is not None:
                    self.document = saved.case
            self.path = target
            self._confirmed_overwrite = target
            state.case_path, state.case_name = str(target), target.name
            state.case_modified = False
            self._show_error(None)
            self._lay_out()
            self._say(f"Saved {target.name}.")
        return None

    async def request_save(self) -> None:
        """Save over the open file -- after a first warning that its comments will not be kept."""
        if self.path is not None and self._confirmed_overwrite != self.path:
            self.server.state.case_confirm = True
            return
        await self.save()

    async def confirm_save(self) -> None:
        """The warning was accepted: save over the open file."""
        self.server.state.case_confirm = False
        self._confirmed_overwrite = self.path
        await self.save()

    async def check(self) -> None:
        """Check the saved file against its mesh."""
        if self.path is None:
            return
        ok, message = await self._ask(self.commands.check, self.path)
        with self.server.state:
            self._say(message or ("Checked." if ok else "The check failed."), ok=ok)

    def _chosen(self, path: str, mode: str) -> str | None:
        """The file browser chose ``path``: open it, or save to it."""
        self._spawn(self._report(self.open(path)) if mode == "open" else self.save(path))
        return None

    def _spawn(self, coroutine) -> None:
        """Run ``coroutine`` on the page's event loop, keeping it alive until it finishes."""
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _report(self, task) -> None:
        error = await task
        if error:
            with self.server.state:
                self._say(error, ok=False)

    def browse(self, mode: str) -> None:
        """Open the file browser, starting where the open case is."""
        start = self.path.parent if self.path is not None else None
        name = self.path.name if (mode == "save" and self.path is not None) else "case.yaml"
        self.browser.show(mode, start, name if mode == "save" else "")

    # --- editing --------------------------------------------------------------------------------

    def edit(self, path: str, value) -> None:
        """A typed box, a switch or a choice changed: set, or unset when emptied."""
        row = self._rows.get(path)
        if row is None or self.document is None:
            return
        steps = json.loads(path)
        try:
            if row.widget == "boolean" and value is None:
                keep, parsed = False, None
            else:
                keep, parsed = case_form.parse_input(row.widget, value)
        except ValueError as error:
            self._say(str(error), ok=False)
            self._lay_out()  # put the box back to what the case holds
            return
        if keep:
            self._apply(case_form.set_value(self.document, steps, parsed, self.schema))
        elif row.is_set:
            self._apply(case_form.unset(self.document, steps))

    def choose_kind(self, path: str, kind: str) -> None:
        """A nested value, an item or an entry was given another kind (or the default)."""
        if self.document is not None:
            self._apply(
                case_form.set_kind(self.document, json.loads(path), kind or "", self.schema)
            )

    def add_item(self, path: str, kind: str) -> None:
        """Append a value of ``kind`` to the list at ``path``."""
        if self.document is not None and kind:
            self._apply(case_form.add_item(self.document, json.loads(path), kind, self.schema))

    def add_entry(self, path: str, name: str) -> None:
        """Add a table entry ``name``, of the table's first kind."""
        row = self._rows.get(path)
        if self.document is None or row is None:
            return
        entry_kinds = case_form.entry_kinds(self.schema, self.document, json.loads(path))
        try:
            edited = case_form.add_entry(
                self.document, json.loads(path), str(name or ""), entry_kinds[0], self.schema
            )
        except (ValueError, IndexError) as error:
            self._say(str(error) or "This table takes no entries.", ok=False)
            return
        self._apply(edited)

    def remove(self, path: str) -> None:
        """Remove a list item or table entry, or return a setting to its default."""
        if self.document is not None:
            self._apply(case_form.remove_at(self.document, json.loads(path)))

    def toggle(self, path: str) -> None:
        """Fold or unfold a nested value's settings."""
        collapsed = list(self.server.state.case_collapsed)
        self.server.state.case_collapsed = (
            [p for p in collapsed if p != path] if path in collapsed else [*collapsed, path]
        )

    def _apply(self, document: dict) -> None:
        if self.document is not None and document.get("mesh") != self.document.get("mesh"):
            self.server.state.mesh_stale = True
        self.document = document
        self.server.state.case_modified = True
        self._lay_out()

    # --- state ----------------------------------------------------------------------------------

    def _show_error(self, error: str | None) -> None:
        state = self.server.state
        state.case_error = error or ""
        match = _AT.search(error or "")
        state.case_error_path = match.group(1) if match else ""

    def _lay_out(self) -> None:
        """Rebuild the form's rows from the document."""
        sections = form_sections(self.schema, self.document)
        self._rows = {}
        bad = self.server.state.case_error_path
        laid = []
        for section in sections:
            rows = []
            for row in section["rows"]:
                state = row.to_state()
                dotted = ".".join(str(step) for step in row.path)
                state["invalid"] = bool(bad) and (bad == dotted or bad.startswith(dotted + "."))
                state["ancestors"] = [
                    json.dumps(list(row.path[:n])) for n in range(1, len(row.path))
                ]
                state["doc"] = row.doc
                rows.append(state)
                self._rows[state["path"]] = row
            laid.append(
                {
                    "key": section["key"],
                    "title": section["title"],
                    "doc": section["doc"],
                    "is_set": section["is_set"],
                    "invalid": any(r["invalid"] for r in rows),
                    "rows": rows,
                }
            )
        self.server.state.case_sections = laid
        self.server.state.case_revision += 1

    def _default_collapsed(self) -> list[dict]:
        """The groups folded when a case is opened: every one below a section's own top level."""
        return [
            row
            for section in self.server.state.case_sections
            for row in section["rows"]
            if row["has_inside"] and row["depth"] >= 1
        ]

    def _say(self, message: str, ok: bool = True) -> None:
        state = self.server.state
        state.case_message, state.case_message_ok, state.case_show_message = message, ok, True

    # --- page -----------------------------------------------------------------------------------

    def drawer(self) -> None:
        """The file's actions, then the open case's form."""
        self.browser.dialog()
        self._confirm_dialog()
        v3.VSnackbar(
            v_model=("case_show_message",),
            text=("case_message",),
            color=("case_message_ok ? 'primary' : 'error'",),
            timeout=6000,
            location="bottom right",
        )
        with html.Div(classes="pa-3 d-flex flex-column", style="gap: 8px;"):
            with html.Div(classes="d-flex", style="gap: 8px;"):
                v3.VBtn(
                    "Open", prepend_icon="mdi-folder-open-outline", color="primary",
                    variant="flat", classes="flex-grow-1", click=(self.browse, "['open']"),
                )  # fmt: skip
                v3.VBtn(
                    "Save", v_if="case_loaded", prepend_icon="mdi-content-save-outline",
                    variant="tonal", color="primary", classes="flex-grow-1",
                    disabled=("!case_modified || case_busy",), click=self.request_save,
                )  # fmt: skip
                v3.VBtn(
                    "Save as", v_if="case_loaded", variant="tonal", classes="flex-grow-1",
                    disabled=("case_busy",), click=(self.browse, "['save']"),
                )  # fmt: skip
                v3.VBtn(
                    "Check", v_if="case_loaded", prepend_icon="mdi-check-decagram-outline",
                    variant="tonal", classes="flex-grow-1", disabled=("case_modified || case_busy",),
                    click=self.check,
                )  # fmt: skip
            with html.Div(
                v_if="case_opening", classes="d-flex align-center text-body-2", style="gap: 8px;"
            ):
                v3.VProgressCircular(indeterminate=True, size=16, width=2, color="primary")
                html.Span("Opening {{ case_opening }}…")
            with html.Div(
                v_if="case_loaded && !case_opening",
                classes="d-flex align-center",
                style="gap: 8px;",
            ):
                v3.VIcon("mdi-file-document-outline", color="primary", size="small")
                with html.Div(classes="flex-grow-1", style="min-width: 0;"):
                    html.Div("{{ case_name }}", classes="text-subtitle-2 text-truncate")
                    html.Div(
                        "{{ case_path }}", classes="text-caption text-medium-emphasis text-truncate",
                        title=("case_path",),
                    )  # fmt: skip
                v3.VProgressCircular(
                    v_if="case_busy", indeterminate=True, size=16, width=2, color="primary"
                )
                v3.VChip(
                    "Modified", v_if="case_modified", size="small", color="warning",
                    variant="tonal",
                )  # fmt: skip
            v3.VAlert(
                v_if="case_error",
                text=("case_error",),
                title="The solver refuses this case",
                type="error",
                variant="tonal",
                density="compact",
            )
            html.Div(
                "Open a case file to see and change its settings. Every choice offered is one the "
                "installed solver reads.",
                v_if="!case_loaded && !case_opening",
                classes="text-body-2 text-medium-emphasis pa-2",
            )
        v3.VDivider(v_if="case_loaded")
        with v3.VExpansionPanels(
            v_if="case_loaded", v_model=("case_open_sections",), multiple=True, flat=True,
            variant="accordion",
        ):  # fmt: skip
            with v3.VExpansionPanel(
                v_for="section in case_sections", key="section.key", value=("section.key",)
            ):
                with v3.VExpansionPanelTitle():
                    v3.VIcon(
                        icon=(
                            "section.invalid ? 'mdi-alert-circle-outline' "
                            ": section_icons[section.key] || 'mdi-cog-outline'",
                        ),
                        color=("section.invalid ? 'error' : 'primary'",),
                        size="small",
                        classes="mr-3",
                    )
                    html.Span("{{ section.title }}", classes="text-subtitle-2")
                    html.Span(
                        "default", v_if="!section.is_set",
                        classes="text-caption text-medium-emphasis ml-3",
                    )  # fmt: skip
                with v3.VExpansionPanelText():
                    html.Div("{{ section.doc }}", classes="text-caption text-medium-emphasis mb-2")
                    self._rows_template()

    def main(self) -> None:
        """The case's mesh, read or generated by the solver; a prompt to open a case until one is."""
        with html.Div(
            v_if="!case_loaded && case_opening",
            classes="d-flex align-center justify-center pa-6",
            style="height: 100%;",
        ):
            with v3.VCard(elevation=1, max_width=560, classes="pa-8 text-center"):
                v3.VProgressCircular(
                    indeterminate=True, size=48, width=4, color="primary", classes="mb-4"
                )
                html.Div("Opening {{ case_opening }}", classes="text-h6 mb-2")
                html.Div(
                    "The solver is reading the case file. Its mesh is read or generated next.",
                    classes="text-body-2 text-medium-emphasis",
                )
        with html.Div(v_if="!case_loaded && !case_opening", style="height: 100%;"):
            empty_state(
                "mdi-file-cog-outline",
                "Case setup",
                "Open a case file to see and change its settings, and its mesh: the mesh, fluid, "
                "physics, boundary patches, numerics, solver and outputs.",
            )
        with html.Div(
            v_show="case_loaded",
            classes="d-flex flex-column pa-3",
            style="height: 100%; gap: 12px;",
        ):
            with v3.VCard(elevation=1, style="flex: none;"):
                with v3.VCardItem(classes="py-2"):
                    with v3.Template(v_slot_prepend=True):
                        v3.VIcon("mdi-vector-triangle", color="primary", size="small")
                    v3.VCardTitle("Mesh", classes="text-subtitle-1")
                    v3.VCardSubtitle("Loading the mesh…", v_if="mesh_busy")
                    v3.VCardSubtitle(
                        "{{ mesh_cells.toLocaleString() }} cells, {{ mesh_dim }}D, "
                        "{{ mesh_patches.length }} boundary patches",
                        v_else_if="mesh_loaded",
                    )
                    with v3.Template(v_slot_append=True):
                        v3.VProgressCircular(
                            v_if="mesh_busy", indeterminate=True, size=18, width=2,
                            color="primary", classes="mr-2",
                        )  # fmt: skip
                        v3.VBtn(
                            "{{ mesh_stale ? 'Mesh settings changed: reload' : 'Reload' }}",
                            prepend_icon="mdi-refresh", size="small",
                            variant=("mesh_stale ? 'flat' : 'text'",),
                            color=("mesh_stale ? 'warning' : undefined",),
                            disabled=("mesh_busy",), click=self.load_mesh,
                        )  # fmt: skip
                with html.Div(
                    v_if="mesh_loaded", classes="px-4 pb-3 d-flex flex-wrap", style="gap: 6px;"
                ):
                    v3.VChip(
                        "{{ patch.name }} · {{ patch.faces }}",
                        v_for="patch in mesh_patches",
                        key="patch.name",
                        size="small",
                        color=("mesh_selected.includes(patch.name) ? 'warning' : undefined",),
                        variant=("mesh_selected.includes(patch.name) ? 'flat' : 'tonal'",),
                        click=(self.select_boundary, "[patch.name]"),
                    )
                html.Div(
                    "Choose a patch here or under Boundaries to highlight it.",
                    v_if="mesh_loaded && mesh_patches.length",
                    classes="px-4 pb-3 text-caption text-medium-emphasis",
                )
                v3.VAlert(
                    v_if="mesh_error",
                    text=("mesh_error",),
                    title="The mesh could not be read",
                    type="error",
                    variant="tonal",
                    density="compact",
                    classes="mx-4 mb-3",
                )
            with v3.VCard(classes="af-view flex-grow-1", elevation=1, style="min-height: 0;"):
                view = vtk_widgets.VtkRemoteView(self.plotter.render_window, interactive_ratio=1)
                self.server.controller.mesh_view_update = view.update

    def _rows_template(self) -> None:
        """One row of the form, drawn by its widget."""
        with html.Div(
            v_for="row in section.rows",
            key="row.id",
            v_show="!row.ancestors.some(a => case_collapsed.includes(a))",
            # `af-row` lays the row out as a flex row in the stylesheet, not with Vuetify's `d-flex`:
            # that utility is `display: flex !important`, which would defeat the `v_show` folding it.
            classes=(
                "['af-row', 'af-depth-' + Math.min(row.depth, 2), "
                "{ 'af-row-invalid': row.invalid, 'af-row-unknown': row.unknown }]",
            ),
            # One guide line per enclosing group, drawn in the row's own background: rows stack
            # without gaps, so each level's lines join into one rail down the group.
            style=(_RAIL_STYLE,),
        ):
            # Fold and unfold a nested value.
            v3.VBtn(
                v_if="row.has_inside",
                icon=("case_collapsed.includes(row.path) ? 'mdi-chevron-right' : 'mdi-chevron-down'",),
                size="x-small", variant="text", density="comfortable",
                click=(self.toggle, "[row.path]"),
            )  # fmt: skip
            html.Div(v_else=True, style="width: 28px; flex: none;")
            with html.Div(classes="af-row-label"):
                html.Span(
                    v_if=(
                        "['kind', 'item', 'entry'].includes(row.widget) && row.set_inside "
                        "&& case_collapsed.includes(row.path)"
                    ),
                    classes="af-set-dot",
                    title=("row.set_inside + ' set inside'",),
                )
                html.Span(
                    "{{ row.label }}",
                    classes=(
                        "[row.is_set ? 'font-weight-medium' : 'text-medium-emphasis', "
                        "{ 'af-patch': row.widget === 'entry' && mesh_names.includes(row.label), "
                        "'af-patch-selected': row.widget === 'entry' && mesh_selected_key === row.label }]",
                    ),
                    click=(self._label_clicked, "[row.widget, row.label]"),
                )
                with v3.VTooltip(v_if="row.doc", text=("row.doc",), location="top", max_width=420):
                    with v3.Template(v_slot_activator=("{ props }",)):
                        v3.VIcon(
                            "mdi-information-outline", v_bind="props", size="x-small",
                            classes="ml-1 text-medium-emphasis",
                        )  # fmt: skip
                # Under a group's name: what is set inside it while it is folded, else what it is.
                html.Div(
                    "{{ " + _GROUP_SUBTITLE + " }}",
                    v_if="['kind', 'item', 'entry'].includes(row.widget) && ("
                    + _GROUP_SUBTITLE
                    + ")",
                    title=(_GROUP_SUBTITLE,),
                    classes="af-row-sub",
                )
            with html.Div(classes="af-row-input"):
                v3.VSelect(
                    v_if="['kind', 'entry', 'choice'].includes(row.widget)",
                    model_value=("row.value",),
                    items=("row.items",),
                    placeholder=("row.placeholder",),
                    hide_details=True,
                    update_modelValue=(self._select, "[row.path, row.widget, $event]"),
                )
                v3.VSelect(
                    v_else_if="row.widget === 'choices'",
                    model_value=("row.value",),
                    items=("row.items",),
                    placeholder=("row.placeholder",),
                    multiple=True,
                    chips=True,
                    closable_chips=True,
                    hide_details=True,
                    update_modelValue=(self.edit, "[row.path, $event]"),
                )
                v3.VSelect(
                    v_else_if="row.widget === 'boolean'",
                    model_value=("row.is_set ? row.value : null",),
                    items=("row.items",),
                    placeholder=("row.placeholder",),
                    hide_details=True,
                    update_modelValue=(self.edit, "[row.path, $event]"),
                )
                v3.VTextField(
                    v_else_if=f"{list(_TYPED)}.includes(row.widget)",
                    model_value=("row.value",),
                    placeholder=("row.placeholder",),
                    hide_details=True,
                    keyup_enter=(self.edit, "[row.path, $event.target.value]"),
                    blur=(self.edit, "[row.path, $event.target.value]"),
                    __events=[("keyup_enter", "keyup.enter"), "blur"],
                    classes=("row.widget === 'raw' ? 'af-raw' : ''",),
                )
                v3.VSelect(
                    v_else_if="row.widget === 'list' && row.items.length",
                    model_value=None,
                    items=("row.items",),
                    placeholder="Add…",
                    prepend_inner_icon="mdi-plus",
                    hide_details=True,
                    update_modelValue=(self.add_item, "[row.path, $event]"),
                )
                # Keyed by the form's revision, so the box is a fresh, empty one once an entry is added.
                v3.VTextField(
                    v_else_if="row.widget === 'table'",
                    key=("'add-' + row.id + '-' + case_revision",),
                    placeholder="New entry name, then Enter",
                    prepend_inner_icon="mdi-plus",
                    hide_details=True,
                    keyup_enter=(self.add_entry, "[row.path, $event.target.value]"),
                    __events=[("keyup_enter", "keyup.enter")],
                )
                html.Span(
                    "{{ row.value }}",
                    v_else_if="row.widget === 'item'",
                    classes="text-body-2 text-medium-emphasis",
                )
            with html.Div(style="width: 32px; flex: none;"):
                with html.Div(v_if="row.removable"):
                    icon_button(
                        (
                            "['item', 'entry'].includes(row.widget) || row.unknown ? 'mdi-close' : 'mdi-backspace-outline'",
                        ),
                        (
                            "['item', 'entry'].includes(row.widget) || row.unknown ? 'Remove' : 'Back to the default'",
                        ),
                        (self.remove, "[row.path]"),
                    )

    def _select(self, path: str, widget: str, value) -> None:
        """A dropdown changed: a kind for a nested value or an entry, or a choice's value."""
        if widget == "choice":
            self.edit(path, value)
        else:
            self.choose_kind(path, value)

    def _confirm_dialog(self) -> None:
        with v3.VDialog(v_model=("case_confirm",), max_width=480):
            with v3.VCard(rounded="lg"):
                v3.VCardTitle("Save over {{ case_name }}?", classes="text-subtitle-1 pt-4")
                v3.VCardText(
                    "The file is rewritten from its settings: comments and hand formatting in it are "
                    "not kept, and a setting written out at its default value is dropped. Save as a "
                    "new file instead to keep the original."
                )
                with v3.VCardActions(classes="px-4 pb-4"):
                    v3.VSpacer()
                    v3.VBtn("Cancel", variant="text", click="case_confirm = false")
                    v3.VBtn("Save as…", variant="text", click=(self._save_as_instead,))
                    v3.VBtn("Save", color="primary", variant="flat", click=self.confirm_save)

    def _save_as_instead(self) -> None:
        self.server.state.case_confirm = False
        self.browse("save")

    def toolbar(self) -> None:
        """Reframe the mesh, once one is shown."""
        with html.Div(v_if="mesh_loaded"):
            icon_button("mdi-fit-to-screen-outline", "Reset camera", self.reset_camera)

    def reset_camera(self) -> None:
        """Frame the whole mesh, with a margin."""
        self.mesh_view.frame()
        self.server.controller.mesh_view_update()

    def shown(self) -> None:
        """Send the mesh view a fresh frame of its current size."""
        self.server.controller.mesh_view_update()
