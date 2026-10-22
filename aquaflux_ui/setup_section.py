"""The Setup section: open a case file, change its settings, save it, and check it against its mesh.

The case is edited through a :class:`~aquaflux_ui.document_form.DocumentForm` -- its form built from
the case-file schema the installed solver prints, opened and saved through the solver's own commands
-- and its mesh, read by the solver, is drawn beside it, its patches highlighted from the form.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Mapping
from pathlib import Path

import pyvista as pv
from trame.widgets import html
from trame.widgets import vtk as vtk_widgets
from trame.widgets import vuetify3 as v3

from .document_form import DocumentForm
from .mesh_view import MeshView, patches_addressed
from .solver_commands import CASE_FILE, SolverCommands
from .theme import LIGHT, THEMES
from .widgets import empty_state, icon_button

__all__ = ["SECTION_ICONS", "SetupSection"]

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

#: A row's name, marked as a patch it can highlight when it is a table entry naming one, as a page
#: expression given ``row``.
_PATCH_LABEL = (
    "{ 'af-patch': row.widget === 'entry' && mesh_names.includes(row.label), "
    "'af-patch-selected': row.widget === 'entry' && mesh_selected_key === row.label }"
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
        self.form = DocumentForm(
            server, "case", self.commands, CASE_FILE,
            icons=SECTION_ICONS, default_name="case.yaml", opened=self.load_mesh,
            changed=self._changed, label_clicked=self._label_clicked,
            label_classes=_PATCH_LABEL,
        )  # fmt: skip
        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.enable_anti_aliasing("fxaa")
        self.mesh_view = MeshView(self.plotter, THEMES[LIGHT])
        self._mesh_files = tempfile.TemporaryDirectory(prefix="aquaflux-ui-mesh-")
        self._mesh_loads = 0
        self._groups: dict[str, list[str]] = {}
        server.state.update(
            {
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
            }
        )
        server.state.change("theme")(self._on_theme)
        if case is not None:
            server.controller.on_server_ready.add(lambda **_: self.form.spawn(self.form.open(case)))

    def _changed(self, before: Mapping | None, after: Mapping) -> None:
        """An edit under ``mesh`` leaves the mesh drawn out of date."""
        if before is not None and after.get("mesh") != before.get("mesh"):
            self.server.state.mesh_stale = True

    def use_mesh(self, section: Mapping) -> str | None:
        """Make ``section`` the open case's mesh, as an edit; returns why it could not be, or ``None``."""
        if self.form.document is None:
            return "Open or create a case in Setup first."
        self.form.apply({**self.form.document, "mesh": dict(section)})
        return None

    async def load_mesh(self) -> None:
        """Read the case's mesh, as its mesh section stands, and draw it."""
        document, directory = self.form.document, self.form.directory
        if document is None or directory is None:
            return
        section = document.get("mesh")
        if not isinstance(section, dict):
            with self.server.state as state:
                state.mesh_error, state.mesh_loaded = "The case names no mesh.", False
            return
        self._mesh_loads += 1
        written = Path(self._mesh_files.name) / str(self._mesh_loads)
        with self.server.state as state:
            state.mesh_busy, state.mesh_error = True, ""
        try:
            export = await asyncio.to_thread(self.commands.mesh, section, directory, written)
            if export.error is None:
                await asyncio.to_thread(self.mesh_view.load, written, export.dim)
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

    async def check(self) -> None:
        """Check the saved file against its mesh."""
        if self.form.path is None:
            return
        ok, message = await self.form.ask(self.commands.check, self.form.path)
        with self.server.state:
            self.form.say(message or ("Checked." if ok else "The check failed."), ok=ok)

    # --- page -----------------------------------------------------------------------------------

    def drawer(self) -> None:
        """The file's actions, then the open case's form."""
        self.form.drawer(
            buttons=self._check_button,
            empty="Open a case file to see and change its settings. Every choice offered is one "
            "the installed solver reads.",
        )
        self.form.form()

    def _check_button(self) -> None:
        v3.VBtn(
            "Check", prepend_icon="mdi-check-decagram-outline", variant="tonal",
            classes="flex-grow-1", disabled=("case_modified || case_busy",), click=self.check,
        )  # fmt: skip

    def main(self) -> None:
        """The case's mesh, read by the solver; a prompt to open a case until one is."""
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
                    "The solver is reading the case file. Its mesh is read next.",
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
