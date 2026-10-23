"""The Mesh section: open or start a mesh file, choose its patches on the drawing, make the mesh, and
use it in the case.

A mesh file says how to mesh a region of a STEP drawing. Its form is a
:class:`~aquaflux_ui.document_form.DocumentForm` built from the mesh-file schema the solver prints,
like Setup's case form. Beside it the drawing is drawn as the solver reads it (``aquaflux drawing``):
the region's faces, each in its patch's colour. Choosing a patch in the list and clicking a face adds
the face to it, or takes it away again (:mod:`~aquaflux_ui.patch_picking`); every click is an edit of
the file, and the solver is asked again which patch each face is in.

Making the mesh runs ``aquaflux mesh`` on the saved file in a process of its own, its stages shown as
they start; the mesh made is then drawn with the quality its file records, and can be made the open
case's mesh in Setup.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import pyvista as pv
from trame.widgets import html
from trame.widgets import vtk as vtk_widgets
from trame.widgets import vuetify3 as v3

from . import case_form
from .case_form import ANY
from .document_form import DocumentForm
from .drawing_scene import DrawingScene
from .file_browser import FileBrowser
from .mesh_view import MeshView
from .patch_picking import face_colours, patch_rows, picked
from .run_process import SolverRun
from .solver_commands import MESH_FILE, DrawingView, MeshExport, SolverCommands
from .theme import LIGHT, THEMES
from .widgets import empty_state, icon_button

__all__ = ["DRAWING_SUFFIXES", "MESH_SECTION_ICONS", "POLL_SECONDS", "MeshSection", "quality_tiles"]

#: The icon a mesh-file section is headed by, by the section's name; one not named gets a cog.
MESH_SECTION_ICONS = {
    "drawing": "mdi-file-cad",
    "region": "mdi-cube-outline",
    "sizes": "mdi-ruler",
    "patches": "mdi-border-outside",
    "others": "mdi-border-none-variant",
    "layers": "mdi-layers-outline",
    "layered": "mdi-layers-outline",
    "mapped": "mdi-grid",
    "pipes": "mdi-pipe",
    "pipe_core": "mdi-pipe",
    "pipe_grading": "mdi-pipe",
    "planarizer": "mdi-vector-square",
    "core": "mdi-cube-outline",
    "periodic": "mdi-sync",
    "output": "mdi-folder-outline",
}

#: The files a drawing is chosen from.
DRAWING_SUFFIXES = (".step", ".stp")

#: Seconds between reads of a meshing's console while it runs.
POLL_SECONDS = 1.0

#: The settings that change what the drawing shows: when one is edited, the solver is asked again.
_DRAWN = ("drawing", "region", "patches", "others")


def quality_tiles(made: Mapping | None) -> list[dict]:
    """The tiles a mesh's quality is shown in, from what its file records of how it was made.

    Parameters
    ----------
    made : mapping or None
        The ``.aqmesh`` header's provenance, as ``aquaflux export-mesh`` prints it.

    Returns
    -------
    list of dict
        ``{"label", "value"}`` per tile; empty when the file records no quality.
    """
    quality = (made or {}).get("quality")
    if not quality:
        return []
    tiles = [
        ("Cells", f"{quality['cells']:,}"),
        (
            "Non-orthogonality p99 / max",
            f"{quality['non_orthogonality_p99']:.0f}° / {quality['non_orthogonality_max']:.0f}°",
        ),
        ("Volume ratio p99", f"{quality['volume_ratio_p99']:.1f}"),
        ("Concave cells", f"{quality['concave_cells']:,}"),
    ]
    if made.get("seconds") is not None:
        tiles.append(("Meshed in", f"{made['seconds']:.1f} s"))
    return [{"label": label, "value": value} for label, value in tiles]


class MeshSection:
    """The Mesh section.

    Parameters
    ----------
    server : trame server
        The workspace's server.
    commands : SolverCommands
        How the solver is asked.
    setup : SetupSection
        The Setup section, whose open case a mesh made here is given to.
    mesh_file : path-like, optional
        A mesh file to open when the page is first shown.
    """

    key, title, icon = "mesh", "Mesh", "mdi-cube-scan"

    def __init__(self, server, commands: SolverCommands, setup, mesh_file=None) -> None:
        self.server = server
        self.commands = commands
        self.setup = setup
        self.form = DocumentForm(
            server, "meshfile", commands, MESH_FILE,
            icons=MESH_SECTION_ICONS, default_name="mesh.yaml", opened=self._opened,
            changed=self._changed, names=self._names,
        )  # fmt: skip
        self.drawings = FileBrowser(
            server, "meshing_drawings", DRAWING_SUFFIXES, self._drawing_chosen, what="drawing"
        )
        self.view = DrawingView(None)
        self.run: SolverRun | None = None
        self._files = tempfile.TemporaryDirectory(prefix="aquaflux-ui-meshing-")
        self._console = Path(self._files.name) / "console.txt"
        self._loads = 0
        self._loading: asyncio.Task | None = None
        self._again = False
        self._pending: list[str] = []
        self._hidden: set[int] = set()
        self.drawing_plotter = pv.Plotter(off_screen=True)
        self.drawing_plotter.enable_anti_aliasing("fxaa")
        self.drawing = DrawingScene(self.drawing_plotter, THEMES[LIGHT])
        self.mesh_plotter = pv.Plotter(off_screen=True)
        self.mesh_plotter.enable_anti_aliasing("fxaa")
        self.mesh_view = MeshView(self.mesh_plotter, THEMES[LIGHT])
        server.state.update(
            {
                "meshing_view": "drawing",
                "meshing_drawing_busy": False,
                "meshing_drawing_error": "",
                "meshing_selections_error": "",
                "meshing_has_region": False,
                "meshing_patches": [],
                "meshing_chosen": "",
                "meshing_click_mode": "assign",
                "meshing_hidden": 0,
                "meshing_new_patch": "",
                "meshing_running": False,
                "meshing_state": "idle",
                "meshing_state_text": "",
                "meshing_log": "",
                "meshing_overwrite": False,
                "meshing_tiles": [],
                "meshing_result": False,
                "meshing_result_error": "",
                "meshing_result_path": "",
            }
        )
        server.state.change("theme")(self._on_theme)
        atexit.register(self._stop_quietly)  # closing the page stops a meshing under way
        if mesh_file is not None:
            server.controller.on_server_ready.add(
                lambda **_: self.form.spawn(self.form.open(mesh_file))
            )

    # --- the file -------------------------------------------------------------------------------

    def start_new(self) -> None:
        """Start a new mesh file: first choose its drawing."""
        self.drawings.show("open", self.form.directory, "")

    def _drawing_chosen(self, path: str, mode: str) -> str | None:
        del mode
        step = Path(path)
        self.form.spawn(self.form.new({"drawing": {"file": step.name}}, directory=step.parent))
        return None

    async def _opened(self) -> None:
        """A mesh file was opened or started: draw its drawing, and the mesh it made, if any."""
        self._pending, self._hidden = [], set()
        with self.server.state as state:
            state.meshing_chosen, state.meshing_tiles = "", []
            state.meshing_click_mode, state.meshing_hidden = "assign", 0
            state.meshing_result, state.meshing_result_error = False, ""
            state.meshing_state, state.meshing_state_text, state.meshing_log = "idle", "", ""
        await self.load_drawing(frame=True)
        made = self.mesh_path()
        if made is not None and made.is_file():
            await self.load_result()

    def _changed(self, before: Mapping | None, after: Mapping) -> None:
        """An edit of what the drawing shows asks the solver again."""
        if before is None or any(before.get(key) != after.get(key) for key in _DRAWN):
            self.form.spawn(self.load_drawing())

    def _names(self, document: Mapping) -> dict[tuple, list[str]]:
        """The names a list in the form is chosen from: the drawing's solids, the file's patches."""
        del document
        solids = list(self.view.solids)
        patches = [row["name"] for row in self.server.state.meshing_patches]
        return {
            ("region", "solids"): solids,
            ("region", "without"): solids,
            ("patches", ANY, "solids"): solids,
            ("layered",): patches,
            ("mapped",): patches,
            ("pipes",): self._pipe_patches(),
        }

    def _pipe_patches(self) -> list[str]:
        """The patches whose every face is a pipe."""
        names = self.view.patches or []
        pipes = set(self.view.pipes)
        return [
            row["name"]
            for row in self.server.state.meshing_patches
            if (faces := [f for f, name in enumerate(names) if name == row["name"]])
            and set(faces) <= pipes
        ]

    # --- the drawing ------------------------------------------------------------------------------

    async def load_drawing(self, frame: bool = False) -> None:
        """Ask the solver for the drawing as the file stands, and draw it; one request at a time."""
        if self._loading is not None and not self._loading.done():
            self._again = True  # the file changed while the solver was asked: ask once more after
            return
        self._loading = asyncio.ensure_future(self._load_drawing(frame))
        await self._loading
        while self._again:
            self._again = False
            self._loading = asyncio.ensure_future(self._load_drawing(False))
            await self._loading

    async def _load_drawing(self, frame: bool) -> None:
        document, directory = self.form.document, self.form.directory
        if document is None or directory is None:
            return
        self._loads += 1
        written = Path(self._files.name) / f"drawing-{self._loads}"
        with self.server.state as state:
            state.meshing_drawing_busy = True
        try:
            view = await asyncio.to_thread(self.commands.drawing, document, directory, written)
            if view.error is None and view.surface is not None:
                await asyncio.to_thread(self.drawing.load, view.surface)
        finally:
            with self.server.state as state:
                state.meshing_drawing_busy = False
        with self.server.state as state:
            if view.error is not None:
                state.meshing_drawing_error = view.error
                return
            first = not self.view.faces
            self.view = view
            state.meshing_drawing_error = ""
            state.meshing_selections_error = view.selections_error or ""
            state.meshing_has_region = view.surface is not None
            self._colour()
            if view.surface is None:
                self.drawing.clear()
            if frame or first:
                self.drawing.frame()
            self.form.lay_out()  # the form's lists of names follow the drawing
        self.server.controller.meshing_drawing_update()

    def _colour(self) -> None:
        """Lay out the patch list and colour the faces by it."""
        if self.form.document is None:
            return
        state = self.server.state
        style = self.drawing.style
        rows = patch_rows(
            self.form.document or {}, self.view, style.categories, style.context, self._others(),
            self._pending,
        )  # fmt: skip
        state.meshing_patches = [row.to_state() for row in rows]
        if self.view.surface is not None:
            self.drawing.show(face_colours(rows, self.view), self._hidden)
        state.meshing_hidden = len(self._hidden)

    def _others(self) -> str:
        """The patch the faces no selection chooses go to, as the file sets it or by default."""
        name = case_form.setting(self.form.schema, self.form.document or {}, ["others"])
        return str(name) if name else "walls"

    # --- picking --------------------------------------------------------------------------------

    def choose(self, name: str) -> None:
        """Make ``name`` the patch a click on a face adds to; again to choose none."""
        state = self.server.state
        state.meshing_chosen = "" if state.meshing_chosen == name else name

    def add_patch(self, name: str) -> None:
        """Start a patch called ``name``, chosen, so the next face clicked is its first."""
        name = (name or "").strip()
        state = self.server.state
        taken = {row["name"] for row in state.meshing_patches}
        if not name:
            return
        if name in taken:
            self.form.say(f"There is a patch called {name!r} already.", ok=False)
            return
        self._pending.append(name)
        state.meshing_new_patch, state.meshing_chosen = "", name
        self._colour()
        self.server.controller.meshing_drawing_update()

    def remove_patch(self, name: str) -> None:
        """Take patch ``name`` out of the file: its faces go back to the others patch."""
        state = self.server.state
        if name in self._pending:
            self._pending.remove(name)
            self._colour()
        document = self.form.document
        if document is not None and name in (document.get("patches") or {}):
            patches = {key: value for key, value in document["patches"].items() if key != name}
            edited = {**document, "patches": patches}
            if not patches:
                edited.pop("patches")
            self.form.apply(edited)
        if state.meshing_chosen == name:
            state.meshing_chosen = ""

    def show_all(self) -> None:
        """Draw every face again."""
        self._hidden.clear()
        self._colour()
        self.server.controller.meshing_drawing_update()

    def clicked(self, event: Mapping) -> None:
        """A click in the drawing: the face under it joins or leaves the chosen patch, or, hiding,
        is hidden.

        A click that ends a drag -- the camera moved since the view was last still -- picks nothing.
        """
        state = self.server.state
        if self.drawing.dragged():
            return
        position = event.get("position") or {}
        x, y = float(position.get("x", -1)), float(position.get("y", -1))
        if state.meshing_click_mode == "hide":
            face = self.drawing.pick(x, y)
            if face is not None:
                self._hidden.add(face)
                self._colour()
                self.server.controller.meshing_drawing_update()
            return
        if not state.meshing_chosen:
            self.form.say("Choose a patch in the list first, then click its faces.", ok=False)
            return
        if self._loading is not None and not self._loading.done():
            self.form.say("The drawing is being read again; click once it is shown.", ok=False)
            return
        face = self.drawing.pick(x, y)
        if face is None or self.form.document is None:
            return
        chosen = state.meshing_chosen
        pick = picked(self.form.document, self.view, face, chosen, self._others())
        if pick.message:
            self.form.say(pick.message, ok=False)
        if pick.document != self.form.document:
            in_file = chosen in (pick.document.get("patches") or {})
            if in_file and chosen in self._pending:
                self._pending.remove(chosen)
            elif not in_file and chosen != self._others() and chosen not in self._pending:
                self._pending.append(chosen)  # its last face left: kept, to be given another
            self.form.apply(pick.document)

    # --- making the mesh --------------------------------------------------------------------------

    def mesh_path(self) -> Path | None:
        """The ``.aqmesh`` file the open mesh file writes: its output, else the default."""
        directory = self.form.directory
        if self.form.document is None or directory is None:
            return None
        path = case_form.setting(self.form.schema, self.form.document, ["output", "path"])
        return (directory / str(path or "mesh.aqmesh")).resolve()

    def make(self) -> None:
        """Make the saved mesh file's mesh, in a process of its own."""
        state = self.server.state
        if self.form.path is None or state.meshfile_modified or self.running:
            return
        self.run = SolverRun(
            "mesh", self.form.path, self._console, overwrite=bool(state.meshing_overwrite)
        )
        self.run.start()
        state.update(
            {
                "meshing_running": True,
                "meshing_state": "running",
                "meshing_state_text": "Meshing",
                "meshing_log": "",
                "meshing_view": "mesh",
                # Permission to replace a mesh is for this meshing; the next one is asked again.
                "meshing_overwrite": False,
            }
        )
        self.form.spawn(self._follow())

    @property
    def running(self) -> bool:
        """Whether a meshing is under way."""
        return self.run is not None and self.run.running

    def stop(self) -> None:
        """Stop the meshing under way."""
        if self.running:
            self.server.state.meshing_state_text = "Stopping"
            self.form.spawn(asyncio.to_thread(self.run.stop, 5.0))

    def _stop_quietly(self) -> None:
        if self.run is not None:
            with contextlib.suppress(Exception):
                self.run.stop(grace=3.0)

    async def _follow(self) -> None:
        """Show the meshing's stages as it prints them, then the mesh it made."""
        run = self.run
        while run.running:
            with self.server.state as state:
                state.meshing_log = run.console_tail()
            await asyncio.sleep(POLL_SECONDS)
        with self.server.state as state:
            state.meshing_log = run.console_tail()
            state.meshing_running = False
            if run.stopped:
                state.meshing_state, state.meshing_state_text = "stopped", "Stopped"
            elif run.status == 0:
                state.meshing_state, state.meshing_state_text = "done", "Meshed"
            elif "already there" in state.meshing_log:
                state.meshing_state = "failed"
                state.meshing_state_text = "A mesh is already there: turn on Replace to remake it"
            else:
                state.meshing_state, state.meshing_state_text = "failed", "Failed: see the log"
        if run.status == 0 and not run.stopped:
            await self.load_result()

    async def load_result(self) -> None:
        """Read the mesh the file made, as a case would, and draw it with its quality."""
        path, directory = self.mesh_path(), self.form.directory
        if path is None or directory is None:
            return
        self._loads += 1
        written = Path(self._files.name) / f"mesh-{self._loads}"
        section = {"kind": "AquafluxMesh", "path": str(path)}
        export: MeshExport = await asyncio.to_thread(
            self.commands.mesh, section, directory, written
        )
        if export.error is None:
            await asyncio.to_thread(self.mesh_view.load, written, export.dim)
        with self.server.state as state:
            state.meshing_result_error = export.error or ""
            state.meshing_result = export.error is None
            state.meshing_result_path = str(path)
            state.meshing_tiles = quality_tiles(export.made) if export.error is None else []
        self.server.controller.meshing_mesh_update()

    def use_in_case(self) -> None:
        """Make the mesh made the open case's mesh in Setup, by a path from the case's directory."""
        path = self.mesh_path()
        case_directory = self.setup.form.directory
        if path is None or not path.is_file():
            self.form.say("Make the mesh first.", ok=False)
            return
        relative = str(path)
        if case_directory is not None:
            with contextlib.suppress(ValueError):
                relative = os.path.relpath(path, case_directory)
        refusal = self.setup.use_mesh({"kind": "AquafluxMesh", "path": relative})
        if refusal:
            self.form.say(refusal, ok=False)
            return
        self.server.state.section = "setup"
        self.setup.form.say(f"The case now reads {relative}. Save it to keep the change.")
        self.setup.form.spawn(self.setup.load_mesh())

    def _on_theme(self, theme, **_) -> None:
        style = THEMES[theme if theme in THEMES else LIGHT]
        self.drawing.style = self.mesh_view.style = style
        self._colour()
        self.mesh_view.show(())
        self.server.controller.meshing_drawing_update()
        self.server.controller.meshing_mesh_update()

    # --- page -----------------------------------------------------------------------------------

    def drawer(self) -> None:
        """The file's actions, the patches, making the mesh, then the file's settings."""
        self.drawings.dialog()
        self.form.drawer(
            new=self.start_new,
            empty="Open a mesh file, or start a new one from a STEP drawing. Every choice offered "
            "is one the installed mesher reads.",
        )
        with html.Div(
            v_if="meshfile_loaded", classes="px-3 pb-3 d-flex flex-column", style="gap: 8px;"
        ):
            self._patch_panel()
            self._make_panel()
        self.form.form()

    def _patch_panel(self) -> None:
        with v3.VCard(variant="tonal", classes="pa-2"):
            html.Div("Patches", classes="text-subtitle-2 px-1")
            html.Div(
                "Choose a patch, then click its faces in the drawing; click a face again to take "
                "it away.",
                classes="text-caption text-medium-emphasis px-1 mb-1",
            )
            with html.Div(
                v_for="patch in meshing_patches",
                key="patch.name",
                classes=(
                    "['d-flex', 'align-center', 'px-1', 'rounded', "
                    "{ 'bg-primary': meshing_chosen === patch.name }]",
                ),
                style="gap: 6px; min-height: 30px; cursor: pointer;",
                click=(self.choose, "[patch.name]"),
            ):
                html.Span(
                    style=("{ width: '12px', height: '12px', borderRadius: '3px', "
                           "background: patch.colour, flex: 'none' }",),
                )  # fmt: skip
                html.Span("{{ patch.name }}", classes="text-body-2 flex-grow-1 text-truncate")
                html.Span(
                    "{{ patch.chosen_by === 'new' ? 'click a face' : patch.faces + "
                    "(patch.faces === 1 ? ' face' : ' faces') + ' · ' + patch.chosen_by }}",
                    classes="text-caption",
                )
                with html.Div(v_if="patch.chosen_by !== 'others'"):
                    icon_button(
                        "mdi-close", "Remove the patch", (self.remove_patch, "[patch.name]")
                    )
            v3.VTextField(
                v_model=("meshing_new_patch",),
                placeholder="New patch name, then Enter",
                prepend_inner_icon="mdi-plus",
                hide_details=True,
                classes="mt-1",
                keyup_enter=(self.add_patch, "[$event.target.value]"),
                __events=[("keyup_enter", "keyup.enter")],
            )
            v3.VAlert(
                v_if="meshing_selections_error",
                text=("meshing_selections_error",),
                type="warning",
                variant="tonal",
                density="compact",
                classes="mt-2",
            )

    def _make_panel(self) -> None:
        with v3.VCard(variant="tonal", classes="pa-2 d-flex flex-column", style="gap: 6px;"):
            with html.Div(classes="d-flex", style="gap: 8px;"):
                v3.VBtn(
                    "Make mesh", prepend_icon="mdi-play", color="primary", variant="flat",
                    classes="flex-grow-1", v_if="!meshing_running",
                    disabled=("!meshfile_path || meshfile_modified",), click=self.make,
                )  # fmt: skip
                v3.VBtn(
                    "Stop", prepend_icon="mdi-stop", color="error", variant="tonal",
                    classes="flex-grow-1", v_if="meshing_running", click=self.stop,
                )  # fmt: skip
                v3.VBtn(
                    "Use in the case", prepend_icon="mdi-arrow-right", variant="tonal",
                    classes="flex-grow-1", disabled=("!meshing_result || meshing_running",),
                    click=self.use_in_case,
                )  # fmt: skip
            v3.VSwitch(
                v_model=("meshing_overwrite",), label="Replace the mesh made before",
                density="compact", hide_details=True, color="primary",
            )  # fmt: skip
            html.Div(
                "Save the mesh file to make its mesh.",
                v_if="!meshfile_path || meshfile_modified",
                classes="text-caption text-medium-emphasis",
            )
            html.Div("{{ meshing_state_text }}", v_if="meshing_state_text", classes="text-body-2")

    def main(self) -> None:
        """The drawing to pick faces on, or the mesh made with its quality."""
        with html.Div(v_if="!meshfile_loaded", style="height: 100%;"):
            empty_state(
                self.icon,
                "Meshing",
                "Open a mesh file, or start one from a STEP drawing: choose the region to mesh, "
                "name its patches by clicking the drawing's faces, set the sizes and layers, and "
                "make the mesh.",
            )
        with html.Div(
            v_show="meshfile_loaded",
            classes="pa-3",
            style="display: flex; flex-direction: column; height: 100%; gap: 12px;",
        ):
            with html.Div(classes="d-flex align-center", style="gap: 8px; flex: none;"):
                with v3.VBtnToggle(
                    v_model=("meshing_view",), mandatory=True, density="compact", color="primary",
                    variant="outlined",
                ):  # fmt: skip
                    v3.VBtn("Drawing", value="drawing", prepend_icon="mdi-file-cad")
                    v3.VBtn("Mesh", value="mesh", prepend_icon="mdi-cube-scan")
                v3.VProgressCircular(
                    v_if="meshing_drawing_busy", indeterminate=True, size=18, width=2,
                    color="primary",
                )  # fmt: skip
                with v3.VBtnToggle(
                    v_if="meshing_view === 'drawing'", v_model=("meshing_click_mode",),
                    mandatory=True, density="compact", variant="outlined",
                ):  # fmt: skip
                    v3.VBtn("Assign faces", value="assign", prepend_icon="mdi-cursor-default-click")
                    v3.VBtn("Hide faces", value="hide", prepend_icon="mdi-eye-off-outline")
                v3.VBtn(
                    "Show all ({{ meshing_hidden }} hidden)",
                    v_if="meshing_view === 'drawing' && meshing_hidden", variant="text",
                    density="compact", click=self.show_all,
                )  # fmt: skip
                html.Span(
                    "{{ meshing_click_mode === 'hide' ? 'Click a face to hide it, to reach the "
                    "faces behind.' : 'Click a face to add it to ' + meshing_chosen + ', or to take "
                    "it away.' }}",
                    v_if="meshing_view === 'drawing' && (meshing_chosen || meshing_click_mode === 'hide')",
                    classes="text-body-2 text-medium-emphasis",
                )
            v3.VAlert(
                v_if="meshing_view === 'drawing' && meshing_drawing_error",
                text=("meshing_drawing_error",),
                title="The drawing could not be read",
                type="error", variant="tonal", density="compact", style="flex: none;",
            )  # fmt: skip
            v3.VAlert(
                v_if="meshing_view === 'drawing' && !meshing_drawing_error && !meshing_has_region",
                text="Choose the solids of the region to mesh under Region, in the side panel.",
                type="info", variant="tonal", density="compact", style="flex: none;",
            )  # fmt: skip
            with v3.VCard(
                v_show="meshing_view === 'drawing'", classes="af-view flex-grow-1", elevation=1,
                style="min-height: 0;",
            ):  # fmt: skip
                view = vtk_widgets.VtkRemoteView(
                    self.drawing_plotter.render_window, interactive_ratio=1,
                    picking_modes=("['click']",), click=(self.clicked, "[$event]"),
                )  # fmt: skip
                self.server.controller.meshing_drawing_update = view.update
            # Flex by an inline style, not `d-flex`: that utility is `display: flex !important`,
            # which would defeat the `v_show` hiding it.
            with html.Div(
                v_show="meshing_view === 'mesh'",
                style="display: flex; flex-direction: column; flex: 1 1 0; gap: 12px; min-height: 0;",
            ):
                self._result_panel()

    def _result_panel(self) -> None:
        with html.Div(
            v_if="meshing_tiles.length", classes="d-flex flex-wrap", style="gap: 8px; flex: none;"
        ):
            with v3.VCard(v_for="tile in meshing_tiles", key="tile.label", classes="pa-2 px-3"):
                html.Div("{{ tile.label }}", classes="text-caption text-medium-emphasis")
                html.Div("{{ tile.value }}", classes="text-h6")
        with v3.VCard(
            v_if="meshing_running || meshing_log", classes="pa-2", style="flex: none;",
        ):  # fmt: skip
            html.Div("{{ meshing_state_text || 'Meshing' }}", classes="text-subtitle-2 mb-1")
            html.Pre(
                "{{ meshing_log }}",
                classes="text-caption",
                style="max-height: 180px; overflow: auto; white-space: pre-wrap; margin: 0;",
            )
        v3.VAlert(
            v_if="meshing_result_error",
            text=("meshing_result_error",),
            title="The mesh could not be read",
            type="error", variant="tonal", density="compact", style="flex: none;",
        )  # fmt: skip
        html.Div(
            "No mesh made yet: save the file, then Make mesh.",
            v_if="!meshing_result && !meshing_running && !meshing_log",
            classes="text-body-2 text-medium-emphasis pa-2",
        )
        with v3.VCard(
            v_show="meshing_result", classes="af-view flex-grow-1", elevation=1,
            style="min-height: 0;",
        ):  # fmt: skip
            view = vtk_widgets.VtkRemoteView(self.mesh_plotter.render_window, interactive_ratio=1)
            self.server.controller.meshing_mesh_update = view.update

    def toolbar(self) -> None:
        """Reframe the view shown."""
        with html.Div(v_if="meshfile_loaded"):
            icon_button("mdi-fit-to-screen-outline", "Reset camera", self.reset_camera)

    def reset_camera(self) -> None:
        """Frame the drawing, or the mesh."""
        if self.server.state.meshing_view == "mesh":
            self.mesh_view.frame()
            self.server.controller.meshing_mesh_update()
        else:
            self.drawing.frame()
            self.server.controller.meshing_drawing_update()

    def shown(self) -> None:
        """Send the views fresh frames of their current size."""
        self.server.controller.meshing_drawing_update()
        self.server.controller.meshing_mesh_update()
