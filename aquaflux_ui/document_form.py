"""A file the solver reads, as a form: open it, change its settings, save it.

The form is built from the schema the installed solver prints for the file (:mod:`aquaflux_ui.case_form`),
so every choice it offers is one the solver reads. Opening and saving go through the solver's own
commands (:mod:`aquaflux_ui.solver_commands`): the file is read exactly as the solver reads it, and
written by the solver's own writer, which checks the file first and writes nothing it would refuse.

Saving rewrites the file from its settings, so comments and hand formatting in it are not kept, and a
setting written out at its default value is dropped. The page says so before the first save over a
file it opened.

One :class:`DocumentForm` edits one kind of file -- a case file in Setup, a mesh file in Mesh. Its
page state is named by its ``key`` (``case_path``, ``meshfile_path``, ...), so two forms share a page
without sharing a setting, and the section around it supplies what is particular to its file through
the hooks it is built with.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path

from trame.widgets import html
from trame.widgets import vuetify3 as v3

from . import case_form
from .case_form import CaseSchema, form_sections
from .file_browser import FileBrowser
from .solver_commands import CASE_FILE, CaseDocument, DocumentKind, SolverCommands
from .widgets import icon_button

__all__ = ["DOCUMENT_SUFFIXES", "GROUP_INDENT", "DocumentForm"]

#: The files the file browser offers as files to open: YAML documents.
DOCUMENT_SUFFIXES = (".yaml", ".yml")

#: Where in the file a refusal says the trouble is: ``... at 'boundaries.inlet': ...``.
_AT = re.compile(r" at '([^']+)'")

#: Widgets whose rows are edited by typing.
_TYPED = ("integer", "number", "string", "numbers", "strings", "raw")

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


class DocumentForm:
    """One file's form: its file actions, its settings, and the solver commands behind them.

    Parameters
    ----------
    server : trame server
        The workspace's server.
    key : str
        The prefix of its page state's names, unique in the page: ``case`` gives ``case_path``.
    commands : SolverCommands
        How the solver is asked.
    kind : DocumentKind
        The kind of file it edits.
    icons : mapping of {str: str}
        The icon each top-level section is headed by; a section not named gets a cog.
    default_name : str
        The file name offered when a file is first saved.
    opened : callable, optional
        Awaited after a file is opened or a new one started, to show what goes with it.
    changed : callable, optional
        Told ``(before, after)`` whenever an edit changes the document.
    names : callable, optional
        ``document -> {path pattern: names}``: lists of strings chosen from names the schema cannot
        know (see :func:`~aquaflux_ui.case_form.form_sections`).
    label_clicked : callable, optional
        Told ``(widget, label)`` when a row's name is clicked.
    label_classes : str, optional
        A page expression adding classes to a row's name, given ``row``.
    """

    def __init__(
        self,
        server,
        key: str,
        commands: SolverCommands,
        kind: DocumentKind = CASE_FILE,
        *,
        icons: Mapping[str, str],
        default_name: str,
        opened: Callable[[], Awaitable[None]] | None = None,
        changed: Callable[[dict | None, dict], None] | None = None,
        names: Callable[[Mapping], Mapping[tuple, Sequence[str]]] | None = None,
        label_clicked: Callable[[str, str], None] | None = None,
        label_classes: str = "{}",
    ) -> None:
        self.server = server
        self.key = key
        self.commands = commands
        self.kind = kind
        self.default_name = default_name
        self.schema: CaseSchema | None = None
        self.document: dict | None = None
        self.path: Path | None = None
        self._base: Path | None = None
        self._opened = opened
        self._changed = changed
        self._names = names
        self._label_clicked = label_clicked
        self._label_classes = label_classes
        self._confirmed_overwrite: Path | None = None
        self._rows: dict[str, case_form.Row] = {}
        self._tasks: set[asyncio.Task] = set()
        self._schema_task: asyncio.Future | None = None
        self.browser = FileBrowser(
            server, f"{key}_browser", DOCUMENT_SUFFIXES, self._chosen, what=kind.name
        )
        server.state.update(
            {
                self.state("loaded"): False,
                self.state("path"): "",
                self.state("name"): "",
                self.state("modified"): False,
                self.state("error"): "",
                self.state("error_path"): "",
                self.state("sections"): [],
                self.state("open_sections"): [],
                self.state("collapsed"): [],
                self.state("busy"): False,
                self.state("opening"): "",
                self.state("message"): "",
                self.state("message_ok"): True,
                self.state("show_message"): False,
                self.state("confirm"): False,
                self.state("revision"): 0,
                self.state("icons"): dict(icons),
            }
        )
        # Asking for the schema starts the solver in a process of its own, which takes seconds: ask
        # as soon as the page is served, so it is ready by the time a file is chosen.
        server.controller.on_server_ready.add(lambda **_: self.spawn(self._prefetch_schema()))

    def state(self, name: str) -> str:
        """The page-state name of this form's ``name``: ``state("path")`` is ``case_path``."""
        return f"{self.key}_{name}"

    # --- opening, saving --------------------------------------------------------------------------

    async def ask(self, function, *arguments):
        """Run a solver command off the page's event loop, showing the form as busy meanwhile."""
        with self.server.state as state:
            state[self.state("busy")] = True
        try:
            return await asyncio.to_thread(function, *arguments)
        finally:
            with self.server.state as state:
                state[self.state("busy")] = False

    async def load_schema(self) -> CaseSchema:
        """The file's schema, asked for once; a request already under way is waited for."""
        if self.schema is None:
            if self._schema_task is None:
                self._schema_task = asyncio.ensure_future(
                    asyncio.to_thread(self.commands.schema, self.kind)
                )
            try:
                self.schema = await self._schema_task
            except RuntimeError:
                self._schema_task = None  # asked again by the next file opened
                raise
        return self.schema

    async def _prefetch_schema(self) -> None:
        """Ask for the schema before it is needed; a failure is reported when a file is opened."""
        with contextlib.suppress(RuntimeError):
            await self.load_schema()

    async def open(self, path) -> str | None:
        """Open a file; returns why it could not be read, or ``None``."""
        with self.server.state as state:
            state[self.state("opening")] = Path(path).name
        try:
            await self.load_schema()
            document: CaseDocument = await self.ask(self.commands.show, path, self.kind)
        except RuntimeError as error:
            self.say(str(error), ok=False)
            return str(error)
        finally:
            with self.server.state as state:
                state[self.state("opening")] = ""
        if document.case is None or not isinstance(document.case, dict):
            return document.error or f"{path} holds no {self.kind.name}."
        self._show(document.case, Path(path).resolve(), document.error)
        if self._opened is not None:
            await self._opened()
        return None

    async def new(self, document: Mapping | None = None, directory=None) -> str | None:
        """Start a new file from ``document`` (unset, an empty one), saved nowhere until Save as.

        Parameters
        ----------
        document : mapping, optional
            What the new file holds to begin with.
        directory : path-like, optional
            The directory its relative paths are taken from until it is saved, where Save as starts.
        """
        try:
            await self.load_schema()
        except RuntimeError as error:
            self.say(str(error), ok=False)
            return str(error)
        self._show(dict(document or {}), None, None)
        self._base = None if directory is None else Path(directory).resolve()
        with self.server.state as state:
            state[self.state("modified")] = True
        if self._opened is not None:
            await self._opened()
        return None

    def _show(self, document: dict, path: Path | None, error: str | None) -> None:
        self.document, self.path, self._base = document, path, None
        self._confirmed_overwrite = None
        with self.server.state as state:
            state[self.state("loaded")] = True
            state[self.state("path")] = "" if path is None else str(path)
            state[self.state("name")] = "Not saved" if path is None else path.name
            state[self.state("modified")] = False
            state[self.state("collapsed")] = []
            self._show_error(error)
            self.lay_out()
            state[self.state("collapsed")] = [row["path"] for row in self._default_collapsed()]
            state[self.state("open_sections")] = [
                section["key"] for section in state[self.state("sections")][:1]
            ]

    @property
    def directory(self) -> Path | None:
        """The directory the open file's relative paths are taken from: its own once saved, else the
        one a new file was started in."""
        return self._base if self.path is None else self.path.parent

    async def save(self, path=None) -> str | None:
        """Save to ``path`` (its own file, unset); returns why it was refused, or ``None``."""
        if self.document is None:
            return f"No {self.kind.name} is open."
        target = Path(path).resolve() if path is not None else self.path
        if target is None:
            self.browse("save")
            return None
        origin = self.directory
        error = await self.ask(self.commands.write, target, self.document, origin, self.kind)
        with self.server.state as state:
            if error:
                self._show_error(error)
                self.lay_out()
                self.say(
                    f"Not saved: the {self.kind.name} is refused. See the message above the form.",
                    ok=False,
                )
                return error
            if origin is not None and origin != target.parent:
                # The relative paths were re-based on writing; show the file as it now reads.
                saved = await self.ask(self.commands.show, target, self.kind)
                if saved.case is not None:
                    self.document = saved.case
            self.path, self._base = target, None
            self._confirmed_overwrite = target
            state[self.state("path")], state[self.state("name")] = str(target), target.name
            state[self.state("modified")] = False
            self._show_error(None)
            self.lay_out()
            self.say(f"Saved {target.name}.")
        return None

    async def request_save(self) -> None:
        """Save over the open file -- after a first warning that its comments will not be kept."""
        if self.path is not None and self._confirmed_overwrite != self.path:
            self.server.state[self.state("confirm")] = True
            return
        await self.save()

    async def confirm_save(self) -> None:
        """The warning was accepted: save over the open file."""
        self.server.state[self.state("confirm")] = False
        self._confirmed_overwrite = self.path
        await self.save()

    def _chosen(self, path: str, mode: str) -> str | None:
        """The file browser chose ``path``: open it, or save to it."""
        self.spawn(self._report(self.open(path)) if mode == "open" else self.save(path))
        return None

    def spawn(self, coroutine) -> None:
        """Run ``coroutine`` on the page's event loop, keeping it alive until it finishes."""
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _report(self, task) -> None:
        error = await task
        if error:
            with self.server.state:
                self.say(error, ok=False)

    def browse(self, mode: str) -> None:
        """Open the file browser, starting where the open file is."""
        name = self.path.name if (mode == "save" and self.path is not None) else self.default_name
        self.browser.show(mode, self.directory, name if mode == "save" else "")

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
            self.say(str(error), ok=False)
            self.lay_out()  # put the box back to what the file holds
            return
        if keep:
            self.apply(case_form.set_value(self.document, steps, parsed, self.schema))
        elif row.is_set:
            self.apply(case_form.unset(self.document, steps))

    def choose_kind(self, path: str, kind: str) -> None:
        """A nested value, an item or an entry was given another kind (or the default)."""
        if self.document is not None:
            self.apply(case_form.set_kind(self.document, json.loads(path), kind or "", self.schema))

    def add_item(self, path: str, kind: str) -> None:
        """Append a value of ``kind`` to the list at ``path``."""
        if self.document is not None and kind:
            self.apply(case_form.add_item(self.document, json.loads(path), kind, self.schema))

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
            self.say(str(error) or "This table takes no entries.", ok=False)
            return
        self.apply(edited)

    def remove(self, path: str) -> None:
        """Remove a list item or table entry, or return a setting to its default."""
        if self.document is not None:
            self.apply(case_form.remove_at(self.document, json.loads(path)))

    def toggle(self, path: str) -> None:
        """Fold or unfold a nested value's settings."""
        collapsed = list(self.server.state[self.state("collapsed")])
        self.server.state[self.state("collapsed")] = (
            [p for p in collapsed if p != path] if path in collapsed else [*collapsed, path]
        )

    def apply(self, document: dict) -> None:
        """Make ``document`` the file's content, as an edit: marked modified, the form laid out."""
        before = self.document
        self.document = document
        self.server.state[self.state("modified")] = True
        if self._changed is not None:
            self._changed(before, document)
        self.lay_out()

    # --- state ----------------------------------------------------------------------------------

    def _show_error(self, error: str | None) -> None:
        state = self.server.state
        state[self.state("error")] = error or ""
        match = _AT.search(error or "")
        state[self.state("error_path")] = match.group(1) if match else ""

    def lay_out(self) -> None:
        """Rebuild the form's rows from the document."""
        names = self._names(self.document) if self._names is not None else None
        sections = form_sections(self.schema, self.document, names)
        self._rows = {}
        bad = self.server.state[self.state("error_path")]
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
        self.server.state[self.state("sections")] = laid
        self.server.state[self.state("revision")] += 1

    def _default_collapsed(self) -> list[dict]:
        """The groups folded when a file is opened: every one below a section's own top level."""
        return [
            row
            for section in self.server.state[self.state("sections")]
            for row in section["rows"]
            if row["has_inside"] and row["depth"] >= 1
        ]

    def say(self, message: str, ok: bool = True) -> None:
        """Show ``message`` for a few seconds: a confirmation, or a refusal when not ``ok``."""
        state = self.server.state
        state[self.state("message")] = message
        state[self.state("message_ok")] = ok
        state[self.state("show_message")] = True

    # --- page -----------------------------------------------------------------------------------

    def drawer(
        self,
        *,
        new: Callable[[], None] | None = None,
        buttons: Callable[[], None] | None = None,
        empty: str,
    ) -> None:
        """The file's actions, then its form.

        Parameters
        ----------
        new : callable, optional
            Starts a new file; given, a New button is shown.
        buttons : callable, optional
            Builds further buttons beside Save as, shown while a file is open.
        empty : str
            What the panel says before a file is open.
        """
        k = self.state
        self.browser.dialog()
        self._confirm_dialog()
        v3.VSnackbar(
            v_model=(k("show_message"),),
            text=(k("message"),),
            color=(f"{k('message_ok')} ? 'primary' : 'error'",),
            timeout=6000,
            location="bottom right",
        )
        with html.Div(classes="pa-3 d-flex flex-column", style="gap: 8px;"):
            with html.Div(classes="d-flex flex-wrap", style="gap: 8px;"):
                if new is not None:
                    v3.VBtn(
                        "New", prepend_icon="mdi-file-plus-outline", variant="tonal",
                        color="primary", classes="flex-grow-1", disabled=(k("busy"),), click=new,
                    )  # fmt: skip
                v3.VBtn(
                    "Open", prepend_icon="mdi-folder-open-outline", color="primary",
                    variant="flat", classes="flex-grow-1", click=(self.browse, "['open']"),
                )  # fmt: skip
                v3.VBtn(
                    "Save", v_if=k("loaded"), prepend_icon="mdi-content-save-outline",
                    variant="tonal", color="primary", classes="flex-grow-1",
                    disabled=(f"!{k('modified')} || {k('busy')}",), click=self.request_save,
                )  # fmt: skip
                v3.VBtn(
                    "Save as", v_if=k("loaded"), variant="tonal", classes="flex-grow-1",
                    disabled=(k("busy"),), click=(self.browse, "['save']"),
                )  # fmt: skip
                if buttons is not None:
                    with html.Div(v_if=k("loaded"), classes="d-flex flex-grow-1"):
                        buttons()
            with html.Div(
                v_if=k("opening"), classes="d-flex align-center text-body-2", style="gap: 8px;"
            ):
                v3.VProgressCircular(indeterminate=True, size=16, width=2, color="primary")
                html.Span("Opening {{ " + k("opening") + " }}…")
            with html.Div(
                v_if=f"{k('loaded')} && !{k('opening')}",
                classes="d-flex align-center",
                style="gap: 8px;",
            ):
                v3.VIcon("mdi-file-document-outline", color="primary", size="small")
                with html.Div(classes="flex-grow-1", style="min-width: 0;"):
                    html.Div("{{ " + k("name") + " }}", classes="text-subtitle-2 text-truncate")
                    html.Div(
                        "{{ " + k("path") + " }}",
                        classes="text-caption text-medium-emphasis text-truncate",
                        title=(k("path"),),
                    )
                v3.VProgressCircular(
                    v_if=k("busy"), indeterminate=True, size=16, width=2, color="primary"
                )
                v3.VChip(
                    "Modified", v_if=k("modified"), size="small", color="warning",
                    variant="tonal",
                )  # fmt: skip
            v3.VAlert(
                v_if=k("error"),
                text=(k("error"),),
                title=f"The solver refuses this {self.kind.name}",
                type="error",
                variant="tonal",
                density="compact",
            )
            html.Div(
                empty,
                v_if=f"!{k('loaded')} && !{k('opening')}",
                classes="text-body-2 text-medium-emphasis pa-2",
            )

    def form(self, skip: Sequence[str] = ()) -> None:
        """The file's settings, one folding panel per top-level section but those in ``skip``."""
        k = self.state
        shown = f"!{json.dumps(list(skip))}.includes(section.key)"
        v3.VDivider(v_if=k("loaded"))
        with v3.VExpansionPanels(
            v_if=k("loaded"), v_model=(k("open_sections"),), multiple=True, flat=True,
            variant="accordion",
        ):  # fmt: skip
            with v3.VExpansionPanel(
                v_for=f"section in {k('sections')}",
                key="section.key",
                value=("section.key",),
                v_show=shown,
            ):
                with v3.VExpansionPanelTitle():
                    v3.VIcon(
                        icon=(
                            "section.invalid ? 'mdi-alert-circle-outline' "
                            f": {k('icons')}[section.key] || 'mdi-cog-outline'",
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

    def _rows_template(self) -> None:
        """One row of the form, drawn by its widget."""
        k = self.state
        collapsed, subtitle = k("collapsed"), _group_subtitle(k("collapsed"))
        with html.Div(
            v_for="row in section.rows",
            key="row.id",
            v_show=f"!row.ancestors.some(a => {collapsed}.includes(a))",
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
                icon=(f"{collapsed}.includes(row.path) ? 'mdi-chevron-right' : 'mdi-chevron-down'",),
                size="x-small", variant="text", density="comfortable",
                click=(self.toggle, "[row.path]"),
            )  # fmt: skip
            html.Div(v_else=True, style="width: 28px; flex: none;")
            with html.Div(classes="af-row-label"):
                html.Span(
                    v_if=(
                        "['kind', 'item', 'entry'].includes(row.widget) && row.set_inside "
                        f"&& {collapsed}.includes(row.path)"
                    ),
                    classes="af-set-dot",
                    title=("row.set_inside + ' set inside'",),
                )
                html.Span(
                    "{{ row.label }}",
                    classes=(
                        "[row.is_set ? 'font-weight-medium' : 'text-medium-emphasis', "
                        f"{self._label_classes}]",
                    ),
                    click=(self._clicked, "[row.widget, row.label]"),
                )
                with v3.VTooltip(v_if="row.doc", text=("row.doc",), location="top", max_width=420):
                    with v3.Template(v_slot_activator=("{ props }",)):
                        v3.VIcon(
                            "mdi-information-outline", v_bind="props", size="x-small",
                            classes="ml-1 text-medium-emphasis",
                        )  # fmt: skip
                # Under a group's name: what is set inside it while it is folded, else what it is.
                html.Div(
                    "{{ " + subtitle + " }}",
                    v_if="['kind', 'item', 'entry'].includes(row.widget) && (" + subtitle + ")",
                    title=(subtitle,),
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
                    key=(f"'add-' + row.id + '-' + {k('revision')}",),
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

    def _clicked(self, widget: str, label: str) -> None:
        if self._label_clicked is not None:
            self._label_clicked(widget, label)

    def _select(self, path: str, widget: str, value) -> None:
        """A dropdown changed: a kind for a nested value or an entry, or a choice's value."""
        if widget == "choice":
            self.edit(path, value)
        else:
            self.choose_kind(path, value)

    def _confirm_dialog(self) -> None:
        k = self.state
        with v3.VDialog(v_model=(k("confirm"),), max_width=480):
            with v3.VCard(rounded="lg"):
                v3.VCardTitle("Save over {{ " + k("name") + " }}?", classes="text-subtitle-1 pt-4")
                v3.VCardText(
                    "The file is rewritten from its settings: comments and hand formatting in it are "
                    "not kept, and a setting written out at its default value is dropped. Save as a "
                    "new file instead to keep the original."
                )
                with v3.VCardActions(classes="px-4 pb-4"):
                    v3.VSpacer()
                    v3.VBtn("Cancel", variant="text", click=f"{k('confirm')} = false")
                    v3.VBtn("Save as…", variant="text", click=(self._save_as_instead,))
                    v3.VBtn("Save", color="primary", variant="flat", click=self.confirm_save)

    def _save_as_instead(self) -> None:
        self.server.state[self.state("confirm")] = False
        self.browse("save")


def _group_subtitle(collapsed: str) -> str:
    """A group heading's subtitle, as a page expression: what is set inside it while it is folded,
    else the one-line description of the kind it holds."""
    return f"({collapsed}.includes(row.path) && row.contents) ? row.contents : row.kind_summary"
