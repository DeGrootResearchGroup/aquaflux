"""A dialog for choosing a file on this machine: to open one, or to save under a name.

A browser's own file picker hands a page the file's content and hides where it lives. A case file's
mesh path is relative to the case file, so the page needs the real path -- and since the page is served
by this machine, it can list this machine's directories itself. :func:`list_directory` does the
listing and is pure; :class:`FileBrowser` is the dialog drawn from it.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from trame.widgets import html
from trame.widgets import vuetify3 as v3

__all__ = ["FileBrowser", "Listing", "list_directory"]


@dataclasses.dataclass(frozen=True)
class Listing:
    """One directory, as the dialog shows it.

    Attributes
    ----------
    directory : pathlib.Path
        The directory, absolute.
    entries : list of dict
        Its subdirectories, then its files with one of the wanted suffixes, each sorted by name, case
        ignored: ``{"name", "path", "is_dir"}``. Hidden entries (a leading dot) are left out.
    crumbs : list of dict
        The directory's ancestors and itself, from the top: ``{"name", "path"}``.
    error : str or None
        Why it could not be read, if it could not; ``entries`` is then empty.
    """

    directory: Path
    entries: list[dict]
    crumbs: list[dict]
    error: str | None = None


def list_directory(directory: str | os.PathLike[str], suffixes: Sequence[str]) -> Listing:
    """List a directory's subdirectories and the files ending in one of ``suffixes``.

    Parameters
    ----------
    directory : path-like
        The directory; a file is taken as the directory it is in.
    suffixes : sequence of str
        The file suffixes shown (``".yaml"``, say), compared ignoring case.

    Returns
    -------
    Listing
    """
    directory = Path(directory).expanduser().resolve()
    if directory.is_file():
        directory = directory.parent
    crumbs = [
        {"name": part.name or part.anchor, "path": str(part)}
        for part in (*reversed(directory.parents), directory)
    ]
    try:
        children = list(directory.iterdir())
    except OSError as error:
        return Listing(
            directory, [], crumbs, f"{directory} cannot be read: {error.strerror or error}"
        )
    wanted = {suffix.lower() for suffix in suffixes}
    visible = [child for child in children if not child.name.startswith(".")]
    folders = sorted((c for c in visible if c.is_dir()), key=lambda c: c.name.lower())
    files = sorted(
        (c for c in visible if c.is_file() and c.suffix.lower() in wanted),
        key=lambda c: c.name.lower(),
    )
    entries = [{"name": c.name, "path": str(c), "is_dir": True} for c in folders] + [
        {"name": c.name, "path": str(c), "is_dir": False} for c in files
    ]
    return Listing(directory, entries, crumbs)


class FileBrowser:
    """A dialog listing this machine's directories, to open a file or to save under a name.

    Parameters
    ----------
    server : trame server
        The page's server; the dialog's state is named under ``key``.
    key : str
        A prefix for its state names, unique on the page.
    suffixes : sequence of str
        The files shown.
    on_choose : callable
        ``(path, mode) -> str | None``, called with the chosen file and the mode the dialog was opened
        in; it returns why the choice failed (shown in the dialog, which stays open) or ``None``.
    """

    def __init__(
        self,
        server,
        key: str,
        suffixes: Sequence[str],
        on_choose: Callable[[str, str], str | None],
    ) -> None:
        self.server = server
        self.key = key
        self.suffixes = tuple(suffixes)
        self.on_choose = on_choose
        server.state.update(
            {
                f"{key}_open": False,
                f"{key}_mode": "open",
                f"{key}_listing": {},
                f"{key}_name": "",
                f"{key}_error": "",
            }
        )

    def show(self, mode: str, start: str | os.PathLike[str] | None = None, name: str = "") -> None:
        """Open the dialog in ``mode`` (``"open"`` or ``"save"``) at ``start`` (the working directory)."""
        state = self.server.state
        state[f"{self.key}_mode"] = mode
        state[f"{self.key}_name"] = name
        state[f"{self.key}_error"] = ""
        self.go(start or os.getcwd())
        state[f"{self.key}_open"] = True

    def go(self, directory: str) -> None:
        """Show ``directory``."""
        listing = list_directory(directory, self.suffixes)
        self.server.state[f"{self.key}_listing"] = {
            "directory": str(listing.directory),
            "entries": listing.entries,
            "crumbs": listing.crumbs,
        }
        self.server.state[f"{self.key}_error"] = listing.error or ""

    def pick(self, path: str, is_dir: bool) -> None:
        """An entry was clicked: enter a directory; open a file, or take its name to save over."""
        state = self.server.state
        if is_dir:
            self.go(path)
        elif state[f"{self.key}_mode"] == "save":
            state[f"{self.key}_name"] = Path(path).name
        else:
            self._choose(path)

    def save(self) -> None:
        """Save under the typed name, in the shown directory."""
        state = self.server.state
        name = str(state[f"{self.key}_name"]).strip()
        if not name:
            state[f"{self.key}_error"] = "Give the file a name."
            return
        if Path(name).suffix.lower() not in {s.lower() for s in self.suffixes}:
            name += self.suffixes[0]
        self._choose(str(Path(state[f"{self.key}_listing"]["directory"]) / name))

    def _choose(self, path: str) -> None:
        error = self.on_choose(path, self.server.state[f"{self.key}_mode"])
        if error:
            self.server.state[f"{self.key}_error"] = error
        else:
            self.server.state[f"{self.key}_open"] = False

    def dialog(self) -> None:
        """Build the dialog; it is shown by :meth:`show`."""
        k = self.key
        with v3.VDialog(v_model=(f"{k}_open",), max_width=640, scrollable=True):
            with v3.VCard(rounded="lg"):
                with v3.VCardItem(classes="pb-1"):
                    with v3.Template(v_slot_prepend=True):
                        v3.VIcon(
                            icon=(
                                f"{k}_mode === 'save' ? 'mdi-content-save-outline' "
                                ": 'mdi-folder-open-outline'",
                            ),
                            color="primary",
                        )
                    v3.VCardTitle(
                        f"{{{{ {k}_mode === 'save' ? 'Save case file' : 'Open case file' }}}}",
                        classes="text-subtitle-1",
                    )
                with html.Div(classes="px-4 d-flex align-center flex-wrap", style="gap: 2px;"):
                    v3.VBtn(
                        icon="mdi-home-outline", size="x-small", variant="text",
                        click=(self.go, f"[{_js_string(str(Path.home()))}]"),
                    )  # fmt: skip
                    with html.Template(v_for=f"(crumb, i) in {k}_listing.crumbs", key="crumb.path"):
                        v3.VIcon(
                            "mdi-chevron-right",
                            size="x-small",
                            v_if="i > 0",
                            classes="text-medium-emphasis",
                        )
                        v3.VBtn(
                            "{{ crumb.name }}", size="x-small", variant="text", classes="px-1",
                            click=(self.go, "[crumb.path]"),
                        )  # fmt: skip
                v3.VDivider(classes="mt-2")
                with v3.VCardText(style="height: 360px;", classes="pa-1"):
                    with v3.VList(density="compact", nav=True):
                        v3.VListItem(
                            v_for=f"entry in {k}_listing.entries",
                            key="entry.path",
                            title=("entry.name",),
                            prepend_icon=(
                                "entry.is_dir ? 'mdi-folder-outline' : 'mdi-file-document-outline'",
                            ),
                            click=(self.pick, "[entry.path, entry.is_dir]"),
                            rounded="lg",
                        )
                        with html.Div(
                            v_if=f"!{k}_listing.entries || !{k}_listing.entries.length",
                            classes="text-body-2 text-medium-emphasis pa-4 text-center",
                        ):
                            html.Span(f"No folders or {', '.join(self.suffixes)} files here.")
                v3.VDivider()
                with v3.VCardActions(classes="px-4 py-3", style="gap: 8px;"):
                    v3.VTextField(
                        v_if=f"{k}_mode === 'save'",
                        v_model=(f"{k}_name",),
                        label="File name",
                        hide_details=True,
                        keyup_enter=self.save,
                        __events=[("keyup_enter", "keyup.enter")],
                    )
                    v3.VSpacer(v_else=True)
                    v3.VBtn("Cancel", variant="text", click=f"{k}_open = false")
                    v3.VBtn(
                        "Save", v_if=f"{k}_mode === 'save'", color="primary", variant="flat",
                        click=self.save,
                    )  # fmt: skip
                v3.VAlert(
                    v_if=f"{k}_error",
                    text=(f"{k}_error",),
                    type="error",
                    variant="tonal",
                    density="compact",
                    classes="mx-4 mb-3",
                )


def _js_string(text: str) -> str:
    """``text`` as a single-quoted JavaScript string, safe inside a double-quoted page attribute."""
    escaped = text.replace("\\", "\\\\").replace("'", "\\'").replace('"', "\\x22")
    return f"'{escaped}'"
