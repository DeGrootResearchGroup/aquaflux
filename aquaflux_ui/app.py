"""The workspace page: one interface for a case's setup, its run, and its results.

A narrow rail on the left switches between the sections. Each section is an object with its own side
panel, main area and app-bar buttons (:class:`Section`); the workspace builds them all once and shows
one at a time, so a section keeps its state -- the camera, the open panels, the slices -- while another
is in front. The shell owns only what is common to every section: the app bar, the rail, the theme,
and which section is shown.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from trame.app import get_server
from trame.ui.vuetify3 import VAppLayout
from trame.widgets import client, html
from trame.widgets import vuetify3 as v3

from .results_section import ResultsSection
from .run_section import RunSection
from .setup_section import SetupSection
from .solver_commands import SolverCommands
from .sources import ResultSource
from .theme import DARK, LIGHT, PAGE_CSS, vuetify_config
from .widgets import empty_state, icon_button

__all__ = ["NoResultsSection", "Section", "Workspace"]


class Section(Protocol):
    """One section of the workspace.

    Attributes
    ----------
    key : str
        Its name in the page's state; unique in a workspace.
    title, icon : str
        Its label in the rail, and a Material Design icon name.
    """

    key: str
    title: str
    icon: str

    def drawer(self) -> None:
        """Build its side panel."""

    def main(self) -> None:
        """Build its main area."""

    def toolbar(self) -> None:
        """Build its buttons in the app bar, shown while it is."""

    def shown(self) -> None:
        """React to being brought into view."""


class NoResultsSection:
    """The Results section when the page was opened with no results to show."""

    key, title, icon = ResultsSection.key, ResultsSection.title, ResultsSection.icon

    def drawer(self) -> None:
        """Nothing to control."""

    def main(self) -> None:
        """How to open results."""
        empty_state(
            ResultsSection.icon,
            "No results open",
            "Results are shown from the output directory a run writes, or from a VTK file.",
            steps=("Open them by starting the interface on them: `aquaflux-ui results/`.",),
        )

    def toolbar(self) -> None:
        """No buttons of its own."""

    def shown(self) -> None:
        """Nothing to refresh."""


class Workspace:
    """The page: Setup, Run and Results, switched from a rail, under one app bar.

    Parameters
    ----------
    source : ResultSource or None
        The results the Results section shows; given, the page opens on that section.
    server : trame server, optional
        Serves the page; unset, a new one.
    case : path-like, optional
        A case file the Setup section opens; given with no ``source``, the page opens on Setup.
    commands : SolverCommands, optional
        How the Setup section asks the solver; unset, the installed one.
    """

    def __init__(
        self,
        source: ResultSource | None = None,
        server=None,
        case=None,
        commands: SolverCommands | None = None,
    ) -> None:
        self.server = server if server is not None else get_server(client_type="vue3")
        self.title = source.title if source is not None else (Path(case).name if case else "")
        state = self.server.state
        state.update(
            {
                "theme": LIGHT,
                "trame__title": f"{self.title} · aquaflux" if self.title else "aquaflux",
                "section": ResultsSection.key if source is not None else SetupSection.key,
                "drawer": True,
            }
        )
        self.setup = SetupSection(self.server, commands, case)
        self.results = (
            ResultsSection(source, self.server) if source is not None else NoResultsSection()
        )
        self.run = RunSection(self.server, self.setup)
        self.sections: tuple[Section, ...] = (self.setup, self.run, self.results)
        if len({section.key for section in self.sections}) != len(self.sections):
            raise ValueError("every section of a workspace needs its own key.")
        self._build_page()
        state.change("section")(self._on_section)

    def _on_section(self, section, **_) -> None:
        for each in self.sections:
            if each.key == section:
                each.shown()

    def _build_page(self) -> None:
        with VAppLayout(self.server, theme=("theme",), vuetify_config=vuetify_config()):
            client.Style(PAGE_CSS)
            self._app_bar()
            with v3.VNavigationDrawer(rail=True, permanent=True, border=True, rail_width=76):
                with v3.VList(nav=True, density="compact", classes="pa-2 af-rail"):
                    for section in self.sections:
                        with v3.VListItem(
                            active=(f"section === '{section.key}'",),
                            click=f"section = '{section.key}'",
                            color="primary",
                            rounded="lg",
                            classes="mb-1 af-rail-item",
                        ):
                            v3.VIcon(section.icon)
                            html.Div(section.title, classes="af-rail-label")
            with v3.VNavigationDrawer(
                v_model=("drawer",), width=("section === 'setup' ? 520 : 360",), permanent=True,
                border=True,
            ):  # fmt: skip
                for section in self.sections:
                    with html.Div(v_show=f"section === '{section.key}'"):
                        section.drawer()
            with v3.VMain(classes="bg-background"):
                for section in self.sections:
                    with html.Div(v_show=f"section === '{section.key}'", style="height: 100%;"):
                        section.main()

    def _app_bar(self) -> None:
        with v3.VAppBar(flat=True, border="b", density="comfortable"):
            icon_button("mdi-dock-left", "Show or hide the side panel", "drawer = !drawer")
            v3.VIcon("mdi-water-outline", color="primary", classes="ml-2 mr-2")
            html.Span("aquaflux", classes="text-h6 font-weight-bold")
            v3.VDivider(vertical=True, classes="mx-4 my-3")
            html.Span(
                self.title,
                v_if="section !== 'setup' || !case_loaded",
                classes="text-body-1 text-medium-emphasis",
            )
            html.Span(
                "{{ case_name }}", v_if="section === 'setup' && case_loaded",
                classes="text-body-1 text-medium-emphasis",
            )  # fmt: skip
            v3.VSpacer()
            v3.VProgressCircular(
                v_show="trame__busy", indeterminate=True, size=18, width=2, color="primary",
                classes="mr-3",
            )  # fmt: skip
            for section in self.sections:
                # Not `d-flex` on this element: that utility is `display: flex !important`, which
                # overrides the `display: none` that hides it.
                with html.Div(v_show=f"section === '{section.key}'"):
                    with html.Div(classes="d-flex"):
                        section.toolbar()
            icon_button(
                ("theme === 'dark' ? 'mdi-weather-sunny' : 'mdi-weather-night'",),
                ("theme === 'dark' ? 'Light theme' : 'Dark theme'",),
                f"theme = theme === '{DARK}' ? '{LIGHT}' : '{DARK}'",
            )
