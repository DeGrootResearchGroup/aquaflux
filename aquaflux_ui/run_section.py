"""The Run section: where a case will be started and followed. A placeholder holding its place.

Running a case needs the solver, which this package does not import; the section will start one as
a separate ``aquaflux run`` process and follow it through the files it writes as it goes -- its log
and its convergence history, flushed one step at a time -- which is also how the Results section's
Reload already follows a run in progress.
"""

from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3

from .widgets import empty_state

__all__ = ["RunSection"]


class RunSection:
    """The Run section, holding its place in the workspace.

    Parameters
    ----------
    server : trame server
        The workspace's server.
    """

    key, title, icon = "run", "Run", "mdi-play-circle-outline"

    def __init__(self, server) -> None:
        self.server = server

    def drawer(self) -> None:
        """The controls a run will have, shown but not yet active."""
        with html.Div(classes="pa-4 d-flex flex-column", style="gap: 12px;"):
            html.Div("Run", classes="text-subtitle-2")
            with html.Div(classes="d-flex", style="gap: 8px;"):
                v3.VBtn("Start", prepend_icon="mdi-play", color="primary", disabled=True, flat=True)
                v3.VBtn("Stop", prepend_icon="mdi-stop", variant="tonal", disabled=True)
            v3.VSwitch(label="Overwrite earlier results", disabled=True, hide_details=True)

    def main(self) -> None:
        """How a case is run and followed today."""
        empty_state(
            "mdi-play-circle-outline",
            "Run and monitor",
            "Starting a case, following its march step by step, and stopping it are not available "
            "here in this version.",
            steps=(
                "Run it in a terminal with `aquaflux run case.yaml`, and open its output directory "
                "in this viewer.",
                "A run still in progress can be followed in Results: the convergence plot's Reload "
                "reads its history as it grows.",
            ),
        )

    def toolbar(self) -> None:
        """No buttons of its own."""

    def shown(self) -> None:
        """Nothing to refresh."""
