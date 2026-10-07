"""Page pieces every section uses: a collapsible side-panel section, an icon button, an empty state."""

from __future__ import annotations

from collections.abc import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3

__all__ = ["empty_state", "icon_button", "panel"]


def panel(
    value: str, title: str, icon: str, body: Callable[[], None], count: str | None = None
) -> None:
    """One collapsible section of a side panel: a header with its icon (and a count), then its body.

    Parameters
    ----------
    value : str
        The section's key in its panel set's open-sections state.
    title, icon : str
        Its heading, and a Material Design icon name (``mdi-...``).
    body : callable
        Builds the section's controls.
    count : str, optional
        A page expression for a number shown beside the heading when it is not zero.
    """
    with v3.VExpansionPanel(value=value):
        with v3.VExpansionPanelTitle():
            v3.VIcon(icon, size="small", color="primary", classes="mr-3")
            html.Span(title, classes="text-subtitle-2")
            if count is not None:
                v3.VChip(
                    f"{{{{ {count} }}}}", v_if=count, size="x-small", color="primary",
                    variant="flat", classes="ml-2",
                )  # fmt: skip
        with v3.VExpansionPanelText():
            body()


def icon_button(icon, tooltip, click) -> None:
    """A small icon button with a tooltip; each argument may be a value or a page expression."""
    with v3.VTooltip(text=tooltip, location="bottom"):
        with v3.Template(v_slot_activator=("{ props }",)):
            v3.VBtn(icon=icon, v_bind="props", size="small", variant="text", click=click)


def empty_state(icon: str, title: str, text: str, steps: tuple[str, ...] = ()) -> None:
    """A centred card saying what a section is for, when it has nothing to show.

    Parameters
    ----------
    icon, title, text : str
        A Material Design icon, a heading and a paragraph.
    steps : tuple of str
        Lines shown beneath the paragraph -- what to do instead, for example. Text between
        backticks is set as code.
    """
    with html.Div(classes="d-flex align-center justify-center pa-6", style="height: 100%;"):
        with v3.VCard(elevation=1, max_width=560, classes="pa-8 text-center"):
            v3.VIcon(icon, size=56, color="primary", classes="mb-4")
            html.Div(title, classes="text-h6 mb-2")
            html.Div(text, classes="text-body-2 text-medium-emphasis")
            for step in steps:
                with html.Div(classes="text-body-2 mt-3"):
                    for index, part in enumerate(step.split("`")):
                        # Text between backticks is a command, set as code.
                        if index % 2:
                            html.Code(part, classes="af-code")
                        elif part:
                            html.Span(part)
