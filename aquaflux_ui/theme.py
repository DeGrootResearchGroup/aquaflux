"""The page's look: its colour themes, the rendered view's palette in each, and colormap previews.

Every colour the viewer uses is defined here, once, for a light and a dark theme: the page's own
(handed to Vuetify, the page's component library) and the rendered view's (its background gradient,
the colour of the colour bar's text, the plain colour of a surface drawn as context). Switching theme
switches both together, so the 3D view never sits light-on-dark inside a dark page.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from matplotlib import colormaps

__all__ = [
    "DARK",
    "LIGHT",
    "PAGE_CSS",
    "PLOT_STYLES",
    "THEMES",
    "PlotStyle",
    "RenderStyle",
    "colormap_gradient",
    "vuetify_config",
]

LIGHT, DARK = "light", "dark"


@dataclasses.dataclass(frozen=True)
class RenderStyle:
    """The colours of the rendered view in one theme.

    Attributes
    ----------
    background_top, background_bottom : str
        The view's background, a vertical gradient between the two.
    text : str
        The colour bar's title and labels.
    context : str
        The plain colour of a surface drawn as context beside a slice or a threshold.
    highlight : str
        The colour a chosen boundary patch is drawn in, set apart from everything else.
    """

    background_top: str
    background_bottom: str
    text: str
    context: str
    highlight: str


#: The rendered view's palette by theme name.
THEMES: dict[str, RenderStyle] = {
    LIGHT: RenderStyle("#FFFFFF", "#DDE3EA", "#334155", "#B8C2CC", "#EA580C"),
    DARK: RenderStyle("#1E293B", "#0B1120", "#E2E8F0", "#64748B", "#FB923C"),
}


@dataclasses.dataclass(frozen=True)
class PlotStyle:
    """The colours of the convergence plot in one theme.

    Attributes
    ----------
    lines : tuple of str
        One colour per plotted series, in order.
    text, grid : str
        The axes' text and grid lines. The plot's own background is transparent, so it takes the
        page's.
    equations : tuple of str
        One colour per equation of a per-equation residual plot, in the order the equations come.
        Six hues of a palette validated for colour-vision deficiency on adjacent lines, in its own
        order, leaving out the two the event marks use; on the light surface the worst adjacent pair
        is legal only beside a second encoding, which is why each line is labelled at its end.
    retry, refit : str
        The marks of a redone step and of a preconditioner refit.
    band : str
        The shading behind the steps a continuation spends before it reaches the case's own problem.
    muted : str
        Limit lines (a target, a budget) and secondary bars.
    """

    lines: tuple[str, ...]
    text: str
    grid: str
    equations: tuple[str, ...]
    retry: str
    refit: str
    band: str
    muted: str


#: The plots' palette by theme name.
PLOT_STYLES: dict[str, PlotStyle] = {
    LIGHT: PlotStyle(
        ("#0E7490", "#94A3B8"),
        "#334155",
        "#E2E8F0",
        ("#2A78D6", "#1BAF7A", "#EDA100", "#E87BA4", "#008300", "#E34948"),
        "#EA580C",
        "#4A3AA7",
        "rgba(148, 163, 184, 0.16)",
        "#64748B",
    ),
    DARK: PlotStyle(
        ("#22D3EE", "#64748B"),
        "#CBD5E1",
        "#1F2937",
        ("#3987E5", "#199E70", "#C98500", "#D55181", "#008300", "#E66767"),
        "#FB923C",
        "#9085E9",
        "rgba(100, 116, 139, 0.18)",
        "#94A3B8",
    ),
}


#: The page's colours by theme, in Vuetify's theme vocabulary. ``surface-variant`` is a tooltip's
#: background, and its text colour ``on-surface-variant`` is given with it: Vuetify works out most text
#: colours from their background, but keeps its own fixed near-white for this one, which on a light
#: ``surface-variant`` leaves every tooltip unreadable.
_PAGE_COLOURS = {
    LIGHT: {
        "primary": "#0E7490",
        "secondary": "#475569",
        "background": "#F1F4F8",
        "surface": "#FFFFFF",
        "surface-variant": "#E8EDF2",
        "on-surface-variant": "#1E293B",
        "error": "#DC2626",
    },
    DARK: {
        "primary": "#22D3EE",
        "secondary": "#94A3B8",
        "background": "#0B1120",
        "surface": "#111827",
        "surface-variant": "#1F2937",
        "on-surface-variant": "#E2E8F0",
        "error": "#F87171",
    },
}

#: Page-wide style: a system font stack (the viewer runs offline, so no web font is fetched), tighter
#: panel padding, and a rounded frame for the rendered view.
PAGE_CSS = """
html, body, .v-application {
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.v-expansion-panel-text__wrapper { padding: 4px 16px 16px; }
.v-expansion-panel-title { min-height: 48px !important; padding: 12px 16px; }
.af-view { border-radius: 12px; overflow: hidden; }
.af-swatch { width: 72px; height: 12px; border-radius: 3px; flex: none; }
.af-label { font-size: 0.75rem; letter-spacing: 0.02em; opacity: 0.7; }
.af-rail-item { flex-direction: column; justify-content: center; text-align: center;
  padding: 8px 0 !important; min-height: 56px !important; }
.af-rail-item .v-list-item__content { display: flex; flex-direction: column; align-items: center; }
.af-rail-label { font-size: 0.7rem; margin-top: 2px; }
.af-row { display: flex; align-items: center; gap: 8px; min-height: 38px; padding-right: 4px;
  --af-rail: rgba(var(--v-theme-on-surface), 0.24); }
.af-row:hover { box-shadow: inset 0 0 0 100vmax rgba(var(--v-theme-on-surface), 0.03); }
.af-depth-1 { background-color: rgba(var(--v-theme-on-surface), 0.022); }
.af-depth-2 { background-color: rgba(var(--v-theme-on-surface), 0.04); }
.af-row-sub { font-size: 0.72rem; line-height: 1.2; color: rgba(var(--v-theme-on-surface), 0.6);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.af-set-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px;
  vertical-align: middle; background: rgb(var(--v-theme-primary)); }
.af-row-label { flex: 1 1 40%; min-width: 0; font-size: 0.875rem; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }
.af-row-input { flex: 1 1 60%; min-width: 0; }
/* A box keeps the page's surface colour inside a tinted group, so an unset one still reads as a box
   to type in rather than a disabled one. */
.af-row-input .v-field { background: rgb(var(--v-theme-surface)); }
/* The form's boxes in the labels' own size: Vuetify's compact field is still 1rem text in a 40px box,
   which reads as larger than the setting it belongs to. */
.af-row-input .v-field { font-size: 0.875rem; --v-input-control-height: 32px;
  --v-field-input-min-height: 32px; }
.af-row-input .v-field__input { min-height: 32px; padding-top: 4px; padding-bottom: 4px; font-size: 0.875rem; }
.af-row-input .v-field__input input { font-size: 0.875rem; }
.af-row-input .v-select__selection-text { font-size: 0.875rem; }
.af-row-input .v-chip { font-size: 0.75rem; }
.v-select__content .v-list-item-title { font-size: 0.875rem; }
.v-select__content .v-list-item { min-height: 32px; }
.af-row-invalid { background-color: rgba(var(--v-theme-error), 0.08) !important; }
.af-row-unknown .af-row-label { text-decoration: line-through; }
.af-raw input { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8rem; }
.af-patch { cursor: pointer; text-decoration: underline dotted; text-underline-offset: 3px; }
.af-patch-selected { color: rgb(var(--v-theme-warning)); font-weight: 600; }
.af-code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8rem;
  padding: 1px 6px; border-radius: 4px; background: rgba(var(--v-theme-primary), 0.1); }
/* The Run section: headline tiles, the plots two by two, the key/value lists, the events. */
.af-tiles { display: grid; grid-template-columns: 1.6fr repeat(4, minmax(0, 1fr)); gap: 12px; }
.af-plot-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
@media (max-width: 1100px) {
  .af-tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .af-plot-grid { grid-template-columns: minmax(0, 1fr); }
}
.af-number { font-variant-numeric: tabular-nums; }
.af-kv { display: grid; grid-template-columns: auto 1fr; column-gap: 12px; row-gap: 6px;
  font-size: 0.8125rem; }
.af-kv > :nth-child(even) { text-align: right; }
.af-legend { display: flex; align-items: center; gap: 14px; font-size: 0.75rem;
  color: rgba(var(--v-theme-on-surface), 0.7); }
.af-mark { display: inline-block; vertical-align: middle; margin-right: 5px; }
.af-mark-dash { width: 16px; border-top: 2px dashed var(--af-muted); }
.af-mark-retry { width: 10px; height: 10px; border-radius: 50%; background: var(--af-retry); }
.af-mark-refit { width: 2px; height: 12px; background: var(--af-refit); }
.af-event { display: grid; grid-template-columns: 64px 12px 1fr auto; gap: 8px; align-items: center;
  padding: 6px 0; font-size: 0.8125rem; border-top: 1px solid rgba(var(--v-theme-on-surface), 0.08); }
.af-event:first-child { border-top: 0; }
.af-event-dot { width: 10px; height: 10px; border-radius: 50%; }
.af-event-retry { background: var(--af-retry); }
.af-event-refit { background: var(--af-refit); }
.af-event-arrived { background: var(--af-muted); }
.af-log { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.75rem;
  max-height: 320px; overflow: auto; white-space: pre; padding: 8px;
  background: rgba(var(--v-theme-on-surface), 0.04); border-radius: 6px; }
"""


def vuetify_config(default: str = LIGHT) -> dict:
    """The configuration Vuetify is created with: both themes and the default component style.

    Parameters
    ----------
    default : {"light", "dark"}
        The theme the page opens in.

    Returns
    -------
    dict
    """
    return {
        "theme": {
            "defaultTheme": default,
            "themes": {
                name: {"dark": name == DARK, "colors": colours}
                for name, colours in _PAGE_COLOURS.items()
            },
        },
        "defaults": {
            "VTextField": {"variant": "outlined", "density": "compact", "color": "primary"},
            "VSelect": {"variant": "outlined", "density": "compact", "color": "primary"},
            "VSwitch": {"color": "primary", "inset": True, "density": "compact"},
            "VSlider": {"color": "primary", "density": "compact"},
            "VRangeSlider": {"color": "primary", "density": "compact"},
            "VBtn": {"class": "text-none", "rounded": "lg"},
            "VCard": {"rounded": "lg"},
        },
    }


def colormap_gradient(name: str, stops: int = 8) -> str:
    """A CSS ``linear-gradient`` previewing a Matplotlib colormap from its low end to its high end.

    Parameters
    ----------
    name : str
        The colormap.
    stops : int
        How many colours the gradient is sampled at, ``>= 2``.

    Returns
    -------
    str
        For example ``"linear-gradient(90deg, #440154, ..., #fde725)"``.

    Raises
    ------
    KeyError
        If Matplotlib has no colormap of that name.
    ValueError
        If ``stops < 2``.
    """
    if stops < 2:
        raise ValueError(f"a gradient needs at least 2 stops, got {stops}.")
    rgb = np.rint(255 * colormaps[name](np.linspace(0.0, 1.0, stops))[:, :3]).astype(int)
    colours = ", ".join("#{:02x}{:02x}{:02x}".format(*row) for row in rgb)
    return f"linear-gradient(90deg, {colours})"


def _mark_colours(theme: str) -> str:
    """The Run section's legend and event marks in ``theme``, from the same colours as its plots."""
    style = PLOT_STYLES[theme]
    return (
        f".v-theme--{theme} {{ --af-retry: {style.retry}; --af-refit: {style.refit}; "
        f"--af-muted: {style.muted}; }}\n"
    )


PAGE_CSS += "".join(_mark_colours(theme) for theme in PLOT_STYLES)
