"""The look every plot on the page shares: transparent backgrounds, the theme's text and grid colours.

Each plot builds on :func:`base_figure` and adds its own traces, so the convergence plot in Results
and the run's plots in Run read as one set and change together with the theme.
"""

from __future__ import annotations

import plotly.graph_objects as go

from .theme import LIGHT, PLOT_STYLES, PlotStyle

__all__ = ["base_figure"]


def base_figure(
    theme: str = LIGHT, *, margin: dict | None = None, x_title: str = "Step"
) -> tuple[go.Figure, PlotStyle]:
    """An empty figure in the page's theme, and the theme's plot colours for its traces.

    Parameters
    ----------
    theme : {"light", "dark"}
        The page's theme; an unknown one is taken as light.
    margin : dict, optional
        Plotly's ``{"l", "r", "t", "b"}`` margins, in pixels.
    x_title : str
        The horizontal axis title; empty for none.

    Returns
    -------
    tuple of (plotly.graph_objects.Figure, PlotStyle)
    """
    style = PLOT_STYLES[theme if theme in PLOT_STYLES else LIGHT]
    axis = {"gridcolor": style.grid, "zerolinecolor": style.grid, "linecolor": style.grid}
    figure = go.Figure()
    figure.update_layout(
        margin=margin or {"l": 60, "r": 20, "t": 30, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "system-ui, sans-serif", "color": style.text, "size": 12},
        xaxis={"title": x_title, **axis},
        yaxis=axis,
        legend={"orientation": "h", "y": 1.15, "x": 0},
    )
    return figure, style
