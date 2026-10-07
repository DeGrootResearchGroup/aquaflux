"""Tests for the viewer's look: colormap previews, and that every theme defines every colour set."""

from __future__ import annotations

import pytest

pytest.importorskip("pyvista")

from aquaflux_ui.theme import (
    DARK,
    LIGHT,
    PLOT_STYLES,
    THEMES,
    colormap_gradient,
    vuetify_config,
)


def test_a_colormap_preview_runs_from_its_low_end_to_its_high_end():
    # Viridis is defined from #440154 (dark purple) to #fde725 (yellow).
    gradient = colormap_gradient("viridis", stops=3)
    assert gradient.startswith("linear-gradient(90deg, #440154, ")
    assert gradient.endswith(", #fde725)")
    assert gradient.count("#") == 3


def test_a_preview_needs_two_stops_and_a_real_colormap():
    with pytest.raises(ValueError, match="at least 2 stops"):
        colormap_gradient("viridis", stops=1)
    with pytest.raises(KeyError):
        colormap_gradient("no-such-map")


def test_every_theme_has_a_page_a_view_and_a_plot_palette():
    themes = vuetify_config()["theme"]["themes"]
    assert set(themes) == set(THEMES) == set(PLOT_STYLES) == {LIGHT, DARK}
    assert themes[DARK]["dark"] is True and themes[LIGHT]["dark"] is False
    assert vuetify_config(DARK)["theme"]["defaultTheme"] == DARK


def _luminance(colour):
    """The relative luminance of a ``#RRGGBB`` colour, as the WCAG contrast ratio defines it."""
    channels = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a, b):
    light, dark = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


@pytest.mark.parametrize("theme", [LIGHT, DARK])
def test_a_tooltips_text_is_readable_on_its_background(theme):
    # A tooltip is drawn on `surface-variant` in `on-surface-variant`, which Vuetify does not derive:
    # left unset it is a near-white, invisible on the light theme's pale background.
    colours = vuetify_config()["theme"]["themes"][theme]["colors"]
    assert _contrast(colours["surface-variant"], colours["on-surface-variant"]) >= 4.5
