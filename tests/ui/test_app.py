"""Tests for the Results section's convergence figure. The control rules are in ``test_controls.py``."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("trame")

from aquaflux_ui import ConvergenceHistory
from aquaflux_ui.results_section import convergence_figure
from aquaflux_ui.theme import PLOT_STYLES


def test_the_figure_plots_both_residual_columns_on_a_log_axis():
    history = ConvergenceHistory(
        {
            "step": np.array([1.0, 2.0]),
            "residual_norm": np.array([1.0, 0.1]),
            "residual_ratio": np.array([0.5, 0.05]),
        }
    )
    figure = convergence_figure(history)
    assert [trace.name for trace in figure.data] == ["Residual / initial residual", "Residual"]
    np.testing.assert_array_equal(figure.data[0].y, [0.5, 0.05])
    assert figure.layout.yaxis.type == "log"


def test_the_figure_is_drawn_in_the_pages_theme():
    history = ConvergenceHistory({"step": np.array([1.0]), "residual_ratio": np.array([0.5])})
    for theme, style in PLOT_STYLES.items():
        figure = convergence_figure(history, theme)
        assert figure.data[0].line.color == style.lines[0]
        assert figure.layout.font.color == style.text


@pytest.mark.parametrize(
    ("history", "says"),
    [
        (None, "No convergence history saved"),
        (
            ConvergenceHistory({"step": np.empty(0), "residual_norm": np.empty(0)}),
            "took no march steps",
        ),
    ],
)
def test_the_figure_says_why_it_is_empty(history, says):
    figure = convergence_figure(history)
    assert figure.data == ()
    assert says in figure.layout.annotations[0].text
