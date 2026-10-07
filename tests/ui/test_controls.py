"""Tests for the page's control rules: what each edit does to a slice or threshold entry, and how the
controls become a view. Plain data throughout, so no page is built."""

from __future__ import annotations

import pytest

pytest.importorskip("pyvista")

from aquaflux_ui.controls import (
    MAGNITUDE,
    edit_slice,
    edit_threshold,
    fit_slice,
    fit_threshold,
    new_slice,
    new_threshold,
    parse_number,
    view_from_state,
)
from aquaflux_ui.scene import Slice, Threshold

BOUNDS = (-2.0, 2.0, -1.0, 3.0, 0.0, 3.0)
RANGES = {"G": (0.0, 40.0), "T": (280.0, 300.0)}


@pytest.mark.parametrize(
    ("value", "number"),
    [
        (0.6, 0.6),
        ("0.6", 0.6),
        ("-1e-3", -1e-3),
        ("-", None),
        ("", None),
        ("nan", None),
        ("inf", None),
        (None, None),
    ],
)
def test_a_typed_value_is_read_as_a_finite_number_or_not_at_all(value, number):
    assert parse_number(value) == number


def test_a_new_slice_crosses_the_middle_of_its_axis():
    assert new_slice(7, BOUNDS, "y") == {
        "id": 7,
        "axis": "y",
        "coordinate": 1.0,
        "min": -1.0,
        "max": 3.0,
    }


def test_a_typed_coordinate_is_taken_and_kept_inside_the_dataset():
    entry = new_slice(1, BOUNDS)
    assert edit_slice(entry, "coordinate", "0.603", BOUNDS)["coordinate"] == 0.603
    assert edit_slice(entry, "coordinate", 9.0, BOUNDS)["coordinate"] == 3.0
    assert edit_slice(entry, "coordinate", "-4", BOUNDS)["coordinate"] == 0.0


def test_a_coordinate_that_is_not_a_number_leaves_the_slice_alone():
    entry = new_slice(1, BOUNDS)
    assert edit_slice(entry, "coordinate", "0.", BOUNDS) == entry | {"coordinate": 0.0}
    assert edit_slice(entry, "coordinate", "-", BOUNDS) == entry


def test_a_new_axis_recentres_the_slice_on_it_and_keeps_its_identity():
    moved = edit_slice(new_slice(4, BOUNDS, "z") | {"coordinate": 0.1}, "axis", "x", BOUNDS)
    assert moved == {"id": 4, "axis": "x", "coordinate": 0.0, "min": -2.0, "max": 2.0}
    entry = new_slice(4, BOUNDS)
    assert edit_slice(entry, "axis", "w", BOUNDS) == entry


def test_a_slice_refits_to_another_dataset():
    fitted = fit_slice(new_slice(1, BOUNDS) | {"coordinate": 2.5}, (0, 1, 0, 1, 0, 1))
    assert (fitted["min"], fitted["max"], fitted["coordinate"]) == (0, 1, 1)


def test_a_new_threshold_keeps_the_upper_half_of_its_field():
    assert new_threshold(2, "G", RANGES["G"]) == {
        "id": 2, "field": "G", "low": 20.0, "high": 40.0, "min": 0.0, "max": 40.0,
    }  # fmt: skip


def test_a_new_field_resets_the_threshold_to_that_field():
    entry = new_threshold(2, "G", RANGES["G"])
    assert edit_threshold(entry, "field", "T", RANGES) == new_threshold(2, "T", RANGES["T"])
    assert edit_threshold(entry, "field", "absent", RANGES) == entry


def test_typed_ends_are_taken_and_never_invert_the_range():
    entry = new_threshold(2, "G", RANGES["G"])  # [20, 40]
    assert edit_threshold(entry, "low", "5.5", RANGES)[("low")] == 5.5
    raised = edit_threshold(entry, "low", 45.0, RANGES)
    assert (raised["low"], raised["high"]) == (45.0, 45.0)
    lowered = edit_threshold(entry, "high", 10.0, RANGES)
    assert (lowered["low"], lowered["high"]) == (10.0, 10.0)
    assert edit_threshold(entry, "high", "x", RANGES) == entry


def test_the_range_slider_sets_both_ends_in_order():
    entry = new_threshold(2, "G", RANGES["G"])
    assert edit_threshold(entry, "range", [30, 10], RANGES) | {} == entry | {
        "low": 10.0,
        "high": 30.0,
    }
    assert edit_threshold(entry, "range", [1], RANGES) == entry


def test_a_threshold_whose_field_a_dataset_lacks_moves_to_the_default_field():
    entry = new_threshold(3, "G", RANGES["G"])
    assert fit_threshold(entry, {"T": RANGES["T"]}, "T") == new_threshold(3, "T", RANGES["T"])
    assert fit_threshold(entry | {"low": 1.0}, RANGES, "T")["low"] == 1.0


def test_an_unknown_control_is_refused():
    with pytest.raises(KeyError, match="no control 'colour'"):
        edit_slice(new_slice(1, BOUNDS), "colour", 1, BOUNDS)
    with pytest.raises(KeyError, match="no control 'colour'"):
        edit_threshold(new_threshold(1, "G", None), "colour", 1, RANGES)


def _state(**overrides):
    state = {
        "dataset": "fields",
        "field": "G",
        "component": MAGNITUDE,
        "colormap": "inferno",
        "log_scale": True,
        "auto_range": True,
        "range_min": 1.0,
        "range_max": 2.0,
        "surface": True,
        "surface_opacity": 0.2,
        "edges": False,
        "slices": [],
        "thresholds": [],
    }
    return state | overrides


def test_the_controls_become_a_view():
    view = view_from_state(_state())
    assert (view.dataset, view.field, view.component) == ("fields", "G", None)
    assert (view.colormap, view.log_scale, view.color_range) == ("inferno", True, None)
    assert (view.slices, view.thresholds) == ((), ())
    assert view.surface_opacity == 0.2


def test_every_entry_becomes_part_of_the_view_in_order():
    state = _state(
        auto_range=False,
        component=2,
        slices=[new_slice(1, BOUNDS) | {"coordinate": 0.6}, new_slice(2, BOUNDS, "x")],
        thresholds=[
            new_threshold(3, "T", RANGES["T"]),
            {"id": 4, "field": "", "low": 0, "high": 1},
        ],
    )
    view = view_from_state(state)
    assert view.color_range == (1.0, 2.0) and view.component == 2
    assert view.slices == (Slice("z", 0.6), Slice("x", 0.0))
    # An entry with no field yet is not drawn.
    assert view.thresholds == (Threshold("T", 290.0, 300.0),)
