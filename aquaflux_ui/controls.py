"""The page's control values, and how they change: plain data in, plain data out, no page needed.

The page keeps its slices and thresholds as lists of small dictionaries -- what a browser can hold
and send back -- and every edit to them goes through the functions here, which return a new entry
rather than changing one. :func:`view_from_state` is the one place those values become a
:class:`~aquaflux_ui.scene.View`. Nothing here imports the page, so the rules for what an edit does
(a slice moved to another axis is re-centred on it; a typed number that does not read as one is
ignored; a range keeps its low end below its high end) are tested directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .scene import AXES, Slice, Threshold, View

__all__ = [
    "COLORMAPS",
    "MAGNITUDE",
    "edit_slice",
    "edit_threshold",
    "fit_slice",
    "fit_threshold",
    "new_slice",
    "new_threshold",
    "parse_number",
    "view_from_state",
]

#: The colormaps offered; the first is the default.
COLORMAPS = ("viridis", "inferno", "plasma", "cividis", "turbo", "coolwarm", "gray")

#: The "component" choice that shows a vector's magnitude.
MAGNITUDE = -1

#: A range a control falls back to when its field has no finite value to take one from.
_NO_RANGE = (0.0, 1.0)


def parse_number(value: object) -> float | None:
    """A typed value as a finite number, or ``None`` if it does not read as one.

    Parameters
    ----------
    value : object
        What a box or a slider sent: a number, or text such as ``"0.6"`` or ``"-"`` while a value is
        being typed.

    Returns
    -------
    float or None
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _axis_extent(bounds: Sequence[float], axis: str) -> tuple[float, float]:
    i = AXES.index(axis)
    return float(bounds[2 * i]), float(bounds[2 * i + 1])


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def new_slice(ident: int, bounds: Sequence[float], axis: str = "z") -> dict:
    """A slice entry through the middle of the dataset, normal to ``axis``.

    Parameters
    ----------
    ident : int
        A number unique among the page's entries, which keeps each entry's controls its own.
    bounds : sequence of 6 float
        The dataset's ``(x_min, x_max, y_min, y_max, z_min, z_max)``.
    axis : {"x", "y", "z"}
        The plane's normal.

    Returns
    -------
    dict
        ``id``, ``axis``, ``coordinate``, and the axis's extent as ``min`` and ``max``.
    """
    low, high = _axis_extent(bounds, axis)
    return {"id": ident, "axis": axis, "coordinate": 0.5 * (low + high), "min": low, "max": high}


def fit_slice(entry: Mapping, bounds: Sequence[float]) -> dict:
    """``entry`` refitted to a dataset's bounds: its extent replaced, its coordinate kept inside it.

    Parameters
    ----------
    entry : mapping
        A slice entry.
    bounds : sequence of 6 float
        The dataset's bounds.

    Returns
    -------
    dict
    """
    low, high = _axis_extent(bounds, entry["axis"])
    return dict(entry) | {
        "min": low,
        "max": high,
        "coordinate": _clamp(float(entry["coordinate"]), low, high),
    }


def edit_slice(entry: Mapping, key: str, value: object, bounds: Sequence[float]) -> dict:
    """``entry`` with one control changed.

    Parameters
    ----------
    entry : mapping
        A slice entry.
    key : {"axis", "coordinate"}
        Which control.
    value : object
        Its new value. A new axis re-centres the plane on that axis. A coordinate that does not read
        as a number leaves the entry as it was; one outside the dataset is moved to its edge.
    bounds : sequence of 6 float
        The dataset's bounds.

    Returns
    -------
    dict

    Raises
    ------
    KeyError
        If ``key`` is not a slice's control.
    """
    if key == "axis":
        if value not in AXES or value == entry["axis"]:
            return dict(entry)
        return new_slice(entry["id"], bounds, str(value))
    if key == "coordinate":
        number = parse_number(value)
        return (
            dict(entry)
            if number is None
            else fit_slice(dict(entry) | {"coordinate": number}, bounds)
        )
    raise KeyError(f"a slice has no control {key!r}.")


def new_threshold(ident: int, field: str, field_range: tuple[float, float] | None) -> dict:
    """A threshold entry on ``field``, keeping the upper half of its range.

    Parameters
    ----------
    ident : int
        A number unique among the page's entries.
    field : str
        The field tested.
    field_range : tuple of (float, float) or None
        The field's finite extent.

    Returns
    -------
    dict
        ``id``, ``field``, ``low``, ``high``, and the field's extent as ``min`` and ``max``.
    """
    low, high = field_range or _NO_RANGE
    return {
        "id": ident,
        "field": field,
        "low": low + 0.5 * (high - low),
        "high": high,
        "min": low,
        "max": high,
    }


def fit_threshold(
    entry: Mapping, field_ranges: Mapping[str, tuple[float, float] | None], default_field: str
) -> dict:
    """``entry`` refitted to a dataset: its extent updated, or a new entry if its field is gone.

    Parameters
    ----------
    entry : mapping
        A threshold entry.
    field_ranges : mapping of {str: tuple or None}
        The dataset's fields and their extents.
    default_field : str
        The field a threshold whose field the dataset lacks is moved to.

    Returns
    -------
    dict
    """
    if entry["field"] not in field_ranges:
        return new_threshold(entry["id"], default_field, field_ranges.get(default_field))
    low, high = field_ranges[entry["field"]] or _NO_RANGE
    return dict(entry) | {"min": low, "max": high}


def edit_threshold(
    entry: Mapping, key: str, value: object, field_ranges: Mapping[str, tuple[float, float] | None]
) -> dict:
    """``entry`` with one control changed.

    Parameters
    ----------
    entry : mapping
        A threshold entry.
    key : {"field", "low", "high", "range"}
        Which control: the field tested, one end typed, or both ends from the range slider.
    value : object
        Its new value. A new field resets the range to that field's upper half. An end that does not
        read as a number leaves the entry as it was; an end moved past the other one moves that one
        with it, so the range never inverts.
    field_ranges : mapping of {str: tuple or None}
        The dataset's fields and their extents.

    Returns
    -------
    dict

    Raises
    ------
    KeyError
        If ``key`` is not a threshold's control.
    """
    entry = dict(entry)
    if key == "field":
        if value not in field_ranges or value == entry["field"]:
            return entry
        return new_threshold(entry["id"], str(value), field_ranges[value])
    if key == "range":
        ends = [parse_number(end) for end in value] if isinstance(value, Sequence) else []
        if len(ends) != 2 or None in ends:
            return entry
        low, high = sorted(ends)
        return entry | {"low": low, "high": high}
    if key in ("low", "high"):
        number = parse_number(value)
        if number is None:
            return entry
        entry[key] = number
        if entry["low"] > entry["high"]:
            other = "high" if key == "low" else "low"
            entry[other] = number
        return entry
    raise KeyError(f"a threshold has no control {key!r}.")


def view_from_state(state: Mapping[str, object]) -> View:
    """The view the page's controls describe.

    Parameters
    ----------
    state : mapping
        The control values by state name: ``dataset``, ``field`` (empty for none), ``component``
        (:data:`MAGNITUDE` for a vector's magnitude), ``colormap``, ``log_scale``, ``auto_range``
        with ``range_min`` / ``range_max``, ``surface``, ``surface_opacity``, ``edges``, and the
        entry lists ``slices`` and ``thresholds``.

    Returns
    -------
    View
    """
    field = state.get("field") or None
    component = state.get("component")
    manual = not state.get("auto_range", True)
    return View(
        dataset=str(state["dataset"]),
        field=field,
        component=None if component in (None, MAGNITUDE) else int(component),
        colormap=str(state.get("colormap", COLORMAPS[0])),
        log_scale=bool(state.get("log_scale", False)),
        color_range=(float(state["range_min"]), float(state["range_max"])) if manual else None,
        surface=bool(state.get("surface", True)),
        surface_opacity=float(state.get("surface_opacity", 1.0)),
        edges=bool(state.get("edges", False)),
        slices=tuple(
            Slice(entry["axis"], float(entry["coordinate"])) for entry in state.get("slices") or ()
        ),
        thresholds=tuple(
            Threshold(entry["field"], float(entry["low"]), float(entry["high"]))
            for entry in state.get("thresholds") or ()
            if entry.get("field")
        ),
    )
