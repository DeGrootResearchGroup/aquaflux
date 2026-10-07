"""What is drawn, from what the user chose: a dataset's surface, slices through it, threshold regions.

:class:`View` is the user's choices as one value. :class:`Pipeline` turns a dataset and a view into
:class:`Layer` objects -- each a mesh to draw and how to colour it -- and keeps what it derived, so
moving a slider recomputes only the one piece that depends on it. :class:`Scene` puts layers into a
PyVista plotter. The first two never render anything, so the pipeline is tested without a display.

A large volume is never drawn as a volume. What is drawn is its outer surface, extracted once per
dataset; each slice, which cuts only the cells it crosses; and the outer surface of the cells each
threshold keeps. Each of those is a small fraction of the volume's cells.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Literal

import numpy as np
import pyvista as pv

from .theme import LIGHT, THEMES, RenderStyle

__all__ = [
    "AXES",
    "Layer",
    "Pipeline",
    "Scene",
    "Slice",
    "Threshold",
    "View",
    "automatic_range",
    "field_components",
    "field_values",
    "focus_values",
    "slice_origin",
]

#: The axes a slice can be normal to.
AXES: tuple[str, ...] = ("x", "y", "z")

#: The name of the outer-surface layer; slice and threshold layers are numbered (``slice 1``, ...).
SURFACE = "surface"

#: On a logarithmic colour scale, the lowest value shown, relative to the largest: a field that is
#: zero somewhere (a fluence rate in full shadow) has no smallest positive value worth scaling to.
LOG_DECADES = 6


@dataclasses.dataclass(frozen=True)
class Slice:
    """A plane through a dataset, normal to one axis.

    Attributes
    ----------
    axis : {"x", "y", "z"}
        The plane's normal.
    coordinate : float
        Where the plane crosses that axis, in the dataset's own coordinates.

    Raises
    ------
    ValueError
        If ``axis`` is not one of :data:`AXES`.
    """

    axis: Literal["x", "y", "z"]
    coordinate: float

    def __post_init__(self) -> None:
        if self.axis not in AXES:
            raise ValueError(f"Slice.axis is one of {AXES}, got {self.axis!r}.")


@dataclasses.dataclass(frozen=True)
class Threshold:
    """The cells of a dataset whose value of one field lies in a range.

    Attributes
    ----------
    field : str
        The field tested -- not necessarily the one the region is coloured by.
    low, high : float
        The range, inclusive at both ends.
    component : int or None
        For a vector field, the component tested; ``None`` tests its magnitude.

    Raises
    ------
    ValueError
        If ``low > high``.
    """

    field: str
    low: float
    high: float
    component: int | None = None

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(
                f"Threshold.low must not exceed high, got {self.low!r} > {self.high!r}."
            )


@dataclasses.dataclass(frozen=True)
class View:
    """What the user has chosen to see.

    Attributes
    ----------
    dataset : str
        The dataset, by its name in the snapshot.
    field : str or None
        The field it is coloured by; ``None`` draws it in one colour.
    component : int or None
        For a vector field, the component shown; ``None`` shows its magnitude.
    colormap : str
        A Matplotlib colormap name.
    log_scale : bool
        Colour by the logarithm of the field.
    color_range : tuple of (float, float) or None
        The values at the two ends of the colour scale; ``None`` takes them from the field.
    surface : bool
        Draw the dataset's outer surface. Beside a slice or a threshold region it is context, and is
        drawn in one plain colour.
    surface_opacity : float
        Its opacity, in ``[0, 1]``; lowered, a slice inside it can be seen.
    edges : bool
        Draw the cell edges.
    slices : tuple of Slice
        The planes drawn through the dataset, any number.
    thresholds : tuple of Threshold
        The threshold regions drawn, any number.

    Raises
    ------
    ValueError
        If the opacity is outside ``[0, 1]``.
    """

    dataset: str
    field: str | None = None
    component: int | None = None
    colormap: str = "viridis"
    log_scale: bool = False
    color_range: tuple[float, float] | None = None
    surface: bool = True
    surface_opacity: float = 1.0
    edges: bool = False
    slices: tuple[Slice, ...] = ()
    thresholds: tuple[Threshold, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.surface_opacity <= 1.0:
            raise ValueError(f"View.surface_opacity is in [0, 1], got {self.surface_opacity!r}.")

    @property
    def has_focus(self) -> bool:
        """Whether it draws anything beside the outer surface: a slice or a threshold region."""
        return bool(self.slices or self.thresholds)


@dataclasses.dataclass(frozen=True)
class Layer:
    """One mesh to draw, and how.

    Attributes
    ----------
    name : str
        Unique in a scene; drawing a layer of the same name replaces the one before it.
    mesh : pyvista.DataSet
        What is drawn.
    values : np.ndarray or None
        One value per cell (or per point, for point data) to colour by; ``None`` for one colour.
    opacity : float
        In ``[0, 1]``.
    """

    name: str
    mesh: pv.DataSet
    values: np.ndarray | None
    opacity: float = 1.0


def field_components(mesh: pv.DataSet) -> dict[str, int]:
    """The fields a dataset can be coloured by, with their number of components.

    Parameters
    ----------
    mesh : pyvista.DataSet
        The dataset.

    Returns
    -------
    dict of {str: int}
        Cell fields, then point fields, by name. A name held by both is the cell field.
    """
    found: dict[str, int] = {}
    for data in (mesh.cell_data, mesh.point_data):
        for name in data.keys():
            array = data[name]
            if name not in found and np.issubdtype(array.dtype, np.number):
                found[name] = 1 if array.ndim == 1 else int(array.shape[1])
    return found


def field_values(mesh: pv.DataSet, field: str, component: int | None = None) -> np.ndarray:
    """A field's values as one number per cell (or point): a scalar itself, a vector's component or magnitude.

    Parameters
    ----------
    mesh : pyvista.DataSet
        The dataset holding the field, as cell or point data.
    field : str
        Its name.
    component : int or None
        For a vector, the component; ``None`` for its magnitude.

    Returns
    -------
    np.ndarray of float, shape ``(n,)``
        ``n`` cells for a cell field, points for a point field.

    Raises
    ------
    KeyError
        If the dataset holds no field of that name.
    IndexError
        If ``component`` is not one the field has.
    """
    if field in mesh.cell_data:
        array = np.asarray(mesh.cell_data[field], dtype=float)
    elif field in mesh.point_data:
        array = np.asarray(mesh.point_data[field], dtype=float)
    else:
        raise KeyError(f"no field {field!r}; the dataset has {list(field_components(mesh))}.")
    if array.ndim == 1:
        return array
    if component is None:
        return np.linalg.norm(array, axis=1)
    if not 0 <= component < array.shape[1]:
        raise IndexError(f"{field!r} has {array.shape[1]} components, not a component {component}.")
    return array[:, component]


def automatic_range(values: np.ndarray, log_scale: bool = False) -> tuple[float, float] | None:
    """The colour range a field's own values give: its finite minimum and maximum.

    On a logarithmic scale the lower end is the smallest positive value, but no lower than
    ``10**-LOG_DECADES`` of the largest, so a field that is zero or nearly so over part of the
    domain does not stretch the scale over decades of nothing.

    Parameters
    ----------
    values : np.ndarray
        The values coloured by.
    log_scale : bool
        Whether the scale is logarithmic.

    Returns
    -------
    tuple of (float, float) or None
        ``(low, high)``, ``low < high`` -- widened about a constant field so the scale is not empty;
        ``None`` if there is no finite value, or no positive one on a logarithmic scale.
    """
    finite = values[np.isfinite(values)]
    if log_scale:
        finite = finite[finite > 0]
    if finite.size == 0:
        return None
    low, high = float(finite.min()), float(finite.max())
    if log_scale:
        low = max(low, high * 10.0**-LOG_DECADES)
        if low == high:
            return low / 10.0, high * 10.0
        return low, high
    if low == high:
        pad = abs(low) * 0.01 or 1.0
        return low - pad, high + pad
    return low, high


def focus_values(layers: list[Layer]) -> np.ndarray:
    """The values an automatic colour range is taken from: those of what the view is looking at.

    Slices and threshold regions are what a view that draws them is about; the outer surface beside
    them is context. So the range comes from every layer but the surface when there are any, and
    from the surface only when it is drawn alone. Taken from everything drawn, the surface's
    extremes would set the scale for the slices -- the walls beside a lamp are orders of magnitude
    brighter than the room it lights, and a slice through the room would use a sliver of the scale.

    Parameters
    ----------
    layers : list of Layer
        What is drawn.

    Returns
    -------
    np.ndarray
        The coloured values of the focus layers, concatenated; empty if none is coloured.
    """
    coloured = [layer for layer in layers if layer.values is not None]
    focus = [layer for layer in coloured if layer.name != SURFACE] or coloured
    return np.concatenate([layer.values for layer in focus]) if focus else np.empty(0)


def slice_origin(bounds: tuple[float, ...], plane: Slice) -> tuple[float, float, float]:
    """A point the plane passes through: the bounds' centre, moved along the plane's axis to it.

    Parameters
    ----------
    bounds : tuple of 6 float
        ``(x_min, x_max, y_min, y_max, z_min, z_max)``.
    plane : Slice
        The plane.

    Returns
    -------
    tuple of 3 float
    """
    centre = [0.5 * (bounds[2 * i] + bounds[2 * i + 1]) for i in range(3)]
    centre[AXES.index(plane.axis)] = float(plane.coordinate)
    return tuple(centre)


class Pipeline:
    """Turns a dataset and a :class:`View` into the layers to draw, keeping what it derived.

    Each derived mesh -- the outer surface, each slice, each threshold region -- is recomputed only
    when something it depends on has changed, and only what the latest view drew is kept, so a
    session holds a bounded amount beside the dataset itself. A dataset is recognized by identity,
    which is what a :class:`~aquaflux_ui.sources.ResultSource` promises for one that has not
    changed.
    """

    def __init__(self) -> None:
        self._derived: dict[str, tuple[tuple, pv.DataSet]] = {}

    def layers(self, mesh: pv.DataSet, view: View) -> list[Layer]:
        """The layers that draw ``view`` of ``mesh``.

        Parameters
        ----------
        mesh : pyvista.DataSet
            The chosen dataset.
        view : View
            What is to be seen of it.

        Returns
        -------
        list of Layer
            In drawing order: ``surface``, then ``slice 1``, ``slice 2``, ..., then ``threshold 1``,
            ..., numbered by position in the view. The surface is present when the view asks for
            it, a slice when it cuts a cell and a threshold when it keeps one.
        """
        wanted: dict[str, tuple[tuple, object]] = {}
        if view.surface:
            wanted[SURFACE] = ((id(mesh),), lambda: _outer_surface(mesh))
        for number, plane in enumerate(view.slices, start=1):
            origin = slice_origin(mesh.bounds, plane)
            wanted[f"slice {number}"] = (
                (id(mesh), plane.axis, origin),
                lambda plane=plane, origin=origin: mesh.slice(normal=plane.axis, origin=origin),
            )
        for number, region in enumerate(view.thresholds, start=1):
            wanted[f"threshold {number}"] = (
                (id(mesh), region),
                lambda region=region: _threshold(mesh, region),
            )

        # Keep only what this view draws; a removed slice's mesh is not held on to.
        self._derived = {name: self._derived[name] for name in wanted if name in self._derived}
        layers = []
        for name, (key, build) in wanted.items():
            derived = self._cached(name, key, build)
            if name == SURFACE:
                # Beside a slice or a threshold the surface is context: one plain colour.
                values = None if view.has_focus else _colour(derived, view)
                layers.append(Layer(name, derived, values, view.surface_opacity))
            elif derived.n_cells:
                layers.append(Layer(name, derived, _colour(derived, view)))
        return layers

    def _cached(self, name: str, key: tuple, build) -> pv.DataSet:
        held = self._derived.get(name)
        if held is None or held[0] != key:
            held = (key, build())
            self._derived[name] = held
        return held[1]


class Scene:
    """Draws layers into a PyVista plotter, replacing what it drew before.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Where they are drawn; usually off-screen, its frames sent to a browser.
    style : RenderStyle, optional
        The view's colours; unset, the light theme's. Assign :attr:`style` to change theme -- the
        next :meth:`show` draws in it.
    """

    def __init__(self, plotter: pv.Plotter, style: RenderStyle | None = None) -> None:
        self.plotter = plotter
        self.pipeline = Pipeline()
        self.style = style if style is not None else THEMES[LIGHT]
        self._shown: str | None = None
        self._drawn: tuple[str, ...] = ()

    def show(self, frame: Mapping[str, pv.DataSet], view: View) -> list[Layer]:
        """Draw ``view`` of its dataset in ``frame``, framing the camera when the dataset changes.

        Parameters
        ----------
        frame : mapping of {str: pyvista.DataSet}
            A snapshot's datasets.
        view : View
            What to draw.

        Returns
        -------
        list of Layer
            What was drawn.
        """
        mesh = frame[view.dataset]
        layers = self.pipeline.layers(mesh, view)
        self.plotter.set_background(self.style.background_bottom, top=self.style.background_top)
        values = [layer.values for layer in layers if layer.values is not None]
        clim = view.color_range or automatic_range(focus_values(layers), view.log_scale)
        for name in self._drawn:
            self.plotter.remove_actor(name, render=False)
        self._drawn = tuple(layer.name for layer in layers)
        if self.plotter.scalar_bars:
            self.plotter.remove_scalar_bar()
        for layer in layers:
            colour = layer.values is not None and clim is not None
            self.plotter.add_mesh(
                layer.mesh,
                name=layer.name,
                scalars=layer.values if colour else None,
                cmap=view.colormap,
                clim=clim if colour else None,
                log_scale=view.log_scale and colour,
                opacity=layer.opacity,
                show_edges=view.edges,
                color=self.style.context if not colour else None,
                smooth_shading=False,
                show_scalar_bar=False,
                render=False,
            )
        if values and clim is not None:
            self.plotter.add_scalar_bar(
                title=_bar_title(mesh, view),
                n_labels=5,
                fmt="%.3g",
                color=self.style.text,
                font_family="arial",
                title_font_size=16,
                label_font_size=13,
                bold=False,
                vertical=True,
                position_x=0.86,
                position_y=0.15,
                width=0.06,
                height=0.7,
            )
        if self._shown != view.dataset:
            self.plotter.view_isometric(render=False)
            self._shown = view.dataset
        return layers


def _bar_title(mesh: pv.DataSet, view: View) -> str:
    """The colour bar's title: the field, and for a vector which component or the magnitude."""
    if field_components(mesh).get(view.field, 1) == 1:
        return view.field
    return f"|{view.field}|" if view.component is None else f"{view.field}[{view.component}]"


def _colour(mesh: pv.DataSet, view: View) -> np.ndarray | None:
    """The values a derived mesh is coloured by, or ``None`` if the view colours by nothing."""
    if view.field is None or view.field not in field_components(mesh):
        return None
    return field_values(mesh, view.field, view.component)


def _outer_surface(mesh: pv.DataSet) -> pv.PolyData:
    """A dataset's outer surface, carrying its cell fields onto the faces; a surface is its own."""
    if isinstance(mesh, pv.PolyData):
        return mesh
    return mesh.extract_surface(algorithm="dataset_surface")


def _threshold(mesh: pv.DataSet, region: Threshold) -> pv.DataSet:
    """The outer surface of the cells whose value lies in the region's range."""
    values = field_values(mesh, region.field, region.component)
    keep = np.flatnonzero((values >= region.low) & (values <= region.high))
    if region.field in mesh.point_data and region.field not in mesh.cell_data:
        kept = mesh.extract_points(keep, adjacent_cells=False)
    else:
        kept = mesh.extract_cells(keep)
    return _outer_surface(kept) if kept.n_cells else kept
