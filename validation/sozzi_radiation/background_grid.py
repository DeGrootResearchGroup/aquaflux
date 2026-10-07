"""aquaflux's fluence rate on a Cartesian background grid, interpolated onto the Sozzi CFD mesh.

The direct gather costs the same at every receiver, so its cost is set by how many points it is
asked for. On the wall-resolved uvmesh mesh (1,232,629 cells) every cell centre is a receiver; most
of those cells are there for the flow's wall layers, not because the fluence rate needs them. This
study gathers ``G`` only at the nodes of a uniform Cartesian grid, interpolates it trilinearly to
the cell centres, and asks, for a series of grid spacings, how far that field is from the gather at
every cell centre (``uvmesh_swap.py``'s ``swap/G_aquaflux.npy``) and what the difference is worth in
dose, with the tracker following the same particles as the swap set.

Which nodes are gathered. The grid covers the fluid's bounding box with spacing ``h`` and an origin
at the box's low corner. A node is gathered when it is a corner of a grid cell that holds a CFD cell
centre and it lies in the water: inside one of the three cylinders (``compare_fluence``'s chamber,
inlet pipe and riser) and outside the lamp sleeve (radius 10 mm, from x = 0 to 0.80 m, closed by a
hemispherical tip). Each node is gathered as a cell at that point would be: unmasked in the chamber,
through ``compare_fluence.BranchOpenings`` in the pipes.

How a cell centre is interpolated. Trilinear weights from the eight corners of its grid cell; a
corner outside the water (in the sleeve, or beyond a wall) is dropped and the remaining weights are
rescaled to sum to one, so a cell beside the sleeve takes its value from the water side only. A cell
whose eight corners are all dry takes its nearest wet node's value. The interpolated value is
therefore always a convex combination of gathered values -- which, beside the sleeve, is below the
true value, since every wet corner there is farther from the lamp than the cell. Three variants
(``VARIANTS``) take the same node values to the cells: ``G_average`` as just described;
``logG_average``, the same on log G; and ``logG_extrapolated``, log G with a one-sided linear fit
where a cell has dry corners and at least four wet ones not in one plane, which extends the wet
corners' gradient toward the sleeve (capped at ``G_CAP``, the most G the water can hold). A fourth, ``logG_surface``,
adds to that fit the value G takes on the sleeve itself (:class:`SleeveSurface`: ``2 M``), at the
nearest surface point of every corner inside the sleeve, so a cell beside it is fitted between known
values on both sides rather than extrapolated. The interpolate
stage checks the interpolator before using it: it must reproduce a linear field exactly at cells
whose corners are all wet and at every fitted cell, and stay within the corners' range elsewhere.

Stages, run as ``python background_grid.py <stage>`` or
``SOZZI_GRID_STAGE=<stage> validation/run_case.sh validation/sozzi_radiation/background_grid.py``:

``gather``
    For each spacing in ``SOZZI_GRID_SPACINGS`` (metres, space-separated; default
    ``0.008 0.006 0.004 0.003 0.002``): build the grid, gather ``G`` at its wet nodes with the
    swap set's lamp (the mesh's own lamp patches coarsened as ``uvmesh_swap.mesh_lamp`` does),
    and keep the nodes and the gather's time under ``grid/h<mm>/``.
``interpolate``
    Take each grid's saved node values to the cell centres by every variant, after checking the
    interpolator; no gathering, so a variant can be added or changed without one.
``write``
    One tracking case per spacing and variant, ``grid/h<mm>/<variant>``, with its ``G`` written as
    ``uvmesh_swap.tracking_case`` writes aquaflux's: interior values, ``zeroGradient`` patches.
``compare``
    After the tracker has run in each case: the field's error against the gather at every cell
    centre (volume-weighted means by region, the ratio's percentiles over lit cells and in bands of
    distance from the sleeve, and ``integral(G dV) / Q``), and the dose against ``swap/aquaflux``,
    particle by particle (the runs must follow the same paths), to ``grid/summary.json``.

Paths as ``uvmesh_swap.py`` (``SOZZI_UVMESH_RUN``); this study writes under ``<run>/grid``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import compare_fluence  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.io import read_openfoam  # noqa: E402
from compare_fluence import EXITANCE, BranchOpenings, gather  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from uvmesh_swap import (  # noqa: E402
    FLOW_RATE,
    GATHER_CHUNK,
    K_INACT,
    MESH,
    RUN,
    TO_MJ_PER_CM2,
    _say,
    fluid_region,
    log_reduction,
    mesh_lamp,
    tracking_case,
    tracks,
)
from uvmesh_swap import OUT as SWAP  # noqa: E402

OUT = RUN / "grid"
DEFAULT_SPACINGS = (0.008, 0.006, 0.004, 0.003, 0.002)
#: The lamp sleeve: its radius, and the axial position of its hemispherical tip's centre.
SLEEVE_RADIUS, SLEEVE_TIP_X = 0.010, 0.80
#: Bands of distance from the sleeve's surface (m) for the field's error.
DISTANCE_BANDS = (0.0, 0.001, 0.002, 0.005, 0.010, 0.020, np.inf)
#: The largest fluence rate the water can hold: at the surface of a convex, diffusely emitting lamp
#: G is at most its radiance M / pi over a hemisphere, 2 M, and absorption only lowers it. An
#: extrapolated value is capped here.
G_CAP = 2.0 * EXITANCE
#: Determinant of a cell's normal equations (corner offsets in cell units) below which its wet
#: corners are taken as coplanar and a linear fit as undetermined.
FIT_DETERMINANT = 1e-6
#: The eight corners of a grid cell, as offsets from its low corner.
CORNERS = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)])


def sleeve_distance(points: np.ndarray) -> np.ndarray:
    """Signed distance from the sleeve's surface, negative inside it (the cylinder and its tip)."""
    along = np.minimum(points[:, 0], SLEEVE_TIP_X)
    axis_point = np.column_stack([along, np.zeros((len(points), 2))])
    return np.linalg.norm(points - axis_point, axis=1) - SLEEVE_RADIUS


def wet(points: np.ndarray) -> np.ndarray:
    """In the water: in one of the three cylinders and outside the sleeve."""
    in_fluid = ~np.asarray(BranchOpenings().contains(jnp.asarray(points)))
    return in_fluid & (sleeve_distance(points) > 0.0)


class SleeveSurface:
    """The lamp sleeve's surface, where the fluence rate is known without gathering.

    At a point just outside a diffusely emitting surface, the radiance leaving it is ``M / pi``
    over the whole outward hemisphere, which contributes ``2 M`` to ``G`` before any path through
    the water; what arrives from the other hemisphere is light from elsewhere. The sleeve is convex
    and the walls are black, so nothing arrives and ``G = 2 M`` all over it.
    """

    value = 2.0 * EXITANCE

    @staticmethod
    def holds(points: np.ndarray) -> np.ndarray:
        """Which points (n, 3) are inside the sleeve, so have a nearest point on its surface."""
        return sleeve_distance(points) <= 0.0

    @staticmethod
    def nearest(points: np.ndarray) -> np.ndarray:
        """The nearest point (n, 3) on the sleeve's surface to each point inside it: radially out
        from the axis along the cylinder, out from the tip's centre past it. A point on the axis
        goes out along +y."""
        along = np.minimum(points[:, 0], SLEEVE_TIP_X)
        axis_point = np.column_stack([along, np.zeros((len(points), 2))])
        out = points - axis_point
        length = np.linalg.norm(out, axis=1)
        on_axis = length == 0.0
        out[on_axis] = (0.0, 1.0, 0.0)
        length[on_axis] = 1.0
        return axis_point + SLEEVE_RADIUS * out / length[:, None]


class BackgroundGrid:
    """A uniform Cartesian grid: nodes at ``origin + h (i, j, k)``, numbered by one integer key.

    Parameters
    ----------
    origin : array_like, shape (3,)
        The node with index (0, 0, 0).
    spacing : float
        ``h``, the same along every axis.
    shape : array_like of int, shape (3,)
        Nodes along each axis.
    """

    def __init__(self, origin, spacing: float, shape) -> None:
        self.origin = np.asarray(origin, dtype=float)
        self.spacing = float(spacing)
        self.shape = np.asarray(shape, dtype=np.int64)

    @classmethod
    def covering(cls, points: np.ndarray, spacing: float) -> BackgroundGrid:
        """The grid of spacing ``h`` whose cells cover ``points`` (shape (n, 3)), one cell to spare."""
        low, high = points.min(axis=0), points.max(axis=0)
        shape = np.floor((high - low) / spacing).astype(np.int64) + 2
        return cls(low, spacing, shape)

    def cell_of(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each point's grid cell, as its low corner's index (n, 3), and the point's fractional
        position in that cell (n, 3), each component in [0, 1)."""
        scaled = (points - self.origin) / self.spacing
        low = np.clip(np.floor(scaled).astype(np.int64), 0, self.shape - 2)
        return low, scaled - low

    def key(self, index: np.ndarray) -> np.ndarray:
        """The integer key of each node index (..., 3)."""
        return (index[..., 0] * self.shape[1] + index[..., 1]) * self.shape[2] + index[..., 2]

    def position(self, keys: np.ndarray) -> np.ndarray:
        """The positions (n, 3) of the nodes with these keys."""
        k = keys % self.shape[2]
        j = (keys // self.shape[2]) % self.shape[1]
        i = keys // (self.shape[1] * self.shape[2])
        return self.origin + self.spacing * np.column_stack([i, j, k])

    def corner_keys(self, points: np.ndarray) -> np.ndarray:
        """The keys (n, 8) of the eight corners of each point's grid cell, in ``CORNERS`` order."""
        low, _ = self.cell_of(points)
        return self.key(low[:, None, :] + CORNERS[None, :, :])


class WetTrilinear:
    """Trilinear interpolation from a grid's wet nodes, dry corners dropped and weights rescaled.

    Parameters
    ----------
    grid : BackgroundGrid
    keys : numpy.ndarray of int, shape (m,)
        Sorted keys of the nodes that carry a value (wet or not).
    is_wet : numpy.ndarray of bool, shape (m,)
        Which of those nodes are in the water; only these are used.
    """

    def __init__(self, grid: BackgroundGrid, keys: np.ndarray, is_wet: np.ndarray) -> None:
        self.grid = grid
        self.keys = keys
        self.is_wet = is_wet
        wet_keys = keys[is_wet]
        self._wet_slots = np.flatnonzero(is_wet)
        self._tree = cKDTree(grid.position(wet_keys))

    def corners(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each point: the slots (n, 8) of its grid cell's corners in ``keys``, in ``CORNERS``
        order, their trilinear weights (n, 8), and which of them are wet (n, 8)."""
        _, fraction = self.grid.cell_of(points)
        keys = self.grid.corner_keys(points)
        slots = np.searchsorted(self.keys, keys)
        found = (slots < len(self.keys)) & (
            self.keys[np.minimum(slots, len(self.keys) - 1)] == keys
        )
        if not found.all():
            raise ValueError(f"{int((~found).sum())} corner nodes were not gathered")
        f = fraction[:, None, :]
        trilinear = np.prod(np.where(CORNERS[None, :, :] == 1, f, 1.0 - f), axis=2)
        return slots, trilinear, self.is_wet[slots]

    def weights(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each point: the slots (n, 8) of the nodes its value is taken from, their weights
        (n, 8), and how many of its corners are wet (n,). The nodes are its cell's corners, unless
        its wet corners all have zero weight (none wet, or the point on a face whose corners are
        dry): then its nearest wet node, with weight one in the first column."""
        slots, trilinear, usable = self.corners(points)
        w = np.where(usable, trilinear, 0.0)
        total = w.sum(axis=1)
        none = total <= 0.0
        w = np.where(none[:, None], 0.0, w / np.where(none, 1.0, total)[:, None])
        if none.any():
            _, nearest = self._tree.query(points[none])
            slots = slots.copy()
            slots[none, 0] = self._wet_slots[nearest]
            w[none, 0] = 1.0
        return slots, w, usable.sum(axis=1)

    def __call__(self, values: np.ndarray, points: np.ndarray) -> np.ndarray:
        """``values`` (m,), one per key, interpolated to ``points`` (n, 3)."""
        slots, w, _ = self.weights(points)
        return np.sum(values[slots] * w, axis=1)

    def fitted(
        self, values: np.ndarray, points: np.ndarray, surface=None, surface_value=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """As :meth:`__call__`, except at a point whose cell has dry corners: there, the value at
        the point of a least-squares linear fit to the cell's wet corners, which extends their
        gradient toward the dry side instead of averaging it away.

        With a ``surface`` (holding ``holds`` and ``nearest``, as :class:`SleeveSurface`) and a
        ``surface_value`` (a function of positions (k, 3), returning (k,)), every dry corner the
        surface holds adds a datum at its nearest surface point, so a cell beside the surface is
        fitted between known values on both sides. The fit needs four data not in one plane; a
        cell short of them keeps its interpolated value. Returns the values (n,) and which points
        were fitted (n,).
        """
        out = self(values, points)
        slots, _, usable = self.corners(points)
        candidate = np.flatnonzero(~usable.all(axis=1))
        fitted = np.zeros(len(points), bool)
        if candidate.size == 0:
            return out, fitted
        h = self.grid.spacing
        point = points[candidate]
        corner_position = self.grid.position(self.keys[slots[candidate]].ravel()).reshape(-1, 8, 3)
        datum_position = corner_position
        datum_value = values[slots[candidate]]
        use = usable[candidate]
        if surface is not None:
            on = ~use & surface.holds(corner_position.reshape(-1, 3)).reshape(-1, 8)
            projected = surface.nearest(corner_position[on])
            boundary_position = np.zeros_like(corner_position)
            boundary_position[on] = projected
            boundary_value = np.zeros(on.shape)
            boundary_value[on] = surface_value(projected)
            datum_position = np.concatenate([corner_position, boundary_position], axis=1)
            datum_value = np.concatenate([datum_value, boundary_value], axis=1)
            use = np.concatenate([use, on], axis=1)
        offset = (datum_position - point[:, None, :]) / h
        design = np.concatenate([np.ones((*offset.shape[:2], 1)), offset], axis=2)
        weight = use.astype(float)
        normal = np.einsum("nci,ncj,nc->nij", design, design, weight)
        rhs = np.einsum("nci,nc,nc->ni", design, datum_value, weight)
        determined = (use.sum(axis=1) >= 4) & (np.linalg.det(normal) > FIT_DETERMINANT)
        coefficients = np.linalg.solve(normal[determined], rhs[determined][..., None])[..., 0]
        fitted[candidate[determined]] = True
        out[fitted] = coefficients[:, 0]
        return out, fitted


@dataclass(frozen=True)
class FieldVariant:
    """One way of taking the gathered node values to the cells.

    ``log``: interpolate log G rather than G (G falls off from the lamp roughly as an exponential
    over a distance, which log G follows closely and G does not). ``fit``: where a cell has dry
    corners, fit the wet ones (:meth:`WetTrilinear.fitted`) instead of averaging them. ``surface``:
    add the sleeve's known value (:class:`SleeveSurface`) to that fit. A fitted value is capped at
    ``G_CAP``.
    """

    name: str
    log: bool
    fit: bool = False
    surface: bool = False

    def apply(
        self, interpolate: WetTrilinear, values: np.ndarray, points: np.ndarray
    ) -> np.ndarray:
        """``G`` (m,) at the nodes taken to ``points`` (n, 3)."""
        floor = 1e-12 * float(values.max())
        forward = (lambda g: np.log(np.maximum(g, floor))) if self.log else (lambda g: g)
        node = forward(values)
        if self.fit:
            surface = SleeveSurface() if self.surface else None
            value = (
                (lambda x: forward(np.full(len(x), SleeveSurface.value))) if self.surface else None
            )
            cell, fitted = interpolate.fitted(node, points, surface, value)
        else:
            cell, fitted = interpolate(node, points), np.zeros(len(points), bool)
        field = np.exp(cell) if self.log else cell
        return np.where(fitted, np.minimum(field, G_CAP), field)


VARIANTS = (
    FieldVariant("G_average", log=False),
    FieldVariant("logG_average", log=True),
    FieldVariant("logG_extrapolated", log=True, fit=True),
    FieldVariant("logG_surface", log=True, fit=True, surface=True),
)


def check_interpolator(interpolate: WetTrilinear, points: np.ndarray) -> dict:
    """The interpolator must be exact for a linear field where all eight corners are wet, and give
    a convex combination of its corners' values everywhere else; its fits, with and without surface
    data, must be exact for a linear field wherever they fit. Refuse to go on otherwise."""
    nodes = interpolate.grid.position(interpolate.keys)
    linear = 3.0 + nodes @ np.array([20.0, -7.0, 11.0])
    slots, w, n_wet = interpolate.weights(points)
    got = np.sum(linear[slots] * w, axis=1)
    expected = 3.0 + points @ np.array([20.0, -7.0, 11.0])
    full = n_wet == 8
    error = float(np.max(np.abs(got[full] - expected[full]))) if full.any() else 0.0
    if error > 1e-9:
        raise RuntimeError(f"trilinear interpolation of a linear field is off by {error:.3g}")
    used = w > 0
    low = np.min(np.where(used, linear[slots], np.inf), axis=1)
    high = np.max(np.where(used, linear[slots], -np.inf), axis=1)
    if np.any(got < low - 1e-9) or np.any(got > high + 1e-9):
        raise RuntimeError("an interpolated value lies outside the range of the corners it used")
    _, trilinear, usable = interpolate.corners(points)
    nearest = int((np.where(usable, trilinear, 0.0).sum(axis=1) <= 0.0).sum())
    fit, fitted = interpolate.fitted(linear, points)
    fit_error = float(np.max(np.abs(fit[fitted] - expected[fitted]))) if fitted.any() else 0.0
    if fit_error > 1e-9:
        raise RuntimeError(f"the one-sided fit of a linear field is off by {fit_error:.3g}")
    inside = nodes[SleeveSurface.holds(nodes)]
    off_surface = (
        float(np.max(np.abs(sleeve_distance(SleeveSurface.nearest(inside)))))
        if len(inside)
        else 0.0
    )
    if off_surface > 1e-12:
        raise RuntimeError(f"a projected sleeve point is {off_surface:.3g} m off the sleeve")
    field = lambda x: 3.0 + x @ np.array([20.0, -7.0, 11.0])  # noqa: E731
    bounded, with_surface = interpolate.fitted(linear, points, SleeveSurface(), field)
    surface_error = (float(np.max(np.abs(bounded[with_surface] - expected[with_surface])))
                     if with_surface.any() else 0.0)  # fmt: skip
    if surface_error > 1e-9:
        raise RuntimeError(
            f"the fit with surface data, of a linear field, is off by {surface_error:.3g}"
        )
    return {"all_eight_wet": int(full.sum()), "some_dry": int(((n_wet > 0) & ~full).sum()),
            "nearest_node": nearest, "fitted_one_sided": int(fitted.sum()),
            "fitted_with_surface": int(with_surface.sum()),
            "linear_field_max_error": error, "linear_field_fit_max_error": fit_error,
            "linear_field_surface_fit_max_error": surface_error}  # fmt: skip


def _spacings() -> list[float]:
    text = os.environ.get("SOZZI_GRID_SPACINGS")
    return [float(v) for v in text.split()] if text else list(DEFAULT_SPACINGS)


def _label(spacing: float) -> str:
    return f"h{spacing * 1e3:g}mm"


# -- gather ---------------------------------------------------------------------------------------


def stage_gather() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _say("reading the mesh")
    started = time.perf_counter()
    mesh = read_openfoam(MESH)
    geometry = mesh.geometry()
    cells = np.asarray(geometry.cell.centroid)
    read_seconds = time.perf_counter() - started
    _say(f"{len(cells)} cells, read in {read_seconds:.0f} s")
    lamp, lamp_record = mesh_lamp(mesh, geometry)
    compare_fluence.CHUNK = GATHER_CHUNK
    for spacing in _spacings():
        label = _label(spacing)
        grid = BackgroundGrid.covering(cells, spacing)
        started = time.perf_counter()
        keys = np.unique(grid.corner_keys(cells))
        nodes = grid.position(keys)
        is_wet = wet(nodes)
        interpolate = WetTrilinear(grid, keys, is_wet)
        check = check_interpolator(interpolate, cells)
        setup_seconds = time.perf_counter() - started
        _say(f"{label}: grid {grid.shape.tolist()}, {len(keys)} corner nodes, {int(is_wet.sum())} wet; "
             f"cells with all eight corners wet {check['all_eight_wet']}, some dry "
             f"{check['some_dry']}, none wet {check['nearest_node']}")  # fmt: skip
        values = np.zeros(len(keys))
        started = time.perf_counter()
        values[is_wet] = gather(lamp, nodes[is_wet], fluid_region(nodes[is_wet]), label)
        gather_seconds = time.perf_counter() - started
        where = OUT / label
        where.mkdir(parents=True, exist_ok=True)
        np.savez(where / "nodes.npz", keys=keys, position=nodes, wet=is_wet, G=values)
        (where / "gather.json").write_text(json.dumps({
            "spacing_m": spacing,
            "origin_m": grid.origin.tolist(),
            "shape": grid.shape.tolist(),
            "corner_nodes": len(keys),
            "wet_nodes": int(is_wet.sum()),
            "cells": len(cells),
            "interpolation": check,
            **lamp_record,
            "gather_chunk_points": GATHER_CHUNK,
            "mesh_read_seconds": round(read_seconds, 1),
            "grid_setup_seconds": round(setup_seconds, 1),
            "gather_seconds": round(gather_seconds, 1),
            "jax": __import__("jax").__version__,
        }, indent=2) + "\n")  # fmt: skip
        _say(f"{label}: gathered in {gather_seconds:.0f} s")
    np.save(OUT / "cells.npy", cells)


# -- interpolate ----------------------------------------------------------------------------------


def stage_interpolate() -> None:
    cells = np.load(OUT / "cells.npy")
    for spacing in _spacings():
        label = _label(spacing)
        where = OUT / label
        nodes = np.load(where / "nodes.npz")
        grid = BackgroundGrid.covering(cells, spacing)
        if not np.array_equal(grid.position(nodes["keys"]), nodes["position"]):
            raise RuntimeError(f"{label}: the saved nodes are not this grid's")
        interpolate = WetTrilinear(grid, nodes["keys"], nodes["wet"])
        check = check_interpolator(interpolate, cells)
        seconds = {}
        for variant in VARIANTS:
            started = time.perf_counter()
            field = variant.apply(interpolate, nodes["G"], cells)
            seconds[variant.name] = round(time.perf_counter() - started, 1)
            np.save(where / f"G_cells_{variant.name}.npy", field)
            _say(f"{label} {variant.name}: interpolated in {seconds[variant.name]} s; "
                 f"G at the cells {field.min():.3g} to {field.max():.4g}")  # fmt: skip
        (where / "interpolate.json").write_text(
            json.dumps({"interpolation": check, "interpolate_seconds": seconds}, indent=2) + "\n"
        )


# -- write ----------------------------------------------------------------------------------------


def stage_write() -> None:
    for spacing in _spacings():
        where = OUT / _label(spacing)
        for variant in VARIANTS:
            case = where / variant.name
            tracking_case(case, values=np.load(where / f"G_cells_{variant.name}.npy"))
            _say(f"{_label(spacing)} {variant.name}: case written at {case}")


# -- compare --------------------------------------------------------------------------------------


def field_error(
    field: np.ndarray, exact: np.ndarray, volume: np.ndarray, cells: np.ndarray
) -> dict:
    """The interpolated field against the gather at every cell centre."""
    region = fluid_region(cells)
    out = {"volume_mean": {}, "mean_dose_from_integral": float(
        np.sum(np.maximum(field, 0.0) * volume)) / FLOW_RATE * TO_MJ_PER_CM2}  # fmt: skip
    for label, mask in (("all", np.ones(len(cells), bool)), ("chamber", region == 0),
                        ("inlet", region == 1), ("riser", region == 2)):  # fmt: skip
        weights = volume[mask]
        out["volume_mean"][label] = float(np.sum(field[mask] * weights) / np.sum(weights))
    lit = exact > 1e-3 * exact.max()
    ratio = field[lit] / exact[lit]
    out["ratio_over_lit_cells"] = {
        f"p{q}": float(np.percentile(ratio, q)) for q in (1, 10, 50, 90, 99)
    }
    distance = sleeve_distance(cells)
    bands = {}
    for low, high in pairwise(DISTANCE_BANDS):
        band = lit & (distance >= low) & (distance < high)
        if band.any():
            r = field[band] / exact[band]
            name = f"{low * 1e3:g}-{high * 1e3:g}mm" if np.isfinite(high) else f">{low * 1e3:g}mm"
            bands[name] = {"cells": int(band.sum()),
                           **{f"p{q}": float(np.percentile(r, q)) for q in (1, 50, 99)}}  # fmt: skip
    out["ratio_by_sleeve_distance"] = bands
    return out


def stage_compare() -> None:
    _say("reading the mesh for the field statistics")
    geometry = read_openfoam(MESH).geometry()
    cells = np.asarray(geometry.cell.centroid)
    volume = np.asarray(geometry.cell.volume)
    exact = np.load(SWAP / "G_aquaflux.npy")
    reference = tracks(SWAP / "aquaflux")
    escaped = reference["reason"] == "escaped"
    ours = reference["dose"][escaped]
    exact_stats = field_error(exact, exact, volume, cells)
    summary = {"reference": {"mean_dose": float(ours.mean()),
                             "log_reduction": {f"{k:g}": log_reduction(ours, k) for k in K_INACT},
                             "volume_mean": exact_stats["volume_mean"],
                             "mean_dose_from_integral": exact_stats["mean_dose_from_integral"],
                             "gather": json.loads((SWAP / "gather.json").read_text())},
               "grids": {}}  # fmt: skip
    for spacing in _spacings():
        label = _label(spacing)
        where = OUT / label
        record = json.loads((where / "gather.json").read_text())
        interpolated = json.loads((where / "interpolate.json").read_text())
        grid = {"spacing_m": spacing, "wet_nodes": record["wet_nodes"],
                "gather_seconds": record["gather_seconds"],
                "interpolation": interpolated["interpolation"], "variants": {}}  # fmt: skip
        for variant in VARIANTS:
            run = tracks(where / variant.name)
            for key in ("id", "reason", "time", "end"):
                if not np.array_equal(reference[key], run[key]):
                    raise RuntimeError(f"{label} {variant.name} and swap/aquaflux disagree on "
                                       f"'{key}': not paired")  # fmt: skip
            dose = run["dose"][escaped]
            lr = {f"{k:g}": log_reduction(dose, k) for k in K_INACT}
            field = np.load(where / f"G_cells_{variant.name}.npy")
            g = grid["variants"][variant.name] = {
                "interpolate_seconds": interpolated["interpolate_seconds"][variant.name],
                "field": field_error(field, exact, volume, cells),
                "mean_dose": float(dose.mean()),
                "log_reduction": lr,
                "log_reduction_over_reference": {k: v / summary["reference"]["log_reduction"][k]
                                                 for k, v in lr.items()},
                "paired_dose_ratio": {f"p{q}": float(np.percentile(dose / ours, q))
                                      for q in (1, 10, 50, 90, 99)},
            }  # fmt: skip
            _say(f"{label} {variant.name}: mean dose {g['mean_dose']:.2f} "
                 f"({summary['reference']['mean_dose']:.2f}); LR / reference "
                 + ", ".join(f"k={k}: {v:.4f}" for k, v in g["log_reduction_over_reference"].items()))  # fmt: skip
        summary["grids"][label] = grid
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _say(f"wrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    stages = {"gather": stage_gather, "interpolate": stage_interpolate, "write": stage_write,
              "compare": stage_compare}  # fmt: skip
    stage = sys.argv[1] if len(sys.argv) == 2 else os.environ.get("SOZZI_GRID_STAGE")
    if stage not in stages:
        sys.exit(f"usage: {Path(__file__).name} {{{'|'.join(stages)}}} (or SOZZI_GRID_STAGE)")
    stages[stage]()
    sys.exit(0)
