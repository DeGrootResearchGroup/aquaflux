"""aquaflux's fluence rate on an adaptive octree against uniform grids, on the Sozzi reactor.

A uniform background grid (``background_grid.py``) spends almost all of its points where the field
is smooth; the error sits in the first millimetres around the lamp sleeve, and halving the spacing
everywhere multiplies the cost by eight to fix a band a few millimetres thick. This study samples
the same field with :func:`aquaflux.octree.sample_adaptively` -- split a cell, gather only its new
points, compare each with what the coarser tree predicted there, and split again where it missed by
more than the tolerance -- and compares it with uniform grids interpolated the same way.

**No mesh.** Everything comes from the reactor's own drawing (``SozziTaghipour.step``, the tutorial's,
read through :func:`aquaflux.io.cad.read_step`): the water is the three cylinders the drawing holds
less the lamp's cylinder and hemispherical tip, an exact :class:`aquaflux.solids.Solid`; the lamp's
emitting surface is the drawing's lamp triangulated (:data:`LAMP_CHORD`, :data:`LAMP_FACET`), its
base disc left out (it faces the end wall). 35 W at :data:`EXITANCE`, water absorbing
:data:`ABSORPTION`, every wall black, the fluid itself the only occluder (it shadows the pipes).

**The reference** is the gather itself, at random points of the water stratified by distance from the
sleeve's true surface (:data:`BANDS`): the error of any grid is its interpolated value against the
exact value at the same point, with no grid or mesh in the reference. Points right up to the sleeve
are included: a grid has to interpolate there too.

**The grids.** ``uniform_h<mm>``: a tree of root cells at that spacing, never split -- the same
interpolant, cut points and all, so the comparison isolates the adaptivity. ``adaptive_t<tol>``:
root cells at :data:`ADAPTIVE_SPACING`, refined to :data:`ADAPTIVE_LEVELS` levels at that tolerance on
the natural logarithm of ``G`` (a relative tolerance), floored at :data:`FLOOR`, with the library's
default standoff (a thousandth of the finest cell: the lamp is convex and its facets lie inside the
drawn sleeve, so any point of the drawn water is clear of them). Options, appended in any order:
``_m<level>`` (every cell split to that level first), ``_s<mm>`` (another standoff), ``_l<levels>``
(halved more often than :data:`ADAPTIVE_LEVELS`); a uniform arm takes ``_s<mm>``.

Stages, by argument or ``SOZZI_ADAPTIVE_STAGE``:

``ripple``
    How much the gathered field varies at a fixed distance from the sleeve, around it and along one
    facet's length: the structure the lamp's triangulation puts in the field next to it.

``reference``
    Gather at the reference points; kept in ``work/adaptive/reference.npz``.
``grids``
    Build each arm in ``SOZZI_ADAPTIVE_ARMS`` (default :data:`DEFAULT_ARMS`), interpolate it to the
    reference points, and keep its record in ``work/adaptive/<arm>.json`` (and the adaptive tree as
    ``<arm>.vtu``). An arm already recorded is skipped; delete its file to rebuild it.
``summary``
    Collect the records into ``work/adaptive/summary.json`` and print the table.

Run with ``SOZZI_ADAPTIVE_STAGE=<stage> validation/run_case.sh validation/sozzi_radiation/adaptive_grid.py``.
"""

from __future__ import annotations

import json
import os
import pickle
import platform
import sys
import time
from itertools import pairwise
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import aquaflux  # noqa: E402,F401  (enables x64)
import jax  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.io import write_hexahedra  # noqa: E402
from aquaflux.io.cad import Placement, read_step  # noqa: E402
from aquaflux.octree import Logarithm, sample_adaptively  # noqa: E402
from aquaflux.radiation import (  # noqa: E402
    NoOcclusion,
    RadiationSettings,
    Scene,
    Surfaces,
    UniformAbsorption,
    fluence_rate_at,
)
from aquaflux.solids import Difference, Union  # noqa: E402

OUT = HERE / "work" / "adaptive"
DRAWING = HERE.parent / "uvreactor_openfoam" / "of_case" / "SozziTaghipour.step"
#: The drawing has the body axis along y; the case along x.
SWAP_XY = Placement(matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]])
VESSEL = ("reactor_body", "inlet_pipe", "outlet_pipe")

EXITANCE = 696.42  # W/m^2 on the lamp, 35.4 W over its surface
ABSORPTION = 35.67  # 1/m, napierian (70 % UVT)
#: The lamp's triangulation: its facets lie up to the chord inside the true surface.
LAMP_CHORD, LAMP_FACET = 5e-5, 4e-3
#: Below this the logarithm is floored, so shadowed and nearly dark water sets no tolerance (W/m^2).
#: The field reaches 2 M = 1393 at the sleeve and its volume mean over the reactor is ~134, while the
#: pipes' is 0.01 (inlet) and 0.5 (riser): at a floor of 1e-3 the refinement chased every shadow edge
#: in the pipes to the finest level (misses of 10-17 in ln G), at seven times the cost, for water that
#: carries almost none of the dose.
FLOOR = 1.0
#: Bands of distance from the sleeve's true surface (m), and the reference points drawn in each.
BANDS = (0.0, 0.0005, 0.001, 0.002, 0.005, 0.010, 0.030, np.inf)
POINTS_PER_BAND = int(os.environ.get("SOZZI_ADAPTIVE_POINTS", 2000))
#: The adaptive arms' root cells, and how often each may be halved: 16 mm to 0.5 mm. Every root cell
#: holding water is split once (``min_level`` 1), so the coarsest cells are 8 mm -- the coarse uniform
#: grid -- and everything finer is the refinement's own choice.
ADAPTIVE_SPACING = 0.016
ADAPTIVE_LEVELS = 5
DEFAULT_ARMS = (
    "uniform_h8",
    "uniform_h4",
    "adaptive_t0.05_s0.25",
    "adaptive_t0.05",
    "uniform_h2",
)


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class Reactor:
    """The water, the lamp's body and its emitting surface, read from the drawing."""

    def __init__(self) -> None:
        cad = read_step(DRAWING, SWAP_XY)
        fluid = cad.fluid(*VESSEL)
        self.lamp_body = cad.solid("lamp")  # a cylinder and its hemispherical tip
        water = Union(*fluid.regions)
        for piece in self.lamp_body.bodies:
            water = Difference(water, piece)
        self.water = water
        self.fluid = fluid
        vessel = np.concatenate([cad.triangles(name, chord=1e-3).reshape(-1, 3) for name in VESSEL])
        self.low, self.high = vessel.min(axis=0), vessel.max(axis=0)
        triangles = cad.triangles("lamp", chord=LAMP_CHORD, facet_size=LAMP_FACET)
        on_base = np.all(np.abs(triangles[:, :, 0] - triangles[:, :, 0].min()) < 1e-9, axis=1)
        self.lamp = Surfaces.from_triangles(triangles[~on_base], emission=EXITANCE)
        self.scene = Scene(
            lamps=self.lamp,
            occluders=(fluid,),
            absorption=UniformAbsorption(ABSORPTION),
            settings=RadiationSettings(self_occlusion=NoOcclusion()),
        )

    def sleeve_distance(self, points) -> np.ndarray:
        """Distance from the sleeve's true surface, negative inside it."""
        return np.asarray(self.lamp_body.signed_distance(np.asarray(points)))

    def gather(self, points) -> np.ndarray:
        return np.asarray(fluence_rate_at(self.scene, np.asarray(points)))

    def record(self) -> dict:
        return {
            "drawing": DRAWING.name,
            "lamp_facets": int(self.lamp.n_facets),
            "lamp_power_W": float(np.sum(np.asarray(self.lamp.area))) * EXITANCE,
            "lamp_chord_m": LAMP_CHORD,
            "lamp_facet_m": LAMP_FACET,
            "exitance_W_per_m2": EXITANCE,
            "absorption_per_m": ABSORPTION,
            "walls": "black",
            "occluders": "the drawing's fluid (Outside of its three cylinders)",
            "self_occlusion": "none (the lamp is convex)",
            "floor_W_per_m2": FLOOR,
            "box_m": [self.low.tolist(), self.high.tolist()],
        }


def _environment() -> dict:
    return {
        "jax": jax.__version__,
        "machine": platform.machine(),
        "system": platform.system(),
        "cores": os.cpu_count(),
        "python": platform.python_version(),
    }


# -- ripple ---------------------------------------------------------------------------------------


def stage_ripple() -> None:
    reactor = Reactor()
    distances = np.array([1e-6, 1e-5, 3e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3])
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    along = np.linspace(0.400, 0.400 + LAMP_FACET, 9)
    radius = 0.010 + distances[:, None, None]
    points = np.stack(
        np.broadcast_arrays(along[None, :, None], radius * np.cos(angles), radius * np.sin(angles)),
        -1,
    )
    field = reactor.gather(points.reshape(-1, 3)).reshape(len(distances), -1)
    rows = []
    for distance, values in zip(distances, field, strict=True):
        rows.append({"distance_m": float(distance), "mean": float(values.mean()),
                     "relative_std": float(values.std() / values.mean()),
                     "relative_range": float(np.ptp(values) / values.mean())})  # fmt: skip
        _say(f"{distance * 1e3:7.3f} mm: mean {values.mean():8.2f}, relative std {rows[-1]['relative_std']:.2e}, "
             f"range {rows[-1]['relative_range']:.2e}")  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ripple.json").write_text(
        json.dumps({**reactor.record(), "rows": rows, **_environment()}, indent=2) + "\n"
    )


# -- reference -------------------------------------------------------------------------------------


def reference_points(reactor: Reactor, rng) -> tuple[np.ndarray, np.ndarray]:
    """``POINTS_PER_BAND`` points of the water in each band of :data:`BANDS`, uniform in volume."""
    points, band = [], []
    for index, (near, far) in enumerate(pairwise(BANDS)):
        if np.isfinite(far):
            # Around the lamp only: a box reaching ``far`` past the sleeve.
            reach = 0.010 + far
            low = np.array([0.0, -reach, -reach])
            high = np.array([0.81 + far, reach, reach])
        else:
            low, high = reactor.low, reactor.high
        kept: list[np.ndarray] = []
        while sum(map(len, kept)) < POINTS_PER_BAND:
            trial = rng.uniform(low, high, (20 * POINTS_PER_BAND, 3))
            distance = reactor.sleeve_distance(trial)
            inside = np.asarray(reactor.water.contains(trial))
            kept.append(trial[inside & (distance >= near) & (distance < far)])
        taken = np.concatenate(kept)[:POINTS_PER_BAND]
        points.append(taken)
        band.append(np.full(len(taken), index))
    return np.concatenate(points), np.concatenate(band)


def stage_reference() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reactor = Reactor()
    points, band = reference_points(reactor, np.random.default_rng(726))
    _say(
        f"{len(points)} reference points, {POINTS_PER_BAND} per band; lamp {reactor.lamp.n_facets} facets"
    )
    started = time.perf_counter()
    values = reactor.gather(points)
    seconds = time.perf_counter() - started
    _say(f"gathered in {seconds:.0f} s; G from {values.min():.3g} to {values.max():.4g}")
    np.savez(OUT / "reference.npz", points=points, band=band, G=values)
    (OUT / "reference.json").write_text(
        json.dumps({**reactor.record(), "points": len(points), "per_band": POINTS_PER_BAND,
                    "bands_m": [b if np.isfinite(b) else None for b in BANDS],
                    "gather_seconds": round(seconds, 1), **_environment()}, indent=2) + "\n"
    )  # fmt: skip


# -- grids -----------------------------------------------------------------------------------------


def _arms() -> list[str]:
    text = os.environ.get("SOZZI_ADAPTIVE_ARMS")
    return text.split() if text else list(DEFAULT_ARMS)


def build(arm: str, reactor: Reactor):
    """The tree of one arm, with the reactor's field sampled on it."""
    calls = {"seconds": 0.0, "points": 0}
    standoff = None  # the library's: a thousandth of the finest cell

    def sample(points):
        started = time.perf_counter()
        values = reactor.gather(points)
        calls["seconds"] += time.perf_counter() - started
        calls["points"] += len(points)
        return values

    # uniform_h<mm> or adaptive_t<tolerance>, then optionally _s<mm> for a standoff other than the
    # library's default; an adaptive arm also takes _m<min_level> to split every cell holding water
    # to that level first (_m2 starts from a uniform 4 mm grid) and _l<levels> for a tree halved more
    # often than ADAPTIVE_LEVELS.
    if arm.startswith("uniform_h"):
        spacing_text, *options = arm[len("uniform_h") :].split("_")
        spacing, levels, tolerance, min_level = float(spacing_text) * 1e-3, 0, np.inf, 0
    elif arm.startswith("adaptive_t"):
        tolerance_text, *options = arm[len("adaptive_t") :].split("_")
        spacing, levels, tolerance = ADAPTIVE_SPACING, ADAPTIVE_LEVELS, float(tolerance_text)
        min_level = 1
    else:
        raise ValueError(f"unknown arm {arm!r}")
    for option in options:
        if option.startswith("s"):
            standoff = float(option[1:]) * 1e-3
        elif option.startswith("m") and arm.startswith("adaptive"):
            min_level = int(option[1:])
        elif option.startswith("l") and arm.startswith("adaptive"):
            levels = int(option[1:])
        else:
            raise ValueError(f"unknown option {option!r} in arm {arm!r}")
    pad = 0.5 * spacing
    started = time.perf_counter()
    field = sample_adaptively(
        sample,
        reactor.water,
        low=reactor.low - pad,
        high=reactor.high + pad,
        spacing=spacing,
        max_level=levels,
        tolerance=tolerance,
        standoff=standoff,
        transform=Logarithm(FLOOR),
        min_level=min_level,
        report=lambda line: _say(f"  {arm}: {line}"),
    )
    total = time.perf_counter() - started
    settings = {"root_spacing_m": spacing, "max_level": levels, "min_level": min_level,
                "standoff_m": 1e-3 * spacing / 2**levels if standoff is None else standoff,
                "tolerance_ln_G": None if not np.isfinite(tolerance) else tolerance}  # fmt: skip
    return field, settings, total, calls


def errors(estimate, exact, band) -> dict:
    """``|estimate / exact - 1|`` by band, over the points lit above the floor."""
    out = {}
    lit = exact > FLOOR
    for index, (near, far) in enumerate(pairwise(BANDS)):
        rows = lit & (band == index)
        ratio = estimate[rows] / exact[rows]
        name = f"{near * 1e3:g}-{far * 1e3:g}mm" if np.isfinite(far) else f">{near * 1e3:g}mm"
        relative = np.abs(ratio - 1.0)
        out[name] = {"points": int(rows.sum()), "median": float(np.median(relative)),
                     "p90": float(np.percentile(relative, 90)), "p99": float(np.percentile(relative, 99)),
                     "max": float(relative.max()), "ratio_median": float(np.median(ratio))}  # fmt: skip
    return out


def write_tree(field, path: Path) -> None:
    """The tree's leaves holding water, as hexahedra with the sampled G at their corners."""
    points, corners, rows, values = field.hexahedra()
    write_hexahedra(
        points,
        corners,
        path,
        point_fields={"G": values},
        cell_fields={"level": field.tree.levels[rows].astype(float)},
    )


def stage_grids() -> None:
    reference = np.load(OUT / "reference.npz")
    reactor = Reactor()
    for arm in _arms():
        record_path = OUT / f"{arm}.json"
        if record_path.exists():
            _say(f"{arm}: recorded already ({record_path.name}); delete it to rebuild")
            continue
        _say(f"{arm}: building")
        field, settings, total, calls = build(arm, reactor)
        started = time.perf_counter()
        estimate = field(reference["points"])
        interpolate_seconds = time.perf_counter() - started
        result = errors(estimate, reference["G"], reference["band"])
        record = {
            "arm": arm,
            "leaves": int(field.tree.n_leaves),
            "nodes_sampled": int((~field.is_cut).sum()),
            "cut_points_sampled": int(field.is_cut.sum()),
            "samples": len(field.points),
            "leaves_by_level": np.bincount(
                field.tree.levels, minlength=field.tree.max_level + 1
            ).tolist(),
            "ghost_nodes": len(field.ghost_nodes),
            "seconds_total": round(total, 1),
            "seconds_gather": round(calls["seconds"], 1),
            "seconds_interpolate_reference": round(interpolate_seconds, 2),
            "passes": [r.__dict__ for r in field.rounds],
            "error_by_band": result,
            **reactor.record(),
            **settings,
            **_environment(),
        }
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        np.savez(OUT / f"{arm}_samples.npz", points=field.points, values=field.values,
                 misses=field.misses, is_cut=field.is_cut, reference_estimate=estimate)  # fmt: skip
        with open(OUT / f"{arm}_field.pickle", "wb") as file:
            pickle.dump(field, file)
        _say(f"{arm}: {record['samples']} samples ({record['cut_points_sampled']} cut), "
             f"{total:.0f} s ({calls['seconds']:.0f} s gathering)")  # fmt: skip
        for band, stats in result.items():
            _say(
                f"    {band:>10}: median {stats['median']:.2e}, p99 {stats['p99']:.2e}, max {stats['max']:.2e}"
            )
        if arm.startswith("adaptive"):
            write_tree(field, OUT / f"{arm}.vtu")


# -- summary ---------------------------------------------------------------------------------------


def stage_summary() -> None:
    records = {path.stem: json.loads(path.read_text()) for path in sorted(OUT.glob("*_*.json"))}
    records = {name: r for name, r in records.items() if "error_by_band" in r}
    (OUT / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    bands = list(next(iter(records.values()))["error_by_band"])
    print(
        "| arm | samples (cut) | time s (gather) | "
        + " | ".join(f"{b} median / p99" for b in bands)
        + " |"
    )
    print("|---" * (3 + len(bands)) + "|")
    for name, r in records.items():
        cells = " | ".join(
            f"{s['median']:.1e} / {s['p99']:.1e}" for s in r["error_by_band"].values()
        )
        print(f"| {name} | {r['samples']} ({r['cut_points_sampled']}) | {r['seconds_total']:.0f} "
              f"({r['seconds_gather']:.0f}) | {cells} |")  # fmt: skip


if __name__ == "__main__":
    stages = {"ripple": stage_ripple, "reference": stage_reference, "grids": stage_grids,
              "summary": stage_summary}  # fmt: skip
    stage = sys.argv[1] if len(sys.argv) == 2 else os.environ.get("SOZZI_ADAPTIVE_STAGE")
    if stage not in stages:
        sys.exit(f"usage: {Path(__file__).name} {{{'|'.join(stages)}}} (or SOZZI_ADAPTIVE_STAGE)")
    stages[stage]()
    sys.exit(0)
