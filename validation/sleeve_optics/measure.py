"""How much of the fluence rate in a lamp array comes from the sleeves' optics, and what ignoring them costs.

Four sleeved lamps on a square, in water, inside a black wall; every lamp emits one watt per metre from
its arc. The same array is traced four ways by ``tracer.py``:

- **full** -- every interface reflects and refracts (Fresnel, Snell, total internal reflection), on the
  lamp's own sleeve and on its neighbours';
- **own sleeve only** -- the neighbours' sleeves are not there, their arcs still absorb: what light
  reflected off or passed through a neighbour's sleeve adds, against full;
- **straight, from the arc** -- no interfaces anywhere, the arcs absorbing: aquaflux today with the arc
  as the emitter and the other arcs as occluders;
- **straight, from the sleeve** -- the sleeve's outer surface emits into the water and every sleeve is
  opaque: aquaflux today with the meshed sleeve as the lamp (the Sozzi case's ``lampWall``).

Reported per array: where the emitted power goes, the share of what the water absorbs by path class in
the full trace, and per pixel the full fluence rate against each simplified one, over the water between
the lamps and in a band beside the sleeves. **The control is the full trace again with another seed**:
its ratio to the first is the sampling noise every other ratio carries, so a spread no wider than the
control's says nothing.

Configuration by environment: ``SLEEVE_RAYS`` per lamp per trace (default 1,000,000), ``SLEEVE_SPACINGS``
centre-to-centre in metres (default ``0.05,0.08``), ``SLEEVE_UVTS`` in percent through 1 cm (default
``95,65``), ``SLEEVE_PIXEL`` in metres (default 0.002).
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

RESULT = HERE / "work" / "measure.json"
RAYS = int(os.environ.get("SLEEVE_RAYS", 1_000_000))
SPACINGS = tuple(float(s) for s in os.environ.get("SLEEVE_SPACINGS", "0.05,0.08").split(","))
UVTS = tuple(float(s) for s in os.environ.get("SLEEVE_UVTS", "95,65").split(","))
PIXEL = float(os.environ.get("SLEEVE_PIXEL", 0.002))
WALL = 0.15
BAND = 0.005  # the band beside the sleeves, metres


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def variants(base):
    from tracer import WATER

    return {
        "full": base,
        "full, another seed": base,
        "own sleeve only": dataclasses.replace(base, neighbours="invisible"),
        "straight, from the arc": dataclasses.replace(
            base, water=WATER, quartz=WATER, air=WATER, neighbours="invisible"
        ),
        "straight, from the sleeve": dataclasses.replace(
            base, emit_from="sleeve", neighbours="opaque"
        ),
    }


def regions(scene, grid):
    """Masks of the water between the lamps and of the band beside the sleeves, ``(ny, nx)`` each."""
    import numpy as np

    x0, y0, pixel, nx, ny = grid
    xs = x0 + (np.arange(nx) + 0.5) * pixel
    ys = y0 + (np.arange(ny) + 0.5) * pixel
    x, y = xs[None, :], ys[:, None]
    nearest = np.min(
        [np.hypot(x - c[0], y - c[1]) for c in scene.centres], axis=0
    )  # distance to the nearest lamp axis
    half = np.max(np.abs(scene.centres))
    between = (np.abs(x) < half) & (np.abs(y) < half) & (nearest > scene.outer + pixel)
    band = (nearest > scene.outer + pixel) & (nearest < scene.outer + BAND)
    return {"between the lamps": between, "within 5 mm of a sleeve": band}


def percentiles(values):
    import numpy as np

    values = values[np.isfinite(values)]
    return [round(float(np.percentile(values, q)), 4) for q in (5, 50, 95)]


def one_array(spacing, uvt):
    import numpy as np
    from tracer import CLASSES, Scene, fluence, trace, water_fraction

    absorption = -np.log(uvt / 100.0) * 100.0
    half = spacing / 2
    centres = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    base = Scene(centres=centres, wall=WALL, absorption=absorption)
    n = round(2 * WALL / PIXEL)
    grid = (-WALL, -WALL, PIXEL, n, n)
    fraction = water_fraction(base, grid)
    masks = regions(base, grid)
    fields, budgets = {}, {}
    for seed, (name, scene) in enumerate(variants(base).items()):
        started = time.perf_counter()
        tally = trace(scene, RAYS, grid, seed=seed)
        g = fluence(scene, tally, grid, fraction)
        fields[name] = g
        total = tally.emitted
        budgets[name] = {
            "water": tally.water / total,
            "arcs": tally.arcs / total,
            "sleeves (opaque)": tally.sleeves / total,
            "wall": tally.wall / total,
            "still in flight at the event cap": tally.lost / total,
            "unaccounted": 1.0
            - (tally.water + tally.arcs + tally.sleeves + tally.wall + tally.lost) / total,
            "seconds": round(time.perf_counter() - started, 1),
        }
        _say(f"  spacing {spacing} m, UVT {uvt}%: {name}: {budgets[name]}")
    full = fields["full"]
    full_total = np.nansum(full, axis=0)
    deposited = np.nansum(full, axis=(1, 2))
    shares = {cls: float(d / deposited.sum()) for cls, d in zip(CLASSES, deposited, strict=True)}
    comparisons = {}
    for name in (
        "full, another seed",
        "own sleeve only",
        "straight, from the arc",
        "straight, from the sleeve",
    ):
        other = np.nansum(fields[name], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = full_total / other
        comparisons[name] = {
            "water power, full over this": budgets["full"]["water"] / budgets[name]["water"],
            **{
                f"{region}: full over this, p5/p50/p95": percentiles(ratio[mask])
                for region, mask in masks.items()
            },
        }
    with np.errstate(invalid="ignore", divide="ignore"):
        neighbour_share = (full[1] + full[2]) / full_total
    row = {
        "spacing_m": spacing,
        "uvt_percent": uvt,
        "absorption_per_m": round(float(absorption), 3),
        "budgets": budgets,
        "full: share of the water's absorbed power by path": shares,
        **{
            f"full: neighbour-sleeve share of G, {region}, p5/p50/p95": percentiles(
                neighbour_share[mask]
            )
            for region, mask in masks.items()
        },
        "comparisons": comparisons,
    }
    _say(json.dumps({k: v for k, v in row.items() if k != "budgets"}))
    return row


def main() -> None:
    from tracer import AIR, QUARTZ, WATER, Scene

    base = Scene(centres=[[0.0, 0.0]])
    configuration = {
        "rays_per_lamp_per_trace": RAYS,
        "pixel_m": PIXEL,
        "wall_m": WALL,
        "lamps": "4 on a square, 1 W/m each from the arc, Lambertian",
        "arc_inner_outer_m": [base.arc, base.inner, base.outer],
        "indices_water_quartz_air": [WATER, QUARTZ, AIR],
        "quartz and air absorption": 0.0,
        "arcs": "absorb everything that reaches them",
    }
    _say(json.dumps(configuration))
    rows = [one_array(s, u) for s in SPACINGS for u in UVTS]
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps({"configuration": configuration, "rows": rows}, indent=2))
    _say(f"written {RESULT}")


if __name__ == "__main__":
    main()
