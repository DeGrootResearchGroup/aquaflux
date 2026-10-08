"""aquaflux's scene with transparent solids against the tracer, for four sleeved lamps in water.

Four lamps -- arc, air gap, quartz sleeve, as in ``tracer.py`` -- on the corners of a square, each
emitting one watt per metre from its arc, in water inside a black wall. The tracer follows every
interface of every sleeve. aquaflux is :func:`aquaflux.radiation.solve_scene` given the four sleeves as
:class:`aquaflux.radiation.refraction.Media`: each arc's light reaches a point in the water along its
refracted path out of its own sleeve, and a neighbour's sleeve that path passes is crossed **straight**,
its Fresnel losses and absorption taken at the straight line's angles, the path not bent there. The
neighbours' arcs shadow the light, through the ray test of the lamps' own triangles.

As in ``check_refraction.py``, aquaflux follows no reflection, so it is compared with the trace that
ends every ray at its first reflection (``reflections=False``); the full trace is shown beside it for
what the reflections left out are worth. What this check adds to that one is the neighbours: at a
point whose light from some lamp passes another lamp's sleeve, the difference between aquaflux and the
transmitted trace is the straight-through approximation's error, and the tracer's own split of the light
by path (``CLASSES``) says how much of the point's light that is.

Each point's traced value is the mean over the water pixels within ``SLEEVE_ARRAY_DISC`` of it, with
the standard error of that mean; the points are kept a few millimetres from every sleeve, where the
field is smooth over a disc.

Configuration by environment: ``SLEEVE_ARRAY_RAYS`` per lamp (default 20,000,000), ``SLEEVE_UVT`` in
percent through 1 cm (default 95), ``SLEEVE_SECTORS`` around each arc (default 128),
``SLEEVE_ARRAY_PITCH`` the half-side of the square, metres (default 0.025), ``SLEEVE_ARRAY_PIXEL``
(default 0.5 mm) and ``SLEEVE_ARRAY_DISC`` (default 1 mm).
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

RAYS = int(os.environ.get("SLEEVE_ARRAY_RAYS", 20_000_000))
UVT = float(os.environ.get("SLEEVE_UVT", 95))
SECTORS = int(os.environ.get("SLEEVE_SECTORS", 128))
PITCH = float(os.environ.get("SLEEVE_ARRAY_PITCH", 0.025))
PIXEL = float(os.environ.get("SLEEVE_ARRAY_PIXEL", 5e-4))
DISC = float(os.environ.get("SLEEVE_ARRAY_DISC", 1e-3))
HALF_LENGTH = 1.5


def _say(message: str) -> None:
    print(message, flush=True)


def disc_means(field, grid, points, radius):
    """Mean and standard error of a field over the pixels within ``radius`` of each point."""
    import numpy as np

    x0, y0, pixel, nx, ny = grid
    xs = x0 + (np.arange(nx) + 0.5) * pixel
    ys = y0 + (np.arange(ny) + 0.5) * pixel
    out = []
    for x, y in points:
        near = (np.hypot(xs[None, :] - x, ys[:, None] - y) <= radius) & np.isfinite(field)
        values = field[near]
        out.append((float(np.mean(values)), float(np.std(values) / np.sqrt(near.sum()))))
    return out


def main() -> None:
    import aquaflux  # noqa: F401  (enables x64)
    import jax.numpy as jnp
    import numpy as np
    from aquaflux.radiation import (
        RadiationSettings,
        RayCastOcclusion,
        Scene,
        Surfaces,
        UniformAbsorption,
        VolumeReceivers,
        solve_scene,
    )
    from aquaflux.radiation.refraction import Media, Transparent
    from aquaflux.solids import Cylinder
    from check_refraction import graded_tube
    from tracer import AIR, CLASSES, QUARTZ, WATER, Scene as Traced, fluence, trace, water_fraction

    absorption = -np.log(UVT / 100.0) / 0.01
    centres = PITCH * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    traced_scene = Traced(centres=centres, wall=0.1, absorption=absorption)
    n = int(2 * traced_scene.wall / PIXEL)
    grid = (-traced_scene.wall, -traced_scene.wall, PIXEL, n, n)
    # Between the lamps; beside one; beyond one lamp as seen from another (along the square's edge
    # and its diagonal), where light from the far lamp passes the near one's sleeve; and outside.
    points = np.array(
        [
            [0.0, 0.0],
            [0.0, -PITCH],
            [PITCH + 0.02, PITCH],
            [PITCH + 0.03, PITCH],
            [PITCH + 0.012, PITCH + 0.012],
            [PITCH + 0.025, PITCH + 0.025],
            [-PITCH, PITCH + 0.016],
            [0.0, PITCH + 0.05],
            [-PITCH - 0.04, 0.0],
        ]
    )
    for point in points:
        clearance = np.min(np.hypot(*(point - centres).T)) - traced_scene.outer
        assert clearance > 2 * DISC, f"point {point} is {clearance:.4f} m from a sleeve"

    traced = {}
    for name, variant in (
        ("full", traced_scene),
        ("transmitted", dataclasses.replace(traced_scene, reflections=False)),
    ):
        started = time.perf_counter()
        tally = trace(variant, RAYS, grid, seed=3)
        by_class = fluence(variant, tally, grid, water_fraction(variant, grid))
        total = np.sum(np.nan_to_num(by_class), axis=0)
        total = np.where(np.isfinite(by_class[0]), total, np.nan)
        traced[name] = {
            "total": disc_means(total, grid, points, DISC),
            "classes": [disc_means(part, grid, points, DISC) for part in by_class],
            "tally": tally,
        }
        _say(f"traced {name}: {RAYS} rays a lamp in {time.perf_counter() - started:.0f} s")

    arc = graded_tube(traced_scene.arc, HALF_LENGTH, SECTORS)
    pieces = [arc + np.array([x, y, 0.0]) for x, y in centres]
    triangles = np.concatenate(pieces)
    area = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    per_lamp = area.sum() / len(centres)
    lamps = Surfaces.from_triangles(
        jnp.asarray(triangles),
        solid_id=np.repeat(np.arange(len(centres)), len(arc)),
        solid_names=tuple(f"lamp{k}" for k in range(len(centres))),
        emission=(1.0 * 2 * HALF_LENGTH) / per_lamp,
    )
    axis, half = [0.0, 0.0, 1.0], HALF_LENGTH + 0.25
    sleeves = tuple(
        Transparent(
            Cylinder([x, y, 0.0], axis, traced_scene.outer, half),
            QUARTZ,
            inside=(Transparent(Cylinder([x, y, 0.0], axis, traced_scene.inner, half), AIR),),
        )
        for x, y in centres
    )
    water = UniformAbsorption(absorption)
    receivers = np.concatenate([points, np.zeros((len(points), 1))], axis=1)
    settings = RadiationSettings(self_occlusion=RayCastOcclusion(grid=True))
    started = time.perf_counter()
    solved = solve_scene(
        Scene(
            lamps,
            media=Media(WATER, sleeves, water),
            volume=VolumeReceivers(receivers),
            settings=settings,
        ),
        report=_say,
    )
    seconds = time.perf_counter() - started
    mine = np.asarray(solved.fluence_rate_direct)
    _say(f"aquaflux: {lamps.n_facets} facets, {len(points)} points, {seconds:.0f} s")
    # The straight gather with nothing transparent in the way -- the neighbours' arcs shadowing --
    # for scale: what the sleeves change.
    plain = Scene(lamps, absorption=water, volume=VolumeReceivers(receivers), settings=settings)
    straight = np.asarray(solve_scene(plain).fluence_rate_direct)

    rows = []
    for k, point in enumerate(points):
        full, full_error = traced["full"]["total"][k]
        alone, alone_error = traced["transmitted"]["total"][k]
        shares = [
            traced["transmitted"]["classes"][c][k][0] / alone if alone > 0 else float("nan")
            for c in range(len(CLASSES))
        ]
        rows.append(
            {
                "point_mm": [round(float(v) * 1e3, 1) for v in point],
                "traced_full": full,
                "traced_full_error": full_error,
                "traced_transmitted": alone,
                "traced_transmitted_error": alone_error,
                "transmitted_share_by_path": dict(zip(CLASSES, shares, strict=True)),
                "aquaflux": float(mine[k]),
                "aquaflux_over_transmitted": float(mine[k]) / alone,
                "aquaflux_over_transmitted_in_errors": (float(mine[k]) - alone) / alone_error,
                "transmitted_over_full": alone / full,
                "straight_over_full": float(straight[k]) / full,
            }
        )
        _say(json.dumps(rows[-1]))
    result = {
        "rays_per_lamp": RAYS,
        "uvt_percent": UVT,
        "absorption_per_m": absorption,
        "pitch_m": PITCH,
        "pixel_m": PIXEL,
        "disc_m": DISC,
        "facets": int(lamps.n_facets),
        "aquaflux_seconds": round(seconds, 1),
        "energy": {
            name: {
                "emitted": entry["tally"].emitted,
                "water": entry["tally"].water,
                "arcs": entry["tally"].arcs,
                "wall": entry["tally"].wall,
                "discarded": entry["tally"].discarded,
                "lost": entry["tally"].lost,
            }
            for name, entry in traced.items()
        },
        "rows": rows,
    }
    out = HERE / "work" / "check_array.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    _say(f"written {out}")


if __name__ == "__main__":
    main()
