# ruff: noqa: E402  (imports follow the sys.path bootstrap)
"""What the four-lamp scene's direct light costs with and without the coaxial route cull.

The scene of ``check_array.py`` (``optics`` neighbours: every sleeve in the way of every lamp), its
points read from the stored trace ``work/check_array-optics.json``, solved by ``solve_scene`` with
the shadow mask's ray test on a grid. ``SLEEVE_ROUTE_CULL=0`` keeps every receiver on every route,
as before the cull; anything else (the default) culls. Prints the time of each step the scene
reports, the whole solve, and per point the field against the tracer's transmitted light, and
writes them to ``work/cull_cost-<on|off>-<sectors>.json``. Run each arm in its own process and
compare the two files: the fields must agree, since a receiver left out of a route has no path on
it. ``SLEEVE_SECTORS`` round each arc (default 16).
"""

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import aquaflux  # noqa: F401  (enables x64)
import numpy as np
from aquaflux.radiation import (
    RadiationSettings,
    RayCastOcclusion,
    Scene,
    VolumeReceivers,
    coaxial_routes,
    solve_scene,
)
from four_lamps import scene

CULL = os.environ.get("SLEEVE_ROUTE_CULL", "1") != "0"
SECTORS = int(os.environ.get("SLEEVE_SECTORS", 16))
if not CULL:
    coaxial_routes.kept_receivers = lambda media, chain, sources, points: np.ones(
        len(np.asarray(points).reshape(-1, 3)), dtype=bool
    )

stored = json.load(open(HERE / "work" / "check_array-optics.json"))
points = np.array([[*row["point_mm"], 0.0] for row in stored["rows"]]) / 1e3
lamps, media = scene(SECTORS)
print(f"cull {'on' if CULL else 'off'}, {lamps.n_facets} facets, {len(points)} points", flush=True)
steps = []
start = time.perf_counter()


def report(message):
    steps.append((round(time.perf_counter() - start, 1), message))
    print(f"[{steps[-1][0]:8.1f} s] {message}", flush=True)


solution = solve_scene(
    Scene(
        lamps,
        media=media,
        volume=VolumeReceivers(points),
        settings=RadiationSettings(self_occlusion=RayCastOcclusion(grid=True)),
    ),
    report=report,
)
field = np.asarray(solution.fluence_rate_direct)
total = time.perf_counter() - start
print(f"solve {total:.0f} s", flush=True)
for row, value in zip(stored["rows"], field, strict=True):
    traced = row["traced_transmitted"]
    print(f"  {row['point_mm']}: {value:.6f}, tracer {traced:.6f} ({value / traced:.4f})", flush=True)
name = f"cull_cost-{'on' if CULL else 'off'}-{SECTORS}.json"
json.dump(
    {"cull": CULL, "sectors": SECTORS, "seconds": total, "steps": steps, "field": field.tolist()},
    open(HERE / "work" / name, "w"),
    indent=1,
)
