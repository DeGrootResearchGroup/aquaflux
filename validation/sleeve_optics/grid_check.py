# ruff: noqa: E402  (imports follow the sys.path bootstrap)
"""Is the shooting grid enough for the four-lamp square? ``check_shooting_grid`` at its points.

For the routes round a neighbouring sleeve's rim, counts the paths the grid found against those
found only from a neighbouring corner, says why the grid missed each (rows or columns), and solves
again with that count doubled to see whether the grid lost a whole patch. ``SLEEVE_SECTORS`` round
each arc (default 16); ``SLEEVE_SHOT_GRID`` rows and columns as ``R,C`` (default the module's).
"""

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import aquaflux  # noqa: F401  (enables x64)
import numpy as np
from aquaflux.radiation import check_shooting_grid
from four_lamps import scene

SECTORS = int(os.environ.get("SLEEVE_SECTORS", 16))
GRID = os.environ.get("SLEEVE_SHOT_GRID")
POINTS_MM = [[55.0, 25.0], [50.0, 50.0], [37.0, 37.0], [-25.0, 41.0]]

lamps, media = scene(SECTORS)
points = np.array([[*p, 0.0] for p in POINTS_MM]) / 1e3
grid = tuple(int(n) for n in GRID.split(",")) if GRID else None
print(f"{SECTORS} sectors, {lamps.n_facets} facets, points {POINTS_MM} mm", flush=True)
started = time.perf_counter()
check = check_shooting_grid(lamps, media, points, grid=grid)
print(f"checked in {time.perf_counter() - started:.0f} s", flush=True)
for name, value in vars(check).items():
    print(f"{name}: {value}", flush=True)
