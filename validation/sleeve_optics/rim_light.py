# ruff: noqa: E402  (imports follow the sys.path bootstrap)
"""The light that skims a neighbouring sleeve's rim: aquaflux's shot paths against the backward gather.

A path from one lamp's arc that crosses another sleeve's quartz wall without entering its air gap is
a saddle of its optical length, found by shooting (``aquaflux.radiation.shooting``). This gathers
that route's light alone -- every route ``_shot`` selects, no shadow mask, since the only things in
the way are the source's own arc (whose far side faces away and sends nothing) and the other
sleeves (which such a path must miss) -- and sets it beside ``backward_gather.py``'s share of light
through a wall alone, both for each lamp emitting 1 W per metre.

``SLEEVE_SECTORS`` around each arc (default 32); ``SLEEVE_SLICES`` splits each arc triangle into
that many squared (default 1); ``SLEEVE_BACKWARD`` the backward gather's azimuths
(default 2^17, with 160 axial cosines); ``SLEEVE_SHOT_GRID`` the shooting grid's
rows and columns, to see whether it misses paths (default the module's).
"""

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
from aquaflux.radiation import refracted, shooting
from aquaflux.radiation.refraction import _shot
from aquaflux.radiation.work import DEFAULT_PAIR_LIMIT
from backward_gather import gather
from four_lamps import scene

SECTORS = int(os.environ.get("SLEEVE_SECTORS", 32))
SLICES = int(os.environ.get("SLEEVE_SLICES", 1))
BACKWARD = int(os.environ.get("SLEEVE_BACKWARD", 2**17))
GRID = os.environ.get("SLEEVE_SHOT_GRID")
POINTS_MM = [[55.0, 25.0], [50.0, 50.0], [37.0, 37.0], [-25.0, 41.0]]


lamps, media = scene(SECTORS, SLICES)
points = np.array([[*p, 0.0] for p in POINTS_MM]) / 1e3
grid = shooting.checked_grid((int(GRID), int(GRID)) if GRID else None)
print(
    f"{SECTORS} sectors x {SLICES} slices, {lamps.n_facets} facets, shooting grid "
    f"{grid[0]} x {grid[1]}",
    flush=True,
)

started = time.perf_counter()
plan = [
    g for g in refracted._plan(lamps, media, points, DEFAULT_PAIR_LIMIT) if _shot(media, g.chain)
]
shot = np.zeros(len(points))
for group in plan:
    reach = refracted._route_reach(lamps, media, group, points, DEFAULT_PAIR_LIMIT, grid)
    shot[group.rows] += np.asarray(
        refracted._group_field(
            lamps, media, group, jnp.asarray(points), None, None, reach, DEFAULT_PAIR_LIMIT
        )
    )
print(f"shot routes ({len(plan)} groups): {time.perf_counter() - started:.0f} s", flush=True)
for k, (x, y) in enumerate(POINTS_MM):
    G, _ = gather(np.array([x, y]) * 1e-3, BACKWARD, 160)
    wall = G[1].sum()
    ratio = shot[k] / wall if wall > 0 else float("nan")
    print(
        f"({x:.0f}, {y:.0f}) mm: shot {shot[k]:.5f}, backward {wall:.5f} "
        f"({wall / G.sum():.4%} of its G), shot/backward {ratio:.3f}",
        flush=True,
    )
