# ruff: noqa: E402  (imports follow the sys.path bootstrap)
"""Is the coaxial route cull sound on the four-lamp scene, and how much does it leave out?

For every route through a neighbouring sleeve, the receivers the cull leaves out
(``coaxial_routes.kept_receivers``) are solved anyway -- every distinct corner of every source
facet, on every start or side, the shot routes with continuation -- and any path found there is a
receiver the cull should have kept. Prints, per route kind, the rows kept and left out, the paths
found on rows left out (must be none for the cull to be sound), and the paths on rows kept.

``SLEEVE_SECTORS`` (default 4) sets the arcs; ``CULL_POINTS`` (default 40) the receivers drawn
uniformly in the water round the lamps, on top of the six points ``route_split.py`` reads.
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
from aquaflux.radiation import coaxial_routes, refracted
from aquaflux.radiation.refraction import Chain, _shot
from aquaflux.radiation.work import DEFAULT_PAIR_LIMIT
from four_lamps import CENTRES, scene

SECTORS = int(os.environ.get("SLEEVE_SECTORS", 4))
N_POINTS = int(os.environ.get("CULL_POINTS", 40))
KEEP = np.array([[0.0, 0.0], [0.0, -25.0], [45.0, 25.0], [55.0, 25.0], [50.0, 50.0], [-25.0, 41.0]])

lamps, media = scene(SECTORS)
rng = np.random.default_rng(0)
drawn = []
while len(drawn) < N_POINTS:
    xy = rng.uniform(-0.07, 0.07, size=2)
    if np.min(np.linalg.norm(CENTRES - xy, axis=1)) > 0.0125:
        drawn.append(xy)
xy = np.concatenate([KEEP / 1e3, np.array(drawn)])
z = np.concatenate([np.zeros(len(KEEP)), rng.uniform(-0.3, 0.3, size=N_POINTS)])
points = np.column_stack([xy, z])
print(f"{lamps.n_facets} facets, {len(points)} points", flush=True)

regions = media.region_of_facets(lamps)
receiver_region = media.region_of(points, "receiver")
parents = media.parents
vertices = np.asarray(lamps.vertices, dtype=float)
totals = {}
start = time.perf_counter()
for source in np.unique(regions):
    facets = np.flatnonzero(regions == source)
    corners, shared = refracted._shared_corners(lamps, facets)
    corners = np.asarray(corners)
    edges = refracted._corner_edges(shared)
    rows = np.flatnonzero(receiver_region == -1)
    for chain in Chain.routes(parents, int(source), -1):
        if not chain.passing:
            continue
        kind = ("shot" if _shot(media, chain) else "descended") + f" depth {len(chain.passing)}"
        keep = coaxial_routes.kept_receivers(media, chain, vertices[facets], points[rows])
        if _shot(media, chain):
            _, valid, _, _ = refracted._continued_paths(
                media, chain, corners, points[rows], edges, DEFAULT_PAIR_LIMIT
            )
            found = valid.any(axis=(0, 2))
        else:
            _, valid = refracted._host_paths(media, chain, corners, points[rows], DEFAULT_PAIR_LIMIT)
            found = valid.any(axis=(0, 2))
        entry = totals.setdefault(kind, np.zeros(4, dtype=int))
        entry += [keep.sum(), (~keep).sum(), (found & ~keep).sum(), (found & keep).sum()]
        print(
            f"source {source} through {chain.passing}: kept {keep.sum()}, left out {(~keep).sum()},"
            f" reached and left out {(found & ~keep).sum()}, reached and kept {(found & keep).sum()}"
            f"  [{time.perf_counter() - start:.0f} s]",
            flush=True,
        )
print("\nper route kind: kept, left out, reached and left out (must be 0), reached and kept")
for kind, (kept, out, wrong, right) in totals.items():
    print(f"  {kind}: {kept} {out} {wrong} {right}")
unsound = sum(int(t[2]) for t in totals.values())
print("SOUND" if unsound == 0 else f"UNSOUND: {unsound} rows left out that a path reaches")
