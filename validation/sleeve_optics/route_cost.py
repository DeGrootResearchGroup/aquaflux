# ruff: noqa: E402  (imports follow the sys.path bootstrap)
"""What each route of the four-lamp scene costs to solve, and how many paths it finds.

For each lamp and each point, every route from the lamp's arc to the point (:meth:`Chain.routes`,
including the routes the gather leaves out) is solved for the paths to every arc facet's centroid --
one point per call, as the shadow mask solves them -- and timed warm. The table says which routes the
gather keeps, what each costs per path, and how many paths it found: a route that costs as much as
any other and finds nothing is one whose paths the solve cannot reach.

Same scene as ``check_array.py``. ``SLEEVE_SECTORS`` around each arc (default 12; the time scales with
the facet count, and the counts of paths found barely move with it).
"""

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.radiation import Surfaces, UniformAbsorption
from aquaflux.radiation.refraction import Chain, Media, Transparent, _only_saddles, solve_paths
from aquaflux.solids import Cylinder
from check_refraction import graded_tube
from tracer import AIR, QUARTZ, WATER
from tracer import Scene as Traced

SECTORS = int(os.environ.get("SLEEVE_SECTORS", 12))
POINTS_MM = [[55.0, 25.0], [50.0, 50.0], [-25.0, 41.0]]

pitch, absorption = 0.025, -np.log(0.95) / 0.01
centres = pitch * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
traced = Traced(centres=centres, wall=0.1, absorption=absorption, neighbours="optics")
arc = graded_tube(traced.arc, 1.5, SECTORS)
lamps = Surfaces.from_triangles(
    jnp.asarray(np.concatenate([arc + np.array([x, y, 0.0]) for x, y in centres]))
)
sleeves = tuple(
    Transparent(
        Cylinder([x, y, 0.0], [0, 0, 1], traced.outer, 1.75),
        QUARTZ,
        inside=(Transparent(Cylinder([x, y, 0.0], [0, 0, 1], traced.inner, 1.75), AIR),),
    )
    for x, y in centres
)
media = Media(WATER, sleeves, UniformAbsorption(absorption))
region = media.region_of_facets(lamps)
centroid = np.asarray(lamps.centroid)
print(f"{SECTORS} sectors, {lamps.n_facets} facets, points {POINTS_MM} mm", flush=True)
print("point | lamp | passing | kept | paths | warm s | us/path | found", flush=True)
for point in np.asarray(POINTS_MM) / 1e3:
    receiver = jnp.asarray([*point, 0.0])[None, None]
    for source in np.unique(region):
        facets = np.flatnonzero(region == source)
        for chain in Chain.routes(media.parents, int(source), -1):
            jax.clear_caches()
            sources = jnp.asarray(centroid[facets])[None]
            jax.block_until_ready(solve_paths(media, chain, sources, receiver))
            started = time.perf_counter()
            paths = jax.block_until_ready(solve_paths(media, chain, sources, receiver))
            seconds = time.perf_counter() - started
            n = int(np.asarray(paths.valid).size)
            print(
                f"{np.round(point * 1e3).tolist()} | {int(source) // 2} | {chain.passing!s:7} | "
                f"{'no' if _only_saddles(media, chain) else 'yes'} | {n} | {seconds:.2f} | "
                f"{1e6 * seconds / n:.0f} | {int(np.sum(np.asarray(paths.valid)))}",
                flush=True,
            )
