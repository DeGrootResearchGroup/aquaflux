# ruff: noqa: E402  (imports follow the sys.path bootstrap)
"""Does shooting find every path the descent finds on a route through a neighbour's air gap?

On each route through a neighbouring sleeve's quartz and air gap, in the four-lamp scene, the paths
from every arc corner to each point are found both ways: by the descent from four starts, distinct
paths kept (what the gather solves today), and by shooting past each side of the shell's axis
(``refraction._shot_paths``, the route's band running from the axis to the core's edge). Each valid
path of one is matched to the other's by its crossing points, to ``1e-6`` of the separation. A path
the descent finds and shooting does not is a path the switch would lose; one shooting finds alone is
one the descent missed. Also timed, warm, per path searched.

Same scene as ``check_array.py`` with no shadow mask. ``SLEEVE_SECTORS`` (default 8).
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
from aquaflux.radiation import Surfaces, UniformAbsorption, refracted
from aquaflux.radiation.refraction import Media, Transparent, _shot_paths, solve_paths
from aquaflux.radiation.work import DEFAULT_PAIR_LIMIT
from aquaflux.solids import Cylinder
from check_refraction import graded_tube
from tracer import AIR, QUARTZ, WATER
from tracer import Scene as Traced

SECTORS = int(os.environ.get("SLEEVE_SECTORS", 8))
POINTS_MM = [[55.0, 25.0], [50.0, 50.0], [-25.0, 41.0], [0.0, -25.0]]
SAME = 1e-6

pitch, absorption = 0.025, -np.log(0.95) / 0.01
centres = pitch * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
traced = Traced(centres=centres, wall=0.1, absorption=absorption, neighbours="optics")
arc = graded_tube(traced.arc, 1.5, SECTORS)
triangles = np.concatenate([arc + np.array([x, y, 0.0]) for x, y in centres])
lamps = Surfaces.from_triangles(jnp.asarray(triangles), emission=1.0)
sleeves = tuple(
    Transparent(
        Cylinder([x, y, 0.0], [0, 0, 1], traced.outer, 1.75),
        QUARTZ,
        inside=(Transparent(Cylinder([x, y, 0.0], [0, 0, 1], traced.inner, 1.75), AIR),),
    )
    for x, y in centres
)
media = Media(WATER, sleeves, UniformAbsorption(absorption))
points = np.array([[*p, 0.0] for p in POINTS_MM]) / 1e3
print(f"{SECTORS} sectors, {lamps.n_facets} facets, points {POINTS_MM} mm", flush=True)


def timed(solve):
    jax.block_until_ready(solve())
    started = time.perf_counter()
    result = jax.block_until_ready(solve())
    return result, time.perf_counter() - started


print(
    "point | lamp -> gap of | pairs | descent paths | shot paths | descent only | shot only | "
    "us/search descent | us/search shot",
    flush=True,
)
for group in refracted._plan(lamps, media, points, DEFAULT_PAIR_LIMIT):
    chain = group.chain
    if len(chain.passing) != 2:
        continue
    corners, _ = refracted._shared_corners(lamps, group.facets)
    corners = np.asarray(corners)
    for row in group.rows:
        receiver = points[row]
        n = len(corners)
        descent, t_descent = timed(
            lambda chain=chain, corners=corners, receiver=receiver: solve_paths(
                media, chain, jnp.asarray(corners), jnp.asarray(receiver)
            )
        )
        sources = jnp.asarray(np.concatenate([corners, corners]))
        targets = jnp.asarray(np.repeat(receiver[None], 2 * n, axis=0))
        sides = jnp.asarray(np.repeat([1.0, -1.0], n))
        shot, t_shot = timed(
            lambda chain=chain, sources=sources, targets=targets, sides=sides: _shot_paths(
                media, chain, sources, targets, sides
            )
        )
        d_valid, d_points = np.asarray(descent.valid), np.asarray(descent.points)
        s_valid = np.asarray(shot.valid).reshape(2, n)
        s_points = np.asarray(shot.points).reshape(2, n, *d_points.shape[2:])
        scale = SAME * np.linalg.norm(receiver - corners, axis=1)
        # (descent start, side, corner): the largest distance between matching crossings.
        apart = np.max(np.linalg.norm(d_points[:, None] - s_points[None], axis=-1), axis=-1)
        same = apart <= scale[None, None]
        d_only = np.sum(d_valid & ~np.any(same & s_valid[None], axis=1))
        s_only = np.sum(s_valid & ~np.any(same & d_valid[:, None], axis=0))
        print(
            f"{POINTS_MM[row]} | {chain.legs[0] // 2} -> {chain.passing[0] // 2} | {n} | "
            f"{int(d_valid.sum())} | {int(s_valid.sum())} | {int(d_only)} | {int(s_only)} | "
            f"{1e6 * t_descent / (4 * n):.0f} | {1e6 * t_shot / (2 * n):.0f}",
            flush=True,
        )
