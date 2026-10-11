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
(default 2^17, with 160 axial cosines).
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
from aquaflux.radiation import Surfaces, UniformAbsorption, refracted
from aquaflux.radiation.refraction import Media, Transparent, _shot
from aquaflux.radiation.work import DEFAULT_PAIR_LIMIT
from aquaflux.solids import Cylinder
from backward_gather import gather
from check_refraction import graded_tube
from tracer import AIR, QUARTZ, WATER
from tracer import Scene as Traced

SECTORS = int(os.environ.get("SLEEVE_SECTORS", 32))
SLICES = int(os.environ.get("SLEEVE_SLICES", 1))
BACKWARD = int(os.environ.get("SLEEVE_BACKWARD", 2**17))
POINTS_MM = [[55.0, 25.0], [50.0, 50.0], [37.0, 37.0], [-25.0, 41.0]]


def sliced(triangles, parts):
    """Each triangle split into ``parts**2`` by an even lattice on its own corners, winding kept."""
    if parts == 1:
        return triangles
    out = []
    for first, second, third in triangles:
        u, v = (second - first) / parts, (third - first) / parts
        for i in range(parts):
            for j in range(parts - i):
                a = first + i * u + j * v
                out.append([a, a + u, a + v])
                if j < parts - i - 1:
                    out.append([a + u, a + u + v, a + v])
    return np.asarray(out)


pitch, absorption = 0.025, -np.log(0.95) / 0.01
centres = pitch * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
traced = Traced(centres=centres, wall=0.1, absorption=absorption, neighbours="optics")
arc = sliced(graded_tube(traced.arc, 1.5, SECTORS), SLICES)
triangles = np.concatenate([arc + np.array([x, y, 0.0]) for x, y in centres])
area = 0.5 * np.linalg.norm(
    np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
)
lamps = Surfaces.from_triangles(jnp.asarray(triangles), emission=3.0 / (area.sum() / 4))
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
print(f"{SECTORS} sectors x {SLICES} slices, {lamps.n_facets} facets", flush=True)

started = time.perf_counter()
plan = [
    g for g in refracted._plan(lamps, media, points, DEFAULT_PAIR_LIMIT) if _shot(media, g.chain)
]
shot = np.zeros(len(points))
for group in plan:
    shot[group.rows] += np.asarray(
        refracted._group_field(
            lamps, media, group, jnp.asarray(points), None, None, DEFAULT_PAIR_LIMIT
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
