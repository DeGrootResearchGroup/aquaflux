# ruff: noqa: E402  (imports follow the sys.path bootstrap)
"""aquaflux's four-lamp field split by route.

Diagnostic for the shortfall behind a neighbouring sleeve (#604 step 2b-i): which routes deliver the
light at each point, against the tracer's share that entered another sleeve. Same scene and settings as ``check_array.py``; reads the stored trace
``work/check_array-optics.json`` for the tracer's side. ``SLEEVE_SECTORS`` (default 64).
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
import jax.numpy as jnp
import numpy as np
from aquaflux.radiation import RayCastOcclusion, Surfaces, UniformAbsorption, refracted
from aquaflux.radiation.refraction import Media, Transparent
from aquaflux.radiation.work import DEFAULT_PAIR_LIMIT
from aquaflux.solids import Cylinder
from check_refraction import graded_tube
from tracer import AIR, QUARTZ, WATER
from tracer import Scene as Traced

KEEP = [[0.0, 0.0], [0.0, -25.0], [45.0, 25.0], [55.0, 25.0], [50.0, 50.0], [-25.0, 41.0]]

stored = json.load(open(HERE / "work" / "check_array-optics.json"))
rows = [r for r in stored["rows"] if r["point_mm"] in KEEP]
pitch, absorption = stored["pitch_m"], stored["absorption_per_m"]
centres = pitch * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
ts = Traced(centres=centres, wall=0.1, absorption=absorption, neighbours="optics")
points = np.array([[*r["point_mm"], 0.0] for r in rows]) / 1e3
arc = graded_tube(ts.arc, 1.5, int(os.environ.get("SLEEVE_SECTORS", 64)))
tri = np.concatenate([arc + np.array([x, y, 0.0]) for x, y in centres])
area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
lamps = Surfaces.from_triangles(
    jnp.asarray(tri),
    solid_id=np.repeat(np.arange(4), len(arc)),
    solid_names=tuple(f"lamp{k}" for k in range(4)),
    emission=3.0 / (area.sum() / 4),
)
sleeves = tuple(
    Transparent(
        Cylinder([x, y, 0.0], [0, 0, 1], ts.outer, 1.75),
        QUARTZ,
        inside=(Transparent(Cylinder([x, y, 0.0], [0, 0, 1], ts.inner, 1.75), AIR),),
    )
    for x, y in centres
)
media = Media(WATER, sleeves, UniformAbsorption(absorption))
parents = media.parents
lamp_of_facet = np.repeat(np.arange(4), len(arc))
print(f"{lamps.n_facets} facets, {len(points)} points", flush=True)


def label(chain):
    if not chain.passing:
        return "own sleeve only"
    # ``passing`` is outermost first: (quartz,) through the quartz alone, (quartz, air) across the gap.
    sleeve = int(chain.passing[0]) // 2
    kind = "quartz only" if len(chain.passing) == 1 else "quartz and gap"
    return f"through sleeve {sleeve}, {kind}"


def split():
    started = time.perf_counter()
    visibility = refracted.build_refracted_visibility(
        (), lamps, media, points, self_occlusion=RayCastOcclusion(grid=True)
    )
    plan = refracted._plan(lamps, media, points, DEFAULT_PAIR_LIMIT)
    masks = refracted._route_masks(visibility, None, points, plan)
    parts = {}
    for group, mask in zip(plan, masks, strict=True):
        field = np.asarray(
            refracted._group_field(
                lamps, media, group, jnp.asarray(points), None, mask, DEFAULT_PAIR_LIMIT
            )
        )
        lamp = int(lamp_of_facet[group.facets[0]])
        key = (lamp, label(group.chain))
        parts.setdefault(key, np.zeros(len(points)))[group.rows] += field
    print(f"split: {time.perf_counter() - started:.0f} s", flush=True)
    return parts


parts = split()
total = sum(parts.values())
through = sum(v for (_, name), v in parts.items() if name != "own sleeve only")
for k, r in enumerate(rows):
    t, e = r["traced_transmitted"], r["traced_transmitted_error"]
    entered = r["transmitted_share_by_path"]["entered another sleeve"]
    print(
        f"{r['point_mm']}: aquaflux/traced {total[k] / t:.4f} ({(total[k] - t) / e:+.1f} se), "
        f"through-routes {through[k] / t:.4f} of the traced value, tracer entered {entered:.4f}, "
        f"stored aquaflux/traced {r['aquaflux_over_transmitted']:.4f}",
        flush=True,
    )
    for (lamp, name), v in sorted(parts.items()):
        if v[k] > 0:
            print(f"    lamp {lamp} {name}: {v[k] / t:.5f}", flush=True)
