"""aquaflux's side of check_array.py rerun against its stored traces (work/check_array-<mode>.json).

Temporary: same scene and settings as check_array.py, without re-tracing. Delete once the
neighbour pass-through routes are measured.
"""

import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import numpy as np
import aquaflux  # noqa
import jax.numpy as jnp
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
from tracer import AIR, QUARTZ, WATER, Scene as Traced

mode = os.environ.get("SLEEVE_ARRAY_NEIGHBOURS", "optics")
stored = json.load(open(HERE / "work" / f"check_array-{mode}.json"))
PITCH = stored["pitch_m"]
absorption = stored["absorption_per_m"]
centres = PITCH * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
ts = Traced(centres=centres, wall=0.1, absorption=absorption, neighbours=mode)
points = np.array([r["point_mm"] for r in stored["rows"]]) / 1e3
arc = graded_tube(ts.arc, 1.5, int(os.environ.get("SLEEVE_SECTORS", 128)))
tri = np.concatenate([arc + np.array([x, y, 0.0]) for x, y in centres])
area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
lamps = Surfaces.from_triangles(
    jnp.asarray(tri),
    solid_id=np.repeat(np.arange(4), len(arc)),
    solid_names=tuple(f"lamp{k}" for k in range(4)),
    emission=3.0 / (area.sum() / 4),
)
half = 1.75
sleeves = tuple(
    Transparent(
        Cylinder([x, y, 0.0], [0, 0, 1], ts.outer, half),
        QUARTZ,
        inside=(Transparent(Cylinder([x, y, 0.0], [0, 0, 1], ts.inner, half), AIR),),
    )
    for x, y in centres
)
receivers = np.concatenate([points, np.zeros((len(points), 1))], axis=1)
settings = RadiationSettings(self_occlusion=RayCastOcclusion(grid=True))
own = np.asarray(lamps.solid_id)
scenes = (
    [(lamps, sleeves)]
    if mode == "optics"
    else [
        (lamps.with_optics(emission=jnp.where(own == k, lamps.emission, 0.0)), (sleeves[k],))
        for k in range(4)
    ]
)
t0 = time.perf_counter()
mine = np.zeros(len(points))
for emitting, regions in scenes:
    s = solve_scene(
        Scene(
            emitting,
            media=Media(WATER, regions, UniformAbsorption(absorption)),
            volume=VolumeReceivers(receivers),
            settings=settings,
        ),
        report=lambda m: print(m, flush=True),
    )
    mine += np.asarray(s.fluence_rate_direct)
print(f"aquaflux {time.perf_counter() - t0:.0f} s", flush=True)
for r, m in zip(stored["rows"], mine):
    t, e = r["traced_transmitted"], r["traced_transmitted_error"]
    print(
        r["point_mm"],
        f"traced {t:.4f}+-{e:.4f} old {r['aquaflux']:.4f} ({r['aquaflux_over_transmitted']:.4f}) new {m:.4f} ({m / t:.4f}, {(m - t) / e:+.1f} se)",
        flush=True,
    )
