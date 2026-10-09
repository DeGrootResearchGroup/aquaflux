"""aquaflux's refracted gather against the tracer, for one sleeved lamp in water.

One lamp -- arc, air gap, quartz sleeve, as in ``tracer.py`` -- emitting one watt per metre from its
arc, in water with no wall in reach. The tracer follows every interface (Fresnel reflection and
refraction, total internal reflection), its arc absorbing whatever comes back to it. aquaflux gathers
the arc's facets at points in the water along their refracted paths
(``aquaflux.radiation.refracted_fluence_rate``), each path crossing the air-quartz and quartz-water
surfaces once and the Fresnel reflections counted as lost.

So the two differ by **design** in one way: light reflected inside the sleeve that leaves it anyway
(off the inner surface, across the air gap past the arc, out the far side; or round the quartz) is in
the tracer and not in aquaflux. The tracer is therefore run twice: in full, and with every ray ended at
its first reflection (``reflections=False``), which counts exactly the transmitted paths aquaflux
follows. aquaflux against the second is the error of the gather itself -- the refracted image of each
arc triangle taken with straight edges, one path per corner; the second against the first is what the
reflected paths, left out, are worth.

Reported along a radius at the lamp's mid-plane: both traces' ring averages and standard errors,
aquaflux's value, and aquaflux's straight-line gather from the same arc (no interfaces), which is
what the optics change.

Configuration by environment: ``SLEEVE_CHECK_RAYS`` (default 2,000,000), ``SLEEVE_UVT`` in percent
through 1 cm (default 95), ``SLEEVE_SECTORS`` around the arc (default 256), ``SLEEVE_CHECK_PIXEL``
the tally's pixel side in metres (default 0.25 mm: the field beside the sleeve is steep and curved,
so a coarse pixel's average reads above the value at its ring).
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

RAYS = int(os.environ.get("SLEEVE_CHECK_RAYS", 2_000_000))
UVT = float(os.environ.get("SLEEVE_UVT", 95))
SECTORS = int(os.environ.get("SLEEVE_SECTORS", 256))
PIXEL = float(os.environ.get("SLEEVE_CHECK_PIXEL", 2.5e-4))
HALF_LENGTH = 1.5


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def graded_tube(radius: float, half_length: float, sectors: int):
    """An open tube about the z axis, wound outward, its slices finest at the mid-plane.

    The receivers are at ``z = 0``, a few millimetres from the arc, so the slices there are a
    fraction of that distance and grow geometrically away from it, where a triangle's size only has
    to be small next to its distance. Wound as ``check_tracer.tube`` winds its even slices.
    """
    import numpy as np

    finest, growth = 2e-4, 1.06
    steps = [0.0]
    while steps[-1] < half_length:
        steps.append(steps[-1] + finest * growth ** len(steps))
    z = np.asarray(steps) / steps[-1] * half_length
    levels = np.concatenate([-z[::-1], z[1:]])
    angle = np.linspace(0.0, 2.0 * np.pi, sectors, endpoint=False)
    ring = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    a, b = ring, np.roll(ring, -1, axis=0)
    lower, upper = levels[:-1], levels[1:]

    def corner(xy, zs):
        return np.concatenate(
            [
                np.broadcast_to(xy[:, None, :], (sectors, len(zs), 2)),
                np.broadcast_to(zs[None, :, None], (sectors, len(zs), 1)),
            ],
            axis=2,
        )

    p00, p10 = corner(a, lower), corner(b, lower)
    p01, p11 = corner(a, upper), corner(b, upper)
    first = np.stack([p00, p10, p11], axis=2)
    second = np.stack([p00, p11, p01], axis=2)
    return np.stack([first, second], axis=2).reshape(-1, 3, 3)


def main() -> None:
    import aquaflux  # noqa: F401  (enables x64)
    import jax.numpy as jnp
    import numpy as np
    from aquaflux.radiation import Surfaces, UniformAbsorption, direct_fluence_rate
    from aquaflux.radiation.refraction import Media, Transparent
    from aquaflux.radiation.refracted import refracted_fluence_rate
    from aquaflux.solids import Cylinder
    from check_tracer import radial_profile
    from tracer import AIR, QUARTZ, WATER, Scene, fluence, trace, water_fraction

    absorption = -np.log(UVT / 100.0) / 0.01
    scene = Scene(centres=np.zeros((1, 2)), wall=0.08, absorption=absorption)
    pixel = PIXEL
    n = int(2 * scene.wall / pixel)
    grid = (-scene.wall, -scene.wall, pixel, n, n)
    radii = np.array([0.0125, 0.0135, 0.016, 0.02, 0.03, 0.045, 0.06])

    traced = {}
    for name, variant in (
        ("full", scene),
        ("transmitted", dataclasses.replace(scene, reflections=False)),
    ):
        started = time.perf_counter()
        tally = trace(variant, RAYS, grid, seed=1)
        traced[name] = (
            radial_profile(
                fluence(variant, tally, grid, water_fraction(variant, grid))[0], grid, radii, 0.6e-3
            ),
            tally,
        )
        _say(f"traced {name}: {RAYS} rays in {time.perf_counter() - started:.0f} s")

    triangles = graded_tube(scene.arc, HALF_LENGTH, SECTORS)
    area = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    surfaces = Surfaces.from_triangles(
        jnp.asarray(triangles), emission=(1.0 * 2 * HALF_LENGTH) / area.sum()
    )
    axis, length = [0.0, 0.0, 1.0], 2 * HALF_LENGTH + 0.5
    gap = Transparent(Cylinder([0, 0, 0], axis, scene.inner, length / 2), AIR)
    media = Media(
        WATER,
        (Transparent(Cylinder([0, 0, 0], axis, scene.outer, length / 2), QUARTZ, inside=(gap,)),),
        absorption=UniformAbsorption(absorption),
    )
    points = np.stack([radii, np.zeros_like(radii), np.zeros_like(radii)], axis=1)
    started = time.perf_counter()
    refracted = np.asarray(refracted_fluence_rate(surfaces, media, points))
    seconds = time.perf_counter() - started
    _say(f"refracted gather: {surfaces.n_facets} facets, {len(radii)} points, {seconds:.0f} s")
    # The straight gather absorbs along the whole segment, the air and quartz included; the
    # tracer's straight variant would not, but this one is shown only for scale.
    straight = np.asarray(
        direct_fluence_rate(surfaces, points, absorption=UniformAbsorption(absorption))
    )

    rows = []
    for k, (radius, mine, plain) in enumerate(zip(radii, refracted, straight, strict=True)):
        full, full_error = traced["full"][0][k]
        alone, alone_error = traced["transmitted"][0][k]
        rows.append(
            {
                "radius_mm": round(float(radius) * 1e3, 2),
                "traced_full": full,
                "traced_full_error": full_error,
                "traced_transmitted": alone,
                "traced_transmitted_error": alone_error,
                "refracted": float(mine),
                "refracted_over_transmitted": float(mine) / alone,
                "refracted_over_transmitted_in_errors": (float(mine) - alone) / alone_error,
                "transmitted_over_full": alone / full,
                "refracted_over_full": float(mine) / full,
                "straight_over_full": float(plain) / full,
            }
        )
        _say(str(rows[-1]))
    result = {
        "rays": RAYS,
        "uvt_percent": UVT,
        "absorption_per_m": absorption,
        "facets": int(surfaces.n_facets),
        "gather_seconds": round(seconds, 1),
        "energy": {
            name: {
                "emitted": tally.emitted,
                "water": tally.water,
                "arcs": tally.arcs,
                "wall": tally.wall,
                "discarded": tally.discarded,
                "lost": tally.lost,
            }
            for name, (_, tally) in traced.items()
        },
        "rows": rows,
    }
    out = HERE / "work" / "check_refraction.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    _say(f"written {out}")


if __name__ == "__main__":
    main()
