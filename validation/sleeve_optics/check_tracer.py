"""Checks of the sleeve-optics tracer against independent answers, before it is used to measure anything.

1. **The estimator and the emission**: a single lamp whose sleeve surface emits straight into absorbing
   water, nothing else in the scene. aquaflux's direct gather is exact for that (a closed-form solid angle
   per facet, a long tessellated tube standing in for the infinite one), so the tracer's fluence rate along
   a radius must match it to within its own sampling noise.
2. **Energy**: everything emitted is absorbed by the water, the arcs, an opaque sleeve or the wall, or is
   still in flight when a ray reaches its event cap -- reported, not assumed.
3. **Fresnel**: reflectance plus transmittance carries the power across an interface (checked through the
   cosine-weighted flux, ``1 - R = T``), the critical angle is where reflectance reaches one, and at normal
   incidence the reflectance is ``((n1 - n2) / (n1 + n2))^2``.

Configuration by environment: ``SLEEVE_CHECK_RAYS`` (default 2,000,000 per lamp).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

RAYS = int(os.environ.get("SLEEVE_CHECK_RAYS", 2_000_000))


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def tube(radius: float, half_length: float, sectors: int, slices: int):
    """Triangles of an open tube about the z axis, wound to face outward, ``(n, 3, 3)``."""
    import numpy as np

    angle = np.linspace(0.0, 2.0 * np.pi, sectors, endpoint=False)
    ring = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    z = np.linspace(-half_length, half_length, slices + 1)
    triangles = []
    for i in range(sectors):
        a, b = ring[i], ring[(i + 1) % sectors]
        for k in range(slices):
            p00 = [a[0], a[1], z[k]]
            p10 = [b[0], b[1], z[k]]
            p01 = [a[0], a[1], z[k + 1]]
            p11 = [b[0], b[1], z[k + 1]]
            triangles.append([p00, p10, p11])
            triangles.append([p00, p11, p01])
    return np.asarray(triangles)


def radial_profile(g, grid, radii, width):
    """Mean of a field over thin rings about the origin, ignoring pixels that are not water."""
    import numpy as np

    x0, y0, pixel, nx, ny = grid
    xs = x0 + (np.arange(nx) + 0.5) * pixel
    ys = y0 + (np.arange(ny) + 0.5) * pixel
    r = np.hypot(xs[None, :], ys[:, None])
    out = []
    for radius in radii:
        ring = (np.abs(r - radius) < width / 2) & np.isfinite(g)
        out.append((float(np.mean(g[ring])), float(np.std(g[ring]) / np.sqrt(ring.sum()))))
    return out


def check_against_the_direct_gather() -> dict:
    import aquaflux  # noqa: F401  (enables x64)
    import jax.numpy as jnp
    import numpy as np
    from aquaflux.radiation import Surfaces, UniformAbsorption, direct_fluence_rate
    from tracer import Scene, fluence, trace, water_fraction

    scene = Scene(centres=np.zeros((1, 2)), emit_from="sleeve", neighbours="opaque", wall=0.08)
    pixel = 1e-3
    n = int(2 * scene.wall / pixel)
    grid = (-scene.wall, -scene.wall, pixel, n, n)
    started = time.perf_counter()
    tally = trace(scene, RAYS, grid, seed=1)
    g = fluence(scene, tally, grid, water_fraction(scene, grid))[0]
    radii = np.array([0.0135, 0.016, 0.02, 0.03, 0.045, 0.06])
    traced = radial_profile(g, grid, radii, 0.6e-3)
    seconds = time.perf_counter() - started

    # The same lamp in aquaflux: a tube long enough to stand in for an infinite one, emitting one watt
    # per metre over its own (inscribed) area.
    half = 1.5
    triangles = tube(scene.outer, half, sectors=256, slices=600)
    area = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    exitance = (1.0 * 2 * half) / area.sum()
    surfaces = Surfaces.from_triangles(jnp.asarray(triangles), emission=exitance)
    points = np.stack([radii, np.zeros_like(radii), np.zeros_like(radii)], axis=1)
    exact = np.asarray(
        direct_fluence_rate(surfaces, points, absorption=UniformAbsorption(scene.absorption))
    )
    rows = []
    for radius, (value, error), reference in zip(radii, traced, exact, strict=True):
        rows.append(
            {
                "radius_mm": round(radius * 1e3, 2),
                "traced": value,
                "standard_error": error,
                "direct_gather": float(reference),
                "relative_difference": value / float(reference) - 1.0,
                "in_standard_errors": (value - float(reference)) / error,
            }
        )
        _say(str(rows[-1]))
    balance = {
        "emitted": tally.emitted,
        "water": tally.water,
        "sleeves": tally.sleeves,
        "wall": tally.wall,
        "arcs": tally.arcs,
        "lost": tally.lost,
        "unaccounted": tally.emitted
        - (tally.water + tally.sleeves + tally.wall + tally.arcs + tally.lost),
    }
    _say(f"energy {balance}")
    return {"rows": rows, "balance": balance, "seconds": round(seconds, 1)}


def check_fresnel() -> dict:
    import numpy as np
    from tracer import AIR, QUARTZ, WATER, fresnel

    normal_r, _ = fresnel(np.array([1.0]), WATER, QUARTZ)
    expected = ((WATER - QUARTZ) / (WATER + QUARTZ)) ** 2
    critical = np.sqrt(1.0 - (AIR / QUARTZ) ** 2)
    # A cosine just under the critical one is an angle just past it, and the other way round.
    past, _ = fresnel(np.array([critical * (1 - 1e-9)]), QUARTZ, AIR)
    short, _ = fresnel(np.array([critical * (1 + 1e-9)]), QUARTZ, AIR)
    # Flux carried across: the transmitted share of the cosine-weighted flux, 1 - R, against the
    # Fresnel transmittance written out independently from the amplitude coefficients.
    cos_i = np.linspace(0.05, 1.0, 50)
    r, cos_t = fresnel(cos_i, WATER, QUARTZ)
    ts = 2 * WATER * cos_i / (WATER * cos_i + QUARTZ * cos_t)
    tp = 2 * WATER * cos_i / (QUARTZ * cos_i + WATER * cos_t)
    t = 0.5 * (ts**2 + tp**2) * (QUARTZ * cos_t) / (WATER * cos_i)
    result = {
        "normal_incidence": float(normal_r[0]),
        "normal_incidence_closed_form": expected,
        "angle_just_past_critical": float(past[0]),
        "angle_just_short_of_critical": float(short[0]),
        "max_abs_r_plus_t_minus_one": float(np.max(np.abs(r + t - 1.0))),
    }
    _say(f"fresnel {result}")
    return result


def main() -> None:
    import json

    result = {
        "fresnel": check_fresnel(),
        "direct_gather": check_against_the_direct_gather(),
    }
    out = HERE / "work" / "check_tracer.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    _say(f"written {out}")


if __name__ == "__main__":
    main()
