"""How many facets does the lamp actually need?

The gather costs ``n_receivers x n_facets``, and the shadow mask costs that again times whatever
it must test, so the emitter's facet count sets the price of everything. The Sozzi comparison
uses the tutorial's own ``lampWall.stl`` -- **7,516 facets** at ~4 mm, which is a mesh generated
for `snappyHexMesh` to snap to, not a number anyone chose for radiation. This measures what that
buys.

**What a facet count can cost, and what it cannot.** For a Lambertian emitter the radiance
leaving a facet is the same in every direction, and the solid angle it subtends is exact at any
distance, so in vacuum a coarse lamp is not approximate at all. Three things do depend on the
facet count:

1. **Absorption is evaluated once per facet**, along the centroid-to-receiver path, so a facet
   whose far corner sits at a different optical depth is attenuated as though it did not. This
   is the term that bites near the lamp, where the path length varies most across a facet.
2. **The area is inscribed**: a polygonal cylinder is thinner than the round one, so at fixed
   exitance a coarse lamp radiates less total power. That is a pure area effect, known in closed
   form, and it is reported separately -- rescaling to equal power removes it.
3. **Shadows are resolved per facet**, which this study excludes by design: it samples chamber
   cells, which see the whole lamp.

So the ladder is run twice: as the case would run it (fixed exitance, so the area deficit is
included) and rescaled to equal emitted power (so only the shape and the absorption sampling
remain).

**The lamp read from the reactor's CAD drawing** is a further set of rungs when the CAD kernel is
installed: the drawing's ``lamp`` solid triangulated by :meth:`aquaflux.io.cad.CadModel.triangles`
at several chord tolerances and facet sizes. Every vertex of those lies on the drawing's surface
and their facets are bounded in size, so they are the rungs a user of a STEP file would actually
run. The drawing's lamp has a base disc the analytic lamp and the case's patch both leave out, for
the reason given in :func:`lamp`; it is dropped here too, so every rung radiates from the same
surface. (It could not light a cell anyway: its normal points into the end wall.)

Run with ``validation/run_case.sh validation/sozzi_radiation/lamp_resolution.py``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import aquaflux  # noqa: E402,F401  (enables x64)
import jax.numpy as jnp  # noqa: E402
from aquaflux.io import read_openfoam  # noqa: E402
from aquaflux.radiation import (  # noqa: E402
    Surfaces,
    UniformAbsorption,
    direct_fluence_rate,
    read_stl,
)

WORK = HERE / "work"
CASE = WORK / "case"
OUT = WORK / "compare"

R_LAMP, X_TIP = 0.010, 0.80  # the lamp: a cylinder to x = 0.80 m, then a hemispherical cap
EXITANCE, ABSORPTION = 696.42, 35.67
#: (sectors around the lamp, slices along it). The last is the reference.
LADDER = ((8, 16), (16, 32), (24, 64), (32, 128), (48, 256), (64, 512), (128, 1024))
SAMPLE = 4000
#: (chord tolerance, facet size) in metres, for the lamp read from the drawing. The chord sets the
#: spacing around the lamp -- about 50 sectors at 2e-5 on a 10 mm radius -- and the facet size the
#: spacing along it.
DRAWING_LADDER = (
    (1e-4, 0.02),
    (2e-5, 0.01),
    (2e-5, 0.005),
    (5e-6, 0.0025),
    # A coarse chord with a fine facet: spacing along the lamp, not around it, is what the cells
    # nearest it are sensitive to.
    (1e-4, 0.005),
    (1e-4, 0.0025),
)
DRAWING = HERE.parent / "uvreactor_openfoam" / "of_case" / "SozziTaghipour.step"


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def lamp(sectors: int, slices: int) -> np.ndarray:
    """The lamp as a closed cylinder with a hemispherical tip, wound outward.

    Built analytically rather than by re-triangulating the STL, so a resolution is a property of
    the geometry and not of whatever a mesher did to it. The base disc at x = 0 is left out for
    the same reason it is absent from the case's own patch: it sits against the end wall.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, sectors, endpoint=False)
    ring = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    faces = []
    x = np.linspace(0.0, X_TIP, slices + 1)
    for i in range(slices):
        for j in range(sectors):
            k = (j + 1) % sectors
            a = [x[i], R_LAMP * ring[j, 0], R_LAMP * ring[j, 1]]
            b = [x[i], R_LAMP * ring[k, 0], R_LAMP * ring[k, 1]]
            c = [x[i + 1], R_LAMP * ring[k, 0], R_LAMP * ring[k, 1]]
            d = [x[i + 1], R_LAMP * ring[j, 0], R_LAMP * ring[j, 1]]
            faces += [[a, b, c], [a, c, d]]
    # The cap: rings of constant polar angle from the equator to the pole.
    rings = max(2, sectors // 4)
    polar = np.linspace(0.0, np.pi / 2.0, rings + 1)
    for i in range(rings):
        for j in range(sectors):
            k = (j + 1) % sectors

            def point(theta, index):
                return [
                    X_TIP + R_LAMP * np.sin(theta),
                    R_LAMP * np.cos(theta) * ring[index, 0],
                    R_LAMP * np.cos(theta) * ring[index, 1],
                ]

            a, b = point(polar[i], j), point(polar[i], k)
            c, d = point(polar[i + 1], k), point(polar[i + 1], j)
            faces += [[a, b, c], [a, c, d]]
    return np.array(faces)


def drawing_lamps() -> list[tuple[str, np.ndarray]]:
    """The drawing's lamp at each rung of :data:`DRAWING_LADDER`, base disc removed; [] without CAD."""
    try:
        from aquaflux.io.cad import Placement, read_step
    except ImportError:
        _say("CAD kernel not installed: the rungs read from the drawing are skipped")
        return []
    cad = read_step(DRAWING, Placement(matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
    rungs = []
    for chord, size in DRAWING_LADDER:
        triangles = cad.triangles("lamp", chord=chord, facet_size=size)
        on_base = np.all(np.abs(triangles[:, :, 0]) < 1e-9, axis=1)
        rungs.append((f"drawing, chord {chord:g} m, facets {size:g} m", triangles[~on_base]))
    return rungs


def field(vertices: np.ndarray, receivers: np.ndarray) -> tuple[np.ndarray, float]:
    """``G`` at the receivers from a lamp of these facets, and the power it emits."""
    surfaces = Surfaces.from_triangles(vertices, emission=EXITANCE)
    outward = np.einsum(
        "ij,ij->i",
        np.asarray(surfaces.normal),
        np.asarray(surfaces.centroid)
        - np.column_stack(
            [np.minimum(np.asarray(surfaces.centroid)[:, 0], X_TIP), np.zeros((len(vertices), 2))]
        ),
    )
    if (outward > 0).mean() < 0.5:
        surfaces = Surfaces.from_triangles(vertices[:, ::-1, :], emission=EXITANCE)
    power = float(np.sum(np.asarray(surfaces.area)) * EXITANCE)
    # ⚠️ The gather's chunk is a number of RECEIVERS, so at these facet counts the default
    # 4096 would form a chunk of 4096 x 270,000 entries -- about 9 GB, which is what killed the
    # first run of this study. Bound the entries instead, as the intersection test does.
    values = np.asarray(
        direct_fluence_rate(
            surfaces,
            jnp.asarray(receivers),
            absorption=UniformAbsorption(ABSORPTION),
            chunk_size=max(1, int(4_000_000 // len(vertices))),
        )
    )
    return values, power


def main() -> None:
    rng = np.random.default_rng(0)
    centres = np.load(WORK / "cell_centres.npy") if (WORK / "cell_centres.npy").exists() else None
    if centres is None:
        _say("reading the mesh")
        centres = np.asarray(read_openfoam(CASE).geometry().cell.centroid)
        np.save(WORK / "cell_centres.npy", centres)
    radius = np.hypot(centres[:, 1], centres[:, 2])
    chamber = (centres[:, 0] >= 0.0) & (centres[:, 0] <= 0.889) & (radius <= 0.0445)
    near = chamber & (radius - R_LAMP < 0.005) & (centres[:, 0] < X_TIP)
    bands = {
        "near the lamp (<5 mm)": np.flatnonzero(near),
        "the rest of the chamber": np.flatnonzero(chamber & ~near),
    }
    receivers = np.concatenate(
        [rng.choice(rows, size=min(SAMPLE, len(rows)), replace=False) for rows in bands.values()]
    )
    points = centres[receivers]
    _say(
        f"{len(points)} sample cells: "
        + ", ".join(f"{k} {min(SAMPLE, len(v))}" for k, v in bands.items())
    )

    summary = {}
    fine_sectors, fine_slices = LADDER[-1]
    reference_facets = lamp(fine_sectors, fine_slices)
    started = time.perf_counter()
    reference, reference_power = field(reference_facets, points)
    _say(
        f"reference {len(reference_facets)} facets ({fine_sectors}x{fine_slices}), "
        f"{reference_power:.4f} W, {time.perf_counter() - started:.0f} s"
    )

    stl = np.asarray(read_stl(CASE / "constant" / "triSurface" / "lampWall.stl").vertices)
    arms = [("the case's STL", stl)] + [
        (f"{sectors}x{slices}", lamp(sectors, slices)) for sectors, slices in LADDER[:-1]
    ]
    arms += drawing_lamps()
    for name, vertices in arms:
        started = time.perf_counter()
        values, power = field(vertices, points)
        elapsed = time.perf_counter() - started
        error = np.abs(values - reference) / reference
        scaled = np.abs(values * (reference_power / power) - reference) / reference
        summary[name] = {
            "facets": len(vertices),
            "power_W": round(power, 4),
            "seconds": round(elapsed, 1),
            "rays_for_the_full_mesh": float(len(centres) * len(vertices)),
        }
        for band, rows in bands.items():
            which = np.isin(receivers, rows)
            summary[name][band] = {
                "median": float(np.median(error[which])),
                "p99": float(np.percentile(error[which], 99)),
                "median_at_equal_power": float(np.median(scaled[which])),
            }
        _say(f"{name}: {json.dumps(summary[name])}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lamp_resolution.json").write_text(json.dumps(summary, indent=2) + "\n")
    _say(f"done: {OUT / 'lamp_resolution.json'}")


if __name__ == "__main__":
    main()
