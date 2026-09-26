"""What does the triangle grid's walk cost, and at which grid resolution?

``TriangleGrid.blocks`` walks each segment through the voxels it crosses, in one compiled loop per
ray (``aquaflux/radiation/grid_walk.py``), testing the triangles those voxels hold until one blocks
it. This times that walk on two scenes at the default grid and at multiples of it per axis, and
checks its answers against testing every triangle (``segment_is_cut``) on a sample of the rays.

The resolution sweep is there because the default's target of about ten triangles per occupied
voxel was tuned when the walk was array code, which paid ~150-180 ns per ray per voxel step in numpy
bookkeeping and so favoured coarse grids. A compiled step costs far less, so the best resolution
may have moved; this is the instrument for asking. The array walk this replaced is at commit
``d9fc017``, where this harness timed both side by side (17-37x, identical answers).

Two scenes, both triangles only, no primitives:

1. **a long thin vessel**: a cylindrical wall, 0.05 m in radius and 1.6 m long, 160 x 160 sectors
   and slices (51,200 triangles, close to the Sozzi reactor wall's 53,500); rays between random
   points inside it, a fifth of them aimed out through the wall so they are blocked.
2. **an annular reactor**: a lamp sleeve (radius 11.5 mm) inside a vessel wall (radius 44.5 mm),
   both 0.9 m long; rays from random points in the annulus to random sleeve facet centroids, the
   target facet excluded and a 1e-6 margin at the receiver, as a receiver mask casts them. The
   rays to the sleeve's far side are blocked by the sleeve itself.

Each resolution is warmed, then timed twice alternating with the others, and the faster pass kept.
The first call of a process also compiles the loop; that time is reported separately.

``RADIATION_GRID_WALK_RAYS`` sets the rays per scene (default 200,000) and
``RADIATION_GRID_WALK_CHECKED`` how many of them are checked against every triangle (default 5,000).

Run with ``validation/run_case.sh validation/radiation_grid_walk.py``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aquaflux  # noqa: F401  (enables x64)
from aquaflux.radiation.grid import TriangleGrid
from aquaflux.radiation.triangles import segment_is_cut
from tests.unit.radiation_references import cylinder_triangles

RAYS = int(os.environ.get("RADIATION_GRID_WALK_RAYS", "200000"))
CHECKED = int(os.environ.get("RADIATION_GRID_WALK_CHECKED", "5000"))
MULTIPLES = (0.5, 1, 2, 4)
PASSES = 2


def inside_cylinder(rng, count, inner, outer, half_length):
    """Points uniform in an annulus of the given radii (a disc when ``inner`` is zero)."""
    radius = np.sqrt(rng.uniform(inner**2, outer**2, count))
    angle = rng.uniform(0.0, 2.0 * np.pi, count)
    return np.column_stack(
        [
            radius * np.cos(angle),
            radius * np.sin(angle),
            rng.uniform(-half_length, half_length, count),
        ]
    )


def long_vessel(rng):
    """The long thin vessel: rays between interior points, a fifth aimed out through the wall."""
    wall = cylinder_triangles(0.05, 0.8, sectors=160, slices=160)
    origin = inside_cylinder(rng, RAYS, 0.0, 0.045, 0.75)
    target = inside_cylinder(rng, RAYS, 0.0, 0.045, 0.75)
    out = rng.uniform(size=RAYS) < 0.2
    target[out, :2] *= 2.0
    return wall, origin, target, np.full(RAYS, 1e-9), None


def annular_reactor(rng):
    """The annular reactor: receivers in the water, targets on the sleeve, the target excluded."""
    sleeve = cylinder_triangles(0.0115, 0.45, sectors=32, slices=128)
    wall = cylinder_triangles(0.0445, 0.45, sectors=96, slices=128)
    vertices = np.concatenate([sleeve, wall])
    facet = rng.integers(0, len(sleeve), RAYS)
    origin = inside_cylinder(rng, RAYS, 0.013, 0.043, 0.44)
    target = sleeve[facet].mean(axis=1)
    exclude = facet[:, None]
    return vertices, origin, target, np.full(RAYS, 1e-6), exclude


def timed(grid, origin, target, near, exclude):
    start = time.perf_counter()
    found = grid.blocks(origin, target, near, exclude=exclude)
    return time.perf_counter() - start, found


def main() -> None:
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    TriangleGrid.build(cylinder_triangles(1.0, 1.0, 8, 8)).blocks(
        np.zeros((1, 3)), np.full((1, 3), 2.0), np.zeros(1)
    )
    print(
        f"first call of the process, including compiling the loop: "
        f"{time.perf_counter() - start:.2f} s",
        flush=True,
    )
    for name, scene in (("long thin vessel", long_vessel), ("annular reactor", annular_reactor)):
        vertices, origin, target, near, exclude = scene(rng)
        sample = slice(0, CHECKED)
        brute = np.asarray(
            segment_is_cut(
                origin[sample], target[sample], vertices, near[sample],
                exclude=None if exclude is None else exclude[sample],
            )
        )  # fmt: skip
        print(
            f"\n### {name}: {len(vertices):,} triangles, {RAYS:,} rays; fastest of {PASSES} "
            f"alternating warm passes; the first {CHECKED:,} checked against every triangle\n",
            flush=True,
        )
        print(f"{'grid':>18} {'x default':>9} {'rays/s':>11} {'blocked':>8} {'matches':>8}")
        default = TriangleGrid.build(vertices).resolution
        grids = {
            m: TriangleGrid.build(
                vertices, resolution=tuple(np.maximum(np.round(m * default), 1).astype(int))
            )
            for m in MULTIPLES
        }
        for grid in grids.values():
            timed(
                grid,
                origin[:2000],
                target[:2000],
                near[:2000],
                None if exclude is None else exclude[:2000],
            )
        best, answers = {}, {}
        for _ in range(PASSES):
            for m, grid in grids.items():
                seconds, answers[m] = timed(grid, origin, target, near, exclude)
                best[m] = min(best.get(m, np.inf), seconds)
        for m, grid in grids.items():
            shape = "x".join(str(int(n)) for n in grid.resolution)
            print(
                f"{shape:>18} {m:9g} {RAYS / best[m]:11,.0f} {answers[m].mean():8.3f} "
                f"{np.array_equal(answers[m][sample], brute)!s:>8}",
                flush=True,
            )


if __name__ == "__main__":
    main()
