"""What does walking a segment through the triangle grid cost, as array passes and as a compiled loop?

``TriangleGrid.blocks`` walks each segment through the voxels it crosses and tests the triangles
they hold. It has two walks that visit the same voxels and test the same triangles in the same order:

- ``walk="array"`` steps every ray still in flight together, as whole-array numpy operations, and
  tests each step's (ray, triangle) pairs in one traced call;
- ``walk="compiled"`` walks each ray to its end in one compiled loop (Numba), one ray per iteration
  of a parallel loop.

The table times both on the same rays and checks that they answer identically, at the default grid
and at finer ones -- the array walk's cost per voxel step is what makes a finer grid slower for it,
so the resolution sweep is where the two differ in kind and not only in rate.

Two scenes, both triangles only, no primitives:

1. **a long thin vessel**: a cylindrical wall, 0.05 m in radius and 1.6 m long, 160 x 160 sectors
   and slices (51,200 triangles, close to the Sozzi reactor wall's 53,500); rays between random
   points inside it, a fifth of them aimed out through the wall so they are blocked.
2. **an annular reactor**: a lamp sleeve (radius 11.5 mm) inside a vessel wall (radius 44.5 mm),
   both 0.9 m long; rays from random points in the annulus to random sleeve facet centroids, the
   target facet excluded and a 1e-6 margin at the receiver, as a receiver mask casts them. The
   rays to the sleeve's far side are blocked by the sleeve itself.

Each arm is warmed, then run twice alternating with the other, and the faster pass kept. The first
compiled call of a process also compiles the loop; that time is reported separately.

``RADIATION_GRID_WALK_RAYS`` sets the rays per scene (default 200,000).

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
from tests.unit.radiation_references import cylinder_triangles

RAYS = int(os.environ.get("RADIATION_GRID_WALK_RAYS", "200000"))
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


def timed(grid, walk, origin, target, near, exclude):
    start = time.perf_counter()
    found = grid.blocks(origin, target, near, exclude=exclude, walk=walk)
    return time.perf_counter() - start, found


def main() -> None:
    rng = np.random.default_rng(0)
    first_compile = None
    for name, scene in (("long thin vessel", long_vessel), ("annular reactor", annular_reactor)):
        vertices, origin, target, near, exclude = scene(rng)
        print(
            f"\n### {name}: {len(vertices):,} triangles, {RAYS:,} rays; fastest of {PASSES} "
            "alternating warm passes\n",
            flush=True,
        )
        print(
            f"{'grid':>18} {'array rays/s':>13} {'compiled rays/s':>16} {'speed-up':>9} "
            f"{'identical':>10} {'blocked':>8}"
        )
        for resolution in (None, 2, 4):
            grid = TriangleGrid.build(vertices)
            if resolution is not None:  # a multiple of the default, per axis
                grid = TriangleGrid.build(vertices, resolution=tuple(resolution * grid.resolution))
            warm = slice(0, min(RAYS, 5000))
            ex = None if exclude is None else exclude[warm]
            for walk in ("array", "compiled"):
                seconds, _ = timed(grid, walk, origin[warm], target[warm], near[warm], ex)
                if walk == "compiled" and first_compile is None:
                    first_compile = seconds
            best, answers = {}, {}
            for _ in range(PASSES):
                for walk in ("array", "compiled"):
                    seconds, answers[walk] = timed(grid, walk, origin, target, near, exclude)
                    best[walk] = min(best.get(walk, np.inf), seconds)
            shape = "x".join(str(int(n)) for n in grid.resolution)
            print(
                f"{shape:>18} {RAYS / best['array']:13,.0f} {RAYS / best['compiled']:16,.0f} "
                f"{best['array'] / best['compiled']:8.1f}x "
                f"{np.array_equal(answers['array'], answers['compiled'])!s:>10} "
                f"{answers['compiled'].mean():8.3f}",
                flush=True,
            )
    print(
        f"\nfirst compiled call of the process, including compiling the loop: {first_compile:.2f} s"
    )


if __name__ == "__main__":
    main()
