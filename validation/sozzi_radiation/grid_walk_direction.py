"""What does a segment cost the triangle grid's walk on the real reactor, and what moves it?

``TriangleGrid.blocks`` walks each segment from its origin towards its target and stops at the
first triangle that blocks it. A shadow mask hands it the lamp facet as the origin. The lamp sits
on the axis in open water, and what blocks a sight line to a cell in an outlet pipe is the pipe's
own wall, beside the cell -- so it was reasoned (#503) that walking from the cell would stop
sooner, and that the rest of the cost was the walk through empty voxels. This measures both, on
the 53,500 triangles of ``bodyWall.stl`` and the 7,516 lamp facets, for three receiver populations
-- cells in the outlet pipes, cells in the chamber, and cells drawn uniformly from the whole mesh
-- at whichever grids are asked for.

It prints two tables. The first is the rate both ways round, and how many answers the reversal
changed. The second is where a lamp-to-cell walk spends its steps, from a copy of the walk with
counters (:func:`_counted_walk`): voxels stepped, how many were occupied, triangles tested and how
many were distinct, over each population's blocked and clear segments. The copy's answers are
compared with ``TriangleGrid.blocks`` ray for ray, so a copy that drifted from the walk it
describes says so.

**The reversed walk is not the same predicate, exactly.** The mask's margin is at the lamp end:
a hit within ``1e-6 * sqrt(facet area)`` of the facet does not count. Walked from the cell there is
no way to say that, so the reversed walk runs with no margin at either end, and the agreement
column says how many answers that changed. It is a timing arm, not a candidate implementation.

Every corner is timed in one process: warmed, then two passes alternating over all corners, the
faster pass kept.

``SOZZI_DIRECTION_CELLS`` sets the cells per population (default 40, which with every lamp facet is
300,640 segments per corner). ``SOZZI_DIRECTION_GRIDS`` lists the grids, comma-separated: ``x<m>``
is the default times ``m`` per axis, ``a:b:c`` those voxels per axis, and a bare number that many
voxels on every axis (default ``212:11:114,x1,64:128:128`` -- the near-cubic grid that was the
default before #503, the default, and the best grid measured on this wall).

Run with ``validation/run_case.sh validation/sozzi_radiation/grid_walk_direction.py``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import jax.numpy as jnp  # noqa: E402
import numba  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.radiation import grid as grid_module  # noqa: E402
from aquaflux.radiation import read_stl  # noqa: E402
from aquaflux.radiation.grid import TriangleGrid  # noqa: E402
from aquaflux.radiation.grid_walk import _cuts, _ray_frame  # noqa: E402
from compare_fluence import CASE, WORK, BranchOpenings, _in_chamber, lamp_surfaces  # noqa: E402

CELLS = int(os.environ.get("SOZZI_DIRECTION_CELLS", 40))
PASSES = 2
#: The grids timed; see the module docstring.
GRIDS = os.environ.get("SOZZI_DIRECTION_GRIDS", "212:11:114,x1,64:128:128").split(",")
OFFSET_SCALE = 1e-6  # build_visibility's default margin, relative to each facet's size


@numba.njit(parallel=True)
def _counted_walk(
    ray, voxel, until, step, delta, origin, direction, near, vertices, starts, triangles,
    resolution, max_steps, blocked, steps, occupied, tested, distinct,
):  # fmt: skip
    """``grid_walk._walk``'s loop with four counters per ray, and no exclusions.

    ``distinct`` counts the different triangles a ray tested, from a stamp per triangle holding the
    last ray that tested it -- one stamp array per block of rays, so each thread has its own.

    A copy of the stepping, kept here rather than in the library so the shipped loop carries no
    counters; :func:`counted` checks its answers against ``TriangleGrid.blocks`` ray for ray, so a
    copy that drifted from the walk it describes would say so.
    """
    block = 4096
    for first in numba.prange((len(ray) + block - 1) // block):
        stamp = np.full(len(vertices), -1)
        for live in range(first * block, min((first + 1) * block, len(ray))):
            _count_one(
                live, ray, voxel, until, step, delta, origin, direction, near, vertices, starts,
                triangles, resolution, max_steps, blocked, steps, occupied, tested, distinct, stamp,
            )  # fmt: skip


@numba.njit
def _count_one(
    live, ray, voxel, until, step, delta, origin, direction, near, vertices, starts, triangles,
    resolution, max_steps, blocked, steps, occupied, tested, distinct, stamp,
):  # fmt: skip
    """One ray of :func:`_counted_walk`."""
    if True:
        r = ray[live]
        frame = _ray_frame(direction[r])
        at = voxel[live].copy()
        crossing = until[live].copy()
        none = np.full(1, -1)
        for _ in range(max_steps):
            steps[r] += 1
            flat = (at[0] * resolution[1] + at[1]) * resolution[2] + at[2]
            if starts[flat + 1] > starts[flat]:
                occupied[r] += 1
            for entry in range(starts[flat], starts[flat + 1]):
                triangle = triangles[entry]
                tested[r] += 1
                if stamp[triangle] != r:
                    stamp[triangle] = r
                    distinct[r] += 1
                if _cuts(origin[r], frame, near[r], vertices[triangle], triangle, none):
                    blocked[r] = True
                    break
            if blocked[r]:
                break
            axis = 0
            if crossing[1] < crossing[axis]:
                axis = 1
            if crossing[2] < crossing[axis]:
                axis = 2
            leaving = crossing[axis]
            moved = at[axis] + step[live, axis]
            if not (leaving <= 1.0 and 0 <= moved < resolution[axis]):
                break
            at[axis] = moved
            crossing[axis] = leaving + delta[live, axis]


def counted(grid, origin, target, near):
    """Per segment: blocked, voxels stepped, occupied ones, triangles tested, distinct ones."""
    direction = target - origin
    length = np.sqrt(np.sum(direction * direction, axis=-1))
    share = near / np.where(length == 0.0, 1.0, length)
    alive, entry = grid_module._enters_grid(
        origin, direction, grid.low, grid.spacing, grid.resolution
    )
    ray = np.flatnonzero(alive)
    voxel, until, step, delta = grid_module._walk_state(
        origin[ray], direction[ray], entry[ray], grid.low, grid.spacing, grid.resolution
    )
    n = len(origin)
    blocked = np.zeros(n, dtype=bool)
    steps, occupied, tested, distinct = (np.zeros(n, dtype=np.int64) for _ in range(4))
    _counted_walk(
        ray, voxel.astype(np.int64), until, step.astype(np.int64), delta, origin, direction,
        share, grid.vertices, grid.starts.astype(np.int64), grid.triangles.astype(np.int64),
        grid.resolution.astype(np.int64), int(grid.resolution.sum()) + 3, blocked, steps,
        occupied, tested, distinct,
    )  # fmt: skip
    return blocked, steps, occupied, tested, distinct


def populations(rng) -> dict[str, np.ndarray]:
    """Cell centres from the outlet pipes, from the chamber, and from the whole mesh."""
    centres = np.load(WORK / "cell_centres.npy")
    chamber = np.asarray(_in_chamber(jnp.asarray(centres)))
    solid = np.asarray(BranchOpenings().contains(jnp.asarray(centres)))
    piped = np.flatnonzero(~chamber & ~solid)
    inside = np.flatnonzero(chamber)
    return {
        "pipe cells": centres[rng.choice(piped, CELLS, replace=False)],
        "chamber cells": centres[rng.choice(inside, CELLS, replace=False)],
        "any cell": centres[rng.choice(len(centres), CELLS, replace=False)],
    }


def segments(lamp, cells):
    """Every (cell, facet) segment: lamp-side start, cell-side end, and the lamp-end margin."""
    centroid = np.asarray(lamp.centroid)
    margin = OFFSET_SCALE * np.sqrt(np.asarray(lamp.area))
    source = np.broadcast_to(centroid[None], (len(cells), *centroid.shape)).reshape(-1, 3)
    receiver = np.broadcast_to(cells[:, None], (len(cells), *centroid.shape)).reshape(-1, 3)
    near = np.broadcast_to(margin[None], (len(cells), len(margin))).reshape(-1)
    return np.ascontiguousarray(source), np.ascontiguousarray(receiver), np.ascontiguousarray(near)


def main() -> None:
    rng = np.random.default_rng(0)
    lamp = lamp_surfaces()
    wall = np.asarray(read_stl(CASE / "constant" / "triSurface" / "bodyWall.stl").vertices)
    default = TriangleGrid.build(wall).resolution
    grids = {}
    for arm in GRIDS:
        if arm.startswith("x"):
            multiple = float(arm[1:])
            counts = tuple(int(n) for n in np.maximum(np.round(multiple * default), 1))
            name = "default" if multiple == 1 else f"default {arm}"
        elif ":" in arm:
            counts, name = tuple(int(n) for n in arm.split(":")), arm.replace(":", "x")
        else:
            counts, name = int(arm), f"{arm}^3"
        start = time.perf_counter()
        grids[name] = TriangleGrid.build(wall, resolution=counts)
        print(f"built {name} in {time.perf_counter() - start:.1f} s", flush=True)
    for name, grid in grids.items():
        held = np.diff(grid.starts)
        print(
            f"{name}: {tuple(int(n) for n in grid.resolution)}, voxel "
            f"{', '.join(f'{1e3 * s:.1f}' for s in grid.spacing)} mm, "
            f"{int((held > 0).sum()):,} occupied holding {held[held > 0].mean():.1f} each",
            flush=True,
        )
    cases = {label: segments(lamp, cells) for label, cells in populations(rng).items()}
    rays = len(next(iter(cases.values()))[0])
    print(
        f"\n{len(wall):,} wall triangles, {lamp.n_facets:,} lamp facets, {CELLS} cells per "
        f"population, {rays:,} segments per corner; fastest of {PASSES} alternating warm passes\n",
        flush=True,
    )

    def walk(grid, case, direction):
        source, receiver, near = case
        if direction == "lamp -> cell":
            return grid.blocks(source, receiver, near)
        return grid.blocks(receiver, source, np.zeros_like(near))

    corners = [(g, p, d) for g in grids for p in cases for d in ("lamp -> cell", "cell -> lamp")]
    for g, p, d in corners:  # warm: compile once, touch every array once
        walk(grids[g], tuple(a[:2000] for a in cases[p]), d)
    best, answers = {}, {}
    for _ in range(PASSES):
        for corner in corners:
            g, p, d = corner
            start = time.perf_counter()
            answers[corner] = walk(grids[g], cases[p], d)
            best[corner] = min(best.get(corner, np.inf), time.perf_counter() - start)

    print(
        f"{'grid':>20} {'receivers':>14} {'lamp->cell':>11} {'cell->lamp':>11} {'reversed is':>11} "
        f"{'blocked':>8} {'answers differ':>14}"
    )
    for g in grids:
        for p in cases:
            forward, backward = (g, p, "lamp -> cell"), (g, p, "cell -> lamp")
            print(
                f"{g:>20} {p:>14} {rays / best[forward]:11,.0f} {rays / best[backward]:11,.0f} "
                f"{best[forward] / best[backward]:10.2f}x {answers[forward].mean():8.3f} "
                f"{int((answers[forward] != answers[backward]).sum()):14,}",
                flush=True,
            )

    print(
        "\nWhere a lamp -> cell walk spends its steps, per segment (mean over the population, "
        "then over its blocked and clear segments)\n"
    )
    print(
        f"{'grid':>20} {'receivers':>22} {'segments':>9} {'steps':>7} {'occupied':>9} "
        f"{'tested':>8} {'distinct':>9} {'same answers':>12}"
    )
    for g, grid in grids.items():
        for p, (source, receiver, near) in cases.items():
            blocked, steps, occupied, tested, distinct = counted(grid, source, receiver, near)
            same = np.array_equal(blocked, answers[(g, p, "lamp -> cell")])
            for label, rows in (("all", slice(None)), ("blocked", blocked), ("clear", ~blocked)):
                if not np.any(np.ones(len(blocked), bool)[rows]):
                    continue
                print(
                    f"{g:>20} {p + ' ' + label:>22} {len(steps[rows]):9,} "
                    f"{steps[rows].mean():7.1f} {occupied[rows].mean():9.1f} "
                    f"{tested[rows].mean():8.1f} {distinct[rows].mean():9.1f} {same!s:>12}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
