"""Does the grid-accelerated ray mask reproduce the analytic one, on the real reactor?

``compare_fluence.py`` shadows this reactor *analytically*: the fluid is three convex cylinders,
so a cell in a pipe sees a lamp point exactly when the segment between them leaves the chamber
through that pipe's opening (:class:`~compare_fluence.BranchOpenings`). That is exact, and it is
a description of one reactor rather than a method. This runs the general method against it --
the vessel wall as the 53,500 triangles its STL actually is, culled by
:class:`~aquaflux.radiation.grid.TriangleGrid` -- and asks whether the two agree.

**They cannot agree exactly, and the disagreement is the measurement.** The analytic occluder
uses the ideal cylinders the geometry was designed from; the STL is a faceted approximation of
them, so its pipe openings are polygons inscribed in the design circles and its walls sit
slightly inside the design radius. Pairs whose sight line passes near an opening's rim are
therefore genuinely blocked by one and not the other. What this measures is how much of the
field that moves, which is the discretization of the *wall*, not an error in either mask.

**What it costs.** The chamber's cells see the whole lamp and the pipes' cells are where the
mask does anything, so the receivers are all sampled from the two pipes, plus a chamber sample
as a control -- there the grid must find nothing, and anything it finds is the faceted wall
cutting a line the ideal one does not.

Run with ``validation/run_case.sh validation/sozzi_radiation/grid_mask_check.py`` after
``generate_dom_reference.py`` has produced ``work/case``. ``SOZZI_PIPE_CELLS`` and
``SOZZI_CHAMBER_CELLS`` override the sample sizes; ``SOZZI_GRID`` overrides the grid resolution.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.io import read_openfoam  # noqa: E402
from aquaflux.radiation import (  # noqa: E402
    NoOcclusion,
    UniformAbsorption,
    build_visibility,
    direct_fluence_rate,
    read_stl,
)
from aquaflux.radiation.grid import TriangleGrid  # noqa: E402
from aquaflux.radiation.occluders import Occluder  # noqa: E402
from compare_fluence import (  # noqa: E402
    ABSORPTION,
    CASE,
    WORK,
    BranchOpenings,
    _in_chamber,
    lamp_surfaces,
)

OUT = WORK / "compare"
CHUNK = 1_000  # receivers per pass: with 7,516 facets that is ~7e6 rays of walking at a time
PIPE_CELLS = int(os.environ.get("SOZZI_PIPE_CELLS", 40_000))
CHAMBER_CELLS = int(os.environ.get("SOZZI_CHAMBER_CELLS", 4_000))
RESOLUTION = os.environ.get("SOZZI_GRID")


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class WallTriangles(Occluder):
    """The vessel wall as the triangles it really is, culled by a uniform grid.

    Attributes
    ----------
    grid : TriangleGrid
        Held as static: the mask is frozen and built on the host, and this body carries plain
        arrays rather than anything traced.
    """

    grid: TriangleGrid = eqx.field(static=True)

    def contains(self, position) -> jnp.ndarray:
        """Nothing is inside this body, and here that is exact rather than a simplification.

        The wall is a zero-thickness sheet bounding the fluid: the metal is on its far side, so
        no cell and no lamp facet is embedded in it. The guard this stands in for -- refusing a
        scene whose points sit inside a solid -- is still applied to this run by the analytic
        arm, whose own ``contains`` is exact for the three cylinders.
        """
        return jnp.zeros(jnp.asarray(position).shape[:-1], dtype=bool)

    def blocks(self, origin, target, min_distance) -> jnp.ndarray:
        """See :meth:`aquaflux.radiation.occluders.Occluder.blocks`."""
        source, receiver = np.broadcast_arrays(
            np.asarray(origin, dtype=float), np.asarray(target, dtype=float)
        )
        shape = source.shape[:-1]
        near = np.broadcast_to(np.asarray(min_distance, dtype=float), shape)
        blocked = self.grid.blocks(source.reshape(-1, 3), receiver.reshape(-1, 3), near.reshape(-1))
        return jnp.asarray(blocked.reshape(shape))


def receivers(rng) -> tuple[np.ndarray, np.ndarray]:
    """Cells to compare on: a sample from the pipes, and a control sample from the chamber."""
    cached = WORK / "cell_centres.npy"
    if cached.exists():
        centres = np.load(cached)
    else:
        _say("reading the mesh")
        centres = np.asarray(read_openfoam(CASE).geometry().cell.centroid)
        np.save(cached, centres)
    chamber = np.asarray(_in_chamber(jnp.asarray(centres)))
    piped = np.flatnonzero(~chamber & ~np.asarray(BranchOpenings().contains(jnp.asarray(centres))))
    inside = np.flatnonzero(chamber)
    take = np.concatenate(
        [
            rng.choice(piped, size=min(PIPE_CELLS, len(piped)), replace=False),
            rng.choice(inside, size=min(CHAMBER_CELLS, len(inside)), replace=False),
        ]
    )
    _say(
        f"{len(take)} receivers: {min(PIPE_CELLS, len(piped))} of {len(piped)} pipe cells, "
        f"{min(CHAMBER_CELLS, len(inside))} of {len(inside)} chamber cells"
    )
    return centres[take], np.concatenate(
        [
            np.ones(min(PIPE_CELLS, len(piped)), bool),
            np.zeros(min(CHAMBER_CELLS, len(inside)), bool),
        ]
    )


def main() -> None:
    rng = np.random.default_rng(0)
    lamp = lamp_surfaces()
    points, in_pipe = receivers(rng)
    water = UniformAbsorption(ABSORPTION)

    wall = np.asarray(read_stl(CASE / "constant" / "triSurface" / "bodyWall.stl").vertices)
    started = time.perf_counter()
    grid = TriangleGrid.build(wall, resolution=None if RESOLUTION is None else int(RESOLUTION))
    held = np.diff(grid.starts)
    _say(
        f"grid over {len(wall)} wall triangles: {tuple(int(n) for n in grid.resolution)}, "
        f"{int((held > 0).sum())} occupied voxels holding {held[held > 0].mean():.1f} each, "
        f"built in {time.perf_counter() - started:.1f} s"
    )

    bodies = {"analytic": BranchOpenings(), "grid": WallTriangles(grid=grid)}
    fields = {name: np.empty(len(points)) for name in bodies}
    differing = np.zeros(len(points), dtype=int)
    seconds = dict.fromkeys(bodies, 0.0)
    rays = len(points) * lamp.n_facets

    started = time.perf_counter()
    at_rim = []
    for start in range(0, len(points), CHUNK):
        chunk = points[start : start + CHUNK]
        masks = {}
        for name, body in bodies.items():
            at = time.perf_counter()
            mask = build_visibility([body], lamp, chunk, self_occlusion=NoOcclusion())
            fields[name][start : start + CHUNK] = np.asarray(
                direct_fluence_rate(lamp, jnp.asarray(chunk), absorption=water, visibility=mask)
            )
            seconds[name] += time.perf_counter() - at
            masks[name] = np.asarray(mask.blocked[0])
        disagrees = masks["analytic"] != masks["grid"]
        differing[start : start + CHUNK] = disagrees.sum(axis=1)
        # Where each disputed sight line crosses the opening, in units of its radius. If the
        # two masks differ only because one describes the opening as a circle and the other as
        # the polygon an STL makes of it, the disputed lines pass at the rim -- and if they do
        # not, something is wrong with a mask rather than with the geometry.
        if disagrees.any():
            ratio = np.asarray(
                bodies["analytic"].crossing_ratio(
                    jnp.asarray(lamp.centroid)[None, :, :], jnp.asarray(chunk)[:, None, :]
                )
            )
            at_rim.append(ratio[disagrees])
        if (start // CHUNK) % 5 == 0 or start + CHUNK >= len(points):
            done = min(start + CHUNK, len(points))
            _say(
                f"  {done}/{len(points)} cells, {done * lamp.n_facets / 1e6:.0f}M rays, "
                f"analytic {seconds['analytic']:.0f} s, grid {seconds['grid']:.0f} s"
            )
    elapsed = time.perf_counter() - started

    relative = np.abs(fields["grid"] - fields["analytic"]) / fields["analytic"]
    summary = {
        "wall_triangles": len(wall),
        "lamp_facets": int(lamp.n_facets),
        "grid_resolution": [int(n) for n in grid.resolution],
        "receivers": len(points),
        "rays": int(rays),
        "seconds": {name: round(value, 1) for name, value in seconds.items()},
        "grid_rays_per_second": round(rays / seconds["grid"], 1),
        "whole_mesh_hours_at_this_rate": round(
            1_635_909 * lamp.n_facets / (rays / seconds["grid"]) / 3600.0, 2
        ),
        "wall_clock_s": round(elapsed, 1),
    }
    for label, rows in (("pipes", in_pipe), ("chamber", ~in_pipe)):
        if not rows.any():
            continue
        summary[label] = {
            "cells": int(rows.sum()),
            "pairs_masked_differently_per_cell_mean": float(differing[rows].mean()),
            "cells_with_any_pair_differing": int((differing[rows] > 0).sum()),
            "relative_difference_median": float(np.median(relative[rows])),
            "relative_difference_p99": float(np.percentile(relative[rows], 99)),
            "relative_difference_max": float(relative[rows].max()),
        }
    if at_rim:
        ratio = np.concatenate(at_rim)
        finite = ratio[np.isfinite(ratio)]
        summary["where_the_masks_disagree"] = {
            "pairs": len(ratio),
            "crossing_radius_over_opening_radius": {
                "p1": float(np.percentile(finite, 1)),
                "median": float(np.median(finite)),
                "p99": float(np.percentile(finite, 99)),
            },
            "share_within_10pc_of_the_rim": float(np.mean(np.abs(finite - 1.0) < 0.1)),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "grid_mask_check.json").write_text(json.dumps(summary, indent=2) + "\n")
    _say(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
