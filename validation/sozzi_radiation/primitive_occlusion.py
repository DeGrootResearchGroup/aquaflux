"""Does the reactor described as CAD primitives shadow itself the way the hand-derived occluder does?

``compare_fluence.py`` shadows this reactor with :class:`~compare_fluence.BranchOpenings`: a
hand-derived closure that knows, for this reactor, that a cell in a pipe sees a lamp point
exactly when the segment crosses that pipe's opening. It is exact and it is fast, and it is a
description of **one reactor** -- somebody worked out where the openings are and wrote the
algebra for each.

This runs the general construction against it. The fluid is declared as three cylinders and
nothing else::

    Outside(chamber, inlet, riser)

and a segment is clear exactly when those three regions cover it end to end. No opening is
identified, no shadow edge is derived, and the same three lines describe a chamber-pipe-elbow
chain of any length. **If the two agree, the hand-derived occluder can be replaced by a
declaration** -- and the agreement is the measurement, because a general construction matching a
bespoke one is evidence only while the bespoke one is still there to disagree.

**Why it is cheap, and what is measured about that.** Each region is convex, so it clips a
segment to a single interval and the whole test is a handful of comparisons. Two points inside
one convex region need no test at all -- the straight line between them cannot leave -- which is
the structural reason a chamber full of cells costs nothing to shadow. That share is *counted*
here rather than asserted, per region and overall.

Receivers come from the meshed case when ``work/case`` exists, which is what makes this
comparable with the other harnesses in this directory. Without it the same three cylinders are
sampled directly, so the comparison still runs anywhere; the summary says which was used, because
the two are not the same population.

Run with ``validation/run_case.sh validation/sozzi_radiation/primitive_occlusion.py``.
``SOZZI_RECEIVERS`` overrides the sample size.
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
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.radiation import (  # noqa: E402
    NoOcclusion,
    Surfaces,
    UniformAbsorption,
    build_visibility,
    direct_fluence_rate,
)
from aquaflux.radiation.occluders import Cylinder, Outside  # noqa: E402
from compare_fluence import (  # noqa: E402
    ABSORPTION,
    CASE,
    EXITANCE,
    R_BODY,
    R_PIPE,
    WORK,
    X_BODY_END,
    X_RISER,
    BranchOpenings,
    lamp_surfaces,
)
from lamp_resolution import lamp as analytic_lamp  # noqa: E402

OUT = WORK / "compare"
RECEIVERS = int(os.environ.get("SOZZI_RECEIVERS", 24_000))
CHUNK = 2_000

#: How far the pipes reach back into the chamber. Neighbouring regions must overlap or touch --
#: a gap between them reads as solid, silently, as a shadow rather than as an error -- and a
#: curved junction leaves slivers of neither region unless one region is carried through it. The
#: overlap is inside the chamber, so it adds nothing to the fluid and costs nothing to describe.
REACH_BACK = 0.02
#: Where the pipes end. Beyond the meshed case's own extent, so the regions cannot cut a cell off.
INLET_END, RISER_TOP = 1.10, 0.40


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fluid() -> Outside:
    """The reactor's water, as three cylinders at the tutorial's own dimensions."""
    chamber = Cylinder(
        centre=[X_BODY_END / 2.0, 0.0, 0.0],
        axis=[1.0, 0.0, 0.0],
        radius=R_BODY,
        half_length=X_BODY_END / 2.0,
    )
    inlet_start = X_BODY_END - REACH_BACK
    inlet = Cylinder(
        centre=[(inlet_start + INLET_END) / 2.0, 0.0, 0.0],
        axis=[1.0, 0.0, 0.0],
        radius=R_PIPE,
        half_length=(INLET_END - inlet_start) / 2.0,
    )
    riser_foot = -R_BODY  # through the chamber, so the curved junction leaves no sliver
    riser = Cylinder(
        centre=[X_RISER, 0.0, (riser_foot + RISER_TOP) / 2.0],
        axis=[0.0, 0.0, 1.0],
        radius=R_PIPE,
        half_length=(RISER_TOP - riser_foot) / 2.0,
    )
    return Outside(chamber, inlet, riser)


def lamp() -> tuple[Surfaces, str]:
    """The emitting patch: the case's own STL where it exists, else an equivalent analytic lamp."""
    if (CASE / "constant" / "triSurface" / "lampWall.stl").exists():
        return lamp_surfaces(), "the case's lampWall.stl"
    # 32 x 128 is the rung of the resolution ladder that matches the STL's own error, so the
    # comparison is made against a lamp of the same fidelity rather than a coarser stand-in.
    surfaces = Surfaces.from_triangles(analytic_lamp(32, 128), emission=EXITANCE)
    outward = np.einsum(
        "ij,ij->i",
        np.asarray(surfaces.normal),
        np.asarray(surfaces.centroid)
        - np.column_stack(
            [
                np.minimum(np.asarray(surfaces.centroid)[:, 0], 0.80),
                np.zeros((surfaces.n_facets, 2)),
            ]
        ),
    )
    if (outward > 0).mean() < 0.5:
        surfaces = Surfaces.from_triangles(analytic_lamp(32, 128)[:, ::-1, :], emission=EXITANCE)
    return surfaces, "an analytic lamp, 32 sectors x 128 slices"


def receivers(rng, water: Outside) -> tuple[np.ndarray, str]:
    """Cells to compare on, from the meshed case if it is here and from the geometry if not."""
    cached = WORK / "cell_centres.npy"
    if cached.exists() or (CASE / "constant" / "polyMesh").exists():
        from aquaflux.io import read_openfoam

        if cached.exists():
            centres = np.load(cached)
        else:
            _say("reading the mesh")
            centres = np.asarray(read_openfoam(CASE).geometry().cell.centroid)
            cached.parent.mkdir(parents=True, exist_ok=True)
            np.save(cached, centres)
        # The pipes are where a mask does anything; the chamber is the control, where an
        # occluder that shadows one pair has a defect rather than a discretization.
        source = f"{len(centres)} cell centres of the meshed case"
        inside = np.flatnonzero(~np.asarray(water.contains(jnp.asarray(centres))))
        take = rng.choice(inside, size=min(RECEIVERS, len(inside)), replace=False)
        return centres[take], source
    box = np.array([[0.0, -R_BODY, -R_BODY], [INLET_END, R_BODY, RISER_TOP]])
    kept: list[np.ndarray] = []
    while sum(len(block) for block in kept) < RECEIVERS:
        trial = rng.uniform(box[0], box[1], (4 * RECEIVERS, 3))
        kept.append(trial[~np.asarray(water.contains(jnp.asarray(trial)))])
    return (
        np.concatenate(kept)[:RECEIVERS],
        "points sampled inside the three cylinders (no meshed case present)",
    )


def field(surfaces: Surfaces, points: np.ndarray, body) -> tuple[np.ndarray, float]:
    """``G`` at every point with ``body`` standing in the light, and what it cost."""
    water = UniformAbsorption(ABSORPTION)
    values = np.empty(len(points))
    started = time.perf_counter()
    for start in range(0, len(points), CHUNK):
        chunk = points[start : start + CHUNK]
        mask = build_visibility([body], surfaces, chunk, self_occlusion=NoOcclusion())
        values[start : start + CHUNK] = np.asarray(
            direct_fluence_rate(surfaces, jnp.asarray(chunk), absorption=water, visibility=mask)
        )
    return values, time.perf_counter() - started


def by_region(water: Outside, points: np.ndarray) -> dict:
    """How many of the receivers each region holds, so the reader can see what was exercised.

    The pair shares below are dominated by the chamber, which holds most of the fluid by volume.
    A count of receivers says separately whether the pipes -- the only place the mask does
    anything at all -- were sampled in useful numbers.
    """
    return {
        f"region {index}": int(
            (~np.asarray(region.signed_distance(jnp.asarray(points)) > 0.0)).sum()
        )
        for index, region in enumerate(water.regions)
    }


def shared_region(water: Outside, sources: np.ndarray, points: np.ndarray) -> dict:
    """What share of pairs lie in one convex region, and so are clear with no segment test.

    Counted from each end's region membership alone -- an outer product of two short boolean
    arrays -- because that is exactly the information the shortcut would act on. ⚠️ **The test
    as written does not skip those pairs**: it is one branch-free expression over the whole
    chunk, so every pair pays the same handful of comparisons. What convexity buys is that the
    handful is all there is, against a search over the wall's triangles; this number says how
    much of the reactor is answered by convexity alone rather than by the covering test finding
    an opening.
    """
    at_source = np.stack(
        [
            ~np.asarray(region.signed_distance(jnp.asarray(sources)) > 0.0)
            for region in water.regions
        ]
    )
    at_point = np.stack(
        [~np.asarray(region.signed_distance(jnp.asarray(points)) > 0.0) for region in water.regions]
    )
    together = np.zeros((len(points), len(sources)), dtype=bool)
    per_region = {}
    for index in range(len(water.regions)):
        both = at_point[index][:, None] & at_source[index][None, :]
        per_region[f"region {index}"] = float(both.mean())
        together |= both
    return {"share_in_one_convex_region": float(together.mean()), "by_region": per_region}


def main() -> None:
    rng = np.random.default_rng(0)
    water = fluid()
    surfaces, lamp_source = lamp()
    points, receiver_source = receivers(rng, water)
    _say(f"lamp: {surfaces.n_facets} facets from {lamp_source}")
    _say(f"receivers: {len(points)} from {receiver_source}")
    _say(f"fluid: {len(water.regions)} regions, {water.fluid.interval_count} intervals per ray")

    arms = {"primitives": water, "hand-derived": BranchOpenings()}
    fields, seconds, masks = {}, {}, {}
    for name, body in arms.items():
        fields[name], seconds[name] = field(surfaces, points, body)
        _say(
            f"  {name}: {seconds[name]:.1f} s for {len(points) * surfaces.n_facets / 1e6:.0f}M rays"
        )

    differing = np.zeros(len(points), dtype=int)
    for start in range(0, len(points), CHUNK):
        chunk = points[start : start + CHUNK]
        for name, body in arms.items():
            masks[name] = np.asarray(
                build_visibility([body], surfaces, chunk, self_occlusion=NoOcclusion()).blocked[0]
            )
        differing[start : start + CHUNK] = (masks["primitives"] != masks["hand-derived"]).sum(
            axis=1
        )

    lit = fields["hand-derived"] > 0.0
    relative = (
        np.abs(fields["primitives"][lit] - fields["hand-derived"][lit])
        / fields["hand-derived"][lit]
    )
    rays = len(points) * int(surfaces.n_facets)
    summary = {
        "lamp": {"facets": int(surfaces.n_facets), "source": lamp_source},
        "receivers": {"count": len(points), "source": receiver_source},
        "rays": rays,
        "regions": len(water.regions),
        "intervals_per_ray": water.fluid.interval_count,
        "seconds": {name: round(value, 2) for name, value in seconds.items()},
        "rays_per_second": {name: round(rays / value) for name, value in seconds.items()},
        "whole_mesh_seconds_at_this_rate": {
            name: round(1_635_909 * int(surfaces.n_facets) / (rays / value))
            for name, value in seconds.items()
        },
        "pairs_masked_differently": int(differing.sum()),
        "cells_with_any_pair_differing": int((differing > 0).sum()),
        "relative_field_difference": {
            "cells_compared": int(lit.sum()),
            "median": float(np.median(relative)),
            "p99": float(np.percentile(relative, 99)),
            "max": float(relative.max()),
        },
        "receivers_per_region": by_region(water, points),
        "convex_shortcut": shared_region(water, np.asarray(surfaces.centroid), points),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "primitive_occlusion.json").write_text(json.dumps(summary, indent=2) + "\n")
    _say(json.dumps(summary, indent=2))
    if summary["pairs_masked_differently"]:
        _say(
            "⚠️ the two occluders disagree. They describe the same ideal cylinders, so unlike the "
            "triangle comparison there is no faceting to explain a difference away."
        )


if __name__ == "__main__":
    main()
    sys.exit(0)
