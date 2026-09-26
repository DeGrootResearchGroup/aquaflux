"""How much of the Sozzi reactor's analytic shadow mask shaft culling decides without a test.

The fluid is ``Outside(chamber, inlet, riser)``, three cylinders, as in ``primitive_occlusion.py``,
and the lamp is the case's own STL (or the analytic 32 x 128 lamp when the case is absent). The
mask is built two ways in **one process on one ray set**:

- :class:`~aquaflux.radiation.EveryPair`: every body against every pair -- the shipped default;
- :class:`~aquaflux.radiation.ShaftCulling`, at several group sizes: receivers and facet
  centroids grouped along a space-filling curve, a tile certified clear wherever every point of
  it lies in one convex region, and only the remaining tiles tested pair by pair.

What is reported, per arm: whether the mask is **bit-identical** to the reference (the claim, so
any difference is a defect); the share of pairs certified without a test; and the wall clock of
the build. Timings alternate between the arms, two passes after a warm-up that pays compilation,
and the fastest pass per arm is kept with the spread between passes beside it -- a ratio between
two arms is read from one process, never across runs.

Receivers come from the meshed case when ``work/case`` exists, as in ``primitive_occlusion.py``,
and are otherwise sampled uniformly inside the three cylinders. The two populations are not the
same -- the mesh refines towards the walls and the lamp, which is where the uncertifiable pairs
are -- so the summary says which was used, and a certified share from one is not a figure for the
other.

``SOZZI_RECEIVERS`` overrides the sample size; ``SOZZI_CULLING_SIZES`` the arms, separated by
semicolons, each ``blocks:clusters`` with its coarse-to-fine sizes separated by commas -- the
default ``32:32;32,8:32,8;64,16:64,16`` is one level at 32 and two refinement ladders.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.radiation import EveryPair, ShaftCulling  # noqa: E402
from primitive_occlusion import OUT, fluid, lamp, receivers  # noqa: E402


def culling_arms(spec: str) -> dict[str, ShaftCulling]:
    """The shaft-culling arms a ``blocks:clusters;blocks:clusters`` specification names."""
    arms = {}
    for arm in spec.split(";"):
        blocks, clusters = (tuple(int(size) for size in side.split(",")) for side in arm.split(":"))
        name = "shaft " + "/".join(f"{b}x{c}" for b, c in zip(blocks, clusters, strict=True))
        arms[name] = ShaftCulling(receiver_blocks=blocks, source_clusters=clusters)
    return arms


SIZES = os.environ.get("SOZZI_CULLING_SIZES", "32:32;32,8:32,8;64,16:64,16")
#: The mask build's own default margin at the source end, as a fraction of sqrt(facet area).
OFFSET_SCALE = 1e-6
PASSES = 2


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _timed(arm, bodies, sources, near, points):
    """One build of the mask, returning the mask and its wall clock."""
    started = time.perf_counter()
    mask = np.asarray(arm.blocked(bodies, sources, near, points))
    return mask, time.perf_counter() - started


def main() -> None:
    rng = np.random.default_rng(0)
    water = fluid()
    surfaces, lamp_source = lamp()
    points, population = receivers(rng, water)
    sources = np.asarray(surfaces.centroid)
    near = OFFSET_SCALE * np.sqrt(np.asarray(surfaces.area))
    bodies = [water]
    n_pairs = len(points) * len(sources)
    region = np.argmin(
        np.stack([np.asarray(r.signed_distance(jnp.asarray(points))) for r in water.regions]),
        axis=0,
    )
    _say(
        f"{len(points)} receivers ({population}; deepest inside chamber/inlet/riser = "
        f"{np.bincount(region, minlength=3).tolist()}), {len(sources)} facets ({lamp_source}), "
        f"{n_pairs:,} pairs; jax {jax.__version__}, {platform.system()} {platform.machine()}, "
        f"{os.cpu_count()} cores"
    )

    arms = {"every pair": EveryPair()} | culling_arms(SIZES)
    reference = None
    results = {}
    for name, arm in arms.items():
        mask, seconds = _timed(arm, bodies, sources, near, points)
        if reference is None:
            reference = mask
        certified = (
            int(arm.certified_pairs(bodies, sources, points)[0])
            if isinstance(arm, ShaftCulling)
            else 0
        )
        results[name] = {
            "identical": bool(np.array_equal(mask, reference)),
            "differing_pairs": int(np.count_nonzero(mask != reference)),
            "certified_share": certified / n_pairs,
            "warm_up_s": seconds,
            "passes_s": [],
        }
        _say(
            f"{name}: warm-up {seconds:.2f} s, certified {certified / n_pairs:.1%}, "
            f"identical to every-pair: {results[name]['identical']}"
        )
    del mask

    for index in range(PASSES):
        for name, arm in arms.items():
            _, seconds = _timed(arm, bodies, sources, near, points)
            results[name]["passes_s"].append(seconds)
            _say(f"pass {index + 1}, {name}: {seconds:.2f} s")

    base = min(results["every pair"]["passes_s"])
    _say("summary (fastest pass; spread is slowest over fastest pass of the same arm):")
    for name, result in results.items():
        fastest = min(result["passes_s"])
        result["fastest_s"] = fastest
        result["spread"] = max(result["passes_s"]) / fastest
        result["pairs_per_s"] = n_pairs / fastest
        _say(
            f"  {name:>16}: {fastest:7.2f} s  {n_pairs / fastest / 1e6:7.1f} M pairs/s  "
            f"spread {result['spread']:.2f}x  speed-up {base / fastest:5.2f}x  certified "
            f"{result['certified_share']:6.1%}  identical {result['identical']}"
        )
    blocked_share = float(reference.mean())
    _say(f"pairs blocked by the fluid: {blocked_share:.2%}")

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "receivers": len(points),
        "population": population,
        "lamp": lamp_source,
        "facets": len(sources),
        "blocked_share": blocked_share,
        "jax": jax.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "cores": os.cpu_count(),
        "arms": results,
    }
    path = OUT / "body_culling.json"
    path.write_text(json.dumps(summary, indent=2))
    _say(f"wrote {path}")
    if not all(result["identical"] for result in results.values()):
        raise SystemExit("a culled mask differs from the every-pair mask")


if __name__ == "__main__":
    main()
