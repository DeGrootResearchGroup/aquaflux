"""What should one pass of the gather be allowed to form? Measured, not copied.

The gather, the shadow-mask build and the ray test all visit receivers in passes of
receiver-by-facet pairs, bounded by ``pair_limit``. The default was first copied from the
intersection test's own budget of four million entries; but a gather entry is not an
intersection-test entry -- it carries a solid angle, an absorption factor and a radiance -- so the
right number is whatever this measures.

**What is swept.** Three analytic lamps (the Sozzi & Taghipour lamp, a cylinder with a hemispherical
tip, at 8,704 / 67,584 / 270,336 facets) against receivers drawn uniformly in the chamber, at a
range of limits, on two paths. The receiver count falls as the facet count rises, so every point
does the same total work (about 35 million pairs) and differs only in how it is cut into passes:

* ``gather`` -- :func:`~aquaflux.radiation.direct_fluence_rate` with no bodies: the traced scan and
  nothing else;
* ``streamed`` -- the same with the vessel as an :class:`~aquaflux.solids.Outside` of three
  cylinders, so each pass also builds and drops a shadow mask. This is the path a reactor read from
  CAD takes.

**Each point runs in its own process**, because the quantity that decides a default is the peak
memory *footprint* (``/usr/bin/time -l``'s "peak memory footprint", which counts the compressed
pages that the resident set size leaves out), and a footprint can only be read per process. The
cost of that is that each wall-clock figure is from its own process: the sweep is therefore run
twice in alternating order and the faster of the two kept, with the spread between them reported
beside it, so a difference smaller than that spread is read as none. ``peak_footprint.py`` runs each
child, and forwards a ``kill`` of this process to it rather than leaving it orphaned.

Run with ``validation/run_case.sh validation/radiation_gather_pair_limit.py``. ``--point`` is the
child-process entry and is not for direct use.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "sozzi_radiation"))

#: The limits swept, in receiver-by-facet pairs.
LIMITS = (1_000_000, 4_000_000, 16_000_000, 64_000_000)
#: (sectors, slices) of the analytic lamp: 8,704, 67,584 and 270,336 facets.
LAMPS = ((32, 128), (64, 512), (128, 1024))
PATHS = ("gather", "streamed")
#: Receiver-by-facet pairs each point computes in all; the receiver count follows.
TOTAL_PAIRS = 35_000_000
REPEATS = 2


def point(path: str, sectors: int, slices: int, limit: int) -> dict:
    """One measurement, in this process: the second of two calls, after compiling."""
    import aquaflux  # noqa: F401  (enables x64)
    import jax.numpy as jnp
    import numpy as np
    from aquaflux.radiation import (
        NoOcclusion,
        Surfaces,
        UniformAbsorption,
        direct_fluence_rate,
    )
    from aquaflux.solids import Cylinder, Outside
    from lamp_resolution import lamp

    vertices = lamp(sectors, slices)
    count = TOTAL_PAIRS // len(vertices)
    rng = np.random.default_rng(0)
    radius = rng.uniform(0.011, 0.044, count)
    angle = rng.uniform(0.0, 2.0 * np.pi, count)
    receivers = jnp.asarray(
        np.column_stack(
            [rng.uniform(0.01, 0.88, count), radius * np.cos(angle), radius * np.sin(angle)]
        )
    )
    surfaces = Surfaces.from_triangles(vertices, emission=696.42)
    options = {"absorption": UniformAbsorption(35.67), "pair_limit": limit}
    if path == "streamed":
        chamber = Cylinder([0.4445, 0, 0], [1, 0, 0], 0.0445, 0.4445)
        options |= {"occluders": [Outside(chamber)], "self_occlusion": NoOcclusion()}

    direct_fluence_rate(surfaces, receivers, **options).block_until_ready()
    started = time.perf_counter()
    values = direct_fluence_rate(surfaces, receivers, **options).block_until_ready()
    seconds = time.perf_counter() - started
    return {
        "seconds": seconds,
        "receivers": count,
        "pairs_per_second": count * len(vertices) / seconds,
        "checksum": float(jnp.sum(values)),
    }


def measure(path: str, sectors: int, slices: int, limit: int) -> dict:
    """One point in a fresh process, with its peak memory footprint."""
    from peak_footprint import run_with_footprint

    command = [sys.executable, "-u", __file__, "--point", path]
    command += [str(sectors), str(slices), str(limit)]
    run = run_with_footprint(command, capture_output=True)
    if run.returncode:
        return {"failed": run.returncode, "stderr": run.stderr[-500:]}
    result = json.loads(run.stdout.strip().splitlines()[-1])
    result["peak_footprint_GB"] = run.peak_footprint_gb
    return result


def main() -> None:
    results: dict = {}
    order = [(p, s, t, lim) for p in PATHS for (s, t) in LAMPS for lim in LIMITS]
    for repeat in range(REPEATS):
        for path, sectors, slices, limit in order if repeat % 2 == 0 else order[::-1]:
            key = f"{path} {sectors}x{slices} limit {limit:,}"
            got = measure(path, sectors, slices, limit)
            results.setdefault(key, []).append(got)
            shown = {k: round(v, 3) if isinstance(v, float) else v for k, v in got.items()}
            print(f"[{time.strftime('%H:%M:%S')}] pass {repeat + 1} {key}: {shown}", flush=True)

    summary = {}
    for key, runs in results.items():
        good = [r for r in runs if "failed" not in r]
        if not good:
            summary[key] = {"failed": runs}
            continue
        seconds = [r["seconds"] for r in good]
        summary[key] = {
            "seconds": round(min(seconds), 3),
            "repeat_spread": round(max(seconds) / min(seconds), 3),
            "pairs_per_second_M": round(max(r["pairs_per_second"] for r in good) / 1e6, 1),
            "peak_footprint_GB": round(max(r["peak_footprint_GB"] or 0 for r in good), 2),
            "checksums_agree": len({round(r["checksum"], 6) for r in good}) == 1,
        }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--point":
        path, sectors, slices, limit = sys.argv[2], *map(int, sys.argv[3:6])
        print(json.dumps(point(path, sectors, slices, limit)), flush=True)
    else:
        main()
