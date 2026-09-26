"""The whole Sozzi reactor field through the public radiation model, with its shadows streamed.

``compare_fluence.py`` computes this reactor's fluence rate on all 1,635,909 cells by hand: it cuts
the cells into chunks itself, builds each chunk's shadow mask and throws it away, because a model
holding the receiver mask whole would need 12 GB per body. This runs the same field the way a user
would — one model, one call —

    model = build_radiation_model(cells, lamp, occluders=[water],
                                  settings=RadiationSettings(stream_receiver_mask=True, ...))
    G, _ = fluence_rate(model, lamp, absorption=UniformAbsorption(35.67))

with the water read from the reactor's own CAD drawing, and compares it with that hand-built field.
Without the CAD kernel the water is ``primitive_occlusion.py``'s three hand-typed cylinders instead,
which mask the mesh's cells identically (0 of 180 million pairs differ); the summary says which.

**How the bodies' layer is decided** is chosen with ``SOZZI_CULLING``, so every arm of a cost
comparison runs through this one harness: unset, the library's default; ``every-pair``, every body
against every pair (:class:`~aquaflux.radiation.EveryPair`); or a ladder of group sizes such as
``32,8``, a :class:`~aquaflux.radiation.ShaftCulling` with that ladder for both receivers and
facets. The masks are identical whichever is chosen, so the fields agree to the last bit and only
the time differs. Each arm writes its own summary, ``model_at_mesh_scale[-<arm>].json``.

**What must agree, and why exactly.** The lamp (the case's ``lampWall.stl``), the exitance and the
medium are the same; the walls are black, so the reflected pass the model adds carries only the
interreflection solve's own residual. The shadows differ in *description* — the hand-built field
tests the pipe openings with a closure derived for this reactor, the model with the drawing's three
cylinders — and on a 24,000-cell sample the two masked no pair differently. So the fields should
agree to the solve's tolerance everywhere, and a larger difference anywhere is a finding.

The run re-executes itself as a child process so its peak memory *footprint* can be read
(``/usr/bin/time -l``), which counts the compressed pages a resident-set figure leaves out — the
number that says whether the model really stayed within a chunk. ``validation/peak_footprint.py``
runs the child, and forwards a ``kill`` of this process to it rather than leaving it orphaned.

Run with ``validation/run_case.sh validation/sozzi_radiation/model_at_mesh_scale.py`` in an
environment with the CAD kernel (``pip install "aquaflux[cad]"``). Needs ``work/case`` (for the
lamp), ``work/cell_centres.npy``, and ``work/compare/G_aquaflux.npy`` from ``compare_fluence.py``.
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
sys.path.insert(0, str(HERE.parent))

WORK = HERE / "work"
CULLING = os.environ.get("SOZZI_CULLING", "")
RESULT = WORK / "compare" / f"model_at_mesh_scale{'-' + CULLING if CULLING else ''}.json"


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def culling_settings() -> dict:
    """The ``RadiationSettings`` entry ``SOZZI_CULLING`` names: none, to keep the library default."""
    from aquaflux.radiation import EveryPair, ShaftCulling

    if not CULLING:
        return {}
    if CULLING == "every-pair":
        return {"body_culling": EveryPair()}
    ladder = tuple(int(size) for size in CULLING.split(","))
    return {"body_culling": ShaftCulling(receiver_blocks=ladder, source_clusters=ladder)}


def run() -> dict:
    """Build the model and compute the field, in this process."""
    import aquaflux  # noqa: F401  (enables x64)
    import numpy as np
    from aquaflux.radiation import (
        NoOcclusion,
        RadiationSettings,
        UniformAbsorption,
        build_radiation_model,
        fluence_rate,
    )
    from compare_fluence import ABSORPTION, lamp_surfaces
    from primitive_occlusion import fluid, from_drawing

    cells = np.load(WORK / "cell_centres.npy")
    reference = np.load(WORK / "compare" / "G_aquaflux.npy")
    lamp = lamp_surfaces()
    water = from_drawing()
    source = "the CAD drawing"
    if water is None:
        water, source = fluid(), "three hand-typed cylinders (no CAD kernel)"
    culling = culling_settings()
    described = repr(culling.get("body_culling", "library default"))
    _say(f"{len(cells)} cells, {lamp.n_facets} lamp facets, water from {source}, {described}")

    started = time.perf_counter()
    model = build_radiation_model(
        cells,
        lamp,
        occluders=[water],
        # A convex lamp cannot shadow itself, so its own triangles are not ray-tested; that is the
        # same choice the hand-built field makes.
        settings=RadiationSettings(
            self_occlusion=NoOcclusion(), stream_receiver_mask=True, **culling
        ),
    )
    built = time.perf_counter() - started
    _say(f"model built in {built:.1f} s")

    started = time.perf_counter()
    field, cycles = fluence_rate(model, lamp, absorption=UniformAbsorption(ABSORPTION))
    field = np.asarray(field)
    called = time.perf_counter() - started
    _say(f"field computed in {called:.0f} s ({int(cycles)} solver cycles)")

    lit = reference > 1e-3 * reference.max()
    relative = np.abs(field - reference)[lit] / reference[lit]
    radius = np.hypot(cells[:, 1], cells[:, 2])
    far = (cells[:, 0] > 1.10) | (cells[:, 2] > 0.40)
    return {
        "cells": len(cells),
        "water": source,
        "body_culling": described,
        "lamp_facets": int(lamp.n_facets),
        "build_seconds": round(built, 1),
        "field_seconds": round(called, 1),
        "solver_cycles": int(cycles),
        "finite": bool(np.isfinite(field).all()),
        "relative_difference_over_lit_cells": {
            "cells": int(lit.sum()),
            "median": float(np.median(relative)),
            "p99": float(np.percentile(relative, 99)),
            "max": float(relative.max()),
        },
        "max_absolute_difference_W_m2": float(np.abs(field - reference).max()),
        "far_pipe_cells": int(far.sum()),
        "far_pipe_max_absolute_difference_W_m2": float(np.abs(field - reference)[far].max()),
        "chamber_mean_G": float(field[(radius <= 0.0445) & (cells[:, 0] <= 0.889)].mean()),
    }


def main() -> None:
    """Run the child, its progress streaming, and read its footprint."""
    from peak_footprint import run_with_footprint

    result = RESULT
    child = run_with_footprint([sys.executable, "-u", __file__, "--child"])
    if child.returncode:
        sys.exit(child.returncode)
    summary = json.loads(result.read_text())
    gb = child.peak_footprint_gb
    summary["peak_footprint_GB"] = None if gb is None else round(gb, 2)
    result.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        RESULT.write_text(json.dumps(run()))
    else:
        main()
