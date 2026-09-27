"""Where a streamed field's time goes: each chunk's mask, layout and gather, each timed to completion.

JAX dispatches asynchronously, so a profile of a streamed call charges each chunk's gather to
whatever next waits on a device array -- the following chunk's mask build, which converts its
inputs to numpy. That reading once put the mask at 10.6 s of an 11.5 s call; timed to completion it
is under half the gather. Here every piece is blocked until ready before its clock stops.

Two arms: the default streamed chunk (one mask and one layout per ~460 receivers here), and the
whole problem as **one** streamed chunk with the gather's traced chunk left at the default -- done
by making the streamed path's own chunk count, and nothing else, cover every point. (Raising
``pair_limit`` instead would enlarge the traced chunk too, which is slower and measures something
else.) The second arm is a diagnostic, not a setting: one mask for a mesh would not fit.
Scene as ``backface_share.py``.

Run with ``validation/run_case.sh validation/sozzi_radiation/streamed_cost_split.py``.
"""

from __future__ import annotations

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
import numpy as np  # noqa: E402
from aquaflux.radiation import NoOcclusion, UniformAbsorption, gather  # noqa: E402
from backface_share import outside_lamp  # noqa: E402
from primitive_occlusion import fluid, lamp, receivers  # noqa: E402

PASSES = 2


def _timed(pieces: dict, name: str, function):
    """``function``, its time to completion added to ``pieces[name]``."""

    def timed(*args, **kwargs):
        started = time.perf_counter()
        result = jax.block_until_ready(function(*args, **kwargs))
        pieces[name] += time.perf_counter() - started
        return result

    return timed


def main() -> None:
    rng = np.random.default_rng(0)
    water = fluid()
    surfaces, lamp_source = lamp()
    sampled, population = receivers(rng, water)
    points = outside_lamp(sampled, np.asarray(surfaces.vertices))
    medium = UniformAbsorption(35.67)
    print(
        f"{len(points)} receivers ({population}), {surfaces.n_facets} facets ({lamp_source}); "
        f"jax {jax.__version__}, {platform.system()} {platform.machine()}, {os.cpu_count()} cores",
        flush=True,
    )
    pieces = dict.fromkeys(("mask", "layout", "point sources", "areal segments"), 0.0)
    gather._Shadows.mask = _timed(pieces, "mask", gather._Shadows.mask)
    gather.areal_layout = _timed(pieces, "layout", gather.areal_layout)
    gather._CompiledParts.points = _timed(pieces, "point sources", gather._CompiledParts.points)
    gather._CompiledParts.segment = _timed(pieces, "areal segments", gather._CompiledParts.segment)
    per_chunk = gather.receivers_per_pass
    for label, whole in (("default chunk", False), ("one streamed chunk", True)):
        # Read in the streamed path alone; the traced passes use their own binding.
        gather.receivers_per_pass = (lambda *_: len(points)) if whole else per_chunk
        for index in range(PASSES + 1):
            for key in pieces:
                pieces[key] = 0.0
            started = time.perf_counter()
            np.asarray(
                gather.streamed_fluence_rate(
                    (surfaces,), points, shadow_geometry=surfaces, occluders=[water],
                    self_occlusion=NoOcclusion(), absorption=medium,
                )
            )  # fmt: skip
            total = time.perf_counter() - started
            split = ", ".join(f"{key} {value:.2f} s" for key, value in pieces.items())
            print(
                f"{label}, {'warm-up' if index == 0 else f'pass {index}'}: {total:.2f} s ({split})",
                flush=True,
            )


if __name__ == "__main__":
    main()
