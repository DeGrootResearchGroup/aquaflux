"""What does a model call pay for work that could have been compiled, and is it paid once or every time?

Two per-call costs, each measured the only way that separates it from the arithmetic -- against the
same call wrapped in ``jax.jit``, or against a second call of the same size:

1. **The gather's chunk loop.** ``work.in_passes`` runs a gather's full chunks as one scan and its
   short last chunk after it. A scan is compiled as a whole even when nothing around it is; a bare
   call of the body runs it one operation at a time. The table times ``direct_fluence_rate``
   called plainly against the same call inside ``jax.jit``, at receiver counts that give no
   remainder, a half-chunk remainder, a one-receiver remainder and a remainder alone. What is left
   of the ratio once the remainder is compiled is the loop's per-call trace and compile.
2. **The radiosity solve.** A linear solve handed a new operator closure is traced again on every
   call. The table times four consecutive ``radiosity`` calls on one model: the first pays the
   compile (and, since ``build_radiation_model`` returns before its asynchronous work finishes,
   whatever of the build is still running -- which is why the model is waited on first), and the
   rest should cost the solve alone.

Answers are compared across every pair of arms; how the work is compiled must not change them
beyond rounding.

Run with ``validation/run_case.sh validation/radiation_per_call_compile.py``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aquaflux  # noqa: F401  (enables x64)
import jax
from aquaflux.radiation.absorption import UniformAbsorption
from aquaflux.radiation.gather import direct_fluence_rate
from aquaflux.radiation.model import RadiationSettings, build_radiation_model, radiosity
from aquaflux.radiation.self_occlusion import NoOcclusion
from aquaflux.radiation.surfaces import Surfaces
from tests.unit.radiation_references import box, cylinder_triangles

PAIR_LIMIT = 4_000_000
PASSES = 2


def seconds(call) -> float:
    """Wall clock of one call, waited on."""
    start = time.perf_counter()
    jax.block_until_ready(call())
    return time.perf_counter() - start


def gather_loop() -> None:
    """``direct_fluence_rate`` plainly and inside ``jax.jit``, at four receiver counts."""
    lamp = Surfaces.from_triangles(
        cylinder_triangles(0.01, 0.2, sectors=32, slices=64), emission=1.0
    )
    per_pass = PAIR_LIMIT // lamp.n_facets
    absorption = UniformAbsorption(35.0)
    rng = np.random.default_rng(0)
    print(
        f"### gather: {lamp.n_facets} lamp facets, {per_pass} receivers a pass, "
        f"UniformAbsorption(35), no mask; fastest of {PASSES} warm calls\n",
        flush=True,
    )
    print(
        f"{'receivers':>9} {'passes':>18} {'plain s':>8} {'jit s':>8} {'ratio':>6} {'rel diff':>9}"
    )
    for n in (3 * per_pass, 3 * per_pass + per_pass // 2, 3 * per_pass + 1, per_pass // 2):
        radius = np.sqrt(rng.uniform(0.02**2, 0.08**2, n))
        angle = rng.uniform(0.0, 2.0 * np.pi, n)
        points = np.column_stack(
            [radius * np.cos(angle), radius * np.sin(angle), rng.uniform(-0.2, 0.2, n)]
        )

        def plain(points=points):
            return direct_fluence_rate(lamp, points, absorption=absorption, pair_limit=PAIR_LIMIT)

        compiled_gather = jax.jit(
            lambda p: direct_fluence_rate(lamp, p, absorption=absorption, pair_limit=PAIR_LIMIT)
        )

        def compiled(points=points, compiled_gather=compiled_gather):
            return compiled_gather(points)

        a, b = plain(), compiled()  # warm-up, and the answers
        best_plain = min(seconds(plain) for _ in range(PASSES))
        best_jit = min(seconds(compiled) for _ in range(PASSES))
        moved = float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
        passes = f"{n // per_pass} full + {n % per_pass}"
        print(
            f"{n:9d} {passes:>18} {best_plain:8.3f} {best_jit:8.3f} "
            f"{best_plain / best_jit:5.2f}x {moved:9.1e}",
            flush=True,
        )


def solve() -> None:
    """Four consecutive ``radiosity`` calls on one model of each size."""
    print("\n### radiosity: closed box, reflectance 0.5, UniformAbsorption(2), NoOcclusion\n")
    print(
        f"{'facets':>7} {'call 1 s':>9} {'call 2 s':>9} {'call 3 s':>9} {'call 4 s':>9} {'cycles':>7}"
    )
    for divisions in (8, 16):
        surfaces = box(divisions, emission=1.0, reflectance=0.5)
        model = build_radiation_model(
            np.array([[0.5, 0.5, 0.5]]),
            surfaces,
            settings=RadiationSettings(self_occlusion=NoOcclusion()),
        )
        jax.block_until_ready(jax.tree.leaves(model))  # the build finishes before anything is timed
        absorption = UniformAbsorption(2.0)
        answers, times = [], []
        for scale in (1.0, 1.0, 2.0, 2.0):
            lit = surfaces.with_optics(emission=scale * np.asarray(surfaces.emission))
            start = time.perf_counter()
            outgoing, cycles = radiosity(model, lit, absorption=absorption)
            jax.block_until_ready(outgoing)
            times.append(time.perf_counter() - start)
            answers.append(np.asarray(outgoing) / scale)
        spread = max(float(np.max(np.abs(x - answers[0]))) for x in answers)
        print(
            f"{surfaces.n_facets:7d} " + " ".join(f"{t:9.3f}" for t in times) + f" {int(cycles):7d}"
            f"   (answers per unit emission agree to {spread:.1e})",
            flush=True,
        )


if __name__ == "__main__":
    gather_loop()
    solve()
