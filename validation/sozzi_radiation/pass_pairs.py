"""How many pairs one step of the gather's traced loop should form: ``work.PASS_PAIRS``, swept.

``pair_limit`` bounds what one pass *may* form, for memory. ``work.PASS_PAIRS`` bounds what one
step of the traced loop (``work.in_passes``) *does* form, for speed: the compiled body writes each
per-pair intermediate out and reads it back, which is cheap while a step's intermediates fit in a
core's cache and about twice the cost past it. Where that edge falls is a property of the machine,
so this measures it rather than assuming it.

**Each value runs in its own process.** The segment programs are compiled once per scene and
cached across calls under keys that do not include the bound, so in one process every value after
the first would silently reuse the first one's program and measure nothing. The sweep runs twice
in alternating order and keeps the faster of each, with the spread beside it.

**Paths**, each warm, fastest of its repeats: the whole streamed field
(``direct_fluence_rate(..., occluders=...)``, masks and gather together), its gradient with
respect to emission over 4,000 of the receivers, and the held gather against a mask built once
(12,000 receivers), eagerly and under ``jit``. Checksums must agree across values: how the work is cut changes nothing
about the answer.

**Scene** as ``backface_share.py``: the Sozzi water as ``Outside(chamber, inlet, riser)``, the
case's lamp STL when the case is present and otherwise the analytic 32 x 128 lamp,
``UniformAbsorption(35.67)``, ``NoOcclusion`` for the lamp's own triangles.

Run with ``validation/run_case.sh validation/sozzi_radiation/pass_pairs.py``. ``--point`` is the
child-process entry and is not for direct use.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

#: The bounds swept, in receiver-by-facet pairs. The largest is the default pair limit, i.e. the
#: traced loop as it was before it had a bound of its own.
BOUNDS = (1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 20, 4_000_000)
SWEEPS = 2


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def point(bound: int) -> dict:
    """Every path at one bound, in this process."""
    import aquaflux  # noqa: F401  (enables x64)
    from aquaflux.radiation import work

    work.PASS_PAIRS = bound
    import jax
    import jax.numpy as jnp
    import numpy as np
    from aquaflux.radiation import (
        NoOcclusion,
        UniformAbsorption,
        build_visibility,
        direct_fluence_rate,
    )
    from backface_share import outside_lamp
    from primitive_occlusion import fluid, lamp, receivers

    rng = np.random.default_rng(0)
    water = fluid()
    surfaces, _ = lamp()
    sampled, _ = receivers(rng, water)
    points = outside_lamp(sampled, np.asarray(surfaces.vertices))
    medium = UniformAbsorption(35.67)
    held = points[:12000]
    mask = build_visibility([water], surfaces, held, self_occlusion=NoOcclusion())

    def streamed():
        return direct_fluence_rate(
            surfaces, points, absorption=medium, occluders=[water], self_occlusion=NoOcclusion()
        )

    def gradient():
        def total(emission):
            return jnp.sum(
                direct_fluence_rate(
                    surfaces.with_optics(emission=emission), points[:4000], absorption=medium,
                    occluders=[water], self_occlusion=NoOcclusion(),
                )
            )  # fmt: skip

        return jax.grad(total)(surfaces.emission)

    def held_gather():
        return direct_fluence_rate(surfaces, held, absorption=medium, visibility=mask)

    held_under_jit = jax.jit(held_gather)

    result = {}
    for name, function, repeats in (
        ("streamed field", streamed, 2),
        ("streamed gradient", gradient, 2),
        ("held gather", held_gather, 3),
        ("held gather under jit", held_under_jit, 3),
    ):
        jax.block_until_ready(function())
        fastest = float("inf")
        for _ in range(repeats):
            started = time.perf_counter()
            value = jax.block_until_ready(function())
            fastest = min(fastest, time.perf_counter() - started)
        result[name] = {"seconds": fastest, "checksum": float(jnp.sum(value))}
    return result


def main() -> None:
    import jax

    _say(
        f"PASS_PAIRS sweep {BOUNDS}; jax {jax.__version__}, {platform.system()} "
        f"{platform.machine()}, {os.cpu_count()} cores"
    )
    runs = {bound: [] for bound in BOUNDS}
    for sweep in range(SWEEPS):
        for bound in BOUNDS if sweep % 2 == 0 else BOUNDS[::-1]:
            child = subprocess.run(
                [sys.executable, __file__, "--point", str(bound)],
                capture_output=True, text=True, check=True,
            )  # fmt: skip
            measured = json.loads(child.stdout.strip().splitlines()[-1])
            runs[bound].append(measured)
            _say(
                f"sweep {sweep + 1}, bound {bound}: "
                + ", ".join(f"{name} {value['seconds']:.2f} s" for name, value in measured.items())
            )
    names = list(runs[BOUNDS[0]][0])
    for name in names:
        checksums = {run[name]["checksum"] for bound in BOUNDS for run in runs[bound]}
        cells = []
        for bound in BOUNDS:
            seconds = [run[name]["seconds"] for run in runs[bound]]
            cells.append(
                f"{bound}: {min(seconds):.2f} s (spread {max(seconds) / min(seconds):.2f}x)"
            )
        _say(f"{name}: " + "; ".join(cells))
        # Differently cut work adds its terms in another order, so a rounding is allowed.
        values = sorted(checksums)
        if (values[-1] - values[0]) > 1e-12 * abs(values[-1]):
            raise SystemExit(f"{name}: the answer moved with the bound: {values}")
    from primitive_occlusion import OUT

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pass_pairs.json").write_text(json.dumps({str(k): v for k, v in runs.items()}, indent=2))
    _say(f"wrote {OUT / 'pass_pairs.json'}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--point"]:
        print(json.dumps(point(int(sys.argv[2]))), flush=True)
    else:
        main()
