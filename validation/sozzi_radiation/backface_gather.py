"""What leaving facets off the gather's lists buys on the whole streamed field.

The gather lays receivers out in small blocks, each against the facets its bounding box is not
proven to lie behind (``aquaflux/radiation/lit_blocks.py``); for a lamp dark behind itself what is
left off is exactly zero. This times the whole streamed field -- the public
``direct_fluence_rate(..., occluders=...)``, masks and gather together -- two ways in **one
process on the same points**:

- *every facet listed*: the same layout with the planes withheld, so every block is gathered
  against every facet;
- *facets behind left out*: as shipped.

A warm-up pays compilation, then two alternating passes, fastest kept, with the spread beside
it. The fields must agree to rounding (the terms are the same, added in another order), and the
share of pairs the lists hold is reported from the layout itself.

**Scene** as ``backface_share.py``: the Sozzi water as ``Outside(chamber, inlet, riser)``, the
case's lamp STL when the case is present and otherwise the analytic 32 x 128 lamp,
``UniformAbsorption(35.67)``, ``NoOcclusion`` for the lamp's own triangles, receivers from
``primitive_occlusion.receivers`` less any inside the lamp. ``SOZZI_RECEIVERS`` sets the count.

Run with ``validation/run_case.sh validation/sozzi_radiation/backface_gather.py``.
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
import numpy as np  # noqa: E402
from aquaflux.radiation import (  # noqa: E402
    NoOcclusion,
    UniformAbsorption,
    direct_fluence_rate,
    gather,
)
from backface_share import outside_lamp  # noqa: E402
from primitive_occlusion import OUT, fluid, lamp, receivers  # noqa: E402

PASSES = 2


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _listing_every_facet(points, geometry, groups, *, segments=None):
    """The shipped layout with the planes withheld: every block against every facet."""
    return tuple(
        gather.lit_segments(points, group.facets, None, segments=segments) for group in groups
    )


def main() -> None:
    rng = np.random.default_rng(0)
    water = fluid()
    surfaces, lamp_source = lamp()
    sampled, population = receivers(rng, water)
    points = outside_lamp(sampled, np.asarray(surfaces.vertices))
    medium = UniformAbsorption(35.67)
    _say(
        f"{len(points)} receivers ({population}), {surfaces.n_facets} lamp facets "
        f"({lamp_source}); jax {jax.__version__}, {platform.system()} {platform.machine()}, "
        f"{os.cpu_count()} cores"
    )
    shipped = gather.areal_layout
    listed = []

    def counting(points, geometry, groups, *, segments=None):
        layout = shipped(points, geometry, groups, segments=segments)
        listed.append(
            sum(int(np.asarray(s.valid).sum()) * s.block for part in layout for s in part)
        )
        return layout

    arms = {"every facet listed": _listing_every_facet, "facets behind left out": counting}

    def field(arm):
        gather.areal_layout = arms[arm]
        try:
            started = time.perf_counter()
            value = direct_fluence_rate(
                surfaces, points, absorption=medium, occluders=[water],
                self_occlusion=NoOcclusion(),
            ).block_until_ready()  # fmt: skip
            return np.asarray(value), time.perf_counter() - started
        finally:
            gather.areal_layout = shipped

    fields, seconds = {}, {arm: [] for arm in arms}
    for index in range(PASSES + 1):
        for arm in arms:
            listed.clear()
            fields[arm], elapsed = field(arm)
            if index:
                seconds[arm].append(elapsed)
            _say(f"{'warm-up' if index == 0 else f'pass {index}'}, {arm}: {elapsed:.2f} s")
    share = sum(listed) / (len(points) * int((~surfaces.is_point_source).sum()))
    full, skipped = fields["every facet listed"], fields["facets behind left out"]
    lit = full > 0.0
    relative = np.abs(skipped - full)[lit] / full[lit]
    fastest = {arm: min(values) for arm, values in seconds.items()}
    spread = {arm: max(values) / min(values) for arm, values in seconds.items()}
    ratio = fastest["every facet listed"] / fastest["facets behind left out"]
    _say(
        f"every facet listed {fastest['every facet listed']:.2f} s (spread "
        f"{spread['every facet listed']:.2f}x), facets behind left out "
        f"{fastest['facets behind left out']:.2f} s (spread "
        f"{spread['facets behind left out']:.2f}x): {ratio:.2f}x; lists hold {share:.3f} of the "
        f"pairs; field relative difference median {np.median(relative):.1e}, max "
        f"{relative.max():.1e} over {int(lit.sum())} lit receivers"
    )
    summary = {
        "receivers": len(points),
        "facets": surfaces.n_facets,
        "fastest_s": fastest,
        "spread": spread,
        "speed_up": ratio,
        "listed_share": share,
        "max_relative_difference": float(relative.max()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "backface_gather.json").write_text(json.dumps(summary, indent=2))
    _say(f"wrote {OUT / 'backface_gather.json'}")
    if relative.max() > 1e-12:
        raise SystemExit("the field moved by more than a rounding")


if __name__ == "__main__":
    main()
