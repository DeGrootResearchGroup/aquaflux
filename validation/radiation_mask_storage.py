"""What a receiver shadow mask stores, and what building it peaks at.

The scene is ``radiation_receiver_ray_mask.py``'s annular reactor in triangles -- the Sozzi lamp,
a dark sleeve, the dark vessel wall, 4,992 facets -- with its receivers in the annulus, and one
analytic body in the way as well: a ``Cylinder`` just inside the sleeve -- inside, so none of the
sleeve's own facet centroids lies within it -- so the mask has a body layer beside the surface's
own. Each arm is built in a process of its own, so that
``/usr/bin/time -l`` can read that build's peak memory footprint, and reports:

* the bytes the built ``Visibility`` holds per receiver-facet pair, counted from its arrays;
* the build's peak footprint and wall clock;
* a checksum of what it records -- pairs blocked by the body, and the surface's hidden share
  summed -- read from its arrays directly, so a before-and-after pair can be seen to describe
  the same shadows without forming a floating-point array of the whole mask to check it.

Two arms: ``NoOcclusion`` on ``HELD_RECEIVERS`` receivers -- the Sozzi field's own configuration,
where the surface shadows nothing and the body is the only layer -- and the brute-force
``RayCastOcclusion`` on ``RAY_RECEIVERS``, whose passes form the rays.

Run with ``validation/run_case.sh validation/radiation_mask_storage.py``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

#: Receivers for the arm without self-occlusion, and for the ray-cast arm.
HELD_RECEIVERS = 40_000
RAY_RECEIVERS = 2_000


def build(arm: str) -> dict:
    """Build one arm's mask in this process and say what it holds."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from aquaflux.radiation import NoOcclusion, RayCastOcclusion, build_visibility
    from aquaflux.solids import Cylinder
    from radiation_receiver_ray_mask import SLEEVE, SLEEVE_AT, receivers, scene

    surfaces = scene()
    body = Cylinder(
        centre=SLEEVE_AT, axis=[0.0, 0.0, 1.0], radius=0.9 * SLEEVE[0], half_length=0.9 * SLEEVE[1]
    )
    strategy, count = {
        "none": (NoOcclusion(), HELD_RECEIVERS),
        "ray": (RayCastOcclusion(), RAY_RECEIVERS),
    }[arm]
    points = receivers(np.random.default_rng(0), count)
    started = time.perf_counter()
    mask = jax.block_until_ready(
        build_visibility([body], surfaces, points, self_occlusion=strategy)
    )
    seconds = time.perf_counter() - started
    hidden = 0 if mask.hidden_by_geometry is None else int(jnp.sum(mask.hidden_by_geometry))
    stored = sum(
        leaf.nbytes
        for leaf in jax.tree.leaves(mask)
        if hasattr(leaf, "nbytes") and leaf is not mask.receivers
    )
    return {
        "arm": arm,
        "receivers": count,
        "facets": int(surfaces.n_facets),
        "bytes_per_pair": round(stored / (count * surfaces.n_facets), 3),
        "seconds": round(seconds, 1),
        "checksum": f"{int(jnp.sum(mask.blocked))} blocked, {hidden} hidden",
    }


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        print(json.dumps(build(sys.argv[2])), flush=True)
        return
    from peak_footprint import run_with_footprint

    for arm in ("none", "ray"):
        measured = run_with_footprint(
            [sys.executable, "-u", __file__, "--child", arm], capture_output=True
        )
        if measured.returncode != 0:
            print(f"{arm}: child failed with {measured.returncode}\n{measured.stderr}", flush=True)
            continue
        report = json.loads(measured.stdout.strip().splitlines()[-1])
        print(
            f"{report['arm']}: {report['receivers']} receivers x {report['facets']} facets, "
            f"{report['bytes_per_pair']} B/pair stored, peak footprint "
            f"{measured.peak_footprint_gb:.2f} GB, {report['seconds']} s, "
            f"checksum {report['checksum']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
