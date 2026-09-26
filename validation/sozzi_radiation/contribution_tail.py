"""How few lamp facets carry a receiver's fluence rate, in water that absorbs strongly?

The third measurement of issue #565. ``cluster_approximation.py`` found that approximating a
cluster of lamp facets by one term buys little in the Sozzi water (``a`` = 35.67 /m): a facet is
already nearly as large as the absorption allows, so clusters stay small. The same absorption
makes distant facets contribute almost nothing -- ``exp(-35.67 x 0.3 m)`` is 2e-5 -- which points at
a different lever that approximates nothing: **skip the pairs whose contribution is below a budget.**

This measures the ceiling of that lever. For each receiver the exact per-pair terms (as in
``cluster_visibility.py``: ``M rad Omega exp(-a d)`` times the shadow bit) are sorted, largest
first, and the report is the share of the receiver's lit pairs needed for the kept sum to reach
``1 - tol`` of ``G``, for each ``tol``. **It is an oracle** -- it knows every term's value -- so it
is the most any cutoff could achieve at that accuracy; a practical one decides from upper bounds
over tiles of receivers and clusters of facets, and keeps more.

**Scene** and receivers as ``cluster_visibility.py``: the analytic Sozzi reactor, the lamp,
``UniformAbsorption(ABSORPTION)``, receivers sampled per region outside the lamp,
``SOZZI_CLUSTER_RECEIVERS`` per region (default ``3000,1500,1500``).

Run with ``validation/run_case.sh validation/sozzi_radiation/contribution_tail.py``.
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
from aquaflux.radiation import NoOcclusion, build_visibility  # noqa: E402
from cluster_visibility import COUNTS, REGIONS, _pair_terms, sample  # noqa: E402
from compare_fluence import ABSORPTION  # noqa: E402
from primitive_occlusion import OUT, fluid, lamp  # noqa: E402

TOLERANCES = (1e-2, 1e-3, 1e-4, 1e-6)
CHUNK = 500


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    rng = np.random.default_rng(0)
    water = fluid()
    surfaces, lamp_source = lamp()
    points, region = sample(rng, water, np.asarray(surfaces.vertices))
    _say(
        f"{len(points)} receivers ({', '.join(f'{c} {n}' for c, n in zip(COUNTS, REGIONS, strict=True))}), "
        f"{surfaces.n_facets} lamp facets ({lamp_source}), absorption {ABSORPTION} /m; "
        f"jax {jax.__version__}, {platform.system()} {platform.machine()}, {os.cpu_count()} cores"
    )
    needed = {tol: np.zeros(len(points)) for tol in TOLERANCES}
    lit_share = np.zeros(len(points))
    for start in range(0, len(points), CHUNK):
        chunk = points[start : start + CHUNK]
        mask = build_visibility([water], surfaces, chunk, self_occlusion=NoOcclusion())
        visible = 1.0 - np.asarray(mask.blocked[0], dtype=float)
        w, _, gate = (
            np.asarray(term)
            for term in _pair_terms(
                jnp.asarray(chunk),
                surfaces.vertices,
                surfaces.centroid,
                surfaces.normal,
                surfaces.emission,
                ABSORPTION,
            )
        )
        terms = np.sort(w * visible, axis=1)[:, ::-1]
        kept = np.cumsum(terms, axis=1) / terms.sum(axis=1, keepdims=True)
        lit = np.maximum(gate.sum(axis=1), 1)
        lit_share[start : start + CHUNK] = gate.sum(axis=1) / surfaces.n_facets
        for tol in TOLERANCES:
            needed[tol][start : start + CHUNK] = ((kept < 1.0 - tol).sum(axis=1) + 1) / lit
        _say(f"{start + len(chunk)} / {len(points)} receivers")
    summary = {}
    for r, name in enumerate(REGIONS):
        rows = region == r
        summary[name] = {
            "lit_share_of_all_pairs": float(np.median(lit_share[rows])),
            **{
                f"tol {tol:g}": [float(np.median(needed[tol][rows])),
                                 float(np.quantile(needed[tol][rows], 0.9))]
                for tol in TOLERANCES
            },
        }  # fmt: skip
        _say(
            f"{name}: lit pairs are {summary[name]['lit_share_of_all_pairs']:.3f} of all; share of lit "
            "pairs needed, median / p90: "
            + ", ".join(
                f"tol {tol:g} {summary[name][f'tol {tol:g}'][0]:.4f} / "
                f"{summary[name][f'tol {tol:g}'][1]:.4f}"
                for tol in TOLERANCES
            )
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contribution_tail.json").write_text(json.dumps(summary, indent=2))
    _say(f"wrote {OUT / 'contribution_tail.json'}")


if __name__ == "__main__":
    main()
