"""What coarsening the Sozzi reactor's own patches costs, at the lamp's size and the vessel's.

The snapped ``lampWall`` patch is 194,636 centre-fan triangles and the ``bodyWall`` patch about
1.3 million. The lamp is coarsened to be the emitter (#492); the vessel wall is what a reflecting
wall's optical surface would be coarsened from (#491), which makes it the size the coarsener has to
be fast at. Each patch is coarsened at two bounds and the time, facet count and realized bounds are
reported; the first call of the process also pays for compiling the distance kernel, so a small
warm-up runs first.

Run with ``validation/run_case.sh validation/sozzi_radiation/coarsen_speed.py``. The patches are
cached beside the case (``work/lamp_patch.npy``, ``work/body_patch.npy``); reading the 1.6M-cell mesh
for them takes several minutes the first time.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import aquaflux  # noqa: E402,F401  (enables x64)
from aquaflux.io import read_openfoam  # noqa: E402
from aquaflux.mesh import patch_triangles  # noqa: E402
from aquaflux.radiation import coarsen_to_size  # noqa: E402

WORK = HERE / "work"
CASE = WORK / "case"
OUT = WORK / "compare"

PATCHES = {"lampWall": WORK / "lamp_patch.npy", "bodyWall": WORK / "body_patch.npy"}
#: (longest edge, chord) in metres. 4 mm / 1e-4 m is the lamp's chosen resolution; the wider edge is
#: nearer what a reflecting wall's optical surface would want.
BOUNDS = ((4e-3, 1e-4), (1e-2, 1e-4))


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def patches() -> dict[str, np.ndarray]:
    """Each patch's centre-fan triangles, read from the mesh once and cached."""
    if all(path.exists() for path in PATCHES.values()):
        return {name: np.load(path) for name, path in PATCHES.items()}
    _say("reading the mesh for its patches")
    mesh = read_openfoam(CASE)
    geometry = mesh.geometry()
    found = {}
    for name, path in PATCHES.items():
        found[name] = patch_triangles(mesh, geometry, [name]).vertices
        np.save(path, found[name])
    return found


def main() -> None:
    triangles = patches()
    for name, vertices in triangles.items():
        _say(f"{name}: {len(vertices):,} triangles")
    started = time.perf_counter()
    coarsen_to_size(triangles["lampWall"][:2000], max_edge=4e-3, chord=1e-4)
    _say(f"warm-up (compiles the distance kernel): {time.perf_counter() - started:.1f} s")

    summary = {}
    for name, vertices in triangles.items():
        for max_edge, chord in BOUNDS:
            started = time.perf_counter()
            result = coarsen_to_size(vertices, max_edge=max_edge, chord=chord)
            seconds = time.perf_counter() - started
            key = f"{name}, edge {max_edge:g} m, chord {chord:g} m"
            summary[key] = {
                "input_triangles": len(vertices),
                "facets": result.n_facets,
                "seconds": round(seconds, 1),
                "ms_per_input_triangle": round(1e3 * seconds / len(vertices), 4),
                "longest_edge_max_m": float(result.longest_edge.max()),
                "longest_edge_median_m": float(np.median(result.longest_edge)),
                "chord_max_m": float(result.chord.max()),
                "angle_max_rad": float(result.angle.max()),
                "area_after_over_before": float(result.area_after[0] / result.area_before[0]),
            }
            _say(f"{key}: {json.dumps(summary[key])}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coarsen_speed.json").write_text(json.dumps(summary, indent=2) + "\n")
    _say(f"done: {OUT / 'coarsen_speed.json'}")


if __name__ == "__main__":
    main()
