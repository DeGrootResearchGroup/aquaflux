"""aquaflux's fluence rate at every cell centre of the room.

The volume counterpart of ``aquaflux_floor.py``, with the same lamp and the same bunny: the
fluence rate ``G`` -- radiance integrated over the whole sphere, no receiver cosine -- at the cell
centres OpenFOAM wrote for the mesh (``work/<mesh>/cells.npz``), which are the points the
discrete-ordinates ``G`` belongs to. Receivers go through in chunks, each with its own shadow mask,
so the mask never holds more than one chunk's pairs.

The whole volume is gathered against the lamp patch's 1,120 fan triangles. On the cells of the
two slices in ``room.SLICES`` it is gathered again against the lamp split to ~0.8 mm, as on the
floor, and the difference is reported.

Writes ``work/aquaflux/<mesh>_volume.npz`` (``G`` per cell, and ``G_refined`` on each slice's
cells) and ``<mesh>_volume.json``.

Environment: ``RAY_MESHES`` (default ``"empty bunny"``).
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
import patches  # noqa: E402
import room  # noqa: E402
from aquaflux.radiation import (  # noqa: E402
    NoOcclusion,
    Surfaces,
    TriangleBody,
    build_visibility,
    direct_fluence_rate,
)
from aquaflux.radiation.subdivide import subdivide_to_width  # noqa: E402
from aquaflux_floor import REFINED_RATIO, lamp_surfaces  # noqa: E402

OUT = HERE / "work" / "aquaflux"
CHUNK = 40_000  # receivers per shadow mask: 40,000 x 1,120 pairs, ~45 MB of mask


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fluence(lamp: Surfaces, points: np.ndarray, body, label: str) -> tuple[np.ndarray, dict]:
    """``G`` at ``points``, chunk by chunk, and the time spent masking and gathering."""
    out = np.empty(len(points))
    timing = {"visibility_s": 0.0, "gather_s": 0.0}
    for start in range(0, len(points), CHUNK):
        chunk = points[start : start + CHUNK]
        visibility = None
        if body is not None:
            began = time.perf_counter()
            visibility = build_visibility([body], lamp, chunk, self_occlusion=NoOcclusion())
            timing["visibility_s"] += time.perf_counter() - began
        began = time.perf_counter()
        out[start : start + len(chunk)] = np.asarray(
            jax.block_until_ready(
                direct_fluence_rate(lamp, jnp.asarray(chunk), visibility=visibility)
            )
        )
        timing["gather_s"] += time.perf_counter() - began
        if (start // CHUNK) % 10 == 0 or start + CHUNK >= len(points):
            _say(f"  {label}: {min(start + CHUNK, len(points))}/{len(points)} cells")
    return out, {key: round(value, 1) for key, value in timing.items()}


BANDS = ((0.05, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 6.0))  # m from the lamp's centre


def refinement_check(coarse: np.ndarray, fine: np.ndarray, points: np.ndarray) -> dict:
    """The coarse lamp's relative error against the refined one, by distance from the lamp.

    Over lit cells only (refined ``G`` above 1e-3 of its 99th percentile): cells level with the
    ceiling beside the window, or deep in a shadow, receive nothing, and a relative error there
    divides by zero. Banded because the error is a near-field one -- a 3 mm facet's one-point
    radiance -- and falls off with distance, so one number would describe only the worst band.
    """
    lit = fine > 1e-3 * np.percentile(fine, 99)
    distance = np.linalg.norm(points - np.array([0.0, 0.0, room.ROOM_HIGH[2]]), axis=1)
    relative = np.abs(coarse - fine) / np.where(fine > 0.0, fine, 1.0)
    out = {}
    for low, high in BANDS:
        band = lit & (distance > low) & (distance <= high)
        if band.any():
            out[f"{low}-{high} m"] = {
                "cells": int(band.sum()),
                "max_rel": float(relative[band].max()),
                "p99_rel": float(np.percentile(relative[band], 99)),
            }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for mesh in os.environ.get("RAY_MESHES", "empty bunny").split():
        started = time.perf_counter()
        geometry = patches.load(mesh)
        cells = np.load(HERE / "work" / mesh / "cells.npz")
        centre, volume = cells["centre"], cells["volume"]
        lamp = lamp_surfaces(geometry["lamp"]["triangles"])
        body = (
            TriangleBody.build(geometry["bunny"]["triangles"], sheet=True)
            if mesh == "bunny"
            else None
        )
        _say(f"{mesh}: {len(centre)} cells, {lamp.n_facets} lamp facets")
        values, timing = fluence(lamp, centre, body, f"{mesh} volume")

        on_slices = np.unique(
            np.concatenate([room.slice_cells(centre, volume, name) for name in room.SLICES])
        )
        refined_triangles = subdivide_to_width(
            np.asarray(lamp.vertices), geometry["floor"]["centre"], max_ratio=REFINED_RATIO
        ).vertices
        refined = Surfaces.from_triangles(
            refined_triangles, emission=lamp.emission[0], profiles=lamp.profiles
        )
        fine, fine_timing = fluence(
            refined, centre[on_slices], body, f"{mesh} slices, refined lamp"
        )
        check = refinement_check(values[on_slices], fine, centre[on_slices])
        power = room.lamp_power()
        record = {
            "mesh": mesh,
            "cells": len(centre),
            "lamp_facets": int(lamp.n_facets),
            "refined_lamp_facets": int(refined.n_facets),
            "slice_cells": len(on_slices),
            "coarse_vs_refined_on_slices": check,
            "volume_integral_G_W_m": float(np.sum(values * volume)),
            "lamp_power_W": power,
            "timing": {"volume": timing, "slices_refined": fine_timing},
            "total_s": round(time.perf_counter() - started, 1),
            "backend": jax.default_backend(),
            "host": platform.platform(),
            "code": room.code_version(),
            "cpu_count": os.cpu_count(),
            "jax": jax.__version__,
        }
        np.savez(OUT / f"{mesh}_volume.npz", G=values, slice_cells=on_slices, G_refined=fine)
        (OUT / f"{mesh}_volume.json").write_text(json.dumps(record, indent=2) + "\n")
        _say(f"{mesh}: {json.dumps(record['coarse_vs_refined_on_slices'])}, {timing}")
    _say("done")


if __name__ == "__main__":
    main()
