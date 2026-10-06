"""The fluence rate on the two slices with the lamp split fine, and what the coarse lamp costs there.

``cases/<mesh>_volume.yaml`` gathers every cell against the lamp patch's own 1,120 triangles, since
splitting them against every cell is not affordable at 2.5M cells. On the cells of the two slices in
``room.SLICES`` -- where the reference is computed -- this harness gathers again with the lamp split as
``cases/<mesh>_floor.yaml`` splits it (2.7e-4 of the distance to the nearest floor point, ~0.8 mm,
17,920 triangles), and reports the coarse lamp's error by distance from the lamp. Each lamp triangle
casts its own hard shadow, so a soft shadow on a slice is a sum of as many as there are triangles.

It is a developer harness: everything but the lamp's refinement and the points gathered at is the case
file's, read and built by the library (``CaseFile.check().build()``), so the two cannot describe
different rooms.

Writes ``work/cases/<mesh>_volume/slices_refined.npz`` (``slice_cells``, ``G_refined``) and
``slices_refined.json``. Run after the case: ``RAY_MESHES="bunny empty"
validation/run_case.sh validation/ray_effects_room/slices_refined.py --wait``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import outputs  # noqa: E402
import room  # noqa: E402
from aquaflux.case import read_case  # noqa: E402
from aquaflux.radiation import VolumeReceivers, refine_for_receivers, solve_scene  # noqa: E402

#: The floor case's lamp refinement, ``cases/<mesh>_floor.yaml``'s ``lamp_refinement``.
REFINED_RATIO = 2.7e-4
BANDS = ((0.05, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 6.0))  # m from the lamp's centre


def refinement_check(coarse: np.ndarray, fine: np.ndarray, points: np.ndarray) -> dict:
    """The coarse lamp's relative error against the refined one, by distance from the lamp.

    Over lit cells only (refined ``G`` above 1e-3 of its 99th percentile), since a relative error
    divides by nothing in a shadow; banded, because the error is a near-field one.
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
    for mesh in os.environ.get("RAY_MESHES", "bunny empty").split():
        started = time.perf_counter()
        case = f"{mesh}_volume"
        scene = read_case(HERE / "cases" / f"{case}.yaml").check().build()
        centre = scene.volume.points
        # The slices are cut from OpenFOAM's own cell centres, as the reference's are, so the two
        # name the same cells; the light is gathered at aquaflux's, which agree with them to rounding.
        cells = np.load(HERE / "work" / mesh / "cells.npz")
        chosen = np.unique(
            np.concatenate(
                [room.slice_cells(cells["centre"], cells["volume"], name) for name in room.SLICES]
            )
        )
        lamps, _ = refine_for_receivers(
            scene.lamps, scene.surfaces["floor"].points, max_ratio=REFINED_RATIO
        )
        print(f"{mesh}: {len(chosen)} slice cells, {lamps.n_facets} lamp facets", flush=True)
        solution = solve_scene(
            dataclasses.replace(
                scene,
                lamps=lamps,
                volume=VolumeReceivers(points=centre[chosen]),
                surfaces={},
            ),
            report=lambda line: print(f"  {line}", flush=True),
        )
        fine = solution.fluence_rate
        coarse = outputs.cell_field(case, "G")[chosen]
        record = {
            "case": f"cases/{case}.yaml",
            "lamp_facets": int(lamps.n_facets),
            "slice_cells": len(chosen),
            "coarse_vs_refined_on_slices": refinement_check(coarse, fine, centre[chosen]),
            "seconds": round(time.perf_counter() - started, 1),
            "code": room.code_version(),
        }
        directory = outputs.run_directory(case)
        np.savez(directory / "slices_refined.npz", slice_cells=chosen, G_refined=fine)
        (directory / "slices_refined.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"{mesh}: {json.dumps(record['coarse_vs_refined_on_slices'])}", flush=True)


if __name__ == "__main__":
    main()
