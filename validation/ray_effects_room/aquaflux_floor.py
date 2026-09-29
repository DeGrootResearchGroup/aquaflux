"""aquaflux's irradiance on the room's floor, from the lamp, past the bunny.

Every surface is black and the air transparent, so the floor's irradiance is the lamp's direct
light alone: at each floor face centre, the exact projected solid angle of every lamp facet,
weighted by the facet's radiance towards that face from the Care222 photometry, and gated by
whether the segment between them crosses the bunny. No angular grid is involved, which is the
point of the comparison.

**The geometry is the mesh's.** The lamp is the ``lamp`` patch's faces (280 of them, 3.125 mm),
split by ``patches.py`` into the fan of triangles OpenFOAM measures them by and wound to face the
room; the receivers are the ``floor`` patch's face centres, facing up; and the bunny is the
``bunny`` patch -- the snapped surface the discrete-ordinates solver saw, not the STL it was
snapped to -- as a :class:`~aquaflux.radiation.TriangleBody`. So the two solvers differ in their
angular treatment and nothing else.

**What the lamp's discretization costs** is measured, not assumed: the floor is gathered again
against the lamp's triangles split to ``REFINED_WIDTH``, and the difference is reported. Each
lamp facet casts its own hard shadow, so a soft shadow is a sum of as many hard ones as there are
facets, and the question is whether their offsets are fine against a floor face.

Writes ``work/aquaflux/<mesh>.npz`` (``E``, the floor's irradiance in W/m^2, per floor face, and
the refined gather's) and ``<mesh>.json`` (timings, counts, the configuration).

Environment: ``RAY_MESHES`` (default ``"empty bunny"``).

Run with ``validation/run_case.sh validation/ray_effects_room/aquaflux_floor.py`` after
``generate_dom.py`` has built the meshes and ``patches.py`` has extracted their patches.
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
    check_profiles,
    direct_irradiance,
)
from aquaflux.radiation.subdivide import subdivide_to_width  # noqa: E402

OUT = HERE / "work" / "aquaflux"
# The refined lamp's triangles: split until each is under this fraction of the 3 m to the nearest
# floor point -- about 0.8 mm, two splits of the 3.1 mm faces' fan triangles.
REFINED_RATIO = 2.7e-4


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def lamp_surfaces(triangles: np.ndarray) -> Surfaces:
    """The lamp: the patch's triangles facing the room, at exitance P / A, with the photometry.

    A boundary face's normal points out of the air, so up into the ceiling; the emitter must face
    the room, so the winding is reversed. Checked, rather than assumed, against the fixture axis.
    """
    facing = triangles[:, ::-1, :]
    area = 0.5 * np.linalg.norm(
        np.cross(facing[:, 1] - facing[:, 0], facing[:, 2] - facing[:, 0]), axis=1
    )
    profile = room.lamp_photometry().profile(up=room.LAMP_UP)
    lamp = Surfaces.from_triangles(
        facing, emission=room.lamp_power() / area.sum(), profiles=(profile,)
    )
    if not np.allclose(np.asarray(lamp.normal), room.LAMP_AXIS, atol=1e-9):
        msg = "the lamp's facets do not all face along the fixture axis"
        raise ValueError(msg)
    check_profiles(lamp)
    return lamp


def floor_irradiance(lamp: Surfaces, floor: dict, body) -> tuple[np.ndarray, dict]:
    """``E`` at every floor face centre, and what it cost."""
    points = jnp.asarray(floor["centre"])
    normals = jnp.asarray(-floor["normal"])  # into the room
    timing = {}
    visibility = None
    if body is not None:
        started = time.perf_counter()
        visibility = build_visibility(
            [body], lamp, np.asarray(points), self_occlusion=NoOcclusion()
        )
        timing["visibility_s"] = round(time.perf_counter() - started, 2)
    started = time.perf_counter()
    values = jax.block_until_ready(direct_irradiance(lamp, points, normals, visibility=visibility))
    timing["gather_s"] = round(time.perf_counter() - started, 2)
    return np.asarray(values), timing


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for mesh in os.environ.get("RAY_MESHES", "empty bunny").split():
        started = time.perf_counter()
        geometry = patches.load(mesh)
        floor = geometry["floor"]
        lamp = lamp_surfaces(geometry["lamp"]["triangles"])
        body = None
        if mesh == "bunny":
            built = time.perf_counter()
            # A sheet: the snapped surface stops light from either side, and nothing is inside it
            # that the floor's receivers could be.
            body = TriangleBody.build(geometry["bunny"]["triangles"], sheet=True)
            _say(f"{mesh}: bunny body of {len(geometry['bunny']['triangles'])} triangles, "
                 f"{time.perf_counter() - built:.1f} s")  # fmt: skip
        _say(f"{mesh}: {lamp.n_facets} lamp facets, {len(floor['area'])} floor faces")
        coarse, timing = floor_irradiance(lamp, floor, body)
        _say(f"{mesh}: gathered, {timing}")

        refined_triangles = subdivide_to_width(
            np.asarray(lamp.vertices), floor["centre"], max_ratio=REFINED_RATIO
        ).vertices
        refined = Surfaces.from_triangles(
            refined_triangles, emission=lamp.emission[0], profiles=lamp.profiles
        )
        fine, fine_timing = floor_irradiance(refined, floor, body)
        _say(f"{mesh}: refined lamp of {refined.n_facets} facets, {fine_timing}")

        lit = fine > 1e-3 * fine.max()
        difference = np.abs(coarse - fine)
        power = room.lamp_power()
        record = {
            "mesh": mesh,
            "lamp_power_W": power,
            "lamp_facets": int(lamp.n_facets),
            "refined_lamp_facets": int(refined.n_facets),
            "floor_faces": len(floor["area"]),
            "floor_power_W": float(np.sum(fine * floor["area"])),
            "floor_share_of_lamp_power": float(np.sum(fine * floor["area"]) / power),
            "coarse_vs_refined": {
                "max_abs_W_per_m2": float(difference.max()),
                "max_rel_of_peak": float(difference.max() / fine.max()),
                "p99_rel_lit": float(np.percentile(difference[lit] / fine[lit], 99)),
            },
            "timing": {"coarse": timing, "refined": fine_timing},
            "total_s": round(time.perf_counter() - started, 1),
            "backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "host": platform.platform(),
            "cpu_count": os.cpu_count(),
            "jax": jax.__version__,
        }
        np.savez(OUT / f"{mesh}.npz", E=coarse, E_refined=fine)
        (OUT / f"{mesh}.json").write_text(json.dumps(record, indent=2) + "\n")
        _say(f"{mesh}: {json.dumps(record['coarse_vs_refined'])}, "
             f"floor receives {record['floor_share_of_lamp_power']:.4f} of the lamp's power")  # fmt: skip
    _say("done")


if __name__ == "__main__":
    main()
