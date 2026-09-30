"""aquaflux in the reflecting room: the floor, ceiling and walls reflect diffusely.

The direct light is the black room's, already computed (``aquaflux_floor.py``,
``aquaflux_volume.py``); this adds everything that has bounced off the room's surfaces at least
once, with interreflection closed to convergence -- the number of bounces is not a parameter.

**The surfaces.** Each face of the room is cut into squares of ``room.REFLECTING_FACET`` (two
triangles each), reflectance ``room.WALL_REFLECTANCE``, no emission. The bunny stays black and
stands between them as a :class:`~aquaflux.radiation.TriangleBody`; the lamp's window, which
of-optical-radiation's ``iesEmitter`` treats as black to arriving light, is 0.0027 m^2 of the
16 m^2 ceiling and is left reflecting with it.

**How the lamp enters.** Its direct irradiance on each room facet -- with the measured
photometry, through the bunny's shadows -- is gathered directly and handed to the surface solve
as an external irradiance, averaged over ``SUBPOINTS**2`` points of each triangle so the bunny's
shadow on the floor is not aliased by a single sample. The transfer matrix therefore carries only
reflected, Lambertian light; it never needs the lamp's direction-dependent profile.

**The reflected field.** The solve gives each facet's radiosity ``B``; the reflected light is
then gathered as a Lambertian set of exitance ``B`` -- at the floor faces as irradiance, at every
cell centre as fluence rate -- through the bunny's shadows. On the slice cells the same field is
also taken from the library's own :func:`~aquaflux.radiation.fluence_rate` and the two routes
compared. The squares are repeated at twice the side on the floor and the slices, to show how far
the facet size matters.

Energy: what the room's surfaces absorb, ``sum (1 - rho) H A``, with ``H`` their total irradiance
from :func:`~aquaflux.radiation.surface_irradiance`; the bunny absorbs the rest of the lamp's power.

Writes ``work/aquaflux/<mesh>_reflecting.npz`` (``E`` on the floor, ``G`` per cell, ``G_slices``
and ``slice_cells``, each the direct plus the reflected, and the reflected parts alone) and
``.json``. Environment: ``RAY_MESHES`` (default ``"bunny"``).
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
    Lambertian,
    NoOcclusion,
    Surfaces,
    TriangleBody,
    build_radiation_model,
    build_visibility,
    direct_irradiance,
    fluence_rate,
    radiosity,
    surface_irradiance,
)
from aquaflux.radiation.subdivide import subdivide_to_width  # noqa: E402
from aquaflux_floor import REFINED_RATIO, lamp_surfaces  # noqa: E402
from aquaflux_volume import fluence  # noqa: E402

OUT = HERE / "work" / "aquaflux"
SUBPOINTS = 4  # per triangle edge: 16 points per room triangle for the lamp's direct irradiance


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def subpoints(triangles: np.ndarray, per_side: int) -> np.ndarray:
    """Centroids of each triangle's ``per_side**2`` similar sub-triangles, ``(n, per_side**2, 3)``."""
    k, rows = per_side, []
    for i in range(k):
        for j in range(k - i):
            for u, v in (((i + 1 / 3) / k, (j + 1 / 3) / k), ((i + 2 / 3) / k, (j + 2 / 3) / k)):
                if u + v <= 1.0 + 1e-12:
                    rows.append(
                        triangles[:, 0]
                        + u * (triangles[:, 1] - triangles[:, 0])
                        + v * (triangles[:, 2] - triangles[:, 0])
                    )
    return np.stack(rows, axis=1)


def irradiance(sources: Surfaces, points, normals, body) -> np.ndarray:
    """Irradiance at oriented points from ``sources``, through the bunny when there is one."""
    visibility = None
    if body is not None:
        visibility = build_visibility(
            [body], sources, np.asarray(points), self_occlusion=NoOcclusion()
        )
    return np.asarray(
        jax.block_until_ready(
            direct_irradiance(
                sources, jnp.asarray(points), jnp.asarray(normals), visibility=visibility
            )
        )
    )


def reflecting_room(side: float, lamp: Surfaces, body, slice_points, floor) -> dict:
    """Solve the room's interreflection at one facet size; the reflected floor and slice fields."""
    started = time.perf_counter()
    walls = Surfaces.from_triangles(
        room.room_facets(side), emission=0.0, reflectance=room.WALL_REFLECTANCE
    )
    samples = subpoints(np.asarray(walls.vertices), SUBPOINTS)
    normals = np.repeat(np.asarray(walls.normal)[:, None, :], samples.shape[1], axis=1)
    lamp_on_walls = irradiance(lamp, samples.reshape(-1, 3), normals.reshape(-1, 3), body)
    external = lamp_on_walls.reshape(samples.shape[:2]).mean(axis=1)
    occluders = () if body is None else (body,)
    model = build_radiation_model(slice_points, walls, occluders=occluders)
    outgoing, cycles = radiosity(model, walls, external_irradiance=jnp.asarray(external))
    arriving, _ = surface_irradiance(model, walls, external_irradiance=jnp.asarray(external))
    library_slices, _ = fluence_rate(model, walls, external_irradiance=jnp.asarray(external))
    bounced = walls.with_optics(emission=outgoing, profiles=(Lambertian(),))
    floor_reflected = irradiance(bounced, floor["centre"], -floor["normal"], body)
    slices_reflected, _ = fluence(bounced, slice_points, body, f"reflected slices, {side} m facets")
    area = np.asarray(walls.area)
    return {
        "walls": walls,
        "bounced": bounced,
        "cycles": int(cycles),
        "floor": floor_reflected,
        "slices": slices_reflected,
        "slices_library": np.asarray(library_slices),
        "absorbed_by_room_W": float(
            np.sum((1.0 - room.WALL_REFLECTANCE) * np.asarray(arriving) * area)
        ),
        "lamp_on_room_W": float(np.sum(external * area)),
        "seconds": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for mesh in os.environ.get("RAY_MESHES", "bunny").split():
        started = time.perf_counter()
        geometry = patches.load(mesh)
        floor = geometry["floor"]
        cells = np.load(HERE / "work" / mesh / "cells.npz")
        centre = cells["centre"]
        direct_floor = np.load(OUT / f"{mesh}.npz")["E_refined"]
        direct_volume = np.load(OUT / f"{mesh}_volume.npz")
        slice_cells = direct_volume["slice_cells"]
        body = (
            TriangleBody.build(geometry["bunny"]["triangles"], sheet=True)
            if mesh == "bunny"
            else None
        )
        # The lamp split finely, as for the direct floor and slices: its light on the room's own
        # surfaces carries the bunny's penumbrae too.
        coarse = lamp_surfaces(geometry["lamp"]["triangles"])
        lamp = Surfaces.from_triangles(
            subdivide_to_width(
                np.asarray(coarse.vertices), floor["centre"], max_ratio=REFINED_RATIO
            ).vertices,
            emission=coarse.emission[0],
            profiles=coarse.profiles,
        )
        _say(f"{mesh}: {lamp.n_facets} lamp facets, {len(slice_cells)} slice cells")

        solved = {}
        for side in (room.REFLECTING_FACET, 2 * room.REFLECTING_FACET):
            solved[side] = reflecting_room(side, lamp, body, centre[slice_cells], floor)
            _say(f"{mesh}: {side} m facets, {solved[side]['walls'].n_facets} triangles, "
                 f"{solved[side]['cycles']} cycles, {solved[side]['seconds']} s")  # fmt: skip
        fine, half = solved[room.REFLECTING_FACET], solved[2 * room.REFLECTING_FACET]

        # The reflected field in every cell, at the finer facets.
        volume_reflected, volume_timing = fluence(
            fine["bounced"], centre, body, f"{mesh} volume, reflected"
        )

        routes = np.abs(fine["slices"] - fine["slices_library"]) / np.maximum(
            fine["slices_library"], 1e-300
        )
        power = room.lamp_power()
        record = {
            "mesh": mesh,
            "wall_reflectance": room.WALL_REFLECTANCE,
            "facet_side_m": room.REFLECTING_FACET,
            "room_triangles": int(fine["walls"].n_facets),
            "lamp_facets": int(lamp.n_facets),
            "radiosity_cycles": fine["cycles"],
            "lamp_power_W": power,
            "lamp_direct_on_room_W": fine["lamp_on_room_W"],
            "absorbed_by_room_W": fine["absorbed_by_room_W"],
            "absorbed_elsewhere_W": power - fine["absorbed_by_room_W"],
            "reflected_share_of_floor_power": float(
                np.sum(fine["floor"] * floor["area"])
                / np.sum((fine["floor"] + direct_floor) * floor["area"])
            ),
            "manual_vs_library_slices_max_rel": float(routes.max()),
            "facet_size_check": {
                "floor_reflected_max_rel_of_peak": float(
                    np.abs(fine["floor"] - half["floor"]).max() / fine["floor"].max()
                ),
                "slices_reflected_median_rel": float(
                    np.median(np.abs(fine["slices"] - half["slices"]) / fine["slices"])
                ),
            },
            "timing_s": {
                "fine": fine["seconds"],
                "coarse": half["seconds"],
                "volume_reflected": volume_timing,
                "total": round(time.perf_counter() - started, 1),
            },
            "backend": jax.default_backend(),
            "host": platform.platform(),
            "code": room.code_version(),
            "cpu_count": os.cpu_count(),
            "jax": jax.__version__,
        }
        np.savez(
            OUT / f"{mesh}_reflecting.npz",
            E=direct_floor + fine["floor"],
            E_reflected=fine["floor"],
            G=direct_volume["G"] + volume_reflected,
            G_reflected=volume_reflected,
            slice_cells=slice_cells,
            G_slices=direct_volume["G_refined"] + fine["slices"],
            G_slices_reflected=fine["slices"],
        )
        (OUT / f"{mesh}_reflecting.json").write_text(json.dumps(record, indent=2) + "\n")
        _say(
            f"{mesh}: {json.dumps({k: record[k] for k in ('absorbed_by_room_W', 'absorbed_elsewhere_W', 'reflected_share_of_floor_power', 'manual_vs_library_slices_max_rel', 'facet_size_check')})}"
        )
    _say("done")


if __name__ == "__main__":
    main()
