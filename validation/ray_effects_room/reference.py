"""The exact floor irradiance and fluence rate, by brute force: the ground truth both solvers meet.

At each receiver ``r`` the lamp's window is integrated point by point. On the floor, the
irradiance

    E(r) = sum over sample points s of  I(d_sr) / A_lamp * dA_s * cos(theta_r) / |r - s|^2 * V(s, r)

with ``I`` the measured intensity of the Care222 module towards ``d_sr`` (W/sr), ``A_lamp`` the
window's area, ``theta_r`` the angle at the floor, and ``V`` one if the segment misses the bunny.
In the air, the fluence rate ``G`` is the same sum without the receiver's cosine. A window
radiating ``I(d)`` as a whole has radiance ``I(d) / (A cos gamma)`` at every point, which is what
makes this the field of the extended source rather than of a point.

**Independent of aquaflux on purpose.** It imports nothing from the package: the photometric file
is parsed and interpolated here, from the LM-63 conventions directly, and visibility is decided
by Embree's ray caster (through trimesh). What it shares with the solvers is the geometry -- the
lamp, floor and bunny patches of the one mesh from ``patches.npz``, and the cell centres from
``cells.npz`` -- and the choice of slices, from ``room.py`` (numpy only), so that a difference from
it is a difference in method.

The volume is done on the cells of the two slices in ``room.SLICES`` only: every cell of the mesh
would be some 44 billion sample-receiver pairs.

**Discretization.** Each of the lamp patch's fan triangles is sampled at ``SAMPLES_PER_SIDE**2``
barycentric points; the run repeats a strip of the floor with four times as many and reports the
change. Pairs whose segment cannot reach the bunny's bounding box are clear without a ray.

The bunny can also be the STL the mesh was snapped to (``RAY_REFERENCE_BUNNY=stl``), to show how
much of any difference the snapping itself accounts for.

Writes ``work/reference/<mesh>[_stl].npz`` (``E`` per floor face; ``G_<slice>`` and
``cells_<slice>`` per slice) and a ``.json`` record.

Environment: ``RAY_MESHES`` (default ``"empty bunny"``); ``RAY_REFERENCE_BUNNY`` (``mesh`` or
``stl``). Needs ``numpy``, ``trimesh`` and ``embreex``, and no aquaflux.
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import room  # numpy only

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
OUT = WORK / "reference"
IES = HERE / "ushio_b1.ies"
UP = np.array([1.0, 0.0, 0.0])  # the fixture's h = 0 direction
AXIS = np.array([0.0, 0.0, -1.0])  # the fixture's gamma = 0 direction
SAMPLES_PER_SIDE = 4  # barycentric subdivisions per edge: SAMPLES_PER_SIDE**2 points a triangle
CHUNK = 256


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_ies_table(path: Path):
    """Vertical and horizontal angles (degrees) and the intensity table, ``[h][gamma]``, in W/sr.

    The file states milliwatts per steradian; that is checked. A full table (last horizontal
    angle 360) is the only kind this module handles, which is what the Care222 file is.
    """
    text = path.read_text()
    unit = re.search(r"^\[_INTENSITYUNITS\]\s*(\S+)", text, re.M).group(1)
    if unit != "mW/sr":
        msg = f"expected mW/sr, got {unit}"
        raise ValueError(msg)
    numbers = np.array(text.split("TILT=NONE")[1].split(), dtype=float)
    multiplier, nv, nh = numbers[2], int(numbers[3]), int(numbers[4])
    vertical = numbers[13 : 13 + nv]
    horizontal = numbers[13 + nv : 13 + nv + nh]
    table = numbers[13 + nv + nh :].reshape(nh, nv) * multiplier * numbers[10] * 1e-3
    if horizontal[-1] != 360.0 or horizontal[0] != 0.0:
        msg = "this reference handles a full 0-360 table only"
        raise ValueError(msg)
    return vertical, horizontal, table


def intensity(directions: np.ndarray, vertical, horizontal, table) -> np.ndarray:
    """``I`` towards each unit direction, W/sr: bilinear in (gamma, h) in degrees, gamma clamped.

    ``h`` is measured from ``UP`` towards ``AXIS x UP``, as LM-63 Type C and of-optical-radiation's
    ``iesEmitter`` define it; zero behind the window.
    """
    cos_gamma = directions @ AXIS
    gamma = np.degrees(np.arccos(np.clip(cos_gamma, -1.0, 1.0)))
    second = np.cross(AXIS, UP)
    h = np.degrees(np.arctan2(directions @ second, directions @ UP)) % 360.0
    gamma = np.clip(gamma, vertical[0], vertical[-1])
    iv = np.clip(np.searchsorted(vertical, gamma, side="right") - 1, 0, len(vertical) - 2)
    fv = (gamma - vertical[iv]) / (vertical[iv + 1] - vertical[iv])
    ih = np.clip(np.searchsorted(horizontal, h, side="right") - 1, 0, len(horizontal) - 2)
    fh = (h - horizontal[ih]) / (horizontal[ih + 1] - horizontal[ih])
    low = (1 - fv) * table[ih, iv] + fv * table[ih, iv + 1]
    high = (1 - fv) * table[ih + 1, iv] + fv * table[ih + 1, iv + 1]
    return np.where(cos_gamma > 0.0, (1 - fh) * low + fh * high, 0.0)


def window_samples(triangles: np.ndarray, per_side: int) -> tuple[np.ndarray, np.ndarray]:
    """Points on each triangle at the centroids of its ``per_side**2`` similar sub-triangles.

    Returns the points ``(n, 3)`` and the area each stands for ``(n,)``: exact for a flat
    triangle, and each sub-triangle's centroid is the one-point rule's optimum.
    """
    points, weights = [], []
    area = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    k = per_side
    for i in range(k):
        for j in range(k - i):
            # Upward sub-triangle (i, j), and the downward one beside it where there is room.
            for u, v in (((i + 1 / 3) / k, (j + 1 / 3) / k), ((i + 2 / 3) / k, (j + 2 / 3) / k)):
                if u + v > 1.0 + 1e-12:
                    continue
                points.append(
                    triangles[:, 0]
                    + u * (triangles[:, 1] - triangles[:, 0])
                    + v * (triangles[:, 2] - triangles[:, 0])
                )
                weights.append(area / k**2)
    return np.concatenate(points), np.concatenate(weights)


def exact_field(receivers, sources, weights, table, bunny, window_area, *, facing_up) -> np.ndarray:
    """``E`` at receivers on the floor (``facing_up``), else ``G`` at receivers in the air, W/m^2."""
    vertical, horizontal, values = table
    if bunny is not None:
        low, high = bunny.bounds
        caster = bunny.ray
    result = np.zeros(len(receivers))
    for start in range(0, len(receivers), CHUNK):
        chunk = receivers[start : start + CHUNK]
        offset = chunk[:, None, :] - sources[None, :, :]
        distance_sq = np.einsum("ijk,ijk->ij", offset, offset)
        direction = offset / np.sqrt(distance_sq)[..., None]
        # The floor faces +z; a point in the air counts every direction alike.
        cos_receiver = np.clip(-direction[..., 2], 0.0, None) if facing_up else 1.0
        light = intensity(direction.reshape(-1, 3), vertical, horizontal, values).reshape(
            distance_sq.shape
        )
        term = light * weights[None, :] / window_area * cos_receiver / distance_sq
        if bunny is not None:
            # Only a segment that enters the bunny's bounding box can be blocked.
            maybe = _segment_meets_box(sources[None, :, :], chunk[:, None, :], low, high)
            i, j = np.nonzero(maybe & (term > 0.0))
            if len(i):
                origins = sources[j]
                vectors = chunk[i] - origins
                length = np.linalg.norm(vectors, axis=1)
                hit, ray = caster.intersects_location(
                    origins, vectors / length[:, None], multiple_hits=False
                )[:2]
                blocked = np.zeros(len(i), dtype=bool)
                if len(ray):
                    reach = np.linalg.norm(hit - origins[ray], axis=1)
                    blocked[ray[reach < length[ray]]] = True
                term[i[blocked], j[blocked]] = 0.0
        result[start : start + len(chunk)] = term.sum(axis=1)
    return result


def _segment_meets_box(a, b, low, high) -> np.ndarray:
    """Whether each segment ``a -> b`` meets the axis-aligned box (the slab test)."""
    d = b - a
    safe = np.where(d == 0.0, 1e-300, d)
    t0, t1 = (low - a) / safe, (high - a) / safe
    near = np.minimum(t0, t1).max(axis=-1)
    far = np.maximum(t0, t1).min(axis=-1)
    return (near <= far) & (far >= 0.0) & (near <= 1.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = read_ies_table(IES)
    source_kind = os.environ.get("RAY_REFERENCE_BUNNY", "mesh")
    for mesh in os.environ.get("RAY_MESHES", "empty bunny").split():
        started = time.perf_counter()
        stored = np.load(WORK / mesh / "patches.npz")
        receivers = stored["floor/centre"]
        lamp = stored["lamp/triangles"]
        window_area = float(stored["lamp/area"].sum())
        bunny = None
        if mesh == "bunny":
            if source_kind == "stl":
                bunny = trimesh.load(
                    WORK / "bunny" / "case" / "constant" / "triSurface" / "bunny.stl"
                )
            else:
                triangles = stored["bunny/triangles"]
                bunny = trimesh.Trimesh(
                    vertices=triangles.reshape(-1, 3),
                    faces=np.arange(3 * len(triangles)).reshape(-1, 3),
                    process=False,
                )
            if not isinstance(bunny.ray, trimesh.ray.ray_pyembree.RayMeshIntersector):
                msg = "Embree is not in use; install embreex"
                raise RuntimeError(msg)
        sources, weights = window_samples(lamp, SAMPLES_PER_SIDE)
        _say(f"{mesh}: {len(sources)} window samples, {len(receivers)} floor faces, bunny "
             f"{'none' if bunny is None else source_kind}")  # fmt: skip
        values = exact_field(receivers, sources, weights, table, bunny, window_area, facing_up=True)
        elapsed = time.perf_counter() - started

        # Discretization: a strip across the shadow at four times the samples per triangle.
        strip = np.flatnonzero(np.abs(receivers[:, 1] - 0.1) < 0.02)
        dense_sources, dense_weights = window_samples(lamp, 2 * SAMPLES_PER_SIDE)
        dense = exact_field(
            receivers[strip],
            dense_sources,
            dense_weights,
            table,
            bunny,
            window_area,
            facing_up=True,
        )
        change = np.abs(values[strip] - dense)
        cells = np.load(WORK / mesh / "cells.npz")
        slices = {}
        for slice_name in room.SLICES:
            chosen = room.slice_cells(cells["centre"], cells["volume"], slice_name)
            began = time.perf_counter()
            slices[slice_name] = (
                chosen,
                exact_field(
                    cells["centre"][chosen], sources, weights, table, bunny, window_area,
                    facing_up=False,
                ),
            )  # fmt: skip
            _say(
                f"{mesh}: {slice_name} slice, {len(chosen)} cells, {time.perf_counter() - began:.0f} s"
            )
        power = float(np.sum(values * stored["floor/area"]))
        record = {
            "mesh": mesh,
            "bunny": None if bunny is None else source_kind,
            "window_samples": len(sources),
            "window_area_m2": window_area,
            "floor_faces": len(receivers),
            "floor_power_W": power,
            "strip_faces": len(strip),
            "strip_samples_x4_max_change_of_peak": float(change.max() / values.max()),
            "seconds": round(elapsed, 1),
            "slice_cells": {slice_name: len(chosen) for slice_name, (chosen, _) in slices.items()},
            "host": platform.platform(),
            "python": sys.version.split()[0],
            "trimesh": trimesh.__version__,
        }
        name = mesh if bunny is None or source_kind == "mesh" else f"{mesh}_stl"
        arrays = {"E": values}
        for slice_name, (chosen, field) in slices.items():
            arrays[f"cells_{slice_name}"], arrays[f"G_{slice_name}"] = chosen, field
        np.savez(OUT / f"{name}.npz", **arrays)
        (OUT / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
        _say(f"{name}: {json.dumps(record)}")


if __name__ == "__main__":
    main()
