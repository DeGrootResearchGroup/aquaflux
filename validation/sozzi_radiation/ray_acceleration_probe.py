"""What a uniform grid is worth for the shadow mask, measured on a real reactor.

The shadow mask tests every (receiver, facet) segment against every triangle of the surface
with no early exit, so it costs ``n_receivers x n_facets x n_triangles``: on this reactor
1.6M x 7,516 x 53,500 ~ **6e14** intersections, weeks at the measured 120-150 Mtest/s. The
design record's remedy is a uniform grid -- register each triangle into the voxels its bounding
box spans, walk each segment through the grid, and test only what the voxels entered hold --
with an assumed "~20 triangles tested per ray" that **has never been measured**. That figure is
the whole decision: at 20 tests a ray the mask is minutes, at 2000 it is days.

This measures it, and measures the two things the assumption hides:

1. **Voxels entered per segment.** Under `jax`, a traced grid walk needs a *static* trip count,
   so it pays for the longest possible walk on every ray whether or not the voxels hold
   anything. The useful quantity is therefore not only how many triangles are tested but how
   many *steps* are taken -- ``steps x triangles_per_voxel`` is what a static implementation
   actually costs, and it can be worse than brute force.
2. **What early exit buys.** A host-side walk can stop at the first opaque hit; a traced one
   cannot. Both are reported, because the gap between them is the cost of staying traceable.

It also measures the alternative that fits this package's existing idiom -- the silhouette clip
already compacts candidates on the host, since the mask is frozen and nothing about it has to
be traceable. **Blocked culling**: sort receivers into spatial blocks, take each block of
receivers against each block of facets, and keep only the triangles whose bounding box meets
the box around that pair. Empty pairs cost nothing, which is what matters in a reactor where
most of the domain has nothing between it and the lamp.

Run with ``validation/run_case.sh validation/sozzi_radiation/ray_acceleration_probe.py`` after
``generate_dom_reference.py`` has meshed the case.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import aquaflux  # noqa: E402,F401  (enables x64)
from aquaflux.io import read_openfoam  # noqa: E402
from aquaflux.radiation import read_stl  # noqa: E402

WORK = HERE / "work"
CASE = WORK / "case"
RESOLUTIONS = (32, 64, 128)
RAYS = 4000
RECEIVER_BLOCK = 4096
FACET_BLOCK = 256


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def cell_centres() -> np.ndarray:
    """The mesh's cell centres, cached: reading 1.6M cells of ASCII costs ~30 s."""
    cached = WORK / "cell_centres.npy"
    if cached.exists():
        return np.load(cached)
    _say("reading the mesh")
    centres = np.asarray(read_openfoam(CASE).geometry().cell.centroid)
    np.save(cached, centres)
    return centres


def triangle_voxels(vertices: np.ndarray, origin, spacing, resolution: int):
    """Which voxels each triangle's bounding box spans, as a flat (voxel, triangle) pair list."""
    low = np.floor((vertices.min(axis=1) - origin) / spacing).astype(int)
    high = np.floor((vertices.max(axis=1) - origin) / spacing).astype(int)
    np.clip(low, 0, resolution - 1, out=low)
    np.clip(high, 0, resolution - 1, out=high)
    voxel_of, triangle_of = [], []
    for index, (lo, hi) in enumerate(zip(low, high, strict=True)):
        spans = np.meshgrid(
            *(np.arange(lo[axis], hi[axis] + 1) for axis in range(3)), indexing="ij"
        )
        flat = np.stack([span.ravel() for span in spans], axis=1)
        voxel_of.append((flat[:, 0] * resolution + flat[:, 1]) * resolution + flat[:, 2])
        triangle_of.append(np.full(len(flat), index))
    return np.concatenate(voxel_of), np.concatenate(triangle_of)


def walk(origin_point, target_point, origin, spacing, resolution: int):
    """Voxels a segment passes through, in order (Amanatides & Woo's 3D-DDA)."""
    start = np.clip(((origin_point - origin) / spacing).astype(int), 0, resolution - 1)
    finish = np.clip(((target_point - origin) / spacing).astype(int), 0, resolution - 1)
    direction = target_point - origin_point
    step = np.where(direction > 0, 1, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.abs(spacing / np.where(direction == 0.0, np.inf, direction))
        boundary = origin + (start + (step > 0)) * spacing
        until = np.abs((boundary - origin_point) / np.where(direction == 0.0, np.inf, direction))
    until = np.where(np.isfinite(until), until, np.inf)
    delta = np.where(np.isfinite(delta), delta, np.inf)
    voxel = start.copy()
    visited = [voxel.copy()]
    for _ in range(3 * resolution):
        if np.array_equal(voxel, finish):
            break
        axis = int(np.argmin(until))
        voxel[axis] += step[axis]
        if not (0 <= voxel[axis] < resolution):
            break
        until[axis] += delta[axis]
        visited.append(voxel.copy())
    return np.array(visited)


def measure_grid(receivers, facets, vertices, resolution, rng) -> dict:
    """Steps and triangle tests per ray at one grid resolution."""
    low = np.minimum(vertices.reshape(-1, 3).min(axis=0), receivers.min(axis=0)) - 1e-6
    high = np.maximum(vertices.reshape(-1, 3).max(axis=0), receivers.max(axis=0)) + 1e-6
    spacing = (high - low) / resolution
    voxel_of, triangle_of = triangle_voxels(vertices, low, spacing, resolution)
    order = np.argsort(voxel_of, kind="stable")
    voxel_of, triangle_of = voxel_of[order], triangle_of[order]
    counts = np.bincount(voxel_of, minlength=resolution**3)
    starts = np.concatenate([[0], np.cumsum(counts)])

    steps, tested, tested_until_hit = [], [], []
    for _ in range(RAYS):
        receiver = receivers[rng.integers(len(receivers))]
        facet = facets[rng.integers(len(facets))]
        visited = walk(facet, receiver, low, spacing, resolution)
        flat = (visited[:, 0] * resolution + visited[:, 1]) * resolution + visited[:, 2]
        per_voxel = counts[flat]
        steps.append(len(visited))
        tested.append(int(per_voxel.sum()))
        # Early exit: stop at the first voxel that holds anything, a lower bound on the work a
        # host-side walk does (it stops at the first *hit*, which is no earlier than this).
        first = np.flatnonzero(per_voxel)
        tested_until_hit.append(int(per_voxel[: first[0] + 1].sum()) if len(first) else 0)
    del starts, triangle_of
    return {
        "resolution": resolution,
        "occupied_voxels": int(np.count_nonzero(counts)),
        "triangles_per_occupied_voxel_mean": float(counts[counts > 0].mean()),
        "triangles_per_occupied_voxel_max": int(counts.max()),
        "steps_mean": float(np.mean(steps)),
        "steps_max": int(np.max(steps)),
        "tested_mean": float(np.mean(tested)),
        "tested_p95": float(np.percentile(tested, 95)),
        "tested_until_first_occupied_mean": float(np.mean(tested_until_hit)),
        "static_cost_per_ray": float(np.max(steps) * counts.max()),
    }


def measure_blocks(receivers, facets, vertices, rng) -> dict:
    """Candidate triangles per (receiver block, facet block) pair, and how many pairs are empty.

    Receivers are blocked by a Morton-style spatial sort, so a block is compact in space; the
    box around a block pair is the box holding every segment between them, and a triangle whose
    own box misses it cannot block any of those segments.
    """
    keys = np.lexsort(
        tuple(
            np.floor((receivers[:, axis] - receivers[:, axis].min()) / 0.01).astype(int)
            for axis in (2, 1, 0)
        )
    )
    blocked = receivers[keys]
    low, high = vertices.min(axis=1), vertices.max(axis=1)
    receiver_blocks = range(0, len(blocked), RECEIVER_BLOCK)
    facet_blocks = range(0, len(facets), FACET_BLOCK)
    candidates, empty_pairs, pairs = [], 0, 0
    for start in receiver_blocks:
        block = blocked[start : start + RECEIVER_BLOCK]
        for facet_start in facet_blocks:
            patch = facets[facet_start : facet_start + FACET_BLOCK]
            box_low = np.minimum(block.min(axis=0), patch.min(axis=0))
            box_high = np.maximum(block.max(axis=0), patch.max(axis=0))
            meets = np.all((high >= box_low) & (low <= box_high), axis=1)
            count = int(meets.sum())
            candidates.append(count)
            empty_pairs += count == 0
            pairs += 1
    del rng
    return {
        "receiver_block": RECEIVER_BLOCK,
        "facet_block": FACET_BLOCK,
        "block_pairs": pairs,
        "empty_pairs_fraction": empty_pairs / pairs,
        "candidates_mean": float(np.mean(candidates)),
        "candidates_p95": float(np.percentile(candidates, 95)),
        "candidates_max": int(np.max(candidates)),
    }


def main() -> None:
    rng = np.random.default_rng(0)
    surface = CASE / "constant" / "triSurface"
    lamp = np.asarray(read_stl(surface / "lampWall.stl").vertices)
    body = np.asarray(read_stl(surface / "bodyWall.stl").vertices)
    receivers = cell_centres()
    facets = lamp.mean(axis=1)
    _say(f"{len(receivers)} receivers, {len(facets)} lamp facets, {len(body)} wall triangles")
    brute = len(receivers) * len(facets) * len(body)
    _say(f"brute force: {brute:.3g} intersection tests for the whole mask")

    sample = receivers[rng.choice(len(receivers), size=200_000, replace=False)]
    for resolution in RESOLUTIONS:
        started = time.perf_counter()
        found = measure_grid(sample, facets, body, resolution, rng)
        _say(f"grid {resolution}^3 ({time.perf_counter() - started:.0f} s): {found}")
    started = time.perf_counter()
    _say(
        f"blocked culling ({time.perf_counter() - started:.0f} s): {measure_blocks(sample, facets, body, rng)}"
    )


if __name__ == "__main__":
    main()
