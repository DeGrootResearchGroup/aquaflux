"""How much of a TRIANGULATED wall's shadow mask shaft culling decides without walking a ray.

``grid_mask_check.py`` measured the triangulated wall's mask at ~39k rays/s, the one piece of the
Sozzi calculation that costs hours rather than minutes. This measures what
:class:`~aquaflux.radiation.ShaftCulling` saves on it: a tile of cells and lamp facets whose
bounding box overlaps no occupied voxel of the wall's occupancy grid is recorded clear without a
ray being walked, and a tile that fails is split and asked again.

**The wall.** The case's own ``bodyWall.stl`` when ``work/case`` exists, as a sheet (the metal is
on its far side, as in ``grid_mask_check.py``). Otherwise the chamber alone, triangulated here: a
closed cylinder at the tutorial's radius and length, 64 sides by 400 slices plus its two end
caps, wound to face the water, so the body is a vessel. The two are not the same wall -- the
generated one has no pipe openings -- and the summary says which was used.

**The receivers are a dense lattice, not a scattered sample, and that is load-bearing.** What a
block of 32 receivers can be vouched for depends on how small the block is, and a block is small
only where the receivers are dense, as a mesh's cells are. So the receivers are every point of a
``SOZZI_LATTICE`` spaced lattice (default 2 mm, near the case mesh's cell size away from the
walls) in a slab of the chamber ``SOZZI_SLAB`` thick (default 6 mm) at mid-length. A scattered
sample of the same count spreads each block across the whole chamber and understates the saving.

**The lamp** is the analytic 24 x 64 one (3,360 facets), which keeps the every-pair arm to a few
minutes a pass; its facets, not its fidelity, are what a cluster groups.

All arms run in one process on one ray set: a warm-up that pays compilation, then ``PASSES``
alternating passes, fastest kept, with the spread beside it. Every mask is compared with the
every-pair one bit for bit and the run fails if any differs. ``SOZZI_CULLING_SIZES`` names the
culling arms as in ``body_culling.py``.
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
from aquaflux.radiation import EveryPair, Surfaces, TriangleBody, read_stl  # noqa: E402
from body_culling import culling_arms  # noqa: E402
from compare_fluence import CASE, EXITANCE, R_BODY, X_BODY_END  # noqa: E402
from lamp_resolution import lamp as analytic_lamp  # noqa: E402
from primitive_occlusion import OUT  # noqa: E402

LATTICE = float(os.environ.get("SOZZI_LATTICE", 2e-3))
SLAB = float(os.environ.get("SOZZI_SLAB", 6e-3))
SIZES = os.environ.get("SOZZI_CULLING_SIZES", "32:32;32,8:32,8;32,8,2:32,8,2")
OFFSET_SCALE = 1e-6
PASSES = 2


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def generated_chamber(sectors: int = 64, slices: int = 400) -> np.ndarray:
    """The chamber as a closed cylinder of triangles along x, wound to face in."""
    angle = np.linspace(0.0, 2.0 * np.pi, sectors, endpoint=False)
    ring = R_BODY * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    x = np.linspace(0.0, X_BODY_END, slices + 1)
    faces = []
    for i in range(slices):
        for j in range(sectors):
            k = (j + 1) % sectors
            a, b = [x[i], *ring[j]], [x[i], *ring[k]]
            c, d = [x[i + 1], *ring[k]], [x[i + 1], *ring[j]]
            faces += [[a, c, b], [a, d, c]]
    for end, flip in ((0.0, False), (X_BODY_END, True)):
        centre = [end, 0.0, 0.0]
        for j in range(sectors):
            k = (j + 1) % sectors
            triangle = [centre, [end, *ring[j]], [end, *ring[k]]]
            faces.append(triangle[::-1] if flip else triangle)
    return np.asarray(faces)


def wall() -> tuple[TriangleBody, str]:
    """The wall as a body: the case's STL where it exists, else the generated chamber."""
    stl = CASE / "constant" / "triSurface" / "bodyWall.stl"
    if stl.exists():
        return TriangleBody.build(np.asarray(read_stl(stl).vertices), sheet=True), str(stl.name)
    body = TriangleBody.build(generated_chamber())
    if body.inward_pieces != 1:
        raise SystemExit("the generated chamber should be one vessel wound to face the water")
    return body, "the chamber as 64 x 400 triangles plus caps (no case present)"


def lattice_slab() -> np.ndarray:
    """Every lattice point in a slab of the chamber at mid-length, clear of the lamp."""
    x = np.arange(0.5 * (X_BODY_END - SLAB), 0.5 * (X_BODY_END + SLAB) + 1e-12, LATTICE)
    across = np.arange(-R_BODY, R_BODY + 1e-12, LATTICE)
    points = np.stack(np.meshgrid(x, across, across, indexing="ij"), axis=-1).reshape(-1, 3)
    radius = np.linalg.norm(points[:, 1:], axis=1)
    return points[(radius < R_BODY - 0.5 * LATTICE) & (radius > 0.012)]


def main() -> None:
    body, wall_source = wall()
    surfaces = Surfaces.from_triangles(analytic_lamp(24, 64), emission=EXITANCE)
    sources = np.asarray(surfaces.centroid)
    near = OFFSET_SCALE * np.sqrt(np.asarray(surfaces.area))
    points = lattice_slab()
    n_pairs = len(points) * len(sources)
    _say(
        f"{len(points)} receivers (lattice {LATTICE * 1e3:.1f} mm, slab {SLAB * 1e3:.1f} mm), "
        f"{len(sources)} lamp facets, {n_pairs:,} pairs; wall: {wall_source}, "
        f"{len(body.grid.vertices)} triangles, walk grid {tuple(int(n) for n in body.grid.resolution)}, "
        f"occupancy grid {tuple(int(n) for n in body.occupancy.resolution)}; jax {jax.__version__}, "
        f"{platform.system()} {platform.machine()}, {os.cpu_count()} cores"
    )
    arms = {"every pair": EveryPair()} | culling_arms(SIZES)
    reference, results = None, {}
    for name, arm in arms.items():
        started = time.perf_counter()
        mask = np.asarray(arm.blocked([body], sources, near, points))
        seconds = time.perf_counter() - started
        if reference is None:
            reference = mask
        certified = (
            int(arm.certified_pairs([body], sources, points)[0]) if name != "every pair" else 0
        )
        results[name] = {
            "identical": bool(np.array_equal(mask, reference)),
            "certified_share": certified / n_pairs,
            "warm_up_s": seconds,
            "passes_s": [],
        }
        _say(
            f"{name}: warm-up {seconds:.1f} s, certified {certified / n_pairs:.1%}, "
            f"identical {results[name]['identical']}"
        )
    for index in range(PASSES):
        for name, arm in arms.items():
            started = time.perf_counter()
            arm.blocked([body], sources, near, points)
            results[name]["passes_s"].append(time.perf_counter() - started)
            _say(f"pass {index + 1}, {name}: {results[name]['passes_s'][-1]:.1f} s")
    base = min(results["every pair"]["passes_s"])
    _say(f"blocked by the wall: {reference.mean():.2%} of pairs")
    _say("summary (fastest pass; spread is slowest over fastest of the same arm):")
    for name, result in results.items():
        fastest = min(result["passes_s"])
        result.update(fastest_s=fastest, spread=max(result["passes_s"]) / fastest)
        _say(
            f"  {name:>22}: {fastest:8.1f} s  {n_pairs / fastest:10,.0f} pairs/s  spread "
            f"{result['spread']:.2f}x  speed-up {base / fastest:5.2f}x  certified "
            f"{result['certified_share']:6.1%}  identical {result['identical']}"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "triangle_culling.json"
    path.write_text(
        json.dumps(
            {
                "wall": wall_source,
                "receivers": len(points),
                "lattice_m": LATTICE,
                "slab_m": SLAB,
                "facets": len(sources),
                "blocked_share": float(reference.mean()),
                "arms": results,
            },
            indent=2,
        )
    )
    _say(f"wrote {path}")
    if not all(result["identical"] for result in results.values()):
        raise SystemExit("a culled mask differs from the every-pair mask")


if __name__ == "__main__":
    main()
