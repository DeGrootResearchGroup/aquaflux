"""What the ray-cast receiver mask costs, held and streamed, with and without the grid.

The scene is the shape of a real annular reactor, with every surface a triangle so the surface's
own triangles are the only thing casting shadows: a lamp (the Sozzi & Taghipour lamp's radius and
length) emitting 696.42 W/m², a second, dark sleeve beside it that shadows part of the water, and
the vessel wall around both, dark and facing inward. Receivers are drawn uniformly in the annulus
between the lamp and the wall, outside the sleeve, and the water is ``UniformAbsorption(35.67)``.

Three builds of the volume-receiver mask by ``RayCastOcclusion``, each timed warm (the median of
``CALLS`` after one untimed call) and followed by a checksum of the mask and of the fluence rate
it gives, so a before-and-after pair can be seen to compute the same field:

* **every triangle** -- ``grid=False`` -- on ``BRUTE_RECEIVERS`` receivers;
* **the grid** -- ``grid=True`` -- on ``GRID_RECEIVERS``;
* **streamed** -- ``direct_fluence_rate(occluders=[], self_occlusion=RayCastOcclusion(grid=True))``
  over ``GRID_RECEIVERS`` at ``STREAMED_PAIR_LIMIT``, which cuts it into many passes, so what a
  pass repeats is visible.

It also prints the share of areal receiver-facet pairs that face their receiver, since only
those can carry light from a facet that is dark behind itself.

With ``RADIATION_WORK_LIMIT_SWEEP=1`` in the environment it instead times the brute-force ray
test alone, ``WORK_SWEEP_RAYS`` random lamp-to-receiver rays against the first
``WORK_SWEEP_TRIANGLES`` of the scene's triangles, at each of ``WORK_LIMITS``: two passes
alternating through the limits, fastest kept, with the answers checked identical across limits.

Run with ``validation/run_case.sh validation/radiation_receiver_ray_mask.py``.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.radiation import (  # noqa: E402
    RayCastOcclusion,
    Surfaces,
    UniformAbsorption,
    build_visibility,
    direct_fluence_rate,
)
from aquaflux.radiation.triangles import segment_is_cut  # noqa: E402
from tests.unit.radiation_references import cylinder_triangles  # noqa: E402

#: Radius, half-length, sectors and slices of the lamp, the dark sleeve and the vessel wall.
LAMP = (0.0115, 0.2, 24, 32)
SLEEVE = (0.004, 0.25, 12, 16)
SLEEVE_AT = np.array([0.03, 0.0, 0.0])
WALL = (0.044, 0.25, 48, 32)
BRUTE_RECEIVERS = 500
GRID_RECEIVERS = 4_000
STREAMED_PAIR_LIMIT = 1_000_000
CALLS = 3
WORK_SWEEP_RAYS = 400_000
WORK_SWEEP_TRIANGLES = 1_532
WORK_LIMITS = (250_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000)


def scene():
    """The lamp, the sleeve and the wall as one surface set, and where the sleeve is."""
    lamp = cylinder_triangles(*LAMP)
    sleeve = cylinder_triangles(*SLEEVE) + SLEEVE_AT
    wall = cylinder_triangles(*WALL)[:, ::-1, :]
    emission = np.concatenate([np.full(len(lamp), 696.42), np.zeros(len(sleeve) + len(wall))])
    return Surfaces.from_triangles(np.concatenate([lamp, sleeve, wall]), emission=emission)


def receivers(rng, n):
    """Points uniform in angle and height between the lamp and the wall, outside the sleeve."""
    points = np.zeros((0, 3))
    while len(points) < n:
        radius = rng.uniform(0.0125, 0.043, 2 * n)
        angle = rng.uniform(0.0, 2.0 * np.pi, 2 * n)
        height = rng.uniform(-0.24, 0.24, 2 * n)
        batch = np.stack([radius * np.cos(angle), radius * np.sin(angle), height], axis=1)
        clear = np.linalg.norm(batch[:, :2] - SLEEVE_AT[:2], axis=1) > 0.005
        points = np.concatenate([points, batch[clear]])
    return points[:n]


def timed(build):
    """The result of one untimed call, and the median seconds of ``CALLS`` more."""
    result = jax.block_until_ready(build())
    seconds = []
    for _ in range(CALLS):
        start = time.perf_counter()
        jax.block_until_ready(build())
        seconds.append(time.perf_counter() - start)
    return result, statistics.median(seconds)


def facing_share(surfaces, points):
    """Of the areal receiver-facet pairs, the share whose facet faces its receiver."""
    areal = ~surfaces.is_point_source
    centroid = np.asarray(surfaces.centroid)[areal]
    normal = np.asarray(surfaces.normal)[areal]
    offset = points[:, None, :] - centroid[None, :, :]
    return float(np.mean(np.sum(offset * normal[None, :, :], axis=-1) > 0.0))


def work_limit_sweep(surfaces, rng):
    """Mtest/s of the brute-force ray test at each work limit, fastest of two passes."""
    vertices = jnp.asarray(np.asarray(surfaces.vertices)[:WORK_SWEEP_TRIANGLES])
    origin = jnp.asarray(np.asarray(surfaces.centroid)[rng.integers(0, 1_536, WORK_SWEEP_RAYS)])
    target = jnp.asarray(receivers(rng, WORK_SWEEP_RAYS))
    near = jnp.full(WORK_SWEEP_RAYS, 1e-9)
    tests = WORK_SWEEP_RAYS * WORK_SWEEP_TRIANGLES
    best, answers = {}, {}
    for _ in range(2):
        for limit in WORK_LIMITS:
            cut = segment_is_cut(origin, target, vertices, near, work_limit=limit)
            start = time.perf_counter()
            cut = jax.block_until_ready(
                segment_is_cut(origin, target, vertices, near, work_limit=limit)
            )
            best[limit] = min(best.get(limit, np.inf), time.perf_counter() - start)
            answers[limit] = np.asarray(cut)
    same = all(np.array_equal(answers[limit], answers[WORK_LIMITS[0]]) for limit in WORK_LIMITS)
    for limit in WORK_LIMITS:
        print(
            f"work limit {limit}: {tests / best[limit] / 1e6:.0f} Mtest/s "
            f"({WORK_SWEEP_RAYS} rays x {WORK_SWEEP_TRIANGLES} triangles)",
            flush=True,
        )
    print(f"answers identical across limits: {same}", flush=True)


def main():
    surfaces = scene()
    if os.environ.get("RADIATION_WORK_LIMIT_SWEEP") == "1":
        print(f"jax {jax.__version__}", flush=True)
        work_limit_sweep(surfaces, np.random.default_rng(1))
        return
    water = UniformAbsorption(35.67)
    rng = np.random.default_rng(0)
    brute_points = receivers(rng, BRUTE_RECEIVERS)
    grid_points = receivers(rng, GRID_RECEIVERS)
    print(
        f"scene: {surfaces.n_facets} facets; jax {jax.__version__}; "
        f"facing share {facing_share(surfaces, grid_points):.3f}",
        flush=True,
    )

    for name, points, grid in (
        ("every triangle", brute_points, False),
        ("grid", grid_points, True),
    ):
        strategy = RayCastOcclusion(grid=grid)
        mask, seconds = timed(
            lambda points=points, strategy=strategy: (
                build_visibility([], surfaces, points, self_occlusion=strategy).hidden_by_geometry
            )
        )
        field = direct_fluence_rate(
            surfaces,
            points,
            absorption=water,
            visibility=build_visibility([], surfaces, points, self_occlusion=strategy),
        )
        print(
            f"{name}: {len(points)} receivers, {seconds:.2f} s, "
            f"{float(jnp.mean(mask)):.6f} of pairs hidden, "
            f"field checksum {float(jnp.sum(field)):.12e}",
            flush=True,
        )

    field, seconds = timed(
        lambda: direct_fluence_rate(
            surfaces,
            grid_points,
            absorption=water,
            occluders=[],
            self_occlusion=RayCastOcclusion(grid=True),
            pair_limit=STREAMED_PAIR_LIMIT,
        )
    )
    passes = -(-GRID_RECEIVERS // (STREAMED_PAIR_LIMIT // surfaces.n_facets))
    print(
        f"streamed, grid: {GRID_RECEIVERS} receivers in {passes} passes, {seconds:.2f} s, "
        f"field checksum {float(jnp.sum(field)):.12e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
