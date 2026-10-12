# ruff: noqa: E402  (imports follow the sys.path bootstrap)
"""What the refracted gather's corner rule costs, route by route, in the four-lamp scene.

The refracted gather counts a source triangle along a route only where the paths to all three of its
corners exist, and on a route through a region it pairs the three corners' paths by the starting
point each was found from. Two questions, per route and point:

- **Corners matched by start index or by path.** On a route searched from several starting points, a
  start can find one path at one corner and another path, or none, at the next. Re-pairing each
  corner's path with its nearest path (by crossing points) at the other corners, a triangle attributed
  to the first corner it is found at, gives what matching by path would count; the difference is what
  pairing by start index miscounts.
- **Triangles with some corners unreached.** What a triangle reached at one or two corners would add
  if counted in proportion to its reached corners -- the arrival directions at its unreached corners
  are unknown, so its solid angle is taken along straight lines from the receiver. An estimate of the
  scale of what the rule drops, not a value.

No shadow mask: the source arc's far side faces away and sends nothing, but a path's last legs may
cross a neighbour's arc unshadowed, so the totals here read a little above the masked gather's.
``SLEEVE_SECTORS`` (default 16).
"""

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.radiation import Surfaces, UniformAbsorption, refracted
from aquaflux.radiation.refraction import Media, Transparent, _path_count, _shot, solve_paths
from aquaflux.radiation.solid_angle import solid_angle
from aquaflux.radiation.work import DEFAULT_PAIR_LIMIT
from aquaflux.solids import Cylinder
from check_refraction import graded_tube
from tracer import AIR, QUARTZ, WATER
from tracer import Scene as Traced

SECTORS = int(os.environ.get("SLEEVE_SECTORS", 16))
POINTS_MM = [[55.0, 25.0], [50.0, 50.0], [-25.0, 41.0], [0.0, -25.0]]

pitch, absorption = 0.025, -np.log(0.95) / 0.01
centres = pitch * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
traced = Traced(centres=centres, wall=0.1, absorption=absorption, neighbours="optics")
arc = graded_tube(traced.arc, 1.5, SECTORS)
triangles = np.concatenate([arc + np.array([x, y, 0.0]) for x, y in centres])
area = 0.5 * np.linalg.norm(
    np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
)
lamps = Surfaces.from_triangles(jnp.asarray(triangles), emission=3.0 / (area.sum() / 4))
sleeves = tuple(
    Transparent(
        Cylinder([x, y, 0.0], [0, 0, 1], traced.outer, 1.75),
        QUARTZ,
        inside=(Transparent(Cylinder([x, y, 0.0], [0, 0, 1], traced.inner, 1.75), AIR),),
    )
    for x, y in centres
)
media = Media(WATER, sleeves, UniformAbsorption(absorption))
points = np.array([[*p, 0.0] for p in POINTS_MM]) / 1e3
print(f"{SECTORS} sectors, {lamps.n_facets} facets, points {POINTS_MM} mm", flush=True)


def label(chain):
    if not chain.passing:
        return "base"
    kind = "wall" if len(chain.passing) == 1 else "gap"
    return f"{kind} of sleeve {chain.passing[0] // 2}" + (" (shot)" if _shot(media, chain) else "")


def weights(paths, corners_of, receiver, normal, emission, ratio, profile, slots):
    """``(n_facets,)`` each facet's weight along the corner paths ``slots`` picks, 0 where not all
    three exist. ``slots`` is ``(n_facets, 3)`` start indices, -1 for none."""
    facets = np.arange(corners_of.shape[0])
    picked = [np.clip(slots[:, k], 0, None) for k in range(3)]
    arrival = np.stack([paths.arrival[picked[k], corners_of[:, k]] for k in range(3)], axis=1)
    departure = np.stack([paths.departure[picked[k], corners_of[:, k]] for k in range(3)], axis=1)
    carried_t = np.stack([paths.transmittance[picked[k], corners_of[:, k]] for k in range(3)], 1)
    ok = np.all(slots >= 0, axis=1)
    ok &= np.all(
        np.stack([paths.valid[picked[k], corners_of[:, k]] for k in range(3)], axis=1), axis=1
    )
    omega = np.asarray(solid_angle(receiver[None], receiver[None, None] + arrival))
    radiance = np.asarray(profile.radiance_per_exitance(departure, normal[facets, None]))
    return np.where(ok, emission * ratio * np.mean(radiance * carried_t, axis=1) * omega, 0.0)


def matched_slots(paths, corners_of):
    """Every distinct triple of paths a triangle is reached along, its corners matched by path.

    For each corner in turn and each path found there, the nearest path at each other corner (by
    crossing points); a triple is kept once, attributed to the first corner it is found at.
    Returns a list of ``(n_facets, 3)`` arrays of start indices, -1 where a corner has no path.
    """
    n_starts = paths.valid.shape[0]
    pts = paths.points  # (S, C, m, 3)
    seen = [set() for _ in range(corners_of.shape[0])]
    out = []
    for j in range(3):
        for s in range(n_starts):
            here = paths.valid[s, corners_of[:, j]]
            slots = np.full((corners_of.shape[0], 3), -1)
            slots[:, j] = np.where(here, s, -1)
            for k in range(3):
                if k == j:
                    continue
                gap = np.max(
                    np.linalg.norm(
                        pts[:, corners_of[:, k]] - pts[s, corners_of[:, j]][None], axis=-1
                    ),
                    axis=-1,
                )  # (S, F)
                gap = np.where(paths.valid[:, corners_of[:, k]], gap, np.inf)
                best = np.argmin(gap, axis=0)
                slots[:, k] = np.where(here & np.isfinite(gap.min(axis=0)), best, -1)
            keep = np.zeros(corners_of.shape[0], bool)
            for f in np.flatnonzero(here):
                key = tuple(slots[f])
                if key not in seen[f]:
                    seen[f].add(key)
                    keep[f] = True
            out.append(np.where(keep[:, None], slots, -1))
    return out


started = time.perf_counter()
plan = refracted._plan(lamps, media, points, DEFAULT_PAIR_LIMIT)
rows_out = {}
for group in plan:
    chain = group.chain
    corners, shared = refracted._shared_corners(lamps, group.facets)
    normal = np.asarray(lamps.normal)[group.facets]
    emission = float(np.asarray(lamps.emission)[group.facets][0])
    ratio = float((media.index_of(chain.legs[-1]) / media.index_of(chain.legs[0])) ** 2)
    receivers = jnp.asarray(points[group.rows])
    paths = solve_paths(media, chain, corners[None], receivers[:, None])
    if not chain.passing:
        paths = jax.tree.map(lambda leaf: leaf[None], paths)
    paths = jax.tree.map(np.asarray, paths)
    n_starts = _path_count(media, chain)
    for r, row in enumerate(group.rows):
        at = jax.tree.map(lambda leaf, r=r: leaf[:, r], paths)
        receiver = points[row]
        by_start = sum(
            weights(
                at,
                shared,
                receiver,
                normal,
                emission,
                ratio,
                group.profile,
                np.full((len(group.facets), 3), s),
            ).sum()
            for s in range(n_starts)
        )
        if n_starts > 1:
            by_path = sum(
                weights(at, shared, receiver, normal, emission, ratio, group.profile, slots).sum()
                for slots in matched_slots(at, shared)
            )
        else:
            by_path = by_start
        # Triangles reached at one or two corners by one start, counted in proportion.
        straight = np.asarray(lamps.vertices)[group.facets]
        omega_straight = np.asarray(solid_angle(receiver[None], straight))
        partial = 0.0
        for s in range(n_starts):
            reached = at.valid[s][shared]
            count = reached.sum(axis=1)
            mixed = (count > 0) & (count < 3)
            rad = np.asarray(
                group.profile.radiance_per_exitance(at.departure[s][shared], normal[:, None])
            )
            carried = np.where(reached, rad * at.transmittance[s][shared], 0.0).sum(axis=1) / 3
            partial += np.sum(np.where(mixed, emission * ratio * carried * omega_straight, 0.0))
        key = (tuple(POINTS_MM[row]), label(chain))
        acc = rows_out.setdefault(key, [0.0, 0.0, 0.0])
        acc[0] += by_start
        acc[1] += by_path
        acc[2] += partial
print(f"solved in {time.perf_counter() - started:.0f} s", flush=True)
print("point | route | by start | by path | path/start | partial estimate | partial/by start")
for point in POINTS_MM:
    total = sum(v[0] for (p, _), v in rows_out.items() if p == tuple(point))
    for (p, name), (start, path, partial) in sorted(rows_out.items()):
        if p != tuple(point) or (start == 0.0 and partial == 0.0):
            continue
        print(
            f"{point} | {name} | {start:.6g} | {path:.6g} | "
            f"{path / start if start else float('nan'):.4f} | {partial:.4g} | "
            f"{partial / total:.4%} of the point's total",
            flush=True,
        )
    print(f"{point} | total by start {total:.6g}", flush=True)
