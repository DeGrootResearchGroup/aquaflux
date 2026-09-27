"""How many of the pairs the shadow mask tests face away from their receiver, and could go untested?

A lamp facet whose outward normal points away from a receiver sends it nothing: the gather weights
the pair by the profile's radiance towards the receiver, which is exactly zero behind a facet for
every profile declared ``dark_behind``. So neither its shadow test nor its gather term can change
the field, and both could be skipped -- exactly, with no tolerance. The ray mask of the surface's
own triangles already skips such pairs (``Visibility.clear_behind``); the bodies' layer, which
``ShaftCulling`` decides, does not. This measures what skipping them there would buy, before
anything is built.

**Scene.** The Sozzi reactor's water as ``Outside(chamber, inlet, riser)``, the case's lamp STL if
the case is present and otherwise the analytic 32 x 128 lamp, ``ShaftCulling()`` at its default
ladder. Receivers are ``primitive_occlusion.receivers`` -- the case's cell centres when the case is
present, else points uniform in the three cylinders (the population ``body_culling.py`` records) --
**less any inside the lamp**, which the vessel body does not exclude.

**What is reported**, all as shares of pairs:

- *facing away, of all pairs*: what the gather could skip;
- *undecided*: pairs in the finest tiles the certificates could not vouch for -- what is tested;
- of those, *facing away* (the ceiling for a per-pair skip), *in finest tiles wholly facing away*
  (what a per-tile skip decides exactly), and *in tiles a bounding-box test proves face away*
  (a conservative per-tile test from the tile's receivers' box and each facet's plane, which is
  what an implementation can afford to decide before forming any pair).

"Facing away" is the gather's own gate: ``(receiver - centroid) . normal <= 0``.

**Then what the built skip realizes**: the bodies' layer built by ``ShaftCulling()`` with and
without the facets' planes (``BackFaces``), in one process on the same rays, a warm-up and then
two alternating passes, fastest kept; the masks must agree on every pair that faces its receiver
and the skipped one must be clear on every pair that does not.

Run with ``validation/run_case.sh validation/sozzi_radiation/backface_share.py``.
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
from aquaflux.radiation import ShaftCulling  # noqa: E402
from aquaflux.radiation.back_faces import BackFaces  # noqa: E402
from aquaflux.radiation.culling import _Curve  # noqa: E402
from primitive_occlusion import OUT, fluid, lamp, receivers  # noqa: E402


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def outside_lamp(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """The points not inside the lamp's circumscribed cylinder (its axis along x, through 0)."""
    radius = float(np.hypot(vertices[..., 1], vertices[..., 2]).max())
    low, high = float(vertices[..., 0].min()), float(vertices[..., 0].max())
    inside = (np.hypot(points[:, 1], points[:, 2]) <= radius) & (
        (points[:, 0] >= low) & (points[:, 0] <= high)
    )
    return points[~inside]


def main() -> None:
    rng = np.random.default_rng(0)
    water = fluid()
    surfaces, lamp_source = lamp()
    vertices = np.asarray(surfaces.vertices)
    sampled, population = receivers(rng, water)
    points = outside_lamp(sampled, vertices)
    centroid = np.asarray(surfaces.centroid)
    normal = np.asarray(surfaces.normal)
    strategy = ShaftCulling()
    _say(
        f"{len(points)} receivers ({population}; {len(sampled) - len(points)} inside the lamp "
        f"removed), {surfaces.n_facets} lamp facets ({lamp_source}), ShaftCulling"
        f"{strategy.receiver_blocks}x{strategy.source_clusters}; jax {jax.__version__}, "
        f"{platform.system()} {platform.machine()}, {os.cpu_count()} cores"
    )
    blocks = _Curve.of(points, strategy.receiver_blocks[0])
    clusters = _Curve.of(centroid, strategy.source_clusters[0])
    block, cluster = strategy.receiver_blocks[-1], strategy.source_clusters[-1]
    started = time.perf_counter()
    rows, cols = strategy._undecided(water, centroid, points, blocks, clusters)
    _say(f"{len(rows)} undecided finest tiles, found in {time.perf_counter() - started:.1f} s")
    row_members = blocks.members(block)[rows]  # (n_tiles, block) receiver indices, padded
    col_members = clusters.members(cluster)[cols]  # (n_tiles, cluster) facet indices, padded
    row_real = np.stack([blocks.counts(block)[rows] > k for k in range(block)], axis=1)
    col_real = np.stack([clusters.counts(cluster)[cols] > k for k in range(cluster)], axis=1)

    facing_all = 0
    for start in range(0, len(points), 2000):
        chunk = points[start : start + 2000]
        facing_all += int(
            (np.einsum("rfk,fk->rf", chunk[:, None, :] - centroid[None], normal) > 0.0).sum()
        )
    all_pairs = len(points) * surfaces.n_facets

    undecided = facing = whole_tile_away = box_away = 0
    for start in range(0, len(rows), 200_000):
        r = row_members[start : start + 200_000]
        c = col_members[start : start + 200_000]
        real = row_real[start : start + 200_000, :, None] & col_real[start : start + 200_000, None]
        height = np.einsum(
            "trfk,tfk->trf", points[r][:, :, None, :] - centroid[c][:, None, :, :], normal[c]
        )
        faces = (height > 0.0) & real
        undecided += int(real.sum())
        facing += int(faces.sum())
        away_tile = ~faces.any(axis=(1, 2))
        whole_tile_away += int(real[away_tile].sum())
        # The conservative test: the largest height over the receivers' bounding box, per facet.
        low = np.where(real.any(axis=2)[..., None], points[r], np.inf).min(axis=1)
        high = np.where(real.any(axis=2)[..., None], points[r], -np.inf).max(axis=1)
        middle, half = 0.5 * (low + high), 0.5 * (high - low)
        n = normal[c]
        largest = np.einsum("tfk,tfk->tf", middle[:, None, :] - centroid[c], n) + np.einsum(
            "tk,tfk->tf", half, np.abs(n)
        )
        proven = ~((largest > 0.0) & col_real[start : start + 200_000]).any(axis=1)
        box_away += int(real[proven].sum())
    summary = {
        "receivers": len(points),
        "facets": surfaces.n_facets,
        "facing_away_of_all_pairs": 1.0 - facing_all / all_pairs,
        "undecided_of_all_pairs": undecided / all_pairs,
        "of_undecided": {
            "facing_away": 1.0 - facing / undecided,
            "in_finest_tiles_wholly_facing_away": whole_tile_away / undecided,
            "in_finest_tiles_a_box_test_proves_face_away": box_away / undecided,
        },
    }
    _say(
        f"facing away, of all pairs: {summary['facing_away_of_all_pairs']:.3f}; undecided (tested) "
        f"{summary['undecided_of_all_pairs']:.3f} of all pairs"
    )
    for label, share in summary["of_undecided"].items():
        _say(f"of the undecided pairs, {label.replace('_', ' ')}: {share:.3f}")
    summary["built"] = _time_the_skip(water, surfaces, points, strategy)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "backface_share.json").write_text(json.dumps(summary, indent=2))
    _say(f"wrote {OUT / 'backface_share.json'}")


def _time_the_skip(water, surfaces, points, strategy, passes: int = 2) -> dict:
    """The bodies' layer with and without the facets' planes: wall clock, and the masks agree."""
    sources = np.asarray(surfaces.centroid)
    near = 1e-6 * np.sqrt(np.asarray(surfaces.area))
    facing = BackFaces.of(surfaces)
    arms = {"every pair asked": None, "pairs behind skipped": facing}
    masks, seconds = {}, {name: [] for name in arms}
    for index in range(passes + 1):
        for name, given in arms.items():
            started = time.perf_counter()
            masks[name] = np.asarray(
                strategy.blocked([water], sources, near, points, facing=given)
            )[0]
            elapsed = time.perf_counter() - started
            if index:
                seconds[name].append(elapsed)
            _say(f"{'warm-up' if index == 0 else f'pass {index}'}, {name}: {elapsed:.2f} s")
    behind = np.concatenate(
        [np.asarray(facing.every_pair(points[start : start + 2000])) for start in
         range(0, len(points), 2000)]
    )  # fmt: skip
    full, skipped = masks["every pair asked"], masks["pairs behind skipped"]
    agree = bool(np.array_equal(skipped, full & ~behind))
    fastest = {name: min(values) for name, values in seconds.items()}
    spread = {name: max(values) / min(values) for name, values in seconds.items()}
    ratio = fastest["every pair asked"] / fastest["pairs behind skipped"]
    _say(
        f"every pair asked {fastest['every pair asked']:.2f} s (spread "
        f"{spread['every pair asked']:.2f}x), pairs behind skipped "
        f"{fastest['pairs behind skipped']:.2f} s (spread {spread['pairs behind skipped']:.2f}x): "
        f"{ratio:.2f}x; blocked pairs behind their source in the full mask "
        f"{int((full & behind).sum())}; masks agree where it matters: {agree}"
    )
    if not agree:
        raise SystemExit("the skipped mask differs from the full one on a pair facing its receiver")
    return {"fastest_s": fastest, "spread": spread, "speed_up": ratio, "agree": agree}


if __name__ == "__main__":
    main()
