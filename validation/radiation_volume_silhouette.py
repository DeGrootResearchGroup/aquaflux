"""The silhouette clip for points in the water: how much it fixes the fluence rate, and its cost.

A cell in the water weights each source by its plain solid angle, and ``SilhouetteOcclusion`` now
takes its covered share of that measure, so the fluence rate can see a sleeve's shadow edge as the
fraction it is rather than as all or nothing per (cell, facet) pair. This measures what that buys
and what it costs, on the two-sleeve reactor of ``radiation_overlap_overcount.py`` -- the fixture
where the clip's one known defect, adding the shares of two overlapping silhouettes, can bite.

1. **The fluence rate near shadow edges**, the sleeves emitting and the walls black so that only
   the volume mask is being judged. Three fields at each cell: the clip's, the one-ray mask's, and
   a reference with every evaluated pair's hidden share replaced by a sampled *union* over the
   source -- a different algorithm, which cannot double count. Each goes through the public model
   (``fluence_rate``), the reference by injecting its hidden shares as a precomputed strategy, so
   all three are the same gather over three masks.
2. **The cost of the volume mask**, clip against ray test, at two facet counts and two receiver
   counts, with the cone cull's survival per receiver -- the quantity that decides whether the
   clip is usable on a mesh's cells.

Run with ``validation/run_case.sh validation/radiation_volume_silhouette.py``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
from aquaflux.radiation import (
    RadiationSettings,
    RayCastOcclusion,
    SilhouetteOcclusion,
    build_radiation_model,
    build_visibility,
    fluence_rate,
)
from radiation_overlap_overcount import PrecomputedOcclusion, two_sleeve_reactor, union_hidden

#: Sleeve axes, radius and half-height, as `two_sleeve_reactor` builds them.
SLEEVES_X = (0.35, 0.65)
RADIUS, HALF_HEIGHT = 0.1, 0.3

#: Edge cells whose pairs are referenced, and the random control pairs neither mask hides.
EDGE_CELLS = 300
CONTROLS = 2000

#: A share strictly inside this band marks a partly hidden pair.
PARTIAL = (0.05, 0.95)


def water_points(per_side: int) -> np.ndarray:
    """A lattice of cell-like points in the box's water, clear of the walls and both sleeves."""
    axis = (np.arange(per_side) + 0.5) / per_side
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    clear = np.ones(len(points), dtype=bool)
    for x in SLEEVES_X:
        radial = np.hypot(points[:, 0] - x, points[:, 1] - 0.5)
        inside = (radial < RADIUS + 0.02) & (np.abs(points[:, 2] - 0.5) < HALF_HEIGHT + 0.02)
        clear &= ~inside
    return points[clear]


def lit_by_sleeves(surfaces):
    """The reactor with its walls black, so the only light is the sleeves' direct emission."""
    return surfaces.with_optics(reflectance=jnp.zeros(surfaces.n_facets))


def field_of(surfaces, points, strategy):
    """The fluence rate and the volume mask's hidden shares, through the public model."""
    model = build_radiation_model(
        points,
        surfaces,
        settings=RadiationSettings(self_occlusion=RayCastOcclusion(), receiver_occlusion=strategy),
    )
    field, _ = fluence_rate(model, surfaces)
    return np.asarray(field), np.asarray(model.receiver_shadows.visibility.hidden_by_geometry)


def accuracy(divisions: int, sectors: int, per_side: int, seed: int = 0) -> None:
    """Section 1: the three fields at the cells beside a shadow edge."""
    surfaces = lit_by_sleeves(two_sleeve_reactor(divisions, sectors))
    points = water_points(per_side)
    sleeve = np.asarray(surfaces.emission) > 0.0
    print(
        f"\n## fluence rate, two-sleeve reactor divisions={divisions} sectors={sectors}: "
        f"{surfaces.n_facets} facets, {len(points):,} water points\n",
        flush=True,
    )

    start = time.perf_counter()
    clip_field, clip = field_of(surfaces, points, SilhouetteOcclusion())
    print(f"silhouette model and field: {time.perf_counter() - start:.1f} s", flush=True)
    start = time.perf_counter()
    ray_field, ray = field_of(surfaces, points, RayCastOcclusion())
    print(f"ray-mask model and field:   {time.perf_counter() - start:.1f} s", flush=True)

    partial = (clip > PARTIAL[0]) & (clip < PARTIAL[1]) & sleeve[None, :]
    edge = np.nonzero(partial.any(axis=1))[0]
    rng = np.random.default_rng(seed)
    if len(edge) > EDGE_CELLS:
        edge = np.sort(rng.choice(edge, EDGE_CELLS, replace=False))
    print(
        f"cells with a partly hidden sleeve facet: {int(partial.any(axis=1).sum()):,} of "
        f"{len(points):,}; referencing {len(edge)}",
        flush=True,
    )

    # Every sleeve pair either mask hides at all, at the edge cells, plus a control of pairs
    # neither hides -- where a shadow the clip missed would show.
    hidden = ((clip[edge] > 1e-9) | (ray[edge] > 0.0)) & sleeve[None, :]
    cells, sources = np.nonzero(hidden)
    clear_cells, clear_sources = np.nonzero(~hidden & sleeve[None, :])
    pick = rng.choice(len(clear_cells), min(CONTROLS, len(clear_cells)), replace=False)
    start = time.perf_counter()
    reference, error = union_hidden(
        surfaces, points[edge][cells], None, sources, sources, seed=seed + 1
    )
    control, control_error = union_hidden(
        surfaces,
        points[edge][clear_cells[pick]],
        None,
        clear_sources[pick],
        clear_sources[pick],
        seed=seed + 2,
    )
    print(
        f"union reference for {len(cells):,} hidden pairs and {len(pick):,} controls: "
        f"{time.perf_counter() - start:.1f} s",
        flush=True,
    )
    missed = control > 3.0 * control_error + 1e-3
    print(
        f"controls the reference finds hidden (a shadow both masks missed): "
        f"{int(missed.sum())} of {len(pick)}, worst {control.max():.4f}",
        flush=True,
    )

    gap = clip[edge][cells, sources] - reference
    over = gap > 3.0 * error + 1e-3
    under = gap < -(3.0 * error + 1e-3)
    print(
        f"\nper pair, clip minus reference: {int(over.sum())} over, {int(under.sum())} under, of "
        f"{len(cells):,}; worst over {gap.max():.4f}, worst under {gap.min():.4f}",
        flush=True,
    )
    ray_gap = ray[edge][cells, sources] - reference
    print(
        f"per pair, ray minus reference:  mean |gap| {np.abs(ray_gap).mean():.4f}, "
        f"worst {np.abs(ray_gap).max():.4f}   (clip: mean |gap| {np.abs(gap).mean():.4f})",
        flush=True,
    )

    corrected = clip[edge].copy()
    corrected[cells, sources] = reference
    # A host array rather than a device one: a strategy is a static setting, and JAX warns about
    # an array of its own held static.
    reference_field, _ = field_of(surfaces, points[edge], PrecomputedOcclusion(corrected))
    print("\n### fluence rate at the edge cells, against the reference\n", flush=True)
    print(f"{'mask':>12} {'mean |rel|':>11} {'p95 |rel|':>10} {'worst |rel|':>12}", flush=True)
    for name, value in (("silhouette", clip_field[edge]), ("ray mask", ray_field[edge])):
        relative = np.abs(value - reference_field) / reference_field
        print(
            f"{name:>12} {relative.mean():11.4f} {np.percentile(relative, 95):10.4f} "
            f"{relative.max():12.4f}",
            flush=True,
        )
    everywhere = np.abs(ray_field - clip_field) / clip_field
    print(
        f"\nray mask against the clip over all {len(points):,} points: mean |rel| "
        f"{everywhere.mean():.4f}, worst {everywhere.max():.4f}",
        flush=True,
    )


def cost(divisions: int, sectors: int, per_side: int) -> None:
    """Section 2: the volume mask alone, clip against ray test, and the cull's survival."""
    surfaces = two_sleeve_reactor(divisions, sectors)
    points = jnp.asarray(water_points(per_side))
    n = int(surfaces.n_facets)
    timings = {}
    for name, strategy in (("silhouette", SilhouetteOcclusion()), ("ray mask", RayCastOcclusion())):
        # Warm once on a handful of points, so the figure is the build and not its compilation.
        build_visibility((), surfaces, points[:4], self_occlusion=strategy)
        start = time.perf_counter()
        build_visibility((), surfaces, points, self_occlusion=strategy)
        timings[name] = time.perf_counter() - start

    near = jnp.zeros(n)
    survivors = [
        len(
            SilhouetteOcclusion()._candidates(
                points[k],
                None,
                surfaces.vertices,
                surfaces.centroid,
                surfaces.normal,
                near,
                jnp.zeros(n, dtype=bool),
            )[1]
        )
        for k in range(0, len(points), max(1, len(points) // 50))
    ]
    print(
        f"{n:>7} {len(points):>9,} {timings['silhouette']:10.1f} {timings['ray mask']:9.1f} "
        f"{1e3 * timings['silhouette'] / len(points):12.2f} "
        f"{timings['silhouette'] / timings['ray mask']:7.1f} "
        f"{np.mean(survivors):12,.0f} {100.0 * np.mean(survivors) / n**2:9.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    print(__doc__.split("Run with")[0].strip(), flush=True)
    print(
        "\nConfiguration: union reference 4096 samples per pair, plain solid-angle weighting; a "
        "gap is real past 3 standard errors + 1e-3. Warm timings, one process.",
        flush=True,
    )
    accuracy(4, 12, per_side=14)
    print("\n## cost of the volume mask alone\n", flush=True)
    print(
        f"{'facets':>7} {'receivers':>9} {'clip s':>10} {'ray s':>9} {'clip ms/rcv':>12} "
        f"{'ratio':>7} {'survivors':>12} {'of n^2':>10}",
        flush=True,
    )
    for divisions, sectors in ((4, 12), (6, 16)):
        for per_side in (8, 12):
            cost(divisions, sectors, per_side)
