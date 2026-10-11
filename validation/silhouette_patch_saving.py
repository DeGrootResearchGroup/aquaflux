"""What would merging coplanar triangles into convex patches save the silhouette clip? (issue #474)

``SilhouetteOcclusion`` clips each (receiver, source) pair against every candidate blocker
TRIANGLE. A flat wall meshed as many triangles is, for occlusion, one convex blocker, so merging
coplanar, edge-connected triangles into convex polygons first would cut the clip's items. This
harness measures how many, on a real build, before anything is built:

1. **Where a build spends its time today** -- ``radiation_mask_build_cost.silhouette_stages`` on
   the box-plus-sleeve reactor ladder, so a saving in the clip can be priced against the build.
2. **The clip items a merge could remove.** One instrumented build per scene records exactly the
   (receiver, source, blocker) triples handed to the clip, and whether each covered anything. Each
   blocker is mapped to its patch (coplanar, edge-connected, same facing, convex tiling), and the
   triples are counted again as distinct (receiver, source, patch). That is a **lower bound on the
   merged clip's items** -- a best case for the merge: a patch's cone and second-stage tests are
   looser than its members', so a merged build would keep at least these.
3. **What one merged item costs.** A patch of ``k`` vertices is ``k`` edge-plane stages against a
   triangle's three, so the items are also reported weighted by vertex count.
4. **What ``overlapping`` would count**: pairs with more than one contributing triangle, against
   pairs with more than one contributing patch.

Scenes: the reactor ladder (box walls, flat, the case the merge is for); a sphere vessel (an
icosphere, no two facets coplanar -- the curved control); and a faceted tube vessel (a cylinder
meshed in strips along its length, which is how a curved vessel arrives from CAD: each strip is
flat). Each with the same lamp sleeve.

Run with ``validation/run_case.sh validation/silhouette_patch_saving.py``; set
``RADIATION_PATCH_SKIP_STAGES=1`` to skip the stage timing.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aquaflux  # noqa: F401  (enables x64)
import jax
import radiation_mask_build_cost as cost
from aquaflux.radiation.self_occlusion import SilhouetteOcclusion, _PairPipeline
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.visibility import build_visibility
from tests.unit.radiation_references import closed_drum

#: Two directions are one plane's normal when their dot product is within this of one, and a
#: facet lies in a plane when its centroid is within this share of the scene's extent of it.
COPLANAR = 1e-9


# ---------------------------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------------------------


def sleeve() -> np.ndarray:
    """The reactor ladder's lamp sleeve: a closed drum down the vessel's axis."""
    return closed_drum(16, radius=0.15, half_height=0.3) + np.array([0.5, 0.5, 0.5])


def icosphere(subdivisions: int) -> np.ndarray:
    """A sphere of radius 0.5 about (0.5, 0.5, 0.5), wound to face inward; no two facets coplanar."""
    t = (1.0 + 5.0**0.5) / 2.0
    points = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]  # fmt: skip
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    vertices = [np.array(p, dtype=float) / np.linalg.norm(p) for p in points]
    for _ in range(subdivisions):
        midpoints: dict[tuple[int, int], int] = {}

        def middle(a, b, midpoints=midpoints):
            key = (min(a, b), max(a, b))
            if key not in midpoints:
                m = vertices[a] + vertices[b]
                vertices.append(m / np.linalg.norm(m))
                midpoints[key] = len(vertices) - 1
            return midpoints[key]

        refined = []
        for a, b, c in faces:
            ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
            refined += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = refined
    table = np.array(vertices)
    # Outward as built; reversed to face the water inside.
    return 0.5 * table[np.array(faces)[:, ::-1]] + 0.5


def faceted_tube(sectors: int, slices: int) -> np.ndarray:
    """A closed cylinder vessel, radius 0.5 and height 1, meshed in flat strips, facing inward.

    One vertex table, so the seam closes. Each side strip is ``slices`` quads in one plane, and
    each cap a fan of ``sectors`` triangles in one plane -- the faceting a CAD reader gives a
    curved vessel.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, sectors, endpoint=False)
    ring = np.column_stack([0.5 + 0.5 * np.cos(angle), 0.5 + 0.5 * np.sin(angle)])
    heights = np.linspace(0.0, 1.0, slices + 1)
    faces = []
    for k in range(sectors):
        nxt = (k + 1) % sectors
        for s in range(slices):
            a = [*ring[k], heights[s]]
            b = [*ring[nxt], heights[s]]
            c = [*ring[nxt], heights[s + 1]]
            d = [*ring[k], heights[s + 1]]
            # Inward: reverse of the outward a, b, c / a, c, d.
            faces += [[a, c, b], [a, d, c]]
        low, high = [0.5, 0.5, 0.0], [0.5, 0.5, 1.0]
        faces.append([low, [*ring[k], 0.0], [*ring[nxt], 0.0]])
        faces.append([high, [*ring[nxt], 1.0], [*ring[k], 1.0]])
    return np.array(faces)


def with_sleeve(walls: np.ndarray) -> Surfaces:
    return Surfaces.from_triangles(np.concatenate([walls, sleeve()]))


# ---------------------------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------------------------


def _hull(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Indices of a 2D convex hull's corners, collinear points dropped (Andrew's monotone chain)."""
    order = np.lexsort((points[:, 1], points[:, 0]))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def chain(indices):
        kept: list[int] = []
        for i in indices:
            while (
                len(kept) >= 2 and cross(points[kept[-2]], points[kept[-1]], points[i]) <= tolerance
            ):
                kept.pop()
            kept.append(i)
        return kept

    lower, upper = chain(order), chain(order[::-1])
    return np.array(lower[:-1] + upper[:-1])


def coplanar_patches(surfaces: Surfaces):
    """Each facet's patch: coplanar, edge-connected, same facing, and a convex tiling.

    Returns ``(patch, corners, merged)``: the patch index of every facet ``(n,)``, each patch's
    corner count ``(n_patches,)`` (3 for an unmerged triangle), and whether each patch merged more
    than one triangle. A coplanar component whose triangles do not tile their convex hull is not
    convex and is left as triangles -- counted, so the report can say whether that ever happens.
    """
    vertices = np.asarray(surfaces.vertices, dtype=float)
    normal = np.asarray(surfaces.normal, dtype=float)
    centroid = np.asarray(surfaces.centroid, dtype=float)
    area = np.asarray(surfaces.area, dtype=float)
    n = len(vertices)
    extent = float(np.ptp(vertices.reshape(-1, 3), axis=0).max())

    keys = np.round(vertices.reshape(-1, 3) / (extent * 1e-12)).astype(np.int64)
    _, vertex_id = np.unique(keys, axis=0, return_inverse=True)
    vertex_id = vertex_id.reshape(n, 3)

    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges: dict[tuple[int, int], list[int]] = {}
    for f in range(n):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            u, v = vertex_id[f, a], vertex_id[f, b]
            edges.setdefault((min(u, v), max(u, v)), []).append(f)
    for faces in edges.values():
        if len(faces) != 2:
            continue
        i, j = faces
        same_plane = (
            float(normal[i] @ normal[j]) > 1.0 - COPLANAR
            and abs(float(normal[i] @ (centroid[j] - centroid[i]))) < COPLANAR * extent
        )
        if same_plane and area[i] > 0 and area[j] > 0:
            parent[find(i)] = find(j)

    roots = np.array([find(i) for i in range(n)])
    patch = np.full(n, -1)
    corners: list[int] = []
    merged: list[bool] = []
    non_convex = 0
    for root in np.unique(roots):
        members = np.flatnonzero(roots == root)
        if len(members) == 1:
            patch[members] = len(corners)
            corners.append(3)
            merged.append(False)
            continue
        n0 = normal[members[0]]
        e1 = np.cross(n0, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 0.5:
            e1 = np.cross(n0, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n0, e1)
        flat = vertices[members].reshape(-1, 3) @ np.column_stack([e1, e2])
        hull = _hull(flat, 1e-12 * extent**2)
        h = flat[hull]
        hull_area = 0.5 * abs(
            np.sum(h[:, 0] * np.roll(h[:, 1], -1) - np.roll(h[:, 0], -1) * h[:, 1])
        )
        if abs(hull_area - area[members].sum()) <= 1e-9 * hull_area:
            patch[members] = len(corners)
            corners.append(len(hull))
            merged.append(True)
        else:
            non_convex += 1
            for f in members:
                patch[f] = len(corners)
                corners.append(3)
                merged.append(False)
    return patch, np.array(corners), np.array(merged), non_convex


# ---------------------------------------------------------------------------------------------
# What reaches the clip
# ---------------------------------------------------------------------------------------------


def clipped_pairs(surfaces: Surfaces):
    """Run one default silhouette build and record every triple handed to the clip.

    Returns ``(rejected_input, row, source, blocker, hit, seconds)``: how many pairs reached the
    second stage, the clipped triples, whether each covered anything, and the build's wall clock.
    """
    original = _PairPipeline._run
    record = {"culled": 0, "rows": [], "sources": [], "blockers": [], "hits": []}

    def run(self, kernel, pairs):
        answers = original(self, kernel, pairs)
        if kernel is _PairPipeline._worth_clipping:
            record["culled"] += len(pairs)
        elif kernel is _PairPipeline._covered:
            record["rows"].append(np.array(pairs.row))
            record["sources"].append(np.array(pairs.source))
            record["blockers"].append(np.array(pairs.blocker))
            record["hits"].append(np.concatenate([np.asarray(h) for _, h in answers]))
        return answers

    _PairPipeline._run = run
    try:
        n = surfaces.n_facets
        start = time.perf_counter()
        jax.block_until_ready(
            build_visibility(
                (),
                surfaces,
                surfaces.centroid,
                receiver_facet=np.arange(n),
                self_occlusion=SilhouetteOcclusion(),
            ).hidden_by_geometry
        )
        seconds = time.perf_counter() - start
    finally:
        _PairPipeline._run = original
    joined = [
        np.concatenate(record[k]) if record[k] else np.zeros(0, dtype=int)
        for k in ("rows", "sources", "blockers", "hits")
    ]
    return record["culled"], *joined, seconds


def _contributors(row, source, key, hit, n):
    """Pairs with more than one distinct ``key`` among the triples that covered something."""
    triples = np.unique(np.stack([row[hit], source[hit], key[hit]]), axis=1)
    pair = triples[0].astype(np.int64) * n + triples[1]
    _, counts = np.unique(pair, return_counts=True)
    return int(np.count_nonzero(counts > 1)), len(counts)


def report(name: str, surfaces: Surfaces) -> None:
    n = surfaces.n_facets
    start = time.perf_counter()
    patch, corners, merged, non_convex = coplanar_patches(surfaces)
    patch_seconds = time.perf_counter() - start

    culled, row, source, blocker, hit, seconds = clipped_pairs(surfaces)
    hit = hit.astype(bool)
    items = len(row)
    patch_of = patch[blocker]
    unique = np.unique(np.stack([row, source, patch_of]).astype(np.int64), axis=1)
    merged_items = unique.shape[1]
    # Edge-plane stages: a triangle blocker is 3, a patch is its corner count.
    stages_before = 3 * items
    stages_after = int(corners[unique[2]].sum())
    on_merged = float(np.mean(merged[patch_of])) if items else 0.0
    many_tri, flagged_pairs = _contributors(row, source, blocker, hit, n)
    many_patch, _ = _contributors(row, source, patch_of, hit, n)

    print(
        f"{name:>18} {n:6,} | patches {len(corners):6,} (merged {int(merged.sum()):4,}, "
        f"non-convex {non_convex}, corners max {int(corners.max()):3}) {patch_seconds:6.2f} s | "
        f"build {seconds:8.2f} s",
        flush=True,
    )
    print(
        f"{'':>18} {'':6} | reject in {culled:13,}  clip items {items:12,} "
        f"({100 * items / max(1, culled):5.1f}%)  on merged patches {100 * on_merged:5.1f}%",
        flush=True,
    )
    print(
        f"{'':>18} {'':6} | merged items >= {merged_items:12,} "
        f"(x{items / max(1, merged_items):5.2f} fewer)  edge stages "
        f"{stages_before:13,} -> {stages_after:13,} (x{stages_before / max(1, stages_after):5.2f})",
        flush=True,
    )
    print(
        f"{'':>18} {'':6} | overlapping: {many_tri:10,} of {flagged_pairs:10,} covered pairs "
        f"by triangles -> {many_patch:10,} by patches",
        flush=True,
    )


if __name__ == "__main__":
    print(__doc__.split("Run with")[0].strip(), flush=True)
    print(f"\njax {jax.__version__}, {os.cpu_count()} cores\n", flush=True)

    ladder = [(4, 8), (6, 12), (8, 16), (11, 20), (14, 24), (16, 28)]

    print("### Clip items with and without coplanar patches\n", flush=True)
    for divisions, sectors in ladder:
        report(f"reactor d={divisions}", cost.reactor(divisions, sectors))
    for subdivisions in (2, 3):
        report(f"sphere s={subdivisions}", with_sleeve(icosphere(subdivisions)))
    for sectors, slices in ((24, 8), (32, 16)):
        report(f"tube {sectors}x{slices}", with_sleeve(faceted_tube(sectors, slices)))

    if not os.environ.get("RADIATION_PATCH_SKIP_STAGES"):
        cost.silhouette_stages(ladder[2:])
