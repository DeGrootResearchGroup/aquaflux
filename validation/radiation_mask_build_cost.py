"""How many seconds does the occlusion mask actually cost to build, and what would an exact one cost?

The subsystem record prices analytic silhouette occlusion at *fifteen to twenty times the mask
build*. That is a ratio, and the decision it is meant to inform -- build it or document the
binary mask as a limitation -- is a question about **minutes**. This harness supplies the
denominator in seconds, so the ratio can be multiplied out into a build time rather than argued
about in the abstract.

Four things it establishes:

1. **The absolute cost of the shipped build**, on a closed reactor-shaped scene, at facet counts
   spanning the range a user would actually mesh.
2. **That the cost is geometry-independent**, which is why one ladder of numbers settles it for
   every scene of that size. ``segment_is_cut`` tests every ray against every triangle with no
   early exit, so the self-occlusion pass performs exactly ``n_receivers x n_facets x
   n_triangles`` intersection tests whatever the geometry is doing. The harness checks this
   rather than asserting it, by timing two scenes of equal facet count and very different
   occlusion structure.
3. **What ``work_limit`` is worth now that the pass is traced.** The bound caps rays times
   triangles per compiled call, and the call takes triangles first, so a larger bound means a
   larger ray chunk per call. Two middle rungs are timed at a much larger bound to show which way
   that moves the build.
4. **What the analytic strategy actually costs**, measured rather than projected:
   ``silhouette_ladder`` times ``SilhouetteOcclusion`` beside ``RayCastOcclusion`` on the same scenes
   in the same run. This supersedes the ratio-times-seconds columns of the ray ladder, which apply
   a band taken at a far smaller size and are kept only to show why that is unsafe.

⚠️ **The cubic is the whole story and it is worth seeing written out.** Receivers are facet
centroids for a transfer build, so ``n_receivers == n_facets == n_triangles`` and the pass is
``n^3``. Doubling the mesh costs eight times the build. Any judgement about affordability that
is made at one facet count transfers to another only through that cube.

Run with ``validation/run_case.sh validation/radiation_mask_build_cost.py``; set
``RADIATION_SILHOUETTE_ONLY=1`` to run only the silhouette's two sections.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
from aquaflux.radiation.clusters import FacetClusters
from aquaflux.radiation.self_occlusion import (
    RayCastOcclusion,
    SilhouetteOcclusion,
    _PairPipeline,
)
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.triangles import padded_length, segment_is_cut
from aquaflux.radiation.visibility import build_visibility
from tests.unit.radiation_references import closed_drum, inward_box

#: The ratio band on record for analytic silhouette occlusion: clip throughput against ray-test
#: throughput, times the surviving fraction of the conservative frustum reject.
#:
#: ⚠️ **Both of its throughputs were measured at 200,000 work items**, four orders below a real
#: build, and on a ray test whose call shape has since changed. Applying this band to the seconds
#: measured here is an extrapolation, not a projection, and the columns that do so are labelled as
#: such. The measured ratio is ``silhouette_ladder``'s, and that is the one to quote.
ANALYTIC_RATIO = (15.0, 20.0)


def reactor(divisions: int, sectors: int, radius: float = 0.15) -> Surfaces:
    """A closed box with a lamp sleeve down its axis -- one self-occluding surface set.

    This is the shape the package exists for: an enclosure whose own walls shadow each other
    around an internal body. The sleeve is built from one vertex table so its seam actually
    closes.
    """
    walls = inward_box(divisions)
    sleeve = closed_drum(sectors, radius=radius, half_height=0.3) + np.array([0.5, 0.5, 0.5])
    return Surfaces.from_triangles(np.concatenate([walls, sleeve]))


def empty_box(divisions: int, sectors: int, radius: float = 0.15) -> Surfaces:
    """The same facet count with nothing in the middle -- the geometry-independence control.

    The sleeve's triangles are moved out beyond the box rather than deleted, so the two scenes
    carry the identical number of rays and the identical number of triangles and differ only in
    what the rays *hit*. If the build is geometry-independent these time the same.
    """
    walls = inward_box(divisions)
    sleeve = closed_drum(sectors, radius=radius, half_height=0.3) + np.array([9.0, 9.0, 9.0])
    return Surfaces.from_triangles(np.concatenate([walls, sleeve]))


def timed(build, repeats: int = 3) -> float:
    """Median wall clock of a warmed call.

    The inner kernel is jitted per distinct block shape, so the first call pays compilation that
    no later one does; it is run and discarded. Wall clock on a shared desktop carries ~20%
    spread, so the median of a few is the honest reading and a single run is not.
    """
    jax.block_until_ready(build().hidden_by_geometry)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(build().hidden_by_geometry)
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def ladder(cases, work_limit: int) -> None:
    """Time the shipped build across a range of facet counts at one `work_limit`."""
    print(
        f"\n### build_visibility, self_occlusion=RayCastOcclusion(), work_limit={work_limit:,}\n",
        flush=True,
    )
    print(
        f"{'facets':>7} {'tests':>15} {'seconds':>9} {'Mtest/s':>9} "
        f"{'x15 (see note)':>15} {'x20':>10}",
        flush=True,
    )
    for divisions, sectors in cases:
        surfaces = reactor(divisions, sectors)
        n = surfaces.n_facets
        centroid = surfaces.centroid
        facet = np.arange(n)
        # The cube means the top rungs dominate this harness's own wall clock; two samples
        # there and three below keeps it under half an hour without thinning the range.
        seconds = timed(
            lambda s=surfaces, c=centroid, f=facet: build_visibility(
                (),
                s,
                c,
                receiver_facet=f,
                self_occlusion=RayCastOcclusion(work_limit=work_limit),
            ),
            repeats=2 if n > 2000 else 3,
        )
        tests = n**3
        print(
            f"{n:7,} {tests:15,} {seconds:9.2f} {tests / seconds / 1e6:9.1f} "
            f"{_minutes(seconds * ANALYTIC_RATIO[0]):>14} "
            f"{_minutes(seconds * ANALYTIC_RATIO[1]):>10}",
            flush=True,
        )

    print(
        "\n⚠️ The last two columns apply a ratio measured at 200,000 work items to seconds\n"
        "   measured here; the Mtest/s column is what shows why that is not safe. Read them as\n"
        "   an upper bound pending the scale-matched ratio, not as a projection.",
        flush=True,
    )


def silhouette_ladder(cases) -> None:
    """Time the analytic strategy on the same scenes, beside the ray mask it would replace.

    The ratio the rest of this harness can only extrapolate, measured directly: both builds,
    same reactor, same receivers, same machine, each the median of warmed runs. The clip's cost
    is not a fixed multiple of ``n**3`` -- its cone cull discards a share of the candidate pairs
    that grows as the mesh refines -- so the ratio is reported per rung rather than as one band.
    """
    print("\n### build_visibility: SilhouetteOcclusion() against RayCastOcclusion()\n", flush=True)
    print(f"{'facets':>7} {'ray s':>9} {'clip s':>9} {'clip/ray':>9} {'clip':>9}", flush=True)
    for divisions, sectors in cases:
        surfaces = reactor(divisions, sectors)
        n = surfaces.n_facets
        centroid, facet = surfaces.centroid, np.arange(n)
        repeats = 2 if n > 2000 else 3
        times = [
            timed(
                lambda s=surfaces, c=centroid, f=facet, st=strategy: build_visibility(
                    (), s, c, receiver_facet=f, self_occlusion=st
                ),
                repeats=repeats,
            )
            for strategy in (RayCastOcclusion(), SilhouetteOcclusion())
        ]
        print(
            f"{n:7,} {times[0]:9.2f} {times[1]:9.2f} {times[1] / times[0]:9.2f} "
            f"{_minutes(times[1]):>9}",
            flush=True,
        )


def second_stage(cases, sampled: int = 30) -> None:
    """What the silhouette's second-stage rejection removes, and that it removes nothing real.

    For a sample of receivers on each rung: the pairs the cone cull keeps, how many of those the
    second stage (``covers_nothing``) rejects, and -- clipping the rejected pairs anyway -- the
    largest share any of them covers and the largest per-source sum, both of which must sit at
    the clip's own floor. The unit tests sweep this exhaustively on small reactors; this is the
    same check at the sizes a build actually runs.
    """
    print("\n### SilhouetteOcclusion: the second stage on the cone cull's survivors\n", flush=True)
    print(
        f"{'facets':>7} {'receivers':>9} {'cull kept':>12} {'rejected':>9} "
        f"{'worst rejected':>15} {'worst source sum':>17}",
        flush=True,
    )
    strategy = SilhouetteOcclusion()
    for divisions, sectors in cases:
        surfaces = reactor(divisions, sectors)
        n = surfaces.n_facets
        vertices = jnp.asarray(surfaces.vertices)
        centroid = jnp.asarray(surfaces.centroid)
        normal = np.asarray(surfaces.normal)
        either_side = jnp.zeros(n, dtype=bool)
        clusters = FacetClusters.build(np.asarray(surfaces.vertices), strategy.cluster_size)
        kept_total, rejected_total, worst, worst_sum = 0, 0, 0.0, 0.0
        rows = range(0, n, max(1, n // sampled))
        for row in rows:
            view, source, blocker = strategy._candidates(
                np.asarray(centroid[row]),
                normal[row],
                vertices,
                centroid,
                jnp.asarray(normal),
                jnp.zeros(n),
                either_side,
                clusters,
            )
            legal = (source != blocker) & (source != row) & (blocker != row)
            source, blocker = source[legal], blocker[legal]
            # Padded to a power of two, repeating the last pair, so a handful of compiled shapes
            # serve every receiver -- one program per distinct pair count exhausts memory -- and
            # the padding's answers are cut off.
            count = len(source)
            index = np.minimum(np.arange(padded_length(count)), count - 1)
            receiver = np.broadcast_to(np.asarray(centroid[row]), (len(index), 3))
            facing = np.broadcast_to(normal[row], (len(index), 3))
            pair_view = view.take(source[index])
            worth = np.asarray(
                _PairPipeline._worth_clipping(
                    vertices, receiver, facing, pair_view, source[index], blocker[index]
                )
            )[:count]
            covered = np.asarray(
                _PairPipeline._covered(
                    vertices, receiver, facing, pair_view, source[index], blocker[index]
                )[0]
            )[:count]
            rejected = np.where(worth, 0.0, covered)
            per_source = np.zeros(n)
            np.add.at(per_source, source, rejected)
            kept_total += len(source)
            rejected_total += int(np.count_nonzero(~worth))
            worst = max(worst, float(rejected.max(initial=0.0)))
            worst_sum = max(worst_sum, float(per_source.max()))
        print(
            f"{n:7,} {len(rows):9,} {kept_total / len(rows):12,.0f} "
            f"{100.0 * rejected_total / max(1, kept_total):8.1f}% {worst:15.2e} {worst_sum:17.2e}",
            flush=True,
        )


def silhouette_stages(cases, threads: int = 1) -> None:
    """Where one silhouette build spends its time: cull, reject, clip, and the host work around them.

    Each compiled stage is wrapped to block and time itself, and the host work is what is left of
    the Python around it. With ``threads`` above one the stages overlap, so the seconds are summed
    over threads and are shares of the work, not of the wall clock. The wrappers are removed
    afterwards.
    """
    import threading
    from collections import defaultdict

    spent = defaultdict(float)
    lock = threading.Lock()

    def timed(name, fn, block=True):
        def run(*args, **kwargs):
            start = time.perf_counter()
            out = fn(*args, **kwargs)
            if block:
                out = jax.block_until_ready(out)
            with lock:
                spent[name] += time.perf_counter() - start
            return out

        return run

    wrapped = {
        (SilhouetteOcclusion, "_per_triangle"): "per triangle",
        (SilhouetteOcclusion, "_cluster_bounds"): "cull (device)",
        (SilhouetteOcclusion, "_cluster_pairs"): "cull (device)",
        (SilhouetteOcclusion, "_member_pairs"): "cull (device)",
        (_PairPipeline, "_worth_clipping"): "reject (device)",
        (_PairPipeline, "_covered"): "clip (device)",
    }
    totals = {
        (SilhouetteOcclusion, "_candidates"): "cull total",
        (_PairPipeline, "_call"): "chunks",
    }
    saved = {key: key[0].__dict__[key[1]] for key in (*wrapped, *totals)}
    try:
        for (owner, name), label in wrapped.items():
            setattr(owner, name, staticmethod(timed(label, getattr(owner, name))))
        for (owner, name), label in totals.items():
            setattr(owner, name, timed(label, getattr(owner, name), block=False))
        print(f"\n### SilhouetteOcclusion stages, threads={threads}\n", flush=True)
        for divisions, sectors in cases:
            surfaces = reactor(divisions, sectors)
            n = surfaces.n_facets
            strategy = SilhouetteOcclusion(threads=threads)
            facets = np.arange(n)

            def build(s=surfaces, st=strategy, f=facets):
                return build_visibility((), s, s.centroid, receiver_facet=f, self_occlusion=st)

            jax.block_until_ready(build().hidden_by_geometry)
            spent.clear()
            start = time.perf_counter()
            jax.block_until_ready(build().hidden_by_geometry)
            wall = time.perf_counter() - start
            row = dict(spent)
            cull_total, chunks = row.pop("cull total"), row.pop("chunks")
            row["cull (host)"] = cull_total - row["per triangle"] - row["cull (device)"]
            row["chunks (host)"] = chunks - row["reject (device)"] - row["clip (device)"]
            summed = sum(row.values())
            shares = "  ".join(f"{k} {100 * v / summed:.1f}%" for k, v in row.items())
            print(f"{n:7,} facets: wall {wall:8.2f} s  |  {shares}", flush=True)
    finally:
        for (owner, name), original in saved.items():
            setattr(owner, name, original)


def _minutes(seconds: float) -> str:
    """Seconds as the unit a build-time decision is actually made in."""
    if seconds < 90.0:
        return f"{seconds:.0f} s"
    if seconds < 5400.0:
        return f"{seconds / 60.0:.0f} min"
    return f"{seconds / 3600.0:.1f} h"


def geometry_independence(divisions: int, sectors: int, work_limit: int) -> None:
    """Does what the rays hit change what the pass costs? It should not, and this is the check.

    A pass with an early exit would run faster on the scene where more rays are blocked. This one
    has none, so the two arms should agree inside the ~20% spread wall clock carries here -- and
    if they ever stop agreeing, the cost model in this file's docstring is the thing that broke.
    """
    print(f"\n### geometry independence, work_limit={work_limit:,}\n", flush=True)
    print(f"{'scene':>12} {'facets':>7} {'blocked pairs':>14} {'seconds':>9}", flush=True)
    timings = {}
    for name, builder in (("sleeve", reactor), ("empty", empty_box)):
        surfaces = builder(divisions, sectors)
        n = surfaces.n_facets
        centroid = surfaces.centroid
        facet = np.arange(n)

        def build(s=surfaces, c=centroid, f=facet):
            return build_visibility(
                (),
                s,
                c,
                receiver_facet=f,
                self_occlusion=RayCastOcclusion(work_limit=work_limit),
            )

        blocked = int(np.count_nonzero(np.asarray(build().hidden_by_geometry)))
        timings[name] = timed(build)
        print(f"{name:>12} {n:7,} {blocked:14,} {timings[name]:9.2f}", flush=True)
    ratio = max(timings.values()) / min(timings.values())
    verdict = "geometry-independent" if ratio < 1.25 else "⚠️ GEOMETRY-DEPENDENT"
    print(f"\nslower / faster = {ratio:.2f}x  -- {verdict}", flush=True)


def throughput_against_ray_count(n_triangles: int, work_limit: int) -> None:
    """Does the pass slow down because there is more work, or because there are more RAYS?

    Written when the ladder showed throughput falling steeply as the mesh grew. Two explanations
    fit the ladder equally well and they have opposite consequences for what an exact mask would
    cost, so they have to be told apart rather than guessed between.

    **Answered, and by neither of the two below:** the sweep did fall with the ray count, and the
    cause was ``segment_is_cut`` giving the rays the whole call budget and leaving a one-triangle
    block. With triangles first the sweep is flat -- which is what this now checks stays true.

    * **Total work.** Some fixed overhead is being amortized differently, or the cache is
      spilling as the output grows. Nothing to be done; the cost is the cost.
    * **Ray-array traffic.** ``build_visibility`` hands ``segment_is_cut`` a *flattened outer
      product*: ``broadcast_to(origin, (rays, n_facets, 3)).reshape(...)`` materializes one
      origin and one target per (receiver, facet) pair, so at ``n`` facets the pass writes and
      re-reads ``O(n**2)`` positions that were only ``O(n)`` distinct values. That is avoidable
      work, and if it dominates then the mask build is carrying a factor that an exact
      treatment would not have to match.

    Holding the triangle count fixed and sweeping the ray count separates them: on the first
    explanation throughput is flat in the ray count at fixed total work per call; on the second
    it falls as the rays grow. The test count per call is held constant across the sweep, so
    neither arm is given more work than the other.
    """
    print(f"\n### throughput against ray count, {n_triangles:,} triangles\n", flush=True)
    print(
        "Test count per call is held fixed, so only the SHAPE of the work changes. Flat means\n"
        "the cost is the tests; falling with the ray count means it is the ray arrays.\n",
        flush=True,
    )
    rng = np.random.default_rng(11)
    triangles = jnp.asarray(rng.uniform(-1, 1, (n_triangles, 3, 3)) + np.array([0.0, 0.0, 1.5]))
    budget = 600_000_000

    print(
        f"{'rays':>12} {'repeats':>8} {'tests/call':>14} {'seconds':>9} {'Mtest/s':>9}", flush=True
    )
    for rays in (50_000, 200_000, 800_000, 3_200_000):
        origin = jnp.asarray(rng.uniform(-1, 1, (rays, 3)))
        target = origin + jnp.asarray(rng.uniform(-1, 1, (rays, 3)) + np.array([0.0, 0.0, 3.0]))
        near = jnp.zeros(rays)
        # Fewer, larger calls at the big end; more, smaller ones at the small end. Equal total
        # tests either way, so a difference is the shape and not the budget.
        repeats = max(1, budget // (rays * n_triangles))

        def once(o=origin, t=target, nr=near, r=repeats):
            for _ in range(r):
                out = segment_is_cut(o, t, triangles, nr, work_limit=work_limit)
            return out

        jax.block_until_ready(once())
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            jax.block_until_ready(once())
            samples.append(time.perf_counter() - start)
        seconds = float(np.median(samples))
        tests = rays * n_triangles * repeats
        print(
            f"{rays:12,} {repeats:8d} {rays * n_triangles:14,} {seconds:9.2f} "
            f"{tests / seconds / 1e6:9.1f}",
            flush=True,
        )


if __name__ == "__main__":
    # 12 * divisions^2 wall triangles plus 4 * sectors for the sleeve. The top of the ladder is
    # set by the cube: 3184 facets is 3.2e10 tests, and one more rung would be most of an hour.
    cases = [(4, 8), (6, 12), (8, 16), (11, 20), (14, 24), (16, 28)]

    print(__doc__.split("Run with")[0].strip(), flush=True)

    # The silhouette's sections alone, for a change to the clip that leaves the ray mask alone.
    if os.environ.get("RADIATION_SILHOUETTE_ONLY"):
        silhouette_ladder(cases)
        second_stage(cases[:4])
        silhouette_stages(cases[2:])
        sys.exit(0)

    # The full range at the SHIPPED default, which is the number a user actually pays.
    ladder(cases, 4_000_000)

    # A much larger bound on two middle rungs only: it bounds the pass's memory, and through the
    # size of each call, its speed. Two rungs are enough to see which way it moves the build
    # without paying the cube twice.
    ladder(cases[2:4], 100_000_000)

    # The measured replacement for the extrapolated columns above. Run to the same top rung.
    silhouette_ladder(cases)
    second_stage(cases[:4])

    geometry_independence(11, 20, 4_000_000)
    throughput_against_ray_count(1532, 4_000_000)
