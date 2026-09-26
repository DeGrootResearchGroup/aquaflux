"""How often does a cluster of lamp facets straddle a shadow, and what does one ray per cluster cost?

The design question of issue #565, measured before anything is built. Clustering the emitter
(lightcuts, a fast multipole method) pays for itself only by NOT evaluating a cluster's members
one by one for a distant receiver: one geometry term and one shadow ray per (receiver, cluster)
instead of one per (receiver, facet). A cluster whose members are partly shadowed from a receiver
-- it *straddles* a shadow edge -- is then answered wrongly by that one ray, whichever way it
falls. This measures how often that happens, where, how much of the field it carries, and what the
two approximations a cluster makes cost on their own and together.

**Scene.** The Sozzi reactor as ``Outside(chamber, inlet, riser)`` at the tutorial's dimensions
(``primitive_occlusion.fluid``), the case's lamp STL if the case is present and otherwise the
analytic 32 x 128 lamp, exitance ``EXITANCE``, ``UniformAbsorption(ABSORPTION)``, the lamp's own
facets not occluding (it is convex; its emitter cosine gate is its visibility). Receivers are
sampled uniformly inside each region separately -- ``SOZZI_CLUSTER_RECEIVERS`` gives the chamber,
inlet and riser counts, default ``3000,1500,1500`` -- because the pipes are where the shadows are
and a volume-uniform sample holds very few of them. A receiver belongs to the chamber if it is
inside the chamber, else to the pipe it is in.

**Clusters** are the leaves of a median-split tree over the facet centroids: a set is halved
along the longest axis of its bounding box until it holds at most ``size`` facets, for each size
in ``SOZZI_CLUSTER_SIZES`` (default ``4,16,64,256``). ⚠️ **Not** ``FacetClusters``: its Morton
keys normalize each axis of the bounding box separately, and on a lamp 0.8 m long and 2 cm across
that makes the curve anisotropic, so some runs of four facets span half the lamp (radius 0.405 m
at every size); a cut built on those would be measuring the clusters' shape rather than the idea.
A cluster's representative point is the area-weighted mean of its members' centroids, and its
extent the radius of the smallest sphere about its bounding-box centre that holds every vertex;
``distance / radius`` is the ratio a cut would refine on, and ``a * radius`` its optical size.

**What is computed, per (receiver, facet) pair, exactly**: ``w = M * rad(cos) * Omega * exp(-a d)``
with the centroid distance and cosine, and the shadow bit ``v`` from ``build_visibility``. The
exact field is ``sum w v``. **Control**: that sum is compared with ``direct_fluence_rate`` through
the same mask on the first chunk, and must agree to rounding.

**The two approximations, each alone and then together**:

- *one ray per cluster*: every member's ``w`` exact, but one visibility bit for the whole cluster,
  from a ray between the representative point and the receiver;
- *absorption at the representative point*: every member's shadow bit exact, but ``exp(-a d)``
  evaluated once, at the representative point's distance;
- *the same, with the cluster's second moment*: the members' offsets ``x`` from the representative
  point make ``d_m ~ d - x . n`` along the direction ``n`` to the receiver, so the mean of
  ``exp(-a d_m)`` is ``exp(-a d) <exp(a x . n)>``, whose second-order cumulant form is
  ``exp(-a d + a^2 n.S.n / 2)`` with ``S`` the members' area-weighted covariance. ``S`` is frozen
  per cluster and ``a`` stays live, so this costs one quadratic form per (receiver, cluster);
- *both, applied only to clusters farther than* ``tau`` *radii* (``SOZZI_CLUSTER_TAUS``, default
  ``2,4,8,16``), nearer clusters left exact -- which is what an error-bounded cut does. Reported
  with the share of (receiver, cluster) pairs that the approximation replaced, which is the
  saving; the geometry term (the members' solid angles) is still exact there, so this measures the
  visibility and absorption halves of the approximation, not the geometry half.

A cluster size of one would make every approximation exact; that reduction is not printed as a
check, because it holds by construction.

Run with ``validation/run_case.sh validation/sozzi_radiation/cluster_visibility.py``.
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
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.radiation import (  # noqa: E402
    NoOcclusion,
    UniformAbsorption,
    build_visibility,
    direct_fluence_rate,
)
from aquaflux.radiation.solid_angle import solid_angle  # noqa: E402
from compare_fluence import ABSORPTION  # noqa: E402
from primitive_occlusion import INLET_END, OUT, R_BODY, RISER_TOP, fluid, lamp  # noqa: E402

COUNTS = tuple(
    int(n) for n in os.environ.get("SOZZI_CLUSTER_RECEIVERS", "3000,1500,1500").split(",")
)
SIZES = tuple(int(n) for n in os.environ.get("SOZZI_CLUSTER_SIZES", "4,16,64,256").split(","))
TAUS = tuple(float(t) for t in os.environ.get("SOZZI_CLUSTER_TAUS", "2,4,8,16").split(","))
REGIONS = ("chamber", "inlet", "riser")
#: Distance-over-radius bands the straddle shares are reported in.
BANDS = ((0.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, np.inf))
CHUNK = 500


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class _Leaves:
    """Median-split leaves: ``members`` (padded with -1) and each leaf's bounding ``radius``."""

    def __init__(self, vertices: np.ndarray, size: int):
        centroid = vertices.mean(axis=1)
        leaves: list[np.ndarray] = []
        stack = [np.arange(len(vertices))]
        while stack:
            index = stack.pop()
            if len(index) <= size:
                leaves.append(index)
                continue
            spread = np.ptp(centroid[index], axis=0)
            axis = int(np.argmax(spread))
            order = index[np.argsort(centroid[index, axis], kind="stable")]
            stack += [order[: len(order) // 2], order[len(order) // 2 :]]
        self.members = np.full((len(leaves), size), -1, dtype=np.int64)
        self.radius = np.empty(len(leaves))
        for row, leaf in enumerate(leaves):
            self.members[row, : len(leaf)] = leaf
            corners = vertices[leaf].reshape(-1, 3)
            centre = 0.5 * (corners.min(axis=0) + corners.max(axis=0))
            self.radius[row] = np.linalg.norm(corners - centre, axis=1).max()


def sample(rng, water) -> tuple[np.ndarray, np.ndarray]:
    """``COUNTS`` receivers in the chamber, the inlet and the riser, and each one's region index."""
    box = np.array([[0.0, -R_BODY, -R_BODY], [INLET_END, R_BODY, RISER_TOP]])
    points, labels = [], []
    for index, count in enumerate(COUNTS):
        found: list[np.ndarray] = []
        while sum(len(block) for block in found) < count:
            trial = rng.uniform(box[0], box[1], (20 * count, 3))
            wet = ~np.asarray(water.contains(jnp.asarray(trial)))
            depth = np.stack(
                [np.asarray(region.signed_distance(jnp.asarray(trial))) for region in water.regions]
            )
            region = np.where(depth[0] <= 0.0, 0, np.where(depth[1] <= 0.0, 1, 2))
            found.append(trial[wet & (region == index)])
        points.append(np.concatenate(found)[:count])
        labels.append(np.full(count, index))
    return np.concatenate(points), np.concatenate(labels)


@eqx.filter_jit
def _pair_terms(points, vertices, centroid, normal, emission, coefficient):
    """Per (receiver, facet): the exact weight ``w`` without the shadow, and the emitter gate."""
    offset = points[:, None, :] - centroid[None, :, :]
    distance = jnp.sqrt(jnp.sum(offset * offset, axis=-1))
    cosine = jnp.sum(offset * normal[None, :, :], axis=-1) / distance
    omega = solid_angle(points[:, None, :], vertices[None, ...])
    radiance = jnp.where(cosine > 0.0, emission[None, :] / jnp.pi, 0.0)
    return radiance * omega * jnp.exp(-coefficient * distance), radiance * omega, cosine > 0.0


@eqx.filter_jit
def _ray_blocked(water, origin, target, near):
    return water.blocks(origin, target, near)


def main() -> None:
    rng = np.random.default_rng(0)
    water = fluid()
    surfaces, lamp_source = lamp()
    points, region = sample(rng, water)
    medium = UniformAbsorption(ABSORPTION)
    vertices = np.asarray(surfaces.vertices)
    centroid = np.asarray(surfaces.centroid)
    area = np.asarray(surfaces.area)
    _say(
        f"{len(points)} receivers ({', '.join(f'{c} {n}' for c, n in zip(COUNTS, REGIONS, strict=True))}), "
        f"{surfaces.n_facets} lamp facets ({lamp_source}), absorption {ABSORPTION} /m; "
        f"jax {jax.__version__}, {platform.system()} {platform.machine()}, {os.cpu_count()} cores"
    )
    clusters = {size: _Leaves(vertices, size) for size in SIZES}
    representative, covariance = {}, {}
    for size, found in clusters.items():
        members = found.members
        present = members >= 0
        weights = np.where(present, area[np.maximum(members, 0)], 0.0)
        representative[size] = (
            np.einsum("cm,cmk->ck", weights, centroid[np.maximum(members, 0)])
            / weights.sum(axis=1)[:, None]
        )
        spread = centroid[np.maximum(members, 0)] - representative[size][:, None, :]
        covariance[size] = (
            np.einsum("cm,cmi,cmj->cij", weights, spread, spread)
            / weights.sum(axis=1)[:, None, None]
        )

    exact = np.zeros(len(points))
    one_ray = {size: np.zeros(len(points)) for size in SIZES}
    one_absorption = {size: np.zeros(len(points)) for size in SIZES}
    moment = {size: np.zeros(len(points)) for size in SIZES}
    far = {(size, tau): np.zeros(len(points)) for size in SIZES for tau in TAUS}
    replaced = {(size, tau): 0 for size in SIZES for tau in TAUS}
    lit_pairs = {size: 0 for size in SIZES}
    # [size][band][region] -> [lit (r, C) pairs, mask-straddling pairs, gate-straddling pairs,
    #                          exact G carried by all lit pairs, exact G carried by straddlers]
    tally = {size: np.zeros((len(BANDS), len(REGIONS), 5)) for size in SIZES}
    control = None
    started = time.perf_counter()
    for start in range(0, len(points), CHUNK):
        chunk = points[start : start + CHUNK]
        labels = region[start : start + CHUNK]
        mask = build_visibility([water], surfaces, chunk, self_occlusion=NoOcclusion())
        visible = 1.0 - np.asarray(mask.blocked[0], dtype=float)
        w, geometric, gate = (
            np.asarray(term)
            for term in _pair_terms(
                jnp.asarray(chunk),
                surfaces.vertices,
                surfaces.centroid,
                surfaces.normal,
                surfaces.emission,
                ABSORPTION,
            )
        )
        chunk_exact = (w * visible).sum(axis=1)
        exact[start : start + CHUNK] = chunk_exact
        if control is None:
            reference = np.asarray(
                direct_fluence_rate(
                    surfaces, jnp.asarray(chunk), absorption=medium, visibility=mask
                )
            )
            control = float(np.max(np.abs(chunk_exact - reference) / np.abs(reference)))
        for size, found in clusters.items():
            members = found.members
            present = members >= 0
            pick = np.maximum(members, 0)
            w_c = np.where(present, w[:, pick], 0.0)  # (n_r, n_C, size)
            geometric_c = np.where(present, geometric[:, pick], 0.0)
            v_c = visible[:, pick]
            gate_c = gate[:, pick] & present
            lit = gate_c.any(axis=2)
            carried = (w_c * v_c).sum(axis=2)
            weighted = w_c > 0.0
            some_shown = (weighted & (v_c > 0.5)).any(axis=2)
            some_hidden = (weighted & (v_c < 0.5)).any(axis=2)
            straddles = some_shown & some_hidden
            gate_straddles = lit & (present & ~gate[:, pick]).any(axis=2)
            rep = representative[size]
            offset = chunk[:, None, :] - rep[None, :, :]
            distance = np.linalg.norm(offset, axis=-1)
            ratio = distance / found.radius[None, :]
            near = np.full((len(chunk), len(rep)), 1e-6 * np.sqrt(area.mean()))
            ray_clear = 1.0 - np.asarray(
                _ray_blocked(
                    water,
                    jnp.asarray(np.broadcast_to(rep[None], offset.shape)),
                    jnp.asarray(np.broadcast_to(chunk[:, None, :], offset.shape)),
                    jnp.asarray(near),
                ),
                dtype=float,
            )
            by_ray = w_c.sum(axis=2) * ray_clear
            by_absorption = (geometric_c * v_c).sum(axis=2) * np.exp(-ABSORPTION * distance)
            direction = offset / distance[..., None]
            stretch = np.einsum("rci,cij,rcj->rc", direction, covariance[size], direction)
            by_moment = by_absorption * np.exp(0.5 * ABSORPTION**2 * stretch)
            both = geometric_c.sum(axis=2) * np.exp(-ABSORPTION * distance) * ray_clear
            one_ray[size][start : start + CHUNK] = by_ray.sum(axis=1)
            one_absorption[size][start : start + CHUNK] = by_absorption.sum(axis=1)
            moment[size][start : start + CHUNK] = by_moment.sum(axis=1)
            lit_pairs[size] += int(lit.sum())
            for tau in TAUS:
                coarse = lit & (ratio >= tau)
                far[size, tau][start : start + CHUNK] = np.where(coarse, both, carried).sum(axis=1)
                replaced[size, tau] += int(coarse.sum())
            for b, (low, high) in enumerate(BANDS):
                in_band = lit & (ratio >= low) & (ratio < high)
                for r in range(len(REGIONS)):
                    rows = labels == r
                    sel = in_band[rows]
                    tally[size][b, r] += [
                        sel.sum(),
                        (sel & straddles[rows]).sum(),
                        (sel & gate_straddles[rows]).sum(),
                        np.where(sel, carried[rows], 0.0).sum(),
                        np.where(sel & straddles[rows], carried[rows], 0.0).sum(),
                    ]
        _say(
            f"{start + len(chunk)} / {len(points)} receivers, {time.perf_counter() - started:.0f} s"
        )

    _say(f"control: exact per-pair sum against direct_fluence_rate, max relative {control:.1e}")

    def relative(values):
        return np.abs(values - exact) / exact

    summary = {"control": control, "sizes": {}}
    for size in SIZES:
        radius = clusters[size].radius
        _say(
            f"\n### clusters of at most {size} facets ({len(radius)} clusters; radius median "
            f"{np.median(radius):.4f} / max {radius.max():.4f} m, a*radius median "
            f"{ABSORPTION * np.median(radius):.2f} / max {ABSORPTION * radius.max():.2f})"
        )
        _say(
            f"{'distance/radius':>16} {'region':>8} {'lit (r,C)':>11} {'straddle':>9} "
            f"{'gate-str.':>9} {'G share of straddlers':>22}"
        )
        rows = []
        for b, (low, high) in enumerate(BANDS):
            for r, name in enumerate(REGIONS):
                n, straddle, gate_straddle, g_all, g_straddle = tally[size][b, r]
                share = straddle / n if n else 0.0
                band = f"{low:g}-{high:g}"
                _say(
                    f"{band:>16} {name:>8} {int(n):11d} {share:9.4f} "
                    f"{(gate_straddle / n if n else 0.0):9.4f} "
                    f"{(g_straddle / g_all if g_all else 0.0):22.4f}"
                )
                rows.append(
                    {"band": band, "region": name, "lit": int(n), "straddle": share,
                     "gate_straddle": gate_straddle / n if n else 0.0,
                     "straddle_share_of_G": g_straddle / g_all if g_all else 0.0}
                )  # fmt: skip
        errors = {}
        for label, values in (
            ("one ray per cluster", one_ray[size]),
            ("absorption at the representative point", one_absorption[size]),
            ("the same with the second moment", moment[size]),
        ):
            per = relative(values)
            text = ", ".join(
                f"{name} median {np.median(per[region == r]):.2e} / p99 "
                f"{np.quantile(per[region == r], 0.99):.2e} / max {per[region == r].max():.2e}"
                for r, name in enumerate(REGIONS)
            )
            _say(f"{label}, everywhere: {text}")
            errors[label] = {
                name: [float(np.median(per[region == r])), float(np.quantile(per[region == r], 0.99)),
                       float(per[region == r].max())]
                for r, name in enumerate(REGIONS)
            }  # fmt: skip
        for tau in TAUS:
            per = relative(far[size, tau])
            text = ", ".join(
                f"{name} median {np.median(per[region == r]):.2e} / p99 "
                f"{np.quantile(per[region == r], 0.99):.2e} / max {per[region == r].max():.2e}"
                for r, name in enumerate(REGIONS)
            )
            saved = replaced[size, tau] / lit_pairs[size]
            _say(f"both, beyond {tau:g} radii: replaces {saved:.3f} of lit (r, C) pairs; {text}")
            errors[f"both beyond {tau:g}"] = {
                "replaced": saved,
                **{
                    name: [float(np.median(per[region == r])), float(np.quantile(per[region == r], 0.99)),
                           float(per[region == r].max())]
                    for r, name in enumerate(REGIONS)
                },
            }  # fmt: skip
        summary["sizes"][size] = {"bands": rows, "errors": errors}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cluster_visibility.json").write_text(json.dumps(summary, indent=2))
    _say(f"wrote {OUT / 'cluster_visibility.json'}")


if __name__ == "__main__":
    main()
