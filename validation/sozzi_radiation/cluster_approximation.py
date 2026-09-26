"""What does a cluster of lamp facets cost as ONE term, and does grouping by orientation fix it?

The second measurement of issue #565, after ``cluster_visibility.py``. That one found a cluster's
absorption error set by the emitter's cosine gate -- from any receiver part of a cluster faces away
and emits nothing, so the lit members' centre is not the cluster's -- and proposed grouping facets by
orientation as well as position. This measures that, and the half ``cluster_visibility.py`` kept
exact: a cluster's **solid angle as a single term**, which is what a cut has to use to save anything.

**Scene**, receivers and per-pair terms as in ``cluster_visibility.py`` (the analytic Sozzi
reactor, the lamp, ``UniformAbsorption(ABSORPTION)``, ``SOZZI_CLUSTER_RECEIVERS`` per region), and
the same control: the exact per-pair sum against ``direct_fluence_rate``.

**Clusterings**, each the leaves of a median-split tree over the facets:

- ``position``: halve along the longest axis of the centroids' bounding box until at most ``size``;
- ``oriented, cone C``: as ``position``, except that a set whose normals spread by more than
  ``C`` degrees from their mean is first halved along the normal component that spreads most;
- ``optical k, cone C``: split by normal as ``oriented``, then by position until the cluster's
  optical radius ``a * R`` is at most ``k`` -- bounded by size in the medium, not by member count,
  because ``cluster_visibility.py`` found the absorption error set by ``a * R`` whatever the distance.

**The one-term cluster**, for a receiver at ``r`` and a cluster with vector area ``S = sum A_m n_m``
and area-weighted representative point ``p``: ``(M / pi) max(S . u, 0) / d^2 exp(-a d)``, with
``d = |r - p|`` and ``u`` the unit direction from ``p`` to ``r`` -- one far-field solid angle, one
gate and one attenuation, instead of a solid angle, a gate and an attenuation per member. ``M`` is
uniform over the lamp, as a cluster needs it to be. Beside it, the partial approximation that keeps
the members' exact solid angles and gates and takes only the attenuation at ``p``.

**Where a cluster is used**: for lit pairs beyond ``tau`` radii (``SOZZI_CLUSTER_TAUS``) whose
cluster is **wholly visible** from the receiver. The visibility here is an oracle -- every member's
own shadow bit -- standing in for a shaft certificate, which is conservative: it would use clusters
in fewer places and so replace less, but it adds no error. Everywhere else the members are exact. A cluster **wholly hidden** from a receiver contributes
exactly zero; a "fully hidden" certificate would skip it too, so the work is reported both without
that (hidden clusters paid member by member) and with it (hidden clusters cost one term). A third
figure, **hidden only**, skips wholly hidden clusters and approximates nothing -- exact by
construction, so it carries no error column -- which is what "fully hidden" certificates alone
(#554 item C) would buy with no cluster approximation at all. Work is reported **per region** as
well as overall, because the share of hidden pairs depends on how many receivers are in the pipes,
and this sample over-represents them (the Sozzi mesh has 310,886 of its 1,635,909 cells in the pipes).
Reported per arm: the share of lit (receiver, cluster) pairs replaced, the work as a fraction of
the exact per-pair gather (``(exact pairs + replaced clusters) / exact pairs``), and the relative
error in ``G`` per region.

``SOZZI_CLUSTER_ARMS`` names the arms, ``kind:size[:cone]`` separated by semicolons (default below).

Run with ``validation/run_case.sh validation/sozzi_radiation/cluster_approximation.py``.
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
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.radiation import (  # noqa: E402
    NoOcclusion,
    UniformAbsorption,
    build_visibility,
    direct_fluence_rate,
)
from cluster_visibility import COUNTS, REGIONS, _pair_terms, sample  # noqa: E402
from compare_fluence import ABSORPTION  # noqa: E402
from primitive_occlusion import OUT, fluid, lamp  # noqa: E402

ARMS = os.environ.get(
    "SOZZI_CLUSTER_ARMS",
    "position:16;position:64;oriented:16:30;optical:0.25:30;optical:0.1:30;optical:0.25:15",
)
TAUS = tuple(float(t) for t in os.environ.get("SOZZI_CLUSTER_TAUS", "4,8,16").split(","))
CHUNK = 500


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class _Clusters:
    """Median-split leaves, optionally split by normal first while the normals spread too far.

    Attributes: ``members`` (padded with -1), ``radius`` (of the sphere about the bounding-box
    centre holding every vertex), ``point`` (area-weighted centroid), ``vector_area``
    (``sum A_m n_m``) and ``cone`` (the largest angle, in degrees, between a member's normal and
    the members' mean normal).
    """

    def __init__(self, vertices, centroid, normal, area, size, cone, max_radius=None):
        leaves: list[np.ndarray] = []
        stack = [np.arange(len(vertices))]
        while stack:
            index = stack.pop()
            spread = _spread(normal[index], area[index])
            if cone is not None and spread > cone and len(index) > 1:
                values = normal[index]
            elif (size is not None and len(index) > size) or (
                max_radius is not None and len(index) > 1 and _radius(vertices[index]) > max_radius
            ):
                values = centroid[index]
            else:
                leaves.append(index)
                continue
            axis = int(np.argmax(np.ptp(values, axis=0)))
            order = index[np.argsort(values[:, axis], kind="stable")]
            stack += [order[: len(order) // 2], order[len(order) // 2 :]]
        width = max(len(leaf) for leaf in leaves)
        self.members = np.full((len(leaves), width), -1, dtype=np.int64)
        self.radius = np.empty(len(leaves))
        self.point = np.empty((len(leaves), 3))
        self.vector_area = np.empty((len(leaves), 3))
        self.cone = np.empty(len(leaves))
        for row, leaf in enumerate(leaves):
            self.members[row, : len(leaf)] = leaf
            self.radius[row] = _radius(vertices[leaf])
            self.point[row] = (area[leaf, None] * centroid[leaf]).sum(axis=0) / area[leaf].sum()
            self.vector_area[row] = (area[leaf, None] * normal[leaf]).sum(axis=0)
            self.cone[row] = _spread(normal[leaf], area[leaf])


def _radius(vertices: np.ndarray) -> float:
    """Radius of the sphere about the bounding-box centre that holds every vertex."""
    corners = vertices.reshape(-1, 3)
    centre = 0.5 * (corners.min(axis=0) + corners.max(axis=0))
    return float(np.linalg.norm(corners - centre, axis=1).max())


def _spread(normals: np.ndarray, area: np.ndarray) -> float:
    """Largest angle in degrees between a normal and the area-weighted mean normal."""
    mean = (area[:, None] * normals).sum(axis=0)
    length = np.linalg.norm(mean)
    if length == 0.0:
        return 180.0
    cosine = np.clip(normals @ (mean / length), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine.min())))


def arms(spec: str, surfaces) -> dict[str, _Clusters]:
    vertices = np.asarray(surfaces.vertices)
    centroid = np.asarray(surfaces.centroid)
    normal = np.asarray(surfaces.normal)
    area = np.asarray(surfaces.area)
    built = {}
    for arm in spec.split(";"):
        kind, first, *rest = arm.split(":")
        if kind == "optical":
            name = f"optical a*R <= {first}, cone {rest[0]} deg"
            built[name] = _Clusters(
                vertices, centroid, normal, area, None, float(rest[0]), float(first) / ABSORPTION
            )
            continue
        cone = float(rest[0]) if kind == "oriented" else None
        name = f"{kind} {first}" + (f", cone {rest[0]} deg" if cone is not None else "")
        built[name] = _Clusters(vertices, centroid, normal, area, int(first), cone)
    return built


def main() -> None:
    rng = np.random.default_rng(0)
    water = fluid()
    surfaces, lamp_source = lamp()
    points, region = sample(rng, water, np.asarray(surfaces.vertices))
    medium = UniformAbsorption(ABSORPTION)
    emission = float(np.asarray(surfaces.emission)[0])
    assert np.all(np.asarray(surfaces.emission) == emission), "a cluster needs uniform emission"
    clusters = arms(ARMS, surfaces)
    _say(
        f"{len(points)} receivers ({', '.join(f'{c} {n}' for c, n in zip(COUNTS, REGIONS, strict=True))}), "
        f"{surfaces.n_facets} lamp facets ({lamp_source}), absorption {ABSORPTION} /m; "
        f"jax {jax.__version__}, {platform.system()} {platform.machine()}, {os.cpu_count()} cores"
    )
    exact = np.zeros(len(points))
    partial = {(name, tau): np.zeros(len(points)) for name in clusters for tau in TAUS}
    one_term = {(name, tau): np.zeros(len(points)) for name in clusters for tau in TAUS}
    replaced = {(name, tau): 0 for name in clusters for tau in TAUS}
    kept_pairs = {(name, tau): np.zeros(len(REGIONS)) for name in clusters for tau in TAUS}
    hidden_pairs = {(name, tau): np.zeros(len(REGIONS)) for name in clusters for tau in TAUS}
    hidden_clusters = {(name, tau): np.zeros(len(REGIONS)) for name in clusters for tau in TAUS}
    replaced_by_region = {(name, tau): np.zeros(len(REGIONS)) for name in clusters for tau in TAUS}
    only_hidden_pairs = {name: np.zeros(len(REGIONS)) for name in clusters}
    only_hidden_clusters = {name: np.zeros(len(REGIONS)) for name in clusters}
    region_pairs = np.zeros(len(REGIONS))
    lit_pairs = dict.fromkeys(clusters, 0)
    gate_straddle = dict.fromkeys(clusters, 0)
    all_pairs = 0
    control = None
    started = time.perf_counter()
    for start in range(0, len(points), CHUNK):
        chunk = points[start : start + CHUNK]
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
        exact[start : start + CHUNK] = (w * visible).sum(axis=1)
        all_pairs += int(gate.sum())
        labels = region[start : start + CHUNK]

        def by_region(values, labels=labels):
            return np.array([values[labels == r].sum() for r in range(len(REGIONS))])

        region_pairs += by_region(gate.sum(axis=1))
        if control is None:
            reference = np.asarray(
                direct_fluence_rate(
                    surfaces, jnp.asarray(chunk), absorption=medium, visibility=mask
                )
            )
            control = float(np.max(np.abs(exact[start : start + CHUNK] - reference) / reference))
        for name, found in clusters.items():
            present = found.members >= 0
            pick = np.maximum(found.members, 0)
            w_c = np.where(present, w[:, pick], 0.0)
            geometric_c = np.where(present, geometric[:, pick], 0.0)
            v_c = np.where(present, visible[:, pick], 1.0)
            gate_c = gate[:, pick] & present
            lit = gate_c.any(axis=2)
            carried = (w_c * v_c).sum(axis=2)
            wholly_visible = (v_c > 0.5).all(axis=2)
            wholly_hidden = ~(np.where(present, visible[:, pick], 0.0) > 0.5).any(axis=2)
            lit_pairs[name] += int(lit.sum())
            gate_straddle[name] += int((lit & (present & ~gate[:, pick]).any(axis=2)).sum())
            offset = chunk[:, None, :] - found.point[None, :, :]
            distance = np.linalg.norm(offset, axis=-1)
            direction = offset / distance[..., None]
            ratio = distance / found.radius[None, :]
            attenuation = np.exp(-ABSORPTION * distance)
            by_partial = (geometric_c * v_c).sum(axis=2) * attenuation
            projected = np.maximum(np.einsum("rci,ci->rc", direction, found.vector_area), 0.0)
            by_one_term = emission / np.pi * projected / distance**2 * attenuation
            lit_members = gate_c.sum(axis=2)
            dark_anywhere = lit & wholly_hidden
            only_hidden_pairs[name] += by_region(np.where(dark_anywhere, lit_members, 0).sum(1))
            only_hidden_clusters[name] += by_region(dark_anywhere.sum(1))
            for tau in TAUS:
                use = lit & wholly_visible & (ratio >= tau)
                partial[name, tau][start : start + CHUNK] = np.where(use, by_partial, carried).sum(
                    1
                )
                one_term[name, tau][start : start + CHUNK] = np.where(
                    use, by_one_term, carried
                ).sum(1)
                replaced[name, tau] += int(use.sum())
                kept_pairs[name, tau] += by_region(np.where(use, 0, lit_members).sum(1))
                dark = lit & wholly_hidden & ~use
                hidden_pairs[name, tau] += by_region(np.where(dark, lit_members, 0).sum(1))
                hidden_clusters[name, tau] += by_region(dark.sum(1))
                replaced_by_region[name, tau] += by_region(use.sum(1))
        _say(
            f"{start + len(chunk)} / {len(points)} receivers, {time.perf_counter() - started:.0f} s"
        )
    _say(f"control: exact per-pair sum against direct_fluence_rate, max relative {control:.1e}")

    summary = {"control": control, "arms": {}}
    for name, found in clusters.items():
        _say(
            f"\n### {name}: {len(found.radius)} clusters; radius median {np.median(found.radius):.4f} m "
            f"(a*R {ABSORPTION * np.median(found.radius):.2f}); normal cone median "
            f"{np.median(found.cone):.1f} / max {found.cone.max():.1f} deg; lit pairs straddling the "
            f"gate {gate_straddle[name] / lit_pairs[name]:.3f}"
        )
        entry = {"clusters": len(found.radius), "radius_median": float(np.median(found.radius)),
                 "cone_median": float(np.median(found.cone)),
                 "gate_straddle": gate_straddle[name] / lit_pairs[name], "taus": {}}  # fmt: skip
        only = region_pairs - only_hidden_pairs[name] + only_hidden_clusters[name]
        entry["hidden_only_work"] = {
            "all": float(only.sum() / region_pairs.sum()),
            **{r: float(only[i] / region_pairs[i]) for i, r in enumerate(REGIONS)},
        }
        _say(
            "hidden only (exact): work "
            + ", ".join(f"{k} {v:.3f}" for k, v in entry["hidden_only_work"].items())
        )
        for tau in TAUS:
            per_region = kept_pairs[name, tau] + replaced_by_region[name, tau]
            per_region_hidden = per_region - hidden_pairs[name, tau] + hidden_clusters[name, tau]
            work = float(per_region.sum() / region_pairs.sum())
            with_hidden = float(per_region_hidden.sum() / region_pairs.sum())
            _say(
                f"beyond {tau:g} radii, work by region with hidden certificates: "
                + ", ".join(
                    f"{r} {per_region_hidden[i] / region_pairs[i]:.3f}"
                    for i, r in enumerate(REGIONS)
                )
            )
            line = {"replaced": replaced[name, tau] / lit_pairs[name], "work": work,
                    "work_with_hidden_certificates": with_hidden,
                    "work_with_hidden_by_region": {
                        r: float(per_region_hidden[i] / region_pairs[i]) for i, r in enumerate(REGIONS)
                    }}  # fmt: skip
            for label, values in (
                ("partial", partial[name, tau]),
                ("one term", one_term[name, tau]),
            ):
                per = np.abs(values - exact) / exact
                stats = {
                    region_name: [float(np.median(per[region == r])),
                                  float(np.quantile(per[region == r], 0.99)),
                                  float(per[region == r].max())]
                    for r, region_name in enumerate(REGIONS)
                }  # fmt: skip
                line[label] = stats
                _say(
                    f"beyond {tau:g} radii, {label:>8}: replaces {line['replaced']:.3f} of lit pairs, "
                    f"work {work:.3f} of exact ({with_hidden:.3f} with hidden certificates); "
                    + ", ".join(
                        f"{r} {s[0]:.1e} / {s[1]:.1e} / {s[2]:.1e}" for r, s in stats.items()
                    )
                )
            entry["taus"][tau] = line
        summary["arms"][name] = entry
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cluster_approximation.json").write_text(json.dumps(summary, indent=2))
    _say(f"wrote {OUT / 'cluster_approximation.json'} (errors: median / p99 / max per region)")


if __name__ == "__main__":
    main()
