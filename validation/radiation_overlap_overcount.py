"""How much does the silhouette clip over-count where two blockers' silhouettes overlap?

``SilhouetteOcclusion`` clips each source against every blocker separately and **adds** the covered
fractions, so where two front-facing silhouettes overlap in angle the overlap is counted twice and
the pair errs dark. That is exact for a sight line crossing front-facing geometry once -- one
sleeve, a baffle, a wall -- and wrong for one sleeve standing behind another, which is how a
multi-lamp reactor is built. This harness sizes the error before any fix is designed.

**The reference takes the union, so it cannot double count.** For each (receiver, source) pair it
samples points over the source, weights each by the projected-solid-angle measure the clip's
fraction is a share of (``max(cos_r, 0) * |cos_s| / d**2``), and asks whether the segment from the
receiver to the sample is cut by *any* triangle of the surface -- the ray test
``segment_is_cut``, a different algorithm from the clip, with the receiver's and the source's
own facets excluded. A ray blocked by two sleeves is blocked once. The hidden share is the
weighted blocked fraction. It is cross-checked against the independent brute-force sampler in
the test references on a handful of pairs before anything is concluded from it.

What it reports, on a box with two sleeves side by side so that from the end walls one stands
directly behind the other:

1. **How often ``overlapping`` fires, and how often that is a real overlap.** The flag means "more
   than one blocker contributed", which a tiling of one sleeve also satisfies without overlapping.
2. **The over-count on the flagged pairs** -- clip minus reference -- as a count, a mean and a
   worst case, and as a share of each receiver's hemisphere wrongly reported dark.
3. **What it does to the surface irradiance**, the sleeves emitting and the walls reflecting:
   the clip's field against the same field with every evaluated pair corrected to the
   reference, and against the one-ray mask.

Run with ``validation/run_case.sh validation/radiation_overlap_overcount.py``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax.numpy as jnp
from aquaflux.radiation import (
    OcclusionField,
    RadiationModel,
    RadiationSettings,
    RayCastOcclusion,
    SelfOcclusion,
    SilhouetteOcclusion,
    Surfaces,
    build_transfer,
    build_visibility,
    projected_solid_angle,
    segment_is_cut,
    surface_irradiance,
)
from tests.unit.radiation_references import closed_drum, inward_box, sampled_fraction

#: Sample points per source facet for the reference. The standard error of a hidden share near a
#: half is then about 0.01 at the effective sample counts the weighting leaves; each pair's own
#: error is computed and reported rather than assumed.
SAMPLES = 4096

#: Pairs whose reference is formed per batch: bounds the ray arrays at SAMPLES times this.
BATCH = 256

#: A disagreement is called real only past this many standard errors of the reference, plus a
#: floor for the clip's own rounding.
SIGMAS = 3.0
FLOOR = 1e-3


def two_sleeve_reactor(divisions: int, sectors: int) -> Surfaces:
    """A closed box with two lamp sleeves side by side along x, one behind the other from x = 0.

    The sleeves emit and do not reflect; the walls reflect and do not emit -- the arrangement of a
    two-lamp reactor, and the one where the clip's additive fractions can overlap.
    """
    walls = inward_box(divisions)
    sleeves = [
        closed_drum(sectors, radius=0.1, half_height=0.3) + np.array([x, 0.5, 0.5])
        for x in (0.35, 0.65)
    ]
    triangles = np.concatenate([walls, *sleeves])
    is_sleeve = np.arange(len(triangles)) >= len(walls)
    return Surfaces.from_triangles(
        triangles,
        emission=np.where(is_sleeve, 1.0, 0.0),
        reflectance=np.where(is_sleeve, 0.0, 0.5),
    )


class PrecomputedOcclusion(SelfOcclusion):
    """A self-occlusion field supplied whole, so a corrected field can be solved like any other."""

    fraction: jnp.ndarray
    overlapping: jnp.ndarray = eqx.field(default=None)

    def field(self, surfaces, points, near, receiver_facet) -> OcclusionField:
        overlapping = (
            jnp.zeros_like(self.fraction, dtype=bool)
            if self.overlapping is None
            else self.overlapping
        )
        return OcclusionField(fraction=self.fraction, overlapping=overlapping)


def union_reference(surfaces: Surfaces, receiver, source, samples: int = SAMPLES, seed: int = 0):
    """The hidden share of each source, seen from each receiver, with no double counting.

    Parameters
    ----------
    surfaces : Surfaces
    receiver, source : np.ndarray of int, shape ``(n_pairs,)``
        Facet indices; the receiver sits at its facet's centroid.

    Returns
    -------
    tuple of np.ndarray, each shape ``(n_pairs,)``
        The hidden share and its standard error.
    """
    rng = np.random.default_rng(seed)
    u, v = rng.random(samples), rng.random(samples)
    outside = u + v > 1.0
    u, v = np.where(outside, 1.0 - u, u), np.where(outside, 1.0 - v, v)

    vertices = np.asarray(surfaces.vertices)
    normal = np.asarray(surfaces.normal)
    centroid = np.asarray(surfaces.centroid)
    triangles = jnp.asarray(vertices)

    hidden = np.empty(len(receiver))
    error = np.empty(len(receiver))
    for start in range(0, len(receiver), BATCH):
        r, s = receiver[start : start + BATCH], source[start : start + BATCH]
        a, b, c = vertices[s, 0], vertices[s, 1], vertices[s, 2]
        points = (
            a[:, None] + u[None, :, None] * (b - a)[:, None] + v[None, :, None] * (c - a)[:, None]
        )
        offset = points - centroid[r][:, None, :]
        squared = np.sum(offset * offset, axis=-1)
        unit = offset / np.sqrt(squared)[..., None]
        weight = (
            np.clip(np.einsum("pnd,pd->pn", unit, normal[r]), 0.0, None)
            * np.abs(np.einsum("pnd,pd->pn", unit, normal[s]))
            / squared
        )
        n_pairs = len(r)
        exclude = np.stack([np.repeat(s, samples), np.repeat(r, samples)], axis=-1)
        blocked = np.asarray(
            segment_is_cut(
                points.reshape(-1, 3),
                np.repeat(centroid[r], samples, axis=0),
                triangles,
                np.full(n_pairs * samples, 1e-9),
                exclude=exclude,
            )
        ).reshape(n_pairs, samples)
        total = weight.sum(axis=1)
        safe = np.where(total > 0.0, total, 1.0)
        share = (weight * blocked).sum(axis=1) / safe
        effective = total**2 / np.where(total > 0.0, (weight**2).sum(axis=1), 1.0)
        hidden[start : start + n_pairs] = np.where(total > 0.0, share, 0.0)
        # Agresti-Coull, not the plain binomial error. A share of exactly 0 or 1 gives the plain
        # form an error of ZERO, however few samples carried the weight -- and a pair whose small
        # shadow no sample happened to land in then reads as a 26-sigma disagreement. Measured
        # here: 0.0000 +/- 0.0000 at 4096 samples, 0.036 +/- 0.03 at 65536, the clip at 0.026.
        adjusted = (share * effective + 2.0) / (effective + 4.0)
        error[start : start + n_pairs] = np.sqrt(adjusted * (1.0 - adjusted) / (effective + 4.0))
    return hidden, error


def cross_check(surfaces: Surfaces, receiver, source) -> None:
    """The reference against the independent brute-force sampler, before it is trusted."""
    print("\n### the reference against the independent brute-force sampler\n", flush=True)
    print(f"{'pair':>12} {'union ref':>10} {'sampler':>10} {'gap':>8} {'3 sigma':>8}", flush=True)
    vertices = np.asarray(surfaces.vertices)
    normal = np.asarray(surfaces.normal)
    centroid = np.asarray(surfaces.centroid)
    hidden, error = union_reference(surfaces, receiver, source, samples=16384, seed=1)
    for k, (r, s) in enumerate(zip(receiver, source, strict=True)):
        others = np.delete(vertices, [r, s], axis=0)
        brute = sampled_fraction(centroid[r], normal[r], vertices[s], others, samples=100_000)
        print(
            f"{f'({r},{s})':>12} {hidden[k]:10.4f} {brute:10.4f} {hidden[k] - brute:8.4f} "
            f"{3 * error[k]:8.4f}",
            flush=True,
        )


def study(divisions: int, sectors: int, max_pairs: int | None, seed: int = 0) -> None:
    """Every measurement for one mesh."""
    surfaces = two_sleeve_reactor(divisions, sectors)
    n = int(surfaces.n_facets)
    facets = np.arange(n)
    print(
        f"\n## two-sleeve reactor, divisions={divisions}, sectors={sectors}: {n} facets\n",
        flush=True,
    )

    start = time.perf_counter()
    clip = SilhouetteOcclusion().field(surfaces, surfaces.centroid, jnp.zeros(n), facets)
    fraction, flagged = np.asarray(clip.fraction), np.asarray(clip.overlapping)
    print(f"silhouette field: {time.perf_counter() - start:.1f} s", flush=True)

    hidden = fraction > 1e-9
    print(
        f"pairs hidden at all: {int(hidden.sum()):,}   flagged overlapping: {int(flagged.sum()):,}"
        f"   flagged and hidden: {int((flagged & hidden).sum()):,}"
        f"   read fully hidden (clipped at 1): {int((fraction >= 1.0 - 1e-12).sum()):,}",
        flush=True,
    )

    # The pairs the reference is formed for: every flagged pair (or a random subset of them on a
    # large mesh), plus two controls -- hidden pairs with ONE contributor, which the clip claims
    # exact, and unhidden pairs, where a miss by the clip would show.
    rng = np.random.default_rng(seed)
    flagged_pairs = np.argwhere(flagged & hidden)
    if max_pairs is not None and len(flagged_pairs) > max_pairs:
        flagged_pairs = flagged_pairs[rng.choice(len(flagged_pairs), max_pairs, replace=False)]
    single = np.argwhere(hidden & ~flagged)
    single = single[rng.choice(len(single), min(len(single), 2000), replace=False)]
    clear = np.argwhere(~hidden & ~np.eye(n, dtype=bool))
    clear = clear[rng.choice(len(clear), min(len(clear), 2000), replace=False)]

    cross_check(surfaces, flagged_pairs[:6, 0], flagged_pairs[:6, 1])

    groups = {}
    for name, pairs in (
        ("flagged", flagged_pairs),
        ("one contributor", single),
        ("unhidden", clear),
    ):
        start = time.perf_counter()
        reference, error = union_reference(surfaces, pairs[:, 0], pairs[:, 1])
        groups[name] = (pairs, reference, error)
        print(
            f"reference for {len(pairs):,} {name} pairs: {time.perf_counter() - start:.1f} s",
            flush=True,
        )

    print("\n### clip against the union reference\n", flush=True)
    print(
        f"{'group':>16} {'pairs':>8} {'exact':>8} {'over':>8} {'under':>8} "
        f"{'mean over':>10} {'worst over':>11} {'worst under':>12}",
        flush=True,
    )
    for name, (pairs, reference, error) in groups.items():
        gap = fraction[pairs[:, 0], pairs[:, 1]] - reference
        band = SIGMAS * error + FLOOR
        over, under = gap > band, gap < -band
        print(
            f"{name:>16} {len(pairs):8,} {int((~over & ~under).sum()):8,} {int(over.sum()):8,} "
            f"{int(under.sum()):8,} {gap[over].mean() if over.any() else 0.0:10.4f} "
            f"{gap.max():11.4f} {gap.min():12.4f}",
            flush=True,
        )

    pairs, reference, error = groups["flagged"]
    gap = fraction[pairs[:, 0], pairs[:, 1]] - reference
    real = gap > SIGMAS * error + FLOOR
    print(
        f"\n`overlapping` precision: {int(real.sum()):,} of {len(pairs):,} flagged pairs over-count "
        f"({100.0 * real.mean():.1f}%) -- the rest are tilings, which add without overlapping.",
        flush=True,
    )

    # The error as the receiver sees it: the share of its hemisphere wrongly reported dark.
    share = (
        np.asarray(
            projected_solid_angle(
                jnp.asarray(surfaces.centroid)[pairs[:, 0]],
                jnp.asarray(surfaces.normal)[pairs[:, 0]],
                jnp.asarray(surfaces.vertices)[pairs[:, 1]],
            )
        )
        / np.pi
    )
    dark = np.zeros(n)
    np.add.at(dark, pairs[:, 0], share * np.clip(gap, 0.0, None))
    print(
        f"hemisphere wrongly dark, per receiver: mean {dark.mean():.4f}, worst {dark.max():.4f} "
        f"(of evaluated pairs{'' if max_pairs is None else ', a random subset'})",
        flush=True,
    )

    # Only meaningful when EVERY flagged pair was corrected; a subset leaves the rest uncorrected
    # and the "corrected" field would not be the truth it is compared against.
    if max_pairs is None:
        irradiance_effect(surfaces, fraction, groups["flagged"])


def irradiance_effect(surfaces: Surfaces, fraction, flagged_group) -> None:
    """The clip's surface irradiance against the corrected field and against the one-ray mask."""
    pairs, reference, _ = flagged_group
    corrected = fraction.copy()
    corrected[pairs[:, 0], pairs[:, 1]] = reference

    probe = np.asarray(surfaces.centroid)[:1]
    fields = {}
    for name, strategy in (
        ("silhouette", PrecomputedOcclusion(jnp.asarray(fraction))),
        ("corrected", PrecomputedOcclusion(jnp.asarray(corrected))),
        ("ray mask", RayCastOcclusion()),
    ):
        # Assembled directly: the volume-receiver mask is irrelevant to a surface quantity, and
        # the ray mask serves it whatever the facet strategy is.
        model = RadiationModel(
            receivers=jnp.asarray(probe),
            transfer=build_transfer(surfaces, self_occlusion=strategy),
            receiver_visibility=build_visibility((), surfaces, probe),
            settings=RadiationSettings(),
        )
        fields[name] = np.asarray(surface_irradiance(model, surfaces)[0])

    walls = ~np.asarray(surfaces.emission > 0.0)
    truth = fields["corrected"][walls]
    print(
        "\n### surface irradiance on the walls (sleeves emitting, walls reflecting 0.5)\n",
        flush=True,
    )
    print(
        f"{'field':>12} {'mean rel. error':>16} {'worst rel. error':>17} {'total':>10}", flush=True
    )
    for name in ("silhouette", "ray mask", "corrected"):
        value = fields[name][walls]
        relative = (value - truth) / np.where(truth > 0.0, truth, 1.0)
        print(
            f"{name:>12} {np.mean(np.abs(relative)):16.4f} {np.max(np.abs(relative)):17.4f} "
            f"{value.sum():10.4f}",
            flush=True,
        )


if __name__ == "__main__":
    print(__doc__.split("Run with")[0].strip(), flush=True)
    print(
        f"\nConfiguration: reference {SAMPLES} samples per source, a gap is real past "
        f"{SIGMAS:g} standard errors + {FLOOR:g}.",
        flush=True,
    )
    study(4, 12, max_pairs=None)
    study(6, 16, max_pairs=20_000)
