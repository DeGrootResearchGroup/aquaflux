"""Can the occlusion mask be made EXACT geometrically, instead of sampled?

``validation/radiation_partial_occlusion.py`` measured what the binary per-pair mask costs, and
a separate sweep measured what sampling buys: receiver quadrature and source pixelation each
improve the mean two- to four-fold and leave the worst pair untouched. This harness measures the
third option, which is neither sampling nor meshing.

**The idea.** For a receiver point and a source triangle, the blocked part of the source's
angular extent is the spherical intersection of the source with the blocker's silhouette. Clip
the source's directions against the three planes through the receiver carrying the blocker's
edges, take the signed projected solid angle of what survives, and divide. No rays, no sampling,
so there is no straddling pair and nothing to converge. This is the classical analytic
form-factor treatment — Nishita and Nakamae (1983), then Baum, Rushmeier and Winget (*Computer
Graphics* 23(3), 1989), who project blockers onto the source's supporting plane and clip.

Everything here works in **direction space** relative to the receiver rather than on the source's
plane: no perspective divide, so no projective infinity when a blocker straddles that plane.

**What makes it expressible at all** is that the contour form of the projected solid angle is
*signed* and additive over loops — the magnitude is taken only at the very end of
:func:`~aquaflux.radiation.solid_angle.projected_solid_angle`. So the visible part never has to
be constructed: it is the whole minus the covered part, and the covered part is an intersection
of two convex regions, which is convex with a statically bounded vertex count.

Run with ``validation/run_case.sh validation/radiation_analytic_occlusion.py``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
from aquaflux.radiation import Surfaces
from aquaflux.radiation.solid_angle import _clip_to_front, _unit
from aquaflux.vectors import dot
from tests.unit.radiation_references import cylinder_triangles, inward_box

#: Samples for the brute-force reference. Its noise floor is about ``1/sqrt(SAMPLES)``, which is
#: what the analytic answer is eventually compared against rather than to zero.
SAMPLES = 1_500_000


def signed_projected(normal, loop):
    """The contour form of the projected solid angle with the final magnitude NOT taken."""
    direction, _ = _unit(loop)
    following = jnp.roll(direction, -1, axis=-2)
    edge = jnp.cross(direction, following)
    squared = dot(edge, edge)
    flat = squared == 0.0
    span = jnp.sqrt(jnp.where(flat, 1.0, squared))
    axis = edge / span[..., None]
    span = jnp.where(flat, 0.0, span)
    angle = jnp.arctan2(span, dot(direction, following))
    return 0.5 * jnp.sum(angle * dot(axis, normal[..., None, :]), axis=-1)


def clip_halfspace(loop, plane_normal):
    """Keep the part of a direction loop with ``d . plane_normal >= 0``.

    The same fixed-shape Sutherland-Hodgman the kernel already uses for the receiver's own
    plane: each vertex emits itself and the crossing on the edge after it, and a rejected
    candidate repeats its predecessor. The repeats are zero-length edges of the same closed loop
    and contribute no angle, so the output shape is static at twice the input — which is what
    makes this expressible under ``jit`` at all.
    """
    height = dot(loop, plane_normal[..., None, :])
    candidates, kept = [], []
    for k in range(loop.shape[-2]):
        following = (k + 1) % loop.shape[-2]
        here, there = height[..., k], height[..., following]
        candidates.append(loop[..., k, :])
        kept.append(here >= 0.0)
        gap = here - there
        crossing = jnp.clip(here / jnp.where(gap == 0.0, 1.0, gap), 0.0, 1.0)
        candidates.append(
            loop[..., k, :] + crossing[..., None] * (loop[..., following, :] - loop[..., k, :])
        )
        kept.append(here * there < 0.0)
    last = jnp.zeros_like(loop[..., 0, :])
    for survived, candidate in zip(kept, candidates, strict=True):
        last = jnp.where(survived[..., None], candidate, last)
    out = []
    for survived, candidate in zip(kept, candidates, strict=True):
        last = jnp.where(survived[..., None], candidate, last)
        out.append(last)
    return jnp.stack(out, axis=-2)


def blocked_fraction(receiver, receiver_normal, source, blocker):
    """Fraction of the source's projected solid angle the blocker covers, analytically."""
    receiver = jnp.asarray(receiver, dtype=float)
    receiver_normal = jnp.asarray(receiver_normal, dtype=float)
    to_source = jnp.asarray(source, dtype=float) - receiver
    to_blocker = jnp.asarray(blocker, dtype=float) - receiver

    loop = _clip_to_front(to_source, receiver_normal)
    whole = signed_projected(receiver_normal, loop)
    # The blocker's winding as seen from here decides which side of each edge plane is inside,
    # so the orientation is read off rather than assumed -- an imported file's winding is
    # whatever the exporter wrote.
    corner = [to_blocker[..., k, :] for k in range(3)]
    facing = jnp.sign(dot(corner[0], jnp.cross(corner[1], corner[2])))[..., None]
    for first, second in ((0, 1), (1, 2), (2, 0)):
        loop = clip_halfspace(loop, jnp.cross(corner[first], corner[second]) * facing)
    covered = signed_projected(receiver_normal, loop)
    return jnp.abs(covered) / jnp.where(jnp.abs(whole) == 0.0, 1.0, jnp.abs(whole))


def sampled_fraction(receiver, receiver_normal, source, blockers, samples=SAMPLES, seed=0):
    """Brute force: area-sample the source, weight by the projected solid angle, ray-test."""
    rng = np.random.default_rng(seed)
    a, b, c = np.asarray(source, dtype=float)
    u, v = rng.random(samples), rng.random(samples)
    outside = u + v > 1.0
    u, v = np.where(outside, 1 - u, u), np.where(outside, 1 - v, v)
    points = a + u[:, None] * (b - a) + v[:, None] * (c - a)

    receiver = np.asarray(receiver, dtype=float)
    offset = points - receiver
    squared = np.sum(offset * offset, axis=1)
    unit = offset / np.sqrt(squared)[:, None]
    source_normal = np.cross(b - a, c - a)
    source_normal /= np.linalg.norm(source_normal)
    weight = np.abs(unit @ np.asarray(receiver_normal)) * np.abs(unit @ source_normal) / squared

    blocked = np.zeros(samples, dtype=bool)
    for p0, p1, p2 in np.atleast_3d(np.asarray(blockers, dtype=float)).reshape(-1, 3, 3):
        e1, e2 = p1 - p0, p2 - p0
        h = np.cross(unit, e2)
        det = e1 @ h.T
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = receiver - p0
        bu = inv * (s @ h.T)
        q = np.cross(s, e1)
        bv = inv * (unit @ q)
        t = inv * (e2 @ q)
        inside = ok & (bu >= 0) & (bu <= 1) & (bv >= 0) & (bu + bv <= 1)
        # Strictly between the receiver and the sample: a triangle beyond the source does not
        # occlude it, and one at the origin is the receiver's own facet.
        blocked |= inside & (t > 1e-12) & (t < np.sqrt(squared))
    return float(np.sum(weight * blocked) / np.sum(weight))


def split_once(triangle):
    """Midpoint split into four similar triangles, windings kept with the parent."""
    a, b, c = np.asarray(triangle, dtype=float)
    ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
    return [np.array(x) for x in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))]


def across_the_gap(radius=0.30, sectors=48, slices=4):
    """A closed triangulated tube lying across a z-separated gap -- the STL blocker."""
    return cylinder_triangles(radius, 2.0, sectors=sectors, slices=slices)[:, :, [0, 2, 1]]


def front_facing(blockers, receiver):
    """Which blocker triangles turn their outward normal back towards the receiver."""
    edge_a = blockers[:, 1] - blockers[:, 0]
    edge_b = blockers[:, 2] - blockers[:, 0]
    normal = np.cross(edge_a, edge_b)
    return np.sum(normal * (np.asarray(receiver) - blockers.mean(axis=1)), axis=1) > 0


def reactor(box_divisions=4, sectors=24, slices=4, radius=0.15):
    """A closed box with a lamp sleeve down its axis: one self-occluding surface set."""
    walls = inward_box(box_divisions)
    sleeve = cylinder_triangles(radius, 0.45, sectors=sectors, slices=slices) + 0.5
    return Surfaces.from_triangles(np.concatenate([walls, sleeve])), len(walls)


def candidate_fraction(surfaces, chunk=2):
    """Share of (receiver, source, blocker) triples a conservative frustum reject keeps.

    Exact-conservative: a blocker is discarded only when all three of its vertices lie outside
    one single plane of the frustum, so nothing that could occlude is ever dropped. The mask is
    frozen geometry built once and off the differentiation path, so the survivors can be
    compacted on the host before any clipping — which is what decides whether the analytic
    treatment is affordable, since the clip itself is far dearer than a ray test.
    """
    v = np.asarray(surfaces.vertices)
    centroid = np.asarray(surfaces.centroid)
    normal = np.asarray(surfaces.normal)
    n = surfaces.n_facets
    kept = 0
    for first in range(0, n, chunk):
        origin = centroid[first : first + chunk]
        relative = v[None, :, :, :] - origin[:, None, None, :]
        middle = relative.mean(axis=2)
        side = np.stack(
            [np.cross(relative[:, :, k, :], relative[:, :, (k + 1) % 3, :]) for k in range(3)],
            axis=2,
        )
        side = side * np.sign(np.einsum("rskd,rsd->rsk", side, middle))[..., None]
        plane = np.cross(
            relative[:, :, 1] - relative[:, :, 0], relative[:, :, 2] - relative[:, :, 0]
        )
        plane = plane * -np.sign(np.einsum("rsd,rsd->rs", plane, middle))[..., None]

        out = np.any(np.all(np.einsum("rskd,rbvd->rskbv", side, relative) < 0.0, axis=-1), axis=2)
        out |= np.all(np.einsum("rsd,rbvd->rsbv", plane, relative) < 0.0, axis=-1)
        out |= np.all(
            np.einsum("rd,rbvd->rbv", normal[first : first + chunk], relative) < 0.0, axis=-1
        )[:, None, :]
        kept += int(np.sum(~out))
    return kept / n**3


def main() -> None:
    started = time.time()
    receiver = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    source = np.array([[-1.0, -1.0, 2.0], [1.4, -0.9, 2.0], [0.1, 1.2, 2.0]])

    print("1. Is the analytic fraction exact? Against brute force, refining the brute force.")
    blocker = np.array([[-0.2, -0.2, 1.0], [0.2, -0.2, 1.0], [0.0, 0.2, 1.0]])
    analytic = float(blocked_fraction(receiver, normal, source, blocker))
    print(f"   analytic (one evaluation): {analytic:.9f}", flush=True)
    for samples in (25_000, 100_000, 400_000, 1_600_000, 6_400_000):
        gap = abs(analytic - sampled_fraction(receiver, normal, source, blocker, samples))
        print(
            f"   {samples:>9,} samples: |diff| {gap:.2e}   sampler floor {1 / np.sqrt(samples):.2e}"
        )
    print("   -- the gap tracks the SAMPLER's floor, so the analytic value is the exact one.\n")

    print("2. Does a blocker tiled into many triangles sum to the whole? (the STL question)")
    for name, one in (
        ("partly covering", np.array([[-3.0, -3.0, 1.0], [3.0, -3.0, 1.0], [0.0, 0.05, 1.0]])),
        ("fully inside", blocker),
    ):
        whole = float(blocked_fraction(receiver, normal, source, one))
        pieces = [one]
        for _level in (1, 2, 3):
            pieces = [child for piece in pieces for child in split_once(piece)]
            total = sum(float(blocked_fraction(receiver, normal, source, p)) for p in pieces)
            print(f"   {name:>16}: 1 -> {len(pieces):3d} triangles, diff {abs(whole - total):.2e}")
    print("   -- exact, because a tiling does not overlap in projection.\n", flush=True)

    print("3. A CLOSED triangulated body: sum over front-facing triangles only.")
    tube = across_the_gap()
    print(f"   {len(tube)} triangles.  receiver x | front-facing sum | dense truth | error")
    for x in (0.0, 0.25, 0.5, 0.8, 1.2):
        r = np.array([x, 0.0, -1.0])
        patch = np.array([[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.0, 0.5, 1.0]])
        front = front_facing(tube, r)
        got = sum(float(blocked_fraction(r, normal, patch, t)) for t in tube[front])
        every = sum(float(blocked_fraction(r, normal, patch, t)) for t in tube)
        truth = sampled_fraction(r, normal, patch, tube)
        print(
            f"   {x:+18.2f} {got:18.6f} {truth:13.6f} {got - truth:+11.2e}"
            f"   (all triangles would give {every:.4f})",
            flush=True,
        )
    print("   -- front-facing is exact; summing every triangle counts the far wall too.\n")

    print("4. How much work is there really? Share of triples a conservative reject keeps.")
    for divisions, sectors in ((2, 12), (3, 16), (4, 24)):
        surfaces, n_wall = reactor(divisions, sectors)
        share = candidate_fraction(surfaces)
        print(
            f"   {surfaces.n_facets:4d} facets ({n_wall} wall + {surfaces.n_facets - n_wall} "
            f"sleeve): {100 * share:.2f}% of n^3 survives",
            flush=True,
        )
    print("   -- and it falls as the mesh refines: a finer pair sweeps a narrower pencil.")
    print(f"\n({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
