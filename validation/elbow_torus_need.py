"""Does a pipe elbow need a torus, or is a chain of cylinders enough to shadow it? (#504)

An elbow's fluid is a quarter torus, and a torus is the one primitive the solids package does not
have: a line can pass through its tube twice, and the ray test is a quartic. Before the quartic is
written, this asks whether it is needed -- whether describing the elbow as a chain of ``N``
cylinders, which :class:`~aquaflux.solids.Outside` already takes, shadows the light the same way.

**The reference does not need the quartic either.** A pipe is everything within its radius of its
centreline, so the fluid's signed distance is the distance to that curve -- two segments and a
quarter circle -- less the radius, cut flat at the two far ends. That function has slope at most
one everywhere, so a segment can be walked from its start in steps of the distance to the wall
without ever stepping through it (sphere tracing), stopping where the wall is reached. It is a
different algorithm from the interval arithmetic every arm uses, so agreement is evidence rather
than an identity, and it is checked twice before it is trusted:

- its zero set against an independent membership test (which part of the pipe a point is in, then
  whether it is within the radius of that part's axis), on a million random points;
- the same walk run over the *signed distance of a mitred chain*, against
  :class:`~aquaflux.solids.Outside` of the same chain built from primitives, which is exact for
  that geometry -- so a defect in the walk shows up as pairs masked differently where neither side
  has any excuse.

A segment that comes within ``_TOUCH`` of the wall without leaving -- an internal tangency the
walk cannot tell from a crossing by its steps alone -- is probed a short distance ahead, and is
counted. Those counts are reported, so how many pairs the reference decides at a micron is
visible beside how many pairs the arms dispute.

**The arms.** The elbow's centreline is replaced by ``N`` chords whose ends lie on the bend's arc,
and the fluid by one cylinder per chord, two ways:

- *mitred* -- each cylinder cut by the planes bisecting its joints, which is exactly a fabricated
  ``N``-piece mitre bend;
- *overlapping* -- plain cylinders, each carried past its joints far enough to close the outer
  corner, which is what is quickest to write. In the bend's plane that union is the mitre bend
  exactly -- what one cylinder carries past a joint on the inside of the turn lies in the next one
  -- and off that plane its round end reaches a little past the mitre on the outside, so the two
  arms differ by a sliver and are expected to read alike.

The straight legs are the same in every arm and in the reference, so every difference is the
elbow's.

**What is reported, per bend and arm.** Pairs masked differently from the reference; receivers that
lie outside the arm's fluid (a mesh of the real elbow has cells there, which ``Outside`` refuses as
embedded in the wall, or reads as dark once given a tolerance); and the fluence rate against the
reference -- per cell, median, 99th percentile and worst over the cells the reference lights, and as
a volume mean per region, which is what a dose or an absorbed power integrates. The straight leg
round the lamp is the control: every segment there lies in one convex cylinder, so any difference
in it is a defect, not a discretization.

Run with ``validation/run_case.sh validation/elbow_torus_need.py``. ``ELBOW_RECEIVERS`` sets the
receivers per region, ``ELBOW_PIECES`` the chord counts (comma separated).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import aquaflux  # noqa: E402,F401  (enables x64)
import jax.numpy as jnp  # noqa: E402
import numba  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.radiation import (  # noqa: E402
    NoOcclusion,
    Surfaces,
    UniformAbsorption,
    absorption_from_uvt,
    build_visibility,
    direct_fluence_rate,
)
from aquaflux.solids import Body, Cylinder, HalfSpace, Intersection, Outside  # noqa: E402

OUT = HERE / "elbow_torus_need"

#: Pipe radius: a DN100 line.
RADIUS = 0.05
#: Straight length either side of the bend.
LEG = 0.5
#: Bend radius of the centreline, by name: a short-radius elbow (one diameter) and a long-radius
#: one (one and a half), the two standard fittings.
BENDS = {"short radius, R = 1.0 D": 2.0 * RADIUS, "long radius, R = 1.5 D": 3.0 * RADIUS}
#: The lamp: a sleeve-sized emitting tube on the upstream leg's axis, ending short of the bend.
LAMP_RADIUS = 0.0115
LAMP_START, LAMP_END = -0.45, -0.05
LAMP_SECTORS, LAMP_SLICES = 16, 48
EXITANCE = 696.42
#: 95 % transmittance through 1 cm: light reaches far round a bend in clear water, which is where
#: the elbow's description has the most to do.
WATER = UniformAbsorption(absorption_from_uvt(95.0))
#: How far down the outlet leg receivers are drawn. Beyond about this the outlet sees no lamp at
#: all round either bend, so cells there would only dilute the shares below with zeros.
OUTLET_REACH = 0.30

RECEIVERS = int(os.environ.get("ELBOW_RECEIVERS", 3_000))
PIECES = tuple(int(n) for n in os.environ.get("ELBOW_PIECES", "1,2,4,8,12,16,24,32,48,64").split(","))

#: Joints are overlapped by this much, so neighbouring regions share a sliver rather than meeting
#: on one plane, where two separately rounded intersections can leave a gap of a rounding.
_OVERLAP = 1e-7
#: The walk stops to look ahead once the wall is nearer than this.
_TOUCH = 1e-9
#: How far ahead it looks, in steps of ``_PROBE_STEP``, to tell a crossing from a tangency.
_PROBE_STEP, _PROBES = 1e-7, 20
_MAX_STEPS = 200_000

#: The faceting error #501 measured on the Sozzi pipes -- an STL's 15-sided inscribed polygon in
#: place of a round pipe -- and judged worth removing. The yardstick for what "enough" means.
FACETING_YARDSTICK = {"median": 0.012, "p99": 0.089}


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------------------------
# The reference: the pipe as everything within its radius of its centreline
# ---------------------------------------------------------------------------------------------
#
# The bend turns the flow from +x to +z in the plane y = 0. The inlet leg's axis runs from
# (-LEG, 0, 0) to the origin; the bend's centreline is the quarter circle about (0, 0, R) from the
# origin to (R, 0, R); the outlet leg's axis runs from there to (R, 0, R + LEG).


@numba.njit(cache=False)
def _pipe_distance(x, y, z, bend, radius, leg):
    """Signed distance to the pipe's wall: negative in the water, at most one in slope."""
    along = min(max(x, -leg), 0.0)
    inlet = np.sqrt((x - along) ** 2 + y * y + z * z)
    qz = z - bend
    if x >= 0.0 and qz <= 0.0:
        elbow = np.sqrt((np.sqrt(x * x + qz * qz) - bend) ** 2 + y * y)
    else:
        elbow = np.inf  # the arc's nearest point is an end, which the legs already hold
    up = min(max(z, bend), bend + leg)
    outlet = np.sqrt((x - bend) ** 2 + y * y + (z - up) ** 2)
    return max(min(inlet, elbow, outlet) - radius, -(x + leg), z - (bend + leg))


@numba.njit(cache=False)
def _regions_distance(x, y, z, cylinders, planes):
    """Signed distance of a union of capped cylinders each cut by half-spaces, at most one in slope.

    ``cylinders`` rows are ``centre (3), axis (3), radius, half_length``; ``planes`` is
    ``(n_regions, n_planes, 6)``, rows ``point (3), outward normal (3)``, a zero normal unused.
    """
    best = np.inf
    for k in range(cylinders.shape[0]):
        cx, cy, cz = x - cylinders[k, 0], y - cylinders[k, 1], z - cylinders[k, 2]
        ax, ay, az = cylinders[k, 3], cylinders[k, 4], cylinders[k, 5]
        axial = cx * ax + cy * ay + cz * az
        radial = np.sqrt(max(cx * cx + cy * cy + cz * cz - axial * axial, 0.0))
        value = max(radial - cylinders[k, 6], abs(axial) - cylinders[k, 7])
        for j in range(planes.shape[1]):
            nx, ny, nz = planes[k, j, 3], planes[k, j, 4], planes[k, j, 5]
            if nx == 0.0 and ny == 0.0 and nz == 0.0:
                continue
            value = max(
                value,
                (x - planes[k, j, 0]) * nx
                + (y - planes[k, j, 1]) * ny
                + (z - planes[k, j, 2]) * nz,
            )
        best = min(best, value)
    return best


@numba.njit(cache=False)
def _distance(kind, x, y, z, shape, cylinders, planes):
    if kind == 0:
        return _pipe_distance(x, y, z, shape[0], shape[1], shape[2])
    return _regions_distance(x, y, z, cylinders, planes)


@numba.njit(parallel=True, cache=False)
def _distances(points, kind, shape, cylinders, planes, out):
    for i in numba.prange(points.shape[0]):
        out[i] = _distance(kind, points[i, 0], points[i, 1], points[i, 2], shape, cylinders, planes)


@numba.njit(parallel=True, cache=False)
def _walk(origin, target, near, kind, shape, cylinders, planes, blocked, grazed, unresolved):
    """Walk each segment by the distance to the wall; blocked where it leaves the water."""
    for i in numba.prange(origin.shape[0]):
        dx = target[i, 0] - origin[i, 0]
        dy = target[i, 1] - origin[i, 1]
        dz = target[i, 2] - origin[i, 2]
        length = np.sqrt(dx * dx + dy * dy + dz * dz)
        t = near[i] / length
        end = 1.0 - t
        hit = False
        done = False
        for _ in range(_MAX_STEPS):
            if t >= end:
                done = True
                break
            value = _distance(
                kind, origin[i, 0] + t * dx, origin[i, 1] + t * dy, origin[i, 2] + t * dz,
                shape, cylinders, planes,
            )  # fmt: skip
            if value > 0.0:
                hit = True
                done = True
                break
            if -value < _TOUCH:
                for j in range(1, _PROBES + 1):
                    ahead = t + j * _PROBE_STEP / length
                    if ahead >= end:
                        break
                    if (
                        _distance(
                            kind,
                            origin[i, 0] + ahead * dx,
                            origin[i, 1] + ahead * dy,
                            origin[i, 2] + ahead * dz,
                            shape,
                            cylinders,
                            planes,
                        )  # fmt: skip
                        > 0.0
                    ):
                        hit = True
                        break
                if hit:
                    done = True
                    break
                grazed[i] += 1
                t += _PROBES * _PROBE_STEP / length
                continue
            t += -value / length
        blocked[i] = hit
        unresolved[i] = not done


class Walked(Body):
    """A pipe's water, its wall found by walking each segment. Host code: a search that stops."""

    kind: int
    shape: np.ndarray
    cylinders: np.ndarray
    planes: np.ndarray
    tolerance: float
    counts: dict

    def __init__(self, kind, shape=(0.0, 0.0, 0.0), cylinders=None, planes=None, tolerance=0.0):
        self.kind = kind
        self.shape = np.asarray(shape, dtype=float)
        self.cylinders = np.zeros((0, 8)) if cylinders is None else np.asarray(cylinders, float)
        self.planes = np.zeros((0, 0, 6)) if planes is None else np.asarray(planes, float)
        self.tolerance = tolerance
        self.counts = {"segments": 0, "grazed": 0, "unresolved": 0}

    def signed_distance(self, position):
        points = np.ascontiguousarray(np.asarray(position, dtype=float).reshape(-1, 3))
        values = np.empty(len(points))
        _distances(points, self.kind, self.shape, self.cylinders, self.planes, values)
        return values.reshape(np.shape(position)[:-1])

    def contains(self, position):
        return jnp.asarray(self.signed_distance(position) > self.tolerance)

    def blocks(self, origin, target, min_distance):
        origin, target = np.broadcast_arrays(np.asarray(origin, float), np.asarray(target, float))
        shape = origin.shape[:-1]
        near = np.broadcast_to(np.asarray(min_distance, float), shape).reshape(-1).copy()
        origin, target = origin.reshape(-1, 3).copy(), target.reshape(-1, 3).copy()
        blocked = np.zeros(len(origin), dtype=np.bool_)
        grazed = np.zeros(len(origin), dtype=np.int64)
        unresolved = np.zeros(len(origin), dtype=np.bool_)
        _walk(
            origin, target, near, self.kind, self.shape, self.cylinders, self.planes,
            blocked, grazed, unresolved,
        )  # fmt: skip
        self.counts["segments"] += len(origin)
        self.counts["grazed"] += int((grazed > 0).sum())
        self.counts["unresolved"] += int(unresolved.sum())
        return jnp.asarray(blocked.reshape(shape))


def membership(points, bend):
    """Which part of the pipe each point is in, and whether it is in the water: an independent
    reading of the same solid, written from the parts rather than from a distance."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    inlet = (x >= -LEG) & (x <= 0.0) & (y * y + z * z <= RADIUS**2)
    qz = z - bend
    elbow = (x >= 0.0) & (qz <= 0.0) & ((np.hypot(x, qz) - bend) ** 2 + y * y <= RADIUS**2)
    outlet = (z >= bend) & (z <= bend + LEG) & ((x - bend) ** 2 + y * y <= RADIUS**2)
    return inlet, elbow, outlet


# ---------------------------------------------------------------------------------------------
# The arms: the elbow as a chain of cylinders
# ---------------------------------------------------------------------------------------------


def chain(bend, pieces):
    """The centreline's corners and the chords between them, inlet axis to outlet axis.

    Returns the joints (``pieces + 1`` points on the arc) and the unit direction of every straight
    run: the inlet leg, each chord, the outlet leg.
    """
    angles = np.linspace(0.0, np.pi / 2.0, pieces + 1)
    joints = np.column_stack(
        [bend * np.sin(angles), np.zeros_like(angles), bend * (1 - np.cos(angles))]
    )
    chords = np.diff(joints, axis=0)
    chords /= np.linalg.norm(chords, axis=1, keepdims=True)
    runs = np.vstack([[1.0, 0.0, 0.0], chords, [0.0, 0.0, 1.0]])
    return joints, runs


def _cylinder_row(start, end, extra_start=0.0, extra_end=0.0):
    axis = (end - start) / np.linalg.norm(end - start)
    first, last = start - extra_start * axis, end + extra_end * axis
    return np.concatenate(
        [(first + last) / 2.0, axis, [RADIUS, np.linalg.norm(last - first) / 2.0]]
    )


def mitred(bend, pieces):
    """Each straight run cut by the planes bisecting its two joints: an ``N``-piece mitre bend."""
    joints, runs = chain(bend, pieces)
    starts = np.vstack([[-LEG, 0.0, 0.0], joints])
    ends = np.vstack([joints, [bend, 0.0, bend + LEG]])
    # A mitre plane's normal is the sum of the two directions it joins; a run is kept on the side
    # after its first joint and before its last, each pushed out by the overlap.
    normals = runs[:-1] + runs[1:]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    cylinders, planes = [], []
    for k in range(len(runs)):
        reach = RADIUS + _OVERLAP  # past either joint, so the cut and not the cap ends the run
        cylinders.append(
            _cylinder_row(
                starts[k], ends[k], reach if k > 0 else 0.0, reach if k < len(runs) - 1 else 0.0
            )
        )
        cut = np.zeros((2, 6))
        if k > 0:
            cut[0] = np.concatenate([joints[k - 1] - _OVERLAP * normals[k - 1], -normals[k - 1]])
        if k < len(runs) - 1:
            cut[1] = np.concatenate([joints[k] + _OVERLAP * normals[k], normals[k]])
        planes.append(cut)
    return np.asarray(cylinders), np.asarray(planes)


def overlapping(bend, pieces):
    """Plain cylinders, each carried past a joint by ``radius * tan(turn / 2)``: far enough to close
    the joint's outer corner in the bend's plane, and a little past the mitre off it."""
    joints, runs = chain(bend, pieces)
    starts = np.vstack([[-LEG, 0.0, 0.0], joints])
    ends = np.vstack([joints, [bend, 0.0, bend + LEG]])
    turns = np.arccos(np.clip(np.einsum("ij,ij->i", runs[:-1], runs[1:]), -1.0, 1.0))
    reach = RADIUS * np.tan(turns / 2.0) + _OVERLAP
    cylinders = [
        _cylinder_row(
            starts[k],
            ends[k],
            reach[k - 1] if k > 0 else 0.0,
            reach[k] if k < len(runs) - 1 else 0.0,
        )
        for k in range(len(runs))
    ]
    return np.asarray(cylinders), np.zeros((len(runs), 0, 6))


def as_primitives(cylinders, planes, tolerance):
    """The same regions as :class:`~aquaflux.solids.Outside` of primitives."""
    regions = []
    for row, cuts in zip(cylinders, planes):
        body = Cylinder(centre=row[:3], axis=row[3:6], radius=row[6], half_length=row[7])
        halves = [HalfSpace(point=c[:3], normal=c[3:]) for c in cuts if np.any(c[3:] != 0.0)]
        regions.append(Intersection(body, *halves) if halves else body)
    return Outside(*regions, tolerance=tolerance)


# ---------------------------------------------------------------------------------------------
# The scene
# ---------------------------------------------------------------------------------------------


def lamp() -> Surfaces:
    """An emitting tube on the inlet's axis, wound so its normals face the water."""
    angles = np.linspace(0.0, 2.0 * np.pi, LAMP_SECTORS + 1)
    rings = np.linspace(LAMP_START, LAMP_END, LAMP_SLICES + 1)
    ring = np.column_stack(
        [np.zeros(LAMP_SECTORS + 1), LAMP_RADIUS * np.cos(angles), LAMP_RADIUS * np.sin(angles)]
    )
    ring[-1] = ring[0]  # close the tube on one vertex table, not on trigonometry evaluated twice
    triangles = []
    for a, b in zip(rings[:-1], rings[1:]):
        for j in range(LAMP_SECTORS):
            p00, p01 = ring[j] + [a, 0, 0], ring[j + 1] + [a, 0, 0]
            p10, p11 = ring[j] + [b, 0, 0], ring[j + 1] + [b, 0, 0]
            triangles += [[p00, p10, p11], [p00, p11, p01]]
    vertices = np.asarray(triangles)
    surfaces = Surfaces.from_triangles(vertices, emission=EXITANCE)
    centroid, normal = np.asarray(surfaces.centroid), np.asarray(surfaces.normal)
    outward = np.einsum("ij,ij->i", normal, centroid * [0.0, 1.0, 1.0]) > 0.0
    if not outward.all():
        surfaces = Surfaces.from_triangles(vertices[:, ::-1, :], emission=EXITANCE)
    return surfaces


def receivers(rng, bend):
    """``RECEIVERS`` points in each of three regions of the water, drawn uniformly by volume.

    The inlet round the lamp is the control; the bend and the first ``OUTLET_REACH`` of the outlet
    are where the elbow's description decides what is lit. The control stops a radius short of
    the bend: a mitre cut through the first joint leans into the inlet's last radius, so cells
    there are part of what is being measured, and they are left out of both rather than counted
    in the control.
    """
    boxes = {
        "inlet": ([LAMP_START, -RADIUS, -RADIUS], [-RADIUS, RADIUS, RADIUS]),
        "bend": ([0.0, -RADIUS, -RADIUS], [bend + RADIUS, RADIUS, bend]),
        "outlet": ([bend - RADIUS, -RADIUS, bend], [bend + RADIUS, RADIUS, bend + OUTLET_REACH]),
    }
    regions = {}
    for index, (name, (low, high)) in enumerate(boxes.items()):
        kept = []
        while sum(len(k) for k in kept) < RECEIVERS:
            trial = rng.uniform(low, high, (8 * RECEIVERS, 3))
            parts = membership(trial, bend)
            in_lamp = (trial[:, 0] >= LAMP_START - 1e-3) & (
                trial[:, 1] ** 2 + trial[:, 2] ** 2 <= (LAMP_RADIUS + 1e-4) ** 2
            )
            kept.append(trial[parts[index] & ~in_lamp])
        regions[name] = np.concatenate(kept)[:RECEIVERS]
    return regions


def mask_and_field(surfaces, points, body):
    started = time.perf_counter()
    visibility = build_visibility([body], surfaces, points, self_occlusion=NoOcclusion())
    blocked = np.asarray(visibility.blocked[0])
    built = time.perf_counter() - started
    field = np.asarray(
        direct_fluence_rate(surfaces, jnp.asarray(points), absorption=WATER, visibility=visibility)
    )
    return blocked, field, built


def differences(field, reference, blocked, reference_blocked, regions, outside):
    """The arm against the reference, per region, over the cells the reference lights."""
    result = {}
    start = 0
    for name, block in regions.items():
        rows = slice(start, start + len(block))
        start += len(block)
        g, g_ref = field[rows], reference[rows]
        lit = g_ref > 0.0
        relative = np.abs(g[lit] - g_ref[lit]) / g_ref[lit]
        inside = ~outside[rows]
        inside_lit = lit & inside
        relative_inside = np.abs(g[inside_lit] - g_ref[inside_lit]) / g_ref[inside_lit]
        result[name] = {
            "cells": len(block),
            "cells_lit_by_reference": int(lit.sum()),
            "cells_outside_the_arm_s_water": int((~inside).sum()),
            "cells_lit_by_arm_but_dark_in_reference": int(((g > 0.0) & ~lit).sum()),
            "pairs_masked_differently": int((blocked[rows] != reference_blocked[rows]).sum()),
            "relative_difference_lit": _stats(relative),
            "relative_difference_lit_inside_the_arm_s_water": _stats(relative_inside),
            "volume_mean_G_relative_difference": float(g.mean() / g_ref.mean() - 1.0)
            if g_ref.mean() > 0
            else None,
        }
    return result


def _stats(values):
    if len(values) == 0:
        return None
    return {
        "median": float(np.median(values)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


# ---------------------------------------------------------------------------------------------
# Checks on the reference before it is used
# ---------------------------------------------------------------------------------------------


def check_zero_set(rng, bend):
    """The walk's distance against the parts read separately, on a million points."""
    points = rng.uniform(
        [-LEG - 0.01, -0.06, -0.06], [bend + 0.06, 0.06, bend + LEG + 0.01], (1_000_000, 3)
    )
    inside = np.any(np.stack(membership(points, bend)), axis=0)
    distance = Walked(0, (bend, RADIUS, LEG)).signed_distance(points)
    disagree = inside != (distance <= 0.0)
    return {
        "points": len(points),
        "inside": int(inside.sum()),
        "disagree": int(disagree.sum()),
        "disagree_further_than_1e-12_from_the_wall": int(
            (disagree & (np.abs(distance) > 1e-12)).sum()
        ),
    }


def check_walk(surfaces, points, bend, pieces=4):
    """The walk over a mitred chain against ``Outside`` of the same chain, which is exact for it."""
    cylinders, planes = mitred(bend, pieces)
    walked = Walked(1, cylinders=cylinders, planes=planes, tolerance=RADIUS)
    exact = as_primitives(cylinders, planes, tolerance=RADIUS)
    a = np.asarray(
        build_visibility([walked], surfaces, points, self_occlusion=NoOcclusion()).blocked[0]
    )
    b = np.asarray(
        build_visibility([exact], surfaces, points, self_occlusion=NoOcclusion()).blocked[0]
    )
    return {
        "pieces": pieces,
        "pairs": int(a.size),
        "blocked": int(b.sum()),
        "pairs_masked_differently": int((a != b).sum()),
        **{f"walk_{k}": v for k, v in walked.counts.items()},
    }


def main() -> None:
    rng = np.random.default_rng(0)
    surfaces = lamp()
    _say(
        f"lamp: {surfaces.n_facets} facets; pipe radius {RADIUS} m; water "
        f"{float(WATER.coefficient):.3f} /m; {RECEIVERS} receivers per region; pieces {PIECES}"
    )
    summary = {
        "configuration": {
            "pipe_radius": RADIUS,
            "leg": LEG,
            "lamp": {
                "radius": LAMP_RADIUS,
                "from": LAMP_START,
                "to": LAMP_END,
                "facets": int(surfaces.n_facets),
                "exitance": EXITANCE,
            },
            "absorption_per_m": float(WATER.coefficient),
            "receivers_per_region": RECEIVERS,
            "outlet_reach": OUTLET_REACH,
            "faceting_yardstick_from_501": FACETING_YARDSTICK,
        },
        "bends": {},
    }
    for bend_name, bend in BENDS.items():
        _say(f"== {bend_name} (R = {bend} m)")
        regions = receivers(rng, bend)
        points = np.concatenate(list(regions.values()))
        result = {"bend_radius": bend, "zero_set": check_zero_set(rng, bend)}
        _say(f"  zero set: {result['zero_set']}")
        sample = points[rng.choice(len(points), size=min(len(points), 1_500), replace=False)]
        result["walk_check"] = check_walk(surfaces, sample, bend)
        _say(f"  walk against Outside of a 4-piece mitre: {result['walk_check']}")

        reference = Walked(0, (bend, RADIUS, LEG))
        reference_blocked, reference_field, seconds = mask_and_field(surfaces, points, reference)
        rays = reference_blocked.size
        result["reference"] = {
            "rays": rays,
            "seconds": round(seconds, 1),
            "blocked_share": float(reference_blocked.mean()),
            **reference.counts,
        }
        _say(f"  reference: {result['reference']}")

        result["arms"] = {}
        for style, builder in (("mitred", mitred), ("overlapping", overlapping)):
            for pieces in PIECES:
                cylinders, planes = builder(bend, pieces)
                water = as_primitives(cylinders, planes, tolerance=1.0)
                outside = np.asarray(water.fluid.signed_distance(jnp.asarray(points))) > 0.0
                depth = np.asarray(water.fluid.signed_distance(jnp.asarray(points)))
                blocked, field, built = mask_and_field(surfaces, points, water)
                arm = {
                    "regions": len(water.regions),
                    "mask_seconds": round(built, 2),
                    "rays_per_second": round(rays / built),
                    "deepest_receiver_outside_the_water_m": float(depth.max())
                    if outside.any()
                    else 0.0,
                    "by_region": differences(
                        field, reference_field, blocked, reference_blocked, regions, outside
                    ),
                }
                result["arms"][f"{style} {pieces}"] = arm
                bend_part, out_part = arm["by_region"]["bend"], arm["by_region"]["outlet"]
                _say(
                    f"  {style:>11} {pieces:>2}: pairs differ bend {bend_part['pairs_masked_differently']:>8} "
                    f"outlet {out_part['pairs_masked_differently']:>8} "
                    f"inlet {arm['by_region']['inlet']['pairs_masked_differently']}; "
                    f"outside {bend_part['cells_outside_the_arm_s_water'] + out_part['cells_outside_the_arm_s_water']}; "
                    f"G bend {_brief(bend_part)} outlet {_brief(out_part)}; {built:.1f} s"
                )
        summary["bends"][bend_name] = result
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _say(f"wrote {OUT / 'summary.json'}")


def _brief(part):
    stats = part["relative_difference_lit"]
    mean = part["volume_mean_G_relative_difference"]
    if stats is None:
        return "unlit"
    return f"med {stats['median']:.2%} p99 {stats['p99']:.2%} mean {mean:+.2%}"


if __name__ == "__main__":
    main()
    sys.exit(0)
