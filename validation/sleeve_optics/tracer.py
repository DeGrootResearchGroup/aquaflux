"""A brute-force optical reference for an array of sleeved lamps: forward Monte Carlo, exact cylinders.

Each lamp is an infinitely long Lambertian arc of radius ``arc`` inside a quartz sleeve (air gap out to
``inner``, quartz out to ``outer``), standing in water. Because every surface is an infinite cylinder
along z, the geometry is two-dimensional while every ray keeps its full three-dimensional direction: a
ray is traced in the cross-section, but the distance it travels, the angle it meets each surface at, and
so Snell's law and the Fresnel coefficients, are all taken in three dimensions. The fluence rate is then
per metre of lamp and independent of z.

At every interface the ray is reflected with the unpolarized Fresnel reflectance and refracted otherwise
-- one choice per ray, so a ray's weight changes only by absorption. Total internal reflection is the
reflectance reaching one. A ray reaching any lamp's arc is absorbed there; one reaching the outer wall
(a black circle) leaves. In water the weight decays as ``exp(-a s)``; every water segment deposits what
it loses at a point drawn along it from the truncated exponential, so the absorbed power per pixel,
divided by ``a`` and the pixel's water area, is the fluence rate.

Each ray carries what it has met, so the field is split by path: light that never touched another lamp's
sleeve, light that reflected off one (and never entered it), and light that entered one.

This module imports no aquaflux code: it is the reference the library is checked against.
"""

from __future__ import annotations

import dataclasses

import numpy as np

#: Refractive indices at 254 nm: water (Hale & Querry 1973), fused silica (Malitson 1965), air.
WATER, QUARTZ, AIR = 1.376, 1.5048, 1.0003

#: Path classes, by what the ray had met when it deposited.
OWN, NEIGHBOUR_REFLECTED, NEIGHBOUR_ENTERED = 0, 1, 2
CLASSES = ("own sleeve only", "reflected off another sleeve", "entered another sleeve")

_EPSILON = 1e-10


@dataclasses.dataclass(frozen=True)
class Scene:
    """An array of sleeved lamps in water, inside a black circular wall.

    Attributes
    ----------
    centres : numpy.ndarray, shape ``(k, 2)``
        Lamp axes in the cross-section, metres.
    arc, inner, outer : float
        Radii of the emitting arc, of the sleeve's inner (air side) and outer (water side) surfaces.
    wall : float
        Radius of the black wall, centred on the origin.
    absorption : float
        Napierian absorption coefficient of the water, per metre.
    water, quartz, air : float
        Refractive indices. Setting all three equal removes every interface: straight rays, no
        reflection, the arcs still absorbing.
    neighbours : str
        How a lamp's sleeve meets light from another lamp: ``"optics"`` (reflect and refract),
        ``"invisible"`` (its sleeve is not there, its arc still absorbs) or ``"opaque"`` (its sleeve's
        outer surface absorbs).
    emit_from : str
        ``"arc"`` -- each lamp's arc emits into its air gap; or ``"sleeve"`` -- each lamp's outer sleeve
        surface emits into the water, the lamp's interior never entered, which is how a model that meshes
        the sleeve as the lamp sees it.
    """

    centres: np.ndarray
    arc: float = 7.5e-3
    inner: float = 10.25e-3
    outer: float = 11.5e-3
    wall: float = 0.15
    absorption: float = 5.13
    water: float = WATER
    quartz: float = QUARTZ
    air: float = AIR
    neighbours: str = "optics"
    emit_from: str = "arc"


@dataclasses.dataclass
class Tally:
    """What the rays did, per metre of lamp, for one watt per metre from each lamp."""

    deposited: np.ndarray  # (classes, ny, nx) watts per metre absorbed in each pixel's water
    emitted: float = 0.0
    water: float = 0.0
    arcs: float = 0.0
    sleeves: float = 0.0
    wall: float = 0.0
    lost: float = 0.0


def fresnel(cos_i, n1, n2):
    """Unpolarized Fresnel reflectance and the transmitted cosine, ``(r, cos_t)``; ``r = 1`` past the
    critical angle, where ``cos_t`` is returned as zero."""
    cos_i = np.clip(cos_i, 0.0, 1.0)
    ratio = n1 / n2
    sin_t2 = ratio * ratio * (1.0 - cos_i * cos_i)
    total = sin_t2 >= 1.0
    cos_t = np.sqrt(np.clip(1.0 - sin_t2, 0.0, 1.0))
    rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    r = np.where(total, 1.0, 0.5 * (rs * rs + rp * rp))
    return r, np.where(total, 0.0, cos_t)


def _first_crossing(position, direction, centre, radius, outside):
    """3D path length to a circle along each ray, or ``inf``: entering it from ``outside``, else
    leaving it. ``position (n, 2)``, ``direction (n, 3)``, ``centre (n, 2)``, ``outside (n,)``."""
    planar = direction[:, :2]
    a = np.sum(planar * planar, axis=1)
    offset = position - centre
    b = np.sum(planar * offset, axis=1)
    c = np.sum(offset * offset, axis=1) - radius * radius
    disc = b * b - a * c
    safe = np.where(a > 1e-300, a, 1.0)
    root = np.sqrt(np.clip(disc, 0.0, None))
    near, far = (-b - root) / safe, (-b + root) / safe
    hit = np.where(outside, near, far)
    ok = (a > 1e-300) & (disc > 0.0) & (hit > _EPSILON)
    if np.any(outside):
        ok &= ~outside | (c > 0.0)
    return np.where(ok, hit, np.inf)


def trace(scene: Scene, rays_per_lamp: int, grid: tuple, *, seed=0, batch=200_000, max_events=400):
    """Trace ``rays_per_lamp`` rays from every lamp and return their :class:`Tally`.

    ``grid`` is ``(x0, y0, pixel, nx, ny)``: the lower-left corner, the pixel side, and the counts.
    """
    rng = np.random.default_rng(seed)
    _, _, _, nx, ny = grid
    tally = Tally(deposited=np.zeros((len(CLASSES), ny, nx)))
    centres = np.asarray(scene.centres, dtype=float)
    weight_each = 1.0 / rays_per_lamp
    for lamp in range(len(centres)):
        left = rays_per_lamp
        while left:
            n = min(batch, left)
            left -= n
            _trace_batch(scene, centres, lamp, n, weight_each, rng, tally, grid, max_events)
    return tally


def _emit(scene, centre, n, rng):
    """Lambertian emission from a cylinder about ``centre``: positions, directions, and the radius."""
    radius = scene.arc if scene.emit_from == "arc" else scene.outer
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    normal = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    tangent = np.stack([-np.sin(phi), np.cos(phi)], axis=1)
    u1, u2 = rng.random(n), rng.random(n)
    s, t = np.sqrt(u1), 2.0 * np.pi * u2
    along_normal = np.sqrt(1.0 - u1)
    planar = along_normal[:, None] * normal + (s * np.cos(t))[:, None] * tangent
    direction = np.concatenate([planar, (s * np.sin(t))[:, None]], axis=1)
    return centre + radius * normal, direction


def _seen_from_water(scene, j, lamp):
    """Which of lamp ``j``'s circles a ray in water meets first (0 arc, 2 outer surface), for rays
    emitted by ``lamp``."""
    if scene.emit_from == "sleeve" or j == lamp or scene.neighbours != "invisible":
        return 2
    return 0


def _trace_batch(scene, centres, lamp, n, weight_each, rng, tally, grid, max_events):
    x0, y0, pixel, nx, ny = grid
    position, direction = _emit(scene, centres[lamp], n, rng)
    weight = np.full(n, weight_each)
    tally.emitted += weight.sum()
    # Where each ray is: -1 in water, else 3 * lamp + layer (0 air gap, 1 quartz).
    if scene.emit_from == "arc":
        where = np.full(n, 3 * lamp)
    else:
        where = np.full(n, -1)
    met = np.zeros(n, dtype=np.int8)
    home = np.full(n, lamp)
    alive = np.ones(n, dtype=bool)
    k = len(centres)
    radii = (scene.arc, scene.inner, scene.outer)
    for _ in range(max_events):
        idx = np.flatnonzero(alive)
        if not idx.size:
            break
        p, d, w, at = position[idx], direction[idx], weight[idx], where[idx]
        m = idx.size
        best = np.full(m, np.inf)
        surface = np.full(m, -1)  # 3 * lamp + circle (0 arc, 1 inner, 2 outer), or -2 for the wall
        in_water = at < 0
        # The wall, from inside, for rays in water.
        t = _first_crossing(p, d, np.zeros((m, 2)), scene.wall, np.zeros(m, dtype=bool))
        take = in_water & (t < best)
        best, surface = np.where(take, t, best), np.where(take, -2, surface)
        for j in range(k):
            c = np.broadcast_to(centres[j], (m, 2))
            mine = ~in_water & (at // 3 == j)
            layer = at % 3
            circle = _seen_from_water(scene, j, lamp)
            t = _first_crossing(p, d, c, radii[circle], np.ones(m, dtype=bool))
            take = in_water & (t < best)
            best = np.where(take, t, best)
            surface = np.where(take, 3 * j + circle, surface)
            # Inside lamp j: the air gap meets the arc (entering) or the inner surface (leaving);
            # the quartz meets the inner surface (entering) or the outer one (leaving).
            for circle, from_outside, in_layer in (
                (0, True, 0),
                (1, False, 0),
                (1, True, 1),
                (2, False, 1),
            ):
                sel = mine & (layer == in_layer)
                if not np.any(sel):
                    continue
                t = _first_crossing(p, d, c, radii[circle], np.full(m, from_outside))
                take = sel & (t < best)
                best = np.where(take, t, best)
                surface = np.where(take, 3 * j + circle, surface)
        # Water segments deposit what they lose, at a point drawn along them.
        a = scene.absorption
        wet = in_water
        if np.any(wet):
            length = best[wet]
            survive = np.exp(-a * np.where(np.isfinite(length), length, np.inf))
            lost = w[wet] * (1.0 - survive)
            u = rng.random(int(wet.sum()))
            s = -np.log1p(-u * (1.0 - survive)) / a
            point = p[wet] + s[:, None] * d[wet, :2]
            ix = np.floor((point[:, 0] - x0) / pixel).astype(np.int64)
            iy = np.floor((point[:, 1] - y0) / pixel).astype(np.int64)
            inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            cls = met[idx][wet]
            np.add.at(tally.deposited, (cls[inside], iy[inside], ix[inside]), lost[inside])
            tally.water += lost.sum()
            w = w.copy()
            w[wet] = w[wet] * survive
        # Move to the surface met.
        finite = np.isfinite(best)
        p = p + np.where(finite, best, 0.0)[:, None] * d[:, :2]
        stop = ~finite | (surface == -2)
        tally.wall += w[(surface == -2)].sum()
        lamp_of = np.where(surface >= 0, surface // 3, -1)
        circle_of = np.where(surface >= 0, surface % 3, -1)
        arc_hit = circle_of == 0
        tally.arcs += w[arc_hit].sum()
        stop |= arc_hit
        foreign = (lamp_of >= 0) & (lamp_of != home[idx])
        absorbing_sleeve = (circle_of == 2) & in_water & foreign & (scene.neighbours == "opaque")
        if scene.emit_from == "sleeve":
            absorbing_sleeve |= (circle_of == 2) & in_water
        tally.sleeves += w[absorbing_sleeve].sum()
        stop |= absorbing_sleeve
        # Interfaces: which media meet there.
        go = ~stop
        n1 = np.where(in_water, scene.water, np.where(at % 3 == 0, scene.air, scene.quartz))
        n2 = np.where(
            circle_of == 2,
            np.where(in_water, scene.quartz, scene.water),
            np.where(at % 3 == 0, scene.quartz, scene.air),
        )
        centre = centres[np.clip(lamp_of, 0, k - 1)]
        normal2 = (p - centre) / np.where(circle_of == 2, scene.outer, scene.inner)[:, None]
        normal = np.concatenate([normal2, np.zeros((m, 1))], axis=1)
        cos_n = np.sum(d * normal, axis=1)
        facing = np.where((cos_n < 0)[:, None], normal, -normal)  # against the ray
        cos_i = np.abs(cos_n)
        r, cos_t = fresnel(cos_i, n1, n2)
        reflect = rng.random(m) < r
        reflected = d + 2.0 * cos_i[:, None] * facing
        ratio = (n1 / n2)[:, None]
        refracted = ratio * d + (ratio[:, 0] * cos_i - cos_t)[:, None] * facing
        new_d = np.where(reflect[:, None], reflected, refracted)
        new_d /= np.linalg.norm(new_d, axis=1, keepdims=True)
        # The medium after a transmission.
        crossed = go & ~reflect
        new_at = at.copy()
        out_of_quartz_outward = crossed & (circle_of == 2) & ~in_water
        into_quartz_from_water = crossed & (circle_of == 2) & in_water
        into_quartz_from_air = crossed & (circle_of == 1) & (at % 3 == 0)
        into_air_from_quartz = crossed & (circle_of == 1) & (at % 3 == 1)
        new_at = np.where(out_of_quartz_outward, -1, new_at)
        new_at = np.where(into_quartz_from_water, 3 * lamp_of + 1, new_at)
        new_at = np.where(into_quartz_from_air, 3 * lamp_of + 1, new_at)
        new_at = np.where(into_air_from_quartz, 3 * lamp_of + 0, new_at)
        # What a ray from water has now met at a foreign sleeve.
        at_foreign = go & in_water & foreign & (circle_of == 2)
        new_met = met[idx].copy()
        new_met = np.where(at_foreign & reflect, np.maximum(new_met, NEIGHBOUR_REFLECTED), new_met)
        new_met = np.where(at_foreign & ~reflect, NEIGHBOUR_ENTERED, new_met)
        # Index-matched interfaces (equal indices) transmit straight on, which the formulas give.
        position[idx] = p
        direction[idx] = np.where(go[:, None], new_d, d)
        weight[idx] = w
        where[idx] = new_at
        met[idx] = new_met
        alive[idx[stop]] = False
        # A ray with almost no weight left is ended by roulette, keeping the expectation.
        faint = alive & (weight < 1e-6 * weight_each)
        if np.any(faint):
            keep = rng.random(int(faint.sum())) < 0.1
            ids = np.flatnonzero(faint)
            weight[ids[keep]] *= 10.0
            alive[ids[~keep]] = False
    tally.lost += weight[alive].sum()


def water_fraction(scene: Scene, grid: tuple, samples: int = 8) -> np.ndarray:
    """The share of each pixel's area that is water, from ``samples x samples`` points per pixel."""
    x0, y0, pixel, nx, ny = grid
    offsets = (np.arange(samples) + 0.5) / samples
    fraction = np.zeros((ny, nx))
    xs = x0 + (np.arange(nx)[None, :] + offsets[:, None]) * pixel  # (samples, nx)
    ys = y0 + (np.arange(ny)[None, :] + offsets[:, None]) * pixel  # (samples, ny)
    for sy in range(samples):
        for sx in range(samples):
            x = xs[sx][None, :]
            y = ys[sy][:, None]
            wet = (x * x + y * y) < scene.wall**2
            for c in scene.centres:
                wet &= (x - c[0]) ** 2 + (y - c[1]) ** 2 > scene.outer**2
            fraction += wet
    return fraction / samples**2


def fluence(scene: Scene, tally: Tally, grid: tuple, fraction: np.ndarray) -> np.ndarray:
    """Fluence rate per class, W/m^2 for one watt per metre from each lamp, ``(classes, ny, nx)``;
    NaN where a pixel holds less than a quarter of water."""
    pixel = grid[2]
    area = fraction * pixel * pixel
    with np.errstate(invalid="ignore", divide="ignore"):
        g = tally.deposited / (scene.absorption * area)
    return np.where(fraction >= 0.25, g, np.nan)
