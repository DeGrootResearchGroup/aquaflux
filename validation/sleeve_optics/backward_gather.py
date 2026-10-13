"""A deterministic backward gather through infinite coaxial sleeves, split by the route light took.

From a point in the water, rays are cast over the sphere (azimuth x axial direction cosine) and
followed on their transmitted branch only -- no reflected branch, total internal reflection lost --
weighted by each Fresnel transmittance and the water's absorption, until one meets a lamp's arc, whose
basic radiance (radiance over the index squared, conserved along a ray) it then carries. Each arrival
is classed by the sleeves it passed through before reaching its source's: none, one through its quartz
wall alone, one through its air gap too, or two and more. That split is what the tracer cannot give at
a point and what tells a missing route from a missing amount.

The scene is ``check_array.py``'s four lamps, with the sleeves infinite (the scene's are 3 m long),
each lamp emitting 1 W per metre. It is independent of aquaflux: it imports only the tracer's indices
and Fresnel coefficient.

Run: ``python backward_gather.py [azimuths] [axial cosines]`` (defaults 2^17 and 160, under a minute a
point on four cores); results are printed and saved per point under ``work/``.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tracer import AIR, QUARTZ, WATER, fresnel  # noqa: E402

ARC, INNER, OUTER, WALL = 7.5e-3, 10.25e-3, 11.5e-3, 0.1
A = -np.log(0.95) / 0.01
CENTRES = 0.025 * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
L_ARC = 1.0 / (2 * np.pi * ARC) / np.pi  # W/m^2/sr in air, 1 W/m per lamp
RADII = (ARC, INNER, OUTER)


def crossing(p, d, c, r, outside):
    pl = d[:, :2]
    a = np.sum(pl * pl, 1)
    o = p - c
    b = np.sum(pl * o, 1)
    cc = np.sum(o * o, 1) - r * r
    disc = b * b - a * cc
    s = np.where(a > 1e-300, a, 1.0)
    root = np.sqrt(np.clip(disc, 0, None))
    t = np.where(outside, (-b - root) / s, (-b + root) / s)
    ok = (a > 1e-300) & (disc > 0) & (t > 1e-12)
    ok &= ~outside | (cc > 0)
    return np.where(ok, t, np.inf)


def gather(P, nphi=2**17, nmu=160):
    """Return dict class -> G (W/m^2 per W/m), and per-(class, mu) arrays."""
    phis = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
    mus = (np.arange(nmu) + 0.5) / nmu  # mu in (0,1); doubled for -mu
    dphi, dmu = 2 * np.pi / nphi, 1.0 / nmu
    # classes: 0 own only, 1 one pass quartz-only, 2 one pass via gap, 3 two+ passes
    G = np.zeros((4, nmu))
    for im, mu in enumerate(mus):
        st = np.sqrt(1 - mu * mu)
        d = np.stack([st * np.cos(phis), st * np.sin(phis), np.full(nphi, mu)], 1)
        p = np.broadcast_to(P, (nphi, 2)).copy()
        at = np.full(nphi, -1)  # -1 water, 3j+0 gap, 3j+1 quartz
        w = np.ones(nphi)
        passes = np.zeros(nphi, int)
        gap_now = np.zeros(nphi, bool)
        first_gap = np.zeros(nphi, bool)
        alive = np.ones(nphi, bool)
        for _ in range(40):
            idx = np.flatnonzero(alive)
            if not idx.size:
                break
            pp, dd, aa = p[idx], d[idx], at[idx]
            m = idx.size
            best = np.full(m, np.inf)
            surf = np.full(m, -1)
            wet = aa < 0
            t = crossing(pp, dd, np.zeros((m, 2)), WALL, np.zeros(m, bool))
            tk = wet & (t < best)
            best, surf = np.where(tk, t, best), np.where(tk, -2, surf)
            for j in range(4):
                c = np.broadcast_to(CENTRES[j], (m, 2))
                t = crossing(pp, dd, c, OUTER, np.ones(m, bool))
                tk = wet & (t < best)
                best, surf = np.where(tk, t, best), np.where(tk, 3 * j + 2, surf)
                mine = (~wet) & (aa // 3 == j)
                lay = aa % 3
                for circ, fo, inl in ((0, True, 0), (1, False, 0), (1, True, 1), (2, False, 1)):
                    sel = mine & (lay == inl)
                    if not sel.any():
                        continue
                    t = crossing(pp, dd, c, RADII[circ], np.full(m, fo))
                    tk = sel & (t < best)
                    best, surf = np.where(tk, t, best), np.where(tk, 3 * j + circ, surf)
            ww = w[idx] * np.where(wet, np.exp(-A * np.where(np.isfinite(best), best, np.inf)), 1.0)
            fin = np.isfinite(best)
            pp = pp + np.where(fin, best, 0)[:, None] * dd[:, :2]
            lampof = np.where(surf >= 0, surf // 3, -1)
            circof = np.where(surf >= 0, surf % 3, -1)
            stop = ~fin | (surf == -2)
            hit = circof == 0
            # record arc hits: basic radiance -> water radiance (n_w/n_air)^2
            cls = np.where(
                passes[idx] == 0, 0, np.where(passes[idx] == 1, np.where(first_gap[idx], 2, 1), 3)
            )
            val = ww * L_ARC * (WATER / AIR) ** 2 * dphi * dmu * 2  # x2 for -mu
            for k in range(4):
                G[k, im] += val[hit & (cls == k)].sum()
            stop |= hit
            go = ~stop
            n1 = np.where(wet, WATER, np.where(aa % 3 == 0, AIR, QUARTZ))
            n2 = np.where(
                circof == 2, np.where(wet, QUARTZ, WATER), np.where(aa % 3 == 0, QUARTZ, AIR)
            )
            cen = CENTRES[np.clip(lampof, 0, 3)]
            nrm2 = (pp - cen) / np.where(circof == 2, OUTER, INNER)[:, None]
            nrm = np.concatenate([nrm2, np.zeros((m, 1))], 1)
            cn = np.sum(dd * nrm, 1)
            face = np.where((cn < 0)[:, None], nrm, -nrm)
            ci = np.abs(cn)
            r, ct = fresnel(ci, n1, n2)
            tir = r >= 1.0
            stop |= go & tir
            go &= ~tir
            ratio = (n1 / n2)[:, None]
            nd = ratio * dd + (ratio[:, 0] * ci - ct)[:, None] * face
            nd /= np.linalg.norm(nd, axis=1, keepdims=True)
            ww = np.where(go, ww * (1 - r), ww)
            na = aa.copy()
            out_q = go & (circof == 2) & ~wet
            in_q = go & (circof == 2) & wet
            na = np.where(out_q, -1, na)
            na = np.where(in_q, 3 * lampof + 1, na)
            na = np.where(go & (circof == 1) & (aa % 3 == 0), 3 * lampof + 1, na)
            into_gap = go & (circof == 1) & (aa % 3 == 1)
            na = np.where(into_gap, 3 * lampof, na)
            g_now = np.where(in_q, False, gap_now[idx]) | into_gap
            ps = passes[idx]
            fg = first_gap[idx]
            fg = np.where(out_q & (ps == 0), g_now, fg)
            ps = ps + out_q
            p[idx], d[idx], w[idx], at[idx] = pp, np.where(go[:, None], nd, dd), ww, na
            passes[idx], gap_now[idx], first_gap[idx] = ps, g_now, fg
            alive[idx[stop]] = False
    return G, mus


if __name__ == "__main__":
    pts = [(55, 25), (50, 50), (-25, 41), (37, 37), (0, 0), (45, 25)]
    nphi = int(sys.argv[1]) if len(sys.argv) > 1 else 2**17
    nmu = int(sys.argv[2]) if len(sys.argv) > 2 else 160
    for x, y in pts:
        G, mus = gather(np.array([x, y]) * 1e-3, nphi, nmu)
        tot = G.sum()
        s = G.sum(1)
        print(
            f"({x},{y}) G={tot:.4f}  own={s[0] / tot:.5f} quartz-only={s[1] / tot:.2e} "
            f"via-gap={s[2] / tot:.4f} two+={s[3] / tot:.2e}",
            flush=True,
        )
        (HERE / "work").mkdir(exist_ok=True)
        np.save(HERE / "work" / f"backward_{x}_{y}.npy", G)
