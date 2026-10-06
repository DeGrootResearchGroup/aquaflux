# Sleeve optics in a lamp array: what they contribute, by brute force

A Monte Carlo reference for light leaving sleeved ultraviolet lamps in water, written to decide what an
optical model of the sleeves must carry before one is built into the library. It imports nothing from
aquaflux.

- `tracer.py` -- the tracer. Infinitely long lamps (arc, air gap, quartz sleeve) in water inside a black
  circular wall; 2D geometry, 3D ray directions; unpolarized Fresnel reflection, Snell refraction and
  total internal reflection at every interface; arcs absorb what reaches them; the fluence rate from
  the power the water absorbs. Each ray records the path classes it has met.
- `check_tracer.py` -- the tracer against independent answers.
- `measure.py` -- the measurement below.

## The tracer is checked first

`check_tracer.py`, numbers from a run at 200,000 rays (its default is 2,000,000):

- A single lamp's **sleeve surface emitting straight into water** (no optics) against aquaflux's
  `direct_fluence_rate` on a 3 m tube of 256 x 600 sectors and slices, `UniformAbsorption(5.129)`:
  within one standard error of the tracer's ring average at 13.5, 16, 20, 30, 45 and 60 mm
  (|difference| 0.5-2.9 %, 0.2-1.1 standard errors).
- **Energy**: emitted = water + arcs + opaque sleeves + wall + in flight, to 2e-16.
- **Fresnel**: normal incidence water-quartz 0.0019990 against `((n1 - n2) / (n1 + n2))^2`; reflectance
  plus the independently written transmittance is one to 4e-16 over 50 angles; total reflection just
  past the critical angle.

## Configuration

Four lamps on a square, centre spacing 50 mm or 80 mm; arc radius 7.5 mm, sleeve inner and outer radii
10.25 and 11.5 mm; indices at 254 nm water 1.376, fused silica 1.5048, air 1.0003; no absorption in
quartz or air; each lamp 1 W/m from its arc, Lambertian; **arcs absorb everything that reaches them**;
black wall at radius 150 mm; 2 mm pixels; 4,000,000 rays per lamp per trace. Water 95 % UVT
(5.129 /m napierian) and 65 % (43.078 /m). numpy 2.4.6, Linux x86_64, 2026-10-06, one run.

Each array is traced five ways:

| trace | what it is |
|---|---|
| full | every interface, own sleeve and neighbours' |
| full, another seed | **the control**: its ratio to full is the sampling noise |
| own sleeve only | neighbours' sleeves absent, their arcs still absorbing |
| straight, from the arc | no interfaces: aquaflux today with the arc as emitter, other arcs as occluders |
| straight, from the sleeve | the sleeve's outer surface emits, every sleeve opaque: aquaflux today with the meshed sleeve as the lamp |

## Results

Ratios are the full trace over the other, per pixel, p5 / p50 / p95; "between" is the water inside the
square of lamp axes, "band" the water within 5 mm of any sleeve.

| spacing, UVT | control (noise), between | full / own sleeve only, between | full / straight from arc, between · band | full / straight from sleeve, between · band |
|---|---|---|---|---|
| 50 mm, 95 % | 0.963 / 1.000 / 1.039 | 1.022 / **1.078** / 1.137 | **0.890** · 0.876 | **0.874** · **0.852** |
| 50 mm, 65 % | 0.971 / 1.001 / 1.030 | 0.989 / **1.023** / 1.052 | 0.964 · 0.873 | 0.996 · **0.845** |
| 80 mm, 95 % | 0.942 / 0.999 / 1.060 | 0.965 / **1.032** / 1.093 | **0.872** · 0.835 | **0.871** · **0.795** |
| 80 mm, 65 % | 0.948 / 0.999 / 1.058 | 0.947 / 1.001 / 1.056 | 1.008 · 0.834 | 1.059 · **0.801** |

Share of the power the water absorbs in the full trace, by path:

| spacing, UVT | own sleeve only | reflected off another sleeve | entered another sleeve |
|---|---|---|---|
| 50 mm, 95 % | 90.2 % | **0.25 %** | 9.6 % |
| 50 mm, 65 % | 97.6 % | 0.05 % | 2.4 % |
| 80 mm, 95 % | 95.1 % | 0.12 % | 4.7 % |
| 80 mm, 65 % | 99.6 % | 0.01 % | 0.35 % |

Where the emitted power goes (water / arcs / wall), full against straight from the arc: 50 mm 95 %
0.460 / 0.129 / 0.412 against 0.513 / 0.102 / 0.385; 80 mm 65 % 0.930 / 0.063 / 0.007 against
0.992 / 0.003 / 0.006 -- with one lamp's neighbours far, **6 % of what an arc emits comes back to it**
through its own sleeve's reflections.

## What it says

1. **Reflection off a neighbour's sleeve exterior is negligible**: at most 0.25 % of the water's power.
   A convex-mirror model of the sleeve's outside alone buys nothing measurable.
2. **Light passing through a neighbour's sleeve matters at close spacing and clear water**: +7.8 %
   median between lamps at 50 mm and 95 %, +3 % at 80 mm, nothing at 80 mm and 65 %. The full trace's
   arcs absorb less (0.129 against 0.159 at 50 mm, 95 %): a neighbour's sleeve bends light around its
   arc.
3. **A lamp's own sleeve is the largest effect**: a straight model from the arc reads 12-13 % high
   between lamps in clear water and 13-17 % high in the band beside the sleeves at both transmittances;
   the meshed-sleeve model reads 13 % high between lamps in clear water and **15-26 % high beside the
   sleeves**, at both transmittances. Fresnel losses at three interfaces, light returned to the arc,
   and refraction's change to the angular distribution entering the water, together.

**Bound on what this does not settle**: every figure assumes the arcs absorb all that reaches them, no
absorption in the quartz, infinitely long lamps and a black wall. The arc assumption moves items 2 and 3
directly (light returned to an arc, light a neighbour's sleeve bends around its arc) and is not
bracketed here.
