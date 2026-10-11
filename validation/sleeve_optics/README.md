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
| straight, from the arc | no interfaces, no absorption inside the sleeves |
| straight, from the sleeve | the sleeve's outer surface emits 1 W/m into the water, every sleeve opaque: aquaflux today with the meshed sleeve as the lamp |

⚠️ **"Straight, from the arc" is not what aquaflux computes with the arc as emitter.** aquaflux's
`UniformAbsorption` absorbs along the whole straight segment, the 4 mm of air and quartz inside the
sleeve included; this trace absorbs only in water. ⚠️ **"Straight, from the sleeve" emits the arc's full
watt from the sleeve**, where a real sleeve passes about 0.94 of it (the rest returns to the arc), so
about six points of its gap is power a user would calibrate away; the rest is angular and is not.

## Results

Ratios are the full trace over the other, per pixel, p5 / p50 / p95; "between" is the water inside the
square of lamp axes, "band" the pixels whose centres lie 13.5 to 16.5 mm from a lamp axis. ⚠️ **The band
does not reach the first 2 mm of water** (11.5 to 13.5 mm), where the gaps are largest. **Read the
medians**: the control shows the p5/p95 columns are mostly sampling noise. A model reads high by
`1/ratio - 1`: 0.890 is 12.4 % high, 0.795 is 25.8 %.

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

1. **Reflection off a neighbour's sleeve exterior is negligible**: at most 0.25 % of the water's power,
   as the 0.2 % water-quartz reflectance at normal incidence predicts. A convex-mirror model of the
   sleeve's outside alone buys nothing measurable.
2. **Light passing through a neighbour's sleeve adds up to 8 % between lamps (median, 50 mm, 95 %)
   against neighbours' sleeves that are not there**; 3 % at 80 mm; nothing at 80 mm and 65 %. The full
   trace's arcs absorb less (0.129 against 0.159 at 50 mm, 95 %): a neighbour's sleeve bends light around
   its arc. Two qualifications: a model that meshes the sleeves treats them as **opaque**, not absent,
   which this table does not measure; and the whole effect rests on the arcs absorbing what reaches them.
3. **A lamp's own sleeve is the largest effect.** The meshed-sleeve model reads 14.4-14.8 % high between
   lamps in clear water and 17-26 % high in the band, at both transmittances, and more in the first 2 mm
   of water the band misses. Only about 6 % of the arc's emission is a power loss (light returned to the
   arc, almost all of it reflected at the inner air-quartz surface); the rest is **refraction changing
   the directions light leaves the sleeve in**, which moves fluence from beside the sleeve outwards and
   which no scalar transmittance corrects.

**Bound on what this does not settle**: every figure assumes the arcs absorb all that reaches them, no
absorption in the quartz, infinitely long lamps and a black wall. The arc assumption moves items 2 and 3
directly (light returned to an arc, light a neighbour's sleeve bends around its arc) and is not
bracketed here. A low-pressure mercury plasma traps its own 254 nm line (optically thick, re-emitting),
so a fully absorbing arc is the pessimistic end, not the physical one.

## aquaflux's refracted gather against the tracer

`check_refraction.py`: one lamp of the array above (arc, air gap, quartz sleeve, 1 W/m from the arc,
Lambertian) in 95 % UVT water, no wall in reach. aquaflux gathers the arc's facets at points along a
radius of the mid-plane along their refracted paths (`aquaflux.radiation.refracted.refracted_fluence_rate`:
out through the air-quartz and quartz-water surfaces, Fresnel and Snell at each, every reflection
counted as lost). The tracer runs twice -- in full, and with each ray ended at its first reflection
(`Scene(reflections=False)`), which counts exactly the transmitted paths aquaflux follows -- so the
gather's own error and the worth of the reflected paths it leaves out are measured apart.

Configuration: 40,000,000 rays per trace, 0.25 mm pixels (a 1 mm pixel's average reads above the
value at its ring this close to the sleeve), rings 0.6 mm wide; the arc a 3 m tube of 256 sectors,
slices 0.2 mm at the mid-plane growing by 6 % (106,496 facets); jax 0.10.2, CPU, Linux (4 cores),
2026-10-08. The gather took 90 s for the 7 points, compile included (one run; the tracer 155 and 136 s).

| radius, mm | aquaflux / transmitted trace | in standard errors | transmitted / full trace | straight gather / full trace |
|---|---|---|---|---|
| 12.5 | 0.9993 | -0.41 | 0.9997 | 1.255 |
| 13.5 | 0.9996 | -0.24 | 1.0000 | 1.244 |
| 16 | 0.9995 | -0.30 | 0.9990 | 1.221 |
| 20 | 1.0025 | +1.52 | 0.9971 | 1.201 |
| 30 | 0.9991 | -0.54 | 1.0012 | 1.171 |
| 45 | 1.0008 | +0.47 | 0.9976 | 1.138 |
| 60 | 0.9974 | -1.38 | 1.0024 | 1.116 |

- **The refracted gather agrees with the transmitted paths to within the tracer's noise** at every
  radius, the first millimetre of water beside the sleeve included.
- **The reflected paths it leaves out are worth under half a percent** (transmitted over full, 0.997 to
  1.006, itself within about two standard errors of one): almost all the light a sleeve reflects goes
  back to the arc, which absorbs it.
- **The straight-line gather from the same arc reads 12-25 % high** -- it absorbs along the air and
  quartz too, which the tracer's straight variant above does not, so read it for scale only.
- ⚠️ **The first version of the gather read 1.5-2 % low within 2 mm of the sleeve, converged under
  refinement.** It was the path solve giving up on paths seen at grazing incidence (fixed by solving
  as a descent on the optical length), not physics; the tracer's emitting-sleeve check at the same
  pixels and rays (aquaflux exact there) agreed with the tracer to 0.2 % at every ring, which is what
  ruled the tracer out.

## Four lamps: the scene with transparent solids against the tracer (`check_array.py`)

Four sleeved lamps on the corners of a 50 mm square (`SLEEVE_ARRAY_PITCH` 0.025), each 1 W/m from its
arc, 95 % UVT water inside a black wall at 100 mm. aquaflux is `solve_scene` with the four sleeves as
`Media` (128 sectors per arc on the graded tube, 212,992 facets, `RayCastOcclusion(grid=True)`); the
tracer at 20,000,000 rays a lamp, 0.5 mm pixels, each point the mean over the water pixels within 1 mm.
aquaflux follows no reflection, so it is compared with the trace that ends each ray at its first one.
"Entered" is the tracer's share of the transmitted light at the point that passed through another
lamp's sleeve. In this first run aquaflux crossed a neighbour's sleeve straight (its Fresnel losses and
absorption along the unbent line); the routes that replaced it are measured in the next section. jax 0.10.2, CPU, x64, Linux x86_64, 4 cores, one run,
2026-10-08; aquaflux 977 s, compile included; the traces 476 and 348 s.

| point (mm) | aquaflux / transmitted | in standard errors | traced error | entered | transmitted / full | straight / full |
|---|---|---|---|---|---|---|
| (0, 0) | 1.0015 | +0.25 | 0.60 % | 0 | 0.945 | 1.097 |
| (0, -25) | 0.9971 | -0.35 | 0.85 % | 0.8 % | 0.926 | 1.077 |
| (45, 25) | 0.9875 | -1.44 | 0.87 % | 1.8 % | 0.925 | 1.124 |
| (55, 25) | 0.9673 | -2.26 | 1.44 % | 2.4 % | 0.940 | 1.040 |
| (37, 37) | 1.0244 | +1.30 | 1.87 % | 0 | 0.943 | 1.586 |
| (50, 50) | 0.9647 | -3.31 | 1.06 % | 2.3 % | 0.940 | 1.025 |
| (-25, 41) | 0.9871 | -1.33 | 0.97 % | 1.9 % | 0.930 | 1.202 |
| (0, 75) | 0.9722 | -2.33 | 1.19 % | 1.0 % | 0.927 | 1.113 |
| (-65, 0) | 0.9904 | -0.59 | 1.63 % | 0.7 % | 0.927 | 1.058 |

- Where no light passed another sleeve, aquaflux agrees with the transmitted trace within its
  statistical error.
- **Where light passes a neighbouring sleeve, aquaflux reads 2.8-3.5 % low, at 2.3-3.3 standard
  errors** (beyond a lamp along the edge and along the diagonal, and outside the array). The straight
  crossing of the neighbour is the likely cause, but this is **not decomposed**: the deficit is larger
  than the light the tracer says entered a sleeve (1.0-2.4 %), so it is not only the bending of that
  light; a straight line meeting a neighbour's arc where the bent path misses it would also count, and
  was not measured.
- The reflections left out are worth 5.5-7.5 % here against under 0.5 % beside one lamp: with four
  lamps, light reflected off the neighbours' sleeves reaches the water. That is step 4's.
- The straight gather with the sleeves absent reads 2.5-59 % high.

### Bent routes through a neighbour's sleeve

Light between a lamp and a point may now pass through one neighbouring sleeve along a path bent there
(a route of its own: through the quartz, or through the quartz and the air gap), searched for from four
starting points inside that sleeve, and a path meeting a sleeve it does not pass is no path. Measured on
the branch at `535b0fa7`: `optics` mode, **64 sectors** per arc (106,496 facets; 64 against 128 sectors
agree to 0.9998-1.0000 at every point in `invisible` mode), the stored 60M-ray traces (2 mm disc),
`RayCastOcclusion(grid=True)`; jax 0.10.2, CPU, x64, macOS arm64, 11 cores, one run, 2026-10-10;
aquaflux by `rerun_aquaflux.py` against `work/check_array-optics.json`.

| point (mm) | traced | straight crossing (60M rays, 64 sectors) | bent routes | in standard errors |
|---|---|---|---|---|
| (0, 0) | 16.058 +- 0.034 | 0.9985 | 0.9985 | -0.7 |
| (0, -25) | 16.733 +- 0.029 | 0.9903 | 0.9985 | -0.9 |
| (45, 25) | 11.030 +- 0.078 | 0.9825 | 0.9977 | -0.3 |
| (55, 25) | 8.331 +- 0.039 | 0.9706 | **0.9897** | **-2.2** |
| (37, 37) | 9.508 +- 0.089 | 0.9857 | 0.9852 | -1.6 |
| (50, 50) | 7.007 +- 0.036 | 0.9719 | 0.9947 | -1.0 |
| (-25, 41) | 12.796 +- 0.108 | 0.9735 | 0.9927 | -0.9 |
| (0, 75) | 5.740 +- 0.028 | 0.9853 | 0.9947 | -1.1 |
| (-65, 0) | 7.563 +- 0.042 | 0.9971 | 1.0036 | +0.7 |

- **Every point is now within 1.1 % of the transmitted trace, and only (55, 25) is beyond two standard
  errors.** Split by route (`route_split.py`, six points), the routes through a neighbour carry
  0.8-2.3 % of the traced value against the tracer's 0.8-2.6 % that entered another sleeve.
- **Where the light goes at (-25, 41)**: a 2D forward trace from lamp 1's arc finds two rays reaching
  it, both missing every arc -- one through lamp 2's air gap, turned about 45 degrees near the critical
  angle, whose straight line passes nowhere near lamp 2; one through lamp 2's gap and then lamp 3's. The
  first is found once the starts stand inside the sleeve; the second passes two neighbours and is not
  carried (it bounds to about 0.08 % here). The shortest path through lamp 3 alone runs into its arc.
- **Left open**: (55, 25) still misses about 0.6 % of light that passed a neighbour (a path round the
  arc that is not a shortest path is the candidate, unmeasured); 0.2-0.65 % is short on light that
  entered no other sleeve at some points (largest at (-25, 41)), not decomposed; (37, 37) is unchanged at
  0.985 (-1.6 standard errors), with only 0.1 % of its light having passed another sleeve.
- **Cost**: the gather took 9,456 s against 1,456 s with the first version's row cull (6.5x). No route
  is culled: a route through a quartz sleeve can turn light by up to 120 degrees, and through its air gap
  too by up to 217 (the sum of the largest turn at each surface, the air ones limited by the critical
  angle), so with a facet emitting into its front half no direction can be ruled out from geometry alone.
