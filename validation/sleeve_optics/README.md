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
aquaflux by a since-deleted rerun harness against `work/check_array-optics.json` (`cull_cost.py` reruns
the same scene now).

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
- **Cost**: the gather took 9,456 s against 1,456 s with the first version's row cull (6.5x).

### The light round a sleeve's rim, and the route that cannot find it

**Where the time went** (measured with an instrumented copy of the solver that also counted steps, not
kept; `route_cost.py` repeats the timing and the paths found per route): at 12 sectors (19,968 facets), (55, 25) /
(50, 50) / (-25, 41), centroid paths, one point a call, warm; jax 0.10.2, CPU, x64, Linux x86_64, 4
cores, 2026-10-11. The base route costs 0.5-0.7 s a lamp. A route through a neighbour's quartz wall
alone, missing its gap, costs 16-18 s and **found no path at any point** -- 13 of the 28 route groups,
about 43 % of the solve. A route across the gap too costs 17-30 s. On the wall-only route about 80 %
of the searches ran the full 60 steps; their Newton step stayed large while the step the line search
accepted was about the path tolerance, and their state at 30 steps matched that at 60 to three figures.
Stopping a search once its accepted step falls to the tolerance saved nothing (522.0 against 521.7 s:
a batched loop runs until its slowest search ends) and lost 2-4 valid paths in three groups; a lower
cap would drop valid paths (some take 46-56 steps); seeding the corners' searches from the centroid's
path agrees with four fresh starts on 0.977-1.000 of facets but only shortens searches that find a path.

**Why** (three independent reviews of that evidence, each from its own calculation): every path that
crosses a sleeve's quartz wall without entering its gap is a **saddle** of the optical length -- the
water-quartz annulus bends it as a converging lens, so the length falls one way and rises the other
(Hessian eigenvalues in the two crossing angles (-4.2, +3.5), (-4.8, +2.7), (-5.6, +1.5) at impact
parameters 11.25, 11.35, 11.45 mm; seven of seven such paths found by shooting in 2D were saddles).
The descent finds only minima: on this route those are the path through the sleeve's middle, which
meets the gap, and a path pinched to a reflection off the rim, where the searches end (their quartz
chord had shrunk to 6e-10 mm, entry and exit at one point, the legs obeying the law of reflection).
Paths across the gap too are true minima (the air gap is a diverging lens; all four eigenvalues
positive). The band exists for in-plane impact parameters between 10.25 x 1.5048/1.376 = 11.21 mm and
11.5 mm, i.e. incidence of 77 degrees or more, Fresnel transmittance 0.78 at its inner edge falling to 0.

**What that light is worth** (`backward_gather.py`: a deterministic backward gather from each point
through infinite sleeves, transmitted branch only, split by route; 262,144 azimuths x 160 axial
cosines, a coarse 32,768 x 40 run agreeing to 0.07 %; numpy, independent of aquaflux, 2026-10-11):

| point (mm) | gather | traced | through a wall alone | through a gap | two or more | aquaflux - gather |
|---|---|---|---|---|---|---|
| (55, 25) | 8.307 | 8.331 +- 0.039 | **0.71 %** | 1.91 % | 0.013 % | -0.75 % |
| (50, 50) | 6.989 | 7.007 +- 0.036 | **0.26 %** | 2.26 % | 0.0015 % | -0.27 % |
| (-25, 41) | 12.710 | 12.796 +- 0.108 | 0 | 1.93 % | 0.025 % | -0.06 % |
| (37, 37) | 9.370 | 9.508 +- 0.089 | 0.0016 % | 0 | 0 | about 0 |

At every point aquaflux's shortfall against this gather is the wall-alone share plus the share through
two neighbours, to within 0.04 %: at (55, 25) the missing 0.6 % is lamp 3's light bent round lamp 2's
rim, the candidate the section above left unmeasured. Two-neighbour light is under the 0.08 % bound.

**So the route is shot rather than descended** (2026-10-11, agreed with the project owner). For a
while it was simply left out; now `aquaflux.radiation.shooting` finds its paths. Every surface such a
path crosses is the side of a cylinder parallel to one axis, so `beta = n u_z` and, about each
cylinder's axis, the index times the projected impact parameter are conserved: a path is fixed by
`beta` and the impact parameter `b` of the receiver's leg about the sleeve, and `b` lies between the
radius at which the path would just miss the gap and the sleeve's own. A ray is traced backwards from
the receiver over a grid of the two, the cell where both misses of the source change sign starts a
Newton iteration, and the path found is handed to the same stationarity conditions the descent's
paths are differentiated through. One path is kept per side of the sleeve. Used when the route passes
one sleeve's wall, the receiver lies in the water round it, and every body crossed is a parallel
cylinder (`refraction._shot`); a wall route to a receiver inside another region is still left out.
`rim_light.py` gathers the shot routes alone against `backward_gather.py`'s share through a wall
(16 sectors, 26,624 facets, no shadow mask, backward gather 2^17 x 160; jax 0.10.2, CPU, Linux x86_64,
4 cores, 2026-10-11; shot / backward):

| shooting grid | (55, 25) | (50, 50) | (37, 37) | shot routes, s |
|---|---|---|---|---|
| 32 x 32, uniform samples | 0.720 | 0.544 | 0.265 | 711 |
| 96 x 96, uniform samples | 0.769 | 0.629 | 0.472 | 1,114 |
| 32 x 32, each row's run of existing rays found by its ends | 0.761 | 0.631 | 0.515 | 874 |
| the same, each straddling facet cut where its paths stop | **0.924** | **0.910** | 0.799 | 796* |
| the same, each arc triangle split four ways (`SLEEVE_SLICES=2`, 106,496 facets) | 0.932 | 0.912 | 0.806 | 1,878* |
| the same, with a 96 x 96 shooting grid (`SLEEVE_SHOT_GRID=96`) | 0.967 | 0.968 | 0.853 | 1,532 |
| 32 x 32, each corner the grid misses polished from a neighbouring corner's path (shipped, `a443caf`) | **0.988** | **0.995** | 0.952 | 730 |

\* Unit tests shared the 4 cores for part of the run, so the time is an upper bound.

(-25, 41) is zero in both. Uniform samples missed paths between the last sample where the ray exists
and the edge of that run; finding the ends recovers what a 96 x 96 grid does. Most of what was left
was the gather's rule, not the solve: a facet counted only where all three corners had a path, so
the small patch of arc that lights this route lost its boundary facets (the prototype, with the same
rule: 0.79 / 0.61 at 16 sectors, 0.925 / 0.836 at 64, 0.982 / 0.964 at 256 sectors with the arc's
slices split four ways; about 0.17 % of G at (55, 25) at 16 sectors). Cutting each straddling facet
where its paths stop took it to 0.924 / 0.910. What remains, 8-9 % of this route's light (about
0.05 % of G at (55, 25)), is not decomposed. One candidate is a patch narrower than a facet along
the lamp: the arc's facets are long along it, and a patch that misses all three of a facet's corners
is still dropped. **Measured, and refuted**: splitting every arc triangle four ways moves the ratio
only to 0.932 / 0.912, so facet size is not what is left. **The shooting grid is**: at 96 x 96 the same
run reads 0.967 / 0.968, for about twice the time (1,532 s, run alone, against 796 s run beside tests).
So the shipped 32 x 32 grid misses paths, the same kind of miss as the gap routes' finding below:
paths that exist only in a sliver of a row. The 3 % left at 96 x 96 is not decomposed.
**Fixed by continuation, not by a finer grid** (2026-10-12, run alone): a path's unknowns (the
fraction across the band and `beta`) range over a band fixed by the shell alone, so a neighbouring
corner's path is a start inside the polish's basin. Every corner the grid misses beside one it found
is polished from that one's unknowns, until no corner gains a path; the cut's bisection and the traced
pass polish from where each point was found too, so the traced pass scans no grid. At the shipped
32 x 32 grid that reads 0.988 / 0.995 in 730 s -- above the 96 x 96 grid without it, in half its time.
The grid must still hit each patch once: on the two-sleeve fixture a 4 x 4 grid missed one whole patch
(six paths), which nothing can seed; 8 x 8 found exactly the 96 x 96 set. The 1.2 % / 0.5 % left is not
decomposed.
**Checked** (`grid_check.py`, `check_shooting_grid`, 16 sectors, the four points, 32 x 32, run alone,
1,168 s, 2026-10-12): 4,195 corner paths in 6 patches; the grid found 3,758 itself, and all 437 it
missed lie between rows (none along them), so rows are the direction to refine; with 64 rows the
routes find **no new patch**. So at these points 32 x 32 hits every patch and continuation does the
rest; none of its patches was hit only once.

- **Not decomposed**: the 0.2-0.65 % short on light that entered no other sleeve at some points. At
  (37, 37), 0.985 with 0.1 % of the light having passed another sleeve, the backward gather averaged
  over the trace's 2 mm disc reads 0.7 % above its value at the point (the field is curved beside the
  sleeve; 1.0002 at (55, 25)), so most of that gap is likely the reference's averaging -- read off a
  scratch calculation, not yet a committed one.
- **The deflection argument above was the wrong bound.** It summed the largest turn at each surface,
  which no single path can reach: a ray through coaxial cylinders keeps its impact parameter and its
  axial direction times the index, so its turn and transmittance through a sleeve are closed forms of
  those two numbers. Through a quartz wall alone light turns 28-47 degrees in the cross-section, not up
  to 120, and only 1-5 of the 24 routes through a neighbour carry any light at each point. A cull on
  these invariants would be sound for coaxial cylinders; it is not built.

### What the corner rule costs, route by route (`corner_rule.py`)

The refracted gather counts a source triangle only where the paths to all three of its corners
exist, and on a route searched from several starts it pairs the corners' paths by start index. Two
questions, per route and point: does pairing by start index miscount, and how much light sits in
triangles reached at one or two corners. 8 sectors (13,312 facets), points (55,25), (50,50),
(-25,41), (0,-25) mm, no shadow mask, each lamp 1 W per metre; jax 0.10.2, CPU, x64, Linux x86_64,
4 cores, one run, 1,522 s, 2026-10-12.

- **Pairing by start index miscounts nothing that matters.** On every descended route through a
  neighbour's gap, re-pairing each corner with its nearest path changes the route's light by
  0.01-0.03 % where the route carries more than 1e-3 of the point's G. The changes are 1.2-1.8 % on
  routes carrying under 1e-4 of G, which is under 0.001 % of any point's total. The shot route is
  already paired by side.
- ⚠️ **Nearest-path pairing without a tolerance is wrong, and the first run showed by how much.** It
  re-paired the shot route too. Where a corner's own side had no path, it took the other side's,
  forming triangles that span the sleeve: 69x and 199x the shot route's light, 31 % and 21 % of G
  against the backward gather's 0.71 % and 0.26 %. The probe no longer re-pairs shot routes.
- **Triangles reached at only some corners**, counted in proportion with straight-line solid angles:
  at most 0.07 % of G on a descended route (gap of sleeve 1 at (55,25)), 0.0002-0.0006 % on the base
  routes. On the shot route the same estimate reads 1.9-2.2 % of G, which cannot be right: the backward
  gather puts all of that route's light at 0.71 % and 0.26 %. The rim is a strong lens, so a straight
  solid angle overstates the image of a facet seen through it. The measured loss on that route is
  `rim_light.py`'s: 0.761 / 0.631 of the backward gather at 16 sectors, about 0.17 % of G at (55,25).
- **What was done with it (2026-10-12):** pairing by path was not built (the first bullet). On the
  shot route, a straddling facet is now cut where its paths stop and the part with paths counted,
  which took the route to 0.924 / 0.910 of the backward gather at 16 sectors (table above). The
  descended routes keep the all-corners rule, at most ~0.07 % of G by the estimate here.

### Shooting does not find every path through a neighbour's air gap (`gap_shooting.py`, at `a680743`)

The plan was to move the routes through a neighbour's quartz and air gap from the descent to
shooting, but only once shooting finds every path the descent finds there. It does not, so they
stay descended. The probe and the six-line generalization of shooting it needs (the receiver leg's
impact parameter running from the shell's axis to the core's edge) are in commit `a680743` and were
taken out in the next commit. To re-run, check out that commit and run
`validation/sleeve_optics/gap_shooting.py`.

8 sectors (13,312 facets), every distinct corner of lamp 0's arc and of lamp 1's against four
points, no mask; jax 0.10.2, CPU, x64, Linux x86_64, 4 cores, run beside another case (timings
contended and not quoted), 2026-10-12:

- **Shooting finds no path the descent misses, and misses many it finds**: 986 of 1,672 at
  (55, 25) through sleeve 1, 922 of 1,672 through sleeve 2; at (-25, 41) through sleeve 2, none of 868.
- **The missed paths carry light**: on the first of those routes, 10 % of the route's summed path
  transmittance (55.4 of 528.7). They start more than 0.1 m along the 3 m arc, and climb the air
  gaps almost along the axis (`beta = n u_z` within 3e-4 of the air's index). At such a `beta`, the
  receiver leg's impact parameters whose backward ray reaches the source's own gap form a sliver
  narrower than the 32 samples along a row can find. The trace itself is right: at the descent's
  path, both misses are ~1e-10.
- **Rows spaced evenly in angle, so they crowd towards the band's ends, did not help**: 666 missed
  with cell-centred rows, 718 with the band's ends as rows, against 686. What is missing is along
  each row, not across the rows.
- Many of these paths cross the receiver's neighbour's arc inside its gap, and the shadow mask
  would take them out. The criterion agreed for switching was to find every path, so that was not
  used to excuse them.
