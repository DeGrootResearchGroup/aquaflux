# Sozzi & Taghipour reactor — fluence rate against discrete ordinates

`generate_dom_reference.py` meshes of-optical-radiation's `uvReactorSozzi2006-DOM` tutorial and
runs its discrete-ordinates (DOM) solver at several angular resolutions. See its docstring for the
case and the environment it needs.

## Measured (2026-09-22)

Configuration: of-optical-radiation `726714d`, image built locally from its `Dockerfile`
(OpenFOAM 13, arm64), Apple Silicon 11 cores / 19 GB, Docker's full allocation. Mesh from the
tutorial's own `snappyHexMesh` settings: **1,635,909 cells**, `checkMesh` OK (not the ~365k an
earlier estimate recorded). Absorption 35.67 /m, lamp exitance 696.42 W/m², walls black, no
scattering. Both runs stopped on DOM's convergence test (1e-5) after **15 outer sweeps**, well
under the cap of 100.

| directions (`2 nPhi nTheta`) | wall time | median G | mean G (unweighted) | min G |
|---|---|---|---|---|
| 64 (8 x 4, the tutorial's) | 1035 s | 35.5 | 146.5 | **-6.0** |
| 256 (16 x 8) | 4193 s | 41.0 | 147.9 | -0.12 |

**DOM is not converged in angle here, pointwise.** Over cells above 1e-3 of the peak (78% of
them), 64 against 256 directions differ by a median of **21%**, 90th percentile **84%**, 99th
**115%** — the ray effect, which is worst for a small bright source in weakly absorbing water.
The unweighted mean moves by **1%**, so energy is placed wrongly rather than lost, and the
negative values at 64 directions are a symptom of the same under-resolution. 1024 directions would
cost ~4-5 h and, holding 1024 radiance fields of 1.6M cells (~13 GB), plausibly more memory than
this machine has. So a pointwise gate against DOM is not defensible at any resolution reachable
here; a volume-integrated comparison is.

## aquaflux against DOM (2026-09-22, `compare_fluence.py`)

aquaflux: the lamp patch's own STL (7516 facets, 35.26 W at 696.42 W/m²), direct gather at every
cell centre, exact visibility through the pipe openings (checked against brute-force segment
sampling: 0 disagreements in 8000). Its own discretization error, against a lamp refined to 67,381
facets on 1500 cells within 5 mm of the lamp at x = 0.39-0.41 m: **median 0.44%, p99 4.1%, max
6.1%**. Gather: 557 s for 1.6M cells on this machine.

| volume-weighted mean G (W/m²) | aquaflux | DOM 256 | DOM 64 |
|---|---|---|---|
| whole reactor | 133.28 | 133.16 | 131.67 |
| chamber | 145.26 | 145.13 | 143.51 |
| riser | 0.528 | 0.440 | 0.260 |
| inlet | 0.0096 | ~0 (-8e-5) | 1.8e-4 |

DOM / aquaflux over cells above 1e-3 of the peak, percentiles 1 / 10 / 50 / 90 / 99:
**64 directions -0.11 / 0.36 / 0.974 / 1.65 / 1.97; 256 directions 0.80 / 0.886 / 0.988 / 1.076 /
1.17.** Every percentile moves toward 1 as DOM's directions quadruple — DOM converging on
aquaflux, which is the evidence that aquaflux is near the converged field. The whole-reactor means
agree to 0.09%, which also rules out a lamp-area mismatch between the STL and the snapped patch.

Three places they differ, visible in `work/compare/`:
- **Angular scatter at fixed radius** (radial profile at x = 0.40 m): the true field is
  axisymmetric there and aquaflux gives one curve; DOM 64 scatters by up to ~10x with angle, DOM 256
  by ~15% near the wall. The ray effect, seen directly.
- **Beyond the lamp tip** (x = 0.81-0.89 m): DOM 256 shows rays of alternating sign fanning from
  the tip.
- **The pipes**: aquaflux decays smoothly up the riser and along the inlet; DOM falls in banded
  steps to 1e-37 (256) and 1e-67 (64) — only a narrow cone of directions reaches down a 19 mm pipe,
  and the discrete directions mostly miss it. Negligible for dose (G < 1 W/m² there against
  hundreds in the chamber), but it is where DOM is wrong by orders of magnitude.

## How many facets the lamp needs (2026-09-23, `lamp_resolution.py`)

The comparison above uses the tutorial's own `lampWall.stl` — 7,516 facets at ~4 mm, a mesh made
for `snappyHexMesh` to snap to rather than a number chosen for radiation. This measures what that
buys, against an analytic lamp refined to 270,336 facets (35.4397 W), on 8,000 sampled cells,
exitance 696.42 W/m², absorption 35.67 /m.

| lamp | facets | power W | near the lamp (<5 mm), median / p99 | rest of the chamber, median / p99 |
|---|---|---|---|---|
| the case's STL | 7,516 | 35.2596 | 1.95% / 7.28% | 1.08% / 2.63% |
| 8 x 16 | 288 | 34.4966 | 25.4% / 52.6% | 8.82% / 26.3% |
| 16 x 32 | 1,152 | 35.205 | 10.4% / 27.7% | 2.21% / 8.38% |
| 24 x 64 | 3,360 | 35.3374 | 4.08% / 13.4% | 0.84% / 2.28% |
| 32 x 128 | 8,704 | 35.3837 | 1.42% / 6.51% | 0.38% / 0.88% |
| 48 x 256 | 25,728 | 35.4169 | 0.43% / 2.77% | 0.15% / 0.35% |
| 64 x 512 | 67,584 | 35.4285 | 0.15% / 0.98% | 0.07% / 0.17% |

Error falls about in proportion to the facet count, and the cells nearest the lamp set the
requirement: one facet subtends a large angle from a millimetre away, and the absorption along its
path is evaluated once, at its centroid. A Lambertian emitter's solid angle is exact at any
distance, so none of this is a solid-angle error — a facet count buys absorption sampling and the
inscribed area, nothing else. The STL is slightly worse than its count suggests (1.95% against
1.42% at 8,704) because its triangles are irregular and its area is 0.5% under the true cylinder;
rescaling to equal emitted power gives 1.45% / 0.58%.

The mesh's own patches are the expensive way to get this: the snapped `lampWall` patch carries
48,550 faces for about what 25,728 analytic facets buy.

## Shadowing arbitrary geometry (2026-09-23, `ray_acceleration_probe.py`, `grid_mask_check.py`)

The comparison above shadows this reactor analytically, which is exact because the fluid is three
convex cylinders — and is a description of one reactor rather than a method. These two measure the
general alternative: the vessel wall as the 53,500 triangles `bodyWall.stl` holds, with a uniform
grid deciding which of them a sight line is worth testing against.

`ray_acceleration_probe.py`, on 4,000 sampled rays: at a 128-cubed grid an occupied voxel holds 9.7
triangles, a segment enters 65 of them and reaches the first occupied one after about 5, so it
tests **52 triangles instead of 53,500**. A traced implementation cannot do this — needing a static
trip count and a static per-voxel count, it would pay 183 steps times 36 triangles, 6,588 tests a
ray, worse than testing everything — which is why the walk is ordinary host code over the rays
still in flight.

`grid_mask_check.py` runs that mask against the analytic one on the same rays: 7,516 lamp facets,
24,000 receivers (20,000 pipe cells, where a mask does anything, plus 4,000 chamber cells as a
control), 180,384,000 rays, 78 min.

| | pipes, 20,000 cells | chamber, 4,000 cells |
|---|---|---|
| pairs masked differently, per cell | 88.8 of 7,516 | 0 |
| relative difference in G, median / p99 / max | 1.2% / 8.9% / 54% | 0 / 0 / 0 |

The chamber control is exactly zero, and of the 1,776,306 pairs the two masks disagree on, 99.993%
cross the pipe opening between **0.976 and 0.9997** of its radius. An STL draws a round pipe as an
inscribed polygon — `cos(pi/15) = 0.978` puts this one at about fifteen sides — and that sliver
between the polygon and the circle is the entire disagreement. Neither mask is wrong; they are
given different geometry.

**The cost is the finding.** In that run the analytic arm took 6.9 s against the grid's 4,659.8 s
on the same rays, and the whole field with analytic visibility takes 557 s where the triangulated
mask alone extrapolates to tens of hours. Read any such ratio with its scene: the grid's cost is
set by how far a segment travels through empty voxels, so it depends on both the receivers and the
voxel size. Measured in one process, 300,000 rays per corner, two alternating passes, fastest per
corner:

| | area-sized default grid (212, 11, 114) | 128 cubed |
|---|---|---|
| pipe cells | 28,248 rays/s | 92,038 |
| randomly placed cells | 79,963 | 141,451 |

The two axes interact, so neither has a single factor. Pipe-cell rays run the length of the
chamber, so the coarser axial voxel of the 128-cubed grid is worth 3.26x to them and 1.77x to
randomly aimed rays: what sets the cost is the voxel size along the axis the rays actually
traverse. Ratios taken across separate runs of this machine are not reliable to better than about
1.4x; within one process, repeats here held to 1.01-1.10x.
