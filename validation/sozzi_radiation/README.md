# Sozzi & Taghipour reactor — fluence rate against discrete ordinates

`generate_dom_reference.py` meshes of-optical-radiation's `uvReactorSozzi2006-DOM` tutorial and
runs its discrete-ordinates (DOM) solver at several angular resolutions. See its docstring for the
case and the environment it needs.

## The reactor as CAD primitives (2026-09-23, `primitive_occlusion.py`)

`compare_fluence.py` shadows this reactor with `BranchOpenings`: a closure that knows, for this
reactor, where each pipe's opening is and what crossing it means. `primitive_occlusion.py` runs
the general construction against it — the fluid declared as three cylinders,

```python
Outside(chamber, inlet, riser)
```

with a segment clear exactly when those regions cover it end to end. No opening is identified and
no shadow edge is derived, so the same three lines describe a chamber–pipe–elbow chain of any
length.

Configuration: the case's own `lampWall.stl` (7,516 facets), 24,000 receivers drawn uniformly
from the mesh's 1,635,909 cell centres (22,250 chamber, 1,200 riser, 640 inlet), exitance
696.42 W/m², absorption 35.67 /m, jax 0.10.2, CPU, x64, macOS arm64, 11 cores.

| | rays/s | 180M rays | repeat spread over 3 passes |
|---|---|---|---|
| `Outside` of three cylinders | 20.7M | 8.72 s | 1.003× |
| `BranchOpenings`, hand-derived | 28.8M | 6.27 s | 1.033× |

**0 of 180 million pairs masked differently**, and `G` agrees to `0.0` relative at the median, the
99th percentile and the maximum. Both describe the same ideal cylinders, so there is no faceting
to explain a difference away and a disagreement would have meant a defect in one of them. The
1,840 pipe receivers carry 13.8M of those pairs — the only place the mask does anything — so the
agreement is not an artifact of testing mostly-clear geometry, and it survives `BranchOpenings`
being reformulated — the same comparison against its `crossing_ratio` rewrite is still 0 pairs, so
`Outside` matches two independent spellings of the bespoke test. The general construction costs
1.39× the bespoke one, which is the price of not being told where the openings are; single runs of
the pair gave 1.67× and 1.48×, so the three passes are what make that a number.

Against the triangle grid the primitive arm is 146–732× faster, and the span is over the *grid's*
configurations rather than one number — from its worst measured corner (pipe-cell receivers on the
area-sized default grid, 28,248 rays/s) to its best (random cells on a 128³ grid, 141,451). The
two axes interact, so neither has a single factor: resolution is worth 3.26× on pipe cells and
1.77× on random ones, receiver placement 2.83× at the default grid and 1.54× at 128³. The
primitive arm is one branch-free expression, so its own rate does not depend on either.

⚠️ Those ratios divide this harness's number by a separate run's, so read them as approximate and
the grid's internal comparisons (measured in one process, repeating to 1.10×) as sharp. Running
both arms in one process is what would make the ratios as solid as the square.

92.8% of pairs lie inside one convex region, where a straight segment cannot leave and nothing has
to be tested at all. ⚠️ That share depends on where the receivers are, and sampling the fluid by
*volume* gets it wrong — 97.1% — because the snapped mesh refines near the walls and near the
lamp, which is where the pairs that are not in one region live. Quote the cell-centre figure.

`BranchOpenings` stays. It was derived independently, so it is the reference the general
construction is checked against, and a general construction agreeing with a bespoke one stops
being evidence the moment the bespoke one is deleted. The harness uses the meshed case's cells
when `work/case` is present and samples the three cylinders when it is not, saying which in its
summary — so it runs without OpenFOAM, at the cost of a different receiver population.

## The reactor read from its CAD drawing (2026-09-24, `primitive_occlusion.py`, `lamp_resolution.py`)

The tutorial's STLs are generated from a STEP drawing, `SozziTaghipour.step` (an Onshape export in
metres, body axis along `y`, declared tolerance 10 µm; committed at
`validation/uvreactor_openfoam/of_case/`). `aquaflux.io.cad.read_step` reads it with the axes swapped
to the case's frame, and both harnesses take it as a further arm when the CAD kernel is installed
(`pip install "aquaflux[cad]"`); without it the arm is skipped and the log says so.

**Shadowing.** `cad.fluid("reactor_body", "inlet_pipe", "outlet_pipe")` is read and checked against
the drawing in 0.8 s — its boundary lies within **0.75 µm** of the drawing's against the declared
10 µm — and, on the same 24,000 cells, 7,516-facet STL lamp and 180,384,000 rays as the two arms
above, masks **0 pairs differently** from `BranchOpenings`, with `G` equal to 0.0 relative at the
median, p99 and max. Its cylinders are the drawing's, not the hand-typed ones — the pipes run 850 mm
where the hand-written ones stop at the meshed domain, and the riser is carried into the chamber by
recognition — so the parameters differ by design and the mask on the mesh's cells is identical.
Single-pass times: `BranchOpenings` 8.7 s, hand-typed `Outside` 12.8 s, drawing `Outside` 11.0 s —
one pass each, so not a ratio to quote (#513).

**The lamp as an emitter.** `cad.triangles("lamp", chord=..., facet_size=...)`, base disc dropped
(the analytic ladder and the case's patch omit it too), against the same 270,336-facet reference on
the same 8,000 cells, in the same process as the ladder above:

| drawing's lamp: chord, facet size | facets | power W | near the lamp (<5 mm), median / p99 | rest of the chamber, median / p99 |
|---|---|---|---|---|
| 1e-4 m, 20 mm | 3,144 | 35.3839 | 6.65% / 19.1% | 0.85% / 4.71% |
| 2e-5 m, 10 mm | 14,888 | 35.4317 | 2.22% / 9.27% | 0.18% / 0.89% |
| 1e-4 m, 5 mm | 17,453 | 35.4155 | 0.74% / 4.40% | 0.18% / 0.52% |
| 2e-5 m, 5 mm | 28,521 | 35.4329 | 0.60% / 4.13% | 0.07% / 0.25% |
| 1e-4 m, 2.5 mm | 66,011 | 35.4347 | 0.16% / 1.75% | 0.04% / 0.18% |
| 5e-6 m, 2.5 mm | 116,753 | 35.4410 | 0.11% / 1.62% | 0.00% / 0.04% |

- **Size the facets along the lamp and keep the chord coarse.** The cells nearest the lamp are
  sensitive to the spacing *along* it, which `facet_size` sets; the chord sets the spacing *around*
  it. At 2.5 mm and a 1e-4 chord the drawing's lamp matches the analytic 64 x 512 lamp at equal
  facet count near it (0.16% against 0.15% median at 66,011 against 67,584 facets) and beats it
  everywhere else (0.04% against 0.07%). Its near-lamp p99 is worse (1.75% against 0.98%): the plane
  grid that bounds facet size leaves some irregular patches beside the lamp.
- **Its vertices are on the true surface, so its area is not inscribed.** The finest rung radiates
  35.4410 W, closer to the true 35.443 W (`2 pi R L + 2 pi R^2` at 696.42 W/m²) than the
  270,336-facet analytic reference itself (35.4397 W); read its sub-0.01% errors as at the
  reference's own floor.

Configuration: jax 0.10.2, CPU, x64, macOS arm64, 11 cores; `cadquery-ocp-novtk` 8.0.1.0.0 on
CPython 3.13; `work/case` and `work/cell_centres.npy` from the meshed case of the DOM run below.

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
