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

| | rays/s | 180M rays |
|---|---|---|
| `Outside` of three cylinders | 17.2M | 10.5 s |
| `BranchOpenings`, hand-derived | 28.6M | 6.3 s |

**0 of 180 million pairs masked differently**, and `G` agrees to `0.0` relative at the median, the
99th percentile and the maximum. Both describe the same ideal cylinders, so there is no faceting
to explain a difference away and a disagreement would have meant a defect in one of them. The
1,840 pipe receivers carry 13.8M of those pairs — the only place the mask does anything — so the
agreement is not an artifact of testing mostly-clear geometry. The general construction costs
1.7× the bespoke one, which is the price of not being told where the openings are.

Against the triangle grid the primitive arm is 122–609× faster, and the span is over the *grid's*
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
