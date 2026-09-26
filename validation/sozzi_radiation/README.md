# Sozzi & Taghipour reactor — fluence rate against discrete ordinates

`generate_dom_reference.py` meshes of-optical-radiation's `uvReactorSozzi2006-DOM` tutorial and
runs its discrete-ordinates (DOM) solver at several angular resolutions. See its docstring for the
case and the environment it needs.

## Dose: what the fluence-rate differences are worth to a particle (2026-09-25, `dose_comparison.py`)

The reactor is designed against dose, not `G`. `dose_comparison.py` solves the tutorial's own flow
and runs of-optical-radiation's Lagrangian tracker (`radiationDose`) once per fluence rate, so the
fluence rate is the only thing that changes between runs. **The particles are the same particles**:
the tracker's generator is seeded, one stream per thread on a static schedule, and dose does not
feed back into the motion, so all four runs end every one of the 9,998 particles at the same time
and point — checked by the harness, which refuses to compare otherwise. Each comparison below is
therefore paired, particle by particle.

| `G` | mean | p1 / p5 / p50 / p95 | min – max | LR k=0.1 | LR k=0.2 | LR k=0.5 |
|---|---|---|---|---|---|---|
| aquaflux | 62.37 | 18.8 / 20.7 / 42.2 / 182.3 | 16.6 – 541 | 1.459 | 2.523 | 5.310 |
| DOM 256 | 62.56 | 18.5 / 20.4 / 42.0 / 184.3 | 16.3 – 559 | 1.451 | 2.504 | 5.256 |
| DOM 64 | 62.00 | 12.6 / 18.6 / 41.6 / 183.0 | 7.5 – 566 | 1.389 | 2.282 | 4.063 |
| DOM 256, its own patch values | 63.48 | 18.5 / 20.4 / 42.0 / 189.6 | 16.3 – 610 | 1.452 | 2.505 | 5.258 |
| Sozzi & Taghipour (2006) | 68 | | ~21 – ~270 | 1.87 | | |

Dose in mJ/cm², log reduction (LR) `-log10 mean(exp(-k D))` over all particles (all escaped).
Per particle, DOM / aquaflux at percentiles 1 / 10 / 50 / 90 / 99: **DOM 256 0.940 / 0.975 / 0.999 /
1.018 / 1.041; DOM 64 0.584 / 0.865 / 0.987 / 1.099 / 1.343.**

- **The mean dose cannot see the ray effect.** The three compared fields agree to 0.9%
  (DOM's own patch values add 1.5%), as the volume-mean `G` did (0.09%): energy misplaced rather than lost averages out along
  a path.
- **The low-dose tail can, and it is what disinfection depends on.** DOM 64 puts its least-exposed
  particles at 7.5 mJ/cm² against 16.6, and its 1st percentile at 12.6 against 18.8, so its log
  reduction falls short by 5% at k = 0.1 and by **23% at k = 0.5** — a sensitive organism's
  inactivation is set by the few particles that dodge the light. DOM 256 is within 1.1% of aquaflux
  at every k measured (0.01 – 0.5). The per-particle spread is widest at low dose (`dose_paired.png`).
- **DOM 256 converging on aquaflux is the evidence, as it was for `G`.** Quadrupling DOM's directions
  shrinks the per-particle 1–99% band from 0.58–1.34 to 0.94–1.04 around aquaflux.
- **Patch values move the high-dose tail only.** The tracker interpolates `G` from cell and boundary
  values and aquaflux computes cells only, so the three compared fields carry `zeroGradient` on every
  patch. Keeping DOM 256's own patch values instead raises its 99th-percentile ratio from 1.041 to
  1.093 and its mean by 1.5% — the particles that graze the lamp — and moves no log reduction by more
  than 0.002.
- **The gap to the paper is common to every `G`** (mean 62–63.5 against 68, LR 1.45 against 1.87, a
  maximum near 550 against ~270), so it lies in the flow, the tracker or the model and not in the
  fluence rate. The tutorial's own record for DOM 64 (LR ~1.39, mean ~64) is reproduced: 1.389 and
  62.0 (63.5 at 256 with DOM's patch values).

Outputs in `work/dose/compare/`: `dose_distribution.png`, `dose_paired.png`, `log_reduction.png`,
`summary.json`, and `sozzi_fluence_rates.vtu` holding the three `G` fields and
`log10 |G_DOM - G_aquaflux|`; each run's `trajectories.vtk` (every 20th vertex, per-vertex dose)
sits in `work/dose/runs/<name>/`, and `work/dose/case` opens in ParaView for `U`.

Configuration: of-optical-radiation `726714d`, image `oor:local` built from its `Dockerfile` (OpenFOAM
13, arm64), Docker's full allocation (11 cores, 18 GB) on Apple Silicon. Flow: the tutorial's
`incompressibleFluid` PIMPLE with local time stepping, realizable k-epsilon, inlet 5.51 m/s (25 US
gal/min), 8 MPI ranks (scotch); stopped by its residual control after **514 iterations, 837 s**, with
the kinematic pressure drop 45.10 m²/s² varying 0.23% over the last 100. Tracker: the tutorial's
`postProcess.dict` (10,000 requested → 9,998 seeded at the inlet, seed 42, discrete random walk
`Cl` 0.15, `dtMax` 5 ms, CFL 0.5, wall reflection) plus `trajectoryStride 20`, 8 OpenMP threads; about
34 s per run. `G`: aquaflux's is `compare_fluence.py`'s field (the case's `lampWall.stl`, 7,516
facets, exact visibility through the pipe openings); DOM's are the 64- and 256-direction runs above.
Lamp exitance 696.42 W/m², absorption 35.67 /m, walls black.

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
from the mesh's 1,635,909 cell centres (19,437 chamber, 2,214 inlet, 2,433 riser), exitance
696.42 W/m², absorption 35.67 /m, jax 0.10.2, CPU, x64, macOS arm64, 11 cores.

⚠️ **Corrected 2026-09-24.** The harness's hand-typed pipes used to stop at x = 1.10 / z = 0.40 while
the mesh's run to 1.739 / 0.894, and its sampler keeps only cells inside the regions, so it silently
never sampled the 206,713 far pipe cells (13% of the mesh). They now end at 1.75 / 0.90 and the
figures below are from the corrected run, except the timing table, which was measured on the old
population (22,250 / 640 / 1,200) — the primitive arm's per-ray cost does not depend on where the
receivers are.

| | rays/s | 180M rays | repeat spread over 3 passes |
|---|---|---|---|
| `Outside` of three cylinders | 20.7M | 8.72 s | 1.003× |
| `BranchOpenings`, hand-derived | 28.8M | 6.27 s | 1.033× |

**0 of 180 million pairs masked differently**, and `G` agrees to `0.0` relative at the median, the
99th percentile and the maximum. Both describe the same ideal cylinders, so there is no faceting
to explain a difference away and a disagreement would have meant a defect in one of them. The
4,647 pipe receivers carry 34.9M of those pairs — the only place the mask does anything — so the
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

81.1% of pairs lie inside one convex region, where a straight segment cannot leave and nothing has
to be tested at all (92.8% was once recorded here, from the truncated population). ⚠️ That share
depends on where the receivers are — the snapped mesh refines near the walls and near the lamp,
which is where the pairs that are not in one region live — so quote the cell-centre figure, never a
volume-uniform sample of the geometry.

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
median, p99 and max — on the corrected population too. Its cylinders are the drawing's, not the
hand-typed ones — the riser is carried into the chamber by recognition rather than by a chosen
`REACH_BACK` — so the parameters differ by design and the mask on the mesh's cells is identical.
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

## The whole field through the public model, shadows streamed (2026-09-24, `model_at_mesh_scale.py`)

`build_radiation_model(cells, lamp, occluders=[cad.fluid(...)], settings=RadiationSettings(
stream_receiver_mask=True, self_occlusion=NoOcclusion()))` then `fluence_rate(...)`, on all 1,635,909
cells with the case's `lampWall.stl`, against `compare_fluence.py`'s hand-chunked field: **max relative
difference 4.4e-16** over the 1,285,221 lit cells (median 0, p99 1.8e-16), and 6.8e-21 W/m² at the
206,713 far pipe cells. Build 65 s; field 1,170 s, about twice the hand-chunked run because the model
also gathers the (here empty) reflected field. Peak footprint 11.15 GB, repeated at 11.07 GB after the
transfer build was cut from 8.33 to 4.07 GB (`transfer_build_peak.py`). So the peak is set by computing
the field, not by building the model. (Measured before the radiation speed-ups of 2026-09-24 to 26; the next
section re-measures it.) The receiver mask, which would add 12 GB if held whole, is not
held. Configuration as in the table above, plus OCP 8.0.1 on CPython 3.13.

## The whole field, before and after the radiation speed-ups (2026-09-26, `model_at_mesh_scale.py`)

The run above, repeated on the same 1,635,909 cells and 7,516-facet `lampWall.stl`, at the commit
before the radiation performance merges of 2026-09-24 to 26 and at the main that followed them.
Each run matches `compare_fluence.py`'s field to **4.4e-16** relative over the 1,285,221 lit cells,
so only the time differs.

| code | how the bodies' layer is decided | build s | field s | vs before | peak footprint |
|---|---|---|---|---|---|
| before (`99c472c`) | every pair (its only strategy) | 65.2 | **1138.4** | 1.00x | 10.94 GB |
| main (`e12f214`) | every pair, `EveryPair()` | 60.0 | **471.7** | 2.41x | 5.57 GB |
| main + #561 (default flipped) | the new default, `ShaftCulling()` = ladder (32, 8, 2) | 57.6 | **313.5** | 3.63x | 6.49 GB |
| main (`e12f214`) | `ShaftCulling`, ladder (32, 8) | 57.3 | **251.9** | 4.52x | 5.48 GB |

- **Without culling, main is 2.4x faster on its own**, and its peak is half. Which merges buy which
  part is not separated here; the mask's narrower storage (#525) is the likely source of the memory.
- **Culling is worth 1.50x at the default ladder and 1.87x stopping at eight**, measured on the mesh's
  own cell centres. The volume-sampled population the culling harness uses gave 5-9x, and the rules
  predicted less here because the mesh refines towards the walls and the lamp. The field also pays
  for the gather, which culling does not touch, so this is not the mask's own speed-up.
- **The third level of the default ladder costs 1.24x on this analytic scene**, as `body_culling.py`
  found on sampled receivers (5.35x against 8.53x). The default keeps it because it is worth ~2.7x
  on a triangulated wall; `ShaftCulling(receiver_blocks=(32, 8), source_clusters=(32, 8))` is the
  choice for a scene of analytic bodies alone.

Configuration: jax/jaxlib 0.10.2, CPU, x64, macOS arm64, 11 cores, 19 GB; black walls, absorption
35.67 /m, exitance 696.42 W/m², `NoOcclusion` self-occlusion, `stream_receiver_mask=True`. The water
is the hand-typed `Outside(chamber, inlet, riser)` — the CAD kernel was not installed, and the two
mask these cells identically (above). One arm per process, run one after another through
`validation/run_case.sh` in one session with nothing else heavy running (`--force` past its free-page
check, which saw 3 GB strictly free while `memory_pressure` reported 40% free). The "before" row
predates this harness's `SOZZI_CULLING` switch and ran from an equivalent scratch copy of it with
the same settings; the two unculled-main and (32, 8) arms were each also run once that way, at
471.5 s and 251.4 s, so repeats agree to 0.2%. Each other row is one run.

## Where the whole-field call spends its time (2026-09-26, `field_cost_breakdown.py`)

The default-culling arm above (313.5 s), broken into its pieces: the same model and call, each piece
wrapped in a timer that waits for its result. Instrumented, the call took **316.8 s** (1% over the
plain run) and repeated to 0.3 s.

| piece | s | share of call |
|---|---|---|
| **shadow masks**, one per chunk (3,076 chunks of ~532 cells) | **243.9** | **77%** |
| · pair-by-pair test of the tiles culling could not certify | 186.7 | 59% |
| ·· the compiled body test (695 calls) | 141.8 | 45% |
| ·· host work around it: index gathers, padding, writing answers into the mask | 44.8 | 14% |
| · deciding which tiles are certified | 45.5 | 14% |
| ·· lamp-facet clearance, the same 7,516 facets recomputed every chunk | 12.6 | 4% |
| ·· receiver clearance | 12.1 | 4% |
| ·· tile certificates | 14.3 | 5% |
| · curve ordering (3.2 s), mask allocation and conversion (~8.5 s) | ~12 | 4% |
| **fluence gather**, compiled | **66.5** | **21%** |
| surface solve: transfer assembly 0.3 s, interreflection 1.2 s | 1.5 | 0.5% |

The build (57 s) is the transfer's row blocks, **56.4 s** — the 7,516 x 7,516 facet-to-facet geometry;
its shadow mask takes **0.2 s**, because culling certifies every lamp-to-lamp pair.

- **Culling is at its ceiling on this geometry.** It certifies **81.0%** of the 12.3 billion
  receiver-facet pairs — the share lying inside one convex region (81.1%, above). The rest cross
  between chamber and pipe, where no clearance certificate can hold, so finer or smarter tiling
  cannot reduce what is tested.
- **Padding is not the cost.** The 2.33 billion pairs in undecided tiles become 2.56 billion
  tested (+9.6%) after each batch is padded to a power of two.
- **The leftover pairs are tested slowly.** 2.56 billion in 141.8 s is ~18 M pairs/s, against
  ~30 M pairs/s for testing every pair on the lamp's own facet mask (1.85 s for 56.5 M pairs,
  separate process, so approximate). The finest tiles are 2 x 2, so each compiled batch is many
  tiny blocks; the (32, 8) ladder, finest 8 x 8, is faster overall (251.9 s).
- **Where time could come back**: testing the leftover pairs at every-pair speed and batching the
  host work (up to ~100 s together); computing the lamp side once per call instead of once per
  chunk (~14 s); a "fully hidden" certificate, the only way to test fewer pairs. The gather is the
  physics. Those two changes would plausibly land the call near 175-200 s — an estimate, not a
  measurement.

**How much a "fully hidden" certificate could remove** (a second run, with the harness counting, per
chunk and along the strategy's own curve, the pairs in tiles whose every pair is blocked):

| | pairs | of all pairs | of the 2.33 billion in undecided tiles |
|---|---|---|---|
| blocked by the water's walls | 1,185,423,023 | 9.6% | 50.8% |
| in wholly blocked 32 x 32 tiles | 985,667,984 | 8.0% | **42.3%** |
| in wholly blocked 8 x 8 tiles | 1,121,866,464 | 9.1% | **48.1%** |
| in wholly blocked 2 x 2 tiles | 1,173,627,432 | 9.5% | **50.3%** |

- **A dark-tile certificate could at most halve what is tested.** About half the undecided pairs are
  blocked, and nearly all of those sit in tiles that are dark throughout, even at 32 x 32. The other
  half are clear pairs that today's clearance certificate cannot vouch for, because it proves a tile
  clear only inside one convex region and these pairs cross from the chamber into a pipe. A
  certificate on the plane of the port where the pipe meets the chamber could in principle decide
  crossing tiles of both kinds -- clear where the shaft passes wholly through the opening, dark where
  it passes wholly outside it -- so the dark-only figure below is not the ceiling of every
  certificate, only of that one.
- **So its ceiling is roughly 70-90 s of the call**: half of the 141.8 s compiled test and of its
  44.8 s host work, plus the gather skipping the 9.6% of pairs that are dark. That is an upper bound
  on the pairs such a certificate could skip, not a prediction of what one would prove.
- The count is harness work (73.1 s, timed on its own outside the bodies' layer); with it subtracted
  the call was 317.7 s, and every other piece repeated the table above to within 0.4 s.

Configuration: as the section above (main `392f935`, the new default culling, water as the
hand-typed cylinders, `NoOcclusion`, streamed receiver mask), run through `validation/run_case.sh`
with nothing else heavy running. The build rows come from a separate 3,000-cell run after the
row-block timer was added (the build does not depend on the receivers); without that timer the
row blocks' work, still running asynchronously, was charged to the facet mask as 30 s.

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

## The lamp from the mesh's own patch (2026-09-25, `lamp_resolution.py`, #492)

A user with only a mesh has the lamp already: the snapped `lampWall` patch. `patch_triangles` cuts
its 48,550 faces into their 194,636 centre-fan triangles, facing into the water, and
`coarsen_surfaces` coarsens them by edge collapse under a longest-edge and a chord bound (angle
0.5 rad), holding the exact patch's emitted power. Same reference, cells, exitance and absorption as
the ladder above, in the same process; every other rung reproduced that table to the digit.

| lamp | facets | power W | near the lamp (<5 mm), median / p99 | rest of the chamber, median / p99 |
|---|---|---|---|---|
| the patch, exact | 194,636 | 35.1511 | 2.01% / 4.97% | 1.33% / 3.04% |
| coarsened, edge 2.5 mm, chord 1e-4 m | 40,083 | 35.1511 | 2.30% / 5.90% | 1.40% / 3.03% |
| coarsened, edge 4 mm, chord 1e-4 m | 18,432 | 35.1511 | 2.53% / 6.70% | 1.46% / 3.16% |
| coarsened, edge 6 mm, chord 1e-4 m | 10,860 | 35.1511 | 2.76% / 7.99% | 1.49% / 3.29% |
| coarsened, edge 4 mm, chord 2.5e-4 m | 15,479 | 35.1511 | 2.83% / 7.64% | 1.59% / 3.71% |
| coarsened, edge 8 mm, chord 2.5e-4 m | 5,995 | 35.1511 | 3.64% / 11.01% | 1.75% / 4.15% |

- **The patch is not the cylinder, and that dominates.** Its 194,636 facets sit 2.0% from the true
  lamp near it — no better than the 7,516-facet STL (1.95%) — because its area is the STL's, shrunk
  again by snapping: 0.050474 m², 0.8% under the true lamp, so it radiates 35.15 W at the case's
  exitance. At equal power the gap is 1.20% near the lamp and 0.52% elsewhere, so it is shape as well
  as area. Against the STL-built field, which is the other description of the *same* lamp, the exact
  patch differs by **0.30% median / 3.65% p99** near it and **0.24% / 0.59%** elsewhere. #492 set
  the bar at the STL's own discretization error from the DOM comparison below, median 0.44% / p99
  4.1%; that was measured on a narrower band (1,500 cells within 5 mm at x = 0.39-0.41 m, against a
  67,381-facet lamp), so read this as inside the bar on the nearest like-for-like band available,
  not as a replicate of it.
- **What coarsening costs, judged against the exact patch** (which isolates it from the patch's own
  departure from the cylinder):

  | coarsened | near the lamp, median / p99 | rest, median / p99 | realized longest edge, max / median | coarsening s |
  |---|---|---|---|---|
  | 2.5 mm, 1e-4 m | 0.31% / 2.05% | 0.08% / 0.26% | 2.50 / 2.19 mm | 20 |
  | 4 mm, 1e-4 m | 0.52% / 2.97% | 0.14% / 0.34% | 4.00 / 3.18 mm | 20 |
  | 6 mm, 1e-4 m | 0.73% / 4.35% | 0.17% / 0.42% | 6.00 / 4.37 mm | 28 |
  | 4 mm, 2.5e-4 m | 0.88% / 3.95% | 0.32% / 0.85% | 4.00 / 3.48 mm | 16 |
  | 8 mm, 2.5e-4 m | 1.65% / 7.37% | 0.46% / 1.22% | 8.00 / 6.00 mm | 26 |

  At 4 mm and 1e-4 m the lamp has 18,432 facets — a tenth of the patch's, and 2.5x the STL's — for
  a 0.52% median change near it, a quarter of the patch's own 2.0% from the cylinder. As with the
  drawing's lamp, **the spacing along the lamp matters more than the chord**: loosening the chord
  from 1e-4 to 2.5e-4 m at 4 mm saves 16% of the facets and adds 70% to the error near the lamp.
  The realized chord and angle stayed inside their bounds on every rung (angle at most 0.44 rad); the
  area fell by 0.2-0.6%, which the power-holding exitance puts back.
- **Which surface to use.** Against DOM, the patch is exactly the surface the reference emits from, so
  a patch-built lamp removes the area mismatch the STL comparison carried. Against the true lamp,
  the drawing's lamp is better at every facet count (0.16% at 66,011 facets) because its vertices
  are on the cylinder rather than on a mesher's approximation of it.

Measured with the batched coarsener (the coarsening times are its own; the first, one-at-a-time
version took 72-124 s per rung on the same rungs, for facet counts within 2% of these and the same
errors to within 0.03 points). Configuration: jax 0.10.2, CPU, x64, macOS arm64, 11 cores, nothing
else running; every non-coarsened rung identical to the digit across three runs. Reading the 1.6M-cell mesh
for its patch took ~8.5 min the first time (`work/lamp_patch.npy` caches the triangles after). The
drawing's rungs were skipped: the CAD kernel is not installed in this interpreter.

## What coarsening costs, at the lamp's size and the vessel's (2026-09-25, `coarsen_speed.py`)

The vessel wall is what a reflecting wall's optical surface would be coarsened from (#491), so it
is the size the coarsener has to be fast at. Same machine and configuration as above, nothing else
running, after a 0.5 s warm-up that compiles the distance kernel:

| patch | input triangles | edge, chord | facets | seconds | ms per input triangle |
|---|---|---|---|---|---|
| `lampWall` | 194,636 | 4 mm, 1e-4 m | 18,432 | 20.8 | 0.107 |
| `lampWall` | 194,636 | 10 mm, 1e-4 m | 7,652 | 35.7 | 0.183 |
| `bodyWall` | 1,322,096 | 4 mm, 1e-4 m | 116,989 | 140.0 | 0.106 |
| `bodyWall` | 1,322,096 | 10 mm, 1e-4 m | 37,431 | 210.0 | 0.159 |

The cost is linear in the input at a fixed bound: 0.106-0.107 ms per triangle on a patch 6.8x the
size. Against the first, one-at-a-time version on the same machine, same inputs, same process
family: the lamp at 4 mm took 93 s for 18,292 facets and the wall **555 s for 117,175** — so the
batched coarsener is about **4x** faster at both sizes, for facet counts within 1%. Realized bounds
held on every run (chord at most the bound, angle at most 0.49 rad).

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
