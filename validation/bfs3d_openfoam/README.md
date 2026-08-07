# 3D backward-facing step — aquaflux coupled k-ω SST vs OpenFOAM k-ω SST

The first **three-dimensional** same-mesh, cell-for-cell cross-code validation: a finite-width
backward-facing step solved with the k-ω SST closure by OpenFOAM and, on the **same imported mesh**, by
aquaflux's coupled RANS solver. The three-dimensional sibling of `validation/pitzdaily_openfoam`.

## Why

The 2D cases exercise the coupled solver on a plane flow, where the coupled Jacobian can be preconditioned
by a complete or incomplete LU factorization. **In three dimensions those factorizations hit a wall** — a
complete LU's fill is `O(n^{4/3})` (out of memory past a few 10⁴ cells) and even a threshold-ILU's
factorization of the distance-3 3D coupled Jacobian (hundreds of nonzeros per row) is prohibitively slow to
build. This case is the first real exercise of the **algebraic-multigrid preconditioner** built for that
regime, and the first test that the coupled solver — and its exact adjoint — carry over to a genuinely
three-dimensional flow.

**Genuinely 3D, not a 2D extrusion.** The step spans `4h` (`h` = step height) between two **no-slip side
walls**, so the finite span drives corner and secondary flow: the spanwise velocity reaches ≈ 20 % of the
inlet velocity, and the reattachment length varies across the span. A slip/symmetry-walled extrusion would
collapse to the 2D problem; the viscous side walls are what make this a three-dimensional test.

## Setup

- **OpenFOAM** (`of_case/`): a clean `blockMesh` L-shaped domain — step height `h = 0.01 m`, inlet channel
  height `h` (expansion ratio 2), upstream length `3h`, downstream length `20h`, span `4h`. Boundary
  patches are `inlet` / `outlet` / `upperWall` / `lowerWall` / `sideWalls` — all six bounding planes are
  named (no `empty`), so the mesh stays fully three-dimensional on import. `incompressibleFluid` steady
  SIMPLE solver with `kOmegaSST`; U_in = 10 m/s, ν = 1e-5 → **Re_h = 10000**; ≈ 23040 orthogonal-hex cells,
  wall-function near-wall treatment. The mesh is graded two-sided toward the walls and the shear layer shed
  from the step lip.
- **aquaflux**: the **coupled** RANS solver (`solve_coupled` — one monolithic Newton on `R(u, p, k, ω)`)
  with **hybrid initialization**, **second-order upwind** momentum advection (Venkatakrishnan-limited
  `LimitedUpwind`), **corrected Green-Gauss** gradients, and **log-variable ω** (`omega_transform=LogScalars()`),
  on the imported mesh at the same operating point. The coupled Jacobian is preconditioned by a **monolithic
  algebraic-multigrid V-cycle** (`coupled_amg_continuation`) — see below.

## The preconditioner is the point of this case

The 2D cases run on a factorization of the coupled Jacobian; in 3D that factorization is the wall. On this
mesh the assembled coupled Jacobian has ≈ 38.7 million nonzeros (≈ 280 per row), and a single incomplete-LU
factorization at the usual fill runs for many minutes. The **algebraic-multigrid V-cycle** instead keeps the
heavy fill on only the small coarsest grid — a **direct-LU coarse solve** — so its memory stays bounded and
its setup is a matter of seconds. It is one V-cycle (a fixed linear operator), applied inside the coupled
Newton's Krylov solve; a stationary ILU(1) level smoother reaches the solve tolerance on the indefinite
saddle where a plain zero-fill smoother stalls. Because it is a fixed linear operator it is also
transposable, so the exact coupled adjoint (the point of a differentiable solver) reuses its transpose
V-cycle — verified against finite differences on a channel case
(`tests/integration/test_coupled_amg.py`). It needs the optional `petsc` dependency (`pip install aquaflux[petsc]`).

## Near-wall caveat

This is a **wall-function** mesh (first-cell `y+` above the viscous sublayer), while aquaflux's SST is
**wall-resolving** (it fixes the analytical sublayer `ω` at the wall-adjacent cell). The comparison
therefore focuses on the **outer** flow — the shear-layer growth, the recirculation bubble, and the
reattachment length — where the near-wall treatment matters least, and reports the near-wall fields as the
expected point of departure. This mirrors the 2D case.

## The comparison target is the TRANSIENT run, not the steady one

The steady SIMPLE run **limit-cycles** on this separated 3D flow: run to `endTime`, `of_case/` plateaus at a
residual around 1e-3 rather than converging. Unlike the 2D pitzDaily case the steady field is *physical*
(no inlet checkerboard — ω spans about two decades, not the nine of a decoupled field), but it is not a
converged steady root. A **time-accurate** run (`of_transient/`, the `incompressibleFluid` solver with
unsteady `ddtSchemes`, adjustable time step at `maxCo = 0.9`) reaches a statistically-steady state; its
`fieldAverage` mean fields are the comparison target. `run_transient.sh` copies the mesh, properties and
boundary conditions from `of_case/` at run time, so the transient reference is byte-identical in geometry
and setup to the steady case and to what aquaflux imports.

**Consequence for reading aquaflux results:** as in 2D, a short coupled march under-predicts the bubble
because the reattachment length grows as the solve converges. Judge against the transient reattachment and
the peak `k`/`ν_t` of the mean field, and check how far the march got before concluding anything about the
closure.

## Layout

- `of_case/` — the OpenFOAM **steady** case template + `run_of.sh` (runs blockMesh + checkMesh + foamRun in
  the openfoam13 container, writing the fields, the mesh, the 3D cell centres, and the SIMPLE residual
  history to `runs/kwsst/`). Supplies the mesh and the non-convergence history — **not** the comparison
  target.
- `of_transient/` — the **time-accurate** case + `run_transient.sh`, whose time-averaged field *is* the
  comparison target (`runs/kwsst_transient/`). Its mesh, constant and boundary conditions are copied from
  `of_case/` at run time (one definition of the geometry) and are git-ignored.
- `compare.py` — imports that mesh into aquaflux, runs the coupled AMG solve on it, compares cell-for-cell,
  and writes `report.md` + `figures/comparison.png`. Reports a **mid-span** reattachment length and the
  **spanwise variation** of it — the 3D structure the 2D case cannot show.

### Reproducing the aquaflux result

`compare.py` runs the **full Reynolds-continuation solve** — it is the driver, not a thin wrapper
around solver defaults. The target Reynolds number is **not reachable without the ramp**: a direct
cold start diverges on its first step, growing the residual by ~100 orders of magnitude at a
line-search factor already clipped to 0.002. Every non-default setting in the constants block is a
measurement on this case rather than a preference — notably the zero-fill smoother (a level-1 fill
*diverges* at the low shifts the march's tail runs at) and the absolute row-scaled stopping bar.

```bash
python3 validation/bfs3d_openfoam/compare.py
```

writes `march.log` (one framed block per step, readable while it runs), rolling state checkpoints
under `checkpoints/`, and then `report.md` + `figures/`. Expect hours, not minutes.

### Putting a convergence bar on the reference

The transient reference stops each timestep's solve at 1% of its own initial residual
(`relTol 0.01`, two outer correctors). Its per-timestep residuals therefore sit on a **floor** — `p`
pinned at ~9.7e-3 from `t = 0.1` to `t = 0.5` with no trend — which is the tolerance, not
unsteadiness: `Ux` decays three decades to ~1e-5 by `t = 0.15` and stays there, so the flow is
**steady**, and time-averaging from `t = 0.25` averages an already-settled field.

Steady is not the same as tightly converged, though, and a reference converged to ~1% per step is a
soft baseline for judging a reattachment difference of order 10%. `run_transient.sh --tight` re-runs
with `system/fvSolution.tight` (final solves at `relTol 0`, four outer correctors, one
non-orthogonal corrector) into `runs/kwsst_transient_tight`, leaving the reference untouched, so the
two reattachment lengths can be compared directly. If `x_r/h` moves, part of any discrepancy belongs
to the reference rather than to the code under test.

### Why the reattachment length is measured mid-span

`reattachment_length` locates the **last** wall-adjacent cell with reversed streamwise velocity. Measured
across the **full span**, that is whichever cell separates furthest downstream — and in this geometry the
side walls carry their own corner separation that reattaches well behind the primary bubble. On the
reference field the two outermost spanwise slabs read `x_r/h = 10.28` while all six interior slabs read
`7.24`, so a full-span number overstates the primary bubble by ~40%.

The mid-span slab (`mid_span_slab`) is therefore the primary metric, and it is the **same helper** used
both for the final comparison and for the reattachment length reported live during a solve, so the
quantity a run is watched by and the one it is judged by cannot drift apart. The full-span value is still
reported alongside as `xr/h_full`: the gap between the two *is* the corner separation, which is worth
seeing — it is simply not the primary bubble.
- `report.md`, `figures/` — the tracked deliverables, produced by running `compare.py`. The OpenFOAM run
  tree (`runs/`, time dirs, generated `polyMesh`) is git-ignored.

## Reproduce

```bash
# 1. OpenFOAM kOmegaSST references (needs the openfoam13 image)
cd validation/bfs3d_openfoam
docker run --rm -v "$PWD":/work -w /work/of_case openfoam13:latest bash run_of.sh
docker run --rm -v "$PWD":/work -w /work/of_transient openfoam13:latest bash run_transient.sh

# 2. aquaflux coupled solve + comparison (from the repo root; needs the `petsc` extra)
cd ../..
python3 validation/bfs3d_openfoam/compare.py
```

## Matched discretization

| term | OpenFOAM (`fvSchemes`) | aquaflux |
|---|---|---|
| momentum advection | `Gauss linearUpwind grad(U)` | `LimitedUpwind(VenkatakrishnanLimiter())` (second order, bounded) |
| k / ω advection | `Gauss limitedLinear 1` | `FirstOrderUpwind()` (bounded) |
| gradient | `Gauss linear` | `CorrectedGreenGauss()` |
| laplacian / surface-normal gradient | `corrected` | `DiffusionFlux` non-orthogonal correction |
| ω positivity | bounded scalar scheme + clipping | `omega_transform=LogScalars()` (`ω = e^w`) |

The scheme choices match the 2D case; the same reasoning applies (second-order momentum, first-order k/ω to
keep the coupled-Newton ω step from overshooting negative, log-ω for floor-free positivity).
