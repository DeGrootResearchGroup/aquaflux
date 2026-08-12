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
  on the imported mesh at the same operating point. The coupled Jacobian is preconditioned by a
  **field-split algebraic-multigrid V-cycle** (`coupled_amg_continuation(field_split=True)`) — see below.

## The preconditioner is the point of this case

> **The two tables in this section were measured before the defaults moved**, at the host (`petsc`)
> trailing inverse, the `dirichlet` wall condition on `k`, and no positivity floor. All three defaults
> have since changed (see *The trailing inverse* below), so these numbers describe the arms they
> compare and not the configuration `compare.py` now runs. What they establish — that splitting the
> hierarchy by field wins, and that the trailing half wants one smoother sweep rather than four — is
> unaffected: both were controlled pairs differing only in the thing named.

### The hierarchy is split by field

The six coupled fields are not one kind of equation: `[u, v, w, p]` is a pressure–velocity saddle, while
`k` and `ω` are advection-dominated transported scalars. They are given **separate multigrid
hierarchies**, with one triangle of the coupling between them retained exactly from the assembled
Jacobian (not dropped — dropping it is a weaker preconditioner, and the coupling is load-bearing).
Measured on this case against the otherwise-identical monolithic V-cycle:

| | monolithic | field split |
|---|---|---|
| wall | 3140 s | **2161 s** (−31%) |
| steps | 58 | 66 |
| Krylov cycles | 293 | 324 (+11%) |
| mid-span `x_r/h` | 8.361 | 8.361 |

**Note the cycle count moves the wrong way and the solve is still much faster.** Two smaller V-cycles
plus one sparse coupling product apply far more cheaply than one six-field V-cycle, so the split buys
more cycles at a lower price per cycle. A cycle count is only a fair proxy for cost between candidates
that share a per-application cost; once the preconditioner's shape changes it stops being one. Set
`BFS3D_FIELD_SPLIT=0` to run the monolithic arm.

### …and the two halves are smoothed differently

Splitting the hierarchies is only half the point: the halves can then be *tuned* apart, and they want
different things. The saddle needs its four incomplete-LU sweeps — Jacobi-class smoothers do not
converge on it at all — while `[k, ω]` is a transported-scalar pair with a genuine diagonal, a far
easier operator, and it does not. Dropping the trailing half from four sweeps to **one**:

| | four sweeps | one sweep |
|---|---|---|
| wall | 1959 s | **1636 s** (−16.5%) |
| steps | 58 | 58 |
| refresh | 21 events / 318 s | 19 / 286 s |
| Krylov cycles | 277 | 282 (+1.8%) |
| mid-span `x_r/h` | 8.361 | 8.361 |

The two marches follow the **same trajectory step for step** — same shift, same per-step cycle counts,
same residuals to four figures, and the single line-search escalation fires at the same step to the same
shift — so the 323 s is the identical path at a lower price per matrix-vector product, not a different
path taken faster. That makes it a much stronger single-run result than a bare 16.5% would be. One sweep
is now the library default (`coupled_amg_continuation(trailing_smoother_sweeps=…)`); vary it here with
`BFS3D_TRAILING_SWEEPS`, and the smoother *method* with `BFS3D_TURBULENCE_SMOOTHER`.

Two cautions carried from the screening that chose it. The cycle count rose while the wall fell, again.
And a candidate needs three things, not two — cheap per application, convergent on a hard operator, and
**not materially weaker than what it replaces**: the arms that were markedly weaker (point-block and
plain Jacobi at low sweep counts) rank well on per-solve cost and are not settled on a march, so cost
ranking alone does not select a smoother.

### The trailing inverse: the in-framework hierarchy, and a comparison that was not one

The trailing `[k, ω]` half can be inverted either by the host GAMG V-cycle or by the framework's own
nodal hierarchy, configured to match it. The host arm was believed faster on this case — 58 steps
against 67 — and that reading was wrong, in a way worth recording because nothing in either log said so.
The two runs behind it differed in three things at once: the wall condition on `k`, the presence of the
positivity floor, and the code they ran on. Their trajectories separate at step one.

Run as a controlled pair — same code, same `zerogradient` wall condition, same `1e-08` floor, everything
else at the settings above — the ranking reverses:

| | in-framework | host GAMG |
|---|---|---|
| wall | **2124 s** | 2893 s (+36%) |
| steps | **67** | 72 |
| Krylov cycles | **329** | 371 |
| escalations | **4** | 8 |
| mid-span `x_r/h` | 8.36 | 8.36 |

Same root by either route; the difference is entirely path cost. The in-framework hierarchy is now the
default (`BFS3D_TURBULENCE_INVERSE=petsc` selects the host arm), which also takes the host callback off
the trailing half and leaves it as plain array work. The flow half still runs on the host V-cycle.

**And the whole difference is one event, not a diffuse quality gap.** The two arms are identical for
**49 steps** — same shift, same residual to four figures (8.810e-03, then 7.274e-03), same full step
length — and part at step 50. What happens there is not a preconditioner failure: it is the positivity
cap collapsing, and each arm meets it at a different step. The in-framework arm hits it first, at 50
(cap 1.01e-02, then 1.44e-05); the host arm sails through 50–52 and hits it at 53–55 instead
(`a_min` 0.316, 0.075, 0.005). Both then pay the same cascade, and the host arm's is the more expensive
one — its shift is driven to 4.44 against 0.94, and the step control can only unwind it at `/grow` per
step, so it spends **ten** steps walking back where the other spends six:

> **walk-back steps = log(β_peak / β_resume) / log(grow)** — predicted 6.0 and 10.0 for the two arms,
> observed 6 and 10.

So the arms are separated by *when* they meet the cap and *how far* the shift is driven when they do,
not by how well either inverts its block. That also says where the remaining wall time is: the cascade,
not the preconditioner.

**The lesson is about the comparison, not the arms.** These marches were read against each other on
the strength of matching aggregate step counts, and no single log was wrong — each stated its own
configuration correctly. What was missing was any check that the arms differed in *one*
thing. A cheap one exists and is now used: two runs that differ only in the preconditioner should track
each other's residuals for the first few steps, because the operator is identical. These separated at
step one, which was visible in the logs the whole time. `march_log_compare.py` reports exactly this —
per-rung counts for one run, and for two runs the first step at which their residuals part.

The 2D cases run on a factorization of the coupled Jacobian; in 3D that factorization is the wall. On this
mesh the assembled coupled Jacobian has ≈ 38.7 million nonzeros (≈ 280 per row), and a single incomplete-LU
factorization at the usual fill runs for many minutes. The **algebraic-multigrid V-cycle** instead keeps the
heavy fill on only the small coarsest grid — a **direct-LU coarse solve** — so its memory stays bounded and
its setup is a matter of seconds. It is one V-cycle (a fixed linear operator), applied inside the coupled
Newton's Krylov solve; a stationary **zero-fill** incomplete-LU level smoother reaches the solve
tolerance on the indefinite saddle, where adding fill is what fails — a level-1 factorization develops
negative pivots as the pseudo-transient shift falls and diverges at the low shifts this march's tail
runs at. Because it is a fixed linear operator it is also
transposable, so the exact coupled adjoint (the point of a differentiable solver) reuses its transpose
V-cycle — verified against finite differences on a channel case
(`tests/integration/test_coupled_amg.py`). It needs the optional `petsc` dependency (`pip install aquaflux[petsc]`).

## Near-wall caveat

This mesh **straddles** the sublayer/log crossover rather than sitting cleanly on one side, and that is
the awkward regime for any near-wall treatment. Measured from the reference `k` at the converged state
(`wall_layer_comparison.py`), the wall-adjacent `y*` median is **2.1 on the floor behind the step** and
**2.5–3.0 on the lower wall** — the viscous sublayer, entirely below the `y* = 11.53` crossover — while
the **upper wall sits at 11.1**, on the crossover itself, and the **side walls at 34**, fully in the log
layer. So the three walls are in three different regimes at once. The comparison
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
- `march_log_compare.py` — reads archived march logs back as data: per-rung steps, cycles, wall time,
  clips, escalations and preconditioner cost for one run, and for two runs the first step at which their
  residuals part. That last number is the check that a pair of arms differs in one thing only; runs that
  should share an operator and separate at step one are not a controlled comparison, however well their
  totals line up.
- `zero_pattern_pivots.py` — what the preconditioner's sparsity pattern actually contains at **zero
  pseudo-transient shift**, and the incomplete-LU pivots it produces. Zero shift is the operator the
  adjoint solves, so every gradient goes through it, and no forward step ever visits it (the march floors
  its preconditioner at a positive shift) — which is why it needs a harness of its own rather than a march.
  It sweeps the probing reach against how the shift and the equilibration are written, since both decide
  whether the assembler's stored *exactly-zero* positions survive into the factorization, and reports the
  pattern per `(row field, column field)` block, the pivot census, the hierarchy shape, and the true
  residual through GMRES. Both spellings are implemented in the harness rather than imported, so an arm
  describes the spelling and not whichever version of the library is checked out, and the arms are
  fingerprinted over their nonzero entries so a comparison whose arms differ in their *values* cannot be
  mistaken for one that differs only in pattern. `BFS3D_CENSUS_ONLY=1` skips the solves when the pattern
  and the factorization answer the question on their own.
- `step_policy_replay.py` — pre-screens candidate step-control policies against those same archived logs,
  in milliseconds rather than in 35–50-minute marches. The rule that sets the shift strength β is
  arithmetic over three recorded numbers per step (the accepted step length, the steady residual, and the
  escalation count), so it can be replayed exactly — which it self-tests by reproducing every logged β
  before printing anything — and then re-run with a candidate rule in place. A candidate's trajectory is a
  **counterfactual under frozen recorded inputs**, never a predicted saving: a different β would have
  produced a different step length and residual, which the replay cannot know. Its use is elimination, and
  its labelling says so. Alongside it reports quantities that *are* measurements: the walk-back a
  β-escalation commits the march to, and the steps whose residual did not move at all.

### Reproducing the aquaflux result

`compare.py` runs the **full Reynolds-continuation solve** — it is the driver, not a thin wrapper
around solver defaults. The target Reynolds number is **not reachable without the ramp**: a direct
cold start diverges on its first step, growing the residual by ~100 orders of magnitude at a
line-search factor already clipped to 0.002. Every non-default setting in the constants block is a
measurement on this case rather than a preference — notably the zero-fill smoother (a level-1 fill
*diverges* at the low shifts the march's tail runs at) and the absolute row-scaled stopping bar.

**The ramp uses two lower-Reynolds rungs** — `Re/100`, then `Re/10`, then the target. Whether the
`Re/100` anchor earns its place was measured, and it is a close call worth knowing about: the cold
start reaches `Re/10` unaided, and reaching a converged `Re/10` costs ~800 s directly against ~1027 s
by way of the anchor — yet the one-rung ladder still finishes slower overall (2007 s against 1959 s),
because it repays the saving in the target rung. Both margins are ~2% on single runs, so neither is
decisive; two rungs is kept because the measured total favours it and the anchor is cheap insurance at
a higher Reynolds number. `BFS3D_N_POINTS=1` runs the one-rung ladder.

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
