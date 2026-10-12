# Turbulent channel — aquaflux k-ω SST vs the law of the wall

A fully-developed plane-channel validation of the aquaflux k-ω SST turbulence solver against the
**law of the wall**, built on the streamwise-periodic flow + mass-flow-constraint machinery.

The channel is **streamwise-periodic** (two no-slip walls, periodic in x) driven to a fixed bulk
velocity by a mass-flow constraint — the canonical fully-developed setup. Because the flow is
x-homogeneous the streamwise mesh is trivial (`nx = 4`) and high `Re_τ` stays cheap; the wall-normal
mesh is graded to `y+ < 1`. Each Reynolds number is solved with the segregated SST driver and reduced
to wall units.

## The headline diagnostic

The convincing test of a logarithmic layer is **not** a least-squares slope over a `y+` window (the
buffer below and the wake above bias it low), but the **log-law indicator function**

    Xi(y+) = y+ dU+/dy+ = 1/kappa   (locally).

A genuine log layer shows a **flat plateau** in `Xi`, and the plateau value *is* the realized von
Kármán constant. This study reports that plateau.

## Layout

- `compare.py` — solves the periodic SST channel at a sweep of Reynolds numbers, reduces to wall
  units, computes the indicator function and the plateau `kappa`, and writes `report.md` +
  `figures/law_of_the_wall.png` (mean profile, indicator function, realized `kappa` vs `Re_τ`).
- `report.md`, `figures/` — the tracked deliverables. Heavy run artifacts are git-ignored.

## Reproduce

```bash
# from the repo root
PYTHONPATH=. python3 validation/turbulent_channel/compare.py
```

(`nx = 4`, direct linear flow solve; the three shipped cases run in a few minutes each.)

## Headline

- **Viscous sublayer `u+ = y+`: exact** at every `Re_τ`.
- **A genuine logarithmic layer**: the indicator function `Xi` develops a flat plateau over a decade
  of `y+` at high `Re_τ`.
- **Realized `kappa ≈ 0.39`**, a few percent below the nominal 0.41 and climbing slowly with `Re_τ`
  — the expected realized-vs-nominal gap of standard k-ω SST (the nominal 0.41 is an idealized
  calibration value; the full coupled model realizes `~0.38–0.39`). All model inputs verified correct
  (constants, strain magnitude, `F1` blend, wall distance, cross-flow, mesh independence).

## Status

**Preliminary.** The open question — whether the ~5%-below-nominal `kappa` is standard SST behaviour
or an aquaflux-specific gap — is to be settled by a **direct same-model, same-mesh comparison against
OpenFOAM SST** (as in `validation/skewed_cavity`: solve an OpenFOAM channel tutorial and read its mesh
into aquaflux via `read_openfoam`). That is the next step.

## Holding the bulk velocity with the coupled march (`mass_flow_march.py`)

`mass_flow_march.py` solves one case file two ways and reduces each to wall units: the file's own
`Segregated` solve, and `solve_coupled` with the force appended to the state (single step, a
three-iteration dual-time loop, or a viscosity ramp inside the march). It prints one line per step,
the measure's per-block residuals, and a `RESULT` line. Settings are environment variables
(`CHANNEL_CASE`, `CHANNEL_ARM`, `CHANNEL_MAX_STEPS`, `CHANNEL_PRECONDITIONER`, `CHANNEL_MEASURE`);
see its docstring. Run it through the case runner:

```bash
CHANNEL_CASE=re20000.yaml CHANNEL_ARM=coupled CHANNEL_PRECONDITIONER=convection \
  validation/run_case.sh validation/turbulent_channel/mass_flow_march.py --wait
```

Measured 2026-10-11 (jax 0.11.2, CPU, Linux x86_64, 4 cores, one run per arm). "Before" is the
separate coupled `RootSolver` path that a held bulk velocity ran on until it moved onto the staged march
(commit `795b289`, its Euclidean stopping test); "after" is the staged march. Coupled arms use
`BlockDiagonal(scalar=ScalarAir(), velocity=ConvectionTwoLevel())` unless noted, `rtol 1e-10,
atol 1e-12`, default shift schedule.

| case | arm | measure | result |
|---|---|---|---|
| re20000 | segregated | -- | Re_tau 537.5, kappa 0.340, 47 sweeps, 144 s |
| re20000 | before, single step, `BlockDiagonal()` | Euclidean | not converged in 200 steps (`|R|` 1e3) |
| re20000 | before, single step | Euclidean | converged, 23 steps, 82 s |
| re20000 | before, dual-time | Euclidean | converged, 13 steps, 86 s |
| re20000 | after, single step | Euclidean | converged, 33 steps, 84 s |
| re20000 | after, single step | row-scaled (default) | converged, 101 steps, 133 s |
| re20000 | after, dual-time | row-scaled (default) | converged, 271 steps, 401 s |
| re240000 | segregated | -- | Re_tau 4946.3, kappa 0.382, 41 sweeps, 406 s |
| re240000 | before, single step | Euclidean | frozen (0 cycles, `|R|` 4.3x its start), stopped at step 20 |
| re240000 | before, dual-time | Euclidean | diverged (`|R|` 1e9), stopped at step 38 |
| re240000 | after, single step | Euclidean | frozen (alpha 0, `|R|` 3x its start), stopped at step 84 |
| re240000 | after, single step | row-scaled (default) | not converged in 200 steps (ratio 0.14, still falling) |
| re240000 | after, dual-time | row-scaled (default) | crawling (ratio 0.054 at step 71), stopped |
| re240000 | after, ramp (100x, 16 stations) | row-scaled (default) | diverged (`|R|` 1e12), stopped at step 31 |

Every converged coupled arm reaches the segregated root to the printed digits (Re_tau 537.5, kappa
0.340, force 2.889499e-03). What this shows:

- **re20000 converges on the staged march.** It is the same root, at about the old cost under the
  Euclidean measure.
- **Under the default row-scaled measure it is 3-20x slower.** The constraint row sits at machine
  zero throughout (the plug starts at the target), and the slow block is `k`. The row-scaled measure
  falls by about 20 % per step from this start, so the shift schedule keyed to it relaxes slowly.
- **re240000 is converged by no coupled arm, before or after.** The segregated loop remains the solve
  for that file.
