# pitzDaily backward-facing step — aquaflux coupled k-ω SST vs OpenFOAM k-ω SST

A **same-mesh, cell-for-cell** cross-code validation on the classic backward-facing step: the
OpenFOAM `pitzDailySteady` tutorial is run with the k-ω SST closure, and aquaflux's coupled RANS
solver is then run on the **same imported mesh**, so the two solutions are compared directly on
identical cells.

## Why

`validation/turbulent_channel_openfoam` compares the two SST implementations on an equivalent (but
independently built) fully-developed channel. This study is the stricter test in two ways: a
**separating, recirculating** flow (the backward-facing step, not a parallel channel), and the **same
mesh** imported into aquaflux via `read_openfoam` rather than a matched-setup mesh — so a difference
is a solver/closure difference, not a meshing one.

## Setup

- **OpenFOAM** (`of_case/`): the shipped `pitzDailySteady` tutorial with **one change** — the RAS
  model switched from the shipped `kEpsilon` to `kOmegaSST` in `constant/momentumTransport` (the
  tutorial's own comment lists `kOmegaSST` as a tested option). `incompressibleFluid` steady SIMPLE
  solver; U_in = 10 m/s, ν = 1e-5 (Re ≈ 25000 on the 25.4 mm inlet height); ≈ 12225 cells;
  inlet / outlet / no-slip walls / empty (2D) front-and-back. The `residuals` function object records
  the SIMPLE convergence history.
- **aquaflux**: the **coupled** RANS solver (`solve_coupled` — one monolithic Newton on
  `R(u, p, k, ω)`) with **hybrid initialization**, **second-order upwind** momentum advection
  (Venkatakrishnan-limited `LimitedUpwind`), **corrected Green-Gauss** gradients (`CorrectedGreenGauss`,
  swept `A_g⁻¹`), and **log-variable ω** (`omega_transform=LogScalars()`) on the imported mesh at the
  same operating point. Log-ω is what this case forced: a direct-ω Newton step drives ω negative once
  the recirculation forms and poisons `ν_t = k/ω` (the residual stays finite, so the divergence guard
  never trips); `ω = e^w` removes that structurally. See the cost note below.

## Near-wall caveat

The pitzDaily mesh is a **wall-function** mesh (first-cell `y+` well above the viscous sublayer),
while aquaflux's SST is **wall-resolving** (it fixes the analytical sublayer `ω` at the wall-adjacent
cell). The comparison therefore focuses on the **outer** flow — the shear-layer growth, the
recirculation bubble, and the lower-wall **reattachment length** — where the near-wall treatment
matters least, and reports the near-wall fields as the expected point of departure.

## Status / cost

The OpenFOAM kOmegaSST reference (fields, mesh, SIMPLE residual history) is fully reproducible via
`run_of.sh`. On the aquaflux side, **log-ω keeps ω strictly positive** and the **per-field relative
residuals descend** on this case — but the coupled log-ω solve on the full ~12k-cell mesh is
**compute-heavy** (each Newton step is several minutes; a full run is hours). The log-ω transport is
validated on a smaller channel (`tests/integration/test_coupled_rans.py`: converges to the direct
fixed point, ω > 0 by construction, exact coupled adjoint); efficient large-mesh convergence — the
reparametrized-block preconditioner scaling and the globalization — is a **known tuning follow-up**.

When judging convergence, read the **per-field relative** residuals, not the absolute `||R||`: the
latter is dominated by ω's ~1e5 near-wall scale and looks stalled while the flow is nearly converged.

## Layout

- `of_case/` — the OpenFOAM case template + `run_of.sh` (runs blockMesh + foamRun inside the
  openfoam13 container, writing the converged fields, the mesh, and the residual history to
  `runs/kwsst/`).
- `compare.py` — imports that mesh into aquaflux, runs the coupled solve on it, compares cell-for-cell,
  and writes `report.md` + `figures/comparison.png`.
- `report.md`, `figures/` — the tracked deliverables, **produced by running `compare.py`**. They are
  not committed yet: that run is the compute-heavy step (see the status/cost note above). The OpenFOAM
  run tree (`runs/`, `of_case/` time dirs, the generated `polyMesh`) is git-ignored.

## Reproduce

```bash
# 1. OpenFOAM kOmegaSST reference (needs the openfoam13 image) -> runs/kwsst/
cd validation/pitzdaily_openfoam
docker run --rm -v "$PWD":/work -w /work/of_case openfoam13:latest bash run_of.sh

# 2. aquaflux coupled solve + comparison (from the repo root)
cd ../..
python3 validation/pitzdaily_openfoam/compare.py
```

## Matched discretization

| term | OpenFOAM (`fvSchemes`) | aquaflux |
|---|---|---|
| momentum advection | `Gauss linearUpwind grad(U)` | `LimitedUpwind(VenkatakrishnanLimiter())` (second order, bounded) |
| k / ω advection | `Gauss limitedLinear 1` | `FirstOrderUpwind()` (bounded) |
| gradient | `Gauss linear` | `CorrectedGreenGauss()` (swept `A_g⁻¹`) |
| laplacian / surface-normal gradient | `corrected` | `DiffusionFlux` non-orthogonal correction |
| ω positivity | bounded scalar scheme + clipping | `omega_transform=LogScalars()` (`ω = e^w`) |

Momentum is second-order (Venkatakrishnan-limited), matching OpenFOAM's `linearUpwind`. The stiff k/ω
scalars use **first-order** upwind: even a *limited* second-order scalar stencil weakens the ω transport
operator's diagonal dominance enough that a coupled-Newton step overshoots ω negative (a Newton-update /
M-matrix effect, not a face-value one the limiter guards). Log-ω keeps ω positive regardless, but
first-order scalars are also much cheaper on the coupled solve, so they are the pragmatic choice here.
OpenFOAM avoids the same failure with `limitedLinear` **plus** per-iteration clipping under its
segregated SIMPLE loop; aquaflux's floor-free coupled residual instead relies on the log transform.
