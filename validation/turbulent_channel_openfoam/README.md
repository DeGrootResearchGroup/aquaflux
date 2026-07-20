# Turbulent channel — aquaflux k-ω SST vs OpenFOAM k-ω SST

A cross-code validation: the **same turbulence model** (k-ω SST) run in **both aquaflux and
OpenFOAM** on an equivalent fully-developed channel, to check that aquaflux's realized von Kármán
constant matches the reference implementation.

## Why

The law-of-the-wall study (`validation/turbulent_channel`) found that aquaflux's SST reproduces the
viscous sublayer exactly and a genuine log layer, but with a **realized `κ ≈ 0.34–0.39`** — a few
percent below the nominal 0.41. Everything upstream checked out (constants, strain, F1 blend, wall
distance, mesh independence), pointing to the standard *realized-vs-nominal* gap of k-ω SST. This
study confirms that directly: OpenFOAM's `kOmegaSST`, on an equivalent channel, gives the **same**
below-nominal `κ`.

## Setup

Both codes solve a 2D, streamwise-periodic, resolved-wall (`y+ < 1`) channel driven to the same bulk
Reynolds number, at a **low** (Re_τ ≈ 380) and a **high** (Re_τ ≈ 3600) point:

- **OpenFOAM** (`of_case/`): `incompressibleFluid` steady solver + `kOmegaSST` RAS + `meanVelocityForce`
  (the mass-flow constraint holding `Ubar = 0.1335`), cyclic in x, no-slip walls, empty (2D) in z.
- **aquaflux**: the streamwise-periodic SST channel with the mass-flow controller, at matched `Re_b`.

The comparison is the mean velocity in wall units `u+(y+)` and the **log-law indicator**
`Ξ = y+ dU+/dy+ = 1/κ`, whose flat-plateau value is the realized `κ`.

## Layout

- `of_case/` — the OpenFOAM case template + `run_of.sh` (runs both Re_τ inside the openfoam13
  container, writing the converged fields to `runs/{low,high}`).
- `compare.py` — reads the OpenFOAM fields, solves aquaflux at matched `Re_b`, and writes `report.md`
  + `figures/comparison.png`.
- `report.md`, `figures/` — the tracked deliverables. The OpenFOAM run trees (`runs/`, `of_case/`
  time dirs) are git-ignored.

## Reproduce

```bash
# 1. OpenFOAM: low + high Re_tau (needs the openfoam13 image)
cd validation/turbulent_channel_openfoam
docker run --rm -v "$PWD":/work -w /work/of_case openfoam13:latest bash run_of.sh

# 2. aquaflux + comparison (from the repo root)
cd ../..
python3 validation/turbulent_channel_openfoam/compare.py
```

## Headline

At Re_τ ≈ 380 the two independent SST implementations agree on the mean profile, `u_τ/U_bulk`
(0.0569 vs 0.0567), `ν_t/ν` (50.4 vs 50.3), and the **realized `κ`** (0.340 vs 0.343) — and both
carry `κ` a few percent below the nominal 0.41. aquaflux's below-nominal `κ` is **standard k-ω SST
behaviour**, not an aquaflux gap.

The Re_τ ≈ 3600 point is currently **not compared**: the aquaflux segregated solve does not converge
there (its scalar k/ω sub-solve exhausts its step budget). `compare.py` records the point as
unconverged and reports the rest rather than aborting.

## Matched discretization

The comparison isolates the *model*, so the schemes are matched term by term:

| term | OpenFOAM (`fvSchemes`) | aquaflux |
|---|---|---|
| momentum advection | `Gauss linearUpwind grad(U)` | `LimitedUpwind()` (unlimited) |
| k / ω advection | `Gauss upwind` | `FirstOrderUpwind()` |
| gradient | `Gauss linear` (uncorrected) | `CompactGreenGauss()` |
| laplacian / surface-normal gradient | `corrected` | `DiffusionFlux` non-orthogonal correction |

Two notes on why these pairings are the accurate ones here, not just the matching ones:

- The wall-normal grading keeps the cells rectangular, so the skewness offset `x_ip − x_g` vanishes
  and a skewness-corrected gradient reproduces the compact one to ~1e-13 relative.
- A fully-developed channel carries **no mean momentum advection** (the converged wall-normal
  velocity is ~1e-15 and the flow is streamwise-homogeneous), so the momentum advection scheme has
  nothing to act on — first-order and second-order upwind give identical results to four significant
  figures. The scheme is matched for correctness of the claim, not because it moves this case.

## Follow-up

This is a **matched-setup** comparison (equivalent, independently-built meshes). The stricter
same-mesh import — reading the OpenFOAM cyclic mesh into aquaflux via `read_openfoam` and comparing
cell-for-cell — needs cyclic-patch support in the reader, tracked as a separate feature.
