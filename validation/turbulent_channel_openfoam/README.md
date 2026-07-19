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

The two independent SST implementations agree on the mean profile, `u_τ/U_bulk`, `ν_t/ν`, and the
**realized `κ`** to a few percent at both Reynolds numbers — and both carry `κ` a few percent below
the nominal 0.41, rising with Re_τ. aquaflux's below-nominal `κ` is **standard k-ω SST behaviour**,
not an aquaflux gap.

## Follow-up

This is a **matched-setup** comparison (equivalent, independently-built meshes). The stricter
same-mesh import — reading the OpenFOAM cyclic mesh into aquaflux via `read_openfoam` and comparing
cell-for-cell — needs cyclic-patch support in the reader, tracked as a separate feature.
