# 3D backward-facing step: aquaflux coupled k-omega SST vs OpenFOAM k-omega SST

A finite-width 3D backward-facing step (step height h = 0.01 m, expansion ratio 2, span 4h
between no-slip side walls), solved by OpenFOAM `incompressibleFluid` (kOmegaSST) and, on the
**same imported mesh**, by aquaflux's coupled RANS solver (hybrid initialization, second-order
upwind momentum advection, corrected Green-Gauss gradients, log-omega, monolithic algebraic-
multigrid preconditioner). U_in = 10 m/s, nu = 1e-5 -> Re_h = 10000.

## Results

| quantity | aquaflux | OpenFOAM |
|---|---|---|
| reattachment length x_r/h (mid-span) | 8.36 | 7.24 |
| peak nu_t/nu | 150 | 147 |
| rel. L2 U_x error (cell-for-cell) | 0.062 | -- |
| rel. L2 U_y error (cell-for-cell) | 0.007 | -- |
| rel. L2 U_z error (cell-for-cell) | 0.006 | -- |

### Spanwise reattachment (the 3D structure)

| z (m) | aquaflux x_r/h | OpenFOAM x_r/h |
|---|---|---|
| 0.0033 | 12.53 | 10.28 |
| 0.0080 | 8.97 | 7.24 |
| 0.0128 | 8.36 | 7.24 |
| 0.0176 | 8.36 | 7.24 |
| 0.0224 | 8.36 | 7.24 |
| 0.0272 | 8.36 | 7.24 |
| 0.0320 | 8.97 | 7.24 |
| 0.0367 | 12.53 | 10.28 |

See `figures/comparison.png`.

## Reproduce

```bash
# 1. OpenFOAM kOmegaSST reference (needs the openfoam13 image)
cd validation/bfs3d_openfoam
docker run --rm -v "$PWD":/work -w /work/of_case openfoam13:latest bash run_of.sh
docker run --rm -v "$PWD":/work -w /work/of_transient openfoam13:latest bash run_transient.sh
# 2. aquaflux coupled solve + comparison (from the repo root)
cd ../..
python3 validation/bfs3d_openfoam/compare.py
```
