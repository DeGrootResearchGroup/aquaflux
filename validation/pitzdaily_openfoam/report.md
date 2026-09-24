# pitzDaily backward-facing step: aquaflux coupled k-omega SST vs OpenFOAM k-omega SST

The OpenFOAM `pitzDailySteady` tutorial -- its RAS model switched from the shipped `kEpsilon`
to `kOmegaSST` -- run in OpenFOAM, then solved on the **same imported mesh** by aquaflux's
coupled RANS solver (hybrid initialization, second-order upwind momentum advection, corrected
Green-Gauss gradients). U_in = 10 m/s, nu = 1e-5 (Re ~ 25000 on the 25.4 mm inlet).

## Results

| quantity | aquaflux | OpenFOAM |
|---|---|---|
| reattachment length x_r/h (lower wall) | 8.07 | 7.74 |
| peak nu_t/nu | 418 | 423 |
| rel. L2 U_x error (cell-for-cell) | 0.019 | -- |
| rel. L2 U_y error (cell-for-cell) | 0.010 | -- |

See `figures/comparison.png`.

## Reproduce

```bash
# 1. OpenFOAM kOmegaSST reference (needs the openfoam13 image) -> runs/kwsst/
cd validation/pitzdaily_openfoam
docker run --rm -v "$PWD":/work -w /work/of_case openfoam13:latest bash run_of.sh
# 2. aquaflux coupled solve + comparison (from the repo root)
cd ../..
python3 validation/pitzdaily_openfoam/compare.py
```
