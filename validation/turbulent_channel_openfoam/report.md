# Turbulent channel: aquaflux k-omega SST vs OpenFOAM k-omega SST

The **same model** (k-omega SST) run in **both codes** on an equivalent fully-developed,
streamwise-periodic, resolved-wall channel, driven to the same bulk Reynolds number. This
settles whether aquaflux's realized `kappa ~ 0.34-0.39` (below the nominal 0.41; see
`validation/turbulent_channel`) is standard SST behaviour or an aquaflux gap.

- OpenFOAM: `incompressibleFluid` + `kOmegaSST` + `meanVelocityForce` (`of_case/`).
- aquaflux: streamwise-periodic SST channel + mass-flow controller, matched `Re_b`.

## Results

| Re_tau | code | u_tau/U_bulk | nu_t/nu peak | realized kappa (indicator plateau) |
|---|---|---|---|---|
| 368 | aquaflux | 0.0551 | 48.5 | 0.333 |
| 379 | OpenFOAM | 0.0567 | 50.3 | 0.343 |
| 3569 | aquaflux | 0.0420 | 519.7 | 0.388 |
| 3630 | OpenFOAM | 0.0427 | 529.0 | 0.389 |

See `figures/comparison.png`.

## Conclusion

The two independent SST implementations **agree** on every metric to a few percent at both
Reynolds numbers -- the mean profile, the wall stress `u_tau/U_bulk`, the eddy-viscosity ratio,
and the **realized `kappa`**, which both codes carry a few percent below the nominal 0.41 and
which both increase with `Re_tau` (kappa ~ 0.34 at Re_tau ~ 380, ~ 0.39 at Re_tau ~ 3600). So
aquaflux's below-nominal realized `kappa` is **standard k-omega SST behaviour**, reproduced by
the reference implementation -- not an aquaflux discretization or model error.

## Reproduce

```bash
# 1. OpenFOAM (needs the openfoam13 image): low + high Re_tau -> runs/{low,high}
cd validation/turbulent_channel_openfoam
docker run --rm -v "$PWD":/work -w /work/of_case openfoam13:latest bash run_of.sh
# 2. aquaflux + comparison (from the repo root)
cd ../..
python3 validation/turbulent_channel_openfoam/compare.py
```
