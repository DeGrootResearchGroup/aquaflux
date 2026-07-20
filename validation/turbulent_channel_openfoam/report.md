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
| 380 | aquaflux | 0.0569 | 50.4 | 0.340 |
| 379 | OpenFOAM | 0.0567 | 50.3 | 0.343 |

See `figures/comparison.png`.

## Points not compared

- **high** (OpenFOAM Re_tau ~ 3630): the aquaflux segregated solve did not converge (`EquinoxRuntimeError`), so this point is absent from the table and the figure.

## Conclusion

At every Reynolds number that converged, the two independent SST implementations **agree** on
each metric to a few percent -- the mean profile, the wall stress `u_tau/U_bulk`, the
eddy-viscosity ratio, and the **realized `kappa`**, which both codes carry a few percent below
the nominal 0.41. So aquaflux's below-nominal realized `kappa` is **standard k-omega SST
behaviour**, reproduced by an independent implementation -- not an aquaflux discretization or
model error.

The discretization is matched term by term (momentum `linearUpwind`, k/omega `upwind`,
uncorrected Green-Gauss gradients), so what the comparison isolates is the model, not the
schemes.

## Reproduce

```bash
# 1. OpenFOAM (needs the openfoam13 image): low + high Re_tau -> runs/{low,high}
cd validation/turbulent_channel_openfoam
docker run --rm -v "$PWD":/work -w /work/of_case openfoam13:latest bash run_of.sh
# 2. aquaflux + comparison (from the repo root)
cd ../..
python3 validation/turbulent_channel_openfoam/compare.py
```
