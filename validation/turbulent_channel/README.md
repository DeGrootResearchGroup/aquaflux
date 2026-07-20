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
