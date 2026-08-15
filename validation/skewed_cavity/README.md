# Skewed lid-driven cavity — aquaflux vs OpenFOAM

A laminar, non-orthogonal validation of the aquaflux coupled flow solver against
OpenFOAM 13, built to exercise the **AD-linearised non-orthogonal correction**.

The case is the inclined (parallelogram) lid-driven cavity — the
Demirdžić–Lilek–Perić (1992) non-orthogonal benchmark — skewed at **β = 45°**
(uniform 45° mesh non-orthogonality), **Re = 100**. Both codes solve on the
*identical* mesh: OpenFOAM's `blockMesh` builds it, and aquaflux reads that same
ASCII `polyMesh` back in via `read_openfoam`.

## Layout

- `of_case/` — the OpenFOAM 13 case template (`foamRun -solver incompressibleFluid`,
  steady SIMPLE), skewed at β = 45°. Two `empty` caps (`front`/`back`) rather than the
  usual single `frontAndBack`, because aquaflux's 2D collapse currently needs two cap
  patches.
- `run_of.sh` — meshes β = 45° once, then sweeps `nNonOrthogonalCorrectors ∈ {0,1,2}`.
- `compare.py` — the detailed β = 45° comparison: reads the mesh into aquaflux, solves
  it two ways (corrected gradient **in** the AD Jacobian vs `stop_gradient`-lagged /
  deferred correction), compares against the OpenFOAM converged field, and writes
  `report.md` + `figures/comparison.png`.
- `sweep.py` + `run_of_beta.sh` — the skew-angle sweep (β = 30/45/60°): iterations vs
  mesh non-orthogonality for each solver. Writes `report_sweep.md` +
  `figures/sweep.png`.
- `report.md`, `report_sweep.md`, `figures/` — the tracked deliverables. The heavy,
  regenerable OpenFOAM run trees (`runs/`, `runs_sweep/`) are git-ignored.

## Reproduce

```bash
# 1. OpenFOAM: mesh + SIMPLE sweep (needs the openfoam13 image)
cd validation/skewed_cavity
docker run --rm -v "$PWD":/work -w /work openfoam13:latest bash run_of.sh

# 2. aquaflux: solve the same mesh and compare (from the repo root)
cd ../..
PYTHONPATH=. python3 validation/skewed_cavity/compare.py
```

## Headline

| solver | non-orthogonal correction | nonlinear iterations |
|---|---|---|
| OpenFOAM SIMPLE, `nNonOrthogonalCorrectors 0` | absent | **diverges** |
| OpenFOAM SIMPLE, `nNonOrthogonalCorrectors 1`/`2` | deferred | ~213 outer |
| aquaflux, corrected gradient in the AD Jacobian | implicit | **6 Newton (quadratic)** |
| aquaflux, gradient `stop_gradient`-lagged | deferred | 29 (linear) |

Folding the corrected-gradient reconstruction into the residual and letting AD
place its full linearisation in the coupled Jacobian is what turns the lagged /
deferred-correction convergence (linear, or divergent) into quadratic Newton
convergence on the non-orthogonal mesh.
