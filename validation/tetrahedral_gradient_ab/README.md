# tetrahedral_gradient_ab — a real mesh with corner tetrahedra

## Why

Issue #432: `MultipleCorrectionGradient`'s default (`boundary_closure=OwnerGradient()`,
`fallback=None`) leaves the reconstruction underdetermined at a tetrahedron owning two or more
boundary faces, and `fallback=SkewCorrectedGradient()` repairs it — but every prior measurement of
either half of that was on a synthetic unit fixture (`tetrahedral_grid_3d`, a perturbed unit cube) or
a probe, never on a mesh with an inlet, an outlet, and boundary conditions attached. This case builds
one: a short rectangular duct meshed as tetrahedra rather than the hexahedra every other 3D case here
uses.

Any tetrahedral mesh of a box has cells owning two or more boundary faces wherever an element touches
an edge of the box — no special construction is needed, only a genuinely unstructured tet mesh of a
domain with edges. At the shipped mesh size: 2462 cells, 176 of them corner cells.

## What this answers

`report_m2_conditioning` (part 1 of `compare.py`) measures `max|M2⁻¹|` per cell directly — the actual
quantity the corner-cell defect is about — before and after the local repair:

| | `fallback=None` | `fallback=SkewCorrectedGradient()` |
|---|---|---|
| worst corner cell's `max\|M2⁻¹\|` | 3.29e16 | 10.1 |
| cells above the 1e4 threshold | 176 | 0 |

This is now recorded in issue #432: the repair works exactly as documented, on a real mesh, for the
first time.

## What this does not yet answer

Whether the repair is *safe on a real march* — the question `fallback` exists for. `run_march_ab`
(part 2 of `compare.py`) attempts a coupled RANS march under both closures and is currently blocked
before either arm reaches a converged state, for reasons independent of the corner-cell defect above
(both arms are affected almost identically): tracked separately as **#435**. As shipped, expect
`compare.py` to report the M2 numbers above cleanly and then both march attempts as `FAILED` — that
second half is the expected, currently-unresolved state, not a result about the gradient closure.

⚠️ **For a while it failed for a different reason and said the same thing.** `run_march_ab` called
`solve_coupled` with `rtol=`/`atol=` after those keywords had moved onto `Convergence`, and caught
*every* exception, so the resulting `TypeError` (raised in 0.1 s, before a single step) was reported
as the expected #435 failure. It now passes `convergence=Convergence(...)` and catches only
`EquinoxRuntimeError`, the march's own non-convergence guard, so an API break raises instead. With
the march actually running (anchor rung, dual time, complete LU, both closures), measured 2026-09-21
on a commit carrying the per-field binding and its corner-cell repair: `owner` diverges to `inf` at
step 1 (step 0 at 40 cycles); `repaired` takes four steps at 3–4 cycles each with `|R|` near 2.5 and
diverges at step 5.

## Status

- **Not a physics-validated case.** No OpenFOAM reference is run; the mesh is coarse and the duct
  short, deliberately, to keep the march cheap once it can run at all.
- **Marched for laminar flow, not yet for coupled RANS.** `laminar_duct_march.py` marches the same mesh
  as a laminar duct (below); the coupled RANS march is still blocked, see #435.

## The laminar march (issue #448)

`laminar_duct_march.py` marches the same mesh as a laminar duct (Re_Dh 50: unit density, mu 5e-4, inlet
speed 1 m/s, 0.025 m hydraulic diameter, first-order-upwind advection) with the flow-only path's
robustness machinery, to separate the flow-only path's configuration from the case itself. Arms:
`bare` (`newton_march` over `momentum_continuation`), `staged` (`solve_flow_march`, one shifted step),
`lu` and `simple` (`solve_flow_march` with `DualTimeLoop(inner_steps=3)` and a `MaterializedJacobian`
whose inverse is a complete LU or `SimpleSmoothed`). One run per row, 2462 tetrahedra, `CorrectedGreenGauss`
unless a row says otherwise, 60-step cap. Steps to convergence, or `failed` with the last residual
(Euclidean norm unless noted):

| start | measure | `bare` | `staged` | `lu` | `simple` |
|---|---|---|---|---|---|
| uniform plug | row-scaled (`bare`: Euclidean) | 12 | 15 | 17 | 17 |
| rest | Euclidean | failed, 1.7e-3 | failed, 1.6e-3 | 13 | 13 |
| rest | row-scaled | failed, 1.7e-3 | NaN at step 0 | NaN at step 0 | NaN at step 0 |

- From a plug every arm converges, so the mesh, the case and the gradient reconstruction can be marched;
  the reported failure was the single-step pseudo-transient march from rest, which fails with or without
  the staged driver. Dual time plus a materialized-Jacobian inverse converges it from rest in 13 steps,
  and `SimpleSmoothed` does so without PETSc.
- The row-scaled measure divides by the mean speed and the mass throughput, both zero at rest, so it is
  NaN there (issue #459). Use `LAM_MEASURE=euclid` from rest.
- With `LAM_SCHEME=multiple` (`MultipleCorrectionGradient` with the corner-cell fallback), the `lu` arm
  from rest with the Euclidean measure does not converge, before or after the boundary first pass was
  reweighted by the boundary condition's own dependence on the owner gradient (#463). Before that
  change the march stalled at 4.79e-2 with the linear solve at its cycle cap every step (a singular
  per-cell inverse made the compiled and eager residuals differ); on a working commit carrying it, the
  march diverges slowly instead (7.6e2 at step 59, one run). The investigation of #435 found the same
  duct converging with that scheme on an orthogonal hexahedral mesh and with its first pass alone, so
  the second pass on tetrahedra is where it breaks. That is #435's open question, not a property of
  the flow-only path.
- **Binding the scheme against each field's boundary conditions (#467) first made this worse, and the
  repair restored it.** One run per row, same settings (`LAM_SCHEME=multiple LAM_START=rest
  LAM_MEASURE=euclid LAM_ARMS=lu`), 2026-09-21:

  | commit | step 9 | step 29 | step 59 | cycles per step |
  |---|---|---|---|---|
  | before the per-field binding (`9f33eca`) | 1.24e-2 | 4.70e-1 | 7.64e2 | 7–83 |
  | per-field binding as first merged | 1.78e-4 | 1.78e-4 | 1.78e-4 | 120 (the cap), every step |
  | with the corner-cell repair | 1.40e-2 | 8.83 | 3.12e1 | 13–85 |

  The middle row did not move at all: binding against the pressure's zero-gradient walls left 94
  corner tetrahedra with a singular Hessian correction (`max|M2^-1|` 3.1e16), and the linear solve
  could not make progress past it. Those cells now keep the geometry-only correction. From a uniform
  plug (row-scaled measure) the first two commits both diverge to `inf` at step 1; the repaired one
  was not run from a plug.

## Layout

- `of_case/make_mesh.py` — gmsh script building the tetrahedral duct mesh (OpenCASCADE + Delaunay),
  with three physical surface groups (`inlet`, `outlet`, `walls`) and one physical volume (`fluid`),
  written as Gmsh MSH format 2.2 ASCII. The element size is tuned (see the script's own `MESH_SIZE`
  comment) to avoid a separate, unrelated hazard: an unstructured Delaunay mesh occasionally leaves a
  cell whose immediate neighbourhood is nearly coplanar, which makes the Hessian-correction normal
  equations singular for a reason unrelated to the corner-face question this case is about.
- `of_case/constant/polyMesh/` — the OpenFOAM mesh `gmshToFoam` writes; not tracked (regenerable, see
  below).
- `compare.py` — the M2 conditioning report and the march attempt.
- `laminar_duct_march.py` — the laminar march above; settings are `LAM_*` environment variables.
- `rhie_chow_sign_probe.py` — assembles the Rhie–Chow pressure operator from geometry alone for the
  multiple-correction gradient with its second pass withheld on chosen cells, and reports where the damping's
  sign flips (`TET_POLYMESH` is not read; it uses the case mesh). It orders the laminar march's outcomes
  (the march converges only with the second pass withheld out to ring 3).

### Investigation-only probes (issue #435)

The rest of the scripts here are single-purpose harnesses from the investigation into why the coupled
march does not start on this mesh, kept for a re-adjudication rather than removed. They are a
point-in-time snapshot — see the module docstring of each for its own configuration, and read one before
trusting its output against a checkout that has moved on.

- `diagnose_435.py` — reconstruction exactness on this mesh and where a plug state's residual lives.
- `diffusion_operator_probe.py` — a scalar Laplace operator split into its orthogonal part and its
  non-orthogonal correction, across gradient schemes and on control meshes; also the schemes used to
  withhold or cap the second pass on chosen cells or fields (`FirstPassOnly`, `FirstPassOnCells`,
  `FirstPassWhereIllConditioned`, `CorrectionCapped`).
- `coupled_operator_probe.py` — the coupled Jacobian at a plug state: probe accuracy, an exact-LU Newton
  solve, and non-positive omega diagonals.
- `laminar_duct_probe.py` — the laminar control: the flow residual alone through `solve_flow_march`,
  four gradient schemes on three meshes (`TET_MESHES`, `TET_LAMINAR_RE`, `TET_STEPS`); the full
  multiple-correction scheme fails on both tetrahedral meshes and converges on the orthogonal hex, so
  the failure needs no turbulence. `TET_ARMS` also selects arms that withhold the second pass on cell
  subsets, from one field's gradient only, that cap it, or that give the first-pass gradient to the
  Rhie-Chow damping and/or the viscous flux only, or hand the full gradient to one consumer of an
  otherwise first-pass residual (mixed configurations, so not attribution).
- `seed_state_probe.py`, `warm_start_probe.py`, `mixed_scheme_probe.py`, `limited_hessian_march_probe.py`
  — the same questions at a smooth state converged with corrected Green–Gauss, with the scheme split
  between the velocity and k/omega blocks, or with the Hessian correction withheld on chosen cells.
- `wall_velocity_gradient_probe.py` — imposes a wall-model velocity gradient at wall cells (the
  velocity analogue of omega's `ImposedGradient` wall treatment), optionally combined with withholding
  the Hessian correction on ring 1, and marches the result.
- `divergence_diagnosis_probe.py` — checkpoints a diverging march step by step and decomposes the
  residual by field block and by distance from the wall at a spread of the states it actually visits,
  rather than only at the seed.
- `scheme_march_comparison_probe.py` — marches the same converged seed under several different
  gradient schemes, to separate what is specific to `MultipleCorrectionGradient` from what every
  scheme shares.
- `gradient_difference_probe.py` — every scheme's gradients, strain rate and `grad k`/`grad omega` at
  one converged state, compared per cell with `CorrectedGreenGauss` and grouped by wall ring, boundary
  faces owned, `max|M2^-1|`, non-orthogonality and streamwise position, plus each scheme's error
  against an exact analytic gradient.
- `exact_function_probe.py` — every scheme's gradient of constant, linear, quadratic, cubic and smooth
  fields with exact values, per cell group, to test whether the reconstruction itself is correct.
- `mesh_quality_probe.py` — non-orthogonality, skewness, neighbour-volume ratio and cells-across-duct
  compared against `pitzdaily_openfoam`'s mesh, to judge whether this mesh is realistically coarse or
  unrepresentative.
- `of_case/make_hex_mesh.py` — the same duct as an orthogonal structured hexahedral mesh, the control
  for the tetrahedral mesh's cell shape.
- `of_case/make_mesh.py` also takes `TET_MESH_OPTIMIZE=netgen`, an opt-in control mesh with no corner cells.
- `of_case/perturb_mesh.py` — moves the nodes of the hexahedral duct by a fraction of a cell without
  changing the boundary, to add controlled skew.
- `own_anchor_march_probe.py` — marches each scheme's own low-Re anchor from the plug, never
  warm-starting from another scheme's converged state, to separate a warm-start artifact from a
  property intrinsic to the scheme on this mesh.

## Regenerating the mesh

`of_case/duct.msh` and `of_case/constant/polyMesh/` are not committed (regenerable). Rebuild with a
local `gmsh` install and the `openfoam13` Docker image:

```bash
cd validation/tetrahedral_gradient_ab/of_case
python3 make_mesh.py
docker run --rm -v "$PWD":/case -w /case openfoam13:latest bash -lc "gmshToFoam duct.msh"
```

## Running

```bash
validation/run_case.sh validation/tetrahedral_gradient_ab/compare.py
```
