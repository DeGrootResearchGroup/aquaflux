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
