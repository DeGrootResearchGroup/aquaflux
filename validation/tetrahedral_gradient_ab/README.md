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

⚠️ **The march itself is no longer blocked — the multiple-correction closures are.** `TET_ARMS=projected`
converges this same case (below), so a failing `owner`/`repaired` arm is now a statement about *those
weights*, not about this mesh. What the repair does to a march that runs is still unanswered, because
the corner-cell repair is a `MultipleCorrectionGradient` setting and that scheme is what fails here.

⚠️ **For a while it failed for a different reason and said the same thing.** `run_march_ab` called
`solve_coupled` with `rtol=`/`atol=` after those keywords had moved onto `Convergence`, and caught
*every* exception, so the resulting `TypeError` (raised in 0.1 s, before a single step) was reported
as the expected #435 failure. It now passes `convergence=Convergence(...)` and catches only
`EquinoxRuntimeError`, the march's own non-convergence guard, so an API break raises instead. With
the march actually running (anchor rung, dual time, complete LU, both closures), measured 2026-09-21:
with each probe given its own boundary data, both arms diverge to `inf` at step 3 (`owner` from
`|R|` 1.08 at 40 cycles, `repaired` from 5.9 at 43); under the geometry-only corner-cell tier that
preceded it, `owner` failed at step 1 and `repaired` took four steps at 3–4 cycles before failing
at step 5.

## Status

- **Not a physics-validated case.** No OpenFOAM reference is run; the mesh is coarse and the duct
  short, deliberately, to keep the march cheap once it can run at all.
- **Marched, laminar and coupled RANS, under `ProjectedStencilGradient`** (13 steps and 16 target
  steps respectively; see its section below). Under `MultipleCorrectionGradient` — either corner-cell
  closure — both marches still fail, which is what #435 now covers.
- **The self-start works under `ProjectedStencilGradient`** (`potential_flow_probe.py`, 2026-09-23):
  `hybrid_initialize` returns in 5.9 s with the uniform 10 m/s plug this straight duct's potential
  flow is, and the coupled march from *that* seed converges in 16 target steps at `alpha = 1` to
  `|R|` 5.4e-6 — the same count as the hand-built seed. The hand seeds stay in the harness because the
  multiple-correction arms still need them, not because the shipped path is unusable.
  **The Laplace operator's conditioning is the reconstruction's, not the mesh's:** the exact Jacobian
  of the same scalar problem is **4.67e4** under the projected weights against the 3.3e19 / 1.7e19
  recorded under the multiple-correction closures. The old lead that this duct's aspect ratio might
  simply be badly scaled is refuted.

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
  | each probe given its own boundary data | 7.43e-3 | 2.16e1 | 3.80e2 | 9–84 |

  The second row did not move at all: binding against the pressure's zero-gradient walls left 94
  corner tetrahedra with a singular Hessian correction (`max|M2^-1|` 3.1e16), and the linear solve
  could not make progress past it. The third row kept the geometry-only correction on those cells;
  the fourth is what ships, and determines them instead (pressure binding `max|M2^-1|` 7.8e2), by
  probing each quadratic with its own normal derivative on a zero-gradient face rather than zero.
  None of the four converges -- that is #435. From a uniform plug (row-scaled measure) the first,
  second and fourth diverge to `inf` at step 1; the third was not run from a plug.

## Betchen's coupled reconstruction on this mesh (2026-09-22)

`HessianCorrectedGradient` -- the coupled gradient and Hessian reconstruction of Betchen and
Straatman (2010) -- solved as a fixed number of coupled block sweeps (default 20). One run per row,
commit `bd6bb8b` (#483's branch) plus the harness changes that add these arms.

**The Rhie–Chow damping operator** (`rhie_chow_sign_probe.py`, `RC_BINDING=pressure`, i.e. the laminar
duct's pressure conditions; `nnz/row` is that operator's mean nonzeros per row; the quadratic error is
the worst cell's `|g - g_exact| / max|g_exact|` under exact Dirichlet values, `RC_BINDING=geometry`):

| scheme | flipped diagonals | largest anti-damping eigenvalue | nnz/row | worst quadratic error |
|---|---|---|---|---|
| `MultipleCorrectionGradient` (fallback on), first pass only | 0 | 2.9e-4 | -- | -- |
| `MultipleCorrectionGradient` (fallback on), full | 24 | 2.2 | -- | exact by construction (not measured here) |
| `CorrectedGreenGauss` (default sweeps) | 0 | 9.0e-4 | 80 | 7.0 |
| Betchen, `OwnerHessian` closure, 20 sweeps | 1027 | 5.9e54 (the sweep diverges; 100 sweeps too) | -- | diverges |
| Betchen, `AveragedNeighbourHessian`, 1 sweep | 0 | 4.4e-4 | 12 | 0.77 |
| same, 2 sweeps | 0 | 4.1e-4 | 49 | 0.32 |
| same, 3 sweeps | 0 | 4.1e-4 | 120 | 0.14 |
| same, 5 sweeps | 0 | 4.6e-4 | 336 | 5.6e-2 |
| same, 10 sweeps | 0 | 4.5e-4 | 685 | 5.9e-3 |
| same, 20 sweeps (default) | 0 | 4.5e-4 | 857 | 6.8e-4 |
| same, 100 sweeps | 0 | 7.4e-4 (geometry binding) | -- | 6.9e-10 |
| Betchen, `AveragedInteriorHessian`, 20 sweeps | 0 | 7.2e-4 | -- | 7.1e-4 (100 sweeps: 7.1e-4) |

Two things follow. Betchen's scheme with either averaged closure damps as well as the scheme that
converges here -- and its truncations do too, from a single sweep, so the damping does not come from
the global solve; the sweep count buys accuracy only, roughly halving the worst error per sweep early
on. And `AveragedInteriorHessian` stalls at 7e-4 however far it is swept: it is not exact for
quadratics on this mesh, where `AveragedNeighbourHessian` is. ⚠️ The shipped `CorrectedGreenGauss`
misses a quadratic's gradient by 7x its magnitude at the worst cell: the scheme that converges on this
mesh is the least accurate one measured.

**The marches**, `LAM_SCHEME=hessian LAM_HESSIAN_CLOSURE=neighbour` (20 sweeps):

- **Laminar, from rest** (`LAM_START=rest LAM_MEASURE=euclid LAM_ARMS=lu`, 60-step cap): **converges in
  13 steps to `|R|` 5.6e-9, 287 s**, 3 linear cycles per step after the first -- the same step count
  and final residual as `CorrectedGreenGauss` (13, 5.4e-9), where `MultipleCorrectionGradient` never
  converges (above).
- **Coupled RANS** (`compare.py`, `TET_ARMS=hessian`, anchor rung at Re/10 then the target, dual time,
  complete LU): **converges -- 9 anchor steps and 16 target steps, `alpha = 1` at every step with no
  escalation, to 4.9e-6 against the 1e-5 stop, 1583 s** (`TET_MAX_STEPS=20`; at the shipped cap of 15
  it stops at 1.7e-5 one step short). k stays in [0.36, 1.36] and omega in [618, 4124]. Both
  `MultipleCorrectionGradient` arms diverge to `inf` at step 3 of the anchor rung.

So a reconstruction exact for quadratics can march this mesh: #435's failure is specific to
`MultipleCorrectionGradient`'s second pass, not to second-order gradients meeting Rhie–Chow on
tetrahedra. What these runs do not establish: a cost comparison (no matched wall time for the
converging control), anything beyond this one mesh, or whether a local scheme can have both
properties -- truncated Betchen is local and damps but is not exact for quadratics at any finite sweep
count, and `MultipleCorrectionGradient` is local and exact but anti-damps.

## `ProjectedStencilGradient` — choosing the weights (2026-09-22)

What Betchen's scheme buys here is not its accuracy but the *size* of its weights, and that can be had
on a compact stencil: of all the per-cell weights on the two-hop stencil that are exact for
quadratics, take the ones nearest `blend` x a reference that damps (one block sweep of the
gradient--Hessian system). `aquaflux.schemes.ProjectedStencilGradient` is that scheme;
`rhie_chow_sign_probe.py` measures it with `RC_SCHEMES=projected-<blend>`.

**The damping operator** under the duct pressure's own conditions (`RC_BINDING=pressure`; the `nnz/row`
column is that operator's width, the gradient's own reach being 12.5 — the same as the
multiple-correction scheme's, measured directly):

| scheme | flipped diagonals | largest eigenvalue | worst quadratic error |
|---|---|---|---|
| `MultipleCorrectionGradient` (fallback on) | 24 | +2.17 | exact by construction |
| its first pass alone | 0 | +2.9e-4 | — |
| `CorrectedGreenGauss` | 0 | +9.0e-4 | 6.99 |
| `HessianCorrectedGradient` (neighbour closure, 20 sweeps) | 0 | +4.5e-4 | 6.8e-4 |
| `ProjectedStencilGradient`, blend 0 | 0 | +2.15e-4 | 4.2e-10 |
| blend 0.5 | 0 | +1.95e-4 | 3.7e-10 |
| **blend 0.75 (the default)** | **0** | **+1.92e-4** | **4.0e-10** |
| blend 1 | 0 | +2.03e-4 | 5.3e-10 |

Every blend damps better than the scheme that converges here, and every one is exact for quadratics;
what the blend buys is accuracy on fields that are not quadratic, which
`betchen_projected_probe.py` measures (at 0.75, ~1.6x more accurate in the volume mean than the
minimum-norm end on a smooth field, and within 9% of aiming at the reference entirely).

**The marches**, `LAM_SCHEME=projected`, from rest (`LAM_MEASURE=euclid LAM_ARMS=lu`, 60-step cap):

| scheme | steps | final \|R\| | wall |
|---|---|---|---|
| `CorrectedGreenGauss` | 13 | 5.4e-9 | — |
| `HessianCorrectedGradient` (neighbour closure) | 13 | 5.6e-9 | 287 s |
| **`ProjectedStencilGradient`, blend 0.75** | **13** | **5.0e-9** | **19 s** |
| `MultipleCorrectionGradient` | fails (3.8e2 at step 59) | | |
| `ProjectedStencilGradient`, blend 0 | 13 | 4.8e-9 | 19 s |

**Coupled RANS** (`compare.py`, `TET_ARMS=projected`, `TET_MAX_STEPS=20`, anchor rung at Re/10 then
the target, dual time, complete LU): **converges -- 16 target steps, `alpha = 1` throughout with no
escalation, to 5.8e-6 against the 1e-5 stop, 160 s**, against Betchen's 1583 s for the same case and
both multiple-correction arms diverging to `inf` at anchor step 3. k stays in [0.36, 1.54] and omega
in [615, 4503]. This needed the turbulence closure to bind its schemes per field at build time
(`k_gradient_scheme` / `omega_gradient_scheme`), since `k` and `omega` build their assembler inside
each residual evaluation and this scheme cannot be bound against a traced mesh.

**The shipped self-start, and what the old conditioning number was really measuring** (2026-09-23,
`potential_flow_probe.py`, one run per arm). `hybrid_initialize` seeds a coupled RANS problem with
potential flow: one scalar Laplace solve whose operator carries the field's own gradient
reconstruction. It had stagnated here under both multiple-correction closures, and the exact Jacobian
of that scalar problem measured 3.3e19 / 1.7e19 -- read at the time as possibly a property of this
duct's 10:1 aspect ratio.

| arm | `hybrid_initialize` | exact Laplace Jacobian | march from that seed |
|---|---|---|---|
| `projected` | returns in 5.9 s, plug at `U_IN` | **4.67e4** | converges, 16 target steps, `alpha = 1`, `\|R\|` 5.4e-6 |
| `owner` | stagnates | unreadable (the solve stagnates) | — |
| `repaired` | stagnates | 1.4e5 (see below) | — |

So the ill-conditioning was the **reconstruction's**, not the mesh's, and the aspect-ratio lead is
refuted. ⚠️ **One unexplained reading**: the `repaired` arm's operator materialized at 1.4e5 on a
second call, moments after `hybrid_initialize`'s own solve of what should be the identical operator
stagnated, and it matches neither the recorded 1.7e19 nor the stagnation. Treat that cell as an
anomaly rather than a measurement; `owner` stagnated on both calls, as recorded.

**Which reference the projected weights aim at** (2026-09-23, `reference_target_probe.py`, geometry
binding, one run, `blend=0.75`). The reference is a target, not an ingredient -- the exact weights
form an affine set and the projection lands in it whatever it aims at -- so only its magnitude
reaches the answer, and a cheap target is legitimate:

| target | build | max eig | worst retention | smooth-field error |
|---|---|---|---|---|
| diagonal block **with** the local Schur correction (former) | 2.83 s | +5.42e-4 | 0.4698 | 1.50e-5 |
| **diagonal block alone (shipped)** | **0.05 s** | +5.59e-4 | 0.4671 | 1.51e-5 |
| compact Green--Gauss (`V I`) | 0.02 s | **+1.30e-3** | 0.4777 | 1.57e-5 |

The Schur correction is 2.83 s of a 2.88 s build for a 3 % change in the damping eigenvalue, so it
went; compact is cheaper still and gives away 2.4x on that eigenvalue, which is the quantity the
scheme exists to control, so it did not win. At `blend=0` all three are bit-identical -- the target is
not consulted there at all, which is the probe's own wiring check. `ProjectedStencilGradient` lost its
`boundary_weight` setting with the correction: the block is bit-identical under all three Hessian
boundary closures once the correction is off.

## Layout

- `of_case/make_mesh.py` — gmsh script building the tetrahedral duct mesh (OpenCASCADE + Delaunay),
  with three physical surface groups (`inlet`, `outlet`, `walls`) and one physical volume (`fluid`),
  written as Gmsh MSH format 2.2 ASCII. The element size is tuned (see the script's own `MESH_SIZE`
  comment) to avoid a separate, unrelated hazard: an unstructured Delaunay mesh occasionally leaves a
  cell whose immediate neighbourhood is nearly coplanar, which makes the Hessian-correction normal
  equations singular for a reason unrelated to the corner-face question this case is about.
- `of_case/constant/polyMesh/` — the OpenFOAM mesh `gmshToFoam` writes; not tracked (regenerable, see
  below).
- `compare.py` — the M2 conditioning report and the coupled RANS march; `TET_ARMS` picks the arms
  (`owner`, `repaired`, `hessian`) and `TET_MAX_STEPS` the per-rung cap.
- `laminar_duct_march.py` — the laminar march above; settings are `LAM_*` environment variables.
- `betchen_variants_probe.py` — Betchen's equations written out (`BetchenPrototype`), with two
  reformulations as switches, solved directly and swept, to measure what each costs and buys.
- `betchen_projected_probe.py` — the weight-projection study the shipped scheme came from: which
  target, which blend, which stencil, measured for damping, exactness and accuracy.
- `betchen_locality_probe.py` — how far the converged reconstruction's weights reach, and how fast
  its sweep converges (`BL_WEIGHTS` sweeps the boundary Hessian's weight).
- `rhie_chow_sign_probe.py` — assembles the Rhie–Chow pressure operator for a gradient reconstruction and
  reports where the damping's sign flips: the multiple-correction gradient with its second pass withheld
  on chosen cells, or whole schemes (`RC_SCHEMES`), against geometry alone or the laminar duct's
  pressure conditions (`RC_BINDING`). It orders the laminar march's outcomes (the multiple-correction
  march converges only with the second pass withheld out to ring 3).

### Investigation-only probes (issue #435)

The rest of the scripts here are single-purpose harnesses from the investigation into why the coupled
march does not start on this mesh, kept for a re-adjudication rather than removed. They are a
point-in-time snapshot — see the module docstring of each for its own configuration, and read one before
trusting its output against a checkout that has moved on.

- `potential_flow_probe.py` — the shipped self-start on this mesh: whether `hybrid_initialize` returns
  under each reconstruction, the exact Laplace Jacobian's conditioning, and whether the coupled march
  runs from that seed rather than the hand-built one (`TET_SEED_ARMS`, `TET_SEED_MARCH`).
- `reference_target_probe.py` — which reference the projected weights should be projected toward:
  three per-cell blocks compared on build time, weight size, damping sign and accuracy
  (`REF_TARGETS`, `REF_BLENDS`).
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
