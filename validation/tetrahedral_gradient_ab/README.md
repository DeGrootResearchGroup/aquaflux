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

## Status

- **Not a physics-validated case.** No OpenFOAM reference is run; the mesh is coarse and the duct
  short, deliberately, to keep the march cheap once it can run at all.
- **Not (yet) a marched case.** See #435.

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
