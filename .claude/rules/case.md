---
paths:
  - "aquaflux/case/**"
  - "validation/*/case.yaml"
---

# Rules — `aquaflux/case/` (a whole case described in one YAML file)

> **Provenance boundary (binding).** As with every rule file: what you read here informs your
> understanding, and none of it may reach the shipped surface. See the root `CLAUDE.md`
> **Comment Convention**. Issue numbers below are for *you*; they never go in a docstring or an
> error message.

Tracking issue: #437. Design constraints it is held to: #374 (a loaded case yields a builder, never a
built solver) and #375 (no flat case object — a small core plus two discriminators).

## Status — phases A–D BUILT (2026-09-24); the solver section is NOT

The construction order #374 records is A spec → B topology → C geometry → D equations → E state →
F frozen solver → G drive.
- **A–B, the cheap "check the file" stop:** `read_case(path)` → `CaseFile(spec, directory)`, then
  `CaseFile.check()` → `CheckedCase(spec, mesh)` (the mesh read and `validate()`d, the spec checked
  against its topology, **no geometry**). ~1 s on pitzDaily (12225 cells, ASCII read; one run,
  2026-09-24, macOS arm64).
- **C–D:** `CheckedCase.build()` computes the geometry once and hands it to `spec.physics.build(spec,
  mesh, geometry)`, which returns **the problem the initializers and solves already take** —
  `MomentumContinuity` for `Laminar`, `CoupledRANS` for `RANS` — never a case-specific wrapper (#375) and
  never a built step (#374: the step is rebuilt from mid-march states at every Reynolds rung and refresh,
  so the solver section, when it exists, must be a `state -> NewtonStrategy` builder). ~8 s on pitzDaily
  including the wall distance (same run).
- **E onward are the driver's**: `hybrid_initialize(problem)` and the march, configured in code.

## The layout, and why each part is where it is

| module | holds |
|---|---|
| `spec.py` | `CaseSpec` (mesh, fluid, physics, boundaries, numerics, drive), `Numerics`, the one `SettingsMapping` registry `_CASE_MAPPING`, `case_spec_from_mapping` / `_to_mapping`, `CaseSpec.check_against(mesh)` |
| `case_file.py` | the YAML parse (`_CaseLoader`), `read_case` / `write_case`, `CaseFile`, `CheckedCase` |
| `mesh_source.py` | `MeshSource` → `OpenFOAMMesh` |
| `fluid.py` | `Fluid` |
| `physics.py` | `Physics` → `Laminar` / `RANS` |
| `boundaries.py` | `PatchCondition` → `Inlet` / `Outlet` / `Wall`; `InletTurbulence` → `FixedTurbulence` |

- **`case/` is the TOP layer: nothing else in `aquaflux` imports it**, pinned by
  `tests/unit/test_layering.py::test_nothing_below_the_case_layer_imports_it` (resolves relative imports,
  mutation-checked in all three import forms). The description must not become a dependency of the thing
  it describes.
- **The case-file vocabulary lives here, not beside each runtime class.** The first proposal said "each
  spec value lives in the package that owns its runtime class" (as the preconditioner spec lives in
  `turbulence/`). It does not fit these values: a patch kind is one statement for *every* field — flow,
  `k`, `omega`, wall membership — so no single package owns it, and a `Physics` base shared by a
  `Laminar` in `flow/` and a `RANS` in `turbulence/` would need a home below both that `case/` then
  imports from — a cycle through `case/__init__`. The **schemes, SST constants, variable transforms and
  `BoundaryDriven` are the library's own classes**, read directly (they are plain dataclasses whose
  annotations `SettingsMapping` can check); only what a file needs that no library class expresses gets a
  case value.
- **The top level names no `kind`** — the whole document is the case — and `fluid` / `numerics` may
  omit theirs (one form each). `case_spec_from_mapping` injects them; `_to_mapping` strips them.

## Binding decisions

- **Boundaries are one PHYSICAL kind per patch (decided by the project owner, 2026-09-24), not per-field
  closure dicts.** The per-field form (what every validation driver writes today, and OpenFOAM's `0/U`,
  `0/p`, `0/k`) states one physical boundary four times — momentum, `k`, `omega`, and `wall_patches` —
  which is exactly the double registration #355/#514 has to reconcile by checking. Here the closures and
  the wall set are **derived** from one statement (see below), so they cannot disagree. The cost, accepted: a
  per-field choice becomes a field on the kind (`Wall.k: zero_gradient | zero` — the one that varies in
  the existing cases), and an unusual combination needs a new kind.
- **Turbulence settings sit on the patch; the PHYSICS decides whether they may.** Each `PatchCondition`
  reports `turbulence_settings()` (given here, read only by a closure) and `missing_turbulence_settings()`
  (a closure needs it here and it is absent); `Laminar.refuse_boundaries` refuses the first,
  `RANS.refuse_boundaries` the second, each listing **every** offending path at once. Polymorphic on
  purpose — no `isinstance(physics, RANS)` in `CaseSpec`, so a third physics adds its own rule.
- **The fluid is stated once, with exactly one viscosity (decided by the project owner, 2026-09-24).**
  Either `kinematic_viscosity` or `dynamic_viscosity`, never both, never neither — the other follows from
  `density`. This makes #367's momentum/SST viscosity mismatch unwritable in a file. ⚠️ `SSTTurbulence.build`
  already derives `molecular_viscosity` from a `PropertyModel` (it no longer takes a raw kinematic array),
  so #367's *code* gap is narrower than its text says; the density cross-check in `CoupledRANS.build`
  survives because the two assemblers are still handed two models.
- **The pressure level is fixed exactly once (#500, 2026-09-24).** A top-level `pressure_datum:
  {kind: PinnedPoint, point: [...], value: ...}` is **required** when no patch is an `Outlet` and
  **refused** beside one, checked at read by the flow's own `refuse_an_unsuitable_pressure_datum` over the
  patches' `flow_closure()`s — so `PatchCondition.prescribes_pressure()` was **deleted**: which closure
  fixes the level is the flow's knowledge (`FlowBoundary.prescribes_pressure()`), not restated here.
  `check_against` refuses a point of the wrong dimension or outside the mesh's bounding box (node
  coordinates only; a point inside the box but outside a non-convex domain is harmless — any cell may
  carry a datum). `MassFlow` is still not a registered drive: the channels also need a structured-grid
  mesh source.
- **A moving wall is `Wall` with an optional `velocity` (decided by the project owner, 2026-09-24), not
  its own kind.** A moving wall is a wall in every other respect — no through-flow, a wall to the closure,
  the same `k` option — so one kind keeps those shared by construction. `flow_closure()` is
  `NoSlipWall()` unset, `MovingWall(velocity)` set; `refuse_for_dimension` checks the velocity.
- **`Numerics.momentum_advection` and `RANS.advection` are REQUIRED.** `MomentumContinuity.build`'s
  `advection_scheme=None` means *Stokes flow* — a different problem, not a default — so a file cannot
  reach it by omission. `RANS.advection` is required because `SSTTurbulence.build` takes it positionally.
- **`check_against` needs topology only, and reports every misfit at once**: an unknown patch name (with
  the mesh's boundary patches listed), a named patch that is not a boundary patch (`interior`), boundary
  faces in no named patch (via `FacePatches.uncovered_boundary_faces` — the same query
  `BoundaryConditions.resolve` refuses on), and a patch that does not fit the dimension (an inlet velocity
  of the wrong length). `CaseFile.check` calls `mesh.validate()` even though the OpenFOAM reader already
  validates, so phase B does not depend on which reader produced the mesh.

## ⚠️ The YAML parse is YAML 1.2, deliberately — do NOT swap in `yaml.safe_load`

PyYAML implements YAML **1.1**, whose plain-scalar rules misread ordinary case settings *without a
word*: `1e-5` is a **string** (1.1 floats need a dot and a signed exponent), `no`/`off`/`yes`/`on` are
**booleans**, and `010` is **eight** (octal). And PyYAML keeps the **last** of two duplicate keys, so a
second `inlet:` under `boundaries` would silently drop the first. `_CaseLoader` replaces the bool, int
and float resolvers with the 1.2 core ones, reads integers as decimal, and refuses duplicate keys. Each
is pinned by a test and mutation-checked. `write_case` uses `safe_dump`, whose output the 1.2 loader
reads back equal (floats come out as `1.0e-05`).

## `SettingsMapping` grew three things for this (in `solve/settings_mapping.py`; see `solve.md`)

A **table** (`Mapping[str, X]`: names chosen by the file → entries of one form, read back as a
`MappingProxyType`); a **missing required field** refused by name and path; and a nested value's **own
constructor refusal re-raised with the path prepended** (so `Inlet`'s bad velocity names
`boundaries.inlet`). `LimitedUpwind.limiter`'s `Limiter` annotation had to become a runtime import in
`discretization/advection.py` for the mapping to resolve it.

## What a file cannot describe yet — and where each goes

- **The solver section** (march, preconditioner, convergence, Reynolds schedule) — a `state -> step`
  builder per #374. The preconditioner part already reads through `preconditioner_spec_from_mapping`.
  The validation drivers carry closures no file can state (`point_setup`, the damping tapers reading a
  residual at each rung's seed, the Reynolds companion function, logger metrics); decide which become
  named values and which stay code.
- **`MassFlow`** / a `UniformBodyForce` source and a structured-grid mesh source (the channels).
- **Patch types / `inGroups`** (#364) — would let a file say "all walls" instead of listing each patch.
- **`IntensityLength`** inlet turbulence (`inlet_k` / `inlet_omega`) — `InletTurbulence.inflow(velocity)`
  already takes the velocity for it.
- **A passive-scalar case referring to another case's converged flow** (#375; `bfs3d_species` imports the
  flow driver by path today) — a dependency edge between cases, not a field on one.
- **Profiles** (`DirichletField`, a callable) — not plain data; a named-profile kind if ever needed.

## How the build derives what the drivers used to restate (binding)

- **Each patch kind builds its own closures**: `PatchCondition.flow_closure()` (`Inlet` → `VelocityInlet`,
  `Outlet` → `PressureOutlet`, `Wall` → `NoSlipWall`, or `MovingWall` when it has a `velocity`) and `turbulence_closures()` → `(k, omega)`
  (`Inlet` → both `Dirichlet` from `InletTurbulence.inflow(velocity)`; `Outlet` → both `ZeroGradient`;
  `Wall` → `k` by `Wall.k` (`ZeroGradient` unset, `Dirichlet(0)` for `zero`) and a **placeholder**
  `ZeroGradient` for `omega`, which the closure fixes in the wall cells instead). That table is what every
  validation driver wrote by hand.
- **The wall set is `flow.sheared_patches(momentum.boundary)`** — the one predicate (#514 declined to add
  an `is_wall`), exported from `aquaflux.flow` for this. So the flow/turbulence wall reconciliation #514
  checks at `CoupledRANS.build` holds by construction for a case file.
- **One `PropertyModel`**, `Fluid.property_model()`, built once in the momentum builder and handed to
  `SSTTurbulence.build` as `momentum.properties` — the same object, not an equal one.
  ⚠️ **Its viscosity is `Constant(jnp.asarray(mu))` and its density a plain float — deliberately, and
  this is load-bearing.** A Python number in a module is static to a jitted function, so a continuation
  that rescales the viscosity would recompile the coupled solve at every rung; as an array leaf it is a
  value change. Density is never rescaled. `mu = rho * nu` is computed in that order so it is
  bit-identical to the drivers' `RHO * NU`.
- **An unset setting is not passed**, so the builder's own default applies (`_set`), and nothing restates
  a default here: `gradient_scheme`, `explicit_production_limiter`, `drive`, `k_transform`/`omega_transform`
  (`CoupledRANS.build` takes `None`), and `SSTModel()` when `model` is unset.
- **`_momentum(spec, mesh, geometry)` is the one flow builder** both physics start from, so a setting
  added to the flow reaches laminar and RANS cases together.

## Checking a build: the parity harness, and the trap it had

`validation/case_file_parity.py` builds each case with a `case.yaml` two ways and requires **one pytree**
(same tree structure, static fields included, every array leaf bit-equal) **and** a bit-identical residual
at the reference's hybrid initial condition. References, independent of the files: pitzDaily — a
**frozen copy** of the assembly `compare.py` used to write by hand (the driver cannot be the reference any
more, since it builds from the file); bfs3d — its driver's own `build_case()`, which stays hand-built for
exactly this reason. **Both pass** (2026-09-24, commit 9476606 + this change): pitzDaily |R| 2.884371e+02,
bfs3d |R| 2.178624e+00, both bit-identical. bfs3d is kept out of CI **by design** (project owner, 2026-09-24: its run is too long and its generated
mesh, gitignored under `runs/`, too large), so the harness is a local check — link `runs/` from a checkout
that has the mesh.
The fast tier's own guard is `tests/unit/test_case_file.py`'s build tests on the 2D slab fixture (laminar,
RANS with mixed wall `k`, and all-unset), against hand-built references by the same pytree comparison.
⚠️ **The first version of that comparison could not see a float leaf standing in for an array** — it
converted both with `np.asarray`, so `Constant(4e-3)` and `Constant(jnp.asarray(4e-3))` compared equal,
and the mutation that makes every Reynolds rung recompile passed. Both the test and the harness now
require an array leaf to be matched by an array leaf. **15 of 15 build mutations RED after the fix; two
equivalent mutations dismissed**: building a second, equal `PropertyModel` for the closure (value-identical
by construction), and not passing `drive` (the only nameable drive is the builder's own default).

## The case files in the repository

- **`validation/pitzdaily_openfoam/case.yaml` IS what `compare.py` solves.** `compare.case_spec(model=,
  gradient_scheme=)` reads it and applies study overrides as **edits of the spec** (`dataclasses.replace`),
  so what is not overridden is exactly the file; `build_case()` builds that and returns the same dict as
  before (`coupled`, `momentum`, `turbulence`, `geom`) plus `spec`. ⚠️ **`PITZ_GRADIENT` and `PITZ_K_WALL`
  no longer carry defaults of their own**: unset means "the file's choice", set overrides it — so the
  default is stated once, in the file. `turbulence` is now the coupled system's (boundaries pre-resolved),
  which is what every harness reads. `U_IN` and `NU` (the comparison's scales, read by
  `compare_reynolds_continuation.py`) are read out of the file; `RHO`, `K_IN`, `OMEGA_IN`, `WALLS`,
  `K_WALL_BC` are gone. The reasons for each choice moved into the file's comments.
- **`validation/bfs3d_openfoam/case.yaml` states bfs3d at its driver's defaults and is NOT read by it** —
  kept hand-built as the parity reference. Switching that driver is a separate decision: it is where most
  of the `BFS3D_*` environment configuration lives.
- `tests/unit/test_case_file.py::test_the_shipped_pitzdaily_file_reads_as_the_case_its_driver_builds_and_fits_its_mesh`
  still compares the pitzDaily file against a `CaseSpec` written in the test — now a **second** copy rather
  than a third, and it catches the reader misreading the file, not a physics change (a changed file value
  is a deliberate change to the case, and should be reflected there).
