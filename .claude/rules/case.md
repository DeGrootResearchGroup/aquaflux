---
paths:
  - "aquaflux/case/**"
  - "aquaflux/__main__.py"
  - "validation/*/case.yaml"
  - "validation/*/cases/*.yaml"
---

# Rules — `aquaflux/case/` (a whole case described in one YAML file)

> **Provenance boundary (binding).** As with every rule file: what you read here informs your
> understanding, and none of it may reach the shipped surface. See the root `CLAUDE.md`
> **Comment Convention**. Issue numbers below are for *you*; they never go in a docstring or an
> error message.

Tracking issue: #437. Design constraints it is held to: #374 (a loaded case yields a builder, never a
built solver) and #375 (no flat case object — a small core plus two discriminators).

## ⚠️ THE CASE FILE IS THE USER'S WHOLE INTERFACE (binding — project owner, 2026-09-24)

**Users do not write scripts. Everything a normal user would want to set belongs in the file, and a case
must run from its file alone.** Consequences, each already decided:
- **"Keep it in the script" is never the answer for a user setting** — solver/march settings, initial
  guesses (`BulkVelocity.initial_force`), sources, outputs all go in the file.
- **One file per configuration; a sweep is a set of files.** The channel studies have one file per
  Reynolds number rather than one file a script edits.
- **The validation `compare.py` scripts are developer harnesses** that *read* case files and compare
  against a reference. Where a reference fixes a value (the OpenFOAM channel's ν), the file states it and
  the harness **checks** the file against the reference rather than filling it in.
- **The solver section (BUILT 2026-09-24) and `aquaflux run case.yaml` with an outputs section (BUILT
  2026-09-25) were the critical path**: a case now runs from its file alone. Every validation harness
  takes its solve from its file. A starting state (`initial`, #544) is BUILT: `Checkpoint` (an earlier run's) and `Fields` (an OpenFOAM time directory).
- Do not call the harnesses "drivers" to the project owner — it collides with the *drive* (`MassFlow`).

## Status — phases A–D, the solver section and `run` with outputs BUILT (2026-09-24/25)

The construction order #374 records is A spec → B topology → C geometry → D equations → E state →
F frozen solver → G drive.
- **A–B, the cheap "check the file" stop:** `read_case(path)` → `CaseFile(spec, directory)`, then
  `CaseFile.check()` → `CheckedCase(spec, mesh, directory)` (the mesh read and `validate()`d, the spec checked
  against its topology, **no geometry**). ~1 s on pitzDaily (12225 cells, ASCII read; one run,
  2026-09-24, macOS arm64).
- **C–D:** `CheckedCase.build()` computes the geometry once and hands it to `spec.physics.build(spec,
  mesh, geometry, directory)` (the case file's directory, since 2026-10-05), which returns **the problem the initializers and solves already take** —
  `MomentumContinuity` for `Laminar`, `CoupledRANS` for `RANS` — never a case-specific wrapper (#375) and
  never a built step (#374: the step is rebuilt from mid-march states at every Reynolds rung and refresh).
  ~8 s on pitzDaily including the wall distance (same run).
- **E–F:** `CheckedCase.solve(problem, **observers)` runs `solver_for(spec)` — the file's `solver`, or the
  physics' march with every setting unset — from the state the file's `initial` names, if it has one. #374's "`state -> step` builder" needed no new builder: the
  library solves already rebuild the step from **specs** at every rung and refresh, so the solver section
  turns into the solve's keyword arguments and never into a built step. See "The solver section" below.

## The layout, and why each part is where it is

| module | holds |
|---|---|
| `scopes.py` | `FLOW` / `TURBULENCE` / `RADIATION`, `SCOPES`, the `Scoped` mixin (`setting_scopes`, `required_in_scope`, `settings_in`, `missing_in`) |
| `spec.py` | `CaseSpec` (mesh, physics, boundaries, optional fluid and numerics — the physics decides —, drive, pressure datum, sources, solver, initial, outputs), `Numerics` (momentum and turbulence advection, gradient), the one `SettingsMapping` registry `_CASE_MAPPING`, `case_spec_from_mapping` / `_to_mapping`, `CaseSpec.check_against(mesh)` |
| `case_file.py` | the YAML parse (`_CaseLoader`), `read_case` / `write_case`, `CaseFile`, `CheckedCase` |
| `mesh_source.py` | `MeshSource.read(directory) -> Mesh` → `OpenFOAMMesh` (read) / `StructuredGrid` + `AxisGrading` → `GeometricGrading` (generated) |
| `forcing.py` | `DriveSpec` → `BulkVelocity` (builds `flow.MassFlow`); `SourceSpec` → `BodyForce` (builds `flow.UniformBodyForce`) |
| `fluid.py` | `Fluid` |
| `physics.py` | `Physics` → `_Flow` → `Laminar` / `RANS`, and `Radiation`; the flow physics also map a march state to named physical fields (`restart_fields`) and back to a solve's starting state (`initial_fields`), which a physics with no march (radiation) refuses |
| `initial.py` | `InitialState` → `Checkpoint` / `Fields` (a case's `initial` section: where a starting state is read from, `read(case_directory, spec, mesh)` → `StartingFields`, and `starting_arguments`; `path_fields = ("path",)` so the path rules re-base it) |
| `axes.py` | `AXES`, `AxisName` (the `x`/`y`/`z` a file names; `BulkVelocity.direction`, `Fields.extruded_axis` and `OpenFOAMTime.extruded_axis` all use it) and the one refusal of an extruded axis stated for a three-dimensional mesh |
| `kinematic.py` | `kinematic_pressure` / `pressure_from_kinematic`: the one home of OpenFOAM's pressure-per-unit-density factor, used by `OpenFOAMTime` going out and `Fields` coming in |
| `restart_file.py` | the checkpoint file a case writes and reads: `RestartHeader` (physics, `n_cells`, `dim`, mesh digest, case digest, problem digest), `mesh_digest`, `checkpoint_writer`, `read_restart` → `RestartFile` (with `resumption`), `refuse_non_finite` |
| `boundaries.py` | `PatchCondition` → `Inlet` / `Outlet` / `Wall` / `Lamp`; `InletTurbulence` → `FixedTurbulence` |
| `radiation.py` | the radiation vocabulary: `LampProfile` → `LambertianProfile` / `CosinePowerProfile` / `IesProfile`; `UniformMedium`; `SurfaceSource` → `MeshPatch` / `StlSurface` / `CadSurface` (+ `Coarsen`, `CadPlacement`); `OccluderSpec` → `PatchBody` / `StlBody` / `CadSolid` / `CadFluid`; `Receivers`; the internal `PatchSurface`, `_Drawings`, `_facing` |
| `paths.py` | `named_paths` / `with_paths` / `relocated`: every value naming a file declares `path_fields` (a `ClassVar`), found at any depth for the existence check; `relocated(value, source, target)` is THE re-basing rule (relative → re-based, absolute kept, `outputs.directory` never — it names no file), used by the `case.yaml` run record and `aquaflux write --relative-to` |
| `solver.py` | `SolverSpec` → `CoupledMarch` / `FlowMarch` (both on the private `_March`, their shared settings) / `Segregated` / `RadiationSolve`; `ViscosityRamp`; `RootSolve`; `solver_for(spec)`; `NotConverged`; each kind's `observers_for(logger, recorder)`, `refuse_initial(initial)` and `solve(problem, initial=..., resume=...)` |
| `outputs.py` | `Outputs(directory, fields, log, history, checkpoints)`; `RunFields(cells, patches)`; `FieldWriter` → `Vtk` / `OpenFOAMTime` / `PatchVtk`; `Checkpoints` |
| `run.py` | `prepare_run(path, overwrite)` → `PreparedRun.run(terminal)` → `RunRecord` |
| `aquaflux/__main__.py` | the `aquaflux check` / `aquaflux run` command (console script `aquaflux`) — the ONE module outside `case/` allowed to import it |

- **`case/` is the TOP layer: nothing else in `aquaflux` imports it** — except `aquaflux/__main__.py`,
  the command-line entry point, which runs a case file and so sits above everything (project owner,
  2026-09-25; `ABOVE_THE_CASE_LAYER` in the test) — pinned by
  `tests/unit/test_layering.py::test_nothing_below_the_case_layer_imports_it` (resolves relative imports,
  mutation-checked in all three import forms). The description must not become a dependency of the thing
  it describes.
- **The case-file vocabulary lives here, not beside each runtime class.** The first proposal said "each
  spec value lives in the package that owns its runtime class" (as the preconditioner spec lives in
  `turbulence/`). It does not fit these values: a patch kind is one statement for *every* field — flow,
  `k`, `omega`, wall membership — so no single package owns it, and a `Physics` base shared by a
  `Laminar` in `flow/` and a `RANS` in `turbulence/` would need a home below both that `case/` then
  imports from — a cycle through `case/__init__`. The **schemes, SST constants and variable transforms are
  the library's own classes**, read directly (they are plain dataclasses whose
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
- **Which physics reads a setting is DECLARED once, as a scope (`case/scopes.py`, 2026-10-06), and both
  the refusals and the browser form read that declaration.** A value (each `PatchCondition`, `Numerics`)
  lists `setting_scopes` (`{"k": TURBULENCE, "velocity": FLOW, "reflectance": RADIATION, ...}`) and
  `required_in_scope`; each physics lists `reads_scopes` (Laminar `flow`; RANS `flow`+`turbulence`;
  Radiation `radiation`). `Scoped.settings_in(scope)` / `missing_in(scope)` replaced the hand-written
  `turbulence_settings()` / `missing_turbulence_settings()` / `flow_settings()` / `radiation_settings()`
  (deleted; each was reproduced exactly by the derivation, including the lamp's `("profile", "power")`).
  The physics refuses a stated setting of a scope it does not read and a missing required one, listing
  **every** offending path at once (`_refuse_stray` per scope on the boundaries; `_Flow.refuse_sections`
  generically over `numerics`). The schema publishes `scope` / `required_in_scope` per field and `reads`
  per physics, with `scopes_from: "physics"`, so the form shows a laminar case no turbulence setting and
  a flow case no light setting (`ui.md`). Polymorphic on purpose — no `isinstance(physics, RANS)` in
  `CaseSpec`. Mutation-checked 9/9 (2026-10-06).
- **The fluid is stated once, with exactly one viscosity (decided by the project owner, 2026-09-24).**
  Either `kinematic_viscosity` or `dynamic_viscosity`, never both, never neither — the other follows from
  `density`. This makes #367's momentum/SST viscosity mismatch unwritable in a file. ⚠️ `SSTTurbulence.build`
  already derives `molecular_viscosity` from a `PropertyModel` (it no longer takes a raw kinematic array),
  and `CoupledRANS.build` cross-checks both properties (#367, `refuse_a_fluid_the_flow_disagrees_with`),
  because the two assemblers are still handed two models.
- **The pressure level is fixed exactly once (#500, 2026-09-24).** A top-level `pressure_datum:
  {kind: PinnedPoint, point: [...], value: ...}` is **required** when no patch is an `Outlet` and
  **refused** beside one, checked at read by the flow's own `refuse_an_unsuitable_pressure_datum` over the
  patches' `flow_closure()`s — so `PatchCondition.prescribes_pressure()` was **deleted**: which closure
  fixes the level is the flow's knowledge (`FlowBoundary.prescribes_pressure()`), not restated here.
  `check_against` refuses a point of the wrong dimension or outside the mesh's bounding box (node
  coordinates only; a point inside the box but outside a non-convex domain is harmless — any cell may
  carry a datum). `MassFlow` is still not a registered drive: the channels also need a structured-grid
  mesh source.
- **Inlet turbulence is `FixedTurbulence(k, omega)` or `IntensityLength(intensity, length)` (2026-09-25).**
  `k = 1.5 (I |U|)^2` (`|U|` the inflow SPEED, so direction does not matter), `omega = sqrt(k) /
  (beta_star^(1/4) L)` — the OpenFOAM `turbulentIntensityKineticEnergyInlet` / `turbulentMixingLength
  FrequencyInlet` pair. **`C_mu` is the case's own `SSTModel.beta_star` (project owner, 2026-09-25), not
  a fixed 0.09**, so a file that changes the model constant cannot disagree with its inlet; this is why
  `PatchCondition.turbulence_closures(model)` and `InletTurbulence.inflow(velocity, model)` take the model,
  and `RANS.build` hands every patch the SAME model it builds the closure with. A zero inflow speed is
  refused (no turbulence to take an intensity of). The step cases' files stay `FixedTurbulence`: their
  OpenFOAM `omega` values are rounded (`IntensityLength(0.05, 0.1 h)` gives 440.17 against pitzDaily's
  440.15, `(0.05, 0.07 H1)` 1597.19 against bfs3d's "~1600"), so switching would change the problem.
  Mutation-checked 8/8, including the build ignoring the case's model and the inlet ignoring it.
- **A moving wall is `Wall` with an optional `velocity` (decided by the project owner, 2026-09-24), not
  its own kind.** A moving wall is a wall in every other respect — no through-flow, a wall to the closure,
  the same `k` option — so one kind keeps those shared by construction. `flow_closure()` is
  `NoSlipWall()` unset, `MovingWall(velocity)` set; `refuse_for_dimension` checks the velocity.
- **`Numerics.gradient` offers four schemes: `CompactGreenGauss`, `CorrectedGreenGauss`,
  `MultipleCorrectionGradient`, `ProjectedStencilGradient`.** The last was BUILT 2026-09-22 (two days
  before the case file existed) and left out of `_CASE_MAPPING` until 2026-10-06, though `schemes.md`
  names it THE scheme for tetrahedral meshes; its `prepared` weight cache is `not_settings`. Checked
  2026-10-06 by two coupled steps on pitzDaily through a case file naming it (finite residual, 3.4e-2).
  `HessianCorrectedGradient` is still unregistered (dominated on cost per `schemes.md`; not asked for).
- **Every advection scheme is in `numerics` (project owner, 2026-10-06): `momentum_advection` (REQUIRED)
  and `turbulence_advection` (scope `turbulence`: required by RANS, refused by Laminar).**
  `MomentumContinuity.build`'s `advection_scheme=None` means *Stokes flow* — a different problem, not a
  default — so a file cannot reach it by omission; the turbulence one is required because
  `SSTTurbulence.build` takes it positionally. ⚠️ There is no `RANS.advection` or
  `RANS.turbulence_advection`: the key was `physics.advection`, renamed `turbulence_advection` and then
  moved to `numerics` the same day (`numerics.gradient` is already every field's, so the advection schemes
  sit beside it).
- **A `boundaries` key names a patch OR a patch group (decided by the project owner, 2026-09-25, #364).**
  `CaseSpec.boundaries` keeps the file's keys as written (so a case round-trips unchanged);
  `CaseSpec.patch_conditions(mesh)` resolves them to `{patch: condition}` through
  `FacePatches.addressed_by`, and **every build reads that per-patch form, never `spec.boundaries`
  directly** — both `_momentum` and `RANS.build` do; a third build reading the keys would hand a group
  name to `BoundaryConditions.resolve`, which knows only patches. Refused, all at once, by
  `_patch_conditions` (shared by the check and `patch_conditions`, so the two cannot disagree): a
  patch reached twice (its own name and a group, or two groups), a name that is both a patch and a group
  of *other* patches (a group holding only its namesake is harmless and allowed), and a group member that
  is not a boundary patch. The mesh-free checks (`refuse_boundaries`, the pressure-datum rule) still read
  the keys: neither depends on how many patches a key reaches. Tests: a group key builds the same problem
  as its patches stated one by one, under Laminar and RANS (with `k: zero` so a dropped condition shows);
  mutation-checked with the mesh-side record, 24/24 red.
- **`check_against` needs topology only, and reports every misfit at once**: an unknown patch name (with
  the mesh's boundary patches and patch groups listed), a named patch that is not a boundary patch
  (`interior`), a patch reached twice, boundary faces in no named patch (via `FacePatches.uncovered_boundary_faces` — the same query
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

- **A starting state on a mesh that is not an OpenFOAM one** — `Fields` is the cells of an OpenFOAM case in its own numbering, so it refuses any other `MeshSource`; the only cross-check is the cell count.
- **A laminar case holding a bulk velocity (#541)** — `FlowMarch` refuses one (`solve_flow_march` refuses a
  `MassFlow` drive) and there is no laminar segregated solve; `bulk_velocity_flow_solve` exists but is a
  bordered Newton, not a march.
- **A coupled march holding a bulk velocity (#541)** — `CoupledMarch` refuses one and points at `Segregated`.
  `solve_coupled_mass_flow` exists, but it is the older `RootSolver` path: `BlockDiagonal` only, no dual
  time, no retry, no step control, no ramp. Routing a file there would publish a surface that silently
  cannot take half the section's settings; the gap is the library's (#277/#448 made the march generic but
  not the bordered constraint).
- **A graded or periodic 3D structured grid** — `structured_grid_3d` has neither, so a 3D `StructuredGrid`
  (three counts, three lengths; patches `left/right/bottom/top/back/front`) is uniform and refuses both.
  It was 2D only until a radiation case needed a small box (2026-10-05).
- **Other source kinds** — a `sources:` section beyond `BodyForce` waits on #362 (sources declaring their
  inputs); `BodyForce` reads no field and no gradient, so it needs nothing #362 would add.
- **A passive-scalar case referring to another case's converged flow** (#662, split from #375; `bfs3d_species`
  imports the flow driver by path today) — a dependency edge between cases, not a field on one.
- **Profiles** (`DirichletField`, a callable) — not plain data; a named-profile kind if ever needed.

## How the build derives what the drivers used to restate (binding)

- **Each patch kind builds its own closures**: `PatchCondition.flow_closure()` (`Inlet` → `VelocityInlet`,
  `Outlet` → `PressureOutlet`, `Wall` → `NoSlipWall`, or `MovingWall` when it has a `velocity`) and `turbulence_closures(model)` → `(k, omega)`
  (`Inlet` → both `Dirichlet` from `InletTurbulence.inflow(velocity, model)`; `Outlet` → both `ZeroGradient`;
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

## Periodic channels: the mesh source, the drive and the sources (2026-09-24)

- **`MeshSource.read(directory) -> Mesh` is the contract**, not `reader()`: a generated grid has nothing to
  read. `OpenFOAMMesh` keeps its `reader()` and implements `read` through it.
- **`StructuredGrid(cells, lengths, periodic, grading)`** builds `structured_grid_2d(named_boundaries=True)`
  — a file always gets the side-named patches, minus a periodic axis's two sides. `periodic` is
  `tuple[Literal["x"], ...]` because the generator supports `x` only; `grading` is a table keyed by axis.
  A whole-number float cell count (`96.0`) is normalized to an int, since the settings mapping accepts one
  in an int position.
- **The drive is a case-side family, `DriveSpec` → `BulkVelocity(target, direction, initial_force)`,** and
  the library's `BoundaryDriven` is **no longer a kind a file names**: unset *is* boundary-driven, and a
  second spelling of the default was removed rather than kept. `BulkVelocity` exists (instead of
  registering `flow.MassFlow`) because `MassFlow.force` is an array leaf the mapping cannot check;
  `direction` is `x`/`y`/`z`, mapped to the index. `initial_force` is the multiplier's seed — state, not
  problem — kept in the file under the user-interface rule above, and documented as a guess.
- **`sources: [BodyForce(force)]`** → `UniformBodyForce`. Named `BodyForce`, not `UniformBodyForce`, so
  the case vocabulary and the flow's classes do not share a name (the mapping's registry would allow it;
  a reader would not).
- **Each has `refuse_for_dimension`, and `check_against` runs every one** — patches, drive, sources — from
  one list, reporting all misfits at once.
- **Mutation-checked: 21 of 21 RED** (grid periodic/grading/naming, every refusal, the float-count
  normalization, the drive's direction/seed/dimension, the force's sign/count/dimension, and the build
  dropping the drive or the sources).

## The solver section (2026-09-24) — binding decisions

- **Three kinds, each the library solve of the same shape.** `CoupledMarch` → `solve_coupled`, or
  `solve_reynolds_ramp` when `continuation` is set; `FlowMarch` → `solve_flow_march`; `Segregated` →
  `solve_segregated` (flow solve `bulk_velocity_flow_solve` under a `MassFlow` drive, else
  `reused_flow_solve`; scalar solve `scalar_pseudo_transient_solve`; start `sst_initial_fields`). Each
  refuses the physics it does not solve (`refuse_for(physics, drive)`, run in `CaseSpec.__post_init__`) and
  the two marches refuse a `BulkVelocity` drive, which they would hold at its starting force.
- **The shared march settings are ONE dataclass, `_March`, that both marches derive from** — `max_steps`,
  `convergence`, `preconditioner`, `dual_time`, `linear_solve`, `step_control`, `retry` — so a setting added
  there reaches both (Principle 2's sibling rule, at the file surface). `CoupledMarch` adds the closure's:
  `turbulence_damping` (→ `shift=CoupledShiftSettings(turbulence_damping=...)`), `positivity_floor`,
  `positivity_projection`, `continuation`.
- **The library's own settings values are read directly** — `Convergence` (+ `RowScaled`/`BlockScaled`/
  `Euclidean`), `DualTimeLoop`, `LinearSolveSettings`, the three dual-time controls (equinox modules are
  dataclasses; their float fields read), `RetryPolicy`, and every coupled-preconditioner kind via
  `turbulence.PRECONDITIONER_SPEC_MAPPING.kinds` (made public for this, so a kind added there reaches a
  case file). Case-side values exist only where no library value fits: `ViscosityRamp` (the ramp's
  arguments are loose keywords, and its `companion` is a function — `scale: flow|both` names
  `scale_momentum_only` / `scale_both_blocks`), and `RootSolve` (`RootSolveSettings` holds `lineax`
  solvers; `test_a_root_solve_states_every_setting_of_the_librarys_root_solve` pins the two field lists
  equal, so they cannot drift). `RetryPolicy.solver` became a `LinearSolverSpec` so that no case value was
  needed there (see `solve-globalization.md`).
- **Unset means the library's default, never one restated here** (`_set`). The file-level consequence:
  a pitzDaily file with **no** solver section runs `CoupledMarch()` — a single-step march with no ramp —
  which is the recorded reachability crawl and does not converge in `max_steps`. The shipped default was
  not changed (that needs the project owner, #542); the validation files state their calibrated values.
- **A script can observe a solve, never configure it.** `solve(problem, **observers)` refuses any keyword
  that is one of the solve's settings **whether the file sets it or not** (`_owned()`, derived from the
  `_March` fields plus `shift`/`positivity_*`/`homotopy`) — an unset setting is still the file's, since its
  default is part of what the file says. A materialized-Jacobian preconditioner is **opened as a session
  here** (`open_session(spec, problem, **session_options)`), so a harness passes the session's *observers*
  (`observer`, `reports`, `on_build`, the wrappers) rather than a session of its own that might hold other
  settings; `session_options` beside a `BlockDiagonal` is refused. A `point_setup` may be passed to observe
  a ramp's anchor, and is refused if it returns any settings. `Segregated` takes no observers, because
  `solve_segregated` has none, and it only WARNS when it runs out of sweeps (#543).
- ⚠️ **`station_step` and `jacobian_gradient_sweeps` are NOT in `_owned()` and are not file settings.**
  The pitzDaily harness's study arms need them; they reach the library through that harness's own study
  path, not through `solve(**observers)`. `station_step` reshapes the path, so passing it as an "observer"
  would be a configuration back door — the harness does not, and nothing else should.
- **Why the parity is on the keywords, not on a pytree.** The solve's arguments are specs and settings
  values, so `tests/unit/test_case_solver.py` replaces each library solve with a recorder and compares
  what the case hands it with a **frozen copy** of what each harness passed by hand before it read its
  file (pitzDaily, bfs3d, all five channel files). The recorder tests also pin every dispatch branch.
  That a solve really runs is `tests/integration/test_case_solve.py` (a laminar channel from a file,
  bit-identical to the direct `solve_flow_march` call with the same settings, ~45 s).

## Running a case (2026-09-25) — binding decisions

- **`aquaflux check case.yaml` / `aquaflux run case.yaml [--overwrite]`** (also `python -m aquaflux`).
  Exit status: `0` converged (or checked), `1` a run whose solve stopped short, `2` a refused file (reason
  on stderr, no traceback). Only reading/checking errors map to `2` — `prepare_run` does all of that
  before `run()` starts, so an error from inside a solve still surfaces as a traceback rather than as a
  one-line "refused file".
- **`prepare_run` is the cheap stop**: read, refuse an occupied output directory (or an existing
  `OpenFOAMTime` target), resolve the solver (a bulk-velocity case with no solver is refused HERE), check
  the mesh, and read the `initial` state against it — all before any geometry. **What counts as occupied
  has one home, `run._plan`**, shared with **`plan_run(path) -> RunPlan(directory, log, history,
  occupied)`** (2026-10-07), which answers the same question without checking the mesh; `aquaflux plan
  case.yaml` prints it as `{"error", "directory", "log", "history", "occupied"}` and `serve` answers it. The
  browser interface's Run section asks it before offering to replace results (project owner, 2026-10-07:
  "only overwrite with permission"). `--overwrite` replaces the files the run writes and clears
  `checkpoints/` (whose names collide across runs) **after** the starting state has been read; anything
  else in the directory is left alone. See "A starting state" below.
- **Editing commands for a program (2026-10-05, for the browser interface's Setup section)** — `aquaflux
  schema` prints `case_schema()` (root `CaseSpec`, `one_form_sections`, and `_CASE_MAPPING.schema()`);
  `aquaflux show case.yaml` prints `{"error", "case"}` — the document by `read_case_document` (the SAME
  YAML 1.2 parse `read_case` uses, factored out so the two cannot differ) and the refusal of reading it
  into a case, if any (a refused file is still shown so it can be corrected; unparseable YAML → exit 2);
  `aquaflux write case.yaml [--relative-to DIR] < {"case": ...}` validates through
  `case_spec_from_mapping` and writes with `write_case`, writing NOTHING when refused (exit 2, JSON
  error). Saving rewrites the file: comments and hand formatting are not kept, settings at their default
  are dropped — the UI warns before the first save over an opened file. The UI never imports the
  solver: it asks these through ONE long-lived `aquaflux serve` process (below), not a process per command. Pinned by
  `tests/unit/test_case_editing_commands.py`. Fourth command, `aquaflux mesh OUT [--relative-to DIR] <
  {"mesh": ...}`: reads only the mesh section (`mesh_source_from_mapping`, the same rules as within a
  case, so a mesh can be previewed while the rest is unfinished), reads or generates and validates it,
  writes `OUT/mesh.vtu`, and prints `cells`, `dim`, boundary `patches` (name + face count, holding faces
  only) and `groups`.
- **`aquaflux serve` (2026-10-06, project owner: "build the persistent worker")** — answers `SERVED`
  (`schema`/`show`/`write`/`mesh`/`check`/`plan`, never `run`) one JSON line per request
  (`{"arguments", "stdin"}` → `{"status", "output", "errors"}`) by calling `main(arguments)` in-process
  with stdin/stdout/stderr swapped for strings; a `SystemExit` (argparse) or any exception becomes that
  request's status, and the loop goes on; Ctrl-C (run by hand) or a closed reply pipe ends it quietly. **Replies go on a `dup` of fd 1, and fd 1 is then pointed at
  stderr**, so a library or compiled code writing to stdout cannot corrupt the protocol
  (`test_serve_replies_on_a_channel_nothing_else_writes_to`). Why it exists: each command's work is
  0.01–0.4 s while a fresh process spends 2.5–5 s importing (JAX ~1.1–1.9 s, SciPy via `mesh.distance`,
  the flow package via `case.boundaries`) — measured 2026-10-06 on pitzDaily, macOS arm64, under a load
  average of 12–24 from a concurrent fast gate, so the absolute seconds are inflated; opening a case was
  two such processes (schema, show) then a third (mesh). Through the worker: start + schema 1.5 s
  (in the background at page load), show 0.02 s, first mesh 2.0 s, a repeat mesh 0.3 s.
- **`paths.relocated(value, source, target)` — one home for "this case file moved directories"** (on
  `with_paths`, so it reaches every `path_fields` path at any depth: mesh, OpenFOAM writer case, lamp
  photometry, STL/STEP surfaces); absolute paths are unchanged and `outputs.directory` is NOT re-based (a
  copy's runs write beside the copy). The run record's `case.yaml` and `aquaflux write --relative-to`
  both use it. ⚠️ There is no `CaseSpec.relocated` / `MeshSource.relocated` / `rebased_path`: a
  per-class version was built for the browser interface in parallel with `with_paths` (2026-10-06) and
  deleted at the merge as dominated — it reached only the mesh and `OpenFOAMTime`.
- **`outputs` is optional; its default writes `results/fields.vtu` + `results/march.log` +
  `results/history.csv`** (project owner, 2026-09-25; `history` added 2026-10-05 for the browser viewer's
  convergence plot). `history` is `solve.StepHistory` — one CSV row per step, every `StepReport` field
  at full precision, flushed per row, plus refits, retry reasons and the per-equation residuals (see
  `solve-march.md`); the runner's `_StepCount` forwards each march hook to every recorder that has it
  (`on_checkpoint` to the history and the checkpointer; `on_retry`, `on_refresh` and `on_residuals` to
  the history). Its header is written with the first row, so under `Segregated` (no steps) it stays
  empty. `log` and `history` may not name
  the same file. `Outputs` is a `default_factory` field, so a file stating none round-trips with no section.
  Like `fluid`/`numerics`, the section may omit its `kind`; nested values (`Checkpoints`, writers) may not.
- **Three writers, all the library's** (`PatchVtk` → `write_patches` since 2026-10-05, see Radiation cases): `Vtk` → `write_vtu` (any mesh) and `OpenFOAMTime` →
  `write_openfoam_time` (writes into the named OpenFOAM case, not the output directory; refused at read
  unless the mesh is an `OpenFOAMMesh`). `OpenFOAMTime.case` is **required**, not derived: pitzDaily's
  mesh path is `runs/kwsst/polyMesh`, not a `constant/polyMesh` layout, so the case directory cannot be
  recovered from it. `time` is a string (quote it). Each writer's `fields` selects by name, because an
  OpenFOAM template may not hold every field (`nut`).
- **What is written is each physics' `output_fields(problem, solution)`** — `U`, `p` (the SOLVED pressure,
  not `coupled_fields`' gauge-free one), and under RANS `k`, `omega`, `nut`; the log's per-step field
  changes are `progress_fields(problem)` (`coupled_fields` for RANS, none for laminar).
- **Each solver kind attaches the log and the recorder itself** (`observers_for(logger, recorder)`): a
  march wires `on_checkpoint` and `on_retry` to both (via `combine_observers`, in `solver._both`),
  `on_residuals` to the recorder only when it has a taker (`_StepCount.on_residuals` is `None` without
  one, since each costs a residual evaluation per step), and `inner_observer` only with a dual-time loop;
  `CoupledMarch` adds the session's refresh observer (log and recorder) and the ramp's point label;
  `Segregated` gets nothing (#543). The runner never names a library keyword itself.
- **A solve that stops short is a record, not an error**: `NotConverged` (the case layer's, raised by
  `Segregated` from the loop's own warning — a stopgap until #543) and `EquinoxRuntimeError` (the marches'
  non-root refusal) end the run with `converged: false`, no fields, but the log, checkpoints and records
  written. **An interrupt is one too (2026-10-07)**: `KeyboardInterrupt` during the solve — Ctrl-C, or
  the browser interface's Stop, which sends SIGINT to the run's process group — records `message:
  interrupted before it converged` and exits 1, so a stopped run leaves `run.yaml` rather than a
  traceback and nothing. Any other exception propagates. `_SEGREGATED_NOT_CONVERGED` is matched against the loop's
  warning by its opening words, and a test asserts those words are in `solve_segregated`'s source.
- **Provenance**: `case.yaml` is the spec as it ran — `solver_for(spec)` written out, relative mesh and
  `OpenFOAMTime.case` paths **re-based on the output directory** (`os.path.relpath`), `outputs.directory`
  `"."` — so the copy reads and checks where it lies. `run.yaml` records case path, aquaflux version,
  commit and whether tracked files were modified (`git -C <package dir>`; `null` outside a checkout),
  start time, seconds, solver kind, converged, steps and last residual (counted by `_StepCount` on
  `on_checkpoint`; `null` for `Segregated`), message, and what was written.
- **Tests**: `tests/unit/test_case_run.py` (section reading/refusals, writer selection, both physics'
  output fields, observer wiring per kind, the segregated refusal, `prepare_run`'s refusals and overwrite,
  the command's exit statuses, `python -m aquaflux --help`); `tests/integration/test_case_run.py` (a
  laminar channel run: files, the checkpoint's `U` and `p` equal to the direct `solve_flow_march` root bit for bit, the
  records; a run that stops short; an `OpenFOAMTime` directory on the slab fixture read back against the
  direct solve's pressure, and the re-based record re-checked). ⚠️ The two-cell slab leaves the default
  `MultipleCorrectionGradient` underdetermined and the potential-flow initializer's Laplace operator
  NaN — that test states `CompactGreenGauss`.

## A starting state (2026-10-02, #544) — binding decisions

- **One optional top-level `initial` section, one kind so far: `Checkpoint {path, step: latest | <int>}`.**
  `path` is the earlier run's `checkpoints` directory, relative to the case file. Unset is today's
  self-start. `Checkpoint` is singular and `Checkpoints` (the `outputs` writer) plural: one reads, one writes.
- **A checkpoint written by a case holds PHYSICAL fields plus a header, not the march's solved state
  (project owner, 2026-10-02).** The solved state holds `log(omega)` under `LogScalars`; a file of it read
  by a direct-`omega` case has the right length and so passes every size check, and loads as garbage.
  `Physics.restart_fields(problem, state)` maps a march state to `U`, `p` (the solved, real pressure) and
  under RANS `k`, `omega`; `Physics.initial_fields(problem, fields)` is its inverse (`state_from_physical`
  for RANS). Written by `checkpoint_writer`, injected as the `save=` of the run's `StateCheckpointer`, so
  `solve/` stays mesh-free. A library `StateCheckpointer` with its default serializer still writes a bare
  `state` array — the validation harnesses (`pitzdaily_openfoam`, `bfs3d_openfoam`) use that and are
  unchanged; `read_restart` refuses such a file with a message saying why.
- **The header is the physics kind, `n_cells`, `dim`, a mesh digest, and the writing case's two digests.**
  The case digest (`CaseSpec.digest()`, the whole file) is provenance and is never compared (a restart is
  usually a case with something changed); the PROBLEM digest (`CaseSpec.problem_digest()`) is compared, for
  one purpose only — see the reference residual below. A file written before the problem digest was added
  is refused as "not a case checkpoint" (pre-release: no shim). `mesh_digest` hashes the face-to-cell connectivity and the node coordinates **binned to a
  millionth of the mesh's largest extent**, so the same mesh read on another machine (last-bit
  differences) agrees while a stretched or renumbered one does not — a cell count alone would load a
  renumbered mesh's field without complaint. It needs no geometry, so the check runs in `prepare_run`.
  The digest is reported only when count and dimension agree (it differs whenever they do not).
- **`read` and `starting_arguments` are split.** `InitialState.read(case_directory, spec, mesh)` finds the
  file and checks it against the mesh and the case (no geometry, no equations) → `StartingFields`;
  `starting_arguments(starting, physics, problem)` maps those onto the built problem as the solve's
  keywords (`initial=`, `resume=`). `prepare_run` reads, so a
  state that does not fit is refused before the 8 s build; `PreparedRun.starting` carries it.
  `CheckedCase` gained `directory` (a relative path needs it) and `starting_fields()`, and
  `CheckedCase.solve` honours the file's `initial` so a script cannot silently drop it.
- **Every solve takes `initial=` and `resume=`, on the ABC.** `CoupledMarch` →
  `solve_coupled(problem, *initial, resume=)`, `FlowMarch` → `solve_flow_march(...,
  state=initial, resume=)` (each passed only when set, so the parity recorders are
  unchanged), `Segregated` replaces `sst_initial_fields` and refuses a `resume` (its loop stops on the
  sweep-to-sweep change and has no residual history). Neither is one of `_owned()`'s settings: they are
  the case's starting state, not march settings. `RadiationSolve.solve` takes both and refuses a non-`None`
  one; a radiation CASE refuses an `initial` section when read, through
  `Radiation.refuse_sections` beside its fluid/numerics/drive refusals (`restart_fields` / `initial_fields`
  on a physics with no march raise).
- **A `CoupledMarch` with a `continuation` REFUSES `initial` when the file is read (project owner,
  2026-10-02), and again at `solve`.** `solve_reynolds_ramp` seeds from `hybrid_initialize(companion(coupled,
  anchor))` because the seed must match the anchor's viscosity (a seed at the target's put `omega` 90x off at
  the near wall on `bfs3d`). The refusal is `SolverSpec.refuse_initial`, run in `CaseSpec.__post_init__`
  beside `refuse_for`; the one message is shared by both sites. Resuming mid-ramp (at the station a
  checkpoint came from) needs the checkpoint to record its station — **not built**; the message says to drop
  `continuation`, which restarts at the case's own viscosity.
- **A restart writes to its own output directory (project owner, 2026-10-02).** `--overwrite` clears
  `<output>/checkpoints`, and without it an occupied directory is refused, so a case could never read its
  own checkpoints. `_refuse_a_start_from_the_output` refuses an `initial` whose resolved location is inside
  the output directory, **whether or not overwrite is set** and before anything else. In `prepare_run` the
  checkpoints are cleared only **after** the state has been read and checked — a refused restart must not
  have cost an earlier run its checkpoints (`test_a_restart_that_does_not_fit_leaves_the_checkpoints_it_was_told_to_overwrite`).
- **A checkpoint with non-finite fields is refused (`RestartFile.refuse_if_not_finite`).** The
  checkpointer writes the failed step too (the "known defect" in `solve-march.md`), so the newest file can
  be the state the march died in. The message says to name an earlier `step`.
- **A restart carries the stopped march's HISTORY — three numbers, one value `solve.Resumption` (built
  2026-10-07; the shift and the anchor 2026-10-08).** ⚠️ **An earlier entry said the *shift*
  (`ShiftStrengthControl`, `beta_start`) of a default flow march restarted, and that was a misdiagnosis —
  kept as a trap.** The default flow march's damping is the memoryless switched-evolution schedule
  `beta = max(floor, beta0 (|R|/|R0|)^p)`, which reads its anchor `|R0|` from the state the march is
  *handed* and holds nothing between steps (a report's `shift` reads `0.0` for it because it is not a
  readable beta, which looked like a reset). Resumed at the fresh march's step-3 state, `|R0|` is
  re-measured there, so the ratio is ~1 and beta opens at `beta0` again. Verified by giving the resumed
  march `beta0 = 2 |R0_resumed| / |R0_fresh|`: its first iterate equalled the fresh march's fourth (diff
  0.0), all later ones to 7.8e-16, 9 steps as the fresh march had left (8x4 laminar channel, `FlowMarch`
  defaults, `FirstOrderUpwind`, Re ~ 50; one run, 2026-10-07). A **dual-time** march, by contrast, does
  hold a shift between steps — the Courant control's `(beta, memo)` — and that one was not carried until
  2026-10-08. Measured then (24 x 16 laminar channel, `mu = 5e-3`, `DualTimeLoop(inner_steps=3)`, the
  default `DualTimeControl`, row-scaled `atol = 1e-9`): the fresh march takes 17 steps, its shift walking
  2.0 → 0.02; resumed at its step 3, 6 or 10 with only the reference it took 16, 16 and 14 steps (against
  14, 11 and 7 left) and opened at shift 2.0 each time.
  - **What is carried, and where each number comes from.** The step record a checkpoint stores holds
    `residual_norm`, `residual_ratio = residual_norm / reference_norm`, `shift` and
    `damping_reference` (a `StepReport` field: the `|R0|` the step's damping was anchored at). `Resumption`
    is `(reference_residual, damping_reference, shift)`: the reference is the quotient (the original
    `|R0|` in the measure it was taken in, and the scale of the stopping bar `atol + rtol * reference`);
    the anchor is the record's own (it differs from the reference once the march has refreshed its
    preconditioner, since a later segment re-bases its damping at the refresh); the shift is the last step's, a `0.0` ("no step control") read as none.
    `RestartFile.resumption` builds it (`None` for a record with no usable ratio) and each number a record
    does not hold reads as unset.
  - **Carried only when the case states the SAME PROBLEM in the SAME MEASURE.** The numbers are for the
    problem and the measure they were taken in. `CaseSpec.problem_digest()` is the digest of
    the case with `outputs` and `initial` dropped and the solver reduced to its kind and its
    `convergence.measure` (the solver is the one `solver_for` resolves, so a file stating its default
    solver and one leaving it out are the same problem); it is stored in the checkpoint header beside the
    whole-file `case_digest` (provenance, still never compared), and `Checkpoint.read` hands the history
    on only when the two agree. So a restart with a larger `max_steps`, another preconditioner or another
    output directory continues the march; one with another fluid, inlet or residual measure starts a new
    march from the fields. `run.yaml`'s `initial.resumption` holds the three numbers (null when not
    carried). ⚠️ A `dual_time` loop is not part of the digest, so a checkpoint written without one and
    resumed with one carries a shift of `None` (the record's `0.0`), and the control opens at its own start.
  - **Library seam: `resume=` on `solve_flow_march`, `solve_coupled`, `staged_march`, `RootSolver.solve`
    and `solve_coupled_mass_flow`.** `staged_march` applies the reference as the stopping scale, the anchor
    as the **first segment's** damping reference (a later segment follows a refresh and re-bases as in any
    march) and the shift through the control's `resumed_at(shift)` when the control has one — a march with
    no control, or a control without the hook, has no shift to seed and starts afresh. `RootSolver.solve`
    uses the first two (a root solve has no control). Refused beside a `homotopy` (its anchor is the first
    station's). `ShiftStrengthControl.resumed_at` returns `(clamp(shift), None)`: no memo, so a ratio rule
    forms its first ratio a step later (the path `rebase` already takes), and the first step **holds** the
    shift — there is no previous report to adapt it from, the treatment a refresh boundary gets.
    ⚠️ **Still not carried:** a march resumed is a new march for the refresh BUDGET (`refresh.limit` starts
    over) and for a retry's escalated β beyond the shift it ended on.
  - ⚠️ **Nothing produces a `Resumption` for `RootSolver` or `solve_coupled_mass_flow` but a script:** a
    `RootSolver` march exposes no observer, so no checkpoint of one exists to read it from, and a
    case file reaches neither (`CoupledMarch` refuses a bulk velocity; `Segregated` resumes nothing). The seam
    is there so the capability is reachable from every residual, not because a case file uses it.
  - Pinned by `tests/integration/test_flow_march.py` (a resume with the reference reproduces the steps
    that followed it, residual and ratio to 1e-8 and the root to 1e-9, against a control that does not;
    the reference moves the stopping target; the anchor reaches the first segment and the reports say so;
    no shift reaches a march with no control; the homotopy refusal), `test_flow_march_resume.py` (a dual-time
    march resumed with its shift opens at it and continues the ramp, against a control that reopens at the
    start — its own module, because that compile aborted the process when it ran after thirty others in one
    worker, and the suite clears compiled programs only between modules),
    `tests/unit/test_resumption.py`, `test_step_control.py` (`resumed_at`: held, clamped, no memo),
    `test_root_solver.py`, `test_coupled_rans.py` (the anchor reaches only the first segment; the
    mass-flow solve hands its root solve the resumption), `tests/unit/test_case_initial.py` (the quotient
    not the residual; the record's own anchor and shift; an unusable ratio; what does and does not change
    the problem digest) and `tests/integration/test_case_run.py` (a stopped run resumed takes exactly the
    steps the uninterrupted run had left, against the same logged reference; a dual-time run's resumed
    first shift is the stopped run's last; a changed fluid carries none).
  - **Mutation-checked 2026-10-08 (26 single-line mutations across the seam, the control's resume, the
    record, the restart file, the case solvers and the extruded-axis setting): all caught.** The wiring
    mutations that drop `resume` at each hand-off (flow march, coupled solve, mass-flow solve, each case
    solver) and the ones that swap the anchor for the reference, drop the shift, skip the clamp, read a
    recorded zero as a number, or carry the history across another problem each turn at least one test
    red; so do the axis ones (letters misordered, a stated axis ignored by the reader or the writer, the
    dimension refusal disabled, inverted, or left out of `check_against`). No mutation survived and none
    was dismissed.

- **`Fields` — an OpenFOAM time directory as the starting state (built 2026-10-07, #544's second change).**
  `Fields(path, time)`: `path` the OpenFOAM case directory (re-based by `path_fields` like every path),
  `time` a string (a bare number is refused by the settings mapping, which is the "quote it" prompt).
  - **The file names come from the physics**: `Physics.state_fields` (`("U","p")` laminar, plus `k`,
    `omega` under RANS) is the one list `initial_fields` requires and `Fields.read` reads, so the two
    cannot disagree.
  - **Pressure is OpenFOAM's kinematic pressure, both ways, in ONE module** (`kinematic.py`): the reader
    multiplies by `spec.fluid.density`, and `OpenFOAMTime.write` divides by it. The writer used to write the
    solved (real) pressure unscaled into a file whose template dimensions say `[0 2 -2 ...]`, invisible at
    `density: 1` — **a behaviour change for any run at another density that writes an OpenFOAM time**
    (decided with the project owner, 2026-10-02: "Option 1", shipped together with the reader so the two
    stay symmetric). `RunFields.density` carries the factor to the writer (`None` for a radiation case,
    which has no pressure); the other writers still write the pressure itself, and checkpoints hold the
    real pressure.
  - **The mesh must be an `OpenFOAMMesh`** — a generated grid is numbered differently, and the cell count is
    the only cross-check the file allows (a different mesh with as many cells is read without complaint;
    documented on the class). A 2D case drops the extruded axis' component, whose axis is `Fields.extruded_axis`
    (`x`/`y`/`z`, via `case/axes.py`) when the file states it and otherwise recovered from the **case's mesh
    path** (`case_directory / spec.mesh.path`, not the fields' directory, which need not hold the
    polyMesh); `read_openfoam_time` refuses a field with a nonzero component there (the wrong axis, or a
    flow that is not planar) rather than discard it. **`OpenFOAMTime.extruded_axis` is the same setting for
    the writer** (where the padded zero goes), and a stated axis on a three-dimensional mesh is refused by
    `refuse_an_extruded_axis_of_a_three_dimensional_mesh` through `FieldWriter.refuse_for_dimension` /
    `InitialState.refuse_for_dimension`, which `CaseSpec.check_against` runs with the other dimension
    checks. Added 2026-10-08 because the committed 2 x 1 x 1 slab leaves the axis ambiguous and the error
    asked for an argument a file could not give; a real extrusion is unambiguous and needs neither.
  - **No march history is carried** (`resumption` is `None`): another program's solution has no
    reference of ours, so the march begins as a new one from the state.
  - Pinned by `tests/unit/test_case_initial_fields.py` (cells, components, the pressure at density 2, RANS
    fields, each refusal), `tests/unit/test_openfoam_fields.py` (`parse_vector_field`,
    `read_openfoam_time`, a writer round trip) and `tests/integration/test_case_run.py` (a run writes an
    OpenFOAM time at density 2, another starts from it and takes ≤ 2 steps, its checkpoint equal to the
    first run's solution).
  - **Mutation-checked 2026-10-07 (35 single-line mutations across the reference carry, `Fields`, the
    problem digest, the io reader, the solver seams and the pressure factor): all caught, after three
    survived and were fixed.** Each survivor was a test that could not tell the right answer from the wrong:
    the extruded axis taken from the fields' directory rather than the case's mesh path (the fixture put
    both in one directory — now a test with them apart); the stopping-scale test also moved the damping
    anchor, so it passed with the reference ignored in the stopping bar (now a reference large enough
    that the march takes no step, which isolates it); and the `Fields` round trip reached the root anyway
    from a pressure off by the density (now asserts the second run takes no step). Not mutation-checked:
    the section's YAML round trip.

- **Provenance.** `run.yaml` gains `initial:` (a checkpoint's `kind`, `file`, the residual the earlier run
  had reached, its case digest and the `resumption` carried; a `Fields`' `kind`, `file` and
  `density`; `null` when none). `InitialState` declares `path_fields = ("path",)`, so `relocated` re-bases
  it on the output directory with every other path and `case.yaml` reads where it lies (there is no
  initial-specific re-basing code).
- **Tests.** `tests/unit/test_case_initial.py` (digest, header refusals each with a unique match, the
  file round trip under `LogScalars`, the section, the ramp refusal, the run-order guards),
  `test_checkpoint.py` (`find_checkpoint`: highest by step not mtime, `.partial` ignored, prefix),
  `test_case_solver.py` (what each solve is handed), and `tests/integration/test_case_run.py` (a stopped
  run resumed to the same root; a converged checkpoint needs no steps). Mutation-checked 2026-10-02, each
  turning at least one test red: the digest's node binning and its connectivity, the step lookup and its
  `.partial`/prefix rules, the `omega` transform, the physics/count/mesh header refusals, the bare-file
  refusal, the ramp refusal, the three solves' seams, the output-directory guard, clearing after the read,
  the finiteness guard, a run's use of its seed (both integration tests) and the `case.yaml` re-base.
  Not mutation-checked: the section's YAML round trip.

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
- **`validation/pitzdaily_openfoam/compare.py` solves with the file's solver.** `SOLVER =
  _solver_with_overrides(CASE.spec.solver)` applies each set `PITZ_*` march variable as an **edit** of the
  file's solver (unset = the file), and every module constant its probes import (`INNER_STEPS`, `CONTROL`,
  `RETRY`, `STENCIL_REACH`, `LEADING_INVERSE`, ...) is read back from `SOLVER`, keeping its measurement
  record beside it. The default run is `solver.solve(coupled, session_options={observer}, point_setup=<log
  the label>, inner_observer=..., on_checkpoint=..., on_retry=...)`. **Script-only study arms** (project
  owner, 2026-09-24: kept, labelled, off by default) — the rung ladder (`PITZ_RAMP=off`, with
  `BETA_START_WARM` and `SEED_REPAIR`), `PITZ_TURB_TAPER`, `PITZ_TURB_DAMPING_TARGET` and capped Jacobian
  gradient sweeps — run through `_solve_study_arm`, which starts from `SOLVER.settings()` and calls the
  library directly; the banner names the arm in force.
- **`validation/bfs3d_openfoam/case.yaml` states bfs3d at its driver's defaults, and bfs3d now SOLVES with
  its solver section** while still assembling the *problem* by hand (the parity reference). Every march
  constant defaults to the file's value and a `BFS3D_*` variable overrides it; `SOLVER` is reassembled from
  those constants and **the module refuses to import if, with no `BFS3D_*` variable set, it differs from
  `FILE_SOLVER`** — the check that the constants still read the file. The ladder arm (`BFS3D_RAMP=off`)
  is its one script-only study arm.
- **Both step scripts are pinned** by `test_each_step_case_script_runs_its_files_solver_when_no_override_is_set`,
  which imports each in a subprocess with the `PITZ_*`/`BFS3D_*` variables removed.
- **`validation/turbulent_channel/cases/re{20000,45000,240000}.yaml` and
  `validation/turbulent_channel_openfoam/cases/{low,high}.yaml` are what the channel harnesses solve** —
  one file per configuration, each stating ν as the harness used to compute it, to the last bit, and its
  `Segregated` solve (sweeps, relaxation, the direct flow solve, the scalar budget and stop, `ScalarAir` for
  the law-of-the-wall study). The harnesses hold **no** solver setting any more — they call
  `checked.solve(coupled)`. The OpenFOAM harness refuses to compare if the file's ν is not the OpenFOAM
  run's `nu_of / Ubar` (at U_bulk = 1). Both read the channel height, bulk velocity and column stride
  (`nx`) from the file rather than restating them.
- **`tests/unit/test_channel_case_files.py` is their parity check, in the FAST tier** — the meshes are
  generated, so nothing is gitignored. Each file is compared as one pytree against a frozen copy of the
  harnesses' old hand assembly, with one deliberate difference: the harnesses held the viscosity as a
  Python float, a case file holds it as an array (see "One `PropertyModel`" above); a test shows the two
  evaluate to identical values, so the difference is in the pytree only. ~36 s, dominated by the five builds.
- `tests/unit/test_case_file.py::test_the_shipped_pitzdaily_file_reads_as_the_case_its_driver_builds_and_fits_its_mesh`
  still compares the pitzDaily file against a `CaseSpec` written in the test — now a **second** copy rather
  than a third, and it catches the reader misreading the file, not a physics change (a changed file value
  is a deliberate change to the case, and should be reflected there).


## Radiation cases (2026-10-05) — a third physics, decided with the project owner

The radiation package's lamp-and-room studies were scripts writing ad-hoc `.npz` files; the owner
decided against converting those to VTK in a script, and for radiation being a physics of the case file
whose runs write what every run writes. Decisions taken with the owner before building:

- **Patch kinds are `Wall` + a new `Lamp`, shared with flow** (not radiation-only kinds), so a coupled
  flow + UV case later states each patch once. `Wall` gained `reflectance` (black when unset) and
  `geometry`; `Lamp(profile, power, geometry)` is a stationary wall to a flow (`flow_closure()` is
  `Wall()`'s). Its settings carry `flow` / `radiation` scopes beside `turbulence` (see the scope bullet
  above): a flow physics refuses the radiation ones (`_refuse_light`), `Radiation` refuses the flow and
  turbulence ones and requires at least one `Lamp` — the same polymorphic rule as Laminar/RANS, through
  one `_refuse_stray` helper. An `Inlet`/`Outlet` is refused in a radiation case
  (both carry required flow settings).
- **The medium lives inside the physics** (`Radiation.medium: UniformMedium(absorption | transmittance)`),
  not on `fluid`: a radiation case has **no `fluid` and no `numerics` section**. So `CaseSpec.fluid` /
  `numerics` became optional (`kw_only=True` dataclass, every constructor call was by keyword already), and
  **which sections must and must not be present is the physics' knowledge** — `Physics.refuse_sections(spec)`:
  `_Flow` (Laminar, RANS) requires `fluid` and `numerics` and runs the pressure-datum rule (moved out of
  `CaseSpec.__post_init__`); `Radiation` refuses fluid, numerics, drive, sources and pressure datum.
  Revisit the medium's home when a coupled flow + radiation physics exists.
- **Surfaces come from the drawing first** (owner's request): a `Lamp`'s or reflecting `Wall`'s `geometry`
  is `CadSurface` (a STEP solid's whole surface via `CadModel.triangles`), `StlSurface` (named solids of
  an STL), or, **unset, `MeshPatch()`** — the mesh patch's centre-fan triangles, the fallback. A reflecting
  surface normally coarsens (`MeshPatch(coarsen=Coarsen(max_edge, chord))` → `coarsen_to_size`), since the
  transfer is dense `n^2`. **Orientation is DERIVED from the mesh patch** (owner's choice over stating it):
  `_facing` compares each source triangle's normal with the inward normal of the patch's nearest face;
  only `|cos| > 0.5` counts (a crease's nearest face may be across it); all decisive ones agreeing →
  kept, all disagreeing → reversed wholesale, both → refused (a winding defect), none → refused. ⚠️ **A CAD
  solid's WHOLE surface is used**, so a closed solid only fits a patch that is a closed body in the medium
  (a lamp sleeve); a flat wall's slab or a vessel drawn as its fluid has opposite-facing parts and is
  refused. **A file surface given under a patch-GROUP key is refused** in `mesh_misfits` (each member
  would take the whole surface); `MeshPatch` under a group is fine.
- **Occluders** (`physics.occluders`): `CadSolid` / `CadFluid` (exact bodies from `CadModel.solid` /
  `fluid`), `StlBody` and `PatchBody` (`TriangleBody`s). ⚠️ **A black wall (no `reflectance`) shadows nothing
  unless listed** — it is not in the scene at all; exact in a convex room, wrong in an L-shaped one;
  documented on the docs page. A REFLECTING wall shadows the lamps' direct light and the reflected light
  through the scene's own triangles whenever self-occlusion is on (#622, 2026-10-10;
  `.claude/rules/radiation.md` → THE SCENE).
  `PatchBody` builds with `patch_triangles(..., allow_folded=True)` (`mesh.md`): the snapped bunny has
  **291 faces not star-shaped from their vertex mean**, which the first at-scale run refused.
- **Receivers** (`Receivers(cells=True, patches=None)`): unset patches = every `Wall` not in a
  `PatchBody`. A lamp's faces are never gathered on (they lie on the emitting surface); a body's are not
  either (on the body). `mesh_misfits` refuses a named receiver patch that is a lamp or in a body.
- **Numerics inside the physics**: `lamp_refinement` (a `refine_for_receivers` max ratio against the
  gathered points — refused with no points), `lamp_samples` (sub-points per edge for the lamps' light on
  each reflecting facet), and `settings`, **the library's `RadiationSettings` read directly** (registered
  with `NoOcclusion`, `RayCastOcclusion`, `SilhouetteOcclusion`, `ShaftCulling`, `EveryPair`; `ShaftCulling`'s
  bare `tuple` annotations became `tuple[int, ...]` so the mapping can check them). Unset = the library's.
- **Lamp power**: `Lamp.power` in W, or unset for an `IesProfile` (`can_state_power` ClassVar) whose file
  states `[_INTENSITYUNITS]` `W/sr` / `mW/sr` / `uW/sr`: then `photometry.flux * scale`. **The unit
  reading is the case layer's, deliberately** — `photometry.py`'s docstring says converting units is the
  caller's decision. A candela file with no power is refused at build. Exitance by `lamp_exitance`
  (each lamp one body named by its patch, its own profile). **`Lamp.reflectance`** (unset = 0, the same
  `[0, 1]` check as `Wall.reflectance` through `boundaries._refuse_a_bad_reflectance`) becomes the lamp
  facets' `diffuse_reflectance`; a lamp now reflects and shadows in the scene's exchange (#604 step 1).
- **What is built is the library's `radiation.Scene`** (`radiation/scene.py`, see `radiation.md`): lamps
  whose emission is kept out of the transfer but whose facets exchange reflected light, reflectors, bodies, medium, `VolumeReceivers` (cell centroids + volumes),
  `SurfaceReceivers` per patch (face centroids, `-normal`, areas, reflectance, `reflector` = the patch's
  name when it reflects). Solved by `RadiationSolve(rtol)` → `solve_scene`, observer `report` = the log's
  `note`. **`Physics.build` now takes `directory`** (files are relative to the case file);
  `CheckedCase` carries `directory`.
- **Outputs (agreed with the owner; the `aquaflux_viz` viewer reads them):** cell fields in `fields.vtu`
  (`G`, plus `G_direct`/`G_reflected` when anything reflects); **`PatchVtk` → `patches.vtm` indexing
  `patches/<patch>.vtp`, one block per boundary patch named by it, EVERY boundary patch written**, face
  fields `E`, `E_absorbed` = (1−ρ)E (read from `SceneSolution.irradiance_absorbed` since #622), plus `E_direct`/`E_reflected`. Writers now take `RunFields(cells,
  patches)`; `Physics.output_patch_fields` and `Physics.results` default to empty for flow. `run.yaml`
  gains `results:` (lamp power and facets, reflector facets, radiosity cycles, `volume_integral_G`,
  `medium_absorbed_power`, `lamp_absorbed_power` when anything reflects, per patch
  `area`/`incident_power`/`absorbed_power`, `unaccounted_power` -- which subtracts the lamps' share).
- **Files**: `CaseFile.check` refuses a missing file named under `physics` or `boundaries` (inputs; the
  mesh is checked by reading it, an `OpenFOAMTime` target is an output) before reading the mesh;
  `_write_case_record` re-bases every `path_fields` entry through `with_paths` — this replaced its two
  ad-hoc branches for `OpenFOAMMesh.path` and `OpenFOAMTime.case`.
- **Tests**: `tests/unit/test_case_radiation.py` (build parity with a hand-built `Scene` from library
  calls — every array leaf; defaults reaching the library; IES power from the file; STL orientation
  flipped and refused; paths found/checked/re-based; round trip; every refusal; a group key with a file
  surface; a small run end to end), `test_radiation_scene.py`, `test_vtk_patches.py`.
- **The case files**: `validation/ray_effects_room/cases/*.yaml` (see `validation.md`), run by
  `run_cases.py` through `run_case.sh`.
