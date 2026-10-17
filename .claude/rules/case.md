---
paths:
  - "aquaflux/case/**"
  - "aquaflux/__main__.py"
  - "validation/*/case.yaml"
  - "validation/*/cases/*.yaml"
---

# Rules — `aquaflux/case/` (a whole case described in one YAML file)

> **Provenance boundary (binding).** What you read here informs your understanding; none of it may reach
> the shipped surface (root `CLAUDE.md`, **Comment Convention**). Issue numbers are for you, never for a
> docstring or error message. The dated record — measurements, test inventories, mutation passes — is in
> `.claude/notes/case-log.md`.

Tracking issue #437. Held to #374 (a loaded case yields a builder, never a built solver) and #375 (no flat
case object — a small core plus discriminators).

## ⚠️ The case file is the user's whole interface (binding — project owner, 2026-09-24)

**Users do not write scripts. Everything a normal user would set belongs in the file, and a case runs from
its file alone.**
- "Keep it in the script" is never the answer for a user setting — solver settings, initial guesses
  (`BulkVelocity.initial_force`), sources, outputs all go in the file.
- **One file per configuration; a sweep is a set of files** (the channel studies: one per Reynolds number).
- **The validation `compare.py` scripts are developer harnesses** that read case files. Where a reference
  fixes a value (the OpenFOAM channel's ν), the file states it and the harness **checks** it.
- Do not call the harnesses "drivers" to the project owner — it collides with the *drive* (`MassFlow`).

## Construction stages (all built)

- **Read and check, no geometry:** `read_case(path)` → `CaseFile(spec, directory)`; `CaseFile.check()` →
  `CheckedCase(spec, mesh, directory)` — the mesh read and `validate()`d (even when the reader already
  validated, so the check does not depend on the reader) and the spec checked against its topology.
- **Build:** `CheckedCase.build()` computes geometry once and calls `spec.physics.build(spec, mesh,
  geometry, directory)`, which returns **the problem the library solves already take** —
  `MomentumContinuity` (Laminar), `CoupledRANS` (RANS), a `radiation.Scene` (Radiation) — never a
  case-specific wrapper and never a built step (the step is rebuilt at every Reynolds rung and refresh).
- **Solve:** `CheckedCase.solve(problem, **observers)` runs `solver_for(spec)` (the file's `solver`, or the
  physics' default with every setting unset) from the file's `initial` state if any. The solver section
  turns into the library solve's keyword arguments, never a built step.

## Layout

| module | holds |
|---|---|
| `scopes.py` | `FLOW` / `TURBULENCE` / `RADIATION`, `SCOPES`, the `Scoped` mixin (`setting_scopes`, `required_in_scope`, `settings_in`, `missing_in`) |
| `spec.py` | `CaseSpec` (mesh, physics, boundaries, optional fluid and numerics, drive, pressure datum, sources, solver, initial, outputs), `Numerics`, the one `SettingsMapping` registry `_CASE_MAPPING`, `case_spec_from_mapping` / `_to_mapping`, `CaseSpec.check_against(mesh)` |
| `case_file.py` | the YAML parse (`_CaseLoader`), `read_case` / `read_case_document` / `write_case`, `CaseFile`, `CheckedCase` |
| `mesh_source.py` | `MeshSource.read(directory) -> Mesh` → `OpenFOAMMesh` / `StructuredGrid` + `AxisGrading` → `GeometricGrading` |
| `forcing.py` | `DriveSpec` → `BulkVelocity` (builds `flow.MassFlow`); `SourceSpec` → `BodyForce` (builds `flow.UniformBodyForce`) |
| `fluid.py` | `Fluid` |
| `physics.py` | `Physics` → `_Flow` → `Laminar` / `RANS`, and `Radiation`; flow physics map a march state to named fields (`restart_fields`) and back (`initial_fields`) |
| `initial.py` | `InitialState` → `Checkpoint` / `Fields`: `read(case_directory, spec, mesh)` → `StartingFields`, `starting_arguments` |
| `axes.py` | `AXES`, `AxisName`, and the one refusal of an extruded axis stated for a 3D mesh |
| `kinematic.py` | `kinematic_pressure` / `pressure_from_kinematic` — the one home of OpenFOAM's pressure-per-unit-density factor |
| `restart_file.py` | `RestartHeader`, `mesh_digest`, `checkpoint_writer`, `read_restart` → `RestartFile` (`resumption`), `refuse_non_finite` |
| `boundaries.py` | `PatchCondition` → `Inlet` / `Outlet` / `Wall` / `Lamp`; `InletTurbulence` → `FixedTurbulence` / `IntensityLength` |
| `radiation.py` | `LampProfile` → `LambertianProfile` / `CosinePowerProfile` / `IesProfile`; `UniformMedium`; `SurfaceSource` → `MeshPatch` / `StlSurface` / `CadSurface` (+ `Coarsen`, `CadPlacement`); `OccluderSpec` → `PatchBody` / `StlBody` / `CadSolid` / `CadFluid`; `Receivers` |
| `paths.py` | `named_paths` / `with_paths` / `relocated` (see "Paths") |
| `solver.py` | `SolverSpec` → `CoupledMarch` / `FlowMarch` (on `_March`) / `Segregated` / `RadiationSolve`; `ViscosityRamp`; `RootSolve`; `solver_for`; `NotConverged` |
| `outputs.py` | `Outputs`, `RunFields(cells, patches)`, `FieldWriter` → `Vtk` / `OpenFOAMTime` / `PatchVtk`, `Checkpoints` |
| `run.py` | `prepare_run(path, overwrite)` → `PreparedRun.run(terminal)` → `RunRecord`; `plan_run` → `RunPlan` |
| `aquaflux/__main__.py` | the `aquaflux` command |

- **`case/` is the top layer: nothing in `aquaflux` imports it except `aquaflux/__main__.py`** (project
  owner, 2026-09-25; `ABOVE_THE_CASE_LAYER`, pinned by
  `tests/unit/test_layering.py::test_nothing_below_the_case_layer_imports_it`). Only `relocated` is reached
  across that boundary, so it alone is exported for it.
- **The case vocabulary lives here, not beside each runtime class**: a patch kind is one statement for every
  field, so no single physics package owns it, and a shared `Physics` base below `flow/` and `turbulence/`
  would make a cycle through `case/__init__`. **The library's own classes are read directly** where a file
  can state them (schemes, SST constants, variable transforms, solve settings); a case value exists only
  where no library class expresses what a file needs.
- The top level names no `kind`; `fluid`, `numerics` and `outputs` may omit theirs (one form each), injected
  by `case_spec_from_mapping`, stripped by `_to_mapping`. Nested values may not.

## Binding decisions

- **Boundaries are one physical kind per patch** (project owner, 2026-09-24), not per-field closure dicts, so
  momentum, `k`, `omega` and the wall set are derived from one statement and cannot disagree. A per-field
  choice becomes a field on the kind (`Wall.k: zero_gradient | zero`); an unusual combination needs a new
  kind. **A moving wall is `Wall` with an optional `velocity`**, not its own kind.
- **Which physics reads a setting is declared once, as a scope** (`scopes.py`). Values list
  `setting_scopes` and `required_in_scope`; each physics lists `reads_scopes` (Laminar `flow`; RANS
  `flow`+`turbulence`; Radiation `radiation`). A physics refuses a stated setting of a scope it does not read
  and a missing required one, listing every offending path at once (`_refuse_stray`,
  `_Flow.refuse_sections`). The schema publishes `scope` / `required_in_scope` / `reads` for the browser form
  (`ui.md`). No `isinstance(physics, RANS)` in `CaseSpec`.
- **Which sections must be present is the physics' knowledge** (`Physics.refuse_sections(spec)`): `_Flow`
  requires `fluid` and `numerics` and runs the pressure-datum rule; `Radiation` refuses fluid, numerics,
  drive, sources, pressure datum and `initial`.
- **The fluid is stated once, with exactly one viscosity** (project owner, 2026-09-24): `kinematic_viscosity`
  or `dynamic_viscosity`, the other from `density`. `CoupledRANS.build` still cross-checks the two
  assemblers' property models (`refuse_a_fluid_the_flow_disagrees_with`, #367).
- **The pressure level is fixed exactly once** (#500): a top-level `pressure_datum: {kind: PinnedPoint,
  point, value}` is required when no patch fixes the pressure and refused beside one — decided by the flow's
  own `refuse_an_unsuitable_pressure_datum` over the patches' `flow_closure()`s. `check_against` refuses a
  point of the wrong dimension or outside the mesh's bounding box.
- **Inlet turbulence is `FixedTurbulence(k, omega)` or `IntensityLength(intensity, length)`**: `k = 1.5 (I
  |U|)^2`, `omega = sqrt(k) / (beta_star^(1/4) L)`. **`C_mu` is the case's own `SSTModel.beta_star`**
  (project owner, 2026-09-25), so `turbulence_closures(model)` / `InletTurbulence.inflow(velocity, model)`
  take the model and `RANS.build` hands every patch the model it builds the closure with. Zero inflow speed
  is refused. The step cases stay `FixedTurbulence`: their OpenFOAM ω values are rounded, so switching would
  change the problem.
- **`Numerics.gradient`** registers `CompactGreenGauss`, `CorrectedGreenGauss`,
  `MultipleCorrectionGradient`, `ProjectedStencilGradient` (the tetrahedral-mesh scheme). `HessianCorrectedGradient`
  is unregistered (dominated on cost, `schemes.md`).
- **Every advection scheme is in `numerics`** (project owner, 2026-10-06): `momentum_advection` (required —
  `advection_scheme=None` means Stokes flow, a different problem, so it must not be reachable by omission) and
  `turbulence_advection` (scope `turbulence`). There is no `physics.advection`.
- **A `boundaries` key names a patch or a patch group** (project owner, 2026-09-25, #364).
  `CaseSpec.boundaries` keeps the keys as written (round trip); `CaseSpec.patch_conditions(mesh)` resolves
  them through `FacePatches.addressed_by`, and **every build reads the per-patch form, never
  `spec.boundaries`** (`BoundaryConditions.resolve` knows only patches). `_patch_conditions` (shared by the
  check and the build) refuses a patch reached twice, a name that is both a patch and a group of other
  patches, and a group member that is not a boundary patch.
- **`check_against` needs topology only and reports every misfit at once**: unknown patch names (listing the
  mesh's patches and groups), `interior`, a patch reached twice, uncovered boundary faces
  (`FacePatches.uncovered_boundary_faces`, the query `BoundaryConditions.resolve` refuses on), and every
  `refuse_for_dimension` (patches, drive, sources, writers, initial).

## ⚠️ The YAML parse is YAML 1.2 — do not swap in `yaml.safe_load`

PyYAML's 1.1 rules read `1e-5` as a string, `no`/`off`/`yes`/`on` as booleans and `010` as octal, and keep
the last of two duplicate keys. `_CaseLoader` uses the 1.2 core resolvers, reads integers as decimal and
refuses duplicate keys (each pinned). `write_case` uses `safe_dump`, which the loader reads back equal.
`read_case_document` is that same parse, shared by `read_case` and `aquaflux show`.

`SettingsMapping` (`solve/settings_mapping.py`, `solve.md`) supports tables (`Mapping[str, X]`), refuses a
missing required field by path, and re-raises a nested constructor's refusal with the path prepended.

## How the build derives what the drivers used to restate (binding)

- **Each patch kind builds its own closures**: `flow_closure()` (`Inlet` → `VelocityInlet`, `Outlet` →
  `PressureOutlet`, `Wall` → `NoSlipWall` or `MovingWall`, `Lamp` → as `Wall()`) and
  `turbulence_closures(model)` → `(k, omega)` (inlet: both `Dirichlet` from the inflow; outlet: both
  `ZeroGradient`; wall: `k` by `Wall.k`, and a placeholder `ZeroGradient` ω the closure fixes in the wall
  cells).
- **The wall set is `flow.sheared_patches(momentum.boundary)`**, so the flow/turbulence wall reconciliation
  at `CoupledRANS.build` holds by construction.
- **One `PropertyModel`** (`Fluid.property_model()`), built in the momentum builder and handed to
  `SSTTurbulence.build` as `momentum.properties` — the same object.
  ⚠️ **Its viscosity is `Constant(jnp.asarray(mu))`, an array leaf, deliberately**: a Python float is static
  to a jitted function, so a continuation rescaling viscosity would recompile at every rung. `mu = rho * nu`
  in that order, bit-identical to the hand-built drivers.
- **An unset setting is not passed** (`_set`), so the builder's default applies; nothing restates a default.
- **`_momentum(spec, mesh, geometry)` is the one flow builder** both flow physics start from.

## Checking a build

- `validation/case_file_parity.py` builds each step case two ways and requires one pytree (static fields
  included, array leaves bit-equal) and a bit-identical residual at the hybrid initial condition. References:
  pitzDaily — a frozen copy of the old hand assembly; bfs3d — its driver's own `build_case()`, which stays
  hand-built for this reason. bfs3d is kept out of CI by design (project owner: run too long, generated mesh
  gitignored under `runs/`), so the harness is a local check.
- In the fast tier: `tests/unit/test_case_file.py` (2D slab fixture), `tests/unit/test_channel_case_files.py`
  (the five channel files against a frozen copy of the old assembly; the one deliberate difference is the
  array-leaf viscosity).
- ⚠️ **A pytree comparison must require an array leaf to be matched by an array leaf** — comparing through
  `np.asarray` let `Constant(4e-3)` equal `Constant(jnp.asarray(4e-3))`, hiding the per-rung recompile.

## Periodic channels: mesh source, drive, sources

- **`MeshSource.read(directory) -> Mesh` is the contract** (a generated grid has nothing to read).
- **`StructuredGrid(cells, lengths, periodic, grading)`** builds `structured_grid_2d(named_boundaries=True)`
  (side-named patches minus a periodic axis's two sides); `periodic` supports `x` only; a whole-number float
  cell count is normalized to an int. A 3D `StructuredGrid` is uniform and non-periodic
  (`structured_grid_3d` has neither).
- **The drive is `DriveSpec` → `BulkVelocity(target, direction, initial_force)`**; unset is
  boundary-driven (there is no `BoundaryDriven` kind). `BulkVelocity` exists because `MassFlow.force` is an
  array leaf the mapping cannot check; `initial_force` is the multiplier's seed, documented as a guess.
- **`sources: [BodyForce(force)]`** → `UniformBodyForce`, named differently so the case and flow vocabularies
  do not share a name.

## The solver section (binding)

- **Kinds, each the library solve of the same shape**: `CoupledMarch` → `solve_coupled`, or
  `solve_reynolds_ramp` with `continuation`; `FlowMarch` → `solve_flow_march`; `Segregated` →
  `solve_segregated` (`bulk_velocity_flow_solve` under a `MassFlow` drive, else `reused_flow_solve`; scalar
  `scalar_pseudo_transient_solve`; start `sst_initial_fields`); `RadiationSolve(rtol)` → `solve_scene`. Each
  refuses the physics it does not solve (`refuse_for(physics, drive)`, in `CaseSpec.__post_init__`). All flow
  kinds hold a `BulkVelocity` drive; a march refuses one only beside a `MaterializedJacobian` preconditioner
  (`_March._refuse_a_border_the_preconditioner_lacks`, #541).
- **The shared march settings are one dataclass, `_March`**: `max_steps`, `convergence`, `preconditioner`,
  `dual_time`, `linear_solve`, `step_control`, `retry`, `adaptive_tolerance` (unset = the library's
  `DEFAULT_ADAPTIVE_TOLERANCE`; off = both rules `null`). `CoupledMarch` adds `turbulence_damping` (→
  `CoupledShiftSettings`), `omega_step_ratio` (unset = e under log-ω, infinite = unbounded) and
  `continuation`. A setting added to `_March` reaches both marches.
- **The library's settings values are read directly** — `Convergence` (+ `RowScaled`/`BlockScaled`/
  `Euclidean`), `DualTimeLoop`, `LinearSolveSettings`, the dual-time controls, `RetryPolicy`,
  `AdaptiveLinearTolerance`, and every coupled-preconditioner kind via
  `turbulence.PRECONDITIONER_SPEC_MAPPING.kinds`. Case-side values only where none fits:
  - `ViscosityRamp` — `scale: flow|both` names `scale_momentum_only` / `scale_both_blocks`; `end: settled`
    (+ optional `finish`) names the march's own `step_control.settled`. `ViscosityRamp.schedule(step_control)`
    is the one place it becomes a `solve.RampSchedule` (the pitzDaily study path calls it too). `end` without
    a `step_control` is refused at read.
  - `RootSolve` — `RootSolveSettings` holds `lineax` solvers;
    `test_a_root_solve_states_every_setting_of_the_librarys_root_solve` pins the field lists equal.
- **Unset means the library's default, never one restated here.** So a pitzDaily file with no solver
  section runs a single-step `CoupledMarch()` that does not converge in `max_steps`; changing the shipped
  default needs the project owner (#542), and the validation files state their calibrated values.
- **A script can observe a solve, never configure it.** `solve(problem, **observers)` refuses any keyword
  that is a solve setting, set in the file or not (`_owned()`). A materialized-Jacobian preconditioner is
  opened as a session here (`open_session(spec, problem, **session_options)`), so a harness passes the
  session's observers, not a session; `session_options` beside `BlockDiagonal` is refused. A `point_setup`
  may observe a ramp's anchor and is refused if it returns settings. `Segregated` takes no observers and only
  warns when it runs out of sweeps (#543).
- ⚠️ **`station_step` and `jacobian_gradient_sweeps` are not file settings and not in `_owned()`**; the
  pitzDaily harness's study path passes them to the library itself. Passing them as "observers" would be a
  configuration back door.
- **Parity is on the keywords**: `tests/unit/test_case_solver.py` replaces each library solve with a
  recorder and compares against a frozen copy of what each harness used to pass;
  `tests/integration/test_case_solve.py` runs one for real.

## Running a case (binding)

- **`aquaflux check|run case.yaml [--overwrite]`** (also `python -m aquaflux`). Exit `0` converged/checked,
  `1` a solve that stopped short, `2` a refused file (reason on stderr). Only read/check errors map to `2` —
  `prepare_run` does them all before `run()`, so an error inside a solve is a traceback.
- **`prepare_run` is the cheap stop**: read, refuse an occupied output directory or existing
  `OpenFOAMTime` target, resolve the solver, check the mesh, read the `initial` state — all before geometry.
  **What counts as occupied has one home, `run._plan`**, shared with `plan_run(path) -> RunPlan(directory,
  log, history, occupied)` (`aquaflux plan`), which the browser asks before replacing results (project
  owner: "only overwrite with permission"). `--overwrite` replaces the files the run writes and clears
  `checkpoints/` **after** the starting state has been read.
- **Editing commands for the browser interface**: `aquaflux schema` (`case_schema()`), `aquaflux show
  case.yaml` (`{"error", "case"}`; a refused file is still shown), `aquaflux write case.yaml [--relative-to
  DIR]` (validates through `case_spec_from_mapping`; writes nothing when refused; drops comments and
  default-valued settings), `aquaflux mesh OUT` (the mesh section alone, via `mesh_source_from_mapping`;
  writes `OUT/mesh.vtu`, prints cells, dim, patches with faces, groups). Pinned by
  `tests/unit/test_case_editing_commands.py`.
- **`aquaflux serve`** answers `SERVED` commands (`schema`/`show`/`write`/`mesh`/`check`/`plan`, never `run`)
  one JSON line per request by calling `main(arguments)` in-process with stdio swapped; an exception or
  `SystemExit` becomes that request's status. ⚠️ **Replies go on a `dup` of fd 1, and fd 1 is then pointed at
  stderr**, so stray stdout writes cannot corrupt the protocol. It exists because a command's work is
  0.01–0.4 s against 2.5–5 s of process start-up (2026-10-06, macOS arm64, loaded machine; breakdown in the
  log).
- **`outputs` is optional; its default writes `results/fields.vtu`, `results/march.log`,
  `results/history.csv`** (project owner). `history` is `solve.StepHistory` (`solve-march.md`); `log` and
  `history` may not name the same file. `SolverSpec.observers_for(logger, recorder)` takes a
  `solve.MarchRecorder` and reads its hooks as attributes; the runner's `_StepCount` forwards to recorders.
  `on_residuals` is wired only when a recorder takes it (each costs a residual evaluation per step).
- **Writers are the library's**: `Vtk` → `write_vtu`; `OpenFOAMTime` → `write_openfoam_time` (into the named
  OpenFOAM case; only for an `OpenFOAMMesh`; `case` required because the mesh path need not be a
  `constant/polyMesh` layout; `time` a quoted string; `fields` selects by name); `PatchVtk` → `write_patches`.
  What is written is `output_fields(problem, solution)` (`U`, the solved `p`, and `k`, `omega`, `nut` under
  RANS) and, for radiation, `output_patch_fields`.
- **A solve that stops short is a record, not an error**: `NotConverged` (from `Segregated`'s warning,
  matched by `_SEGREGATED_NOT_CONVERGED` — a test asserts its words are in `solve_segregated`'s source),
  `EquinoxRuntimeError` from a march, and `KeyboardInterrupt` (Ctrl-C or the browser's Stop, SIGINT to the
  process group) all end with `converged: false`, the log, checkpoints and records written, exit 1.
- **Provenance**: the output's `case.yaml` is the spec as it ran (`solver_for(spec)` written out, every path
  re-based by `relocated`, `outputs.directory` `"."`); `run.yaml` records case path, version, commit and
  dirty flag, times, solver kind, convergence, steps, last residual, message, what was written, `initial`,
  and for radiation `results`.
- ⚠️ The two-cell slab fixture leaves the default `MultipleCorrectionGradient` underdetermined — tests on it
  state `CompactGreenGauss`.

### Paths

Every value naming a file declares `path_fields` (a `ClassVar`), found at any depth by `with_paths`.
**`relocated(value, source, target)` is the one re-basing rule** (relative re-based, absolute kept,
`outputs.directory` never — it names no file), used by the run record and `aquaflux write --relative-to`.
There is no per-class `relocated`/`rebased_path`. `CaseFile.check` refuses a missing input file under
`physics` or `boundaries` before reading the mesh (an `initial` path is checked when it is read).

## A starting state (#544) — binding decisions

- **One optional `initial` section**: `Checkpoint {path, step: latest | <int>}` (an earlier run's
  `checkpoints` directory) or `Fields {path, time, extruded_axis?}` (an OpenFOAM time directory). Unset is a
  self-start. `Checkpoint` reads, `Checkpoints` (the output) writes.
- **A case checkpoint holds physical fields plus a header** (project owner, 2026-10-02), not the solved
  state — a `log(omega)` state read by a direct-ω case passes every size check and loads as garbage.
  `restart_fields` / `initial_fields` map both ways; `checkpoint_writer` is injected as the `save=` of the
  run's `StateCheckpointer`, so `solve/` stays mesh-free. A library `StateCheckpointer` with its default
  serializer (the step-case harnesses) still writes a bare `state`, which `read_restart` refuses.
- **The header**: physics kind, `n_cells`, `dim`, `mesh_digest` (face-to-cell connectivity plus node
  coordinates binned to 1e-6 of the largest extent, so another machine's rounding agrees and a renumbered
  mesh does not), the whole-file `case_digest` (provenance, never compared) and the `problem_digest`
  (compared — below). A file without a problem digest is refused (no shim).
- **`read` and `starting_arguments` are split**: `InitialState.read` checks against mesh and case without
  geometry (so `prepare_run` refuses a misfit before the build); `starting_arguments` maps onto the built
  problem as `initial=` / `resume=`. `CheckedCase.solve` honours the file's `initial`.
- **Every solve takes `initial=` and `resume=`** (passed only when set). `Segregated` refuses `resume`;
  `RadiationSolve` refuses both. Neither is in `_owned()`.
- **A `CoupledMarch` with `continuation` refuses `initial`** (project owner, 2026-10-02) at read
  (`SolverSpec.refuse_initial`) and at `solve`: the ramp seeds at the anchor's viscosity. Resuming mid-ramp
  would need the checkpoint to record its station — not built.
- **A restart may not read from its own output directory** (`_refuse_a_start_from_the_output`, whether or
  not `--overwrite` is set), and checkpoints are cleared only after the state is read.
- **A checkpoint with non-finite fields is refused** (`RestartFile.refuse_if_not_finite`) — the checkpointer
  also writes the failed step, so the newest file can be the one the march died in.
- **A restart carries the stopped march's history as `solve.Resumption(reference_residual,
  damping_reference, shift)`**, built by `RestartFile.resumption` from the last step record: the reference is
  `residual_norm / residual_ratio` (the stopping scale), the anchor is the record's `damping_reference`, the
  shift its last `shift` (`0.0` read as none). **Carried only when the problem digest matches** — the case
  with `outputs` and `initial` dropped and the solver reduced to its kind and `convergence.measure` — so a
  larger `max_steps` or another preconditioner continues the march; another fluid, inlet or measure starts a
  new one. ⚠️ `dual_time` is not in the digest: a checkpoint written without one carries no shift.
  - **Library seam**: `resume=` on `solve_flow_march`, `solve_coupled`, `staged_march`, `RootSolver.solve`.
    `staged_march` applies the reference as the stopping scale, the anchor to the first segment only, and the
    shift through the control's `resumed_at(shift)` (`ShiftStrengthControl`: clamped, no memo, held for the
    first step). Refused beside a `homotopy`.
  - ⚠️ The default flow march's switched-evolution damping is memoryless (it re-anchors on the state it is
    handed); only a dual-time control carries a shift. ⚠️ Still not carried: the refresh budget and a retry's
    escalated β. A held bulk velocity's force is re-solved from the drive's seed.
- **`Fields`**: the file names come from `Physics.state_fields` (`U`, `p`, plus `k`, `omega` under RANS).
  **Pressure is kinematic both ways, in `kinematic.py`**: read × density, `OpenFOAMTime` writes ÷ density
  (project owner, 2026-10-02; `RunFields.density` carries the factor, other writers and checkpoints hold the
  real pressure). The mesh must be an `OpenFOAMMesh`, and only the cell count is cross-checked. On a 2D case
  the extruded axis is `Fields.extruded_axis` or, unset, recovered from the **case's mesh path** (not the
  fields' directory); `OpenFOAMTime.extruded_axis` is the writer's twin. No march history is carried.

## The case files in the repository

- **`validation/pitzdaily_openfoam/case.yaml` is what `compare.py` solves.** `compare.case_spec(model=,
  gradient_scheme=)` applies study overrides as edits of the spec (`dataclasses.replace`); `build_case()`
  returns `coupled`, `momentum`, `turbulence`, `geom`, `spec`. `PITZ_GRADIENT` / `PITZ_K_WALL` unset mean the
  file's choice. `SOLVER = _solver_with_overrides(CASE.spec.solver)` applies each set `PITZ_*` march variable
  as an edit; probe-imported constants are read back from `SOLVER`. Script-only study arms (project owner,
  2026-09-24: kept, labelled, off by default — the rung ladder `PITZ_RAMP=off`, `PITZ_TURB_TAPER`,
  `PITZ_TURB_DAMPING_TARGET`, capped Jacobian gradient sweeps) run through `_solve_study_arm`.
- **`validation/bfs3d_openfoam/case.yaml` states bfs3d at its driver's defaults; bfs3d solves with its
  solver section** but assembles the problem by hand (the parity reference). `BFS3D_*` variables override
  constants that default to the file; the module refuses to import if, with none set, `SOLVER` differs from
  `FILE_SOLVER`. `BFS3D_RAMP=off` is its one script-only arm.
- Both pinned by `test_each_step_case_script_runs_its_files_solver_when_no_override_is_set`.
- **`validation/turbulent_channel/cases/re{20000,45000,240000}.yaml` and
  `validation/turbulent_channel_openfoam/cases/{low,high}.yaml`** each state ν to the last bit and a
  `Segregated` solve; the harnesses call `checked.solve(coupled)` and read height, bulk velocity and `nx` from
  the file. The OpenFOAM harness refuses if the file's ν is not `nu_of / Ubar`.
- `validation/ray_effects_room/cases/*.yaml` (`validation.md`).

## Radiation cases

Radiation is a physics of the case file (project owner, 2026-10-05), and its runs write what every run writes.

- **Patch kinds are `Wall` + `Lamp`, shared with flow**, so a coupled flow + UV case states each patch once.
  `Wall` has `reflectance` (black when unset) and `geometry`; `Lamp(profile, power, geometry, reflectance)`
  is a stationary wall to a flow. Radiation refuses `Inlet`/`Outlet` and requires at least one `Lamp`.
- **The medium lives in the physics** (`Radiation.medium: UniformMedium(absorption | transmittance)`), so a
  radiation case has no `fluid` and no `numerics`. Revisit when a coupled flow + radiation physics exists.
- **Surfaces come from the drawing first**: `CadSurface` (a STEP solid's whole surface), `StlSurface`, or,
  unset, `MeshPatch()` (centre-fan triangles; `MeshPatch(coarsen=Coarsen(max_edge, chord))` for reflectors,
  since the transfer is dense). **Orientation is derived from the mesh patch** (project owner's choice):
  `_facing` compares each triangle's normal with the nearest patch face's inward normal where `|cos| > 0.5`;
  all agree → kept, all disagree → reversed, mixed or none → refused. ⚠️ A CAD solid's whole surface is used,
  so it fits only a patch that is a closed body in the medium. A file surface under a patch-group key is
  refused.
- **Occluders** (`physics.occluders`): `CadSolid` / `CadFluid` (exact bodies), `StlBody`, `PatchBody`
  (`TriangleBody`; built with `allow_folded=True`, since real patches have faces not star-shaped from their
  vertex mean). ⚠️ **A black wall shadows nothing unless listed** — exact in a convex room, wrong in an
  L-shaped one. A reflecting wall shadows through the scene's own triangles (`radiation.md`, THE SCENE).
- **Receivers** (`Receivers(cells=True, patches=None)`): unset patches = every `Wall` not in a `PatchBody`;
  never a lamp's or a body's faces.
- **Numerics inside the physics**: `lamp_refinement`, `lamp_samples`, and `settings` — the library's
  `RadiationSettings` read directly (occlusion and culling kinds registered). Unset = the library's.
- **Lamp power**: `Lamp.power` in W, or unset for an `IesProfile` whose `[_INTENSITYUNITS]` is `W/sr` /
  `mW/sr` / `uW/sr` (`can_state_power`). The unit reading is the case layer's on purpose; a candela file
  with no power is refused. `Lamp.reflectance` becomes the lamp facets' `diffuse_reflectance`.
- **What is built is `radiation.Scene`** (`radiation.md`), solved by `RadiationSolve(rtol)` → `solve_scene`.
- **Outputs** (agreed with the project owner; the `aquaflux_ui` viewer reads them): `fields.vtu` (`G`, plus
  `G_direct`/`G_reflected` when anything reflects); `PatchVtk` → `patches.vtm` + `patches/<patch>.vtp` for
  every boundary patch (`E`, `E_absorbed`, `E_direct`/`E_reflected`). `run.yaml`'s `results:` holds lamp
  power and facets, reflector facets, radiosity cycles, `volume_integral_G`, `medium_absorbed_power`,
  `lamp_absorbed_power`, per-patch area / incident / absorbed power, and `unaccounted_power`.
- Tests: `tests/unit/test_case_radiation.py` (build parity with a hand-built `Scene`, every refusal, a run
  end to end).

## What a file cannot describe yet

- A starting state on a non-OpenFOAM mesh (`Fields` is in OpenFOAM's cell numbering).
- A held bulk velocity beside a `MaterializedJacobian` preconditioner (#541).
- A graded or periodic 3D structured grid.
- Source kinds beyond `BodyForce` (waits on #362).
- A passive-scalar case referring to another case's converged flow (#662; `bfs3d_species` imports the flow
  harness by path).
- Profiles (`DirichletField`, a callable).
