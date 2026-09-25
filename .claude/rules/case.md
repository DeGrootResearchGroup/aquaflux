---
paths:
  - "aquaflux/case/**"
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
- **The solver section (BUILT 2026-09-24), then a `run` entry point plus outputs, are the critical path**:
  until the second exists a user still has to write the three lines that read, build and solve a file,
  and to write the results out. Every validation harness now takes its solve from its file.
- Do not call the harnesses "drivers" to the project owner — it collides with the *drive* (`MassFlow`).

## Status — phases A–D and the solver section BUILT (2026-09-24); a `run` entry point and outputs are NOT

The construction order #374 records is A spec → B topology → C geometry → D equations → E state →
F frozen solver → G drive.
- **A–B, the cheap "check the file" stop:** `read_case(path)` → `CaseFile(spec, directory)`, then
  `CaseFile.check()` → `CheckedCase(spec, mesh)` (the mesh read and `validate()`d, the spec checked
  against its topology, **no geometry**). ~1 s on pitzDaily (12225 cells, ASCII read; one run,
  2026-09-24, macOS arm64).
- **C–D:** `CheckedCase.build()` computes the geometry once and hands it to `spec.physics.build(spec,
  mesh, geometry)`, which returns **the problem the initializers and solves already take** —
  `MomentumContinuity` for `Laminar`, `CoupledRANS` for `RANS` — never a case-specific wrapper (#375) and
  never a built step (#374: the step is rebuilt from mid-march states at every Reynolds rung and refresh).
  ~8 s on pitzDaily including the wall distance (same run).
- **E–F:** `CheckedCase.solve(problem, **observers)` runs `solver_for(spec)` — the file's `solver`, or the
  physics' march with every setting unset. #374's "`state -> step` builder" needed no new builder: the
  library solves already rebuild the step from **specs** at every rung and refresh, so the solver section
  turns into the solve's keyword arguments and never into a built step. See "The solver section" below.

## The layout, and why each part is where it is

| module | holds |
|---|---|
| `spec.py` | `CaseSpec` (mesh, fluid, physics, boundaries, numerics, drive, pressure datum, sources, solver), `Numerics`, the one `SettingsMapping` registry `_CASE_MAPPING`, `case_spec_from_mapping` / `_to_mapping`, `CaseSpec.check_against(mesh)` |
| `case_file.py` | the YAML parse (`_CaseLoader`), `read_case` / `write_case`, `CaseFile`, `CheckedCase` |
| `mesh_source.py` | `MeshSource.read(directory) -> Mesh` → `OpenFOAMMesh` (read) / `StructuredGrid` + `AxisGrading` → `GeometricGrading` (generated) |
| `forcing.py` | `DriveSpec` → `BulkVelocity` (builds `flow.MassFlow`); `SourceSpec` → `BodyForce` (builds `flow.UniformBodyForce`) |
| `fluid.py` | `Fluid` |
| `physics.py` | `Physics` → `Laminar` / `RANS` |
| `boundaries.py` | `PatchCondition` → `Inlet` / `Outlet` / `Wall`; `InletTurbulence` → `FixedTurbulence` |
| `solver.py` | `SolverSpec` → `CoupledMarch` / `FlowMarch` (both on the private `_March`, their shared settings) / `Segregated`; `ViscosityRamp`; `RootSolve`; `solver_for(spec)` |

- **`case/` is the TOP layer: nothing else in `aquaflux` imports it**, pinned by
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

- **A starting state** (#544) — no `initial` section, no checkpoint loader, and the ramp takes no seed.
- **Where results go** — a `run` entry point (`python -m aquaflux run case.yaml`) and an outputs section
  (VTK, OpenFOAM time directories, checkpoints). The next PR.
- **A laminar case holding a bulk velocity (#541)** — `FlowMarch` refuses one (`solve_flow_march` refuses a
  `MassFlow` drive) and there is no laminar segregated solve; `bulk_velocity_flow_solve` exists but is a
  bordered Newton, not a march.
- **A coupled march holding a bulk velocity (#541)** — `CoupledMarch` refuses one and points at `Segregated`.
  `solve_coupled_mass_flow` exists, but it is the older `RootSolver` path: `BlockDiagonal` only, no dual
  time, no retry, no step control, no ramp. Routing a file there would publish a surface that silently
  cannot take half the section's settings; the gap is the library's (#277/#448 made the march generic but
  not the bordered constraint).
- **A 3D structured grid** — `structured_grid_3d` has neither grading nor periodic axes, so `StructuredGrid`
  is 2D only rather than half-supporting a uniform 3D box no case needs.
- **Other source kinds** — a `sources:` section beyond `BodyForce` waits on #362 (sources declaring their
  inputs); `BodyForce` reads no field and no gradient, so it needs nothing #362 would add.
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
