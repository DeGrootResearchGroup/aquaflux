# Case files

A case file describes a whole case — its mesh, fluid, physics, boundary patches, numerics,
and how it is solved — as one YAML document, which {func}`~aquaflux.case.read_case` reads into a
{class}`~aquaflux.case.CaseSpec`. Every setting is checked where it appears, so a misspelt
name, a value of the wrong form, or a setting the case's physics would never read is refused
with the path to it, rather than loading and failing (or being silently ignored) later.

Reading a file and checking it against its mesh builds no equation and computes no geometry,
so a case can be checked in about the time its mesh takes to read; building it then assembles
the equations, and solving it runs the solve the file describes.

## An example

The two-dimensional backward-facing step (the OpenFOAM `pitzDaily` tutorial), solved with the
k–ω shear-stress transport (SST) closure:

```yaml
mesh:
  kind: OpenFOAMMesh
  path: runs/kwsst/polyMesh

fluid:
  density: 1.0
  kinematic_viscosity: 1.0e-5

physics:
  kind: RANS
  omega_variable: {kind: LogScalars}
  explicit_production_limiter: true

boundaries:
  inlet:
    kind: Inlet
    velocity: [10.0, 0.0]
    turbulence: {kind: FixedTurbulence, k: 0.375, omega: 440.15}
  outlet: {kind: Outlet, pressure: 0.0}
  upperWall: {kind: Wall, k: zero_gradient}
  lowerWall: {kind: Wall, k: zero_gradient}

numerics:
  momentum_advection:
    kind: LimitedUpwind
    limiter: {kind: VenkatakrishnanLimiter}
  turbulence_advection: {kind: FirstOrderUpwind}
  gradient: {kind: MultipleCorrectionGradient}

solver:
  kind: CoupledMarch
  convergence: {kind: Convergence, rtol: 0.0, atol: 1.0e-5}
  dual_time: {kind: DualTimeLoop, inner_steps: 5, inner_tol: 0.01}
  step_control: {kind: CflResidualDualTimeControl, beta_start: 0.5, beta_min: 0.005}
  continuation: {kind: ViscosityRamp, anchor: 100.0, stations: 16, steps_per_station: 1, scale: flow}
```

```python
from aquaflux.case import read_case

case = read_case("case.yaml")         # the file, checked on its own terms
checked = case.check()                # its mesh read, and the case checked against it
checked.mesh.n_cells                  # 12225
coupled = checked.build()             # the coupled flow and SST closure
flow, k, omega = checked.solve(coupled)   # marched as the solver section says
```

Each value in the file is a mapping whose `kind` names what it is, and whose other keys are
its settings; a setting left out takes its default. The `fluid` and `numerics` sections have
one form each, so they need no `kind`, and the top level is the case itself.

## The sections

**`mesh`** — where the mesh comes from, read or generated:

- {class}`~aquaflux.case.OpenFOAMMesh` reads an OpenFOAM polyMesh, or a case directory holding
  `constant/polyMesh`; a relative `path` is taken from the directory the case file sits in. A
  case one cell thick between `empty` patches reads as a two-dimensional mesh, and those patches
  are not named in the file.
- {class}`~aquaflux.case.StructuredGrid` generates a two-dimensional structured grid on the box
  `[0, lx] x [0, ly]`, with patches named by side — `left`, `right`, `bottom`, `top` — or, given
  three counts and three lengths, a uniform three-dimensional box with `back` and `front` as well. An axis
  listed as `periodic` wraps around, and its two sides are not patches at all; an axis given a
  `grading` has its cells sized by it
  ({class}`~aquaflux.case.GeometricGrading`: finest at the walls, growing by `growth`):

  ```yaml
  mesh:
    kind: StructuredGrid
    cells: [4, 96]
    lengths: [1.0, 2.0]
    periodic: [x]
    grading:
      y: {kind: GeometricGrading, growth: 1.09}
  ```

**`fluid`** — a constant-density fluid, stated once for every equation. Give the `density`
and **exactly one** of `kinematic_viscosity` and `dynamic_viscosity`: the other follows from
the density, so a file cannot state two viscosities that disagree.

**`physics`** — which equations are solved:

- {class}`~aquaflux.case.Laminar` — laminar incompressible flow.
- {class}`~aquaflux.case.Radiation` — the light of ultraviolet lamps rather than a flow; it has no
  `fluid` or `numerics` section, and its patches are lamps and walls. See {doc}`radiation_case`.
- {class}`~aquaflux.case.RANS` — Reynolds-averaged flow closed by k–ω SST, the flow and the
  closure solved together. It optionally takes the model constants (`model: {kind: SSTModel, ...}`), the variable each
  field is solved in (`k_variable`, `omega_variable`: `DirectScalars` or `LogScalars`), and
  `explicit_production_limiter`.

A setting that only the turbulence closure reads, such as a wall's `k` condition or
`numerics.turbulence_advection`, is refused by a laminar case; the browser interface does not
show one for a laminar case at all.

**`boundaries`** — one entry per boundary patch, by the patch's name in the mesh. Each patch
is described once, for every field:

| kind | velocity | pressure | in a `RANS` case |
|---|---|---|---|
| {class}`~aquaflux.case.Inlet` | `velocity`, prescribed | follows the interior | requires `turbulence`, the inflow `k` and `omega` |
| {class}`~aquaflux.case.Outlet` | follows the interior | `pressure`, prescribed | `k` and `omega` follow the interior |
| {class}`~aquaflux.case.Wall` | no slip — at rest, or at `velocity` for a moving wall | follows the interior | a wall for the closure; `k: zero_gradient` (default) or `k: zero` |

The closures each equation needs, and the set of walls the turbulence closure measures its
wall distance from, all follow from this one statement. A moving wall — the driven lid of a
cavity, `{kind: Wall, velocity: [1.0, 0.0]}` — is a wall in every other respect: it passes no
fluid, and it is a wall to the turbulence closure.

An entry may instead name a **patch group**, and its condition then applies to every patch in
the group. An OpenFOAM mesh declares its groups in the `boundary` file's `inGroups` entries —
most meshes put every `wall` patch in a group named `wall` — so a mesh with many walls needs
one entry for all of them:

```yaml
boundaries:
  inlet: {kind: Inlet, velocity: [10.0, 0.0], turbulence: {kind: FixedTurbulence, k: 0.375, omega: 440.15}}
  outlet: {kind: Outlet, pressure: 0.0}
  wall: {kind: Wall}
```

No patch may be reached twice, by its own name and by a group, or by two groups; and a name
that is both a patch and a group of other patches is refused, since it does not say which
faces it means.

An inlet's `turbulence` is given either outright or the way it is usually known:

- {class}`~aquaflux.case.FixedTurbulence` — `k` and `omega` as values;
- {class}`~aquaflux.case.IntensityLength` — a turbulence `intensity` (the r.m.s. velocity
  fluctuation as a fraction of the inflow speed, `0.05` for 5%) and a `length` scale, giving
  `k = 1.5 (I |U|)^2` and `omega = sqrt(k) / (C_mu^(1/4) L)`, with `C_mu` the case's own SST
  constant `beta_star`:

  ```yaml
  turbulence: {kind: IntensityLength, intensity: 0.05, length: 2.54e-3}
  ```

**`numerics`** — the discretization choices of a flow: `momentum_advection`
(required, since a flow with no advection is Stokes flow rather than a default),
`turbulence_advection` (how `k` and `omega` are advected; required by a RANS case, refused by a
laminar one) and
`gradient` (the cell-gradient reconstruction, for every field; unset,
{data}`~aquaflux.schemes.DEFAULT_GRADIENT_SCHEME`). The choices are
{class}`~aquaflux.schemes.CompactGreenGauss`, {class}`~aquaflux.schemes.CorrectedGreenGauss`,
{class}`~aquaflux.schemes.MultipleCorrectionGradient`, which suits hexahedral and polyhedral
meshes, and {class}`~aquaflux.schemes.ProjectedStencilGradient`, the one to use on a
tetrahedral mesh. {doc}`gradient_reconstruction` compares them.

**`drive`** — what sets the flow in motion when the boundary conditions do not. Unset, they
do. A streamwise-periodic channel prescribes the velocity nowhere, and is held at a bulk
(volume-averaged) velocity instead, by a uniform streamwise force solved for with the flow:

```yaml
drive: {kind: BulkVelocity, target: 1.0, direction: x, initial_force: 0.004}
```

{class}`~aquaflux.case.BulkVelocity` holds `target` along `direction` (default `x`).
`initial_force` is where the solve for the force starts — a guess, not a setting of the
problem: the force it converges to does not depend on it, though how quickly it gets there can.

**`sources`** — terms added to the momentum balance. {class}`~aquaflux.case.BodyForce` is a
prescribed uniform force per unit volume, the other way to drive a periodic channel — at a
fixed force, with the bulk velocity whatever it sustains:

```yaml
sources:
  - {kind: BodyForce, force: [0.004, 0.0]}
```

**`pressure_datum`** — where the pressure level is fixed, in a domain that no patch fixes it
in. Incompressible flow determines the pressure only up to a constant; an `Outlet` supplies
the level, and a domain with none — a lid-driven cavity, say — needs it fixed somewhere else:

```yaml
pressure_datum: {kind: PinnedPoint, point: [0.0, 0.0], value: 0.0}
```

The cell nearest `point` has its pressure held at `value` (0 unless given). Which cell that
is changes the pressure only by a constant, and a point names the same place however the mesh
is numbered. A datum is **required** when no patch is an `Outlet` and **refused** when one is:
without one the system is singular, and beside an outlet it would fix the level twice.

**`solver`** — how the case is solved. It names one of four solves, and every setting in it
is optional unless noted: an unset one leaves that solve's own default in force. Unset
altogether, a `RANS` case is marched by `CoupledMarch`, a `Laminar` one by `FlowMarch` and a
`Radiation` one solved by `RadiationSolve`, each with every setting at its default.

| kind | physics | solves by |
|---|---|---|
| {class}`~aquaflux.case.CoupledMarch` | `RANS` | one coupled march of the flow and the closure ({func}`~aquaflux.turbulence.solve_coupled`) |
| {class}`~aquaflux.case.FlowMarch` | `Laminar` | one coupled march of the flow ({func}`~aquaflux.flow.solve_flow_march`) |
| {class}`~aquaflux.case.Segregated` | `RANS` | alternating the flow and the closure ({func}`~aquaflux.turbulence.solve_segregated`) |
| {class}`~aquaflux.case.RadiationSolve` | `Radiation` | the lamps' light and the walls' reflections ({func}`~aquaflux.radiation.solve_scene`); see {doc}`radiation_case` |

The two marches share their settings, each a value of the solver library written in the file:

- `max_steps` — the outer-step cap;
- `convergence` — the stopping test, {class}`~aquaflux.solve.Convergence`: a `measure`
  (`RowScaled`, `BlockScaled` or `Euclidean`) and its `rtol` and `atol`;
- `preconditioner` — {class}`~aquaflux.solve.MaterializedJacobian`, or for `CoupledMarch`
  {class}`~aquaflux.turbulence.BlockDiagonal`, with the settings
  {func}`~aquaflux.turbulence.preconditioner_spec_from_mapping` reads;
- `dual_time` — run each outer step as an inner loop, {class}`~aquaflux.solve.DualTimeLoop`;
- `linear_solve` — each step's Krylov regime, {class}`~aquaflux.solve.LinearSolveSettings`;
- `step_control` — how the pseudo-time shift adapts: {class}`~aquaflux.solve.DualTimeControl`,
  {class}`~aquaflux.solve.ResidualRatioDualTimeControl` or
  {class}`~aquaflux.solve.CflResidualDualTimeControl`;
- `retry` — when a bad step is redone, {class}`~aquaflux.solve.RetryPolicy`, whose tighter
  `solver` is a {class}`~aquaflux.solve.GmresSolve`.

`CoupledMarch` adds the closure's: `turbulence_damping` (a multiplier on the `k` and `omega`
rows' shift), `positivity_floor` and `positivity_projection` (how `k` is kept positive), and
`continuation` — {class}`~aquaflux.case.ViscosityRamp`, which starts the march at `anchor` times
the case's viscosity and walks it down to the case's in `stations` geometric steps, each held
for `steps_per_station` outer steps, keeping the state and the preconditioner throughout. Its
`scale` says which viscosity a station scales: the flow's only (`flow`), or the flow's and the
closure's (`both`, the default). A high-Reynolds-number case needs one: from a cold start at
its own viscosity the march integrates a long transient before the flow develops.

`Segregated` takes `sweeps` (required: the most it may take), `relaxation` and
`relaxation_max` (the closure update's under-relaxation), `increment_tol` (the change over a
sweep at which it stops), `flow_solve` and `scalar_solve` ({class}`~aquaflux.case.RootSolve`:
each Newton solve's `max_steps`, `convergence` and `linear_solver`, a
{class}`~aquaflux.solve.GmresSolve` or a {class}`~aquaflux.solve.DirectSolve`), and
`scalar_preconditioner`. It is the solve that holds a `BulkVelocity`: the two marches solve the
fields with the force fixed, so they refuse one rather than converge at the force's starting
guess.

```yaml
solver:
  kind: Segregated
  sweeps: 100
  relaxation: 0.9
  flow_solve: {kind: RootSolve, linear_solver: {kind: DirectSolve}}
  scalar_solve: {kind: RootSolve, max_steps: 400}
  scalar_preconditioner: {kind: ScalarAir}
```

**`initial`** — what the case starts from, when it does not start from scratch. Optional: with no
section the solve builds its own starting state, a potential flow (and, for a `RANS` case, a
hybrid initial condition for `k` and `omega`).

```yaml
initial:
  kind: Checkpoint
  path: ../first/checkpoints   # the checkpoints directory of an earlier run, relative to the case file
  step: latest                 # or the number of a checkpoint that run kept
```

{class}`~aquaflux.case.Checkpoint` resumes a run that stopped short from the checkpoints it
wrote. It reads the physical fields the earlier run saved, so the two cases need not solve in
the same variables (one may solve for the logarithm of `omega` and the other for `omega`); they
must have the same physics and the same mesh. The viscosity, the boundary values and the solver
settings may differ. Write the new run to its own `outputs.directory`: a run replaces what it
finds in its output directory, so one that reads from there is refused.

A restart also resumes the march. The checkpoint records where the stopped march had got to: the
residual it began at, the residual its damping was anchored at (which is a later one once it has
refreshed its preconditioner), and the shift its step control had come down to. The new run
measures its damping and its stopping bar against those residuals rather than against the
residual at the state it was handed, and its step control opens at that shift rather than at the
top of its ramp, so it takes the steps the stopped march had left. That holds when the new file
states the same problem -- the same mesh, physics, fluid, boundaries and numerics -- and judges a
residual in the same measure (`convergence.measure`); the step budget, the preconditioner, the
outputs and the like may change. If the file differs in any of the first, the run starts from the
checkpoint's fields as a new march, and its `run.yaml` records `resumption: null`. The first step
of a resumed march with a step control holds the shift it resumes from, since there is no step
before it to adapt from; a control that adapts by the ratio of successive residuals forms its
first ratio one step later.

```yaml
initial:
  kind: Fields
  path: of_case        # an OpenFOAM case directory, relative to the case file
  time: "1000"         # the time directory to start from; quote it
```

{class}`~aquaflux.case.Fields` starts the case from one time directory of an OpenFOAM case: another
program's converged solution, or the output of an earlier run's
{class}`~aquaflux.case.OpenFOAMTime`. It reads the files `U` and `p` (and under `RANS` also `k` and
`omega`). The case's mesh must be that OpenFOAM case's own, which is what numbers the cells; only
the number of cells can be checked against it. The pressure is read as an incompressible OpenFOAM
solver holds it, per unit density, and multiplied by the case's fluid density; `OpenFOAMTime`
writes it the same way, so the two are inverses and a case of density one reads it unchanged. It
carries no march to continue, so the run begins as a new march from the state.

A two-dimensional OpenFOAM mesh was extruded along one axis, and its vector fields carry a
component along it, which must be zero. The axis is recovered from the mesh's points, which
decide it for any real extrusion; a mesh whose extents leave it ambiguous (a slab as thick as it
is tall) needs `extruded_axis: z` (or `x`, `y`) stated. It is refused for a three-dimensional mesh.

A {class}`~aquaflux.case.ViscosityRamp` opens on a seed fitted to its own anchor station, so a
case with a `continuation` refuses an `initial` section: drop the `continuation` to resume at
the case's own viscosity.

**`outputs`** — what a run writes, and where. Every part is optional; with no section at all a
run writes the fields as VTK, the log and the history into `results/` beside the case file.

```yaml
outputs:
  directory: results                    # relative to the case file
  log: march.log                        # the march's per-step table; null for the terminal only
  history: history.csv                  # the same steps as comma-separated values; null for none
  checkpoints: {kind: Checkpoints, every: 1, keep: 3}
  fields:
    - {kind: Vtk, file: fields.vtu}
    - {kind: OpenFOAMTime, case: of_case, time: "1000", fields: [U, p]}
```

- {class}`~aquaflux.case.Vtk` writes the converged fields as one VTK unstructured-grid file,
  for any mesh.
- {class}`~aquaflux.case.OpenFOAMTime` writes them as a time directory of an OpenFOAM case —
  `case` is that case's directory, relative to the case file — taking each field's dimensions
  and boundary conditions from the file of the same name in its `template_time` directory
  (`0` unless given), so the result restarts in the solver the case is set up for. It needs an
  OpenFOAM mesh, and a template for every field it writes. Quote `time`: a bare number reads as
  a number. On a two-dimensional mesh the zero component of each vector goes on the axis the mesh
  was extruded along, recovered from its points; state `extruded_axis` (`x`, `y` or `z`) when
  they leave it ambiguous. The pressure is written per unit density (`p / density`), as an incompressible
  OpenFOAM solver holds it; the other writers write the pressure itself.
- {class}`~aquaflux.case.PatchVtk` writes the boundary patches, and the fields on their faces, as
  one VTK polygonal-data file per patch (`patches/<patch>.vtp`) indexed by a multiblock file
  (`patches.vtm`), one block per patch named by it. A radiation case's irradiance is written this
  way; a flow case has no patch fields, and writes the patches alone.
- Each writer's `fields` names what it writes — `U` and `p`, and under `RANS` also `k`, `omega`
  and `nut`; left out, all of them. The pressure is the solved one.
- The history ({class}`~aquaflux.solve.StepHistory`) has one row per step of the march — the
  step, the seconds since the run started, the residual and its ratio to the starting one, the
  line-search factor, the shift, the linear-solve cost, the continuation station the step drove
  and whether that was the case's own problem, how many times the preconditioner was refitted for
  it and what that took, why the step was redone (if it was), and the residual split by equation
  (`residual_of_u`, `residual_of_p`, …, under the default row-scaled measure) — every number at
  full precision, each row written as the step ends. It is what a convergence plot reads. Its
  header is written with the first row, so the file is empty until then; the segregated solve
  takes no steps, so its history stays empty.
- {class}`~aquaflux.case.Checkpoints` writes the march's state every `every` steps into
  `checkpoints/`, keeping the latest `keep`, so a run that stops has not lost its work. Each
  file holds the physical fields (`U`, `p` and, under `RANS`, `k` and `omega`) with what they
  belong to — the physics, the number of cells and a digest of the mesh — so a later case can
  start from one (see `initial`). The segregated solve takes no steps a checkpoint could be
  written at.

## Editing a case file from another program

Four commands let a program read and write case files as aquaflux does — the
browser interface's Setup section is built on them. Each prints JSON:

```bash
aquaflux schema                                   # every kind, field and choice a case file may hold
aquaflux show case.yaml                           # the file as aquaflux reads it, and why it is refused
aquaflux write copy.yaml --relative-to . < case.json  # check, then write; re-base relative paths
aquaflux mesh out/ --relative-to . < mesh.json        # read or generate a mesh section, as VTK
```

`schema` is read off the same definitions a file is checked against, so a program
that builds its choices from it offers exactly what aquaflux accepts. `write`
writes nothing it would refuse, and prints the reason instead (exit status 2).

Starting Python and importing aquaflux takes a few seconds, far longer than any of
these commands takes itself, so a program asking many of them can keep one process
instead:

```bash
aquaflux serve        # answers schema, show, write, mesh and check requests, one per line
```

Each request is one line of JSON, `{"arguments": ["show", "case.yaml"], "stdin": null}`,
and each answer is one line, `{"status": 0, "output": "...", "errors": "..."}`: the
exit status the command would have had and what it would have printed. It stops at
the end of its input.

## What is checked, and when

When the file is **read**:

- every `kind` is known and every setting is a setting of it, of a form it can hold — a
  number where a number belongs, one of the offered choices where there is a choice;
- every required setting is present;
- the physics accepts the boundaries — no turbulence setting in a laminar case, and inflow
  turbulence at every inlet of a `RANS` one;
- the pressure level is fixed exactly once — by an `Outlet`, or, with none, by a
  `pressure_datum`;
- the solver solves this physics, and can hold its drive, and can start from the `initial`
  state given — a viscosity ramp cannot;
- an `OpenFOAMTime` output has an OpenFOAM mesh to write for.

When the case is **checked** against its mesh ({meth}`~aquaflux.case.CaseFile.check`):

- the mesh's topology is valid;
- every name is a boundary patch of the mesh or a group of them, no patch is reached twice, and
  every boundary face lies in a patch given a condition — a face given no condition would
  otherwise keep a zero face value, a boundary condition nobody chose;
- each patch fits the mesh — an inlet or wall velocity has one component per dimension — and so
  do the drive's direction and each source's force;
- a pressure datum's point has one coordinate per dimension and lies within the mesh's
  bounding box.

When a run is **prepared** ({func}`~aquaflux.case.prepare_run`), before anything is built, an
`initial` state is read and checked against the case. A checkpoint must be of the same physics and
the same mesh (the same number of cells, and the same cell numbering and node positions); an
OpenFOAM time directory must be on the case's own OpenFOAM mesh, with the right number of cells
and finite values. Either must lie outside the run's own output directory. A state that does not
fit is refused before an earlier run's checkpoints are cleared.

Every problem found is reported at once.

## How YAML is read

A case file is parsed by the YAML 1.2 rules for plain values, which differ from the older
rules many parsers default to in ways that would otherwise misread an ordinary setting:

- `1e-5` is a number (not the string `"1e-5"`);
- only `true` and `false` are booleans — `yes`, `no`, `on` and `off` are strings, and are
  refused where a boolean belongs;
- `010` is ten, not eight;
- a mapping that names one key twice — two `inlet` entries, say — is refused, rather than
  keeping whichever came last.

## Building the case

{meth}`~aquaflux.case.CheckedCase.build` turns a checked case into the problem it describes —
the same assembler the initializers and the solves already take:

| physics | builds |
|---|---|
| `Laminar` | {class}`~aquaflux.flow.MomentumContinuity` |
| `RANS` | {class}`~aquaflux.turbulence.CoupledRANS`, holding the flow and the SST closure |

```python
checked = read_case("case.yaml").check()
coupled = checked.build()
flow, k, omega = checked.solve(coupled)          # the solver section's march
```

Nothing is stated twice on the way. Each patch builds its own closures — the flow's, and under
`RANS` the `k` and `omega` ones:

| patch | flow | `k` | `omega` |
|---|---|---|---|
| `Inlet` | prescribed velocity | the inflow `k` | the inflow `omega` |
| `Outlet` | prescribed pressure | zero gradient | zero gradient |
| `Wall` | no slip | zero gradient, or zero (`k: zero`) | fixed in the cells next to the wall |

The turbulence closure's walls are the patches that are walls to the flow, and both equations read
the one property model the fluid gives, so the flow and the closure cannot describe different
walls or different fluids. The geometry is computed once, here — this is where loading a case
starts to cost something, where checking it did not.

## Solving the case

{meth}`~aquaflux.case.CheckedCase.solve` solves the built problem as the `solver` section says,
and returns the converged fields — `(flow, k, omega)` for a `RANS` case, the flow state for a
`Laminar` one. What the file holds is settings, never a built solve: a march fits its
preconditioner to the state it has reached, again at each continuation station and each
refresh, and each fit is made from the file's settings.

A case with an `initial` section starts from the state it names
({meth}`~aquaflux.case.CheckedCase.starting_fields` reads it); without one, the solve starts
from its own.

A solve can be watched without being changed: keywords that only observe it — `on_step`,
`on_checkpoint`, `on_retry`, `inner_observer` — are passed to the library solve beside the
file's settings. A keyword that is one of the solve's settings is refused, whether the file
sets it or leaves it at its default, so code that runs a case cannot change what the case says.

## Running a case

The `aquaflux` command runs a file from start to finish:

```bash
aquaflux check case.yaml            # read the file and check it against its mesh, in seconds
aquaflux plan case.yaml             # where a run would write, and which earlier results it would replace
aquaflux run case.yaml              # read, check, build, solve, and write the outputs
aquaflux run case.yaml --overwrite  # replace an earlier run's results
```

`python -m aquaflux` is the same command. A run writes, into the output directory:

- the converged fields, by each of the section's writers;
- the log, one row per outer step, echoed to the terminal as it goes;
- the checkpoints, when asked for;
- `case.yaml`, the case as it ran — the solver written out even when the file left it to the
  default, and its relative paths re-based so the copy reads where it lies;
- `run.yaml`, a record of the run: the aquaflux version and the commit it ran from, when it
  started and how long it took, the solver, how many steps it took, where its residual ended,
  whether it converged, what it wrote and, under `results`, the scalar results the physics
  reports (a radiation case's lamp power and where it goes); for a run that started from an earlier
  state, which one: its file, the residual the earlier run had reached and a digest of the case that
  wrote it.

`run` exits with status 0 when the solve converges. A solve that stops short of its stopping
test — or is interrupted, with Ctrl-C or the browser interface's Stop — writes no fields (what it
holds is not a solution) but still writes its log, its checkpoints and `run.yaml`, and exits with
status 1. A file that is refused exits with status
2, with the reason. A run refuses an output directory that already holds results unless given
`--overwrite`, which replaces the files the run writes and clears its old checkpoints.

In code, {func}`~aquaflux.case.prepare_run` reads and checks a file for a run and
{meth}`~aquaflux.case.PreparedRun.run` runs it, returning a {class}`~aquaflux.case.RunRecord`;
{func}`~aquaflux.case.plan_run` says where a run would write and what it would replace, by the
same rule a run refuses on, without checking the mesh.

## What a case file cannot describe yet

- **A boundary profile** — an inlet velocity or value varying across the patch. Those are
  functions of position, built in code.
- **A starting state on a mesh that is not an OpenFOAM one** — `Fields` reads an OpenFOAM time
  directory onto that case's own mesh; there is no way yet to start from another program's fields on
  a generated grid.
- **A laminar case holding a bulk velocity** — the flow march solves with the force fixed, and
  no laminar solve holds the constraint from a file.

## Writing a case

{func}`~aquaflux.case.write_case` writes a {class}`~aquaflux.case.CaseSpec` built in code as a
file that reads back as an equal case, omitting every setting left at its default;
{func}`~aquaflux.case.case_spec_to_mapping` and {func}`~aquaflux.case.case_spec_from_mapping`
do the same to and from a plain mapping, for a caller with its own format.
