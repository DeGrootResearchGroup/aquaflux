# Case files

A case file describes a whole case — its mesh, fluid, physics, boundary patches and
numerics — as one YAML document, which {func}`~aquaflux.case.read_case` reads into a
{class}`~aquaflux.case.CaseSpec`. Every setting is checked where it appears, so a misspelt
name, a value of the wrong form, or a setting the case's physics would never read is refused
with the path to it, rather than loading and failing (or being silently ignored) later.

Reading a file and checking it against its mesh builds no equation and computes no geometry,
so a case can be checked in about the time its mesh takes to read; building it then assembles
the equations.

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
  advection: {kind: FirstOrderUpwind}
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
  gradient: {kind: MultipleCorrectionGradient}
```

```python
from aquaflux.case import read_case

case = read_case("case.yaml")   # the file, checked on its own terms
checked = case.check()          # its mesh read, and the case checked against it
checked.mesh.n_cells            # 12225
coupled = checked.build()       # the coupled flow and SST closure, ready to solve
```

Each value in the file is a mapping whose `kind` names what it is, and whose other keys are
its settings; a setting left out takes its default. The `fluid` and `numerics` sections have
one form each, so they need no `kind`, and the top level is the case itself.

## The sections

**`mesh`** — where the mesh is read from. {class}`~aquaflux.case.OpenFOAMMesh` reads an
OpenFOAM polyMesh, or a case directory holding `constant/polyMesh`; a relative `path` is taken
from the directory the case file sits in. A case one cell thick between `empty` patches reads
as a two-dimensional mesh, and those patches are not named in the file.

**`fluid`** — a constant-density fluid, stated once for every equation. Give the `density`
and **exactly one** of `kinematic_viscosity` and `dynamic_viscosity`: the other follows from
the density, so a file cannot state two viscosities that disagree.

**`physics`** — which equations are solved:

- {class}`~aquaflux.case.Laminar` — laminar incompressible flow.
- {class}`~aquaflux.case.RANS` — Reynolds-averaged flow closed by k–ω SST, the flow and the
  closure solved together. It requires `advection` (how `k` and `omega` are advected) and
  optionally takes the model constants (`model: {kind: SSTModel, ...}`), the variable each
  field is solved in (`k_variable`, `omega_variable`: `DirectScalars` or `LogScalars`), and
  `explicit_production_limiter`.

Everything that belongs to the turbulence closure lives inside `RANS` or on a boundary patch,
and a laminar case refuses any of it.

**`boundaries`** — one entry per boundary patch, by the patch's name in the mesh. Each patch
is described once, for every field:

| kind | velocity | pressure | in a `RANS` case |
|---|---|---|---|
| {class}`~aquaflux.case.Inlet` | `velocity`, prescribed | follows the interior | requires `turbulence`, the inflow `k` and `omega` |
| {class}`~aquaflux.case.Outlet` | follows the interior | `pressure`, prescribed | `k` and `omega` follow the interior |
| {class}`~aquaflux.case.Wall` | no slip | follows the interior | a wall for the closure; `k: zero_gradient` (default) or `k: zero` |

The closures each equation needs, and the set of walls the turbulence closure measures its
wall distance from, all follow from this one statement.

**`numerics`** — the discretization choices common to every case: `momentum_advection`
(required, since a flow with no advection is Stokes flow rather than a default) and
`gradient` (the cell-gradient reconstruction, for every field; unset,
{data}`~aquaflux.schemes.DEFAULT_GRADIENT_SCHEME`).

**`drive`** — what sets the flow in motion. Unset, and the only value a file can name today,
it is {class}`~aquaflux.flow.BoundaryDriven`: the boundary conditions do.

## What is checked, and when

When the file is **read**:

- every `kind` is known and every setting is a setting of it, of a form it can hold — a
  number where a number belongs, one of the offered choices where there is a choice;
- every required setting is present;
- the physics accepts the boundaries — no turbulence setting in a laminar case, and inflow
  turbulence at every inlet of a `RANS` one;
- some patch fixes the pressure level (an `Outlet`).

When the case is **checked** against its mesh ({meth}`~aquaflux.case.CaseFile.check`):

- the mesh's topology is valid;
- every patch named is a boundary patch of the mesh, and every boundary face lies in a named
  patch — a face given no condition would otherwise keep a zero face value, a boundary
  condition nobody chose;
- each patch fits the mesh — an inlet velocity has one component per dimension.

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
from aquaflux.turbulence import solve_coupled

coupled = read_case("case.yaml").check().build()
flow, k, omega = solve_coupled(coupled)          # starts from hybrid_initialize(coupled)
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

The result is the problem, not a solve: how it is marched — the preconditioner, the continuation,
the stopping test — is configured against it in code.

## What a case file cannot describe yet

- **A closed domain.** A domain with no `Outlet` has a free pressure level and needs a datum,
  which a file cannot state yet; such a case (a lid-driven cavity, a streamwise-periodic
  channel held at a bulk velocity) is built in code, with
  `MomentumContinuity.build(pressure_pin=...)`.
- **A boundary profile** — an inlet velocity or value varying across the patch. Those are
  functions of position, built in code.
- **The solve** — the march, its preconditioner and its convergence test. A case file
  describes the problem; the solve is configured in code, and
  {func}`~aquaflux.turbulence.preconditioner_spec_from_mapping` reads the preconditioner's
  part from a mapping of the same form.

## Writing a case

{func}`~aquaflux.case.write_case` writes a {class}`~aquaflux.case.CaseSpec` built in code as a
file that reads back as an equal case, omitting every setting left at its default;
{func}`~aquaflux.case.case_spec_to_mapping` and {func}`~aquaflux.case.case_spec_from_mapping`
do the same to and from a plain mapping, for a caller with its own format.
