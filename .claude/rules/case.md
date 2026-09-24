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

## Status — phases A–C BUILT (2026-09-24); phase D (building the assemblers) is NOT

The construction order #374 records is A spec → B topology → C geometry → D equations → E state →
F frozen solver → G drive. **What exists stops before any geometry**: `read_case(path)` →
`CaseFile(spec, directory)` (phase A), and `CaseFile.check()` → `CheckedCase(spec, mesh)` (phase B: the
mesh read and `validate()`d, and the spec checked against its topology). ~1 s on pitzDaily (12225
cells, ASCII read; one run, 2026-09-24, macOS arm64) — the "check the file" stop #374 asked to be kept
cheap. **Nothing here builds a `MomentumContinuity`, an `SSTTurbulence` or a `CoupledRANS` yet**; the
patch kinds hold the settings the closures will be built from, but no code derives a closure from them.
When phase D lands, `CheckedCase` is where `build()` goes, and the solver section must produce a
`state -> NewtonStrategy` builder, never a built step (#374: the step is rebuilt from mid-march states
at every Reynolds rung and every refresh).

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
  the wall set will be **derived** from one statement, so they cannot disagree. The cost, accepted: a
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
- **A closed domain is REFUSED at read, not left to solve a singular system.** No `Outlet` ⇒ the pressure
  level is free ⇒ it needs a datum, and a file cannot state one until #500 (`PinnedPoint` / `PinnedPatch`)
  exists. The message tells the reader to build such a case in code. This is also why `MassFlow` is not
  a registered drive: both periodic channels are closed.
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

- **Phase D** (build the assemblers from a `CheckedCase`) — the next step. It must write the one
  `PropertyModel` the fluid implies (dynamic viscosity as a **JAX array**, so continuation rungs share
  compiled code — bfs3d's driver records why) and hand it to both assemblers.
- **The solver section** (march, preconditioner, convergence, Reynolds schedule) — a `state -> step`
  builder per #374. The preconditioner part already reads through `preconditioner_spec_from_mapping`.
  The validation drivers carry closures no file can state (`point_setup`, the damping tapers reading a
  residual at each rung's seed, the Reynolds companion function, logger metrics); decide which become
  named values and which stay code.
- **The pressure datum** (#500), then **`MassFlow`** and a structured-grid mesh source (the channels).
- **Patch types / `inGroups`** (#364) — would let a file say "all walls" instead of listing each patch.
- **`IntensityLength`** inlet turbulence (`inlet_k` / `inlet_omega`) — `InletTurbulence.inflow(velocity)`
  already takes the velocity for it.
- **A passive-scalar case referring to another case's converged flow** (#375; `bfs3d_species` imports the
  flow driver by path today) — a dependency edge between cases, not a field on one.
- **Profiles** (`DirichletField`, a callable) — not plain data; a named-profile kind if ever needed.

## The one case file in the repository

`validation/pitzdaily_openfoam/case.yaml` states the case `compare.py` builds in code (same mesh, fluid,
physics, boundaries, numerics at that driver's defaults). `compare.py` does **not** read it yet.
`tests/unit/test_case_file.py::test_the_shipped_pitzdaily_file_reads_as_the_case_its_driver_builds_and_fits_its_mesh`
compares it against a `CaseSpec` written out independently in the test and checks it against the real
mesh, so a renamed patch or a setting the loader stops accepting fails the fast gate. ⚠️ It does **not**
catch the driver's constants moving (`U_IN`, `K_IN`, …): the test's expected value is a third copy. When
phase D lets the driver read the file, delete that duplication rather than maintaining three.
