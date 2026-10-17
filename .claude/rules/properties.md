---
paths:
  - "aquaflux/properties/**"
---

# Rules — `aquaflux/properties/` (physical property model)

> **Provenance boundary (binding).** This file may cite the C++/Fortran precursors; none of that may
> reach the shipped surface (`.py`, `docs/`) — root `CLAUDE.md` Comment Convention.

Physical properties (density, viscosity, conductivity, diffusivity, ...) as per-cell fields, decoupled
from the numerics: operators consume them, this package owns them (the C++ precursor put a "transport
coefficient" inside discretization — the wrong layer).

## Module layout
- `property.py` — `Property` (`evaluate(cell_zones, fields) -> (n_cells,)`) → `Constant`, `ZoneConstant`,
  `FieldProperty`. Each object is **one** property; the kind (constant / zonal / per-cell field) is
  orthogonal to which property it is. `FieldProperty(values)` carries an arbitrary per-cell array (e.g. an
  externally computed or frozen effective viscosity); `values` is a differentiable leaf, length-checked.
- `model.py` — `PropertyModel(properties: dict[str, Property])`: `evaluate(cell_zones, fields=None)` →
  `{name: (n_cells,)}`; `require(*names)` fail-fast checks an operator's named coefficient is supplied.

## Binding decisions
- **Separate top-level layer.** Dependency is one-way: `discretization`/`flow` → `properties` → `mesh`
  (`CellZones` only). Numerics never own properties; `properties` never imports numerics.
- **One property per object, collected by `PropertyModel`** (decided with the author). Operators name the
  property they read (`DiffusionFlux(coefficient="conductivity")`), so more physics adds model entries, not
  fields to a context bundle.
- **Values are plain scalars, not forced JAX arrays.** `Constant.value: float`; differentiate by
  constructing the property inside the differentiated function with a traced value (as
  `boundary.Convective(h=…)` does). `ZoneConstant.values` is an array because the label gather needs one.
  - ⚠️ **A float value is a static `eqx.filter_jit` leaf compared by value**, so every rescaled float is a
    fresh compile key for anything taking the owning assembler. Reynolds continuation rescales viscosity
    per rung, so a float viscosity recompiles the coupled solve each rung. The library does not force
    arrays; the **case opts in** (`validation/bfs3d_openfoam/compare.py` uses
    `Constant(jnp.asarray(RHO * NU))`; density stays a float since nothing rescales it). Both halves are
    pinned by `test_properties.py::test_scaling_an_array_valued_constant_shares_compiled_code_and_a_float_does_not`.
    Promoting to an array converter would reverse the decision above — measure first.
- **Scalar per-cell only.** `(n_cells, dim, dim)` anisotropy is left open, not built.
- **`ZoneConstant.from_dict(cell_zones, {zone: value})`** keys to existing zone labels; a populated zone
  left unspecified is a `ValueError`, an empty zone may be omitted.
- **Evaluate per residual**; no caching unless a profile demands it (state-independent values constant-fold).
- **Uniform rescale is one primitive: `Property.scaled(factor)`** (delegated per kind) and
  `PropertyModel.with_scaled(name, factor)` (one `require`-checked entry). Built for Reynolds continuation;
  `factor` may be traced.

## Wired in
`aquaflux.context.MeshContext` (`context.mesh`) carries the evaluated `properties` map.
`ResidualAssembler` takes a `PropertyModel` (+ a `coefficient` name for Robin/Neumann `Gamma`) and evaluates
it each residual. `MomentumContinuity` takes one and exposes per-cell `.density` and `.effective_viscosity`
(molecular `μ` + `ρν_t`).
**Density interpolation (binding):** the Rhie–Chow mass flux interpolates `ρu` together (`interp(ρu)·n`,
not `interp(ρ)·interp(u)`); the pressure-correction / Schur term uses `interp(ρ)`. Variable-density
Rhie–Chow is not validated.

## Not yet built
- **`Calculated`** — a property computed from state fields (`field_names`, differentiable `params`,
  `formula`) evaluated inside the residual so AD carries its state-dependence. The plumbing exists:
  `ResidualAssembler.residual`/`.gradient` take `fields: Mapping[str, Array]` and forward it verbatim to
  `PropertyModel.evaluate`. The map is **caller-supplied** (the assembler does not insert its own `phi`
  under a guessed name). Missing: the property itself and the cross-equation convention for building the map.

## Testability seam
`tests/unit/test_properties.py` tests each kind with a hand-built `CellZones` and no geometry or solve.
