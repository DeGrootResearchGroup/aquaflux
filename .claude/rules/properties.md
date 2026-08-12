---
paths:
  - "aquaflux/properties/**"
---

# Rules — `aquaflux/properties/` (physical property model)

> **Provenance boundary (binding).** This file cites the C++/Fortran precursors to inform
> *your* understanding — that is its job, and why it loads into your
> context. Per the root `CLAUDE.md` **Comment Convention**, none of that provenance may
> reach the shipped surface (`.py` comments/docstrings, `docs/`): cite the *math*, never the
> reference code, the `.claude/` rules, the design notes, or the author's own papers.

Physical properties (density, viscosity, conductivity, diffusivity, ...) as per-cell fields,
**decoupled from the numerics**. The C++ precursor put a "transport coefficient" object *inside*
discretization — the wrong layer. Properties are a physical-model concern operators *consume*; this
module owns them. Governed by the root `CLAUDE.md` Engineering Principles.

## Module layout
- `property.py` — **`Property`** (interface, `evaluate(cell_zones, fields) -> (n_cells,)`) →
  `Constant`, `ZoneConstant`, `FieldProperty`. Each is a **single** property (density is one object,
  viscosity another); the *kind* (constant / zonal / per-cell field / calculated) is orthogonal to
  *which* property it is. `FieldProperty(values)` holds an arbitrary per-cell array (the point
  between uniform `Constant` and per-zone `ZoneConstant`) — the carrier for an externally computed
  or frozen-from-a-previous-solve field (e.g. an effective viscosity); `values` is a differentiable
  leaf and `evaluate` fail-fast checks its length against the cell count.
- `model.py` — **`PropertyModel`** (`properties: dict[str, Property]`): the named collection
  fully describing what the active physics models need. `evaluate(cell_zones, fields=None)` →
  `{name: (n_cells,) array}`; `require(*names)` fail-fast checks an operator's named coefficient is
  supplied.

## Binding decisions
- **Placement is a separate top-level layer** (peer of mesh/discretization/schemes/flow). One-way
  dependency `discretization`/`flow` → `properties` → `mesh` (needs only `CellZones`). Never let the
  numerics own properties, and never let `properties` import the numerics.
- **One property per object, collected by `PropertyModel`** (decided with the author). Operators
  **name** the property they consume (`DiffusionFlux(coefficient="conductivity")` — Stage 2), so
  multi-species / multi-physics adds *entries to the model*, not *fields to a bundle* — this is what
  retires the `FaceContext.gamma` union-bundle-regrows finding.
- **Values are plain scalars, not forced JAX arrays.** `Constant.value: float`; differentiate by
  passing the value as a traced argument (constructing the property inside the differentiated
  function) — the same pattern `boundary.Convective(h=…)` uses. `ZoneConstant.values` is the one
  array (an indexed collection of per-zone scalars — the label gather needs it). Do **not** wrap
  constants in `jnp.asarray` "for differentiability"; it is unnecessary.
  - **⚠️ BUT A FLOAT VALUE IS A *STATIC* JIT LEAF, AND THAT COSTS A FULL RECOMPILE PER REYNOLDS RUNG
    (measured 2026-08-12).** The decision above is about differentiability and stands on that ground.
    It has a second consequence nobody had looked at: a Python float is not a JAX array, so
    `eqx.filter_jit` puts it on the **static** side and compares it **by value** — and every function
    taking the owning assembler as an argument is then keyed on it. `Property.scaled` exists for
    Reynolds continuation, whose every rung rescales exactly this value, so on `bfs3d` each rung was a
    fresh key for `_march_step`, i.e. a full compilation of the coupled solve. `_jacobian_matvec`'s
    docstring claims a rung is a cache hit because "a scaled viscosity changes only two leaf values";
    that is true **only if those leaves are arrays**. Verified by trace count: with a float viscosity a
    companion assembler retraces, with `Constant(jnp.asarray(...))` it does not.
  - **The library still does not force it; the CASE opts in.** `validation/bfs3d_openfoam/compare.py`
    builds `Constant(jnp.asarray(RHO * NU))` for viscosity (density stays a float — nothing rescales
    it, so its static value is the same every rung). `Constant.scaled`'s docstring carries the rule, and
    `test_properties.py::test_scaling_an_array_valued_constant_shares_compiled_code_and_a_float_does_not`
    pins **both** halves, so the float behaviour cannot be "fixed" by accident either. If a second
    continuation case appears and this keeps catching people, promoting it to a converter on the field
    is the change to make — but that would reverse the binding decision above, so measure first.
- **Scalar per-cell only for now.** `evaluate` returns `(n_cells,)`. The contract is left open for a
  future `(n_cells, dim, dim)` anisotropic case — not built.
- **`ZoneConstant` reuses `CellZones`.** `from_dict(cell_zones, {zone: value})` keys values to the
  existing zone labels (`values[cell_zones.label]`). A populated zone left unspecified is a
  `ValueError`; an empty zone (e.g. an empty `"default"`) may be omitted (its slot is never
  gathered).
- **Static caching is acceptable but not required.** Evaluate per residual (state-independent
  properties are cheap and XLA constant-folds them); optimize only if a profile demands it.
- **Uniform rescale is one primitive, `Property.scaled(factor)` (+ `PropertyModel.with_scaled(name,
  factor)`).** `scaled` returns a copy with every value multiplied by `factor`, delegated to each kind
  (`Constant.value`, `ZoneConstant.values`, `FieldProperty.values`), so a caller can rescale a material
  coefficient without knowing its representation; `with_scaled` applies it to one named entry
  (`require`-checked) and leaves the rest untouched. Built for Reynolds-number continuation — raising the
  molecular viscosity of a lower-Re companion problem — but it is a generic property operation. `factor`
  is a plain multiplier, so a tracer flows through it under differentiation, the same as a property value.
  Unit-tested per kind in `test_properties.py`.

## Wired in (Stage 2 — DONE)
`FaceContext` carries the evaluated `properties: {name: (n_cells,) array}` map (not a per-coefficient
field, so it never grows with more properties). `DiffusionFlux(coefficient=…)` names the property it
reads. `ResidualAssembler` takes a `PropertyModel` (+ a `coefficient` name for the Robin/Neumann BC
`Gamma`) and evaluates it each residual. `MomentumContinuity` takes a `PropertyModel` and exposes
`.viscosity` / `.density` (per-cell); **`rho` is now per-cell**. Retired review findings D1/F1/F5.
**Density interpolation (binding):** the Rhie–Chow mass flux is `ρu`, so density interpolates *with*
the velocity as the momentum `interp(ρu)·n` (not `interp(ρ)·interp(u)`); the pressure-correction /
Schur term uses the face density `interp(ρ)`. Constant-density is bit-for-bit unchanged; genuine
variable-density Rhie–Chow physics is not yet validated.

## Not yet built (staged)
- **Stage 3 — `Calculated`.** A value computed from state fields via a formula (temperature-dependent
  viscosity, ...): `field_names` (static) + differentiable `params` + `formula(params, *fields)`,
  evaluated inside the residual so AD carries its state-dependence into the Jacobian (the limiter
  `psi(phi)` pattern). Needs the assembler to expose its state as **named fields** — also the bridge
  to the eventual DSL's named fields.

## Testability seam
Each property/kind is unit-tested with a hand-built `CellZones` and no mesh geometry, no solve
(`tests/unit/test_properties.py`): `Constant` broadcasts and is grad-able in its value; `ZoneConstant`
maps each zone to its value, is grad-able per zone, and rejects unknown / missing populated zones;
`PropertyModel` evaluates every named property and `require` flags a missing one.
