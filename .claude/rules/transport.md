---
paths:
  - "aquaflux/transport/**"
---

# Rules — `aquaflux/transport/` (scalar transport by a converged flow)

> **Provenance boundary (binding).** This file may cite the C++/Fortran precursors and project history;
> none of that may reach the shipped surface (`.py`, `docs/`) — root `CLAUDE.md` Comment Convention.

A tracer, reagent or contaminant transported by the flow the coupled block produces. This is the
**aquakin coupling seam**: a reaction network attaches as `VolumeSource` terms.

## Status — BUILT (steady and transient single scalar)

- **`scalar.py` — `ScalarTransport`, `effective_diffusivity`, `DIFFUSIVITY`.** `build(mesh, geometry,
  diffusivity, boundary, advection_scheme, *, gradient_scheme, sources, transient)` fixes the
  configuration; `residual(flux)` returns the residual for one flow. It composes `AdvectionFlux` +
  `DiffusionFlux` + `VolumeSource`s + optional `TransientTerm` through `ResidualAssembler` and adds no
  numerics. `build` refuses multi-field closures (`refuse_a_closure_that_closes_other_fields`), so a flow
  bundle is named by patch rather than failing later.
- **`prescribed_range(boundary, geometry)` sizes a scaled advection scheme.** `build` sets
  `advection_scheme.with_reference_scale(...)` from `max − min` over every `Dirichlet`/`DirichletField`
  value (the latter read at face centroids), falling back to the largest magnitude when there is one level;
  nothing prescribed → a scaled scheme is refused. ⚠️ A temperature prescribed at one level gets its
  absolute level (e.g. 300 K) as the scale, which is wrong for the variation — such a case should state the
  limiter's scale.
- **`initialization.py` — `hybrid_initialize(transport)`**: the harmonic interpolant (`flow.laplace_field`)
  of the boundary values, reconstructed with the scalar's own `gradient_scheme`, so it stays in the
  prescribed range. With no `Dirichlet`/`DirichletField` patch it starts at **zero** rather than through a
  singular pure-Neumann solve. A transport solve on the potential-flow flux would be a better start; not built.

## Binding decisions

- **The scalar advects on the flow's own Rhie–Chow flux, never a rebuilt one.** Callers pass
  `flow.volume_flux(momentum.mass_flux(state), rho)`. `(u·n)A` from cell velocities satisfies no discrete
  continuity, so a uniform tracer would not stay uniform. `tests/support/fields.py::face_mass_flux` is for
  prescribed divergence-free verification fields only and stays out of the library.
- **A concentration rides the volumetric flux.** It is per unit volume, so its balance
  `∂C/∂t + ∇·(uC) = ∇·(D∇C) + S` has no fluid density. `flow.volume_flux` is the one conversion (also used
  by `SSTTurbulence._volume_flux`).
  - ⚠️ Do not carry the k/ω "exact for constant density" caveat across: k and ω are per unit mass, so
    their conservative form carries ρ; a concentration has none to divide out. What does need uniform ρ,
    for both, is the discrete guarantee that a uniform field is preserved (`Σ ṁ_f = 0` ⇒ `Σ Q_f = 0`).
- **`effective_diffusivity(D, ν_t, turbulent_number)` = `D + ν_t/Sc_t`, deliberately not shared with the
  k/ω diffusivity.** Species (Schmidt) and energy (Prandtl) share it; the k/ω coefficient is an F₁-blended
  closure constant, and unifying would remove no decision from either caller. `turbulent_number` defaults
  to `0.7`, a modelling choice — a result sensitive to it must state the value.
- **A sub-patch injection is a `DirichletField` on the existing patch, never a new patch.** Splitting a
  patch changes the `polyMesh` and invalidates every checkpoint and measurement on the old one.
  - ⚠️ The injection geometry is differentiable only if the injector is an `equinox.Module` (its array
    fields are leaves). `validation/bfs3d_species`'s `injected_value` is a plain function, accepted but
    static, so its geometry is not yet a parameter.

## Testing
`tests/integration/test_scalar_transport.py`: order of accuracy (1-D advection–diffusion exponential),
conservation, uniform-field preservation, boundedness under sub-patch injection, flux magnitude vs the
inlet's volumetric rate.
⚠️ Uniform-field preservation is blind to a mis-scaled flux (any divergence-free flux preserves it, and
ρ-scaling keeps it divergence-free) — that is why the flux-magnitude test exists. **Use `rho ≠ 1`** in any
test meant to tell mass and volume flux apart; at `rho = 1` they are identical. These tests use 998.

## The cross-code case — `validation/bfs3d_species`
Passive tracer on the 3D backward-facing step vs OpenFOAM `scalarTransport`.
- **Two arms under different names**, because the flow differs between codes (`x_r/h` 8.36 vs 7.24): a
  **same-flux** arm (both on OpenFOAM's `phi` and `nut`, isolating the scalar discretization) and an
  **own-flow** arm (end to end). Their difference is the flow's contribution.
- **`Sc_t = 1`** (`turbulent_number=1.0`) to match `scalarTransport`'s `ν + ν_t` with no `D` entry; every
  number the case reports is at `Sc_t = 1`.
- **One injector definition** (`injector.injected_value`); the OpenFOAM inlet values are generated from it
  as a `nonuniform List<scalar>`. Edges are tapered so the comparison is not dominated by limiter
  tie-breaking at a jump.
- ⚠️ **A steady tracer here needs a preconditioner**: away from the shear layer `ν_t ~ 4e-11`, cell Péclet
  ~10³, and an unpreconditioned solve does not converge. It reuses
  `turbulence.scalar_transport_preconditioner` (frozen convection-diffusion V-cycle).
  - 📌 Placement smell, recorded not fixed: that generic function lives in
    `aquaflux/turbulence/preconditioner.py` because k/ω was its first caller; moving it to `transport/` is
    a public-API change on the turbulence path and has not been done.

## Not yet built
- **Multi-species / reacting systems (aquakin).** Wants one `(n_cells × n_species)` residual; the present
  `VolumeSource` seam carries other fields as frozen constructor state (a lagged coupling). Use
  `solve/state.py::FieldLayout` (e.g. `CellFields("species", n_species)`) for packing, block slices and
  block-wise measures — do not add another layout.
- **Residence-time distribution** — a transient pulse on a frozen steady flow; reachable now
  (`TransientTerm` is wired) and needs no transient momentum.
- **Energy** — the same equation with `Pr_t`; `effective_diffusivity` already covers it.
