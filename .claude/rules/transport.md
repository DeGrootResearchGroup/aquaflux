---
paths:
  - "aquaflux/transport/**"
---

# Rules — `aquaflux/transport/` (scalar transport by a converged flow)

> **Provenance boundary (binding).** This file may cite the C++/Fortran precursors and the project's
> own history to inform *your* understanding. Per the root `CLAUDE.md` **Comment Convention**, none
> of that may reach the shipped surface (`.py` comments/docstrings, `docs/`): cite the *math*, never
> the reference code, the `.claude/` rules, the design notes, or the author's own papers.

The equation the project exists to answer questions about — what a tracer, a reagent or a
contaminant does in a reactor — solved on the flow the coupled block produces. This is also the
**aquakin coupling seam**: a reaction network attaches as `VolumeSource` terms.

## Status — BUILT (steady and transient single scalar)

- **`scalar.py` — `ScalarTransport`, `effective_diffusivity`, `DIFFUSIVITY`.** A configured scalar
  transport equation. `build(mesh, geometry, diffusivity, boundary, advection_scheme, *,
  gradient_scheme, sources, transient)` fixes the configuration; `residual(flux)` returns the
  residual function for a particular flow. It composes `AdvectionFlux` + `DiffusionFlux` +
  the injected `VolumeSource`s + the optional `TransientTerm` through the ordinary
  `ResidualAssembler` — it adds no numerics of its own, only the composition and the two
  conventions below.

## Binding decisions

- **The scalar advects on the flow's OWN Rhie–Chow flux, never on a rebuilt one.** `residual(flux)`
  takes it; the caller forms it as `flow.volume_flux(momentum.mass_flux(state), rho)`. Rebuilding
  `(u·n)A` from cell velocities satisfies no discrete continuity, so a uniform tracer would not stay
  uniform. `tests/support/fields.py::face_mass_flux` exists for *prescribed divergence-free*
  verification fields only and is deliberately not in the library.

- **A concentration rides the VOLUMETRIC flux, and this is not a stylistic choice.** A species
  concentration is per unit **volume** (`kg/m³`, `mol/m³`), so its balance is
  `∂C/∂t + ∇·(uC) = ∇·(D∇C) + S` — a mass balance on the species, with **no fluid density in it**,
  already in conservative form as written. `flow.volume_flux(mdot, rho)` is the one definition of the
  conversion (`.claude/rules/flow.md`); `SSTTurbulence._volume_flux` calls it too.
  - ⚠️ **Do NOT carry the k/ω "exact for constant density" caveat across to a concentration.** That
    caveat is about `k` and `ω` being per-unit-**mass** quantities: their conservative form is
    `∂(ρk)/∂t + ∇·(ρu k) = …` and dividing ρ out to reach the kinematic form is what needs ρ uniform.
    A concentration has no ρ to divide out. The two equations look alike and their density
    assumptions are not the same.
  - **What does depend on constant density, for both:** the *discrete* guarantee that a uniform field
    is preserved. Continuity closes on `Σ ṁ_f = 0`, so `Σ Q_f = 0` follows only while ρ is uniform.
    That is a note for whenever variable density lands, not a restriction on the equation.

- **`effective_diffusivity(D, ν_t, turbulent_number)` = `D + ν_t/Sc_t` — and it is deliberately NOT
  shared with the k/ω diffusivity.** Species (Schmidt) and energy (Prandtl) are genuinely the same
  relation and share this one home. The k/ω form is `ν + blend(F₁, σ₁, σ₂)·ν_t`, whose coefficient is
  an F₁-blended **model constant** of the closure, not a turbulent number. They agree only in having
  the shape `molecular + coefficient·ν_t`; unifying them would take "the coefficient multiplying
  `ν_t`" as an argument, which removes no decision from either caller and would contort the SST side.
  `turbulent_number` defaults to `0.7` — a **modelling choice**, so any result sensitive to it must
  say which value it was taken at.

- **A sub-patch injection is a `DirichletField` on the EXISTING patch, never a new patch.** An
  injector covering part of an inlet is a boundary *value* that varies with the face centroid, not a
  topology change. Splitting a patch in the mesh generator would produce a different `polyMesh` and
  invalidate every checkpoint and measurement taken on the old one — on `bfs3d` that is `state-00067`
  and effectively the whole design record.
  - ⚠️ **`DirichletField.field_fn` is a STATIC field, so the injection geometry is not a
    differentiable parameter.** Fine for a validation case; it blocks the obvious optimization demo
    ("where should the injector go to maximize mixing"), which would need that position to be a leaf.

## Testability seam

`tests/integration/test_scalar_transport.py` — order of accuracy against the 1-D
advection–diffusion exponential, discrete conservation, uniform-field preservation, boundedness
under a sub-patch injection, and the flux magnitude against the inlet's known volumetric rate.

⚠️ **The flux-magnitude test is separate from the uniform-field one on purpose, and the reason
generalizes.** A uniform field is preserved by *any* divergence-free flux, and scaling a flux by a
constant density leaves it divergence-free — so uniform-field preservation is **completely blind** to
a mis-scaled flux. Verified by mutation: feeding the mass flux where the volumetric one belongs left
the uniform result unchanged to 1e-10, while the inlet-flow-rate test fails it by 997×. Two further
consequences worth carrying:
- **Set `rho ≠ 1` in any test that means to discriminate the two fluxes.** At `rho = 1` they are
  numerically identical and every such test is vacuous. These use water, 998.
- **Mutation-check a test that claims to catch a specific error** — feed it the wrong quantity and
  confirm it fails. Both of the above were found that way, after the tests were green.

## The cross-code case — `validation/bfs3d_species` (aquaflux side BUILT)

A passive tracer on the 3D backward-facing step, compared against an OpenFOAM `scalarTransport` run.
Three things about it are worth carrying, because each was a decision rather than a detail:

- **Two arms, under DIFFERENT names, because the FLOW does not agree between the codes** (`x_r/h`
  8.36 vs 7.24). A species comparison on each code's own flow is dominated by the flow difference and
  can attribute nothing. So: a **same-flux** arm (both codes on OpenFOAM's own `phi` *and* its `nut`,
  isolating the scalar discretization) and an **own-flow** arm (each on its own, the honest end-to-end
  number). They differ *only* in the flux and `ν_t`, so their difference is the flow's contribution.
- **`Sc_t = 1` there, not the 0.7 default.** OpenFOAM's `scalarTransport` with no `D` entry uses the
  momentum transport model's `ν + ν_t`, so matching it means `turbulent_number=1.0`. Every number the
  case reports is a number at `Sc_t = 1` — which is exactly the "say which value it was taken at"
  obligation above, discharged.
- **The injector has ONE definition** (`injector.injected_value`), from which the OpenFOAM case's
  inlet values are *generated* as an explicit `nonuniform List<scalar>`. Two implementations of one
  profile would drift, and a drifted injector is indistinguishable from a transport discrepancy. Its
  edges are tapered rather than sharp: at a jump the leading cross-code difference is limiter
  tie-breaking at one cell, not the transport being measured.

⚠️ **A steady tracer at this Reynolds number NEEDS a preconditioner** — measured, not anticipated. An
unpreconditioned solve does not converge: away from the shear layer `ν_t` falls to `~4e-11`, so `Γ` is
essentially molecular and the cell Péclet number reaches order `10³`. The fix is to reuse
`turbulence.scalar_transport_preconditioner` (the frozen convection-diffusion V-cycle already built
for k/ω), which then converges in 10 s at 23040 cells to `|Σ R| = 8.2e-15`.
- 📌 **That function's home is a placement smell now that it has a second, non-turbulence consumer.**
  It is generic scalar transport living in `aquaflux/turbulence/preconditioner.py` because that is
  where the first caller was. Moving it to `aquaflux/transport/` would be the honest placement; it is
  a public-API change touching the flagship turbulence path, so it has NOT been done — it is recorded
  here rather than silently carried.

## Not yet built

- **Multi-species / reacting systems (the aquakin coupling).** A reaction network couples N species
  tightly, so it wants one `(n_cells × n_species)` residual rather than N scalar ones; the present
  `VolumeSource` seam carries other fields as frozen constructor state, which is a *lagged* coupling.
  Prerequisite: the flat multi-block layout needs one home — `flow/state.py::BlockStateLayout` and
  `turbulence/coupled.py::CoupledRANSLayout` are already two hand-rolled versions of it, and a
  species system would be the third.
- **Residence-time distribution.** A transient tracer pulse on a frozen steady flow. Reachable now —
  `TransientTerm` is wired — and it does **not** need transient momentum, which is a separate and
  much larger piece (it needs a transient-consistent Rhie–Chow; see `.claude/rules/flow.md`).
- **Energy.** The same equation with `Pr_t`; `effective_diffusivity` already covers the closure.
