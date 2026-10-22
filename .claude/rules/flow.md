---
paths:
  - "aquaflux/flow/**"
---

# Rules — `aquaflux/flow/` (coupled pressure–velocity)

The coupled p–U block — the project's central bet. Solved **monolithically** `(u, v[, w], p)` per cell,
not segregated SIMPLE/PISO; the whole Jacobian comes from AD. This file may cite the C++/Fortran
precursors for *your* understanding; none of that provenance may reach `.py` or `docs/` (root
`CLAUDE.md` Comment Convention). Dated measurements and investigation records: `.claude/notes/flow-log.md`.

**Names that no longer exist** (grep lands here): `coupled_continuation`, `coupled_lu_continuation`,
`coupled_amg_continuation`, `lu_beta_tracking_refresh`, `amg_beta_tracking_refresh` → one builder
`coupled_step` given a preconditioner value (`BlockDiagonal` or `MaterializedJacobian(FieldSplit |
BlockInverse)`), sessions via `open_session` (rename table in `turbulence.md`). `BlockPreconditioner.build
(velocity="smoothed"/"convection"/"convection-air")` → values `ViscousMultilevel()` / `ConvectionTwoLevel()`
/ `ConvectionAir()` (a string is refused; `VelocityBlock` and `_ConvectionVelocityBlock` are abstract and
`SettingsValue.__new__` refuses them). `MomentumContinuity.viscosity` → `effective_viscosity` (returns
`μ + ρν_t`; bare `viscosity` means the molecular property). The mass-flow drive's solved force is `f` in
code (`MassFlow.forced(momentum, force)`); older entries in the log writing it `β` mean `f`, since `β` is
the pseudo-transient shift strength. `pressure_pin=` / `pressure_pin_value=` → `pressure_datum=`. The
`body_force` leaf → a `UniformBodyForce` source or a `MassFlow` drive. There is no `MASS_FLOW_CONVERGENCE`,
`solve_coupled_mass_flow`, `mass_flow_coupled_continuation`, `simple_preconditioner`,
`viscous_face_coefficient`, `convection_diffusion_operator`, `BlockStateLayout`, `inner=` selector,
`AggregationSchur`, `DampedJacobiSchur` or `DiagonalVelocity`.

## Status — BUILT (steady laminar, Poiseuille-validated)

- **`FlowBoundary.closes() -> FLOW_FIELDS = ("velocity", "pressure", "mdot")`.** A flow condition is a
  bundle of three closures; `MomentumContinuity.build` refuses a collection declaring anything else,
  naming the patch (mechanism in `boundary.md`).
- **`FlowBoundary.shears_flow()` is this package's `is_wall` predicate — do not add a second.** Three
  consumers, one inside the residual: the wetted area for `scales.hydraulic_length`; the wall-function
  viscosity override the momentum assembler applies on exactly these faces (solution-affecting); and
  `CoupledRANS.build`'s wall reconciliation.
- **`datum.py` — `PressureDatum` → `PinnedPoint(point, value=0.0)`.** `MomentumContinuity.build
  (pressure_datum=...)`. The datum is a place: the cell whose centroid is nearest `point` (ties to the
  lowest index), resolved once at build into the static `pressure_pin` field (which the Schur geometry,
  SIMPLE kernels and `potential_flow` read). A point survives `permute_cells` and a remesh; an index does
  not (`test_a_point_pins_the_same_physical_cell_however_the_cells_are_numbered`).
  - Whether a datum is needed is derived and enforced both ways by
    `refuse_an_unsuitable_pressure_datum(boundary, datum, caller)`: no closure `prescribes_pressure()`
    (`True` for `PressureOutlet` alone) ⇒ a datum is required (else singular); some does ⇒ refused (it
    over-determines the level). The case file asks the same function.
  - No `PinnedPatch`/mean-pressure datum (a patch mean is a row coupling every patch face, which the
    coloured probe and the Schur preconditioner do not handle) and no `PinnedCell` (a point does it and
    survives renumbering).
- **`drive.py` — `Drive` → `BoundaryDriven` / `MassFlow`.** `MomentumContinuity.drive` says what forces
  momentum and what that makes of the state. A *prescribed* uniform force is a `UniformBodyForce` source,
  not a drive; the two drives are "unknowns are the fields" vs "fields plus a multiplier". `Drive` is
  abstract in all seven members on purpose (incl. `held_speed`, the speed a drive holds, and
  `held_velocity(dim)`, the plug a starting state takes so `body_force_velocity` starts a held bulk
  velocity at its target) — a default would let a bordered drive inherit an answer that drops its multiplier.
  - `MassFlow` owns everything the bordered form needs of the drive: `layout` / `join` / `split`,
    `driven_state` / `fields_state` / `settled`, `constraint_vectors` / `constraint_vectors_in`, `forced`,
    `bulk_velocity`. The border is attached and detached through the declared layout
    (`fields.appended(GlobalDofs("body_force", 1))`, read with `slice_of`) — never by `jnp.append` / `[:-1]`.
  - Entry points read the drive off the **assembler**, never take a target (a target named twice lets the
    constraint and the applied force disagree). `mass_flow_drive(momentum, caller)` refuses an assembler
    without one — raise it early, because otherwise the failure is an `equinox.tree_at` path error inside a
    traced step.
  - ⚠️ The refusal runs both ways: `refuse_a_constraint_this_solve_cannot_hold` is called by a solve that
    marches the fields alone (now only `reused_flow_solve`). A `MassFlow` force is a valid force, so such a
    solve would hold it fixed at the seed, converge, and silently drop the constraint.
  - `MomentumContinuity.uniform_body_force()` is the one home for "the sum of uniform forces";
    `Drive.volumetric_force` returns `None` for "no force at all", not a zero vector (a mass-flow solve is
    force-driven even when its multiplier is zero, i.e. at every first step). Do not infer "body-force
    driven" from a value.
- **`momentum.py` — `MomentumContinuity`.** Residual over the flat state `[vel_0..vel_{dim-1}, pressure]`.
  The layout is `state.py::flow_state_layout(dim, n_cells)`, which only *names* blocks over
  `solve/state.py::FieldLayout` (owner of all `f * n_cells + i` arithmetic). `MomentumContinuity.layout` is
  public because the coupled RANS state nests it as its `flow` block; `pack`/`unpack`/`initial_state`
  delegate to it. Per component the momentum balance is
  `CellBalance((DiffusionFlux, PressureForce, AdvectionFlux)).residual(component, context)` over one
  shared `MeshContext` and a per-component `FieldContext`. Continuity is `Σ mdot_f = 0` via
  `mesh.face_cells.scatter_conservative(mdot)` (there is no `_scatter`).
  - **`PressureForce`** (exported) is the one flow-specific operator: `p_f n_i A`, face pressure as
    constructor state, ignoring the transported field. AD still recovers `dR_momentum/dp` because the face
    pressure is a differentiable function of the pressure unknowns.
  - ⚠️ **The operator order `(diffusion, pressure, advection)` is load-bearing arithmetic.** `CellBalance`
    sums in tuple order; reordering perturbs the last bits, and `bfs3d` march trajectories are compared
    bit-for-bit (`march_log_compare.py`).
  - **Reuse, not reimplementation:** momentum viscous/advection are the existing operators with μ as the
    diffusion coefficient and `mdot` as the mass flux; do not re-derive a momentum flux.
- **`source.py` — `MomentumSource` / `UniformBodyForce`.** Vector volume-integrated force per cell,
  production positive, subtracted from the balance; injected via `build(sources=(...))` and applied at
  the **vector** level in `_momentum_residual` (a momentum source couples components, e.g. `−2ρΩ×u`, so
  it cannot ride a `CellBalance`'s scalar `source_operators`).
  - `requires()` / `uses_gradient()` come from `discretization.DeclaredInputs`; `build` unions
    `declared_properties` with `viscosity`/`density` into one `properties.require(...)`, so a mistyped
    property is a build-time `ValueError`. `uses_gradient()` is declared but not enforced (the flow always
    reconstructs).
  - It takes the `VelocityFields` bundle, `geometry` and evaluated `properties` — **not** a `FieldContext`
    (that carries one scalar component's data).
  - **`source`, `face_force`, `diagonal` are all abstract on purpose** — a silent "no face treatment" or
    "no diagonal" default is wrong exactly where it matters.
  - **`diagonal(cell velocity, geometry, properties)` is wired** into `momentum_matrix_diagonal` (both
    residual and frozen paths, via `_source_diagonal`) and the dissipative bucket of
    `momentum_matrix_diagonal_parts`. `a_P` is not preconditioner-only (it sets the differentiated
    Rhie–Chow damping `V/a_P`), so an unused or wrong `diagonal` changes the converged answer and the
    adjoint. The total is added **unfloored** (operator coefficient); the bucket contribution is
    **component-averaged and floored at 0** (a forward-path stabilization scale). It takes the cell
    velocity only (a diagonal is `∂row/∂own unknown`, and the frozen/shift sites hold a bare velocity).
    Pin any non-trivial `diagonal` against AD of the source's own `source`, with a control showing the
    sourceless `a_P` fails by exactly the source term (`test_momentum_source.py`). The velocity AMG needs
    nothing: its symmetric rescaling to the current `a_P` reproduces a diagonal source exactly.
  - **`face_force` is declared, not implemented**: only `None` (uniform) is supported, and a non-`None`
    return is refused at build by `source.reject_unsupported_face_force` (the mass flux carries only the
    pressure gradient, so it would be silently dropped). See "Not yet built" (buoyancy).
  - A *solved* force cannot be a source: the bordered solve adds `(f − f₀)a` with the analytic
    `a = dR/df = −V`, valid only for a uniform, state-independent force.
- **One shared assembly per state — `flow_fields` / `residual_from_fields` (binding).** `flow_fields(state)
  -> FlowFields` assembles boundary fields, both gradients, the lagged `a_P` and the Rhie–Chow flux once;
  `residual` and `mass_flux` read from it. A caller needing several of these at one state calls
  `flow_fields` once — separate accessors re-assemble the Rhie–Chow flux each time. `velocity_fields` is
  deliberately the lightweight path (boundary velocity + `_velocity_gradient`, no `a_P`/`mdot`): the eddy
  viscosity is needed before `mdot` exists.
  - **The fields-based forms are the implementation; the flat ones delegate (2026-10-20).**
    `flow_fields_of(velocity, pressure)` and `velocity_fields_of(velocity)` do the work, `flow_fields(state)`
    / `velocity_fields(state)` unpack and call them. `unpack_with_field(state, field, values)` /
    `residual_with_field(state, field, values)` replace one per-cell field (layout order `[u.., p]`) and
    assemble from the separate arrays, so `jax.jvp` in that field never forms the others' tangents — the
    coloured probe's per-field programs (`solve-direct-preconditioners.md`; `_FlowProblem.probes_by_field`).
    ⚠️ **Never repack fields into a flat state on that path** (`layout.pack`, `concatenate`, `.at[].set` on
    the flat vector): the repack's derivative hands every field an explicit zero tangent and the pruning is
    gone, silently. Pinned by `tests/unit/test_fieldwise_probe.py`.
  - **`VelocityFields` (`velocity`, `boundary_velocity`, `gradient`) is nested inside `FlowFields`**, not a
    second view; it is the whole of what the turbulence closure reads (`SSTTurbulence.closure_fields`).
- **Properties and boundary.** `build(mesh, geom, properties, boundary, *, gradient_scheme=…)`;
  `properties` is a `PropertyModel` supplying `"viscosity"` + `"density"` (`properties.md`); `boundary` a
  `BoundaryConditions({name: FlowBoundary})` bound in `build` via `boundary.resolve(mesh.face_patches,
  mesh.face_cells)` (every boundary face must be covered) to one `boundary` field — no loose tuples or bare
  dict (`boundary.md`). `rho` is per cell; the Rhie–Chow flux interpolates `interp(ρu)·n` and the
  correction/Schur term uses `interp(ρ)`. Variable-density physics is not validated.
- ⚠️ **The gradient scheme is bound once per solved field.** `build` carries `velocity_gradient_schemes`
  (one per component) and `pressure_gradient_scheme`, each `gradient_scheme.bind(mesh, geometry,
  linearization)` against that field's `BoundaryLinearization` (from `_build_time_velocity_linearizations`
  / `_build_time_pressure_linearization`); `_velocity_gradient` / `_pressure_gradient` apply those. One
  binding cannot serve the flow state: a `PressureOutlet` prescribes pressure and extrapolates velocity, a
  wall the reverse, so the weights are nonzero on disjoint patches (single-binding error ~4e-3 vs ~4e-15
  per field on a perturbed 8×8 grid; numbers and configuration in the log). Pins:
  `test_the_flow_assembler_binds_a_scheme_per_solved_field`,
  `test_the_velocity_and_pressure_linearizations_differ_on_the_same_patch`,
  `test_the_flow_is_exact_on_an_inlet_outlet_wall_duct`.
  - The weights are read by `jax.jvp` at rest, exact because every flow closure is affine in the gradient
    (`test_the_flow_boundary_linearizations_do_not_depend_on_the_state`). `build` is therefore two-phase:
    construct with geometry-only bindings, then `dataclasses.replace` once the closures exist.
  - ⚠️ `momentum.gradient_scheme` survives as the **condition-free** form, for an initializer solving a
    different equation on this mesh to re-bind (`potential_flow` → `laplace_field` →
    `ResidualAssembler.build`). It is not what the residual applies.
- **`rhie_chow.py` — `interior_mass_flux` + `momentum_diagonal`.** `mdot_f = ρ(u_ip·n − d̂[(p_N−p_P) −
  ∇̄p·d]/(d·n))A`. Skewness-consistent: `ρu` is reconstructed to the face integration point via
  `interpolate_to_face`, and the damping compares compact and reconstructed jumps along `d`, so they cancel
  on a linear pressure (`tests/integration/test_skewed_flow.py`). `momentum_diagonal` returns per-component
  `a_P` `(n_cells, dim)` (isotropic today; the seam for directional sources); the Rhie–Chow coefficient is
  `d̂ = Σ_i n_i²(V/a_P_i)_f`. Keeps `g`-weighting. The block preconditioner uses the component-averaged
  `a_P`. The momentum pressure force (`_face_pressure`) is reconstructed to the integration point too.
  - `_lagged_mdot_estimate` is velocity-only (no pressure), which breaks the `a_P` ↔ `mdot` circularity.
    ⚠️ `a_P` does **not** widen the coupled Jacobian's velocity stencil; the eddy viscosity's strain-rate
    dependence does (`turbulence.md`; harness `validation/bfs3d_openfoam/column_reach_probe.py`; log).
  - **Boundary faces respect the BC type** (`boundary_owner_coeff`, from each patch's
    `momentum_diagonal_coefficient`): a zero-gradient outlet adds no viscous diagonal, a wall no convective
    one. Applied to the **residual** `a_P` only (`momentum_matrix_diagonal`, `boundary_corrected=True`).
  - **The residual's viscous boundary term uses the operator's per-face boundary viscosity**
    (`_boundary_face_viscosity()` → `boundary_momentum_diagonal`), the same coefficient `DiffusionFlux`
    carries — on wall-function meshes the wall model's `μ + ρν_t,wall` differs from the owner's log-layer
    `μ_eff` by a large factor. Pinned by a wall-function `a_P` vs `diag(jacfwd(residual))` test
    (`test_preconditioner.py`).
  - ⚠️ The **frozen** diagonal (`frozen_momentum_diagonal`, `boundary_corrected=False`) keeps the plain
    all-faces form: it is a forward-path stabilization scale, its extra boundary damping carries the
    high-Reynolds pseudo-transient march (correcting it regressed `test_channel_high_reynolds`), and it
    never affects the converged residual or adjoint. Do not "correct" it.
- **`a_P`'s viscous term is the diffusion operator's own diagonal (binding).** `momentum_diagonal` /
  `momentum_diagonal_parts` use `discretization.flux_continuous_conductance(μ_eff, geometry, face_cells)`
  (`Γ_P A / ((D_P·n) − (Γ_P/Γ_N)(D_N·n))`; harmonic mean on an orthogonal face). A g-weighted arithmetic
  face viscosity agrees only for constant `μ`, which is why constant-`μ` tests cannot catch a wrong form.
  The same helper feeds the frozen velocity AMG and the scalar shift/AMG (`turbulence/preconditioner.py`).
  Pinned: `momentum_matrix_diagonal` vs `diag(jacfwd(residual))` at graded `ν_t` (`test_preconditioner.py`).
- **The velocity-block AMG's frozen operator comes from the same diagonal definitions — no numpy
  re-derivation (binding).** Viscous coupling from `flux_continuous_conductance`; boundary diagonal is the
  boundary-face owner coefficient scattered by `face_cells.scatter`, so the assembled diagonal *is* the
  frozen `a_P` by construction. It stays on the uncorrected all-faces form, matching
  `frozen_momentum_diagonal` (what the runtime rescaling divides by).
- **`boundary.py` — `FlowBoundary` → `NoSlipWall`, `MovingWall`, `VelocityInlet`, `PressureOutlet`.** Each
  closes velocity, pressure and boundary `mdot`, and exposes `pressure_schur_coefficient`
  (`d(mdot)/d(p_owner)`, nonzero only at `PressureOutlet`) and `momentum_diagonal_coefficient`.
  - **Compose, do not inherit (binding).** `FlowBoundary` is not a `BoundaryCondition` subtype (three
    coupled quantities; `mdot` has no scalar analogue). `velocity_face`/`pressure_face` delegate to scalar
    `Dirichlet`/`DirichletField`/`ZeroGradient` closures per component (`_prescribed_components`,
    `_face_value`, `_velocity_face`); only `mass_flux` is flow-specific. A constant velocity is a floating
    `(dim,)` array leaf and a profile an `equinox.Module` (read via `_Component`, not a lambda), so the
    velocity is differentiable (`boundary.md`).
  - **Two-pass boundary reconstruction is `schemes.BoundaryClosure.reconstruct`**, the same object
    `ResidualAssembler._gradient` composes: leading-order boundary values (closures at zero gradient) feed
    the reconstruction, then closures are re-evaluated at the reconstructed gradient for the fluxes.
    `_pressure_closure()` wraps `_boundary_pressure`; `_velocity_closure(velocity, i)` is the per-component
    view, exact only because every current flow closure closes each component on its own. Face values the
    fluxes read are still taken from the vector closures at the whole tensor, so a future component-coupling
    closure (slip/symmetry) needs only its reconstruction hooks revisited. Nothing moves on an orthogonal mesh.
  - **`mass_flux`'s through-flow term is the patch's own boundary velocity, for every patch (binding).**
    `_boundary_mass_flux` passes each patch its `boundary_velocity` slice (the value just computed at the
    reconstructed gradient). `PressureOutlet.mass_flux` reads it, which makes the open Couette case exact
    on a skewed mesh (`test_skewed_flow.py`; numbers in the log). `pressure_schur_coefficient` is unaffected.
  - `FlowBoundary.mass_flux` takes nine positional arrays — past the ~5-array trigger; bundling them is
    part of the buoyancy change (below).
- **Turbulence enters through `eddy_viscosity`; `μ_eff = μ + ρν_t` is formed once, in
  `MomentumContinuity.effective_viscosity` (binding).** Kinematic `ν_t` rides its own differentiable leaf
  (`with_eddy_viscosity(nu_t)`); callers never restate the closure relation. **Do not put `ν_t` into the
  `PropertyModel`** — it is not a material property, and overwriting `"viscosity"` destroys the molecular
  value (log). `SSTTurbulence` keeps its own `density`/`molecular_viscosity` for the k/ω diffusion.
  - `with_scaled_molecular_viscosity(factor)` rescales molecular `μ` only (`PropertyModel.with_scaled`) —
    the momentum half of a Reynolds rescale (`CoupledRANS.with_scaled_molecular_viscosity` composes both).
  - The wall-function eddy viscosity is a second per-face kinematic leaf:
    `with_eddy_viscosity(nu_t, wall_eddy_viscosity=None)` (from `turbulence.wall_face_eddy_viscosity(k)`).
    `_wall_boundary_viscosity()` makes the diffusion's boundary coefficient `μ + ρ·ν_t,wall` on
    `shears_flow()` faces, owner `μ_eff` elsewhere; the residual `a_P` reads the same coefficient. `None`
    and wall-resolved meshes (`ν_t,wall = 0`) are no-ops, so it is always on.
  - The same call means different things in the two paths: `turbulence/driver.py` applies it outside the
    residual (frozen per sweep, invisible to AD); `turbulence/coupled.py` inside (cross-block Jacobian
    terms). That is why gradients go through `solve_coupled`, never `solve_segregated`. Contract pinned in
    `tests/unit/test_momentum_coupling.py` on a `ρ≠1` fluid.
- **Convection.** `advection_scheme=` turns on `mdot·u` (upwind or limited), making the residual
  nonlinear; `a_P`'s convective part uses the lagged velocity-flux estimate. Closed domains fix the
  pressure level with `pressure_datum`.
- **Body force.** `sources=(UniformBodyForce(f),)`: `R = flux − f·V` per component; with `p = p̃ + G·x`,
  `f>0` drives `+x` and `G = −f`. `force` is a differentiable leaf. Drives a streamwise-periodic channel
  (`structured_grid_2d(periodic=("x",))` + a datum), verified against Poiseuille in
  `test_periodic_channel.py`. No Rhie–Chow term, no continuity entry.
- **`drive.FlowSolve` — `(momentum, state) -> (momentum, state)`**, the segregated loop's flow-solve
  signature, exported; `bulk_velocity_flow_solve` and `reused_flow_solve` both return one.
- **Bulk-velocity constraint — `flow/mean_velocity.py`, `bulk_velocity_flow_solve`.** The force is a
  scalar Lagrange multiplier on `⟨U_dir⟩ − U_bar = 0`, appended to the state and solved jointly by the
  production `RootSolver`; AD assembles the bordered Jacobian (no hand-derived border in the residual).
  `⟨U⟩ = U_bar` holds by construction at the root, so the bulk velocity cannot overshoot while `ν_t`
  develops (the failure of the deleted proportional controller). Convergence-gated and
  reverse-differentiable; returns `(momentum, state)` carrying the converged force.
  - **Preconditioning: `precondition.bordered_preconditioner(M, border, fields, a, c)`** Schur-eliminates the
    scalar (`y = M·r_flow`, `dβ = (cᵀy − r_β)/(cᵀMa)`, `dw = y − dβ·Ma`). Hand-building `a`/`c` here is
    legitimate (a preconditioner changes only Krylov convergence). ⚠️ It is built once in the builder from a
    concrete `reference` (required whenever `preconditioner` is given), never from the traced `momentum` —
    a captured tracer breaks `jax.grad`.
  - ⚠️ **The assembler is threaded as the Newton `theta`, not captured in the residual closure** — a captured
    assembler raises a `custom_vjp` closed-over-value error. The adjoint transpose solve uses the same
    bordered preconditioner transposed (`solve/root_adjoint.py::_transposed_preconditioner`,
    `jax.linear_transpose`). Pinned by `tests/unit/test_mean_velocity.py` (incl. analytic `dβ/dμ`) and
    `test_bordered_preconditioner.py`.
  - The same primitives border every march: `flow/bordered.py` (below). Do not re-derive the border elsewhere.
  - The periodic seam is **not** a preconditioner problem — do not look there again; the cold-start crawl
    was, fixed by the velocity scale + plug start (log § "The periodic seam").
- **Field initialization — `flow/initialization.py`.** `laplace_field(mesh, geometry, boundary, …)` solves
  scalar Laplace in one multigrid-preconditioned linear step (smoothed-aggregation V-cycle; needed because
  the condition number grows like the square of the near-wall aspect ratio — Jacobi is not enough).
  `potential_flow(momentum)`: `u = ∇φ` with Neumann `u_in·n` at each `VelocityInlet`, a Dirichlet datum at
  the `PressureOutlet`, `ZeroGradient` at walls; returns `[u, p]` with the Bernoulli seed. With no
  through-flow boundary it returns the plug `scales.body_force_velocity(momentum)` (body-force channel);
  a moving-lid cavity correctly returns zero (`test_potential_flow_is_zero_on_a_closed_domain` — do not
  widen the fallback to `characteristic_velocity`, which reports the lid speed).
  - ⚠️ `potential_flow` takes **no `gradient_scheme`** and must not be given one back: it uses
    `momentum.gradient_scheme` (condition-free) so the initializer and the solve share a discretization
    (`schemes.md`).
- **Bernoulli pressure seed — `bernoulli_pressure`.** `p = ½ρ(|u_ref|² − |u|²)`, anchored so the mean over
  `PressureOutlet` cells is zero (`_pressure_outlet_cells`; else the domain mean). Halves the first coupled
  Newton correction. Closed-form on purpose: a pressure-Poisson seed reads `∇u`, which is singular at sharp
  corners (backward-facing step) and was far worse (log). Remaining first-step pressure correction is a
  solver-side concern. Pinned by `test_bernoulli_pressure_*`.
- **Velocity scale — `flow/scales.py`, the one home for "how fast is this flow".**
  `characteristic_velocity(assembler)`: fastest patch `reference_velocity`, else `body_force_velocity`.
  `body_force_speed` closes `f V_total = τ_w A_wall` via `hydraulic_length` (`V_total/A_wall`, wetted area
  from `shears_flow()`), taking **`min`** of laminar `f h²/(3μ)` and turbulent `20·u_τ` (the laminar branch
  diverges as `μ→0`). `friction_velocity = √(f h/ρ)` follows from the force balance alone; the turbulence
  initializer sizes `k`/`ω` from it. `wetted_length(mesh, geometry, patches)` is the geometric half.
  `BlockPreconditioner.build` warns when the convection block is requested with zero reference mass flux;
  a caller that knows the speed should pass `reference_state=`.
- **`reference_speed(assembler)` enters the residual.** `MomentumContinuity.build` scales momentum
  advection (`with_reference_scale`, `schemes.md`) by it, so a softened Venkatakrishnan limiter's `eps` is
  a fraction of it. Order: `Drive.held_speed()` (`MassFlow` → `|target|`, `BoundaryDriven` → `None`), else
  `|characteristic_velocity|`. Fixed at build — `with_scaled_molecular_viscosity` does not re-derive it
  (deliberate: a Reynolds ramp changes viscosity, not the limiter). An undriven domain refuses a scaled
  scheme at build. ⚠️ A flow driven only by a pressure difference between two `PressureOutlet`s is not
  covered and must state the limiter's scale. Pinned in `tests/unit/test_advection_reference_scales.py`.
- **Validated:** Poiseuille (`test_poiseuille.py`; 2nd order; Stokes in one Newton step; differentiable)
  and the Re=100 lid-driven cavity (`test_cavity.py`; ~5 Newton steps to 1e-12; Ghia et al. (1982)
  centrelines; differentiable through the IFT adjoint).
- **Structure.** The preconditioner is `block_preconditioner.py`'s `BlockPreconditioner`, composing three
  injected abstract strategy families: an `InnerSchurSolver` (`SmoothedAmgSchur`, the one pressure Schur), a
  `VelocityBlockSolver` (`SmoothedAmgVelocity` viscous / `TwoLevelConvectionVelocity` /
  `AirConvectionVelocity`), and a `SaddleComposition` (default `BlockTriangularComposition`). Build with
  `BlockPreconditioner.build(assembler, velocity=…).factory()`. Add a strategy as a subclass, never an
  `if … == …` branch. `strength_threshold=θ` (default 0, byte-identical) turns on strength-of-connection
  aggregation; the coupled RANS path uses 0.25 (`solve-amg-multigrid.md`).
  - **Dependency is one-way `block_preconditioner → momentum`**; the assembler must not import the
    preconditioner (a convenience wrapper on `MomentumContinuity` once created that cycle). Low-level
    kernels (`pressure_schur_laplacian`, `damped_jacobi_solve`) stay in `preconditioner.py`.
  - `frozen_momentum_diagonal_parts(assembler, state)` is a public function needing only the assembler,
    so a shift can be built without a preconditioner (building one per rung re-aggregates and recompiles).
  - **`BlockPreconditioner.on(assembler)`** re-points a built preconditioner at another problem, frozen
    hierarchies kept (they are rescaled to the current `a_P` per application). A viscosity ramp's station
    change uses it (`turbulence._BlockSession._repointed`); a fresh `build` re-aggregates and recompiles
    (`test_a_re_pointed_block_session_step_is_a_compilation_cache_hit`, `test_preconditioner_session.py`).
  - Frozen operators are assembled by `precondition/frozen_operator.py`'s `ConvectionDiffusionStencil(...)
    .assemble()` with `decouple_dof` for the closed-domain pin; do not re-assemble a stencil here.
  - **Symmetric rescaling and the per-component lift are single-homed (binding).** `A_cur⁻¹ ≈ D⁻¹ A_ref⁻¹
    D⁻¹`, `D = sqrt(diag_cur/diag_ref)`, lives once in `_symmetric_rescaled`; the per-component lift in
    `_per_component`. Velocity blocks inherit one `apply` from `_RescaledAmgVelocity` and supply `build` +
    `_inner_solve`. Pinned in `tests/unit/test_preconditioner.py`.
  - **Strategies build from narrow geometry bundles, not the assembler (binding).** `_SchurGeometry`
    (runtime, stored; owns `coefficient(a_P)` / `boundary_diagonal(a_P)`) and `_VelocityGeometry`
    (build-time only) are siblings, not one union bundle. Assembler behaviour (the reference `mdot`) is
    computed by the builder and handed in as a frozen array. The shared connectivity + multigrid knobs are
    one `_StrategyInputs` NamedTuple.
  - Trap: a non-h-independent inner makes the outer count grow — do not resurrect a damped-Jacobi Schur or
    `diag(a_P)⁻¹` velocity block (both deleted as dominated).

## Hybrid initialization is generic — `aquaflux.initialization.hybrid_initialize`

`hybrid_initialize(problem, **settings)` is a `functools.singledispatch` in a physics-free top-level module;
each package registers its own types (`flow/initialization.py`: `MomentumContinuity` → `potential_flow`;
`transport/initialization.py`: `ScalarTransport`; `turbulence/coupled_initialization.py`: `CoupledRANS` →
`sst_initial_fields`). A new problem type registers its own — do not add a branch to a central function.
Unregistered types raise `TypeError`. `solve_flow_march(state=None)` uses the same initializer. On the
tetrahedral duct (`validation/tetrahedral_gradient_ab`) a scalar Laplace stagnates (reconstruction
Jacobian condition ~3e19); whether laminar `potential_flow` fails there is untested.

## The flow march — `solve_flow_march` / `flow_march_step` / `FlowMeasures`

A laminar problem runs on the **same staged march as the turbulent one** (`solve.staged_march`,
`solve.shifted_step`; `solve.md`), configured the same way — a laminar case is a valid control for a solver
question.
- **`solve_flow_march(momentum, state=None, *, strategy, reference_state, preconditioner_options, …,
  **march)`** starts from `potential_flow`, marches on `stop_gradient` copies, attaches `root_adjoint` at the
  root (pinned against finite differences in `tests/integration/test_flow_march.py`). **`flow_march_step`**:
  `momentum_shift_policy` (block-SIMPLE + velocity `a_P` shift, shared with `momentum_continuation`) into
  `shifted_step`. Krylov regime `_FLOW_LINEAR_SOLVE` = rtol 0.3 / restart 120 / max_restarts 15, carried
  from the coupled block-diagonal family and **not re-measured on a flow-only residual**.
  `tools/sibling_builders.py` pairs it with `coupled_step`; differences are genuine (the coupled
  `positivity_*` are `k`'s). `shift` is `solve.ShiftSettings(basis, velocity_parts)` here and its subclass
  `CoupledShiftSettings` there; `momentum_continuation` / `momentum_shift_policy` take it too.
  `jacobian_gradient_sweeps` is on both (a residual property).
- **`FlowMeasures`** supplies `RowScaled` / `BlockScaled`: momentum rows by the shift's base diagonal,
  continuity by the cell's mass throughput, field scales the mean speed and one. `flow_row_scales` is shared
  with `turbulence.coupled_scaled_norm`, so the flow rows cannot drift.
- **`flow_row_scales` floors both scales by the characteristic speed `|U|`, always on**, so a march can start
  from rest: velocity at `REST_SPEED_FRACTION = 0.05 |U|`, throughput at `REST_THROUGHPUT_FRACTION = 1e-4
  ρ|U|A` (value chosen by the project owner for guard margin). A rest state with `|U| = 0` is refused,
  naming `Euclidean()`. Floors applied only at exact rest fail (the measure rebuilt at low speed trips
  `DivergenceGuard`); evidence in the log. Pinned by `tests/unit/test_flow_row_scales.py` and
  `test_a_lid_driven_cavity_marches_from_rest_under_the_default_measure`. ⚠️ It reads the speed as a
  concrete `float`, so it cannot run under a trace.
- ⚠️ **`BlockScaled()` stalls from the default `potential_flow` start** (potential flow is divergence-free,
  so the continuity block's scale starts ~0). It converges from rest; the default `RowScaled()` has no
  such sensitivity (24×16 channel, Re 200, 2026-09-19).
- **`MaterializedJacobian` march.** `solve_flow_march(preconditioner=MaterializedJacobian(SimpleSmoothed()))`
  or a session from `open_flow_session`. Use `SimpleSmoothed` (traced pressure–velocity hierarchy
  `precondition.block_preconditioner.MaterializedBlockPreconditioner`). `_FlowProblem` (`flow/march.py`)
  supplies the assembler, the graph probe (`_GradientSweepCap`; default stencil reach 3 — over-reach is
  exact, only costlier; the laminar reach was not measured), **no field groups** (so `FieldSplit` is refused
  and a bare block inverse allowed — the mirror of the coupled problem), the shift-only
  `momentum_shift_only_policy`, and the step. On a 384-cell channel it matched block-SIMPLE's trajectory
  (log). The step carries the inverse's traced cycle (`TracedSingleBlock` in a `TracedFormShiftPolicy`,
  `solve-field-split.md`), so a Krylov iteration makes no host round trip.
- **The tetrahedral duct**: its failure was the single-step march from rest, fixed by dual-time with a
  materialized inverse; `MultipleCorrectionGradient` does not converge there (#435). Log § "the
  tetrahedral duct re-run".

## A held bulk velocity on every march — `flow/bordered.py`

`solve_flow_march` and `turbulence.solve_coupled` take a `MassFlow`-driven assembler and border themselves;
the case file's `FlowMarch` / `CoupledMarch` hold a `BulkVelocity`.
- **`MassFlowBorder(drive, fields, force, average)`**, built by `mass_flow_border(momentum, layout)`
  (`None` when boundary-driven; `layout` the flow's own or an outer one whose first block is the flow).
  Methods: `attach` / `detach` / `force_of`, `residual(fn)` (`BorderedResidual`), `policy(inner)`
  (`BorderedShiftPolicy`), `measures(inner)` (`BorderedMeasures`), `homotopy(inner)`, `source(inner)`
  (hands sessions and caller builders field states only), `on_fields(measure)`.
- **The border wraps any residual because the force enters linearly**: `R(w; f) = R(w; f₀) + (f − f₀)a`,
  `a = −V`; it never writes the force into the assembler, so it borders a bare callable (a homotopy
  station) too. ⚠️ Rests on the force being uniform and state-independent.
- **Measures**: `BorderedMeasures.row_scaled` appends the constraint row with row scale 1 and field scale
  `drive.held_speed()` (the fractional miss; refused for a zero target), named `bordered.BULK_VELOCITY`.
  `block_scaled` gives it the leading block's scale.
- **The constraint row is unshifted, and the march starts at the target (decided 2026-10-11).**
  `BorderedShiftPolicy` gives the force a zero shift; `bordered_preconditioner` takes no border shift.
  Starting the plug at the target (`Drive.held_velocity`) is what makes this work — a start sized from the
  seed force froze the march. Shifting the row by its Schur magnitude was built, measured and rejected
  (log). At the default `beta0 = 2` a cold start crawls through viscous spin-up with or without the border,
  so the integration test runs at `beta0 = 0.1`.
- **Turbulent channels**: the coupled march holds re20000, not re240000; record in
  `.claude/notes/turbulence-log.md`, "Held bulk velocity on the law-of-the-wall files", and
  `validation/turbulent_channel/README.md`.
- Two-argument residuals for the adjoint: `BorderedFlowResidual` and
  `turbulence.coupled._BorderedCoupledResidual` read everything, border vectors included, from `theta`.
- Observers see fields: `on_checkpoint` gets the detached fields state; a restart re-solves the force from
  the drive's seed.
- **Not built: a materialized-Jacobian preconditioner beside a held bulk velocity** — `open_flow_session` /
  `turbulence.open_session` and the case's marches refuse it. Missing piece: a bordered `MaterializedProblem`.

## Binding decisions
- **`a_P` is differentiated in the residual; only the preconditioner freezes it.** Do not describe it as a
  lagged/`stop_gradient`-ed coefficient. It enters the Rhie–Chow damping `V/a_P`, so freezing it would make
  the IFT adjoint linearize a different residual (same converged value, wrong sensitivity). The block
  preconditioner `stop_gradient`s `a_P` and its state.
- **Monolithic block, AD Jacobian, Newton–Krylov** — replacing the precursor's hand-assembled 4×4-per-face
  block solved in an outer Picard loop with BiCGStab.
- **Reuse over reimplementation**: new flow code is only the pressure force, Rhie–Chow and the flow BCs.

## Not yet built (follow-ons, in order)
- **Porous / conjugate interface.** The drag half has a wired home (a `MomentumSource` returning
  `−(μ/K)u` with a matching `diagonal`); remaining are porosity in continuity/transport and the interface branch.
- **Buoyancy, and the balanced-force face treatment.** A spatially varying force must be reconstructed to
  faces consistently with the pressure gradient and enter the mass flux beside it, including the boundary
  closures — so bundling `FlowBoundary.mass_flux`'s nine arrays into a per-face-state object is part of that
  change. It is a variable-density regime, due with the conservative forms flow and k/ω defer. A species
  concentration is not implicated (only its discrete uniform-`C` preservation leans on constant density).
- **Transient momentum** (BDF, reusing `TransientTerm`). ⚠️ Must use a transient-consistent Rhie–Chow:
  `a_P^{RC} = a_P − ρV/Δt` in the mass-flux d-coefficient, or the pressure smoothing and the `-C` block
  vanish as `Δt → 0` (checkerboarding and a singular SIMPLE Schur). `momentum_diagonal(_parts)` already
  carry an unused `dt` + `rho` seam (`ρV/Δt`; `rho` required with `dt`). Then energy coupling.
- **Outer block preconditioner — Stage 2: mesh-independent AMG inner is done.** Smoothed-aggregation
  V-cycle Schur (`SmoothedAmgSchur`), built off-jit with `scipy.sparse`; ~0.25 contraction from a direct
  coarse solve, Chebyshev smoother and pin decoupling; symmetric diagonal rescaling is essential. Both blocks
  get an AMG — the velocity block, not the Schur, was the √N limit, so PCD/LSC would not help.
  Block-triangular: `δp = Ŝ⁻¹(r_p − D·δu)` with `D·δu` a jvp through a frozen (`stop_gradient`) residual.
  Outer GMRES grew ~O(N^0.25). Two cheaper diagonals were rejected (`.claude/notes/solve-log.md`, "Binding
  decisions — where preconditioning must attach (skewed cavity)"). Numbers in the log.
- **`FlowBlocks` re-runs its `jvp` per application on purpose** — `jax.linearize` per iterate buys nothing
  (harness `validation/flow_blocks_linearize.py`; log § "FlowBlocks linearize").
- **Stage 3: the remaining limit is the Schur approximation (binding, do not re-attempt).** More V-cycles,
  rebuilding at the developed state, and rescaling MSIMPLE's `k` do not help; the stabilized
  least-squares commutator Schur lost and was deleted. Validate preconditioner changes on the real march,
  not the spectral radius. Log § "Stage 3"; ledger `.claude/notes/solve-refuted-directions.md`.
- **SIMPLE-type compositions.** `schur_scaling` ∈ `{"simple", "msimple"}` picks the `InnerSchurSolver`;
  `composition` ∈ `{"triangular" (default), "simple", "simpler"}` picks the `SaddleComposition`. MSIMPLER is
  `msimple` + `simpler`; `mass_scale` is the `M` variant's `k`. MSIMPLER saves ~30–45 % of outer cycles over
  triangular but stays behind the shipped `bfs3d` leading inverse; plain SIMPLER does not converge; no
  default moved. `SimplerComposition` is the paper's Algorithm 2 at `ω_p = 1`, with `F̃` the diagonal the
  Schur was assembled from. Pinned in `tests/unit/test_preconditioner.py` by the defining property: on the
  saddle with a diagonal momentum block `simple`/`simpler` are exact inverses and `triangular` is exact in
  pressure only; plus linearity and adjoint consistency. Measurements: log § "SCOPE", "MSIMPLE(R) measurements".
- **Fully-AD `a_P`** — possible refinement, not needed.
- **Gradient-scheme cost — use `SweptGradientSolve`** (fixed matrix-free Richardson sweeps on constant
  `A_g`, an exact O(n) drop-in, ~5× cheaper than a nested GMRES) on skewed meshes (`schemes.md`,
  `solve.md`). A dense-LU variant is O(n²) and dominated; do not rebuild.

## Testability seam
`interior_mass_flux` and the BCs are pure functions of per-face arrays, unit-tested with no mesh
(`test_rhie_chow.py`, `test_flow_boundary.py`). `momentum_diagonal` is checked for positivity and
linear-in-μ scaling. The coupled solve is gated on analytical Poiseuille and its 2nd-order convergence,
plus an AD-through-solve check.
