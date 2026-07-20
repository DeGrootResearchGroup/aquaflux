---
paths:
  - "aquaflux/flow/**"
---

# Rules — `aquaflux/flow/` (coupled pressure–velocity)

> **Provenance boundary (binding).** This file cites the C++/Fortran precursors to inform
> *your* understanding — that is its job, and why it loads into your
> context. Per the root `CLAUDE.md` **Comment Convention**, none of that provenance may
> reach the shipped surface (`.py` comments/docstrings, `docs/`): cite the *math*, never the
> reference code, the `.claude/` rules, the design notes, or the author's own papers.

The coupled p–U block — the project's central bet. Solved **monolithically**
`(u, v[, w], p)` per cell, not segregated SIMPLE/PISO. Governed by the root `CLAUDE.md`
Engineering Principles.

## Status — BUILT (steady laminar, Poiseuille-validated)
- **`momentum.py` — `MomentumContinuity`.** The coupled residual over the flat state
  `[vel_0..vel_{dim-1}, pressure]` (the system-first layout; `pack`/`unpack` convert to
  `(velocity, pressure)`). Per component the momentum balance is a scalar transport of `u_i`
  — advection (`mdot·u_i`) + viscous diffusion (μ as the coefficient) + pressure force
  (`p_f n_i A`) — **reusing `AdvectionFlux` and `DiffusionFlux` verbatim**: per component it builds
  a `FaceContext` (properties `{"viscosity": μ}` via `DiffusionFlux(coefficient="viscosity")`, the
  component gradient, the boundary velocity) and calls the shared operators' `face_flux(component,
  context)`; only the pressure term is new. Continuity is `Σ mdot_f = 0`. The whole Jacobian comes
  from AD; solved by the existing `NewtonSolver` / `ImplicitNewtonSolver`.
  - **One shared assembly per state — `flow_fields` / `residual_from_fields` (binding, #106).** The
    boundary fields, both gradients, the lagged `a_P`, and the Rhie–Chow flux are assembled once by
    the public `flow_fields(state) -> FlowFields`; `residual` = `residual_from_fields(flow_fields(state))`
    and `mass_flux(state)` = `flow_fields(state).mdot`. A coupling caller that needs several of these at
    one state (a coupled RANS residual wanting the residual **and** `mdot`, a segregated sweep wanting
    the gradient **and** `mdot`) must call `flow_fields` **once** and read the fields — not call the
    accessors separately, which re-assembles the whole Rhie–Chow flux per accessor (was 3× per coupled
    residual eval; the pre-optimization HLO / AD-tape size scaled with that). **`velocity_gradient` is
    deliberately the lightweight path** (boundary velocity + the shared `_velocity_gradient` only, **no**
    `a_P`/`mdot`): the eddy viscosity a segregated sweep needs comes before the mass flux is even
    defined, so dragging the Rhie–Chow assembly through it would defeat the point. Same formula as the
    bundle (both compose `_boundary_fields` + `_velocity_gradient`), so no duplication. Pinned by
    `test_flow_fields_accessors_agree_with_the_bundle` / `test_velocity_gradient_skips_the_rhie_chow_assembly`.
  - **Fluid properties come
  from a `PropertyModel`** (`build(mesh, geom, properties, gradient_scheme, boundary, …)`, must supply
  `"viscosity"`+`"density"`; `.viscosity`/`.density` evaluate them per-cell) — see
  `.claude/rules/properties.md`. **`boundary` is a `BoundaryConditions({name: FlowBoundary})`**
  collection (constructed like a `PropertyModel`), bound inside `build` via
  `boundary.resolve(mesh.face_patches)` to a single `boundary` field (not three parallel
  `names`/`conditions`/`faces` tuples); `_apply_per_patch` now just binds `mesh.face_cells` to
  `boundary.apply` — do not reintroduce the loose-tuple form or a bare-dict arg (see
  `.claude/rules/boundary.md`). **`rho` is per-cell**, and the Rhie–Chow mass flux interpolates it
  *with* the velocity as the momentum `interp(ρu)·n` (density rides with velocity; the correction/
  Schur term uses `interp(ρ)`). Constant density is bit-for-bit unchanged; variable-density physics
  is not yet validated.
- **`rhie_chow.py` — `interior_mass_flux` + `momentum_diagonal`.** The Rhie–Chow face flux
  `mdot_f = ρ(u_ip·n − d̂[(p_N−p_P) − ∇̄p·d]/(d·n))A` couples pressure implicitly and kills
  checkerboarding; it reduces to the interpolated velocity flux where pressure is smooth. **Skewness-
  consistent (2nd-order on non-orthogonal meshes):** the momentum `ρu` is reconstructed to the face
  **integration point** via `interpolate_to_face` (blend + `∇·(x_ip−x_g)`), not stopped at the
  projection foot `x_g`; and the pressure damping compares the compact and reconstructed jumps **along
  the connection vector `d`** (`(Δp − ∇̄p·d)/(d·n)`), not projected on the face normal, so the two are
  the same directional derivative and cancel on a linear pressure field. Validated:
  `tests/integration/test_skewed_flow.py` (linear Couette reproduced to solver tolerance on a skewed
  mesh). **`momentum_diagonal` returns per-component `a_P` `(n_cells, dim)`** (isotropic today —
  viscous/convective/transient are component-independent — but the seam a directional source, e.g.
  porous resistance, would fill); the Rhie–Chow coefficient is the **directional** projection
  `d̂ = Σ_i n_i²(V/a_P_i)_f`, which equals the scalar `V/a_P` for equal components. Keeps `g`-weighting
  (not the Fortran's `0.5`). The block preconditioner reduces `a_P` to the isotropic
  (component-averaged) scalar — the directional form enters only the operator. The momentum **pressure
  force** (`momentum.py::_face_pressure`) is likewise reconstructed to the integration point; the
  lagged-`a_P` momentum estimate uses `interpolate_to_face` too when `grad_velocity` is available
  (the preconditioner keeps the cheap leading-order blend, exact at convergence since `a_P` is lagged).
  **Boundary faces respect the BC type** (`boundary_owner_coeff`, from each patch's
  `momentum_diagonal_coefficient`): a zero-gradient outlet adds **no** viscous diagonal (its viscous
  flux `μ(u_owner−u_owner)/(d·n)` is zero), and a no-through-flow wall adds **no** convective diagonal
  (its mass flux is exactly zero) — assembling over *all* faces uniformly over-counted both. Applied to
  the **residual** `a_P` only (`momentum_matrix_diagonal`, default `boundary_corrected=True`). The
  **frozen** preconditioner / continuation-shift diagonal (`frozen_momentum_diagonal`) keeps the plain
  all-faces form (`boundary_corrected=False`): it is a forward-path *stabilization* scale, not the
  operator coefficient, and the extra boundary damping is what carries the high-Reynolds pseudo-transient
  march — correcting it there regressed `test_channel_high_reynolds` and never affects the converged
  residual or its adjoint (the shift vanishes at the fixed point). The broader assembler unification is #58.
- **The velocity-block AMG derives its frozen operator from the shared diagonal definitions — no numpy
  re-derivation (binding, #45).** The velocity strategies used to rebuild the interior upwind stencil in
  numpy (`np.add.at`) purely to subtract it back out of `a_P` and recover a boundary diagonal — a
  reconstruction that had to match the `jnp` formula bit-for-bit across two modules and two languages.
  Now: the viscous coupling comes from `rhie_chow.viscous_face_coefficient` (the *same* helper
  `momentum_diagonal` uses, so the two cannot drift), and the boundary diagonal is the boundary-face
  owner coefficient scattered by `face_cells.scatter` — built from the same `viscous` / reference-flux
  arrays as the off-diagonals, so the assembled operator's diagonal **is** the frozen `a_P` by
  construction and the per-iterate rescaling is diagonal-exact without a bit-matching assumption
  (verified equal to the old path to ~1e-16). It stays on the **uncorrected all-faces** form, matching
  `frozen_momentum_diagonal` (`boundary_corrected=False`) — that is what the runtime rescaling divides
  by, and what carries the high-Reynolds march; do not "correct" it to the residual's form.
- **`boundary.py` — `FlowBoundary` → `NoSlipWall`, `MovingWall`, `VelocityInlet`,
  `PressureOutlet`.** Each closes velocity (viscous BC + gradient), pressure, and boundary
  `mdot`. `VelocityInlet`/`MovingWall` take a constant or a profile callable; `MovingWall` passes
  no fluid (`mdot=0`, for a driven lid). Beyond those three closures a patch also exposes the two
  **preconditioner/diagonal contributions** it alone knows: `pressure_schur_coefficient`
  (`d(mdot)/d(p_owner)`, non-zero only at a `PressureOutlet`) and `momentum_diagonal_coefficient`
  (the owner-velocity linearization of its velocity flux — viscous for a Dirichlet-velocity patch,
  convective for a through-flow patch; see the `rhie_chow.py` `a_P` note).
  - **Compose, do NOT inherit (decided; composition now realized).** A `FlowBoundary` is a *bundle*
    {velocity closure, pressure closure, mass-flux closure}, not a subtype of
    `boundary.BoundaryCondition`: it returns three coupled quantities, and `mdot` (Rhie–Chow) has no
    single-field analogue, so `FlowBoundary(BoundaryCondition)` would be an LSP violation. Its
    velocity/pressure parts, though, are **not re-implemented** here: `velocity_face`/`pressure_face`
    now **delegate to the scalar `Dirichlet`/`ZeroGradient` closures** — one per velocity component
    (`_prescribed_components` → per-component `Dirichlet`/`DirichletField`; a zero-gradient outlet →
    `ZeroGradient` per component) and one for the pressure — via the module helpers
    `_leading_order_face_value` / `_velocity_face` / `_pressure_face`. Only `mass_flux` stays
    flow-specific (and `VelocityInlet.mass_flux` reuses its own `velocity_face`). This also removed
    the duplicated `MovingWall._wall` / `VelocityInlet._inlet` broadcasts. Still **no shared base
    class** with `BoundaryCondition` — composition, not inheritance. (A constant-velocity spec is a
    static `(dim,)` sequence, so `_prescribed_components` indexes it directly — a `jnp.asarray` +
    `float()` there does not concretize under `jit`.)
  - **`corr` reconciliation (leading-order now; two-pass is the tracked follow-up).** The delegated
    scalar closures are evaluated at a **zero reconstructed gradient** (the flow assembler passes no
    boundary-face gradient), so the tangential non-orthogonal correction vanishes and the value is
    the bare owner value / prescribed value — deliberately leading-order, exact on orthogonal grids.
    Because the flow path now *reuses* `ZeroGradient` (and its single-homed `_tangential_correction`)
    rather than re-deriving it, the deferred fix is purely to feed a real gradient in place of the
    zero. The scalar path additionally does a one-step flux refinement (`residual.py::_gradient`:
    leading-order boundary → reconstruct gradient → full corr-included boundary → flux); the flow
    assembler is single-pass and lacks it. Bringing corr into the flow boundary flux **is** the
    deferred boundary-gradient fold-in (it needs that two-pass and entangles the gradient scheme with
    the BC closures) and must land for both paths together, with an analytical skewed-flow test — not
    as a local edit. Both paths are currently leading-order at non-Dirichlet skewed boundaries;
    documented as deliberate, not drift.
- **The turbulence closure enters through `eddy_viscosity`, and `μ_eff = μ + ρν_t` is formed ONCE, in
  `MomentumContinuity.viscosity` (binding).** `ν_t` (**kinematic**, the closure's own quantity) rides
  on its own differentiable leaf, set by `with_eddy_viscosity(nu_t)`; `viscosity` adds it to the
  molecular `μ` from `properties`. Callers pass only `ν_t` and never restate the closure relation.
  - **Do not put `ν_t` into the `PropertyModel`.** It is not a material property — water has no eddy
    viscosity, a turbulent flow of water does. The previous design overwrote the `"viscosity"` entry
    with a pre-summed `ρ(ν+ν_t)` field, which **destroyed the molecular value**, forced a second home
    for it (`turbulence.molecular_viscosity` — `solve_segregated`'s docstring openly said the
    assembler's own molecular viscosity "is ignored"), made the swap non-idempotent, duplicated the
    `ρ(ν+ν_t)` formula across three call sites, and changed the property's type `Constant`→
    `FieldProperty` on the first sweep. Keeping the material properties intact fixed all five.
  - The redundant `density=` parameter of `solve_segregated` and the `density` field of `CoupledRANS`
    were **removed** with it: both existed only to form `μ_eff`, which the assembler now does from its
    own `ρ`. (`SSTTurbulence` keeps its `density` and `molecular_viscosity` — the k/ω equations use
    them for their own `ν + σν_t` diffusion, a genuinely different coefficient.)
  - `with_eddy_viscosity` is part of the contract the segregated driver requires of any injected
    momentum stand-in. Pinned in `tests/unit/test_momentum_coupling.py` on a `ρ≠1` fluid, so a dropped
    density factor cannot pass.
  - **The same call means different things in the two paths**, and nothing at the call site says
    which: `driver.py` applies it *outside* the residual, so `ν_t` is frozen for the sweep and the
    coupling is invisible to AD; `coupled.py` applies it *inside*, so `dR_momentum/d(k,ω)` flows
    through it and the monolithic Jacobian gets its cross-block terms. That is why gradients go
    through `solve_coupled` and never `solve_segregated`.
- **Convection — BUILT.** `advection_scheme=` turns on momentum convection (`mdot·u`, upwind or
  limited), which makes the residual **nonlinear** (mass flux and advected velocity both depend
  on velocity). `a_P`'s convective part uses a lagged velocity-flux estimate (breaks the
  `a_P`↔`mdot` circularity). Closed domains pin the pressure at one cell (`pressure_pin=`).
- **Body force — BUILT.** `body_force=(β,…)` adds a uniform per-volume source to the momentum
  residual (subtracted per component: `R = flux − β·V`). Sign: with `p = p̃ + G·x`, `β>0` drives
  `+x` and the mean gradient is `G = −β`. It is a differentiable leaf (a mass-flow controller swaps
  it via `eqx.tree_at`), and drives a **streamwise-periodic** channel (`structured_grid_2d(periodic=
  ("x",))` + `pressure_pin`): verified against exact fully-developed Poiseuille in
  `test_periodic_channel.py`. A uniform force needs no Rhie–Chow term and does not enter continuity.
  - **The periodic seam is NOT a preconditioner problem — do not go looking there again.** A
    `reused_flow_solve` on a periodic mesh was suspected of a block-SIMPLE defect "across the seam";
    it was measured and the seam is clean. The offset is absorbed one layer down (`interpolation_factor`
    and `normal_distance` both route through `face_cells.neighbour_centroid`; the Schur matvec compares
    *field* values, which are periodic and need no offset), so: preconditioned GMRES converges in 9–18
    iterations on the periodic mesh and returns the same Newton update as unpreconditioned to `5e-8`;
    the velocity-AMG `boundary_diagonal` has no negative entries (min `-4e-16`); every shifted-step
    attempt is accepted with `|δ| ∝ 1/β`. **The real failure was the cold start**: from rest, a
    body-force channel's residual barely moves during the viscous spin-up (measured ratio `0.984` per
    step), so the feedforward SER schedule `β = β₀(‖R‖/‖R₀‖)^p` never relaxes, the pseudo-timestep stays
    tiny, and the march exhausts `max_steps` at ~`0.28‖R₀‖` — a crawl, not a stall. Two independent
    confirmations: `β₀ ≤ 0.1` converges to `1e-16` in 30 steps, and starting from `potential_flow`'s plug
    reaches `7e-13`. **Fix shipped = the velocity scale + plug start** (`flow/scales.py`), not a change
    to the schedule. A cold rest start on a body-force domain still crawls and is left to fail loudly
    (`ImplicitNewtonSolver` raises); revisiting that means revisiting cross-step SER, which was measured
    ~24% slower on the inlet channel and rejected.
- **Field initialization — BUILT (`flow/initialization.py`).** `laplace_field(mesh, geometry, boundary,
  …)` solves a scalar Laplace `div(Γ∇φ)=0` (one exact linear step) — the harmonic interpolant of any
  boundary data; general-purpose, not flow-specific. The step is **multigrid-preconditioned**
  (smoothed-aggregation V-cycle on the symmetric graph Laplacian; boundary/Dirichlet stiffness from a
  single `J·1`, fixed cells decoupled): the Laplacian's condition number grows like the **square of the
  near-wall cell aspect ratio**, so unpreconditioned it stagnated to a non-finite iterate past AR≈600
  — i.e. on every genuinely wall-resolved mesh (AR 10³–10⁵). Jacobi is not enough (removes one factor
  of AR, not two). `potential_flow(momentum)` uses it to build a
  Fluent-style irrotational velocity `u=∇φ`: a `Neumann` `∂φ/∂n = u_in·n` at each `VelocityInlet`, a
  `Dirichlet` datum at the `PressureOutlet`, no-penetration (`ZeroGradient`) at walls; returns the flat
  `[u, p=0]` state. Divergence-free, respects geometry (≈plug only in a straight duct), and — being a
  real discrete gradient — carries the tiny asymmetry that avoids the coupled solve's perfectly-symmetric
  degeneracy. A domain with **no through-flow boundary** has no potential to solve for, but may still be
  driven: a body-force periodic channel returns the **plug** `scales.body_force_velocity(momentum)` (see
  the velocity-scale bullet below). A moving-lid cavity is deliberately *not* that case (no net
  through-flow → its potential really is zero; a plug would violate the stationary walls), so the test
  `test_potential_flow_is_zero_on_a_closed_domain` is the guard — do not widen the fallback to
  `characteristic_velocity`, which reports the *lid* speed. Otherwise closed domains return the zero
  state (or pin `pressure_pin`). Usable to warm-start **any** solve (flow-only, segregated, coupled).
- **Velocity scale — BUILT (`flow/scales.py`), the one home for "how fast is this flow".** Two consumers
  need a representative speed before any flow exists — the convection velocity block's frozen
  linearization and the initializer's plug — so it is derived once here rather than in either.
  `characteristic_velocity(assembler)`: the **fastest** patch `reference_velocity` if any patch prescribes
  one, else `body_force_velocity`. `body_force_speed` closes the global balance `β V_total = τ_w A_wall`
  via the hydraulic length `h = V_total/A_wall` (`hydraulic_length`, wetted area from the new
  `FlowBoundary.shears_flow()` predicate — True for `NoSlipWall`/`MovingWall`), taking **`min`** of the
  laminar `β h²/(3μ)` (exact in that regime — unit-tested against the closed-form `βH²/12μ`) and the
  turbulent `20·u_τ`. `friction_velocity(assembler)` = `√(βh/ρ)` is the shared primitive: it follows from
  the force balance alone (no viscous or turbulence assumption), so it is the one velocity scale such a
  domain always has — which is why `turbulence/initialization.py` also sizes its equilibrium `k`/`ω` from
  it (see `.claude/rules/turbulence.md`). `min` not `max` here: the laminar branch is an extrapolation that
  diverges as `μ→0`, unlike the prescribed-velocity rule where every candidate is genuinely imposed.
  **Why it exists:** a streamwise-periodic channel prescribes velocity *nowhere*, so the old
  boundary-only estimate returned exactly zero and silently degraded `velocity="convection"` to the
  viscous block. `BlockPreconditioner.build` now warns (`RuntimeWarning`) when the convection block is
  asked for but the reference mass flux is zero. A caller that *knows* the speed (the mass-flow
  controller holds `bulk_velocity_target`) should pass `reference_state=` explicitly instead.
- **Validated:**
  - Poiseuille (`test_poiseuille.py`) — parabolic `u`, `v≈0`, linear `p`; **2nd-order**; Stokes
    is linear so **one Newton step**; differentiable.
  - **Lid-driven cavity** (`test_cavity.py`, `Re=100`) — the first genuinely nonlinear flow;
    **multi-step Newton converges** (~5 steps to `1e-12`); centreline velocities match **Ghia et
    al. (1982)** (primary-vortex `u_min≈−0.21`, `v` centreline within ~0.02); reverse-mode
    **differentiable through the nonlinear solve via the IFT adjoint**.
- **Structure (post-encapsulation-refactor):** the preconditioner is `block_preconditioner.py`'s
  `BlockPreconditioner` — a builder composing two **injected strategy families**: an `InnerSchurSolver`
  (`SmoothedAmgSchur`, the mesh-independent smoothed-aggregation multigrid) for the pressure block and a
  `VelocityBlockSolver` (`SmoothedAmgVelocity` on the viscous operator / `SmoothedAmgConvectionVelocity`
  on the convection-diffusion operator) for the momentum block, **always** block-triangular (the `D·δu`
  coupling). Build it with `BlockPreconditioner.build(assembler, velocity=…).factory()` (the factory is
  the `preconditioner` argument `newton_step` expects). The dominated Stage-1 fallbacks
  (`AggregationSchur`/`inner="multigrid"`, `DampedJacobiSchur`/`inner="jacobi"`, `DiagonalVelocity`) and
  the geometric coefficient-flow multigrid they used were **deleted** as superseded (see root `CLAUDE.md`
  Principle 0); there is no `inner=` selector — the smoothed AMG is the one pressure Schur. **The
  dependency is one-way** `block_preconditioner → momentum` — the assembler does **not** know about the
  preconditioner (the old `MomentumContinuity.simple_preconditioner` convenience wrapper was removed
  precisely because it made `momentum.py` import `block_preconditioner`, a cycle; do not reintroduce it).
  The low-level coefficient/Laplacian kernels stay in `preconditioner.py`. `a_P` comes from
  `MomentumContinuity.lagged_momentum_diagonal`. `InnerSchurSolver` / `VelocityBlockSolver` stay abstract
  as the extension seam — when adding a strategy, add a subclass; do **not** grow an `if … == …` branch.
- **Frozen operators are assembled by `aquaflux/solve/frozen_operator.py`, not here (#45).** All three preconditioner hierarchies (pressure Schur, viscous velocity block,
  convection velocity block) build their scipy CSR operator with
  `convection_diffusion_operator(owner, nb, coefficient, n, *, flux=None, boundary_diagonal=None)` and
  regularize the closed-domain pin with `decouple_dof`, then hand the **assembled matrix** to the
  coarsening builders. Do not re-assemble a stencil inside `block_preconditioner.py`.
- **The symmetric rescaling and the per-component lift are single-homed (binding — Principle 2).** Every
  block here freezes a multigrid hierarchy at a reference operator and tracks the current one by the
  symmetric congruence `A_cur⁻¹ ≈ D⁻¹ A_ref⁻¹ D⁻¹`, `D = sqrt(diag_cur/diag_ref)` — the most subtle
  numerical invariant in the file. It lives **once**, in `_symmetric_rescaled(inner_solve, diag_ref,
  diag_cur)`; the momentum block's per-component lift lives once in `_per_component(scalar_solve, dim)`.
  `SmoothedAmgSchur.apply` composes the first; both velocity blocks inherit a **single** `apply`
  (`_per_component` ∘ `_symmetric_rescaled`) from the intermediate base `_RescaledAmgVelocity`, which
  owns the shared `hierarchy` / `dim` / `v_cycles` fields and defers to an abstract `_inner_solve` — the
  only thing the viscous and convection-aware blocks actually differ in (besides how they *build* the
  hierarchy). A new AMG-based velocity block subclasses `_RescaledAmgVelocity` and supplies `build` +
  `_inner_solve`; do **not** re-write the rescale sandwich or the component loop in a strategy's `apply`.
  The invariant is pinned directly (rescaled dense solve == exact `A_cur⁻¹` for a diagonal congruence)
  in `tests/unit/test_preconditioner.py`, independent of any multigrid. (Was issue #51: the sandwich was
  hand-written in three `apply` methods and the component loop byte-identical in two.)
- **Every strategy builds from a narrow geometry bundle, not the assembler (binding — Principle 3).**
  Both strategy families take a frozen geometry seam rather than reaching into the full
  `MomentumContinuity`: the Schur family holds a `_SchurGeometry` (`face_cells`, `mesh_geometry`,
  `boundary`, `interp_factor`, `normal_distance`, `rho`, `pressure_pin`; it also *owns* the Schur-coeff
  computation `coefficient(a_P)` / `boundary_diagonal(a_P)`, so it is a **stored, runtime** object), and
  the velocity family builds from a sibling `_VelocityGeometry` (`face_cells`, `mesh_geometry`,
  `interp_factor`, `normal_distance`, `viscosity`, `dim`) that is **build-time-only** — the velocity
  strategies freeze their AMG hierarchy at build and never store it. Keep these **siblings, not one union
  bundle** (the runtime-vs-build-time split and the disjoint Schur-only / velocity-only fields are why).
  Anything that is assembler *behaviour* rather than geometry — the Rhie–Chow `mass_flux(reference_state)`
  the convection velocity block freezes at — is computed by the builder (which has the assembler) and
  handed in as a frozen array (`reference_mdot`), so the strategy `build` stays assembler-free and
  unit-testable from mesh primitives alone. When adding a strategy, extend/consume the matching bundle;
  do **not** thread the assembler into a strategy.
- **Outer block preconditioner — Stage 1 (SIMPLE block-diagonal) was BUILT then REMOVED as superseded.**
  The original composition was a `DampedJacobiSchur` (a fixed damped-Jacobi sweep on the compact
  Rhie–Chow **pressure Laplacian** `pressure_schur_laplacian`, coefficient `c_f = ρ(V/a_P)_f A_f/(d·n)_f`)
  + `DiagonalVelocity` (`diag(a_P)⁻¹`) block-diagonal preconditioner. It gave ~8–12× fewer outer GMRES
  iterations (skewed cavity: 92→11 at 432 dof, 309→25 at 1200) — but the count still *grew* with mesh
  because the Jacobi inner is not h-independent. Stage 2 (the mesh-independent smoothed-AMG inner) made
  it obsolete, so `DampedJacobiSchur` / `DiagonalVelocity` (and the `inner="jacobi"` selector) were
  deleted as dominated (root `CLAUDE.md` Principle 0). The underlying kernels `pressure_schur_laplacian`
  / `damped_jacobi_solve` remain in `preconditioner.py` (still directly unit-tested); the composed Stage-1
  strategy is gone.

## Binding decisions
- **`a_P` (momentum diagonal) is a LAGGED stabilization coefficient, not the AD Jacobian.**
  Computed from the standard central-coefficient formula (viscous + convective + transient) and
  `stop_gradient`-ed. Justified because the Rhie–Chow term vanishes at convergence (continuity
  holds for any positive `a_P`), so lagging affects only the convergence path, not the converged
  solution — matches the Fortran reference (which hand-extracts `diag = A(0,i,i)` and lags it).
  A fully AD-linearized `a_P` is a possible refinement (the diffusion Gate-C / limiter pattern),
  not yet needed.
- **Monolithic block, AD Jacobian.** The reference (`reference_codes/…conjugatecfd`) assembles
  the 4×4-per-face block `A(0:6,4,4)` **by hand** in an outer **Picard** loop with lagged
  coefficients, one **BiCGStab** solve (default PC, **no AMG** — "reasonable convergence" for
  laminar/porous). Our version replaces the hand-assembled block with the AD Jacobian and Newton;
  the block **preconditioner is the top research risk** — `lineax` GMRES is the
  current default; AMG (`jaxamg`, unverified) is deferred exactly as the reference deferred it.
- **Reuse over reimplementation (Principle 2).** Momentum viscous/advection are the existing
  operators with μ as the diffusion coefficient and `mdot` as the mass flux — do **not** re-derive
  a momentum flux. New code is only the pressure force, Rhie–Chow, and the flow BCs.

## Not yet built (follow-ons, in order)
- **Porous / conjugate interface** (`eps` porosity terms, `addCont` interface branch) — the
  reference's distinguishing capability.
- **Transient** momentum (BDF, reusing `TransientTerm`). **Must use a transient-consistent Rhie–Chow:
  subtract the transient component of `a_P` in the mass-flux d-coefficient** — use `a_P^{RC} = a_P −
  ρV/Δt` (spatial only), not the full `a_P`, or the pressure smoothing (and the `-C` block) vanishes as
  `Δt → 0`, giving checkerboarding *and* a singular SIMPLE Schur in the preconditioner (author's ANSYS
  experience — two `a_P` roles: spatial for Rhie–Chow/`-C`,
  full for SIMPLE's `diag(F)⁻¹`). `momentum_diagonal` already carries an unused `dt` seam. Then **energy
  coupling** (the scalar transport already exists — couple the temperature field to the flow).
- **Outer block preconditioner — Stage 2: MESH-INDEPENDENT AMG inner is done; the residual is the Schur
  approximation.** The inner Schur solve is a **smoothed-aggregation multigrid** V-cycle
  (`aquaflux/solve/multigrid.py`, `SmoothedAmgSchur`, the one inner — `BlockPreconditioner.build(asm)`).
  Built once off-jit with `scipy.sparse` (no PyAMG needed),
  applied as a frozen matrix-free V-cycle; **~0.25 flat contraction** via three fixes — **direct coarse
  solve** (dense pinv), **Chebyshev smoother**, and **pin decoupling** (pin zeroed out of the operator →
  SPD singleton, null-space-matched to the pinned Jacobian; this fixed the earlier stagnation/wrong-solution
  correctness bugs). A **symmetric diagonal rescaling** `√(diag_cur/diag_ref)` tracks the reference scale
  (essential). **BOTH blocks get an AMG:** a block-by-block diagnostic (make each block's inverse exact in
  turn) showed the Schur-only version's residual ~√N was the **velocity block** (`diag(a_P)⁻¹` is
  Jacobi-quality for the viscous `F⁻¹`), *not* the Schur (making the Schur exact changed nothing) — so
  **PCD/LSC would not have helped.** The velocity momentum block gets its own smoothed AMG (Dirichlet
  no-slip → SPD-nonsingular via a `boundary_diagonal`, no pin; per component). The preconditioner is **block-TRIANGULAR**: the
  pressure block sees the divergence of the velocity predictor, `δp = Ŝ⁻¹(r_p − D·δu)`, with `D·δu`
  applied as a **jvp through a frozen residual** (`stop_gradient(self)`, so `D` is constant and
  adjoint-transparent — verified). **Result: outer GMRES 4→8 over 144→2304 cells (~O(N^0.25)), ≈37 iters
  extrapolated to 1e6 cells** vs unpreconditioned 95→1818; correct/differentiable, all
  `test_flow_preconditioner.py` pass. (Progression: Schur-AMG-only 9→34 ~√N; +velocity-AMG 7→14 ~O(N^0.3);
  +D-coupling 4→8 ~O(N^0.25).) The tiny residual growth is the 1-cycle-AMG block approximation, not the
  structure. **Two cheaper diagonals were measured and rejected** — velocity block-Jacobi and
  inverse-volume-Jacobi on the gradient solve; see `solve.md`, do not re-attempt.
- **Fully-AD `a_P`** — a possible refinement (the diffusion Gate-C / limiter pattern), not yet needed.
- **Gradient-scheme cost — largely solved (use `SweptCorrectedGradient`).** The *per-matvec* and
  *compile* cost of the nested corrected-gradient solve (distinct from the outer iteration count) is
  cut ~5× by inverting the constant `A_g` with fixed matrix-free Richardson sweeps instead of a
  nested implicit-diff GMRES — an exact `O(n)` drop-in (N=32 coupled Newton step 112 s → 23 s). It is
  the default advanced-gradient scheme to inject into `MomentumContinuity` on skewed meshes. See
  `schemes.md` / `solve.md`. (A dense-LU variant was measured and removed — `O(n²)`, strictly dominated;
  do not rebuild.)

## Testability seam
`interior_mass_flux` and the BCs are pure functions of per-face arrays — unit-tested with no
mesh (`test_rhie_chow.py`, `test_flow_boundary.py`). `momentum_diagonal` is checked for positivity
and linear-in-μ scaling. The coupled solve is gated on the analytical Poiseuille field and its
2nd-order convergence, plus an AD-through-solve no-NaN check.
