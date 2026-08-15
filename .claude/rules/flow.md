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
  (`p_f n_i A`) — **and all three are `FaceFluxOperator`s composed by the shared `CellBalance`**,
  the same object that assembles every scalar transport equation (`.claude/rules/discretization.md`).
  Per component it builds a `FaceContext` (properties `{"viscosity": μ}` via
  `DiffusionFlux(coefficient="viscosity")`, the component gradient, the boundary velocity) and hands
  it to `CellBalance((DiffusionFlux, PressureForce, AdvectionFlux)).residual(component, context)`.
  Continuity is `Σ mdot_f = 0`. The whole Jacobian comes from AD; solved by the existing
  `ImplicitNewtonSolver`.
  - **`PressureForce` (`momentum.py`, exported from `aquaflux.flow`) is the one flow-specific
    operator.** `p_f n_i A` — the momentum pressure term written as what it is, a surface flux of
    momentum — carrying the already-reconstructed face pressure as constructor state and the
    component as a static field, and **ignoring the transported field** (pressure is a different
    unknown, exactly as `AdvectionFlux` ignores it through `mass_flux`). That loses no coupling:
    `face_pressure` is a differentiable function of the pressure unknowns, so AD still recovers the
    full `dR_momentum/dp`. Before this it was added by hand inside `_momentum_residual`, which is
    what forced the momentum block to open-code the flux sum and scatter instead of composing the
    shared balance.
  - **⚠️ The operator order `(diffusion, pressure, advection)` is load-bearing arithmetic, not
    style.** A `CellBalance` sums in tuple order and floating-point addition is not associative, so
    reordering perturbs the residual in the last bits — and `bfs3d` march trajectories are compared
    bit-for-bit across runs (`march_log_compare.py` exists because a trajectory difference was once
    mis-attributed to a change rather than to the base). This order reproduces the hand-assembled
    `(diffusion + pressure) + advection` it replaced; the whole refactor was gated on the coupled
    `bfs3d` residual at `state-00067` being bit-unchanged, not on tests passing.
  - **`MomentumContinuity` has no `_scatter`** — there is no such method; the balance scatters, and
    continuity calls `mesh.face_cells.scatter_conservative(mdot)` directly.
  - **One shared assembly per state — `flow_fields` / `residual_from_fields` (binding, #106).** The
    boundary fields, both gradients, the lagged `a_P`, and the Rhie–Chow flux are assembled once by
    the public `flow_fields(state) -> FlowFields`; `residual` = `residual_from_fields(flow_fields(state))`
    and `mass_flux(state)` = `flow_fields(state).mdot`. A coupling caller that needs several of these at
    one state (a coupled RANS residual wanting the residual **and** `mdot`, a segregated sweep wanting
    the gradient **and** `mdot`) must call `flow_fields` **once** and read the fields — not call the
    accessors separately, which re-assembles the whole Rhie–Chow flux per accessor (was 3× per coupled
    residual eval; the pre-optimization HLO / AD-tape size scaled with that). **`velocity_fields` is
    deliberately the lightweight path** (boundary velocity + the shared `_velocity_gradient` only, **no**
    `a_P`/`mdot`): the eddy viscosity a segregated sweep needs comes before the mass flux is even
    defined, so dragging the Rhie–Chow assembly through it would defeat the point. Same formula as the
    bundle (both compose `_boundary_fields` + `_velocity_gradient`), so no duplication. Pinned by
    `test_flow_fields_accessors_agree_with_the_bundle` / `test_velocity_fields_skips_the_rhie_chow_assembly`.
  - **The kinematic half is its own bundle, `VelocityFields` (`velocity`, `boundary_velocity`,
    `gradient`), nested inside `FlowFields` (binding — Principle 3).** It is exactly the part of a flow
    state that is a pure function of the velocity unknowns (no pressure, no `a_P`, no `mdot`), and it is
    the *whole* of what the turbulence closure reads from the flow — `SSTTurbulence.closure_fields`
    takes this one object. It exists because the adaptive wall treatment needs `velocity` and
    `boundary_velocity` **as well as** the gradient (the near-wall shear rate is `|U_P − U_wall|/d`
    measured against the patch's own velocity), and threading three arrays that always travel together
    is the primitive-obsession smell. `MomentumContinuity.velocity_fields(state)` returns it directly;
    `flow_fields(state).velocity_fields` is the same record inside the full bundle — **nested, not a
    second view**, so there is no duplicate spelling of those three arrays. It replaced
    `velocity_gradient(state)` outright (pre-release, no shim); a caller that only wants the tensor
    writes `velocity_fields(state).gradient`.
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
  - **⚠️ `a_P` does NOT widen the coupled Jacobian's velocity stencil — measured, and the opposite is an
    easy conclusion to reach from this code (2026-08-11).** The argument that looks right: the mass flux
    carries `V/a_P` averaged across the face, so a cell's residual reads its *neighbour's* momentum
    diagonal, and that diagonal is a sum over the neighbour's own faces of a lagged convective flux which
    is itself gradient-reconstructed — a ring on the far side. It is wrong. Removing the velocity gradient
    from `_lagged_mdot_estimate` leaves the velocity columns reaching graph distance **3**, unchanged; so
    does first-order momentum advection, and so does a one-shot compact gradient. What actually spends the
    ring is the **eddy viscosity's dependence on the strain rate** (`.claude/rules/turbulence.md`). Note
    the one asymmetry here that *is* real and is worth keeping: `_lagged_mdot_estimate` is a function of
    velocity alone — it carries no pressure, which is what breaks the `a_P` ↔ `mdot` circularity — so
    pressure never enters `a_P`. That is true, and it is not what sets the reach.
    Harness: `validation/bfs3d_openfoam/column_reach_probe.py`.
  **Boundary faces respect the BC type** (`boundary_owner_coeff`, from each patch's
  `momentum_diagonal_coefficient`): a zero-gradient outlet adds **no** viscous diagonal (its viscous
  flux `μ(u_owner−u_owner)/(d·n)` is zero), and a no-through-flow wall adds **no** convective diagonal
  (its mass flux is exactly zero) — assembling over *all* faces uniformly over-counted both. Applied to
  the **residual** `a_P` only (`momentum_matrix_diagonal`, default `boundary_corrected=True`).
  **The viscous boundary term uses the operator's per-face boundary viscosity, not the owner cell's
  (binding, #155).** `boundary_momentum_diagonal` builds `μ_f A/(d·n)` from `_boundary_face_viscosity()`
  — the same per-face coefficient the residual's `DiffusionFlux` carries at the boundary
  (`_wall_boundary_viscosity()`, falling back to the owner-cell `μ_eff` off the shearing walls). On a
  **wall-function** mesh the wall model's `μ + ρ·ν_t,wall` and the wall-adjacent cell's *log-layer*
  `k/ω` `μ_eff` differ by a large factor (measured ~6.5× on a `y*≈38` cavity), so gathering the owner
  cell's viscosity left `a_P ≠ diag(J_momentum)` exactly where the wall model is active (suppressing the
  Rhie–Chow damping `V/a_P` there — a converged-solution effect, like #154). Wall-**resolved** meshes
  have `ν_t,wall = 0` (a no-op), which is why the wall-resolved `test_channel_law_of_the_wall` cannot see
  it; pinned instead by a wall-function `a_P` vs `diag(jacfwd(residual))` test (`test_preconditioner.py`). The
  **frozen** preconditioner / continuation-shift diagonal (`frozen_momentum_diagonal`) keeps the plain
  all-faces form (`boundary_corrected=False`): it is a forward-path *stabilization* scale, not the
  operator coefficient, and the extra boundary damping is what carries the high-Reynolds pseudo-transient
  march — correcting it there regressed `test_channel_high_reynolds` and never affects the converged
  residual or its adjoint (the shift vanishes at the fixed point). The broader assembler unification is #58.
- **`a_P`'s viscous term is the flux-continuous conductance, i.e. the diffusion operator's own diagonal
  (binding, #154).** `momentum_diagonal`/`momentum_diagonal_parts` build the viscous coupling from
  `discretization.flux_continuous_conductance(μ_eff, geometry, face_cells)` — `Γ_P A / denom`,
  `denom = (D_P·n) − (Γ_P/Γ_N)(D_N·n)`, the *same* per-face value AD gets by differentiating
  `DiffusionFlux`'s owner-outward flux, so scattering it to both cells reproduces the residual's momentum
  diagonal. On an orthogonal graded face it is the **harmonic mean** `2Γ_PΓ_N/(Γ_P+Γ_N)·A/h`. The old
  g-weighted **arithmetic** face viscosity (`viscous_face_coefficient`, now deleted) agreed only for
  constant `μ`; on a face with viscosity ratio `r` it over-estimated the coupling by `(1+r)²/(4r)`, so
  under a graded turbulent `μ_eff = μ + ρν_t` `a_P` was *not* the operator diagonal — which mattered
  because `a_P` is not solver-internal: it sets the differentiated Rhie–Chow damping `V/a_P` (so it
  changed the *converged* solution in the shear layer/near-wall), and it is the base of the
  pseudo-transient shift (so `βd` was not the spatially-uniform relaxation `1/(1+β)` the shift-basis
  assumes). **Constant `μ` is byte-identical** to the old form on any mesh (`Γ_P=Γ_N ⇒ denom = (x_N−x_P)·n`),
  which is why every prior test — all constant-`μ` — passed the wrong form. The **same** helper feeds the
  frozen velocity AMG operators and the scalar shift/AMG (`turbulence/preconditioner.py`), so none can
  drift from the residual. Note `g`-weighting still governs the *other* face interpolations
  (`interp(ρu)`, `interp(∇p)`, `interp(V/a_P)`); only the viscosity face value moved to the owner-based
  conductance. Regression-pinned: `momentum_matrix_diagonal` vs `diag(jacfwd(residual))` at graded `ν_t`
  (`test_preconditioner.py`), the conductance vs the `DiffusionFlux` diagonal (`test_diffusion.py`).
- **The velocity-block AMG derives its frozen operator from the shared diagonal definitions — no numpy
  re-derivation (binding, #45).** The velocity strategies used to rebuild the interior upwind stencil in
  numpy (`np.add.at`) purely to subtract it back out of `a_P` and recover a boundary diagonal — a
  reconstruction that had to match the `jnp` formula bit-for-bit across two modules and two languages.
  Now: the viscous coupling comes from `discretization.flux_continuous_conductance` (the *same* helper
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
  - **`with_scaled_molecular_viscosity(factor)` rescales the molecular `μ` only** (the `"viscosity"`
    entry in `properties`, via `PropertyModel.with_scaled`), leaving the eddy-viscosity leaves alone.
    It is the momentum half of the coupled Reynolds-number rescale a continuation applies (the
    turbulence half scales `SSTTurbulence.molecular_viscosity`; `CoupledRANS.with_scaled_molecular_viscosity`
    composes both) — see `.claude/rules/turbulence.md`. Molecular `μ` and eddy `ν_t` stay separate leaves,
    so the rescale cannot disturb the closure.
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
  - **The wall-function eddy viscosity enters the SAME seam, as a second (per-face) leaf.**
    `with_eddy_viscosity(nu_t, wall_eddy_viscosity=None)` also carries an optional per-face **kinematic**
    `ν_t,wall` (shape `(n_faces,)`), the adaptive wall function's value (`turbulence.wall_face_eddy_viscosity(k)`,
    the `nut_wall` blend on wall faces, zero off them). `_wall_boundary_viscosity()` turns it into the
    momentum diffusion's `boundary_coefficient`: on the flow's **shearing walls** (`FlowBoundary.shears_flow()`)
    the wall-face coefficient is `μ + ρ·ν_t,wall` — the wall model's value replacing the owner-cell
    `k/ω` closure — every other face keeps the owner `μ_eff`, and interior faces are ignored by
    `DiffusionFlux`. `μ_eff = μ + ρν_t` stays formed **here** (the closure still supplies only the
    kinematic value). The **residual `a_P` reads this same per-face coefficient at the boundary**
    (`_boundary_face_viscosity()` → `boundary_momentum_diagonal`), so the momentum diagonal matches the
    operator at wall faces rather than gathering the (large, log-layer) owner-cell `μ_eff` there — see
    the `a_P` boundary bullet above (#155). `None` (default) leaves walls resolved, so the momentum block is **bit-identical**;
    on a wall-resolved mesh `ν_t,wall = 0` (below `y*_lam`), so it is a no-op there too — this is what
    lets it be always-on. Applied in **both** paths so they solve the identical near-wall model:
    `coupled.py` inside the residual (live, in the coupled Jacobian) and `driver.py` per sweep (frozen).
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
  `+x` and the mean gradient is `G = −β`. It is a differentiable leaf (the mass-flow constraint below
  swaps it via `eqx.tree_at`), and drives a **streamwise-periodic** channel (`structured_grid_2d(periodic=
  ("x",))` + `pressure_pin`): verified against exact fully-developed Poiseuille in
  `test_periodic_channel.py`. A uniform force needs no Rhie–Chow term and does not enter continuity.
- **Bulk-velocity constraint — BUILT (`flow/mean_velocity.py`, `bulk_velocity_flow_solve`).** A
  streamwise-periodic channel is driven to a target **bulk velocity** `U_bar` by making the body force
  `β` a **scalar Lagrange multiplier** on the constraint `⟨U_dir⟩ − U_bar = 0`, solved *jointly* with
  the flow — not by an outer proportional controller. `β` is **appended to the flow state** and the
  flow residual is augmented with the constraint equation, `R_aug([w,β]) = [R_flow(w;β); ⟨U_dir⟩(w) −
  U_bar]`; this **one honest residual** is handed to the production `ImplicitNewtonSolver`, and **AD
  assembles the whole bordered Jacobian** (the force column `∂R/∂β = −V` on the flow-direction momentum
  rows, since `R = flux − βV`, and the averaging row `∂⟨U⟩/∂w = V/ΣV`) — **no bespoke solver, no
  hand-derived border** (an earlier version hand-rolled a two-solve rank-one elimination purely to keep
  the block preconditioner applicable to the border; that is a preconditioner concern, and the study
  uses a direct solve where it buys nothing — so it was dropped for the augmented residual, which also
  respects the "AD assembles the Jacobian" rule the hand-derived form broke). `⟨U⟩ = U_bar` holds **by
  construction** at the converged root, so the bulk velocity can never overshoot while the eddy
  viscosity is still developing (the failure the old proportional controller had at high Re / high
  aspect ratio: it measured `U_bulk` at a fixed β, which spiked ~17× before the feedback reacted,
  collapsing the near-wall `k` onto its floor). Being the production solve it is **convergence-gated**
  (stops on tolerance, not a fixed count) and **reverse-differentiable** through the IFT adjoint at the
  root — the constraint is part of the differentiable model at no extra cost. Returns `(momentum,
  state)` — the assembler carries the converged β out, so a segregated outer loop threads it forward
  with no controller. The **proportional mass-flow controller in `solve_segregated` was DELETED** (its
  `bulk_velocity_target`/`bulk_velocity_gain`/`flow_direction` args gone); the driver's flow-solve seam
  is now `solve_flow(momentum, state) → (momentum, state)`.
  - **Preconditioning the augmented system — constraint preconditioning (`_bordered_preconditioner`).**
    β is a multiplier, not a function of `w`, so — unlike a nested gradient sub-solve, which AD absorbs
    into the outer Jacobian — the border cannot be absorbed inside the residual; it is eliminated one
    layer down, **in the preconditioner**. Given the flow block preconditioner `M ≈ J⁻¹` (the block-
    SIMPLE AMG `BlockPreconditioner.factory()`), `_bordered_preconditioner(M, a, c)` returns a
    preconditioner for the `(dim+1)·n_cells + 1` augmented system that Schur-eliminates the scalar β:
    `y = M·r_flow`, `dβ = (cᵀy − r_β)/(cᵀMa)`, `dw = y − dβ·(Ma)` — one flow-preconditioner apply plus
    O(n) dots per augmented Krylov iteration, exact when `M = J⁻¹` (converges in one step), inheriting
    the flow block's mesh-independence otherwise. **Hand-building `a`/`c` here is legitimate** — a
    preconditioner is an approximate, `stop_gradient`-ed inverse that changes only Krylov convergence,
    never the solution or adjoint (contrast the *residual*, whose Jacobian is AD-assembled). This is the
    robust production path (large iterative solve); `preconditioner=None` (a direct or small solve) needs
    none of it. **The bordered preconditioner is built once in the builder from a concrete `reference`**
    (required whenever `preconditioner` is given), **not** inside the jitted solve from its traced
    `momentum` argument — a tracer captured into the (non-differentiated) preconditioner breaks `jax.grad`
    ("no constant handler" / closed-over-value). Pinned by `test_mean_velocity.py`: with an exact `M` the
    bordered preconditioner is exactly `J_aug⁻¹` (and AD's border matches the hand-built `a`/`c`), and the
    block-preconditioned GMRES augmented solve lands on the direct solve's answer.
  - **The solve is reverse-differentiable in `momentum` (#127).** The assembler is threaded as the Newton
    **`theta`** (not captured in the residual closure), so the IFT adjoint returns its cotangent —
    `jax.grad` of an objective through the solve gives e.g. `d/dμ` (verified vs finite differences and the
    analytic `dβ/dμ = 12U_b/H²` on the laminar channel). A captured assembler instead raises a `custom_vjp`
    closed-over-value error — the reason `theta` threading is load-bearing, not cosmetic. The **adjoint**
    transpose solve is preconditioned by the *same* bordered preconditioner **transposed**, which the
    generic adjoint machinery (`solve/implicit.py::_adjoint_preconditioner`) forms with `jax.linear_transpose`
    (`M_aug` is linear ⇒ `M_augᵀ ≈ J_augᵀ⁻¹` exactly) — no hand-written transpose needed. Pinned: the
    preconditioned gradient equals the unpreconditioned (direct) one to ~1e-9 (a preconditioner never
    changes the sensitivity, only the adjoint Krylov iteration count).
  Pinned by `tests/unit/test_mean_velocity.py` (constraint met to machine precision; analytic β
  recovered; initial-force-independent; the preconditioner and gradient tests above).
  - **The same primitives border the monolithic coupled RANS solve (#128).** `_constraint_vectors`,
    `_bordered_preconditioner`, and `_with_body_force` are imported by
    `turbulence/coupled.py::solve_coupled_mass_flow` to append `β` to the *coupled* `[flow…, k, ω]`
    state and Schur-eliminate it in the coupled preconditioner — so the bulk-velocity constraint is
    enforced by the same one place whether the forward solve is segregated (this flow-block solve) or
    monolithic. Do not re-derive the border there.
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
  `[u, p]` state with `p` the **Bernoulli** seed (below). Divergence-free, respects geometry (≈plug only
  in a straight duct). A domain with
  **no through-flow boundary** has no potential to solve for, but may still be
  driven: a body-force periodic channel returns the **plug** `scales.body_force_velocity(momentum)` (see
  the velocity-scale bullet below). A moving-lid cavity is deliberately *not* that case (no net
  through-flow → its potential really is zero; a plug would violate the stationary walls), so the test
  `test_potential_flow_is_zero_on_a_closed_domain` is the guard — do not widen the fallback to
  `characteristic_velocity`, which reports the *lid* speed. Otherwise closed domains return the zero
  state (or pin `pressure_pin`). Usable to warm-start **any** solve (flow-only, segregated, coupled).
- **Bernoulli pressure seed — BUILT (`flow/initialization.py`, `bernoulli_pressure`).** `potential_flow`
  no longer returns `p=0`: it seeds the **dynamic** pressure `p = ½ρ(|u_ref|² − |u|²)` consistent with
  the irrotational velocity (`p + ½ρ|u|² = const`), anchored so the mean pressure over the
  `PressureOutlet` cells is **zero** — consistent with the `p=0` outlet datum (`_pressure_outlet_cells`;
  a domain with no outlet anchors the domain-mean to zero instead, so a uniform plug / quiescent cavity
  stays at 0). **Why:** a coupled Newton solve otherwise starts with the *whole* dynamic-head pressure
  field as a first-step correction — measured `‖δ_flow‖≈2694` with `p=0`, of which `‖δ_pressure‖` is
  ~100% while `‖δ_vel‖≈8`; the seed **halves** it (`δ_flow≈1348`). **Closed-form on purpose (not the
  pressure-Poisson equation):** the PPE source `ρ·tr(∇u·∇u)` blows up at a sharp geometric corner (the
  backward-facing step, whose potential-flow `∇u` is singular) — measured `‖p_ppe‖≈9.4e4` and `δ_flow`
  *35× worse*; Bernoulli reads `|u|`, not `∇u`, so it is immune. This matches how ANSYS/OpenFOAM actually
  behave — ANSYS hybrid init solves a **source-free** pressure Laplace (`∇²p=0`) from the pressure BCs
  and, for a velocity-inlet/pressure-outlet case with no pressure-inlet info, seeds `p≈const` and lets
  the SIMPLE pressure under-relaxation build the field. So the remaining intrinsic `δ_pressure` (the
  potential→viscous change) is a **solver-side** concern (per-block step / pressure under-relaxation in
  the coupled step), not an IC one. Pinned by `test_bernoulli_pressure_*` (outlet-anchored, dynamic-head
  ordering, uniform-flow → 0).
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
  asked for but the reference mass flux is zero. A caller that *knows* the speed (a bulk-velocity
  constraint targets `U_bar`) should pass `reference_state=` explicitly instead.
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
  the `preconditioner` argument `newton_step` expects). **`strength_threshold=θ` (default `0`) turns on
  strength-of-connection aggregation** for the velocity/Schur AMGs — aggregate along strong connections
  only, the fix that keeps the V-cycle contracting on a high-aspect-ratio / skewed mesh where isotropic
  aggregation coarsens across the stiff wall-normal direction and stalls; `θ=0` is byte-identical to the
  historical build, and the coupled RANS path turns it on at `0.25` (a no-op on low-AR pitzDaily, the
  payoff is the wall-resolved regime). Full evidence + the value-dependence/refresh caveat live in the
  AMG section of `.claude/rules/solve.md`. The dominated Stage-1 fallbacks
  (`AggregationSchur`/`inner="multigrid"`, `DampedJacobiSchur`/`inner="jacobi"`, `DiagonalVelocity`) and
  the geometric coefficient-flow multigrid they used were **deleted** as superseded (see root `CLAUDE.md`
  Principle 0); there is no `inner=` selector — the smoothed AMG is the one pressure Schur. **The
  dependency is one-way** `block_preconditioner → momentum` — the assembler does **not** know about the
  preconditioner (the old `MomentumContinuity.simple_preconditioner` convenience wrapper was removed
  precisely because it made `momentum.py` import `block_preconditioner`, a cycle; do not reintroduce it).
  The low-level coefficient/Laplacian kernels stay in `preconditioner.py`. `a_P` comes from
  `MomentumContinuity.momentum_matrix_diagonal` / `BlockPreconditioner.frozen_momentum_diagonal`. `InnerSchurSolver` / `VelocityBlockSolver` stay abstract
  as the extension seam — when adding a strategy, add a subclass; do **not** grow an `if … == …` branch.
  - **`frozen_momentum_diagonal_parts(assembler, state)` is a PUBLIC module-level function, not only a
    `BlockPreconditioner` method (2026-08-12).** The convective/dissipative buckets a
    `ShiftBasis` combines need **the assembler and nothing else** — the method has always just
    delegated to it. Exposing it is what lets a pseudo-transient shift be built without a
    preconditioner, and that is not cosmetic: `CoupledShiftPolicy` used to hold a whole
    `BlockPreconditioner` to read these two arrays, so a monolithically-preconditioned coupled step
    (which supplies its own inverse and never applies that one) built two multigrid hierarchies per
    build that were never used — *and*, because their aggregation reads the operator's values, their
    coarse-grid array shapes moved with the molecular viscosity, recompiling the whole coupled solve at
    every Reynolds-continuation rung. See `.claude/rules/turbulence.md`. The method stays as the one
    call site's convenience; the function is the one home.
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
- **`a_P` (momentum diagonal) is DIFFERENTIATED in the residual; only the PRECONDITIONER freezes it
  (binding — do not describe `a_P` as a lagged/`stop_gradient`-ed coefficient).** It is computed from
  the standard central-coefficient formula (viscous + convective + transient), with its convective part
  using a velocity-flux *estimate* for the mass flux (not the Rhie–Chow `mdot` itself), which breaks the
  `a_P` ↔ `mdot` circularity and makes `a_P` a non-circular function of the velocity. It then enters the
  Rhie–Chow damping `V/a_P`, whose term is non-zero for a non-linear pressure field, so `a_P` genuinely
  affects the **converged solution's sensitivity** — freezing it (`stop_gradient`) would leave the
  implicit-function-theorem adjoint linearizing a *different* residual than the one solved (the converged
  *value* would be unchanged, the *sensitivity* not). So the residual differentiates `a_P` (verified: on
  pitzDaily the AD-jvp of the coupled residual matches a central finite difference to ~1e-7 in the flow
  block). The **block preconditioner**, which needs a constant operator, `stop_gradient`-s `a_P` on its
  side (and evaluates it at a `stop_gradient`-ed state) — that is where the freezing lives.
  (Historical note: an earlier design lagged `a_P` in the residual; the code moved to differentiating it.
  See `rhie_chow.py` / `momentum.py`.)
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
  full for SIMPLE's `diag(F)⁻¹`). `momentum_diagonal`/`momentum_diagonal_parts` already carry an unused
  `dt`+`rho` seam whose transient term is the **density-weighted** `ρV/Δt` (conservative `∂(ρu)/∂t`,
  matching the ρ-weighted convective mass-flux and dynamic-viscosity terms on the same diagonal; `rho`
  is required whenever `dt` is given — #153). Then **energy
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
- **Outer block preconditioner — Stage 3: the remaining limit IS the Schur approximation, and no amount
  of inner accuracy reaches it (measured on a developed Re=1e5 SST channel; binding, do not re-attempt).**
  The `v_cycles` knob and the MSIMPLER scale are both exhausted: **velocity-AMG V-cycles ×2/×4/×8 leave
  the flow block's error operator `ρ(I − A_flow·M_flow)` at 34.02 / 33.99 / 34.03 / 34.05 (no effect);
  Schur V-cycles ×2/×4/×8 make it *worse* (41.6 / 48.7 / 48.5)**; both-exact never beats the 1-cycle
  baseline; and **rebuilding the whole block at the developed state does not help** (34.0 → 31.6 on the
  channel, 49.9 → 91.9 on pitzDaily). Inverting `Ŝ` more accurately making the preconditioner worse is
  the signature that `Ŝ` is the **wrong operator**. **Rescaling MSIMPLER's `k` is a ρ mirage:** it
  collapses ρ (34.0 → 9.6) while barely moving the one-shot error (24.1 → 22.6) and the ρ-optimal `k`
  sits ~40× above the *maximum* of the per-cell `ρV/a_P` distribution — the degenerate limit that
  switches the pressure correction off — and on the **real march it is slower** (auto-`k` 348 s / 8 steps
  vs `k×4` 447 s, identical trajectory). So the shipped per-apply `mean(ρV/a_P)` calibration is
  near-optimal, and **preconditioner changes must be validated on the real march, not on ρ** (ρ here is
  dominated by isolated outlier eigenvalues GMRES kills anyway). **Root cause:** the MSIMPLER Schur is a
  constant-coefficient (scaled pressure-mass-matrix) Poisson — a near-Stokes/low-Re approximation that
  degrades as convection strengthens. **A better Schur was the obvious next move — it is BUILT
  (`schur_scaling="lsc"`, the algebraic nonuniform-mesh stabilized least-squares commutator of Elman,
  Howle, Shadid, Silvester & Tuminaro 2007) and it LOSES BADLY on the coupled solve. Do not re-derive
  it.** At a developed/separated pitzDaily state, one shifted solve: msimpler **13 cycles / 38.9 s** vs
  lsc **96 cycles / 526 s** (`v_cycles=4`) and **82 / 662 s** (`v_cycles=8`) — 6–7× the cycles, 13–17×
  the wall time, both genuinely converged (`lin_rel ~2e-9`); and ~2.9× slower on the coupled channel at
  an identical residual trajectory. **The flow-only win does not transfer:** LSC beats MSIMPLER on the
  *isolated* flow block (9 vs 15 GMRES at Re=1e4), but under the coupled block-**diagonal**
  preconditioner plus the pseudo-transient shift, the coupled iteration is not limited by the flow
  Schur's quality, so improving it buys nothing. The strategy stays available for a flow-only solve;
  it is not the coupled default and is not the cure for coupled cost. PCD remains deprioritized
  independently (finite-element boundary recipes that do not transfer to FVM). Full numbers and the
  matching "what a preconditioner can and cannot change" rule are in `.claude/rules/solve.md`.
- **Fully-AD `a_P`** — a possible refinement (the diffusion Gate-C / limiter pattern), not yet needed.
- **Gradient-scheme cost — largely solved (use `SweptGradientSolve`).** The *per-matvec* and
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
