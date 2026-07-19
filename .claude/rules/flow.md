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
  from AD; solved by the existing `NewtonSolver` / `ImplicitNewtonSolver`. **Fluid properties come
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
  residual or its adjoint (the shift vanishes at the fixed point). Wiring the velocity-block AMG to the
  same shared decomposition (robustly) is deferred to #45; the broader assembler unification is #58.
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
- **Validated:**
  - Poiseuille (`test_poiseuille.py`) — parabolic `u`, `v≈0`, linear `p`; **2nd-order**; Stokes
    is linear so **one Newton step**; differentiable.
  - **Lid-driven cavity** (`test_cavity.py`, `Re=100`) — the first genuinely nonlinear flow;
    **multi-step Newton converges** (~5 steps to `1e-12`); centreline velocities match **Ghia et
    al. (1982)** (primary-vortex `u_min≈−0.21`, `v` centreline within ~0.02); reverse-mode
    **differentiable through the nonlinear solve via the IFT adjoint**.
- **Structure (post-encapsulation-refactor):** the preconditioner is `block_preconditioner.py`'s
  `BlockPreconditioner` — a builder composing two **injected strategy families**: an `InnerSchurSolver`
  (`SmoothedAmgSchur` / `AggregationSchur` / `DampedJacobiSchur`) for the pressure block and a
  `VelocityBlockSolver` (`DiagonalVelocity` / `SmoothedAmgVelocity`), with the block-triangular `D·δu`
  coupling. Build it with `BlockPreconditioner.build(assembler, inner=…).factory()` (the factory is
  the `preconditioner` argument `newton_step` expects). **The dependency is one-way** `block_preconditioner
  → momentum` — the assembler does **not** know about the preconditioner (the old
  `MomentumContinuity.simple_preconditioner` convenience wrapper was removed precisely because it made
  `momentum.py` import `block_preconditioner`, a cycle; do not reintroduce it). The low-level
  coefficient/Laplacian/Jacobi kernels stay in `preconditioner.py`. `a_P` comes from
  `MomentumContinuity.lagged_momentum_diagonal`. When adding an inner solver, add an `InnerSchurSolver`
  subclass — do **not** grow an `if inner == …` branch (that god-method was the thing the refactor
  removed; see root `CLAUDE.md` Principle 3).
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
- **Outer block preconditioner — BUILT (Stage 1: SIMPLE block-diagonal).** The `DampedJacobiSchur` +
  `DiagonalVelocity` composition is a frozen left preconditioner
  `M = blkdiag(diag(a_P)⁻¹, Ŝ⁻¹)`: the velocity block is the momentum-diagonal solve (reusing the
  lagged `a_P`), and `Ŝ` is the compact Rhie–Chow **pressure Laplacian** (`pressure_schur_laplacian`,
  coefficient `c_f = ρ(V/a_P)_f A_f/(d·n)_f` — same `d = V/a_P` as the mass flux, one source of truth)
  solved by a **fixed** damped-Jacobi sweep (`damped_jacobi_solve` — constant operator, so plain
  left-preconditioned GMRES suffices, no FGMRES). **Measured (skewed cavity):** outer GMRES per Newton
  step 92→11 (432 dof), 191→19 (768), 309→25 (1200) — **~8–12× fewer iterations**, an exact drop-in
  (update diff ~1e-7) and adjoint-transparent (`tests/{unit/test_preconditioner,integration/test_flow_preconditioner}.py`).
  The count still *grows* with mesh because the Jacobi inner is not h-independent — **Stage 2 (a
  fixed-cycle multigrid inner, built once off-jit and frozen) is what flattens it at >1M cells.**

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
  approximation.** Default inner Schur solve is a **smoothed-aggregation multigrid** V-cycle
  (`aquaflux/solve/multigrid.py`, `BlockPreconditioner.build(asm, inner="smoothed")`; `"multigrid"` =
  unsmoothed fallback, `"jacobi"` = Stage-1 fallback). Built once off-jit with `scipy.sparse` (no PyAMG needed),
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
