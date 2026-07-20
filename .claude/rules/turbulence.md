---
paths:
  - "aquaflux/turbulence/**"
---

# Rules — `aquaflux/turbulence/` (RANS closure: k–ω SST + the segregated coupling)

> **Provenance boundary (binding).** This file cites the internal design record
> (`turbulence-design-note.md`) and the precursor codes to inform *your* understanding — that
> is its job, and why it loads into your context. Per the root `CLAUDE.md` **Comment
> Convention**, none of that provenance may reach the shipped surface (`.py`
> comments/docstrings, `docs/`): cite the *math*, never the reference code, the `.claude/`
> rules, the design notes, or the author's own papers.

The k–ω shear-stress-transport (SST) closure and the loop that couples it to the coupled p–U
flow. The forward coupling is **segregated** (an outer Picard loop, decided with the author) —
but segregation is a *forward-solve strategy only*; the differentiable promise still requires
the adjoint of the **unfrozen coupled residual**. Governed by the root `CLAUDE.md` Engineering
Principles; the flow block it feeds is `.claude/rules/flow.md`, and the Newton / linear-solve
adjoint machinery it must reuse is `.claude/rules/solve.md`.

## Status — BUILT (segregated forward solve **and** monolithic coupled solve + coupled adjoint)
- **`sst.py` — `SSTModel`.** Menter's SST constants and the quantities derived directly from
  them (the F₁/F₂ blend, the eddy-viscosity limiter).
- **`strain.py`** — the strain-rate magnitude `S = sqrt(2 S_ij S_ij)` the production terms read.
- **`sources.py`** — the k and ω production / destruction / cross-diffusion terms as
  `VolumeSourceFn` volume-source operators (the transport equations reuse the shared advection
  and diffusion flux operators; only the sources are turbulence-specific).
- **`transport.py` — `SSTTurbulence`, `SSTClosureFields`.** Assembles the k and ω scalar
  transport residuals on the flow's Rhie–Chow mass flux, with μ_t a **frozen per-cell field**
  recomputed once per outer sweep.
- **`preconditioner.py`** — the convection-diffusion AMG preconditioner for the stiff k/ω scalar
  Krylov solves at high Reynolds number (the scalar analogue of the velocity-block work). It assembles
  its frozen operator with the shared `aquaflux.solve.frozen_operator.convection_diffusion_operator` and
  hands the **assembled matrix** to `build_convection_hierarchy` / `build_air_hierarchy` (the coarsening
  library is operator-in, #45); its reaction+boundary diagonal still comes from its own `J·1`
  derivation, which is a genuinely different source, not a copy of the interior stencil.
  `scalar_transport_preconditioner` returns a **`ScalarTransportPreconditioner`** strategy
  (`ConvectionAmgPreconditioner` / `AirAmgPreconditioner`) rather than the old opaque `lambda phi: solve`.
  These are plain frozen dataclasses, **not `equinox.Module`s** — see the binding note in
  `.claude/rules/solve.md`; making them pytrees breaks both the IFT adjoint and the jit cache.
- **The scalar policy's two halves have different lifetimes (binding, #105).** `ScalarShiftPolicy` carries
  a **shift diagonal rebuilt every sweep** (so the pseudo-time damping keeps tracking the operator as
  ν_t grows — freezing it would under-damp the march and lean on `DivergenceGuard` escalation) and an
  **AMG preconditioner built once and carried** (it only accelerates the Krylov iteration, and rebuilding
  it per sweep cost ~0.9 s (k) + ~1.0 s (ω) at 4k cells *and* re-compiled the whole solve every sweep).
  `SSTTurbulence` therefore splits `k_preconditioner`/`omega_preconditioner` (frozen, `method=`) from
  `k_shift_policy`/`omega_shift_policy` (per sweep, `preconditioner=`); `solve_segregated` builds the
  former on the first sweep and the latter every sweep. Measured: traces per sweep went `[5,5,5,5,5]` →
  `[5,5,0,0,0]` with the converged field bit-identical. Pinned by
  `test_a_carried_preconditioner_compiles_the_scalar_solve_once`.
- **`transport.py`'s `omega_residual` returns a `WallFixedResidual`, not a closure (binding, #105).** It is
  rebuilt every sweep and passed into the jitted scalar solve, so as a bare closure it landed on
  `filter_jit`'s static side and identity-missed the cache every sweep. As an `equinox.Module` its arrays
  ride on the traced side and only their *values* change. (`k_residual` already returned a bound
  `ResidualAssembler.residual`, which equinox treats as a pytree — that one was always fine.) Note the
  contrast with the preconditioner above: a *per-sweep* callable must be a pytree, a *frozen* one must not.
- **`boundary.py`** — inlet/wall closures for k and ω over the generic scalar boundary machinery.
  - **The wall ω is `6ν/(β₁d²)`, NOT `60ν/(β₁d²)` (binding — the constant depends on where it is
    imposed).** `omega_wall_value` is fixed at the wall-adjacent **cell centroid** (`FixedValueCells`
    at `wall_distance[wall_cells]`), so it must be the analytical sublayer solution *at that distance*:
    `ν d²ω/dy² = β₁ω²` with `ω = A/y²` gives `A = 6ν/β₁`. The `60` form is a wall-**face** value — 10×
    the asymptote, standing in for the singularity at `y = 0` (Menter, 1994) — and was being imposed at
    the cell centroid, putting near-wall ω 10× high (suppressed near-wall `ν_t`, stiffer ω equation, and
    a realized κ below the reference). Do **not** "restore" the 60 without also moving the imposition to
    the wall face. The unit test pins the **ODE residual**, not the coefficient, so the two forms cannot
    be swapped silently again.
- **`driver.py` — `solve_segregated`.** The outer Picard loop: μ_t → flow solve → k solve → ω
  solve, with under-relaxation and positivity floors as the stabilizers, and injected
  `solve_flow` / `solve_scalar` so the driver is pure orchestration. The per-sweep coupling is
  `momentum.with_eddy_viscosity(ν_t)` — the driver hands over the closure's **kinematic** `ν_t` and
  the flow assembler forms `μ_eff = μ + ρν_t` from its own material properties, so the driver never
  restates the closure relation and takes **no `density=` argument** (see `.claude/rules/flow.md`).
  An injected momentum stand-in must therefore provide `with_eddy_viscosity`. The loop **stops on the coupled
  Picard increment** (`_relative_change` — the largest per-field relative L2 change over a sweep <
  `rtol`), with `max_sweeps` only a backstop; the outer under-relaxation is the **SER ramp**
  `_sweep_relaxation` (opens from the `relaxation` floor toward `relaxation_max` as that increment
  falls, constant when `relaxation_max is None`). Hitting `max_sweeps` without converging warns.
  - **Flow-solve seam is `solve_flow(momentum, state) → (momentum, state)` (binding).** The flow solve
    returns the assembler as well as the state, because a **bulk-velocity-constrained** solve
    (`flow.bulk_velocity_flow_solve`) carries its converged body force out on the assembler — so a
    mass-flow-driven periodic channel needs **no separate controller**, the constraint is enforced
    inside the flow Newton. The old inline **proportional mass-flow controller was DELETED** (its
    `bulk_velocity_target`/`bulk_velocity_gain`/`flow_direction` args gone): it updated β *after* a
    fixed-β flow solve, so at high Reynolds / high aspect ratio it measured a bulk velocity that had
    already spiked ~17× (β tripled while μ_t was stale) and collapsed the near-wall `k` onto its floor.
    The bordered solve makes `⟨U⟩ = U_bar` hold by construction; see `.claude/rules/flow.md`. An
    unconstrained `solve_flow` returns the assembler unchanged.
  - **The sweep body between the injected solves is jitted and assembles the flow fields once
    (binding, #106).** The pre-solve μ_t and the post-solve `(mdot, closure)` run in two module-level
    `eqx.filter_jit` prologues (`_sweep_eddy_viscosity`, `_sweep_closure`) instead of op-by-op eagerly
    (the eager path dispatched `velocity_gradient` / `mass_flux` / `closure_fields` one op at a time —
    ~130 ms/sweep of avoidable overhead at 1600 cells). `_sweep_closure` calls
    `momentum.flow_fields(flow)` **once** for both the velocity gradient the closure reads and the
    Rhie–Chow `mdot` the scalars advect on (the pre-solve μ_t uses the lightweight `velocity_gradient`,
    which is all it needs before `mdot` exists). `solve_segregated` binds the k/ω boundaries once via
    `turbulence.resolve_boundaries()` before the loop, so those compiled prologues never re-run the
    dynamic-shape patch resolve inside `closure_fields`'s gradient assembler. Bit-identical to the old
    eager path; pinned by `test_segregated_prologues_match_the_eager_assembly`.
- **`coupled.py` — `CoupledRANS`, `solve_coupled` (Option 2, the target engine).** The monolithic
  residual `R(u, p, k, ω)` over the flat `[flow…, k, ω]` state (`CoupledRANSLayout`, whose `unpack`
  yields the momentum block's own `[u,p]` sub-vector so `MomentumContinuity` runs on it unchanged),
  with **nothing frozen**: μ_t, the strain `S(u)`, the Rhie–Chow flux, and the closure are live, so
  one Newton solve sees the exact cross-block Jacobian. Globalized by `coupled_continuation`
  (a block `CoupledShiftPolicy` = velocity `a_P` shift ⊕ the k/ω transport-diagonal shifts, and a
  block-diagonal preconditioner gluing `BlockPreconditioner` to the two scalar CD-AMGs; the AMG
  hierarchies + numpy-built scalar shift diagonals **frozen at a reference state** off-jit à la
  `reused_flow_solve`, the velocity `a_P` live). Handed to `ImplicitNewtonSolver`, it gives the
  **exact coupled adjoint** (§5) — a single transpose solve on the unfrozen `R_coupled`. The ω wall
  rows are `FixedValueCells`. `CoupledRANS.build` pre-resolves the k/ω boundaries (via
  `turbulence.resolve_boundaries()`, the shared idempotent bind the segregated driver also uses) so the
  per-eval assembler rebuild's `resolve` is an idempotent no-op (else a dynamic-shape `nonzero` on
  traced mesh labels breaks the jit). **`CoupledRANS.residual` assembles the Rhie–Chow flow fields once
  (#106):** it builds the `closure` first and takes `nu_t` from it (rather than a separate
  `eddy_viscosity` recomputing the same strain), then one `momentum.flow_fields(flow)` feeds both
  `residual_from_fields` and the `mdot` the scalars advect on — was 3× `_flow_fields` per eval, ~1.85×
  the trace/compile and AD-tape size. **Direct k/ω variables** (log-variables held in reserve, below); positivity
  under a full Newton step is carried by the pseudo-transient shift + divergence guard, no in-residual
  floor. FD-verified: coupled ‖R‖→machine-zero, agrees with the segregated fixed point, adjoint
  matches finite differences.
- **`initialization.py` — `hybrid_initialize` (cold-start, the reason `solve_coupled` self-starts).**
  The monolithic Newton is a *local* method: from a raw cold start (`u=0`, uniform k/ω) it **stalls** —
  the near-wall ω fixation alone injects a `~6ν/(β₁d²)` jump, and a uniform interior is far from a
  consistent field the inner solve can precondition. `hybrid_initialize(momentum, turbulence)` builds a
  cheap physical IC (a few linear Laplace solves): **potential-flow velocity** (`flow/initialization.py`
  `potential_flow`), **Laplace-smoothed k** (harmonic interpolant of its BCs), and **ω** =
  boundary-propagated interior with the near-wall cells set to the analytical wall value (a *Laplace*-ω
  over-diffuses that large value and slows the solve — set only the wall cells). From this IC the coupled
  Newton converges from nothing (~10–15 steps, FD-verified). `solve_coupled(coupled)` with no initial
  state calls it automatically; the segregated pre-smooth is no longer required to reach the basin (still
  available as a fallback). **Do not init with an exactly symmetric velocity** — a perfectly symmetric
  `u` (e.g. exact plug, `u_y≡0`) hits a measure-zero degeneracy in the coupled inner solve that stalls;
  the potential flow's discrete-gradient roundoff (`|u_y|~1e-10`) lifts it, and any perturbation ≥1e-10
  converges. The IC is a forward device (the converged-state adjoint is IC-independent); when
  differentiating, pass an explicit state built outside `jax.grad`.
  - **Body-force-driven domains need equilibrium levels, not interpolants (binding).** A
    streamwise-periodic channel has **no inlet**, so both smoothed fields are degenerate: `k` is the
    harmonic interpolant between all-zero wall Dirichlets (**identically zero**), and `ω` is a
    pure-Neumann solve whose interior carries nothing. Left alone that starts the solve at `k=0` →
    `ν_t=0` — not a poor guess but the **laminar** problem, which for a turbulent case is the wrong
    equations. Both levels therefore come from the **friction velocity the force balance fixes**,
    `u_τ = √(βh/ρ)` (`flow/scales.py::friction_velocity`, `h = V/A_wall`): `k = u_τ²/√β*`
    (`boundary.py::equilibrium_k`) and `ω = inlet_omega(k, 0.09h)`, applied with `jnp.maximum` so an
    inlet-driven domain (whose `u_τ` is zero) is bit-unchanged. **Fix k and ω together or not at
    all** — raising `k` while `ω` sits at its `1e-8` floor gives `ν_t = k/ω ~ 10⁶`, far worse than the
    laminar start. The length scale is the **outer mixing length `0.09h`**, not the `0.07·D_h`
    inlet-specification convention: the latter is for an inlet, and here overshot the developed-channel
    `ν_t` by ~3.5× (measured `ν_t/ν` 373 vs the correct 120 = `0.09u_τh/ν`, which the shipped default
    now hits exactly). Pinned by `test_hybrid_initialize_gives_a_developed_channel_eddy_viscosity`.

**Issue #69 — CLOSED path (do not re-derive without reading it):** all three planned steps shipped —
scalar continuation (#73), Option 1 hardening (convergence stop + adaptive relaxation), and Option 2
(the monolithic coupled residual + its IFT adjoint, the target engine). The segregated loop is
**retained as a forward pre-smoother / fallback**, not the sensitivity model; for gradients use the
coupled `solve_coupled` (its adjoint is exact) — never differentiate `solve_segregated` (forward-only,
unrolls the Picard sweeps, which §5 forbids). The remaining held-in-reserve item is the **log-variable
fallback** (below), promoted only if a stiff high-Re case shows the direct coupled form non-robust.

## Binding decisions

- **Segregated forward, coupled adjoint (design note §5 — binding) — BUILT via `solve_coupled`.**
  Segregation is a **forward-solve strategy only**. For exact sensitivities the adjoint is the
  implicit-function-theorem solve on the full **unfrozen** coupled residual
  `R_coupled(k, ω, U, p; params) = 0` at the converged state — the `solve/` two-level
  implicit-diff machinery — **not** a differentiation of the Picard iteration. This is now realized:
  `coupled.py`'s `CoupledRANS.residual` **is** that unfrozen `R_coupled`, and `solve_coupled` hands it
  to `ImplicitNewtonSolver`, whose adjoint is a single transpose solve (FD-verified). At the fixed
  point the frozen fields equal the live values, so the coupled residual is satisfied and its
  adjoint is exact; the segregated outer loop is a forward convergence device that is **absent from
  the sensitivity model**. Differentiate **`solve_coupled`, never `solve_segregated`** (the latter is
  forward-only and its docstring says so). When building the coupled continuation for a differentiated
  solve, construct it **outside `jax.grad`** (concrete preconditioner params) and pass it in — see the
  flow preconditioner's same constraint.

- **Never unroll the outer loop onto the differentiation path.** A fixed-count `for` over sweeps
  that is differentiated directly is exactly the failure `solve.md` names ("no loops on the
  differentiation path"). If the coupled solve is not yet wrapped in the coupled-residual IFT
  adjoint, it is **not done** — it is an intermediate step (Principle 0), and the deferred adjoint
  must be filed as a tracked issue at merge time, not left implicit.

- **Convergence-based outer stop, not a fixed sweep count — BUILT.** The loop tests the coupled
  Picard increment and stops on it (`rtol`), with `max_sweeps` only a backstop and a warning when the
  cap is hit unconverged. Do **not** reintroduce a hard-coded `sweeps` count. The increment measure
  is the residual-agnostic per-field relative change, not a raw combined norm (the field scales
  differ by orders of magnitude).

- **Globalize the outer loop and the scalar sub-solves like everything else.** The flow block is
  globalized by pseudo-transient continuation; the k/ω transport sub-solves and the outer coupling
  must reach the same standard (a scalar `ShiftPolicy` continuation on the transport diagonal for the
  sub-solves; adaptive under-relaxation — **the SER ramp is built** — with Aitken/Anderson or a
  monolithic coupled residual as the further steps, for the loop). Constant under-relaxation plus
  positivity floors is the *stabilizer of last resort*, not the globalization.

- **Positivity floors must be inactive at convergence (adjoint honesty, design note §3.3 —
  binding).** `k ← max(k, k_floor)`, `ω ← max(ω, ω_floor)` and the `CD_kω` / F-blend floors have zero
  gradient in the clamped region; they pollute the sensitivity **unless inactive at the fixed point**
  (`k, ω > floor` everywhere, which holds for any properly resolved RANS field). State this precondition
  in code and **check it**: if a case converges with a floor active, the sensitivity through that cell is
  wrong — surface it, do not ship it. (Log-variable `k = e^{k̃}` is the held-in-reserve structural fix.)

- **Frozen coupling data rides as injected pytree leaves** (μ_t, the frozen ∇u, mdot), the same
  blessed mechanism the coupled solver already uses to inject `mdot` — no new freezing mechanism, and
  no re-coupling μ_t ↔ (k, ω) inside a residual via a `Calculated` property in the segregated path.

## Testability seam
- `solve_segregated` takes **injected** `solve_flow` / `solve_scalar` closures, so the loop is
  tested against trivial stub solvers (e.g. identity / one-step) with a known fixed point — no full
  coupled solve needed to test the orchestration.
- Every turbulence operator (sources, strain, transport) ships an operator-level unit test on an
  analytic field (Principle 1), independent of the coupled solve.
- **The coupled solve needs an adjoint-correctness gate**, not only a smoke test: a test that
  `jax.grad` through the converged coupled turbulent solve is **iteration-count-independent** (the
  coupling analogue of Gate C; see the root `CLAUDE.md` Testing Architecture). An existence check
  ("stays stable, fields positive, μ_t active") does not establish the adjoint.
- **Never assert the positivity floors back (binding — they are tautologies).** `solve_segregated`
  clamps every sweep with `jnp.maximum(k, k_floor)` / `jnp.maximum(ω, ω_floor)`, so `min(k) >= 0`
  and `min(ω) > 0` hold for a *diverged* field exactly as well as a converged one. Likewise
  `max(μ_t)/ν > 1` is reached within a single sweep. A test built from these asserts only that the
  process did not crash. `test_high_reynolds_turbulent_channel_solves` was exactly this shape and
  was deleted rather than tuned: it cost ~45 minutes and its four assertions were all near-free,
  while its docstring claimed an isolation (unpreconditioned scalar solves) that the code
  contradicted. **A segregated-loop test must assert convergence** — that the Picard increment
  actually reached `rtol` (the driver only `warnings.warn`s otherwise, and returns the
  under-converged fields), or that the result matches an independently converged reference. The
  model is `test_coupled_rans.py`, which drives the loop to `rtol=1e-9` and asserts it reaches the
  coupled solve's fixed point to 1e-4.

## Post-change
Keep this file's Status and Binding decisions true as the coupling globalization (issue #69) lands —
per the root `CLAUDE.md` Post-Change Checklist's Documentation-sync item.
