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

## The gradient scheme is bound PER FIELD, at build (2026-09-22)

`SSTTurbulence.build` binds the injected scheme twice — `k_gradient_scheme` and
`omega_gradient_scheme` — against each field's own conditions, the shape `MomentumContinuity` has
carried since #469. Two reasons, either sufficient:

- **The conditions differ.** A wall prescribes `k` and lets `omega` extrapolate, so the linearization
  a scheme prepares itself from is not the same for the two equations; one binding applied to both is
  wrong for one of them, silently (both still reconstruct).
- **`_assembler` runs inside the residual.** Every k/omega assembler is built per evaluation, so a
  scheme bound there is bound against a **traced** mesh. A scheme whose preparation reads the
  connectivity (`ProjectedStencilGradient`) cannot be, and raises; binding at build is what lets the
  turbulence closure use one at all. `ResidualAssembler.build(..., bind_gradient_scheme=False)` is
  how the pre-bound scheme is passed through.

⚠️ **Storing the bindings is half of it — each equation must READ its own**, and that half is what a
test looking at the stored schemes cannot see. `tests/unit/test_sst_transport.py` pins it by
substitution (a stub scheme reconstructing zero), which needs two things to discriminate: a **frozen**
closure, because the blending function reads both gradients and would otherwise carry one field's
swap into the other's equation through the physics; and a **perturbed** mesh, because on an
orthogonal one the diffusion's non-orthogonal correction vanishes and the k equation never reads its
gradient at all. Mutation-verified four ways.


## The advection scheme is bound PER FIELD too, for its field's magnitude (2026-10-09, #144)

`SSTTurbulence` carries `k_advection_scheme` and `omega_advection_scheme` — **there is no
`advection_scheme` field any more**; `build` still takes one positional scheme and binds it twice
with `with_reference_scale` (see `schemes.md`). A scheme that reads no scale (first-order upwind,
what both step cases run) comes back unchanged in both slots; a softened Venkatakrishnan limiter gets
`k_ref` in one and `omega_ref` in the other, because the two fields differ by units and by orders of
magnitude. `build(velocity_scale=...)` supplies the flow's speed; the case's `RANS.build` passes
`flow.reference_speed(momentum)`. Unset, a scaled scheme is refused at build naming the keyword; an
unscaled one needs nothing. Pinned by substitution
(`test_each_turbulence_equation_advects_with_its_own_scheme`) and by the case build
(`test_a_limited_turbulence_advection_takes_k_and_omega_scales_from_the_flow_s_speed`), both
mutation-checked.

**The scales (`turbulence/scales.py`, `turbulence_scales(U, h, model)`), decided with the project
owner, who asked for them to come from the velocity rather than the inlet turbulence** — an inlet's
`k`/`omega` is not representative of the domain (pitzDaily's inlet k is 0.375; OpenFOAM's shear-layer
peak is ~5.0):
- **`k_ref = 1.5 (0.1 U)² = 0.015 U²`** (`REFERENCE_INTENSITY = 0.1`, through `inlet_k`). Peak `k`
  in developed flows is roughly 0.01 U² (plane channel, bulk U; peak k⁺ ≈ 4.5 at u_τ ≈ 0.05 U_b) to
  0.05 U² (mixing layer / separated shear layer); 0.015 sits at the low end, where an error makes the
  limiter stricter, which is the safe side for `k`'s positivity. pitzDaily: `k_ref` = 1.5 against the
  OpenFOAM peak of 5.0 (U = 10).
- **`omega_ref = sqrt(k_ref) / (β*^¼ · 0.09 h)`**, `h = flow.wetted_length(mesh, geometry,
  wall_patches)` = V / A_wall (the same hydraulic length `flow.hydraulic_length` and the body-force
  initializer use), through `inlet_omega`; `OUTER_MIXING_LENGTH_FACTOR = 0.09` is now the one home of
  the 0.09 `sst_initial_fields` defaults to. It is the **outer-flow** level on purpose: `omega`'s
  range is set by the first cell height (`~6ν/(β₁y₁²)`, pitzDaily OF range 160–1.1e5), so a range-based
  scale would be mesh-dependent and would turn limiting off everywhere but the near-wall cells.
  pitzDaily estimate (h ≈ 0.02 m, not measured from the mesh): `omega_ref` ≈ 2000 /s against shear-layer
  values ~10³ and the inlet's 440.
- **No wall** (h = 0) or **no speed** → refused, only when a scaled scheme asks. A wall-less RANS
  domain with a limited turbulence advection must state the scale on the limiter itself.
- **Not measured on any case.** Both step cases run first-order turbulence advection, so nothing
  shipped reaches these scales; the choice is argued, not calibrated. The alternative considered and
  not built: a softening relative to the LOCAL value (`eps = K |phi_i|`), which suits ω's dynamic range
  but collapses for `k → 0` at walls.

## How to read this file (read this before grepping it)

Same three rules as `.claude/rules/solve.md`: **every entry sits under a `##` section** (scan up to the
nearest one); **a superseded entry is DELETED, never struck through or annotated in place** (`~~tildes~~`
are invisible to `grep`, and "see above" expresses supersession by adjacency, which a hit does not
carry); and **a measurement without its configuration is unfalsifiable** — name the case, state,
bundle and shift, or the number cannot be re-adjudicated when a default moves.

**Check any symbol or default against the source before quoting it.** Stale entries here have been
lifted by `grep` and asserted as current. See `CLAUDE.md` → **Stale-Record Check**.

## ⚠️ Generic machinery does NOT live here (2026-09-19, #448 / #277) — read before adding to `coupled.py`

`coupled.py` used to hold the whole robust march, and a laminar flow problem could not reach any of it
(#448). The residual-agnostic parts now live in `aquaflux/solve/`, and the flow-only solve
(`aquaflux.flow.solve_flow_march`) runs on the same ones. **What moved, and where it is now:**

| was in `turbulence/coupled.py` | is | note |
|---|---|---|
| the segment loop, the once-measured stopping target, the last-segment-without-trigger rule, the non-root refusal in `solve_coupled` | `solve.staged_march` (`solve/driver.py`) | `solve_coupled` now supplies the residual, `_CoupledMeasures`, the `eddy_viscosity_drift` measure and a source |
| `_ContinuationSource`, `_CallerBuiltContinuation`, `_FinishedContinuation` | `solve.ContinuationSource`, `CallerBuiltSource`, `FinishedSource` | `_SessionContinuation` stays: it wraps a coupled session |
| `_refuse` and the strategy/builder branch of `_continuation_source` | `solve.explicit_source`, `solve.driver.refuse_unforwardable_settings` (not exported since #541 left `solve/` its only caller) | the raise now names the caller passed in |
| the tail of `_coupled_step` (dual-time vs single step, the hook and refresh refusals, the default `relative_residual_gmres(norm=None)`) | `solve.shifted_step` (`solve/shifted_step.py`) | `_coupled_step` keeps only the coupled operator stand-in and passes `line_search=_COUPLED_LINE_SEARCH` |
| `_LinearSolveRegime`, `_resolved_regime`, `_resolved_linear_solve`, and `LinearSolveSettings` from `march_settings.py` | `solve.LinearSolveRegime`, `solve.resolve_linear_solve`, `solve.LinearSolveSettings` | `aquaflux.turbulence.LinearSolveSettings` no longer exists; the per-family regime constants (`_BLOCK_LINEAR_SOLVE`, …) stay here because they are calibrations of *these* preconditioners |
| `CoupledJacobianProbe` (plan + gather map), `_coupled_jacobian_plan`'s graph logic | `precondition.JacobianProbe`, `precondition.jacobian_probe_plan` (`precondition/jacobian_probe.py`) | the probe holds a `narrowing` callable for the assembler stand-in; `coupled_jacobian_probe(coupled, …)` builds one with `_CoupledNarrowing(gradient_sweeps, production_viscosity_frozen)`. `probe.gradient_sweeps` is now `probe.narrowing.gradient_sweeps`. `_coupled_jacobian_plan` stays as the coupled adapter (mesh graph + layout) |
| `MonolithicFactorShiftPolicy`, `FrozenTransposeFactory` | `precondition.MonolithicFactorShiftPolicy`, `precondition.monolithic_policy.FrozenTransposeFactory` (`precondition/monolithic_policy.py`) | `base` is any `ShiftPolicy`; no longer exported from `aquaflux.turbulence` |
| `_MaterializedSession`, `_beta_tracking_refresh`, `PreconditionerSession`, `_jacobian_matvec`, `_batched_jacobian_matvec`, `_frozen_shift_diagonal`, `_PROBE_BATCH_SIZE`, `_BUILD_BETA`, `_FACTORIZATION_LINEAR_SOLVE`, `_VCYCLE_LINEAR_SOLVE`, `_is_traced` | `precondition.MaterializedSession`, `BetaTrackingRefresh`, `solve.PreconditionerSession` (`solve/session.py`), `jacobian_matvec`, `batched_jacobian_matvec`, `frozen_shift_diagonal`, `PROBE_BATCH_SIZE`, `BUILD_BETA`, `VCYCLE_LINEAR_SOLVE` (`precondition/materialized_session.py`; `FACTORIZATION_LINEAR_SOLVE` was deleted with the complete LU, 2026-10-11) | the session is written against `precondition.MaterializedProblem`; coupled RANS supplies `_CoupledProblem` (assembler, probe, `[flow] / [k, omega]` groups, `_monolithic_shift_source`, `_monolithic_factor_step` with the `k` positivity guards). `_beta_tracking_refresh(coupled, stencil_reach, …, probe=)` is now the class `BetaTrackingRefresh(assembler, probe, preconditioner, refit_beta_floor=, observer=)` (a function `beta_tracking_refresh` until #661; `every_step=` went with the complete LU, 2026-10-11). Since #290 `_monolithic_factor_step(coupled, reference_state, policy, …)` takes the shift policy the session glued (traced split or host factor) rather than building one |
| `MaterializedJacobian`, `FieldSplit`, `JacobianProbeSpec` (and `MonolithicVCycle`, deleted with PETSc 2026-10-10, and `CompleteLu`, deleted 2026-10-11) | the same names in `aquaflux.precondition` (`precondition/materialized_spec.py`) | no longer exported from `aquaflux.turbulence`; `BlockDiagonal` stays here. The registry `_SPEC_MAPPING` **extends** `precondition.MATERIALIZED_MAPPING` rather than restating its kinds; a laminar spec file loads with `precondition.materialized_spec_from_mapping` |
| `_SessionContinuation` | `solve.SessionSource` | shared with the flow march |
| the flow rows of `coupled_scaled_norm`, and the per-block reference scales | `flow.flow_row_scales`, `solve.block_reference_scales` | `coupled_scaled_norm` appends its `k`/`ω` rows to the flow's |

**Still here**: `open_session` and `_BlockSession` (the block-diagonal family is turbulence's own). The mass-flow border moved to `flow/bordered.py` (#541): `coupled_border` and `_BorderedCoupledResidual` are its coupled adapters. The materialized-Jacobian session, the probe, the monolithic shift policy and the specs have moved to `precondition/` (via `solve/`, which held them until #284 part 1), and `solve_flow_march` uses them (`MaterializedJacobian` with a bare `BlockInverse`). The size ratchet in `tests/unit/test_layering.py` is what to lower as they move.
⚠️ **Do not add a capability to the march here without checking `solve_flow_march` can reach it** — that is
the drift this section exists to prevent (root `CLAUDE.md`, Principle 3.6).

## ⚠️ The coupled builders were unified (2026-09-14, #371) — these names no longer exist

Many entries below are dated history written against the old API. Read them through this table:

| was | is |
|---|---|
| `coupled_continuation(coupled, state, method=M, **flow_opts, **march)` | `coupled_step(coupled, state, preconditioner=BlockDiagonal(scalar=S, **flow_opts), **march)` |
| `coupled_lu_continuation(..., lu_beta=b, backend=B, stencil_reach=r, ...)` | *nothing* — the complete LU was deleted 2026-10-11 (its `backend` the day before, with PETSc); use `MaterializedJacobian(FieldSplit(L, T), build_beta=b, probe=JacobianProbeSpec(stencil_reach=r))` |
| `coupled_amg_continuation(..., smoother_fill_levels=…, amg_beta=b)` | *nothing* — the monolithic PETSc V-cycle (`MonolithicVCycle`) was deleted 2026-10-10; use `MaterializedJacobian(FieldSplit(L, T), build_beta=b)` |
| `coupled_amg_continuation(..., field_split=True, leading_inverse=L, trailing_inverse=T)` | `MaterializedJacobian(FieldSplit(L, T))` — `L`/`T` are `precondition.BlockInverse` values |
| `probe=` / `preconditioner=` shared across rungs, `amg_beta_tracking_refresh(..., beta_floor=f, observer=o)`, `lu_beta_tracking_refresh` | one session: `open_session(MaterializedJacobian(..., refit_beta_floor=f), coupled, observer=o)`, passed as `solve_coupled(preconditioner=session)`; its `refresh_preconditioner` / `rebind` replace the hooks' |
| `reuse=previous.shift_policy, residual_norm=m` | `session.refresh(state, previous, **march)` — since #370 the measure is not passed; the march hands every step its own |
| `solve_coupled(rtol=, atol=, scaled_norm=)`, `solve_coupled_mass_flow(rtol=, atol=)` | `convergence=Convergence(measure=…, rtol=…, atol=…)` (#370); `scaled_norm` is gone because the row-scaled measure is now always rebuilt |
| `coupled_step(block_scaled_norm=True)` / `(residual_norm=m)`, the same on `mass_flow_coupled_continuation` | nothing on the builder: `solve_coupled(convergence=Convergence(measure=BlockScaled()))` (#370) |
| `solve_reynolds_continuation(intermediate_rtol=…, intermediate_atol=…)` | `intermediate=Convergence(rtol=…, atol=…)`, unset fields from the target's `convergence`; `None` = no loosening (#370) |
| `solve_coupled(method=M, velocity=…)` | `solve_coupled(preconditioner=BlockDiagonal(scalar=S, velocity=…))` |
| `mass_flow_coupled_continuation(..., method=M, **flow_opts)`, `solve_coupled_mass_flow(method=M, **flow_opts)` | both **deleted (#541)**: `coupled_step(coupled, state, preconditioner=BlockDiagonal(...))` and `solve_coupled(coupled, ...)` border themselves when `coupled.momentum.drive` is a `MassFlow`; the held force is `held_force(coupled, flow, k, omega)` |
| `BlockDiagonal(velocity="convection")`, `"smoothed"`, `"convection-air"` (and the same strings on `BlockPreconditioner.build`, `momentum_continuation`, `reused_flow_solve`) | `ConvectionTwoLevel()`, `ViscousMultilevel()`, `ConvectionAir()` from `aquaflux.flow` (#390); a string is refused |
| `shift_basis=…`, `velocity_shift_parts=…`, `turbulence_damping=…` on `coupled_step` / the mass-flow builder / `solve_coupled` | `shift=CoupledShiftSettings(basis=…, velocity_parts=…, turbulence_damping=…)` (#387; since #450 stage 3 it subclasses `solve.ShiftSettings(basis, velocity_parts)`, the flow's part, and adds `turbulence_damping`; `ShiftSettings` from `aquaflux.turbulence` no longer exists); a Reynolds `point_setup` value merges field by field over the shared one (so does a `Globalization`; an unset field takes the shared setting, so a point cannot reset one to its default). `merged_march_options` merges any value that is a `solve.MergeableSettings` (`isinstance`, #669), not anything with a `filled_from` attribute |
| `inner_steps=N` (N > 1), `inner_tol=…`, `cycle_budget=…`, `refresh_on_cycles=…` | `dual_time=DualTimeLoop(inner_steps=N, …)` (#388); unset is the single shifted step. `inner_observer` without a loop is refused where it used to be dropped, and a `refresh_on_cycles` with no refresh hook to take its re-fit (a frozen step) is refused by `newton_march` before the first step, and by a block-diagonal session when it builds the step (its hook only follows a re-pointing, #386). There is no `inner_refresh` (removed 2026-10-10, #290): the step ENDS for the re-fit and the march takes it |
| `krylov_solver=S`, `forward_rtol=…` / `forward_restart=…` / `forward_max_restarts=…` | `linear_solve=S` or `linear_solve=LinearSolveSettings(rtol=…, restart=…, max_restarts=…)` (#388) — one slot, so a solver beside a regime setting cannot be written |
| the scalar multigrid string `M` = `"twolevel"` / `"air"` / `None` — on `BlockDiagonal.method`, `scalar_transport_preconditioner(method=)`, `SSTTurbulence.k_preconditioner`/`omega_preconditioner(method=)`, `solve_segregated(scalar_preconditioner=)`, and `_coupled_shift_policy(coupled, state, M)` | a `ScalarBlock` value `S` = `ScalarTwoLevel()` / `ScalarAir()` / `UnpreconditionedScalars()` from `aquaflux.turbulence` (#394), on `BlockDiagonal.scalar` and a `scalar=` keyword everywhere else; `_UNSET` and `resolved_method` are gone (`resolved_scalar`); a string is refused |
| `point_setup` returning `strategy` + `RefreshPolicy(refresh_preconditioner=hook)`, `_rebinding` | `solve_reynolds_continuation` / `solve_reynolds_ramp` given `preconditioner=`; `point_setup` keeps only per-point march settings |
| `solve_segregated(rtol=)` | `increment_tol=` (#373) — it is a tolerance on the coupled Picard **increment**, not on a residual |
| `scalar_pseudo_transient_solve(max_steps=, rtol=, atol=, solver=)` | `root_solve=RootSolveSettings(max_steps=…, convergence=…, linear_solver=…)` (#428); its `adjoint_solver` reaches the driver but is inert, since this solve is forward-only |
| `MaterializedJacobian(beta_floor=)` | `refit_beta_floor=` (#373) — the shift the inverse is re-fitted at, against the march's own `Globalization.beta_floor` |

## The closure — model, strain, sources, transport, preconditioner

- **`sst.py` — `SSTModel`.** Menter's SST constants and the quantities derived directly from
  them (the F₁/F₂ blend, the eddy-viscosity limiter).
- **`strain.py`** — the strain-rate magnitude `S = sqrt(2 S_ij S_ij)` the production terms read.
  - **The `sqrt` is guarded at `S = 0` (binding, `_safe_sqrt`).** `S` is a Euclidean norm, so it has a
    cone point at zero (like `|x| = sqrt(x²)`): the value is continuous but the `sqrt` chain rule is
    `dS = dq/(2S) = 0/0 = NaN` there. A **uniform** velocity region has `S = 0` *identically* (zero
    gradient), and a body-force periodic channel's hybrid IC is the exactly-uniform plug
    (`scales.body_force_velocity`, `u_y ≡ 0`), so **every interior cell** was `S = 0` and the coupled
    Jacobian came back NaN in all of them — the monolithic Newton then stalled immediately (every step
    NaN → `DivergenceGuard` rejects → escalates to the cap → `max_steps` → raises). The double-`where`
    `_safe_sqrt` clamps the `sqrt` argument to 1 on the branch discarded at zero, so `dS = 0` (the
    minimum-norm subgradient) at exactly `S = 0` while the value and derivative are **bit-identical to a
    plain `sqrt` wherever `S > 0`** (verified: forward value, jvp, and jacrev all bit-equal). Returning
    `0` is the *correct* local derivative, not just NaN-avoidance: every consumer of `S` is locally flat
    in the flow there — production reads `S²` (`d/dt S² = 2S·dS → 0`) and the eddy-viscosity limiter's
    `max(a1 ω, S F2)` picks the strain-independent `a1 ω` branch — and `S = 0` never occurs at a
    converged (sheared) field, so the exact adjoint at the fixed point is untouched. This is what lets
    the coupled solve self-start from the symmetric plug with **no symmetry-breaking perturbation** (see
    the `initialization.py` note). Pinned by `test_strain.py` (finite/zero Jacobian at `S = 0`; FD-match
    where `S > 0`).
- **`S` is what makes the velocity columns of the coupled Jacobian reach one ring further than
  everything else** (`nu_t` reads `S`, built from `∇u`, and multiplies a gradient-based flux; `k` and `ω`
  enter `nu_t` pointwise and `p` not at all) — a real cost of differentiating the closure exactly, which
  a lagged `nu_t` would remove only as a quasi-Newton linearization. Measurement (`bfs3d` colour counts,
  harness `validation/bfs3d_openfoam/column_reach_probe.py`): `.claude/notes/turbulence-log.md`, "`S` makes the velocity columns".
- **`sources.py`** — the k and ω production / destruction / cross-diffusion terms as
  `VolumeSourceFn` volume-source operators (the transport equations reuse the shared advection
  and diffusion flux operators; only the sources are turbulence-specific).
  - **Both productions are limited at the destruction scale (binding).** `KProduction` caps
    `P_k = min(ν_t S², 10 β* k ω)`; `OmegaProduction` caps the *same way* — `α min(S², 10 β* k ω/ν_t)`,
    i.e. `α/ν_t` times the limited k-production (equivalently OpenFOAM's `(c1/a1)β*ω·max(a1ω, F2 S)`,
    c1=10). It reads the frozen closure (`nu_t`, `k`, `omega`, `strain_rate`), so it has **no derivative
    in the solved ω** (adds no ω-Jacobian diagonal) and differentiates exactly through the *live*
    closure in the coupled residual. A tiny `_EDDY_VISCOSITY_FLOOR` guards the `1/ν_t` at the `k→0`
    edge only (k/ν_t is finite where the cap bites). The unlimited `α S²` over-stiffened the ω equation
    in high-strain / transient regions — one of the robustness gaps behind the near-wall `k` collapse
    (#126). ω needs no such flag (its cap is already field-independent).
  - **The near-wall `k` row's Jacobian diagonal goes negative** (`P_k ∝ k` through `nu_t`), so only the
    pseudo-time shift keeps it positive; at the shipped `turbulence_damping` 3 it is well clear of zero.
    The 2026-08-24 measurement at damping 1 and the frozen-production operator probe that confirmed the
    mechanism: `.claude/notes/turbulence-log.md`, "`sources.py` — the `k` row's Jacobian diagonal is negative".
  - **⚠️ `explicit_production_limiter` now defaults to `False` (the EXACT operator) — measured
    2026-08-15, and the old `True` default was a silent adjoint hazard.** The flag freezes the cap's
    `k` in the **linearization** only (a Patankar / deferred-correction treatment). That is free only
    while the cap is **inactive at the converged root**; where it binds there, the IFT adjoint
    linearizes a residual different from the one solved, so the fields are right and the
    **sensitivity is silently wrong**. `KProduction`'s own docstring had always said the coupled path
    uses the exact operator "so the adjoint stays exact" — `SSTTurbulence` defaulted the opposite way,
    and `k_residual` serves both paths from one construction site, so there was no seam by which the
    coupled residual got the exact operator.
    - Measured inert on the channel and on `bfs3d` (limiter ON/OFF bit-identical), with an unexplained
      transient binding mid-march on `bfs3d`: `.claude/notes/turbulence-log.md`, "`explicit_production_limiter`".
    - **The scale-free criterion, worth carrying:** on the unlimited branch `ν_t = k/ω` the ratio is
      `production/limit = S²/(10 β* ω²)`, so the cap binds at **`S/ω > sqrt(10 β*) = 0.949`**,
      independent of `k`, against an equilibrium boundary-layer value of `sqrt(β*) = 0.3`. Measured
      peaks: channel 0.333, `bfs3d` 0.530 at the root. Separation does push it up, and not nearly far
      enough. (Where Menter's shear limiter is itself active the threshold rises further, to
      `S/ω > 10 β* F2/a1 ≈ 2.9 F2`.)
    - **Opting back in is now GUARDED, not merely documented.** `solve_coupled` refuses to return a
      root reached with the limiter set whose cap is active
      (`_reject_a_root_the_frozen_cap_invalidates`, an `eqx.error_if` so it fires on the traced
      `jax.grad` path too). `turbulence.production_cap_active(coupled, state)` reports which cells
      bind, and `turbulence.production_and_limit` is the ONE definition of the cap's two sides, shared
      by the residual and the guard so they cannot drift. This is the discipline the positivity floors
      are already held to, applied to the other stabilization that alters the linearization.
    - **⚠️ WHO STILL WANTS IT — and this is MEASURED, not inferred. The flip is not "the limiter is
      useless".** The exact derivative of the cap is **indefinite where the cap is active**, and an
      *unpreconditioned* k solve stagnates there. Flipping the default **broke
      `test_sst_transport.py::test_k_equation_solves_to_a_finite_bounded_field`** — a bare
      `RootSolver` (no preconditioner, no globalization) on the k equation, which stopped
      converging inside `max_steps` and raised the convergence guard.
      **Why, exactly:** at that solve's starting field (`k = 0.01`) the cap is active in **24 of 24
      cells — 100 %** — so the exact Jacobian carries the indefinite term *everywhere*; at the
      closure's own `k` it is active in **0**. That is the whole mechanism in two numbers, and it is
      consistent with every other measurement here: the cap binds **far from the solution and not at
      it**. `test_turbulent_channel.py` says the same thing from the other side — the
      convection-diffusion AMG rescues the exact operator, so a *preconditioned* exact-Newton k-solve
      converges quadratically to machine precision.
      **So the opt-in has exactly one home: a bare or weakly preconditioned SEGREGATED SCALAR solve.**
      It takes no gradients, so the validity guard never fires on it. Both tests now opt in explicitly,
      and the transport one **asserts the cap really is active at its starting field**, so the flag
      cannot silently become cargo if that ever stops being true. The coupled path always carries a
      preconditioner *and* takes gradients, which is why the default belongs on the exact operator.
    - Harnesses: `validation/production_cap_activity.py` (channel: activity, forward equivalence, and
      the adjoint against finite differences) and `validation/bfs3d_openfoam/production_cap_activity.py`
      (root activity from a checkpoint). `BFS3D_PRODUCTION_LIMITER=0` selects the exact operator on the
      case, and the banner records which arm ran.
- **`transport.py` — `SSTTurbulence`, `SSTClosureFields`.** Assembles the k and ω scalar
  transport residuals on the flow's Rhie–Chow mass flux, with μ_t a **frozen per-cell field**
  recomputed once per outer sweep.
- **`preconditioner.py`** — the convection-diffusion AMG preconditioner for the stiff k/ω scalar
  Krylov solves at high Reynolds number (the scalar analogue of the velocity-block work). It assembles
  its frozen operator as the shared `aquaflux.precondition.ConvectionDiffusionStencil` (#89) and
  hands the **assembled matrix** to `build_convection_hierarchy` / `build_air_hierarchy` (the coarsening
  library is operator-in, #45); its reaction+boundary diagonal still comes from its own `J·1`
  derivation, which is a genuinely different source, not a copy of the interior stencil. Its interior
  diffusion coupling (`_scalar_stencil`, one stencil feeding both the AMG operator — `assemble()`, with
  the fixed cells `detached` — and the pseudo-time shift — `diagonal_parts()`; it was a six-tuple
  `_scalar_operator_pieces` unpacked at both and the shift re-derived the diagonal with `np.add.at`)
  is `discretization.flux_continuous_conductance(Γ, geometry, face_cells)` — the scalar transport
  operator's own diagonal contribution, harmonic on a graded diffusivity `Γ = ν + σν_t`, the *same*
  conductance the k/ω residual's `DiffusionFlux` carries (binding, #154). It replaced a g-weighted
  **arithmetic** face `Γ` that agreed only for constant `Γ` and over-counted a graded face by
  `(1+r)²/(4r)`; for a pure-diffusion scalar the shift diagonal now equals `diag(jacfwd(residual))`
  exactly (pinned in `test_scalar_transport_preconditioner.py`).
  `scalar_transport_preconditioner` returns a **`ScalarTransportPreconditioner`** strategy
  (`ConvectionAmgPreconditioner` / `AirAmgPreconditioner`) rather than the old opaque `lambda phi: solve`.
  They are **`equinox.Module`s since 2026-10-12 (#386)**, hierarchies on the traced side — reversing
  #105's binding "not a pytree"; see the reversed note in `.claude/rules/solve-globalization.md` for why
  and what it costs. `ScaledScalarPreconditioner(inner, scale)` wraps one with a fixed per-cell output
  factor — the reciprocal chain-rule scaling a log-transformed scalar block needs (above); also a module.
  - **Which continuation a solve runs is ONE injected source, not two parallel branches (binding, #278,
    2026-08-20).** `solve_coupled` needs a continuation twice — the initial build and every refresh —
    and those were written as two independent two-way branches (`refresh.builder` vs the default
    `coupled_continuation`), one at the build and one inside the refresh loop. `solve.ContinuationSource`
    (`build(state)` / `refresh(state, previous)`) makes it one decision, with
    `CallerBuiltSource`, `_SessionContinuation` and `FinishedSource` as its three cases (the first and last
    moved to `solve/driver.py`).
    That is the shape #282 had just been fixed for one level down; here it also carried a live defect.
    - **⚠️ `preconditioner` / `reference_state` / `**strategy_kwargs` are REFUSED where they cannot be
      forwarded, not dropped.** They configure the continuation `solve_coupled` builds. On the two paths
      where it builds none — an explicit `strategy`, or a `RefreshPolicy(builder=...)` — they reached
      nothing at all, with no error and no log line, so a march asked for `inner_steps=3` /
      `positivity_floor=1e-6` (a keyword since deleted) and silently ran the library defaults. `**kwargs` is what made it quiet:
      it accepts every keyword and checks none, and it is the main entry point's door. This had already
      cost a study harness (`lu_vs_hostilu.py`, since deleted, #371, carried a warning comment about a `refresh_preconditioner=`
      swallowed here instead of reaching its `RefreshPolicy`).
    - **There is no `_UNSET` sentinel any more (#394, 2026-09-16).** It existed because the scalar
      multigrid was a string whose `None` meant "no preconditioner", so "not given" needed a third
      value. "No preconditioner" is now the value `UnpreconditionedScalars()`, so `None` means unset
      here as on every other settings field.
  - **✅ `preconditioner_spec.py`, the preconditioner as a value (#371, 2026-09-14) — consumed by every
    session, and readable from a case file (#391).** `BlockDiagonal` | `MaterializedJacobian(inverse=
    FieldSplit(leading, trailing) | BlockInverse, probe=JacobianProbeSpec, build_beta,
    refit_beta_floor)` (a `MonolithicVCycle` inverse existed until PETSc was removed, 2026-10-10, and a
    `CompleteLu` one until 2026-10-11). `BlockDiagonal` and `JacobianProbeSpec` are `SettingsValue`s with `None`-unset fields; **`FieldSplit` and `MaterializedJacobian` are not** — both
    have required fields (`leading`/`trailing`, `inverse`) and `probe` defaults to `JacobianProbeSpec()`.
    `solve_coupled`, `coupled_step`, `open_session` and both Reynolds drivers take them.
    - **Their registry is PUBLIC, `PRECONDITIONER_SPEC_MAPPING` (was the private `_SPEC_MAPPING`, renamed
      2026-09-24)**, because the case file's registry (`case/spec.py`) takes its preconditioner kinds from
      `PRECONDITIONER_SPEC_MAPPING.kinds` rather than listing them a second time.
    - **`preconditioner_spec_from_mapping` / `preconditioner_spec_to_mapping` read and write one as a
      nested plain mapping** (`kind` = class name at every level, a field at its default omitted, a list
      read as a tuple), over the generic `solve.SettingsMapping`. **The kind vocabulary is the class
      names, deliberately** — a second, snake_case naming would be a second spelling of every value to
      keep in step. Unknown kinds and fields raise `ValueError` with the path (`inverse.leading`,
      `probe.column_reach[0]`); a nested value of the wrong kind reaches that value's own constructor
      refusal (`TypeError`); the outermost kind must be one of the two families. The writer checks a value
      is the registered class itself (not merely one of the same name) and refuses anything that is not
      plain data — a numpy scalar, a callable — naming its path, so its output always survives a JSON or
      YAML writer. `test_preconditioner_spec_mapping.py` finds every public `VelocityBlock` /
      `ScalarBlock` / `BlockInverse` subclass from the exports of **every** `aquaflux` subpackage and fails if the mapping
      does not accept it, so a new value class cannot be silently unwritable.
    - **✅ FIELD VALUES are checked per position too (#424, 2026-09-23).** `backend: umfpak`,
      `backend: {kind: CompleteLu}` (both fields since deleted with the LU) and `smoother_sweeps: true` — the three that used to load and fail
      at build, or be ignored — are refused where they appear, with the path and what that field takes.
      The rules are read off the fields' **own annotations**, so there is no second table to drift from
      the dataclass; the four choice sets that lived only in docstrings (`backend`, `schur_scaling`,
      `composition`, `prolongation_smoothing`) are `Literal`s now, pinned equal to their builders' own
      sets by `test_preconditioner_spec_mapping.py`. ⚠️ **A wrong nested kind is now refused by the
      LOADER, not by the value's constructor** — so a refusal message moved from `TypeError` naming the
      family to `ValueError` naming the path. Constructing the same value in Python still meets the
      constructor's refusal, which is deliberately unchanged: #424 is about the file. Cross-field rules
      (a setting that means nothing beside a given kind — `mass_scale` beside `schur_scaling: simple`,
      `strength_threshold` beside a `ConvectionAir` velocity block) are #663's, not yet built: both load
      and are silently ignored.
    - **"Default omitted" is judged by EQUALITY WITH THE FIELD'S DEFAULT, not by `None`.** One field
      breaks the "`None` for unset" wording: `MaterializedJacobian.probe`, where an absent key is the
      default probe and `probe: null` is refused. (`BlockDiagonal.method` was a second, with an `_UNSET`
      default that made an explicit `null` mean "no scalar preconditioner"; #394 replaced it with
      `scalar`, where `null` and an absent key are both unset and no preconditioner is
      `{kind: UnpreconditionedScalars}`.)
    - **The scalar block is a value family beside the velocity block (#394, 2026-09-16).**
      `ScalarBlock` (abstract `SettingsValue`) with `ScalarTwoLevel(cycles)`, `ScalarAir(cycles)` and
      `UnpreconditionedScalars()`, in `turbulence/preconditioner.py` beside
      `scalar_transport_preconditioner`, which takes `scalar=` and **returns `None` for
      `UnpreconditionedScalars` without assembling the operator** (the value's `_build` is handed a
      zero-argument `operator()` closure, so a value that builds nothing never pays the `J·1` residual
      evaluation). That `None` is what `k_shift_policy(preconditioner=)` and
      `_reparametrized_preconditioner` already read as shift-only, so neither the coupled policy nor the
      segregated driver branches on the choice. Defaults are byte-identical: `_DEFAULT_SCALAR_BLOCK =
      ScalarTwoLevel()` is the one home for the builder's, `SSTTurbulence`'s and the spec's unset value,
      and `solve_segregated` still defaults to no preconditioner (`UnpreconditionedScalars()`). Each
      multigrid value's fields are pinned to its preconditioner dataclass's fields minus `hierarchy`
      (`test_scalar_transport_preconditioner.py`), which made the scalar `cycles` reachable from a
      spec for the first time; `BlockDiagonal.cycles` stays the **flow** block's alone.
    - `JacobianProbeSpec.column_reach` is stored as a tuple of `int`s **however it arrives** — a
      tuple of floats from a parser included; it used to convert only non-tuples, and the loader hands
      over tuples.
    - **It yields a spec, never a preconditioner (#374):** the march re-fits from states no file names.
      Session destinations (`observer`, `reports`, `on_build`) are not spec fields, so they cannot be
      written. **No flagship driver reads a case file yet** — both still assemble their spec from
      environment variables in Python.
    Two design facts about the values themselves:
    - **The three materialized inverses are ONE family with a nested choice**, because they share the
      probe, the build shift and the refresh floor and differ only in the inverse. That nesting is also
      what makes defect D1 unrepresentable: a monolithic smoother setting (`smoother_fill_levels`, …)
      has no field beside a `FieldSplit`, where the deleted `coupled_amg_continuation(field_split=True)`
      accepted it and silently ignored it.
    - **Each spec's field set is pinned to the constructor it feeds** (`test_preconditioner_spec.py`):
      `BlockDiagonal` to `BlockPreconditioner.build` minus `reference_state`, `JacobianProbeSpec` to
      `coupled_jacobian_probe`'s free settings (not `active_rows`, which follows from the inverse, nor
      `production_viscosity_frozen`, which follows from the operator). `FieldSplit` requires `precondition.BlockInverse` values, never a factory closure, so a
      build-record sink is attached where the session is opened rather than bound into the inverse.
  - **✅ `open_session` / `PreconditionerSession` / `coupled_step` — the ONE coupled builder, and what
    `solve_coupled`, both Reynolds drivers and both flagship cases run on (#371, 2026-09-14).**
    `_BlockSession` and `_MaterializedSession` are `ContinuationSource` promoted: `build(state,
    **march)`, `refresh(state, previous, **march)`, `refresh_preconditioner`, `rebind`.
    `coupled_step` is the one frozen-step builder and opens a private session. The new path was proven
    **array-identical** to the old builders first (`test_preconditioner_session.py` for block / LU /
    field split, `test_coupled_amg.py` for the V-cycle), then wired:
    - **`solve_coupled(preconditioner=…)` replaced `method=`**, which no longer exists, and
      `_DefaultContinuation` is gone: the default is a `_SessionContinuation` over a `BlockDiagonal()`
      session. Block-preconditioner settings (`velocity=`, `schur_scaling=`, …) are **no longer march
      keywords** — `**strategy_kwargs` binds against `coupled_step`'s signature and an unknown name
      raises naming where it belongs. A materialized spec's session supplies `refresh_preconditioner`, which
      makes the march observed (forward-only); a second hook on `RefreshPolicy` beside it is refused, as
      is `jacobian_production_viscosity` beside a session object.
    - **`solve_reynolds_continuation`** opens **one** session for a `MaterializedJacobian` spec, shares
      it across points and `rebind`s it per point; a `BlockDiagonal` spec is built per point at its own
      viscosity, exactly as before. `_SOLVE_ONLY` excludes `preconditioner`. **`solve_reynolds_ramp`**
      opens a session on the **anchor** for EITHER family (a block spec too, since #386) and hands the
      homotopy its `rebind`. A `point_setup` refresh hook beside a session is refused.
    - **The three builders, both public β-tracking hooks and `_rebinding` are DELETED** (see the rename
      table at the top of this file). Both flagship drivers and every harness open a session or build a
      spec; the drivers' inert monolithic smoother settings under the field split (D1) went with them,
      and `bfs3d` now exposes its preconditioner as `compare.PRECONDITIONER`. The array-identity tests
      that pinned the new path to the old builders were deleted with the builders. `CallerBuiltSource`
      stays: a `RefreshPolicy(builder=...)` is still a supported way to rebuild a caller's own step.
    - **The mass-flow sibling is gone (#541): a held bulk velocity is `coupled_step` / `solve_coupled` on a
      `MassFlow`-driven problem** (see the bulk-velocity entry under "The coupled solve"). A
      `MaterializedJacobian` beside one is refused by `open_session`. `_refuse_unknown_flow_block_options`
      is **deleted**: every flow-block option arrives from a spec pinned to `BlockPreconditioner.build`.
    - Both flagship drivers were dry-run against the pre-migration drivers (2026-09-14) and matched
      step 1 line for line: `.claude/notes/turbulence-log.md`, "`open_session` — dry run".
    Facts to hold while finishing it:
    - **March defaults live once, on `coupled_step`'s signature.** A session binds its `**march` against
      that signature (`_march_keywords`), so an unknown keyword is a `TypeError` and no default is
      restated. `preconditioner` and `jacobian_production_viscosity` are refused there: the session owns
      them (user decision Q1).
    - **A frozen `coupled_step` has no refresh hook; a session has `refresh_preconditioner`**, created
      once per session so every march it serves calls the same object. There is no mid-step hook any more
      (2026-10-10, #290): a loop's `refresh_on_cycles` ends the step and the march calls the session's hook
      with `due=True` before the next one (see `solve-march.md`).
    - **`_BlockSession` follows a re-pointing (2026-10-12, #386; it was a no-op until then).** It is a
      `solve.RebuildingSession`: `rebind` stores the companion, and the march's next call to the session's
      `refresh_preconditioner` derives a step on it at the march's state from the held one (`_repointed`:
      the held policy refreshed with `repointed=True` — k/omega hierarchies on their reused coarsening, the
      flow block's anchor hierarchy applied to the station's momentum by `flow.BlockPreconditioner.on`)
      and grafts its shift policy, adjoint factory and operator stand-in into the held step
      (`ShiftedStep.preconditioned_as`), keeping the step control's shift strength. Unrebound, the hook
      returns `None` (and the default-family test pins that). A segment `refresh` keeps carrying the flow
      block with its build-state assembler, as before. Before this, `solve_reynolds_ramp` left a
      block spec to the solve, which built it on the TARGET, so every station ran the target's
      preconditioner and shift. **Measured on the re20000 law-of-the-wall channel ramp (2026-10-12, table
      in `validation/turbulent_channel/README.md`):** before, the march diverged at station 5 of 16;
      after, stations 0-10 take full steps at 4-17 restart cycles and the march diverges on arriving at
      the target with the shift at its floor (0.02). ⚠️ **That "after" arm was the first version, which
      rebuilt each station from scratch and recompiled the march step at every station** (~40 s against
      ~7 s per step): the rebuilt preconditioners were new static objects. The station step now keeps the
      held step's structure and is a cache hit (`test_a_re_pointed_block_session_step_is_a_compilation_cache_hit`),
      which needed the scalar preconditioners to become data (the reversed #105 note in
      `solve-globalization.md`). ⚠️ **NOT re-measured since**, and the change is numerical too (the flow
      block is no longer re-aggregated per station), so the 4-17 cycles above describe the first version.
    - **#701 — a materialized ramp HOLDS the anchor's shift diagonal for the whole march, deliberately,
      and is judged at the target in the case's own row scales (2026-10-14).** The issue read the held
      diagonal as a stale bug: the march's `beta * d`, the inverse's re-fit at `J(station) + beta * d`
      and the row-scaled measure (`_CoupledMeasures.row_scaled` reads `step.shift_policy`) all used the
      anchor's diagonal at every station and at the target. **Every arm says the diagonal should stay,
      and only the stop should move.** The step control adapts the scalar shift strength on the premise
      that the diagonal it scales is fixed; a diagonal swapped under it changes the damping by a different
      amount on every row (the target's velocity diagonal over the anchor's, measured on pitzDaily at the
      seed state: 0.64 median, 0.44 at the 10th percentile, 0.13 at the 1st, 0.030 at the worst cell --
      ~1.6x less damping in the bulk, up to ~33x in the most viscous cells; k and omega within
      0.95-1.12), which the control cannot see, and every arm that did it lost on bfs3d. The shift
      vanishes at the root, so which fixed diagonal is held changes only the path.
      ⚠️ "The anchor shift is 100x the target's viscous `a_P`" (an earlier draft) is wrong: the
      viscosity ratio is 100, the diagonal is convective plus viscous.
      **Built:** `MaterializedSession.rebind` re-fits only (as before #701), and `solve_reynolds_ramp`
      passes the session's `stopping_step(case)` (`solve.StoppingStepSource`) to the solve, so the stop
      -- and its `rtol` reference -- is read in the case's row scales while every step is main's
      (`solve-march.md`, `stopping_step`; `solve.md`, `materialized_session`). The block-diagonal
      family (#704's `RebuildingSession`) still swaps the station's shift in at every station; whether
      it should hold the anchor's too is #711, unmeasured.
      The per-arm tables (pitzDaily and bfs3d, E1–E8) and the retired swap sweep:
      `.claude/notes/turbulence-log.md`, "#701 — the arms".
    - `tools/sibling_builders.py` reports `coupled_step` (beside `flow_march_step`);
      `test_sibling_builders.py` pins that it still sees it. History: `.claude/notes/turbulence-log.md`, "`tools/sibling_builders.py`".
  - **`solve_coupled(refresh=RefreshPolicy(trigger=…))` segments the march to re-freeze the preconditioner — and a refresh
    must CARRY the shift diagonals, not rebuild them (binding).** With a trigger set, the march runs as a
    sequence of segments of one `aquaflux.solve.newton_march`: each steps until the trigger judges
    the frozen preconditioner stale, the k/ω AMGs are re-derived at the state reached, and the next
    segment continues; the state the last segment reaches is checked for convergence and handed to
    `root_adjoint`. Segments exist because the AMG rebuild is off-jit scipy work that cannot run inside
    a traced loop; that part is not subtle. **The trigger is the drift of `ν_t` since the freeze
    state** — `CoefficientDriftTrigger` reading `StepReport.drift`, which `solve_coupled` fills from
    `eddy_viscosity_drift(coupled, <segment start>)`. `ν_t` is the right coefficient because it is what
    the frozen k/ω transport operators are assembled from, so its movement *is* the staleness. The
    earlier signals are both superseded: `refresh_rtol` (a residual threshold) was replaced first, and
    `CycleGrowthTrigger` (the restart-cycle count) is dominated because cost rises from the SER `β` ramp
    as well as from staleness — on a separating flow, by more — which is why it needs a residual gate
    and `patience` that drift needs neither of. Do not reintroduce either. **Re-base the measure at
    every refresh** (`solve_coupled` builds a fresh one per segment); carrying one across segments
    reports drift the refresh already absorbed and re-fires immediately. Measured evidence for
    preferring drift: on the pitzDaily cold-IC march the per-step cycle count exploded **identically in
    two arms with very different step boldness** (monotone 10→12→21→53→119, relaxed 15→27→40→134), i.e.
    cost tracked flow development, not the stepping — so cost alone cannot separate the two causes. **What is subtle:** a refresh rebuilds the
    AMGs **and the shift's transport time scale**, but carries the shift's **coordinate factor**
    `jacobian_scale` (and the flow block) over from the reused policy. The shift diagonal is
    `d = transport_diagonal(state) × jacobian_scale(field)`, and under `LogScalars`
    `jacobian_scale(ω) = ω`. **Rebuilding the whole product at the developed state freezes the march**:
    both factors grow, `d` blows up, the pseudo-transient shift `β·d` over-damps, and the step collapses
    — the relative residual creeps *upward* ~1e-5/step with the recirculation and `k` static, no error
    and no divergence-guard trip (on pitzDaily the un-fixed FULL refresh raised the convergence guard).
    **This is independent of the SER `β`** — do not attribute it to the `β` reset: a controlled
    discriminator from one post-stage-one state showed rebuilding the *product* + carrying the AMG froze
    the march *byte-identically* to rebuilding both, while carrying the *product* + refreshing the AMG
    descended — so the shift rebuild is the freeze, at whatever `β`. **The cure (issue #156) is to store
    the two factors separately, not to freeze the shift.** `CoupledShiftPolicy` carries
    `k_shift_transport`/`k_jacobian_scale` (likewise ω) rather than the product; a refresh rebuilds the
    transport time scale (physics that should track the flow — measured on a real march that upgrades its
    shift every refresh, it holds a full unclipped step) while carrying the coordinate factor frozen, so
    the temporal ratio it presents is `transport(state)/transport(reference)` in which `ω` cancels (the
    `>2×` over-damped tail — 15 % of ω cells when the product is rebuilt — drops to the 0.0–0.1 % of the
    velocity/`k` blocks). `_coupled_shift_policy(..., reuse=…)` therefore rebuilds `k_shift_transport`/
    `omega_shift_transport` at the new state and takes `k_jacobian_scale`/`omega_jacobian_scale` from
    `reuse`. (The preconditioner's copy of the factor, `k_scale`/`omega_scale`, is *re-derived* at the
    new state instead, because its AMG is refreshed at the new physical operator — the same quantity from
    two states, deliberately.) Carrying the frozen factor is safe because the shift vanishes at the root,
    so a slightly-stale factor changes only the path, never the converged state or its adjoint (the same
    argument that carries the flow block). Rebuilding the transport was measured ~1.87× faster end-to-end
    on pitzDaily than a stale baseline (~925 s vs 1726 s to rel 3e-2, flat ~22 s/step vs 100–300 s/step). Also: `max_steps` applies to **each** segment (so up to
    `(refresh.limit+1)·max_steps` march steps, deliberately not split — either segment may need the full
    allowance, and since 2026-09-15 there is no finishing solve to add a second budget). ⚠️ **The last
    segment is marched WITHOUT the trigger** (it has no refresh left to spend): with no finishing solve
    after the march, a last segment stopped where its trigger fired ends the whole solve short of the root.
    That was the first version of the unified march, and four staged-refresh integration tests caught it
    (residual 4.9e-02 against a 1.37e-11 target); pinned fast by
    `test_the_last_refresh_segment_marches_without_the_trigger`. The stopping
    target `atol + rtol·‖R0‖` is measured once, at the initial state, and held across every segment, so a
    refreshed solve stops exactly where an unrefreshed one does for **any** number of refreshes (a
    target re-measured per segment would compound a silent tightening per refresh — this is what the old
    `rtol/refresh_rtol` compensation approximated, and why the `refresh_rtol <= rtol` constraint existed;
    both are now gone). The constrained path
  - ⚠️ **The constrained (held bulk velocity) adjoint test is decided at β = 0 against the
    block-diagonal preconditioner, its weakest regime, so a mass-flow adjoint result is not a stable
    gate** — the test passes an explicit `relative_residual_gmres(1e-8, restart=120,
    stagnation_iters=200)`. Measurement: `.claude/notes/turbulence-log.md`, "The constrained adjoint test".
    - **The refresh, the per-step re-fit, controls and retries all work under `jax.grad` (2026-09-15,
      phase 2 of unifying the two Newton loops; this reverses a binding "raises under `jax.grad`").**
      `solve_coupled` runs its one march on `stop_array_gradients` copies of the assembler and the initial
      state, so every preconditioner it builds or refreshes sees concrete arrays, and `root_adjoint`
      attaches the adjoint at the root reached. The old refusal existed because the mid-march state was a
      tracer and a refreshed preconditioner would capture it; with stopped copies there is no tracer to
      capture. Pinned by `test_a_refreshed_solve_is_differentiable_and_gives_the_unrefreshed_gradient`
      and, for the materialized re-fit, `test_a_solve_that_re_fits_its_split_every_step_is_differentiable` (on the complete LU until it was deleted, 2026-10-11; ported onto the split in `test_coupled_field_split.py`).
      ⚠️ **Two traps survive.** Anything a hook caches must be built from those stopped copies: an array
      built from un-stopped values does not fail during `jax.grad`, only later, as
      `UnexpectedTracerError`. And the solve **cannot** run inside a traced program -- `jax.jit`,
      `jax.vmap`, or a traced loop such as `lax.scan` -- where a
      stopped copy is still abstract — `refuse_a_transform_the_march_cannot_run_in` refuses that before
      any work (it moved to `solve/march.py` and took a `caller=` argument in phase 3, since
      `RootSolver.solve` now refuses on the same terms). A session the **caller** re-points at a live assembler (`session.rebind(coupled)` inside
      `jax.grad`) still refuses to build, since the session holds that assembler itself.
  - **`on_step` / `on_checkpoint` instrument the march, and work WITHOUT a refresh trigger.** `on_step`
    receives each `StepReport` (step, raw_cycles, ‖R‖, ratio); `on_checkpoint` additionally receives the
    *solved-variable* state (map with `physical_fields`). Both are the seam a solver study logs a long
    march through — needed because a multi-hour coupled march that prints nothing cannot be told from a
    hung one, and this case's documented failure mode is a march that keeps stepping while the residual
    creeps *upward*. Every step calls back, and a callback changes nothing about the march. See the
    `march.py` bullets in `.claude/rules/solve-march.md` for why observation is not gated on the trigger and why
    the state rides a separate seam from the report history.
  - **`solve_coupled(step_control=…)` — the dual-time march DEFAULTS to the `DualTimeControl` Courant
    ramp; other controls are opt-in.** A `StepControl` reshapes the shift strength β each step from the
    previous report; it runs between steps on concrete values and never reaches the gradient.
    - **Default (`dual_time` set):** `solve_coupled` auto-selects `DualTimeControl` (the α-based Courant
      ramp) whenever the march is a `DualTimeStep` and no control was supplied — since 2026-09-15 whether
      or not anything observes the march, which changes the path of dual-time solves that used to run
      traced (the root is unchanged) (`default_dual_time_control` -- **in `solve/step_control.py` since
      2026-08-15**, beside the controls it chooses between; it lived here only because an import cycle
      made it inexpressible in `solve/`. Unit-tested in `test_coupled_rans.py`). It grows
      the pseudo-timestep while the inner loop stays comfortable and **carries β across refreshes**,
      reaching a developed pitzDaily recirculation in materially fewer outer steps than the residual-keyed
      control, and carrying a full cold ramp to the target Reynolds number (step counts measured on pitzDaily,
      configuration not recorded — re-measure before relying on them). See the DualTimeStep bullet
      in `.claude/rules/solve-march.md`.
    - **Opt-in `ResidualRatioDualTimeControl`:** ramps β by the steady-residual ratio; safe when that
      residual is a reliable progress signal, but it pins β on the flat `β×travel` pitzDaily plateau
      (the slower arm), so it is not the default.
    - **There is no `AlphaTargetingControl` — the single-step α-targeter was DELETED (2026-08-14).** It
      reshaped β toward the α=1 boundary that SER misses, but it never converged standalone (stalled rel
      ~0.03), its gains were hand-set placeholders and it had no production caller. The α signal is kept
      where it is measured to work: in the two *dual-time* controls above. Full analysis: the "SER β
      schedule runs backwards" bullet in `.claude/notes/solve-globalization-log.md`.
  - **`reuse=` refreshes a stale k/ω preconditioner without changing the compilation signature.**
    `scalar_transport_preconditioner(..., reuse=old)` (threaded through
    `SSTTurbulence.k_preconditioner` / `omega_preconditioner`) re-derives the *values* at a new state on
    the reused hierarchy's **frozen coarsening**. This is what makes a mid-march refresh affordable, and
    the measured reason to want one: on a separated pitzDaily state, refreshing the **scalar** AMGs cuts
    the outer GMRES cycle count materially while refreshing the *flow* block does not (measured at β=2
    with the production lAIR scalars, but on a separated pitzDaily state that is not replayable and with
    no smoother/aggregation recorded — re-measure before relying on the sizes). It matters **only for
    `ScalarAir()`** — lAIR's C/F split
    reads operator values, so a plain rebuild changes every shape below the first level or two and would
    force a recompile of the solve it accelerates (`reuse` routes to
    `~aquaflux.precondition.refresh_air_hierarchy`). For `ScalarTwoLevel()` the aggregation reads only the
    graph, so a rebuild is already structure-preserving and `reuse` is accepted but changes nothing.
    A `ScaledScalarPreconditioner` wrapper is unwrapped (the log chain-rule scale is re-derived at the
    new state by the caller), and reusing across *different* kinds of scalar block raises. Pinned in
    `tests/unit/test_scalar_transport_preconditioner.py`: the lAIR refresh preserves shapes **where a
    rebuild provably does not**, the twolevel path is structure-preserving either way, and a refreshed
    preconditioner **beats the stale one on the developed operator** (so the reused split is a real
    trade, not a no-op).
- **The scalar policy's two halves have different lifetimes (binding, #105).** `ScalarShiftPolicy` carries
  a **shift diagonal rebuilt every sweep** (so the pseudo-time damping keeps tracking the operator as
  ν_t grows — freezing it would under-damp the march and lean on `DivergenceGuard` escalation) and an
  **AMG preconditioner built once and carried** (it only accelerates the Krylov iteration, and rebuilding
  it per sweep cost ~0.9 s (k) + ~1.0 s (ω) at 4k cells *and* re-compiled the whole solve every sweep).
  `SSTTurbulence` therefore splits `k_preconditioner`/`omega_preconditioner` (frozen, `scalar=`) from
  `k_shift_policy`/`omega_shift_policy` (per sweep, `preconditioner=`), each policy bundling the bare
  diagonal `k_shift_diagonal`/`omega_shift_diagonal` — which the coupled `_coupled_shift_policy` reads
  directly (#158), rather than building a policy to discard all but its `.shift_diagonal`, so one
  definition damps a scalar on both paths; `solve_segregated` builds the
  former on the first sweep and the latter every sweep. Measured: traces per sweep went `[5,5,5,5,5]` →
  `[5,5,0,0,0]` with the converged field bit-identical — **under the traced solve of the day; the counts
  moved when the solve became an eager march (2026-09-15), because a converged sweep now traces nothing;
  since 2026-10-14 the march's reference norm is compiled too, so a sweep that recompiles nothing
  executes the residual not at all (`[n, 0, 0, 0]`; it was one eager evaluation per sweep).** The property is
  unchanged and pinned by
  `test_the_scalar_solve_compiles_once_whether_its_preconditioner_is_carried_or_rebuilt`, whose fixture
  had to start disturbing the state to keep measuring it (see `.claude/notes/solve-log.md`, "Contracts: `implicit.py` — the compile-count test that measured nothing"). ⚠️ **Since the preconditioner
  became data (#386) a REBUILT two-level preconditioner is a cache hit too** (its coarsening reads only the
  graph, so the shapes do not move) — carrying it now saves only the ~1 s build, not a recompile; a
  rebuilt lAIR still misses, by shape.
- **`transport.py`'s `omega_residual` returns a `WallFixedResidual`, not a closure (binding, #105).** It is
  rebuilt every sweep and reaches the compiled march step as an argument, so as a bare closure it landed on
  `filter_jit`'s static side and identity-missed the cache every sweep. As an `equinox.Module` its arrays
  ride on the traced side and only their *values* change. (`k_residual` already returned a bound
  `ResidualAssembler.residual`, which equinox treats as a pytree — that one was always fine.) The
  preconditioner is a pytree too since #386, for the same reason.

## Which patches are walls is stated twice, and reconciled at the coupled build (#355, 2026-09-24)

- **Both cross-block consistency checks are METHODS ON THE CLOSURE, not inline in
  `CoupledRANS.build`** — `refuse_a_fluid_the_flow_disagrees_with` and
  `refuse_a_wall_set_the_flow_disagrees_with`. Each compares a declaration the closure itself owns
  against the flow block's, so the closure is where they belong; the density one was inline until
  #355 and moved with the wall one so the two read as one pattern rather than two shapes.
- **`SSTTurbulence.wall_patches` is kept on the built assembler, and `CoupledRANS.build` checks it
  against the flow closures.** The flow block says "this patch is a wall" by giving it a closure whose
  `shears_flow()` is true (`NoSlipWall`, `MovingWall`); the closure says it by naming the patch in
  `wall_patches`, which drives the wall distance, the wall-adjacent cell set and hence the `omega`
  fixation. `_refuse_wall_patches_the_flow_does_not_agree_are_walls` raises on either difference,
  naming each side. **Both directions are defects and neither surfaces later:** a wall the closure
  does not list gets no wall distance, so `omega` is fixed nowhere near it; a patch listed that passes
  fluid adds a spurious zero-distance surface, pulling the wall distance down across the region
  nearest it.
- ⚠️ **Checked, not derived, and the reason is #364.** An imported mesh drops the patch `type` its
  polyMesh file declares, so there is no third, authoritative source to derive both from. When #364
  lands the set can be derived and the third registration disappears.
- ⚠️ **`wall_patches` is kept ALONGSIDE `wall_faces`, and that is not a second spelling.** The faces
  are what the arrays are built from; the names are the *declaration*, and only a declaration can be
  compared with another block's — indices also do not survive a face renumbering.
- ⚠️ **Do NOT add an `is_wall` predicate: `shears_flow()` already is one** (`flow/boundary.py`). Its
  docstring used to say it was "consumed only by the frozen preconditioner's velocity scale; it never
  enters the residual" — **false**: the momentum assembler selects the wall-function viscosity
  override on exactly these faces, which is solution-affecting. Corrected in the same change.

## Near-wall treatment — `boundary.py` (the four pieces, and the wall BC question)

- **`boundary.py`** — inlet/wall closures for k and ω over the generic scalar boundary machinery.
  - **The wall ω is the adaptive (`y+`-insensitive) blend `omega_wall`, imposed at the wall-adjacent
    cell centroid (binding).** `omega_wall(nu, d, k, model) = [omega_vis^p + omega_log^p]^{1/p}` — a
    **generalized power mean** of exponent `p = SSTModel.wall_omega_exponent` (default `2.0`, the Menter
    (2003) quadrature `sqrt(omega_vis² + omega_log²)`) — with the **viscous branch**
    `omega_vis = C·6ν/(β₁d²)` (`C = SSTModel.wall_omega_viscous_coeff`, default `1.0`; the raw branch is
    the single-homed `omega_wall_value`) and the **log branch** `omega_log = √k/(β*^{1/4}·κ·d)` (equilibrium
    log layer, `κ = SSTModel.kappa = 0.41`). The blend recovers `omega_vis` as `d→0` (it grows `1/d²`,
    the log branch only `1/d`) and `omega_log` once the first cell is out in the log layer, so the same
    wall value is correct across `y+` with **no switch** — on a wall-resolved (`y+~1`) mesh it reduces
    to the old pure-viscous fixation exactly (`k→0` kills the log branch). **Why it replaced the
    pure-viscous fixation:** on the wall-**function** pitzDaily mesh (`y+~30`) the sublayer value was
    ~2× too low (measured OF wall ω ~5715 vs the fixation ~3027), so near-wall `ν_t=k/ω` was ~2× too
    high, over-diffusing the free shear layer and pushing interior `k` below the reference — the
    diagnosed k anti-correlation. **Imposition is still at the cell centroid** (`FixedValueCells` at
    `wall_distance[wall_cells]`), so the viscous branch must stay the `6ν/(β₁d²)` centroid value, **not**
    the `60ν/(β₁dy²)` wall-**face** surrogate (10× the asymptote, standing in for the `y=0` singularity,
    Menter 1994) — imposing `60` at the centroid puts near-wall ω 10× high. Do **not** "restore" the 60
    without also moving the imposition to the wall face. **The blend reads `k` at the wall cells, so the
    fixation is state-dependent:** frozen per sweep in the segregated path (`closure.k`), and a **live
    `dω_wall/dk` coupling in the coupled Jacobian** (AD carries it). It is `d(omega_wall)/dk`-**finite at
    `k=0`** for any `p≥1`: the log branch carries `√k` through a guarded `safe_sqrt` (zero derivative at
    `k=0`) and enters the mean raised to `p`, so its contribution *and* its derivative vanish as `k→0`
    while `omega_vis>0` keeps the mean bounded below — a naive `√k` differentiated at zero would give a
    NaN derivative and poison the wall rows. The power mean is computed max-factored
    (`m·[(omega_vis/m)^p+(omega_log/m)^p]^{1/p}`, `m=max`) so no power overflows for large `p`, and the
    log branch's power is double-`where` guarded (the `safe_sqrt` trick) so `0^p` never differentiates
    through `exp(p·log 0)` — which also keeps `grad` w.r.t. `p`/`C` clean (both are differentiable
    leaves). `k` is clamped `≥0` for the log term (off-solution, inactive at convergence). The unit test pins the **ODE residual** of the
    viscous branch (so the `6`/`60` swap cannot recur silently) plus the blend's `k→0` recovery, log-layer
    limit, and finite `k=0` derivative. Consumed by `omega_residual` (transport.py) for **both** the
    segregated and coupled paths (one change point). `omega_wall_value` is retained as the viscous branch
    and for the IC seed (`initialization.py`, the smooth near-wall ω ramp — an IC device, unchanged).
  - **The fixed cells' ω GRADIENT is imposed analytically too, not reconstructed (binding — bug fix).**
    Those cells carry an *imposed* value rather than a solved balance, so their gradient is a model
    quantity as well; inferring it from neighbours is both inconsistent and badly inaccurate. Measured
    on pitzDaily against the analytical gradient of the field we impose: the reconstruction is
    **0.256× the exact magnitude** (p5 0.205, p95 0.374) in the fixed cells, and the error does not stay
    local — the **first interior ring reconstructs 2.24× too large** (p5 0.845, p95 5.04). Two causes,
    both structural: `ω_wall ∝ 1/d²` is strongly convex while Green–Gauss is a *linear* fit over cells
    whose `d` spans 8.5e-5→5.6e-4 here; and the stencil folds in the **wall face**, whose ω a
    zero-gradient closure sets to the cell value although the true profile diverges there.
    - **A gradient-scheme A/B CANNOT detect this** — every scheme treats these as ordinary cells, so all
      are wrong identically and the difference cancels. Measured: `CorrectedGreenGauss` vs
      `CompactGreenGauss` give **bit-identical** ratios (and differ by 0.03 % in ‖R_ω‖). Compare against
      the *analytical* gradient, never against another scheme.
    - **What it corrupts:** `∇k·∇ω` in `OmegaCrossDiffusion` and the `F1` blend, which set the blended
      constants for **both** scalar equations — and the k rows at these cells are **not** fixed, so a
      genuinely solved equation was reading a 4×-wrong gradient (measured `CD_kω` there was **3.6×** too
      small). Also the diffusion's non-orthogonal `corr` on faces to interior neighbours (measured
      negligible on this mesh, 0.03 %).
    - **The fix:** `omega_wall_gradient` = `dω_wall/dd·∇d + dω_wall/dk·∇k`, with both partials taken by
      **automatic differentiation of `omega_wall` itself**, so it cannot drift from whatever blend that
      function implements. `∇d` is reconstructed **once at build** (`wall_distance_gradient`, pure
      geometry) with the exact boundary closure `d = 0` on the wall patches; the distance field is
      smooth and O(geometry), so unlike ω it reconstructs well. Only the wall-adjacent rows are
      replaced. Safe because ω needs **no wall-normal flux** there (the row is a value fixation, the
      wall closure is zero-gradient), so the only consumers are inward.
    - **⚠️ IT IS PASSED *INTO* THE RECONSTRUCTION, NOT APPLIED TO WHAT IT RETURNS (changed
      2026-08-24).** `closure_fields` used to overwrite the array `gradients()` handed back
      (`_imposed_wall_omega_gradient`, **gone**); `_wall_omega_gradient` now returns a
      `schemes.ImposedGradient` that rides through `_field_gradient` →
      `ResidualAssembler.gradient(imposed=…)` → the scheme. The difference only shows for a scheme
      that **consumes its own reconstructed gradient** — `MultipleCorrectionGradient` differentiates
      its first estimate to build the Hessian that corrects what it returns, so a patch applied
      afterwards left that Hessian built on the 0.256× estimate. It also settles the **wall faces'**
      gradient, which a boundary closure would otherwise invent from an ω boundary value that is not
      data. The shipped `CorrectedGreenGauss` reconstructs nothing from its own output, so its
      result is unchanged — this is the same latent-trap removal as the entry above, one layer down.
      See `.claude/rules/schemes.md`'s multiple-correction entry for the seam.
    - Measured inert on pitzDaily (`F1` saturates at the fixed cells); do not cite the fix as a
      convergence or accuracy improvement there: `.claude/notes/turbulence-log.md`, "The imposed wall ω gradient".
    - **Distinction to keep in mind if this is ever extended:** the *point* gradient (right for the
      cell-centred sources) is **not** the best *linear reconstruction slope* over a finite cell for a
      convex profile. For ω the latter barely arises — advection is first-order upwind and the
      flux-continuous diffusion eliminates the face value — but it would matter for a gradient-using
      advection scheme.
  - **The near-wall ω fixation row is written in the SOLVED variable, not in physical ω (binding —
    this was the single biggest defect in the coupled march, fixed 2026-07-25).** `FixedValueCells`
    now carries an injected `FixationRow` (`discretization/fixed_value.py`), and
    `ScalarVariableTransform.fixation_row()` picks it: `DirectScalars` → `DifferenceRow`
    (`ω − ω_wall`, **bit-identical** to the old behaviour), `LogScalars` → `LogRatioRow`
    (`log(ω/ω_wall) = w − log ω_wall`). **Why it matters, two ways:**
    - *Newton.* Under log-ω the old physical row `e^w − ω_wall` gives a correction `δw = r − 1` with
      `r = ω_wall/ω` — the **linearization of an exponential**, landing at `ω·e^(r−1)` instead of the
      target `ω·r`. The log-ratio row is **exactly linear in `w` (derivative 1 at any ratio)**, so a
      full step satisfies the constraint in one iteration. This is what makes the *zeroed shift* on
      those cells correct: the "an exact fixation converges in one Newton step, so it needs no
      pseudo-time damping" justification is true under `DirectScalars` and was **false under
      `LogScalars`** from the day the transform landed until this fix.
    - *Measurement (the bigger effect).* The physical row is scaled by ω, which spans 160→1.1e5 near a
      wall, so **472 of 12 225 cells dominated the residual norm** — the metric that drives the line
      search, the SER β ramp, the divergence guard and the stopping test. Measured on the clean
      pimpleFoam field, ‖R‖ fell **1.533e5 → 20.7 (7 400×)** with the log-ratio row; what remains is
      the genuine wall-blend model difference, not scaling. And the metric now **orders states the way
      the physics does**: before the fix raw ‖R‖ ranked the const-β state at "rel 0.032" *better* than
      the SER state at "rel 0.052", though the former's bubble is 4× worse (`x_r/h` 0.29 vs 1.16,
      against OF's 7.74); after the fix the ranking matches `x_r/h`, `k_peak` and `ν_t`. **Consequence
      to internalize: every conclusion drawn from comparing raw ‖R‖ across march states before this
      date is suspect** — including the choice to prefer the const-β march and the α-controller
      because they reached a "deeper" residual.
    - **The fixation row's derivative must also reach the PRECONDITIONER — this was a real regression
      the row change introduced, caught 2026-07-25 (binding).** `_reparametrized_preconditioner`
      rescales the frozen physical-operator scalar AMG by `1/(dφ/dw)`, because a reparametrized block's
      Jacobian is `J_φ·diag(dφ/dw)`. **That identity holds only for rows assembled in physical φ.** The
      472 wall-fixation rows are not: `LogRatioRow` writes them directly in `w`, so their true
      derivative is **1**, while the frozen operator carries a unit identity row there
      (`boundary_diagonal[fixed] = 1.0`). Scaling them by `1/ω` anyway left the preconditioned operator
      with a `1/ω ≈ 1e-5` eigenvalue cluster on those rows. Measured at the cold IC, capping GMRES at 5
      restart cycles: linear residual **1.03e-3 → 3.87e-5 (27×)** once the fixation rows are exempted.
      Under the *old* `DifferenceRow` the row was `e^w − target`, so `J_ii = ω` and `(1/ω)·ω = 1`
      matched exactly — i.e. **fixing the residual metric silently broke the preconditioner on the same
      rows**, and the two changes must always be made together.
      - **The fix is on the row, not on ω.** `FixationRow.jacobian_scale(phi, chain)` gives each row its
        own derivative (`DifferenceRow` → `chain`; `LogRatioRow` → `chain/phi`, hence exactly 1 under
        `LogScalars`), and `coupled._row_jacobian_scale` assembles the per-row array the preconditioner
        is rescaled by. Correct for either row under either transform, so a new transform/row pair
        cannot silently reintroduce this. The directly-solved path stays all-ones, i.e. bit-identical.
        Pinned by tests that check `jacobian_scale` against **AD of the row itself**, so it cannot drift.
      - **Generalize the lesson:** anything that rescales a scalar block *per row* — a diagonal
        preconditioner rescale, a row-equilibrated norm — must ask each row for its derivative rather
        than assume every row is a transport balance. The shift diagonal escapes this only because it is
        **zeroed** on fixed cells, so mis-scaling zero is still zero.
  - ⚠️ **Taking `abs` of the k shift's net reaction diagonal (#312, PR #317) HALVES the shift and was
    reverted; adding the magnitude (`|r| + |f|`) is inert at the shipped damping 3 and was deleted.** No
    test tier measures step productivity, so a shift-diagonal change must be measured with
    `validation/pitzdaily_gradient_ab/closure_stall_probe.py`'s `analyze` mode before it merges. Full
    record: `.claude/notes/turbulence-log.md`, "#312"; ledger: `.claude/notes/solve-refuted-directions.md`.
  - **The ω shift diagonal inherits ω's dynamic range** under `LogScalars` (`d_ω = transport × ω`):
    rebuilding the product at a developed state over-damps a tail of ω cells and destroys the step, and
    dropping the `× ω` factor (a log-space shift) wins single steps but loses the march, because it
    inverts the near-wall weighting. The built cure is the factor split (#156) in the refresh bullet
    above. Investigation: `.claude/notes/turbulence-log.md`, "The ω shift diagonal inherits ω's dynamic range".
  - The OF-vs-aquaflux ω residual at the clean pimpleFoam field is a uniform ~13 % relative imbalance,
    not a near-wall defect: `.claude/notes/turbulence-log.md`, "The first ring is not special".
  - **The near-wall blend is a power mean** (`SSTModel.wall_omega_exponent` p, `wall_omega_viscous_coeff`
    C; default p = 2, C = 1, Menter's quadrature). OpenFOAM's `omegaWallFunction` is NOT a power mean but
    a hard switch at `yPlusLam` (a `max` is a good proxy except between `y*` ≈ 9.84 and 11.53), and
    Fluent's `correlation` default is p = 1.3, C = 1/3. Which default to ship is an open model decision;
    the mechanism to select any of them is shipped. Measurements: `.claude/notes/turbulence-log.md`, "The blend shape".
  - ⚠️ **Do not compare against `validation/pitzdaily_openfoam/runs/kwsst/`** — that steady OpenFOAM
    reference is corrupt (ω checkerboards in the inlet channel); the clean reference is the
    time-accurate pimpleFoam field. The 2026-07-24 measurements (the `max` blend accepts that field as
    an on-root IC): `.claude/notes/turbulence-log.md`, "pitzDaily reference and the max blend".
  - **The momentum companion is the adaptive wall-face eddy viscosity `nut_wall` (binding).** The
    `y+`-insensitive treatment also needs the momentum wall shear to follow the law of the wall on a
    non-sublayer mesh, not the molecular gradient. `nut_wall(nu, d, k, model) = nu·max(0, y*·κ/ln(E·y*) − 1)`
    with the **k-based** wall coordinate `y* = β*^{1/4}√k·d/ν` is the `nutkWallFunction`: **velocity-independent**,
    so it has **no reattachment singularity** (a velocity-based law blows up where the near-wall velocity
    vanishes) and the wall shear `(μ+ρ·nut_wall)|U|/d` passes through zero there on its own — the correct
    behaviour on a reattaching flow like pitzDaily. Below the laminar/log crossover `y*_lam`
    (`SSTModel.wall_y_star_lam`, the fixed point of `y=ln(E y)/κ`, ~11) it is **zero** — a resolved wall,
    reducing to the plain no-slip molecular shear — so it is a no-op on a wall-resolved mesh and can be
    **always-on** (verified: the wall-resolved `test_channel_law_of_the_wall` is unaffected). The `ln`
    argument is floored at its crossover value on the discarded sublayer branch, so the switch is finite
    and differentiable (no `ln` singularity), and `k` is clamped `≥0`. `E = SSTModel.e_wall = 9.8` (the
    log-law constant). `SSTTurbulence.wall_face_eddy_viscosity(k)` scatters it onto the stored `wall_faces`
    (zero elsewhere) and hands it to `MomentumContinuity.with_eddy_viscosity(nu_t, wall_nu_t)`; the momentum
    block applies it **only at the shearing-wall boundary faces** via the diffusion `boundary_coefficient`
    (the interior closure stays `ν_t=k/ω`). Applied in **both** forward paths (coupled residual live, in
    the Jacobian; segregated driver per sweep) so they solve the identical model. Operator-tested in
    `test_turbulence_boundary.py` (sublayer→0, log-law value, velocity-independence/finiteness, `y*_lam`,
    and the **derivative in `k` against a central difference**); the flow-side seam is in
    `.claude/rules/flow.md` / `.claude/rules/discretization.md`.
    ⚠️ The `k`-derivative test compares against a finite difference (a finiteness check could not fail);
    the log branch's VALUE is still integration-unverified, since the coupled fixtures are wall-resolved:
    `.claude/notes/turbulence-log.md`, "`nut_wall`'s derivative test".
  - **The near-wall `k` budget is closed by FOUR pieces that only work together (binding — measured on the
    periodic channel; do not remove or reorder one in isolation).** Wiring `omega_wall` + `nut_wall` alone
    left the wall-function channel predicting **−25%** of the wall-resolved `u_τ`. The full set brings the
    same mesh to **−2.0%** with the wall-resolved mesh a no-op (`u_τ` unchanged to all printed digits) and
    the segregated loop *converging* where it previously hit `max_sweeps`. All four cross the sublayer/log
    boundary on the **one** smooth weight `wall_function_weight(nu,d,k,model) = tanh((y*/y*_lam)⁴)`
    (never a `y*` switch — an AD-Newton residual cannot converge through a jump; see the docstring), and
    all live in `boundary.py` as pure functions with a `NearWallKClosure` collaborator in `sources.py`
    holding the per-wall-cell data (`cells`/`distance`/`kinematic_viscosity`/`shear_rate`) that always travels
    together.
    1. **The production carries the WALL-FACE shear, not the cell strain.** `wall_shear_stress =
       (ν+ν_t,wall)·|dU/dn|_wall` with `|dU/dn|_wall = |U_P − U_wall|/d` (`SSTTurbulence.wall_shear_rate`,
       area-averaged over each wall cell's faces, guarded `sqrt` so a quiescent field has no NaN
       derivative), and `k_wall_production = wall_shear_stress · log_layer_shear_rate(d,k)`. The *stress*
       is the discrete wall flux momentum actually applies; the *mean shear* is the analytical
       `u_τ/(κd)`. Substituting `nut_wall` gives `τ_w = u_k·u_log` — the geometric mean of the k-based and
       velocity-based friction velocities — so the balance holds **only** where they agree: a genuine
       equation for `k`, unlike the pure-`k` form `β*^{3/4}k^{1.5}/(κd)`, which cancels the destruction
       identically (still forbidden). The previous version passed the *cell strain-rate magnitude* here;
       measured on a `y+~26` channel that shear is `11.5` where the wall gradient is `17.9` and the true
       log-layer shear `2.8`, and it left production 19% under destruction, `k/k_eq = 0.72`.
    2. **The wall-face `k` diffusivity is faded out** (`wall_k_diffusivity = (1−f)·γ`, applied through
       `DiffusionFlux(boundary_coefficient=…)`). A modelled sublayer carries no turbulent-energy flux to
       the wall (the `kqRWallFunction` zero-gradient condition), and retaining `Dirichlet(0)`'s drain costs
       ~7.5% of the local destruction. **Fade the COEFFICIENT, not the face value:** a `k`-dependent face
       value `f·k_P` has `d(φ_ip)/d(k_P) = f + k_P f′ > 1` near the crossover, which makes the wall face a
       `k`-amplifying source and the solve **does not converge** (this is why the earlier `AdaptiveWallK`
       `BoundaryCondition` failed and was **deleted**). The identical flux, a clean linearization.
    3. **The wall cells' k-destruction reads the LIVE wall `ω`** (`NearWallKClosure.dissipation_rate`
       substitutes `omega_wall(k)` there, no blend — the ω equation fixes exactly that value at every
       `y+`). **Mandatory alongside piece 1, not optional:** out in the log layer the modelled production
       is ≈linear in `k` (`ν_t,wall ∝ √k`, `u_τ ∝ √k`), so against a *frozen* `ω` the destruction `β*kω` is
       linear too — the wall row degenerates into a homogeneous equation whose diagonal flips sign as soon
       as production exceeds destruction, and the k solve **runs away** (measured: piece 1 alone raises the
       `EquinoxRuntimeError`). The live `ω` restores the physical `k^{1.5}` destruction. A no-op on a
       resolved mesh, where the fixed `ω` is the `k`-independent viscous branch.
    4. **The strain rate the CLOSURE sees is blended onto `log_layer_shear_rate` in the wall cells**
       (`SSTTurbulence.strain_rate`, used by both `eddy_viscosity` and `closure_fields`). The sensitive
       consumer is not a production term but the **SST shear limiter** `ν_t = a₁k/max(a₁ω, F₂S)`: in an
       equilibrium log layer `a₁ω` beats `S` by only a few percent (2.52 vs 2.44 `u_k/d`), so the limiter is
       *just* inactive; a wall-function mesh's reconstructed `S` overshoots several-fold and throws it hard
       the other way. Measured with pieces 1–3 in place but not this one: wall-cell `ν_t` ~5× low, `U+`
       jumping **6.6** across the first cell spacing where the log law gives 2.7, and `u_τ` still **−12%**
       despite the wall stress itself being right (first-cell `U+` 13.6 vs log-law 13.9). With it, the
       profile tracks the log law and the gap closes to −2%.

    The near-wall `k` collapse at `bfs3d`'s trihedral corners (2026-08-10): the corner `k` row does
    have a non-negative root, and the multi-wall reduction is OpenFOAM's; investigation and the OpenFOAM
    comparison in `.claude/notes/turbulence-log.md`, "Near-wall k collapse at trihedral wall corners". The trap that transfers:

    **⚠️ BUT THE CAUSE IS THIS BLEND, USED OUTSIDE ITS RANGE, AND THAT IS WORTH KNOWING WHEN CHOOSING A
    CONTINUATION LADDER.** `y* = β*^0.25 √k d/ν` scales as **1/ν**, so at a Re/100 anchor
    `wall_function_weight = tanh((y*/y*_lam)⁴)` is ~1e-7 and the **log-layer production is switched off
    even at fully turbulent `k`**, while `ω_wall = 6ν/(β₁d²)` is simultaneously 100× larger. Destruction
    then dominates a homogeneous linear row and `k` decays; raising Re reverses both (weight median
    5.6e-20 → 1.9e-10 → 1.000 across the three rungs). **Anchoring a ladder below the Reynolds number at
    which the wall function turns itself on manufactures a near-wall `k` trough.** Full account in
    `.claude/notes/solve-amg-multigrid-log.md`, "The traced multigrid — in progress".

    **Known residual — the buffer layer.** A wall-function mesh landing at `y+ ≈ 11–16` (the crossover
    itself) is the worst case for any wall function, and the blend smooths it without making it exact:
    measured `u_τ` error on the channel is **−0.6% at y+≈68, −2.0% at y+≈33, −6.9% at y+≈16, −5.0% at
    y+≈11**, versus 0 on the wall-resolved mesh. Place the first cell either inside the sublayer or out
    past `y+ ~ 30`; the buffer-layer dip is a model limitation, not a bug to chase.

## The segregated driver — `solve_segregated`

- **`driver.py` — `solve_segregated`.** The outer Picard loop: μ_t → flow solve → k solve → ω
  solve, with under-relaxation and positivity floors as the stabilizers, and injected
  `solve_flow` / `solve_scalar` so the driver is pure orchestration. The per-sweep coupling is
  `momentum.with_eddy_viscosity(ν_t)` — the driver hands over the closure's **kinematic** `ν_t` and
  the flow assembler forms `μ_eff = μ + ρν_t` from its own material properties, so the driver never
  restates the closure relation and takes **no `density=` argument** (see `.claude/rules/flow.md`).
  An injected momentum stand-in must therefore provide `with_eddy_viscosity`. **The per-sweep call is now
  `with_eddy_viscosity(ν_t, turbulence.wall_face_eddy_viscosity(k))`** — the driver also applies the adaptive
  wall-function eddy viscosity, so the segregated and coupled paths solve the identical near-wall model — which
  widens the injected contract by two: a **momentum** stand-in's `with_eddy_viscosity` must accept the optional
  second (per-face) argument, and a **turbulence** stand-in must provide `wall_face_eddy_viscosity(k)`. A
  resolved-wall stub returns zeros for it (`tests/unit/test_segregated_convergence.py`). The loop **stops on the coupled
  Picard increment** (`_relative_change` — the largest per-field relative L2 change over a sweep <
  `increment_tol`), with `max_sweeps` only a backstop; the outer under-relaxation is the **SER ramp**
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
    ⚠️ **Both library builders return this shape, typed `flow.drive.FlowSolve` (#158, 2026-10-10) —
    and until then `reused_flow_solve` did NOT.** It returned the state alone, so the tests wrapped it
    (`lambda m, s: (m, solve_flow(m, s))`) and the case runner, which hands it to the loop unwrapped
    (`case/solver.py`, `Segregated` on any non-`MassFlow` drive), died on the first sweep with
    `too many values to unpack`. Every case-runner test replaces `solve_segregated` with a recorder, so
    nothing ran the handover. Pinned now at the builder (`test_flow_solve_seam.py`, solver stubbed,
    mutation-checked three ways) and end to end (`test_case_solve.py`'s segregated RANS channel,
    `slow`).
  - **The sweep body between the injected solves is jitted and assembles the flow fields once
    (binding, #106).** The pre-solve μ_t and the post-solve `(mdot, closure)` run in two module-level
    `eqx.filter_jit` prologues (`_sweep_eddy_viscosity`, `_sweep_closure`) instead of op-by-op eagerly
    (the eager path dispatched `velocity_fields` / `mass_flux` / `closure_fields` one op at a time —
    ~130 ms/sweep of avoidable overhead at 1600 cells). `_sweep_closure` calls
    `momentum.flow_fields(flow)` **once** for both the velocity gradient the closure reads and the
    Rhie–Chow `mdot` the scalars advect on (the pre-solve μ_t uses the lightweight `velocity_fields`,
    which is all it needs before `mdot` exists). `solve_segregated` binds the k/ω boundaries once via
    `turbulence.resolve_boundaries()` before the loop, so those compiled prologues never re-run the
    dynamic-shape patch resolve inside `closure_fields`'s gradient assembler. Bit-identical to the old
    eager path; pinned by `test_segregated_prologues_match_the_eager_assembly`.

## The coupled solve — `CoupledRANS` / `solve_coupled`

- **`coupled.py` — `CoupledRANS`, `solve_coupled` (Option 2, the target engine).** The monolithic
  residual `R(u, p, k, ω)` over the flat `[flow…, k, ω]` state — `coupled_rans_layout(momentum.layout)`,
  a `solve/state.py::FieldLayout` that **nests the momentum block's own layout** as its `flow` block, so
  `unpack` yields the `[u,p]` sub-vector and `MomentumContinuity` runs on it unchanged (#285; there is no
  `CoupledRANSLayout` class and no `layout.dim` — `layout.n_fields`, `layout.field_offset("k")`,
  `layout.slice_of("k")`, and the mesh's own `dim`) —
  with **nothing frozen**: μ_t, the strain `S(u)`, the Rhie–Chow flux, and the closure are live, so
  one Newton solve sees the exact cross-block Jacobian. Globalized by `coupled_continuation`
  (a block `CoupledShiftPolicy` = velocity `a_P` shift ⊕ the k/ω transport-diagonal shifts, and a
  block-diagonal preconditioner gluing `BlockPreconditioner` to the two scalar CD-AMGs; the AMG
  hierarchies + numpy-built scalar shift diagonals **frozen at a reference state** off-jit à la
  `reused_flow_solve`, the velocity `a_P` live). Handed to `RootSolver`, it gives the
  **exact coupled adjoint** (§5) — a single transpose solve on the unfrozen `R_coupled`. The ω wall
  rows are `FixedValueCells`. `CoupledRANS.build` pre-resolves the k/ω boundaries (via
  `turbulence.resolve_boundaries()`, the shared idempotent bind the segregated driver also uses) so the
  per-eval assembler rebuild's `resolve` is an idempotent no-op (else a dynamic-shape `nonzero` on
  traced mesh labels breaks the jit). **`CoupledRANS.build` now checks the two densities agree (issue
  #157's second finding, fixed 2026-08-18).** `SSTTurbulence.density` (a scalar, used to form the k/ω
  volume flux `mdot / density`) and `MomentumContinuity.density` (a per-cell array from the flow's
  `PropertyModel`) are two independent values with nothing forcing them to agree, so nothing previously
  caught a caller passing a different value to each — the flow block never reads
  `SSTTurbulence.density`, so it solves fine either way, and the k/ω equations silently solve the wrong
  Peclet regime (a 998× density error, say, gives a 998× wrong volume flux in **both** scalar residuals,
  **both** frozen AMGs, and **both** pseudo-transient shift diagonals, with no other symptom). `build`
  now raises `ValueError` unless `turbulence.density` is close (`jnp.isclose`) to every cell of
  `momentum.density` — which also catches a non-uniform flow density, since a scalar cannot match a
  non-constant array. This is a build-time-only check (plain Python, not `eqx.error_if`): `build` is
  eager setup code on the same footing as `SSTTurbulence.build`'s own wall-distance reconstruction, never
  called from inside a differentiated residual, so it does not need to survive tracing the way the
  forward-march's mid-solve guards do. Pinned by
  `test_coupled_build_rejects_a_turbulence_density_that_disagrees_with_the_flow_assembler`; every
  existing case and test already passes one `RHO` literal to both builders, so this is silent-until-now,
  not a behaviour change for any of them.
  ⚠️ **The two builders each take a `PropertyModel` now (issue #353) — the check above is UNCHANGED,
  only where `SSTTurbulence.density` comes from moved.** `SSTTurbulence.build(..., density=..., molecular_viscosity=...)` took two independent
  numbers in two unit systems (dynamic `mu` implicit in the flow's `PropertyModel`, kinematic `nu`
  passed directly here); it now takes a `properties: PropertyModel` supplying `"viscosity"` (dynamic)
  and `"density"`, matching `MomentumContinuity.build`'s slot, and derives `density` (asserted uniform
  at build) and `molecular_viscosity = viscosity / density` from it. This still does **not** eliminate
  the mismatch this guard exists for: the two `properties` arguments are still two separate models with
  no shared source unless a caller passes the same object to both (as every case and test now does), so
  the guard above stays in place rather than being deleted — consolidating the two builders behind one
  is follow-on work the issue explicitly deferred.
  **The guard compares the whole fluid, not only its density (#367, 2026-09-28):**
  `SSTTurbulence.refuse_a_fluid_the_flow_disagrees_with` checks the density and then the kinematic
  molecular viscosity, the flow's `viscosity / density` read from its **material** `properties` (so an
  eddy viscosity the flow already carries never enters it) against the closure's `molecular_viscosity`.
  A viscosity mismatch is the same silent defect as a density one: the closure runs at a different
  Reynolds number from the flow it is coupled to while the flow block solves fine. Pinned by
  `test_coupled_build_rejects_a_turbulence_viscosity_that_disagrees_with_the_flow_assembler`, whose
  fixture keeps the densities equal so only the viscosity branch can raise. It runs only in
  `CoupledRANS.build`: the Reynolds ramp's scaled companions are made by
  `CoupledRANS.with_scaled_molecular_viscosity`, which moves both copies together. **`CoupledRANS.residual` assembles the Rhie–Chow flow fields once
  (#106):** it builds the `closure` first and takes `nu_t` from it (rather than a separate
  `eddy_viscosity` recomputing the same strain), then one `momentum.flow_fields(flow)` feeds both
  `residual_from_fields` and the `mdot` the scalars advect on — was 3× `_flow_fields` per eval, ~1.85×
  the trace/compile and AD-tape size. **Per-scalar variable parametrization** (`k_transform` /
  `omega_transform`, both `ScalarVariableTransform`, default `DirectScalars` = identity): the coupled
  residual is always written in the *physical* `k`/`ω` (recovered by `physical_fields`), so a transform
  changes only the Newton iterate space, not the root — the residual at the mapped state equals the
  direct residual at the same physical fields (unit-pinned to 1e-13). `DirectScalars` carries positivity
  by the pseudo-transient shift + divergence guard (no in-residual floor); `LogScalars` (`φ = e^w`) makes
  the field `> 0` **by construction under any Newton step** — the fix for the stiff high-Re case where a
  full step drives `ω` negative and `ν_t = k/ω` flips sign without the residual going non-finite (so the
  guard never trips). **Use `omega_transform=LogScalars()`, `k` direct (binding):** `ω` is the field that
  goes negative and `log(ω)` is well-conditioned (`ω` bounded away from 0, large near walls); `log(k)` is
  **not** — `k → 0` at a no-slip wall (Dirichlet 0) so `log(k) → −∞` there stalls the near-wall cells (the
  full-log form descends then freezes; measured). FD-verified for both forms: coupled ‖R‖→machine-zero,
  agrees with the segregated fixed point, adjoint matches finite differences.
    **`k` direct needs a POSITIVITY-PRESERVING LINE SEARCH, not just the shift and the guard (binding —
    this invariant was stated and was wrong).** Nothing structurally stops a Newton step carrying a
    directly-solved `k` negative, and the closure's `sqrt(k)` then NaNs the whole residual from a single
    cell. The divergence guard cannot catch it: it fires on a non-finite residual, which is already the
    poisoned state. Measured on `bfs3d`: 62 healthy steps, then **two cells of 23040** at
    `k = -3.3e-4`, every field finite, only the derived `nu_t` NaN.
    **Every** coupled builder therefore installs `_closure_step_projection(coupled, omega_step_ratio)`
    as its `step_projection` (2026-10-14): each cell's `k` correction clipped to its own room
    (`positive_k_projection`, unconditional for a directly-solved `k`), then each cell's log-ω change
    bounded (`omega_step_bound`, `omega_step_ratio` default `DEFAULT_OMEGA_STEP_RATIO = e` under
    `LogScalars`; `math.inf` = unbounded; a finite ratio with direct ω raises). There is no whole-step
    `step_limit` to set beside it: that seam was deleted the same day as unused.
    See `.claude/rules/solve-march.md`. **There is no `positive_k_limit`, `_k_positivity_guards`,
    `positivity_floor` or `positivity_projection` any more** — the global fraction-to-the-boundary cap
    and its floor/projection switches were deleted 2026-10-14, because one or two near-zero-k `bfs3d`
    side-wall corner cells set that whole-step cap to 0.001–0.002 and caused every redone attempt; the
    per-cell clip + ω bound e took `bfs3d` from 19 steps / 3 redone / 616 s to 17 / 0 / 558 s at the same
    x_r/h 8.361 (`.claude/notes/solve-globalization-log.md`, "bfs3d's retried steps: the global k cap,
    replaced by a per-cell clip with an omega bound (2026-10-14)"). The trap the floor refusal (#365)
    taught still holds for any future cap: **a projection applied before a limit leaves the limit nothing
    to bind**, so a floor fed only to the limit is inert.
  - **⚠️ THE FORWARD SOLVE'S STOPPING MEASURE IS `_coupled_step`'s, NOT A BUILDER'S (binding, #282,
    2026-08-20) — and the surfaces above `_coupled_step` had drifted TWICE MORE after the tail was
    extracted.** `_coupled_step` builds the default forward solver with `norm=None`, which each step binds
    to the progress measure the march hands it (#370; before, it captured the step's build-time
    `residual_norm`, which a rebuilt measure then left behind) — so the solve is steered by and judged by
    one definition. What is per-family is a
    `solve.LinearSolveRegime` (rtol, restart, cap), and the coupled builders take it as one value,
    `linear_solve=LinearSolveSettings(...)` (#388 — the `forward_*` keywords are gone). **Move it through that value,
    never by passing a whole solver as `linear_solve`, which replaces the measure too.**
    How the old builders stopped before this, and where the `0.3` came from: `.claude/notes/turbulence-log.md`,
    "The forward stopping measure".
    - **The bordered (held bulk velocity) step takes the block regime too since #541** — its Euclidean
      `1e-2` exception (`_CONSTRAINED_LINEAR_SOLVE`, deleted) existed only because there was no bordered
      row-scaled measure; there is now (`flow.BorderedMeasures`).
    **`log(k+1)` does not fix this**: `k = e^w − 1` bounds `k > −1`, not `k > 0`, so the failure above is
    still reachable. It is regular at the wall (unlike `log k`) but that solves the other problem, not
    this one.
  - **The reparametrized block's preconditioner/shift are chain-rule-scaled at the reference (binding).**
    The physics Jacobian w.r.t. `w` picks up `d(φ)/d(w) = jacobian_scale(φ)` (`= φ` for log). `coupled_continuation`
    recovers the physical reference via `physical_fields`, scales each scalar shift diagonal by that factor,
    and wraps its (physical-operator) AMG in `ScaledScalarPreconditioner` by the reciprocal — so the frozen
    preconditioner acts on the reparametrized block without rebuilding the hierarchy. `_reparametrized_preconditioner`
    returns the preconditioner **unchanged** when the factor is one, so the `DirectScalars` path is bit-identical.
  - **Every coupled builder's unset `Globalization.line_search` takes `_COUPLED_LINE_SEARCH`, and the
    block family's linear-solve regime is restart-120 GMRES (`_BLOCK_LINEAR_SOLVE`).** The full coupled
    Newton step from the hybrid start overshoots by orders of magnitude, so β escalation stays the
    fallback for a bad direction, not an overshoot. The measurements behind both, and the stale
    "tight tolerance is load-bearing" claim: `.claude/notes/turbulence-log.md`, "The coupled line search".
  - **The coupled solve's default residual measure is the row-equilibrated `RowScaledNorm`
    (`coupled_scaled_norm`), REBUILT at every outer iteration, NOT the plain Euclidean ‖R‖.** The Euclidean
    coupled residual is dominated by the `ω` block (`ω` O(1e5), `k` O(1e-3)), so it barely moves while the
    flow develops and *mis-ranks* states — a converged field can score worse than a badly wrong one, and a
    step collapsing `k` is accepted (see the mis-ranking warning in `.claude/notes/solve-globalization-log.md`).
    `RowScaledNorm` divides each row by its own diagonal and each block by its field magnitude, reporting a
    fractional change per equation, so steering and the stopping test judge every block comparably.
    **Since #370 (2026-09-16) the measure is the solve's, not the step's:** `solve_coupled(convergence=
    Convergence(measure=…))` names it (`RowScaled()` unset; `BlockScaled()` for the coarser one-scale-per-block
    `BlockScaledNorm`, held at the initial state across refreshes — #156 seam 4; `Euclidean()` for the plain
    norm), the march builds it from `_CoupledMeasures` at each iteration's starting state and hands it to the
    step, and the step's default linear solve stops in it (`relative_residual_gmres(norm=None)`). The builders
    take no measure keyword. **The frozen-at-build-state form, the old default, is deleted** — measured cheap to
    rebuild and needed by nothing; the evidence is in `.claude/notes/solve-log.md`, "Contracts: `convergence.py` — deleting the frozen row-scaled measure".
    (A held bulk velocity's bordered `[flow, k, ω, β]` state takes the same default since #541: the
    constraint row is scaled by the held speed, `flow.BorderedMeasures`.) **This does not *fix* the forward stall** —
    the pitzDaily march is globalization-bound and plateaus under any measure (the row-scaled measure is the
    honest signal of that, where the Euclidean fall was a `β×travel` + `ω`-magnitude artifact). It makes the
    measure honest, and it is REQUIRED for the case to be judged correctly.
  - **`Globalization.beta_floor` is off by default (a measured wash) and reachable from every march; the
    coupled flow block uses the convection-aware velocity AMG (`ConvectionTwoLevel()`), not the
    viscous-smoothed default, frozen once — refreshing it does not pay, refreshing the k/ω scalar AMGs
    does.** Overridable through `BlockDiagonal(velocity=…)`. Measurements:
    `.claude/notes/turbulence-log.md`, "`beta_floor` and the convection-aware velocity AMG".
  - **✅ The convection velocity block is frozen at the POLICY'S OWN reference flow, not the plug (#389,
    2026-10-13).** `_coupled_shift_policy` used to call `BlockPreconditioner.build` without
    `reference_state`, so the block's mass flux came from `_characteristic_reference_state` (a uniform
    flow at the fastest prescribed speed) while its eddy viscosity, and every other frozen operator in the
    policy, came from the reference state. It now passes `flow_ref` whenever that flow carries mass flux,
    and leaves the builder its plug otherwise (a closed domain's hybrid start is at rest; freezing there
    would give a viscous-only block). The flow-only march already froze at its own reference state, so
    the two paths now agree where the reference flows; the at-rest fallback is the coupled policy's only.
    - **Why it is a fix and not a preference: the plug cannot build on a contracting duct.** A uniform
      flow pushed into a narrowing channel is far from divergence-free (worst cell net flux 75 % of the
      largest face flux on a 0.3 H bump, against 14 % for the hybrid start), the frozen upwind operator
      loses positivity on its first coarse level, and the hierarchy refuses with a negative diagonal —
      so `solve_coupled` with its default preconditioner could not take a first step. Bump-height sweep
      (2304 cells, Re_H 1e4): the plug builds at 0.05 and 0.10 H and fails at 0.15, 0.20, 0.25, 0.30 H;
      the reference flow builds at all. Pinned by `tests/unit/test_velocity_block_reference_state.py`
      (432-cell 0.3 H duct; mutation-checked: dropping the argument, dropping the fallback and passing a
      rescaled flow each turn one test red).
    - Where both build, plug and reference are indistinguishable (harness
      `validation/velocity_block_reference.py`): `.claude/notes/turbulence-log.md`, "#389".
    - ⚠️ **NOT fixed: the plug's failure on a CLOSED domain (#706).** The 6 x 6 lid cavity at Re 100
      (`test_coupled_rans._cavity`) is at rest at its hybrid start, so it still takes the plug, and the
      plug's hierarchy has a negative coarse diagonal there too (−8.1e-3): `_coupled_shift_policy` cannot
      build the default block for that cavity from its hybrid start, before or after this change. The
      same holds for `BlockPreconditioner.build(velocity=ConvectionTwoLevel())` on a 6 x 6 lid cavity at
      viscosity 1e-2. A better at-rest default than a uniform plug (one that is divergence-free in a
      closed box) is the open question; it was deliberately not moved into the builder, because the
      flow-only march from rest currently degrades to the viscous block with a warning, which builds.
  - **`_coupled_shift_policy` leaves `schur_scaling` at `BlockPreconditioner`'s own default** (it
    hardcoded `"msimple"` until 2026-08-18); MSIMPLE stays a non-dominated option for a standalone,
    flow-only, convection-dominated solve
    (`tests/integration/test_channel_high_reynolds.py::test_mass_scaled_schur_reaches_beyond_the_a_p_schur`).
    Evidence: `.claude/notes/turbulence-log.md`, "MSIMPLE no longer hardcoded".
  - **The complete-LU (`coupled_lu_continuation`, `lu_beta_tracking_refresh`) and PETSc monolithic-AMG
    (`coupled_amg_continuation`, `amg_beta_tracking_refresh`) coupled builders are deleted**
    (2026-10-10/11; the rename table above, and `.claude/notes/solve-refuted-directions.md`, "PETSc" and
    "Complete LU"). What outlived them, on the materialized session's re-fit hook
    (`precondition.BetaTrackingRefresh`):
    - **The re-fit is cost-triggered** (first call, after `rebind`, and when the march passes `due` after
      a step ended on `refresh_on_cycles`); the scheduled β-mismatch and drift gates were deleted (#371).
      Trap: a badly mismatched preconditioner collapses α, which freezes the state and every drift
      measure on it, so a drift trigger cannot fire on that failure mode.
    - The cost-trigger measurement, the deleted β-diagonal split, the `column_reach` and `probe_gradient_sweeps` knobs as first
      recorded here, and the refresh timing: `.claude/notes/turbulence-log.md`, "Materialized re-fit". The probe knobs' current
      record is `.claude/rules/solve-direct-preconditioners.md`.
      **Trap (from the deleted gates, #371): a refresh condition nested inside a β gate is unreachable
      below a preconditioner-only `refit_beta_floor`**, where the gate's clamped input never moves — 91 % of
      sub-floor steps once refreshed nothing while ν_t drifted ~20 % per step. Full data in
      `.claude/rules/solve-amg-multigrid.md`.
      **The jitted probes take the assembler as an ARGUMENT, not as a closure capture (binding).**
      `_jacobian_matvec` / `_batched_jacobian_matvec` are module-level `eqx.filter_jit` functions taking
      `coupled`; the six call sites bind it in a plain `def`. Written as local `jax.jit` closures over
      `coupled` — as they were — every Reynolds-continuation rung is a **fresh cache entry**, because
      `filter_jit` caches per function object. That is pure waste here: scaling the molecular viscosity
      changes exactly **two leaf values** and leaves the pytree structure identical (pinned by
      `test_scaling_the_viscosity_leaves_the_pytree_structure_identical`), so every rung *could* be a
      cache hit. Same defect and same fix as `eddy_viscosity_drift`. Pinned by
      `test_the_jacobian_probe_is_a_cache_hit_across_reynolds_rungs`.
      ⚠️ **That "two leaf values" holds only if those leaves are ARRAYS, and on this case one of them
      was not** — see the per-rung-recompile entry below, which is where a rung is now made a cache
      hit end to end.
    - **The per-rung recompile (2026-08-12) had three causes, all closed:** a fresh preconditioner object
      per rung (a static field, compared by identity), an unused flow block in the shift policy whose
      value-dependent coarsening moved array shapes, and a Python-`float` viscosity on the static side.
      Full account and the `bfs3d` measurement: `.claude/notes/turbulence-log.md`, "The per-rung recompile". Two method traps
      from it: the process-global compilation cache contaminates a leave-one-out cache-key test (use a
      fresh jitted wrapper per arm); and build a control from your own base, never from another branch's
      archived march log (`validation/bfs3d_openfoam/march_log_compare.py`).
  - **pitzDaily cost history (2026-07)** — the retired "k drift is the remaining limiter" stall, the
    SER β schedule under-damping, preconditioner staleness as the secondary cost, and the slope limiter
    ruled out: `.claude/notes/turbulence-log.md`, "pitzDaily cost history". The drift-keyed refresh trigger they motivated is the
    refresh bullet above.
  - **The per-scalar transform is layout-consistent through the coupled solve (binding).** `solve_coupled`
    maps the physical IC into the solved space with `state_from_physical` (before attaching a held force)
    and returns `physical_fields` — so `LogScalars` is correct through a held bulk velocity too (identity
    for `DirectScalars`, which is all the mass-flow tests exercise). Do not reintroduce a bare
    `pack_state`/`layout.unpack` at a solve boundary: it packs physical values as if they were the solved
    unknown, silently wrong under any non-identity transform.
  - **A held bulk velocity is `solve_coupled` on a `MassFlow`-driven problem (#128; one march since
    #541).** A streamwise-periodic channel is driven to a target bulk velocity by a body force `β` that is
    a **coupled unknown**: appended to the state, the residual bordered with `⟨U_dir⟩ − U_bar`. Until #541
    this was a separate `RootSolver` path (`solve_coupled_mass_flow` over
    `mass_flow_coupled_continuation`, **both deleted**) that reached none of the staged march's step
    control, retry, refresh, homotopy or observers, and a case file could solve it only with `Segregated`.
    Now `solve_coupled` asks `coupled_border(coupled)` (`flow.mass_flow_border` in the coupled layout) and,
    given one, attaches the seed force, borders the residual / measures / drift measure / homotopy and
    wraps the source so sessions build from the fields; `_BlockSession._step` borders its policy (and
    unwraps it at a refresh, so the reuse path is the unbordered one); `_coupled_step` borders a
    Jacobian stand-in. The root's adjoint is `_BorderedCoupledResidual` (theta-read border), so `jax.grad`
    is the sensitivity *at fixed bulk velocity* (FD-verified in `test_coupled_mass_flow.py`). It returns
    the fields; `held_force(coupled, flow, k, omega)` reads the force back as
    `β₀ − a·R(fields; β₀)/a·a`, exact at a root because the force enters linearly. A
    `MaterializedJacobian` beside one is refused (`open_session`). The design pieces are flow's — see
    `.claude/rules/flow.md`, "A held bulk velocity on every march".
    ⚠️ **"The segregated loop does not converge on this body-force channel"**, said here before, is a
    property of `test_coupled_mass_flow.py`'s 48-cell fixture: on the law-of-the-wall case files the
    file's `Segregated` solve converges (re20000: 47 sweeps), and the coupled march reaches the same root.
    The law-of-the-wall measurements (#541: the default row-scaled measure costs steps on a held bulk
    velocity, and re240000 is converged by no coupled arm), which is why `Segregated` stays for a held
    bulk velocity: `.claude/notes/turbulence-log.md`, "Held bulk velocity on the law-of-the-wall files".

## Initialization, diagnostics, Reynolds continuation

- **`initialization.py` — `sst_initial_fields`, registered as `hybrid_initialize(coupled)`'s initializer (cold-start, the reason `solve_coupled` self-starts).** ⚠️ **`hybrid_initialize` is no longer turbulence's (2026-09-19): it is `aquaflux.initialization.hybrid_initialize`, a `functools.singledispatch` on the problem's type, and this module's function is the SST body under the name `sst_initial_fields(momentum, turbulence, *, k_floor, omega_floor, length_scale_factor)`.** `CoupledRANS` is registered in `coupled.py` (settings pass through); `aquaflux.turbulence.hybrid_initialize` no longer exists. Every entry below that says `hybrid_initialize(momentum, turbulence)` describes `sst_initial_fields`; `hybrid_initialize(coupled)` is the call.
  ⚠️ **It takes NO `gradient_scheme` argument (#361, 2026-09-16).** k and omega are smoothed with
  `turbulence.gradient_scheme` and the flow through `potential_flow`, which reads
  `momentum.gradient_scheme` — each field's initial condition reconstructed the way its own residual
  will be. It used to substitute `CompactGreenGauss()`, and every caller omitted the argument, so every
  case started on a discretization it was not solving. Do not add the keyword back; the assemblers'
  own default is `schemes.DEFAULT_GRADIENT_SCHEME` (see `.claude/rules/schemes.md`).
  The monolithic Newton is a *local* method: from a raw cold start (`u=0`, uniform k/ω) it **stalls** —
  the near-wall ω fixation alone injects a `~6ν/(β₁d²)` jump, and a uniform interior is far from a
  consistent field the inner solve can precondition. `hybrid_initialize(momentum, turbulence)` builds a
  cheap physical IC (a few linear Laplace solves): **potential-flow velocity** (`flow/initialization.py`
  `potential_flow`), **Laplace-smoothed k** (harmonic interpolant of its BCs), and **ω** =
  boundary-propagated interior **raised to the SAME near-wall closure the residual imposes — the adaptive
  blend `omega_wall = sqrt(ω_vis²+ω_log²)`, `ω_vis = 6ν/(β₁y²)` (binding, see below) — at
  every cell's own wall distance** (via `jnp.maximum`). A *Laplace*-ω over-diffuses the large wall value
  into the interior; seeding only the wall cells (the earlier form) leaves a **cliff** between the fixed
  wall cell and its neighbour on the flat interpolant, and that neighbour's ω equation then carries
  almost the entire initial ω residual. The profile is the exact solution of the near-wall balance
  `ν d²ω/dy² = β₁ω²`, so every near-wall cell starts on the same decay curve; it falls off as `1/y²`, so
  a few cells out it drops below the interpolant and the `maximum` leaves the core untouched, and at the
  wall cells it equals the fixation value (same distance/expression) so those rows stay consistent.
  Measured: this roughly **halves** the initial ‖R_ω‖ (otherwise ~99% concentrated in the wall-adjacent
  cells — the discrete **diffusion-vs-quadratic-destruction** balance, independent of convection /
  production / cross-diffusion). The profile is also the **smooth ramp the held-in-reserve log-ω form
  wants** (`w=log ω`, below): `w(y)=log(6ν/β₁)−2 log y`, whose largest cross-face `Δw` is set by the
  mesh growth ratio (~2, Reynolds-independent), where the cliff would be a `~log(ω_wall/ω_core)` jump in
  `w` that **grows with Reynolds number** as the wall spacing shrinks (measured max `Δw` 5.4→8.3 from Re
  2.5k→25k, vs ~2.4 for the profile). From this IC the coupled Newton converges from nothing
  (~10–15 steps, FD-verified). `solve_coupled(coupled)` with no initial
  state calls it automatically; the segregated pre-smooth is no longer required to reach the basin (still
  available as a fallback). **An exactly symmetric velocity is fine** — the coupled solve self-starts
  from the exactly-uniform body-force plug (`u_y ≡ 0`) with no perturbation. (Earlier this stalled, and
  was misread as a "measure-zero degeneracy in the inner solve"; it was actually the `sqrt`-at-zero NaN
  in `strain_rate_magnitude` — a uniform plug has `S = 0` in every interior cell — now fixed at the
  source by the guarded `sqrt`, see the `strain.py` note. Do **not** reintroduce an IC perturbation to
  "lift" it: the degeneracy was never in the IC.) The IC is a forward device (the converged-state
  adjoint is IC-independent); when differentiating, pass an explicit state built outside `jax.grad`.
  - **The seed MUST be the closure the residual imposes, not the viscous branch alone (binding — this
    was a real regression).** The whole point of seeding the profile is that the wall-adjacent cells start
    *on their own boundary condition* and therefore do not carry the initial ω residual. When the wall
    treatment gained its log branch (`omega_wall`, the adaptive blend) the seed was left on
    `omega_wall_value` (viscous only), so on a **wall-function** mesh the IC disagreed with the BC by
    ~3× at y⁺=30 and ~10× at y⁺=100 — and `‖R₀‖` on pitzDaily rose ~350× (≈2.2e2 → ≈7.8e4), with the
    march then spending its early steps repairing the IC instead of developing the flow. Wall-resolved is
    unaffected (the blend → `ω_vis` as y⁺→0), **which is exactly why a wall-resolved no-op check could not
    catch it**, and why the unit test now asserts against `omega_wall` on a *coarse, high-Re* mesh where
    the branches genuinely differ (a low-Re or fine mesh makes the assertion vacuous). The seed therefore
    runs **after** `k` is settled — the log branch reads `sqrt(k)`, so seeding it against the bare Laplace
    interpolant (k≈0 in a wall-bounded interior) would evaluate the closure at a `k` the solve never sees.
    - **The same rule applies at every Reynolds-continuation handover:** `wall_consistent_omega` /
      `wall_consistent_state` (default off, through `solve_reynolds_continuation(seed_projection=…)`)
      re-impose the fixation rows the viscosity change invalidates. A correctness fix, measured to buy no
      steps: `.claude/notes/turbulence-log.md`, "The wall-consistent seed".
  - **Body-force-driven domains need equilibrium levels, not interpolants (binding).** A
    streamwise-periodic channel has **no inlet**, so both smoothed fields are degenerate: `k` is the
    harmonic interpolant between all-zero wall Dirichlets (**identically zero**), and `ω` is a
    pure-Neumann solve whose interior carries nothing. Left alone that starts the solve at `k=0` →
    `ν_t=0` — not a poor guess but the **laminar** problem, which for a turbulent case is the wrong
    equations. Both levels therefore come from the **friction velocity the force balance fixes**,
    `u_τ = √(βh/ρ)` (`flow/scales.py::friction_velocity`, `h = V/A_wall`): `k = u_τ²/√β*`
    (`boundary.py::equilibrium_k`) and `ω = inlet_omega(k, 0.09h)`, applied with `jnp.maximum` so it
    only ever raises the fields (the `u_τ>0` branch). **Fix k and ω together or not at
    all** — raising `k` while `ω` sits at its `1e-8` floor gives `ν_t = k/ω ~ 10⁶`, far worse than the
    laminar start. The length scale is the **outer mixing length `0.09h`**, not the `0.07·D_h`
    inlet-specification convention: the latter is for an inlet, and here overshot the developed-channel
    `ν_t` by ~3.5× (measured `ν_t/ν` 373 vs the correct 120 = `0.09u_τh/ν`, which the shipped default
    now hits exactly). Pinned by `test_hybrid_initialize_gives_a_developed_channel_eddy_viscosity`.
  - **Inlet-driven wall-bounded domains collapse k too — floor it at the inlet level (binding).** The
    body-force degeneracy has a subtler inlet-driven twin: even *with* an inlet, the walls carry
    `k=Dirichlet(0)` over the whole domain and **dominate the small inlet patch by area**, so the
    harmonic `k` interpolant decays toward zero a few channel heights downstream (measured median `k`
    `~1e-6` at L/H≈8, collapsing further with length — the **laminar** field again). `friction_velocity`
    is zero here, so the equilibrium branch does not fire; the `else` branch instead floors `k` at
    **`jnp.max(k)`** — the interpolant's peak, which by the maximum principle is the inlet Dirichlet
    value — giving a uniform inlet-level interior. **ω needs no matching floor**: its walls are
    *zero-gradient*, not Dirichlet-0, so its interpolant stays at `~ω_in` (verified: exactly `ω_in` for
    a constant-`ω_in` inlet) and `(k_in, ω_in)` is the consistent inlet eddy viscosity `ν_t=k_in/ω_in`.
    Low interior `k` is a prime suspect for the coupled Newton's large near-wall k-swing on a separating
    high-Re case, so this is a coupled-convergence fix, not only a cosmetic IC one. Pinned by
    `test_hybrid_initialize_floors_inlet_driven_k_at_the_turbulent_level`.

**Issue #69 — CLOSED path (do not re-derive without reading it):** all three planned steps shipped —
scalar continuation (#73), Option 1 hardening (convergence stop + adaptive relaxation), and Option 2
(the monolithic coupled residual + its IFT adjoint, the target engine). The segregated loop is
**retained as a forward pre-smoother / fallback**, not the sensitivity model; for gradients use the
coupled `solve_coupled` (its adjoint is exact) — never differentiate `solve_segregated` (forward-only,
unrolls the Picard sweeps, which §5 forbids). The formerly-held-in-reserve **log-variable form is now
built** (`LogScalars` on `omega_transform`, above), promoted exactly as anticipated: the stiff high-Re
separating pitzDaily case (`validation/pitzdaily_openfoam`) drives the direct `ω` negative, and
`omega_transform=LogScalars()` keeps `ω > 0` so the coupled solve no longer poisons its closure. The
form is validated (channel + tests); efficient convergence on the *full* pitzDaily mesh is the open
tuning follow-up noted above.

- **`measures.py` — the coupled row-scaled measure and its equation names (moved out of `coupled.py`
  2026-10-07, where the names had briefly gone and pushed it past its size budget).**
  `coupled_scaled_norm(coupled, shift_policy, state)` and `coupled_equation_names(dim)` → `(u, v, w,
  p, k, omega)` (`(u, v, p, k, omega)` in 2D) — the flat layout's block order, the **single home** for
  these names, built on `flow.flow_equation_names` (the flow's `u, v[, w], p`) plus the closure's two.
  The measure names its blocks by them, which is what lets the march report the residual per equation
  (`newton_march(on_residuals=)`, see `solve-march.md`). `coupled.py` imports the measure for
  `_CoupledMeasures`; `CoupledRANS`/`CoupledShiftPolicy` appear in `measures.py` only as annotations
  (under `TYPE_CHECKING`), so there is no import cycle.
- **`diagnostics.py` — the march log's per-equation view (BUILT).** Two exports over the names above,
  so the two grids under a step row join instead of drifting:
  - `coupled_fields(coupled)` → named **physical** fields for `field_change_metrics` (each solved
    scalar mapped back through its variable transform, so the log reads the same whether ω is solved
    directly or in log form). Velocity is **split per component** so each lines up with its own
    momentum equation; `p` is gauge-free; `ν_t` rides along (derived, not solved — it is what the
    momentum equations actually see, and it moves when k and ω move in ways their own norms hide).
    **Compiled since 2026-10-14**: the values come from the module-level `eqx.filter_jit`
    `_field_values(coupled, state)` (the assembler an argument, so a read at a new state is a cache hit),
    and are named OUTSIDE it from a tuple -- ⚠️ a dict returned from a compiled program comes back with
    its keys sorted, which would reorder the log's rows. Read once per logged step, it cost 34 s of an
    800 s pitzDaily march op by op (see `.claude/notes/solve-log.md`, "Contracts: `norm.norm_of_residual` — measured (#715)"). Pinned by
    `test_coupled_fields_are_each_field_s_own_value_read_through_one_compiled_program`.
    `coupled_residuals` below reads its per-block terms through `_per_block_residual`, compiled the same
    way (its measure is still built per step, a few ms); bfs3d's `compare.py` logs it every step. Pinned by
    `test_the_per_equation_residuals_are_read_through_one_compiled_program`.
  - `coupled_residuals(coupled, continuation, reference_state=None)` → the **per-equation residual** on
    the march's own measure, `coupled_scaled_norm(...).per_block(coupled.residual(state))`. Reads
    `strategy` **late** for its `shift_policy`, so a refreshed segment's rebuilt diagonals are the
    ones used — the same late-read `norm_builder` does. Under a per-rung setup (`point_setup`) the case
    *and* its continuation are both rebuilt, so the reporter must be rebuilt with them (the bfs3d driver
    keeps the current rung's in a list the logger's callable defers to).
  ⚠️ **Equilibrate at the PREVIOUS state, never the logged one** — this is what makes the per-equation
  rows add up to the steered `R` reported beside them (pinned `rel=1e-12`; once a `stopping_step` march arrives the `R` column is the stopping measure and the rows add up to the `steered R` aside). `newton_march` re-derives the
  measure at the state each outer iteration *starts* from and holds it for the whole iteration, so a
  step's residual is `norm_at_start(R(state_at_end))`; scaling at the end state measures the right
  residual in the wrong scales. Consequences: the reporter is **stateful and order-dependent** (once per
  step, in order — `field_change_metrics`'s contract), and a reporter built partway through a march
  **must be seeded** with that segment's `seed_state` or its first step is scaled at its own end state.
- **`ViscosityRampHomotopy` — the SAME Reynolds span walked inside ONE march (`reynolds.py`, BUILT
  2026-09-09).** The `ResidualHomotopy` alternative to the rung ladder below: `stations` geometric
  viscosity stations from `anchor` down to exactly `1.0`, each held for `steps_per_station` outer steps,
  handed to `solve_coupled(homotopy=…)`. ⚠️ **Since 2026-10-11 the stations come from a
  `solve.RampSchedule` (`ViscosityRampHomotopy(coupled, anchor=, ramp=RampSchedule(stations,
  steps_per_station, end_when=, finish=))`, the same `ramp=` on `solve_reynolds_ramp`)**: there are no
  `stations`/`steps_per_station` keywords and no `ramp_steps` property on the homotopy any more (the
  latter is `RampSchedule.ramp_steps`), `scale(progress)` takes the schedule's progress rather than a
  station index, and steps must be entered in order (`enter(step, previous)`). The schedule may END the
  ramp on the march's settle (`solve-march.md`, `RampSchedule`); a fixed schedule walks exactly the
  stations it always did. What it removes is measured in
  `.claude/notes/solve-march-log.md`, "`ResidualHomotopy` — what it is for, and the four-arm result" — read the numbers there rather than
  restating them: in one sentence, the ladder converges every seed rung to a bar the next rung's
  viscosity jump undoes by three to four orders of magnitude, and restarts the pseudo-timestep ramp at
  every rung.
  - **The target station is the CALLER'S OWN assembler, by identity, not a rescale by one.**
    Numerically the same problem; **not** the same object, and the assembler is the differentiable
    parameter pytree the adjoint is taken with respect to. The homotopy dissolves at the target exactly
    as the ladder does, so the root and its adjoint belong to the case. The `companion` strategy is
    never consulted for the target, so this holds under **either** scaling. Pinned by an `is` assertion
    per strategy.
  - **`rebind` is what keeps the preconditioner honest, and it is per station CHANGE.** The hook is
    `amg_beta_tracking_refresh(...).rebind`, which discards the previous companion's probe and forces a
    full re-materialize; the ramp calls it only when the station index moves, which is the whole reason
    a station spans several steps.
  - **It is a plain mutable object, not an `equinox.Module`** — it caches the current station and
    re-points the refresh as a side effect. Like the refresh hook it drives, it runs only on the eager
    forward-only march and must never be on a differentiated path.
  - **⚠️ `redamping` defaults to `ratio ** 0.6` — the station's OWN size, not a constant (2026-09-10).**
    A station warrants damping in proportion to how far it moves the problem, which a constant cannot
    express: `2.0` is calibrated for a 4-station ramp (`3.162 ** 0.6`) and is a *runaway* on a 24-station
    one. Forced to exactly `1.0` at `steps_per_station == 1`, and an explicit value refused there,
    because every step then enters a station, the control is held on every step and never adapts, and
    β becomes `beta_start * redamping ** n`. The exponent is an empirical fit through one measured
    point, not a law — see `_REDAMPING_EXPONENT`.
  - **Schedule — current status.** Both step cases ship a 12-station, one-step-per-station ramp from
    anchor 100 with `scale: flow` (momentum-only), `turbulence_damping` 3 and a `CflResidualDualTimeControl`
    at `grow` 3 with the release (`case.yaml`); the station count is the shift's descent in disguise,
    not an independent lever. Adaptive `steps_per_station` is not expressible (the `ResidualHomotopy`
    protocol keys on the outer-step index only). The schedule, refresh, seed, scaling and damping
    sweeps on both cases:
    `.claude/notes/turbulence-log.md`, "ViscosityRampHomotopy".
        - ⚠️ **WHICH VISCOSITY A STATION SCALES IS AN INJECTED STRATEGY (`companion=`, BUILT
          2026-09-10), and the alternative exists to make the seed question UNREPRESENTABLE.**
          `scale_both_blocks` (the default) moves the momentum block's `mu` and the closure's `nu`
          together, so each station is a genuine lower-Reynolds problem — the path the rung ladder
          walks. `scale_momentum_only` moves the momentum block alone. It is injected rather than
          branched on a flag because the two make different claims about what an intermediate station
          *is*, and because the choice decides which seed the march opens from —
          `solve_reynolds_ramp` builds its seed from the companion it is given, so the two cannot
          disagree.
          - **The structural point, which outweighs the cycle counts.** `hybrid_initialize` reads the
            **closure's** molecular viscosity to place the near-wall `omega`. Leave the closure alone
            and the anchor's and the target's turbulence blocks are the *same object*, so a hybrid start
            built at either is identical — the seed defect above cannot be expressed at all. Pinned by
            `test_scaling_momentum_only_makes_the_anchor_and_target_hybrid_starts_IDENTICAL`, which also
            asserts the contrast (scaling both blocks *does* move the seed).
          - ⚠️ **AN INTERMEDIATE STATION IS THEN NOT A PHYSICAL REYNOLDS NUMBER.** The momentum equation
            sees a viscosity the closure does not, so the SST blending functions and the wall treatment
            are evaluated against the case's own viscosity throughout and no single `Re` describes the
            station. That is legitimate for a homotopy — the path need only be traversable and end
            exactly at the target, which it does, since the target is the caller's own assembler by
            identity under either strategy (pinned) — but a result quoted from an intermediate station
            is **not** a lower-Reynolds solution, and "Reynolds continuation" is a misnomer for this arm.
  - **`turbulence_damping` (built 2026-09-10/11): the k/ω rows' shift strength is multiplied by a
    damping strategy while the velocity rows keep `beta`.** The facts that stay binding:
              - **⚠️⚠️ IT MULTIPLIES THE STRENGTH, NOT THE BASE DIAGONAL — and the first implementation
                did the latter, which silently moved the march's own stopping bar.** The shift's base
                diagonal is *also* the row scale of `coupled_scaled_norm`, the measure the march is
                steered by, stopped on (`atol = 1e-5`) and compared across arms with. Folding the ratio
                into it divided the k/ω rows of that measure by `gamma`, so **every damped arm ran to a
                different physical bar, the looser bar going to the larger gamma.** Nothing reports
                this: each arm's log looks normal and its `|R|` is self-consistent. The first recorded
                sweep — "27/191, 29/206, 41/305 at gamma 2/5/10, gamma=10 stalling twice and going
                backwards" — was taken under it and is **deleted**; re-measured honestly, `gamma = 10`
                takes **64 steps with zero stalls**, so the confound had been hiding a third of that
                arm's length. Damping now rides on `ShiftTerm.row_shift_scale`, a closure of `beta`
                applied in the step, and `coupled_scaled_norm` reads the undamped diagonal. Pinned both
                ways (`test_damping_leaves_the_BASE_DIAGONAL_alone_...`).
              - **The strategies, all reachable and all measurable.** `ConstantDamping(ratio)`,
                `BetaTaperedDamping(initial, beta_start, beta_min, exponent)` (linear in `log beta`,
                since a Courant control moves `beta` geometrically), and
                `ResidualTaperedDamping(initial, reference, layout, exponent)`. ⚠️ **Ending at `1` is a
                choice about the PATH, not a correctness requirement** — the shift term is
                `beta d (phi - phi_n)`, zero at the root whatever multiplies it, which is what licenses
                a constant ratio at all. A taper ending *above* 1 is equally legitimate and is not
                currently expressible.
              - **⚠️ `ConstantDamping.ratio` IS STORED AS A JAX ARRAY, NOT THE FLOAT IT IS CONSTRUCTED
                FROM.** `equinox.filter_jit` partitions on "is this an array", so a Python float rides
                on the **static** side and is compared by value — swapping one per station would
                recompile the entire coupled solve at every station, the same trap a `float` molecular
                viscosity sprang on the Reynolds ramp. Pinned by
                `test_swapping_the_damping_RATIO_is_a_compilation_cache_hit_not_a_recompile`.
    - ⚠️ **A wrapper that rebuilds a `ShiftTerm` from `.diagonal` drops everything else in silence** (the
      residual a residual-keyed policy reads, the per-row multiplier), and a `DualTimeStep` must hand the
      policy its residual; both were plumbing defects that made a residual-keyed damping inert. Find the
      implementers of a Protocol structurally (AST), not by grepping a signature. Detail:
      `.claude/notes/turbulence-log.md`, "ViscosityRampHomotopy — momentum-only scaling".
              - **`solve_coupled(station_step=...)` / `newton_march(station_step=...)` — the seam a
                station-varying setting needs (BUILT 2026-09-11).** `(step, station, arrived) -> step`,
                called once per outer step when a homotopy is running, letting the caller reshape the
                Newton step for the station it is about to run. It exists because the discriminator a
                per-station setting needs — *which station is this* — is known only to the march, while
                everything the shift policy can read is a progress measure. ⚠️ It must swap **array**
                leaves over a fixed structure (`eqx.tree_at`), exactly as the march's own per-step
                measure swap does, or every station recompiles the solve. **Its measured verdict is
                negative** (the non-composition result above) — it is kept because it is the instrument
                that established that, and because nothing else can express a per-station setting.
  - **The ramp is the default on both step cases** (`BFS3D_RAMP` / `PITZ_RAMP`, default `continuous`);
    `off` returns to `solve_reynolds_continuation` as a comparison arm, not a supported path. The ramp
    arm builds its engine through the ladder arm's own `point_setup`. History: `.claude/notes/turbulence-log.md`, "The ramp became the default".

- **`reynolds.py` — Reynolds-number continuation (BUILT).** `solve_reynolds_continuation(coupled,
  n_points, *, schedule=None, **solve_kwargs)` reaches a high-Re coupled root through a homotopy in
  Reynolds number: `n_points` lower-Re solves from an easy anchor up to the target, each seeded by the
  previous converged solution (the lowest self-starts from `hybrid_initialize`, the rest are warm-started).
  Raising the molecular viscosity weakens the convective nonlinearity — **measured on the 560-cell
  channel a cold coupled solve takes 8 steps at Re=250 vs 12 at Re=2500, both at α=1** — so the anchor is
  easy and each up-step is a small jump from a converged neighbour. The **user surface is one integer**
  (`n_points`); everything else is automatic.
  - **It is an outer wrapper around `solve_coupled`, agnostic to the per-Re globalization.** Every keyword
    in `solve_kwargs` is forwarded to each per-Re solve, so the pseudo-transient march, the dual-time
    march (`inner_steps>1`, whose observed rungs default to the `DualTimeControl` Courant ramp), the
    preconditioner options and the observers all compose unchanged — no coupling to which globalization runs.
  - **⚠️ EXCEPT the continuation keywords, which are split BOTH WAYS between the ramp and the target
    (binding, #278, 2026-08-20).** A pre-built `strategy` / `reference_state` is frozen at the
    *target* viscosity, so it reaches the **final** solve only and each lower-Re point builds its own
    (`ramp_kwargs`, long-standing). The mirror image is that everything which *configures* a build —
    `preconditioner`, and every keyword bound for `coupled_step` — reaches the **ramp** only, because
    the target is not building one when the caller supplied it (`target_kwargs`, keyed on `_SOLVE_ONLY`,
    derived from `solve_coupled`'s own signature so the two cannot drift). **Passing both at once is the
    ordinary case, not a mistake.** Only the ramp half existed, so such a call reached `solve_coupled`
    with a continuation *and* the settings for one — ignored in silence there until #278 made it a
    `TypeError`, at which point the missing half surfaced as a failing adjoint test.
  - **⚠️ A forwarding wrapper hides call sites from an audit scoped to the callee.** An AST scan of
    `solve_coupled(` sites cleared this change and missed this one entirely, because the offending call
    is spelled `solve_reynolds_continuation(...)` and the keywords travel through `**solve_kwargs`.
    When you change a contract on `solve_coupled`, scan its wrappers too — `solve_reynolds_continuation`
    and `solve_reynolds_ramp` are the ones that exist.
  - Rung count on `bfs3d` (two kept; the Re/100 anchor is not needed for reachability, measured
    2026-08-09): `.claude/notes/turbulence-log.md`, "How many rungs".
  - **The continuation DISSOLVES at the target (binding).** The final solve runs at the case's true
    viscosity on the **live** `coupled`, so the root and its exact IFT adjoint are identical to a direct
    `solve_coupled` — the continuation changes only the path. Pinned: `n_points=2` reaches the direct
    solve's fields to `1e-6`, and `n_points=0` is bit-identical to a direct solve
    (`tests/integration/test_reynolds_continuation.py`).
  - **Intermediate points converge LOOSELY (`intermediate=Convergence(rtol=1e-2)` default).** A lower-Re
    point is only an initial guess for the next Reynolds number, so converging it to the target `rtol` is
    wasted work — the `intermediate` test applies to the lower-Re solves only, its unset fields (the
    measure, and here `atol`) taken from the target's `convergence`; the target keeps the caller's own.
    `None` disables the loosening. Measured necessary on pitzDaily: the wall-resolved 12k-cell mesh is
    stiff *independent of Re* (Re continuation removes the convective nonlinearity, not the mesh-induced
    linear stiffness), so an anchor converged to `1e-6` grinds for many iterations for no benefit to the
    seed. Pinned by monkeypatched-`solve_coupled` unit tests that the lower-Re points receive the
    intermediate test, the target its own, and that the intermediate one inherits the target's measure.
  - **`point_setup` — a PER-POINT continuation/precondition seam for a per-companion, per-state
    preconditioner (BUILT).** The single `strategy`/`reference_state` is dropped for the ramp rungs
    (target-specific), so it cannot express a preconditioner that must be rebuilt at *each* rung's own
    viscosity **and** seed state — chiefly the complete-LU β-tracking hook (`coupled_lu_continuation`
    frozen at the point's `(state, β)` + `lu_beta_tracking_refresh` closing over the point's residual),
    which is what lets the aggressive Courant control reach the developed pitzDaily root (the block PC
    stalls/diverges at the overshoot — see the pitzDaily case). `point_setup(companion, seed_state, point) ->
    dict` is called for **every** point (lower-Re and target) with that point's companion, its **packed
    seed coupled state**, and a `ReynoldsPoint` (1-based `index`, `total`, `viscosity_scale`, plus
    `is_target` / `label`) telling it where in the ramp it is — so a per-point builder never counts its
    own invocations to recover the loop's index, and can reach the total and the scaling at all, and its keys are merged over `solve_kwargs` (overriding any `strategy`/
    `reference_state`). To give the built continuation the state the solve begins from, the loop
    **materializes the lowest point's seed** (`hybrid_initialize`) when `point_setup` is set, rather than
    letting `solve_coupled` self-start internally. ⚠️ Under `jax.grad` the **target** point's companion is
    the live assembler, so a `point_setup` that builds a preconditioner from it must build from
    `stop_array_gradients(companion)`, or the preconditioner captures a tracer. **`None` (default) is
    byte-identical** — each point self-starts and builds its own default continuation. Pinned by a
    monkeypatched-`solve_coupled` unit test (called per point, kwargs merged, lowest seed materialized;
    and `None` reproduces the plain ramp). Used by
    `validation/pitzdaily_openfoam/compare_reynolds_continuation.py` (the complete-LU + aggressive-control
    ramp that reaches `x_r/h` ~ 8).
    - **⚠️ "Per-rung" is about FITTING, not about OBJECTS — and conflating the two cost ~10 % of the
      `bfs3d` march.** A rung genuinely needs its V-cycle fitted at its own viscosity and seed state;
      it does **not** need a new preconditioner *object*, and a new object recompiles the entire
      coupled solve (a static field, compared by identity). The `bfs3d` driver therefore builds the
      preconditioner, the probe, the refresh hook and the inner observer **once**, outside
      `point_setup`, and each rung reuses them — `refresh.rebind(companion)` re-fits the shared V-cycle
      at that rung. `point_setup` is still the right seam; what it should vary per rung is the
      *residual assembler and the row scales*, which are ordinary data and cost nothing. See the
      per-rung-recompile entry under `amg_beta_tracking_refresh` for all three causes.
  - **Prefer an absolute intermediate bar, `intermediate=Convergence(atol=…)` with the target at
    `rtol=0`, on a row-scaled measure** — a relative bar re-bases on each rung's own `‖R₀‖`, so a later
    rung stops at a looser absolute residual. And budget a march by how long it takes to reach the
    Newton basin (`α → 1` and the cycle count collapsing together), not by the final residual level.
    Measurements: `.claude/notes/turbulence-log.md`, "An absolute intermediate bar".
  - **Schedule is an injected `ReynoldsSchedule` (a `Protocol`), default `GeometricReynoldsSchedule`
    (one decade per step). ⚠️ IT IS ASKED FOR ONE SCALE AT A TIME, NOT FOR THE WHOLE LADDER — there is
    no `scales(n_points)` method (removed 2026-09-08).** Three pure methods: `anchor(n_points)` (where
    the ramp starts, `ratio ** n_points`), `next_scale(converged, failed)` (the next scale to attempt,
    `1.0` for the target, `None` to give up) and `planned_total(n_points)` (the label's denominator, or
    `None` when the length is not knowable). `converged` is the scales of the rungs solved so far and
    `failed` is the scale that has just failed, or `None` after a success — so one method covers both
    "where next" and "that was too big", as a pure function of the history with no hidden state.
    Geometric because the nonlinearity scales multiplicatively with Re and each up-step is seeded by a
    converged neighbour; `n_points` is the number of decades (anchor at `Re_target/10^N`), and `ratio`
    is an advanced knob. Unit-tested directly, by walking the ladder.
    - **Why the interface changed: the fixed ladder could not express a step-size control, and a
      continuation without one is unusual.** The whole ladder was previously computed before any solve
      ran, so a rung that failed ended the continuation and the caller was told to re-run at a larger
      `n_points` — discarding every rung already converged. That is the standard step-size rejection of
      numerical continuation with the human as the controller and a full restart as the retry.
    - **`AdaptiveReynoldsSchedule` closes that loop (BUILT 2026-09-08).** On a failure it retreats to
      the **geometric mean** of the root in hand and the scale that failed — a bisection in `ln(scale)`,
      the parameterization the ramp is geometric in — keeping the converged rungs and re-seeding from
      the last root. Repeated failures bisect again; it gives up below `min_ratio` (default 1.05), since
      a step that small is evidence the difficulty is not the jump size. After a retreat the step grows
      back by `recovery` (default 1.5) per successful rung, capped at `ratio`: without it one hard rung's
      caution is carried to the target, with it unbounded the next rung jumps straight back to the step
      that just failed. **It reacts only to whether a rung converged** — the one signal available without
      observing the march, and deliberately so, given that three independent attempts to *predict* a bad
      step have all failed here (see the entry above).
    - **The rung and the target take the SAME path through the loop**, differing only in which assembler
      and which keywords they get, which is what lets a failed **target** retreat too — it is the hardest
      point on the ramp and so the one most worth inserting a rung in front of. The target still runs on
      the live `coupled`, so its adjoint is unchanged.
    - **`GeometricReynoldsSchedule` returns `None` from `next_scale` on a failure, so the DEFAULT IS
      EXACTLY AS UNFORGIVING AS BEFORE.** Pinned by a pair of tests that drive the same simulated failure
      through both schedules — one recovers, one raises — so the retreat cannot leak into the default.
    - `ReynoldsPoint.total` is now `int | None`, and `label` renders the index alone when it is `None`.
    - **Ladder studies (2026-09-08)**, in
      `.claude/notes/turbulence-log.md`, "Reynolds-continuation ladder studies": a tangent predictor between rungs makes
      the seed worse at a decade; a finer ladder at the same span loses wall to per-rung shift descents;
      a lower warm `beta_start` wins at 0.1 and is catastrophic at 0.05 (the cliff is per rung); path
      geometry does not predict a dangerous rung (the third refuted predictor: adapt from observed cost
      instead); a local smoother does not remove the ladder; where a 3-rung march's time goes; and a
      literature survey (no order continuation or grid sequencing on separated flows).
  - **Case reconstruction at a scaled ν is `CoupledRANS.with_scaled_molecular_viscosity(factor)` — one
    home for where ν lives.** The molecular viscosity sits in **two** leaves that must move together —
    the momentum block's dynamic `μ` (its `PropertyModel` `"viscosity"`) and the turbulence block's
    kinematic `ν` (`molecular_viscosity`) — and each object scales its own: `MomentumContinuity`/
    `SSTTurbulence.with_scaled_molecular_viscosity`, composed by the coupled method (density untouched;
    `μ = ρν` stays consistent because both scale by the same factor). Under it sits the generic
    `Property.scaled(factor)` / `PropertyModel.with_scaled(name, factor)` primitive (see
    `.claude/rules/properties.md`). The case is never restated — the assembled objects are scaled.
  - **Differentiability.** The lower-Re ramp only makes an initial guess: companions are built from a
    `stop_gradient` copy of `coupled` and each intermediate result is `stop_gradient`-ed before it seeds
    the next, so the ramp never tapes. The final solve runs on the live `coupled` from a stopped seed, so
    `jax.grad` through the wrapper is the target solve's IFT adjoint — **exact and `n_points`-independent**
    (pinned: grad through `n_points=1` equals grad through `n_points=0` and finite differences). Same
    contract as `solve_coupled`: to differentiate, pass a target-viscosity `strategy` built outside
    `jax.grad` (used by the final solve only — `strategy`/`reference_state` are dropped from the ramp
    kwargs, since each lower-Re point builds its own preconditioner at its own viscosity) and no
    forward-only keywords.
  - **Failure handling.** A lower-Re point that fails to converge (`EquinoxRuntimeError` from the
    convergence guard) is re-raised as a `RuntimeError` naming the point and its scale and suggesting a
    larger `n_points`; the march never continues from a non-root.

## Binding decisions

- **Segregated forward, coupled adjoint (design note §5 — binding) — BUILT via `solve_coupled`.**
  Segregation is a **forward-solve strategy only**. For exact sensitivities the adjoint is the
  implicit-function-theorem solve on the full **unfrozen** coupled residual
  `R_coupled(k, ω, U, p; params) = 0` at the converged state — the `solve/` two-level
  implicit-diff machinery — **not** a differentiation of the Picard iteration. This is now realized:
  `coupled.py`'s `CoupledRANS.residual` **is** that unfrozen `R_coupled`, and `solve_coupled` hands it
  to `RootSolver`, whose adjoint is a single transpose solve (FD-verified). At the fixed
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
  Picard increment and stops on it (`increment_tol`), with `max_sweeps` only a backstop and a warning
  when the cap is hit unconverged. Do **not** reintroduce a hard-coded `sweeps` count. The increment measure
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
  wrong — surface it, do not ship it. (Log-variable transport `φ = e^w` — **built** as `LogScalars`, above —
  is the structural fix: it removes the floor entirely for the transformed field, which stays `> 0` by
  construction, so there is no clamped region to pollute the sensitivity. Use it on `ω`, not `k`.)
  - **The ω floor is the k-tied realizability floor `ω ≥ k/(nut_max_coeff·ν)` (default `nut_max_coeff
    = 1e5`), NOT a fixed value (#126).** It caps `ν_t = k/ω` at `nut_max_coeff·ν`; being tied to the
    current `k` it is **inactive at convergence** for a physical field (`ν_t/ν` is O(10²) ≪ 1e5), so it
    honours the precondition above rather than pinning near-wall cells the way the old fixed `1e-8` ω
    floor could. `omega_floor` remains only as a tiny absolute backstop (`max(realizability, ω_floor)`).
    Pinned by the law-of-the-wall test asserting `ω > k/(1e5 ν)` everywhere at the converged state.

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
  actually reached `increment_tol` (the driver only `warnings.warn`s otherwise, and returns the
  under-converged fields), or that the result matches an independently converged reference. The
  model is `test_coupled_rans.py`, which drives the loop to `increment_tol=1e-9` and asserts it reaches the
  coupled solve's fixed point to 1e-4.

## Post-change
Keep this file's Status and Binding decisions true as the coupling globalization (issue #69) lands —
per the root `CLAUDE.md` Post-Change Checklist's Documentation-sync item.
