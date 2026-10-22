---
paths:
  - "aquaflux/turbulence/**"
---

# Rules — `aquaflux/turbulence/` (RANS closure: k–ω SST + the segregated coupling)

The k–ω shear-stress-transport (SST) closure and its coupling to the p–U flow block. The forward
coupling may be **segregated** (an outer Picard loop, decided with the author) but segregation is a
forward-solve strategy only: sensitivities come from the adjoint of the **unfrozen coupled residual**.
Flow block: `flow.md`; the generic march, measures and adjoint machinery: `solve.md` and its siblings.
Provenance (precursor codes, design notes) informs this file only — none of it may reach `.py` or
`docs/` (root `CLAUDE.md`, Comment Convention).

Every entry sits under a `##` section; check any symbol or default against the source before quoting
it. Dated measurements and the full pre-2026-10-17 entries are in `.claude/notes/turbulence-log.md`
(the last section, "Pre-condensing text of `turbulence.md`", holds them verbatim by topic).

## Generic machinery does NOT live here — read before adding to `coupled.py`

The residual-agnostic march (staged driver, continuation sources, shifted-step tail, linear-solve
regimes, measures' generic parts) lives in `solve/`; the materialized-Jacobian session, probe,
monolithic shift policy and the `MaterializedJacobian`/`FieldSplit`/`JacobianProbeSpec` specs live in
`precondition/` (#448/#277/#284). `flow.solve_flow_march` runs on the same pieces. What stays
here is physics or this family's own: `CoupledRANS`, `_CoupledMeasures`, `eddy_viscosity_drift`,
`_CoupledProblem` (the `MaterializedProblem` coupled RANS supplies), `coupled_jacobian_probe` /
`_CoupledNarrowing`, `BlockDiagonal` with `open_session`/`_BlockSession`, and the per-family regime
constants (`_BLOCK_LINEAR_SOLVE`, …) because they calibrate *these* preconditioners. The mass-flow
border is in `flow/bordered.py`; `coupled_border` / `_BorderedCoupledResidual` are its adapters.
⚠️ **Do not add a march capability here without checking `solve_flow_march` can reach it** (Principle
3.6). The size ratchet in `tests/unit/test_layering.py` is what to lower as code leaves. The full
move table: log, "Where the generic march machinery moved".

## The rename table — names that no longer exist

Other files point here. If you find one of these names, read it through this table.

| was | is |
|---|---|
| `coupled_continuation(…, method=M, **flow_opts)` | `coupled_step(coupled, state, preconditioner=BlockDiagonal(scalar=S, **flow_opts), **march)` |
| `coupled_lu_continuation`, `lu_beta_tracking_refresh`, `CompleteLu` | deleted (2026-10-11); use `MaterializedJacobian(FieldSplit(L, T), build_beta=b, probe=JacobianProbeSpec(stencil_reach=r))` |
| `coupled_amg_continuation`, `amg_beta_tracking_refresh`, `MonolithicVCycle` | deleted with PETSc (2026-10-10); field split as above. A shared hook → one session: `open_session(MaterializedJacobian(..., refit_beta_floor=f), coupled, observer=o)` passed as `solve_coupled(preconditioner=session)`; its `refresh_preconditioner` / `rebind` replace the hooks' |
| `_SessionContinuation`, `_DefaultContinuation`, `_rebinding` | `solve.SessionSource`; default is a session over `BlockDiagonal()` |
| `solve_coupled(method=M, velocity=…)` | `solve_coupled(preconditioner=BlockDiagonal(scalar=S, velocity=…))` |
| scalar method strings `"twolevel"`/`"air"`/`None`, `_UNSET`, `resolved_method` | `ScalarTwoLevel()` / `ScalarAir()` / `UnpreconditionedScalars()` on `scalar=` (#394); `resolved_scalar`; a string is refused |
| `velocity="convection"`/`"smoothed"`/`"convection-air"` | `flow.ConvectionTwoLevel()` / `ViscousMultilevel()` / `ConvectionAir()` (#390) |
| `solve_coupled(rtol=, atol=, scaled_norm=)`, builder `block_scaled_norm=`/`residual_norm=` | `convergence=Convergence(measure=…, rtol=…, atol=…)` on the solve (#370) |
| `reuse=previous.shift_policy, residual_norm=m` | `session.refresh(state, previous, **march)` |
| `solve_reynolds_continuation(intermediate_rtol=, intermediate_atol=)` | `intermediate=Convergence(...)` |
| `mass_flow_coupled_continuation`, `solve_coupled_mass_flow` | deleted (#541): `coupled_step` / `solve_coupled` border themselves on a `MassFlow` drive; force via `held_force(coupled, flow, k, omega)` |
| `shift_basis=`, `velocity_shift_parts=`, `turbulence_damping=` | `shift=CoupledShiftSettings(basis=, velocity_parts=, turbulence_damping=)` (#387; subclasses `solve.ShiftSettings`) |
| `inner_steps=N`, `inner_tol=`, `cycle_budget=`, `refresh_on_cycles=`, `inner_refresh` | `dual_time=DualTimeLoop(...)` (#388); `inner_refresh` removed (#290) |
| `krylov_solver=`, `forward_rtol/_restart/_max_restarts=` | `linear_solve=S` or `LinearSolveSettings(rtol=, restart=, max_restarts=)` |
| `aquaflux.turbulence.LinearSolveSettings` / `ShiftSettings` / `hybrid_initialize` | `solve.LinearSolveSettings` / `solve.ShiftSettings` / `aquaflux.initialization.hybrid_initialize(coupled)` |
| `point_setup` returning `strategy` + a refresh hook | `preconditioner=` on the Reynolds drivers; `point_setup` keeps per-point march settings only |
| `solve_segregated(rtol=)` | `increment_tol=` (#373) |
| `scalar_pseudo_transient_solve(max_steps=, rtol=, atol=, solver=)` | `root_solve=RootSolveSettings(...)` (#428) |
| `MaterializedJacobian(beta_floor=)` | `refit_beta_floor=` (#373) |
| `positive_k_limit`, `_k_positivity_guards`, `positivity_floor`, `positivity_projection`, `step_limit` | deleted 2026-10-14; `_closure_step_projection` (below) |
| `_safe_sqrt`, `_jacobian_matvec`, `_imposed_wall_omega_gradient`, `AlphaTargetingControl`, `AdaptiveWallK` | `strain.safe_sqrt`, `jacobian_matvec` (`precondition/materialized_session.py`), `_wall_omega_gradient`, deleted, deleted |
| `SSTTurbulence.advection_scheme` / `gradient_scheme` | per-field `k_/omega_advection_scheme`, `k_/omega_gradient_scheme` |

## Schemes are bound per field, at build

- **Gradient (2026-09-22).** `SSTTurbulence.build` binds the injected scheme twice,
  `k_gradient_scheme` / `omega_gradient_scheme`, against each field's own conditions (a wall fixes `k`,
  lets `omega` extrapolate), and binds at build because `_assembler` runs inside the residual on a
  traced mesh, where a connectivity-reading scheme (`ProjectedStencilGradient`) cannot prepare;
  `ResidualAssembler.build(..., bind_gradient_scheme=False)` passes the bound scheme through.
  ⚠️ Each equation must *read* its own binding; `test_sst_transport.py` pins it by substitution, which
  needs a frozen closure (F₁ reads both gradients) and a perturbed mesh (an orthogonal one never reads
  `k`'s gradient).
- **Advection (#144).** `build` takes one scheme and binds it per field with `with_reference_scale`
  (`schemes.md`); a scale-free scheme (first-order upwind, what both step cases run) is unchanged, a
  softened limiter gets `k_ref` / `omega_ref`. `build(velocity_scale=...)` supplies the speed (the case
  passes `flow.reference_speed(momentum)`); a scaled scheme without it is refused at build.
- **The scales, `turbulence_scales(U, h, model)` in `scales.py`** — from the velocity, not the inlet
  turbulence (owner's decision: an inlet's k/ω is not representative of the domain):
  `k_ref = 1.5(0.1U)² = 0.015U²` (`REFERENCE_INTENSITY`; the low end of developed-flow peaks, the safe
  side for `k` positivity) and `omega_ref = sqrt(k_ref)/(β*^¼·0.09h)` with
  `h = flow.wetted_length` = V/A_wall (`OUTER_MIXING_LENGTH_FACTOR = 0.09`, also `sst_initial_fields`'
  default). Outer-flow level on purpose: ω's range is set by the first cell height, so a range-based
  scale would be mesh-dependent. No wall or no speed is refused only when a scaled scheme asks.
  **Argued, not calibrated** — nothing shipped reaches these scales. A local-value softening
  (`eps = K|φ_i|`) was considered and not built: it collapses for `k → 0` at walls.

## The closure — model, strain, sources, transport, preconditioner

- **`sst.py` — `SSTModel`**: Menter's constants, the F₁/F₂ blend, the eddy-viscosity limiter.
- **`strain.py`** — `S = sqrt(2 S_ij S_ij)` through `safe_sqrt` (binding): a double-`where` gives
  `dS = 0` at `S = 0` and is bit-identical to `sqrt` where `S > 0`. Without it a uniform plug (the
  body-force channel's hybrid start, `S = 0` in every interior cell) NaNs the whole coupled Jacobian.
  Zero is the correct local derivative (production reads `S²`; the limiter picks `a1 ω`), and `S = 0`
  never occurs at a sheared root. Do not reintroduce an IC perturbation to "lift" the plug — the
  degeneracy was never in the IC. Pinned in `test_strain.py`.
- **`S` makes the velocity columns of the coupled Jacobian reach one ring further** than everything
  else (`nu_t` reads `S` from `∇u` and multiplies a gradient-based flux) — the cost of differentiating
  the closure exactly. Measurement: log, "`S` makes the velocity columns".
- **`sources.py`** — production / destruction / cross-diffusion as `VolumeSourceFn`s; transport reuses
  the shared advection and diffusion operators.
  - **Both productions are limited at the destruction scale (binding).** `P_k = min(ν_t S², 10β*kω)`;
    ω: `α min(S², 10β*kω/ν_t)` (OpenFOAM's `c1 = 10` form), read from the frozen closure in the
    segregated path and live in the coupled residual. `_EDDY_VISCOSITY_FLOOR` guards `1/ν_t` at the
    `k→0` edge only.
  - **The near-wall `k` row's Jacobian diagonal can be negative** (`P_k ∝ k` through `nu_t`), and the
    pseudo-time shift does not always keep it positive: it does on pitzDaily at `turbulence_damping` 3,
    ⚠️ **not in bfs3d's top-wall / side-wall corner cells once their `k` is small**, where the local shift
    is too weak below β ≈ 0.2. A shifted `k` pivot near zero breaks a **float32** field split around it;
    the materialized preconditioner's pivot floor (`solve.md`, `PivotFloor`) is what keeps the march
    working there, and `_CoupledProblem.floorable_fields = ("k",)` names the rows it may change (a momentum
    row's pivot is small beside its pseudo-time diagonal without being singular). Log, "`sources.py` —
    the `k` row's Jacobian diagonal is negative" and "bfs3d's corner `k` collapse (2026-10-18)".
  - ⚠️ **`explicit_production_limiter` defaults to `False` — the exact operator.** `True` freezes the
    cap's `k` in the linearization only; where the cap binds at the root the IFT adjoint then
    linearizes a different residual and the sensitivity is silently wrong. The cap binds at
    `S/ω > sqrt(10β*) = 0.949` independent of `k` (equilibrium value 0.3; measured root peaks: channel
    0.333, `bfs3d` 0.530), so it binds far from the root, not at it. Opting in is **guarded**:
    `solve_coupled` refuses a root where the frozen cap is active
    (`_reject_a_root_the_frozen_cap_invalidates`, `eqx.error_if`, fires under `jax.grad`);
    `production_cap_active(coupled, state)` reports the cells, and `production_and_limit` is the one
    definition shared by residual and guard. **The opt-in's one home is a bare or weakly preconditioned
    segregated scalar solve**, where the exact cap's indefinite derivative stalls an unpreconditioned
    `k` solve (active in 24 of 24 cells at `test_sst_transport.py`'s start, which asserts that).
    Harnesses: `validation/production_cap_activity.py`, `validation/bfs3d_openfoam/production_cap_activity.py`
    (`BFS3D_PRODUCTION_LIMITER`). Log, "`explicit_production_limiter`".
- **`transport.py` — `SSTTurbulence`, `SSTClosureFields`**: k and ω residuals on the Rhie–Chow mass
  flux, μ_t a frozen per-cell field in the segregated path.
  - `omega_residual` returns a `WallFixedResidual` module, not a closure (#105): it reaches the
    compiled step as an argument, and a closure lands on `filter_jit`'s static side and misses the
    cache every sweep.
- **`preconditioner.py`** — the convection-diffusion AMG for the k/ω Krylov solves. Its frozen operator
  is the shared `precondition.ConvectionDiffusionStencil` (`_scalar_stencil`: `assemble()` with fixed
  cells `detached` for the AMG, `diagonal_parts()` for the shift); its interior conductance is
  `discretization.flux_continuous_conductance(Γ, …)`, harmonic on `Γ = ν + σν_t` — the same conductance
  the residual's `DiffusionFlux` carries (binding, #154), so for pure diffusion the shift diagonal
  equals `diag(jacfwd(residual))` exactly. The reaction+boundary diagonal comes from its own `J·1`.
  - `scalar_transport_preconditioner(scalar=…)` returns a `ScalarTransportPreconditioner`
    (`ConvectionAmgPreconditioner` / `AirAmgPreconditioner`), an `equinox.Module` since #386
    (hierarchies on the traced side; reversed #105 note in `solve-globalization.md`), or **`None` for
    `UnpreconditionedScalars` without assembling the operator** — `None` is what the shift policies
    already read as shift-only. `ScaledScalarPreconditioner(inner, scale)` applies a fixed per-cell
    output factor (the log-transform chain rule).
  - **Scalar block values (#394):** `ScalarBlock` = `ScalarTwoLevel(cycles)` / `ScalarAir(cycles)` /
    `UnpreconditionedScalars()`; `_DEFAULT_SCALAR_BLOCK = ScalarTwoLevel()` is the one home of the
    coupled default; `solve_segregated` defaults to `UnpreconditionedScalars()`. Each value's fields are
    pinned to its preconditioner's minus `hierarchy`. `BlockDiagonal.cycles` is the flow block's alone.
  - **`reuse=old` re-derives values on the old hierarchy's frozen coarsening** (threaded through
    `k_preconditioner`/`omega_preconditioner`). It matters only for `ScalarAir()` — lAIR's C/F split
    reads values, so a rebuild changes shapes and recompiles (`precondition.refresh_air_hierarchy`); the
    two-level aggregation reads only the graph. Reusing across different scalar kinds raises. Pinned in
    `test_scalar_transport_preconditioner.py` (shape preservation where a rebuild provably does not, and
    a refreshed preconditioner beating the stale one).
- **The scalar policy's two halves have different lifetimes (binding, #105).** The shift diagonal is
  rebuilt every sweep (it must track the operator as ν_t grows); the AMG is built once and carried.
  Hence `k_preconditioner`/`omega_preconditioner` (frozen, `scalar=`) vs `k_shift_policy`/
  `omega_shift_policy` (per sweep), each bundling `k_shift_diagonal`/`omega_shift_diagonal`, which
  `_coupled_shift_policy` reads directly (#158) so one definition damps a scalar on both paths. Since
  #386 a rebuilt two-level preconditioner is a cache hit too, so carrying saves only the ~1 s build; a
  rebuilt lAIR still misses. Pinned by
  `test_the_scalar_solve_compiles_once_whether_its_preconditioner_is_carried_or_rebuilt`.
- **`continuation.py`** — `ScalarShiftPolicy` and `scalar_pseudo_transient_solve`: the k/ω solves
  globalized by `solve.PseudoTransientStep` with a transport-diagonal shift (forward-only; its
  `adjoint_solver` is inert).

## The preconditioner as a value — `preconditioner_spec.py`

- `BlockDiagonal` (a `SettingsValue`, `None`-unset fields; `scalar=`, `velocity=`, flow-block options)
  | `MaterializedJacobian(inverse=FieldSplit(leading, trailing) | BlockInverse, probe=
  JacobianProbeSpec(), build_beta, refit_beta_floor)` (from `precondition`; required fields, so not
  `None`-unset). Taken by `solve_coupled`, `coupled_step`, `open_session` and both Reynolds drivers.
  The three materialized inverses share probe/build shift/refit floor and differ only in the inverse,
  which makes a monolithic smoother setting beside a `FieldSplit` unrepresentable.
- **Each spec's field set is pinned to the constructor it feeds** (`test_preconditioner_spec.py`):
  `BlockDiagonal` to `BlockPreconditioner.build` minus `reference_state`; `JacobianProbeSpec` to
  `coupled_jacobian_probe`'s free settings. `FieldSplit` takes `precondition.BlockInverse` values,
  never factory closures.
- **Mapping form:** `PRECONDITIONER_SPEC_MAPPING` is public because `case/spec.py` takes its kinds from
  it; it extends `precondition.MATERIALIZED_MAPPING`. `preconditioner_spec_from_mapping` /
  `_to_mapping` read and write nested plain mappings over `solve.SettingsMapping`: `kind` = class name
  at every level (deliberately — a snake_case vocabulary would be a second spelling), a field equal to
  its default omitted, lists read as tuples. Field values are checked per position from the fields' own
  annotations (#424; the choice sets are `Literal`s pinned to the builders'); unknown kinds/fields and
  a wrong nested kind raise `ValueError` naming the path. The writer refuses non-plain data.
  `test_preconditioner_spec_mapping.py` fails if any public `VelocityBlock`/`ScalarBlock`/`BlockInverse`
  subclass in any subpackage is unwritable. Not built: cross-field refusals (#663 — e.g. `mass_scale`
  beside `schur_scaling: simple` loads and is ignored).
- `MaterializedJacobian.probe` is the one field where absent = default and `null` is refused.
  `JacobianProbeSpec.column_reach` is stored as a tuple of `int`s however it arrives.
- A spec yields a spec, never a preconditioner (#374): session destinations (`observer`, `reports`,
  `on_build`) are not fields.

## Sessions and the one coupled builder — `open_session`, `coupled_step`

- **`coupled_step` is the one frozen-step builder (#371)**; `_BlockSession` and
  `precondition.MaterializedSession` are `solve.PreconditionerSession`s (`build`, `refresh`,
  `refresh_preconditioner`, `rebind`). `solve_coupled`'s default is a `solve.SessionSource` over a
  `BlockDiagonal()` session; an explicit `strategy` or `RefreshPolicy(builder=…)` goes through
  `solve.CallerBuiltSource`. `tools/sibling_builders.py` reports `coupled_step` beside
  `flow_march_step`.
- **March defaults live once, on `coupled_step`'s signature**: a session binds `**march` against it
  (`_march_keywords`), so an unknown keyword is a `TypeError`. `preconditioner` and
  `jacobian_production_viscosity` belong to the session and are refused as march keywords (user
  decision). In `solve_coupled`, `**strategy_kwargs` binds against `coupled_step` likewise.
- ⚠️ **Keywords that cannot be forwarded are refused, not dropped (#278).** `preconditioner` /
  `reference_state` / `**strategy_kwargs` beside an explicit `strategy` or a `RefreshPolicy(builder=)`
  raise (`solve.driver.refuse_unforwardable_settings`); a `**kwargs` door that accepted them silently
  once ran harnesses on library defaults.
- **A frozen `coupled_step` has no refresh hook; a session has `refresh_preconditioner`**, one object
  per session. A loop's `refresh_on_cycles` ends the step and the march calls the hook with `due=True`
  before the next (#290; `solve-march.md`). A materialized session's hook makes the march
  forward-only-observed; a second hook on `RefreshPolicy` beside it is refused.
- **The materialized re-fit (`precondition.BetaTrackingRefresh`) is cost-triggered**: first call, after
  `rebind`, and when `due`. Traps from the deleted β/drift gates: a badly mismatched preconditioner
  collapses α and freezes every drift measure, so a drift trigger cannot catch it; and a refresh
  condition nested in a β gate is unreachable below a preconditioner-only `refit_beta_floor`. Probe
  knobs: `solve-direct-preconditioners.md`. Log, "Materialized re-fit".
- **`_BlockSession` follows a re-pointing (#386)** as a `solve.RebuildingSession`: `rebind` stores the
  companion and the next `refresh_preconditioner` derives a step at the march's state (`_repointed`:
  k/ω hierarchies on their reused coarsening, the flow block's anchor hierarchy applied via
  `flow.BlockPreconditioner.on`) and grafts it into the held step (`ShiftedStep.preconditioned_as`),
  keeping the shift strength. Unrebound it returns `None`. The station step keeps the held structure and
  is a cache hit (`test_a_re_pointed_block_session_step_is_a_compilation_cache_hit`). ⚠️ The re20000
  channel ramp table in `validation/turbulent_channel/README.md` was measured on the first, recompiling
  version and has not been re-measured.
- **A materialized ramp holds the anchor's shift diagonal for the whole march, deliberately (#701).**
  The step control adapts a scalar strength on the premise that the diagonal it scales is fixed; a
  swapped diagonal changes the damping per row by up to ~33× (pitzDaily seed), which the control cannot
  see, and every swapping arm lost on `bfs3d`. The shift vanishes at the root, so only the path
  changes. `MaterializedSession.rebind` re-fits only, and `solve_reynolds_ramp` passes the session's
  `stopping_step(case)` (`solve.StoppingStepSource`) so the stop and its `rtol` reference are read in
  the case's own row scales. Whether the block family should hold the anchor's too is #711,
  unmeasured. Arms: log, "#701 — the arms".
- **The jitted Jacobian probes take the assembler as an argument** (`precondition/materialized_session.py`: `jacobian_matvec` /
  `batched_jacobian_matvec` / `batched_field_jacobian_matvec`, module-level `filter_jit`): a closure over `coupled` is a fresh cache entry
  per rung, while scaling the viscosity leaves the pytree structure identical
  (`test_scaling_the_viscosity_leaves_the_pytree_structure_identical`,
  `test_the_jacobian_probe_is_a_cache_hit_across_reynolds_rungs`).
- **Per-rung recompiles — three causes, all closed:** a fresh preconditioner object per rung (static,
  compared by identity), a value-dependent coarsening in an unused flow block, and a Python-`float`
  viscosity on the static side. **Fit per rung; never build a new object per rung** — share one
  session and `rebind`. Method traps: the process-global compilation cache contaminates a
  leave-one-out cache-key test (fresh jitted wrapper per arm); build a control from your own base, not
  an archived log. Log, "The per-rung recompile".

## Which patches are walls is stated twice, and reconciled at the coupled build (#355)

- The closure names walls in `SSTTurbulence.wall_patches` (wall distance, wall cells, ω fixation); the
  flow block by closures whose `shears_flow()` is true. `CoupledRANS.build` calls
  `refuse_a_wall_set_the_flow_disagrees_with` and `refuse_a_fluid_the_flow_disagrees_with` — methods
  on the closure, since each compares the closure's own declaration with the flow's. Both directions
  of a wall mismatch are silent defects (no ω fixation near a wall / a spurious zero-distance surface).
- Checked, not derived, because an imported mesh drops the polyMesh patch `type` (#364); when that
  lands the set can be derived. `wall_patches` beside `wall_faces` is not a second spelling: only the
  names are a comparable declaration.
- ⚠️ Do not add an `is_wall` predicate: `shears_flow()` is one, and it is solution-affecting (momentum
  selects the wall-function viscosity on those faces).

## Near-wall treatment — `boundary.py`

- **Wall ω is the adaptive blend `omega_wall`, imposed at the wall-adjacent cell centroid (binding).**
  `omega_wall(nu, d, k, model) = [ω_vis^p + ω_log^p]^{1/p}`, `ω_vis = C·6ν/(β₁d²)` (raw branch
  `omega_wall_value`), `ω_log = √k/(β*^¼κd)`; `p = SSTModel.wall_omega_exponent` (2.0, Menter 2003),
  `C = wall_omega_viscous_coeff` (1.0). Reduces to the viscous fixation on a resolved mesh. The viscous
  branch must stay the **centroid** value `6ν/(β₁d²)`; the `60ν/(β₁dy²)` wall-face surrogate at the
  centroid puts ω 10× high — do not "restore" 60 without moving the imposition to the face. The blend
  reads `k` (live `dω_wall/dk` in the coupled Jacobian), finite at `k = 0` via `safe_sqrt` and
  double-`where`-guarded powers, max-factored against overflow; `p`, `C` are differentiable leaves. The
  unit test pins the viscous branch's ODE residual so the 6/60 swap cannot recur.
- **Which blend to ship is an open model decision**: OpenFOAM's `omegaWallFunction` is a hard switch
  at `yPlusLam`; Fluent's default is p = 1.3, C = 1/3. Log, "The blend shape".
- **The fixed cells' ω gradient is imposed analytically, passed *into* the reconstruction (binding).**
  Reconstructed, it was 0.256× exact in the fixed cells and 2.24× in the first ring (pitzDaily);
  ω_wall ∝ 1/d² is convex and the stencil folds in a meaningless wall-face value. It corrupts `∇k·∇ω`
  and F₁ in rows that are *solved*. `omega_wall_gradient = ∂ω_wall/∂d·∇d + ∂ω_wall/∂k·∇k`, partials by
  AD of `omega_wall` itself; `∇d` from `wall_distance_gradient` once at build. `_wall_omega_gradient`
  returns a `schemes.ImposedGradient` that rides `ResidualAssembler.gradient(imposed=…)`, so a scheme
  that consumes its own estimate (`MultipleCorrectionGradient`) builds on the imposed rows. ⚠️ **A
  gradient-scheme A/B cannot detect this class of error** — every scheme is wrong identically; compare
  against the analytical gradient. Measured inert on pitzDaily; do not cite it as an improvement. Log,
  "The imposed wall ω gradient".
- **The ω fixation row is written in the solved variable (binding).** `FixedValueCells` carries a
  `FixationRow` chosen by `ScalarVariableTransform.fixation_row()`: `DirectScalars` → `DifferenceRow`
  (`ω − ω_wall`), `LogScalars` → `LogRatioRow` (`w − log ω_wall`, exactly linear in `w`). A physical
  row under log-ω is not one-step-exact (so the zeroed shift on those cells was wrong) and scales by ω,
  letting ~4 % of cells dominate ‖R‖ — every raw-‖R‖ comparison of march states before 2026-07-25 is
  suspect.
  - **The row's derivative must reach the preconditioner too (binding).** `_reparametrized_preconditioner`
    rescales the physical-operator AMG by `1/(dφ/dw)`, valid only for rows assembled in φ.
    `FixationRow.jacobian_scale(phi, chain)` gives each row its own (`LogRatioRow` → exactly 1), and
    `coupled._row_jacobian_scale` assembles the per-row array; pinned against AD of the row. General
    rule: anything that rescales a block **per row** must ask each row for its derivative. Change the
    row and its preconditioner scaling together.
- ⚠️ **The k shift's reaction diagonal**: `abs` of the net diagonal halves the shift (reverted, #312);
  `|r| + |f|` is inert at damping 3 (deleted). No test tier measures step productivity — measure a
  shift-diagonal change with `validation/pitzdaily_gradient_ab/closure_stall_probe.py analyze` first.
  Ledger: `solve-refuted-directions.md`.
- **The ω shift diagonal inherits ω's range under `LogScalars`** (`d_ω = transport × ω`); a log-space
  shift without the `× ω` wins single steps and loses the march. The cure is the factor split (#156,
  under "The coupled solve").
- ⚠️ **Do not compare against `validation/pitzdaily_openfoam/runs/kwsst/`** — that steady reference is
  corrupt (ω checkerboards); use the time-accurate pimpleFoam field. The OF-vs-aquaflux ω imbalance
  there is a uniform ~13 %, not near-wall. Log, "pitzDaily reference and the max blend", "The first ring
  is not special".
- **Momentum companion: the wall-face eddy viscosity `nut_wall` (binding).**
  `nu·max(0, y*κ/ln(E y*) − 1)`, `y* = β*^¼√k d/ν` (`nutkWallFunction`): velocity-independent, so no
  reattachment singularity; zero below `SSTModel.wall_y_star_lam` (~11), so always-on and a no-op on a
  resolved mesh. `SSTTurbulence.wall_face_eddy_viscosity(k)` →
  `MomentumContinuity.with_eddy_viscosity(nu_t, wall_nu_t)`, applied only on shearing-wall faces via the
  diffusion `boundary_coefficient`, in both forward paths. The `k`-derivative test is against a central
  difference; the log branch's **value** is not integration-verified (coupled fixtures are
  wall-resolved). Log, "`nut_wall`'s derivative test".
- **The near-wall `k` budget needs FOUR pieces together (binding; do not remove one alone).** All cross
  the sublayer/log boundary on one smooth weight `wall_function_weight = tanh((y*/y*_lam)⁴)` (never a
  switch — Newton cannot converge through a jump); `NearWallKClosure` in `sources.py` carries the
  per-wall-cell data. With all four the wall-function channel's `u_τ` is −2.0 % (resolved mesh
  unchanged); with blend + `nut_wall` alone −25 %.
  1. Production carries the **wall-face** shear: `k_wall_production = wall_shear_stress ·
     log_layer_shear_rate(d, k)` (`wall_shear_rate` area-averaged, guarded sqrt). The pure-`k` form
     cancels destruction identically (forbidden).
  2. The wall-face `k` diffusivity is faded (`wall_k_diffusivity = (1−f)γ` via
     `boundary_coefficient`). **Fade the coefficient, not the face value**: a `k`-dependent face value
     amplifies `k` and does not converge.
  3. The wall cells' destruction reads the **live** `omega_wall(k)` (`NearWallKClosure.dissipation_rate`);
     against a frozen ω the row is homogeneous-linear and the k solve runs away.
  4. The closure's strain rate is blended onto `log_layer_shear_rate` in wall cells
     (`SSTTurbulence.strain_rate`), because the SST shear limiter is barely inactive in an equilibrium
     log layer and a reconstructed `S` overshoots it.
  Log, "`nut_wall` and the four near-wall `k` pieces".
- ⚠️ **`y*` scales as 1/ν, so a low-Re anchor switches the log-layer production off** (weight ~1e-7 at
  Re/100) while ω_wall grows 100×: **anchoring a continuation below the Reynolds number at which the wall
  function turns on manufactures a near-wall `k` trough.** The trihedral-corner `k` collapse on `bfs3d`:
  log, "Near-wall k collapse at trihedral wall corners".
- **Known limitation, the buffer layer:** channel `u_τ` error −0.6 % at y+≈68, −2.0 % at 33, −6.9 % at
  16, −5.0 % at 11. Place the first cell in the sublayer or past y+ ~30; do not chase the dip.

## The segregated driver — `solve_segregated`

- **`driver.py`** — outer Picard loop μ_t → flow → k → ω with injected `solve_flow` / `solve_scalar`.
  Coupling is `momentum.with_eddy_viscosity(ν_t, turbulence.wall_face_eddy_viscosity(k))` (kinematic
  `ν_t`; the flow forms `μ_eff`; no `density=`), so a momentum stand-in must accept the second
  argument and a turbulence stand-in must provide `wall_face_eddy_viscosity`. Stops on the coupled
  Picard increment (`_relative_change` < `increment_tol`; `max_sweeps` a backstop that warns); outer
  relaxation is the SER ramp `_sweep_relaxation`.
- **Flow-solve seam: `solve_flow(momentum, state) → (momentum, state)`, typed `flow.drive.FlowSolve`
  (binding).** The assembler comes back because a held-bulk-velocity solve
  (`flow.bulk_velocity_flow_solve`) carries its force on it; there is no separate mass-flow controller.
  Both library builders return this shape (#158; pinned in `test_flow_solve_seam.py` and the slow
  segregated case test).
- **The sweep prologues are jitted (#106):** `_sweep_eddy_viscosity`, `_sweep_closure` (one
  `momentum.flow_fields` for both the closure's gradient and `mdot`); boundaries bound once via
  `turbulence.resolve_boundaries()`. Pinned by `test_segregated_prologues_match_the_eager_assembly`.
- The segregated loop is a forward pre-smoother / fallback, never the sensitivity model.

## The coupled solve — `CoupledRANS` / `solve_coupled`

- **`coupled.py` — the monolithic residual `R(u, p, k, ω)`, nothing frozen** (μ_t, `S(u)`, Rhie–Chow,
  closure all live), over `coupled_rans_layout(momentum.layout)`, a `solve.FieldLayout` nesting the
  momentum layout as its `flow` block (#285). `solve_coupled` hands it to the generic march and attaches
  the IFT adjoint at the root (one transpose solve). `CoupledRANS.build` pre-resolves k/ω boundaries
  (else a dynamic-shape `nonzero` breaks the jit). `CoupledRANS.residual` builds the closure first and
  calls `momentum.flow_fields` once (#106).
  - **The assembly is `residual_of(velocity, pressure, k, omega)` (physical `k`, `omega`, separate
    arrays); `residual(state)` unpacks and calls it** (2026-10-20). `residual_with_field(state, field,
    values)` replaces one per-cell field (layout order `[u.., p, k, omega]`, `values` in the *solved*
    variable) without repacking, so a pressure, `k` or `omega` derivative never forms the velocity
    gradient's tangents — the coloured probe's per-field programs (`_CoupledProblem.probes_by_field`,
    `solve-direct-preconditioners.md`). `_effective_momentum` takes the velocity `(n_cells, dim)`, not the
    flat flow vector, for the same reason. ⚠️ Any new consumer of a field inside `residual_of` must take
    the separate array; reaching back for a flat state re-instantiates every tangent (pinned by
    `test_the_coupled_pressure_and_turbulence_columns_trace_less_than_the_whole_state`).
- **Fluid consistency guard.** Both blocks take a `PropertyModel` (#353); `CoupledRANS.build` refuses a
  closure whose density or kinematic viscosity disagrees with the flow's material properties
  (`refuse_a_fluid_the_flow_disagrees_with`, #157/#367) — the flow block never reads the closure's
  copy, so a mismatch otherwise silently solves the closure at the wrong Péclet/Reynolds number. Build
  time only. `CoupledRANS.with_scaled_molecular_viscosity` moves both copies together.
- **Per-scalar transforms** (`k_transform` / `omega_transform`, default `DirectScalars`): the residual
  is always in physical k/ω, so a transform changes only the Newton iterate space. **Use
  `omega_transform=LogScalars()`, `k` direct (binding)** — ω is the field that goes negative and `log ω`
  is well conditioned; `log k → −∞` at a no-slip wall stalls the near-wall cells. `log(k+1)` bounds
  `k > −1`, not `k > 0`. `solve_coupled` maps in with `state_from_physical` and out with
  `physical_fields`; never a bare `pack_state`/`layout.unpack` at a solve boundary.
- **The reparametrized block's preconditioner and shift are chain-rule scaled at the reference**: each
  scalar shift diagonal × `jacobian_scale(φ)`, the AMG wrapped in `ScaledScalarPreconditioner` by the
  reciprocal; `_reparametrized_preconditioner` returns it unchanged when the factor is one.
- **`k` positivity is a step projection (binding).** Nothing stops a Newton step taking a directly
  solved `k` negative, and the closure's `sqrt(k)` then NaNs everything from one cell (the divergence
  guard sees only the poisoned state). Every coupled builder installs
  `_closure_step_projection(coupled, omega_step_ratio)`: per-cell `k` clip (`positive_k_projection`)
  then a per-cell log-ω bound (`omega_step_bound`, `DEFAULT_OMEGA_STEP_RATIO = e` under `LogScalars`;
  `math.inf` unbounded; finite with direct ω raises). It replaced a global fraction-to-boundary cap that
  one or two corner cells set to ~0.001 (`solve-globalization-log.md`, "bfs3d's retried steps"). Trap
  for any future cap: a projection applied before a limit leaves the limit nothing to bind.
  ⚠️ **The clip only shrinks a cell whose correction keeps pointing down (by `1 − tau` a step); it does
  not cure it.** Where the linearization points below zero and the root does not (bfs3d's corners), the cell
  falls through every scale and the clipped direction can stop descending. Log, "bfs3d's corner `k`
  collapse (2026-10-18)".
- **There is no per-cell production freeze** (removed 2026-10-19, owner's decision; dominated by the
  pivot floor): `jacobian_production_viscosity` (every cell) is the only production freeze. Ledger, "Per-cell
  production freeze — REMOVED (2026-10-19)", says how to reopen it.
- **Stopping measure (binding).** `solve_coupled(convergence=Convergence(measure=…))` names it:
  `RowScaled()` (default — `coupled_scaled_norm`, rebuilt at every outer iteration), `BlockScaled()`
  (held at the initial state), `Euclidean()`. The march builds it from `_CoupledMeasures` at each
  iteration's start and hands it to the step; `_coupled_step`'s default linear solve stops in it
  (`norm=None`). The Euclidean norm is ω-dominated and mis-ranks states. The row-scaled measure does not
  cure the pitzDaily plateau; it makes it honest. Builders take no measure keyword.
- **Linear-solve regime** is per family (`solve.LinearSolveRegime`; block: `_BLOCK_LINEAR_SOLVE`,
  rtol 0.3, restart 120), set through `linear_solve=LinearSolveSettings(...)`. ⚠️ Passing a whole
  solver as `linear_solve` replaces the measure too. Unset `Globalization.line_search` takes
  `_COUPLED_LINE_SEARCH` (the full step from the hybrid start overshoots by orders of magnitude). Log,
  "The coupled line search", "The forward stopping measure".
- **`Globalization.beta_floor` is off by default** (a measured wash). The coupled flow block uses
  `ConvectionTwoLevel()` frozen once (refreshing it does not pay; refreshing the k/ω AMGs does); override
  via `BlockDiagonal(velocity=…)`. `_coupled_shift_policy` leaves `schur_scaling` at
  `BlockPreconditioner`'s default; MSIMPLE stays for flow-only convection-dominated solves. Log,
  "`beta_floor` and the convection-aware velocity AMG", "MSIMPLE no longer hardcoded".
- **The convection velocity block freezes at the policy's own reference flow (#389)** whenever it
  carries mass flux, else the builder's uniform plug. The plug cannot build on a contracting duct (its
  upwind operator loses positivity on the first coarse level); pinned by
  `test_velocity_block_reference_state.py`. ⚠️ **Not fixed: a closed domain at rest (#706)** — the
  6×6 lid cavity still takes the plug, whose hierarchy has a negative coarse diagonal, so the default
  block cannot build there from the hybrid start. A divergence-free at-rest default is the open
  question. Log, "#389".
- **Refresh segments (`refresh=RefreshPolicy(trigger=…)`).** The march runs as segments of one
  `solve.newton_march` (`solve.staged_march`); the k/ω AMGs are re-derived between them (off-jit scipy
  work). The trigger is **`ν_t` drift** (`CoefficientDriftTrigger` on `eddy_viscosity_drift`, re-based
  per segment) — the coefficient the frozen operators are built from; residual- and cycle-count
  triggers are dominated (cost also rises with the β ramp). `max_steps` applies per segment; the
  stopping target is measured once and held; the **last segment runs without the trigger**
  (`test_the_last_refresh_segment_marches_without_the_trigger`).
- **A refresh carries the shift's coordinate factor and rebuilds only its transport time scale
  (binding, #156).** `d = transport_diagonal(state) × jacobian_scale(field)` with `jacobian_scale(ω) = ω`
  under `LogScalars`; rebuilding the product at a developed state over-damps and freezes the march
  (independent of β). `CoupledShiftPolicy` stores `k_shift_transport`/`k_jacobian_scale` (likewise ω)
  separately; `_coupled_shift_policy(..., reuse=…)` rebuilds the transport and takes the scales from
  `reuse`. The preconditioner's own `k_scale`/`omega_scale` are re-derived (its AMG is refreshed at
  the new operator). Safe because the shift vanishes at the root.
- **Everything works under `jax.grad`** (refresh, re-fit, controls, retries): the march runs on
  `stop_array_gradients` copies and `root_adjoint` attaches at the root. ⚠️ Anything a hook caches must
  come from those copies (else a later `UnexpectedTracerError`); the solve cannot run inside
  `jit`/`vmap`/`scan` (`refuse_a_transform_the_march_cannot_run_in`); a session the caller re-points at
  a live assembler inside `jax.grad` refuses to build.
- **Observers** `on_step` / `on_checkpoint` / `on_residuals` work without a trigger and change nothing
  (`solve-march.md`).
- **Step control:** a dual-time march defaults to `DualTimeControl` (`default_dual_time_control`, `solve/step_control.py`);
  `ResidualRatioDualTimeControl` is opt-in (it pins β on the pitzDaily plateau).
- ⚠️ **The constrained (held bulk velocity) adjoint test is decided at β = 0 against the block-diagonal
  preconditioner**, its weakest regime, and passes an explicit `relative_residual_gmres(1e-8,
  restart=120, stagnation_iters=200)`. Log, "The constrained adjoint test".
- **A held bulk velocity is `solve_coupled` on a `MassFlow` drive (#128/#541).** `coupled_border(coupled)`
  borders the residual, measures (`flow.BorderedMeasures`, the constraint row scaled by the held speed),
  drift, homotopy and source; `_BlockSession._step` borders its policy; the root's adjoint is
  `_BorderedCoupledResidual`, so `jax.grad` is at fixed bulk velocity (FD-verified in
  `test_coupled_mass_flow.py`). `held_force` reads the force back exactly at a root. A
  `MaterializedJacobian` beside one is refused. The case files keep `Segregated` for a held bulk
  velocity (re240000 is converged by no coupled arm). Flow side: `flow.md`, "A held bulk velocity on
  every march". Log, "Held bulk velocity on the law-of-the-wall files".
- History of the pitzDaily cost investigations (2026-07): log, "pitzDaily cost history".

## Initialization, diagnostics, Reynolds continuation

- **`initialization.py` — `sst_initial_fields(momentum, turbulence, *, k_floor, omega_floor,
  length_scale_factor)`**, registered for `CoupledRANS` with `aquaflux.initialization.hybrid_initialize`
  in `coupled_initialization.py` (its own module to avoid a cycle). `solve_coupled` with no state calls
  it. The monolithic Newton is local and stalls from a raw cold start; this gives potential-flow velocity
  (`flow.potential_flow`), Laplace-smoothed k, and ω = the interpolant raised (`jnp.maximum`) to the
  **same `omega_wall` the residual imposes**, at every cell's wall distance — the exact near-wall
  balance profile, a smooth ramp in `log ω`. Takes no `gradient_scheme` (#361): each field is smoothed
  with its own residual's scheme. A forward device; to differentiate, pass a state built outside
  `jax.grad`.
  - **The seed must be the full blend, after `k` is settled (binding)**: seeding the viscous branch only
    raised pitzDaily's ‖R₀‖ ~350×, invisible on a resolved mesh — the unit test asserts on a coarse
    high-Re mesh where the branches differ.
  - **Body-force domains:** k and ω from the force balance's `u_τ = √(βh/ρ)`
    (`flow.friction_velocity`): `k = equilibrium_k`, `ω = inlet_omega(k, 0.09h)`, both raised together
    or not at all (raising k alone with ω at its floor gives `ν_t ~ 10⁶`). Pinned by
    `test_hybrid_initialize_gives_a_developed_channel_eddy_viscosity`.
  - **Inlet-driven wall-bounded domains:** walls dominate the harmonic k, so k is floored at
    `jnp.max(k)` (the inlet value); ω needs no floor (zero-gradient walls). Pinned by
    `test_hybrid_initialize_floors_inlet_driven_k_at_the_turbulent_level`.
  - At a Reynolds handover `wall_consistent_omega` / `wall_consistent_state` (via
    `seed_projection=`, default off) re-impose the fixation rows; correct, measured to buy no steps
    (log, "The wall-consistent seed").
- **`measures.py`** — `coupled_scaled_norm(coupled, shift_policy, state)` and
  `coupled_equation_names(dim)` → `(u, v[, w], p, k, omega)`, the one home of the block names (built on
  `flow.flow_equation_names`), which lets the march report per equation.
- **`diagnostics.py`** — `coupled_fields(coupled)` (named physical fields for `field_change_metrics`,
  velocity per component, `ν_t` included) and `coupled_residuals(coupled, continuation,
  reference_state=None)` (per-equation residual on the march's measure). Both compiled
  (`_field_values`, `_per_block_residual`, assembler an argument); names are attached outside the
  compiled program because a returned dict comes back key-sorted. ⚠️ **Equilibrate at the previous
  state, never the logged one** — the march holds the measure from each iteration's start, so the
  reporter is stateful, called once per step in order, and one built mid-march must be seeded with that
  segment's `seed_state`.
- **`ViscosityRampHomotopy` (`reynolds.py`) — the Reynolds span inside ONE march**
  (`solve_coupled(homotopy=…)`), stations from a `solve.RampSchedule(stations, steps_per_station,
  end_when=, finish=)` (`solve-march.md`). Prefer it to the rung ladder, which converges each seed to a
  bar the next viscosity jump undoes and restarts the β ramp per rung (`solve-march-log.md`,
  "`ResidualHomotopy`").
  - The target station is the **caller's own assembler by identity**, so root and adjoint belong to
    the case (pinned per strategy).
  - **`rebind`** is called only when the station changes; `solve_reynolds_ramp` passes the shared
    session's `rebind` (either family since #386). A plain mutable object, forward-only.
  - **`redamping` defaults to `ratio ** _REDAMPING_EXPONENT` (0.6)** — the station's own size; forced to
    1.0 (explicit value refused) at `steps_per_station == 1`, where β would otherwise run away as
    `beta_start · redamping^n`. An empirical one-point fit.
  - **`companion=`** — `scale_both_blocks` (default; a genuine lower-Re problem) or
    `scale_momentum_only`, which leaves the closure's ν alone so anchor and target hybrid starts are
    identical (`test_scaling_momentum_only_makes_the_anchor_and_target_hybrid_starts_IDENTICAL`).
    ⚠️ An intermediate station is then **not a physical Reynolds number**; quote nothing from it.
  - **Shipped schedule:** both step cases' `case.yaml` run a 12-station, one-step-per-station ramp from
    anchor 100 with `scale: flow`, `turbulence_damping` 3 and `CflResidualDualTimeControl` at `grow` 3.
    The station count is the shift's descent in disguise. `PITZ_RAMP`/`BFS3D_RAMP=off` in the
    `compare.py` drivers is a comparison arm. Sweeps: log, "ViscosityRampHomotopy".
  - **`turbulence_damping`** (`CoupledShiftSettings`) multiplies the k/ω rows' shift **strength**
    (`ShiftTerm.row_shift_scale`, a closure of β), ⚠️ **never the base diagonal**, which is also the row
    scale of `coupled_scaled_norm` — folding it in silently moved each arm's stopping bar (pinned both
    ways). Strategies: `ConstantDamping(ratio)` (ratio stored as a JAX array so swapping it is a cache
    hit), `BetaTaperedDamping`, `ResidualTaperedDamping`. Ending at 1 is a path choice; the shift is
    zero at the root regardless.
  - ⚠️ A wrapper that rebuilds a `ShiftTerm` from `.diagonal` drops its other fields silently, and a
    `DualTimeStep` must hand the policy its residual; find a Protocol's implementers structurally (AST),
    not by grepping a signature.
  - **`station_step=`** `(step, station, arrived) -> step` on `solve_coupled`/`newton_march` reshapes a
    step per station; must swap array leaves over a fixed structure (`eqx.tree_at`). Its measured
    verdict was negative; kept as the only way to express a per-station setting.
- **`solve_reynolds_continuation(coupled, n_points, *, schedule, intermediate, point_setup,
  seed_projection, **solve_kwargs)` — the rung ladder.** Each point seeded by the previous root; an
  outer wrapper around `solve_coupled`, agnostic to its globalization.
  - **It dissolves at the target**: the last solve runs on the live `coupled`; `n_points=0` is
    bit-identical to a direct solve, and `jax.grad` is the target's IFT adjoint, `n_points`-independent
    (companions from a stopped copy, each seed stopped).
  - ⚠️ **Keywords split both ways (#278):** a pre-built `strategy`/`reference_state` reaches the target
    only; everything that configures a build (`preconditioner`, `coupled_step` keywords) reaches the
    ramp only (`_SOLVE_ONLY`, derived from `solve_coupled`'s signature). A `MaterializedJacobian` spec
    opens one session shared and `rebind`-ed per point; a `BlockDiagonal` is built per point.
  - ⚠️ **A forwarding wrapper hides call sites**: when changing `solve_coupled`'s contract, scan
    `solve_reynolds_continuation` and `solve_reynolds_ramp` callers too.
  - **`intermediate=Convergence(rtol=1e-2)`** by default; unset fields from the target's
    `convergence`; `None` disables. Prefer an absolute bar (`atol`, target `rtol=0`) on a row-scaled
    measure — a relative bar re-bases per rung. Budget by reaching the Newton basin (α → 1 with cycles
    collapsing). Log, "An absolute intermediate bar".
  - **`point_setup(companion, seed_state, point) -> dict`** — per-point march settings, given a
    `ReynoldsPoint` (`index`, `total`, `viscosity_scale`, `is_target`, `label`); setting it materializes
    the lowest seed. ⚠️ Under `jax.grad` the target companion is live — build from
    `stop_array_gradients(companion)`. A refresh hook from `point_setup` beside a session is refused.
  - **Schedules (`ReynoldsSchedule` Protocol), asked one scale at a time:** `anchor(n_points)`,
    `next_scale(converged, failed)`, `planned_total(n_points)`. `GeometricReynoldsSchedule` (default,
    a decade per point) returns `None` on failure — as unforgiving as ever (pinned against the adaptive
    one). `AdaptiveReynoldsSchedule` bisects in `ln(scale)` on failure, keeps converged rungs, gives up
    below `min_ratio` 1.05, regrows by `recovery` 1.5; it reacts only to convergence, since three
    attempts to predict a bad step failed. The target takes the same path, so it can retreat too.
  - A failed lower-Re point is re-raised as a `RuntimeError` naming it; the march never continues from
    a non-root. Rung count on `bfs3d` and the 2026-09-08 ladder studies (tangent predictor, finer
    ladders, warm `beta_start`, path geometry, literature): log, "How many rungs", "Reynolds-continuation
    ladder studies".
- **`with_scaled_molecular_viscosity(factor)`** on `CoupledRANS` composes `MomentumContinuity`'s and
  `SSTTurbulence`'s (μ and ν move together; density untouched) over `PropertyModel.with_scaled`
  (`properties.md`). The case is scaled, never restated.

## Binding decisions

- **Segregated forward, coupled adjoint.** Sensitivities are the implicit-function-theorem solve on
  the unfrozen `R_coupled(k, ω, U, p; params) = 0` at the root — `CoupledRANS.residual` through
  `solve_coupled`, one transpose solve (FD-verified). At the fixed point the frozen fields equal the
  live ones, so the segregated loop is absent from the sensitivity model. Differentiate
  `solve_coupled`, never `solve_segregated` (forward-only; differentiating it would unroll the Picard
  sweeps onto the tape). Build a continuation for a differentiated solve outside `jax.grad`.
- **Never unroll an outer loop onto the differentiation path**; a coupled solve without the IFT adjoint
  is an intermediate and its adjoint must be a tracked issue.
- **Convergence-based outer stop** (`increment_tol`, per-field relative change), never a fixed sweep
  count.
- **Globalize the loop and the scalar sub-solves** like the flow block (scalar `ShiftPolicy`
  continuation; SER ramp); constant under-relaxation plus floors is a stabilizer of last resort.
- **Positivity floors must be inactive at convergence (adjoint honesty).** A `max(·, floor)` clamp has
  zero gradient where active, so a sensitivity through a clamped cell is wrong — check the
  precondition, surface a violation. `LogScalars` on ω removes the floor structurally. The ω floor is
  the k-tied realizability floor `ω ≥ k/(nut_max_coeff·ν)` (`nut_max_coeff = 1e5`, #126), inactive for
  any physical field; `omega_floor` is only an absolute backstop. Pinned by the law-of-the-wall test.
- **Frozen coupling data rides as injected pytree leaves** (μ_t, frozen ∇u, mdot); no `Calculated`
  property re-coupling μ_t ↔ (k, ω) inside the segregated residual.

## Testability seam

- `solve_segregated` takes injected `solve_flow` / `solve_scalar`, so it is tested with stub solvers
  with a known fixed point (`tests/unit/test_segregated_convergence.py`).
- Every operator (sources, strain, transport) has an operator-level test on an analytic field.
- The coupled solve has an iteration-count-independent adjoint gate (root `CLAUDE.md`, Testing
  Architecture), not only a smoke test.
- **Never assert the positivity floors back** — `min(k) >= 0`, `min(ω) > 0`, `max(μ_t)/ν > 1` hold for a
  diverged field too. A segregated test must assert the increment reached `increment_tol` (the driver
  only warns) or agreement with an independent root; model: `test_coupled_rans.py` (to `1e-9`, matches
  the coupled root to 1e-4).
