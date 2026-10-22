---
paths:
  - "aquaflux/discretization/**"
---

# Rules — `aquaflux/discretization/` (the Layer-0 residual substrate)

> **Provenance boundary (binding).** This file may cite the C++/Fortran precursors; none of that may
> reach the shipped surface (`.py`, `docs/`) — root `CLAUDE.md` Comment Convention.

Assemble the discrete cell residual `R(state, params)`; AD produces the Jacobian/adjoint. Dated
measurements and verification runs: `.claude/notes/discretization-log.md`.

## Binding decisions
- **No hand-derived linearization, ever.** Write the full physical flux — including the non-orthogonal
  correction — as one residual term; AD puts it in the matrix (both precursors defer it to an explicit RHS).
- **Operators are injected strategy classes** (`equinox.Module`s, side-effect free); the assembler holds a
  list of them and imports no specific operator, so a test can assemble from one stub.
- **System-first**: the residual is over the whole state vector; coupling is inferred by AD from what a
  term reads.
- **Sign convention**: every `FaceFluxOperator` returns the **owner-outward flux of the conserved
  quantity**; `R = accumulation + Σ scatter(outward flux)` (owner `+`, neighbour `−`). Advection returns
  `+mdot_f phi_f`, diffusion `−Γ(∇φ·n)A`.
- **Compose, do not re-derive**: face geometry from `aquaflux/mesh`, interpolation/gradients/limiters from
  `aquaflux/schemes`. Gather by direct indexing on `face_cells.owner`/`safe_neighbour`; scatter through
  `face_cells.scatter_conservative` (in `CellBalance`). This package owns the physics, not `segment_sum`.

## The context and self-describing terms
- **No monolithic gathered-state bundle.** Operators get a lean `FieldContext` (`aquaflux/context.py`): a
  shared `MeshContext` (`face_cells`, `geometry`, `properties` — the evaluated `{name: (n_cells,)}` map)
  plus one field's `boundary_values`/`gradient` (the gradient is a solve, formed once). Each operator
  gathers its own owner/neighbour fields. Adding a coefficient is a `PropertyModel` entry, never a context
  field (`.claude/rules/properties.md`); there is no `gamma` field. Do not reintroduce a union bundle.
- **`FieldContext` lives in `aquaflux/context.py`, not here — do not move it back.** It imports only
  `aquaflux.mesh`, so `schemes/` and `boundary/` can consume it without a cycle. `Limiter.limit(field,
  context)` uses it; `GradientScheme` and `BoundaryCondition` deliberately do not (`schemes.md`,
  `boundary.md` give the reasons).
- **`term.py::DeclaredInputs`** — `requires() -> tuple[str, ...]` (default `()`) and `uses_gradient() ->
  bool` (default `False`), inherited by **every** term family: `FaceFluxOperator`, `AdvectionScheme`,
  `VolumeSource`, `TransientTerm`, flow's `MomentumSource`. `AdvectionFlux` delegates both to its scheme.
  `ResidualAssembler.build` checks every term (fluxes, sources, transient):
  `properties.require(*declared_properties(terms))` (`declared_properties` is the one union helper, also
  used by `MomentumContinuity.build`), and refuses `gradient_scheme=None` when any term `uses_gradient()`,
  naming it. Both are build-time `ValueError`s, never a `KeyError` inside a jitted residual.
  - **`uses_gradient()` means: on an orthogonal grid, does a zero gradient change the answer?** So
    `DiffusionFlux` reads the gradient but answers `False` (its correction vanishes there — graceful
    degradation); `LimitedUpwind.uses_gradient()` is unconditionally `True`.
  - `DiffusionFlux.requires()` returns `(self.coefficient,)`.
  - **Declarations are falsified, not trusted** — `tests/unit/test_declared_inputs.py` requires a `CASES`
    entry for every concrete `DeclaredInputs` subclass (new term without one fails; stale fails too) and
    checks: evaluates against exactly its declared properties; each declared property **NaN-poisoned**
    makes the answer non-finite (NaN, not zero, since a silent zero is the defect); `uses_gradient()`
    matches the definition above. Assembler-level sweeps: `test_residual_assembler_validation.py`.
  - ⚠️ No shipped `VolumeSource` reads `context.gradient` (SST sources take frozen fields), so the source
    sweep has nothing to refuse yet — it exists for the first one that does.
  - `TransientTerm` declares nothing and has **no accumulation coefficient**, so `ρ c_p dT/dt` /
    `d(εRC)/dt` cannot be written yet (#655); when it lands, declare the coefficient via `requires()`.
- ⚠️ **`gradient_scheme=None` means no reconstruction, and only that.** It is right for pure diffusion on
  orthogonal meshes, so `ResidualAssembler.build` and `ScalarTransport.build` do not take the library
  default that `MomentumContinuity.build`/`SSTTurbulence.build` carry. `None` never means "choose for me".
- **Boundary-side checks in `build`:** `BoundaryCondition.requires_coefficient()` (`Neumann`/`Convective`)
  adds the assembler's `coefficient` to the same `require` call; `refuse_a_closure_that_closes_other_fields(
  boundary, HOST_EQUATION_FIELD, ...)` refuses a multi-field flow bundle by patch name
  (`.claude/rules/boundary.md`).

## Modules — BUILT
- **`face_flux.py`** — `FaceFluxOperator` (`face_flux(field, context)`), the contract `diffusion.py` and
  `advection.py` both depend on (not on each other).
- **`diffusion.py` — `DiffusionFlux(coefficient=…)`** reads `context.mesh.properties[coefficient]`
  (default `"diffusivity"`; `"viscosity"` for momentum). Flux-continuous non-orthogonal normal derivative:
  one-sided extrapolation from each centroid to the integration point, the face value eliminated via
  `Γ_P dφ/dn = Γ_N dφ/dn`, giving `[(φ_N − φ_P) + corr_N − corr_P] / denom`,
  `denom = (D_P·n) − (Γ_P/Γ_N)(D_N·n)`. Unlike the over-relaxed area-vector split (Jasak), it decomposes
  the centroid-to-face vector, uses each side's own gradient, folds skewness into `corr`, and handles
  coefficient jumps natively.
  - **`flux_continuous_denominator` and `flux_continuous_conductance` are the one home (binding, #154).**
    The conductance `Γ_P A / denom` is exactly `d(owner-outward flux)/d(φ_P)` (harmonic mean on an
    orthogonal graded face), used by momentum `a_P` (`flow/rhie_chow.py`), the scalar pseudo-time shift and
    frozen AMG operators (`turbulence/preconditioner.py`, `flow/block_preconditioner.py`). ⚠️ Never
    re-derive a conductance with an arithmetic `Γ` interpolation: it over-counts a graded face by
    `(1+r)²/(4r)` (`.claude/rules/flow.md` `a_P` note).
  - **`boundary_coefficient`** (`(n_faces,)`, default `None`) overrides the owner `Γ` on boundary faces
    only. Consumers: the momentum wall-function eddy viscosity and the k equation's wall-face diffusivity
    `(1-f)·γ` (`.claude/rules/turbulence.md`). Pattern: a boundary flux that must vanish smoothly fades the
    **coefficient**, not the face value — a state-dependent face value enters the linearization and
    diverged for k.
- **`residual.py` — two objects, context half and balance half (binding — do not re-fuse).**
  `ResidualAssembler.build(...)` binds an injected `BoundaryConditions` (`resolve` off the jit path) and
  builds the **context**: gradients once (injected `GradientScheme`, optional), boundary closures,
  `PropertyModel`. **`CellBalance`** assembles the **balance**: sum the `FaceFluxOperator`s, scatter, subtract
  `VolumeSource`s, add the transient (`R = accumulation − transport`). The assembler holds it as `balance`.
  - `CellBalance` stores **only** `flux_operators`, `source_operators`, `transient`; everything else comes
    on the context, so it is testable with a hand-made context (`test_cell_balance.py`).
  - Why split: `MomentumContinuity` builds its own context (one velocity-gradient tensor from `FlowBoundary`
    closures) and drives a `CellBalance` per component; this is what made the pressure term an ordinary
    operator (`flow.PressureForce`). Both assemblers compose `schemes.BoundaryClosure` for the two-pass
    reconstruction and `jvp` weight probes, so `_gradient` and `_build_time_boundary_linearization` are
    one-liners over it.
  - ⚠️ **Flux-operator tuple order is the summation order** (float addition is not associative), and
    archived march trajectories are compared bit-for-bit. Momentum's order is fixed at
    `(DiffusionFlux, PressureForce, AdvectionFlux)`; pinned by
    `test_cell_balance.py::test_operators_are_summed_in_tuple_order`.
  - ⚠️ For any bit-identity comparison, include quantities the change should **not** touch: per-process
    `hash()` randomization once produced order-one differences that a residual-only check would have read
    as a regression.
  - **`residual(phi, *, gradient_hook=None)`** — a `gradient -> gradient` transform for the distributed
    residual's ghost rows, applied after reconstruction and threaded into
    `GradientScheme.gradients(operator_hook=…)` → `GradientSolve.solve(operator_hook=…)` for iterative
    schemes (`.claude/rules/parallel.md`).
  - **`residual(..., fields=None)` / `gradient(phi, *, fields=None)`** forward a caller-supplied mapping to
    `PropertyModel.evaluate`; the assembler never inserts its own `phi`. Pure plumbing until `Calculated`
    lands (`.claude/rules/properties.md`).
- **`transient.py` — `TransientTerm`**: BDF1 at step 1 (static `first_step`), BDF2 after.
- **`source.py` — `VolumeSource`** (`source(field, context) -> (n_cells,)`): volume-integrated, production
  positive, subtracted by the assembler. The full nonlinear source goes in the residual (no Patankar
  `S_C`/`S_P` split). Frozen coupling fields are constructor state. This is where turbulence production
  and aquakin reactions attach.
- **`fixed_value.py` — `FixedValueCells` + injected `FixationRow`**: replaces chosen cells' rows with an
  algebraic constraint (pinned reference pressure, near-wall ω); the target is a differentiable leaf.
  - **The row form must match the solved variable (binding).** `DifferenceRow` (`phi − target`) when the
    unknown is `phi`; `LogRatioRow` (`log(phi/target)`) when it is solved in log form, where it is exactly
    linear with derivative 1. A difference row under a log unknown is exponential: it overshoots and lets a
    few fixation cells dominate the residual norm. The turbulence `ScalarVariableTransform.fixation_row()`
    chooses; do not branch on the transform inside the residual.
  - **`jacobian_scale(phi, chain)`** returns `d(row)/d(unknown)` (`DifferenceRow` → `chain`; `LogRatioRow`
    → `chain/phi`). Anything rescaling a block per row must ask each row: rescaling ω fixation rows by
    `1/ω` once stalled the Krylov solve (`.claude/rules/turbulence.md`). Pin implementations against AD of
    their own `row`.
- **`advection.py` — `AdvectionScheme` → `FirstOrderUpwind` (linear, monotone), `LimitedUpwind`
  (`phi_f = phi_C + psi_C ∇φ_C·(x_f − x_C)` from the upwind cell); `AdvectionFlux(mass_flux, scheme)`.**
  - **The mass flux is always injected, never built here.** In the coupled flow it is the Rhie–Chow
    `mdot` (`flow/momentum.py`, via `interior_mass_flux`), the same array feeding momentum convection and
    continuity; a transported scalar reuses the flow's flux (`.claude/rules/transport.md`).
    `tests/support/fields.py::face_mass_flux` (`(u·n)A` for prescribed divergence-free fields) is
    verification-only.
  - The limiter lives in `schemes/limiter.py` (`VenkatakrishnanLimiter`), held by
    `LimitedUpwind(limiter=…)` and formed only when that scheme runs (`None` → `psi = 1`). ⚠️
    **`advection.py` imports `Limiter` at run time, not under `TYPE_CHECKING`** — the case reader's
    `SettingsMapping` resolves the annotation with `typing.get_type_hints`, and the registry would raise
    `NameError`. No cycle: `schemes/` imports nothing from `discretization/`.
  - **The limiter is in the residual and AD linearizes it**, recovering quadratic Newton convergence where
    the precursor's lagged limiter converges linearly (`tests/integration/test_limited_advection.py`). The
    smooth limiter damps rather than eliminates over/undershoot — the smoothness is what makes it
    AD-linearizable.
  - `AdvectionScheme.with_reference_scale(provider)` sets a scaled scheme's scale (`schemes.md`).
    `ResidualAssembler` does not call it (it does not know its field); callers must.
- **Gate C (skewed mesh).** With `CorrectedGreenGauss` injected the residual is affine in φ, so Newton is one
  step and exact on a 25 %-skewed grid (`tests/integration/test_skewed_diffusion.py`).
- **Gradient-boundary circularity — resolved for every Green–Gauss scheme but the Hessian-corrected one.**
  Closures are affine in the owner gradient (`phi_f = phi_f0 + w·∇φ_P`); `CompactGreenGauss`,
  `CorrectedGreenGauss` and `MultipleCorrectionGradient` move `w·∇φ_P` to their left-hand side via
  `schemes.boundary_gradient_block`, so Gate C holds with Neumann walls too
  (`test_gate_c_holds_with_flux_walls_on_a_skewed_mesh`). `HessianCorrectedGradient` still uses `phi_f0`
  alone (#648 follow-up; `schemes.md`).

## ⚠️ Single-threaded kernels in the compiled residual and its JVP are NOT a lever

Measured 2026-10-17 on pitzDaily (4-core Linux, jax 0.11.2; harness
`validation/fusion_profile.py` (`FUSION_CASE=pitzdaily`), tables in the log): no kernel is worth more than ~1 % of
the coupled residual or its JVP, and respelling the reduction-rooted fusions through `vectors.dot` removed
them without a wall-time gain. Do not pursue per-kernel threading rewrites here.
- ⚠️ **A standalone JVP timing on Linux is dominated by page faults the march does not pay**: the JVP's
  ~43 MiB temporary exceeds glibc's adaptive mmap threshold (32 MiB cap), so it is `mmap`ed per call
  (~11 000 faults). Raising `MALLOC_MMAP_THRESHOLD_`/`MALLOC_TRIM_THRESHOLD_` cut it 16.5 → ~10 ms and
  changed the shipped march not at all. Quote a standalone JVP time with its fault count and never project a
  march gain from it.

## Testability seam
- The scatter engine is testable with a 2–3 cell mesh and a stub closure (assert cell by cell).
- Each operator ships an order-of-accuracy unit test on an analytic field, independent of any solve.
