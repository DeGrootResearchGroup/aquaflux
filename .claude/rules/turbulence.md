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
    (#126). `KProduction.explicit_limiter` still freezes *its* cap's solved `k` for the M-matrix
    forward path; ω needs no such flag (its cap is already field-independent).
- **`transport.py` — `SSTTurbulence`, `SSTClosureFields`.** Assembles the k and ω scalar
  transport residuals on the flow's Rhie–Chow mass flux, with μ_t a **frozen per-cell field**
  recomputed once per outer sweep.
- **`preconditioner.py`** — the convection-diffusion AMG preconditioner for the stiff k/ω scalar
  Krylov solves at high Reynolds number (the scalar analogue of the velocity-block work). It assembles
  its frozen operator with the shared `aquaflux.solve.frozen_operator.convection_diffusion_operator` and
  hands the **assembled matrix** to `build_convection_hierarchy` / `build_air_hierarchy` (the coarsening
  library is operator-in, #45); its reaction+boundary diagonal still comes from its own `J·1`
  derivation, which is a genuinely different source, not a copy of the interior stencil. Its interior
  diffusion coupling (`_scalar_operator_pieces`, feeding both the AMG operator and the pseudo-time shift)
  is `discretization.flux_continuous_conductance(Γ, geometry, face_cells)` — the scalar transport
  operator's own diagonal contribution, harmonic on a graded diffusivity `Γ = ν + σν_t`, the *same*
  conductance the k/ω residual's `DiffusionFlux` carries (binding, #154). It replaced a g-weighted
  **arithmetic** face `Γ` that agreed only for constant `Γ` and over-counted a graded face by
  `(1+r)²/(4r)`; for a pure-diffusion scalar the shift diagonal now equals `diag(jacfwd(residual))`
  exactly (pinned in `test_scalar_transport_preconditioner.py`).
  `scalar_transport_preconditioner` returns a **`ScalarTransportPreconditioner`** strategy
  (`ConvectionAmgPreconditioner` / `AirAmgPreconditioner`) rather than the old opaque `lambda phi: solve`.
  These are plain frozen dataclasses, **not `equinox.Module`s** — see the binding note in
  `.claude/rules/solve.md`; making them pytrees breaks both the IFT adjoint and the jit cache.
  `ScaledScalarPreconditioner(inner, scale)` wraps one with a fixed per-cell output factor — the
  reciprocal chain-rule scaling a log-transformed scalar block needs (above); also a frozen dataclass.
  - **`solve_coupled(refresh_trigger=…)` segments the march to re-freeze the preconditioner — and a refresh
    must CARRY the shift diagonals, not rebuild them (binding).** With a trigger set, the march runs as a
    sequence of *observed* segments (`aquaflux.solve.forward_march`): each steps until the trigger judges
    the frozen preconditioner stale, the k/ω AMGs are re-derived at the state reached, and the next
    segment continues — then a real `ImplicitNewtonSolver.solve()` finishes and produces the result.
    Segments exist because the AMG rebuild is off-jit scipy work that cannot run inside the
    `lax.while_loop`; that part is not subtle. **The trigger is the drift of `ν_t` since the freeze
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
    `(refresh_limit+1)·max_steps` march steps plus the finishing solve's own allowance, deliberately not
    split — either segment may need the full allowance); the finishing solve is handed the **absolute**
    target `atol + rtol·‖R0‖` measured at the initial state, so a refreshed solve stops exactly where an
    unrefreshed one does for **any** number of refreshes (a relative tolerance would be measured against
    whatever the pre-march reached and compound a silent tightening per refresh — this is what the old
    `rtol/refresh_rtol` compensation approximated, and why the `refresh_rtol <= rtol` constraint existed;
    both are now gone). The absolute form is available precisely *because* the refresh path is
    forward-only, so `‖R0‖` is concrete rather than traced. The constrained path
    (`mass_flow_coupled_continuation` / `solve_coupled_mass_flow`) has **no** staged refresh — thread
    `reuse` through if that driver is added.
    - **The trigger is forward-only — it *raises* under `jax.grad`, and must (binding).** The refresh
      re-derives the preconditioner from the **mid-march** state, which is a tracer when differentiating;
      the refreshed preconditioner would capture it and escape the converged solve's `custom_vjp` as an
      `UnexpectedTracerError` (the general "build the preconditioner from concrete params *outside*
      `jax.grad`" footgun — [[precond-outside-grad]]). A refresh also forbids an explicit `continuation`,
      so there is **no** concrete-preconditioner path through it — hence the honest behaviour is a clear
      up-front `ValueError`, not a leak. `solve_coupled` guards this with `_is_traced((coupled, flow, k,
      omega))` (reliable because the solve is eager-only — the scalar AMGs are off-jit scipy, so a tracer
      leaf can only mean a wrapping transform). To differentiate, drop `refresh_trigger` and take the
      gradient of the single-stage solve with a `continuation` built on concrete params outside
      `jax.grad`; the adjoint is refresh-independent (the preconditioner is `stop_gradient`-ed, both
      marches reach the same converged state, so the IFT adjoint is identical), so nothing is lost. This
      is why the refresh's gradient property is covered by *forward* tests (`same fixed point`) plus the
      existing single-stage adjoint gate, and by a fast unit test that the guard fires — **not** by an
      adjoint test through the staged solve (there is none: that path cannot be differentiated).
  - **`on_step` / `on_checkpoint` instrument the march, and work WITHOUT a refresh trigger.** `on_step`
    receives each `StepReport` (step, cycles, ‖R‖, ratio); `on_checkpoint` additionally receives the
    *solved-variable* state (map with `physical_fields`). Both are the seam a solver study logs a long
    march through — needed because a multi-hour coupled march that prints nothing cannot be told from a
    hung one, and this case's documented failure mode is a march that keeps stepping while the residual
    creeps *upward*. Only the observed segments call back; the finishing solve is traced. See the
    `march.py` bullets in `.claude/rules/solve.md` for why observation is not gated on the trigger and why
    the state rides a separate seam from the report history.
  - **`solve_coupled(step_control=…)` — opt-in α-targeting β control, composes with `refresh_trigger`
    (experimental).** A `StepControl` (currently `AlphaTargetingControl`) reshapes the shift strength β
    each step toward the line-search-factor α=1 boundary, the measured efficiency optimum SER misses.
    Measured to strictly beat SER on pitzDaily (~2.6× to a given residual, reaching deeper) **when paired
    with the AMG refresh** — the two are co-designed, not independent (a bolder β stales the frozen PC
    faster). It is forward-only (raises under `jax.grad`, same guard as the refresh) and does **not**
    converge standalone (stalls rel ~0.03), so it is opt-in and never a default. Full analysis: the "SER β
    schedule runs backwards" bullet in `.claude/rules/solve.md`.
  - **`reuse=` refreshes a stale k/ω preconditioner without changing the compilation signature.**
    `scalar_transport_preconditioner(..., reuse=old)` (threaded through
    `SSTTurbulence.k_preconditioner` / `omega_preconditioner`) re-derives the *values* at a new state on
    the reused hierarchy's **frozen coarsening**. This is what makes a mid-march refresh affordable, and
    the measured reason to want one: on a separated pitzDaily state, refreshing the **scalar** AMGs is
    worth ~2.4× in outer GMRES cycles (30 → 13 at β=2 with the production lAIR scalars; refreshing the
    *flow* block is worth nothing, 30 → 29). It matters **only for `method="air"`** — lAIR's C/F split
    reads operator values, so a plain rebuild changes every shape below the first level or two and would
    force a recompile of the solve it accelerates (`reuse` routes to
    `~aquaflux.solve.refresh_air_hierarchy`). For `method="twolevel"` the aggregation reads only the
    graph, so a rebuild is already structure-preserving and `reuse` is accepted but changes nothing.
    A `ScaledScalarPreconditioner` wrapper is unwrapped (the log chain-rule scale is re-derived at the
    new state by the caller), and reusing across *different* methods raises. Pinned in
    `tests/unit/test_scalar_transport_preconditioner.py`: the lAIR refresh preserves shapes **where a
    rebuild provably does not**, the twolevel path is structure-preserving either way, and a refreshed
    preconditioner **beats the stale one on the developed operator** (so the reused split is a real
    trade, not a no-op).
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
      smooth and O(geometry), so unlike ω it reconstructs well. `closure_fields` overwrites only the
      wall-adjacent rows. Safe because ω needs **no wall-normal flux** there (the row is a value
      fixation, the wall closure is zero-gradient), so the only consumers are inward.
    - **MEASURED INERT ON pitzDaily — the fix is right, its effect here is nil (2026-07-25).** After the
      fix the imposed gradient at the fixed cells is **5.85×** the reconstruction (and off-wall cells are
      untouched, max diff exactly `0`), yet the coupled residual at the clean OpenFOAM field is
      **bit-identical** in every block. The chain is closed: the ω rows at those cells are the *fixation*,
      so their cross-diffusion source is discarded; the diffusion `corr` path is 0.03 %; and the one live
      route — `F1` → blended constants → the (unfixed) k rows — is **saturated**: `F1 = 1.000000` at
      *every* fixed cell (min = median = 1.0, 100 % above 0.999), so a 3.6–5.9× error in `CD_kω` cannot
      move it by one float. **Do not cite this fix as a convergence or accuracy improvement on this
      case.** It is a latent-trap removal: it would bite immediately under a gradient-using ω advection
      scheme, a blend or regime where `F1` does not saturate, or a mesh where the non-orthogonal
      correction is not negligible. Corollary: the first interior ring's residual (91 % of the interior)
      is **still unexplained** — with both the gradient and the non-orthogonal correction now ruled out,
      the leading candidate is the ω advection scheme (first-order upwind here vs OpenFOAM's
      second-order `limitedLinear`).
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
  - **⚠️ THE ω SHIFT DIAGONAL INHERITS ω's DYNAMIC RANGE — the root of the "never rebuild the shift"
    rule, and (measured) a ~11–37× step-productivity penalty (2026-07-25, #33).** The coupled shift for
    the log-solved ω block is `d_ω = transport_diagonal × jacobian_scale`, and under `LogScalars` that
    scale **is ω**. That is the *correct* linearization of a pseudo-time term on ω (`V/Δt (ω^{n+1}−ω^n)`
    → `V/Δt · ω · δw`), but it makes the damping proportional to a field spanning orders of magnitude.
    - **The pathology is a tail, not a level.** Against the cold-IC diagonal, the ω block's ratio at a
      developed state is median 0.87 / p99 **14.4** / max **24**, with **15 % of cells above 2×** —
      while velocity and k have **0 %** above 2× and max < 2. The tail is present within **20 steps**
      (8 % already at step 20), so there is no safe refresh cadence. A *median* comparison reads 0.96
      and shows nothing; this is why the effect was mis-diagnosed twice.
    - **Any diagonal carrying that tail destroys the coupled Newton step**, with the linear solve still
      converging (`lin_rel` 1e-8…1e-10, unchanged cycle count) — so it is not preconditioner mismatch.
      At the march's own β, carried gives α = 1.0 and +0.677 %; diagonals built at *any* later state
      (g0020/g0060/g0100/g0110, all genuine, self-consistent) give α at the ladder floor and **ascent**.
      Confirmed β-independent (same collapse at β = 0.5).
    - **⚠️ A PROPOSED CURE THAT FAILS END-TO-END — dropping the `× ω` factor (marching `w = log ω` in
      pseudo-time). Single-step numbers looked transformative; the march refutes them. Do not retry
      without reading why.** The single-step measurements below are real and reproducible, but they were
      taken at one developed state and judged on ‖R‖, and on this case ‖R‖ has repeatedly failed to
      track the physics. Measured, same state, only the ω block changed:

      | ω shift form | source | β_ω | cyc | α | reduction |
      |---|---|---|---|---|---|
      | shipped (`× ω`) | cold (carried) | march's | 14 | 1.0 | **+0.68 %** |
      | log-space | cold (carried) | matched median | 14 | 1.0 | **+7.75 %** |
      | log-space | cold (carried) | ¼ matched | 15 | 1.0 | **+25.01 %** |
      | log-space | **g0110 (REFRESHED)** | ¼ matched | 16 | 1.0 | **+25.12 %** |
      | log-space | **g0110 (REFRESHED)** | matched median | 15 | 1.0 | **+9.31 %** |

      Three things at once, all real: the tail collapses (>2× from 15 % → **0.17 %**, p99 14.4 →
      **1.73**); at *matched median damping* the step is **11× better**; and it is **refresh-invariant**
      — refreshed matches carried to 0.1 pp, where the shipped form collapses.
    - **AND YET IT LOSES BADLY ON A MARCH.** Cold-IC march, everything held identical to the control
      (same IC, drift trigger, refresh limit, solver) with only the ω shift form changed:

      | step | 10 | 15 | 20 | 25 |
      |---|---|---|---|---|
      | control `x_r/h` | 0.09 | 0.32 | 0.39 | ~0.45 |
      | log-space `x_r/h` | 0.03 | 0.05 | 0.09 | 0.14 |
      | control rel | 1.28e-1 | 9.50e-2 | 4.45e-2 | 3.0e-2 |
      | log-space rel | 3.24e-2 | 3.01e-2 | 2.10e-2 | 1.23e-2 |

      **A far deeper residual with a 3–6× smaller recirculation** — at rel ≈ 0.030 the control has
      `x_r/h` ≈ 0.6 against the log-space arm's 0.05, an order of magnitude worse bubble at equal
      residual. The single-step ‖R‖ gain was largely a *norm* effect, which is exactly what the caveat
      about ‖R‖ on this case warned of. It also clipped hard on the very first step (α = 0.031).
    - **WHY — and this reverses the reading of the `× ω` factor (measured at the cold IC).** Spatial
      spread *within* a state is a different quantity from the temporal tail, and conflating them is
      what produced the wrong proposal. ω spans 440→1.14e5 even at the cold start, and:

      | | near-wall decile / median | bulk decile / median | max/median |
      |---|---|---|---|
      | shipped (`× ω`) | **2.95** | 0.82 | 30.7 |
      | log-space | **0.54** | 0.91 | 3.6 |

      The bare transport diagonal is *smaller* near the wall (0.54× median) — it **under-weights the
      stiffest cells** — and multiplying by ω corrects that to 2.95×. So `× ω` is doing real work:
      dropping it inverts the wall/bulk damping ratio and under-damps the near-wall region by ~20×
      once the global factor is included, which is what clips α on step 0 and starves the bubble after.
    - **The diagnosis: `× ω` entangles two things that should be separated.** *Spatially* it is correct
      (it supplies the near-wall weighting the bare transport diagonal lacks); *temporally* it is the
      problem (it drags ω's evolving range into the shift, creating the tail). Carrying the cold-IC
      diagonal keeps the good half and freezes the bad half — but only by freezing **both**.
    - **✅ THE SHIFT *CAN* BE REFRESHED — `transport_diagonal(state) × ω(cold)` — BUILT (issue #156).**
      `CoupledShiftPolicy` now stores the two factors separately (`k_shift_transport`/`k_jacobian_scale`,
      likewise ω) instead of the product, and `_coupled_shift_policy(reuse=…)` rebuilds the transport
      diagonal at the new state while carrying the coordinate factor frozen — so the "never rebuild the
      shift" rule is replaced by "rebuild the transport, carry the coordinate factor." Refresh the
      *physics* (the transport diagonal, a local time scale that genuinely should track the developing
      flow) and freeze only the *coordinate transformation* (the ω weighting, a property of the log
      parametrization, not of the flow). The temporal ratio is then `transport(state)/transport(cold)` —
      the ω factor cancels exactly:

      | build state | temporal tail >2× (shipped → variant) | near-wall weighting (shipped → variant) |
      |---|---|---|
      | g0020 | 8.01 % → **0.00 %** | 2.92 → **2.83** |
      | g0060 | 12.06 % → **0.06 %** | 3.43 → **2.79** |
      | g0110 | 14.98 % → **0.10 %** | 3.37 → **2.70** |

      Both properties at once: the tail vanishes (p99 1.69 vs 14.4 — the same class as velocity and k,
      which have never needed carrying) *and* the near-wall weighting is preserved (~2.7–2.8 against the
      shipped 2.9–3.4; the failed log-space form inverted it to 0.54).
    - **Confirmed on a march that upgrades its shift at every refresh.** Identical to the control up to
      the first refresh (bit-identical through step 15, as it must be — `transport(cold) × ω(cold)` *is*
      the shipped diagonal), then diverging. Post-upgrade steps 16–20 ran at **α = 1.0000 throughout**,
      9–13 cycles, residual falling steadily — where a shipped-form rebuild collapses α to the ladder
      floor with an ascent direction. At step 20: rel 3.69e-2 vs the control's 4.45e-2, α 1.0 vs 0.5,
      `x_r/h` 0.39 in both. By steps 25–30 the two are level (rel 2.33e-2 vs 2.23e-2; `x_r/h` 0.61 in
      both).
    - **So: structurally sound, performance-neutral on this case.** The march neither gains nor loses,
      because the control *also* refreshes its AMGs and its cycle counts were already healthy (10–16) —
      a stale shift was not costing it much here. The value is that a constraint which should not exist
      is removed: a stabilizer whose correctness depends on being frozen at one smooth initial state
      would fail on a case whose cold start is rougher, or which needs far more development. Expect the
      benefit to appear there, not on pitzDaily. **Do not sell this as a speed-up.**
    - **Not a free lunch either way:** over-damping the ω block destroys the direction by *any* route —
      a uniform 4× on the log-space form collapses exactly like the tail does (α floor, ascent, 34
      cycles). The unifying statement is **the coupled direction fails when the ω block is over-damped**.
    - **Untested variant, if this is revisited:** only `factor = 0.25` was marched, chosen because it
      maximized single-step ‖R‖ reduction — i.e. selected on the metric we do not trust. `factor = 1.0`
      (matched median damping) still gave +9.3 %/step and would pace ω much closer to the shipped form.
      That is the fair second attempt; it does **not** fix the inverted wall/bulk ratio, so expect it to
      help but not to win.
  - **OPEN DEFECT: the reconstructed ω *gradient* in the fixed cells is ~4× too small (measured
    2026-07-25; fix tracked, not yet built).** We impose a value on those cells but let their gradient
    be *inferred from neighbours* as if they were ordinary unknowns. Measured against the analytical
    gradient of the field we impose (`∇ω_wall = dω_wall/dd · ∇d`): **reconstructed/exact = 0.256**
    (p5 0.205, p95 0.374) at the fixed cells, and **2.236** (p5 0.845, p95 5.037) at the first interior
    ring — so the error does not stay local. Two causes, both structural: `ω_wall ∝ 1/d²` is strongly
    convex while Green–Gauss is a *linear* fit across cells whose `d` spans 8.5e-5→5.6e-4; and the
    reconstruction folds in the **wall face**, whose ω comes from the `ZeroGradient` closure (face value
    = cell value) although the true profile diverges there.
    - **A gradient-scheme A/B CANNOT detect this** — every scheme treats the fixed cells as ordinary
      cells, so all are wrong identically and the difference cancels. Measured: `CorrectedGreenGauss` vs
      `CompactGreenGauss` give **bit-identical** ratios, and their residual difference is 0.03 % of
      ‖R_ω‖. Do not re-run that comparison expecting an answer; compare against the *analytical*
      gradient instead.
    - **Consumers.** Face interpolation and the diffusion `corr` on faces to interior neighbours
      (measured negligible here, 0.03 %), and — the one with teeth — `∇k·∇ω` in `OmegaCrossDiffusion`
      and the `F1` blend, which set the blended constants for **both** scalar equations. The k rows at
      those cells are **not** fixed, so a wrong gradient corrupts a genuinely solved equation: measured
      `CD_kω` there is **3.6× too small** (median exact/reconstructed 3.569, p5 2.98, p95 6.34). Whether
      that moves `F1 = tanh(arg₁⁴)` is **not yet measured** — it saturates near a wall by design and may
      absorb the error, so treat this as a correctness/consistency defect until shown otherwise.
    - **Fix (agreed direction):** impose the gradient alongside the value, from the closed form
      `dω_wall/dd · ∇d` — differentiating the smooth **power-mean** blend (not the bare viscous branch,
      which kinks where the branches cross), with `∇d` a well-posed reconstruction of a smooth O(1)
      field. Safe to overwrite because ω needs **no wall-ward flux** at those cells (the row is fixed and
      the wall closure is zero-gradient), so the only consumers are inward. Note the distinction when
      building it: the *point* gradient (right for the cell-centred sources) is **not** the best *linear
      reconstruction slope* over a finite cell for a convex profile — for ω the latter barely arises,
      since advection is first-order upwind and the flux-continuous diffusion eliminates the face value.
  - **THE "FIRST RING" IS NOT SPECIAL — the OF-vs-aquaflux ω imbalance is UNIFORM ~13 % across the
    domain (term decomposition, 2026-07-25). This retires a whole line of investigation.** At the clean
    pimpleFoam field, 91 % of the interior ω residual sits in the 471 cells adjacent to the wall-fixed
    band, which reads as a near-wall defect. Decomposing the residual term by term shows it is not:

    | | first ring | bulk |
    |---|---|---|
    | advection / diffusion | 27.8 / 49.9 | 18.1 / 17.5 |
    | production / destruction | 119 / 169 | 44.9 / 64.1 |
    | cross-diffusion | 24.1 | 14.3 |
    | **residual** | **20.5** | **9.33** |
    | **residual ÷ largest term** | **0.121** | **0.146** |

    The *relative* imbalance is the same everywhere — the bulk is if anything slightly worse. The ring
    dominates the absolute residual only because its terms are 2–3× larger there (destruction 169 vs 64).
    **So there is no localized near-wall defect**, and the "91 % in the first ring" framing was an
    artifact of reading absolute magnitudes in the stiffest region — the same error as the wall-row
    scaling defect, one level down. It is also **not** a near-cancellation of stiff terms (that would be
    ~1e-3, not 0.12).
    - **What it is:** a **global** discretization/model difference — OF's converged field leaves a ~13 %
      relative imbalance in *our* discrete ω equation everywhere. That is a two-codes statement, not a
      defect in ours, and **not a convergence blocker**.
    - **Four local candidates were eliminated first, all by measurement — do not re-open them without new
      evidence:** the non-orthogonal correction (0.03 % of ‖R_ω‖); the fixed-cell ω gradient (genuinely
      4–6× wrong, fixed, and measured **inert** because `F1` saturates at 1.0 there); the wall-blend
      exponent (removed by the max blend, wall rows 4.91 → 1.33); and the ω **advection scheme**, which
      makes the ring *worse* (18.4 → 19.9) while improving the bulk.
    - **Actionable follow-up (accuracy, not convergence):** second-order scalar advection cuts the *bulk*
      ω residual **15 %** (8.26 → 6.98), consistent with OpenFOAM using `Gauss limitedLinear 1` for both
      k and ω while we use `FirstOrderUpwind`. The original reason for first-order — ω driven negative by
      a second-order Newton update — was explicitly conditioned on log-variable transport not existing.
      **It exists now (`LogScalars`)**, so that choice is worth re-testing rather than inherited.
  - **The blend SHAPE is a power-mean choice, and OpenFOAM / Fluent pick DIFFERENT exponents — this is
    the source of the near-wall ω disagreement, and it is a modelling choice, not a bug (measured
    2026-07-24 against a *clean* reference; supersedes the corrupt-reference wall-ω numbers above).** All
    three codes use the *same* two branches `omega_vis`, `omega_log`; they differ only in how they combine
    them, which is now the **implemented** power-mean family (`SSTModel.wall_omega_exponent = p`,
    `wall_omega_viscous_coeff = C`, `omega_vis = C·6ν/(β₁d²)`):
    - **aquaflux default: `p = 2`, `C = 1`** — `sqrt(omega_vis² + omega_log²)` (Menter's quadrature). The
      default is **unchanged** by the parametrization (all existing wall tests pin it).
    - **OpenFOAM `omegaWallFunction` (default): `p → ∞`, `C = 1`** — `max(omega_vis, omega_log)`; reached in
      aquaflux with a large exponent (`p ≈ 60` is `max` to <2 %). On pitzDaily wall cells `max` matches OF's
      field to **<2 %** (median ratio 1.00); aquaflux's `p = 2` runs **~20 % high in the buffer layer**
      (`y+≈8–15`, `sqrt(a²+b²)` exceeds `max(a,b)` by up to 41 %; median aquaflux/OF `omega_wall` = 1.20).
      The wall distance and constants **agree** — only the exponent differs.
    - **Ansys Fluent (`correlation` default, Theory Guide §4.18.3, eqs 4.404–4.407): `p = C_exp = 1.3`,
      `C = C_calib = 1/3`** — both fit on plane Couette flow (Re 1e6) to flatten the wall shear across `y+`
      (Fluent also blends `u*`, `u_τ`, and the k-production consistently, and offers a `tabulated` option).

    So there is **no single "the" near-wall ω model**: each code picks an exponent, and Fluent recalibrates
    the coefficient. Whether to change the aquaflux *default* (match OF `max`, keep Menter `sqrt`, or adopt
    the Couette-calibrated Fluent blend) is still an open model decision; the *mechanism* to select any of
    them is now shipped.
  - **The max blend makes aquaflux accept the clean pimpleFoam field as an on-root IC (measured
    2026-07-24, `pimple_ic_blend`).** Feeding the converged pimpleFoam field (`of_transient/0.14`, the
    clean reference — *not* the corrupt steady run) into the coupled residual: the **flow, k, and interior-ω
    blocks are already ~0** (`|R_flow|≈6e-3`, `|R_k|≈1e-2`, interior `|R_ω|≈20` over 11.8k cells) for every
    blend — the bulk field is accepted; the *entire* ω-block residual lives in the **472 wall-fixation
    cells**. There the scale-free `|R_ω|/ω` per wall cell is **median 0.20 under the default `p=2`** (exactly
    the ~20 % blend bias) but **median 7e-5 under the `max` blend** (`p=60`) — the imposed near-wall ω then
    matches OF cell-for-cell, ~3500× smaller. Fluent's `p=1.3, C=1/3` gives median 0.13 — *worse*, because
    `C=1/3` is calibrated to Fluent's own treatment, not OF's, confirming **`max` (not Fluent's constants) is
    what matches OF**. A residual tail (p95 ≈ 0.19) survives the `max` blend only at the highest-ω cells (step
    lip `x≈0`, upper-wall separation `x≈0.12–0.17`, ω ~ 3e4–1e5), where the transient pimpleFoam field is
    itself not deeply converged; the absolute `|R_ω|` L2 (~4e4) is large only because those few ω~1e5 cells
    dominate the norm. Bottom line: the ~20 % near-wall disagreement was **entirely** the blend exponent, and
    it is removed by the `max` blend — the interior model was never in question.
  - **pitzDaily validation status (2026-07-24, binding for whoever re-runs it — read before trusting any
    OF-vs-aquaflux number).** The shipped OpenFOAM *steady* reference (`validation/pitzdaily_openfoam/runs/kwsst/`,
    `foamRun` with `ddtSchemes: steadyState` = SIMPLE) is **CORRUPT**: its ω field *checkerboards* in the
    inlet channel (adjacent cells oscillate ω ≈ 0.2 ↔ 1e8 — a non-converged `omegaWallFunction` limit-cycle;
    the steady solver's residuals swing ~500× and never settle). **Do not compare aquaflux against it** —
    aquaflux's residual on that field is ~4e8, which is aquaflux *correctly rejecting a non-physical field*,
    not a bug (verified by reading the raw OF ω). A **stable steady root DOES exist**: a time-accurate
    `pimpleFoam` transient (Euler ddt, PIMPLE, CFL≈0.9) started from that field *relaxes* (velocity residuals
    decay ~50×) and holds reattachment `x_r/h = 7.74`; its ω is clean (`[160, 1.1e5]`, no checkerboard). **Use
    that transient-converged field as the reference.** aquaflux's turbulence under-prediction (`x_r/h` 1.16
    vs 7.74, `k` 1.6 vs 5.0, `ν_t/ν` 85 vs 422 at rel 0.052) is **UNDER-CONVERGENCE, not a model bug** —
    verified three ways: (1) the closure reproduces OF's `ν_t = 422` *exactly* when fed OF's own converged
    `k`,`ω`; (2) the "flat ν_t ≈ 85" is the **inlet** value `k_in/ω_in`, not a cap (interior ν_t is *below*
    inlet — under-developed); (3) `x_r`,`k` climb *monotonically* toward OF as the march progresses, stalling
    only at rel ~0.05 (the SER-schedule convergence problem — see `.claude/rules/solve.md`). On the clean field
    aquaflux accepts the **bulk** to `|R|/ω ~2e-6`; the only residual is the near-wall fixed-cell blend
    difference above. (`compare.py` was also silently broken — it called the renamed `momentum.velocity_gradient`;
    fixed to `turbulence.closure_fields(...).nu_t`, so the cell-for-cell profile comparison had *never actually
    run* until this session.)
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
    differentiability); the flow-side seam is in `.claude/rules/flow.md` / `.claude/rules/discretization.md`.
  - **The near-wall `k` budget is closed by FOUR pieces that only work together (binding — measured on the
    periodic channel; do not remove or reorder one in isolation).** Wiring `omega_wall` + `nut_wall` alone
    left the wall-function channel predicting **−25%** of the wall-resolved `u_τ`. The full set brings the
    same mesh to **−2.0%** with the wall-resolved mesh a no-op (`u_τ` unchanged to all printed digits) and
    the segregated loop *converging* where it previously hit `max_sweeps`. All four cross the sublayer/log
    boundary on the **one** smooth weight `wall_function_weight(nu,d,k,model) = tanh((y*/y*_lam)⁴)`
    (never a `y*` switch — an AD-Newton residual cannot converge through a jump; see the docstring), and
    all live in `boundary.py` as pure functions with a `NearWallKClosure` collaborator in `sources.py`
    holding the per-wall-cell data (`cells`/`distance`/`viscosity`/`shear_rate`) that always travels
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
    **Known residual — the buffer layer.** A wall-function mesh landing at `y+ ≈ 11–16` (the crossover
    itself) is the worst case for any wall function, and the blend smooths it without making it exact:
    measured `u_τ` error on the channel is **−0.6% at y+≈68, −2.0% at y+≈33, −6.9% at y+≈16, −5.0% at
    y+≈11**, versus 0 on the wall-resolved mesh. Place the first cell either inside the sublayer or out
    past `y+ ~ 30`; the buffer-layer dip is a model limitation, not a bug to chase.
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
    (the eager path dispatched `velocity_fields` / `mass_flux` / `closure_fields` one op at a time —
    ~130 ms/sweep of avoidable overhead at 1600 cells). `_sweep_closure` calls
    `momentum.flow_fields(flow)` **once** for both the velocity gradient the closure reads and the
    Rhie–Chow `mdot` the scalars advect on (the pre-solve μ_t uses the lightweight `velocity_fields`,
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
  - **The reparametrized block's preconditioner/shift are chain-rule-scaled at the reference (binding).**
    The physics Jacobian w.r.t. `w` picks up `d(φ)/d(w) = jacobian_scale(φ)` (`= φ` for log). `coupled_continuation`
    recovers the physical reference via `physical_fields`, scales each scalar shift diagonal by that factor,
    and wraps its (physical-operator) AMG in `ScaledScalarPreconditioner` by the reciprocal — so the frozen
    preconditioner acts on the reparametrized block without rebuilding the hierarchy. `_reparametrized_preconditioner`
    returns the preconditioner **unchanged** when the factor is one, so the `DirectScalars` path is bit-identical.
  - **`coupled_continuation` globalizes with a line search + a larger-restart Krylov (the pitzDaily
    performance fix).** Two measured facts drove this. **(1) The full coupled Newton step
    from the hybrid IC overshoots by ~10⁷×** (‖R‖ 220 → 5.8e9); the pseudo-transient step's only recourse
    used to be escalating β — a *full re-solve* — and escalating β (16/64) still did **not** descend
    (rel ≈ 1.0 → the full-mesh march *stalled*, which had been misread as "slow, compute-heavy"). A
    **backtracking line search** on the one β₀ solve finds α≈¼ → rel≈0.48 (residual halved), so
    `coupled_continuation` sets `line_search=_COUPLED_LINE_SEARCH` (see `.claude/rules/solve.md`); β
    escalation stays the fallback for a bad *direction*, not an overshoot. With it the full-mesh solve
    **descends** (rel 1.0 → 0.48 → 0.44 → 0.31 → 0.20 → ~0.18 over ~6 steps) instead of *stalling at
    rel 1.0* — the case is now solvable at all, a correctness fix, not just speed. **(2) The shifted
    solve needs a large Krylov subspace:** `_COUPLED_FORWARD_SOLVER` is restart-120 GMRES (the shared
    restart-40 default discards too much Arnoldi history on this stiff saddle system; ~1.4× faster to
    the same tight solution). Tolerances stay **tight** — an inexact solve is unsafe under log-`ω` (an
    inaccurate log step is exponentiated and diverges), so loosening the linear tolerance is **not** a
    lever here (measured: it breaks the march).
  - **The march's residual measure is the plain Euclidean ‖R‖ by default; the block-scaled per-field
    measure is opt-in (`block_scaled_norm=True`).** A `BlockScaledNorm` over `[flow, k, ω]` (each block
    divided by its own initial magnitude, `_coupled_residual_norm`) was built so the globalization weighs
    every field rather than the `ω` block that dominates ‖R‖ (`ω` O(1e5), `k` O(1e-3)) — the concern being
    that a step collapsing `k` barely moves the `ω`-dominated ‖R‖ and is accepted. But **measured, it
    *stalls* the pitzDaily march**: the per-block relative norm plateaus long before the fields converge,
    so `coupled_continuation`/`mass_flow_coupled_continuation` default to `jnp.linalg.norm` and expose
    `block_scaled_norm` (default `False`) to request the block measure for experimentation. The helper and
    the `BlockScaledNorm` class are kept as that opt-in path, not deleted. **When a march refreshes, the
    measure is held fixed at the initial state** — `solve_coupled` passes `coupled_continuation(residual_norm=
    base_norm)` on every refresh rather than letting it rebuild `_coupled_residual_norm` at the developed
    state, or the self-normalising block scales would re-base and the convergence test become unreachable
    (#156 seam 4; see `.claude/rules/solve.md`).
  - **`beta_floor` (SER lower bound) is available but off by default (a measured wash).** Bounding
    `β = max(beta_floor, β₀(‖R‖/‖R₀‖)^p)` keeps each late shifted solve out of the ill-conditioned low-`β`
    regime (correctness-safe — the floor scales the correction `δ`, which vanishes at the root, so it never
    moves the converged state). But end-to-end it is a **net wash** (cheaper late solves cancel the extra
    Newton steps), so it defaults to `0`; wired through `coupled_continuation` for further evaluation. The
    settled coupled-solve cost is the diagonal-block-preconditioner weakness at high Reynolds number, **not**
    the residual measure, `β` floor, or missing cross-coupling (a block-triangular preconditioner was worse
    — non-convergent on recirculating pitzDaily). See `.claude/rules/solve.md`.
  - **The coupled flow block uses the convection-aware AMG + MSIMPLER Schur, not the smoothed/SIMPLE
    default (`_coupled_shift_policy`).** A RANS case is high-Reynolds, and the default
    `BlockPreconditioner.build` config (viscous-**smoothed** velocity AMG, which is Peclet-blind, + the
    **SIMPLE** `a_P` Schur, which degrades with convection) produces a poor momentum-block direction once
    the flow separates. Measured on the developed pitzDaily field (shifted Newton direction vs the true
    one): smoothed+SIMPLE gives **cos 0.40** and the march stalls at rel ~0.18; **`velocity="convection"`
    + `schur_scaling="msimpler"` gives cos 0.998** *and* cuts the shifted solve from ~120–580 GMRES
    cycles to ~17 (each march step ~8× cheaper). Both stay valid **frozen at the cold initial state**
    (MSIMPLER's Schur is velocity-independent; the convection linearization is Peclet-robust), so **the
    FLOW block needs no reference refresh** — verified two ways: IC-frozen cos 0.996 vs plateau-rebuilt
    0.998, and refreshing the flow block alone at a separated pitzDaily state is if anything slightly
    *worse* (31 → 34 outer cycles). It is **not** the flow↔turbulence cross-coupling (the block-*diagonal*
    preconditioner with the right config already reaches cos 0.998 — a block-triangular coupling was
    built, measured, and is worse; see `.claude/rules/solve.md`). **The k/ω *scalar* AMGs are the
    exception: they do go stale, and refreshing them alone once the flow separates is worth ~2.6× in
    outer cycles** (31 → 12) — the one staleness lever that pays; see the staleness bullet in
    `.claude/rules/solve.md`. Overridable via `preconditioner_kwargs`.
  - **~~Remaining limiter — the k equation drift~~ — RETIRED: the stall no longer reproduces (measured
    2026-07-22).** This bullet used to record that past rel ~0.09 the direct-`k` residual grew (rel 1 →
    ~5×) and re-stalled the march, and named high-Reynolds `k` stability as the open follow-up. **It
    does not happen on the current code.** Re-measured on the full ~12k-cell pitzDaily from the cold
    hybrid IC, second-order (Venkatakrishnan-limited) momentum, log-`ω`, conv+MSIMPLER: the march
    descends **monotonically straight through 0.09** with no plateau —
    rel 9.4e-2 (step 14) → 6.6e-2 (18) → 4.7e-2 (21) → 3.5e-2 (24) → 2.7e-2 (26) —
    while the recirculation keeps growing (110 → 416 cells). **`k` does not diverge**: its peak *falls*
    (13.1 → 11.2) once separation establishes. The fixes that landed after the original observation —
    the inlet-driven `k` floor and the near-wall `ω` profile in the hybrid IC (#139), the Bernoulli
    pressure seed, and the backtracking line search — appear to have removed it. Do **not** reopen `k`
    stability, nonlinear elimination of the k–ω closure, or Reynolds-number continuation on the strength
    of the old claim; re-measure first.
  - **What is actually left on this case is COST — and the DOMINANT part is the SER β schedule
    under-damping, not the preconditioner (measured, corrected-IC run).** An instrumented full march
    (E1: `solve_coupled` to `rtol=1e-6`, cold hybrid IC, per-step logged) does **not** converge — it
    *decelerates* (per-step residual decay 0.918 → 0.971, step efficiency down 19×) instead of entering
    the quadratic basin. The cause is the globalization: SER lowers β as ‖R‖ falls, but the
    efficiency-optimal β *rises* (≈2 at rel 0.38, ≥5 at rel 0.05), so in the tail SER runs at β ~50× too
    low, where the full Newton step overshoots ~33× and the line search claws back ~0.4%/step (diagnosed
    directly via the step-length factor α — the full analysis and data are the "SER β schedule runs
    backwards" bullet in `.claude/rules/solve.md`). **A ~1.9× preconditioner refresh cannot rescue a march
    the schedule is grinding to a halt** — this reorders the priorities: fixing the β schedule (an
    α-targeting PTC step control) is ahead of calibrating the refresh.
  - **Preconditioner staleness is the SECONDARY cost, and it is coupled to the β schedule (measured).**
    Over the march the wall time per step also grows (~7×: 27 s @ step 8 → 197 s @ step 26 on the older
    run) as the recirculation develops and the frozen scalar preconditioner degrades — the same
    post-separation regime where refreshing the k/ω AMGs is worth ~2.4–2.6× in outer cycles (staleness
    bullet in `.claude/rules/solve.md`). Driving a refresh **from the march** is BUILT:
    `solve_coupled(refresh_trigger=CoefficientDriftTrigger(…))`. **The β coupling that motivated it:** a
    bolder β moves the state faster and stales the IC-frozen PC faster, so a *cost*-based trigger is
    confounded (cycles rise from β→0 **and** staleness — #19). The β-independent staleness trigger keyed
    on `‖Δν_t‖` is now **BUILT** and is the default recommendation; a `‖Δṁ‖` measure would be a second
    `drift_measure` against the same trigger, needing no new trigger. The threshold is **calibrated on an
    instrumented cold-IC pitzDaily march** (`threshold = 0.1`, firing where the cycle count has just
    doubled off its floor and the bubble has formed) — the table, and the validation that drift really
    does track cost, are in `.claude/rules/solve.md`. One geometry, so re-calibrate by offline replay
    on a new case rather than assuming it transfers.
  - **The slope limiter is NOT implicated (measured — do not re-derive this).** pitzDaily is the first
    case that genuinely exercises `LimitedUpwind` (Poiseuille / cavity / smooth channels never activate a
    limiter), so it was the natural suspect for the second-order march being slower than first-order.
    Measured over an identical 14-step march: first-order rel 6.96e-2, limited `K=5` 9.39e-2, limited
    `K=100` 9.63e-2, **unlimited (`limiter=None`, ψ≡1) 9.71e-2**. Removing the limiter entirely
    reproduces the limited result, so the first-vs-second-order difference is inherent to the
    *reconstruction*, not to limiting. (Two genuine limiter defects were found in that audit and filed —
    a periodic-image inconsistency, and a dimensionally inconsistent `eps²` softening — but neither
    causes this, and neither should be pursued as a convergence fix.)
  - **The per-scalar transform is layout-consistent through both coupled solves (binding).** `solve_coupled`
    and `solve_coupled_mass_flow` both map the physical IC into the solved space with `state_from_physical`
    and return `physical_fields` — so `LogScalars` is correct through the mass-flow-constrained path too
    (identity for `DirectScalars`, which is all the mass-flow tests exercise). Do not reintroduce a bare
    `pack_state`/`layout.unpack` at a solve boundary: it packs physical values as if they were the solved
    unknown, silently wrong under any non-identity transform.
  - **`solve_coupled_mass_flow` — the coupled solve with the bulk velocity held by a Lagrange
    multiplier (#128).** A streamwise-periodic channel is driven to a target bulk velocity `U_bar`, so
    the body force `β` along the flow direction is itself a **coupled unknown** appended to the state
    and the coupled residual bordered with the constraint row `⟨U_dir⟩ − U_bar = 0`: one honest
    augmented residual `R_aug([flow…, k, ω, β]) = [R_coupled(state; β); ⟨U_dir⟩ − U_bar]`, driven by a
    single `ImplicitNewtonSolver`. The border column/row `(a, c)` and the Schur (constraint)
    preconditioner are the flow block's own primitives (`_constraint_vectors`,
    `_bordered_preconditioner`, `_with_body_force` from `flow/mean_velocity.py`) reused in the coupled
    `[flow…, k, ω]` layout by `_coupled_constraint_vectors` — the same Schur elimination one careful
    place keeps consistent, not re-derived. Globalized by `mass_flow_coupled_continuation`, which
    borders the **same** `_coupled_shift_policy` (extracted from `coupled_continuation` for exactly this
    reuse) with a `_MassFlowBorderedPolicy`: the shift diagonal gains a **zero** for `β` (the linear
    constraint row needs no pseudo-time damping) and the block preconditioner is wrapped by the
    constraint preconditioner. Because the constraint lives *inside* the coupled residual, the coupled
    IFT adjoint **carries it** — `jax.grad` through the converged constrained solve is the sensitivity
    of the turbulent field *at fixed bulk velocity* (FD-verified). This is the monolithic counterpart of
    the segregated bordered flow solve (`flow.bulk_velocity_flow_solve`): the segregated loop does **not**
    converge on this body-force channel, so the constrained fixed point is cross-validated by two
    independent AMG coarsenings (`air` ≡ `twolevel`, same `β`) — the periodic analogue of the inlet
    coupled-vs-segregated cross-check. Pinned by `test_coupled_mass_flow.py` (constraint met + turbulent
    + floors inactive; method-independence; the adjoint FD gate).
- **`initialization.py` — `hybrid_initialize` (cold-start, the reason `solve_coupled` self-starts).**
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
  actually reached `rtol` (the driver only `warnings.warn`s otherwise, and returns the
  under-converged fields), or that the result matches an independently converged reference. The
  model is `test_coupled_rans.py`, which drives the loop to `rtol=1e-9` and asserts it reaches the
  coupled solve's fixed point to 1e-4.

## Post-change
Keep this file's Status and Binding decisions true as the coupling globalization (issue #69) lands —
per the root `CLAUDE.md` Post-Change Checklist's Documentation-sync item.
