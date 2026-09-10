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

## How to read this file (read this before grepping it)

Same three rules as `.claude/rules/solve.md`: **every entry sits under a `##` section** (scan up to the
nearest one); **a superseded entry is DELETED, never struck through or annotated in place** (`~~tildes~~`
are invisible to `grep`, and "see above" expresses supersession by adjacency, which a hit does not
carry); and **a measurement without its configuration is unfalsifiable** — name the case, state,
bundle and shift, or the number cannot be re-adjudicated when a default moves.

**Check any symbol or default against the source before quoting it.** Stale entries here have been
lifted by `grep` and asserted as current. See `CLAUDE.md` → **Stale-Record Check**.

⚠️ **Two defaults that have moved and that older entries in this file were measured under:**
the AMG smoother fill (the validated `bfs3d` bundle is **ILU(0) × 4 sweeps**; the library still defaults
to ILU(1) × 2) and the aggregation (**plain**, not smoothed). Any AMG-adjacent number written before
those moves is un-adjudicable — treat it as a lead, not a fact.

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
- **⚠️ `S` IS WHAT MAKES THE VELOCITY COLUMNS OF THE COUPLED JACOBIAN REACH ONE RING FURTHER THAN
  EVERYTHING ELSE — a closure property with a direct solver cost (measured 2026-08-11).** The coloured
  probe that materializes the coupled Jacobian is charged per (colour, **column** field), and the colour
  count climbs steeply with the stencil reach: on `bfs3d` 11 colours at reach 1, 39 at 2, 94 at 3. The
  velocity columns need reach 3 while `p`, `k` and `ω` close inside reach 2, and the whole of that
  difference is `ν_t`'s dependence on `S`. Holding `S` constant (a `stop_gradient`, i.e. what a lagged-`ν_t`
  linearization would do) takes the velocity columns to **2**; every other arm — the flux limiter, the
  second-order reconstruction, a one-shot compact Green-Gauss gradient, and the velocity gradient inside
  the momentum diagonal's lagged flux — leaves them at **3**.
  **Why:** `ν_t = a1 k / max(a1 ω, S F2)` reads `S`, which is built from `∇u`, so `ν_t` is a
  *velocity-dependent coefficient* multiplying a flux that is already gradient-based, and the composition
  spends the extra ring. It equally explains the columns that do **not**: `k` and `ω` enter `ν_t`
  **pointwise** (no gradient), and pressure does not enter it at all.
  **Consequence to hold onto:** this is a real cost of differentiating the closure exactly, not a defect.
  A lagged `ν_t` would make the whole operator reach 2 and roughly halve the probe, but it is a
  quasi-Newton linearization — and the reach-2 *pattern* is separately measured to break the multigrid
  hierarchy, so it is not a free lever. Harness `validation/bfs3d_openfoam/column_reach_probe.py`; the
  probe-cost side is in `.claude/rules/solve-direct-preconditioners.md`.
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
  - **⚠️⚠️ THE `k` ROW'S JACOBIAN DIAGONAL IS NEGATIVE AT NEAR-WALL CELLS, AND ONLY THE PSEUDO-TIME
    SHIFT KEEPS IT POSITIVE — measured 2026-08-24 on pitzDaily.** `nu_t = a_1 k / max(a_1 omega, S F_2)`
    is proportional to `k`, so `P_k` is proportional to `k`, and subtracting a source that grows with
    the variable puts a **negative** term on that row's diagonal. Measured at a wall cell of the
    pitzDaily target rung: `J_kk` between **-1.22e-03 and -1.95e-03** against a shift `beta d_k` of
    **+2.31e-03** at the case's `beta_start = 0.5` — so the effective diagonal keeps only **16-47 %**
    of the shift, and is a difference of two similar numbers.
    - **The consequence is an amplifier on everything upstream.** Four gradient reconstructions at that
      one cell differ by **12 %** in `|grad u|` and in `P_k`, and by **30×** in the `k` correction the
      Newton step asks for, purely because a 12 % move in `J_kk` is a 3× move in the near-cancelling
      sum. The `k` correction then reaches **+4000×** the local `k`, in the increasing direction that
      `positive_block_limit` does not guard (it bounds only entries that could cross zero). The
      near-wall `omega` fixation row `log omega = log omega_wall(k)` hands that straight to `omega`,
      whose log transport exponentiates it — which is how a 12 % gradient difference loses a march.
      The full trail, with the four-way tables, is in `.claude/rules/schemes.md` under the
      `SkewCorrectedGradient` stall; `validation/pitzdaily_gradient_ab/closure_stall_probe.py`
      re-measures any of it at a saved state.
    - **The Patankar treatment this points at is NOT the flag below.** `explicit_production_limiter`
      freezes the **cap's** `k`; what drives the diagonal negative here is the **production's own**
      `k`-dependence through `nu_t`, which is differentiated whatever that flag says (the production is
      well under its cap at these cells — 83 against 315 — so the cap is not involved at all).
    - ⚠️ **Build it through `_shifted_solve`'s `jacobian_fn` seam, not with a `stop_gradient` in the
      residual.** The flag below is an adjoint hazard precisely because it changes what AD linearizes;
      `jacobian_fn` replaces only the **forward** Krylov operator, leaving the residual, the root and
      the IFT adjoint untouched.
    - **✅ BUILT AND TESTED AT ONE STATE, AND IT WORKS (2026-08-24).**
      `validation/pitzdaily_gradient_ab/closure_stall_probe.py::frozen_production_residual` evaluates
      the `k` equation's eddy viscosity at a `stop_gradient`-ed `k` and is used as the operator only.
      At the pitzDaily target-rung iterate, β = 0.5, cell 10824: `J_kk` goes **-1.74e-03 → +4.80e-03**,
      the effective diagonal **5.77e-04 → 7.12e-03**, `max |dk/k|` over the wall cells **2136 → 6.4**,
      `max |d log omega|` **15.6 → 0.119**, and the line search goes from `alpha` 0.25 to a **full
      step** reaching `|G|/|G0|` 0.103 against 0.783.
    - **The result that confirms the mechanism is that the arms COLLAPSE ONTO EACH OTHER.** Three
      gradient reconstructions whose exact `J_kk` spans 60 % (-1.22e-03, -1.74e-03, -1.95e-03) — and
      whose step quality orders the same way — give **+4.7892e-03, +4.7958e-03, +4.7985e-03** under the
      freeze, identical to 0.2 %, with `dk/k` of 6.0/6.4/6.7 and full steps to within 1 % of each other.
      So that one derivative was the whole of the reconstruction sensitivity.
    - ⚠️ **One state, one shift, NO MARCH.** A quasi-Newton operator can cost convergence rate near the
      root, and this iterate is not near it. The probe also replaces `closure.nu_t`, which reaches the
      `k` **diffusivity** as well as the production, so it is not surgically the production term; and
      the near-wall blend's own `k` and the Menter cap's `k` are untouched. The march A/B is the test
      that matters.
  - **⚠️ `explicit_production_limiter` now defaults to `False` (the EXACT operator) — measured
    2026-08-15, and the old `True` default was a silent adjoint hazard.** The flag freezes the cap's
    `k` in the **linearization** only (a Patankar / deferred-correction treatment). That is free only
    while the cap is **inactive at the converged root**; where it binds there, the IFT adjoint
    linearizes a residual different from the one solved, so the fields are right and the
    **sensitivity is silently wrong**. `KProduction`'s own docstring had always said the coupled path
    uses the exact operator "so the adjoint stays exact" — `SSTTurbulence` defaulted the opposite way,
    and `k_residual` serves both paths from one construction site, so there was no seam by which the
    coupled residual got the exact operator.
    - **Measured inert on both cases available.** Turbulent channel (Re 2500, 280 cells): cap active
      in **0** cells at the root, and the limiter ON/OFF forward solves are **bit-identical**
      (`0.000e+00` max field difference), gradients identical and both 1.2e-05 from central FD.
      `bfs3d` (23040 cells, Re 10000, separating), a controlled ON/OFF pair at the shipped bundle
      (petsc flow inverse, `inner_tol` 0.01): **59 steps / 232 cycles / final ‖R‖ 1.861e-06 in BOTH
      arms**, `x_r/h` 8.3611, report byte-identical. So the stabilization was buying nothing on either
      case, forward or adjoint.
    - **The cap DID bind transiently mid-march on `bfs3d`** — 4 steps of 59, at 0.013–0.065 % of cells
      (3–15 cells) — and changed nothing, which is what the identical trajectories say. **Why those
      cells is UNEXPLAINED.** ⚠️ The obvious reading, that they are the numerically-dead-`k` cells
      whose limit collapses through `maximum(k, 0)`, is **REFUTED by direct test**: at `k = 0` the
      eddy viscosity is zero too, so production and limit are *both* zero and nothing binds. Peak
      `S/ω` over that march was 0.5717 against a binding threshold of 0.9487, so the simple
      unlimited-branch criterion does not explain it either. Settle it by re-running with
      `BFS3D_CHECKPOINT_KEEP=80` and reading the binding cells; the per-step `cap%` / `S/w` metric is
      now in the march log (`compare.production_cap_metrics`), so the question is answerable rather
      than lost.
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
      `ImplicitNewtonSolver` (no preconditioner, no globalization) on the k equation, which stopped
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
  `.claude/rules/solve-globalization.md`; making them pytrees breaks both the IFT adjoint and the jit cache.
  `ScaledScalarPreconditioner(inner, scale)` wraps one with a fixed per-cell output factor — the
  reciprocal chain-rule scaling a log-transformed scalar block needs (above); also a frozen dataclass.
  - **Which continuation a solve runs is ONE injected source, not two parallel branches (binding, #278,
    2026-08-20).** `solve_coupled` needs a continuation twice — the initial build and every refresh —
    and those were written as two independent two-way branches (`refresh.builder` vs the default
    `coupled_continuation`), one at the build and one inside the refresh loop. `_ContinuationSource`
    (`build(state)` / `refresh(state, previous, residual_norm)`) makes it one decision, with
    `_CallerBuiltContinuation`, `_DefaultContinuation` and `_FinishedContinuation` as its three cases.
    That is the shape #282 had just been fixed for one level down; here it also carried a live defect.
    - **⚠️ `method` / `reference_state` / `**continuation_kwargs` are REFUSED where they cannot be
      forwarded, not dropped.** They configure the continuation `solve_coupled` builds. On the two paths
      where it builds none — an explicit `continuation`, or a `RefreshPolicy(builder=...)` — they reached
      nothing at all, with no error and no log line, so a march asked for `inner_steps=3` /
      `positivity_floor=1e-6` and silently ran the library defaults. `**kwargs` is what made it quiet:
      it accepts every keyword and checks none, and it is the main entry point's door. This had already
      cost a study harness (`lu_vs_hostilu.py` carried a warning comment about a `precondition_step=`
      swallowed here instead of reaching its `RefreshPolicy`).
    - **`method` now defaults to a sentinel (`_UNSET`), resolving to `"twolevel"` when the solve builds
      the continuation.** Both a real default and an explicit `None` ("no preconditioner method") are
      meaningful, so neither could stand for "not given" — and without that distinction the guard could
      not refuse an explicitly-passed `method` without refusing the default nobody asked for.
  - **`solve_coupled(refresh=RefreshPolicy(trigger=…))` segments the march to re-freeze the preconditioner — and a refresh
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
    `(refresh.limit+1)·max_steps` march steps plus the finishing solve's own allowance, deliberately not
    split — either segment may need the full allowance); the finishing solve is handed the **absolute**
    target `atol + rtol·‖R0‖` measured at the initial state, so a refreshed solve stops exactly where an
    unrefreshed one does for **any** number of refreshes (a relative tolerance would be measured against
    whatever the pre-march reached and compound a silent tightening per refresh — this is what the old
    `rtol/refresh_rtol` compensation approximated, and why the `refresh_rtol <= rtol` constraint existed;
    both are now gone). The absolute form is available precisely *because* the refresh path is
    forward-only, so `‖R0‖` is concrete rather than traced. The constrained path
  - **⚠️ THE CONSTRAINED ADJOINT TEST WAS PASSING BY ~1e-11, AND THAT IS A PROPERTY OF THE
    PRECONDITIONER, NOT OF THE TEST (measured 2026-08-19).**
    `test_constrained_coupled_adjoint_matches_finite_difference` differentiates at a warm state produced
    by an `air`-preconditioned solve. Two warm states differing by **6.6e-12 relative / 4e-9 absolute**
    — the same solve, before and after an unrelated change to the lAIR *setup* — take the transpose
    solve from converging to raising `EquinoxRuntimeError: A stagnation in an iterative linear solve`,
    **with the code held identical**. That was verified the only way that settles it: stash the code
    change out entirely, then feed each saved state to the same binary. So the test was gating on the
    stagnation detector's threshold rather than on the adjoint.
    **The underlying fact is the block-diagonal preconditioner's zero-shift limit**, which this path
    meets head-on: the constrained adjoint is a transpose solve at β = 0 against `BlockPreconditioner`,
    the regime it is weakest in. The test now passes an explicit
    `relative_residual_gmres(1e-8, restart=120, stagnation_iters=200)`, so it reports the gradient it
    claims to. **`solve_coupled_mass_flow` gained `adjoint_solver` to make that possible** — it was the
    only coupled driver without it, the same capability drift as the four continuation builders.
    **Consequence for anyone measuring here: a mass-flow adjoint result is not a stable gate.** Any
    change that perturbs the warm state in its last bits — a preconditioner tweak, a library version,
    a different accelerator — can flip it, and the flip says nothing about the change.
    (`mass_flow_coupled_continuation` / `solve_coupled_mass_flow`) has **no** staged refresh — thread
    `reuse` through if that driver is added.
    - **The trigger is forward-only — it *raises* under `jax.grad`, and must (binding).** The refresh
      re-derives the preconditioner from the **mid-march** state, which is a tracer when differentiating;
      the refreshed preconditioner would capture it and escape the converged solve's `custom_vjp` as an
      `UnexpectedTracerError` (the general "build the preconditioner from concrete params *outside*
      `jax.grad`" footgun: a `BlockPreconditioner` must be constructed once, from concrete parameter
      values, *outside* the differentiated region — build it inside and it captures a tracer and leaks).
      A refresh also forbids an explicit `continuation`,
      so there is **no** concrete-preconditioner path through it — hence the honest behaviour is a clear
      up-front `ValueError`, not a leak. `solve_coupled` guards this with `_is_traced((coupled, flow, k,
      omega))` (reliable because the solve is eager-only — the scalar AMGs are off-jit scipy, so a tracer
      leaf can only mean a wrapping transform). To differentiate, drop `refresh.trigger` and take the
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
    `march.py` bullets in `.claude/rules/solve-march.md` for why observation is not gated on the trigger and why
    the state rides a separate seam from the report history.
  - **`solve_coupled(step_control=…)` — the dual-time march DEFAULTS to the `DualTimeControl` Courant
    ramp; other controls are opt-in.** A `StepControl` reshapes the shift strength β each observed step
    from the previous report; all are forward-only (raise under `jax.grad`, same guard as the refresh).
    - **Default (`inner_steps > 1`, observing):** `solve_coupled` auto-selects `DualTimeControl` (the
      α-based Courant ramp) when the march is a `DualTimeStep`, a refresh/observer is active, and no
      control was supplied (`default_dual_time_control` -- **in `solve/step_control.py` since
      2026-08-15**, beside the controls it chooses between; it lived here only because an import cycle
      made it inexpressible in `solve/`. Unit-tested in `test_coupled_rans.py`). It grows
      the pseudo-timestep while the inner loop stays comfortable and **carries β across refreshes**,
      reaching a developed pitzDaily recirculation in materially fewer outer steps than the residual-keyed
      control, and carrying a full cold ramp to the target Reynolds number (step counts measured on pitzDaily,
      configuration not recorded — re-measure before relying on them). The injection never turns
      observation on, so the differentiable single-stage solve is untouched. See the DualTimeStep bullet
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
    `method="air"`** — lAIR's C/F split
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
  - **⚠️⚠️ REFUTED BY MEASUREMENT — "give the k shift the production feedback and take `abs` of the
    reaction diagonal" (issue #312, shipped as PR #317, REVERTED 2026-08-26).** The diagnosis was right
    and the prescription was wrong; it shipped green and made the case it targeted **worse**. Do not
    re-propose the `abs` half without reading this.
    - **The diagnosis stands.** `_coupled_shift_policy` builds the k shift from `k_residual(mdot,
      closure)`, whose `KProduction` reads `closure.nu_t` as a frozen array — so `nu_t·S²` holds no `k`
      and the `J·1` row sum carries the destruction feedback and none of the production's. The coupled
      residual's `nu_t = a₁k/max(a₁ω, S F₂)` is live, so `dP_k/dk = P_k/k`. On pitzDaily at the target
      rung's clipped iterate, cell 10824, that term is **+6.57e-03** against a shift of **+4.61e-03**.
    - **The prescription does not follow.** "A destabilizing source means the shift must be larger" is
      right about intent, but `abs` is taken of the **net**: the reaction term goes `r → |r − f|`, which
      is *smaller* than `r` whenever `f < 2r`. Measured `f/r = 1.48`, so the shift **halved**.
    - **Measured, both arms, same iterate, β = 0.5, `analyze` mode of
      `validation/pitzdaily_gradient_ab/closure_stall_probe.py` at `stall-iterate-skew.npz`** (the
      `exact` operator rows; compiled ILU(0) live, `PITZ_STALL_BETAS` default ladder):

      | arm | `βd_k` before → after | `J_kk + βd` before → after | kept α before → after |
      |---|---|---|---|
      | skew | 2.3066e-03 → 1.1448e-03 | **+3.6004e-04 → −8.0181e-04** | **0.03125 → 0.005371** |
      | owner | 2.3140e-03 → 1.0393e-03 | +5.7462e-04 → −7.0015e-04 | **0.25 → 0.006645** |

      `J_kk + βd` flips sign and the kept step degrades **6×** (skew) and **38×** (owner). The
      pre-merge `d_k` = 4.6132e-03 reproduces the issue's own 4.6196e-03, so the baseline is the state
      the issue measured.
    - **⚠️ NO TEST TIER CAN SEE THIS, which is why it shipped.** Fast, slow and validation were all
      green, and unit tests pinned `feedback_rate` against AD to 2.2e-16 — correctly, since the closed
      form is right; it is the *consumer* that is wrong. **Nothing anywhere asserts step productivity on
      pitzDaily**, so a change that halves a shift is invisible to every gate. A shift-diagonal change
      must be measured on `closure_stall_probe.py`'s `analyze` mode before it merges; green tiers are
      not evidence about it.
    - **What was NOT refuted, and is worth re-landing on its own:** `KProduction.feedback_rate` (exact
      against AD in both cap branches under either linearization, an estimate only under the near-wall
      blend), and returning the reaction diagonal **unclamped** from `_scalar_operator_pieces` with the
      preconditioner keeping its own `max(·, 0)` — that pair is a behaviour-preserving separation. What
      is refuted is `abs(·)` on the shift and `live_eddy_viscosity=True` on the coupled path.
    - **The untried variant is `|r| + |f|`** — add the destabilizing magnitude rather than net it, which
      at this cell predicts `J_kk + βd` = **+3.65e-03**. Arithmetic only; **unmeasured**, and it must be
      measured on the probe, not on the tiers.
    - **`explicit_production_viscosity` already addresses #312** and is the reason the revert costs
      nothing: at this same cell and iterate its `frozen` rows give `J_kk` **+4.78e-03**, α **1**, and
      `|G|/|G0|` **0.103**, against the exact operator's stalled 0.975.

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
    - **OpenFOAM `omegaWallFunction` (default): ⚠️ NOT a power mean at all — a HARD SWITCH at `yPlusLam`.**
      Read from source: `blended_(dict.lookupOrDefault<Switch>("blended", false))`
      (`omegaWallFunctionFvPatchScalarField.C:215`), and the `blended_` branch at `:81` selects
      `omega_vis` below `yPlusLam` and `omega_log` above, with no combination. (`blended = true` is an
      *exponential* `exp(-Rey/11)` interpolation — also not a max.) The earlier entry here described it as
      `p → ∞`, `max(omega_vis, omega_log)`; **that names the wrong mechanism**, and the correction matters
      in a specific band: the two branches cross at `y* ≈ 9.84` while the switch is at `y*_lam = 11.53`, so
      between those a `max` takes `omega_log` where OpenFOAM takes `omega_vis`. Everywhere else `max` is a
      good numerical proxy, which is why the measurement below still stands.
      **Measurements, each with its case** (they differ, and the rule is to name what each was taken on):
      on **pitzDaily** wall cells `max` matched OF's field to **<2 %** (median ratio 1.00) while aquaflux's
      `p = 2` ran **~20 % high in the buffer layer** (`y+≈8–15`, median aquaflux/OF `omega_wall` = 1.20);
      on **bfs3d** at the converged root the median `omega_aq/omega_OF` over the fine wall layer is
      **1.129** (p90 1.28, max 1.41). The wall distance and constants **agree** in both.
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
    only at rel ~0.05 (the SER-schedule convergence problem — see `.claude/notes/solve-globalization-log.md`). On the clean field
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
    and the **derivative in `k` against a central difference**); the flow-side seam is in
    `.claude/rules/flow.md` / `.claude/rules/discretization.md`.
    ⚠️ **That last one used to read "differentiability" and was a finiteness check, which cannot fail:**
    `0.0` is finite, so wrapping `nut_wall`'s return in `stop_gradient` left all 28 tests in that file
    green (measured 2026-08-21), and nothing anywhere else noticed either — 119 more turbulence/coupled
    unit tests and the wall-resolved `test_channel_law_of_the_wall` all passed. The derivative is in every
    Jacobian and every adjoint this closure appears in, so it now has a finite-difference comparison and a
    non-zero assertion. **Its VALUE in the log branch is still integration-unverified**: doubling that
    branch kills three unit tests and neither turbulent-channel integration test, because the coupled
    fixtures are wall-resolved (`y*` = 1.44 at the wall-adjacent cells against `y*_lam` = 11.53), so the
    `jnp.where` selects the zero branch at every wall face.
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

    **⚠️ (2026-08-10) A cell with three wall faces DOES have a non-negative root — the alarm below is
    withdrawn, but read it: the near-wall `k` collapse it describes is real and unexplained.**
    Measured directly: holding every other field fixed, `R_k` at the worst cell is linear to seven digits
    with root `k* = +1.99e-14`, strictly positive, and the iterate sits eight decades *below* it. The
    corner `k` system is homogeneous and an M-matrix — production, destruction and the `Dirichlet(0)` wall
    term all vanish linearly at `k = 0` — so `k = 0` solves it exactly when the neighbours are 0 and no
    negative root exists. The near-wall `k` trough seen at that state is a **Reynolds-
    continuation transient, not a defect in this closure** — the converged target-Re root has **zero**
    cells below `k = 1e-6` (`k_min` 1.30e-05, median 0.7017 against OpenFOAM's 0.7468).

    **⚠️ BUT THE CAUSE IS THIS BLEND, USED OUTSIDE ITS RANGE, AND THAT IS WORTH KNOWING WHEN CHOOSING A
    CONTINUATION LADDER.** `y* = β*^0.25 √k d/ν` scales as **1/ν**, so at a Re/100 anchor
    `wall_function_weight = tanh((y*/y*_lam)⁴)` is ~1e-7 and the **log-layer production is switched off
    even at fully turbulent `k`**, while `ω_wall = 6ν/(β₁d²)` is simultaneously 100× larger. Destruction
    then dominates a homogeneous linear row and `k` decays; raising Re reverses both (weight median
    5.6e-20 → 1.9e-10 → 1.000 across the three rungs). **Anchoring a ladder below the Reynolds number at
    which the wall function turns itself on manufactures a near-wall `k` trough.** Full account in
    `.claude/rules/solve-amg-multigrid.md`.

    ❌ **REFUTED — the "ω is exactly 10×, i.e. still at its 60ν seed" lead.** That ω is `6ν/(β₁d²)` at the
    *rung's own* viscosity to 16 digits; the 10× compared a rung-2 state against the target-Re formula.
    (The unclamped `jnp.sqrt(k)` in `SSTModel.f1`/`.f2` that this bullet once flagged is **fixed** —
    both now call `safe_sqrt(jnp.maximum(k, 0.0))`, as does `eddy_viscosity` and both production caps.)

    **Open, and now the real physics question:** the near-wall `k` at the converged `bfs3d` root sits below
    OpenFOAM's. Leftover of the trough, or an independent wall-treatment difference? Untested. **Do not quote
    a bare ratio for this.** The ratio is dominated by how "the first wall layer" is defined — all wall-face
    owners, the finest-spacing layer, and the floor alone are three different cell sets whose OpenFOAM `k`
    medians differ by more than the discrepancy being investigated — and the reattachment lengths it is said
    to bear on are two grid stations apart on a metric quantized at ~0.5 h. Run
    `validation/bfs3d_openfoam/wall_layer_comparison.py <state.npz>`, which reports both quantities every
    defensible way with the cell set named beside each, and quote it *with* the cell set and the state.

    The original alarm, now known to be a wrong inference from a right observation: On the 3D backward-facing step the coupled march is
    killed by the `k`-positivity limiter binding on a single cell of 23040, and the two cells that own
    the cap are the mirror pair of **trihedral wall corners** behind the step — `lowerWall` twice (the
    floor plus the vertical step face) and `sideWalls` once, so half of every face is no-slip. Across the
    four tightest cells the ranking is exactly the wall-face count, 3 / 3 / 2 / 1. There the Newton
    direction keeps demanding a `k` change of ~1e-13 while `k` itself has been ratcheted down to 1e-22
    against a mesh median of 2.97e-02. **The root there is `+1.99e-14` — positive**; the constraint is
    active because the iterate sits eight decades below its own root, not because no root exists. (Full
    measurement, and the step-length lock-up it causes, in `.claude/rules/solve-amg-multigrid.md`.)

    **❌ The first suspicion — that a three-wall-face cell takes several times the destruction against
    one cell's worth of production, because the two terms are reduced over a cell's wall faces
    differently — is REFUTED. Our multi-wall reduction is already the same one OpenFOAM uses.**
    - `wall_cells = jnp.unique(mesh.face_cells.owner[wall_faces])`, so a cell appears **once** however
      many wall faces it has, and `NearWallKClosure` writes with `.at[cells].set(...)` on those unique
      indices. Nothing is summed per face.
    - `SSTTurbulence.wall_shear_rate` reduces to `Σ_f |S_f| r_f / Σ_f |S_f|` over the cell's wall faces
      — an **area-weighted average**, which is exactly OpenFOAM's `patchFieldsToWallCellField`
      (`wallCellWallFunctionFvPatchScalarField.C`: `Σ|Sf|·φ / max(Σ|Sf|, vSmall)`, accumulated over
      *all* wall-function patches so a corner cell spanning two patches is still one cell).

      OpenFOAM adds two mechanisms we do not have, neither of which is a reduction: a **master patch**
      elected across every wall-function patch on the field, so the correction is computed once per wall
      *cell* rather than once per patch; and `wallCellFraction`, the ratio of finite-volume to polyhedral
      wall area with `tol_ = 1e-1`, which is 1 on a conformal mesh like this one.

    **What the two codes DO differ by, read from the OpenFOAM 13 sources, and neither difference is
    about corners:**
    1. **OpenFOAM does not solve `ω` in a wall cell, and its `G` is a lagged constant.**
       `omegaWallFunctionFvPatchScalarField::manipulateMatrixMaster` calls
       `matrix.setValues(wallCells, wallCellOmega, wallCellFraction)` — the row is *replaced* — and
       `updateCoeffsMaster` overwrites `G` in those cells with the wall-function value computed from the
       **previous iterate's** `k`. So its wall-cell `k` equation is linear in `k` with a lagged,
       non-negative source and a fixed positive destruction coefficient. Ours keeps both live functions
       of `k` inside one Newton residual, which is deliberate (see pieces 1 and 3: against a *frozen* `ω`
       the row degenerates and the solve runs away) — but it is what makes a negative root reachable.
    2. **OpenFOAM does not rely on the `k` equation having a non-negative root at all; it PROJECTS.**
       `kOmegaSSTBase.C` ends every `k` solve with `solve(kEqn); fvConstraints.constrain(k_);
       bound(k_, kMin_)`. And `bound` is not a plain clip — where the field went negative it takes the
       **average of the neighbouring cells**, then floors at `kMin`:
       `isf = max(max(isf, fvc::average(max(vsf,min))*pos0(-isf)), min)`. A cell whose `k` equation has
       no non-negative root is therefore a non-event in OpenFOAM: it is overwritten each iteration.
       We instead constrain the *step*, which is what locks the march up.

    So the wall-face-count ranking (3/3/2/1) is most likely reading **stagnation** rather than any
    double counting — three wall faces means the deepest dead zone, hence the least production — and the
    open question is the one in item 1, not a corner-specific defect in the reduction.

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
  `reused_flow_solve`, the velocity `a_P` live). Handed to `ImplicitNewtonSolver`, it gives the
  **exact coupled adjoint** (§5) — a single transpose solve on the unfrozen `R_coupled`. The ω wall
  rows are `FixedValueCells`. `CoupledRANS.build` pre-resolves the k/ω boundaries (via
  `turbulence.resolve_boundaries()`, the shared idempotent bind the segregated driver also uses) so the
  per-eval assembler rebuild's `resolve` is an idempotent no-op (else a dynamic-shape `nonzero` on
  traced mesh labels breaks the jit). **`CoupledRANS.build` now checks the two densities agree (issue
  #157's second finding, fixed 2026-08-18).** `SSTTurbulence.density` (a scalar, used to form the k/ω
  volume flux `mdot / density`) and `MomentumContinuity.density` (a per-cell array from the flow's
  `PropertyModel`) are supplied to two independent builders that never see each other, so nothing
  previously caught a caller passing a different value to each — the flow block never reads
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
  not a behaviour change for any of them. **`CoupledRANS.residual` assembles the Rhie–Chow flow fields once
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
    **Every** continuation builder therefore wires `positive_k_limit(coupled)` — the fraction-to-the-
    boundary cap — automatically for a directly-solved `k`, and passes `None` for a log-solved one, where
    positivity is already structural and a cap would only throttle. See `.claude/rules/solve-march.md`.
    ⚠️ **Until 2026-08-19 only `coupled_amg_continuation` did, and that is worth knowing before citing
    any comparison between the builders.** `coupled_continuation` and `coupled_lu_continuation` never set
    `step_limit`, whose default is `None`, so the block-diagonal and complete-LU paths marched with **no
    positivity guard at all**. Two consequences. Any recorded block-SIMPLE-versus-monolithic result
    compared a guarded march against an unguarded one. And the recorded verdict that "the low-β wall is
    the block-SIMPLE preconditioner — its coupled solve goes NaN at a low shift"
    (`solve-globalization-log.md`) is now **suspect**: an unguarded `k` crossing zero produces exactly
    that signature, so the preconditioner may have been blamed for a missing globalization. Not
    established — the experiment is one flag and has not been run — but that verdict is dead anyway on
    its own terms: its remedy, the monolithic ILUT, was measured dominated and **deleted**, and the
    shipped AMG preconditioners reach adjoint grade at zero shift, which is the regime it claimed only a
    complete factorization could reach (the entry is collapsed to one line in
    `solve-globalization-log.md`). All **four** builders now route through `_coupled_step`, and
    `test_every_continuation_builder_installs_the_same_globalization` fails if one loses the guard.
  - **⚠️ THE FORWARD SOLVE'S STOPPING MEASURE IS `_coupled_step`'s, NOT A BUILDER'S (binding, #282,
    2026-08-20) — and the surfaces above `_coupled_step` had drifted TWICE MORE after the tail was
    extracted.** `_coupled_step` builds the default forward solver from `residual_norm` — the march's own
    progress measure — so the solve is steered by and judged by one definition. What is per-family is a
    `_ForwardSolveRegime` (rtol, restart, cap), and all four builders take `forward_rtol` /
    `forward_restart` / `forward_max_restarts`. **Move either through those, never by passing a whole
    `forward_solver`, which replaces the measure too.**
    Two things this changed that a reader of an older measurement needs:
    - **`coupled_continuation` and `coupled_lu_continuation` previously stopped on a plain 2-norm at
      `1e-2`.** They now stop on the row-scaled measure at `0.3`. The 2-norm of the coupled residual is
      ~100% `ω`, so it halts once `ω` is resolved while the flow-dominated Newton step is still coarse
      — a measured ~116 % velocity error, which is what the multigrid builder's own docstring had been
      calling effectively blind while the *default* path did exactly that.
    - **`0.3` is the multigrid family's calibration, carried across as a property of the MEASURE and NOT
      re-measured on those two families.** It is documented as such. The tolerance costs ~1.5× the
      plain-2-norm cycle count where it was measured; neither flagship case runs those builders (both
      build `coupled_amg_continuation` explicitly), so nothing on record was taken under it.
    - **`mass_flow_coupled_continuation` is the one genuine exception and stays Euclidean at `1e-2`**,
      because the row-equilibrated measure has no constraint-aware form — it would scale the border row
      by a diagonal the constraint does not have. Its tolerance differs *because its measure does*; the
      reason is recorded at `_CONSTRAINED_FORWARD` itself so it reads as a decision, not an omission.
    **`log(k+1)` does not fix this**: `k = e^w − 1` bounds `k > −1`, not `k > 0`, so the failure above is
    still reachable. It is regular at the wall (unlike `log k`) but that solves the other problem, not
    this one.
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
    `coupled_continuation` sets `line_search=_COUPLED_LINE_SEARCH` (see `.claude/rules/solve-globalization.md`); β
    escalation stays the fallback for a bad *direction*, not an overshoot. With it the full-mesh solve
    **descends** (rel 1.0 → 0.48 → 0.44 → 0.31 → 0.20 → ~0.18 over ~6 steps) instead of *stalling at
    rel 1.0* — the case is now solvable at all, a correctness fix, not just speed. **(2) The shifted
    solve needs a large Krylov subspace:** the block family's regime (`_BLOCK_FORWARD`, ⚠️ **the symbol
    `_COUPLED_FORWARD_SOLVER` no longer exists**) is restart-120 GMRES (the shared
    restart-40 default discards too much Arnoldi history on this stiff saddle system; ~1.4× faster to
    the same solution). **The forward-solve TERMINATION, not a tight tolerance, was the dominant coupled
    cost — and the old "tight tolerance is load-bearing under log-`ω` / loosening breaks the march" claim
    was STALE (corrected 2026-07-28).** That solve moved from `lineax`'s componentwise stop to a
    *relative-residual* stop (`relative_residual_gmres`, ~1% per inexact-Newton step): it reproduces the
    over-solving march's `x_r/h` trajectory to 3-4 significant figures per step with no log-`ω`
    divergence, at ~4× fewer matvecs. ⚠️ **The norm it stops in has since moved again** — since #282
    every family stops in the march's own row-scaled measure at `forward_rtol = 0.3`, not a global
    2-norm at 1e-2, so the "~1%" here describes the arrangement this measurement was taken under and not
    the current default. See the `forward_solver` bullet in `.claude/rules/solve-globalization.md` for the mechanism
    (`lineax`'s componentwise stop plus the near-zero-right-hand-side ω wall-fixation rows pinned it to
    the absolute `atol=1e-10` floor, ~9 orders past the requested 1e-3) and the two-arm refutation.
  - **The march's default residual measure is the row-equilibrated `RowScaledNorm` (`coupled_scaled_norm`),
    NOT the plain Euclidean ‖R‖.** The Euclidean coupled residual is dominated by the `ω` block (`ω` O(1e5),
    `k` O(1e-3)), so it barely moves while the flow develops and *mis-ranks* states — a converged field can
    score worse than a badly wrong one, and a step collapsing `k` is accepted (see the mis-ranking warning
    in `.claude/notes/solve-globalization-log.md`). `RowScaledNorm` divides each row by its own diagonal and each block by its
    field magnitude, reporting a fractional change per equation, so steering and the stopping test judge
    every block comparably. `coupled_continuation` / `coupled_lu_continuation` build it by default;
    `block_scaled_norm=True` selects the coarser one-scale-per-block `BlockScaledNorm` (`_coupled_residual_norm`),
    and `residual_norm=jnp.linalg.norm` recovers the plain Euclidean measure. (`mass_flow_coupled_continuation`
    still defaults to Euclidean — its bordered `[flow, k, ω, β]` state needs a constraint-aware row-scaled
    variant not yet built; a follow-up.) **This does not *fix* the forward stall** — the pitzDaily march is
    globalization-bound and plateaus under any measure (the row-scaled measure is the honest signal of that,
    where the Euclidean fall was a `β×travel` + `ω`-magnitude artifact). It makes the measure honest, and it
    is REQUIRED for the case to be judged correctly. **When a march refreshes, the measure is held fixed at
    the initial state** — `solve_coupled` passes `coupled_continuation(residual_norm=base_norm)` on every
    refresh rather than rebuilding it at the developed state, or the self-normalising scales would re-base and
    the convergence test become unreachable (#156 seam 4; see `.claude/notes/solve-globalization-log.md`). `scaled_norm=True`
    opts the *observed* march into rebuilding the row scales per outer step (finer, more expensive).
  - **`beta_floor` (SER lower bound) is available but off by default (a measured wash).** Bounding
    `β = max(beta_floor, β₀(‖R‖/‖R₀‖)^p)` keeps each late shifted solve out of the ill-conditioned low-`β`
    regime (correctness-safe — the floor scales the correction `δ`, which vanishes at the root, so it never
    moves the converged state). But end-to-end it is a **net wash** (cheaper late solves cancel the extra
    Newton steps), so it defaults to `0`; wired through `coupled_continuation` for further evaluation. The
    settled coupled-solve cost is the diagonal-block-preconditioner weakness at high Reynolds number, **not**
    the residual measure, `β` floor, or missing cross-coupling (a block-triangular preconditioner was worse
    — non-convergent on recirculating pitzDaily). See `.claude/notes/solve-globalization-log.md`.
  - **The coupled flow block uses the convection-aware velocity AMG, not the viscous-smoothed default
    (`_coupled_shift_policy`).** A RANS case is high-Reynolds, and the default `BlockPreconditioner.build`
    velocity config (viscous-**smoothed** AMG, which is Peclet-blind) produces a poor momentum-block
    direction once the flow separates. Measured on the developed pitzDaily field (shifted Newton direction
    vs the true one): smoothed gives a direction badly misaligned with the true Newton step and the march
    stalls at rel ~0.18, while **`velocity="convection"`** is nearly aligned. The convection block's
    linearization stays valid **frozen at the cold initial state**, so **the flow block needs no reference
    refresh** — verified two ways: an IC-frozen direction is as well aligned as a plateau-rebuilt one, and
    refreshing the flow block alone at a separated pitzDaily state is if anything slightly *worse*. It is
    **not** the flow↔turbulence cross-coupling (the block-*diagonal* preconditioner with the right config
    is already nearly aligned on its own — a block-triangular coupling was built, measured, and is worse;
    see `.claude/notes/solve-globalization-log.md`). **The k/ω *scalar* AMGs are the exception: they do go
    stale, and refreshing them alone once the flow separates cuts the outer cycle count materially**
    (configuration not recorded — re-measure before relying on the size) — the one staleness lever that
    pays; see the staleness bullet in `.claude/notes/solve-globalization-log.md`. Overridable via
    `preconditioner_kwargs`.
  - **⚠️ THE PRESSURE SCHUR NO LONGER HARDCODES `schur_scaling="msimple"` (fixed 2026-08-18) — it was
    never necessary at the scale this policy is actually used at, and is dominated where it matters.**
    Superseded finding, kept for the trap: this bullet used to pair `velocity="convection"` with
    `schur_scaling="msimple"`, on the belief that a RANS-scale coupled solve needed MSIMPLE's
    velocity-independent Schur to avoid the plain SIMPLE `a_P` Schur degrading under convection — cutting
    the shifted solve by roughly an order of magnitude in GMRES cycles on a developed pitzDaily state (no
    β, tolerance or norm recorded — unfalsifiable, and superseded regardless). Two things closed it: (1)
    neither flagship case (`bfs3d`, `pitzDaily`) reaches this policy at all — both build
    `coupled_amg_continuation(...)` explicitly, whose `_monolithic_shift_source` constructs the shift
    policy with `build_flow_block=False`, so no `BlockPreconditioner` — hence no `schur_scaling` — is ever
    built, regardless of `FIELD_SPLIT`/`FLOW_INVERSE`; this policy is only reached by `solve_coupled`'s
    zero-config fallback and by small (hundreds-of-cells) unit/integration fixtures. (2) On EVERY one of
    those small fixtures (`test_coupled_rans.py`, `test_coupled_mass_flow.py`, `test_coupled_lu.py`,
    `test_coupled_periodic_channel.py`, `test_periodic_channel.py`, `production_cap_activity.py` — Re up
    to 20000), dropping `schur_scaling="msimple"` in favour of `BlockPreconditioner`'s own default reaches
    the identical converged fixed point (fields and residual agree to the arms' own tight tolerances; the
    shift vanishes at the root so a Schur choice cannot move it). And on `bfs3d`'s real coupled Jacobian, a
    controlled single-variable swap (`validation/bfs3d_openfoam/simple_type_swap_probe.py`, at the converged
    root, everything but the leading inverse held at the shipped bundle) shows MSIMPLE costs ~8-9x the
    shipped bundle's Krylov cycles (6→53 at the adjoint operator, 5→40 at the march's own shift) — see
    `.claude/notes/solve-flow-block-log.md` § "MSIMPLE swapped in for the SHIPPED leading inverse".
    ⚠️ **That ~8-9x is measured against a `hostilu` leading inverse this case no longer ships, and on a
    method whose pressure prediction was missing; re-measured 2026-08-20 with Klaij & Vuik's Algorithm 2
    actually built and against the current `simplesmooth` default, MSIMPLER is 32 cycles against 17 at the
    adjoint operator but 22 against 19 at the march's own shift** — still behind, but not by an order of
    magnitude, and not by the same factor at both operators (same file, the following section). It does
    not change the conclusion below, since the fixtures that reach this policy converge identically
    either way. So
    `_coupled_shift_policy` now leaves `schur_scaling` at `BlockPreconditioner`'s own default; `velocity=
    "convection"` is unaffected and stays. `schur_scaling="msimple"` remains a real, non-dominated option
    of `BlockPreconditioner` for a **standalone, flow-only, convection-dominated** solve — see
    `tests/integration/test_channel_high_reynolds.py::test_mass_scaled_schur_reaches_beyond_the_a_p_schur`,
    where plain SIMPLE's inner GMRES genuinely stalls and MSIMPLE converges — just not for a coupled RANS
    solve at any scale this project has measured.
  - **`coupled_lu_continuation` / `coupled_lu_refreshing_continuation` — the COMPLETE-LU coupled PC, the
    preferred coupled PC on 2D/moderate meshes (BUILT).** A drop-in for `solve_coupled(continuation=…)`
    that preconditions the whole `[flow, k, ω]` saddle by factoring the assembled coupled Jacobian
    *completely* (`MonolithicLuPreconditioner`, `.claude/rules/solve-direct-preconditioners.md`), instead of
    the block-diagonal SIMPLE composition, so the preconditioner is the operator's exact inverse and a
    Krylov solve converges in **one** iteration. It **forms the true pressure Schur through the
    factorization's fill** rather than approximating it — the block PC's measured wall is the Schur
    *approximation* (the "Stage 3" note above / `.claude/rules/flow.md`), which the complete factorization
    sidesteps. `MonolithicFactorShiftPolicy` **reuses `CoupledShiftPolicy`'s
    pseudo-transient shift diagonal** (the physics — same velocity `a_P` + k/ω transport diagonals) and
    swaps only the preconditioner. It takes **only that diagonal**, so both monolithic builders
    construct their base through `_monolithic_shift_source`, which builds the policy with
    `flow_preconditioner=None` — the shift's velocity buckets come straight from the assembler
    (`flow.frozen_momentum_diagonal_parts`), and the block preconditioner that used to be built for them
    contributed two never-applied multigrid hierarchies whose value-dependent coarsening recompiled the
    coupled solve at every Reynolds rung. Asking a shift-only policy for a composed preconditioner
    raises. The factorization is a host object so it rides as a **static**
    field and is applied via `jax.pure_callback`, with the adjoint's `Mᵀ` supplied directly through a
    `TransposedPreconditioner` (the generic `jax.linear_transpose` machinery cannot transpose a callback —
    `.claude/rules/solve-direct-preconditioners.md`). **Its forward-solve regime is
    `_FACTORIZATION_FORWARD` (restart-10),
    NOT the block path's restart-120 `_BLOCK_FORWARD`:** an exact factorization clusters the preconditioned
    spectrum so tightly that the 1% stop is reached within a handful of vectors, and `lineax` GMRES only tests
    convergence at each restart boundary (its sole mid-cycle exit is exact Arnoldi breakdown, which does
    not fire while the residual is merely small), so a 120-vector restart pays many wasted back-solves per
    cycle where a small restart stops as soon as it has converged. The two
    regimes are tuned oppositely on purpose — the block PC genuinely needs a large subspace per cycle, the
    exact factorization needs a small one. That **restart** difference is real; the stopping *measure* is
    not a per-family choice and is `_coupled_step`'s (#282). Verified: `solve_coupled(continuation=coupled_lu_continuation(...))`
    converges to the **same fixed point** as the block PC and passes the **coupled-adjoint FD gate**
    (`tests/integration/test_coupled_lu.py`, run under the `scipy` backend so CI needs no optional dep —
    the complete factorization is exact regardless of backend). With the UMFPACK backend
    (optional `petsc4py` dep, `backend="auto"|"umfpack"|"scipy"`) it factors the developed pitzDaily
    coupled Jacobian quickly, exact (1 GMRES iter), verified on the real forward operator
    and the β=0 adjoint. **Cheap in-place mid-march refresh —
    `coupled_lu_refreshing_continuation` (BUILT, forward-march only).** For a differentiable solve the
    factorization is frozen at the reference state (state drift costs only a few cycles, and freezing
    keeps the adjoint valid). For a long developing march it instead goes stale — on a low-shift dual-time
    path it can NaN — so `coupled_lu_refreshing_continuation(coupled, …)` returns a `refresh.builder`
    for `solve_coupled` that re-factors the LU **in place in the SAME continuation object**
    (`MonolithicLuPreconditioner.refresh_in_place`), so the jitted march-step is a compilation cache hit
    (no recompile) — pair it with a `CoefficientDriftTrigger` so the re-factor leads the staleness. This
    is impure and **forward-march only** (never differentiate through it); see `.claude/rules/solve-direct-preconditioners.md`
    for the mechanism (static preconditioner field + callback reads `self.factors` at call time). It is a
    reasonable default for a *differentiable* coupled solve on a 2D/moderate mesh — but it is
    **NOT** the only PC with a working β=0 coupled adjoint: `coupled_amg_continuation`
    also passes the coupled-adjoint finite-difference gate with a shipped test
    (`test_coupled_amg.py::test_amg_adjoint_matches_finite_difference`). It does **not** shorten the
    reachability crawl (a per-step-cost + adjoint-correctness lever, not a globalization one). NOTE: it is
    not yet `solve_coupled`'s default — the two continuations have different parameter surfaces, so making
    it the default needs a selector seam, not a swap (tracked).
  - **SCOPE: a 2D / moderate-mesh tool** — the
    complete LU's fill (`O(n^{4/3})` in 3D) is a memory wall past ~10⁴ 3D cells (measured), so large 3D
    stays on the algebraic-multigrid path (`.claude/rules/solve-direct-preconditioners.md`).
  - **`coupled_amg_continuation` — the ALGEBRAIC-MULTIGRID counterpart, the coupled PC for large 3D
    (BUILT).** Same drop-in as the LU builder but preconditions with one smoothed-aggregation
    multigrid V-cycle (`MonolithicAmgPreconditioner`, `.claude/rules/solve-amg-multigrid.md`) instead of a factorization —
    a **direct-LU coarse solve** keeps the heavy fill on only the small coarsest grid, so it builds in
    ~seconds with bounded memory where the complete LU hits the 3D wall (its fill OOMs; a monolithic
    threshold-incomplete-LU factorization was tried and measured to hit the same wall from the time side
    instead — its `spilu` on the distance-3 3D Jacobian — measured 38.7M nnz on `bfs3d` — ran >7.5 min and
    never finished, before being deleted as dominated). Shares
    `MonolithicFactorShiftPolicy` + `_monolithic_factor_step`; it builds its forward solver **inline** —
    `forward_rtol = 0.3` measured in the **row-scaled** `coupled_scaled_norm`, `restart=15`,
    `max_restarts=60` (there is no `_COUPLED_AMG_FORWARD_SOLVER` symbol; the row-scaled stop is the
    substantive part, since a plain 2-norm stop is ~100% `omega` and leaves the flow correction blind). The V-cycle is a **fixed linear operator** (one apply), so the coupled-adjoint reuses
    its **transpose** V-cycle — verified: reaches the block PC's fixed point AND passes the coupled-adjoint FD
    gate (`tests/integration/test_coupled_amg.py`). **Needs `petsc4py`** (the one builder that does; the module
    raises a clear install hint otherwise). The smoother must stay **stationary** (a Krylov-accelerated one
    makes the V-cycle nonlinear, so it needs flexible GMRES and has no clean transpose — it can never be the
    adjoint path). ⚠️ **Fill level: the validated `bfs3d` bundle is ILU(0) × 4 sweeps**; "ILU(0) stalls" was a
    HIGH-β measurement and is superseded — at low β it is ILU(1) that breaks down. The library defaults still
    ship `smoother_fill_levels=1, smoother_sweeps=2`, so the case bundle and the library disagree; the
    `sweeps=2` optimum was tuned against ILU(1) and does not carry over. Restart-15
    stops the forward solve at the ~1% inexact-Newton tolerance, and extra smoother sweeps pay for themselves
    on the low-shift operator the march's tail runs at, at one extra cheap incomplete-LU back-solve each. (The
    sweep figures once quoted here were measured at ILU(1), where the optimum was two sweeps; the validated
    `bfs3d` bundle is ILU(0) × 4 and the ILU(1) numbers do not carry over — see the zero-fill smoother bullet
    in `.claude/rules/solve-amg-multigrid.md`.) This is the coupled
    preconditioner the first 3D validation case (`validation/bfs3d_openfoam`) runs on.
  - **`amg_beta_tracking_refresh` — rebuild the V-cycle as β drifts, the enabler of the 3D
    dual-time march (BUILT).** The AMG sibling of `lu_beta_tracking_refresh`.
    A dual-time march ramps β down to develop the recirculation, and a V-cycle frozen at `amg_beta` degrades
    sharply as β leaves that value (measured on the `bfs3d` cold march: an order-of-magnitude rise in outer
    cycles per solve, and the march stalling, as β fell — but that run's smoother fill, sweeps, aggregation and
    coarse-eq limit were not recorded and two of those defaults have since moved, so re-measure before relying
    on the counts). Re-materializing the Jacobian and rebuilding the GAMG at the step's `(state, β)` restores
    the matched cheap solve — the ~tens-of-seconds rebuild is far cheaper than the hundreds of extra
    Jacobian-vector-product matvecs a stale V-cycle costs. This is the only β-tracking that carries the 3D
    case: the complete LU's factorization is out of memory there. Same forward-only contract as the LU
    hook (raises under `jax.grad`); pass it to
    `solve_coupled(refresh=RefreshPolicy(precondition_step=…))` (or a `solve_reynolds_continuation` `point_setup`) with a
    `coupled_amg_continuation` step and a `DualTimeControl`.
    - **The refresh cadence is GATED, not every-step (measured on the developed-low-β tail).** Refreshing
      unconditionally every step is wasteful once the march has developed and β is nearly constant, and — the
      failure that motivated the gate — a *fixed step-count* cadence (`refresh_every`) fails at the tail: a
      long low-β cruise between refreshes lets the V-cycle go stale mid-interval and the cycle count explodes
      before the next scheduled refresh (same unrecorded bundle as the bullet above). The honest staleness
      signal for this hook is
      **β itself** (a V-cycle's degradation is a function of the β-mismatch, above), so with
      `beta_rel_change` set the refresh fires when `|β − β_last|/β_last` exceeds it (`_staleness_beta_gate`,
      shared with the shift-tracking mechanism above), OR after `refresh_every` steps as a development
      backstop, and the β-move prong is what catches an overshoot / rung
      restart before the solve stalls. **Why a β-move trigger, not a drift trigger, for this job:** a
      badly mismatched frozen factorization collapses the line search (α→0), which freezes the state and
      hence every drift measure taken on it — so a drift trigger cannot fire on exactly the failure mode
      a β-move trigger is watching for. Default (`beta_rel_change=None`) keeps the every-step behaviour.
    - **Cheaper refresh — the β-diagonal split, drift-gated (`materialize_drift` / `materialize_every`).** The
      shifted operator is `J(state) + β·d(state)`; between two refreshes at the same developed state only β (and
      the shift diagonal) has moved, and re-materializing `J` by graph-coloured probing is ~half the refresh
      cost (~18–40 s of the ~36–62 s total; the GAMG refactor is the other ~18 s). `MonolithicAmgPreconditioner`
      therefore caches the un-shifted Jacobian (`_materialize_jacobian` / `_shifted` split) and exposes
      `refresh_shift_in_place(shift_diagonal)`, which re-forms only `J + β·d` on the cached `J` and re-sets-up
      the GAMG. The full re-materialize is **gated** (`_materialize_gate`, the drift/step-cap analogue of
      `_staleness_beta_gate`): `materialize_drift=τ` fires it when the ν_t drift since the last one exceeds `τ`
      (`eddy_viscosity_drift`, the honest staleness signal — prefer this), `materialize_every=K` is the
      step-count safety cap; both `None` (default) re-materialize every refresh. Since gradients are never
      taken through the forward march, the in-place mutation is safe. **⚠️ Measured: DON'T under-materialize.**
      On a fast-developing flow a *fresh* `J` cuts the Krylov cycle count on `bfs3d` (fewer/cheaper steps) by
      more than the materialize costs, so a stale frozen `J` is a net loss (the reduction was recorded with no
      state, β or preconditioner bundle — re-measure before relying on its size); the right lever is a
      *cheaper* materialize, not a rarer one — hence the `sparse_jacobian` speedups (batched probe,
      gather de-compression, the saturation colouring and the per-column reach; see the
      materialize-efficiency bullet in `.claude/rules/solve-amg-multigrid.md`), which `coupled_amg_continuation` and the
      refresh wire in (built once, reused). **DON'T lower `stencil_reach` to 2 to cheapen it** — reach-2 is
      numerically near-exact at *every* state, but GAMG(reach-2) DIVERGES as a preconditioner. ⚠️ The
      mechanism once given here — the ILU(1) smoother's pattern-dependent fill — **does not survive**:
      reach-2 fails at ILU(0) too, where there is no fill to be pattern-dependent about, and the surviving
      reason is that it builds a *different hierarchy* (3 levels / 480 coarse equations against 2 / 1296).
      **`column_reach` is a DIFFERENT knob — it keeps the reach-3 pattern, so the hierarchy is untouched.**
      Two things were wrong with how it was first recorded. It is not free: shortening a column **aliases**
      its far couplings onto its near entries, at 53.4 % of the entries of every shortened column on
      `bfs3d`. And `(3,3,3,2,2,2)` **diverges the march at step 1** (issue #191) — though not
      through the aliasing, whose assembled error is at the float64 floor there; the cause is sparse
      arithmetic pruning the stored zeros out of the smoother's pattern. Preserving them cures that and is
      not available, because a zero-fill incomplete factorization cannot be handed stored zeros (measured at
      zero shift: 58 restart cycles at 2.299e-02 against 11 at 8.474e-11), so they are pruned at the
      factorization boundary and the shipped value is `(3,3,3,3,2,2)` — `p` at reach 3, only k and ω
      shortened. Keep the two reaches apart when reasoning; both are in the solve rules.
      **`probe_gradient_sweeps` is a THIRD knob, and it moves the residual rather than the probe (BUILT
      2026-08-16).** The two above choose how much of the Jacobian to recover; this one bounds how far the
      Jacobian *goes*. `CorrectedGreenGauss` couples one further ring per Richardson sweep, so on a skewed
      mesh the shipped `sweeps=4` puts the coupled residual at reach **6** against a `stencil_reach` of 3 —
      and a residual reaching past the pattern aliases in **every** column, which no `column_reach` choice
      fixes and `jacobian_relative_error` cannot see. `CoupledJacobianProbe(gradient_sweeps=n)` (and the
      `probe_gradient_sweeps=n` keyword on `coupled_amg_continuation`, `amg_beta_tracking_refresh` and the
      LU builders) materializes the preconditioner from a copy of the residual whose gradient
      solve is capped at `n` sweeps — `CoupledJacobianProbe.narrow` is the one place that decides it, and
      the refresh hook re-narrows on `rebind` so a Reynolds rung's companion is capped too. The forward
      matvec keeps the exact `coupled`, so the root and the adjoint are unmoved. **`None` (default) is
      byte-identical, and this is INERT on `bfs3d`**, whose mesh is skew-free to round-off (0 of 66368
      interior faces above 1e-6 relative) — so it changes no `bfs3d` result. ⚠️ **It is NOT inert on
      pitzDaily**, which is skewed enough that reach 3 materializes a measurably wrong matrix there; and
      the cap is a **trade** rather than a free win, since narrowing makes a zero-fill elimination harder
      (pitzDaily ILU(0) negative pivots 9 → 263 as the sweeps narrow). Both are in
      `.claude/rules/solve-direct-preconditioners.md`; do not describe this knob as free. Reach/accuracy tables and the refuted "use GMRES instead" alternative are in
      `.claude/rules/schemes.md` and `.claude/rules/solve-direct-preconditioners.md`; harness
      `validation/gradient_stencil_reach.py`.
      **⚠️ THE DRIFT GATE MUST NOT BE NESTED INSIDE THE β GATE — it was, and a PC-only `beta_floor` then
      made it unreachable.** The β gate sees `max(β, beta_floor)`, so below the floor its input is pinned
      and it answers "no change" forever; asking the drift gate only inside it therefore froze the
      Jacobian through the entire low-shift tail — 91 % of sub-floor steps refreshed nothing while ν_t
      drifted ~20 % per step, and those steps carried 47 % of the march's Krylov cost. The decision is now
      the pure `_refresh_branch(stale_state, moved_beta, split)`: drift ⇒ `full` regardless of β, β move
      alone ⇒ `shift`, neither ⇒ `none`. Because the materialize gate is now consulted **every step**,
      `materialize_every` counts steps rather than refreshes. Full data in `.claude/rules/solve-amg-multigrid.md`.
      The `assemble` half of the refresh is also precomputed now (`ShiftedCellMajorOperator`), and the
      observer receives a `RefreshTiming` with per-phase costs instead of one aggregate — so "the
      materialize is ~half the refresh" is measured per run rather than inferred.
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
    - **⚠️ THE PER-RUNG RECOMPILE — THREE independent causes, all three now closed, and finding only
      two of them would have bought nothing (2026-08-12).** Each rung's first step was the most
      expensive step of that rung by a wide margin *at a cycle count no higher than its cheap ones* —
      on one march the target rung's first step cost 141 s at **7** cycles where a **19**-cycle step in
      the same rung cost 57 s. Measured per rung as the first step's excess over the median of its
      rung's same-cycle-count peers (after subtracting that step's own reported `pc ... Ns`), across
      four archived marches: rung 1 **86–90 s**, rung 2 **98–106 s**, rung 3 **112–250 s**. Rung 1's is
      the unavoidable first compile; rungs 2–3 are **~190–230 s, ~10 % of the march**.

      The compiled unit is `_march_step`, whose key is the forward step, the residual and the solver.
      **Any one difference recompiles the whole coupled solve, so this had to be closed on every axis
      at once** — which is why the earlier probe fix (above) moved only 20 %:
      1. **The preconditioner object.** It rides in a *static* field of the step and is compared by
         **identity**, so a rung that fits its own V-cycle is a new key. Fixed by reusing one object:
         `coupled_amg_continuation(preconditioner=…)` glues in an existing V-cycle instead of fitting
         one, and `amg_beta_tracking_refresh(...).rebind(companion)` points the shared refresh hook at
         the new rung and forces its next refresh to a **full** re-materialize — so the V-cycle is
         still fitted to each rung's own state and shift (the march calls `precondition_step` before a
         segment's first step), it is just not a new *object*. The adjoint factory follows for free,
         being a value object over that preconditioner.
      2. **The shift policy's unused block preconditioner.** `MonolithicFactorShiftPolicy` reads only
         `base.shift_term(phi).diagonal` and supplies its own inverse, yet `base` held a whole
         `BlockPreconditioner` — two multigrid hierarchies, built and never applied. They aggregate
         along strong connections (`strength_threshold=0.25`), which reads the operator's **values**,
         so their coarse-grid array shapes moved with the viscosity. `_monolithic_shift_source` now
         builds the policy with **no flow block at all** (`build_flow_block=False`); the shift's
         velocity buckets come from `flow.frozen_momentum_diagonal_parts(assembler, flow)`, which was
         always the whole dependency. Asking such a policy for a composed preconditioner raises.
         ⚠️ Turning the aggregation down to graph-only instead **looks** equivalent (the shift diagonal
         is bit-identical and the shapes do stabilize) and is **not safe**: with graph-only coarsening
         the hierarchy can refuse to build on a degenerate coarse row, giving a failure mode to
         something that has no consumer. Do not re-propose it.
      3. **A `float` molecular viscosity.** `Constant(RHO * NU)` is a Python float, which is not a JAX
         array, so it sits on the **static** side and is compared by value — and `Property.scaled`
         rescales exactly it, once per rung. This one is invisible: it defeats the "two leaf values"
         argument above wholesale. The case now builds `Constant(jnp.asarray(RHO * NU))`; the library
         still does not force it (`.claude/rules/properties.md` carries the rule and both halves are
         pinned).
      Also removed while sharing: the probe plan and its gather map (`CoupledJacobianProbe`, mesh-fixed
      — a three-rung march built the largest allocation the case makes **six** times, once per rung per
      consumer), the per-rung `combine_observers` closure (a static field, so a fresh one is its own
      recompile), and the per-rung engine's V-cycle fit itself.

      **MEASURED END TO END on the 3-rung `bfs3d` cold march** (field split, traced trailing inverse,
      ILU(0)×4, plain aggregation, `coarse_eq_limit` 2000, `refresh_on_cycles` 3, PC β floor 0.05,
      `retry.on_alpha` 0.01, `zerogradient` k wall, positivity floor 1e-08, forward restart 15 —
      **and `BFS3D_COLUMN_REACH=0`, a uniform reach 3** -- the case's default carried `p` at reach 2
      when this ran, which does not converge; the shipped default is now `(3,3,3,3,2,2)`, so a
      re-run needs no override and probes 454 columns rather than this run's 564):

      | | this change | archived `march-20260811-132658` |
      |---|---|---|
      | steps | 67 | 67 |
      | wall | **1883 s** | 2124 s |
      | mid-span `x_r/h` | **8.361** | 8.361 |
      | rung-3 first step | **71 s at 7 cycles** | 161 s at 7 cycles |
      | rung-2 first-step excess | **1 s** | 98 s |
      | rung-3 first-step excess | **25 s** | 112 s |
      | **rungs 2+ excess** | **26 s** | 210 s |

      Same step count, same rung-3 step count (29), same answer. The excess is the first step's wall
      minus its own reported `pc`, against the median of its rung's *same-cycle-count* peers —
      self-controlled within one march, which is what makes it quotable across runs at all
      (`validation/bfs3d_openfoam/rung_compile_cost.py`; it reproduces the archived baselines' figures
      exactly). Read the wall row with its caveat: this run probed **564** columns against the
      baseline's 399, so the 241 s is if anything an under-statement.
      **Rung 3's residual 25 s is NOT established as compilation** — its peer set is four steps, and it
      is the rung whose excess was noisiest in the baselines too (112–250 s). Per-rung work the metric
      cannot subtract (the row-scale rebuild, seeding the per-equation residual reporter) is a live
      alternative; attribute it before removing it.
      Pinned end to end by
      `test_coupled_amg.py::test_sharing_one_preconditioner_makes_a_new_rung_a_march_step_cache_hit`,
      which drives the real builder and counts residual traces through `_march_step`: a rung that
      rebuilds retraces, one that shares does not.
      ⚠️ **Method note for any future cache-key work: the process-global compilation cache contaminates
      a leave-one-out test.** Once a key is compiled a later identical key reads as a hit, so an
      arm that drops one shared field can look unnecessary purely because an earlier arm compiled it.
      Use a fresh jitted wrapper (or a uniquely-shaped stand-in) per arm.
      ⚠️⚠️ **AND BUILD THE CONTROL FROM YOUR OWN BASE, NOT FROM AN ARCHIVED LOG.** Validating this
      change against archived `bfs3d` marches showed step 1 stagnating at 47 Krylov cycles where the
      archived logs took 1, which read as a regression and cost hours of bisection. It was not: those
      logs were produced on a *different branch*, and the base being worked on carried a
      `COLUMN_REACH` with **pressure at reach 2**, which corrupts that column and poisons the operator
      the preconditioner is fitted to (since fixed on the case; `.claude/rules/solve-direct-preconditioners.md` carries it).
      Three arms — the change, the change with the old driver, and the
      **untouched base** — came out identical to the digit, which is what identified the base rather
      than the change; restoring pressure to reach 3 then reproduced the archived trajectory exactly
      (`3.896e-02, 4.115e-03, 5.275e-04, 2.851e-05` at 1/1/1/0 cycles). A march log from another branch
      is not a control, and `validation/bfs3d_openfoam/march_log_compare.py` exists to say so.
    - **✅ CARRYING THE COARSE SPACE ACROSS A REYNOLDS RUNG IS INERT — measured, and structurally so
      (2026-08-12, `validation/bfs3d_openfoam/rung_hierarchy_reuse.py`).** Reusing one preconditioner
      object across the ramp means its GAMG coarse space is built at the *anchor* and carried down two
      decades of viscosity by `refactor`'s `pc_gamg_reuse_interpolation`, which was the standing reason
      to re-test rather than assume the reuse was safe. *Configuration:* `bfs3d`, field split, ILU(0)×4,
      trailing ×1 traced, `coarse_eq_limit` 2000, column reach (3,3,3,2,2,2), shift β = 0.5 (a rung's
      first step, not the low-shift tail), one state throughout (the anchor's cold hybrid
      initialization, so the viscosity is the only variable), GMRES restart 15 to rtol 1e-8 on the
      **true** residual.

      | viscosity | carried | fresh |
      |---|---|---|
      | 100× ν (the build) | 9 cycles, 1.64e-09 | — |
      | 10× ν | 58 cycles, 1.651e-03 | 58 cycles, 1.651e-03 |
      | 1× ν | 47 cycles, 1.477e+00 | 47 cycles, 1.477e+00 |

      *(one state throughout — the anchor's cold hybrid initialization — and the column reach the case
      shipped at the time, `p` at 2 -- since fixed. See the warning below before reading any number
      here as a cost.)*

      **Identical to the digit**, which is stronger than "close" and is the tell for the mechanism: this
      bundle runs **plain aggregation** (`pc_gamg_agg_nsmooths = 0`) and sets **no** `pc_gamg_threshold`,
      so GAMG's strength graph is every nonzero — the sparsity pattern, which the coloured probe fixes
      at the mesh. The coarsening therefore never reads the operator's *values*, and a carried coarse
      space is not merely as good as a fresh one, it is **the same one**. Expect this to stop holding if
      a strength threshold is ever turned on for the monolithic V-cycle.
      ⚠️⚠️ **THE ABSOLUTE NUMBERS IN THAT TABLE ARE NOT INTERPRETABLE — only the arm-vs-arm equality is.**
      Two things spoil them, and the second was found afterwards: the anchor's cold field is not a state
      the target rung ever occupies (its real seed is a converged Re/10 root), **and** the run inherited
      the case's then-current `COLUMN_REACH` with **pressure at reach 2** (since fixed), which corrupts
      that column and poisons the operator being solved (see `.claude/rules/solve-direct-preconditioners.md`). That is why both arms fail so
      badly at the lower viscosities — at 1× ν the true residual ends *above* the initial guess.
      Neither spoiler touches the **comparison**, which holds the state, the operator and the reach fixed
      and varies only the hierarchy's provenance — and the arms agree *to the digit*, which is a stronger
      form of agreement than any operating point could manufacture. But do not quote a cycle count from
      this table for anything else, and **re-run it at reach 3 before extending it.**
    - The rebuild REUSES the aggregation coarse space (`MonolithicAmgPreconditioner.refactor` overwrites
    the operator values in place over a persistent CSR array and re-sets-up the PC with
    `pc_gamg_reuse_interpolation`), since the graph-coloured probe's sparsity is fixed across β; only the
    Galerkin coarse operators and the incomplete-LU factor values recompute, cutting the multigrid setup
    substantially with the coarse space no worse. (The setup-cost figures once quoted here were taken on
    `bfs3d` under **smoothed** aggregation, which is no longer the validated default, and with no
    smoother fill recorded — re-measure.) `coarse_eq_limit` (`pc_gamg_coarse_eq_limit`, threaded through
    the `coupled_amg_continuation` builder) sets the direct-LU coarse-grid size; raising it to 2000 is a
    large cut in the outer cycle count on the hard `bfs3d` state — for the figure with its β and bundle
    use the `coarse_eq_limit` bullet in `.claude/rules/solve-amg-multigrid.md` rather than repeating an unanchored
    number here. The experimental host-exact-solve forward path and the FGMRES-forward
      optimization remain follow-ups (`.claude/rules/solve-amg-multigrid.md`).
  - **`lu_beta_tracking_refresh` — re-factor the LU at the current β EVERY step (the correct LU treatment
    for a dual-time march; BUILT).** A frozen LU is exact only for the β it was factored at; a dual-time
    march's β ramps (0.5 → 0.005), so a factorization frozen at `lu_beta` mis-preconditions the operator
    actually solved — measured: frozen@0.05 needs 25/111/217/**474** GMRES iters at β=0.1/0.5/1/2 (vs
    **1** matched), and it **NaN'd** on a real cold ramp's overshot low-β state (215 cycles → failure).
    An approximate factorization can shrug off a β-mismatch at a few extra cycles; the exact LU is
    *exact-and-brittle* instead, so a frozen-and-occasionally-refreshed design does not suit it. Since the
    LU factor is cheap (~1 s), the fix is to re-factor at the current `(state, β)` **every step**:
    `lu_beta_tracking_refresh(coupled)` returns a `precondition_step(active_step, state)` (the
    `forward_march` seam, `.claude/rules/solve-march.md`) that reads β from the step's `ConstantRelaxation` (set
    by a `DualTimeControl`) and `refresh_in_place`s the LU at `J(state)+β·d(state)` — exact each step (1
    iter), robust through overshoots. Measured: `solve_coupled(continuation=coupled_lu_continuation(...),
    step_control=DualTimeControl(...), precondition_step=lu_beta_tracking_refresh(coupled))` **completes
    the cold pitzDaily Reynolds ramp** (rung0 12 + rung1 23 steps to rtol 1e-3) where the **frozen** LU
    failed at the rung-1 overshoot, cyc ≤ 18 throughout. **Forward-march only** (impure host re-factor;
    raises under `jax.grad`, same guard as the refresh/control); the finishing solve and adjoint keep the
    last frozen factorization, exact enough at the converged β → 0 root — so the coupled adjoint is
    unchanged (still use the plain `coupled_lu_continuation`, no `precondition_step`, for a differentiated
    solve). Requires a `DualTimeControl` (β must be a readable constant); raises with the fix if paired
    with the default switched-evolution schedule. Pinned in `tests/integration/test_coupled_lu.py`
    (exact-at-current-β, cold-march convergence to the block PC's root, grad-guard).
  - **RETIRED — "the k equation drift is the remaining limiter": the stall no longer reproduces (measured
    2026-07-22).** This bullet used to record that past rel ~0.09 the direct-`k` residual grew (rel 1 →
    ~5×) and re-stalled the march, and named high-Reynolds `k` stability as the open follow-up. **It
    does not happen on the current code.** Re-measured on the full ~12k-cell pitzDaily from the cold
    hybrid IC, second-order (Venkatakrishnan-limited) momentum, log-`ω`, conv+MSIMPLE: the march
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
    backwards" bullet in `.claude/notes/solve-globalization-log.md`). **A ~1.9× preconditioner refresh cannot rescue a march
    the schedule is grinding to a halt** — this reorders the priorities: fixing the β schedule (an
    α-driven pseudo-transient step control — the dual-time controls are where that direction survives)
    is ahead of calibrating the refresh.
  - **Preconditioner staleness is the SECONDARY cost, and it is coupled to the β schedule (measured).**
    Over the march the wall time per step also grows several-fold as the recirculation develops and the
    frozen scalar preconditioner degrades — the same post-separation regime where refreshing the k/ω AMGs
    cuts the outer cycle count (staleness bullet in `.claude/notes/solve-globalization-log.md`). Neither figure was recorded
    with its state or its preconditioner bundle, so re-measure before relying on either.
    Driving a refresh **from the march** is BUILT:
    `solve_coupled(refresh=RefreshPolicy(trigger=CoefficientDriftTrigger(…)))`. **The β coupling that motivated it:** a
    bolder β moves the state faster and stales the IC-frozen PC faster, so a *cost*-based trigger is
    confounded (cycles rise from β→0 **and** staleness — #19). The β-independent staleness trigger keyed
    on `‖Δν_t‖` is now **BUILT** and is the default recommendation; a `‖Δṁ‖` measure would be a second
    `drift_measure` against the same trigger, needing no new trigger. The threshold is **calibrated on an
    instrumented cold-IC pitzDaily march** (`threshold = 0.1`, firing where the cycle count has just
    doubled off its floor and the bubble has formed) — the table, and the validation that drift really
    does track cost, are in `.claude/rules/solve-march.md`. One geometry, so re-calibrate by offline replay
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

## Initialization, diagnostics, Reynolds continuation

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
    - **⚠️ AND THE SAME RULE APPLIES AT EVERY REYNOLDS-CONTINUATION HANDOVER, WHERE IT WAS BEING BROKEN
      (fixed 2026-09-09, `wall_consistent_omega` / `wall_consistent_state`, default OFF).** The rule
      above is honoured by `hybrid_initialize` at the **cold start** and was violated at every **rung
      boundary**: a rung inherits the previous rung's converged root, and `omega_wall`'s viscous branch
      `C·6ν/(β₁d²)` is **linear in ν**, so the carried value is wrong for the new rung by the viscosity
      ratio — a decade at the default ladder, the same magnitude as the original regression's "~10× at
      y+ = 100". Measured on pitzDaily's rung 2 → 3 handover: those 472 rows sit at **4.7e-16** at the
      rung they were converged at and at **mean 2.2890 / max 2.3026** at the next, against
      `ln(10) = 2.302585`; re-imposing them takes `|R0|` **1.1729e-01 → 3.0318e-02 (3.87×)** with every
      other block unchanged to four significant figures *in fixed scales*, and moves `ω` at those cells
      by a median factor of exactly **0.1000**.
      **It is a correctness fix and nothing more, which is measured rather than assumed**: marched, the
      better seed buys nothing (14 steps on the rung either way) and is not what carries an aggressive
      warm shift through the target rung. The reason to expect that is structural — the shift is exactly
      zero on a fixation row, so the first inner solve enforces it whatever `beta` is; the march repairs
      in step 1 what the seed repair does for free. Reached through
      `solve_reynolds_continuation(seed_projection=…)`; harness
      `validation/pitzdaily_openfoam/seed_repair_probe.py`.
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

- **`diagnostics.py` — the march log's per-equation view (BUILT).** Three exports over one set of
  names, so the two grids under a step row join instead of drifting:
  - `coupled_equation_names(dim)` → `(u, v, w, p, k, omega)` (`(u, v, p, k, omega)` in 2D) — the flat
    layout's block order, and the **single home** for these names. Raises above `dim = 3`.
  - `coupled_fields(coupled)` → named **physical** fields for `field_change_metrics` (each solved
    scalar mapped back through its variable transform, so the log reads the same whether ω is solved
    directly or in log form). Velocity is **split per component** so each lines up with its own
    momentum equation; `p` is gauge-free; `ν_t` rides along (derived, not solved — it is what the
    momentum equations actually see, and it moves when k and ω move in ways their own norms hide).
  - `coupled_residuals(coupled, continuation, reference_state=None)` → the **per-equation residual** on
    the march's own measure, `coupled_scaled_norm(...).per_block(coupled.residual(state))`. Reads
    `continuation` **late** for its `shift_policy`, so a refreshed segment's rebuilt diagonals are the
    ones used — the same late-read `norm_builder` does. Under a per-rung setup (`point_setup`) the case
    *and* its continuation are both rebuilt, so the reporter must be rebuilt with them (the bfs3d driver
    keeps the current rung's in a list the logger's callable defers to).
  ⚠️ **Equilibrate at the PREVIOUS state, never the logged one** — this is what makes the per-equation
  rows add up to the `R` reported beside them (pinned `rel=1e-12`). `forward_march` re-derives the
  measure at the state each outer iteration *starts* from and holds it for the whole iteration, so a
  step's residual is `norm_at_start(R(state_at_end))`; scaling at the end state measures the right
  residual in the wrong scales. Consequences: the reporter is **stateful and order-dependent** (once per
  step, in order — `field_change_metrics`'s contract), and a reporter built partway through a march
  **must be seeded** with that segment's `seed_state` or its first step is scaled at its own end state.
- **`ViscosityRampHomotopy` — the SAME Reynolds span walked inside ONE march (`reynolds.py`, BUILT
  2026-09-09).** The `ResidualHomotopy` alternative to the rung ladder below: `stations` geometric
  viscosity stations from `anchor` down to exactly `1.0`, each held for `steps_per_station` outer steps,
  handed to `solve_coupled(homotopy=…)`. What it removes is measured in
  `.claude/rules/solve-march.md`'s `ResidualHomotopy` bullet — read the numbers there rather than
  restating them: in one sentence, the ladder converges every seed rung to a bar the next rung's
  viscosity jump undoes by three to four orders of magnitude, and restarts the pseudo-timestep ramp at
  every rung.
  - **The target station is the CALLER'S OWN assembler, by identity, not `with_scaled_molecular_viscosity(1.0)`.**
    Numerically the same problem; **not** the same object, and the assembler is the differentiable
    parameter pytree the adjoint is taken with respect to. The homotopy dissolves at the target exactly
    as the ladder does, so the root and its adjoint belong to the case. Pinned by an `is` assertion.
  - **`rebind` is what keeps the preconditioner honest, and it is per station CHANGE.** The hook is
    `amg_beta_tracking_refresh(...).rebind`, which discards the previous companion's probe and forces a
    full re-materialize; the ramp calls it only when the station index moves, which is the whole reason
    a station spans several steps.
  - **It is a plain mutable object, not an `equinox.Module`** — it caches the current station and
    re-points the refresh as a side effect. Like the refresh hook it drives, it runs only on the eager
    forward-only march and must never be on a differentiated path.
  - **⚠️ `redamping` (default 2.0) is NOT a tuning knob — the ramp walks into a wall without it.** The
    shift control divides β by `grow` every step, so an unopposed station costs `grow ** steps` (3.375
    at the defaults) and four of them take β 0.5 → 0.013, onto this case's measured wall at β ≈ 0.012.
    See the `redamp` bullet in `.claude/rules/solve-march.md` for the failure and the four-arm
    measurement; the short version is 40 steps / 261 cycles with it against 50 / 301 without.
  - **⚠️ IT IS THE DEFAULT on `validation/pitzdaily_openfoam/compare.py` since 2026-09-10** (`PITZ_RAMP`,
    with `PITZ_RAMP=off` returning to `solve_reynolds_continuation` as the comparison arm rather than as
    a supported path). Flipped on the user's decision that the coupled march's upcoming work lands here
    and the ladder is not being developed further. **⚠️ The ramp has been measured on THIS CASE ONLY and
    on one run per arm** — `bfs3d_openfoam` has never run it, and that case is where the ladder's rung
    structure was originally calibrated, so it is the one most likely to disagree. The defaults are
    configured with
    `PITZ_RAMP_STATIONS` / `PITZ_RAMP_STEPS`. It anchors at `RATIO ** N_POINTS`, i.e. **the same span the
    ladder walks**, so the two arms differ in how the span is traversed and not in how far — and it
    builds its engine by calling the ladder arm's own `point_setup`, so they are preconditioned, logged
    and step-controlled identically. A second builder written beside it would drift a keyword at a time.

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
    (binding, #278, 2026-08-20).** A pre-built `continuation` / `reference_state` is frozen at the
    *target* viscosity, so it reaches the **final** solve only and each lower-Re point builds its own
    (`ramp_kwargs`, long-standing). The mirror image is that everything which *configures* a build —
    `method`, and every keyword bound for `coupled_continuation` — reaches the **ramp** only, because
    the target is not building one when the caller supplied it (`target_kwargs`, keyed on `_SOLVE_ONLY`,
    derived from `solve_coupled`'s own signature so the two cannot drift). **Passing both at once is the
    ordinary case, not a mistake.** Only the ramp half existed, so such a call reached `solve_coupled`
    with a continuation *and* the settings for one — ignored in silence there until #278 made it a
    `TypeError`, at which point the missing half surfaced as a failing adjoint test.
  - **⚠️ A forwarding wrapper hides call sites from an audit scoped to the callee.** An AST scan of
    `solve_coupled(` sites cleared this change and missed this one entirely, because the offending call
    is spelled `solve_reynolds_continuation(...)` and the keywords travel through `**solve_kwargs`.
    When you change a contract on `solve_coupled`, scan its wrappers too — `solve_reynolds_continuation`
    and `solve_coupled_mass_flow` are the ones that exist.
  - **HOW MANY rungs: `bfs3d` keeps TWO, but the Re/100 anchor is a closer call than it looks
    (measured 2026-08-09).** Configuration for both numbers: field-split AMG preconditioner,
    `refresh_on_cycles=3`, `retry.on_alpha` 0.01, ILU(0)×4, plain aggregation, `coarse_eq_limit` 2000,
    cold hybrid initialization, `Re_h` 10000.

    | | 2 rungs | 1 rung |
    |---|---|---|
    | wall | **1959 s** | 2007 s |
    | steps | 58 | **43** |
    | Krylov cycles | 277 | **264** |
    | mid-span `x_r/h` | 8.361 | 8.361 |

    - **The anchor is NOT needed for reachability.** The cold initialization converges at **Re/10** to
      the same root, in two independent runs. Only `n_points = 0` fails. Do not justify the anchor on
      reachability grounds — that was the standing assumption and it is false at this Reynolds number.
    - **As a route to Re/10 it costs more than it saves:** a converged Re/10 costs **800 s from cold**
      against **1027 s** via the anchor (359 s of anchor plus 668 s of a better-seeded Re/10) — ~227 s
      of net cost that the improved seed does not return.
    - **Yet the total still favours two**, because the one-rung ladder repays that saving with interest
      in the **target rung**: 932 s against **1207 s**, the two ladders arriving at the low-β wall in
      different states and needing one β escalation against three.
    - **Neither margin decides it.** Both are ~2% on single runs, the target-rung spread (±275 s) is
      larger than the ladder effect it would have to be smaller than, and an earlier pair of runs
      (before the α-collapse escalation existed) ordered the totals the *other* way by a similar margin.
      Two rungs is kept because the measured total favours it and nothing measured argues for moving —
      not because the anchor was shown to be necessary.
    - **Method note worth carrying to the next ladder question:** two attempts at this measurement were
      confounded by the same thing, the second time even after the first confound was fixed. A
      single-event cost (one stall, one escalation cascade) that varies by arm will swamp a structural
      difference of a few per cent, and equalizing the *response* to that event does not equalize its
      *cost* — the arms still meet it in different states. Compare **per-rung** costs, which reproduced
      to within 2 s across runs, rather than totals, which flipped sign.
  - **The continuation DISSOLVES at the target (binding).** The final solve runs at the case's true
    viscosity on the **live** `coupled`, so the root and its exact IFT adjoint are identical to a direct
    `solve_coupled` — the continuation changes only the path. Pinned: `n_points=2` reaches the direct
    solve's fields to `1e-6`, and `n_points=0` is bit-identical to a direct solve
    (`tests/integration/test_reynolds_continuation.py`).
  - **Intermediate points converge LOOSELY (`intermediate_rtol=1e-2` default).** A lower-Re point is
    only an initial guess for the next Reynolds number, so converging it to the target `rtol` is wasted
    work — it overrides `rtol` for the lower-Re solves only (the target keeps the caller's `rtol`);
    `None` disables the loosening. Measured necessary on pitzDaily: the wall-resolved 12k-cell mesh is
    stiff *independent of Re* (Re continuation removes the convective nonlinearity, not the mesh-induced
    linear stiffness), so an anchor converged to `1e-6` grinds for many iterations for no benefit to the
    seed. Pinned by a monkeypatched-`solve_coupled` unit test that the lower-Re points receive
    `intermediate_rtol` and the target receives `rtol`.
  - **`point_setup` — a PER-POINT continuation/precondition seam for a per-companion, per-state
    preconditioner (BUILT).** The single `continuation`/`reference_state` is dropped for the ramp rungs
    (target-specific), so it cannot express a preconditioner that must be rebuilt at *each* rung's own
    viscosity **and** seed state — chiefly the complete-LU β-tracking hook (`coupled_lu_continuation`
    frozen at the point's `(state, β)` + `lu_beta_tracking_refresh` closing over the point's residual),
    which is what lets the aggressive Courant control reach the developed pitzDaily root (the block PC
    stalls/diverges at the overshoot — see the pitzDaily case). `point_setup(companion, seed_state, point) ->
    dict` is called for **every** point (lower-Re and target) with that point's companion, its **packed
    seed coupled state**, and a `ReynoldsPoint` (1-based `index`, `total`, `viscosity_scale`, plus
    `is_target` / `label`) telling it where in the ramp it is — so a per-point builder never counts its
    own invocations to recover the loop's index, and can reach the total and the scaling at all, and its keys are merged over `solve_kwargs` (overriding any `continuation`/
    `reference_state`). To give the built continuation the state the solve begins from, the loop
    **materializes the lowest point's seed** (`hybrid_initialize`) when `point_setup` is set, rather than
    letting `solve_coupled` self-start internally. **Forward-only** (the `precondition_step` it returns
    raises under `jax.grad`), so leave it `None` when differentiating. **`None` (default) is
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
  - **`intermediate_atol` — stop every rung at one PHYSICAL standard (BUILT), and prefer it over the
    relative bar on a row-scaled measure.** `intermediate_rtol` sets each rung's bar as a fraction of
    *that rung's own* starting residual, and under continuation every rung **re-bases `‖R₀‖`** — so a
    later rung, starting from a better seed, stops at a *looser absolute* residual than an earlier one
    already achieved. Worse, the row-scaled measure is already a fractional per-equation change, so
    dividing it again by `‖R₀‖` makes the bar a property of the initial guess rather than of the physics.
    `intermediate_atol` (with `rtol=0`) gives every rung the same absolute bar.
    **Measured on the 3D backward-facing step**, where the case metric is the reattachment length:
    under the relative bar, rung 1 stopped at `‖R‖ = 6.3e-3` with the bubble **still moving**; under an
    absolute `1e-4` it ran 5 more steps to `6.3e-5` and the bubble was **bit-identical for the last five
    steps**. Five steps bought a seed that was actually converged rather than a snapshot of a moving field.
    (The whole-ramp step / cycle / wall totals once recorded here named no preconditioner bundle, and were
    taken on a *three*-rung ladder where the measured `bfs3d` ladder choice above keeps **two** — so they are
    deleted rather than carried. Re-measure on the current bundle if a ramp total is needed.)
  - **The nonlinearity, not the shift, is what makes a step expensive — convergence is abrupt once the
    Newton basin is reached.** Observed on every rung of the 3D march: the last few steps take the last
    ~20× of the residual drop for a few per cent of the rung's total cycles, and the basin announces
    itself by **`α → 1.0` and the cycle count collapsing at the same time** — even though `β` is
    *smaller* there, i.e. the operator is nearer the singular limit that used to break the preconditioner.
    Near the root the Jacobian is what an algebraic-multigrid V-cycle is good at; far from it the
    nonlinearity is what makes the shifted operator hard. Practical consequence: **budget a march by how
    long it takes to *reach* the basin**, not by the residual level it must ultimately hit, and do not
    read a rising cycle count near the start as a preconditioner failure.
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
    - **⚠️ A TANGENT (EULER) PREDICTOR BETWEEN RUNGS IS COUNTERPRODUCTIVE AT THE DEFAULT DECADE, AND ONLY
      PAYS ON A MUCH FINER LADDER (measured 2026-09-08, `validation/continuation_seed_error.py`).**
      The seed handed to the next rung is the previous rung's converged fields unchanged, wrong by first
      order in the continuation parameter, so the obvious repair is to carry it along the solution path:
      parameterize by `lam = ln(scale)` (constant `dlam = -ln(ratio)` on a geometric schedule), solve
      `J v = -dR/dlam` at the converged state, and hand over `u* + dlam v`. **The benefit is boundable
      without building any of it**, which is what makes this cheap to settle: expanding the residual about
      `u*` gives `R(u* + dlam v, lam+dlam) ~= R(u*, lam+dlam) - dlam dR/dlam`, so the predictor removes
      *exactly* the term `dlam dR/dlam` and nothing else — one parameter-direction JVP, no Jacobian and no
      linear solve. `dR/dlam` is exact by AD (`with_scaled_molecular_viscosity` scales live leaves; matched
      a central difference to **7.4e-10** relative), so no difference quotient is needed.

      *Configuration:* `bfs3d` (23040 cells), converged target root at `lam = 0` (checkpoints 69 and 62 of
      `march-20260825-213445.log`, agreeing to four figures), scored in the row-equilibrated
      `coupled_scaled_norm` the march stops on, rebuilt at each companion viscosity. Ratio of the
      tangent's residual to the plain seed's, per rung spacing:

      | spacing | `dlam` | onward (the direction of travel) | back along the arc |
      |---|---|---|---|
      | **10 (the default)** | 2.303 | **1.23 — 23 % WORSE** | 0.52 |
      | 5 | 1.609 | 0.90 | 0.45 |
      | 3.16 | 1.151 | 0.71 | 0.38 |
      | 2 | 0.693 | 0.50 | 0.31 |
      | 1.5 | 0.405 | 0.29 | 0.21 |
      | 1.2 | 0.182 | 0.09 | 0.10 |

      **At a decade the predictor makes the seed worse**, because `dlam = 2.303` is not a small number and
      the second-order term it cannot cancel exceeds the first-order term it does. Its advantage is
      `O(dlam^2)` against the plain seed's `O(dlam)`, so it only appears once the rungs are packed closely
      — which makes "add a predictor" and "use a finer ladder" **one change, not two**. Read the ratio
      column only: the absolute residuals are not comparable across rows, because the row scales are
      rebuilt at each companion's viscosity; the ratio uses one measure for both of its terms and is clean.
      ⚠️ **Direction matters and a one-ended probe cannot settle the decade case.** A converged checkpoint
      sits at the *target*, so positive `dlam` walks back up the arc rather than along the direction of
      travel; the two signs differ by 2.4x at a decade and agree by ratio 2. The direction of travel gives
      the pessimistic answer. ⚠️ **Measured at the converged target root only** — both checkpoints are that
      same root, so their agreement shows the instrument is deterministic, not that the path curvature is
      the same at lower Re. The back-along-the-arc column is the less curved one at equal `|dlam|`, which
      hints the early rungs may do better than the target rung; untested.
    - **⚠️ AND A FINER LADDER IS NOT CURRENTLY AFFORDABLE, BECAUSE beta RESTARTS AT EVERY RUNG.** Each rung
      is its own `solve_coupled`, and `control_state` is initialized to `None` *inside* it, threaded across
      the refresh segments, and discarded at the return — so every rung re-seeds `beta_start` and re-walks
      the shift down. At the cases' `CflResidualDualTimeControl(beta_start=0.5, beta_min=0.005, grow=1.5)`
      that is `ceil(ln(100)/ln(1.5))` = **12 steps of pure descent per rung**, independent of seed quality;
      rung 1 of the march below hit exactly 12 with zero backoffs, and all three rungs start at 0.5000.
      Spanning `Re/100` to the target at ratio 1.5 needs 12 rungs, so 144+ steps against the present 69.
      **Do not read this as "carry beta across rungs" being the missing plumbing.** The carry across
      *refresh* boundaries is built and shipped (the sawtooth fix, worth ~4x: a pitzDaily rung went ~75
      steps to ~22 at `beta_min` 0.02); a *cross-step* carry of the escalated beta was built and
      **rejected** at ~24 % slower, from damping inertia. The rung-boundary carry is untried, and what is
      recorded points against it: on pitzDaily a constant `beta = 0.05` (a jump straight to a big
      pseudo-timestep) went non-finite at step 2 — "the gradual ramp is essential" — and at
      `beta_min = 0.005` a 10x Reynolds jump from a loose seed overshot the bubble in one step (`x_r/h`
      3.7 to 8.7, relaxing back) at a recovery cost of several expensive steps. Carrying beta into a new
      rung *is* that configuration. **But every one of those measurements was taken at a 10x jump**, and
      the fine ladder's whole premise is that the jump is small, so they do not transfer by themselves.
      The safer lever is to make `beta_start` a function of the rung spacing rather than carrying beta
      across a discontinuity — the control can still back off, where a carry starts past the point of no
      return — and `PITZ_BETA_START` is on record as **untried in the low direction** (see
      `.claude/rules/solve-amg-multigrid.md`, where `beta_start = 4` measured worse than `0.5`).
    - **✅ A LOWER STARTING SHIFT ON THE *WARM* RUNGS IS A REAL WIN AT 0.1 AND CATASTROPHIC AT 0.05 — and
      the cliff is PER RUNG, not per case (measured 2026-09-08, pitzDaily).** `beta_start` was one constant
      serving two unlike situations: the lowest rung self-starts from `hybrid_initialize` and genuinely
      needs heavy damping, while every rung above it is handed a *converged root* one Reynolds step below.
      Splitting them (`PITZ_BETA_START_WARM`, applied through the existing `point_setup` seam keyed on
      `ReynoldsPoint.index`, so no library change) and sweeping the warm value:

      | `beta_start` warm | rung 1 `Re/100` | rung 2 `Re/10` | rung 3 target | total steps | cycles | wall | esc |
      |---|---|---|---|---|---|---|---|
      | **0.5** (the single-constant baseline) | 28 | 19 | 17 | 64 | 421 | 750 s | 1 |
      | **0.1** | 28 | **17** | **14** | **59** | **403** | **712 s** | **0** |
      | 0.05 | 28 | **14** | **64, killed** | 106+ | 327 | 824+ | **9** |

      *Configuration:* pitzDaily 12225 cells, `N_POINTS=2`, `CflResidualDualTimeControl(beta_min=0.005,
      grow=1.5, backoff=2.0, grow_above=0.5, backoff_below=0.25)`, stop `(rtol, atol) = (0.0, 1e-5)`,
      `MAX_STEPS` 150/rung, compiled ILU(0) live, all three arms run back to back on one machine state.
      ⚠️ **Swept corrected Green–Gauss at 4 sweeps and probe reach 5 — the case default WHEN THIS RAN,
      since moved to `MultipleCorrectionGradient` at reach 3, which is 32 % faster on the same root.**
      The arms are still a valid comparison with each other; their absolute walls are not comparable to
      anything measured after the move.

      **0.1 is a clean win and costs nothing in robustness:** −8 % steps, −4 % cycles, −5 % wall, the same
      root (`x_r/h` 8.0686, `nut` peak within 0.04 %), and it *removed* the baseline's one escalation
      rather than adding any. Rung 1 is bit-identical across all three arms (28 steps / 130 cycles) and the
      trajectories part exactly at the first warm step, so this is a controlled experiment rather than a
      pair of runs.

      **0.05 gives the sharpest result in the sweep: it is the BEST arm on rung 2 (14 steps against the
      baseline's 19) and a catastrophe on rung 3.** The target rung's *first* step at 0.05 escalated, the
      march then diverged to `‖R‖ = 2.2e+15`, ground back down ~150x per step at zero Krylov cycles, and
      stalled with the line search collapsed to `alpha = 0.000` while the retry ladder drove beta to
      **16.0** — 320x its start and 32x the baseline's. Killed at step 106 (818 s, nine escalations) three
      orders from the stopping bar. ⚠️ **Its cycle total (327) is LOWER than the baseline's 421, because
      the diverged steps cost no cycles at all — cycles are meaningless for a failed arm; read steps and
      the escalation count.**

      **What this settles, and it is not "0.1 is the right number".** A single warm constant is still the
      wrong *shape*: the value that was best on rung 2 destroyed rung 3, so what the shift has to track is
      **how hard this particular rung is**, which rises with Reynolds number — not whether the rung is warm
      or cold. The cold/warm split is strictly better than one constant and strictly worse than a rule. It
      also locates the safety boundary precisely: at 0.1 the one clipped step (`alpha = 0.125`) was caught
      **inside the step control**, which backed beta off x2 and carried on; at 0.05 the clip escaped into
      the **retry ladder** and from there into divergence. That is the line a rule has to stay on the right
      side of, and it is a property of the rung, not of the case.
    - **❌ PATH GEOMETRY DOES NOT PREDICT WHICH RUNG IS DANGEROUS — the curvature measure ranks the one
      that blew up as the SAFEST of the three (measured 2026-09-08, `validation/continuation_seed_error.py`).**
      The natural way to choose a rung spacing rather than fixing it at a decade is to ask how far the
      solution path stays linear over the step: with `E0(dlam) = ‖R(u*, lam0+dlam)‖` and
      `E1(dlam) = ‖R(u*, lam0+dlam) - dlam dR/dlam‖`, the ratio `tau = E1/E0` is the dimensionless
      fraction of the step the first-order model fails to explain, and `tau -> 1` marks the point where
      neglected curvature matches the entire linear term. A schedule holding `tau` constant instead of
      `dlam` constant would then take large steps where the path is straight and small ones where it
      bends. **It does not work.** Measured at each rung's own converged root on pitzDaily (`N_POINTS=2`,
      the run reproducing 28/19/17 steps and 421 cycles exactly, under the swept gradient at reach 5
      that was the default then), for the shipped decade step:

      | converged root | `lam0` | `E0` | `tau` | what the next rung actually did |
      |---|---|---|---|---|
      | rung 1 (`Re/100`) | `ln 100` | 1.663e-1 | **0.599** | 19 steps; **survived** `beta_start` 0.05 (14 steps, 0 escalations) |
      | rung 2 (`Re/10`) | `ln 10` | 1.175e-1 | **0.278** | 17 steps; **diverged** at `beta_start` 0.05 to `‖R‖ = 2.2e+15` |
      | target | `0` | 2.668e-2 | 1.105 | — (a step *past* the target; matches `bfs3d`'s 1.23) |

      `tau` calls the step into the target **twice as safe** as the step into `Re/10`, and it is the one
      that catastrophically fails. `E0` ranks them the same wrong way (the dangerous step starts at the
      *lower* residual). So neither the seed error nor the path curvature sees the danger, and the reason
      is structural rather than a bad threshold: **both are parameter-space quantities, and what makes a
      high-Reynolds rung dangerous is state-space stiffness** — how far the shifted Newton step stays
      trustworthy — which no amount of information about how `R` varies with `lam` can reach.

      **What the probe IS validated for: `E0` predicts the next rung's starting residual exactly**, before
      running it — 1.6629e-1 against the log's `reference |R0| = 1.6629e-1`, and 1.1745e-1 against
      1.1745e-1, five figures each. That makes it a sound way to *equalize* seed quality across a ladder,
      which is a different job from predicting difficulty, and it costs nothing: a residual evaluation is
      **3.7 ms against 14.5 s for a march step** on this case, so a six-point search over `dlam` is 0.15 %
      of one step.

      ⚠️ **This is now the THIRD refuted predictor of a bad continuation step**, after the sibling case's
      step-0 diagnostic (every static signal) and an `alpha`-trend rule tried here (growth taken on a
      falling `alpha` 9 times with 0 bad outcomes — 9 false positives for the 1 event it would have
      caught). Three independent attempts from three different signal families all fail, which is strong
      support for the detect-then-react design the sibling case already settled on. **Adapt the spacing
      from the previous rung's observed cost; do not predict it from the path.**
    - **❌ A LOCAL SMOOTHER DOES NOT REMOVE THE LADDER — no local direction makes real progress at the
      cold start (measured 2026-09-08, `validation/point_implicit_step.py`).** The hypothesis was
      attractive: the recorded cold-start failure is "no `beta` makes step 1 descend", which is a
      statement about the *shifted Newton family* `-(J + beta D)^-1 R` only. Mavriplis (Computers and
      Fluids 220:104859, 2021) makes the small-pseudo-timestep limit `-D^-1 R` for a local operator `D`
      instead of an explicit step, on the observation that local nonlinear solvers converge where PTC
      stagnates. If that direction reached the target root, the whole ladder — 64 of pitzDaily's steps,
      against 17 in the target rung — would be unnecessary.

      *Configuration:* pitzDaily 12225 cells, cold `hybrid_initialize` at the **true target** viscosity,
      scored in `coupled_scaled_norm`, step length swept over 15 decades. Best `‖R‖/‖R0‖` reached:

      | direction | `‖step‖/‖u‖` | best ratio | iterated x5 |
      |---|---|---|---|
      | explicit `-R/V` (the PTC small-shift limit) | 7.8e+05 | 1.0000 (none) | — |
      | row-scaled `-R/d` on the shift diagonal | 1.0e-02 | 0.9974 | — |
      | point-implicit `-D^-1 R`, per-cell block | 1.2e-01 | 0.9991 | **non-finite** |
      | strength-of-connection aggregate blocks | 2.1e-01 | 1.0000 (none) | **non-finite** |

      **The best any of them achieves is a 0.26 % reduction**, and the iterated form — five damped
      nonlinear sweeps, which is the shape of the actual smoothing term — goes non-finite at every
      damping tried. Descent directions exist at this state; they are simply worthless.

      **The decisive number is not in that table.** The cold state at the target has `‖R‖ = 2.35e-2`,
      while the anchor rung at `Re/100` *starts* at `8.71e-2` and converges in 28 steps — **the
      unreachable state has the LOWER residual**. So this is a basin question, not a descent question,
      and it is the same shape as the two refuted predictors above: the ladder's measured contribution is
      mean-flow topology (79 % of the final reattachment length at the anchor, turbulence 90 % wrong),
      and no local relaxation manufactures a recirculation bubble.

      ⚠️ **What this does NOT refute, stated precisely.** `D` here is a *linear* block solve over Vaněk
      aggregates (median size 7); Mavriplis's is a three-stage **line**-preconditioned nonlinear RK, and
      a line through an anisotropic layer is a thin wall-normal chain, not a blob — the aggregation used
      here is the multigrid coarsening, which is built to be isotropic. And the published method is the
      **blend** `(M/tau + J) dw = -(I + (M/tau) D^-1) R`, a Newton step with a smoothed right-hand side;
      only its small-`tau` endpoint was tested. A faithful test needs a real line construction and the
      blended right-hand side, which is an implementation rather than a probe.

      ⚠️ **And a self-inflicted trap worth carrying: `_materialize_jacobian` returns the Jacobian
      FIELD-major.** The cell-major permutation is applied *around* the V-cycle, not baked in, so reading
      per-cell blocks with `tobsr(blocksize=(n_fields, n_fields))` — as the cell-block harnesses do only
      *after* permuting — groups `n_fields` consecutive **cells of one field**. Doing that here produced a
      step 600x the state norm and an apparent refutation that was pure indexing.
    - **❌ A FINER LADDER AT THE SAME SPAN LOSES ~22 % OF WALL — the decade is not obviously the wrong
      round number (measured 2026-09-08, pitzDaily).** Every earlier ladder study varied `n_points`,
      which moves the *span* (how deep the anchor sits) and the *granularity* together; this holds the
      span at `Re/100` and varies only granularity, which is the comparison the "why 10?" question
      actually needs. `PITZ_RATIO` (paired with `PITZ_N_POINTS` so the product is fixed) is the hook.

      | ladder | rungs | steps | cycles | wall | escalations | `x_r/h` |
      |---|---|---|---|---|---|---|
      | **ratio 10** (shipped) | 3 | **64** | **421** | **750 s** | 1 | 8.0686 |
      | ratio 3.1623 | 5 | 99 | 455 | 918 s | **0** | 8.0686 |

      *Configuration:* pitzDaily 12225 cells, `beta_start` 0.5 both arms, `CflResidualDualTimeControl(
      beta_min=0.005, grow=1.5, backoff=2.0, grow_above=0.5, backoff_below=0.25)`, stop `(0.0, 1e-5)`,
      compiled ILU(0) live, run back to back. Rung 1 is identical in both (28 steps / 130 cycles — the
      same anchor), so the arms differ only below it. ⚠️ **Swept gradient, probe reach 5 — the default
      when this ran, since moved; the per-rung walls do not carry across that change.**

      **The finer rungs really are individually cheaper, and it still loses.** Per-rung wall runs
      261 / 173 / 158 / 150 / 176 s against the baseline's 259 / 275 / 216 — every rung after the anchor
      is cheaper than either of the baseline's, and cycles rise only 8 % (421 → 455) because a smaller
      jump makes each step easier. What sinks it is **step count**: 99 against 64, because each rung pays
      its own `beta_start`-to-`beta_min` descent. Fitting `wall = steps x f + cycles x c` across the two
      arms gives **f ~ 3.6 s per step and c ~ 1.23 s per cycle**, so the 35 extra steps cost ~126 s of
      pure per-step overhead against ~42 s of extra Krylov — i.e. **the loss is the descents, not the
      work**.

      **What the finer ladder DOES buy is robustness: zero escalations against the baseline's one.** That
      is the axis an adaptive schedule is for, and it is not visible in the wall clock. Note also that
      this makes the warm-`beta_start` result worth more here than its own 5 % suggests: a ladder paying
      four descents instead of two has twice as much to gain from shortening them, and the two were not
      measured together.
    - **Where a 3-rung march's time actually goes (measured, `march-20260825-213445.log`).** Same bundle as
      the equilibration A/B in `.claude/rules/solve-amg-multigrid.md`: `bfs3d`, `N_POINTS=2`, field split,
      ILU(0)x4, probe reach 3/3/3/3/2/2, dual-time `inner_steps`/`inner_tol` 5/0.01, `refresh_on_cycles` 3,
      retry on cycles/alpha 10/0.01, cycle budget 42, forward restart 15, PC beta floor 0.05, stop
      `(rtol, atol) = (0.0, 1e-5)`, k wall `zerogradient`, k positivity floor 1e-8.

      | rung | steps | wall | s/step | cycles | cycles/step | s/cycle |
      |---|---|---|---|---|---|---|
      | 1 `Re/100` | 14 | 292 s | 20.9 | 41 | 2.9 | 7.1 |
      | 2 `Re/10` | 25 | 666 s | 26.6 | 121 | 4.8 | 5.5 |
      | 3 target | 30 | 1310 s | 43.7 | 203 | 6.8 | 6.5 |
      | total | **69** | **2268 s** | 32.9 | **365** | 5.3 | 6.2 |

      **The cost per cycle is flat across rungs; what grows is cycles per step** (2.9 to 6.8), i.e. the
      preconditioner weakening as the case hardens — so the target rung is 58 % of the march because it
      needs more cycles, not dearer ones. Preconditioner refresh is **466 s (21 %)** over 69 logged
      asides, of which the coloured Jacobian probe is ~320 s.
      ⚠️ **That figure was first recorded as 364 s / 16 %, and both that and `march_log_compare.py`'s own
      1 % were parser undercounts of the same shape.** A step's preconditioner aside is written
      `pc full 3.6s`, `pc none 0.0s`, `pc none inner 3.0s` **or** `pc none 2x inner 2x 6.3s` — the kinds
      *compose*, so any pattern that enumerates them (`pc (full|inner|none) Ns`) or caps the words before
      the number silently drops the compound forms, which is where the cost is. The tool now matches `pc`
      and takes the first seconds figure on the line; both totals were re-derived independently before
      this entry was changed. Steps cost about the same whatever
      beta is doing (target rung: 43.8 s/step while beta descends, 41 s/step at the floor), so **a step
      removed is ~33 s saved wherever it is removed**. The eight most expensive steps are 31.5 % of the
      march, and four consecutive ones in the target rung (52-55, spanning a beta escalation to 2.22)
      are ~20 % on their own.
      ⚠️ **`t(s)` in a march log is cumulative over the WHOLE march and does not reset at a rung
      boundary.** Differencing it per rung inflates every rung after the first — read here as 3518 s
      before it was caught, against the true 2268 s. The per-rung wall is the difference of the boundary
      values, never the last row.
    - **WHAT THE LITERATURE ACTUALLY DOES — a survey against primary sources (2026-09-08).** Five parallel
      literature threads, checked against source code (ADflow, PETSc, SU2, AUTO), manuals (FUN3D, CFL3D,
      Fluent) and the books themselves (Allgower & Georg 2003; Deuflhard 2004). Recorded because a
      finding that lives outside these files can be cited but never re-adjudicated.
      - **Reynolds-number continuation is established in two communities and essentially absent from a
        third.** It is standard infrastructure for *bifurcation* work, and routine as a Newton
        globalization for *laminar incompressible* Navier–Stokes — Farrell, Mitchell & Wechsung (SIAM J.
        Sci. Comput. 41(5):A3073, 2019, §5.1): "we employ simple continuation in Reynolds number as a
        globalization device". In *engineering aerodynamics* it is absent, and deliberately: the Toronto
        homotopy line continues in artificial **dissipation**, Bristol in **angle of attack**.
      - **⚠️ THERE IS A PUBLISHED OBJECTION TO RE SPECIFICALLY, AND THIS PROJECT MEASURED IT
        INDEPENDENTLY.** Hicken & Zingg (AIAA-2009-4139; AIAA-2011-3237, verbatim in both):
        "Reynolds-number continuation affects only the momentum and energy equations. Moreover, the
        influence of the Reynolds number is different in the momentum and energy equations … folds can
        be introduced when using Reynolds-number continuation" (citing Walker, SIAM J. Sci. Comput.
        21(3), 1999). The first clause is exactly the momentum-only measurement above: scaling both
        blocks displaces the target root's ω residual **1355×** more than scaling momentum alone. Theory
        says the regularization is non-uniform; this case says by three orders of magnitude. The **fold**
        clause names a failure mode unrelated to step size or damping, which a monotone sweep cannot
        pass — though Farrell et al. reach Re 10000 on a cavity *and a backward-facing step* with only a
        line search, so it does not appear to bite there.
      - **⚠️⚠️ ORDER CONTINUATION AND GRID SEQUENCING ARE CONTRAINDICATED FOR THIS PROJECT'S CASES, AND
        THE REASON IS SEPARATION.** Both are near-universal practice (FUN3D `first_order_iterations`,
        CFL3D `nitfo`, Fluent's first-to-higher-order blending, ADflow's first-order Jacobian, SU2's
        permanent `MUSCL_TURB=NO`), and the ramp is `stop_gradient`-ed so either would be free of the
        adjoint. **Do not reach for them here anyway.** Hemker & Koren (*Numerical Methods for Fluid
        Dynamics III*, 1988, p. 162), on a shock/boundary-layer interaction with a separation bubble:
        "the **first-order distribution typically is the distribution belonging to a non-separating
        flow** … the first-order solution has to be rejected." It does not misplace the bubble, it has
        **no bubble** — and both flagship cases are separated flows reported by reattachment length,
        with the ladder's measured contribution being mean-flow bubble development (79 % of final length
        at the anchor). Chisholm & Zingg (JCP 228(9), 2009) agree from the coupling side on a separated
        high-lift case: "a fully coupled turbulence model appears to be an important aspect of achieving
        convergence for such flows." CFL3D ships its own knob **recommended off** (`nitfol = 0`).
      - **The per-rung shift restart has named precedent on BOTH sides**, so it is a choice: ADflow's
        `ANKCFLReset` documents carrying the CFL across successive solves as an option, and PETSc's
        `TSPSEUDO` offers `-ts_pseudo_increment_dt_from_initial_dt`. The step-control literature is
        one-sided though — **every rule in it is a recursion on the previous step**: Deuflhard's
        (5.25)/(5.46) cannot be written without `Δλ_{ν−1}`, AUTO's `RDS` is `INTENT(INOUT)`,
        LOCA/MATCONT/pde2path all carry it. Two refinements: a failure-reduced step should climb back
        **geometrically, not snap back** (LOCA states this), and a carried step needs a **dead band** or
        it oscillates between the grow and shrink bands (AUTO, MATCONT, pde2path all have one;
        `AdaptiveReynoldsSchedule` does **not**, and should).
      - **⚠️ A CONTRACTION-BASED (Deuflhard / Newton–Kantorovich) STEP CONTROL IS THE WRONG FAMILY HERE,
        AND SOMEONE ELSE ALREADY PAID TO FIND OUT.** It is the theoretically founded rule — affine
        covariant, `(g(Θ̄)/g(Θ₀))^{1/p}` with `g(Θ) = √(1+4Θ) − 1`, computable from the first two
        corrector corrections. Brown & Zingg (AIAA-2013-2370, §V.E) tried it on RANS and abandoned it:
        "**the relaxation that we apply to both the linear system and nonlinear sub-problems reduces the
        effectiveness of these error models**". This case runs `forward_rtol = 0.3` — exactly that
        condition. They kept the *distance* and *angle* monitors and dropped contraction. A fourth
        independent argument for detect-then-react, alongside the three refuted predictors above.
      - **Two existing design choices are independently corroborated.** Yildirim, Kenway, Mader &
        Martins (JCP 397:108741, 2019) use `θ_phys = 0.99` on the turbulence variable, checking **only
        negative-direction updates** — the same rule and constant as `positive_k_limit`. And Hicken et
        al. (2011) §II.C recommend "equation scaling … that ensures the 2-norm of the residuals are the
        same order of magnitude", which is `coupled_scaled_norm` arrived at separately.
      - **Grid sequencing is not the prize it looks like.** CFL3D's Table 7-1 gives multigrid ~10× but
        full multigrid only **1.33× on top in 2D and nothing in 3D**. Yildirim et al.'s 1172-case study
        finds deepening a multigrid startup 3→5 levels worth 2.9×, yet a well-globalized
        approximate-Newton solver **on the finest grid alone** matches the best of them. Wackers & Koren
        (JCP 226(2), 2007) find nonlinear multigrid "does not work for the RANS equations" — the
        turbulence source is a small difference of large terms and "the model needs a minimum grid
        resolution … typically about 20 cells over the thickness of a boundary layer", the same shape of
        objection as this case's wall-function-off measurement one level down.
      - **Continuation in a turbulence-model parameter is an open problem**, flagged as future work by
        both groups best placed to do it (Hicken et al. 2011; Brown & Zingg 2013 conclusions). Anything
        built there is novel work, not adapted practice.
      - ⚠️ **Not verified, and not to be cited as if it were:** Seydel's step-control formula (which both
        Allgower & Georg's lineage and Tuckerman & Barkley defer to); Keller on step selection; Carey &
        Krishnan (1985), the single classical citation the whole Toronto line uses for
        Re-as-globalization; and Xu et al. (2023), doi:10.3390/aerospace10030230, likely the best single
        survey, which returned 403 everywhere. Pueyo & Zingg's numbers come from the thesis, not the
        journal paper. The LOCA report's printed Eq. (2.8) contradicts its own shipped code — use the code.
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
    contract as `solve_coupled`: to differentiate, pass a target-viscosity `continuation` built outside
    `jax.grad` (used by the final solve only — `continuation`/`reference_state` are dropped from the ramp
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
