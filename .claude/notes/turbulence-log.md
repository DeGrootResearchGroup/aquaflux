# Investigation log — `aquaflux/turbulence/`

> ⚠️ **`coupled_continuation`, `coupled_lu_continuation`, `coupled_amg_continuation`, `lu_beta_tracking_refresh` and `amg_beta_tracking_refresh` no longer exist (deleted 2026-09-14, #371; the complete LU and the PETSc V-cycle themselves 2026-10-10/11).** Entries below that name them are dated history; read them through the rename table in `.claude/rules/turbulence.md`.

> Split out of `.claude/rules/turbulence.md` (2026-10-17). **Lives in `.claude/notes/`, outside the
> auto-loaded `.claude/rules/` tree, so it never auto-loads.** It holds the dated investigation behind
> the k–ω SST closure, the near-wall treatment, the coupled solve and the Reynolds continuation —
> measurements with their configurations, probe results, and findings a later measurement qualified or
> retracted — kept because a wrong-but-plausible finding gets re-derived if the trail that refuted it is
> deleted rather than archived. `turbulence.md` holds the current, load-bearing status and points here
> by heading. Sections mirror the `##` sections of `turbulence.md` the entries came from; entries are
> verbatim, so their indentation is that of the bullet they were nested in.

## The closure — model, strain, sources, transport, preconditioner

### `S` makes the velocity columns of the coupled Jacobian reach one ring further (2026-08-11)

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

### `sources.py` — the `k` row's Jacobian diagonal is negative at near-wall cells (2026-08-24)

  - **⚠️⚠️ THE `k` ROW'S JACOBIAN DIAGONAL IS NEGATIVE AT NEAR-WALL CELLS, AND ONLY THE PSEUDO-TIME
    SHIFT KEEPS IT POSITIVE — measured 2026-08-24 on pitzDaily.** `nu_t = a_1 k / max(a_1 omega, S F_2)`
    is proportional to `k`, so `P_k` is proportional to `k`, and subtracting a source that grows with
    the variable puts a **negative** term on that row's diagonal. Measured at a wall cell of the
    pitzDaily target rung: `J_kk` between **-1.22e-03 and -1.95e-03** against a shift `beta d_k` of
    **+2.31e-03** at the case's `beta_start = 0.5` — so the effective diagonal keeps only **16-47 %**
    of the shift, and is a difference of two similar numbers. ⚠️ **That was at `turbulence_damping` 1;
    the shipped case runs 3, which triples the shift on this row and removes the near-cancellation**
    (measured 2026-10-12, the #312 entry below) — the numbers in this entry describe an undamped march.
    - **The consequence is an amplifier on everything upstream.** Four gradient reconstructions at that
      one cell differ by **12 %** in `|grad u|` and in `P_k`, and by **30×** in the `k` correction the
      Newton step asks for, purely because a 12 % move in `J_kk` is a 3× move in the near-cancelling
      sum. The `k` correction then reaches **+4000×** the local `k`, in the increasing direction that
      the k positivity guard (then the global cap, now the per-cell clip) does not touch — it bounds only
      entries that could cross zero; the ω step bound added 2026-10-14 now limits what reaches `omega`. The
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

### `explicit_production_limiter` — measured inert, and the cap's transient binding on `bfs3d` (2026-08-15)

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

### `open_session` — dry run of both flagship drivers against the pre-migration drivers (2026-09-14)

    - **Both flagship drivers were dry-run to their first step against the pre-migration drivers
      (2026-09-14), and match line for line.** Control: this branch's own commit `4eea966` (S3b), whose
      drivers still called the deleted builders — not an archived log, which is another branch's run.
      Migrated: `c4d9cb1`. Each case run through `run_case.sh` on its shipped defaults, one at a time,
      stopped at step 2. **pitzDaily** (12225 cells, ramp 16x1, momentum-only, damping 3, field split
      `simplesmooth`/`jacobi_smoothed`): banner identical; step 1 `G` 3.596e-02 → 6.796e-03 → 1.411e-04
      at 1+1 cycles, `alpha` 1.000, `|R0|` 2.3038e-02, `R` 3.331e-02, `pc full 1.2s` in both.
      **bfs3d** (23040 cells, same bundle): step 1 `G` 1.911e-01 → 4.840e-02 → 9.770e-04 at 1+1 cycles,
      `|R0|` 9.9215e-02, `R` 1.511e-01, per-field residuals and the SIMPLE-smoothed hierarchy prints
      identical. Only wall clocks differ (35/36 s and 76/74 s). One step is what this establishes — the
      refresh hook, the ramp's per-station `rebind` and the target station are not reached by it.

### #701 — the arms (2026-10-14)

      **The arms.** pitzDaily: `validation/pitzdaily_openfoam/compare.py`, 12 x 1 momentum-only ramp from
      anchor 100, damping 3, `grow` 3, release 1e-4, `beta_min` 0.005, dual-time 5 / 0.01,
      `refresh_on_cycles` 3, residual stop, field split `SimpleSmoothed`/`JacobiSmoothed`, `atol` 1e-5 /
      `rtol` 0; Linux x86_64, 4 cores, `run_case.sh --force` (its free-memory reading is macOS-only); one
      run each; every arm `x_r/h` 8.0686. bfs3d: `validation/bfs3d_openfoam/compare.py`, its `case.yaml`
      as it was BEFORE #712 (`CorrectedGreenGauss`, `lineax` stop, `refresh_on_cycles` 3, 12 x 1 ramp,
      `grow` 1.5, no release, refit floor 0.05, field split),
      no-floor arms by the owner's local `BFS3D_REFIT_BETA_FLOOR` override; Apple M3 Pro, 11 cores, jax
      0.10.2, owner's runs (PR #710 comments); every arm `x_r/h` 8.361, converged states within relative
      L2 2.3e-4 of main's. Steps / restart cycles / escalating retries:

      | arm | what moves | pitzDaily, floor 0.05 | pitzDaily, no floor | bfs3d, floor 0.05 | bfs3d, no floor |
      |---|---|---|---|---|---|
      | main (`78e0bbf` / `159536c`) | nothing | 20 / 154 / 0 | 19 / 144 / 0 | 36 / 166 / 0 | 33 / 155 / 0 |
      | station's shift each station, built at the iterate | D, measure, nu_t/mdot/omega factor | 28 / 157 / 6 | -- | -- | -- |
      | ... built at the held steps' state (E1) | D, measure | 22 / 129 / 2 | -- | -- | -- |
      | ... + measure rebuilt after the graft (E1 + E4) | D, measure | 25 / 145 / 0 | 18 / 136 / 0 | 40 / 178 / 2 | 43 / 195 / 2 |
      | anchor's shift, target's once on arrival (E5) | D, measure at arrival | -- | 18 / 134 / 0 | 38 / 192 / 1 | 44 / 202 / 2 |
      | target's shift throughout (E7) | D from step 1 | -- | 27 / 183 / 0 | -- | -- |
      | **anchor's shift, stop in the target's scales (E6, built)** | the stop | -- | **19 / 144 / 0** | **38 / 176 / 0** | **34 / 157 / 0** |
      | E6 + steered in the target's scales from arrival (E8 `arrived`) | the stop, the measure at arrival | -- | 17 / 128 / 0 | 35 / 151 / 0 | 41 / 194 / 1 |
      | E6 + steered in the target's scales from step 1 (E8 `always`) | the stop, the measure from step 1 | -- | 19 / 142 / 0 | 35 / 136 / 1 | 38 / 159 / 1 |

      (Columns: pitzDaily's `refit_beta_floor` was 0.05 until #709 and none after; the "floor 0.05"
      pitzDaily arms ran at `78e0bbf`-based commits, the rest on the merge with main at `159536c`.)
      **What each arm showed.** Building the station's policy at the march's iterate re-froze `nu_t`, the
      mass flux and the log-omega coordinate factor as well as the viscosity (the #156 hazard; omega shift
      moved by a median 1.86x, up to 25x, on pitzDaily), which two independent reviews caught. With that
      fixed, any change of `D` still cost: on bfs3d every swapping arm parted from main at the first
      change, ran line-search clipped and positivity-limited, and had to walk the shift strength back up
      (to ~0.1 where main stays at or below 0.035) before a plateau near `|R|` 2.2e-2; E5 matched main
      exactly through step 12 and paid the same plateau after arrival. On pitzDaily the arms that damped
      the viscous rows at about the target's level late in the ramp (the per-station swap, E7) stalled
      at steps 10-14 with one-inner, 3-cycle steps; E5 kept the anchor's there and descended, and E7 had a
      `rho1` of 1.6e7 three steps after arriving. **E6 is main step for step** -- pitzDaily's step table identical to
      main's on every row (main re-run at `159536c` on the same machine, 591 s against E6's 618 s, noise);
      bfs3d identical on every row through main's stop, then 2 more steps at the floor and 1 without,
      because the case's own scales read its residual 1.4-4x higher than the anchor's at bfs3d's end
      state. That extra is the stated `atol` actually being met: main stopped at a looser bar than the
      case file says. The owner kept `atol` 1e-5 (2026-10-14).
      **E8, steering in the target's scales too (prototype `AQUAFLUX_STEER_BY_STOPPING`, branch
      `materialized-ramp-e8` at `61331d6`, not merged), loses on the case file bfs3d ships now.** Asked
      because one measure for steering and stopping is the consistent design; it keeps the anchor's shift
      and changes only the measure the line search, step control and Krylov stop read. It won on pitzDaily
      (`arrived`) and on the pre-#712 bfs3d file at the 0.05 floor, and lost without the floor (`arrived`'s
      first full-length trial at arrival had a residual ratio of 4e11, then a six-step park near `|R|`
      8e-3 and a twice-redone step). On **#712's bfs3d file** (multiple-correction gradient, residual stop,
      `grow` 3 with the release, no refit floor, damping 3, uniform probe reach 3; owner's runs at
      `61331d6` + #712's `case.yaml`/`compare.py`, same machine), steps / restart cycles (with rejected
      attempts) / steps redone: main 19 / 126 (157) / 2, **E6 20 / 134 (165) / 2**, E8 `arrived` 21 / 159
      (194) / 2, E8 `always` 25 / 153 (203) / 7; main's two redone steps (6 twice, 14) are #712's own and
      E6 repeats them. E6 again follows main row for row and takes one more step: main's last state reads
      1.95e-5 in the case's scales, 3.8x its steered 5.18e-6. So the steering measure stays the anchor's.
      ⚠️ **Every E8 number above was taken under the global k cap, deleted 2026-10-14, and the cap is
      what E8 lost to:** each of its redone steps is the cap set by one or two side-wall corner cells. Under
      the per-cell clip + omega bound `e` that replaced it (same machine and file, E8 at `61331d6` with the
      guard ported), E8 `arrived` takes **17 / 133 (133) / 0 / 566 s** against merged main's (`da37958a`,
      #710 + the guard) **18 / 137 (137) / 0 / 552 s** -- a tie on single runs, not a loss. At bound `e^2`
      or unbounded E8 regresses (28 and 24 steps); `always` at `e^2`, 38. So the decision stands (no gain
      measured), but its reason is now "no gain", not "loses"; arms and configuration in
      `.claude/notes/solve-globalization-log.md`, "The shift ramp is not the cause".
      **Retired with the swap:** a retune sweep of the swapping arm (`beta_min` 0.01: 24 / 176 / 0;
      `refresh_on_cycles` 4: 35 / 339; `beta_min` 0.02: stopped at step 37; two arms damping the flow
      rows harder at the closure's shift, both worse and non-finite early) ran from `sweep-*.yaml` files
      beside pitzDaily's `case.yaml` at commit `0074a1a`, deleted since; `PITZ_CASE_FILE` (an edited copy
      of the case file) stays in `compare.py`. **Not fixed, found by the reviews:** a `station_step` swap
      is undone on a station-entering step under `RebuildingSession`'s graft (the block family; reachable
      from the `PITZ_TURB_DAMPING_TARGET` harness arm only).

### `tools/sibling_builders.py` and `coupled_step` (#392)

    - **`tools/sibling_builders.py` reports `coupled_step` (fixed 2026-09-14, #392) — beside
      `flow_march_step` since #541 deleted its mass-flow sibling, which it paired with until then.** It was blind to `coupled_step` for the life of #371 — the call
      `session._build(...)` names a method four classes define, and the tool dropped it — so the coupled
      family was absent from the report, which reads as clean. It now resolves the method on the classes
      `open_session` returns. `test_sibling_builders.py` pins that it still sees `coupled_step`.
      It is the only pair the fix added; the mechanism, and the six invented pairs its first version
      reported before review, are in `CLAUDE.md`'s sibling-builder item.

### The constrained adjoint test was passing by ~1e-11 (2026-08-19)

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
    claims to (through `solve_coupled(adjoint_solver=...)`, since #541 the one coupled driver).
    **Consequence for anyone measuring here: a mass-flow adjoint result is not a stable gate.** Any
    change that perturbs the warm state in its last bits — a preconditioner tweak, a library version,
    a different accelerator — can flip it, and the flip says nothing about the change.

## Near-wall treatment — `boundary.py`

### The imposed wall ω gradient — measured inert on pitzDaily (2026-07-25)

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

### #312 — `abs` on the k shift (reverted 2026-08-26) and `|r| + |f|` (inert, 2026-10-12)

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
      blend), and returning the reaction diagonal **unclamped** from `_scalar_stencil` (was `_scalar_operator_pieces`) with the
      preconditioner keeping its own `max(·, 0)` — that pair is a behaviour-preserving separation. What
      is refuted is `abs(·)` on the shift and `live_eddy_viscosity=True` on the coupled path.
    - **`|r| + |f|` — add the magnitude rather than net it — was MEASURED 2026-10-12 and is INERT on the
      shipped case. #312 is closed on it.** Built opt-in (an `SSTTurbulence.k_production_feedback`
      method, exact against the coupled-level difference to 2.6e-16), measured, and **deleted**; it is in
      branch history (`probe-312-additive-shift`, dda9be7) if a case ever needs it. The decisive fact is
      that **the shipped pitzDaily case damps the k and omega rows at `turbulence_damping: 3`**, which
      already keeps the k row's effective diagonal well clear of zero. Every #312 number above was taken
      at damping 1, because `closure_stall_probe.py` built its engine without the case's shift settings
      (fixed in the same change; see `validation.md`).
      - **March A/B, the shipped case as shipped against the same with `d + f` on the k shift** (both
        2026-10-12, Linux x86_64, 4 cores, jax 0.11.2, `case.yaml` at `8d65a8d`: field split
        `SimpleSmoothed`/`JacobiSmoothed`, residual stop, `grow` 3, release 1e-4, 12 x 1 momentum-only
        ramp from anchor 100, damping 3): **16 steps / 111 restart cycles in BOTH arms**, the same
        per-step minimum inner alpha throughout (0.5 from step 3, 0.25 at step 5), per-step `|R|` to
        three figures, final `|R|` 1.979e-06 / 1.965e-06, `x_r/h` 8.0686 both. Walls 751 / 666 s differ
        by the second run's warm compilation cache (step 1 alone: 144 / 80 s), not by the change.
      - **The probe, at the regenerated target-station iterate (`stall-iterate-skew.npz`, captured and
        marched WITHOUT damping, so the state is the damping-1 trajectory's), beta 0.5.** At damping 1
        `d + f` was a real local repair — hotspot cell 10823, `J_kk + beta d` +8.2e-04 / +6.7e-04
        (owner/skew) to +3.6e-03 / +3.5e-03, `max |dk/k|` at the wall 2168 / 4346 to 32 / 37, kept
        alpha 1 / 0.5 to 1 / 1, `|G|/|G0|` 0.216 / 0.593 to 0.102 / 0.103 — matching the frozen
        production operator (0.103). **At the case's own damping 3 there is nothing to repair:** the
        hotspot moves off the wall-fixed band (cells 504/36/18, `J_kk` already **+5.9e-03**),
        `max |d log omega|` is 0.11, every arm keeps alpha 1 and reaches `|G|/|G0|` **0.034**; `d + f`
        moves that to 0.0335 and `max |dk/k|` 6 to 1.2.
      - **What would re-open it:** a case that runs undamped closure rows (`turbulence_damping` 1) and
        shows the k-row near-cancellation in the march. The opt-in is then one revert away.
    - **`explicit_production_viscosity` still addresses the same row at damping 1** (its `frozen` rows
      gave `J_kk` **+4.78e-03**, alpha **1**, `|G|/|G0|` **0.103** against the exact operator's 0.975 at
      the August iterate); at damping 3 it is as inert as `d + f` (0.035 against 0.034).

### The ω shift diagonal inherits ω's dynamic range (2026-07-25, #33)

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

### The first ring is not special — a uniform ~13 % ω imbalance (2026-07-25)

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

### The blend shape: OpenFOAM and Fluent pick different exponents (2026-07-24)

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

### pitzDaily reference and the max blend (2026-07-24)

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

### `nut_wall`'s derivative test, and its unverified log-branch value (2026-08-21)

    ⚠️ **That last one used to read "differentiability" and was a finiteness check, which cannot fail:**
    `0.0` is finite, so wrapping `nut_wall`'s return in `stop_gradient` left all 28 tests in that file
    green (measured 2026-08-21), and nothing anywhere else noticed either — 119 more turbulence/coupled
    unit tests and the wall-resolved `test_channel_law_of_the_wall` all passed. The derivative is in every
    Jacobian and every adjoint this closure appears in, so it now has a finite-difference comparison and a
    non-zero assertion. **Its VALUE in the log branch is still integration-unverified**: doubling that
    branch kills three unit tests and neither turbulent-channel integration test, because the coupled
    fixtures are wall-resolved (`y*` = 1.44 at the wall-adjacent cells against `y*_lam` = 11.53), so the
    `jnp.where` selects the zero branch at every wall face.

### Near-wall k collapse at trihedral wall corners (2026-08-10)

    **⚠️ (2026-08-10) A cell with three wall faces DOES have a non-negative root — the alarm below is
    withdrawn, but read it: the near-wall `k` collapse it describes is real and unexplained.**
    Measured directly: holding every other field fixed, `R_k` at the worst cell is linear to seven digits
    with root `k* = +1.99e-14`, strictly positive, and the iterate sits eight decades *below* it. The
    corner `k` system is homogeneous and an M-matrix — production, destruction and the `Dirichlet(0)` wall
    term all vanish linearly at `k = 0` — so `k = 0` solves it exactly when the neighbours are 0 and no
    negative root exists. The near-wall `k` trough seen at that state is a **Reynolds-
    continuation transient, not a defect in this closure** — the converged target-Re root has **zero**
    cells below `k = 1e-6` (`k_min` 1.30e-05, median 0.7017 against OpenFOAM's 0.7468).

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

## The coupled solve — `CoupledRANS` / `solve_coupled`

### The forward stopping measure — what a reader of older measurements needs (#282)

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

### The coupled line search and the large-restart Krylov (the pitzDaily performance fix)

  - **`coupled_continuation` globalizes with a line search + a larger-restart Krylov (the pitzDaily
    performance fix).** Two measured facts drove this. **(1) The full coupled Newton step
    from the hybrid IC overshoots by ~10⁷×** (‖R‖ 220 → 5.8e9); the pseudo-transient step's only recourse
    used to be escalating β — a *full re-solve* — and escalating β (16/64) still did **not** descend
    (rel ≈ 1.0 → the full-mesh march *stalled*, which had been misread as "slow, compute-heavy"). A
    **backtracking line search** on the one β₀ solve finds α≈¼ → rel≈0.48 (residual halved), so
    an unset `Globalization.line_search` takes `_COUPLED_LINE_SEARCH` on every coupled builder (see `.claude/rules/solve-globalization.md`); β
    escalation stays the fallback for a bad *direction*, not an overshoot. With it the full-mesh solve
    **descends** (rel 1.0 → 0.48 → 0.44 → 0.31 → 0.20 → ~0.18 over ~6 steps) instead of *stalling at
    rel 1.0* — the case is now solvable at all, a correctness fix, not just speed. **(2) The shifted
    solve needs a large Krylov subspace:** the block family's regime (`_BLOCK_LINEAR_SOLVE`, ⚠️ **the symbol
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
    the current default. See the `krylov_solver` bullet in `.claude/rules/solve-globalization.md` for the mechanism
    (`lineax`'s componentwise stop plus the near-zero-right-hand-side ω wall-fixation rows pinned it to
    the absolute `atol=1e-10` floor, ~9 orders past the requested 1e-3) and the two-arm refutation.

### `beta_floor` and the convection-aware velocity AMG

  - **`beta_floor` (SER lower bound) is available but off by default (a measured wash).** ⚠️ It is a
    field of the shared `Globalization` now, not a builder keyword —
    `coupled_continuation(globalization=Globalization(beta_floor=…))`, which keeps the coupled line
    search — and it reaches the flow-only and scalar marches too. Bounding
    `β = max(beta_floor, β₀(‖R‖/‖R₀‖)^p)` keeps each late shifted solve out of the ill-conditioned low-`β`
    regime (correctness-safe — the floor scales the correction `δ`, which vanishes at the root, so it never
    moves the converged state). But end-to-end it is a **net wash** (cheaper late solves cancel the extra
    Newton steps), so it defaults to `0`; reachable from every march for further evaluation. The
    settled coupled-solve cost is the diagonal-block-preconditioner weakness at high Reynolds number, **not**
    the residual measure, `β` floor, or missing cross-coupling (a block-triangular preconditioner was worse
    — non-convergent on recirculating pitzDaily). See `.claude/notes/solve-globalization-log.md`.
  - **The coupled flow block uses the convection-aware velocity AMG, not the viscous-smoothed default
    (`_coupled_shift_policy`).** A RANS case is high-Reynolds, and the default `BlockPreconditioner.build`
    velocity config (viscous-**smoothed** AMG, which is Peclet-blind) produces a poor momentum-block
    direction once the flow separates. Measured on the developed pitzDaily field (shifted Newton direction
    vs the true one): smoothed gives a direction badly misaligned with the true Newton step and the march
    stalls at rel ~0.18, while **`velocity="convection"`** is nearly aligned. The convection block's
    linearization stays valid **frozen once**, so **the flow block needs no reference refresh** — verified
    two ways: an IC-frozen direction is as well aligned as a plateau-rebuilt one, and refreshing the flow
    block alone at a separated pitzDaily state is if anything slightly *worse*. ⚠️ Both were measured
    while "frozen at the IC" meant the **boundary-derived uniform plug** (`BlockPreconditioner.build` with
    no `reference_state`), not the hybrid start; see the next bullet. It is
    **not** the flow↔turbulence cross-coupling (the block-*diagonal* preconditioner with the right config
    is already nearly aligned on its own — a block-triangular coupling was built, measured, and is worse;
    see `.claude/notes/solve-globalization-log.md`). **The k/ω *scalar* AMGs are the exception: they do go
    stale, and refreshing them alone once the flow separates cuts the outer cycle count materially**
    (configuration not recorded — re-measure before relying on the size) — the one staleness lever that
    pays; see the staleness bullet in `.claude/notes/solve-globalization-log.md`. Overridable through
    `BlockDiagonal(velocity=…)`.

### #389 — where plug and reference both build (2026-10-13)

    - **Where both build, they are indistinguishable.** Harness `validation/velocity_block_reference.py`
      (default `BlockDiagonal()`, hybrid start, single shifted step, `max_steps` 300, one run per arm,
      Linux x86_64, 4 cores, jax 0.11.2): 0.10 H bump — plug 39 steps / 97 restart cycles, reference 39 /
      91, roots agree to 2e-13, separated length 0.4165 both; 0.30 H bump — plug refused to build,
      reference 56 / 207, separated length 1.50; straight channel (`test_coupled_rans.py`'s, where the
      two frozen fluxes agree to 2e-14) — 20 / 17 against 20 / 15. The control is what reads the 0.10 H
      gap as noise: identical blocks to 5e-14 still differ by two restart cycles (~12 %).

### MSIMPLE no longer hardcoded in `_coupled_shift_policy` (2026-08-18)

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
    those small fixtures (`test_coupled_rans.py`, `test_coupled_mass_flow.py`, `test_coupled_lu.py` (since deleted),
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

### Materialized re-fit: the deleted β-diagonal split and the probe knobs

    - **The refresh cadence is COST-TRIGGERED; the scheduled gates are DELETED (2026-09-13, #371).** The
      hook re-fits on its first call, after `rebind`, and when the march passes `due` -- before the step
      after one that stopped because an inner solve reached `refresh_on_cycles` (it re-fitted mid-step,
      through a `refresh_at` callback, until 2026-10-10, #290). A fixed step-count cadence failed at the low-β tail (stale
      mid-interval); the β-mismatch and drift gates that replaced it measured 14 % slower than the cost
      trigger on `bfs3d` (3632 s against 3140 s at unchanged cycles, monolithic; see
      `.claude/notes/solve-refuted-directions.md`) and were switched off by both cases before being
      deleted. **Trap worth keeping:** a badly mismatched factorization collapses α, which freezes the
      state and every drift measure on it, so a drift trigger cannot fire on that failure mode.
    - **The β-diagonal split (`refresh_shift_in_place`, drift-gated) is DELETED with the gates (#371).**
      Re-materializing `J` by graph-coloured probing is ~half a refresh on `bfs3d` (~18–40 s of ~36–62 s);
      the split re-added `β d` to a cached `J` instead. Since gradients are never
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
      fixes and `jacobian_relative_error` cannot see. `JacobianProbe(gradient_sweeps=n)` (and the
      `probe_gradient_sweeps=n` keyword on `coupled_amg_continuation`, `amg_beta_tracking_refresh` and the
      LU builders) materializes the preconditioner from a copy of the residual whose gradient
      solve is capped at `n` sweeps — `JacobianProbe.narrow` is the one place that decides it, and
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

      The `assemble` half of the refresh is also precomputed now (`ShiftedCellMajorOperator`), and the
      observer receives a `RefreshTiming` with per-phase costs instead of one aggregate — so "the
      materialize is ~half the refresh" is measured per run rather than inferred.

### The per-rung recompile — three causes (2026-08-12)

    - **⚠️ THE PER-RUNG RECOMPILE — THREE independent causes, all three now closed, and finding only
      two of them would have bought nothing (2026-08-12).** Each rung's first step was the most
      expensive step of that rung by a wide margin *at a cycle count no higher than its cheap ones* —
      on one march the target rung's first step cost 141 s at **7** cycles where a **19**-cycle step in
      the same rung cost 57 s. Measured per rung as the first step's excess over the median of its
      rung's same-cycle-count peers (after subtracting that step's own reported `pc ... Ns`), across
      four archived marches: rung 1 **86–90 s**, rung 2 **98–106 s**, rung 3 **112–250 s**. Rung 1's is
      the unavoidable first compile; rungs 2–3 are **~190–230 s, ~10 % of the march**.

      The compiled unit is `_march_step`, whose key is the Newton step, the residual and the solver.
      **Any one difference recompiles the whole coupled solve, so this had to be closed on every axis
      at once** — which is why the earlier probe fix (above) moved only 20 %:
      1. **The preconditioner object.** It rides in a *static* field of the step and is compared by
         **identity**, so a rung that fits its own V-cycle is a new key. Fixed by reusing one object:
         `coupled_amg_continuation(preconditioner=…)` glues in an existing V-cycle instead of fitting
         one, and `amg_beta_tracking_refresh(...).rebind(companion)` points the shared refresh hook at
         the new rung and forces its next refresh to a **full** re-materialize — so the V-cycle is
         still fitted to each rung's own state and shift (the march calls `refresh_preconditioner` before a
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
      Also removed while sharing: the probe plan and its gather map (`JacobianProbe`, mesh-fixed
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

### pitzDaily cost history (2026-07)

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
    `K=100` 9.63e-2, **unlimited (`limiter=None`, ψ≡1) 9.71e-2** — the `K`s are of the OLD
    `eps² = vol K³` softening, deleted 2026-10-09 (#144). Removing the limiter entirely
    reproduces the limited result, so the first-vs-second-order difference is inherent to the
    *reconstruction*, not to limiting. (Two genuine limiter defects were found in that audit and filed —
    a periodic-image inconsistency (#143) and the dimensionally inconsistent `eps²` softening (#144),
    both since fixed — but neither caused this, and neither was a convergence fix.)

### Held bulk velocity on the law-of-the-wall files (2026-10-11, #541)

    ⚠️ **Measured on the law-of-the-wall files (2026-10-11, #541; table and configuration in
    `validation/turbulent_channel/README.md`, harness `mass_flow_march.py`): re20000 converges on the
    staged march, and re240000 is converged by no coupled arm, before or after the move.** The staged march
    reaches the segregated root to the printed digits. Under `Euclidean()` it costs about what the old
    `RootSolver` path did (33 steps against 23). **Under the default `RowScaled()` it is 101 steps
    single-step and 271 dual-time**, against the old path's 23 and 13 under its Euclidean test. The per-block
    line shows the constraint row at machine zero and `k` as the slow block, so the shift schedule relaxes
    slowly. The default measure costs time here; the border does not. At re240000 the default single step
    falls slowly (ratio 0.14 after 200 steps), `Euclidean()` freezes as the old path did, and a viscosity
    ramp (100x, 16 stations of one step) diverges. So the coupled march does **not** yet dominate
    `Segregated` for a held bulk velocity, and `Segregated` stays.

## Initialization, diagnostics, Reynolds continuation

### The wall-consistent seed at every Reynolds handover (2026-09-09)

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

### ViscosityRampHomotopy — schedule, per-station refresh and the anchor seed (2026-09-10/11)

  - **⚠️ THE SCHEDULE WAS AN ARBITRARY GUESS AND A POOR ONE — 11 arms, 2026-09-10, pitzDaily,
    uncontended, warm cache, all reaching `x_r/h` 8.0686.** `24 stations x 1 step, redamping 1.0` costs
    **191** restart cycles against **261** for the original `4 x 3, 2.0` and **417** for the ladder.
    - **The mechanism, which is the finding.** Net shift change per station is
      `redamping * grow ** (1 - steps_per_station)` — an entering step is HELD and forgoes its own
      `/grow`. At `(3, 2.0)` that is `2 / 1.5² = 0.889`, so **β barely descends across the ramp**: the
      target station is handed β = **0.936** and then spends ~13 of its 28 steps walking β down. That
      is the *same* per-rung shift restart the homotopy exists to delete, removed from between the rungs
      and recreated inside the ramp. It is why the target station's cost is near-invariant across coarse
      schedules (231/232/235/238/242 cycles for five of them) — the schedule cannot reach it.
    - **Why the fine ramp is safe at β ≈ 0.012, which was the measured wall.** At 24 stations the
      viscosity moves `1.21x` per station instead of `3.16x`, so the state barely moves and no damping
      response is warranted. That is the `(state, β)` rule one level down: **the lever is to make the
      problem move less per step, not to damp after it has moved a lot.**
    - **Interior optimum, broad basin:** 12 → 246, 24 → **191**, 32 → 225, 48 → 245 cycles. Past ~32 the
      ramp's own steps and rebuilds outweigh what the target station has left to give (its cost floors
      near 40 cycles).
    - **Granularity is the lever, not ramp length** — at a fixed 24 ramp steps, `8 x 3, r=2` costs 286
      against `24 x 1, r=1`'s 191.
    - **Anchor depth: 100 is the cheapest start measured on BOTH cases (2026-10-11).** pitzDaily at its
      shipped march (`3 x 12`, `grow` 3, release 1e-4, momentum-only scaling; `PITZ_RAMP_ANCHOR`), anchor
      1 (no ramp) / 3 / 10 / **100** / 300: **246 / 217 / 178 / 111 / 255** restart cycles, all to the same
      root; 1000 was stopped at 502 cycles by step 38. With the release backoff (2026-10-12) 300 costs 210
      and 1000 converges at 612; 100 is unchanged at 111. bfs3d at its shipped `1.5 x 12` (#685), anchor
      10 / **100** / 1000: 239 / **193** / 329 all-attempt cycles. Shallower costs redone steps early, at a
      small shift; deeper gives the cleanest ramp and then pays at the target -- pitzDaily's releases
      failing until the residual is small (re-fired every other step until the backoff,
      `solve-globalization.md`), bfs3d's 64 floor steps with the release off. ⚠️ At fixed stations a deeper anchor is also a larger per-station viscosity jump, landing at
      the floor; deep anchors with stations scaled to hold the jump are unmeasured. No ramp still
      converges on pitzDaily (16 steps), at 2.2x the cycles. Detail in
      `.claude/notes/solve-globalization-log.md` ("Where the ramp starts").
    - **Adaptive `steps_per_station` is NOT expressible** — the whole `ResidualHomotopy` protocol keys on
      the outer-step index and is never handed the state, the residual or a `StepReport`. Widening it is
      a design change, and the fine-ramp result says the answer is small stations rather than a clever
      rule for leaving one.
    - **⚠️ THE OPTIMUM ABOVE WAS MEASURED AT `grow` 1.5, AND THE STATION COUNT IS THE SHIFT'S DESCENT IN
      DISGUISE (measured 2026-10-11, with the shift released).** `beta` takes 12 steps to reach the floor
      at `grow` 1.5, which is where the optimum sits; at `grow` 3 it takes 7, and `3 x 12` costs **111**
      cycles in 16 steps against `1.5 x 16`'s 141 in 22. The viscosity step size does not move `alpha`
      (identical histories at 1.33x and 2x per step); what is conserved is 6-9 steps AT the floor before
      the state settles (at those paces: at 1.21x per step it settled at the 4th), and the cheapest ramp ends one step after that. Table and the design it closes
      in `.claude/notes/solve-globalization-log.md` ("The ramp's length is the shift's descent").
      **pitzDaily's `case.yaml` ships `3 x 12` with the release since 2026-10-11, and bfs3d's since
      2026-10-13** -- with the residual stop, which it needs: there `3 x 12` with the release cost 214 accepted / 459 all-attempt cycles and nine steps
      redone against the shipped `1.5 x 12`'s 174 / 193 under bfs3d's lineax inner stop, but 195 with three
      redone under pitzDaily's `stop: residual` -- the stop, not the rate, was most of the loss. Its anchor
      100 was the cheapest of 10 / 100 / 1000 (anchor 1000: 92 steps, a 64-step floor tail). (2026-10-11,
      #678; `solve-globalization-log.md`, "bfs3d: the release, `grow` 3 and the settle-ended ramp".)
  - **⚠️ THE PRECONDITIONER IS REFRESHED PER STATION CHANGE, AND THAT IS CORRECT — but "a full rebuild is
    the most expensive operation in the march" is REFUTED on this case.** Measured, a full
    re-materialize is **1.2-1.6 s** against a **~9 s** outer step. A control arm isolates it: `4 x 3` with
    `PITZ_REFRESH_ON_CYCLES=1` (a rebuild essentially every step) costs 253 cycles against 261 — so
    preconditioner freshness is worth ~3 % and the remaining ~27 % of the fine ramp's win is the
    viscosity/β path, not the refresh.
    - **A shift-only refresh could not have substituted** — it re-added `β d` to a frozen Jacobian, while
      viscosity enters every viscous edge coefficient — and it no longer exists (#371).
    - **`refresh_on_cycles` needs no reset at a station change** — its state lives in the dual-time step's
      `while_loop` carry and never crosses a step.
    - **Drift within a station is nil on the ramp and heavy on the target**: all 16 cost-triggered inner
      refreshes fire in the target station, zero across the 12 ramp steps.
    - ⚠️ **The balance shifts on `bfs3d` by less than the record claimed, and the "overhead" framing
      DOUBLE-COUNTS.** Measured 2026-09-10 from that case's own march logs: a full re-materialize is
      **11.5 s** against a ~34 s mean outer step. The **~36 s** figure previously carried here and
      quoted into two docstrings came from the record rather than from a log and is **wrong by 3x**; it
      made a fine ramp look disqualified on `bfs3d` when it is not. But "24 stations cost ~8 steps of
      overhead" is wrong for a second, independent reason: **the ladder rebuilds too.** Counting the
      `pc …` lines in both logs (23040 cells, field split on, `simplesmooth`, column reach 3/3/3/3/2/2,
      ILU(0)/4 sweeps, `REFRESH_ON_CYCLES=3`, `PC_BETA_FLOOR=0.05`):

      | arm | steps | preconditioner builds | pc seconds | share of wall |
      |---|---|---|---|---|
      | ladder (converged) | 59 | 33 (28 cost-triggered inner refreshes) | 391 s | 20 % |
      | ramp, 24x1 (aborted) | 32 | 37 (23 station-triggered + 7 inner) | 490 s | 29 % |

      Comparable **totals per march**, because the ladder's `refresh_on_cycles` fires on nearly every
      step of its long target rung. The station rebuilds largely **replace** those rather than adding
      to them, so a schedule may not be costed as `stations x rebuild`.
    - ⚠️ **THE FIRST `bfs3d` RAMP ARM WAS SEEDED AT THE WRONG VISCOSITY — a defect in
      `solve_reynolds_ramp`, not a property of the ramp (found and fixed 2026-09-10).** It called
      `hybrid_initialize(coupled…)` — the **target** assembler — while the march's first station solves
      the **anchor**. The ladder calls it on the companion. `hybrid_initialize` reads
      `turbulence.molecular_viscosity` to seed the near-wall `omega` at the profile that assembler's
      residual *fixes* there, so the seed put every wall-adjacent cell off its own boundary condition —
      exactly the residual the seeding exists to remove. Measured on `bfs3d` at anchor 100: `omega`
      differed by up to **90x**, `flow` and `k` not at all. In the march that is the anchor's first step
      opening at a row-scaled **9.60e-01 against 3.29e-01**, a line search collapsing to `alpha` 0.016,
      **two discarded attempts**, a retry escalation of β to **4.0**, and ~10 further steps spent
      walking the shift back down.
      - ⚠️ **So the earlier reading of that arm — "`bfs3d` opens at `beta_start = 2.0` against
        pitzDaily's 0.5, the ramp began four times stiffer" — is FALSE and is deleted.** Both cases set
        `beta_start = 0.5`; the 2.0/4.0 in that log is the **retry ladder escalating**, i.e. an effect
        of the bad seed, not a setting. A shift regime read off a march is not a case parameter until
        the parameter itself has been read.
      - **Fixed by seeding from the anchor companion**, which is what makes the docstring's claim that
        the two arms are "configured identically" true. Pinned by
        `test_the_ramp_arm_seeds_the_hybrid_start_from_the_anchor_not_from_the_target`; the pre-existing
        ramp tests stub `hybrid_initialize` **without inspecting its arguments**, which is how this
        shipped and why the new test records what it was handed.
      - `tools/sibling_builders.py` could not see this pair until 2026-09-13: both builders
        `return solve_coupled(...)`, a lowercase function it did not follow. It follows unambiguous
        functions and methods now, so both are visible; they share two parameters (`coupled`,
        `point_setup`) and are correctly not reported. See the tool's entry in `CLAUDE.md`.
      - ⚠️ **RE-MEASURED ON pitzDaily (2026-09-10), AND THE FIX COSTS THAT CASE 55 % MORE CYCLES.** All
        four numbers below are from this tree (commit `a403752` + the fix), `PITZ_RAMP_STATIONS=24`,
        `PITZ_RAMP_STEPS=1`, `N_POINTS=2` ratio 10 (anchor Re/100), `BETA_START=0.5`, `beta_min=0.005`,
        `grow=1.5`, `backoff=2.0`, field split on, `simplesmooth` flow inverse, `jacobi_smoothed`
        trailing inverse, `REFRESH_ON_CYCLES=3`, `PC_BETA_FLOOR=0.05`, stop `rtol=0, atol=1e-5`,
        compiled ILU(0) live. Every arm reaches `x_r/h` **8.0686** and none takes a retry.

        | arm | steps | restart cycles |
        |---|---|---|
        | rung ladder | 69 | 417 |
        | ramp 24 x 1, **pre-fix** (two runs, identical) | 33 | **191** |
        | ramp 24 x 1, **post-fix** | 38 | **297** |

        - **The tree reproduces the recorded ladder EXACTLY, rung for rung** — 28/123, 21/139, 20/155,
          final `|R|` 5.771e-06 — so these are comparable to the older record rather than a new baseline.
          That also makes the pre-fix ramp's two identical runs (33/191 both times) a genuine
          determinism check: this arm does not vary run to run, so a single post-fix run is decisive.
        - **⚠️ THE MECHANISM, WHICH INVERTS THE OBVIOUS READING OF THE DEFECT.** The ramp **terminates at
          the target**, so a seed built at the *target's* viscosity starts `omega` where it has to end —
          only the flow and `k` need develop. The anchor seed, correct for station 0, is ~90x too large
          near the walls and must be walked back **down** across all 24 stations. **The defect was
          accidentally a better warm start for a homotopy that ends at the target.** Correctness and
          cheapness point opposite ways here, and the fix buys the former.
        - **What the fix is still right for:** it makes the two arms genuinely comparable. Verified —
          the post-fix ramp's step 1 (`|R|` 5.668e-02, 2 inners, 2 cycles, `a_min` 1.000) is *identical*
          to the ladder's, as it must be, since both solve the anchor from the same seed with the same
          preconditioner. Pre-fix, that step read `|R|` 2.393e-01 at `a_min` **0.125** and pushed β up
          to 1.0, costing three steps to walk back down — degraded quietly, **without ever tripping the
          retry ladder** (0 retries in every arm), which is why it went unnoticed here.
        - **The ramp still beats the ladder either way** — 1.40x on cycles post-fix against 2.18x
          pre-fix — so nothing here argues for reverting to the ladder on this case.
        - ⚠️ **NOTHING HERE ESTABLISHES THAT THE FIX HELPED `bfs3d`.** That case's comparison is
          fixed-ramp (240) against ladder (349); its *unfixed* ramp was killed unconverged, so no
          fix-vs-no-fix number exists there. On the only case where both seedings ran to convergence,
          the unfixed one is cheaper. Do not generalize the `bfs3d` win into a claim about the seed.

### ViscosityRampHomotopy — momentum-only scaling and `turbulence_damping` (2026-09-10/11)

          - **⚠️ MEASURED ON pitzDaily (2026-09-10, same bundle as the table above), and BOTH predicted
            effects appear and partly cancel:** 32 outer steps / **264** cycles, `x_r/h` 8.0686, no
            retries — against the correctly-seeded both-blocks ramp's 38 / **297** and the
            defect-seeded 33 / **191**.

            | arm | steps | ramp stations | target station | total |
            |---|---|---|---|---|
            | both blocks, post-fix | 38 | 179 | 118 | 297 |
            | momentum only | 32 | **214** | **50** | **264** |

            The split *is* the mechanism. The transported scalars carry their target stiffness from the
            first step, so the ramp costs **214 cycles against 179** with `alpha` clipped to 0.25–0.5 for
            a dozen consecutive steps and the inner loop at its cap — the predicted risk, real. Against
            it, `omega` arrives already at its target profile, so the target station costs **50 against
            118**. Net 33 cycles better than the correct both-blocks arm, and the fewest outer steps of
            any arm measured on this case.
          - **⚠️ STATION SWEEP (2026-09-10, same bundle, `steps_per_station=1` throughout): the station
            count is a WEAK lever for this strategy, and the prediction that drove the sweep was wrong.**

            | stations | steps | cycles | retries |
            |---|---|---|---|
            | 4 | 27 | 237 | 2 |
            | 8 | 28 | 240 | 3 |
            | 12 | 28 | 238 | 4 |
            | **16** | **27** | **227** | **0** |
            | 24 | 32 | 264 | 0 |

            All five reach `x_r/h` **8.0686**. The expectation was a monotone "coarser is better",
            because momentum-only front-loads its cost into the stations. Instead **4/8/12 are a flat
            plateau** — 237/240/238, a **3-cycle spread over a 3x range of station counts** — and the
            best point is **16, in the interior**.
            - **⚠️ READ THE RETRY COLUMN, NOT ONLY THE CYCLES.** 4/8/12 reach ~238 cycles by **grinding
              through the low-β wall**: 2, 3 and 4 retries respectively, with `alpha` pinned at 0.25 for
              a stretch of consecutive steps. 16 and 24 take **none**. So 16 is not a marginal 4 % win
              over 237 — it is the only coarse-ish point that gets there *cleanly*, which is a different
              and more durable property. A reader comparing only totals would pick 4 and inherit the
              grind.
            - **Where flow-only stands on this case at its own best schedule:** ladder 417, both-blocks
              post-fix 297, **flow-only 16 x 1 = 227**, both-blocks pre-fix (defect-seeded) 191. So it is
              **24 % cheaper than the correctly-seeded both-blocks ramp** and 1.84x the ladder, while
              still not reaching what the defect achieved by accident. It is the best *principled*
              configuration measured here.
            - One run per point, one case, one step per station. `steps_per_station > 1` is untested for
              this strategy, and so is every point above 24.
          - **⚠️ MEASURED ON `bfs3d` TOO (2026-09-10), AND IT WINS BY MORE THERE — 21 % over the
            both-blocks ramp and 46 % over the ladder.** Schedule held at **24 x 1 deliberately**, the
            same as that case's both-blocks arm, so the *only* difference is the scaling strategy;
            everything else is the case's own defaults (23040 cells, anchor Re/100, `beta_start=0.5`,
            field split on, `simplesmooth` / `jacobi`, ILU(0) inert — both blocks supply their own
            inverse — `PC_BETA_FLOOR=0.05`, `REFRESH_ON_CYCLES=3`, stop `rtol=0, atol=1e-5`).

            | arm | steps | ramp stations | target station | total | retries | `x_r/h` |
            |---|---|---|---|---|---|---|
            | rung ladder | 59 | — | — | 349 | — | 8.3611 |
            | ramp, both blocks | 39 | 122 | 118 (15 steps) | 240 | 3 | 8.966 |
            | **ramp, momentum only** | **31** | **151** | **39 (7 steps)** | **190** | 3 | **8.3611** |

            - **⚠️ THE SPLIT IS THE RESULT, and it is starker here than on the sibling.** Momentum-only
              pays **more** in the ramp (151 against 122) — the `k`/`omega` stiffness is real and arrives
              immediately: step 1's residual **grew 6.8x** (`G` 1.911e-01 -> 1.295e+00, `alpha` 0.002),
              took three attempts, and escalated β to 2.0 before the march could start, against a clean
              3-cycle `alpha = 1.000` first step for both-blocks. `alpha` then sat at 0.25-0.5 through the
              mid-ramp. Against that, its target station costs **39 cycles over 7 steps against 118 over
              15** — a **3x** saving, because `omega` arrives already at its target profile. Net -50.
            - **Every cost is in the `k`/`omega` block and every gain is in not having to move `omega`.**
              That is the cleanest evidence yet that the two blocks want *different instruments*:
              continuation for the momentum block's convective nonlinearity, and damping (a larger
              pseudo-transient shift) for the closure's stiff source terms. `CoupledShiftPolicy.shift_term`
              already packs the shift **per block** (`coupled.py`, `flow_diagonal` and the two scalar
              transport diagonals, packed a line apart), so a per-block β multiplier is a small change
              rather than a redesign.
            - **⚠️ BUILT AND SWEPT (`turbulence_damping`, 2026-09-10/11). A CONSTANT RATIO OF 2 WINS
              12 %, THE OPTIMUM IS INTERIOR, AND EVERY SCHEDULE TRIED IS DOMINATED BY IT.** The k/ω
              rows' shift **strength** is multiplied by `turbulence_damping`, so they run at an
              effective `turbulence_damping * beta` while the velocity rows keep `beta`.

              - **The honest sweep.** pitzDaily, momentum-only scaling at **16 x 1**, everything else
                that case's defaults, one run per arm, all reaching `x_r/h` 8.0686, zero escalations
                except where noted. `gamma = 1` is a control and reproduces the recorded 27 / 227
                exactly, so the knob is a true no-op at 1.0.

                | arm | steps | cycles | ramp (16 steps) | target station | `|R|` handed over |
                |---|---|---|---|---|---|
                | `gamma = 1` | 27 | 227 | — | ~84 | — |
                | **`gamma = 2`** | 29 | **200** | 129 | **71** | 2.814e-03 |
                | `gamma = 5` | 33 | 212 | 102 | 110 | 5.796e-03 |
                | `gamma = 10` | 64 | 329 | **100** | 229 | 1.005e-02 |
                | `gamma = 20` | — | — | 119 | — | — |
                | taper 2 → 1 on `beta` | 28 | 234 | 150 | 84 | — |
                | taper 10 → 1 on `beta` | 39 | 319 | 135 | 184 | — |
                | taper 10 → 1 on `|R_turb|` | killed | — | 183 | stalling | — |
                | station-keyed 5 → 2 | 29 | 211 | 102 | 109 | 5.796e-03 |

              - **⚠️⚠️ THE FINDING THAT SUBSUMES THE REST: DAMPING DOES NOT REMOVE THE CLOSURE'S WORK, IT
                DEFERS IT.** The phase split looks real and large — the ramp's cost falls monotonically
                with `gamma` (129 → 102 → 100, turning back up at 20) while the target station's rises
                (71 → 110 → 229) — so "damp the ramp, release at the target" appears to be worth
                ramp 100 + target 71 = ~171 against the best constant's 200. **It is not available, and
                that was measured rather than argued.** The station-keyed arm gets its cheap ramp exactly
                as asked (102 cycles, matching constant `gamma = 5` *to the cycle*), and switching the
                target station to `gamma = 2` then buys **one** cycle: 109 against constant-5's 110.
                The target's cost is set by **the state the ramp hands it**, not by its own ratio, and
                the handover residual scales with how hard the ramp damped (2.81e-3 / 5.80e-3 / 1.01e-2
                at `gamma` 2 / 5 / 10). The two phases are **one budget**, and it is smallest at a
                constant 2. Do not quote a per-phase cost as a saving.
              - **⚠️ EVERY SIGNAL A SHIFT POLICY CAN READ MEASURES PROGRESS, AND PROGRESS SATURATES
                INSIDE THE RAMP.** That is why both tapers fail, and it is the same defect in each:
                keyed on `beta` the release is complete by **step 12 of the 16-step ramp** (the control
                floors `beta` there), keyed on `|R_turb|` by **step 6** — the closure residual falls
                100x in six steps, being dominated by the initial transient rather than by how far the
                closure is from equilibrium. Both therefore spend the entire release inside the phase
                that wants the damping and hand the target station a ratio of 1. Replay it on any run
                with `validation/pitzdaily_openfoam/damping_taper_trace.py`, which reconstructs the
                ratio actually applied from the checkpoints — a march log reports the ratio nowhere, and
                a taper whose signal never moves is a *constant* wearing a taper's name.
              - **⚠️ NO ARM TOOK A RETRY ON A CONSTANT RATIO, `gamma = 10` INCLUDED** (the beta-keyed
                taper took five, see below). Damping does what it is for — it stabilizes the step — and
                its cost is **lag**: `gamma = 10` runs 64 steps at `alpha = 1.000` throughout with the
                residual crawling at ~0.74 per step for 34 steps at 3 cycles each. **Read a damping
                failure as a residual that will not fall at healthy `alpha`, never as a divergence.**
              - **⚠️ A `beta`-KEYED RATIO AND THE CONTROL'S OWN ESCALATION FORM A POSITIVE FEEDBACK
                LOOP.** The retry ladder raises `beta` on a bad step, which raises the ratio, which makes
                the closure lag harder, which keeps the residual from falling. Measured on the
                `10 -> 1` arm: over steps 20-27 `gamma` climbed back to 6.4 while the residual rose
                monotonically (3.15e-3 → 4.96e-3) and the control kept growing `beta` on cheap 2-cycle
                steps; five escalations, ~90 cycles lost. ⚠️ This refutes the defence that "a retry
                raising the damping is the response a bad step wants".

              - **⚠️ TWO PLUMBING DEFECTS MADE A RESIDUAL-KEYED POLICY STRUCTURALLY INERT, AND NEITHER
                WAS THE CAUSE FIRST RECORDED.** The first recorded diagnosis — a reference frozen at the
                anchor while a continuation makes the problem harder, so the clamp pins the factor — is
                **true and was not load-bearing**; `rebased` exists and is tested, but the taper could
                not have released whatever its reference was, because: (a)
                `MonolithicFactorShiftPolicy.shift_term` rebuilt its `ShiftTerm` from
                `base.shift_term(phi).diagonal` alone, dropping the residual on the way in and the
                per-row multiplier on the way out; and (b) **`DualTimeStep` never passed the residual at
                all** — and every case that wants a damping runs `inner_steps > 1`, i.e. that step. Both
                fixed and pinned (`test_the_dual_time_step_hands_the_policy_the_residual_it_just_computed`,
                `test_the_residual_taper_survives_the_MONOLITHIC_WRAPPER_the_cases_actually_run`). The
                standing lesson: **a wrapper that rebuilds a `ShiftTerm` from `.diagonal` drops
                everything else in silence** — the march runs and the behaviour simply never happens.
              - **⚠️ Widening `ShiftPolicy.shift_term` broke 21 tests, all of them STUB policies in
                `tests/`.** The six library implementations were found and updated; seven stubs were not,
                because the search was a string match on one exact type annotation. **Find implementers
                of a Protocol structurally (AST), not by grepping a signature** — the same shape as
                `sibling_builders.py`'s blind spots: a check that cannot see the case reports clean.

              - **✅ IT TRANSFERS TO `bfs3d`, AND THERE IT IS FREE IN BOTH PHASES (2026-09-11).** The
                same ratio, measured on the case the whole idea came from. Momentum-only at **24 x 1**
                (matched to that case's recorded control), everything else its defaults: 23040 cells,
                anchor Re/100, `beta_start` 0.5, field split on, `simplesmooth` / `jacobi`, column reach
                3/3/3/3/2/2, ILU(0) x 4, coarse 2000, `PC_BETA_FLOOR` 0.05, `REFRESH_ON_CYCLES` 3, stop
                `rtol=0, atol=1e-5`, compiled ILU(0) live. Both arms reach mid-span `x_r/h` **8.3611**.

                | arm | steps | cycles | ramp (24 steps) | target station | escalations |
                |---|---|---|---|---|---|
                | `gamma = 1` (control) | 31 | 190 | 151 | 39 | 2 |
                | **`gamma = 2`** | 31 | **163** | **130** | **33** | **0** |

                **14 % cheaper, better in BOTH phases, and it removes both escalations** — against 12 %
                on the sibling. The control reproduces the recorded 31 / 190 / ramp 151 / target 39
                exactly, so this is one tree, two arms, one variable.
              - **⚠️ SO THE DEFERRAL IS A PROPERTY OF THE RATIO, NOT OF DAMPING.** On `bfs3d` the
                cheaper ramp costs nothing at the handover — both arms enter the target station at the
                same residual (4.27e-4 against 4.34e-4) — whereas on pitzDaily the handover degrades
                monotonically from `gamma` 2 to 10 (2.81e-3 / 5.80e-3 / 1.01e-2). The consistent reading
                across both cases is an **interior optimum near 2**: up to it the damping removes work,
                past it the cheaper ramp is borrowed from the target. Do not quote the deferral as an
                argument against damping; it is an argument against a *large* ratio.
              - **⚠️ THE RATIO AND THE STATION COUNT INTERACT, so neither transfers alone (2026-09-11).**
                Swept jointly on `bfs3d`'s momentum-only ramp, one run per arm, all reaching mid-span
                `x_r/h` 8.3611. `march` is the march's own monotonic clock, which does not tick through
                a machine sleep:

                | stations | gamma | steps | cycles | ramp | target | esc | pc | march |
                |---|---|---|---|---|---|---|---|---|
                | 24 | 3 | 32 | **148** | 114 | 34 | 1 | 453 s | 1322 s |
                | 12 | 2 | 27 | 154 | 42 | 112 | 2 | 249 s | 1145 s |
                | 12 | 3 | 29 | 159 | 47 | 112 | 0 | 259 s | 1059 s |
                | **12** | **5** | 29 | 150 | 44 | 106 | **0** | 245 s | **1001 s** |
                | 12 | 8 | 29 | **141** | 54 | 87 | 4 | 256 s | 1186 s |
                | 8 | 3 | 26 | 157 | 33 | 124 | 1 | 231 s | 1063 s |

                **At 24 stations `gamma = 5` is worse than 3 (169 against 148); at 12 it is better (150
                against 159).** A coarser ramp is a larger disturbance per station for the closure to
                absorb, so the optimum ratio rises as the ramp coarsens — which means a station count
                calibrated at `gamma = 1` does not survive the introduction of damping, and a `gamma`
                calibrated at one station count does not transfer to another.
              - **⚠️ THE TWO CASES WANT OPPOSITE SCHEDULES, AND THE DISCRIMINATOR IS THE REBUILD COST.**
                `bfs3d` shipped **12 stations at `gamma = 5`** (until 2026-10-13: it now takes pitzDaily's
                `gamma = 3` with `grow` 3, unswept on bfs3d), pitzDaily shipped **16 at `gamma = 3`** at the
                control's `grow` 1.5 (since 2026-10-11 it ships 12 at `gamma = 3` with `grow` 3 and the
                release, which is a different optimum for a different reason: the "station count is the
                shift's descent" entry under the ramp). Swept on
                pitzDaily under momentum-only scaling, one run per arm, all reaching `x_r/h` 8.0686:

                | stations | gamma | steps | cycles | ramp | target | pc | march |
                |---|---|---|---|---|---|---|---|
                | 16 | 1 | 27 | 227 | — | — | — | — |
                | 16 | 2 | 29 | 200 | 129 | 71 | — | — |
                | **16** | **3** | 31 | **196** | 112 | 84 | 47 s | 290 s |
                | 16 | 5 | 33 | 212 | 102 | 110 | — | — |
                | 16 | 10 | 64 | 329 | 100 | 229 | — | — |
                | 12 | 3 | 34 | 254 | 62 | 192 | 44 s | 328 s |

                **Coarsening loses here and wins there**, for one reason: a station change forces a full
                preconditioner re-materialize, which costs 11–16 s on `bfs3d` (24–34 % of that march,
                so buying fewer of them pays for a worse handover) and ~1.5 s on pitzDaily (44–47 s
                total, 13 %, so there is nothing to buy and the worse handover simply costs — 12
                stations cuts the ramp 112 → 62 and explodes the target 84 → 192). **Neither schedule
                transfers; sweep the pair on any new case.**
              - **The pitzDaily gain decomposes cleanly**, each step measured: both-blocks 24 stations
                `gamma = 1` **297** → momentum-only 24 **264** (−11 %, scaling) → momentum-only 16
                **227** (−14 %, stations) → `gamma = 3` **196** (−14 %, damping). 34 % in total, and
                `bfs3d` is 349 (rung ladder) → 150 by the same three levers.
              - **⚠️⚠️ RESTART CYCLES ARE THE WRONG INSTRUMENT FOR A SCHEDULE ON THIS CASE, TWICE OVER,
                and both ways they flatter the wrong arm.** *(a)* They cannot see preconditioner
                rebuilds, and a station change **forces a full re-materialize** — so the station count
                is literally a count of forced-full rebuilds, at 11-16 s each here. By cycles, 24
                stations wins (148 against 159); by the clock it loses by 25 %. On pitzDaily a rebuild
                is ~1.5 s and cycles do decide, which is why the two cases' station optima differ.
                *(b)* They omit **rejected attempts** — the count is recorded only on acceptance, so a
                step whose attempts were all rejected contributes `0`. `gamma = 8` therefore posts the
                fewest cycles of any arm (141) while being the second slowest (1186 s): its four
                escalations are four discarded shifted solves the count never mentions. Judge a
                schedule here on the march clock **and** the escalation count, with cycles third.
              - One run per arm. The pitzDaily arm is **deterministic** — an earlier configuration ran
                twice and gave 33 / 191 both times — so single runs separate arms there; that is a
                property of that case, not a general licence, and `bfs3d`'s own target-rung cost is on
                record as swinging +-275 s with how many escalations it needs (the `gamma = 2` arm takes
                none, which is part of why it is cheaper).
            - **It also lands the ladder's `x_r/h` (8.3611) where the both-blocks ramp landed 8.966.**
              Suggestive only: `atol = 1e-5` does not pin that metric to a cell on this mesh (see the
              entry above), and this is one run per arm. Do not quote it as an accuracy result.
            - One run, one schedule. **`bfs3d`'s own station sweep is untested**, and the sibling's sweep
              puts momentum-only's optimum at **16**, not the 24 used here — so 190 is this strategy's
              cost at a schedule fitted for a different one, and is probably not its best.
        - **The remaining 11-arm sweep is still orphaned** (225 / 245 / 246 / 261 / 269 / 286 cycles, the
          24-station interior optimum, the granularity and anchor-depth arms): all were taken pre-fix,
          and the one arm re-measured moved by 55 %, so the fitted station count in particular cannot be
          carried forward.
  - **⚠️ THE MEASURED `bfs3d` RESULT — the ramp WINS on this case too, 2026-09-10.** Both arms at commit
    `a403752` plus the seed fix, 23040 cells, `N_POINTS=2` (anchor Re/100), `RAMP_STATIONS=24`,
    `RAMP_STEPS=1`, `redamping` derived (= 1.0 at one step per station), `CflResidualDualTimeControl`
    `beta_start=0.5 beta_min=0.005 grow=1.5 backoff=2.0`, `INNER_STEPS/TOL` 5 / 1e-2, field split on,
    `simplesmooth` flow inverse, `jacobi` turbulence inverse, column reach 3/3/3/3/2/2, ILU(0) / 4
    sweeps / coarse 2000, `PC_BETA_FLOOR=0.05`, `REFRESH_ON_CYCLES=3`, `CYCLE_BUDGET=42`,
    `RETRY_ON_CYCLES/ALPHA` 10 / 0.01, stop `rtol=0, atol=1e-5` on the row-scaled residual, `k` wall
    `zerogradient`, compiled ILU(0) live.

    | arm | steps | restart cycles | retries | final \|R\| | mid-span `x_r/h` |
    |---|---|---|---|---|---|
    | rung ladder (shipped) | 59 | **349** | — | 1.830e-06 | 8.361 |
    | ramp, 24 x 1 | **39** | **240** | 3 | 9.887e-06 | 8.966 |

    **34 % fewer outer steps, 31 % fewer restart cycles.** Per phase: the ramp's 24 stations cost 122
    cycles and its target station 15 steps / **118**, against the ladder's target rung of 21 steps /
    **192** — which is the homotopy's whole thesis working, because the ramp *enters* the target at
    `|R|` **8.09e-04** where the ladder's target rung opened at **5.93e-02**, 73x worse, having just
    thrown away a converged rung.
    - ⚠️ **ONE RUN PER ARM — no spread.** The record already warns that this case's target-rung cost
      swings by +-275 s with how many β escalations it needs, and an earlier ladder pair ordered two
      totals the other way round. A 31 % cycle margin is well outside that, but it is one run.
    - ⚠️ **NO WALL-CLOCK RESULT, and the reason is worth knowing.** The ramp run spanned a **clamshell
      sleep** (lid closed 15:01:35, six minutes in; the machine then cycled through DarkWake for the
      rest). `run_case.sh`'s `caffeinate -ims` holds off *idle* sleep and cannot stop a lid close, which
      that script's own header already says. The case reported 4569 s and its march log 2101 s **because
      they use different clocks** — `march_log` uses `time.monotonic`, which on macOS does **not tick
      during sleep**, so a large gap between the two is a *sleep detector*. Counts are deterministic and
      unaffected; per this project's own rule the timings are discarded.
    - ⚠️ **THE TWO ARMS REPORT DIFFERENT `x_r/h` (8.966 vs 8.361) AND IT IS THE STOPPING BAR, NOT THE
      METHOD.** Established by comparing the checkpointed states rather than by arguing from a metric
      quantized to ~0.6h cell bins: the two converged states differ by rel L2 **3.75e-03**, while the
      **ladder's own** state moved **4.95e-03** between its step 57 (`|R|` 9.75e-05) and step 59
      (1.83e-06). The arms are closer to each other than the ladder was to itself two steps earlier, so
      they are converging to one root — the ramp merely stopped at 9.89e-06 where the ladder overshot
      to 1.83e-06.
      **The standing consequence is about the BAR: `atol = 1e-5` does not pin this case's headline
      quantity to a cell.** That indicts the ladder's 8.361 exactly as much as the ramp's 8.966 — the
      ladder's number is not more trustworthy, it is luckier. Anything comparing `x_r/h` across arms
      must converge tighter or report the metric with the residual it was measured at.

### The ramp became the default on both cases (2026-09-10/11)

  - **⚠️ IT IS THE DEFAULT ON BOTH CASES** — `validation/pitzdaily_openfoam/compare.py` since 2026-09-10
    (`PITZ_RAMP`) and `validation/bfs3d_openfoam/compare.py` since 2026-09-11 (`BFS3D_RAMP`), with `off`
    returning to `solve_reynolds_continuation` as the comparison arm rather than as a supported path.
    Flipped on the user's decision that the coupled march's upcoming work lands here and the ladder is
    not being developed further. `bfs3d` was the case most likely to disagree — it is where the ladder's
    rung structure was originally calibrated — and it does not: the ramp is 31 % cheaper there
    (240 against 349 cycles at both-blocks scaling, 190 at momentum-only, 148 with that case's damping
    optimum). **Still one run per arm on each case.** `bfs3d` additionally defaults to
    `BFS3D_RAMP_SCALE=flow`, i.e. momentum-only stations; pitzDaily does too (`scale: flow` in its `case.yaml`). The
    defaults are configured with
    `PITZ_RAMP_STATIONS` / `PITZ_RAMP_STEPS`. It anchors at `RATIO ** N_POINTS`, i.e. **the same span the
    ladder walks**, so the two arms differ in how the span is traversed and not in how far — and it
    builds its engine by calling the ladder arm's own `point_setup`, so they are preconditioned, logged
    and step-controlled identically. A second builder written beside it would drift a keyword at a time.

### How many rungs (2026-08-09)

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

### An absolute intermediate bar, and where a rung's cost goes

  - **An absolute intermediate bar, `intermediate=Convergence(atol=…)` with the target at `rtol=0` — stop
    every rung at one PHYSICAL standard, and prefer it over the relative bar on a row-scaled measure.** A
    relative intermediate `rtol` sets each rung's bar as a fraction of
    *that rung's own* starting residual, and under continuation every rung **re-bases `‖R₀‖`** — so a
    later rung, starting from a better seed, stops at a *looser absolute* residual than an earlier one
    already achieved. Worse, the row-scaled measure is already a fractional per-equation change, so
    dividing it again by `‖R₀‖` makes the bar a property of the initial guess rather than of the physics.
    An absolute intermediate `atol` (with `rtol=0`) gives every rung the same absolute bar.
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

### Reynolds-continuation ladder studies and a literature survey (2026-09-08)

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
        negative-direction updates** — the same rule and constant as the since-deleted global cap `positive_k_limit` (today's per-cell `positive_k_projection` keeps the 0.99). And Hicken et
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
