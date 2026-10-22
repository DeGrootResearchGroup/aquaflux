# Investigation log — `aquaflux/solve/` globalization

> ⚠️ **`coupled_continuation`, `coupled_lu_continuation`, `coupled_amg_continuation`, `lu_beta_tracking_refresh` and `amg_beta_tracking_refresh` no longer exist (deleted 2026-09-14, #371).** Entries below that name them are dated history. The coupled march is now one builder, `coupled_step`, given a preconditioner value (`BlockDiagonal` or `MaterializedJacobian(FieldSplit | BlockInverse)` (a `CompleteLu` inverse existed until 2026-10-11)), and a march that keeps its preconditioner current runs on a session (`open_session`) — see the rename table in `.claude/rules/turbulence.md`.

> Split out of `solve.md` / `solve-globalization.md` (2026-08-18). **Lives in `.claude/notes/`,
> outside the auto-loaded `.claude/rules/` tree, so it never auto-loads.** It holds the dated
> investigation behind the current globalization architecture — the residual-measure choice, the
> shift-basis probing, the SER schedule reversal, and the line-search behaviour — including entries
> that a later measurement corrected or superseded in place. See `solve-globalization.md` for the
> current, load-bearing architecture and defaults.

    - **First measurement of the convective basis: WORSE at a weakly-separated state, but NOT yet a fair
      test of its regime (`local_ts_ab.py`, 2026-07-24).** Probing both bases on pitzDaily checkpoints at
      `β ∈ {3, 12}` with a fresh preconditioner:

      | state | basis | β | cycles | α | ‖R‖ kept |
      |---|---|---|---|---|---|
      | known-good rel 0.052 | spectral (`w=1`) | 3 | 17 | **1.00** | **29.3 %** |
      | known-good rel 0.052 | convective (`w=0`) | 3 | 14 | 0.25 | 7.8 % |
      | plateau rel 0.032 | either | 3 / 12 | 13–163 | **0.001** | ~0 |

      At the productive state the convective basis **lowers** α (1.0 → 0.25) and the residual reduction
      (29 → 8 %), and does not help the flow or k blocks either (d(flow) −0.7 %, d(k) −0.5 %) — consistent
      with the dissipative diagonal being **load-bearing stabilization** on a wall-resolved,
      high-aspect-ratio mesh: the near-wall and recirculation cells are diffusion-controlled, so dropping
      their damping makes the coupled step overshoot and the line search clip harder. It is also cheaper
      per step at low β (14 vs 17 cycles) but degrades faster at high β (dropping the dissipative diagonal
      weakens diagonal dominance). **Caveat that keeps this open, not settled: the only state where steps
      are productive (rel 0.052) is barely separated, so this probe never exercised the developed
      recirculation the convective basis is *meant* for** — and the one genuinely separated state available
      (the plateau) is direction-limited (α = 0.001 for *every* basis and β), so it cannot discriminate
      between bases at all. Re-test at a state that is both **well separated and still productive** before
      concluding. Hence: shipped as an opt-in with the **default unchanged** (`w=1` = the historical `a_P`),
      not adopted and not withdrawn.
    - **RE-TESTED PROPERLY (2026-07-25, #28): the convective basis is NOT dominated — it is 2.2× better
      at the march's own operating point, with an optimum at Co ≈ 1.** ⚠️ **SUPERSEDED — this conclusion
      is wrong and was overturned on marches; see "the convective basis is DOMINATED" below. It is kept
      only as the third recorded instance of a single-step %/s sweep picking the wrong winner.** The
      earlier probe above rebuilt
      the continuation at the state it measured, and used a state that was barely separated; this one
      uses the **carried protocol** at a state that is both developed (`x_r/h` ~1.2) and productive
      (α = 1, 14 cycles), with β taken from the **segment-local** ratio. Residual reduction, %/s:

      | β (= 1/Co for w=0) | 0.5 | 1.0 | **1.79 ← the march** | 3.0 | 6.0 |
      |---|---|---|---|---|---|
      | `w = 1` (a_P, uniform relaxation) | **0.056** | 0.030 | 0.016 | 0.007 | 0.002 |
      | `w = 0` (convective local Δt) | 0.035 | **0.051** | **0.035** | 0.019 | 0.007 |

      The convective basis wins at every β ≥ 1 (1.7–3×) and needs fewer cycles there (10–11 vs 14). It
      has a genuine **interior optimum at Co ≈ 1** with symmetric fall-off — the canonical stability
      limit, i.e. a target with physical meaning that should transfer between cases. `a_P` has **no**
      optimum in range: its reduction follows a strict inverse law (`β × keep% ≈ 1.23` across a 12× span)
      with cycles flat below β ≈ 2, so it simply improves as damping falls and its best measured value
      is at the bottom of the sweep. Two roughly equivalent routes to ~3.5× over the march's 0.016 —
      lower β on `a_P` (0.056), or convective at Co ≈ 1 (0.051) — which do **not** compose (convective at
      β = 0.5 is 0.035).
    - **A non-uniform shift creates INTERIOR optima that the backtracking ladder cannot find — but the
    obvious fix is already REFUTED, so this is a description, not a lever.** With `w = 1` the ideal step
    length was `α = 1` at every β, so the powers-of-½ quantization cost nothing; with `w = 0` the ideal
    moves off a rung, and `backtracking_line_search` accepts the *first* rung that reduces and never asks
    whether a shorter step is better (a sufficient-decrease search, not a minimizing one), so some
    per-step residual reduction is left unclaimed. **⚠️ CONFLICT, settled — do not re-propose the
    minimizing search.** The "unclaimed reduction" argument was acted on: a minimizing search was built,
    measured and REVERTED, because a deeper residual per step bought far less recirculation development
    (see "THE LINE SEARCH TAKES THE LONGEST ADMISSIBLE STEP" below). The unclaimed-percentage figures
    that motivated it recorded no configuration and are deleted; the conflict is settled on the march
    evidence, which judges the physics rather than ‖R‖. Note the directional derivative is available
    almost free here, since the shifted solve gives `J δ = −R − β D δ` exactly.
    - **⚠️ MARCHES REVERSE THE SWEEP: on this case the productive lever is the damping LEVEL, not the
      basis (2026-07-25).** The %/s table above is single-step at one state, and it picked the wrong
      winner. Four cold-IC marches, all with the drift refresh, judged on the recirculation length:

      | arm | steps | **x_r/h** | k_peak | rel | α |
      |---|---|---|---|---|---|
      | a_P, β₀ = 2, shift carried (the shipped config) | 158 / 80 min | 1.67 | 3.10 | 8.1e-3 | 1.0 |
      | a_P, β₀ = 2, shift refreshable | 89 / 41 min | 1.22 | 2.04 | 1.3e-2 | 1.0 |
      | **a_P, β₀ = 0.5, shift refreshable** | **109 / 76 min** | **2.43** | **3.04** | **4.6e-3** | **1.0** |
      | convective at nominal Co ≈ 1, refreshable | 85 / 77 min | 1.07 | 2.58 | 1.5e-2 | **0.001 stalled** |

      At the time, `β₀ = 0.5` was the best this case had produced (a 46 % larger bubble than the shipped
      configuration). **⚠️ That is now SUPERSEDED and REVERSED — see the post-`a_P`-fix re-profile below;
      `β₀ = 0.5` is currently the worst of the three and `β₀ = 2` is right.** The convective arm did not
      merely lose, it **stalled** — α pinned at the 0.001 ladder sentinel with the residual frozen to
      five figures for four consecutive steps.
      - **The single-step sweep rated convective at Co ≈ 1 (0.051 %/s) level with a_P at β = 0.5 (0.056)
        and better than a_P at every β ≥ 1.** The marches say the opposite. That is the **second** time
        in one session that a single-state single-step ‖R‖ measurement chose the wrong winner (the first
        was the log-space ω shift, `.claude/notes/turbulence-log.md`, "The ω shift diagonal inherits ω's dynamic range"). **Treat %/s sweeps as a way to find
        candidates, never as a way to choose between them** — the choice needs a march judged on physics.
        (The 2026-07-26 re-profile is the **third** instance: the same sweep's `Co ≈ 1` optimum did not
        survive a constant-β march either.)
      - **This does NOT close out local timestepping (open question).** The Co ≈ 1 optimum was measured
        with the shift frozen at the cold initial condition, so the Courant number it optimized was
        *nominal*: `d_conv` was built from potential-flow mass fluxes, not the developed ones. With a
        refreshable shift `Co` finally means what it says, and the optimum can move — if developed fluxes
        exceed cold-IC fluxes, nominal Co ≈ 1 corresponded to a larger *actual* Co, which would leave the
        refreshable arm under-damped and is consistent with the stall. Re-sweep Co **on marches** with the
        refreshable shift before concluding; do not reuse the frozen-shift optimum.
        **— CLOSED 2026-07-26 by the re-profile below: re-swept, and the under-damping explanation is
        refuted. The pure convective basis is dominated at every damping level tested.**
    - **⚠️⚠️ THE EUCLIDEAN ‖R‖ MIS-RANKS STATES — a converged field scores WORSE than a badly wrong one
      (measured 2026-07-26). This invalidates ‖R‖-based comparisons throughout this file; read this
      before trusting any of them.** Raw norm against a scale-free per-cell measure, four states, same
      mesh and model:

      | state | raw ‖R‖ | **`\|R_ω\|/ω` median** | `\|R_k\|/k` | flow | x_r/h |
      |---|---|---|---|---|---|
      | cold initial condition | 286.3 | 4.54e-05 | 5.73e-05 | 2.52e-01 | 0.00 |
      | cold march, step 90 | **3.68** | 2.44e-05 | 3.41e-05 | 5.21e-02 | 1.22 |
      | OpenFOAM reference (converged) | 21.2 | **3.48e-06** | 3.06e-06 | 6.74e-03 | 7.74 |
      | our own warm-started root | 4.57 | **2.32e-06** | 2.84e-06 | 1.27e-03 | 8.07 |

      **The scale-free measure ranks all four correctly; the raw norm inverts the middle two**, rating a
      state whose recirculation is six times too short (3.68) above both converged fields. Two
      compounding causes, both measured:
      - **The ω residual is not dimensionless.** OpenFOAM's field is converged to a *relative* imbalance
        of 3.5e-6 — 7× better than the cold march's 2.4e-5 — but its ω is sharp and developed, so the
        same relative error yields a far larger absolute residual. Raw ω residuals cannot be compared
        across states with different turbulence levels, which is exactly what a march does.
      - **At converged states the ω L2 is a TEN-CELL statistic.** At the reference the top **1** cell
        carries **41.9 %** of ‖R_ω‖² and the top 10 carry **75.9 %** (the sharp near-wall peaks); at the
        under-developed marched state it is spread out (top 1 = 1.1 %, top 10 = 5.4 %). A metric that
        concentrates into a handful of cells precisely as the solution becomes correct is backwards.
      - **The flow block's raw residual already ranks correctly** (2.5e-1 → 5.2e-2 → 6.7e-3 → 1.3e-3).
        Only k and ω are mis-scaled — and ω is ~100 % of the norm.
      **⚠️ CORRECTION (same day): the "mis-ranks states" framing above is OVERSTATED — do not repeat
      it.** The OpenFOAM field is *not* a root of these equations (another discretization, a different
      wall treatment, an instantaneous snapshot of an unsteady shear layer), so a residual measure
      rating it by its own nonzero imbalance is **correct behaviour, not a defect**. Demanding that a
      measure rank a foreign field as converged is a broken test, and the row-equilibrated measure does
      not do it either (cold march 1.16e-2 vs that field's 1.34e-2, scales rebuilt per state). On
      states that *are* ours both measures already rank correctly: our warm-started root scores best
      under the raw norm (0.98 vs the march's 3.68) as well as the scaled one.
      **What actually survives, and it is enough:** the raw norm is ~100 % ω, so it does not *report*
      flow progress. Measured directly at a state known to be near the correct root — the warm-start
      run — raw ‖R‖ moved 13.87 → 13.70 (**−0.2 %, reading as stalled**) while the flow block fell
      6.30e-3 → 5.03e-3 (**−20 %**). That is what starves the line search and SER of flow information,
      and it is the real case for equilibration; the block spread narrows from ~100 %-ω to within ~10×
      across blocks.
      **Consequence (binding):** SER's β ramp, the line-search acceptance, the divergence guard, the
      convergence test, and every β / basis / preconditioner comparison recorded in this file are
      computed on a measure that is ~100 % one block. This is the concrete,
      quantified case for row equilibration (#29) — divide each row by its own scale so `|R_ω|/ω` is
      what is measured — and for per-block reporting (#24). Note this is **not** the earlier claim that
      "the answer is unreachable by descent": that was measured against the OpenFOAM field as the
      endpoint and was **wrong**, because that field is another discretization's instantaneous snapshot
      and not a root of these equations. Our own root scores 2.77–4.57, *below* the cold march's 3.68 —
      the landscape around the true solution is fine.
    - **⚠️ THE COLD-START CRAWL IS A REACHABILITY PROBLEM, NOT A WRONG ROOT — settled 2026-07-26 by a
      warm start, and this reframes every globalization result below.** A cold march reaches only
      `x_r/h` 1.22 in 91 steps against the reference's 7.74, which is consistent with two completely
      different stories: the solver cannot *reach* the right root, or it converges correctly to its
      *own* root, which has a short bubble. Starting **from** the time-accurate reference separates
      them, and the answer is unambiguous — the root is ours and it is in the right place:

      | arm (near-wall ω blend) | x_r/h | k_peak | ‖R‖ from → after 5-6 steps | flow | k |
      |---|---|---|---|---|---|
      | shipped power mean, `p = 2` | 7.74 → **7.82** | 5.03 | 21.20 → **13.70** | 6.7e-3 → 5.0e-3 | 2.0e-2 → 1.5e-2 |
      | `max` limit, `p = 60` | 7.74 → **7.99** | 5.03 | 20.67 → **6.70** | 6.7e-3 → **1.8e-3** | 1.2e-2 → **2.6e-3** |

      Every block descends and the reattachment holds. **So no closure/model work is required to get
      the bubble** — the closure was never the problem (which also confirms the older three-way
      verification recorded in `.claude/notes/turbulence-log.md`, "pitzDaily reference and the max blend"), and the entire remaining gap is the
      solver's inability to travel from a cold start to a root it is perfectly happy to sit on.
      - **Consequence for what to build:** de-emphasizing ω in the *measure* is now a justified lever
        rather than a guess, because a reachable correct root is known to exist. The target is
        quantitative: along the straight segment from a cold-march state to the reference the total
        residual **peaks at ~5.9×** while the **flow block falls monotonically 7.7×** (5.2e-2 →
        6.7e-3). Any measure that lets the march traverse that must stop ω's few-cell L2 from vetoing
        flow progress. The acceptance rule currently tolerates `1.107×` (`RelaxedFarFromRoot` at
        rel 1.3e-2), i.e. **~5× too little**.
      - **The ω-dominance pathology is visible even at the root.** In the `p = 2` arm, ‖R‖ moves
        13.87 → 13.77 → 13.74 → 13.72 → 13.70 (−0.2 % per checkpoint, reading as "stalled") while the
        flow block falls 6.30e-3 → 5.03e-3 (−20 %). Since this state is *known* to be near the correct
        root, that cannot be confused with a genuine stall: it is the global norm failing to report
        flow convergence. This is the concrete case for per-block reporting (#24) and row
        equilibration (#29).
      - **A `max` near-wall blend is better on CONVERGENCE, not only on accuracy.** `.claude/notes/turbulence-log.md` ("pitzDaily reference and the max blend")
        justified `p → ∞` on agreement grounds (per-wall-cell `|R_ω|/ω` 3500× smaller) and left the
        default open as "a model decision". This adds an independent argument: from the same start it
        reaches **2× the residual depth** (6.70 vs 13.70) and a **2.8× lower flow residual** in the same
        number of steps, while `p = 2` flattens. Note the ω **L2 at the reference** barely moves between
        blends (21.20 vs 20.67, 2.5 %) — that norm is a handful of ω~1e5 cells and is not the quantity
        that discriminates, which is itself a caution against judging the blend on ‖R‖.
      - **Honest caveat on the number:** `x_r/h` does not settle exactly on OpenFOAM's 7.74. It creeps
        to 7.82 (`p = 2`) and 7.99 (`p = 60`, still rising when measured) — a ~1–3 % longer bubble.
        Report that as a solver-to-solver difference, not as a match; it is expected for a
        wall-resolving closure on a wall-function mesh, and it is small beside the cold-start gap.
      - **Also measured: the residual at the reference is UNCHANGED by the 2026-07-25/26 fixes.**
        Current code gives flow 6.74e-3, k 1.99e-2, ω 21.2 against the 2026-07-24 record's ~6e-3,
        ~1e-2, ~20. The fixation-row fix and both `a_P` fixes mattered for the **march**, not for the
        root.
      - **Trap that cost an hour here (binding for any future reference comparison):**
        `compare.read_openfoam_reference()` used to read `runs/kwsst`, the **corrupt steady** case, so a
        probe calling it silently inherited the inlet checkerboard — ω spanning 0.03 to 1.15e8, with
        **ten cells carrying 100 % of the ω residual** and a total ‖R‖ of 4.1e8. That produced a
        spurious "10⁸ residual ridge blocks the path" and a spurious `cos(step, error) = −0.087`
        (the true value against the transient field is **+0.13**, i.e. weakly *aligned*). The loader now
        reads `of_transient/0.14`. **Sanity-check any reference measurement against the recorded
        ‖R‖ ≈ 20 before drawing conclusions from it.**
    - **⚠️ THE CRAWL IS A CORRECT PSEUDO-TIME INTEGRATION OF A GENUINELY LONG TRANSIENT — measured
      2026-07-27, and it re-scopes the "de-emphasize ω in the measure" lever above.** The `a_P` shift is
      backward-Euler local time stepping: `β·a_P` on the diagonal is the transient term `ρV/Δτ`, so the
      per-step pseudo-time is `Δτ = α·V/(β·a_P)` (α the accepted line-search factor, ρ = 1). Accumulating
      that per cell over two stored cold marches (`profile_base`, β ≈ 1.9; `basis_march_aP05`, β₀ = 0.5),
      sampled in the recirculation region behind the step, settles what the reachability crawl actually is:
      - **`a_P` is ~constant (~8.0e-3) for the whole march.** The potential-flow seed already carries
        free-stream-magnitude velocity, so the momentum diagonal barely moves — hence `Δτ` per step is
        *fixed and tiny* (~6e-5 s at the median cell), regardless of how the bubble develops.
      - **`x_r/h` grows smoothly, monotonically, and decelerating with accumulated pseudo-time in both
        arms — no stall, no reversal.** The step direction is never the problem; every step buys real
        bubble. `base` reaches `x_r/h` 1.22 at ~5 ms of bubble-median pseudo-time in 90 steps; the
        β₀ = 0.5 arm reaches 2.43 at ~28 ms in 109 steps. Physical yardstick: the free-stream
        flow-through of the reattached bubble length (7.74·h ≈ 0.20 m at 10 m/s) is ~20 ms, and `base`'s
        growth extrapolates to reach 7.74 near **~55 ms ≈ ~800 steps at this `Δτ`**. So the march has
        elapsed only a small fraction of the transient — the crawl is *insufficient elapsed pseudo-time*,
        not a wrong direction, a bad merit function, or a stuck state.
      - **Lower β = larger backward-Euler step = further per step (2.43 vs 1.22) — this confirms `Δτ` is
        the lever.** The two arms trace the same qualitative decelerating growth but do **not** collapse
        onto one `x_r/h`(pseudo-time) curve: the β₀ = 0.5 arm sits at ~1.6–2× more pseudo-time per unit
        bubble, because a larger implicit step integrates the transient more coarsely and its pseudo-time
        bookkeeping overstates true transient progress. The small-step arm is the truer `x_r(t)`; do not
        read the imperfect collapse as a defect.
      - **CONSEQUENCE (binding): no merit function, acceptance rule, filter, or shift *basis* changes
        this** — every one of those is a direction/measure lever, and the direction is fine. The only
        levers are the **effective `Δτ` per step** (`α·V/(β·a_P)` — a larger stable step) or a **different
        homotopy/seed nearer the developed bubble** (physical continuation in Re or a `ν_t` ramp, a
        coarse-grid or eddy-viscosity-augmented start). A measure change cannot *manufacture* pseudo-time,
        which bounds the "de-emphasize ω in the measure" target above (#24/#29): worth it for readable
        per-block reporting, **not** as the reachability fix it was framed as.
      - **WHAT LIMITS `Δτ`: the cold-start β floor is a NONLINEARITY, and diffusion continuation lifts it
        — measured 2026-07-27. *(harness not in the repository — this finding cannot be re-adjudicated as recorded)*** A single
        shifted cold step at the target Re = 25000 is stable at β = 2 (α = 1, ‖R‖ ×0.49) and β = 0.5
        (α = 0.5) but **blows up at β = 0.25** (ω → 5.6e32, no reducing rung) — the recorded floor. Raising
        the molecular viscosity (a clean Reynolds continuation, self-consistent seed, no state
        perturbation) removes it: at Re = 2500, β = 0.5 goes α = 0.5 → **1** and β = 0.25 becomes finite
        and productive (α = 0.5, ‖R‖ ×0.49); at Re = 250, β = 0.5 takes a near-Newton step (‖R‖ **×0.045**,
        22× in one step). So the floor is set by the convective nonlinearity, and reducing it (diffusion
        homotopy) buys a lower β = a larger `Δτ` from step 1 — the automatable, knob-light lever (one
        scalar that **dissolves at the target Re**, like the shift, so the root is unchanged).
      - **A `ν_t` seed applied by perturbing the k/ω *state* BACKFIRES — do not.** Scaling ω down to raise
        `ν_t` unbalances the ω transport equation, and since ‖R‖ is ~100 % ω the coupled step then fights
        that artificial deficit: measured α = 0 (no reducing rung) at β = 2 where the unperturbed state
        gives α = 1. An eddy-viscosity seed must add diffusion to the **momentum closure** (a `μ_eff`
        floor, ramped out), *not* to the k/ω fields — i.e. it is a spatially-varying diffusion
        continuation, the same family as the Reynolds ramp above.
      - **This is the `β × travel` finding seen in the residual, and it explains why ‖R‖ points opposite
        to the physics.** On every shifted row `R(φ+δ) ≈ −βDδ = −(ρV/Δτ)δ ≈ −ρV·(dφ/dt)` — the *physical
        unsteady term*, nonzero for the entire transient and independent of step size. The equilibrated
        residual therefore literally cannot fall until the transient completes; it is behaving exactly as
        an unsteady residual should, which is why judging on `x_r/h` (never a residual) is mandatory here.
      - **The `of_transient` reference has NO bubble-growth curve — it was restarted from the developed
        steady field.** `of_transient/0/U` carries `location "2000"` in its header (copied from the steady
        run's converged step), so `x_r/h` ≈ 7.74 at *every* written time including t = 0. The transient
        confirms the developed state is stable; it is **not** a growth transient and cannot be overlaid
        against the march's `x_r` vs pseudo-time. Treat 7.74 as the asymptote only.
    - **⚠️ RE-PROFILED AFTER THE `a_P` FIX (2026-07-26) — the two conclusions above REVERSE. Read this
      bullet, not them.** The flux-continuous (harmonic) face viscosity and the wall-model boundary
      viscosity changed `a_P` itself, and the shift is `β·a_P`, so **every β calibration measured before
      that fix is void** — the harmonic mean is ≤ the arithmetic one it replaced, so the same β now buys
      *less* damping and the optimal β moves **up**. Three cold-IC marches, shipped `solve_coupled`,
      drift refresh, judged on the recirculation length:

      | arm | steps | **x_r/h** | k_peak | rel | α (tail) | cyc/step |
      |---|---|---|---|---|---|---|
      | **`a_P`, β₀ = 2 (the shipped default)** | **67** | **0.99** | 1.61 | **1.4e-2** | **1.00** | **12.5** |
      | `a_P`, β₀ = 0.5 (the former "best") | 16 | 0.39 | 1.41 | 9.5e-2 | **0.13** | 29.0 |
      | convective, Co adapted from α | 2 | — | — | 8.9e-1 | 0.125 | 22 → killed |

      `β₀ = 0.5` is now **under-damped and stalling** (α 0.13, 29 cycles/step, bubble frozen at 0.39),
      exactly the failure the convective arm shows — and for the same reason, too little effective
      damping. **Take `β₀ = 2`.**
    - **The convective basis (`w = 0`) is DOMINATED — settled by a controlled 2×2 plus a β sweep, do not
      re-open on a %/s sweep (2026-07-26).** Three steps from the same cold IC at **constant** β
      (`exponent = 0`, so β is genuinely fixed and the arms are compared at equal damping, not equal
      residual history). The probe reproduces the real march bit-for-bit at step 0, which is the harness
      validation that must precede any such claim:

      | basis | β | cyc 0/1/2 | α 0/1/2 | rel after 3 steps |
      |---|---|---|---|---|
      | **`a_P`** | 2 | 15 / 14 / 13 | **1.000 / 1.000 / 1.000** | **0.2995** |
      | convective | 1 | 18 / 14 / 13 | 0.125 / 0.125 / 0.125 | 0.8035 |
      | convective | 3.3 (= matched effective damping) | 36 / 22 / 24 | 0.250 / 0.125 / **0.0039** | 0.8980 |

      The convective basis is clipped at **every** step while `a_P` takes full steps at the same cost,
      and at *matched* effective damping it is worse still and collapses into the ladder by step 2
      (α → 0.0039 → 0.0020, the 0.001 sentinel again). Three candidate explanations were each proposed
      and each **refuted by measurement** — record them so they are not re-proposed:
      - *Preconditioner inconsistency* (the MSIMPLE Schur ignores the shift, which for a non-uniform
        basis is a spatially-varying error): refuted by the 2×2 below. **Issue #163, closed as
        refuted.**
      - *Damping level / wrong Co calibration* (the convective diagonal is only ~0.61 of `a_P`, so
        "Co = 1" under-damps 3×): refuted by the β = 3.3 row — matching effective damping does not
        recover α, and makes progress *worse*.
      - *Weakened diagonal dominance / near-wall cells left undamped*: refuted directly — `a_P + βd` is
        **more** diagonally dominant than `a_P`, and the measured convective share bottoms out at
        p1 = 0.30 (never near zero), with the least-damped cells **mid-channel**, not at the wall.
      - *The recirculation is left undamped* (a convective-only `Δt → ∞` where the mass flux vanishes,
        i.e. no damping in the most nonlinear region): refuted, and the correlation runs the **other**
        way. At a developed state (`x_r/h` 1.22) the reversed-flow cells have a **higher** convective
        share than the forward-flow ones (median 0.778 vs 0.652), and they are strongly
        *under*-represented among the least-damped — 0.00× the base rate in the bottom 1 % by share,
        0.14× in the bottom 5 %. The least-damped 2 % are at `x/h ≈ +10.8`, `y/h ≈ −0.10`, moving at
        **7.03 m/s against a 5.10 m/s domain median**: the fast downstream core, where the developed
        eddy viscosity makes the viscous diagonal dominate.
      **No mechanism is offered — four were proposed and all four were refuted by measurement. The
      empirical result stands without one; do not add a fifth without a measurement that discriminates
      it.** What *is* established: the shipped `w = 1` basis is the classical local time step (the shift
      `β a_P` is `V/Δt` with `Δt = Co·V/λ`, `Co = 1/β`, `λ` the **combined** convective + viscous
      spectral radius — Blazek's form), and it holds `α = 1.0` for 90+ consecutive steps. `w = 0` is
      that same formula with the viscous stability limit deleted, on a mesh where the developed `ν_t`
      makes the viscous half the **larger** one almost everywhere (share median 0.66). So this is not
      evidence against local timestepping; the default *is* local timestepping and it is what works.
    - **The Schur's blindness to the shift is NOT a defect — measured, do not "fix" it (2026-07-26,
      #163).** `apply_at` feeds the velocity block the shifted diagonal `a_P + β d`, while the MSIMPLE
      Schur uses `Q̂/k` calibrated from the **un-shifted** diagonal, i.e. it ignores the shift entirely.
      That looks like an inconsistency, and for a non-uniform basis the discrepancy is spatially varying
      (`1/(1 + β·share)`, share 0.30–0.97) rather than a global scalar. It costs nothing. A 2×2 at fixed
      β, varying only `schur_scaling` (`simple` uses the shifted `a_p` and is consistent by
      construction):

      | basis | `msimple` (shift-blind) | `simple` (consistent) | ratio | α (both) | rel (both) |
      |---|---|---|---|---|---|
      | `a_P`, β = 2 | **15** cyc | 36 cyc | 2.4× | 1.000 | 4.8530e-01 |
      | convective, β = 1 | **18** cyc | 34 cyc | 1.9× | 0.125 | 9.2719e-01 |

      Within each basis `α` and the residual ratio are **bit-identical across all three steps measured**
      — the "a preconditioner changes cost, not the converged step" property, which also confirms these
      solves genuinely converge. There is **no interaction**: the consistent Schur is uniformly ~2×
      worse, and *less* bad on the convective basis (1.9× vs 2.4×) — the opposite of the hypothesis.
      `Ŝ` is an approximation chosen for **spectral quality**, not a derivation of the true Schur
      complement; MSIMPLE's whole premise is replacing `a_P` with a velocity-independent mass-matrix
      stand-in, so being more faithful to `(A + βD)⁻¹` does not make it a better preconditioner. This
      also confirms the earlier "shift-consistent Schur is strictly worse at every β" finding **does**
      transfer to a non-uniform basis, contrary to what was argued when #163 was filed.
    - **⚠️ CONFLICT — "neither α nor the cycle count can serve as a controller target on this problem" is
    contradicted by the shipped default; do not act on either side without re-measuring.** One side: a
    single-step β/basis sweep found α and the cycle count constant across its whole range while the
    efficiency varied, so only residual reduction per unit time discriminated *(configuration not
    recorded — no preconditioner, forward solver or tolerance — and taken under the superseded
    ω-dominated norm; re-measure before relying on it)*. The other side: `DualTimeControl`, the
    **α-driven** ramp, is the shipped default for a dual-time observed march and is the arm that reaches
    a developed recirculation, while the residual-keyed control pins β on the flat `β×travel` plateau.
    The likely reconciliation is that a *single-step* sweep at one state cannot see the α signal a
    dual-time inner loop produces — but that is an inference, not a measurement.
    - **⚠️ SUSPECT — the "plateau is a step-DIRECTION problem" conclusion was measured through a broken
      preconditioner (see the fixation-row/`1/ω` bug below) and a corrected re-measurement CONTRADICTS
      part of it. Re-derive before relying on any of it (#31).** As originally written: every basis/β
      combination at rel 0.032 gave α = 0.001 and zero descent, so neither a preconditioner, nor a
      per-block β, nor a shift basis could move it — the argument being that a **preconditioner** only
      changes Krylov cycles, since for fixed `J`, `β`, `d`, `R` the shifted step `δ` is unique regardless
      of `M`.
      - **That uniqueness argument is only valid for a CONVERGED linear solve, and the coupled forward
        solver is deliberately inexact** (the block family's forward solve ran at `rtol = 1e-3` then;
        that symbol is gone and the default is now `forward_rtol = 0.3` in the row-scaled measure). At a finite
        tolerance `δ` depends on `M`, so a preconditioner *can* change α. Measured directly: refreshing
        the scalar AMGs mid-march on the pitzDaily cold-IC run took α from **0.5 → 1.0** (sustained over
        the following steps) while cutting cycles 53 → 10. So "a preconditioner cannot change α" is false
        as stated here; state it as "cannot change the *converged* `δ`, hence not the fixed point".
      - The α = 0.001 observations themselves came from probes at a state reached through the
        `1/ω` preconditioner mis-scaling, i.e. through solves that were not converging. Treat the whole
        plateau diagnosis as unverified until re-measured on a cold-IC march with the fixed code.
      - What is *not* in doubt: a **shift basis** only redistributes damping, and altering the operator
        itself (the pseudo-time shift, a grad-div / augmented-Lagrangian augmentation vanishing at
        `∇·u = 0`, or physical continuation) is what changes `δ` at convergence.

  - **The `growth` parameter is NOT a performance regression — hypothesis raised, then DISPROVEN by
    measurement (2026-07-25, settled; do not re-open on the original evidence).** A
    `newton_march(..., max_steps=1)` on the pitzDaily *plateau* state ran **27 min 42 s** without
    returning, shortly after `backtracking_line_search` gained its `growth` argument, and the
    coincidence was recorded here as a suspected regression with a traced-bound hypothesis. Both the
    hypothesis and the attribution are wrong:
    - **`admissible` was never inside the loop.** `admissible = growth * reference_norm` is computed
      *outside* the ladder body (`implicit.py`), so the bound is a loop-invariant closure capture
      exactly as the bare `reference_norm` was before. The "compute it outside" fix that was filed as
      a candidate is already the code as written.
    - **The default bound is not even traced.** `MonotoneLineSearch.growth` returns a **concrete**
      `jnp.asarray(1.0)`, not a tracer, so under the default the comparison is structurally identical
      to the pre-change one. The recorded observation that the slowdown appeared *with the monotone
      schedule active* was read as evidence **for** the traced-bound hypothesis; it is evidence
      against it.
    - **The decisive measurement.** A jitted `_march_step` at the plateau with **`max_escalations = 0`
      and `MonotoneLineSearch`** — i.e. exactly ONE shifted linear solve, default growth, no
      escalation ladder, strictly less work than the step that took 27 min — ran **> 57 min without
      returning**. Since that configuration contains neither the escalation ladder nor any non-default
      growth, neither can be the cost.
    - **The ladder was never a plausible candidate on size grounds either:** one coupled residual
      evaluation is ~0.3 s, so all 11 rungs cost ~3 s against a 27-minute step.
    - **What the cost actually was: a PRECONDITIONER BUG introduced by the fixation-row change in the
      same PR — found and fixed the same day.** The scalar block's frozen preconditioner was rescaled
      by `1/(dφ/dw) = 1/ω` on *every* row, including the 472 near-wall ω **fixation** rows, whose row
      (`LogRatioRow`, written in the solved variable) has derivative **1** against the frozen
      operator's unit identity row — a `1e-5` eigenvalue cluster that stalls GMRES. So the coupled
      solves were not converging and ground to their step cap. Fixed by giving each `FixationRow` its
      own derivative; measured 27× better linear residual at fixed cycle count (see
      `.claude/rules/turbulence.md` for the full finding).
      **The chain of three wrong attributions is the lesson:** the `growth` argument, the fixation row
      itself, and nested `jax.grad` in the residual were each blamed in turn because all three landed
      in the same window. What discriminated was measuring *components* rather than the whole step —
      the residual is 8.1 ms, `jvp/residual` is **1.5×** (healthy AD, killing the nested-grad theory),
      and one 120-vector restart cycle is ~1.5 s, so a healthy solve is seconds. Any step costing
      minutes therefore had to be **iteration count**, not per-matvec cost — which pointed straight at
      the preconditioner and away from everything else.
    - **Staleness is still real but was NOT the main term here.** The per-solve wall-time figures once
    quoted here (and elsewhere in this file) named no preconditioner, forward solver, restart or state,
    so they are deleted rather than defended. Rebuild-vs-carry belongs in the refresh-trigger
    calibration (#17) on a *cold-IC* march; re-measure it now that the solves converge, since the
    pre-fix carried-vs-rebuilt comparison (#31) was taken through the broken preconditioner and cannot
    be trusted.
    - **Methodological trap this cost an hour to learn (binding for future probes):** timing a
      `solve_linear` **eagerly** measures nothing comparable to the march, which runs the whole step
      inside one `eqx.filter_jit`; eager JAX dispatches each Krylov operation separately. An eager
      version of this same probe was still inside a single solve after 60 min of busy CPU. Always
      time the jitted `_march_step`, and prefer *differencing two configurations* over instrumenting
      inside the compiled region.
    - **⚠️ THE CARRIED PROTOCOL — how to probe a coupled step at all (binding; four separate sweeps
      were invalidated by getting this wrong in one session).** A probe that loads a checkpoint and
      calls `coupled_continuation(coupled, checkpoint)` builds a **self-consistent** (state, shift,
      preconditioner) triple. **The march never occupies that configuration**: it freezes the shift at
      the cold IC and carries it through every refresh, so its shift always *lags* the state. The
      difference is not academic — at the same state and the same β, a rebuilt continuation took
      **148 cycles and found no descent** where the march takes **14 cycles at α = 1 and descends**.
      Reproduce a refresh instead:
      ```
      cold_cont = coupled_continuation(coupled, cold_ic, method=...)
      cont      = coupled_continuation(coupled, state, method=..., reuse=cold_cont.shift_policy)
      ```
      **Validate the harness before trusting it**: the `reuse=` build at the march's own β must
      reproduce the march's cycle count and α from its log. When it finally did (14 cycles, α = 1.0000,
      +0.677 %), every earlier number in that session turned out to have been measuring something else.
    - **Take β from the SEGMENT-LOCAL residual ratio, never the global one (this is what made three of
      those four sweeps wrong).** SER's reference is reset at every refresh, so on a refreshing march
      β stays pinned near β₀ = 2 rather than decaying — see the `RelaxationSchedule` section. Computing
      "the march's operating β" from the global ratio gave 0.022 where the true value was 1.79, an 80×
      error that silently produced an apparently-solid "ascent at every Courant number" result.
    - **A median is the wrong statistic for a shift diagonal.** Comparing rebuilt against carried gave
      a median ratio of 0.96 — "no effect" — while the effect was a >2× tail over 15 % of ω cells,
      reaching 24×. Report p99/max and the fraction above 2×, per block; a whole-state statistic also
      hides that the pressure block is identically zero (20 % of the vector) and dilutes everything.
    - **Do not probe step cost at the plateau.** Every configuration measured there costs ~1 h/step
      because the state is direction-limited (α = 0.001 at every β and shift basis) *and* maximally
      stale for a carried preconditioner. Cost questions belong on a cold-IC march, where steps are
      accepted on the first attempt.
  - **⚠️⚠️ THE EQUILIBRATED MEASURE BARELY FALLS ON A MARCH THE EUCLIDEAN NORM LOVES — the most
    consequential measurement of 2026-07-27. Read this before designing around either measure.**
    Evaluating the row-equilibrated measure along the *default* march's own checkpoints (the march whose
    Euclidean residual falls 78×, from 2.86e2 to 3.68):

    | step | Euclidean | equilibrated | u0 | u1 | cont | k | ω | x_r/h |
    |---|---|---|---|---|---|---|---|---|
    | cold | 2.86e+2 | 2.229e-2 | 4.76e-3 | 1.80e-3 | 2.30e-3 | 1.64e-2 | 1.41e-2 | 0.00 |
    | 25 | 3.96e+1 | 2.175e-2 | 5.99e-3 | 6.21e-3 | 1.14e-3 | 1.14e-2 | 1.64e-2 | 0.05 |
    | 90 | 3.68e+0 | 1.158e-2 | **5.23e-3** | **4.61e-3** | 6.84e-7 | 7.55e-3 | 5.34e-3 | 1.22 |

    **The Euclidean norm falls 78×; the equilibrated measure falls 1.9×.** Composition: continuity
    improves ~3000× (a negligible absolute contributor), k and ω ~2.5× each, and the **velocity blocks
    get WORSE** — one component 1.80e-3 → 4.61e-3, **2.6× worse** — over a march the Euclidean norm
    reports as converging.
    - **⚠️⚠️ THE TABLE ABOVE IS AN ARTIFACT OF THE MEASURE'S OWN CONSTRUCTION — MEASURED 2026-07-27,
      and it supersedes the reading that stood here before.** On every row the shift owns, the
      equilibrated measure after a full step is **`β × per-step travel`, not a distance to the root.**
      `coupled_scaled_norm` takes its velocity/k/ω row scales from `shift_policy.shift_term(state)
      .diagonal` — *the very array the shift multiplies* — so with `(J + βD)δ = −R` giving
      `R(φ+δ) = −βDδ + O(‖δ‖²)`, the equilibrated row is exactly `β|δᵢ|`. Continuity carries `D_c = 0`
      (the shift packs `jnp.zeros(n_cells)` on pressure), so it is **annihilated to first order**
      whatever the physics does. Measured against real shifted solves at real march checkpoints
      (harness not in the repository), actual ÷ predicted `βDδ` floor:

      | state | β = 2 | β = 1 | β = 0.25 |
      |---|---|---|---|
      | cold | 0.995–1.008 | 0.975–1.060 | *diverges, see below* |
      | g0020 | 1.013–1.038 | 1.049–1.189 | *NaN, see below* |
      | g0045 | **1.000 ×4** | 0.999–1.005 | 0.994–1.046 |
      | g0090 | **1.000 ×4** | 1.000–1.003 | 0.999–1.023 |

      Continuity's floor is *exactly* zero everywhere; its actual residual after the step is 5.9e-7 at
      g0090 against 6.8e-7 before. Nonlinear defect is 0.1–0.2 % of the block value at the developed
      states, and the Krylov residual is 1e-11–1e-13 throughout — so this is neither nonlinear
      truncation nor solver inexactness. **The identity is β-independent**: it holds across a factor of
      eight in β, which is much stronger evidence than the operating point alone.
      **Therefore: continuity's ~3000× is first-order annihilation of an unshifted row; the velocity
      blocks' 2.6× "degradation" is the steps getting BIGGER. Neither is a statement about the flow —
      stop citing them as one.** The measure cannot fall below a floor proportional to `β ×` step while
      β ≈ 2, which is the whole explanation of "equilibrated stalls at 1.9× while Euclidean falls 78×":
      the Euclidean norm has no β-proportional floor. Both measures were correct about what they
      actually measure; neither was measuring convergence.
    - **⚠️ CORRECTION (2026-07-27, from reading the reference coupled p–U C++): the MEASURE is sound —
      the `β×travel` is aquaflux feeding it the wrong residual, not a flaw in the measure's
      construction.** The reference code's scaled-residual convergence measure is the *same* construction
      (divide each row by its diagonal coefficient, then normalize by field magnitude) and is robust
      there. What differs is the residual each divides:
      - **The reference measures the residual of the equation it actually solves** — `transient + flux` =
        `ρV/Δτ·(φ − φ⁰) + flux(φ)`, scaled by `a_P + ρV/Δτ`, read as the *initial* residual before the
        field update (the standard finite-volume convergence judge). The pseudo-time term is present in
        the residual, the matrix diagonal, **and** the scaling, all three consistently, and its reference
        `φ⁰` is **held fixed across the inner iterations of a timestep**. That residual is `O(‖δ‖²)` after
        a Newton step and collapses to the pure steady imbalance at each timestep's start — so it
        converges.
      - **aquaflux measures the bare *steady* residual `R`** while the shift `βD` is on the **Jacobian
        only** (`(J + βD)δ = −R`, `R` the unshifted steady residual — `continuation.py`), and the shift
        reference **resets to the previous iterate every step**. Both take the *same* Newton step on
        `G = R + βD(φ − φ_k)`; the reference measures `G(φ⁺) = O(‖δ‖²)`, aquaflux measures
        `R(φ⁺) = G(φ⁺) − βDδ = −βDδ` — the `β×travel`. The missing `−βDδ` is exactly the pseudo-time term
        the reference keeps in its residual and aquaflux drops.
      So the earlier conclusion ("valid convergence test near the root, not a merit function far from
      it") mislocated the fault: the measure is **sound**, and it is being fed the steady residual of a
      **single-step PTC with a per-step reference** instead of the backward-Euler *initial* residual of a
      **held-reference dual-time march**. This is the same gap the pseudo-time finding named — aquaflux
      approximates a transient with single PTC steps. The fix is structural, not a norm change: a true
      dual-time march (hold `φ⁰`; put the shift in the residual **and** Jacobian as
      `G = R + (ρV/Δτ)(φ − φ⁰)`, scaled by `a_P + ρV/Δτ`; inner-iterate `G → 0`; advance `φ⁰`; judge on
      the initial residual per outer step). Then the measure behaves exactly as in the reference — and it
      is the same change the pseudo-time finding calls for, so the two motivate one build.
    - **PROTOTYPE VALIDATED (2026-07-27; prototype not in the repository, superseded by the shipped
      `DualTimeStep`) — the diagnosis holds, and
      the fix is a per-timestep inner loop, not a norm change.**
      - **Confirmed current PTC = dual-time with K = 1.** At β = 2 the inner Newton converges
        `G = R + βd(φ − φⁿ)` in a **single** step (`‖G‖` 2.2e-2 → 6e-4, quadratic — it *is* the shifted
        Newton step), reproducing the single-step march exactly; the scaled measure then stalls
        (2.229e-2 → 2.204e-2) while euclidean halves — the β×travel signature.
      - **At β = 0.5 the inner loop engages (K = 2–3) and is stable**, converging `G` (2.2e-2 → ~3e-4)
        with a **line search on the scaled `‖G_n‖`** (first inner step clipped α = 0.5, then full). This is
        the legitimate inner merit (`G_n = 0` is a well-posed fixed-`φⁿ` solve), distinct from the refuted
        `G`-as-*outer*-merit.
      - **The measure is now honest.** The scaled `‖R(φⁿ)‖` holds ~2.1e-2 while `x_r/h ≈ 0` — it correctly
        reports that the slow bubble has not developed — while euclidean falls fast (2.86e2 → 4.6e1 over 3
        steps) on the quick pressure/momentum modes. That split is physical, not the β×travel artifact.
      - **CAVEAT — the dual-time STEP alone does not accelerate reachability; the Δτ RAMP is what does.
      This and the carried-`DualTimeControl` result below are ONE finding, not a conflict.** Development
      rate is Δτ-governed, so a dual-time step held at a fixed β is the same crawl. Its contribution is
      (a) an honest `‖R(φⁿ)‖` that can *drive* a Δτ ramp (single-step's stalling measure is why SER ran
      backwards) and (b) the inner-line-search-on-`G` tolerating a larger Δτ than one shifted step.
      Reachability needs that ramp **and** the cold-start diffusion/Re continuation (they compose:
      dual-time is the honest gauge + robust per-step solve, continuation lowers the cold stiffness so Δτ
      can grow early). Read the ramp's own result below as a measurement of the *ramp*, never as evidence
      that the step alone accelerates anything.
      - **CFL-ramp A/B (2026-07-27; prototype not in the repository) — the hypothesis holds, the
        gate is now the low-β linear-solve cost.** A `DualTimeStep` + `CflController` (grow Δτ / drop β
        when the inner loop meets η within ≤ 3 steps with α ≥ 0.5; back off otherwise), cold start:

        | inner solves | β | inner | x_r/h | scaled ‖R(φⁿ)‖ | euclid |
        |---|---|---|---|---|---|
        | 9 | 0.263 | 3 | 0.031 | 2.12e-2 | 44 |
        | 12 | 0.176 | 3 | 0.095 | 2.01e-2 | 29 |
        | 15 | 0.117 | 3 | 0.227 | **1.51e-2** | 18 |

        - **CONFIRMED: the inner loop unlocks β far below the single-step floor.** β ramped 2.0 → 0.117
          (still dropping) with every step converging (met, α = 1, ≤ 3 inner) — single-step blows up at
          β = 0.25 cold, dual-time is stable at less than half that.
        - **The measure fix is now visible in a march:** once the bubble formed (x_r/h 0.095 → 0.227) the
          scaled ‖R(φⁿ)‖ fell 25 % in one step, where the single-step scaled measure stalled at 1.9×
          forever. x_r/h accelerates as β drops (0.031 → 0.095 → 0.227, ~doubling per Δτ doubling).
        - **NOT YET more efficient per solve, and the reason is the low-β cost.** (i) The controller
          started at β = 2 (safe cold) and spent ~6 solves in the unproductive high-β regime before the
          bubble moved, so at 15 solves it trails aP05 (single-step β₀ = 0.5: x_r/h 0.58 @ 15). Fix: start
          the controller at β = 0.5 (proven stable cold). (ii) As β drops the shifted saddle loses diagonal
          dominance, so each solve costs more GMRES cycles *and* the inner loop needs 2–3 steps — stability
          is bought, not cheaply. **That low-β linear-solve cost is exactly what automated Re/ν_t
          continuation removes** (lower cold stiffness → cheap low-β solves), so dual-time (stability +
          honest gauge) and continuation (cheap big-Δτ steps) compose — the point to move to Re continuation.
      - **BUILT (opt-in): `DualTimeStep` (`solve/continuation.py`) + `DualTimeControl`
        (`solve/step_control.py`).** `DualTimeStep` is a `NewtonStrategy` whose `stepper()` holds a reference
        `φⁿ` and runs an inner Newton loop on `G = R + β d (φ − φⁿ)` to `‖G‖ ≤ inner_tol·‖R(φⁿ)‖` (or
        `inner_steps`), line-searched **monotonically on ‖G‖** (a well-posed fixed-`φⁿ` solve, unlike the
        non-monotone steady residual). The shift is in the residual *and* the Jacobian, so the measured
        steady residual is the honest discrete time derivative, not `β×travel`; `inner_steps = 1` is one
        shifted step (the pseudo-transient attempt, minus the escalation ladder the inner loop replaces).
        β still vanishes at the root, so the IFT adjoint is unchanged — pinned by
        `tests/unit/test_dual_time.py` (converges, exact gradient, **iteration-count-independent**).
        **Inner-loop observability — `DualTimeStep.inner_observer` (opt-in, shipped).** The outer
        `StepReport` only summarizes the inner loop (the inner *count* and the *summed* solve cycles),
        which conflates the two costs and hides the inner `‖G‖` trajectory. `inner_observer` is a
        `(inner_index, ‖G‖_before, ‖G‖_after, cycles, alpha) -> None` hook called **once per inner
        iteration** via `jax.debug.callback` — so it surfaces exactly how many inner iterations ran, each
        inner solve's cycle count, its `‖G‖` reduction and line-search factor. It is forward-only and
        transform-transparent (a no-op under `jax.grad`); `None` (default) elides the call at trace time,
        leaving the step **byte-identical** (do not set it on a differentiated solve). Threaded through
        `coupled_continuation` / `coupled_amg_continuation` / `coupled_lu_continuation` (the ILUT
        builder named here no longer exists), so a profiling march can pass one straight through. Pinned by
        `test_dual_time_inner_observer_surfaces_the_trajectory_without_changing_the_step`.
        `DualTimeControl` is the Courant β-ramp (grow the pseudo-timestep while the inner α = 1, shrink
        when it clips), a `StepControl` on the eager march. The step's
        reported α is the **min** inner line-search factor, and an inner step that fails to reduce ‖G‖
        (the line search's non-descent fallback, which otherwise reports α = 1) is folded to **α = 0** so
        the control reads it as struggling and backs off rather than growing — the α-only `StepReport`
        signal cannot otherwise distinguish a clean full step from a non-descending fallback. Wired
        through `coupled_continuation(inner_steps=…, inner_tol=…)` (returns a `DualTimeStep` when
        `inner_steps > 1`, else the unchanged `PseudoTransientStep`) and reachable as
        `solve_coupled(coupled, inner_steps=…)`. **The default path (`inner_steps = 1`) is byte-unchanged.**
      - **`DualTimeControl` IS NOW THE DEFAULT for a dual-time observed march, and it CARRIES β across
      refreshes — this reaches a developed recirculation several-fold faster than the residual-keyed
      control (measured 2026-07-30, and it SUPERSEDES the "runs the transient away" verdict just
      below).** The reachability crawl to develop the pitzDaily bubble was a **step-control defect**,
      not a pseudo-time limit. Two defects, both fixed/retired here:
        - `DualTimeControl` used to **reset β to `beta_start` on the first step of every post-refresh
          segment** (`previous is None`); with a ~3-step drift refresh β *sawtoothed* `0.5→0.33→0.22→
          (refresh)→0.5→…` and Δτ never grew — so the α-ramp was byte-identical to the pinned SER control.
          `next_step` now **carries β** on a segment boundary (`state` present, `previous is None` → hold),
          exactly as `ResidualRatioDualTimeControl` does. Its carried state is a **bare β** (SER's is
          `(β, prev ‖R‖)`). Pinned by `test_dual_time_control_holds_beta_across_a_refresh`.
        - `solve_coupled` **auto-defaults** `step_control=DualTimeControl()` when the march is a
          `DualTimeStep`, is already observing (a refresh or observer is set), and no control was supplied
          (`default_dual_time_control(step_control, observing, continuation)`, **in
          `solve/step_control.py` since 2026-08-15** — it lived in `turbulence/coupled.py` only because the
          import cycle below made it inexpressible in `solve/`; unit-tested in
          `test_coupled_rans.py`). It is injected **only where a control runs** and **never turns
          observation on**, so the differentiable single-stage solve (guarded `_is_traced`) is untouched;
          pass an explicit control to override. `solve_reynolds_continuation` inherits it (kwarg forward).
        Measured 2026-07-30 on a matched-seed pitzDaily rung-1 testbed and on a full cold Re ramp (hybrid
        IC → Re/100 → Re/10 → target Re 25000), carrying `DualTimeControl` against the SER control:
        **carrying β cut the outer-step count several-fold, and the ramp reached a developed `x_r/h` close
        to the OpenFOAM value.** *(the step counts and `x_r/h` recorded no continuation builder,
        preconditioner or forward solver — re-measure before quoting a number.)* The qualitative behaviour
        is what to rely on: it is self-regulating — α clips in the steepest development, recovers to 1.0,
        then β falls to the `beta_min` floor and the tail converges near-quadratically. `beta_min` is a
        speed↔smoothness knob (a smaller floor is faster but can overshoot the steady bubble on a cold
        rung with a loose seed + big Re jump, costing a couple of expensive recovery steps; the class
        default is the smoother choice).
      - **⚠️ THE "`DualTimeControl` RUNS THE TRANSIENT AWAY" VERDICT IS SUPERSEDED — do not cite it.**
        It held that the α-control grows Δτ blind to the steady residual and drives `x_r/h` past the
        steady state without settling, and was measured on the Re/100 anchor **before the β-carry fix and
        without the `beta_min` floor**. With β carried and the
        floor bounding Δτ, the ramp converges standalone (rung-1 to rtol 1e-6; full ramp to target Re) — the
        "runaway" the residual-keyed control was built to prevent does not block convergence here, and its
        residual-feedback instead *pins* β on the flat `β×travel` plateau (the slower arm). Do not cite the
        old verdict as a reason to prefer the residual-keyed control.
      - **`ResidualRatioDualTimeControl` (`solve/step_control.py`) is now the OPT-IN alternative — switched
        evolution relaxation / Kelley–Keyes pseudo-transient continuation.** It ramps Δτ by the steady-
        residual reduction ratio: `β ← β · (‖Rₙ‖/‖Rₙ₋₁‖)` (residual drop → β down / Δτ up; residual rise →
        β up / Δτ down), clipped to `[1/max_change, max_change]`, clamped to `[beta_min, beta_max]`, with a
        hard inner-clip (`α < backoff_below`) safety shrink, and carrying β across a refresh. A rising
        residual *automatically* shrinks Δτ, so it cannot run away — but on the pitzDaily ramp the row-scaled
        steady residual is nearly flat while the flow develops (`β×travel`), so it **pins β near `beta_start`
        and stalls Δτ**, taking several-fold more outer steps than the α-based default. Prefer it only where the steady
        residual is a reliable monotone progress signal. Its `next_step` state is `(β, prev ‖R‖)`. Unit-tested
        in `tests/unit/test_step_control.py`.
      - **DELETED — "the low-β wall is the block-SIMPLE preconditioner, and the ILUT breaks it" was
      superseded on both halves.** The remedy is gone: the monolithic ILUT was measured **dominated** and
      removed, and there is no `coupled_ilut_continuation` / `IlutFactors` / `ilut_beta_tracking_refresh`
      any more (record in git history of `solve-direct-preconditioners.md` before 2026-10-17). The premise is gone too: the shipped AMG
      preconditioners solve to **adjoint grade at zero shift**, which is the very regime this entry
      claimed only a complete factorization could reach. The one line worth keeping is the trap: a
      dual-time march driving β low is where a preconditioner's low-shift behaviour is tested, so measure
      there and not at the march's floor. ⚠️ And note what the entry never controlled for — the
      block-SIMPLE path had **no k-positivity limiter at all** until 2026-08-19 (record in git history of `.claude/rules/turbulence.md` before 2026-10-17), so an
      unguarded `k` crossing zero produces exactly the NaN it attributed to conditioning.
      - **Residual FLOOR + over-development past the minimum = loose `inner_tol`, NOT the preconditioner.**
      Even with the ILUT (a flat cycle count — the linear solve is fine), the march bottoms out at a
      residual floor and then slowly over-develops. Cause: dual-time's unconditional stability comes from
      the inner loop driving `G = R + βd(φ−φⁿ)` to zero each step; at `inner_tol = 0.05` the implicit
      step is only 5%-solved, so a large-Δτ backward-Euler step on a half-solved system overshoots. Fix =
      tighten `inner_tol` (with enough `inner_steps` to reach it) — **affordable precisely because the
      ILUT makes the low-β inner solves cheap**, where block-SIMPLE could not. ILUT removes the
      conditioning wall; tight `inner_tol` restores dual-time stability; the two together are what settle
      the rung. *(the floor and the `x_r/h` it corresponded to shared the unrecorded configuration of the
      bullet above — the mechanism stands, the numbers are deleted.)*
      ⚠️ **"Tighten" has since been bounded on the OTHER case, and it does not mean 1e-3.** On `bfs3d`
      a three-point sweep measured `inner_tol` 1e-2 as a **33 % shorter march than 1e-3 at an identical
      step count**, with 0.05 the first value to cost an outer step — see the dual-time inner-tolerance
      entry above. So this bullet's direction is right and its magnitude is case- and Δτ-specific: do not
      read it as an argument for 1e-3 anywhere else.
      - **⚠️ READING SMALL CYCLE COUNTS (binding — two offsets fooled a whole investigation).** Two things
        inflate the reported linear-solve cost at the low end, so a "6" is NOT six times a "1":
        (1) **lineax's `num_steps` has a +2 offset and is blind within a restart cycle.** Calibrated: a
        system GMRES solves in 1, few, or ~100 matvecs (all inside one 120-restart cycle) ALL report
        `num_steps = 3` (a dummy r0=0 first pass + deferred breakdown); it only climbs when the solve
        genuinely spills past a restart cycle. So **`num_steps = 3` means "converged in one cycle" = ideal**,
        and `solve_linear`'s count cannot distinguish 1 matvec from ~100. (2) **`DualTimeStep` reports the
        SUM of `num_steps` over its inner Newton iterations** (`stepper` docstring). So a dual-time
        `cyc = 6` is **~2 inner Newton iterations × an ideal 1-cycle solve**, and `cyc = 9` is ~3 —
        the inner-iteration count to reach `inner_tol`, NOT a per-solve penalty. **Consequence measured
        this session:** the coupled ILUT is a NEAR-DIRECT preconditioner — 1 restart cycle (~4 matvecs) per
        solve at every pitzDaily state, flow-only and full `[u,v,p,k,ω]` alike, fresh or mildly stale
        (record not in the repository). The march's "6–9" is the dual-time inner-loop sum, and
        **β-matching the frozen factorization to the march's β is a no-op on it** (fixed-`ilut_beta` and
        `ilut_beta`-matched runs gave IDENTICAL `cyc`). The only lever on the "6" is `inner_steps`/`inner_tol`
        (globalization/accuracy), which is deliberately kept tight for stability — not the preconditioner.
        The "coupled ≈ 6 vs flow-only ≈ 2 → k/ω degrades the ILUT → build ILUT+AMG" premise is REFUTED; the
        only live reason for ILUT+AMG is 3D `spilu` fill scalability, which cannot be judged on 2D pitzDaily.
    - **Lowering β is not the escape, and the reason is specific — state it precisely.** At `β = 0.25`
      the k/ω blocks reach 1e24 / 1e52 at the cold IC and go NaN at step 20, but are **perfectly stable
      at steps 45 and 90** (ratios 0.994–1.046). So the under-damping is an *early-state* property, not
      a general one: β can be lowered once the flow is developed, and cannot be lowered at exactly the
      cold start where the reachability problem lives. This independently re-kills `descent_backoff`,
      whose whole premise is lowering β from a cold state.
    - **Still open:** the *across-iteration* weight drift (`a_P` and the field magnitudes both grow as
      the flow develops, so the denominators move between iterations) is a **separate** effect from the
      β floor and remains unmeasured. Settle it by replaying one `RowScaledNorm` with scales frozen at
      the warm-started root over the stored `profile_base/g*.npz` history — seconds of compute.
  - **⚠️ THE MEASURE'S WEIGHTS ARE STATE-DEPENDENT, so there is no single objective across iterations.**
  `f(x) = Σ wᵢ(x)|Rᵢ(x)|` with `w` from the operator diagonals and field magnitudes. **This governs the
  OUTER-ITERATION boundary only:** when a `norm_builder` is supplied, `newton_march` rebuilds the
  measure at the state each outer iteration begins from and freezes it for that whole iteration (so the
  line search compares like with like) — so a direction that descends in *this* iteration's frozen `f`
  need not reduce the *next* iteration's `f`. Do not assume the frozen-per-iteration measure behaves
  like a fixed merit function. This is **not** in conflict with "the measure must be held FIXED across a
  refresh" below: that rule governs the *segment/refresh* boundary — the `base_norm` `solve_coupled`
  builds once and re-injects into every refreshed continuation, which is what the convergence test and
  the finishing solve are judged in.
  - **⚠️ `descent_backoff` IS COUNTERPRODUCTIVE ON THIS CASE — measured, do not enable it blindly.**
    Backing β off until the correction descends does produce a descending direction, but the finite-step
    profile along it is *worse*: at β = 0.5 the full step raises the measure 2.59× and is not admissible,
    forcing α ≤ 0.5. On a march the arm's α fell 1.0 → 0.5 → 0.5 → 0.031 while the measure *rose* every
    step. **Descent is necessary but not sufficient** — strong positive curvature along δ swamps the
    negative slope. Note ‖δ‖ *decreases* as β is backed off (1049 → 856 → 760 for β = 2 → 1 → 0.5), so
    "α collapsing" is not a large-correction artefact.
  - **⚠️ EXTENDING THE LADDER ABOVE α = 1 (`grow`): inert on the Euclidean measure, live on the
    equilibrated one — and it exposed a fallback bug (2026-07-27).** ⚠️ **`grow` no longer exists
    (deleted 2026-10-14):** from 2026-08-15 (`7c1361bd`) the pseudo-transient step passed the line search
    `max_alpha = 1.0` whenever no whole-step limit was set, which capped every growth rung at the full
    step, so nothing measured after that date ran with it; deleted as inert with the whole-step-limit
    seam. The measurements below predate the cap. (The equilibrated/row-scaled measure
    is now the *default*, so `grow` is live on the shipped configuration; the Euclidean result below is the
    now-non-default measure.)
    - On the **Euclidean** march, `grow = 2` produced a trajectory **bit-identical** to the
      control across 10 steps and both checkpoints: α = 2 is never admissible there, so the extended
      ladder is inert on that measure.
    - On the **equilibrated** measure it fires: α = 2 was selected at step 1 and was productive. A
      cold-start scan confirms α = 2 sits inside the tolerance (ratio 1.291 against a 2× bound) and
      travels twice as far as the full step.
    - **The bug it exposed:** extending the ladder upward also extended the *fallback* upward, so a step
      with no admissible length fell back onto **α = 4** and multiplied the measure by **4.6** in one
      step. The fallback is now capped at the full step — **a growth rung must only ever be reachable by
      passing the acceptance test, never by falling back onto it.** Pinned by a unit test.
  - **⚠️ THE SHIFTED CORRECTION IS NOT A DESCENT DIRECTION, AND THE CAUSE IS THE UNSHIFTED CONSTRAINT
    ROW (measured 2026-07-27). This is the mechanism behind the α-at-the-smallest-rung stalls recorded
    throughout this file.** For the *exact* Newton direction (`J δ = −R`) the derivative of any
    positively-weighted residual measure along `δ` is `−‖R‖ < 0` — descent, for free. The **shifted**
    direction satisfies `J δ = −R − β D δ`, whose second term has no fixed sign, and its damage grows
    with β. Measured directly (`∇f·δ` by forward-mode AD through the measure) on a stiff coupled state:

    | β | 0.05 | 0.2 | 0.5 | 1.0 | **2.0** |
    |---|---|---|---|---|---|
    | `∇f·δ` | −7.7e-3 | −1.7e-3 | −4.9e-4 | −1.3e-4 | **+3.9e-5** |
    | ladder minimum α | 1.0 | 1.0 | 0.25 | 0.25 | **0.00098** |

    **The sign changes between β = 1 and β = 2, and the march was running at β ≈ 1.9.** At β = 2 the
    best rung on the whole ladder is the shortest one, which reproduces the observed stall exactly. Note
    the lower bound too: at β ≤ 0.2 the trial states go non-finite, so the usable window at that state
    was roughly **0.5 ≤ β ≤ 1**.
    - **What causes it is the MIXTURE of shifted and unshifted rows — not the weighting, and not the
      off-diagonal coupling.** Both of those were proposed and refuted on toy systems: a scalar residual
      gives `∇f·δ = −|R|·J/(J + βD) < 0` for *any* β, and a symmetric system with strong off-diagonal
      coupling, or with strongly skewed row weights, still descends at every β tested. What reproduces it
      is a **saddle system whose constraint row carries no shift** — the exact shape of the flow policy,
      where momentum rows get the operator diagonal and continuity gets zero:

      | β | 0 | 0.5 | **2** | 10 |
      |---|---|---|---|---|
      | `∇f·δ` | −2.30 | −1.15 | **+2.30** | **+20.70** |

      ⚠️ **CORRECTION (2026-07-27, same day): "shift every row uniformly and the derivative stays
      negative at any β" — as first written here — is WRONG.** That was only ever tested on a
      *symmetric* system, never on the saddle. Damping the constraint row on the saddle above gives:

      | `d_p` | 0 | 0.1 | 1.0 | 5.0 |
      |---|---|---|---|---|
      | `∇f·δ` at β = 2 | +2.300 | +1.438 | +0.329 | +0.074 |
      | `∇f·δ` at β = 50 | **+112.7** | +0.440 | +0.044 | — |
      | **crossover β** | **0.987** | **0.987** | **0.987** | **0.987** |

      So constraint damping **does not move the descent threshold at all** — it is 0.987 for every
      `d_p` tested, including zero. What it changes is the *magnitude* past the threshold: with an
      unshifted constraint row the failure **grows without bound in β** (+2.3 → +113), with a shifted
      one it **decays toward zero** (+0.33 → +0.044). And on this toy no rung of the ladder reduces the
      measure at β = 2 for **any** `d_p` — the profile is monotone in α throughout.
      **Consequence: damping the constraint row is not a fix for non-descent, and should not be sold as
      one.** What remains true and useful is that the unshifted row makes the failure unbounded rather
      than bounded, and that the threshold itself (β ≈ 1 here, and between 1 and 2 on the real coupled
      case) is set by the momentum shift against the Jacobian scale, not by the constraint row. Whether
      bounding the damage buys anything on the real nonlinear system is unmeasured — the toy is a 2×2
      linear system and cannot answer it.
    - **Escalation moves β the WRONG WAY for this failure (binding).** A rejected step escalates
      `β *= escalation_factor`, which is right for an overshoot or an ill-conditioned shifted system.
      Against a non-descent direction it is worse than useless: more shift makes `∇f·δ` *less* negative,
      so the loop spends a solve per attempt making the direction worse. `PseudoTransientStep` therefore
      carries **`descent_backoff`** (lower β until the direction descends, then escalate from there) and
      **`descent_test`** (reject a non-descent direction outright rather than judging the candidate's
      norm). Both default off. `∇f·δ` itself is cheap: one `jvp` on a direction already computed.
    - **A backoff probe is a COMPLETE attempt and is carried into the escalation loop — do not go back
      to discarding it.** The probe already computes the correction, the line search, the measure and
      `∇f·δ` at exactly the β the escalation loop then starts from, so re-solving there made every step
      pay **two** shifted solves on the path where nothing is backed off — the common one. The five
      values travel as one `_Attempt` record, and the loop folds its final probe into the escalation
      carry (`record(fresh(β), trial, probed & admits(trial, 0))`, selected by the loop's own descent
      flag). The seeding is used **only** when the carried attempt really was taken at the starting β:
      if the backoff instead exhausts its tries it exits at a *lower, unprobed* β and the escalation
      loop starts cold there, which is the pre-existing ladder. A backoff that has to lower β still
      costs one solve per rung; what is now free is the case where the first probe already descends.
  - **⚠️ THE LINE SEARCH TAKES THE LONGEST ADMISSIBLE STEP, NOT THE BEST ONE — a minimizing search was
    built, measured, and REVERTED (2026-07-27). Do not re-propose it.** Replacing "first rung that is
    admissible, walking longest-first" with "the rung that minimizes the measure" lowers the residual per
    step and is far worse on the physics: on the same cold-start case, judged at identical checkpoints,

    | checkpoint | minimizing | first-acceptable-largest |
    |---|---|---|
    | 2 | 0.01 | **0.09** |
    | 3 | 0.03 | **0.16** |
    | 4 | 0.05 | **0.34** |
    | 5 | 0.05 | **0.46** |

    **9× less recirculation development, while reporting BETTER residuals at every early step** (0.377
    vs 0.430, 0.254 vs 0.293, 0.193 vs 0.212). The α sequences show the mechanism: the minimizing search
    systematically picks 4–8× shorter steps. **Residual depth per step and distance travelled per step
    are different objectives, and on a march that has to transport a front across the domain, distance
    is the one that matters.** This is the fourth time on this case that a residual improvement has
    pointed the opposite way from the physics — judge a march on `x_r/h`, never on ‖R‖.
    - **The fallback when NOTHING is admissible is the longest ADMISSIBLE rung, not the shortest
      (binding).** Returning the shortest is a near-null step that changes nothing, which the divergence
      guard then accepts as finite: the march reports a step and stands still. That is a *guaranteed*
      stall rather than a slow one, and it is what produced the `α = 0.001` signature (`0.001 = 1/2**10`,
      the smallest rung of the shipped 10-rung ladder — a value that means "nothing passed", not a
      sentinel).
      ⚠️ **"Admissible" used to mean only FINITE, and finite is a very weak bound — see
      `fallback_growth` at the end of this file.** A rung must now also be under
      `fallback_growth * reference_norm`; the *longest qualifying* one still wins, so this reasoning is
      unchanged, but a rung 130 decades above the reference no longer qualifies at all.
    - **The ladder could extend ABOVE α = 1 (`grow` rungs of doubling; default 0 = off; deleted
      2026-10-14, inert since 2026-08-15 -- see the `grow` entry above).** Measured on a
      developed state: the full step moved the reattachment not at all, while `α ≈ 5.7` moved it four
      times further **and already sat inside the tolerance the acceptance rule allowed** — it was simply
      unreachable from a ladder that starts at 1. Any scan or study of step length must therefore not
      hard-bound its grid at 1.0, which an earlier one did, making "α = 1 is optimal" unfalsifiable.

  - **⚠️ THE "TIGHT TOLERANCE IS LOAD-BEARING UNDER LOG-ω" CLAIM WAS STALE — corrected 2026-07-28.** The
    old note here (and in `turbulence.md`) said an inexact/loose forward solve is unsafe under log-ω
    ("an inaccurate log step is exponentiated and diverges; loosening breaks the march"). Two-arm cold-IC
    pitzDaily marches — the over-solving default vs a 4-cycle-capped ~1e-3 solve, then the adaptive
    `relative_residual_gmres` — **refute it**: the honest ~1e-3 solve reproduces the over-solve march to
    3-4 significant figures per step (**including the line-search α**), tracks the same `x_r/h`, and never
    diverges, at ~3-4× fewer matvecs. The original evidence was almost certainly an artifact — "loosening
    rtol" never loosened the solve, because `atol=1e-10` bound regardless of rtol (the componentwise floor
    above), so it kept over-solving to ~1e-12. Do **not** reinstate a tight fixed tolerance on the forward
    solve. Same-root-safe by construction: inexact Newton, the shift vanishes at the root, the nonlinear
    stop is on ‖R‖, and the adjoint is a separate transpose solve — none touched by the forward-solve
    looseness (pinned by the `slow` `test_coupled_rans` convergence + adjoint gates).
  - **⚠️ READ FIRST: every raw-‖R‖ comparison ACROSS coupled-RANS march states recorded below predates
    the 2026-07-25 fixation-row fix and is suspect.** Until then the near-wall ω fixation was written
    in physical ω under a log-ω unknown, so 472 of 12 225 rows — scaled by an ω spanning 160→1.1e5 —
    dominated the norm that drives the line search, the SER ramp, the divergence guard and the stopping
    test (see `.claude/rules/turbulence.md`). Measured consequences: ‖R‖ at the *converged* pimpleFoam
    field was **1.533e5**, i.e. the metric rated the right answer ~1.7e4× worse than a half-developed
    state and 800× worse than the cold IC; and it ranked the const-β state at "rel 0.032" above the SER
    state at "rel 0.052" although the former's recirculation bubble is **4× worse** (`x_r/h` 0.29 vs
    1.16, target 7.74). After the fix the same field reads **20.7** and the ranking matches the physics.
    **So: "reached a deeper rel" was not evidence of a better state, and the preference for the const-β
    march and the α-targeting controller rests on exactly that comparison.** Re-measure before trusting
    any relative-residual claim in this section; the *mechanistic* findings (exact linear solves, the
    α-sentinel, the modal attenuation) are unaffected because they were measured at a single state.
  - **⚠️ SCOPE FIRST: the "SER runs backwards" finding below applies ONLY to a march that does NOT
  refresh (2026-07-25).** SER's `residual_norm_0` is **segment-local** — recomputed at each
  `newton_march` entry, hence reset at every preconditioner refresh. With a refresh every handful of
  steps the ratio `‖R‖/‖R₀‖` never falls far below one, so **β is pinned near β₀ for the entire march**
  rather than decaying (measured on a drift-refreshed cold-IC pitzDaily march; the per-step β/α table
  that stood here is deleted — it named no preconditioner or forward solver and was taken under the
  superseded ω-dominated norm).

  Three consequences. (i) Enabling the refresh silently converts SER into **constant β ≈ β₀** — if a
  different damping is wanted it must come from `β₀` or a different schedule, not from expecting SER to
  ramp. **`β₀ = 2` is the settled value.** The earlier reading here — "β₀ = 2 is too much, `β₀ = 0.5`
  develops a longer bubble" — was measured *before* the flux-continuous / wall-model `a_P` fixes that
  changed the shift itself, and the post-fix re-profile reverses it (`β₀ = 0.5` is under-damped and
  stalls); see "RE-PROFILED AFTER THE `a_P` FIX" above, which is the surviving side of that conflict.
  (ii) **An α-targeting controller may have nothing to push against here**: α was reported saturated at
  its set-point on this march while the residual barely moved, i.e. the productivity ceiling is not
  obviously an α problem, which re-scopes #22 — same unrecorded configuration, so treat it as a
  hypothesis, and note the shipped dual-time default *is* α-driven (see the conflict recorded above).
  (iii) Any probe that derives "the march's operating β" from the global ratio is wrong by a large
  factor at a developed state.
  - **THE SER β SCHEDULE RUNS BACKWARDS FOR STIFF COUPLED RANS (pitzDaily — the claim is that the
  dominant cost is the globalization, not the preconditioner).** The switched-evolution-relaxation
  schedule `β = β₀(‖R‖/‖R₀‖)^p` *lowers* β as the residual falls, on the premise that a smaller shift
  means a more Newton-like, more productive step near the root. **On this problem the premise was
  measured false: the efficiency-optimal β *rises* as ‖R‖ falls, so SER drives β the wrong way and the
  coupled march grinds instead of entering the quadratic basin.** ⚠️ **The supporting numbers — the
  efficiency optima, the step-efficiency gap and the α-vs-β table — are DELETED. All of them predate the
  2026-07-25 fixation-row fix and were taken under the ω-dominated norm that mis-ranked states, with the
  preconditioner rebuilt at each probed state (which the march never is). Re-measure before relying on
  any of this quantitatively.** What survives is the mechanism and one design consequence:
  - **The mechanism is line-search CLIPPING, seen directly via the step-length factor α.** α (the
  fraction of the shifted step the backtracking search keeps) rises with β and reaches **α = 1 at the
  efficiency-optimal β** — the point where the full damped step *just stops overshooting*. Below it
  the step overshoots and is clipped to near-nothing; at it the step is full and productive. So the
  grind is over-damped clipping, not near-convergence and not preconditioner cost.
  - **α is the usable controller signal; the per-step residual reduction ρ is not** — ρ swung
  several-fold at fixed β and wrecked a first, ρ-driven controller that ratcheted β into a runaway.
    - **Caveat — β-schedule and PC-refresh are COUPLED; the deleted optimal-β measurements above all used
      a PC rebuilt at each probed state.** In a real march the preconditioner is frozen at the cold IC, and
      a bolder β moves the state faster, staling that frozen PC faster (the ρ-driven controller's runaway
      got *more* expensive as β rose, where a bolder shift should be *cheaper* — so that was PC staleness,
      not the shift). So an α-targeting β schedule and the scalar-AMG refresh (below) must be co-designed,
      not tuned in isolation. A **β-independent staleness
      indicator** — the drift of the frozen operator's coefficients, `‖Δν_t‖`/`‖Δṁ‖` relative to the
      freeze state — is the clean refresh trigger this motivates (it fixes the `CycleGrowthTrigger`
      confound, #19: cycle count rises from β→0 *and* staleness, drift rises only from staleness).
    - **A/B'd end-to-end against SER (α-targeting controller + PC refresh) — the numbers are DELETED.**
    The whole comparison was a race between two ‖R‖ trajectories measured under the superseded
    ω-dominated norm, which the scoping entry above shows mis-ranked states; "reached a deeper rel" is
    exactly the claim the fixation-row fix invalidated. A prototype controller — raise β toward the α=1
    boundary (`β ← β/α`, capped), ease gently when α=1 — with the k/ω AMGs refreshed periodically and the
    step `filter_jit`'d (to match SER's compiled `while_loop` footing) was reported faster than SER at
    every overlapping residual from the cold hybrid IC. *(re-measure on `x_r/h` before citing it.)* Two
    structural findings from the same arms are worth keeping, because they say *which* configuration wins
    rather than by how much: (a) the **frozen-PC** α-controller *lost* — cycles rose with β, the β↔PC-
    refresh coupling biting, so the refresh is load-bearing; (b) the **eager** (un-jitted) version was
    handicapped per cycle, so the jit is needed for a fair comparison, not for the physics.
    - **The controller has a CEILING — it stalls short of a root, deeper than SER but not converged.** The
    cause is its own **over-damped hunting**: the `β/α` raise overshoots *past* the α=1 boundary to where
    the full step is tiny, then eases slowly; α saturates at 1 above the boundary, so the controller is
    blind there and cannot sit at the productive edge. So the direction is right, but a dynamics rework
    is needed: approach α=1 *from below* without overshooting, or pair α with a step-productivity signal.
    *(the residual levels quoted for both arms shared the superseded ω-dominated norm — the stall is the
    finding, its depth is not.)*
    - **PRODUCTIONIZED as an injected strategy pair (the schedule half is shipped).** The β schedule is
      the injected `RelaxationSchedule` (SER = `SwitchedEvolutionRelaxation`, the default; see the
      `continuation.py` bullet). **The α-targeting control itself is DELETED (2026-08-14) — there is no
      `AlphaTargetingControl`.** It never converged standalone, its gains were hand-set placeholders, it
      had no production caller, and it was the one control that both lacked `carry_beta` and reset β at a
      refresh boundary — so a shared `ShiftStrengthControl` base would have had to carry a seam for a
      member nothing selected. The α signal survives where it is measured to work: inside
      `DualTimeControl` and `CflResidualDualTimeControl`, which drive the *dual-time* pseudo-timestep.
      The single-step α-targeting *direction* is unrefuted and unbuilt; rebuild it as a
      `ShiftStrengthControl` subclass if it is ever wanted. Study harnesses in the scratchpad
      (`beta_sweep.py`, `alpha_probe.py`, `alpha_controller_march.py` = frozen-PC, `alpha_refresh_march.py`
      = the winning arm) remain as the calibration/replay tools.
    - **A PER-BLOCK β (separate shift damping for flow / k / ω) is DOMINATED — measured, do not re-attempt
      (`per_block_sweep.py`).** The Euclidean ‖R‖ on the coupled state is ~100 % ω (ω O(1e1) vs flow O(1e-2),
      k O(1e-3)), so a natural idea is to damp each block by its own β — the block-diagonal shift already
      supports it (unpack the shift diagonal `[a_P·u, 0·p, d_k·k, d_ω·ω]`, scale each slice, repack; the flow
      preconditioner keys off `β_flow` via its `a_P(1+β)`, the scalar AMGs are β-independent). Swept at the
      developed state (rel 0.05), holding `β_ω` high and lowering `β_k`/`β_flow`, it loses on every axis
      against uniform β. *(the per-block sweep table is deleted: it recorded no preconditioner or forward
      solver, and its judging quantity was the ω-dominated Euclidean norm that mis-ranked states —
      re-measure before treating the ruling as settled.)*

      Two failure modes were read off it, **neither a damping problem**: (i) **k is acceptance-limited** —
      a smaller `β_k` *does* let k descend, but the bigger k-step makes the *coupled* full step overshoot
      the ω-dominated norm, so the line search clips α and ω progress collapses; crediting k would need a
      block-aware *acceptance* norm, which is the dead `BlockScaledNorm` (below). (ii) **flow is
      coupling-limited** — no `β_flow` un-sticks it, because flow is waiting on ω through the two-way ν_t
      coupling. The blocks are coupled through **both** the direction (flow↔ω) and the acceptance (ω-norm),
      so per-block *damping* cannot separate them. This re-confirms the old "Lever D" per-block
      under-relaxation ruling, now with the mechanism visible under log-ω + the adaptive wall.
    - **The lever is a HIGHER uniform β, not a per-block one — but the numbers behind "β=5 ≫ β=3" are
    DELETED (same sweep, same ω-dominated norm, preconditioner rebuilt per state).** The reading was that
    at the developed state the efficiency-optimal β sits above the value SER reaches; that a higher β is
    not cheaper *per cycle* but wins on **step count and overhead** (fewer Newton steps → fewer PC
    refreshes, recompiles, line searches) while staying productive (α = 1); and that a constant-β march
    descended past SER's floor and then ground in the tail — the too-low-β symptom. So the direction taken
    was the β-climbing controller (#22: climb β while α = 1), **not** a per-block β, a norm change, or
    physical/order continuation. ⚠️ Re-measure before quoting any β from this: the post-`a_P`-fix
    re-profile moved the whole β calibration, and it is judged on `x_r/h`, not on ‖R‖.
  - **Where the coupled-solve cost actually is.** As the SER ramp drives `β → 0` through the march, the
    *unshifted* coupled saddle Jacobian is severely ill-conditioned, so the diagonally-shifted GMRES burns
    many matvecs per solve and the cost rises sharply as β falls. *(the per-solve wall times once quoted
    here named no preconditioner, forward solver, restart or state, and predate the fixation-row fix —
    deleted; the β-dependence is the mechanism, the seconds are not evidence.)* Note lineax `num_steps`
    counts restart **cycles**, not iterations, and carries a fixed offset — see the reading rule above.
    **The `β → 0` here is SER-induced and correctable, not inevitable — see the schedule-runs-backwards
    finding above.** Several levers were probed: two are wired but **off by
    default** (kept for further evaluation, not the fix), one is dead, and one — refreshing the **scalar**
    k/ω AMGs after the flow separates — is a real ~2.6× win, now BUILT (see below):
    - **Flooring the SER `β` below (`β = max(beta_floor, β₀(‖R‖/‖R₀‖)^p)`) — correctness-safe, reported a
    WASH, kept off-by-default.** The field is **`SwitchedEvolutionRelaxation.beta_floor`** (it lives on
    the schedule, not on `PseudoTransientStep`, whose `beta0`/`exponent`/`beta_floor` fields were
    removed); default 0 = off. It never moves the converged root (the shift `β d` scales the correction
    `δ`, which vanishes at `R=0`; it only damps the *path*, linear instead of quadratic terminal steps)
    and it does make each late solve cheaper, but end-to-end the cheaper late solves were reported to
    cancel the extra Newton steps. *(configuration not recorded — case, state, preconditioner and norm
    all unnamed — so treat "wash" as the reason it is off by default, not as a measured fact.)* Wired
    through the shared `Globalization` for further evaluation (it was `coupled_continuation(beta_floor=…)` when this was written; there is no such keyword now — see `solve-globalization.md`).
    - **The default coupled residual measure is the row-equilibrated `RowScaledNorm`
      (`coupled_scaled_norm`), NOT the Euclidean ‖R‖.** The Euclidean coupled residual is `ω`-dominated
      and *mis-ranks* states (a converged field scores worse than a badly wrong one — the warning above);
      `RowScaledNorm` divides each row by its own diagonal and each block by its field magnitude, so every
      equation is judged comparably. **`per_block` is that measure's reporting view and `__call__` is
      literally `norm(per_block(r))`** — one formula, so the per-equation grid in the march log cannot
      describe a convergence history the march never had. `coupled_continuation` builds it by
      default (the ILUT builder named here no longer exists); `block_scaled_norm=True` selects the coarser one-scale-per-block `BlockScaledNorm`
      (`_coupled_residual_norm`), and `residual_norm=jnp.linalg.norm` recovers Euclidean.
      (`mass_flow_coupled_continuation` still defaults to Euclidean pending a constraint-aware variant.)
      The row-scaled measure does **not** fix the forward stall (globalization-bound; it plateaus under any
      measure — that plateau is the *honest* signal, where the Euclidean fall was a `β×travel`/`ω`-magnitude
      artifact); it makes the measure honest and is required to judge this case correctly.
      **The measure must be held FIXED across a refresh (binding, #156 seam 4) — this governs the
      SEGMENT/refresh boundary, and does NOT conflict with the per-outer-iteration rebuild described
      above, which governs what a single iteration's line search and acceptance test compare in.**
      `BlockScaledNorm` is
      self-normalising — at the state its per-block scales were built at it returns `sqrt(n_blocks)` — so
      rebuilding it at each refresh's developed state re-bases every `residual_ratio` back toward one,
      making the convergence test unreachable and mismatching the finishing solve's absolute
      `stage_atol` (computed on the pre-refresh scale). `solve_coupled` therefore captures the initial
      measure once (`base_norm`, the same state the global `reference_norm` is measured at) and passes it
      to every refreshed `coupled_continuation(residual_norm=base_norm)`, which uses it verbatim instead
      of rebuilding. The invariant is "the global progress reference and the norm come from the same
      state." Latent before the fix (only bites with `block_scaled_norm=True` *and* a refresh); pinned by
      a unit test that a refreshed continuation reuses the initial norm object.
    - **A block-*triangular* preconditioner (forward-substituting `∂R_turb/∂flow·δ_flow`) — tried, WORSE,
      dead.** It made the channel worse (measured, configuration not recorded — no mesh, smoother or
      aggregation) and on recirculating pitzDaily was
      so bad GMRES could not converge at all: stronger flow↔turbulence coupling *amplifies* the inexact
      diagonal blocks' inversion error it propagates downstream. So the missing cross-coupling is **not**
      the bottleneck.
    - **The real cost is the pressure-Schur *approximation* at high Reynolds number — and strengthening
      the inner solve CANNOT fix it (measured; do not re-attempt).** The block-diagonal conv+MSIMPLE
      preconditioner is *excellent* at low Re (4 outer cycles on a Re=2500 channel) and weak only at high
      Re / recirculation (17 cycles on a Re=1e5 channel). The weak block is the **flow saddle**, not the
      k/ω scalars (per-block error operator `E_b = I − A_b·M_b` on a developed Re=1e5 channel: flow
      ρ=34.0 / one-shot 24.1, vs ω 13.9 / **2.4** and k 8.5 / 7.9 — ω's high ρ with a low one-shot is an
      isolated outlier eigenvalue GMRES kills in one iteration, a red herring). But every lever *inside*
      that block is dead:
      - **More velocity-AMG V-cycles (×2/×4/×8): ρ 34.019 → 33.995 → 34.031 → 34.046 — no effect at all.**
      - **More Schur V-cycles (×2/×4/×8): ρ 41.6 / 48.7 / 48.5 — strictly worse.** Inverting `Ŝ` *more
        accurately* making the preconditioner *worse* is the signature that `Ŝ` is the **wrong operator**:
        the error is the Schur *approximation*, not its inversion (a partial V-cycle was accidentally
        regularizing it). Driving both sub-solves toward exact never beats the 1-cycle baseline.
      - **Rebuilding the preconditioner at the developed state (staleness) does not help *the flow
        block*** (ρ 34.0 → 31.6 on the channel; 49.9 → 91.9, i.e. worse, on pitzDaily, with an identical
        one-shot). The frozen *flow* reference is fine — the convective linearization is Peclet-robust and
        MSIMPLE's Schur is velocity-independent. **Confirmed on the real solve:** refreshing only the flow
        block at a separated pitzDaily state made it slightly *worse* (31 → 34 outer cycles at β=2).
      - **BUT refreshing the *scalar* k/ω AMGs is a real 2.6× cycle win once the flow separates — the one
        staleness lever that does pay (measured on the real solve, not ρ).** The scalars were noted above
        as going stale (ω ρ 13.9 → 3.3 rebuilt) but dismissed as "not the cycle bottleneck" on the ρ /
        one-shot proxy; on the **real coupled shifted solve** that dismissal does not hold. Marching
        pitzDaily to a genuinely separated state (25 pseudo-transient steps, rel 3.0e-2, 70 recirculation
        cells, `x_r/h` 0.87) and re-solving the **same** shifted system with the preconditioner refreshed
        block-by-block (operator held fixed; every solve converged, `‖Aδ−b‖/‖b‖` ~1e-8):

        | refreshed | cycles | matvecs | wall |
        |---|---|---|---|
        | nothing (all frozen at the cold IC) | 31 | 3720 | 68.9 s |
        | **k/ω scalar AMGs only** | **12** | **1440** | **27.4 s** |
        | flow block only | 34 | 4080 | 71.8 s |
        | everything | 13 | 1560 | 30.4 s |

        So the entire gain is the **scalars** (31 → 12), the flow refresh contributes nothing (everything
        ≈ scalars-only), and this is a textbook instance of the ρ caution above — the scalars' low one-shot
        made them look harmless while they were worth 2.6× on the real iteration. **The benefit only
        appears once the flow has separated**: at a *pre-separation* state (4 march steps, no recirculation)
        a full refresh is worthless (17 → 14 cycles at β=2, and *worse* at β=0.2, 43 → 83), which is why an
        early measurement gives the wrong answer. Full-refresh gains were confirmed at β ∈ {2, 0.5, 0.2}
        (31→13, 19→12, 31→18); the block-by-block isolation above was run at β=2. **Implication for
        implementation: refresh only the two `ScalarTransportPreconditioner`s and leave the flow block
        frozen** — much cheaper than a whole-policy rebuild, and it avoids the flow refresh's small
        regression. It is adjoint-safe (the preconditioner is `stop_gradient`-ed whatever it is frozen at,
        so a refresh changes only the forward Krylov count, never the converged state or its IFT adjoint).
        **BUILT** — `newton_march` + `CycleGrowthTrigger` (see the `march.py` section) segment the march
        around the off-jit rebuild, which is required because the traced solve is one `lax.while_loop` and
        scipy AMG assembly cannot run inside it; `solve_coupled(refresh=RefreshPolicy(trigger=…))` is the driver.
        **⚠️ SETTLED FROM THE CODE — the old claim here, "a refresh still forces a full recompile because these
        are non-pytrees hashed by identity", is SUPERSEDED and deleted.** The fix it proposed as hypothetical was
        built: the coarsening structure is value-independent **at this path's `strength_threshold=0`**, and
        `_SparseLevel` now holds only `n` / `n_coarse`
        static with `val` / `diagonal` / `lam_max` / `coarse_inv` as **traced leaves** — so a refreshed hierarchy
        passed as a jit argument is a **compilation-cache hit**, pinned by
        `test_refreshing_a_hierarchy_is_a_compilation_cache_hit`. ⚠️ **The value-independence is a property of
        the threshold, not of the level split**, so it does not carry to the traced flow block, which runs at
        0.25 and re-partitions on every refresh; that path keeps the cache hit with
        `SmoothedHierarchy.refit` instead (see the flow-block section). What a refresh still costs is the off-jit scipy
        rebuild plus the one-off retrace of the rebuilt `NewtonStrategy`, which is why `refresh.limit` still bounds
        it. The wall figures once attached to this question (a "~60–240 s" recompile and a "~38 s" refresh) were
        both recorded with no configuration and are deleted with it.
      - **The observed march RETURNS ITS OWN CONVERGED STATE — the traced finishing solve is only the
        not-converged fallback (BUILT).** `solve_coupled`'s observed path (`on_step`/`refresh`/`step_control`)
        is never differentiated — those cannot run under a JAX transform (guarded), so the converged eager
        state needs no adjoint. When the eager `newton_march` reaches its stopping tolerance **judged in the
        measure it steered by** (the per-step-rebuilt `RowScaledNorm` under `scaled_norm`), `solve_coupled`
        returns that state directly instead of re-marching it through `RootSolver`. **Why this is
        required, not just an optimization:** the finishing solve targets the *frozen* base measure (state0
        row scales), which over-reports a developed state's residual (#156 seam 4), so it does not see the
        eager convergence — and being traced it cannot refresh or carry the SER step control, so on an
        aggressive low-shift ILUT dual-time path it leaves the converged state chasing the unreachable frozen
        target and **diverges to NaN** (measured on the pitzDaily Re/100 anchor: eager converges row-scaled
        0.009, finishing solve then returns ‖R‖=NaN). Returning the eager state fixes that. The finishing
        solve still runs when the eager march stops short, and is the plain differentiable path's sole march.
        **Open (for the target-Re adjoint):** a differentiated target solve still needs the finishing solve
        to converge deep *in the same row-scaled measure* — restructuring it (Python outer loop so it can
        carry the measure + step control) is the tracked follow-up; the lower-Re continuation rungs are
        `stop_gradient`ed seeds and need no adjoint, so the eager path serves them.
      - **Rescaling the MSIMPLE `k` is a ρ mirage — validate on the real march, never on ρ.** Growing `k`
        collapses ρ but barely moves the one-shot error (figures deleted with the rest of the unconfigured ρ
        evidence above), and the ρ-minimizing
        `k` sits ~40× *above the maximum* of the whole per-cell `ρV/a_P` distribution — i.e. the degenerate
        limit `schur_a_p → 0`, `Ŝ⁻¹ → 0`, which simply switches the pressure correction off. On the real
        production march it is **slower**: shipped auto-`k` 348 s / 8 steps vs `k×4` 447 s (28% slower) at
        an identical residual trajectory. **The shipped per-apply `mean(ρV/a_P)` calibration is
        near-optimal — do not "fix" it**, and do not make the Schur "shift-consistent" with the
        pseudo-transient `a_P(1+β)` either (that direction is strictly worse at every β).
      **Root cause:** the MSIMPLE Schur is a *constant-coefficient* (scaled pressure-mass-matrix) Poisson,
      which is a near-Stokes/low-Re approximation and degrades as convection strengthens — exactly the
      high-Re/recirculating regime here.
      - **⚠️ The "obvious" fix — a better Schur (stabilized LSC) — WAS BUILT, LOST BADLY on the
        coupled solve, and was DELETED 2026-09-13 (#371).** It was the algebraic, nonuniform-mesh
        stabilized least-squares commutator of Elman, Howle, Shadid, Silvester & Tuminaro (2007), with
        the viscosity cancelled so it served a variable-viscosity closure. Measured on one shifted solve
        at a developed/separated pitzDaily state (no shift recorded):

        | Schur | cycles | wall |
        |---|---|---|
        | **msimple** | **13** | **38.9 s** |
        | lsc (`cycles=4`) | 96 | 526 s |
        | lsc (`cycles=8`) | 82 | 662 s |

        6–7× the cycles and 13–17× the wall time, with both solves genuinely converged
        (`lin_rel ~2e-9`), plus ~2.9× slower on the coupled channel at an identical residual trajectory.
        **Why the flow-only win does not transfer:** LSC *does* beat MSIMPLE on the isolated flow block
        (9 vs 15 GMRES at Re=1e4), but on the coupled block-*diagonal* preconditioner under the
        pseudo-transient shift, a better isolated flow-Schur does not reduce *coupled* cycles — the
        coupled iteration is not limited by the flow block's Schur quality. The isolated flow-only win
        named no mesh or configuration and nothing selected the strategy, which is why it was deleted.
      - PCD remains deprioritized regardless: its auxiliary pressure convection–diffusion operator
        carries finite-element boundary recipes that do not transfer cleanly to cell-centred FVM.
      - **What a preconditioner can and cannot change — state this precisely, both halves are measured.**
        *While the linear solve actually converges*, swapping the preconditioner changes **cost only**:
        msimple vs LSC gave coupled residual trajectories identical to **5 significant figures**, so the
        Newton direction, the accepted step, and whether the march converges are all preconditioner-
        independent. That rules out a whole class of experiment — you cannot precondition your way out
        of a stalled march, only out of an expensive one.
        **But the guarantee is conditional on convergence, and it fails when a preconditioner is stale
        enough to degrade the solve.** Measured 2026-07-25 on the pitzDaily cold-IC march, refreshed vs
        unrefreshed at *identical* step indices: the unrefreshed arm sat at **α = 0.5 for steps 16–23
        while needing 53–85 cycles**, and the refreshed arm took **α = 1.0 at 10–13 cycles**, with
        different residual trajectories (rel 4.45e-2 vs 4.13e-2 at step 20). So a sufficiently stale
        preconditioner *does* change the step. The mechanism is not isolated — the natural reading is
        that the degraded solve is truncating or stagnating rather than reaching its tolerance, so the
        returned `δ` is no longer the tolerance-defined one — and the block family's forward solve then ran
        `rtol=1e-3` with `stagnation_iters=40`, which makes that reachable. **Practical rule:** treat
        "preconditioner ⇒ cost only" as true when solves converge comfortably, and stop trusting it
        once the cycle count is climbing toward the solver's limits.

## Bounding the line-search fallback (`fallback_growth`), 2026-09-09

**BUILT, and it removes a recorded catastrophe outright.** The `beta_start_warm = 0.05` arm on pitzDaily
was on record as unusable -- 64 steps on the target rung, nine escalations, `beta` driven to 16.0, killed
three orders from the stopping bar. **That was this hole**, and with it closed the same arm completes in
58 steps / 390 cycles / 521 s with the target rung in 16.

- **The hole.** When no rung of `backtracking_line_search`'s ladder reduces the measure, the search
  returned the longest **finite** one, and "finite" is a very weak bound: on pitzDaily it returned a rung
  the search had *itself measured* at `1.567e+127`, and `newton_march` then took it as the next anchor
  unconditionally (`state = outcome.phi`, no best-of-attempts, no fallback to the pre-step state). The
  fallback exists to avoid a null step, which is a fair argument against the *shortest* rung and no
  argument at all for keeping one 130 decades above the reference. The search had the number in hand and
  did not use it. A 2026-07-27 change had bounded this fallback in *length*; its **damage** was left
  unbounded.

- **The change.** `backtracking_line_search` gains `fallback_growth` (default `100.0`): a fallback rung
  must be finite **and** under `fallback_growth * reference_norm`. Where several qualify the longest
  still wins, so the fallback shortens rather than vanishing. Where **none** qualifies the search returns
  `alpha = 0` -- iterate untouched, reporting `reference_norm`, which is exactly its measure there -- so
  nothing poisoned is handed on, and the zero `alpha` is precisely the signal the march's `on_alpha`
  retry already exists for.
  ⚠️ **A search with no finite rung at all is a separate case and is deliberately UNCHANGED**: it still
  reports a non-finite measure, because the divergence guard, the acceptance policy and the Newton
  driver's convergence guard all test that number with `isfinite`
  (`test_a_search_with_no_finite_rung_reports_a_NON_FINITE_measure` pins it).

- **Why 100, measured rather than chosen.** Across two *healthy* pitzDaily arms the fallback fired on
  **0 of 360** inner iterations -- the dual-time inner search is strict-descent, so an accepted rung
  always reduces the measure and a fallback is by construction never reached. The largest benign fallback
  seen anywhere was **4.5x** (and an existing unit test pins a legitimate 5x one); the harmful ones were
  **9.2e+28** and above. Twenty-odd decades of daylight, so the value is not delicate.

- **Free on a healthy march, confirmed empirically rather than by argument.** A re-run reproduced its
  cold and warm rungs *exactly* -- 26 steps / 130 cycles / 4 clips and 12 / 123 / 4, identical to the
  pre-change run -- which is what the 0-of-360 measurement predicts.

- **What it is worth, measured two ways.** At the excursion it was built for, same anchor and same
  direction: **before** `5.488e-02 -> 5.067e+27` at `alpha = 1.000`; **after** `5.488e-02 -> 7.301e-02`
  at `alpha = 0.125`, the step ending *below* its rung's own reference. And end to end, on the shipped
  bundle (`MultipleCorrectionGradient` at `STENCIL_REACH = 3`), the `0.05` arm that previously could not
  finish:

  | rung | steps | cycles | wall | esc |
  |---|---|---|---|---|
  | Re/100 | 28 | 123 | 186 s | 0 |
  | Re/10 | 14 | 116 | 131 s | 0 |
  | target Re | **16** | 151 | 204 s | 2 |
  | **TOTAL** | **58** | **390** | **521 s** | 2 |

  `x_r/h` **8.0686**, the usual root. The target rung ran at `alpha = 0.008` with two escalations -- the
  globalization working hard and working as designed, which is what it could not do while a poisoned
  anchor was being handed to it.

- ⚠️ **DO NOT READ "16 STEPS" AS A GOOD NUMBER.** A healthy target rung on this case takes **14 to 20**
  steps across the arms on record (`0.5` -> 17, `0.1` -> 14, the `multcorr` arm -> 20). 16 is mid-field.
  What is notable is that the rung completed at all rather than becoming the 64-step failure.

- **CONSEQUENCE, and it is the reason to land this on its own:** a recorded verdict that
  `beta_start_warm = 0.05` is "catastrophic", and the binding statement that carrying `beta` across
  Reynolds rungs is *blocked* because it means starting warm rungs low, were **both properties of this
  hole** rather than of the shift. They are retracted. What remains open is the ordinary trade --
  descent steps saved against a rung meeting a shift below what it can take -- for which the reference
  points are the arm above and the recorded `0.1` arm (59 / 403 / 712, zero escalations, on the previous
  bundle).

- ⚠️ **This change shipped with its documentation sync INCOMPLETE, and the reason is the recorded blind
  spot rather than carelessness (found and repaired 2026-09-09, one merge later).** Four passages went on
  describing the unbounded fallback after it was gone: `backtracking_line_search`'s own **docstring
  summary line** (contradicted by the `fallback_growth` entry a hundred lines below it), the comment in
  `DualTimeStep`'s inner loop that folds a non-descent into `a_min`, `MarchLogger.on_inner`'s explanation
  of the `alpha` column, and `solve-march.md` in two places -- one of them the log-reading rule that
  `a_min=0.000` beside `alpha=1.000` is the non-descent signature, which stopped being the only one the
  moment `alpha = 0` became reachable. **The Stale-Record Check's identifier grep cannot see any of
  them**: `fallback_growth` is a *new* name, so nothing was renamed, and every false claim is spelled in
  ordinary English -- "the longest finite rung", "the longest finite trial step". That is precisely the
  case the checklist's "grep for the CLAIM, not only for the symbol" exists for, and it was still missed.
  **The phrase to grep when this behaviour changes again is `longest finite`** -- it reaches every site
  that describes this fallback, including the several that are correct, which is what a claim-grep is
  supposed to do: it hands you the whole set to adjudicate rather than only the ones that were renamed.

## The shift floor sets the target station's rate; releasing it (`release_floor`), 2026-10-10

**Configuration of every number here.** pitzDaily, the shipped `case.yaml` at commit `7972225`: residual
stop, `rtol` 0.3, restart 15, `max_restarts` 14, `refresh_on_cycles` 2, dual time 5 / 0.01, cycle budget
42, `CflResidualDualTimeControl(beta_start 0.5, beta_min 0.005)`, `refit_beta_floor` 0.05, field split
`SimpleSmoothed` / `JacobiSmoothed`, `turbulence_damping` 3, 16 x 1 momentum-only viscosity ramp from 100,
stop `atol` 1e-5 row-scaled. jax 0.11.2, lineax 0.1.1, CPU, 4-core Linux (16 GB), one run per arm.
**Cycles are restart cycles as the march log's `cyc` column counts them** (`lineax`'s raw count less two
per inner solve); the log's second column is wall-clock seconds, not work.

**The capture.** That march with every step and inner iterate checkpointed (`PITZ_CHECKPOINT_KEEP=500
PITZ_INNER_DUMP_ABOVE=1`): 31 steps, 161 cycles (85 on the ramp, 76 on the target station over 67 inner
solves), final `|R|` 8.345e-06, `x_r/h` 8.07, no retries, 1036 s. `beta` reaches the floor at step 13
(of a 16-step ramp) and sits there for the whole target station (steps 17-31), where the residual falls
linearly at 0.55-0.84 per step.

⚠️ **Unverified since 2026-10-18:** until then the probe built its arms' steps once and discarded each
re-fit's update, so where the split was traced (#290, the same day as this table) its arms applied the
split from the steps' build, not the re-fit at their own state and shift. Whether this table ran on the
host split (re-fitted in place, so correct) or the traced one is not recorded; it was not re-run. The
harness now derives every arm from the replay's current step.

**The probe** (`validation/pitzdaily_openfoam/endgame_shift_probe.py`): ONE outer step from a checkpointed
state at `beta` 0.005 / 0.0015 / 0.0005 / 0, the inverse re-fitted at that state at `max(beta, 0.05)`
(`shipped`) or at the shift itself (`tracking`, zero included), under the file's inner loop (5 / 0.01) or a
tight one (12 / 1e-4); the contraction is the row-scaled residual's, in the measure the march judged that
step by. **Its control -- the march's own step re-taken with the inverse walked to where the march had it
-- reproduced the record at states 16, 21 and 27** (ratio to four figures, raw count, inner count,
`alpha`); a second run at states 18-20 printed a mismatch because it used the tight inner loop for the
control, a harness defect since fixed (its ratios still match the record to four figures). `predict` is
`beta / (lambda + beta)` with `lambda = beta_c (1 / rho_c - 1)` fitted from the control. Cells are
ratio / cycles / inner solves / `alpha`, ✓ where the inner loop met its target; "state" is the number of
steps completed, so state 20 is where step 21 starts.

| state | `beta` | shipped inverse, file inner | tracking inverse, file inner | tracking inverse, tight inner | predict |
|---|---|---|---|---|---|
| 16 (arrival) | 0.005 | 0.989 / 8 / 5 / 0.5 | 0.99 / 9 / 5 / 0.5 | 0.988 / 18 / 10 / 0.5 ✓ | 0.989 |
| | 0.0015 | 1.05 / 46 / 4 / 0.06 | 1.06 / 42 / 3 / 0.06 | 1.06 / 42 / 3 / 0.06 | 0.964 |
| | 0.0005 | 1.03 / 42 / 3 / 0 | 1.00 / 42 / 3 / 0.5 | 1.00 / 42 / 3 / 0.5 | 0.900 |
| | 0 | 1.04 / 42 / 3 / 0 | 1.01 / 42 / 3 / 0 | 1.01 / 42 / 3 / 0 | 0 |
| 18 | 0.0015 | | | 0.39 / 27 / 9 / 0.125 | 0.473 |
| | 0 | | | **0.995** / 42 / 3 / 0 | 0 |
| 19 | 0.0015 | | | 0.32 / 16 / 10 / 0.25 ✓ | 0.356 |
| | 0 | | | 0.0096 / 29 / 8 / 0.06 | 0 |
| 20 | 0.0015 | | | 0.283 / 11 / 9 / 0.5 ✓ | 0.297 |
| | 0 | | | **1e-4** / 20 / 10 / 0.25 ✓ | 0 |
| 21 | 0.005 | 0.555 / 5 / 5 / 1 ✓ | 0.555 / 5 / 5 / 1 ✓ | 0.555 / 8 / 8 / 1 ✓ | 0.556 |
| | 0.0015 | 0.271 / 8 / 5 / 0.5 ✓ | 0.270 / 6 / 5 / 0.5 ✓ | 0.270 / 11 / 9 / 0.5 ✓ | 0.273 |
| | 0.0005 | 0.112 / 9 / 5 / 0.5 | 0.111 / 7 / 5 / 0.5 | 0.110 / 14 / 9 / 0.5 ✓ | 0.111 |
| | 0 | 0.0167 / 10 / 5 / 0.5 | 0.0136 / 8 / 5 / 0.5 | **1e-4** / 14 / 9 / 0.5 ✓ | 0 |
| 27 (tail) | 0.005 | 0.664 / 4 / 4 / 1 ✓ | 0.663 / 4 / 4 / 1 ✓ | 0.663 / 10 / 8 / 1 ✓ | 0.664 |
| | 0.0015 | 0.405 / 8 / 4 / 1 ✓ | 0.404 / 5 / 4 / 1 ✓ | 0.404 / 10 / 8 / 1 ✓ | 0.372 |
| | 0.0005 | 0.204 / 12 / 4 / 1 ✓ | 0.203 / 8 / 4 / 1 ✓ | 0.202 / 12 / 8 / 1 ✓ | 0.165 |
| | 0 | 0.0042 / 17 / 5 / 1 ✓ | 0.0044 / 9 / 5 / 1 ✓ | < 1e-4 / 15 / 9 / 1 ✓ | 0 |

- **At a settled state the floor is the rate and zero shift is Newton.** At state 21 the contraction
  tracks `beta / (lambda + beta)` to three figures at every shift, and one zero-shift step with the tight
  inner loop reaches the 1e-5 bar for 14 cycles, where the march took ten more steps and 43 cycles. At
  the tail (state 27) the gain is steps, not cycles: one step for 15 against four for 16. The single-mode
  fit under-predicts there by 9-23 % at the intermediate shifts (more than one slow mode).
- **At arrival the same step fails, and the inverse is not why.** Every shift below the floor raises the
  residual from state 16, with the line search collapsing and 42-cycle steps, identically under the
  tracking inverse; the tight inner loop at the floor meets its own target and moves the outer residual
  1 %. The transition is sharp: a zero-shift step fails from state 18, strains from 19 (0.0096 with
  `alpha` 0.06 and the inner target unmet) and is Newton from 20 -- **the first state after a step at the
  floor took `alpha` = 1** (steps 17-19 took 0.5). That is the gate the release uses.
- **The refit floor is worth cycles, not rate:** fitting the inverse at the shift instead of at 0.05
  saves 1-8 cycles per step and leaves the contraction unchanged to three figures.
- **The tight inner loop buys nothing a second step does not.** At zero shift from state 21 the file's
  loop (5 inner, forcing 0.3) contracts 0.014-0.017, the tight one (9 inner) 1e-4 for 14 cycles; two
  file-loop steps cost about the same. Inexact Newton at forcing 0.3 converges linearly per inner solve,
  so what matters is the number of inner solves, not how they are grouped into outer steps.

**The march** (`PITZ_RELEASE_FLOOR=1e-4`, `release_alpha` 1.0 -- since renamed `settle_alpha`, everything else as above). Steps 1-20 are
identical to the capture in `|R|`, inner count and cycles; step 20 is the first at the floor with
`alpha` 1.000, step 21 runs released.

| | capture | released to 1e-4 |
|---|---|---|
| outer steps | 31 | **22** |
| cycles | 161 | **141** |
| target station: steps / cycles / inner solves | 15 / 76 / 67 | **6 / 56 / 30** |
| from step 21: steps / cycles / wall | 11 / 48 / 191 s | **2 / 28 / 117 s** |
| final `|R|` | 8.345e-06 | 1.729e-06 |
| `x_r/h` | 8.07 | 8.07 |

The released steps contract 0.040 and 0.030 (step 21 at `alpha` 0.25, step 22 at 1.0), against 0.028
predicted from `lambda` at `beta` 1e-4, and each costs more cycles than a floor step (11 and 17 against
4-5) -- the inverse is still fitted at 0.05, a 500x mismatch, which is the deferred half below.
⚠️ **Wall clock is indicative only:** steps 1-20 did identical work in 661 s and 605 s in the two runs
(different compilation-cache state, light concurrent work in both), so a ~10 % spread; the tail's
191 s against 117 s is outside it, but it is one run each.

**Not built, and why (deviations from the agreed design, stated where they were made):** releasing the
refit floor with the shift (no seam reaches the refresh hook from the control; worth 1-8 cycles per
released step by the table above, i.e. most of the released steps' extra cost) and tightening the inner
loop on the released step (a static field, so a recompile, for nothing per the last bullet).

## The ramp's length is the shift's descent, not the viscosity's: `grow` x `stations`, 2026-10-11

**Configuration.** pitzDaily, shipped `case.yaml` plus `PITZ_RELEASE_FLOOR=1e-4` on every arm (so the
baseline is the released march above), `PITZ_GROW` (the control's growth factor, class default 1.5,
never varied before) and `PITZ_RAMP_STATIONS`; everything else as recorded for the capture (residual
stop, `rtol` 0.3, `refresh_on_cycles` 2, dual time 5 / 0.01, `beta_start` 0.5, `beta_min` 0.005,
`refit_beta_floor` 0.05, momentum-only ramp from 100, `turbulence_damping` 3, stop `atol` 1e-5).
jax 0.11.2, CPU, 4-core Linux, one run per arm, run back to back; cycles are restart cycles as the log
counts them; wall is the marching time after step 1 and is indicative only. Every arm reached `x_r/h`
8.07 with no retry.

| `grow` | stations | steps | cycles (ramp / target) | `beta` floors at | first `alpha` 1 at the floor | released | wall |
|---|---|---|---|---|---|---|---|
| 1.5 | 16 (baseline) | 22 | 141 (85 / 56) | 13 | 20 | 21 | 722 s |
| 1.5 | 8 | 23 | 166 (33 / 133) | 13 | 21 | 22 | 735 s |
| 3 | 16 | 19 | 134 (101 / 33) | 7 | 13 | 18 | 782 s |
| 3 | 8 | 18 | 145 (54 / 91) | 8 | 16 | 17 | 707 s |
| **3** | **12** | **16** | **111 (76 / 35)** | 7 | 14 | 15 | **597 s** |

The last arm was run after the first four, as the prediction the pattern below makes: a ramp that
arrives about where the state settles at the floor.

- **The step count is nearly invariant, and what is conserved is steps AT THE FLOOR before the state
  settles** (⚠️ at these paces only: a slower ramp settles sooner, see the prototype entry below): 6-9 in every arm (the first full-length step at the floor is the 7th, 8th, 6th, 8th and
  7th floor step). That is the recorded "insufficient elapsed pseudo-time" finding in a new form: the
  transient needs a fixed amount of pseudo-time at the largest timestep, and no schedule removes it.
- **The viscosity step does not move `alpha`; the shift does.** The two `grow` 3 arms have identical
  `alpha` histories through step 6 at 1.33x and 2x viscosity per step, and the two `grow` 1.5 arms through
  step 7. The clipping during the ramp is `beta` reaching what the state can take, not the station jump
  (consistent with the predictor probe's 5-15 % jump share). **So `alpha` cannot drive a station
  controller: it is not responding to the ramp.**
- **Reach the floor fast, then ramp AT the floor.** Floor steps at a higher viscosity take their full
  timestep (`alpha` 1 from step 13 in the 16-station `grow` 3 arm, 6-7 cycles); floor steps at the target
  from an unsettled state are clipped to half and cap-limited (`L`, 7-11 cycles, the 8-station arms).
  Ending the ramp when `beta` floors (`grow` 3, 8 stations) is therefore the WRONG pairing, and it is
  what a viscosity keyed to `beta` would do -- that design is closed before being built (ledger entry).
- **The waste is the ramp outlasting the settling.** In the 16-station `grow` 3 arm steps 13-16 ran at
  `alpha` 1 at the floor before the schedule arrived: 24 cycles. Twelve stations end the ramp one step
  after the settle and cost 111 cycles -- **21 % fewer than the released baseline and 31 % fewer than
  the shipped march's 161**, at 16 steps against 31.
- **What an adaptive ramp therefore keys on:** not `alpha` per station, but the settle signal -- the
  first full-length step at the floor, the same gate the release uses. Pace is a secondary knob (2x per
  step cost nothing in `alpha`); the end is the sensitive part. Expressing it needs the homotopy to see
  the previous report (built 2026-10-11: `ResidualHomotopy.enter(step, previous)`, `solve.RampSchedule`).

**pitzDaily's default since 2026-10-11 (project owner's decision, pitzDaily only):** its `case.yaml`
ships `grow` 3, 12 stations and `release_floor` 1e-4. Confirmed by a run of the file as shipped,
no `PITZ_*` overrides: 16 steps, 111 cycles (ramp 76 / target 35), shift at the floor from step 7,
released at step 15, final `|R|` 1.98e-6, `x_r/h` 8.0686, no retry -- the sweep's arm exactly. The class defaults (`grow` 1.5, no release) and
bfs3d's case are unchanged; bfs3d is #678. ⚠️ **"The shipped `case.yaml`" in any pitzDaily entry dated
before 2026-10-11 means `grow` 1.5 x 16 stations with no release** (31 steps / 161 cycles at its last
measurement).

## Ending the ramp on the settle signal: a prototype, 2026-10-11

**Configuration.** pitzDaily, `case.yaml` as shipped that day (`grow` 3, `release_floor` 1e-4, ramp anchor
100, momentum-only, one step per station, everything else as in the sweep above), with
a prototype standing in for `ViscosityRampHomotopy` (`settle_ramp_probe.py`, deleted once the library
reproduced it -- the arms are now `PITZ_RAMP_END=settled`, `PITZ_RAMP_FINISH`, `PITZ_RAMP_STATIONS`, see
"Built" below): station `s`
at step `s` until the first step that ran off the target at `shift <= beta_min` with `alpha >= 1` (the
release's gate), then the target from the next step. `PITZ_RAMP_STATIONS` is then only the pace. jax
0.11.2, CPU, 4-core Linux, one run per arm, run back to back; restart cycles as the log counts them.
Every arm `x_r/h` 8.0686, no retry.

| arm | pace (per step) | settle after step | jump at the settle | steps | cycles (ramp / target) |
|---|---|---|---|---|---|
| 12 fixed stations (shipped) | 1.47x | -- (arrives at 13, settles on the target at 14) | -- | 16 | 111 (76 / 35) |
| settle-ended, pace 16 | 1.33x | 13 (station 12 of 16) | 3.16x | 17 | 124 (83 / 41) |
| settle-ended, pace 24 | 1.21x | **9** (station 8 of 24) | **21.5x** | 16 | 118 (54 / 64) |

- **The jump is tolerated, whatever its size.** A 21.5x viscosity change in one step at the floor
  shift cost no retry and no escalation; the first target step ran at `alpha` 0.5 and 12 cycles.
- **But the jump undoes the settle, and the target has to settle again**: 2 steps after a 3.16x jump, 5
  after a 21.5x one, before the first full-length step there and the release. The shipped 12-station
  ramp avoids this by arriving one step BEFORE its settle, so the one settle it pays is on the target.
- **⚠️ "6-9 floor steps before the settle" (the sweep above) is NOT conserved -- it depends on the pace.**
  At 1.21x per step the state settled at the 4th floor step; at 1.33x and 1.47x, the 7th and 8th. A
  slower-moving problem settles sooner. What WAS roughly conserved across all three: 16-17 steps and
  111-124 cycles in total.
- **Verdict against the pre-registered bar ("the same cycles within a few"): not met.** 118 and 124
  against 111, +6 % and +12 %. What the rule does buy is insensitivity: across paces 16-24 it stays within
  12 % of the hand-tuned optimum, where fixed counts at the same `grow` cost 134 (16 stations) and 145
  (8 stations). A pace of 12 or faster never settles before arriving, so the rule reproduces the shipped
  arm exactly there.
- **What it does not tell apart:** whether a smaller jump at the settle (an accelerated finish over two
  or three steps instead of one jump) would keep the first settle's benefit. Unmeasured.

**Two further arms, same configuration (2026-10-11).**

| arm | ramp ends after step | target settles at step | steps | cycles |
|---|---|---|---|---|
| 12 fixed stations (shipped) | 12 | 14 | 16 | 111 |
| settle-ended, pace 24, the 21.5x walked in 3 steps (`PITZ_RAMP_FINISH=3`, 2.78x each) | 11 | 14 | 16 | **114** |
| settle-ended, pace 24, one jump | 9 | 14 | 16 | 118 |
| settle-ended, pace 16, one jump | 13 | 15 | 17 | 124 |

- **The 3-step finish meets the pre-registered bar (114 against 111) -- but not for the predicted reason.**
  The smaller final steps did NOT keep the ramp's settle: steps 10-13 went back to `alpha` 0.5 and the
  target settled again at step 14.
- **⚠️ THE TARGET SETTLES AT STEP 14 IN THREE OF FOUR ARMS, whatever the ramp did before.** So on this
  case the target's settle is set by the march's own transient, not by the ramp; a ramp only has to be
  over a step or more before it, and costs when it is not (the pace-16 arm, done at 13, settles at 15).
  The residual entering the target's first full-length step was 2.0-2.5e-3 in all four (about 7 % of
  `|R0|`), an observation, not yet a predictor.

**The full-step ratio `rho1` -- what the settle test actually reads (the prototype's trace; now every march's `rho1` column, pace 16,
one jump; the traced march is identical to the untraced one cycle for cycle and to `|R|` 1.788e-06).**
`rho1 = |G(phi + delta)| / |G(phi)|` of each step's FIRST inner iteration, the number the line search
compares with one (strict descent) to accept the full step:

| step | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | **13** | 14 (3.16x jump) | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `rho1` | 1.65 | 4.42 | 3.61 | 3.38 | 1.89 | 1.27 | 1.23 | 1.13 | 1.04 | **0.90** | 1.77 | 0.91 |

- **It approaches one smoothly over the last four floor steps** (1.23, 1.13, 1.04, 0.90: about -0.1 per
  step), so the crossing is visible a step or two ahead -- a two-point linear extrapolation made at step
  12 predicts it at step 13, and at step 11 predicts no crossing on the next step, both right. It is
  not uniformly smooth: the same extrapolation made at step 9 (after the fast fall 3.38 -> 1.89 -> 1.27)
  predicts a crossing at step 10, three steps early.
- **A viscosity jump raises it**: 0.90 -> 1.77 for 3.16x, and the target then re-settles in one step
  (0.91). After the 21.5x jump of the pace-24 arm the re-settle took five steps (`rho1` not traced there).
  So `rho1` responds to the size of the last viscosity step, which is what an adaptive ramp rate needs to
  see -- and what `alpha` (a halving ladder, so 1 or 0.5 here) cannot show.
- **What it does not tell apart yet:** whether `rho1`'s rise after a viscosity step scales predictably
  with the step's size (two points: 3.16x -> +0.87; the 1.33x ramp steps are entangled with the
  transient), which a rate controller would rely on.

**Built into the library, and reproduced exactly (2026-10-11).** `solve.RampSchedule(end_when=
step_control.settled, finish=...)`, the case file's `ViscosityRamp(end: settled, finish: ...)`, and the
step reports' `full_step_ratio` (the log's `rho1`). `PITZ_RAMP_STATIONS=24 PITZ_RAMP_END=settled
PITZ_RAMP_FINISH=3` through `compare.py` (same configuration as above, jax 0.11.2, 4-core Linux):
**16 steps, 114 cycles, per-step costs `3 4 5 6 7 9 7 7 6 7 8 8 7 6 12 12` and final `|R|` 2.378e-06 --
the prototype's arm to the last printed digit**, `x_r/h` 8.0686, no retry, ramp ended after step 9 and
the target settled at step 14. Its `rho1` column: 0.19 0.25 0.86 1.57 3.89 3.43 1.92 1.14 **0.83**
(settle) 1.29 1.53 2.08 1.29 **0.91** (target settles) 3.15 0.34. Opt-in; pitzDaily's file keeps 12 fixed
stations (111 < 114).
The file as shipped (12 fixed stations), on the same code: 16 steps, 111 cycles, per-step costs and
final `|R|` 1.979e-06 identical to the confirmation run before the change -- the fixed walk and the
extra `full_step_ratio` carry move nothing.

## Where the ramp starts: anchor depth at the shipped march, 2026-10-11

**Question (from the project owner):** can a cheap probe choose the ramp's starting multiplier -- far
for hard problems, near (or none) for easy ones? First established whether the start still matters.

**Configuration:** pitzDaily `case.yaml` as shipped at `61a1430` (`CflResidualDualTimeControl`
`beta_start` 0.5, `beta_min` 0.005, `grow` 3, `release_floor` 1e-4; `ViscosityRamp` 12 stations x 1,
`scale: flow`, fixed -- no settle end; field split `simplesmooth`/`jacobi`; dual time 5 / 1e-2), only the
anchor varied by `PITZ_RAMP_ANCHOR` (harness commits `81857c7`, `8891cca`; `1` removes the ramp and the
march self-starts by `hybrid_initialize` at the case's own viscosity). Linux x86_64, 4 cores, jax 0.11.2,
one arm at a time, nothing else running.

| anchor | steps | restart cycles | redone steps | wall |
|---|---|---|---|---|
| 100 | 16 | **111** | 0 | 701 s |
| 10 | 17 | 178 | 2 | 1100 s |
| 3 | 18 | 217 | 3 | 1348 s |
| 1 (no ramp) | 16 | 246 | 5 | 1585 s |
| 300 | 26 | 255 | 3 | 1543 s |
| 1000 | stopped at 38 | 502 by then | 10 by then | 5018 s by then |

All four reach the same root (`Ux` in [-2.693, 10.225]; `x_r/h` 8.07 read for anchors 100 and 1). The
anchor-100 arm reproduces the recorded 16 steps / 111 cycles exactly. ⚠️ The cycle column counts only
ACCEPTED attempts, so the redone steps' discarded work is visible only in wall clock -- which is why the
wall ratios (1.0 / 1.6 / 1.9 / 2.3) run ahead of the cycle ratios.

- **pitzDaily does not NEED a ramp under today's march** (the old ~490-step cold crawl is gone), but the
  ramp is worth 2.2x in cycles and 2.3x in wall. The anchor is a cost lever, not a reachability one, here.
- **Cost falls with depth down to 100 and rises past it, for a different reason on each side.**
  Shallower than 100 the cost is redone steps at a small shift early in the march (the state far from a
  moving station's root while `beta` falls). Deeper than 100 the ramp is the cleanest of any arm
  (anchor 300: nothing redone and `alpha` 1 or 0.5 through step 18; anchor 1000: full-length steps to
  step 5) and the cost is all at the target, in the release (next bullet). Anchors 300 and 1000 were run
  the same day on the same harness; 1000 was stopped at step 38 (owner's call), the fault already clear.
- **⚠️ THE RELEASE RE-FIRES AFTER IT FAILS -- a defect in `ShiftStrengthControl._floored`, not in the
  anchor.** Anchor 300 settled at step 18 (`alpha` 1 at the floor, `rho1` 0.96 against anchor 100's 0.87
  on its settling step) and released at 19; the released step collapsed (`alpha` 0, redone twice, beta
  escalated to 2e-4). The control then put beta back to 0.005, the next step was full length at the
  floor, which IS `settled` again, so the release fired again at 21, 23 and 25 -- every other step --
  until the residual was small enough (about 3e-4) for a released step to hold. Anchor 1000 did the same
  from step 24 for eight cycles (each failed release ~360 s and 28 cycles), the residual creeping
  1.6e-3 -> 7e-4 over 14 steps. Nothing in the rule remembered a failed release. Fixed by the backoff
  the next day ("The release backoff" below), against a bar set here and restated there.
- **The early full-step ratio ranked the four shallow arms** (`rho1` at step 3: 1.08 / 4.51 / 6.43 / 89.25
  for anchors 100 / 10 / 3 / 1, first redo at step -- / 4 / 11 / 3) but is not a probe signal on its own:
  the cheapest arm ran `rho1` 1.3-4.6 at the floor with no step redone, and on bfs3d it gave no lead on
  the settle (#685).
- **The same answer on bfs3d (#685, its shipped `grow` 1.5 x 12, lineax stop):** anchor 10 / 100 / 1000
  cost 239 / 193 / 329 all-attempt cycles; anchor 1000 ran a clean ramp, then 64 steps at the floor with
  the release off (shipped), contracting 0.91-0.95 per step. So anchor 100 is the cheapest start measured
  on both cases, by different routes on each side. ⚠️ At a fixed 12 stations a deeper anchor is also a
  larger viscosity jump per station (1000: 1.78x, 100: 1.47x), and those jumps land with `beta` already
  at the floor (bfs3d's anchor-1000 `rho1` ~1e22 at steps 12-13); a deep anchor with stations scaled to
  hold the per-station jump has not been run.
- **Verdict on the probe the owner asked about:** not needed for these two cases -- one start (100) is
  the cheapest of those measured on both. It would pay only on a case far from both in difficulty, of
  which none is in the repository.

**The inner stop rule on pitzDaily, the same day (counterpart of #685's bfs3d arms).** The shipped file
with `PITZ_FORWARD_STOP=lineax PITZ_REFRESH_ON_CYCLES=3` (bfs3d's stop and refresh), anchor 100. Stopped
at step 15 of an expected 16 by a waiter's time limit (signal 15); compared at step 15 against the
shipped run (residual stop, refresh 2):

| at step 15 | shipped (`residual`, refresh 2) | `lineax`, refresh 3 |
|---|---|---|
| `\|R\|` | 7.08e-5 | 5.41e-5 |
| settle | step 14 | step 14 |
| redone steps | 0 | 0 |
| cumulative cycles | 99 | 157 |
| wall | 626 s | 1190 s |

On pitzDaily the stop moves cost (1.9x wall; the cycle gap understates it, since `lineax`'s count drops
one cycle per solve the residual stop counts) and not the trajectory, even at `grow` 3. On bfs3d it moved
the trajectory too (#685: under lineax `grow` 3 redid nine steps, under `residual` three). So the
stop-and-rate interaction is bfs3d's; whether it runs through the cycle-triggered retry (the two stops
count cycles differently) is unseparated and checkable from #685's logs by splitting each arm's redone
steps by trigger.

## bfs3d: the release, `grow` 3 and the settle-ended ramp (#678), 2026-10-11

⚠️ **Every arm below predates #290/#694 (a re-fit ends the step; traced split), which changed bfs3d's
march**: at `46af0749` the shipped file takes 36 steps / 166 cycles / no retry against arm 1's 28 / 174 /
one, at about the same wall (`solve-open-directions.md` 6b, bfs3d table). Compare these arms with each
other, not with a run on a later tree.

**Configuration.** `validation/bfs3d_openfoam/compare.py` at `61a14308` (#677 merged), `case.yaml` as
shipped (12 stations x 1 step, `turbulence_damping` 5, `CflResidualDualTimeControl` `beta_start` 0.5,
`beta_min` 0.005, `grow` 1.5; field split `simplesmooth` + Jacobi, `refit_beta_floor` 0.05, refresh on 3
cycles, retry on 10 cycles / `alpha` 0.01, cycle budget 42, inner 5 / 0.01, forward rtol 0.3, stop `atol`
1e-5; the inner solve stops by lineax's default rule -- the file sets no `stop`, where pitzDaily's ships
`stop: residual` with `refresh_on_cycles` 2, which counts cycles RUN, one more than lineax for the same
work, so the cycle-triggered refresh and retry fire at a different difficulty), each arm one `BFS3D_*`
edit of it and its banner checked. Apple M3 Pro, 11 cores, 18 GB, macOS,
jax 0.10.2; one run per arm, back to back through `run_case.sh`, nothing else running. Wall clock
indicative. **Cycles are counted two ways**: "accepted" is the log's `cum` (the accepted attempt of each
step only), "all" adds every rejected attempt's inner cycles from the step blocks -- `cum` hides a retry's
cost entirely, and it is most of arm 3's.

| arm | steps (ramp + target) | cycles accepted (ramp / target) | cycles all | inner solves | steps redone | final `\|R\|` | wall (s) |
|---|---|---|---|---|---|---|---|
| 1 shipped | 28 (13 + 15) | 174 (62 / 112) | 193 | 82 | 1 | 9.06e-6 | 1191 |
| 2 `RELEASE_FLOOR=1e-4` | **24** (13 + 11) | **173** (62 / 111) | **192** | 75 | 1 | 2.22e-7 | 1126 |
| 3 + `GROW=3` | 19 (13 + 6) | 214 (125 / 89) | **459** | 70 | **9** | 3.46e-7 | 2011 |
| 4 + `RAMP_END=settled FINISH=3 STATIONS=24` | 19 (9 + 10) | 187 (64 / 123) | 267 | 67 | 5 | 3.00e-7 | 1315 |

Mid-span `x_r/h` 8.361 in all four. Converged states against arm 1: relative L2 1.78e-4 (arms 2, 3 and 4;
arms 3 and 4 agree with each other to 1.1e-6), against 5.67e-4 that arm 1's own state moved over its last
two steps -- one root, the shipped arm merely stopping at the looser residual.

| arm | shift reaches the floor | first full-length step at the floor (settle) | release fires | `rho1` from the floor to the settle |
|---|---|---|---|---|
| 1 | 19 (target) | 22 (target) | -- | 0.92, 0.29 (e1), 0.36, 0.29 |
| 2 | 19 (target) | 22 (target) | 23 | as arm 1 |
| 3 | 12 (ramp) | 17 (target) | 18 | 1.68, 1.15 (e1), 2.23 (e1), 1.80, 0.29 (e1), 0.50 |
| 4 | 6 (ramp, at full length -- the ramp ends) | 6, then 17 on the target | 18 | after the ramp's end: 0.27, 0.82, 1.27, 2.90, 2.22, 0.39 (e1), 0.26 (e1), 0.44 |

- **Arm 2 passes the pre-registered bar, narrowly**: 4 fewer steps, 1 fewer cycle, the same single retry
  (step 20, `alpha`, beta 0.005 -> 0.01), same state. It is bit-for-bit arm 1 through step 22; the release
  fires at 23 and the march ends in two released steps of 12 and 15 cycles (a floor step costs 4-5 here),
  so the saving is in steps, not cycles. No released step crossed `abort_above_cycles` or retried.
- **Arm 3 fails, and the cost is the ramp.** At `grow` 3 the shift reaches 0.006 at step 5, mid-ramp,
  and nine steps between 5 and 16 are redone, twelve attempts in all (five triggered by cycles at an
  unchanged beta, seven by `alpha`): the ramp
  costs 125 accepted / 321 all cycles against arm 1's 62 / 62, and in-step preconditioner refits rise
  from 14 to 25 (rebuild seconds 440 against 315). pitzDaily's `3 x 12` does not transfer to this
  12-station ramp at `turbulence_damping` 5.
- **Arm 4 fails against arm 1 but recovers most of arm 3's loss.** The ramp settles at step 6 (the floor
  reached at full length on the ramp), ends itself after a 3-step finish (9 station changes against 13),
  then the target re-settles at step 17 and releases at 18: 187 / 267 cycles against arm 3's 214 / 459,
  five steps redone (7, 8, 9 in the finish; 15, 16 on the target). The arm-4-vs-arm-3 bar ("within a few
  percent of arm 3") assumed arm 3 passed; it did not, so the question it was meant to answer -- can a
  settle-ended ramp replace a hand-tuned station count here -- is answered by arm 4 against arm 1: no, not
  at `grow` 3.
- **The settle here moves with the ramp** (22 at `grow` 1.5, 17 in both `grow` 3 arms), unlike pitzDaily's
  fixed step 14 -- but both `grow` 3 arms settle at 17 although their ramps ended at 13 and 9.
- **`rho1` gives no lead on the settle here, at either `grow`.** At `grow` 3 (arms 3, 4) it sits above one
  through the floor steps (up to 2.9) and drops below 0.5 only on a step whose shift a retry or the
  control has just raised; at the shipped `grow` 1.5 (arms 1, 2) it is below one from the first floor
  step (0.92, 0.29, 0.36, 0.29). Neither shows pitzDaily's smooth approach to one ahead of the settle.
  What it does track: **below one on the schedule that worked, above one on the two that redid steps** --
  a "the shift outran the state" signal rather than a settle predictor. Necessary, not sufficient: on
  pitzDaily's anchor sweep the cheapest arm ran `rho1` 1.3-4.6 at the floor with no step redone.
- **The positivity limit bounded the step (`L`)** at 8 steps in arms 1 and 2, 5 in arm 3, 4 in arm 4; the
  production cap (`cap%`) was 0 throughout.
- **Not run:** the optional `STATIONS` 8 / 16 at `grow` 3 and `RELEASE_FLOOR=1e-3`. No released step
  retried, so the second has no trigger; arm 3's failure is in the ramp, which a station count at `grow` 3
  might move, but pitzDaily's sweep put the optimum one step after the settle and arm 4 already ends there.
- **What it does not tell apart.** No station change re-damps the shift on either shipped ramp:
  `turbulence_damping` is the closure block's constant shift weight on every step, and at one step per
  station `ViscosityRampHomotopy` forces `redamping` to exactly 1.0. So at `grow` 3 beta reaches 0.006 at
  step 5 with 7 of 12 stations to go, unguarded, and that is where the redone steps sit (5-16); arm 4's
  redone ramp steps (7-9) are its finish, the viscosity jumping at the floor again. pitzDaily tolerated
  viscosity moving at the floor; bfs3d did not. Four candidates are not separated: **beta reaching the
  floor before the ramp arrives**, bfs3d's `turbulence_damping` 5 against pitzDaily's 3, **the inner
  stop rule** (lineax here, `residual` on the case the pairing was tuned on; five of arm 3's twelve extra
  attempts were cycle-triggered), and the harder 3D transient. `grow` 2 moves the floor's arrival (partly
  separating rate from case), `BFS3D_TURB_DAMPING=3` the damping, `BFS3D_FORWARD_STOP=residual` the stop.

**Follow-up arms, same machine and day (#685 review).** At `98aab02` (`probe/ramp-anchor-sweep`), whose
only change on the bfs3d path is the `BFS3D_FORWARD_STOP` override (unset keeps the file's lineax stop;
no library change against `61a14308`), so arm 1 above is still the reference. `refresh_on_cycles` stays at
the file's 3 under both stops. Cycles under the `residual` stop are counted as cycles RUN, one more per
inner solve than lineax for the same work, so set its cycle columns beside the inner solves.

| arm | edit of the file | steps | cycles accepted | cycles all | inner solves | steps redone | final `\|R\|` | wall (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | (shipped, from above) | 28 | 174 | 193 | 82 | 1 | 9.06e-6 | 1191 |
| 5 | `FORWARD_STOP=residual` | 35 | 210 | 294 | 150 | 5 | 6.45e-6 | 1005 |
| 6 | `FORWARD_STOP=residual GROW=3 RELEASE_FLOOR=1e-4` | **25** | 180 | **195** | 110 | 3 | 4.66e-7 | **792** |
| 7 | `GROW=3 RELEASE_FLOOR=1e-4 TURB_DAMPING=3` | 21 | 243 | 347 | 79 | 5 | 8.25e-6 | 1664 |
| 8 | `GROW=2 RELEASE_FLOOR=1e-4` | 21 | 184 | 222 | 71 | 2 | 3.77e-7 | 1217 |
| 9 | `N_POINTS=1` (anchor 10) | 27 | 219 | 239 | 80 | 2 | 9.51e-6 | 1378 |
| 10 | `N_POINTS=3` (anchor 1000) | **92** | 329 | 329 | 213 | 0 | 9.33e-6 | 2054 |

`x_r/h` 8.361 in every arm. Converged states against arm 1: relative L2 0.8-2.1e-4 for arms 5-9 (inside
arm 1's own last-two-step drift, 5.7e-4); **1.6e-3 for arm 10**, which stopped at the bar on a slow linear
tail (below) -- the same `x_r/h`, but not as near the root as the others.

- **The stop rule is most of arm 3's loss.** Under `residual`, pitzDaily's `grow` 3 + release (arm 6)
  redoes 3 steps, not 9, and costs 195 all-attempt cycles against arm 3's 459. **Against its own stop's
  shipped schedule (arm 5) it passes the pre-registered bar outright**: 25 against 35 steps, 180 / 195
  against 210 / 294 cycles, fewer steps redone (3 against 5), same root. Against the file as shipped
  (arm 1, lineax) it is 3 fewer steps and 2 more all-attempt cycles, but about 30 more inner solves (120 against 86, all attempts),
  each counted one cycle higher -- so on the same work it is cheaper, and it is the fastest arm here
  (792 s against 1191, indicative; machine otherwise idle, one run each).
- **The `residual` stop alone on the shipped schedule (arm 5) is not a win in counts** -- 35 steps, five
  redone (16, 18-21, on the target, by cycles at an unchanged beta and then `alpha`) -- although its wall
  clock fell to 1005 s. pitzDaily's halving of the wall time is not reproduced on this schedule, and
  `refresh_on_cycles` was left at 3 where pitzDaily pairs `residual` with 2.
- **Damping 3 (arm 7) recovers part of the loss (459 -> 347) under lineax but still loses to arm 1**, and
  the rate (arm 8, `grow` 2) loses less (222) than `grow` 3 under the same stop. So under lineax the order
  is `grow` 1.5 < 2 < 3 in cost; under `residual`, `grow` 3 beats 1.5. **The stop and the rate interact**,
  and the four-candidate attribution above collapses mostly onto the stop.
- **Where the ramp starts, on bfs3d too: anchor 100 (shipped) is cheapest of 10, 100 and 1000.** Anchor 10
  costs 219 / 239. Anchor 1000 reaches the target at step 14 with no step redone, but its full-length
  trials at steps 12-13 read `rho1` ~1e22 and 4e26 (line search to `alpha` 0.125 / 0.062), and it then
  needs **64 steps at the floor** to meet the bar: from step 28 the residual falls by only ~0.91-0.95 per
  step (release off, as shipped). Not "a deeper start is cheaper" here -- the shipped anchor is the
  optimum of the three, the same place pitzDaily's sweep had its optimum until 300 / 1000 report.
  ⚠️ `compare.py`'s comment that "a direct solve (0) does NOT converge here" was NOT re-checked: arm 9 is
  anchor 10, not a direct solve.
- **What it does not separate yet:** `residual` with `refresh_on_cycles` 2 (pitzDaily's pairing of the
  two), and `residual` with the release alone on `grow` 1.5.

## The release backoff, 2026-10-12

**What was built:** `ShiftStrengthControl.release_backoff` (default 0.5; `step_control.py`). When the rule
backs off from a released step, the control records `bar = release_backoff * residual the release fired
from` in the carried state's `ReleaseMemory`, and releases again only from a settled step whose residual is
at or under the bar. The carried state became `ShiftControlState(beta, memo, release)`.

**The bar set the day before, restated honestly before measuring.** "Anchors 300 and 1000 near anchor 100's
111 cycles" could not be met by any release rule: anchor 300 had already spent 113 cycles before its first
release. The bar that tests this change: (a) no failed release fired again on the next settled step, and a
deep start no longer thrashing; (b) anchor 100 unchanged at 111 (its one release holds, so no bar is ever
set); (c) bfs3d arm 6 of #685 unchanged -- it had no failed release, so it is unchanged by construction and
was not re-run.

**Configuration:** pitzDaily `case.yaml` as shipped (the "Where the ramp starts" configuration), anchor by
`PITZ_RAMP_ANCHOR`, branch `feat/release-backoff` at `8e4639c`; Linux x86_64, 4 cores, jax 0.11.2; one arm at
a time through `run_case.sh`, nothing else running. Logs `run-20261012-013433` (1000), `-023633` (300),
`-025834` (100).

| anchor | before (2026-10-11) | with the backoff |
|---|---|---|
| 100 | 16 steps, 111 cycles, 0 failed, 701 s | **identical step table** (every printed column but `t(s)`), 111 cycles, 657 s |
| 300 | 26 steps, 255 cycles, 3 failed, 1543 s | 26 steps, **210** cycles, 1 failed, 1301 s |
| 1000 | stopped at 38: 502 cycles, 8+ failed, 5018 s, `|R|` 7.1e-4 | **82 steps, 612 cycles, 6 failed, 3702 s**, converged |

All converged arms reach the same root (`Ux` in [-2.693/-2.694, 10.225]). Cycles count accepted attempts
only, as before.

- **(a) met.** Anchor 1000 is bit-identical to the old run through its first failed release (step 14,
  from `|R|` 2.34e-3); afterwards the old run re-released at 16, 18, 20, 22, 24 (28 cycles and ~360 s each)
  while this one ran floor steps at 4-6 cycles. At step 38: 271 cycles / 1867 s against 502 / 5018 s.
  Anchor 300 also skipped two of its three retries; its second release (step 24, from 5.0e-4) clipped to
  `alpha` 0.125 and was counted a failure, but cut the residual enough that the next floor step reached
  5.2e-5 and the third release finished the march.
- **What it does not do:** make a deep anchor competitive. From anchor 1000's state releases failed from
  `|R|` 2.34e-3, 1.14e-3, 5.66e-4, 2.81e-4, 1.38e-4 and 6.39e-5 -- each bar met, each release still
  clipped or collapsing -- and only the seventh, from 3.10e-5, held. The six failed steps are 173 cycles
  of its 612; the rest is the floor's linear tail (0.93-0.97 per step until about 1e-4). Anchor 300's
  release held from 5.2e-5 (after one that clipped from 5.0e-4) and anchor 100's held on its first try
  from **1.53e-3** -- a residual from which anchor 1000's released steps collapsed. So the residual a
  release needs is a property of where the ramp leaves the state, not a number.
- **`rho1` is not a better gate.** On the settled step before each of anchor 1000's six failed releases
  it was 0.97, 0.59, 0.55, 0.51, 0.44, 0.35; before the releases that held, 0.32 (anchor 1000) and 0.87
  (anchor 100).
- **Verdict on the anchor question: unchanged.** 100 remains the cheapest start; the backoff turns a deep
  start from a thrash into a slow but steady march.

## bfs3d takes pitzDaily's march settings, keeping its own `k` positivity (2026-10-13)

**Decision (project owner, 2026-10-13):** bfs3d's `case.yaml` takes pitzDaily's march settings -- the
multiple-correction gradient, `stop: residual`, `grow` 3 with `release_floor` 1e-4 and `release_backoff`
0.5, no `refit_beta_floor`, `turbulence_damping` 3 -- and **keeps its own `k` positivity** (floor 1e-8, the
global step cap, no per-cell projection), its own trailing inverse (mesh-tuned), anchor 100 and 12 stations.
The probe goes from per-column `(3,3,3,3,2,2)` to a **uniform reach of 3**, because the new gradient extends
the p, k and omega columns (audit below).

**Configuration.** `validation/bfs3d_openfoam/compare.py` at `159536cd` plus this change (each candidate
run as its own `case.yaml`, no `BFS3D_*` override, the reassembled solver checked equal to the file's).
Apple M3 Pro, 11 cores, macOS, Python 3.13.4, jax 0.10.2; one run each through `run_case.sh` with no other
case or test tier; desktop apps open, so walls are indicative (the control ran in the afternoon, the rest
in the evening). Cycle counts are not comparable across the lineax and residual stops.

**The probe audit** (`probe_reach_audit.py`, `state-00036` of the 2026-10-12 shipped march), share of each
column's norm beyond reach 2:

| gradient | u | v | w | p | k | omega |
|---|---|---|---|---|---|---|
| corrected Green-Gauss | 1.3e-4 | 1.1e-5 | 4.1e-5 | 1.2e-16 | 3.0e-18 | 1.6e-18 |
| multiple-correction | 8.1e-4 | 2.0e-5 | 5.2e-5 | **1.1e-3** | **6.7e-5** | **3.9e-5** |

The distance-3 share of the leading block rose 4.1e-6 -> 2.9e-3. Reach 4 is unaudited.

| run | steps | cycles accepted / all | steps redone | max `rho1` | solve wall | `x_r/h` | `U_x` err |
|---|---|---|---|---|---|---|---|
| shipped file (control, `corrected`, lineax) | 36 | 166 / 166 | 0 | 1.6 | 1250 s | 8.36 | 0.059 |
| multiple-correction only (uniform reach 3) | 42 | 167 / 172 | 1 (step 30, x2, `alpha`) | 14 | 1114 s | 8.36 | 0.061 |
| full pitzDaily match (positivity floor 0 + per-cell projection) | 20 | 154 / 164 | 1 (step 9, x2) | **4e91** | 660 s | 8.36 | 0.061 |
| **shipped now: match, keeping bfs3d's positivity** | **19** | 126 / 157 | 2 (step 6 x2, step 14) | 64 | **656 s** | 8.36 | 0.061 |
| the committed files, rerun | 19 | identical row for row | | | 660 s | 8.36 | 0.061 |

- All converged states agree within relative L2 2e-4 of the multiple-correction march; the change of
  gradient moves the state 3.7e-3 against the corrected one and the report's `U_x` error 0.059 -> 0.061,
  `U_z` 0.006 -> 0.005, the outermost spanwise slab's `x_r/h` (the side-wall corner) 11.74 -> 12.53.
- **The positivity choice.** Both match variants run in the same time; the full match's full-length
  trials at steps 8-12 read `rho1` 1e7 to 4e91, against 64 at worst with bfs3d's floor and cap, so the
  cap was kept. **Superseded the next day:** the cap was the cause of both retries, and a per-cell clip
  with an omega bound beat both arms; the cap and floor are deleted (next entry).
- **Neither variant met "no new retry"**; the owner accepted the retries for half the wall time. The
  next entry finds their cause and removes them.
- `case_file_parity.py bfs3d`: same pytree, residual at the hybrid start bit-identical (2.140695e+00).

## bfs3d's retried steps: the global k cap, replaced by a per-cell clip with an omega bound (2026-10-14)

**Decision (project owner, 2026-10-14):** the coupled march keeps `k` positive one way only -- each cell's
own `k` correction clipped (`positive_k_projection`), then each cell's `log omega` correction bounded by
`omega_step_ratio` (`omega_step_bound`, default `e`, so `|d log omega| <= 1`, where omega is solved as its
logarithm). The global fraction-to-the-boundary cap (`positive_k_limit` / `positive_block_limit`) and its
`positivity_floor`, and the `positivity_projection` switch between the two, are **deleted**: on both
validation cases the new guard is at least as good, and nothing selected the cap but bfs3d's file.

**Configuration** (every row): main `dfb21bc2` (#712's case file: multiple-correction gradient, uniform
probe reach 3, `grow` 3 with the release, 12 stations), Apple M3 Pro, 11 cores, macOS, jax 0.10.2; one run
each through `run_case.sh`, sharing the machine in turns with another session's marches (never two at
once), so walls are indicative. The guard arms are the library change of this entry with
`BFS3D_K_POSITIVITY_PROJECTION=1 BFS3D_K_POSITIVITY_FLOOR=0 BFS3D_OMEGA_STEP_RATIO=<ratio>` (the switches
were deleted afterwards; the ratio override remains).

**The cause, from replaying the two retried steps** (`validation/bfs3d_openfoam/step_replay.py`, which
reproduces the march's inner rows digit for digit from the capture's checkpoints):
- Step 6 (station 5, beta 0.0062) and step 14 (the target, beta 0.005) each collapse in their second inner
  iteration to alpha 0.004 / 0.001, below `on_alpha` 0.01, so the step is redone at twice the shift.
- **Not** the linear solve (60 restarts instead of 14 change nothing), **not** the station change (the
  previous station's residual fails the same way), **not** the refit floor (0.05 still gives alpha 0.043),
  and **not** an inexact Jacobian (AD agrees with finite differences in every field, error falling with
  the step to round-off).
- The Newton direction is a good one: along it `||G||` falls well below its start at alpha 0.25 (step 6:
  7.22e-3 -> 6.06e-3) and 0.5 (step 14: 2.01e-3 -> 1.40e-3). What stops it is the global cap,
  `0.99 min (k + 1e-8)/(-dk)`, set by one or two cells whose `k` is near zero: step 6, cell 13178
  (x/h 4.5, y/h -0.34, against a side wall in the bubble; cap 0.0020, next cell 0.63); step 14, cells
  12744 / 3144 (x/h 7.2, y/h 0.95, the top-wall / side-wall corners, spanwise mirror pair; caps 0.0010 /
  0.0026, then 0.008, then 1.3). Another session found the same on every one of E8's nine retries, and that
  the binding cell's `dk` does **not** shrink as beta doubles (cell 3142: -0.54, -0.54, -0.70, -1.31 at
  beta 0.01-0.08), so escalation is a poor cure there.
- Disabling the alpha abort (so a cap-limited step is not a "collapse") does nothing: the next solve
  re-derives the same direction and the same cap. Consistent with the earlier A/B recorded in
  `solve/retry.py` that gating the retry on `binding_limit` cost 8 steps.

**Replayed guards** at each step's failing first attempt (march's redo reached `||R||` 9.66e-3 at step 6,
1.517e-3 at step 14):

| guard | step 6 alpha / `||R||` / max `|d log omega|` | step 14 alpha / `||R||` |
|---|---|---|
| per-cell k clip only | 0.25 / 5.38e-3 / **11.6** | 0.5 / 2.27e-3 |
| clip + omega bound 2 | 0.5 / 5.15e-3 / 4.8 | 0.125 / 2.40e-3 |
| clip + omega bound 1 | 0.5 / 4.97e-3 / 2.8 | 0.125 / 2.44e-3 |
| every field scaled per cell by its k room | 0.5 / 4.95e-3 / 1.35 | **0.008** (fails) |

So only the clip variants take both steps without a retry; the clip alone lets one cell's omega move by
1e5 (the 4e91 trials of the 2026-10-13 full match), and scaling all fields per cell breaks the direction
at the target. A single step cannot rank them, so full marches did:

| bfs3d | steps | cycles accepted (all) | redone | worst `rho1` | solve wall | `x_r/h` |
|---|---|---|---|---|---|---|
| global cap + floor 1e-8 (the file) | 19 | 126 (157) | step 6 x2, step 14 | 64 | 616 s | 8.361 |
| per-cell clip, omega unbounded (2026-10-13) | 20 | 154 (164) | step 9 x2 | 4e91 | 660 s | 8.36 |
| **clip + omega bound e** | **17** | **134 (134)** | **none** | **2.54** | **558 s** | 8.361 |
| clip + omega bound e^2 | 17 | 129 (129) | none | 2.54 | 531 s | 8.361 |

The bound-e state matches the cap march's root to relative L2 1.1e-4 (worst field `w`, 5e-4). The two
bounds differ only at steps 6-7; `e` is the default for its margin over the 4e91 failure, at no measured
cost. pitzDaily, which shipped the clip alone, is neutral: 18 steps / 133 cycles / 126 s against 18 / 130 /
132 s with the bound, `x_r/h` 8.0686 both, worst `rho1` 5.68 -> 3.88.

**The shift ramp is not the cause; the viscosity ramp is (2026-10-14).** The question was whether
#712's aggressive shift ramp (`grow` 3 driving `beta` to `beta_min` 0.005 mid-ramp) caused the retries.
It does not, and the measurements below say so for both steering rules. Configuration: the E8 prototype
(`61331d69`, branch `materialized-ramp-e8`, `AQUAFLUX_STEER_BY_STOPPING`; not merged) with #712's
`case.yaml`/`compare.py`, Apple M3 Pro, 11 cores, jax 0.10.2, one run each through `run_case.sh`, in
turns with another session's marches (never two at once) -- walls are indicative. One override per arm
from #712's file. `beta_min` was a local `BFS3D_BETA_MIN` edit of `compare.py`'s `CONTROL` (beside
`BFS3D_GROW`), not on main. Rows above the italic row run the global cap (as shipped then); below it, the
library change of this entry ported onto `61331d69`. Worst `rho1` is the largest first full-length
trial's residual ratio, as in the table above. Wall is the last step's `t(s)`; `x_r/h` 8.361 in
every row.

| E8 arm (one change from #712's file) | steps | cycles (incl. rejected) | redone | worst `rho1` | wall |
|---|---|---|---|---|---|
| `arrived`, as filed | 21 | 159 (194) | 6, 6, 16, 16 | 64 | 721 s |
| `always`, as filed | 25 | 153 (203) | 6, 8, 9, 14, 14, 15, 19, 19, 20 | 1e41 | 789 s |
| `arrived`, `grow` 2 | 23 | 160 (182) | 7, 18 | 1.39 | 739 s |
| `arrived`, `grow` 1.5 | 29 | 193 (199) | 22 | 1e27 | 790 s |
| `arrived`, `beta_min` 0.02 (released to 1e-4 at the target) | 44 | 292 (292) | none | 3e25 | 993 s |
| `arrived`, 15 stations | 23 | 143 (188) | 9, 10, 11, 17, 17 | 2e30 | 719 s |
| `arrived`, 18 stations | 22 | 132 (132) | none | 1.29 | 750 s |
| `always`, 18 stations | 22 | 101 (101) | none | 0.84 | 557 s |
| *per-cell clip + omega bound below* | | | | | |
| `arrived`, bound `e` | **17** | **133 (133)** | **none** | 2.54 | **566 s** |
| `arrived`, bound `e^2` | 28 | 199 (243) | 19, 19, 20, 21, 21, 22, 22 | 129 | 901 s |
| `arrived`, unbounded | 24 | 189 (233) | 9, 9, 19, 19 | 4e91 | 838 s |
| `always`, bound `e^2` | 38 | 349 (407) | 10, 10, 28, 28, 29 | 8.66 | 1474 s |

- **Every one of E8's retries under the cap is the cap**, read from step-limit dumps of the `always`
  march (`BFS3D_DUMP_STEP_LIMIT=0.0101`, deleted with the cap; the march was bit-identical with it on):
  all 11 sub-0.01 caps reproduce to the digit, each set by one or two cells in a spanwise mirror pair
  against a side wall (z/h 0.09 / 3.91), the next cell's room usually >= 0.08. The lower-corner binders
  (y/h -0.56 to -0.74) sit at x/h 4.15, 4.92, 5.34 -- **equal to that step's mid-span `x_r/h`**, so the
  binding cell travels with the reattachment front; later ones are top-wall corners (y/h 0.95-0.99,
  x/h 6.2-9.0). Two shapes: a collapse at inner 0 (steps 14, 15, 19), and a **ratchet** within one step
  (6, 8, 9, 20): inner 0 is cap-bound at alpha 0.5-0.7, which leaves the binding cell at 1 % of its `k`,
  and inner 1 then gets alpha ~0.001.
- **The shift ramp only moves the retries.** `grow` 2 and 1.5 keep one or two; `beta_min` 0.02 removes
  them but each release to 1e-4 collapses alpha and escalates back, and the march thrashes.
- **A gentler viscosity ramp removes them**: 18 stations (not 15) is clean under both steering rules,
  with inner alpha >= 0.25 throughout, and `always` + 18 stations ran the fewest cycles of any arm. It
  costs steps the guard does not, so it was not taken.
- **Under E8 the omega bound must be `e`.** At `e^2` or unbounded, `arrived` keeps a clean ramp but stalls
  after the release to 1e-4 at the target (`G` 2.27e-5 at alpha 1 and rate 0.99, just above `atol` 1e-5)
  and the alpha retry escalates `beta` to 1.28 before it recovers; `always` at `e^2` is the worst arm, with
  inner solves running out their 14 restarts mid-ramp and trial steps 21-92x up. On main `e^2` was clean
  (17 / 129 above), so the margin `e` buys is real, not just over the 4e91 failure.
- **Merged main, as shipped** (`da37958a`: #710's stopping test + this entry's guard, bound `e` by default,
  no overrides, same machine): **18 steps / 137 (137) cycles / none redone / worst `rho1` 2.54 / 552 s**,
  `x_r/h` 8.361, final `|R|` 5.1e-6 in the case's scales. One step more than the guard on `dfb21bc2` (17 /
  134), which is #710 meeting `atol` in the case's own scales.

**Closed the same day:** with nothing installing a whole-step `step_limit`, the generic seam
(`step_limit` on the step classes and `shifted_step`, `backtracking_line_search(max_alpha=)`,
`StepOutcome`/`StepReport.binding_limit`, `newton_march(stop_on_limit_stall=)` / `_limit_collapsing`,
the log's `limit` aside and `L` flag) was deleted as unused, with the harnesses that only read it
(`march_stall_replay.py`, `k_row_scale_probe.py`). The line search's `grow` went with it: the step's `max_alpha = 1.0`
default had capped its rungs at the full step since 2026-08-15, so it too reached nothing.


## The dual-time step's compiled program: five residual copies to three (2026-10-17)

`DualTimeStep` traced the residual five times per `_march_step`: the anchor's `R(φⁿ)`, the Newton
right-hand side `G(p)` at every inner iteration, the line-search ladder body, `R(phi_next)` at the end for
the step's measure, and the Krylov solve's JVP. The second and fourth are redundant in arithmetic (the
right-hand side is the `G` the previous search evaluated at the rung it kept; `R = G − β d (p − φⁿ)`), and
are gone; the contract is in `solve-globalization.md` ("The line search returns its residual VECTOR").

**Configuration.** pitzDaily, `validation/pitzdaily_openfoam/compare.py`, Linux x86_64 4-core container,
jax 0.11.2, Python 3.13; main at `183e9ffa` against `183e9ffa` + this change, each from its own worktree
(`compare.py` puts its OWN repository root first on `sys.path`, so a `PYTHONPATH` pointing at another
checkout is silently ignored — the first copy-4 arm measured main a second time that way, and served as a
noise repeat). Compile figures are the `JAX_LOG_COMPILES=1` lines of a two-step run (`case.yaml` with
`max_steps: 2`, the mesh path absolute) on an empty `AQUAFLUX_COMPILATION_CACHE_DIR`, with
`XLA_FLAGS=--xla_dump_to=… --xla_dump_hlo_module_re=.*march_step.* --xla_dump_hlo_as_text` on in every arm
(the dump adds its own writes to every compile). One run per arm.

| arm | trace `_march_step` (s) | lower to MLIR (s) | XLA compile (s) | HLO lines before / after optimization |
|---|---|---|---|---|
| main | 10.5 | 3.9 | 27.4 | 85,650 / 102,885 |
| main, repeat | 11.1 | 3.9 | 27.6 | 85,650 / 102,885 |
| copy 4 only (`R(phi_next)` from the kept `G`) | 9.6 | 3.7 | 24.4 | 73,438 / 94,910 |
| copies 4 and 2 (final code) | 7.5 | 3.5 | 21.3 | 60,970 / 84,840 |

So about a fifth off XLA's compile (27.5 → 21.3 s) and about 30 % off tracing (10.8 → 7.5 s), the tracing
part paid on every run, warm cache or cold. Lowering moved within noise. The two-step tables are identical
to every printed digit.

**Full march, shipped `case.yaml`, warm cache per arm (each arm's cache filled by its two-step run), back to
back, one run each:** both **21 steps, 50 inner solves, 106 restart cycles**, the step tables identical to
every printed digit but `t(s)` (final step-row `R` 2.618e-06, steered 2.258e-06), `x_r/h` **8.0686**, no
redone step. Solve wall **204 → 195 s**. The run-time saving is one residual evaluation per inner iteration
and one per outer step, 71 here, plus the shorter tracing; one run per arm does not separate the two.

**Not bit-identical in principle, identical here in what is printed.** The step's `R(phi_next)` rounds at
the size of the transient term, and the carried right-hand side was evaluated at `phi + alpha delta` inside
the ladder rather than at the iterate formed after it.


## Moved verbatim from `solve-globalization.md` (2026-10-17 prune) — the rule as it stood

> The whole rule body before it was condensed, kept so measurements, test inventories and the
> reasoning behind each binding entry stay re-adjudicable. The current rule is the authority.


### [from rule] ⚠️ A DEFECT THAT FIRES FOR THE WRONG REASON CAN BE LOAD-BEARING (binding, measured 2026-09-09)

**A globalization defect can be the only thing holding the march out of a bad regime, so fixing it
*correctly* is what breaks the case.** Read this before removing any damping behaviour you have
diagnosed as wrong — the diagnosis being right does not make the removal safe, and the failure lands
one or more steps later, far from the change.

The instance, in full in this file's `redamp` entry and in `.claude/notes/solve-march-log.md`, "`redamp` — the spurious brake `rebase` removed was load-bearing": a
residual-ratio control mis-read a homotopy's station change as divergence and braked. The mis-reading
was real — a station change raises the residual because the *problem* changed, not because the step
was wrong — and removing it was correct. It was also the only thing holding the shift out of a regime
where the frozen preconditioner stops tracking the operator: the shift walked into a wall, the line
search collapsed to `alpha = 0`, `|R|` rose two orders, and the retry ladder had to escalate to
recover. (Not "a shift the preconditioner cannot invert" — the preconditioner's own copy of the shift
is floored separately and never went that low. See the mechanism note in `.claude/rules/solve-march.md`; the measurement is in `.claude/notes/solve-march-log.md`, "`redamp` — the spurious brake `rebase` removed was load-bearing".) **The principled
replacement had to reproduce the accidental brake's own arithmetic before it worked.**

What to take from it, none of which is specific to that mechanism:

- **Before deleting a damping behaviour, ask what it is holding, not only whether it is right.** Those
  are different questions and the second does not answer the first.
  - **This applies to ACCIDENTS as much as to defects — to anything nobody put there deliberately.**
    The brake above was a mis-reading; the preconditioner's own shift floor (`refit_beta_floor`, 0.05 on that
    case) is the same shape from the other direction — a constant chosen against no recorded evidence
    that silently decides where the frozen operator stops following the real one. Neither was designed,
    both were load-bearing, and an unexamined constant is harder to notice than a wrong rule because
    nothing about it looks like a decision.
    ⚠️ An earlier draft named the *march's* `beta_min` here instead. That was a conflation of two
    different floors and it was wrong: `beta_min = 0.005` is fine on that case — the march converges
    there. Left as a correction rather than deleted, because the two floors are easy to mistake for
    one and the mistake changes which knob you reach for.
- **A correct fix that makes the case worse is evidence about the SYSTEM, not about the fix.** The
  right response is to find what the defect was doing and supply it deliberately — not to revert, and
  not to ship the defect on the grounds that it works.
- **Measure the removal on its own before combining it with anything.** The intermediate arm (the fix
  alone, no replacement) is what located the wall; a run that changed both at once would have shown a
  number and no mechanism.

### [from rule] Globalization — Newton step, continuation, line search

- **`RetryPolicy.solver` is a SETTINGS VALUE, `LinearSolverSpec` (`solve/linear_solver_spec.py`, 2026-09-24,
  #437), not a `lineax` solver.** `GmresSolve(rtol, restart, stagnation_iters, max_restarts)` builds
  `relative_residual_gmres(...)` (the Euclidean relative stop the only two callers used,
  `relative_residual_gmres(1e-4, restart=40)`); `DirectSolve()` builds `lx.AutoLinearSolver(well_posed=True)`.
  The reason is the case file: a `lineax` object is not plain data, so a policy holding one could not be
  read by `SettingsMapping`, and `RetryPolicy` is now a case-file kind as it stands. `newton_march` builds
  the solver **once per march** (`tight_solver`) and hands that one object to every divergence retry.
  ⚠️ **Only this field moved.** `RootSolveSettings.linear_solver` / `.adjoint_solver` and
  `solve_coupled(adjoint_solver=)` still take `lineax` solvers; the case layer's `RootSolve` builds them from
  `LinearSolverSpec`s. Moving those too touches ~12 call sites and was left out of scope. The unit tests'
  `"tight"` marker survives as a `LinearSolverSpec` test double (`_TightSolve.build() -> "tight"`), which is
  also what pins the build: a march passing the spec itself, unbuilt, reaches the step as a non-`"tight"`
  object and the retry tests go red.

- **What only SOME strategies or controls can do is a declared protocol in `strategy.py`, asked by
  `isinstance` — never `getattr(x, name, default)` (binding, #281 part 2, 2026-10-10).** All exported:
  - **`NewtonStrategy.with_norm(norm)`** is how the march swaps the per-iteration measure in. It used to
    `eqx.tree_at(lambda s: s.residual_norm, …)` on the step, so a strategy conforming in full but keeping
    its measure under another name failed inside equinox with an error naming a lambda. Implemented on
    `DampedNewtonStep` and on the `ShiftedStep` base (the same `tree_at`, so the swap is still a data-leaf
    change and a compilation-cache hit).
  - **`CarriesRelaxationSchedule`** (`relaxation_schedule`) and **`ReadableShift`** (a schedule's `beta`;
    `ConstantRelaxation` yes, `SwitchedEvolutionRelaxation` no). `ShiftedNewtonStrategy` is now
    `NewtonStrategy + CarriesRelaxationSchedule`. **`shift_of(strategy)`** is the ONE reader of a shift —
    `None` where there is none — used by the march's reports, the escalation's carry, the β-tracking
    refresh and `RetryPolicy.require_shifted`. There is no `march._shift_of`.
    `ShiftStrengthControl.next_step` refuses a step that is not a `CarriesRelaxationSchedule`.
  - **`AbortsInnerLoop`** (`with_inner_abort(*, above_cycles, below_alpha)`, `None` = keep the step's
    own): `RetryPolicy.with_inner_abort` calls it instead of probing two field names and rewriting them
    with `dataclasses.replace` on a type it does not own. `DualTimeStep` implements it.
  - **`StopsForRefresh`** (`refresh_on_cycles: int | None`, a data member): a strategy that can end a
    step early for a preconditioner re-fit, reporting `StepOutcome.refresh_due` (the ninth field, added
    with it, 2026-10-10, #290). `newton_march` asks it before the first step and refuses one with
    `refresh_on_cycles` set and no `refresh_preconditioner`. `DualTimeStep` implements it; see
    `solve-march.md` for the march side.
  - **`ShiftCarryingControl(StepControl)`** declares `carry_beta`, `resumed_at`, `redamp` and `rebase`
    **together**, and the march and `staged_march` call any of them only on a control that is one.
    ⚠️ **All four or none, by decision (project owner, 2026-10-10):** they are one capability — keeping
    the carried β right — so a control offering some would be right across one boundary and silently
    wrong across another. A control implementing only some now gets **none** of them (pinned by
    `test_the_shift_hooks_reach_a_control_offering_all_four_and_none_offering_some`). Only
    `ShiftStrengthControl` implements any, so no shipped control changed behaviour.
  - Why: a probe answers "no" in silence when a class offers the capability under another name, and a
    march that skips a feature looks exactly like one that never needed it. Membership is pinned both
    ways against the real classes in `tests/unit/test_strategy_capabilities.py`; every new test was
    mutation-checked. The remaining `getattr` sites outside this family are tracked in #669.

- **`ShiftedStep` is the shared body of the two shifted Newton steps (`solve/continuation.py`, BUILT
  2026-08-15).** `PseudoTransientStep` and `DualTimeStep` differ entirely in `stepper()` — what one
  *outer step* means — and not at all in how they are configured or interrogated. The fields they
  share (`shift_policy`, `relaxation_schedule`, `line_search`, `step_projection`, `krylov_solver`,
  `residual_norm`, `adjoint_preconditioner_factory`, and `jacobian_residual` since; a ninth, `step_limit`,
  a whole-step cap, was deleted 2026-10-14 as unused) and all three `NewtonStrategy`
  accessors (`norm`, `linear_solver`, `adjoint_preconditioner`) were written out **twice, identically**,
  field comments included. A subclass now supplies `stepper` plus the fields its own step shape needs.
  `DampedNewtonStep` is deliberately NOT a subclass: it has no shift, and its `linear_solver` returns a
  different constant, so folding it in would mean inventing a `relaxation_schedule` it does not possess.
  ⚠️ **`adjoint_preconditioner_factory` is a DATA field since 2026-10-12 (#290); it was static.** A
  traced field split rides in it as array leaves (`_TracedSplitAdjoint`), so a re-fit can replace it
  without a recompile; so does the block-diagonal family's `solve.UnshiftedPreconditioner(policy)` (the
  policy's preconditioner at zero shift, a module where it was a lambda; #386), so a re-pointed session
  can replace it. A value holding no arrays — a closure, a frozen dataclass such as
  `FrozenTransposeFactory`, `None` — is still filtered to the static side and keyed as before.
  **Verified as a pure refactor**: no field and no default changed on either class, compared field by
  field against the previous implementation (`DualTimeStep.line_search` stays **10** against the base's
  **0** — it redeclares it, which is why that difference has to be checked rather than assumed).

- **⚠️ THE RESIDUAL AND THE JACOBIAN MAY BE DIFFERENT FUNCTIONS — `ShiftedStep.jacobian_residual`
  (BUILT 2026-08-22).** An optional `phi -> R~(phi)` whose Jacobian--vector product forms the shifted
  solve's operator, in place of differentiating the residual the step is driving to zero. `None`
  (default) differentiates the residual itself and is byte-identical. Threaded through
  `_shifted_solve(jacobian_fn=…)`, read by **both** steppers (a march runs the dual-time one, so wiring
  the single-step branch alone would leave it inert on the path that matters).

  **The asymmetry it exists to express: `R` decides the answer, `J` decides only the rate.** The
  residual defines *which discrete equations are being solved*, so an approximation there moves the
  root; the Jacobian only decides how fast an inexact-Newton iteration reaches whichever root `R`
  defines, which is the same latitude the forward regime's `rtol = 0.3` (`LinearSolveSettings`) already takes
  on the linear solve. Holding
  both to one accuracy because one function serves both is a coincidence of implementation, not a
  requirement. The concrete consumer is the gradient reconstruction's sweep count — see
  `.claude/rules/schemes.md` (measurements in `.claude/notes/schemes-log.md`) for what it is worth and where the knee is.

  - **The adjoint is exact BY CONSTRUCTION, not by care, and this is the load-bearing property.**
    `root_adjoint`'s backward rule forms `jax.vjp(residual_fn, …)` at the converged state directly and
    never consults the Newton step, so nothing set here can reach a gradient. That is what makes a cheaper
    forward operator legitimate at all: the sensitivity a user asks for stays exact however loosely the
    march got to the root. Pinned by
    `test_a_stand_in_jacobian_leaves_the_adjoint_exact`.
  - **The descent test still reads the TRUE residual.** `PseudoTransientStep`'s `directional` is
    `jvp(norm o residual_fn)`, so the merit and its slope are the real ones; only the operator that
    produced the direction is approximate. Correct — the question a descent test asks is whether the
    step helps the residual being solved.
  - **Data, not static.** The natural argument is a bound method of an assembler, which equinox carries
    as a pytree, so its arrays ride as dynamic leaves and every step stays a compilation-cache hit
    rather than recompiling against baked-in constants. Same reasoning as `residual_norm`.
  - **A reachability test is mandatory and is easy to omit.** An unwired field leaves the correction
    bit-identical, and every property asserted of it then holds for a reason unrelated to the field.
    `test_a_stand_in_jacobian_genuinely_changes_the_step_it_produces` scales the stand-in's Jacobian by
    a fixed factor, whose unshifted exact solve must scale the step by its reciprocal.
  - **⚠️ A FACTOR-OF-TWO-WRONG JACOBIAN IS EXACTLY NEWTON'S STABILITY BOUNDARY — do not reach for it as
    a harmless test fixture (learnt the hard way, 2026-08-22).** An operator scaled by `s` makes the
    step `1/s` times the Newton step, so the iteration's fixed-point derivative at the root is
    `1 - 1/s`, which at `s = 0.5` is `-1`: it oscillates forever without decaying, once the
    pseudo-transient shift has eased to zero near the root. `0.5` is therefore the *least* harmless
    factor available rather than an obvious one, and two tests written with it ran to `max_steps` and
    failed. The fixture is at `0.8` (derivative `-0.25`) — still unmistakably the wrong operator, and
    one that converges. Note this is also the *quantitative* form of the exchange rate the field
    trades in: an operator error of order one is not a slower Newton, it is no Newton at all, so a
    cap's error budget is bounded well below that and not merely "small".

- **`ShiftedStep.preconditioned_as(other)` (2026-10-12, #386)** swaps in `other`'s `shift_policy`,
  `adjoint_preconditioner_factory` and `jacobian_residual` — everything built from the problem — and keeps
  every setting of the step itself, the shift strength included. It is how `RebuildingSession` hands a
  running march a step rebuilt on another problem without discarding the shift its control set. `tree_at`
  with `is_leaf=None` so an unset field swaps too.

- **There is no `_TrailingFirstFieldSplit` and no `_set_order` (deleted 2026-09-13, #371).** The field
  split has one ordering, so `FieldSplitInverse.apply` has one body that branches only on
  `transpose` (which reverses the solve order and uses `Cᵀ`).

- **Forward globalization is ONE injected strategy — `strategy: NewtonStrategy`.** The forward
  Newton loop has a single point of variation: `RootSolver` takes one `strategy`
  implementing the `NewtonStrategy` protocol (`stepper()` → the per-step
  `(residual_fn, phi, ‖R₀‖, solver) -> phi_next`; `linear_solver()` → the inexact-Newton forward
  GMRES for that march; `adjoint_preconditioner()` → the converged-state transpose preconditioner).
  Two concrete strategies: **`DampedNewtonStep`** (default — the backtracking line search, holding
  the forward/adjoint preconditioner and the line-search count) and **`PseudoTransientStep`**
  (`aquaflux/solve/`, the residual-agnostic diagonally-shifted march; the flow configures it via
  `aquaflux/flow/`'s `momentum_continuation` factory — no wrapper class). The march calls the
  injected step unconditionally — there is **no "was a strategy given?" branch**, and the solver grew
  **no separate `line_search` / `preconditioner` / `continuation` constructor arguments** of its own
  (they were unified into the strategy; do not reintroduce them). Each strategy's shift vanishes at the fixed point, so the converged state and
  the IFT adjoint are strategy-independent. When adding a globalization (e.g. a monotone/forcing
  acceptance), add a `NewtonStrategy` — do **not** grow a branch in the march.
  - **`ShiftedNewtonStrategy` is the SECOND contract, and the eager march's beta machinery requires it
    (binding, 2026-08-15).** `NewtonStrategy` says what every strategy must *do*; `ShiftedNewtonStrategy`
    says what one must additionally *carry* — a `relaxation_schedule` holding `beta` as a readable,
    replaceable **dynamic** leaf. It is separate rather than folded in because not every strategy has
    a shift: `DampedNewtonStep` globalizes by backtracking alone, and demanding a `relaxation_schedule`
    of it would be inventing a quantity it does not possess.
    **What it replaces, and why it matters more than it looks:** the requirement was enforced by
    `hasattr` probes scattered through `newton_march`, which fail **silently**. A `DampedNewtonStep`
    satisfies `NewtonStrategy` in full, so a march configured with `RetryPolicy.on_cycles` accepted it and then
    never escalated — from the log, indistinguishable from a march that never needed to. One reporting
    path failed the *opposite* way and read `active_step.relaxation_schedule` unguarded, so the same
    conforming step raised `AttributeError` mid-march. `newton_march` now checks **once, on the first
    iteration, against the step the CONTROL produced** (`RetryPolicy.require_shifted`), naming the
    feature and what it needs.
    **⚠️ THE OBJECT CHECKED IS `active_step`, NOT THE STEP HANDED IN, AND THAT IS LOAD-BEARING (fixed
    2026-08-16).** Checking before the loop looks equivalent and is not: the default schedule is
    `SwitchedEvolutionRelaxation`, which is **memoryless and exposes no `beta`**, while a
    `ShiftStrengthControl` swaps a `ConstantRelaxation` (which does) onto the step every iteration. So
    the readable beta the escalation needs is supplied **by the control**, and a pre-loop gate sees a
    step that never runs. It rejected the shipped configuration outright — every `bfs3d` march runs a
    shift control *and* both retry thresholds, so **not one could start**, under any flow inverse, with
    a `TypeError` whose own message names `DualTimeStep` as acceptable while refusing one. The check
    still precedes any escalation, which is the property it exists for, and with no control
    `active_step` is the base step so the ungated path is unchanged. Pinned in **both** directions by
    `test_the_escalation_guard_accepts_a_shift_a_step_control_installs` and
    `test_the_escalation_guard_still_rejects_a_step_that_can_never_escalate` — the pair matters,
    because a gate can be "fixed" by deleting it and only the second test notices.
    ⚠️ **This was found by RUNNING a march, not by the suite, and it was the second such break in one
    sitting** (the other: `compare.py` handed `solve_coupled` a bare `refresh_preconditioner` callable where
    a `RefreshPolicy` was expected, so reading the policy raised before step 1). **No test tier drives
    this case's own driver**, so a refactor can tighten a seam, take the case out entirely, and leave
    every gate green. Treat "the fast gate passes" as saying nothing about whether `bfs3d` can march.
    - **Gate only what is genuinely silent (binding).** The escalation is gated. The divergence retry
      (`retry_solver` / `retry_divergence_cap`) is **not** — it re-solves at a tighter tolerance and
      never touches beta, so it works on any step; its *reporting* goes through `shift_of`, which
      returns `None` for a step with no shift rather than inventing one. A `StepControl` is **not**
      gated either: the protocol only asks it to return a ready-to-run step, a step-agnostic one is
      legitimate (and exercised), so the protocol stays typed on `NewtonStrategy`. A control that
      *does* drive beta refuses for itself: `ShiftStrengthControl.next_step` is typed on
      `ShiftedNewtonStrategy` and raises a `TypeError` naming the control and the step when the step
      has no `relaxation_schedule` (#158, 2026-10-10). It used to "fail loudly" only as a bare
      `AttributeError` from inside its `tree_at`, naming neither. The march calls it before its first
      step, so nothing is taken first (pinned by `test_step_control.py`, all three controls,
      mutation-checked). It tests the field's presence, not a readable beta, because it replaces the
      schedule rather than reading it. Gating at the march would reject what the protocol permits.
    - **The runtime check tests the SHIFT, not `isinstance(..., ShiftedNewtonStrategy)`.** The argument is
      already typed `NewtonStrategy`, so re-testing those four methods at runtime would reject a
      legitimate duck-typed step for a reason unrelated to the feature asked for — which it did, on
      every test double, when first written that way.
- **`continuation.py` — BUILT (`PseudoTransientStep`, residual-agnostic).** The pseudo-transient
  continuation engine lives **here in `solve/`, not in `flow/`** — it is a `NewtonStrategy`
  (`stepper`/`linear_solver`/`adjoint_preconditioner`) that runs an **injected**
  `RelaxationSchedule` for the shift strength β, the diagonally-shifted solve `(J + diag(βd))δ = −R`
  (`solve_linear(throw=False)`), and the closed-loop accept/escalate `while_loop`. **`stepper()`
  returns a `StepOutcome`** carrying the accepted attempt's cycle count, its line-search factor α (the
  step-quality signal, ≤1; a march reads both off the report), and
  **its `residual_norm`** — the measure at the accepted candidate, which the escalation carry now
  transports beside the candidate it belongs to (a fully-rejected step returns `phi` untouched, so it
  reports `phi`'s own measure, the `residual_norm` the caller handed in). The attempt already formed it
  for the acceptance test, so both drivers read it instead of re-evaluating the residual at the same
  iterate.
  - **The β schedule is an injected `RelaxationSchedule` (`solve/relaxation.py`), SER extracted as the
    default (binding — do not re-inline the β rule).** ⚠️ **Its method is `shift_strength(‖R‖, ‖R₀‖)`,
    renamed from `relaxation()` in #373:** it returns β, which damps *more* as it grows, where an
    under-relaxation factor in `(0, 1]` damps *less* — and both words are live in this package (the
    segregated loop under-relaxes k and ω by a factor). The classes keep their names,
    `SwitchedEvolutionRelaxation` being the literature's. The old `beta0`/`exponent`/`beta_floor` fields on
    `PseudoTransientStep` are gone; the field is `relaxation_schedule: RelaxationSchedule`, defaulting to
    `SwitchedEvolutionRelaxation(beta0=2, exponent=1, beta_floor=0)` — byte-identical to the old inline
    `max(beta_floor, β₀(‖R‖/‖R₀‖)^p)`. It is the direct twin of the injected `ResidualNorm`
    (`solve/norm.py`). A `RelaxationSchedule` is **memoryless** (β from the two residual norms only), which
    is what keeps it on the differentiable traced path. `ConstantRelaxation(β)` carries β as a **dynamic
    0-d leaf** so an external control can vary it per step as a `filter_jit` cache hit (the `lam_max`
    precedent). `beta0`/`exponent`/`beta_floor` are the public spelling still, but they are now fields of
    the shared `Globalization` (below) rather than keywords on each builder, and
    `Globalization`'s step construction translates them — only when one is set; unset, the step's own
    default schedule applies — a factory building the real object, not a shim. A *stateful* or α-driven damping rule is **not** a schedule; it
    is a `StepControl` on the eager march (see `march.py`).

  - **⚠️ ONE `Globalization` FOR ALL SIX BUILDERS — the engine's own settings are not per-builder
    keywords (binding, BUILT 2026-09-13, #372).** `Globalization` (`solve/continuation.py`, exported)
    carries every setting that describes *how a march damps* rather than *what it is solving*: the SER
    schedule (`beta0`, `exponent`, `beta_floor`), the escalation ladder (`max_escalations`,
    `escalation_factor`), the divergence guard's `divergence_cap`, and the backtracking ladder
    (`line_search`, `line_search_growth`). Its two methods, `step(policy, **fields)` and
    `dual_time_step(policy, **fields)`, are the **only** places a globalization becomes a step; the
    per-problem fields (`adjoint_preconditioner_factory`, `krylov_solver`, `residual_norm`,
    `step_projection`, `jacobian_residual`) ride through as passthroughs.
    - **Every field is UNSET by default (`None`), and unset falls through — first to the builder's base,
      then to the step class's own default (binding).** `_coupled_step` calls
      `globalization.with_defaults(line_search=_COUPLED_LINE_SEARCH)`, so `Globalization(beta0=1.5)`
      changes one setting on every builder while the coupled march keeps its ten rungs, and an explicit
      `line_search=0` survives. No default is declared on `Globalization` itself, so none is restated.
      ⚠️ **The first version carried a full default set and two presets** (`DEFAULT_GLOBALIZATION` and a
      `COUPLED_GLOBALIZATION` differing in `line_search`) with a prose rule to start from the right one. A
      bare `Globalization(beta0=1.5)` on a coupled builder then silently reset the line search to zero, and
      a declarative case file's `globalization: {beta0: 1.5}` had no way to say which preset it meant.
      There is no `COUPLED_GLOBALIZATION` now; `DEFAULT_GLOBALIZATION` is the empty override.
    - **What it refuses rather than drops.** `dual_time_step` raises on any escalation-ladder field that is
      set (`max_escalations`, `escalation_factor`, `divergence_cap`, `line_search_growth`): the
      dual-time inner loop has no ladder, so they would reach nothing. A step field the target class does
      not declare raises **even when passed as `None`** — `_supplied` checks names before dropping unset
      values, because filtering first made a misplaced `acceptance=None` vanish — and a setting given both
      on the object and as a step field raises rather than one silently winning.
    - **The membership test is whether a setting's reason can be stated without naming a residual or a
      preconditioner.** It can for all of the above. It cannot for the closure's step projection (per-cell k clip +
      ω step bound), whose value is a function of the state layout — so its one setting,
      `omega_step_ratio`, stays on the coupled builders, which hold the `CoupledRANS` that can construct
      it (`_closure_step_projection`), and arrives at `step()` as an ordinary `step_projection` field.
      (`positivity_floor`/`positivity_projection` no longer exist — deleted 2026-10-14.) Nor can it for the per-family Krylov restart regime or the progress measure's default.
      **Decided (#387, 2026-09-14):** `shift_basis`, the velocity shift parts and `turbulence_damping`
      pass the test but describe how the *shift* is formed rather than how the march damps, so they are
      their own value, `turbulence.CoupledShiftSettings`, on the coupled builders — kept off this object also
      because the damping strategies and the live velocity parts carry state. **Still open:** `divergence_cap` and `line_search_growth.basin`
      are residual ratios read in a measure the object excludes (#370); and `acceptance` and the schedule
      are flattened to scalars, so a non-`DivergenceGuard` rule or a non-SER schedule reaches a step only
      through `step(**fields)` with those fields unset.
    - **What it replaced, and the reach it still does not give.** The four coupled builders carried eight
      of these keywords apiece; `momentum_continuation` and `scalar_pseudo_transient_solve` carried **two**,
      so `beta_floor`, `line_search`, `grow` and the growth rule were unreachable from the flow-only and
      scalar paths, and `line_search_growth` from all six. (⚠️ **There is no `grow` any more** -- the
      ladder's rungs above the full step, on `Globalization`, `PseudoTransientStep` and
      `backtracking_line_search` -- deleted 2026-10-14: a `max_alpha = 1.0` default the step passed
      whenever no whole-step limit was set had capped every rung at the full step since 2026-08-15
      (`7c1361bd`), so it had reached nothing for two months. The ladder starts at the full step.) The problem-specific step fields
      (`step_projection`, `krylov_solver`, `residual_norm`, `jacobian_residual`) are **still** unreachable
      from those two builders — outside #372's settings, not closed by it. The segregated flow solve
      `bulk_velocity_flow_solve` is a `DampedNewtonStep`, a different engine that takes no `Globalization`.
    - **⚠️ Moving keywords off a builder that keeps `**kwargs` turns a retired name into a silent one.**
      The deleted `coupled_continuation` forwarded leftovers to `BlockPreconditioner.build`, which only a
      first build reads — a refresh (`reuse=`) carries the flow block — so `beta0=` on a refresh was
      silently dropped. Closed structurally since #371: flow-block settings travel only as a
      `BlockDiagonal` spec, whose fields are pinned to `build`'s keywords, and march keywords are bound
      against `coupled_step`'s signature (`_march_keywords`), so an unknown name is a `TypeError` at
      construction. There is no `_refuse_unknown_flow_block_options` any more; nothing could reach it.
    - Pinned by `tests/unit/test_globalization_reach.py`: every builder takes the object; a non-default one
      arrives on the built step **field for field**; the shipped defaults as **literal numbers**; one
      override leaves every other setting alone; and each refusal above.
  - **Three boundary responses on `ShiftStrengthControl`, and what each keeps is the whole design
    (`carry_beta` long-standing; `rebase` and `redamp` BUILT 2026-09-09).** The carried state is
    `ShiftControlState(beta, memo, release)` (a named tuple since 2026-10-12; `(beta, memo)` before). All
    three keep `release`, the release's memory of a failed release (below); only `resumed_at` starts a
    state without one. Each of these is an *outside event changed the situation, keep what still applies*
    seam, and they differ only in what still applies:
    | method | called when | keeps | drops |
    |---|---|---|---|
    | `carry_beta` | a β-escalation retry | the memo | — (β replaced) |
    | `rebase` | a `ResidualHomotopy` station change, one step later | β | the ratio reference |
    | `redamp` | that station change, on the entering step | the memo | — (β scaled) |
    `carry_beta` keeps the memo because an escalation changes only how far the march steps through the
    **same** residual; `rebase` drops it because the next residual is a **different function** and a
    ratio across that boundary divides two incomparable numbers.
    - **⚠️ `rebase`'s contract is "drop the ratio REFERENCE, keep everything else that still applies",
      NOT "clear the memo" — a subclass whose memo carries more than the reference MUST override it.**
      The base clears the whole memo, correct only because the base's memo *is* the reference. A richer
      memo (a live growth rate, a cap) would lose those, and the failure is silent in the worst way: the
      ratio then defaults to `1.0` on every adaptation, so the brake `rebase` exists to stop mis-firing
      stops firing *at all*, which is indistinguishable from the rebase working. For the same reason,
      **call these rather than unpacking `(beta, previous_residual) = state`** at a call site — that
      keeps typechecking after a memo grows a field and quietly stops supplying a reference.
    - **`redamp` is not optional and is not a knob.** Without it the shift descends by
      `grow ** steps_per_station` per station, unopposed, and walks through the level the next station
      can carry. The failure, the four-arm measurement, and why a learned floor under β or a cap on the
      growth rate are both *refuted* alternatives are in `.claude/rules/solve-march.md` under
      `ResidualHomotopy` / `redamp` (the four-arm table in `.claude/notes/solve-march-log.md`, "`ResidualHomotopy` — what it is for, and the four-arm result") — read there rather than restating here.
  - **`release_floor` -- the floor comes off once the march has settled at the target (BUILT 2026-10-10,
    opt-in, `None` by default).** A dual-time step at a fixed shift contracts each error mode by
    `beta / (lambda + beta)`, so a march parked at `beta_min` near its root converges linearly at a rate
    the FLOOR sets: on pitzDaily the target station's 15 steps all ran at 0.005 against a slowest mode at
    0.5-0.8 of it. A field pair on the `ShiftStrengthControl` base (`release_floor`, `settle_alpha`
    default 1.0), so all three controls reach it and a case file writes it on any `step_control`. The rule
    lives in one method, `_floored`, applied after every `_adapt`: **released** iff the previous step
    `arrived`, the rule did not raise beta, and either it was `settled(report)` -- ran at or below
    `beta_min` with `alpha >= settle_alpha` -- (then beta = `release_floor`) or it already ran below
    `beta_min` (then the rule's own beta).
    - **`ShiftStrengthControl.settled(report)` is THE definition of settled (2026-10-11), read by the
      release AND by a ramp that ends on it** (`RampSchedule(end_when=control.settled)`, see
      `solve.md`). ⚠️ There is no `release_alpha`: it was renamed `settle_alpha` the same day, before
      the release ever merged, because it now defines settled for both readers. `settled` deliberately
      ignores `arrived` -- that is the release's own further condition, not part of being settled. Anything else -- a clipped released step, a rising residual for a rule that reads one
    -- puts `beta_min` back in one step, not by doubling up from the released shift. `_clamp`'s lower bound
    is `release_floor` when set, so the rules may go below `beta_min` and `_floored` is the one place that
    decides whether they may. "Released" is read off the previous report's shift; what the state carries
    for the release is only the backoff's memory (below).
    - **⚠️ The gate is the measured part, not a guess.** A zero-shift step from the state the ramp ARRIVED
      at raised the residual (line search to `alpha` 0, 42-cycle steps, the same under an inverse fitted
      at zero shift); two states later, after the first step at the floor with `alpha` 1, the same step
      met the stopping bar. So `arrived` alone is the wrong signal, and the shift a state can take
      belongs to the state -- the `(state, beta)` rule in `solve-march.md`, one more time.
    - **`release_floor` must be positive**, refused at construction: a retry escalates beta by scaling it
      (`RetryPolicy.escalate`), and zero scales to zero.
    - **Measured:** pitzDaily 31 -> 22 steps, 161 -> 141 cycles, target station 15 -> 6 steps and 67 -> 30
      inner solves, same `x_r/h`, final `|R|` 1.7e-6 against 8.3e-6. Cycles fall less than steps because
      each released step's solves cost 2-3x a floor step's: the inverse is still re-fitted at
      `refit_beta_floor` (0.05). Re-fitting at the released shift instead was measured 2026-10-13 on the
      post-#290 march and is worth one cycle there (`.claude/notes/solve-open-directions.md` 11a); the floor
      itself was worth 10, and pitzDaily's file now sets none. Full tables and configuration in `solve-globalization-log.md`.
    - **pitzDaily's `case.yaml` ships it since 2026-10-11**, at 1e-4 and paired with `grow` 3 and 12
      stations (111 cycles / 16 steps against the 161 / 31 shipped before; the `grow` x `stations` sweep
      in the log), with `release_backoff: 0.5` stated beside it since 2026-10-12 (the class default,
      written out so the file says what the shipped release does after a failure). **Not the class
      default** (`release_floor=None`) and **not bfs3d's file**.
      ⚠️ Those counts were taken while a cost re-fit ran MID-STEP at `refresh_on_cycles` 2. Since #290 a
      re-fit ends its step, which at 2 moved the release to step 34 of 39 (the settled test needs a
      full-length step, which one-iteration steps rarely give); pitzDaily now ships 3, where it fires at
      step 18 of 20 -- `.claude/notes/solve-open-directions.md` 6b.
    - **bfs3d, measured 2026-10-11 (#678, `61a14308`, M3 Pro, jax 0.10.2):** the release alone at 1e-4 on
      the shipped `grow` 1.5 x 12 stations passes -- 28 -> 24 steps, 174 -> 173 cycles (193 -> 192
      counting rejected attempts), the same one retry, the same root (rel L2 1.8e-4, inside the shipped
      arm's own last-two-step drift of 5.7e-4). The march is bit-identical up to the release, which fires
      the step after the first full-length step at the floor; the two released steps cost 12 and 15
      cycles against 4-5 at the floor, so the saving is steps, not cycles. **pitzDaily's pairing with
      `grow` 3 transfers only with pitzDaily's inner stop**: under bfs3d's lineax stop it costs 214
      accepted / 459 all cycles with nine steps redone; under `stop: residual` it is 25 steps / 195 all
      cycles with three redone, beating the shipped schedule under that stop (35 / 294) and fastest of
      every arm (792 s against 1191, indicative).
      **bfs3d's file releases since 2026-10-13** (owner's decision, with `grow` 3, the residual stop, no
      refit floor and the multiple-correction gradient: 19 steps / 656 s against 36 / 1250 s, same
      `x_r/h`); a library default still waits on the three slow tests #645 names (a loose terminal step is what they measure). Tables in
      `solve-globalization-log.md` ("bfs3d: the release, `grow` 3 and the settle-ended ramp").
    - **A failed release is not retried until the residual falls below a bar (`release_backoff`, BUILT
      2026-10-12, default 0.5).** Without it `_floored` kept no memory: a released step collapsed, the rule
      backed off to `beta_min`, the next full-length floor step was `settled` again, and the release fired
      again -- every other step from a deep ramp start (pitzDaily anchor 300: steps 19, 21, 23, 25; anchor
      1000: eight failures in a row, stopped at step 38 and 502 cycles). Now the rule backing off from a
      released step records `bar = release_backoff * residual the release fired from` in the state's
      `ReleaseMemory`, and the floor is released again only from a settled step at or under it. A release
      that holds sets no bar, so a march whose first release holds is **bit-identical** (pitzDaily anchor
      100: the same 16-step table, 111 cycles, to every printed digit but `t(s)`). Measured on pitzDaily
      (shipped march, `PITZ_RAMP_ANCHOR`, Linux x86_64 4 cores, jax 0.11.2, 2026-10-12): anchor 300 255 ->
      **210** cycles (failed releases 3 -> 1, 1543 -> 1301 s); anchor 1000 converges in 82 steps / **612**
      cycles / 3702 s to the same root, where before it was stopped at 38 / 502 / 5018 s still thrashing.
      ⚠️ **It does NOT make a deep start competitive, and no release rule could:** from anchor 1000's state
      SIX releases failed, from `|R|` 2.3e-3 down to 6.4e-5 (173 cycles together); only the seventh, from
      3.1e-5, held -- while anchor 100's single release holds from 1.5e-3. The residual a release needs
      belongs to the state the ramp leaves, and a deep anchor's cost is the floor's linear tail from it.
      ⚠️ The bar compares residuals across steps in the row-scaled measure, which is rebuilt every step --
      an approximate comparison, adequate for a factor-2 bar and not for a fine one. ⚠️ `rho1` on the
      settled step does not predict which release will hold (0.35-0.97 before anchor 1000's six failures;
      0.32 and 0.87 before the releases that held at anchors 1000 and 100), so it is not a better gate.
      Log: `solve-globalization-log.md` ("The release backoff").
    - Pinned by `test_step_control.py` (each condition; every one of twelve single-line mutations of the
      rule turns a test red; the backoff by its own six tests, nine of ten mutations red -- the tenth,
      keeping a stale `started_from` on re-release, is unreachable because a failure always clears it)
      and `test_newton_march.py::test_a_released_shift_turns_a_linear_tail_into_newton_steps`.
  - **The shift's SPATIAL distribution is an injected `ShiftBasis` (`solve/shift_basis.py`) — the
    spatial twin of the `RelaxationSchedule` (binding).** `RelaxationSchedule` sets *how much* damping
    (the scalar β); `ShiftBasis` sets the per-cell base diagonal `d` the shift `β d` is built on, from
    the operator's two diagonal buckets: `local_diagonal(convective, dissipative) -> d`. The one concrete
    is `LocalCourantBasis(dissipative_weight=w)`: `d = convective + w·dissipative`. **`w = 1` (default) is
    the full operator diagonal `a_P`** — and because `d = a_P` (the same diagonal the operator carries),
    `β a_P` is spatially-*uniform* under-relaxation (relaxation `1/(1+β)` in every cell), byte-compatible
    with the historical shift. **That equivalence is exact and worth stating plainly: on the momentum
    rows, this shift IS the implicit under-relaxation a segregated pressure-correction solver applies,
    at `α_u = 1/(1+β)`.** Both sides match, not just the diagonal — the shift acts on `δ = φ − φ_k` and
    so contributes `β a_P φ_k` to the right-hand side, which is exactly the `((1−α_u)/α_u) a_P φ_old`
    that implicit under-relaxation adds. At the pitzDaily march's operating point `β ≈ 1.9`, that is
    `α_u ≈ 0.345`, against the 0.3 that established segregated codes use as their default momentum
    relaxation. **Consequence (load-bearing for the cold-start work): the momentum treatment is the
    industry-standard one in different notation, so it is NOT where a cold-start reachability gap can
    be hiding, and "our globalization is exotic / uncovered by theory" is false.** Look instead at what
    differs — the coupled pressure/continuity treatment, the turbulence relaxation and limiters, and the
    fact that the reference which reliably reaches this root is a *transient* run. **`w = 0` is a genuine local convective time step** (`d = Σ_f max(mdot_f,0)`
    = ½Σ|mdot|), the non-uniform per-cell `Δt` a Courant condition implies — OF's `Co = ½Δt Σ|φ|/V` and
    Fluent's *segregated* local pseudo-time step are this same convective basis. The buckets are supplied
    by each block: `rhie_chow.momentum_diagonal_parts` (velocity) and
    `preconditioner.scalar_transport_shift_diagonal_parts` (k/ω), each summing to the block's total shift
    diagonal to rounding. **Consequence for the preconditioner (binding):** with a non-`a_P` basis the
    shifted diagonal is `a_P + β d`, *not* `a_P(1+β)`, so `make_preconditioner` must invert `a_P + β d` —
    the velocity block's `apply_at` is fed exactly that (was `a_P(1+β)`; identical when `w=1`). Threaded
    through `momentum_continuation`/`coupled_continuation`/`solve_coupled(shift=CoupledShiftSettings(basis=…))`
    (the coupled builders' loose `shift_basis=` keyword is gone, #387) and the k/ω
    shift policies; **pressure keeps zero shift regardless** (elliptic), which is why a *local* basis is
    defensible on this coupled solver where Fluent uses a global time scale for its coupled path.
    - **The velocity buckets' SOURCE is injected (`VelocityShiftParts`), because the shift and the
      preconditioner have different lifetimes (binding).** `CoupledShiftPolicy` used to take them from
      the frozen flow preconditioner, which welded two unrelated lifetimes together: the preconditioner
      is deliberately frozen for many steps (re-freezing the flow block is measured unhelpful), while the
      shift is a *local time scale* that should describe the operator being solved now. Worse, the
      borrowed quantity was **live in velocity but frozen in viscosity**, so in a coupled solve `μ_eff =
      ρ(ν + ν_t)` was stuck at its freeze-state value and the velocity time scale ignored the eddy
      viscosity that develops — nobody chose that, it fell out of reusing the preconditioner's assembler.
      `FrozenViscosityVelocityParts` (the default, `None`) reproduces the old diagonal **exactly**;
      `LiveViscosityVelocityParts` re-forms the closure at the state being stepped, costing one closure
      evaluation per *step* (milliseconds against a tens-of-seconds solve). It receives the scalar blocks
      **as solved**, so it holds the transforms and maps back to physical — passing `log ω` to the closure
      would silently build the shift from the wrong field. Note `a_p` still comes from the preconditioner
      and must: it is what the velocity block inverts, so the two would otherwise disagree.
      *Measured:* live vs frozen viscosity at a developed state is a modest p50 0.96 / max 1.82 with **no
      tail** — a correctness fix, not a large lever.
      - **Where the pieces live (binding, #156 seam 3).** The `VelocityShiftParts` **protocol** lives in
        `solve/shift_basis.py`, beside `ShiftBasis` — the natural sibling: `ShiftBasis` says *how* to
        combine the buckets, `VelocityShiftParts` says *where they come from*. `FrozenViscosityVelocityParts`
        (needs only a `BlockPreconditioner`) lives in `flow/continuation.py`; only `LiveViscosityVelocityParts`
        (needs `SSTTurbulence` + the transforms) stays in `turbulence/coupled.py`. This is what lets the
        **flow-only** `MomentumShiftPolicy` carry the *same* `velocity_shift_parts` seam (it could not
        before — the protocol lived in `turbulence`, and `flow` importing it would be a cycle). Its
        `parts(flow, k_solved=None, omega_solved=None)` makes the turbulence blocks optional, so the
        flow-only policy calls `parts(flow)` while the coupled one passes all three. The flow-only path is
        still behaviour-unaffected (constant μ ⇒ frozen and live coincide); the seam exists for symmetry
        and a future variable-viscosity flow, and to delete the byte-for-byte duplicated inline pattern.
      - **⚠️ `LiveViscosityVelocityParts.parts` declared the two turbulence blocks WITHOUT the protocol's
        defaults until 2026-08-20 (#282), so it did not satisfy the arity its own protocol promises.**
        Latent — nothing hands the coupled implementation to the flow-only policy today — but it would
        have become a `TypeError` from inside a shift policy the moment the monolithic builders started
        accepting a `velocity_shift_parts`, which is the same change that fixed it. It now takes the
        call and raises a message naming `FrozenViscosityVelocityParts`, because unlike its sibling it
        genuinely cannot do the job without the turbulence context. **A `Protocol` is not checked at
        runtime, so an implementation narrower than its protocol is invisible until the call that needs
        the wider arity — write the defaults even where the implementation will reject them.**
      - **`velocity_shift_parts` reaches all FOUR coupled builders since 2026-08-20 (#282).** It was on
        the two block builders only; `_monolithic_shift_source` now takes and forwards it. Nothing
        structural had excluded it — a live source needs momentum + turbulence + the transforms, not a
        flow preconditioner — and the configuration it was built for, a dual-time low-shift march whose
        shift tracks the developing `ν_t`, is a *monolithic* one. So it was absent from precisely the
        path it was written for.
    Adding a
    basis (e.g. a Fluent-style global min-physical-time-scale) is a new `ShiftBasis`; do not branch the
    policies.

  - **`ShiftTerm` has THREE parts, and two of them are closures of `β` for the same reason (2026-09-11).**
    `diagonal` is the base per-row time scale; `make_preconditioner(β)` builds the inverse of the
    operator that `β` actually shifts; and `row_shift_scale(β)` is an optional per-row multiplier on the
    strength, so different blocks may run at different pseudo-timesteps. `ShiftTerm.shift(β)` is the one
    place they are combined — `β · row_shift_scale(β) · diagonal` — and **every** consumer goes through
    it, including anything assembling a preconditioner for that same shifted operator, so the operator
    and its inverse cannot disagree about what "the shift" is.
    - **⚠️ A BLOCK FACTOR MUST NOT BE FOLDED INTO `diagonal`, for two independent reasons.** `β` reaches
      the step **already clamped** at `beta_min`, so a factor in the diagonal multiplies the *floor* and
      that block never stops being damped after the march has switched the damping off for every other
      one. And the base diagonal doubles as the **row scale of the row-equilibrated residual measure**
      (`coupled_scaled_norm`), so a factor there divides that block's reported residual — silently
      moving the bar the march is steered by, stopped on and compared across arms with. Both bit once;
      see `.claude/rules/turbulence.md` under `turbulence_damping`.
    - **⚠️ A WRAPPER POLICY THAT REBUILDS A `ShiftTerm` FROM `.diagonal` ALONE DROPS THE REST IN
      SILENCE** — the march runs and the dropped behaviour simply never happens. Two shipped wrappers
      did exactly that (`MonolithicFactorShiftPolicy`, and the mass-flow policy that is now
      `flow.BorderedShiftPolicy`, which forwards `row_shift_scale`), which is how a
      per-block damping measured as a no-op. Forward every part, and find the wrappers **structurally**
      (AST), not by grepping the callee's name.
    - **`residual` is `R(φ)` when the caller has it and `None` otherwise**, offered because both steps
      evaluate it immediately before asking for the shift, so a policy whose shift depends on the
      distance from a root reads it for free. ⚠️ **It is optional, so omitting it does not fail** — such
      a policy quietly falls back to its no-residual branch and runs as a constant. `DualTimeStep`
      omitted it until 2026-09-11 while computing the very same value one line above, which made every
      residual-dependent policy inert on every case that runs `inner_steps > 1`. Pinned on both steps.

  - **Two injected seams**, both `Protocol`s: the physics comes from a **`ShiftPolicy`**
  (`shift_term(φ, residual=None) -> ShiftTerm(diagonal, make_preconditioner, row_shift_scale)`;
  `ShiftTerm.diagonal` is the full-state base shift, `make_preconditioner(β)` the frozen shifted `M`),
  and the per-attempt accept/reject
  decision from a **`StepAcceptance`** (`accept(candidate_norm, residual_norm, residual_norm_0,
  attempt) -> bool`). The escalation-loop *mechanics* (grow `β`, cap at `max_escalations`, carry the
  best candidate) stay in the engine; only the decision is delegated. Default acceptance is
  **`DivergenceGuard(divergence_cap=10.0)`** — accept unless the candidate is non-finite or exceeds
  `divergence_cap·‖R₀‖` (a divergence guard, not a descent test, since the march is non-monotone); a
  monotone / forcing rule is a drop-in `StepAcceptance` — do **not** hardwire an acceptance test into
  the `while_loop`. So the engine is
  reusable for **any** nonlinear residual (reaction/energy/turbulence), not just the coupled flow —
  verified in `tests/unit/test_pseudo_transient.py`, which drives it on a scalar root with a trivial
  policy (no mesh, no flow). The flow application is `aquaflux/flow/continuation.py`'s
  `MomentumShiftPolicy` (velocity-block `a_P` shift + shifted SIMPLE preconditioner), configured into a
  `PseudoTransientStep` by the `momentum_continuation(assembler, …)` **factory** (which builds the
  block preconditioner and injects the `DivergenceGuard` + adjoint factory) — **no wrapper/adapter
  class**, since `PseudoTransientStep` is itself the `NewtonStrategy`. The scalar application is
  `aquaflux/turbulence/continuation.py`'s `ScalarShiftPolicy` (the transport operator diagonal — the
  scalar `a_P` analogue from `scalar_transport_shift_diagonal` — as the base shift, with the frozen
  scalar-transport AMG reused **unshifted** as `M`, since the shift only adds positive diagonal),
  globalizing the stiff k/omega solves via `scalar_pseudo_transient_solve` — the **only** scalar path
  the SST driver supports (the fixed-count Newton sub-solve was removed). When a new nonlinear residual
  needs pseudo-time globalization, write a `ShiftPolicy` — do **not** re-implement the march.
  - **`stepper()` returns a `StepOutcome` (`phi_next`, its `residual_norm`, `cycles`, `alpha`, and
    four more — see `solve-march.md`) — ONE step method on the whole `NewtonStrategy` protocol,
    counted/uncounted pair deleted (binding).** Every strategy reports its step's restart-cycle count
    (`DampedNewtonStep` gets it from `newton_correction`, which now returns `(delta, r, cycles)`); a
    consumer with no use for it drops it (`phi, _ = step(…)`). A `counted_stepper()` sibling existed
    briefly, with `stepper()` forwarding to it and dropping the count — deleted for the same reason as
    `solve_linear_counted` above, and note it had **no production consumer at all** while it existed.
    The reported count is the **accepted** attempt's, not the sum over rejected escalation attempts —
    the cost of the step actually taken. **A step whose every attempt was rejected reports `0`**
    (`best_cycles` is only written on acceptance): a consumer must treat `0` as *no measurement*, not
    as *free*, or a rejected step reads as the cheapest in the march. Consumed by `newton_march`
    (`solve-march.md`), which is since 2026-09-15 the only march there is.
  - ⚠️ **"The count is NOT carried out of `_forward`'s `while_loop`" is DEAD — there is no such loop
    (deleted 2026-09-15, phase 3).** Both of its reasons have lapsed: a generic Newton loop no longer
    has to choose which step's count survives, because the one march reports **every** step through
    `StepReport`; and the `int32`-in-a-`custom_vjp`-primal objection went when the loop moved outside
    the `custom_vjp` (the adjoint is attached afterwards by `root_adjoint`). Per-step cost is simply
    observed.

  - **`line_search` — backtrack the shifted step before escalating β (binding, the coupled-RANS fix).**
    The step optionally scales the shifted correction `δ` back along `{1, 1/2, …, 1/2**line_search}`
    (`backtracking_line_search`, extracted from `implicit.py` and shared with `DampedNewtonStep` — one
    home for the ladder) and keeps the largest length that reduces the residual, **before** the
    accept/escalate test. The ladder is a **`lax.while_loop` that stops at the first (largest) reducing
    rung** — a full step that already descends (the common case near the root) costs one residual
    evaluation, not `line_search+1`, and the loop body compiles once instead of unrolling `line_search+1`
    residual copies into the graph. It is safe as a non-differentiable `while_loop` because the search is
    **forward-only**: it runs inside a Newton iteration on `stop_gradient` inputs, whose derivative is
    attached afterwards at the root by `root_adjoint` and never differentiates the iteration (every
    caller is a `NewtonStrategy`; nothing differentiates through it — audited). Do **not** call it on a differentiated
    path. `line_search=0` (default) is the old behaviour: take the full step `φ+δ`, and
    the **only** recourse to an overshoot is escalating β — a *full re-solve*. This was measured to be
    the dominant coupled-RANS cost: from the hybrid IC the full coupled Newton step overshoots by
    ~10⁷× (‖R‖ 220 → 5.8e9), so every step burned ~4–7 expensive re-solves and, worse, escalating β
    (β=16/64) still did **not** descend (rel ≈ 1.0 — the march stalled). A line search on the **one**
    β₀ solve finds α≈¼ → rel≈0.48 (residual halved) in a few cheap residual evaluations. So the
    coupled path sets `line_search>0` (`coupled_continuation`, `_COUPLED_LINE_SEARCH=10`); β escalation
    stays the fallback for a genuinely bad *direction* (an ill-conditioned shifted solve), not an
    overshoot. Like the shift, the search only reshapes the forward path — converged state and IFT
    adjoint unchanged. The flow and scalar paths leave `line_search` unset, so they take the step's own
    `0` and are bit-identical — a default now, not a limit: since #372 either can be handed
    `Globalization(line_search=…)` without constructing the step by hand.
  - **The line search returns its residual VECTOR, and the dual-time step runs on it (BUILT 2026-10-17,
    branch `claude/step-program-size`).** `LineSearchStep.residual` is `R(phi)` at the returned iterate,
    carried through the ladder beside its measure (the last rung walked is the accepted one whenever there
    is one; the fallback rung's travels with `longest_norm`). **`alpha = 0` is the branch to get right:**
    the iterate is the start point, which the ladder never evaluated, so the vector is the caller's own
    `backtracking_line_search(..., residual=R(phi))`, and `None` when the caller passed none (the
    validation probes; a library caller always passes it — all three strategies do, so they read the
    search the same way). With no finite rung it is a NaN seed, non-finite like the measure.
    `DualTimeStep` uses it twice: the carried `G` is the next inner iteration's Newton right-hand side
    (the anchor's `R(φⁿ)` seeds it, the transient term being zero there), and the step's reported
    `R(phi_next)` is that `G` less `transient_term(phi_next)` — one subtraction, through the same helper
    `G` was built with. **Not bit-identical to re-evaluating `R`**: it rounds at the size of
    `β d (phi − φⁿ)`, which vanishes at the march's root; and the carried right-hand side was evaluated at
    `phi + alpha delta` inside the ladder rather than at the iterate formed after it, so it may differ by a
    rounding there too. On the cube root the reported norm matches a fresh `R` to within `8 eps (|R| +
    |β d Δphi|)`; pitzDaily's march is recorded in `solve-globalization-log.md`. `PseudoTransientStep`
    and `DampedNewtonStep` gain nothing inside a step (each evaluates `R(phi)` once, at the start); what
    #649 asks beyond this — carrying the vector ACROSS outer steps through `newton_march`, which would
    also drop the dual-time anchor's `R(φⁿ)` — is not built. Pinned by
    `test_the_dual_time_step_evaluates_the_residual_once_per_inner_iteration` (executions counted by a
    `jax.debug.callback`, Jacobian from an uncounted copy), `..._reports_the_steady_residual_at_the_iterate_it_returns`,
    `test_a_step_whose_every_inner_search_takes_no_step_reports_the_anchor_s_own_residual`,
    `test_a_dual_time_march_step_traces_the_residual_only_where_it_must` and the `.residual` assertions
    in the four line-search fallback tests (`test_newton_march.py`); five of six single-line mutations turn them red. The one
    surviving mutation (handing the stalled search a wrong `g`) is unreachable through a step: a stalled
    inner iteration is a fixed point, so the next solve starts where it did and stalls again.
  - **`krylov_solver` overrides the shared `_INEXACT_CONTINUATION_SOLVER`; the coupled default stops on a
    relative residual in an INJECTED norm (`relative_residual_gmres`, `solve/linear.py`).**
    `linear_solver()` returns the injected `krylov_solver` when set, else the shared restart-40 GMRES.
    The coupled path injects restart 120 (the stiff saddle needs hundreds of restart-40 cycles; a
    40-vector subspace discards too much Arnoldi history). ⚠️ **The norm it stops in has MOVED since this
    was written — the "global 2-norm" below describes the arrangement these measurements were taken
    under, not the current default.** Since #282 every coupled family stops in the march's own row-scaled
    measure at the forward regime's `rtol = 0.3` (`LinearSolveSettings`, #388), built by `_coupled_step` rather
    than by any builder; see
    `solve.md`'s regime table. The mechanism below is unaffected and is why the componentwise stock stop
    was abandoned in the first place. **The dominant waste was the TERMINATION, not the restart.** The
    old `GMRES(rtol=1e-3, atol=1e-10)` reached true_rel ~4e-12 in ~15 restart cycles / 1800 matvecs on the
    cold-IC pitzDaily solve although only rtol=1e-3 was asked — because `lineax`'s stock stop is
    **componentwise** (`|r_i| ≤ atol + rtol|b_i|` under max_norm), and the ~470 near-wall ω wall-fixation
    rows start satisfied (right-hand side ~0), so their per-row scale collapses onto the absolute `atol`
    floor and a handful of them hold the whole solve to ~1e-10 (~9 orders past 1e-3). `relative_residual_gmres`
    scales the system to unit right-hand-side 2-norm and runs GMRES at `rtol=0, atol=target, norm=2-norm`,
    so it stops on `norm(r)/norm(b) ≤ target` — immune to those rows. ⚠️ **That residual is the TRUE
    one, not `‖Mr‖`: `_shifted_solve` takes `solve_linear`'s `preconditioner_side="right"` default, so the
    Krylov residual is `b − A M y = b − A x`. No "solution accuracy" follows from it — an earlier version of
    this line inferred "≈1% solution accuracy since M≈A⁻¹", which holds only under LEFT preconditioning and
    misled a later reader into a wrong hypothesis.** Measured on the real cold-IC march: ~3-5 cycles (often 2-3/step), ~4× fewer
    matvecs to the same `x_r/h`, trajectory unchanged.

  - **The residual measure is an injected `ResidualNorm`, owned by the `NewtonStrategy` (`solve/norm.py`).**
    Every `NewtonStrategy` exposes `norm()`; the march reads it for the outer stopping test
    (`newton_march`'s segment reference, or a per-iteration `norm_builder`) and the strategy
    uses the *same* measure for its own globalization — so the convergence test, the SER ramp
    `β = β₀(‖R‖/‖R₀‖)^p`, `backtracking_line_search` (which now takes a `norm=` kwarg), and the
    `DivergenceGuard` all agree on one scale. Default is `jnp.linalg.norm` (`DampedNewtonStep.norm()` and
    `PseudoTransientStep`'s `residual_norm` field both default to it), so **the flow path is
    bit-identical**. The non-trivial impl is `BlockScaledNorm(sizes, scales)`: it splits the flat
    residual into contiguous blocks, divides each by its own reference magnitude, and returns the L2 of
    those per-block relative residuals — `sqrt(Σ_b (‖R_b‖/scale_b)²)`. **Why it exists (the coupled-RANS
    fix):** the plain Euclidean ‖R‖ on `[flow, k, ω]` is ~100% ω (ω residual O(1e5), k O(1e-3)), so the
    line search can neither *see* nor *protect* the k block — a step that collapses k is accepted (barely
    moves the ω-dominated norm) while one that reduces k is vetoed because ω ticked up, and k gets
    starved (measured: k median collapses to ~7e-5 vs a physical ~0.5, and the march freezes). `coupled.py`
    builds a `BlockScaledNorm` over `[flow, k, ω]` (and `[…, β]` for the mass-flow bordered march) with
    per-field scales `‖R0_field‖` at the reference state, so the whole system is judged. The adjoint never
    forms a residual norm, so the measure is a **forward-only** device — the converged state and IFT
    gradient are norm-independent, and `root_adjoint` does not take one at all.
  - **A `ShiftPolicy`'s preconditioner is DATA — REVERSED 2026-10-12 (#386); it was a binding
    non-pytree from #105.** `ScalarTransportPreconditioner` (`turbulence/preconditioner.py`) and its
    strategies are `equinox.Module`s with their hierarchies on the traced side, and the coupled block
    policy's adjoint factory is `solve.UnshiftedPreconditioner(policy)`, not a lambda. **Why reversed:**
    a viscosity ramp re-points the block session at every station, and a station step holding new static
    preconditioner objects recompiled the coupled solve at every station (~40 s against ~7 s per step on
    the re20000 channel); as data, a step whose hierarchies keep their coarsening differs only in values,
    and the march takes it as a cache hit. **The two #105 hazards, re-examined:** (ii) the cache — a
    carried instance is still a hit, and now so is a rebuild on an unchanged coarsening (two-level, which
    aggregates by the graph); a rebuild whose coarse shapes move (a fresh lAIR split, a flow block
    re-aggregated at another viscosity) is a miss by *shape*, which is why `RebuildingSession._repointed`
    derives the new step from the held one. (i) the adjoint — `UnexpectedTracerError` came from a solve
    taking the preconditioner as a traced argument; the march now runs on `stop_array_gradients` copies,
    so the policy's arrays reach `root_adjoint`'s non-differentiated slot concrete, as
    `_TracedSplitAdjoint`'s have since #290. Confirmed by CI's slow tier on PR #704 (`full-ci`, head
    `78e0bbf`): the tests that `jax.grad` through a block-diagonal coupled solve
    (`tests/integration/test_coupled_rans.py`, `test_reynolds_continuation.py`,
    `test_coupled_mass_flow.py`) all pass. Pinned by
    `test_a_re_pointed_block_session_step_is_a_compilation_cache_hit` (both scalar blocks, with a
    freshly built control that misses) and
    `test_the_scalar_solve_compiles_once_whether_its_preconditioner_is_carried_or_rebuilt`, both
    mutation-checked.
- **`norm_builder` — the residual measure is re-derived every outer iteration, and held FIXED within
  one (binding).** `newton_march(norm_builder=…)` takes a `(step, state) -> ResidualNorm` (the step so a
  measure can read its row diagonals from the step's shift; since #370 a solve builds it from its
  `Convergence` measure with `ResidualMeasure._builder`) and, at the top of
  each iteration, swaps the rebuilt measure onto the step with `eqx.tree_at` (the same mechanism the
  α-control uses for β) and re-measures `residual_norm_0` against it so the SER ratio stays on one
  scale. Every line-search trial step, the acceptance test and the reported norm within that iteration
  then use the *same* measure — **rebuilding per trial step would let a candidate win by shrinking its
  own denominator rather than its residual**, so the search would stop comparing like with like.
  - **This is why `residual_norm` is a DATA field on both `NewtonStrategy`s, not a static one.** A static
    field lives in the treedef, so swapping it would be a new compilation *every step*. As data, and
    with the measure carrying its scales as traced leaves over a fixed block structure
    (`RowScaledNorm`), the swap is a cache hit. A plain callable (the default) has no array leaves and
    is filtered to the static side regardless, so the default path is byte-identical.
  - **⚠️ AND THE STEP'S LINEAR SOLVE MUST READ THE SWAPPED MEASURE, which it did not until #370
    (2026-09-16).** The swap reaches `step.residual_norm` only. The march resolves its Krylov solver
    **once** (`strategy.linear_solver()`), and `_coupled_step` had built that solver with
    `norm=residual_norm` captured at build — so under a rebuilt measure the outer stop, line search and
    shift used the rebuilt one while every inner `rtol = 0.3` stop used the build-time one (on a
    refreshed march, the initial-state measure carried into each refresh). Both flagship cases ran this
    way. The fix: `relative_residual_gmres(norm=None)` means "the measure of the step that runs me", and
    all three steppers (`PseudoTransientStep`, `DualTimeStep`, `DampedNewtonStep`) call
    `solve.linear.in_progress_measure(solver, self.residual_norm)` before solving; an explicit norm or a stock
    `lineax` solver is left alone, and a `norm=None` solver run outside a step raises. Pinned by
    `test_convergence.py::test_each_step_stops_a_measure_following_linear_solve_in_its_own_measure`.
  - **`RowScaledNorm` was march-only because `RootSolver` passed its measure through a
    `custom_vjp` non-differentiable (hashable) slot.** That slot is gone (2026-09-15: the loop runs
    outside the `custom_vjp`, and `root_adjoint` takes no measure), and `solve_coupled` no longer ends
    in a traced finishing solve. Whether `RootSolver` now accepts a rebuilt `RowScaledNorm` is
    untested.

