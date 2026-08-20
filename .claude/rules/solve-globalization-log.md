# Investigation log — `aquaflux/solve/` globalization

> Split out of `solve.md` / `solve-globalization.md` (2026-08-18). **No `paths:` frontmatter — this
> file never auto-loads.** It holds the dated investigation behind the current globalization
> architecture — the residual-measure choice, the shift-basis probing, the SER schedule reversal, and
> the line-search behaviour — including entries that a later measurement corrected or superseded in
> place. See `solve-globalization.md` for the current, load-bearing architecture and defaults.

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
        was the log-space ω shift, `.claude/rules/turbulence.md`). **Treat %/s sweeps as a way to find
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
      verification recorded in `.claude/rules/turbulence.md`), and the entire remaining gap is the
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
      - **A `max` near-wall blend is better on CONVERGENCE, not only on accuracy.** `.claude/rules/turbulence.md`
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
      - *Preconditioner inconsistency* (the MSIMPLER Schur ignores the shift, which for a non-uniform
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
      #163).** `apply_at` feeds the velocity block the shifted diagonal `a_P + β d`, while the MSIMPLER
      Schur uses `Q̂/k` calibrated from the **un-shifted** diagonal, i.e. it ignores the shift entirely.
      That looks like an inconsistency, and for a non-uniform basis the discrepancy is spatially varying
      (`1/(1 + β·share)`, share 0.30–0.97) rather than a global scalar. It costs nothing. A 2×2 at fixed
      β, varying only `schur_scaling` (`simple` uses the shifted `a_p` and is consistent by
      construction):

      | basis | `msimpler` (shift-blind) | `simple` (consistent) | ratio | α (both) | rel (both) |
      |---|---|---|---|---|---|
      | `a_P`, β = 2 | **15** cyc | 36 cyc | 2.4× | 1.000 | 4.8530e-01 |
      | convective, β = 1 | **18** cyc | 34 cyc | 1.9× | 0.125 | 9.2719e-01 |

      Within each basis `α` and the residual ratio are **bit-identical across all three steps measured**
      — the "a preconditioner changes cost, not the converged step" property, which also confirms these
      solves genuinely converge. There is **no interaction**: the consistent Schur is uniformly ~2×
      worse, and *less* bad on the convective basis (1.9× vs 2.4×) — the opposite of the hypothesis.
      `Ŝ` is an approximation chosen for **spectral quality**, not a derivation of the true Schur
      complement; MSIMPLER's whole premise is replacing `a_P` with a velocity-independent mass-matrix
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
        solver is deliberately inexact** (`_COUPLED_FORWARD_SOLVER` runs at `rtol = 1e-3`). At a finite
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
    `forward_march(..., max_steps=1)` on the pitzDaily *plateau* state ran **27 min 42 s** without
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
        (`solve/step_control.py`).** `DualTimeStep` is a `ForwardStep` whose `stepper()` holds a reference
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
      any more (see `solve-direct-preconditioners.md`). The premise is gone too: the shipped AMG
      preconditioners solve to **adjoint grade at zero shift**, which is the very regime this entry
      claimed only a complete factorization could reach. The one line worth keeping is the trap: a
      dual-time march driving β low is where a preconditioner's low-shift behaviour is tested, so measure
      there and not at the march's floor. ⚠️ And note what the entry never controlled for — the
      block-SIMPLE path had **no k-positivity limiter at all** until 2026-08-19 (`turbulence.md`), so an
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
  OUTER-ITERATION boundary only:** when a `norm_builder` is supplied, `forward_march` rebuilds the
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
    equilibrated one — and it exposed a fallback bug (2026-07-27).** (The equilibrated/row-scaled measure
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
    - **The fallback when NOTHING is admissible is the longest FINITE rung, not the shortest (binding).**
      Returning the shortest is a near-null step that changes nothing, which the divergence guard then
      accepts as finite: the march reports a step and stands still. That is a *guaranteed* stall rather
      than a slow one, and it is what produced the `α = 0.001` signature (`0.001 = 1/2**10`, the smallest
      rung of the shipped 10-rung ladder — a value that means "nothing passed", not a sentinel).
    - **The ladder can extend ABOVE α = 1 (`grow` rungs of doubling; default 0 = off).** Measured on a
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
  `forward_march` entry, hence reset at every preconditioner refresh. With a refresh every handful of
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
    through `coupled_continuation(beta_floor=…)` for further evaluation.
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
      the inner solve CANNOT fix it (measured; do not re-attempt).** The block-diagonal conv+MSIMPLER
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
        MSIMPLER's Schur is velocity-independent. **Confirmed on the real solve:** refreshing only the flow
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
        **BUILT** — `forward_march` + `CycleGrowthTrigger` (see the `march.py` section) segment the march
        around the off-jit rebuild, which is required because the traced solve is one `lax.while_loop` and
        scipy AMG assembly cannot run inside it; `solve_coupled(refresh=RefreshPolicy(trigger=…))` is the driver.
        **⚠️ SETTLED FROM THE CODE — the old claim here, "a refresh still forces a full recompile because these
        are non-pytrees hashed by identity", is SUPERSEDED and deleted.** The fix it proposed as hypothetical was
        built: the coarsening structure is value-independent **at this path's `strength_threshold=0`**, and
        `_SparseLevel` now holds only `n` / `n_coarse`
        static with `val` / `diagonal` / `lam_max` / `coarse_inv` as **traced leaves** — so a refreshed hierarchy
        passed as a jit argument is a **compilation-cache hit**, pinned by
        `test_refreshing_a_hierarchy_is_a_compilation_cache_hit`. ⚠️ **The value-independence is a property of
        the threshold, not of the level split**, so it does not carry to the native flow block, which runs at
        0.25 and re-partitions on every refresh; that path keeps the cache hit with
        `SmoothedHierarchy.refit` instead (see the flow-block section). What a refresh still costs is the off-jit scipy
        rebuild plus the one-off retrace of the rebuilt `ForwardStep`, which is why `refresh.limit` still bounds
        it. The wall figures once attached to this question (a "~60–240 s" recompile and a "~38 s" refresh) were
        both recorded with no configuration and are deleted with it.
      - **The observed march RETURNS ITS OWN CONVERGED STATE — the traced finishing solve is only the
        not-converged fallback (BUILT).** `solve_coupled`'s observed path (`on_step`/`refresh`/`step_control`)
        is never differentiated — those cannot run under a JAX transform (guarded), so the converged eager
        state needs no adjoint. When the eager `forward_march` reaches its stopping tolerance **judged in the
        measure it steered by** (the per-step-rebuilt `RowScaledNorm` under `scaled_norm`), `solve_coupled`
        returns that state directly instead of re-marching it through `ImplicitNewtonSolver`. **Why this is
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
      - **Rescaling the MSIMPLER `k` is a ρ mirage — validate on the real march, never on ρ.** Growing `k`
        collapses ρ but barely moves the one-shot error (figures deleted with the rest of the unconfigured ρ
        evidence above), and the ρ-minimizing
        `k` sits ~40× *above the maximum* of the whole per-cell `ρV/a_P` distribution — i.e. the degenerate
        limit `schur_a_p → 0`, `Ŝ⁻¹ → 0`, which simply switches the pressure correction off. On the real
        production march it is **slower**: shipped auto-`k` 348 s / 8 steps vs `k×4` 447 s (28% slower) at
        an identical residual trajectory. **The shipped per-apply `mean(ρV/a_P)` calibration is
        near-optimal — do not "fix" it**, and do not make the Schur "shift-consistent" with the
        pseudo-transient `a_P(1+β)` either (that direction is strictly worse at every β).
      **Root cause:** the MSIMPLER Schur is a *constant-coefficient* (scaled pressure-mass-matrix) Poisson,
      which is a near-Stokes/low-Re approximation and degrades as convection strengthens — exactly the
      high-Re/recirculating regime here.
      - **⚠️ The "obvious" fix — a better Schur (stabilized LSC) — WAS BUILT AND LOSES BADLY on the
        coupled solve. Do not re-derive it (binding).** `schur_scaling="lsc"`
        (`flow/block_preconditioner.py`) implements the algebraic, nonuniform-mesh stabilized
        least-squares commutator of Elman, Howle, Shadid, Silvester & Tuminaro (2007) — the *right*
        variant for a Rhie–Chow collocated (equal-order stabilized) discretization, with the viscosity
        cancelled so it serves a variable-viscosity closure. Measured on one shifted solve at a
        developed/separated pitzDaily state:

        | Schur | cycles | wall |
        |---|---|---|
        | **msimpler** | **13** | **38.9 s** |
        | lsc (`v_cycles=4`) | 96 | 526 s |
        | lsc (`v_cycles=8`) | 82 | 662 s |

        6–7× the cycles and 13–17× the wall time, with both solves genuinely converged
        (`lin_rel ~2e-9`), plus ~2.9× slower on the coupled channel at an identical residual trajectory.
        **Why the flow-only win does not transfer:** LSC *does* beat MSIMPLER on the isolated flow block
        (9 vs 15 GMRES at Re=1e4), but on the coupled block-*diagonal* preconditioner under the
        pseudo-transient shift, a better isolated flow-Schur does not reduce *coupled* cycles — the
        coupled iteration is not limited by the flow block's Schur quality. Keep the strategy (it is a
        legitimate option for a flow-only solve); do **not** make it the coupled default, and do not
        propose it again as the cure for coupled cost.
      - PCD remains deprioritized regardless: its auxiliary pressure convection–diffusion operator
        carries finite-element boundary recipes that do not transfer cleanly to cell-centred FVM.
      - **What a preconditioner can and cannot change — state this precisely, both halves are measured.**
        *While the linear solve actually converges*, swapping the preconditioner changes **cost only**:
        msimpler vs LSC gave coupled residual trajectories identical to **5 significant figures**, so the
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
        returned `δ` is no longer the tolerance-defined one — and `_COUPLED_FORWARD_SOLVER` runs
        `rtol=1e-3` with `stagnation_iters=40`, which makes that reachable. **Practical rule:** treat
        "preconditioner ⇒ cost only" as true when solves converge comfortably, and stop trusting it
        once the cycle count is climbing toward the solver's limits.
