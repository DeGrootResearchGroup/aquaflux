# Refuted / closed directions — `aquaflux/solve/`

> Split out of `solve.md` (2026-08-18). **No `paths:` frontmatter — this file never auto-loads.**
> It is tracked so a refuted idea can be re-adjudicated later (per the root `CLAUDE.md` rule that
> findings belong in tracked files, not memory), but it is deliberately kept out of the
> auto-loaded path so routine solver work does not pay for the full negative-results ledger.
> Read this before proposing a preconditioner/globalization idea that sounds like it should already
> have been tried on `bfs3d` — check here first. Each entry below states what was tried, on what
> case/state, and why it lost; full investigation detail is via the parent-file link at the top of
> the corresponding topic entry.
>
> **Write, don't just read: add an entry here in the SAME change that refutes or closes a direction.**
> One short paragraph — what was tried, on what case/state, and why it lost — plus a pointer to the
> full detail wherever it lives (a topic file, a `-log.md` file, or inline here if it is short enough
> to need no pointer). This ledger is what a future contributor actually greps before re-proposing an
> idea; a refutation that lives only in a `-log.md` file's prose will not be found by that search.

## Low-β directions already measured out — CLOSED, do not re-litigate

- **⚠️ LOW-β DIRECTIONS ALREADY MEASURED OUT — do not re-litigate without new evidence.** The low-shift
  wall on the 3D coupled saddle has absorbed a lot of probing. What is settled:
  - **Turbulence decoupling ("just lag ω") — REFUTED.** A true-residual arm comparison found
    block-diagonal `(u,v,w,p) ⊕ exact(k,ω)` to be the **worst** arm tested: the flow–turbulence coupling
    is load-bearing in the preconditioner, not a nuisance to be segregated away.
  - **A pre-AMG SIMPLE-type transform of the matrix — DEAD.** The published "SIMPLE preconditioning" for
    monolithic coupled AMG *is* Rhie–Chow interpolation; our residual already assembles that matrix, so
    there is no transform left to apply. (Established by reading the primary sources in full, not from
    abstracts.)
  - **PC-only pressure-Poisson augmentation — NO-GO, triple-confirmed.** The `(p,p)` block is *already*
    0.71× the SIMPLE-Schur elliptic operator, and the augmentation degraded cycles ~2.7×.
  - **`coarse_eq_limit` beyond ~2000 — inert, but read this correctly: the ARM was a no-op, which is
    not the same as a null result.** GAMG stops coarsening as soon as the grid falls below the limit,
    and this hierarchy is already **2 levels** — its single aggregation step has landed under 2000
    already, so a *larger* limit cannot make it stop any sooner and produces a bit-identical
    hierarchy. `K=8000 ≡ K=2000` is therefore arithmetic, not evidence, and it says nothing about
    whether the coarse space matters. **To probe the coarse space, degrade it instead**
    (`mg_coarse_pc_type: jacobi` in place of the direct LU) — that reads in both directions, and it
    is the control that decides what a *smoother* plateau means: if degrading the coarse solve barely
    moves the cycle count then the coarse correction is not load-bearing, and a smoother plateau
    cannot be attributed to the coarse space at all. Always print the coarse grid's equation count
    (`AmgVCycle.coarse_size`) beside the level count, so an arm that changed nothing is
    distinguishable from a setting that made no difference.
  - **Additive Vanka + Richardson — invalid by construction** (Richardson on an indefinite saddle).
    **⚠️ THE COARSE-SPACE READING THAT SAT HERE IS DELETED — the two entries conflicted and the configured
    side wins.** The losing claim: a Krylov-Vanka arm "still stalls, insensitive to the inner count, which
    points at the coarse space as the remaining wall" — recorded with no smoother, aggregation, state or
    shift. The winning claim is the 2026-08-08 arm table below, which records its configuration in full:
    degrading the coarse solve to Jacobi leaves the cycle count unchanged, so the coarse correction is not
    load-bearing there. The split was only ever established for the *2-level* GAMG hierarchy we run.
  - **Where the V-cycle actually under-performs:** per-field, pressure is *well* smoothed and **ω** is
    the unsmoothed field, then `u`. (The ratio that quantified "unsmoothed" is deleted — configuration not
    recorded; re-measure before relying on any magnitude.) If a per-field lever is wanted in 3D, ω is it —
    pressure is not.
    **⚠️ PARTLY REHABILITATED (2026-08-08): a second, independent measurement puts ω in the same
    place.** The cell-block singular-value decomposition above finds the near-null direction of the
    worst diagonal blocks to be **pure ω**, in 353 of 23040 cells, under the *current* bundle. That is
    local block conditioning, not per-field V-cycle smoothing rates — a different quantity by a
    different route — so "ω is the 3D per-field lever, not pressure" now rests on two legs instead of
    one unfalsifiable one. The *factor* that once accompanied it is deleted for want of a configuration; treat
    the **field** as corroborated and any magnitude as unmeasured.
    **⚠️ NEITHER THIS NOR THE VANKA BULLET RECORDS WHICH SMOOTHER OR AGGREGATION IT WAS MEASURED
    WITH, so neither can be relied on now.** The smoother default has since moved ILU(1) → ILU(0) and
    the aggregation smoothed → plain, and *both* of those changes have inverted a conclusion on this
    case. Concretely, the Vanka bullet's inference — "a strong smoother still stalls, therefore the
    **coarse space** is the wall" — is valid but **under-determined**: "the coarse space" could mean
    the space is intrinsically inadequate (needing inf-sup-aware coarsening, a research problem) or
    that the coarse *correction* was corrupted by prolongator smoothing (a setting, now changed). A
    stall at 5–6e-2 fits both, so that experiment never distinguished them. Re-measure before building
    on either, and **record the smoother and aggregation** in any replacement.
    **The second reading is now measured, not speculative.** Prolongator smoothing *was* degrading the
    coarse correction on this operator: turning it off (`pc_gamg_agg_nsmooths = 0`) is worth
    **22 → 9 cycles** at a hard state and ~16 % of the whole march's Krylov cost. Every arm in that
    Vanka campaign was judged against a coarse correction built with smoothing **on**, so a *smoother*
    arm failing to rescue it is what you would see whether or not the smoother was any good — which
    also explains why the campaign kept concluding "the smoother is not the lever" whichever smoother
    it tried. **Do not treat "the coarse space is the wall" as settled.**
    **⚠️ FIRST MEASUREMENT UNDER THE CURRENT BUNDLE (2026-08-08) — and it moved the question.**
    Configuration, in full, because that is the point: 3-rung `bfs3d` cold march to ‖R‖ = 2.64e-6 (69
    steps, 60.5 min); state `state-00049` (rung 3, the target Reynolds number); operator β = 0.0293
    with the V-cycle at the floor 0.05; **plain aggregation, ILU(0), 4 sweeps, `coarse_eq_limit` 2000
    → 2 levels with 1296 coarse equations**; real right-hand side `−R(state)`, judged on the true
    residual through GMRES at rtol 1e-6 (restart 15).

    | arm | patch width | worst `\|A_p⁻¹\|` | restart cycles | true relative residual |
    |---|---|---|---|---|
    | self-check, the march's own solver | — | — | 1 | 2.0e-06 |
    | shipped (ILU(0) ×4) | — | — | **2** | 1.3e-14 |
    | ILU(0) ×8 | — | — | 1 | 1.4e-14 |
    | ILU(0) ×16 | — | — | 1 | 1.4e-14 |
    | coarse solve degraded to Jacobi | — | — | 2 | 1.3e-14 |
    | Vanka, 0 neighbours (the cell block) | 6 | 3.011e3 | 58 (cap) | 7.8e-01 |
    | Vanka, 6 velocity neighbours | 24 | 3.006e3 | 58 (cap) | 3.6e-01 |
    | Vanka, 6 neighbours, all fields | 42 | 3.242e3 | 58 (cap) | 2.4e-01 |
    | Vanka, 6 velocity neighbours, damped 0.3 | 24 | 3.006e3 | 58 (cap) | 4.1e-01 |

    Two things follow, and the second is the bigger one.
    - **Vanka does not stall here — it fails outright, at a state where the operator is easy**, and it
      fails for a reason that has nothing to do with the coarse space. It cannot be excused by a hard
      operator: ILU(0) solves the same system to 1.3e-14 in two cycles. Every patch width runs to the
      restart cap; widening helps monotonically (0.78 → 0.36 → 0.24) but nowhere near enough, and
      **the worst patch gain is flat at ~3e3 across all three widths** — on an operator equilibrated
      to unit diagonals, i.e. a local singular value four to five orders down. Flat under widening is
      the informative part: no amount of surrounding a cell repairs it, so the degeneracy is in a row
      of the **centre block** that stays degenerate however large the patch. (Adding the neighbours'
      `k` and `ω` — the "all fields" arm — did not reduce it either; it rose slightly.)
      That the *narrowest* arm is plain point-block Jacobi also settles the "is the implementation
      wrong" question the right way: block-Jacobi failing on a saddle point is textbook, and is
      precisely why patch smoothers exist. **Why ILU(0) succeeds where every patch method fails:** in
      cell-major ordering it is a *global* forward/backward sweep and never inverts a cell block in
      isolation, which is exactly what a patch method is obliged to do.
      **⚠️ It STAGNATES, it does not amplify — and that distinction is not decoration, it points at a
      different cause.** Every arm stops at a finite true residual and runs to the restart cap; none
      diverges. Under-relaxing (damping 0.3), which is what tames an over-correcting smoother, made it
      **worse** (0.36 → 0.41), as it would for a smoother that is too *weak* rather than explosive.
      So the large patch gain is at present a **correlation with the failure, not a demonstrated
      cause** — do not write it up as the mechanism. Two explanations survive the data equally well:
      1. a few near-singular cell blocks poison the recombination; or
      2. the **additive** form is simply too weak here. Weighted-additive Schwarz is block-Jacobi-like
         and needs a relaxation that may not exist for this operator, which predicts exactly what is
         seen: monotone gains with patch width, stagnation at every width, no help from damping. The
         classical Vanka sweep is *multiplicative* and remains **untested**.

      **WHICH row is degenerate — measured, and it is ω.** A batched singular-value decomposition of
      all 23040 diagonal 6×6 blocks of the equilibrated cell-major operator at the same state:
      `σ_min` median 2.87e-2, 1st percentile 8.42e-4, minimum 3.21e-4, with **353 cells below 1e-3**
      and condition numbers there of 5e6–9e6. In every one of the twenty worst blocks the near-null
      right singular vector is **pure ω** — `(u,v,w,p,k,ω) = (0,0,0,0,0,1)` to three decimals. So it
      is the ω *column* that is nearly empty: perturbing ω in such a cell barely changes any of that
      cell's own six equations, because ω's influence there is carried almost entirely by neighbour
      transport rather than locally. That is exactly the quantity a cell-centred patch must invert and
      a global cell-major incomplete-LU sweep never does, which is the cleanest available explanation
      for why every patch smoother fails on this operator while ILU(0) is untroubled. The affected
      cells sit in low-`k` regions (median `k` 0.122 against 0.655 over the mesh) and occur in exact
      spanwise-symmetric pairs, so they are a coherent region of the flow, not scattered noise.
      **⚠️ BUT THE SHIFT DOES NOT CONTROL IT, so this does NOT explain the low-β wall.** Sweeping the
      shift over a 500× range at the same state (one materialization; the Jacobian does not depend on
      β, only the added diagonal does):

      | shift | median `σ_min` | min | cells below 1e-3 |
      |---|---|---|---|
      | 0.5 | 3.450e-2 | 3.929e-4 | 204 |
      | 0.05 (the floor) | 2.873e-2 | 3.214e-4 | 353 |
      | 0.005 | 2.807e-2 | 3.133e-4 | 367 |
      | 0 (unshifted) | 2.799e-2 | 3.124e-4 | 369 |

      Over the range the march's tail actually occupies (0.05 → 0.005) the near-singular count moves
      **4 %**. The degeneracy is a fixed property of the discretization and state, not something β
      governs — so the low-shift conditioning wall is *something else*, and this is not the mechanistic
      account of it that it first looked like.
      **It also kills the obvious lever before anyone builds it.** An ω-only, preconditioner-only shift
      boost cannot work, because the shift is `β·d` and for ω that `d` **is the weak transport coupling
      that is the problem** — scaling a near-zero diagonal by β leaves it near-zero at any β. Anything
      along these lines would have to be an *absolute* floor on the preconditioner's ω diagonal rather
      than a multiple of `d`, which is untested speculation and has no headroom to be demonstrated at a
      state where ILU(0) already converges in two cycles.
      **What the ω finding does license** is narrow and solid: it explains why *cell-local*
      preconditioners fail here, and it gives a two-minute screen for the next one.

      **It also independently corroborates the ω bullet below**, which was recorded without its
      configuration and marked unusable: a completely different measurement — local block
      conditioning rather than per-field V-cycle smoothing rates — lands on the same field. Two
      unrelated routes to "ω is the 3D preconditioner lever" is much better evidence than either
      alone, and it also suggests a concrete arm: **leave ω out of the patch**, since a local solve
      cannot resolve a direction the local block does not see.

      ⚠️ **`VankaSmoother` / `VankaPC` / `CellStarPatches` NO LONGER EXIST — the module was deleted
      2026-08-15 because it never won an arm. If you grepped your way here for one of those names,
      that is why there is no such symbol; the code is in git history. The measurements below stand
      as the reason it went, and cannot be re-run without restoring it.**

      **RESOLVED, and (1) IS REFUTED — the near-singular blocks are NOT the mechanism.** The test was
      pre-registered (`VankaSmoother(max_patch_gain=...)`: converges ⇒ (1), stagnates ⇒ (2)) and the
      answer is unambiguous, because dropping precisely the near-singular patches made the solve
      **worse**, not merely no better. The whole family of drop-arms is monotone in *coverage* and in
      nothing else:

      | patches dropped | fraction of the mesh | true relative residual |
      |---|---|---|
      | 0 | — | **0.360** |
      | 337 (gain > 1e3 — the σ_min < 1e-3 set) | 1.5 % | 0.737 |
      | 1925 (gain > 3e2) | 8.4 % | 0.9975 |
      | 6044 (gain > 1e2) | 26 % | 0.9995 |

      Strictly monotone in how much of the mesh still gets relaxed, which is the tell: these arms
      measure **coverage** and nothing else.

      Those 337 patches were doing *useful* work; removing them leaves their degrees of freedom
      unrelaxed and costs more than their ill-conditioning ever did. So the large patch gains are a
      real property of this operator — and worth knowing, since two independent measurements agree on
      the set (337 patches by inverse gain, 353 cells by block `σ_min`) — but they are **not** why the
      smoother fails. **Explanation (2) is what survives: the ADDITIVE recombination is the limit.** It
      smooths usefully but insufficiently *everywhere* rather than being poisoned anywhere, which is
      also what the width ladder and the damping arm independently say.
      **MULTIPLICATIVE IS NOW TESTED TOO, AND IT IS WORSE — so (2) falls as well.** Compared
      sweep-for-sweep, which is the only fair way to ask whether *sequencing* helps (one multiplicative
      sweep against four additive ones confounds recombination with sweep count): **additive ×1 →
      0.497, multiplicative ×1 → 0.855.** Sequencing the patches does not rescue this smoother; it
      costs. Built as `VankaSmoother(multiplicative=True)`, 16 colours on this mesh, ~33× the additive
      apply cost.

      **So neither hypothesis stands, and what is left is the one fact that survived every arm: the
      cell block is weakly coupled in ω EVERYWHERE, and the 353 near-singular cells are only its tail.**
      Median `σ_min` is 2.9e-2 on an operator equilibrated to unit diagonals — some 34× down — so
      *every* cell-local solve mishandles ω, not just the extremes. That single fact accounts for the
      whole ladder: widening the patch adds neighbour velocities and cannot help ω; dropping the worst
      patches removes useful work without touching the general weakness; damping and sequencing change
      only how corrections are combined, and no recombination repairs a local solve that cannot see the
      field. It equally explains why ILU(0) is untroubled — a global cell-major sweep propagates ω along
      the transport direction, which is where ω's coupling actually lives.

      **Verdict: cell-centred patch relaxation is the wrong shape for this operator, and the reason is
      structural rather than tunable.** Do not re-open it with another patch variant. `VankaSmoother`
      stays in the tree as the evidence and as a testbed; `validation/bfs3d_openfoam/cell_block_conditioning.py`
      screens any future cell-local proposal in ~2 minutes, which is what this campaign cost hours to
      learn.

      **Choose that cap from the gain distribution, not from the maximum**, or the arm measures the
      wrong thing. Measured over the 23040 width-24 patches at this state: gain above **1e1 in 96.6 %**
      of them (22251), above **1e2 in 26.2 %** (6044), maximum 3.0e3. A gain of order ten to a hundred
      is therefore *ordinary* here — it is what `1/σ_min` gives for the median block (`σ_min` ≈ 2.9e-2)
      — and only the ~1.5 % above 1e3 are the near-singular ω blocks the hypothesis is about. Capping
      at 1e2 drops a quarter of the mesh and the true residual goes to **0.9995**, i.e. no reduction at
      all: with that many patches gone, any degree of freedom covered only by them has weight zero and
      is never relaxed. That number says the smoother was gutted, and nothing whatever about whether
      near-singular patches caused the original failure — a confounded arm, not a null result.

      Related and worth keeping either way: this is the same *family* of failure as the two earlier
      patch-smoother attempts (unweighted additive Vanka, "ρ = 9e4"; undamped block-ILU/inexact Uzawa,
      "1-apply reduction 5.10"), and the overlap weighting that was supposed to fix it does not. Note
      also that the published algebraic Vanka does *not* solve the patch exactly: Metsch's (§4.6) local
      solve is an inexact-Uzawa form built on a diagonal `Â > A` with a scaling `β` chosen so
      `Ŝ > C + BÂ⁻¹Bᵀ`, provably convergent precisely because it never inverts a near-singular local
      saddle. The exact patch solve chosen here as the *stronger* option may be the thing that breaks.
    - **⚠️ THE STATE-SELECTION PREMISE IS BROKEN FOR A DUAL-TIME MARCH, and this invalidates the
      comparison above as a test of the *coarse-space* question.** A checkpoint is written at the end
      of a step, so it holds the state the *next* step starts from — and a step's first solve is its
      easy one, from a settled state with a freshly rebuilt preconditioner. Across the whole march:
      **all 70 step-initial solves cost ≤ 2 restart cycles, while solves at inner > 0 reached 15.**
      No checkpointed state in this march poses a hard linear system, so every arm ties there and the
      sweep says nothing about smoother-versus-coarse-space. (It still says plenty about Vanka, which
      *failed* at an easy state — a positive result needs a hard state, a failure does not.) The fix
      is `DualTimeStep.inner_observer`, which now also carries the **iterate**.
      **⚠️ But a march step CANNOT be reconstructed from a checkpoint — its configuration is
      path-dependent.** `validation/bfs3d_openfoam/inner_iterate_probe.py` tried, driving one step from
      the checkpoint, and both plausible arrangements bracket the march without reaching it. At
      `state-00049`, β = 0.0293, where the march's first inner solve costs **1 cycle at α = 0.500**:

      | how the engine was built | inner-0 cycles | ‖G‖ |
      |---|---|---|
      | at the probed state (self-consistent) | 7 | descends cleanly, α = 1 |
      | at the Reynolds rung's seed, 11 steps back | **39** | **no descent at all** |

      The march is outside both because `amg_beta_tracking_refresh` rebuilds the preconditioner
      repeatedly on the way to the step, so by then it is recent, while the shift policy dates from the
      seed with its transport part rebuilt at each refresh — a product of the refresh *history* that no
      checkpoint records. **The route to the hard iterates is therefore to capture them DURING a
      march**, via the observer, not to replay a step afterwards.
      **✅ DONE, and it settles the question: the march's expensive solves are STALENESS, not hard
      operators.** `InnerIterateCheckpointer` (`solve/checkpoint.py`, wired in the case behind
      `BFS3D_INNER_DUMP_ABOVE`) caught the seven solves that reached ≥4 restart cycles on an otherwise
      byte-identical march (`x_r/h` 8.361, 290 cycles, unchanged). Note the replay problem does **not**
      apply once you hold the iterate: the state is on disk, so the operator can be rebuilt around it
      directly. At the march's single hardest solve — attempt 50 inner 3, β = 0.0293, where the march
      took **15 cycles** and the line search collapsed to α = 0:

      | iterate | β | the march | one step stale | matched at the iterate |
      |---|---|---|---|---|
      | attempt 50 inner 3, α → 0 | 0.0293 | **15** | 3 (5.9e-02) | **1 (6.6e-06)** |
      | attempt 40 inner 3, α = 1 | 0.3333 | **8** | 1 (9.9e-05) | 1 (1.7e-10) |

      Read the two rows together: at β = 0.33 even a stale preconditioner is already optimal, so
      **staleness only bites at low β** — and there it bites hard. Note also that both march counts far
      exceed what *one* step of staleness costs (15 against 3, 8 against 1), so the march's real
      staleness — up to four steps in `J` — is worth several times a single step.

      **That operator is easy.** One cycle to 6.6e-06 with a matched preconditioner, and a single step
      of staleness already triples the cost and gives up four orders of accuracy. So there is **no
      preconditioner headroom left on this case at any state the march visits** — which is the
      retrospective explanation for why every arm in the Vanka campaign tied or lost, and why the
      aggregation sweep only separated at all under the *older*, weaker bundle. The lever here is
      **refresh cadence** (`refresh_every=8`, `materialize_every=4`, `beta_rel_change=0.25` are lax
      around the hard steps), consistent with the earlier gate fix, which bought 132 fewer cycles and
      removed three of five retry cascades by refreshing ~50 % more often — the same mechanism found
      from the other end.
      **✅ THE COST TRIGGER IS BUILT — `amg_beta_tracking_refresh(refresh_on_cycles=N)`.** A solve that
      reaches `N` restart cycles refreshes the preconditioner **at the iterate it was handed** and the
      inner loop carries on, rather than aborting the step and escalating β (which discards both the
      work and the pseudo-timestep). Capped at **one refresh per step**, without which a genuinely hard
      operator would refresh on every inner iteration; a new inner loop re-arms it. Off by default
      (`None`), so a march that does not opt in is byte-identical. Pinned by
      `test_inner_refresh_fires_on_an_expensive_solve_at_that_iterate_and_once_per_step`.

      **Measured end to end on the 3D coupled backward-facing step, against the scheduled cadence:**

      | | scheduled | on cost |
      |---|---|---|
      | wall | 3632 s | **3140 s (−14 %)** |
      | refresh | 758 s | **310 s (−59 %)**, 62 refreshes → **19** |
      | Krylov cycles | 290 | 293 (**unchanged**) |
      | steps with α = 0 | 5 | **0** |
      | `x_r/h` | 8.361 | 8.361 (unchanged) |

      **Read the cycle row: this is not a better-conditioned solve.** Cycles are flat, so the whole
      saving is refresh no longer spent maintaining freshness nothing consumed, plus the elimination of
      five dead steps — those where the line search collapsed, the residual froze or rose, and the shift
      escalated twenty-fold before recovering. That the *same* change removes both is the point: an
      inaccurate direction from a stale preconditioner is what α → 0 was, so the retries were downstream
      of the refresh cadence rather than a separate globalization problem.

      **Do not read the arithmetic below as an open proposal — it is the projection this shipped
      against**, and it came out close on the total (−492 s measured vs −515 s predicted) while missing
      the mechanism: it predicted the saving would come from refresh alone, and half of it came from the
      dead steps. The cost arithmetic, measured on the earlier 3501 s
      march: scheduled refreshes are 50 full + 12 shift = **742 s, 21 % of the wall**. A refresh fired
      only when a solve reaches ≥3 cycles would fire **16** times (227 s, **−515 s**); at ≥4, **7**
      times (99 s, **−643 s**) — a 15–18 % whole-march saving, against a ~8 % ceiling for a *perfect*
      preconditioner. Read as an *addition* to the schedule the trigger looks break-even; as a
      **replacement** it is the biggest win on the table, and conflating the two is easy to do.
      **Why no schedule can work here:** 193 of the 232 solves already take 1 cycle, so most scheduled
      refreshes maintain a freshness nothing consumes — and the right interval is regime-dependent (at
      β = 0.333 a one-step-stale preconditioner still gives 1 cycle; at β = 0.029 it gives 3), so a
      fixed cadence necessarily over-refreshes in the easy regime and under-refreshes in the hard one.
      Targeting cannot be predictive either — the Step-0 diagnostic refuted every static signal — which
      leaves reacting to the cost itself.
      **Two things to get right.** The counts above come from a march that *had* the schedule keeping it
      fresh, so removing it shifts the distribution up and the trigger fires more often; the equilibrium
      rate is a feedback loop (staleness → cycles → trigger → freshness) and is unknown until it is run.
      And it needs a **cap of one refresh per step**: with it the worst case is 69 × 14.2 = 980 s against
      today's 742 s, a bounded +238 s downside for a ~515 s upside; without it, a refresh per inner
      iteration is unbounded. `abort_above_inner_cycles` already detects the condition inside the inner
      loop, so the change is to the *reaction* — refresh and continue, keeping β escalation as the
      fallback, which also makes the refresh a diagnostic: if it does not help, the operator really is
      hard.
      **The coloured probe dominates a refresh, and the obvious way to halve it — stencil reach 3 → 2 — is
      MEASURED AND FAILS.** (Its exact share was recorded with no state or mesh and is deleted.) The saving is
      real (112 → 60 colours) but the V-cycle is
      not: at `state-00062`, β = 0.0072 against the 0.05 floor, reach-2 gives **41 cycles at a true
      relative residual of 1.9** — worse than the initial guess — where the shipped reach-3 arm reaches
      1.5e-10 in six. A failure at a *benign* state is conclusive: it cannot be excused by a hard
      operator. Note this also disposes of an appealing argument that does **not** work: the older
      refutation blamed "the pattern-dependent ILU(1) smoother", and at ILU(0) there is no fill to be
      pattern-dependent about — yet reach-2 still fails at ILU(0), so the conclusion outlived the
      mechanism that was offered for it. **Retested under plain aggregation too — it still fails, so the lever is
      CLOSED.** At `state-00049`, β = 0.0293 (where reach-3 does 2 cycles to 1.3e-14): reach-2 plain
      gives **58 cycles at 1.7e-01**, and doubling the smoother sweeps barely moves it (1.4e-01, still
      at the restart cap). Note what that rules out: it is not a smoothing deficit. And the hierarchy
      itself changes — reach-2 yields **3 levels / 480 coarse equations** against reach-3's 2 / 1296 —
      so the dropped couplings are load-bearing for the *coarse space*, not merely for the smoother's
      fill. Reach 3 is required, now measured across both aggregations at ILU(0). **The reach-3 PATTERN
      is required; probing every COLUMN at reach 3 is not, and that distinction was missed here.**
      This entry shrinks the pattern, which is what moves the hierarchy. Keeping the pattern at reach 3
      and shortening only the *columns* (`column_reach`) leaves the coarse space and the smoother
      untouched and is worth 564 → 454 probes at the shipped `(3,3,3,3,2,2)` — but it is **not** the cheap
      win it was recorded as: it aliases each shortened column's far couplings onto its near entries, over
      half the entries of every shortened column on `bfs3d`, and shortening `p` as well
      (`(3,3,3,2,2,2)`) diverges the march at step 1.
      See *"per-column probing reach"* under `sparse_jacobian.py` above before reaching for it. So this
      arm's "there is no cheap way to shrink the probe" stands for the *pattern*, and the column
      variant that looked like the exception has not yet been shown safe on this case.
      **A trap: the cheap `refresh_shift_in_place` branch does NOT help within a step.** The shift is
      formed once per step at the reference state and held fixed across the inner loop, so what drifts
      inside a step is `J(p)`. The shift-only branch only helps *across* steps, where β moves.
      **Untested but cheap and worth doing: whether staleness also drives the GLOBALIZATION cost.** A
      stale preconditioner returns a less accurate Newton direction (5.9e-02 against 6.6e-06 here), and
      an inaccurate direction can fail to descend — which is what α → 0 means. If so, the retries and
      line-search collapses are downstream of the same cause, not a separate problem. The captured
      iterate is all that is needed to check it. Two details worth keeping from the
      attempt: the builder's `amg_beta` defaults to **2.0**, which at a sub-floor β is a two-orders
      mismatch that alone turns a 1-cycle solve into 7 (pass `max(β, beta_floor)`); and a
      preconditioner frozen 11 steps back yields **no descent whatsoever** — α reads 1.000 because
      that is the line search's non-descent fallback, with ‖G‖ flat — which is independent evidence
      that the refresh is load-bearing.

      The probe now **validates itself against `march.log`** (inner-0 cycles and α) and refuses to
      report if it disagrees, which is how both errors above were caught rather than written up. Hold
      any future march-reproduction harness to the same gate.

    **How to re-run it — the smoother and the harness are BUILT (`aquaflux/solve/vanka.py`,
    `validation/bfs3d_openfoam/preconditioner_sweep.py`); what is missing is the measurement.** The
    discriminating question is narrow: **with plain aggregation, does a Vanka smoother still stall?**
    If yes, the coarse space really is the wall and the inf-sup / block-Schur direction is justified.
    If no, the original verdict was an artifact of the aggregation default and the smoother direction
    reopens. Record the smoother, aggregation, state and shift pairing alongside whichever answer
    comes out. The `ARMS` ladder in the harness reads the question two independent ways — a **sweeps**
    ladder on the *shipped* smoother (if cycles keep falling as sweeps rise the smoother is not
    saturated and cannot be the binding constraint; if they plateau, what survives is in the coarse
    space's blind spot) and the **Vanka** arms themselves, the widest deliberately over-strong.
  - **The Jacobian's fill is irreducible.** The coupled `(u,p)` Jacobian is intrinsically **distance-2**
    (~38 nnz/row) because Rhie–Chow damping couples pressure to the neighbour-of-neighbour ring; the
    advection scheme is irrelevant to this. A distance-1 preconditioner pattern is not available for a
    second-order collocated Rhie–Chow discretization, so "make the PC pattern local" is not a lever.


## Flow block (native `[u, v, w, p]` saddle preconditioning) — refuted attempts

Full detail for all of these is in `solve-flow-block-log.md` (no `paths:`, reference-only); the current,
load-bearing status of the flow block itself is in `solve-flow-block.md`.

- **Flat block preconditioners (no hierarchy, no coarse-grid correction) — CLOSED on this case.** Every
  arm tried loses to a hierarchy. See `solve-flow-block-log.md` § "FLAT block preconditioners are CLOSED
  on this case".
- **Smoothed aggregation on the flow saddle, under the SIMPLE smoother — REFUTED.** See
  `solve-flow-block-log.md` § "Smoothed aggregation on the flow saddle — REFUTED under the SIMPLE
  smoother".
- **A deeper Galerkin-coarsened damped-Jacobi convection hierarchy — dominated on both ends, removed.**
  See `solve-amg-multigrid.md`'s "damped-Jacobi convection hierarchy is TWO-LEVEL by design" binding
  decision — a coarse-of-coarse convection operator acquires eigenvalues no single-factor smoother can
  damp; deep convection coarsening is lAIR's job instead.
- **Block-CSR sparse matvec — REFUTED; JAX's own primitives are already at the limit for this case.** See
  `solve-flow-block-log.md` § "The sparse matvec is at the limit of what JAX's primitives give —
  block-CSR REFUTED".
- **The native flow block winning on a real march (on SPEED) — NOT established, and one earlier "10%
  fewer cycles" reading was an unfair A/B (only one arm had received a since-shipped tuning change).**
  ⚠️ The speed verdict stands, but `FLOW_INVERSE` is nonetheless `bfs3d`'s DEFAULT as of 2026-08-18 — the
  case for it moved to robustness (incomplete-LU's elimination-order sensitivity) and GPU-readiness, not
  a reversal of this measurement. The full current verdict is in `solve-flow-block.md` itself, not
  archived here, because it is the live default, not a dead idea. Read it before re-proposing the native
  block as a *speed* win — that specific claim is still refuted.

## Globalization (forward step, continuation, line search) — closed investigations

Full detail is in `solve-globalization-log.md` (no `paths:`, reference-only); the current architecture
and binding defaults are in `solve-globalization.md`.

- **The `growth` line-search parameter causing a march to hang — NOT a performance regression; settled
  2026-07-25, do not re-open on the original evidence.** Both the traced-bound hypothesis and the
  attribution to `growth` were wrong (`admissible` is computed outside the loop and the default bound is
  a concrete, not traced, value). See `solve-globalization-log.md` § "The `growth` parameter is NOT a
  performance regression".
- **`descent_backoff` — COUNTERPRODUCTIVE on this case; measured, do not enable it blindly.** See
  `solve-globalization-log.md` § "`descent_backoff` IS COUNTERPRODUCTIVE ON THIS CASE".
- **The convective `ShiftBasis` (`w = 0`) — DOMINATED by the default `a_P` basis; settled by a controlled
  2×2 plus a β sweep, do not re-open on a %/s sweep.** See `solve-globalization-log.md` § "The convective
  basis (`w = 0`) is DOMINATED".
- **The equilibrated residual measure barely falling where the Euclidean norm looks better — the
  Euclidean norm was found to MIS-RANK states (a converged field can score worse than a badly wrong one),
  which is the larger and more load-bearing finding here.** See `solve-globalization-log.md` §
  "THE EQUILIBRATED MEASURE BARELY FALLS ON A MARCH THE EUCLIDEAN NORM LOVES" and the "EUCLIDEAN ‖R‖
  MIS-RANKS STATES" entry immediately after it.
