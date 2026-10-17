# Investigation log — `aquaflux/precondition/` the traced multigrid (and the monolithic AMG's live-code entries)

> Split out of `.claude/rules/solve-amg-multigrid.md` (2026-10-17). **Lives in `.claude/notes/`, outside
> the auto-loaded `.claude/rules/` tree, so it never auto-loads.** It holds the dated investigation behind
> the traced multigrid (`precondition/multigrid.py`, `precondition/hierarchy_inverse.py`): the
> measurements, probe results and retracted or qualified rounds, moved verbatim under headings that mirror
> where they sat. The record of the deleted monolithic PETSc AMG was deleted rather than moved, except for
> the entries below that concern live code; its refutations are in `solve-refuted-directions.md`. See
> `solve-amg-multigrid.md` for the current, load-bearing status.
>
> ⚠️ **Three things these entries describe no longer exist:** the global `k` positivity cap
> (`positive_block_limit` / `positive_k_limit`, `PositiveBlockLimit` and its `floor`,
> `PositiveBlockProjection` / `positivity_projection`, `stop_on_limit_stall`, the step-limit dumps —
> deleted 2026-10-14; every coupled step now clips `k` per cell with an ω step bound), the monolithic
> PETSc V-cycle and every `petsc` arm (2026-10-10), and `Ilu0` / `hostilu` with the cell orderings
> (2026-09-13, #371). Entries naming them are dated history.

## Preconditioner — monolithic AMG (DELETED 2026-10-10) — entries about live code

### The SIMPLE-smoothed hierarchy is reach-insensitive (2026-08-16)

**✅ AND THE SIMPLE-SMOOTHED HIERARCHY ESCAPES BOTH THE FILL AND THE REACH — the fastest arm on
`pitzDaily`, at HALF the Jacobian (2026-08-16).** Same case, same everything but the leading
inverse and the probe's reach; three Reynolds rungs; `x_r/h` identical to four decimals in every
converging arm, so this is a cost comparison and not an accuracy one:

| leading inverse | reach 3 | reach 5 |
|---|---|---|
| PETSc **ILU(1)** | **fails** (300 matvecs, true 3.36) | converges — 74 steps, 321 cycles, **628 s** |
| PETSc **ILU(0)** | fails | fails (α 0.000, NaN by step 4) |
| our **ILU(0)** (`hostilu`, deleted 2026-09-13, #371) | fails | fails (α 0.000, inf by step 3) |
| **SIMPLE-smoothed** | **converges — 71 steps, 395 cycles, 550 s** | converges — 71, 408, 799 s |

**SIMPLE is REACH-INSENSITIVE, and that is the load-bearing observation.** Between reach 3 and
reach 5 it takes the **same 71 steps**, cycles within 3 % (395 against 408) and a final residual
identical to four figures (5.095e-06 against 5.097e-06) — while the Jacobian halves (6.3M nonzeros
against 13.3M) and the march is **31 % shorter**. It simply absorbs the ~2e-07 perturbation that a
short reach leaves in the pressure column.

**The mechanism is which preconditioners inherit the stored SPARSITY.** An incomplete
factorization takes its pattern from it, so a corrupted pattern gives a corrupted factor and
ILU(1) *requires* reach 5 here. SIMPLE relaxes through diagonal and Schur approximations and takes
no pattern at all, so the same corruption is merely a slightly wrong operator. **Shortening the
reach is free for SIMPLE and fatal for an ILU** — which is the argument for the reach as a
preconditioner-family choice rather than a case constant.

At reach 3 SIMPLE also beats the incumbent outright, 550 s against 628 s (12 %).

⚠️ **One run per arm, and this case has no measured march-level noise floor.** The step counts and
cycle counts are deterministic and carry the result; the 12 % wall margin over PETSc is the softer
number. ⚠️ **And SIMPLE's own settings are still the sibling case's** (`strength_threshold` 0.25,
2 sweeps, 5 levels) on a mesh that coarsens about 3× per level where that one manages 24×, so the
arm is not at its best here — `PITZ_FLOW_SWEEPS` and the threshold are unexplored.

### Narrowing the probe's gradient sweeps (2026-08-17)

**⚠️ Narrowing the probe's gradient sweeps helps a COMPLETE factorization and hurts a ZERO-FILL one —
opposite signs, same knob.** At `probe_gradient_sweeps=1` the LU's factor cost falls 4.1× (72.7 → 17.9 s)
on an exactly-probed matrix, while an ILU(0) pivot census on the equilibrated leading block goes
**9 → 34 → 263** negative pivots at full / 2 / 1 sweeps, minimum |pivot| falling `2.18e-02 → 3.59e-04`.
⚠️ And measured against the **true** Jacobian rather than the narrowed one, `sweeps=1` at reach 3 is
`1.07e-03` away where the *aliased* full-sweep probe at the same reach is only `1.99e-07` — so
narrowing trades a small structured corruption for a larger smooth approximation on this mesh. It is a
preconditioner-family-dependent knob, not a free win: right where sweeps are inert (`bfs3d`, skew-free),
questionable where they are live (`pitzDaily`).

### The cost guard and the shift escalation — re-march and reasoning (2026-08-17)

**✅ `bfs3d` RE-MARCHED UNDER THE SPLIT, AND IT IS UNCHANGED TO EVERY RECORDED FIGURE (2026-08-17).**
Shipped defaults, `hostilu`, `Ilu0` compiled, `abort_above_cycles` 10, `cycle_budget` 42; log
`bfs3d_openfoam/march.log`:

| | recorded (pre-split) | re-marched (post-split) |
|---|---|---|
| steps | 61 | **61** |
| Krylov cycles | 208 | **208** |
| final ‖R‖ | 1.858e-06 | **1.858e-06** |
| mid-span / full-span `x_r/h` | 8.361 / 12.53 | **8.3611 / 12.53** |

**And the reason it is unchanged is the useful part: this case never takes the cost path at all.** Its
two retries are both `alpha`, so there was no `cycles` escalation for the split to remove. The
behaviour change is therefore confined to configurations where a solve actually exceeds the budget
without reaching target — on the evidence so far, `pitzDaily` under the zero-fill smoother, which is
where the defect was found. ⚠️ Wall was 1408 s against a recorded 1246 s, which is the soft number: one
run each, different machine conditions, and the recorded figure has no kernel provenance (this one is
logged as compiled).

**The reasoning that produced this, kept because it generalizes.** The problem was that
the cost threshold is an absolute count and therefore **arm-dependent**, which conflicts with this
project's knob-free-robustness goal: a cycle count is `preconditioner strength × operator difficulty`,
so **any constant encodes an assumption about which preconditioner is installed.** Four directions were
written down; (4) is what shipped, and the other three remain available if the abort threshold itself
proves to need calibrating:

1. **Make the trigger relative to the arm's own history** — an anomaly test against a running median of
   recent accepted solves (say 3×) rather than against a constant. Self-calibrating; needs a cold-start
   rule for the first few steps, where there is no history and the march is at its most fragile.
2. **Trigger on the cycle BUDGET instead.** A solve that exhausts `cycle_budget` was truncated and its
   direction is genuinely suspect; one that finished under it converged. That is a resource decision
   rather than a diagnosis, so it does not need per-arm calibration — but it is blunt, firing only at
   the extreme.
3. **Make the criterion dimensionless: cost per unit residual reduction**, not cost alone. The rule
   wants "this step is going badly", and many cycles that buy a large drop is not that. Something like
   `cycles / log(‖G_in‖/‖G_out‖)` measures what is actually wanted; the present rule reads the numerator
   only.
4. **✅ SHIPPED — a COST signal does not drive the β ladder at all.**
   A high cycle count is evidence about the **preconditioner** (stale factorization, weak smoother), not
   about the **step's stiffness**, and this case already routes it to the right responder:
   `refresh_on_cycles = 3` rebuilds the frozen preconditioner on exactly this signal. Sending the same
   observation to β-escalation as well conflates two diagnoses — and on this case the escalation is not
   even monotone in the right direction. The honest stiffness signal is **α**, which is dimensionless
   and needs no calibration.

**⚠️ AND `on_cycles` DID TWO JOBS — the discovery that shaped the fix.** Besides triggering the
β-escalating retry, `RetryPolicy.thresholded_step` pushes it down as `abort_above_inner_cycles`, which
**cuts the dual-time inner loop short** the moment a solve exceeds it (`retry.py`). It does not truncate
the Krylov solve — the log shows the over-threshold solves running their full 12 cycles — it stops the
step taking further inner iterations. So on the failing run, step 29 ran 3 inners and was discarded;
on the converging one the identical first two inners (`1.663e-01 → 7.841e-03 → 3.865e-03`, bit-identical)
were followed by a 4th and the step was accepted. The threshold therefore withheld *convergence the step
was in the middle of achieving* and then escalated β for the trouble.

**There is weak evidence for (4) already.** In the converging run no solve reached 25 cycles, so the
trigger **never fired** — that run is observationally what "no cycles trigger for this arm" looks like,
and it converged on `alpha` retries alone. ⚠️ Weak because it is one arm on one case. And removing the
trigger outright would **not** have been a pure subtraction — it would have taken the early abort with
it, which `cycle_budget` depends on. That is why what shipped splits the two rather than deleting one.

### Where a refresh's time went (monolithic AMG march)

- **Where a refresh's time now goes (whole-run aggregate, same march): probe 632 s, refactor 195 s,
  assemble 28 s, other 12 s — the probe is 73 %.** The non-probe tail is close to floor, so any
  further work on refresh cost has to attack the coloured probe itself (amortizing colours across
  steps, or a cheaper stencil), not the assembly or the multigrid setup.

## The traced multigrid — in progress

- **⚠️ IN PROGRESS (2026-08-10): the traced trailing V-cycle now MATCHES PETSc on quality AND cost;
  what blocks it is the POSITIVITY LIMITER — read that as the PROXIMATE failure, not the cause.**
  ⚠️ The same limiter, defaults and case run fine under the PETSc ILU(0) control (58 steps to 9.588e-06,
  recorded below), so the limiter is necessary and not sufficient: the traced arm dies at a state the
  control never reaches, and what drives that arm's direction into the boundary is **unexplained**. Read this before the
  2026-08-09 section below, which it supersedes in several places.

  **Where it stands. Configuration, stated here rather than 500 lines away, because the sweep count
  is load-bearing:** `bfs3d` `state-00057`, PC β 0.05, the `[k, ω]` block **ALONE** (46080 dofs,
  4.20M nnz), GMRES restart 15 to rtol 1e-8 on the TRUE residual, **4 smoother sweeps**, the PETSc
  side on its **matched-smoother** arm (plain aggregation, point-block Jacobi ×4 — against ILU(0) ×4
  PETSc does 1 cycle, not 2); harness `trailing_hierarchy_sweep.py`. On that arm the traced nodal
  hierarchy reaches **2 restart cycles against PETSc GAMG's 2**, on a 438-equation coarse space
  against 432. Quality parity, on CPU. Per-apply cost came out at parity too, but that pair of
  timings was recorded with no sweep count and no state and is deleted — measured 2026-08-10,
  configuration not recorded, re-measure before relying on it. The full march converges
  rung 1 (14 steps, 43 cycles, 304 s against the PETSc control's 14 / 45 / 359) and then dies on
  rung 2.


  ⚠️ **Everything from here to the `k`-trough entries below about the k positivity CAP, its `floor` and the
  `positivity_projection` switch is dated history: all three were deleted 2026-10-14** (the global cap
  `positive_block_limit`/`positive_k_limit` let one or two near-zero-k corner cells throttle the whole
  step). Every coupled step now clips `k` per cell and bounds ω's per-cell change (`omega_step_ratio`);
  see `solve-march.md` and `.claude/notes/solve-globalization-log.md`, "bfs3d's retried steps: the global
  k cap, replaced by a per-cell clip with an omega bound (2026-10-14)".

  **⚠️ THE α COLLAPSE IN THESE MARCHES IS THE POSITIVITY LIMITER, AND THE STEP TABLE SAYS SO — READ
  `limit` BEFORE ATTRIBUTING ANYTHING TO THE PRECONDITIONER.** `a_min` and the `limit` aside are the
  **same number** wherever both appear (0.579/5.79e-01, 0.651/6.51e-01, 0.004/3.76e-03, …). Those
  steps are not failing to descend; they are being allowed almost no movement because `k` would go
  negative. Two consequences, both of which cost hours today:
  - **A capped step is not a bad step.** Two arms differed in α at rung-2 step 16 (1.000 against 0.579)
    and reached an **identical** residual, 1.271e-01. Attributing the α difference to preconditioner
    quality was wrong. ⚠️ **Those two numbers are NOT a raw-vs-equilibrated pair** — they are `march.log`
    (**petsc** trailing inverse) against `march-20260810-223702.log` (**jacobi + `equilibrate=True`**),
    both under the `dirichlet` k wall BC. The genuine `equilibrate` A/B at step 16 is **0.699 against
    1.000**, under `zerogradient` (`march-20260810-221936.log` / `march-20260811-003915.log`, both at
    ‖R‖ 1.382e-01). The point about a capped step survives either way; the labelling did not.
  - **⚠️ DO NOT GATE `retry.on_alpha` ON `binding_limit == 1`. It was proposed, built and reverted
    the same day (2026-08-10).** (Moot since 2026-10-14: there is no `binding_limit` and no whole-step
    cap; kept for anyone restoring one.) More damping shrinks the correction, so it *widens*
    `room = k/|dk|`. The measured evidence is already in this file: the single escalation of an
       entire `bfs3d` march fired at step 51, whose cap was **4.37e-10** — constraint-bound — and it
       was worth **8 steps and 199 s** end to end. Gating `retry.on_alpha` on `binding_limit == 1`
       would have suppressed exactly that one useful escalation.

    **What damping genuinely cannot do is un-pin a cell already ON the boundary**, and that is the
    real rung-2 failure — see the lock-up section immediately below, which is what replaced this.

  **⚠️ (2026-08-10, LATER) THE RUNG-2 DEATH IS A FRACTION-TO-THE-BOUNDARY LOCK-UP, AND β ESCALATION
  IS A SYMPTOM RATHER THAN THE CAUSE.** Read this before acting on the two bullets above; it does not
  contradict them but it re-ranks them, and the "~100 dead steps" is now measured rather than
  estimated. Taken from the archived `march.log` of the traced run (banner: `turbulence inverse:
  jacobi`, `equilibrate=True`, `sweeps=4`, `aggressive_levels=1`, `spectral_damping=False`,
  `refresh on cycles 3`, `retry on cycles / alpha 10 / 0.01`, `pc beta floor 0.05`, rung 2 of 3):

  | step | β | cyc | ‖R‖ | a_min | `limit` |
  |---|---|---|---|---|---|
  | 24 | 0.1756 | 5 | 1.123e-01 | 0.694 | 6.94e-01 |
  | 25 | 0.4682 | 2 | 7.567e-02 | 0.004 | 3.76e-03 |
  | 26 | 3.7458 | 0 | 7.316e-02 | 0.000 | 1.00e-05 |
  | 27 | 16.0000 | 0 | 7.316e-02 | 0.000 | 1.95e-06 |
  | 28 | 16.0000 | 0 | 7.316e-02 | 0.000 | 1.95e-08 |
  | … | 16.0000 | 0 | 7.316e-02 | 0.000 | ÷100 every step |
  | 122 | 16.0000 | 0 | 7.316e-02 | 0.000 | 1.95e-196 |

  **The 100× per step IS the rule, not a coincidence.** Taking `α = τ·room` with `τ = 0.99` leaves the
  binding cell at 1% of its `k`, so the next step's room is a hundredth of this one's — forever, for
  as long as the direction keeps pointing at the boundary there. Ninety-six consecutive steps, residual
  frozen to every reported digit, **zero** Krylov cycles, β pinned at its 16.0 ceiling. So:
  - **β is irrelevant from step 27 on, and that is a statement about a PINNED cell, not about damping.**
    Damping is *argued* to widen a fraction-to-the-boundary cap (a smaller correction means more room),
    and that argument is **not measured and cannot be from a log** — only the accepted attempt's cap is
    recorded. ⚠️ An earlier version of this bullet offered "the cap falls only 5.1× across 26→27" as
    evidence; that is the **same cross-step comparison withdrawn above**, used to argue the opposite
    conclusion, and it is struck. The decision rests on the A/B alone. What damping cannot do is recover
    a cell whose `k` is already ~0: the
    room is then small however small the correction gets. **Do not read this as "stop escalating on a
    constraint-bound step"** — that was tried, and it deletes the one escalation on this case that is
    measured to pay (step 51, cap 4.37e-10, 8 steps and 199 s).
  - **Nothing in the ordinary stopping tests can see this.** The state is finite, the residual is finite,
    and the tolerance is simply never reached — so the segment spends its whole budget.
  - **⚠️ Extracting this from a march log needs per-step BLOCKS, not two independent `findall`s.** The
    `limit` line (in a log from before 2026-10-14, when it was removed) is written only when the cap
    bound, so there are fewer of them than steps (105
    against 108 here) and zipping the two lists silently misaligns them. The first pass at this table
    read step 25's cap as 1.95e-08 instead of 3.76e-03 — a three-step shift, invisible because the
    shifted table is just as smooth. Split on the summary-block delimiter and parse within each block.

  **SHIPPED in response, and DELETED 2026-10-14 with the cap it guarded:** a stall stop,
  `newton_march(stop_on_limit_stall=3)` over a predicate `_limit_collapsing` (three consecutive
  constraint-bound, non-widening steps that moved the residual by under 1e-3 relative **in either
  direction**), validated by `march_stall_replay.py` over 11 archived logs / 22 rungs. None of the three
  exists now -- nothing installs a whole-step cap, so nothing can lock up this way. The two traps its
  predicate taught, if a cap ever returns: "the residual did not fall" never fires (a pinned step still
  moves the state by `α·δ`, ~1e-6 relative), and a one-sided "did not fall by 0.1%" fires on a rung
  that recovers (`march-20260810-094635` rung 2 climbed 1.293e-01 → 1.489e-01 under narrowing caps,
  then converged). Judge any such rule against the runs that recovered, not only the one that stalled.

  The march now fails fast and honestly instead of grinding; **what drives `k` to the boundary in those
  cells is still open**, and the global-scalar-cap design (one cell throttling all 23040) is the thing
  to reconsider.

  **Capturing the step direction:** `compare.py`'s `BFS3D_DUMP_STEP_LIMIT` dump of the step at a binding
  cap, read by `singular_cell_probe.py`, located the binding cells below; both were deleted 2026-10-14 with the cap.

  **⚠️ SUPERSEDED THE SAME DAY — equilibration is a TRIGGER, not the cause. Read the floored-limiter
  entry above before using anything here.** The A/B below is real and reproducible, but it holds **only
  at `positivity_floor = 0`**. With the floor at 1e-8 the flag stops mattering: `equilibrate=True`
  converges all three rungs in 69 steps and `equilibrate=False` in 67, to the **same root in every
  reported digit**. So rescaling never caused the failure — it pushed a cell into the numerically-dead
  zone a few steps earlier than the unscaled arm did, and the global positivity limiter's geometric
  ratchet did the killing. The defect was in the limiter. Keep the case default at `False` (it is free,
  and it is what the converging arms were measured with), but do not describe this flag as deciding
  convergence.

  **⚠️ EQUILIBRATION DECIDES WHETHER THE `bfs3d` TRACED MARCH CONVERGES — true ONLY at floor 0 (see
  above). Measured 2026-08-11.** An A/B differing in **one flag** was run end to end:

  *Configuration, both arms:* `bfs3d` (23040 cells), 3-rung Reynolds continuation (`N_POINTS=2` →
  Re/100, Re/10, target), `BFS3D_TURBULENCE_INVERSE=jacobi`, `field_split=True`; traced trailing
  `cycles=1, sweeps=4, max_coarse=2000, aggressive_levels=1, prolongation_smoothing=none,
  spectral_damping=False`; monolithic smoother **ILU(0) ×4**, `coarse_eq_limit` 2000, plain aggregation,
  reach 3, forward restart 15, `refresh_on_cycles` 3, `retry on cycles / alpha` 10 / 0.01, cycle budget
  42, PC β floor 0.05, stop `(rtol, atol) = (0.0, 1e-5)`, `k` wall BC `zerogradient`.

  | arm | outcome | steps | final ‖R‖ | mid-span `x_r/h` |
  |---|---|---|---|---|
  | `equilibrate=True` (archived `march-20260810-221936.log`) | **locked up, rung 2 step 34** (α 0.001; ‖R‖ frozen from step 35; killed by hand at 39, not solver-terminated) | 39 | frozen 1.257e-02 | — |
  | **`equilibrate=False`** (`march-20260811-003915.log`) | **converged, all 3 rungs, 2081 s** | 77 | **3.586e-06** | **8.361** |

  **It REPRODUCES — the claim does not rest on one run each.** Two further `equilibrate=True` runs,
  `march-20260810-130032.log` and `march-20260810-223702.log` (the latter dump-free), both stall at the
  **identical** ‖R‖ 7.316e-02 with the same β ladder — and they ran under the **`dirichlet`** k wall BC,
  where the pair above ran `zerogradient`. So the flag's effect survives a change of wall closure. The
  march is also deterministic: four runs at identical configuration agree in every printed field.

  **⚠️ The dump wrapper is NOT the confound (checked, not argued).** `221936` ran with
  `BFS3D_DUMP_STEP_LIMIT` (a since-deleted instrument) and `003915` did not. A dump-ON/dump-OFF control pair at otherwise equal
  settings differs in **zero** content lines over 16 steps, against the equilibrate flag's **77**.
  The (deleted) dump wrapper returned the inner cap unchanged and preserved static-field value
  equality, so it neither altered the step nor forced a retrace.

  **⚠️ The separation is a GROWING PERTURBATION, not a threshold event — an earlier version of this
  entry said "bit-identical for 15 steps, then separated on α alone" and BOTH halves are wrong.** The
  arms differ at **step 1** (p-block residual 6.217e-07 vs 6.272e-07) and accumulate 77 differing
  content lines before step 17. They agree only to the **4th printed digit of the summary row**, which
  is insensitive to the diverging component: at step 16 the p-block residuals differ by **19×**
  (1.545e-07 vs 2.938e-06) while the reported ‖R‖ matches to four digits. Step 16 differs in the cycle
  count (5 vs 4) and the retry flag as well as α (1.000 vs 0.699 `L`). The correct statement is: **a
  small preconditioner-dependent difference is present from the first step and amplifies for fifteen
  steps until it crosses the positivity limiter**, after which the clipped α trips `retry.on_alpha` →
  β escalation → more clipping → the 16.0 ceiling at zero cycles. Do not describe this as a clean
  bifurcation; that framing implies a threshold the flag crosses cleanly and the data do not show one.

  **⚠️ `x_r/h` 8.361 against OpenFOAM's 7.243 is TWO GRID STATIONS, not a 15 % discrepancy.**
  `reattachment_length` returns the `x` of the last reversed wall cell — a grid station — and the
  stations near reattachment are `… 6.728, 7.243, 7.787, 8.361, 8.966 …`, spacing ~0.55 h. The
  comparison to the shipped PETSc run (`march.log`: 58 steps, 282 cycles, `x_r/h` 8.3611) is **not
  like-for-like**: it differs in the k wall BC (`dirichlet`) as well as the inverse (`petsc`), and it
  lands on the identical station — i.e. the metric did not resolve either change. Quote the sub-cell
  interpolated crossing from `wall_layer_comparison.py` if this number has to bear weight.
  - **The positivity limiter is NOT the failure — failing to recover from it is.** The converged arm hits
    the same constraint repeatedly (`L` at steps 24, 26, 27, 30, 32, 35, 39, 40, 41, 49, 51, 53, 54, 56,
    57, 60, 61, **including α 0.000 at step 61**) and recovers from every one, α returning to 1.000.
  - **A constraint-free α collapse appeared, and nothing reacts to it.** Step 68: α 0.031 with **no `L`
    flag** and 15 cycles (the run's highest) — a poor *direction*, not a clipped step. α 0.031 is above
    `retry.on_alpha` 0.01, no `RefreshTrigger` reads α, and the β-tracking hook re-fits only on a rebind, so no refresh fires. It cost a few steps here, not the run, but it is the
    first live evidence that the refresh gap is a real cost.
  - **⚠️ ONE RUN EACH, and one instrumentation difference:** the archived equilibrated arm ran with
    `BFS3D_DUMP_STEP_LIMIT=0.05/12`, the converged arm with the dumps off. The dump wrapper returns the
    real cap unchanged by construction, so it *should* be neutral, but it is not a matched pair.
  - **The stated reason for the `equilibrate=True` default no longer exists.** `JacobiSmoothedInverse`
    defaults it on because "the per-cell block solve is not otherwise safe" — raw, 4 of 23040 cell blocks
    were flagged singular. That count came from the **Frobenius** guard (`|det| < 1e-12·‖B‖_F`), which is
    not invariant under row scaling and is **the guard that was found wrong and replaced** by the
    Hadamard row-norm bound, which is invariant. The default is **unchanged pending a decision**; flipping
    it is a shipped-default change.

  **✅ THE FLOOR RESCUES A CONFIGURATION THAT PREVIOUSLY DIED — which is the real case for it, not the
  step count (measured 2026-08-11).** `equilibrate=True` had failed on rung 2 **twice**, under both wall
  closures (`march-20260810-221936.log`, ‖R‖ frozen at 1.257e-02 from step 34; `march-20260810-223702.log`,
  frozen at 7.316e-02 from step 26). Re-run with `positivity_floor=1e-8` and **nothing else changed**, it
  converges all three rungs.

  | arm | `equilibrate` | floor | outcome | steps | wall | final ‖R‖ | `x_r/h` | escalations |
  |---|---|---|---|---|---|---|---|---|
  | `221936` | True | 0 | **died, rung 2 step 34** | 39 | — | frozen 1.257e-02 | — | 13 |
  | `223702` | True | 0 | **died, rung 2 step 26** | — | — | frozen 7.316e-02 | — | — |
  | `003915` | False | 0 | converged | 77 | 2081 s | 3.586e-06 | 8.3611 | 8 |
  | — | False | **1e-8** | converged | **67** | 2133 s | 3.586e-06 | 8.3611 | 4 |
  | — | **True** | **1e-8** | **converged** | **69** | 2188 s | 3.586e-06 | 8.3611 | 5 |

  **The chain the floor breaks, visible step by step.** `221936` clipped at rung-2 step 16 (α 0.699,
  `L`), which tripped `retry.on_alpha`, which escalated β, which clipped harder, up to the 16.0 ceiling
  at zero cycles. With the floor, that **same step 16 takes α 1.000 with no flag**, matching the
  unscaled arm exactly — the binding cell there was numerically dead, and buying it out removes the
  first link. From step 22 on, the floored equilibrated and floored unscaled arms run the *same*
  trajectory field for field.

  **What it does NOT fix, confirmed on all three converging arms.** The rung-3 hard point reproduces
  bit-for-bit regardless of floor or flag: α 0.010 at 15 cycles, then α 0.000, then β 0.0293 → 0.9364
  (**32×**) across two steps and six steps of SER walking it back, then a constraint-free α 0.031 with
  **no `L` flag**. That is the bad-direction mode, and the escalation ladder's **overshoot** is now the
  clearest remaining cost on this case.

  **Where the wall time goes** (equilibrated + floored run, per-step sums, one run on a shared machine —
  proportions not a benchmark): preconditioner **19 %** (304 s of 1561 s through step 48), of which the
  coloured Jacobian probe is 243 s and the refactor 56 s; 17 of 48 steps carried a refresh and only 2
  were scheduled — the rest fired on the reactive 3-cycle rule. Rung 3 costs ~56 s/step against ~25–30 s
  on rungs 1–2. The four most expensive steps are all first-of-rung (115–171 s), carrying a compilation
  on top of the PC build.

  **⚠️ THE α COLLAPSES ARE TWO DIFFERENT FAILURE MODES, and only one is the limiter's fault — measured
  2026-08-11 by a floored-limiter A/B.** `PositiveBlockLimit` (deleted 2026-10-14) gained a `floor`: the room becomes
  `(phi_i + floor)/|delta_i|`, so an entry that is numerically zero stops setting the step for all of
  them. `floor=0` (the library default) is bit-identical to the plain rule, and the limiter is inactive
  at a root for **any** floor (there `delta = 0`), which is what keeps it out of the converged state and
  therefore out of the implicit-function-theorem adjoint — the adjoint never sees it in any case, since
  `root_adjoint`'s backward rule reads only `jax.vjp(residual_fn, root)` and a transpose solve, never
  the Newton step.

  *Configuration, both arms:* `bfs3d`, traced trailing inverse with **`equilibrate=False`**, `k` wall BC
  `zerogradient`, 3-rung Reynolds continuation (`N_POINTS=2`), ILU(0) ×4, `coarse_eq_limit` 2000, plain
  aggregation, reach 3, forward restart 15, `refresh_on_cycles` 3, `retry on cycles / alpha` 10 / 0.01,
  PC β floor 0.05, stop `(0.0, 1e-5)`. Floor **1e-8**, calibrated by replaying the recorded clips
  (a harness deleted 2026-10-14 with the floor).

  | | floor 0 | **floor 1e-8** |
  |---|---|---|
  | steps | 77 | **67** |
  | escalations (all `alpha`-triggered) | 8 | **4** |
  | final ‖R‖ | 3.586e-06 | 3.586e-06 |
  | mid-span `x_r/h` | 8.3611 | 8.3611 |
  | `ux`/`uy`/`uz` rel-L2 | 0.0616 / 0.0072 / 0.0061 | identical |
  | ν_t peak | 150.1071 | 150.1071 |
  | wall | 2081 s | 2133 s |

  **The root is unchanged in every reported digit, and the wall clock is a WASH — do not sell this as a
  speed-up.** The floored arm's rung-2 steps run at low β costing 4–9 cycles where the unfloored arm
  escalated into cheap 2-cycle solves, and that cancels the ten steps. (One run each, and this case has
  no measured run-to-run spread, so 2.5% is not resolvable.)

  - **Mode 1, the RATCHET — the floor eliminates it.** A numerically-dead cell with a *tiny* correction:
    `tau` leaves it at `1 - tau` of its value, so the cap falls ×100 per step while its `delta` never
    moves. Rung 2 went 34 steps → 24, three escalations → one, and the clips stopped binding at all by
    step 33.
  - **Mode 2, a BAD DIRECTION — the floor cannot and should not touch it.** Rung 3 steps 50–51
    reproduce bit-for-bit with the floor in place (α 0.000, ‖R‖ 6.191e-03 vs 6.192e-03), and step 58's
    α 0.031 carries **no `L` flag at all**. For α to collapse with a 1e-8 floor the binding cell needs
    `|delta_k| >> 1e-8` — a large correction on a live cell, not a tiny `k`. No physically-sized floor
    rescues a step whose correction dwarfs the field; the escalation ladder is the right response, and
    the open question there is its **32× overshoot** (β 0.0293 → 0.9364 across two steps, then six
    steps of SER walking it back), not its trigger.
  - **Healthy clips survive, which the dumps could NOT show** (they were written only below cap 0.05).
    Steps 7 and 8 are preserved exactly; steps 9 and 44 shift by 0.1–0.3 % — exactly the `floor/k`
    perturbation the softened form predicts for a live cell. That is also how to read a clip: a live
    cell moves by `floor/k`, a dead one is bought out entirely.
  - **⚠️ Choose the floor by REPLAY, not by guess — the dead cells are a graded population, not one
    outlier.** Exempting the worst promotes the next: worst recorded cap 1.05e-09 unfloored, 1.6e-02 at
    a 1e-12 floor, 6.9e-02 at 1e-10, ~0.35 at 1e-08. An earlier estimate of "1e-12 should do it" was
    wrong by four orders. At 1e-06 the binding cell is a live one (`k` 4.7e-03, cap 0.84), so 1e-08
    keeps two orders of margin below anything physical here (inlet `k` 0.375, mesh median ~3e-2).
  - **Safe only because every consumer of the solved `k` clamps at zero.** The limiter's original
    justification — a negative `k` reaching a bare `sqrt` and poisoning the residual — no longer holds:
    `f1`, `f2`, `eddy_viscosity`, the production cap and the wall closures all clamp, and at `k < 0` the
    destruction term runs on the `k`-independent viscous ω branch, whose sign pushes `k` back up.
    **Re-check that before floating this limiter on another field.**

  **✅ BUILT AND MEASURED (2026-08-11): the per-cell PROJECTION removes the cap's failure mode entirely
  and does NOT make the case faster. Read this before proposing anything else aimed at the limiter.**
  `PositiveBlockProjection` clips each cell's own correction, `delta_i <- max(delta_i, -tau(k_i + floor))`,
  instead of scaling the whole step by the worst cell; applied before the cap, which then computes
  exactly `1`. It answers Mode 1 structurally rather than by decades of headroom, since a floored *cap*
  only postpones the ratchet — `(k_new + floor) = (1 - tau)(k + floor)` per capped step, whatever the
  floor.

  *Configuration, both arms:* `bfs3d`, traced trailing inverse, `zerogradient` k wall, floor 1e-08,
  `equilibrate=False`, `refresh_on_cycles` 3, ILU(0) ×4, trailing sweeps 1, `coarse_eq_limit` 2000,
  restart 15, two rungs. **Both predate the probe/memory work merged in #188/#189**, so their wall
  clocks are not comparable to anything measured after it.

  | | projection | cap only |
  |---|---|---|
  | steps | 65 | 67 |
  | **Krylov cycles** | **329** | **329** |
  | wall | 2021 s | 2124 s |
  | positivity-limited (`L`) steps | **0** | 22 |
  | escalations | 5 | 4 |
  | mid-span `x_r/h` | 8.36 | 8.36 |

  **The cycle count is IDENTICAL, which is the number to read.** Wall differs by 4.8 % between runs
  taken at different machine loads and this case has no measured run-to-run spread; the linear-solve
  work did not move. **Do not cite this as a speed-up.**

  **Why it bought nothing: the cap was one DOOR into the cascade, not its cause.** The mechanism worked
  exactly as designed — the cap never binds once, against 22 times — and the same cascade reappeared
  through the line search's descent test instead. Rung 3 steps 53–55: α 0.000 twice with **no `L`
  flag**, β driven 0.0548 → 0.8769 (**16×**), cured at 0.8769, then six steps walking back to 0.0770 —
  the walk-back law again, `log(0.8769/0.0770)/log(1.5)` = **6.00** predicted, **6** observed. This is
  Mode 2 above, and it confirms that entry empirically: no treatment of the *limiter* reaches it.

  **So the lever is the RETURN, not the trigger and not the constraint.** Three separate attacks on the
  trigger side are now measured or refuted — gating `retry.on_alpha` on `binding_limit`, raising the
  floor, and removing the cap's global coupling altogether — and none moved the cycle count.

  **Keep the projection for ROBUSTNESS, not speed.** It eliminates the positivity lock-up outright (the
  mode that killed the 77-step march and the β = 16 runaways). It is not *dominated* — it removes a
  failure mode nothing else does — so it is not a deletion candidate, but it must not be sold as a
  performance feature.

  ⚠️ **The `positivity_projection` switch no longer exists** (deleted 2026-10-14): the per-cell clip is
  unconditional on every case, bfs3d included, now composed with a per-cell ω step bound. The 2026-08-25
  pair below (projection costing bfs3d 41 % more cycles) is the trap that bound fixes — full-length
  steps into bad iterates once the global cap stopped shortening them.

  ✅ **RE-ADJUDICATED AT TODAY'S DEFAULTS (2026-08-25, #314): the neutral result above has become a
  41 % CYCLE COST, and the conclusion — robustness, not speed — is unchanged and now has a price.**
  Same controlled pair, only `positivity_projection` varying, back to back on one machine.

  *Configuration:* `bfs3d`, three-rung Reynolds continuation, field split on, `simplesmooth` flow inverse
  (2 sweeps, frozen coarsening), in-framework trailing inverse, `zerogradient` `k` wall, floor `1e-08`,
  `retry_on_cycles` 10, `cycle_budget` 42, `forward_rtol` 0.3, compiled ILU(0) live, at `e46564a` (i.e.
  after the flow boundary-closure and coupled-`k`-shift fixes of #312/#313, which move this march:
  the cap arm is 69 steps here against the 67 of the last pair on record).

  | | cap only | projection |
  |---|---|---|
  | steps | 69 | **62** |
  | **Krylov cycles** | **365** | 515 (+41 %) |
  | wall | **2268 s** | 3166 s (+40 %) |
  | positivity-limited (`L`) steps | 26 | **0** |
  | step redos (`eN` flag) | **7** | 11 |
  | mid-span `x_r/h` | 8.3611 | 8.3611 |

  **The loss is entirely the TARGET rung, and the projection WINS the two below it** — 35 steps / 142
  cycles / 851 s against 39 / 162 / 958. On the target rung it spends 373 cycles / 2315 s against 203 /
  1310, which is **13.8 Krylov cycles per step against 6.8**.

  **Why: the global cap was doing globalization work, and the per-cell clip removes it without replacing
  it.** With the step no longer shortened, the line search takes full steps into iterates the carried
  preconditioner solves badly — inner solves pinned at 12 cycles where the cap arm's run 2–5, and one
  attempt at α = 1.000 whose inner residual reaches **2.7e+10** — and each such solve trips
  `retry_on_cycles` and redoes the step at unchanged β: **8 cycle-triggered redos against 2**. Note the
  summary row's cycle count is the ACCEPTED attempt only, so the +150 cycles is real per-solve cost and
  the discarded attempts are extra wall on top.
  ⚠️ **This does NOT re-open Mode 2 as "the cascade reappears through the descent test".** The 2026-08-11
  pair saw the descent test take over; here the trigger is the *cycle* bailout, which did not exist in
  that bundle. Same conclusion by a different route.
  ⚠️⚠️ **THESE NUMBERS WERE TAKEN AT `e46564a`, WHICH CARRIED THE COUPLED-`k`-SHIFT CHANGE OF #317 —
  REVERTED THE NEXT DAY (#322, `db2b87c`). RE-ADJUDICATE BEFORE RELYING ON THEM.** The revert removes
  `live_eddy_viscosity=True` from `_coupled_shift_policy`'s `k` transport, so the `k` shift diagonal is
  a different quantity now — and that is not a distant confound, it is the term the positivity limiter
  clips against: the shift sets `dk`, and the cap and the projection are two ways of bounding `dk`.
  Measured on the sibling, the reverted change **halved** the shift, flipped the sign of `J_kk + βd`,
  and degraded the kept step length 6–38×. Both arms here ran under it, so the PAIR is still controlled
  and the qualitative reading (the cap binds 26 times against 0; the projection wins the two lower rungs
  and loses the target one) is what the logs say — but every number in the table above describes a code
  state `main` no longer has. The case default stays off either way: it was off before this measurement
  and nothing here argues for flipping it.

  ⚠️ **UNTESTED, and the obvious next probe:** `retry_on_cycles = 10` was calibrated under the cap, so
  whether the target-rung loss belongs to the projection or to a threshold that no longer suits it is
  not established. It is a hardcoded constant in the case (no environment override), so a third arm
  needs a code edit.

  **⚠️ REFUTED — "rescaling promotes collapsed-`k` rows and inflates their corrections" is FALSE. Do not
  re-propose it (measured 2026-08-11, `k_row_scale_probe.py` -- deleted 2026-10-14: the `step-limit-*`
  dumps it read went with the global k cap and can no longer be produced).** The proposed explanation for why the
  `equilibrate` flag changes the step length was: symmetric rescaling divides row `i` by `sqrt(A_ii)`; a
  cell whose `k` has collapsed has a tiny diagonal there; so rescaling promotes that row to unit weight
  and un-scaling inflates the correction in exactly the cells the cap (a **minimum** over cells) is
  decided by. **The first clause is false**, so the rest cannot hold.

  *Configuration:* `bfs3d`, states `step-limit-04`/`-11` (both from `march-20260810-223702.log`: jacobi
  trailing inverse, `equilibrate=True`, **`k` wall BC `dirichlet`**, rung 2 = Re/10, β 0.468 and 4).
  Exact `∂R_k/∂k` per cell by one-hot Jacobian-vector product — no materialization — 40 cells per decile
  of `k` plus the three cells the limiter is observed to bind on.

  | decile of `k` | median `k` | median `∂R_k/∂k` | scale `1/sqrt(|diag|)` |
  |---|---|---|---|
  | 0 | 8.974e-13 | 3.543e-05 | 168 |
  | 3 | 1.702e-03 | 1.246e-05 | 283 |
  | 9 | 4.893e-01 | 3.176e-05 | 177 |

  Across **twelve orders of magnitude in `k`** the diagonal moves ~1.1× and the scale 1.7×, peaking in
  the MIDDLE deciles — lowest-decile/highest-decile scale is **0.95×**. The binding cells sit **below**
  the median scale (12800 at 0.89×, 3181 at 0.51×, 22400 at 0.89×), i.e. rescaling mildly *demotes*
  them. The reason is structural: `∂R_k/∂k` is set by the destruction `β* ω V`, face transport and the
  pseudo-transient shift, **none of which vanish as `k -> 0`** — a collapsing `k` does not weaken its own
  equation.

  **The within-cell control is stronger than the table and removes the last confound.** Cell 12800's `k`
  differs by **six orders of magnitude** between the two dumps (3.082e-16 at β 0.468, 3.082e-22 at β 4)
  and its diagonal is **identical to four digits in both, 2.864e-05**. Same cell, same position, `k`
  varying a millionfold, `∂R_k/∂k` unchanged — so the flat decile trend is not an artifact of which
  cells populate the low deciles. (Cell 22400 reports the same 2.864e-05, so the two share a structural
  situation; cell 3181 differs at 8.753e-05.)

  **⚠️ The probe that first "measured" this question was structurally incapable of answering it — check
  for this failure mode before trusting any arm comparison.** It preconditioned with
  `CoupledShiftPolicy.make_preconditioner`, which is block-SIMPLE on `[u,v,w,p]` plus (with
  `method=None`, now `scalar=UnpreconditionedScalars()`) **identity on `k` and `ω`**. `equilibrate` lives only inside the engine's
  `FieldSplitPreconditioner` (via `trailing_inverse`), so **both arms ran identical code** and
  returned identical corrections — reported as "no effect". Two ~2 GB Jacobians were built and discarded
  to produce it. The faithfulness gate could not catch it: the gate forms `operator(δ) − b`, which
  contains **no preconditioner at all**. *An A/B needs an assertion that the arms actually differ
  (`assert not array_equal(...)`), not only a gate that the system is right.*

  **⚠️ A Euclidean gate cannot establish solve accuracy on this system.** That probe read its
  8.4e-07 **2-norm** gate as "the march over-delivers by orders of magnitude against its 0.3 stop". The
  coupled Euclidean residual is ~100 % `ω` — the reason the row-scaled stop exists — so 8.4e-07 there is
  consistent with the `k` and velocity rows sitting at 0.1–0.3. Report the gate **per block**, in the
  measure the solve actually stopped in.

  **What the dumps DO show, and it points away from the preconditioner.** At fixed β, `δk` at the
  binding cell is unchanged to 7 digits between dumps while `k` falls ×100 per clipped step —
  `3.0816e-16 -> 3.0816e-22`, mantissa preserved, exactly `(1 − τ)` at `τ = 0.99`. The correction is not
  driving the collapse; **the limiter is ratcheting one cell toward zero and the global `min` lets that
  one cell of 23040 throttle the march.** The binding component is also ~13 orders below its block's
  norm — order 40× the round-off floor — so *which* arm's correction clips is likely not a reproducible
  quantity at any tolerance. The live target is the limiter's design (a `k`-relative floor, `τ` tapering,
  or a per-cell rather than global cap), not the hierarchy.


  **The tree is verified NEUTRAL on the shipped path.** A control march (PETSc ILU(0)×1) on all of
  this measured **58 steps / 282 cycles / final ‖R‖ 9.588e-06 / mid-span `x_r/h` 8.36** — identical
  in every reported digit to the recorded baseline. Wall was 1809 s against 1636 s — **10.6%, unexplained**, and
  on one run each. ⚠️ This case has **no measured run-to-run spread** to judge that against; the "~2%" it
  was previously compared to was a remembered figure with no configuration and has been deleted. The
  neutrality conclusion rests on the cycle count and the reported digits, not on the wall.

  ⚠️ **SIX OF THESE HARNESSES WERE DEAD ON ARRIVAL UNTIL 2026-08-14, AND NOTHING ANNOUNCED IT.** Each
  opened with `march_beta, _, description = STATES[name]` — a **positional** unpack of a record that has
  since grown to five fields — so every one of them raised `ValueError: too many values to unpack` before
  reaching its first line of work: `trailing_hierarchy_sweep.py`, `cell_block_scaling.py`,
  `trailing_block_conditioning.py`, `turbulence_smoother_sweep.py` (since deleted, #371), `zero_pattern_pivots.py` and
  `field_coupling.py`. All six now read the fields **by name**, which is what makes them survive the next
  field. **The lesson is about the keep-the-harness rule rather than about these six**: a harness kept in
  the repository is only re-adjudicable if it still *runs*, and nothing in the test suite exercises these,
  so a shared record can grow a field and silently retire the entire probe fleet. Every measurement in
  this file whose harness is named above was taken before that breakage and is unaffected — but any
  attempt to re-take one between the field's addition and this fix would have failed at startup.

  **✅ ANSWERED: the cap binds on ONE cell, it is a step-corner cell whose `k` is already numerically
  zero, and it is NOT one of the ill-conditioned ones.** Measured from twelve `BFS3D_DUMP_STEP_LIMIT`
  dumps taken on the traced march at rung 2 (`equilibrate=True`, `sweeps=4`, `aggressive_levels=1`,
  `spectral_damping=False`, `refresh on cycles 3`, `retry on cycles / alpha 10 / 0.01`, PC β floor
  0.05), the operator swept over β 0 … 0.4. The probe reproduces the recorded cap exactly
  (3.761982e-03 both ways), so the capture is reading the quantity the march acted on.

  | dump | cap | cell | room | `k` | `dk` | next cell | next room |
  |---|---|---|---|---|---|---|---|
  | 04 | 3.7620e-03 | **12800** | 3.80e-03 | 3.08e-16 | −8.11e-14 | 22400 | 2.60e-01 |
  | 05 | 1.4517e-04 | **12800** | 1.47e-04 | 3.08e-18 | −2.10e-14 | 12840 | 2.09e+00 |
  | 08 | 1.0516e-07 | **12800** | 1.06e-07 | 3.08e-20 | −2.90e-13 | 12840 | 1.12e-01 |
  | 11 | 1.0516e-09 | **12800** | 1.06e-09 | 3.08e-22 | −2.90e-13 | 12840 | 1.12e-01 |

  - **One cell out of 23040 throttles the whole march,** and it is not a crowd: at dump 04 exactly
    **one** cell sits within 100× of the tightest room, the runner-up 68× behind; by dump 11 the gap is
    **eight orders**. Cell 12800 owns ten of the twelve dumps and every one from step 25 on.
  - **`k` there ratchets by exactly 100× per step** (the per-step evidence is the checkpoint mantissa
    below; these dumps are non-consecutive) — 3.08e-16 → 3.08e-18 → 3.08e-20 → 3.08e-22 — which
    is `1 − τ` at `τ = 0.99`, the same factor the cap collapses by. Against a mesh median `k` of
    **2.97e-02**, so the binding cell's `k` is ~20 orders below typical. It is not small; it is zero.
  - **`dk` stays ~1e-13 throughout while `k` collapses.** ⚠️ **The inference drawn from this — "the
    cell's `k` equation has no root at `k ≥ 0`" — is WRONG, and was refuted the same day by direct
    measurement. It is the opposite.** Holding every other field at the `step-limit-11` iterate and
    scanning `R_k[12800]` over `k_P` gives a straight line to seven digits,
    `R_k = A·k − b` with `A = 2.8740e-06`, `b = 5.7236e-20`, so the **root is `k* = +1.99e-14`,
    strictly positive** — and the iterate sits **eight decades BELOW its own root**. Newton should be
    pushing `k` *up*.

    Production, destruction and the `Dirichlet(0)` wall term all vanish linearly at `k = 0`, so `k = 0`
    solves the row exactly when the neighbours are 0. (⚠️ An earlier version called the corner `k` system
    "**homogeneous and an M-matrix**" — the M-matrix half is **false**: `A_kk` carries **14 non-positive
    diagonals**, min −9.03e-07.)

    **Why the returned correction is wrong there, corrected — it is Krylov truncation, NOT roundoff.**
    An adversarial re-derivation (independent colour-recovery of `A_kk`, verified row-wise against a
    `jax.vjp` to 5e-16) settles the mechanism:
    - `R_k[12800] = −5.72e-20` is **deterministic and 13 orders ABOVE its own row's floating-point
      floor** (~1.3e-33). "Roundoff floor" was the wrong description.
    - What is true is stronger: `|R_row| / ‖R_k‖₂ = 2.94e-16 ≈ machine epsilon`, and
      `|R_row| / ‖R‖₂ = 2.15e-22`. **Any Krylov solve stopping on a relative-residual test would need
      ~1e-22 relative accuracy in float64 to resolve this row — unreachable in principle, not merely
      under-solved.** Measured: `‖A_kk·dk − rhs_k‖/‖rhs_k‖ = 1.0055`, i.e. the returned correction does
      not reduce the k-block linear residual at all, and dumps 08/09/10 sit at a **bit-identical state**
      yet return `dk[12800]` of −2.901e-13, −8.825e-14, −1.567e-14, an **18× spread**.
    - The returned `dk` also provably violates its own row for *every* admissible shift: the row demands
      `βD·(k_ref + 2.901e-13) = −3.958e-19`, and the left side is ≥ 0 while the right is < 0.
    - ⚠️ **"14× larger than the exact local step" was wrong** — that froze the neighbours. Given the
      dumped neighbours the row-consistent `dk` is −1.52e-13, still negative; it is the **exact k-BLOCK
      solve** that comes out positive, at `+3.90e-10` (β = 0) through `+1.90e-15` (β = 10), i.e.
      **positive at every β from 0 to 10** while the dumped value is negative and 3–5 orders off.

    `positive_block_limit` (the global cap, deleted 2026-10-14) then computed a purely **relative** room
    `0.99·k/|dk|` and let that cell veto the global step.
    - **⚠️ THE MOST USEFUL LEAD OF THE LOT: at β ≥ 0.5 an exact k-block solve produces NO binding cell at
      all (cap = 1.0).** Only at β ≤ 0.05 does an exact solve give a tight cap, and then on a *different*
      cell (7.6e-14 at cell 2679). So the cap is **not intrinsic to the `k` equations at a healthy
      shift** — it is a low-β-plus-inexact-solve artifact, which points back at the low-β conditioning
      wall rather than at the closure.

    **The ratchet is proved by the mantissa.** `k[12800]` reads `3.0816e-22` at `step-limit-11` and
    `3.0816e-208 / e-210 / e-212` at checkpoints 120 / 121 / 122 — **the same five digits, 190 decades
    apart**, i.e. ninety-five successive ×0.01 cuts, while `u/v/w/p` report a relative change of
    exactly 0.
  - **It is not one cell, either: 1876 of 23040 cells sit below `k = 1e-6`** at this state, against an
    OpenFOAM global minimum of 1.672e-6. ⚠️ **BOTH HALVES OF THAT COMPARISON WERE WRONG, AND THE
    CONVERGED ROOT HAS NO SUCH DEFICIT — see the resolution below.**
  - **Geometry: it is a step-corner cell, and its mirror is the runner-up.** 12800 sits at
    `(0.0007, −0.0099, 0.0009)` and 22400 at `(0.0007, −0.0099, 0.0391)` — bottom wall (`y ≈ −h`),
    immediately behind the step (`x ≈ 0`), against each side wall (span `4h = 0.04`). The stagnant
    bottom/side-wall corner, i.e. the same corner-separation region that makes the full-span
    reattachment (16.14 here) disagree with the mid-span one (5.34). ω there is 4.01e+05, a wall value.
  - **The tightness ranking tracks the wall-face count — on n = 4 cells, one dump, one state.** Counted
    off `face_patches`, the four tightest cells carry **3, 3, 2, 1** no-slip faces out of six:

    | cell | room (dump 04) | boundary faces |
    |---|---|---|
    | 12800 | 3.80e-03 | `lowerWall`, `lowerWall`, `sideWalls` |
    | 22400 | 2.60e-01 | `lowerWall`, `lowerWall`, `sideWalls` |
    | 12840 | 1.38e+00 | `lowerWall`, `sideWalls` |
    | 13129 | 2.17e+00 | `sideWalls` |

    The two cells that own the cap are **trihedral wall corners** — half of every face is a no-slip
    wall, the two `lowerWall` faces being the floor and the vertical step face. **Hypothesis, not yet
    measured:** the near-wall `k` closure is per-wall-cell, and its production is area-averaged over a
    cell's wall faces while the destruction `β*kω_wall` is not obviously averaged the same way — so a
    three-wall-face cell could take up to 3× the destruction against one cell's worth of production.
    **❌ REFUTED the same day by reading both sources** — our reduction is already OpenFOAM's:
    `wall_cells` is a `jnp.unique`, so a cell appears once however many wall faces it has, and
    `wall_shear_rate` area-averages `Σ|S_f|r_f / Σ|S_f|` over them, which is exactly
    `patchFieldsToWallCellField` in OpenFOAM 13's `wallCellWallFunctionFvPatchScalarField`. Nothing is
    summed per face. The ranking most likely reads **stagnation** — three wall faces means the deepest
    dead zone and so the least production — not double counting. See `.claude/notes/turbulence-log.md`, "Near-wall k collapse at trihedral wall corners", for
    the two differences that are real, of which one bears directly on the lever below.
  - **NOT the ill-conditioned cells — the standing hypothesis is refuted at this state.** **Zero**
    singular blocks at every β from 0 to 0.4. The binding cells run cond 3.6e5 … 2.9e7, ranks
    1088–2165 of 23040, and **none** of them is among the twelve worst-conditioned. `cond > 1e6`
    catches 50% of them against an 8.7% base rate — mildly enriched — but `cond > 1e9` catches none,
    and the separately characterised cond ~1e12 / ‖B⁻¹‖ ~9.5e8 cells are absent entirely (the worst
    here is 2.9e7 / 6.6e6). **Caveat on comparing those two numbers:** the 1e12 figure was measured on
    `state-00057` under *symmetric equilibration*, and this is a different state, raw — so this
    refutes the coincidence at this state rather than retiring the 1e12 finding.

  **✅ RESOLVED (2026-08-10, by re-running the shipped march end to end): the `k` trough is a REYNOLDS-
  CONTINUATION TRANSIENT, and the converged root does not have it.** At the converged target-Re root
  (step 58, `R` 9.588e-06, `x_r/h` 8.36) there are **ZERO** cells below `k = 1e-6`: `k_min` 1.2997e-05,
  median 0.7017 against OpenFOAM's 0.7468, a ratio of **0.98**. Cells below 1e-6 go 190 (step 37) → 130
  → **0** (step 46) → 0 at convergence.
  - **Both halves of the "1876 vs 1.672e-6" comparison were wrong.** It is **cross-Reynolds** —
    `state-00122` is rung 2 (ν = 1e-4), the OpenFOAM field is target Re — and 1.672e-6 is the **steady**
    OpenFOAM run's minimum, the run this case's own docstring rules out as a reference. The valid
    transient reference has `k_min` **6.2247e-03**.
  - **The mechanism is the continuation's own doing.** `y* = β*^0.25 √k d/ν` scales as **1/ν**, so at the
    Re/100 anchor `wall_function_weight = tanh((y*/y*_lam)⁴)` is ~1e-7 — **the log-layer wall production
    is switched off even at fully turbulent `k`** — while `ω_wall = 6ν/(β₁d²)` is **100×** larger, so
    `β*kω` destruction is 100× the target's. Homogeneous, linear, destruction-dominated ⇒ `k` decays.
    Raising Re reverses both: the weight median goes 5.6e-20 (rung-1 root) → 1.9e-10 → **1.000** (target).
  - The rung-1 root is **correctly** laminar (`ν_t/ν` median 1.7e-5 at `Re_h = 100`), and **1732 of the
    1876 cells (92%) are inherited from it**.
  - **Population is pre-stall, depth is stall-caused**: `#k<1e-6` is 1876 by dump 04 and frozen there for
    all 97 remaining steps, while `k[12800]` falls 3.08e-16 → 3.08e-212 over the same span.
  - **The stalled arm is `BFS3D_TURBULENCE_INVERSE=jacobi`; the default `petsc` arm walks the same trough
    and recovers** — agreeing with the exact-solve finding that at β ≥ 0.5 there is no binding cell.
  - ❌ **REFUTED — the initialization.** `hybrid_initialize` gives a **uniform** `k = 0.2489585`, zero
    cells below 1e-6.
  - ❌ **REFUTED — the "ω is exactly 10×, still at its 60ν seed" lead.** `ω[12800] = 400910.4` **is**
    `6ν/(β₁d²)` at **rung 2's own** ν = 1e-4 to 16 digits (all 4490 wall cells within 4.8e-5). The three
    rungs give 4.0091e6 / 4.0091e5 / 4.0091e4 — the "10×" compared a rung-2 state against the
    **target-Re** formula. Those rows are converged; the `60ν` seed was replaced by `omega_wall` in
    `c324021` (2026-07-23), well before these checkpoints.
  - **The sustaining/ambient source has no motivating evidence left**: it targets a defect absent from
    the converged root. The defensible levers are the absolute floor in the limiter, and not anchoring
    the ladder below the Reynolds number at which the wall function turns itself on.
  - **Genuinely open, and now the real physics question — but BOTH numbers needed a definition before
    they meant anything (harness: `validation/bfs3d_openfoam/wall_layer_comparison.py`).**
    - "**First wall layer**" spans a 4× range of defensible meanings, and the choice matters more than
      the discrepancy: **all wall-adjacent (4490 cells) is 1.054**, the **finest layer (1600) is 0.459**,
      the **floor behind the step (640) is 0.342**. The recorded 0.164/0.46× is the *finest layer*. Side
      walls, 64% of wall-adjacent cells, agree at **1.113** and do not participate.
    - `x_r/h` **7.24 and 8.36 are both grid stations, two apart.** The floor's stations near
      reattachment are `… 6.728, 7.243, 7.787, 8.361, 8.966 …`, local spacing **0.375–0.749 h**, so the
      metric's resolution is about half a step height and "15%" is a two-cell offset. A sub-cell
      interpolated crossing puts the reference at **~7.6**, i.e. the quoted 7.24 carries a systematic
      −0.36 h truncation, and the interior spanwise columns scatter by **sd 0.13 h**. A defensible
      reference is **x_r/h ≈ 7.6 ± 0.3**; the gap survives it (~1.0 h, ~13%) but is smaller than quoted.
    - **The wall closure is NOT the cause.** The `Dirichlet(0)` vs `kqRWallFunction` (zero-gradient)
      difference is real but worth only ~7.4% of local destruction (k ×0.93–0.98); `nut_wall` is
      *algebraically identical* to `nutkWallFunction`; every verified wall-closure difference multiplies
      to **~0.87–0.93, not 0.46**. The wall-cell `k` budget is **transport-dominated**
      (|transport|/production ≈ 1.16), so that `k` is inherited, not locally made.
    - **Where it is inherited from, and this is the causally clean part:** the deficit is already there
      **upstream of the step** (x/h −2.85 … −0.15, no recirculation), first-cell `k` **0.54–0.68×** with
      the channel core at parity (0.987), and the lip-shed rows carry the same ratios. In wall units
      `k⁺` ≈ 1.3–2.4 against OpenFOAM's 2.4–4.0, DNS channel ≈3.9–4.4, and the SST log-layer equilibrium
      `1/√Cμ` = 3.33 — **low against both references**, which is what makes it a defect rather than a
      difference. It seeds a shear layer carrying 20–25% less `ν_t` through x/h 0.3–2.
    - ❌ **Exonerated with evidence:** the momentum scheme (aquaflux is *more* dissipative yet its shear
      layer is *thinner* — wrong sign); the SST shear limiter (identical branch in both); the mesh
      (cell-for-cell identical, max centroid distance 4.7e-9 m); and first-order upwind on k/ω (false
      diffusion ~5% of `ν_t S²` at x/h 0.4, <1% beyond — an order of magnitude too small).
    - ⚠️ **THE EXPANSION RATIO CHANGES WHICH NUMBER LOOKS WRONG, AND NEITHER OF US HAD ACCOUNTED FOR
      IT.** The famous benchmarks are ER 1.125–1.2 (Driver & Seegmiller 1985: 6.26 ± 0.10; Le, Moin &
      Kim 1997 DNS: 6.28) and **must not be used as the target here — this case is ER = 2**, and the
      published trend is that larger ER *lengthens* the normalized bubble (Armaly et al. 1983, quoting
      Durst & Tropea 1981). **At ER = 2 the 2D references cluster at 8.0–8.8**: Pont-Vílchez, Trias,
      Gorobets & Oliva (2019, *JFM* 863) DNS gives **X_r = 8.8h** at Re_τ = 395 (verified independently
      of the agent that reported it — a later LES cites it as its reference, calling its own 8.15h a
      7.3% under-estimate); Durst & Tropea (1981) 8.5 at Re_H 1.5e4; Rothe & Johnston (1975) 7.8.
      **So aquaflux's 8.36 sits INSIDE the ER = 2 band and OpenFOAM's 7.24 sits below it** — the
      opposite of the "aquaflux over-predicts by 15%" framing this section started from.
    - ⚠️ **"SST is a known under-predictor" is also wrong.** On Driver & Seegmiller, NASA's turbulence-
      model resource puts SST at x/H ≈ **6.50** against 6.26 ± 0.10 — a 4% **over**-prediction,
      reproduced across four codes; Menter (1994) reports 6.5 himself. The classic under-prediction is
      **k-ε** (Menter's Jones–Launder 5.5), and even the widely-repeated "20–25%" is disowned by
      Thangam & Speziale (ICASE 91-23) as a resolution artifact.
    - **What is genuinely unresolved is the FINITE SPAN.** span/h = 4 is 2.5× below the span/h > 10
      that de Brederode & Bradshaw (1972) require for two-dimensionality at the centreline (quoted
      verbatim in Jovic & Driver 1994, NASA TM 108807). Lower aspect ratio **shortens** reattachment —
      direction unanimous across every source found — which would pull both codes below the 2D band.
      **No open-access `x_r`-vs-aspect-ratio numbers exist**, so the magnitude is unknown and neither
      7.24 nor 8.36 can be called correct.
    - ⚠️ **One anomaly worth chasing: our spanwise profile has the WRONG SIGN against the literature.**
      Both codes here reattach *later* at the outer slabs (10.28) than in the interior (7.24), but the
      two published spanwise measurements go the other way — Sugiyama et al. (2013) find near-side-wall
      reattachment ~60% of the centreline value at AR 16, and Armaly, Li & Nie (2003) find a *minimum*
      near the side wall. Both are laminar/transitional, so the transfer is uncertain, but this is the
      one place our result contradicts published structure rather than merely differing in magnitude.
    - **Untested and the largest remaining unknown: grid convergence.** One mesh exists.

  **⚠️ THE LEVER, as corrected earlier — with the scan's reach stated.** The measurement below is a **1-D scan of
  one row at one iterate with every other field frozen**. It establishes a strictly positive root of cell
  12800's frozen-field `k` row, which removes the motivation for a projection *at this state*. It does
  **not** establish the sign of the coupled Newton direction, nor anything about the other 1875 collapsed
  cells, nor that no state has a negative root. Two defects remain, and they are different from
  each other:
  1. *(Moot since 2026-10-14: the cap, and the floor built for it 2026-08-11, are deleted; `k` is clipped
     per cell.)* **`positive_block_limit` is a purely RELATIVE rule** — `room = tau·k/|dk|` — applied to a field
     whose physical floor is 0. A relative rule has no lower bound, so a roundoff-level `dk` in an
     already-collapsed cell produces an arbitrarily small cap. An **absolute floor**
     (`room = tau·(k + k_abs)/|dk|`) fixes it, is pure globalization, sits off the IFT path entirely,
     and — *provided the cap is inactive at the converged state* — changes nothing at the solution.
     Scale `k_abs` off the block (`eps·max(k)`), not a constant, so it carries `k`'s units.
     **⚠️ IT IS NOT SAFE ALONE — clamping `k` in the closure is a PREREQUISITE, in the same change.**
     Relaxing the cap lets `k` go slightly negative, and two sites then misbehave badly (verified):
     `SSTModel.f1`/`.f2` (`sst.py:136,178`) take a raw `jnp.sqrt(k)` and NaN the whole residual, where
     every closure in `boundary.py` uses `safe_sqrt(jnp.maximum(k, 0.0))`; and `OmegaProduction`
     (`sources.py:288-294`) divides by `jnp.maximum(nu_t, 1e-30)`, so a negative `ν_t` selects the floor
     and the cap becomes ~**−1e23** at the measured corner values. `KProduction`'s cap
     (`sources.py:209`) also flips sign, turning production into a sink. **Unbuilt and unmeasured.**
  2. **A large part of the near-wall field has laminarized** (1876 cells below 1e-6). That is the real
     physics defect and the absolute floor does not address it. The candidate is a **sustaining /
     ambient source**. ⚠️ **Both the form and the citation first recorded here were WRONG; corrected:**
     - The literature terms are **constant**, and there are **two** of them, one per equation:
       `k: + β* ω_amb k_amb` and `ω: + β ω_amb²`, where `ω_amb` is an **ambient constant, not the local
       ω**. Source: **Rumsey & Spalart, AIAA J. 47(4), 2009, 982–993** (NASA calls it `SST-sust`).
       **Spalart & Rumsey 2007** — cited here originally — proposes *floor values*, not source terms.
     - The form written here first, `+β* k_amb ω` with the **local** ω, is a different term: it makes
       the destruction `−β* ω (k − k_amb)`, pinning `k ≥ k_amb` *uniformly including deep in the
       boundary layer* where ω is O(1e5). At the `bfs3d` corner that is ~80× the literature term and
       would override the `Dirichlet(0)` wall condition the case sets — which is exactly what Rumsey &
       Spalart warn against. It also is **not** diagonal-free: `∂R_k/∂ω = −β* k_amb` is an off-diagonal
       the frozen AMG operator would not see. The literature (constant) form does have both properties.
     - **Motivating evidence is weak.** The `k` deficit is measured at an *unconverged* state whose ω
       rows have not moved off initialization, and OpenFOAM reaches min `k` 1.672e-6 on this mesh with
       **no** sustaining term at all. It would also make two existing assertions tautological
       (`test_coupled_mass_flow.py:138`, `test_coupled_periodic_channel.py:124`, both of which assert
       `min(k) > 1e-6` and document themselves as *non*-tautological).
     - **Order of work: do the step-limiter fix first and re-measure.** If the march then converges with
       `min(k)` near OpenFOAM's, this has no motivating evidence left. Unbuilt and unmeasured.

  Two further leads found while measuring this, both unverified: `ω[12800] = 4.0091e5` is **exactly
  10×** the `omega_wall` value `6ν/(β₁d²)` its own residual imposes, which is the `60ν` initialization
  value never relaxed (so this state's `ω` rows are unconverged too); and `SSTModel.f1`/`.f2`
  (`sst.py:136,178`) use **plain unclamped `jnp.sqrt(k)`** where every closure in `boundary.py` uses
  `safe_sqrt(jnp.maximum(k, 0.0))` — clamping those two would make a transiently negative `k`
  survivable, which is what an absolute-floor cap needs.

  **The rest of this paragraph is the superseded framing, kept because the OpenFOAM comparison in it
  still stands on its own:** `kOmegaSSTBase.C` ends every `k` solve with
  `bound(k_, kMin_)`, and `bound` replaces a negative cell by the **average of its neighbours** before
  flooring at `kMin` (`isf = max(max(isf, fvc::average(max(vsf,min))*pos0(-isf)), min)`). It also
  *replaces* the `ω` row in wall cells (`matrix.setValues`) and lags `G`, so its wall-cell `k` equation
  is linear with a non-negative source. We do the opposite on both counts: both terms stay live
  functions of `k` in one Newton residual (deliberately — a frozen `ω` degenerates the row), and we
  constrain the **step** rather than projecting the state. Constraining the step is what locks up; a
  projection cannot. That reframes the choice, and it is the one to make with the user:
  a positivity **projection** after the step (OpenFOAM's answer, and it changes the forward path but
  not the root provided the floor is inactive there) against keeping the step constraint and finding
  why that cell's row is hard to satisfy (its root is **positive**, at `+1.99e-14`). Neither is built.

  **⚠️ (2026-08-10, LATER STILL) THE `k` WALL BC A/B, RUN AS A CONTROLLED PAIR — and the crash is NOT
  fixed by any of it.** Two full 3-rung marches from the initial state, `BFS3D_TURBULENCE_INVERSE=jacobi`,
  everything identical but `BFS3D_K_WALL`, both carrying the four negative-`k` clamps and
  `stop_on_limit_stall=3`:

  | | `dirichlet` (control) | `zerogradient` |
  |---|---|---|
  | rung 1 | 14 steps, 43 cycles, ‖R‖ 5.882e-06 | 14 steps, 45 cycles, ‖R‖ 7.821e-06 |
  | rung-2 step it locks at | **25** | **35** |
  | rung-2 ‖R‖ at lock | **7.316e-02** | **1.257e-02** |
  | rung-2 steps before the guard ends it | 15 | 24 |
  | outcome | `RuntimeError` from `solve_reynolds_continuation` | same |

  - **❌ Zero-gradient does NOT cure the lock-up.** The same fraction-to-the-boundary ratchet appears,
    same β ceiling of 16.0, same ~100×-per-step cap collapse — just later. Consistent with the
    term-by-term account that put the whole wall closure at ~0.87–0.93.
  - **✅ But it is worth keeping on the evidence: a 5.8× lower residual and 10 more steps of rung 2**,
    and the attribution is clean because the clamps are verified neutral (below) and both arms had the
    guard. It also produced five consecutive *uncapped* full steps in rung 2, which the control never
    does.
  - **✅ THE CLAMPS ARE EXACTLY NEUTRAL, end to end.** The control's rung 1 is **14 steps / 43 cycles /
    5.882e-06 / ‖R₀‖ 3.3078e-01 — identical in every digit to the pre-clamp recorded baseline** — and its
    rung-2 steps 25–29 reproduce that baseline's trajectory exactly (step 25 β 0.4682, ‖R‖ 7.567e-02,
    `a_min` 0.004, cap 3.76e-03; then 1.00e-05 → 1.95e-06 → 1.95e-08 → 1.95e-10). The per-guard
    `array_equal` unit tests said this; a real coupled march now confirms it.
  - **✅ `stop_on_limit_stall` worked, on two independent arms** (deleted 2026-10-14 with the cap). The control's rung 2 ends at **15 steps
    instead of the recorded 108**; the zero-gradient arm's at 24. ~90 dead steps saved each time, and the
    run now fails with a `RuntimeError` naming the rung instead of grinding out `MAX_STEPS`.
  - **The dead grind that then moved into `solve_coupled`'s traced finishing solve is CLOSED (2026-09-15):**
    there is no finishing solve any more, so a segment the stall guard ends is judged unconverged and
    raises at once.
  - The step-limit dumps (gone with the cap, 2026-10-14) carried **β and the anchor** (`cap 2.2923e-06 beta 1.754 anchor yes`), so the
    falsifier named against the earlier diagnosis is closed: these dumps can be paired with the linear
    system that produced them.

  **✅ GATES GREEN on all of the above** (CSR level operator, traced trailing inverse, and both march
  changes), with the tiers named because a default-on march guard reaches further than the fast gate:
  - fast gate **967 passed / 1 skipped** (899 unit `-n auto`, 68 integration `-n 1`);
  - `test_coupled_rans` + `test_coupled_amg` + `test_coupled_field_split` + `test_reynolds_continuation`
    — 33 tests, 18 of them `slow` — **33 passed**;
  - `test_coupled_lu` + `test_coupled_ilut` (the latter since deleted along with the ILUT it tested;
    its record is in git history of `solve-direct-preconditioners.md` before 2026-10-17) — **15 passed**. These two matter and were nearly missed:
    they drive `newton_march` with `step_control` + `refresh_preconditioner`, so they pick up
    `stop_on_limit_stall` exactly as the four above do, and they are in neither the fast gate nor the
    list a first pass would think to run;
  - **`-m validation` — 18 passed.**

## Faithful smoothed aggregation — matching PETSc GAMG

- **⚠️ (2026-08-09): making the traced multigrid a FAITHFUL smoothed aggregation, so a
  comparison against PETSc GAMG means something. Uncommitted work sits on `claude/block-aware-aggregation`.**
  Read this before touching `precondition/multigrid.py`'s aggregation path.

  **⚠️ THE EQUILIBRATION HYPOTHESIS WAS WRONG, AND MEASURING IT IS WHAT SETTLED THE COMPARISON
  (2026-08-09).** The suspicion recorded here was that ours and PETSc's were not built on the same
  matrix — `MonolithicVCycleInverse` calls `equilibrate_cell_major` before handing the operator to PETSc, so GAMG
  coarsens a **unit-diagonal** matrix, while `build_convection_hierarchy` coarsened the raw block,
  whose diagonal on `bfs3d`'s `[k, ω]` slice spans **7.96e5×** (1.26e-06 … 1.0). That description of
  the code is accurate, and the σ_max figures reproduce exactly (**4.365** raw against **2.832e3**
  equilibrated, so the standard prolongator step `1.4/σ_max` is ~0.32 for us and ~5e-4 for PETSc).
  The **inference** from it — that no ours-vs-PETSc number meant anything until it was fixed — does
  not survive. `equilibrate=True` is now built and measured, and it makes our hierarchy **worse**:
  8 → 11 cycles on the RCM arm, 5 → 10 on the plain-aggregation MIS arm.


  **❌ "EQUILIBRATION CHANGES THE GRAPH THE COARSENING SEES" — PROPOSED AND REFUTED BY MEASUREMENT
  (2026-08-12). Do not re-propose it; the falsifying run is cheap and is in the tree.** The reasoning
  was sound as far as it went: `symmetrically_equilibrate` returns `(diagonal @ a @ diagonal).tocsr()`,
  **a sparse product stores only entries whose result is nonzero, so it DROPS every explicit zero**
  (verified in isolation), and at `strength_threshold = 0` the aggregation reads the graph and nothing
  else. The dropped counts are large and real — on `bfs3d` at the target-Re cold initial field, the
  trailing `[k, ω]` block loses **26.8 %** of its stored entries to equilibration (5 245 488 → 3 839 276,
  1 406 212 exact zeros) while the leading block loses **0.0 %** (5 662 of 20 981 952).
  **It does not reach the coarsening, because the cell collapse absorbs it**
  (`validation/bfs3d_openfoam/equilibration_graph_effect.py`, running the real `_cell_graph` /
  `_aggregation_edges` / `_mis_aggregate`):

  | trailing `[k, ω]` | cell edges | aggregates | max aggregate |
  |---|---|---|---|
  | raw | 644 166 | 2300 | 45 |
  | equilibrated | 643 967 | 2302 | 45 |

  **199 of 644 166 edges — 0.03 %.** `_cell_graph` sums `\|A_ij\|` over each `block_size²` block, so a
  cell pair keeps its edge if **any** of its entries is nonzero; the dropped degrees of freedom are
  scattered across blocks rather than clustered within them, so almost every edge survives. The coarse
  space is effectively identical. The leading-block control loses **1** edge and its partition is
  bit-identical.
  **So the similarity-transform argument stands, and the recorded explanation — the non-equivariant
  tentative prolongation and the coarse operator it builds — remains the live one for the 5 → 10.**
  **What survives, and it is worth keeping:** *the aggregation is robust to explicit-zero pruning while
  an incomplete factorization is not.* The smoother takes its pattern from the stored pattern directly,
  with no cell collapse to absorb the loss — which is why the same pruning is fatal to `column_reach`'s
  pressure column (see the per-column reach entry under `sparse_jacobian.py`) and inert here. When
  reasoning about a pattern change, **ask which consumer sees it**: a nodal coarsening and a zero-fill
  factorization have opposite sensitivities.

  **Reference numbers, re-measured on the current code** — `bfs3d` `state-00057`, PC β 0.05, the
  `[k, ω]` block **ALONE** (46080 dofs, 4.20M nnz, 91/row), GMRES restart 15 to rtol 1e-8 on the
  TRUE residual, random right-hand side; harness
  `validation/bfs3d_openfoam/trailing_hierarchy_sweep.py`:

  | preconditioner | coarse eq | cycles |
  |---|---|---|
  | PETSc GAMG, plain aggregation, ILU(0) ×4 | 432 | **1** |
  | PETSc GAMG, plain aggregation, ILU(0) ×1 | 432 | **2** |
  | **PETSc GAMG, plain aggregation, point-block Jacobi ×4** | **432** | **2** |
  | ours, MIS, standard prolongator, ×8, EQUILIBRATED | 2150 | 4 |
  | ours, MIS, plain, ×4, raw | 2150 | 5 |
  | ours, RCM, symmetric-part prolongator, ×4, raw | 598 | 8 |
  | ours, MIS, plain, ×4, EQUILIBRATED | 2150 | 10 |
  | ours, RCM, symmetric-part prolongator, ×4, EQUILIBRATED | 598 | 11 |
  | ours, MIS, standard prolongator, ×4, raw / EQUILIBRATED | 2150 | 44 / 44 — both fail (true rel 1.0) |

  **Read the THIRD row, not the first — the like-for-like arm.** Comparing our Jacobi-class smoother
  against PETSc's incomplete-LU measures the smoother, not the hierarchy, and an ILU sweep is a much
  stronger and much less parallel unit of work.

  **✅ THE GAP IS CLOSED: matched to PETSc's algorithm, our V-cycle equals it — 2 cycles against 2,
  on a 436-equation coarse space against 432.** Two differences accounted for the whole of it, and
  neither was scaling. Both were found by reading `agg.c`/`misk.c`/`rich.c` against our code rather
  than by tuning:

  | arm, `[k, ω]` block alone, matched smoother class | coarse eq | cycles |
  |---|---|---|
  | PETSc GAMG, plain aggregation, point-block Jacobi ×4 | 432 | **2** |
  | ours, MIS ×4, damped, no aggressive level | 2150 | 5 |
  | ours + **aggressive level** ×4, damped | **436** | 10 |
  | ours + aggressive level ×8, damped | 436 | 3 |
  | ours + aggressive level + **undamped** ×2 | 436 | 11 |
  | **ours + aggressive level + undamped ×4** | **436** | **2** |

  1. **The aggressive first level, which PETSc applies BY DEFAULT and we did not.**
     `monolithic_vcycle_inverse` never sets `pc_gamg_aggressive_coarsening`, and GAMG's default is
     `aggressive_coarsening_levels 1` with `use_aggressive_square_graph` — so on level 0 it coarsens
     the **squared** graph. That is the entire 5× coarse-space difference: ours coarsened 21×, GAMG
     107×, and with `aggressive_levels=1` ours lands at 436 equations against PETSc's 432.
     **Note `use_aggressive_square_graph` and `aggressive_mis_k` are ALTERNATIVES, not a pair** —
     `mis_k` is read only in the non-squared branch (`agg.c:1314`), so at the default the coarsener
     is plain MIS at distance **1** on the squared graph.
  2. **PETSc's level smoother is UNDAMPED.** `mg_levels_ksp_type richardson` at its default
     `scale = 1.0` (`rich.c:277`) is `x += D⁻¹(b − Ax)`, while ours relaxed by `omega/lam_max` with
     `omega = 0.8`. That is never a *smaller* factor than 0.8 and usually much smaller: `D⁻¹A` has
     unit diagonal blocks, so its eigenvalues average one and `lam_max ≥ 1` always. Dividing by it
     under-relaxes every mode except the extreme one. Worth **10 → 2 cycles** at four sweeps.
     Reachable as `convection_multigrid_cycles(..., smoother_weight=1.0, spectral_damping=False)`; the
     spectral default is unchanged, because an undamped sweep is not a contraction standing alone —
     it does not need to be under a coarse correction and an outer Krylov.

  **Equilibration is NOT among the causes**, and the two that are were invisible until the sources
  were read side by side. The earlier "ours is behind on both axes" reading was correct as a
  measurement and wrong as a diagnosis: it described a hierarchy that was simply not running the
  same algorithm.


  **⚠️ THE SECOND, COARSENING-INDEPENDENT BENEFIT RECORDED HERE WAS A PSEUDO-INVERSE ARTIFACT, AND IT IS
  GONE (2026-08-14).** The claim was that the coarse solve's singular-value truncation is what a badly
  scaled operator defeats first, so rescaling buys accuracy there: on the badly-scaled chain fixture
  (condition ~1.6e8) the one-level direct solve went from 1.2e-09 unscaled to 5.3e-13 rescaled. That is a
  property of the **pseudo-inverse**, which truncates small singular values, and not of the operator. The
  coarse solve is now a factorization (`_dense_inverse`), which the scaling does not defeat, and the same
  fixture reads **2.8e-13 unscaled against 3.9e-13 rescaled** — both at the floating-point floor, the
  unscaled one four thousand times better than it was, and their ordering now noise. The unit test that
  pinned the gap asserted `scaled < raw` and duly failed on the change; it now pins only what remains
  true, that both paths invert `A`. **The surviving case for rescaling is per-cell block conditioning**
  (measured separately, orders of magnitude in the median) and its effect on the coarse operator the
  tentative prolongation builds — not coarse-solve accuracy.
