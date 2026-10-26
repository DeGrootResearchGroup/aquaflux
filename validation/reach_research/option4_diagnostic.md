# Option 4, Step 0: can a preconditioner fitted from the reach-2-probed Jacobian work on bfs3d?

Status: in progress (2026-10-26). Measurement record for the project owner; not yet reflected in the
rules or notes.

## The question

On bfs3d, a preconditioner fitted from the Jacobian with only the pressure column probed at reach 2
("`p`") stalls the hard inner solves, where one fitted from the uniform-reach-3 Jacobian ("`3`")
converges. Two readings:

- **H1, the fitted hierarchy is fragile.** `J_p⁻¹` would precondition the exact operator `J₃` well
  enough, but the fixed one-V-cycle SIMPLE-smoothed leading inverse fitted from `J_p` is a poor
  approximation of it. A more robust leading preconditioner (option 4) could then recover reach 2.
- **H2, the premise fails.** `J₃ J_p⁻¹` itself has outlying eigenvalues, so no preconditioner fitted
  from `J_p`, however good, can do much better than the one we have, and option 4 cannot fix `p@drop`.

## Pre-registered reading (written before any 0a run)

**0a.** At bfs3d step 5, inner solve 2, re-fit the split from `3`, `p` and `p@drop`, and raise the
leading `SimpleSmoothed`'s fixed V-cycles per application (`cycles`: 1, 2, 4, 8, 16; trailing block
unchanged). Each arm reports raw restart cycles at the march tolerance (rtol 0.3, cap 16), the
row-scaled residual of the tight solve (rtol 1e-5), and **how close the leading inverse `L` is to the
inverse of the block it was fitted to**: `|b − A_fit L b| / |b|` on a seeded random `b` and on the
solve's own right-hand side (the harness's `leading inverse` line).

- **H1**: arm `3`'s count falls with cycles, and `p@drop`'s falls toward it (within 1.5x of arm `3` at
  the same cycle count, i.e. ≤ 12 at the march tolerance if arm `3` stays at 8).
- **H2**: `p@drop`'s count stays flat or rises (or stays at the cap) while arm `3`'s falls, **and** at
  the highest cycle count the leading inverse is close to the fitted block's inverse (fitted-block
  residual ≤ 1e-3 on both right-hand sides). If the leading inverse has not converged to its fitted
  block by 16 cycles, the sweep did not reach `J_p⁻¹` and H2 cannot be read from it.
- Anything else (both flat; `p@drop` falls but stays above 1.5x) is reported as such, not forced into
  H1 or H2.

**0b (only if H1).** Re-fit `p@drop`, keeping one fitted component from a uniform-reach-3 re-fit on the
same frozen coarsening. A component carries the failure if swapping it alone brings `p@drop` within
1.5x of arm `3` (≤ 12 raw cycles at step 5 inner 2).

## Configuration common to every table

- Library and harness: branch `claude/reach-option4-diagnostic`, from `8dd92e7` (PR #837's head, which
  has `main`'s #829 and #832); the harness commits are this branch's. Nothing under `aquaflux/` changed.
- Machine: 4-core Linux container, 15 GiB, jax 0.12.0.1, one case at a time through
  `validation/run_case.sh`, one or two arms per process.
- Shipped bfs3d `case.yaml`: `FieldSplit(SimpleSmoothed, JacobiSmoothed)`, float32 split,
  `PivotFloor(0.15)`, leading `SimpleSmoothed` with `cycles` 1 (the class default; the file does not set
  it), `sweeps` 2, `pressure_sweeps` 2, `smoother_weight` 1.0, `pressure_smoother_weight` 1.0 (default),
  Frobenius on both halves (defaults), `block_splitting`, strength 0.25, 5 levels, frozen coarsening.
  The linear solve is `LinearSolveSettings(rtol 0.3, restart 15, max_restarts 14, stop residual)`. Both
  the march-tolerance and the tight (rtol 1e-5) solves stop at **16 raw cycles**, the cap.
- States: a uniform-reach-3 march captured for this study (`BFS3D_COLUMN_REACH=0
  BFS3D_CHECKPOINT_KEEP=60 BFS3D_INNER_DUMP_ABOVE=1`, 2026-10-26): 17 steps, `cum 143` cycles, mid-span
  `x_r/h` 8.3611, as recorded before. The solve replayed is **step 5, inner solve 2** (station 4,
  β 0.00617, linearized at the iterate after inner 1), re-fitted at the step's start φ as the ramp march
  does, unless a table says otherwise.
- Replay check against the capture march's log: arm `3` replays step 5 inner 2 at the march's own 6 raw
  cycles (logged 5) and step 10 inner 0 at its 4 (logged 3). Step 6 inner 1 replays at 5 where the
  march ran 6: its inner 0 took α 0.5, so the march solved inner 1 at the post-setback tolerance 0.1,
  which the harness does not reproduce (it solves every arm at 0.3, so arms stay comparable).
- Cycles are raw restart cycles (the march log prints one fewer). "Row-scaled" is the step's own
  measure, `|A d + g| / |g|`. "Tight" is the same solve at rtol 1e-5, read after its cycles.

## Baseline re-established on this tree (step 5, inner 2)

| arm (fit) | cycles, rtol 0.3 | tight: cycles / row-scaled | at `1c5a187` (proposal) |
|---|---|---|---|
| `3` | 6 | 16 / 1.35e-4 | 8 / 7.3e-4 |
| `p` | 16, stagnant | 16 / 2.48 | 16 / 2.72 |
| `p@drop` | 14 | 16 / 8.74e-2 | 16 / 0.39 |
| `p@fold` | 6 | 16 / 1.34e-4 | 8 / 1.7e-3 |

The pattern reproduces (the fold is harmless; the drop is what costs) but every number moved after
#829/#832: arm `3` is two cycles cheaper, and `p@drop` now degrades (14 cycles, tight 8.7e-2) rather
than reaching the cap. The 1.5x criterion below is therefore read against arm `3`'s **6**, i.e. ≤ 9.
The stalled residual is in the continuity rows, ~80 % of it in the 41 % of cells above 10:1 (arm `p`:
1.81 of p's 2.30; `p@drop` tight: 6.7e-2 of 8.4e-2).

## 0a: raising the leading inverse's V-cycles (step 5, inner 2)

Arms `3` and `p@drop`, leading `cycles` edited with `REACH_FOLDING_LEADING=cycles=N`, everything else as
shipped. The leading-inverse line is `|b − A L b| / |b|` for the leading block `A` the inverse was fitted
to (on the solve's own right-hand side, and on a seeded random one).

| leading `cycles` | arm | cycles, rtol 0.3 | tight row-scaled | `L` vs its fitted block: rhs / random |
|---|---|---|---|---|
| 1 | `3` | 6 | 1.35e-4 | 0.92 / 2.18 |
| 1 | `p@drop` | 14 | 8.74e-2 | 0.93 / 2.18 |
| 4 | `3` | **16, stagnant** | 2.77 | 0.52 / 0.90 |
| 4 | `p@drop` | 16, stagnant | 2.73 | 0.53 / 0.91 |

**Reading against the pre-registration: neither H1 nor H2; the sweep cannot be read.** Arm `3`'s count
does not fall with cycles: four V-cycles break the shipped arm itself (16, stagnant, residual above the
right-hand side). And the stacked V-cycle is not approaching the inverse of its own fitted block: four
cycles leave half the right-hand side (0.52), where the H2 reading needed ≤ 1e-3 at sixteen. A likely
reading (an inference, not measured mode by mode): the one-cycle leading map is a useful preconditioner
but a poorly contracting stationary iteration, and repeating it spreads the eigenvalues of
`I − A L` that lie near or outside the unit circle into outliers the outer GMRES cannot absorb. Either
way, "raise the cycles until `L ≈ J_p⁻¹`" is not available with this smoother, on either arm, and the
proposal's 0a premise ("at 16 cycles the leading inverse is close to `J_p⁻¹`") does not hold here
(16 cycles: see the arm-`3` follow-up below).

## The question re-posed: which operator does each fit precondition? (step 5, inner 2)

Since 0a cannot reach `J_p⁻¹` through the V-cycle count, the harness was given two more levers
(`REACH_FOLDING_OPERATORS`, `REACH_FOLDING_FIT_AT`; this is a change of method from the proposal's 0a,
flagged here). The preconditioner stays as fitted; the **operator** the outer GMRES inverts is varied:

- `iterate` — the march's own: the exact shifted Jacobian at the solve's iterate (two inner solves past
  the fit). Every table above used this.
- `start` — the exact shifted Jacobian `J₃(φ)` at the step's start, the state every arm is fitted at.
- `start-drop:p` — `J₃(φ)` with the 4 535 188 entries the `p` probe drops removed: exactly the operator
  the `p@drop` arm is fitted from, `J_drop(φ)`, so a solve against it asks how well that fit inverts its
  own operator.

and, separately, the fit is moved to the iterate (`REACH_FOLDING_FIT_AT=iterate`, so the preconditioner
has no lag behind its operator; the shift is the step's either way). Right-hand side: the march's.

| fit state | operator | arm `3`: cycles; tight cycles / row-scaled | `p@drop`: cycles; tight cycles / row-scaled |
|---|---|---|---|
| φ (march) | `iterate` | 6; 16 / 1.35e-4 | **14; 16 / 8.74e-2** |
| φ | `start` = `J₃(φ)` | 3; **8 / 9.9e-6** | 3; **11 / 1.0e-5** |
| φ | `start-drop:p` = `J_drop(φ)` | 3; 16 / **0.141 (stagnant)** | 3; 16 / **0.133 (stagnant)** |
| iterate | `iterate` | 3; 14 / 9.5e-6 | **3; 16 / 3.5e-5** |

Three readings, each from this one state:

1. **At its own fit state, the `p@drop` fit preconditions the exact `J₃` about as well as the reach-3
   fit** (11 against 8 cycles to 1e-5; 3 against 3 at the march tolerance; with no lag, 3 against 3
   and 3.5e-5 against 9.5e-6 after 16/14 cycles). That is well inside the pre-registered 1.5x at the
   march tolerance. So the claim behind H2 — that anything fitted from `J_p` must precondition `J₃`
   badly — is refuted here, for the fitted hierarchy we have.
2. **`J_drop` itself is a hard operator: neither fit inverts it.** Both stagnate near 0.13–0.14 tight,
   the drop's own fit included. The hierarchy fitted from `J_drop` is therefore *not* close to
   `J_drop⁻¹`. It approximates `J₃⁻¹` better than it approximates the operator it was fitted from,
   which is why reading this as "the fitted hierarchy versus `J_p⁻¹`" (H1 against H2, as posed) does
   not describe what was measured. Whether the exact `J_drop⁻¹` would precondition `J₃` badly (H2 for
   the exact inverse) is not measured; given (2) it may well, and it does not matter for a fitted
   preconditioner.
3. **What fails is the combination of the drop with the lag.** The march re-fits at the step's start
   and then runs the inner loop on that fit. Two inner iterates later, the reach-3 fit loses a factor of
   2 at the march tolerance (3 → 6), and the drop fit a factor of ~5 (3 → 14), with three orders of
   magnitude at the tight tolerance (1e-5 → 8.7e-2). Without the lag the drop costs nothing at the march
   tolerance. Not distinguished by these runs: whether the drop makes the fit more sensitive to *any*
   change of operator, or to the specific change across the inner loop (the iterate moved `k` enough
   that the pivot floor moved 85 `k` rows there against 32 at φ).

### The same at other states, and for the arm the reach-2 probe actually gives (`p`, drop and fold)

Cycles at rtol 0.3; tight row-scaled after 16 cycles unless a count is given.

| state | arm | fit at φ (march), operator iterate | fit at φ, operator `J₃(φ)` | fit at iterate (no lag) |
|---|---|---|---|---|
| step 5, inner 2 | `3` | 6; 1.35e-4 | 3; 8 cycles to 9.9e-6 | 3; 14 cycles to 9.5e-6 |
| step 5, inner 2 | `p@drop` | 14; 8.74e-2 | 3; 11 cycles to 1.0e-5 | 3; 3.5e-5 |
| step 5, inner 2 | `p` | **16, stagnant; 2.48** | 3; 11 cycles to 9.0e-6 | **3; 1.03e-5** |
| step 6, inner 1 | `3` | 5; 2.61e-4 | — | 3; 6.77e-5 |
| step 6, inner 1 | `p@drop` | 8; 1.08e-2 | — | 4; 7.13e-5 |
| step 6, inner 1 | `p` | (see the attribution table) | — | 3; 6.30e-5 |
| step 10, inner 0 (iterate = φ) | `3` | 4; 1.91e-4 | (same operator) | (same fit) |
| step 10, inner 0 (iterate = φ) | `p@drop` | 5; 4.94e-4 | (same operator) | (same fit) |

At every state measured, the reach-2 fit (`p`, or its drop alone) **made at the state it is applied
at** is level with the reach-3 fit at the march tolerance (3–5 against 3–4 cycles) and within a factor
of ~3 at 1e-5. It loses heavily only when applied two inner iterates (step 5) or one (step 6) after its
fit, which is how the dual-time inner loop uses it. Step 10's first inner solve has no lag and loses
only one cycle and a factor 2.6 at the tight tolerance (the "rate halves" of the record).

## Attribution: which fitted component carries the lagged failure (step 5, inner 2)

⚠️ **Deviation from the pre-registration, flagged:** 0b was pre-registered to run only if 0a said H1,
and 0a was unreadable. It is run anyway, on the failure actually observed (the lagged `p` fit against
the march's operator), because it is the same experiment and decides where option 4 would have to act.
The arm is `p` (the probe the march would use), not `p@drop`. Each row re-fits `p` at φ and puts back
one fitted component from a uniform-reach-3 re-fit at φ on the same frozen coarsening
(`ComponentSwap`); everything else is `p`'s own. Pre-registered bar: within 1.5x of arm `3`'s 6 cycles,
i.e. ≤ 9.

| `p` with this component from the reach-3 fit (step 5, inner 2) | cycles, rtol 0.3 | tight row-scaled (16 cycles) |
|---|---|---|
| none (`p`) | 16, stagnant | 2.48 |
| finest leading level operator (the smoother's residual products) | 16, stagnant | 1.72 |
| every finest-level SIMPLE piece (`F` inverse, `M G`, `D`, `C`, Schur diagonal) | 16, stagnant | 2.48 |
| the finest Schur diagonal inverse alone | 16, stagnant | 2.48 |
| the finest `C` alone | 16, stagnant | 2.48 |
| the split's coupling block | 16, stagnant | 1.08 |
| **every coarse level (Galerkin operators, coarse solve, their SIMPLE pieces)** | **6** | **2.98e-4** |
| all of the above (null control: must reproduce arm `3`) | 6 | 1.47e-4 (arm `3`: 1.35e-4) |

At step 6, inner 1 the same swap repairs `p` again: `p` 6 cycles / 4.77e-3, `p` with the coarse levels
from the reach-3 fit 5 / 2.80e-4, arm `3` 5 / 2.61e-4. (At `1c5a187` the record had `p` at this state
16, stagnant at 0.21; on this tree it degrades rather than fails.)

**The coarse levels carry the failure.** Swapping them alone meets the bar exactly (6, the arm-`3`
count); nothing on the finest level does, the Schur diagonal included (so H1-a, the proposal's
over-relaxed Frobenius pressure sweep, is not what fails here). The null control reproduces arm `3`, so
the swap machinery is sound. The fine-piece swaps that leave every digit unchanged did move their arrays
(the census below: the `p` probe drops 773 476 of `C`'s 1 311 372 entries, 3.5e-3 of its norm); those
values simply do not matter to this solve.

Why the coarse levels: the fitted fine operators of `p` and `3` differ by 3.7e-3 (largest relative
change of the finest operator's values), but their coarse levels by 5.3e-2, fourteen times more, at the
same aggregates. A Galerkin coarse operator `Pᵀ A P` over these aggregates sums each aggregate's
couplings, and on the pressure rows those sums nearly cancel, so removing couplings that took part in
the cancellation moves the coarse operator far more, relatively, than the fine one. (An inference from
the two numbers; the cancellation itself was not measured.)

Census of what the `p` probe drops at φ (`validation/reach_research/dropped_census.py`, step 5; the numbers were taken with its scratch first draft, same computation): per
row field of the `p` column, 773 476 of 1 311 372 entries in every row field (667 808 of 1 122 986 for
ω), of norm 4.6e-3 (u), 9.9e-4 (v), 1.4e-3 (w), 3.5e-3 (p), 4.1e-3 (k), 8.5e-4 (ω) of each block.

## Settings screen on the leading smoother (step 5, inner 2)

The proposal's cheap screen, on `p` with arm `3` beside it in the same process, each setting at two
smoother sweep counts (`sweeps` 2, shipped, and 4; never rank smoother arms at one). Edits through
`REACH_FOLDING_LEADING`; Frobenius-off is moved together with the pressure weight, as the rules require.
Cycles at rtol 0.3 / tight row-scaled after 16.

| leading edit | `sweeps` | arm `3` | `p` |
|---|---|---|---|
| none | 2 | 6 / 1.35e-4 | 16 / 2.48 |
| `pressure_smoother_weight=0.7` | 2 | 6 / 1.18e-3 | 16 / 2.28 |
| `pressure_smoother_weight=0.85` | 2 | 6 / 4.46e-4 | 16 / 2.34 |
| `pressure_sweeps=4` | 2 | 5 / 4.43e-5 | 16 / 2.55 |
| `schur_frobenius=false, pressure_smoother_weight=0.7` | 2 | 6 / 2.19e-4 | 16 / 2.34 |
| none | 4 | 6 / 1.51e-4 | 12 / 8.77e-2 |
| `pressure_smoother_weight=0.7` | 4 | 6 / 2.36e-4 | 16 / 3.00 |
| `pressure_smoother_weight=0.85` | 4 | 6 / 2.97e-4 | 15 / 0.189 |
| `pressure_sweeps=4` | 4 | 5 / 4.96e-5 | **10 / 3.14e-2** |
| `schur_frobenius=false, pressure_smoother_weight=0.7` | 4 | 6 / 1.64e-4 | 13 / 6.32e-2 |

No smoother setting brings the lagged `p` fit within the bar (≤ 9); the best, four sweeps of four inner
pressure sweeps, reaches 10 at about twice the fine-level smoothing work per application. Damping the
pressure sweep (the H1-a remedy) makes it worse at both sweep counts. Arm `3` is insensitive to all of
it. This agrees with the attribution: the fine-level smoother is not where the failure lives.

## pitzDaily control (step 5, inner 2)

States from a shipped pitzDaily march captured for this study (`PITZ_CHECKPOINT_KEEP=60
PITZ_INNER_DUMP_ABOVE=1`, 2026-10-26, same tree): 21 steps, `cum 102`, `x_r/h` 8.069 (OpenFOAM 7.741);
its figure step then failed for want of matplotlib, after the march. pitzDaily ships every column at
reach 2 (`column_reach (2, 2, 2, 2, 2)`). Step 5 is station 4, β 0.00617, solved at the iterate after
inner 1 (2 cycles, α 0.25). Cycles at rtol 0.3 / tight row-scaled after 16.

| fit | `3` | `2` (shipped) | `p` | `p@drop` |
|---|---|---|---|---|
| at φ (march) | 4 / 2.92e-4 | 4 / 3.03e-4 | 4 / 2.88e-4 | 4 / 2.99e-4 |
| at the iterate | 3 / 1.01e-5 | 3 / 1.09e-5 | — | — |

On pitzDaily the lag costs every fit the same one cycle, and the reach-2 fits lose nothing either way:
the drop does not make its fit lag-sensitive there. (The march logged 5 for this solve, 6 raw, after two inner solves at α
0.25; the replay's 4 is at rtol 0.3, not the post-setback 0.1, as the earlier record also found.)

(Continued below.)
