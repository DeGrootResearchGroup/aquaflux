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

(Continued below.)
