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

(Results below.)
