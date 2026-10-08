# Open directions — making the fluid solve faster

> The forward-looking sibling of
> `solve-refuted-directions.md`. It lives in `.claude/notes/` (tracked, greppable, **never auto-loaded**)
> for the same reason the refutation ledger does: it is read on demand before proposing work, not paid for
> by every session. Each entry below was checked against the refutation ledger, the `solve-*` rules and
> the `-log.md` files on 2026-10-07; none of them is recorded as tried.
>
> **Status discipline.** An entry here is a hypothesis with a pre-registered measurement. When one is
> measured it moves: a win becomes a rule entry (and a default, if it earns one), a loss becomes an entry
> in `solve-refuted-directions.md` with its configuration recorded in full. Nothing stays here after it
> has been run. Open one GitHub issue per entry *when its investigation starts*, linking back to this
> file, so the issue tracker carries the work in flight and this file carries the backlog.

## Where the time is — the reading these entries rest on

The recorded numbers (`solve-march.md`, `solve-refuted-directions.md`):

| case | cells | dofs | wall | compile | Krylov matvec |
|---|---|---|---|---|---|
| pitzDaily coupled RANS | 12 225 | ~73k | 363 s (ramp arm) | 396 s | 3.2–9.1 ms |
| bfs3d coupled RANS | 23 040 | ~138k | ~50 min | — | 16.6–33.2 ms |

The matvec actually run is `jax.jvp` through the full residual, which reconstructs nine gradients per
evaluation on pitzDaily. Measured 2026-10-07 on pitzDaily (4-core Linux, jax 0.11.2): the jvp is 44 ms,
while a SciPy matvec on the materialized `J` (7.4M nnz, ~120 per row) is 6.5 ms, about 7× cheaper. bfs3d's
`J` is 47.2M structural nnz (~340 per row). The matrix is not used for the matvec because *materializing*
it costs 165–564 batched JVPs (`sparse_jacobian.py`, coloured probing), which is why the preconditioner is
refreshed only every few steps — and the ledger already found that **staleness, not hard operators, is the
dominant linear-solve cost at low β** (the `InnerIterateCheckpointer` result: 15 cycles stale vs 1 cycle
matched at the march's hardest solve).

So the two costs are coupled: the matvec is expensive because the matrix is expensive to build, and the
preconditioner is stale because the matrix is expensive to build. Entries 1–2 tried to break that
coupling and are closed (below): the coloured probe is already near the forward-mode floor. 4–6 attack the
*number* of linear solves rather than their cost; 7–9 are platform and sweep-level levers.

---

## 1–2. CLOSED — local Jacobian assembly and the materialized-`J` matvec

Both were measured on 2026-10-07 and moved to `solve-refuted-directions.md` ("Local (staged) AD assembly
of the coupled Jacobian…"). Local assembly gains ~1× to ~1.3× at best, not 15–30×, so the per-iterate
build item 2 relied on is not cheap. The numbering below is kept so cross-references stay valid.

## 3. CLOSED — tangent / secant predictor between continuation stations

Measured on 2026-10-07 and moved to `solve-refuted-directions.md` ("Tangent / secant predictor between
continuation stations"). It was already measured on the old ladder (worse at a decade). On the shipped
one-step-per-station ramp an exact predictor changes the starting residual by −2.6 % to +12 %, for one
extra linear solve per station.

## 4. CLOSED — Eisenstat–Walker forcing terms; replaced by 4b

Measured 2026-10-08 and moved to `solve-refuted-directions.md` ("Eisenstat–Walker adaptive forcing
terms"): tightening the inner tolerance buys no nonlinear iterations on pitzDaily, and the schedule would
tighten most of the time. The gain is the loose end, which needs no schedule.

## 4b. Loosen the fixed inner forcing term from 0.3

**Deferred — tracked in #637** (needs bfs3d and a robustness check before any default moves).

**What.** Raise `LinearSolveSettings.rtol` (the `_VCYCLE_LINEAR_SOLVE` / `VCYCLE_LINEAR_SOLVE` family
value, and the case files' `linear_solve.rtol`) from 0.3 to ~0.6.

**Why.** On pitzDaily, 0.6 took 166 restart cycles against 203 (−18 %), 0.9 took 161, with identical steps,
inner-solve counts and `x_r/h` (table in the ledger entry above). 0.3 was calibrated on bfs3d's
multigrid family and carried to the others.

**Pre-registered measurement.** bfs3d at `BFS3D_FORWARD_RTOL` 0.3 / 0.6 / 0.9 (one case at a time):
steps, cycles, retries, refreshes, wall, mid-span `x_r/h`. A pass is ≥ 10 % fewer cycles with no extra
steps or retries and the same `x_r/h`; then a library-default change needs the project owner. Watch the
retry ladder: a looser direction near the low-β wall could trip `abort_above_cycles` or the line search.

## 5. Anderson acceleration / nonlinear GMRES around the outer step

**What.** Treat one pseudo-transient Newton step as a fixed-point map `x ↦ G(x)` and accelerate the
sequence with Anderson mixing (depth 3–5, damping 1), or NGMRES (Washio & Oosterlee; De Sterck). Vector
operations only; one small least-squares per step.

**Why it should win.** The ladder descends β one notch per outer step with `α_min = 1.000` on eleven of
twelve steps — the control "had no reason for caution and crawled anyway". Anderson extrapolates
exactly such a well-behaved sequence. The notes mention Aitken/Anderson once, in passing, as something
that *could* sit on the SER ramp; it was never built.

**What it does not change.** Like every globalization here, the mixing vanishes at the fixed point:
converged state and adjoint unchanged.

**Pre-registered measurement.** pitzDaily ramp arm, steps/cycles/wall, Anderson depth 0 / 3 / 5.
Watch the retry region: Anderson on a stiff sequence can extrapolate into non-descent; keep the line
search as the acceptance gate and reset the history on any rejected step.

**Probed 2026-10-08 (`validation/pitzdaily_openfoam/anderson_replay_probe.py`) — modest, tail-only.**
Replay on the shipped pitzDaily march (`case.yaml` as shipped, 31 steps / 203 cycles / `x_r/h` 8.0686,
every step checkpointed; jax 0.11.2, CPU): at each target-station step the Anderson proposal of depth
1/2/3/5, built from the unaccelerated iterates, scored in one row-scaled measure built at `x_{k+1}`.
⚠️ The step labels below are corrected (2026-10-08): the harness keyed checkpoints by their `step`
field, which counts from zero, so it printed every state one lower than its file index and skipped the
station's first step. The numbers are the harness's; only the labels moved. The harness now keys by file.
- **Steps 18–23** (early target station): proposal ≈ the march's own next iterate (ratio 0.80–1.25), never
  better than `x_{k+2}`; depth 5 sometimes worse. No saving.
- **Steps 24–30** (the linear tail, β at its 0.005 floor, the march's own rate 0.64 → 0.82 per step and
  slowing): every depth beats `x_{k+2}` (ratio 0.55–0.77), i.e. about one saved step per step.
- **Bound:** compounding that factor over the tail gives ~2.5 steps for the march's 6, i.e. ~3–4 outer
  steps and ~14 of 203 cycles (~7 %). Optimistic — a one-step replay on the unaccelerated sequence, not a
  march feeding its own proposals back. The pre-registered march (depth 0 / 3 / 5, history reset on a
  station change or a rejected step, line search as the acceptance gate) is what would settle it, and it
  needs `newton_march` to carry the history: an implementation, not a probe.
- Depth 2–3 is as good as 5; the gain is confined to where the shifted step converges linearly.

**Deferred — tracked in #639** (needs the real march above, and bfs3d, before it is worth building).

## 6. CLOSED — Krylov subspace recycling (GCRO-DR); it found 6b instead

Measured 2026-10-08 and moved to `solve-refuted-directions.md` ("Krylov subspace recycling (GCRO-DR)").
On the replayed sequence of pitzDaily's 42 final-station solves, recycling costs 60–105 % MORE operator
applications than plain GMRES under the same stop, and still more with its recompute charge removed. At
`rtol = 0.3` a solve is ~12 iterations; there are no slow modes worth carrying.

## 6b. The GMRES stopping rule over-solves about 4×

**What.** Stop the inner Krylov solve on its residual alone, tested every iteration, instead of
`lineax`'s rule (tested only at a restart boundary, and only once the solution has also stopped moving
over a whole cycle). In-house restarted GMRES, or a `lineax` subclass; the adjoint's transpose solve is
untouched.

**Why it should win.** On the same 42 replayed systems (`krylov_recycling_probe.py`, exact reproduction
of every recorded cycle count) the march's solver used **2186** applications of `A M`; a residual-only
test at restart boundaries needs **890**, and one checked every iteration **514**. Krylov work is ~half
of pitzDaily's wall (fit over the baseline log: ~4.4 s per restart cycle against a ~2 s per-step
constant, 4-core Linux). Every solve pays at least two full restart cycles today; the median one needs
about one. This plausibly IS 4b's gain: loosening `rtol` mostly makes the solution-change test easier.

**What it does change.** The inner corrections become less over-solved (the march's second cycle drives
the residual far below 0.3), so the inner Newton loop may take more iterations or clip differently. That
is exactly what a single-system replay cannot see, so it needs a march.

**Pre-registered measurement.** pitzDaily and bfs3d marches, shipped stop vs per-iteration residual stop
at 0.3 (and 0.6 on pitzDaily, to see whether 4b survives it): steps, inner solves, cycles AND operator
applications (cycles are no longer comparable across stops), retries, refreshes, wall, `x_r/h`.

**✅ Measured on pitzDaily 2026-10-08 — 1.88× faster wall, same root.** `residual_stop_gmres`
(`solve/linear.py`, opt-in) against the shipped stop, both through the study path
(`PITZ_KRYLOV_STOP=residual` / `shipped`), back to back, shipped `case.yaml`, jax 0.11.2, CPU, 4-core
Linux, one run each (this case's march is deterministic in its counts):

| | shipped stop | residual stop |
|---|---|---|
| outer steps | 31 | 31 |
| inner solves | 99 | 137 (+38 %) |
| applications of `A M` | 4931 | **2147 (−56 %)** |
| mid-step preconditioner refreshes | 12 | 4 |
| wall | 1306 s | **693 s** |
| final `|R|` / `x_r/h` | 7.818e-06 / 8.0686 | 7.700e-06 / 8.0686 |

The control reproduces the shipped march exactly (31 / 99 / 203). The looser corrections cost about one
inner Newton iteration per step and the cheaper solves pay for it twice over. ⚠️ **Not a pure stop
effect:** the residual stop reports cycles run where `lineax` reports cycles run minus one, so the
`refresh_on_cycles = 3` trigger fired 4 times against 12; part of the wall gain is fewer rebuilds.
**Still open before any default moves:** bfs3d (its mesh is not in this container), the slow and
validation tiers, the cost thresholds' re-calibration, and the follow-ups below.

**Follow-ups planned on the replay harness, then marches:** minimize the residual in the stop's own
(row-scaled) measure rather than the 2-norm; sweep the restart length; re-measure 4b and the
Eisenstat–Walker refutation under the new stop (both were measured under `lineax`'s, whose
solution-change test is likely most of what loosening `rtol` relaxed).

## 7. Mixed precision: float64 outside, float32 inside the preconditioner

**What.** Keep the residual, the Krylov true-residual recurrence, the outer stopping test and the
adjoint in float64. Store the preconditioner's `J`, hierarchy, smoothers and V-cycle in float32, with
casts at the apply boundary.

**Why it should win.** Halves memory traffic in the part of the solve that is bandwidth-bound, which
is the whole game on a GPU; on CPU the gain is smaller but non-zero. A float32 preconditioner is still a
*fixed linear map*, so plain GMRES and `jax.linear_transpose` remain valid — the three properties
`preconditioning.md` relies on survive.

**What it does not change.** The preconditioner never enters the state or its gradient; a rounding
change inside it changes Krylov convergence only. `solve-amg-multigrid.md` already notes the one place
a float32 hierarchy *forces* a change (the small-block pivoting at §1313): treat that as the checklist.

**Pre-registered measurement.** bfs3d, matched preconditioner at `state-00049`: cycles and apply wall,
float64 vs float32 PC. Pass is cycles within +10 % at lower wall. The real measurement is on a GPU
(next entry).

## 8. The GPU path, and what it rules out

Not a new idea — it is the stated plan — but worth stating as a solver direction because it decides
others. bfs3d at 138k dofs should be of order one second per Newton step on a single modern GPU, which
would make the whole march minutes rather than fifty. Every host-callback inverse (PETSc GAMG via
`pure_callback`, scipy LU) is serialized through the host on that path, which is why the traced
hierarchies (`SimpleSmoothed`, `MaterializedBlockPreconditioner`) exist. Consequences for this file:
entry 7 compounds on GPU and is worth more there than its CPU measurement will show;
entry 6 is a JAX solver by necessity; a GPU-friendly smoother (coloured Gauss–Seidel, Chebyshev where
the operator allows, or more ILU-free sweeps) replaces ILU(0), whose quality the ledger credits to a
sequential global sweep that does not parallelize.

**Measurement.** Unavailable until a GPU environment exists (`solve-amg-multigrid.md` §1958). Record
the first GPU march's step/cycle counts against the CPU march's — counts should match; clocks are the
result.

## 9. Batch the parameter sweep with `vmap` over the whole Newton step

**What.** For calibration and gradient sweeps (`uvreactor_openfoam/gradient_sweep_calibration.py`,
Reynolds sweeps), `jax.vmap` the compiled Newton step over 4–8 parameter values and march them in
lock-step, with per-member convergence masks.

**Why it should win.** The ledger already measured per-call overhead amortizing under `vmap`
(9.08 → 3.22 ms per matvec in 2D, 33 → 17 ms in 3D). On GPU the amortization is far larger because
each member is too small to fill the device alone. Independent solves at nearby parameters also share
their compile.

**Caveats.** `RootSolver` reads residual norms back as Python numbers and refuses to run under
`vmap`/`jit` by design (`steady_state_solving.md`); this needs a batched march driver with traced
convergence masks — a new driver, not a wrapper. Members converging at different step counts waste
work on the finished ones; members starting closer together would help (entry 3, a predictor, is closed).

## 10. Nonlinear elimination of the ω stiffness

**What.** Before each global Newton step, run a few cheap local nonlinear solves of the ω equation
(flow and `k` frozen) in the cells the cell-block SVD flags (`σ_min < 1e-3`, ~1.5 % of bfs3d), then
take the global step from the eliminated state (Cai & Keyes, nonlinear elimination / ASPIN family).

**Why it might win.** The ledger's finding that *the cell block is weakly coupled in ω everywhere* and
near-singular in a coherent low-`k` region is a *nonlinear* preconditioning argument: the linear PC
cannot see a direction the local block does not contain, but a local nonlinear solve along the
transport direction can remove the stiffness before the Jacobian is formed. It is the one idea the
measured data points at that has not been tried; the notes name it once as a possibility.

**What it does not change.** At the root the elimination is the identity, so state and adjoint are
unchanged — but only if the eliminated residual is the *same* residual; do not hand the IFT a different
operator (the warm-start refutation is the cautionary tale).

**Risk.** Highest in this file; measure on the captured hard iterates before touching the march.

---

## Smaller housekeeping

- **Compile on the critical path.** 396 s of XLA compilation before step 1 on pitzDaily. Ahead-of-time
  lower-and-compile the *next* station's program (new viscosity leaf, new measure) in a background
  thread while the current station marches; the persistent cache already makes repeat runs cheap, so
  this is for first runs and parameter changes.
- **First-order seed for the second-order residual.** March the `FirstOrderUpwind` residual to a loose
  tolerance and hand the state to the second-order residual as a seed (defect-correction seeding). Not
  recorded as tried; cheap to measure on pitzDaily's opening residual.
- **Gradient reconstruction `sweeps`.** `schemes.md` is explicit that `sweeps=4` must not be lowered
  globally; the open question is whether the *tangent* reconstruction inside the JVP needs the same
  count as the primal (still open: entry 2, which would have made this moot, is closed).

## Where this file does not go

Not `.claude/rules/` (it would auto-load into every session); not `docs/` (user-facing, and these are
hypotheses); not only an issue (an issue is not greppable from a checkout, and the project's rule is
that findings live in tracked files). The right shape is this note as the backlog, one issue per entry
once it is in progress, and the result filed in a rule or the refutation ledger when it is measured.
