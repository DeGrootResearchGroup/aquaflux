---
paths:
  - "aquaflux/precondition/multigrid.py"
  - "aquaflux/precondition/hierarchy_inverse.py"
---

# Rules — `aquaflux/precondition/` the traced multigrid (and the record of the deleted monolithic AMG)

> ⚠️ **THE MONOLITHIC PETSc AMG IS DELETED (2026-10-10), with the `petsc` dependency.** There is no
> `amg_preconditioner.py`, `MonolithicVCyclePreconditioner`, `MonolithicVCycleInverse`, `monolithic_vcycle_inverse`,
> `ShiftedCellMajorOperator` or `MonolithicVCycle`; `tests/unit/test_amg_preconditioner.py`,
> `tests/unit/test_cell_major_operator.py` and `tests/integration/test_coupled_amg.py` went with them, as
> did the PETSc-only harnesses (`petsc_free_march.py`, `preconditioner_sweep.py`, `zero_pattern_pivots.py`,
> `inner_iterate_probe.py`). The shared base `MaterializedJacobianPreconditioner` lives in
> `precondition/materialized_preconditioner.py` (documented in `solve-direct-preconditioners.md`). Its
> record here was deleted 2026-10-17 except for the traps that transfer (first section below); a PETSc/GAMG
> arm still quoted in a live-code entry is a comparison that cannot be re-run from this tree. Recover the
> code from git history before `2026-10-10`. Why it went: no shipped case or default selected it, and the
> traced nodal hierarchy matches GAMG on the `[k, ω]` block.

> ⚠️ **`coupled_continuation`, `coupled_lu_continuation`, `coupled_amg_continuation`, `lu_beta_tracking_refresh` and `amg_beta_tracking_refresh` no longer exist (deleted 2026-09-14, #371).** Entries below that name them are dated history. The coupled march is now one builder, `coupled_step`, given a preconditioner value (`BlockDiagonal` or `MaterializedJacobian(FieldSplit | BlockInverse)` (a `CompleteLu` inverse existed until 2026-10-11)), and a march that keeps its preconditioner current runs on a session (`open_session`) — see the rename table in `.claude/rules/turbulence.md`.

> Split out of `solve.md` (2026-08-18) to keep routine `aquaflux/solve/` work from loading the full
> AMG/multigrid investigation narrative. See `solve.md` for the package-wide contracts, current
> configuration, and general binding decisions this file assumes.
>
> **Split 2026-10-17.** The dated investigation behind the live traced multigrid moved, verbatim, to
> `.claude/notes/solve-amg-multigrid-log.md` (outside the auto-loaded tree, so it never auto-loads),
> under headings mirroring this file's. The record of the deleted monolithic PETSc AMG was deleted,
> leaving one line per trap that transfers to live code; its refutations are in
> `.claude/notes/solve-refuted-directions.md`.

## The deleted monolithic AMG — traps that transfer to live code

**Current status (2026-10-17).** Nothing in this package is a monolithic AMG any more: the PETSc GAMG
V-cycle went 2026-10-10, the zero-fill `Ilu0` / `hostilu` smoother and the elimination orderings
2026-09-13 (#371), Vanka 2026-08-15, the threshold ILU 2026-08-18 and the complete LU 2026-10-11. Both
flagship cases run `MaterializedJacobian(FieldSplit(SimpleSmoothed, JacobiSmoothed))`. The section that
recorded the monolithic AMG's smoother, fill, ordering, coarse-size and aggregation campaigns is
deleted; the ledger entries that record why each lost are, in `.claude/notes/solve-refuted-directions.md`:
"PETSc (monolithic GAMG V-cycle, UMFPACK complete LU) — DELETED as dominated", "Complete LU (SciPy
SuperLU) — DELETED as dominated", "Incomplete-LU preconditioning of the pitzDaily flow block — CLOSED as
a direction", "Coupled preconditioner options deleted as dominated (#371)" (`Ilu0`, the orderings) and
"Low-β directions already measured out" (Vanka, plain aggregation, `coarse_eq_limit`). The entries of
that section that concern live code (the SIMPLE-smoothed hierarchy's reach-insensitivity, the probe's
gradient sweeps, the cost-guard split's re-march and reasoning, a refresh's time breakdown) are in
`.claude/notes/solve-amg-multigrid-log.md`, "Preconditioner — monolithic AMG (DELETED 2026-10-10)". The
shared base `MaterializedJacobianPreconditioner` is documented in `solve-direct-preconditioners.md`.

**Traps it taught that still apply:**
- **Any preconditioner that takes its pattern from the STORED sparsity is decided by which entries are stored, and at a degenerate state (near-zero fluxes, a closure switch sitting on its kink) rounding decides that** — an ILU(1) V-cycle read as a 1-cycle preconditioner on a channel fixture only because ~1e-12 garbage survived explicit-zero pruning; treat a preconditioner measured at such a state as unverified.
- **Two independent implementations agreeing is evidence about a SHARED cause, not about which shared thing it is** — the zero-fill failure on pitzDaily was blamed on fill when both implementations also shared the elimination order, which was the lever.
- **Interacting knobs must be varied together** — stencil reach and fill each "did nothing" alone on pitzDaily and together took one matvec; a one-variable sweep measured the wrong pair and was reverted on that basis.
- **A zero-shift measurement is meaningless without its state** — at pitzDaily's cold self-start the Jacobian is nearly singular (a complete LU of it was not an accurate inverse either), while at the converged root the field split converges; the pseudo-transient shift is what makes that Jacobian factorizable at all (harness: `validation/bfs3d_openfoam/zero_shift_adjoint.py`).
- **Nothing cheap computable at build time ranked the factorization arms** — `condest`, diagonal dominance, stationary-sweep contraction and a pivot census each failed, the correlation flipping sign with the shift; settle a preconditioner question on a true-residual solve, and finally on a march.
- **Pressure-last elimination is an ordering for a factorization that KEEPS fill** (eliminating the velocities fills the pressure block with the Schur complement); under zero fill it failed at five of six points. Re-ask it only for a level-of-fill factorization.
- **Sweep preconditioners at the march's HARD states** (the checkpoints identify them by cycle count and clipped α) — an easy operator does not discriminate, and a sweep taken at the converged tail nearly closed the plain-vs-smoothed aggregation question with a tie.
- **With a smoothed prolongator, adding a strength threshold gave dense coarse operators and NaN V-cycles; with plain aggregation the same threshold built a 5-level hierarchy** — the NaN is the smoothing/threshold interaction, so do not re-refute strength-of-connection on smoothed-aggregation evidence.
- **Don't under-materialize** — a fresher Jacobian cut a march's Krylov cycles ~23 % despite more refresh work, so the lever is a cheaper materialize (the probe), not a rarer one.
- **A cycle-count threshold encodes which preconditioner is installed** (a count is preconditioner strength × operator difficulty) — `on_cycles = 10`, set for one arm, fired on a healthy 12-cycle arm and its β escalations killed the march; read a `cycles` retry as a statement about the threshold before reading it as one about the step.
- **Do not instrument a long run with a per-step notification stream** — a log monitor's notifications drove the desktop to ~70 % CPU against the solve; cycles, not the clock, are the cost measure.

### ✅ RESOLVED — the cost guard and the shift escalation are now separate responses (2026-08-17)

**Binding, and it changed shipped behaviour on both cases.** `RetryPolicy.on_cycles` was one number
doing two jobs: it stopped the dual-time inner loop *and* escalated β. It is now
**`abort_above_cycles`**, a cost guard only. `retry_reason` returns `"diverged" | "alpha" | "cycles"`,
and `ESCALATING_REASONS = {"diverged", "alpha"}` — one home, read by both the march and the log — says
which of them raise the shift. A `"cycles"` retry **redoes the step at the shift it already had**, on
a re-fitted factorization (the dual-time loop's mid-step refresh until 2026-10-10; now the march's
`due=True` re-fit before the redo, #290).

- **Why cost must not escalate.** A cycle count says the frozen preconditioner is struggling with this
  operator, not that the step is stiff. The march already has the right response — `refresh_on_cycles`
  rebuilds it (mid-step when this was written; since #290 the step ends and the re-fit precedes the next) — and escalating on the same observation additionally assumes a stiffer operator
  is an easier one, which is false on this case (140 applications at β 0.5 against 32 at 0.05).
  **Stiffness is what `on_alpha` measures, and a step length is dimensionless**, so it needs no per-arm
  calibration. The abort threshold remains a number, but it is now a *resource* decision rather than a
  diagnosis.
- **⚠️ The redo is capped at ONE per step** (`redone_on_fresh` in `march.py`). It differs from the first
  attempt only in starting on the refreshed factorization, so a second at the same shift would repeat
  it exactly — burning the escalation budget on identical attempts and spinning to `max_retries` every
  step.
- **⚠️ `cycle_budget` DEPENDS on this redo and would otherwise silently accept a bad iterate.** It
  deliberately truncates a grinding solve and returns a partial, non-converged iterate *expecting it to
  be discarded*. Pair it with `abort_above_cycles < cycle_budget` or the truncation is simply accepted.
  That pairing was the reason the two jobs were fused in the first place, and it is what made "just
  delete the cost trigger" wrong.
- **`escalates` now means `on_alpha is not None`**, so a cost-only policy needs no readable β leaf and
  `require_shifted` no longer rejects one.

### Other live facts first recorded in the monolithic-AMG section

- **Trap the deleted gates taught, kept because it applies to any future refresh condition:** a
  preconditioner-only `refit_beta_floor` pins a β gate's input below the floor, so a condition nested
  inside a β gate is unreachable in exactly the low-shift tail. Measured on the 3-rung `bfs3d` cold
  march: 91 % of sub-floor steps refreshed nothing while ν_t drifted ~20 % per step, those steps
  carried 47 % of the Krylov cycles, and un-nesting the gates cut the march 480 → 348 cycles.
- **Where a refresh's time actually goes — REPORTED, not inferred (`RefreshTiming`, `solve/refresh_timing.py`).**
  The observer used to receive `(kind, seconds)`, so a breakdown had to be recovered by differencing
  the `full` and `shift` branches across a run. It now receives a record carrying ordered
  `(phase, seconds)` pairs — AMG: `probe` / `assemble` / `refactor` — and `MarchLogger` renders them
  under the `"pc"` detail (`pc full 23.0s (probe 14.6 assemble 3.2 refactor 5.2)`), with any
  unattributed remainder shown as `other` so a breakdown cannot silently fail to add up. The
  factorization preconditioners report no phases (empty tuple), which the record documents as valid.
  Differencing the same 56-step march gave probe ≈ 14.6 s of a 23.0 s `full` (63 %) against 8.4 s for
  `shift` — consistent with the older ~40-of-60 s reading, and the reason the probe is the lever.
- **⚠️ EVERY refresh between two step rows is counted and summed, because more than one fires
  (fixed 2026-08-17).** `on_refresh` used to *overwrite* its record, which was safe only while the
  hook ran once per step. A rebuild triggered by solve cost (`refresh_on_cycles`) fires **inside** a
  step, and a **retried** step runs its inner loop again, so several land between step rows — six in
  one observed step. The old rendering therefore showed one refresh per step however many fired, and
  printed one refresh's seconds as the step's whole preconditioner cost: on that step,
  `pc inner 15.2s` for what was really **65.4 s, 22 % of a 301 s step** rather than 5 %. Seconds and
  phases now sum across the step and a branch that ran more than once is marked `Nx` —
  `pc full inner 3x none 2x 65.4s (probe 11.8 assemble 0.5 refactor 52.0)`. A single refresh renders
  exactly as before. **Any refresh-versus-step cost comparison drawn from a log written before this
  fix understates the refresh side, by however many refreshes that step happened to fire.**

- **Three small clones removed (BUILT 2026-08-15), and one deliberately left.**
  - The SIMPLE smoother's sweep loop is `sweeps_from(level, rhs, guess, count)` in
    `saddle_multigrid._simple_smoothed_cycle`. `smooth` and `smooth_zero` each declared the identical
    10-line `fori_loop` body; they differ only in where they start and how many sweeps that leaves.
    Bit-identical at 1, 2 and 4 sweeps.
  - `block_stencil_colouring` calls `frozen_operator.require_valid_graph` instead of re-inlining three
    of its four checks. That helper takes the caller's name for its messages, which is precisely why it
    is shared; only the `reach` check is the colouring's own. It also loses its underscore, since it now
    crosses a module boundary.
  - `checkpoint._atomically(path, write)` is the one home for write-then-rename. It was written twice —
    once with the reasoning attached and once without, and the copy without it is the one a reader would
    have had to reconstruct the argument for. `write` takes a **path**, matching the serializer contract
    ("write exactly to this path"), so an injected serializer needs no wrapping. Verified that a crashed
    write leaves **no** file that reads as a checkpoint, which is the whole point of the idiom.


## The traced multigrid — in progress

**Current status (2026-10-17).** The traced nodal hierarchy (`JacobiSmoothedInverse` over
`HierarchyBlockInverse`) is built, differentiable, and is the `[k, ω]` trailing inverse of both flagship
cases' field splits. On the `bfs3d` `[k, ω]` block alone it matched PETSc GAMG's matched-smoother arm
(2 restart cycles against 2) once the four differences below were closed. The march failures this
section chased — the rung-2 fraction-to-the-boundary lock-up, the `equilibrate` A/B, the positivity
floor, the per-cell projection, the stall stop — all belonged to the global `k` positivity cap, deleted
2026-10-14 in favour of a per-cell clip with an ω step bound (`solve-march.md`); the `k` trough behind it
was a Reynolds-continuation transient absent from the converged root. Both case files set
`equilibrate: false`; the library default is still `True`, whose stated reason (the Frobenius
singularity guard) is gone, and flipping it is an open shipped-default decision. The full dated record —
the lock-up tables, the floor and projection A/Bs, the `k`-trough and wall-layer investigation, the
`x_r/h` grid-station caveat and the gates run — is in `.claude/notes/solve-amg-multigrid-log.md`,
"The traced multigrid — in progress".

  **Four things closed the gap, and each was found by reading the reference rather than tuning:**
  1. **The aggressive first level.** `monolithic_vcycle_inverse` never sets `pc_gamg_aggressive_coarsening`,
     so GAMG applies its default of one aggressive level over the SQUARED graph. Ours had none:
     21× coarsening against GAMG's 107×, the whole 5× coarse-space difference. (`use_aggressive_
     square_graph` and `aggressive_mis_k` are ALTERNATIVES; at the default the coarsener is plain
     MIS at distance 1 on the squared graph.)
  2. **PETSc's level smoother is UNDAMPED** — `richardson` at its default scale of 1. Ours relaxed
     by `omega/lam_max`, and `D⁻¹A` has a unit diagonal so `lam_max ≥ 1` always: the spectral factor
     can only ever under-relax. Worth **10 → 2 cycles**. Reachable as `spectral_damping=False`.
  3. **A CSR matvec instead of the COO `segment_sum`.** A scatter-add collides on output rows; a CSR
     row walk does not. Measured on the 4.2M-nnz block: `segment_sum` 13.3 ms, `BCOO @ x` 13.4 ms,
     scipy CSR on the host 2.6 ms, **`BCSR @ x` 1.4 ms**. The level operator is applied ~10× per
     V-cycle, so this alone took the apply from **117.8 ms to 13.3 ms (8.8×)**. Landed as
     `_CsrOperator`, which owns its matvec and replaced the loose `row`/`col`/`val` — and was itself
     replaced 2026-10-14 by the padded-row `_PaddedRowOperator` (see "Binding decisions" below). **Not** a jit
     or marshalling problem — both were measured out (1 trace everywhere; a numpy↔jnp round trip is
     0.02 ms), which also retires the older claim that "marshalling is ~half the gap".
  4. **The singularity guard was WRONG, and it was aborting the march.** `_cell_block_inverse`
     tested `|det| < 1e-12 · ‖B‖_F^b`. On the coupled turbulence pair a cell reads
     `[[8.8e-06, 1.7e-12], [-1.3e+03, 1.5e-01]]` — rows differing by **1.5e8** — so `‖B‖_F²` ≈ 1.6e6
     is set entirely by the ω row and the bar lands at 1.6e-06, just above the determinant of
     1.35e-06. The block is **not singular**: on the row-norm (Hadamard) bound `|det| ≤ ∏‖rowᵢ‖` it
     scores **1.2e-04**, eight orders clear. Now tested that way, which is invariant under rescaling
     any row or column where the Frobenius form is not. A structurally empty row is named
     explicitly, since it makes the bound zero and no determinant compares below it.

  **Equilibration: KEEP it, for conditioning, and stop expecting it to fix anything else.**
  - It improves per-cell block conditioning by orders of magnitude in the median, which is why it is kept.
  (Measured 2026-08-10 with `cell_block_scaling.py`; state, β and bundle not recorded, so the figures that
  quantified it are deleted — re-measure before relying on the size of the gain.)
  - It **cannot** change whether a cell block is singular. `det B̂ = det B / |b₁₁b₂₂|`, so
    `det B̂ = 0 ⟺ det B = 0`, and the coupling ratio `|a_kω a_ωk| / |a_kk a_ωω|` measures **identical**
    raw and rescaled (7.369e-04 both). Using it as the fix for the guard was wrong from the start;
    the "4 singular → 2 singular" reading that seemed to support it compared *different states*.
  - It does **not** help the worst cells: on `bfs3d` `state-00057`, symmetric `D B D` leaves cond at 1.22e12
  because it moves the imbalance from the rows into the subdiagonal (`[[1, 1.5e-9], [-1.1e6, 1]]`). The
  next two bullets are the same worst cell at that state.
  - **Independent row/column scaling would**: row-equilibrating the cell block gives cond 1.68e4,
    two-sided gives **2.41**, and both are *exact* rebracketings of `B⁻¹`. Unnecessary at float64
    and 2×2 (relative error is already 2.5e-16), but it becomes mandatory in **float32** — which is
    the GPU case this whole exercise is for.
  - The deeper issue is upstream: the k row has norm 8.8e-06 and the ω row 1.4e+03, so the
    **equations** are eight orders apart before any preconditioner sees them. That is what
    `RowScaledNorm` already recognizes in the convergence measure. Fixing the residual's row scaling
    would help the smoother, the coarsening and any factorization at once.

  **`‖B⁻¹‖ = 9.5e8` (`bfs3d` `state-00057`, symmetrically equilibrated) is not an error and no rescaling
  removes it.** The block is essentially
  lower-triangular (`∂R_k/∂ω ≈ 1.7e-12`), so ω is slaved to k there and a k correction legitimately
  produces one ~1e9 times larger in ω. Whether a *cell-local* smoother should apply that is the real
  open question, and it is the same "ω is not locally determined" theme as the Vanka campaign.

- **A harness kept in the repository is re-adjudicable only while it still RUNS** — six harnesses unpacked a shared `STATES` record positionally and all died at startup when it grew a field; read shared records by name (detail in the log).

  **Harnesses kept (all in `validation/bfs3d_openfoam/`):** `trailing_hierarchy_sweep.py` (the block
  alone, every arm), `cell_block_scaling.py` (per-cell conditioning, raw vs equilibrated).
  (`k_row_scale_probe.py`, which refuted the row-promotion explanation above, was deleted 2026-10-14 with
  the step-limit dumps it read; its method -- exact `∂R_k/∂k` per cell by one-hot Jacobian-vector
  product -- is in git history.)
  `compare.py` gained `BFS3D_TURBULENCE_INVERSE`, `BFS3D_NATIVE_EQUILIBRATE`,
  `BFS3D_DUMP_TRAILING_BLOCK`, a `pbjacobi1` (undamped) smoother arm, march-log **archival**, and a
  banner that records the inverse and all its settings — **including the three knobs that install
  wrappers or change retention (`CHECKPOINT_KEEP`, `INNER_DUMP_ABOVE`, `DUMP_TRAILING_BLOCK`), which it
  used to read but never print, so two differently-configured runs could produce identical banners.**

  **⚠️ FOUR METHODOLOGICAL TRAPS, each of which produced a wrong write-up today:**
  1. **Probe the state the failure happens at.** The refusal fired from a *mid-step* refresh (before #290; it now fires from the re-fit before a step); step
     checkpoints and the inner-iterate dump both miss it (the inner observer writes only after an
     iteration succeeds). Three capture runs were wasted before dumping the operator *before* the
     build, which needs no state and no shift pairing — and even that missed twice, first by
     wrapping only the factory when the refresh goes through `refactor_block`.
  2. **Pair the operator with the right β.** Probing state-N with state-N's β when the failing
     refresh uses state-N+1's is the recorded trap; sweep β instead.
  3. **Never quote an arm at one smoother-sweep count.** The standard prolongator was recorded as
     "void" from its 4-sweep numbers; at 8 it is the best traced arm on the scalar problem. The rule
     cuts both ways — on the flow saddle it fails at 4 sweeps and fails *worse* at 8, which is what
     distinguishes an under-smoothed arm from an amplified one.
  4. **A block-alone probe ties where a march separates.** Raw and equilibrated are both 2 cycles on
     the block and behave differently in a march.

  **The trap, recorded because it nearly landed:** a default-on guard on a *shared* seam is not covered
  by "the tests for the subsystem I changed". `newton_march` has one production caller, but everything
  that reaches `solve_coupled` with an observer picks the new default up. Enumerate by **who calls the
  seam**, not by which file the change is in.


## Faithful smoothed aggregation — matching PETSc GAMG

**Current status (2026-10-17).** `precondition/multigrid.py`'s aggregation path is a faithful port of
PETSc GAMG's: the aggressive first level, undamped Richardson smoothing, the fix-up pass, magnitude
before symmetrization and the coarse-size stopping rule are built, and the matched arm equals GAMG on the
`bfs3d` `[k, ω]` block. The investigation that got there (the equilibration hypothesis and its
refutation, the reference tables, the coarse solve's pseudo-inverse artifact) is in
`.claude/notes/solve-amg-multigrid-log.md`, "Faithful smoothed aggregation — matching PETSc GAMG".
Read this section before touching the aggregation path.

  **A symmetric equilibration is a SIMILARITY TRANSFORM, so the SPECTRAL half of the setup is
  invariant under it.** `D̂⁻¹Â = S⁻¹(D⁻¹A)S`, so both spectral estimates are unchanged and the
  damped-Jacobi smoother `x += (ω/λ)D⁻¹r` is unchanged with them. Measured on a badly scaled chain:
  the **fine** level's `lam_max` is 1.966 raw against 1.976 equilibrated (equal to power-iteration
  accuracy) while the **coarse** level's is 1.81 against 15.3. The **tentative prolongation** — a
  fixed 0/1 aggregate indicator, which does not transform with the operator — is non-equivariant.

- **Aggregation over the cell graph is robust to explicit-zero pruning; a zero-fill factorization is not** — equilibration drops 26.8 % of the `[k, ω]` block's stored entries yet moves 0.03 % of the cell graph's edges (`validation/bfs3d_openfoam/equilibration_graph_effect.py`). When reasoning about a pattern change, ask which consumer sees it.

  **The three remaining differences, now BUILT and measured (same block, same state, matched arm):**

  | arm | coarse eq | ×1 | ×2 | ×4 |
  |---|---|---|---|---|
  | matched, before | 436 | 58 | 11 | **2** |
  | + fix-up pass + magnitude-first graph | 438 | 52 | 10 | **2** |
  | + coarse-size stopping rule | 438 | 52 | 10 | **2** |

  - **The fix-up pass — BUILT, `_reattach_to_adjacent_root`.** PETSc's `fixAggregatesWithSquare`
    (`agg.c:1032-1075`): after the squared-graph MIS, each selected root **in ascending index
    order** steals every distance-1 neighbour in the *unsquared* graph that belongs to another
    aggregate. Only members move, never roots, so the count is unchanged and no aggregate empties.
    **Its guarantee is conditional and must not be stated as absolute** — a member whose neighbours
    are all themselves members has no adjacent root to move to and keeps its distant one. Worth a
    cycle at ×2 and six at ×1; nothing at ×4, where the arm was already at PETSc's 2.
  - **Magnitude before symmetrization — BUILT.** PETSc block-sums `|A_ij|` and *then* symmetrizes;
    we took the symmetric part first, so an edge with `A_ij ≈ −A_ji` **cancelled out of the graph
    entirely**. On an M-matrix (every frozen upwind transport operator) the sparsity is identical
    either way, so at `strength_threshold = 0` — the default, and what every scalar hierarchy runs —
    this is a no-op. **It is NOT a no-op at a threshold**, because the weights differ (`|A_ij|`
    against `|A_ij + A_ji|/2`) and the strong-connection set is chosen from them: the coupled flow
    block runs at **0.25** (`turbulence/coupled.py`), so its hierarchy genuinely moves. Do not quote
    the M-matrix equivalence without that caveat — it is a statement about the pattern only.
  - **The stopping rule — measured INERT here, and that is a real result rather than a null one.**
    PETSc coarsens until the grid is under `coarse_eq_limit`; we stop at `max_levels`, at which
    point `max_coarse` can never fire. Setting `max_coarse=2000, max_levels=20` gives a
    **bit-identical** hierarchy on this block, because one aggregation already lands at 438 < 2000
    and both rules then stop. So it changes nothing *at this size* and remains the correct rule for
    a mesh where it would bind. The two knobs are still not equivalents and must not be quoted as
    matched.

  **Still different, verified by reading the source, and NOT measured:**
  - **Strength-of-connection semantics differ** — PETSc thresholds absolutely on the
    diagonally-scaled graph (Vanek), we use the row-max-relative classical criterion. Inert at
    threshold 0, so it affects no measurement here, but the recorded threshold arms on the two sides
    are not comparable.
  - **Tentative prolongation — NOW MEASURED, and it is REFUTED as a deep-hierarchy fix.** PETSc
    orthonormalizes each aggregate's block by QR (`agg.c:690-714`), giving unit-2-norm columns where
    ours are 0/1 — i.e. `P_ours = P_petsc · diag(√|agg|)`. The inertness claim is confirmed exactly (a
    2-level pair is bit-identical), but deeper it is **17–30× worse**, not better. Built as
    `orthonormal_prolongation`; see *"Depth, singletons, and the prolongation"* below for the arms and
    for the mechanism that was proposed and refuted.
  - **Singletons — NOW MEASURED, and this one was a real defect worth 1.7×.** PETSc drops a
    neighbourless vertex from the coarse space entirely (zero row in `P`, left to the smoother); we gave
    it its own aggregate. On this operator most such vertices are not neighbourless at all — they are
    an artifact of the random sweep's arrival order — and a three-level hierarchy came out with 49
    singletons of 161 aggregates on its second level. Fixed behind `avoid_singletons`; see below.

  **What equilibration IS worth (keep it). ⚠️ "Default off" was a scope error — settled from source
  2026-08-10:** `build_convection_hierarchy` and `_build_aggregation_hierarchy` default `equilibrate=False`,
  so "default off" is true of the **traced builder only**; `JacobiSmoothedInverse` overrides it to `True`
  (its per-cell block solve is not otherwise safe); and the PETSc `MonolithicVCycleInverse` path equilibrates
  **unconditionally** via `equilibrate_cell_major`. Three different objects, no contradiction.

  **✅ THE COARSE SOLVE IS A FACTORIZATION, NOT A PSEUDO-INVERSE — `_dense_inverse` (2026-08-14).** A
  pseudo-inverse is a singular value decomposition, and it was **59 % of a whole hierarchy build**
  (1.37 s of 2.32 s on a 92160-dof, 8.7M-nnz flow block coarsening to 2156). The generality is only
  needed for a *singular* coarse operator, which this module works to avoid producing — an empty
  aggregate is given a unit diagonal precisely so no coarse row is structurally zero. So the LU is taken
  where it is valid, falling back to the decomposition where it is not. Cost at the coarse sizes that
  arise, and at the ceiling `_MAX_DENSE_COARSE_DOFS` permits:

  | n | pinv | inv | LU factor alone |
  |---|---|---|---|
  | 868 | 0.090 s | 0.013 s (7.0×) | 0.011 s |
  | 2000 | 1.027 s | 0.118 s (8.7×) | 0.032 s |
  | 4096 | 12.06 s | 1.154 s (10.5×) | 0.263 s |
  | **8192** (the cap) | **101.8 s** | **9.7 s** | 2.0 s |

  Whole-build effect **1.97×** (2.30 s → 1.17 s), with the coarse inverse agreeing to 6.4e-15 and the
  cycle output to 1.1e-16. **The gap widens with coarse size**, so this matters more as the mesh grows,
  which is the direction 3D is heading.
  ⚠️ **Singularity is judged by LAPACK's reciprocal-condition estimate, not by whether the factorization
  raised.** An operator can be numerically singular without being exactly so, and a factorization of one
  returns finite nonsense rather than failing — measured, `np.linalg.inv` of a matrix with a 1e-18
  diagonal entry returns finite garbage without complaint. The threshold is the same `n · eps` the
  pseudo-inverse itself uses when truncating, so the two paths agree on where invertibility stops
  instead of drawing the line in two places. All three cases are pinned
  (`test_the_coarse_solve_is_a_factorization_where_the_operator_admits_one`); a fallback that never
  fires and one that always does look identical from the outside.

  **What is BUILT and measured good (keep):**
  - **Nodal (block-aware) aggregation** — `_cell_graph` collapses the dof graph to cell connectivity
    (exact, since field-major means `index % n_cells` *is* the cell) and `_block_tentative` gives each
    field its own coarse unknown. With a **block smoother** (`_cell_block_inverse`,
    `_block_diagonal_inverse_operator`) this makes the two-field slice buildable and convergent.
    **All three are required; no pair suffices** (measured 2×2).
  - **MIS aggregation** (`_mis_aggregate`) — greedy maximal independent set over a **randomized** visit
    order, single sweep, selector-claims-neighbours, faithful to `MatCoarsenApply_MISK_private`. Worth
    **8 → 5 cycles** over the two-pass RCM scheme. The old scheme only seeded from a fully-free
    neighbourhood, so it seeded few aggregates and left most vertices to a ragged cleanup pass.
  - `jax.jit` on the traced applies (they were dispatching eagerly, ~18 % of apply cost) and
    `indices_are_sorted=True` in `_coo_apply` (CSR→COO is row-sorted by construction).
  - **There is no `PerFieldNativeInverse` — deleted 2026-08-15 (binding).** It had no production caller
    (only its own tests and one sweep arm), its own docstring recorded it as superseded by
    `JacobiSmoothedInverse` wherever that works, and it was never ported onto `HierarchyBlockInverse` — so
    it had neither `refactor_block` nor `refactor` and `FieldSplitInverse.refactor` (since #281
    `refactor_block`; there is no `split.refactor`) **raised**
    the first time a march refreshed with it, while being exported from `__all__` as public API. It also
    re-committed two costs the shared base was written to remove (eager per-field transposes; a fresh
    closure per apply). Its measurements never transferred to the nodal inverse in any case — a
    per-field pair is a weaker object than one block-aware hierarchy — so the `nativeN` arm in
    `turbulence_smoother_sweep.py` went with it, leaving `nodal[N][cM]` (the whole harness was deleted
    2026-09-13, #371).
  - `JacobiSmoothedInverse` (`precondition/field_split.py`, over the shared
    `precondition/hierarchy_inverse.HierarchyBlockInverse` base), both transposable in
    closed form and fixed linear operators, so adjoint-legal. **⚠️ "Neither is wired into production" was
    wrong — settled from source 2026-08-10.** The nodal inverse is reachable through
    `JacobiSmoothedInverse` and is the `BFS3D_TURBULENCE_INVERSE=jacobi` arm; what is true is that it is not a
    *default*.

  - **`prolongation_smoothing` is its own parameter, no longer welded to `mis_aggregation`.** The old
    flag chose the aggregation *and* the prolongator formula together, so "MIS with and without
    prolongator smoothing" was not expressible and the two effects could not be separated — which is
    how the smoothed prolongator came to be recorded as void when what it actually needs is more
    smoother sweeps. `"symmetric-part"` (default, the historical formula), `"standard"` (the textbook
    σ_max form on the true operator and scalar diagonal), `"none"` (plain aggregation, what the
    shipped PETSc bundle runs). An unknown value raises.
  - **`_mis_aggregate` returns its ROOTS**, not just the aggregate index per vertex — the fix-up pass
    cannot be written without knowing which vertex seeded each aggregate, and re-deriving it
    afterwards is not possible (an aggregate's root is not recoverable from the labelling).

  **What is BUILT and measured BAD (revert or gate):**
  - The **standard prolongator at 4 sweeps** fails outright (44 cycles, true rel 1.0) — raw *and*
    equilibrated, so this is not the scaling. At **8 sweeps equilibrated it is the best traced arm
    (4 cycles)**, so it is under-smoothed rather than wrong: the earlier "VOID / much worse" reading
    conflated a smoothing deficit with a broken formula. Treat sweeps as part of that arm's
    specification, never quote it at a single sweep count.
    ⚠️ **Measured on the 2150-equation scalar problem under a point-block-Jacobi / ILU smoother. It does
    NOT carry to the flow saddle**, where both smoothed prolongators are refuted under the SIMPLE
    smoother at 4 *and* 8 sweeps — see *Smoothed aggregation on the flow saddle*. "Best traced arm"
    below always means best on the operator it was measured on.
  - **⚠️ GRAPH SQUARING ON THE FIRST LEVEL IS GAMG'S DEFAULT COARSENING, NOT A SIZE KNOB — and
    recording it as harmful was the single thing holding the comparison open.** Squaring the graph on the first level is not a size knob, it is
    GAMG's *default coarsening* (`aggressive_coarsening_levels 1`), and its 106× ratio is not
    over-coarsening — it is PETSc's 107×, i.e. the target. The 5 → 10 reading is real but was taken
    with the damped smoother; at the matched undamped smoother the same arm is **2 cycles**. A knob
    measured harmful in one configuration was recorded as harmful in general, and that closed off
    the one change that mattered.

  **Two corrections to older entries in this file, both of which cost real time:**
  - The recorded *"the builders refuse the `[k,ω]` slice because its diagonal is negative from the live
    source linearizations"* is **wrong**. The fine slice is clean — **0 of 23040** cells non-positive at
    β = 0, at the march shift, and at the floor. The refusal names `level 1`, a **coarse** operator, and
    is manufactured by **field-blind aggregation** merging a k-dof with an ω-dof of another cell. The
    error text said `level 1` all along; an inference was stapled to a verbatim quote and only the quote
    was ever checked.
  - **A field split HIDES the quality of its trailing half.** That half is ~11 % of the nonzeros, so the
    flow block carries the solve: a trailing inverse that does not converge *at all* in isolation still
    produced a plausible 1.36× blended march estimate. **Measure a block's preconditioner on the block
    alone first**, then in situ.

  - **A THIRD correction, from this round: an arm quoted at one smoother-sweep count is not a
    result about the method.** The standard prolongator was written down as "much worse" from its
    4-sweep numbers; at 8 it is the best traced arm. Two of the three wrong conclusions in this
    section came from holding one axis fixed while attributing the outcome to another.

  **Also established:** block Jacobi is a perfectly good smoother class here given enough sweeps (PETSc
  ×4 = 2 cycles against ILU ×1's 2), and on CPU the two cost the *same* per apply (102 vs 101 ms) — so
  the penalty for a GPU-friendly smoother is quality, buyable with sweeps, not cost. The remaining
  question is a GPU one and **cannot be measured in the current environment** (CPU-only JAX).

  **⚠️ COMPARE LIKE WITH LIKE, or the arm measures the smoother and gets attributed to the hierarchy
  (binding for this campaign).** Our multigrid has only Jacobi-class smoothers; PETSc's default here
  is an incomplete-LU sweep, which is both far stronger and the least parallelizable piece in it. An
  ours-vs-PETSc table whose PETSc row is ILU therefore cannot say anything about the *coarsening*,
  which is the thing under development. Every such table must carry a **matched-smoother** row —
  PETSc `pbjacobi` against our block Jacobi, at the same sweep count and the same aggregation — and
  that row is the one to quote. The ILU rows stay as the incumbent's absolute bar, not as the
  comparison.

  **⚠️ HOW THE GAP WAS ACTUALLY FOUND, because the method generalizes and the alternative cost
  days.** Three rounds of tuning our own knobs (equilibration, prolongator variants, sweep ladders)
  moved nothing and produced two wrong write-ups. What worked was **an independent agent reading
  `agg.c` / `misk.c` / `rich.c` against `multigrid.py` and reporting differences with citations on
  both sides** — it found the undamped Richardson scale, the default aggressive level, and the
  fix-up pass in one pass, none of which was visible from any measurement we had. When a
  reimplementation of a *published, available* algorithm underperforms it, read the source before
  running another sweep; and do not describe a port as faithful until something has checked it
  line by line, which is exactly the claim that was wrong here.

  **NEXT STEPS, in order:**
  1. **Decide whether the matched configuration becomes the DEFAULT, and for which consumers.**
     ⚠️ **This shipped: `JacobiSmoothedInverse` now defaults to the whole matched bundle**
     (`aggressive_levels=1`, `prolongation_smoothing="none"`, `spectral_damping=False`,
     `equilibrate=True`) and is the `BFS3D_TURBULENCE_INVERSE=jacobi` arm. The open part is whether it
     transfers to other consumers; the library `monolithic_vcycle_inverse` defaults are untouched. The measurement says the matched
     bundle is 5 → 2 cycles on the turbulence block; whether that transfers to the scalar transport
     and velocity blocks is unmeasured, and flipping a default is a march-level decision, not a
     block-probe one.
  2. Then, if still short: sparse-LU coarse solve (ours is a dense inverse, `_dense_inverse`),
     QR-orthonormalized
     tentative columns (inert at 2 levels, not deeper), and Chebyshev with the SA-cached bounds.
  3. **✅ SCALABILITY — BUILT AND SHIPPED as `bfs3d`'s trailing-block default (2026-08-18).** The
     convection hierarchy was capped at 2 levels (`_CONVECTION_LEVELS`) with a **dense inverse** coarse
     solve — at a 77× ratio that is ~26k coarse dofs and 5.4 GB to store at 1M cells, infeasible.
     `compare.py`'s `JACOBI_TRAILING` now defaults `JacobiSmoothedInverse` to `max_levels=20,
     max_coarse=200, strength_threshold=0.25, aggressive_levels=0, frozen_coarsening=True`. Depth or
     the threshold *alone* did nothing on a block-alone probe of the converged root's zero-shift
     operator (every arm at the old 2-level default shows no real residual reduction after 50+ restart
     cycles); combined, the same probe reached rtol 1e-8 in 11 cycles, and a full 3-rung march at this
     bundle reproduced the shipped root exactly (`x_r/h` 8.3611, 59 steps). `frozen_coarsening` is
     load-bearing, not optional decoration: a nonzero threshold reads the operator's values, so a plain
     refresh re-coarsens from scratch and can move the hierarchy's shapes, retracing the compiled cycle
     every refresh — measured ~50 s per refresh unfrozen against ~4 s frozen, stable across a full march.
     PETSc reaches only 2 levels here because it coarsens until under `coarse_eq_limit`; our
     `max_coarse` is NOT the equivalent knob and has no effect while `max_levels` binds first.
     **Still open:** this is validated at `bfs3d`'s 23040 cells only, and the per-row-field
     normalization of `_cell_graph`'s edge weights (this block's ω rows sit ~8 orders above its k rows,
     so a raw threshold risks reading as an ω-only strength measure) remains unbuilt — the shipped
     bundle uses the raw threshold as-is and it did not visibly misbehave at this size, which is not the
     same as having ruled the risk out at a much larger one. Re-run
     `validation/bfs3d_openfoam/trailing_depth_probe.py` on whatever bigger mesh next triggers this.
     **A separate, likely LARGER wall this does not touch:** `FieldSplitPreconditioner` materializes
     the *entire* coupled Jacobian (all fields, reach 3) to slice out this block regardless of its own
     depth — see the field-split scaling note in `solve-field-split.md`.

  **Local PETSc source** for continuing the port (shallow sparse clone, may need re-cloning):
  `src/ksp/pc/impls/gamg/{agg,gamg}.c` and `src/mat/graphops/coarsen/impls/misk/misk.c` — note
  `coarsen` moved under `graphops` in current PETSc. GAMG-AGG defaults: `nsmooths 1`,
  `aggressive_coarsening_levels 1`, `use_aggressive_square_graph TRUE`,
  `use_minimum_degree_ordering FALSE` (so: random order), `aggressive_mis_k 2`, `graph_symmetrize TRUE`.



## The coupled AMG builder (gone)

- **There is no coupled AMG builder** — `coupled_amg_continuation` was replaced by `coupled_step` (#371)
  and the monolithic V-cycle it built was deleted with PETSc (2026-10-10). The multigrid path a coupled
  march takes now is `MaterializedJacobian(FieldSplit(...))`, whose adjoint gate is
  `tests/integration/test_coupled_field_split.py` (no longer PETSc-gated, so it runs in CI).


## Binding decisions — `precondition/multigrid.py`

> The rest of `solve.md`'s "Binding decisions" carries the general Newton/adjoint decisions and a
> short pointer to this section; these are the `multigrid.py`-specific ones, moved here in full.

- **A level operator is a PADDED-ROW (ELLPACK) operator, `_PaddedRowOperator` — there is no
  `_CsrOperator` and no `BCSR` product in the package (2026-10-14).** Every row is padded to one width:
  `columns` / `values` are `(n_rows, width)`, `row_lengths` marks which slots are stored entries, and
  `apply` is `sum(values * x[columns], axis=1)`. ⚠️ **It is NOT one fused loop** (an earlier version of
  this entry said so; the compiled program says otherwise, 2026-10-15): XLA writes `x[columns]` to memory
  in a gather fusion and a second kernel reads it back for the multiply and row sum — on CPU a
  `__ynn_fusion` (`kCustom`) with empty `outer_dimension_partitions`. On 4-core Linux both kernels are
  threaded (4 cores ≈ 2× one core). `jax.experimental.sparse.BCSR @ x` lowers instead to one opaque
  kernel call (`cpu_csr_sparse_dense_ffi`).
  - **⚠️ ON macOS THAT ROW-SUM KERNEL IS THE BOTTLENECK FOR WIDE ROWS, AND NO FORM OF THE SUM AVOIDS IT
    CHEAPLY (measured 2026-10-15/16, Apple M3 Pro, bfs3d `state-00018`, `split_apply_timing.py`, best of
    10, the `ell-csr-forms-probes` branch).** Split apply: padded 140 ms against CSR 105–109; with
    `XLA_FLAGS=--xla_cpu_experimental_ynn_fusion_type=-reduce` **91 ms** — same on jax 0.10.2 and 0.11.2,
    so not a version artefact. On Linux the same flag is neutral to worse (a 228-wide row 36 → 45 ms), so
    it cannot be a global setting. Measured and rejected: `einsum` and a chunked two-stage sum (compile to
    the same kernels), a column-major `(width, n_rows)` layout (~1.4× slower on Linux, 3× on Mac per
    operator), and a reduce-free halving sum (2× slower on Linux, **6× slower** on the Mac: 902 ms). What
    does fix the Mac's forward apply is float32 (82 ms), shipped by both cases (`solve-field-split.md`).
    pitzDaily's narrow rows (≤ ~85) win on both platforms (Mac 32.6 → 15.7 ms). Harness:
    `validation/padded_row_forms.py` on that branch (per-operator forms, precisions and compiled fusions).
  - **The TRANSPOSE is stored (`t_columns` / `t_values`) and applied as a gather (2026-10-16).** `apply`
    goes through `jax.custom_derivatives.linear_call`, so `jax.linear_transpose` applies `A^T` as a
    forward product instead of deriving a scatter-add over every slot from the gather. Mac bfs3d split
    transpose (float64, prototype): **480 → 143 ms** (CSR `main`: 258–263); per operator on the Mac the
    stored form was 4× the derived one and 7× CSR. On the 4-core Linux container after its 2026-10-16
    restart the pitzDaily split transpose was within noise (91–111 against 100–110 ms, three interleaved
    rounds), where the same prototype on the earlier host gave 110 → 76 ms. Costs: a second copy of the
    operator storage (pitzDaily split 115 + 115 MB); its width is budgeted per coarse level too
    (`ShapeBudget.transpose_width`) and compared by `_require_matching_structure`; and **`apply` has no
    `jax.vmap` rule** (`linear_call` has none) — nothing in the package vmaps a preconditioner apply (the
    materialized probe vmaps the RESIDUAL's JVP, and no residual applies a multigrid level). Pinned by
    `test_a_padded_row_operator_transposes_by_gathering_its_stored_transpose` (no scatter in the
    transposed program), `..._transpose_takes_its_own_budget` and the budgeted-widths assertions in
    `test_shape_budget_makes_a_repartitioning_rebuild_a_compilation_cache_hit`; mutation-checked (rule
    removed, budget not passed, overflow unchecked). `_require_matching_structure`'s transpose-width
    comparison is not mutation-checked: no fixture refreshes into a different transposed width.
  - **Measured** (pitzDaily `case.yaml`, the shipped field split built at the hybrid start, Linux
    x86_64, 4 cores, jax 0.11.2, best of 15, two runs per arm, `main` `a5d61d49` against this change, a
    quiet machine): one split apply **168–176 → 113–126 ms**, its transpose (the adjoint's direction)
    **171–174 → 118–129 ms**, results equal to 1.2e-14; first-call compile 1.0 → 1.7 s. Per operator on
    the finest saddle level: the level operator (36675 rows, 2.68M entries) 6.5 → 4.4–5.2 ms, the Schur
    complement (0.99M entries) 6.3 → 1.8–2.0 ms, divergence and `D⁻¹G` (0.60M each) 4.5 → 0.6–1.2 ms.
    ⚠️ **The kernel's cost is machine-dependent and the `BCSR` figure recorded in the traced-multigrid
    section (1.4 ms on a 4.2M-entry block) was a different machine**, so re-measure there before
    quoting this gain for it: `validation/pitzdaily_openfoam/csr_kernel_precision.py` times the
    padded-row, CSR and segment-sum forms side by side.
  - **bfs3d, at ONE fixed state (2026-10-15, `validation/bfs3d_openfoam/split_apply_timing.py`,
    same Linux machine, best of 10, two runs):** both checkouts build the identical split (every level
    equal, outputs within 9e-16); apply **779–791 → 606–693 ms**, transpose **870–918 → 911–1017 ms** — the
    forward gain is 1.14–1.28× and the transpose gains nothing, since the transpose of a padded-row gather
    is a scatter-add over `n_rows * width` slots. On the project owner's Apple M3 Pro (jax 0.10.2, PR #721)
    the pitzDaily kernel is 2.1× faster forward and 1.3× transposed, but a bfs3d march ran **8.5 % slower
    per step** (529 → 574 s). ⚠️ That march pair compared two DIFFERENT hierarchies (see `validation.md`,
    "bfs3d's leading coarsening moves under rounding"), so it separates neither format nor hierarchy;
    the fixed-state harness is the measurement that does: on macOS it showed the padded form **slower**
    at a fixed state (apply 105–109 → 138–139 ms, transpose 258–263 → 461–462 ms), the cause and
    remedies being the two bullets above.
  - **What the padding costs:** storage `n_rows * width`. On pitzDaily's split the fine levels pad
    3–5 %, aggregation's coarse levels up to 1.95× (they are small), 1.09× over the whole split. On
    bfs3d's: fine levels 1.11×, the leading coarse levels 1.7–2.4×, the trailing ones' deepest up to
    6.6×, about 1.3× over the two hierarchies. An
    operator with a few very wide rows (a dense constraint row) would pay that width on every row; no
    level operator here has one.
  - **The reduction form, not an unrolled sum.** `vectors.dot` unrolls its sum because a
    reduction-rooted fusion ran on one thread on macOS arm64; here an unrolled row sum was ~10 % faster
    standalone on Linux, multiplies the compiled program by the row width (up to 364 on bfs3d) and its
    transpose was up to 4× slower, so the reduction was kept. The Mac's wide-row loss is that same
    reduction kernel (bullet above); the reduce-free sum measured there was far worse, not better.
  - **The shape budget is a WIDTH per coarse level now (`ShapeBudget.operator_width`, was
    `operator_nnz`)**, and `from_scipy(a, width=)` refuses a row wider than the budget rather than
    dropping entries. `_require_matching_structure` compares widths: the width is the compilation
    signature.
  - **`to_scipy` returns exactly the stored sparsity, explicit zeros included, padding excluded** (the
    structural patterns of a refit are read from it). The old CSR form's entry-count budget appended its
    padding to the last row as stored zeros, so a budgeted coarse level read back with extra entries;
    this one does not. `nnz` is the stored-entry count.
  - Pinned by `test_a_padded_row_operator_reproduces_the_matrix_it_froze`,
    `..._takes_a_wider_budget_and_refuses_a_narrower_one` and `..._transposes_to_the_transposed_matrix`
    (a ragged rectangular operator with an empty row and a stored zero), mutation-checked six ways (slot
    placement, stored zeros dropped, the overflow check, the diagonal, `nnz`, `row_lengths`).

- **Reading a frozen level back onto the host has one home per array it owns — `_PaddedRowOperator.to_scipy()`
  and `_SparseLevel`/`_AirLevel.prolongation_scipy()` (binding, #287).** Every host-side consumer of a
  traced level's arrays — the host-smoothed inverse factorizing an operator, a march refresh re-deriving
  a hierarchy at a new operator, a SIMPLE-smoothed level forming its Schur pieces — needs the exact same
  operator → `scipy.sparse.csr_matrix` and `p_*` → prolongation reconstructions, and
  four call sites (`ilu_inverse.py` — since deleted, #371 — `saddle_multigrid.py`'s `derive_extras`, `SmoothedHierarchy.refit`,
  `refresh_air_hierarchy`) each open-coded them. `_PaddedRowOperator.to_scipy()` is the exact inverse of the
  existing `from_scipy`; `prolongation_scipy()` is level-kind-specific because `_SparseLevel` stores a
  COO triple (`p_frow`/`p_ccol`/`p_val`, one prolongation doubling as the transposed restriction) while
  `_AirLevel` stores an independent CSR-shaped one (`p_row`/`p_col`/`p_val`, restriction and
  prolongation kept apart) — so there is no single shared method across the two classes, only one per
  class. No behaviour change; each call site is a byte-identical reconstruction of what it built inline.
- **`precondition/multigrid.py` is a pure operator-coarsening library — operator-in, uniformly (binding).**
  **Every** builder takes an assembled `a: sp.csr_matrix` — `build_smoothed_hierarchy(a)`,
  `build_convection_hierarchy(a)`, `build_air_hierarchy(a)` — and none takes a mesh, edge arrays, or
  flow quantities. Assembly lives beside it in `aquaflux/precondition/frozen_operator.py`
  (numpy + scipy only — no mesh, no field, no `jax`):
  `ConvectionDiffusionStencil(owner, nb, coefficient, n, *, flux=None, boundary_diagonal=None).assemble()` —
  symmetric graph Laplacian when `flux is None`, first-order-upwind convection-diffusion otherwise —
  plus `decouple_dof(a, index)` for the closed-domain pressure pin. It is the **one** assembler for
  all four consumers (pressure Schur, viscous velocity block, convection velocity block, k/ω scalar
  transport). Do not reintroduce a `(owner, nb, coeff, …)` signature into `multigrid.py`, and do not
  add a second stencil assembler — the old `_laplacian_csr` was exactly the stencil's assembly
  at `flux=None` and was deleted (and the free function `convection_diffusion_operator` became the
  stencil's `assemble`, #89). `build_convection_air_hierarchy` was likewise **deleted**: once it
  took `a` it was a pure alias for `build_air_hierarchy`.
  *Why it is a solver concern, not a flow one:* the first-order-upwind stencil is the
  **preconditioner's** choice, independent of what the residual discretizes advection with — it is
  chosen to give a diagonally dominant M-matrix an aggregation hierarchy can coarsen. Its parameters
  are a weighted graph (`coefficient`, `flux`), not flow quantities. Keeping it in `solve/` also adds
  no new dependency edge: every consumer already imports `precondition.multigrid`.
- **Aggregate-size diagnostics are an EXPLICIT OUTPUT PARAMETER, not a module-level accumulator
  (binding, #287, 2026-09-12) — `_AGGREGATE_STATS` is DELETED.** `_build_aggregation_hierarchy` (and
  both public wrappers, `build_smoothed_hierarchy` / `build_convection_hierarchy`) take an optional
  `stats: list[dict] | None`, appended to in place with this build's own per-level
  `aggregate_size_histogram`; `None` (the default) skips it entirely. This replaces a bare module
  global that every hierarchy in the process shared and that a caller "cleared before" and "read after"
  by convention rather than by construction — a real bug on the one path that broke the convention:
  `HierarchyBlockInverse.refactor_block`'s `frozen_coarsening` branch calls `SmoothedHierarchy.refit`,
  which reuses the stored prolongations and aggregates **nothing new**, then still called
  `_after_coarsening()` — which read the global as if this hierarchy's aggregation had just run. With a
  leading+trailing pair (`field_split_inverse`) the trailing block generally aggregates
  last, so the leading block's frozen-coarsening refresh reported the **trailing block's** aggregate
  stats as its own. Diagnostic-only (never reaches the operator, the residual, or the adjoint), but
  wrong on a shipped opt-in report path, and it was the only reason `saddle_multigrid.py` reached into
  `multigrid.py`'s privates for *data* rather than behaviour.
  ⚠️ **The obvious fix — carry the histograms as a field on `SmoothedHierarchy` — was considered and
  REJECTED.** `SmoothedHierarchy` is passed as a **jit argument** in the traced consumers (see the
  static/traced level split two bullets below), and a build-to-build-varying histogram field would
  either have to be a *static* field — retracing the compiled cycle on every refresh purely to update a
  diagnostic, which is exactly the per-rung-recompile hazard `turbulence.md` documents at length — or a
  traced one, which a list of Python dicts cannot be. Keeping the histograms **off the pytree
  entirely** avoids the question. `HierarchyBlockInverse._coarsen` (`hierarchy_inverse.py`) is the one
  place both smoothers' hierarchies are built, so it is the one place that owns a fresh list per build
  and stores it as `self._aggregate_stats` on the **inverse**, never on the hierarchy — a frozen-
  coarsening refresh then reports this inverse's own last real aggregation (stale but correctly
  attributed) rather than a stranger's most recent one. `SimpleSmoothedInverse` no longer overrides
  `_after_coarsening` at all: capturing the stats moved to where the aggregation actually happens, so
  there is nothing left for the override to do. Pinned by
  `test_a_frozen_coarsening_refresh_reports_its_own_last_aggregation_not_a_strangers`
  (`tests/unit/test_saddle_multigrid.py`), which reproduces the leading+trailing ordering directly.
- **The V-cycle recursion AND its outer fixed-cycle driver are single-homed (binding, #52).** A
  family (`smoothed_multigrid_cycles`, `convection_multigrid_cycles`, `air_multigrid_cycles`) contributes
  **only** its `_VCycleOps` — restriction, prolongation, smoother. The recursion is `_frozen_v_cycle`
  and the outer loop (zero initial guess, `cycles` residual-correction passes,
  `x += _frozen_v_cycle(levels, b - A x, …)`) is `_fixed_cycles`; both are written once. That
  outer loop is what makes `b -> x` a constant linear operator — the property the frozen-left-PC and
  the adjoint transpose depend on — so it must not be re-typed per family where one copy could drift.
  A new family adds a `_VCycleOps` builder and a thin entry point that calls `_fixed_cycles`; do
  **not** re-write the cycle loop in it.
- **A level is STATIC indices + TRACED values, so a hierarchy refresh is a jit cache hit (binding).**
  `_SparseLevel` / `_AirLevel` are `equinox.Module`s in which **only `n` and `n_coarse` are static** —
  they size the sparse matvec output (`_coo_apply`'s `n_out`), so they must be concrete. Everything
  else (`val`, `diagonal`, `coarse_inv`, the prolongation/restriction values, and **`lam_max`**) is a
  traced leaf. `lam_max` is deliberately a **0-d array, not a Python `float`**: it is only smoother
  arithmetic, and as a static field any refreshed value would be a new compilation-cache key —
  defeating the point. Consequence: a hierarchy passed as a **jit argument** survives a refresh as a
  *cache hit* (one compiled V-cycle), which is what lets a frozen preconditioner track a developing
  flow without paying a recompile per refresh (the ~2.6× scalar-AMG staleness win above). This is only
  sound because **the aggregation coarsening is a pure function of the graph at the default
  `strength_threshold=0`** — `_aggregate` reads `owner`/`nb`/`n` and never the coefficients — so on a
  fixed mesh a hierarchy re-derived at a new operator has identical aggregates, coarse sizes and array
  shapes, and only values differ (this is why the refreshed *scalar k/ω* AMGs keep `strength_threshold=0`
  — the flow block is frozen, so it can afford the value-dependent SoC below; the scalars cannot without
  a value-refresh). Both properties are pinned in `tests/unit/test_multigrid.py`
  (`test_aggregation_hierarchy_structure_is_value_independent`,
  `test_refreshing_a_hierarchy_is_a_compilation_cache_hit`).
  ⚠️ **The shape half of that claim was FALSE on a real operator until 2026-10-12 (#290), and the
  chain fixtures could not show it.** The Galerkin `Pᵀ A P` was a plain `scipy` product, which drops
  an entry summing to exactly zero, so on a block whose stored entries are sometimes exactly zero (a
  coupled `[k, ω]` Jacobian, where closure switches zero a derivative) the coarse operator's stored
  count moved with the values even at threshold 0. Measured on pitzDaily's trailing block (`JacobiSmoothed`,
  `max_coarse` 2000, threshold 0, two levels) across its hybrid start and ten march checkpoints:
  level-1 entries 9360 → 9535 → 9515, so the jitted trailing cycle on the shipped host split
  retraced on refresh. Both the build and `SmoothedHierarchy.refit` now use `_galerkin_coarse`
  (`Pᵀ` as the restriction), whose stored pattern is the structural one; afterwards the same probe
  (`validation/traced_split_refresh_shapes.py`) finds the trailing signature unchanged at every
  state and shift. Values are bit-identical, and so is the strength graph at a nonzero threshold (a
  stored zero is never strong); at threshold 0, deeper levels' aggregation now sees structural
  edges carrying zero weight, which is the value-independence the paragraph above claims. Pinned by
  `test_a_nodal_hierarchy_keeps_its_shapes_when_a_coupling_becomes_exactly_zero` (mutation-checked
  on the build and the refit separately).
  ⚠️ **Two more zero-dropping sites moved a REBUILD's shapes, and the Galerkin fix alone did not reach
  them (found the same day, #290).** *(i)* The aggregation graph: at threshold 0 the MIS path
  symmetrized the cell graph by a sparse add and squared it (aggressive level) by a sparse product,
  both of which drop an exactly-zero result, so a cell pair whose coupling was stored but zero at one
  state lost its edge there — measured as pitzDaily's trailing coarse level moving 650 → 648 cells
  between early-march states. `_build_aggregation_hierarchy` now hands the coarsening `_ones(graph)`
  when `strength_threshold <= 0` (the MIS helpers read only the pattern, checked: none reads `.data`).
  *(ii)* A smoothed prolongation `P_tent − c D⁻¹ A_agg P_tent` drops zeros the same way; both smoothed
  branches now store it on `_smoothing_pattern` (whole cell blocks for `D⁻¹`, `A + Aᵀ` for `A_agg`).
  After both, `validation/traced_split_refresh_shapes.py` reports **0 moved signatures and 1 trace of
  the jitted apply across 40 builds** (pitzDaily hybrid start + ten checkpoints, both blocks). Pinned by
  `test_a_threshold_free_mis_coarsening_keeps_its_shapes_when_a_coupling_becomes_exactly_zero`,
  parametrized over the three prolongation forms; each fix mutation-checked on its own (no structural
  graph fails all three, an unpatterned standard or symmetric-part branch fails its own). A nonzero
  threshold still reads values, so a rebuild there is shape-stable only under `frozen_coarsening`.
  ⚠️ **Those structural patterns doubled the refactor phase of a refit, and a cache now pays for them
  once (`StructuralPatterns`, 2026-10-13).** Each pattern is the same product recomputed over
  `_ones` copies of its operands, so every Galerkin operator and every SIMPLE `D⁻¹G`/Schur cost two
  products plus a key search on every refit. Measured with `validation/refit_cost_probe.py` (pitzDaily
  `state-00031`, shift 0.05, leading block frozen on both trees, best of 3, Linux x86_64, 4 cores, jax
  0.11.2): the split's `refactor_block` took 0.88 s at `61a14308` (before #290) and 1.06 s on `main`
  (leading block 0.50 → 0.69 s); bfs3d's refactor phase went ~2.8 → ~5.3 s per re-fit (#699 run 0, not
  re-measured here). `StructuralPatterns` (`multigrid.py`) holds one pattern per slot (call site and
  level), keyed by a checksum of the operands' stored patterns, so a pattern is recomputed only when an
  operand's pattern moved; a collision can only hand `_Structure.store` a pattern missing an entry,
  which it refuses, so it cannot change an answer. Each `HierarchyBlockInverse` owns one
  (`self._patterns`) and passes it to `SmoothedHierarchy.refit(a, patterns=)` and, for the SIMPLE
  smoother, to `_simple_pieces(patterns=)`; `None` computes every pattern afresh. Same probe after: split
  **0.83 s**, leading block **0.48 s**. ⚠️ **On bfs3d that recovered under half** (#708 comment, `state-00036`,
  shift 0.05, best of 5, M3 Pro, jax 0.10.2: split 2.46 s before #290, 4.80 s on `main`, 3.75 s), because
  `_Structure.store` then searched every key of the pattern (`np.searchsorted`, 1.42 s of 1.76 s; the
  7.1M-entry Schur complement dominates). So `_Structure` also remembers where the last operator's
  entries went, reused while the new operator's `indptr` and `indices` are EXACTLY equal to the last
  one's (compared, not checksummed: a hit skips the subset check, so a collision would scatter values
  silently); a scipy product that drops an exact zero changes them and is located afresh. Unsorted
  operators are placed by key, so the sort before the search is gone too. Pinned by
  `test_a_structure_reuses_positions_only_for_identical_stored_entries` (mutation-checked three ways:
  row pointer only, always reuse, never reuse). **bfs3d after it** (#708 comment, same probe, state and
  machine, `e3c61519` re-run the same session as the control at 3.78 s): split **2.03 s**, leading block
  1.71 s, trailing 0.20 s -- 2.4x `main` and 0.43 s below the pre-#290 2.46 s, the patterns and positions
  now being computed once where before #290 every refit did its own sparse bookkeeping; what remains is
  the products (`__matmul__` 1.28 s) and the spectral-radius estimates. Refit only; no bfs3d march on it. A hierarchy BUILD does not use it (a rebuild re-derives the
  prolongations, so the Galerkin operands change); the pitzDaily trailing block, unfrozen, still
  rebuilds and still pays (0.25 s). Pinned by
  `test_structural_patterns_compute_a_pattern_once_while_the_operands_patterns_hold` and
  `test_a_frozen_refresh_reuses_its_structural_patterns_and_returns_what_computing_them_would` (no
  `_ones` call on a second frozen refresh, and every leaf equal to an uncached inverse's), mutation-checked
  five ways (always reuse, never reuse, the inverse or the smoother not passing the cache, one slot for
  every level).
  **Caveat — lAIR does NOT get this for
  free, and this is measured, not hypothetical.** Its C/F split comes from `_strength_classical`, which
  thresholds on `|A_ij|`, so re-deriving a reduction hierarchy at a new operator changes the split and
  the shapes: on a 600-cell chain, cold vs developed coefficients (5000× flux, 1000× viscosity ramp)
  gave **identical L0–L2 but divergent L3–L5** (`n_coarse` 37→38, then `n` 37→38 / `nnz` 109→112,
  18→19 / 52→55), i.e. a different jit signature and therefore a recompile anyway. The aggregation
  path was invariant at *every* level in the same comparison. **Consequence:** for `ScalarAir()` (and
  `velocity=ConvectionAir()`) a cheap refresh requires **reusing the reference's frozen C/F split and
  prolongation and recomputing only the values on it** — legitimate, since any valid split gives a
  valid preconditioner. That is **`refresh_air_hierarchy(hierarchy, a_new)`** (below),
  whereas the aggregation path gets it for free by rebuilding. (No `degree=` — that argument was
  removed 2026-08-19 when the refresh started reusing the frozen neighbourhoods, which carry the
  degree, so a refresh can no longer disagree with its build about it.)
  - **Strength-of-connection (SoC) aggregation IS available now — opt-in via `strength_threshold`,
    and it makes the aggregation path value-dependent (so it forfeits the free refresh above).**
    `build_smoothed_hierarchy(a, strength_threshold=θ)` / `build_convection_hierarchy(a, …)` (and
    `BlockPreconditioner.build(…, strength_threshold=θ)`) aggregate on the **strong** subgraph only —
    edge `(i,j)` kept iff `|A_ij| ≥ θ·max_k|A_ik|` from either endpoint (`_aggregation_edges`, reusing
    `_strength_classical`, symmetrized) — instead of the full graph. **Why it exists:** isotropic
    aggregation coarsens *across* the stiff direction of a high-aspect-ratio / skewed operator and the
    V-cycle stalls; SoC coarsens *along* the strong coupling and keeps contracting. Measured on a
    uniformly-anisotropic Poisson (cycles to true 1%): plain `>20` and rising toward failure as the
    aspect ratio grows (contraction 0.22→0.56→0.90→0.97 at AR 1→10→100→1000), **SoC a flat 3 cycles at
    every AR, mesh-independent** — the difference between a converging and a non-converging solve at
    AR≥100 (`test_multigrid.py::test_strength_of_connection_aggregation_fixes_an_anisotropic_operator`).
    **`θ=0` (default) is byte-identical to the historical isotropic build.** It is turned **on (θ=0.25)
    for the coupled flow block** (`_coupled_shift_policy`: the convection velocity AMG), which is frozen
    at the reference state so the value-dependence costs no refresh; it is a **no-op on
    the low-aspect-ratio pitzDaily case**, and the payoff is the future wall-resolved / skewed regime.
    It does **not** apply to the reduction-based `air` block (already strength-based), and the
    refreshed scalar k/ω AMGs stay `θ=0` to keep their refresh cache-hit — a value-refresh (à la
    `refresh_air_hierarchy`) to let them use SoC too is the tracked follow-up.
- **⚠️⚠️ THE lAIR RESTRICTION WALKED THE OPERATOR'S FULL SPARSITY PATTERN, NOT THE STRENGTH GRAPH —
  a conformance defect, and it is why lAIR "did not finish" on every real operator (found and FIXED
  2026-08-19).** `_lair_restriction` grew each C-point's F-neighbourhood by stepping over
  `a.indices`. Manteuffel, Ruge & Southworth (SISC 40(6), 2018) specify the walk over the
  **strong-connection graph**, and name this exact failure: a wider neighbourhood "can lead to
  intractable complexity due to larger local linear systems as a result of coarse-grid operator
  fill-in". We met it by construction on anything denser than a nearest-neighbour stencil, which is why
  it looked fine on the 2D unit fixtures and died on a real block.
  - **The mechanism is compounding fill, not Python overhead — and that distinction was nearly missed.**
    A profile at level 0 puts only 12 % of the time in the dense local solves and the rest in Python and
    `scipy` indexing, which reads as "Cythonize this". One level down the ratio inverts completely.
    Instrumented `build_air_hierarchy` on a convection-dominated distance-2 operator (8192 rows,
    45 nnz/row, `theta` 0.25, `degree` 2): stored entries per row went **45 → 319 → 856** (a coarse
    operator 80 % dense) with `R` at **1167 nnz/row** on level 1, and level 1 alone cost 20 s. Changing
    **only** which graph the walk is given, everything else identical: **45 → 62 → 78**, `R` at 20
    nnz/row, and the total restriction time **34.9 s → 0.4 s**.
  - **At a real shape the whole hierarchy now builds in ~4 s.** Six levels on a 46080-row, 2.2M-nnz
    convection-dominated operator: restriction 3.2 s, splitting 0.7 s, Galerkin 0.07 s, densities
    47/67/86/83/47/15. The same build under the old walk does not finish. For scale, the traced flow
    hierarchy builds in ~6 s and PETSc GAMG in ~3 s, so lAIR setup is now in normal range and **the
    method is evaluable on `bfs3d` for the first time**.
  - **It is a real accuracy-against-cost trade, now a knob: `build_air_hierarchy(restriction_theta=…)`,
    defaulting to `theta`.** Every admitted connection makes the local solve a better approximation of
    the ideal restriction and makes the coarse operator denser. Per-cycle contraction on the 2D
    convection-diffusion fixture at cell Peclet ~40 (n=24, n=48), against peak coarse density on the
    3D-shaped operator above:

    | `restriction_theta` | contraction n=24 | n=48 | 3D setup | 3D peak nnz/row |
    |---|---|---|---|---|
    | `0.0` — the old full-pattern walk | 0.0001 | **0.0030** | **does not finish** | 856+ |
    | 0.05 | 0.0543 | 0.0867 | 5.9 s | 296 |
    | 0.10 | 0.0543 | 0.1390 | 5.2 s | 198 |
    | 0.20 (with `theta` 0.40 — the paper's own pair) | 0.0543 | 0.1523 | — | — |
    | **`theta` = 0.25 — the default** | 0.0543 | **0.1660** | **3.9 s** | **86** |

    So the old code sat at the accurate/unaffordable extreme, and `restriction_theta=0.0` still
    reproduces it exactly where an operator is sparse enough to afford it. **`0.0` is also the only
    setting for which the exact-Schur property holds**, because the ideal restriction must cancel
    *every* term of `A[g, F]`, including weak ones — pinned by
    `test_lair_restriction_reproduces_the_exact_schur_complement`, which now walks at zero threshold
    and says why.
  - **⚠️ The contraction test's old bars were calibrated against the unaffordable extreme.**
    `test_air_v_cycle_is_mesh_independent_at_high_peclet` required contraction `< 0.1` and a
    cross-refinement spread `< 0.05`; **no** paper-conformant configuration meets them, the paper's own
    (0.40, 0.20) included. It was renamed `test_air_v_cycle_contracts_strongly_at_high_peclet` and
    recalibrated to `< 0.25` / `< 0.15` against the default, with
    `test_a_wider_restriction_neighbourhood_buys_accuracy_and_costs_density` pinning both halves of the
    trade so neither reads alone as a reason to move the default. Note the old arm was also the *least*
    mesh-independent of the lot (22.5× degradation across the refinement against the default's 3.1×) —
    it passed the mesh-independence bar on headroom, not on the property the name claimed.
  - **`_require_bounded_coarsening` (new) refuses a densifying hierarchy at `_MAX_COARSENING_GROWTH`
    (4× stored entries per row against the level above), skipping levels at or below `max_coarse`
    since a small bottom level is routinely near-dense and compounds into nothing.** The failure it
    catches has no symptom other than a build that stops finishing — the growth lands in dense local
    solves rather than in anything that raises — so it is a guard, not a diagnostic. Pinned in both
    directions by `test_air_coarse_operators_stay_sparse_on_a_distance_two_operator` and
    `test_air_build_rejects_a_densifying_hierarchy_rather_than_grinding`.
  - **⚠️ A SEPARATE, PRE-EXISTING DEFECT SURFACED WHILE VERIFYING THIS, AND IT BLOCKED THE REFRESH ON
    EVERY REAL OPERATOR — fixed in the same change (`_galerkin_coarse`).** `scipy` **drops an exactly
    zero entry from a sparse product**, and lAIR generates those *structurally*: a degree-2
    neighbourhood can hold an F-point that the C-point's row does not couple to, and the local solve
    then returns exactly zero for it. So the Galerkin coarse operator's stored entry count depended on
    the operator's **values**, and `refresh_air_hierarchy` — whose whole purpose is to hold the jit
    signature — failed its own `_require_matching_structure` check on a perfectly legitimate operator,
    with a message blaming the caller's sparsity pattern. Measured on the 46080-row operator: 14 such
    weights at the build state, zero at the developed one, moving the level-1 count by 160 of 1.52M.
    **Not an artifact of a contrived fixture** — it reproduced with generic random values, because the
    zeros come from the neighbourhood's *pattern* rather than from cancellation. `_galerkin_coarse`
    takes the pattern from the same triple product over matrices of **ones**, where no cancellation is
    possible, and scatters the real values into it; the build costs ~11 % more (5.0 s against 4.5 s on
    that operator) and the refresh then preserves structure and runs in 3.5 s. Pinned by
    `test_galerkin_coarse_pattern_does_not_move_when_a_weight_becomes_zero`. **This was latent before
    the neighbourhood fix and would have bitten whoever next ran a refreshing lAIR march** — the old
    walk was equally value-dependent in `z`; it was simply unreachable because no such hierarchy could
    be built.
  - **⚠️ THE lAIR REFRESH'S STRUCTURAL BENEFIT IS SMALLER THAN IT LOOKED, and the old evidence for it
    was partly the same `scipy` artifact.** `test_refreshing_an_air_scalar_preconditioner_preserves_
    its_structure` asserted that a plain *rebuild* at a developed state changes the shapes — the
    non-vacuity half of the test — and that assertion **broke** once `_galerkin_coarse` made the coarse
    pattern value-independent. Measured on that fixture (a uniform 24x12 transport operator): scaling
    the diffusivity and the flux *together* over a 400x swing leaves the level-0 C/F split **bit-
    identical**, because every relative strength is unchanged, so a rebuild really is shape-stable there
    and the old shape difference was entirely the zero-pruning artifact. Only moving the convection/
    diffusion **balance** far enough to reorder an edge's strength re-splits (a diffusion-dominated arm
    takes level 1 from `n_coarse` 72 to 48). The test now uses a diffusion-dominated third arm and says
    why. **Consequence: on a uniformly-scaled operator, `reuse=` buys nothing structural** — lAIR's
    value-dependence is real (`test_lair_structure_is_value_dependent_unlike_aggregation` still passes
    on a chain whose coefficients ramp non-uniformly) but it is a property of the *balance* moving, not
    of the values moving. Do not cite the refresh as load-bearing without checking which of the two a
    given march actually does.
  - **✅ lAIR IS NOW BLOCK-AWARE — `build_air_hierarchy(block_size=…)` — AND THAT DISSOLVES THE
    RECORDED `[k, ω]` BLOCKER (2026-08-19).** Three things change together above ``block_size = 1`` and
    none suffices alone, which is the same trio `JacobiSmoothedInverse` needed and for the same measured
    reasons. The C/F splitting runs on the **cell** graph (`_cell_graph`), so a cell is entirely coarse
    or entirely fine — a field-blind split can put one field of a cell on the coarse grid and another on
    the fine one, which is what produces the degenerate Galerkin row. The local approximate-ideal solve
    carries **one right-hand side per field** (the block form of the paper's eq. 17), so the within-cell
    coupling stays inside it rather than being approximated away outside. And the FC-Jacobi smoother
    inverts each cell's own block (`_cell_block_inverse`) rather than the scalar diagonal, because on
    the coupled scalar pair that coupling exceeds the diagonal.
    - **The third of those is why the recorded blocker falls.** `solve-field-split.md` records the
      trailing Jacobian slice going **diagonal-negative at level 1**, with smoothed/convection/lAIR all
      refusing to build, and names that — not row density — as "the decisive fact". It is a **point**-
      smoother requirement: `_require_positive_diagonal` demands a positive scalar diagonal, while
      `_cell_block_inverse` demands only that each cell block be *invertible*, and says so outright
      ("a block can be perfectly well conditioned with a negative entry on its diagonal"). Pinned
      directly by `test_block_air_builds_where_the_scalar_diagonal_is_negative`: the **same** operator,
      scalar build refused, block build succeeding. So that blocker was never a property of the
      operator.
    - **`block_size = 1` is byte-identical** — all 63 pre-existing multigrid tests pass unchanged.
    - **The contraction test carries its own non-vacuity check**, comparing against the *same hierarchy
      with its block inverse stripped out*. Without that a fixture whose within-cell coupling is weak
      would pass for the wrong reason and read as evidence for the block smoother.
    - **Wired for a march as `AirReductionInverse(...)` / `AirReductionInverse` (`precondition/field_split.py`)**, a
      `trailing_inverse` factory satisfying the split's three-method contract (`n_dofs`, `apply`,
      `refactor_block`). It does **not** subclass `HierarchyBlockInverse`: that base owns a
      `SmoothedHierarchy` and refreshes by re-fitting the aggregation, this owns an `AirHierarchy` and
      refreshes via `refresh_air_hierarchy` on a frozen split. The transpose is `jax.linear_transpose`
      of the whole fixed-cycle map, since `R != Pᵀ` makes swapping the transfers by hand wrong.
      Selected on `bfs3d` by `BFS3D_TURBULENCE_INVERSE=air`.
  - **What is NOT yet re-measured: whether lAIR is any good on `bfs3d`.** Everything above is setup
    cost and a 2D contraction fixture. lAIR is opt-in in both consumers
    (`scalar_transport_preconditioner` via `scalar=ScalarAir()`, `BlockPreconditioner` via
    `velocity=ConvectionAir()`) and both default to two-level aggregation, so no shipped default moved.
- **`refresh_air_hierarchy` — the lAIR refresh that keeps the compilation signature (BUILT).** It
  re-derives an lAIR hierarchy's **values** at a new operator while holding the coarsening fixed: each
  level reuses its stored C/F split (recovered from the level's own masks) and its stored
  prolongation, and re-solves only the local approximate-ideal restriction against the new `a`. The
  prolongation must be *carried over*, not re-derived, because `_one_point_interpolation` picks each
  F-point's strongest C-neighbour by `argmax |a_ij|` — a value-dependent column choice.
  ⚠️ **The restriction's neighbourhoods must now be carried over too, and the reason this entry used to
  give for not carrying them is FALSE as of 2026-08-19.** It said the restriction's sparsity "depends
  only on the split and on `a`'s *pattern*", which was true only while the walk ran over the operator's
  full sparsity pattern — the conformance defect fixed that day (see the lAIR-neighbourhood entry
  above). The walk now runs over the **strength graph**, which reads values, so it is exactly as
  value-dependent as the split and the prolongation. `_frozen_neighbourhoods(level)` recovers each
  level's walk from its own stored `R` (row `k` holds `{g} ∪ N`, so dropping the one coarse entry per
  row returns `N`), and `_restriction_over_neighbourhoods` re-solves on it. **Structure is therefore
  preserved by construction rather than by the check** — nothing structural is re-derived at all — and
  the Galerkin `R A P` patterns below follow inductively. The result is still verified before returning
  (`_require_matching_structure`) and a
  mismatched operator raises rather than silently returning a hierarchy that would recompile. Pinned
  by `test_refresh_air_hierarchy_keeps_the_structure_and_is_a_cache_hit` (shapes preserved, values
  changed, jitted V-cycle traces once), `test_refreshed_air_hierarchy_preconditions_the_new_operator`
  (a refreshed cycle beats the stale one on the new operator, so the reused split is a real trade and
  not a no-op), and `test_refresh_air_hierarchy_rejects_a_mismatched_operator`. **Why it matters:**
  measured on the separated pitzDaily state with the production lAIR scalars, refreshing the k/ω AMGs
  is worth ~2.4× in outer cycles (30 → 13 at β=2; the flow block 30 → 29, i.e. nothing), and this is
  the only way to take that win without paying a recompile per refresh.
- **Degenerate-mesh guard (binding — validated where the graph is consumed).** Because the
  hierarchies are built once off-jit and then frozen, a degenerate mesh must fail *there*, not as a
  silently stalling runtime V-cycle. Now that the builders are operator-in, the **graph** check lives
  with the assembler: `frozen_operator.require_valid_graph` (`n ≥ 1`, matched `owner`/`nb`, in-range
  endpoints) runs when a `ConvectionDiffusionStencil` is constructed; the two build loops
  (`_build_aggregation_hierarchy` for smoothed/convection, `build_air_hierarchy` for lAIR) call
  `_require_positive_diagonal` on **every** level's operator diagonal before inverting/freezing it,
  so a zero diagonal (disconnected component, isolated/zero-volume cell, degenerate `R A P` row)
  raises `ValueError` at setup instead of baking `inf` into the frozen operator. The diagonal is
  checked *after* boundary stiffness is folded into the operator, so a boundary-only cell is
  correctly allowed. This one build-time guard is why the runtime smoothers (`_chebyshev_smooth`,
  `_jacobi_smooth`, `_fc_jacobi`) and the block-preconditioner rescales, which invert the frozen
  diagonal / the positive momentum `a_P`, need no per-apply floor.
- **`_coo_apply`'s sorted-segments promise is now per-call, not blanket (issue #110's first finding,
  fixed 2026-08-13 in `60806e8`, independently of the issue).** `sorted_segments` (default `True`)
  states whether the COO row array is non-decreasing; it is a promise, not a hint — a backend that
  acts on it returns silently wrong output if it is false. A frozen operator taken from `scipy.sparse`
  CSR is row-major and satisfies the default; the smoothed-aggregation **restriction** is the
  prolongation applied transposed, so its segment ids are the prolongation's COLUMN array, which a
  row-major COO does not leave sorted — `_smoothed_ops`'s `restrict` passes `sorted_segments=False`
  for exactly this. lAIR's restriction/prolongation are each an **independent** `scipy.sparse` CSR
  matrix (never a transpose of another), so their COO round-trip is sorted by construction and
  `_air_ops` uses the default correctly as-is — checked, not merely assumed. `_operator_matvec` itself
  does not go through `_coo_apply` at all: every level's own operator applies via `_PaddedRowOperator`
  (a padded-row product, one output per row), which needs no sortedness hint.
- **Frozen level diagonals are inverted ONCE, in `__post_init__`, not on every smoother apply (issue
  #110's second finding, fixed 2026-08-18).** `_SparseLevel`/`_AirLevel` both carry `inv_diagonal`,
  derived automatically the moment either is constructed — the same `__post_init__` +
  `object.__setattr__` pattern `FaceCellConnectivity` uses for `interior` / `safe_neighbour` (issue
  #109), so every existing constructor call site (the builders, `refit`, and the one hand-built test
  level) gets it for free without passing it explicitly.
  `_chebyshev_smooth`/`_jacobi_smooth`/`_jacobi_smooth_zero` read `level.inv_diagonal` instead of
  re-dividing; `_fc_jacobi` additionally hoists `omega * mask * inv_diagonal` out of its per-sweep
  loops (computed once per call instead of once per sweep) — `smoother_weight` stays a **solve-time** argument
  rather than a build-time field, since it is chosen per `air_multigrid_cycles` call and is not itself
  a frozen property of the hierarchy; baking it into `_AirLevel` would mean a different `smoother_weight` needs
  a different hierarchy, which is a real behaviour change this fix does not make. Verified two ways
  in `test_multigrid.py`: `inv_diagonal` matches `1.0 / diagonal` exactly on both level kinds, and each
  smoother's jaxpr, traced in isolation, closes over `inv_diagonal` as a constant and never over the
  raw `diagonal` (checked via `ClosedJaxpr.consts`, which distinguishes this from Chebyshev's other,
  unrelated scalar-derived divides that a blanket "no `div` primitive" check would have flagged too).
  - **⚠️ THE TWO `__post_init__` BODIES DRIFTED, AND THE DRIFT WAS A LATENT DEFECT, NOT JUST DUPLICATION
    (issue #271, fixed 2026-09-12).** Both originally computed the bare `1.0 / diagonal` this entry
    describes. `_AirLevel` later grew a guard (block-aware lAIR, above) — a nodal level's smoother reads
    `block_inverse` and never `inv_diagonal`, and the *scalar* diagonal a nodal level's `diagonal` field
    carries is not required to be nonzero there (`_diagonal_inverse_operator` skips
    `_require_positive_diagonal` whenever `block_size > 1`, since only each cell's own block need be
    invertible) — so a bare reciprocal could store an infinity nothing reads. `_SparseLevel` never got
    the same guard, so a nodal `_SparseLevel` (from `build_convection_hierarchy(..., block_size>1)`,
    consumed by `SimpleSmoothedInverse`/`JacobiSmoothedInverse`) could carry `inv_diagonal = inf` at any
    cell whose own scalar diagonal entry is zero or negative — inert today only because every consumer
    of a nodal level's smoother branches on `block_inverse` before ever reading `inv_diagonal`
    (`_chebyshev_smooth`, the only unconditional reader, is reachable only through
    `smoothed_multigrid_cycles`, whose hierarchies come solely from `build_smoothed_hierarchy`, always
    `block_size=1`). Both bodies now call one shared `_level_inv_diagonal(diagonal, block_inverse)`, so
    the guard cannot drift off one of the two levels again. Pinned by
    `test_block_sparse_level_stores_a_finite_inv_diagonal_on_a_zero_scalar_diagonal`
    (`tests/unit/test_multigrid.py`), the `_SparseLevel` mirror of the block-AIR fixture
    `test_block_air_builds_where_the_scalar_diagonal_is_negative` already pins for `_AirLevel`.
- **⚠️ NARROWED (2026-08-18): the defect below is specific to a PLAIN (untresholded) deep coarsening,
  not to depth itself — a strength-thresholded deep hierarchy does not reproduce it.** The prohibition
  below was against adding a depth knob at all; `HierarchyBlockInverse`'s shared base (#197) exposed
  `max_levels`/`strength_threshold` since, and `bfs3d`'s trailing-block bundle now ships
  `max_levels=20, strength_threshold=0.25` (case-level override, library default unchanged — see the
  scalability item earlier in this file). Depth *alone* (`strength_threshold=0`, the untresholded case
  this bullet is about) reproduces the original failure exactly, measured directly: no real residual
  reduction after 50+ restart cycles on the same operator this bullet's ρ(S) ≈ 1.0–1.36 was measured on.
  Depth *combined with* the threshold reached rtol 1e-8 in 11 cycles. So the reasoning below — why a
  coarse-of-coarse operator from *structural* coarsening is pathological — still stands and still governs
  the library default; what changed is that it is no longer the last word on depth in general.
- **The damped-Jacobi convection hierarchy defaults to TWO-LEVEL (binding for the library default; NOT
  for every consumer — see the note just above).** `build_convection_hierarchy(a)` builds exactly a
  smoothed fine level + a single **direct** (dense inverse) coarse solve at its own defaults.
  **`max_levels` exists** (`multigrid.py`, default `_CONVECTION_LEVELS = 2`); raising it *alone* (with
  `strength_threshold` left at its structural default) re-opens the defect below. On the fine level the
  upwind operator is a diagonally dominant M-matrix, so one damping factor `ω/λ_max` contracts
  (`_jacobi_smooth`, ρ ≈ 0.7 at high cell Peclet). A *deeper* Galerkin recursion over a **structural**
  (value-blind) coarsening is what was found broken: a coarse-of-coarse operator of a strongly
  convection-dominated problem acquires near-imaginary-axis eigenvalues that **no single-factor
  damped-Jacobi smoother can damp** — the smoother becomes non-contractive (measured ρ(S) ≈ 1.0–1.36 on
  such levels), so the coarse level must be an exact solve there. Deep, mesh-independent convection
  coarsening is the job of the reduction-based lAIR hierarchy (`build_air_hierarchy` + `_fc_jacobi`)
  where a strength threshold is not wanted. Library defaults are unchanged and both original production
  callers (the flow `TwoLevelConvectionVelocity` path and the segregated turbulence
  preconditioner) still use two levels; the deep-and-untresholded damped-Jacobi build that motivated this
  bullet was dominated on both ends (worse than two-level shallow, worse than lAIR deep) and was the sole
  source of the non-contractive-smoother defect.
