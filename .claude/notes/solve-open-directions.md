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

A sparse matvec on a 138k-row, ~38-nnz/row matrix is a few million flops — well under a millisecond on
one CPU core. The matvec actually run is `jax.jvp` through the full residual (five gradient
reconstructions per evaluation, `schemes.md`), so the Krylov operator costs roughly **10× the operator
it represents**. The matrix is not used for the matvec because *materializing* it costs ~560 global JVPs
(`sparse_jacobian.py`, colored probing at reach 3), which is why the preconditioner is refreshed only
every few steps — and the ledger already found that **staleness, not hard operators, is the dominant
linear-solve cost at low β** (the `InnerIterateCheckpointer` result: 15 cycles stale vs 1 cycle matched
at the march's hardest solve).

So the two costs are coupled: the matvec is expensive because the matrix is expensive to build, and the
preconditioner is stale because the matrix is expensive to build. Entries 1–2 break that coupling; 3–6
attack the *number* of linear solves rather than their cost; 7–9 are platform and sweep-level levers.

---

## 1. Local AD assembly of the exact Jacobian (replace coloured probing)

**What.** The residual is gather → compute → scatter (`discretization.md`). Differentiate the
*per-face / per-cell contribution* with respect to its own fixed-size, padded stencil using
`jax.vmap(jax.jacfwd(local_contribution))` over faces, and scatter the resulting small dense blocks
straight into a COO/BCOO Jacobian. The gradient-reconstruction stage (least-squares + `sweeps` of the
skewness correction) composes the same way: its per-cell linear map over the reach-1 ring is itself a
small dense block, and the chain rule over two stencil rings is a sparse product of two sparse operators
that can be formed once per materialization.

**Why it should win.** Cost is ~(stencil width) × one fused *local* evaluation, with no colouring and
no de-compression, against 564 full residual JVPs. On bfs3d that is a 15–30× cheaper materialization
in flop terms, and the work is embarrassingly parallel (ideal on GPU, where probing is a sequence of
full-mesh passes).

**What it does not change.** The result is the same AD Jacobian to rounding — the exactness the
project's Principle on hand-linearization protects is preserved, because nothing is derived by hand;
only the *order* of differentiation and summation changes.

**Pre-registered measurement.** At `bfs3d` `state-00049`: assemble by both routes, report
`‖J_local − J_probe‖_max / ‖J_probe‖_max` (expect ≤ 1e-13) and wall time for each, CPU, warm compile.
Then the same on pitzDaily. A pass is agreement to rounding at < 1/5 of the probe's wall time.

**Risks.** Padding ragged stencils to a fixed width wastes work on meshes with high-valence polyhedra;
measure the padding ratio. Boundary-patch closures (`boundary.md`'s per-patch fold) need their own local
blocks. The reach-3 pattern is required (ledger) — the local route reproduces it exactly, so this is a
non-issue, but assert it in the test.

## 2. Run the Krylov matvec on the materialized `J`, not the JVP

**What.** Once (1) makes assembly cheap enough to run **every Newton iterate**, hand GMRES
`lambda v: J @ v` (sparse) and the adjoint `Jᵀ @ v`, with the preconditioner built from the same `J` at
the same iterate.

**Why it should win.** ~10× cheaper per matvec (see table), and it deletes the staleness mechanism the
ledger identifies as the low-β wall: the preconditioner is matched to the operator at every inner
iterate by construction, with no refresh policy, no cost trigger and no `beta_rel_change` heuristics.
The adjoint's single tight transpose solve gets the same speed-up.

**Distinguish from the refuted entries.** The `jax.linearize` refutation concerned hoisting primal work
*inside the JVP path*; this removes the JVP path from the linear solve. The block-CSR refutation
concerned a BCOO-vs-block-CSR *implementation* of a sparse matvec, not whether the sparse matvec should
replace the JVP. Neither bears on this.

**What it does not change.** `J` is exact at the iterate, so the Newton direction and the IFT adjoint
are identical to the JVP path to rounding; `custom_vjp` / `linear_transpose` plumbing in `implicit.py`
is unaffected in principle, but the preconditioner's `stop_gradient` discipline must extend to `J`
*as used in the PC* while `J` *as used in the matvec* stays on the differentiable path.

**Pre-registered measurement.** `pitzDaily` ramp arm (the shipped 40-step configuration): identical
step count and `x_r/h` 8.0686; report Krylov cycles, matvec wall, materialization wall, total wall.
Then bfs3d. The comparison that matters is total wall at identical trajectory.

**Risks.** Memory: `J` on bfs3d is ~5M nonzeros × 8 B = 40 MB — trivial. The failure mode is (1) being
less cheap than hoped on a real polyhedral mesh; then materialize every *k* iterates and keep the JVP
matvec for the in-between iterates — still a win on staleness.

## 3. Tangent-predictor continuation along the Reynolds ladder / homotopy

**What.** Each rung (or `ResidualHomotopy` station) currently seeds the next from the previous *root*
with zero slope. Compute `dx/dλ = −J⁻¹ ∂R/∂λ` at the converged root — one linear solve with the
operator you already have factored/preconditioned — and seed the next station at
`x₀ + Δλ · dx/dλ`. A secant predictor from the last two roots costs nothing at all.

**Why it should win.** The ledger's own evidence: rung 1 closed at `|R|` 7.2e-6 and rung 2 opened at
4.5e-2, *six thousand times worse*. A first-order predictor makes the opening residual `O(Δλ²)`; most
of the 12–16 "comfortable" β-ladder steps per rung are spent recovering exactly that loss. This is the
single most natural lever in a differentiable solver and is absent from the notes.

**What it does not change.** A seed is a seed; the converged state and its adjoint are unaffected.

**Pre-registered measurement.** pitzDaily shipped ladder: opening `|R|` per rung, outer steps and
cycles per rung, total wall; arms: no predictor / secant / tangent. Expect the tangent arm to cut rung
2 and 3 steps by roughly half.

**Risks.** A predictor overshoots across a bifurcation or a turning point in the ladder; the SER control
already re-damps a bad step, so the downside is one wasted step.

## 4. Eisenstat–Walker forcing terms

**What.** Replace the fixed inner tolerances (`rtol=0.3` in `VCYCLE_LINEAR_SOLVE` /
`FACTORIZATION_LINEAR_SOLVE`, `1e-3` in `_INEXACT_CONTINUATION_SOLVER`) with Eisenstat & Walker (1996)
choice 2, safeguarded: `η_k = γ (‖R_k‖/‖R_{k−1}‖)^α`, `γ = 0.9, α = 2`, floors and the standard
"do not let the tolerance drop below the previous step's achieved residual" guard.

**Why it should win.** Standard result for inexact Newton on pseudo-transient CFD: 20–40 % fewer
Krylov iterations by over-solving less far from the root and solving tightly only where quadratic
convergence can use it. It also aligns the inner tolerance with where staleness already makes the
direction inexact.

**What it does not change.** The stopping test on the outer residual and the adjoint's tight solve
are untouched.

**Pre-registered measurement.** Both flagship cases, identical `x_r/h`; cycles and wall vs the fixed
tolerance. ~20 lines; cheapest entry in this file.

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

## 6. Krylov subspace recycling across Newton steps (GCRO-DR)

**What.** Keep a ~10–20-vector approximate invariant subspace (harmonic Ritz vectors of the slowest
modes) from each GMRES solve and deflate it from the next (Parks, de Sturler, Mackey, Johnson & Maiti
2006). Consecutive Newton systems share their slow modes; at low β those few modes are what turn a
1-cycle solve into 15.

**Why it should win.** The ledger's cell-block SVD puts the near-null directions in ω, coherently
located in low-`k` regions — a *small* subspace, which is exactly what recycling captures and a V-cycle
cannot. It is the standard remedy for sequences of related nonsymmetric systems in Newton–Krylov CFD.

**Cost.** Not in `lineax`; a `GCRO-DR` solver would be written in-house (the restart machinery and the
`linear_transpose` requirement constrain the design — recycling the *transpose* for the adjoint is a
separate question, and the adjoint is one solve, so leave it un-recycled).

**Pre-registered measurement.** Replay the seven ≥4-cycle solves `InnerIterateCheckpointer` captured on
bfs3d, with and without a recycle space carried from the preceding solve; report cycles and true residual.

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
entries 1, 2 and 7 compound on GPU and are worth more there than their CPU measurements will show;
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
work on the finished ones; pair with entry 3 so members start closer together.

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
  count as the primal — irrelevant once entry 2 removes the JVP from the Krylov loop.

## Where this file does not go

Not `.claude/rules/` (it would auto-load into every session); not `docs/` (user-facing, and these are
hypotheses); not only an issue (an issue is not greppable from a checkout, and the project's rule is
that findings live in tracked files). The right shape is this note as the backlog, one issue per entry
once it is in progress, and the result filed in a rule or the refutation ledger when it is measured.
