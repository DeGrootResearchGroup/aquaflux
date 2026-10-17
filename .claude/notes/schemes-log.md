# Investigation log — `aquaflux/schemes/` (gradient reconstruction and the other swappable numerics)

> Split out of `.claude/rules/schemes.md` (2026-10-17). **Lives in `.claude/notes/`, outside the
> auto-loaded `.claude/rules/` tree, so it never auto-loads.** It holds the dated measurements and
> investigation trails behind the gradient schemes — probes, marches, cost profiles, and several
> rounds of "found the cause" that a later measurement qualified or retracted — kept because a
> wrong-but-plausible finding gets re-derived if the trail that refuted it is deleted rather than
> archived. Each `##` heading below names the part of `schemes.md` the entries were moved from; that
> file holds the current, load-bearing status and points back here. The issue-#435 tetrahedral
> investigation has its own log, `.claude/notes/schemes-435-tetrahedral-gradient-log.md`.
>
> ⚠️ **Everything below is a point-in-time record, verbatim as it stood in `schemes.md`.** Symbols it
> names may since have been renamed or deleted — among them `coupled_continuation`,
> `coupled_lu_continuation`, `coupled_amg_continuation` and `mass_flow_coupled_continuation` (all now
> `coupled_step`), the global k-positivity cap `positive_k_limit` and its projection switch (deleted
> 2026-10-14), the line search's `max_alpha` cap, the `petsc` bundle and `Ilu0`. Check a symbol, default
> or tolerance against the source before quoting it. Cross-references such as "above", "below" or
> "the entry above" refer to the layout of `schemes.md` when the entry was written.


## The swept gradient sets the Jacobian's stencil (2026-08-16)

**⚠️ THIS ENTRY WAS ALREADY STALE WHEN FOUND (2026-08-20) — the diagnostic is gated, and has been for
longer than this file said.** It read "`SweptGradientSolve`'s `warn_tol` diagnostic fires
**unconditionally at `sweeps=1`**"; `solve` in fact skips it entirely when `sweeps <= 1`
(`... and self.sweeps > 1`), where it would carry no information rather than a little. What remains
true, and is the reason the entry is kept rather than deleted: the diagnostic measures the residual
from *before* the last update, so at one sweep that is the right-hand side itself and the ratio is
exactly 1 whatever the mesh — which is why gating it was the fix. **If you meet that warning in an old
log from a skew-free mesh, it was never evidence of non-orthogonality.**


## The sweeps are a `lax.scan` (2026-08-22)

**Two tests had to change lens, and neither was weakened — this is the instructive part.**
- `test_the_swept_solve_spends_one_apply_fewer_than_its_sweep_count` counted applies with a **Python
  counter inside the operator**. A scan body is traced **once** however many times it executes, so
  that counter reads `1` at every sweep count and the test would have passed whatever the peel did.
  It now reads the scan's **trip count out of the jaxpr**, which is what actually executes.
- `test_peeling_the_zero_apply_leaves_the_answer_BIT_identical` compared the peeled solve against an
  unpeeled **Python loop**. With one arm scanned it was comparing loop constructs rather than the
  peel, and failed by ~1 ulp. The reference is a scan too now, and bit-equality is restored — the peel
  itself is exact, since `A·0` is exactly zero.


## The Jacobian's gradient can be cheaper than the residual's (2026-08-22)

**Per-matvec headroom, priced first** (pitzDaily 12225 cells, `state-00082`, `eqx.filter_jit`, warm,
min of 7, x64, compiled ILU(0) live):

| gradient sweeps | operator applies | `R` (ms) | `jvp(R)` (ms) |
|---|---|---|---|
| **4 (shipped)** | 3 | 3.85 | **8.27** |
| 3 | 2 | 2.97 | 6.71 |
| **2** | 1 | 1.97 | **4.63** |
| 1 | 0 | 1.33 | **3.72** |

A sweep costs ~0.9 ms in the residual and ~1.6 ms in the tangent, because **`jvp` spends two operator
applies per sweep — a primal and a tangent — against the residual's one**. (The residual is nonlinear
in the reconstructed gradient, through `nu_t` and the limiter, so the primal gradient cannot be
eliminated.) Hence the sweeps weigh nearly twice as much on the differentiated path as on the evaluated
one, and a march pays a tangent per Krylov iteration against a residual once per step.

**On a whole march (three arms back to back in one process, `simplesmooth` bundle, three Reynolds
rungs, `forward_rtol` 0.3, probe reach 5, residual held at swept-4):**

| Jacobian | steps | Krylov cycles | worst step | wall | `x_r/h` | field difference vs control |
|---|---|---|---|---|---|---|
| **full (exact) — control** | 71 | 404 | 14 | 713.3 s | 8.069 | — |
| **swept-2** | **71** | **404** | **14** | **644.0 s (−9.7 %)** | 8.069 | ≤ 1.2e-07 L2 |
| swept-1 | 71 | 410 | 15 | 644.4 s | 8.069 | ≤ 2.3e-06 L2 |

- **swept-2 is free: identical step count, identical cycle count, identical worst step, and the same
  root.** The step tables are equal column for column — `beta`, inner count, cycles, `alpha`, flags and
  residual to four figures — with only the wall-clock column moving. The whole 9.7 % is cheaper
  matrix-vector products.
- **THE KNEE IS AT 2, AND swept-1 IS PAST IT.** It costs 6 cycles (+1.5 %) and returns nothing in wall
  clock (644.4 s against 644.0), even though its tangent is a further 20 % cheaper per matvec. That is
  the exchange rate this idea lives or dies on, seen directly: below some accuracy the extra iterations
  eat the cheaper product. **Run the ladder one rung past where it stops helping** — the two endpoints
  alone would have read as "keep going".
- **Why 2 is exactly right here is predictable in advance, and that is the useful part.** This mesh's
  measured contraction rate is `rho = 5.07e-03`, so a swept-2 gradient departs from swept-4 by
  `~1.4e-05` relative and a swept-1 one by `~4.0e-03`. Against a linear solve stopping at 30 %, the
  first is invisible and the second is at the edge. **Size the cap from `contraction_rate`, not by
  feel** — and note the residual's own calibrated count on this mesh is also 2, so at `tol=1e-4` the
  two questions happen to give the same answer here; they will not in general.
- ⚠️ **One run per arm**, but the wall clock here is better anchored than usual. The **step and cycle
  counts are contention-immune and carry the verdict** — bit-identical trajectories are not a
  wall-clock claim — and the three arms ran back to back in one process. Beyond that, the control arm
  was run **twice**, in two separate invocations on the same machine, at **713.3 s and 720.1 s** with
  identical steps, cycles and reattachment: a **~1 % march-level repeatability**, which is what puts
  the 9.7 % comfortably outside it. That is a much finer instrument than this project's recorded ~15 %
  *per-application* noise floor, and the difference is worth knowing: a whole march averages away the
  per-application spread that a single-state probe is at the mercy of.
- ⚠️ **Measured on `simplesmooth`, not on the `petsc` bundle this case shipped until the same day** (see
  the case comment: `petsc` no longer marches it). Cycle counts do not transfer across preconditioner
  families.
- **The adjoint is untouched, by construction rather than by care.** `root_adjoint`'s backward rule differentiates
  the residual it was handed at the converged state and never consults the Newton step, so a cheaper
  forward operator cannot reach a gradient. See `.claude/rules/solve-globalization.md`.
- **Not measured: a genuinely skewed mesh, where `rho` is 0.14–0.26 rather than 5e-03.** There the
  residual needs many more sweeps and the *ratio* between the two counts should be much larger — which
  is where this lever should pay most, and where it is also most likely to start costing steps. Both
  shipped cases are near-orthogonal (`bfs3d` calibrates to `k = 1`), so neither can answer it.
  **The UV reactor is the wrong instrument for it** despite being the one skewed mesh in the tree: at
  1.6M cells a single march is far too expensive to walk a ladder on, and the ladder — not any single
  arm — is what carries the verdict here (the knee at 2 was only visible because 1 was also run).
  **What this needs is a SMALL skewed case**, on the order of the two existing ones, and the natural
  source is an automatically-generated mesh over a simple geometry rather than a perturbed grid: a
  synthetic perturbation makes `rho` a knob, where the question is what a real mesh generator produces.
  The strongest form is a **re-mesh of a geometry already validated on a block mesh**, so mesh quality
  is the only variable against a known answer and the metric stays the judge.
  **⚠️ BUT THE GEOMETRY MUST NOT BE AXIS-ALIGNED, AND `bfs3d` IS — so re-meshing IT would produce
  another orthogonal mesh.** An automatic hex mesher distorts cells only where it must **snap** to a
  surface the background mesh does not already conform to; a box with an axis-aligned step castellates
  and stops, leaving the background hexes intact. This project's own two meshes are the demonstration:
  `bfs3d` is a pure box and is skew-free to `1.9e-12`, while `pitzDaily` — the same class of geometry
  but with an **inclined lower wall and a contraction** — reaches `7.5e-02`. The non-alignment is where
  the skew comes from. So the candidate geometry needs a genuinely angled or curved surface (and
  refinement-level transitions, the other source, help), and `pitzDaily`'s own geometry is the better
  host for the idea than `bfs3d`'s for exactly this reason.
  **⚠️ Whatever is chosen, MEASURE THE MESH BEFORE BUILDING A CASE ON IT** — generate it, then run
  `contraction_rate` and a skew census. It costs minutes, and the failure it prevents is a case built
  around a mesh that turns out orthogonal, which is precisely why `bfs3d` cannot answer this question
  despite being the newer and better-tuned of the two.

**And the other half of the question, measured on the same case: the RESIDUAL's exactness buys nothing
here either — because the sweep series has already converged by two.** Same harness, `residual` group,
each arm's probing reach moved with its sweep count (`sweeps + 2`, so the probe is not the thing
degrading):

| residual | probe reach | steps | cycles | wall | `x_r/h` | field difference vs swept-4 |
|---|---|---|---|---|---|---|
| swept-2 | 4 | 71 | 391 | 545.8 s | 8.069 | ≤ 2.2e-06 L2 |
| **swept-4 (shipped)** | 5 | 71 | 404 | 720.1 s | 8.069 | — |
| swept-6 | 7 | 71 | 416 | **1059.1 s** | 8.069 | ≤ 7.1e-07 L2 |

**A 1.9× cost spread across the ladder, for a converged field that moves by ~1e-06 relative and a
reattachment length that does not move at all.** Read it as the sweep series having converged rather
than as accuracy being unnecessary: swept-2 and swept-6 each sit ~1e-06 from swept-4 *and from each
other*, which is what "all three are the same discretization" looks like. The direct answer to "what is
the exactness buying?" on this mesh is **nothing measurable** — and that is a property of `rho = 5.07e-03`,
not a general licence.
⚠️ **Do NOT read this as an argument to lower the shipped `sweeps=4`.** The same count is *insufficient*
at 30 % skew, the failure is silent (there is no residual test to trip), and the count also sets the
residual's Jacobian reach, which each case's probing reach is matched to — so moving it is a change to
the discretization that the case's reattachment result would have to be re-validated against. The
per-case answer is `CorrectedGreenGauss.calibrated`, which already gives **2** on this mesh.
⚠️ **The wall-clock column of this group mixes two changes** — the arms differ in probe reach as well as
in sweep count, deliberately, since a probe shorter than the residual's stencil would fold the far
coupling onto near entries and degrade the arm for the wrong reason. The **field** comparison is what
this group is for; the seconds are indicative.
**Contrast the two groups, because that is the finding.** Both cheapen the reconstruction and both leave
the answer where it was — but the residual group does it by *changing the discretization and being
lucky that this mesh does not care*, while the Jacobian group does it *without touching the
discretization at all*. Only the second is safe on a mesh nobody has calibrated.


## Responsibility → `limiter.py`

  - **pitzDaily under the new softening (2026-10-09, commit `f084c65`, Linux x86_64, 4 cores, jax
    0.11.2, the case's own `case.yaml` defaults — Venkatakrishnan on momentum only, `softening` 0.05,
    scale = inlet speed 10 m/s, so `eps` = 0.5 m/s; one run):** 31 steps, 202 restart cycles, final
    row-scaled `|R|` 8.511e-06, `x_r/h` 8.07, peak `nu_t/nu` 418, rel. L2 against the OpenFOAM field
    `U_x` **0.017** / `U_y` **0.009** — against the last recorded run under `vol K³` (K = 5): 31 / 202,
    7.84e-06, 8.07, 418, **0.019 / 0.010**. As the issue predicted, at this scale the limiter is all
    but inactive on this smooth flow, and the velocity field moves slightly toward OpenFOAM's.
    `bfs3d` and `bfs3d_species` (which also limit) are NOT re-run: their meshes are not in the
    repository, so their recorded numbers predate this change.


## Responsibility → `projected_stencil.py`

  - **The target DROPPED its local Schur correction, 2026-09-23**
    (`validation/tetrahedral_gradient_ab/reference_target_probe.py`, geometry binding, one run, at
    `blend=0.75`). Three targets differing only in the per-cell block: with the Schur correction (the
    former target) **2.83 s** to build, max eig `+5.42e-4`, worst retention 0.4698, smooth-field error
    1.50e-5; **without it (shipped) 0.05 s, `+5.59e-4`, 0.4671, 1.51e-5**; compact Green--Gauss `V·I`
    0.02 s, **`+1.30e-3`**, 0.4777, 1.57e-5. The correction costs 2.83 s of a 2.88 s build and moves
    nothing it is aimed for (3 % on the damping eigenvalue, 0.3 % on the weights); compact is cheaper
    still and gives away **2.4x** on the damping eigenvalue, which is the quantity the scheme exists to
    control. At `blend=0` all three arms are bit-identical, as the geometry requires — a wiring check
    for that probe. **`boundary_weight` was DELETED with the correction**: with it off the block is
    bit-identical under all three Hessian closures, so the setting reached nothing.
    ⚠️ **Every projected-scheme measurement recorded before that date was taken under the Schur-corrected
    target** — the tetrahedral marches, pitzDaily's 31 steps / 198 cycles, bfs3d's `x_r/h` 8.3611 and the
    blend sweep. **The tetrahedral coupled march was re-run under the new target and is unchanged** --
    9 anchor + 16 target steps at `alpha = 1`, no escalation, `|R|` 5.82e-6 in 164 s against the old
    target's 16 target steps, 5.8e-6, 160 s. The two hexahedral cases were **not** re-run.

  - **Measured on the tetrahedral duct (2026-09-22, one run each):** laminar from rest **13 steps,
    5.0e-9, 19 s** at `blend=0.75` and 13 steps, 4.8e-9 at `blend=0` — the same step count as
    `CorrectedGreenGauss`, against `HessianCorrectedGradient`'s 287 s and the multiple-correction
    scheme failing; coupled RANS **16 target steps at `alpha = 1`, 160 s** against Betchen's 1583 s
    and both multiple-correction arms at `inf` by anchor step 3. Damping under the pressure's own
    conditions: no flipped diagonal at any blend, largest eigenvalue 1.9–2.2e-4 against
    `CorrectedGreenGauss`'s 9.0e-4 and the multiple-correction scheme's `+2.17`.
  - **And it costs nothing on the hexahedral case the shipped scheme already suits**
    (`PITZ_GRADIENT=projected`, one run, 2026-09-22): **31 steps and 198 cumulative cycles against the
    shipped `multcorr`'s 31 and 200**, `x_r/h` 7.9857 against 8.0686 (OpenFOAM 7.7409, so marginally
    closer), peak `nu_t` 418.9 against 418.0 (OpenFOAM 422.9), `ux`/`uy` errors unchanged at
    0.0191/0.0102. ⚠️ No wall-clock comparison: the baseline's timing is from another session and
    bundle. This is also the second case to exercise the per-field turbulence binding.
  - **`bfs3d` (orthogonal, graded hexahedra, 23040 cells) reproduces its baseline and costs ~10 %
    more** (`BFS3D_GRADIENT=projected`, one run, 2026-09-22): mid-span `x_r/h` **8.3611** against the
    shipped `CorrectedGreenGauss`'s 8.36 (OpenFOAM 7.2434), same spanwise shape including the
    side-wall corner separation, peak `nu_t` 149.8 against OpenFOAM's 146.9, in 31 steps / 224
    cumulative cycles against 28 / 205.
    ⚠️⚠️ **AND ITS CONDITIONING GUARD FIRED THERE: 928 of 23040 cells, worst condition number
    2.59e7.** The near-wall cells of a graded mesh are thin, so a two-hop stencil through them is
    nearly flat and fitting a quadratic on it is ill-conditioned -- the weights on those cells are
    large, which is the failure this scheme exists to avoid elsewhere. It still converged to the
    right answer here, so the threshold reports rather than refuses; what is **not** measured is
    whether a larger `reach` or a blend nearer 0 (smaller weights) is the right answer on a stretched
    mesh. Read the warning as a real property of that mesh, not noise.
    **✅ SWEPT, AND THE BLEND IS A MARGIN THERE RATHER THAN A CLIFF**
    (`validation/bfs3d_openfoam/gradient_weight_probe.py`, one run, 2026-09-22): the flagged set is
    the same 1056 cells at every blend -- the fit is the **stencil's**, not the aim's, so no blend
    changes the conditioning -- and **no blend flips a damping diagonal**. What moves is the worst
    retention of the compact two-point difference: `0.933` at blend 0, `0.898` at 0.25, `0.848` at
    0.5, **`0.792` at the default 0.75**, `0.735` at 1. So the guard is informative on a graded mesh
    and the default is comfortable; blend 0.25 is the cheap extra margin if one is wanted, and 0 is
    the least accurate end on non-polynomial fields. Left at 0.75.
    ⚠️ **Do not compare weight MAGNITUDES across meshes**: a weight carries units of 1/length, so this
    mesh's hundreds and the tetrahedral duct's ~1 are the cell sizes, not a difference in behaviour.
    The scale-free quantity is the retention (or the ratio to a reference reconstruction's own
    weights, which is what the duct's `2900x` is).


## Responsibility → `HessianCorrectedGradient`: slivers and the local Schur block

  **Robustness on degenerate cells (measured 2026-08-22, VALID single-cell fixture).** At a volume
  ratio of 1.9e+06 the shipped swept solve leaves **1.09e+00** on the sliver cell and **4.94e-07** in
  the far field, against a clean mesh's 8.6e-15 — so a single bad cell does cost the scheme its
  exactness globally. But this is an **iteration** effect and not a discretization one: an exact
  Krylov solve on the same mesh gives **1.79e-10** on the sliver and **2.41e-15** far. The assembled
  system holds the right answer and the fixed-sweep Richardson fails to find it, because the
  preconditioner is `A_gg`'s diagonal block while the operator is the Schur complement — and on a
  cell whose volume vanishes the neglected elimination term stops being a perturbation. Two guards
  are absent and worth knowing: `jnp.linalg.inv` on the per-cell blocks has no singularity check, and
  `interpolation_factor` guards `d → 0` on boundary faces only; neither has been observed to trigger.

  **⚠️⚠️ THE DEFAULT WENT `True` → `False` → `True` IN ONE DAY (2026-08-22). It is `True`. The middle
  step is kept because what moved it back is a change to the SCHEME, not a re-reading of the same
  evidence — with the Hessian solved as nine components the divergence below is real and reproducible;
  with six it is gone and the other arm diverges instead. Read the entries in order.**

  **⚠️⚠️ THE NINE-COMPONENT MEASUREMENT — IT DIVERGED ON A REAL
  MESH, AND THE REASON INVERTS THE OBVIOUS INTUITION.** On a 1.6M-cell snappyHexMesh reactor,
  measured in one process against a known analytic gradient
  (`validation/uvreactor_openfoam/schur_block_diagnosis.py`):

  | | median | p99 | max | cells > 100 % error |
  |---|---|---|---|---|
  | `A_gg` block | 1.137e-07 | **4.876e-05** | **7.011e-02** | 0 |
  | Schur block | **8.957e-08** | 7.398e-05 | **5.599e+18** | **4599 of 1 635 909** |

  A 27 % better median, a worse p99, and 0.28 % of the mesh diverging outright. ⚠️ **"A population,
  not an edge case, so no per-cell guard is proportionate" was the reading here and it is REFUTED**:
  the divergent mode's participation ratio is ~1 cell (below), so the population is one bad row's
  contamination spread by 20 sweeps, and a per-cell guard is exactly the right shape after all.

  **The failing cells are ORDINARY.** Four- and five-faced cells (tetrahedra and pyramids) against
  the mesh's median of six, volumes 0.24–0.42 of median (one is 9× *larger*), planarity 0.89–0.9996,
  closure ~1e-22. Not slivers, not warped, not invalid — every geometric hypothesis was refuted.

  **⚠️ AND THE SCHUR BLOCK IS THE MORE ACCURATE ONE. That is the finding.** Against the true local
  Schur block — obtained preconditioner-independently by lighting one cell's degree of freedom and
  reading back its own row:

  | cell | faces | `A_gg` vs truth | Schur vs truth | cond Schur | error |
  |---|---|---|---|---|---|
  | 1595711 | 4 | 3.284e-01 | **2.452e-02** | 2.6e+03 | 5.6e+18 |
  | 60355 | 5 | 2.535e-01 | **6.025e-02** | **6.37** | 1.4e+17 |

  **⚠️⚠️ SOLVING THE HESSIAN AS SIX COMPONENTS REMOVES THE DIVERGENCE ENTIRELY — AND SWAPS WHICH ARM
  DIVERGES (measured 2026-08-22).** Reproduce with
  `UV_MESH=<polyMesh> validation/run_case.sh validation/uvreactor_openfoam/schur_block_diagnosis.py`
  — the run logs are gitignored, so the harness and its defaults are the record. Same 1.6M-cell
  `snappyHexMesh` reactor mesh both times, same analytic quadratic, scheme defaults of 20 outer
  sweeps and `UV_INNER=12` on the Hessian, both preconditioner arms in one process. Judged against the analytic gradient:

  | Hessian | preconditioner | median | p99 | max | cells > 100 % |
  |---|---|---|---|---|---|
  | nine components | `A_gg` block | 1.137e-07 | 4.876e-05 | 7.011e-02 | 0 |
  | nine components | Schur block | 8.957e-08 | 7.398e-05 | **5.599e+18** | **4599** |
  | six components | `A_gg` block | 8.147e-08 | 4.918e-05 | **5.787e+05** | — |
  | **six components** | **Schur block** | **8.055e-08** | **4.320e-05** | **5.184e-03** | **0** |

  **The Schur block improves by twenty-one orders in the max and no longer diverges anywhere**, and
  the `A_gg` block — which was the safe arm — now blows up instead. Every diverging cell in both
  regimes is a four-faced cell at ~0.2 of the median volume, so it is the same population changing
  hands rather than a new one.

  **This is a controlled comparison, and that was checked rather than assumed.** Diffing
  `aquaflux/schemes/gradient.py` across every commit between the two runs, the only change to the
  reconstruction other than the symmetry reduction is `outer`'s `use_local_schur_block` default —
  which the harness passes explicitly on both arms, so it is inert here. The two intervening merges
  add opt-in paths (`CoupledBlockSweep`, the probe's gradient narrowing) that this harness does not
  select.

  **Consequence: `local_schur_block=False` is now the DOMINATED option, which inverts the default
  question rather than settling it.** The reason to prefer `A_gg`'s block was that the Schur block
  diverged on this mesh; it does not any more, and `A_gg`'s does. Note also that the under-resolution
  warning fires on the `A_gg` arm at 20 sweeps and not on the Schur arm — the fixed sweep count is
  visibly short for it, which is the same failure seen from the solver's side.

  **Why the representation should decide this at all**: the elimination term is
  `A_gH A_HH⁻¹ A_Hg`, and with the unsymmetrized Hessian `A_HH`'s per-cell block was neither symmetric
  nor definite, so the correction subtracted from `A_gg`'s block could be wrong-signed. The reduction
  makes that block `(A_P E)ᵀ(A_P E)` — SPD by construction — so the correction is a well-signed one.
  ⚠️ That is the mechanism the measurement is consistent with, **not** one this run isolates: the
  reduction also changes `S` itself, since a different reduction of an over-determined system picks a
  different solution. Do not report it as demonstrated.

  **THE RATE INVERTS WITH THE ARMS, AND PREDICTS BOTH MAGNITUDES.** `rho(I - P⁻¹S)`, 20 power
  iterations, half-budget estimate beside it:

  | Hessian | `A_gg` block | Schur block |
  |---|---|---|
  | nine components | 0.7148 ✓ | **6.2679 ✗** |
  | six components | **2.0397 ✗** (half 2.0199) | **0.3070 ✓** (half 0.2223) |

  `2.04²⁰ ~ 1.5e+06` against an observed max of 5.8e+05, and `6.27²⁰ ~ 1e+16` against 5.6e+18 — the
  rate accounts for the error magnitude in both regimes, which is what ties the spectral measurement
  to the reconstruction one rather than leaving them as two separate observations. The Schur block
  under six components has the best rate of the four. ⚠️ Its half-budget estimate is the least
  settled of them (0.2223 against 0.3070, a ratio of 1.38 where the others are within 1 %), so treat
  **0.31 as approximate** — unambiguously below one, not accurate to two figures.

  **⚠️⚠️ AND THE FAILURE IS LOCAL, NOT GLOBAL — this REFUTES the reading recorded here on
  2026-08-22, which was mine and which the measurement built to test it overturned.** The dominant
  mode's participation ratio:

  | block | effective cells | top 1 | top 10 |
  |---|---|---|---|
  | `A_gg` (the diverging arm) | **1.1 of 1 635 909** | **0.933** | 0.999 |
  | Schur | 16.9 | 0.091 | 0.688 |

  **Ninety-three per cent of the divergent mode's energy sits on a single cell.** So `rho > 1` was
  never evidence of a whole-mesh cause, a per-cell guard would work, and the earlier "the diverging
  cells are geometrically ordinary because they are merely where the dominant eigenvector has
  support, which is why no per-cell guard would help" had the inference backwards: a mode on one cell
  is exactly what a per-cell guard catches. The population of 4599 was the contamination of a tiny
  set spread by 20 sweeps, which is what the dilation test was added to check.

  **The general lesson, and it is the transferable one: a spectral radius is a whole-mesh scalar and
  says nothing about whether its cause is whole-mesh.** Reading locality out of it requires the
  eigenvector, which costs nothing extra once the power iteration is already running.

  **THE RELAXATION LADDER CONFIRMS THE SPECTRAL PICTURE INDEPENDENTLY, and rules relaxation OUT for
  this arm.** `rho(I - w P⁻¹S)` under the Schur block, against what the two-point model
  `max|1 - w·lambda|` over `lambda ∈ [0.693, 1.065]` predicts from the governing eigenvalue and the
  modulus bound:

  | `w` | 1.00 | 0.80 | 0.50 | 0.25 | 0.10 |
  |---|---|---|---|---|---|
  | measured | 0.3070 | 0.4597 | 0.6170 | 0.7863 | 0.9089 |
  | predicted | 0.3070 | 0.4456 | 0.6535 | 0.8267 | 0.9307 |

  Within 6 % at every rung, and **monotone increasing** — every amount of under-relaxation makes this
  arm strictly worse, which is what an all-positive spectrum requires and is the opposite of what a
  diverging arm needs. So Betchen & Straatman's relaxation requirement, which is stated for *their*
  block-Jacobi on an arbitrary grid, is **not** what this configuration wants: it converges undamped
  and damping only slows it. The prediction is a genuine one — the model was written down from the
  `w = 1` measurement alone, before the other four rungs were computed.

  The model also says the optimum is *above* one (`w = 2/(lambda_min + lambda_max) = 1.14`, giving
  `rho ~ 0.21` against 0.31). **Not measured, and not proposed**: a 30 % rate improvement bought by a
  knob calibrated from a two-point spectral estimate on one mesh is exactly the kind of tuning this
  project's own record shows sitting one grid step from a cliff.

  ⚠️ **Cost note for anyone re-running the ladder:** it used to build a fresh `jax.jit` per rung,
  closing `w` in as a constant — so every rung recompiled a program capturing 2.3 GB of constants at
  this mesh size, which dominated the twenty applies it then ran. `w` is now a traced argument and one
  compilation serves the whole ladder.

  **⚠️ THE SIGN TEST MEASURED THE WRONG EIGENVALUE, AND ITS PRINTED VERDICT IS THE OPPOSITE OF THE
  TRUE ONE.** The test was meant to split the two candidate fixes: a positive governing eigenvalue of
  `P⁻¹S` means some relaxation stabilizes the arm, a negative one means none does. It power-iterated
  over `P⁻¹S` and reported `|lambda_max| = 1.0612`, `Rayleigh +1.0597`, "relaxation can stabilize it"
  for `A_gg` — and `1.0649 / +1.0634` for the Schur arm, i.e. **nearly the same answer for an arm that
  diverges at `rho = 2.04` and one that converges at `rho = 0.31`.** Two arms whose behaviour differs
  by that much returning the same number is the tell.

  **A power iteration over `P⁻¹S` returns the largest-MODULUS eigenvalue. The sweep is governed by the
  eigenvalue FURTHEST FROM 1.** Those are different eigenvalues here, and the first is uninformative
  about the second: on this operator most cells have `P` nearly exact, so there is a large cluster of
  eigenvalues near `+1` — that cluster is what both iterations found, which is why both arms returned
  ~1.06.

  **Solved properly, `rho = |1 - lambda|` admits two candidates and the modulus bound kills one:**

  | arm | `rho` | candidates | `\|lambda\|_max` | survivor |
  |---|---|---|---|---|
  | `A_gg` | 2.0397 | `-1.0397` or `+3.0397` | 1.0612 | **`-1.0397`** (`+3.04` exceeds the bound) |
  | Schur | 0.3070 | `+0.6930` or `+1.3070` | 1.0649 | **`+0.6930`** (`+1.307` exceeds the bound) |

  So `A_gg`'s governing eigenvalue is **negative**, and `1 - w·lambda > 1` for every `w > 0`: **no
  positive relaxation stabilizes that arm** — the exact opposite of what the harness printed. Betchen
  & Straatman's own under-relaxation prescription would not have rescued it.

  **The instrument is fixed rather than annotated.** The governing eigenvalue is now read from the
  Rayleigh quotient of the `I - P⁻¹S` iteration already being run for `rho` — whose dominant mode *is*
  the governing one, and on which `I - P⁻¹S` acts as `1 - lambda` — so `lambda` comes out directly and
  free. The `P⁻¹S` iteration is kept, relabelled as the modulus bound it actually provides, because it
  is what discriminates the two candidates; when the two routes disagree, neither is trusted.

  ⚠️ **What made this catchable was a second measurement that had to agree and did not** — the same
  discipline that caught the `local_schur_block` reversal. A verdict from one power iteration, printed
  with a confident English gloss, would have been recorded as fact; it was already written into this
  file once as "relaxation can stabilize it" before the arithmetic was checked.

  None of this affects the six-component result, which needs no relaxation: the Schur block converges
  undamped at `rho = 0.31` with a governing eigenvalue of `+0.69`.

  **BOUNDARY PROXIMITY: the hardest cells ARE boundary cells — 12 of the 12 worst own a boundary
  face, against 23.1 % of the mesh doing so (2026-08-22).** Under the six-component Hessian nothing
  diverges, so the enrichment table over the diverging *set* is vacuous (0 of 0) and says nothing;
  what carries the finding is the residual ranking. Every one of the twelve worst-reconstructed cells
  sits at boundary hop **0**, where 2.8 of 12 would be the chance expectation — and every one is also
  four-faced. This scheme closes the Hessian at a boundary by taking the owner's, which Betchen &
  Straatman note can leave the system under-determined; their remedy — an inverse-distance average of
  the Hessian over interior neighbours not themselves adjacent to a boundary — is **not implemented
  here**, and this is the first evidence on a real mesh that it is the right place to look next.

  ⚠️ **Association, not mechanism.** A four-faced cell at a boundary is unusual in two ways at once
  and this cannot separate them: the boundary closure could be the cause, or low face count could be,
  or the two could simply co-occur because that is what `snappyHexMesh` produces where it cuts the
  geometry. The discriminating arm is to implement the averaged closure and re-measure the same
  ranking; a face-count-matched comparison of interior against boundary tetrahedra would separate
  them without writing any new closure, and is cheaper.

  ⚠️ **And the collapse the earlier regime showed is gone too**: `sigma_min` shrink now has a
  *minimum* of 5.4e-01 across the whole mesh (median 1.000, **zero** cells below 1e-1), against a
  minimum of 7.7e-03 and eight cells below 1e-1 under the unsymmetrized Hessian. The reduction did
  not merely re-rank the arms — it stopped the correction from nearly cancelling the block anywhere.

  **⚠️ IT IS THE SYMMETRY THAT FIXES THE REACTOR MESH, NOT THE LEAST-SQUARES WEIGHTING — measured
  2026-08-23, and it retires the SPD mechanism this file offered a day earlier.** The six-component
  change carried two things at once: solving the symmetric components, and removing the surplus
  equations by a weighting that makes the reduced block SPD. Separated by swapping the reduction for
  an unweighted projection (`contract_symmetric(residual)`, no `A_P` contraction, no row scaling) and
  re-running the same harness on the same 1.6M-cell mesh:

  | reduction | Schur median | p99 | max | Schur diverging | `A_gg` max | `A_gg` diverging |
  |---|---|---|---|---|---|---|
  | weighted (least squares) | 8.055e-08 | 4.320e-05 | 5.184e-03 | **0** | 5.787e+05 | — |
  | unweighted (projection) | 8.139e-08 | 4.467e-05 | 6.137e-03 | **0** | 7.095e+16 | 6680 |

  **Both fix it completely.** The unweighted block carries no definiteness guarantee whatever, so
  "the reduced block is SPD, therefore the elimination term is well-signed" — offered here as the
  mechanism — **is not what is doing the work.** What matters is that the Hessian is symmetric. Say
  that, and stop citing definiteness as the cause.

  **The weighting is kept anyway, and the reason is not the reactor.** It is the source's own
  formulation, it costs one contraction per apply, and it makes the block SPD — a property worth
  having even where it is not load-bearing. But its cost is real and shows up elsewhere: on a
  tetrahedral mesh under the owner boundary closure it squares an already-bad conditioning,
  `cond(S)` 4.184e+04 unweighted against **4.040e+19** weighted. That difference is moot once the
  averaged closure is in force (which fixes tetrahedra outright, `cond(S) ~ 6`), but it is the
  measurement to remember if a mesh ever turns up where the closure alone is not enough.

  ⚠️ **Note what the `A_gg` column also shows:** unweighted, that arm's worst cell is 7.095e+16
  against the weighted arm's 5.787e+05, and 6680 cells diverge. So the two reductions are not
  interchangeable for the arm that fails — they are interchangeable only for the one that works.


## Responsibility → Hessian boundary closures

    Measured:

    | mesh | calibrated weight | inner sweeps |
    |---|---|---|
    | tetrahedral n=2 | 0.10 | 38 |
    | tetrahedral n=3 | 0.20 | 36 |
    | hex 5³, perturb 0.25 | **0.00** | 10 |
    | 2D, perturb 0.3 | **0.00** | 9 |

    Costs `iters` applies per candidate — about one reconstruction for the whole ladder, once, off
    the differentiated path.

  - **✅ THE WAY TO IMPROVE THE INNER SOLVE IS NOT TO NEST IT — `CoupledBlockSweep` is worth ~12x
    here, far more than the 1.2--1.5x it was worth before this closure existed (2026-08-23).** The
    nested path re-converges the Hessian from zero on **every outer apply**, and this closure is
    precisely what makes that convergence expensive — so the arrangement that keeps the Hessian
    iterate between sweeps gains exactly where the closure costs. At matched accuracy (~1e-10),
    counting face-kernel passes as `outer x (1 + inner)` against the coupled sweep's `2 x sweeps`
    (it was `4 x sweeps` when this table was taken, and the counts below are the corrected ones):

    | mesh | nested | coupled sweep | ratio |
    |---|---|---|---|
    | tetrahedral n=3, weight 0.2 | 1665 passes → 6.2e-11 | **~80 → 2.2e-10** | **~21x** |
    | hex 5³, perturb 0.25 | 165 passes → 1.9e-11 | **34 → 8.9e-13** | **~4.9x** |

    On the tetrahedral mesh the coupled sweep reaches **5.8e-15** at 192 passes, which the nested path
    does not approach at nine times the work. **So the pairing to recommend with this closure is the
    coupled sweep, not a bigger inner count.**

  - **⚠️ AND THE INNER COUNT THE CALIBRATION ASKS FOR IS LARGELY WASTED ON A HARD MESH.** Holding the
    outer count at 20 and sweeping the inner on the tetrahedral mesh, the reconstruction plateaus by
    **inner 8** (4.06e-06) and 36 buys nothing (3.48e-06) — the *outer* count is what limits the
    answer there, so 28 of the 36 calibrated inner sweeps are spent for a 13 % improvement. On the
    hexahedral mesh the same sweep keeps paying to inner 14. **The composition is the thing to
    calibrate, and calibrating each system separately to the same tolerance does not do it** — this
    file's older "three orders of margin" note is the same observation, and the margin is now measured
    at ~7e-4 attenuation on tetrahedra against ~1e-2 on hexahedra. ⚠️ **A blanket loosening factor is
    therefore NOT safe**: those differ by 14x, and the conservative end would over-resolve one mesh
    while under-resolving the other.

  - **⚠️ `CoupledBlockSweep.calibrated` SATURATES ITS CAP on the tetrahedral mesh, and over-counts by
    ~3x when it does not.** It reports 64 at the default cap, 128 at 128, and asks for **193** at 256
    — while 64 sweeps already reconstruct to **5.8e-15**. That is the conservatism this file already
    records structurally (the rate is measured on the packed `[g, h]` error, whose Hessian half
    dominates while the gradient's falls faster), quantified here on a mesh hard enough for it to
    matter. **A reported count equal to the cap is not a measurement** — raise the cap to find out
    what it wanted, and judge the sweep on the reconstruction rather than on the count.


## Responsibility → the all-tetrahedral `A_HH` singularity and the reactor divergence diagnosis

  **⚠️⚠️ THE SCHEME DOES NOT SOLVE ON AN ALL-TETRAHEDRAL MESH UNDER THE DEFAULT BOUNDARY CLOSURE —
  `A_HH` IS NUMERICALLY SINGULAR THERE, AND THIS PREDATES THE SIX-COMPONENT CHANGE (measured
  2026-08-22). ✅ SOLVED 2026-08-23 by `AveragedInteriorHessian` (above), which takes `cond(A_HH)`
  from ~1e17 to ~8 — this entry is the diagnosis that led there, and the "not per-cell / not only the
  boundary" reasoning below is what pointed at the closure rather than at a guard.** `tests/support/meshes.py::
  tetrahedral_grid_3d` builds a conforming Kuhn subdivision of the unit cube (six tetrahedra per
  cube, every cell four-faced, closure ~1e-17, positive volumes, validated by
  `test_the_tetrahedral_fixture_is_a_valid_mesh_of_four_faced_cells`). On it, materialized densely:

  | mesh | formulation | `cond(A_HH)` | `cond(S)` |
  |---|---|---|---|
  | tet, 48 cells, perturb 0.1 | nine components | **4.696e+17** | 7.175e+02 |
  | tet, 48 cells, perturb 0.1 | six components | **2.851e+17** | **4.040e+19** |
  | hex, 3³, perturb 0.25 | nine components | 4.622e+00 | 1.995e+00 |
  | hex, 3³, perturb 0.25 | six components | 2.084e+01 | 1.995e+00 |

  Double precision carries ~1e16, so `A_HH` is **singular in both formulations** — the nine-component
  column is the pre-existing state and was checked by running the diagnostic against the parent
  commit's `gradient.py`, precisely so this could not be mis-attributed to the same day's change. A
  Krylov solve raises (lineax stagnation) and a swept solve returns `inf`, at every perturbation from
  0 to 0.25 and at every size from 48 to 750 cells.

  **What is NOT yet established, and the candidates are not equivalent:**
  - **It is NOT per-cell.** Every cell's diagonal block inverts; the null space is a property of the
    assembled operator. So a per-cell guard is the wrong shape here, unlike the `A_gg` divergence
    above.
  - **It is not only the boundary closure**, though that is the prime suspect: Betchen & Straatman
    introduce their averaged Hessian (Eq. 29) *explicitly* to stop the simple closure leaving the
    system under-determined, and this scheme implements the simple one. But raising the mesh from 75 %
    to 36 % boundary-touching cells does not rescue it, so if the closure is the cause its effect is
    not confined to the cells that touch a boundary.
  - **The six-component `cond(S)` is nine orders worse than the nine-component one** (4.0e+19 against
    7.2e+02) even though `A_HH` is comparably singular in both. The reading that fits: with the full
    tensor the null direction lies where `A_gH` cannot see it — the gradient equation contracts `H`
    against the face-curvature tensor, which is **symmetric**, so an antisymmetric null direction is
    annihilated and never reaches the Schur complement. Removing the antisymmetric components removes
    that shelter. ⚠️ Offered as the reading that fits, **not measured** — the null vector was not
    extracted.
  - **The least-squares weighting squares the conditioning** (`cond(NᵀN) = cond(N)²`), which is
    harmless where `C` is well conditioned (2.4 median on the reactor mesh) and is not where it is
    not. An unweighted projection would avoid the squaring at the cost of the SPD property that the
    reactor measurement above turns on. **Whether the reactor result needs the weighting or only the
    symmetry is UNSEPARATED** — both arrived in one change, and separating them is one more run.

  ⚠️ **THE FIXTURE GAP THAT LET THIS SHIP TWICE IS ONLY HALF CLOSED.** Both the original wrong
  default and this reversal were invisible to every synthetic mesh in the test suite, because those
  are perturbed hexahedral grids and every diverging cell here has **four** faces. A tetrahedral
  fixture now exists (`tetrahedral_grid_3d`) and is validated as a mesh — but the scheme does not
  *solve* on it (see the entry above), so it cannot yet carry a reconstruction test. **Until it can,
  a change to this preconditioner is not tested by a green fast gate — it is tested by this one case
  and nothing else.**

  **⚠️ THE RATE IS THE MECHANISM: `rho(I - P⁻¹S)` is 6.2679 under the Schur block and 0.7148 under
  `A_gg`'s.** The sweep is expansive by 6.3 per iteration, so twenty sweeps is `6.27²⁰ ~ 1e+16` --
  which is the observed 1e+16--1e+18 exactly. Half-budget estimates 5.42 and 0.58, so both are settled
  rather than transient.

  **⚠️⚠️ BUT `rho` IS A WHOLE-MESH NUMBER AND DOES NOT ESTABLISH A WHOLE-MESH CAUSE — an earlier
  version of this entry read it as "GLOBAL rather than per-cell" and concluded "NO PER-CELL FIX
  EXISTS". Neither follows.** A spectral radius above one is equally consistent with a mode spread
  over the mesh and with one pinned to a handful of rows, and nothing measured distinguishes them: a
  fixed sweep count propagates a bad row one cell per sweep, so **8 collapsed cells dilated 20 hops is
  a population of the same order as the 4599 that diverge**. The discriminators are the eigenvector's
  participation ratio, whether the diverging set is the dilation of the collapsed one, and the SIGN of
  the dominant eigenvalue of `P⁻¹S`; all three are now in
  `validation/uvreactor_openfoam/schur_block_diagnosis.py` and **none has been run** — the 1.6M-cell
  mesh is gitignored, was not kept, and the run-file did not record `UV_MESH`, so re-running needs
  `of_case/Allmesh` first.

  **⚠️ AND THE HARNESS THAT PRODUCED `rho` WAS DELETED IN THE SAME COMMIT THAT CITED IT.** The
  `sigma_min` and contraction-rate sections ran at 21:50 and the file committed at 22:05 did not
  contain them, so the headline finding was citable but not re-runnable — the precise state the
  measurement rule exists to prevent. Rebuilt 2026-08-22.

  **THE LEADING EXPLANATION IS THAT WE RUN THE ONE RELAXATION THE METHOD'S AUTHORS EXCLUDE.** Betchen
  and Straatman solve this reconstruction by block-Jacobi over the **coupled** `[g, h]` system, with
  the full per-cell block `A_P` (their Eq. 20) — and the gradient corner of `A_P⁻¹` **is** the local
  Schur complement `(A_gg,loc - A_gH,loc A_HH,loc⁻¹ A_Hg,loc)⁻¹`, by the standard block-inverse
  identity. So `local_schur_block=True` is their preconditioner and `A_gg`'s block is not: the latter
  is the `(1,1)` block of `A_P` itself rather than the inverse of the `(1,1)` block of `A_P⁻¹`, and
  the two differ by exactly the elimination term. Of their own iteration they state that **on an
  arbitrary grid a relaxation strictly below one is required for convergence at all**, and they run it
  at 0.8 (their `a = 0.2`; see the convention note in `SweptGradientSolve.solve`), converging in 33
  iterations. **Our outer solver runs undamped.** So the block that diverges is theirs run at a
  relaxation they exclude, and the block that survives is a weaker, more diagonally dominant stand-in
  that happens to contract undamped.

  **A SECOND DEVIATION WAS A CANDIDATE FOR THE SAME FAILURE — SINCE CLOSED (2026-08-22).** They solve
  the **six independent** Hessian components, reducing the over-determined nine-equation system by
  least squares (their Eq. 16--17), which makes their Hessian block `(A_P E)ᵀ(A_P E)` **symmetric
  positive definite by construction**; this scheme solved all `dim²` components unsymmetrized, so its
  block was neither. A non-definite `A_HH` can give the elimination term the wrong sign, and a
  negative eigenvalue of `P⁻¹S` is precisely the case **no** positive relaxation repairs
  (`1 - w·lambda > 1` for every `w > 0`), so the two candidate fixes are not interchangeable and the
  **sign of the dominant eigenvalue of `P⁻¹S`** is what separates them: positive says Betchen's
  relaxation is the fix, negative implicates the Hessian's representation.

  The representation is now Betchen's — see the six-components entry above — so that arm is settled
  and the block is SPD. What that change buys **on this failure** is a measurement, not an inference,
  and it is taken with the same harness on the same mesh.

  **⚠️ BETCHEN'S OWN "UNDER-DETERMINED" REMARK IS ABOUT NEITHER OF THESE — it is a BOUNDARY-closure
  degeneracy** (their Eq. 28--29): taking the boundary Hessian as the owner's leaves the system
  under-determined in some cases, their illustration being a grid one cell thick in a direction, where
  the second derivative across it is arbitrary. Their fix is an inverse-distance-averaged Hessian over
  interior neighbours not themselves adjacent to a boundary, which this scheme does not implement (see
  the boundary-treatment note above). It is **not** a statement that the per-cell Hessian block goes
  rank-deficient on low-face-count cells, and the measurement agrees: `cond C` is 1.17 mesh-wide
  median and, on four-faced cells, 2.45 median with a **maximum of 3.37** — the best-conditioned group
  on the mesh. **Whether the failing cells touch a boundary is unchecked**, and is the one route by
  which their remark could still bear on this.

  A limiter on the correction remains refuted as a *general* fix -- the `sigma_min` collapse it would
  target affects **8 cells of 1 635 909** (median shrink 1.000, only 3 below 1e-2) — but "8 cells
  cannot matter" is exactly the inference the dilation test exists to check, so do not treat it as
  closed either. A Krylov outer solve does not require `rho < 1`, which is the whole reason it is
  untroubled.

  The block that diverges is **13× closer to the truth**, and cell 60355 is **well conditioned** as
  well as accurate. So neither accuracy nor conditioning is the mechanism: **a stationary iteration
  does not want an accurate preconditioner, it wants a diagonally dominant one.** The correction
  *subtracts* from `A_gg`'s block, the diagonal shrinks, `P⁻¹` grows, and on a cell whose off-diagonal
  coupling is already comparable the iteration stops contracting. ⚠️ **This reading was marked
  "superseded by the global rate" and that marking was WRONG — the rate is this mechanism measured,
  not a competitor to it.** Loss of block diagonal dominance is *how* an iteration comes to have a
  rate above one, so the two are one finding at two levels of description; and it is corroborated
  independently by Betchen and Straatman requiring a relaxation below one for their own block-Jacobi
  on an arbitrary grid, which is what one does to an iteration not dominant enough to contract
  undamped. Three earlier mechanisms
  (near-singular `A_HH` from an under-determined Hessian on tetrahedra — refuted, `cond C` is 1.8 on
  the worst cells and 3.37 at worst over every four-faced cell; preconditioner/operator mismatch at
  low inner counts — refuted, invariant across inner 4–20; ill-conditioning generally — refuted by
  cell 60355) were each proposed and each rejected by measurement.

  **A Krylov outer solve is untroubled by the same preconditioner** (max 2.378e-02), spending
  iterations rather than diverging, because it does not require `rho < 1`. ⚠️ That was the *workaround*
  while the Hessian was unsymmetrized; under six components the fixed sweep converges too
  (`rho = 0.31`), so it is no longer a reason to reach for `GmresGradientSolve` here — and the fixed
  sweep remains far cheaper to differentiate.

  **⚠️ HOW THIS SHIPPED, because the process failure matters more than the bug.** It was defaulted on
  the strength of measurements taken **entirely on 8³ synthetic meshes**, which cannot contain the
  cells that break it — while "measure `local_schur_block` at 1.6M cells" was already an open item.
  A default was changed on evidence that structurally could not see the failure mode. The same run
  also exposed a probe-scheduling bug (below) whose justification was measured at 13824 cells, where
  its premise does not hold either. **Two independent faults in one feature, both from measuring on
  meshes too small to discriminate.**


## Responsibility → `local_schur_block`: probe count and memory (measured at nine Hessian components)

  - **`A_gH` needs one probe per Hessian unknown** — no Kronecker shortcut, because the gradient
    equation contracts **both** of the Hessian's indices (face curvature, each side's Hessian moment,
    the warp moment). Against the full tensor that was a rank-three block at `dim²` probes; against the
    six independent components it is a plain matrix at `n_sym` probes. ⚠️ The count below and the
    memory figures with it were measured at `dim² = 9`.
  - **Probe sequentially, not under `vmap`.** Same wall clock (248 ms against 247 at 13824 cells), but
    `vmap` holds all of the probes' face intermediates at once and those are the largest arrays in the
    scheme: peak 1.29 GB sequential against 1.39 GB vmapped, on a 1.10 GB baseline — a third of the
    feature's memory cost, for nothing.


## Responsibility → `CoupledBlockSweep` as the default

  | fixture | nested 20/10 | coupled 20 |
  |---|---|---|
  | 2D perturb 0.2 | 5.20e-15 | 5.15e-15 |
  | 2D perturb 0.3 | 2.42e-13 | 2.82e-13 |
  | 3D perturb 0.25 | 2.98e-15 | 2.43e-15 |

  ⚠️ **It is NOT uniformly better and the record should not be read as saying so** — it is 0.6x on an
  orthogonal mesh (slower, the nested outer solve being nearly trivial there), 1.4x at 30 % skew,
  1.8x at 40 %, and 3--12x with `AveragedNeighbourHessian`. The case for it is that it is better on
  the meshes this scheme exists for and a wash on the ones it does not, at consistently better
  accuracy.


## Responsibility → where the time goes in a reconstruction (2026-08-23)

  **⚠️⚠️ WHERE THE TIME ACTUALLY GOES IN A RECONSTRUCTION — profiled 2026-08-23, and it redirects the
  optimization effort this scheme had been attracting.** Configuration, in full: `pitzDaily`
  (12225 cells, `dim` 2, `n_sym` 3), `CoupledBlockSweep(sweeps=20)`, **`AveragedNeighbourHessian`**
  at its default weight — ⚠️ **NOT the shipped closure, which is `OwnerHessian`**; see the rate
  caveat below before carrying the sweeps ladder anywhere — `local_schur_block=True` (the shipped
  default), jitted and warmed, minimum of four to six runs on an otherwise idle machine. Harness:
  `validation/gradient_reconstruction_profile.py` — kept in the repository, so this can be re-asked
  when a default moves.

  | piece | time | share |
  |---|---|---|
  | **whole reconstruction** | **26.9 ms** | 100 % |
  | 20 sweeps (from differencing 40 against 20) | **22.0 ms** | **~82 %** |
  | geometry-only prologue | ~4.9 ms | ~18 % |
  | — of which `local_schur_block` | **3.1 ms** | **~11.5 %** |
  | one sweep | 1.10 ms | 4.1 % |

  Three consequences, and the first is the one that matters:

  - **THE SWEEP COUNT IS THE COST, NOT THE WORK INSIDE A SWEEP.** Merging each row into a single
    face-kernel evaluation halved the passes per sweep (four → two) and moved the whole
    reconstruction **9 %**. The reason is that the wasted arguments were **literal zeros**, so XLA
    was already folding most of that work away — a redundancy that is obvious in the source and
    largely absent from the compiled program. Against that, the *count* is worth several times as
    much: the sweep contracts at **0.3152** on this mesh, so

    | sweeps | time | vs 20 | ‖g − g₂₀‖/‖g₂₀‖ |
    |---|---|---|---|
    | 20 (shipped) | 26.9 ms | 1.00x | — |
    | 12 | 18.4 ms | **0.68x** | **1.06e-09** |
    | 8 | 13.9 ms | **0.52x** | 4.58e-07 |
    | 6 | 11.8 ms | 0.44x | 1.07e-05 |
    | 4 | 9.8 ms | 0.36x | 2.99e-04 |

    **Twelve sweeps reproduce twenty to 1e-09 for a third less time.** The class default of 20 is
    not wrong — it is the count that preserves exactness on quadratic fields — it is simply far
    tighter than a flow solve stopping at ‖R‖ ~ 1e-6 has any use for. ⚠️ Do **not** read this as
    licence to lower the class default; a case wanting less should call
    `CoupledBlockSweep.calibrated(mesh, geometry, tol=...)`.

    **⚠️ THE RATE — AND SO THE WHOLE LADDER — IS A PROPERTY OF THE BOUNDARY CLOSURE, and the table
    above was taken under the non-default one.** Measured on pitzDaily exactly as
    `CoupledBlockSweep.calibrated` measures it (relaxation 1.0, `outer(SweptGradientSolve(...))`'s
    preconditioner, `iters=24`, `seed=0`):

    | closure | rate | sweeps for 1e-4 | for 1e-10 |
    |---|---|---|---|
    | **`OwnerHessian` (the shipped default)** | **0.2263** | **7** | **16** |
    | `AveragedNeighbourHessian` | 0.3152 | 8 | 20 |

    The averaged closure couples each boundary cell to its neighbours, so it contracts more slowly.
    Note the trap this set: under the averaged closure 1e-10 needs *exactly* 20 sweeps, which matches
    the class default and made "the default is precisely sized for exactness" look like a clean
    finding. Under the closure that actually ships it needs **16**, so the default carries margin.
    `HessianCorrectedGradient.calibrated` returns 7 / 10 / 13 / 16 at 1e-4 / 1e-6 / 1e-8 / 1e-10,
    which is the `OwnerHessian` row — that disagreement with a hand-computed count is what exposed
    the closure mismatch, and is why `validation/gradient_reconstruction_profile.py` now measures the
    rate the way the calibrator does and defaults `PROFILE_CLOSURE` to the shipped one.
  - **The geometry-only prologue is ~18 % of ONE reconstruction — but a residual reconstructs several
    fields on one geometry, and XLA ALREADY SHARES IT ACROSS THEM.** ⚠️ An earlier version of this
    entry said the prologue is "rebuilt on every reconstruction" and sized a geometry cache from that;
    it is wrong, and it is the *same* mistake as reading the face-pass count as cost — the redundancy
    is visible in the source and absent from the compiled program. Measured by reconstructing `N`
    fields on one mesh inside one jit and fitting `t(N) = P + N·S` (pitzDaily, shipped defaults):

    | sweeps | `N`=1 | `N`=2 | `N`=4 | `N`=6 | marginal per extra field | intercept `P` |
    |---|---|---|---|---|---|---|
    | 20 | 27.9 ms | 49.0 ms | 91.5 ms | 133.4 ms | **21.1 ms** (≈ the 20 sweeps) | **6.8 ms** |
    | 2 | 6.6 ms | 7.6 ms | 8.4 ms | 9.6 ms | **0.60 ms** | ~6.0 ms |

    Six fields cost 9.6 ms against one field's 6.6 ms at two sweeps — common-subexpression
    elimination collapses the six identical prologues to one. **So the prologue is ~5 % of a
    six-field residual, not 18 %**, and a `bind`-style cache can only collect it *across separate
    residual evaluations* (successive Krylov matvecs), never within one.
  - **`local_schur_block` is 11.5 % of a single reconstruction** (≈3 % of a six-field residual) and is
    load-bearing on skewed meshes — dropping it raises the sweep's contraction rate on the reactor
    mesh — so this is not an argument for turning it off. It is also **the one piece that would cost
    no extra memory to cache**: it modifies the outer block in place before inversion rather than
    producing an array of its own.
  - **⚠️ WEIGH ANY CACHE AGAINST ITS RESIDENCY, WHICH AT REACTOR SCALE IS THE BINDING CONSTRAINT.**
    The three cacheable arrays are `hessian_cell_block` `(n, dim, dim)`, the inner preconditioner's
    inverse `(n, n_sym, n_sym)` and the outer's `(n, dim, dim)`: 1.6 MB total at pitzDaily, 9.5 MB at
    `bfs3d`, and **659 MB at a 1.6M-cell 3D mesh, 440 MB of it the 6×6 inner inverse**. Peak memory
    during a reconstruction is *unchanged* — all three are live through the sweep either way — so a
    cache buys ~5 % of reconstruction time in exchange for making 659 MB permanently resident rather
    than transient. On the memory-bound target this scheme exists for, that is the wrong trade;
    calibrating the sweep count is worth six to ten times as much and costs no memory at all.

  **✅ THE SWEEP COUNT IS THE LEVER, CONFIRMED ON A FULL MARCH — 2.61x, BIT-IDENTICAL (2026-08-23).**
  `pitzDaily` gradient A/B, both arms calibrated to an L2 tolerance of 1e-4 by their own factories
  (`PITZ_AB_CALIBRATE=1e-4`), against the same case at each scheme's shipped defaults:

  | arm | sweeps | wall | cycles | steps | `x_r/h` |
  |---|---|---|---|---|---|
  | corrected Green--Gauss, default | 5 | 736.2 s | 439 | 71 | 8.069 |
  | corrected Green--Gauss, calibrated | **2** | **605.1 s** | 431 | 71 | **8.069** |
  | Hessian-corrected, default | 20 | 6690.0 s | 510 | 72 | 8.069 |
  | Hessian-corrected, calibrated | **7** | **2565.3 s** | **510** | **72** | **8.069** |

  **Read the cycle and step columns: 510 and 72 in BOTH Hessian-corrected arms.** Seven sweeps
  produce Newton directions the coupled solver cannot distinguish from twenty — not merely a similar
  trajectory but the same one — and the scheme-versus-scheme field differences are unchanged to four
  significant figures (`U` 6.665e-03 in both runs). So this is a 2.61x saving for no change in the
  answer whatsoever, and the identical trajectory is evidence of headroom *below* seven rather than
  at it. The matched-tolerance cost ratio between the schemes is **4.24x**, against 9.09x at defaults
  — most of that gap was the two schemes' differing slack, which is the reason this case has a
  calibrated mode at all. ⚠️ The 2565 s arm also carries the two row merges, worth ~9 % of a
  reconstruction, so the calibration alone is nearer 2.4x than 2.61x.

  **⚠️⚠️ AND SPENDING THE SWEEP LEVER PROMOTES THE PROLOGUE, WHICH INVERTS THE CACHING VERDICT
  ABOVE.** The prologue is geometry-only, so it is near-constant while the sweeps shrink around it.
  Measured by the same `t(N) = P + N·S` fit, shipped closure:

  | sweeps | shared prologue `P` | marginal per field `S` | `P` share of a six-field residual |
  |---|---|---|---|
  | 20 | 7.42 ms | 17.61 ms | **6.6 %** |
  | **7 (calibrated)** | 6.82 ms | 5.64 ms | **16.8 %** |

  At one reconstruction it is starker still: **48 % at seven sweeps against 18 % at twenty**, of which
  `local_schur_block` alone is ~3.5 ms. So the "~5 % for 659 MB, do not cache" reading recorded above
  was taken at the *un*-calibrated operating point and does not survive calibration. At the calibrated
  point the best-ratio cache is the **outer preconditioner's inverse alone**: it carries
  `local_schur_block`, costs ~4.2 ms of the 6.82 ms prologue, and is `(n, dim, dim)` — **110 MB at
  1.6M cells against 659 MB for the whole prologue**, whose bulk is the `(n, n_sym, n_sym)` inner
  inverse that this scheme now rebuilds by algebra anyway. Expect the share to be **higher again in
  3D**, where `local_schur_block` probes `dim + n_sym = 9` columns rather than 5.

  **The general lesson, and it is the one to carry: optimize in the order that keeps the ranking
  honest.** Every share here is a fraction of a total that the previous lever changed, so a
  cost-share table is only valid at the configuration it was taken at — and the largest item at the
  default configuration was not the largest item after the largest lever was spent.


## Responsibility → `GradientScheme.bind`

  **⚠️ MEASURE THE SAVING AT THE FIELD COUNT A RESIDUAL ACTUALLY USES, NOT AT ONE FIELD.** The
  prologue is built **once** per compiled residual (the compiler shares it across fields), so binding
  removes a fixed amount of work whatever the field count — which reads as a huge saving on a
  one-field probe and a modest one on the six-field residual that is the real consumer. Measured at a
  calibrated seven sweeps, arms interleaved so machine drift hits both equally, minimum of seven to
  nine runs:

  | mesh | `N`=1 | `N`=6 | absolute saving |
  |---|---|---|---|
  | pitzDaily, 2D, 12225 cells | 38.5 % | **7.8 %** | 4.3 ms → 2.9 ms |
  | perturbed hex, 3D, 21952 cells | 37.2 % | **13.0 %** | **26.1 ms → 26.2 ms** |

  Read the last column: in **3D the absolute saving is unchanged between one field and six**
  (26.1 against 26.2 ms), which is what "built once per residual, removed entirely" predicts. In 2D
  it falls (4.3 → 2.9 ms), so some of the prologue there is being hidden behind the independent
  per-field sweeps rather than costing wall clock. **3D is the case that matters and it is the
  better one**, ~1.7x the 2D share, as `local_schur_block` probing `dim + n_sym` = 9 columns against
  5 predicts.

  ⚠️ **A ~17 % projection from the serial cost share was too high** — the honest figure is 13 % in
  3D and 8 % in 2D, because a cost share assumes the work is on the critical path and some of it is
  not. Project from a share only as an upper bound.

  **Memory, which is the whole reason this is the outer inverse and not the prologue:** `(n, dim, dim)`
  is **110 MB at 1.6M cells**, against 659 MB for all three geometry arrays — the bulk of which is the
  `(n, n_sym, n_sym)` inner inverse this scheme now rebuilds by algebra anyway. Peak memory during a
  reconstruction is unchanged either way; binding converts a transient allocation into a resident one.


## Responsibility → the UV reactor, and directions closed for the coupled sweep (2026-08-23)

  **⚠️⚠️ MEASURED ON THE UV REACTOR (2026-08-23) — the mesh this scheme exists for, and the numbers
  do NOT resemble pitzDaily's.** 1 635 909 cells, 5 249 365 faces, `dim` 3, snappyHexMesh; interior
  face skewness median 0.0022 / p99 0.2013 / max 0.3505, face planarity min 0.877 (pitzDaily is
  planar to 1.000000 and p99-skewed 0.0158). Contraction rate of the coupled sweep, measured as
  `CoupledBlockSweep.calibrated` measures it, `iters=24`, peak RSS 9.3–10.8 GB.
  Harness: `validation/uvreactor_openfoam/gradient_sweep_calibration.py --rate` (or `UV_RATE=1`).

  | closure | `local_schur_block` | ω | rate | sweeps @1e-4 | @1e-6 |
  |---|---|---|---|---|---|
  | `OwnerHessian` (shipped) | **True** | **1.00** | **0.5378** | **15** | 23 |
  | `OwnerHessian` | True | 1.05 | 0.5585 | 16 | 24 |
  | `OwnerHessian` | True | 1.10 | 0.5894 | 18 | 27 |
  | `OwnerHessian` | **False** | 1.00 | **2.0145** | **DIVERGES** | — |
  | **`AveragedNeighbourHessian`** | **True** | **1.00** | **0.4385** | **12** | 17 |
  | `AveragedNeighbourHessian` | True | 1.10 | 0.4913 | 13 | 20 |
  | `AveragedNeighbourHessian` | **False** | 1.00 | **4.9196** | **DIVERGES** | — |

  Four things, and the first two are binding.

  - **`local_schur_block=False` DIVERGES on this mesh** — 2.01 and 4.92, not merely slower. So it is
    not an optimization to be traded off here; it is required. This also sets the true price of the
    calibrator defect fixed the same day: measuring the `False` preconditioner would have returned a
    rate above one, which `sweeps_for` clamps to `cap` — **64 sweeps where 15 are needed**, on the
    target mesh, silently.
  - **OVER-RELAXATION IS REFUTED HERE, AND THE OPTIMUM IS AT OR JUST BELOW 1.0 — closure-dependent.**
    ⚠️ An earlier version of this entry said "do not ship a non-unit relaxation", which is broader
    than the data: it was written from the over-relaxed half of the ladder before the under-relaxed
    half had run. The full sweep (`local_schur_block=True`):

    | ω | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 | 1.05 | 1.10 |
    |---|---|---|---|---|---|---|---|
    | `OwnerHessian` | 0.7078 | 0.6203 | 0.5347 | **0.4946** | 0.5378 | 0.5585 | 0.5894 |
    | `AveragedNeighbourHessian` | 0.7101 | 0.6238 | 0.5419 | 0.4666 | **0.4385** | 0.4715 | 0.4913 |

    The owner closure has a genuine interior optimum at **0.90** (14 sweeps against 15); the averaged
    closure — the one to prefer on this mesh — is optimal at **1.00**, with both sides worse. So the
    correct statement is that *over*-relaxation is refuted and the optimum sits at or just below
    unity, not that unity is always right. Note the consequence for polynomial acceleration: under
    the preferred closure the degree-one polynomial is **already at its optimum**, so any gain from
    that family has to come from degree two or higher. A relaxation tuned on pitzDaily (1.05–1.10,
    worth a sweep there) costs 1–3 sweeps here, which is the transferable warning.
  - **`AveragedNeighbourHessian` is the FASTER closure on this mesh** (0.4385 against 0.5378, 12
    sweeps against 15) where on pitzDaily it is the slower one (0.3038 against 0.1978). It was
    adopted for exactness on cells whose `A_HH` the owner closure leaves singular; that it also
    converges faster here is a second, independent reason to prefer it on skewed meshes — and a
    reminder that closure choice is a per-mesh question, not a global ranking.
  - **The sweep count does not transfer between meshes and the spread is wide**: 6 sweeps on
    pitzDaily, 12–15 here, 23 on synthetic tetrahedra. Any cost projection for this scheme taken from
    pitzDaily understates the reactor by 2–2.5x on the sweeps alone, before 3D's dearer per-sweep
    cost. Calibrate per mesh; the class default of 20 is neither safe (it is below what the tet mesh
    needs) nor economical (it is above what pitzDaily needs).

  **⚠️ REPLACING THE COUPLED RECONSTRUCTION WITH A LOCAL LEAST-SQUARES FIT IS A POOR BET ON A SKEWED
  MESH — and the evidence is in the source paper, so check there before re-proposing it (2026-08-23).**
  The reasoning that leads here is sound and will recur: the sweeps are ~82 % of a reconstruction, the
  count is set by a contraction rate of 0.54 on the reactor, and information propagates one cell-ring
  per sweep — so ~15 rings of propagation are being spent to produce what is intrinsically a local
  quadratic fit. A k-exact least-squares reconstruction gets exactness for quadratics *directly*, at
  roughly one pass, stays exactly linear in the field (so the cheap tangent survives), and is a far
  better shape for a GPU than fifteen sequential global sweeps with reductions.

  **What kills it is the MAXIMUM error in poor cells, which is the regime this scheme exists for.**
  Betchen & Straatman (2010) benchmark against Baserinia & Stubley's **second-order** least-squares
  reconstruction — a quadratic fit, so the comparison is close to this proposal rather than to the
  ordinary linear-LSQ gradient — and report its maximum gradient and Hessian errors as "of the order
  of those observed in the first-order Gauss' theorem approach", i.e. the second-order advantage is
  lost entirely in the worst cells. The worst error sat one layer of tetrahedra off a curved
  boundary, where the local aspect ratios and the Hessian components were both large, on a grid whose
  **mean** aspect ratio was only 1.23 (max 3.44) — a good grid. They attribute it to Mavriplis (1993,
  *Revisiting the least squares procedure for gradient reconstruction on unstructured meshes*), who
  found that even inverse-distance-weighted least squares fails near highly skewed volumes on
  tetrahedral grids. The UV reactor is exactly that regime: skewness p99 0.20, face planarity down to
  0.877, with the worst cells in the boundary layers.

  ⚠️ **One distinction to keep, because it decides what the evidence does and does not cover.**
  Baserinia & Stubley's scheme is **coupled**, not a local fit: the same paper reports it *diverging*
  under block-Jacobi and needing "a stabilized, pre-conditioned conjugate gradient procedure". So the
  head-to-head in that paper is not against a purely local k-exact fit. The Mavriplis result it rests
  on **is** about local weighted least squares, so the max-error objection still reaches the local
  version — but do not cite Betchen's table as a direct measurement of it.

  **THE SURVIVING DIRECTION IS LEAST SQUARES AS A PRECONDITIONER, NOT AS THE ANSWER.** Use a local
  fit to precondition the existing coupled system. The fixed point stays Betchen's, so accuracy and
  poor-cell robustness are unchanged **by construction**; a preconditioner that is bad in a skewed
  cell costs convergence *rate* there and never accuracy, which neutralizes precisely the documented
  failure. It attacks the only quantity that sets the cost — the rate — which neither relaxation
  (refuted on the reactor, both directions) nor Krylov acceleration (forfeits the exact linearity
  that keeps the tangent cheap) can touch. Unbuilt and unmeasured.

  **Two calibration points from the same paper, worth having before judging our own numbers:**
  - **They report 33 block-Jacobi iterations** to converge the reconstruction on their tetrahedral
    grid. Our 12–15 sweeps on the reactor — Gauss–Seidel with a Schur-block preconditioner rather
    than block-Jacobi — is **this method's cost, not an inefficiency in this implementation.**
  - **They exclude building the coefficient matrices from their reported timings**, treating them as
    precomputed per geometry. That is our prologue, and it is independent support for `bind`: the
    same split, and the same judgement about which side of it that work belongs on.

  **❌ SUB-SOLVING EITHER BLOCK OF THE COUPLED SWEEP IS REFUTED (2026-08-23) — the rate is set by the
  INTER-CELL coupling, and no cell-local preconditioner can reach it.** The proposal is a natural one
  and will recur: the gradient block `A_gg` is essentially the corrected Green--Gauss operator, so
  replace the per-cell block-Jacobi inverse in the sweep's gradient update with `k` fixed sweeps of
  that block (a truncated Neumann series -- fixed count, so still exactly linear in the field). The
  same for the Hessian block, and for both. Measured as the coupled sweep's contraction rate, at
  `local_schur_block=True`, `iters=24`:

  | arm | inner=1 | 2 | 3 | 4 |
  |---|---|---|---|---|
  | pitzDaily, gradient block | **0.1978** | 0.2285 | 0.2254 | 0.2256 |
  | pitzDaily, Hessian block | **0.1978** | 0.1980 | 0.1980 | — |
  | pitzDaily, both | **0.1978** | 0.2287 | 0.2255 | — |
  | hex 3D p=0.35, gradient block | **0.2701** | 0.2698 | 0.2685 | 0.2686 |
  | hex 3D p=0.35, Hessian block | 0.2701 | **0.2489** | 0.2484 | — |
  | hex 3D p=0.35, both | 0.2701 | 0.2350 | 0.2349 | — |

  Every arm **saturates by two or three inner sweeps** — so the limit is what an *exact* block inverse
  would give — and none pays for its cost. Counting kernel passes (one Hessian pass plus `k` gradient
  passes per outer sweep), the best arm is 3D Hessian sub-solve at 21 passes against the baseline's
  16. On pitzDaily the gradient arm is **worse than the per-cell block**, which is the recorded
  "inverting the wrong operator more accurately" signature: block Gauss--Seidel's gradient update
  wants the Schur complement, not `A_gg`, and the per-cell local Schur block approximates the former
  while an exact `A_gg` sub-solve converges to the latter.

  **The three results together say where the rate actually lives, which is the durable finding.**
  `local_schur_block` is worth everything on the reactor (2.0145 → 0.4385, divergence to convergence),
  yet sub-solving either block is worth nothing. That is consistent, because the local Schur block
  **is** the exact per-cell Schur complement of the joint `[g, H]` block — so the sweep already
  inverts each cell's coupled block essentially exactly, and the intra-cell coupling is exhausted.
  What remains in a 0.44--0.54 rate is **inter-cell**, which is why a preconditioner that cannot see
  past one cell moves it not at all.

  **CONSEQUENCE FOR WHAT TO TRY NEXT: any real gain needs REACH, not accuracy.** A better cell-local
  inverse is closed. What is not closed is a preconditioner spanning more than one cell — a local
  least-squares operator over a two-ring stencil is the obvious candidate, and its documented weakness
  (large maximum error in skewed cells) is harmless in a preconditioner, where a bad cell costs
  convergence rate and never accuracy. See the least-squares entry above for why that weakness rules
  it out as a *replacement* and not as a preconditioner.

  **❌ POLYNOMIAL ACCELERATION AND EVERY PER-SWEEP RELAXATION SCHEDULE ARE BOUNDED AT ~1 SWEEP —
  measured on the reactor 2026-08-23, and the bound covers more than it looks like.** The proposal
  recurs in several disguises: Chebyshev on the error operator, fourth-kind Chebyshev, "cascading"
  smoothers with an escalating per-sweep ω, or simply a schedule `ω₁…ω_N` instead of one ω. **They are
  all the same object.** Starting from zero, `N` steps of `x ← x + ωₖ P⁻¹(b − A x)` leave the residual
  `Π(I − ωₖ P⁻¹A) r₀` — a degree-`N` polynomial with `q(0) = 1` — and GMRES minimizes over exactly
  that class. So **one GMRES run bounds the entire family**, and a fixed-coefficient member does
  strictly worse because it cannot adapt to the right-hand side.

  Measured on the reactor (`AveragedNeighbourHessian`, `local_schur_block=True`, ω = 1, residuals of
  the same system on the same right-hand side):

  | k | sweep residual | best possible polynomial | headroom |
  |---|---|---|---|
  | 5 | 1.579e+00 | 5.967e-01 | 2.65x |
  | 7 | 1.154e-01 | 6.013e-02 | 1.92x |
  | 10 | 3.051e-03 | 1.451e-03 | **2.10x** |

  The headroom **collapses** from 552x at k=1 to ~2x in the operating range, and 2x in residual at a
  per-step factor of ~0.3 is **one sweep of thirteen**. Harness:
  `validation/uvreactor_openfoam/gradient_sweep_calibration.py --spectrum`.

  ⚠️ **DO NOT COMPARE EITHER AGAINST `rate^k` — that compares an ERROR with a RESIDUAL and is
  meaningless on an operator this nonnormal.** A first version of this measurement did exactly that
  and reported GMRES as *worse* than the sweep, which is impossible for a minimizer over the class the
  sweep belongs to; that impossibility is what caught it. On pitzDaily the sweep's residual **grows to
  ~650 before it contracts at all**, so the asymptotic rate is nowhere near achievable early — which is
  also why rate-based projections of Chebyshev gains (1.5–2x) are far too optimistic here.

  **❌ COARSE-GRID AND MULTIGRID METHODS ARE CLOSED.** An *ideal* two-level cycle — geometric
  aggregation, exact Galerkin coarse operator, exact coarse solve, the existing exact per-cell block as
  smoother — measured **0.1–1.4 %, and 0.0 % at ν = 2**, across meshes bracketing the reactor's rate
  from both sides, on a harness first shown to find the textbook 1.5x on a Poisson control. Two-level
  with an exact coarse solve **bounds every deeper hierarchy**, so this closes deep AMG, W-cycles and
  every coarsening rate at once. The mechanism: this operator is **second kind** — a bounded
  perturbation of the identity, spectrum measured in [0.71, 1.18] — so its rate is set by mesh
  *quality*, not mesh *size* (0.151 at 4³ falling to 0.151 at 10³, while a Poisson control climbs
  0.919 → 0.969). Multigrid removes h-dependence; there is none here to remove.
  ⚠️ **Those two-level numbers were measured on synthetic fixtures with a scratchpad harness and are
  NOT re-adjudicable as recorded** — treat them as strong but unreproducible. What *is* re-adjudicable,
  and corroborates them independently on the real mesh, is the mode locality below: a coarse space
  represents smooth error, and this error is not smooth.

  **✅ THE SLOW MODE IS EXTREMELY LOCALIZED, WHICH IS THE POSITIVE RESULT OF THE CAMPAIGN.** Dominant
  eigenvector of the error operator, same probe:

  | mesh | worst 0.1 % of cells hold | worst 1 % hold | roughness `\|v − avg v\|/\|v\|` |
  |---|---|---|---|
  | pitzDaily | 34.3 % | **95.4 %** | 0.49 |
  | **UV reactor** | **100.0 %** (1000x population share) | 100.0 % | **1.15** |

  On the reactor the entire slow mode lives on ~1600 cells of 1.6M, at cell-scale roughness. That
  closes the coarse-grid family from the other direction and **opens the one candidate the memory
  arithmetic previously killed**: a Schwarz/patch correction costed over *all* cells is 7.4 GB and is
  dominated, but over the worst 0.1 % it is ~1600 × 9 × 63 doubles ≈ **7 MB**, at ~0.1 % of a sweep.
  The selection is legal under the linearity constraint because the eigenvector is a property of the
  **geometry-only** operator, so the patch set is a mesh constant chosen once — not a field-dependent
  branch. ⚠️ Unmeasured and decisive before building: **the sub-dominant rate.** If a cluster of
  similarly-slow modes sits on *different* cells, the patch set grows and the memory argument returns.

  **From the literature, two items that bear directly on this scheme:**
  - **Betchen & Straatman measured the failure mode of a fixed sweep count in the source paper**:
    terminating after four sweeps gave **22 % error in a Hessian component** where 33 were needed for
    1e-4, on a tetrahedral mesh of *mean* aspect ratio 1.23 and max 3.44. Syrakos et al. (2017)
    measure the same effect from the other side — a fixed corrector count buys a pre-asymptotic window
    of apparent higher order that a fine enough mesh destroys. **Validate any reduced count against an
    accuracy fixture, not only against a residual rate.**
  - **Nishikawa (2018) uses a skewness-dependent diagonal boost**, geometry-only, introduced because
    "a serious convergence difficulty may be encountered in the case that cells have all skewed faces".
    That lands exactly on the localized mode measured above, preserves exact linearity, and is cheap.
    Unbuilt.

  **The structural alternative, if the sweep is ever to be removed rather than accelerated:**
  Setzwein, Ess & Gerlinger (JCP 446, 2021) apply the Green–Gauss operator to the 1-exact gradient and
  correct the result with a **per-cell 6×6 matrix depending solely on mesh geometry, inverted before
  the simulation** — an O(h) Hessian and a 2-exact gradient in two face passes plus one precomputed
  matvec, with **no global system, no iteration and no convergence test**. Against 13 sweeps × 2 passes
  that is ~3 passes; memory is `(n, n_sym, n_sym)` = 440 MB at 1.6M cells, and the matrix reduces to
  the identity on Cartesian grids. ⚠️ **Its accuracy claim is derived by Taylor analysis and never
  measured** — the paper carries no error-versus-`h` table for the Hessian, and the authors' own flux
  benchmark found the correction had "only minor effects". Closing that gap is the prerequisite, and
  would itself be a contribution.


## `MultipleCorrectionGradient` — measurements

  **⚠️ ON THE SHIPPED MARCH THE GAIN IS ~2–5 %, NOT THE ~10–14 % THE PER-CALL JVP PROJECTED.**
  `validation/run_case.sh validation/pitzdaily_openfoam/compare.py`, shipped `case.yaml` and driver
  defaults, no `PITZ_*` overrides, one run per arm, back to back on the same 4-core / 15 GB Linux
  container, jax 0.11.2, persistent compile cache warm from earlier in-session runs:

  | arm | solve wall | log `t(s)` at step 1 → 18 | steps / cycles | `x_r/h` | final `R` |
  |---|---|---|---|---|---|
  | `main` `44ba609f` | 430 s | 94 → 416 s (322 s) | 18 / 123 | 8.0686 | 9.574e-06 |
  | this change `5b2c3dd2` | 411 s | 83 → 397 s (314 s) | 18 / 123 | 8.0686 | 9.574e-06 |

  The trajectories are identical row for row (`beta`, inner and cycle counts, `R` to four figures).
  −19 s overall (−4.4 %), of which 11 s is before the step-1 row (compile and the first step) and
  8 s (−2.5 %) is steps 2–18. One run per arm with no measured noise floor, so the march-time part is
  within plausible run-to-run spread. **Why the projection overshot is not established.** One
  candidate, unchecked: if the Krylov operator is a linearization taken once per step, its matvec is
  the tangent of `inv` (two batched matrix products, `−A⁻¹ dA A⁻¹`), with no factorization, so the
  standalone `jax.jvp` timing, which re-runs the primal LU on every call, overstates what each
  matvec saved. Check the compiled step program's matvec before using the per-call figures to
  project a march gain again.

  **✅✅ `MultipleCorrectionGradient` IS FASTER THAN CORRECTED GREEN--GAUSS ON A REAL MARCH
  (pitzDaily, 2026-08-24)** — a quadratic-exact reconstruction for less than the linear-exact one.
  Same case, same field-split preconditioner, same probe reach 5, same continuation, same commit:

  | arm | wall | cycles | steps | `x_r/h` |
  |---|---|---|---|---|
  | `CorrectedGreenGauss` (default sweeps) | 739.5 s | 439 | 71 | 8.069 |
  | **`MultipleCorrectionGradient`** (`OwnerGradient`) | **703.7 s** | **437** | 73 | **8.069** |
  | `HessianCorrectedGradient`, calibrated 7 sweeps | 2565.3 s | 510 | 72 | 8.069 |
  | `HessianCorrectedGradient`, 20 sweeps | 6690.0 s | 510 | 72 | 8.069 |

  **Read the cycle column first: 437 against 439.** The preconditioner sees the two arms identically,
  so the wall clock is the reconstruction and not a preconditioning artifact — the check this case
  exists to make. ⚠️ The two Betchen rows predate `bind` and the row merges and are stale in its
  favour, but not by a factor that reaches this conclusion.

  *Configuration:* pitzDaily 12225 cells, `N_POINTS=2`, `beta_start` 0.5, `CflResidualDualTimeControl`
  defaults, stop `(0.0, 1e-5)`, `simplesmooth` leading inverse, compiled ILU(0) live; `PITZ_GRADIENT` and
  `PITZ_STENCIL_REACH` the only variables.

  | arm | steps | cycles | wall | s/cycle | esc | `x_r/h` | `nut` peak |
  |---|---|---|---|---|---|---|---|
  | `CorrectedGreenGauss` (swept 4), reach 5 | 64 | 421 | **750 s** | 1.78 | 1 | 8.0686 | 417.36 |
  | **`MultipleCorrectionGradient`, reach 3** | 69 | 417 | **506 s** | **1.21** | **0** | 8.0686 | 417.52 |

  **Same root** (`x_r/h` identical to four decimals, `nut` peak within 0.04 %, `ux` 0.0189 against
  0.0191) and **essentially the same work** — 417 cycles against 421 — at **32 % less wall clock**,
  because each cycle meets a Jacobian of roughly half the nonzeros. Per-cycle cost 1.21 s against 1.78 s
  is the whole of it, and the preconditioner refresh halves outright (59 s against 124 s, its coloured
  probe 28 s against 56 s). One escalation becomes none. Well outside this case's ~3 % replicate spread.

  **Measured on pitzDaily, and it is a perturbation of the start, not of the root** (A/B against a
  control worktree at `6df9944`, the branch point; case defaults: `multcorr`, reach 3,
  `CflResidualDualTimeControl`, compiled ILU(0) live). Only pitzDaily's and bfs3d's initial conditions
  move at all — every test that names `CompactGreenGauss` was already initializing with it. Both arms:
  **31 steps, 194 cumulative cycles, identical beta and inner-iteration columns on every step**; cycles
  differ at two steps (step 14: 11 → 12, step 18: 13 → 12, where the control also carried a
  line-search flag the arm did not); residuals agree to 3–4 significant figures throughout; `x_r/h`
  8.0686 in both, `nut` peak 417.9254 → 417.9255, and the generated `report.md` byte-identical.
  ⚠️ **The arm's wall clock (658 s against 306 s) is VOID**: a virtual machine started at 11:10:44,
  between the two runs, and held ~400 % CPU through the whole arm — its slowdown is a uniform ~2.2x
  on every step including step 1, which is contention, not the change (the change touches only the
  initial condition, never the per-step residual or Jacobian). bfs3d was not run.


## `MultipleCorrectionGradient` — the initializer re-bind, and the `SkewCorrectedGradient` stall on pitzDaily (2026-08-24/25)

  **The re-bind an initializer now pays is negligible, measured rather than assumed.** `potential_flow`
  hands its already-bound scheme to `laplace_field`, whose `ResidualAssembler.build` binds again, so
  `hybrid_initialize` rebuilds the correction matrices three times. Measured on perturbed 2D quad
  meshes (jax 0.10.2, CPU, x64): **1.06 s** for the first bind in a process and **0.014 s** for every
  one after it, at both 1600 and 10000 cells — the first is XLA compiling the monomial probe, which
  is cached. No bind-avoidance mechanism was built, and one would need a geometry identity the
  `bind` contract deliberately does not carry.
  **⚠️⚠️ `SkewCorrectedGradient` STALLS THIS CASE DEAD, and the reason is about what a boundary value
  MEANS — not about the closure's accuracy.** It converges the first two Reynolds rungs to the
  standard arm's residual to three significant figures (7.277e-06 against 7.255e-06 at step 28, and a
  step-for-step identical rung-2 transition) and then fails on the **first step of the target rung**:
  α = 0.001 immediately, β escalated to its 16.0 cap, residual frozen at 1.074e-01 for every
  subsequent step. Not a degradation — and **not, as this line read until the cause was found, "no descent direction at all": the direction descends, and it is the finite-step curvature that is inadmissible.** See below.

  **⚠️⚠️ THE CAUSE IS NOW MEASURED, AND IT IS NEITHER OF THE TWO ACCOUNTS PREVIOUSLY RECORDED HERE:
  `SkewCorrectedGradient` CORRECTS TWICE ON EVERY GRADIENT-TYPE PATCH (2026-08-24).** A
  reconstruction is fed **leading-order** boundary values — the closures evaluated at *zero*
  gradient, deliberately, so the residual stays a single-pass function of the field
  (`residual.py::_gradient`). A Dirichlet value does not depend on the gradient and is unaffected. A
  gradient-type one is: `ZeroGradient.face_value` is `phi_owner + tangential_correction(...)`, whose
  correction is evaluated away — so the boundary value handed in is **exactly** `phi_owner`. The
  closure then forms `rise = bval − phi_owner − non_orthogonal_correction(...)` = `−corr`, and
  divides it by `d·n`, the smallest length in the mesh at a wall. **The entire normal derivative it
  reports on such a patch is an artifact of subtracting a correction the boundary value never
  added.**

  Measured on pitzDaily (`validation/multiple_correction/boundary_closure_probe.py`,
  MultipleCorrection + skew, analytical near-wall omega, leading-order boundary values):

  | patch | kind | max \|bval − phi_P\| | med \|d·n\| | med \|rise/d·n\| | max |
  |---|---|---|---|---|---|
  | inlet | Dirichlet | **3.225e+04** | 7.9e-04 | 3.3e+05 | 4.1e+07 |
  | outlet | ZeroGradient | **0.000e+00** | 2.6e-03 | 1.7e+03 | **3.0e+07** |
  | upperWall | ZeroGradient | **0.000e+00** | 1.8e-04 | 1.1e+02 | **1.4e+05** |
  | lowerWall | ZeroGradient | **0.000e+00** | 5.1e-04 | 8.0e-01 | 1.5e+04 |

  **Exactly zero on every gradient-type patch, and only there** — the Dirichlet inlet carries a real
  difference, which is what the correction exists for. `OwnerGradient` never reads a boundary value
  and marches.

  ⚠️ **Two earlier accounts are SUPERSEDED and should not be repeated.** "The wall value is huge and
  gets amplified by `1/(d·n)`" is wrong — the value is the cell's own. "The wall `omega` value is a
  modelling device rather than data, so the closure imposes a spurious near-zero normal derivative"
  is the right *neighbourhood* but the wrong mechanism, and it mis-predicts: it says the problem is
  `omega`-specific and would yield to imposing that field's gradient. It does not (below), because
  the defect is in the closure's arithmetic and applies to every field on every gradient-type patch
  — the outlet included, where `d·n` is not even small and the artifact still reaches `3e7`.

  **⚠️⚠️ MEASURED: THE `ImposedGradient` SEAM DOES NOT RESCUE THIS CLOSURE — two cold marches,
  2026-08-24.** With the analytical wall-`omega` gradient imposed on the wall cells *and* their
  boundary faces, and (second run) also threaded into the omega equation's own assembler so the
  residual and the closure fields impose the same derivative, the march reproduces the stall
  **exactly**: first step of the target rung, α = 0.001, β at its 2.0 start, `|R|` 1.079e-01 and
  1.082e-01 against the pre-fix 1.074e-01. Both runs cleared Re/100 and Re/10 at α = 1.000
  throughout. That is the direct evidence for the paragraph above: fixing `omega` cannot fix a defect
  that is not about `omega`.

  Two sub-mechanisms were tested and **refuted** on the way, both worth not re-proposing:
  - **Not the correction matrices.** On this mesh `cond(M2⁻¹)` is median 2 / max **2.14** under skew
    against median 2 / max **3.49** under owner, and `|gradient_defect|` is identical to three
    figures. Skew is the *better*-conditioned of the two here.
  - **Not "the Hessian correction is negligible so the closure cannot matter".** It is not
    negligible: `|defect·hessian|` reaches **41 %** of `|first|`, and skew's Hessian tail is **2.8×**
    owner's (1.94e12 against 6.96e11). The closure does reach the returned gradient — through the
    Hessian, which is the only route, since the first pass reads no face gradient.

  **✅ THE DOUBLE CORRECTION IS FIXED — `boundary_values_at` — AND ⚠️ IT DOES NOT FIX THE MARCH.** A
  scheme that *differentiates* a boundary value now asks the caller to re-evaluate its closures at
  the reconstruction's own gradient (`GradientScheme.gradients(boundary_values_at=…)`, supplied by
  `ResidualAssembler._gradient` and — since #313 — by `MomentumContinuity`'s velocity and pressure
  reconstructions too); a closure declares whether it needs them
  (`GradientBoundaryClosure.reads_boundary_values`, `False` on `OwnerGradient`, so that path pays
  nothing). Measured on the same patches, with the Dirichlet arm as the control that makes it
  readable — a fix that merely suppressed the term would have flattened that one too:

  | patch | kind | leading order | corrected |
  |---|---|---|---|
  | inlet | Dirichlet | 4.075e+07 | **4.075e+07** |
  | outlet | ZeroGradient | 3.023e+07 | **8.841e-11** |
  | upperWall | ZeroGradient | 1.434e+05 | **7.946e-08** |
  | lowerWall | ZeroGradient | 1.523e+04 | **1.070e-08** |

  **Eighteen orders down on the gradient-type patches, and the march is BIT-FOR-BIT AS STALLED** —
  step 50 of the target rung, α = 0.001, `|R|` 1.082e-01, identical to the two runs before it. So the
  double correction was a genuine defect worth fixing on its own terms, and it is **not** the cause.

  ⚠️ **Three mechanisms were proposed and refuted before the cause was found; do not re-propose
  them.** (i) the wall `omega` gradient, (ii) the correction matrices' conditioning, (iii) the
  closure's double correction. Each was measured, not argued, and each left the stall unchanged to
  three figures. **The reason all three missed is the same and is worth more than any of them: every
  one was measured at the cold initial condition or on a single reconstruction, and the arms are
  IDENTICAL at every state either of those reaches.** The cause is below.

  **✅✅ FOUND (2026-08-24): A LOW-SHIFT WALL AT BOUNDARY CELLS THAT **BOTH** CLOSURES HIT, WITH
  `beta_start` SITTING ON ITS EDGE — and the arms part at the SECOND INNER NEWTON ITERATION of the
  target rung's first step, not at its anchor.**
  `validation/pitzdaily_gradient_ab/closure_stall_probe.py` captures the rung's seed from a real ramp
  (`capture`), drives the real `DualTimeStep` from it with one line per inner iteration (`march`), and
  compares the two closures at any saved state (`analyze`). Configuration for everything below:
  pitzDaily, 12225 cells, `MultipleCorrectionGradient(fallback=None)` so the closure is global,
  `CflResidualDualTimeControl(beta_start=0.5, beta_min=0.005)`, `inner_steps` 5 / `inner_tol` 1e-2,
  field-split SIMPLE-smoothed leading inverse + Jacobi trailing at `run_ab.py`'s own settings, probe
  reach 5 uniform, `forward_rtol` 0.3, compiled ILU(0) live, measure `coupled_scaled_norm`.

  **The fork, from the march's own inner trajectory** (`|G|` before → after, restart cycles, `alpha`):

  | inner | `OwnerGradient` | `SkewCorrectedGradient` |
  |---|---|---|
  | 0 | 1.17395e-01 → 3.62639e-03, 3 cyc, **α = 1** | 1.17415e-01 → 3.65210e-03, 3 cyc, **α = 1** |
  | 1 | 3.62639e-03 → 2.96989e-03, **3 cyc**, α = 0.5 | 3.65210e-03 → 3.64112e-03, **14 cyc**, α = **0.003108** |

  **The first inner iteration is the same step in both arms to three figures.** The second is where
  they separate, and the attempt is then rejected, `beta` escalates 0.5 → 2 → 16, every retry restarts
  from the anchor, and the march freezes at `|R|` 1.06479e-01 with `alpha` underflowing to 0. So the
  reported "α = 0.001 on the first step" is a step's **minimum inner** `alpha`: the anchor state says
  nothing, which is why no measurement taken at one ever found this.

  **At that iterate, at the march's own β = 0.5, BOTH arms overshoot — and the difference is entirely
  the `omega` block.** The full step, per block, `|delta|` and the scaled `|G|` it lands at:

  | arm | `|delta|` | u | v | p | k | **omega** | whole step `|G|/|G0|` |
  |---|---|---|---|---|---|---|---|
  | owner | 1.97e+02 | 6.79e-04 | 8.07e-05 | 2.34e-04 | 6.79e-03 | **2.04e+02** | **7.6e+03** |
  | skew | 3.80e+02 | 2.17e-03 | 2.49e-04 | 5.45e-04 | 2.98e-02 | **3.75e+21** | **1.4e+23** |

  Every other block lands at 1e-4 to 3e-2. **The descent slope is the same in both arms and negative**
  (−2.6831e-02 against −2.6830e-02), so this is not a direction problem — it is the finite-step
  curvature, which is the shape `.claude/notes/solve-globalization-log.md` already records as
  "descent is necessary but not sufficient".

  **Why twenty orders: `omega` is transported as `log omega`, so the correction is a LOG increment and
  the residual after the step is its exponential.** Where that increment sits, at the same iterate and
  shift, resolved by how many boundary faces a cell owns:

  | boundary faces owned | cells | max `|d log omega|` owner | max `|d log omega|` skew |
  |---|---|---|---|
  | 0 (interior) | 11670 | 4.5416e-01 | **4.5391e-01 — identical** |
  | 1 | 550 | 6.6361e+00 | 1.6066e+01 |
  | **2 or more (corners)** | **5** | **1.5611e+01** | **5.8363e+01** |

  Monotone in boundary-face count in both arms, with the skew/owner ratio rising 1.00 → 2.4 → 3.7. The
  single worst cell is **10824 at the `lowerWall`/`outlet` corner** (`omega` 8.75e+03), at 15.61 under
  owner and **58.36** under skew, decaying into its two wall neighbours (6.64/16.07, 2.52/4.38).
  `exp(58.36 − 15.61) = 3.7e+18` against the measured `omega`-block ratio of 1.8e+19 — the same order,
  so the whole gap between the arms is the exponential of that one extra log increment.

  **✅✅ SETTLED BY A FOUR-WAY COMPARISON: THE RECONSTRUCTION DECIDES THIS, THROUGH BOUNDARY-OWNING
  CELLS, AND THE FOUR SCHEMES ORDER THE SAME WAY ON BOTH SIDES OF THE CHAIN.** Two arms of one scheme
  could never have answered it — three *other* reconstructions march this benchmark to `x_r/h` 8.069
  and only this one loses it, which is the fact any account has to fit. All four at the **same**
  iterate and the **same** β = 0.5, `max |d log omega|` by boundary-face count, beside the rung the
  ladder then keeps:

  | reconstruction | interior (11670) | 1 boundary face (550) | corners (5) | kept α → ratio | marches? |
  |---|---|---|---|---|---|
  | `CorrectedGreenGauss` | 4.5562e-01 | 1.3581e+00 | **1.9120e+00** | **1 → 0.128** | yes |
  | `HessianCorrectedGradient` | 4.3789e-01 | 4.7132e+00 | **9.8349e+00** | 0.5 → 0.605 | yes |
  | multiple-correction, `OwnerGradient` | 4.5416e-01 | 6.6361e+00 | **1.5611e+01** | 0.25 → 0.783 | yes |
  | multiple-correction, `SkewCorrectedGradient` | 4.5391e-01 | 1.6066e+01 | **5.8363e+01** | 0.0625 → 0.955 | **no** |

  **The interior column is the control and it is flat — 0.438 to 0.456, a 4 % spread over four
  structurally different reconstructions.** The boundary columns span **30×** and the corner column
  **30×**, and the kept step orders identically: 0.128 < 0.605 < 0.783 < 0.955. The acceptance
  threshold the march applies sits between 0.783 and 0.955, which is exactly where the fourth arm
  falls off. Nothing about the state, the shift, the preconditioner or the measure varies across that
  table; only the reconstruction does.

  **✅✅ THE CHAIN IS NOW CLOSED, AND THE WALL CELLS' `omega` CORRECTION IS NOT THE `omega` EQUATION'S
  AT ALL — IT IS THE `k` EQUATION'S, TRANSMITTED BY THE FIXATION ROW.** The 472 near-wall `omega` rows
  are replaced by the algebraic fixation `log omega - log omega_wall(k)`
  (`FixedValueCells` + `LogRatioRow`), and `omega_wall`'s log-layer branch carries `sqrt(k)`. So such
  a row has exactly two entries — **1** on its own unknown and `-d(log omega_target)/dk` on `k` — and
  `coupled_scaled_norm` records that the shift leaves it alone. The correction there is therefore an
  identity, not something to be inferred:

      d log omega  =  -R_row  +  (d log omega_target / dk) * dk

  Measured at the stall iterate, β = 0.5, all four arms — `-R_row` contributes essentially nothing and
  the `k` term reproduces the whole thing:

  | reconstruction | max `|d log omega|` at a wall cell | the `k` term alone | identity residual | max `dk/k` at a wall cell |
  |---|---|---|---|---|
  | `CorrectedGreenGauss` | 1.9120e+00 | **1.9136e+00** | 1.3e-06 | +6.8719e+02 |
  | `HessianCorrectedGradient` | 9.8349e+00 | **9.8365e+00** | 3.9e-06 | +1.7674e+03 |
  | multiple-correction, owner | 1.5611e+01 | **1.5612e+01** | 1.5e-05 | +2.1362e+03 |
  | multiple-correction, skew | 5.8363e+01 | **5.8364e+01** | 3.6e-05 | +4.1737e+03 |

  **⚠️ AND THE `k` CORRECTION THAT DRIVES IT IS UNBOUNDED IN THE DIRECTION NOTHING GUARDS.** The
  *decrease* side is `min dk/k = -0.270` in **all four arms** — identical to three figures, and nowhere
  near `-1`, which is exactly why `positive_k_limit` (the global cap, deleted 2026-10-14) reports a cap of 1 at every
  shift measured. The **increase** side is where the arms differ, and that cap was one-sided by construction:
  it bounds only entries with `delta < 0`, the ones that could cross zero. A Newton step asking to
  raise a near-wall `k` by **four thousand times** passes it untouched. At the worst cell
  (`k` = 3.998e-02) the four arms ask for `dk` of +1.43, +7.34, +11.65, **+43.55**.

  So the causal chain, every link measured: **reconstruction → velocity gradient at boundary-owning
  cells → near-wall `k` production → an unbounded upward `dk` there → the `omega` fixation row slaves
  `omega` to `k` → log transport exponentiates it → the line search cannot find a step that is both
  admissible and worth taking.** The chain factor itself is modest (median 0.683, max 4.93, since the
  viscous branch dominates `omega_wall` on this mesh and the log branch's share is small); the size
  comes from `dk/k`, not from the chain.

  **✅✅ AND ONE LEVEL FURTHER UP — WHY `dk` IS THAT LARGE: THE `k` ROW'S OWN JACOBIAN DIAGONAL IS
  NEGATIVE, AND THE SHIFT ONLY JUST OUTWEIGHS IT.** Measured term by term at the worst cell (10824),
  same state, same β = 0.5, all four arms:

  | arm | `\|grad u\|_F` | `P_k` | cap | `R_k` | **`J_kk`** | **`beta d_k`** | **sum** | `-R/(J+bd)` | actual `dk` |
  |---|---|---|---|---|---|---|---|---|---|
  | gauss | 6.0085e+03 | 73.95 | 314.96 | -1.9491e-04 | **-1.2240e-03** | +2.3242e-03 | **1.100e-03** | 0.177 | 1.43 |
  | betchen | 6.4569e+03 | 79.46 | 314.96 | -2.1611e-04 | **-1.6588e-03** | +2.3192e-03 | **6.604e-04** | 0.327 | 7.34 |
  | owner | 6.5384e+03 | 80.47 | 314.96 | -2.2014e-04 | **-1.7403e-03** | +2.3172e-03 | **5.769e-04** | 0.382 | 11.65 |
  | skew | 6.7402e+03 | 82.96 | 314.96 | -2.2787e-04 | **-1.9482e-03** | +2.3098e-03 | **3.616e-04** | 0.630 | 43.55 |

  **Read the first three columns and the last together, because the disproportion is the point.** The
  velocity gradient — the one thing the reconstructions actually differ in — differs by **12 %**;
  the production tracks it at 12 % and the residual at 17 %; and `dk` differs by **30×**. The
  production is well under its Menter cap (83 against 315), so the cap is not masking anything: it is
  strain-driven, and the reconstruction does feed it.

  **What turns 12 % into 30× is that `J_kk` is negative and `beta d_k` is nearly identical across the
  arms, so their sum is a difference of two similar numbers.** The net diagonal keeps only **47 % of
  the shift under `CorrectedGreenGauss` and 16 % under the skew closure**; a 12 % move in `J_kk` is a
  **3×** move in the effective diagonal, and `-R/(J + beta d)` goes 0.177 → 0.630 on it. (The actual
  `dk` is 8–69× that one-row estimate, so off-diagonal coupling amplifies further — but the ordering
  is set here.)

  **`J_kk < 0` is the `k` equation's own positive feedback, and it is the textbook Patankar case.**
  `nu_t = a_1 k / max(a_1 omega, S F_2)` is proportional to `k`, so `P_k` is proportional to `k`, and
  subtracting a source that grows with the variable puts a negative term on that row's diagonal. This
  case already half-treats it: `explicit_production_limiter=True` freezes the **cap's** derivative for
  exactly this reason ("the Jacobian carries the k-production cap's own derivative, which is
  destabilizing; freezing it is the Patankar treatment"), but the **production's own** `k`-derivative
  is still differentiated, and that is what makes the diagonal negative here.

  **So the ordering of the four reconstructions is real and the chain is real, but the reconstruction
  is not the fault — it is the perturbation that a near-cancelling diagonal amplifies thirtyfold.**
  Any of the four could be on the wrong side of it; on this case, at this shift, the skew closure is.

  **✅✅ THE PATANKAR TREATMENT WAS BUILT AND TESTED AT THAT STATE, AND IT WORKS — 2026-08-24.**
  `closure_stall_probe.frozen_production_residual` rebuilds the coupled residual exactly as the
  library assembles it, changing one thing: the eddy viscosity handed to the `k` equation is evaluated
  at a `stop_gradient`-ed `k`. It is used as the **operator only** (`_shifted_solve`'s `jacobian_fn`),
  so the residual, the root and the IFT adjoint are untouched — the `|R|` column below is identical
  between each pair, which is the control. Same state, same β = 0.5, same cell (10824):

  | arm | operator | `J_kk` | `J_kk + beta d` | max `\|dk/k\|` | max `\|d log omega\|` | kept α | `\|G\|/\|G0\|` |
  |---|---|---|---|---|---|---|---|
  | owner | exact | **-1.7403e-03** | 5.7698e-04 | 2136 | 15.61 | 0.25 | 0.7827 |
  | owner | **frozen** | **+4.7985e-03** | 7.1158e-03 | **6.43** | **0.119** | **1** | **0.1029** |
  | skew | exact | **-1.9482e-03** | 3.6159e-04 | 4174 | 58.36 | 0.0625 | 0.9547 |
  | skew | **frozen** | **+4.7892e-03** | 7.0990e-03 | **6.68** | **0.124** | **1** | **0.1032** |
  | gauss | exact | **-1.2240e-03** | 1.1002e-03 | 687 | 1.912 | 1 | 0.1276 |
  | gauss | **frozen** | **+4.7958e-03** | 7.1200e-03 | **6.01** | **0.110** | **1** | **0.1020** |

  **The unforeseen result is the strongest evidence, and it retro-confirms the whole chain: after the
  freeze the three reconstructions become INDISTINGUISHABLE.** `J_kk` goes from -1.22e-03 / -1.74e-03
  / -1.95e-03 — a 60 % spread that orders the arms exactly as their step quality does — to
  **+4.7892e-03 / +4.7958e-03 / +4.7985e-03, identical to 0.2 %**. So the reconstruction-dependence of
  that diagonal *was* the production's `k`-derivative and nothing else. Everything downstream follows:
  `dk/k` collapses three orders to ~6 in every arm, `d log omega` to ~0.12, and all three take a
  **full step** to `|G|/|G0|` ≈ 0.102 — better than the best exact-operator arm (gauss at 0.128).

  ⚠️ **What this does NOT yet establish.** One state, one shift, no march. A quasi-Newton operator can
  cost convergence rate near the root, where this iterate is not. The probe freezes `nu_t` throughout
  the **`k` equation** (it replaces `closure.nu_t`, which reaches the `k` diffusivity as well as the
  production), so it is not surgically the production term alone. And the production's other `k`
  paths — the adaptive near-wall blend's own `k`, and the Menter cap unless
  `explicit_production_limiter` is set — are untouched. **The march A/B is the test that matters and
  has not been run.**

  ⚠️ **This SUPERSEDES the earlier fix proposed here (freezing the `omega` fixation row's `k`
  column).** That would have stopped the `omega` rows from *learning* about a `dk` which is itself the defect —
  treating the symptom one link downstream of the cause. The live candidates are now, in order:
  1. **Patankar-treat the `k` production** — freeze `nu_t`'s `k`-dependence in the production term's
     Jacobian contribution, so the diagonal is not driven negative by the very feedback the shift is
     then asked to cancel. Precedent and seam both exist (`explicit_production_limiter`, and
     `_shifted_solve`'s `jacobian_fn`). **Unmeasured.**
  2. **Keep the shift above the cancellation.** The recorded β sweep already shows β ≥ 0.6 fixing every
     arm at this state, which is the same statement from the other side: `beta d_k` at 0.6 is 2.79e-3
     against `J_kk` of -1.95e-3, restoring the margin. A floor is cruder than (1) but is a
     step-control setting.
  3. Not a step-length cap, and not the closure. See the checks above.

  ⚠️ **A cap is the wrong instrument, and this is now checkable rather than arguable.** Capping
  `d log omega` at 3 would give α = 1.57 (uncapped) for `CorrectedGreenGauss`, 0.305 for the
  Hessian-corrected arm, 0.192 for owner and 0.051 for skew — i.e. *approximately what the ladder
  already picks in each case*, because the ladder is already shortening the step until `omega` is
  sane. The same is true of a relative cap on `dk`: the arms that **work** also carry `dk/k` of 687
  and 1767 at a wall cell and take full steps happily. **The problem is the direction, not the
  length** — skew's correction requires a 4000× near-wall `k` increase in order to reduce the
  residual, and any step short enough to keep that sane makes negligible progress everywhere else.

  **The lever is therefore upstream of the step, and there is a precedent in this very case.** The
  fixation row's `k` column is what converts a `k` overshoot into an `omega` overshoot; dropping it
  from the **operator only** — via `_shifted_solve`'s existing `jacobian_fn` seam, which takes a
  stand-in residual for the Jacobian while the true residual still decides where the march lands —
  would leave the root and the adjoint untouched and remove the amplifier from the direction. That is
  the same trade `explicit_production_limiter=True` already makes on this case for the `k`-production
  cap ("the Jacobian carries the cap's own derivative, which is destabilizing; freezing it is the
  Patankar treatment"). **Proposed, NOT measured.**

  **The mechanism this supports, offered as a reading of the ordering and NOT separately isolated:**
  the four differ in how much of the returned gradient comes from a **boundary face's gradient**.
  `CorrectedGreenGauss` builds no Hessian, so a boundary face contributes its *value* and nothing
  else — the smallest boundary term and the best-behaved arm. The other three form a Hessian by a
  second pass over the gradient field, in which a boundary face contributes `face_gradient · A/V`, and
  at a wall cell `A/V` is `~1/(d·n)` ≈ 5.5e+03 per metre; the second-order correction built from that
  Hessian is separately measured at up to **41 %** of the first-order gradient. What the two closures
  then differ in is precisely the boundary face gradient they supply: `OwnerGradient` hands over the
  full cell gradient, while `SkewCorrectedGradient` on a gradient-type patch zeroes the normal
  component and keeps only the tangential part. A cell owning **two** boundary faces has two
  independent normal directions so replaced, which is why the corner column is the extreme in every
  arm (4.2×, 22×, 34×, 128× that arm's own interior maximum).

  ⚠️⚠️ **THIS ENTRY WAS WRITTEN WRONG TWICE, AND BOTH ERRORS ARE WORTH MORE THAN THE RESULT.**
  1. **First it attributed the whole thing to the closure's corner behaviour, from a single shift.**
     That was under-determined: two arms at one β cannot separate "the closure does this" from "this
     state is fragile and the closure nudged it".
  2. **Then a β sweep was read as refuting the closure entirely, and that was worse.** The sweep
     showed both closures producing `d log omega` of order 10–70 below β ≈ 0.55, with *which arm is
     worse alternating* — owner 5.5× worse than skew at β = 0.45 — and that was written up as "a
     low-shift wall both closures hit, `beta_start` sits on its edge, the closure is not the cause".
     **The alternation is between two configurations that have BOTH already failed**: every ratio at
     β ≤ 0.45 is 0.983–0.998, i.e. no usable step in either arm. Ranking two dead configurations is
     not evidence, and reading a ranking out of one is the same error as reading a tie at a benign
     operating point as parity — this file's own recorded trap, arrived at from the other end.
     At every β where a usable step exists the ordering is stable, and the four-way table settles it.

  **What survives of the low-shift observation, correctly scoped:** below β ≈ 0.45 *every* arm
  measured fails at this state, the linear solve degrades monotonically (3.4e-07 at β = 0.7 to
  3.4e-03 at β = 0.3), and the k-positivity cap starts to bind (0.0022–0.0040, against exactly 1 at
  every β ≥ 0.5). That is a real background fragility of this state and a reason not to let a control
  drive β far down here — but it is **not** what separates the arms, since at the operating β = 0.5
  `CorrectedGreenGauss` takes a full step with room to spare.

  ⚠️ **What log-`omega` does and does not do, since the first write-up blurred it.** It does **not**
  corrupt the reconstruction — the interior column above is flat across all four schemes. It is the
  amplifier at the *end* of the chain: the correction is an increment to `log omega`, so an entry of
  58 means "multiply this cell's `omega` by `e**58`", and the residual at the trial point is that
  exponential. Its real cost is to the **line search**, since halving α only takes the square root of
  the `omega` factor — recovering from 58 needs six halvings, and a step that short makes no progress
  even once it is admissible. That is the whole failure: not "no admissible step", but "no step that
  is both admissible and worth taking". Direct `omega` would be worse, not better — the same
  correction would drive it far negative, which is what log transport exists to prevent and what
  the k positivity guard (then `positive_k_limit`, deleted 2026-10-14) prevents for `k`.

  **✅✅✅ RESOLVED (2026-08-25): THE STALL IS THE k-POSITIVITY CAP, AND THE PER-ENTRY PROJECTION FIXES
  IT. `SkewCorrectedGradient` MARCHES THIS CASE.**

  | arm | wall | cycles | steps | `x_r/h` |
  |---|---|---|---|---|---|
  | `OwnerGradient`, as shipped (no projection) | 703.7 s | 437 | 73 | 8.069 |
  | **`OwnerGradient` + `positivity_projection`** | **664.0 s** | 459 | **67** | 8.069 |
  | **`SkewCorrectedGradient` + `positivity_projection`** | **660.4 s** | 467 | **67** | 8.069 |
  | `SkewCorrectedGradient`, as shipped | — | — | **stalls at the target rung** | — |

  Both projection arms on the **exact** operator (`frozen production viscosity: False`), final `|R|`
  5.77e-06 against a 1e-05 target, `alpha` = 1.000 into the tail. The projection **also improves the
  arm that already worked** (−5.6 % wall, −8 % steps, +5 % cycles, same answer) and collapses the two
  arms onto each other — 660.4 s against 664.0 s, 67 steps both. At march level the reconstruction
  sensitivity this whole entry is about is *gone*.

  ⚠️⚠️ **AND THE MECHANISM RECORDED ABOVE IS WRONG WHERE IT NAMES THE LINE SEARCH.** The tables above
  are correct about `dk`, `d log omega` and the four-way ordering — but they were measured in a probe
  that builds a **fresh preconditioner at the state it measures** and reaches linear residuals of
  1e-05 to 1e-10. **The march is never in that configuration** (`forward_rtol = 0.3`, a carried
  preconditioner), and in the march the step length was set by `positive_k_limit`, not by the ladder:

  - `_COUPLED_LINE_SEARCH = 10`, so a rung is `0.5**k` down to **9.766e-04**. The failing march
    `alpha`s are **0.003108, 0.154, 0.03744, 0.0004484, 0.1149, 0.001161** — not one is a rung, and
    **0.0004484 is BELOW the shortest**, which only the line search's `max_alpha` cap (deleted 2026-10-14) could produce.
  - The terminal freeze is the **fraction-to-the-boundary ratchet** `positive_block_projection`'s own
    docstring derives: the inner block runs `8.462e-06 → 8.353e-08 → 8.353e-10 → 8.353e-12`, ratios
    `9.871e-03, 1.000e-02, 1.000e-02` — exactly `1 - tau` at `tau = 0.99`.
  - So this entry's bullet "the k-positivity limiter is **not** binding — its cap reports exactly 1 for
    both arms at every shift measured" is **true of the probe and false of the march**, and the reading
    "the ladder reaches an admissible step but not one worth taking" describes the probe only.

  **The lesson, and it is the third time in this entry:** a probe that rebuilds a self-consistent
  (state, shift, preconditioner) triple measures a configuration the march never occupies — which is
  the trap `solve-globalization-log.md` already records as "THE CARRIED PROTOCOL", arrived at here
  from a new direction. The α values being non-rungs was visible in the march log the whole time.

  **What the reconstruction difference actually did:** it produced a wildly different near-wall `dk` in
  both arms (the four-way ordering is real), and the **global** cap then let that one entry set the
  step length for all 61125 degrees of freedom. `positive_block_projection` clips each entry's own
  correction instead, so a bad entry costs only itself.

  ⚠️ **THIS IS A pitzDaily RESULT AND IT DOES NOT GENERALIZE — the 3D sibling was A/B'd on 2026-08-25
  (#314) and the projection costs it 41 % more Krylov cycles and 40 % more wall**, same root
  (`x_r/h` 8.3611 both arms), 69 steps → 62. `bfs3d_openfoam` therefore keeps
  `K_POSITIVITY_PROJECTION` **off**, deliberately out of step with the library default this table set.
  The projection still does exactly what it claims there (the cap binds 26 times against 0) and still
  wins the first two continuation rungs outright; the whole loss is the target rung, where the global
  cap turns out to have been doing **globalization** work — with the step no longer shortened, the line
  search takes full steps into iterates the carried preconditioner solves badly (inner solves pinned at
  12 cycles against 2–5, one attempt at α = 1.000 reaching an inner residual of 2.7e+10), each tripping
  that case's per-solve cycle bailout. Full numbers and configuration in
  `.claude/notes/solve-amg-multigrid-log.md`, "The traced multigrid — in progress". ⚠️ **All of this table was measured with the global cap and its
  projection switch, both deleted 2026-10-14**: every coupled step now clips `k` per cell AND bounds ω's
  per-cell change, which removed bfs3d's bad full-length trials (`.claude/notes/solve-globalization-log.md`,
  "bfs3d's retried steps: the global k cap, replaced by a per-cell clip with an omega bound (2026-10-14)").
  **So read the table above as "the cap can lose a march and the projection can rescue it", which is
  what justifies the default — NOT as "the projection is free".** On a case where the cap is not losing
  the march, it is not.

  **WHERE TO LOOK NEXT:**
  - **A cap on the log-`omega` increment does NOT work — checked against the measured ladders, not
    assumed.** A fraction-to-the-boundary analogue limiting `omega` to a factor `e**2` per step gives
    α = 2/58.4 = 0.034 for the skew arm (ratio ≈ 0.96, still not worth taking) and α = 2/15.6 = 0.128
    for the owner arm, **worse than the 0.25 its ladder already finds**. It penalizes the arms that
    work. The step is not too long; the boundary-cell correction it is made of is too large.
  - **The lever is the boundary-face gradient a Hessian-forming scheme is fed**, since that is the one
    thing the four arms vary. Whether the right move is to damp its contribution at cells owning two
    or more boundary faces, or to keep the normal component there as `OwnerGradient` does, is
    unmeasured — but `_undetermined_cells` already identifies such cells and `CellwiseFallback`
    already applies a per-cell closure, so the machinery exists.
  - **Separately, the escalation does not rescue the march even though β = 2 is comfortable at this
    iterate, and that is unexplained.** The retry **from the anchor** at β = 2 keeps α = 0.154 and a
    ratio of 0.846, while a solve built fresh at that same anchor and shift keeps **α = 1 and 0.027**.
    The measure is not the explanation (both are the anchor's). The candidate is the preconditioner:
    the 14-cycle solve trips `refresh_on_cycles = 3`, so the rebuild happens at the *bad* iterate and
    the retry then runs on a preconditioner built elsewhere — and at `forward_rtol = 0.3` a worse
    preconditioner returns a materially worse `delta` at the same reported cycle count. **Candidate,
    not measured.** It is what makes the failure permanent rather than a one-step stumble.


## `boundary_gradient_weight` — measurements

  **Marched end to end on pitzDaily, and the answer does not move** -- which is the acceptance
  criterion for a discretisation change at a boundary:

  | configuration | wall | cycles | steps | `x_r/h` |
  |---|---|---|---|---|
  | skew closure, before either change | — | — | **stalls at the target rung** | — |
  | skew + positivity projection | 660.4 s | 467 | 67 | 8.069 |
  | **skew + projection + the first-form weight (`w = d`)** | **707.6 s** | **456** | **68** | **8.069** |

  Same reattachment length to four figures, 2 % fewer Krylov cycles, one more outer step. ⚠️ Measured
  with the first form of the weight, not re-run with the tangential one. The +7 %
  wall is **one run**, and this case has no measured march-level noise floor -- read it as neutral,
  not as a cost, until a repeat says otherwise.

  **⚠️ SEPARATELY, AND FOUND ON THE WAY: THE `boundary_values_at` FIX HAD NEVER REACHED THE FLOW
  BLOCK. FIXED IN #313; the census below is what it was measured on, and is kept for the trap it
  taught.** Three call sites in the library reconstruct a gradient — `ResidualAssembler._gradient` and
  `MomentumContinuity`'s velocity and pressure reconstructions — and only the first passed
  `boundary_values_at`. The flow block could not: `FlowBoundary.pressure_face` / `velocity_face` took
  **no gradient argument at all**, so a zero-gradient flow face value was the owner cell's own value
  permanently, whatever gradient was reconstructed, and the double correction was fully live there for
  any closure that reads a boundary value. Both closures now take the owner gradient and the
  owner-centroid→face displacement, and both flow reconstructions run the two-pass shape and pass
  `boundary_values_at` — see `.claude/rules/flow.md`'s `corr` reconciliation bullet for the shape and
  the analytical test. **The numbers below are therefore a record of the defect, not of current
  behaviour**; the ratio they establish (a spurious normal derivative four times the real gradient at
  the outlet) is what makes the repair worth its cost, and the closing "1e-4 on this mesh" is why it
  was never *visible* on pitzDaily.

  Measured at the divergence iterate, on the flow block's own faces (`closure_stall_probe.py`'s census
  mirrors `boundary_closure_probe.py`, which only ever reached the scalar equations):

  | field | patch | kind | faces | max `|bval − phi_P|` | med `|d.n|` | max `|rise/(d.n)|` | med `|grad phi|` |
  |---|---|---|---|---|---|---|---|
  | p | inlet | ZeroGradient | 30 | **0.000e+00** | 7.9e-04 | 1.1e-13 | 8.1e+01 |
  | p | upperWall | ZeroGradient | 223 | **0.000e+00** | 1.8e-04 | 1.7e+01 | 7.1e+01 |
  | p | lowerWall | ZeroGradient | 250 | **0.000e+00** | 5.1e-04 | 2.4e+01 | 2.0e+01 |
  | p | outlet | Dirichlet | 57 | 1.918e+00 | 2.6e-03 | 7.5e+02 | 3.8e+02 |
  | U0 | outlet | ZeroGradient | 57 | **0.000e+00** | 2.6e-03 | 8.7e+02 | 2.1e+02 |

  Exactly zero on every gradient-type flow patch and only there — 503 of 560 boundary faces for
  pressure — so the normal derivative reported on them is entirely the artifact. **But on this mesh it
  does not matter**: supplying the corrected value (`p_face + chain * non_orthogonal_correction`, with
  `chain = d(p_face)/d(p_owner)` read off the case's own closures by differentiating them) moves the
  reconstructed pressure gradient by **1e-4 relative** — 2.577e-02 against 2.578e-02 as a difference
  from the owner arm. pitzDaily's worst face angle is ~6°, so the non-orthogonal correction is ~0.1 %
  of the local variation and the double correction is negligible here. **That is why it survived: a
  defect measured not to bite on a nearly-orthogonal mesh reads as "not a problem" until someone runs
  a skewed one.** Repaired in #313; the analytical pin lives on an 8×8 grid perturbed by 0.2 of a cell,
  where the same defect moves the outlet's face velocity by 1.15e-2 and the momentum residual at the
  exact Stokes field by 2.8e-2 — i.e. the size of the effect is a property of the *mesh*, and pitzDaily
  is the benign end of it, not the representative one.

  Worth keeping beside that: the two closures' reconstructed flow gradients differ **only on the 555
  boundary-owning cells** (relative L2 6.7e-02 for `grad p`, 4.5e-02 and 1.9e-02 for the velocity
  components) and are **bit-identical on all 11670 interior cells** — 0.000e+00, not "small". A closure
  can only reach a cell that owns a boundary face, and it does not reach any other.

  ⚠️ **Caveat on the `alpha` values, which is the one place the probe is not the march.** The probe
  rebuilds `coupled_scaled_norm` at the state it measures, where the march holds the anchor's measure
  fixed across a step — so the *rung the ladder keeps* differs (probe 0.25 against the march's 0.5 for
  owner). Every per-block and per-cell figure above is unaffected, and the ratios within one arm are
  taken in one fixed measure; do not quote the probe's `alpha` as the march's.

  **⚠️⚠️ MEASURED ON THE REACTOR MESH (2026-08-24), AND IT REFUTES THE OBVIOUS INFERENCE: BOTH
  CLOSURES RECONSTRUCT A QUADRATIC EXACTLY THERE.** The reasoning that "the reactor has boundary
  tetrahedra, therefore `OwnerGradient` fails there" was written into this file and is **wrong**.
  Measured with `validation/uvreactor_openfoam/multiple_correction_closures.py` on the full 1,635,909-cell
  snappyHexMesh mesh (5,249,365 faces, 377,421 cells owning a boundary face), reusing
  `gradient_sweep_calibration.probe_field`'s quadratic and its median/p99/max error:

  | closure | gradient error median / p99 / max | `max \|M2⁻¹\|` |
  |---|---|---|
  | **`OwnerGradient`** (the default) | 1.317e-13 / 7.006e-13 / **4.906e-12** | **1.86e+01** |
  | `SkewCorrectedGradient` | 1.305e-13 / 6.987e-13 / 2.056e-11 | 9.35e+00 |

  Both exact; the default is *better* on the max. **So the default is right for the reactor, and no
  closure conflict exists between it and pitzDaily.**

  ⚠️ **What this does NOT do is explain the stall of GLOBAL skew, which is still unknown.** The
  fallback removes the need to apply it globally; it does not make it safe to. On a mesh that *does*
  need repair the closure runs on those few cells, and whether that is safe on a march is still
  untested — a case with the cell shape now exists (below) but cannot currently be marched at all,
  under either closure, for reasons unrelated to this defect (#435). A fourth candidate difference was
  measured and is
  recorded as unquantified rather than as a cause: at a developed near-wall `omega` field the two
  closures' reconstructed `|grad omega|` differs by up to **16.6 % on 490 of 12225 cells** (median
  0), which persists *after* the `boundary_values_at` fix. At the cold initial condition the coupled
  residual agrees to 0.9–14 % per block and the Jacobian action to **2 %**, so whatever it is, it is
  a developed-state effect.


## `CoupledBlockSweep` — cost and boundary treatment

  **⚠️ MEASURE IT AGAINST A CALIBRATED NESTED SOLVE, NOT THE SHIPPED DEFAULT.** Against `20/10` it
  looks like 2.0× forward and 3.1× on the tangent — but `20/10` is heavily over-provisioned on the
  meshes that comparison used, so most of that gap is the baseline's slack rather than this sweep's
  merit. Both calibrated at the shared default tolerance (8000 cells, wall clock):

  | mesh | arm | forward | jvp | error vs analytic |
  |---|---|---|---|---|
  | orthogonal | nested 5×1 | **24.5 ms** | 28.3 ms | 3.42e-06 |
  | orthogonal | coupled 6 | 27.9 ms | 28.2 ms | **5.04e-07** |
  | warped p=0.3 | nested 6×4 | 36.6 ms | 46.6 ms | 5.20e-06 |
  | warped p=0.3 | **coupled 7** | **30.7 ms** | **30.5 ms** | **1.09e-06** |

  So honestly: **1.2× forward and 1.5× on the tangent on a skewed mesh, at 5× the accuracy** — and
  slightly *slower* forward on an orthogonal one, where the nested outer solve is nearly trivial and
  there is little to save. In calibrated face-passes: 0.6× on orthogonal, 1.4× at 30 % perturbation,
  1.8× at 40 %. The advantage grows with non-orthogonality, which is the regime this scheme is for.

  **It beats the tolerance it is given, and that is structural.** The rate is measured on the packed
  `[g, h]` error because the blocks converge together, but the Hessian's error dominates that
  estimate while the gradient's falls faster — so the count is conservative for what is returned. A
  comparison at equal *tolerance* is therefore not a comparison at equal *accuracy*.

  **Boundary treatment: audited and correct.** Boundary cells reconstruct as exactly as interior ones
  (1.95e-15 vs 2.26e-15 median on a warped mesh). `f = 0` there, `skew` collapses to `d_own`, and the
  unconditionally-computed neighbour side is discarded *structurally* by `scatter` — its index points
  one row past the last cell and is sliced off — rather than by a mask. ⚠️ Betchen's own boundary
  caveat (his Eq. 29, an inverse-distance-averaged Hessian over interior neighbours) is **not**
  implemented; we use the simpler Eq. (28) form. He raises it for a 3D mesh one cell thick, which the
  empty-patch collapse turns into a genuine 2D mesh here, so it likely cannot arise — but that has not
  been proven.
  **Validated inside the solve** (`tests/integration/test_betchen_solve.py`): injected as the
  residual's gradient scheme it nests correctly (the outer Newton `jvp` differentiates through
  Betchen's Schur solve *and* its inner `A_HH` solve), converges in one Newton
  step, and is differentiable on a skewed mesh. **Finding:** in *pure diffusion* the gradient
  enters only as a small non-orthogonal correction, so the solved **field** order is set by the
  operator floor (~2nd) and matches `CorrectedGreenGauss`; Betchen's win shows in the
  **reconstructed gradient / flux** of the solved field (~4× smaller error, higher order on
  skewed, where corrected Gauss caps near 1st — the `ResidualAssembler.gradient(phi)` accessor).
  The scheme matters most where the gradient enters a face value at leading order — advection /
  Rhie–Chow.


## The inner `A_HH` solve (2026-08-21)

  - **⚠️ THE `I_dim ⊗ C` BLOCK STRUCTURE IS GONE — it was a property of the UNSYMMETRIZED Hessian and
    the scheme now solves the six independent components (2026-08-22).** It held exactly (measured
    departure `0.000e+00` in 2D and 3D) because `H` entered its own equation only as `H·a` for per-face
    vectors `a`, contracting one index and leaving the other untouched — so the block stored as
    `dim×dim` rather than `dim²×dim²`, 72 B/cell against 648 in 3D. Imposing symmetry couples those two
    indices, so the reduced block is a dense `(n_sym, n_sym)`: **288 B/cell in 3D**, four times the old
    figure, while the unknown drops from 72 to 48. There is no such test any more; the property now
    pinned is that the reduced block is symmetric positive definite
    (`test_the_reduced_hessian_block_is_symmetric_positive_definite`). The unreduced `C` still exists
    and is still `I ⊗ C` — it is what weights the reduction — so the extraction below is unchanged.

  - **Cost, and the honest caveat.** On pitzDaily (12225 cells, skewness p99 0.016) the swept inner at the
    shipped 10 sweeps reconstructs **bit-identically** to the exact Krylov inner (`0.000e+00`) at
    **1.04 s against 4.25 s**; `CorrectedGreenGauss` is 0.15 s on the same field, so Betchen is ~7×
    that scheme and would be ~3.6× at a calibrated 4 sweeps. **Single runs on a shared machine — quote
    the structural claim (no nested Krylov, no implicit-diff tangent on the inner solve, cell-local
    preconditioner) rather than the seconds.**
  - **Choosing the sweep count is a MESH property and cannot be inferred from a test mesh.** Departure
    from exactness for a quadratic falls ~2.5 orders per 2 sweeps (3D hex, p=0.25): 4.8e-04 at 2,
    1.3e-06 at 4, 3.0e-09 at 6, 8.6e-12 at 8, machine precision at 10 = the default. A fixed sweep has
    no convergence test and the inner solver's warning is off by necessity, so an under-resolved mesh
    loses quadratic exactness **silently** — calibrate with
    `validation/uvreactor_openfoam/gradient_sweep_calibration.py`, which walks the ladder against a
    converged reference on the real mesh. On pitzDaily it says 4–6 sweeps suffice, i.e. the default is
    conservative there.
  - **A swept OUTER solve reaches the Krylov gradient to 1e-12** (60 sweeps, p=0.3), so an entirely
    Krylov-free, inner-product-free, unrolled-differentiable reconstruction is available —
    `HessianCorrectedGradient(solver=SweptGradientSolve(...))`. Not the default, and **not yet
    cost-compared on a march**.
  - **NOT re-enabled: the distributed path.** `HessianCorrectedGradient` still raises on
    `operator_hook`. The pieces are now much closer — the preconditioner is cell-local and a swept
    outer solve forms no global inner product — but the inner solve's own ghost exchange is not
    threaded, so this is an opportunity, not a claim.
  - **⚠️ NOT DONE: the blocks are rebuilt on every `gradients()` call.** They are geometry-only and
    could be built once per mesh, but `gradients()` builds all its terms inline (as
    `CorrectedGreenGauss` does), so `2·dim` extra applies are paid per reconstruction. Hoisting
    geometry-only reconstruction terms out of the residual is a shared refactor with
    `CorrectedGreenGauss.terms`, not a Betchen-local one.
  - **Measured but NOT acted on: `CellBlockJacobi` helps `CorrectedGreenGauss` too** — ρ 0.1365 → 0.0419
    at p=0.2 and 0.2568 → 0.0682 at p=0.3, a 3–4× better rate, which would make `sweeps=4` far more
    accurate on a skewed mesh or allow fewer sweeps. Its default is **unchanged** (`InverseVolume`);
    changing a shipped default is not something to do without being asked.
  - **REFUTED as the production route: solving the full coupled `[g, H]` system instead of eliminating.**
    With the exact per-cell 12×12 block it converges at ρ ≈ 0.38 for **one** operator apply per sweep,
    against the Schur route's ρ ≈ 0.13 for `k+2` applies — so per unit work the coupled route is ~2.5×
    cheaper (24–27 applies to 1e-10 against 60–108). It is nonetheless **not** the route: the eliminated
    system keeps the gradient as the only primary unknown, which is the point of the elimination and what
    lets it Schur-couple into the flow Newton at gradient size. Recorded because the cost argument is
    real and will be re-derived by anyone who measures it.


## The Betchen residual's stencil on pitzDaily (2026-08-21)

  - **⚠️⚠️ BUT IT DOES NOT BITE ON A NEARLY-ORTHOGONAL MESH, AND THE REACH THAT MATTERS IS A PROPERTY OF
    THE SMOOTHER FAMILY, NOT OF THE RESIDUAL'S STENCIL ALONE — measured on `pitzDaily` 2026-08-21. Read
    this before acting on the alarm above; the entry as first written over-weighted the obstacle.**
    - **Both gradient arms take the SAME cycle counts at the benchmark's own reach 5.** A capped march
      (6 steps, the case's shipped incomplete-LU bundle) gives corrected Green–Gauss `[2,2,2,3,3,6]` and
      the Hessian-corrected scheme `[2,2]` over the steps it reached — so the long stencil is not
      starving the probe here at all. The far-field mass scales with skewness, and this mesh's median
      skew is `2.2e-09` against the 5–20 % perturbed grids the table above was measured on.
    - **⚠️⚠️ THE SWEEP BELOW IS VOID — THE HARNESS PRINTED A REACH IT WAS NOT PROBING AT. Do not cite the
      "bit-identical cycles at 2, 3 and 5" table.** `run_ab.py`'s `solve_arm` derived a local `reach`
      from its argument, used it in the **banner**, and built the probe from the module-level `REACH`.
      So the three arms printed 2/3/5 and probed the same value each time — which explains
      bit-identical cycles far better than "the reach is inert" did. This is the validation rule *a
      setting the banner prints must be a setting that is in force* violated exactly as written, and it
      is the second time this same claim has failed for a see-the-variable reason (the first being that
      it was swept on an arm that could not feel it). **What survives** are the runs driven by the
      `PITZ_AB_REACH` environment variable, which sets the global the probe really read: the standard
      arm at reach 3 vs 5 (432 vs 439 cycles, 527 vs 711 s) and the Betchen arm (834 vs 511 cycles).
      Those are genuine, and they are what the entries below rest on. Fixed 2026-08-21.
    - **On the SIMPLE-smoothed field-split bundle the reach is inert, and the probe is 3.8× oversized.**
      Same case, standard arm, 6 steps of one Reynolds rung, leading inverse `SimpleSmoothedInverse`
      (sweeps 2, pressure_sweeps 2, θ=0.25, no singletons, 5 levels, max_coarse 500, block splitting,
      ω=1.0) with a nodal trailing inverse:

      | probe reach | probes per refresh | plan build | march cycles |
      |---|---|---|---|
      | 2 | **100** | 0.27 s | `[4, 3, 3, 3, 3, 8]` |
      | 3 | **165** | 0.41 s | `[4, 3, 3, 3, 3, 8]` |
      | 5 (the benchmark's) | **380** | 0.86 s | `[4, 3, 3, 3, 3, 8]` |
      | 7 | 670 | 1.60 s | — |

      **Bit-identical cycles, at 3.8× the probes.** ⚠️ And the null is a real one: the knob demonstrably
      has teeth (100 → 670 probes, 0.27 → 1.60 s to build), which is the check that had to pass before
      a flat result could be read as "the reach buys nothing" rather than as "the reach was ignored".
    - **Why the two bundles differ, and it is the mechanism that generalizes:** `pitzDaily`'s reach-5
      calibration was measured against an **incomplete-LU** smoother, where the folded far entries
      become *pivots* — its record has reach 3 + fill 1 failing outright (300 matvecs, true residual
      3.36). A SIMPLE-smoothed hierarchy never eliminates the matrix; it forms an approximate Schur
      complement and applies V-cycles, whose rate is set by the strong near couplings, so a perturbed
      far entry degrades a rate instead of wrecking a factorization. **Do not carry a reach calibration
      across smoother families.**
    - **⚠️⚠️ AND IT IS ARM-SPECIFIC — MEASURED ON THE STANDARD GRADIENT ONLY, AND IT DOES NOT TRANSFER
      TO THE HESSIAN-CORRECTED ONE (2026-08-21).** The sweep above varied the reach with
      `CorrectedGreenGauss` in place, whose Jacobian carries *exactly zero* mass past its own reach — so
      it could not have detected a reach effect on a long-stencil reconstruction, and reading it as a
      property of the preconditioner was wrong. Measured on full `pitzdaily_gradient_ab` marches, the
      Betchen arm at outer/inner swept-5:

      | Betchen arm | cycles | per step | max | wall |
      |---|---|---|---|---|
      | probe reach 3 | 834 over 74 steps | 11.27 | 36 | 1715 s |
      | probe reach 5 | **511 over 72 steps** | **7.10** | **20** | **1604 s** |

      **A third fewer Krylov cycles**, and the cycle ratio against the standard arm falls from **1.93×
      to 1.18×** — i.e. at reach 3 the probe was genuinely under-resolving that arm's stencil and the
      cost difference was being mis-attributed to the reconstruction. ⚠️ **Note the wall clock barely
      moves** (1715 → 1604 s): the longer probe costs 380 residual evaluations per refresh against 165,
      which eats most of the cycle saving. So a longer reach here buys a *comparable measurement*, not a
      faster march — and the two are easy to confuse.
    - **⚠️ What this does NOT license.** It is six steps of the *first* Reynolds rung, on the standard
      gradient arm. This case's own history is that step-initial solves are the cheap ones and the hard
      operators appear mid-step and in retries, so this supports shortening the reach on this bundle
      pending a full march — not a default change on the validated benchmark, whose reach-5 pairing with
      its own smoother is untouched by this. The sweep's **wall-clock column is void** (the first arm
      paid JIT compilation); quote the probe counts and the cycles.
  - **Still open:** whether the shortened reach survives a full three-rung march, and whether the
    genuinely probe-free family (`coupled_continuation`'s block-diagonal SIMPLE + scalar AMGs, which
    takes no `stencil_reach` at all) is competitive on this case.


## Krylov outer cost, the first Betchen A/B, and the sweep-count lever (2026-08-21)

- **⚠️⚠️ THE KRYLOV OUTER SOLVE COSTS 5× THE SWEPT ONE ON THE PATH A MARCH ACTUALLY PAYS — profiled
  2026-08-21, and it is the single largest number in this subsystem.** *Configuration: `pitzDaily`
  (12225 cells, 2D), the case's own assembly, `eqx.filter_jit`-ed, warm, min of 7 reps, on an
  otherwise-idle machine. `fwd` is one reconstruction; `jvp` is `jax.jvp` of it, which is what every
  Krylov iteration of the flow solve costs.*

  | arm | fwd ms | ×base | jvp ms | ×base | **jvp/fwd** |
  |---|---|---|---|---|---|
  | `CorrectedGreenGauss` swept-4 (baseline) | 0.77 | 1.0× | 0.81 | 1.0× | 1.1× |
  | Betchen outer swept-**1** / inner swept-1 | 1.94 | 2.5× | 2.06 | 2.5× | 1.1× |
  | Betchen outer swept-10 / inner swept-10 | 25.50 | 33.3× | 25.55 | **31.5×** | **1.0×** |
  | Betchen outer swept-20 / inner swept-10 | 50.87 | 66.5× | 48.61 | 59.9× | 1.0× |
  | Betchen outer **GMRES** / inner swept-10 — *the default until 2026-08-21* | 71.11 | **92.9×** | 128.68 | **158.6×** | **1.8×** |

  - **The `jvp/fwd` column is the finding.** A fixed-sweep solve differentiates by unrolling and costs
    **1.0×** to differentiate; the Krylov solve differentiates by the implicit function theorem, which
    solves an *entire second Schur system* per JVP — each with its own 10-sweep inner solve inside every
    one of its iterations — and costs **1.8×** on top of an already 2.8× dearer forward pass. Net: the
    swept outer is **5× cheaper on the JVP path** (158.6× → 31.5×) and reaches the same gradient to
    1e-12. **✅ ACTED ON 2026-08-21: the outer default is now `SweptGradientSolve(sweeps=20)`.** The
    count is 20 rather than the 10 profiled here because 20 is what holds quadratic exactness on the
    meshes the exactness tests use — departure from an exactly-solved reconstruction is `1.7e-15`
    (2D, 20 % perturbed), `7.9e-14` (2D, 30 %) and `1.9e-13` (3D hex, 25 %), against `1.2e-08` /
    `7.4e-08` / `1.5e-07` at outer 10. So the swap is **equal-accuracy**: the Krylov default solved to
    `rtol` 1e-10, and anything below 20 would have quietly made the scheme *less* exact while changing
    the solver. At 20 the comparison is 66.5× forward / **59.9× jvp** against the Krylov 92.9× / 158.6×
    — **2.6× cheaper on the path a march pays, at the same accuracy.** No exactness test needed
    re-pinning; four tests that read the *default* rather than naming their arms did, which is its own
    lesson.
  - **Cost is LINEAR in outer sweeps** (10 → 20 doubles it), and the inner solve dominates each one, so
    the remaining lever is the inner count: 10 buys machine-precision quadratic exactness where 4 gives
    1.3e-06, far below any discretization error a flow solve has. Not yet measured as a pair.
  - **The fixed overhead is 2.5× the whole baseline reconstruction** — geometry, the two per-cell block
    builds, and the right-hand side. ⚠️ Read the `outer swept-1` rows correctly: at `sweeps=1` the swept
    solver returns `P⁻¹b` and **never applies the operator** (the peeled first sweep), so the inner solve
    is never invoked and those two rows are identical by construction, not by measurement. That is what
    makes them a clean isolation of the overhead. The blocks are geometry-only and rebuilt every call;
    hoisting them is the shared refactor with `CorrectedGreenGauss.terms`, and this prices it.
  - **On a whole residual the ratio is much smaller, because a residual is not only gradients:** the full
    coupled residual is **5.0 ms** (1.0 GB peak RSS) under corrected Green–Gauss and **93.2 ms** (2.2 GB)
    under Betchen outer-swept-10 — **18.6×**. Note also that five reconstructions at 0.77 ms is ~85 % of
    the baseline residual's 4.4–5.0 ms, so **the gradient is most of a residual on this case** — an
    intuition that it is a small percentage does not hold here.
  - **⚠️ Memory is a live constraint, not a footnote.** The Betchen residual peaks at 2.2 GB against
    1.0 GB at 12225 cells, and a profiling process was killed running three such arms in sequence. The
    swept path unrolls outer × inner sweeps onto the tape; at 1.6M cells that scales, and it should be
    checked before the scheme is pointed at a large mesh.
  - ⚠️ **Two earlier attempts at this measurement were void and neither announced itself:** one ran two
    copies of the profile concurrently (two background waiters fired on the same event), and the
    march-derived "10–15× per step" that prompted it rested on a *single* step-to-step delta per arm,
    with step 1 contaminated by setup and compilation. Timings on this machine are only meaningful when
    it is running one job.

- **WHAT THE BETCHEN GRADIENT ACTUALLY BUYS AND COSTS ON A REAL CASE — first end-to-end A/B,
  2026-08-21 (`validation/pitzdaily_gradient_ab`).**
  ⚠️ **EVERY `pitzdaily_gradient_ab` FIGURE IN THIS FILE WAS MEASURED UNDER `local_schur_block=False`,
  WHICH IS NO LONGER THE DEFAULT (moved 2026-08-22).** The case pins the two sweep counts but not the
  preconditioner, so it now runs the Schur block instead. On a mild all-hexahedral 2D mesh the two
  blocks agree closely on synthetic fixtures, so these numbers are **expected** to carry — but that is
  an expectation, not a measurement, and nothing here has been re-run. Treat the cost ratios and cycle
  counts below as provisional until one march is repeated, and pin the flag explicitly in any arm that
  is meant to reproduce them.
  Matched marches, both arms at probe reach 5,
  Betchen at outer/inner swept-5, everything else `pitzdaily_openfoam`'s own configuration:

  | | wall | cycles | steps | max cyc | `x_r/h` |
  |---|---|---|---|---|---|
  | `CorrectedGreenGauss` | 711.1 s | 439 | 71 | 15 | 8.069 |
  | `HessianCorrectedGradient` swept 5/5 | 1603.5 s | 511 | 72 | 20 | 8.069 |
  | ratio | **2.25×** | **1.16×** | — | — | same cell |

  - **It does not move the judged quantity.** Both reattach at `x_r/h` 8.069 (reference 7.741). ⚠️ Read
    that carefully: `reattachment_length` returns the **cell-centre** x of the last reversed-flow cell,
    and cells there are ~0.076 h wide — so "identical" means *the same cell*, i.e. agreement to ~1%,
    not agreement to three decimals. The fields do differ (1–2 % in L2, up to 12 % locally in `nu_t`);
    that difference simply does not reach the reattachment length on a mesh ~6° off orthogonal at
    worst. **This is the measurement that says the cheap scheme suffices at this mesh quality.**
  - **⚠️ THE FIRST COST NUMBER WAS 3.25× AND IT WAS WRONG — the arms were at different probe reaches.**
    At reach 3 the Betchen arm needed 1.93× the standard arm's cycles, and that got charged to the
    reconstruction. Matched, the cycle ratio is 1.16× and the wall ratio is the reconstruction. The
    lesson generalizes past this case: **when a scheme change also changes the operator the
    preconditioner sees, a wall-clock ratio is only a scheme comparison once the cycle counts match** —
    which is why the harness prints the cycle ratio next to the wall ratio and says what to do about it.
  - **The probe is a large share of a step, and this prices it:** the standard arm goes 527 s → 711 s
    (+35 %) from reach 3 to reach 5 at essentially unchanged cycles (432 → 439), i.e. ~184 s of pure
    probe cost for 380 residual evaluations per refresh instead of 165.
  - ⚠️ **One run per arm**, on a machine whose per-application noise floor is ~15 %. Step and cycle
    counts are contention-immune and are what to quote; the seconds are single samples.
  - **⚠️ AND THE REASON THIS MATTERS BEYOND BETCHEN: the reconstruction is 12–20 % of a march step even
    for the CHEAP scheme.** With the reconstruction's own jvp cost measured at 0.84 ms
    (`CorrectedGreenGauss`) against 9.30 ms (Betchen swept 5/5) — a factor `k` = 11.11 — a step ratio
    `S` pins the share directly, since `k·g + (1−g) = S` gives `g = (S−1)/(k−1)`:

    | basis | `S` | gradient share, corrected Gauss | under Betchen 5/5 |
    |---|---|---|---|
    | matched, both at probe reach 5 | 2.25× | **12.4 %** | 61 % |
    | standard at reach 3 (cheaper probe) | 3.04× | **20.2 %** | 74 % |

    The share moves with the *rest* of the step (a dearer probe dilutes it), which is why it is quoted
    as a range with its basis rather than as one number. Either way, an eighth to a fifth of every step
    spent reconstructing a term that enters the residual as a **correction** is the standing argument
    for attacking reconstruction cost directly — the geometry-only terms rebuilt per call
    (`CorrectedGreenGauss.terms`, and Betchen's two per-cell blocks, priced at 2.5× the whole baseline
    reconstruction) being the first place to look, since nothing in the current design lets them be
    cached across calls.

- **⚠️⚠️ THE SWEEP COUNT IS THE COST LEVER, AND THE FIXED DEFAULT OF 4 IS WRONG BY UP TO FIVE ORDERS IN
  BOTH DIRECTIONS — measured 2026-08-21.** Two independent investigations converged on this after both
  ruled out the thing that looked more promising.
  - **The contraction rate varies ~50× across meshes.** `ρ(I − P⁻¹A_g)` is **0.0053** on `pitzDaily`,
    against **0.14** at 20 % grid perturbation and **0.26** at 30 %. A single static `sweeps=4` is
    therefore *five orders over-resolved* on the first and under-resolved on the last. Cold relative
    error on pitzDaily at `k` = 1/2/3/4: `4.0e-3 / 1.4e-5 / 5.2e-8 / 1.6e-10`.
  - **On pitzDaily `sweeps=2` costs ONE operator apply** (the first sweep's apply is peeled) and leaves
    `1.4e-05` relative gradient error — three orders below the 1–2 % field difference between the two
    *schemes*. Corroborated independently: `schemes.md`'s own recorded reach-3 Jacobian error of
    1.99e-07 at `sweeps=4` is `ρ³` to one digit.
  - **It compounds, because the sweep count sets the residual's Jacobian reach** — but ⚠️ **NOT by
    `sweeps + 1` for every column, and a first version of this entry said `sweeps=2` collapses the reach
    from 5 to 3, which is WRONG.** Measured directly (coloured probe against the true jvp, per column
    field, at `hybrid_initialize` on pitzDaily):

    | sweeps | reach 3 (165 probes) | reach 4 (265) | reach 5 (380) |
    |---|---|---|---|
    | **4 (shipped)** | 2.45e-07 (**p column 4.0e-07**) | 4.66e-10 | **9.5e-16** ← needs reach 5 |
    | **2** | 1.39e-09 (p column 2.1e-16) | **2.6e-16** ← needs reach 4 | 2.6e-16 |
    | **1** | **2.6e-16** ← needs reach 3 | 2.6e-16 | 2.6e-16 |

    The **pressure** column does track `sweeps + 1` exactly (float64-exact at reach 5/3/2 for sweeps
    4/2/1) — it is the gradient-carried column. The **`u`/`v`** columns carry a further ring from the
    eddy viscosity's strain-rate dependence, which is *not* the gradient's, so they reach `sweeps + 2`.
    So `sweeps=2` needs reach **4**, worth ~14 % of the march rather than the ~35 % first claimed.
    (This run also independently reproduces this file's recorded 1.99e-07 at reach 3 — measured
    2.45e-07 — and its attribution to the pressure column.) At reach 3, `sweeps=2` leaves 1.4e-09 where
    `sweeps=4` leaves 2.45e-07, and 2e-07 is the value recorded as *breaking* the incomplete-LU bundle
    — so reach 3 may yet be safe at `sweeps=2`, but that needs a march and has not been run.
  - **The fix is to stop guessing it: estimate `ρ` once at case-build time** (a few power iterations on
    the geometry-only iteration matrix, which is constant) and set `sweeps = ceil(log(tol)/log(ρ))`.
    That gives the adaptivity a runtime convergence test would, at zero per-call cost, with no
    `lax.while_loop` (which JAX cannot reverse-differentiate) and no implicit-diff tangent — and it
    subsumes `validation/uvreactor_openfoam/gradient_sweep_calibration.py`, which answers the same
    question by hand. ⚠️ **The caution that stood here — that a global `ρ` would size the count for the
    worst cell and so over-resolve the bulk — is REFUTED, and the error runs the OTHER way; see the
    measured L2-versus-worst-cell entry below.**
  - ⚠️ **What this does NOT license:** lowering the shipped default without per-mesh calibration. The
    same count that is wasteful on pitzDaily is *insufficient* at 30 % skew, and a fixed sweep fails
    **silently** — there is no residual test to trip.


## Build-time sweep calibration, the `bind` proposal, and probe narrowing (2026-08-21)

  - ⚠️ **`tools/sibling_builders.py` COULD NOT SEE THIS PAIR, and that was fixed in the same change.**
    It recognized only `build`/`create`/`make`/`from_*` as factory methods, and credited construction
    by capitalized-name convention — so a `@classmethod` returning `cls(...)` looked like it
    constructed *nothing* and dropped out of the report entirely. Not a quiet pair: **no pair at all**,
    which reads exactly like a clean tree. It now knows `calibrated` and credits `cls(...)` to its
    owning class, pinned by `test_it_reaches_classmethod_factories_that_return_cls`. With that, the
    pair reports with `schur` as the only difference — a genuine property of the scheme with two
    modes. A signature-parity test derives the shared surface from `dataclasses.fields(
    SweepCalibration)`, so adding a setting and wiring it into one factory fails there.

- **Per-case calibrated counts, measured on the meshes in this repository (2026-08-21):**

  | case mesh | cells / dim | `ρ` | `k` at 1e-4 | `k` at 1e-10 | ships |
  |---|---|---|---|---|---|
  | `pitzdaily_openfoam/of_case` | 12225 / 2D | 5.07e-03 | **2** | 5 | 4 |
  | `bfs3d` (read from `bfs3d_species/of_case`) | 23040 / 3D | **5.06e-15** | **1** | **1** | 4 |

  ⚠️ `bfs3d_openfoam/of_case/constant/polyMesh` is generated by running OpenFOAM and is **absent from a
  fresh checkout**, which is why the row above names the species case's copy — the same geometry (23040
  cells, 66368 interior faces, matching this file's own reach table), and the one actually in the
  repository. A harness that reads the flagship path will skip rather than measure.

  `bfs3d`'s mesh is a graded but orthogonal block, so the skewness correction is essentially absent and
  the first sweep is already at machine precision — the shipped `4` spends three operator applies per
  reconstruction, on every residual evaluation and every Jacobian--vector product, to no effect. It
  corroborates the reach table at the top of this file (`0 of 66368` faces above the skewness
  threshold). **Neither default was changed**: the sweep count sets the residual's Jacobian reach,
  which each case's probing reach is matched to, so moving it is a change to the discretization that
  the case's reattachment result would have to be re-validated against. Both numbers are recorded in
  their case READMEs beside the mesh.

- **PROPOSED, NOT BUILT — `GradientScheme.bind(mesh, geometry) -> BoundGradient`.**
  - **The seam.** `gradients()` takes `mesh` and `geometry` on every call and rebuilds every
    geometry-only term inside them. `bind` would split the *choice* (the injected, mesh-free
    `GradientScheme`) from the *derived product* (arrays, no mesh argument), exactly as `mesh.geometry()`
    already does and as `MomentumContinuity.build` already does for `interp_factor`/`normal_distance` —
    the gradient scheme is the one thing that opted out. `gradients()` then loses both parameters, which
    is the real payoff: today you can hand it a different mesh than the assembler holds and get a
    silently wrong answer. Three call sites change. ⚠️ **Do not sell it as performance** — XLA already
    CSEs the shared geometry, so it is worth ~2 % of a residual and ~0.4 % of a jvp; the Betchen prologue
    is the only piece large enough to matter (~4.7 % of a march step).
  - **⚠️ Binding OUTSIDE the differentiated region silently kills mesh-shape gradients** —
    `‖d(objective)/d(node_coords)‖ = 0.0`, no error. Binding inside is bit-identical to today. Not a new
    hazard (the same rule already governs `mesh.geometry()`), but now measured.
  - **Calibration was expected to need this seam and did not** — it shipped as a classmethod factory
    (above). `bind` would still be a natural home for it if it is ever built, but it is no longer a
    reason to build it.
  - **Betchen's own calibrated counts, measured:** outer `ρ` = 0.159 (pitzDaily) to 0.238 (30 %
    perturbed) — **barely moving with mesh quality**, which corroborates that the outer Schur system's
    difficulty is intra-cell gradient–Hessian coupling rather than skewness, and means the outer count is
    nearly mesh-independent. Inner `ρ` = 0.0074 to 0.091. At `tol=1e-4` that is outer **6**, inner **2**.
    ⚠️ Read against the shipped 20/10, which target ~1e-10 (the accuracy the Krylov default delivered),
    not 1e-4 — the two are answering different questions, and the shipped pair is not "20 against 6".
  - **The inner truncation sets a floor no number of outer sweeps removes** (pitzDaily: inner 1 → 4.7e-08,
    2 → 7.2e-11, 3 → 5.1e-13), but it is attenuated into the gradient by 1e-5 to 1e-3, so calibrating both
    at the same tolerance carries three orders of margin. Measured on two meshes only — check the
    composition per mesh rather than assuming it.
  - ⚠️ **Distributed:** the estimator norms the whole local vector, which double-counts ghost rows.
    Calibrate on the global mesh **before** partitioning, or supply an owned-only reduction.

- **✅ NARROWING THE PROBE BEATS LENGTHENING IT — measured 2026-08-21, and it is the largest single
  saving found on this case.** The long reach a long-stencil reconstruction seems to demand is paying
  to *tolerate folding*, not to capture coupling the preconditioner needs. Cap the gradient's sweeps
  **for the probe copy only** (`coupled_jacobian_probe(gradient_sweeps=…)` /
  `narrow_gradient_sweeps`) and the residual's stencil genuinely shortens, so the colouring is
  collision-free and the recovered matrix is **exact for the narrowed residual** instead of corrupted
  for the true one. Full `pitzdaily_gradient_ab` marches, Betchen arm at outer/inner swept-5:

  | probe | probes | standard | betchen | betchen/standard |
  |---|---|---|---|---|
  | reach 5, full sweeps | 380 | 711 s / 439 cyc | 1604 s / 511 cyc | 2.25× |
  | reach 3, full sweeps | 165 | 527 s / 432 cyc | 1715 s / 834 cyc | 3.25× |
  | **reach 3, `gradient_sweeps=1`** | **165** | **473 s / 421 cyc** | **918 s / 451 cyc** | **1.94×** |

  All six marches reach `x_r/h` 8.069.

  - **1.75× faster than the reach-5 arm, at 43 % of its probe cost, with FEWER cycles** (451 against
    511) and the identical root. The expectation was that narrowing would merely *recover* reach-5
    quality; it beats it. So the far entries were not helping the aggregation — a sparser, cleanly
    probed operator is both a better preconditioner and a cheaper one to build and apply.
  - **It is NOT a Betchen-specific fix — the standard scheme gains 1.50×** (711 → 473 s, cycles 439 →
    421). The benchmark next door ships reach 5 with full sweeps and is leaving that on the table.
    ⚠️ But do not assume it transfers: this case runs a SIMPLE-smoothed field split, and the benchmark
    runs an **incomplete-LU** smoother, which is the family folding hurts most (folded entries become
    pivots) *and* the family most exposed to the narrowed matrix being a cruder approximation. Both
    effects point opposite ways; it has to be measured there, not inferred.
  - **The matched cost of the Betchen scheme is 1.94×, and this is the trustworthy version of that
    number** — its cycle ratio is **1.07×**, against 1.16× at reach 5 and 1.93× at reach 3 unnarrowed.
    The figure has now read 3.25× / 2.25× / 1.94× across the session, and every correction came from
    removing a probe artifact rather than from anything about the reconstruction.
  - **The middle row is the cleanest demonstration of why the stencil matters.** Same probe, same mesh:
    the standard arm is untouched by the short reach (432 cycles, essentially its reach-5 value) while
    the Betchen arm degrades to 834. That is also why this went unnoticed — with the corrected gradient
    the folded mass on a mesh this mild is negligible, so the probe reach looks like a free parameter.
  - **The Krylov matvec keeps the exact jvp of the full residual**, so the converged state and its
    adjoint are untouched — only the preconditioner's materialize sees the narrowed copy. That is what
    makes this a free lunch rather than a discretization change.
  - **It is only reachable because the Betchen outer solver is now a fixed sweep.** A Krylov outer
    cannot be narrowed — `narrow_gradient_sweeps` rewrites `SweptGradientSolve` nodes and would return
    the tree unchanged, silently. At `sweeps=1` the swept solver returns `P⁻¹b` without ever applying
    its operator, so *both* Betchen systems collapse to a one-ring stencil.
  - **The floor is reach 3, not 1**: the coupled `(u,p)` Jacobian is intrinsically distance-2 (Rhie–Chow
    damping couples pressure to the neighbour-of-neighbour ring), and no narrowing removes that.
  - ⚠️ **One run per configuration.** Cycle and step counts are contention-immune and carry the verdict;
    the seconds are single samples. **Not yet measured:** the standard arm with the same narrowing (its
    reach-3 full-sweep probe already works, folding being negligible at this mesh's skew, so the gain
    there should be smaller); and whether this holds on a genuinely skewed mesh, where the folded mass
    is larger and the effect should be *stronger*.


## `narrow_gradient_sweeps`: latent on bfs3d, live on pitzDaily (2026-08-16)

    ⚠️⚠️ **IT IS LATENT ON `bfs3d` AND LIVE ON pitzDaily — an earlier version of this entry said "latent
    on every case shipped today" and that is FALSE (corrected 2026-08-16).** The two shipped cases run
    *identical schemes* and differ only in the mesh, and only `bfs3d` is skew-free:

    | mesh | `|skew|/d` median | max | interior faces > 1e-6 |
    |---|---|---|---|
    | `bfs3d` | 7.0e-15 | 1.9e-12 | **0 of 66368** |
    | **pitzDaily** | 2.2e-09 | **7.5e-02** | **11567 of 24170** |

    pitzDaily's distribution is bimodal — most of it is a structured block at round-off, and the slanted
    lower wall and the contraction carry the tail, whose **maximum skew exceeds a 5 %-perturbed synthetic
    grid's**. Measured consequence (a `pitzDaily` session, at the `hybrid_initialize` seed):
    `jacobian_relative_error` against the true matrix-free jvp is **1.99e-07 at reach 3** and only reaches
    the float64 floor at **reach 5** (1.48e-15), where `bfs3d` is already at 2.34e-16 at reach 3 — and
    pitzDaily at `sweeps=1` floors at reach 3 exactly as `bfs3d` does, which is what ties the difference to
    the sweeps rather than to anything else. **So running pitzDaily at the shipped `stencil_reach=3` costs
    a real, measurable error today; the trap is not hypothetical.** The shortfall is carried almost entirely
    by the **pressure column**, which enters the residual only through gradients and so inherits the
    sweep-extended stencil undiluted — read `jacobian_relative_error` per (row field, column field), since
    one random vector under a global norm cannot see it.
    ⚠️ `validation/pitzdaily_openfoam/compare.py`'s own docstring still says this mesh is "only mildly
    non-orthogonal ... reaches the converged corrected-gradient to machine precision in the default few
    sweeps". Re-adjudicate that before quoting it.
