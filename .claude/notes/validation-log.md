# Investigation log — `validation/` (the scientific cases and study harnesses)

> Split out of `.claude/rules/validation.md` (2026-10-17). Lives in `.claude/notes/`, so it never auto-loads. It holds the dated record — measurements with their configuration, investigation narratives, mutation-pass lists, test inventories — behind the current-status entries in `validation.md`. Moved verbatim: names in it may since have been renamed or deleted; check the rule and the code before acting on any of it.

## From `.claude/rules/validation.md` — the recorded failure (2026-08-16)

## ⚠️ THE RECORDED FAILURE — one case, three simultaneous breaks, suite green

Found 2026-08-16, all in `validation/pitzdaily_openfoam/compare.py`, none detected by anything:

1. **A settings object was introduced and the case was not updated.** `RefreshPolicy` replaced four
   loose keyword arguments; the case still passed a bare `refresh_preconditioner` callable, so
   `solve_coupled` asked a function for `observes` and raised **before the first step**, under every
   configuration and every preconditioner.
2. **A guard was tightened onto an object the case does not hand it.** The escalation gate began
   validating the base step rather than the control's output, and refused to start with a `TypeError`
   whose own message named `DualTimeStep` as acceptable while rejecting one.
3. **The case had no `sys.path` bootstrap**, so it could not be launched through
   `validation/run_case.sh` at all. Its sibling has had one for as long as it has existed.

And beyond outright breakage, the same case had silently fallen a long way **behind**: it still ran a
single-step pseudo-transient march with no dual-time inner loop, no Courant control, no retry ladder
and no per-step log, while all of that was built and calibrated on the other case. Under that
configuration this case is a documented reachability crawl — on the order of eight hundred outer steps
to develop the recirculation, against a two-hundred step cap — so it could not converge however long
it was left, and any timing taken from it measured the globalization rather than the thing under study.

## From `.claude/rules/validation.md` — pitzDaily's shipped preconditioner stopped marching it (2026-08-22)

## ⚠️ pitzDaily's SHIPPED PRECONDITIONER STOPPED MARCHING IT — the default moved to `simplesmooth` (2026-08-22)

Under `PITZ_FLOW_INVERSE=petsc` (the case's default until this date; the arm was removed 2026-09-13, #371) `pitzdaily_openfoam/compare.py`
collapses at the **first step of the second Reynolds rung**: `alpha` 0, `beta` escalating 0.5 → 2 → 16
through the whole ladder, the residual rising 1.674e-01 → 5.754e-01 → `inf`. Reproduced three times,
including on a tree carrying **no local change at all**, with the step tables bit-identical.

**The default is now `simplesmooth`** — the same leading inverse the 3D sibling defaults to, and for
the same reason: an incomplete factorization's behaviour on this saddle depends on the elimination
**order** and the **fill**, neither of which is predictable in advance, while a SIMPLE-smoothed
hierarchy never eliminates the matrix at all. This case's own record already carries the extreme
version of that sensitivity — a zero-fill factorization going from *amplifying* a residual 5.5× per
sweep to contracting it, on nothing but a reordering. `petsc` and `hostilu` were both
removed 2026-09-13 (#371); their recorded measurements stand as history.

**What was ruled out before reaching for a preconditioner swap**, because "the case broke" invites
blaming whatever merged most recently:

- **Not the four merges of that day.** `|R|` at a fixed saved state is identical to **twelve digits**
  across `1c3c874 → 2829965 → eeb5c60 → f05eafb → 2c6ffae`, the last digit moving only at the
  `lax.scan` change that is recorded as not bit-identical. The discretization did not move.
- **Not the scan specifically.** Reverting it to the Python loop reproduces the collapse unchanged
  (1.679e-01 against 1.674e-01).
- **`simplesmooth` marches the same case to the same answer** — `x_r/h` 8.0686, `ux` 0.0191, 404
  cycles, 711 s — against the last good `petsc` run's 8.0686, 0.0191, 743 s. That agreement is what
  makes this a swap of preconditioner rather than of result.

⚠️ **WHEN it started is NOT going to be chased, and that is a decision rather than a gap.** The last
good `petsc` run is 2026-08-17 and the failure is present at every commit from 2026-08-21 onward, so it
entered somewhere in the ~30 commits between — a bisect of march-hours. **Incomplete-LU preconditioning
is no longer a direction this project is taking** (2026-08-22, project owner), precisely because its
behaviour depends on the elimination order and the fill in ways that are hard to predict, so the bisect
would buy a diagnosis of an arm nothing selects. What the reader needs instead is the consequence:
**any `petsc`-bundle number recorded for this case is unreproducible on the current tree** — do not
re-measure against one, and do not read the collapse as evidence about anything other than that
preconditioner family.

⚠️ **The trailing `[k, omega]` inverse was NOT aligned with the sibling in the same change.** This case
runs `{max_coarse: 2000, equilibrate: False}` where the 3D one runs `{max_levels: 20, max_coarse: 200,
strength_threshold: 0.25, aggressive_levels: 0, frozen_coarsening: True}`. That is a genuine open
question rather than an oversight — the two meshes coarsen at very different rates and this case has
never been measured at the sibling's settings — but it is the next thing to try if the alignment is
carried further.

**The leading `SimpleSmoothed` block now sets `frozen_coarsening: true` (2026-10-12, #290, the project
owner's choice).** At `strength_threshold` 0.25 its coarsening reads values, so unfrozen it re-coarsened
between checkpoints and each refit would recompile the on-device split. Frozen, the march takes 20 steps
/ 154 restart cycles against the unfrozen 20 / 140, to the same `x_r/h` 8.0686 (`solve-field-split.md`).

## From `.claude/rules/validation.md` — what the static guard cannot catch (full history)

## ⚠️ What the static guard CANNOT catch

`tests/unit/test_validation_api.py` is a static check — it reads the cases with `ast` and never builds
a mesh. **A green run there does not mean the cases work.**

✅ **It now also checks the bootstrap (2026-09-24):** every script with a `__main__` entry point that
imports `aquaflux` must put the repository root on `sys.path` *before* that import
(`test_every_runnable_case_can_import_the_package_it_runs`). Four scripts failed it —
`turbulent_channel/compare.py`, `turbulent_channel_openfoam/compare.py`, `continuation_seed_error.py`,
`point_implicit_step.py` — so **none of them could be launched through `run_case.sh`**, with every check
here green, although this module's own docstring names a missing bootstrap as a motivating break. Found by
running a channel study through the runner, not by the guard. All four are fixed.

It is still blind to:

- **A method that no longer exists on an object the case holds.** `turbulent_channel_openfoam/compare.py`
  called `momentum.velocity_gradient(flow)` after that method became `velocity_fields(flow).gradient`, so
  its OpenFOAM comparison raised `AttributeError` *after* every solve — found 2026-09-24 by running it.
  The guard checks imported names and keyword arguments, never attribute access on a returned object.

- **Semantic breaks.** A parameter that still exists and now means something different, a default that
  moved, a type that changed under an unchanged name. Break 1 above is exactly this, and the guard
  would **not** have caught it.
- **Anything behind `**kwargs`.** `solve_coupled` takes `**strategy_kwargs` and forwards them to
  whichever builder it is given, so every keyword is "accepted" *statically* and none is checked here —
  and this is the main entry point. ✅ **Narrowed 2026-08-20 (#278): the case where there is no builder
  to forward to is now a `TypeError` at the call rather than silence.** Given an explicit `strategy`
  or a `RefreshPolicy(builder=...)`, `preconditioner` / `reference_state` / `**strategy_kwargs` are refused
  by name instead of dropped — which is how `refresh_preconditioner=` used to vanish on its way to a
  `RefreshPolicy`. What is still unchecked is a keyword that *does* reach the default builder and is
  wrong there; that raises at run time, in the builder, not here.
- **A call on an object the case built itself.** `Class.method(kw=...)` on an *imported* name is
  resolvable and is checked (see below); `instance.method(kw=...)` is not, because the instance's type
  is not knowable statically. Most of what a case does after construction is this shape.
- **Behaviour.** It cannot tell a converging march from one that crawls, which is how a case can be
  runnable and useless at the same time (the "fallen behind" failure above).

⚠️ **Until 2026-08-21 it was also blind to every constructor call in every case**, because it resolved
only *bare* imported names and the cases build their assemblers through class methods — so
`MomentumContinuity.build(...)`, `SSTTurbulence.build(...)` and `CoupledRANS.build(...)` were
unchecked, and those carry most of the keywords a case passes. Measured at the time: 60 such call
sites across `validation/`, 28 of them carrying keywords, none of them seen. The guard now resolves
that form too, and `test_the_checker_reaches_a_call_on_an_imported_CLASS_not_only_a_bare_name` pins
that it does — the coverage gap sat exactly where the cases spend their configuration, and looked
identical to coverage that worked.

⚠️ **And until 2026-08-25 nothing anywhere could see a case whose DEFAULT configuration does not start.**
A case settings module configures itself by branching — one block per arm, each binding its own per-arm
name — and then reads the arms back through a branch of its own. When a rename moves one arm and not the
other's reader, the surviving branch names a global only the *other* arm binds, and the case dies with a
`NameError` in the one line whose job is to say what the run is. Three things are quiet about it at once:
the name **is** bound at module level, so `ruff`'s undefined-name rule is correct to say nothing; the read
sits in a function body, so even importing the module would not reach it; and no tier runs these files.
That is how `bfs3d_openfoam/compare.py` spent five days unable to start at its own default —
`FLOW_INVERSE == 'native'` survived the `native` → `simplesmooth` rename in the banner ternary alone
(`a2ed044`), so the default arm reached `_HOST_FLOW`, which only the `hostilu` arm defines. The guard now
reports this shape (`test_the_cases_do_not_read_a_global_that_only_one_branch_binds`), and
`test_the_branch_checker_separates_a_real_hazard_from_the_idioms_around_it` pins both directions, because
the idioms it must stay quiet about — a name bound by both arms, an arm that raises instead of binding, a
helper defined inside the branch that binds it — are what a naive version of this check drowns in.
**The structural fix is the one to prefer over the guard**: have each arm record its settings under ONE
shared name (`LEADING_SETTINGS`) beside the object it built, so the reader never branches and the hazard
cannot be written. The guard is what catches the next module that does branch.

## From `.claude/rules/validation.md` — known API gaps these cases exposed

## Known API gaps these cases exposed

- **The k-positivity guard was wired on one builder of four until 2026-08-19 (FIXED).** All builders now
  route through `_coupled_step`, which installs the per-cell k clip + ω step bound
  (`_closure_step_projection`; the global cap `positive_k_limit` and the `positivity_floor` /
  `positivity_projection` keywords, with their `K_POSITIVITY_*` / `POSITIVITY_PROJECTION` harness
  constants and env vars, were deleted 2026-10-14 — the one knob left is `omega_step_ratio`, overridden by
  `BFS3D_OMEGA_STEP_RATIO` / `PITZ_OMEGA_STEP_RATIO`, `none` = unbounded);
  `test_every_continuation_builder_installs_the_same_globalization` fails if one loses it. The confound —
  a cross-builder comparison between a guarded march and an unguarded one — still applies to archived
  runs before 2026-08-19.
- **✅ FIXED 2026-08-20 (#282) — the SAME defect recurred TWICE MORE on the same four builders, with the
  shared tail already extracted.** `forward_rtol` / `forward_restart` / `forward_max_restarts` were on
  `coupled_amg_continuation` alone, so the *default* path — what `solve_coupled` builds when nothing is
  passed — stopped its forward solve on the plain 2-norm that same builder's docstring calls effectively
  blind to the flow block. And `velocity_shift_parts` was on the two block builders only, absent from
  exactly the monolithic path it was written for. Both are now on all four, and the stopping *measure*
  belongs to `_coupled_step` rather than to any builder: it is the march's own progress measure, so a
  solve cannot converge in a quantity the march does not read. ⚠️ The advice recorded here — that
  `forward_rtol`/`restart`/`max_restarts` were "reachable via `krylov_solver=`" — was true and a trap:
  building a solver to move the tolerance silently replaced the *stopping measure* too, a far larger
  change. Pass the parameter — since #388 that is `linear_solve=LinearSolveSettings(...)`, and a whole solver goes
  in the same `linear_solve` slot, so the two cannot be given together.
- **`_mis_aggregate`'s return annotation is stale** — it says `tuple[np.ndarray, int]` and returns
  three values (labels, roots, count). Cost one debugging cycle.

## From `.claude/rules/validation.md` — traps when writing a harness (full text)

## ⚠️ Traps when writing a harness (each one produced a wrong result here)

- **⚠️ A LOG PARSER KEYED ON THE TABLE'S SHAPE RETURNS NOTHING, SILENTLY, WHEN THE LOGGER GAINS A
  COLUMN.** `bfs3d_openfoam/march_log_compare.py` matched the step row by its eight columns; the march
  log gained `rho1` on 2026-10-11 (`af98897`), and for three days the script summarized every new log as
  zero steps -- no error, a well-formed table of zeros (found 2026-10-14 when it was asked for a study
  table). `step_policy_replay.py` reads logs through the same parser. The column is now optional, and the
  script refuses a log with no step rows. Any reader of the march log should do both: accept the old
  and the new shape, and treat "found nothing" as an error rather than as an answer.
- **Build the reference state with `state_from_physical`, NOT `pack_state`.** `pack_state` takes the
  **solved** variables; a case transporting `log(omega)` differs from the physical fields by an
  exponential. Packing physical omega where a log is expected exponentiates ~1e5, and the residual is
  **silently NaN** while the state still reads finite. Every factorization then fails in its own idiom
  — "out of memory", "exactly singular", "SVD did not converge" — and each invites a confident and
  completely wrong story about the method. This happened, and a whole comparison had to be withdrawn;
  a recorded mechanism was even fitted to it (`_cell_graph`'s field-scale problem), because NaN
  comparisons return False and so *look* exactly like a threshold rejecting every edge.
  **Gate it: assert the starting residual is finite before measuring anything.**
- **Do not copy a wiring idiom from a test without checking the case matches.** The `pack_state` error
  above came from `tests/integration/test_coupled_lu.py` (deleted 2026-10-11; its channel fixture lives in
  `test_coupled_field_split.py` now), where it is correct — that fixture builds `CoupledRANS` with no
  transform.
- **⚠️ A PIVOT CENSUS MUST READ THE FACTOR, NOT THE OPERATOR HANDED TO IT.** Every consumer here
  symmetrically equilibrates before factorizing, which forces the *operator's* diagonal to magnitude
  exactly 1 — so a census written as `matrix.diagonal()` reports "zero negative pivots, min |pivot|
  1.00" for every arm at every shift, including arms whose sweep diverges by 1e+59. It looks like a
  finding ("the pivots are all healthy, so it is not a pivot problem") and it is a measurement of the
  conditioning transform. This shipped in a sweep on 2026-08-17 and a conclusion was drawn from it
  before being retracted. Use the factorization's own stored pivots (`Ilu0.pivots` did this until it was deleted, #371); and note
  a host factorization may store the pivot itself where PETSc stores its **reciprocal**, so a census ported between the two reports the inverse
  of what it claims.
- **⚠️ A HARNESS THAT SPAWNS A CHILD MUST FORWARD `kill` TO IT — use `validation/peak_footprint.py`,
  never a bare `subprocess.run`.** Harnesses that read a peak memory footprint re-execute themselves under
  `/usr/bin/time -l`. Launched with `subprocess.run`, a plain `kill <pid>` of the parent (what
  `run_case.sh` documents, and what its wrapper forwards) killed **only the parent**: the `/usr/bin/time`
  child and its Python grandchild were reparented to launchd and ran on holding several GB, **invisible to
  `run_case.sh --status`** because the run-file's pid was dead. Observed 2026-09-24 with
  `sozzi_radiation/model_at_mesh_scale.py`. `run_with_footprint` starts the child in its own session and
  forwards SIGTERM/SIGHUP/SIGINT to the whole group (SIGINT because a new session also leaves the
  terminal's foreground group, so Ctrl-C would otherwise stop reaching it), then dies by the same signal;
  it returns the exit status and the parsed `peak memory footprint`. Verified by killing a parent over a
  `sleep` child: no survivors, exit 143; with the forward disabled both processes survive. **`kill -9`
  still orphans** (uncatchable) — stop runs with `kill`. After any interrupted run, check
  `pgrep -fl "usr/bin/time"` rather than trusting `--status`.
- **⚠️ A TIMER THAT WAITS ON ITS OWN RESULT ALSO WAITS FOR EARLIER, UNTIMED WORK.** JAX dispatches
  asynchronously, so a wrapper that calls `jax.block_until_ready` on a piece's output before stopping
  its clock is charged whatever was dispatched before it and is still running. In
  `sozzi_radiation/field_cost_breakdown.py` the facet shadow mask read **30 s** — it takes **0.2 s**; the
  time was the transfer's row blocks, returned un-materialized just before it (2026-09-26, caught by a
  standalone probe of the mask). A timer breakdown is only sound where the work before each timed piece
  is timed too; when a small piece reads large, time it alone before believing it.
- **⚠️ ON LINUX, A STANDALONE CALL OF A PROGRAM WHOSE TEMPORARY BUFFER EXCEEDS 32 MiB PAGE-FAULTS ALL OF
  IT, EVERY CALL.** glibc's adaptive mmap threshold stops at 32 MiB, so such a buffer is mapped and
  unmapped per call. pitzDaily's JVP (42.8 MiB) takes ~11 000 minor faults per call, 16.5 ms against
  9.8–11.1 ms with the threshold raised. Its residual (22.6 MiB) takes none. The march does not pay
  this (its wall is unchanged with the threshold raised), so a per-call timing taken this way
  overstates the in-march cost (2026-10-17, `.claude/rules/discretization.md`, "Single-threaded
  kernels"). Report `resource.getrusage(...).ru_minflt` per call beside any such timing, or run with
  `MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=2147483647` and say so. ⚠️ Through `mallopt`
  both are C `int`s: a 4 GiB trim threshold wraps to 0 and multiplies the faults instead. ⚠️ (2026-10-18)
  The raised thresholds stop at 64 MiB for a buffer allocated on an XLA worker thread (a non-main arena's
  heap is at most 64 MiB, so a larger request is always `mmap`ed): the composite `(J + s) M v` program's
  ~74 MiB buffer kept its 19 145 faults per call with them set, and lost them under jemalloc or
  `MALLOC_ARENA_MAX=1` (`solve-open-directions.md`, "Inside the step program (2026-10-18)").
- **A harness that builds a step must build it with the case's OWN shift settings, and must form the
  shift through `ShiftTerm.shift(beta)`, never `beta * term.diagonal`.** `closure_stall_probe.py` did
  neither until 2026-10-12: it built its engine, capture and march without `turbulence_damping`, and
  pitzDaily damps the k and omega rows at 3, so every k-row number it produced (#312's included)
  described a march the case does not run. `beta * term.diagonal` drops `row_shift_scale`, which is where
  that damping lives, so even an engine built with the setting would have been measured without it.
- **A harness that applies the march's preconditioner must apply what the STEP holds, and take it as an
  argument of anything it compiles.** Since #290 a shipped field split is applied as a traced copy
  (`TracedSplitShiftPolicy.split`) taken at each fit; a refit re-fits the session's host object in place
  and reaches the step only through the replacement `refresh_preconditioner` returns. `pitzdaily_openfoam/
  replay.py` still read `policy.preconditioner` (a host-policy field) and so raised `AttributeError` on
  construction, which broke every probe built on it (`batch_bound_probe.py`, `krylov_recycling_probe.py`,
  `precision_replay_probe.py`, `csr_kernel_precision.py`, `endgame_shift_probe.py`) with nothing
  reporting it. Repaired 2026-10-14: `MarchReplay.refit` applies the returned replacement as the march
  does, and `apply_pc` / `solve` / `preconditioned` compile with the step's policy as an argument, since
  a split captured in a closure keeps applying the first fit's hierarchies after every refit, silently.
  ⚠️ **Since #645 (2026-10-16) the march solves a step after a setback to 0.1, not the regime's 0.3**
  (`AdaptiveLinearTolerance`, the library default). A replay at a fixed 0.3 read 27 of 50 solves 1-2
  cycles low (found 2026-10-17). `System.rtol` now carries the tolerance the march's rule gave that step,
  rebuilt from the checkpointed reports (`MarchReplay.tolerance`), and `march_solver(measure, rtol)`
  applies it; with it all 50 solves of a `5661273a` capture reproduce. `krylov_recycling_probe.py`,
  `precision_replay_probe.py` and `batch_bound_probe.py` still build fixed-tolerance solvers.
  `MarchReplay(host=True)` applies the host object instead, for a probe that changes the host inverse in
  place (`precision_replay_probe.py` swaps its blocks' precision; the traced copy would not see that).
  Checked on a capture of `main` `da37958a`'s shipped pitzDaily march (18 steps, 4-core Linux, jax
  0.11.2): all 68 inner solves, steps 1-18 with every station refit, reproduce the recorded counts
  (`restart_cycles` of the replayed raw count; the inner dump stores the corrected count).
  `pitzdaily_openfoam/zero_shift_arms.py` and `bfs3d_openfoam/zero_shift_adjoint.py` read
  `engine.shift_policy.preconditioner.inverse` the same way, inside a `try` that reports a failure as an
  arm's result, so they printed `BUILD FAILED: AttributeError` as if it were a measurement. They now
  build through `open_session` and read `session.preconditioner.inverse`, the host object whose
  `apply(..., transpose=True)` they need (pitzDaily smoke test at a self-start: finite both ways,
  `w.Mv = v.M^T w` to 3.5e-14).
- **A harness that calls `solve_linear` outside a Newton step must bind the measure itself**
  (`solve.linear.in_progress_measure(engine.linear_solver(), measure)`). Since #370 the step's GMRES is
  built with `norm=None` ("the measure of the step that runs me") and raises when run anywhere else;
  `closure_stall_probe.py`'s `analyze` mode was broken on main by exactly this until 2026-10-12, and
  nothing reported it because no tier drives it.
- **Print one line per outer step, flushed.** A harness that collects reports and prints at the end is
  indistinguishable from a hung one, and cost thirty minutes of a run that could not have converged.
- **State the operating point before measuring.** A harness whose banner prints `? cells` is one whose
  author does not know what it is measuring; the mesh size decided the whole question in that instance.
- **A setting the banner prints must be a setting that is in force.** Printing an intended value that
  the builder never received is worse than printing nothing.
- **⚠️ Do NOT gate a loaded checkpoint on its own recorded `residual_norm`.** That number is whatever
  measure the march was *steered* by, and both cases march in the row-scaled measure (the default since #370) — a row-equilibrated
  norm, not a Euclidean one. Comparing the two rejects a perfectly good state: `bfs3d`'s `state-00069`
  records `2.64e-06` and computes `1.04e-03` under `jnp.linalg.norm`, a factor of **395** that is
  entirely the change of measure. Gate against the case's **own self-start** in whichever single norm
  the harness uses — both ends then move together, and a genuine configuration mismatch (which moves the
  residual by orders) still trips it.
- **A saved `.npz` is not necessarily a checkpoint.** A self-start cached only so repeated runs skip
  rebuilding it is not a converged root, and measuring "at the converged root" against it silently answers
  a different question: the zero-shift coupled Jacobian is nearly singular at a self-start and is not at a
  root. Read what wrote a state before trusting it.
- **⚠️ bfs3d's leading coarsening moves under ROUNDING, so a bfs3d march A/B of two checkouts that
  differ in rounding is not a controlled pair (found 2026-10-15, PR #721).** The `[u, v, w, p]` block
  coarsens at `strength_threshold` 0.25, reading values. #721 changed only how a sparse product is
  stored; that moved the hybrid start by ~1e-13 (its potential-flow solve runs multigrid cycles on such
  operators), and bfs3d's leading
  hierarchy by tens of aggregates (26252 against 26248 dofs at level 1, 284 against 236 at the bottom;
  applies 2.5 % apart), so the two marches differed from step 1 (`R` 0.1468 against 0.1474) and took
  137 against 129 cycles. Built from the SAME state, the two checkouts' hierarchies were identical and
  their applies agreed to 4e-16. pitzDaily's march happened not to move. **Any change that moves
  rounding anywhere upstream of the first fit does this** -- a different gradient inverse, a jax upgrade.
  Compare a preconditioner's cost at one fixed checkpoint
  (`bfs3d_openfoam/split_apply_timing.py` with `BFS3D_SPLIT_TIMING_STATE`; its level table must agree
  between the runs), and read a march pair's step and cycle counts as two samples, not as a difference.
- **A per-column probe reach is licensed by the scheme AND the split, so re-audit it when either changes.**
  `bfs3d` shipped `COLUMN_REACH = (3,3,3,3,2,2)` under `CorrectedGreenGauss`, licensed for the field split
  only (a flow-first split never applies `dR_flow/dturb`; a monolithic factorization does, and a short
  colouring folds far couplings onto near entries rather than dropping them). Since 2026-10-13 it ships
  `MultipleCorrectionGradient` at a **uniform reach of 3**: under that scheme `probe_reach_audit.py` finds
  p, k and omega carrying 1.1e-3 / 6.7e-5 / 3.9e-5 of their norm beyond reach 2, so no column may be
  shortened. Reach 4 is unaudited (the harness's widest pattern is 3).
  ⚠️ **"No column may be shortened" was an exact-recovery rule, and pitzDaily now breaks it on purpose
  (2026-10-14):** it probes every column at reach 2 inside its reach-3 pattern, folding ~4e-4 of the
  Jacobian's norm onto near entries, and its march is unchanged (18 steps, 123 restart cycles, `x_r/h`
  8.0686, as uniform reach 3 and reach 4) for a probe 40 % cheaper. The flow block is SIMPLE-smoothed and
  no factorization runs on that case; the folding that broke bfs3d's `p` column (#191) was a zero-fill
  factorization's. **bfs3d's answer is no** (2026-10-15, PR #722): the same colouring took it from 18
  steps / 137 cycles / 522 s to 21 / 216 / 697 s, same `x_r/h` 8.361, so its uniform reach 3 stands
  (`solve-direct-preconditioners.md` has the table).

## From `.claude/rules/validation.md` — case/tier collisions and what contention moves (2026-09-09/13)

## A validation case can collide with a test tier, or a test tier with another tier (both now closed)

**The runner's mutual exclusion was originally over *cases* only. `tools/fastgate.sh` is not a case, so
nothing stopped it starting on top of a running one** — and the fast tier is unambiguously a heavy job
on this machine. On 2026-09-09 that happened **twice in one evening**, from two different worktrees, one
minute and four minutes into someone else's case — and the two gates took **17:25** and **13:00** against
a documented 6:34-8:53 for that tier. Every wall-clock number in a case log overlapping such a window is
contaminated, and so was the gate's own. **`tools/fastgate.sh` now refuses to start while
`validation/run_case.sh --running` reports a live case** (exit `3`, `FASTGATE_FORCE=1` to override), so
this specific collision cannot recur — the numbers above stand as evidence for the cost of contention
and for why the guard exists, not as an open gap.

**⚠️ The "peaks around 6.4 GB" figure this incident was originally recorded with was RSS, and RSS is
not what a machine runs out of.** It excludes compressed pages, which is most of a JAX worker's real
footprint. A footprint re-measurement on the same class of machine (`top -l 1 -o mem -stats
pid,ppid,mem,cmprs,command`, 2026-09-13, `-n auto` = 11 xdist workers, jax 0.10.2) found the workers
holding **2.5–5.9 GB each, ~49.5 GB combined**, against a summed RSS at the same instant of only
**4.45 GB** — an order of magnitude apart, and the reason nothing caught the next collision: on
2026-09-13 **two fast tiers** started five minutes apart, from two different worktrees, neither
touching a case, and drove the machine to a hard reset (~100 GB of combined footprint on a 19 GB
machine). `tools/fastgate.sh` now also holds a machine-wide lock (`~/.cache/aquaflux/fastgate.lock`)
while any tier runs anywhere on the machine, refusing a second one — from any worktree, any session —
with the same `FASTGATE_FORCE=1` override and the same CI exemption (exit `4`, distinct from the case
guard's exit `3`). The case guard above and this lock are two different mutual exclusions: a tier can
collide with either a case or another tier, and closing one said nothing about the other.

**What contention does and does not move — and this is the useful half, because it needs no clock.** The
cleanest instance is a matched pair in ONE worktree at ONE commit, differing only in machine load, whose
two logs align line for line. Both print rung 1's closing row on their own line 738:

    |   28 |    895 | 0.0050 |  1 |   2 | 7.214e-06 | 1.000 |     |     <- loaded machine
    |   28 |    194 | 0.0050 |  1 |   2 | 7.214e-06 | 1.000 |     |     <- quiet machine

**Every column is identical except `t(s)`, which differs 4.6x** — step count, `beta`, inner count, cycles,
`|R|` to all four figures, `a_min`. The same holds over the **whole** march: both runs finish at
**69 steps / 417 cycles**, final `|R|` **5.771e-06**, `Ux` in **[-2.694, 10.222]** — every digit — against
**1325 s and 555 s** of wall clock, a **2.4x** spread on an identical trajectory. (Both on the compiled
host ILU kernel, per their banners; see the cross-checkout confounds below for why that has to be said.)
Steps, cycles, escalations, line-search clips and the converged root are deterministic and contention
cannot move them. Wall clock is a different
matter: this project already records ~15 % run-to-run spread on an *uncontended* per-application timing,
and 4.6x is far outside that. **A contended run is still good evidence about counts and worthless about
seconds** — keep its step and cycle columns, discard its timings, exactly as for a run that spanned a
machine sleep.

**`895 -> 194` IS an attributable cost, and the attribution is graded.** The loader is known and was read
from its own log rather than reported: a fast tier in another worktree, `21:55:02 -> 22:12:27`,
`1045.13s (0:17:25)`, 1498 passed — one minute after the case launched at `21:54:04`. Splitting the slow
march at its rung boundaries against the quiet one, the ratio tracks **how much of each rung overlapped
that window**:

| rung | overlap with the gate | slow | quiet | ratio |
|---|---|---|---|---|
| 1 | entirely inside it | 499 s | 157 s | **3.2x** |
| 2 | ~90 %, the gate ends mid-rung | 229 s | 168 s | **1.4x** |
| 3 | none — it starts after the gate ends | 194 s | 186 s | **1.04x** — *the control* |

**That graded structure is the evidence, not the endpoint ratio, and the control is what makes it one.**
The ratio reaches unity exactly where the overlap does: 8 s in 190 on a machine with a documented ~15 %
spread is not a small effect, it is *no* effect. Anyone arguing the slow arm differed in some other way
now has to explain why that difference confined itself to the two rungs that overlapped a known job.
(Measuring each rung from the previous rung's last step partitions the march with no gaps; measuring from
each rung's own first step instead gives 499/157, 215/160, 176/170 — the same three ratios within 2 %, so
the conclusion does not rest on the convention.)

⚠️ **And the cost is MUTUAL, which is the part that argues for a guard rather than a convention.** The
gate was slowed by the case as surely as the case was slowed by it: it took **17:25**, and the evening's
second gate **13:00**, for two jobs that alone are roughly 8-9 minutes each. **Concurrency here is not a
trade of latency for throughput — it is a loss on both sides**, which matters when arguing for a guard,
because nobody is giving up throughput they were actually getting.

**It is a measurement, not a shape — the uncontended arm was run the same evening.** Three gates, one
machine, one night, within 1 % of each other in test count, so composition is not the variable:

| gate | tests | wall | vs clean |
|---|---|---|---|
| clean (no case running) | 1494 | **8:44** | — |
| overlapping a case | 1498 | **17:25** | **2.0x** |
| overlapping a case | 1482 | **13:00** | **1.5x** |

And the makespan is worse too, which is the part with no counter-argument. The gate alone is 8:44 and that
case alone is 9:15; **back to back the pair finishes in ~18:00, overlapped the later one finished at
~22:05** — four minutes later, with *both* jobs individually slowed on the way. There is no throughput
being bought here to trade latency for. (This supersedes an earlier reading of the 17:25 against the
tier's documented 6:34-8:53, which was not a fair anchor: different worktree, tier composition and date.
The three rows above are the comparison that holds.)

**The `--status` line was clean throughout.** At 22:09, mid-slow-arm, the machine sat at load **13.4** with
**0.73 GB** free while `run_case.sh --status` reported one case and nothing else — because what was
loading it was a *test tier*, and a test tier is not a case. **That is the whole argument for this
section**: not that the runner missed something unexplained, but that it is blind to a job class we run
constantly, and here that job class is named, timed, and measured.

⚠️ **There is currently NO trustworthy wall-clock baseline for pitzDaily from any session.** A figure of
518 s circulated this evening as "uncontended" and has been withdrawn by the session that produced it: it
was taken under `run_case.sh`'s guarantee, which establishes only that no other *case* was running and
says nothing about a test tier or anything else on the machine. Any speedup ratio built on it inherits
that, so do not quote one. This is the "record what a measurement was taken under" rule biting in its
sharpest form — the number was not wrong, it was **unfalsifiable**, and it had already been adopted by a
second session before its author caught it.

⚠️ **A run that changes two variables at once separates neither — and the tempting attribution here is
wrong twice over.** The contaminated march reached step 1 at `t = 396 s`; a later run reached it at
**37 s**. That run was quiet **and** cache-warm, against a bad run that was contended **and** cache-cold,
so the honest statement is *"roughly 10x, cause not isolated"* — **not** "a cold cache costs six minutes",
which is what both sessions involved were about to write. Isolating it needs a third arm: quiet machine,
cache deliberately cold. Until someone runs one, quote the pair only with this caveat attached.
What the same comparison *does* establish, at no cost: step 1's `|R|` is **bit-identical** (`5.668e-02`)
across the contended and the quiet run — a stronger form of the determinism above than agreement to four
figures.

⚠️ **Contention is only ONE of the reasons a timing does not travel, and the other two are cross-checkout
rather than cross-process — the same lesson, three instances, all invisible in the log.** The root
briefing already records both of the others; what follows is only the connection to this section, so that
a reader chasing an unexplained wall-clock difference checks all three rather than the one they happened
to read about.

- **The compiled ILU(0) kernel was a gitignored artifact** (deleted with it 2026-09-13, #371). A fresh
  worktree silently ran the pure-Python twin, so a `hostilu` timing from before then is only comparable
  if its run log's `host ILU kernel` banner line says `compiled`.
- **The JAX compilation cache is shared but keyed on the compiled program.** A branch carrying different
  solver code takes misses in a warm checkout, so the first run on a new branch is partly measuring
  compilation. Nothing in the log distinguishes that from the case being slower.

Together with the step-1 confound above: **a cold-cache premium and a contention premium are two of at
least three ways the same seconds can go missing, and a single run separates none of them.** A bound of
roughly 90-360 s has been proposed for the cold-cache half by grading it against the contention ratios;
it is **not recorded here**, because it assumes single-threaded compilation is starved by a `-n auto`
tier in the same proportion as the numerics, and that assumption is exactly the one this section declines
to make elsewhere. The run that would settle it — quiet machine, deliberately cold cache, extension
built — is cheap and has not been done.

⚠️ **`pgrep -f "tools/fastgate.sh"` DOES NOT TELL YOU WHETHER A GATE IS RUNNING.** It matches any
*watcher* whose own command line contains that string — an `until ! pgrep -f "tools/fastgate.sh"; do
sleep` loop matches itself and waits forever — so the poll reports "running" long after the gate exited,
and a session gating on it will either stall or raise a false alarm at a peer. This is the same
self-matching trap the root briefing records for a case waiter, one tool along. Match the **real**
process (`bash tools/fastgate.sh`, or the `python -m pytest` it spawns) with the watcher shells excluded,
or read the gate's own log — `$TMPDIR/aquaflux-tests-<worktree>-fast-*.log`, whose last line is the
pytest summary. For the record, the gate described above finished at **13:00** for 1482 passed / 1
skipped, half again the documented 6:34-8:53 range and a further measure of what the collision cost.

**Wherever the gate does not refuse on its own, the check is manual and it is on the person starting the
*tests*, not the case:** run `validation/run_case.sh --status` before `tools/fastgate.sh`, not only before
another case. Checking once and launching twice is the specific way this failed — the status was clean
when the case was queued and stale by the time the gate followed it. **A guard closing this direction has
been written and is pending review**; when it lands, the manual check still applies under `CI`, under any
force-override, and in a checkout that predates it.

⚠️ **The reverse direction stays open even then, and it is worth knowing which half is covered.** A guard
on the gate can ask `run_case.sh` whether a case is live, because a case writes a run-file. Nothing asks
the question the other way: **a case cannot see a running test tier, because no tier writes a run-file**.
Both collisions recorded here were gate-after-case, so that half covers every instance actually observed
— but "case starts on top of a gate" remains available, and closing it means giving the gate a run-file
of its own, which is a larger change than the one that was asked for.

⚠️ **Do not read this as a knowledge gap to be closed by documentation.** Three sessions on the evening
this was recorded all knew the one-heavy-job-at-a-time rule and it happened anyway, because the rule is
enforced for one pair of jobs and merely known for the other. The durable fix is to make the collision
unavailable rather than better documented — teach the gate to consult the machine-global run-file and
refuse or warn — and this entry's job is to stop the wall-clock numbers being trusted, never to stand in
for that. Read it as a record of what happened and what the measurements are worth, not as the remedy.

## From `.claude/rules/validation.md` — run_case.sh --wait exit status (2026-09-22) and liveness

## ⚠️ `run_case.sh --wait` exited 0 for crashed cases until 2026-09-22 — it now exits with the case's status

**Both `--wait` forms used to exit `0` however the case ended**, because the case runs detached and
nothing collected its status. Observed 2026-09-22 on `validation/sozzi_radiation/compare_fluence.py`: one
run died of a `ModuleNotFoundError` (traceback in its log) and another was killed mid-script with **no
traceback at all** (almost certainly out of memory) — both returned `0`, so `run_case.sh … --wait &&
next-step` went on after a crash. That is the silent-success class the script exists to remove, sitting
in the script itself. **Any `--wait` status read before this fix says nothing about the run; read the
log.**

How it works now, and what not to undo:

- **The case runs inside a wrapper subshell that writes `run-<stamp>.exit` beside the log** (and a
  `[run_case] the case exited with status N` line at the foot of the log) when the case ends. A signal
  death is `128+N` — an out-of-memory SIGKILL reads `137 (killed by signal 9)`, the only record such a
  run leaves. Both waiters print `exit status: N` and exit with it; **a missing `.exit` file is reported
  as failure**, since "we do not know how it ended" is not success (the wrapper was killed, or the run was
  launched by the old script).
- **The status lives in a sibling file, not the run-file**, because `live_pid` deletes the run-file the
  moment the pid is dead — exactly when a waiter wants the status, and possibly from some other caller's
  `--status`.
- **`pid=` in the run-file is now the WRAPPER's pid, not python's.** `kill -0` on it is still the
  liveness test (a waiter never matches itself), and the wrapper outlives the case by exactly the status
  write, so a waiter that sees it gone always finds the status on disk. `kill <pid>` still stops the run:
  the wrapper traps TERM/HUP and **forwards** them to the case, then re-`wait`s until the case itself has
  exited — `wait` returns early (143) when a trapped signal lands, so without the loop the status of a
  case that shuts down on SIGTERM with its own code would be misreported. **`kill -9 <pid>` now kills
  only the wrapper and orphans the case** (and `--status` then reports nothing running); stop a run with a
  plain `kill`.
- **`caffeinate` watches the wrapper's pid (`-w`) instead of wrapping the case**, so it is out of the path
  the status travels. The wrapper's stdio is detached so a caller capturing the launch output does not
  block for the case's lifetime.
- **`--force` skips taking the memory/load readings, not only judging them**, so a reading that cannot be
  taken never blocks a forced launch.
- **`free_gb` reads Linux too (2026-10-14).** It read only macOS's `vm_stat` (free + inactive pages), so
  on Linux every UNFORCED launch died in a Python traceback before the case started, and runs here were
  launched by hand around the runner. It now falls back to `/proc/meminfo`'s `MemAvailable`, the kernel's
  own estimate of the same thing. Every test above passes `--force`, which is why nothing saw it; the
  unforced pre-flight is pinned by
  `test_the_PRE_FLIGHT_reads_free_memory_on_this_platform_and_refuses_below_its_bar` and
  `test_an_unforced_launch_that_PASSES_the_pre_flight_runs_the_case` (both red with the `vm_stat`-only
  reading restored, on Linux). Entries elsewhere saying a run went around `run_case.sh` because "it needs
  `vm_stat`" record how those runs were made, not a limit that still holds.
- **Liveness is the pid AND its start time (2026-10-12).** The run-file records `pid_started=` (`ps -o
  lstart=`, spaces squeezed), and `live_pid` treats the run as live only when a process with that pid
  exists and started then; otherwise the run-file is stale and removed. After a cloud-container
  restart a finished run's pid belonged to a kernel thread, `kill -0` succeeded, and every launch was
  refused (`--force` does not skip that refusal). A run-file without `pid_started=` reads as stale.
  Pinned by `test_a_recorded_pid_now_held_by_ANOTHER_process_is_not_a_running_case` (red with the
  comparison removed); `test_fastgate.py`'s fake run-file carries the start time too.
  ⚠️ **Both sides go through `canonical_time` (blanks squeezed AND trimmed), not `tr -s ' '` alone
  (2026-10-14).** macOS `ps` pads `lstart` to its column width, so `tr -s` kept one trailing blank there
  and none on Linux. The script matched its own run-files, but the two tests above write the time
  `split()`-and-joined, so both failed on every Mac while CI (Linux) stayed green. Canonicalizing the
  *recorded* value too keeps a run-file written before the fix live.
  `test_a_start_time_is_compared_whatever_its_blanks` pads the time on every platform.
- `AQUAFLUX_CASE_POLL_SECONDS` (default 20) sets the waiter's poll interval; it exists for the tests. `run-*.exit` is gitignored beside `run-*.log`.

Pinned by `tests/unit/test_run_case.py` (fast tier): exit 3, an uncaught exception, a self-SIGKILL
(137), a success (0), `--wait` attached to an already-running case, `kill <recorded pid>` reaching a
case that exits 7 on SIGTERM, and a SIGKILLed wrapper reporting failure. Mutation-checked 2026-09-22:
restoring the always-0 return in each `--wait` form, dropping the TERM forward, dropping the re-`wait`
loop, and treating a missing status as success each turn at least one test red.

## From `.claude/rules/validation.md` — a report about a run is not the run's own record

## ⚠️ A report about a run is not the run's own record (the evening's actual lesson)

**Every correction in the section above was an account being preferred to a record that was already on
disk.** The pattern is worth more than any of its instances, and it generalizes past machine contention —
one of these had nothing to do with load:

| got wrong | from | settled by |
|---|---|---|
| which job loaded a slow march | a peer's account plus timing structure | the gate's own `$TMPDIR` log |
| whether a gate was still running | a process-table match | that log's last line |
| which PR changed which files | a `HEAD..origin/main` range spanning two merges | `git show --stat <sha>` |
| a wall-clock baseline | the runner's guarantee about *cases* | nothing — it was withdrawn |

**The records were right every time; the accounts were wrong every time.** Three records earned that:
the gate's log (`$TMPDIR/aquaflux-tests-<worktree>-fast-<stamp>.log`, pytest summary on the last line),
`run_case.sh`'s run-file, and `git show`. Three sources did not: a peer's report, one's own recollection,
and the process table.

⚠️ **And a record can be authoritative about the wrong question, which is the failure mode hardest to
see.** On a machine where every session pushes as the same GitHub account, **a pull request's author field
does not identify which session wrote it** — three PRs opened that evening by three different sessions all
report the same author. It arrives from an API rather than from anyone's memory, so it *reads* as a record,
but it records the account and the question being asked is about the actor. That is circumstantial in
exactly the way the process table is. What settles authorship is the work itself: which worktree contains
the code (`grep` for a symbol the change introduces), and which branch has a remote ref at all. Before
trusting a field, check that it answers the question you are asking and not a neighbouring one.

**This is not a new rule — it is the one `run_case.sh` was built to embody**, and its own header says so:
the most valuable thing it produces is not the log but that *"is this run mine, and what is it testing?"
has a written answer*, a question that has been got wrong from the process table alone. Reach for the
written answer first. The cost of not doing so, measured here, was four wrong claims in one evening
between two sessions that were each checking the other's work.

## From `.claude/rules/validation.md` — case files: solving from them, inert PITZ_* overrides, stale channel report

## Every case with a case file now SOLVES from it (2026-09-24)

The channel harnesses call `checked.solve(coupled)` and hold no solver setting; pitzDaily and bfs3d read
their march from `case.yaml`'s `solver` section, with each `PITZ_*` / `BFS3D_*` march variable an **edit**
of it (see `.claude/rules/case.md` → "The case files in the repository"). Two consequences for anyone
writing a harness:
- **A module constant a probe imports (`compare.INNER_STEPS`, `compare.CONTROL`, `compare.PRECONDITIONER`,
  ...) is now read back from the file**, not a literal. Change the value in `case.yaml`, never in the
  script — bfs3d refuses to import when its reassembled `SOLVER` differs from the file's with no override
  set, and a fast-tier test imports both scripts that way.
- **A setting cannot be passed to a case's `solve` as a keyword** — it is refused. A study that varies one
  edits the spec (`dataclasses.replace`), as the `PITZ_*` overrides do; one that varies something no file
  can state is a script-only study arm that calls the library itself (pitzDaily's `_solve_study_arm`).

⚠️ **From that change (2026-09-24, #437) until 2026-10-08, pitzDaily's `PITZ_*` MARCH overrides were
INERT.** `solve_aquaflux` solved `case["spec"].solver` (the file's) while the banner printed the edited
`SOLVER`, so every override — `BETA_START`, `REFRESH_ON_CYCLES`, `RETRY_ON_CYCLES`, `TURB_DAMPING`,
`K_POSITIVITY_PROJECTION` (since deleted), `FLOW_SWEEPS`, `FIELD_SPLIT`, `PROBE_GRADIENT_SWEEPS`, `RAMP_*`, the new
`FORWARD_RTOL` — was reported as in force and ran the file unchanged; the study arms received the same
unedited solver. Found when three marches at `FORWARD_RTOL` 0.01 / 0.1 / 0.6 came back bit-identical to
the 0.3 baseline. Fixed (`solver = SOLVER`) and pinned by
`test_a_pitzdaily_march_override_reaches_the_solve_not_only_the_banner`, mutation-checked. The
`stencil_reach` and `gradient_scheme` *arguments* were unaffected. **A pitzDaily arm in that window that
set a `PITZ_*` march variable is a run of the file's solver**; none is recorded in these files. bfs3d
reassembles its solver from its constants and was not affected.

## The channel studies read case files (2026-09-24) — and their tracked report was stale

`turbulent_channel` and `turbulent_channel_openfoam` build each configuration from a file under `cases/`
(see `.claude/rules/case.md`). ⚠️ **`turbulent_channel/report.md` dated from 2026-07-20 and said Re_b =
240000 "did not converge (issue #99)".** It converges on today's code: Re_tau 4946, u_tau 0.0412 (Dean
0.0406), kappa 0.382, 116 s — and **`main`'s own hand-built script gives the identical numbers** (run in
isolation, 121 s), so it was fixed by some solver change since July, not by the case-file switch. The
regenerated report is committed. The other two rows reproduced exactly (Re_tau 538 / 1093, kappa 0.340 /
0.361). The OpenFOAM study's aquaflux `low` side reproduces its report exactly (Re_tau 380, u_tau
0.0569, nu_t/nu 50.4, kappa 0.340); its OpenFOAM results are gitignored and were not on the machine, so
the full comparison was not rerun. One run each, macOS arm64, 2026-09-24.

## From `.claude/rules/validation.md` — ray_effects_room

## `ray_effects_room` — aquaflux against discrete ordinates in a far-UVC room (2026-09-28)

A demonstration case, not a solver study: a 4 x 4 x 3 m room lit by a Care222 module (the
OSLUV-measured `ushio_b1.ies`, committed with its MIT licence), a Voronoi-lattice bunny hanging in
it, black walls (and a 0.5-reflecting variant); aquaflux, of-optical-radiation's DOM and an
independent brute-force reference (`reference.py`, numpy + Embree, no aquaflux import) on one
snapped mesh. Its README carries every number with its configuration. What to know before touching it:

- **DOM needs of-optical-radiation's `add-incident-flux-output` branch** (`qin`, the exact-power
  `iesEmitter`, no scatter snapshot); `generate_dom.py` records the commit beside every run.
- **Killing the case must reach Docker.** A `kill` of `generate_dom.py` used to leave its container
  running on, unseen by `run_case.sh`; containers are now named and stopped from a SIGTERM/SIGHUP
  handler. The same shape as the `peak_footprint.py` trap above, one tool along.
- **A mesh directory is not a finished mesh**: `snappyHexMesh` writes `polyMesh` before the lamp
  patch is cut, so each meshing step leaves its own completion marker.
- **The cell count is not `owner.max() + 1`**: the highest cells may appear only as a neighbour
  (2-3 cells short on both meshes); `cells.npz`, written by OpenFOAM itself, is the authority.
- **aquaflux's side is CASE FILES since 2026-10-05**: `cases/{bunny,empty}_{floor,volume}.yaml`,
  `bunny_reflecting.yaml` and `bunny_reflecting_coarse.yaml` (0.4 m reflecting triangles), run by
  `run_cases.py` (`RAY_CASES`, `RAY_OVERWRITE=1`) through `run_case.sh` into `work/cases/<name>/`
  (`fields.vtu`, `patches.vtm` + `patches/*.vtp`, `run.yaml` with `results`). `aquaflux_floor.py`,
  `aquaflux_volume.py` and `aquaflux_reflecting.py` are **deleted**; `figures.py` reads the run
  directories through `outputs.py` (a numpy decoder of the VTK files, independent of the writer);
  `slices_refined.py` is the one harness left, re-gathering the slice cells of `<mesh>_volume.yaml`'s
  built scene with the floor case's refined lamp. `work` may be a symlink to the worktree that meshed
  the rooms (the `.gitignore` entry is `/work`, which matches a link; `work/` did not).
- **The port reproduces the scripts (2026-10-05/06; `943389aa` + the uncommitted port, jax 0.10.2,
  CPU, macOS arm64, 11 cores; the deleted scripts' outputs at `8dd4caae` as the comparison).** Floor `E`
  (bunny, 17,920-facet lamp) to 9.1e-12 of its peak, floor power 0.0994887477227159 W both; empty floor
  9.1e-12. `bunny_volume` `G` p99 6.8e-10 of max, `volume_integral_G` to 3e-10 relative, but **26 cells
  move by about one lamp facet's share** (shadow-edge ray decisions: the bunny's fan triangulation
  differs from the old script's), one near-bunny cell 3.36e-4 -> 0. `bunny_reflecting` is NOT the same
  configuration and is not a bit-for-bit check: 9,762 coarsened mesh-patch triangles (<= 0.2 m) against
  2,000 squares, the lamp's own 1,120 facets (the script refined it for the floor and the walls' light),
  a black lamp window. Under that: room absorbs 113.133 vs 113.129 mW, bunny 5.693 vs 5.697 mW,
  reflected share of floor power 0.13443 vs 0.13442; reflected `G` p99 2.6 % of its peak, worst 40 %
  in a cell 6 mm from the floor (0.7 % p99 past 10 cm from the walls). 0.2 -> 0.4 m moves the reflected
  floor <= 4.5 % of its own peak (the script's squares: 5.1 %). Wall times (`run.yaml` `seconds`, mesh
  read included): bunny floor 306 s, empty floor 148, bunny volume 805, empty volume 99 and bunny
  reflecting 5,277 (both shared with unit tests: upper bounds), coarse 2,009 (its tail beside a test tier).
- **`test_validation_api.py` matched `from aquaflux...` by string prefix**, so a case's sibling
  module named `aquaflux_floor` was taken for the package and every name imported from it reported
  missing. It now matches the package or its submodules (`module == "aquaflux"` or
  `startswith("aquaflux.")`), pinned by `test_a_sibling_module_whose_name_begins_with_aquaflux_is_not_the_package`.
- **`run_case.sh` records `RAY_` settings** (added to its capture prefixes; `test_check_env_prefixes.py`
  fails a case whose prefix is missing, which is how this was caught).
- ⚠️ **of-optical-radiation's DOM grid is uniform only when `nPhi == nTheta`.** It divides the polar angle over
  the WHOLE sphere into `nTheta` bins of `pi/nTheta` and the azimuth into `2 nPhi` bins of `pi/nPhi` (its tutorial's
  "polar bins per hemisphere" comment was wrong). `8 x 4` / `16 x 8`, used for every 64- and 256-direction figure
  below and in `sozzi_radiation/` before 2026-10-02, have polar bins twice the azimuthal; `n x n` gives `2 n^2`
  directions and matches Fluent's `n/2 x n/2` per octant (6 x 6 = 72 = Fluent 3 x 3). On the Sozzi uvmesh swap the
  8 x 4 grid's dose shortfall at k = 0.5 was 12.4 %, the uniform 6 x 6's 2.2 %: most of it was the grid's shape.
  Every figure below quoted "at 64 / 256 directions" is on the non-uniform grids.
- **`sozzi_radiation/background_grid.py`: aquaflux on a Cartesian grid, interpolated to the mesh (2026-10-02).**
  Gathering only at 4 mm grid nodes (90,788, 43 s against 714 s at every cell) gives the dose within 0.65 % in
  LR at every k, **but only with the sleeve's own value as data**: G = 2M at a diffuse emitter's surface
  (outgoing hemisphere at zero path; the sleeve is convex and the walls black). Without it every variant
  under-predicts the cells beside the sleeve (median 0.57-0.81 within 1 mm, 8-3 mm), because each wet corner
  is farther from the lamp than the cell. ⚠️ `G_average`'s LR looking right at 3-4 mm is a cancellation (low
  near the sleeve, high in the bulk), not accuracy: its mean dose is 2 % low and its p1 per-particle ratio 0.83.
  ⚠️ Twelve tracker runs at once in Docker's VM killed three (memory); run five at a time. Numbers in its README.
- **A diffusely reflecting DOM sweep costs 3.3-3.8x a black one** (209 against 63 s at 64 directions,
  984 against 257 s at 256, 8 ranks) and needs ~60 % more sweeps (57 against 36 at 64), so the
  reflecting 256 run was stopped. ⚠️ Not quadratic in the directions, as was first said here: 64 -> 256
  multiplied a reflecting sweep by 4.7 and a black one by 4.1. The likely cost is the `reflective` BC's
  sum over every direction at each wall face, which is read from its code, not profiled.
- **The error window decides the trend, so state it with the number.** Over the central 2 x 2 m the
  DOM floor error ROSE with directions (L2 / peak 0.77 -> 1.76, 64 -> 256); over the whole floor it
  FELL (2.03 -> 1.07), because at 64 every beam lands outside the window. "More directions made it
  worse" was put on a slide from the window figure and had to be withdrawn. The README's table is
  the window; the full-floor figures are in its bullets and in the talk's `export_room_data.py`.
- **Pixels are a run parameter (`bunny:16x8/3x3`) and every record carries them**; runs before
  2026-09-29 had 1 x 1 from the template, backfilled into their records and marked so. On this mesh
  3 x 3 pixels change the floor error by nothing to four figures at 64 and 256 directions (only the
  bunny's snapped cells have faces a bin overhangs). Do not read that as "pixels never matter": the
  room is hex-dominant with axis-aligned faces and bin edges; a tet or polyhedral mesh is untested.

## From `.claude/rules/validation.md` — tubular_reactor

## `tubular_reactor` — a ten-lamp reactor on a background grid, and its resolution study (2026-10-15/16)

A radiation-only case with no flow mesh: 10 Sozzi lamps (35.4 W each) in a 300 mm vessel, with a
100-hole, 50 %-open perforated plate at the inlet and walls reflecting 0.1. `G` is gathered at the
nodes of a Cartesian grid in the water (`run.py`, `TUBULAR_*` settings, outputs in
`work/runs/<tag>/`). `study.sh` runs the grid and tessellation studies, and `compare.py` compares
them. Numbers and configuration are in its README. What to know before touching it:

- **Shadows are exact bodies, and the plate is a bespoke `Body` (`geometry.PerforatedPlate`).** A
  sight line crossing the slab is clear only if its part inside the slab lies in one hole, found by
  one hexagonal-lattice lookup. ⚠️ **The first version put the 100 holes in an `Outside` as 104 convex
  regions, and that ran a smoke test to a 54 GB footprint before it was killed.** Its pair test forms
  intermediates per region at the 4M-pair default. Verified against brute-force hole sampling:
  1 of 4,000 lines differed, and finer sampling resolved it (a 7 µm graze).
- **Lamps are emitters only; their shadows are separate exact bodies.** A cylinder plus hemisphere,
  `TUBULAR_SHRINK` (0.251 mm) inside the sleeve so no facet centroid lies inside. That is fixed
  rather than derived, so a lamp-tessellation study does not move the shadows.
- **The lamps are kept out of the radiosity on purpose: the solve is hand-assembled, not
  `solve_scene`.** `solve_scene` puts every lamp facet into the dense transfer, and 50k lamp facets
  plus 12k wall triangles cannot be held. Wall triangles at 16k and 18k (20 mm edges, 12-sided
  holes) did fit, on a 19 GB machine.
- **Grid results (h = 8/6/4/3 mm):** away from the sleeves the interpolation error falls as h². Within
  2 mm of a sleeve it is 17-36 % at every spacing (issue #726). ⚠️ **The node-averaged mean G is not a
  converged volume integral**: it moves non-monotonically, 112.9 / 114.8 / 115.1 / 114.1 W/m².
- **Figures are drawn by `render.py`**: 0.5 mm pixels masked by the exact water test and interpolated
  from the nodes, so boundaries are true curves. The plate-plane figures are direct light only and
  say so on the figure.

## From `.claude/rules/validation.md` — the aggressive Reynolds continuation does not need the complete LU (2026-10-11)

## The aggressive Reynolds continuation does NOT need the complete LU (2026-10-11)

`pitzdaily_openfoam/compare_reynolds_continuation.py` used to say its aggressive Courant control could
reach the developed reattachment only with an exact complete-LU preconditioner, re-factored every step,
because "a block-triangular SIMPLE preconditioner loses diagonal dominance and the step goes non-finite".
That was the last claimed regime for the LU, so it was re-run on the field split before the LU was deleted.
**Measured 2026-10-11** (`validation/pitzdaily_openfoam/compare_reynolds_continuation.py` at `10449ed` plus
the harness changes of the LU-removal PR; Linux x86_64, 4 cores, jax 0.11.2, Python 3.13): the aggressive
`DualTimeControl(beta_start=0.5, beta_min=0.005)` Reynolds continuation (Re/100, Re/10, target;
`inner_steps=10`, `inner_tol=1e-3`, intermediate `rtol` 3e-2, target `rtol` 1e-3), preconditioned by
`case.yaml`'s own `FieldSplit(SimpleSmoothed, JacobiSmoothed)` at `refit_beta_floor` 0.05 with
`refresh_on_cycles=2`, built at β 0.5: **converged in 40 outer steps / 852 restart cycles, no retries and
no non-finite step, `x_r/h` 8.069 (OpenFOAM 7.741; `compare.py`'s root 8.0686), peak `nu_t/nu` 416 (423)**.
Rungs took 12 / 11 / 17 steps. The one hard step was rung 2's last (β 0.009): 133 cycles, inner α 0.016,
finite, and it met the rung's tolerance. ⚠️ Wall clock (3305 s) is NOT quotable: a unit-test run shared
the machine for the first ~10 minutes. ⚠️ A first attempt with NO `refresh_on_cycles` (split fitted once
per rung, never refreshed) was stopped at step 8 with cycles rising 9 → 31 — that configuration is not one
anything ships, and the LU it was compared with refactored every step, so it was not a fair arm. The LU arm
itself was not re-run (the harness's claim was recorded without a run log in the repository).
The harness now runs the split only. Two defects in it predated this and were fixed on the way: its
step-control wrapper called `float()` on `DualTimeControl`'s carried state, which is a `(β, previous
|R|)` tuple, so it crashed on its first step on `main`; it now reads the shift off the step with
`shift_of`. And it does not put its own repository on `sys.path`, so `import aquaflux` resolves to the
editable install — a run from a second worktree must set `PYTHONPATH` to that worktree, or it silently
runs the other checkout's code (this happened, mid-edit, and crashed on an import the edit had removed).

## From `.claude/rules/validation.md` — fails locally, green on CI (2026-08-19)

## ⚠️ "Fails locally, green on CI" — the 2026-08-19 case was a SKIP, not a platform

Three slow-tier tests failed locally with `RootSolver did not converge` while `main` was green on GitHub,
and this section once explained it as a platform difference (arm64/Python 3.13 vs x86_64) with a
`max_steps` hypothesis. **The real cause was simpler: all three were `pytest.importorskip("petsc4py")`-gated,
CI did not install PETSc, so CI never ran them** (root `CLAUDE.md`, "CI is not a superset of a local run").
Two of them (`test_coupled_amg.py`) were deleted with PETSc on 2026-10-10; the third, now
`test_the_split_continuation_reaches_the_block_preconditioned_root` (renamed when the complete LU it
compared against was deleted, 2026-10-11), is PETSc-free and runs in CI.
**The trap that survives: before attributing a local-only failure to your platform, check the CI skip
counts** — a guard was once reverted on the strength of the platform story.

## bfs3d on a 4-core Linux container: the march, the re-fit's cost, and the allocator (2026-10-17/18)

Container: 4-core Intel Xeon @ 2.1 GHz, 15 GiB, no swap; jax 0.11.2; commit `b1648725`; compilation
cache cold for the march. Shipped `case.yaml` throughout.

**The march** (`validation/run_case.sh validation/bfs3d_openfoam/compare.py --wait`, glibc defaults, no
OpenFOAM fields present, so the comparison was skipped): 29 steps, 48 inner solves, 173 restart cycles
(none rejected), 29 re-fits (13 full, 16 cost), final `R` 3.69e-06, mid-span `x_r/h` **8.3611**,
solve 3155 s (3178 s with set-up), peak `VmHWM` 12.1 GiB. Re-fits 1844 s of it (probe 1247 s, refactor
543 s), each 56–64 s; steps without their re-fit ~20 s. Against the M3 Pro record (18 / 137 / 5.1e-06 /
522 s, jax 0.10.2) the counts differ as expected (rounding moves the leading coarsening); the root agrees.
The case process used ~2.1 of 4 cores, 21 % of CPU in the kernel, ~146k minor faults/s.

**One re-fit at `checkpoints/state-00029`, β 0.005, mean of 3** (`validation/refit_profile.py`):

| allocator | probe | assemble | slice | leading | trailing | whole | faults |
|---|---|---|---|---|---|---|---|
| glibc defaults | 38.35 s (2.88 cores) | 1.77 | 3.24 | 13.83 | 1.05 | 58.29 s | 15.9M |
| mmap 1 GiB + trim 2^31−1 | 36.54 | 0.19 | 1.00 | 8.14 | 0.97 | 46.87 | 15.1M |
| … + `MALLOC_ARENA_MAX=1` | **28.75** | 0.16 | 0.80 | 7.94 | 0.95 | **38.63** | 0.15M |
| jemalloc `LD_PRELOAD`, default decay | 37.38 | 0.86 | 3.34 | 13.23 | 1.30 | 56.17 | 11.6M |

The thresholds cure the host refactor's faults; the probe's batched JVP runs on XLA worker threads, whose
arenas `mmap` above 64 MiB regardless, until one arena is forced. jemalloc with decay disabled
(`MALLOC_CONF=dirty_decay_ms:-1,muzzy_decay_ms:-1`, which the pitzDaily session found removes the faults
there) was not run on bfs3d.

**pitzDaily on the same machine, hybrid start, β 0.005**: whole re-fit 2.87 s (probe 1.54 s, 100 probes)
at defaults, 2.66 s (probe 1.40 s) single-arena. **Batched JVP per tangent, single-arena, best of 5**
(`REFIT_JVP_BATCHES`): pitzDaily 22.3 / 13.0 / 8.5 / 8.4 / 7.6 ms at batch 1 / 2 / 4 / 8 / 16; bfs3d
68.8 / 77.8 / 54.2 / 44.8 / 53.5 ms (batch 16: 388k faults per call even single-arena). So on this machine
bfs3d's probe is ~25× pitzDaily's: 5.76× the probes (576 against 100) times ~5.3× per tangent at batch 8
for 2.26× the unknowns. Why: next subsection ("Why bfs3d's probe tangent costs 5.5× pitzDaily's").

**The single-arena march arm died**: `MALLOC_ARENA_MAX=1` plus both thresholds, otherwise identical;
re-fits 45.8 s and 42.4 s (probe 28.8 / 28.1 s) at steps 1–2 against 63.8 / 61.7 s at defaults, but
12.3 GiB resident by step 2 (9.5 GiB at defaults), and the container restarted at ~00:42 with no exit
record and no surviving kernel log. Inferred cause: one arena with a 2 GiB trim threshold retains freed
memory. Both arms left unmeasured there are measured below.

## The shipped bfs3d march fails at step 35 on `fe772f59` (2026-10-18)

`validation/run_case.sh validation/bfs3d_openfoam/compare.py`, shipped `case.yaml`, glibc defaults, 4-core
Xeon, 15.7 GiB, jax 0.11.2; on `main` at `fe772f59` and on `claude/bfs3d-memory` (identical step tables
through step 34). Both raise at the re-fit before step 35, in the trailing `[k, omega]` block:
`SmoothedHierarchy.refit (level 0): block smoothing needs every cell block invertible, but 1 of 23040 are
singular`, with `R` 4.557e-04 (target 1e-05) and the shift at its 0.005 floor. The last recorded
convergence (`b1648725`, 2026-10-17: 29 steps, `x_r/h` 8.3611) predates #757 and #761; which of the commits
between moved the trajectory onto the singular block is not bisected. Logs: `march.log` of each run (not
kept in the repository).

## bfs3d with the pivot floor, by machine (2026-10-19)

Shipped `case.yaml` (`PivotFloor(0.15)`, float32 split, `simplesmooth` leading, uniform reach 3), no
OpenFOAM reference present, `validation/run_case.sh validation/bfs3d_openfoam/compare.py`, warm-ish cache.
Container: Linux x86_64, 4 cores, 15 GB, jax 0.12.0, numpy 2.5.3, scipy 1.18.1.

| checkout | steps | cycles | `x_r/h` | peak `VmHWM` (15 s samples) |
|---|---|---|---|---|
| #766 head `1e69b4ab` (main's march: the production freeze it deleted before landing was off) | 39 | 254 | 8.361 | 12.15 GiB |
| #763 merged with it (`_fitted_shift` on both re-fit paths) | 39 | 254 | 8.361 | 8.85 GiB |

The two step tables are identical on every row but the time column. The owner's machine gave 21 steps /
115 cycles for the same floor (solve-log, "Near-zero pivots and the float32 split"). The container march
has 14–16-cycle solves with `a_min` 0.125–0.25 at steps 12, 14, 19, 20, 23, 28 — the float32-stall
signature `solve.md` flags for a healthy negative pivot — but no probe was run to confirm it. Platform
and jax version were not separated (the container's earlier marches ran jax 0.11.2). The #766 run's wall
clock (3777 s) overlapped a 4-minute unit-test run and is void; the merged run's is 3568 s.

Two arms on the same container, both at `refresh_on_cycles` 3, each stopped once it had answered:
- **jax 0.10.2** (the owner's version; a separate venv, numpy pinned to 2.5.3, `main` `f7fcb6f2`): step 1
  `R` 1.469e-01 against 1.473e-01 under 0.12.0, the paths apart from step 4 (6 cycles against 5); at step
  17 `R` 7.95e-03 with capped solves at 13, 15 and 17. Not the cause.
- **float64 split** (`BFS3D_SPLIT_PRECISION=float64`, #763 at `4cfc758d`, jax 0.12.0): the float32 table
  to every printed digit through step 12, whose solve stops at `max_restarts` 14 (`a_min` 0.25) in both.
  The capped solve is not float32 rounding. `pivot_floor_probe.py` at step 12 was killed out of memory in
  float64 (2 GB of captured constants) and not run to completion in float32.
#768's cadence 4 then gave 17 steps on both the owner's laptop and a 4-core container.

**At the shipped cadence 4** (same container, jax 0.12.0, glibc defaults, nothing else running, one run
each): `main` `1b68caf6` and #763 at `6161f715` give identical step tables (17 steps, 143 cycles, `x_r/h`
8.361); peak `VmHWM` 12.25 against 8.46 GiB (15 s samples), solve 1601 against 1526 s.


### Why bfs3d's probe tangent costs 5.5× pitzDaily's, and what moves it (2026-10-18)

Same machine class (another 4-core Xeon @ 2.1 GHz container, 15 GiB, no swap, 4 KiB pages, transparent
huge pages `madvise`, L3 260 MiB shared with the host), jax 0.11.2, glibc 2.39, `main` `fe772f59` (the
library unchanged by this work), cold-then-warm cache. bfs3d at `state-00032` of the march below (β
0.005, uniform reach 3, 576 probes, batch 8); pitzDaily at its hybrid start (reach 2, 100 probes).
Instruments: `validation/fusion_profile.py` (`FUSION_PROGRAMS=probe`: the session's own probe residual
through `batched_jacobian_matvec`, seeded with the plan's first colours) and `validation/refit_profile.py`.

**The probe's batched program, batch 8** (`fusion_profile.py`, best of 15 untraced calls):

| | pitzDaily | bfs3d | ratio |
|---|---|---|---|
| unknowns | 61125 | 138240 | 2.26 |
| per call, 4 cores / one core | 77.5 / 227 ms | 426 / 1278 ms | 5.5 / 5.6 |
| per tangent, 4 cores | 9.7 ms | 53.2 ms | 5.5 |
| compiled temporary buffer (faulted afresh every call at glibc defaults) | 151 MiB | 816 MiB | 5.4 |
| bytes touched per call (operands + outputs, summed over kernels) | 2.2 GB | 9.9 GB | 4.4 |
| time in partitioned kernels | 78 % | 89 % | |
| parallel efficiency (one-core / (4 x four-core)) | 73 % | 75 % | |

No custom calls in either program, so no `lapack_*`. The time is in copy/bitcast, scatter and gather loop
fusions over `[71872 faces, 8 tangents, (3, 3)]` arrays (bfs3d top kernel: a `select` over
`f64[8, 71872]`, 4.4 %; no kernel above 5 %). Every partitioned kernel has 2 parts: the CPU backend gives a
memory-bound kernel `ceil(sqrt(cores))` = 2 on 4 cores, so ~3 cores is what the program can use by
construction (`XLA_FLAGS` was unset; `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` unset; cpuset 0-3; the work
runs on the 4 `tf_XLAEigen` threads: 97-102 processor-seconds per re-fit there against ~20 on the Python
threads). So the 5.5x is: **4.4x the bytes** (1.95x per unknown: 3D gradients and 3x3 blocks on 2.9x the
faces) **times ~1.25x lower throughput**, ~23 GB/s, plausibly the 816 MiB temporary spilling the L3
that pitzDaily's 151 MiB fits, plus faulting it in. Not fewer threads (the efficiency is the same) and not a
serial host step (below).

**Allocation, separated** (`refit_profile.py`, `state-00032`, β 0.005, mean of 3 re-fits; per-tangent
batch sweep best of 5; one process per arm, back to back; VmHWM includes the sweep's batch-16 program):

| arm | probe | re-fit | probe faults | per tangent at batch 1/2/3/4/6/8/12/16 (ms) | VmHWM / VmRSS after |
|---|---|---|---|---|---|
| glibc defaults | 36.0 s | 48.7 s | 15.06M | 127/94/95/70/61/**52**/60/56 | 10.86 / 8.31 GiB |
| mmap 1 GiB + trim 2^31-1 | 36.0 | 42.3 | 15.06M | 112/87/89/76/63/59/62/52 | 12.13 / 11.33 |
| `MALLOC_ARENA_MAX=1`, default trim | 37.3 | 49.8 | 15.05M | 109/91/87/69/82/59/65/58 | 10.88 / 8.11 |
| jemalloc, `dirty_decay_ms:-1,muzzy_decay_ms:-1` | **26.7** | **33.9** | 3k | 79/68/50/52/51/**38**/42/42 | **13.22 / 13.18** |
| `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` | **29.6** | **39.7** | 66k | 91/66/56/50/51/**42.5**/55/43 | **10.98 / 8.28** |

- **Faulting is ~a quarter of the probe here** (36.0 → 26.7 s with none). It is the whole temporary,
  every call: faults per call grow ~26k (100 MiB) per tangent on ~26k.
- **A single arena alone does nothing**: above glibc's adaptive mmap ceiling (32 MiB) the temporary is
  mapped per call whatever the arena count; #758's fault-free single arena came from the raised thresholds
  with it, and that combination is what retained memory. The thresholds alone cure the host refactor's
  faults (leading refactor 9.2 → 5.1 s) and keep ~3 GiB more resident.
- **jemalloc with decay off is the fastest and unusable for a march here**: +2.4 GiB high-water and +4.9
  GiB resident after one re-fit, on a march that already peaks at 12 GiB at defaults.
- **`glibc.malloc.hugetlb=1` is the footprint-neutral one**: glibc `madvise`s its mapped chunks for
  transparent huge pages, so the temporary faults in 2 MiB pages (66k faults per re-fit), and freed memory
  is still returned. Re-fit −18 %, probe −18 %, VmHWM and VmRSS within 1 % of defaults. Not measured on a
  march (the arm below ran at defaults) and not a `run_case.sh` default (it would make timings
  incomparable with the record); it is the owner's choice for this machine class. It needs THP `madvise`
  or `always`; under `never` it does nothing.
- **Batch 8 is the optimum in every arm** (non-power-of-two batches are erratic: 3, 6, 12), so
  `PROBE_BATCH_SIZE` stands on this machine for this case, as it did on the M3 Pro (2026-08-11): no
  evidence it is a machine property. #758's batch 2 slower than batch 1 (77.8 against 68.8 ms, single
  arena with both thresholds) did not recur in any arm here (batch 2 is 66-94 ms against 79-127).
- **The host steps between chunks** (fill seeds 0.12-0.17 s, scatter 2.5-3.1 s per re-fit, one core)
  are ~8 % of the probe, and overlapping them with the next chunk bought nothing (ledger:
  `solve-refuted-directions.md`, "Overlapping the probe's host de-compression with the next chunk").

**The M3 Pro figure cannot be reconciled from here, and the sentence that is true is this**: the record's
~14 ms per bfs3d tangent is the whole probe's wall (8.2-8.5 s, host scatter included) over 576 probes at
jax 0.10.2, commit `77b03c18`, on an 11-core machine with 16 KiB pages and several times this container's
memory bandwidth; here the same quantity is 62 ms (36 s / 576), of which ~8 % is the host scatter and
~25 % faulting, and the remaining ~40 ms a memory-bound program at ~23 GB/s. Nobody has profiled the
program on the Mac, so bandwidth, page size and the jax version are not separated. The record's pitzDaily
M3 figure (2.5-3.4 s / 100 probes, against 9.7 ms per tangent here) is a whole re-fit's probe line, not a
per-tangent measurement, and is not comparable either.

### One probe program per column field (2026-10-20)

Container: 4-core Intel Xeon @ 2.1 GHz, 15 GiB, no swap; jax 0.12.0, numpy 2.5.4, scipy 1.18.1; glibc
defaults; nothing else running. Base `main` `1290aa11` (in a second worktree) against `claude/per-field-probe`
(`0bf40601`..`bd1a6c42`, library identical across them). Shipped `case.yaml` throughout. The probe's chunks
now follow column fields and each field's program differentiates the residual in that field alone
(`solve-direct-preconditioners.md`, "One program per column field").

**The programs, batch 8, hybrid start** (`fusion_profile.py`, `FUSION_PROGRAMS=probe,probe-fields`, best of
15 untraced calls; bytes are operands + outputs summed over kernels):

| | whole state | u | v | w | p | k | ω | per-field mean |
|---|---|---|---|---|---|---|---|---|
| bfs3d GB per call | 10.39 | 6.91 | 6.91 | 6.91 | 2.77 | 4.30 | 3.51 | 5.22 |
| bfs3d temporary MiB | 829 | 524 | 524 | 524 | 222 | 284 | 244 | |
| bfs3d ms per tangent | 59.3 | 33.8 | 35.6 | 36.9 | 16.1 | 24.3 | 18.4 | 27.5 |
| pitzDaily GB per call | 2.36 | 1.43 | 1.43 | — | 0.63 | 1.05 | 0.85 | 1.08 |
| pitzDaily temporary MiB | 154 | 93 | 93 | — | 44 | 60 | 49 | |
| pitzDaily ms per tangent | 12.1 | 7.1 | 7.0 | — | 3.6 | 5.0 | 4.6 | 5.5 |

Every program shrinks, the velocity ones too (they no longer carry pressure, `k` and `ω` tangents); the
pressure column, which never forms the velocity gradient's `[faces, 8, 3, 3]` tangents, is the cheapest.
Throughput is unchanged (bytes and time fall together), so the program is still memory-bound.

**One re-fit, hybrid start, β 0.005, mean of 3** (`refit_profile.py`, field split's probe now timed through
`_probed_blocks`): bfs3d probe **43.1 → 18.5 s**, whole re-fit 54.0 → 29.4 s, probe faults 15.3M → 7.1M,
VmHWM after the re-fits 8.20 → 8.58 GiB; pitzDaily probe 1.60 → 0.94 s, re-fit 2.53 → 1.64 s.

**Exactness** (`fieldwise_probe_identity.py`): at both hybrid starts the per-field matrix differs from the
whole-state one on 2–8 % of entries by at most 2.4e-16 of its (row field, column field) block's largest
entry, none non-finite; under `XLA_FLAGS=--xla_cpu_max_isa=AVX` (no FMA instructions) the two are identical,
and pitzDaily's equals `main`'s own matrix exactly. The branch's whole-state product (chunks now aligned to
fields) equals `main`'s bit for bit at default flags. At bfs3d's march `state-00015` (β 1e-4, `main`'s shipped march): ≤ 3.6e-16 of the block's largest entry
at default flags, identical under `max_isa=AVX`.

**pitzDaily marches** (`run_case.sh validation/pitzdaily_openfoam/compare.py`, `PITZ_CHECKPOINT_KEEP=100`):

| arm | steps | cycles | re-fits | probe s over the march | step tables | solve | `x_r/h` |
|---|---|---|---|---|---|---|---|
| default flags, cache warm, `main` | 21 | 102 | 21 | 34.3 | | 238 s | 8.0686 |
| default flags, cache warm, branch | 21 | 102 | 21 | 18.9 | first differ at step 12, `R` 8.988e-03 / 8.987e-03 | 233 s | 8.0686 |
| `max_isa=AVX`, `main` | 21 | 102 | 21 | 28.9 | | 239 s | 8.0686 |
| `max_isa=AVX`, branch | 21 | 102 | 21 | 18.3 | **identical** | 247 s | 8.0686 |

pitzDaily's wall moves little: steps 2–21 save ~14 s, and the first step pays ~10 s more even with a warm
cache, for tracing and lowering five probe programs instead of one, once per process (the AVX branch run
paid 12 s on a cold cache).

**bfs3d marches** (project owner, PR #780, 2026-10-21): Apple-silicon Mac (arm64, 11 cores, 18 GB), shipped
`case.yaml`, `BFS3D_CHECKPOINT_KEEP=100`, `validation/bfs3d_openfoam/run_arm.sh`; `main` at `d2522fd8`
(merge-base plus #779, meshing only) against the branch at `295cfd45`, two runs each, alternated; the
machine was shared (load average ~4–10, 8–13 during the branch's first run), so the seconds are indicative:

| arm | coupled solve | probe over the march | refactor | outside the preconditioner (steps 2–17) |
|---|---|---|---|---|
| `main` | 511 s | 131.9 s | 62.3 s | 227 s |
| `main`, again | 481 s | 124.7 s | 58.9 s | 213 s |
| branch, first run (cold compilation cache) | 604 s | 88.7 s | 72.4 s | 326 s |
| branch, second run | 455 s | 70.6 s | 60.6 s | 232 s |

All four take 17 steps and 149 cycles and end at R 5.319e-06, `x_r/h` 8.361, their step tables agreeing in
every column but time to the printed digits. That is not a bit-identity test: `--xla_cpu_max_isa=AVX` is an
x86 flag, so arm64 has no FMA-off arm. The probe falls from ~9–12 s to 5–7 s a re-fit (~45 % over the
march). On the 4-core Linux container the pair could not be run: the container restarted under `main`'s
march twice (steps 15 and 11, not memory), which is where `state-00015` above comes from.

### The re-fit cadence on this machine class, and a corner cell's `k` (2026-10-18/19)

**Read the second table.** The first was taken without the pivot floor, where (PR #766) a near-zero `k`
pivot makes the float32 split wrong by order one, every solve near it runs to its cap and rounding decides
the step's direction: those arms measure where rounding sent each march, not the cadence. The floored
arms (2026-10-19) are the cadence measurement.

#### Without the pivot floor (2026-10-18)

Same container and configuration as the subsection above (`main` `fe772f59`; this branch changes only
harnesses), `validation/run_case.sh validation/bfs3d_openfoam/compare.py`, shipped `case.yaml` but for the
variable named, glibc defaults unless named, one run per arm, back to back, nothing else running, cache warm
after the first. #758's run (`b1648725`, before #757's entry maps and #761) could not serve as the baseline;
it was re-run. Peak is `VmHWM` sampled every 20 s (the first arm was sampled only at step 1, the second not
at all). Cycles count accepted steps' rows (the log's `cum`).

| arm | steps | inner solves | cycles | redone | re-fits (full / cost) | re-fit s (probe / refactor) | solve | final `R` | `x_r/h` | peak |
|---|---|---|---|---|---|---|---|---|---|---|
| `refresh_on_cycles` 3 (shipped) | **failed at step 35** | 60 | 302 | 2 | 13 / 22 | 1710 (1237 / 427) | 3911 s to the failure | 4.56e-04 | (8.361 at step 34) | >= 12.05 GiB |
| 4 | 31 | 64 | 325 | 6 | 13 / 23 | 1779 (1284 / 438) | 4388 s | 7.89e-07 | 8.3611 | not sampled |
| **5** | **19** | 48 | 233 | 0 | 13 / 6 | 944 (663 / 248) | **2439 s** | 2.15e-06 | 8.3611 | 12.63 GiB |
| 5, `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` | 19 | 48 | 233 | 0 | 13 / 6 | 736 (530 / 202) | **2180 s** | 2.15e-06 | 8.3611 | 12.15 GiB |
| 3, `BFS3D_RAMP_STATIONS=8` | stalled; stopped at step 29 | 58 | 195 | 9 | 9 / 24 | 1645 (1188 / 408) | — | 2.49e-03 | — | 12.28 GiB |

- **All arms share steps 1-5 exactly** (only `t(s)` differs), so the march's counts are deterministic here and
  the arms part where the threshold first changes a decision (step 6 for 4, step 9 for 5). The two arms at
  cadence 5 have identical step tables, so the huge-page row is a controlled pair with the row above it.
- **On this machine a re-fit costs ~2.5 steps** (46-53 s against ~20 s without one). The ranking of these
  arms is rounding's (above); the cadence's is the floored table.
- **The huge-page setting on a whole march** (unaffected by the floor question: a controlled pair, identical step tables): solve -11 %, re-fits -22 %, peak 0.48 GiB *lower*, every count
  identical. The setting a march on this machine class can use; not a `run_case.sh` default (it would make
  timings incomparable with the record) -- the owner's call.
- **Fewer stations (8) stalled**: `R` flat at 2.49e-03 from step 26 while the step control doubled the shift
  per step to 4.55 at step 29, line-search steps 0.001-0.008; stopped by hand.

**Why the shipped arm failed, and the 8-station arm stalled: one cell's `k`.** The shipped arm's step-35 cost
re-fit raised `SmoothedHierarchy.refit (level 0): ... 1 of 23040 [cell blocks] are singular` in the trailing
`[k, ω]` hierarchy; the same state refuses the first fit as well (`refit_profile.py` at `state-00034`, the
build at β 2.0), so it is the operator, not the entry-map re-fit. Cell 12107 had `k` 2.1e-17 → 1.1e-17 → 8.0e-18
over steps 32-34 (five other cells also below 1e-12, none as low); there `∂R_ω/∂k` is -9.5e14 and its block is
`[[1.48e-05, 2.5e-23], [-9.47e14, 0.645]]`: lower-triangular, determinant 9.5e-06, invertible, but its
Hadamard ratio `|det| / prod(row norms)` is 6.8e-16, under `_cell_block_inverse`'s 1e-12 cut. That ratio is
invariant to row scaling but **not to column scaling**, contrary to its comment there (measuring `k` in other
units moves it by the same factor). **Since 2026-10-19 the guard is `_block_singularity`, invariant under both,
and passes this block (~1.0); `state-00034` then fits and re-fits** (`refit_profile.py`, 51.6 s); the rest of
this paragraph is the record before that change. The 8-station arm's stall had the mirror cell 3667 at `k` 5.5e-20 (across
the span: both at x = 0.0897 m, ~9h downstream, 0.4-1.8 mm below the upper wall and 1.8 mm from a side wall).
The converged arms kept `k` ~0.72 in cell 12107 and a minimum of 1.7e-05. Not changed here (out of this
work's scope): whether that corner's `k` should be kept from collapsing is the owner's. To re-adjudicate, re-run the shipped arm: two marches at one setting gave identical step
tables here (the cadence-5 pair), and `refit_profile.py` with `REFIT_STATE` at its `state-00034`
reproduces the refusal. **The pivot floor does not cure it**: the same `state-00034` still raises the
guard on `main` `f7fcb6f2` with `PivotFloor(0.15)` in force (2026-10-19, `refit_profile.py`, the first fit
at β 2.0). The cell's `k` pivot is not small against its own pseudo-time diagonal (`|p|` ~2e-06 against
`0.15 d` ~1.3e-06 at either β), so the floor leaves it, and the huge `∂R_ω/∂k` keeps the Hadamard ratio
~1e-15. The floored arms below simply never reached such a state (lowest `k` ~1.8e-05).

#### With the pivot floor, #766 (2026-10-19)

**`case.yaml` moved `refresh_on_cycles` from 3 to 4 on these and the M3 Pro arms below (project owner,
2026-10-19).** "Shipped" in this section's tables means 3.

#766's head before merge, `1e69b4ab` (base `main` `de339a1f`; `case.yaml` sets
`pivot_floor: {kind: PivotFloor, threshold: 0.15}`), run from its own worktree. It merged as `f7fcb6f2`,
whose only further library change deleted the production freeze, off by default and off in these runs, so
the shipped march is the same path (not re-run on `f7fcb6f2`); otherwise as above; glibc
defaults; one run per arm, back to back; peak sampled every 20 s.

| `refresh_on_cycles` | steps | inner solves | cycles | redone | re-fits (full / cost) | re-fit s (probe / refactor) | solve | final `R` | `x_r/h` | peak |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 (shipped) | 32 | 58 | 209 | 0 | 13 / 18 | 1647 (1176 / 422) | 3141 s | 1.72e-07 | 8.3611 | 11.88 GiB |
| **4** | **17** | 55 | 148 | 0 | 13 / 1 | 782 (541 / 210) | **1860 s** | 5.30e-06 | 8.3611 | 11.79 GiB |
| 5 | 28 | 87 | 339 | 4 | 13 / 8 | 1150 (816 / 293) | 3693 s | 6.85e-06 | 8.3611 | 11.96 GiB |

- **All three converge, to the same root.** The shipped march no longer fails on this machine.
- **4 was the fastest of these three samples (0.59x the shipped 3)**: one cost re-fit against 18, at the
  price of nothing: fewer steps too (17 against 32). 5 skipped re-fits the march needed: 87 inner solves, 339 cycles, 4 redone steps.
  The arms share steps 1-6 (the floor moves the residual in the fourth digit from step 5) and part at step 7
  or 8 where the threshold first changes a decision. Single samples of trajectories the threshold moves:
  this ranks three runs; what it does not distinguish is whether 4 wins because it re-fits less or because
  its trajectory happened to avoid the long stall at `R` ~3e-3 that 3 and 5 both went through.
- **The floored shipped march takes 32 steps here against 21 on PR #766's machine** (its march table,
  476-501 s per solve), so with the floor rounding still moves the step count between machines; it no longer
  moves it to a failure.
- **On the M3 Pro (project owner, 2026-10-19, PR #768 comment)**, `main` `a1fd3d1a` (#766's merged library),
  shipped `case.yaml` with the floor, only `BFS3D_REFRESH_ON_CYCLES` changed, one run each, back to back;
  ~5.6 GB of swap already in use at the start and the peak not sampled, so its seconds are uncertain:

  | `refresh_on_cycles` | steps | cycles | re-fits (full / cost) | solve | final `R` | `x_r/h` |
  |---|---|---|---|---|---|---|
  | 3 (shipped) | 21 | 115 | 13 / 7 | 531 s | 4.99e-08 | 8.3611 |
  | 4 | 17 | 149 | 13 / 1 | 475 s | 1.76e-06 | 8.3611 |
  | 5 | 17 | 155 | 13 / 1 | 497 s | 3.18e-07 | 8.3611 |

  4 was fastest there too, by less (0.89x the shipped 3, against 0.59x here): a re-fit costs ~13 s there.
  Arm 4 took the same path class on both machines (17 steps, 13 / 1 re-fits, 149 against 148 cycles); arm 3
  repeated #766's own march (21 / 115) against 32 steps here; arm 5 followed arm 4 to step 17 there and took
  28 steps here, so it is the arm still decided by the machine. The arms part at step 6 there, step 7-8 here.
  Still one sample per arm per machine: whether 4 wins by re-fitting less or by avoiding a stall is not
  distinguished.
- **Without the floor the ranking was 5 < 4 < 3 (failed)**; with it, 4 < 3 < 5. The unfloored ranking was
  rounding's, as PR #766's diagnosis predicts.
