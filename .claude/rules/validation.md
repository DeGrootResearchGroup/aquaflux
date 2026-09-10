---
paths:
  - "validation/**"
---

# Rules — `validation/` (the scientific cases and the study harnesses)

> **Provenance boundary (binding).** As with every rule file: what you read here informs your
> understanding, and none of it may reach the shipped surface. See the root `CLAUDE.md`
> **Comment Convention**.

## What lives here, and why it is fragile

Two kinds of file, with the same failure mode:

- **Cases** (`*/compare.py`) — a full scientific run against a reference solution. Tens of minutes each.
- **Harnesses** (everything else) — single-state probes that measure one question and print a table.
  These are the project's **re-adjudication instruments**: most numbers in the `.claude/rules/solve*.md`
  files were measured with one, and a finding whose harness no longer runs cannot be re-asked, only cited.

**Nothing in any test tier drives these files.** They are too slow for CI and for the fast gate, so
the suite is green whether or not a single one of them still works. That is the whole problem this
file exists to address.

## ⚠️ THE RECORDED FAILURE — one case, three simultaneous breaks, suite green

Found 2026-08-16, all in `validation/pitzdaily_openfoam/compare.py`, none detected by anything:

1. **A settings object was introduced and the case was not updated.** `RefreshPolicy` replaced four
   loose keyword arguments; the case still passed a bare `precondition_step` callable, so
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

## ⚠️ pitzDaily's SHIPPED PRECONDITIONER STOPPED MARCHING IT — the default moved to `simplesmooth` (2026-08-22)

Under `PITZ_FLOW_INVERSE=petsc` (the case's default until this date) `pitzdaily_openfoam/compare.py`
collapses at the **first step of the second Reynolds rung**: `alpha` 0, `beta` escalating 0.5 → 2 → 16
through the whole ladder, the residual rising 1.674e-01 → 5.754e-01 → `inf`. Reproduced three times,
including on a tree carrying **no local change at all**, with the step tables bit-identical.

**The default is now `simplesmooth`** — the same leading inverse the 3D sibling defaults to, and for
the same reason: an incomplete factorization's behaviour on this saddle depends on the elimination
**order** and the **fill**, neither of which is predictable in advance, while a SIMPLE-smoothed
hierarchy never eliminates the matrix at all. This case's own record already carries the extreme
version of that sensitivity — a zero-fill factorization going from *amplifying* a residual 5.5× per
sweep to contracting it, on nothing but a reordering. `petsc` and `hostilu` both remain reachable and
both remain measured; what is no longer defensible is either as the default.

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

## The obligation (binding)

**A change to the library's public surface is not complete until it has been checked against the cases
that call it.** Concretely, when you change a signature, a default, a settings object, a type, or the
shape of a seam:

1. **Run the static guard** — it is in the fast gate and costs milliseconds:
   `pytest tests/unit/test_validation_api.py`. It checks that every name a case imports still exists
   and every literal keyword it passes is still accepted — **including keywords passed to a method on
   an imported class** (`CoupledRANS.build(...)`, `SSTTurbulence.build(...)`, `MomentumContinuity
   .build(...)`), which is how every case constructs its assemblers — and that no function in a case
   reads a module global that only one branch of an `if`/`try` binds.
2. **Then use judgement on what the guard cannot see** (below), and if the change plausibly reaches a
   case, *run that case* — `validation/run_case.sh <case>` — before considering the change done.
3. **When you change the march machinery, ask whether the OTHER case should get it too.** Every
   improvement in this project has been developed on whichever case was in front of someone, and the
   sibling has repeatedly been left behind. Carrying it across is part of the change, not a follow-up.

The `.githooks/pre-commit` reminder raises this whenever a commit touches `aquaflux/`. It does not
block: whether a given change reaches a case is exactly the judgement a script cannot make.

## ⚠️ What the static guard CANNOT catch

`tests/unit/test_validation_api.py` is a static check — it reads the cases with `ast` and never builds
a mesh. **A green run there does not mean the cases work.** It is blind to:

- **Semantic breaks.** A parameter that still exists and now means something different, a default that
  moved, a type that changed under an unchanged name. Break 1 above is exactly this, and the guard
  would **not** have caught it.
- **Anything behind `**kwargs`.** `solve_coupled` takes `**continuation_kwargs` and forwards them to
  whichever builder it is given, so every keyword is "accepted" *statically* and none is checked here —
  and this is the main entry point. ✅ **Narrowed 2026-08-20 (#278): the case where there is no builder
  to forward to is now a `TypeError` at the call rather than silence.** Given an explicit `continuation`
  or a `RefreshPolicy(builder=...)`, `method` / `reference_state` / `**continuation_kwargs` are refused
  by name instead of dropped — which is how `precondition_step=` used to vanish on its way to a
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

## Known API gaps these cases exposed

- **`positivity_floor` was a parameter of `coupled_amg_continuation` ALONE — FIXED, see the entry
  below; it is now on all four builders.** `pitzdaily_openfoam/compare.py` spells it
  `K_POSITIVITY_FLOOR`, matching the sibling case so a future diff lines up.
- **✅ FIXED 2026-08-19 — this was a behavioural gap between the arms, and it was wider than recorded
  here.** It said `coupled_lu_continuation` marched with no k-positivity limiter while
  `coupled_amg_continuation` always carried it. True, and **three** of the four builders lacked it, not
  one: `coupled_continuation` and `mass_flow_coupled_continuation` too, and even the AMG builder only
  wired it on its dual-time branch. All four now route through `_coupled_step`, which passes
  `step_limit=positive_k_limit(...)` on both branches;
  `test_every_continuation_builder_installs_the_same_globalization` fails if one loses it. The confound
  this warned about — an LU-versus-AMG comparison being between a guarded march and an unguarded one —
  applied to every cross-builder comparison on record, so treat any of them with that in mind. The
  confound still applies to archived runs.
- **✅ FIXED 2026-08-20 (#282) — the SAME defect recurred TWICE MORE on the same four builders, with the
  shared tail already extracted.** `forward_rtol` / `forward_restart` / `forward_max_restarts` were on
  `coupled_amg_continuation` alone, so the *default* path — what `solve_coupled` builds when nothing is
  passed — stopped its forward solve on the plain 2-norm that same builder's docstring calls effectively
  blind to the flow block. And `velocity_shift_parts` was on the two block builders only, absent from
  exactly the monolithic path it was written for. Both are now on all four, and the stopping *measure*
  belongs to `_coupled_step` rather than to any builder: it is the march's own progress measure, so a
  solve cannot converge in a quantity the march does not read. ⚠️ The advice recorded here — that
  `forward_rtol`/`restart`/`max_restarts` were "reachable via `forward_solver=`" — was true and a trap:
  building a solver to move the tolerance silently replaced the *stopping measure* too, a far larger
  change. Pass the parameter.
- **`_mis_aggregate`'s return annotation is stale** — it says `tuple[np.ndarray, int]` and returns
  three values (labels, roots, count). Cost one debugging cycle.

## ⚠️ Traps when writing a harness (each one produced a wrong result here)

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
  above came from `tests/integration/test_coupled_lu.py`, where it is correct — that fixture builds
  `CoupledRANS` with no transform.
- **⚠️ A PIVOT CENSUS MUST READ THE FACTOR, NOT THE OPERATOR HANDED TO IT.** Every consumer here
  symmetrically equilibrates before factorizing, which forces the *operator's* diagonal to magnitude
  exactly 1 — so a census written as `matrix.diagonal()` reports "zero negative pivots, min |pivot|
  1.00" for every arm at every shift, including arms whose sweep diverges by 1e+59. It looks like a
  finding ("the pivots are all healthy, so it is not a pivot problem") and it is a measurement of the
  conditioning transform. This shipped in a sweep on 2026-08-17 and a conclusion was drawn from it
  before being retracted. Use `Ilu0.pivots`, which exists for this; and note it stores the pivot
  itself where PETSc stores its **reciprocal**, so a census ported between the two reports the inverse
  of what it claims.
- **Print one line per outer step, flushed.** A harness that collects reports and prints at the end is
  indistinguishable from a hung one, and cost thirty minutes of a run that could not have converged.
- **State the operating point before measuring.** A harness whose banner prints `? cells` is one whose
  author does not know what it is measuring; the mesh size decided the whole question in that instance.
- **A setting the banner prints must be a setting that is in force.** Printing an intended value that
  the builder never received is worse than printing nothing.
- **⚠️ Do NOT gate a loaded checkpoint on its own recorded `residual_norm`.** That number is whatever
  measure the march was *steered* by, and both cases march with `scaled_norm=True` — a row-equilibrated
  norm, not a Euclidean one. Comparing the two rejects a perfectly good state: `bfs3d`'s `state-00069`
  records `2.64e-06` and computes `1.04e-03` under `jnp.linalg.norm`, a factor of **395** that is
  entirely the change of measure. Gate against the case's **own self-start** in whichever single norm
  the harness uses — both ends then move together, and a genuine configuration mismatch (which moves the
  residual by orders) still trips it.
- **A saved `.npz` is not necessarily a checkpoint.** `pitzdaily_openfoam/ilu0_remedy_state.npz` is the
  case's *self-start*, cached only so repeated runs skip rebuilding it. Measuring "at the converged root"
  against it silently answers a different question — and the two differ enormously: at the self-start the
  zero-shift coupled Jacobian is nearly singular (smallest pivot `1.3e-12` against a matrix 1-norm of
  `278`), so even a complete LU is not an accurate inverse of it, while at a converged root the shipped
  field split solves the same zero-shift operator to `6e-09`. Read what wrote a state before trusting it.
- **`bfs3d`'s shipped `COLUMN_REACH = (3,3,3,3,2,2)` is licensed for the FIELD SPLIT ONLY.** A flow-first
  split never applies `dR_flow/dturb`, so it never touches the shortened k/ω columns. A **monolithic**
  factorization (the complete LU) does apply them, and a short colouring does not truncate a column —
  it folds far couplings onto near entries. Probe every arm at a uniform reach whenever a monolithic arm
  is in the comparison, or the arms are not being compared on the same matrix.

## ⚠️ `run_case.sh` GUARDS AGAINST A SECOND CASE AND NOT AGAINST A TEST TIER (measured 2026-09-09)

**The runner's mutual exclusion is over *cases*. `tools/fastgate.sh` is not a case, so nothing stops it
starting on top of a running one — and the fast tier is unambiguously a heavy job on this machine.** It
runs `pytest -n auto --dist loadfile`, which on the 11-core, 19 GB machine these cases are measured on
peaks around **6.4 GB** and, launched beside a live pitzDaily march, drove the load average past **22**
with free memory at **0.73 GB**. It happened **twice in one evening**, from two different worktrees, one
minute and four minutes into someone else's case — and the two gates took **17:25** and **13:00** against
a documented 6:34-8:53 for that tier. Every wall-clock number in a case log overlapping such a window is
contaminated, and so is the gate's own.

**What contention does and does not move — and this is the useful half, because it needs no clock.** The
cleanest instance is a matched pair in ONE worktree at ONE commit, differing only in machine load, whose
two logs align line for line. Both print rung 1's closing row on their own line 738:

    |   28 |    895 | 0.0050 |  1 |   2 | 7.214e-06 | 1.000 |     |     <- loaded machine
    |   28 |    194 | 0.0050 |  1 |   2 | 7.214e-06 | 1.000 |     |     <- quiet machine

**Every column is identical except `t(s)`, which differs 4.6x** — step count, `beta`, inner count, cycles,
`|R|` to all four figures, `a_min`. A second, cross-session pair agrees the same way on whole-march totals
(**69 steps / 417 cycles**, `x_r/h` **8.069 against 8.0686**). Steps, cycles, escalations, line-search
clips and the converged root are deterministic and contention cannot move them. Wall clock is a different
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
| 3 | none — it starts after the gate ends | 194 s | *pending* | *the control* |

**That graded structure is the evidence, not the endpoint ratio.** A confound would have to be graded the
same way to explain it, and the one arm that overlapped nothing is the control. (Its quiet half was still
marching when this was written. The extrapolation looked like noise; it is deliberately **not** recorded
here, because an extrapolated control is not a control — fill it in when the run lands.)

⚠️ **And the cost is MUTUAL, which is the part that argues for a guard rather than a convention.** That
gate took **17:25** against a documented 6:34-8:53 for the same tier — it was slowed by the case as surely
as the case was slowed by it. Run in sequence the two jobs are roughly 8 and 9 minutes; run together the
gate alone took 17. **Concurrency here is not a trade of latency for throughput, it is a loss on both.**

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

- **The compiled ILU(0) kernel is a gitignored artifact.** A fresh worktree silently runs the pure-Python
  twin, and its timings are incomparable to any other checkout's until `tools/build_ext.sh` has been run
  there. `ilu0.COMPILED` says which is live and both cases' banners print it — check it, because nothing
  else will. (This is not hypothetical housekeeping: the worktree this entry was written in reported
  `COMPILED = False` while it was being written.)
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

**Until a guard exists, the check is manual and it is on the person starting the *tests*, not the case:**
run `validation/run_case.sh --status` before `tools/fastgate.sh`, not only before another case. Checking
once and launching twice is the specific way this failed — the status was clean when the case was queued
and stale by the time the gate followed it.

⚠️ **Do not read this as a knowledge gap to be closed by documentation.** Three sessions on the evening
this was recorded all knew the one-heavy-job-at-a-time rule and it happened anyway, because the rule is
enforced for one pair of jobs and merely known for the other. The durable fix is to make the collision
unavailable — teach the gate to consult the machine-global run-file and refuse or warn — and this entry
exists to stop the wall-clock numbers being trusted in the meantime, not to substitute for that.

## `bfs3d_species` — the newest case, and what it depends on

A passive tracer on `bfs3d_openfoam`'s flow (see `.claude/rules/transport.md` for the two-arm design
and the `Sc_t = 1` choice). Its dependency structure is unusual and is the thing to know:

- **It carries NO mesh and NO flow of its own.** `of_case/run_of.sh` assembles a scratch case from
  `bfs3d_openfoam/of_case`'s `polyMesh` and converged time directory, and `compare.py` loads that
  case's `build_case` **by path** (`importlib`), because both cases have a module named `compare` and
  a plain `import compare` resolves by whichever path order happens to be in force.
- **Its own-flow arm reads the flow case's rolling checkpoint** rather than re-marching. ⚠️ That
  checkpoint predates the production-limiter default moving OFF, so it is *not* a root of today's
  coupled residual. This does not invalidate the arm: what a transported scalar needs of a flux is
  that it close discretely, which `compare.py` measures directly rather than inferring from the
  state's residual.
- **`tests/integration/test_bfs3d_species.py` is the one test that reaches into `validation/`.** Every
  test there skips when the case data is absent, since none of it is in the repository. It pins the
  properties the comparison rests on — sub-patch injection, conservation on the imported flux,
  boundedness, mixing — so a break shows up as a failure rather than as a meaningless log.

⚠️ **Do not edit files under `validation/` while the fast gate is running.**
`tests/unit/test_validation_api.py` statically reads every case, so it will parse a half-written file
and fail for a reason that has nothing to do with the change under test. This happened once and cost a
28-minute gate run.

## Recovering a converged state (both cases)

Both `compare.py` files take `checkpoint_dir` and write a rolling per-step state through the shared
`StateCheckpointer` + `combine_observers` (`PITZ_CHECKPOINT_KEEP` / `BFS3D_CHECKPOINT_KEEP`, default 3;
`main()` writes to `<case>/checkpoints/`). This matters because **a converged state otherwise exists only
inside the process that computed it** — and the adjoint's operator is the Jacobian at that root, which is
the one operator a march never exercises, since the continuation ramps the shift and the preconditioner
is additionally floored. Without a checkpoint, every question about the zero-shift operator costs a full
re-march to ask. The `zero_shift_arms.py` / `zero_shift_adjoint.py` harnesses are the consumers.

## ⚠️ THE SLOW TIER FAILS LOCALLY BUT NOT ON CI — an ENVIRONMENT difference (observed 2026-08-19)

Three tests fail here with `ImplicitNewtonSolver did not converge`, **at `dd1ea73` itself** (verified by
stashing all local work and re-running each one, not inferred) — while `main` is **green on GitHub**:

- `tests/integration/test_coupled_amg.py::test_amg_solve_converges_and_matches_the_block_preconditioned_solve`
- `tests/integration/test_coupled_amg.py::test_amg_adjoint_matches_finite_difference`
- `tests/integration/test_coupled_field_split.py::test_the_split_continuation_converges_to_the_monolithic_fixed_point`

So this is not a code defect — it is a **platform-dependent** one, and that is worse in one specific
way: it makes the local slow tier useless as a gate without telling you. A branch that genuinely breaks
something in that tier is indistinguishable from this baseline.

**Known environment differences** (local against CI): Python **3.13 on arm64** here versus **3.11/3.12
on x86_64 Linux** in CI (`.github/workflows/ci.yml` installs `.[test]`, unpinned beyond
`requires-python >= 3.10`), hence different BLAS and different floating-point summation order.

**The likely mechanism, stated as a hypothesis and NOT yet measured:** these solves are marginal against
their `max_steps = 40` budget, and a platform-level difference in the last bits tips them over. That is
the same shape as the constrained mass-flow adjoint, where a **6.6e-12** relative difference in the warm
state flipped a transpose solve from converging to raising — measured, on this machine, the same day.
**To settle it, re-run one with `max_steps` raised and read how far past 40 it needs**; a couple of steps
means fragility to be given headroom and documented, while an order of magnitude means a real
platform-specific defect.

**Binding until then: run the three at `HEAD` before believing any local slow-tier failure is yours.**
Not doing so cost real time on 2026-08-19 — the failures were attributed to a local change, a guard was
reverted on that basis, and a comment was committed claiming the revert was "measured" when the baseline
had never been run.
