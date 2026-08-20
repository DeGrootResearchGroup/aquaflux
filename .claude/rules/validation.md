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

## The obligation (binding)

**A change to the library's public surface is not complete until it has been checked against the cases
that call it.** Concretely, when you change a signature, a default, a settings object, a type, or the
shape of a seam:

1. **Run the static guard** — it is in the fast gate and costs milliseconds:
   `pytest tests/unit/test_validation_api.py`. It checks that every name a case imports still exists
   and every literal keyword it passes is still accepted.
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
  whichever builder it is given, so every keyword is "accepted" there and none is checked — and this is
  the main entry point.
- **Behaviour.** It cannot tell a converging march from one that crawls, which is how a case can be
  runnable and useless at the same time (the "fallen behind" failure above).

## Known API gaps these cases exposed

- **`positivity_floor` is a parameter of `coupled_amg_continuation` ALONE.** The default builder
  (`coupled_continuation`) and the complete-LU and threshold-ILU builders expose neither it nor the
  `step_limit` it would be set on, so a study arm built on one of those cannot switch the k-positivity
  safeguard on at all, and a march that meets the ratchet there has no way out of it.
  `pitzdaily_openfoam/compare.py` reaches it only because that case now uses the AMG builder; the
  symbol there is `K_POSITIVITY_FLOOR`, matching the sibling case so a future diff lines up.
- **✅ FIXED 2026-08-19 — this was a behavioural gap between the arms, and it was wider than recorded
  here.** It said `coupled_lu_continuation` marched with no k-positivity limiter while
  `coupled_amg_continuation` always carried it. True, and **three** of the four builders lacked it, not
  one: `coupled_continuation` and `mass_flow_coupled_continuation` too, and even the AMG builder only
  wired it on its dual-time branch. All four now route through `_coupled_step`, which passes
  `step_limit=positive_k_limit(...)` on both branches;
  `test_every_continuation_builder_installs_the_same_globalization` fails if one loses it. The confound
  this warned about — an LU-versus-AMG comparison being between a guarded march and an unguarded one —
  applied to every cross-builder comparison on record, so treat any of them with that in mind. The
  original text follows, because the confound it names still applies to archived runs. The tail already accepts all
  four parameters (`refresh_on_cycles`, `inner_refresh`, `cycle_budget`, `step_limit`/`step_projection`);
  the LU builder simply does not forward them. `forward_rtol`/`restart`/`max_restarts` ARE reachable, via
  `forward_solver=`.
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
