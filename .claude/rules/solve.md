---
paths:
  - "aquaflux/solve/**"
---

# Rules — `aquaflux/solve/` (Newton + implicitly-differentiated linear solve)

> **Provenance boundary (binding).** This file cites the C++/Fortran precursors to inform
> *your* understanding — that is its job, and why it loads into your
> context. Per the root `CLAUDE.md` **Comment Convention**, none of that provenance may
> reach the shipped surface (`.py` comments/docstrings, `docs/`): cite the *math*, never the
> reference code, the `.claude/` rules, the design notes, or the author's own papers.

Drive the residual to zero and expose an exact, iteration-count-independent adjoint.
Governed by the root `CLAUDE.md` Engineering Principles.


## ⚠️ "native" was renamed away (2026-08-20) — there is no such symbol

The word meant **two opposite things** and neither described a method:

- *ours, not PETSc* — `NativeSimpleInverse`, `NodalNativeInverse`, `NativeHierarchyInverse`,
  `native_saddle_inverse`, `native_nodal_inverse`, `FLOW_INVERSE="native"`;
- *run the whole solve natively on the host, **in PETSc*** — `AmgVCycle(native=True)`,
  `has_native_solve`, `is_exact_native`, `native_forward_solve`.

The family is now named by its **level smoother**, which is the only thing its members differ in:

| was | is |
|---|---|
| `NativeHierarchyInverse` | `HierarchyBlockInverse` |
| `NativeSimpleInverse` / `native_saddle_inverse` | `SimpleSmoothedInverse` / `simple_smoothed_inverse` — *the factory is now the value object `SimpleSmoothed` (#371)* |
| `NodalNativeInverse` / `native_nodal_inverse` | `JacobiSmoothedInverse` / `jacobi_smoothed_inverse` — *the factory is now the value object `JacobiSmoothed` (#371); `air_inverse` is `AirReduction`* |
| `HostVCycleInverse` / `host_ilu_inverse` | `IluSmoothedInverse` / `ilu_smoothed_inverse` — *deleted 2026-09-13 with the ILU(0) kernel, #371; neither name exists* |
| `AmgVCycle(native=)` / `has_native_solve` / `is_exact_native` / `native_forward_solve` | *(deleted 2026-09-13, #371 — there is no host exact forward solve, and none of these four names exists)* |
| `solve/native_inverse.py` / `solve/host_vcycle.py` | `solve/hierarchy_inverse.py` / `solve/ilu_inverse.py` (*the latter deleted 2026-09-13, #371*) |
| `BFS3D_FLOW_INVERSE=native` | `BFS3D_FLOW_INVERSE=simplesmooth` |
| `BFS3D_TURBULENCE_INVERSE=native` | `BFS3D_TURBULENCE_INVERSE=jacobi` |

Recorded measurements in these files that said "the native arm" now say "the traced arm" — *traced*
(runs inside JAX, on device) against *host* is the distinction the old word was reaching for, and it
is the one that matters for a GPU. `hostilu` and `petsc` arm values were unchanged (both arms were removed 2026-09-13, #371): both already say
what they are. See the shipped `docs/preconditioning.md` for the user-facing description.

**The three value objects share a public base, `BlockInverse` (`block_inverse.py`), which in turn
derives from `SettingsValue` (`settings_value.py`, #371, 2026-09-14).** `SettingsValue` is the one home
of "a frozen dataclass whose `None` fields are unset, and whose `settings()` are the set ones" — the
coupled-preconditioner specs in `turbulence/preconditioner_spec.py` derive from it too, so the
comprehension is written once rather than per configuration family. `BlockInverse` is public so a
configuration can *require* a value rather than an arbitrary `(block, n_fields)` callable
(`FieldSplit` does). There is no `_BlockInverseSpec`; that was its private name for one commit.
**An abstract family base is refused at construction, in one place: `SettingsValue.__new__` (#398
review).** `BlockInverse` and `flow.VelocityBlock` (and the private `_ConvectionVelocityBlock`) derive from
`abc.ABC` with an abstract method, and constructing any such base raises `TypeError` naming the family's
public concrete members (found from `__subclasses__`, private and abstract ones omitted). Before, a bare
`BlockInverse()` or `VelocityBlock()` passed the `isinstance` refusals on `FieldSplit` / `BlockDiagonal` /
`BlockPreconditioner.build` and failed only at build, on an empty `NotImplementedError` — for the velocity
block after the pressure Schur was already built. A new family base gets this by deriving from `abc.ABC`
and marking its build hook abstract; do not add a per-family `__new__`.
**How two partial settings values combine is written once too: `solve.filled_from(value, base)` (#399
review), and so is how one BECOMES the arguments of the class it configures — `_supplied(target, fields)`
and `_merged(target, settings, fields)`, moved from `continuation.py` into `settings_value.py` with
`RootSolveSettings` (#428, 2026-09-22), since `implicit.py` cannot import `continuation.py` (that is the
direction the import already runs).** Each `None` field of `value` takes `base`'s. `SettingsValue.filled_from` delegates to it, and
so do the settings objects that are `equinox` modules (`Globalization.filled_from` and `with_defaults`,
`turbulence.CoupledShiftSettings.filled_from`); before, `CoupledShiftSettings`, `LinearSolveSettings` and `DualTimeLoop` each
carried an identical body and `Globalization.with_defaults` a fourth, and because the Reynolds merge
(`merged_march_options`) recognizes a mergeable value by its `filled_from`, `Globalization` alone was
still replaced whole per point. ⚠️ **`None` cannot reset a shared field to its default** — it means "take
the base's"; write a numeric default out, and keep a setting whose default is `None` itself
(`CoupledShiftSettings.velocity_parts`) out of the shared options if a point needs it back.

**`SettingsMapping` (`settings_mapping.py`, #391) writes and reads any family of such values as a nested
plain mapping** — `kind` = class name, a field equal to its dataclass default omitted, nested values as
nested mappings, tuples as lists — and refuses an unknown kind or field with the path to it. It is
generic and parses nothing (no YAML dependency, per `pyproject.toml`'s note); the coupled-preconditioner
wrappers live in `turbulence/preconditioner_spec.py` (see `turbulence.md`). Omission is decided by
**equality with the default, not by `None`**, so a field whose default is a sentinel (`BlockDiagonal.method`)
round-trips, and a required field is always written. ⚠️ A value whose default is compared by `!=`
must compare by value — an array-valued field would need its own rule; none exists today.

## Responsibility
- A Newton driver on `R(state, params) = 0` using the AD Jacobian (JVP/VJP), and a
  linear solve wrapped so its gradient comes from **implicit differentiation**, not by
  unrolling Krylov iterations onto the tape.

> **⚠️ The guard scanned `aquaflux/` ONLY, and the boundary eroded where it did not look (extended
> 2026-08-15).** `validation/` held **16** deep imports of names `__all__` already advertises —
> `restart_cycles`, `MonolithicAmgPreconditioner`, `symmetrically_equilibrate`, the probe-plan types —
> so "this cannot erode silently" was true of the directory under guard and false next door. Those
> harnesses are the project's re-adjudication instruments, so a rename inside a preconditioner breaks a
> *study* rather than a test: the more expensive failure, and the later-discovered one. The rule now
> enforced over `validation/` is the one needing no API decision — **if the package exports it, import
> it from the package** — and the harnesses' genuinely-internal reaches are an explicit list
> (`VALIDATION_INTERNAL_REACHES`) asserted in **both** directions, so an unlisted reach fails and so
> does a stale entry. Note the guard immediately found four violations an ad-hoc regex sweep had
> missed, because it parses imports rather than matching lines.

> **`solve/__init__.py` is the API boundary (binding, #48).** Everything consumable from this
> package is re-exported there, and **library code imports `from aquaflux.solve import …`, never
> `from aquaflux.solve.<submodule> import …`**. A name absent from `__all__` is internal (reach for it
> only from that submodule's own unit tests, which are exempt). When you add a public entry point,
> export it in the *same* change — a partial surface is what pushes consumers into deep imports and
> makes `__init__` stop describing the package (the block preconditioner once pulled nine names
> straight out of `solve.multigrid` while `__all__` advertised only the smoothed-aggregation third of
> the AMG toolkit). `tests/unit/test_solve_api.py` pins both halves and fails with the offending
> file named, so this cannot erode silently.
- Milestone 0: a single scalar diffusion system; the plumbing must generalize to the
  coupled p–U block later without redesign.

## ⚠️ RENAMED 2026-09-15 — grep this table before believing an old name is missing

Phase 4 of the Newton-loop unification renamed the solver vocabulary, because four words each meant
four to seven different things: "forward" named the strategy, the loop, the Krylov settings *and*
"not differentiable"; "solver" named the nonlinear driver and the linear one; "continuation" named a
method and any strategy at all. **Every old name below is gone from the tree**, so a search for one
lands here rather than on nothing — which is the point of keeping the table (issues, PR bodies, the
archived march logs and any private notes still use the left column).

| was | is | note |
|---|---|---|
| `ImplicitNewtonSolver` | `RootSolver` | "implicit" read as implicit time-stepping. **Not** `NewtonSolver`: a class of that name was deleted in #102 and the record of that deletion is still binding, so the name would collide with it |
| `ForwardStep` | `NewtonStrategy` | the protocol; `ShiftedForwardStep` → `ShiftedNewtonStrategy` |
| `forward_step.py` | `strategy.py` | the contract module (`NewtonStrategy`, `StepOutcome`, `StepReport`, `StepControl`) |
| `forward_march` | `newton_march` | there is only one march now (phase 3), and "forward" no longer distinguishes it from anything |
| `RootSolver.forward_step=` | `RootSolver.strategy=` | it holds a `NewtonStrategy` |
| `solve_coupled(continuation=)` | `strategy=` | it accepts any strategy, not only a continuation; `**continuation_kwargs` → `**strategy_kwargs` |
| `ForwardStep.default_solver()` | `NewtonStrategy.linear_solver()` | "solver" alone means the nonlinear driver everywhere else |
| `RootSolver.solver=` | `linear_solver=` | same reason; `adjoint_solver` was already unambiguous and is unchanged |
| `PseudoTransientStep.forward_solver=` | `krylov_solver=` | the caller's override for the inner linear solve; `linear_solver()` is the method that resolves it |
| `coupled_step(forward=)`, `ForwardSolve` | `linear_solve=`, `LinearSolveSettings` | the Krylov settings, which are neither forward-mode nor the march |
| `precondition_step` | `refresh_preconditioner` | a per-step hook, not a step |
| `_ForwardSolveRegime`, `_BLOCK_FORWARD`, … | `LinearSolveRegime` (public, `solve/shifted_step.py`; was `_LinearSolveRegime` in `turbulence/coupled.py` until 2026-09-19), `_BLOCK_LINEAR_SOLVE`, … | the regimes of the same inner solve; the per-family constants stay in `turbulence/coupled.py` |
| `RootSolver(rtol=, atol=)`, `solve_coupled(rtol=, atol=, scaled_norm=)`, `solve_coupled_mass_flow(rtol=, atol=)` | `convergence=Convergence(measure=…, rtol=…, atol=…)` (#370, 2026-09-16) | the measure and its tolerances as one value; `RootSolver` also takes `measures=` for a structured residual |
| `RelaxationSchedule.relaxation(…)` | `shift_strength(…)` (#373, 2026-09-22) | it returns the shift strength β, which grows with damping, where an under-relaxation factor in `(0, 1]` shrinks with it — the word ran in both directions. The classes keep their names: `SwitchedEvolutionRelaxation` is the literature's |
| `solve_segregated(rtol=)` | `increment_tol=` (#373, 2026-09-22) | it is a tolerance on the **state increment** (the largest per-field relative change over a sweep), not on a residual, and it sat beside `solve_coupled(rtol=)`, which is one |
| `MaterializedJacobian(beta_floor=)`, `beta_tracking_refresh(beta_floor=)` | `refit_beta_floor=` (#373, 2026-09-22) | the shift the **inverse is re-fitted at**. `Globalization.beta_floor` (the march's own) keeps its name; the step control's `beta_min` is a third floor and is unchanged |
| `reused_flow_solve(max_steps=)`, `bulk_velocity_flow_solve(max_steps=, solver=)`, `scalar_pseudo_transient_solve(max_steps=, rtol=, atol=, solver=)` | `root_solve=RootSolveSettings(max_steps=…, convergence=…, linear_solver=…, adjoint_solver=…)` (#428, 2026-09-22) | one value on all three, so a setting reachable from one is reachable from all — `adjoint_solver` was reachable from none |

**Deliberately NOT renamed**, so do not "finish the job":

- **`stepper()`** stays. It is the only use of that word in the package, so it is already unambiguous,
  and `step()` would collide with `Globalization.step()` and `coupled_step()`, which build the step
  *object* — one word, two jobs, which is the defect this exercise removes.
- **`DampedNewtonStep`, `PseudoTransientStep`, `DualTimeStep`, `StepOutcome`, `StepReport`,
  `StepControl`** keep "Step". Across all of them "step" means one outer iteration, consistently; the
  full `Iteration` vocabulary was considered and rejected as churn without a gain in clarity.
- **`momentum_continuation`, `mass_flow_coupled_continuation`, `coupled_step`** keep their names: they
  build genuine pseudo-transient continuations, which is the word used correctly.

## Index — where the detail lives

**This file used to be one 8,500-line document; it is now split by subsystem, each part scoped so it
auto-loads only when you touch the files it actually governs** (mirroring the root `CLAUDE.md`'s own
per-package split). What stays here — always loaded on any `aquaflux/solve/` edit — is the package-wide
contracts, the current configuration, the general Newton/adjoint binding decisions, the gates, and the
testability seam. Everything subsystem-specific moved out:

| File | `paths:` | Covers |
|---|---|---|
| `solve-direct-preconditioners.md` | `lu_preconditioner.py`, `sparse_jacobian.py` | The monolithic complete-LU preconditioner (and the now-deleted ILUT it once shared a family with), and the shared frozen-host contract |
| `solve-amg-multigrid.md` | `amg_preconditioner.py`, `multigrid.py`, `hierarchy_inverse.py` | The monolithic AMG coupled PC, the traced multigrid, faithful smoothed aggregation, and `multigrid.py`'s own binding decisions |
| `solve-flow-block.md` | `saddle_multigrid.py`, `shift_basis.py` | Traced preconditioning of the `[u, v, w, p]` saddle — current status only |
| `.claude/notes/solve-flow-block-log.md` | *(never auto-loads)* | The full dated investigation behind the flow block, including qualified/retracted findings |
| `solve-field-split.md` | `field_split.py` | The block-triangular field split (saddle plus two transported scalars) |
| `solve-globalization.md` | `strategy.py`, `continuation.py`, `step_control.py`, `retry.py`, `relaxation.py`, `line_search_growth.py` | Forward-step architecture, pseudo-transient continuation, line search — current status only |
| `.claude/notes/solve-globalization-log.md` | *(never auto-loads)* | The dated investigation behind the globalization architecture |
| `solve-march.md` | `march.py`, `march_log.py`, `checkpoint.py` | The observed march: `newton_march`, triggers, controls, logging |
| `.claude/notes/solve-refuted-directions.md` | *(never auto-loads)* | A cross-cutting ledger of closed/refuted ideas — check here before proposing something that sounds already tried |

The two `-log.md` files and `solve-refuted-directions.md` live in **`.claude/notes/`, outside the
auto-loaded `.claude/rules/` tree, and never auto-load** — they are tracked (so a finding can be
re-adjudicated later, per the root `CLAUDE.md` rule that findings belong in tracked files, not memory)
but deliberately kept out of the auto-loaded path so routine solver work does not pay for the full
investigation history. Read them deliberately: before re-investigating a subsystem, or before
proposing an idea that might already be closed.

⚠️ **They used to sit in `.claude/rules/` with no `paths:` frontmatter, and that did NOT keep them out
— it did the opposite.** A rules file without `paths:` is not scoped to nothing, it is scoped to
everything: all three loaded into **every** session in the repository, whatever was being worked on,
at ~285 KB (roughly 70k tokens, over a third of the context window) before the first tool call. Moving
them out of `.claude/rules/` is what makes the "never auto-loads" claim true. **A file under
`.claude/rules/` must carry `paths:`; reference-only material belongs in `.claude/notes/`.**

## Where new content goes (binding — read before adding a finding)

The table above says where content *is*; this says where NEW content *goes*, so the split does not
silently regrow into another 8,500-line file. Four rules:

1. **Route by the file you are documenting, not by where the last entry on the topic happened to
   land.** Find the row above whose `paths:` covers the `.py` file your change touches, and write
   there. A finding that is genuinely package-wide (the Newton driver, the linear-solve contract, an
   adjoint/implicit-diff decision) belongs in this core file; everything else belongs in its subsystem
   file, never here "for visibility" — that is exactly how this file reached 8,510 lines the first time.
2. **Topic file vs. `-log.md`: durable fact here, dated investigation there.** `solve-flow-block.md` and
   `solve-globalization.md` model this — copy the pattern, do not just read past it. A new **durable**
   fact (what is built, what the shipped default is, a binding decision) goes in the topic file. A new
   **dated** entry (a measurement, a probe result, an investigation step — anything that reads "on
   DATE we found X") goes in the matching `-log.md` file instead. When an investigation in a `-log.md`
   file reaches a durable verdict, update the topic file's "current status" paragraph to match and
   point at the `-log.md` entry for the full trail — do not duplicate the trail into the topic file.
3. **No `-log.md` sibling yet does not mean dated entries are welcome in the topic file — it means one
   has not been needed yet.** `solve-direct-preconditioners.md`, `solve-amg-multigrid.md`,
   `solve-field-split.md`, and `solve-march.md` currently hold both current facts and investigation
   history together. **The moment one of them is next edited after crossing ~1,800 lines** (roughly
   `turbulence.md`'s 1,779 — the largest still-unsplit rule file in the project, and a reasonable outer
   bound not to exceed), split it the same way as part of that change: peel its dated/historical content
   into a new `.claude/notes/<name>-log.md` (outside the auto-loaded tree — NOT a `paths:`-less file
   in `.claude/rules/`, which loads always rather than never), add it to the table above, and leave
   a synthesized current-status paragraph behind, mirroring rule 2. Do not wait for someone to notice
   the file is huge — that is what happened to this file the first time. **`solve-amg-multigrid.md` is
   already past this bound (2,088 lines as of 2026-08-18)** — it is the one candidate that should be
   split on its next substantial edit rather than waited on further.
4. **Closing or refuting a direction is not done until `solve-refuted-directions.md` has an entry for
   it, added in the SAME change.** One short paragraph — what was tried, on what case/state, why it
   lost — plus a pointer to wherever the full detail lives (a topic file, a `-log.md` file, or inline
   in the ledger itself if it is short enough to need no pointer). The ledger is the thing a future
   contributor actually greps before re-proposing an idea; a refutation that lives only in a `-log.md`
   file's prose will not be found by that search.

## How to read this file (and every file above)

This file, and the ones it splits into, accumulate. Three rules make a `grep` hit trustworthy:

1. **Every entry sits under a `##` section** — scan up to the nearest one to see what a hit is about.
   The sections are topical, not chronological; a 2026-07 and a 2026-08 finding on the same subject
   sit together.
2. **A superseded entry is DELETED, never struck through and never annotated in place.** `~~tildes~~`
   are invisible to `grep`, and "SUPERSEDED — see below" expresses supersession by *adjacency*, which
   a hit does not carry. Where a dead finding taught a trap, one line states the trap and the body is
   gone. If you find a strikethrough here, it is a bug — delete it.
3. **A measurement without its configuration is unfalsifiable, and worse than a wrong one** — a wrong
   number gets corrected, an unanchored one gets cited. Every number should name the case, the state,
   the preconditioner bundle and the shift it was taken at. Some older entries do not; they are marked.

**Before quoting any symbol, default or tolerance from any of these files, check it against the source.**
Three wrong facts were lifted from here by `grep` and asserted as current in a single session — a march
solver that had been replaced, a tolerance that had moved, and a preconditioning side that had been
deliberately reversed. See `CLAUDE.md` → **Stale-Record Check**.

## Current configuration (check here FIRST — the library and the case deliberately differ)

**The library defaults and the validated `bfs3d` case bundle are not the same, and conflating them is a
recorded error.** A default here that disagrees with the code is a defect — fix it in the same change.

| | library default | validated `bfs3d` bundle | where |
|---|---|---|---|
| smoother fill | `smoother_fill_levels=1` (ILU(1)) | 0 (ILU(0)) — **inert**: monolithic only, and the case runs the split | `MonolithicVCycle` / `compare.py` |
| smoother sweeps | `smoother_sweeps=2` | 4 — **inert**, as above | same |
| coarse-eq limit | `coarse_eq_limit=None` (~50) | 2000 — **inert**, as above | same |
| PC shift floor | `refit_beta_floor=0.0` | **0.05** | same |
| aggregation | plain (`pc_gamg_agg_nsmooths=0`) | plain | `amg_preconditioner.py` |
| field split | `field_split=False` | **True** | `compare.py` |
| stencil reach | `stencil_reach=3` | 3 | — |
| probe column reach | `column_reach=None` (uniform) | **(3,3,3,3,2,2)** | `compare.py` `COLUMN_REACH` |
| dual-time inner tol | `inner_tol=0.05` | **1e-2** | `compare.py` `INNER_TOL` |
| flow (leading) inverse | none — `field_split=True` requires `leading_inverse` and `trailing_inverse` (#371) | **`SimpleSmoothedInverse`** (`FLOW_INVERSE="simplesmooth"`) | `compare.py` |
| trailing hierarchy depth | `HierarchyBlockInverse` class default: `max_levels=2, strength_threshold=0.0, aggressive_levels=1` | **`max_levels=20, max_coarse=200, strength_threshold=0.25, aggressive_levels=0, frozen_coarsening=True`** | `compare.py` `JACOBI_TRAILING` |

**The coupled forward solve: one MEASURE, per-family RESTART REGIMES (restructured 2026-08-20, #282).**
⚠️ There are no `_COUPLED_FORWARD_SOLVER` / `_COUPLED_FACTORIZATION_FORWARD_SOLVER` / `_COUPLED_AMG_FORWARD_SOLVER`
symbols — every one of those is dead. `_coupled_step` builds the default solver itself, and the two
halves of the decision are now separated:

* **The stop is the march's own progress measure**, whatever the solve's `Convergence` names — the
  row-equilibrated `coupled_scaled_norm` (`RowScaled()`, rebuilt every outer iteration) by default,
  `BlockScaledNorm` under `BlockScaled()`, the plain Euclidean norm by default on the bordered
  mass-flow path. The default solver is `relative_residual_gmres(norm=None)`, which each step binds to
  the measure it is being judged by at that moment (see `solve-globalization.md`'s `norm_builder`
  entry), so steering and judging come from one definition even as the measure is rebuilt. ⚠️ **A
  tolerance is therefore only meaningful beside its measure** — `0.3` row-scaled and `1e-2` Euclidean
  are not comparable numbers.
* **The restart regime is per preconditioner family**, which is the part that genuinely differs, as
  `LinearSolveRegime` values in `turbulence/coupled.py`:

| regime | preconditioner (`coupled_step` / a session) | rtol | restart | max_restarts |
|---|---|---|---|---|
| `_BLOCK_LINEAR_SOLVE` | `BlockDiagonal` (block-SIMPLE) | 0.3 | 120 | 15 |
| `_FACTORIZATION_LINEAR_SOLVE` | `MaterializedJacobian(CompleteLu)` | 0.3 | 10 | 40 |
| `_VCYCLE_LINEAR_SOLVE` | `MaterializedJacobian(MonolithicVCycle \| FieldSplit)` (3D `bfs3d`) | 0.3 | 15 | 60 |
| `_CONSTRAINED_LINEAR_SOLVE` | `mass_flow_coupled_continuation` | **1e-2, Euclidean** | 120 | 15 |

Both coupled builders take the regime as one value, `linear_solve=LinearSolveSettings(rtol=…, restart=…,
max_restarts=…)` (#388 — there are no `forward_*` keywords any more). ⚠️ **Move the tolerance or the
restart with that value, never by passing a whole solver as `linear_solve`** — a solver also replaces the
stopping measure, which is a far larger change than the one intended.

⚠️ **`0.3` on the block and complete-LU families is the multigrid family's CALIBRATION, carried across
because it is a property of the measure, not of multigrid — it has not been re-measured there.** Those
two ran `1e-2` in a plain 2-norm before #282, so any recorded cost measured on them predates the change;
the earlier arrangement is the drift the issue documents, not a calibration.
(`_FACTORIZATION_LINEAR_SOLVE` was `_COUPLED_ILUT_FORWARD_SOLVER` while the now-deleted monolithic ILUT was
its other consumer, then `_COUPLED_FACTORIZATION_FORWARD_SOLVER` — see `solve-direct-preconditioners.md`.)

**Preconditioning side: RIGHT** (`solve_linear`'s default, taken by `_shifted_solve`), so the Krylov
residual is the **true** residual `b − Ax`. No solution-accuracy bound follows from the stop. `left` is
used only by `potential_flow`, where `M` is strong and the operator well-behaved.

## Contracts — the API boundary

- **`state.py` — BUILT (#285): `FieldLayout` is the ONE flat field-major state layout, and nothing
  else may re-derive `f * n_cells + i` (binding).** A coupled state is one flat vector, field-major,
  described by an ordered tuple of named `StateBlock`s over a cell count. Three block kinds cover
  everything the solvers need: `CellFields(name, n_fields)` (whole per-cell fields, read out
  `(n_cells,)` for a scalar and `(n_cells, n_fields)` for a vector), `SubLayout(name, layout)` (a
  nested sub-state, read out **flat** so its owner unpacks it with its own layout), and
  `GlobalDofs(name, count)` (degrees of freedom attached to no cell — a constraint multiplier). The
  object is mesh-free, array-free, hashable, and a pytree with **zero** leaves, so it rides inside a
  differentiated module or as a static field indifferently.
  - **Why it exists.** The same concept had **three** implementations in three subpackages —
    `flow/state.py::BlockStateLayout`, `turbulence/coupled.py::CoupledRANSLayout`, and
    `solve/field_split.py::FieldGroups` — each with its own vocabulary, none composing the others,
    and each of the first two carrying a docstring saying it existed so the arithmetic was not
    open-coded. `CoupledRANSLayout` said it carried the flow layout "verbatim" and then re-derived
    `flow_size = (dim + 1) * n_cells` and hand-sliced `pack`/`unpack`; `coupled.py::_k_block`
    computed a block range a fourth time. Adding an unknown to the flow state meant edits in three
    packages.
  - **The three are now compositions of the one.** `flow/state.py::flow_state_layout(dim, n_cells)`
    names the flow system's `velocity` / `pressure` blocks; `turbulence/coupled.py::coupled_rans_layout(flow)`
    **nests that layout** as the `flow` block and adds `k` and `omega`, so the flow block's widths are
    stated exactly once; `_k_block` is gone (`coupled.layout.slice_of("k")`). `MomentumContinuity.layout`
    is public for exactly this — the coupled builder takes the assembler's own layout object rather
    than rebuilding one from `dim` and `n_cells`.
  - **A bordered state is an extra named block, not a special case.** The mass-flow-constrained march
    carries `[flow…, k, omega, beta]`; `_mass_flow_layout(coupled)` is
    `coupled.layout.appended(GlobalDofs("mass_flow", 1))`, and the block-scaled measure reads its
    `sizes` instead of hand-building `(flow_size, n, n, 1)`.
  - **`FieldGroups` is a partition VIEW over a layout, not parallel arithmetic** — see
    `solve-field-split.md`. It holds the layout and a leading field count, derives `n_dofs` /
    `leading` / `trailing` from it, and refuses a layout carrying a `GlobalDofs` block, because a
    partition into whole fields cannot describe a multiplier belonging to no field.
  - ⚠️ **`FieldLayout` has no `dim`, and there is no `layout.dim` anywhere.** Number of *fields* is
    `layout.n_fields`; the fields *before* a block are `layout.field_offset(name)`; the spatial
    dimension is a property of the mesh (`coupled.momentum.mesh.dim`), which is where every former
    `layout.dim` call site now reads it. The old spellings `layout.dim + 3` and `layout.dim + 1` are
    now `layout.n_fields` and `layout.field_offset("k")` — both of which survive a block being added.
  - **The surface is deliberately only what is used.** `span(first, last)`, `sub(name)`,
    `width(name)` and `block(name)` were written and **deleted before the change landed**: nothing
    outside their own tests called any of them, and a nested layout is reached from the assembler
    that owns it (`momentum.layout`) rather than back out of the state it was nested into. Add one
    back when a consumer needs it, not in anticipation.
  - Pinned by `tests/unit/test_state.py` (mesh-free: shape, addressing, packing, the bordered
    extension, the construction refusals, and the zero-leaf/hashable pytree properties).

- **`RootSolveSettings` (`implicit.py`) — the settings of a root solve, as one value (BUILT 2026-09-22,
  #428).** `RootSolveSettings(convergence, max_steps, linear_solver, adjoint_solver)`, a `SettingsValue`
  with `None`-unset fields, and `DEFAULT_ROOT_SOLVE` the empty override every builder defaults to. Its one
  method, `solver(strategy, **fields)`, is the only place a value becomes a `RootSolver`; `measures` rides
  through as a field, because it says what the *problem's* residual can be measured in. Taken by all three
  builders of a root solve — `flow.reused_flow_solve`, `flow.bulk_velocity_flow_solve` and
  `turbulence.scalar_pseudo_transient_solve` — each beneath its own step cap (80 / 20 / 40) via
  `filled_from`.
  - **The membership test is the same one `Globalization` uses:** a setting belongs here when its reason
    can be stated without naming the residual. `strategy` (the step a builder exists to construct) and
    `measures` cannot, so they stay with the builder.
  - **What it replaced.** All three took `max_steps`, two a forward linear solver, one `rtol`/`atol`, and
    **none** could reach `adjoint_solver` — the setting `solve_coupled` had to grow because its absence
    makes a transpose solve on a hard case raise an error naming a remedy the entry point cannot reach.
    The scalar builder's `rtol=1e-10, atol=1e-12` were a second copy of `_ROOT_CONVERGENCE`'s and are gone
    rather than moved, so the default is declared once.
  - **The drivers `solve_coupled` and `solve_coupled_mass_flow` still spell these as keywords** — they run
    a solve rather than returning one, their surface is what a case file will describe (#375), and their
    forward linear solve is reached through `linear_solve=LinearSolveSettings(...)` on the step instead.
  - **⚠️ `tools/sibling_builders.py` reports nothing for these three and cannot**: each returns a closure
    it defines, which the tool credits with constructing nothing, and their public surfaces share fewer
    parameters than its threshold anyway. That silence was not evidence before this change and is not now.
  - Pinned by `tests/unit/test_root_solve_reach.py`: every builder takes the value, a non-default one
    arrives on the built solver field for field, the three step caps as literal numbers, one override
    changes one setting, and each refusal.

- **`convergence.py` — the stopping test as ONE value (BUILT 2026-09-16, #370).** `Convergence(measure,
  rtol, atol)`, a `SettingsValue` with `None`-unset fields, and the measure family `ResidualMeasure`
  (abstract) = `Euclidean()` / `RowScaled()` / `BlockScaled()`. Each measure builds a
  `MeasureBuilder = (step, state) -> ResidualNorm` against a `ResidualMeasures` source the problem
  supplies (`row_scaled(step, state)`, `block_scaled(state)`); `PLAIN_RESIDUAL` supports only Euclidean
  and refuses the others by name. `RowScaled` is rebuilt at every state it is asked about;
  `BlockScaled` takes the initial state's scales once and holds them (it normalizes itself, so
  re-basing at a refresh would put the stop out of reach — #156 seam 4).
  - **Why one value:** `rtol` meant a row-scaled tolerance on one path and a Euclidean one on another,
    and the measure was chosen by three settings in two places (`block_scaled_norm` / `residual_norm`
    on the step builders, `scaled_norm` on `solve_coupled` for *when* the scales were rebuilt). There is
    no step-level measure choice any more: `coupled_step` and `mass_flow_coupled_continuation` take
    neither keyword, and the march hands the step its measure every outer iteration. A case file can
    therefore write `convergence: {measure: {kind: RowScaled}, rtol: 0.0, atol: 1.0e-5}` with one meaning.
  - **`RootSolver`:** unset measure → the strategy's own `residual_norm` (no builder, byte-identical to
    before); a set one is built against `RootSolver.measures` (default `PLAIN_RESIDUAL`).
    `solve_coupled` defaults to `RowScaled()` against `_CoupledMeasures`; `solve_coupled_mass_flow` to
    `Euclidean()` against `_MassFlowMeasures` (block-scaled allowed, row-scaled refused — the border row
    has no diagonal).
  - **⚠️ THE FROZEN ROW-SCALED MEASURE WAS DELETED, AND THIS IS WHAT DECIDED IT (measured 2026-09-16).**
    The default used to freeze the row scales at the step's build state; `scaled_norm=True` rebuilt them.
    *Cost:* on pitzDaily (12225 cells) at a converged checkpoint (step 31 of a shipped run), commit
    `e7fa44d`, jax 0.10.2, macOS arm64, an **eager** rebuild (as the march calls it) is **8.2 ms**, a
    jitted one **0.25 ms**, one coupled residual **8.6 ms**, its jvp **13.9 ms**, against **~19 s** per
    outer step in that run — ~0.04 % of a step (`validation/pitzdaily_openfoam/measure_rebuild_cost.py`).
    *Behaviour:* with the rebuild forced on every row-scaled `solve_coupled`, the fast tier (1843) and
    slow tier (43 of 44) passed; the one failure was
    `test_the_coupled_adjoint_is_independent_of_the_forward_iteration_count`, whose two arms both took
    20 steps (17/20 frozen) so its distinct-path guard refused — a fixture, not a gradient. Steps over
    26 matched tests rose **1213 → 1302 (+7 %)**, both periodic-channel tests ~48 → 72. Scoring each final
    state in all three measures showed **neither form consistently tighter** (the periodic
    AMG-independence test stopped *looser* after more steps), so the difference is steering, not
    stopping bar. These fixtures are small, mildly developed flows and cannot show the recorded
    developed-flow failure of the frozen form; they show nothing *needs* it. Both flagship cases already
    rebuilt.
  - ⚠️ **That measurement predates the linear-solve binding fix above**, so its step counts are for a
    rebuilt outer measure with a build-time inner stop. The inner stop now follows too; re-measure before
    quoting counts.
  - **pitzDaily with the whole change, against a control built from `main` (2026-09-16).** Control:
    `e7fa44d` in a scratch worktree; arm: this change. Both `validation/pitzdaily_openfoam/compare.py` at
    its shipped defaults (viscosity ramp 16 stations x 1, momentum-only scaling, turbulence damping 3,
    dual-time 5 / 0.01, field split `simplesmooth` / `jacobi_smoothed`, stop `rtol=0, atol=1e-5`,
    row-scaled measure rebuilt per step in both). **Both 31 steps, `x_r/h` 8.0686, final `|R|`
    7.841e-06 / 7.839e-06, no retries.** The two logs are identical through step 17; from step 18 (the
    target station) the per-step residuals agree to four figures while the restart cycles differ by ±1–2
    per step, **194 → 202 in total (+4 %)** — the inner Krylov stop now reading the rebuilt measure
    instead of the one the step was built with. Wall clock is not quotable: another session loaded the
    machine during the arm (1-minute load ~10).
- **`driver.py` + `shifted_step.py` — BUILT 2026-09-19 (#448, #277): the robust march is residual-agnostic,
  and lives HERE (root `CLAUDE.md`, Principle 3.6).** Both were `turbulence/coupled.py` internals until a
  laminar flow problem found it could not reach them. **`solve.staged_march(residual_fn, state, *, strategy,
  source, refresh, convergence, measures, drift_measure, max_steps, step_control, …, caller)`** is the
  segment loop and every rule the sequence obeys: the stopping target `atol + rtol * reference` measured
  **once** at the initial state in the measure `convergence` names and held across segments; the measure
  builder made once and handed to every segment; the **last segment marched without the trigger** (there
  is no second solve, so a segment stopped where its trigger fires would end short of a root); the step
  control threaded across segments while the damping reference and `drift_measure` restart per segment
  (`drift_measure` is `state -> drift(state)` so the caller re-bases it at each segment's start); the
  final state judged in the measure it was steered by and **refused with `EquinoxRuntimeError` if not a
  root** (the adjoint is valid only at one). It returns `StagedResult(state, strategy)`; attaching
  `root_adjoint` stays the caller's, because only the caller holds the differentiable parameter pytree.
  `solve.explicit_source` picks `FinishedSource` / `CallerBuiltSource` when the caller gave a strategy or
  a `RefreshPolicy(builder=…)` and **refuses** (`refuse_unforwardable_settings`) any step-configuring
  setting beside them rather than dropping it. **`solve.shifted_step(policy, *, globalization, dual_time,
  regime, krylov_solver, adjoint_preconditioner_factory, …, line_search)`** is the tail every builder
  ends in: single shifted step vs dual-time loop, the refusal of inner-loop hooks with no loop and of a
  `refresh_on_cycles` with nothing to fire, and the default `relative_residual_gmres(norm=None)` built
  from a `LinearSolveRegime` (`regime=None` leaves the step class's own solve — what
  `momentum_continuation` wants). `LinearSolveSettings`, `LinearSolveRegime` and `resolve_linear_solve`
  moved here with it: nothing in them names a preconditioner or a residual.
  - Callers: `turbulence.solve_coupled` (`_CoupledMeasures`, `eddy_viscosity_drift`, a session source),
    `flow.solve_flow_march` (`flow.FlowMeasures`, no drift measure, `_FlowSource`), and
    `flow.momentum_continuation` for `shifted_step`. A residual with a coefficient to watch drift on
    supplies `drift_measure`; one without (constant-viscosity laminar flow) passes `None` and uses a cost
    trigger.
  - `tests/unit/test_coupled_rans.py` monkeypatches `newton_march` **on `solve.driver`**, where the loop
    now calls it.
  - `block_reference_scales(layout, residual)` (`norm.py`) is the per-block scale a `BlockScaledNorm` is
    built from — one home for what `_coupled_block_scales` and the flow measure both need.
  - **`solve/` imports nothing outside itself** (`tests/unit/test_layering.py`, always-on): it is the layer
    that lets every residual run on this machinery.

- **`jacobian_probe.py` + `monolithic_policy.py` — MOVED HERE 2026-09-19 (#450 stage 1).** `JacobianProbe(plan, structure, narrowing)` and `jacobian_probe_plan(face_cells, n_cells, n_fields, …)` are the coloured-probe plan and de-compression map, which depend on the cell graph and reaches alone; `narrowing` (`assembler -> assembler`, compared by value) is the one residual-specific part — a stand-in whose Jacobian is materialized. `MonolithicFactorShiftPolicy(base, preconditioner)` pairs any base `ShiftPolicy`'s shift diagonal with a frozen monolithic inverse, and `FrozenTransposeFactory` is its value-equal transposed apply (equality is what keeps a rung's engine rebuild a compile-cache hit). Their coupled-RANS builders are `turbulence.coupled_jacobian_probe` and `_CoupledNarrowing`.

- **`materialized_session.py` + `materialized_spec.py` — MOVED HERE 2026-09-19 (#450 stage 2).** `MaterializedSession(spec, problem, …)` is the lifecycle of a materialized-Jacobian preconditioner: one probe, one inverse and one refresh hook, each created at most once, so every step built from it carries the identical static objects. It is written against **`MaterializedProblem`**, an ABC that is everything the lifecycle needs from a residual: `assembler`, `layout`, `with_assembler`, `probe(settings, active_rows)`, `groups()` (`None` ⇒ nothing to split), `bind_march`, `shift_source`, `build_step`. `bind_march`'s result **must** carry `dual_time` and `inner_refresh`, which the session reads and may set; it is also where a residual validates early (coupled RANS raises its `k`-positivity refusals there, before any inverse is fitted). Implementations: `turbulence._CoupledProblem`, `flow._FlowProblem`. A `FieldSplit` inverse is refused at construction for a problem whose `groups()` is `None`. `beta_tracking_refresh(assembler, probe, every_step=, refit_beta_floor=, observer=)`, `jacobian_matvec` / `batched_jacobian_matvec` (module-level `filter_jit`, assembler an ARGUMENT so a rung is a cache hit), `frozen_shift_diagonal`, `PROBE_BATCH_SIZE` and the two family regimes (`FACTORIZATION_LINEAR_SOLVE`, `VCYCLE_LINEAR_SOLVE`) came with it. The specs (`MaterializedJacobian`, `CompleteLu`, `MonolithicVCycle`, `FieldSplit`, `JacobianProbeSpec`) and `MATERIALIZED_MAPPING` / `materialized_spec_from_mapping` / `materialized_spec_to_mapping` are here too; a solve with more kinds extends `MATERIALIZED_MAPPING.kinds` rather than restating them. `SessionSource` (`driver.py`) is the `ContinuationSource` over any session.

- **`block_preconditioner.py` — BUILT 2026-09-19.** `MaterializedBlockPreconditioner`: one `BlockInverse` (e.g. `SimpleSmoothed`) fitted to the whole materialized, shifted Jacobian, sharing the probe/shift/in-place-refresh of `MaterializedJacobianPreconditioner` with the field split. It is the inverse of a problem whose fields form a **single group** (`MaterializedProblem.groups()` is `None`), and the session enforces the mirror pair: `FieldSplit` refused for one group, a bare `BlockInverse` refused for two. `MaterializedJacobian.inverse` accepts a bare `BlockInverse`. Needs no optional dependency (the hierarchy is traced JAX), unlike `MonolithicVCycle` (PETSc GAMG). Tested for exact transposition and for being an approximate inverse of the SHIFTED operator (mutating the shift away fails it).

- **`linear.py` — BUILT.** `solve_linear(matvec, b, solver, preconditioner=None)` is a
  matrix-free wrapper over `lineax` (default restarted GMRES); `lineax` supplies the
  **implicit-diff of the linear solve** (the Krylov loop is not taped). This is the load-bearing
  adjoint primitive. The optional **preconditioner** `M` (a matvec ≈ `A⁻¹`) is applied on a
  caller-chosen side (`preconditioner_side`, default **right**); since the caller `stop_gradient`s
  `M`'s coefficients, it changes only Krylov convergence, not the solution or its gradient —
  **verified transparent** in `test_preconditioning.py` (solution and gradient identical with/without
  `M`). This is the seam the **outer block preconditioner** (`solve-direct-preconditioners.md`,
  `solve-amg-multigrid.md`) attaches to.
  - **The side is a real numerical choice, and forcing one broke a solve — keep both (binding).**
    **Right** (`A∘M` and `b`, recover `x = M y`) stops on the **true** residual `‖A x − b‖`, so a
    **weak** `M` cannot falsely report convergence — the honest stop on the shifted coupled saddle at
    low pseudo-transient shift (where the single V-cycle degrades as β falls), and the default. **Left**
    (`M∘A` and `M b`) stops on the **preconditioned** residual `‖M(A x − b)‖`, the right measure when
    `M` is a **strong** inverse of a well-behaved SPD operator: on a wall-resolved mesh the near-wall
    anisotropy makes `potential_flow`'s Laplacian condition ~1e6–1e10, where the multigrid drives `‖M r‖`
    to tolerance but the **true** residual cannot in `max_steps` — so a right-preconditioned solve there
    exhausts the Krylov budget and raises. Making the solve *universally* right (the honesty fix for the
    saddle) therefore broke `potential_flow` (its `newton_step` now passes `preconditioner_side="left"`);
    the converged solution is identical either way — only which regime converges in a bounded step count
    differs. Threaded `solve_linear` → `newton_correction`/`newton_step`. Pinned by
    `test_initialization.py::test_potential_flow_survives_a_wall_resolved_aspect_ratio`.
  - **`solve_linear` returns `(x, cycles)` — there is ONE linear-solve entry point, not a counted/
    uncounted pair (binding, do not re-split).** A caller that only wants the answer writes
    `x, _ = solve_linear(...)`. A `solve_linear_counted` sibling existed briefly and was **deleted**: it
    held the real body while `solve_linear` forwarded to it and dropped the count, i.e. the old shape
    preserved across a refactor — the delegating-wrapper form the pre-release no-shims policy bans. It
    also duplicated the whole signature and `Parameters` block (its docstring had already degenerated to
    "the arguments mean exactly what they do there", which cannot stand alone), and it was *dominated*:
    it did `solve_linear`'s job plus more. The blast radius of collapsing was ~9 lines — the function has
    exactly **two** library call sites (`newton.py`'s `newton_correction`, `implicit.py`'s adjoint
    transpose solve); the "it has too many callers to change" intuition is false, so do not resurrect the
    pair on that argument. **The count is restart CYCLES, not matvecs** (each cycle is up to `restart`
    matvecs — the standing misreading of `lineax`'s `num_steps`), pinned to `int32` so a caller can carry
    it through a `lax.while_loop` (whose carry structure must be invariant — exactly one call site does,
    the pseudo-transient escalation loop), and `0` for a solver that reports none (a direct
    factorization). **Why the count:** a frozen preconditioner going stale shows up first as a *rising
    cycle count on an otherwise-unchanged system*, before the residual history shows anything — so it is
    the honest trigger for re-freezing the preconditioner mid-march, and a robust one. Wall-clock time is
    the tempting proxy and a bad one: it moves with machine load or a suspended process while the linear
    algebra has not changed at all. Pinned in `test_preconditioning.py`, including the load-bearing
    behavioural check that **a better preconditioner strictly lowers the count** (otherwise it measures
    nothing).
- **`newton.py` — BUILT: `newton_step` / `newton_correction`, one correction each. There is NO
  `NewtonSolver` class — it was deleted (binding, #102).** Each forms `J` matrix-free via `jax.jvp`
  and calls `solve_linear`; no hand-derived Jacobian.
  - **Why it went.** The class was `newton_step` plus a **fixed-count, unchecked loop**, which is
    redundant at `iterations=1` (19 of its 28 call sites — a *linear* residual, where one correction
    is exact) and **forbidden** above it (a fixed count cannot tell convergence from exhaustion, and
    taping the unrolled steps is the gradient path the two-level implicit differentiation exists to
    avoid). Its one production use, `laplace_field`, was `iterations=1` — not Newton at all, just an
    exact linear solve. The turbulence scalars had already migrated off it for exactly these reasons
    (see `turbulence/continuation.py`).
  - **The split to hold to.** *Linear* residual → `newton_step`, exact in one call. *Nonlinear*
    residual → `RootSolver` (converges, globalizes, IFT adjoint). Do **not** reintroduce a
    fixed-count loop over `newton_step` in library code. A *test* may write one inline when the point
    is to show unglobalized Newton is insufficient (`test_scalar_continuation.py`) or to isolate a
    preconditioner (`test_turbulent_channel.py`) — that is 2 lines and self-documenting, not a class.
  - **`newton_step` is the only path that differentiates in FORWARD mode.** `RootSolver` is
    a `jax.custom_vjp`, which registers only the reverse rule, so `jacfwd`/`jvp` through it raises
    `TypeError` — a JAX API consequence, not a mathematical one (the IFT gives the tangent just as
    readily; a `custom_jvp` would serve both, at the cost of the separate tight `adjoint_solver` the
    current design deliberately controls). `newton_step` is plain traced operations, so both modes
    work. This matters: the plane-wall sensitivity gate takes `jacfwd` through the whole transient
    march — one linear solve per input, the efficient direction for a scalar parameter against a
    whole field. Pinned in `tests/unit/test_newton.py`.
- **Neither function jits internally — the caller owns the jit boundary.** Wrap calls in
  `eqx.filter_jit`; un-jitted, every operation dispatches eagerly. This is about `newton_step` and
  `newton_correction`, which are plain traced operations: ⚠️ **`RootSolver.solve` is the
  opposite case and REFUSES `jit`/`vmap`** since 2026-09-15 — it marches in Python (see below), and its
  step is compiled for it. The cache-hit discipline still applies, one level down: pass the assembler as
  an `equinox.Module` **argument** so its arrays are dynamic leaves, and hand the solve a residual whose
  identity is stable (`assembler_residual`, a bound method, or a small `equinox.Module`) — a lambda built
  at the call site is hashed by identity and recompiles the march step on every call.
- **`implicit.py` — BUILT (`RootSolver`).** The nonlinear counterpart: Newton to
  convergence on `stop_gradient` copies of `phi0` and `theta` (`stop_array_gradients`, which passes
  non-array leaves through), with the reverse-mode **IFT adjoint** attached afterwards at the root it
  reaches by `root_adjoint` — one transpose linear solve,
  `dphi*/dtheta = -(dR/dphi)^{-1}(dR/dtheta)`, no Newton loop taped.
  `solve(residual_fn, phi0, theta)` takes the differentiable params `theta` explicit so the adjoint
  returns their cotangents. Reverse-mode only (`jax.grad`), which is what a scalar objective through
  the solver needs. This is the "IFT on the converged Newton state" half of the two-level scheme; it
  activates with the first nonlinear residual (the flux limiter). Verified
  (`test_root_solver.py`): converges a nonlinear root, gradient matches the closed form to
  1e-10, and is iteration-count-independent. Used by the limited-advection solve.
  **⚠️ THE LOOP IS `newton_march`, AND THERE IS NO OTHER ONE (binding, 2026-09-15, phase 3 of the
  unification).** `_forward` — the traced `lax.while_loop` this class used to carry — is **deleted**.
  One driver now runs every Newton solve in the package, so the hooks (observer, refresh trigger, step
  control, retries, homotopy, per-step preconditioner re-fit) are reachable from one place and a
  capability cannot exist on one loop and not the other, which is the defect class #369 was. The price,
  decided deliberately rather than discovered: **a Python loop cannot run under `jax.jit` or
  `jax.vmap`** — ⚠️ **nor inside `lax.scan` / `lax.fori_loop`, which is the case that was MISSED when
  this was scoped and is the one with a real consumer.** A transient march over a *nonlinear* residual
  scanned the solve (`test_limiter_reduces_overshoot_on_advected_step`, validation tier — the only
  affected site in the repository, found by running that tier rather than by the grep for `jit`/`vmap`
  that preceded the decision). It is now a Python loop whose per-step residual is an `equinox.Module`
  holding the previous states (`_BdfStep`), so the later timesteps run on the compiled Newton step the
  opening ones build — measured 7/4/4 residual executions for the first three steps and exactly 1 for
  every step after, that one being the march's own eager reference-norm evaluation. `docs/` carries the
  pattern for users. **The lesson for the next decision of this shape: "which transforms does this
  break" is not answered by grepping for the transforms' names — `lax.scan` traces its body just as
  `jit` does, and a test tier found what the grep did not.** So `solve()` refuses a transform up front via
  `refuse_a_transform_the_march_cannot_run_in` (defined in `march.py` and **exported**, since
  `solve_coupled` and `solve_coupled_mass_flow` refuse on the same terms and library code may not
  deep-import a `solve` submodule — `tests/unit/test_solve_api.py` fails the gate if it does). `jax.grad` is unaffected and is the mode the project needs — the march runs on stopped,
  concrete values and the derivative is attached at the root afterwards.
  - **What that cost at the four call sites, and the trap to avoid repeating.** `reused_flow_solve`,
    `bulk_velocity_flow_solve`, `scalar_pseudo_transient_solve` and `solve_coupled_mass_flow` each wrapped
    their solve in `eqx.filter_jit` for compile-cache reuse across sweeps; those wrappers are **removed**
    (they would now hit the refusal). The reuse is preserved one level down, because `_march_step` is
    itself `filter_jit`-compiled — **but only if the residual handed to the solve has a stable identity**.
    Each of those call sites passed a freshly built `lambda`, which is a new static cache key per call and
    would have recompiled the whole march every sweep. They now pass `assembler_residual` (the shared
    module-level `(state, assembler) -> assembler.residual(state)`), `_BulkVelocityResidual`,
    `_ParameterFreeResidual` or `_MassFlowConstrainedResidual` — small `equinox.Module`s whose settings
    compare by value and whose arrays ride as dynamic leaves. `RootSolver.solve` binds `theta`
    into `_ResidualAt`, a module for the same reason. Pinned by
    `test_a_carried_preconditioner_compiles_the_scalar_solve_once`.
  - ⚠️ **That test measured nothing until this change fixed its fixture, and the reason generalizes.**
    Under a `lax.while_loop` the body is traced whether or not it executes, so the test's later sweeps —
    which started from the previous sweep's converged state and took **zero** steps — still reported
    compilations. An eager loop that takes no steps compiles nothing, so both arms collapsed to the same
    count. The fixture now disturbs the state each sweep so the march actually runs. The general form:
    **a fixture that reaches a solver in a state with no work to do can pin a compilation property while
    exercising none of the solve.**
- **`root_adjoint.py` — BUILT 2026-09-15 (phase 1 of unifying the two Newton loops): the IFT adjoint is
  a standalone step, not a property of the loop.** `root_adjoint(residual_fn, root, theta, *,
  adjoint_solver=None, adjoint_preconditioner=None)` returns `root` unchanged and carries its derivative
  (a `custom_vjp` whose backward rule is the transpose solve); `TransposedPreconditioner` moved here with
  it. Whatever derivative `root` itself carries is **discarded** — its dependence on `theta` is the
  adjoint's to supply — so any loop may produce the root, including the eager `newton_march` with its
  hooks. **It does not check that `root` is a root**: the caller owns the convergence test, because only
  the caller knows the tolerance and measure. Extracted from `_implicit_solve` with
  `RootSolver` rewired onto it; values and gradients were compared **bit for bit** before and
  after across `DampedNewtonStep` (with and without a transposable and a `TransposedPreconditioner`),
  `PseudoTransientStep`, `DualTimeStep`, module-valued `theta`, `jit(grad)`, `vmap(grad)` and the
  `phi0` gradient — all identical. Why the extraction: a throwaway toy spike showed the gradient does not
  depend on the loop, so the eager march can gain the adjoint and the traced/eager split in
  `solve_coupled` (the `observing` switch behind #369) can be removed. **Phase 2 did that
  (2026-09-15):** `solve_coupled` marches once with `newton_march` on `stop_array_gradients` copies and
  attaches `root_adjoint`; see `turbulence.md`. `stop_array_gradients` (also in `root_adjoint.py`) is the
  one helper both loops use to stop a pytree's array leaves.
  ⚠️ A `theta` holding a **callable leaf** was already refused before this change (the `custom_vjp`
  rejects non-JAX-type arguments) and still is; `jax.lax.stop_gradient` on such a tree also raises,
  which is why `stop_array_gradients` filters by `eqx.is_array`.
  - **Convergence guard (binding — the IFT adjoint is only valid at a root).** `solve()` reads
    `MarchResult.converged` and **raises `eqx.EquinoxRuntimeError`** rather than returning a state that
    is not a root: exhausted `max_steps`, stopped on a collapsing constraint cap, or a non-finite
    residual norm. It runs before `root_adjoint` is reached, on the stopped inputs, so it fires for both
    the forward value and the `jax.grad` path, closing the silent-wrong-gradient hole where the transpose
    solve at a non-root stays well-posed and raises no `NaN`. A plain `raise` since the march is eager —
    it was an `eqx.error_if` on a traced value while the loop was a `while_loop`; the exception type is
    unchanged, which is what `solve_reynolds_continuation`'s retreat catches. A `NaN` mid-iteration is
    often caught first by `lineax`'s own non-finite guard at the next linear solve — both are hard
    errors, neither is silent.
    - ⚠️ **`within_tolerance` alone cannot reject a diverged march, and the fix lives in the march
      (2026-09-15).** It compares with `<=`, so when the residual norm *and* the threshold it is judged
      against have both run away to `+inf`, `inf <= inf` is `True` — a false "converged" on a state that
      solves nothing. (A `NaN` fails the comparison on its own; `+inf` is the case that needs help.) The
      finiteness test therefore sits inside `newton_march`'s own `converged_at`, so **every** consumer of
      `MarchResult.converged` inherits it rather than each driver re-deriving it, and the march refuses to
      *step* from a non-finite residual at all — otherwise the step's Krylov solve raises first, reporting
      a bug upstream of itself instead of the fact that the march never left a state solving nothing.
  - **✅ `solve_coupled(adjoint_solver=…)` — the transpose solve's Krylov settings are REACHABLE
    (BUILT 2026-08-14).** `RootSolver` has carried an `adjoint_solver` field all along, but
    `solve_coupled` did not expose it: it forwarded the forward-only retry policy's `retry.solver`, and
    nothing for the transpose. So every `jax.grad` through a coupled solve fell through to
    `default_linear_solver()` = `lx.GMRES(rtol=1e-10, atol=1e-10)` at **lineax's own** restart length
    and stagnation budget — and on `bfs3d` that combination raises *"A stagnation in an iterative
    linear solve has occurred. Try increasing `stagnation_iters` or `restart`"*, i.e. **the remedy the
    error names was unreachable from the coupled entry point.** Now threaded;
    `solve_reynolds_continuation` forwards `**solve_kwargs`, so it inherits the argument for free.
    `None` (default) keeps `default_linear_solver()` and is byte-identical.
    Build one with `relative_residual_gmres(rtol, restart=…, stagnation_iters=…, max_restarts=…)`.
    **Why it is a separate injection point from the forward solver, and not a knob to unify with it:**
    the two meet different operators. The Newton steps solve `J + β d`, which the pseudo-transient
    shift keeps diagonally dominant; the transpose solve meets `J` itself at β = 0 with no shift to
    soften it, once, and its accuracy *is* the gradient's accuracy.
    Pinned by `test_the_injected_adjoint_solver_reaches_the_transpose_solve_and_only_it`
    (`tests/integration/test_coupled_rans.py`, `slow`), which is a **reachability** test rather than an
    accuracy one: an adjoint solver crippled to a single Krylov vector and a single restart cycle must
    make `jax.grad` raise while leaving the forward value bit-identical. That is the property the gap
    destroyed, and an accuracy test cannot see it — the default solver returns the right gradient on a
    2D channel whether or not the argument is wired to anything.
    **Outcome on the case it was built for:** with this threaded and a budget large enough not to bind,
    `jax.grad` runs on `bfs3d` for the first time and matches a central finite difference to 1.9e-04 —
    but the passthrough alone was not sufficient, because the transpose solve needs ~1450 preconditioner
    applications and the first budget allowed ~900. See `solve-flow-block.md`'s *"`jax.grad` RUNS ON THIS
    CASE"* section for the costs, the arms, and the finite-difference trap that a loose root sets.


## Measurement discipline for preconditioner probes (BINDING)

- **⚠️ MEASUREMENT DISCIPLINE FOR PRECONDITIONER PROBES (binding — every one of these produced a wrong
  verdict that had to be retracted).** Judge a candidate preconditioner **only** by running it through
  GMRES and reading the **true** residual `‖Ax−b‖`, **at a state and shift pairing where the operator
  is actually hard**. Seven cheaper-looking shortcuts are all invalid on this indefinite saddle:

  **Shortcut 0, and the most expensive one found so far: judging a preconditioner by its CYCLE COUNT
  rather than by the march's WALL CLOCK.** Every other entry here is about measuring the residual
  honestly; this one is about measuring the wrong *quantity* honestly, which is harder to notice. The
  field split was measured at a captured hard iterate to cost a cycle (4 against the monolithic's 3) and
  was very nearly abandoned on that basis. Run end to end at the identical configuration it is **31%
  faster** (2161 s against 3140 s) to the identical reattachment length — **while taking 11% MORE
  cycles** (324 against 293) and triggering 21% more refreshes. A cycle is not a unit of cost: two
  smaller V-cycles plus one sparse coupling product apply far more cheaply than one six-field V-cycle,
  so the split buys more cycles at a lower price. **A cycle count is only a valid proxy when the
  candidates share a per-application cost** — true when comparing smoother sweeps or aggregation
  settings on one hierarchy, false the moment the preconditioner's *shape* changes. When the shape
  changes, the only honest measure is wall clock over a whole march, and a single-state probe cannot
  give it. (Two corollaries worth keeping: the same run's mean cycles per inner solve was *lower* for
  the split, 1.49 against 1.68 — the higher total came from more, cheaper steps, so even the direction
  of the cycle difference depends on whether you count per solve or per march; and the monolithic run
  contained a single 40-cycle solve where the split's worst was 8, which no average shows.)

  1. **The preconditioned residual `‖Mr‖`.** PETSc's default convergence norm. SOR/Krylov-smoothing
     report `reason=2` (converged) at a **true** residual of 1.0. Force `KSP_NORM_UNPRECONDITIONED`.
     A level-ILU "win" was once entirely this artifact.
  2. **One-apply contraction `‖M A x − x‖ / ‖x‖ < 1`.** Rejected a candidate on this; it is not a
     convergence criterion for a *Krylov-accelerated* preconditioner. Counter-example from our own
     data: ILU(0) at β=0.02 has a one-apply contraction of **4.5** and still converges in 97 matvecs.
  3. **The spectral radius of the iteration operator.** The largest eigenmode of a smoothed operator is
     the *smooth* mode — which is the coarse grid's job, not the smoother's. A "ρ = 9e4, diverges"
     reading nearly killed a Vanka smoother that had never actually been run through GMRES.
  4. **A probe at a BENIGN operating point — an easy operator cannot discriminate between
     preconditioners.** This is the one that nearly buried the largest preconditioner win found on this
     case. A GAMG aggregation sweep run at the march's *converged tail* returned **6 cycles for every
     arm** — shipped, plain aggregation, and two strength thresholds all identical — and the honest
     reading of that sweep was "no difference, close the question". Re-run at the march's own **hard**
     states, the same arms separated **22 → 9 cycles** (2.4×, and 66× lower true residual). Where the
     operator is well conditioned, every candidate looks the same, so a null result there is *no
     information*, not evidence of no effect.
     **Pick the hard states from the march's own log, not by intuition:** the checkpoints plus the step
     table identify them directly — highest cycle count, clipped `a_min`, and any step carrying a retry
     flag. Probe the state *entering* such a step (the checkpoint written after the previous one).
     The same caution applies to the *pairing*: use the operator at the march's own β with the V-cycle
     at `max(β, refit_beta_floor)`, because that mismatch is the shipped configuration. A probe that builds
     the V-cycle at the march's raw β instead measures a configuration the floor exists to prevent —
     it reported "the V-cycle does not converge at all in the tail" (true residual 1.0), where the real
     pairing takes **6 cycles to 1.5e-10**.
     **Two traps in "highest cycle count", both of which pick a BENIGN state if you get them wrong:**
     - **Rank on the hardest SINGLE solve, never on the step's summed cycles.** The sum rewards a step
       that took many easy inner iterations over one that took a single hard solve, and on a real march
       the two orderings disagree outright: on the 3-rung `bfs3d` cold march the summed count picks a
       step whose hardest solve is **6** cycles over one whose hardest is **15**. `StepReport` has
       `max_inner_cycles` for exactly this, and `StateCheckpointer` now serializes it (with
       `inner_iterations`) so a later study can rank without re-parsing the log.
     - **A step's record describes only its ACCEPTED attempt, and the hardest operators live in the
       REJECTED ones.** A solve that blows past `retry.abort_above_cycles` gets the step redone (at an unchanged
       β, and the retry then succeeds easily — so the record shows the *easy* attempt. Same march: step
       50's hardest solve is **15 cycles at β = 0.0293** with α collapsing to 0, in attempt 1; the step
       reports **3 cycles at β = 0.0585**. That is also why the escalated attempts are where the
       *sub-floor* operators are — the escalation is what lifts β back above the floor. Until the
       rejected attempts are recorded, read them out of `march.log` (`redo step N (attempt 2): …` plus
       the per-inner table above it) and name the state and β explicitly.
  5. **A probe driven by a hand-built "march" solver.** Every family now shares the row-scaled stop and
     differs only in its restart regime (see the table above), which removes the specific trap recorded
     here — reaching for the complete-LU path's plain-2-norm solver while believing it was the AMG
     builder's row-scaled one, done twice in one session, once where it would have replaced a loose
     row-scaled stop with a tight Euclidean one and reported the difference as a restart-length effect.
     The general form still bites: **a hand-built forward solver replaces the stopping MEASURE, not
     just the tolerance**, and **it does not announce itself** — at a state where both converge in one
     cycle a self-check still passes and reports a validation it never performed. Take the restart or
     the tolerance through `linear_solve=LinearSolveSettings(restart=…, rtol=…)`, never by passing a whole solver
     in that same `linear_solve` slot (#388).
     Note also what the regime's `rtol = 0.3` *is*: an inexact-Newton forcing term on the **linear** residual
     per inner solve, not a solution tolerance — accuracy comes from the inner loop iterating. And the
     **achieved** reduction is routinely tighter than the requested one, because a restarted GMRES tests
     the stop only at restart boundaries, so a solve that would cross 30 % after three matrix-vector
     products still builds fifteen.
  6. **A probe on a Jacobian sliced with the wrong layout.** `vk_J.npz` and the materialized coupled
     Jacobian are **field-major**: DOF `(cell i, field f)` sits at `f·n_cells + i`, fields ordered
     `[u, v, w, p, k, ω]`. Slicing it cell-major silently yields a *different matrix* that still looks
     plausible — two probes were invalidated this way. (`equilibrate_cell_major` reorders internally, so
     *after* that reorder `field = row % n_fields`. Know which side of it you are on.)


## Gates

- **Gate C — PASSED (`tests/integration/test_skewed_diffusion.py`).** With
  `CorrectedGreenGauss` injected into the residual on a 25%-skewed mesh, one Newton step
  drives `‖R‖` ~24 → ~1e-12 and reproduces a harmonic linear field to ~5e-13 (linear-exact
  on a skewed grid). The reference's lagged correction is emulated with `stop_gradient` on
  the gradient (residual value real, Jacobian omits the correction) and needs ~8
  deferred-correction sweeps — the concrete before/after. The nested gradient GMRES is
  differentiated through cleanly (forward-mode `jvp` inside the outer Newton).


## Binding decisions
- **Two-level implicit differentiation**: IFT on the converged
  Newton state (skip Newton iterations) + `custom_vjp`/adjoint on each linear solve
  (skip Krylov iterations). **Neither loop is unrolled onto the tape.** Say "no loops on
  the differentiation path," not "no loops."
- Prefer **lineax** (or `jax.scipy.sparse.linalg`) for the solve with built-in implicit
  diff; add a `custom_vjp` only where the library's differentiation is not exact through
  the converged solve. **Verify** the adjoint is a single transpose solve, not an
  unrolled iteration — this is the whole correctness claim.
- The **preconditioner is the top research risk.** Literature synthesis is done; the chosen
  direction and the traps follow. Headline: a
  **block-triangular SIMPLE-type** preconditioner using the lagged `a_P` for the Schur approximation,
  with a **fixed-cycle multigrid inner** pressure solve built once off-jit and frozen; keep the inner
  *fixed* (constant operator) so plain GMRES suffices (a *variable* inner would force FGMRES); the
  preconditioner is applied on the **RIGHT** (`solve_linear`'s default), so the Krylov residual is the true
  residual — a left-preconditioned stop is honest only for a strong `M` on a well-behaved operator
  (`potential_flow` passes `preconditioner_side="left"`), never on the shifted saddle. On **`jaxamg`**: the search confirmed it is **NVIDIA/AmgX-locked and
  scalar-only** (no coupled/saddle-point, no AMD/TPU) — usable at most as a pressure-Poisson *inner*
  escape-hatch on NVIDIA hardware, **not** the coupled solver or an architectural commitment. Do not
  adopt it on the README's word. **`LSC` original / `PCD` carry equal-order/FEM traps** (stabilized
  LSC is the Rhie–Chow form, and was built and deleted as dominated on the coupled solve; PCD needs
  FEM-BC re-derivation). **The `multigrid.py`-specific binding decisions
  this headline expands into — the pure operator-coarsening contract, the single-homed V-cycle
  recursion, the static/traced level split, strength-of-connection aggregation, `refresh_air_hierarchy`,
  the degenerate-mesh guard, and the two-level damped-Jacobi convection hierarchy — moved to
  `solve-amg-multigrid.md`'s own "Binding decisions" section in full.**
- **Where preconditioning must attach — measured, do not repeat the wrong lever.** For the
  skewed lid-driven cavity (`CorrectedGreenGauss`, `FirstOrderUpwind`) the per-Newton-step cost
  splits cleanly: the **outer coupled saddle-point GMRES takes 67 steps at 432 dof, 127 at 768
  dof — growing ~O(N)**, while the **inner gradient `A_g` solve is a flat 4 steps** regardless of
  mesh (it is volume-dominated and inherently well-conditioned). So the outer block solve is the
  whole bottleneck. **Preconditioning the *inner gradient* solve was built, measured, and
  reverted:** an inverse-volume-Jacobi `M≈A_g⁻¹` took the inner solve 4→3 steps and was
  *net-negative* end-to-end (132 s vs 121 s/step — the extra matvec per iteration outweighs the
  one iteration saved). This is the same outcome as the block-Jacobi velocity-diagonal experiment
  (`flow.md`): the cheap diagonal is not the missing physics. The real lever is an **outer**
  pressure-Schur / SIMPLE-style block preconditioner on the coupled `(u,p)` system, attaching via
  the `solve_linear(preconditioner=…)` seam.
- **The "gradient Schur elimination" is already exact and free from AD — it was never a numerical
  gap.** Feeding a gradient scheme (nested `lineax` solve `A_g g = Bφ`) into the flow residual and
  taking `jax.jvp` makes `lineax`'s implicit-diff form the exact Schur complement
  `S = ∂R/∂x + (∂R/∂g)A_g⁻¹B` *without unrolling* the inner Krylov loop. The skewed cavity with
  `CorrectedGreenGauss` **converges quadratically** (‖R‖ → 6e-12, `u_min=-0.204` vs Ghia −0.211),
  full Newton, differentiable. What remained was purely performance — not correctness or
  convergence of the absorbed gradient.
- **The efficient realization of the absorbed gradient — `SweptGradientSolve` (built, measured,
  a ~5× win).** Two costs of applying `A_g⁻¹` inside every outer matvec are separable from the outer
  iteration count above: the *per-matvec* cost and the *compile* cost of a nested implicit-diff GMRES.
  Both collapse if the constant, well-conditioned `A_g` is inverted by a **fixed number of matrix-free
  Richardson sweeps, unrolled** (no `lineax`, no implicit-diff tangent solve, no dense matrix). On the
  N=32 skewed cavity this cut a coupled Newton step from **112 s → 23 s run and 96 s → 23 s compile**
  (the compile collapse localizes the earlier blow-up to the nested Krylov + control flow), staying an
  exact drop-in (3.8e-10). Sweep count is **mesh-independent** ⇒ `O(n)`. (A *dense* LU of `A_g` was also
  built and measured but **removed** — exact yet `O((n·dim)²)`, so strictly dominated by the swept apply
  at every size; see `schemes.md`, do not rebuild.) The remaining lever is still the **outer** block
  preconditioner (the 67→127 outer iterations), which is independent of the gradient scheme.
- **Gate C (the improvement-over-reference claim):** on a non-orthogonal mesh the AD-exact
  Jacobian must converge the linear problem in **one** Newton step, where the reference
  needed several. Guard this with a test.


## Testability seam
- The Newton solver is a class constructed with an **injected residual object and
  linear-solver strategy** (CLAUDE Principle 1), so it is tested against a trivial
  analytic residual (e.g. a quadratic) with a known root and known Jacobian — no FVM
  mesh required.
- The adjoint is tested by finite-difference agreement of `jax.grad` through `solve()`
  on a small problem, plus the AD-correctness / no-NaN gate every integration suite
  carries (CLAUDE Testing Architecture).
