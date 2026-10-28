---
paths:
  - "aquaflux/solve/**"
  - "aquaflux/precondition/**"
---

# Rules — `aquaflux/solve/` (Newton + implicitly-differentiated linear solve)

Drive the residual to zero and expose an exact, iteration-count-independent adjoint. This file holds the
package-wide contracts, the current configuration, the Newton/adjoint binding decisions and the gates;
subsystem detail is in the files the Index names. Dated history: `.claude/notes/solve-log.md`.

## Index — where the detail lives

| File | `paths:` | Covers |
|---|---|---|
| `solve-direct-preconditioners.md` | `sparse_jacobian.py`, `probe_layout.py`, `kronecker_pattern.py`, `host_preconditioner.py`, `materialized_preconditioner.py` | The frozen-host contract and the materialized Jacobian; traps from the deleted incomplete-/complete-LU preconditioners |
| `.claude/notes/solve-direct-preconditioners-log.md` | *(never auto-loads)* | Dated investigation behind Jacobian materialization: probing reach, #191, materialize cost |
| `solve-amg-multigrid.md` | `multigrid.py`, `hierarchy_inverse.py`, `entry_map.py`, `padded_rows.py`, `level_fit.py`, `kronecker_pattern.py` | Traced multigrid, smoothed aggregation, `multigrid.py`'s binding decisions, the entry maps a refit applies; traps from the deleted monolithic AMG |
| `.claude/notes/solve-amg-multigrid-log.md` | *(never auto-loads)* | Dated investigation behind the traced multigrid and smoothed aggregation |
| `solve-flow-block.md` | `saddle_multigrid.py`, `simple_fit.py`, `shift_basis.py` | Traced preconditioning of the `[u, v, w, p]` saddle |
| `.claude/notes/solve-flow-block-log.md` | *(never auto-loads)* | Full dated flow-block investigation, including qualified/retracted findings |
| `solve-field-split.md` | `field_split.py`, `traced_field_split.py`, `traced_cycle.py` | The block-triangular field split (saddle plus two transported scalars) |
| `.claude/notes/solve-field-split-log.md` | *(never auto-loads)* | Dated investigation behind the field split |
| `solve-globalization.md` | `strategy.py`, `continuation.py`, `step_control.py`, `retry.py`, `relaxation.py`, `line_search_growth.py` | Forward-step architecture, pseudo-transient continuation, line search |
| `.claude/notes/solve-globalization-log.md` | *(never auto-loads)* | Dated investigation behind the globalization architecture |
| `solve-march.md` | `march.py`, `march_log.py`, `march_history.py`, `checkpoint.py`, `step_control.py`, `ramp_schedule.py` | The observed march: `newton_march`, triggers, controls, logging, ramp schedules |
| `.claude/notes/solve-march-log.md` | *(never auto-loads)* | Dated measurements behind the march's triggers, retries, controls and homotopies |
| `.claude/notes/solve-log.md` | *(never auto-loads)* | Dated investigation behind this file's contracts; the rename tables (2026-08-20 "native", 2026-09-15, 2026-10-10) |
| `.claude/notes/solve-refuted-directions.md` | *(never auto-loads)* | Cross-cutting ledger of closed/refuted ideas — grep before proposing one |
| `.claude/notes/solve-open-directions.md` | *(never auto-loads)* | Backlog of unmeasured fluid-solve speed-ups, each with a pre-registered measurement; an entry leaves once measured (win → the matching rule, loss → the ledger) |

## Where new content goes (binding — read before adding a finding)

Follow the root `CLAUDE.md` "Writing the design records". Solve-specific:
1. **Route by the Index row whose `paths:` covers the `.py` file you changed.** Only genuinely
   package-wide facts (Newton driver, linear-solve contract, adjoint decisions, current configuration)
   belong here — never subsystem detail "for visibility".
2. **Closing or refuting a direction is not done until `solve-refuted-directions.md` has its entry**, in
   the same change: what was tried, on what case/state, why it lost, where the detail is.

## Package layout and boundary

- **Two packages, one direction.** `aquaflux/solve/` is the Newton/march engine; `aquaflux/precondition/`
  (since #284) holds the preconditioners (`multigrid`, `frozen_operator`, `sparse_jacobian`,
  `jacobian_probe`, `field_split`, `traced_field_split`, `traced_cycle`, `hierarchy_inverse`,
  `saddle_multigrid`, `host_preconditioner`, `block_inverse`, `block_preconditioner`,
  `materialized_preconditioner`, `materialized_spec`, `materialized_session`, `monolithic_policy`,
  `bordered`, `phase_timer`, `entry_map`, `padded_rows`, `fused_product`, `level_fit`, `simple_fit`, `probe_layout`,
  `kronecker_pattern`). An older entry saying `solve/<one of those>.py` means the `precondition/`
  one. `precondition/` imports `solve/` and nothing else; `solve/` imports nothing outside itself (so
  every residual can run on it); every scipy/numba import is on the `precondition/` side. Pinned by
  `tests/unit/test_layering.py` (`BUILT_ON = {"precondition": ("solve",)}`). The engine holds a session
  only through the `PreconditionerSession` protocol (`solve/session.py`).
- **`__init__.py` is the API boundary (binding).** Library code imports `from aquaflux.solve import …` /
  `from aquaflux.precondition import …`, never from a submodule (enforced package-wide by
  `test_layering.py::test_one_package_reaches_another_only_through_its_public_surface`); export a new
  entry point in the same change. `__all__` holds what a user **configures, implements or receives**,
  not whatever another module happens to need — pinned by
  `test_solve_api.py::test_every_export_is_used_elsewhere_or_declared_user_facing` (an export must be
  imported by another package, named in hand-written docs, or listed in `USER_IMPLEMENTS` /
  `USER_CONFIGURES` / `USER_CALLS`; a listed name that no longer needs its reason fails). The same file
  pins that no name is exported by both packages. `validation/` harnesses' internal reaches are listed in
  `VALIDATION_INTERNAL_REACHES`, asserted both ways. `docs/conf.py` publishes both packages.
- **Deleted, so do not look for them:** PETSc (2026-10-10; `MonolithicFactorShiftPolicy.preconditioner`
  is typed `HostPreconditioner`) and the complete LU (2026-10-11; `MaterializedJacobian.inverse` is
  `FieldSplit | BlockInverse`, and a case file naming `kind: CompleteLu` is an unknown kind). Both have
  ledger entries. Any PETSc/GAMG/UMFPACK number in these files is history that cannot be re-run.

## Names

Old names are in `.claude/notes/solve-log.md` ("Renamed 2026-08-20", "Renamed 2026-09-15", "Renamed
2026-10-10", "Renamed 2026-10-22"); retired names are refused by `tests/unit/test_vocabulary.py`. No aliases exist.
- **Frozen-inverse family: the suffix names the contract, the stem the settings value that builds it.**
  `*Inverse` is a `FrozenInverse` (`n_dofs` + `apply(residual, *, transpose)`); `*Preconditioner` in the
  host family is a `HostPreconditioner` (the `pure_callback` matvec a jitted Krylov solve takes, holding
  one `FrozenInverse` as `.inverse`). Value `X` builds `XInverse` or `XPreconditioner`; a factory function
  is `<method>_inverse`. The `*_multigrid_cycles` functions run a fixed cycle count (no convergence test).
- **Deliberately not renamed:** the traced `state -> M` preconditioners in physics packages
  (`flow.BlockPreconditioner`, the `turbulence.ScalarTransportPreconditioner` family,
  `schemes.GradientPreconditioner`) — a different contract; `TracedFieldSplit` (a traced cycle, not a
  `FrozenInverse`); `stepper()` (`step()` would collide with `Globalization.step()`/`coupled_step()`);
  the `*Step` classes (`DampedNewtonStep`, `PseudoTransientStep`, `DualTimeStep`, `StepOutcome`,
  `StepReport`, `StepControl` — "step" consistently means one outer iteration); `momentum_continuation`,
  `coupled_step` (genuine pseudo-transient continuations). "Traced" (inside JAX) vs "host" is the
  distinction the retired word "native" reached for.

## Settings values — `settings_value.py`, `settings_mapping.py`

- **`BlockInverse.bound(*, report=None, stored_precision="float64")`** binds a block-inverse factory;
  `_reports` / `_stores_precision` say which a family accepts (lAIR takes neither), and asking for one it
  cannot is a `TypeError`. The session passes the split's precision as `stored_precision` where allowed.
- **`SettingsValue`** is the one home of "a frozen dataclass whose `None` fields are unset and whose
  `settings()` are the set ones"; `BlockInverse` (`block_inverse.py`, public so a configuration can
  require a value rather than a callable) and the specs in `turbulence/preconditioner_spec.py` derive
  from it. **An abstract family base is refused at construction in one place, `SettingsValue.__new__`**,
  with a `TypeError` naming the public concrete members; a new family gets this by deriving from
  `abc.ABC` with an abstract build hook — do not add a per-family `__new__`.
- **Combining two partial values is `solve.filled_from(value, base)`** (each `None` takes `base`'s), and
  turning one into a class's arguments is `_supplied` / `_merged` (in `settings_value.py`, beside
  `RootSolveSettings`, because `implicit.py` cannot import `continuation.py`). The equinox settings
  objects (`Globalization.filled_from`/`with_defaults`, `CoupledShiftSettings.filled_from`) delegate to
  it. **`MergeableSettings`** (exported, `runtime_checkable`) declares the capability;
  `merged_march_options` merges a value only when it is one (same type both sides).
  ⚠️ `None` cannot reset a shared field to its default — it means "take the base's"; write a numeric
  default out, and keep a setting whose default is itself `None` (`CoupledShiftSettings.velocity_parts`)
  out of shared options.
- **`SettingsMapping` (#391) writes and reads any family as a nested plain mapping**: `kind` = class
  name, a field equal to its default omitted (by equality, not by `None`, so a required field is always
  written; ⚠️ an array-valued field would need its own equality rule — none exists), nested values
  nested, tuples as lists. It parses nothing — the YAML parse is `case/case_file.py`; the case registry
  is `case/spec.py` (see `case.md`).
- **It checks each setting against its own field's annotation** (`typing.get_type_hints`, read once at
  construction): union → alternatives, `Literal` → values, nested → any registered kind deriving from the
  annotated class, `tuple[int, ...]` → list. Load-bearing details:
  - `bool` is a subclass of `int`: a boolean is accepted only where a boolean is annotated (this is how
    `smoother_sweeps: true` once reached a builder as `1`); an `int` position takes a whole-numbered float.
  - An unknown `kind` is reported as unknown, not as misplaced, including inside a list.
  - **An annotation the rules cannot express is a `TypeError` at construction, never a skipped field** —
    the mappings are module constants, so the package fails to import rather than load unchecked. Widen
    `_atoms` for a genuinely new form.
  - A table (`Mapping[str, X]`, e.g. a boundary condition per patch) has no `kind`: **position decides
    the reading, never the presence of a `kind` key**, so `_atoms` refuses a union of a table and a nested
    value, and a table keyed by anything but `str`. Read back as a `MappingProxyType` in file order.
  - A missing required field is refused by name and path; a nested constructor's exact
    `ValueError`/`TypeError` is re-raised with its path prepended (not at the root).
  - ⚠️ String choice sets are `Literal`s, spelled twice (a `Literal` cannot be computed from a variable):
    `BlockDiagonal.schur_scaling`/`.composition` and `prolongation_smoothing` against `SCHUR_SCALINGS`,
    `_COMPOSITIONS`, `_PROLONGATION_SMOOTHING`, pinned equal by a test.
- **`SettingsMapping.schema()` publishes the per-field rules as JSON-able data** (kinds, fields,
  `accepts` alternatives, `required`, `default`, `doc` = first paragraph of the field's NumPy `Attributes`
  entry through the MRO). Each `_atoms` alternative has `schema()` beside `accepts`/`describe`, so the form
  the case editor builds (`aquaflux schema`, `case.md`) and the check a file is read by come from one
  parse. `nested.kinds` is in registration order. Pinned by `tests/unit/test_settings_schema.py`. A kind
  may declare three `ClassVar`s (read by `getattr`):
  - `unset_resolves_to` — callables a `None` setting falls through to, searched in order for a parameter
    of the field's name; the schema reports `resolved_default`. On the block inverses it is also the
    build (`_inverse_class()` is `unset_resolves_to[0]`). ⚠️ A target is only right if no call site
    overrides it: `BlockDiagonal` is not linked to `BlockPreconditioner.build`, because the coupled march
    overrides `velocity`/`strength_threshold` inline.
  - `unset_means_off` — fields whose `None` turns the feature off (`"off": true`).
  - `not_settings` — a value's own cache (`MultipleCorrectionGradient.prepared`, `ShaftCulling.sources`),
    excluded from schema and both mapping directions via `_settings(kind)`. ⚠️ `init=False` is not an
    alternative: an equinox `init=False` field leaves the pytree, so `eqx.tree_at` cannot fill it.
  Pinned by `test_every_case_file_kinds_declarations_name_real_fields_and_readable_targets`.

## Current configuration (check here FIRST — the library and the case deliberately differ)

The library defaults and the validated `bfs3d` bundle (`validation/bfs3d_openfoam/compare.py`, from
`case.yaml`) differ; conflating them is a recorded error. A default here that disagrees with the code is a
defect — fix it in the same change.

| | library default | validated `bfs3d` bundle | where |
|---|---|---|---|
| PC shift floor | `refit_beta_floor=0.0` | 0.0 (since #712) | `PC_BETA_FLOOR` |
| inverse | none — `MaterializedJacobian.inverse` required | **`FieldSplit`** (case refuses others) | `compare.py` |
| stencil reach | `stencil_reach=3` | 3 | — |
| probe column reach | `column_reach=None` (uniform); `strong_couplings=None` | `[2, 2, 2, 3, 2, 2]` (since #839); no strong couplings | `COLUMN_REACH`, `STRONG_COUPLINGS` |
| dual-time inner tol | `inner_tol=0.05` | **1e-2** | `INNER_TOL` |
| flow (leading) inverse | none — `field_split=True` requires `leading_inverse` and `trailing_inverse` | **`SimpleSmoothedInverse`** (`FLOW_INVERSE="simplesmooth"`) | `compare.py` |
| field split precision | `FieldSplit.precision="float64"` | **`float32`** (pitzDaily too; the adjoint's preconditioner stays float64) — ⚠️ breaks at near-zero pivots without the pivot floor (below), which both case files set | `SPLIT_PRECISION` |
| pivot floor | `MaterializedJacobian.pivot_floor=None` (off) | **`PivotFloor(0.15)`** in both case files (project owner, 2026-10-19; `BFS3D_PIVOT_FLOOR` / `PITZ_PIVOT_FLOOR`, `none` off) | `PIVOT_FLOOR` |
| trailing hierarchy | `HierarchyBlockInverse`: `max_levels=2, strength_threshold=0.0, aggressive_levels=1` | **`max_levels=20, max_coarse=200, strength_threshold=0.25, aggressive_levels=0, frozen_coarsening=True`** | `JACOBI_TRAILING` |

**The coupled forward solve: one measure, per-family restart regimes.**
- **The stop is the march's own progress measure**, whatever `Convergence` names: row-scaled
  `RowScaled()` (rebuilt every outer iteration) by default, `BlockScaledNorm` under `BlockScaled()`, and
  on a held bulk velocity the row-scaled default with the constraint row appended
  (`flow.BorderedMeasures`, #541). The default solver is `relative_residual_gmres(norm=None)`, bound each
  step to the measure it is judged by (`solve-globalization.md`, `norm_builder`). ⚠️ A tolerance is only
  meaningful beside its measure — `0.3` row-scaled and `1e-2` Euclidean are not comparable.
- **The restart regime is per preconditioner family** (`LinearSolveRegime`):

| regime | where | preconditioner | rtol | restart | max_restarts |
|---|---|---|---|---|---|
| `_BLOCK_LINEAR_SOLVE` | `turbulence/coupled.py` | `BlockDiagonal` (block-SIMPLE) | 0.3 | 120 | 15 |
| `_FLOW_LINEAR_SOLVE` | `flow/march.py` | block-SIMPLE on a flow-only residual (the block calibration, not re-measured) | 0.3 | 120 | 15 |
| `VCYCLE_LINEAR_SOLVE` | `precondition/materialized_session.py` | `MaterializedJacobian(FieldSplit \| BlockInverse)` | 0.3 | 15 | 60 |

  A held bulk velocity runs on its block family's regime. Every regime carries `stop` (default
  `"residual"` since #645; see `residual_stop_gmres` below). ⚠️ `0.3` on the block family is the
  multigrid family's calibration carried across as a property of the measure; it has not been re-measured
  there.
- **Move the tolerance or restart with `linear_solve=LinearSolveSettings(rtol=…, restart=…,
  max_restarts=…)`, never by passing a whole solver as `linear_solve`** — a solver also replaces the
  stopping measure, and does not announce it.
- **Preconditioning side: RIGHT** (`solve_linear`'s default), so the Krylov residual is the true
  `b − Ax`. Only `potential_flow` (`flow/initialization.py`) passes `left` — see `linear.py` below.

## Contracts — the API boundary

- **`frozen_operator.py` — `ConvectionDiffusionStencil` is the one description of a frozen
  convection-diffusion operator** (#89; there is no `convection_diffusion_operator` /
  `_scalar_operator_pieces`). A frozen numpy dataclass (`eq=False`, off the jit path) of
  `owner, nb, coefficient, n` plus keyword-only `flux`, `boundary_diagonal`, validated at construction
  (`require_valid_graph`). Operations: `assemble()` (CSR); `diagonal_parts()` → `(convective,
  dissipative)` summing to the assembled diagonal — the k/ω pseudo-time shift is this split; `detached(cells)`
  (drop the cells' edges from both ends, unit diagonal — the k/ω fixed-cell treatment). ⚠️ `detached` is
  not `decouple_dof`, which zeroes an assembled row/column and leaves neighbours' diagonals holding the
  dropped edge (used by `flow/initialization.py`); deliberately not unified (it would change an operator).
  The turbulence builder is `_scalar_stencil(...)`. Pinned by `tests/unit/test_convection_diffusion_stencil.py`.

- **`state.py` — `FieldLayout` is the one flat field-major state layout; nothing else may re-derive
  `f * n_cells + i` (binding, #285).** An ordered tuple of named blocks over a cell count:
  `CellFields(name, n_fields)`, `SubLayout(name, layout)` (nested, read out flat), `GlobalDofs(name, count)`
  (a multiplier on no cell). Mesh-free, hashable, a zero-leaf pytree.
  - `flow/state.py::flow_state_layout(dim, n_cells)`; `turbulence/coupled.py::coupled_rans_layout(flow)`
    nests the flow layout and adds `k`, `omega` (`MomentumContinuity.layout` is public for this).
  - **A bordered state is an extra named block, and the block does the packing:**
    `MassFlow.layout(fields)` appends `GlobalDofs("body_force", 1)`, and `MassFlow.join`/`.split`
    (`flow/drive.py`) are the only packing, reading the position with `slice_of`. Never `jnp.append` /
    `[:-1]` a border.
  - `FieldGroups` is a partition view over a layout (see `solve-field-split.md`); it refuses a
    `GlobalDofs` block.
  - ⚠️ `FieldLayout` has no `dim`: use `layout.n_fields`, `layout.field_offset(name)`, and the mesh's
    `dim`. The surface is only what is used — add a method when a consumer needs it.
  - Pinned by `tests/unit/test_state.py`.

- **`RootSolveSettings` (`implicit.py`, #428) — a root solve's settings as one `SettingsValue`**
  (`convergence, max_steps, linear_solver, adjoint_solver`; `DEFAULT_ROOT_SOLVE` the empty override).
  `solver(strategy, **fields)` is the only place one becomes a `RootSolver`. Taken by
  `flow.reused_flow_solve`, `flow.bulk_velocity_flow_solve`, `turbulence.scalar_pseudo_transient_solve`,
  each beneath its own step cap (80 / 20 / 40) via `filled_from`. A setting belongs here when its reason
  needs no residual's name; `strategy` and `measures` stay with the builder. `solve_coupled` still spells
  these as keywords (its surface is what a case file describes). ⚠️ `tools/sibling_builders.py` cannot
  see these three (each returns a closure). Pinned by `tests/unit/test_root_solve_reach.py`.

- **`convergence.py` — the stopping test as one value (#370).** `Convergence(measure, rtol, atol)` with
  measures `Euclidean()` / `RowScaled()` / `BlockScaled()` (`ResidualMeasure`), each building a
  `MeasureBuilder = (step, state) -> ResidualNorm` against the problem's `ResidualMeasures`
  (`PLAIN_RESIDUAL` supports only Euclidean). `RowScaled` is rebuilt at every state; `BlockScaled` holds
  the initial state's scales (it normalizes itself, so re-basing would put the stop out of reach). There
  is no step-level measure choice: the march hands the step its measure every outer iteration.
  `RootSolver`: unset measure → the strategy's own `residual_norm`; a set one is built against
  `RootSolver.measures`. `solve_coupled` defaults to `RowScaled()` against `_CoupledMeasures`
  (bordered by `flow.BorderedMeasures` for a held bulk velocity, constraint row scaled by the held speed).

- **A residual norm read outside the compiled step goes through `norm.norm_of_residual(measure,
  residual_fn, state)` — never `measure(residual_fn(state))` eagerly (binding).** It is an
  `eqx.filter_jit` program returning `(norm, residual)` (the march keeps the vector for its first step);
  `newton_march` and `staged_march` read every reference and final norm through it (eager, each was one
  dispatch per residual operation — seconds on a coupled problem). A norm of a vector already in hand
  (the march's per-step stopping norm) is `norm.norm_of(measure, residual)`, compiled likewise.
  It inherits `_march_step`'s obligations: a residual with stable identity and a measure that changes
  only in array values (a fresh `lambda` measure recompiles every call). ⚠️ A test counting a march's
  traces as its step's compiles must warm this program first (`_warm_the_march_norm` in
  `test_newton_march.py`).
  ⚠️ **Do not jit the eager set-up phases (build, hybrid initialization, the shift source) one at a
  time**: their ~980 eager programs are one pool cached per (primitive, shape, dtype) across phases, so
  jitting one moves programs to the next, and jitting arithmetic moves rounding at ~1e-16. Ledger:
  "Compiling the eager set-up phases into a few programs".

- **`driver.py` + `shifted_step.py` — the robust march is residual-agnostic and lives here** (#448;
  Principle 3.6). **`solve.staged_march(residual_fn, state, *, strategy, source, refresh, convergence,
  measures, drift_measure, max_steps, step_control, …)`** is the segment loop: the stop target
  `atol + rtol·reference` measured once at the initial state and held; the measure builder made once;
  **the last segment marched without the trigger** (else it ends short of a root); step control
  threaded across segments while the damping reference and `drift_measure` (`state -> drift`) restart
  per segment; the final state judged in its steering measure and **refused with `EquinoxRuntimeError`
  if not a root**. Returns `StagedResult(state, strategy)`; attaching `root_adjoint` is the caller's (only
  it holds the parameter pytree). `explicit_source` picks `FinishedSource`/`CallerBuiltSource` and refuses
  (`refuse_unforwardable_settings`) any step setting beside a caller's strategy rather than drop it.
  `resume=` takes a `Resumption` (`resumption.py`; see `solve-march.md`). **`solve.shifted_step(policy,
  *, globalization, dual_time, regime, krylov_solver, adjoint_preconditioner_factory, …)`** is the tail
  every builder ends in (single shifted step vs dual-time loop; refuses inner-loop hooks with no loop and
  `refresh_on_cycles` with nothing to fire; `regime=None` keeps the step class's own solve).
  `LinearSolveSettings`, `LinearSolveRegime`, `resolve_linear_solve` live here.
  - Callers: `turbulence.solve_coupled`, `flow.solve_flow_march`, `flow.momentum_continuation`.
  - `tests/unit/test_coupled_rans.py` monkeypatches `newton_march` **on `solve.driver`**.
  - `block_reference_scales(layout, residual)` (`norm.py`) is the one home of block scales. Both block
    measures carry static `names` (refused unless naming every block once) and `per_block`;
    `norm.named_blocks` reads per-block terms for `newton_march(on_residuals=)`. `BlockScaledNorm.scales`
    is an array leaf (only `sizes` static), so a rebuild is a cache hit.
  - **`DEFAULT_SHIFT_BASIS` (`shift_basis.py`) is the one default shift basis** (`LocalCourantBasis()`);
    every shifting builder defaults to that object, pinned by identity in
    `test_every_builder_that_shifts_a_block_defaults_to_the_one_shift_basis`.

- **`jacobian_probe.py` + `monolithic_policy.py` (`precondition/`).** `JacobianProbe(plan, structure,
  narrowing)` / `jacobian_probe_plan(face_cells, n_cells, n_fields, …)` — the coloured probe, depending
  on graph and reaches alone; `narrowing` (`assembler -> assembler`, compared by value) is the one
  residual-specific part. `MonolithicFactorShiftPolicy(base, preconditioner)` pairs a base shift
  diagonal with a frozen inverse; `FrozenTransposeFactory` is its value-equal transposed apply (equality
  keeps a rung rebuild a cache hit). The traced counterpart `TracedFormShiftPolicy(base, form)` (and
  `_TracedFormAdjoint`), for a field split or a single block whose cycles are all traced, is replaced
  together by `with_traced_form` (#290; `solve-field-split.md`, `TracedForm`).
  Coupled-RANS builders: `turbulence.coupled_jacobian_probe`, `_CoupledNarrowing`.

- **`materialized_session.py` + `materialized_spec.py` (`precondition/`).** `MaterializedSession(spec,
  problem, …)` is the lifecycle of a materialized-Jacobian preconditioner: one probe, one inverse, one
  refresh hook, each made at most once, so every step carries identical static objects. It is written
  against the ABC **`MaterializedProblem`** (`assembler`, `layout`, `with_assembler`,
  `probe(settings, groups)`, `groups()` — `None` ⇒ one group, `bind_march` (where a residual
  validates early), `shift_source`, `build_step(state, policy, march, base_regime)`, and the class
  attributes `floorable_fields` and `probes_by_field` — `True` when the assembler and its narrowed stand-in
  are a `FieldwiseResidual`, so the probe runs one program per column field); the session hands
  it the glued shift policy, so no problem chooses the preconditioner's form. Implementations:
  `turbulence._CoupledProblem`, `flow._FlowProblem`. `FieldSplit` is refused for one group, a bare
  `BlockInverse` for two.
  - **`BetaTrackingRefresh`** is a plain mutable class, **deliberately not an `equinox.Module`**: called
    per step as `(active_step, state, *, due=False)`, re-fits the host preconditioner it was given in
    place and returns whether it did; rides in a static field compared by identity (one hook shared
    across rungs is a cache hit). After a re-fit of a traced form the session returns
    `with_traced_form(step, …)` for the march to apply. Tests: `tests/unit/test_beta_tracking_refresh.py`.
  - Also here: `jacobian_matvec`/`batched_jacobian_matvec`/`batched_field_jacobian_matvec` and
    `_probe_products` (the one place a fit and a re-fit choose their products) (module-level `filter_jit`, assembler an
    argument), `frozen_shift_diagonal`, `PROBE_BATCH_SIZE`, `VCYCLE_LINEAR_SOLVE`; the specs
    (`MaterializedJacobian`, `FieldSplit`, `JacobianProbeSpec`) and `MATERIALIZED_MAPPING` (extend its
    kinds rather than restate them). `SessionSource` (`driver.py`) is the `ContinuationSource` over any session.
  - **One traced form per fit, shared by every step the session builds (binding).** `_form` is taken
    at the fit and replaced after each re-fit, which also re-points the cached stopping steps
    (`with_traced_form`), so no held step pins an old copy; each copy is a whole hierarchy in the
    split's precision (0.63 GiB float32 plus the int32 indices it pins on bfs3d). A per-step split held
    three copies at once and cost 1.1 GiB of the 2-step peak (`memory_attribution.py`, 2026-10-18; log,
    "bfs3d memory by owner"). Pinned by
    `test_every_step_built_from_one_fit_carries_the_one_traced_split_it_made` (array identity, since
    `eqx.tree_at` rebuilds the module it inserts). A caller that keeps a built step for its shift policy
    should keep `step.shift_policy.base` (bfs3d's residual reporter does).
  - **A re-fit releases the superseded traced split (binding, the owner's choice of 2026-10-18):**
    `_release_superseded` calls `Array.delete()` on every leaf of the replaced split that neither the new
    split nor the inverse's current traced state holds (a frozen coarsening's reused index arrays are
    shared, so kept). **Why deleted rather than dropped:** `solve.staged_march` builds the step and passes
    it to `newton_march`, and a caller holds its arguments for the whole call, so the first split stayed
    resident for the march (~1 GiB on bfs3d, found by a stack-locals scan; `gc.get_referrers` cannot see
    a running frame's locals). **The alternative not taken (option 3):** have `newton_march` take its
    starting step through a hand-off its caller does not keep (a one-shot holder it empties), a change to
    the march's interface; with it the release could go. Consequence of the release: a step kept across
    a re-fit and then applied raises (`RuntimeError: Array has been deleted`) instead of silently applying
    a stale preconditioner — apply the update the hook returns (`MarchReplay.refit`,
    `endgame_shift_probe.py`'s `refreshed`). Pinned by
    `test_a_re_fit_releases_the_arrays_only_the_superseded_split_held` (both precisions).
  - **…and frees most of it before the fit, not after (2026-10-25).** `BetaTrackingRefresh(before_refit=)`
    is called before each re-fit's probe (ahead of the trim); the session passes `_release_outgoing_form`,
    which deletes every leaf of `_form` the inverse does not name in `refit_keeps()` (a frozen coarsening's
    plan, the SIMPLE plans, the coupling's packing layouts; `solve-amg-multigrid.md`), so the superseded
    split is not alive beside the one being fitted. An inverse that re-coarsens, or a block without
    `refit_keeps`, answers `None` and nothing is freed early. Nothing applies the preconditioner between
    the start of a re-fit and the update it returns, so no value moves (pinned bit for bit,
    `test_freeing_the_outgoing_split_early_changes_no_value_of_the_re_fit`, and the timing,
    `test_a_re_fit_frees_the_outgoing_split_before_it_fits`). ⚠️ **A re-fit that raises after the release
    leaves the session with no preconditioner, form or hook**, so the next `build` fits afresh rather than
    gluing deleted arrays (`test_a_re_fit_that_fails_after_the_early_release_leaves_the_session_to_fit_afresh`).
    `_release_superseded` still runs after the fit, for whatever was kept. bfs3d (2 steps, `main`
    `1c5a1875`, one run each, 2026-10-25): live arrays at the leading re-fit's peak 1.36 -> 1.05 GiB, peak
    5.80 -> 5.58 GiB, now set by the step program's first call; every split leaf bit-identical
    (`.claude/notes/solve-direct-preconditioners-log.md`, "Freeing the superseded split at the start of a
    re-fit").
  - **Every session gives glibc's freed memory back before each re-fit (default, the owner's decision of
    2026-10-24).** `MaterializedSession(memory_trim=)` defaults to `GlibcTrim()` (`aquaflux/allocator.py`,
    `malloc_trim(0)`, a no-op without glibc) and hands it to
    `BetaTrackingRefresh`, which calls it once per re-fit, before the probe (the hook's own default is
    `None`). `None` turns it off; a case file cannot reach it. It changes no value. **Why:** the step
    program's first call leaves ~0.6 GiB of freed memory in XLA's worker-thread arenas, and the next re-fit's
    transient landed on top of it: bfs3d 2-step peak 7.78 -> 7.30 GiB on `2e7949e8` (after #817; mean
    of three runs each, spread 0.21 -> 0.06; there re-fit #2 set the peak, ~1 GiB above the step
    program's first call, which sets it since the early release below), no cost in time; 9.04 -> 8.52 before #817
    (`.claude/notes/solve-direct-preconditioners-log.md`, "A `malloc_trim(0)` before each re-fit"). Read its effect from the resident set, not `mallinfo2`'s free
    bytes, which count trimmed pages as still free. Pinned by
    `test_the_memory_trim_runs_once_before_each_refit_and_never_on_a_skipped_step` and
    `test_a_session_gives_freed_memory_back_before_each_refit_by_default`.
  - **A re-pointed session re-fits the inverse and deliberately keeps the held step's shift (#701)** —
    the step control adapts the shift strength on the premise that the diagonal it scales is fixed, and
    every variant changing it mid-march lost (`.claude/notes/turbulence-log.md`, "#701 — the arms").
    The target is judged through **`MaterializedSession.stopping_step(target)`** (implements
    `solve.StoppingStepSource`, `session.py`): `held -> held.preconditioned_as(step on target)`, built
    once per build, never instrumented by `on_build`; `turbulence.solve_reynolds_ramp` passes it as
    `stopping_step` (`solve-march.md`). Pinned in `test_preconditioner_session.py`.
  - **`PivotFloor` (`pivot_floor.py`; `MaterializedJacobian.pivot_floor`, off when unset) moves each
    near-zero shifted pivot out of the fit.** Asked inside `BetaTrackingRefresh._materialize_at` after
    the probe and before the fit, through `refresh_in_place(diagonal_correction=)` (it sees the Jacobian's
    diagonal only, so it survives a probe straight into a split's blocks; the initial build `_fit` does not
    ask, the hook's forced first re-fit does, before any step): a row of a field the problem names (`MaterializedProblem.floorable_fields`,
    default none) whose pivot `p = J_ii + s_i` has `|p| < threshold · d_i` is fitted at `sign(p) · d_i`, with
    `d` the base pseudo-time diagonal (`ShiftTerm.diagonal`, read off the same term as the shift, never
    `shift / β`, which folds in `row_shift_scale`). The shift, the scale and the floor's arithmetic stay on
    the device (`PivotFloor.correction` is one jitted function returning a device array); only the floored
    row set is read back, for `floored_rows`. The operator, the root and the adjoint are untouched;
    the adjoint's preconditioner sees the floored fit. `BetaTrackingRefresh.floored_rows` holds the rows
    changed at the last re-fit. A floor on a problem naming no field is refused at open.
    - **Why:** a near-zero pivot makes the fitted split amplify its rows ~10⁶–10⁷× beyond any other (bfs3d
      step 19: `|M e|` 2.8e11 on two corner `k` rows, ~1e5 elsewhere). In float64 GMRES copes (`A·M`
      linear to 1e-12); in **float32** the amplified entries' rounding swamps their neighbours and `A·M`
      is wrong by 10–900 % within ~3 cells (linear only to ~1e-4 overall), so every solve hits its cap
      and the direction is decided by rounding — bfs3d's 1e-14 sensitivity. Floored, float32 and float64
      give the same direction for every perturbation.
    - **Floor only the fields that can pass through zero.** Applied to every shifted row it changed
      11 664 rows on bfs3d (11 538 momentum, whose pivots are small beside `d` without being singular)
      and the solve got worse; the coupled RANS problem names `k` alone. Raise a pivot to the row's scale,
      not just above zero (raising them to 6 % of `d` made the amplification worse), and keep healthy negative
      pivots as they are (flooring them all changed 75 rows and every solve hit its cap).
    - ⚠️ **Open: a healthy negative pivot can still amplify, and the floor leaves it alone.** At bfs3d's
      step-19 state cell 3068 (`k` pivot −7.4e-5, well outside the band) still has `|M e|` ~7e8 on its `k`
      row, ~1e3× a typical row, and `A·M` stays linear only to ~1e-4 on random vectors with the floor on.
      That step converged anyway (7 cycles). **If another case's float32 solve stalls at its cap with the
      floor on, look here first**: `pivot_floor_probe.py`'s linearity line and the solve's Euclidean residual
      above 1 are the signature; the candidates are the trailing smoother's 2×2 `[k, ω]` blocks around a
      negative pivot, or float64 for the trailing block only.
    - Measurements and the march A/B: `.claude/notes/solve-log.md`, "Near-zero pivots and the float32 split
      (2026-10-18)". Pinned in `test_pivot_floor.py`.

- **`session.py` — `RebuildingSession` (#386)**, the generic session with nothing to re-fit: steps are
  built whole from `assembler`; `rebind` re-points and marks steps stale; `refresh_preconditioner` makes
  one step on the new problem derived from the held one (`_repointed(state, held, march)`) and returns
  `held -> held.preconditioned_as(rebuilt)` (`None` when unrebound). A build after a re-pointing clears
  it. `build` refuses a `StopsForRefresh` step with `refresh_on_cycles`. Subclass supplies `_build`,
  `_repointed`, `refresh`; the one subclass is `turbulence._BlockSession`. ⚠️ **`_repointed` must keep the
  held step's structure, or every re-pointing recompiles the march step** (hierarchy coarse shapes follow
  values); `_BlockSession` refreshes the held policy (`reuse=`, `repointed=True`), which needs every
  preconditioner to be data. Pinned by `tests/unit/test_rebuilding_session.py` and
  `test_a_re_pointed_block_session_step_is_a_compilation_cache_hit`.

- **`block_preconditioner.py` — `MaterializedBlockPreconditioner`**: one `BlockInverse` (e.g.
  `SimpleSmoothed`) fitted to the whole materialized, shifted Jacobian of a single-group problem, sharing
  probe/shift/refresh with the field split; traced JAX hierarchy. Its probe is the one-group layout, so a
  refit keeps the values, shift and pivot floor on the device like the split's
  (`solve-direct-preconditioners.md`, "Every probe structure has a `BlockLayout`").

- **`linear.py` — `solve_linear(matvec, b, solver, preconditioner=None, preconditioner_side="right")`
  returns `(x, cycles)`**, a matrix-free wrapper over `lineax` whose implicit differentiation is the
  load-bearing adjoint primitive (the Krylov loop is not taped). `M`'s coefficients are `stop_gradient`ed,
  so it changes convergence only (pinned transparent in `test_preconditioning.py`).
  - **Keep both sides (binding).** Right stops on the true residual, so a weak `M` cannot falsely report
    convergence — the default and the only honest stop on the shifted saddle. Left stops on `‖M r‖`,
    right for a strong `M` on a badly conditioned SPD operator: `potential_flow` on a wall-resolved mesh
    cannot drive the true residual to tolerance in budget. Pinned by
    `test_initialization.py::test_potential_flow_survives_a_wall_resolved_aspect_ratio`.
  - **One entry point (binding, do not re-split into counted/uncounted).** The count is **restart
    cycles, not matvecs**, `int32` (a `while_loop` carry), `0` for a direct solver. A rising count on an
    unchanged system is the honest stale-preconditioner signal (wall clock moves with load). Pinned,
    including that a better preconditioner strictly lowers it.
  - **`default_linear_solver()`** (`lx.GMRES(rtol=1e-10, atol=1e-10)`, lineax's restart 20) is the
    fallback for every adjoint transpose solve; no restart wins on both operator classes measured (#114,
    `validation/adjoint_gmres_restart.py`). Re-open only for a new operator class.

- **`adaptive_tolerance.py` — `AdaptiveLinearTolerance`, the DEFAULT on every march
  (`DEFAULT_ADAPTIVE_TOLERANCE`, #645, the owner's decision with the residual stop).** Reaches
  `newton_march`, `staged_march`, `solve_coupled`, `solve_flow_march` (and the Reynolds drivers through
  `**solve_kwargs`) and the case file's `_March`; not `RootSolver`. Each step starts from the base
  tolerance (the regime's `rtol`); after an accepted step `next_tolerance(base, alpha, reduction,
  stopping_norm, target)` picks: after a setback (`alpha < 1` or `reduction > 1`) → `after_setback`
  (0.1); in the tail (full step with `reduction <= tail_gate · base`, 1.5) → `tail_scale · reduction²`
  (0.9) floored at `target_margin · target / stopping_norm` (0.5) and `minimum` (1e-10); else the base.
  Never looser than the base. Applies only where the solver is a relative-residual GMRES; `None` turns it
  off (a case file writes both rules off). `tight_solver` (the divergence retry) is never varied.
  - ⚠️ `reduction` is the step's reported residual over the **previous report's**, deliberately across
    the rebuilt row-scaled measure — do not "fix" it to re-measure the start in the step's own measure;
    that lost (a rise in the record is what the setback rule needs). Gate on the **base**, not the last
    tolerance. Measurements: `.claude/notes/solve-log.md`.
  - The tolerance rides as a float64 **array leaf** (`linear.with_relative_tolerance`;
    `linear.relative_tolerance(solver)` reads it, `None` for other solvers), so a change is a cache hit;
    a Python float would recompile. `StepReport.linear_tolerance` records it.
  - Eisenstat–Walker on every step is refuted (ledger): most pseudo-transient steps are shift-limited.
  - Pinned by `tests/unit/test_adaptive_tolerance.py` and reach tests in `test_case_solver.py`,
    `test_coupled_rans.py`, `tests/integration/test_flow_march.py`.

- **`newton.py` — `newton_step` / `newton_correction`, one correction each, `J` by `jax.jvp`. There is
  no `NewtonSolver` (binding).** Linear residual → `newton_step` (exact in one call); nonlinear →
  `RootSolver`. No fixed-count loop over `newton_step` in library code (a test may write one inline).
  `newton_step` is the only path that differentiates in **forward** mode (`RootSolver` is a
  `custom_vjp`, so `jacfwd`/`jvp` raises); the plane-wall sensitivity gate needs that. Neither function
  jits internally — the caller owns the jit boundary.

- **`implicit.py` — `RootSolver`.** Newton to convergence on `stop_array_gradients` copies of `phi0`
  and `theta`, then the IFT adjoint attached at the root by `root_adjoint` (one transpose solve);
  `solve(residual_fn, phi0, theta)`, reverse mode only.
  - **The loop is `newton_march` and there is no other (binding)**, so every hook is reachable from one
    driver. The price: **`RootSolver.solve` (and `solve_coupled`) refuse `jit`, `vmap`, `lax.scan` and
    `lax.fori_loop`** (`refuse_a_transform_the_march_cannot_run_in`, `march.py`, exported); `jax.grad`
    is unaffected. A transient march over a nonlinear residual is a Python loop whose per-step residual is
    an `equinox.Module` holding previous states (`_BdfStep`). ⚠️ "Which transforms does this break" is not
    answered by grepping for transform names — `lax.scan` traces its body like `jit`.
  - **Compile reuse needs a residual with stable identity**: pass `assembler_residual`,
    `flow.BorderedFlowResidual`, `_ParameterFreeResidual` or another small `equinox.Module` (`theta` is
    bound into `_ResidualAt`), never a call-site `lambda` (a new cache key every call). Pinned by
    `test_the_scalar_solve_compiles_once_whether_its_preconditioner_is_carried_or_rebuilt`. ⚠️ A fixture
    that reaches the solver with no work to do pins a compile property while exercising nothing.
  - **Convergence guard (binding — the adjoint is valid only at a root):** `solve()` raises
    `eqx.EquinoxRuntimeError` on exhausted `max_steps`, a collapsing constraint cap, or a non-finite norm,
    before `root_adjoint`, for forward and `grad` alike (`solve_reynolds_continuation`'s retreat catches
    it). ⚠️ `within_tolerance` compares with `<=`, so `inf <= inf` passes; the finiteness test lives in
    `newton_march`'s `converged_at`, which also refuses to step from a non-finite residual.

- **`root_adjoint.py` — `root_adjoint(residual_fn, root, theta, *, adjoint_solver=None,
  adjoint_preconditioner=None)`** returns `root` unchanged carrying its derivative (a `custom_vjp` whose
  backward rule is the transpose solve), **discarding** whatever derivative `root` carried, so any loop
  may produce it. **It does not check that `root` is a root** — the caller owns the convergence test.
  `TransposedPreconditioner` and `stop_array_gradients` (filters by `eqx.is_array`; a callable leaf in
  `theta` is refused by the `custom_vjp`) live here.
  - **`with_transposes(tree)` / `PacksTranspose` (2026-10-24) — the adjoint asks for transposes; nothing
    else holds them.** The backward rule applies `with_transposes` to the adjoint preconditioner factory
    before `jax.linear_transpose`, so every leaf with `with_transpose()` (a padded-row operator,
    `solve-amg-multigrid.md`) packs its transpose for that transpose solve only. Generic, so every
    residual reaches it; a factory must hold its operators as leaves, not close over them.
  - **`solve_coupled(adjoint_solver=…)` reaches the transpose solve** (also through
    `solve_reynolds_continuation`'s `**solve_kwargs`); `None` → `default_linear_solver()`. Build one with
    `relative_residual_gmres(rtol, restart=…, stagnation_iters=…, max_restarts=…)`. **Keep it separate
    from the forward solver**: the forward solve meets the shifted `J + βd`, the transpose solve meets
    `J` itself once, and its accuracy is the gradient's. Pinned by the reachability test
    `test_the_injected_adjoint_solver_reaches_the_transpose_solve_and_only_it` (slow).

## Measurement discipline for preconditioner probes (BINDING)

Judge a candidate preconditioner **only** by running it through GMRES, reading the **true** residual
`‖Ax−b‖`, at a state and shift pairing where the operator is hard. Each shortcut below produced a wrong
verdict that had to be retracted (full text with numbers: `.claude/notes/solve-log.md`, "Measurement
discipline — the full text").

0. **Cycle count instead of whole-march wall clock.** A cycle is a unit of cost only when candidates share
   a per-application cost (smoother sweeps, aggregation on one hierarchy). When the preconditioner's
   shape changes, only wall clock over a whole march is honest: the field split took 11% more cycles than
   the monolithic V-cycle and was 31% faster end to end.
1. **The preconditioned residual `‖Mr‖`** — it has reported convergence at a true residual of 1.0.
2. **One-apply contraction `‖MAx − x‖/‖x‖ < 1`** — not a criterion for a Krylov-accelerated
   preconditioner (ILU(0) at β=0.02 has contraction 4.5 and converges).
3. **Spectral radius of the iteration operator** — the largest mode is the smooth one, the coarse
   grid's job.
4. **A benign operating point.** Where the operator is easy every candidate looks the same, so a null
   result there is no information (an aggregation sweep read 6/6/6/6 cycles at the converged tail and
   22 → 9 at the march's hard states). Pick hard states from the march's own log; probe the state
   entering such a step; pair the operator at the march's β with the V-cycle at
   `max(β, refit_beta_floor)` (the shipped mismatch). Rank by the hardest **single** solve
   (`StepReport.max_inner_cycles`, serialized by `StateCheckpointer`), never summed cycles; and the
   hardest operators live in **rejected** attempts, which a step's record does not describe — read them
   from `march.log` (`redo step N (attempt 2)` plus the inner table).
5. **A hand-built forward solver.** It replaces the stopping **measure**, not just the tolerance, and does
   not announce itself. Use `linear_solve=LinearSolveSettings(...)`. The regime `rtol = 0.3` is an
   inexact-Newton forcing term per inner solve, not a solution tolerance; achieved reduction is often
   tighter because restarted GMRES tests only at restart boundaries. ⚠️ Under `stop: lineax`, `rtol` also
   bounds the solution's change over the last cycle, and the first counted cycle reduces nothing. The march
   is insensitive to tightening `0.3` and sensitive to loosening it; moving it is the owner's call (#637).
   **`residual_stop_gmres` is the default stop (#645)**: residual only, tested every Krylov iteration in
   the step's measure (`norm=None` binds it); only the cycle that meets the target recomputes `b − Ax`.
   Selected by `LinearSolveSettings.stop` / `LinearSolveRegime.stop` (`"lineax"` | `"residual"`);
   `LinearSolveRegime.solver()` is the one place a regime becomes a solver. Its count is cycles **run**
   (one more than lineax's for the same work), so cost triggers and `max_restarts` shift. Pinned by
   `test_the_stop_reaches_the_step_s_solver_with_the_regime_and_the_step_s_measure`. Measurements:
   `.claude/notes/solve-log.md`, "Measurement discipline item 5"; `solve-open-directions.md` 6b.
6. **A Jacobian sliced with the wrong layout.** The materialized coupled Jacobian is field-major: DOF
   `(cell i, field f)` at `f·n_cells + i`, fields `[u, v, w, p, k, ω]`; a cell-major slice is a different,
   plausible matrix. (`equilibrate_cell_major` reorders internally — after it, `field = row % n_fields`.)

## Gates

- **Gate C — PASSED (`tests/integration/test_skewed_diffusion.py`).** On a non-orthogonal mesh the
  AD-exact Jacobian converges the linear problem in **one** Newton step (with `CorrectedGreenGauss`,
  `‖R‖` ~24 → ~1e-12, linear field to ~5e-13), where a lagged correction needs ~8 sweeps. Keep it guarded.

## Binding decisions

- **Two-level implicit differentiation**: IFT on the converged Newton state + implicit diff of each linear
  solve. Neither loop is on the tape — "no loops on the differentiation path", not "no loops". Prefer
  `lineax`'s implicit diff; add a `custom_vjp` only where it is not exact, and verify the adjoint is a
  single transpose solve.
- **The preconditioner is the top research risk.** Chosen direction: block-triangular SIMPLE-type with
  the lagged `a_P` Schur approximation and a fixed-cycle frozen multigrid inner (fixed so plain GMRES
  suffices, not FGMRES), applied on the right. `jaxamg` is NVIDIA/AmgX-locked and scalar-only — at most a
  pressure-Poisson inner on NVIDIA, never the coupled solver. Original LSC/PCD carry FEM traps
  (stabilized LSC was built and deleted as dominated). The `multigrid.py` decisions are in
  `solve-amg-multigrid.md`'s "Binding decisions".
- **Preconditioning attaches to the outer coupled solve** (`solve_linear(preconditioner=…)`):
  preconditioning the inner gradient `A_g` solve was built, measured net-negative and reverted — do not
  repeat it.
- **The absorbed gradient's Schur elimination is exact and free from AD** (a nested `lineax` gradient
  solve under `jax.jvp` gives the exact Schur complement without unrolling). Its efficient realization is
  `SweptGradientSolve` — fixed, unrolled, matrix-free Richardson sweeps on the constant `A_g`
  (mesh-independent sweep count, ~5× faster per coupled step and to compile on the N=32 skewed cavity).
  A dense LU of `A_g` is dominated — do not rebuild (`schemes.md`).

## Testability seam

- `RootSolver` takes an injected residual and linear-solver strategy, so it is tested on a trivial
  analytic residual with a known root — no mesh.
- The adjoint is tested by finite-difference agreement of `jax.grad` through `solve()`, plus the no-NaN
  gate every integration suite carries.
