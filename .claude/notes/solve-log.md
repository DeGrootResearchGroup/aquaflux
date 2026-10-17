# Investigation log — `aquaflux/solve/` core (`solve.md`)

> Split out of `.claude/rules/solve.md` (2026-10-17). **Lives in `.claude/notes/`, outside the
> auto-loaded `.claude/rules/` tree, so it never auto-loads.** It holds the dated measurements and
> investigation steps behind the core file's contracts, and the two long rename tables, moved here
> verbatim so a `grep` for an old name or an old number still lands somewhere. Headings mirror the
> section or contract entry of `solve.md` each block came from; see `solve.md` for the current,
> load-bearing status. Numbers here carry the configuration they were measured under — check that
> configuration against the source before quoting one as current.

## Renamed 2026-08-20: "native" (from `solve.md`'s section of that name)


The word meant **two opposite things** and neither described a method:

- *ours, not PETSc* — `NativeSimpleInverse`, `NodalNativeInverse`, `NativeHierarchyInverse`,
  `native_saddle_inverse`, `native_nodal_inverse`, `FLOW_INVERSE="native"`;
- *run the whole solve natively on the host, **in PETSc*** — `MonolithicVCycleInverse(native=True)`,
  `has_native_solve`, `is_exact_native`, `native_forward_solve`.

The family is now named by its **level smoother**, which is the only thing its members differ in:

| was | is |
|---|---|
| `NativeHierarchyInverse` | `HierarchyBlockInverse` |
| `NativeSimpleInverse` / `native_saddle_inverse` | `SimpleSmoothedInverse` / `simple_smoothed_inverse` — *the factory is now the value object `SimpleSmoothed` (#371)* |
| `NodalNativeInverse` / `native_nodal_inverse` | `JacobiSmoothedInverse` / `jacobi_smoothed_inverse` — *the factory is now the value object `JacobiSmoothed` (#371); `air_inverse` is `AirReduction`* |
| `HostVCycleInverse` / `host_ilu_inverse` | `IluSmoothedInverse` / `ilu_smoothed_inverse` — *deleted 2026-09-13 with the ILU(0) kernel, #371; neither name exists* |
| `MonolithicVCycleInverse(native=)` / `has_native_solve` / `is_exact_native` / `native_forward_solve` | *(deleted 2026-09-13, #371 — there is no host exact forward solve, and none of these four names exists)* |
| `solve/native_inverse.py` / `solve/host_vcycle.py` | `precondition/hierarchy_inverse.py` / `solve/ilu_inverse.py` (*the latter deleted 2026-09-13, #371*) |
| `BFS3D_FLOW_INVERSE=native` | `BFS3D_FLOW_INVERSE=simplesmooth` |
| `BFS3D_TURBULENCE_INVERSE=native` | `BFS3D_TURBULENCE_INVERSE=jacobi` |

Recorded measurements in these files that said "the native arm" now say "the traced arm" — *traced*
(runs inside JAX, on device) against *host* is the distinction the old word was reaching for, and it
is the one that matters for a GPU. `hostilu` and `petsc` arm values were unchanged (both arms were removed 2026-09-13, #371): both already say
what they are. See the shipped `docs/preconditioning.md` for the user-facing description.

## Renamed 2026-09-15 — the Newton-loop unification and #373 vocabulary (from `solve.md`'s section of that name)

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
| `ShiftTerm.row_relaxation`, and every `relaxation` parameter carrying `β` (`ShiftTerm.shift`, `make_preconditioner`, `TurbulenceDamping.factor`) | `row_shift_scale`, `shift_strength` (#373, 2026-10-12) | the rest of #373's `relaxation` clash: those parameters carried `β` under the word the segregated loop uses for an under-relaxation factor, which runs the other way. `relaxation` now means only the latter. Retired names are refused by `tests/unit/test_vocabulary.py` |
| `RetryPolicy.cycles_limit` | `max_retries` (#373, 2026-10-12) | it counted redone steps -- every redo, escalating or not -- not cycles. **Not** `max_escalations`, which was the proposal on the issue: that name is `Globalization`'s cap on the shift escalations *within* one attempt, a different setting that can sit in the same case file |
| `MomentumContinuity.viscosity` (and `_VelocityGeometry.viscosity`) | `effective_viscosity` (#373, 2026-10-12) | it returns `μ + ρν_t`, while `properties["viscosity"]` -- the word everywhere else -- is the molecular value |
| `MassFlow.forced(momentum, beta)`, and `beta` for the mass-flow body force in `flow/` prose and locals | `force`, written `f` in formulas (#373, 2026-10-12) | `β` means the pseudo-transient shift strength; the solved body force is a different quantity |
| `omega` / `pressure_omega` for a smoother's damping weight (`SimpleSmoothed`, `AirReduction`, `ConvectionTwoLevel`, the multigrid cycle functions, `damped_jacobi_solve`) | `smoother_weight` / `pressure_smoother_weight` (#373, 2026-10-12) | `omega` is the SST specific dissipation rate; a case file set both a few lines apart. **A case-file key**: `SimpleSmoothed`'s `omega: 1.0` in both flagship `case.yaml`s is now `smoother_weight: 1.0` |
| `omega_smooth` (smoothed-aggregation prolongation) | `prolongation_weight` (#373, 2026-10-12) | the same clash; it weights the prolongation smoother, not a level smoother |
| `mu` for the multigrid cycle index (`SimpleSmoothed`, `SimpleSmoothedInverse`, `_VCycleOps`) | `cycle_index` (#373, 2026-10-12) | `mu` is the dynamic viscosity; 1 is a V-cycle, 2 a W-cycle |
| `v_cycles` (`BlockDiagonal`, `ScalarTwoLevel`, `ScalarAir`, the velocity blocks, `BlockPreconditioner.build`) | `cycles` (#373, 2026-10-13) | the name every other multigrid value already used for V-cycles per application. **A case-file key** on `BlockDiagonal` and the scalar blocks; no shipped case file set it |
| `f_iters` / `c_iters` (`AirReduction`, `AirReductionInverse`, `air_multigrid_cycles`) | `f_sweeps` / `c_sweeps` (#373, 2026-10-13) | smoother passes are `sweeps` everywhere else. **A case-file key** on `AirReduction`; no shipped case file set it |
| `n_group_fields` (`AirReductionInverse`) | `n_fields` (#373, 2026-10-13) | the field count of the block, as every other inverse spells it |
| `StepReport.cycles`, `StepOutcome.cycles`, and the checkpoint key `"cycles"` | `raw_cycles` (#373, 2026-10-13) | it is lineax's raw count, offset included, beside `restart_cycles` (corrected) and `max_inner_cycles`; `cycles` now means V-cycles. The march-history column follows the field. ⚠️ **Checkpoints written before this carry `cycles`**, and the two harnesses that read the step record (`damping_taper_trace.py`, `endgame_shift_probe.py`) now read `raw_cycles` with no fallback, by the project owner's choice. An inner-iterate checkpoint's `cycles` (offset-corrected, `InnerIterateCheckpointer`) is unchanged, and so are the harnesses that read one (`field_split_probe.py`, `probe_reach_audit.py`, `replay.py`; `singular_cell_probe.py` was deleted 2026-10-14) |
| `KProduction(explicit_limiter=)` | `explicit_production_limiter` (#373, 2026-10-13) | the name `SSTTurbulence` already gave the same setting |
| `FacePatches.from_dict(groups)` | `patches` (#373, 2026-10-13) | beside `patch_groups`, `groups` read as groups of patches. `CellZones` keeps the inherited `LabelledGroups.from_dict(groups)`: an override only to rename it would be a forwarding wrapper |
| `NearWallKClosure.viscosity` | `kinematic_viscosity` (#373, 2026-10-13) | `viscosity` is the molecular DYNAMIC viscosity everywhere else; this one is kinematic. `SSTTurbulence.molecular_viscosity` is kinematic too and keeps its name |
| `MaterializedJacobian(beta_floor=)`, `beta_tracking_refresh(beta_floor=)` | `refit_beta_floor=` (#373, 2026-09-22); the function itself is now the class `BetaTrackingRefresh` (#661) | the shift the **inverse is re-fitted at**. `Globalization.beta_floor` (the march's own) keeps its name; the step control's `beta_min` is a third floor and is unchanged |
| `reused_flow_solve(max_steps=)`, `bulk_velocity_flow_solve(max_steps=, solver=)`, `scalar_pseudo_transient_solve(max_steps=, rtol=, atol=, solver=)` | `root_solve=RootSolveSettings(max_steps=…, convergence=…, linear_solver=…, adjoint_solver=…)` (#428, 2026-09-22) | one value on all three, so a setting reachable from one is reachable from all — `adjoint_solver` was reachable from none |

## Responsibility / API boundary — the guard over `validation/` (2026-08-15)

> **⚠️ The guard scanned `aquaflux/` ONLY, and the boundary eroded where it did not look (extended
> 2026-08-15).** `validation/` held **16** deep imports of names `__all__` already advertises —
> `restart_cycles`, `MonolithicVCyclePreconditioner`, `symmetrically_equilibrate`, the probe-plan types —
> so "this cannot erode silently" was true of the directory under guard and false next door. Those
> harnesses are the project's re-adjudication instruments, so a rename inside a preconditioner breaks a
> *study* rather than a test: the more expensive failure, and the later-discovered one. The rule now
> enforced over `validation/` is the one needing no API decision — **if the package exports it, import
> it from the package** — and the harnesses' genuinely-internal reaches are an explicit list
> (`VALIDATION_INTERNAL_REACHES`) asserted in **both** directions, so an unlisted reach fails and so
> does a stale entry. Note the guard immediately found four violations an ad-hoc regex sweep had
> missed, because it parses imports rather than matching lines.

## Contracts: `frozen_operator.py` — migration verification (2026-10-09, #89)

  **Verified value-identical at migration (2026-10-09):** a throwaway pytest plugin wrapped the three
  hierarchy builders and `scalar_transport_shift_diagonal_parts` in every `aquaflux` module that
  imports them, and recorded each operator (canonical CSR) and shift pair per test over 16 unit modules
  (343 tests, `-m "not slow and not validation"`, jax 0.10.2, Linux, Python 3.13); all **1744** arrays
  were `np.array_equal` before and after. ⚠️ **The comparison's first run caught a behaviour change
  that no review had:** the first version refused a **scalar** coefficient by shape, which the deleted
  function had quietly broadcast, and five `test_multigrid.py` equilibration tests that build a chain
  with `coefficient=2.0` failed — read as missing recordings, not as differing ones, so a comparison
  that counted only mismatches would have reported clean. A scalar now broadcasts to a uniform array
  (pinned); an array of the wrong length is still refused. Pinned by `tests/unit/test_convection_diffusion_stencil.py` (hand-written
  three-cell operator, mutation-checked nine ways — the "drops only owner-side edges" mutation
  survived until a neighbour-only detach test was added).

## Contracts: `state.py` — why `FieldLayout` exists (#285)

  - **Why it exists.** The same concept had **three** implementations in three subpackages —
    `flow/state.py::BlockStateLayout`, `turbulence/coupled.py::CoupledRANSLayout`, and
    `precondition/field_split.py::FieldGroups` — each with its own vocabulary, none composing the others,
    and each of the first two carrying a docstring saying it existed so the arithmetic was not
    open-coded. `CoupledRANSLayout` said it carried the flow layout "verbatim" and then re-derived
    `flow_size = (dim + 1) * n_cells` and hand-sliced `pack`/`unpack`; `coupled.py::_k_block`
    computed a block range a fourth time. Adding an unknown to the flow state meant edits in three
    packages.

## Contracts: `RootSolveSettings` — what it replaced (#428)

  - **What it replaced.** All three took `max_steps`, two a forward linear solver, one `rtol`/`atol`, and
    **none** could reach `adjoint_solver` — the setting `solve_coupled` had to grow because its absence
    makes a transpose solve on a hard case raise an error naming a remedy the entry point cannot reach.
    The scalar builder's `rtol=1e-10, atol=1e-12` were a second copy of `_ROOT_CONVERGENCE`'s and are gone
    rather than moved, so the default is declared once.

## Contracts: `convergence.py` — deleting the frozen row-scaled measure (2026-09-16, #370)

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

## Contracts: `norm.norm_of_residual` — measured (#715)

  - **Measured, pitzDaily `case.yaml` at `66aec63b` (#715) against this change, back to back, warm
    compile cache, Linux x86_64, 4 cores, jax 0.11.2, one run each under `py-spy` (25 Hz):** identical
    step tables (18 steps / 123 cycles, `x_r/h` 8.0686), solve 804 -> **727 s**. By profile bucket: the
    stopping test 25.3 -> 1.6 s, the driver's references 20.4 -> 7.4 s, and the march log's field
    extraction (`turbulence.coupled_fields`, compiled in the same change) 34.4 -> 0.3 s; compilation
    122 -> 124 s.

## Contracts: `adaptive_tolerance.py` — measured on the built setting (#645)

  - **Measured on the built setting** (`validation/adaptive_tolerance_probe.py`, whose docstring holds the
    table; main `a5d61d4` + this change, Linux x86_64, 4 cores, jax 0.11.2, one run per arm; `lineax` /
    residual stop / residual + adaptive, outer steps): the field-split test file's 20×14 channel at the
    default `Convergence`, block-diagonal 27 / 41 / **31**, shipped field split 28 / 41 / **30**; the
    Reynolds-continuation adjoint test's march (`n_points=1`, both rungs on the stop under test) 26 /
    FAILS at 60 target steps (4.8e-2) / **28** (16 anchor + 12 target). On the env-switched prototype
    (same base) the target march alone also measured: fixed 0.1 / 0.03 / 0.01 throughout 16 / 13 / 12,
    tail rule alone 44, tail + setback-to-0.03 24 (so tighter is not monotonically better). Re-solving the cut-back step itself tighter (the other remedy #645 proposed)
    fails — ledger entry. pitzDaily `case.yaml` (residual stop already shipped): 18 → 21 steps, 123 → 112
    cycles, 553 → 566 s, same `x_r/h` 8.0686 — a wash there; the setback rule fired on steps 3–18 because
    a dual-time step's `alpha` is the inner minimum and sits at 0.5 / 0.25 through the ramp.

## Contracts: `root_adjoint.py` — `adjoint_solver` outcome on bfs3d (2026-08-14)

    **Outcome on the case it was built for:** with this threaded and a budget large enough not to bind,
    `jax.grad` runs on `bfs3d` for the first time and matches a central finite difference to 1.9e-04 —
    but the passthrough alone was not sufficient, because the transpose solve needs ~1450 preconditioner
    applications and the first budget allowed ~900. See `solve-flow-block.md`'s *"`jax.grad` RUNS ON THIS
    CASE"* section for the costs, the arms, and the finite-difference trap that a loose root sets.

## Contracts: `root_adjoint.py` — `default_linear_solver()` restart sweep (2026-10-09, #114)

  - **`default_linear_solver()` KEEPS lineax's restart 20 — measured, not overlooked (2026-10-09, #114).**
    The continuation path's GMRES carries `restart=40, stagnation_iters=40`, and #114 asked whether the
    adjoint's fallback should too. Swept with `validation/adjoint_gmres_restart.py` (restart 10/20/40/80,
    `stagnation_iters` 20 throughout, `rtol=atol=1e-10`, forward solve unchanged so every arm differentiates
    one root; operator applications counted exactly by wrapping the transpose matvec, true residual read
    after the solve; `81a05c5`, jax 0.11.2, lineax 0.1.1, Linux x86_64, 4 cores). **No restart wins on
    both operator classes:** on `test_flow_adjoint.py`'s skewed cavity (432–3072 unknowns, block
    preconditioner transposed) restart 20 has the lowest backward time at every size, 40 is up to 1.7× slower,
    80 is 1.3–2.1× slower; on `test_coupled_rans.py`'s 28×20 channel (2800 unknowns, `BlockDiagonal`) 40
    is ~11 % and 80 ~30 % faster than 20 (736 → 698 → 487 applications), and **10 stagnates** — the default
    sits one halving above failure there. All converged arms agree on the gradient to 4e-11. The table is
    in the harness docstring. **Why a longer restart costs on an easy solve:** the confirming-cycle cost
    measurement-discipline item 5 records for the forward solve applies to the adjoint too — restart 80
    spends exactly `2 (80 + 1) + 1 = 163` applications on every cavity, converging inside one cycle and
    paying for a second. So restart is a floor on easy solves and pays only on hard ones; an operator that
    needs a real budget (`bfs3d`, ~1450 applications) passes its own `adjoint_solver`, which no shared
    default would reach. Re-open only with a new operator class in the sweep, or if the adjoint adopts a
    residual-only stop (`residual_stop_gmres`, forward-only today), which removes the confirming cycle and
    would change this trade.

## Measurement discipline item 5 — the forward `rtol` measured (2026-10-08, #699)

     **Measured on pitzDaily (shipped `case.yaml`, 2026-10-08, jax 0.11.2, CPU; `PITZ_FORWARD_RTOL`), the
     march is not sensitive to tightening and IS to loosening:** `rtol` 0.1 / 0.3 / 0.6 / 0.9 → 245 / 203 /
     166 / 161 restart cycles, all 31 steps, 98–103 inner solves, `x_r/h` 8.0686. The shipped 0.3 is not
     changed. On bfs3d (where it was calibrated), under the residual stop at `refresh_on_cycles` 4
     (2026-10-12, #699, `46af0749`, Apple M3 Pro, jax 0.10.2, one run each), 0.1 / 0.3 / 0.6 took 774 /
     787 / 590 s at the same `x_r/h`, with one retry at 0.6 only; moving 0.3 is the owner's call (#637).
     See `solve-refuted-directions.md`'s Eisenstat–Walker entry and the open-directions note's 4b.

## Measurement discipline item 5 — what the stop costs; `residual_stop_gmres` history (2026-10-08 to 2026-10-16)

     ⚠️ **What the stop costs, replayed (2026-10-08, `krylov_recycling_probe.py`):** a solve reported as
     "1 cycle" ran TWO (`restart_cycles` subtracts the start-up pass and the cycle the solution-change
     test forces), `2 (restart + 1) + 1` applications of `A M`. On pitzDaily's 42 final-station solves the
     march's solver used 2186 applications where a residual-only stop checked every iteration needs 514.
     **`residual_stop_gmres` (BUILT, opt-in, 2026-10-08)** stops on the residual alone, tested every
     iteration in the same measure (`norm=None` binds the step's, like `relative_residual_gmres`). **A cycle
     that falls short hands the next its Arnoldi residual; only the cycle whose recurrence meets the target
     recomputes `b − A x`, and the verdict reads that** (2026-10-16): a solve applies `A M` once per
     Krylov iteration plus once at the end. From 2026-10-09 (`6d52eb1`) it recomputed at EVERY cycle end
     (one more application per non-final cycle); before that it never did. Why the end still pays: on the
     replayed pitzDaily march the recurrence sits within round-off of `‖b‖` of the true residual, harmless
     as a restart vector, but at a 1e-10 target it once declared converged a solve 0.13 % above it
     (open-directions 6b, `arnoldi_drift_probe.py`). On a
     pitzDaily march it took 693 s against 1306 s to the same root (open-directions 6b); on bfs3d (#699,
     2026-10-12, `46af0749`, Apple M3 Pro, jax 0.10.2, one run each) 810 s at `refresh_on_cycles` 3 and
     787 s at 4 against the shipped stop's 1197 s, same `x_r/h`, no retry (refresh 2 added one). **Selected by
     `LinearSolveSettings.stop` / `LinearSolveRegime.stop` (`"lineax"` | `"residual"`, 2026-10-09)**, so a
     case file writes `linear_solve: {stop: residual}`; `LinearSolveRegime.solver()` is the ONE place a
     regime becomes a solver (`shifted_step` calls it), so every march family reaches it. **It is the default
     since 2026-10-15 (#645, the project owner's request, on a PR for testing bfs3d)**, together with
     `AdaptiveLinearTolerance`, the forcing safeguard #645 asked for — see its contract entry below. Alone,
     forced as the default, it failed four slow tests (linear convergence at the end of the march, and one
     Reynolds-continuation march that goes erratic); with the adaptive tolerance all 39 pass. Its reported
     count is cycles RUN (one more than `lineax`'s for the same work), so the cost triggers fire at a
     different difficulty, and `max_restarts` counts cycles under it rather than raw `lineax` steps;
     `stop: lineax` restores the old rule. **pitzDaily's `case.yaml`
     opts in** (`stop: residual`, 2026-10-09; `refresh_on_cycles` 2 then, 3 since a cost re-fit ends the step, #290, 2026-10-11 — see `.claude/notes/solve-open-directions.md` 6b). Pinned by `test_the_stop_reaches_the_step_s_solver_with_the_regime_and_the_step_s_measure`.

## Contracts: `implicit.py` — the compile-count test that measured nothing (2026-09-15)

  - ⚠️ **That test measured nothing until this change fixed its fixture, and the reason generalizes.**
    Under a `lax.while_loop` the body is traced whether or not it executes, so the test's later sweeps —
    which started from the previous sweep's converged state and took **zero** steps — still reported
    compilations. An eager loop that takes no steps compiles nothing, so both arms collapsed to the same
    count. The fixture now disturbs the state each sweep so the march actually runs. The general form:
    **a fixture that reaches a solver in a state with no work to do can pin a compilation property while
    exercising none of the solve.**

## Binding decisions — where preconditioning must attach (skewed cavity)

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
