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

## Settings values — case-editor default resolution coverage (from `solve.md` "Settings values", 2026-10-06)

  drift from what is built. Measured 2026-10-06: of 165 `None`-default case-file fields, 62 resolve this
  way, 24 are off, 79 stay "not set" (path- or case-dependent, or a default only an inline literal holds).

## API boundary — the #284 part 2 surface cut (from `solve.md` "Responsibility", 2026-10-11)

> **What belongs in `__all__` is what a user CONFIGURES, IMPLEMENTS or RECEIVES — not everything
> another module happens to need (#284 part 2, 2026-10-11).** 37 names left the surface that day
> (185 → 148 that day; `refuse_unforwardable_settings` followed once #541 left `solve/` its only caller): regime and batch constants, the coloured probe's plan/gather/matvecs, equilibration and
> reordering helpers, `PositiveBlockProjection` (users call `positive_block_projection`; there is no
> `positive_block_limit`/`PositiveBlockLimit` any more — the global k cap was deleted 2026-10-14),
> `CallerBuiltSource`/`FinishedSource` (what
> `explicit_source` picks), `BetaTrackingRefresh`, `FrozenTransposeFactory`,
> `MaterializedBlockPreconditioner`, `PhaseTimer`, `ShapeBudget`, `PLAIN_RESIDUAL`, `named_blocks`,
> `shift_of`, `default_dual_time_control`, `checkpoint_name`, `combine_metrics`, `in_progress_measure`,
> `InnerIterateCheckpointer`. Each still lives in its submodule; tests import it from there, and the
> harnesses that need one are listed in `VALIDATION_INTERNAL_REACHES`. (Since #284 part 1 `shift_of` is
> exported again, and the preconditioner names among these live in `precondition/` submodules —
> `PhaseTimer` in `precondition/phase_timer.py`.) **Kept although nothing outside
> `solve/` imports them**, because a user implements or chooses them: the protocols and family bases
> (`StepAcceptance`, `RefreshTrigger`, `ResidualMeasure`, `BlockInverse`, `StateBlock`, `FrozenInverse`
> and its two refinements, the five capability protocols (`StopsForRefresh` the fifth, #290), `NamedBlockMeasure`, `MeasureBuilder`), the
> `Globalization` values (`DivergenceGuard`, `MonotoneLineSearch`, `RelaxedFarFromRoot`), the results
> of public drivers (`MarchResult`, `StagedResult`) and the materialized spec's mapping reader/writer.
> **Pinned by `test_solve_api.py::test_every_export_is_used_elsewhere_or_declared_user_facing`**: an
> export must be imported by another package, named in the hand-written docs (`README.md`, `docs/*.md`
> bar the generated `api.md` and `package_structure.md`), or listed in `USER_IMPLEMENTS` /
> `USER_CONFIGURES` / `USER_CALLS` — and a listed name that no longer needs its reason fails too.
> Mutation-checked six ways. Adding to a list is a decision a reviewer sees; the list is the place to
> ask whether a user writes against the name or it merely leaked.

## Renamed 2026-10-10 (#281 part 3) — the frozen-inverse family (from `solve.md`'s section of that name)

**The rule: the suffix names the contract, the stem names the settings value that builds it.** `*Inverse`
is a `FrozenInverse` (`n_dofs` + `apply(residual, *, transpose)`); `*Preconditioner` in the host family is
a `HostPreconditioner` (the `pure_callback` matvec seam a jitted Krylov solve takes, holding one
`FrozenInverse` as `.inverse`). The value `X` builds `XInverse` (block level) or `XPreconditioner`
(materialized level). A factory function for an inverse is `<method>_inverse`. Every left-column name is
gone from the tree — no aliases.

| was | is |
|---|---|
| `HostFactors` / `RefactorableFactors` / `ReleasableFactors` | `FrozenInverse` / `RefactorableInverse` / `ReleasableInverse` |
| `HostPreconditioner.factors` | `HostPreconditioner.inverse` |
| `LuFactors` / `factorize_lu` | **deleted with the complete LU (2026-10-11)** — briefly `CompleteLuInverse` / `complete_lu_inverse` |
| `AmgVCycle` / `build_amg_vcycle` | **deleted with PETSc (2026-10-10)** — briefly `MonolithicVCycleInverse` / `monolithic_vcycle_inverse` |
| `BlockTriangularFieldSplit` / `build_block_triangular_field_split` | `FieldSplitInverse` / `field_split_inverse` |
| `AirBlockInverse` | `AirReductionInverse` (built by `AirReduction`) |
| `MonolithicLuPreconditioner` | **deleted with the complete LU (2026-10-11)** — briefly `CompleteLuPreconditioner` (built by `CompleteLu`, also deleted) |
| `MonolithicAmgPreconditioner` | **deleted with PETSc (2026-10-10)** — briefly `MonolithicVCyclePreconditioner` (built by `MonolithicVCycle`, also deleted) |
| `FieldSplitAmgPreconditioner` | `FieldSplitPreconditioner` (built by `FieldSplit`; its blocks are any `FrozenInverse`, so "Amg" was false) |
| `smoothed_multigrid_solve` / `convection_multigrid_solve` / `air_multigrid_solve`, private `_fixed_cycle_solve` | `smoothed_multigrid_cycles` / `convection_multigrid_cycles` / `air_multigrid_cycles`, `_fixed_cycles` — a FIXED cycle count with no convergence test, which "solve" contradicted |

**Deliberately NOT renamed:** the traced `state -> M` preconditioners in physics packages
(`flow.BlockPreconditioner`, the `turbulence.ScalarTransportPreconditioner` family,
`schemes.GradientPreconditioner`) — a different contract, accurately called a preconditioner, so the word
alone does not say host vs traced; the `HostPreconditioner` base does. `TracedFieldSplit` is a traced
cycle (`OffersTracedCycle`), not a `FrozenInverse`, so it takes no `Inverse` suffix. The surviving
module files keep their names; `lu_preconditioner.py` and `amg_preconditioner.py` are deleted, and `MaterializedJacobianPreconditioner` now lives in `materialized_preconditioner.py`. Measurements in these files that
name an old class were taken with the same class under its old name; the rename changed no behaviour.

## Contracts: `state.py` — the bordered state before `MassFlow.join`/`.split` (#375) and the deleted surface

  - **A bordered state is an extra named block, not a special case — AND THE BLOCK HAS TO DO THE
    PACKING (2026-09-23, #375).** The mass-flow-constrained march carries `[flow…, k, omega, beta]`;
    `MassFlow.layout(fields)` is `fields.appended(GlobalDofs("body_force", 1))`, and the block-scaled
    measure reads its `sizes` instead of hand-building `(flow_size, n, n, 1)`. ⚠️ **Declaring the
    layout is only half of it**: for months that declaration's *only* consumer was that one measure,
    while the actual packing and unpacking was `jnp.append` / `[:-1]` / `[-1]` at five sites across
    `turbulence/coupled.py` and `flow/mean_velocity.py` — so an object that read as "the bordered
    state's layout" was describing a partition nothing was packed by, and a border added anywhere but
    the end would have been wrong in five places and right in one. `MassFlow.join` / `.split` are now
    the only packing, and they read the entry's position off the layout with `slice_of`.
  - **The surface is deliberately only what is used.** `span(first, last)`, `sub(name)`,
    `width(name)` and `block(name)` were written and **deleted before the change landed**: nothing
    outside their own tests called any of them, and a nested layout is reached from the assembler
    that owns it (`momentum.layout`) rather than back out of the state it was nested into. Add one
    back when a consumer needs it, not in anticipation.

## Contracts: `materialized_session.py` — #701 and the deleted closure (from `solve.md`)

- **`materialized_session.py` + `materialized_spec.py` — MOVED to `solve/` 2026-09-19 (#450 stage 2), and on to `precondition/` (#284 part 1); the `PreconditionerSession` protocol stayed behind in `solve/session.py`.** `MaterializedSession(spec, problem, …)` is the lifecycle of a materialized-Jacobian preconditioner: one probe, one inverse and one refresh hook, each created at most once, so every step built from it carries the identical static objects. It is written against **`MaterializedProblem`**, an ABC that is everything the lifecycle needs from a residual: `assembler`, `layout`, `with_assembler`, `probe(settings, active_rows)`, `groups()` (`None` ⇒ nothing to split), `bind_march`, `shift_source`, `build_step(state, policy, march, base_regime)` — the session hands it the glued shift policy (traced split or host factor, chosen at the fit; #290), so no problem chooses the preconditioner's form. `bind_march`'s result is the problem's own to interpret (it carried `inner_refresh` for the session to set until #290, 2026-10-11 — there is no mid-step hook now); it is also where a residual validates early (coupled RANS raises its `k`-positivity refusals there, before any inverse is fitted). Implementations: `turbulence._CoupledProblem`, `flow._FlowProblem`. A `FieldSplit` inverse is refused at construction for a problem whose `groups()` is `None`. `BetaTrackingRefresh(assembler, probe, preconditioner, refit_beta_floor=, observer=)` — a plain mutable class (called per step as `(active_step, state, *, due=False)`, re-fitting when the march passes `due` and **returning whether it re-fitted**; `rebind(companion)`; its `refresh_at` mid-step method went with #290), **deliberately not an `equinox.Module`**: it re-fits the host preconditioner it is given in place (it no longer reads one off the step, which a traced policy cannot supply) and rides in a static field compared by identity, which is what makes one hook shared across rungs a compile-cache hit. The session wraps it: after a re-fit of a traced split it returns `with_traced_split(step, traced_field_split(inverse))` for the march to apply, else `None` (see `solve-march.md`). ⚠️ **There is no `beta_tracking_refresh` function any more (#661, 2026-10-10)**: it was a closure over mutable dicts with `refresh_at`/`rebind` assigned onto it as attributes while its annotation said `Callable`; its tests moved from `test_coupled_rans.py` to `tests/unit/test_beta_tracking_refresh.py` (mutation-checked eight ways). `jacobian_matvec` / `batched_jacobian_matvec` (module-level `filter_jit`, assembler an ARGUMENT so a rung is a cache hit), `frozen_shift_diagonal`, `PROBE_BATCH_SIZE` and the family regime (`VCYCLE_LINEAR_SOLVE`) came with it. The specs (`MaterializedJacobian`, `FieldSplit`, `JacobianProbeSpec`) and `MATERIALIZED_MAPPING` / `materialized_spec_from_mapping` / `materialized_spec_to_mapping` are here too; a solve with more kinds extends `MATERIALIZED_MAPPING.kinds` rather than restating them. `SessionSource` (`driver.py`) is the `ContinuationSource` over any session. **A re-pointed session re-fits the inverse and DELIBERATELY keeps the held step's shift; the target is judged through `stopping_step` (2026-10-14, #701).** `rebind` reaches only the inverse (`BetaTrackingRefresh.rebind` forces a full re-fit, at the station's Jacobian plus the shift the held step carries), so a viscosity ramp holds the anchor's shift diagonal for the whole march. That is the design, not the bug #701 first read it as: the step control adapts the scalar shift strength on the premise that the diagonal it scales is fixed, and every variant that changed the diagonal mid-march lost to holding it (table and configurations in `.claude/notes/turbulence-log.md`, "#701 — the arms"). What #701 did get right is that the row-scaled measure, which reads its row scales from the step's shift policy, judged the CASE in the anchor's scales. **`MaterializedSession.stopping_step(target)`** returns `held -> held.preconditioned_as(step on target)`, the target's step built once per build at the build's state and march keywords (`_reference_state`, `_march`; `_stopping_steps` cache cleared by every `_build`), never instrumented by `on_build` (`_step` is the uninstrumented half of `_build`); it implements **`solve.StoppingStepSource`** (`session.py`, exported, `runtime_checkable`), and `turbulence.solve_reynolds_ramp` passes it to the solve as `stopping_step` (see `solve-march.md`). ⚠️ **There is no `solve.Repointing`**: it existed on this branch for the per-station graft and went with it; `RebuildingSession` keeps its own `_march` / `_repointed_since_build`. Pinned by `test_a_re_pointed_materialized_session_keeps_the_shift_the_held_step_carries` and `..._refits_against_the_shift_the_step_carries` (both fail against the grafting session), and the four `stopping_step` tests in `test_preconditioner_session.py` (mutation-checked: built on the session's own problem, cache kept across a rebuild, held step returned unchanged, instrumented by `on_build`).

## Contracts: `session.py` — `RebuildingSession` recompile measurement (2026-10-12, #386)

  ⚠️ **`_repointed` must keep the held step's STRUCTURE, or every re-pointing recompiles the march step.**
  The march compiles against the whole step it holds; a step built from scratch on the new problem has
  hierarchies whose coarse shapes follow its values (lAIR's split, the flow block's strength
  aggregation), so its first version (a plain `build`) recompiled at every station of a viscosity ramp,
  ~40 s against ~7 s for a step (re20000 channel, 2026-10-12). `_BlockSession._repointed` refreshes the
  held policy instead (`_coupled_shift_policy(..., reuse=held policy, repointed=True)`): k/omega
  hierarchies on their reused coarsening, the flow block's hierarchy carried and applied to the new
  momentum (`flow.BlockPreconditioner.on`). That needs every preconditioner to be data, not a static
  object (see `solve-globalization.md`, the reversed #105 note). Pinned mesh-free by
  `tests/unit/test_rebuilding_session.py` and on the coupled cavity by
  `test_a_re_pointed_block_session_step_is_a_compilation_cache_hit` (both mutation-checked).
  A build after a re-pointing clears the pending re-pointing, so a ramp that rebinds to its anchor
  before the first build rebuilds nothing on step 1 (pinned,
  `test_a_build_after_a_re_pointing_is_already_on_the_new_problem`).

## Contracts: `linear.py` — side choice and the deleted `solve_linear_counted` (from `solve.md`)

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

## Contracts: `adaptive_tolerance.py` — why the reduction is report-to-report, and the gate (from `solve.md`)

  `reduction` is the step's reported residual over the PREVIOUS REPORT's (the initial residual for the
  first step), although the row-scaled measure is rebuilt every outer iteration so the two are in
  different scales; `stopping_norm`/`target` are the stopping test's (`judged(...)`, `atol + rtol·judged_reference()`).
  ⚠️ **Do not "fix" that to re-measure the step's start in its own measure** — the first build did
  (`norm_of_residual(active_step.norm(), …, prestep_state)`), unmeasured, and it lost: on the
  Reynolds-continuation target march (both rungs on the residual stop) step 1 started at 0.869 in its own
  scales against 0.217 reported, then reported 0.366, so the in-measure ratio read a 0.42 reduction (tail
  rule) where the record rose 1.7× (setback rule, 0.1); that march failed at 60 target steps (6.5e-3),
  the report-to-report rule converged in 12. A rise in the record is the signal the setback rule needs.
  - **Why not Eisenstat–Walker on every step:** that was measured and refuted (2026-10-08,
    `solve-refuted-directions.md`) — on a pseudo-transient march most steps are limited by the shift, so
    tightening every step buys no steps and costs cycles. This tightens only where the step's own
    reduction says the solve was the limit (the gate compares against the BASE, not the tolerance just
    used: a prototype gating on `1.5·eta_k` dropped back to the base one step too early, at ρ 0.145 against
    a gate of 0.137).

## Contracts: `newton.py` / `implicit.py` — why `NewtonSolver` and `_forward` went (from `solve.md`)

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
  opening ones build — measured 4/4/4 residual executions for the first three steps and **none** for
  any step after (2026-10-14, once the march's reference norm became a compiled program; it was 1 per
  step before, that one evaluation eager). `docs/` carries the
  pattern for users. **The lesson for the next decision of this shape: "which transforms does this
  break" is not answered by grepping for the transforms' names — `lax.scan` traces its body just as
  `jit` does, and a test tier found what the grep did not.** So `solve()` refuses a transform up front via
  `refuse_a_transform_the_march_cannot_run_in` (defined in `march.py` and **exported**, since
  `solve_coupled` refuses on the same terms and library code may not
  deep-import a `solve` submodule — `tests/unit/test_layering.py` fails the gate if it does). `jax.grad` is unaffected and is the mode the project needs — the march runs on stopped,
  concrete values and the derivative is attached at the root afterwards.
  - **What that cost at the four call sites, and the trap to avoid repeating.** `reused_flow_solve`,
    `bulk_velocity_flow_solve`, `scalar_pseudo_transient_solve` and `solve_coupled_mass_flow` (since deleted) each wrapped
    their solve in `eqx.filter_jit` for compile-cache reuse across sweeps; those wrappers are **removed**
    (they would now hit the refusal). The reuse is preserved one level down, because `_march_step` is
    itself `filter_jit`-compiled — **but only if the residual handed to the solve has a stable identity**.
    Each of those call sites passed a freshly built `lambda`, which is a new static cache key per call and
    would have recompiled the whole march every sweep. They now pass `assembler_residual` (the shared
    module-level `(state, assembler) -> assembler.residual(state)`), `flow.BorderedFlowResidual`
    (was `_BulkVelocityResidual`) or `_ParameterFreeResidual` — small `equinox.Module`s whose settings
    compare by value and whose arrays ride as dynamic leaves. `RootSolver.solve` binds `theta`
    into `_ResidualAt`, a module for the same reason. Pinned by
    `test_the_scalar_solve_compiles_once_whether_its_preconditioner_is_carried_or_rebuilt` (renamed
    2026-10-12, #386, when the preconditioner became data and a rebuilt one stopped recompiling).

## Contracts: `root_adjoint.py` — extraction and `adjoint_solver` reachability (2026-08-14, 2026-09-15)

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

## Measurement discipline — the full text with its numbers (from `solve.md`, as of 2026-10-17)

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

  1. **The preconditioned residual `‖Mr‖`** — a stop on it reported converged at a TRUE residual of 1.0 (the deleted PETSc path's default norm), and one level-ILU "win" was entirely this artifact. Read the true residual.
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
     ⚠️ **And `rtol` is not only a residual tolerance (found 2026-10-08).** lineax's GMRES also stops only
     once the solution's change over the last restart cycle is ≤ `rtol` (in the measure, `‖b‖` scaled to
     one), and its first counted cycle is a start-up pass that reduces nothing. A toy solve whose residual
     was 0.02 after one real cycle still ran another at `rtol = 0.1`, because the solution had moved 0.265.
     The march is insensitive to tightening the shipped `0.3` and sensitive to loosening it; moving it is the owner's call (#637). Measurements: `.claude/notes/solve-log.md`, "Measurement discipline item 5"; see also `solve-refuted-directions.md`'s Eisenstat–Walker entry.
     **`residual_stop_gmres` — the DEFAULT stop since 2026-10-15 (#645), with `AdaptiveLinearTolerance`.** It stops on the residual alone, tested every Krylov iteration in the step's measure (`norm=None` binds it, like `relative_residual_gmres`); a cycle that falls short hands the next its Arnoldi residual, and only the cycle whose recurrence meets the target recomputes `b − A x`, which the verdict reads. Selected by `LinearSolveSettings.stop` / `LinearSolveRegime.stop` (`"lineax"` | `"residual"`), and `LinearSolveRegime.solver()` is the ONE place a regime becomes a solver, so every march family reaches it. Its reported count is cycles RUN (one more than `lineax`'s for the same work), so cost triggers fire at a different difficulty and `max_restarts` counts cycles; `stop: lineax` restores the old rule (whose confirming cycle costs `2 (restart + 1) + 1` applications for a "1 cycle" solve). Pinned by `test_the_stop_reaches_the_step_s_solver_with_the_regime_and_the_step_s_measure`. History and measurements: `.claude/notes/solve-log.md`, "what the stop costs"; `.claude/notes/solve-open-directions.md` 6b.
  6. **A probe on a Jacobian sliced with the wrong layout.** `vk_J.npz` and the materialized coupled
     Jacobian are **field-major**: DOF `(cell i, field f)` sits at `f·n_cells + i`, fields ordered
     `[u, v, w, p, k, ω]`. Slicing it cell-major silently yields a *different matrix* that still looks
     plausible — two probes were invalidated this way. (`equilibrate_cell_major` reorders internally, so
     *after* that reorder `field = row % n_fields`. Know which side of it you are on.)

## Binding decisions — Gate C, `jaxamg`, absorbed gradient, `SweptGradientSolve` (from `solve.md`)

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
- **Where preconditioning must attach:** on the skewed cavity the outer coupled saddle GMRES is the bottleneck (it grows with the mesh; the inner gradient `A_g` solve is a flat few steps), and preconditioning the inner gradient solve was built, measured net-negative, and reverted — do not repeat it. The lever is the outer block preconditioner through `solve_linear(preconditioner=…)`. Measurement: `.claude/notes/solve-log.md`, "where preconditioning must attach".
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
