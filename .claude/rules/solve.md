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

## Index — where the detail lives

**This file used to be one 8,500-line document; it is now split by subsystem, each part scoped so it
auto-loads only when you touch the files it actually governs** (mirroring the root `CLAUDE.md`'s own
per-package split). What stays here — always loaded on any `aquaflux/solve/` edit — is the package-wide
contracts, the current configuration, the general Newton/adjoint binding decisions, the gates, and the
testability seam. Everything subsystem-specific moved out:

| File | `paths:` | Covers |
|---|---|---|
| `solve-direct-preconditioners.md` | `lu_preconditioner.py`, `ilu0.py`, `_ilu0.pyx` | The monolithic complete-LU preconditioner (and the now-deleted ILUT it once shared a family with), and the shared frozen-host contract |
| `solve-amg-multigrid.md` | `amg_preconditioner.py`, `multigrid.py`, `native_inverse.py`, `host_vcycle.py` | The monolithic AMG coupled PC, the JAX-native multigrid, faithful smoothed aggregation, and `multigrid.py`'s own binding decisions |
| `solve-flow-block.md` | `saddle_multigrid.py`, `shift_basis.py` | Native preconditioning of the `[u, v, w, p]` saddle — current status only |
| `solve-flow-block-log.md` | *(none — reference only)* | The full dated investigation behind the flow block, including qualified/retracted findings |
| `solve-field-split.md` | `field_split.py` | The block-triangular field split (saddle plus two transported scalars) |
| `solve-globalization.md` | `forward_step.py`, `continuation.py`, `step_control.py`, `retry.py`, `relaxation.py`, `line_search_growth.py` | Forward-step architecture, pseudo-transient continuation, line search — current status only |
| `solve-globalization-log.md` | *(none — reference only)* | The dated investigation behind the globalization architecture |
| `solve-march.md` | `march.py`, `march_log.py`, `checkpoint.py` | The observed march: `forward_march`, triggers, controls, logging |
| `solve-refuted-directions.md` | *(none — reference only)* | A cross-cutting ledger of closed/refuted ideas — check here before proposing something that sounds already tried |

The two `-log.md` files and `solve-refuted-directions.md` carry **no `paths:` frontmatter and never
auto-load** — they are tracked (so a finding can be re-adjudicated later, per the root `CLAUDE.md` rule
that findings belong in tracked files, not memory) but deliberately kept out of the auto-loaded path so
routine solver work does not pay for the full investigation history. Read them deliberately: before
re-investigating a subsystem, or before proposing an idea that might already be closed.

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
   into a new `<name>-log.md` sibling with no `paths:` frontmatter, add it to the table above, and leave
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

## ⏳ DEFERRED TO A BIGGER MESH — read this before scaling the case up

**Everything in this file is measured on `bfs3d` at 23040 cells, and one open question is deferred
*specifically* until that number grows. It is recorded here rather than buried because the trigger is a
new case, not a new idea, and whoever builds that case is the one who needs to know.**

### The trailing `[k, ω]` block's coarse grid does not scale, and nothing on this mesh can show it

**The cycle case for improving the trailing block is CLOSED — three independent measurements agree
there is nothing left there** (2026-08-14): the ablation says its *quality* buys nothing coupled (a
near-exact factorization is worse than the shipped inverse); the ordering measurement says its *retained*
coupling is worth two cycles; and the block norms say its *discarded* coupling is 0.09 % of the diagonal
blocks. **Do not go looking for coupled cycles in that half again.**

**What remains is a pure scaling argument, and it is unfalsifiable at this size.** The trailing hierarchy
runs 2 levels, so its coarse grid grows **linearly** with the mesh while its coarsest solve is a **dense
inverse — cubic to build, quadratic to store, and rebuilt at every refresh**:

| cells | trailing coarse dofs | status |
|---|---|---|
| 23040 (`bfs3d` today) | ~438 | free — invisible in any measurement here |
| ~230k | ~4400 | the dense solve starts to dominate the refresh |
| — | **8192** | `_MAX_DENSE_COARSE_DOFS` **raises outright** |

**So the symptom, when it comes, is a refresh that grows superlinearly with the mesh, or an outright
raise from `_MAX_DENSE_COARSE_DOFS`. If you are reading this because you hit one of those, this is the
entry you want.**

**The fix is to let the trailing block coarsen deeper, and both knobs now exist** — 197's shared
`NativeHierarchyInverse` base gave it `strength_threshold` and `max_levels`, which it previously lacked
(that absence was an accident of a duplicated seam, not a decision). Three conditions, each of which has
already cost someone a wrong conclusion elsewhere in this file:

1. **Sweep `max_levels` and `strength_threshold` TOGETHER, never separately.** A threshold coarsens
   *less*, so it enlarges the coarse grid. Testing one at the old 2-level cap reproduces the flow block's
   original failure and refutes the transfer for the wrong reason.
2. **Normalize `_cell_graph`'s edge weights per row-field first (UNBUILT).** It weights each cell edge by
   the **sum** of `|A_ij|` over the block, and this block's ω rows sit ~8 orders above its k rows — so a
   raw threshold there is an **ω-only** strength measure. This is the same collapse-over-row-fields
   failure that let a column-reach audit approve a configuration which then diverged the march, and that
   made the trailing bound arm unreadable for a day.
3. **A nonzero threshold forfeits the refresh compilation-cache hit** unless paired with
   `frozen_coarsening` or `shape_headroom`, because the aggregation then reads values. Safe for a
   single-state sweep, which never refreshes; **not** obviously safe in a march, and the binding decision
   keeping the k/ω path at θ = 0 in a march is unchanged.

**The harness is `validation/bfs3d_openfoam/trailing_hierarchy_sweep.py`**, which runs the block alone at
every arm. ⚠️ It, and five sibling probes, were dead on arrival until 2026-08-14 on a positional unpack —
if it fails at startup again, check that first rather than the numerics.

### The materialize used to be monolithic in a way that wasted a quarter of it — now fixed

**Found while checking whether the trailing-block fix above was the only scaling wall on this path; it
was not, and it is now closed too — see `.claude/rules/solve-field-split.md` for the measurement, the
mechanism, and the tests.** `FieldSplitAmgPreconditioner` materializes the whole six-field coupled
Jacobian with one coloured jvp probe and slices the four field-pair blocks out of it; on `bfs3d` at
23040 cells, the block that gets sliced out and discarded (`∂R_flow/∂turb`, never applied by a
flow-first split) was **22 % of the total stored nonzeros (10.5M of 47.2M)**. `ColumnProbePlan` now
takes an `active_rows` table (`solve/sparse_jacobian.py`) excluding a field-pair block from the pattern
**before** it is built, derived from the partition itself
(`FieldGroups.active_rows`, `solve/field_split.py`) and threaded through automatically wherever this case
builds its own probe under `field_split=True`. Confirmed on the real mesh through the real production
call: `nnz` **47.209M → 36.718M (−22.2 %)**, and the resulting preconditioner's `apply()` — forward and
transpose — is bit-identical to the unrestricted build's.

## Current configuration (check here FIRST — the library and the case deliberately differ)

**The library defaults and the validated `bfs3d` case bundle are not the same, and conflating them is a
recorded error.** A default here that disagrees with the code is a defect — fix it in the same change.

| | library default | validated `bfs3d` bundle | where |
|---|---|---|---|
| smoother fill | `smoother_fill_levels=1` (ILU(1)) | **0** (ILU(0)) | `coupled_amg_continuation` / `compare.py` |
| smoother sweeps | `smoother_sweeps=2` | **4** | same |
| coarse-eq limit | `coarse_eq_limit=None` (~50) | **2000** | same |
| PC shift floor | `beta_floor=0.0` | **0.05** | same |
| aggregation | plain (`pc_gamg_agg_nsmooths=0`) | plain | `amg_preconditioner.py` |
| field split | `field_split=False` | **True** | `compare.py` |
| stencil reach | `stencil_reach=3` | 3 | — |
| probe column reach | `column_reach=None` (uniform) | **(3,3,3,3,2,2)** | `compare.py` `COLUMN_REACH` |
| dual-time inner tol | `inner_tol=0.05` | **1e-2** | `compare.py` `INNER_TOL` |

**The two coupled forward solvers — always name which path you mean.** There is no
`_COUPLED_AMG_FORWARD_SOLVER` symbol.

| path | stop | norm | restart |
|---|---|---|---|
| `coupled_amg_continuation` (3D `bfs3d`) | `forward_rtol = 0.3` | **row-scaled** `coupled_scaled_norm` | 15 |
| `_COUPLED_FORWARD_SOLVER` (block-SIMPLE 2D) | `1e-2` | global 2-norm | 120 |

`_COUPLED_FACTORIZATION_FORWARD_SOLVER` (`1e-2`, global 2-norm, restart 10) is `coupled_lu_continuation`'s
default — a small-restart GMRES matched to an exact factorization. (It was named
`_COUPLED_ILUT_FORWARD_SOLVER` while the now-deleted monolithic ILUT was its other consumer; renamed
when that preconditioner was removed as dominated — see `solve-direct-preconditioners.md`.)

**Preconditioning side: RIGHT** (`solve_linear`'s default, taken by `_shifted_solve`), so the Krylov
residual is the **true** residual `b − Ax`. No solution-accuracy bound follows from the stop. `left` is
used only by `potential_flow`, where `M` is strong and the operator well-behaved.

## Contracts — the API boundary

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
    residual → `ImplicitNewtonSolver` (converges, globalizes, IFT adjoint). Do **not** reintroduce a
    fixed-count loop over `newton_step` in library code. A *test* may write one inline when the point
    is to show unglobalized Newton is insufficient (`test_scalar_continuation.py`) or to isolate a
    preconditioner (`test_turbulent_channel.py`) — that is 2 lines and self-documenting, not a class.
  - **`newton_step` is the only path that differentiates in FORWARD mode.** `ImplicitNewtonSolver` is
    a `jax.custom_vjp`, which registers only the reverse rule, so `jacfwd`/`jvp` through it raises
    `TypeError` — a JAX API consequence, not a mathematical one (the IFT gives the tangent just as
    readily; a `custom_jvp` would serve both, at the cost of the separate tight `adjoint_solver` the
    current design deliberately controls). `newton_step` is plain traced operations, so both modes
    work. This matters: the plane-wall sensitivity gate takes `jacfwd` through the whole transient
    march — one linear solve per input, the efficient direction for a scalar parameter against a
    whole field. Pinned in `tests/unit/test_newton.py`.
- **Neither function jits internally — the caller owns the jit boundary**, matching
  `ImplicitNewtonSolver`. Wrap calls in `eqx.filter_jit`; un-jitted, every operation dispatches
  eagerly. For a caller that re-solves in a loop, pass the assembler as an `equinox.Module`
  **argument** to the jitted function (`reused_flow_solve`'s pattern) so its arrays are dynamic
  leaves and the compiled solve is a cache hit; a bare captured closure is hashed by identity and
  misses every time.
- **`implicit.py` — BUILT (`ImplicitNewtonSolver`).** The nonlinear counterpart: Newton to
  convergence (`lax.while_loop`, data-dependent stop) with a reverse-mode **IFT adjoint** via
  `custom_vjp` — one transpose linear solve at the converged state, `dphi*/dtheta =
  -(dR/dphi)^{-1}(dR/dtheta)`, no Newton loop taped. `solve(residual_fn, phi0, theta)` takes the
  differentiable params `theta` explicit so the adjoint returns their cotangents. Reverse-mode
  only (`jax.grad`), which is what a scalar objective through the solver needs. This is the
  "IFT on the converged Newton state" half of the two-level scheme; it activates with the first
  nonlinear residual (the flux limiter). Verified
  (`test_implicit_solve.py`): converges a nonlinear root, gradient matches the closed form to
  1e-10, and is iteration-count-independent. Used by the limited-advection solve.
  - **Convergence guard (binding — the IFT adjoint is only valid at a root).** `_forward` carries the
    terminal residual norm out of the `while_loop` and wraps the returned field in `eqx.error_if`: if
    the residual is non-finite or above `atol + rtol·‖R₀‖` (exhausted `max_steps`, or a `NaN`/`Inf`
    that used to make `residual_norm > tol` short-circuit to `False` and exit with a poisoned field),
    it **raises `eqx.EquinoxRuntimeError`** instead of returning. The guard sits in `_forward`, so it
    fires for both the forward value and the `jax.grad` path (the fwd pass saves the guarded field),
    closing the silent-wrong-gradient hole where the transpose solve at a non-root stays well-posed
    and raises no `NaN`. The stopping test is one helper, `_within_tolerance`, shared by the loop
    `cond` and the guard. A `NaN` mid-iteration is often caught first by `lineax`'s own non-finite
    guard at the next linear solve — both are hard errors, neither is silent.
  - **✅ `solve_coupled(adjoint_solver=…)` — the transpose solve's Krylov settings are REACHABLE
    (BUILT 2026-08-14).** `ImplicitNewtonSolver` has carried an `adjoint_solver` field all along, but
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
    the two meet different operators. The forward steps solve `J + β d`, which the pseudo-transient
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
     at `max(β, beta_floor)`, because that mismatch is the shipped configuration. A probe that builds
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
  5. **A probe driven by the WRONG "march" solver — there are two, and they look interchangeable.**
     `_COUPLED_FACTORIZATION_FORWARD_SOLVER` (the complete-LU path's default) is 1 % in a plain 2-norm at
     restart 10; the coupled **AMG** builder's
     default (what `bfs3d` actually runs) is `forward_rtol` = **0.3** in the **row-scaled**
     `coupled_scaled_norm` at restart **15**. Reaching for the first while believing it is the second was
     done twice in one session — once in a sweep's self-check arm, once when adding a `forward_solver`
     seam, where it would have replaced a loose row-scaled stop with a tight Euclidean one and reported
     the difference as a restart-length effect. **It does not announce itself:** at a state where both
     converge in one cycle the self-check still passes and reports a validation it never performed.
     Build the solver from the same pieces the builder does, or take the restart through
     `coupled_amg_continuation(forward_restart=...)` rather than by supplying a whole solver.
     Note also what `forward_rtol = 0.3` *is*: an inexact-Newton forcing term on the **linear** residual
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
  adopt it on the README's word. **`LSC` original / `PCD` carry equal-order/FEM traps** (use stabilized
  LSC for Rhie–Chow; PCD needs FEM-BC re-derivation). **The `multigrid.py`-specific binding decisions
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
