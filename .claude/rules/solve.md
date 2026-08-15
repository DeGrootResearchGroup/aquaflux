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

## How to read this file (read this before grepping it)

This file is long and accumulates. Three rules make a `grep` hit trustworthy:

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

**Before quoting any symbol, default or tolerance from this file, check it against the source.** Three
wrong facts were lifted from here by `grep` and asserted as current in a single session — a march
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

**The three coupled forward solvers — always name which path you mean.** There is no
`_COUPLED_AMG_FORWARD_SOLVER` symbol.

| path | stop | norm | restart |
|---|---|---|---|
| `coupled_amg_continuation` (3D `bfs3d`) | `forward_rtol = 0.3` | **row-scaled** `coupled_scaled_norm` | 15 |
| `_COUPLED_FORWARD_SOLVER` (block-SIMPLE 2D) | `1e-2` | global 2-norm | 120 |
| `_COUPLED_ILUT_FORWARD_SOLVER` (2D ILUT) | `1e-2` | global 2-norm | 10 |

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
  `M`). This is the seam the **outer block preconditioner** (below) attaches to.
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
    applications and the first budget allowed ~900. See *"`jax.grad` RUNS ON THIS CASE"* under the flow
    block for the costs, the arms, and the finite-difference trap that a loose root sets.

## Preconditioner — the frozen host family (shared contract)

- **`cell_major_permutation` / `equilibrate_cell_major` live in `frozen_operator.py`, NOT in the ILUT
  (binding, moved 2026-08-15).** They are the reorder half of one transform whose rescale half
  (`symmetrically_equilibrate`, `equilibration_scale`, `apply_symmetric_scale`, `row_chunks`) was
  already there, and every consumer applies the two together -- a factorization or a coarsening wants
  the matrix both unit-diagonal and grouped by cell.
  - **Three of the four consumers were never the ILUT**, and the V-cycle uses them *more* than it does
    (7 references against 4). They sat in `ilut_preconditioner.py` only because the threshold ILU
    needed them first.
  - **The concrete cost that removes:** the ILUT is the family member most likely to be deleted --
    dominated by the complete LU at 2D and by the AMG at 3D, per its own docstring -- and deleting it
    would have taken the monolithic AMG and the field split down with it. Those two sibling imports
    are gone; nothing outside the ILUT's own module imports from it but `MonolithicIlutPreconditioner`.
  - **Both are now exported from `aquaflux.solve`.** They were internal by `__all__` yet deep-imported
    by three study harnesses, i.e. public in practice and unguarded in principle; the harnesses now
    take them from the package surface. The permutation's unit test moved with the function, into
    `test_frozen_operator_scaling.py` beside the rescale half it belongs with.


- **The three host preconditioners share ONE application path and ONE declared contract —
  `solve/host_preconditioner.py` (BUILT, 2026-08-14).** The ILUT, the complete LU and the AMG V-cycle
  differ entirely in how the inverse is *fitted* and not at all in how it is *applied*, so
  `HostPreconditioner` owns `__init__` and `matvec()` and each subclass supplies only `build` /
  `refresh_in_place`. Those two genuinely differ (different inputs, different refresh costs) and are
  deliberately **not** unified behind a signature that would be the union of three.
  - **`HostFactors` is the contract, and it is exactly `n_dofs` + `apply(residual, *, transpose=…)`.**
    That pair was already a real structural contract satisfied by **seven** classes — the three
    factorizations, `AmgVCycle`, `NativeHierarchyInverse`, both `BlockTriangularFieldSplit`s and
    `VankaSmoother` — and declared by none, so `matvec` was written out three times byte for byte and
    `FieldSplitAmgPreconditioner` obtained it by subclassing a *concrete sibling*.
  - **⚠️ Anything a base reads off `self.factors` beyond that pair is a requirement on ALL of them
    (binding).** This is not hypothetical: `has_native_solve` read `self.factors.has_native_solve`,
    which only `AmgVCycle` has, so the property **raised** on the field split — and both call sites ask
    through `getattr(pc, "is_exact_native", False)`, whose default swallows an `AttributeError` raised
    inside a property body exactly as it swallows a missing name. The answer it produced was
    accidentally the correct `False`. If a capability is not in `HostFactors`, answer it on the
    subclass. Pinned by an AST check on the base's own source
    (`test_the_base_asks_its_factors_for_nothing_beyond_the_declared_contract`) — read off the source
    rather than exercised, because the failure is a lookup that is *never taken* on the paths a test
    would naturally drive, which is why the original went unseen.
  - **The pseudo-transient shift has one home: `sparse_jacobian.shifted_jacobian`.** All three added
    `β d` before factoring, and they **disagreed**: the AMG used a pattern-preserving `setdiag` while
    the ILUT and LU used `a + sp.diags(shift)` — the form the AMG's own docstring says is wrong, since
    a sparse *addition* stores only entries whose result is nonzero and so drops the explicit zeros a
    fixed-pattern probe deliberately kept. Benign only because the ILUT and LU never pass `structure=`;
    it would have become a silent divergence the moment either took the fixed-pattern path, which is
    the cheap one. **Measured:** the two spellings are identical in values *and* pattern on a full
    diagonal and on a matrix with diagonal entries missing (both create them), and differ only where
    explicit zeros are stored — so adopting `setdiag` everywhere is a verified **no-op** for the ILUT
    and LU and a correctness fix in general. The whole refactor is **bit-identical** end to end: an
    ILUT and an LU built and refreshed under both implementations return byte-equal `matvec` and
    transpose output.

## Preconditioner — monolithic ILUT

- **Monolithic ILUT preconditioner — BUILT (`sparse_jacobian.py` + `ilut_preconditioner.py`).** An
  incomplete-LU (threshold ILU) factorization of the **assembled coupled Jacobian**, the alternative to
  the block-triangular SIMPLE preconditioner for the coupled saddle. The block PC approximates the
  pressure Schur; the ILUT **forms the true Schur coupling `B F⁻¹ G` through its fill** instead — measured
  on the coupled RANS saddle it reaches the forward tolerance in a handful of GMRES cycles where the block
  PC needs hundreds (the block PC's wall is the Schur *approximation*, not its inversion — see the Stage-3
  note in `.claude/rules/flow.md`). Three ingredients are each load-bearing and measured: **enough fill**
  (zero-fill ILU(0) drops exactly the Schur-forming fill → a singular factor; `drop_tol=1e-6` — not
  `fill_factor` — is the binding control, keeps it); **symmetric √-diagonal equilibration** (the momentum
  and continuity rows differ in scale by orders of magnitude, which otherwise gives near-singular pivots —
  a ratio was measured but its case and state were not recorded, so re-measure before quoting a number); and
  **cell-major ordering** (interleave `[u,v,p,k,ω]` per cell so the indefinite saddle factors without a
  zero pressure pivot). The distance-1 *truncation* of the operator is catastrophic — the coupled saddle
  is intrinsically distance-2 (Rhie–Chow) and the fill is essential, so this is **not** a compact-operator
  play.
  - **`sparse_jacobian.py`** materializes the coupled Jacobian from the *same* residual the solver uses
    (no re-derived assembly): `block_stencil_colouring(owner, nb, n, reach)` (pure NumPy — the cell-block
    pattern at a stencil `reach` and a collision-free CPR colouring, the conflict graph is the pattern
    squared) then `materialize_block_jacobian(matvec, plan)` (one `jax.jvp` per colour×column-field).
    **The `(colouring, n_fields)` pair is GONE — both take a `ColumnProbePlan`** (`ColumnProbePlan.uniform(
    colouring, n_fields)` for one reach throughout, `column_probe_plan(owner, nb, n, column_reach,
    pattern_reach)` to give each column its own); `block_stencil_gather_map(plan)` likewise takes one
    argument. `jacobian_relative_error` guards that `reach` covers the stencil (coupled RANS reaches
    distance **3**). Field-major DOF layout `(cell i, field f) = f·n + i`.
    - **The colouring is by SATURATION degree, not by degree — 112 → 94 colours on `bfs3d` (BUILT,
      2026-08-11).** `_saturation_colouring` picks the uncoloured vertex whose neighbours already show
      the most *distinct* colours. The colour count **is** the probe count, so this is 108 fewer
      directional derivatives per materialize for free; the build costs 0.6 s against the degree-ordered
      greedy's 0.2 s, paid a handful of times per march. **Any collision-free colouring de-compresses to
      the same matrix**, so this changes what a materialize costs and not what it returns — which is why
      it needed no re-validation of anything downstream. Colour counts on this mesh: reach 1 **16 → 11**,
      reach 2 **60 → 39**, reach 3 **112 → 94**. Pinned by
      `test_saturation_colouring_uses_fewer_colours_than_degree_ordering` on a 3D lattice.
    - **✅ SEEDS ARE BUILT ONE CHUNK AT A TIME, not all at once (BUILT, 2026-08-11).** The probe seeds
      are `n_probes x n_fields x n_cells` floats — **441 MB on `bfs3d` at 399 probes, 624 MB at 564** —
      and the old code materialized the whole set before chunking it, holding it for the entire
      materialize. `ColumnProbePlan.seed_block(start, stop, out=)` fills a **reused** buffer of one chunk
      instead, so the seeds cost `probe_batch_size x nf` (a few MB) rather than `n_probes x nf`. Measured
      on an 18³ lattice at 546 probes: **153 MB → 1.1 MB, 136x less.** The buffer keeps one shape across
      every chunk, so the batched map still compiles a single time, and a final short chunk is padded by
      the rows `seed_block` leaves zero — the previous `np.vstack` pad is gone with it.
    - **✅ THE DE-COMPRESSION RUNS PER CHUNK TOO, so a materialize costs the matrix and nothing else
      (BUILT, 2026-08-11).** `block_stencil_gather_map` now returns a **`ProbeGather`** rather than an
      `(indptr, indices, gather_map)` tuple, and it sorts the CSR entries by the probe that feeds them.
      Each chunk's responses are scattered in as they are computed and then dropped, so the full
      `n_probes x nf` response array is gone — and with it the `np.concatenate` that appended a zero
      sentinel, which was a **second** full copy. Measured on an 18³ lattice, 546 probes, 11.5M nnz:

      | | |
      |---|---|
      | matrix data alone (irreducible) | 92 MB |
      | responses + sentinel copy (old, on top) | 306 MB |
      | **peak now** | **95 MB** |

      Scaled to `bfs3d` at 47.2M nnz that is roughly **1260 MB → 380 MB**. Out-of-reach entries need no
      sentinel: having no source probe, they never enter the sorted order, so a zero-initialized `data`
      leaves them zero. The index arrays are `int32` (entry counts and `n_probes x nf` both sit well
      inside its range), so the permanent structure is no larger than the `int64` gather map it replaced.
      **Verified against the TRUE matrix-free jvp on the real case** — `jacobian_relative_error` 2.5e-16
      (uniform) and 6.5e-16 (per-column), i.e. the float64 floor, which is an independent check rather
      than the two paths agreeing with each other.

    - **⚠️ THE GATHER MAP'S OWN CONSTRUCTION PEAKS HIGHER THAN ANYTHING ELSE IN A RUN, and the entry
      above is about the MATERIALIZE, not the build (measured 2026-08-14).** The de-compression entry
      says "the peak is the matrix plus one chunk", which is true *per materialize* and was read as
      though it described the whole path. It does not: `block_stencil_gather_map` plus
      `ProbeGather.__init__` run **once**, at case construction, and cost **100 bytes per pattern
      entry** in transient peak. The retained `ProbeGather` is ~12 B/entry, so the recorded "gather map
      ~1.2 GB" is the *retained* delta and understates the peak by roughly four times.

      *Configuration:* synthetic 14³ = 2744-cell lattice, reach-3 block-stencil pattern, 6 fields,
      143 464 cell blocks, 5 164 704 entries (`bfs3d`'s shape at ~1/9 its size); peak resident set via
      `ru_maxrss`, **one arm per process**, 5 interleaved repetitions on an idle machine, minimum
      reported. The pre-change arm is the previous commit's module imported side by side with the
      shipped one, so both are the real code rather than a re-typed stand-in. Harness
      `validation/bfs3d_openfoam/gather_map_memory.py`. ⚠️ The first run of a session reads low (a cold
      allocator gave 47 B/entry where the same arm repeated at 71) — discard it, as with every other
      probe in this file.

      | | before | after |
      |---|---|---|
      | transient peak | 517 MB (100.1 B/entry) | **339 MB (65.7 B/entry), −34%** |
      | retained | 62 MB | 62 MB, unchanged |
      | build time | 0.31 s / 0.26 s | 0.27 s / 0.24 s (**0.85× / 0.93×**) |

      Scaled to `bfs3d`'s 47.2M entries that is roughly **4.7 GB → 3.1 GB**, which makes this the
      largest single allocation in a run either way — the case build after the blocked wall-distance
      search is ~850 MB. All five repetitions of the pre-change arm read **100.1 B/entry to the
      decimal**, and reproduce the value measured earlier while a march was competing for the machine:
      peak resident set is an allocation high-water mark, so unlike wall clock it is insensitive to
      contention. Use it, not timing, when a probe must run beside other work.

      Two causes, both incidental rather than structural. The build routed an **integer** index array
      through `float64` purely to use `coo_matrix.tocsr()` as a sorter and converted back to `int64` —
      four full-length arrays at double the necessary width — and it applied the out-of-reach mask with
      `np.where`, allocating a second copy of a grid that can be written in place. Every index here has
      a bound known before it is formed (`zero_slot = n_probes · nf`), which is what lets the width be
      chosen up front (`_index_dtype`) so the wide form never materializes. **32-bit is not assumed:**
      the type falls back to `int64` above `iinfo(int32).max`, which no mesh this path is used on
      reaches (`bfs3d` is 78M against a 2.1e9 ceiling) but a much larger one would. Choosing from the
      bound also removed a latent wrap: `ProbeGather` narrowed to `int32` *unconditionally*, which
      silently truncates on a mesh that does not fit rather than falling back.

      **⚠️ FIXING THE BUILD ALONE WOULD HAVE LEFT MOST OF THIS ON THE TABLE, because
      `ProbeGather.__init__` has its own ~57 B/entry peak that was HIDDEN under the build's.** Measured
      per phase before the change: the compressed-sparse-row build 56.0 B/entry and the regrouping
      ~57 B/entry, the second invisible end to end because it fitted under the first. That is the
      transferable part — a phase whose peak sits under a larger one cannot be seen from outside, and
      becomes the ceiling the moment the larger one is reduced — and it is why both were narrowed in
      one change.

      **Verified identical, not merely convergent.** Against the pre-change module on the same plans:
      `indptr`, `indices`, `_position`, `_source` and `_probe_start` all `array_equal`, **and** the
      `scatter` output agrees on random responses — on the uniform-reach path *and* on the
      `(3,3,3,3,2,2)` per-column path, which is the one that exercises the out-of-reach mask where the
      in-place write lives. Fast (1043 passed), `slow` (46) and `validation` (18) tiers all green.

      **The one remaining full-width array is the sort permutation**, `np.argsort(probe_of)`, at 8
      B/entry — numpy returns an index array and there is no narrower form. A counting sort over the
      probe index (a few hundred values) would remove it, and is admissible because `scatter` writes to
      unique positions, so the order *within* a probe is free. It was deliberately not taken: it would
      reorder `_position`/`_source` within each probe, which forfeits the array-for-array equality that
      makes this change checkable. Take it only with a test that pins the scatter's *result* rather than
      its index arrays.

      **⚠️ Read this before tuning `_PROBE_BATCH_SIZE` against memory.** Most of the peak the batch size
      was being traded against was never the batch: seeds and responses are both `n_probes x nf` and
      neither scaled with `probe_batch_size`. Both are now gone, so the peak is the matrix plus one chunk
      — which is what makes a memory-budgeted batch size a meaningful knob rather than a knob on the
      small term. The recorded "16 vs 4: ~2.2 GB vs ~0.7 GB" was measured against the old allocations and
      **no longer describes this code**; re-measure before using it.

    - **⚠️⚠️ PER-COLUMN PROBING REACH — shipped at `(3,3,3,3,2,2)`; `p` MUST stay at reach 3.**
      `(3,3,3,2,2,2)` diverges this case on its first step (issue #191), and the cause is not the reach's
      accuracy — the matrix is exact to the floating-point floor — but the *sparsity* the shortening
      leaves behind. Keeping those positions as stored zeros does cure the divergence and is **not
      available**, because the zero-fill incomplete factorization cannot be handed stored zeros; see
      *"stored exactly-zero positions break the ZERO-SHIFT V-cycle"* below. Read the root-cause section at
      the end of this bullet before using anything in it. The 564 → 399 figure and the 5.7e-16 Frobenius agreement below are both real and both fail
      to predict the divergence, which is the instructive part — the matrix was exact and the *sparsity*
      was not.

      The probe
      is charged per (colour, **column** field), so the reach is worth choosing per column rather than
      once. Measured with `validation/bfs3d_openfoam/probe_reach_audit.py` at three states — a cold inner
      iterate (`inner-00003-02`), a developed step (`state-00072`, step 33, ‖R‖ 3.1e-6, shift 0.0068) and
      the march's hardest solve (`inner-00050-03`, 15 cycles, α → 0) — the share of each column's norm
      lying **beyond reach 2**:

      | arm | stored pattern | exact zeros | after equilibration | lost |
      |---|---|---|---|---|
      | uniform reach 3 | 47 209 392 | 8 487 718 | **38 721 674** | 18.0 % |
      | `column_reach=(3,3,3,2,2,2)` | 47 209 392 | 14 899 552 | **32 309 840** | 31.6 % |

      **6 411 834 positions — 16.6 % — that the uniform arm keeps are stripped from the operator the
      smoother factorizes**, and a zero-fill incomplete factorization takes its pattern from exactly that.
      (38.7M also matches the independently recorded live-nnz figure, which corroborates the reading.)
      **This retro-explains the padding experiment that established the reach-3 requirement:** it padded
      the distance-3 shell with **~1e-30, not exact zero** — precisely what survives the product. That is
      why positions read as causal and values did not.
      It equally explains the p/kω asymmetry: pressure is in the incomplete-LU-smoothed **leading** block,
      while k/ω are in the trailing block, whose native cell-block inverse has no factorization pattern to
      weaken.
      ⚠️ **It also violated the "full pattern, no `eliminate_zeros`, so the structure is truly fixed"
      invariant the in-place GAMG refactor relies on — for EVERY arm**, since the pruned set is whichever
      entries happen to be exactly zero at that state, which moves with the state.
      **✅ FIXED — pattern-preserving assembly, and there were TWO pruning sites, not one (BUILT).**
      Sparse *arithmetic* is what prunes, so both the shift add and the equilibration had to change:
      - `frozen_operator.apply_symmetric_scale(data, indptr, indices, scale, chunks=…)` scales the stored
        values in place (row-chunked, so the row factor's per-nonzero expansion never allocates an array
        the size of the values). `symmetrically_equilibrate` uses it in place of `diags(s) @ a @ diags(s)`.
      - `MonolithicAmgPreconditioner._shifted` adds the shift by **diagonal assignment**, not
        `a + sp.diags(shift)` — a sparse **addition** drops explicit zeros just as the product does, so
        the zeros were being lost before the equilibration ever ran. Found only because the bit-identity
        test kept failing after the first fix.
      - `ShiftedCellMajorOperator.assemble` already had the correct chunked form privately; it now calls
        the shared helper, and `_row_chunks` moved to `frozen_operator.row_chunks` rather than being a
        second copy.
      Strictly cheaper than the sparse products it replaces. Pinned by `test_frozen_operator_scaling.py`
      (pattern preserved including explicit zeros, values bit-identical to the product where it stores
      them, **and an explicit test that the old product form would have dropped them**, so it cannot
      quietly return).
      ⚠️ **The bit-identity test had SEEN this and worked around it:** it compared the two paths as dense
      arrays under the comment *"the fast path keeps the full pattern's explicit zeros, so compare as
      dense rather than by nnz"* — a dense comparison passes whether or not a path drops positions. It now
      compares `indptr`/`indices`/`data`, and against the production `_shifted` rather than a raw
      `jacobian + sp.diags(shift)` written in the test. **A workaround in a test is a defect report;
      read it as one.**
      **✅ CONFIRMED ON THE CASE — the acceptance test passes bit for bit.** With the fix and the case
      back at its `COLUMN_REACH=(3,3,3,2,2,2)` default, `bfs3d` rung-1 step 1 reproduces the uniform-reach
      baseline in **every reported digit**: `‖R₀‖` 3.2901e-01, β 0.5000, 4 inner, **3 cycles**,
      `‖R‖` 2.046e-01, α 1.000, no flags — against the diverged arm's 44 cycles, `‖R‖` 3.758e-01, α 0.000.
      So the values were never the problem and the stored pattern always was.
      **Expect a cost shift, not only a win:** the trailing block carries ~37 % more stored entries
      (5 245 488 against 3 839 276) into its factorization than it did when equilibration pruned them, so
      per-cycle cost there may rise even where cycle counts fall. Judge on march wall clock, not cycles —
      and **not on the confirming run above, whose wall clock is void** (forced past the load guard with
      the test tiers running; cycles and step counts are unaffected, timings absorbed the contention).
      Still open: a full march to mid-span `x_r/h` 8.361, and what the restored pattern does to the
      `equilibrate=True` arm, which is a different question from this one.
      **THE MECHANISM IS COLOUR ALIASING, and it is the MAJORITY of a shortened column, not an edge
      case (measured structurally 2026-08-12, `validation/bfs3d_openfoam/column_reach_collisions.py` —
      mesh graph only, no state, no Jacobian, no solve).** A colouring is collision-free only on the
      pattern it was built at, so under the reach-3 assembly pattern two cells sharing a reach-2 colour
      can both couple to one row; the response holds their **sum** and de-compression charges all of it
      to whichever lies in reach. On `bfs3d` (23040 cells, 1 311 372 pattern cell-blocks):

      | column | reach | entries in reach | corrupted | share | far entries folded in | rows touched |
      |---|---|---|---|---|---|---|
      | u, v, w | 3 | 1311372 | 0 | 0% | 0 | 0 |
      | p, k, ω | 2 | 537896 | 287315 | **53.4%** | 369396 | **23025 of 23040** |

      **By this measure `p`, `k` and `ω` all close inside reach 2 and the velocities do not.** The
      table is kept because a column-norm measure does **not** license shortening a column — it collapses
      over row fields, and on this system the ω rows set it alone — so read it as evidence about the
      method, not as the licence. The shipped value is `(3,3,3,3,2,2)` — `p` at reach 3 with only k and ω
      shortened, the pattern held at reach 3 — giving **454 probes against 564** and a **−16 % probe cost
      per build**.
      **Why it is a case property and not a library constant:** it follows from the *schemes*. First-order
      upwind on k/ω plus a non-orthogonal diffusion correction reaches 2; the velocity columns carry the
      limited second-order upwind reconstruction out to 3. Re-measure for any case that changes them —
      the library default (`column_reach=None`) probes every column at `stencil_reach`, unchanged.

      **⚠️ THE MEASUREMENT THAT SETS THE DEFAULT (2026-08-12).** At `(3,3,3,2,2,2)` the `bfs3d` march took **44 restart cycles
      at step 1 instead of 3**, α collapses to 0.000, and by step 2 β is at its 16.0 ceiling with ‖R‖ an
      order of magnitude ABOVE its start. The p row is the signature: **1.402e-01 after step one against
      6.217e-07**. Restoring p to reach 3 — `(3,3,3,3,2,2)` — reproduces the uniform-reach trajectory
      *exactly* (identical ‖R‖, cycles, β and α over the first 24 steps to the log's four figures) and
      keeps a **−16 % probe cost per build** (8.4 s against 10.0 s; 13.9 s at the pre-#188 baseline).
      Verified 67 steps / 319 cycles / 1960 s, mid-span `x_r/h` 8.36 unchanged.

      **Why the audit above did not catch it, as a HYPOTHESIS and not a measurement.** The shell norm is
      a **column-relative** quantity — the share of a column's norm beyond the reach — while the damage
      from folding is **entry-relative**: the aliased mass lands on particular positions, and what
      matters is its size against *the entry it lands on*. In an incompressible collocated saddle the
      (p,p) block is near-zero by construction (it carries only the Rhie–Chow damping), so folding even
      1e-15 of column mass onto it can be a large relative perturbation of exactly the entries an
      incomplete LU turns into pivots. That would explain why p audits as clean at 9.8e-16 and still
      breaks the smoother, while k and ω — whose columns are read only by the turbulence rows, under a
      per-cell block inverse — are genuinely inert. **This is not established.** What would settle it is
      an entry-relative audit (folded mass over the magnitude of the receiving entry, per block) rather
      than a column-norm one; `probe_reach_audit.py` reports the column-relative measure only.

      **The safe reading until then: a shell norm licenses shortening a column only when that column's
      entries are not themselves near-zero.** Shortening `k`/`ω` is measured safe on this case;
      shortening `p` is measured catastrophic on this case; the two are not distinguished by any number
      in the table above.
      **⚠️ "Safe" here means AT POSITIVE SHIFT, which is the only regime a march visits. At β = 0 —
      the adjoint's operator — shortening `k`/`ω` is measured to DOUBLE the cost** (22 restart cycles
      against uniform reach's 11 to the same 1e-11 floor; see the stored-zeros entry below for the full
      configuration). It still converges, so this is a cost to know about rather than a reason to widen
      the march's reach, but do not carry "proven safe" across the β boundary.
      **⚠️ AND `(3,3,3,2,2,2)` IS ONLY SOUND IF THE FACTORIZATION KEEPS STORED ZEROS, WHICH IT MUST NOT.**
      Shortening `p` is safe only when the dropped positions are held in the pattern as stored zeros —
      and stored zeros are exactly what `AmgVCycle._live` removes, because the incomplete factorization
      cannot take them. So with the operator pruned at the factorization the `p` column is back in the
      configuration that diverges the case, and the case default must stay `(3,3,3,3,2,2)`. **Predicted
      from the mechanism, not re-measured on a march** — the divergence itself is the measured #191
      result, at β > 0 under a pruning assembly.
      **⚠️ A SHORT REACH CORRUPTS RATHER THAN TRUNCATES, and this is the trap the design turns on.** A
      colouring is collision-free only for the pattern it was built at, so two cells sharing a reach-2
      colour may still both couple to a common row at distance 3; the response then holds the *sum* and
      the de-compression charges all of it to the near entry. So a column with far couplings has its
      **near** entries perturbed — it is not simply missing the far ones. Correspondingly, every pattern
      entry outside its column's own reach is written as an **explicit zero** rather than gathered
      (`ColumnProbePlan.in_reach` plus a zero sentinel row): gathering there would deposit a *different*
      cell's near coupling into a position the matrix has nothing in. Both halves are pinned
      (`test_a_short_probed_column_zeroes_its_out_of_reach_entries`,
      `test_per_column_plan_recovers_a_mixed_reach_matrix_exactly_and_more_cheaply`).

      **✅ ROOT CAUSE AND FIX (2026-08-12) — it was NOT the aliasing, and it was NOT the reach. Sparse
      arithmetic was pruning the explicit zeros out of the smoother's pattern.** Two readings of #191
      preceded this and both are refuted; the falsifying harnesses are in `validation/bfs3d_openfoam/`.
      - **Aliasing is real and pervasive but is not the cause.** A reach-2 colouring is collision-free
        only on its own pattern, so under the reach-3 assembly pattern two cells sharing a colour can
        both couple to one row and the response is charged entirely to the near one:
        `column_reach_collisions.py` (mesh graph only, no state, no solve) finds **287 315 of 537 896
        entries — 53.4 % — corrupted in every shortened column**, touching 23 025 of 23 040 rows.
        Yet `trailing_column_reach_probe.py`, which measures the assembled error *exactly* (seed the
        colour class, minus seed the one cell), puts the pressure column's error at **1.4e-17 … 4.0e-16
        at the rung ICs — the float64 floor, and the divergence is at step 1, so no other state is
        involved.** A perturbation that size cannot take a solve from 3 cycles to 44.
      - **The cause: `a + sp.diags(shift)` and `diags(s) @ a @ diags(s)` are sparse arithmetic, which
        stores only entries whose result is nonzero.** `column_reach` writes out-of-reach entries as
        **exact `0.0`**; a uniform build stores the true value there (~1e-26, nonzero). Exact zeros were
        stripped, 1e-26 survived. Measured (`column_reach_zero_pruning.py`): same 47 209 392 stored
        pattern in both arms, but **38 721 674 against 32 309 840 after assembly — 6 411 834 positions,
        16.6 %, removed from what the ILU(0) smoother factorizes.** A zero-fill factorization takes its
        pattern from exactly that.
      - **Why `p` and not `k`/`ω`:** pressure is in the incomplete-LU-smoothed **leading** block; k and ω
        are in the trailing block, whose cell-block inverse has no factorization pattern to weaken. That
        also retro-explains the padding experiment which established the reach-3 requirement — it padded
        with **~1e-30, not exact zero**, precisely what survives sparse arithmetic.
      - **⚠️ CURED BY PATTERN-PRESERVING ASSEMBLY, AND THAT CURE IS NOT AVAILABLE — so `p` STAYS at reach
        3.** Preserving the pattern (`frozen_operator.apply_symmetric_scale`, plus a diagonal assignment
        for the shift) does exactly what this entry says: `(3,3,3,2,2,2)` then reproduces the uniform-reach
        baseline in **every reported digit** at rung-1 step 1 — `‖R₀‖` 3.2901e-01, β 0.5000, 4 inner,
        **3 cycles**, `‖R‖` 2.046e-01, α 1.000, no flags — and the **full march** converges to the recorded
        root: 67 steps, 320 cycles, 4 escalations, final ‖R‖ **3.586e-06**, mid-span `x_r/h` **8.3611**,
        ν_t peak **150.1071**, with the trailing inverse's `equilibrate` setting inert (both settings
        step-for-step identical). ⚠️ That run's **wall clock is void** (its early steps ran against the
        test tiers); its step and cycle counts are not.
        **But what it preserves are stored EXACT ZEROS, and the zero-fill incomplete factorization cannot
        take those** — measured at zero shift, carrying them costs 58 restart cycles at a true relative
        residual of 2.299e-02 against 11 cycles to 8.474e-11 without (see *"stored exactly-zero positions
        break the ZERO-SHIFT V-cycle"*). They are therefore pruned at the factorization boundary
        (`AmgVCycle._live`), which puts `p` at reach 2 back in the configuration above. **The shipped
        default is `(3,3,3,3,2,2)`** — 454 probes against 564, −16 % — and `p` at reach 3 IS a correctness
        constraint after all. The march numbers here stand as a measurement of the preserving arm; they are
        not a licence for the shortened default under the shipped one.
      - ⚠️ **How it got through, which is the transferable part: the audit that licensed it AND the gate
        that verified it both collapse over row fields**, on a matrix whose row norms differ by ~8 orders
        (k row 8.8e-06, ω row 1.4e+03). `column_reach_ladder` takes a whole column block;
        `jacobian_relative_error` is one random vector over the whole matrix. Neither can see a wrong
        pressure block, and `probe_reach_audit.py` already had the right function — `block_shell_fractions`,
        per (row field, column field) pair. **Read a reach or exactness measurement per PAIR, never over a
        whole column or a whole matrix.**

      **✅ MEASURED, and CPU time tracks the probe count 1:1** (`state-00077`, at the then-default
      `_PROBE_BATCH_SIZE = 4` — since moved to 8, so the RATIO below stands and the absolute seconds
      are ~10 % lower now,
      four interleaved repetitions after a discarded warm-up, minimum reported):

      | | uniform reach 3 | per-column | |
      |---|---|---|---|
      | probes | 564 | 399 | −29% |
      | **process CPU** | 65.74 s | 46.74 s | **−29%** |
      | wall | 10.36 s | 7.11 s | −31% |

      Per-arm spread was ≤5% and the arms do not overlap (worst per-column 7.50 s beats best uniform
      10.36 s). **The CPU row is the load-bearing one:** it is nearly insensitive to other processes
      competing for cores, and it moves exactly with the probe count, which is what establishes that the
      saving is work removed rather than a scheduling artifact.
      **⚠️ Against the ORIGINAL shipped probe (672, degree-ordered colouring at uniform reach 3) the
      combined saving is 672 → 399 = −41%, but that arm was NOT timed in this run** — it is a probe-count
      ratio extrapolated on the 1:1 CPU-vs-probes relation above. Time it before quoting a second.
      **⚠️ THREE EARLIER ATTEMPTS AT THIS NUMBER WERE EACH CONFOUNDED, in three different ways, and the
      method is the finding.** (1) The batched jvp compiles on its first call, so whichever arm ran first
      carried the trace and the second looked cheap — 12.4 s vs 6.3 s, which is the compile, not the
      probe. (2) A concurrent 4.4 GB march drove the machine 8 GB into swap and the result came out
      *inverted*, 10.4 s vs 11.8 s. (3) The machine is shared and rarely idle. The method that works:
      discard a warm-up, interleave the arms so a load excursion hits both, repeat, report the minimum,
      and carry CPU time beside wall as the cross-check.
      **✅ WHY the velocity columns reach further than the scalars: the EDDY VISCOSITY's dependence on
      the strain rate (measured 2026-08-11, `validation/bfs3d_openfoam/column_reach_probe.py`).** The
      momentum and scalar transport equations look alike, so the split invites an explanation, and the
      two obvious ones are both wrong. The harness perturbs one degree of freedom and takes a single
      directional derivative — that *is* the column, and how far the response spreads on the cell graph
      is its reach — so it costs megabytes rather than a materialize's gigabytes and can sweep arms
      freely. At `state-00077` (step 28, ‖R‖ 3.586e-06, shift 0.0064), from the four deepest interior
      cells:

      | arm | u | v | w | p | k | ω |
      |---|---|---|---|---|---|---|
      | Venkatakrishnan-limited linear upwind (the case) | 3 | 3 | 3 | 2 | 2 | 2 |
      | unlimited second-order upwind | 3 | 3 | 3 | 2 | 2 | 2 |
      | first-order upwind on momentum | 3 | 3 | 3 | 2 | 2 | 2 |
      | compact (one-shot) Green-Gauss gradient | 3 | 3 | 3 | 2 | 2 | 2 |
      | `a_P`'s lagged flux with no velocity gradient | 3 | 3 | 3 | 2 | 2 | 2 |
      | **`nu_t` with `stop_gradient` on the strain rate** | **2** | **2** | **2** | 2 | 2 | 2 |

      **`nu_t = a1 k / max(a1 omega, S F2)` reads the strain-rate magnitude, which is built from the
      velocity gradient.** That makes the eddy viscosity a *velocity-dependent coefficient* of a flux
      that is already gradient-based, and the composition of the two is what spends the extra ring.
      It also accounts for the columns that do NOT reach three: `k` and `omega` enter `nu_t` **pointwise**
      (no gradient), and pressure does not enter it at all.
      **Three explanations are REFUTED — do not re-propose them:** the flux limiter and the second-order
      reconstruction (first-order upwind on momentum, the very scheme k/ω use, still reaches three); the
      gradient scheme's own stencil (a one-shot compact Green-Gauss still reaches three); and the
      velocity gradient inside `a_P`'s lagged convective flux. The last was *my* account, argued from the
      source and killed by the arm built to test it — the lesson being that the ring is not where the
      pressure-velocity coupling is, it is in the closure.
      **⚠️ The arm is a DIAGNOSTIC, not a proposal.** `stop_gradient` on the strain rate changes the
      *Jacobian*, not the residual, so it is a quasi-Newton linearization rather than a model change.
      It is worth knowing that it would take the whole operator to reach two — but the probe feeds the
      **preconditioner**, and a reach-2 *pattern* is separately measured to break the hierarchy (below),
      so this is not a route to 234 probes without re-opening that.

      **This does NOT reopen the reach-2 lever below, and must not be read as contradicting it.** That
      entry is about probing *every* column at reach 2 **and coarsening the reach-2 pattern**, which
      changes the hierarchy (3 levels / 480 coarse equations against 2 / 1296) and fails. Here the
      pattern stays at reach 3, so the coarse space and the smoother see exactly what they see today; only
      columns with *nothing* beyond reach 2 are shortened, and the velocities — which do carry content
      there — are not.
    - **Materialize efficiency — two shipped speedups, both AMG-path-only, bit-identical (BUILT).** The
      probe dominates a refresh, so `materialize_block_jacobian` takes two optional accelerators the AMG
      preconditioner passes (LU/ILUT keep the plain loop, which any NumPy matvec supports). **(1) Batched
      probing** — `batched_matvec` (a `jax.vmap` of the jvp, **built once and reused** so it compiles a
      single time; `probe_batch_size` chunks it for memory) runs the coloured probes as a few fused passes
      instead of a Python loop of separate calls. Measured 22.4→14.0 s (~1.6×) on `bfs3d` — modest because
      CPU forward-AD does not vectorize across the batch like a GPU (the win is dispatch amortization). For
      the SAME reason the chunk was once kept **small** (`_PROBE_BATCH_SIZE` was 4, not 16): a larger
      batch holds more simultaneous forward-AD tapes. **The default is now 8** — see the sweep below,
      which is what retired that reasoning.
      **✅ RE-MEASURED, and the memory half of that trade NO LONGER EXISTS (2026-08-11,
      `validation/bfs3d_openfoam/probe_batch_sweep.py`; `bfs3d` `state-00077`, 399 probes at the
      per-column reach, 47.2M nnz, warm-up discarded per chunk shape, min of 3):**

      | batch | wall (s) | cpu (s) | python peak | vs batch 1 |
      |---|---|---|---|---|
      | 1 | 11.73 | 91.37 | 380 MB | 1.00× |
      | 2 | 8.42 | 70.29 | 383 MB | 1.39× |
      | **4 (shipped)** | 6.69 | 53.89 | 388 MB | 1.75× |
      | **8** | 5.94 | **46.35** | 399 MB | 1.98× |
      | 16 | **5.81** | 49.13 | 419 MB | 2.02× |
      | 32 | 6.47 | 52.33 | 460 MB | 1.81× |

      - **Memory barely moves: 380 → 460 MB across batch 1 → 32**, about 2.5 MB per unit of batch. The
        old "16 vs 4: ~2.2 GB vs ~0.7 GB" was the seed set and the response array, and **neither ever
        scaled with the batch**; both are gone. So the memory argument for a small chunk is void.
      - **The curve has an interior optimum and TURNS: 32 is worse than 16** (6.47 s against 5.81 s), so
        "bigger is better up to memory" was never the shape. CPU time — the cleaner axis on a shared
        machine — bottoms at **8**.
      - **The default is now `_PROBE_BATCH_SIZE = 8`** (was 4), which is where processor time bottoms.
        16 is a hair faster in wall and slower in CPU; on a shared machine CPU is the more trustworthy
        axis, and the wall difference between 8 and 16 is 0.13 s on a 6 s probe.
      **⚠️ AUTO-SIZING THE BATCH TO AVAILABLE MEMORY IS NOT WORTH BUILDING, and this is why.** The
      motivation was that the right chunk depends on machine state and case size. At 2.5 MB per unit it
      does not: any sane fixed value is safe, and the constraint that motivated the mechanism was an
      artifact of allocations that have since been removed. Fix the allocation, not the knob.
      **⚠️ WHERE THE MEMORY ACTUALLY GOES, measured by attribution rather than assumed:** peak RSS is
      149 MB after imports, **6889 MB after `build_case`**, 7154 MB after the colouring and plan, 8359 MB
      after the gather map. So the case assembly is ~6.7 GB, the gather map ~1.2 GB, and a whole
      materialize 0.4 GB. **A run's memory ceiling is set by building the case, not by probing it** — the
      probe is now a rounding error against it, and an initial reading that blamed the structure build was
      wrong.
      **(2) Gather de-compression** — `block_stencil_gather_map(plan)` precomputes, once, the
      **fixed full-pattern** CSR structure and the de-compression that fills it (no scatter loop, no
      per-materialize CSR re-sort), passed as `structure=`. ⚠️ It used to return an
      `(indptr, indices, gather_map)` tuple consumed as one vectorized `data = responses.ravel()[gather_map]`;
      it now returns a `ProbeGather` that scatters **chunk by chunk**, so no such expression exists and the
      full response array it indexed is gone (see the chunked-de-compression entry above). The **full** pattern (no `eliminate_zeros`) is the fixed mesh distance-3 colouring graph
      — a superset of the Jacobian's live nonzeros at *any* state — so the structure stays fixed
      cold→developed and no entry is ever dropped as values change: a guaranteed-fixed structure the in-place
      GAMG refactor needs. On `bfs3d` the full pattern is ~47.2M positions, but that is a **structural
      over-estimate**, mostly explicit zeros at every state — LIVE nnz is ~constant (**38.7M cold / 39.0M
      developed**; there is *no* cold→developed nnz collapse — an earlier "47.2M cold → 39.0M developed"
      reading conflated the fixed pattern with live nnz). **The stored zeros must be pruned before the
      operator reaches an incomplete factorization, and `AmgVCycle._live` is where that now happens** — see
      *"stored exactly-zero positions break the ZERO-SHIFT V-cycle"* below, which measures what they cost and
      corrects the two mechanisms this bullet used to assert for it. De-compression 22.4→11.2 s (2.0× vs
      loop); full build (materialize + GAMG) only 56.0→54.2 s (**1.03×** — GAMG-dominated), so the gather's
      real value is the fixed-structure invariant, not the wall-clock.
    - **⚠️⚠️ STORED EXACTLY-ZERO POSITIONS BREAK THE ZERO-SHIFT V-CYCLE, and BOTH obvious mechanisms are
      REFUTED (measured 2026-08-12, harness `validation/bfs3d_openfoam/zero_pattern_pivots.py`).** The
      coloured probe stores the full block-stencil pattern, of which **8.03M of 47.21M positions are exactly
      zero**; whether they survive into the preconditioner depends only on how the shift and the
      equilibration are *spelled*. A sparse product or addition stores only nonzero results and deletes them;
      an in-place diagonal assignment and value scale cannot, and keeps them.

      *Configuration:* `bfs3d` `state-00067` (converged, ‖R‖ 3.586e-06), **operator and preconditioner both
      at β = 0**, right-hand side the real steady residual `−R(state)`, monolithic V-cycle, plain
      aggregation, ILU(0) ×4, `coarse_eq_limit` 2000, pattern reach 3, GMRES restart 15 to rtol 1e-8 judged
      on the **TRUE** residual.

      | column reach | shift / equilibration | nnz | stored zeros | cycles | TRUE rel |
      |---|---|---|---|---|---|
      | uniform 3 | prune / prune | 39.18M | 0 | **11** | **8.474e-11** |
      | uniform 3 | preserve / preserve | 47.21M | 8.03M | 58 | **2.299e-02** |
      | uniform 3 | **preserve / prune** | 39.18M | 0 | **11** | **8.474e-11** |
      | 3/3/3/3/2/2 | prune / prune | 36.97M | 0 | 22 | 8.545e-11 |
      | 3/3/3/3/2/2 | preserve / preserve | 47.21M | 10.24M | 58 | 2.299e-02 |

      - **⚠️ AT THE FORWARD MARCH'S OWN OPERATING POINT EVERY ARM TIES, so a march cannot reveal any of
        this.** Same harness at `state-00066` (step-initial, operator β 0.0096 with the V-cycle at the 0.05
        floor — the shipped mismatch): **all five arms give 4 restart cycles at a true relative residual of
        1.435e-13** (1.444e-13 at the shortened reach). Not a cycle of difference, from either the stored
        zeros or the reach. So the damage is **confined to the zero-shift adjoint**, and the pattern-preserving
        change's clean march revalidation was correct rather than lucky — its validation and its failure mode
        genuinely do not overlap. This is the "a benign operating point cannot discriminate" rule biting on
        exactly the axis that was being changed: β = 0 is the only state on this case that separates these arms.
      - **Pruning at EITHER stage restores the good operator bit-identically** (11 cycles, 8.474e-11), so the
        fix is not "which stage prunes" but "prune before the factorization". Shipped as
        **`AmgVCycle._live`**, at the boundary where the operator reaches PETSc, so the assemblers can stay
        pattern-preserving (which is what makes the fixed-pattern fast path and the generic sparse path agree
        entry for entry) while the factorization still gets the live pattern.
      - **⚠️ A PIVOT CENSUS CANNOT DETECT THIS — all four arms are IDENTICAL: min |pivot| 1.546e-01, zero
        negative pivots, median 1.020.** So is the coarse space's size (2 levels, 1296 coarse equations in
        every arm). Both were the natural diagnoses, both were measured, and both are blind here; do not
        spend a run on either again. **The blindness holds at BOTH states, i.e. over nine arms:** at
        `state-00066` every arm reads min |pivot| 2.145e-01 and median 1.017 — the shift lifts the smallest
        pivot from 1.546e-01, which is the shift doing its job on the diagonal — while the arms there tie in
        convergence and at β = 0 they differ five-fold. A census that is constant across arms that converge
        identically *and* across arms that do not is not measuring the thing that separates them. What *is* demonstrated (small-scale, PETSc directly) is that the stored
        zeros are genuinely kept by PETSc (`nz_used` counts them, in the `Mat` and in the ILU factor) and
        that the smoother's **action** differs, i.e. the elimination does deposit fill into them.
      - **⚠️ Why a denser incomplete factor is a WORSE smoother on this operator at zero shift is NOT
        established.** Retaining fill can only make an incomplete factorization a closer approximation to
        `A⁻¹`, and the pivots stay healthy, so the ILU(1)-style "fill produces negative pivots" account does
        **not** transfer. The open instrument is to compare the two V-cycles' action per level, or to degrade
        the coarse solve to `jacobi` in both arms and see whether the gap survives.
      - **Column shortening is a separate and much milder effect: it DEGRADES but does not break.** At the
        tight stop `3/3/3/3/2/2` converges in 22 cycles against uniform's 11 — 2× the cost, same 1e-11 floor
        — where preserving does not converge at all. A loose march-solver stop reads its 4.779e-03 as a
        failure, but that is the stop, not divergence. Under preservation the reach is genuinely irrelevant
        (58 cycles / 2.299e-02 at both reaches), which is the one claim of the pattern-preserving change that
        the measurement fully confirms.
      - **The reach-3-positions padding experiment is consistent with all of this, and the distinction is now
        load-bearing:** that experiment padded with **~1e-30**, which is *nonzero* and therefore survives a
        pruning assembly. An exact zero does not. Do not read "explicit zeros are required in the pattern" as
        licensing stored exact zeros.
    - **Probe REACH is a preconditioner choice — reach-2 is NOT a safe drop-in (measured, SHELVED — a
      GENUINE failure, not a build artifact).** The materialized `J` is only the preconditioner's operator
      (the solve matvec is always the exact matrix-free jvp), so a lower `stencil_reach` gives an approximate
      PC. On the orthogonal `bfs3d` mesh reach-2 is numerically near-*exact* at *every* state
      (‖A2−A3‖_F/‖A3‖ = 6e-6 cold, ~1e-15 developed — the dropped distance-3 shell is negligible; swapping
      Corrected→Compact Green-Gauss leaves it bit-identical, so the non-orthogonal skew correction
      contributes ~0 here) and ~2.2x cheaper to probe (60 vs 112 colours under the degree-ordered colouring this was measured with; 39 vs 94 under the saturation colouring now shipped, ~half the nnz). **Yet GAMG(reach-2)
      DIVERGES as a preconditioner** (true residual 1e3–1e8) at cold AND developed, on its own operator, with
      a verified-correct build — so it is genuine, not a build/scaling bug. **The cause is the ILU(1)
      smoother, whose fill is PATTERN-dependent (a symbolic/graph operation), not value-dependent:** halving
      the graph gives a structurally weaker incomplete factorization that is non-convergent on the indefinite
      saddle. The ≤6e-6 magnitude argument bounds only the value-based *aggregation* (consistent
      reach-2↔reach-3); it does not touch the smoother. Proof: padding reach-2's values onto the reach-3
      *positions* (distance-3 shell as ~**1e-30**, which is *nonzero*) recovers reach-3 convergence
      **bit-identically** (32 matvecs) — so the distance-3 *positions* are causal, their values are not.
      **This is why the reach-3 pattern is REQUIRED for smoother convergence (not merely a
      structure-invariance nicety), and why you must not lower the default reach.** ⚠️ **Read "positions
      carrying a tiny NONZERO", not "stored exact zeros" — the two are opposite in effect and the padding here
      was 1e-30.** A stored *exact* zero survives no pruning assembly and, when it does reach the
      factorization, is measured to break the zero-shift V-cycle outright (see the stored-zeros entry above).
      Recovery is not cheap:
      `smoother_sweeps=3` / ILU(2) still diverge *and* erase the setup win; restoring the reach-3 pattern
      converges but forfeits the GAMG-setup half of the economy (the larger half — refresh is
      setup-dominated), leaving only the marginal probe-colour saving. The one open lever is a fundamentally
      different smoother that tolerates the sparser graph (a smoother-design task). On a genuinely SKEWED mesh
      reach-2 would additionally be *lossy* (the non-orthogonal ring pushes real content to distance-3) — a
      second reason to keep reach-3.
      **One correction worth carrying, because it was believed for a while:** an apparent "47.2M → 39M
      nnz decay" in the pattern was a conflation of a *pattern* count with a *live* count — the live
      Jacobian holds ~38.7–39.0M nnz throughout, roughly constant. The reach-3 requirement above rests on
      the padding experiment, which is sound; it does not rest on any decay.
  - **`ilut_preconditioner.py` — `MonolithicIlutPreconditioner`.** Built off the jit path (`scipy.spilu`);
    a **host** object, so it is **not** an `equinox.Module` — it rides as a static field and is applied
    inside the jitted Krylov solve through `jax.pure_callback` (`.matvec()` / `.matvec(transpose=True)`).
    Frozen at a reference state+shift like the AMG blocks; being far stronger it tolerates the freezing at
    a few extra cycles, and the shift vanishes at the root so it never changes the converged state or its
    adjoint. `IlutFactors`/`factorize_ilut` are the pure host core (testable without JAX); the JAX
    wrapper is thin. `cell_major_permutation`/`equilibrate_cell_major` are **not** the ILUT's -- they
    live in `frozen_operator.py`; see the placement note below.
  - **Adjoint transpose wiring — `TransposedPreconditioner` (in `implicit.py`, binding).** The generic
    adjoint machinery `_adjoint_preconditioner` derives `Mᵀ` from the forward `M` with
    `jax.linear_transpose` — which works for a traceable AMG V-cycle but **cannot transpose a
    `jax.pure_callback`** (and `jax.custom_transpose` is absent in the pinned JAX). So a preconditioner
    that supplies its own transpose wraps its `state → Mᵀ` factory in `TransposedPreconditioner`, and
    `_adjoint_preconditioner` applies it directly rather than transposing. The ILUT's `Mᵀ` is the same
    factorization with `ilu.solve(trans='T')`. Pinned by the coupled-adjoint FD gate
    (`tests/integration/test_coupled_ilut.py`); the plain-callable path (every AMG preconditioner) is
    unchanged.
  - **Cheap in-place mid-march refresh — `refresh_in_place` (BUILT; forward-march only).** The frozen
    ILUT goes stale as the flow develops — the shifted solve slows, and on a low-shift dual-time path it
    can NaN. `MonolithicIlutPreconditioner.refresh_in_place(matvec, plan, shift_diagonal,
    …)` re-materializes and re-factors at the developed state and swaps `self.factors` **in place**. Two
    facts make this a **compilation cache hit** rather than a recompile: the preconditioner is a *static*
    field of `MonolithicFactorShiftPolicy` (so its identity is the jit treedef, unchanged by mutating its
    factors), and `matvec()` reads `self.factors` **at callback time** (not captured), so the mutation is
    seen by the already-compiled solve. `build` and `refresh_in_place` share one form-and-factor path
    (`_factor`). **This is sound only because the forward march is NEVER differentiated** — the mutation
    is impure and would corrupt the adjoint's transpose solve (which reads the same `self.factors`), so
    it is forward-march only; the converged root and its adjoint are refresh-independent anyway (the shift
    vanishes at the root). Measured on pitzDaily: the in-place refresh removes a large fixed overhead per
    refresh (a march-step recompile, a base-policy rebuild and a jvp recompile), leaving only the intrinsic
    materialize + factor — of which **`spilu` is the overwhelming majority** and the coloured-probe
    materialize a small remainder, so a sparser (cheaper-materialize) stencil would save almost nothing.
    (The wall-clock breakdown was recorded with no machine, thread count, state or β — the *ratio* is the
    load-bearing part and the seconds are deleted; re-measure if a cost model needs them.) `spilu` is a hard floor: a *threshold* ILU's fill pattern is
    value-dependent, so the symbolic factorization cannot be frozen and re-used (and scipy exposes no
    symbolic/numeric split), leaving **amortization (refresh less often) as the only cheap lever**. The
    coupled driver wiring is `coupled_ilut_refreshing_continuation` (a `refresh_builder` for
    `solve_coupled` — see `.claude/rules/turbulence.md`); it pairs with a `CoefficientDriftTrigger` so the
    re-factor *leads* the staleness. Pinned by `test_refresh_in_place_repreconditions_the_same_compiled_matvec`
    (unit) and `test_ilut_refreshing_continuation_refreshes_the_same_step_in_place` (integration).
  - **Scope / follow-ups (MVP).** The heavy fill is affordable at 2D /
    moderate mesh sizes but is the weak point at large 3D — the **monolithic AMG V-cycle**
    (`amg_preconditioner.py`, below) is the built scaling path (its direct-LU coarse solve is what tames the
    naive monolithic V-cycle's coarse-grid-correction instability on the indefinite saddle). The coupled builder still assembles the
    unused block AMG as the `a_P` source — a lightweight shift-diagonal-only policy would remove that. The
    coupled integration (`coupled_ilut_continuation`) lives in `.claude/rules/turbulence.md`.

## Preconditioner — monolithic complete-LU

- **Monolithic COMPLETE-LU preconditioner — BUILT (`lu_preconditioner.py`), the preferred 2D/moderate
  coupled preconditioner.** The sibling of the ILUT: it factors the assembled coupled Jacobian
  *completely* (`MonolithicLuPreconditioner`), so it is the operator's **exact** inverse and a Krylov
  solve converges in **one** iteration. Measured on the developed pitzDaily coupled Jacobian (61k dof):
  UMFPACK factors it in **~1.2 s vs the ILUT's ~32 s (~26×)**, exact (1 GMRES iter vs 2–4), verified on
  the real forward operator and the β=0 adjoint (true-residual checked). Because the fill is pattern-determined it is also **state-robust**
  (no `drop_tol` tail that shifts with the flow). Same interface as the ILUT — now shared rather than
  restated, via `HostPreconditioner` (`build` / `refresh_in_place`
  / `matvec`), a host object applied via `pure_callback`, riding as a static field; the adjoint reuses the
  factorization's transpose. **No equilibration / cell-major reordering** (unlike the ILUT — the complete
  factorization's own pivoting + fill-reducing ordering handle the indefinite saddle on the raw
  field-major matrix; equilibrating + cell-major actually *hurt* it, measured).
  - **Pluggable backend (`factorize_lu(backend=…)`):** `"umfpack"` (SuiteSparse via the optional
    `petsc4py` dep) is the fast path — a fill-reducing (nested-dissection/AMD) ordering + a multifrontal
    BLAS-3 numeric kernel. A refresh **re-factors from scratch** (NOT a fixed-pattern numeric-only
    refactor): the coupled Jacobian's sparsity *grows* as the flow develops — cross-coupling entries that
    are exactly zero at the cold reference become nonzero — so a frozen-pattern refactor is both wrong and
    a shape error; the full factor is fast enough (~1 s at 2D/moderate) that re-analysing each refresh is
    cheap. `"scipy"` (`scipy.sparse.linalg.splu`, SuperLU) is the always-available fallback: exact and
    correct (what the tests run under) but, lacking nested dissection, no faster to factor than the ILUT.
    `"auto"` (default) picks UMFPACK when importable, else SciPy. So the module imports with no optional
    dependency; the 26× is opt-in via `pip install aquaflux[petsc]`.
  - **SCOPE — a 2D / moderate-mesh tool (binding).** A complete LU's fill is `O(n log n)` in 2D but
    `O(n^{4/3})` in 3D, so **memory is the wall in 3D** — measured (synthetic block grids): 2D factor time
    ~`dof^1.37` (comfortable to ~10⁵ cells, seconds, <10 GB), but 3D hit **out-of-memory at ~10⁴ cells**.
    So this preconditioner is the fast, exact choice for 2D / moderate meshes; large 3D still needs the
    ILUT / block / (parked) multigrid-smoothed paths, or a **rank-structured direct solver** (MUMPS-BLR /
    STRUMPACK — the fill-taming way to keep this exact-factor paradigm in 3D, reachable via the same PETSc
    dep). It does **not** dominate the ILUT everywhere; it is an additional strategy, best at 2D/moderate.
  - **Why the ILUT's "spilu is a hard floor" is not the whole story (measured, corrected):** the ILUT
    floor is against reducing *fill* (a sparser threshold stencil saves little) — a different lever than
    switching to a *complete* factorization with a fill-reducing ordering + fast kernel, which is what
    breaks the floor here. A separately-tried level-based ILU(k) via PETSc looked faster but was a
    **preconditioned-norm artifact** (PETSc's KSP converges on ‖Mr‖, not the true ‖Ax−b‖); it is weaker,
    not stronger — always verify the TRUE residual.
  - **FROZEN is wrong for the β-ramping dual-time march — track β (binding, measured).** A complete LU is
    *exact* only for the operator it factored, `J + β d`. In a dual-time march β ramps (0.5 → 0.005), so a
    factorization frozen at one β **mis-preconditions** the operator actually solved — measured on rung2:
    a LU frozen at β = 0.05 needs 25 / 111 / 217 / **474** GMRES iters at β = 0.1 / 0.5 / 1 / 2 (vs **1**
    when factored at the matching β), and on a real cold ramp the frozen LU **NaN'd** on the overshot
    low-β state (215 cycles → failure). This is the *opposite* of the ILUT, whose *approximate*
    factorization tolerates the β-mismatch at a few extra cycles — so the ILUT's frozen + drift-refresh
    design does **not** carry over to the exact LU. The fix, because the LU factor is cheap (~1 s), is to
    **re-factor at the current `(state, β)` every step** (`forward_march`'s `precondition_step` seam +
    `lu_beta_tracking_refresh`, `.claude/rules/turbulence.md`): exact each step (1 Krylov iter), and robust
    through overshoots (measured: completes the cold ramp where the frozen LU failed, cyc ≤ 18). The
    finishing solve and adjoint keep the last frozen factorization (exact enough at the converged β → 0).
  - **Coupled builders (`coupled_lu_continuation` / `coupled_lu_refreshing_continuation`, and the
    β-tracking `lu_beta_tracking_refresh`) live in `.claude/rules/turbulence.md`;** they share the
    `MonolithicFactorShiftPolicy` and the `_monolithic_factor_step` builder tail with the ILUT (one
    implementation, parameterized by the factorization).

## Preconditioner — monolithic AMG (the coupled PC)

- **Monolithic ALGEBRAIC-MULTIGRID preconditioner — BUILT (`amg_preconditioner.py`), the coupled PC for
  large 3D.** The third member of the family: instead of factoring the assembled coupled Jacobian it applies
  **one smoothed-aggregation multigrid V-cycle** (`MonolithicAmgPreconditioner`, PETSc `PCGAMG`) whose only
  exact solve is a **direct LU on the small coarsest grid**, so the heavy fill never lives on the fine grid.
  This is the answer to both factorizations' 3D wall: the complete LU's fill OOMs (`O(n^{4/3})`), and the
  threshold-ILU's `spilu` is *prohibitively slow to build* on a distance-3 3D coupled Jacobian — measured on
  the 23k-cell `bfs3d` case the assembled Jacobian is **38.7M nnz (~280/row)** and a single `spilu` at
  `fill_factor=30` ran **>7.5 min and never finished** (RSS <2 GB — the 3D wall here is *time*, not memory,
  and the earlier "~11 min XLA compile" reading was a CPU-contention artifact; on idle CPU the probe compile
  is fast and `spilu` dominates). The V-cycle instead **builds in ~seconds with bounded memory** and scales.
  - **The V-cycle is a fixed LINEAR operator (one `pc.apply`, not an inner Krylov solve), so it is a
    drop-in for the same callback-matvec interface as the ILUT/LU and — being linear and transposable —
    serves the adjoint's transpose solve through the multigrid's own transpose (`pc.applyTranspose`), with
    no flexible outer Krylov.** It preconditions the **equilibrated, cell-major** matrix via the shared
    `equilibrate_cell_major` (**in `frozen_operator.py`** -- the one home for the sqrt-diagonal
    equilibration + cell-major reorder, moved there 2026-08-15; see the placement note below); the
    `Mat` block size is `n_fields` so GAMG aggregates cell-blocks. Host object, applied via `pure_callback`,
    riding as a static field; `build`/`refresh_in_place` its own and `matvec` inherited from the shared
    `HostPreconditioner`, so it plugs into
    `MonolithicFactorShiftPolicy` unchanged. **It is the one family member that needs `petsc4py`** (no
    pure-SciPy AMG fallback); the module lazily imports PETSc and raises a clear install hint otherwise.
  - **⚠️ SUPERSEDED BELOW — "ILU(0) stalls" is a HIGH-β measurement and does NOT hold at low β.** The
    original comparison was made near the OpenFOAM-converged state at a *large* shift, where the operator
    is diagonally dominant and extra fill is harmless. At the low shifts the march's tail actually runs
    at, the ranking **inverts**: ILU(1) breaks down and ILU(0) is the one that converges. See
    *"Zero-fill is the low-β smoother"* below, which is the current default-setting evidence. Read the
    two together: neither is wrong, they are measurements at opposite ends of the β range, and the march
    lives at the low end.
  - **The smoother is the research variable, and the first measured MVP config was a STATIONARY ILU(1) level
    smoother (`richardson`) + direct-LU coarse.** Measured on the `bfs3d` shifted coupled Jacobian
    (true 2-norm residual, `KSP_NORM_UNPRECONDITIONED`, at 2 sweeps): plain GMRES + stationary **ILU(1)**
    reached 1e-8, **ILU(0) stalled** and **SOR diverged** (measured, configuration not recorded — no shift,
    aggregation, coarse-eq limit or state, and both the smoother-fill and aggregation defaults have since
    moved; re-measure, and read it with the superseding low-β result below). A Krylov-accelerated (GMRES)
    smoother is a few iterations *faster* but makes the V-cycle **nonlinear** — it
    needs flexible GMRES and has no clean transpose, so it is a deferred forward-only optimization, **not** the
    adjoint path. ⚠️ **Name the forward solver's PATH — there are three and they differ.**
    `coupled_amg_continuation` builds its own inline: `forward_rtol = 0.3` in the **row-scaled**
    `coupled_scaled_norm`, `restart=15`, `max_restarts=60`. (`_COUPLED_FORWARD_SOLVER`, block-SIMPLE 2D:
    `relative_residual_gmres(1e-2)`, 2-norm, restart 120. `_COUPLED_ILUT_FORWARD_SOLVER`, 2D ILUT: 1e-2
    2-norm, restart 10.) There is no `_COUPLED_AMG_FORWARD_SOLVER` symbol.
  - **Per-step cost tuning (measured): `smoother_sweeps=2` default and the forward restart 15 (from 40).**
    The restart-15 forward loop stops as soon as the ~1% inexact-Newton tolerance is met instead of running
    out a 40-vector subspace (the dominant per-step saving). The **smoother-sweeps knob is the second lever,
    and more is better on this saddle**: the outer Krylov cost is governed by the *smoother work* per V-cycle,
    and adding a second incomplete-LU Richardson sweep — one extra cheap triangular back-solve — roughly
    quarters the outer iteration count on the low-shift operator the march's tail runs at, and is worth much
    less at a high shift where the operator is already diagonally dominant (measured on the `bfs3d` coupled
    Jacobian to a 1% stop — configuration not recorded: no β value, aggregation or coarse-eq limit, and it
    was tuned against ILU(1), which is no longer the validated smoother). Each outer iteration pays a full
    Jacobian-vector product (and, on the JAX-side `lineax` path, a `pure_callback` into PETSc), so trading one
    cheap extra sweep for far fewer outer iterations is a large net win — `sweeps=2` is the sweet spot
    (`sweeps=3` helps a little more at low shift but costs at high shift). Adding *fill* to the smoother
    (`smoother_fill_levels`) instead would cut iterations too, but it is the expensive incomplete-factorization
    build the ILUT hits in three dimensions; sweeps add smoother work without that build cost, and the
    coarsening choice (selective vs smoothed-aggregation) is a minor knob by comparison — but do not read that
    as covering `pc_gamg_agg_nsmooths`: plain-vs-smoothed *prolongator* smoothing is measured below as the
    largest preconditioner win found on this case. (The whole-march wall figure that used to sit here
    predated several march-wide wins and is deleted.) An **experimental, opt-in native-PETSc forward path**
    (`coupled_amg_continuation(native_forward_solve=True)`) is a far larger per-step lever — a native KSP
    whose shell matvec calls the eager JAX jvp (true Newton), 1 native GMRES iteration vs the JAX-side
    lineax path's ~90 on the identical system — but it currently under-converges the *march* (the lineax
    path over-solves each step to ~machine zero and the pseudo-transient globalization leans on those
    near-exact steps; the native honest-tolerance step descends slower and overruns the step budget), so it
    stays off by default. The follow-up to make it the default is a **β-tracking GAMG refresh** (the AMG
    analogue of `lu_beta_tracking_refresh`), so the frozen V-cycle matches the ramping β and the native
    solve stays accurate at low β.
  - **`coarse_eq_limit` — grow the coarsest-grid direct LU (BUILT).** GAMG's default coarsens to a tiny
    (~50-equation) coarse grid, whose direct LU captures only the crudest global mode; the indefinite
    saddle's wall is exactly that global pressure coupling. `build_amg_vcycle(coarse_eq_limit=K)` /
    `coupled_amg_continuation(coarse_eq_limit=K)` (default `None` = PETSc's ~50, byte-identical) stops
    coarsening at `K` equations so the coarse LU inverts more of the global coupling **exactly** — a
    stronger V-cycle *and* transpose V-cycle (so the adjoint benefits). Measured on the `bfs3d` hard state
    (honest right-PC cycles to 1e-6): **baseline ~50 → 652 cycles at β=0.5, `K=2000` → 24 (~27×)**, and it
    **saturates by ~2000** (`K=8000` identical), at negligible extra build cost (the 2000-eq coarse LU is
    trivial against the materialize). The `bfs3d` combo-sweep. *(A `cycle_type` V/W knob was tried and
    **dropped**: the W-cycle came back byte-identical to V — GAMG did not honour `pc_mg_cycle_type` here —
    and `coarse_eq_limit` dominates regardless.)*
  - **Zero-fill is the low-β smoother: ILU(0), 4 sweeps, `coarse_eq_limit=2000`, PC-only `beta_floor`
    (the validated bundle).** The shifted operator is `J + β d`; as the march's shift falls the diagonal
    weakens and the factorization, not the coarse space, is what fails first. Measured on the `bfs3d`
    coupled Jacobian at **adjoint-grade rtol 1e-8 on the TRUE residual**:

    | β | ILU(1) (was default) | ILU(0) |
    |---|---|---|
    | 0.10 | converges | 30 its |
    | 0.05 | converges | 51 its |
    | 0.02 | **DIVERGES** (534 its, true rel. 3.7) | 97 its |

    **Ground truth, not inference:** a pivot census of both factorizations at β=0.02 found **303 negative
    pivots in ILU(1)** (min 6.2e-4) and **zero** in ILU(0) (min pivot 0.29–0.36 at every β tested). The
    *fill* is what destroys the factorization — dropping it is not an approximation here, it is the fix.
    The worst pivots sit in **velocity rows, not pressure rows**, which is why "the pressure block is the
    problem" intuitions kept failing. ILU(0) is also **3–4× cheaper to build**. This is consistent with the
    literature: every saddle-point-AMG ILU smoother in the published work we surveyed is **zero-fill**
    (ILUC0 / DILU); the ILU(1) default was ours alone.

    The bundle members were each measured alone and then together on the same state:
    - `smoother_fill_levels=0` — the table above.
    - `smoother_sweeps=4` — ILU(0) is a *weaker* smoother than ILU(1), so extra sweeps pay more than they
      did for ILU(1): 390 → 69 its at β=0.01. (This is why the `sweeps=2` sweet spot recorded above does
      not carry over: it was tuned against ILU(1).)
    - `coarse_eq_limit=2000` — `None` **stalls at every low β** (552 its, true rel. 1.3–67). Not optional.
    - **PC-only `beta_floor=0.05`** — the V-cycle is built at `max(β, 0.05)` while the march still solves
      at its own β: 43/69/95 → 29/45/47 its at β = 0.02/0.01/0.005. The *operator* is untouched, so the
      converged root and the adjoint are unchanged and the mismatch saturates at `beta_floor·d` rather than
      growing as β → 0. Flooring the k/ω rows *alone* hurt; flooring all rows was best.

    **`alpha_u` and a velocity-row `beta_floor` are the same knob.** A velocity under-relaxation `α_u=0.95`
    adds `(1−α_u)/α_u = 0.0526` of the diagonal — a β-equivalent of ~0.05 applied to the velocity rows only.
    Worth knowing before adding a second spelling of it: prefer the floor, which is explicit about being
    preconditioner-only.

    **PLAIN aggregation, not smoothed — `pc_gamg_agg_nsmooths = 0` (measured, and the largest
    preconditioner win found on this case).** Smoothing the tentative prolongator with a Jacobi step is
    GAMG's default and is right for an M-matrix-like operator; on a strongly indefinite saddle it
    degrades the coarse correction. Measured with the march's **own** states, right-hand sides
    (`-R(state)`) and shift pairing (operator at the march's β, V-cycle at `max(β, floor)`):

    | state | smoothed (was) | plain (now) |
    |---|---|---|
    | entering a step below the shift floor (β 0.0154, α clipped to 0.200) | 22 cyc, 3.2e-8 | **9 cyc, 4.7e-10** |
    | entering the retried step (α → 0) | 4 cyc, 1.2e-12 | **3 cyc, 5.7e-14** |
    | the converged tail | 6 cyc | 6 cyc |

    **The tie at the converged tail is the methodological point:** an easy operator does not discriminate
    between preconditioners, so the first sweep — taken there — reported no difference on any arm and
    nearly closed the question. Sweep preconditioners at the march's *hard* states, which the checkpoints
    identify by cycle count and clipped α. Plain aggregation is also marginally cheaper to set up.
    **It is also what makes a DEEP hierarchy usable:** with smoothing ON, adding levels via
    `pc_gamg_threshold` gives dense coarse operators, builds of 96–1224 s, and a V-cycle returning
    **NaN** at every threshold tried (0.01/0.05/0.1/0.25); with it OFF, `threshold=0.05` builds a
    **5-level** hierarchy in 26 s at the same cycle count. That matters for scaling — the shipped
    2-level design leans on a direct LU at `coarse_eq_limit` equations to carry the global modes, which
    is cheap at 23k cells and will not be at 10× — so the threshold is left OFF as the default (it buys
    nothing on *this* mesh) while being known to work.
    **Corollary worth keeping: the NaN is the smoothing/threshold INTERACTION, not the threshold.** Do
    not re-refute strength-of-connection on the smoothed-aggregation evidence.
    **VALIDATED ON THE FULL 3-RUNG MARCH** — 4050 s / 62 steps / **347 cycles** / 3 retry cascades →
    3474 s / 69 steps / **290 cycles** / **1** cascade, converging deeper (2.6e-6 vs 8.9e-6) with the
    answer unchanged (mid-span `x_r/h` 8.36). Per rung the effect lands exactly where the sweep said:
    rung 1 (the easy high-β anchor) **37 → 37, identical**, rung 2 110 → 75, rung 3 200 → 178. Note the
    march takes *more* steps (62 → 69): cheaper solves hold α higher, so the Courant control grows β
    differently and the trajectory diverges from step 24 — this is a whole-march total, not a per-step
    improvement, and part of the wall saving is the two retry cascades that stop happening. That same march
    ran to the target Reynolds number with no breakdown and β reaching **0.0077** at 6–11 cycles per solve,
    well past the 0.02 where ILU(1) diverged — the qualitative claim that matters.

    **⚠️ CONFLICTING CYCLE TOTALS for this one 62-step march, unresolved.** It is recorded here as **347
    cycles** and elsewhere as **883 "raw" cycles**, with "raw" nowhere defined. Neither is recoverable from
    source. Treat the *ratios* as the finding and the absolute total as unestablished — re-measure with the
    counter's definition stated in the same breath if a cycle total ever becomes decisive.
  - **β-diagonal split — track β without re-materializing the Jacobian (BUILT).** The operator is
    `J(φ) + β d`, and the shift `β d` touches only the **diagonal**, so a β-tracking refresh does **not**
    need the coloured-probe materialization of `J` (the dominant refresh cost — hundreds of jvps).
    `MonolithicAmgPreconditioner.refresh_shift_in_place(shift)` reuses the **cached** Jacobian (stored at
    the last `build` / `refresh_in_place`), re-adds the new `β d` diagonal (`O(nnz)` numpy) and re-factors.
    Measured on the `bfs3d` hard state: **full `refresh_in_place` 36 s vs shift-only 18 s (2×)** — the 18 s
    saved is the materialize, the remaining 18 s the equilibrate + GAMG refactor. The frozen `J` does not
    track *state* drift, so the full materialize is **gated** (`_materialize_gate`, mirroring
    `_staleness_beta_gate`): `amg_beta_tracking_refresh(materialize_drift=τ, materialize_every=K)` does the
    cheap shift-only refresh in between and a full materialize when the ν_t drift since the last one exceeds
    `τ` OR after `K` steps (both `None` = full every refresh, unchanged). Prefer `materialize_drift` (the
    honest state-staleness signal via `eddy_viscosity_drift`) with a large `K` as the safety cap — the fixed
    step count was the "fixed cadence" antipattern. **Measured caveat: a *fresh* Jacobian is worth its cost
    on a fast-developing flow** — driving the materialize *more* often (via a tight `τ`) cut the march's
    Krylov cycles ~23 % (fewer/cheaper steps) despite more refresh, so under-materializing was costing more
    in solve than the materialize saves; the lever is a *cheaper* materialize (batched probe + gather
    de-compression above), not a rarer one. Forward-march only. Pinned by `test_amg_refresh_shift_in_place_*`
    (`test_amg_preconditioner.py`) and `test_materialize_gate_*` / `test_batched_probing_*` /
    `test_gather_de_compression_*`.
    - **⚠️ THE TWO GATES ARE COMBINED, NOT NESTED — the β floor used to make the drift trigger UNREACHABLE
      (fixed; `_refresh_branch`).** `_beta_tracking_refresh` asked the β gate first and the materialize gate
      only *inside* it. With a PC-only `beta_floor` the gate's input is `max(β, floor)`, so once the march
      drops below the floor that input is **pinned** and the β gate answers "no change" on every step
      forever — taking the drift gate down with it. That is exactly the low-shift tail where the flow
      develops fastest. Measured on the 3-rung `bfs3d` cold march (56 steps, `beta_floor = 0.05`):

      | | steps | refresh declined | mean cycles |
      |---|---|---|---|
      | β ≥ floor | 34 | 12 % | 6.9 |
      | β < floor | 22 | **91 %** | 7.5 |

      13 steps had >5 % ν_t drift *and* no refresh, and they carried **189 of the march's 399 Krylov
      cycles (47 %)** — steps 29–31 ran 24/23/34 cycles on a V-cycle nothing was allowed to refresh, and
      the step after them blew up into a 3-attempt β-escalation retry costing 380 s. The decision is now
      the total function `_refresh_branch(stale_state, moved_beta, split)`: **state drift ⇒ `full`**
      whatever β says, β move alone ⇒ `shift`, neither ⇒ `none`. Note the shift branch is real work below
      the floor too — the shift is `pc_beta · d(state)` and the per-cell `d` tracks the state even where
      `pc_beta` is clamped, so the old comment's "would rebuild an identical V-cycle" was only true of the
      β factor. **Consequence to know:** the materialize gate is now consulted every step rather than only
      on refresh steps, so its `materialize_every` cap counts **steps**, not refreshes.
      **MEASURED END-TO-END on the 3-rung `bfs3d` cold march, and the trade is strongly favourable:**

      | | before | after |
      |---|---|---|
      | outer steps | 69 | **61** |
      | **Krylov cycles** | **480** | **348 (−27 %)** |
      | starved steps (>5 % drift, no refresh) | 15, holding 201 cycles (42 %) | **0** |
      | sub-floor steps declining a refresh | 89 % | **19 %** |
      | full refreshes | 32 @ 23.1 s | 48 @ **17.3 s** |
      | refresh total | 803 s (15.7 % of wall) | 865 s (19.1 %) |
      | β-escalation retry steps | 5, **1745 s** | 2, **655 s** |
      | final ‖R‖ / mid-span `x_r/h` | 2.65e-6 / 8.36 | 2.42e-6 / **8.36** |

      Read the refresh row correctly: the preconditioner now costs **more** in absolute terms (865 s vs
      803 s) because it refreshes 50 % more often — that is the trade working, not a regression. It buys
      132 fewer Krylov cycles and, far larger, removes three of the five retry cascades (−1090 s), which
      were the single biggest line item in the march. The converged answer is **unchanged** (`x_r/h` 8.36
      mid-span against OpenFOAM's 7.24, exactly as before), which is the constraint that matters: this is
      a path change, not a solution change.
    - **Where a refresh's time now goes (whole-run aggregate, same march): probe 632 s, refactor 195 s,
      assemble 28 s, other 12 s — the probe is 73 %.** The non-probe tail is close to floor, so any
      further work on refresh cost has to attack the coloured probe itself (amortizing colours across
      steps, or a cheaper stencil), not the assembly or the multigrid setup.
    - **⚠️ Wall-clock from that comparison is a LOWER bound, and the reason is a trap worth naming.** The
      "after" run was watched by a per-refresh log monitor whose notifications drove the desktop UI to
      ~70 % CPU against the solve; its first ~20 minutes are inflated (steps 1–3 were *bit-identical* in
      cycles and residual to the baseline while taking 254 s against 224 s). Cycles, phase times and
      step/retry counts are unaffected — which is exactly why the cycle count, not the clock, is this
      project's cost measure. **Do not instrument a long run with a per-step notification stream.**
    - **Where a refresh's time actually goes — REPORTED, not inferred (`RefreshTiming`, `solve/refresh_timing.py`).**
      The observer used to receive `(kind, seconds)`, so a breakdown had to be recovered by differencing
      the `full` and `shift` branches across a run. It now receives a record carrying ordered
      `(phase, seconds)` pairs — AMG: `probe` / `assemble` / `refactor` — and `MarchLogger` renders them
      under the `"pc"` detail (`pc full 23.0s (probe 14.6 assemble 3.2 refactor 5.2)`), with any
      unattributed remainder shown as `other` so a breakdown cannot silently fail to add up. The
      factorization preconditioners report no phases (empty tuple), which the record documents as valid.
      Differencing the same 56-step march gave probe ≈ 14.6 s of a 23.0 s `full` (63 %) against 8.4 s for
      `shift` — consistent with the older ~40-of-60 s reading, and the reason the probe is the lever.
    - **The per-refresh sparse-matrix work is precomputed (`ShiftedCellMajorOperator`).** The `assemble`
      phase — add `β d` to the diagonal, symmetrically equilibrate, reorder to cell-major — was a sparse
      add plus two sparse products plus two fancy-index permutations, each allocating and re-sorting a
      matrix the size of the coupled Jacobian, **repeated identically every refresh** because for a fixed
      stencil reach the pattern never changes and only the values do. The class hoists the
      pattern-dependent part (which base nonzero feeds each output nonzero, where the diagonals sit, the
      cell-major CSR structure) into a one-time build, leaving one gather + an `O(n_dofs)` diagonal add +
      a symmetric scale into a **preallocated** buffer. Bit-identical to `equilibrate_cell_major(J +
      diags(shift))` — pinned by `tests/unit/test_cell_major_operator.py`, which runs without `petsc4py`
      because the class holds none. Two details worth keeping: the returned matrix **aliases the reused
      buffer** (consume it immediately; `AmgVCycle._build`/`refactor` both copy), and the symmetric scale
      is applied **chunked over rows** so it never allocates a second Jacobian-sized temporary — the
      point of the exercise being to stop allocating those. It engages only when a precomputed
      `structure` was used, which is precisely the guarantee the pattern is fixed; otherwise the generic
      path runs unchanged. `AmgVCycle.refactor`'s pattern re-check is likewise memoized on the index
      arrays' identity, since comparing tens of millions of indices twice per refresh re-confirms
      something fixed by construction.
    - **Two redundant full copies of the operator's VALUES are gone (2026-08-14).** `_build` wrote
      `cell_major.data.astype(ScalarType).copy()` — `astype` already returns a fresh array, which is the
      ownership the persistent `Mat` needs, so the `.copy()` allocated the values a second time — and
      `refactor` wrote `self._data[:] = cell_major.data.astype(ScalarType)`, whose `astype` builds a full
      temporary that the assignment then copies out of, on **every** refresh. A slice assignment casts to
      the destination's type as it copies, so it needs no `astype` at all. Each is one array as long as
      the Jacobian's nonzeros. Behaviour-identical; not measured.
    - **Already done, do not re-propose:** `refactor` reuses the aggregation and prolongation
      (`pc_gamg_reuse_interpolation` + smoother `reuse_ordering`) and overwrites the operator values in
      place, so only the Galerkin coarse operators and the incomplete-LU factor values recompute.
    - **A Vanka patch smoother is BUILT and installable as a level smoother — `aquaflux/solve/vanka.py`.**
      Not shipped as a default and **not exported from `aquaflux.solve`**: it exists so the recorded
      "the coarse space is the wall" verdict can be re-adjudicated (see the low-β bullet below), and it
      earns a place in the tree only if it wins. Three facts to keep:
      - **Route: PETSc's *shell* preconditioner, not `PCPATCH`.** `PCPATCH` with
        `pc_patch_construct_type = vanka` needs a `DM` the plain-AIJ path does not supply, which is why
        the earlier plan flagged it as a risk. `pc_type python` + `pc_python_type
        aquaflux.solve.vanka.VankaPC` needs **nothing but the assembled matrix** — verified on a plain
        `createAIJWithArrays` matrix with no `DM`, and it reaches every level (block size propagates to
        the Galerkin operators, so the coarse levels are configurable too). Reached through the
        `extra_options` seam, so no change to `AmgVCycle` was needed. Options carry the level prefix:
        `mg_levels_pc_type python`, `mg_levels_vanka_neighbours`, `vanka_neighbour_fields`
        (`before_centre`|`all`), `vanka_centre_field`, `vanka_damping`. On the shipped 2-level hierarchy
        `mg_levels_` *is* the fine level (level 0 is `mg_coarse_`), so no level-index bookkeeping.
      - **The patch is chosen ALGEBRAICALLY, by coupling strength, and it has to be.** The classical
        collocated Vanka patch is "the cell plus the cells its continuity row couples to" — but this
        Jacobian's stencil is distance-3, ~47 cells per row, so that patch would be a 280-unknown dense
        solve. `CellStarPatches` instead ranks the pressure row's cell-collapsed `Σ|a_ij|` and keeps the
        `n_neighbours` strongest. That also keeps the module mesh-free (`numpy`/`scipy` only, no `jax`),
        so it unit-tests on a three-cell matrix. **The strength is summed only over the fields the patch
        will actually contain** — with `before_centre` (the classical choice: the neighbours contribute
        velocities, each neighbour's pressure staying in its own patch) that is the divergence entries
        alone. Ranking on the whole row instead lets the collocated pressure-pressure damping term pick
        the patch's cells, which are not the cells the patch exists to invert.
      - **Additive with overlap averaging, deliberately — the recombination is the part that failed
        before.** The recorded "additive Vanka + Richardson diverges (ρ 9e4)" arm was an *unweighted*
        additive sum, which over-corrects every unknown that several patches share; the undamped
        block-ILU/inexact-Uzawa arm ("1-apply reduction 5.10") amplified for the same reason. The weight
        is `W^½ (Σ RᵀA_pp⁻¹R) W^½` with `W` the reciprocal coverage — split symmetrically across
        restriction and prolongation, which is the normalization Schöberl–Zulehner (2003) use and what
        Metsch's algebraic Vanka (dissertation §4.6) implements. It is a partition of unity, so on
        non-overlapping patches the smoother is exactly the block inverse (pinned by a unit test), and
        one application is a **bounded, fixed linear** operator — which is what the non-flexible outer
        GMRES and the adjoint's transpose both require, and what an inner-Krylov ("Krylov-Vanka")
        smoother is not.
        **Two deviations to state with any result, both making this weaker than the textbook smoother:**
        the classical Vanka sweep is *multiplicative* (patches solved in sequence against a residual
        updated as it goes) and this is additive; and the patch is truncated by strength rather than
        taking the whole continuity row. A null result from it is therefore *not* on its own evidence
        that patch relaxation cannot help — which is why the harness also runs a sweeps ladder on the
        Vanka arm, not only on the incomplete-LU one.
        `VankaSmoother.worst_patch_gain` (max `|A_p⁻¹|`) is printed at setup, because a near-singular
        patch produces an enormous additive correction and looks from the outside exactly like "patch
        relaxation does not work here". Cost at the `bfs3d` shape (23k cells, 6 fields, 280 nnz/row),
        extrapolated from a 5k-cell synthetic of the same density: patch selection ~0.1 s, factorization
        0.5–3 s, one apply 7–110 ms, and the stored inverses 0.1–1.1 GB depending on patch width — which
        is why the harness's widest arm is 12 velocity neighbours (~325 MB) and not 12 full cells.
    - **`AmgVCycle.destroy()` / `.levels`.** The V-cycle owns several PETSc objects and a factored
      hierarchy; `destroy()` releases them on the caller's schedule rather than the collector's, which a
      loop that builds one preconditioner per arm needs (two live copies of a 3D coupled operator plus
      factors is enough to exhaust a workstation — the standing "one heavy probe at a time" rule).
      `refactor`'s rebuild branch calls it too, so the teardown has one home.

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
       REJECTED ones.** A solve that blows past `retry.on_cycles` gets the step redone at an escalated
       β, and the retry then succeeds easily — so the record shows the *easy* attempt. Same march: step
       50's hardest solve is **15 cycles at β = 0.0293** with α collapsing to 0, in attempt 1; the step
       reports **3 cycles at β = 0.0585**. That is also why the escalated attempts are where the
       *sub-floor* operators are — the escalation is what lifts β back above the floor. Until the
       rejected attempts are recorded, read them out of `march.log` (`redo step N (attempt 2): …` plus
       the per-inner table above it) and name the state and β explicitly.
  5. **A probe driven by the WRONG "march" solver — there are two, and they look interchangeable.**
     `_COUPLED_ILUT_FORWARD_SOLVER` is 1 % in a plain 2-norm at restart 10; the coupled **AMG** builder's
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

## The field split — a saddle plus two transported scalars

- **⚠️ WE ARE NOT SOLVING A SADDLE-POINT PROBLEM — we are solving a saddle point PLUS two
  advection-dominated transported scalars, and that is probably why the saddle-point literature keeps
  not transferring.** Worth stating plainly because a long run of failures is explained by it:
  - **Every published method tried here targets a 2-field `(u,p)` system** — Vanka, Webster's
    stabilization, Metsch's algebraic Vanka, the SIMPLE pre-transform, monolithic saddle-AMG. Our block
    is **six** fields, two of them transported scalars, one solved in a log variable.
  - **The closest published work to this discretization segregates the turbulence.** Uroić–Jasak match
    us on every axis that usually matters (collocated Rhie–Chow finite volume, k–ω SST, backward-facing
    step, monolithic coupled AMG) and still put only `(u,p)` in the coupled block, solving k/ω
    separately with BiCGStab + ILU(0). Their papers contain **zero coverage** of turbulence-in-the-block.
    There is no published precedent for the 6-field monolithic block, so importing a fix from that
    literature is importing it across a regime boundary.
  - **The measured failure is not a saddle pathology.** The near-null direction of the degenerate cell
    blocks is **pure ω** — nothing on `u,v,w,p,k` — and per-field V-cycle smoothing has pressure *well*
    handled with ω the outlier. The saddle part of our operator is not what is hurting.
  **The direction this points at is a FIELD SPLIT that keeps the coupling** — a block-triangular
  preconditioner with the `(u,p)` saddle handled as now and k/ω preconditioned by something suited to
  transport (`scalar_transport_preconditioner` already exists and already serves those blocks in the
  block-preconditioner family). Note this is **not** the refuted arm below: that one was
  block-*diagonal*, i.e. the coupling **dropped**, and it established that the coupling is load-bearing
  — not that ω has to live in the same multigrid hierarchy. Architecturally the split is free: the
  operator stays monolithic, so the AD Jacobian and the coupled adjoint are untouched, and a
  block-triangular preconditioner is a fixed linear operator and therefore transposable.
  **Before building it, get an operator that can show a difference.** At the states currently reachable
  the monolithic ILU(0) V-cycle converges in **2 cycles to 1.3e-14**, so every candidate ties; and the
  march's cost is no longer preconditioner-bound (~65 % Krylov, largely fixed per-step matvec rather
  than cycles; 21 % refresh; the rest globalization). The upside on *this* case is bounded, and the
  test needs the hard inner iterates.
  - **✅ THE SPLIT IS BUILT — `solve/field_split.py` (`FieldGroups`,
    `BlockTriangularFieldSplit`, `build_block_triangular_field_split`).** Three facts worth keeping,
    independent of whether it ever wins:
    - **The partition is free, because the coupled state is FIELD-major.** Degree of freedom
      `(cell i, field f)` sits at `f·n_cells + i`, so a split on a *field* boundary is a split into two
      **contiguous ranges**: `[u,v,w,p]` is `[0, (dim+1)·n)` and `[k,ω]` the rest. Vectors are sliced,
      not gathered, and the four blocks are contiguous submatrices. `FieldGroups` owns that arithmetic
      so no consumer re-derives `f·n_cells + i` inline; `tests/integration/test_coupled_field_split.py`
      pins the partition against `CoupledRANSLayout.unpack`, which is the one thing that would be
      silently wrong rather than loudly wrong — a partition off by one field still preconditions, it
      just preconditions a mislabelled operator.
    - **It needs no JAX wrapper of its own.** `MonolithicAmgPreconditioner.matvec()` reads only
      `factors.n_dofs` and `factors.apply(r, transpose=…)`, both of which the split has, so it rides the
      existing `pure_callback` path unchanged. Each diagonal block is an ordinary `AmgVCycle`
      (`build_amg_vcycle` on the sub-block), which equilibrates and reorders *within its own group* and
      aggregates at its own block size — the whole point, since a four-field saddle and a two-field
      transport pair coarsen differently. `AmgVCycle.apply` returns the inverse in the **original**
      (unequilibrated, field-major) space, so the retained coupling block is applied raw between the two
      block solves, with no scaling bookkeeping.
      **⚠️ But `factors.n_dofs` + `factors.apply` is the WHOLE of what the split satisfies, and the base
      asks for more elsewhere — `has_native_solve` reads `self.factors.has_native_solve`, which only an
      `AmgVCycle` has, so on the split the inherited property RAISED (fixed 2026-08-14 by an explicit
      `has_native_solve = False` override; a split never forms the whole shifted operator, so there is no
      native solve to offer).** It went unseen for the reason worth carrying: **both call sites ask
      through `getattr(pc, "is_exact_native", False)` — the right spelling for the ILUT and LU, which
      genuinely lack the attribute — and a `getattr` default swallows an `AttributeError` raised *inside*
      a property body exactly as it swallows a missing name.** The value it produced was accidentally the
      correct `False`, so nothing failed. Two consequences: a test of such a property must read it
      **directly**, never through `getattr` with a default (a `getattr` test passes against the defect —
      `test_the_field_split_answers_the_native_solve_question_without_raising` reads it both ways for
      this reason); and the unnamed `factors` contract the family shares is **`n_dofs` + `apply` only**,
      so anything else the base reads off `self.factors` is an inheritance leak, not a contract.
    - **The transpose is closed-form, so the adjoint is served.** The transpose of a
      block-lower-triangular inverse is the block-upper-triangular one over the transposed blocks, so
      `apply(transpose=True)` reverses the two block solves and uses `Cᵀ` — pinned both as an exact dense
      transpose (unit) and as `⟨y, Mx⟩ = ⟨Mᵀy, x⟩` over real V-cycles on the real coupled Jacobian
      (integration).
      **`Cᵀ` is formed on FIRST USE, not at build (2026-08-14).** It was built eagerly in both
      `__init__`s and again in every `refactor`, and it is read **only** by the transpose apply — the
      adjoint's transpose solve. A forward march therefore carried a second full copy of the coupling
      block for its whole life, rebuilt at every refresh, and never touched it. It is now cached behind
      `_transposed_coupling`, with `_set_coupling` the one place that stores a coupling and drops the
      stale transpose with it (previously two sites set both fields in step, which is the pairing a
      third site would eventually get wrong). The reason it must not be re-derived *per apply* is
      unchanged and still recorded on the property: `A.T` on a compressed-sparse-row matrix is a
      compressed-sparse-column view whose product converts on every call. Not measured — the block's
      size is known but its share of a march's peak is not.
    **⚠️ A ONE-APPLICATION CONTRACTION RANKED THE TWO ORDERINGS AND WAS WRONG — invalid shortcut 2, in
    miniature, caught in a test rather than a write-up.** On the small coupled channel one application of
    the turbulence-first split leaves ~3× the input residual where flow-first leaves ~0.3×, which reads as
    a large quality gap. Through **GMRES on the true residual the two are indistinguishable**: both reach
    ~1e-14 inside one restart cycle, as does the monolithic control. Two lessons, and the second is the
    one that keeps costing time: a contraction ratio is not a convergence criterion for a
    Krylov-accelerated preconditioner; and *that state cannot rank the orderings at all*, because an
    operator every candidate solves in one cycle discriminates between none of them.
  - **✅ SHIPPED ON `bfs3d` — 31% FASTER END TO END, and the single-state probe below got the sign
    wrong.** A full 3-rung cold march at the identical configuration (`refresh_on_cycles=3`, ILU(0)×4,
    plain aggregation, `coarse_eq_limit` 2000, reach 3, restart 15), `field_split=True` against the
    shipped monolithic:

    | | monolithic | field split | |
    |---|---|---|---|
    | **wall** | 3140 s | **2161 s** | **−31%** |
    | steps | 58 | 66 | +14% |
    | Krylov cycles | 293 | 324 | **+11%** |
    | refresh | 19 events / 310 s | 23 / 352 s | +42 s |
    | mid-span `x_r/h` | 8.361 | **8.361** | identical |

    **The cycle row is the point.** The split is much faster *while doing more cycles*, because two
    smaller V-cycles plus one sparse coupling product apply far more cheaply than one six-field V-cycle
    — the coupling never enters a factorization or a coarse hierarchy. Per matched step at equal cycle
    counts the split's steps ran ~38–40% faster. Its mean cycles per **inner solve** is *lower* (1.49 vs
    1.68); the higher total is more, cheaper steps. It also crosses the refresh trigger slightly more
    often (9.7% of inner solves vs 9.0%), costing ~42 s of the ~980 s saved — the feedback loop is real
    and small. Refresh cost **per event** is unchanged (~14 s), so an earlier claim that the split
    refreshes more cheaply was wrong: it compared against the *scheduled* run's average.
    **Machine-load control:** the coloured jvp probe is identical work in both runs and took 11.3–14.6 s
    (monolithic) vs 11.7–15.1 s (split), so the faster run was not the quieter machine.
  - **⚠️ THE SINGLE-STATE PROBE BELOW SAID THE OPPOSITE — read it as a lesson, not as a result.** Harness
    `validation/bfs3d_openfoam/field_split_probe.py`. **Configuration, in full:** 3-rung cold march's
    own states; plain aggregation, **ILU(0) ×4** where not overridden, `coarse_eq_limit` 2000, stencil
    reach 3, block sizes 4 and 2; GMRES restart 15 to **rtol 1e-8 on the TRUE residual**; right-hand
    side the steady residual `−R(state)`; one materialization per state shared by every arm.

    | arm (flow / turbulence smoother) | hard iterate, β=0.0293, PC 0.05 | converged, **β=0** |
    |---|---|---|
    | monolithic ILU(0) — control | **3** (1.2e-12) | **16** (1.8e-11) |
    | split flow-first, ILU(0) / ILU(0) | 4 (1.3e-13) | **11** (3.7e-11) |
    | split turbulence-first, ILU(0) / ILU(0) | 4 (5.2e-14) | 13 (1.3e-10) |
    | split flow-first, ILU(0) / **damped Jacobi** | 6 (2.8e-12) | 16 (5.2e-10) |
    | split flow-first, ILU(0) / Chebyshev | 58 cap (3.3e-01) | 58 cap (2.6e-01) |
    | split flow-first, Chebyshev / ILU(0) | 58 cap (3.4e-02) | 58 cap (3.8e-03) |
    | split flow-first, Chebyshev / Chebyshev | 58 cap (9.97e-01) | 58 cap (9.5e-01) |
    | monolithic Chebyshev | 58 cap (9.9e-01) | 58 cap (5.4e-01) |
    | monolithic damped Jacobi | 58 cap (6.0e-01) | 58 cap (2.8e-01) |

    - **Forward: a small loss (4 vs 3), so do not adopt it for the march.** Both orderings tie there.
    - **Adjoint (`β = 0`, converged state): 11 vs 16, a 1.45× reduction** — and flow-first genuinely beats
      turbulence-first (11 vs 13), so the ordering *does* matter once the operator is hard enough to
      discriminate. This is the operator behind every `jax.grad` through a converged coupled solve, and
      it is the same place the monolithic ILUT's value turned out to lie. **Measure a coupled
      preconditioner at `β = 0`, not only on the march.**
    - **⚠️ `β = 0` must be taken at the CONVERGED state.** Stripping the shift off a mid-march iterate
      gives an operator neither the forward march nor the adjoint ever solves.
    - **⚠️ The probe's right-hand side is `−R`, and a dual-time step's is `−G = −(R + βd(φ−φₙ))`.** They
      coincide only at inner 0 (`φ = φₙ`) — which is why a sweep over end-of-step *checkpoints* is right
      to use `−R` — and on the hardest captured **inner** iterate they differ by ~200× (`|G|` 3.8e-03 vs
      `|R|` 8.3e-01). `φₙ` is not recorded by the inner observer, so `G` cannot be reconstructed. The
      self-check therefore gates on the **cycle count** (which reproduced the recorded 1) and reports the
      achieved residual without asserting it. Every arm sees the identical right-hand side, so the
      between-arm comparison is unaffected — but do not compare an absolute residual across the two.
  - **✅ THE INCOMPLETE-LU SWEEP CAN BE CONFINED TO THE SADDLE — the reachable half of the GPU prize.**
    Every smoother `solve/multigrid.py` has is Jacobi-class; the only thing PETSc supplies that it does
    not is the incomplete-LU sweep, which is also the least parallelizable piece (a sequential triangular
    solve). Two claims must be separated, because they have opposite answers:
    - **Remove it everywhere — REFUTED.** A Jacobi-class smoother on the four-field `[u,v,w,p]` block
      fails as badly as on the six-field block (both run to the restart cap). Taking ω out of the
      Chebyshev-smoothed block *helps a great deal* — 9.9e-01 → 3.4e-02 at the hard iterate, 5.4e-01 →
      3.8e-03 at `β=0`, one to two orders — so **ω-locality is real and is part of the obstruction**, as
      the cell-block singular-value decomposition predicted. But it never converges: the `[u,v,w,p]`
      block **is the saddle**, and a field split does not make it definite. Removing ω is necessary and
      not sufficient.
    - **Confine it to the saddle — SUPPORTED.** `[k,ω]` is not a saddle but a two-field
      advection-diffusion-reaction pair with a genuine diagonal, and **damped Jacobi on it converges** at
      both states (6 cycles / 2.8e-12 forward, 16 / 5.2e-10 at `β=0`), for a consistent ~1.5× cycle cost
      against ILU(0) on both blocks. At `β=0` it *ties the shipped monolithic* (16) while taking the
      incomplete factorization off two of six fields. So the k/ω hierarchy could be JAX-native, with
      PETSc confined to the four-field saddle.
    - **The Chebyshev-vs-Jacobi asymmetry on the SAME block is the mechanistic tell, and it is why
      "Jacobi-class" must not be treated as one arm.** Chebyshev **fails** on `[k,ω]` (58 cap, 2.6e-01 at
      `β=0`) where damped Jacobi converges. Chebyshev's polynomial is built for a bounded positive
      **real** spectrum; the transport pair is advection-dominated and strongly nonsymmetric, so its
      spectrum is complex and the real-interval polynomial is the wrong instrument. Damped Jacobi assumes
      only diagonal dominance, which first-order-upwind advection-diffusion-reaction has. So Chebyshev
      fails on both blocks for **two different reasons** — indefiniteness on the saddle, nonsymmetry on
      the scalars — and only the second is cured by dropping to Jacobi.
    - **Untested, and the honest caveat on the refuted half:** PETSc estimates Chebyshev's eigenvalue
      bounds with a few GMRES iterations, which is meaningless on an indefinite operator. A hand-bounded
      polynomial, or one designed for a complex spectrum, is not covered by these arms.
  - **✅ VANKA RE-OPENED ON THE FOUR-FIELD BLOCK, AND RE-CLOSED — but the ω half of the old mechanism is
    now CONFIRMED, quantitatively.** The standing verdict ("cell-centred patch relaxation is the wrong
    shape; do not re-open it with another patch variant") was measured on the **six-field** cell block,
    and its stated mechanism was that the block is weakly coupled in ω *everywhere*. A field split
    removes ω from the patch by construction, leaving the classical velocity-pressure patch the entire
    Vanka literature is built on — so that verdict does not transfer, and this is the new evidence its
    "do not re-litigate without new evidence" clause asks for. Same configuration as the table above,
    `state-00069`, β = 0, `vanka_centre_field = 3` (**mandatory** — the default is three fields from the
    end, which finds `p` in `[u,v,w,p,k,ω]` and would silently centre patches on `v` in a four-field
    block), patch width 22 = centre `[u,v,w,p]` + 6 neighbours × `[u,v,w]`:

    | arm | cycles | TRUE rel | worst patch gain |
    |---|---|---|---|
    | split, ILU(0) / ILU(0) | 11 | 3.7e-11 | — |
    | split, **Vanka** flow / ILU(0) | 58 cap | 2.9e-03 | **12.8** |
    | split, Vanka flow / damped Jacobi | 58 cap | 1.4e-01 | 12.8 |
    | split, **multiplicative** Vanka flow / ILU(0) | 40 | **1.55** (worse than the initial guess), 994 s | 12.8 |

    - **ω WAS the cause of the catastrophic patch conditioning — measured from the opposite direction.**
      The six-field campaign found the worst patch gain **flat at ~3e3 across every patch width**, which
      was the evidence that widening cannot help. On the four-field patch it is **12.8**, ~235× better,
      with **zero** patches dropped. The classical velocity-pressure patch is well conditioned on this
      operator. That is a clean confirmation of the ω-locality finding, obtained by *removing* ω rather
      than by inferring it from a singular-value decomposition.
    - **But patch conditioning was never the binding constraint, and this settles it.** A perfectly
      conditioned classical patch **still fails to converge** (stalls at 2.9e-03). The earlier campaign
      reached the same conclusion by *dropping* the near-singular patches, which was a confounded arm
      (it measured coverage); this reaches it by removing the ill-conditioning at the source, which is
      not confounded. Removing ω does help a great deal in absolute terms — the six-field Vanka stalled
      at 0.24–0.78, a 1.3–4× reduction, against 340× here — so the split genuinely strengthens the
      smoother, just nowhere near ILU(0)'s 11 cycles.
    - **Multiplicative is worse on the four-field block too, and hugely more expensive** (true residual
      1.55, 994 s vs 176 s, 17 colours). The six-field finding that sequencing costs rather than helps
      survives the split. Treat patch relaxation as closed on this operator, now for a *measured*
      reason rather than an inferred one.
    - **Two weak smoothers compound**: Vanka flow + Jacobi k/ω is 1.4e-01, far worse than either
      weakness alone. The two blocks' smoothers are independent *settings* but not independent in effect.
  - **⚠️ THE JAX-NATIVE HIERARCHIES CANNOT BE BUILT ON THE `[k,ω]` JACOBIAN SLICE AT ALL — and the
    reason is structural, not a cost.** Both were tried as the trailing block's inverse and both failed,
    for **one shared cause** rather than two incidental ones:
    - `build_convection_hierarchy` (aggregation) **refuses**: *"operator diagonal must be finite and
      strictly positive, but its minimum is −2.129e+06"*.
      **⚠️ THE OBVIOUS EXPLANATION — that the true `[k,ω]` block carries negative diagonals from its
      live source linearizations — IS WRONG, AND THE ERROR MESSAGE ITSELF SAYS SO (corrected
      2026-08-09).**
      The refusal names **`level 1`**, a *coarse* operator. The fine slice is clean: measured on `bfs3d`
      `state-00057`, **0 of 23040 cells** have a non-positive diagonal, at β = 0, at the march's shift
      and at the preconditioner floor alike (`validation/bfs3d_openfoam/trailing_block_conditioning.py`).
      The negative diagonal is *manufactured by the aggregation*, in the Galerkin `R A P` row — which is
      one of the three causes the guard's own docstring lists.
      **The real cause is that `_aggregate` is FIELD-BLIND.** It takes a bare matrix with no block size,
      so on a multi-field block it can merge a `k` degree of freedom with an `ω` one from a *different*
      cell into a single aggregate, and on a strongly nonsymmetric operator that produces a degenerate
      coarse row. PETSc GAMG does not hit this because it is told `setBlockSize(n_fields)` and coarsens
      whole **cells**. So the obstruction is the coarsening's blindness to fields, not the fine
      operator, not the source linearizations, and not the sign of anything the caller supplies.
      **Consequence: one hierarchy PER FIELD works, on the REAL Jacobian sub-blocks.** An aggregate
      cannot mix fields when there is only one, and `build_convection_hierarchy` accepts `A_kk`
      (diagonal 1.16e-06 … 2.66e-04, 57 nnz/row) and `A_ωω` (3.03e-03 … 1.0, 49 nnz/row) at every
      shift. That keeps the full stencil fill and the true source linearizations, and needs **no**
      reparametrization scaling, because the Jacobian is already in the solved variable — the log-ω
      chain factor and its wall-fixation trap simply do not arise. Built as
      `solve/field_split.PerFieldNativeInverse` / `native_per_field_inverse` (**both DELETED 2026-08-15**
      — see the deletion note in the field-split section; the surviving inverse is `NodalNativeInverse`),
      with the two fields
      composed block-triangularly (k leading).
      **This corrected a real cost: the wrong explanation was taken as an accepted blocker and sent the
      first implementation down a transport-operator detour** — a 13×-sparser, source-clamped, per-field
      stand-in needing the closure, the mass flux and the reparametrization scale — which measured 5
      cycles against the true sub-blocks' 1 on the channel and was deleted. The generalisable lesson:
      the record quoted an error verbatim and stapled an *inference* to it, and only the quotation was
      ever verified. The word `level 1` was in the quotation the whole time.
    - `build_air_hierarchy` (lAIR) ran **~50 minutes without finishing** and was killed. Contributing:
      the slice carries the distance-3 coupled fill at **91 nnz/row** where the frozen transport stencil
      is ~7 (the leading block's slice is 227), and lAIR's local approximate-ideal-restriction solves run
      over a degree-2 F-neighbourhood whose size grows roughly with the square of the row density.
    **Neither is a verdict on the method, and this is MEASURED rather than argued.** Both builders assume
    an M-matrix-like operator, and `scalar_transport_preconditioner` never hands them one that isn't:
    `_scalar_operator_pieces` **clamps its reaction diagonal non-negative** for exactly this reason (an
    anti-diffusive source would make the operator indefinite and its V-cycle diverge). The Jacobian slice
    is the unclamped truth, so it is simply not in these builders' domain. PETSc GAMG with an ILU(0) or
    Jacobi smoother is untroubled because it assumes none of this. Built on the **transport** operator
    instead, at the same state on the same mesh, both are perfectly healthy:

    | hierarchy on the transport operator (`bfs3d`, 23040 cells, `state-00069`) | k | ω |
    |---|---|---|
    | `twolevel` (the shipped scalar default) | 5.0 s | 44.3 s |
    | `air` (lAIR) | 80.0 s | 79.8 s |

    So lAIR builds in ~80 s where it did not finish in 50 minutes on the slice — a ≥40× gap on the same
    mesh — and is ~3.2× the shipped aggregation's combined build (1.8× on ω alone), which matches the
    independent recollection that production lAIR on the 2D case worked and was only somewhat slower.
    **Do not read the slice failures as anything about lAIR's cost in normal use.**
    **Note also that row density was only part of the story and was the weaker part.** The decisive fact
    is the **negative diagonal**, which is a property of the true Jacobian block whatever its sparsity;
    the 91-vs-7 nnz/row gap explains lAIR's *time* but not aggregation's outright refusal.
    **So the correct arm builds the hierarchy on the TRANSPORT operator** —
    `SSTTurbulence.k_preconditioner` / `omega_preconditioner`, which is what the segregated scalar path
    already does — accepting that it then approximates a *different* matrix from `A_tt` (no k↔ω coupling,
    no reach-3 fill, clamped diagonal), the same approximation the block-preconditioner family already
    makes. That arm is **not yet built**: it needs `mdot` and the closure at the state, a `k ⊕ ω`
    block-diagonal composition, and the log-ω chain-rule scaling (`ScaledScalarPreconditioner`) — the
    last of which is a known trap, since a rescale that ignores the wall-fixation rows' own derivative
    cost 27× on the linear residual once already.
  - **⚠️⚠️ SUPERSEDED (2026-08-10) — EVERY COST NUMBER IN THIS BULLET IS VOID, AND ITS CONCLUSION IS
    REVERSED. The native trailing inverse is now AT PARITY on cycles (2 restart cycles against 2 on the
    `[k, ω]` block alone; full configuration below) and at parity on per-apply cost — the pair of timings
    once quoted here carried no sweep count and no state and is deleted.** Both stated blockers are gone.
    The "~2.8× more expensive"
    figure was the **COO `segment_sum`** matvec, not anything intrinsic to a framework-native
    V-cycle — a CSR operator took the apply from 117.8 ms to 13.3 ms (8.8×), and the level operator
    is applied ~10× per cycle so that was essentially all of it. The "learn a block size" half was
    built and is what closed the *quality* gap. Read the 2026-08-10 section at the top of this
    IN-PROGRESS group; what remains open there is the positivity limiter, not the preconditioner.
    **The bullet is kept, unedited below, for two reasons that are worth more than the numbers:** it
    records that the deleted per-field inverse (two per-field hierarchies) was a different and weaker object
    than the single nodal one now used, so its measurements never transferred; and it is the clearest
    instance in this file of a *cost* attributed to a method when it belonged to a data structure.

    **✅ CONFIRMED ON A MARCH (2026-08-11), and the parity above is now a WIN.** Block-level parity does
    not by itself say which arm marches faster, so it was measured end to end as a controlled pair.
    *Configuration, both arms:* `bfs3d`, field split, `zerogradient` k wall BC, `K_POSITIVITY_FLOOR`
    1e-08, `equilibrate=False`, `refresh_on_cycles` 3, β-mismatch gate off, ILU(0) × 4 on the saddle,
    trailing sweeps 1, `coarse_eq_limit` 2000, restart 15, `forward_rtol` 0.3, two continuation rungs.

    | | native (`march-20260811-132658.log`) | petsc (`march-20260811-161642.log`) |
    |---|---|---|
    | wall | **2124 s** | 2893 s (+36 %) |
    | steps | **67** | 72 |
    | Krylov cycles | **329** | 371 |
    | escalations | **4** | 8 |
    | mid-span `x_r/h` | 8.361 | 8.361 |

    **The entire difference is ONE event, and the arms are otherwise bit-identical.** They track for
    **49 steps** — same β, same ‖R‖ to four figures (8.810e-03, 7.274e-03), same α — and part at step
    50. That is the controlled-pair signature the confounded comparison lacked, and it is worth more
    than the totals: it localizes the whole 769 s to the positivity-cap collapse, which each arm meets
    at a *different* step. Native meets it at 50 (cap 1.01e-02 → 1.44e-05); petsc sails through 50–52
    and meets it at 53–55 (α 0.316 / 0.075 / 0.005). Both then pay the cascade, and petsc's is worse —
    β driven to **4.4394** against **0.9364**, hence ten walk-back steps against six by the law below.
    So the arms differ in *when* they hit the cap and *how far* β is driven, **not** in how well either
    inverts its block — and the remaining wall time is in the cascade, not the preconditioner.

    So the native inverse is not merely at parity on a march, it is ahead, and **the case defaults moved
    to it** (`BFS3D_TURBULENCE_INVERSE` now defaults to `native`, with `K_WALL` `zerogradient` and
    `K_POSITIVITY_FLOOR` 1e-08 moved in the same change so the default configuration *is* the arm
    measured above; `compare.py` states all three in its banner).

    **⚠️ THE "PETSC IS FASTER" READING THAT THIS REPLACES WAS A THREE-WAY CONFOUND, AND NO SINGLE LOG
    WAS WRONG.** The belief was 58 steps against 67. The 58-step run (`march-20260811-095526.log`) used
    the **`dirichlet`** k wall BC, had **no positivity floor**, and predated both `ad3e144` and
    `bb46032` — three differences at once, each correctly stated in its own banner. Held like-for-like
    the ranking reverses, as above. **The cheap check that would have caught it on day one:** two arms
    differing only in the preconditioner solve the *same* operator, so their residuals must track for
    the first few steps. These separated at **step 1** (2.051e-01 against 2.046e-01). That test is now
    `validation/bfs3d_openfoam/march_log_compare.py`, which reports the first step at which two logs
    part. Matching aggregate step counts are not evidence of a controlled comparison.

    **The step-count gap is the positivity limiter, not the inverse** — consistent with the `limit`
    warning further down this file. Minimum step cap over a whole march: **2.02e-01** under `dirichlet`,
    against **1.44e-05** (native) and **5.41e-03** (petsc) under `zerogradient`. Under `zerogradient`
    near-wall `k` is free to ratchet down until one numerically dead cell (12800, `k ≈ 3e-20`) sets the
    global cap, which is why the `dirichlet` arm looked fast. It is a different discrete problem, not a
    faster solver.

    **The wall condition's own cost, as a controlled pair** (2026-08-11, `march-bc-dirichlet-native.log`
    against `march-20260811-132658.log`): same code, same native inverse, same 1e-08 floor, only the k
    wall BC differing — `dirichlet` **59 steps / 1911 s / 292 cycles**, `zerogradient` **67 / 2124 /
    329**, both to mid-span `x_r/h` 8.36. So the BC is worth ~11 % of the wall, and the earlier
    cross-version "58 against 67" is superseded by this same-code pair. Note what it does *not* say:
    under `dirichlet` the two inverses are within one step of each other (native 59, petsc 58 — and the
    58 is on older code), so **the native inverse's margin is regime-dependent**, clear under
    `zerogradient` and inside the noise under `dirichlet`. The default rests on `zerogradient` being the
    selected physics, not on the inverse winning everywhere.

    **The reaction to a collapsed cap is quantified, and the RETURN is the waste — not the ladder.** A
    capped step trips `retry.on_alpha` and the ladder escalates β (×2 per retry, to
    `retry.cycles_limit`); β then has to be walked back at `/grow` per step, and that unwind is
    arithmetic: **walk-back steps = log(β_peak / β_resume) / log(grow)** — predicted 2.4 / 6.0 / 10.0
    across the three marches above, observed ~2 / 6 / 10. Replaying `CflResidualDualTimeControl` over
    each logged (α, ‖R‖) sequence reproduces every logged β to four decimals on **all 22 archived
    marches**, so the law is checkable rather than inferred. The cascade regions are **12.4 %** of wall
    (native trio) and **24.7 %** (petsc).

    **⚠️ THREE OBVIOUS FIXES ARE REFUTED BY THE ARCHIVED LOGS — do not re-propose them (2026-08-11).**
    Each was proposed here first and killed by a counterexample:

    - **"Stop a ladder whose attempts move nothing."** `march-20260811-161642.log` step 56: attempts 1
      and 2 are bit-null (`G` 5.670e-03 → 5.669e-03 → 5.668e-03, α 0.000, **0 cycles each**), and
      attempt 3 at β 4.4394 is the one that **cures** the step (α 1.000). Any "stop after N nulls" with
      N ≤ 2 stops one attempt short of the cure, and the accepted null step then triggers the control's
      backoff to the same β anyway — paying a whole outer step (12–30 s) to save two 0-cycle attempts.
    - **Decaying `carry_beta`.** At `132658` step 52 the ladder *tried* β = 0.4682 and it was rejected;
      the cure was 0.9364. Carrying any decayed fraction (half → 0.47, geometric → 0.66) lands the next
      step at or below a β just demonstrated to fail at that state, and re-pays the escalation at once.
    - **"The control should skip its own backoff after an escalation."** They are a **serial search, not
      a redundancy**: across all 24 archived logs there is **not one instance** of the control backing
      off after a *successful* escalation (α ≥ 0.25 ⇒ it grows or holds). The two only co-occur when the
      ladder was *exhausted*, where the control's doubling continues the same search — the ladder gave
      up at 0.2341 / 0.5549 and the cures were 0.9364 / 4.4394. Suppressing it lengthens each cascade.

    **⚠️ "Damping cannot widen a positivity cap" is FALSE as a general statement.** The per-attempt
    `checkpoints/step-limit-*.npz` dumps (which `compare.py` wrongly called unobservable) show
    `d(cap)/dβ` with **no fixed sign** at a fixed anchor: doubling β narrowed the cap ×9.4 then ×1.6 on
    one anchor and widened it ×3.3 then ×5.6 on another, and on a third the binding cell *changed
    between attempts* (3181 → 22400 → 12800) so consecutive caps do not even measure the same thing.
    This does not reopen gating on `binding_limit == 1` (measured and reverted, below).

    **⚠️ AND THE FUTILITY READING IS CONFOUNDED.** The native trio's step-51 attempts all ran with
    `pc none 0.0s` — `REFRESH_ON_BETA` defaults to `inf`, so the `precondition_step` the march calls on
    every escalation does nothing, and every escalated attempt was solved against a V-cycle built for
    β = 0.0293. Step 52 escaped only because its inner solve happened to trip the mid-step rebuild. So
    **"the ladder is futile" and "the ladder was never given a matched preconditioner" are NOT separated
    by this data.** `BFS3D_REFRESH_ON_BETA=0.9` is the existing knob that discriminates them, at roughly
    +35 s of rebuild on one march.

    **What survives as the lever: the asymmetric return.** β is driven up by ×2 (backoff) and up to ×4
    (ladder) but recovers at only ÷1.5 per step — ~5:1 in log space — and every walk-back step is
    massively over-damped (inner counts 2,2,2,2,3,5 and 2,2,2,2,2,2,2,2,2,3, converging the implicit
    step to ~1e-7 of its start). Accelerating the ramp *only while unwinding an escalation* is
    byte-identical on any march that never escalates, and is worth ≈3 steps on the trio and 5–6 on
    `161642` by exact control arithmetic. Second lever: **the ladder is not clamped by the control** —
    it multiplies the β leaf past `beta_max`, giving β = 4.4394 against a ceiling of 4.0 and a stable
    β = 16.0 limit cycle for **74 consecutive steps** in `march-20260810-090711.log`.

    **⚠️ HISTORICAL — measured 2026-08-09, void since:** taking PETSc off the trailing half is not a
    win, and the two blockers are now specific (`bfs3d`, three states, arms `native`/`native2` in
    `turbulence_smoother_sweep.py`). With the per-field hierarchies above wired in as the trailing
    inverse, against the shipped PETSc V-cycle at `ilu0` on both halves, `refresh_on_cycles` 3, plain
    aggregation, `coarse_eq_limit` 2000, reach 3, restart 15:

    | trailing inverse | 00057 | 00058 | hard | blended | apply |
    |---|---|---|---|---|---|
    | PETSc V-cycle (shipped) | 8.66 | 8.83 | 8.88 | **8.76** | 102.0 ms |
    | native, 1 V-cycle/field | 13.09 | 13.22 | 13.23 | 13.16 (**1.50×**) | 144.3 ms |
    | native, 2 V-cycles/field | 16.50 | 17.03 | 12.94 | 16.34 (1.87×) | 204.0 ms |

    Uniformly 1.50× (spread 1.49–1.51) with apply flat to 0.3 % — a low-variance loss, unlike the
    block-Jacobi arm's. The second row shows the two deficits are separable and neither is free: more
    V-cycles close the **quality** gap (tight cycles 8 → 5 against ILU's 4) and double the **cost**.
    **The apply gap decomposes, and a traced-side port would NOT rescue it.** Timed on the same block:
    the two JAX hierarchies jitted, `jnp` in and out, cost **32.6 ms**; through the study adapter's
    numpy↔jnp marshalling, **53.8 ms**; the PETSc V-cycle they replace, **~11.7 ms** (by difference
    against the shipped split's 102.0 ms total, the flow half being identical). So marshalling is only
    ~half the gap and **the JAX V-cycle is ~2.8× more expensive than GAMG on the identical operator**.
    Removing the callback would take 144 → 123 ms, still 20 % above the incumbent and still weaker.
    (If anything 32.6 ms flatters it: XLA constant-folded the frozen hierarchy arrays into the jit.)
    **So the next attempt starts from "the V-cycle must get ~3× faster and learn a block size", not
    from "remove the callback".** The second half is a defined enhancement rather than a research
    problem — teach `_aggregate` a block size so it coarsens *cells* as GAMG does, which would let one
    two-field native hierarchy replace the per-field pair and close the quality gap without doubling
    the cycles. It does nothing about the 2.8×. **`PerFieldNativeInverse` is GONE (deleted 2026-08-15);**
    the paragraph below is why it was kept until then, and it
    is tested, transposable, adjoint-legal, and is what a block-aware aggregation would slot into — it
    is **not** wired into the production builder.

  - **NOT YET BUILT: the production wiring.** There is no `coupled_field_split_continuation` / shift
    policy, deliberately — the forward march is where a continuation builder would be used and the split
    *loses* there. What the measurement argues for is a **`β = 0` adjoint-only** preconditioner seam
    (`ForwardStep.adjoint_preconditioner()` already exists as the natural home), plus the JAX-native k/ω
    hierarchy the damped-Jacobi result unlocks. Both are unbuilt.
  - **⚠️⚠️ `trailing_smoother_sweeps` DOES NOT REACH AN INJECTED TRAILING INVERSE, SO THE NATIVE ARM
    SHIPS AT FOUR SWEEPS, NOT ONE (verified in source, 2026-08-14).** In
    `build_block_triangular_field_split` the injected inverse is called as
    `trailing_inverse(trailing_block, groups.n_trailing_fields)` — two arguments — while
    `smoother_sweeps=trailing_smoother_sweeps` is passed **only on the `else` branch** that builds its own
    `build_amg_vcycle`. `native_nodal_inverse`'s own default is `sweeps=4`, and the case's
    `NATIVE_TRAILING` sets only `max_coarse` and `equilibrate`. **So the entry below, and its measured
    16.5 % wall saving, describe the PETSc trailing V-cycle — which this case no longer uses.** Cutting
    the NATIVE trailing sweeps is therefore an untaken lever, not a shipped setting: measured on a
    same-shape synthetic, its apply runs 10.7 / 6.9 / 5.3 ms at 4 / 2 / 1 sweeps.
    ⚠️ **Two more shipped-vs-record mismatches found in the same pass:** the case runs
    `equilibrate=False` (the class default is `True`), so every statement here about the *equilibrated*
    `[k, ω]` cell block — unit diagonal, determinant exactly 1, subdiagonal 100–340 — describes an
    operator the shipped configuration does not build; and `NATIVE_TRAILING`'s `max_coarse=2000` is
    **inert**, because at `max_levels = _CONVECTION_LEVELS = 2` the level cap fires first regardless
    (`max_coarse=16` builds a bit-identical hierarchy).

  - **✅ THE TWO HALVES ARE NOW SMOOTHED APART, AND THE TRAILING DEFAULT IS ONE SWEEP —
    `trailing_smoother_sweeps=1` (BUILT, SHIPPED, 2026-08-09).** Splitting the hierarchies is only half
    the value; the other half is that they can then be *tuned* apart, which the shipped bundle was not
    doing — both halves inherited the four incomplete-LU sweeps tuned against the **six-field**
    monolithic block. The saddle needs them (Jacobi-class smoothers do not converge on it at all); the
    transported-scalar pair does not. `build_block_triangular_field_split` /
    `FieldSplitAmgPreconditioner.build` / `coupled_amg_continuation` all carry
    `smoother_sweeps` (leading) and `trailing_smoother_sweeps` (trailing, default **1**) as separate
    parameters, plus `leading_options` / `trailing_options` as the raw-PETSc escape hatch. Measured on
    the full 3-point `bfs3d` Reynolds-continuation march (field split, `retry.on_alpha` 0.01,
    `refresh_on_cycles` 3, ILU(0), plain aggregation, `coarse_eq_limit` 2000, reach 3, restart 15):

    | | 4 sweeps | 1 sweep |
    |---|---|---|
    | wall | 1959 s | **1636 s (−16.5 %)** |
    | steps | 58 | 58 |
    | refresh | 21 events / 318 s | 19 / 286 s |
    | Krylov cycles | 277 | **282 (+1.8 %)** |
    | final ‖R‖ | 9.589e-06 | 9.588e-06 |
    | mid-span `x_r/h` | 8.361 | 8.361 |

    **The two marches follow the same trajectory step for step** — identical β, identical per-step
    cycle counts, identical residuals to four figures, and the single α-collapse escalation fires at
    the same step for the same reason to the same β. So this is the *same* path at a lower price per
    matrix-vector product, which is a far stronger single-run result than a 16.5 % margin would
    normally be. (An earlier version justified that with a "~2 %" run-to-run noise figure for this case;
    it was a remembered number with no configuration behind it and is deleted — the strength of the result
    rests on the step-for-step identity, not on a noise floor.) Note again that **cycles rose while wall
    fell**.
  - **⚠️ HOW THAT SMOOTHER WAS CHOSEN, AND THE TWO WAYS THE SCREEN NEARLY GOT IT WRONG
    (`validation/bfs3d_openfoam/turbulence_smoother_sweep.py`).** The screen holds the leading half at
    ILU(0) and varies only the trailing one, ranking on **wall time** at real march states rather than
    on cycles. Two failures are worth carrying, because both produced a wrong answer first:
    - **A HARD state cannot rank candidates — it can only screen them.** The standing caution is about
      *benign* states not discriminating; the complement is equally real and was not written down. On
      the shipped march **139 of 194 inner solves cost one restart cycle and only 7 exceeded three**,
      so the worst iterate is not what a march pays for. Point-block Jacobi buys an extra cycle *only*
      where the operator is hard: it ranks **1.15× (worst of ten)** on the hard iterate and **0.94×**
      over the march's real mix. Rank on step-initial states, screen on the hard one, and weight the
      blend by the march's own solve distribution.
    - **A cost screen is blind to the DIRECTION a preconditioner returns, and a march is not.** The
      forward solve stops at `forward_rtol = 0.3`; a strong preconditioner overshoots that by orders of
      magnitude inside one restart cycle while a weak one lands near it, and **both report one cycle**.
      Since the march takes an inexact-Newton step from whatever comes back, a materially weaker arm
      hands back a worse direction, the line search clips and the step control escalates — none of
      which a timing screen registers. The usable proxy is the screen's **tight** cycle count read as a
      *magnitude*, not as a pass/fail gate: shipped 3/3/4 (two step-initial states, then the hard one),
      `ilu0x1` 3/4/5 — about the same strength, and it reproduced the baseline trajectory exactly —
      against `jacobix2` 5/7/7 and `jacobix1` 8/10/13. **A candidate needs three things: cheap per
      application, convergent at all, and not materially weaker than the incumbent.**
    - **THE `[k, ω]` CELL BLOCK IS TRIANGULAR, AND THAT EXPLAINS THE WHOLE SMOOTHER RANKING ON THIS
      BLOCK.** Equilibrated, each cell's 2×2 is **lower-triangular with unit diagonal and a subdiagonal
      of order 100–340**, every determinant exactly 1.0, and the k↔ω coupling is **~100 % same-cell**
      (‖∂R_ω/∂k‖ splits 2.14e3 same-cell against 23.9 neighbour on the channel;
      `validation/bfs3d_openfoam/field_coupling.py` finds the same asymmetry on `bfs3d` — ∂R_ω/∂k at
      121× the diagonal blocks, ∂R_k/∂ω at 0.19 % of them). ω depends enormously on same-cell k through
      the production limiter and the βkω destruction pair; k barely depends on ω, because wherever the
      SST limiter is active `ν_t = a₁k/max(a₁ω, S F₂)` is **ω-independent**. Consequences:
      - **ILU(0) in cell-major order already captures it exactly** — the forward/backward pair is a
        complete factorization of a 2×2 with one off-diagonal (no fill), so the subdiagonal costs it
        nothing. That is why one sweep is nearly as good as four here (tight cycles 3/4/5 against
        3/3/4) and why `sor`, also a sweep in the favourable order, matches ILU(0)×4 at 3/3/4.
      - **Point Jacobi discards the entire subdiagonal**, which is precisely the term the others get.
      - **Point-block Jacobi recovers it, and measurably does**: split by field, a `pbjacobi`-smoothed
        V-cycle lands **10× closer to an ILU-smoothed one than point Jacobi does in the ω rows**
        (1.48e-4 against 1.49e-3). The end-to-end near-tie is real all the same — the V-cycle output is
        ω-dominated (‖ω‖ 2318 vs ‖k‖ 29.7), all three get the bulk right, and the coarse correction
        plus outer Krylov absorb the rest.
      - **Re-ordering the fields does NOT help.** A symmetric permutation to `(ω, k)` makes the block
        upper-triangular; Jacobi and block-Jacobi are permutation-invariant, ILU holds both triangles
        either way, and a *forward* sweep (SOR/Gauss-Seidel) is made strictly worse — the current order
        lets it solve k first and use the fresh value in ω. The shipped order is already the good one.
      - **For a JAX-native smoother this is the cheap prize:** with the equilibrated unit diagonal the
        block solve is one fused multiply-add per cell (`x_ω -= c·x_k`), fully parallel across cells,
        no factorization and no sequential dependency.
    - **⚠️ BLOCK JACOBI CANNOT REPLACE THE TRAILING HIERARCHY — measured, and the reason is
      ROBUSTNESS rather than reduction.** The `[k, ω]` block is strongly *block*-diagonally dominant
      (neighbour coupling ~12 % of same-cell for the diagonal fields, 1.1 % for `∂R_ω/∂k`), and block
      Jacobi's error operator is exactly that neighbour part — so it looks as though a coarse grid has
      nothing to do here, and on a synthetic operator at that neighbour weight it contracts ~10× per
      sweep. It does not carry over. Measured as the **whole** trailing inverse (a native batched 2×2
      solve, no hierarchy at all) against the shipped V-cycle, on two step-initial states and the hard
      iterate, blended by the march's own solve mix:

      | trailing inverse | 00057 | 00058 | hard | blended |
      |---|---|---|---|---|
      | `ilu0` V-cycle (shipped) | 8.68 | 8.74 | 8.72 | **8.71** |
      | block Jacobi ×4 | **8.63** | 10.94 | 11.17 | 9.94 (1.14×) |
      | block Jacobi ×8 | 9.09 | 11.32 | 9.29 | 10.10 (1.16×) |
      | block Jacobi ×2 | 10.40 | 10.47 | 10.88 | 10.48 (1.20×) |
      | block Jacobi ×1 | 13.72 | 14.09 | 14.29 | 13.95 (1.60×) |

      **What the coarse grid buys is variance, not average.** The V-cycle is 8.68/8.74/8.72 — flat to
      under 1 % across all three states. Block Jacobi ×4 swings 8.63 → 10.94 between *adjacent steps of
      the same rung*, because it sits on the edge of converging inside one restart cycle and small
      state changes flip it to two. And more sweeps cannot buy the edge back: by 8 sweeps its apply
      cost (109 ms) exceeds the V-cycle's (102 ms), so it has spent the whole cost advantage
      recovering what the coarse grid gave for free.
      **This does NOT refute block Jacobi as a SMOOTHER inside a JAX-native hierarchy**, which is the
      actual route for taking PETSc off this block and is untested. The implementation
      (`BlockJacobiInverse` in `validation/bfs3d_openfoam/field_split_probe.py`) is verified exact on a
      block-diagonal operator in one sweep, transposable in closed form (⟨y,Mx⟩ = ⟨Mᵀy,x⟩ to 1e-15, so
      adjoint-legal) and a fixed **linear** operator (1e-16, so the non-flexible outer Krylov is
      legal) — it is the smoother such a hierarchy would need.
      **Methodological note, because it went wrong twice in one session:** the first easy state alone
      showed a tie and was reported as a result. The second easy state, same class and adjacent step,
      was 25 % worse. **Hold a screen's conclusion until every state has landed** — the per-class
      spread is itself the finding here.
    - **⚠️ CHECKPOINT NAMES ARE REUSED ACROSS MARCHES, so a probe can silently run at a state that is
      not the one its table documents.** `StateCheckpointer` keeps only the last few files
      (`BFS3D_CHECKPOINT_KEEP`, default 3) and numbers them from a counter that restarts each run, so a
      later march *replaces* `state-000NN` with a different state under the same name. Observed: a name
      documented as the converged zero-shift adjoint operator came back holding a mid-march iterate at
      shift 0.98 from an abandoned run — a probe would have paired an operator built at the documented
      shift with a state that never had it and reported it as a measurement at that operating point.
      `field_split_probe.load_state` now checks the checkpoint's recorded shift against its `STATES`
      entry and refuses on a mismatch (loosely, at 2 %: the table stores ~4 figures, and what this must
      catch differs by orders of magnitude). Raise the keep count for a study that needs a trajectory.
    - **⚠️ `equilibrate_cell_major` RETURNS UNSORTED COLUMN INDICES, and PETSc's AIJ format requires
      them ascending.** `AmgVCycle._build` and `refactor` both `sort_indices()` before wrapping the
      matrix, and `ShiftedCellMajorOperator` genuinely produces sorted output (its
      `has_sorted_indices = True` is honest, verified), so **every shipped path is correct**. But a
      probe that calls `equilibrate_cell_major` directly and feeds `createAIJWithArrays` gets **NaN in
      most entries from `pbjacobi`**, while `jacobi` and `ilu` survive it — a diagonal scan does not
      care about column order and a block extraction does. That asymmetry reads exactly like "PETSc's
      point-block Jacobi is broken on this operator", which was written up and had to be retracted.
      **Check `has_sorted_indices` before concluding anything about a block method.**
    - **⚠️ TWO MARCH ARMS WERE RUN AT THE WRONG REFRESH TRIGGER AND ARE VOID** — launched without
      `BFS3D_REFRESH_ON_CYCLES=3` when that variable still defaulted to `0` (the *scheduled* cadence,
      measured at 3632 s against 1959 s). Both "results" were the refresh trigger, not the smoother.
      The trap was already written down in bold and that did not prevent it, so the fix was made
      structural: **the case default is now 3**, and every run writes its full configuration into its
      own `march.log` header before any result. A default nobody wants is a trap, not a setting.
    - **Still unsolved, and it bites any future arm comparison:** `refresh_on_cycles` and
      `cycle_budget` are denominated in **cycles**, so a preconditioner that shifts the cycle
      distribution changes the *effective* trigger point — penalizing a weaker arm twice, once in
      cycles and again in the refreshes those cycles provoke. (Same argument as the case's
      `_RESTART_SCALE`.) Report the refresh **count** beside the wall in every arm comparison; it came
      out level for `ilu0x1` (21 vs 19 events), which is what licensed attributing its saving to the
      smoother.

## The JAX-native multigrid — in progress

- **⚠️ IN PROGRESS (2026-08-10): the native trailing V-cycle now MATCHES PETSc on quality AND cost;
  what blocks it is the POSITIVITY LIMITER — read that as the PROXIMATE failure, not the cause.**
  ⚠️ The same limiter, defaults and case run fine under the PETSc ILU(0) control (58 steps to 9.588e-06,
  recorded below), so the limiter is necessary and not sufficient: the native arm dies at a state the
  control never reaches, and what drives that arm's direction into the boundary is **unexplained**. Read this before the
  2026-08-09 section below, which it supersedes in several places.

  **Where it stands. Configuration, stated here rather than 500 lines away, because the sweep count
  is load-bearing:** `bfs3d` `state-00057`, PC β 0.05, the `[k, ω]` block **ALONE** (46080 dofs,
  4.20M nnz), GMRES restart 15 to rtol 1e-8 on the TRUE residual, **4 smoother sweeps**, the PETSc
  side on its **matched-smoother** arm (plain aggregation, point-block Jacobi ×4 — against ILU(0) ×4
  PETSc does 1 cycle, not 2); harness `trailing_hierarchy_sweep.py`. On that arm the JAX-native nodal
  hierarchy reaches **2 restart cycles against PETSc GAMG's 2**, on a 438-equation coarse space
  against 432. Quality parity, on CPU. Per-apply cost came out at parity too, but that pair of
  timings was recorded with no sweep count and no state and is deleted — measured 2026-08-10,
  configuration not recorded, re-measure before relying on it. The full march converges
  rung 1 (14 steps, 43 cycles, 304 s against the PETSc control's 14 / 45 / 359) and then dies on
  rung 2.

  **Four things closed the gap, and each was found by reading the reference rather than tuning:**
  1. **The aggressive first level.** `build_amg_vcycle` never sets `pc_gamg_aggressive_coarsening`,
     so GAMG applies its default of one aggressive level over the SQUARED graph. Ours had none:
     21× coarsening against GAMG's 107×, the whole 5× coarse-space difference. (`use_aggressive_
     square_graph` and `aggressive_mis_k` are ALTERNATIVES; at the default the coarsener is plain
     MIS at distance 1 on the squared graph.)
  2. **PETSc's level smoother is UNDAMPED** — `richardson` at its default scale of 1. Ours relaxed
     by `omega/lam_max`, and `D⁻¹A` has a unit diagonal so `lam_max ≥ 1` always: the spectral factor
     can only ever under-relax. Worth **10 → 2 cycles**. Reachable as `spectral_damping=False`.
  3. **A CSR matvec instead of the COO `segment_sum`.** A scatter-add collides on output rows; a CSR
     row walk does not. Measured on the 4.2M-nnz block: `segment_sum` 13.3 ms, `BCOO @ x` 13.4 ms,
     scipy CSR on the host 2.6 ms, **`BCSR @ x` 1.4 ms**. The level operator is applied ~10× per
     V-cycle, so this alone took the apply from **117.8 ms to 13.3 ms (8.8×)**. Landed as
     `_CsrOperator`, which owns its matvec and replaced the loose `row`/`col`/`val`. **Not** a jit
     or marshalling problem — both were measured out (1 trace everywhere; a numpy↔jnp round trip is
     0.02 ms), which also retires the older claim that "marshalling is ~half the gap".
  4. **The singularity guard was WRONG, and it was aborting the march.** `_cell_block_inverse`
     tested `|det| < 1e-12 · ‖B‖_F^b`. On the coupled turbulence pair a cell reads
     `[[8.8e-06, 1.7e-12], [-1.3e+03, 1.5e-01]]` — rows differing by **1.5e8** — so `‖B‖_F²` ≈ 1.6e6
     is set entirely by the ω row and the bar lands at 1.6e-06, just above the determinant of
     1.35e-06. The block is **not singular**: on the row-norm (Hadamard) bound `|det| ≤ ∏‖rowᵢ‖` it
     scores **1.2e-04**, eight orders clear. Now tested that way, which is invariant under rescaling
     any row or column where the Frobenius form is not. A structurally empty row is named
     explicitly, since it makes the bound zero and no determinant compares below it.

  **⚠️ THE α COLLAPSE IN THESE MARCHES IS THE POSITIVITY LIMITER, AND THE STEP TABLE SAYS SO — READ
  `limit` BEFORE ATTRIBUTING ANYTHING TO THE PRECONDITIONER.** `a_min` and the `limit` aside are the
  **same number** wherever both appear (0.579/5.79e-01, 0.651/6.51e-01, 0.004/3.76e-03, …). Those
  steps are not failing to descend; they are being allowed almost no movement because `k` would go
  negative. Two consequences, both of which cost hours today:
  - **A capped step is not a bad step.** Two arms differed in α at rung-2 step 16 (1.000 against 0.579)
    and reached an **identical** residual, 1.271e-01. Attributing the α difference to preconditioner
    quality was wrong. ⚠️ **Those two numbers are NOT a raw-vs-equilibrated pair** — they are `march.log`
    (**petsc** trailing inverse) against `march-20260810-223702.log` (**native + `equilibrate=True`**),
    both under the `dirichlet` k wall BC. The genuine `equilibrate` A/B at step 16 is **0.699 against
    1.000**, under `zerogradient` (`march-20260810-221936.log` / `march-20260811-003915.log`, both at
    ‖R‖ 1.382e-01). The point about a capped step survives either way; the labelling did not.
  - **⚠️ DO NOT GATE `retry.on_alpha` ON `binding_limit == 1`. It was proposed, built and reverted
    the same day (2026-08-10).** More damping shrinks the correction, so it *widens*
    `room = k/|dk|`. The measured evidence is already in this file: the single escalation of an
       entire `bfs3d` march fired at step 51, whose cap was **4.37e-10** — constraint-bound — and it
       was worth **8 steps and 199 s** end to end. Gating `retry.on_alpha` on `binding_limit == 1`
       would have suppressed exactly that one useful escalation.

    **What damping genuinely cannot do is un-pin a cell already ON the boundary**, and that is the
    real rung-2 failure — see the lock-up section immediately below, which is what replaced this.

  **⚠️ (2026-08-10, LATER) THE RUNG-2 DEATH IS A FRACTION-TO-THE-BOUNDARY LOCK-UP, AND β ESCALATION
  IS A SYMPTOM RATHER THAN THE CAUSE.** Read this before acting on the two bullets above; it does not
  contradict them but it re-ranks them, and the "~100 dead steps" is now measured rather than
  estimated. Taken from the archived `march.log` of the native run (banner: `turbulence inverse:
  native`, `equilibrate=True`, `sweeps=4`, `aggressive_levels=1`, `spectral_damping=False`,
  `refresh on cycles 3`, `retry on cycles / alpha 10 / 0.01`, `pc beta floor 0.05`, rung 2 of 3):

  | step | β | cyc | ‖R‖ | a_min | `limit` |
  |---|---|---|---|---|---|
  | 24 | 0.1756 | 5 | 1.123e-01 | 0.694 | 6.94e-01 |
  | 25 | 0.4682 | 2 | 7.567e-02 | 0.004 | 3.76e-03 |
  | 26 | 3.7458 | 0 | 7.316e-02 | 0.000 | 1.00e-05 |
  | 27 | 16.0000 | 0 | 7.316e-02 | 0.000 | 1.95e-06 |
  | 28 | 16.0000 | 0 | 7.316e-02 | 0.000 | 1.95e-08 |
  | … | 16.0000 | 0 | 7.316e-02 | 0.000 | ÷100 every step |
  | 122 | 16.0000 | 0 | 7.316e-02 | 0.000 | 1.95e-196 |

  **The 100× per step IS the rule, not a coincidence.** Taking `α = τ·room` with `τ = 0.99` leaves the
  binding cell at 1% of its `k`, so the next step's room is a hundredth of this one's — forever, for
  as long as the direction keeps pointing at the boundary there. Ninety-six consecutive steps, residual
  frozen to every reported digit, **zero** Krylov cycles, β pinned at its 16.0 ceiling. So:
  - **β is irrelevant from step 27 on, and that is a statement about a PINNED cell, not about damping.**
    Damping is *argued* to widen a fraction-to-the-boundary cap (a smaller correction means more room),
    and that argument is **not measured and cannot be from a log** — only the accepted attempt's cap is
    recorded. ⚠️ An earlier version of this bullet offered "the cap falls only 5.1× across 26→27" as
    evidence; that is the **same cross-step comparison withdrawn above**, used to argue the opposite
    conclusion, and it is struck. The decision rests on the A/B alone. What damping cannot do is recover
    a cell whose `k` is already ~0: the
    room is then small however small the correction gets. **Do not read this as "stop escalating on a
    constraint-bound step"** — that was tried, and it deletes the one escalation on this case that is
    measured to pay (step 51, cap 4.37e-10, 8 steps and 199 s).
  - **Nothing in the ordinary stopping tests can see this.** The state is finite, the residual is finite,
    and the tolerance is simply never reached — so the segment spends its whole budget.
  - **⚠️ Extracting this from a march log needs per-step BLOCKS, not two independent `findall`s.** The
    `limit` line is written only when `binding_limit < 1`, so there are fewer of them than steps (105
    against 108 here) and zipping the two lists silently misaligns them. The first pass at this table
    read step 25's cap as 1.95e-08 instead of 3.76e-03 — a three-step shift, invisible because the
    shifted table is just as smooth. Split on the summary-block delimiter and parse within each block.

  **SHIPPED in response:**
  1. `forward_march(stop_on_limit_stall=3)` (**a new default-on guard**) ends the segment after three
     consecutive steps that are constraint-bound, non-widening, and **changed the residual by less
     than 1e-3 relative, in either direction** (`_limit_collapsing`).

     **⚠️ The residual half of that predicate was wrong twice, and the fix for the first attempt caused
     the second. Do not re-derive it from the failing run alone.**
     - *"the residual did not fall"* — **never fires.** A locked-up step is not a bit-exact no-op: it
       still moves the state by `α·δ`, so at a cap of ~1e-6 the residual genuinely falls, by ~1e-6
       relative, and the counter resets every step. Shipped, and observed doing nothing on a march that
       had otherwise reproduced the lock-up step for step (it reached step 31 and was still going).
     - *"the residual did not fall by 0.1%"* — **fires on a converging rung.** `march-20260810-094635`
       rung 2 steps 19–21 ran caps 0.983 → 0.928 → 0.253 with the residual climbing
       1.293e-01 → 1.335e-01 → 1.489e-01, then recovered to 9.241e-02 and converged to 4.994e-06. A
       *rising* residual means the step did something; a pseudo-transient path is a march in pseudo-time,
       not a descent method. A one-sided rule ends that rung at its worst moment.
     - **The failure is a step that changes NOTHING, so the test is two-sided.** The two populations sit
       orders apart: null steps move the residual by ~1e-6 relative, productive ones by percents.

     **Validated by replaying the predicate over every march log on disk** — `march_stall_replay.py`,
     kept in `validation/bfs3d_openfoam/`, because a march that recovers looks exactly like one that does
     not until it does, so a candidate rule cannot be judged on the run that motivated it. Across 11 logs
     / 22 rungs it fires on exactly the three locked-up rungs (090711 rung 2 at step 21, cutting a
     73-step stall; 130032 rung 2 at step 29, cutting 96) and on **nothing** that went on to converge —
     including the shipped `x_r/h` 8.36 baseline rung. Both cases are pinned as unit tests with the real
     recorded numbers inlined, since the logs themselves are not tracked.

     The cap *narrowing* is the third condition and is what separates a lock-up from a march working
     productively along a constraint. `None` disables the guard.

  The march now fails fast and honestly instead of grinding; **what drives `k` to the boundary in those
  cells is still open**, and the global-scalar-cap design (one cell throttling all 23040) is the thing
  to reconsider.

  **Capturing the step DIRECTION, which no checkpoint holds.** The cap is a property of `delta`, so
  "which cells does it bind on" cannot be answered from a state — a step checkpoint holds where a step
  ended and the inner-iterate dump holds where an inner iteration reached, and neither carries a
  direction. `compare.py` gained `BFS3D_DUMP_STEP_LIMIT=<cap>` (with `BFS3D_DUMP_STEP_LIMIT_KEEP`),
  which wraps the engine's `step_limit` in a `_DumpingStepLimit` — a **frozen dataclass**, not a
  closure, because the limiter rides in a *static* field compared by `__eq__` and a closure there
  recompiles the whole coupled solve on every rebuild. It returns the real cap unchanged, so the march
  it instruments is the march that would have run. Dumps **stop** after `KEEP` rather than wrapping, so
  the first binding event survives the escalation ladder that follows it. `singular_cell_probe.py`
  reads such a dump and reports the exact per-cell room, the binding set, and the binding cells'
  conditioning against the **base rate** for the mesh.

  **⚠️ SUPERSEDED THE SAME DAY — equilibration is a TRIGGER, not the cause. Read the floored-limiter
  entry above before using anything here.** The A/B below is real and reproducible, but it holds **only
  at `positivity_floor = 0`**. With the floor at 1e-8 the flag stops mattering: `equilibrate=True`
  converges all three rungs in 69 steps and `equilibrate=False` in 67, to the **same root in every
  reported digit**. So rescaling never caused the failure — it pushed a cell into the numerically-dead
  zone a few steps earlier than the unscaled arm did, and the global positivity limiter's geometric
  ratchet did the killing. The defect was in the limiter. Keep the case default at `False` (it is free,
  and it is what the converging arms were measured with), but do not describe this flag as deciding
  convergence.

  **⚠️ EQUILIBRATION DECIDES WHETHER THE `bfs3d` NATIVE MARCH CONVERGES — true ONLY at floor 0 (see
  above). Measured 2026-08-11.** An A/B differing in **one flag** was run end to end:

  *Configuration, both arms:* `bfs3d` (23040 cells), 3-rung Reynolds continuation (`N_POINTS=2` →
  Re/100, Re/10, target), `BFS3D_TURBULENCE_INVERSE=native`, `field_split=True`; native trailing
  `cycles=1, sweeps=4, max_coarse=2000, aggressive_levels=1, prolongation_smoothing=none,
  spectral_damping=False`; monolithic smoother **ILU(0) ×4**, `coarse_eq_limit` 2000, plain aggregation,
  reach 3, forward restart 15, `refresh_on_cycles` 3, `retry on cycles / alpha` 10 / 0.01, cycle budget
  42, PC β floor 0.05, stop `(rtol, atol) = (0.0, 1e-5)`, `k` wall BC `zerogradient`.

  | arm | outcome | steps | final ‖R‖ | mid-span `x_r/h` |
  |---|---|---|---|---|
  | `equilibrate=True` (archived `march-20260810-221936.log`) | **locked up, rung 2 step 34** (α 0.001; ‖R‖ frozen from step 35; killed by hand at 39, not solver-terminated) | 39 | frozen 1.257e-02 | — |
  | **`equilibrate=False`** (`march-20260811-003915.log`) | **converged, all 3 rungs, 2081 s** | 77 | **3.586e-06** | **8.361** |

  **It REPRODUCES — the claim does not rest on one run each.** Two further `equilibrate=True` runs,
  `march-20260810-130032.log` and `march-20260810-223702.log` (the latter dump-free), both stall at the
  **identical** ‖R‖ 7.316e-02 with the same β ladder — and they ran under the **`dirichlet`** k wall BC,
  where the pair above ran `zerogradient`. So the flag's effect survives a change of wall closure. The
  march is also deterministic: four runs at identical configuration agree in every printed field.

  **⚠️ The dump wrapper is NOT the confound (checked, not argued).** `221936` ran with
  `BFS3D_DUMP_STEP_LIMIT` and `003915` did not. A dump-ON/dump-OFF control pair at otherwise equal
  settings differs in **zero** content lines over 16 steps, against the equilibrate flag's **77**.
  `_DumpingStepLimit` returns the inner cap unchanged and preserves static-field value equality, so it
  neither alters the step nor forces a retrace.

  **⚠️ The separation is a GROWING PERTURBATION, not a threshold event — an earlier version of this
  entry said "bit-identical for 15 steps, then separated on α alone" and BOTH halves are wrong.** The
  arms differ at **step 1** (p-block residual 6.217e-07 vs 6.272e-07) and accumulate 77 differing
  content lines before step 17. They agree only to the **4th printed digit of the summary row**, which
  is insensitive to the diverging component: at step 16 the p-block residuals differ by **19×**
  (1.545e-07 vs 2.938e-06) while the reported ‖R‖ matches to four digits. Step 16 differs in the cycle
  count (5 vs 4) and the retry flag as well as α (1.000 vs 0.699 `L`). The correct statement is: **a
  small preconditioner-dependent difference is present from the first step and amplifies for fifteen
  steps until it crosses the positivity limiter**, after which the clipped α trips `retry.on_alpha` →
  β escalation → more clipping → the 16.0 ceiling at zero cycles. Do not describe this as a clean
  bifurcation; that framing implies a threshold the flag crosses cleanly and the data do not show one.

  **⚠️ `x_r/h` 8.361 against OpenFOAM's 7.243 is TWO GRID STATIONS, not a 15 % discrepancy.**
  `reattachment_length` returns the `x` of the last reversed wall cell — a grid station — and the
  stations near reattachment are `… 6.728, 7.243, 7.787, 8.361, 8.966 …`, spacing ~0.55 h. The
  comparison to the shipped PETSc run (`march.log`: 58 steps, 282 cycles, `x_r/h` 8.3611) is **not
  like-for-like**: it differs in the k wall BC (`dirichlet`) as well as the inverse (`petsc`), and it
  lands on the identical station — i.e. the metric did not resolve either change. Quote the sub-cell
  interpolated crossing from `wall_layer_comparison.py` if this number has to bear weight.
  - **The positivity limiter is NOT the failure — failing to recover from it is.** The converged arm hits
    the same constraint repeatedly (`L` at steps 24, 26, 27, 30, 32, 35, 39, 40, 41, 49, 51, 53, 54, 56,
    57, 60, 61, **including α 0.000 at step 61**) and recovers from every one, α returning to 1.000.
  - **A constraint-free α collapse appeared, and nothing reacts to it.** Step 68: α 0.031 with **no `L`
    flag** and 15 cycles (the run's highest) — a poor *direction*, not a clipped step. α 0.031 is above
    `retry.on_alpha` 0.01, no `RefreshTrigger` reads α or `binding_limit`, and this bundle sets
    `beta_rel_change=inf`, so no refresh fires. It cost a few steps here, not the run, but it is the
    first live evidence that the refresh gap is a real cost.
  - **⚠️ ONE RUN EACH, and one instrumentation difference:** the archived equilibrated arm ran with
    `BFS3D_DUMP_STEP_LIMIT=0.05/12`, the converged arm with the dumps off. The dump wrapper returns the
    real cap unchanged by construction, so it *should* be neutral, but it is not a matched pair.
  - **The stated reason for the `equilibrate=True` default no longer exists.** `native_nodal_inverse`
    defaults it on because "the per-cell block solve is not otherwise safe" — raw, 4 of 23040 cell blocks
    were flagged singular. That count came from the **Frobenius** guard (`|det| < 1e-12·‖B‖_F`), which is
    not invariant under row scaling and is **the guard that was found wrong and replaced** by the
    Hadamard row-norm bound, which is invariant. The default is **unchanged pending a decision**; flipping
    it is a shipped-default change.

  **✅ THE FLOOR RESCUES A CONFIGURATION THAT PREVIOUSLY DIED — which is the real case for it, not the
  step count (measured 2026-08-11).** `equilibrate=True` had failed on rung 2 **twice**, under both wall
  closures (`march-20260810-221936.log`, ‖R‖ frozen at 1.257e-02 from step 34; `march-20260810-223702.log`,
  frozen at 7.316e-02 from step 26). Re-run with `positivity_floor=1e-8` and **nothing else changed**, it
  converges all three rungs.

  | arm | `equilibrate` | floor | outcome | steps | wall | final ‖R‖ | `x_r/h` | escalations |
  |---|---|---|---|---|---|---|---|---|
  | `221936` | True | 0 | **died, rung 2 step 34** | 39 | — | frozen 1.257e-02 | — | 13 |
  | `223702` | True | 0 | **died, rung 2 step 26** | — | — | frozen 7.316e-02 | — | — |
  | `003915` | False | 0 | converged | 77 | 2081 s | 3.586e-06 | 8.3611 | 8 |
  | — | False | **1e-8** | converged | **67** | 2133 s | 3.586e-06 | 8.3611 | 4 |
  | — | **True** | **1e-8** | **converged** | **69** | 2188 s | 3.586e-06 | 8.3611 | 5 |

  **The chain the floor breaks, visible step by step.** `221936` clipped at rung-2 step 16 (α 0.699,
  `L`), which tripped `retry.on_alpha`, which escalated β, which clipped harder, up to the 16.0 ceiling
  at zero cycles. With the floor, that **same step 16 takes α 1.000 with no flag**, matching the
  unscaled arm exactly — the binding cell there was numerically dead, and buying it out removes the
  first link. From step 22 on, the floored equilibrated and floored unscaled arms run the *same*
  trajectory field for field.

  **What it does NOT fix, confirmed on all three converging arms.** The rung-3 hard point reproduces
  bit-for-bit regardless of floor or flag: α 0.010 at 15 cycles, then α 0.000, then β 0.0293 → 0.9364
  (**32×**) across two steps and six steps of SER walking it back, then a constraint-free α 0.031 with
  **no `L` flag**. That is the bad-direction mode, and the escalation ladder's **overshoot** is now the
  clearest remaining cost on this case.

  **Where the wall time goes** (equilibrated + floored run, per-step sums, one run on a shared machine —
  proportions not a benchmark): preconditioner **19 %** (304 s of 1561 s through step 48), of which the
  coloured Jacobian probe is 243 s and the refactor 56 s; 17 of 48 steps carried a refresh and only 2
  were scheduled — the rest fired on the reactive 3-cycle rule. Rung 3 costs ~56 s/step against ~25–30 s
  on rungs 1–2. The four most expensive steps are all first-of-rung (115–171 s), carrying a compilation
  on top of the PC build.

  **⚠️ THE α COLLAPSES ARE TWO DIFFERENT FAILURE MODES, and only one is the limiter's fault — measured
  2026-08-11 by a floored-limiter A/B.** `PositiveBlockLimit` gained a `floor`: the room becomes
  `(phi_i + floor)/|delta_i|`, so an entry that is numerically zero stops setting the step for all of
  them. `floor=0` (the library default) is bit-identical to the plain rule, and the limiter is inactive
  at a root for **any** floor (there `delta = 0`), which is what keeps it out of the converged state and
  therefore out of the implicit-function-theorem adjoint — the adjoint never sees it in any case, since
  `_implicit_solve_bwd` reads only `jax.vjp(residual_fn, phi_star)` and a transpose solve, never
  `forward_step_fn`.

  *Configuration, both arms:* `bfs3d`, native trailing inverse with **`equilibrate=False`**, `k` wall BC
  `zerogradient`, 3-rung Reynolds continuation (`N_POINTS=2`), ILU(0) ×4, `coarse_eq_limit` 2000, plain
  aggregation, reach 3, forward restart 15, `refresh_on_cycles` 3, `retry on cycles / alpha` 10 / 0.01,
  PC β floor 0.05, stop `(0.0, 1e-5)`. Floor **1e-8**, calibrated by replaying the recorded clips
  (`positivity_floor_calibration.py`).

  | | floor 0 | **floor 1e-8** |
  |---|---|---|
  | steps | 77 | **67** |
  | escalations (all `alpha`-triggered) | 8 | **4** |
  | final ‖R‖ | 3.586e-06 | 3.586e-06 |
  | mid-span `x_r/h` | 8.3611 | 8.3611 |
  | `ux`/`uy`/`uz` rel-L2 | 0.0616 / 0.0072 / 0.0061 | identical |
  | ν_t peak | 150.1071 | 150.1071 |
  | wall | 2081 s | 2133 s |

  **The root is unchanged in every reported digit, and the wall clock is a WASH — do not sell this as a
  speed-up.** The floored arm's rung-2 steps run at low β costing 4–9 cycles where the unfloored arm
  escalated into cheap 2-cycle solves, and that cancels the ten steps. (One run each, and this case has
  no measured run-to-run spread, so 2.5% is not resolvable.)

  - **Mode 1, the RATCHET — the floor eliminates it.** A numerically-dead cell with a *tiny* correction:
    `tau` leaves it at `1 - tau` of its value, so the cap falls ×100 per step while its `delta` never
    moves. Rung 2 went 34 steps → 24, three escalations → one, and the clips stopped binding at all by
    step 33.
  - **Mode 2, a BAD DIRECTION — the floor cannot and should not touch it.** Rung 3 steps 50–51
    reproduce bit-for-bit with the floor in place (α 0.000, ‖R‖ 6.191e-03 vs 6.192e-03), and step 58's
    α 0.031 carries **no `L` flag at all**. For α to collapse with a 1e-8 floor the binding cell needs
    `|delta_k| >> 1e-8` — a large correction on a live cell, not a tiny `k`. No physically-sized floor
    rescues a step whose correction dwarfs the field; the escalation ladder is the right response, and
    the open question there is its **32× overshoot** (β 0.0293 → 0.9364 across two steps, then six
    steps of SER walking it back), not its trigger.
  - **Healthy clips survive, which the dumps could NOT show** (they were written only below cap 0.05).
    Steps 7 and 8 are preserved exactly; steps 9 and 44 shift by 0.1–0.3 % — exactly the `floor/k`
    perturbation the softened form predicts for a live cell. That is also how to read a clip: a live
    cell moves by `floor/k`, a dead one is bought out entirely.
  - **⚠️ Choose the floor by REPLAY, not by guess — the dead cells are a graded population, not one
    outlier.** Exempting the worst promotes the next: worst recorded cap 1.05e-09 unfloored, 1.6e-02 at
    a 1e-12 floor, 6.9e-02 at 1e-10, ~0.35 at 1e-08. An earlier estimate of "1e-12 should do it" was
    wrong by four orders. At 1e-06 the binding cell is a live one (`k` 4.7e-03, cap 0.84), so 1e-08
    keeps two orders of margin below anything physical here (inlet `k` 0.375, mesh median ~3e-2).
  - **Safe only because every consumer of the solved `k` clamps at zero.** The limiter's original
    justification — a negative `k` reaching a bare `sqrt` and poisoning the residual — no longer holds:
    `f1`, `f2`, `eddy_viscosity`, the production cap and the wall closures all clamp, and at `k < 0` the
    destruction term runs on the `k`-independent viscous ω branch, whose sign pushes `k` back up.
    **Re-check that before floating this limiter on another field.**

  **✅ BUILT AND MEASURED (2026-08-11): the per-cell PROJECTION removes the cap's failure mode entirely
  and does NOT make the case faster. Read this before proposing anything else aimed at the limiter.**
  `PositiveBlockProjection` clips each cell's own correction, `delta_i <- max(delta_i, -tau(k_i + floor))`,
  instead of scaling the whole step by the worst cell; applied before the cap, which then computes
  exactly `1`. It answers Mode 1 structurally rather than by decades of headroom, since a floored *cap*
  only postpones the ratchet — `(k_new + floor) = (1 - tau)(k + floor)` per capped step, whatever the
  floor.

  *Configuration, both arms:* `bfs3d`, native trailing inverse, `zerogradient` k wall, floor 1e-08,
  `equilibrate=False`, `refresh_on_cycles` 3, ILU(0) ×4, trailing sweeps 1, `coarse_eq_limit` 2000,
  restart 15, two rungs. **Both predate the probe/memory work merged in #188/#189**, so their wall
  clocks are not comparable to anything measured after it.

  | | projection | cap only |
  |---|---|---|
  | steps | 65 | 67 |
  | **Krylov cycles** | **329** | **329** |
  | wall | 2021 s | 2124 s |
  | positivity-limited (`L`) steps | **0** | 22 |
  | escalations | 5 | 4 |
  | mid-span `x_r/h` | 8.36 | 8.36 |

  **The cycle count is IDENTICAL, which is the number to read.** Wall differs by 4.8 % between runs
  taken at different machine loads and this case has no measured run-to-run spread; the linear-solve
  work did not move. **Do not cite this as a speed-up.**

  **Why it bought nothing: the cap was one DOOR into the cascade, not its cause.** The mechanism worked
  exactly as designed — the cap never binds once, against 22 times — and the same cascade reappeared
  through the line search's descent test instead. Rung 3 steps 53–55: α 0.000 twice with **no `L`
  flag**, β driven 0.0548 → 0.8769 (**16×**), cured at 0.8769, then six steps walking back to 0.0770 —
  the walk-back law again, `log(0.8769/0.0770)/log(1.5)` = **6.00** predicted, **6** observed. This is
  Mode 2 above, and it confirms that entry empirically: no treatment of the *limiter* reaches it.

  **So the lever is the RETURN, not the trigger and not the constraint.** Three separate attacks on the
  trigger side are now measured or refuted — gating `retry.on_alpha` on `binding_limit`, raising the
  floor, and removing the cap's global coupling altogether — and none moved the cycle count.

  **Keep the projection for ROBUSTNESS, not speed.** It eliminates the positivity lock-up outright (the
  mode that killed the 77-step march and the β = 16 runaways), and it is off by default and
  byte-identical off (`positivity_projection=False`, `BFS3D_K_POSITIVITY_PROJECTION` unset). It is not
  *dominated* — it removes a failure mode nothing else does — so it is not a deletion candidate, but it
  must not be sold as a performance feature.

  **⚠️ REFUTED — "rescaling promotes collapsed-`k` rows and inflates their corrections" is FALSE. Do not
  re-propose it (measured 2026-08-11, `k_row_scale_probe.py`).** The proposed explanation for why the
  `equilibrate` flag changes the step length was: symmetric rescaling divides row `i` by `sqrt(A_ii)`; a
  cell whose `k` has collapsed has a tiny diagonal there; so rescaling promotes that row to unit weight
  and un-scaling inflates the correction in exactly the cells the cap (a **minimum** over cells) is
  decided by. **The first clause is false**, so the rest cannot hold.

  *Configuration:* `bfs3d`, states `step-limit-04`/`-11` (both from `march-20260810-223702.log`: native
  trailing inverse, `equilibrate=True`, **`k` wall BC `dirichlet`**, rung 2 = Re/10, β 0.468 and 4).
  Exact `∂R_k/∂k` per cell by one-hot Jacobian-vector product — no materialization — 40 cells per decile
  of `k` plus the three cells the limiter is observed to bind on.

  | decile of `k` | median `k` | median `∂R_k/∂k` | scale `1/sqrt(|diag|)` |
  |---|---|---|---|
  | 0 | 8.974e-13 | 3.543e-05 | 168 |
  | 3 | 1.702e-03 | 1.246e-05 | 283 |
  | 9 | 4.893e-01 | 3.176e-05 | 177 |

  Across **twelve orders of magnitude in `k`** the diagonal moves ~1.1× and the scale 1.7×, peaking in
  the MIDDLE deciles — lowest-decile/highest-decile scale is **0.95×**. The binding cells sit **below**
  the median scale (12800 at 0.89×, 3181 at 0.51×, 22400 at 0.89×), i.e. rescaling mildly *demotes*
  them. The reason is structural: `∂R_k/∂k` is set by the destruction `β* ω V`, face transport and the
  pseudo-transient shift, **none of which vanish as `k -> 0`** — a collapsing `k` does not weaken its own
  equation.

  **The within-cell control is stronger than the table and removes the last confound.** Cell 12800's `k`
  differs by **six orders of magnitude** between the two dumps (3.082e-16 at β 0.468, 3.082e-22 at β 4)
  and its diagonal is **identical to four digits in both, 2.864e-05**. Same cell, same position, `k`
  varying a millionfold, `∂R_k/∂k` unchanged — so the flat decile trend is not an artifact of which
  cells populate the low deciles. (Cell 22400 reports the same 2.864e-05, so the two share a structural
  situation; cell 3181 differs at 8.753e-05.)

  **⚠️ The probe that first "measured" this question was structurally incapable of answering it — check
  for this failure mode before trusting any arm comparison.** It preconditioned with
  `CoupledShiftPolicy.make_preconditioner`, which is block-SIMPLE on `[u,v,w,p]` plus (with
  `method=None`) **identity on `k` and `ω`**. `equilibrate` lives only inside the engine's
  `FieldSplitAmgPreconditioner` (via `trailing_inverse`), so **both arms ran identical code** and
  returned identical corrections — reported as "no effect". Two ~2 GB Jacobians were built and discarded
  to produce it. The faithfulness gate could not catch it: the gate forms `operator(δ) − b`, which
  contains **no preconditioner at all**. *An A/B needs an assertion that the arms actually differ
  (`assert not array_equal(...)`), not only a gate that the system is right.*

  **⚠️ A Euclidean gate cannot establish solve accuracy on this system.** That probe read its
  8.4e-07 **2-norm** gate as "the march over-delivers by orders of magnitude against its 0.3 stop". The
  coupled Euclidean residual is ~100 % `ω` — the reason the row-scaled stop exists — so 8.4e-07 there is
  consistent with the `k` and velocity rows sitting at 0.1–0.3. Report the gate **per block**, in the
  measure the solve actually stopped in.

  **What the dumps DO show, and it points away from the preconditioner.** At fixed β, `δk` at the
  binding cell is unchanged to 7 digits between dumps while `k` falls ×100 per clipped step —
  `3.0816e-16 -> 3.0816e-22`, mantissa preserved, exactly `(1 − τ)` at `τ = 0.99`. The correction is not
  driving the collapse; **the limiter is ratcheting one cell toward zero and the global `min` lets that
  one cell of 23040 throttle the march.** The binding component is also ~13 orders below its block's
  norm — order 40× the round-off floor — so *which* arm's correction clips is likely not a reproducible
  quantity at any tolerance. The live target is the limiter's design (a `k`-relative floor, `τ` tapering,
  or a per-cell rather than global cap), not the hierarchy.

  **Equilibration: KEEP it, for conditioning, and stop expecting it to fix anything else.**
  - It improves per-cell block conditioning by orders of magnitude in the median, which is why it is kept.
  (Measured 2026-08-10 with `cell_block_scaling.py`; state, β and bundle not recorded, so the figures that
  quantified it are deleted — re-measure before relying on the size of the gain.)
  - It **cannot** change whether a cell block is singular. `det B̂ = det B / |b₁₁b₂₂|`, so
    `det B̂ = 0 ⟺ det B = 0`, and the coupling ratio `|a_kω a_ωk| / |a_kk a_ωω|` measures **identical**
    raw and rescaled (7.369e-04 both). Using it as the fix for the guard was wrong from the start;
    the "4 singular → 2 singular" reading that seemed to support it compared *different states*.
  - It does **not** help the worst cells: on `bfs3d` `state-00057`, symmetric `D B D` leaves cond at 1.22e12
  because it moves the imbalance from the rows into the subdiagonal (`[[1, 1.5e-9], [-1.1e6, 1]]`). The
  next two bullets are the same worst cell at that state.
  - **Independent row/column scaling would**: row-equilibrating the cell block gives cond 1.68e4,
    two-sided gives **2.41**, and both are *exact* rebracketings of `B⁻¹`. Unnecessary at float64
    and 2×2 (relative error is already 2.5e-16), but it becomes mandatory in **float32** — which is
    the GPU case this whole exercise is for.
  - The deeper issue is upstream: the k row has norm 8.8e-06 and the ω row 1.4e+03, so the
    **equations** are eight orders apart before any preconditioner sees them. That is what
    `RowScaledNorm` already recognizes in the convergence measure. Fixing the residual's row scaling
    would help the smoother, the coarsening and any factorization at once.

  **`‖B⁻¹‖ = 9.5e8` (`bfs3d` `state-00057`, symmetrically equilibrated) is not an error and no rescaling
  removes it.** The block is essentially
  lower-triangular (`∂R_k/∂ω ≈ 1.7e-12`), so ω is slaved to k there and a k correction legitimately
  produces one ~1e9 times larger in ω. Whether a *cell-local* smoother should apply that is the real
  open question, and it is the same "ω is not locally determined" theme as the Vanka campaign.

  **The tree is verified NEUTRAL on the shipped path.** A control march (PETSc ILU(0)×1) on all of
  this measured **58 steps / 282 cycles / final ‖R‖ 9.588e-06 / mid-span `x_r/h` 8.36** — identical
  in every reported digit to the recorded baseline. Wall was 1809 s against 1636 s — **10.6%, unexplained**, and
  on one run each. ⚠️ This case has **no measured run-to-run spread** to judge that against; the "~2%" it
  was previously compared to was a remembered figure with no configuration and has been deleted. The
  neutrality conclusion rests on the cycle count and the reported digits, not on the wall.

  ⚠️ **SIX OF THESE HARNESSES WERE DEAD ON ARRIVAL UNTIL 2026-08-14, AND NOTHING ANNOUNCED IT.** Each
  opened with `march_beta, _, description = STATES[name]` — a **positional** unpack of a record that has
  since grown to five fields — so every one of them raised `ValueError: too many values to unpack` before
  reaching its first line of work: `trailing_hierarchy_sweep.py`, `cell_block_scaling.py`,
  `trailing_block_conditioning.py`, `turbulence_smoother_sweep.py`, `zero_pattern_pivots.py` and
  `field_coupling.py`. All six now read the fields **by name**, which is what makes them survive the next
  field. **The lesson is about the keep-the-harness rule rather than about these six**: a harness kept in
  the repository is only re-adjudicable if it still *runs*, and nothing in the test suite exercises these,
  so a shared record can grow a field and silently retire the entire probe fleet. Every measurement in
  this file whose harness is named above was taken before that breakage and is unaffected — but any
  attempt to re-take one between the field's addition and this fix would have failed at startup.

  **Harnesses kept (all in `validation/bfs3d_openfoam/`):** `trailing_hierarchy_sweep.py` (the block
  alone, every arm), `cell_block_scaling.py` (per-cell conditioning, raw vs equilibrated),
  `singular_cell_probe.py` (which cells, from a checkpoint **or** a dumped block),
  `k_row_scale_probe.py` (exact `∂R_k/∂k` per cell by one-hot Jacobian-vector product, stratified by
  `k` — the harness that refuted the row-promotion explanation above, and the cheap way to ask whether
  any field's rows are what a rescaling promotes).
  `compare.py` gained `BFS3D_TURBULENCE_INVERSE`, `BFS3D_NATIVE_EQUILIBRATE`,
  `BFS3D_DUMP_TRAILING_BLOCK`, a `pbjacobi1` (undamped) smoother arm, march-log **archival**, and a
  banner that records the inverse and all its settings — **including the three knobs that install
  wrappers or change retention (`CHECKPOINT_KEEP`, `INNER_DUMP_ABOVE`, `DUMP_TRAILING_BLOCK`), which it
  used to read but never print, so two differently-configured runs could produce identical banners.**

  **⚠️ FOUR METHODOLOGICAL TRAPS, each of which produced a wrong write-up today:**
  1. **Probe the state the failure happens at.** The refusal fires from a *mid-step* refresh; step
     checkpoints and the inner-iterate dump both miss it (the inner observer writes only after an
     iteration succeeds). Three capture runs were wasted before dumping the operator *before* the
     build, which needs no state and no shift pairing — and even that missed twice, first by
     wrapping only the factory when the refresh goes through `refactor_block`.
  2. **Pair the operator with the right β.** Probing state-N with state-N's β when the failing
     refresh uses state-N+1's is the recorded trap; sweep β instead.
  3. **Never quote an arm at one smoother-sweep count.** The standard prolongator was recorded as
     "void" from its 4-sweep numbers; at 8 it is the best native arm on the scalar problem. The rule
     cuts both ways — on the flow saddle it fails at 4 sweeps and fails *worse* at 8, which is what
     distinguishes an under-smoothed arm from an amplified one.
  4. **A block-alone probe ties where a march separates.** Raw and equilibrated are both 2 cycles on
     the block and behave differently in a march.

  **✅ ANSWERED: the cap binds on ONE cell, it is a step-corner cell whose `k` is already numerically
  zero, and it is NOT one of the ill-conditioned ones.** Measured from twelve `BFS3D_DUMP_STEP_LIMIT`
  dumps taken on the native march at rung 2 (`equilibrate=True`, `sweeps=4`, `aggressive_levels=1`,
  `spectral_damping=False`, `refresh on cycles 3`, `retry on cycles / alpha 10 / 0.01`, PC β floor
  0.05), the operator swept over β 0 … 0.4. The probe reproduces the recorded cap exactly
  (3.761982e-03 both ways), so the capture is reading the quantity the march acted on.

  | dump | cap | cell | room | `k` | `dk` | next cell | next room |
  |---|---|---|---|---|---|---|---|
  | 04 | 3.7620e-03 | **12800** | 3.80e-03 | 3.08e-16 | −8.11e-14 | 22400 | 2.60e-01 |
  | 05 | 1.4517e-04 | **12800** | 1.47e-04 | 3.08e-18 | −2.10e-14 | 12840 | 2.09e+00 |
  | 08 | 1.0516e-07 | **12800** | 1.06e-07 | 3.08e-20 | −2.90e-13 | 12840 | 1.12e-01 |
  | 11 | 1.0516e-09 | **12800** | 1.06e-09 | 3.08e-22 | −2.90e-13 | 12840 | 1.12e-01 |

  - **One cell out of 23040 throttles the whole march,** and it is not a crowd: at dump 04 exactly
    **one** cell sits within 100× of the tightest room, the runner-up 68× behind; by dump 11 the gap is
    **eight orders**. Cell 12800 owns ten of the twelve dumps and every one from step 25 on.
  - **`k` there ratchets by exactly 100× per step** (the per-step evidence is the checkpoint mantissa
    below; these dumps are non-consecutive) — 3.08e-16 → 3.08e-18 → 3.08e-20 → 3.08e-22 — which
    is `1 − τ` at `τ = 0.99`, the same factor the cap collapses by. Against a mesh median `k` of
    **2.97e-02**, so the binding cell's `k` is ~20 orders below typical. It is not small; it is zero.
  - **`dk` stays ~1e-13 throughout while `k` collapses.** ⚠️ **The inference drawn from this — "the
    cell's `k` equation has no root at `k ≥ 0`" — is WRONG, and was refuted the same day by direct
    measurement. It is the opposite.** Holding every other field at the `step-limit-11` iterate and
    scanning `R_k[12800]` over `k_P` gives a straight line to seven digits,
    `R_k = A·k − b` with `A = 2.8740e-06`, `b = 5.7236e-20`, so the **root is `k* = +1.99e-14`,
    strictly positive** — and the iterate sits **eight decades BELOW its own root**. Newton should be
    pushing `k` *up*.

    Production, destruction and the `Dirichlet(0)` wall term all vanish linearly at `k = 0`, so `k = 0`
    solves the row exactly when the neighbours are 0. (⚠️ An earlier version called the corner `k` system
    "**homogeneous and an M-matrix**" — the M-matrix half is **false**: `A_kk` carries **14 non-positive
    diagonals**, min −9.03e-07.)

    **Why the returned correction is wrong there, corrected — it is Krylov truncation, NOT roundoff.**
    An adversarial re-derivation (independent colour-recovery of `A_kk`, verified row-wise against a
    `jax.vjp` to 5e-16) settles the mechanism:
    - `R_k[12800] = −5.72e-20` is **deterministic and 13 orders ABOVE its own row's floating-point
      floor** (~1.3e-33). "Roundoff floor" was the wrong description.
    - What is true is stronger: `|R_row| / ‖R_k‖₂ = 2.94e-16 ≈ machine epsilon`, and
      `|R_row| / ‖R‖₂ = 2.15e-22`. **Any Krylov solve stopping on a relative-residual test would need
      ~1e-22 relative accuracy in float64 to resolve this row — unreachable in principle, not merely
      under-solved.** Measured: `‖A_kk·dk − rhs_k‖/‖rhs_k‖ = 1.0055`, i.e. the returned correction does
      not reduce the k-block linear residual at all, and dumps 08/09/10 sit at a **bit-identical state**
      yet return `dk[12800]` of −2.901e-13, −8.825e-14, −1.567e-14, an **18× spread**.
    - The returned `dk` also provably violates its own row for *every* admissible shift: the row demands
      `βD·(k_ref + 2.901e-13) = −3.958e-19`, and the left side is ≥ 0 while the right is < 0.
    - ⚠️ **"14× larger than the exact local step" was wrong** — that froze the neighbours. Given the
      dumped neighbours the row-consistent `dk` is −1.52e-13, still negative; it is the **exact k-BLOCK
      solve** that comes out positive, at `+3.90e-10` (β = 0) through `+1.90e-15` (β = 10), i.e.
      **positive at every β from 0 to 10** while the dumped value is negative and 3–5 orders off.

    `positive_block_limit` then computes a purely **relative** room `0.99·k/|dk|` and lets that cell veto
    the global step.
    - **⚠️ THE MOST USEFUL LEAD OF THE LOT: at β ≥ 0.5 an exact k-block solve produces NO binding cell at
      all (cap = 1.0).** Only at β ≤ 0.05 does an exact solve give a tight cap, and then on a *different*
      cell (7.6e-14 at cell 2679). So the cap is **not intrinsic to the `k` equations at a healthy
      shift** — it is a low-β-plus-inexact-solve artifact, which points back at the low-β conditioning
      wall rather than at the closure.

    **The ratchet is proved by the mantissa.** `k[12800]` reads `3.0816e-22` at `step-limit-11` and
    `3.0816e-208 / e-210 / e-212` at checkpoints 120 / 121 / 122 — **the same five digits, 190 decades
    apart**, i.e. ninety-five successive ×0.01 cuts, while `u/v/w/p` report a relative change of
    exactly 0.
  - **It is not one cell, either: 1876 of 23040 cells sit below `k = 1e-6`** at this state, against an
    OpenFOAM global minimum of 1.672e-6. ⚠️ **BOTH HALVES OF THAT COMPARISON WERE WRONG, AND THE
    CONVERGED ROOT HAS NO SUCH DEFICIT — see the resolution below.**
  - **Geometry: it is a step-corner cell, and its mirror is the runner-up.** 12800 sits at
    `(0.0007, −0.0099, 0.0009)` and 22400 at `(0.0007, −0.0099, 0.0391)` — bottom wall (`y ≈ −h`),
    immediately behind the step (`x ≈ 0`), against each side wall (span `4h = 0.04`). The stagnant
    bottom/side-wall corner, i.e. the same corner-separation region that makes the full-span
    reattachment (16.14 here) disagree with the mid-span one (5.34). ω there is 4.01e+05, a wall value.
  - **The tightness ranking tracks the wall-face count — on n = 4 cells, one dump, one state.** Counted
    off `face_patches`, the four tightest cells carry **3, 3, 2, 1** no-slip faces out of six:

    | cell | room (dump 04) | boundary faces |
    |---|---|---|
    | 12800 | 3.80e-03 | `lowerWall`, `lowerWall`, `sideWalls` |
    | 22400 | 2.60e-01 | `lowerWall`, `lowerWall`, `sideWalls` |
    | 12840 | 1.38e+00 | `lowerWall`, `sideWalls` |
    | 13129 | 2.17e+00 | `sideWalls` |

    The two cells that own the cap are **trihedral wall corners** — half of every face is a no-slip
    wall, the two `lowerWall` faces being the floor and the vertical step face. **Hypothesis, not yet
    measured:** the near-wall `k` closure is per-wall-cell, and its production is area-averaged over a
    cell's wall faces while the destruction `β*kω_wall` is not obviously averaged the same way — so a
    three-wall-face cell could take up to 3× the destruction against one cell's worth of production.
    **❌ REFUTED the same day by reading both sources** — our reduction is already OpenFOAM's:
    `wall_cells` is a `jnp.unique`, so a cell appears once however many wall faces it has, and
    `wall_shear_rate` area-averages `Σ|S_f|r_f / Σ|S_f|` over them, which is exactly
    `patchFieldsToWallCellField` in OpenFOAM 13's `wallCellWallFunctionFvPatchScalarField`. Nothing is
    summed per face. The ranking most likely reads **stagnation** — three wall faces means the deepest
    dead zone and so the least production — not double counting. See `.claude/rules/turbulence.md` for
    the two differences that are real, of which one bears directly on the lever below.
  - **NOT the ill-conditioned cells — the standing hypothesis is refuted at this state.** **Zero**
    singular blocks at every β from 0 to 0.4. The binding cells run cond 3.6e5 … 2.9e7, ranks
    1088–2165 of 23040, and **none** of them is among the twelve worst-conditioned. `cond > 1e6`
    catches 50% of them against an 8.7% base rate — mildly enriched — but `cond > 1e9` catches none,
    and the separately characterised cond ~1e12 / ‖B⁻¹‖ ~9.5e8 cells are absent entirely (the worst
    here is 2.9e7 / 6.6e6). **Caveat on comparing those two numbers:** the 1e12 figure was measured on
    `state-00057` under *symmetric equilibration*, and this is a different state, raw — so this
    refutes the coincidence at this state rather than retiring the 1e12 finding.

  **✅ RESOLVED (2026-08-10, by re-running the shipped march end to end): the `k` trough is a REYNOLDS-
  CONTINUATION TRANSIENT, and the converged root does not have it.** At the converged target-Re root
  (step 58, `R` 9.588e-06, `x_r/h` 8.36) there are **ZERO** cells below `k = 1e-6`: `k_min` 1.2997e-05,
  median 0.7017 against OpenFOAM's 0.7468, a ratio of **0.98**. Cells below 1e-6 go 190 (step 37) → 130
  → **0** (step 46) → 0 at convergence.
  - **Both halves of the "1876 vs 1.672e-6" comparison were wrong.** It is **cross-Reynolds** —
    `state-00122` is rung 2 (ν = 1e-4), the OpenFOAM field is target Re — and 1.672e-6 is the **steady**
    OpenFOAM run's minimum, the run this case's own docstring rules out as a reference. The valid
    transient reference has `k_min` **6.2247e-03**.
  - **The mechanism is the continuation's own doing.** `y* = β*^0.25 √k d/ν` scales as **1/ν**, so at the
    Re/100 anchor `wall_function_weight = tanh((y*/y*_lam)⁴)` is ~1e-7 — **the log-layer wall production
    is switched off even at fully turbulent `k`** — while `ω_wall = 6ν/(β₁d²)` is **100×** larger, so
    `β*kω` destruction is 100× the target's. Homogeneous, linear, destruction-dominated ⇒ `k` decays.
    Raising Re reverses both: the weight median goes 5.6e-20 (rung-1 root) → 1.9e-10 → **1.000** (target).
  - The rung-1 root is **correctly** laminar (`ν_t/ν` median 1.7e-5 at `Re_h = 100`), and **1732 of the
    1876 cells (92%) are inherited from it**.
  - **Population is pre-stall, depth is stall-caused**: `#k<1e-6` is 1876 by dump 04 and frozen there for
    all 97 remaining steps, while `k[12800]` falls 3.08e-16 → 3.08e-212 over the same span.
  - **The stalled arm is `BFS3D_TURBULENCE_INVERSE=native`; the default `petsc` arm walks the same trough
    and recovers** — agreeing with the exact-solve finding that at β ≥ 0.5 there is no binding cell.
  - ❌ **REFUTED — the initialization.** `hybrid_initialize` gives a **uniform** `k = 0.2489585`, zero
    cells below 1e-6.
  - ❌ **REFUTED — the "ω is exactly 10×, still at its 60ν seed" lead.** `ω[12800] = 400910.4` **is**
    `6ν/(β₁d²)` at **rung 2's own** ν = 1e-4 to 16 digits (all 4490 wall cells within 4.8e-5). The three
    rungs give 4.0091e6 / 4.0091e5 / 4.0091e4 — the "10×" compared a rung-2 state against the
    **target-Re** formula. Those rows are converged; the `60ν` seed was replaced by `omega_wall` in
    `c324021` (2026-07-23), well before these checkpoints.
  - **The sustaining/ambient source has no motivating evidence left**: it targets a defect absent from
    the converged root. The defensible levers are the absolute floor in the limiter, and not anchoring
    the ladder below the Reynolds number at which the wall function turns itself on.
  - **Genuinely open, and now the real physics question — but BOTH numbers needed a definition before
    they meant anything (harness: `validation/bfs3d_openfoam/wall_layer_comparison.py`).**
    - "**First wall layer**" spans a 4× range of defensible meanings, and the choice matters more than
      the discrepancy: **all wall-adjacent (4490 cells) is 1.054**, the **finest layer (1600) is 0.459**,
      the **floor behind the step (640) is 0.342**. The recorded 0.164/0.46× is the *finest layer*. Side
      walls, 64% of wall-adjacent cells, agree at **1.113** and do not participate.
    - `x_r/h` **7.24 and 8.36 are both grid stations, two apart.** The floor's stations near
      reattachment are `… 6.728, 7.243, 7.787, 8.361, 8.966 …`, local spacing **0.375–0.749 h**, so the
      metric's resolution is about half a step height and "15%" is a two-cell offset. A sub-cell
      interpolated crossing puts the reference at **~7.6**, i.e. the quoted 7.24 carries a systematic
      −0.36 h truncation, and the interior spanwise columns scatter by **sd 0.13 h**. A defensible
      reference is **x_r/h ≈ 7.6 ± 0.3**; the gap survives it (~1.0 h, ~13%) but is smaller than quoted.
    - **The wall closure is NOT the cause.** The `Dirichlet(0)` vs `kqRWallFunction` (zero-gradient)
      difference is real but worth only ~7.4% of local destruction (k ×0.93–0.98); `nut_wall` is
      *algebraically identical* to `nutkWallFunction`; every verified wall-closure difference multiplies
      to **~0.87–0.93, not 0.46**. The wall-cell `k` budget is **transport-dominated**
      (|transport|/production ≈ 1.16), so that `k` is inherited, not locally made.
    - **Where it is inherited from, and this is the causally clean part:** the deficit is already there
      **upstream of the step** (x/h −2.85 … −0.15, no recirculation), first-cell `k` **0.54–0.68×** with
      the channel core at parity (0.987), and the lip-shed rows carry the same ratios. In wall units
      `k⁺` ≈ 1.3–2.4 against OpenFOAM's 2.4–4.0, DNS channel ≈3.9–4.4, and the SST log-layer equilibrium
      `1/√Cμ` = 3.33 — **low against both references**, which is what makes it a defect rather than a
      difference. It seeds a shear layer carrying 20–25% less `ν_t` through x/h 0.3–2.
    - ❌ **Exonerated with evidence:** the momentum scheme (aquaflux is *more* dissipative yet its shear
      layer is *thinner* — wrong sign); the SST shear limiter (identical branch in both); the mesh
      (cell-for-cell identical, max centroid distance 4.7e-9 m); and first-order upwind on k/ω (false
      diffusion ~5% of `ν_t S²` at x/h 0.4, <1% beyond — an order of magnitude too small).
    - ⚠️ **THE EXPANSION RATIO CHANGES WHICH NUMBER LOOKS WRONG, AND NEITHER OF US HAD ACCOUNTED FOR
      IT.** The famous benchmarks are ER 1.125–1.2 (Driver & Seegmiller 1985: 6.26 ± 0.10; Le, Moin &
      Kim 1997 DNS: 6.28) and **must not be used as the target here — this case is ER = 2**, and the
      published trend is that larger ER *lengthens* the normalized bubble (Armaly et al. 1983, quoting
      Durst & Tropea 1981). **At ER = 2 the 2D references cluster at 8.0–8.8**: Pont-Vílchez, Trias,
      Gorobets & Oliva (2019, *JFM* 863) DNS gives **X_r = 8.8h** at Re_τ = 395 (verified independently
      of the agent that reported it — a later LES cites it as its reference, calling its own 8.15h a
      7.3% under-estimate); Durst & Tropea (1981) 8.5 at Re_H 1.5e4; Rothe & Johnston (1975) 7.8.
      **So aquaflux's 8.36 sits INSIDE the ER = 2 band and OpenFOAM's 7.24 sits below it** — the
      opposite of the "aquaflux over-predicts by 15%" framing this section started from.
    - ⚠️ **"SST is a known under-predictor" is also wrong.** On Driver & Seegmiller, NASA's turbulence-
      model resource puts SST at x/H ≈ **6.50** against 6.26 ± 0.10 — a 4% **over**-prediction,
      reproduced across four codes; Menter (1994) reports 6.5 himself. The classic under-prediction is
      **k-ε** (Menter's Jones–Launder 5.5), and even the widely-repeated "20–25%" is disowned by
      Thangam & Speziale (ICASE 91-23) as a resolution artifact.
    - **What is genuinely unresolved is the FINITE SPAN.** span/h = 4 is 2.5× below the span/h > 10
      that de Brederode & Bradshaw (1972) require for two-dimensionality at the centreline (quoted
      verbatim in Jovic & Driver 1994, NASA TM 108807). Lower aspect ratio **shortens** reattachment —
      direction unanimous across every source found — which would pull both codes below the 2D band.
      **No open-access `x_r`-vs-aspect-ratio numbers exist**, so the magnitude is unknown and neither
      7.24 nor 8.36 can be called correct.
    - ⚠️ **One anomaly worth chasing: our spanwise profile has the WRONG SIGN against the literature.**
      Both codes here reattach *later* at the outer slabs (10.28) than in the interior (7.24), but the
      two published spanwise measurements go the other way — Sugiyama et al. (2013) find near-side-wall
      reattachment ~60% of the centreline value at AR 16, and Armaly, Li & Nie (2003) find a *minimum*
      near the side wall. Both are laminar/transitional, so the transfer is uncertain, but this is the
      one place our result contradicts published structure rather than merely differing in magnitude.
    - **Untested and the largest remaining unknown: grid convergence.** One mesh exists.

  **⚠️ THE LEVER, as corrected earlier — with the scan's reach stated.** The measurement below is a **1-D scan of
  one row at one iterate with every other field frozen**. It establishes a strictly positive root of cell
  12800's frozen-field `k` row, which removes the motivation for a projection *at this state*. It does
  **not** establish the sign of the coupled Newton direction, nor anything about the other 1875 collapsed
  cells, nor that no state has a negative root. Two defects remain, and they are different from
  each other:
  1. **`positive_block_limit` is a purely RELATIVE rule** — `room = tau·k/|dk|` — applied to a field
     whose physical floor is 0. A relative rule has no lower bound, so a roundoff-level `dk` in an
     already-collapsed cell produces an arbitrarily small cap. An **absolute floor**
     (`room = tau·(k + k_abs)/|dk|`) fixes it, is pure globalization, sits off the IFT path entirely,
     and — *provided the cap is inactive at the converged state* — changes nothing at the solution.
     Scale `k_abs` off the block (`eps·max(k)`), not a constant, so it carries `k`'s units.
     **⚠️ IT IS NOT SAFE ALONE — clamping `k` in the closure is a PREREQUISITE, in the same change.**
     Relaxing the cap lets `k` go slightly negative, and two sites then misbehave badly (verified):
     `SSTModel.f1`/`.f2` (`sst.py:136,178`) take a raw `jnp.sqrt(k)` and NaN the whole residual, where
     every closure in `boundary.py` uses `safe_sqrt(jnp.maximum(k, 0.0))`; and `OmegaProduction`
     (`sources.py:288-294`) divides by `jnp.maximum(nu_t, 1e-30)`, so a negative `ν_t` selects the floor
     and the cap becomes ~**−1e23** at the measured corner values. `KProduction`'s cap
     (`sources.py:209`) also flips sign, turning production into a sink. **Unbuilt and unmeasured.**
  2. **A large part of the near-wall field has laminarized** (1876 cells below 1e-6). That is the real
     physics defect and the absolute floor does not address it. The candidate is a **sustaining /
     ambient source**. ⚠️ **Both the form and the citation first recorded here were WRONG; corrected:**
     - The literature terms are **constant**, and there are **two** of them, one per equation:
       `k: + β* ω_amb k_amb` and `ω: + β ω_amb²`, where `ω_amb` is an **ambient constant, not the local
       ω**. Source: **Rumsey & Spalart, AIAA J. 47(4), 2009, 982–993** (NASA calls it `SST-sust`).
       **Spalart & Rumsey 2007** — cited here originally — proposes *floor values*, not source terms.
     - The form written here first, `+β* k_amb ω` with the **local** ω, is a different term: it makes
       the destruction `−β* ω (k − k_amb)`, pinning `k ≥ k_amb` *uniformly including deep in the
       boundary layer* where ω is O(1e5). At the `bfs3d` corner that is ~80× the literature term and
       would override the `Dirichlet(0)` wall condition the case sets — which is exactly what Rumsey &
       Spalart warn against. It also is **not** diagonal-free: `∂R_k/∂ω = −β* k_amb` is an off-diagonal
       the frozen AMG operator would not see. The literature (constant) form does have both properties.
     - **Motivating evidence is weak.** The `k` deficit is measured at an *unconverged* state whose ω
       rows have not moved off initialization, and OpenFOAM reaches min `k` 1.672e-6 on this mesh with
       **no** sustaining term at all. It would also make two existing assertions tautological
       (`test_coupled_mass_flow.py:138`, `test_coupled_periodic_channel.py:124`, both of which assert
       `min(k) > 1e-6` and document themselves as *non*-tautological).
     - **Order of work: do the step-limiter fix first and re-measure.** If the march then converges with
       `min(k)` near OpenFOAM's, this has no motivating evidence left. Unbuilt and unmeasured.

  Two further leads found while measuring this, both unverified: `ω[12800] = 4.0091e5` is **exactly
  10×** the `omega_wall` value `6ν/(β₁d²)` its own residual imposes, which is the `60ν` initialization
  value never relaxed (so this state's `ω` rows are unconverged too); and `SSTModel.f1`/`.f2`
  (`sst.py:136,178`) use **plain unclamped `jnp.sqrt(k)`** where every closure in `boundary.py` uses
  `safe_sqrt(jnp.maximum(k, 0.0))` — clamping those two would make a transiently negative `k`
  survivable, which is what an absolute-floor cap needs.

  **The rest of this paragraph is the superseded framing, kept because the OpenFOAM comparison in it
  still stands on its own:** `kOmegaSSTBase.C` ends every `k` solve with
  `bound(k_, kMin_)`, and `bound` replaces a negative cell by the **average of its neighbours** before
  flooring at `kMin` (`isf = max(max(isf, fvc::average(max(vsf,min))*pos0(-isf)), min)`). It also
  *replaces* the `ω` row in wall cells (`matrix.setValues`) and lags `G`, so its wall-cell `k` equation
  is linear with a non-negative source. We do the opposite on both counts: both terms stay live
  functions of `k` in one Newton residual (deliberately — a frozen `ω` degenerates the row), and we
  constrain the **step** rather than projecting the state. Constraining the step is what locks up; a
  projection cannot. That reframes the choice, and it is the one to make with the user:
  a positivity **projection** after the step (OpenFOAM's answer, and it changes the forward path but
  not the root provided the floor is inactive there) against keeping the step constraint and finding
  why that cell's row is hard to satisfy (its root is **positive**, at `+1.99e-14`). Neither is built.

  **⚠️ (2026-08-10, LATER STILL) THE `k` WALL BC A/B, RUN AS A CONTROLLED PAIR — and the crash is NOT
  fixed by any of it.** Two full 3-rung marches from the initial state, `BFS3D_TURBULENCE_INVERSE=native`,
  everything identical but `BFS3D_K_WALL`, both carrying the four negative-`k` clamps and
  `stop_on_limit_stall=3`:

  | | `dirichlet` (control) | `zerogradient` |
  |---|---|---|
  | rung 1 | 14 steps, 43 cycles, ‖R‖ 5.882e-06 | 14 steps, 45 cycles, ‖R‖ 7.821e-06 |
  | rung-2 step it locks at | **25** | **35** |
  | rung-2 ‖R‖ at lock | **7.316e-02** | **1.257e-02** |
  | rung-2 steps before the guard ends it | 15 | 24 |
  | outcome | `RuntimeError` from `solve_reynolds_continuation` | same |

  - **❌ Zero-gradient does NOT cure the lock-up.** The same fraction-to-the-boundary ratchet appears,
    same β ceiling of 16.0, same ~100×-per-step cap collapse — just later. Consistent with the
    term-by-term account that put the whole wall closure at ~0.87–0.93.
  - **✅ But it is worth keeping on the evidence: a 5.8× lower residual and 10 more steps of rung 2**,
    and the attribution is clean because the clamps are verified neutral (below) and both arms had the
    guard. It also produced five consecutive *uncapped* full steps in rung 2, which the control never
    does.
  - **✅ THE CLAMPS ARE EXACTLY NEUTRAL, end to end.** The control's rung 1 is **14 steps / 43 cycles /
    5.882e-06 / ‖R₀‖ 3.3078e-01 — identical in every digit to the pre-clamp recorded baseline** — and its
    rung-2 steps 25–29 reproduce that baseline's trajectory exactly (step 25 β 0.4682, ‖R‖ 7.567e-02,
    `a_min` 0.004, cap 3.76e-03; then 1.00e-05 → 1.95e-06 → 1.95e-08 → 1.95e-10). The per-guard
    `array_equal` unit tests said this; a real coupled march now confirms it.
  - **✅ `stop_on_limit_stall` works, on two independent arms.** The control's rung 2 ends at **15 steps
    instead of the recorded 108**; the zero-gradient arm's at 24. ~90 dead steps saved each time, and the
    run now fails with a `RuntimeError` naming the rung instead of grinding out `MAX_STEPS`.
  - **⚠️ THE DEAD GRIND MOVED RATHER THAN VANISHED — a real coverage gap.** Once the guard ends the
    segment, `solve_coupled` falls through to the finishing solve (`ImplicitNewtonSolver`), which has **no
    equivalent guard**, and that grinds at α 0.000 / 0 cycles with the residual frozen until `max_steps`.
    `stop_on_limit_stall` covers `forward_march` only. Fixing that is the next march-level item.
  - The step-limit dumps now carry **β and the anchor** (`cap 2.2923e-06 beta 1.754 anchor yes`), so the
    falsifier named against the earlier diagnosis is closed: these dumps can be paired with the linear
    system that produced them.

  **✅ GATES GREEN on all of the above** (CSR level operator, native trailing inverse, and both march
  changes), with the tiers named because a default-on march guard reaches further than the fast gate:
  - fast gate **967 passed / 1 skipped** (899 unit `-n auto`, 68 integration `-n 1`);
  - `test_coupled_rans` + `test_coupled_amg` + `test_coupled_field_split` + `test_reynolds_continuation`
    — 33 tests, 18 of them `slow` — **33 passed**;
  - `test_coupled_lu` + `test_coupled_ilut` — **15 passed**. These two matter and were nearly missed:
    they drive `forward_march` with `step_control` + `precondition_step`, so they pick up
    `stop_on_limit_stall` exactly as the four above do, and they are in neither the fast gate nor the
    list a first pass would think to run;
  - **`-m validation` — 18 passed.**

  **The trap, recorded because it nearly landed:** a default-on guard on a *shared* seam is not covered
  by "the tests for the subsystem I changed". `forward_march` has one production caller, but everything
  that reaches `solve_coupled` with an observer picks the new default up. Enumerate by **who calls the
  seam**, not by which file the change is in.


## Faithful smoothed aggregation — matching PETSc GAMG

- **⚠️ (2026-08-09): making the JAX-native multigrid a FAITHFUL smoothed aggregation, so a
  comparison against PETSc GAMG means something. Uncommitted work sits on `claude/block-aware-aggregation`.**
  Read this before touching `solve/multigrid.py`'s aggregation path.

  **⚠️ THE EQUILIBRATION HYPOTHESIS WAS WRONG, AND MEASURING IT IS WHAT SETTLED THE COMPARISON
  (2026-08-09).** The suspicion recorded here was that ours and PETSc's were not built on the same
  matrix — `AmgVCycle` calls `equilibrate_cell_major` before handing the operator to PETSc, so GAMG
  coarsens a **unit-diagonal** matrix, while `build_convection_hierarchy` coarsened the raw block,
  whose diagonal on `bfs3d`'s `[k, ω]` slice spans **7.96e5×** (1.26e-06 … 1.0). That description of
  the code is accurate, and the σ_max figures reproduce exactly (**4.365** raw against **2.832e3**
  equilibrated, so the standard prolongator step `1.4/σ_max` is ~0.32 for us and ~5e-4 for PETSc).
  The **inference** from it — that no ours-vs-PETSc number meant anything until it was fixed — does
  not survive. `equilibrate=True` is now built and measured, and it makes our hierarchy **worse**:
  8 → 11 cycles on the RCM arm, 5 → 10 on the plain-aggregation MIS arm.

  **A symmetric equilibration is a SIMILARITY TRANSFORM, so the SPECTRAL half of the setup is
  invariant under it.** `D̂⁻¹Â = S⁻¹(D⁻¹A)S`, so both spectral estimates are unchanged and the
  damped-Jacobi smoother `x += (ω/λ)D⁻¹r` is unchanged with them. Measured on a badly scaled chain:
  the **fine** level's `lam_max` is 1.966 raw against 1.976 equilibrated (equal to power-iteration
  accuracy) while the **coarse** level's is 1.81 against 15.3. The **tentative prolongation** — a
  fixed 0/1 aggregate indicator, which does not transform with the operator — is non-equivariant.

  **❌ "EQUILIBRATION CHANGES THE GRAPH THE COARSENING SEES" — PROPOSED AND REFUTED BY MEASUREMENT
  (2026-08-12). Do not re-propose it; the falsifying run is cheap and is in the tree.** The reasoning
  was sound as far as it went: `symmetrically_equilibrate` returns `(diagonal @ a @ diagonal).tocsr()`,
  **a sparse product stores only entries whose result is nonzero, so it DROPS every explicit zero**
  (verified in isolation), and at `strength_threshold = 0` the aggregation reads the graph and nothing
  else. The dropped counts are large and real — on `bfs3d` at the target-Re cold initial field, the
  trailing `[k, ω]` block loses **26.8 %** of its stored entries to equilibration (5 245 488 → 3 839 276,
  1 406 212 exact zeros) while the leading block loses **0.0 %** (5 662 of 20 981 952).
  **It does not reach the coarsening, because the cell collapse absorbs it**
  (`validation/bfs3d_openfoam/equilibration_graph_effect.py`, running the real `_cell_graph` /
  `_aggregation_edges` / `_mis_aggregate`):

  | trailing `[k, ω]` | cell edges | aggregates | max aggregate |
  |---|---|---|---|
  | raw | 644 166 | 2300 | 45 |
  | equilibrated | 643 967 | 2302 | 45 |

  **199 of 644 166 edges — 0.03 %.** `_cell_graph` sums `\|A_ij\|` over each `block_size²` block, so a
  cell pair keeps its edge if **any** of its entries is nonzero; the dropped degrees of freedom are
  scattered across blocks rather than clustered within them, so almost every edge survives. The coarse
  space is effectively identical. The leading-block control loses **1** edge and its partition is
  bit-identical.
  **So the similarity-transform argument stands, and the recorded explanation — the non-equivariant
  tentative prolongation and the coarse operator it builds — remains the live one for the 5 → 10.**
  **What survives, and it is worth keeping:** *the aggregation is robust to explicit-zero pruning while
  an incomplete factorization is not.* The smoother takes its pattern from the stored pattern directly,
  with no cell collapse to absorb the loss — which is why the same pruning is fatal to `column_reach`'s
  pressure column (see the per-column reach entry under `sparse_jacobian.py`) and inert here. When
  reasoning about a pattern change, **ask which consumer sees it**: a nodal coarsening and a zero-fill
  factorization have opposite sensitivities.

  **Reference numbers, re-measured on the current code** — `bfs3d` `state-00057`, PC β 0.05, the
  `[k, ω]` block **ALONE** (46080 dofs, 4.20M nnz, 91/row), GMRES restart 15 to rtol 1e-8 on the
  TRUE residual, random right-hand side; harness
  `validation/bfs3d_openfoam/trailing_hierarchy_sweep.py`:

  | preconditioner | coarse eq | cycles |
  |---|---|---|
  | PETSc GAMG, plain aggregation, ILU(0) ×4 | 432 | **1** |
  | PETSc GAMG, plain aggregation, ILU(0) ×1 | 432 | **2** |
  | **PETSc GAMG, plain aggregation, point-block Jacobi ×4** | **432** | **2** |
  | ours, MIS, standard prolongator, ×8, EQUILIBRATED | 2150 | 4 |
  | ours, MIS, plain, ×4, raw | 2150 | 5 |
  | ours, RCM, symmetric-part prolongator, ×4, raw | 598 | 8 |
  | ours, MIS, plain, ×4, EQUILIBRATED | 2150 | 10 |
  | ours, RCM, symmetric-part prolongator, ×4, EQUILIBRATED | 598 | 11 |
  | ours, MIS, standard prolongator, ×4, raw / EQUILIBRATED | 2150 | 44 / 44 — both fail (true rel 1.0) |

  **Read the THIRD row, not the first — the like-for-like arm.** Comparing our Jacobi-class smoother
  against PETSc's incomplete-LU measures the smoother, not the hierarchy, and an ILU sweep is a much
  stronger and much less parallel unit of work.

  **✅ THE GAP IS CLOSED: matched to PETSc's algorithm, our V-cycle equals it — 2 cycles against 2,
  on a 436-equation coarse space against 432.** Two differences accounted for the whole of it, and
  neither was scaling. Both were found by reading `agg.c`/`misk.c`/`rich.c` against our code rather
  than by tuning:

  | arm, `[k, ω]` block alone, matched smoother class | coarse eq | cycles |
  |---|---|---|
  | PETSc GAMG, plain aggregation, point-block Jacobi ×4 | 432 | **2** |
  | ours, MIS ×4, damped, no aggressive level | 2150 | 5 |
  | ours + **aggressive level** ×4, damped | **436** | 10 |
  | ours + aggressive level ×8, damped | 436 | 3 |
  | ours + aggressive level + **undamped** ×2 | 436 | 11 |
  | **ours + aggressive level + undamped ×4** | **436** | **2** |

  1. **The aggressive first level, which PETSc applies BY DEFAULT and we did not.**
     `build_amg_vcycle` never sets `pc_gamg_aggressive_coarsening`, and GAMG's default is
     `aggressive_coarsening_levels 1` with `use_aggressive_square_graph` — so on level 0 it coarsens
     the **squared** graph. That is the entire 5× coarse-space difference: ours coarsened 21×, GAMG
     107×, and with `aggressive_levels=1` ours lands at 436 equations against PETSc's 432.
     **Note `use_aggressive_square_graph` and `aggressive_mis_k` are ALTERNATIVES, not a pair** —
     `mis_k` is read only in the non-squared branch (`agg.c:1314`), so at the default the coarsener
     is plain MIS at distance **1** on the squared graph.
  2. **PETSc's level smoother is UNDAMPED.** `mg_levels_ksp_type richardson` at its default
     `scale = 1.0` (`rich.c:277`) is `x += D⁻¹(b − Ax)`, while ours relaxed by `omega/lam_max` with
     `omega = 0.8`. That is never a *smaller* factor than 0.8 and usually much smaller: `D⁻¹A` has
     unit diagonal blocks, so its eigenvalues average one and `lam_max ≥ 1` always. Dividing by it
     under-relaxes every mode except the extreme one. Worth **10 → 2 cycles** at four sweeps.
     Reachable as `convection_multigrid_solve(..., omega=1.0, spectral_damping=False)`; the
     spectral default is unchanged, because an undamped sweep is not a contraction standing alone —
     it does not need to be under a coarse correction and an outer Krylov.

  **Equilibration is NOT among the causes**, and the two that are were invisible until the sources
  were read side by side. The earlier "ours is behind on both axes" reading was correct as a
  measurement and wrong as a diagnosis: it described a hierarchy that was simply not running the
  same algorithm.

  **The three remaining differences, now BUILT and measured (same block, same state, matched arm):**

  | arm | coarse eq | ×1 | ×2 | ×4 |
  |---|---|---|---|---|
  | matched, before | 436 | 58 | 11 | **2** |
  | + fix-up pass + magnitude-first graph | 438 | 52 | 10 | **2** |
  | + coarse-size stopping rule | 438 | 52 | 10 | **2** |

  - **The fix-up pass — BUILT, `_reattach_to_adjacent_root`.** PETSc's `fixAggregatesWithSquare`
    (`agg.c:1032-1075`): after the squared-graph MIS, each selected root **in ascending index
    order** steals every distance-1 neighbour in the *unsquared* graph that belongs to another
    aggregate. Only members move, never roots, so the count is unchanged and no aggregate empties.
    **Its guarantee is conditional and must not be stated as absolute** — a member whose neighbours
    are all themselves members has no adjacent root to move to and keeps its distant one. Worth a
    cycle at ×2 and six at ×1; nothing at ×4, where the arm was already at PETSc's 2.
  - **Magnitude before symmetrization — BUILT.** PETSc block-sums `|A_ij|` and *then* symmetrizes;
    we took the symmetric part first, so an edge with `A_ij ≈ −A_ji` **cancelled out of the graph
    entirely**. On an M-matrix (every frozen upwind transport operator) the sparsity is identical
    either way, so at `strength_threshold = 0` — the default, and what every scalar hierarchy runs —
    this is a no-op. **It is NOT a no-op at a threshold**, because the weights differ (`|A_ij|`
    against `|A_ij + A_ji|/2`) and the strong-connection set is chosen from them: the coupled flow
    block runs at **0.25** (`turbulence/coupled.py`), so its hierarchy genuinely moves. Do not quote
    the M-matrix equivalence without that caveat — it is a statement about the pattern only.
  - **The stopping rule — measured INERT here, and that is a real result rather than a null one.**
    PETSc coarsens until the grid is under `coarse_eq_limit`; we stop at `max_levels`, at which
    point `max_coarse` can never fire. Setting `max_coarse=2000, max_levels=20` gives a
    **bit-identical** hierarchy on this block, because one aggregation already lands at 438 < 2000
    and both rules then stop. So it changes nothing *at this size* and remains the correct rule for
    a mesh where it would bind. The two knobs are still not equivalents and must not be quoted as
    matched.

  **Still different, verified by reading the source, and NOT measured:**
  - **Strength-of-connection semantics differ** — PETSc thresholds absolutely on the
    diagonally-scaled graph (Vanek), we use the row-max-relative classical criterion. Inert at
    threshold 0, so it affects no measurement here, but the recorded threshold arms on the two sides
    are not comparable.
  - **Tentative prolongation — NOW MEASURED, and it is REFUTED as a deep-hierarchy fix.** PETSc
    orthonormalizes each aggregate's block by QR (`agg.c:690-714`), giving unit-2-norm columns where
    ours are 0/1 — i.e. `P_ours = P_petsc · diag(√|agg|)`. The inertness claim is confirmed exactly (a
    2-level pair is bit-identical), but deeper it is **17–30× worse**, not better. Built as
    `orthonormal_prolongation`; see *"Depth, singletons, and the prolongation"* below for the arms and
    for the mechanism that was proposed and refuted.
  - **Singletons — NOW MEASURED, and this one was a real defect worth 1.7×.** PETSc drops a
    neighbourless vertex from the coarse space entirely (zero row in `P`, left to the smoother); we gave
    it its own aggregate. On this operator most such vertices are not neighbourless at all — they are
    an artifact of the random sweep's arrival order — and a three-level hierarchy came out with 49
    singletons of 161 aggregates on its second level. Fixed behind `avoid_singletons`; see below.

  **What equilibration IS worth (keep it). ⚠️ "Default off" was a scope error — settled from source
  2026-08-10:** `build_convection_hierarchy` and `_build_aggregation_hierarchy` default `equilibrate=False`,
  so "default off" is true of the **JAX-native builder only**; `native_nodal_inverse` overrides it to `True`
  (its per-cell block solve is not otherwise safe); and the PETSc `AmgVCycle` path equilibrates
  **unconditionally** via `equilibrate_cell_major`. Three different objects, no contradiction.

  **⚠️ THE SECOND, COARSENING-INDEPENDENT BENEFIT RECORDED HERE WAS A PSEUDO-INVERSE ARTIFACT, AND IT IS
  GONE (2026-08-14).** The claim was that the coarse solve's singular-value truncation is what a badly
  scaled operator defeats first, so rescaling buys accuracy there: on the badly-scaled chain fixture
  (condition ~1.6e8) the one-level direct solve went from 1.2e-09 unscaled to 5.3e-13 rescaled. That is a
  property of the **pseudo-inverse**, which truncates small singular values, and not of the operator. The
  coarse solve is now a factorization (`_dense_inverse`), which the scaling does not defeat, and the same
  fixture reads **2.8e-13 unscaled against 3.9e-13 rescaled** — both at the floating-point floor, the
  unscaled one four thousand times better than it was, and their ordering now noise. The unit test that
  pinned the gap asserted `scaled < raw` and duly failed on the change; it now pins only what remains
  true, that both paths invert `A`. **The surviving case for rescaling is per-cell block conditioning**
  (measured separately, orders of magnitude in the median) and its effect on the coarse operator the
  tentative prolongation builds — not coarse-solve accuracy.

  **✅ THE COARSE SOLVE IS A FACTORIZATION, NOT A PSEUDO-INVERSE — `_dense_inverse` (2026-08-14).** A
  pseudo-inverse is a singular value decomposition, and it was **59 % of a whole hierarchy build**
  (1.37 s of 2.32 s on a 92160-dof, 8.7M-nnz flow block coarsening to 2156). The generality is only
  needed for a *singular* coarse operator, which this module works to avoid producing — an empty
  aggregate is given a unit diagonal precisely so no coarse row is structurally zero. So the LU is taken
  where it is valid, falling back to the decomposition where it is not. Cost at the coarse sizes that
  arise, and at the ceiling `_MAX_DENSE_COARSE_DOFS` permits:

  | n | pinv | inv | LU factor alone |
  |---|---|---|---|
  | 868 | 0.090 s | 0.013 s (7.0×) | 0.011 s |
  | 2000 | 1.027 s | 0.118 s (8.7×) | 0.032 s |
  | 4096 | 12.06 s | 1.154 s (10.5×) | 0.263 s |
  | **8192** (the cap) | **101.8 s** | **9.7 s** | 2.0 s |

  Whole-build effect **1.97×** (2.30 s → 1.17 s), with the coarse inverse agreeing to 6.4e-15 and the
  cycle output to 1.1e-16. **The gap widens with coarse size**, so this matters more as the mesh grows,
  which is the direction 3D is heading.
  ⚠️ **Singularity is judged by LAPACK's reciprocal-condition estimate, not by whether the factorization
  raised.** An operator can be numerically singular without being exactly so, and a factorization of one
  returns finite nonsense rather than failing — measured, `np.linalg.inv` of a matrix with a 1e-18
  diagonal entry returns finite garbage without complaint. The threshold is the same `n · eps` the
  pseudo-inverse itself uses when truncating, so the two paths agree on where invertibility stops
  instead of drawing the line in two places. All three cases are pinned
  (`test_the_coarse_solve_is_a_factorization_where_the_operator_admits_one`); a fallback that never
  fires and one that always does look identical from the outside.

  **What is BUILT and measured good (keep):**
  - **Nodal (block-aware) aggregation** — `_cell_graph` collapses the dof graph to cell connectivity
    (exact, since field-major means `index % n_cells` *is* the cell) and `_block_tentative` gives each
    field its own coarse unknown. With a **block smoother** (`_cell_block_inverse`,
    `_block_diagonal_inverse_operator`) this makes the two-field slice buildable and convergent.
    **All three are required; no pair suffices** (measured 2×2).
  - **MIS aggregation** (`_mis_aggregate`) — greedy maximal independent set over a **randomized** visit
    order, single sweep, selector-claims-neighbours, faithful to `MatCoarsenApply_MISK_private`. Worth
    **8 → 5 cycles** over the two-pass RCM scheme. The old scheme only seeded from a fully-free
    neighbourhood, so it seeded few aggregates and left most vertices to a ragged cleanup pass.
  - `jax.jit` on the native applies (they were dispatching eagerly, ~18 % of apply cost) and
    `indices_are_sorted=True` in `_coo_apply` (CSR→COO is row-sorted by construction).
  - **There is no `PerFieldNativeInverse` — deleted 2026-08-15 (binding).** It had no production caller
    (only its own tests and one sweep arm), its own docstring recorded it as superseded by
    `NodalNativeInverse` wherever that works, and it was never ported onto `NativeHierarchyInverse` — so
    it had neither `refactor_block` nor `refactor` and `BlockTriangularFieldSplit.refactor` **raised**
    the first time a march refreshed with it, while being exported from `__all__` as public API. It also
    re-committed two costs the shared base was written to remove (eager per-field transposes; a fresh
    closure per apply). Its measurements never transferred to the nodal inverse in any case — a
    per-field pair is a weaker object than one block-aware hierarchy — so the `nativeN` arm in
    `turbulence_smoother_sweep.py` went with it, leaving `nodal[N][cM]`.
  - `NodalNativeInverse` (`solve/field_split.py`, over the shared
    `solve/native_inverse.NativeHierarchyInverse` base), both transposable in
    closed form and fixed linear operators, so adjoint-legal. **⚠️ "Neither is wired into production" was
    wrong — settled from source 2026-08-10.** The nodal inverse is reachable through
    `native_nodal_inverse` and is the `BFS3D_TURBULENCE_INVERSE=native` arm; what is true is that it is not a
    *default*.

  - **`prolongation_smoothing` is its own parameter, no longer welded to `mis_aggregation`.** The old
    flag chose the aggregation *and* the prolongator formula together, so "MIS with and without
    prolongator smoothing" was not expressible and the two effects could not be separated — which is
    how the smoothed prolongator came to be recorded as void when what it actually needs is more
    smoother sweeps. `"symmetric-part"` (default, the historical formula), `"standard"` (the textbook
    σ_max form on the true operator and scalar diagonal), `"none"` (plain aggregation, what the
    shipped PETSc bundle runs). An unknown value raises.
  - **`_mis_aggregate` returns its ROOTS**, not just the aggregate index per vertex — the fix-up pass
    cannot be written without knowing which vertex seeded each aggregate, and re-deriving it
    afterwards is not possible (an aggregate's root is not recoverable from the labelling).

  **What is BUILT and measured BAD (revert or gate):**
  - The **standard prolongator at 4 sweeps** fails outright (44 cycles, true rel 1.0) — raw *and*
    equilibrated, so this is not the scaling. At **8 sweeps equilibrated it is the best native arm
    (4 cycles)**, so it is under-smoothed rather than wrong: the earlier "VOID / much worse" reading
    conflated a smoothing deficit with a broken formula. Treat sweeps as part of that arm's
    specification, never quote it at a single sweep count.
    ⚠️ **Measured on the 2150-equation scalar problem under a point-block-Jacobi / ILU smoother. It does
    NOT carry to the flow saddle**, where both smoothed prolongators are refuted under the SIMPLE
    smoother at 4 *and* 8 sweeps — see *Smoothed aggregation on the flow saddle*. "Best native arm"
    below always means best on the operator it was measured on.
  - **⚠️ GRAPH SQUARING ON THE FIRST LEVEL IS GAMG'S DEFAULT COARSENING, NOT A SIZE KNOB — and
    recording it as harmful was the single thing holding the comparison open.** Squaring the graph on the first level is not a size knob, it is
    GAMG's *default coarsening* (`aggressive_coarsening_levels 1`), and its 106× ratio is not
    over-coarsening — it is PETSc's 107×, i.e. the target. The 5 → 10 reading is real but was taken
    with the damped smoother; at the matched undamped smoother the same arm is **2 cycles**. A knob
    measured harmful in one configuration was recorded as harmful in general, and that closed off
    the one change that mattered.

  **Two corrections to older entries in this file, both of which cost real time:**
  - The recorded *"the builders refuse the `[k,ω]` slice because its diagonal is negative from the live
    source linearizations"* is **wrong**. The fine slice is clean — **0 of 23040** cells non-positive at
    β = 0, at the march shift, and at the floor. The refusal names `level 1`, a **coarse** operator, and
    is manufactured by **field-blind aggregation** merging a k-dof with an ω-dof of another cell. The
    error text said `level 1` all along; an inference was stapled to a verbatim quote and only the quote
    was ever checked.
  - **A field split HIDES the quality of its trailing half.** That half is ~11 % of the nonzeros, so the
    flow block carries the solve: a trailing inverse that does not converge *at all* in isolation still
    produced a plausible 1.36× blended march estimate. **Measure a block's preconditioner on the block
    alone first**, then in situ.

  - **A THIRD correction, from this round: an arm quoted at one smoother-sweep count is not a
    result about the method.** The standard prolongator was written down as "much worse" from its
    4-sweep numbers; at 8 it is the best native arm. Two of the three wrong conclusions in this
    section came from holding one axis fixed while attributing the outcome to another.

  **Also established:** block Jacobi is a perfectly good smoother class here given enough sweeps (PETSc
  ×4 = 2 cycles against ILU ×1's 2), and on CPU the two cost the *same* per apply (102 vs 101 ms) — so
  the penalty for a GPU-friendly smoother is quality, buyable with sweeps, not cost. The remaining
  question is a GPU one and **cannot be measured in the current environment** (CPU-only JAX).

  **⚠️ COMPARE LIKE WITH LIKE, or the arm measures the smoother and gets attributed to the hierarchy
  (binding for this campaign).** Our multigrid has only Jacobi-class smoothers; PETSc's default here
  is an incomplete-LU sweep, which is both far stronger and the least parallelizable piece in it. An
  ours-vs-PETSc table whose PETSc row is ILU therefore cannot say anything about the *coarsening*,
  which is the thing under development. Every such table must carry a **matched-smoother** row —
  PETSc `pbjacobi` against our block Jacobi, at the same sweep count and the same aggregation — and
  that row is the one to quote. The ILU rows stay as the incumbent's absolute bar, not as the
  comparison.

  **⚠️ HOW THE GAP WAS ACTUALLY FOUND, because the method generalizes and the alternative cost
  days.** Three rounds of tuning our own knobs (equilibration, prolongator variants, sweep ladders)
  moved nothing and produced two wrong write-ups. What worked was **an independent agent reading
  `agg.c` / `misk.c` / `rich.c` against `multigrid.py` and reporting differences with citations on
  both sides** — it found the undamped Richardson scale, the default aggressive level, and the
  fix-up pass in one pass, none of which was visible from any measurement we had. When a
  reimplementation of a *published, available* algorithm underperforms it, read the source before
  running another sweep; and do not describe a port as faithful until something has checked it
  line by line, which is exactly the claim that was wrong here.

  **NEXT STEPS, in order:**
  1. **Decide whether the matched configuration becomes the DEFAULT, and for which consumers.**
     ⚠️ **This shipped: `native_nodal_inverse` now defaults to the whole matched bundle**
     (`aggressive_levels=1`, `prolongation_smoothing="none"`, `spectral_damping=False`,
     `equilibrate=True`) and is the `BFS3D_TURBULENCE_INVERSE=native` arm. The open part is whether it
     transfers to other consumers; the library `build_amg_vcycle` defaults are untouched. The measurement says the matched
     bundle is 5 → 2 cycles on the turbulence block; whether that transfers to the scalar transport
     and velocity blocks is unmeasured, and flipping a default is a march-level decision, not a
     block-probe one.
  2. Then, if still short: sparse-LU coarse solve (ours is a dense inverse, `_dense_inverse`),
     QR-orthonormalized
     tentative columns (inert at 2 levels, not deeper), and Chebyshev with the SA-cached bounds.
  3. **Scalability, independent of all the above and unaddressed:** the convection hierarchy is capped
     at 2 levels (`_CONVECTION_LEVELS`) with a **dense inverse** coarse solve. At a 77× ratio that is
     ~26k coarse dofs and 5.4 GB to store at 1M cells — infeasible. Depth now *builds* (3 levels, no
     refusal) via the new `max_levels` argument, so the fix is cheap once the method itself works.
     PETSc reaches only 2 levels here because it coarsens until under `coarse_eq_limit`; our
     `max_coarse` is NOT the equivalent knob and has no effect at 2 levels.

  **Local PETSc source** for continuing the port (shallow sparse clone, may need re-cloning):
  `src/ksp/pc/impls/gamg/{agg,gamg}.c` and `src/mat/graphops/coarsen/impls/misk/misk.c` — note
  `coarsen` moved under `graphops` in current PETSc. GAMG-AGG defaults: `nsmooths 1`,
  `aggressive_coarsening_levels 1`, `use_aggressive_square_graph TRUE`,
  `use_minimum_degree_ordering FALSE` (so: random order), `aggressive_mis_k 2`, `graph_symmetrize TRUE`.


## The flow block — native preconditioning of the `[u, v, w, p]` saddle

### ✅ `jax.grad` RUNS ON THIS CASE, AND IS VALIDATED — the first gradient ever taken here (2026-08-15)

**Read this before citing any zero-shift result in this section as an adjoint result.** The whole
section is justified by the transpose solve behind every gradient meeting the unshifted operator — but
every other number in it is a **linear probe**, driven by the right-hand side `-R(state)`
(`field_split_probe.py`). The actual adjoint had never been executed. It now has been, it works, and it
agrees with a finite difference. Harness `validation/bfs3d_openfoam/adjoint_probe.py`.

*Configuration, every run below:* `state-00067` started from its **physical** fields
(`coupled.physical_fields`, not `layout.unpack` — the checkpoint holds `log(omega)`), `forward_rtol`
0.3, β = 0 throughout, field split, **shipped** column reach `(3,3,3,3,2,2)`, native nodal trailing
inverse, `coarse_eq_limit` 2000, positivity floor 1e-8, preconditioner built once on concrete
parameters, adjoint solver `relative_residual_gmres(1e-6, restart=120, max_restarts=150)`. Objective
`sum(k^2)`; the differentiated parameter scales the molecular viscosity.

**TWO things blocked it, and only the first was known.**

1. **The API gap.** `solve_coupled` did not expose `adjoint_solver`, so the transpose solve ran at
   `default_linear_solver()` = lineax's stock restart and stagnation budget and raised *"A stagnation in
   an iterative linear solve has occurred. Try increasing `stagnation_iters` or `restart`"* — the remedy
   the error names was unreachable from the coupled entry point. Threading it (`solve_coupled(
   adjoint_solver=…)`) removes that failure mode outright.
2. **The BUDGET, which is the part nobody had measured.** With the knob threaded, the transpose solve
   needs **~1450 preconditioner applications**; the first attempt gave it 60 restart-15 cycles ≈ 900 and
   died on `The maximum number of solver steps was reached`. The operator was never intractable — it was
   under-resourced by ~1.6×, and the budget had been sized from the stale zero-shift figures below.

**THE GRADIENT, and the finite-difference check.** The mismatch at a loose root is severe and is
entirely the *finite difference's* fault — the adjoint barely moves while the difference walks onto it:

| forward root `rtol` | central difference (`eps` 1e-4) | adjoint | gap |
|---|---|---|---|
| 2e-2 | −2.5807e+03 | −3.1793295e+03 | **23.2 %** |
| 1e-3 | −3.1964e+03 | −3.1793668e+03 | 0.53 % |
| **1e-4** | **−3.1787556e+03** | **−3.1793669e+03** | **1.9e-04 — agrees** |

- **The adjoint is stable to 6e-8 relative across the last two rows** while the difference moves 24 %.
  So the gradient was right throughout, and every "mismatch" was a difference taken between two solves
  that were **not at their roots** — the implicit-function-theorem adjoint is only valid at `R = 0`, and
  a 2 % relative residual is nowhere near it.
- ⚠️ **A LARGER `eps` MAKES IT WORSE, NOT BETTER — so "the finite difference is noisy" is the wrong
  mechanism, and the arithmetic that seems to confirm it is a coincidence.** At `rtol` 2e-2 the gap goes
  **23.2 % at `eps` 1e-4 → 64.2 % at `eps` 1e-3**. Solve *noise* would fall as `1/eps`; what actually
  happens is that a bigger parameter step moves the two solves' stopping points further apart. The
  seductive trap: the `eps` 1e-4 discrepancy (599) matches `objective-noise / 2 eps` to three figures,
  which reads as proof of noise and is not. **Fix the root, not the step size.**

**THE ADJOINT'S COST, measured — and this is the yardstick the campaign lacked.**

| arm at β = 0 (`rtol` 1e-3, otherwise identical) | adjoint applications | derived cycles | per application |
|---|---|---|---|
| **PETSc ILU(0) flow block (incumbent)** | **1454** | **12.0** | ~195 ms |
| native SIMPLE flow block, shipped settings (**4** sweeps) | 1696 | 14.0 | ~383 ms |
| native SIMPLE flow block, **8** sweeps | **1696** | **14.0** | ~607 ms |

- **The native flow block does NOT beat the incumbent on the real adjoint** — ~17 % more applications
  *and* ~2× the cost per application, so ~2.3× the work. This is the one arm the native-preconditioner
  programme exists to test and it had never been run; the recorded 7-against-11 zero-shift win is a
  **linear-probe** result at right-hand side `-R`, not this.
- **⚠️ DOUBLING THE SMOOTHER SWEEPS BUYS EXACTLY NOTHING HERE — 1696 applications either way, not one
  cycle different, at 1.59× the cost per application. So 8 sweeps is STRICTLY DOMINATED by 4 on this
  operator.** That is worth stating loudly because this file's standing rule is the opposite one —
  *"never quote an arm at one smoother-sweep count"*, which earned itself twice when a sweep ladder
  reversed a verdict. On the adjoint's operator the ladder is flat, so the rule's usual remedy (run more
  sweeps before believing a native arm lost) does not apply.
  **The setting is verified to have taken effect, which matters because a no-op produces the same
  identical count:** the coarsening line is unchanged (sweeps do not touch the coarse space, as
  expected) while the measured per-application cost moves 383 → 607 ms (400→800 applications in 153 s
  against 243 s). Cost changed, convergence did not.
- **All three arms' gradients agree to 9–10 significant figures** (−3.179366754e+03 for PETSc and for
  native-8, −3.179366759e+03 for native-4), which is the correctness check behaving exactly as it must:
  a preconditioner changes how the transpose solve reaches the answer, never where it lands.
- ⚠️ **Applications, not cycles, and not wall clock.** A cycle count is a fair proxy only when the
  candidates share a per-application cost, and these arms explicitly do not — the same trap that nearly
  killed the field split (31 % faster end to end *while taking 11 % more cycles*). Wall clock across
  these runs is **contended** (another session ran a test tier throughout) and is not quotable; the
  application counts are contention-immune.
- **Untested, and the honest remaining scope:** only the sweep count was varied on the native side. The
  splitting, `pressure_sweeps`, `omega`, the strength threshold and the level count are all at the
  case's shipped values, and this says nothing about them. What it does establish is that the *cheapest*
  lever on that list — more smoothing — is spent.

**⚠️⚠️ DO NOT SIZE AN ADJOINT SOLVE FROM THE ZERO-SHIFT FIGURES BELOW — treat them as stale until
re-measured.** The 60-restart budget that failed was chosen *from* them (11 cycles at uniform reach, 22
at the shipped one) on the reasoning that 60 leaves ample headroom. That reasoning is unsound twice
over: they are **linear-probe** numbers at a different right-hand side, and they predate much of the
preconditioner work this section records — the smoother, the aggregation, the field split, the trailing
inverse and the column reach have all moved since parts of them were taken.

**The adjoint's right-hand side is NOT `-R`** — it is the cotangent `dL/dphi*`, here `2k`, **localized in
a single field block**, where every linear probe in this file uses the full steady residual. That
difference is why the adjoint's 12 cycles and the probes' 11 are not the same measurement even when the
numbers look alike.

**Instrumentation, and why it counts applications rather than cycles.** The restart-cycle count
`solve_linear` returns is **discarded inside `_implicit_solve_bwd`**, which has no observer, so the
adjoint's cost is unreachable from outside. `adjoint_probe.TransposeApplyCounter` swaps the host
preconditioner's `factors` attribute for a delegating proxy (the same mutation an in-place refresh
performs, seen by an already-compiled solve for the same reason — the callback reads the attribute
rather than capturing it) and counts **transposed** applications. Forward and adjoint share one
factorization, so counting the transpose makes the split exact and free: the forward march never applies
it. Cycles are then *derived*, not measured. A heartbeat every N applications makes the solve watchable;
it separates running from hung, **not** converging from stagnating.

**✅ ITERATION-COUNT INDEPENDENCE IS DEMONSTRATED (2026-08-15) — the coupling analogue of Gate C, and the
one thing a finite-difference check cannot substitute for.** A matching finite difference shows the
derivative is *right*, not that it came from the implicit-function-theorem solve rather than the march
taped onto the tape. The designed test is to repeat the gradient from a **different starting iterate** —
which changes the forward step count while leaving the root, and therefore the gradient, alone. Run at
identical settings (`rtol` 1e-4, `forward_rtol` 0.3, PETSc flow block, adjoint restart 120 to 1e-6),
varying only which checkpoint the solve starts from — `state-00067` is step 28 of the target rung and
`state-00066` is step 27, one step further out on the same trajectory:

| start | forward applications | adjoint applications | adjoint cycles | gradient |
|---|---|---|---|---|
| `state-00067` | 1944 | 1575 | 13.0 | −3.179366936e+03 |
| `state-00066` | **2262 (+16 %)** | **1454 (−8 %)** | 12.0 | −3.179366945e+03 |

- **The forward path got LONGER and the adjoint got CHEAPER.** The cost moved *opposite* to the step
  count, which is stronger than a flat number would have been: a taped reverse pass cannot do that, since
  its cost is the forward pass's by construction.
- **The gradients agree to 9 significant figures from two different starting iterates**, confirming both
  solves landed on the same root — which is what makes the cost comparison a comparison of the *same*
  transpose problem rather than of two different ones.
- **Over all four runs on this case the separation is wide:** forward applications span 1070 → 2262
  (2.1×) while the adjoint stays within 1454–1575 (±8 %) and is **non-monotone** in the forward count.

⚠️ **What the ±8 % is, so it is not misread as drift.** The adjoint's operator is the Jacobian *at the
root*, and `rtol` is relative to each solve's own `‖R₀‖` — so a start further out stops at a slightly
different absolute residual and the operator differs slightly. That is a root-quality effect, not an
iteration-count one, and it is bounded by the same ±8 % seen when the *same* start is converged to
`rtol` 1e-3 against 1e-4.

**Standing configuration for every measurement below**, because none of them mean anything without it:
`bfs3d` `state-00067` (converged, `|R|` 3.586e-06, written at march shift 0.0064), **operator and
preconditioner both at `beta = 0`** — the operator the implicit-function-theorem adjoint solves, and the
only state in this set that discriminates between candidates. Real right-hand side `-R(state)`, GMRES
restart 15 judged on the **TRUE** residual, uniform stencil reach 3, field split with ILU(0) on the
trailing half, harness `validation/bfs3d_openfoam/field_split_probe.py`.

⚠️ **THE INCUMBENT IS THE FIELD SPLIT, NOT THE MONOLITHIC ARM — and calling the monolithic one "shipped"
here cost a day of comparisons against a bar 45 % too slow.** The shipped bundle runs `field split True`
(`compare.py`, and the bundle table at the top of this file), so the arm that differs from a native
leading-block candidate in exactly one place is `split flow/ilu0`. Both reach 11 restart cycles at this
state, but they are **not** interchangeable as a baseline:

| ILU(0) arm at `state-00067`, β = 0 | cycles | solve |
|---|---|---|
| monolithic, six fields | 11 | 42–46 s |
| **`split flow/ilu0` — the matched incumbent** | 11 | **29 s** |

Quote the split arm. A candidate that replaces only the leading block must be measured against the arm
that differs only in the leading block; the monolithic number flatters it by ~45 %.

⚠️ **Two restart caps are in play and residuals across them are NOT comparable:** `max_restarts` 60
(reported as 58 cycles) and 20 (reported as 18). A failing arm runs its cap out, so it costs 3–4× a
converging one — measured, the Krylov solves are ~80 % of a run's wall clock and the fixed setup (case
build, materialize, compile) is a constant ~95 s. Each run therefore carries its own controls.

### Equation (39): the Frobenius-optimal diagonal approximate inverse — the one transferable result

Jemcov & Maruszewski (ECCOMAS/WCCM 2008) minimize `||I - F~^-1 F||_F` over **diagonal** approximate
inverses and obtain the closed form `F~^-1_ii = F_ii / ||F_i||^2`, against Jacobi's `1 / F_ii`. Written
as a ratio the two differ by `F_ii^2 / ||F_i||^2`, which is at most one and is the fraction of row `i`'s
energy sitting on its diagonal — so **the optimal choice is Jacobi with an automatic, per-row
under-relaxation.** That is the derived form of a quantity this solver otherwise sets by hand in at least
three places: a velocity under-relaxation, a preconditioner-only velocity-row shift floor, and the
relative velocity-row relaxation the closest published work on this discretization never manages to drop.

On the `bfs3d` flow block that per-row factor runs min 3.9e-03, **median 0.53**, max 0.998 — a median
47 % under-relaxation. On a small 2D coupled channel it is 0.88–1.0, i.e. nearly inert, which is why a
small case cannot screen this.

**As the velocity predictor inside a SIMPLE smoother it is worth four orders AND changes the sign of the
sweep** (58-cycle cap):

| smoother's `F~^-1` | 2 sweeps | 4 sweeps |
|---|---|---|
| Jacobi `1 / F_ii` | 4.015e-01 | 9.758e-01 |
| Frobenius `F_ii / ||F_i||^2` | **4.281e-05** | **9.399e-06** |

9400× at two sweeps, with nothing else changed and the same 5 s build. **The qualitative change matters
more than the magnitude: under Jacobi more sweeps make it worse (the sweep amplifies), under Frobenius
more sweeps help (it contracts).** A relaxation is exactly what an amplifying sweep lacks, and this
supplies it without a tuned constant.

**On the SCHUR relaxation the answer is sweep-count dependent, and one count gets it backwards** — at 2
sweeps Frobenius is 5.200e-05 against Jacobi's 4.281e-05 (1.2× **worse**); at 4 sweeps it is 3.990e-06
against 9.399e-06 (2.4× **better**). Recorded because the 2-sweep number was written up as the verdict
before the 4-sweep one landed; the file's standing rule against quoting an arm at a single sweep count
earned itself again here.

**The two damping sources are one knob and must move together** (4 sweeps, 20-cycle cap, so not
comparable with the table above):

| Schur diagonal | `pressure_omega` 0.7 | `pressure_omega` 1.0 |
|---|---|---|
| Jacobi | 6.774e-05 | **3.435e-01** |
| Frobenius | 6.828e-05 | **4.199e-05** |

An undamped **Jacobi** pressure sweep blows up; an undamped **Frobenius** one is the best of the four. So
the hand-set 0.7 was standing in for the derived relaxation, and stacking the two over-damps — but the
constant cannot simply be dropped, only replaced.

**Set from this:** Frobenius on **both** the velocity predictor and the Schur relaxation, with
`pressure_omega = 1.0`.

### What does NOT transfer from that paper

- **Multi-step is nearly inert here.** `N` = 1 / 3 / 10 gives 7.399e-03 / 6.690e-03 / 3.823e-03 while the
  solve goes 91 s → 262 s: **1.9× for 10× the work**, where the paper reports `N = 10` working across all
  its cases.
- **Algorithm 2 is WORSE than Algorithm 1 by 45×** (7.399e-03 against 1.632e-04), where the paper reports
  it much better. Untested hypothesis: its velocity seed is one algebraic-multigrid cycle on `F`, which is
  a good solve on a gentle case and may be poor on a developed high-Reynolds velocity block, making a
  seeded start worse than a zero one.
- **The structural mismatch is the likely cause.** Their system has a `(p, p)` block that is exactly zero
  and `D = G^T` (inf-sup-stable unequal-order finite elements); ours has a nonzero Rhie–Chow damping block
  and a nonsymmetric `D`. Their optimality result — two distinct eigenvalues, hence a Krylov method
  converging in two iterations — rests on that zero block and does not carry over. Their cases are a
  driven cavity and a bent duct: no turbulence, no separation, no pseudo-transient shift.

### FLAT block preconditioners are CLOSED on this case

Every arm that replaces the flow block's inverse outright — no hierarchy, no coarse-grid correction over
the saddle — fails at `beta = 0` (58-cycle cap):

| arm | TRUE rel |
|---|---|
| block-SIMPLE, MSIMPLER Schur | 2.554e-06 |
| block-SIMPLE, `a_P` Schur | 7.812e-03 |
| algebraic SIMPLE Schur, native sub-block inverses | 1.049e-02 |
| left block transform (exact) + native multigrid on the transformed operator | 1.473e-02 |
| left block transform (Schur-only) | 1.395e-03 |
| multi-step saddle, Frobenius, 1 step | 1.632e-04 |

**The signature is a floor almost insensitive to the shift** — MSIMPLER moves only ~2× between `beta = 0`
and the forward operating point, where ILU(0) goes 11 cycles to 4. That is an *approximation* ceiling, not
a conditioning one: a flat inverse must get every error component right in one application, and a
SIMPLE-type Schur is worst precisely on the smooth global pressure mode a coarse grid exists to handle.

Two sub-results worth keeping:
- **Neither sub-block inverse is the constraint.** A 2×2 over which half is native: PETSc/PETSc 1.707e-02,
  native/PETSc 1.555e-02, PETSc/native 3.904e-03, native/native 1.049e-02 — a 4.4× spread against an
  eight-order gap to the incumbent.
- **The raw `(p, p)` block is NOT usable as the pressure operator in a split.** With host V-cycles it
  *diverges* (39 cycles, true relative residual 1.198e+00). That it is already 0.71× the SIMPLE-Schur
  elliptic operator does not make it a Schur substitute.

⚠️ **The block-SIMPLE arms were built with `reference_state` at the developed field**, where the production
path passes none and falls back to a characteristic uniform flow at the fastest patch velocity. They were
therefore *flattered* relative to the shipped object, and still failed.

### Coarsening rate and depth — measured, and the direction that helps is the wrong one

Native nodal hierarchy over the flow block, SIMPLE smoother at the settings above, 4 sweeps, 20-cycle cap:

| coarsening | levels | fine → coarse | ratio | TRUE rel | dense coarse solve |
|---|---|---|---|---|---|
| squared graph (`aggressive_levels = 1`) | 2 | 92160 → 872 | 106× | 4.199e-05 | 6 MB |
| plain maximal-independent-set | 2 | 92160 → 4300 | 21× | **2.522e-05** | 148 MB |
| plain MIS + strength threshold 0.10 | 2 | 92160 → 26244 | 4× | aborted | **5.94 GB** |
| plain MIS, three levels | 3 | 92160 → 644 | 143× | 4.964e-05 | — |

- **`max_levels` is INERT while `max_coarse` binds first.** Asking for three levels at the aggressive rate
  produced a bit-identical two-level hierarchy and a bit-identical result, because the first aggregation
  already lands at 872 < `max_coarse` 2000. A level count alone cannot distinguish that from "depth does
  not help", which is why the coarse size is now printed beside it.
- **A strength threshold coarsens LESS, not more.** It aggregates only along strong connections, so the
  aggregates shrink: 0.10 gives a 4× ratio and a 26244-equation coarse grid whose dense inverse is a
  5.94 GB captured constant (JAX warns and the arm was abandoned). It is the wrong instrument for
  controlling coarse-grid size.
- ⚠️ **The aggressive first level on this block is INHERITED, not measured here.** It was adopted to match
  PETSc GAMG on the `[k, omega]` **trailing** block under a point-block-Jacobi smoother, where it was worth
  5 → 2 cycles. On the flow saddle under a SIMPLE smoother it costs **1.7×**.

**A coarse solve is NOT where to worry about a host solver or a sequential algorithm.** The incomplete-LU
this work exists to remove is **fine-grid** work — 92160 rows, 21M nonzeros, on every sweep of every cycle.
A direct solve on a few hundred coarse equations is negligible beside it in both cost and parallelism. An
earlier reading here, that the dense coarse inverse must be replaced before this direction can
proceed, had that backwards: what has to go is the oversized coarse grid, not the density of its solve.

### Depth, singletons, and the prolongation — where the AMG method itself was wrong

**Depth had never actually been tried, and one arm proved it.** Asking for three levels at the
aggressive rate produced a **bit-identical two-level hierarchy and a bit-identical residual**, because
`max_levels` cannot bind while `max_coarse` (2000) stops the coarsening first. A level count alone
cannot distinguish that from "depth does not help", which is why the coarse size and the aggregate-size
distribution are now printed beside it (`aggregate_size_histogram`, and `_AGGREGATE_STATS` for the most
recent build).

**The defect the histogram found: `_mis_aggregate` manufactures SINGLETON aggregates, and they are an
artifact of arrival order rather than of the graph.** The sweep visits vertices in a random permutation
and any unclaimed vertex becomes a selector; a vertex reached late can find *every* neighbour already
claimed, and then opens an aggregate containing only itself. Measured on the `bfs3d` flow block with
plain aggregation, three levels:

| level | aggregates | size min/median/max | singletons |
|---|---|---|---|
| 0 | 1075 | 1 / 17 / 63 | **107** |
| 1 | 161 | 1 / **3** / 35 | **49 of 161** |

A second level with a median aggregate size of three and 30 % singletons is not a coarse space — it is
a slightly smaller copy of the fine grid in which a third of the unknowns stand for one cell each and
couple to almost nothing.

**Fixed by attaching such a vertex to an adjacent aggregate instead** (`_mis_aggregate(
avoid_singletons=True)`; a vertex with genuinely no neighbours is a true isolate and still gets its
own). Same configuration, 20-cycle cap, native SIMPLE smoother at 4 sweeps:

| arm | level-0 aggregates | level-1 aggregates | coarse dofs | TRUE rel |
|---|---|---|---|---|
| three levels, as built | 1075, 107 singletons | 161, median 3, 49 singletons | 644 | 4.964e-05 |
| **three levels, no singletons** | 968, none | 94, median 8, **none** | **376** | **2.844e-05** |

Worth **1.7×**, and it coarsens *further* while doing it (644 → 376). Off by default
(`avoid_singletons=False`) and byte-identical off.

**The practical consequence is that the coarse-space cost problem largely dissolves.** The best
two-level arm is 2.522e-05 on a **4300**-equation coarse grid; three levels without singletons is
2.844e-05 on **376** — within 13 % at **a eleventh of the coarse space**, which is back inside ordinary
practice and trivial to invert densely. (Both figures are residual-after-18-cycles at a 20-restart cap,
not convergence; uncapped, the same two hierarchies converge in 39 and 44 cycles, which corroborates the
conclusion. Every residual quoted in this subsection is capped — see *Where this stands* for the
converged numbers.) The earlier reading here, that gentler coarsening's 1.7× was
unaffordable and therefore not a lever, was measuring a hierarchy whose second level was degenerate.

**⚠️ ORTHONORMALIZING THE TENTATIVE PROLONGATION IS REFUTED — twice, and the mechanism offered for it
was wrong both times.** The recorded prediction was that our 0/1 columns (against PETSc's QR-orthonormal
ones) are "provably inert at two levels with an exact coarse solve" but "start to matter with inexact or
deeper coarse levels", making them the suspect for depth being unhelpful. Built as
`orthonormal_prolongation=True`:

| arm | TRUE rel |
|---|---|
| two levels, with and without orthonormalization | 4.199e-05 **both, bit-identical** |
| three levels, as built | 4.964e-05 |
| three levels + orthonormalization | 8.411e-04 |
| three levels, no singletons | 2.844e-05 |
| three levels, no singletons + orthonormalization | 8.465e-04 |

The two-level identity confirms the inertness claim exactly. Deeper it is **17–30× worse**, and
eliminating the singletons does not rescue it (8.465e-04 against 8.411e-04) — so the proposed mechanism,
that orthonormalization promotes singletons by scaling a column by `1 / sqrt(|agg|)`, is refuted.

**A better hypothesis, offered as one and NOT measured:** the coarse *space* is identical either way, since
scaling a column does not change its span, so this can only be about the coarse *operator*. With 0/1
columns `A_c = P^T A P` **sums** fine rows over each aggregate, preserving row-sum structure — and for a
finite-volume conservation-law discretization the row sums are the conservation statement. Scaling by
`1 / sqrt(|agg|)` destroys it, which is why agglomeration multigrid for conservation laws uses unscaled
sums. Do not act on this without measuring it.

**The singleton fix helps at TWO levels as well — 1.85×, so it is a property of the aggregation and not
a repair for depth.** Measured 2026-08-13 on `state-00067` (β = 0 on both operator and preconditioner,
the adjoint's operator), native SIMPLE smoother at 4 sweeps with the Frobenius diagonal on both halves,
plain aggregation, restart cap 20 (so every arm below stops at 18 cycles — the cap, not a convergence
test; only the residual separates them):

| two-level arm | level-0 aggregates | coarse dofs | build | TRUE rel |
|---|---|---|---|---|
| as built | 1075, 107 singletons | 4300 | 19 s | 2.522e-05 |
| **no singletons** | 968, none | **3872** | 15 s | **1.362e-05** |

It is better, coarser, and cheaper to build at once. **`avoid_singletons` should therefore become the
default rather than an opt-in** — it is currently `False`, and byte-identical when off.

⚠️ **The same fix paired with the SQUARED-GRAPH first level is a NULL TEST, not a negative result.**
Aggressive coarsening builds aggregates of median size 82 and produces **no singletons at all**, so
`-ns` there returns a bit-identical 4.199e-05 and measures nothing. Only the gentler rate strands
vertices. Read a no-change result from that arm as "nothing to fix here", never as "the fix does not
work".

**DEPTH PAST THREE LEVELS DOES NOT PAY, and singletons were not the reason.** With the aggregation
clean at every level, the same configuration continued to coarsen:

| levels | coarse dofs | ratio | TRUE rel |
|---|---|---|---|
| 2 | 3872 | 24× | 1.362e-05 |
| **3** | 376 | 245× | **2.844e-05** |
| 4 | 56 | 1646× | 4.875e-05 |
| 5 | 12 | 7680× | 3.820e-05 |

Four levels is worse than five, so this is not a monotone degradation that a further fix would unwind —
it is noise around a floor the hierarchy cannot get below. Depth buys coarse-grid *size* (376 at 13 %
worse than 3872, which is the trade worth taking) and nothing else.

⚠️ **`max_levels` cannot bind while `max_coarse` stops the coarsening first.** The three-level arm lands
at 376, already under the 2000-equation limit, so `-L4` and `-L5` rebuild the identical hierarchy and
return the identical residual. The depth rows above were obtained by lowering that limit alongside the
level cap. A depth sweep that moves only the level count is measuring one hierarchy N times.

### Smoothed aggregation on the flow saddle — REFUTED under the SIMPLE smoother

Every native arm interpolates the coarse correction piecewise-constant over each aggregate, which is the
textbook reason a hierarchy works at two levels and gains nothing deeper. Smoothing the prolongator once
with the operator is the standard cure, and it had only ever been measured here under a *Jacobi*
smoother — before the Frobenius diagonal turned the velocity predictor from amplifying to contracting —
so it was a fresh question rather than a settled one. It is now settled. Same state and bundle as above,
three levels, plain aggregation, no singletons:

| prolongator | 4 sweeps | 8 sweeps |
|---|---|---|
| **unsmoothed** (`"none"`) | 2.844e-05 | **8.002e-06** |
| symmetric-part (`"symmetric-part"`) | 2.925e-04 | 1.035e-04 |
| standard σ_max (`"standard"`) | 1.611e-01 | 3.005e-01 |
| standard σ_max, equilibrated | — | 8.810e-01 |

Symmetric-part is **10–13× worse** and the gap *widens* with sweeps. The standard formula does not
converge at all, gets **worse** with more relaxation — the signature of a mode the smoother amplifies and
the coarse grid cannot correct, not of under-smoothing — and equilibrating it costs another 3×. Smoothing
also widens the coarse stencil (level-1 operator 0.4M nnz against 0.1M) and changes what the next
aggregation sees, landing a 108-dof coarse grid instead of 376, so the arms differ in coarse space as
well as in interpolation.

**Sweeps beat every structural change tried in this subsection.** Doubling relaxation on the unchanged
three-level hierarchy was worth **3.6×** (2.844e-05 → 8.002e-06) — more than depth, smoothing, or
orthonormalization returned. Weigh it against cost before treating it as a win: eight sweeps roughly
doubles the work in every application, and at a restart cap all arms report the same cycle count, so a
residual gain there is not a like-for-like comparison with a four-sweep arm.

⚠️ **This was written before the strength threshold was measured, and the threshold beats it — 4× against
3.6×, and on cycles rather than on a capped residual.** Every arm in this subsection ran
`strength_threshold = 0`. Do not read "the lever is sweeps, not the hierarchy" out of it; the aggregation
was the larger lever all along, and these arms simply never varied it. See *Where this stands*.

### Where this stands — the native flow block converges: 1.84x at the adjoint, 2.4x on the march

⚠️ **The tables in THIS subsection predate the two free quality levers** (a per-cell block velocity
splitting and an undamped correction) and the per-iteration split measurement. They are kept because the
closures they record still stand, but for the current standing read *The gap is CONVERGENCE, not cost*
below: **7 cycles / 57 s against 11 / 31 s at β = 0**, and **6 / 29 s against 2 / 12 s at β = 0.1**.

**The four-to-five-order gap never existed: it was the GMRES restart cap read as a convergence floor.**
Every earlier reading here was taken at a 20-restart cap, where `restart_cycles` reports 18 for any arm
whatever its quality. Uncapped, the native hierarchy reaches adjoint grade. Two later corrections then
moved the number again, both in the incumbent's favour, so read the table and not the older prose.

Measured 2026-08-13 on `state-00067` (converged, march shift 0.0064, |R| 3.586e-06) with **β = 0 on both
operator and preconditioner**, real right-hand side `-R(state)`, GMRES restart 15 to rtol 1e-8 on the
**TRUE** residual, 60-restart cap, uniform column reach, ILU(0) on the trailing half. Native arms are the
SIMPLE-smoothed hierarchy with the Frobenius diagonal on both halves, plain aggregation, no singleton
aggregates, 5 levels, `max_coarse` 500.

| arm | strength | outer x inner sweeps | cycles | solve | s/cycle |
|---|---|---|---|---|---|
| **`split flow/ilu0` — the matched incumbent** | — | — | 11 | **29 s** | 2.64 |
| monolithic ILU(0) — *not* the right bar | — | — | 11 | 42 s | 3.82 |
| native | 0.10 | 8 x 4 | 11 | 80 s | 7.27 |
| native | 0.25 | 8 x 4 | 10 | 83 s | 8.30 |
| **native — best** | **0.25** | **8 x 2** | 11 | **68 s** | 6.18 |
| native | 0.25 | 8 x 1 | 14 | 70 s | 5.00 |
| **native — fewest cycles** | 0.25 | **16 x 4** | **6** | 98 s | 16.3 |

**STRENGTH OF CONNECTION IS THE LARGEST LEVER, worth 4x.** At `strength_threshold = 0` — the builder
default, and what every native arm ran until now — `_aggregation_edges` returns the full cell adjacency,
so aggregation ignores the operator's values entirely and coarsens across the stiff direction. Turning it
on took the arm from 44 cycles / 213 s to 10 / 83 s. The optimum is interior and sits on the standard
value: 0 → 44 cycles, 0.10 → 11, 0.25 → 10, 0.50 → 11 (and 0.50 costs the most per cycle). **This is not a
new discovery — `turbulence/coupled.py` already ships `strength_threshold: 0.25` on the frozen
velocity/Schur AMGs for exactly this reason.** The native hierarchy was simply built without a setting the
production path already used.

**The value-dependence is free HERE and only here.** A threshold makes the coarsening read `|A_ij|`, so a
rebuilt hierarchy is no longer structurally invariant. The flow block is frozen at the reference state and
never refreshed, so it costs nothing; the k/ω path refreshes (~48 times in a 61-step march) and a binding
decision keeps it at θ = 0 there.

**The INNER pressure sweep count was never varied before this, and is the best cost lever found.** Four
inner relaxations sit inside every outer SIMPLE sweep; the spec parser had tokens for outer sweeps,
levels, coarse size, threshold, aggregation, prolongator and `pressure_omega`, and none for this one — so
the largest single term in the smoother's cost was the one axis held fixed, while the outer count was
swept 4/8/16. Dropping 4 → 2 is worth **18 % of wall clock** for one extra cycle; 4 → 1 removes the Schur
matvec entirely (the peeled first sweep is a pure diagonal solve) and costs three cycles, which is too
many.

**The (outer, inner) frontier is FLAT — swept, and 8 x 2 survives.** Everything from 68 s to 87 s lies
within 28 %, so the pair is not finely tuned and there is no hidden corner:

| outer x inner | 8x2 | 8x1 | 12x2 | 16x1 | 8x4 | 16x2 | 24x1 | 16x4 |
|---|---|---|---|---|---|---|---|---|
| cycles | 11 | 14 | 8 | 8 | 10 | 7 | **6** | **6** |
| solve | **68 s** | 70 s | 73 s | 79 s | 83 s | 86 s | 87 s | 98 s |

One structural fact falls out and is consistent across the whole table: **six cycles is reachable two
ways — 16x4 at 98 s and 24x1 at 87 s — and the cheaper route is more outer sweeps with fewer inner ones.**
At any fixed cycle count, inner pressure relaxations are the expensive way to buy convergence.

Note for the march rather than for this probe: **12 x 2 buys 8 cycles for 73 s**, 7 % above the optimum.
A march's cost is dominated by staleness and refresh cadence rather than by any single solve, so fewer
cycles per solve may be worth more there than wall clock is here. That cannot be settled from a
single-state probe.

**TEN CYCLES IS NOT A FLOOR — REFUTED by measurement.** Three structurally unrelated leading inverses all
landed at 10–11, which raised the real possibility that the count was set by the trailing block or by the
coupling triangle the split discards, and that the comparison was saturated. It is not: at 16 outer
sweeps the native arm reaches **6 cycles**, below the incumbent's 11. The native hierarchy can be made
*stronger* than the incomplete-LU; what it is not yet is cheaper.

**Two structural levers measured and NOT worth taking.** Aggressive (squared-graph) coarsening *combined*
with a threshold does what it promises mechanically — level 0 coarsens 8.7x instead of 3.4x, the hierarchy
drops to 4 levels, the heavy level-1 Schur falls 1.2M → 0.5M nnz — and nets **zero**: per-cycle cost falls
15 % and the cycle count rises by two (11 → 13 at θ = 0.10; 10 → 11 at θ = 0.25). The fine level is ~60 %
of the smoothing cost, so removing intermediate levels can only ever reach the ~28 % that level 1 holds.
**Depth past three levels is separately closed**, non-monotonically (4 levels worse than 5) with a clean
aggregation at every level.

⚠️ **A kernel-level cost model over-predicted its own first result by 6x — treat its remaining estimates
as unmeasured.** A profiling pass modelled the Schur applications as ~64 % of the V-cycle and predicted
16 % from removing the multiply-by-zero first pressure sweep; measured, that change is worth **2–7 %**.
The probe used synthetic matrices with random column patterns, and real mesh-ordered matrices are far more
cache-friendly. Its other figures (per-level sweep schedule 31 %, asymmetric V-cycle 25 %, Schur
truncation 57 %) carry the same flaw and must be measured before being believed — the last one especially,
being the only proposal that would change what `M` is.

**⚠️ SCOPE, BINDING — THE 2.3x IS A ZERO-SHIFT NUMBER. Across the shift range the march runs, the
native hierarchy is ~4x slower.** Swept 2026-08-13 by overriding the operator shift on the fixed
converged state (`BFS3D_PROBE_BETA`, which is the only way to move β without also changing state — the
other entries in `STATES` are step-initial checkpoints and cost every arm a cycle or two). The
preconditioner takes the march's own floor, so β = 0.01 reproduces the shipped operator/preconditioner
mismatch (0.01 against 0.05):

| β | `split flow/ilu0` | native, 8x4 | native, 8x2 | ratio |
|---|---|---|---|---|
| **0** — the adjoint | 11 cyc, 29 s | 10 cyc, 83 s | 11 cyc, **68 s** | **2.34x** |
| 0.01 | 4 cyc, 15 s | 9 cyc, 76 s | 11 cyc, 69 s | **4.6x** |
| 0.05 | 2 cyc, 11 s | 6 cyc, 56 s | 7 cyc, 49 s | 4.45x |
| 0.10 | 2 cyc, 11 s | 5 cyc, 48 s | 6 cyc, 43 s | 3.9x |

**The asymmetry has a mechanism, and it is not going away.** The shift adds `β·d` to the diagonal, and an
incomplete-LU approaches exactness as diagonal dominance grows — so ILU(0) collapses to its floor (2
cycles, 11 s) the moment any shift is present and cannot go lower. An aggregation multigrid gets no such
windfall: its rate is set by smoother/coarse-space complementarity, which the shift barely improves. The
native arm does improve (68 -> 43 s) but from far behind. **The ratio is therefore bounded below by how
cheap the native V-cycle can be made, not by anything about convergence** — at β = 0.1 it needs 6 cycles,
which is a perfectly healthy preconditioner, and still costs 3.9x.

**So the native path's competitiveness is concentrated at β = 0, which is the ADJOINT and nothing else.**
The forward march floors β at 0.05 and never builds a preconditioner at zero shift; the transpose solve
behind every `jax.grad` meets exactly the unshifted operator, with no floor to soften it, and must
converge tightly. That is a real place to win and it is where a differentiable solver spends its gradient
budget — but it is not the march, and a 2.3x quoted without the shift beside it will be read as the march.

**⚠️ THE W-CYCLE AND PRE-SMOOTH REMOVAL ARE BOTH REFUTED ON THIS SADDLE.** Jasak, Jemcov and
Maruszewski (2007) measure, on a segregated LES pressure equation, a W-cycle at 26 iterations against a
V-cycle's 71 on the same coarsener and smoother, and their fastest arms all drop the pre-sweep entirely
— their stated reason being the residual re-evaluation a nonzero pre-sweep forces, which is exactly the
term that dominates a sweep here. Neither survives on the flow saddle. Measured at β = 0.1, strength
0.25, 4 outer x 2 inner sweeps, 5 levels, no singletons:

| arm | cycles | solve | s/cycle |
|---|---|---|---|
| **V-cycle with pre-smoothing** | **8** | **37 s** | 4.63 |
| W-cycle (`mu = 2`) | 6 | 46 s | 7.67 |
| no pre-smoothing | 14 | 41 s | 2.93 |
| W-cycle + no pre-smoothing | 12 | 46 s | 3.83 |

Each lever does mechanically what it promises and neither pays. The W-cycle buys 25 % of the cycles and
costs 66 % per cycle, because a W-cycle visits level *k* **2^k** times — at five levels that multiplies far
faster than the coarse share can absorb, and our level 1 carries a 1.6M-nonzero Schur where their deeper,
more aggressive hierarchies really are nearly free to revisit. Dropping the pre-sweep does cut per-cycle
cost by 37 %, exactly the mechanism claimed, and costs 75 % more cycles.

**⚠️ A first reading of this called convergence here "smoother-dominated, not coarse-grid-dominated".
That is TOO STRONG and two results of the same day contradict it: the strength threshold is a pure
coarse-SPACE change and was the largest win measured (4x), and the W-cycle DID cut cycles 25 % — it failed
on cost, not on effect. "Did not pay" is not "does not matter". What the evidence supports is narrower:
more smoother (8 -> 16 sweeps: 11 -> 6 cycles) and more coarse grid (W-cycle: 8 -> 6 cycles) each buy
25-45 % fewer cycles and each cost more than they return, so this sits at a MARGINAL-COST OPTIMUM rather
than in a regime where one half is limiting.** The mechanism below still explains why each specific lever
loses: On a symmetric Poisson under an incomplete-LU or
symmetric Gauss–Seidel smoother the coarse correction carries the solve, so revisiting it pays and a
post-smooth alone can clean up what an unsmoothed restriction lets through. On this indefinite saddle
under a SIMPLE smoother the fine level carries it, so extra coarse visits buy little and restricting an
unsmoothed residual costs a lot.

**Separating a DEFECT from an INVESTMENT does organize the results, and that much holds.** The coarse
space had a real defect — isotropic aggregation coarsening across the stiff wall-normal direction — and
repairing it was worth 4x. Investing *further* in the coarse grid once repaired has returned nothing at a
price worth paying: more levels, aggressive coarsening, both smoothed prolongators, orthonormalization,
and the W-cycle. But note the W-cycle case carefully — it improved convergence and lost on cost, so this
is a statement about marginal return, not about the coarse grid being irrelevant. Weigh a new
coarse-grid idea against the *cost* it adds, not against a claim that the coarse space is finished.

**The binding constraint is the smoother's absolute quality, which the balance measurement below puts at
`||S|| ~ 1.45` and `||E|| ~ 1.03` — where useful preconditioning needs both far below 1.** Each sweep
contracts very little, which is why many sweeps are needed and why many are expensive. **Improvement
therefore requires a smoother that is better PER UNIT COST, not more of the same one.**

⚠️ **Scoped to this operator, deliberately.** `mu` and `pre_smooth` are kept on `_VCycleOps` (defaulting
to a V-cycle with pre-smoothing, byte-identical) rather than deleted, because `_frozen_v_cycle` is shared
with the smoothed-aggregation path over the **symmetric pressure Schur** — which is the configuration the
literature result was actually measured on, and where it may well hold. What is refuted is the W-cycle on
the saddle under a SIMPLE smoother, not the W-cycle.

### THE NATIVE FLOW BLOCK IN A REAL MARCH — identical trajectories, but the RATIO GROWS WITH THE RUNG

**COMPLETED. Final: 60 steps / 3044 s against the incumbent's 67 / 1957 s — 1.56x — and the SAME
REATTACHMENT LENGTH, `x_r/h` 8.361 mid-span and 12.53 full-span, to four significant figures.** The native
preconditioner converges to the same solution, in SEVEN FEWER STEPS, and pays 1.56x in wall clock: it
produces better Newton directions and pays for them per application. 26 coarsening retraces are inside
that figure, worth roughly 80 s (~7 % of the gap), so removing them helps without changing the verdict.

⚠️ **READ THE RATIO AS A TREND, NOT A NUMBER, AND DO NOT QUOTE THE EARLY RUNGS.** Measured through the
march: **1.32x** (rung 1) → **1.27x** (rung 2) → **1.16x** (target rung, first step) → **1.57x** (target
rung, step 49, where the native arm is at 2342 s against 1489 s and the incumbent's ENTIRE 67-step march
took 1957 s). The early rungs run at high β where both preconditioners are cheap and they flatter the
native arm; the target rung runs at low β, which is exactly where an incomplete factorization's O(eps^2)
advantage is largest. At matched β inside the target rung the native needs **~1.35x the cycles** (9 against
6 at β = 0.0439; 17 against 13 at β = 0.0293). A first write-up of this section quoted "1.16-1.32x" as the
result, which was the most favourable point of a rising sequence.

Everything above is single-state probing. The preconditioner has now been run in the full `bfs3d`
continuation march, which required three things a probe never reaches, and which are the reason no earlier
arm could have been marched at all.

**The native flow inverse needed a `refactor_block`.** `BlockTriangularFieldSplit.refactor` RAISES on an
inverse offering neither `refactor_block` nor `refactor`, because a mid-march refresh must mutate the same
object the compiled Krylov solve holds — replacing it would recompile. A single-state probe exits before
that path. `leading_inverse` is now threaded through `FieldSplitAmgPreconditioner.build` and the coupled
march builder (the underlying `build_block_triangular_field_split` already supported it), and `compare.py`
selects on `BFS3D_FLOW_INVERSE=native`.

**The result, against an archived march of the incumbent** (`zen-newton-3f1b39`, PETSc ILU(0) on the flow
block, same trailing native inverse, same continuation schedule, same `refresh_on_cycles` 3):

| | steps | wall | step-14 `R` | step-38 `R` |
|---|---|---|---|---|
| incumbent | 67 total, 1957 s | — | 7.821e-06 | 4.155e-06 |
| native | in progress | — | **7.822e-06** | **4.155e-06** |
| ratio | | 1.32x (rung 1), 1.27x (rung 2), **1.16x** (step 39) | | |

**The residuals agree to FOUR SIGNIFICANT FIGURES at every step through both early rungs**, with identical
step counts, identical β schedules, and identical line-search factors. The native preconditioner is not
merely converging — it produces the same Newton steps. It also uses **equal or fewer cycles at most
steps**. They first separate in the target rung at step 41, where the native takes a full step
(α = 1.000, `R` 3.343e-02) against the incumbent's clipped one (α = 0.328, `R` 3.757e-02).

⚠️ **The archived march's column reach differs (3/3/3/2/2/2 against the shipped 3/3/3/3/2/2), and it does
NOT matter.** Measured from the two logs' own refresh breakdowns, the difference lands entirely in the
probe phase at **8.3 s against 7.7 s** — about 8 %, not the 24 % a column count suggests — which over ~10
refreshes is 0.6 % of wall. Measure the phase breakdown rather than reasoning from column counts.

**WHERE THE GAP ACTUALLY IS, from the same breakdown at a matched step 28:**

| | native | incumbent |
|---|---|---|
| refreshes | 8 | 9 |
| probe / refresh | 8.3 s | 7.7 s |
| **refactor / refresh** | **5.7 s** | **2.5 s** |
| cumulative wall | 931 s | 776 s |

Refresh accounts for **20 s of the 155 s gap — 13 %. The other 87 % is the V-cycle's per-application
cost**, which the per-iteration split measured independently at 1.45x. So the frozen-coarsening work below
is worth ~20-26 s of a 155 s gap: real, and NOT the lever. The lever is the apply.

**⚠️ `refresh_on_cycles` IS PER INNER ITERATION, NOT PER STEP, AND RAISING IT TO 8 SWITCHED THE REFRESH
OFF.** The trigger is `corrected >= refresh_on_cycles` against a single inner solve, capped at one refresh
per step. A step reporting `cyc 12` is the SUM over three inners at ~4 each, so nothing ever fired: every
step logged `pc none 0.0s`. The march then ground on a never-refreshed preconditioner and the line search
collapsed at step 9 of the LOWEST rung (α 0.469 → 0.030 → 0.000, β escalating, `R` rising) — a rung that
completes in 14 steps in every other configuration. At the shipped threshold of 3 the same march completes
that rung in **14 steps, 429 s**, full Newton steps throughout. The sizing error underneath it: 6-8 was
taken from the PROBE's per-solve count at rtol 1e-8, but the march solves to `forward_rtol` 0.3 and needs
**2-5 cycles per inner**. A cycle count does not survive a change of tolerance any more than it survives a
change of β.

**⚠️ THE COARSENING MOVES ON EVERY REFRESH, AND THE JITTED CYCLE RETRACES.** At `strength_threshold` 0.25
the aggregation reads `|A_ij|`, so a refresh at a developed state produces a different hierarchy: measured
**14 retraces** in one march, level sizes wandering (coarse dofs 312 → 348 → 340 → 288 → 296 → 268 → …).
The sibling nodal inverse rebuilds freely and calls itself structure-preserving only because its
aggregation reads SPARSITY alone — the property this arm gave up to win 4x. That trade was invisible until
the preconditioner met a march.

**✅ THE MECHANISM IS BUILT — `SmoothedHierarchy.refit(a)` (2026-08-13).** It holds this hierarchy's
prolongations and re-derives only what depends on values: each level's Galerkin operator `Pᵀ A P`, its
diagonal or per-cell block inverse, its spectral estimate, and the coarsest level's dense inverse. Shapes
cannot move, so a refitted hierarchy is a compilation-cache hit **by construction** rather than by the
sparsity-pattern-only argument that the strength threshold invalidated. Reached from the case as
`BFS3D_FLOW_FROZEN_COARSENING=1` (and `-fc` on a probe spec), off by default.

**It is EXACT where the flow arm runs, and NOT exact in general — the boundary is the prolongator.** With
`prolongation_smoothing="none"` — what this arm uses — the tentative prolongation is the aggregates' 0/1
indicator and holds no operator values, so freezing it freezes the partition and nothing else: at
`strength_threshold=0` a refit then reproduces a from-scratch rebuild in every level's values. With either
smoothed prolongator it *also* freezes a relaxation built from the old operator, and the coarse operators
then differ from a rebuild's. Both halves are pinned, so the exact agreement cannot be read as
unconditional.

**⚠️ AND THE "MEASURE MEMBERSHIP, NOT SIZE" CAUTION IS NOW DEMONSTRATED, NOT JUST ARGUED.** On a square
grid, aggregating along either direction gives the **identical count** — so rotating an operator's
anisotropy re-partitions the mesh completely while every level size is unchanged. That is now a unit test
(`test_refit_holds_a_partition_that_a_rebuild_would_move`), which is worth more than the argument: **a
march reporting stable level sizes is not reporting a stable coarsening**, and the 10–20 % size drift
recorded above is a lower bound on how much the partition moved, not an estimate of it.

**❌ AND FREEZING LOSES ON A MARCH — the stale coarse space costs FOUR TIMES what the retraces do
(measured 2026-08-13). Do not ship it; the case default stays on the rebuild.** Two full `bfs3d` marches
differing in **one flag**, `BFS3D_FLOW_FROZEN_COARSENING`; otherwise identical (native flow block at
4 outer x 2 inner sweeps, strength 0.25, no singletons, 5 levels, `max_coarse` 500, block splitting,
`omega` 1.0; native trailing inverse; field split; `refresh_on_cycles` 3; three Reynolds rungs):

| | steps | wall | **Krylov cycles** | final ‖R‖ | mid-span `x_r/h` |
|---|---|---|---|---|---|
| rebuild (the default) | 60 | **3044 s** | **327** | 8.076e-06 | 8.361 |
| frozen | 63 | 3153 s | **381 (+16.5 %)** | 1.689e-06 | 8.361 |

**Read the CYCLE row.** The frozen arm converged deeper (1.7e-06 against 8.1e-06) and took three more
steps, so part of the +109 s is work past the tolerance rather than inefficiency; the cycle count is not
subject to that and says the same thing. The like-for-like window — through **step 35**, the last step at
which the two arms agree to four figures — gives **+18 cycles (+12.1 %)** and **+66 s**, so the whole-march
figure is not an artifact of the trajectories parting. Same reattachment length either way.

**The per-rung split is the mechanistic result, and it is cleaner than the total:**

| | cycles | wall |
|---|---|---|
| rung 1 (steps 1–14, Re/100) | 49 → **49, bit-identical at every step** | 429 → **412 s (−17 s)** |
| rung 2 (steps 15–35) | 100 → **118 (+18 %)** | 776 → **859 s (+83 s)** |

**Rung 1 is a clean null: a coarsening derived at the initial condition is worth EXACTLY as much as one
rebuilt at every refresh, as long as the flow has not moved.** So the degradation is not drift in the
aggregation algorithm — it is the anisotropy PATTERN rotating under a partition chosen by reading `|A_ij|`
at a state the flow has since left. That is the "case against" (the eddy viscosity grows to ~150x molecular
and varies spatially) confirmed, against the "case for" (the anisotropy is geometric — wall-normal grading,
within-row conductance ratios of 64 median and 1700 max, 99 % of cells above 4:1), which does not survive.

**⚠️ THE RETRACE IS NOW PRICED, AND IT IS SMALL: ~3.4 s each, ~88 s over a march, ~3 %.** Rung 1's 17 s came
with *bit-identical* cycle counts over ~5 refreshes, so it is pure retrace removal with nothing else moving
— the one place in this comparison where the two effects are separated. That is what makes the verdict
quantitative rather than directional: the whole prize for eliminating retraces is ~88 s, and freezing spent
+109 s of cycles to collect it.
⚠️ **This prices the retrace AT 23040 CELLS and says nothing about how it scales.** The V-cycle is unrolled
over levels x sweeps x inner sweeps and the level count grows like log(n), so the per-retrace cost grows
with the mesh while this measurement does not. Do not carry the 3 % to a larger case.

**Freezing per RUNG is the surviving variant and is NOT measured.** Rung 1 shows a frozen partition is free
while the flow resembles its build state; rung 2 shows it degrades once the flow develops. Rebuilding at
each Reynolds boundary bounds the staleness to one rung. But note the ceiling before spending a run on it:
the whole retrace prize is ~88 s of a 1087 s gap to the incumbent, so this was never the lever.

### Sweep count, the zero-vector peel, and the forward solve cap — 2026-08-14

**FEWER SWEEPS WIN ON THIS MARCH: 2 against 4 is −16.8 % wall for +34.6 % cycles.** Full `bfs3d` marches,
native flow block, everything else identical (strength 0.25, no singletons, 5 levels, `max_coarse` 500,
block splitting, omega 1.0, 2 inner pressure sweeps, `refresh_on_cycles` 3):

| arm | steps | wall | Krylov cycles | final ‖R‖ | mid-span `x_r/h` |
|---|---|---|---|---|---|
| 4 sweeps | 60 | 3044 s | 327 | 8.076e-06 | 8.361 |
| **2 sweeps** | 63 | **2533 s** | 440 | 1.689e-06 | 8.361 |

The apply cost dominates strongly enough that buying cheapness with convergence pays. ⚠️ These arms part
at **step 10** (a materially weaker preconditioner returns different Newton directions), so this is not
the step-for-step controlled pair the frozen-coarsening comparison was; the direction is consistent
across both early rungs independently.

**✅ THE V-CYCLE MULTIPLIED THE LEVEL OPERATOR BY A VECTOR KNOWN TO BE ZERO, TWICE PER CYCLE (fixed).**
`_fixed_cycle_solve` began each pass with `b − A x` at `x = 0`, and every level's pre-smooth ran its
first sweep's `rhs − A g` at `g = 0`. The second is charged at **every level of every cycle**, and inside
a `fori_loop` the body cannot be specialized away. XLA does not fold it: measured 2.88 ms against 1.33 ms
for the peeled form on a 2.0M-nnz operator held as a jit constant.

`_VCycleOps` gained an optional `smooth_zero(level, b)`; families supplying none fall back to
`smooth(level, b, zeros)` and are byte-identical. **Exact, not approximate** — `smooth_zero(level, b)`
matches `smooth(level, b, zeros)` to **0.0** at every level at 1, 2 and 4 sweeps. Worth, on a 5-level
0.7M-nnz block:

| sweeps | 1 | **2 (shipped)** | 4 |
|---|---|---|---|
| saving | 36.0 % | **12.5 %** | 5.7 % |

The peel removes a fixed **two** operator applications out of `2·sweeps + 2`, so it is worth more at
fewer sweeps — which makes low sweep counts more attractive than the sweep table alone suggests. The
argument is the one the `pre_smooth=False` branch already made in its own comment, and the peel is the
one `_simple_correction` already did for the first pressure sweep one loop further in; it simply had not
been applied to the two outer sites.

**✅ A SINGLE FORWARD SOLVE WAS BOUNDED ONLY AT 60 RESTARTS, LONG PAST THE POINT ITS ATTEMPT WAS DOOMED
(fixed).** `cycle_budget` and `abort_above_inner_cycles` are both tested in `DualTimeStep`'s `cond`,
i.e. **between** inner iterations, so neither can stop a solve already running — and neither bounded any
of the 41–45 cycle solves in the archived marches. Meanwhile `forward_march` discards an attempt whose
worst solve passes `retry.on_cycles` without reaching target, so a solve at 11 corrected cycles has
already determined its work will be thrown away.

Measured over **671 solves across three marches**: no accepted attempt exceeded **9** corrected cycles,
discarded ones ran **39–45**, and the distribution is **empty between**. The archived logs put the
recoverable work at ~404 s of a 2533 s march.

**Two traps decide the constant, and both rule out the obvious choice of 10:**
- **`max_restarts` is in RAW `lineax` restarts** (a fixed +2 per solve); `retry.on_cycles` is in
  **corrected** cycles. A corrected cap of 12 is `max_restarts = 14`.
- **The march's test is `max_inner_cycles > retry.on_cycles`, STRICTLY.** A cap landing exactly on the
  threshold does not trip the retry, so the step **accepts the truncated, non-converged direction**
  instead of escalating β — turning a doomed attempt into a bad accepted step. The corrected cap must
  be strictly above the threshold.

Shipped as `coupled_amg_continuation(forward_max_restarts=…)`, library default unchanged at 60; the case
derives it as `retry.on_cycles + 4` from a single scaled constant so the two cannot drift.
⚠️ **The 671-solve distribution describes UNCAPPED solves.** Whether a truncated direction ever needs an
extra escalation rung is not answerable from it, and the first capped march is the first evidence.

**❌ THE FIXED SHAPE LADDER IS DEAD — cycles IDENTICAL to the last step, wall +52.6 %.** Full march at
2 sweeps with `shape_headroom` 1.25 against the 2-sweep control: **431 cycles against 431** through step
62, wall 3808 s against 2495 s. The padding is provably inert (which is the mechanism working exactly as
designed) and unaffordable, because it is charged per **application** (~440 cycles) while the retrace it
avoids is charged per **refresh** (~26). The 25 % headroom hits every level three ways — operator
matvecs, vector-length work in every diagonal scaling, and the dense coarse solve, which goes as the
**square** of its size. Do not revive it at this problem size; the minimum viable headroom is set by how
far the partition genuinely moves (~12 %), which still costs several times the retrace saving.

**⚠️⚠️ AND THE RETRACE WAS NEVER PRICED CORRECTLY — the recompile was UNCONDITIONAL. Read this before
citing any retrace figure.** The flow arm built `jax.jit(lambda …)` on a **fresh lambda every refresh**,
closing over the hierarchy and the smoother pieces. A new `jax.jit` object starts with an empty cache, so
it recompiled **whether or not the coarsening moved**. Consequences that still stand:
- **`refit`'s 17 s rung-1 saving was HOST AGGREGATION, not retrace**: it skips `_mis_aggregate` and
  friends but still rebuilt the jit. So the "~3.4 s per retrace, ~88 s per march, ~3 %" figure recorded
  above is an attribution error; that time is Python, not XLA.
- **Compile was a SEPARATE cost that neither refuted experiment removed**, additional to the 88 s rather
  than part of it.

**✅ INDEPENDENTLY CORROBORATED, by a separate implementation and a separate measurement.** The same
defect was diagnosed and fixed a second time, in parallel and without sight of the above: a different
cycle structure (the hierarchy and per-level state passed to a jit built once in the constructor rather
than to a module-level `filter_jit`), measured on a **different** fixture — a 8000-dof synthetic saddle
at the case's own flow settings, 3 repeats on a quiet machine:

| one refresh, min-max of 3 | refactor | first apply after | total (min) |
|---|---|---|---|
| closure | 0.049-0.050 s | **0.152-0.187 s** | 0.202 s |
| argument | 0.047-0.048 s | **0.005-0.005 s** | **0.051 s** |
| trailing block, either way — the CONTROL | ~6.5 s | 0.002 s | ~6.45 s |

**3.93x**, against the 3.3x measured above at the real block's shape — consistent in direction and
magnitude from two independent implementations. The control is the informative row: the trailing block,
which already passed its hierarchy as an argument, moves **1.01x**, so the effect is the recompile and
not the setup. Note also the *spread*: the closure form's first apply varies 0.152-0.187 s (compile-time
variance) where the argument form is 0.005 s with none.

### The peel and the cap ON A MARCH — the cycles land, the seconds mostly do not (2026-08-14)

Both changes marched at 2 sweeps against the 2-sweep control, native flow block, everything else equal.
The first attempt's wall clock is **void** — pytest tiers, ruff and JAX probes ran on the same machine
throughout it — so it was re-run on a quiet one. Cycle counts and trajectories from both agree, being
contention-immune.

| arm | steps | wall | step cycles | **solve cycles** | max single solve | mid-span `x_r/h` |
|---|---|---|---|---|---|---|
| control, 2 sweeps | 63 | 2533 s | 440 | **622** | 45 | 8.361 |
| peel + cap (contended, wall VOID) | 63 | 3100 s | 440 | 529 | 12 | 8.361 |
| **peel + cap (clean)** | 63 | **2486 s** | 440 | **529** | **12** | 8.361 |

**✅ THE CAP DOES EXACTLY WHAT IT WAS SIZED TO DO, AND COSTS NOTHING IN QUALITY.** Three solves ran
41/43/45 cycles in the control and were truncated to 12; **no solve exceeded the cap**; total solve
cycles fell 622 → 529 (**−93, −15 %**). The accepted trajectory is **identical on all 63 steps** with the
same root and the same reattachment length, so a truncated direction cost **no extra escalation rung** —
the one risk the 671-solve archive could not speak to, since every solve in it ran uncapped.

**⚠️ READ THE STEP TABLE'S CYCLE COLUMN CORRECTLY OR THE CAP LOOKS INERT.** It reports the **accepted**
attempt only, and the cap truncates **discarded** ones — so identical step-table cycles (440 in every
arm) is what a *working* cap produces, not evidence it never fired. The effect is visible only in the
per-inner tables. This was misread once here before the inner-table parse settled it.

**❌ BUT A 15 % CYCLE REDUCTION BOUGHT 1.9 % OF WALL, AND THE PEEL IS INVISIBLE AT MARCH SCALE.** Per
rung: **+1.0 % / +4.9 % / −5.6 %**. The whole saving sits in rung 3, where the cap's three solves live
(~85 s, against ~206 s that a 2.22 s/cycle cost model predicts). The peel measured **12.5 % of the
V-cycle** on a 5-level 0.7M-nnz synthetic and should have shown in every rung; rungs 1 and 2 are
slightly *slower*.

**Two candidate explanations were offered, and the second is now MEASURED to be sufficient on its own:**
- the synthetic does not transfer to the real 21M-nnz block (plausible — a cost model on synthetic
  sparsity already over-predicted itself 6× once here, real mesh-ordered matrices being far more
  cache-friendly); or
- **the instrument's own spread swamps a 12.5 % effect.** ✅ This one is now established, so the peel's
  disappearance needs no further explanation — though it does not *refute* the first, which remains
  untested.

**✅ THE NOISE FLOOR IS MEASURED, AND IT IS LARGE (2026-08-14).** `BFS3D_PROBE_SPLIT=1` at `state-00067`,
β = 0.1, shipped column reach, on a machine with **no competing compute**. The decisive number is an
*internal control*: the two monolithic ILU(0) arms in one run are the **same preconditioner built twice**,
so their apply must cost the same — and they measured **193 ms and 221 ms, 14 % apart, inside a single
quiet process.**

| arm, β = 0.1 | jacobian product | preconditioner | cycles |
|---|---|---|---|
| monolithic ILU(0), march solver | 37 ms | **193 ms** | 1 |
| monolithic ILU(0) | 36 ms | **221 ms** | 2 |
| `split flow/ilu0` | 36 ms | 118 ms | 2 |
| `omega10` — the shipped native bundle | 36 ms | 106 ms | 9 |

**So a per-application timing from this probe cannot resolve anything below roughly 15 %**, and the
12.5 % peel is under that floor. The Jacobian product is the clean counter-example: it is the *same*
computation in every arm and held 36–37 ms across the four (±3 %), which shows the spread lives in the
preconditioner apply rather than in the timer.

⚠️ **AND THE CYCLE COUNTS ARE EXACTLY REPRODUCIBLE**, which is the other half of the result and the
reason to keep quoting them: a second identical invocation returned **1 / 2 / 2 / 9 cycles** and residuals
agreeing to *every digit*, while its timings moved by up to 21 %. Cycles are contention-immune; seconds at
this granularity are not.

⚠️ **A TRAP THIS ALMOST WALKED INTO, worth more than the number.** Run 1 has the native bundle at 106 ms
against the incumbent's 118 ms — the native apply looking **cheaper**, which would overturn the recorded
"~1.45× more expensive per application". The second run has 128 ms against 112 ms, i.e. **the ordering
reverses**. At 2 sweeps the two are indistinguishable per application and **the sign cannot be called**;
what can be said is only that the 1.45× figure was measured at **4 sweeps** and does not describe the
2-sweep bundle the case now ships. A single run would have produced a confident, wrong headline.

⚠️ **Scope: this measures the PER-APPLICATION instrument, not march wall.** Whether a whole march repeats
to better than a few percent is still unmeasured, and the "±1.0 % / +4.9 % / −5.6 %" per-rung figures
above are still uninterpretable for that reason. What is settled is that the cheap discriminator is too
blunt for a 12.5 % effect, so reaching for it again on a sub-15 % question is wasted time.

### Over-relaxing the SIMPLE correction — real, small, and one step from a cliff (2026-08-14)

**The grammar capped `omega` at 1.0, so no arm in this campaign had ever tested over-relaxation.**
`_leading_inverse` read `omega = 1.0 if "-o10" in rest else 0.7` — a binary token — while the class always
took a float. So the recorded "omega 1.0 is worth a cycle" was measuring the **top of the grid**, not an
optimum. `-oNN` (NN/10) now spans it.

*Configuration:* `bfs3d` `state-00067`, **beta = 0 on operator and preconditioner** (the adjoint's
operator), **uniform** column reach, real right-hand side `-R(state)`, GMRES restart 15 to rtol 1e-8 on the
**TRUE** residual, 58-restart cap. Everything but `omega` pinned to the shipped bundle: 2 sweeps x 2 inner,
strength 0.25, no singletons, 5 levels, `max_coarse` 500, per-cell block splitting, ILU(0) trailing.

| omega | 0.7 | **1.0 (shipped)** | **1.2** | 1.4 | 1.6 | 1.8 | 2.0 |
|---|---|---|---|---|---|---|---|
| cycles | 30 | **20** | **19** | 58 cap | 58 cap | 58 cap | 58 cap |
| TRUE rel | 1.99e-10 | 7.24e-10 | 5.62e-10 | 7.1e-05 | 1.2e-02 | 1.8e-02 | 5.0e-02 |
| solve | 84 s | 68 s | **66 s** | 136 s | 139 s | 143 s | 141 s |

**Confirmed in direction, REFUTED in magnitude.** The optimum is above 1.0 and is worth **one cycle (~5 %)**
— not the 1.8x a synthetic saddle predicted. **The synthetic's optimum, 1.4–1.6, is exactly where the real
block stops converging.**

**⚠️ THE DERIVATION IS WRONG, AND PREDICTS ALMOST EXACTLY THE WORST POINT.** The argument was: the Frobenius
velocity predictor is a least-squares fit and so shrinks by construction (median 0.53), SIMPLE's dropped
neighbour terms shrink again, and a correction behaving like `gamma * A^-1` wants a Richardson factor near
`1/gamma ~ 1.9`. Measured, 1.8 is the second-worst arm on the ladder. The premise that fails is
**uniform** shrinkage: the SIMPLE correction's error varies by mode, so one scalar cannot compensate for
it, and pushing the scalar far enough to fix the worst-shrunk mode amplifies the rest. That accounts for
both halves of the result — a small gain from mild over-relaxation, and a hard cliff far below the derived
value.

**DO NOT MOVE THE DEFAULT — 1.0 STAYS.** 1.2 buys 5 % and sits ONE GRID STEP from a cliff that costs
convergence outright, and past the cliff the degradation is monotone (7.1e-05 → 1.2e-02 → 1.8e-02 →
5.0e-02), i.e. genuine amplification rather than a near-miss on the restart budget. The synthetic's one
durable finding was that the optimum **moves with operator hardness**, which makes a fixed 1.2 fragile
across a march that visits a wide range of operators. A per-level, per-refresh derived relaxation would be
a different proposition, and at a 5 % ceiling it is not worth building.

**⚠️ METHOD, worth more than the result: RUN THE WHOLE LADDER, INCLUDING PAST WHERE YOU EXPECT THE
OPTIMUM.** The arms 0.7 / 1.0 / 1.2 alone read as monotone improvement and invite pushing further — and
the next step diverges. A sweep that stops at its best point cannot see a cliff one step beyond it.

**⚠️ AND THE SELF-CHECK EARNED ITS KEEP.** The first attempt ran at the case's shipped column reach and the
control stopped at a TRUE relative residual of **4.779e-03**, so the harness refused to report any arm.
That value is recorded elsewhere in this file as the benign signature of the loose march-solver stop at
`(3,3,3,3,2,2)` — so without the gate there would have been seven plausible omega numbers measured against
a control sitting at 5e-3. Every beta = 0 measurement in this campaign is at **uniform** reach for this
reason; the shipped reach doubles the incumbent's zero-shift cost (22 cycles against 11) and is a separate
axis.

**Unrun:** the same ladder at beta = 0.1. A shift raises diagonal dominance and should pull the optimum
*toward* 1.0, so the march's own regime is where over-relaxation has least to offer; what it would settle
is whether the cliff ever descends BELOW 1.0, which would put the shipped default near an edge.

### The trailing ablation, and the two structural ideas it points at (2026-08-14)

**The question: how far below the coupled cycle count can a better TRAILING inverse alone take it?** The
figure that had stood in for this — "2 restart cycles against a floor of 1" — is a **standalone** solve of
the `[k, ω]` block, which production never performs: one Krylov iteration runs on all six fields with a
block-triangular `M`, so the count is a property of the whole preconditioned operator and of the coupling
the split discards, not of either block alone.

*Configuration:* `bfs3d` `state-00067`, **β = 0**, **uniform** column reach, real right-hand side
`−R(state)`, GMRES restart 15 to rtol 1e-8 on the TRUE residual, 58-restart cap, **leading inverse held at
PETSc ILU(0) in every arm**.

| trailing inverse | cycles |
|---|---|
| PETSc ILU(0) (control, reproduces the recorded 11) | **11** |
| **native nodal, 4 sweeps (SHIPPED)** | **11** |
| native nodal, 2 sweeps | 16 |
| native nodal, 1 sweep | 28 |
| damped Jacobi | 38 |
| near-exact factorization (diagnostic bound) | **58 cap, 2.5e-05 — WORSE than all of them** |

**Three results.** The shipped native trailing inverse **matches PETSc ILU(0)** in the coupled system,
which nothing on record established (the recorded 11 used ILU(0) on *both* halves). **Cutting its sweeps
is measured harmful** — 4 → 2 costs +45 % of coupled cycles, 4 → 1 costs +155 % — which settles the
"4 sweeps is an unexamined default" question in the opposite direction, and is what the record's own
(false) claim that this case already shipped at 1 sweep would have cost. And **more quality does not
help**: nothing in this ablation supports spending on the trailing block's accuracy.

**✅ THE BOUND ARM IS NOW INTERPRETABLE, AND THE BOUND IS REAL — measured per field 2026-08-14.** Its
residual had been reported only as **1.693e-06 globally on a random right-hand side**, and this block's k
and ω rows differ by some eight orders of magnitude, so a global norm is ~100 % ω and would report a
factorization that solves ω beautifully and k not at all as "near-exact". Reported per field, at the
standing configuration of this section (`state-00067`, β = 0, uniform reach, 58-restart cap, leading
inverse held at PETSc ILU(0)):

| trailing factorization, random right-hand side | relative residual |
|---|---|
| global | 1.693e-06 |
| **f0 = k** | **1.936e-06** |
| **f1 = ω** | **1.416e-06** |

**The two fields agree to within 1.4×, so the factorization is as tight in k as in ω** and reading (2) —
"the bound was never tight in k, so it bounds nothing" — is **REFUTED**. Reading (1) stands: an exact
inverse of an ill-conditioned block (recorded cell-block condition ~1e12, ‖B⁻¹‖ ~9.5e8) **amplifies the
coupling the split discards**, where a fixed-cycle V-cycle is bounded and cannot. This has a precedent on
the flow side: inverting the Schur *more* accurately with ×2/×4/×8 V-cycles measured ρ 41.6/48.7/48.5,
strictly worse, and was read as "the operator being inverted is the wrong one".

The same run reproduced every number around it — monolithic ILU(0) 11 cycles / 8.474e-11, `split
flow/ilu0` 11 / 6.393e-11, and the exact arm 58 cap / 2.461e-05 — so the arm is not a one-off.

⚠️ **Quote the CYCLES and the residuals from this run, not its seconds.** Another worktree started a full
test tier partway through it, so the control's timings were taken on a quiet machine and the exact arm's
were not — the two are on opposite sides of that boundary and are **not comparable**. Cycle counts and
residuals are contention-immune and are the load-bearing evidence here. That a near-complete factorization
of this block is far more expensive to build than an incomplete one is not in doubt on other grounds, but
this run does not price it.

**So "the trailing inverse's job is to be WELL-BEHAVED, not accurate" is now a supported claim rather than
a reading.**

⚠️ **BUT THE MECHANISM NAMED IN READING (1) IS WRONG, and the cross-coupling measurement below refutes
it.** Reading (1) says the exact inverse "amplifies **the coupling the split discards**". Measured, the
discarded triangle `∂R_flow/∂turb` is **0.09 %** of the diagonal blocks — there is essentially nothing
there to amplify. What the split *retains*, `∂R_turb/∂flow`, is **83 %**, and that is what feeds the
trailing block's right-hand side. **The surviving mechanism, offered as a hypothesis and NOT measured:** an
exact `B⁻¹` on an ill-conditioned block (cell-block condition ~1e12, ‖B⁻¹‖ ~9.5e8) amplifies whatever error
reaches it, and the error that reaches it is the *approximate flow solve's*, arriving at full strength
through the retained 83 % coupling. A fixed-cycle V-cycle is bounded and cannot do that. Note the flow-side
precedent still stands on its own terms (inverting the Schur more accurately measured ρ 41.6/48.7/48.5,
strictly worse); what changes is which coupling carries the error here.

⚠️ **One scope limit, stated because the arm is a bound.** The residual is measured on a *random* right-hand
side, not on the `r_turb − C·x_flow` the coupled solve actually presents to this block. That is the right
choice for asking "is the factorization tight in k" — an unbiased probe — but it is not evidence about the
per-field scale of the right-hand side the split really sees.

**⚠️ ORDERING (flow-first against turbulence-first) is measured and is nearly a tie**: 11 against 13 at
β = 0, and **4 against 4** at the forward operating point, where the operator discriminates between
neither. Flow-first ships. The ordering decides *which* cross-coupling is discarded — flow-first retains
`∂R_turb/∂flow` and drops `∂R_flow/∂turb`.

**✅ THE TWO CROSS-COUPLING NORMS ARE NOW MEASURED, and the asymmetry is ~900x (2026-08-14).**
`field_coupling.py` at `state-00067`, on the symmetrically equilibrated Jacobian (which is what makes a
block norm comparable across fields whose units differ by orders), 39.2M nonzeros:

| partition (leading \| trailing) | drops | keeps |
|---|---|---|
| **`[u,v,w,p] \| [k,ω]` — the shipped ordering** | **0.09 %** | **83.10 %** |
| `[k,ω] \| [u,v,w,p]` | 83.10 % | 0.09 % |
| `[k] \| [ω]` | 0.19 % | 12016 % |
| `[ω] \| [k]` | 12016 % | 0.19 % |

Both are Frobenius norms of the off-diagonal triangle against the diagonal blocks the two hierarchies
invert. **`∂R_turb/∂flow` dominates `∂R_flow/∂turb` by roughly 900x, so flow-first's win is explained** —
it retains the overwhelmingly larger triangle. The raw grid says why, and it is physically ordinary: the
flow equations depend on turbulence only through the eddy viscosity (`∂R_u/∂k` 26.8 and `∂R_u/∂ω` 0.218
against a `u` diagonal of 244; `∂R_p/∂ω` is 1.6e-03), while the turbulence equations depend on the flow
through production terms in the velocity gradients (`∂R_ω/∂p` 2.12e+04, `∂R_ω/∂v` 1.29e+04 against an `ω`
diagonal of 172). The k↔ω pair repeats the pattern far more extremely: `∂R_ω/∂k` 2.99e+04 against
`∂R_k/∂ω` 0.474, a 63000x asymmetry, which corroborates the recorded k↔ω ordering result independently.

**⚠️ AND READ THE ORDERING TIE AGAINST THIS, because together they say something stronger than either
alone: turbulence-first DISCARDS the 83 % triangle and costs only TWO CYCLES (13 against 11).** So even the
dominant cross-coupling is nearly irrelevant to how fast the preconditioned Krylov solve converges. **The
split's discarded triangle is not what limits it, in either direction** — which is a much sharper statement
than "the ordering is a tie", and it is the fact that kills the alternation idea below.

⚠️ **A Frobenius norm is not an operator norm**: it bounds the block's action rather than equalling it, so
"0.09 %" bounds how much the discarded triangle can matter and does not prove it is inert. The ordering
measurement is the independent leg, and the two agree. The run shared the machine with other test tiers;
block norms are contention-immune, so only its wall clock is affected and nothing here depends on it.

**❌ ALTERNATING THE TWO BLOCKS (block Gauss–Seidel instead of one forward substitution) IS CLOSED BY THE
TWO MEASUREMENTS ABOVE — do not build it.** The idea was that the apply is ONE pass — one flow V-cycle, one
sparse coupling product, one trailing V-cycle (`cycles=1` on both inverses) — so the flow solve never sees
turbulence, and a second pass would pick up the discarded coupling. Its whole value rests on there being
something in `∂R_flow/∂turb` worth capturing, and there is not:

- **The discarded triangle is 0.09 %** of the diagonal blocks. A second pass would buy that, at best.
- **The retained triangle is 83 %, and DISCARDING it costs two cycles** (turbulence-first, 13 against 11).
  So even if the two triangles were swapped in size, the prize would be about two cycles.
- It also needed `∂R_flow/∂turb` **assembled**, which the split does not form — real work, for that prize.

The design was otherwise sound and is worth remembering as a *technique* rather than as a candidate here: a
fixed number of alternations keeps `b → x` linear (which the non-flexible outer GMRES requires) and the
transpose stays closed-form, being the reversed product of transposes that
`BlockTriangularFieldSplit.apply(transpose=True)` already performs for one sweep. What is refuted is
alternation **on this split**, not alternation.

**Where that leaves the trailing block:** its quality buys nothing (the ablation), its coupling buys about
two cycles (the ordering), and the coupling it discards is 0.09 % (the norms). Three independent
measurements now say the same thing, so **stop looking for coupled cycles in the trailing half** — the
remaining reason to touch it is the coarse-grid **scalability** argument, which is about how the dense
coarsest solve grows with the mesh and not about this case's cycle count.

### The zero-vector peel reaches the TRAILING block too (2026-08-14)

`_VCycleOps` has carried an optional `smooth_zero` since the flow block's peel, but the point smoother
the transported scalars run had no such specialization, so the trailing V-cycle kept multiplying its
level operator by a vector known to be zero — once per level per cycle, on the pre-smooth. `_jacobi_smooth_zero`
supplies it and `convection_multigrid_solve` passes it, which reaches the trailing inverse and the frozen
velocity/Schur AMGs alike.

**Exact, and checked as such rather than argued.** `A x` at a zero vector is exactly zero in floating
point, so `b - A x` is `b` bit-for-bit; the peel is the same arithmetic with two operations removed. Pinned
two ways: `_jacobi_smooth_zero` against `_jacobi_smooth(..., zeros)` at 1/2/4 sweeps × both damping
settings (`test_the_zero_guess_peel_matches_the_general_jacobi_sweep_exactly`), and the whole
`fixed_cycle_solve` with and without it across sweeps × damping × cycle counts — **bit-identical
everywhere**.

⚠️ **The size is INHERITED, not measured here.** The flow-side peel measured 36 % / 12.5 % / 5.7 % of a
V-cycle at 1 / 2 / 4 sweeps on a 5-level 0.7M-nnz synthetic; the trailing block ships at **4 sweeps**, so
the low end of that range is the relevant one. And the same synthetic's 12.5 % is separately on record as
having failed to appear at march scale — see the noise-floor entry: a per-application effect of this size
is **below the measurement floor** on this case, so take the peel because it is free and exact, not
because a march will show it.

### ✅ THE DUAL-TIME INNER TOLERANCE WAS COSTING A THIRD OF THE MARCH (2026-08-14)

**`inner_tol` shipped at 1e-3 where `DualTimeStep`'s own docstring says a loose value is enough ("e.g.
0.05 -- the outer march re-solves each timestep anyway"). Fifty times tighter than documented, never
tested, and worth 33 % of the march.** Three full `bfs3d` marches, native flow block at 2 sweeps,
otherwise the case's own settings, on one machine at one commit with only `inner_tol` moved. **Same root,
same reattachment length (`x_r/h` 8.3611 mid-span, 12.53 full-span) in every arm:**

| `inner_tol` | steps | wall | Krylov cycles | inner iterations | final ‖R‖ |
|---|---|---|---|---|---|
| 1e-3 — as shipped | 63 | 2253 s | 445 | 211 | 1.689e-06 |
| **1e-2 — the new default** | **63** | **1510 s** | 338 | 166 | 1.828e-06 |
| 5e-2 — the docstring's value | **64** | 1568 s | **277** | **145** | 1.168e-06 |

**Why loosening is free: `G` and `R` are different quantities.** The inner loop drives
`G = R + β d (φ − φⁿ)`; the march is judged on `R`, and at the next anchor `R = −β d (φ − φⁿ)` — set by
**where φ lands**, not by how far `G` was driven. Two inner iterations place the step as well as three,
and the outer contraction never sees the difference. That is why the step count is identical at 1e-2
while a fifth of the inner Newton iterations disappear.

⚠️ **AND 0.05 IS PAST THE OPTIMUM — the arm with the FEWEST cycles and the FEWEST inner iterations is
the slower one.** It saves 61 cycles and 21 inner iterations against 1e-2 and loses on wall, because it
spends an **extra outer step** (64 against 63), and a step costs a Jacobian assembly and a refresh check.
**Read the failure mode: this knob does not diverge when pushed, it quietly trades inner work for outer
steps.** Nothing looks wrong — ‖R‖ ends *deeper* at 1.168e-06 and the reattachment length is unchanged —
so an arm judged on cycles, or on inner count, or on final residual, picks the slower configuration. Only
wall clock and the step count separate them.

**This is the second time in this file that running the ladder PAST the expected optimum was what
produced the answer** (the first was `omega`, where 0.7/1.0/1.2 read as monotone improvement and 1.4
diverged). There the cliff was loud; here it is silent. Both were invisible from the best measured point.

⚠️ **Each point is ONE march and this case still has no measured march-level noise floor.** The
1e-2/5e-2 wall gap is 4 %, which a single sample cannot resolve on its own — so that comparison rests on
the **step count and cycle count**, which are contention-immune, and not on the seconds. The 1e-3 → 1e-2
result does not need that care: 33 % of wall alongside a 24 % cycle drop at an identical step count.

⚠️ **The optimum is bracketed, not located.** The two points either side of it differ by 5×, and the
region between them is flat to 4 %, so a finer sweep has little to find. `BFS3D_INNER_TOL` and
`BFS3D_INNER_STEPS` are exposed for anyone who wants to try.

⚠️ **THIS DOES NOT OVERTURN THE RECORDED "TIGHTEN `inner_tol`" FINDING — it bounds it.** Elsewhere in
this file a *pitzDaily* (2D, ILUT, large-Δτ dual-time) march is recorded as hitting a residual floor and
over-developing because "at `inner_tol = 0.05` the implicit step is only 5 %-solved, so a large-Δτ
backward-Euler step on a half-solved system overshoots", with the fix being to tighten it. **That
mechanism is real and this sweep shows its onset:** the 0.05 arm is precisely where an extra outer step
appears. What the two together establish is that the threshold is **case- and Δτ-dependent, and that
"tight" meant about 1e-2 rather than 1e-3** — on `bfs3d` at this Courant ramp, 1e-3 bought nothing over
1e-2 while costing a third of the march. Neither number transfers to the other case; the *mechanism*
does. (That pitzDaily entry's own numbers were deleted as unrecorded, so only its mechanism was ever
citable.)

⚠️ **`INNER_STEPS` (5) is INERT and always has been** — the loop exits on the tolerance long before the
cap, at every arm above. It becomes a real knob only if the tolerance is loosened far enough for the cap
to start catching, which none of these arms reach.

### ⚠️ THE FORWARD SOLVE'S β = 0 CONTRACTION IS GEOMETRIC, NOT QUADRATIC — still undiagnosed (2026-08-14)

**A root at β = 0 IS reachable — see *"`jax.grad` RUNS ON THIS CASE"* above, where `rtol` 1e-4 is reached
and the gradient it yields is validated. What has never been explained is the RATE.** An earlier version
of this entry concluded the solve "cannot reach a root at any tolerance a gradient would want"; that is
**false** and is deleted. The rate finding below survives it, and is the part still worth chasing.

*Configuration:* `state-00067` (the converged root), started from its **physical** fields, β = 0
throughout, field split, shipped column reach, `coupled_amg_continuation` built once on concrete
parameters, `forward_rtol` 0.3.

From a converged root the solve decays **geometrically with FULL Newton steps** (`alpha` 1.000 at every
step) where a Newton step at a root should be quadratic. The per-step contraction, measured over 22 steps:

| step | 1 | 5 | 11 | 16 | 19 | 21 | 22 |
|---|---|---|---|---|---|---|---|
| contraction | 0.827 | 0.853 | 0.884 | 0.868 | 0.804 | 0.771 | **0.704** |

**It is roughly flat at 0.83–0.88 for the first ~16 steps and then improves**, reaching 0.70 by step 22
(relative residual 0.019) — mildly superlinear at the tail, **not** the constant rate a first reading took
it for.

**How far it actually gets, measured 2026-08-15 (this supersedes "1e-8 was not demonstrated; how many
steps is unknown"):** the guard in `ImplicitNewtonSolver` passes — i.e. the solve genuinely converged — at

| `rtol` | steps allowed | outcome |
|---|---|---|
| 2e-2 | 30 | reached in **22** steps |
| 1e-3 | 80 | **reached** |
| 1e-4 | 90 | **reached** |

so the reachable tolerance is at least **four orders tighter than the 2e-2 this entry was written at**.
1e-8 is still not demonstrated. ⚠️ The 1e-3 and 1e-4 runs ran with the per-step observer off
(`BFS3D_ADJOINT_SKIP_FORWARD=1`), so the step *counts* at those tolerances are bounded by the budget
rather than measured; the preconditioner-application totals over one gradient evaluation — **1070 → 1702
→ 1944** across the three rows — are the measured proxy.

**Two candidate causes were tested and BOTH are refuted, by controlled arms that share one trajectory:**

| arm | cycles / step | residual trajectory |
|---|---|---|
| `forward_rtol` 0.3 (the march's own), positivity floor 0 | **1** | 8.4766e-06, 7.0091e-06, 5.8031e-06, … |
| `forward_rtol` **1e-6**, positivity floor 0 | **4–8** | **identical to 5 significant figures** |
| `forward_rtol` 1e-6, positivity floor **1e-8** (the case's) | 4–8 | **identical to 5 significant figures** |

- **NOT the linear solve.** Tightening it five orders costs 4–8× the Krylov work per step and moves the
  nonlinear trajectory *not at all*.
- **NOT the k-positivity limiter**, which was the obvious suspect since `coupled_amg_continuation`
  applies `step_limit=positive_k_limit(...)` **unconditionally**. The case's 1e-8 floor reproduces the
  unfloored trajectory exactly.

**So the rate is set by the Jacobian or the residual itself, and that is UNDIAGNOSED.** The residual path
carries no `stop_gradient` (every one in `turbulence/coupled.py` is on the preconditioner or the shift,
which is correct), so a wrong Jacobian is not the cheap explanation. A near-constant rate immune to both
levers is the signature of a fixed-point iteration rather than a Newton one; the next things to look at
are whether the SST closure's blending/limiting makes the residual only piecewise differentiable near
this state, and whether the step actually applied is the one `alpha` reports.

**What it costs, now that the consequence is priced rather than feared.** The rate does not block the
adjoint — it makes a *converged* gradient expensive: reaching `rtol` 1e-4 takes roughly twice the
preconditioner applications of `rtol` 2e-2 (1944 against 1070), and a validated finite-difference check
needs that tight root (at 2e-2 the check reports a 23 % mismatch that is entirely the difference's fault).
So this is a cost and accuracy problem, not a reachability one.

⚠️ **THE ADJOINT-STAGNATION ENTRY THAT USED TO SIT HERE IS DELETED — every claim in it is now false.**
It said `jax.grad` had never succeeded, that the transpose solve stagnated even at a loose root, that both
halves of the adjoint were broken, and that two arms (the Krylov settings, and the native leading inverse)
remained unrun. The stagnation was a **solver-settings artifact reachable only through an API gap that is
now closed** (`solve_coupled(adjoint_solver=…)`), the gradient runs and is validated to 1.9e-04, and both
arms have been run. See *"`jax.grad` RUNS ON THIS CASE, AND IS VALIDATED"* above for the costs and the
native-versus-PETSc comparison. Three findings from that entry are **not** superseded and are kept here
because they cost real time to learn:

- **❌ The column reach is not a factor in the transpose solve** — measured, not assumed. At **uniform**
  reach (`BFS3D_COLUMN_REACH=0`) the forward objective came back bit-identical to the shipped reach's
  (5.337542799e+04), which is the expected control: the reach is a preconditioner-only approximation and
  cannot move the root.
- **⚠️ `on_step` CANNOT BE USED UNDER `jax.grad`, and it is the observer that makes a march readable.**
  `solve_coupled` raises rather than letting it through: `refresh_trigger` / `step_control` / `on_step` /
  `on_checkpoint` / `precondition_step` / `retry` all drive a forward-only **eager** march that
  steps in Python on concrete residual norms, which a differentiation tracer cannot flow through. The
  adjoint is refresh-independent, so the single-stage solve returns the identical gradient; the cost is
  only that **a differentiated evaluation is silent**. The probe therefore builds the objective twice,
  watched and unwatched, and differentiates the unwatched one — and the transpose solve's own progress is
  reported by counting preconditioner applications instead (see the instrumentation note above).
- **⚠️ Two probe defects, either of which produces a confident wrong number**, and neither announces
  itself:
  - **`layout.unpack` is NOT the physical state.** The checkpoint stores the *solved* variables, whose
    scalar blocks are `log(omega)` under this case's log-variable transport, while `solve_coupled` takes a
    **physical** initial condition and maps it in itself. Feeding it the unpacked blocks applies the log
    twice: the solve starts at |R| ~1.4 instead of ~1e-05, converges to *something*, and returns a finite
    gradient of the wrong problem. Use `coupled.physical_fields(state)`.
  - **A probe that builds its own continuation silently gets LIBRARY defaults**, not the case's — here the
    positivity floor (case 1e-8, library 0). It made no difference, but only because it was checked.

  Both were caught only because the probe prints one line per outer step. It did not, at first.

### The gap is CONVERGENCE, not cost — measured, and it reverses the priorities

**The per-iteration split had never been measured, and both directions of inference about it were
wrong.** `BFS3D_PROBE_SPLIT=1` times the exact matrix-free Jacobian product against the preconditioner
apply, separately, on every arm. At β = 0.1:

| arm | jacobian product | preconditioner | cycles |
|---|---|---|---|
| `split flow/ilu0` (incumbent) | 34 ms | **121 ms** | 2 |
| native, 4 sweeps x 2 inner | 38 ms | **175 ms** | 8 |
| native, block splitting | 49 ms | 212 ms | 7 |
| native, 8 sweeps | 46 ms | 319 ms | 5 |

**The native preconditioner is only ~1.45x more expensive per application and needs 4x the cycles.** So
the wall-clock gap is convergence, and **a lever that buys cycles at no cost is worth about four times an
equally-sized cost reduction.** Two independent audits had inferred the Jacobian product to be roughly
half of an iteration and concluded preconditioner work was capped at half the gap; it is **14 %**. That
inference chained three estimates, one resting on a wrong nonzero count, where the direct measurement
takes seconds — take it before ranking any cost work.

⚠️ **TWO QUALIFICATIONS ON THE 1.45x, added 2026-08-14; the CONCLUSION survives both.**
- **It is a FOUR-SWEEP number, and the case ships TWO.** Re-measured at the shipped bundle (`omega10`:
  2 sweeps x 2 inner, strength 0.25, no singletons, 5 levels, `max_coarse` 500, block splitting,
  `omega` 1.0), the native apply came out 106 ms against the incumbent's 118 ms in one run and 128 ms
  against 112 ms in a second — so at two sweeps **the two are indistinguishable per application and the
  ordering is not callable.** Do not quote 1.45x for the shipped configuration.
- **Every number in the table above carries the ~15 % instrument spread** measured under *The noise floor*
  earlier in this file (the same preconditioner built twice in one quiet process timed 193 and 221 ms), so
  the four rows rank reliably only where they differ by much more than that — which 121 against 319 does
  and 121 against 175 does not.

**The conclusion is unaffected and is if anything stronger**: the cost half of the trade is small or
absent, the cycle half is 4x, so a lever that buys cycles still dominates one that buys arithmetic.

**TWO FREE QUALITY LEVERS, AND THEY COMPOSE.**

`omega` — the relaxation on the whole SIMPLE correction — was 0.7, and **no arm in this campaign had ever
varied it; there was no token.** It stacks on the Frobenius per-row relaxation the velocity predictor
already carries (median 0.53) for an effective ~0.37. This is the **third** time the same over-damping has
been found here, after the pressure relaxation (worth 1.6x) and the transported scalars' smoother
(10 cycles against 2). Undamping is worth one cycle at *both* sweep counts, so it is not a single point.

| configuration (β = 0.1, 4 sweeps x 2 inner) | cycles | solve |
|---|---|---|
| baseline, omega 0.7, scalar diagonal | 8 | 34 s |
| + per-cell block splitting | 7 | 32 s |
| + omega 1.0 | 7 | 31 s |
| **+ both** | **6** | **29 s** |

Each is worth one cycle alone and together they are worth two, so they correct **different** deficiencies
rather than being two routes to the same one. Both are properties of the **velocity splitting** — which is
where the balance measurement pointed (`||S|| = 1.449` against `||E|| = 1.029`) before that reading was
mistakenly talked down.

**AT β = 0 THE NATIVE PRECONDITIONER NOW CONVERGES IN FEWER ITERATIONS THAN THE INCUMBENT.** The
composition holds at zero shift, so it is not a march-only effect:

| arm at β = 0 | cycles | solve |
|---|---|---|
| `split flow/ilu0` | 11 | 31 s |
| native, 8 sweeps, omega 0.7, diagonal | 11 | 76 s |
| **native, 8 sweeps, block + undamped** | **7** | **57 s** |

**1.84x at the adjoint point** (from 2.45x), and **2.4x on the march** (from 3.0x). The asymmetry is the
structural claim with the native side finally measured at its best: at zero shift the native hierarchy is
the *better* preconditioner and loses only on arithmetic; once a shift is present an incomplete-LU is
nearly exact in two triangular passes and no multigrid quality catches it.

⚠️ **SIMPLEC FAILS HERE (true relative residual 1.0 at both relaxations) BUT THE RECORDED MECHANISM WAS
WRONG — and the measurement that would settle it has not been run.** Its velocity coefficient is
`1/(a_P - sum a_nb)`, which is this matrix's **row sum**. The first reading was that a conservative
discretization makes the interior row sum vanish — convection cancels by continuity, diffusion telescopes
by conservation — so the coefficient degenerates and that is why the arm failed. **That reading does not
survive its own numbers.**

The guard fell back to the Frobenius diagonal when a row sum dropped below **0.1** of its own diagonal,
and the arm ran at **beta = 0.1**. For an interior cell the row sum is essentially the shift `beta d` with
`d ~ a_P ~ diagonal`, so `row sum / diagonal ~ beta = 0.1` — sitting exactly on the cutoff. The reported
"46 % of fine rows fell back" therefore measures rows scattered either side of a threshold set, by
coincidence, at the physical value. It is not evidence of a degenerate operator.

On the rows where SIMPLEC did apply the coefficient was about `1/(beta d) = 10/d` against Jacobi's `1/d`.
**That is a legitimate SIMPLEC coefficient for this shift, not a pathology** — a larger, consistent `d` is
the method's entire point, and it is why SIMPLEC needs no pressure under-relaxation as a solver.

**The likelier cause is specific to SMOOTHING rather than solving.** In a segregated solver the larger `d`
is bounded by the outer iteration and the velocity under-relaxation. As a fixed-sweep level smoother
there is no outer control, and a velocity coefficient ten times larger pushes the error operator
`I - omega M A` past unit spectral radius — the same failure shape as an undamped **Jacobi** pressure
sweep (3.435e-01), one block over. Note this is consistent with SIMPLEC converging perfectly well as a
solver without relaxation; the two claims do not conflict.

**To settle it, measure `||I - F~^-1 F||` under the SIMPLEC diagonal** (`_splitting_balance` already
computes this quantity for the other splittings — but ⚠️ it currently has no caller, so wiring it back to a
switch comes first) with the fallback threshold dropped far below `beta` so
SIMPLEC is genuinely in force. Above the Frobenius diagonal's 1.449 means it amplifies and the direction
closes for that reason; comparable or lower means something else killed the arm and it deserves another
look. **Until that is run, treat SIMPLEC as failed-but-unexplained, not as closed.**

**SIMPLER is a separate no, on cost**: it targets the smooth global pressure mode, which is what the
coarse grid already owns, at roughly double the pressure work already measured as not paying.

### Q&A on the remaining levers — three closed, one small win

Four questions were put to measurement together, because each earlier answer had been taken with another
axis pinned. All at β = 0.1 on `state-00067`, strength 0.25, 5 levels, no singletons, 60-restart cap;
matched incumbent `split flow/ilu0` is **2 cycles / 11 s**.

**A STRONGER SPLITTING IS A SMALL, FREE WIN — the only Pareto improvement found.** Replacing the velocity
predictor's scalar diagonal with a per-cell block form saves a cycle at both sweep counts and does not cost
wall clock, despite tripling the nonzeros in `dg` and hence in the Schur:

| sweeps | splitting | cycles | solve |
|---|---|---|---|
| 4 | scalar diagonal | 8 | 34 s |
| 4 | **cell block** | **7** | **33 s** |
| 8 | scalar diagonal | 6 | 43 s |
| 8 | **cell block** | **5** | 40 s |

Five cycles is the fewest any native arm has reached at this shift. Everything else measured here trades
cost against cycles; this is the one change that improved the smoother's *quality* at fixed price.

⚠️ **The block form MUST be the Frobenius-optimal one, and the exact inverse of the cell's own block is
much WORSE than the scalar diagonal it would replace.** Measured `||I - F~^-1 F||` per level: scalar
Frobenius diagonal 1.449 / 1.230 / 1.482 / 1.275, against the cell-block **exact inverse** 12.03 / 10.53 /
8.64 / 5.76 — four to eight times worse. Inverting a cell's own block ignores the row's dominant
off-diagonal mass, which on a convection-dominated operator is most of it. This is the same trap as plain
Jacobi versus Eq. (39) on the diagonal, one block size up.

⚠️ **The Frobenius block is `M_i = F_ii^T (R_i R_i^T)^-1`, and getting the transpose wrong is silent at one
field per cell.** Minimizing `||I - M F||_F` over block-diagonal `M` decouples by cell, with `R_i` the
cell's row block. Computing `F_ii (R R^T)^-1` instead — the same expression without the transpose — is
correct only when the cell's own block is symmetric, which a velocity block is not. Measured on a
nonsymmetric test: correct 1.048, transposed 1.246, plain exact inverse 1.101, so the error makes the
"optimal" form worse than simply inverting the block. In the solver it took the arm from **7 cycles to 58**
(no convergence). It reduces exactly to the scalar `F_ii / ||F_i||^2` at one field per cell, so a scalar
reduction check cannot catch it — check against a brute-force minimizer on a nonsymmetric case instead.

**⚠️ AGGRESSIVE COARSENING x W-CYCLE x DEPTH — CLOSED, and the earlier one-axis results were RIGHT.** The
suspicion was that each had been measured with the other pinned at the value that makes it fail, and that
a W-cycle needs the genuinely cheap deep levels aggressive coarsening produces (the configuration the
literature runs at eight to fifteen levels). Tested jointly, it does not hold:

| arm | cycles | solve |
|---|---|---|
| **plain, V-cycle, 5 levels** | 8 | **34 s** |
| aggressive, V-cycle, deep | 9 | 36 s |
| aggressive, W-cycle, 5 levels | 8 | 47 s |
| aggressive, W-cycle, deep | 8 | 60 s |
| plain, W-cycle, deep | 6 | 54 s |

The W-cycle reliably buys ~25 % of the cycles and never once pays for it, across five configurations — the
2^k visits to level k outrun any cheapness depth and aggressive coarsening can create. And aggressive
coarsening makes the W-cycle **worse** on cycles (8 against 6), the opposite of the hypothesis. Note this
is the one place where suspecting a premature closure was itself the error.

**So the marginal-cost optimum is REAL, not an artifact of exploring one axis at a time.** The coarse-grid
side survived a genuine joint test; the smoother side yielded one small Pareto gain that shifts the
optimum rather than contradicting it. Best native arm is now **7 cycles / 33 s** against the matched
incumbent's **2 / 11 s** — a threefold wall-clock gap that is structural for this smoother family.

### The sparse matvec is at the limit of what JAX's primitives give — block-CSR REFUTED

The level operators are applied through `_CsrOperator.apply`, a `jax.experimental.sparse.BCSR` product at
**scalar** block size. Since the flow block carries four fields per cell it has genuine 4x4 dense block
structure, and a true block form stores one column index per **sixteen** nonzeros rather than one per
nonzero — cutting index traffic from 4 bytes per nonzero to 0.25, about 31 % of total traffic on a kernel
whose fine level is bandwidth-limited. Measured on a synthetic operator at the flow block's exact shape
(92160 rows, 11.8M nonzeros, 128 per row), the saving does not materialize:

| variant | time | Gnnz/s |
|---|---|---|
| scalar CSR (what is shipped) | 5.20 ms | 2.27 |
| block 4x4, state already cell-major | **5.20 ms** | 2.27 |
| block 4x4 including the field-major transposes | 6.58 ms | 1.79 |

**Even granting the layout migration for free, the block form exactly TIES the scalar one**, because the
traffic it saves is spent on intermediates a fused kernel never writes: a per-edge gather buffer and a
per-edge product buffer. The transposes that the field-major layout `(cell i, field f) = f n + i` would
actually require cost a further 27 %. So a block-CSR migration is worth zero at best and -27 % as it would
have to be built, against a state-layout change across the whole solver.

**The corollary is the useful part: JAX's CSR matvec is not index-bound.** It matches a formulation
carrying sixteen times less index traffic, so it is near what the primitive can do and there is no easy
headroom in the kernel. Beating it needs a **fused** custom kernel doing gather → block multiply →
scatter without materializing between the steps, which is a Pallas or custom-call project — and one whose
payoff is on an accelerator, where that access pattern is the bottleneck, not on CPU where it would chase
a fraction of a threefold gap.

⚠️ **Per-call launch overhead is NOT why the coarse levels look expensive** — measured at about **10 us**
per matvec, against roughly 144 matvec calls in a V-cycle of ~400 ms, i.e. under 1 %. Nor are the coarse
levels especially inefficient where it matters: level 1 runs at 1.38 Gnnz/s against the fine level's 1.43
and simply holds 27 % of the nonzeros. Only levels 2-3 degrade (0.45-0.75 Gnnz/s) and they hold ~7 % of
the nonzeros, so their excess costs ~5 % of the cycle. **The fine level is ~70 % of the work and is where
any real saving has to come from.**

⚠️ **A kernel timing taken on RANDOM column indices understates the shipped matvec by ~2.2x** (1.43 against
3.13 Gnnz/s at the fine level's shape) purely through locality — real mesh-ordered matrices are far more
cache-friendly. Any cost model built on synthetic sparsity therefore overweights matvec work relative to
everything else, which is exactly why one such model predicted 16 % for removing a multiply-by-zero and it
delivered 2-7 %.

### The SIMPLE smoother is BALANCED and both halves are poor — measured, not argued

Siefert and de Sturler (2006) analyse precisely this operator class: a generalized saddle point whose
(1,2) block differs from the transposed (2,1) block and whose (2,2) block is nonzero but small in norm —
which is what Rhie–Chow interpolation produces, and which they note has received little attention and no
numerical experiments. They show the preconditioned eigenvalues cluster to within a constant times
`max(||S||, ||E||)`, where `S` is the error of the splitting used for the velocity block and `E` the error
of the approximate Schur inverse, and conclude that the two must be **balanced**: shrinking whichever is
already smaller cannot move the bound, so that effort is wasted.

Both have closed forms for this smoother rather than needing an estimate. The splitting is `F ~ diag`, so
`S = I - F_diag^-1 F`. The Schur inverse is `n` damped-Jacobi sweeps, whose recurrence telescopes to
`M_S S = I - G^n` with `G = I - omega D_S^-1 S`, so `E = -G^n` exactly. `_splitting_balance` in the probe
takes both as largest singular values by sparse iteration.

⚠️ **THE DIAGNOSTIC IS PRESENT BUT UNREACHABLE — it has NO caller, and there is no `BFS3D_PROBE_BALANCE`
switch (checked 2026-08-14; the record claimed one).** So the numbers below cannot currently be re-taken,
and neither can the SIMPLEC question further up that names this function as the way to settle it. Wiring
it back to a switch is a few lines and is the prerequisite for either. It takes the level's Schur in host
sparse form as an explicit argument, because the smoother's own pieces no longer carry a host copy — that
would stop them riding into the compiled cycle as a jit argument.

Measured at β = 0.1 on `state-00067`, strength 0.25, 4 outer x 2 inner sweeps, 5 levels, no singletons:

| level | `||S||` splitting | `||E||` Schur inverse | ratio |
|---|---|---|---|
| 92160 (fine) | 1.449 | 1.029 | 1.41 |
| 26828 | 1.230 | 1.113 | 1.10 |
| 6540 | 1.482 | 1.373 | 1.08 |
| 1448 | 1.275 | 1.238 | 1.03 |

**The DEEP levels are balanced (ratios 1.03–1.10); the FINE level is not (1.41), and it carries ~70 % of
the cost.** Read the two separately — a single "they are balanced" reading of this table is wrong where it
matters most. By the `max(||S||, ||E||)` criterion, bringing the fine level's splitting error down to its
Schur error would cut the bound 1.449 -> 1.113, about **23 %**; beyond that, further gains need both halves
together. So a one-sided improvement of the splitting is worth something, but bounded, and only on the
fine level.

It does explain why dropping the inner pressure sweeps 4 -> 2 was nearly free: the Schur side was already
no worse than the splitting side, so the extra sweeps had nothing to buy.

**Both norms are ~1.0–1.5, which is bad in absolute terms and is the real finding.** Their favourable
regime is `||S||, ||E|| << 1`, where the eigenvalues cluster tightly; at `max ~ 1.4` the bound is vacuous,
which matches this arm needing 8 cycles where the incumbent needs 2. Concretely `||I - F_diag^-1 F|| =
1.449` says the diagonal splitting of the velocity block is a ~100 % error, and `||G^2|| = 1.03` says two
damped-Jacobi sweeps on the Schur barely contract in the worst direction. (A 2-norm above 1 does not mean
divergence for a non-normal iteration — the spectral radius governs asymptotically — but it does mean very
little progress.)

**The consequence is structural, not tunable: a SIMPLE smoother with a DIAGONAL splitting is near its
ceiling here.** A real gain needs both halves improved together, and improving the velocity splitting past
a diagonal means solving with `F`, which is the cost this whole direction exists to avoid.

⚠️ **Cao, Du and Niu (2014) shift-splitting does NOT apply here** — there is no such lever to reach for.
Their construction assumes a **symmetric positive definite** (1,1) block, a **zero** (2,2) block, and
off-diagonal blocks that are exact transposes; the convergence proofs use all three (Lemma 2.2 needs
`||(alpha I + A)^-1 (alpha I - A)|| < 1`, Theorem 3.1 forms `A^{1/2}`). Our velocity block is
convection-dominated and nonsymmetric, our (2,2) block is the Rhie–Chow damping, and Rhie–Chow also breaks
the transpose relation. Their central device — adding `alpha I` to a zero (2,2) block to regularize it —
is something this discretization already provides, and augmenting that block further is separately
recorded as a triple-confirmed no-go that degraded cycles 2.7x. Their experiments are a Stokes problem,
i.e. no convection at all.

**What DID transfer: `8 x 2` is the right sweep pair at every β** (49 s against 56 s at 0.05, 43 s against
48 s at 0.1), so that tuning is not an artifact of zero shift.

⚠️ Still measured under a **uniform** column reach the case does not use; at the shipped reach the
incumbent's cost doubles at β = 0 (22 cycles against 11) and the native arm is unmeasured there.

**Why the gap may still be the right trade.** The incomplete-LU is a sequential triangular solve.
This smoother is sparse matrix-vector products, diagonal scalings and one small dense coarse solve, and
its fine level was measured memory-bandwidth-bound at 37–40 GB/s — the profile of something that
vectorizes. Kernel fusion and JAX dispatch overhead were measured out as levers (a chain of 8 matvecs in
one jit costs exactly 8x one matvec); the only thing that helps on CPU is touching fewer nonzeros.

Cell-local smoothing on the saddle is separately closed (point Jacobi, Chebyshev, cell-block Jacobi and
Vanka all fail), the flat-inverse family is closed, smoothed aggregation is closed, and orthonormalized
prolongation is closed.

⚠️ **lAIR is not the way in here**, and two of the three reasons are already measured on this mesh.
`build_air_hierarchy` is **scalar** — no `block_size` — and its classical C/F splitting picks individual
degrees of freedom, destroying the four-field block structure the SIMPLE smoother needs on every coarse
level. It calls `_require_positive_diagonal` at every level, and the saddle's pressure rows are the
Rhie–Chow damping. And on the true Jacobian slice of the `[k, omega]` block on this same mesh it ran
~50 minutes without finishing at 91 nonzeros per row; the flow block carries 128.

**Transferring the threshold to the `[k, omega]` block will not buy CYCLES**, and the trailing ablation
settled that on the right measure: at `state-00067`, β = 0, with the leading inverse held at ILU(0), the
shipped native trailing inverse and PETSc ILU(0) both give **11** coupled cycles, and a near-exact
factorization of that block is **worse than either** (58 cap). Nothing in that ablation supports spending
on the trailing block's accuracy. ⚠️ **The "2 restart cycles against an absolute floor of 1" figure this
entry used to rest on is a STANDALONE solve of the block, which production never performs** — one Krylov
iteration runs on all six fields with a block-triangular `M`, so the count is a property of the whole
preconditioned operator. Do not cite it.

**The live reason to sweep it anyway is SCALABILITY, not cycles.** At 2 levels the coarse grid grows
**linearly** with the mesh while its dense coarse solve is **cubic** to build and is rebuilt on every
refresh: ~438 dofs at 23040 cells, ~4400 at ten times that, and `_MAX_DENSE_COARSE_DOFS = 8192` is where
it raises outright.

✅ **The old blocker is gone: the trailing path is no longer capped at 2 levels.** It ran on
`build_convection_hierarchy`'s `max_levels` default (`_CONVECTION_LEVELS` = 2) because the class exposed
neither `strength_threshold` nor `max_levels`; the shared `NativeHierarchyInverse` base exposes the whole
coarsening surface (see the shared-base entry above). Two conditions on any sweep that uses it:
- **Sweep `max_levels` and `strength_threshold` TOGETHER.** A threshold coarsens LESS, so it enlarges the
  coarse grid; testing one at 2 levels reproduces the flow block's original failure and would refute the
  transfer for the wrong reason.
- ⚠️ **Normalize `_cell_graph`'s edge weights per row-field first.** It weights each cell edge by the SUM
  of `|A_ij|` over the block, and this block's ω rows sit ~8 orders above its k rows — so a raw threshold
  there is an **ω-only** strength measure. This is the same collapse-over-row-fields failure that let a
  column-reach audit approve a configuration which then diverged the march, and that made the trailing
  bound arm unreadable for a day.

⚠️ A nonzero threshold there still forfeits the refresh cache-hit unless paired with `frozen_coarsening`
or `shape_headroom`; the binding decision keeping the k/ω path at θ = 0 in a *march* is unchanged, and the
knob is safe for a single-state sweep precisely because a sweep never refreshes.

**Defects found here, and what was done about each.**
- **FIXED — `_reattach_to_adjacent_root` could CREATE singletons.** It never steals a root but does
  steal every member, so an aggregate could be cut down to its root alone *after* the aggregation's own
  repair had run and could no longer see it. Two passes are needed and both now run under
  `avoid_singletons`: the sweep refuses to open a singleton, and `_absorb_singleton_aggregates`
  dissolves whatever reattachment strands, relabelling contiguously. Measured on a 24x24 anisotropic
  Poisson at threshold 0.25 with one aggressive level: **15 and 11 singletons on the two levels
  unrepaired, 3 and 0 with the sweep repair alone, 0 and 0 with both** — and it coarsens further while
  doing it (260 -> 208 aggregates). Pinned by
  `test_avoid_singletons_reaches_the_aggressively_coarsened_level`.
- **FIXED (documentation) — `max_coarse` is in DEGREES OF FREEDOM, not cells.** It is compared against
  `a.shape[0]`, while both builder docstrings said *cells* — a `block_size`-fold error on the knob
  bounding a dense inverse. Dofs is the correct unit and the code was right: the limit exists to
  bound a solve whose cost is cubic in the dof count. The docstrings now say so.
- **FIXED (guarded, not defaulted) — the coarse grid grows LINEARLY with the mesh when the level cap
  binds before the size limit.** At `max_levels = 5` and a measured 184x total ratio, 23040 cells give
  500 coarse dofs and 1M cells would give ~21500 (~3.7 GB, dense, cubic to build). The level cap is a
  deliberate shipped default on the turbulence path (`_CONVECTION_LEVELS = 2`), so it was NOT changed;
  instead `_MAX_DENSE_COARSE_DOFS` (8192, ~512 MB) makes the case fail with both ways out named, rather
  than silently allocating. Nothing measured here approaches it — the largest coarse level in any arm
  is 4300 dofs. Pinned by `test_dense_coarse_solve_guard_rejects_an_oversized_coarsest_level`.
- **FIXED — `_AGGREGATE_STATS` accumulated across builds** despite its docstring saying to clear it, so
  a build that stopped early shifted every later arm's window in the consumer's tail slice. Cleared at
  the start of each build; the consumer may now read the whole list.

⚠️ **Stale by this section:** the recorded refutation that *"equilibration does not reach the coarsening"*
(0.03 % of edges) was measured at `strength_threshold = 0`, where only the sparsity **pattern** matters.
With a threshold live the aggregation reads **weights**, and these arms run `equilibrate=False` on a block
whose momentum, gradient, divergence and Rhie–Chow entries sit at raw scales. That refutation does not
transfer to any thresholded arm.

## Low-β directions already measured out — CLOSED, do not re-litigate

- **⚠️ LOW-β DIRECTIONS ALREADY MEASURED OUT — do not re-litigate without new evidence.** The low-shift
  wall on the 3D coupled saddle has absorbed a lot of probing. What is settled:
  - **Turbulence decoupling ("just lag ω") — REFUTED.** A true-residual arm comparison found
    block-diagonal `(u,v,w,p) ⊕ exact(k,ω)` to be the **worst** arm tested: the flow–turbulence coupling
    is load-bearing in the preconditioner, not a nuisance to be segregated away.
  - **A pre-AMG SIMPLE-type transform of the matrix — DEAD.** The published "SIMPLE preconditioning" for
    monolithic coupled AMG *is* Rhie–Chow interpolation; our residual already assembles that matrix, so
    there is no transform left to apply. (Established by reading the primary sources in full, not from
    abstracts.)
  - **PC-only pressure-Poisson augmentation — NO-GO, triple-confirmed.** The `(p,p)` block is *already*
    0.71× the SIMPLE-Schur elliptic operator, and the augmentation degraded cycles ~2.7×.
  - **`coarse_eq_limit` beyond ~2000 — inert, but read this correctly: the ARM was a no-op, which is
    not the same as a null result.** GAMG stops coarsening as soon as the grid falls below the limit,
    and this hierarchy is already **2 levels** — its single aggregation step has landed under 2000
    already, so a *larger* limit cannot make it stop any sooner and produces a bit-identical
    hierarchy. `K=8000 ≡ K=2000` is therefore arithmetic, not evidence, and it says nothing about
    whether the coarse space matters. **To probe the coarse space, degrade it instead**
    (`mg_coarse_pc_type: jacobi` in place of the direct LU) — that reads in both directions, and it
    is the control that decides what a *smoother* plateau means: if degrading the coarse solve barely
    moves the cycle count then the coarse correction is not load-bearing, and a smoother plateau
    cannot be attributed to the coarse space at all. Always print the coarse grid's equation count
    (`AmgVCycle.coarse_size`) beside the level count, so an arm that changed nothing is
    distinguishable from a setting that made no difference.
  - **Additive Vanka + Richardson — invalid by construction** (Richardson on an indefinite saddle).
    **⚠️ THE COARSE-SPACE READING THAT SAT HERE IS DELETED — the two entries conflicted and the configured
    side wins.** The losing claim: a Krylov-Vanka arm "still stalls, insensitive to the inner count, which
    points at the coarse space as the remaining wall" — recorded with no smoother, aggregation, state or
    shift. The winning claim is the 2026-08-08 arm table below, which records its configuration in full:
    degrading the coarse solve to Jacobi leaves the cycle count unchanged, so the coarse correction is not
    load-bearing there. The split was only ever established for the *2-level* GAMG hierarchy we run.
  - **Where the V-cycle actually under-performs:** per-field, pressure is *well* smoothed and **ω** is
    the unsmoothed field, then `u`. (The ratio that quantified "unsmoothed" is deleted — configuration not
    recorded; re-measure before relying on any magnitude.) If a per-field lever is wanted in 3D, ω is it —
    pressure is not.
    **⚠️ PARTLY REHABILITATED (2026-08-08): a second, independent measurement puts ω in the same
    place.** The cell-block singular-value decomposition above finds the near-null direction of the
    worst diagonal blocks to be **pure ω**, in 353 of 23040 cells, under the *current* bundle. That is
    local block conditioning, not per-field V-cycle smoothing rates — a different quantity by a
    different route — so "ω is the 3D per-field lever, not pressure" now rests on two legs instead of
    one unfalsifiable one. The *factor* that once accompanied it is deleted for want of a configuration; treat
    the **field** as corroborated and any magnitude as unmeasured.
    **⚠️ NEITHER THIS NOR THE VANKA BULLET RECORDS WHICH SMOOTHER OR AGGREGATION IT WAS MEASURED
    WITH, so neither can be relied on now.** The smoother default has since moved ILU(1) → ILU(0) and
    the aggregation smoothed → plain, and *both* of those changes have inverted a conclusion on this
    case. Concretely, the Vanka bullet's inference — "a strong smoother still stalls, therefore the
    **coarse space** is the wall" — is valid but **under-determined**: "the coarse space" could mean
    the space is intrinsically inadequate (needing inf-sup-aware coarsening, a research problem) or
    that the coarse *correction* was corrupted by prolongator smoothing (a setting, now changed). A
    stall at 5–6e-2 fits both, so that experiment never distinguished them. Re-measure before building
    on either, and **record the smoother and aggregation** in any replacement.
    **The second reading is now measured, not speculative.** Prolongator smoothing *was* degrading the
    coarse correction on this operator: turning it off (`pc_gamg_agg_nsmooths = 0`) is worth
    **22 → 9 cycles** at a hard state and ~16 % of the whole march's Krylov cost. Every arm in that
    Vanka campaign was judged against a coarse correction built with smoothing **on**, so a *smoother*
    arm failing to rescue it is what you would see whether or not the smoother was any good — which
    also explains why the campaign kept concluding "the smoother is not the lever" whichever smoother
    it tried. **Do not treat "the coarse space is the wall" as settled.**
    **⚠️ FIRST MEASUREMENT UNDER THE CURRENT BUNDLE (2026-08-08) — and it moved the question.**
    Configuration, in full, because that is the point: 3-rung `bfs3d` cold march to ‖R‖ = 2.64e-6 (69
    steps, 60.5 min); state `state-00049` (rung 3, the target Reynolds number); operator β = 0.0293
    with the V-cycle at the floor 0.05; **plain aggregation, ILU(0), 4 sweeps, `coarse_eq_limit` 2000
    → 2 levels with 1296 coarse equations**; real right-hand side `−R(state)`, judged on the true
    residual through GMRES at rtol 1e-6 (restart 15).

    | arm | patch width | worst `\|A_p⁻¹\|` | restart cycles | true relative residual |
    |---|---|---|---|---|
    | self-check, the march's own solver | — | — | 1 | 2.0e-06 |
    | shipped (ILU(0) ×4) | — | — | **2** | 1.3e-14 |
    | ILU(0) ×8 | — | — | 1 | 1.4e-14 |
    | ILU(0) ×16 | — | — | 1 | 1.4e-14 |
    | coarse solve degraded to Jacobi | — | — | 2 | 1.3e-14 |
    | Vanka, 0 neighbours (the cell block) | 6 | 3.011e3 | 58 (cap) | 7.8e-01 |
    | Vanka, 6 velocity neighbours | 24 | 3.006e3 | 58 (cap) | 3.6e-01 |
    | Vanka, 6 neighbours, all fields | 42 | 3.242e3 | 58 (cap) | 2.4e-01 |
    | Vanka, 6 velocity neighbours, damped 0.3 | 24 | 3.006e3 | 58 (cap) | 4.1e-01 |

    Two things follow, and the second is the bigger one.
    - **Vanka does not stall here — it fails outright, at a state where the operator is easy**, and it
      fails for a reason that has nothing to do with the coarse space. It cannot be excused by a hard
      operator: ILU(0) solves the same system to 1.3e-14 in two cycles. Every patch width runs to the
      restart cap; widening helps monotonically (0.78 → 0.36 → 0.24) but nowhere near enough, and
      **the worst patch gain is flat at ~3e3 across all three widths** — on an operator equilibrated
      to unit diagonals, i.e. a local singular value four to five orders down. Flat under widening is
      the informative part: no amount of surrounding a cell repairs it, so the degeneracy is in a row
      of the **centre block** that stays degenerate however large the patch. (Adding the neighbours'
      `k` and `ω` — the "all fields" arm — did not reduce it either; it rose slightly.)
      That the *narrowest* arm is plain point-block Jacobi also settles the "is the implementation
      wrong" question the right way: block-Jacobi failing on a saddle point is textbook, and is
      precisely why patch smoothers exist. **Why ILU(0) succeeds where every patch method fails:** in
      cell-major ordering it is a *global* forward/backward sweep and never inverts a cell block in
      isolation, which is exactly what a patch method is obliged to do.
      **⚠️ It STAGNATES, it does not amplify — and that distinction is not decoration, it points at a
      different cause.** Every arm stops at a finite true residual and runs to the restart cap; none
      diverges. Under-relaxing (damping 0.3), which is what tames an over-correcting smoother, made it
      **worse** (0.36 → 0.41), as it would for a smoother that is too *weak* rather than explosive.
      So the large patch gain is at present a **correlation with the failure, not a demonstrated
      cause** — do not write it up as the mechanism. Two explanations survive the data equally well:
      1. a few near-singular cell blocks poison the recombination; or
      2. the **additive** form is simply too weak here. Weighted-additive Schwarz is block-Jacobi-like
         and needs a relaxation that may not exist for this operator, which predicts exactly what is
         seen: monotone gains with patch width, stagnation at every width, no help from damping. The
         classical Vanka sweep is *multiplicative* and remains **untested**.

      **WHICH row is degenerate — measured, and it is ω.** A batched singular-value decomposition of
      all 23040 diagonal 6×6 blocks of the equilibrated cell-major operator at the same state:
      `σ_min` median 2.87e-2, 1st percentile 8.42e-4, minimum 3.21e-4, with **353 cells below 1e-3**
      and condition numbers there of 5e6–9e6. In every one of the twenty worst blocks the near-null
      right singular vector is **pure ω** — `(u,v,w,p,k,ω) = (0,0,0,0,0,1)` to three decimals. So it
      is the ω *column* that is nearly empty: perturbing ω in such a cell barely changes any of that
      cell's own six equations, because ω's influence there is carried almost entirely by neighbour
      transport rather than locally. That is exactly the quantity a cell-centred patch must invert and
      a global cell-major incomplete-LU sweep never does, which is the cleanest available explanation
      for why every patch smoother fails on this operator while ILU(0) is untroubled. The affected
      cells sit in low-`k` regions (median `k` 0.122 against 0.655 over the mesh) and occur in exact
      spanwise-symmetric pairs, so they are a coherent region of the flow, not scattered noise.
      **⚠️ BUT THE SHIFT DOES NOT CONTROL IT, so this does NOT explain the low-β wall.** Sweeping the
      shift over a 500× range at the same state (one materialization; the Jacobian does not depend on
      β, only the added diagonal does):

      | shift | median `σ_min` | min | cells below 1e-3 |
      |---|---|---|---|
      | 0.5 | 3.450e-2 | 3.929e-4 | 204 |
      | 0.05 (the floor) | 2.873e-2 | 3.214e-4 | 353 |
      | 0.005 | 2.807e-2 | 3.133e-4 | 367 |
      | 0 (unshifted) | 2.799e-2 | 3.124e-4 | 369 |

      Over the range the march's tail actually occupies (0.05 → 0.005) the near-singular count moves
      **4 %**. The degeneracy is a fixed property of the discretization and state, not something β
      governs — so the low-shift conditioning wall is *something else*, and this is not the mechanistic
      account of it that it first looked like.
      **It also kills the obvious lever before anyone builds it.** An ω-only, preconditioner-only shift
      boost cannot work, because the shift is `β·d` and for ω that `d` **is the weak transport coupling
      that is the problem** — scaling a near-zero diagonal by β leaves it near-zero at any β. Anything
      along these lines would have to be an *absolute* floor on the preconditioner's ω diagonal rather
      than a multiple of `d`, which is untested speculation and has no headroom to be demonstrated at a
      state where ILU(0) already converges in two cycles.
      **What the ω finding does license** is narrow and solid: it explains why *cell-local*
      preconditioners fail here, and it gives a two-minute screen for the next one.

      **It also independently corroborates the ω bullet below**, which was recorded without its
      configuration and marked unusable: a completely different measurement — local block
      conditioning rather than per-field V-cycle smoothing rates — lands on the same field. Two
      unrelated routes to "ω is the 3D preconditioner lever" is much better evidence than either
      alone, and it also suggests a concrete arm: **leave ω out of the patch**, since a local solve
      cannot resolve a direction the local block does not see.

      **RESOLVED, and (1) IS REFUTED — the near-singular blocks are NOT the mechanism.** The test was
      pre-registered (`VankaSmoother(max_patch_gain=...)`: converges ⇒ (1), stagnates ⇒ (2)) and the
      answer is unambiguous, because dropping precisely the near-singular patches made the solve
      **worse**, not merely no better. The whole family of drop-arms is monotone in *coverage* and in
      nothing else:

      | patches dropped | fraction of the mesh | true relative residual |
      |---|---|---|
      | 0 | — | **0.360** |
      | 337 (gain > 1e3 — the σ_min < 1e-3 set) | 1.5 % | 0.737 |
      | 1925 (gain > 3e2) | 8.4 % | 0.9975 |
      | 6044 (gain > 1e2) | 26 % | 0.9995 |

      Strictly monotone in how much of the mesh still gets relaxed, which is the tell: these arms
      measure **coverage** and nothing else.

      Those 337 patches were doing *useful* work; removing them leaves their degrees of freedom
      unrelaxed and costs more than their ill-conditioning ever did. So the large patch gains are a
      real property of this operator — and worth knowing, since two independent measurements agree on
      the set (337 patches by inverse gain, 353 cells by block `σ_min`) — but they are **not** why the
      smoother fails. **Explanation (2) is what survives: the ADDITIVE recombination is the limit.** It
      smooths usefully but insufficiently *everywhere* rather than being poisoned anywhere, which is
      also what the width ladder and the damping arm independently say.
      **MULTIPLICATIVE IS NOW TESTED TOO, AND IT IS WORSE — so (2) falls as well.** Compared
      sweep-for-sweep, which is the only fair way to ask whether *sequencing* helps (one multiplicative
      sweep against four additive ones confounds recombination with sweep count): **additive ×1 →
      0.497, multiplicative ×1 → 0.855.** Sequencing the patches does not rescue this smoother; it
      costs. Built as `VankaSmoother(multiplicative=True)`, 16 colours on this mesh, ~33× the additive
      apply cost.

      **So neither hypothesis stands, and what is left is the one fact that survived every arm: the
      cell block is weakly coupled in ω EVERYWHERE, and the 353 near-singular cells are only its tail.**
      Median `σ_min` is 2.9e-2 on an operator equilibrated to unit diagonals — some 34× down — so
      *every* cell-local solve mishandles ω, not just the extremes. That single fact accounts for the
      whole ladder: widening the patch adds neighbour velocities and cannot help ω; dropping the worst
      patches removes useful work without touching the general weakness; damping and sequencing change
      only how corrections are combined, and no recombination repairs a local solve that cannot see the
      field. It equally explains why ILU(0) is untroubled — a global cell-major sweep propagates ω along
      the transport direction, which is where ω's coupling actually lives.

      **Verdict: cell-centred patch relaxation is the wrong shape for this operator, and the reason is
      structural rather than tunable.** Do not re-open it with another patch variant. `VankaSmoother`
      stays in the tree as the evidence and as a testbed; `validation/bfs3d_openfoam/cell_block_conditioning.py`
      screens any future cell-local proposal in ~2 minutes, which is what this campaign cost hours to
      learn.

      **Choose that cap from the gain distribution, not from the maximum**, or the arm measures the
      wrong thing. Measured over the 23040 width-24 patches at this state: gain above **1e1 in 96.6 %**
      of them (22251), above **1e2 in 26.2 %** (6044), maximum 3.0e3. A gain of order ten to a hundred
      is therefore *ordinary* here — it is what `1/σ_min` gives for the median block (`σ_min` ≈ 2.9e-2)
      — and only the ~1.5 % above 1e3 are the near-singular ω blocks the hypothesis is about. Capping
      at 1e2 drops a quarter of the mesh and the true residual goes to **0.9995**, i.e. no reduction at
      all: with that many patches gone, any degree of freedom covered only by them has weight zero and
      is never relaxed. That number says the smoother was gutted, and nothing whatever about whether
      near-singular patches caused the original failure — a confounded arm, not a null result.

      Related and worth keeping either way: this is the same *family* of failure as the two earlier
      patch-smoother attempts (unweighted additive Vanka, "ρ = 9e4"; undamped block-ILU/inexact Uzawa,
      "1-apply reduction 5.10"), and the overlap weighting that was supposed to fix it does not. Note
      also that the published algebraic Vanka does *not* solve the patch exactly: Metsch's (§4.6) local
      solve is an inexact-Uzawa form built on a diagonal `Â > A` with a scaling `β` chosen so
      `Ŝ > C + BÂ⁻¹Bᵀ`, provably convergent precisely because it never inverts a near-singular local
      saddle. The exact patch solve chosen here as the *stronger* option may be the thing that breaks.
    - **⚠️ THE STATE-SELECTION PREMISE IS BROKEN FOR A DUAL-TIME MARCH, and this invalidates the
      comparison above as a test of the *coarse-space* question.** A checkpoint is written at the end
      of a step, so it holds the state the *next* step starts from — and a step's first solve is its
      easy one, from a settled state with a freshly rebuilt preconditioner. Across the whole march:
      **all 70 step-initial solves cost ≤ 2 restart cycles, while solves at inner > 0 reached 15.**
      No checkpointed state in this march poses a hard linear system, so every arm ties there and the
      sweep says nothing about smoother-versus-coarse-space. (It still says plenty about Vanka, which
      *failed* at an easy state — a positive result needs a hard state, a failure does not.) The fix
      is `DualTimeStep.inner_observer`, which now also carries the **iterate**.
      **⚠️ But a march step CANNOT be reconstructed from a checkpoint — its configuration is
      path-dependent.** `validation/bfs3d_openfoam/inner_iterate_probe.py` tried, driving one step from
      the checkpoint, and both plausible arrangements bracket the march without reaching it. At
      `state-00049`, β = 0.0293, where the march's first inner solve costs **1 cycle at α = 0.500**:

      | how the engine was built | inner-0 cycles | ‖G‖ |
      |---|---|---|
      | at the probed state (self-consistent) | 7 | descends cleanly, α = 1 |
      | at the Reynolds rung's seed, 11 steps back | **39** | **no descent at all** |

      The march is outside both because `amg_beta_tracking_refresh` rebuilds the preconditioner
      repeatedly on the way to the step, so by then it is recent, while the shift policy dates from the
      seed with its transport part rebuilt at each refresh — a product of the refresh *history* that no
      checkpoint records. **The route to the hard iterates is therefore to capture them DURING a
      march**, via the observer, not to replay a step afterwards.
      **✅ DONE, and it settles the question: the march's expensive solves are STALENESS, not hard
      operators.** `InnerIterateCheckpointer` (`solve/checkpoint.py`, wired in the case behind
      `BFS3D_INNER_DUMP_ABOVE`) caught the seven solves that reached ≥4 restart cycles on an otherwise
      byte-identical march (`x_r/h` 8.361, 290 cycles, unchanged). Note the replay problem does **not**
      apply once you hold the iterate: the state is on disk, so the operator can be rebuilt around it
      directly. At the march's single hardest solve — attempt 50 inner 3, β = 0.0293, where the march
      took **15 cycles** and the line search collapsed to α = 0:

      | iterate | β | the march | one step stale | matched at the iterate |
      |---|---|---|---|---|
      | attempt 50 inner 3, α → 0 | 0.0293 | **15** | 3 (5.9e-02) | **1 (6.6e-06)** |
      | attempt 40 inner 3, α = 1 | 0.3333 | **8** | 1 (9.9e-05) | 1 (1.7e-10) |

      Read the two rows together: at β = 0.33 even a stale preconditioner is already optimal, so
      **staleness only bites at low β** — and there it bites hard. Note also that both march counts far
      exceed what *one* step of staleness costs (15 against 3, 8 against 1), so the march's real
      staleness — up to four steps in `J` — is worth several times a single step.

      **That operator is easy.** One cycle to 6.6e-06 with a matched preconditioner, and a single step
      of staleness already triples the cost and gives up four orders of accuracy. So there is **no
      preconditioner headroom left on this case at any state the march visits** — which is the
      retrospective explanation for why every arm in the Vanka campaign tied or lost, and why the
      aggregation sweep only separated at all under the *older*, weaker bundle. The lever here is
      **refresh cadence** (`refresh_every=8`, `materialize_every=4`, `beta_rel_change=0.25` are lax
      around the hard steps), consistent with the earlier gate fix, which bought 132 fewer cycles and
      removed three of five retry cascades by refreshing ~50 % more often — the same mechanism found
      from the other end.
      **✅ THE COST TRIGGER IS BUILT — `amg_beta_tracking_refresh(refresh_on_cycles=N)`.** A solve that
      reaches `N` restart cycles refreshes the preconditioner **at the iterate it was handed** and the
      inner loop carries on, rather than aborting the step and escalating β (which discards both the
      work and the pseudo-timestep). Capped at **one refresh per step**, without which a genuinely hard
      operator would refresh on every inner iteration; a new inner loop re-arms it. Off by default
      (`None`), so a march that does not opt in is byte-identical. Pinned by
      `test_inner_refresh_fires_on_an_expensive_solve_at_that_iterate_and_once_per_step`.

      **Measured end to end on the 3D coupled backward-facing step, against the scheduled cadence:**

      | | scheduled | on cost |
      |---|---|---|
      | wall | 3632 s | **3140 s (−14 %)** |
      | refresh | 758 s | **310 s (−59 %)**, 62 refreshes → **19** |
      | Krylov cycles | 290 | 293 (**unchanged**) |
      | steps with α = 0 | 5 | **0** |
      | `x_r/h` | 8.361 | 8.361 (unchanged) |

      **Read the cycle row: this is not a better-conditioned solve.** Cycles are flat, so the whole
      saving is refresh no longer spent maintaining freshness nothing consumed, plus the elimination of
      five dead steps — those where the line search collapsed, the residual froze or rose, and the shift
      escalated twenty-fold before recovering. That the *same* change removes both is the point: an
      inaccurate direction from a stale preconditioner is what α → 0 was, so the retries were downstream
      of the refresh cadence rather than a separate globalization problem.

      **Do not read the arithmetic below as an open proposal — it is the projection this shipped
      against**, and it came out close on the total (−492 s measured vs −515 s predicted) while missing
      the mechanism: it predicted the saving would come from refresh alone, and half of it came from the
      dead steps. The cost arithmetic, measured on the earlier 3501 s
      march: scheduled refreshes are 50 full + 12 shift = **742 s, 21 % of the wall**. A refresh fired
      only when a solve reaches ≥3 cycles would fire **16** times (227 s, **−515 s**); at ≥4, **7**
      times (99 s, **−643 s**) — a 15–18 % whole-march saving, against a ~8 % ceiling for a *perfect*
      preconditioner. Read as an *addition* to the schedule the trigger looks break-even; as a
      **replacement** it is the biggest win on the table, and conflating the two is easy to do.
      **Why no schedule can work here:** 193 of the 232 solves already take 1 cycle, so most scheduled
      refreshes maintain a freshness nothing consumes — and the right interval is regime-dependent (at
      β = 0.333 a one-step-stale preconditioner still gives 1 cycle; at β = 0.029 it gives 3), so a
      fixed cadence necessarily over-refreshes in the easy regime and under-refreshes in the hard one.
      Targeting cannot be predictive either — the Step-0 diagnostic refuted every static signal — which
      leaves reacting to the cost itself.
      **Two things to get right.** The counts above come from a march that *had* the schedule keeping it
      fresh, so removing it shifts the distribution up and the trigger fires more often; the equilibrium
      rate is a feedback loop (staleness → cycles → trigger → freshness) and is unknown until it is run.
      And it needs a **cap of one refresh per step**: with it the worst case is 69 × 14.2 = 980 s against
      today's 742 s, a bounded +238 s downside for a ~515 s upside; without it, a refresh per inner
      iteration is unbounded. `abort_above_inner_cycles` already detects the condition inside the inner
      loop, so the change is to the *reaction* — refresh and continue, keeping β escalation as the
      fallback, which also makes the refresh a diagnostic: if it does not help, the operator really is
      hard.
      **The coloured probe dominates a refresh, and the obvious way to halve it — stencil reach 3 → 2 — is
      MEASURED AND FAILS.** (Its exact share was recorded with no state or mesh and is deleted.) The saving is
      real (112 → 60 colours) but the V-cycle is
      not: at `state-00062`, β = 0.0072 against the 0.05 floor, reach-2 gives **41 cycles at a true
      relative residual of 1.9** — worse than the initial guess — where the shipped reach-3 arm reaches
      1.5e-10 in six. A failure at a *benign* state is conclusive: it cannot be excused by a hard
      operator. Note this also disposes of an appealing argument that does **not** work: the older
      refutation blamed "the pattern-dependent ILU(1) smoother", and at ILU(0) there is no fill to be
      pattern-dependent about — yet reach-2 still fails at ILU(0), so the conclusion outlived the
      mechanism that was offered for it. **Retested under plain aggregation too — it still fails, so the lever is
      CLOSED.** At `state-00049`, β = 0.0293 (where reach-3 does 2 cycles to 1.3e-14): reach-2 plain
      gives **58 cycles at 1.7e-01**, and doubling the smoother sweeps barely moves it (1.4e-01, still
      at the restart cap). Note what that rules out: it is not a smoothing deficit. And the hierarchy
      itself changes — reach-2 yields **3 levels / 480 coarse equations** against reach-3's 2 / 1296 —
      so the dropped couplings are load-bearing for the *coarse space*, not merely for the smoother's
      fill. Reach 3 is required, now measured across both aggregations at ILU(0). **The reach-3 PATTERN
      is required; probing every COLUMN at reach 3 is not, and that distinction was missed here.**
      This entry shrinks the pattern, which is what moves the hierarchy. Keeping the pattern at reach 3
      and shortening only the *columns* (`column_reach`) leaves the coarse space and the smoother
      untouched and is worth 564 → 454 probes at the shipped `(3,3,3,3,2,2)` — but it is **not** the cheap
      win it was recorded as: it aliases each shortened column's far couplings onto its near entries, over
      half the entries of every shortened column on `bfs3d`, and shortening `p` as well
      (`(3,3,3,2,2,2)`) diverges the march at step 1.
      See *"per-column probing reach"* under `sparse_jacobian.py` above before reaching for it. So this
      arm's "there is no cheap way to shrink the probe" stands for the *pattern*, and the column
      variant that looked like the exception has not yet been shown safe on this case.
      **A trap: the cheap `refresh_shift_in_place` branch does NOT help within a step.** The shift is
      formed once per step at the reference state and held fixed across the inner loop, so what drifts
      inside a step is `J(p)`. The shift-only branch only helps *across* steps, where β moves.
      **Untested but cheap and worth doing: whether staleness also drives the GLOBALIZATION cost.** A
      stale preconditioner returns a less accurate Newton direction (5.9e-02 against 6.6e-06 here), and
      an inaccurate direction can fail to descend — which is what α → 0 means. If so, the retries and
      line-search collapses are downstream of the same cause, not a separate problem. The captured
      iterate is all that is needed to check it. Two details worth keeping from the
      attempt: the builder's `amg_beta` defaults to **2.0**, which at a sub-floor β is a two-orders
      mismatch that alone turns a 1-cycle solve into 7 (pass `max(β, beta_floor)`); and a
      preconditioner frozen 11 steps back yields **no descent whatsoever** — α reads 1.000 because
      that is the line search's non-descent fallback, with ‖G‖ flat — which is independent evidence
      that the refresh is load-bearing.

      The probe now **validates itself against `march.log`** (inner-0 cycles and α) and refuses to
      report if it disagrees, which is how both errors above were caught rather than written up. Hold
      any future march-reproduction harness to the same gate.

    **How to re-run it — the smoother and the harness are BUILT (`aquaflux/solve/vanka.py`,
    `validation/bfs3d_openfoam/preconditioner_sweep.py`); what is missing is the measurement.** The
    discriminating question is narrow: **with plain aggregation, does a Vanka smoother still stall?**
    If yes, the coarse space really is the wall and the inf-sup / block-Schur direction is justified.
    If no, the original verdict was an artifact of the aggregation default and the smoother direction
    reopens. Record the smoother, aggregation, state and shift pairing alongside whichever answer
    comes out. The `ARMS` ladder in the harness reads the question two independent ways — a **sweeps**
    ladder on the *shipped* smoother (if cycles keep falling as sweeps rise the smoother is not
    saturated and cannot be the binding constraint; if they plateau, what survives is in the coarse
    space's blind spot) and the **Vanka** arms themselves, the widest deliberately over-strong.
  - **The Jacobian's fill is irreducible.** The coupled `(u,p)` Jacobian is intrinsically **distance-2**
    (~38 nnz/row) because Rhie–Chow damping couples pressure to the neighbour-of-neighbour ring; the
    advection scheme is irrelevant to this. A distance-1 preconditioner pattern is not available for a
    second-order collocated Rhie–Chow discretization, so "make the PC pattern local" is not a lever.

## The coupled AMG builder

- **Coupled builder `coupled_amg_continuation`** (`.claude/rules/turbulence.md`) shares
  `MonolithicFactorShiftPolicy` + `_monolithic_factor_step` with the ILUT/LU. Verified: converges to the
  block PC's fixed point AND passes the **coupled-adjoint FD gate** (the transpose V-cycle serves the
  gradient), `tests/integration/test_coupled_amg.py`; V-cycle mechanics in `tests/unit/test_amg_preconditioner.py`.
  Follow-ups: a refreshing/β-tracking variant (the frozen build serves the forward + adjoint; a developing
  3D march would want the refresh), and the FGMRES forward optimization.

## Globalization — forward step, continuation, line search

- **Forward globalization is ONE injected strategy — `forward_step: ForwardStep`.** The forward
  Newton loop has a single point of variation: `ImplicitNewtonSolver` takes one `forward_step`
  implementing the `ForwardStep` protocol (`stepper()` → the per-step
  `(residual_fn, phi, ‖R₀‖, solver) -> phi_next`; `default_solver()` → the inexact-Newton forward
  GMRES for that march; `adjoint_preconditioner()` → the converged-state transpose preconditioner).
  Two concrete strategies: **`DampedNewtonStep`** (default — the backtracking line search, holding
  the forward/adjoint preconditioner and the line-search count) and **`PseudoTransientStep`**
  (`aquaflux/solve/`, the residual-agnostic diagonally-shifted march; the flow configures it via
  `aquaflux/flow/`'s `momentum_continuation` factory — no wrapper class). `_forward` calls the
  injected step unconditionally — there is **no `if continuation is None` branch**, and **no separate
  `line_search`/`preconditioner`/`continuation` constructor args** (they were unified here; do not
  reintroduce them). Each strategy's shift vanishes at the fixed point, so the converged state and
  the IFT adjoint are strategy-independent. When adding a globalization (e.g. a monotone/forcing
  acceptance), add a `ForwardStep` — do **not** grow a branch in `_forward`.
  - **`ShiftedForwardStep` is the SECOND contract, and the eager march's beta machinery requires it
    (binding, 2026-08-15).** `ForwardStep` says what every strategy must *do*; `ShiftedForwardStep`
    says what one must additionally *carry* — a `relaxation_schedule` holding `beta` as a readable,
    replaceable **dynamic** leaf. It is separate rather than folded in because not every strategy has
    a shift: `DampedNewtonStep` globalizes by backtracking alone, and demanding a `relaxation_schedule`
    of it would be inventing a quantity it does not possess.
    **What it replaces, and why it matters more than it looks:** the requirement was enforced by
    `hasattr` probes scattered through `forward_march`, which fail **silently**. A `DampedNewtonStep`
    satisfies `ForwardStep` in full, so a march configured with `retry_on_cycles` accepted it and then
    never escalated — from the log, indistinguishable from a march that never needed to. One reporting
    path failed the *opposite* way and read `active_step.relaxation_schedule` unguarded, so the same
    conforming step raised `AttributeError` mid-march. `forward_march` now checks **once, before the
    first step** (`_require_shifted`), naming the feature and what it needs.
    - **Gate only what is genuinely silent (binding).** The escalation is gated. The divergence retry
      (`retry_solver` / `retry_divergence_cap`) is **not** — it re-solves at a tighter tolerance and
      never touches beta, so it works on any step; its *reporting* goes through `_shift_of`, which
      returns `None` for a step with no shift rather than inventing one. A `StepControl` is **not**
      gated either: the protocol only asks it to return a ready-to-run step, a step-agnostic one is
      legitimate (and exercised), and a control that *does* drive beta already fails loudly from its
      own `tree_at`. Gating those would reject what the protocol permits.
    - **The runtime check tests the SHIFT, not `isinstance(..., ShiftedForwardStep)`.** The argument is
      already typed `ForwardStep`, so re-testing those four methods at runtime would reject a
      legitimate duck-typed step for a reason unrelated to the feature asked for — which it did, on
      every test double, when first written that way.
- **`continuation.py` — BUILT (`PseudoTransientStep`, residual-agnostic).** The pseudo-transient
  continuation engine lives **here in `solve/`, not in `flow/`** — it is a `ForwardStep`
  (`stepper`/`default_solver`/`adjoint_preconditioner`) that runs an **injected**
  `RelaxationSchedule` for the shift strength β, the diagonally-shifted solve `(J + diag(βd))δ = −R`
  (`solve_linear(throw=False)`), and the closed-loop accept/escalate `while_loop`. **`stepper()`
  returns `(φ_next, cycles, alpha)`** — the accepted attempt's cycle count *and* its line-search
  factor α (the step-quality signal, ≤1; `_forward` drops both off the `custom_vjp` primal, a march
  reads them).
  - **The β schedule is an injected `RelaxationSchedule` (`solve/relaxation.py`), SER extracted as the
    default (binding — do not re-inline the β rule).** The old `beta0`/`exponent`/`beta_floor` fields on
    `PseudoTransientStep` are gone; the field is `relaxation_schedule: RelaxationSchedule`, defaulting to
    `SwitchedEvolutionRelaxation(beta0=2, exponent=1, beta_floor=0)` — byte-identical to the old inline
    `max(beta_floor, β₀(‖R‖/‖R₀‖)^p)`. It is the direct twin of the injected `ResidualNorm`
    (`solve/norm.py`). A `RelaxationSchedule` is **memoryless** (β from the two residual norms only), which
    is what keeps it on the differentiable traced path. `ConstantRelaxation(β)` carries β as a **dynamic
    0-d leaf** so an external control can vary it per step as a `filter_jit` cache hit (the `lam_max`
    precedent). The five builder factories (`momentum_continuation`, `coupled_continuation`,
    `mass_flow_coupled_continuation`, the two scalar builders) keep their public `beta0=/exponent=` knobs
    and translate them into `SwitchedEvolutionRelaxation(...)` at the one construction line — a factory
    building the real object, not a shim. A *stateful* or α-driven damping rule is **not** a schedule; it
    is a `StepControl` on the eager march (see `march.py`).
  - **The shift's SPATIAL distribution is an injected `ShiftBasis` (`solve/shift_basis.py`) — the
    spatial twin of the `RelaxationSchedule` (binding).** `RelaxationSchedule` sets *how much* damping
    (the scalar β); `ShiftBasis` sets the per-cell base diagonal `d` the shift `β d` is built on, from
    the operator's two diagonal buckets: `local_diagonal(convective, dissipative) -> d`. The one concrete
    is `LocalCourantBasis(dissipative_weight=w)`: `d = convective + w·dissipative`. **`w = 1` (default) is
    the full operator diagonal `a_P`** — and because `d = a_P` (the same diagonal the operator carries),
    `β a_P` is spatially-*uniform* under-relaxation (relaxation `1/(1+β)` in every cell), byte-compatible
    with the historical shift. **That equivalence is exact and worth stating plainly: on the momentum
    rows, this shift IS the implicit under-relaxation a segregated pressure-correction solver applies,
    at `α_u = 1/(1+β)`.** Both sides match, not just the diagonal — the shift acts on `δ = φ − φ_k` and
    so contributes `β a_P φ_k` to the right-hand side, which is exactly the `((1−α_u)/α_u) a_P φ_old`
    that implicit under-relaxation adds. At the pitzDaily march's operating point `β ≈ 1.9`, that is
    `α_u ≈ 0.345`, against the 0.3 that established segregated codes use as their default momentum
    relaxation. **Consequence (load-bearing for the cold-start work): the momentum treatment is the
    industry-standard one in different notation, so it is NOT where a cold-start reachability gap can
    be hiding, and "our globalization is exotic / uncovered by theory" is false.** Look instead at what
    differs — the coupled pressure/continuity treatment, the turbulence relaxation and limiters, and the
    fact that the reference which reliably reaches this root is a *transient* run. **`w = 0` is a genuine local convective time step** (`d = Σ_f max(mdot_f,0)`
    = ½Σ|mdot|), the non-uniform per-cell `Δt` a Courant condition implies — OF's `Co = ½Δt Σ|φ|/V` and
    Fluent's *segregated* local pseudo-time step are this same convective basis. The buckets are supplied
    by each block: `rhie_chow.momentum_diagonal_parts` (velocity) and
    `preconditioner.scalar_transport_shift_diagonal_parts` (k/ω), each summing to the block's total shift
    diagonal to rounding. **Consequence for the preconditioner (binding):** with a non-`a_P` basis the
    shifted diagonal is `a_P + β d`, *not* `a_P(1+β)`, so `make_preconditioner` must invert `a_P + β d` —
    the velocity block's `apply_at` is fed exactly that (was `a_P(1+β)`; identical when `w=1`). Threaded
    through `momentum_continuation`/`coupled_continuation`/`solve_coupled(shift_basis=…)` and the k/ω
    shift policies; **pressure keeps zero shift regardless** (elliptic), which is why a *local* basis is
    defensible on this coupled solver where Fluent uses a global time scale for its coupled path.
    - **The velocity buckets' SOURCE is injected (`VelocityShiftParts`), because the shift and the
      preconditioner have different lifetimes (binding).** `CoupledShiftPolicy` used to take them from
      the frozen flow preconditioner, which welded two unrelated lifetimes together: the preconditioner
      is deliberately frozen for many steps (re-freezing the flow block is measured unhelpful), while the
      shift is a *local time scale* that should describe the operator being solved now. Worse, the
      borrowed quantity was **live in velocity but frozen in viscosity**, so in a coupled solve `μ_eff =
      ρ(ν + ν_t)` was stuck at its freeze-state value and the velocity time scale ignored the eddy
      viscosity that develops — nobody chose that, it fell out of reusing the preconditioner's assembler.
      `FrozenViscosityVelocityParts` (the default, `None`) reproduces the old diagonal **exactly**;
      `LiveViscosityVelocityParts` re-forms the closure at the state being stepped, costing one closure
      evaluation per *step* (milliseconds against a tens-of-seconds solve). It receives the scalar blocks
      **as solved**, so it holds the transforms and maps back to physical — passing `log ω` to the closure
      would silently build the shift from the wrong field. Note `a_p` still comes from the preconditioner
      and must: it is what the velocity block inverts, so the two would otherwise disagree.
      *Measured:* live vs frozen viscosity at a developed state is a modest p50 0.96 / max 1.82 with **no
      tail** — a correctness fix, not a large lever.
      - **Where the pieces live (binding, #156 seam 3).** The `VelocityShiftParts` **protocol** lives in
        `solve/shift_basis.py`, beside `ShiftBasis` — the natural sibling: `ShiftBasis` says *how* to
        combine the buckets, `VelocityShiftParts` says *where they come from*. `FrozenViscosityVelocityParts`
        (needs only a `BlockPreconditioner`) lives in `flow/continuation.py`; only `LiveViscosityVelocityParts`
        (needs `SSTTurbulence` + the transforms) stays in `turbulence/coupled.py`. This is what lets the
        **flow-only** `MomentumShiftPolicy` carry the *same* `velocity_shift_parts` seam (it could not
        before — the protocol lived in `turbulence`, and `flow` importing it would be a cycle). Its
        `parts(flow, k_solved=None, omega_solved=None)` makes the turbulence blocks optional, so the
        flow-only policy calls `parts(flow)` while the coupled one passes all three. The flow-only path is
        still behaviour-unaffected (constant μ ⇒ frozen and live coincide); the seam exists for symmetry
        and a future variable-viscosity flow, and to delete the byte-for-byte duplicated inline pattern.
    Adding a
    basis (e.g. a Fluent-style global min-physical-time-scale) is a new `ShiftBasis`; do not branch the
    policies.
    - **First measurement of the convective basis: WORSE at a weakly-separated state, but NOT yet a fair
      test of its regime (`local_ts_ab.py`, 2026-07-24).** Probing both bases on pitzDaily checkpoints at
      `β ∈ {3, 12}` with a fresh preconditioner:

      | state | basis | β | cycles | α | ‖R‖ kept |
      |---|---|---|---|---|---|
      | known-good rel 0.052 | spectral (`w=1`) | 3 | 17 | **1.00** | **29.3 %** |
      | known-good rel 0.052 | convective (`w=0`) | 3 | 14 | 0.25 | 7.8 % |
      | plateau rel 0.032 | either | 3 / 12 | 13–163 | **0.001** | ~0 |

      At the productive state the convective basis **lowers** α (1.0 → 0.25) and the residual reduction
      (29 → 8 %), and does not help the flow or k blocks either (d(flow) −0.7 %, d(k) −0.5 %) — consistent
      with the dissipative diagonal being **load-bearing stabilization** on a wall-resolved,
      high-aspect-ratio mesh: the near-wall and recirculation cells are diffusion-controlled, so dropping
      their damping makes the coupled step overshoot and the line search clip harder. It is also cheaper
      per step at low β (14 vs 17 cycles) but degrades faster at high β (dropping the dissipative diagonal
      weakens diagonal dominance). **Caveat that keeps this open, not settled: the only state where steps
      are productive (rel 0.052) is barely separated, so this probe never exercised the developed
      recirculation the convective basis is *meant* for** — and the one genuinely separated state available
      (the plateau) is direction-limited (α = 0.001 for *every* basis and β), so it cannot discriminate
      between bases at all. Re-test at a state that is both **well separated and still productive** before
      concluding. Hence: shipped as an opt-in with the **default unchanged** (`w=1` = the historical `a_P`),
      not adopted and not withdrawn.
    - **RE-TESTED PROPERLY (2026-07-25, #28): the convective basis is NOT dominated — it is 2.2× better
      at the march's own operating point, with an optimum at Co ≈ 1.** ⚠️ **SUPERSEDED — this conclusion
      is wrong and was overturned on marches; see "the convective basis is DOMINATED" below. It is kept
      only as the third recorded instance of a single-step %/s sweep picking the wrong winner.** The
      earlier probe above rebuilt
      the continuation at the state it measured, and used a state that was barely separated; this one
      uses the **carried protocol** at a state that is both developed (`x_r/h` ~1.2) and productive
      (α = 1, 14 cycles), with β taken from the **segment-local** ratio. Residual reduction, %/s:

      | β (= 1/Co for w=0) | 0.5 | 1.0 | **1.79 ← the march** | 3.0 | 6.0 |
      |---|---|---|---|---|---|
      | `w = 1` (a_P, uniform relaxation) | **0.056** | 0.030 | 0.016 | 0.007 | 0.002 |
      | `w = 0` (convective local Δt) | 0.035 | **0.051** | **0.035** | 0.019 | 0.007 |

      The convective basis wins at every β ≥ 1 (1.7–3×) and needs fewer cycles there (10–11 vs 14). It
      has a genuine **interior optimum at Co ≈ 1** with symmetric fall-off — the canonical stability
      limit, i.e. a target with physical meaning that should transfer between cases. `a_P` has **no**
      optimum in range: its reduction follows a strict inverse law (`β × keep% ≈ 1.23` across a 12× span)
      with cycles flat below β ≈ 2, so it simply improves as damping falls and its best measured value
      is at the bottom of the sweep. Two roughly equivalent routes to ~3.5× over the march's 0.016 —
      lower β on `a_P` (0.056), or convective at Co ≈ 1 (0.051) — which do **not** compose (convective at
      β = 0.5 is 0.035).
    - **A non-uniform shift creates INTERIOR optima that the backtracking ladder cannot find — but the
    obvious fix is already REFUTED, so this is a description, not a lever.** With `w = 1` the ideal step
    length was `α = 1` at every β, so the powers-of-½ quantization cost nothing; with `w = 0` the ideal
    moves off a rung, and `backtracking_line_search` accepts the *first* rung that reduces and never asks
    whether a shorter step is better (a sufficient-decrease search, not a minimizing one), so some
    per-step residual reduction is left unclaimed. **⚠️ CONFLICT, settled — do not re-propose the
    minimizing search.** The "unclaimed reduction" argument was acted on: a minimizing search was built,
    measured and REVERTED, because a deeper residual per step bought far less recirculation development
    (see "THE LINE SEARCH TAKES THE LONGEST ADMISSIBLE STEP" below). The unclaimed-percentage figures
    that motivated it recorded no configuration and are deleted; the conflict is settled on the march
    evidence, which judges the physics rather than ‖R‖. Note the directional derivative is available
    almost free here, since the shifted solve gives `J δ = −R − β D δ` exactly.
    - **⚠️ MARCHES REVERSE THE SWEEP: on this case the productive lever is the damping LEVEL, not the
      basis (2026-07-25).** The %/s table above is single-step at one state, and it picked the wrong
      winner. Four cold-IC marches, all with the drift refresh, judged on the recirculation length:

      | arm | steps | **x_r/h** | k_peak | rel | α |
      |---|---|---|---|---|---|
      | a_P, β₀ = 2, shift carried (the shipped config) | 158 / 80 min | 1.67 | 3.10 | 8.1e-3 | 1.0 |
      | a_P, β₀ = 2, shift refreshable | 89 / 41 min | 1.22 | 2.04 | 1.3e-2 | 1.0 |
      | **a_P, β₀ = 0.5, shift refreshable** | **109 / 76 min** | **2.43** | **3.04** | **4.6e-3** | **1.0** |
      | convective at nominal Co ≈ 1, refreshable | 85 / 77 min | 1.07 | 2.58 | 1.5e-2 | **0.001 stalled** |

      At the time, `β₀ = 0.5` was the best this case had produced (a 46 % larger bubble than the shipped
      configuration). **⚠️ That is now SUPERSEDED and REVERSED — see the post-`a_P`-fix re-profile below;
      `β₀ = 0.5` is currently the worst of the three and `β₀ = 2` is right.** The convective arm did not
      merely lose, it **stalled** — α pinned at the 0.001 ladder sentinel with the residual frozen to
      five figures for four consecutive steps.
      - **The single-step sweep rated convective at Co ≈ 1 (0.051 %/s) level with a_P at β = 0.5 (0.056)
        and better than a_P at every β ≥ 1.** The marches say the opposite. That is the **second** time
        in one session that a single-state single-step ‖R‖ measurement chose the wrong winner (the first
        was the log-space ω shift, `.claude/rules/turbulence.md`). **Treat %/s sweeps as a way to find
        candidates, never as a way to choose between them** — the choice needs a march judged on physics.
        (The 2026-07-26 re-profile is the **third** instance: the same sweep's `Co ≈ 1` optimum did not
        survive a constant-β march either.)
      - **This does NOT close out local timestepping (open question).** The Co ≈ 1 optimum was measured
        with the shift frozen at the cold initial condition, so the Courant number it optimized was
        *nominal*: `d_conv` was built from potential-flow mass fluxes, not the developed ones. With a
        refreshable shift `Co` finally means what it says, and the optimum can move — if developed fluxes
        exceed cold-IC fluxes, nominal Co ≈ 1 corresponded to a larger *actual* Co, which would leave the
        refreshable arm under-damped and is consistent with the stall. Re-sweep Co **on marches** with the
        refreshable shift before concluding; do not reuse the frozen-shift optimum.
        **— CLOSED 2026-07-26 by the re-profile below: re-swept, and the under-damping explanation is
        refuted. The pure convective basis is dominated at every damping level tested.**
    - **⚠️⚠️ THE EUCLIDEAN ‖R‖ MIS-RANKS STATES — a converged field scores WORSE than a badly wrong one
      (measured 2026-07-26). This invalidates ‖R‖-based comparisons throughout this file; read this
      before trusting any of them.** Raw norm against a scale-free per-cell measure, four states, same
      mesh and model:

      | state | raw ‖R‖ | **`\|R_ω\|/ω` median** | `\|R_k\|/k` | flow | x_r/h |
      |---|---|---|---|---|---|
      | cold initial condition | 286.3 | 4.54e-05 | 5.73e-05 | 2.52e-01 | 0.00 |
      | cold march, step 90 | **3.68** | 2.44e-05 | 3.41e-05 | 5.21e-02 | 1.22 |
      | OpenFOAM reference (converged) | 21.2 | **3.48e-06** | 3.06e-06 | 6.74e-03 | 7.74 |
      | our own warm-started root | 4.57 | **2.32e-06** | 2.84e-06 | 1.27e-03 | 8.07 |

      **The scale-free measure ranks all four correctly; the raw norm inverts the middle two**, rating a
      state whose recirculation is six times too short (3.68) above both converged fields. Two
      compounding causes, both measured:
      - **The ω residual is not dimensionless.** OpenFOAM's field is converged to a *relative* imbalance
        of 3.5e-6 — 7× better than the cold march's 2.4e-5 — but its ω is sharp and developed, so the
        same relative error yields a far larger absolute residual. Raw ω residuals cannot be compared
        across states with different turbulence levels, which is exactly what a march does.
      - **At converged states the ω L2 is a TEN-CELL statistic.** At the reference the top **1** cell
        carries **41.9 %** of ‖R_ω‖² and the top 10 carry **75.9 %** (the sharp near-wall peaks); at the
        under-developed marched state it is spread out (top 1 = 1.1 %, top 10 = 5.4 %). A metric that
        concentrates into a handful of cells precisely as the solution becomes correct is backwards.
      - **The flow block's raw residual already ranks correctly** (2.5e-1 → 5.2e-2 → 6.7e-3 → 1.3e-3).
        Only k and ω are mis-scaled — and ω is ~100 % of the norm.
      **⚠️ CORRECTION (same day): the "mis-ranks states" framing above is OVERSTATED — do not repeat
      it.** The OpenFOAM field is *not* a root of these equations (another discretization, a different
      wall treatment, an instantaneous snapshot of an unsteady shear layer), so a residual measure
      rating it by its own nonzero imbalance is **correct behaviour, not a defect**. Demanding that a
      measure rank a foreign field as converged is a broken test, and the row-equilibrated measure does
      not do it either (cold march 1.16e-2 vs that field's 1.34e-2, scales rebuilt per state). On
      states that *are* ours both measures already rank correctly: our warm-started root scores best
      under the raw norm (0.98 vs the march's 3.68) as well as the scaled one.
      **What actually survives, and it is enough:** the raw norm is ~100 % ω, so it does not *report*
      flow progress. Measured directly at a state known to be near the correct root — the warm-start
      run — raw ‖R‖ moved 13.87 → 13.70 (**−0.2 %, reading as stalled**) while the flow block fell
      6.30e-3 → 5.03e-3 (**−20 %**). That is what starves the line search and SER of flow information,
      and it is the real case for equilibration; the block spread narrows from ~100 %-ω to within ~10×
      across blocks.
      **Consequence (binding):** SER's β ramp, the line-search acceptance, the divergence guard, the
      convergence test, and every β / basis / preconditioner comparison recorded in this file are
      computed on a measure that is ~100 % one block. This is the concrete,
      quantified case for row equilibration (#29) — divide each row by its own scale so `|R_ω|/ω` is
      what is measured — and for per-block reporting (#24). Note this is **not** the earlier claim that
      "the answer is unreachable by descent": that was measured against the OpenFOAM field as the
      endpoint and was **wrong**, because that field is another discretization's instantaneous snapshot
      and not a root of these equations. Our own root scores 2.77–4.57, *below* the cold march's 3.68 —
      the landscape around the true solution is fine.
    - **⚠️ THE COLD-START CRAWL IS A REACHABILITY PROBLEM, NOT A WRONG ROOT — settled 2026-07-26 by a
      warm start, and this reframes every globalization result below.** A cold march reaches only
      `x_r/h` 1.22 in 91 steps against the reference's 7.74, which is consistent with two completely
      different stories: the solver cannot *reach* the right root, or it converges correctly to its
      *own* root, which has a short bubble. Starting **from** the time-accurate reference separates
      them, and the answer is unambiguous — the root is ours and it is in the right place:

      | arm (near-wall ω blend) | x_r/h | k_peak | ‖R‖ from → after 5-6 steps | flow | k |
      |---|---|---|---|---|---|
      | shipped power mean, `p = 2` | 7.74 → **7.82** | 5.03 | 21.20 → **13.70** | 6.7e-3 → 5.0e-3 | 2.0e-2 → 1.5e-2 |
      | `max` limit, `p = 60` | 7.74 → **7.99** | 5.03 | 20.67 → **6.70** | 6.7e-3 → **1.8e-3** | 1.2e-2 → **2.6e-3** |

      Every block descends and the reattachment holds. **So no closure/model work is required to get
      the bubble** — the closure was never the problem (which also confirms the older three-way
      verification recorded in `.claude/rules/turbulence.md`), and the entire remaining gap is the
      solver's inability to travel from a cold start to a root it is perfectly happy to sit on.
      - **Consequence for what to build:** de-emphasizing ω in the *measure* is now a justified lever
        rather than a guess, because a reachable correct root is known to exist. The target is
        quantitative: along the straight segment from a cold-march state to the reference the total
        residual **peaks at ~5.9×** while the **flow block falls monotonically 7.7×** (5.2e-2 →
        6.7e-3). Any measure that lets the march traverse that must stop ω's few-cell L2 from vetoing
        flow progress. The acceptance rule currently tolerates `1.107×` (`RelaxedFarFromRoot` at
        rel 1.3e-2), i.e. **~5× too little**.
      - **The ω-dominance pathology is visible even at the root.** In the `p = 2` arm, ‖R‖ moves
        13.87 → 13.77 → 13.74 → 13.72 → 13.70 (−0.2 % per checkpoint, reading as "stalled") while the
        flow block falls 6.30e-3 → 5.03e-3 (−20 %). Since this state is *known* to be near the correct
        root, that cannot be confused with a genuine stall: it is the global norm failing to report
        flow convergence. This is the concrete case for per-block reporting (#24) and row
        equilibration (#29).
      - **A `max` near-wall blend is better on CONVERGENCE, not only on accuracy.** `.claude/rules/turbulence.md`
        justified `p → ∞` on agreement grounds (per-wall-cell `|R_ω|/ω` 3500× smaller) and left the
        default open as "a model decision". This adds an independent argument: from the same start it
        reaches **2× the residual depth** (6.70 vs 13.70) and a **2.8× lower flow residual** in the same
        number of steps, while `p = 2` flattens. Note the ω **L2 at the reference** barely moves between
        blends (21.20 vs 20.67, 2.5 %) — that norm is a handful of ω~1e5 cells and is not the quantity
        that discriminates, which is itself a caution against judging the blend on ‖R‖.
      - **Honest caveat on the number:** `x_r/h` does not settle exactly on OpenFOAM's 7.74. It creeps
        to 7.82 (`p = 2`) and 7.99 (`p = 60`, still rising when measured) — a ~1–3 % longer bubble.
        Report that as a solver-to-solver difference, not as a match; it is expected for a
        wall-resolving closure on a wall-function mesh, and it is small beside the cold-start gap.
      - **Also measured: the residual at the reference is UNCHANGED by the 2026-07-25/26 fixes.**
        Current code gives flow 6.74e-3, k 1.99e-2, ω 21.2 against the 2026-07-24 record's ~6e-3,
        ~1e-2, ~20. The fixation-row fix and both `a_P` fixes mattered for the **march**, not for the
        root.
      - **Trap that cost an hour here (binding for any future reference comparison):**
        `compare.read_openfoam_reference()` used to read `runs/kwsst`, the **corrupt steady** case, so a
        probe calling it silently inherited the inlet checkerboard — ω spanning 0.03 to 1.15e8, with
        **ten cells carrying 100 % of the ω residual** and a total ‖R‖ of 4.1e8. That produced a
        spurious "10⁸ residual ridge blocks the path" and a spurious `cos(step, error) = −0.087`
        (the true value against the transient field is **+0.13**, i.e. weakly *aligned*). The loader now
        reads `of_transient/0.14`. **Sanity-check any reference measurement against the recorded
        ‖R‖ ≈ 20 before drawing conclusions from it.**
    - **⚠️ THE CRAWL IS A CORRECT PSEUDO-TIME INTEGRATION OF A GENUINELY LONG TRANSIENT — measured
      2026-07-27, and it re-scopes the "de-emphasize ω in the measure" lever above.** The `a_P` shift is
      backward-Euler local time stepping: `β·a_P` on the diagonal is the transient term `ρV/Δτ`, so the
      per-step pseudo-time is `Δτ = α·V/(β·a_P)` (α the accepted line-search factor, ρ = 1). Accumulating
      that per cell over two stored cold marches (`profile_base`, β ≈ 1.9; `basis_march_aP05`, β₀ = 0.5),
      sampled in the recirculation region behind the step, settles what the reachability crawl actually is:
      - **`a_P` is ~constant (~8.0e-3) for the whole march.** The potential-flow seed already carries
        free-stream-magnitude velocity, so the momentum diagonal barely moves — hence `Δτ` per step is
        *fixed and tiny* (~6e-5 s at the median cell), regardless of how the bubble develops.
      - **`x_r/h` grows smoothly, monotonically, and decelerating with accumulated pseudo-time in both
        arms — no stall, no reversal.** The step direction is never the problem; every step buys real
        bubble. `base` reaches `x_r/h` 1.22 at ~5 ms of bubble-median pseudo-time in 90 steps; the
        β₀ = 0.5 arm reaches 2.43 at ~28 ms in 109 steps. Physical yardstick: the free-stream
        flow-through of the reattached bubble length (7.74·h ≈ 0.20 m at 10 m/s) is ~20 ms, and `base`'s
        growth extrapolates to reach 7.74 near **~55 ms ≈ ~800 steps at this `Δτ`**. So the march has
        elapsed only a small fraction of the transient — the crawl is *insufficient elapsed pseudo-time*,
        not a wrong direction, a bad merit function, or a stuck state.
      - **Lower β = larger backward-Euler step = further per step (2.43 vs 1.22) — this confirms `Δτ` is
        the lever.** The two arms trace the same qualitative decelerating growth but do **not** collapse
        onto one `x_r/h`(pseudo-time) curve: the β₀ = 0.5 arm sits at ~1.6–2× more pseudo-time per unit
        bubble, because a larger implicit step integrates the transient more coarsely and its pseudo-time
        bookkeeping overstates true transient progress. The small-step arm is the truer `x_r(t)`; do not
        read the imperfect collapse as a defect.
      - **CONSEQUENCE (binding): no merit function, acceptance rule, filter, or shift *basis* changes
        this** — every one of those is a direction/measure lever, and the direction is fine. The only
        levers are the **effective `Δτ` per step** (`α·V/(β·a_P)` — a larger stable step) or a **different
        homotopy/seed nearer the developed bubble** (physical continuation in Re or a `ν_t` ramp, a
        coarse-grid or eddy-viscosity-augmented start). A measure change cannot *manufacture* pseudo-time,
        which bounds the "de-emphasize ω in the measure" target above (#24/#29): worth it for readable
        per-block reporting, **not** as the reachability fix it was framed as.
      - **WHAT LIMITS `Δτ`: the cold-start β floor is a NONLINEARITY, and diffusion continuation lifts it
        — measured 2026-07-27. *(harness not in the repository — this finding cannot be re-adjudicated as recorded)*** A single
        shifted cold step at the target Re = 25000 is stable at β = 2 (α = 1, ‖R‖ ×0.49) and β = 0.5
        (α = 0.5) but **blows up at β = 0.25** (ω → 5.6e32, no reducing rung) — the recorded floor. Raising
        the molecular viscosity (a clean Reynolds continuation, self-consistent seed, no state
        perturbation) removes it: at Re = 2500, β = 0.5 goes α = 0.5 → **1** and β = 0.25 becomes finite
        and productive (α = 0.5, ‖R‖ ×0.49); at Re = 250, β = 0.5 takes a near-Newton step (‖R‖ **×0.045**,
        22× in one step). So the floor is set by the convective nonlinearity, and reducing it (diffusion
        homotopy) buys a lower β = a larger `Δτ` from step 1 — the automatable, knob-light lever (one
        scalar that **dissolves at the target Re**, like the shift, so the root is unchanged).
      - **A `ν_t` seed applied by perturbing the k/ω *state* BACKFIRES — do not.** Scaling ω down to raise
        `ν_t` unbalances the ω transport equation, and since ‖R‖ is ~100 % ω the coupled step then fights
        that artificial deficit: measured α = 0 (no reducing rung) at β = 2 where the unperturbed state
        gives α = 1. An eddy-viscosity seed must add diffusion to the **momentum closure** (a `μ_eff`
        floor, ramped out), *not* to the k/ω fields — i.e. it is a spatially-varying diffusion
        continuation, the same family as the Reynolds ramp above.
      - **This is the `β × travel` finding seen in the residual, and it explains why ‖R‖ points opposite
        to the physics.** On every shifted row `R(φ+δ) ≈ −βDδ = −(ρV/Δτ)δ ≈ −ρV·(dφ/dt)` — the *physical
        unsteady term*, nonzero for the entire transient and independent of step size. The equilibrated
        residual therefore literally cannot fall until the transient completes; it is behaving exactly as
        an unsteady residual should, which is why judging on `x_r/h` (never a residual) is mandatory here.
      - **The `of_transient` reference has NO bubble-growth curve — it was restarted from the developed
        steady field.** `of_transient/0/U` carries `location "2000"` in its header (copied from the steady
        run's converged step), so `x_r/h` ≈ 7.74 at *every* written time including t = 0. The transient
        confirms the developed state is stable; it is **not** a growth transient and cannot be overlaid
        against the march's `x_r` vs pseudo-time. Treat 7.74 as the asymptote only.
    - **⚠️ RE-PROFILED AFTER THE `a_P` FIX (2026-07-26) — the two conclusions above REVERSE. Read this
      bullet, not them.** The flux-continuous (harmonic) face viscosity and the wall-model boundary
      viscosity changed `a_P` itself, and the shift is `β·a_P`, so **every β calibration measured before
      that fix is void** — the harmonic mean is ≤ the arithmetic one it replaced, so the same β now buys
      *less* damping and the optimal β moves **up**. Three cold-IC marches, shipped `solve_coupled`,
      drift refresh, judged on the recirculation length:

      | arm | steps | **x_r/h** | k_peak | rel | α (tail) | cyc/step |
      |---|---|---|---|---|---|---|
      | **`a_P`, β₀ = 2 (the shipped default)** | **67** | **0.99** | 1.61 | **1.4e-2** | **1.00** | **12.5** |
      | `a_P`, β₀ = 0.5 (the former "best") | 16 | 0.39 | 1.41 | 9.5e-2 | **0.13** | 29.0 |
      | convective, Co adapted from α | 2 | — | — | 8.9e-1 | 0.125 | 22 → killed |

      `β₀ = 0.5` is now **under-damped and stalling** (α 0.13, 29 cycles/step, bubble frozen at 0.39),
      exactly the failure the convective arm shows — and for the same reason, too little effective
      damping. **Take `β₀ = 2`.**
    - **The convective basis (`w = 0`) is DOMINATED — settled by a controlled 2×2 plus a β sweep, do not
      re-open on a %/s sweep (2026-07-26).** Three steps from the same cold IC at **constant** β
      (`exponent = 0`, so β is genuinely fixed and the arms are compared at equal damping, not equal
      residual history). The probe reproduces the real march bit-for-bit at step 0, which is the harness
      validation that must precede any such claim:

      | basis | β | cyc 0/1/2 | α 0/1/2 | rel after 3 steps |
      |---|---|---|---|---|
      | **`a_P`** | 2 | 15 / 14 / 13 | **1.000 / 1.000 / 1.000** | **0.2995** |
      | convective | 1 | 18 / 14 / 13 | 0.125 / 0.125 / 0.125 | 0.8035 |
      | convective | 3.3 (= matched effective damping) | 36 / 22 / 24 | 0.250 / 0.125 / **0.0039** | 0.8980 |

      The convective basis is clipped at **every** step while `a_P` takes full steps at the same cost,
      and at *matched* effective damping it is worse still and collapses into the ladder by step 2
      (α → 0.0039 → 0.0020, the 0.001 sentinel again). Three candidate explanations were each proposed
      and each **refuted by measurement** — record them so they are not re-proposed:
      - *Preconditioner inconsistency* (the MSIMPLER Schur ignores the shift, which for a non-uniform
        basis is a spatially-varying error): refuted by the 2×2 below. **Issue #163, closed as
        refuted.**
      - *Damping level / wrong Co calibration* (the convective diagonal is only ~0.61 of `a_P`, so
        "Co = 1" under-damps 3×): refuted by the β = 3.3 row — matching effective damping does not
        recover α, and makes progress *worse*.
      - *Weakened diagonal dominance / near-wall cells left undamped*: refuted directly — `a_P + βd` is
        **more** diagonally dominant than `a_P`, and the measured convective share bottoms out at
        p1 = 0.30 (never near zero), with the least-damped cells **mid-channel**, not at the wall.
      - *The recirculation is left undamped* (a convective-only `Δt → ∞` where the mass flux vanishes,
        i.e. no damping in the most nonlinear region): refuted, and the correlation runs the **other**
        way. At a developed state (`x_r/h` 1.22) the reversed-flow cells have a **higher** convective
        share than the forward-flow ones (median 0.778 vs 0.652), and they are strongly
        *under*-represented among the least-damped — 0.00× the base rate in the bottom 1 % by share,
        0.14× in the bottom 5 %. The least-damped 2 % are at `x/h ≈ +10.8`, `y/h ≈ −0.10`, moving at
        **7.03 m/s against a 5.10 m/s domain median**: the fast downstream core, where the developed
        eddy viscosity makes the viscous diagonal dominate.
      **No mechanism is offered — four were proposed and all four were refuted by measurement. The
      empirical result stands without one; do not add a fifth without a measurement that discriminates
      it.** What *is* established: the shipped `w = 1` basis is the classical local time step (the shift
      `β a_P` is `V/Δt` with `Δt = Co·V/λ`, `Co = 1/β`, `λ` the **combined** convective + viscous
      spectral radius — Blazek's form), and it holds `α = 1.0` for 90+ consecutive steps. `w = 0` is
      that same formula with the viscous stability limit deleted, on a mesh where the developed `ν_t`
      makes the viscous half the **larger** one almost everywhere (share median 0.66). So this is not
      evidence against local timestepping; the default *is* local timestepping and it is what works.
    - **The Schur's blindness to the shift is NOT a defect — measured, do not "fix" it (2026-07-26,
      #163).** `apply_at` feeds the velocity block the shifted diagonal `a_P + β d`, while the MSIMPLER
      Schur uses `Q̂/k` calibrated from the **un-shifted** diagonal, i.e. it ignores the shift entirely.
      That looks like an inconsistency, and for a non-uniform basis the discrepancy is spatially varying
      (`1/(1 + β·share)`, share 0.30–0.97) rather than a global scalar. It costs nothing. A 2×2 at fixed
      β, varying only `schur_scaling` (`simple` uses the shifted `a_p` and is consistent by
      construction):

      | basis | `msimpler` (shift-blind) | `simple` (consistent) | ratio | α (both) | rel (both) |
      |---|---|---|---|---|---|
      | `a_P`, β = 2 | **15** cyc | 36 cyc | 2.4× | 1.000 | 4.8530e-01 |
      | convective, β = 1 | **18** cyc | 34 cyc | 1.9× | 0.125 | 9.2719e-01 |

      Within each basis `α` and the residual ratio are **bit-identical across all three steps measured**
      — the "a preconditioner changes cost, not the converged step" property, which also confirms these
      solves genuinely converge. There is **no interaction**: the consistent Schur is uniformly ~2×
      worse, and *less* bad on the convective basis (1.9× vs 2.4×) — the opposite of the hypothesis.
      `Ŝ` is an approximation chosen for **spectral quality**, not a derivation of the true Schur
      complement; MSIMPLER's whole premise is replacing `a_P` with a velocity-independent mass-matrix
      stand-in, so being more faithful to `(A + βD)⁻¹` does not make it a better preconditioner. This
      also confirms the earlier "shift-consistent Schur is strictly worse at every β" finding **does**
      transfer to a non-uniform basis, contrary to what was argued when #163 was filed.
    - **⚠️ CONFLICT — "neither α nor the cycle count can serve as a controller target on this problem" is
    contradicted by the shipped default; do not act on either side without re-measuring.** One side: a
    single-step β/basis sweep found α and the cycle count constant across its whole range while the
    efficiency varied, so only residual reduction per unit time discriminated *(configuration not
    recorded — no preconditioner, forward solver or tolerance — and taken under the superseded
    ω-dominated norm; re-measure before relying on it)*. The other side: `DualTimeControl`, the
    **α-driven** ramp, is the shipped default for a dual-time observed march and is the arm that reaches
    a developed recirculation, while the residual-keyed control pins β on the flat `β×travel` plateau.
    The likely reconciliation is that a *single-step* sweep at one state cannot see the α signal a
    dual-time inner loop produces — but that is an inference, not a measurement.
    - **⚠️ SUSPECT — the "plateau is a step-DIRECTION problem" conclusion was measured through a broken
      preconditioner (see the fixation-row/`1/ω` bug below) and a corrected re-measurement CONTRADICTS
      part of it. Re-derive before relying on any of it (#31).** As originally written: every basis/β
      combination at rel 0.032 gave α = 0.001 and zero descent, so neither a preconditioner, nor a
      per-block β, nor a shift basis could move it — the argument being that a **preconditioner** only
      changes Krylov cycles, since for fixed `J`, `β`, `d`, `R` the shifted step `δ` is unique regardless
      of `M`.
      - **That uniqueness argument is only valid for a CONVERGED linear solve, and the coupled forward
        solver is deliberately inexact** (`_COUPLED_FORWARD_SOLVER` runs at `rtol = 1e-3`). At a finite
        tolerance `δ` depends on `M`, so a preconditioner *can* change α. Measured directly: refreshing
        the scalar AMGs mid-march on the pitzDaily cold-IC run took α from **0.5 → 1.0** (sustained over
        the following steps) while cutting cycles 53 → 10. So "a preconditioner cannot change α" is false
        as stated here; state it as "cannot change the *converged* `δ`, hence not the fixed point".
      - The α = 0.001 observations themselves came from probes at a state reached through the
        `1/ω` preconditioner mis-scaling, i.e. through solves that were not converging. Treat the whole
        plateau diagnosis as unverified until re-measured on a cold-IC march with the fixed code.
      - What is *not* in doubt: a **shift basis** only redistributes damping, and altering the operator
        itself (the pseudo-time shift, a grad-div / augmented-Lagrangian augmentation vanishing at
        `∇·u = 0`, or physical continuation) is what changes `δ` at convergence.
  - **Two injected seams**, both `Protocol`s: the physics comes from a **`ShiftPolicy`**
  (`shift_term(φ) -> ShiftTerm(diagonal, make_preconditioner)`; `ShiftTerm.diagonal` is the full-state
  base shift, `make_preconditioner(β)` the frozen shifted `M`), and the per-attempt accept/reject
  decision from a **`StepAcceptance`** (`accept(candidate_norm, residual_norm, residual_norm_0,
  attempt) -> bool`). The escalation-loop *mechanics* (grow `β`, cap at `max_escalations`, carry the
  best candidate) stay in the engine; only the decision is delegated. Default acceptance is
  **`DivergenceGuard(divergence_cap=10.0)`** — accept unless the candidate is non-finite or exceeds
  `divergence_cap·‖R₀‖` (a divergence guard, not a descent test, since the march is non-monotone); a
  monotone / forcing rule is a drop-in `StepAcceptance` — do **not** hardwire an acceptance test into
  the `while_loop`. So the engine is
  reusable for **any** nonlinear residual (reaction/energy/turbulence), not just the coupled flow —
  verified in `tests/unit/test_pseudo_transient.py`, which drives it on a scalar root with a trivial
  policy (no mesh, no flow). The flow application is `aquaflux/flow/continuation.py`'s
  `MomentumShiftPolicy` (velocity-block `a_P` shift + shifted SIMPLE preconditioner), configured into a
  `PseudoTransientStep` by the `momentum_continuation(assembler, …)` **factory** (which builds the
  block preconditioner and injects the `DivergenceGuard` + adjoint factory) — **no wrapper/adapter
  class**, since `PseudoTransientStep` is itself the `ForwardStep`. The scalar application is
  `aquaflux/turbulence/continuation.py`'s `ScalarShiftPolicy` (the transport operator diagonal — the
  scalar `a_P` analogue from `scalar_transport_shift_diagonal` — as the base shift, with the frozen
  scalar-transport AMG reused **unshifted** as `M`, since the shift only adds positive diagonal),
  globalizing the stiff k/omega solves via `scalar_pseudo_transient_solve` — the **only** scalar path
  the SST driver supports (the fixed-count Newton sub-solve was removed). When a new nonlinear residual
  needs pseudo-time globalization, write a `ShiftPolicy` — do **not** re-implement the march.
  - **`stepper()` returns `(phi_next, cycles, alpha)` — ONE step method on the whole `ForwardStep` protocol,
    counted/uncounted pair deleted (binding).** Every strategy reports its step's restart-cycle count
    (`DampedNewtonStep` gets it from `newton_correction`, which now returns `(delta, r, cycles)`); a
    consumer with no use for it drops it (`phi, _ = step(…)`). A `counted_stepper()` sibling existed
    briefly, with `stepper()` forwarding to it and dropping the count — deleted for the same reason as
    `solve_linear_counted` above, and note it had **no production consumer at all** while it existed.
    The reported count is the **accepted** attempt's, not the sum over rejected escalation attempts —
    the cost of the step actually taken. **A step whose every attempt was rejected reports `0`**
    (`best_cycles` is only written on acceptance): a consumer must treat `0` as *no measurement*, not
    as *free*, or a rejected step reads as the cheapest in the march. Consumed by `forward_march`
    (below); dropped by `_forward`.
  - **The count is NOT carried out of `_forward`'s `while_loop` (binding).** Two reasons, both concrete:
    it would put an `int32` in the primal output of `_implicit_solve`'s `custom_vjp`, so the reverse rule
    would have to handle a `float0` cotangent leaf in the most correctness-critical function in the
    package, for a number the differentiated path can never use; and it would force the *generic* Newton
    loop to pick which step's count survives (last / max / sum), which is a reporting policy the solver
    has no business owning. Per-step cost is observed eagerly instead, by `forward_march`.
  - **The `growth` parameter is NOT a performance regression — hypothesis raised, then DISPROVEN by
    measurement (2026-07-25, settled; do not re-open on the original evidence).** A
    `forward_march(..., max_steps=1)` on the pitzDaily *plateau* state ran **27 min 42 s** without
    returning, shortly after `backtracking_line_search` gained its `growth` argument, and the
    coincidence was recorded here as a suspected regression with a traced-bound hypothesis. Both the
    hypothesis and the attribution are wrong:
    - **`admissible` was never inside the loop.** `admissible = growth * reference_norm` is computed
      *outside* the ladder body (`implicit.py`), so the bound is a loop-invariant closure capture
      exactly as the bare `reference_norm` was before. The "compute it outside" fix that was filed as
      a candidate is already the code as written.
    - **The default bound is not even traced.** `MonotoneLineSearch.growth` returns a **concrete**
      `jnp.asarray(1.0)`, not a tracer, so under the default the comparison is structurally identical
      to the pre-change one. The recorded observation that the slowdown appeared *with the monotone
      schedule active* was read as evidence **for** the traced-bound hypothesis; it is evidence
      against it.
    - **The decisive measurement.** A jitted `_march_step` at the plateau with **`max_escalations = 0`
      and `MonotoneLineSearch`** — i.e. exactly ONE shifted linear solve, default growth, no
      escalation ladder, strictly less work than the step that took 27 min — ran **> 57 min without
      returning**. Since that configuration contains neither the escalation ladder nor any non-default
      growth, neither can be the cost.
    - **The ladder was never a plausible candidate on size grounds either:** one coupled residual
      evaluation is ~0.3 s, so all 11 rungs cost ~3 s against a 27-minute step.
    - **What the cost actually was: a PRECONDITIONER BUG introduced by the fixation-row change in the
      same PR — found and fixed the same day.** The scalar block's frozen preconditioner was rescaled
      by `1/(dφ/dw) = 1/ω` on *every* row, including the 472 near-wall ω **fixation** rows, whose row
      (`LogRatioRow`, written in the solved variable) has derivative **1** against the frozen
      operator's unit identity row — a `1e-5` eigenvalue cluster that stalls GMRES. So the coupled
      solves were not converging and ground to their step cap. Fixed by giving each `FixationRow` its
      own derivative; measured 27× better linear residual at fixed cycle count (see
      `.claude/rules/turbulence.md` for the full finding).
      **The chain of three wrong attributions is the lesson:** the `growth` argument, the fixation row
      itself, and nested `jax.grad` in the residual were each blamed in turn because all three landed
      in the same window. What discriminated was measuring *components* rather than the whole step —
      the residual is 8.1 ms, `jvp/residual` is **1.5×** (healthy AD, killing the nested-grad theory),
      and one 120-vector restart cycle is ~1.5 s, so a healthy solve is seconds. Any step costing
      minutes therefore had to be **iteration count**, not per-matvec cost — which pointed straight at
      the preconditioner and away from everything else.
    - **Staleness is still real but was NOT the main term here.** The per-solve wall-time figures once
    quoted here (and elsewhere in this file) named no preconditioner, forward solver, restart or state,
    so they are deleted rather than defended. Rebuild-vs-carry belongs in the refresh-trigger
    calibration (#17) on a *cold-IC* march; re-measure it now that the solves converge, since the
    pre-fix carried-vs-rebuilt comparison (#31) was taken through the broken preconditioner and cannot
    be trusted.
    - **Methodological trap this cost an hour to learn (binding for future probes):** timing a
      `solve_linear` **eagerly** measures nothing comparable to the march, which runs the whole step
      inside one `eqx.filter_jit`; eager JAX dispatches each Krylov operation separately. An eager
      version of this same probe was still inside a single solve after 60 min of busy CPU. Always
      time the jitted `_march_step`, and prefer *differencing two configurations* over instrumenting
      inside the compiled region.
    - **⚠️ THE CARRIED PROTOCOL — how to probe a coupled step at all (binding; four separate sweeps
      were invalidated by getting this wrong in one session).** A probe that loads a checkpoint and
      calls `coupled_continuation(coupled, checkpoint)` builds a **self-consistent** (state, shift,
      preconditioner) triple. **The march never occupies that configuration**: it freezes the shift at
      the cold IC and carries it through every refresh, so its shift always *lags* the state. The
      difference is not academic — at the same state and the same β, a rebuilt continuation took
      **148 cycles and found no descent** where the march takes **14 cycles at α = 1 and descends**.
      Reproduce a refresh instead:
      ```
      cold_cont = coupled_continuation(coupled, cold_ic, method=...)
      cont      = coupled_continuation(coupled, state, method=..., reuse=cold_cont.shift_policy)
      ```
      **Validate the harness before trusting it**: the `reuse=` build at the march's own β must
      reproduce the march's cycle count and α from its log. When it finally did (14 cycles, α = 1.0000,
      +0.677 %), every earlier number in that session turned out to have been measuring something else.
    - **Take β from the SEGMENT-LOCAL residual ratio, never the global one (this is what made three of
      those four sweeps wrong).** SER's reference is reset at every refresh, so on a refreshing march
      β stays pinned near β₀ = 2 rather than decaying — see the `RelaxationSchedule` section. Computing
      "the march's operating β" from the global ratio gave 0.022 where the true value was 1.79, an 80×
      error that silently produced an apparently-solid "ascent at every Courant number" result.
    - **A median is the wrong statistic for a shift diagonal.** Comparing rebuilt against carried gave
      a median ratio of 0.96 — "no effect" — while the effect was a >2× tail over 15 % of ω cells,
      reaching 24×. Report p99/max and the fraction above 2×, per block; a whole-state statistic also
      hides that the pressure block is identically zero (20 % of the vector) and dilutes everything.
    - **Do not probe step cost at the plateau.** Every configuration measured there costs ~1 h/step
      because the state is direction-limited (α = 0.001 at every β and shift basis) *and* maximally
      stale for a carried preconditioner. Cost questions belong on a cold-IC march, where steps are
      accepted on the first attempt.
  - **⚠️⚠️ THE EQUILIBRATED MEASURE BARELY FALLS ON A MARCH THE EUCLIDEAN NORM LOVES — the most
    consequential measurement of 2026-07-27. Read this before designing around either measure.**
    Evaluating the row-equilibrated measure along the *default* march's own checkpoints (the march whose
    Euclidean residual falls 78×, from 2.86e2 to 3.68):

    | step | Euclidean | equilibrated | u0 | u1 | cont | k | ω | x_r/h |
    |---|---|---|---|---|---|---|---|---|
    | cold | 2.86e+2 | 2.229e-2 | 4.76e-3 | 1.80e-3 | 2.30e-3 | 1.64e-2 | 1.41e-2 | 0.00 |
    | 25 | 3.96e+1 | 2.175e-2 | 5.99e-3 | 6.21e-3 | 1.14e-3 | 1.14e-2 | 1.64e-2 | 0.05 |
    | 90 | 3.68e+0 | 1.158e-2 | **5.23e-3** | **4.61e-3** | 6.84e-7 | 7.55e-3 | 5.34e-3 | 1.22 |

    **The Euclidean norm falls 78×; the equilibrated measure falls 1.9×.** Composition: continuity
    improves ~3000× (a negligible absolute contributor), k and ω ~2.5× each, and the **velocity blocks
    get WORSE** — one component 1.80e-3 → 4.61e-3, **2.6× worse** — over a march the Euclidean norm
    reports as converging.
    - **⚠️⚠️ THE TABLE ABOVE IS AN ARTIFACT OF THE MEASURE'S OWN CONSTRUCTION — MEASURED 2026-07-27,
      and it supersedes the reading that stood here before.** On every row the shift owns, the
      equilibrated measure after a full step is **`β × per-step travel`, not a distance to the root.**
      `coupled_scaled_norm` takes its velocity/k/ω row scales from `shift_policy.shift_term(state)
      .diagonal` — *the very array the shift multiplies* — so with `(J + βD)δ = −R` giving
      `R(φ+δ) = −βDδ + O(‖δ‖²)`, the equilibrated row is exactly `β|δᵢ|`. Continuity carries `D_c = 0`
      (the shift packs `jnp.zeros(n_cells)` on pressure), so it is **annihilated to first order**
      whatever the physics does. Measured against real shifted solves at real march checkpoints
      (harness not in the repository), actual ÷ predicted `βDδ` floor:

      | state | β = 2 | β = 1 | β = 0.25 |
      |---|---|---|---|
      | cold | 0.995–1.008 | 0.975–1.060 | *diverges, see below* |
      | g0020 | 1.013–1.038 | 1.049–1.189 | *NaN, see below* |
      | g0045 | **1.000 ×4** | 0.999–1.005 | 0.994–1.046 |
      | g0090 | **1.000 ×4** | 1.000–1.003 | 0.999–1.023 |

      Continuity's floor is *exactly* zero everywhere; its actual residual after the step is 5.9e-7 at
      g0090 against 6.8e-7 before. Nonlinear defect is 0.1–0.2 % of the block value at the developed
      states, and the Krylov residual is 1e-11–1e-13 throughout — so this is neither nonlinear
      truncation nor solver inexactness. **The identity is β-independent**: it holds across a factor of
      eight in β, which is much stronger evidence than the operating point alone.
      **Therefore: continuity's ~3000× is first-order annihilation of an unshifted row; the velocity
      blocks' 2.6× "degradation" is the steps getting BIGGER. Neither is a statement about the flow —
      stop citing them as one.** The measure cannot fall below a floor proportional to `β ×` step while
      β ≈ 2, which is the whole explanation of "equilibrated stalls at 1.9× while Euclidean falls 78×":
      the Euclidean norm has no β-proportional floor. Both measures were correct about what they
      actually measure; neither was measuring convergence.
    - **⚠️ CORRECTION (2026-07-27, from reading the reference coupled p–U C++): the MEASURE is sound —
      the `β×travel` is aquaflux feeding it the wrong residual, not a flaw in the measure's
      construction.** The reference code's scaled-residual convergence measure is the *same* construction
      (divide each row by its diagonal coefficient, then normalize by field magnitude) and is robust
      there. What differs is the residual each divides:
      - **The reference measures the residual of the equation it actually solves** — `transient + flux` =
        `ρV/Δτ·(φ − φ⁰) + flux(φ)`, scaled by `a_P + ρV/Δτ`, read as the *initial* residual before the
        field update (the standard finite-volume convergence judge). The pseudo-time term is present in
        the residual, the matrix diagonal, **and** the scaling, all three consistently, and its reference
        `φ⁰` is **held fixed across the inner iterations of a timestep**. That residual is `O(‖δ‖²)` after
        a Newton step and collapses to the pure steady imbalance at each timestep's start — so it
        converges.
      - **aquaflux measures the bare *steady* residual `R`** while the shift `βD` is on the **Jacobian
        only** (`(J + βD)δ = −R`, `R` the unshifted steady residual — `continuation.py`), and the shift
        reference **resets to the previous iterate every step**. Both take the *same* Newton step on
        `G = R + βD(φ − φ_k)`; the reference measures `G(φ⁺) = O(‖δ‖²)`, aquaflux measures
        `R(φ⁺) = G(φ⁺) − βDδ = −βDδ` — the `β×travel`. The missing `−βDδ` is exactly the pseudo-time term
        the reference keeps in its residual and aquaflux drops.
      So the earlier conclusion ("valid convergence test near the root, not a merit function far from
      it") mislocated the fault: the measure is **sound**, and it is being fed the steady residual of a
      **single-step PTC with a per-step reference** instead of the backward-Euler *initial* residual of a
      **held-reference dual-time march**. This is the same gap the pseudo-time finding named — aquaflux
      approximates a transient with single PTC steps. The fix is structural, not a norm change: a true
      dual-time march (hold `φ⁰`; put the shift in the residual **and** Jacobian as
      `G = R + (ρV/Δτ)(φ − φ⁰)`, scaled by `a_P + ρV/Δτ`; inner-iterate `G → 0`; advance `φ⁰`; judge on
      the initial residual per outer step). Then the measure behaves exactly as in the reference — and it
      is the same change the pseudo-time finding calls for, so the two motivate one build.
    - **PROTOTYPE VALIDATED (2026-07-27; prototype not in the repository, superseded by the shipped
      `DualTimeStep`) — the diagnosis holds, and
      the fix is a per-timestep inner loop, not a norm change.**
      - **Confirmed current PTC = dual-time with K = 1.** At β = 2 the inner Newton converges
        `G = R + βd(φ − φⁿ)` in a **single** step (`‖G‖` 2.2e-2 → 6e-4, quadratic — it *is* the shifted
        Newton step), reproducing the single-step march exactly; the scaled measure then stalls
        (2.229e-2 → 2.204e-2) while euclidean halves — the β×travel signature.
      - **At β = 0.5 the inner loop engages (K = 2–3) and is stable**, converging `G` (2.2e-2 → ~3e-4)
        with a **line search on the scaled `‖G_n‖`** (first inner step clipped α = 0.5, then full). This is
        the legitimate inner merit (`G_n = 0` is a well-posed fixed-`φⁿ` solve), distinct from the refuted
        `G`-as-*outer*-merit.
      - **The measure is now honest.** The scaled `‖R(φⁿ)‖` holds ~2.1e-2 while `x_r/h ≈ 0` — it correctly
        reports that the slow bubble has not developed — while euclidean falls fast (2.86e2 → 4.6e1 over 3
        steps) on the quick pressure/momentum modes. That split is physical, not the β×travel artifact.
      - **CAVEAT — the dual-time STEP alone does not accelerate reachability; the Δτ RAMP is what does.
      This and the carried-`DualTimeControl` result below are ONE finding, not a conflict.** Development
      rate is Δτ-governed, so a dual-time step held at a fixed β is the same crawl. Its contribution is
      (a) an honest `‖R(φⁿ)‖` that can *drive* a Δτ ramp (single-step's stalling measure is why SER ran
      backwards) and (b) the inner-line-search-on-`G` tolerating a larger Δτ than one shifted step.
      Reachability needs that ramp **and** the cold-start diffusion/Re continuation (they compose:
      dual-time is the honest gauge + robust per-step solve, continuation lowers the cold stiffness so Δτ
      can grow early). Read the ramp's own result below as a measurement of the *ramp*, never as evidence
      that the step alone accelerates anything.
      - **CFL-ramp A/B (2026-07-27; prototype not in the repository) — the hypothesis holds, the
        gate is now the low-β linear-solve cost.** A `DualTimeStep` + `CflController` (grow Δτ / drop β
        when the inner loop meets η within ≤ 3 steps with α ≥ 0.5; back off otherwise), cold start:

        | inner solves | β | inner | x_r/h | scaled ‖R(φⁿ)‖ | euclid |
        |---|---|---|---|---|---|
        | 9 | 0.263 | 3 | 0.031 | 2.12e-2 | 44 |
        | 12 | 0.176 | 3 | 0.095 | 2.01e-2 | 29 |
        | 15 | 0.117 | 3 | 0.227 | **1.51e-2** | 18 |

        - **CONFIRMED: the inner loop unlocks β far below the single-step floor.** β ramped 2.0 → 0.117
          (still dropping) with every step converging (met, α = 1, ≤ 3 inner) — single-step blows up at
          β = 0.25 cold, dual-time is stable at less than half that.
        - **The measure fix is now visible in a march:** once the bubble formed (x_r/h 0.095 → 0.227) the
          scaled ‖R(φⁿ)‖ fell 25 % in one step, where the single-step scaled measure stalled at 1.9×
          forever. x_r/h accelerates as β drops (0.031 → 0.095 → 0.227, ~doubling per Δτ doubling).
        - **NOT YET more efficient per solve, and the reason is the low-β cost.** (i) The controller
          started at β = 2 (safe cold) and spent ~6 solves in the unproductive high-β regime before the
          bubble moved, so at 15 solves it trails aP05 (single-step β₀ = 0.5: x_r/h 0.58 @ 15). Fix: start
          the controller at β = 0.5 (proven stable cold). (ii) As β drops the shifted saddle loses diagonal
          dominance, so each solve costs more GMRES cycles *and* the inner loop needs 2–3 steps — stability
          is bought, not cheaply. **That low-β linear-solve cost is exactly what automated Re/ν_t
          continuation removes** (lower cold stiffness → cheap low-β solves), so dual-time (stability +
          honest gauge) and continuation (cheap big-Δτ steps) compose — the point to move to Re continuation.
      - **BUILT (opt-in): `DualTimeStep` (`solve/continuation.py`) + `DualTimeControl`
        (`solve/step_control.py`).** `DualTimeStep` is a `ForwardStep` whose `stepper()` holds a reference
        `φⁿ` and runs an inner Newton loop on `G = R + β d (φ − φⁿ)` to `‖G‖ ≤ inner_tol·‖R(φⁿ)‖` (or
        `inner_steps`), line-searched **monotonically on ‖G‖** (a well-posed fixed-`φⁿ` solve, unlike the
        non-monotone steady residual). The shift is in the residual *and* the Jacobian, so the measured
        steady residual is the honest discrete time derivative, not `β×travel`; `inner_steps = 1` is one
        shifted step (the pseudo-transient attempt, minus the escalation ladder the inner loop replaces).
        β still vanishes at the root, so the IFT adjoint is unchanged — pinned by
        `tests/unit/test_dual_time.py` (converges, exact gradient, **iteration-count-independent**).
        **Inner-loop observability — `DualTimeStep.inner_observer` (opt-in, shipped).** The outer
        `StepReport` only summarizes the inner loop (the inner *count* and the *summed* solve cycles),
        which conflates the two costs and hides the inner `‖G‖` trajectory. `inner_observer` is a
        `(inner_index, ‖G‖_before, ‖G‖_after, cycles, alpha) -> None` hook called **once per inner
        iteration** via `jax.debug.callback` — so it surfaces exactly how many inner iterations ran, each
        inner solve's cycle count, its `‖G‖` reduction and line-search factor. It is forward-only and
        transform-transparent (a no-op under `jax.grad`); `None` (default) elides the call at trace time,
        leaving the step **byte-identical** (do not set it on a differentiated solve). Threaded through
        `coupled_continuation` / `coupled_amg_continuation` / `coupled_ilut_continuation` /
        `coupled_lu_continuation`, so a profiling march can pass one straight through. Pinned by
        `test_dual_time_inner_observer_surfaces_the_trajectory_without_changing_the_step`.
        `DualTimeControl` is the Courant β-ramp (grow the pseudo-timestep while the inner α = 1, shrink
        when it clips), a `StepControl` on the eager march. The step's
        reported α is the **min** inner line-search factor, and an inner step that fails to reduce ‖G‖
        (the line search's non-descent fallback, which otherwise reports α = 1) is folded to **α = 0** so
        the control reads it as struggling and backs off rather than growing — the α-only `StepReport`
        signal cannot otherwise distinguish a clean full step from a non-descending fallback. Wired
        through `coupled_continuation(inner_steps=…, inner_tol=…)` (returns a `DualTimeStep` when
        `inner_steps > 1`, else the unchanged `PseudoTransientStep`) and reachable as
        `solve_coupled(coupled, inner_steps=…)`. **The default path (`inner_steps = 1`) is byte-unchanged.**
      - **`DualTimeControl` IS NOW THE DEFAULT for a dual-time observed march, and it CARRIES β across
      refreshes — this reaches a developed recirculation several-fold faster than the residual-keyed
      control (measured 2026-07-30, and it SUPERSEDES the "runs the transient away" verdict just
      below).** The reachability crawl to develop the pitzDaily bubble was a **step-control defect**,
      not a pseudo-time limit. Two defects, both fixed/retired here:
        - `DualTimeControl` used to **reset β to `beta_start` on the first step of every post-refresh
          segment** (`previous is None`); with a ~3-step drift refresh β *sawtoothed* `0.5→0.33→0.22→
          (refresh)→0.5→…` and Δτ never grew — so the α-ramp was byte-identical to the pinned SER control.
          `next_step` now **carries β** on a segment boundary (`state` present, `previous is None` → hold),
          exactly as `ResidualRatioDualTimeControl` does. Its carried state is a **bare β** (SER's is
          `(β, prev ‖R‖)`). Pinned by `test_dual_time_control_holds_beta_across_a_refresh`.
        - `solve_coupled` **auto-defaults** `step_control=DualTimeControl()` when the march is a
          `DualTimeStep`, is already observing (a refresh or observer is set), and no control was supplied
          (`_default_dual_time_control(step_control, observing, continuation)`, unit-tested in
          `test_coupled_rans.py`). It is injected **only where a control runs** and **never turns
          observation on**, so the differentiable single-stage solve (guarded `_is_traced`) is untouched;
          pass an explicit control to override. `solve_reynolds_continuation` inherits it (kwarg forward).
        Measured 2026-07-30 on a matched-seed pitzDaily rung-1 testbed and on a full cold Re ramp (hybrid
        IC → Re/100 → Re/10 → target Re 25000), carrying `DualTimeControl` against the SER control:
        **carrying β cut the outer-step count several-fold, and the ramp reached a developed `x_r/h` close
        to the OpenFOAM value.** *(the step counts and `x_r/h` recorded no continuation builder,
        preconditioner or forward solver — re-measure before quoting a number.)* The qualitative behaviour
        is what to rely on: it is self-regulating — α clips in the steepest development, recovers to 1.0,
        then β falls to the `beta_min` floor and the tail converges near-quadratically. `beta_min` is a
        speed↔smoothness knob (a smaller floor is faster but can overshoot the steady bubble on a cold
        rung with a loose seed + big Re jump, costing a couple of expensive recovery steps; the class
        default is the smoother choice).
      - **⚠️ THE "`DualTimeControl` RUNS THE TRANSIENT AWAY" VERDICT IS SUPERSEDED — do not cite it.**
        It held that the α-control grows Δτ blind to the steady residual and drives `x_r/h` past the
        steady state without settling, and was measured on the Re/100 anchor **before the β-carry fix and
        without the `beta_min` floor**. With β carried and the
        floor bounding Δτ, the ramp converges standalone (rung-1 to rtol 1e-6; full ramp to target Re) — the
        "runaway" the residual-keyed control was built to prevent does not block convergence here, and its
        residual-feedback instead *pins* β on the flat `β×travel` plateau (the slower arm). Do not cite the
        old verdict as a reason to prefer the residual-keyed control.
      - **`ResidualRatioDualTimeControl` (`solve/step_control.py`) is now the OPT-IN alternative — switched
        evolution relaxation / Kelley–Keyes pseudo-transient continuation.** It ramps Δτ by the steady-
        residual reduction ratio: `β ← β · (‖Rₙ‖/‖Rₙ₋₁‖)` (residual drop → β down / Δτ up; residual rise →
        β up / Δτ down), clipped to `[1/max_change, max_change]`, clamped to `[beta_min, beta_max]`, with a
        hard inner-clip (`α < backoff_below`) safety shrink, and carrying β across a refresh. A rising
        residual *automatically* shrinks Δτ, so it cannot run away — but on the pitzDaily ramp the row-scaled
        steady residual is nearly flat while the flow develops (`β×travel`), so it **pins β near `beta_start`
        and stalls Δτ**, taking several-fold more outer steps than the α-based default. Prefer it only where the steady
        residual is a reliable monotone progress signal. Its `next_step` state is `(β, prev ‖R‖)`. Unit-tested
        in `tests/unit/test_step_control.py`.
      - **THE LOW-β WALL IS THE BLOCK-SIMPLE PRECONDITIONER, AND THE ILUT BREAKS IT.** With
      `ResidualRatioDualTimeControl` the residual descends cleanly (no runaway) but block-SIMPLE's coupled
      solve goes **NaN at a low shift** — the low-shift conditioning wall (block-SIMPLE cannot solve the
      near-unshifted saddle; the same limit as its adjoint stagnation). The monolithic ILUT forms the true
      coupled inverse, so `coupled_ilut_continuation(inner_steps>1)` (a `DualTimeStep` preconditioned by
      the ILUT — the branch added alongside the single-step one) drives β **monotonically below that wall
      with no NaN, at a flat cycle count**, descending the row-scaled residual on the anchor. So the ILUT
      is what makes the large-Δτ dual-time march reachable at all. *(the β at which block-SIMPLE NaN'd,
      the β the ILUT reached, the cycle count and the residuals recorded no Re rung, state or refresh
      setting — and the cycle count was taken on the ILUT path's restart-10 forward solver, so it is not
      comparable with the restart-15 or restart-120 counts elsewhere in this file. Re-measure before
      relying on any of them.)*
      - **Residual FLOOR + over-development past the minimum = loose `inner_tol`, NOT the preconditioner.**
      Even with the ILUT (a flat cycle count — the linear solve is fine), the march bottoms out at a
      residual floor and then slowly over-develops. Cause: dual-time's unconditional stability comes from
      the inner loop driving `G = R + βd(φ−φⁿ)` to zero each step; at `inner_tol = 0.05` the implicit
      step is only 5%-solved, so a large-Δτ backward-Euler step on a half-solved system overshoots. Fix =
      tighten `inner_tol` (with enough `inner_steps` to reach it) — **affordable precisely because the
      ILUT makes the low-β inner solves cheap**, where block-SIMPLE could not. ILUT removes the
      conditioning wall; tight `inner_tol` restores dual-time stability; the two together are what settle
      the rung. *(the floor and the `x_r/h` it corresponded to shared the unrecorded configuration of the
      bullet above — the mechanism stands, the numbers are deleted.)*
      ⚠️ **"Tighten" has since been bounded on the OTHER case, and it does not mean 1e-3.** On `bfs3d`
      a three-point sweep measured `inner_tol` 1e-2 as a **33 % shorter march than 1e-3 at an identical
      step count**, with 0.05 the first value to cost an outer step — see the dual-time inner-tolerance
      entry above. So this bullet's direction is right and its magnitude is case- and Δτ-specific: do not
      read it as an argument for 1e-3 anywhere else.
      - **⚠️ READING SMALL CYCLE COUNTS (binding — two offsets fooled a whole investigation).** Two things
        inflate the reported linear-solve cost at the low end, so a "6" is NOT six times a "1":
        (1) **lineax's `num_steps` has a +2 offset and is blind within a restart cycle.** Calibrated: a
        system GMRES solves in 1, few, or ~100 matvecs (all inside one 120-restart cycle) ALL report
        `num_steps = 3` (a dummy r0=0 first pass + deferred breakdown); it only climbs when the solve
        genuinely spills past a restart cycle. So **`num_steps = 3` means "converged in one cycle" = ideal**,
        and `solve_linear`'s count cannot distinguish 1 matvec from ~100. (2) **`DualTimeStep` reports the
        SUM of `num_steps` over its inner Newton iterations** (`stepper` docstring). So a dual-time
        `cyc = 6` is **~2 inner Newton iterations × an ideal 1-cycle solve**, and `cyc = 9` is ~3 —
        the inner-iteration count to reach `inner_tol`, NOT a per-solve penalty. **Consequence measured
        this session:** the coupled ILUT is a NEAR-DIRECT preconditioner — 1 restart cycle (~4 matvecs) per
        solve at every pitzDaily state, flow-only and full `[u,v,p,k,ω]` alike, fresh or mildly stale
        (record not in the repository). The march's "6–9" is the dual-time inner-loop sum, and
        **β-matching the frozen factorization to the march's β is a no-op on it** (fixed-`ilut_beta` and
        `ilut_beta`-matched runs gave IDENTICAL `cyc`). The only lever on the "6" is `inner_steps`/`inner_tol`
        (globalization/accuracy), which is deliberately kept tight for stability — not the preconditioner.
        The "coupled ≈ 6 vs flow-only ≈ 2 → k/ω degrades the ILUT → build ILUT+AMG" premise is REFUTED; the
        only live reason for ILUT+AMG is 3D `spilu` fill scalability, which cannot be judged on 2D pitzDaily.
    - **Lowering β is not the escape, and the reason is specific — state it precisely.** At `β = 0.25`
      the k/ω blocks reach 1e24 / 1e52 at the cold IC and go NaN at step 20, but are **perfectly stable
      at steps 45 and 90** (ratios 0.994–1.046). So the under-damping is an *early-state* property, not
      a general one: β can be lowered once the flow is developed, and cannot be lowered at exactly the
      cold start where the reachability problem lives. This independently re-kills `descent_backoff`,
      whose whole premise is lowering β from a cold state.
    - **Still open:** the *across-iteration* weight drift (`a_P` and the field magnitudes both grow as
      the flow develops, so the denominators move between iterations) is a **separate** effect from the
      β floor and remains unmeasured. Settle it by replaying one `RowScaledNorm` with scales frozen at
      the warm-started root over the stored `profile_base/g*.npz` history — seconds of compute.
  - **⚠️ THE MEASURE'S WEIGHTS ARE STATE-DEPENDENT, so there is no single objective across iterations.**
  `f(x) = Σ wᵢ(x)|Rᵢ(x)|` with `w` from the operator diagonals and field magnitudes. **This governs the
  OUTER-ITERATION boundary only:** when a `norm_builder` is supplied, `forward_march` rebuilds the
  measure at the state each outer iteration begins from and freezes it for that whole iteration (so the
  line search compares like with like) — so a direction that descends in *this* iteration's frozen `f`
  need not reduce the *next* iteration's `f`. Do not assume the frozen-per-iteration measure behaves
  like a fixed merit function. This is **not** in conflict with "the measure must be held FIXED across a
  refresh" below: that rule governs the *segment/refresh* boundary — the `base_norm` `solve_coupled`
  builds once and re-injects into every refreshed continuation, which is what the convergence test and
  the finishing solve are judged in.
  - **⚠️ `descent_backoff` IS COUNTERPRODUCTIVE ON THIS CASE — measured, do not enable it blindly.**
    Backing β off until the correction descends does produce a descending direction, but the finite-step
    profile along it is *worse*: at β = 0.5 the full step raises the measure 2.59× and is not admissible,
    forcing α ≤ 0.5. On a march the arm's α fell 1.0 → 0.5 → 0.5 → 0.031 while the measure *rose* every
    step. **Descent is necessary but not sufficient** — strong positive curvature along δ swamps the
    negative slope. Note ‖δ‖ *decreases* as β is backed off (1049 → 856 → 760 for β = 2 → 1 → 0.5), so
    "α collapsing" is not a large-correction artefact.
  - **⚠️ EXTENDING THE LADDER ABOVE α = 1 (`grow`): inert on the Euclidean measure, live on the
    equilibrated one — and it exposed a fallback bug (2026-07-27).** (The equilibrated/row-scaled measure
    is now the *default*, so `grow` is live on the shipped configuration; the Euclidean result below is the
    now-non-default measure.)
    - On the **Euclidean** march, `grow = 2` produced a trajectory **bit-identical** to the
      control across 10 steps and both checkpoints: α = 2 is never admissible there, so the extended
      ladder is inert on that measure.
    - On the **equilibrated** measure it fires: α = 2 was selected at step 1 and was productive. A
      cold-start scan confirms α = 2 sits inside the tolerance (ratio 1.291 against a 2× bound) and
      travels twice as far as the full step.
    - **The bug it exposed:** extending the ladder upward also extended the *fallback* upward, so a step
      with no admissible length fell back onto **α = 4** and multiplied the measure by **4.6** in one
      step. The fallback is now capped at the full step — **a growth rung must only ever be reachable by
      passing the acceptance test, never by falling back onto it.** Pinned by a unit test.
  - **⚠️ THE SHIFTED CORRECTION IS NOT A DESCENT DIRECTION, AND THE CAUSE IS THE UNSHIFTED CONSTRAINT
    ROW (measured 2026-07-27). This is the mechanism behind the α-at-the-smallest-rung stalls recorded
    throughout this file.** For the *exact* Newton direction (`J δ = −R`) the derivative of any
    positively-weighted residual measure along `δ` is `−‖R‖ < 0` — descent, for free. The **shifted**
    direction satisfies `J δ = −R − β D δ`, whose second term has no fixed sign, and its damage grows
    with β. Measured directly (`∇f·δ` by forward-mode AD through the measure) on a stiff coupled state:

    | β | 0.05 | 0.2 | 0.5 | 1.0 | **2.0** |
    |---|---|---|---|---|---|
    | `∇f·δ` | −7.7e-3 | −1.7e-3 | −4.9e-4 | −1.3e-4 | **+3.9e-5** |
    | ladder minimum α | 1.0 | 1.0 | 0.25 | 0.25 | **0.00098** |

    **The sign changes between β = 1 and β = 2, and the march was running at β ≈ 1.9.** At β = 2 the
    best rung on the whole ladder is the shortest one, which reproduces the observed stall exactly. Note
    the lower bound too: at β ≤ 0.2 the trial states go non-finite, so the usable window at that state
    was roughly **0.5 ≤ β ≤ 1**.
    - **What causes it is the MIXTURE of shifted and unshifted rows — not the weighting, and not the
      off-diagonal coupling.** Both of those were proposed and refuted on toy systems: a scalar residual
      gives `∇f·δ = −|R|·J/(J + βD) < 0` for *any* β, and a symmetric system with strong off-diagonal
      coupling, or with strongly skewed row weights, still descends at every β tested. What reproduces it
      is a **saddle system whose constraint row carries no shift** — the exact shape of the flow policy,
      where momentum rows get the operator diagonal and continuity gets zero:

      | β | 0 | 0.5 | **2** | 10 |
      |---|---|---|---|---|
      | `∇f·δ` | −2.30 | −1.15 | **+2.30** | **+20.70** |

      ⚠️ **CORRECTION (2026-07-27, same day): "shift every row uniformly and the derivative stays
      negative at any β" — as first written here — is WRONG.** That was only ever tested on a
      *symmetric* system, never on the saddle. Damping the constraint row on the saddle above gives:

      | `d_p` | 0 | 0.1 | 1.0 | 5.0 |
      |---|---|---|---|---|
      | `∇f·δ` at β = 2 | +2.300 | +1.438 | +0.329 | +0.074 |
      | `∇f·δ` at β = 50 | **+112.7** | +0.440 | +0.044 | — |
      | **crossover β** | **0.987** | **0.987** | **0.987** | **0.987** |

      So constraint damping **does not move the descent threshold at all** — it is 0.987 for every
      `d_p` tested, including zero. What it changes is the *magnitude* past the threshold: with an
      unshifted constraint row the failure **grows without bound in β** (+2.3 → +113), with a shifted
      one it **decays toward zero** (+0.33 → +0.044). And on this toy no rung of the ladder reduces the
      measure at β = 2 for **any** `d_p` — the profile is monotone in α throughout.
      **Consequence: damping the constraint row is not a fix for non-descent, and should not be sold as
      one.** What remains true and useful is that the unshifted row makes the failure unbounded rather
      than bounded, and that the threshold itself (β ≈ 1 here, and between 1 and 2 on the real coupled
      case) is set by the momentum shift against the Jacobian scale, not by the constraint row. Whether
      bounding the damage buys anything on the real nonlinear system is unmeasured — the toy is a 2×2
      linear system and cannot answer it.
    - **Escalation moves β the WRONG WAY for this failure (binding).** A rejected step escalates
      `β *= escalation_factor`, which is right for an overshoot or an ill-conditioned shifted system.
      Against a non-descent direction it is worse than useless: more shift makes `∇f·δ` *less* negative,
      so the loop spends a solve per attempt making the direction worse. `PseudoTransientStep` therefore
      carries **`descent_backoff`** (lower β until the direction descends, then escalate from there) and
      **`descent_test`** (reject a non-descent direction outright rather than judging the candidate's
      norm). Both default off. `∇f·δ` itself is cheap: one `jvp` on a direction already computed.
    - **A backoff probe is a COMPLETE attempt and is carried into the escalation loop — do not go back
      to discarding it.** The probe already computes the correction, the line search, the measure and
      `∇f·δ` at exactly the β the escalation loop then starts from, so re-solving there made every step
      pay **two** shifted solves on the path where nothing is backed off — the common one. The five
      values travel as one `_Attempt` record, and the loop folds its final probe into the escalation
      carry (`record(fresh(β), trial, probed & admits(trial, 0))`, selected by the loop's own descent
      flag). The seeding is used **only** when the carried attempt really was taken at the starting β:
      if the backoff instead exhausts its tries it exits at a *lower, unprobed* β and the escalation
      loop starts cold there, which is the pre-existing ladder. A backoff that has to lower β still
      costs one solve per rung; what is now free is the case where the first probe already descends.
  - **⚠️ THE LINE SEARCH TAKES THE LONGEST ADMISSIBLE STEP, NOT THE BEST ONE — a minimizing search was
    built, measured, and REVERTED (2026-07-27). Do not re-propose it.** Replacing "first rung that is
    admissible, walking longest-first" with "the rung that minimizes the measure" lowers the residual per
    step and is far worse on the physics: on the same cold-start case, judged at identical checkpoints,

    | checkpoint | minimizing | first-acceptable-largest |
    |---|---|---|
    | 2 | 0.01 | **0.09** |
    | 3 | 0.03 | **0.16** |
    | 4 | 0.05 | **0.34** |
    | 5 | 0.05 | **0.46** |

    **9× less recirculation development, while reporting BETTER residuals at every early step** (0.377
    vs 0.430, 0.254 vs 0.293, 0.193 vs 0.212). The α sequences show the mechanism: the minimizing search
    systematically picks 4–8× shorter steps. **Residual depth per step and distance travelled per step
    are different objectives, and on a march that has to transport a front across the domain, distance
    is the one that matters.** This is the fourth time on this case that a residual improvement has
    pointed the opposite way from the physics — judge a march on `x_r/h`, never on ‖R‖.
    - **The fallback when NOTHING is admissible is the longest FINITE rung, not the shortest (binding).**
      Returning the shortest is a near-null step that changes nothing, which the divergence guard then
      accepts as finite: the march reports a step and stands still. That is a *guaranteed* stall rather
      than a slow one, and it is what produced the `α = 0.001` signature (`0.001 = 1/2**10`, the smallest
      rung of the shipped 10-rung ladder — a value that means "nothing passed", not a sentinel).
    - **The ladder can extend ABOVE α = 1 (`grow` rungs of doubling; default 0 = off).** Measured on a
      developed state: the full step moved the reattachment not at all, while `α ≈ 5.7` moved it four
      times further **and already sat inside the tolerance the acceptance rule allowed** — it was simply
      unreachable from a ladder that starts at 1. Any scan or study of step length must therefore not
      hard-bound its grid at 1.0, which an earlier one did, making "α = 1 is optimal" unfalsifiable.
  - **`line_search` — backtrack the shifted step before escalating β (binding, the coupled-RANS fix).**
    The step optionally scales the shifted correction `δ` back along `{1, 1/2, …, 1/2**line_search}`
    (`backtracking_line_search`, extracted from `implicit.py` and shared with `DampedNewtonStep` — one
    home for the ladder) and keeps the largest length that reduces the residual, **before** the
    accept/escalate test. The ladder is a **`lax.while_loop` that stops at the first (largest) reducing
    rung** — a full step that already descends (the common case near the root) costs one residual
    evaluation, not `line_search+1`, and the loop body compiles once instead of unrolling `line_search+1`
    residual copies into the graph. It is safe as a non-differentiable `while_loop` because the search is
    **forward-only**: it runs inside `ImplicitNewtonSolver`'s `custom_vjp` forward pass, whose reverse
    rule is the IFT transpose solve at the root and never differentiates the iteration (every caller is a
    `ForwardStep`; nothing differentiates through it — audited). Do **not** call it on a differentiated
    path. `line_search=0` (default) is the old behaviour: take the full step `φ+δ`, and
    the **only** recourse to an overshoot is escalating β — a *full re-solve*. This was measured to be
    the dominant coupled-RANS cost: from the hybrid IC the full coupled Newton step overshoots by
    ~10⁷× (‖R‖ 220 → 5.8e9), so every step burned ~4–7 expensive re-solves and, worse, escalating β
    (β=16/64) still did **not** descend (rel ≈ 1.0 — the march stalled). A line search on the **one**
    β₀ solve finds α≈¼ → rel≈0.48 (residual halved) in a few cheap residual evaluations. So the
    coupled path sets `line_search>0` (`coupled_continuation`, `_COUPLED_LINE_SEARCH=10`); β escalation
    stays the fallback for a genuinely bad *direction* (an ill-conditioned shifted solve), not an
    overshoot. Like the shift, the search only reshapes the forward path — converged state and IFT
    adjoint unchanged. The flow path leaves `line_search=0`, so it is bit-identical.
  - **`forward_solver` overrides the shared `_INEXACT_CONTINUATION_SOLVER`; the coupled default now stops
    on a GLOBAL 2-norm relative residual (`relative_residual_gmres`, `solve/linear.py`).** `default_solver()`
    returns the injected `forward_solver` when set, else the shared restart-40 GMRES. The coupled path
    injects restart 120 (the stiff saddle needs hundreds of restart-40 cycles; a 40-vector subspace
    discards too much Arnoldi history). **The dominant waste was the TERMINATION, not the restart.** The
    old `GMRES(rtol=1e-3, atol=1e-10)` reached true_rel ~4e-12 in ~15 restart cycles / 1800 matvecs on the
    cold-IC pitzDaily solve although only rtol=1e-3 was asked — because `lineax`'s stock stop is
    **componentwise** (`|r_i| ≤ atol + rtol|b_i|` under max_norm), and the ~470 near-wall ω wall-fixation
    rows start satisfied (right-hand side ~0), so their per-row scale collapses onto the absolute `atol`
    floor and a handful of them hold the whole solve to ~1e-10 (~9 orders past 1e-3). `relative_residual_gmres`
    scales the system to unit right-hand-side 2-norm and runs GMRES at `rtol=0, atol=target, norm=2-norm`,
    so it stops on `norm(r)/norm(b) ≤ target` — immune to those rows. ⚠️ **That residual is the TRUE
    one, not `‖Mr‖`: `_shifted_solve` takes `solve_linear`'s `preconditioner_side="right"` default, so the
    Krylov residual is `b − A M y = b − A x`. No "solution accuracy" follows from it — an earlier version of
    this line inferred "≈1% solution accuracy since M≈A⁻¹", which holds only under LEFT preconditioning and
    misled a later reader into a wrong hypothesis.** Measured on the real cold-IC march: ~3-5 cycles (often 2-3/step), ~4× fewer
    matvecs to the same `x_r/h`, trajectory unchanged.
  - **⚠️ THE "TIGHT TOLERANCE IS LOAD-BEARING UNDER LOG-ω" CLAIM WAS STALE — corrected 2026-07-28.** The
    old note here (and in `turbulence.md`) said an inexact/loose forward solve is unsafe under log-ω
    ("an inaccurate log step is exponentiated and diverges; loosening breaks the march"). Two-arm cold-IC
    pitzDaily marches — the over-solving default vs a 4-cycle-capped ~1e-3 solve, then the adaptive
    `relative_residual_gmres` — **refute it**: the honest ~1e-3 solve reproduces the over-solve march to
    3-4 significant figures per step (**including the line-search α**), tracks the same `x_r/h`, and never
    diverges, at ~3-4× fewer matvecs. The original evidence was almost certainly an artifact — "loosening
    rtol" never loosened the solve, because `atol=1e-10` bound regardless of rtol (the componentwise floor
    above), so it kept over-solving to ~1e-12. Do **not** reinstate a tight fixed tolerance on the forward
    solve. Same-root-safe by construction: inexact Newton, the shift vanishes at the root, the nonlinear
    stop is on ‖R‖, and the adjoint is a separate transpose solve — none touched by the forward-solve
    looseness (pinned by the `slow` `test_coupled_rans` convergence + adjoint gates).
  - **⚠️ READ FIRST: every raw-‖R‖ comparison ACROSS coupled-RANS march states recorded below predates
    the 2026-07-25 fixation-row fix and is suspect.** Until then the near-wall ω fixation was written
    in physical ω under a log-ω unknown, so 472 of 12 225 rows — scaled by an ω spanning 160→1.1e5 —
    dominated the norm that drives the line search, the SER ramp, the divergence guard and the stopping
    test (see `.claude/rules/turbulence.md`). Measured consequences: ‖R‖ at the *converged* pimpleFoam
    field was **1.533e5**, i.e. the metric rated the right answer ~1.7e4× worse than a half-developed
    state and 800× worse than the cold IC; and it ranked the const-β state at "rel 0.032" above the SER
    state at "rel 0.052" although the former's recirculation bubble is **4× worse** (`x_r/h` 0.29 vs
    1.16, target 7.74). After the fix the same field reads **20.7** and the ranking matches the physics.
    **So: "reached a deeper rel" was not evidence of a better state, and the preference for the const-β
    march and the α-targeting controller rests on exactly that comparison.** Re-measure before trusting
    any relative-residual claim in this section; the *mechanistic* findings (exact linear solves, the
    α-sentinel, the modal attenuation) are unaffected because they were measured at a single state.
  - **⚠️ SCOPE FIRST: the "SER runs backwards" finding below applies ONLY to a march that does NOT
  refresh (2026-07-25).** SER's `residual_norm_0` is **segment-local** — recomputed at each
  `forward_march` entry, hence reset at every preconditioner refresh. With a refresh every handful of
  steps the ratio `‖R‖/‖R₀‖` never falls far below one, so **β is pinned near β₀ for the entire march**
  rather than decaying (measured on a drift-refreshed cold-IC pitzDaily march; the per-step β/α table
  that stood here is deleted — it named no preconditioner or forward solver and was taken under the
  superseded ω-dominated norm).

  Three consequences. (i) Enabling the refresh silently converts SER into **constant β ≈ β₀** — if a
  different damping is wanted it must come from `β₀` or a different schedule, not from expecting SER to
  ramp. **`β₀ = 2` is the settled value.** The earlier reading here — "β₀ = 2 is too much, `β₀ = 0.5`
  develops a longer bubble" — was measured *before* the flux-continuous / wall-model `a_P` fixes that
  changed the shift itself, and the post-fix re-profile reverses it (`β₀ = 0.5` is under-damped and
  stalls); see "RE-PROFILED AFTER THE `a_P` FIX" above, which is the surviving side of that conflict.
  (ii) **An α-targeting controller may have nothing to push against here**: α was reported saturated at
  its set-point on this march while the residual barely moved, i.e. the productivity ceiling is not
  obviously an α problem, which re-scopes #22 — same unrecorded configuration, so treat it as a
  hypothesis, and note the shipped dual-time default *is* α-driven (see the conflict recorded above).
  (iii) Any probe that derives "the march's operating β" from the global ratio is wrong by a large
  factor at a developed state.
  - **THE SER β SCHEDULE RUNS BACKWARDS FOR STIFF COUPLED RANS (pitzDaily — the claim is that the
  dominant cost is the globalization, not the preconditioner).** The switched-evolution-relaxation
  schedule `β = β₀(‖R‖/‖R₀‖)^p` *lowers* β as the residual falls, on the premise that a smaller shift
  means a more Newton-like, more productive step near the root. **On this problem the premise was
  measured false: the efficiency-optimal β *rises* as ‖R‖ falls, so SER drives β the wrong way and the
  coupled march grinds instead of entering the quadratic basin.** ⚠️ **The supporting numbers — the
  efficiency optima, the step-efficiency gap and the α-vs-β table — are DELETED. All of them predate the
  2026-07-25 fixation-row fix and were taken under the ω-dominated norm that mis-ranked states, with the
  preconditioner rebuilt at each probed state (which the march never is). Re-measure before relying on
  any of this quantitatively.** What survives is the mechanism and one design consequence:
  - **The mechanism is line-search CLIPPING, seen directly via the step-length factor α.** α (the
  fraction of the shifted step the backtracking search keeps) rises with β and reaches **α = 1 at the
  efficiency-optimal β** — the point where the full damped step *just stops overshooting*. Below it
  the step overshoots and is clipped to near-nothing; at it the step is full and productive. So the
  grind is over-damped clipping, not near-convergence and not preconditioner cost.
  - **α is the usable controller signal; the per-step residual reduction ρ is not** — ρ swung
  several-fold at fixed β and wrecked a first, ρ-driven controller that ratcheted β into a runaway.
    - **Caveat — β-schedule and PC-refresh are COUPLED; the deleted optimal-β measurements above all used
      a PC rebuilt at each probed state.** In a real march the preconditioner is frozen at the cold IC, and
      a bolder β moves the state faster, staling that frozen PC faster (the ρ-driven controller's runaway
      got *more* expensive as β rose, where a bolder shift should be *cheaper* — so that was PC staleness,
      not the shift). So an α-targeting β schedule and the scalar-AMG refresh (below) must be co-designed,
      not tuned in isolation. A **β-independent staleness
      indicator** — the drift of the frozen operator's coefficients, `‖Δν_t‖`/`‖Δṁ‖` relative to the
      freeze state — is the clean refresh trigger this motivates (it fixes the `CycleGrowthTrigger`
      confound, #19: cycle count rises from β→0 *and* staleness, drift rises only from staleness).
    - **A/B'd end-to-end against SER (α-targeting controller + PC refresh) — the numbers are DELETED.**
    The whole comparison was a race between two ‖R‖ trajectories measured under the superseded
    ω-dominated norm, which the scoping entry above shows mis-ranked states; "reached a deeper rel" is
    exactly the claim the fixation-row fix invalidated. A prototype controller — raise β toward the α=1
    boundary (`β ← β/α`, capped), ease gently when α=1 — with the k/ω AMGs refreshed periodically and the
    step `filter_jit`'d (to match SER's compiled `while_loop` footing) was reported faster than SER at
    every overlapping residual from the cold hybrid IC. *(re-measure on `x_r/h` before citing it.)* Two
    structural findings from the same arms are worth keeping, because they say *which* configuration wins
    rather than by how much: (a) the **frozen-PC** α-controller *lost* — cycles rose with β, the β↔PC-
    refresh coupling biting, so the refresh is load-bearing; (b) the **eager** (un-jitted) version was
    handicapped per cycle, so the jit is needed for a fair comparison, not for the physics.
    - **The controller has a CEILING — it stalls short of a root, deeper than SER but not converged.** The
    cause is its own **over-damped hunting**: the `β/α` raise overshoots *past* the α=1 boundary to where
    the full step is tiny, then eases slowly; α saturates at 1 above the boundary, so the controller is
    blind there and cannot sit at the productive edge. So the direction is right, but a dynamics rework
    is needed: approach α=1 *from below* without overshooting, or pair α with a step-productivity signal.
    *(the residual levels quoted for both arms shared the superseded ω-dominated norm — the stall is the
    finding, its depth is not.)*
    - **PRODUCTIONIZED as an injected strategy pair (the schedule half is shipped).** The β schedule is
      the injected `RelaxationSchedule` (SER = `SwitchedEvolutionRelaxation`, the default; see the
      `continuation.py` bullet). **The α-targeting control itself is DELETED (2026-08-14) — there is no
      `AlphaTargetingControl`.** It never converged standalone, its gains were hand-set placeholders, it
      had no production caller, and it was the one control that both lacked `carry_beta` and reset β at a
      refresh boundary — so a shared `ShiftStrengthControl` base would have had to carry a seam for a
      member nothing selected. The α signal survives where it is measured to work: inside
      `DualTimeControl` and `CflResidualDualTimeControl`, which drive the *dual-time* pseudo-timestep.
      The single-step α-targeting *direction* is unrefuted and unbuilt; rebuild it as a
      `ShiftStrengthControl` subclass if it is ever wanted. Study harnesses in the scratchpad
      (`beta_sweep.py`, `alpha_probe.py`, `alpha_controller_march.py` = frozen-PC, `alpha_refresh_march.py`
      = the winning arm) remain as the calibration/replay tools.
    - **A PER-BLOCK β (separate shift damping for flow / k / ω) is DOMINATED — measured, do not re-attempt
      (`per_block_sweep.py`).** The Euclidean ‖R‖ on the coupled state is ~100 % ω (ω O(1e1) vs flow O(1e-2),
      k O(1e-3)), so a natural idea is to damp each block by its own β — the block-diagonal shift already
      supports it (unpack the shift diagonal `[a_P·u, 0·p, d_k·k, d_ω·ω]`, scale each slice, repack; the flow
      preconditioner keys off `β_flow` via its `a_P(1+β)`, the scalar AMGs are β-independent). Swept at the
      developed state (rel 0.05), holding `β_ω` high and lowering `β_k`/`β_flow`, it loses on every axis
      against uniform β. *(the per-block sweep table is deleted: it recorded no preconditioner or forward
      solver, and its judging quantity was the ω-dominated Euclidean norm that mis-ranked states —
      re-measure before treating the ruling as settled.)*

      Two failure modes were read off it, **neither a damping problem**: (i) **k is acceptance-limited** —
      a smaller `β_k` *does* let k descend, but the bigger k-step makes the *coupled* full step overshoot
      the ω-dominated norm, so the line search clips α and ω progress collapses; crediting k would need a
      block-aware *acceptance* norm, which is the dead `BlockScaledNorm` (below). (ii) **flow is
      coupling-limited** — no `β_flow` un-sticks it, because flow is waiting on ω through the two-way ν_t
      coupling. The blocks are coupled through **both** the direction (flow↔ω) and the acceptance (ω-norm),
      so per-block *damping* cannot separate them. This re-confirms the old "Lever D" per-block
      under-relaxation ruling, now with the mechanism visible under log-ω + the adaptive wall.
    - **The lever is a HIGHER uniform β, not a per-block one — but the numbers behind "β=5 ≫ β=3" are
    DELETED (same sweep, same ω-dominated norm, preconditioner rebuilt per state).** The reading was that
    at the developed state the efficiency-optimal β sits above the value SER reaches; that a higher β is
    not cheaper *per cycle* but wins on **step count and overhead** (fewer Newton steps → fewer PC
    refreshes, recompiles, line searches) while staying productive (α = 1); and that a constant-β march
    descended past SER's floor and then ground in the tail — the too-low-β symptom. So the direction taken
    was the β-climbing controller (#22: climb β while α = 1), **not** a per-block β, a norm change, or
    physical/order continuation. ⚠️ Re-measure before quoting any β from this: the post-`a_P`-fix
    re-profile moved the whole β calibration, and it is judged on `x_r/h`, not on ‖R‖.
  - **Where the coupled-solve cost actually is.** As the SER ramp drives `β → 0` through the march, the
    *unshifted* coupled saddle Jacobian is severely ill-conditioned, so the diagonally-shifted GMRES burns
    many matvecs per solve and the cost rises sharply as β falls. *(the per-solve wall times once quoted
    here named no preconditioner, forward solver, restart or state, and predate the fixation-row fix —
    deleted; the β-dependence is the mechanism, the seconds are not evidence.)* Note lineax `num_steps`
    counts restart **cycles**, not iterations, and carries a fixed offset — see the reading rule above.
    **The `β → 0` here is SER-induced and correctable, not inevitable — see the schedule-runs-backwards
    finding above.** Several levers were probed: two are wired but **off by
    default** (kept for further evaluation, not the fix), one is dead, and one — refreshing the **scalar**
    k/ω AMGs after the flow separates — is a real ~2.6× win, now BUILT (see below):
    - **Flooring the SER `β` below (`β = max(beta_floor, β₀(‖R‖/‖R₀‖)^p)`) — correctness-safe, reported a
    WASH, kept off-by-default.** The field is **`SwitchedEvolutionRelaxation.beta_floor`** (it lives on
    the schedule, not on `PseudoTransientStep`, whose `beta0`/`exponent`/`beta_floor` fields were
    removed); default 0 = off. It never moves the converged root (the shift `β d` scales the correction
    `δ`, which vanishes at `R=0`; it only damps the *path*, linear instead of quadratic terminal steps)
    and it does make each late solve cheaper, but end-to-end the cheaper late solves were reported to
    cancel the extra Newton steps. *(configuration not recorded — case, state, preconditioner and norm
    all unnamed — so treat "wash" as the reason it is off by default, not as a measured fact.)* Wired
    through `coupled_continuation(beta_floor=…)` for further evaluation.
    - **The default coupled residual measure is the row-equilibrated `RowScaledNorm`
      (`coupled_scaled_norm`), NOT the Euclidean ‖R‖.** The Euclidean coupled residual is `ω`-dominated
      and *mis-ranks* states (a converged field scores worse than a badly wrong one — the warning above);
      `RowScaledNorm` divides each row by its own diagonal and each block by its field magnitude, so every
      equation is judged comparably. **`per_block` is that measure's reporting view and `__call__` is
      literally `norm(per_block(r))`** — one formula, so the per-equation grid in the march log cannot
      describe a convergence history the march never had. `coupled_continuation` / `coupled_ilut_continuation` build it by
      default; `block_scaled_norm=True` selects the coarser one-scale-per-block `BlockScaledNorm`
      (`_coupled_residual_norm`), and `residual_norm=jnp.linalg.norm` recovers Euclidean.
      (`mass_flow_coupled_continuation` still defaults to Euclidean pending a constraint-aware variant.)
      The row-scaled measure does **not** fix the forward stall (globalization-bound; it plateaus under any
      measure — that plateau is the *honest* signal, where the Euclidean fall was a `β×travel`/`ω`-magnitude
      artifact); it makes the measure honest and is required to judge this case correctly.
      **The measure must be held FIXED across a refresh (binding, #156 seam 4) — this governs the
      SEGMENT/refresh boundary, and does NOT conflict with the per-outer-iteration rebuild described
      above, which governs what a single iteration's line search and acceptance test compare in.**
      `BlockScaledNorm` is
      self-normalising — at the state its per-block scales were built at it returns `sqrt(n_blocks)` — so
      rebuilding it at each refresh's developed state re-bases every `residual_ratio` back toward one,
      making the convergence test unreachable and mismatching the finishing solve's absolute
      `stage_atol` (computed on the pre-refresh scale). `solve_coupled` therefore captures the initial
      measure once (`base_norm`, the same state the global `reference_norm` is measured at) and passes it
      to every refreshed `coupled_continuation(residual_norm=base_norm)`, which uses it verbatim instead
      of rebuilding. The invariant is "the global progress reference and the norm come from the same
      state." Latent before the fix (only bites with `block_scaled_norm=True` *and* a refresh); pinned by
      a unit test that a refreshed continuation reuses the initial norm object.
    - **A block-*triangular* preconditioner (forward-substituting `∂R_turb/∂flow·δ_flow`) — tried, WORSE,
      dead.** It made the channel worse (measured, configuration not recorded — no mesh, smoother or
      aggregation) and on recirculating pitzDaily was
      so bad GMRES could not converge at all: stronger flow↔turbulence coupling *amplifies* the inexact
      diagonal blocks' inversion error it propagates downstream. So the missing cross-coupling is **not**
      the bottleneck.
    - **The real cost is the pressure-Schur *approximation* at high Reynolds number — and strengthening
      the inner solve CANNOT fix it (measured; do not re-attempt).** The block-diagonal conv+MSIMPLER
      preconditioner is *excellent* at low Re (4 outer cycles on a Re=2500 channel) and weak only at high
      Re / recirculation (17 cycles on a Re=1e5 channel). The weak block is the **flow saddle**, not the
      k/ω scalars (per-block error operator `E_b = I − A_b·M_b` on a developed Re=1e5 channel: flow
      ρ=34.0 / one-shot 24.1, vs ω 13.9 / **2.4** and k 8.5 / 7.9 — ω's high ρ with a low one-shot is an
      isolated outlier eigenvalue GMRES kills in one iteration, a red herring). But every lever *inside*
      that block is dead:
      - **More velocity-AMG V-cycles (×2/×4/×8): ρ 34.019 → 33.995 → 34.031 → 34.046 — no effect at all.**
      - **More Schur V-cycles (×2/×4/×8): ρ 41.6 / 48.7 / 48.5 — strictly worse.** Inverting `Ŝ` *more
        accurately* making the preconditioner *worse* is the signature that `Ŝ` is the **wrong operator**:
        the error is the Schur *approximation*, not its inversion (a partial V-cycle was accidentally
        regularizing it). Driving both sub-solves toward exact never beats the 1-cycle baseline.
      - **Rebuilding the preconditioner at the developed state (staleness) does not help *the flow
        block*** (ρ 34.0 → 31.6 on the channel; 49.9 → 91.9, i.e. worse, on pitzDaily, with an identical
        one-shot). The frozen *flow* reference is fine — the convective linearization is Peclet-robust and
        MSIMPLER's Schur is velocity-independent. **Confirmed on the real solve:** refreshing only the flow
        block at a separated pitzDaily state made it slightly *worse* (31 → 34 outer cycles at β=2).
      - **BUT refreshing the *scalar* k/ω AMGs is a real 2.6× cycle win once the flow separates — the one
        staleness lever that does pay (measured on the real solve, not ρ).** The scalars were noted above
        as going stale (ω ρ 13.9 → 3.3 rebuilt) but dismissed as "not the cycle bottleneck" on the ρ /
        one-shot proxy; on the **real coupled shifted solve** that dismissal does not hold. Marching
        pitzDaily to a genuinely separated state (25 pseudo-transient steps, rel 3.0e-2, 70 recirculation
        cells, `x_r/h` 0.87) and re-solving the **same** shifted system with the preconditioner refreshed
        block-by-block (operator held fixed; every solve converged, `‖Aδ−b‖/‖b‖` ~1e-8):

        | refreshed | cycles | matvecs | wall |
        |---|---|---|---|
        | nothing (all frozen at the cold IC) | 31 | 3720 | 68.9 s |
        | **k/ω scalar AMGs only** | **12** | **1440** | **27.4 s** |
        | flow block only | 34 | 4080 | 71.8 s |
        | everything | 13 | 1560 | 30.4 s |

        So the entire gain is the **scalars** (31 → 12), the flow refresh contributes nothing (everything
        ≈ scalars-only), and this is a textbook instance of the ρ caution above — the scalars' low one-shot
        made them look harmless while they were worth 2.6× on the real iteration. **The benefit only
        appears once the flow has separated**: at a *pre-separation* state (4 march steps, no recirculation)
        a full refresh is worthless (17 → 14 cycles at β=2, and *worse* at β=0.2, 43 → 83), which is why an
        early measurement gives the wrong answer. Full-refresh gains were confirmed at β ∈ {2, 0.5, 0.2}
        (31→13, 19→12, 31→18); the block-by-block isolation above was run at β=2. **Implication for
        implementation: refresh only the two `ScalarTransportPreconditioner`s and leave the flow block
        frozen** — much cheaper than a whole-policy rebuild, and it avoids the flow refresh's small
        regression. It is adjoint-safe (the preconditioner is `stop_gradient`-ed whatever it is frozen at,
        so a refresh changes only the forward Krylov count, never the converged state or its IFT adjoint).
        **BUILT** — `forward_march` + `CycleGrowthTrigger` (see the `march.py` section) segment the march
        around the off-jit rebuild, which is required because the traced solve is one `lax.while_loop` and
        scipy AMG assembly cannot run inside it; `solve_coupled(refresh_trigger=…)` is the driver.
        **⚠️ SETTLED FROM THE CODE — the old claim here, "a refresh still forces a full recompile because these
        are non-pytrees hashed by identity", is SUPERSEDED and deleted.** The fix it proposed as hypothetical was
        built: the coarsening structure is value-independent **at this path's `strength_threshold=0`**, and
        `_SparseLevel` now holds only `n` / `n_coarse`
        static with `val` / `diagonal` / `lam_max` / `coarse_inv` as **traced leaves** — so a refreshed hierarchy
        passed as a jit argument is a **compilation-cache hit**, pinned by
        `test_refreshing_a_hierarchy_is_a_compilation_cache_hit`. ⚠️ **The value-independence is a property of
        the threshold, not of the level split**, so it does not carry to the native flow block, which runs at
        0.25 and re-partitions on every refresh; that path keeps the cache hit with
        `SmoothedHierarchy.refit` instead (see the flow-block section). What a refresh still costs is the off-jit scipy
        rebuild plus the one-off retrace of the rebuilt `ForwardStep`, which is why `refresh_limit` still bounds
        it. The wall figures once attached to this question (a "~60–240 s" recompile and a "~38 s" refresh) were
        both recorded with no configuration and are deleted with it.
      - **The observed march RETURNS ITS OWN CONVERGED STATE — the traced finishing solve is only the
        not-converged fallback (BUILT).** `solve_coupled`'s observed path (`on_step`/`refresh`/`step_control`)
        is never differentiated — those cannot run under a JAX transform (guarded), so the converged eager
        state needs no adjoint. When the eager `forward_march` reaches its stopping tolerance **judged in the
        measure it steered by** (the per-step-rebuilt `RowScaledNorm` under `scaled_norm`), `solve_coupled`
        returns that state directly instead of re-marching it through `ImplicitNewtonSolver`. **Why this is
        required, not just an optimization:** the finishing solve targets the *frozen* base measure (state0
        row scales), which over-reports a developed state's residual (#156 seam 4), so it does not see the
        eager convergence — and being traced it cannot refresh or carry the SER step control, so on an
        aggressive low-shift ILUT dual-time path it leaves the converged state chasing the unreachable frozen
        target and **diverges to NaN** (measured on the pitzDaily Re/100 anchor: eager converges row-scaled
        0.009, finishing solve then returns ‖R‖=NaN). Returning the eager state fixes that. The finishing
        solve still runs when the eager march stops short, and is the plain differentiable path's sole march.
        **Open (for the target-Re adjoint):** a differentiated target solve still needs the finishing solve
        to converge deep *in the same row-scaled measure* — restructuring it (Python outer loop so it can
        carry the measure + step control) is the tracked follow-up; the lower-Re continuation rungs are
        `stop_gradient`ed seeds and need no adjoint, so the eager path serves them.
      - **Rescaling the MSIMPLER `k` is a ρ mirage — validate on the real march, never on ρ.** Growing `k`
        collapses ρ but barely moves the one-shot error (figures deleted with the rest of the unconfigured ρ
        evidence above), and the ρ-minimizing
        `k` sits ~40× *above the maximum* of the whole per-cell `ρV/a_P` distribution — i.e. the degenerate
        limit `schur_a_p → 0`, `Ŝ⁻¹ → 0`, which simply switches the pressure correction off. On the real
        production march it is **slower**: shipped auto-`k` 348 s / 8 steps vs `k×4` 447 s (28% slower) at
        an identical residual trajectory. **The shipped per-apply `mean(ρV/a_P)` calibration is
        near-optimal — do not "fix" it**, and do not make the Schur "shift-consistent" with the
        pseudo-transient `a_P(1+β)` either (that direction is strictly worse at every β).
      **Root cause:** the MSIMPLER Schur is a *constant-coefficient* (scaled pressure-mass-matrix) Poisson,
      which is a near-Stokes/low-Re approximation and degrades as convection strengthens — exactly the
      high-Re/recirculating regime here.
      - **⚠️ The "obvious" fix — a better Schur (stabilized LSC) — WAS BUILT AND LOSES BADLY on the
        coupled solve. Do not re-derive it (binding).** `schur_scaling="lsc"`
        (`flow/block_preconditioner.py`) implements the algebraic, nonuniform-mesh stabilized
        least-squares commutator of Elman, Howle, Shadid, Silvester & Tuminaro (2007) — the *right*
        variant for a Rhie–Chow collocated (equal-order stabilized) discretization, with the viscosity
        cancelled so it serves a variable-viscosity closure. Measured on one shifted solve at a
        developed/separated pitzDaily state:

        | Schur | cycles | wall |
        |---|---|---|
        | **msimpler** | **13** | **38.9 s** |
        | lsc (`v_cycles=4`) | 96 | 526 s |
        | lsc (`v_cycles=8`) | 82 | 662 s |

        6–7× the cycles and 13–17× the wall time, with both solves genuinely converged
        (`lin_rel ~2e-9`), plus ~2.9× slower on the coupled channel at an identical residual trajectory.
        **Why the flow-only win does not transfer:** LSC *does* beat MSIMPLER on the isolated flow block
        (9 vs 15 GMRES at Re=1e4), but on the coupled block-*diagonal* preconditioner under the
        pseudo-transient shift, a better isolated flow-Schur does not reduce *coupled* cycles — the
        coupled iteration is not limited by the flow block's Schur quality. Keep the strategy (it is a
        legitimate option for a flow-only solve); do **not** make it the coupled default, and do not
        propose it again as the cure for coupled cost.
      - PCD remains deprioritized regardless: its auxiliary pressure convection–diffusion operator
        carries finite-element boundary recipes that do not transfer cleanly to cell-centred FVM.
      - **What a preconditioner can and cannot change — state this precisely, both halves are measured.**
        *While the linear solve actually converges*, swapping the preconditioner changes **cost only**:
        msimpler vs LSC gave coupled residual trajectories identical to **5 significant figures**, so the
        Newton direction, the accepted step, and whether the march converges are all preconditioner-
        independent. That rules out a whole class of experiment — you cannot precondition your way out
        of a stalled march, only out of an expensive one.
        **But the guarantee is conditional on convergence, and it fails when a preconditioner is stale
        enough to degrade the solve.** Measured 2026-07-25 on the pitzDaily cold-IC march, refreshed vs
        unrefreshed at *identical* step indices: the unrefreshed arm sat at **α = 0.5 for steps 16–23
        while needing 53–85 cycles**, and the refreshed arm took **α = 1.0 at 10–13 cycles**, with
        different residual trajectories (rel 4.45e-2 vs 4.13e-2 at step 20). So a sufficiently stale
        preconditioner *does* change the step. The mechanism is not isolated — the natural reading is
        that the degraded solve is truncating or stagnating rather than reaching its tolerance, so the
        returned `δ` is no longer the tolerance-defined one — and `_COUPLED_FORWARD_SOLVER` runs
        `rtol=1e-3` with `stagnation_iters=40`, which makes that reachable. **Practical rule:** treat
        "preconditioner ⇒ cost only" as true when solves converge comfortably, and stop trusting it
        once the cycle count is climbing toward the solver's limits.
  - **The residual measure is an injected `ResidualNorm`, owned by the `ForwardStep` (`solve/norm.py`).**
    Every `ForwardStep` exposes `norm()`; `ImplicitNewtonSolver` reads it for the outer stopping test
    (threaded through `_forward`/`_implicit_solve` as the extra nondiff arg `norm_fn`) and the strategy
    uses the *same* measure for its own globalization — so the convergence test, the SER ramp
    `β = β₀(‖R‖/‖R₀‖)^p`, `backtracking_line_search` (which now takes a `norm=` kwarg), and the
    `DivergenceGuard` all agree on one scale. Default is `jnp.linalg.norm` (`DampedNewtonStep.norm()` and
    `PseudoTransientStep`'s `residual_norm` field both default to it), so **the flow path is
    bit-identical**. The non-trivial impl is `BlockScaledNorm(sizes, scales)`: it splits the flat
    residual into contiguous blocks, divides each by its own reference magnitude, and returns the L2 of
    those per-block relative residuals — `sqrt(Σ_b (‖R_b‖/scale_b)²)`. **Why it exists (the coupled-RANS
    fix):** the plain Euclidean ‖R‖ on `[flow, k, ω]` is ~100% ω (ω residual O(1e5), k O(1e-3)), so the
    line search can neither *see* nor *protect* the k block — a step that collapses k is accepted (barely
    moves the ω-dominated norm) while one that reduces k is vetoed because ω ticked up, and k gets
    starved (measured: k median collapses to ~7e-5 vs a physical ~0.5, and the march freezes). `coupled.py`
    builds a `BlockScaledNorm` over `[flow, k, ω]` (and `[…, β]` for the mass-flow bordered march) with
    per-field scales `‖R0_field‖` at the reference state, so the whole system is judged. The adjoint never
    forms a residual norm, so `norm_fn` is a **forward-only** device — the converged state and IFT
    gradient are norm-independent (the bwd pass takes it as a `del`-ed nondiff arg). Since it is a static
    field holding an `eqx.Module` with static tuple fields, it stays hashable for the `custom_vjp` nondiff
    slot (like the `lineax` solver already carried there).
  - **A `ShiftPolicy`'s preconditioner must stay a non-pytree (binding, #105).** `ScalarTransportPreconditioner`
    (`turbulence/preconditioner.py`) is a plain `dataclasses.dataclass(frozen=True, eq=False)` ABC with
    `ConvectionAmgPreconditioner` / `AirAmgPreconditioner` concrete strategies — deliberately **not** an
    `equinox.Module`. Two things break if it is made a pytree: (i) a solve taking it as an argument traces
    its hierarchy arrays, which then reach `_implicit_solve`'s `custom_vjp` as tracers in a
    `nondiff_argnums` slot and JAX raises `UnexpectedTracerError`; (ii) it is *because* the object is opaque
    to JAX that carrying one instance across outer sweeps is a `filter_jit` cache **hit** (non-array
    arguments go to the static side, hashed by identity). Both were hit and fixed while building #105 —
    do not "modernize" these into `equinox.Module`s.
- **`norm_builder` — the residual measure is re-derived every outer iteration, and held FIXED within
  one (binding).** `forward_march(norm_builder=…)` takes a `state -> ResidualNorm` and, at the top of
  each iteration, swaps the rebuilt measure onto the step with `eqx.tree_at` (the same mechanism the
  α-control uses for β) and re-measures `residual_norm_0` against it so the SER ratio stays on one
  scale. Every line-search trial step, the acceptance test and the reported norm within that iteration
  then use the *same* measure — **rebuilding per trial step would let a candidate win by shrinking its
  own denominator rather than its residual**, so the search would stop comparing like with like.
  - **This is why `residual_norm` is a DATA field on both `ForwardStep`s, not a static one.** A static
    field lives in the treedef, so swapping it would be a new compilation *every step*. As data, and
    with the measure carrying its scales as traced leaves over a fixed block structure
    (`RowScaledNorm`), the swap is a cache hit. A plain callable (the default) has no array leaves and
    is filtered to the static side regardless, so the default path is byte-identical.
  - **⚠️ `RowScaledNorm` is MARCH-ONLY today.** `ImplicitNewtonSolver` passes `forward.norm()` into
    `custom_vjp`'s `nondiff_argnums`, which requires a hashable object, and a pytree holding arrays is
    not hashable there. So the finishing solve keeps whatever measure it was constructed with. Letting
    the traced solver use it requires reworking that slot — not done.

## The observed march — forward_march, triggers, controls, logging

- **`march.py` — BUILT (`forward_march`, `StepReport`/`MarchResult`, `RefreshTrigger`/`CycleGrowthTrigger`):
  the observed, forward-only march that drives a mid-march preconditioner refresh.**
  - **Two marches, ONE decision layer (binding — this is the shape to hold).** `_forward` (traced,
    inside `custom_vjp`, has the root guard, cannot stop early, cannot be observed) and `forward_march`
    (eager Python loop, forward-only, **no guard by design**, stops on an injected trigger, reports every
    step). They are not duplicates: `forward_march` calls the **same** `forward_step.stepper()`, the same
    `forward_step.norm()`, and the same `_within_tolerance`. The only residue is a ~6-line loop shell,
    pinned against drift by a test that both marches reach the same state on the same residual.
  - **NOTHING in the refresh machinery reads the line-search α, and on `bfs3d` almost nothing reads
    anything else either (source-verified against the current defaults).** Two independent refresh paths
    exist and they key on different things: the post-step `RefreshTrigger`s (`CycleGrowthTrigger` →
    `cycles` + `residual_ratio`; `CoefficientDriftTrigger` → `ν_t` drift) and the per-attempt
    `precondition_step` hook (`amg_beta_tracking_refresh` → `beta_rel_change` / `materialize_drift` /
    `materialize_every` / `refresh_every`). **Neither reads `alpha` or `binding_limit`**, so a collapsed
    line search can only ever escalate β — it can never buy a rebuild. And in the shipped `bfs3d` bundle
    the proactive arms are switched off by construction: with the default `BFS3D_REFRESH_ON_CYCLES=3`,
    `compare.py` passes `beta_rel_change=inf`, `refresh_every=10**9`, `materialize_drift=None`,
    `materialize_every=None`, so the **only** live trigger is the reactive mid-step "one solve reached 3
    restart cycles". Two consequences worth holding: (a) an α-triggered refresh needs **no new trigger** —
    `precondition_step` is already called once per *attempt*, after the control has set β, so a finite
    `beta_rel_change` makes a β escalation pull a matched rebuild for free; (b) that would **not** address
    the lock-ups this case actually hits, which run at `binding_limit < 1` (the positivity ratchet) where
    the direction is measured accurate and the solve already over-delivers against its tolerance. Where α
    *is* the right refresh signal is the **constraint-free** collapse (`binding_limit == 1`, direction
    genuinely bad), and that case is invisible to every trigger today.
  - **Why the early-stop could NOT go inside `ImplicitNewtonSolver` (binding — do not "simplify" it back).**
    `_forward`'s guard raises whenever the terminal state is not a root, and a trigger-stopped segment
    exits un-converged *by design*. Injecting a count-based early stop would therefore require an
    **exemption** in that guard — creating a production path that returns a non-root without raising,
    which is exactly the silent-wrong-gradient hole the guard exists to close. Chunking `_forward` with
    `max_steps=1` fails independently: it recomputes `residual_norm_0` per chunk, pinning the SER ramp at
    β₀ forever.
  - **The eager march NEVER returns the answer.** It is a pure accelerator; every staged solve ends with
    a real `ImplicitNewtonSolver.solve()` that owns the guard, the `custom_vjp`, and the result. So the
    guard has exactly one home and is unconditionally on the path that produces the returned state.
  - **Two reference norms, and conflating them freezes the march (binding).** `residual_norm_0` is
    **segment-local** (recomputed at each `forward_march` entry, handed to `stepper()` for the SER ramp);
    `reference_norm` is **global** (fixed across segments, used for the convergence test and the reported
    ratio). Substituting the second for the first pairs a refreshed, larger shift diagonal with the small
    β belonging to the pre-refresh residual — the over-damping freeze documented in `turbulence.md`.
  - **Per-step jit cache hit is mandatory, not an optimization (top implementation risk).** The per-step
    call goes through the module-level `eqx.filter_jit`'d `_march_step`, taking the `ForwardStep` **and**
    the residual as *arguments*. Two caller obligations: pass the **same** `forward_step` object across a
    segment (a rebuilt one is the intended one-off recompile per refresh), and pass a **bound module
    method** (`coupled.residual`) rather than a freshly-built `lambda`, which `filter_jit` hashes by
    identity. Retracing per step would cost the 60–240 s compile *every step* and dominate the march it
    accelerates. Pinned by a trace-counting test (extra steps add zero traces). Note the residual is
    invoked several times *within one trace* (step, line-search ladder, norm), so trace count ≠ compile
    count — assert that further steps add none, not that the total is 1.
  - **The six retry settings are ONE injected object — `RetryPolicy` (`solve/retry.py`), exported from
    `aquaflux.solve` (BUILT).** `solver` / `divergence_cap` / `on_cycles` / `on_alpha` / `beta_factor` /
    `cycles_limit` used to be six parallel keyword arguments on **both** `forward_march` and
    `solve_coupled`, and they are meaningless apart: a `beta_factor` with no threshold escalates nothing,
    a `cycles_limit` bounds a loop that never runs. The three decisions taken from them read three or
    four each, so they are **methods on the policy** rather than free functions taking a subset —
    `escalation_reason` (was `_escalation_reason`), `has_diverged` (was `_has_diverged`),
    `with_inner_abort` (was `_with_inner_abort`), plus `escalate`, which exists solely to keep the
    "scale the β leaf, never rebuild it from a float" rule in one place (rebuilding changes the leaf's
    dtype/weak type and recompiles the whole coupled solve on every retry).
    - **`NO_RETRIES` is the shared default instance**, so both signatures' defaults name what the default
      *means* rather than how it is spelled; it is byte-identical to the old all-`None` defaults.
    - **The ORDER stays on `forward_march`, not on the policy** — escalation first, `solver` as the
      fallback — because it is a property of the loop, not of the settings. Likewise `on_retry` stays a
      separate argument: it is *reporting*, and it belongs with the other observation seams rather than
      with the thresholds.
    - **`forward_march` takes the whole policy; a refresh policy would NOT be passed the same way.** The
      march uses all six, so the object is the smallest sufficient collaborator there. Do not extend this
      to arguments a callee does not need in full.
  - **Reactive divergence retry — `retry.solver` recovers a step an INEXACT preconditioner poisons,
    without tightening every step (BUILT).** An *inexact* preconditioner (the threshold-ILU) can return a
    non-finite correction on the stiff operator an aggressive Courant overshoot produces, where the
    *exact* complete-LU returns a finite one — the loose default Krylov tolerance is what leaves that
    correction too inaccurate. `forward_march(retry=RetryPolicy(solver=…, divergence_cap=inf))` redoes a diverged
    step **from the same pre-step state** at the tighter `retry.solver`; the trigger is
    `RetryPolicy.has_diverged` (non-finite, or `> divergence_cap·reference` — default `inf`, i.e. non-finite only,
    because the residual legitimately *rises* during development via `β×travel`, so a tight cap would
    false-fire on the reachability descent). **The preconditioner is NOT re-refreshed on retry** — under a
    β-tracking refresh (`ilut_beta_tracking_refresh`, `.claude/rules/turbulence.md`) the factor is already
    fresh at this `(state, β)`, and re-factoring the deterministic factorization at the same point is a
    no-op; the failure is an under-converged *Krylov* solve, not a stale PC, so only the Krylov tolerance
    is tightened. This is orthogonal to the refresh *gating*: the retry recovers a diverged step whatever
    cadence the ILUT was refreshed at. One retry: a still-diverged
    step breaks as before. Default a policy with no `retry.solver` is **byte-identical**, and the exact-LU path never
    triggers it. **Why it beats tightening every step:** measured on the aggressive pitzDaily ILUT ramp,
    rung-1 steps 1–7 ran on the cheap loose solver and *only* the diverged step 8 retried tight —
    recovering to the exact-LU value (ratio 9.72e-2) and tracking the LU on — instead of paying the tight
    solve on every step. Threaded through `solve_coupled(retry=RetryPolicy(solver=…))`; forward-only (raises under
    `jax.grad`, same guard as the refresh/control). Pinned by `test_forward_march.py`
    (`test_march_retries_a_diverged_step_with_the_tighter_solver`, `test_march_does_not_retry_a_finite_step`).
    On 2D the exact LU is cheaper *and* robust for free, so this is really a 3D-readiness lever (where the
    LU's fill is the wall and the ILUT is the only option).
  - **Reactive β-escalation bailout — `retry.on_cycles` ESCALATES β for a bad step, tried BEFORE the tight
    divergence retry (BUILT).** A step goes bad two ways — a *finite-but-expensive* solve (count `> N`) or a
    *non-finite* one — and on the stiff low-β saddle **both have the same cheap cure: more damping.** A
    larger β lifts the correction out of the NaN regime *and* cuts the cycle count (a stronger pseudo-time
    shift makes the same frozen preconditioner more diagonally dominant), and it is far cheaper than the
    tight-Krylov divergence retry. So `forward_march(retry=RetryPolicy(on_cycles=N, beta_factor=2.0,
    cycles_limit=2))` redoes a step whose count exceeds `N` **or** that diverged (non-finite / over
    `retry.divergence_cap`) **from the same pre-step state** with β escalated (`×retry.beta_factor`,
    re-applying `precondition_step` at the new β so a β-tracking refresh re-shifts), up to
    `retry.cycles_limit` times or until it converges/drops below `N`. It reads β off
    `active_step.relaxation_schedule.beta` (a `ConstantRelaxation` / `DualTimeStep`), so it requires a
    readable β and is inert on the default switched-evolution schedule. an unset `retry.on_cycles` (the default) is
    **byte-identical** (and a diverged step then falls straight to `retry.solver`, the pre-reorder
    behaviour). Forward-only; threaded through `solve_coupled(retry=RetryPolicy(on_cycles=…))`. Pinned by
    `test_forward_march.py` (`test_march_escalates_beta_on_a_cycle_count_spike`,
    `…_does_not_escalate_below_the_cycle_cap`, `…_escalates_beta_before_the_tight_divergence_retry`,
    `…_falls_back_to_the_tight_retry_when_escalation_cannot_fix_divergence`).
    - **The escalation must keep `_march_step` a compile-cache HIT (binding — a measured recompile
      hazard).** The retried step redoes the (coupled, minutes-to-compile) `_march_step` at the escalated
      β, so a treedef **or aval** change on that step recompiles the whole solve every retry — which on a
      stiff region that retries most steps is ~half the march wall. β is a dynamic 0-d leaf, so the *value*
      change is fine; the trap is the leaf's **abstract value**. Escalate by **scaling the existing leaf**
      (`beta * retry.beta_factor`), never by rebuilding it from a Python float (`jnp.asarray(float(beta) *
      f)`): the latter yields a *weak*-typed float64 array whose dtype/weak_type need not match the leaf
      the step control set, and any mismatch is a cache miss. Scaling preserves the aval exactly, so the
      retried step is a hit for **any** β dtype (weak/strong f64, f32). The shipped controls happen to set
      weak-f64 (so the old form was accidentally a hit), but that was an unpinned coincidence one JAX
      weak-type-promotion change from breaking. Pinned by
      `test_a_forced_escalation_adds_no_march_step_compilations` (a strong-typed β leaf, the case the
      rebuild recompiled on). Note the `retry.solver` (divergence) fallback is a *separate*, one-off
      recompile — a distinct solver object (restart-40 vs the forward restart-15) is a genuinely different
      static key, compiled once and reused; it is not a per-step cost.
    - **Why escalation leads and `retry.solver` is the FALLBACK (the reorder, measured on `bfs3d`).** The
      two retries used to run divergence-first: a NaN'd step ran the tight `retry.solver` (a 1e-4 Krylov
      solve, restart-40) and *then*, seeing its high count, the cycle bailout escalated β. On the 3D march
      that order was the single worst cost — measured on the cold `bfs3d` cold-continuation, step 28's
      primary NaN'd (α collapsed), the tight retry ground ~325 matvecs (~40 min) to recover it to finite,
      and *then* the β-escalation re-damped the same step to a clean ~5-cycle solve — so the entire tight
      grind was wasted work the escalation superseded. Reordered, the escalation fires first on the NaN,
      recovers the step cheaply, and the tight `retry.solver` fires only as a **fallback** for a non-finite
      step escalation could *not* fix — the genuine inexact-ILUT case (loose Krylov → non-finite δ that a
      tighter Krylov, not more damping, cures), where `retry.on_cycles` is typically `None` anyway so
      escalation is absent and the divergence retry is the sole, original mechanism.
    - **This is the PROACTIVE β-mismatch refresh's reactive twin** — the refresh
      (`amg_beta_tracking_refresh(beta_rel_change=…)`, `.claude/rules/turbulence.md`) re-freezes the PC
      *before* a solve when β has drifted (the stale-PC cause of a spike); the bailout escalates β *after* a
      solve reveals a hard operator. **The bailout is REACTIVE by necessity: a Step-0 diagnostic on `bfs3d`
      (42-step instrumented capture) showed no cheap STATIC operator property predicts a bad step** — the
      diagonal-dominance defect of the frozen shifted operator does not separate bad from good (the rung-1
      trio 10/11/12 have near-identical DD but 302 vs 12 matvecs; the hardness is non-monotone in β and
      refresh-invariant), so a predict-then-avoid β-chooser was refuted and detect-then-react is the honest
      design. Refresh for staleness, escalate β for stiffness — a cost spike does not distinguish the two on
      its own, and neither does any static probe of the operator.
    - **`DualTimeStep(cycle_budget=…)` makes the escalation CHEAP — cap the primary grind, don't run it to
      completion (BUILT).** The escalation above is reactive-after-the-solve, so its cost is set by how
      expensive the doomed primary is *before* it returns. The dual-time step runs up to `inner_steps` inner
      Newton iterations, each a restart-capped GMRES; on a grinding primary every inner ran to the
      stagnation cap, so the step burned `inner_steps ×` a full stagnation before the escalation could fire
      (measured ~5× the necessary cost on `bfs3d` — steps 11/25/27 ground ~300 matvecs where one capped
      inner is ~60). `cycle_budget` stops the inner loop once its *accumulated* Krylov count reaches the
      budget (`cond` gains `& (cycles < cycle_budget)`, elided at trace time when `None`), so a grinding
      primary is cut after ~one over-budget inner iteration and the partial iterate is handed to the
      escalation, which redoes it at a larger β where it converges cheaply. **Pair it with
      `retry.on_cycles < cycle_budget`** so a capped primary's reported count trips the escalation (else the
      partial non-converged step would be accepted). Good steps converge well under the budget, so they are
      byte-identical; only a grinding primary hits it. `cycle_budget=None` (default) is unbounded and
      byte-identical. Threaded through `coupled_amg_continuation(cycle_budget=…)` (and the shared
      `_monolithic_factor_step`, so the ILUT/LU steps can take it too); forward-only, like the escalation it
      feeds. This is Agent C's "small-budget primary + inner abort" realized as an inner-loop cost cap rather
      than a non-attainment flag threaded through every solve layer — same effect (a doomed primary costs
      ~`cycle_budget` matvecs, not `inner_steps ×` a stagnation), far smaller blast radius. Pinned by
      `test_dual_time.py` (`…cycle_budget_caps_the_inner_loop`, `…none_is_the_unbounded_step`).
      **⚠️ CORRECTION: "a doomed primary costs ~`cycle_budget` matvecs" is MEASURABLY FALSE, because the
      budget is checked BETWEEN inner iterations.** A single inner solve is bounded only by its own
      `stagnation_iters=40` / `max_restarts=60`, both ≈ the whole budget, so one solve can blow through
      it. Measured on the 3-rung `bfs3d` march at `cycle_budget=42`: the three discarded attempts cost
      **26 / 56 / 59** cycles, entering their last inner having spent only 14 / 17 / 16. The budget did
      bind — just a whole stagnating solve too late.
    - **`DualTimeStep(abort_above_inner_cycles=…)` — stop the moment the attempt is KNOWN to be
      discarded (BUILT).** `retry.on_cycles` is a **per-solve** quantity, so the instant one solve
      exceeds it with the inner target unmet, `forward_march` is going to bin the whole attempt and redo
      it at a larger β. Yet the check lived only in `forward_march`, *after* the step returned — so the
      step kept running inner iterations whose results were already destined for the bin. The same
      predicate now sits in the inner loop's `cond`, and `forward_march` pushes its own `retry.on_cycles`
      down via `RetryPolicy.with_inner_abort` (using `dataclasses.replace`, not `eqx.tree_at` — the field is static,
      so it is in the treedef, not among the leaves), so there is **one** number rather than two to keep
      in step.
      **It cannot bin an expensive success**, and the ordering is what guarantees that: `cond` tests the
      convergence target *before* either cost bailout, so a costly solve that brings `‖G‖` under the
      target exits normally with `reached_target` set and is kept. The empirical backing is strong on
      this case — across 64 attempts, **no kept attempt ever had a single inner solve above 10 cycles**,
      while all three discarded ones ran 12/15/43 — so the threshold separates them perfectly.
      **MEASURED on the next march, and the prediction held where it applied:**

      | step | discarded attempt before | after | cycles saved |
      |---|---|---|---|
      | 48 | `[2, 9, 5, 43]` | unchanged | 0 (predicted 0 — the trip is on the last inner) |
      | 51 | `[2, 12, 4, 4, 4]` | `[2, 12]` | 12 (predicted 12) |
      | 52 | `[2, 15, 39]` | `[2, 19]` | 35 (predicted 39) |

      Discarded-attempt cycles 141 → 107 (a new retry at step 53 cost 13 of the 47 saved), and the
      retry region's wall fell **250 s** (step 51 −77 s, step 52 −165 s).
      **⚠️ MEASURE IT IN WALL, NOT IN THE CYCLE TOTAL.** The march's reported `cyc` is the **accepted**
      attempt's count only, so discarded work was never in it: total cycles moved 348 → 347 while a real
      250 s came out. A prediction phrased against the cycle total would read as a total miss.
      **⚠️ It is NOT purely a cost change.** `precondition_step` runs per *attempt*, so truncating a
      discarded attempt changes the refresh sequence and hence the V-cycle the next step sees: the two
      marches agree step-for-step through 52 and then diverge (61 vs 62 steps, same `x_r/h` 8.36, both
      converged). Benign here, but do not describe the abort as trajectory-neutral.
      A further ~31 cycles are available if the per-solve `stagnation_iters`/`max_restarts` are brought
      down toward the threshold (still ~4–6× it, which is why one solve can eat the step budget); step 48
      is the case that needs it, since its trip lands on the last inner where the abort cannot help.
      `None` (default) is byte-identical. Forward-only. Pinned by
      `test_dual_time.py::test_abort_above_inner_cycles_{stops_a_doomed_attempt_early,
      never_bins_an_expensive_success, none_is_the_unbounded_step}` and the `RetryPolicy.with_inner_abort` plumbing
      tests beside them.
    - **A COLLAPSED STEP LENGTH is the third way a step dies, and neither the cost bailout nor the
      escalation could see it — `retry.on_alpha` / `abort_below_alpha` (BUILT 2026-08-08).** The
      escalation's two triggers are cost-with-the-target-unmet and divergence. A step whose *solves are
      cheap* and whose *residual is finite* but whose **line search cannot move** is neither, so it was
      accepted as taken, and the only thing raising β was the step control's own backoff — **one doubling
      per outer step, each doubling paying a full step to discover it was not enough.**
      Caught on the shipped 3D `bfs3d` field-split march (`refresh_on_cycles=3`, ILU(0)×4, plain
      aggregation, `coarse_eq_limit` 2000, `N_POINTS=2`), target rung, four consecutive steps 51–54:

      | step | β | cyc | R | a_min | flg |
      |---|---|---|---|---|---|
      | 51 | 0.0390 | 12 | 7.061e-03 | 0.000 | L |
      | 52 | 0.0780 | 10 | 6.409e-03 | 0.000 | L |
      | 53 | 0.1561 | 7 | 8.224e-03 | 0.000 | L |
      | 54 | 0.3121 | 9 | 6.365e-03 | 0.000 | L |
      | 55 | 0.6243 | 5 | 5.530e-03 | **1.000** | |

      Step 51's inner table shows what the step row cannot: inner 0 descends (α 0.974, rate 0.779), then
      **inners 1–4 all report rate 1.000 at α 0.000** — four re-solves from an unchanged iterate,
      returning it unchanged — with `limit 4.37e-10`, i.e. the **positivity cap**, not the descent test,
      is what admits nothing. The four steps cost ~233 s to cross half a decade.
      Both halves are now built and are the same predicate in two places, as the cost bailout already is:
      **`forward_march(retry=RetryPolicy(on_alpha=α))`** escalates β (reason `"alpha"` on `on_retry`), and it is pushed
      into **`DualTimeStep.abort_below_alpha`** by `RetryPolicy.with_inner_abort` so the inner loop exits at the
      collapse instead of iterating on. `RetryPolicy.escalation_reason` now owns which of the three reasons applies,
      so the decision and the string reported for it cannot disagree.
      **Both cost and step-length reasons require the target unmet; divergence does not** — a non-finite
      residual is not a result to keep because the loop happened to meet its tolerance. Both `None`
      (default) is byte-identical.
      **MEASURED END TO END, and ONE retry is worth 8 steps and 199 s.** Same case and configuration as
      the baseline above (field split, `refresh_on_cycles=3`, ILU(0)×4, plain aggregation,
      `coarse_eq_limit` 2000, `N_POINTS=2`), `retry.on_alpha = 0.01`:

      | | baseline | with the trigger | |
      |---|---|---|---|
      | **wall** | 2161 s | **1959 s** | **−9.3%** |
      | steps | 66 | 58 | −8 |
      | Krylov cycles | 324 | 277 | −15% |
      | mid-span `x_r/h` | 8.361 | **8.361** | identical |

      **The run is its own control, which is what makes the attribution safe:** the two lower rungs come
      out *identical to the cycle* — Re/100 14 steps / 359 s / 45 cycles in both, Re/10 23 steps / 99
      cycles, 668 s against 671 s — because the trigger is inert wherever the line search is healthy. The
      whole difference is the target rung, 29 steps / 1131 s / 180 cycles → **21 / 932 / 133**. The
      trigger fired **once in the entire march** (step 51, `alpha`, β 0.0390 → 0.0780), and the inner
      abort cut that attempt from 5 inner iterations / 12 cycles to 2 / 6.
      **Note the cycle count DOES show this one, unlike the cost abort** — there the wasted work sat in
      *discarded* attempts, which the total never counted (348 → 347 cycles while 250 s came out); here
      it sat in **accepted** steps, so both measures agree. Which measure can see a saving depends on
      whether the work being removed was accepted or discarded, so decide that before quoting either.
      **The threshold was calibrated from the baseline's own step table, not chosen:** no productive step
      went below `a_min` 0.191, all four dead ones reported 0.000 (inner collapses at 0.001 and 0.003),
      so 0.01 sits an order of magnitude clear of both. Recalibrate on another case rather than porting
      the number.
      **Known waste left on the table (deliberate, not yet fixed):** the mid-step cost refresh fires in
      the *same* body call as the collapse, so a doomed attempt still pays it (~14.7 s) and the escalated
      retry then re-matches the preconditioner anyway. Suppressing it would need to distinguish a
      constraint-bound step (`binding_limit < 1`, where no preconditioner can help) from a non-descending
      one (where a refresh might be exactly the cure) — `binding_limit` exists for precisely that
      distinction. Measured evidence that the refresh cannot rescue a constraint-bound step: at baseline
      step 51 the mid-step refresh fired at inner 1 and took the following solves from 5 cycles to 2,
      while α stayed 0.000 and `‖G‖` did not move for three more iterations.
    - **The escalated β is CARRIED into the control — so a static β floor can be dropped and the *controller*
      decides how low is safe (BUILT).** β is inverse to the pseudo-timestep, so a static `beta_min` is a cap
      on the *largest* timestep the march may take, applied everywhere — which slows convergence in regions
      that could safely take a bigger step. The escalation is the per-region feedback for "how low is safe
      *here*": it fires exactly where β went too low. But without carrying it back, the next outer step's
      `step_control.next_step` recomputes β from the control's own (floor-ward) trajectory and **re-pays the
      escalation every step** — the observed low-β tail (β pinned at the floor, each step re-escalating). So
      after an escalation `forward_march` seeds the control's carried β with the escalated value via
      `step_control.carry_beta(state, β)` — **one implementation on `ShiftStrengthControl`, over the shared
      `(beta, memo)` state, so no control can be missing it** (it once was: the deleted single-step
      α-targeter had none, and the `hasattr` guard below meant its escalation feedback vanished in
      silence). The memo is preserved across the seed, so a ratio-keyed control does not lose its
      reference and misread the next step as a huge reduction. The control then continues its grow/brake dynamics *from* the discovered-safe β, so
      `beta_min` can be driven toward zero and the controller — with escalation as the safety net and the
      carry as the memory — finds how large a timestep each region tolerates, rather than a global floor
      capping it. Only fires when β was actually escalated and the control exposes `carry_beta`; no
      escalation ⇒ byte-identical. Pinned by `test_forward_march.py`
      (`…carries_the_escalated_beta_into_the_control`) and `test_step_control.py`
      (`test_carry_beta_seeds_the_carried_state`).
  - **`CoefficientDriftTrigger` — the PREFERRED staleness trigger: measure the drift, don't infer it
    from cost (binding for new work).** A frozen preconditioner is stale exactly when the operator it
    approximates has moved, so the honest signal is that movement itself. `StepReport.drift` carries a
    **scalar** relative drift produced by `forward_march(drift_measure=…)`; the coupled RANS measure is
    `turbulence.eddy_viscosity_drift(coupled, reference_state)` — `‖Δν_t‖/‖ν_t,ref‖` — because `ν_t` is
    what the frozen k/ω transport operators are assembled from.
    - **Why it beats `CycleGrowthTrigger` (which it supersedes for this job).** The cycle count rises
      from staleness **and** from `β → 0` ill-conditioning the shifted system, and on a separating flow
      the second is the *larger* — hence that class's `max_residual_ratio` gate and its `patience`.
      Drift has neither confound: it does not respond to `β`, so **no gate**, and it moves smoothly with
      the flow instead of jumping on one stiff solve, so **no patience**. It also cannot fire before the
      flow develops — the regime where a refresh was measured to be actively *harmful* (43 → 83 cycles)
      — because an undeveloped flow is by definition one whose `ν_t` has not moved. This is the fix for
      the confound recorded as #19.
    - **The scalar is the whole design (binding).** Drift needs the *state*, but the state must not
      reach a trigger: `checkpoint` carries state and `observer` carries only numbers precisely so a
      trigger stays a pure function replayable against one logged march (see below). Reducing drift to a
      number on the report keeps that property *and* lets the trigger see the physics — so calibration
      is still "log one march with `trigger=None` and a `drift_measure`, replay thresholds offline".
      Do **not** widen the trigger interface to take the state.
    - **The measure must be RE-BASED at every refresh (binding — a real trap).** `solve_coupled` builds
      a fresh `eddy_viscosity_drift` per segment against that segment's starting state, which *is* the
      state the current preconditioner was frozen at. Carrying one measure across segments would keep
      reporting drift the refresh had already absorbed, so the trigger would re-fire on the next step
      and burn the whole `refresh_limit` in consecutive steps. Same discipline as the segment-local
      `residual_norm_0`. Pinned by
      `test_the_drift_measure_is_rebased_at_every_refresh`.
    - **CALIBRATED, and the premise validated, on an instrumented cold-IC pitzDaily march (2026-07-25 —
      this closes #17 for this case).** One logged march with `refresh_trigger=None` + `on_step`, which
      still records drift because `solve_coupled` observes whenever an observer is supplied:

      | step | 5 | 10 | 12 | 14 | **15** | 17 | 19 | 20 | 22 | 23 |
      |---|---|---|---|---|---|---|---|---|---|---|
      | drift | 0.038 | 0.057 | 0.073 | 0.092 | **0.106** | 0.137 | 0.172 | 0.189 | 0.226 | 0.243 |
      | cycles | 10 | 12 | 13 | 17 | **21** | 28 | 45 | 53 | 85 | 84 |

      **The premise holds:** the cycle count sits flat at its 9–13 floor while drift climbs steadily,
      then rises monotonically with it — so `ν_t` movement is what makes the frozen preconditioner
      expensive, and drift *leads* cost early (drift is already 5× its step-0 value at step 5 while
      cycles are still at the floor). `threshold = 0.1` fires at step 15, where cost has just doubled
      off the floor and the recirculation has formed (`x_r/h` 0.32) — before the steep part
      (45→53→85) and clear of the pre-separation regime where a rebuild was measured to make things
      *worse*.
    - **The shipped default was originally 0.5 and would never have fired on this march** (drift
      reaches only 0.24 by step 23). A "conservative" placeholder chosen by intuition was not
      conservative — it was inert. Trigger numerics must come from a logged march, not from judgement
      about what sounds safe; the replay procedure exists precisely because that judgement is unreliable.
      Calibrated on **one** geometry, so treat 0.1 as a starting point elsewhere, not a constant.
    - **END-TO-END RESULT at `threshold = 0.1`, `refresh_limit = 8`, against the logged control (same
      cold IC, same everything, `refresh_trigger=None`) — a 5–8× cost win, sustained and repeatable:**

      | global step | 16 | 18 | 20 | 21 | 22 | 23 |
      |---|---|---|---|---|---|---|
      | refreshed cycles | 13 | 11 | **10** | 10 | 10 | 11 |
      | control cycles | 24 | 33 | **53** | 74 | 85 | 84 |

      ~21 s/step versus ~190 s/step, and the refreshed march was simultaneously **ahead on residual**
      (rel 2.67e-2 vs 3.03e-2 at step 23). Three further observations worth keeping:
      - **A refresh repays itself inside one step, which is why `refresh_limit` can be generous rather than
      hoarded.** The absolute figure that used to sit here, and the "~60–240 s recompile" it was contradicting,
      were both recorded without a configuration and are deleted; the recompile question is settled from the
      code above — a hierarchy refresh is a jit cache hit.
      - **It repeats across segments.** Refreshes fired at steps 15 and 30, each time on that segment's
        *own* drift accumulating from ~0 to 0.10 — the production confirmation of the per-segment
        re-basing.
      - **α improved 0.5 → 1.0** across the refresh (see the correction to the "preconditioner cannot
        change α" claim above).
    - **What the refresh does NOT fix — state this when reporting it.** The march still grinds: after the
      second refresh the residual moved ~0.3 %/step (rel ~1.77e-2) with cheap (11-cycle) full (α = 1)
      steps, and `x_r/h` crept 0.48 → 0.71 against OpenFOAM's 7.74. So the refresh solved the **cost**
      problem, not the **step-productivity** problem (#22). Its real value beyond the speedup is that the
      cost confound is now *gone*, so a β/α-control experiment finally measures what it claims to.
  - **`CycleGrowthTrigger` — cost growth is the trigger, the residual is the GATE.** Fires only when: the
    `warmup` is past; `residual_ratio <= max_residual_ratio`; and the last `patience` steps each measured
    `>= growth ×` the segment's **running-minimum** non-zero count. **Why the residual is demoted to a
    gate:** the cycle count rises for two reasons, and on this case the *wrong* one is larger. From the
    measurements above — staleness at fixed β=2 is 17 → 31 cycles (**1.8×**), while β alone at a fixed
    pre-separation state is 17 → 43 (**2.5×**). So a bare "cost has doubled" rule fires from the SER ramp
    before the flow separates, and a mis-fire is not neutral: a pre-separation refresh measured
    **43 → 83 cycles**, plus a wasted scipy rebuild and a recompile. Since `β = β₀(‖R‖/‖R₀‖)^p` is a
    function of the residual ratio alone, gating on the ratio normalizes the confound **without**
    re-deriving the schedule or widening the `stepper()` contract to return β.
  - **Zero-count trap (real, pinned).** `stepper()` reports `0` for a fully-rejected step, and a direct
    solver reports `0` too. A running-minimum baseline of `0` makes `cycles >= growth*0` always true and
    latches the trigger on permanently — so the trigger **ignores zero-count reports** for both the
    baseline and the growth test, and stays disarmed until one positive count exists.
  - **`refresh_limit` lives on the driver, not the trigger.** That keeps the trigger a **pure function of
    one segment's history** — which is what lets `warmup`/`patience` re-apply correctly after each
    refresh, lets it be unit-tested on synthetic histories with no solve, and (the big one) lets it be
    **calibrated offline**: log one march with `refresh_trigger=None` and an `on_step` observer, then
    replay candidate parameters against the log. No numeric default here is calibrated — they are chosen
    conservative (late rather than early) and must be set from an instrumented full-mesh run.
  - **Observation does NOT require a refresh (binding — this was a real bug).** `solve_coupled` runs the
    observed pre-march when the caller wants a refresh **or** merely wants to watch
    (`observing = refreshing or on_step or on_checkpoint`). Gating it on the trigger alone makes an
    *instrumented reference march* — `refresh_trigger=None` plus an observer, which is exactly the run a
    trigger is calibrated against, and the longest-running one — produce **no output at all** and sit
    silent for hours. Consequence to keep in mind: an observed solve spends `max_steps` on the pre-march
    and `max_steps` again on the finishing solve, so the budget is larger but *split*; instrumenting a
    solve already near its limit can turn a pass into a convergence-guard raise. Pinned by
    `test_the_march_reports_progress_without_a_refresh_trigger`.
  - **`checkpoint` is a SECOND seam, separate from `observer` (binding).** `checkpoint(report, state)`
    carries the state; `observer(report)` carries only numbers. Keeping the state off the report history
    is what keeps a `RefreshTrigger` a pure function that can be replayed offline against a logged march
    — put the state on that seam and a trigger could read the physics, and trigger calibration would cost
    one full solve per candidate instead of one logged run for all of them.
  - **Reporting seam.** `StepReport(step, cycles, residual_norm, residual_ratio, alpha, drift,
    inner_iterations, shift, escalations, diverged_retry)` + `MarchResult`,
    plus an optional streaming `observer` (a long march must not withhold all logging until it finishes).
    The trigger and a future logger consume the identical objects, so there is no second reporting path.
    Per-step observation exists only where the march is eager — the traced `_forward` would need
    `jax.debug.callback`, a separate decision; do not promise per-step reporting on the differentiable path.
  - **`shift` / `escalations` / `diverged_retry` on the report, and `MarchLogger` (`solve/march_log.py`)
    — the reporting half of the `on_step` seam (BUILT).** Every driver used to write its own
    `on_step`/`on_checkpoint` formatter, so the copies drifted and a gap fixed in one persisted in the
    others; `MarchLogger` owns everything derivable from a `StepReport` and takes an injected
    `metrics: state -> {name: value}` for the case quantities the solver cannot know (a reattachment
    length — `compare.reattachment_metrics` is the bfs3d one). `note()` writes an arbitrary line to the
    log's **own** stream (a driver reaching for `print` alongside it splits the run across two
    destinations, and a file log then loses whichever lines went to stdout); `phase(label, total)`
    auto-numbers, so a caller keeps no counter.
    Three report fields exist because the log could not otherwise carry them. **`shift`** is the `beta`
    the step was taken at — already read at the construction site for `carry_beta`, and otherwise
    recovered by every driver wrapping the step control. **`escalations` / `diverged_retry`** record
    whether the step was redone: `cycles` counts only the **accepted** attempt, so a redone step is
    indistinguishable from a cheap one, **and a retry mechanism left unconfigured never announces its
    absence** — which is not hypothetical (a bfs3d march ran with `cycle_budget` set but
    `retry.on_cycles` at its `None` default, so the beta-escalation never fired and nothing in the log
    said so; the cap exists to trip that trigger).
    The logger also reports the **reference norm and the stopping target**: the test is
    `‖R‖ <= atol + rtol·‖R₀‖`, and the march reports `‖R‖` and `‖R‖/‖R₀‖` but not `‖R₀‖`, so the target
    was invisible. Pinned by `tests/unit/test_march_log.py` (synthetic reports, no solve).
  - **Solve cost is reported offset-corrected, split by inner iteration (BUILT).** The raw count is
    lineax's `num_steps`, which carries **+2 per solve**, so a two-inner step reporting `cycles = 6` did
    two ideal single-cycle solves — the raw number is *entirely* offset and overstates the work
    threefold, worst exactly on the cheap near-root steps where the march's economics are decided.
    `restart_cycles(raw, solves)` in `solve/linear.py` is the one place that offset is stripped;
    `StepReport.restart_cycles` and `MarchLogger` both call it (it was previously open-coded as
    `cycles - 2*inner_iterations` on the report, and the logger printed the raw count instead).
    The step line reports `in` (inner count), `cyc` (corrected total) and `c/in` together, because one
    summed number conflates *how many* solves the step needed with how hard each was.
  - **`MarchLogger.on_inner` + `TextTable` (`aquaflux/text_table.py`) — the per-inner table (BUILT).**
    `DualTimeStep.inner_observer` already emitted `(index, ‖G‖ before, ‖G‖ after, cycles, alpha)` per
    inner Newton iteration via `jax.debug.callback`; nothing formatted it. `on_inner` matches that
    signature (`inner_observer=logger.on_inner`) and renders each iteration as a row — its own solve
    cost and the inner contraction `rate = ‖G‖out/‖G‖in` — inside a ruled table opened at `index == 0`
    and closed by the step line.
    Two ordering consequences, both deliberate: the **summary is a footer, not a header**, because the
    step's outcome is not known until it returns and buffering the block to lead with it would cost the
    live progress the rows exist to give (the title carries what *is* known — the step number, the
    elapsed time, and the residual the step inherits); and a **redone step opens a fresh block per
    attempt**, so the extra blocks are the record of what a retry cost, while the step line still
    reports only the accepted attempt.
    `TextTable`/`Column` is a **root leaf** (no package imports, like `vectors.py`) so any subsystem can
    format a report with it. It is built for streaming — rule, headings, `row` and `spanning` are
    separate methods each returning one line — which is what lets a table appear in a log being tailed.
    `rule(title=None, *, fill="-", segmented=True)` draws all three kinds the framed step block needs:
    the light **segmented** rule that divides one grid, the heavy `fill="="` one that **brackets** a
    block containing several grids, and the `segmented=False` span for the boundary between two grids
    that share a width but not a column layout — where a segmented rule would appear to belong to
    whichever grid its ticks happened to line up with. A step therefore renders as **one framed block**:
    step row, then the per-equation grid, then the asides **one concern per line** (preconditioner /
    case metrics / `limit` / `cum`). Run together on a single line those three had to be read in full to
    find any one of them.
    An over-wide value **widens its row rather than being truncated**: a cut-off number is a wrong
    number. Pinned by `tests/unit/test_text_table.py`.
  - **`StepOutcome` — the forward step's return is a record, not a tuple (BUILT).** It grew to six
    values (`phi, cycles, alpha, inner_iterations, reached_target, max_inner_cycles, binding_limit`),
    which is the missing-object smell: a positional tuple is where a consumer silently mis-unpacks one
    field for another, and every growth broke all five test doubles separately. Three of the fields are
    there because a march could not otherwise act correctly:
    - **`reached_target`** — did the step run to its OWN stopping criterion, or was it cut short? A
      cost-only escalation cannot tell an expensive success from a grind and **discards the success**:
      measured, an inner loop that reached `‖G‖ = 3.0e-6` against a `1.0e-5` target was thrown away for
      costing 54 raw cycles, wasting the work *and* replacing it with a shorter step. `forward_march`
      now escalates only when `cycles > retry.on_cycles` **and not** `reached_target`.
    - **`max_inner_cycles`** — the offset-corrected cost of the step's most expensive SINGLE solve, and
      what `retry.on_cycles` now triggers on. A **summed** threshold is not a difficulty signal: it
      grows with how many times the step solved, so at `retry.on_cycles = 40` a 5-inner step trips at 6
      corrected cycles per solve and a 1-inner step at 38 — a 6× difference in sensitivity decided by a
      count that says nothing about conditioning. (Measured impact on one 63-step march: the two
      triggers agree on 62 steps and disagree on one — the expensive step, which per-inner catches and
      summed misses. The change is correctness, not speed.) **`cycle_budget` stays SUMMED** and should:
      it is a cost cap, and total cost is exactly what it caps.
    - **`binding_limit`** — the step cap where it was the *binding* constraint, else 1. A small `alpha`
      has two opposite causes (the direction overshot; a constraint stopped it being followed further),
      so `alpha` alone cannot be acted on or reported honestly.
  - **Positivity is NOT carried by the shift and the divergence guard (binding — a stated invariant that
    was wrong).** The direct-scalar path documented positivity as following from the pseudo-transient
    shift plus the guard. It does not: the guard fires on a *non-finite residual*, and by the time
    `sqrt` of a negative value has produced one, the state is already poisoned. Measured on `bfs3d`: a
    march ran 62 healthy steps and died because **two cells out of 23040** took `k` to `-3.3e-4` — every
    field still finite, all of `u, p, k, omega` moving by ~1e-4, only the derived `nu_t` NaN, because
    the SST closure takes `sqrt(k)`. The line search had crawled along that boundary for four inner
    iterations (`alpha` halving `0.008 → 0.001`, residual falling under 1% each) with no way to see it.
    - **`positive_block_limit(start, stop, tau)` COMPUTES the admissible fraction, it does not search
      for one** — the fraction-to-the-boundary rule, `alpha_max = tau·min(phi_i / −delta_i)` over
      decreasing entries. Rejecting violating rungs is **not** sufficient: `alpha` was already at the
      ladder's shortest rung when the field crossed zero, so there was nothing shorter to fall back to.
      `backtracking_line_search(max_alpha=…)` caps every rung **including the growth rungs**, and its
      default is `inf`, **not 1** — a default of 1 silently disables the growth rungs, a regression the
      growth tests caught.
    - `turbulence.positive_k_limit(coupled)` returns the limiter for a directly-solved `k` and `None`
      for a log-solved one (positive by construction there, so a cap would only throttle);
      `coupled_amg_continuation` wires it automatically.
    - **BOTH strategies carry `step_limit` / `step_projection` (fixed 2026-08-15).** They were
      `DualTimeStep` fields only, so choosing `PseudoTransientStep` **silently gave up the guard** —
      and the guard exists because its absence is a recorded march death (two cells of 23040 took `k`
      negative and NaN'd the whole residual through a bare `sqrt`, every field still finite, nothing in
      the ordinary stopping tests able to see it). The guard protects the **state**, not the march: a
      field that must stay positive must stay positive whichever strategy steps it. `PseudoTransientStep`
      now applies the identical pair in the identical order — project per entry first, then read the cap,
      which then finds nothing binding — so the two compose the same way on both. Both default `None`,
      which is the unconstrained step exactly, so the default path is unchanged.
    - **The cap is GLOBAL, so one entry near zero throttles the whole step** — a real risk on a field
      spanning `1e-5` to `4.5`. Measured, it does not bite, because the escape is not the cap: the cap
      forces `alpha → 0`, `CflResidualDualTimeControl` reads that as "shift too weak" and escalates
      `beta` `0.5 → 1.0 → 2.0`, and at the larger shift the implicit step is short enough to fit inside
      the constraint — full `alpha = 1`, residual descending. **The two mechanisms compose without
      having been designed together**, and the march then converged in 20 steps where two previous runs
      had died. Do not "fix" the throttling without re-measuring; the globalization already handles it.
    - Why not a log variable for `k`: `k = 0` is the physical no-slip wall condition, so `log k` is
      singular exactly where the mesh is finest (recorded and measured in `.claude/rules/turbulence.md`
      — the full-log form descends then freezes). `log(k+1)` is regular there but bounds `k > −1`, not
      `k > 0`, so it does **not** prevent this failure. `k = w²` would give both properties at the cost
      of a vanishing Jacobian scale at the wall.
  - **`StateCheckpointer` (`solve/checkpoint.py`) — rolling state checkpoints (BUILT).** A march that
    raises at its last step loses everything: the exception propagates out, so a driver saving only on a
    successful return has nothing for the hours before. It cost one `bfs3d` run its two converged rungs.
    Plugs into the same `on_checkpoint` seam, `every` steps, keeping `keep` files. Two properties that
    matter for a job that may be killed: it **writes to a staging name and renames** (a kill mid-write
    would otherwise leave a truncated file that still reads as a checkpoint — and with a small `keep`,
    the only one left), and **retention deletes only paths it wrote**, tracked in a deque rather than by
    globbing, so it cannot remove another run's state. The default serializer hands `numpy` a *file
    object*, because `np.savez` appends `.npz` to any path lacking it and would otherwise silently write
    somewhere other than asked. `combine_observers` fans the single `on_checkpoint` out to logger +
    checkpointer so a driver does not hand-roll a lambda one of them can be dropped from.
    **Known defect, not yet fixed:** the checkpointer writes whatever the march reports *including the
    failed step*, so the newest file can be the poisoned state — and a driver calling it "last good
    state" is then lying. Skip a non-finite report, or do not claim "good".
  - **`on_retry(reason, attempt, beta)` — say WHY a step is being redone (BUILT).** `forward_march`
  calls it immediately before a redo with `"diverged"`, `"cycles"` or `"alpha"` — the three
  `RetryPolicy.escalation_reason` returns, all cured by escalating β — or `"solver"` for the tight-Krylov
  divergence retry. Without it a log shows the same step's work two or three times with nothing between
  the blocks, and the four reasons call for completely different responses. `MarchLogger.on_retry`
  writes the explanation between the abandoned attempt's block and the retry's, and numbers the attempt.
    - **`beta` is the shift the RETRIED attempt will run at — escalated on the three escalation
      reasons, unchanged on `"solver"` (binding, fixed 2026-08-14).** The call therefore sits *after*
      `escalated = beta * retry.beta_factor` is formed and before it is written onto the step. It used
      to fire before the escalation and hand over the abandoned attempt's β, which left `MarchLogger`
      reconstructing the real one as **`beta * 2`** — `retry.beta_factor`'s *default*, hardcoded — so a
      march configured with any other factor logged a shift it never ran at, and the `"solver"` path,
      where β is not touched at all, logged an escalation that never happened. **The trap generalizes:
      a callback that reports a value the caller is about to change makes its consumer re-derive the
      change, and a consumer re-deriving a caller's arithmetic will encode a default as a constant.**
      Pinned by `test_on_retry_reports_the_beta_the_retried_attempt_will_run_at` (at a *non-default*
      factor, since 2 cannot catch this) and two `test_march_log.py` tests.
  - **A self-rescaling measure means two "same" residuals are NOT the same number (binding trap).**
    `forward_march(norm_builder=…)` re-derives the `RowScaledNorm` at the state each outer iteration
    *begins from* and holds it for that whole iteration. So the `R` reported at the end of step N and
    the `‖G‖` entering step N+1 measure the **identical state** in **different scales**. Measured over
    one 62-step `bfs3d` march: they differed on **every** step — up to 2× early on, converging to 1 as
    the state settled and the scales stopped moving (the convergence-to-1 is the signature that
    identifies rescaling rather than a state difference; a rung boundary shows a ~2.8e5 jump, since the
    Reynolds number changed too). Nothing is wrong here, but **never compare two residuals across an
    outer-iteration boundary** and never print them adjacent — the march log deliberately dropped a
    "from |R|=…" field from its inner-block title for exactly this reason.
  - **The step grid stays NARROW; only scan-down quantities get columns (binding).** The step table is
    `step, t(s), beta, in, cyc, R, a_min, flg` — fixed, ~61 characters, comparable to the nested inner
    table so the two read as one document. Everything else — the case metrics, the preconditioner
    branch, the cumulative cycles, the per-field changes — rides in **spanning rows beneath the row it
    belongs to**. The first version put them all in columns and reached 112 characters with
    case-dependent column names; at that width, with the heading several rows up, it stopped being a
    table and became a line of numbers that could not be paired with their labels. **Do not widen the
    grid to add a quantity** — add an aside line, or a column only if it is worth scanning down the
    whole run. A row that follows an inner block is re-headed **compactly** (the label row alone, no
    rules): labelling one row must not cost three lines.
  - **The step summary is a table row, and the stopping test is stated once (BUILT).** The step line had
    grown to ~20 free-text `key=value` fields and ~150 characters — unreadable in a tail, and half of it
    was `‖R₀‖` and the target repeated on every row when both are **constants within a rung**. They now
    ride in a banner emitted once (and again whenever a continuation rung re-bases `‖R₀‖`), and the
    per-step values are a `TextTable` row so a quantity can be scanned *down* the run. Headings re-emit
    every `HEADINGS_EVERY` rows (fewer when the inner table is on, since its blocks push the last
    headings further up-screen), and the inner block is **indented** under the step row it belongs to.
  - **No column heading may contain `|` (binding).** `R`, not `‖R‖`; `G in`, not `‖G‖ in`. A heading
    carrying the grid delimiter makes the log unparseable by any column-splitting tool — not
    hypothetical: it forced regex workarounds in the analysis scripts written against the first version
    of this format. Say what the quantity is in the banner or the docstring, not in a heading that
    breaks the grid.
  - **`a_min` (step) vs `alpha` (inner) — different quantities, deliberately named apart.** An inner
    iteration's `alpha` is its own backtracking factor, and it reads **1.0 even when the line search
    failed to descend** (the non-descent fallback returns the longest finite trial step). The step
    reports the **minimum over its iterations with any non-descending one folded in as 0**. So
    `a_min=0.000` beside `alpha=1.000` is a real state — a full step that did not reduce `‖G‖` — not a
    contradiction. In the inner table **`rate ≥ 1` identifies those iterations exactly**, since the
    search is monotone.
  - **Diagnostics are opt-in through one `detail` argument (BUILT).** `MarchLogger(detail=…)` selects
    from `{"inner", "fields", "residuals", "pc"}`; empty (the default) is the plain one-row-per-step log. They are
    debugging/profiling instruments — several lines per step — not something a routine run should pay
    for. **Every hook stays safe to wire regardless** and no-ops when its name is absent, so a driver
    connects the instrumentation once and switches verbosity with one argument rather than by rewiring
    (which is how a driver ends up hand-rolling conditional plumbing). An **unknown name raises**: a
    silently-ignored typo means losing a diagnostic you believed was enabled.
    - `"fields"` — `MarchLogger(fields=…)` takes a `state -> named arrays` extractor
      (`turbulence.coupled_fields`) and reports each field's relative change per step via
      `field_change_metrics`. **The residual says the equations are unsatisfied; this says whether the
      *solution* still moves** — which a scalar case metric cannot: a mesh-quantized quantity like a
      reattachment length can sit perfectly still while the fields behind it drift several per cent, so
      "the metric stopped moving" is partly a statement about the instrument's resolution. The logger
      builds the (stateful) change measure itself so `detail` alone decides whether it runs — it must be
      called once per step in order, and a caller passing it directly would invite calling it from a
      probe and corrupting the sequence. `field_change_metrics` keys its output by the **field's own
      name** (not a decorated `d<name>/<name>`), so it joins against the per-equation residual below;
      how a quantity is *labelled* is the report's business, not the measure's. `coupled_fields` returns
      pressure **gauge-free** (mean removed; incompressible `p` is defined up to a constant unless a
      boundary pins it), **splits velocity per component** (`u`/`v`/`w`, so each lines up with its own
      momentum equation — a single vector entry averages them and hides a component that has stopped
      moving), and includes `ν_t`, which is derived rather than solved but is what the momentum
      equations actually see.
    - `"residuals"` — `MarchLogger(residuals=…)` takes a `state -> {equation: residual}` extractor
      (`turbulence.coupled_residuals`) and reports **the per-equation residual and its step-on-step
      contraction `rate`** beside each field's relative change, in one grid under the step row. **The
      scalar residual says the solve stopped improving; only this says which equation stopped it.**
      The numbers are `RowScaledNorm.per_block` — the very per-block values the march's scalar measure
      is the Euclidean combination of (`__call__` is now literally `norm(per_block(r))`, one formula),
      so a row near the total *owns* the residual. Names come from `coupled_equation_names(dim)` —
      `(u, v, w, p, k, omega)`, the flat layout's block order — the single home shared with
      `coupled_fields`, so the two grids join. Costs **one extra residual evaluation per logged step**,
      which is why it is opt-in. **The rows add up to the `R` printed above them, exactly** — pinned at
      `rel=1e-12`. That requires equilibrating at the **previous** state, not the logged one:
      `forward_march` re-derives the measure at the state each outer iteration *starts* from and holds
      it for the whole iteration (every trial step, the acceptance test, and the reported norm), so a
      step's residual is `norm_at_start(R(state_at_end))`. Scaling at the end state measures the right
      residual vector in the *wrong* scales and the rows stop adding up. `coupled_residuals` is
      therefore **stateful and order-dependent** — the same once-per-step-in-order contract
      `field_change_metrics` already carries — and takes a `reference_state` seed for the first step,
      which a per-rung reporter must pass (its rung's `seed_state`) or that step alone is scaled at its
      own end state. `ν_t` has no equation, so it gets a row with `--` in the residual cells rather than
      an invented number.
    - `"pc"` — `amg_beta_tracking_refresh(observer=…)` reports which branch a refresh took (`full` /
      `shift` / `none`) and its wall time; `MarchLogger.on_refresh` renders it on the step it preceded.
      **This closes a real gap:** which branch ran was previously invisible, so preconditioner behaviour
      had to be inferred from wall-clock — and was, wrongly, with an occasional expensive re-materialize
      read as a fixed per-step overhead.
    Pinned by `tests/unit/test_march_log.py`, whose assertions read **cells** rather than substrings, so
    a column reordering is not a false failure while a wrong value still is.

  - **`precondition_step` — per-step refresh of the step's frozen host preconditioner (binding,
    forward-only).** `forward_march(precondition_step=…)` calls `precondition_step(active_step, state)`
    before each `_march_step`, *after* the control has set β on `active_step`, to re-derive the step's
    **static** host preconditioner from the current `(state, β)`. It runs in the eager loop (a host op
    outside the jitted step) and mutates the preconditioner in place, so `_march_step` stays a
    compilation-cache hit. Two consumers (`.claude/rules/turbulence.md`), sharing one
    `_beta_tracking_refresh` skeleton: `lu_beta_tracking_refresh` re-factors the complete LU at the current
    `(state, β)` **every step** (cheap + exact → 1 Krylov iter), the fix for the frozen-LU β-mismatch above;
    `ilut_beta_tracking_refresh` does the same for the ILUT but **gated** (β-move OR staleness cap), because
    the ILUT refactor is expensive (~30–40 s) and approximate — the β-move trigger is what averts the α=0 /
    no-drift stall a *drift* trigger would hit on an overshoot. Distinct from the trigger's `refresh_builder`
    (which fires occasionally, restarts a *segment*, and returns a *new* step): this fires every step (the
    consumer may itself no-op) and mutates in place. Forward-only (impure), folded into the same `observing`
    gate and `jax.grad` guard as the trigger/control; `None` is byte-identical to before.
  - **`StepControl` — stateful, feedback-driven step reshaping on the eager march only (binding — the
    twin of `RelaxationSchedule`, deliberately NOT one interface).** A `RelaxationSchedule` is memoryless
    and lives on the differentiable step; a `StepControl` reads the *previous* `StepReport` (α, cost) —
    feedback available only after a step — to reshape the next step, and may raise under `jax.grad`, so it
    lives here beside `RefreshTrigger`, never on the traced path. `forward_march(step_control=…)` calls
    `next_step(base, previous, state) -> (ForwardStep, new_state)`, threading the control's own state; the
    march stays β-ignorant (the control returns a ready-to-run step, typically `base` with a
    `ConstantRelaxation` β leaf via `tree_at`, so `_march_step` stays a cache hit). **The control state
    survives across preconditioner refreshes (issue #156):** `forward_march` takes an incoming
    `control_state` and returns the final one on `MarchResult`, and `solve_coupled` threads it from each
    segment into the next — so a control that climbs β over many steps continues past a refresh instead of
    resetting to `beta_start` at every segment (the α-controller and the refresh were co-designed but
    could not compose before this). This is the *global*-lifetime carry, deliberately opposite the
    segment-local SER `residual_norm_0` and `drift_measure`. Unifying the two into
    one `(rn, rn0, α, state) -> (β, state)` interface was rejected: it would union SER's needs with the
    control's (dead α/state args for SER), drag α onto the differentiable core where the line search
    cannot even produce it before the step, and risk the byte-identity of the default path. Four concrete
    `StepControl`s live in `solve/step_control.py`, all three sharing one body: **`DualTimeControl`** (the
    Courant β-ramp, the **default** for a dual-time observed march — carries β across refreshes, see the
    DualTimeStep bullet above), **`ResidualRatioDualTimeControl`** (the opt-in residual-keyed
    alternative), and **`CflResidualDualTimeControl`** (see the bullet below). There is **no
    `AlphaTargetingControl`** — deleted 2026-08-14, see the "SER β schedule runs backwards" bullet.
    - **`ShiftStrengthControl` is the shared base, and a subclass supplies ONLY `_adapt` (binding).** It
      owns the first-step seed, the hold-across-a-refresh rule, the `[beta_min, beta_max]` clamp, the
      `tree_at` swap of a `ConstantRelaxation` onto a **dynamic** β leaf (the compilation-cache-hit
      property), and `carry_beta`. The carried state is **`(beta, memo)`** for every control — β is
      universal and `memo` is whatever the rule remembers (`None` for the memoryless Courant rule, the
      previous residual for the two ratio rules). **Why it exists:** written three times, the bookkeeping
      drifted — `carry_beta` was byte-identical in two controls and *absent* from the third, which
      `forward_march` probes for with `hasattr`, so that control silently dropped its escalation
      feedback; and the same class reset β at a refresh boundary where the others held it, i.e. the
      sawtooth defect fixed for `DualTimeControl` never reached it. Both were invisible because each
      class carried its own `next_step`. The refactor is verified **bit-for-bit** against the previous
      implementation over 3096 β transitions spanning all three bands, rising/falling/flat residuals, the
      clamps and both boundary paths. `CflResidualDualTimeControl` reduces exactly to `DualTimeControl`
      at infinite ratio thresholds, which is now **pinned by a test** rather than asserted in prose.
  - **`CflResidualDualTimeControl` — the combined control, grows on α but brakes on a rising residual
    (built for the 3D inexact-AMG march the α-only control NaNs).** The two single-signal dual-time
    controls fail in **disjoint** ways: `DualTimeControl` grows Δτ on the inner-loop factor α (fast) but
    is **blind to the trajectory** — it grows into an overshoot while α = 1, which diverges unless the
    linear solve is near-exact (measured: on the 3D `bfs3d` Re-continuation the α-control runs `x_r/h`
    away to 15–19 and NaNs, because the AMG V-cycle is inexact — the same "aggressive control's overshoot
    is tolerated only by a near-exact solve" rule the 2D complete-LU satisfies); `ResidualRatioDualTimeControl`
    keys growth on the steady-residual ratio (safe against overshoot) but is **blind to productive
    development** — on the `β × travel` plateau the residual is flat, so it pins β and crawls (measured:
    the 3D march *survives* the overshoot with it but takes 4 h against OpenFOAM's ~15 min). The combined
    control grows **only when both signals are comfortable** (α ≥ `grow_above` **and** the residual ratio
    ≤ `hold_ratio`) and brakes on **either** wall (α < `backoff_below` **or** ratio > `rise_ratio`), with a
    hold band between the two ratio thresholds so a noisy plateau does not oscillate — so it grows on α
    through the flat-residual development (the residual-only rule's stall) yet the residual-rise term brakes
    the overshoot the α-only rule is blind to. This is the "pair α with a step-productivity signal" lever the
    single-step α-targeter's non-convergence ceiling flagged, applied to the dual-time controls. State is
    `(β, previous residual)` — the shared `(beta, memo)` carry — across refreshes like
    `ResidualRatioDualTimeControl`; opt-in via
    `solve_coupled(step_control=…)`. Unit-tested in `test_step_control.py` (grows on α at a flat residual,
    brakes on a rising residual at α = 1, brakes on an inner clip, holds in the band, carries β). The ratio
    thresholds are march-calibrated numbers — set them from a logged march, not intuition (the 3D
    development overshoot shows ratios ~1.14).

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
  LSC for Rhie–Chow; PCD needs FEM-BC re-derivation).
  - **`solve/multigrid.py` is a pure operator-coarsening library — operator-in, uniformly (binding).**
    **Every** builder takes an assembled `a: sp.csr_matrix` — `build_smoothed_hierarchy(a)`,
    `build_convection_hierarchy(a)`, `build_air_hierarchy(a)` — and none takes a mesh, edge arrays, or
    flow quantities. Assembly lives beside it in `aquaflux/solve/frozen_operator.py`
    (numpy + scipy only — no mesh, no field, no `jax`):
    `convection_diffusion_operator(owner, nb, coefficient, n, *, flux=None, boundary_diagonal=None)` —
    symmetric graph Laplacian when `flux is None`, first-order-upwind convection-diffusion otherwise —
    plus `decouple_dof(a, index)` for the closed-domain pressure pin. It is the **one** assembler for
    all four consumers (pressure Schur, viscous velocity block, convection velocity block, k/ω scalar
    transport). Do not reintroduce a `(owner, nb, coeff, …)` signature into `multigrid.py`, and do not
    add a second stencil assembler — the old `_laplacian_csr` was exactly `convection_diffusion_operator`
    at `flux=None` and was deleted. `build_convection_air_hierarchy` was likewise **deleted**: once it
    took `a` it was a pure alias for `build_air_hierarchy`.
    *Why it is a solver concern, not a flow one:* the first-order-upwind stencil is the
    **preconditioner's** choice, independent of what the residual discretizes advection with — it is
    chosen to give a diagonally dominant M-matrix an aggregation hierarchy can coarsen. Its parameters
    are a weighted graph (`coefficient`, `flux`), not flow quantities. Keeping it in `solve/` also adds
    no new dependency edge: every consumer already imports `solve.multigrid`.
  - **The V-cycle recursion AND its outer fixed-cycle driver are single-homed (binding, #52).** A
    family (`smoothed_multigrid_solve`, `convection_multigrid_solve`, `air_multigrid_solve`) contributes
    **only** its `_VCycleOps` — restriction, prolongation, smoother. The recursion is `_frozen_v_cycle`
    and the outer loop (zero initial guess, `cycles` residual-correction passes,
    `x += _frozen_v_cycle(levels, b - A x, …)`) is `_fixed_cycle_solve`; both are written once. That
    outer loop is what makes `b -> x` a constant linear operator — the property the frozen-left-PC and
    the adjoint transpose depend on — so it must not be re-typed per family where one copy could drift.
    A new family adds a `_VCycleOps` builder and a thin entry point that calls `_fixed_cycle_solve`; do
    **not** re-write the cycle loop in it.
  - **A level is STATIC indices + TRACED values, so a hierarchy refresh is a jit cache hit (binding).**
    `_SparseLevel` / `_AirLevel` are `equinox.Module`s in which **only `n` and `n_coarse` are static** —
    they size the sparse matvec output (`_coo_apply`'s `n_out`), so they must be concrete. Everything
    else (`val`, `diagonal`, `coarse_inv`, the prolongation/restriction values, and **`lam_max`**) is a
    traced leaf. `lam_max` is deliberately a **0-d array, not a Python `float`**: it is only smoother
    arithmetic, and as a static field any refreshed value would be a new compilation-cache key —
    defeating the point. Consequence: a hierarchy passed as a **jit argument** survives a refresh as a
    *cache hit* (one compiled V-cycle), which is what lets a frozen preconditioner track a developing
    flow without paying a recompile per refresh (the ~2.6× scalar-AMG staleness win above). This is only
    sound because **the aggregation coarsening is a pure function of the graph at the default
    `strength_threshold=0`** — `_aggregate` reads `owner`/`nb`/`n` and never the coefficients — so on a
    fixed mesh a hierarchy re-derived at a new operator has identical aggregates, coarse sizes and array
    shapes, and only values differ (this is why the refreshed *scalar k/ω* AMGs keep `strength_threshold=0`
    — the flow block is frozen, so it can afford the value-dependent SoC below; the scalars cannot without
    a value-refresh). Both properties are pinned in `tests/unit/test_multigrid.py`
    (`test_aggregation_hierarchy_structure_is_value_independent`,
    `test_refreshing_a_hierarchy_is_a_compilation_cache_hit`). **Caveat — lAIR does NOT get this for
    free, and this is measured, not hypothetical.** Its C/F split comes from `_strength_classical`, which
    thresholds on `|A_ij|`, so re-deriving a reduction hierarchy at a new operator changes the split and
    the shapes: on a 600-cell chain, cold vs developed coefficients (5000× flux, 1000× viscosity ramp)
    gave **identical L0–L2 but divergent L3–L5** (`n_coarse` 37→38, then `n` 37→38 / `nnz` 109→112,
    18→19 / 52→55), i.e. a different jit signature and therefore a recompile anyway. The aggregation
    path was invariant at *every* level in the same comparison. **Consequence:** for `method="air"` (and
    `velocity="convection-air"`) a cheap refresh requires **reusing the reference's frozen C/F split and
    prolongation and recomputing only the values on it** — legitimate, since any valid split gives a
    valid preconditioner. That is **`refresh_air_hierarchy(hierarchy, a_new, degree=…)`** (below),
    whereas the aggregation path gets it for free by rebuilding.
    - **Strength-of-connection (SoC) aggregation IS available now — opt-in via `strength_threshold`,
      and it makes the aggregation path value-dependent (so it forfeits the free refresh above).**
      `build_smoothed_hierarchy(a, strength_threshold=θ)` / `build_convection_hierarchy(a, …)` (and
      `BlockPreconditioner.build(…, strength_threshold=θ)`) aggregate on the **strong** subgraph only —
      edge `(i,j)` kept iff `|A_ij| ≥ θ·max_k|A_ik|` from either endpoint (`_aggregation_edges`, reusing
      `_strength_classical`, symmetrized) — instead of the full graph. **Why it exists:** isotropic
      aggregation coarsens *across* the stiff direction of a high-aspect-ratio / skewed operator and the
      V-cycle stalls; SoC coarsens *along* the strong coupling and keeps contracting. Measured on a
      uniformly-anisotropic Poisson (cycles to true 1%): plain `>20` and rising toward failure as the
      aspect ratio grows (contraction 0.22→0.56→0.90→0.97 at AR 1→10→100→1000), **SoC a flat 3 cycles at
      every AR, mesh-independent** — the difference between a converging and a non-converging solve at
      AR≥100 (`test_multigrid.py::test_strength_of_connection_aggregation_fixes_an_anisotropic_operator`).
      **`θ=0` (default) is byte-identical to the historical isotropic build.** It is turned **on (θ=0.25)
      for the coupled flow block** (`_coupled_shift_policy`: the convection velocity AMG + MSIMPLER Schur),
      which is frozen at the reference state so the value-dependence costs no refresh; it is a **no-op on
      the low-aspect-ratio pitzDaily case**, and the payoff is the future wall-resolved / skewed regime.
      It does **not** apply to the reduction-based `air`/`lsc` blocks (already strength-based), and the
      refreshed scalar k/ω AMGs stay `θ=0` to keep their refresh cache-hit — a value-refresh (à la
      `refresh_air_hierarchy`) to let them use SoC too is the tracked follow-up.
  - **`refresh_air_hierarchy` — the lAIR refresh that keeps the compilation signature (BUILT).** It
    re-derives an lAIR hierarchy's **values** at a new operator while holding the coarsening fixed: each
    level reuses its stored C/F split (recovered from the level's own masks) and its stored
    prolongation, and re-solves only the local approximate-ideal restriction against the new `a`. The
    prolongation must be *carried over*, not re-derived, because `_one_point_interpolation` picks each
    F-point's strongest C-neighbour by `argmax |a_ij|` — a value-dependent column choice. The
    restriction's sparsity, by contrast, depends only on the split and on `a`'s *pattern* (the
    degree-`d` neighbourhood walk), so it is invariant, and the Galerkin `R A P` patterns below follow
    inductively. The result is verified before returning (`_require_matching_structure`) and a
    mismatched operator raises rather than silently returning a hierarchy that would recompile. Pinned
    by `test_refresh_air_hierarchy_keeps_the_structure_and_is_a_cache_hit` (shapes preserved, values
    changed, jitted V-cycle traces once), `test_refreshed_air_hierarchy_preconditions_the_new_operator`
    (a refreshed cycle beats the stale one on the new operator, so the reused split is a real trade and
    not a no-op), and `test_refresh_air_hierarchy_rejects_a_mismatched_operator`. **Why it matters:**
    measured on the separated pitzDaily state with the production lAIR scalars, refreshing the k/ω AMGs
    is worth ~2.4× in outer cycles (30 → 13 at β=2; the flow block 30 → 29, i.e. nothing), and this is
    the only way to take that win without paying a recompile per refresh.
  - **Degenerate-mesh guard (binding — validated where the graph is consumed).** Because the
    hierarchies are built once off-jit and then frozen, a degenerate mesh must fail *there*, not as a
    silently stalling runtime V-cycle. Now that the builders are operator-in, the **graph** check lives
    with the assembler: `frozen_operator._require_valid_graph` (`n ≥ 1`, matched `owner`/`nb`, in-range
    endpoints) runs inside `convection_diffusion_operator`; the two build loops
    (`_build_aggregation_hierarchy` for smoothed/convection, `build_air_hierarchy` for lAIR) call
    `_require_positive_diagonal` on **every** level's operator diagonal before inverting/freezing it,
    so a zero diagonal (disconnected component, isolated/zero-volume cell, degenerate `R A P` row)
    raises `ValueError` at setup instead of baking `inf` into the frozen operator. The diagonal is
    checked *after* boundary stiffness is folded into the operator, so a boundary-only cell is
    correctly allowed. This one build-time guard is why the runtime smoothers (`_chebyshev_smooth`,
    `_jacobi_smooth`, `_fc_jacobi`) and the block-preconditioner rescales, which invert the frozen
    diagonal / the positive momentum `a_P`, need no per-apply floor.
  - **The damped-Jacobi convection hierarchy is TWO-LEVEL by design (binding — do not add a depth
    knob).** `build_convection_hierarchy(a)` builds exactly a smoothed fine level + a single **direct**
    (dense inverse) coarse solve. **`max_levels` exists** (`multigrid.py`, default
    `_CONVECTION_LEVELS = 2`); raising it re-opens the defect below, so leave it at 2. On the fine level the
    upwind operator is a diagonally dominant M-matrix, so one damping factor `ω/λ_max` contracts
    (`_jacobi_smooth`, ρ ≈ 0.7 at high cell Peclet). A *deeper* Galerkin recursion is deliberately not
    built: a coarse-of-coarse operator of a strongly convection-dominated problem acquires
    near-imaginary-axis eigenvalues that **no single-factor damped-Jacobi smoother can damp** — the
    smoother becomes non-contractive (measured ρ(S) ≈ 1.0–1.36 on such levels), so the coarse level
    must be an exact solve. Deep, mesh-independent convection coarsening is the job of the
    reduction-based lAIR hierarchy (`build_air_hierarchy` + `_fc_jacobi`) instead. Both
    production callers (the flow `SmoothedAmgConvectionVelocity` two-level path and the turbulence
    preconditioner) already used two levels, so this is behaviour-neutral; the deep damped-Jacobi
    build it removed was dominated on both ends (worse than two-level shallow, worse than lAIR deep)
    and was the sole source of the non-contractive-smoother defect.
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
