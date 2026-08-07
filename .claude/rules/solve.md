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

## Status — BUILT (Stage A, linear)
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
  nonlinear residual (the flux limiter). `newton_step` is shared with `NewtonSolver`. Verified
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
- **Monolithic ILUT preconditioner — BUILT (`sparse_jacobian.py` + `ilut_preconditioner.py`).** An
  incomplete-LU (threshold ILU) factorization of the **assembled coupled Jacobian**, the alternative to
  the block-triangular SIMPLE preconditioner for the coupled saddle. The block PC approximates the
  pressure Schur; the ILUT **forms the true Schur coupling `B F⁻¹ G` through its fill** instead — measured
  on the coupled RANS saddle it reaches the forward tolerance in a handful of GMRES cycles where the block
  PC needs hundreds (the block PC's wall is the Schur *approximation*, not its inversion — see the Stage-3
  note in `.claude/rules/flow.md`). Three ingredients are each load-bearing and measured: **enough fill**
  (zero-fill ILU(0) drops exactly the Schur-forming fill → a singular factor; `drop_tol=1e-6` — not
  `fill_factor` — is the binding control, keeps it); **symmetric √-diagonal equilibration** (the momentum
  and continuity rows differ in scale by ~34×, which otherwise gives near-singular pivots); and
  **cell-major ordering** (interleave `[u,v,p,k,ω]` per cell so the indefinite saddle factors without a
  zero pressure pivot). The distance-1 *truncation* of the operator is catastrophic — the coupled saddle
  is intrinsically distance-2 (Rhie–Chow) and the fill is essential, so this is **not** a compact-operator
  play.
  - **`sparse_jacobian.py`** materializes the coupled Jacobian from the *same* residual the solver uses
    (no re-derived assembly): `block_stencil_colouring(owner, nb, n, reach)` (pure NumPy — the cell-block
    pattern at a stencil `reach` and a collision-free CPR colouring, the conflict graph is the pattern
    squared) then `materialize_block_jacobian(matvec, colouring, n_fields)` (one `jax.jvp` per
    colour×field — e.g. 112 colours × 6 fields = **~670 probes** on the 23k-cell reach-3 `bfs3d` mesh, NOT
    ~240). `jacobian_relative_error` guards that `reach` covers the stencil (coupled RANS reaches distance
    **3**). Field-major DOF layout `(cell i, field f) = f·n + i`.
    - **Materialize efficiency — two shipped speedups, both AMG-path-only, bit-identical (BUILT).** The
      probe dominates a refresh, so `materialize_block_jacobian` takes two optional accelerators the AMG
      preconditioner passes (LU/ILUT keep the plain loop, which any NumPy matvec supports). **(1) Batched
      probing** — `batched_matvec` (a `jax.vmap` of the jvp, **built once and reused** so it compiles a
      single time; `probe_batch_size` chunks it for memory) runs the coloured probes as a few fused passes
      instead of a Python loop of separate calls. Measured 22.4→14.0 s (~1.6×) on `bfs3d` — modest because
      CPU forward-AD does not vectorize across the batch like a GPU (the win is dispatch amortization). For
      the SAME reason the chunk is kept **small** (`_PROBE_BATCH_SIZE = 4`, not 16): a larger batch holds
      more simultaneous forward-AD tapes for almost no time gain — measured chunk 16 vs 4 was ~33 s vs ~36 s
      but ~2.2 GB vs ~0.7 GB of transient peak, and on a memory-bounded box (the refresh re-materializes
      every few steps) that peak is what tips a long march into swap/OOM. Four keeps the dispatch
      amortization at a third of the peak; the materialize itself frees cleanly (no cross-refresh growth).
      **(2) Gather de-compression** — `block_stencil_gather_map(colouring, n_fields)` precomputes, once, the
      **fixed full-pattern** CSR structure + a flat `gather_map`, so de-compression is one vectorized
      `data = responses.ravel()[gather_map]` (no scatter loop, no per-materialize CSR re-sort) passed as
      `structure=`. The **full** pattern (no `eliminate_zeros`) is the fixed mesh distance-3 colouring graph
      — a superset of the Jacobian's live nonzeros at *any* state — so the structure stays fixed
      cold→developed and no entry is ever dropped as values change: a guaranteed-fixed structure the in-place
      GAMG refactor needs. On `bfs3d` the full pattern is ~47.2M positions, but that is a **structural
      over-estimate**, mostly explicit zeros at every state — LIVE nnz is ~constant (**38.7M cold / 39.0M
      developed**; there is *no* cold→developed nnz collapse — an earlier "47.2M cold → 39.0M developed"
      reading conflated the fixed pattern with live nnz). The explicit zeros are harmless to aggregation —
      strength-of-connection ignores a zero coupling — but they are why an incomplete/complete factorization
      can't use this path: an explicit zero is fill. De-compression 22.4→11.2 s (2.0× vs
      loop); full build (materialize + GAMG) only 56.0→54.2 s (**1.03×** — GAMG-dominated), so the gather's
      real value is the fixed-structure invariant, not the wall-clock.
    - **Probe REACH is a preconditioner choice — reach-2 is NOT a safe drop-in (measured, SHELVED — a
      GENUINE failure, not a build artifact).** The materialized `J` is only the preconditioner's operator
      (the solve matvec is always the exact matrix-free jvp), so a lower `stencil_reach` gives an approximate
      PC. On the orthogonal `bfs3d` mesh reach-2 is numerically near-*exact* at *every* state
      (‖A2−A3‖_F/‖A3‖ = 6e-6 cold, ~1e-15 developed — the dropped distance-3 shell is negligible; swapping
      Corrected→Compact Green-Gauss leaves it bit-identical, so the non-orthogonal skew correction
      contributes ~0 here) and ~2.2× cheaper to probe (60 vs 112 colours, ~half the nnz). **Yet GAMG(reach-2)
      DIVERGES as a preconditioner** (true residual 1e3–1e8) at cold AND developed, on its own operator, with
      a verified-correct build — so it is genuine, not a build/scaling bug. **The cause is the ILU(1)
      smoother, whose fill is PATTERN-dependent (a symbolic/graph operation), not value-dependent:** halving
      the graph gives a structurally weaker incomplete factorization that is non-convergent on the indefinite
      saddle. The ≤6e-6 magnitude argument bounds only the value-based *aggregation* (consistent
      reach-2↔reach-3); it does not touch the smoother. Proof: padding reach-2's values onto the reach-3
      *positions* (distance-3 shell as ~1e-30 explicit zeros) recovers reach-3 convergence **bit-identically**
      (32 matvecs) — so the distance-3 *positions* are causal, their values are not. **This is why the full
      reach-3 pattern with explicit zeros is REQUIRED for smoother convergence (not merely a
      structure-invariance nicety), and why you must not lower the default reach.** Recovery is not cheap:
      `smoother_sweeps=3` / ILU(2) still diverge *and* erase the setup win; restoring the reach-3 pattern
      converges but forfeits the GAMG-setup half of the economy (the larger half — refresh is
      setup-dominated), leaving only the marginal probe-colour saving. The one open lever is a fundamentally
      different smoother that tolerates the sparser graph (a smoother-design task). On a genuinely SKEWED mesh
      reach-2 would additionally be *lossy* (the non-orthogonal ring pushes real content to distance-3) — a
      second reason to keep reach-3. See `bfs3d-doomed-primary-cost-fixes` in memory.
  - **`ilut_preconditioner.py` — `MonolithicIlutPreconditioner`.** Built off the jit path (`scipy.spilu`);
    a **host** object, so it is **not** an `equinox.Module` — it rides as a static field and is applied
    inside the jitted Krylov solve through `jax.pure_callback` (`.matvec()` / `.matvec(transpose=True)`).
    Frozen at a reference state+shift like the AMG blocks; being far stronger it tolerates the freezing at
    a few extra cycles, and the shift vanishes at the root so it never changes the converged state or its
    adjoint. `IlutFactors`/`factorize_ilut`/`cell_major_permutation` are the pure host core (testable
    without JAX); the JAX wrapper is thin.
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
    can NaN. `MonolithicIlutPreconditioner.refresh_in_place(matvec, colouring, n_fields, shift_diagonal,
    …)` re-materializes and re-factors at the developed state and swaps `self.factors` **in place**. Two
    facts make this a **compilation cache hit** rather than a recompile: the preconditioner is a *static*
    field of `MonolithicIlutShiftPolicy` (so its identity is the jit treedef, unchanged by mutating its
    factors), and `matvec()` reads `self.factors` **at callback time** (not captured), so the mutation is
    seen by the already-compiled solve. `build` and `refresh_in_place` share one form-and-factor path
    (`_factor`). **This is sound only because the forward march is NEVER differentiated** — the mutation
    is impure and would corrupt the adjoint's transpose solve (which reads the same `self.factors`), so
    it is forward-march only; the converged root and its adjoint are refresh-independent anyway (the shift
    vanishes at the root). Measured on pitzDaily: a rebuild-per-refresh is ~72 s (≈27 s of it the
    march-step recompile, ≈13 s the base-policy rebuild, ≈5 s a jvp recompile), the in-place refresh
    ~44 s. The residual ~31 s is the intrinsic materialize + factor, and it splits **materialize (240
    jvps) ~2.4 s / `spilu` ~17.8 s** — so `spilu` is ~88 % and a sparser (cheaper-materialize) stencil
    would save almost nothing. `spilu` is a hard floor: a *threshold* ILU's fill pattern is
    value-dependent, so the symbolic factorization cannot be frozen and re-used (and scipy exposes no
    symbolic/numeric split), leaving **amortization (refresh less often) as the only cheap lever**. The
    coupled driver wiring is `coupled_ilut_refreshing_continuation` (a `refresh_builder` for
    `solve_coupled` — see `.claude/rules/turbulence.md`); it pairs with a `CoefficientDriftTrigger` so the
    re-factor *leads* the staleness. Pinned by `test_refresh_in_place_repreconditions_the_same_compiled_matvec`
    (unit) and `test_ilut_refreshing_continuation_refreshes_the_same_step_in_place` (integration).
  - **Scope / follow-ups (MVP).** The heavy fill (~7–14× the operator's nonzeros) is affordable at 2D /
    moderate mesh sizes but is the weak point at large 3D — the **monolithic AMG V-cycle**
    (`amg_preconditioner.py`, below) is the built scaling path (its direct-LU coarse solve is what tames the
    naive monolithic V-cycle's coarse-grid-correction instability on the indefinite saddle). The coupled builder still assembles the
    unused block AMG as the `a_P` source — a lightweight shift-diagonal-only policy would remove that. The
    coupled integration (`coupled_ilut_continuation`) lives in `.claude/rules/turbulence.md`.
- **Monolithic COMPLETE-LU preconditioner — BUILT (`lu_preconditioner.py`), the preferred 2D/moderate
  coupled preconditioner.** The sibling of the ILUT: it factors the assembled coupled Jacobian
  *completely* (`MonolithicLuPreconditioner`), so it is the operator's **exact** inverse and a Krylov
  solve converges in **one** iteration. Measured on the developed pitzDaily coupled Jacobian (61k dof):
  UMFPACK factors it in **~1.2 s vs the ILUT's ~32 s (~26×)**, exact (1 GMRES iter vs 2–4), verified on
  the real forward operator and the β=0 adjoint (true-residual checked — see
  `reference/ILU_REFRESH_PROFILING.md`). Because the fill is pattern-determined it is also **state-robust**
  (no `drop_tol` tail that shifts with the flow). Same interface as the ILUT (`build` / `refresh_in_place`
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
    not stronger — always verify the TRUE residual. Full record: `reference/ILU_REFRESH_PROFILING.md`.
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
    `equilibrate_cell_major` (extracted from `factorize_ilut` into `ilut_preconditioner.py`, now the one
    home for the sqrt-diagonal equilibration + cell-major reorder both the ILUT and the V-cycle need); the
    `Mat` block size is `n_fields` so GAMG aggregates cell-blocks. Host object, applied via `pure_callback`,
    riding as a static field; `build`/`refresh_in_place`/`matvec` identical to the ILUT/LU, so it plugs into
    `MonolithicFactorShiftPolicy` unchanged. **It is the one family member that needs `petsc4py`** (no
    pure-SciPy AMG fallback); the module lazily imports PETSc and raises a clear install hint otherwise.
  - **The smoother is the research variable, and the measured MVP config is a STATIONARY ILU(1) level
    smoother (`richardson`) + direct-LU coarse.** Measured on the `bfs3d` shifted coupled Jacobian
    (true 2-norm residual, `KSP_NORM_UNPRECONDITIONED`, at 2 sweeps): plain GMRES + stationary **ILU(1)**
    reaches **1e-8 in 21 iterations**; **ILU(0) stalls** ~2e-4 and **SOR diverges**. A Krylov-accelerated
    (GMRES) smoother is a few iterations *faster* (12–18 via FGMRES) but makes the V-cycle **nonlinear** — it
    needs flexible GMRES and has no clean transpose, so it is a deferred forward-only optimization, **not** the
    adjoint path. The MVP forward solver is `_COUPLED_AMG_FORWARD_SOLVER` (restart-15, vs the ILUT's
    restart-10: the V-cycle is a weaker approximate inverse so the loose inexact-Newton solve needs a couple
    dozen vectors, not a handful).
  - **Per-step cost tuning (measured): `smoother_sweeps=2` default and the forward restart 15 (from 40).**
    The restart-15 forward loop stops as soon as the ~1% inexact-Newton tolerance is met instead of running
    out a 40-vector subspace (the dominant per-step saving). The **smoother-sweeps knob is the second lever,
    and more is better on this saddle**: the outer Krylov cost is governed by the *smoother work* per V-cycle,
    and adding a second incomplete-LU Richardson sweep — one extra cheap triangular back-solve — roughly
    quarters the outer iteration count on the low-shift operator the march's tail runs at (measured on the
    `bfs3d` coupled Jacobian to a 1% stop: 211→54 outer cycles at a low shift, ~2.1× the whole solve there;
    ~10% at a high shift, where the operator is already diagonally dominant). Each outer iteration pays a full
    Jacobian-vector product (and, on the JAX-side `lineax` path, a `pure_callback` into PETSc), so trading one
    cheap extra sweep for far fewer outer iterations is a large net win — `sweeps=2` is the sweet spot
    (`sweeps=3` helps a little more at low shift but costs at high shift). Adding *fill* to the smoother
    (`smoother_fill_levels`) instead would cut iterations too, but it is the expensive incomplete-factorization
    build the ILUT hits in three dimensions; sweeps add smoother work without that build cost, and the
    coarsening choice (selective vs smoothed-aggregation) is a minor knob by comparison. The `bfs3d` coupled
    solve reaches ~24–30 min total against OpenFOAM's ~15 min. An **experimental, opt-in native-PETSc forward path**
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
  - **⚠️ TRUE-RESIDUAL TRAP (binding).** PETSc's default convergence norm is the *preconditioned* residual
    `‖Mr‖`; on this indefinite saddle SOR/Krylov-smoothing report `reason=2` (converged) at a **true**
    residual of 1.0. Force `KSP_NORM_UNPRECONDITIONED` and always check `‖Ax−b‖` — same lesson as the
    level-ILU artifact in [[ilu-refresh-cost-levers]].
  - **Coupled builder `coupled_amg_continuation`** (`.claude/rules/turbulence.md`) shares
    `MonolithicFactorShiftPolicy` + `_monolithic_factor_step` with the ILUT/LU. Verified: converges to the
    block PC's fixed point AND passes the **coupled-adjoint FD gate** (the transpose V-cycle serves the
    gradient), `tests/integration/test_coupled_amg.py`; V-cycle mechanics in `tests/unit/test_amg_preconditioner.py`.
    Follow-ups: a refreshing/β-tracking variant (the frozen build serves the forward + adjoint; a developing
    3D march would want the refresh), and the FGMRES forward optimization.
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
    - **A non-uniform shift creates INTERIOR optima that the backtracking ladder cannot find — so a
      local-Δt basis and an interpolating line search are coupled design choices.** With `w = 1` the
      ideal step length was exactly `α = 1` at every β, so the powers-of-½ quantization cost nothing.
      With `w = 0` at β = 0.5 the ideal was `α = 0.658` (+2.26 %) while the ladder took `α = 1` (+1.76 %)
      — **22 % of the available reduction unclaimed**, because `backtracking_line_search` accepts the
      *first* rung that reduces and never asks whether a shorter step is better (a sufficient-decrease
      search, not a minimizing one). The loss grows as the optimum moves further off a rung: on an
      over-damped log-space ω shift the ideal was `α = 0.285` (+0.307 %) while the ladder took the
      neighbouring rung `α = 0.5` (+0.081 %) — **74 % unclaimed**. Note the directional derivative is available almost free here, since
      the shifted solve gives `J δ = −R − β D δ` exactly, so a quadratic/cubic backtrack is cheap; and a
      residual evaluation is ~8 ms against a ~40 s solve, so a finer search is ~0.1 % of a step.
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
        — measured 2026-07-27 (`scratchpad/nut_floor_probe.py`, `re_continuation_probe.py`).** A single
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
    - **Neither α nor the cycle count can serve as a controller target on this problem.** Across the
      whole sweep above — two bases, a 12× span in β — **α is 1.0000 at every single point**, and the
      cycle count is flat at 14 through `a_P`'s entire productive range. Both are constant where the
      efficiency varies 28×. The only quantity that discriminates is **residual reduction per unit
      time**, which is what any Courant/β controller would have to estimate.
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
    - **Staleness is still real but was NOT the main term here, and the old figures stand.** The
      "36 s at β=2 / 127 s at β=0.2" numbers elsewhere in this file predate the bug and remain valid.
      Rebuild-vs-carry belongs in the refresh-trigger calibration (#17) on a *cold-IC* march; re-measure
      it now that the solves converge, since the pre-fix carried-vs-rebuilt comparison (#31) was taken
      through the broken preconditioner and cannot be trusted.
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
      (`scratchpad/measure_is_travel.py`), actual ÷ predicted `βDδ` floor:

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
    - **PROTOTYPE VALIDATED (2026-07-27, `scratchpad/pseudotime/dualtime.py`) — the diagnosis holds, and
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
      - **CAVEAT — dual-time alone does NOT accelerate reachability.** Development rate is Δτ-governed, and
        β = 0.5 is still a small Δτ (same crawl). Its contribution is (a) an honest `‖R(φⁿ)‖` that can
        *drive* a Δτ ramp (single-step's stalling measure is why SER ran backwards) and (b) the
        inner-line-search-on-`G` tolerating a larger Δτ than one shifted step. Reachability still needs the
        Δτ ramp **and** the cold-start diffusion/Re continuation (they compose: dual-time is the honest
        gauge + robust per-step solve, continuation lowers the cold stiffness so Δτ can grow early).
      - **CFL-ramp A/B (2026-07-27, `scratchpad/pseudotime/dualtime_march.py`) — the hypothesis holds, the
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
        when it clips), a `StepControl` on the eager march, sibling to `AlphaTargetingControl`. The step's
        reported α is the **min** inner line-search factor, and an inner step that fails to reduce ‖G‖
        (the line search's non-descent fallback, which otherwise reports α = 1) is folded to **α = 0** so
        the control reads it as struggling and backs off rather than growing — the α-only `StepReport`
        signal cannot otherwise distinguish a clean full step from a non-descending fallback. Wired
        through `coupled_continuation(inner_steps=…, inner_tol=…)` (returns a `DualTimeStep` when
        `inner_steps > 1`, else the unchanged `PseudoTransientStep`) and reachable as
        `solve_coupled(coupled, inner_steps=…)`. **The default path (`inner_steps = 1`) is byte-unchanged.**
      - **`DualTimeControl` IS NOW THE DEFAULT for a dual-time observed march, and it CARRIES β across
        refreshes — this reaches a developed recirculation ~4× faster than the residual-keyed control
        (measured 2026-07-30, and it SUPERSEDES the "runs the transient away" verdict just below).** The
        reachability crawl (~75–90 outer steps/rung to develop the pitzDaily bubble) was a **step-control
        defect**, not a pseudo-time limit. Two defects, both fixed/retired here:
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
        Measured on the matched-seed rung-1 testbed: SER ~75 outer steps to `x_r/h≈7.74`; carrying
        `DualTimeControl` ~22 (β_min 0.02) / ~18 (β_min 0.005), full rtol 1e-6 in 28–51. Full cold ramp
        (hybrid IC → Re/100 → Re/10 → target Re 25000): **59 total outer steps**, `x_r/h` 8.07 vs OF 7.74
        (developed). Self-regulating: α clips to 0.25–0.5 in the steepest development, recovers to 1.0, then
        β falls to the `beta_min` floor and the tail converges near-quadratically. `beta_min` is a
        speed↔smoothness knob (0.005 fastest but can overshoot the steady bubble on a cold rung with a
        loose seed + big Re jump, costing a couple of expensive recovery steps; 0.02 = the class default,
        smoother). Full finding: `reference/REACHABILITY_FINDINGS.md`.
      - **~~`DualTimeControl` RUNS THE TRANSIENT AWAY~~ — SUPERSEDED (see above).** The original bullet
        read: the α-control grows Δτ blind to the steady residual, so it drives `x_r/h` past the steady
        state without settling (residual bottoms ~0.05 then rises to 0.1+). That was measured on the Re/100
        anchor **before the carry fix and without leaning on the `beta_min` floor**. With β carried and the
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
        and stalls Δτ**, taking ~4× more outer steps than the α-based default. Prefer it only where the steady
        residual is a reliable monotone progress signal. Its `next_step` state is `(β, prev ‖R‖)`. Unit-tested
        in `tests/unit/test_step_control.py`.
      - **THE LOW-β WALL IS THE BLOCK-SIMPLE PRECONDITIONER, AND THE ILUT BREAKS IT.** With
        `ResidualRatioDualTimeControl` the residual descends cleanly (no runaway) but block-SIMPLE's coupled
        solve goes **NaN at β ≈ 0.067** — the low-shift conditioning wall (block-SIMPLE cannot solve the
        near-unshifted saddle; the same limit as its adjoint stagnation). The monolithic ILUT forms the true
        coupled inverse, so `coupled_ilut_continuation(inner_steps>1)` (a `DualTimeStep` preconditioned by
        the ILUT — the branch added alongside the single-step one) drives β **monotonically to 0.041 with no
        NaN, ~6 GMRES cycles flat**, residual 0.65 → 0.043 (row-scaled) on the anchor. So the ILUT is what
        makes the large-Δτ dual-time march reachable at all.
      - **Residual FLOOR + over-development past the minimum = loose `inner_tol`, NOT the preconditioner.**
        Even with the ILUT (cycles flat at 6 — the linear solve is fine), the march bottoms ~0.043 (x_r/h
        ≈ 2.9) then slowly over-develops. Cause: dual-time's unconditional stability comes from the inner
        loop driving `G = R + βd(φ−φⁿ)` to zero each step; at `inner_tol = 0.05` the implicit step is only
        5%-solved, so a large-Δτ backward-Euler step on a half-solved system overshoots. Fix = tighten
        `inner_tol` (with enough `inner_steps` to reach it) — **affordable precisely because the ILUT makes
        the low-β inner solves cheap**, where block-SIMPLE could not. ILUT removes the conditioning wall;
        tight `inner_tol` restores dual-time stability; the two together are what settle the rung.
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
        (`reference/ILUT_ITERATION_GAP_FINDINGS.md`). The march's "6–9" is the dual-time inner-loop sum, and
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
    `f(x) = Σ wᵢ(x)|Rᵢ(x)|` with `w` from the operator diagonals and field magnitudes. The weights are
    frozen within an iteration (so the line search compares like with like) and rebuilt each iteration
    — so a direction that descends in *this* iteration's frozen `f` need not reduce the *next*
    iteration's `f`. Do not assume the frozen-per-iteration measure behaves like a fixed merit function.
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
    so it stops on `‖Mr‖₂/‖Mb‖₂ ≤ target` (≈ 1% *solution* accuracy at `target=1e-2`, since M≈A⁻¹) —
    immune to those rows. Measured on the real cold-IC march: ~3-5 cycles (often 2-3/step), ~4× fewer
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
    `forward_march` entry, hence reset at every preconditioner refresh. With refreshes every ~15 steps
    the ratio `‖R‖/‖R₀‖` never falls far below one, so **β is pinned near β₀ = 2 for the entire march**
    rather than decaying. Measured on the drift-refreshed cold-IC pitzDaily march (158 steps, 6
    refreshes):

    | step | 45 | 60 | 90 | 110 | 158 |
    |---|---|---|---|---|---|
    | β from the **global** ratio | 0.033 | 0.027 | 0.024 | 0.022 | 0.016 |
    | β **actually used** (segment-local) | 1.74 | 1.98 | 1.99 | 1.79 | 1.85 |
    | α | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

    Three consequences. (i) Enabling the refresh silently converts SER into **constant-β ≈ 2** — if a
    different damping is wanted it must come from `β₀` or a different schedule, not from expecting SER
    to ramp. **And β₀ = 2 is too much: a march at `β₀ = 0.5` reaches `x_r/h` 2.43 against the shipped
    configuration's 1.67 in fewer steps and less wall time (see the `ShiftBasis` section) — so the
    pinning is not a curiosity, it is holding the solve at ~4× the useful damping.** (ii) **An α-targeting controller has nothing to push against**: α is already 1.0000 at
    every step from 45 on, so the controller sits at its set-point while the residual falls ~0.5 %/step
    — the productivity ceiling is *not* an α problem, which re-scopes #22. (iii) Any probe that derives
    "the march's operating β" from the global ratio is wrong by ~80× at a developed state.
  - **THE SER β SCHEDULE RUNS BACKWARDS FOR STIFF COUPLED RANS (measured, pitzDaily — the dominant
    cost, and it is the globalization, not the preconditioner).** The switched-evolution-relaxation
    schedule `β = β₀(‖R‖/‖R₀‖)^p` *lowers* β as the residual falls, on the premise that a smaller shift
    means a more Newton-like, more productive step near the root. **On this problem the premise is false:
    the efficiency-optimal β *rises* as ‖R‖ falls, so SER drives β the wrong way and the coupled march
    grinds instead of entering the quadratic basin.** Two independent measurements on E1's checkpoints
    (`solve_coupled`, twolevel, corrected hybrid IC; each re-solving one frozen step across fixed β, PC
    rebuilt at that state):
    - **Efficiency (residual reduction per second):** optimum β ≈ 2 at rel 0.38, **≥ 5 at rel 0.05**,
      while SER's β *fell* 0.76 → 0.10. At the developed state SER's β is ~50× below the optimum, a **~190×**
      step-efficiency gap (0.003 vs 0.56 %/s).
    - **The mechanism is line-search CLIPPING, seen directly via the step-length factor α.** α (the
      fraction of the shifted step the backtracking search keeps) is a clean monotone signal: it rises
      with β and hits **α = 1 exactly at the efficiency-optimal β** — the point where the full damped step
      *just stops overshooting*. Below it the step overshoots and is clipped to near-nothing; at it the
      step is full and productive.

      | β | α @ rel 0.38 | α @ rel 0.05 |
      |---|---|---|
      | 0.10 (≈SER in the tail) | 0.016 | **0.031** |
      | 1.0 | 0.50 | 0.25 |
      | 2.0 | **1.00** | 0.50 |
      | 5.0 | 1.00 | **1.00** |

      **SER operates at α ≈ 0.03 in the tail:** the full Newton step overshoots by ~33× (at β=0.05, ~80×),
      and the line search salvages a ~0.4% crawl from it. *That* is the grind — not near-convergence, not
      preconditioner cost. The α = 1 boundary is the controller target (raise β until the full step is
      marginally accepted); α is far less noisy than the per-step residual reduction ρ (which swings
      37%↔6% at fixed β and wrecked a first, ρ-driven controller that ratcheted β into a runaway).
    - **Caveat — β-schedule and PC-refresh are COUPLED; the optimal-β numbers above use a PC rebuilt at
      each state.** In a real march the preconditioner is frozen at the cold IC, and a bolder β moves the
      state faster, staling that frozen PC faster (the ρ-controller runaway hit 119 cycles at β=10.4 — high
      β should be *cheaper*, so that was PC staleness, not the shift). So an α-targeting β schedule and the
      scalar-AMG refresh (below) must be co-designed, not tuned in isolation. A **β-independent staleness
      indicator** — the drift of the frozen operator's coefficients, `‖Δν_t‖`/`‖Δṁ‖` relative to the
      freeze state — is the clean refresh trigger this motivates (it fixes the `CycleGrowthTrigger`
      confound, #19: cycle count rises from β→0 *and* staleness, drift rises only from staleness).
    - **VALIDATED end-to-end (α-targeting controller + PC refresh strictly dominates SER on pitzDaily).**
      A prototype controller — raise β toward the α=1 boundary (`β ← β/α`, capped), ease gently when
      α=1 — with the k/ω AMGs refreshed every 5 steps and the step `filter_jit`'d (to match SER's
      compiled `while_loop` footing, ~2.2 s/cyc), A/B'd from the cold hybrid IC against E1's SER march:

      | reach | SER (E1) | α-controller + refresh |
      |---|---|---|
      | rel 0.10 | 15.5 min | 11.4 min |
      | rel 0.054 | **64 min** | **24 min (2.6×)** |
      | deepest | **rel 0.052** (67 min, then stalled) | **rel 0.032** (41 min) |

      Faster at every overlapping residual, the lead *widens* into the tail (1.3× → 2.6×), and it
      reaches residuals SER never touched. The mechanism is the diagnosis playing out live: as the
      state stiffens α drops below 1 and the controller *raises* β into the 2–5 band (refresh holding
      cycles ~16) while SER collapses to β≈0.10 and grinds. Two prior arms confirm the attribution:
      (a) the **frozen-PC** α-controller *lost* (0.65×) — cycles rose with β (25 vs SER's ≤14),
      the β↔PC-refresh coupling biting, so the refresh is load-bearing; (b) the **eager** (un-jitted)
      version was handicapped ~1.4×/cyc — the jit is needed for a fair comparison, not for the physics.
    - **The controller has a CEILING — it does not converge either (it stalls at rel ~0.03, deeper than
      SER's ~0.05, not at a root).** The cause is its own **over-damped hunting**: the `β/α` raise
      overshoots *past* the α=1 boundary to where the full step is tiny (α=1, ρ~2%), then eases slowly;
      α saturates at 1 above the boundary, so the controller is blind there and cannot sit at the
      productive edge (the sweep's 20–60%/step β). So the direction is right and the win is real, but a
      dynamics rework is needed: approach α=1 *from below* without overshooting, or pair α with a
      step-productivity signal.
    - **PRODUCTIONIZED as an injected strategy pair (the direction is shipped, opt-in).** The β schedule
      is now the injected `RelaxationSchedule` (SER = `SwitchedEvolutionRelaxation`, the default; see the
      `continuation.py` bullet), and the α-targeting control is `AlphaTargetingControl`, a `StepControl`
      on the eager `forward_march` (opt-in via `solve_coupled(step_control=…)`, composes with
      `refresh_trigger`). It is **never a default** — it does not converge standalone and loses without
      the refresh — and its numeric gains (`beta_start`/`growth_cap`/`ease`) are placeholders, like the
      trigger's. The dynamics rework above is the open follow-up. Study harnesses in the scratchpad
      (`beta_sweep.py`, `alpha_probe.py`, `alpha_controller_march.py` = frozen-PC, `alpha_refresh_march.py`
      = the winning arm) remain as the calibration/replay tools.
    - **A PER-BLOCK β (separate shift damping for flow / k / ω) is DOMINATED — measured, do not re-attempt
      (`per_block_sweep.py`).** The Euclidean ‖R‖ on the coupled state is ~100 % ω (ω O(1e1) vs flow O(1e-2),
      k O(1e-3)), so a natural idea is to damp each block by its own β — the block-diagonal shift already
      supports it (unpack the shift diagonal `[a_P·u, 0·p, d_k·k, d_ω·ω]`, scale each slice, repack; the flow
      preconditioner keys off `β_flow` via its `a_P(1+β)`, the scalar AMGs are β-independent). Swept at the
      developed state (rel 0.05), holding `β_ω` high and lowering `β_k`/`β_flow`, it loses on every axis
      against uniform β:

      | (β_flow, β_k, β_ω) | α | ‖R‖ kept | d(flow) | d(k) | d(ω) |
      |---|---|---|---|---|---|
      | **3, 3, 3** (uniform) | **1.00** | **29 %** | −1 | −2 | 29 |
      | 3, **1**, 3 | 0.25 | 10 % | 0 | +0 | 10 |
      | 3, **0.1**, 3 | 0.06 | 1 % | 0 | +2 | 1 |
      | **1**, 3, 3 | 1.00 | 24 % | −2 | −9 | 24 |
      | **0.1**, 3, 3 | 0.50 | 1 % | +2 | −26 | 1 |

      Two failure modes, **neither a damping problem**: (i) **k is acceptance-limited** — a smaller `β_k`
      *does* let k descend (d(k) −2 → +2 %), but the bigger k-step makes the *coupled* full step overshoot the
      ω-dominated norm, so the line search clips α (1.0 → 0.06) and ω progress collapses (29 → 1 %); crediting
      k would need a block-aware *acceptance* norm, which is the dead `BlockScaledNorm` (below). (ii) **flow is
      coupling-limited** — no `β_flow` un-sticks it (d(flow) stays ≤ 0 down to β_flow=0.3; only the ruinous
      β_flow=0.1 at 98 cycles nudges it +2 % while cratering k −26 %), because flow is waiting on ω through the
      two-way ν_t coupling. The blocks are coupled through **both** the direction (flow↔ω) and the acceptance
      (ω-norm), so per-block *damping* cannot separate them. This re-confirms the old "Lever D" per-block
      under-relaxation ruling, now with the mechanism visible under log-ω + the adaptive wall.
    - **The lever is a HIGHER uniform β, not a per-block one — the same sweep shows β=5 ≫ β=3.** At rel 0.05,
      uniform **β=5 keeps α=1 and cuts ‖R‖ 63 % in one step, vs β=3's 29 %**, flow/k barely perturbed
      (−1 %, −1 %) — i.e. the efficiency-optimal β at the developed state is *above* 3, extending the
      "optimum β rises as ‖R‖ falls" table above. Per-cycle efficiency is ~flat (~1.7 %/cyc at both β=3 and
      β=5, α=1), so a higher β is not free per cycle; it wins on **step count and overhead** (fewer Newton
      steps → fewer PC refreshes, recompiles, line searches) and it stays productive (α=1). Confirmed on a
      real march: a **constant β=3** march (`const_beta_march.py`, jit + refresh-every-5, from the cold hybrid
      IC) descends monotonically **past SER's ~0.052 floor** (reached rel ≲ 0.035) but then *grinds* in the
      tail at ρ ~2 %/step — the too-low-β symptom, exactly where β≥5 would nearly halve ‖R‖ per step. So the
      settled next step is the β-climbing controller (#22: climb β while α=1), **not** a per-block β, a norm
      change, or physical/order continuation.
  - **Where the coupled-solve cost actually is (settled by measurement).** As the SER ramp drives `β → 0`
    through the march, the *unshifted* coupled saddle Jacobian is severely ill-conditioned, so the
    diagonally-shifted GMRES burns thousands of matvecs per solve (measured: one shifted solve ≈ 36 s at
    β=2, 127 s at β=0.2 on ~12k-cell pitzDaily — note lineax `num_steps` counts restart **cycles**
    ×`restart`, not iterations). **The `β → 0` here is SER-induced and correctable, not inevitable — see
    the schedule-runs-backwards finding above.** Several levers were probed: two are wired but **off by
    default** (kept for further evaluation, not the fix), one is dead, and one — refreshing the **scalar**
    k/ω AMGs after the flow separates — is a real ~2.6× win, now BUILT (see below):
    - **Flooring the SER `β` below (`β = max(beta_floor, β₀(‖R‖/‖R₀‖)^p)`, `PseudoTransientStep.beta_floor`,
      default 0 = off) — correctness-safe, a measured WASH, kept off-by-default.** It never moves the
      converged root (the shift `β d` scales the correction `δ`, which vanishes at `R=0`; it only damps the
      *path*, linear instead of quadratic terminal steps) and it does make each late solve cheaper. But
      end-to-end it is a net wash: floor 0.0 vs 0.3 reached the same tolerance in the same wall time on
      `solve_coupled`, because the cheaper late solves cancel the extra Newton steps. Wired through
      `coupled_continuation(beta_floor=…)` for further evaluation; not a default because it is a wash.
    - **The default coupled residual measure is the row-equilibrated `RowScaledNorm`
      (`coupled_scaled_norm`), NOT the Euclidean ‖R‖.** The Euclidean coupled residual is `ω`-dominated
      and *mis-ranks* states (a converged field scores worse than a badly wrong one — the warning above);
      `RowScaledNorm` divides each row by its own diagonal and each block by its field magnitude, so every
      equation is judged comparably. `coupled_continuation` / `coupled_ilut_continuation` build it by
      default; `block_scaled_norm=True` selects the coarser one-scale-per-block `BlockScaledNorm`
      (`_coupled_residual_norm`), and `residual_norm=jnp.linalg.norm` recovers Euclidean.
      (`mass_flow_coupled_continuation` still defaults to Euclidean pending a constraint-aware variant.)
      The row-scaled measure does **not** fix the forward stall (globalization-bound; it plateaus under any
      measure — that plateau is the *honest* signal, where the Euclidean fall was a `β×travel`/`ω`-magnitude
      artifact); it makes the measure honest and is required to judge this case correctly.
      **The measure must be held FIXED across a refresh (binding, #156 seam 4).** `BlockScaledNorm` is
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
      dead.** It made the channel worse (85 vs 51 outer cycles at β=0.5) and on recirculating pitzDaily was
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
        scipy AMG assembly cannot run inside it; `solve_coupled(refresh_trigger=…)` is the driver. A
        refresh still forces a full recompile (~60–240 s) because these are non-pytrees hashed by
        identity, which is why `refresh_limit` bounds how often it may happen. That recompile is
        avoidable in principle — **the coarsening structure is value-independent** (`_aggregate` takes only
        `(owner, nb, n)`, pure graph topology, so for a fixed mesh the aggregates, `n_coarse` and every
        sparsity pattern are invariant), so only `val`/`diagonal`/`lam_max`/`coarse_inv` change; making
        those traced leaves over a static index structure would turn a refresh into a cache hit.
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
        collapses ρ (34.0 → 9.6) but barely moves the one-shot error (24.1 → 22.6), and the ρ-minimizing
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
- **`march.py` — BUILT (`forward_march`, `StepReport`/`MarchResult`, `RefreshTrigger`/`CycleGrowthTrigger`):
  the observed, forward-only march that drives a mid-march preconditioner refresh.**
  - **Two marches, ONE decision layer (binding — this is the shape to hold).** `_forward` (traced,
    inside `custom_vjp`, has the root guard, cannot stop early, cannot be observed) and `forward_march`
    (eager Python loop, forward-only, **no guard by design**, stops on an injected trigger, reports every
    step). They are not duplicates: `forward_march` calls the **same** `forward_step.stepper()`, the same
    `forward_step.norm()`, and the same `_within_tolerance`. The only residue is a ~6-line loop shell,
    pinned against drift by a test that both marches reach the same state on the same residual.
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
  - **Reactive divergence retry — `retry_solver` recovers a step an INEXACT preconditioner poisons,
    without tightening every step (BUILT).** An *inexact* preconditioner (the threshold-ILU) can return a
    non-finite correction on the stiff operator an aggressive Courant overshoot produces, where the
    *exact* complete-LU returns a finite one — the loose default Krylov tolerance is what leaves that
    correction too inaccurate. `forward_march(retry_solver=…, retry_divergence_cap=inf)` redoes a diverged
    step **from the same pre-step state** at the tighter `retry_solver`; the trigger is
    `_has_diverged` (non-finite, or `> divergence_cap·reference` — default `inf`, i.e. non-finite only,
    because the residual legitimately *rises* during development via `β×travel`, so a tight cap would
    false-fire on the reachability descent). **The preconditioner is NOT re-refreshed on retry** — under a
    β-tracking refresh (`ilut_beta_tracking_refresh`, `.claude/rules/turbulence.md`) the factor is already
    fresh at this `(state, β)`, and re-factoring the deterministic factorization at the same point is a
    no-op; the failure is an under-converged *Krylov* solve, not a stale PC, so only the Krylov tolerance
    is tightened. This is orthogonal to the refresh *gating*: the retry recovers a diverged step whatever
    cadence the ILUT was refreshed at. One retry: a still-diverged
    step breaks as before. Default `retry_solver=None` is **byte-identical**, and the exact-LU path never
    triggers it. **Why it beats tightening every step:** measured on the aggressive pitzDaily ILUT ramp,
    rung-1 steps 1–7 ran on the cheap loose solver and *only* the diverged step 8 retried tight —
    recovering to the exact-LU value (ratio 9.72e-2) and tracking the LU on — instead of paying the tight
    solve on every step. Threaded through `solve_coupled(retry_solver=…)`; forward-only (raises under
    `jax.grad`, same guard as the refresh/control). Pinned by `test_forward_march.py`
    (`test_march_retries_a_diverged_step_with_the_tighter_solver`, `test_march_does_not_retry_a_finite_step`).
    On 2D the exact LU is cheaper *and* robust for free, so this is really a 3D-readiness lever (where the
    LU's fill is the wall and the ILUT is the only option).
  - **Reactive β-escalation bailout — `retry_on_cycles` ESCALATES β for a bad step, tried BEFORE the tight
    divergence retry (BUILT).** A step goes bad two ways — a *finite-but-expensive* solve (count `> N`) or a
    *non-finite* one — and on the stiff low-β saddle **both have the same cheap cure: more damping.** A
    larger β lifts the correction out of the NaN regime *and* cuts the cycle count (a stronger pseudo-time
    shift makes the same frozen preconditioner more diagonally dominant), and it is far cheaper than the
    tight-Krylov divergence retry. So `forward_march(retry_on_cycles=N, retry_beta_factor=2.0,
    retry_cycles_limit=2)` redoes a step whose count exceeds `N` **or** that diverged (non-finite / over
    `retry_divergence_cap`) **from the same pre-step state** with β escalated (`×retry_beta_factor`,
    re-applying `precondition_step` at the new β so a β-tracking refresh re-shifts), up to
    `retry_cycles_limit` times or until it converges/drops below `N`. It reads β off
    `active_step.relaxation_schedule.beta` (a `ConstantRelaxation` / `DualTimeStep`), so it requires a
    readable β and is inert on the default switched-evolution schedule. `retry_on_cycles=None` (default) is
    **byte-identical** (and a diverged step then falls straight to `retry_solver`, the pre-reorder
    behaviour). Forward-only; threaded through `solve_coupled(retry_on_cycles=…)`. Pinned by
    `test_forward_march.py` (`test_march_escalates_beta_on_a_cycle_count_spike`,
    `…_does_not_escalate_below_the_cycle_cap`, `…_escalates_beta_before_the_tight_divergence_retry`,
    `…_falls_back_to_the_tight_retry_when_escalation_cannot_fix_divergence`).
    - **The escalation must keep `_march_step` a compile-cache HIT (binding — a measured recompile
      hazard).** The retried step redoes the (coupled, minutes-to-compile) `_march_step` at the escalated
      β, so a treedef **or aval** change on that step recompiles the whole solve every retry — which on a
      stiff region that retries most steps is ~half the march wall. β is a dynamic 0-d leaf, so the *value*
      change is fine; the trap is the leaf's **abstract value**. Escalate by **scaling the existing leaf**
      (`beta * retry_beta_factor`), never by rebuilding it from a Python float (`jnp.asarray(float(beta) *
      f)`): the latter yields a *weak*-typed float64 array whose dtype/weak_type need not match the leaf
      the step control set, and any mismatch is a cache miss. Scaling preserves the aval exactly, so the
      retried step is a hit for **any** β dtype (weak/strong f64, f32). The shipped controls happen to set
      weak-f64 (so the old form was accidentally a hit), but that was an unpinned coincidence one JAX
      weak-type-promotion change from breaking. Pinned by
      `test_a_forced_escalation_adds_no_march_step_compilations` (a strong-typed β leaf, the case the
      rebuild recompiled on). Note the `retry_solver` (divergence) fallback is a *separate*, one-off
      recompile — a distinct solver object (restart-40 vs the forward restart-15) is a genuinely different
      static key, compiled once and reused; it is not a per-step cost.
    - **Why escalation leads and `retry_solver` is the FALLBACK (the reorder, measured on `bfs3d`).** The
      two retries used to run divergence-first: a NaN'd step ran the tight `retry_solver` (a 1e-4 Krylov
      solve, restart-40) and *then*, seeing its high count, the cycle bailout escalated β. On the 3D march
      that order was the single worst cost — measured on the cold `bfs3d` cold-continuation, step 28's
      primary NaN'd (α collapsed), the tight retry ground ~325 matvecs (~40 min) to recover it to finite,
      and *then* the β-escalation re-damped the same step to a clean ~5-cycle solve — so the entire tight
      grind was wasted work the escalation superseded. Reordered, the escalation fires first on the NaN,
      recovers the step cheaply, and the tight `retry_solver` fires only as a **fallback** for a non-finite
      step escalation could *not* fix — the genuine inexact-ILUT case (loose Krylov → non-finite δ that a
      tighter Krylov, not more damping, cures), where `retry_on_cycles` is typically `None` anyway so
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
      `retry_on_cycles < cycle_budget`** so a capped primary's reported count trips the escalation (else the
      partial non-converged step would be accepted). Good steps converge well under the budget, so they are
      byte-identical; only a grinding primary hits it. `cycle_budget=None` (default) is unbounded and
      byte-identical. Threaded through `coupled_amg_continuation(cycle_budget=…)` (and the shared
      `_monolithic_factor_step`, so the ILUT/LU steps can take it too); forward-only, like the escalation it
      feeds. This is Agent C's "small-budget primary + inner abort" realized as an inner-loop cost cap rather
      than a non-attainment flag threaded through every solve layer — same effect (a doomed primary costs
      ~`cycle_budget` matvecs, not `inner_steps ×` a stagnation), far smaller blast radius. Pinned by
      `test_dual_time.py` (`…cycle_budget_caps_the_inner_loop`, `…none_is_the_unbounded_step`).
    - **The escalated β is CARRIED into the control — so a static β floor can be dropped and the *controller*
      decides how low is safe (BUILT).** β is inverse to the pseudo-timestep, so a static `beta_min` is a cap
      on the *largest* timestep the march may take, applied everywhere — which slows convergence in regions
      that could safely take a bigger step. The escalation is the per-region feedback for "how low is safe
      *here*": it fires exactly where β went too low. But without carrying it back, the next outer step's
      `step_control.next_step` recomputes β from the control's own (floor-ward) trajectory and **re-pays the
      escalation every step** — the observed low-β tail (β pinned at the floor, each step re-escalating). So
      after an escalation `forward_march` seeds the control's carried β with the escalated value via
      `step_control.carry_beta(state, β)` (the dual-time controls implement it: `DualTimeControl` carries a
      bare β, the `…Residual…` controls carry `(β, prev_residual)` and keep the residual so the ratio signal
      is unbroken). The control then continues its grow/brake dynamics *from* the discovered-safe β, so
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
      - **A refresh costs ~38 s, not the 60–240 s assumed elsewhere in this file** (the refresh step took
        59 s against a 21 s steady step). It repays itself inside one step, which is why
        `refresh_limit` can be generous rather than hoarded.
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
    `retry_on_cycles` at its `None` default, so the beta-escalation never fired and nothing in the log
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
    format a report with it. It is built for streaming — rule, headings and row are separate methods
    each returning one line — which is what lets a table appear in a log being tailed. An over-wide
    value **widens its row rather than being truncated**: a cut-off number is a wrong number.
    Pinned by `tests/unit/test_text_table.py`.

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
    `StepControl`s live in `solve/step_control.py`: **`DualTimeControl`** (the Courant β-ramp, now the
    **default** for a dual-time observed march — carries β across refreshes, see the DualTimeStep bullet
    above), **`ResidualRatioDualTimeControl`** (the opt-in residual-keyed alternative),
    **`CflResidualDualTimeControl`** (see the bullet below), and **`AlphaTargetingControl`** (the
    single-step α-targeter — experimental, opt-in, does not converge standalone; see the "SER β schedule
    runs backwards" bullet).
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
    `AlphaTargetingControl` non-convergence ceiling flagged, applied to the dual-time controls. State is
    `(β, previous residual)`, carried across refreshes like `ResidualRatioDualTimeControl`; opt-in via
    `solve_coupled(step_control=…)`. Unit-tested in `test_step_control.py` (grows on α at a flat residual,
    brakes on a rising residual at α = 1, brakes on an inner clip, holds in the band, carries β). The ratio
    thresholds are march-calibrated numbers — set them from a logged march, not intuition (the 3D
    development overshoot shows ratios ~1.14).
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
  *fixed* (constant operator) so plain GMRES + the verified transparent-left-PC suffices (a *variable*
  inner would force FGMRES). On **`jaxamg`**: the search confirmed it is **NVIDIA/AmgX-locked and
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
    (dense pseudo-inverse) coarse solve; it has no `max_levels` parameter. On the fine level the
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
- **The efficient realization of the absorbed gradient — `SweptCorrectedGradient` (built, measured,
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
