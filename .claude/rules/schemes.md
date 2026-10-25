---
paths:
  - "aquaflux/schemes/**"
---

# Rules — `aquaflux/schemes/` (first-class swappable numerics)

Named, swappable, independently tested numerics: face interpolation, gradient reconstruction,
non-orthogonal correction, slope limiting. Dated measurements and investigation trails behind every
entry here are in `.claude/notes/schemes-log.md` (the #435 tetrahedral investigation in
`.claude/notes/schemes-435-tetrahedral-gradient-log.md`). This file may cite the C++/Fortran
precursors; the shipped surface may not (root Comment Convention).

## Binding decisions
- **Physics and numerics are separate.** Scheme classes live here; operators in `discretization/`
  consume them by constructor injection and never inline a scheme choice. Nothing in `schemes/`
  imports `discretization/` (the limiter lives here for that reason).
- One scheme class, injected into many operators/equations; never copy a reconstruction into two
  operators.
- Green–Gauss gradients are not formally second order and cap accuracy on skewed grids, for both
  the primary and the differentiated fields — that is why the gradient is swappable; keep the
  interface a drop-in.
- **`gradient_scheme=None` means no reconstruction, everywhere.** `ResidualAssembler.build` and
  `ScalarTransport.build` default to `None` (refused once a flux reports `uses_gradient()`); only
  `MomentumContinuity.build` and `SSTTurbulence.build`, where a gradient is mandatory, default to
  `DEFAULT_GRADIENT_SCHEME` (defaulting the others would build `(n, n_sym, n_sym)` correction
  matrices nothing reads).
- **Initializers take no `gradient_scheme`**: `potential_flow` reads `momentum.gradient_scheme`,
  `hybrid_initialize` reads `turbulence.gradient_scheme`, so a run cannot initialize on one
  reconstruction and solve on another (pinned by
  `test_no_initializer_names_a_gradient_scheme_of_its_own`).
- **Precondition any Krylov gradient solve on the RIGHT** (`GmresGradientSolve`). The cell
  preconditioners scale by ~1/V, so a left-preconditioned residual sits ~1e6 off the problem's scale
  and lineax's stagnation detector fires on a solvable system (it broke pitzDaily's omega gradient
  with every unit test green — O(1) test meshes cannot see it).

## Testability seam
Test a scheme by reconstructing an analytic field on a refinement sequence and asserting the order
of accuracy, no physics. Use `tests/support/meshes.py::perturbed_grid_2d` with **random**
perturbation (smooth perturbation flatters the scheme), and measure on interior cells only.
- ⚠️ **A sliver made by squashing a node band is an invalid mesh** (closure residual 6.6e-1): Green–
  Gauss is the divergence theorem, so every result on it is void. Squash one cell's top nodes
  (`tests/unit/test_gradient.py::_sliver_mesh`); check `closed_cell_residual`, never `face_planarity`.
- ⚠️ **Assert what survives the platform on a degenerate mesh** ("does not contract", "is not
  exact"), never the magnitude of a singular result: the same singular `M2` inverts to 1.6e16 under
  macOS Accelerate and NaN under CI's BLAS, and every `x > threshold` is False against NaN.
- ⚠️ **Never reduce before the comparison meant to catch a NaN**: XLA's CPU max-reduction can turn
  an all-NaN row into `-inf`. Compare per entry, then reduce booleans
  (`~jnp.all(|M2⁻¹| <= limit, axis=(1, 2))` in `_undetermined_cells`; pinned by
  `test_a_non_finite_correction_counts_as_undetermined_at_any_mesh_size`).
- A fixture that satisfies its boundary conditions identically, with no cell owning two
  gradient-type faces, hides boundary-probe defects; test the field that satisfies them only at the
  wall.

## `limiter.py`
`Limiter` → `VenkatakrishnanLimiter(softening=0.05, scale)`, `limit(field, context)` on the shared
`FieldContext` (reads its own gradient plus `mesh.face_cells`/geometry). Injected into
`LimitedUpwind(limiter=…)` (`discretization/advection.py`), evaluated only when that scheme runs.
- **`eps = softening · scale`**, not the old `eps² = vol K³` (dimensionally meaningless; there is no
  `k` field). `scale` is the field's reference magnitude — a float array leaf, `not_settings`. The
  vanishing-`d-` regularization is `1e-12 · scale`. `softening` is a differentiable leaf (pinned by
  `test_the_softening_constant_is_a_differentiable_leaf`); under `filter_jit` a Python float is
  still static — pass an array to trace it.
- **The scale is a fixed property of the problem, not the iterate's range** (decided with the
  owner): a state-read range changes the residual during the march and couples every row to the
  extremum cells; `stop_gradient` on it would make the Jacobian inexact and the adjoint drop a term.
- **`with_reference_scale(provider)`** on `Limiter` and `AdvectionScheme` (default returns self;
  `LimitedUpwind` forwards; Venkatakrishnan fills only an unset scale). `provider` is a zero-argument
  callable so schemes that read no scale never derive one. Bound by `MomentumContinuity.build`
  (reference speed), `SSTTurbulence.build` (per field, from `velocity_scale=`),
  `ScalarTransport.build` (`prescribed_range`); `ResidualAssembler` binds nothing, and an unset scale
  at `limit()` raises naming the remedy. Unit and mesh-size invariance are pinned directly.
- The neighbour-side position comes from `face_cells.neighbour_centroid` (periodic seams);
  `face_limiter` takes per-face positions as arguments. The seam test uses a sine (a cosine peaks on
  the seam, where a smooth extremum is limited by design whatever the seam does).

## `boundary_closure.py`
`BoundaryClosure(values)` — one scalar field's boundary face values as `(field, gradient) ->
(n_faces,)`, and the one home of a reconstruction's two boundary passes: `reconstruct(scheme, field,
mesh, geometry, *, operator_hook, imposed)` (constant part at zero gradient, plus
`boundary_gradient_weight`, plus `boundary_values_at` for a later pass), `value_weight` /
`gradient_weight` (`jvp` probes), `linearization(n_cells, dim)` (the `BoundaryLinearization` `bind`
takes). A frozen dataclass (holds a callable, lives for one call). `ResidualAssembler` and
`MomentumContinuity` (pressure, each velocity component) wrap their closures in it. **A new caller
reconstructing against boundary closures wraps them in this; do not write another copy.**
- ⚠️ **`gradient_weight`'s unit tangents are broadcasts (`jnp.broadcast_to(eye[k], shape)`), never
  `zeros_like(...).at[:, k].set(1.0)`**: that scatter is a computation on constants, which XLA folds into
  every compiled program holding the residual as an `(n_cells, dim)` constant (bfs3d 15.87 MiB in the
  residual alone; ~0.05 GiB resident over its programs, measured). Pinned by
  `test_the_gradient_weight_compiles_to_no_per_cell_constant`; any per-cell tangent seeded in a traced
  residual has the same trap (`.claude/notes/solve-direct-preconditioners-log.md`, "What the resident
  memory holds beyond live arrays").

## `GradientScheme` interface and `bind`
`gradients()` is a template method: it calls `_reconstruct_gradient` and applies an optional
`ImposedGradient` (below), so every scheme honours it.
- **It does not take `MeshContext`/`FieldContext` (decided, #280)**: it needs `mesh.node_coords`/
  `face_nodes` for the warp moment, which `MeshContext` deliberately does not carry; and it
  *produces* the gradient a `FieldContext` holds, from leading-order (zero-gradient) boundary values.
- **`bind(mesh, geometry, linearization=None)`** hoists geometry-only (and, with a
  `BoundaryLinearization`, condition-dependent) work out of the per-call path, bit-identical answers.
  The base returns `self` (pinned as *identity* by
  `test_a_scheme_with_nothing_to_prepare_is_returned_unchanged`).
- **The assemblers bind, and that placement is the safety property**: `ResidualAssembler.build`,
  `MomentumContinuity.build` (per solved field: `velocity_gradient_schemes`,
  `pressure_gradient_scheme`; `momentum.gradient_scheme` is the unbound form an initializer
  re-binds) and `SSTTurbulence.build` (`k_gradient_scheme`/`omega_gradient_scheme`; a wall
  prescribes `k` and lets `omega` extrapolate; its assembler is built per residual evaluation, so it
  passes `bind_gradient_scheme=False`). Never bind in `__post_init__` (pytree unflattening reruns it
  at every jit boundary). Dropping the call fails no number, so it is pinned by
  `test_an_assembler_prepares_its_gradient_scheme_for_its_own_geometry`.
- `ResidualAssembler.build` evaluates the linearization once, before any field exists — exact,
  because both derivatives are state-independent — but a property that needs a field to evaluate
  makes it **raise** naming `boundary_linearization=` rather than linearize against a zero `Gamma`.
- ⚠️ **A stale binding is silently wrong, not slow**: fixed-count sweeps mean the bound
  preconditioner decides the answer. A cell-count mismatch raises; a different geometry with the same
  count cannot be detected; binding outside a node-position differentiation freezes the shape
  derivative (field differentiation is unaffected).

## `CorrectedGreenGauss`, `GradientSolve` and the sweep count
`CorrectedGreenGauss(solver, preconditioner)` makes the gradient a sparse system `A_g G = B φ`
(`A_g` geometry-only). Linear-exact on irregular grids, caps near first order there. The diffusion
residual folds it in as the non-orthogonal correction (Gate C:
`tests/integration/test_skewed_diffusion.py`). `CompactGreenGauss` is the one-shot sum
(inconsistent on irregular grids, by design of the test).
- **`solver`**: `SweptGradientSolve(sweeps=4, relaxation=1.0)` (default; matrix-free preconditioned
  Richardson, unrolled under AD, no nested Krylov) or `GmresGradientSolve` (exact, implicit-diff,
  through `solve.solve_linear` right-preconditioned;
  ~180× the swept cost per coupled-residual eval on pitzDaily because its tangent solve re-enters on
  every jvp). Exactness unit tests pin `GmresGradientSolve()` explicitly.
- **`preconditioner`** (renamed 2026-10-17; there is no `CellPreconditioner`, `InverseCellVolume` or
  `ExactCellBlock`: the setting is a `DiagonalBlock`, the applied object a `GradientPreconditioner`):
  `VolumeDiagonalBlock` (default, builds `InverseVolume` or
  `CellBlockJacobi((I−B)⁻¹/V)` when a boundary block is present) or `ProbedDiagonalBlock` (per-cell
  probed block). `1/V` diverges on a flattened sliver (error unbounded with aspect ratio);
  `ProbedDiagonalBlock` stays flat at ~3× the cost — the opt-in for slivers, documented in
  `docs/gradient_reconstruction.md`. Flooring `1/V` is not a substitute.
- **Each sweep extends the residual's stencil one ring, only on a skewed mesh.** The sweep coupling
  is weighted by the skewness offset `x_f − (x_P + g d)`; skew-free, `A_g` is diagonal and sweeps
  past the first add nothing. The scalar residual reaches `sweeps + 1` (coupled RANS on a skewed
  cavity: `sweeps + 2`). `sweeps=1` *is* compact Green–Gauss. ⚠️ **A reach is a property of the
  assembled case and mesh — measure it with `jacobian_relative_error` on the case's own mesh, never
  derive it from the sweep count; `bfs3d`'s 3 is skew-free luck.** The short reach error lands in the
  pressure column (pressure enters only through gradients).
- **Truncating sweeps does not remove far coupling — it folds it onto the last retained shell**; the
  mass beyond a distance is set by skewness. An exact solve has *more* far mass (reach 9–11), so it
  is the wrong lever. The corrected system's exact solution is also less accurate than the
  multiple-correction answer at high skew.
- **Sweeps are a `lax.scan`** (compile-side limit: unrolled, a Hessian-corrected `outer × inner`
  program was OOM-killed in compilation). `sweeps` stays static; the zero-start apply is peeled
  (`sweeps` sweeps cost `sweeps − 1` applies; the compiler does not drop the scatters) and a single
  sweep short-circuits before the scan (`lax.scan` traces its body even at `length=0`); the carry is
  `(x, residual)` so `warn_tol` reads the residual that formed the last update. Not bit-identical to
  the unrolled form (≤1 ulp, flat in depth — a drift growing with depth would be a defect).
- **`GradientSolve.solve(..., operator_hook=None)`** — applied before every operator apply, for a
  ghost-cell exchange. Only `SweptGradientSolve` honours it; `GmresGradientSolve` and
  `HessianCorrectedGradient` raise (`MultipleCorrectionGradient` and `ProjectedStencilGradient` raise
  under decomposition too), so a partitioned run must name `CorrectedGreenGauss`.
- ⚠️ A diagnostic-emitting sweep (`warn_tol`) cannot sit inside an operator a Krylov solve
  transposes (`jax.linear_transpose` rejects the norm; `stop_gradient` does not help): hence
  `requires_linear_operator`/`emits_host_diagnostics` and a named `ValueError`.
- **Rejected: dense LU of `A_g`** — `O((n·dim)²)` per apply, dominated by the swept solve at every
  size; sparse LU needs host callbacks. Do not rebuild.

### `narrow_gradient_sweeps(tree, sweeps)` and `jacobian_gradient_sweeps`
`narrow_gradient_sweeps` returns a copy of any tree with every `SweptGradientSolve` (and
`CoupledBlockSweep`) capped; it only narrows, leaves GMRES/compact untouched, and rebuilds via
`dataclasses.replace` because `sweeps` is static (`eqx.tree_at` cannot reach it). A narrowed copy
must differ in the count and nothing else — it once dropped `relaxation` (pinned by
`test_narrow_gradient_sweeps_carries_the_relaxation`). ⚠️ It cannot narrow a GMRES outer solve; a
caller believing it narrowed one gets an unbounded stencil — check the reach.

Three distinct narrowings, never to be conflated: the residual's `sweeps` (the discretization —
`R` decides the root), `JacobianProbeSpec.gradient_sweeps` (what a coloured probe materializes, so the
recovered matrix is exact for the residual it was taken from rather than folded), and
`jacobian_gradient_sweeps` on `coupled_step` and `flow_march_step` (the copy the Krylov operator is
differentiated from — `J` decides only the rate, the same latitude `forward_rtol = 0.3` takes).
`None` is byte-identical. On pitzDaily swept-2 for the Jacobian is free and the knee is at 2; the
residual's exactness bought nothing measurable there, which is not a licence to lower the shipped
`sweeps` (log: "The Jacobian's gradient can be cheaper than the residual's").

### Sweep calibration (`gradient.py`)
`GradientSystem(preconditioner, operator, shape)` is what a solve consumes and the estimator
measures; `contraction_rate(system, iters=24, …) -> ContractionRate` is a Gelfand estimate of
`ρ(I − P⁻¹A)` in one jitted loop (`settling_ratio` is its self-check; `annihilated` flags a probe
annihilated by arithmetic); `SweepCalibration(tol=1e-4, iters=24, floor=1, cap=64, seed=0)` converts
(`sweeps_for`). Factories `CorrectedGreenGauss.calibrated`, `HessianCorrectedGradient.calibrated`,
`CoupledBlockSweep.calibrated`, `AveragedNeighbourHessian.calibrated`;
the calibrated `SweptGradientSolve` is constructed only in `_calibrated_solver`. Opt-in — no default
changed; each case's count is in its README.
- Traced geometry raises a named `ValueError` (calibrate outside the differentiated region).
- ⚠️ **The tolerance is L2; the worst cell exceeds it about half the time** (worst/L2 median ~12×,
  up to 60×): ask one to two orders tighter than the per-cell accuracy wanted.
- **No safety margin** — the estimator's low bias and the transient's over-delivery cancel (0 L2
  misses in 195 combinations); a margin would double counts.
- ⚠️ **Calibrate the system that will run**: a factory must forward every setting that changes the
  operator or preconditioner (`local_schur_block`, the Hessian `boundary_closure`). Forgetting
  `local_schur_block` once sized tetrahedral sweeps at 64 instead of 23 (pinned by
  `test_the_coupled_sweep_calibrates_against_the_preconditioner_it_will_run_with`).
  `tools/sibling_builders.py` cannot see this family (polymorphic factories constructing different
  classes) — read the surfaces by hand; the factory signature-parity test covers the closure keyword.
- ⚠️ A calibration must never decide from a measurement on a singular system (its rate is
  platform-dependent); `AveragedNeighbourHessian`'s weight ladder starts at 0.05 for that reason, and
  `fastest_boundary_closure` maps a non-finite or annihilated rate to unusable.
- `validation/uvreactor_openfoam/gradient_sweep_calibration.py` measures reconstruction error
  against a converged reference — the independent check on the estimator; keep it.

## `MultipleCorrectionGradient` (`multiple_correction.py`) — the library default
`DEFAULT_GRADIENT_SCHEME = MultipleCorrectionGradient()` (`boundary_closure=OwnerGradient()`,
`fallback=None`), after Pont et al. (JCP 2017). Two face passes and per-cell matrix products, no
system to solve: recover `M1` by applying the Green–Gauss sum to coordinate fields (`D1 = M1⁻¹R` is
linear-exact), apply twice for a Hessian, recover `M2` on the quadratic basis and repair with `M2⁻¹`,
then subtract `D1`'s first-order error. **Every correction matrix is probed through the operators, not
derived**, so it cannot drift from them. Exact for quadratics on quad/hex/tet (with the right
closure); exactly linear in the field (tangent is an unrolled apply).
- **Reach**: the gradient's own reach is 3 on any mesh (no sweeps), and the scalar residual probed at
  reach 3 is exact. ⚠️ **The coupled pitzDaily RANS residual reaches 4 under it** (the velocity
  gradient feeds `ν_t` and the face viscosity); the shipped probe at reach 3 folds ~1e-6 into the `u`,
  `v` columns. Whether that costs Krylov cycles is unmarched; a per-column plan `(4,4,3,4,3)` would be
  exact (`validation/pitzdaily_openfoam/probe_reach_check.py`). On orthogonal `bfs3d` the second pass
  vanishes, but its p/k/omega columns still need a uniform reach 3.
- **The boundary closure must reproduce linear fields exactly** — the corrections absorb what a
  closure does to quadratics but never an error at linear order. `OwnerGradient` (default) is exact on
  quad/hex; `SkewCorrectedGradient` is needed where a tetrahedron owns **two or more** boundary faces
  (the owner closure leaves its six Hessian components underdetermined, `M2⁻¹` ~1e16). A raw
  one-sided difference is linear-inexact and worse; neighbour-averaging (which rescued Betchen's
  `A_HH`) makes it worse here — do not reason by analogy between the schemes.
- **`non_orthogonal_correction` (`schemes/interpolation.py`) is the shared term** the default closure
  and `discretization/diffusion.py` both use; a second copy is the defect.
- **Corner cells**: `bind` measures `M2⁻¹` per cell; with `fallback=SkewCorrectedGradient()` it
  rebuilds through `CellwiseFallback(cells, primary, secondary)` on just those cells, and `Corrections`
  carries the effective closure so `reconstruct` applies what was probed. **`fallback` defaults to
  `None`** (warns naming the count and the opt-in) because the skew closure can destabilize a coupled
  march at an unconverged iterate and no case has both the cell shape and a working march; whether the
  repair is safe in a march is open (#435). Warnings go through the package leaf
  `aquaflux/warn_once.py` (`warn_once`, one record per kind, `forget_warnings` for tests), so a mild
  report cannot silence a graver one; `projected_stencil.py` and `gradient.py` use it too.
- ⚠️ **Not for tetrahedra under a pressure coupling (decided 2026-09-23, #435)**: its second-pass
  weights run ~2900× a damping reference and flip Rhie–Chow damping diagonals; the duct march finds no
  admissible step. Closures, first-pass-only Rhie–Chow, ring/threshold withholding, mixed per-cell
  schemes and Betchen as calibration source were all tried and failed — the fix is a scheme that
  chooses its weights (`ProjectedStencilGradient`). Recorded in the docstring and docs; no default
  moved.
- **First pass inverts `(M1 − B)` with `small_inverse`** (the closed-form 1/2/3-dim adjugate in
  `gradient.py`), never `jnp.linalg.inv`: a batched LAPACK `getrf`/`trsm` cannot fuse and cost ~5× the
  pitzDaily residual (pinned by `test_the_boundary_condition_first_pass_calls_no_library_factorization`).
  `m1_inverse` at bind uses it too so the zero-weight path is bit-identical; `m2_inverse` (bind only)
  stays `jnp.linalg.inv`. ⚠️ pitzDaily/bfs3d timings under this scheme before 2026-10-16 predate it.
- **The whole residual is guarded too**: `tests/unit/test_compiled_residual_census.py` (fast tier, CPU
  only) pins zero `lapack_*` custom calls in the compiled coupled RANS residual and its JVP; restoring
  `jnp.linalg.inv` in the first pass reads 5 `getrf` + 10 `trsm` there (one per reconstructed field).
  Measurement: `.claude/notes/schemes-log.md`, "Compiled-residual census (2026-10-17)".
- ⚠️ A conditioning figure (`cond(M2)`) is per mesh — it runs from ~3 to 3.6e17 at unrepaired corner
  cells; never carry "well conditioned" as a property of the method.
- A design trap found here: a defaulted `eqx.field` on an abstract base makes every subclass field
  defaulted, so `reads_boundary_values` is a plain class attribute.

### `boundary_gradient_weight` (the boundary-condition first pass)
On a patch whose value follows the owner (zero-gradient, Neumann, Robin), the face value the
condition defines depends on the gradient being reconstructed; feeding the zero-gradient value gives
an error at **linear** order that the quadratic-probed corrections cannot remove (the Hessian doubled
per refinement). The value is affine in the owner gradient, so it moves to the left side:
`(M1 − B) ∇φ = raw`, one per-cell inverse, no iteration.
- **`boundary_gradient_weight` = ∂(boundary value)/∂(∇φ_owner)**, shape `(n_faces, dim)`,
  **differentiated from the closures** (one JVP per component), so it cannot disagree with them: zero
  on a prescribed patch, the tangential offset on zero-gradient/Neumann. Supplied by
  `ResidualAssembler._gradient` and `MomentumContinuity` (per velocity component). Its first form
  (`boundary_chain`, the full owner offset) made `M1 − B` singular on tetrahedra with two boundary
  faces; deleted.
- **`B_P` has one home, `schemes.boundary_gradient_block(weight, face_cells, geometry)`** (per volume),
  shared by `_boundary_condition_first_pass` and, since #648, by `CompactGreenGauss` (solves
  `(I − B_P) g = raw/V`) and `CorrectedGreenGauss` (adds `−V B` to `A_g` via
  `_CorrectedTerms.boundary_block`; `None` keeps the terms geometry-only, which is what `calibrated`
  measures). Both cell preconditioners include it. Orthogonal meshes are bit-identical.
  `VolumeDiagonalBlock` inverts `I − B` then divides by `V`, in that order (bit-identity). The owner-only
  scatter is `FaceCellConnectivity.scatter_to_owner`; `small_inverse` is not in `__all__`.
- The asymptotic sweep rate does not move (`B_P² = 0` for single-face cells), so geometry-only
  calibration stays valid; a fixed count sees the transient at corner cells, which is why the default
  preconditioner carries `B`.
- It reaches the first pass and the closure, not `M2` (probed against exact face values), so a
  quadratic keeps an `O(h)` boundary-cell inconsistency.
- ⚠️ Every corrected/compact measurement under derivative-type conditions on a skewed mesh before
  2026-10-09 predates it; Dirichlet-only ones are unaffected.
- **`HessianCorrectedGradient` still ignores the weight** — `B` would have to reach its bound outer
  preconditioner. Open on #648.

### The boundary-linearization probe (`bind` with a `BoundaryLinearization`)
`BoundaryLinearization` carries `value_weight` (∂φ_f/∂φ_owner) and `gradient_weight`
(∂φ_f/∂∇φ_owner), read by `jax.jvp` at rest. `_probe_corrections` drives each boundary face with
`a(ψ_P + w·g₁) + (1−a)ψ_f + a(d·n)∂ₙψ(x_f)` — the condition's construction fed **the basis field's own
value and normal derivative** (`a = 0` Dirichlet, `1` zero-gradient/Neumann, `1/(1+β)` Robin). Two
earlier rules were wrong: geometry-only probing (inexact for every derivative-type condition) and the
condition's own data (told every basis field its normal derivative was zero; froze the tetrahedral duct
march). Classify by `value_weight`, never by `gradient_weight != 0` (zero on an orthogonal face).
Pinned by `test_every_condition_is_exact_for_a_quadratic_satisfying_it_at_the_wall` (the orthogonal
parametrization is the only one that catches the classification mutation). Open: the
`SkewCorrectedGradient` march-loss table was measured under geometry-only binding and not repeated.

### `ImposedGradient` (`gradient.py`)
`ImposedGradient(cells, gradient)` is passed into `gradients()` for a derivative someone knows — the
wall `omega` gradient, built per evaluation by `turbulence/transport.py::_wall_omega_gradient` and
passed via `ResidualAssembler.gradient(imposed=…)`. `MultipleCorrectionGradient` imposes on the
first-order gradient before the second pass, on those cells' boundary faces, and on the result.
`HessianCorrectedGradient` applies it to its converged answer only (imposing on the iterate would
move the Schur fixed point). Imposing nothing is byte-identical; with an imposition the
reconstruction is affine with the same linear part.
- It reaches `ResidualAssembler.gradient`, **not** `.residual` (the diffusion correction on the ω
  fixation cells is measured negligible); the first interior ring's ~2× overestimate is a
  boundary-*value* question for turbulence, not this seam.
- Imposed cells are deliberately off the correction matrices' exactness contract.

## `ProjectedStencilGradient` (`projected_stencil.py`)
Per-cell weights over a two-hop stencil: of the weights exact for quadratics, the ones nearest
`blend` × a damping reference (`P_g⁻¹ b_g`, a Green–Gauss pass under the gradient equation's per-cell
block). `blend=0` is an unweighted quadratic least-squares fit (Barth & Frederickson 1990; pinned by
`test_the_minimum_norm_end_is_an_unweighted_quadratic_least_squares_fit`, the only test sensitive to
*which* exact weights are chosen) — so the novelty is the blend and the boundary rows. Built in `bind`,
applied as one gather and contraction.
- **The reference is a target, not an ingredient**: the projection lands in the exact set whatever it
  aims at, so a cheap low-order target cannot cost exactness — it decides weight size, and through it
  the Rhie–Chow damping sign (exactness and weight size are independent).
- **Boundary datum per field, from the bound linearization (#483)**: a prescribed face constrains a
  monomial's value, an extrapolating one its normal derivative. ⚠️ **A Robin/convective face is
  refused, not approximated** (the mixed row is derived on paper only; #496).
- Serves `k`/`omega` because `SSTTurbulence` binds per field at build; binding against a traced mesh
  raises (the stencil is built from connectivity).
- Not under domain decomposition (raises); the weights are constants after binding, so binding inside
  a geometry differentiation gives a wrong shape derivative.
- ⚠️ **`blend=0.75` came from one mesh** (the tetrahedral duct under its pressure conditions); re-measure
  with `validation/tetrahedral_gradient_ab/rhie_chow_sign_probe.py` before trusting it elsewhere.
  On bfs3d its conditioning guard fires informatively on thin near-wall cells.

## `HessianCorrectedGradient` (Betchen & Straatman), `gradient.py`
Coupled gradient+Hessian reconstruction, 2D and 3D, with every block from AD of the forward
reconstruction (no hand-derived coefficient matrices). Exact for quadratics on irregular grids. Fields:
`hessian_solve` (default `CoupledBlockSweep()`), `local_schur_block=True`,
`boundary_closure=OwnerHessian()`, `prepared_outer` (set by `bind`). Dominated on cost for a march by
`MultipleCorrectionGradient`; unregistered in the case file.
- **`hessian_solve` is one injected strategy**, which makes conflicting solve settings
  unrepresentable: `CoupledBlockSweep(sweeps=20, relaxation=1.0)` (default; alternating Gauss–Seidel
  sweeps of both blocks, 2 face passes per sweep, fixed point provably the Schur solution) or
  `PackedSystemSolve(solver)` (no elimination; with `GmresGradientSolve` it is **the exact reference**
  the tests, harnesses and docs check a sweep count against, which is why it is public). There is no
  `solver`/`hessian_solver`/`schur`/`coupled_sweep` field on the scheme itself any more.
- ⚠️ **`NestedHessianSolve` is deleted (2026-10-17); there is no such class**, nor
  `calibrated(coupled=...)`. It was dominated by the coupled sweep (1.4x–12x its cost, faster only on
  an orthogonal mesh) and kept only as a comparison control; its harness arms went with it. Every
  "nested" measurement in `.claude/notes/schemes-log.md` describes that path and cannot be re-measured
  from the tree. **`_HessianSystems.outer` is kept** though no package code calls it: it is the true
  Schur operator `local_schur_block` approximates, read by `schur_block_diagnosis.py` and
  `gradient_reconstruction_profile.py`.
- **The Hessian is solved as `n_sym` symmetric components**, the surplus equations reduced by
  `(A_P E)ᵀ` (Betchen–Straatman Eq. 16–17, via `contract_symmetric`'s adjoint identity) and **row-scaled
  by `1/vol`** (otherwise the reduced block is `~vol²` and every volume-scaled consumer degrades). The
  reduced block is SPD but dense `(n_sym, n_sym)` — storage per cell goes *up* (9 → 36 doubles in 3D).
- **Warped faces need the warp first moment** (`FaceGeometryScheme.warp_first_moment`, exact from the
  centre-fan triangles) in **both** equations: `+∇φ·P` and `−½Pᵀ(Hd+Hᵀd)` in the gradient equation,
  `+H·P` in the Hessian equation; the `Q` second moment needs none (divergence-theorem identity).
  Worth ~1000× on the snappyHexMesh reactor; planar meshes bit-identical. ⚠️ The moment must be
  oriented to the caller's normal (an unoriented one makes results slightly *worse*, which reads as
  "the term does not matter"; pinned by `test_warp_first_moment_follows_the_supplied_normal`), and
  contracted against the face-centroid gradient from the one `_face_gradient`.
- **`local_schur_block=True`** builds the outer preconditioner from each cell's own Schur block
  (`A_gg − A_gH A_HH⁻¹ A_Hg`, per cell, three blocks extracted independently and contracted) instead
  of `A_gg` alone; required on the 1.6M-cell reactor (the `A_gg` arm diverges there; `False` is kept
  only as the control). ⚠️ Extract all three blocks with owner- **and** neighbour-side columns, and
  never compose the operators (two scatters read neighbours). Verify any new block by lighting one
  cell and reading its own row of `S`. Probe `A_gH` sequentially, not under `vmap`.
- **Per-cell blocks come from `cell_diagonal_block` + `interpolation.blend_owner_neighbour`** — one
  probe per component, no colouring, from the same face kernel as the operator. The inner `A_HH` needs
  a block preconditioner, not a Krylov method (`1/V` gives ρ = 0.5 even orthogonal); `b_H` is zero.
- **Hessian boundary closures** (`HessianBoundaryClosure`): `OwnerHessian` (default; exact for
  quadratics; `cond(A_HH)` ~1e17–1e18 on all-tetrahedral meshes — does not solve there),
  `AveragedNeighbourHessian(weight=0.2)` (exact for quadratics at every weight, solvable on tetrahedra;
  `calibrated(mesh, geometry)` picks the weight by fewest sweeps from a ladder starting at 0.05),
  `AveragedInteriorHessian` (Betchen's Eq. 28–29; solvable but loses quadratic exactness at its empty
  case). A closure is exact for a quadratic iff its weights sum to one. Each declares its own
  `diagonal_probe` (a neighbour average must not be read as a diagonal term); both averaging closures
  share `_inverse_distance_average`. ⚠️ Full-weight averaging costs ~2.3× inner sweeps regardless of
  mesh size (block Jacobi cannot represent the neighbour coupling); the 0.2 blend recovers most of it.
  `cond ~ w^-1.75`, so prefer calibration to the default on a more degenerate mesh. Pair the averaged
  closures with `CoupledBlockSweep`, and treat a calibrated count equal to its cap as no measurement.
- **`CoupledBlockSweep` keeps `h` at zero on every call** — carried across calls the reconstruction
  becomes history-dependent and nonlinear. Its sweep count is the cost (~82 % of a reconstruction):
  calibrate it per mesh rather than lowering the class default; counts do not transfer between meshes.
  Over-relaxation is refuted (optimum ≤ 1.0). Closed for this sweep, do not re-propose: local
  least-squares replacement, sub-solving either block, polynomial acceleration/relaxation schedules,
  coarse-grid correction (log: "Responsibility → the UV reactor, and directions closed for the
  coupled sweep").
- ⚠️ **Its stencil is essentially global** (gradient↔Hessian↔neighbour Hessians), leaving ~3e-5 of
  `|dR/dφ|` beyond reach 5 on a perturbed grid — two orders above what breaks pitzDaily's coloured
  probe — and narrowing barely moves it. It does not bite on near-orthogonal pitzDaily; elsewhere,
  check the reach before probing a residual built on it.
- ⚠️ A face-pass count is a poor predictor of wall clock here (XLA already shares the geometry-only
  prologue across fields); where a ratio in the log comes from counting passes, it says so.
