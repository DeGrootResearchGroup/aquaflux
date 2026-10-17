---
paths:
  - "aquaflux/schemes/**"
---

# Rules — `aquaflux/schemes/` (first-class swappable numerics)

> ⚠️ **`coupled_continuation`, `coupled_lu_continuation`, `coupled_amg_continuation`, `lu_beta_tracking_refresh` and `amg_beta_tracking_refresh` no longer exist (deleted 2026-09-14, #371).** Entries below that name them are dated history. The coupled march is now one builder, `coupled_step`, given a preconditioner value (`BlockDiagonal` or `MaterializedJacobian(FieldSplit | BlockInverse)` (a `CompleteLu` inverse existed until 2026-10-11)), and a march that keeps its preconditioner current runs on a session (`open_session`) — see the rename table in `.claude/rules/turbulence.md`.

> **Provenance boundary (binding).** This file cites the C++/Fortran precursors to inform
> *your* understanding — that is its job, and why it loads into your
> context. Per the root `CLAUDE.md` **Comment Convention**, none of that provenance may
> reach the shipped surface (`.py` comments/docstrings, `docs/`): cite the *math*, never the
> reference code, the `.claude/` rules, the design notes, or the author's own papers.

Named, swappable, independently tested numerics: face interpolation, gradient
reconstruction, non-orthogonal correction, (eventually) Rhie–Chow. Governed by the
root `CLAUDE.md` Engineering Principles.

## ⚠️ The swept gradient sets the JACOBIAN'S STENCIL, and only on a skewed mesh (measured 2026-08-16)

**`CorrectedGreenGauss` solves `A_g G = B phi` by Richardson sweeps (`SweptGradientSolve`, four by
default), and every sweep extends the gradient's stencil by one ring — so the coupled residual reaches
`sweeps + 1`.** That is invisible on a rectilinear mesh and decisive on a skewed one, because the
sweep coupling is weighted entirely by the skewness offset `D_g,ip = x_f - (x_P + g*d)`: where it
vanishes, `A_g` is diagonal, sweeps two onward add exactly nothing, and the scheme degenerates to
compact Green-Gauss at reach 1.

Measured on the two validation meshes, with **identical schemes** on both:

| mesh | median skew | max skew | interior faces > 1e-10 | probed Jacobian exact at |
|---|---|---|---|---|
| `pitzdaily_openfoam` | 2.2e-09 | **7.5e-02** | 20049 of 24170 | **reach 5** |
| `bfs3d_openfoam` | 7.0e-15 | 1.9e-12 | **0 of 66368** | **reach 3** |

Confirmed four ways: both reach ladders; the same 2D case at `sweeps=1` floors at reach 3 exactly as
the 3D one does; a one-hot column probe (immune to colour aliasing) shows one ring per sweep; and a
scheme-level isolation on synthetic meshes gives gradient reach 1/1/1/1/1 rectilinear against 1/2/3/4/5
skewed. The error a short reach leaves is carried by the **pressure column**, which is what one would
expect — pressure enters the residual only through gradients, so it inherits the extended stencil
undiluted.

**⚠️ CONSEQUENCE (binding for any new case): `stencil_reach = 3` is a property of a SKEW-FREE MESH, not
of this discretization.** A case on a genuinely skewed mesh needs `sweeps + 1`, in three dimensions as
much as in two. `bfs3d` gets 3 for free because its blockMesh is rectilinear to roundoff; do not read
its value as a default. Check with `jacobian_relative_error` on the case's own mesh — it costs a minute
and the failure it prevents is a preconditioner built on a matrix that is not the Jacobian.

## ⚠️ THE SWEEPS ARE A `lax.scan`, AND THAT IS A SCALING DECISION (2026-08-22)

`SweptGradientSolve` ran its sweeps as a Python `for`, which emits one copy of the operator apply per
sweep — so the **compiled program** grew with the sweep count. That compounds in the Hessian-corrected
scheme, where the inner solve runs once per outer apply and the program is `outer × inner` applies.

**It is a COMPILE-side limit, not a runtime one.** Runtime memory is flat in the sweep count either
way (measured: 11.7–11.8 GB from 2 to 12 sweeps on a 1.6M-cell mesh). What fails is compilation: at
468 applies the scanned form compiles and runs in **230 s at 7.6 GB** while the unrolled one is
**killed by the operating system during compilation**. On this case that wall sat between 240 and 320
unrolled applies, which is inside the range a real calibration asks for.

Compile time with the scan is **flat at 0.4 s across 50, 200 and 450 applies** (8000 cells).

- **`sweeps` stays a static field** — that is exactly what `length` wants. It never becomes a tracer,
  so `narrow_gradient_sweeps` and the calibration keep working on it unchanged.
- **The peel stays outside the scan**, and a single sweep short-circuits before it. ⚠️ `lax.scan`
  **traces its body even at `length=0`**, so without that short-circuit a one-sweep solve traces an
  operator it never applies.
- **The carry is `(x, residual)`, not just `x`.** The `warn_tol` diagnostic reads the residual that
  *formed* the final update — free, one apply already spent — and a carry of `x` alone silently drops
  the only warning a user gets that their sweep count is short for their mesh.

**⚠️ IT IS NOT BIT-IDENTICAL, and do not record it as such.** XLA contracts the multiply-adds
differently in a scan body than in an unrolled chain — the same mechanism as the `dot` entry in the
root briefing. Measured: **5.4e-17 relative**, i.e. one unit in the last place, **exactly zero at
`sweeps=1`**, and **flat from three sweeps onward rather than accumulating**. That flatness is the
property worth checking on any future change here; a drift that grew with depth would be a defect.

The two tests that had to change lens for the scan (apply count read from the jaxpr trip count; a scanned reference for the bit-identical peel) are recorded in `.claude/notes/schemes-log.md`, "The sweeps are a `lax.scan`".

## ⚠️ THE JACOBIAN'S GRADIENT CAN BE CHEAPER THAN THE RESIDUAL'S, AND ON pitzDaily THAT IS ~10 % OF A MARCH FOR NOTHING (measured 2026-08-22)

The coupled march solves each shifted Newton system to `forward_rtol = 0.3` — a deliberately
30 %-accurate step — while the gradient reconstruction *inside* that residual is solved to near machine
precision, and `jax.jvp` differentiates through the same sweep count. So the expensive gradient is paid
again on **every matrix-vector product**, of which a step takes many, to feed a step that is then taken
inexactly.

**That is only inconsistent if the two accuracies buy the same thing, and they do not.** The
reconstruction inside `R` decides *which discrete equations are being solved*: loosen it and the root
moves (with a fixed sweep count `R` is still exactly linear, deterministic and history-free, so Newton
converges perfectly well — just to the root of a slightly different discretization). The reconstruction
inside `J` decides only how fast the inexact-Newton iteration reaches whichever root `R` defines, which
is the same latitude the `0.3` already takes. **`R` determines the answer; `J` determines only the rate.**

**BUILT: `jacobian_gradient_sweeps` on all four coupled builders** (`coupled_continuation`,
`coupled_lu_continuation`, `coupled_amg_continuation`, `mass_flow_coupled_continuation` — all four since
folded into `coupled_step`, the last in #541, which borders the stand-in for a held bulk velocity) →
`_coupled_step` → `ShiftedStep.jacobian_residual` → `_shifted_solve(jacobian_fn=…)`. It narrows the
gradient's sweeps in the copy of the residual the Krylov **operator** is differentiated from, via the
same `narrow_gradient_sweeps` the probe already uses. `None` everywhere is byte-identical. It is a
**third** thing to narrow and the three must not be conflated: `sweeps` is the residual's (the
discretization), `probe_gradient_sweeps` is what the preconditioner *materializes*, and this is what
the Krylov iteration *applies*.

**Current status:** the per-matvec cost table, the whole-march ladder (pitzDaily: swept-2 is free, the knee is at 2, swept-1 is past it) and the residual-sweep ladder (the residual's exactness bought nothing measurable at `rho = 5.07e-03`, which is NOT a licence to lower the shipped `sweeps`) are in `.claude/notes/schemes-log.md`, "The Jacobian's gradient can be cheaper than the residual's". Not measured: a genuinely skewed small case.

- **⚠️ `narrow_gradient_sweeps` SILENTLY DROPPED `relaxation` — fixed 2026-08-22.** It rebuilt the node
  as `SweptGradientSolve(sweeps=…, warn_tol=…)`, so an under-relaxed solve came back at the class
  default `1.0`. Latent while both shipped cases run undamped, and a real hazard the moment this
  function stopped being probe-only: on a mesh skewed enough to need the damping the undamped
  Richardson iteration **does not converge at all**, so the narrowed copy would diverge where the
  original was fine, with nothing to say a setting had been lost. A narrowed copy must differ from its
  original in the sweep count and in nothing else; pinned by
  `test_narrow_gradient_sweeps_carries_the_relaxation`.

## Responsibility
- Reconstruction/interpolation/gradient/**limiting** **strategy classes** (each an `equinox.Module`
  implementing a scheme `Protocol`), each a **small single-responsibility class with a
  known order of accuracy**, unit-tested in isolation (reconstruct a known analytic
  field, check the convergence rate). All physics-free numerics live here — including the slope
  limiter — so the dependency stays one-way `discretization → schemes` (an operator/scheme injects
  a limiter; nothing in `schemes/` imports up into `discretization`).
- **`limiter.py` — BUILT.** `Limiter` (interface) → `VenkatakrishnanLimiter(softening, scale)`: a
  per-cell slope limiter `psi ∈ [0,1]` (smooth Venkatakrishnan 1993), `limit(field, context)`
  taking the shared `aquaflux.context.FieldContext` (issue #280 step 4 — reads `context.gradient`
  for this field's own cell gradient and `context.mesh.face_cells` / `context.mesh.geometry` for the
  connectivity and metrics; reads no boundary value and no property, so a `MeshContext` would also
  suffice, but it takes the full context because that is what its caller, `LimitedUpwind`, already
  holds). Physics-free (verified in `tests/unit/test_limiter.py`), injected into
  `LimitedUpwind(limiter=…)` in `discretization/advection.py`, and evaluated only when that scheme
  runs (a diffusion-only or first-order solve never forms `psi`). See `.claude/rules/discretization.md`.
  **`softening` is an ordinary pytree leaf, not a static field (#368, 2026-09-29, when it was `k`)**:
  pinned (leaf membership and `eqx.filter_grad` against a central difference on a step) by
  `test_the_softening_constant_is_a_differentiable_leaf`. Under `filter_jit` a plain Python-float
  softening is still hashed as static; pass an array to trace it.
  - **⚠️ THE SOFTENING IS `eps = softening · scale`, NOT `eps² = vol K³` (#144, 2026-10-09). There is
    no `k` field any more** (it was renamed `softening` with the change of meaning; default **0.05**,
    the low end of the 0.01–0.2 range the field-range form is used with). `vol K³` added a volume to
    a squared field increment, so `K` meant nothing portable: measured in the issue on pitzDaily
    (`u_x` in m/s, K=5), `eps²/headroom²` had median 0.57 but was < 0.01 in 15 % of cells, and K=100
    moved the median to 4.5e3. `scale` is the field's reference magnitude (`phi_ref`, the field's
    units), a floating array leaf (a new value is a cache hit), **not a case-file setting**
    (`not_settings`). The regularization of a vanishing `d-` is `1e-12 · scale`, for the same reason.
  - **The scale is a fixed property of the PROBLEM, deliberately not the iterate's range** (the
    Venkatakrishnan–Wang `eps = K (q_max − q_min)` proposed on the issue). Reading the range from the
    state changes the residual as the march develops (a uniform start has range 0, so the first steps
    run a strict limiter), and its exact derivative couples every row to the two cells holding the
    extrema; a `stop_gradient` on it instead makes the Jacobian inexact, so Newton loses quadratic
    convergence and the IFT adjoint drops a term wherever the limiter is active. A fixed scale also
    needs no global reduction in the distributed residual. Decided with the project owner.
  - **Who supplies it: `with_reference_scale(provider)` on `Limiter` and on `AdvectionScheme`**
    (default: return self; `LimitedUpwind` forwards to its limiter; Venkatakrishnan fills an UNSET
    scale only — a stated one wins). `provider` is a zero-argument callable so the magnitude is
    derived only for a scheme that reads one: first-order, unlimited and Stokes builds in a problem
    with no derivable magnitude are not refused. Each assembler binds it for its own field:
    `MomentumContinuity.build` (the flow's `reference_speed`, `flow.md`), `SSTTurbulence.build`
    (per field, `k_advection_scheme` / `omega_advection_scheme`, from `velocity_scale=`;
    `turbulence.md`), `ScalarTransport.build` (`prescribed_range`, `transport.md`). The generic
    `ResidualAssembler` binds nothing — a caller using it with this limiter states `scale` (the
    tests do). An unset scale at `limit()` raises naming the remedy.
  - **Invariance is pinned directly**, the test gap the issue named: the same field in units 1000×
    smaller with its scale likewise, and the same problem on a mesh 1000× larger, give `psi` equal to
    `1e-10` (`test_psi_does_not_depend_on_the_units_of_the_field` / `..._size_of_the_mesh`, on a
    `tanh` step chosen so `psi` is in its softened range). Mutation-checked 2026-10-09: restoring the
    volume form fails both; dropping `eps²` from the numerator or the denominator, or ignoring
    `softening`, each fails the softening test.
  - pitzDaily under the new softening (2026-10-09) is recorded in `.claude/notes/schemes-log.md`, "Responsibility → `limiter.py`": the limiter is all but inactive there at `eps` = 0.5 m/s.
  - **⚠️ The periodic-seam test changed fixture from a cosine to a sine.** A cosine peaks ON the seam,
    and with a softening that no longer swamps the field a smooth extremum is limited by design (its
    headroom is ~0): `psi` there was 0.064 whatever the seam did. With the sine (monotone across the
    seam) the fixed code gives 0.947 and the reverted seam bug 0.11, so the test still discriminates.
  **The per-face unlimited increment gathers its neighbour-side position through
  `face_cells.neighbour_centroid`, not by indexing the cell centroid directly (issue #143, fixed
  2026-09-09).** A raw `cell_geometry.centroid[neighbour]` is the neighbour's true position, which
  across a periodic seam sits a full domain length from the face rather than one cell width — the
  same displacement bug `neighbour_centroid` exists to prevent everywhere else, and the one
  `LimitedUpwind.face_value` already avoided. `face_limiter` now takes the per-face position as an
  argument (`cell_geometry.centroid[owner]` on the owner side, `face_cells.neighbour_centroid(...)`
  on the neighbour side) instead of deriving it from the cell index internally, so both spend the
  same displacement the reconstruction applies. Latent until now: no shipped case pairs a periodic
  mesh with a limiter (the periodic cases run unlimited or first-order advection).
- **`boundary_closure.py` — BUILT (2026-10-09, #58). `BoundaryClosure(values)`: one scalar field's
  boundary face values as a function `(field, gradient) -> (n_faces,)`, and the one home of
  everything a reconstruction needs from them** — `reconstruct(scheme, field, mesh, geometry, *,
  operator_hook, imposed)` (the closures' constant part at a zero gradient into the scheme, plus
  `boundary_gradient_weight` — which a Green–Gauss scheme absorbs, making that constant part exact
  rather than leading-order — and `boundary_values_at` for a later pass that differentiates a boundary
  value; returns the gradient and the values at it; `scheme=None` → a zero gradient), `value_weight` / `gradient_weight` (the `jvp`
  probes), and `linearization(n_cells, dim)` (both at rest — the `BoundaryLinearization` `bind`
  takes). `ResidualAssembler` wraps `boundary_values`; `MomentumContinuity` wraps the pressure and
  a per-component view of each velocity component. Before it, those were written three times
  (scalar, velocity, pressure) and the flow copies had drifted — no `operator_hook`, no `imposed`.
  A frozen dataclass, not an `equinox.Module`: it holds a callable closing over its assembler and
  lives for one method call, never on a pytree. **When a new caller reconstructs a gradient against
  boundary closures, wrap them in this; do not write a fourth copy of the two passes.**
- **`projected_stencil.py` — BUILT (2026-09-22). `ProjectedStencilGradient`: the weights are CHOSEN,
  and that is the whole point of it.** Every linear reconstruction is per-cell weights over a stencil;
  exactness for quadratics fixes only ten of the ~13 numbers per gradient component on a tetrahedron's
  two-hop stencil, so every scheme picks the remainder — most of them implicitly, and one of them
  badly. This scheme takes, of the exact weights, the ones nearest `blend` × a reference that damps
  (`P_g⁻¹ b_g`: a Green--Gauss face pass under the gradient equation's own per-cell block, which is
  `HessianCorrectedGradient`'s system's first iterate from a zero start). `blend=0` is the minimum-norm end, which IS an unweighted quadratic least-squares fit
  on the stencil — the same operator `w = B(BᵀB)⁻¹e`, i.e. the gradient of a k-exact (k=2)
  reconstruction, Barth & Frederickson (AIAA 1990); pinned against an independently built fit by
  `test_the_minimum_norm_end_is_an_unweighted_quadratic_least_squares_fit`, which is the only test in
  that file sensitive to *which* exact weights are chosen (a 0.2 % departure fails it and nothing
  else). ⚠️ **So the novelty is the blend and the boundary rows, not the scheme** — and `blend=0`
  alone already fixes the tetrahedral damping, which is what a paper claim would have to survive.
  `blend=1` is the nearest exact weights to the reference. Built once per field in
  `bind`, applied as one gather and one contraction — no solve, no iteration, and the reach is the
  multiple-correction scheme's, so a Jacobian gains nothing.
  - ⚠️ **The reference is a TARGET, not an ingredient, and that is what makes a cheap one legitimate.**
    The exact weights form an affine set; the projection lands in it whatever it aims at, so the target
    cannot affect exactness and decides only *which* member — i.e. the magnitude, and through it the
    damping sign. Reasoning that a low-order reference must give a low-order scheme is the error to
    avoid here; the target is never summed into a reconstructed gradient.
  - **The target carries no local Schur correction (dropped 2026-09-23) and `boundary_weight` was deleted with it**; the measurement and which records predate the change are in `.claude/notes/schemes-log.md`, "Responsibility → `projected_stencil.py`".
  - **Why it exists: exactness and weight SIZE are independent, and only the second decides the
    Rhie–Chow sign.** On the tetrahedral duct the multiple-correction scheme is exact for quadratics
    and its worst weight is ~2900× the reference's; it flips 117 damping diagonals (geometry binding;
    24 under the pressure binding, eigenvalue `+2.17`). Every exact weight set of order one measured
    on that mesh flips none. `tests/unit/test_projected_stencil.py` pins the size gap on the 162-cell
    tetrahedral fixture (145 against 7.9) — the damping itself is a case-level measurement
    (`validation/tetrahedral_gradient_ab/rhie_chow_sign_probe.py`, `RC_SCHEMES=projected-<blend>`).
  - **The boundary datum is per FIELD, from the linearization `bind` already receives** (#483): a face
    whose value is prescribed constrains the stencil with a monomial's *value*; one whose value
    extrapolates constrains it with the monomial's *normal derivative*, recovered at run time from the
    boundary value evaluated at zero gradient, `(bv − φ_owner)/(d·n)`. ⚠️ **A Robin or convective face
    is REFUSED, not approximated** — its datum is neither, and no stencil constraint expresses the
    mixture yet. The row it needs is the mixed functional the condition imposes,
    `∂ₙq + (h/Γ)q(x_f)` against datum `(h/Γ)T∞`, whose coefficients are recoverable from the
    linearization alone (`β = 1/value_weight − 1`, `h/Γ = β/(d·n)`); that derivation is on paper and
    unverified, which is why the guard ships instead of the branch. Tracked as #496.
  - **It serves `k` and `omega` too, and that took a turbulence-side change.**
    `turbulence/transport.py::_assembler` builds a `ResidualAssembler` *inside* each residual
    evaluation, so a scheme was bound against a **traced** mesh -- which a stencil built from the
    connectivity cannot be (it raises, naming the cause; `MultipleCorrectionGradient` survives that
    path because its whole bind is jnp). `SSTTurbulence` now binds once per field at build
    (`k_gradient_scheme` / `omega_gradient_scheme`, the shape momentum has had since #469) and
    `ResidualAssembler.build` takes `bind_gradient_scheme=False` for a caller that has already bound.
    The two bindings genuinely differ: a wall prescribes `k` and lets `omega` extrapolate. Pinned by
    three tests in `tests/unit/test_sst_transport.py`, mutation-verified — including that each
    *equation* reads its own field's binding, which needs a frozen closure and a perturbed mesh to
    isolate (the blending function reads both gradients, and an orthogonal mesh's k equation does not
    read its gradient at all).
  - **Current status of the case measurements (2026-09-22, one run each):** it marches the tetrahedral duct (laminar and coupled RANS) where the multiple-correction scheme fails, reproduces pitzDaily's and bfs3d's baselines, and on bfs3d its conditioning guard fires on ~1000 thin near-wall cells (informative, not refusing; no blend changes it). Tables in `.claude/notes/schemes-log.md`, "Responsibility → `projected_stencil.py`".
  - **Not under domain decomposition** (two hops past a one-deep halo; raises), and the weights are
    constants after binding, so binding inside a *geometry* differentiation gives a wrong shape
    derivative — the same caveat `HessianCorrectedGradient.bind` carries.
  - ⚠️ **`blend`'s default 0.75 came from ONE mesh** (the tetrahedral duct, under its pressure
    conditions): every blend below 1 flipped no diagonal there while 1 flipped 59, and accuracy rose
    monotonically with the blend. Re-measure with the case probe before trusting it on a mesh unlike
    that one.

- **`gradient.py` — BUILT so far:** `GradientScheme` (interface) → `CompactGreenGauss`
  (one-shot `∇φ_P = (1/V_P) Σ φ_ip S_f`, linear-interpolated interior faces). Verified in
  `tests/unit/test_gradient.py`: linear-exact + 2nd-order on orthogonal grids;
  **inconsistent (order ~0) on irregular grids** — the deliberately-demonstrated
  Green–Gauss deficiency. Differentiable (`jax.grad` flows).
  - **`GradientScheme.gradients`/`_reconstruct_gradient` deliberately do NOT take
    `aquaflux.context.MeshContext` or `FieldContext` (decided, issue #280 step 4 — closed as not
    applicable here, unlike `schemes/limiter.py`'s `Limiter`, which does).** Two independent reasons,
    either alone sufficient:
    1. **It needs mesh topology `MeshContext` does not carry.** `HessianCorrectedGradient`'s face
       warp-moment correction reads `mesh.node_coords` / `mesh.face_nodes` directly (`_HessianSystems`,
       `face_geometry_scheme(dim).warp_first_moment(...)`) — not an edge case, but the term that
       recovers ~13 orders of accuracy on a warped mesh (see the warp-moment entry above). `MeshContext`
       carries `{face_cells, geometry, properties}` only; growing it to also carry node/face-node
       connectivity for this one consumer would be exactly the "union bundle that grows with each new
       operator" the Module Review Rubric warns against, since no other `MeshContext` consumer
       (`DiffusionFlux`, `AdvectionFlux`, `Limiter`) touches mesh topology beyond `face_cells`.
    2. **It is called before a `FieldContext` for this field can exist.** `gradients()` *produces* the
       gradient that `ResidualAssembler._context` later packs into `FieldContext.gradient` — handing it
       a `FieldContext` would be circular (the context it would read from is the one it is building).
       Its `boundary_values` argument is also semantically the **leading-order** closures (evaluated at
       zero gradient), not a `FieldContext`'s final ones — a second reason the shapes do not coincide,
       independent of the first.
    A `MeshContext`-shaped argument was considered and rejected on point 1 alone; `bind(mesh,
    geometry)` (build-time preparation, before `PropertyModel` is even evaluated) was never a candidate
    for either context type, for the same underlying reason: this scheme's inputs are earlier and wider
    than what the shared context is scoped to carry.
- **`CorrectedGreenGauss` — BUILT.** The non-orthogonal correction makes the gradient a
  *sparse coupled system* `A_g·G = B·φ` (`A_g` geometry-only, well-conditioned). **How `A_g⁻¹` is
  applied is an injected `GradientSolve` strategy** — `SweptGradientSolve` (**default**; fixed
  matrix-free Richardson sweeps) or `GmresGradientSolve` (matrix-free `lineax` GMRES, differentiable
  by implicit diff, for a mesh skewed enough that the swept sweep count would grow impractically); the
  discretization (`terms`/`operator`/`rhs`) is identical either way, so the swept path is **not** a
  separate scheme (it was `SweptCorrectedGradient`; retired). **Default is swept, not GMRES (binding,
  changed after the pitzDaily study).** GMRES is a *nested Krylov solve carrying its own implicit-diff
  tangent*, re-entered on every reconstruction; inside a nonlinear (coupled RANS) Newton it is
  re-differentiated by every Jacobian–vector product, which measured **2.16 s vs 0.014 s per
  coupled-residual eval** on the ~12k-cell pitzDaily backward-facing step (≈180×) — impractical for a
  real solve. The swept apply carries no nested solve, so it stays cheap under AD. The discretization-
  *exactness* unit tests (linear-exact, Gate-C one-step, the swept-vs-exact reference) pin
  `GmresGradientSolve()` explicitly, because they assert machine-precision properties the fixed
  `sweeps=4` default does not reach on a skewed mesh (it is exact only to within discretization error
  there — the swept solver's own accuracy is a separate, dedicated test). Verified:
  **linear-exact on irregular grids** (fixes compact GG's inconsistency), reduces to compact
  on orthogonal, but **measured to cap near 1st order** on irregular grids (the DeGroot-2019
  wall). **Now consumed by the diffusion residual** (`discretization/residual.py`) as the
  injected default for the non-orthogonal correction: folding it into `R` and letting AD form
  the Jacobian gives Gate C's one-Newton-step, linear-exact solve on a skewed mesh
  (`tests/integration/test_skewed_diffusion.py`).
- **`HessianCorrectedGradient` — BUILT (2D + 3D). The 2nd-order scheme, and the AD+Schur showcase.**
  Betchen's coupled gradient+Hessian reconstruction, with the **Hessian Schur-eliminated**
  (`schur=True` default) so only the gradient is the primary unknown: `S·g = b_g`,
  `S = A_gg − A_gH·A_HH⁻¹·A_Hg`, `A_HH` geometry-only, applied by an inner **fixed-sweep** solve
  (2026-08-21; it was a nested `lineax` GMRES before — see the block-preconditioner entry below).
  **All blocks come from AD** — the residual is the *forward* reconstruction (interpolations +
  Green–Gauss sums), never the paper's hand-derived coefficient matrices (Eq. 23–25). Verified:
  **exact for linear AND quadratic** on irregular grids (removes the cap), 2nd-order for smooth
  fields, **Schur result == full coupled solve to machine precision** (the elimination is
  exact), well-conditioned (~4 GMRES steps), differentiable through the nested solve. This is
  the drop-in `A_g`/`B` that later Schur-couples into the flow Newton (Hessian pre-eliminated,
  so the flow's inner block is gradient-sized). **Dimension-general (2D + 3D)**: the gradient's
  Hessian term uses Betchen & Straatman's Eq. (7) form — `−½ H:(x_ip−x_P)⊗(x_ip−x_P)` per face, from
  each cell's centroid-to-face vector (`_hessian_moment`) — so **no explicit face second-moment
  tensor** is needed. (An earlier 2D-only build evaluated it with an explicit edge second moment
  `(L³/12)(I−n̂n̂)` — the home-grown detour that caused the 2D limit; retired after checking the paper.)
  The 3D skew test skews a hex grid **in-plane** (`tests/support/meshes.py::columnwise_perturbed_grid_3d`,
  planar faces): exact-for-quadratic vs `CorrectedGreenGauss`'s ~0.08 error.

  **⚠️ THE HESSIAN IS SOLVED AS SIX COMPONENTS, NOT NINE (2026-08-22).** The Hessian of a
  twice-continuously-differentiable field is symmetric, so `n_sym = dim(dim+1)/2` components carry
  it — six in 3D. Solving only those leaves `dim²` equations for `n_sym` unknowns, and the surplus is
  removed **in the least-squares sense weighted by the cell's own block**, which is what Betchen &
  Straatman (2010) do (their Eq. 16–17): the reduced row is `(A_P E)ᵀ` applied to the full row, `E`
  the expansion of the packed components. `contract_symmetric`'s adjoint identity
  `<E u, R> = <u, contract(R)>` turns that into `contract(R A_P)`, so the `(dim², n_sym)` matrix is
  never formed.

  - **The reduced per-cell block is `(A_P E)ᵀ(A_P E)` — SYMMETRIC POSITIVE DEFINITE by construction**,
    where the unreduced block is neither symmetric nor definite. Pinned by
    `test_the_reduced_hessian_block_is_symmetric_positive_definite`, which checks the block the
    preconditioner actually inverts rather than one computed a second way. This is the structural
    reason to weight the reduction rather than project it unweighted: `A_HH⁻¹` sits at the centre of
    the elimination term `A_gH A_HH⁻¹ A_Hg`. ⚠️ **That property is NOT what fixes the reactor mesh —
    measured, see the weighting-versus-symmetry entry below.**
  - **⚠️ STORAGE GOES UP, NOT DOWN — the expectation that motivated this is half wrong, and it is the
    per-cell BLOCK that decides it.** The unknown shrinks by a third (9 → 6 doubles per cell, and
    likewise every Hessian iterate on the reverse-mode tape). But the block **stops factoring**: an
    unsymmetrized `H` enters its own equation only as `H·a`, touching one tensor index and leaving
    the other alone, which is what made the block `I ⊗ C` and storable as `(dim, dim)`. Symmetry
    couples the two indices, so it is a dense `(n_sym, n_sym)` — **9 → 36 doubles per cell in 3D**,
    four times the storage. Net at 1.6M cells: about **−38 MB** on the unknown against **+346 MB** on
    the block. Everything else about the change is favourable; this is not.
  - **⚠️ THE REDUCTION MUST BE ROW-SCALED BY `1/vol`.** Weighting by `A_P` squares the equation's
    volume scaling (`A_P ~ vol`, so the reduced block is `~vol²`), and every consumer assuming a
    volume-scaled Hessian row degrades silently by a factor of the cell volume — the packed system's
    inverse-volume preconditioner most of all, which on a real mesh means a factor of ~1e-9. A
    positive per-cell row scaling changes neither the solution nor the block's definiteness, so this
    costs nothing and is not a tuning.
  - **The elimination term becomes an ordinary triple product.** `A_gH`'s block was a rank-three
    tensor `(n, dim, dim, dim)` because the gradient equation contracts *both* of the Hessian's
    indices; against packed components it is a plain `(n, dim, n_sym)` matrix, and the local Schur
    block is `einsum("nma,nab,nbk->nmk", ...)`. Its probe count drops from `dim² = 9` to `n_sym = 6`,
    while the reduced `A_HH` block's rises from `dim = 3` to `n_sym = 6` (the Kronecker shortcut that
    let one row's probe give every row's is gone with the factorization).
  - **The warp moment simplifies.** `½Pᵀ(Hd + Hᵀd)` carried both contractions because the tensor was
    not symmetrized; the two now coincide and the half cancels the doubling.
  - **`A_P` is now built eagerly rather than inside `inner()`.** The reduction is part of the
    *equation*, not part of the elimination, so the un-eliminated path needs it too — which costs that
    check path `dim` probes it did not previously pay.
  - **Quadratic exactness is unchanged** (median 6.0e-15 at 2D perturb 0.3, 7.6e-15 at 3D perturb
    0.25), the elimination still agrees with the un-eliminated solve, and the default 20/10 sweep
    counts still reach the exactly-solved reconstruction — so this is a change of representation, not
    of accuracy.

  **⚠️ NON-PLANAR FACES: the derivation drops a term, and restoring it is worth ~1000× on a real
  automatically-generated mesh (measured 2026-08-21).** Betchen's Eq. (2)→(3) drops the face integral
  of the linear term by "noting that the planar faces of a polyhedral volume possess a constant
  normal", i.e. by taking `∫_face (x−x_ip) n̂ dS = 0`. On a warped face that integral is **not** zero,
  and it is first order in the warp — where the Hessian correction the scheme exists to apply is
  second. `FaceGeometryScheme.warp_first_moment` computes it exactly from the centre-fan triangles
  (each planar, so `(g_t−x_ip) ⊗ S_t` per triangle is the exact integral — no quadrature), and the
  gradient face kernel carries two terms from it: `+∇φ·P` (first order, dominant) and
  `−½Pᵀ(Hd+Hᵀd)` (second order, from correcting Eq. (5)'s companion assumption).
  Measured on a quadratic, median per-cell relative gradient error:

  | mesh | planarity | Betchen before | Betchen after | `CorrectedGreenGauss` |
  |---|---|---|---|---|
  | orthogonal | 1.0000 | 8.37e-16 | 8.38e-16 | 9.40e-16 |
  | planar-skew, perturb 0.3 | 1.0000 | 5.91e-13 | 5.91e-13 | 4.65e-03 |
  | warped, perturb 0.3 | 0.8875 | 4.07e-02 | **8.61e-15** | 3.70e-02 |
  | warped, perturb 0.4 | 0.6773 | 6.15e-02 | **4.12e-13** | 5.55e-02 |
  | UV reactor, 1.6M cells | 0.8770 | 1.723e-04 | **1.093e-07** | 1.634e-04 |

  Planar meshes are **bit-identical** (`P ≡ 0` there), so this is a correction and not a tuning.
  Before it, Betchen was *worse than the scheme it replaces* on a warped mesh — 300× the cost for no
  gain — which is the state a 1.6M-cell snappyHexMesh reactor was actually in.

  **⚠️ AND THE FIXTURE CHOICE IS WHY THIS HID FOR SO LONG — the entry here used to say a warped grid
  "breaks Green–Gauss exactness for *every* scheme", and used that to justify testing only in-plane
  skew.** The premise is true of the *uncorrected* derivation and false of the scheme once the moment
  is restored, so a statement about a defect was read as a law of nature and became the reason never
  to test the case that would have exposed it. `perturbed_grid_3d` (which warps its quad faces) is
  now tested directly, and the test is verified to FAIL without the correction (4.07e-02 against a
  1e-3 threshold) — a regression test nobody has watched fail is not known to test anything.

  **⚠️ THE MOMENT MUST BE ORIENTED TO THE CALLER'S NORMAL, and getting that wrong looks like the term
  not mattering.** The centre fan's triangle normals follow the node winding — the *unoriented*
  convention `orient_owner_outward` exists to fix — so an unoriented moment is sign-flipped on roughly
  half a real mesh's faces. It does not blow up; it makes the answer slightly *worse* (4.07e-02 →
  4.53e-02), which reads exactly like "this term is not the problem" and nearly closed the
  investigation. A magnitude-only check cannot see it: `P` validated perfectly as a warp detector
  (machine-zero planar, 2.2e-02 warped) while carrying the wrong sign. Pinned by
  `test_warp_first_moment_follows_the_supplied_normal`.

  **BOTH EQUATIONS NEED IT, and the `Q` second moment does NOT.** Three terms were candidates and only
  two were real:
  - **The gradient equation's `+∇φ·P`** — first order in the warp, and the dominant one.
  - **The Hessian equation's `+H·P`** — it is *itself* a Green–Gauss sum (of the gradient), so it
    inherits the identical assumption. Worth ~2× on the fixtures and **28× in the max on the reactor
    mesh** (1.751e+00 → 6.245e-02): the pathological tail was largely this, not solver convergence.
  - **The `Q` second moment needs no correction at all.** Betchen's Eq. (5)–(6) rewriting of it rests
    on `Σ_f ∫ x_i x_j n̂_k dS = 0` in cell-centroid coordinates, which is the divergence theorem over
    the cell and holds for *any* polyhedron. Verified numerically alongside cell closure: both hold to
    ~1e-15 at planarity 0.83, so the identity is exact on a warped mesh and only the `d_i P_jk` term it
    generates has to be carried.

  **⚠️ AND THE WARP TERM'S GRADIENT MUST BE THE ONE AT THE FACE CENTROID.** Contracting `P` against the
  raw interpolated blend instead of the skewness-carried value — and skipping the boundary branch —
  leaves the reconstruction at 9.3e-05 rather than 8.6e-15. That is *visibly better* than the 4.1e-02
  it started from, and entirely plausible as "the residual error the method has on a warped mesh", so
  it is a comfortable place to stop and write up a wrong limit. Both equations now take the value from
  one `_face_gradient` rather than two spellings of it.

  **⚠️⚠️ A SLIVER FIXTURE MADE BY SQUASHING A NODE BAND IS AN INVALID MESH, AND EVERY MEASUREMENT
  TAKEN ON ONE IS VOID (2026-08-22).** Moving a whole plane of nodes leaves cells whose face
  area-vectors no longer sum to zero: measured closure residual **6.6e-01** against 1e-16 for a valid
  mesh, at a planarity of 0.068. Green–Gauss **is** the divergence theorem, so on such a mesh the
  operator is wrong and no solver recovers it — an exact Krylov solve leaves 3e+01 where the same
  solve on a valid sliver reaches 1e-11. It is the natural fixture to write, it looks like a sliver,
  it reports a plausible volume ratio, and it fails silently.

  **Squash ONE cell instead — only its four top nodes — so every cell stays closed**
  (`tests/unit/test_gradient.py::_sliver_mesh`, pinned valid by
  `test_the_sliver_fixture_is_a_valid_mesh`). `closed_cell_residual` is the check;
  **`face_planarity` is NOT a substitute** and does not reliably flag it. This cost a whole line of
  investigation: a "one sliver contaminates the far field by 11 orders" finding, an "error plateaus
  rather than diverging" robustness claim, and a three-scheme robustness comparison were all measured
  on the invalid fixture and are **WITHDRAWN**.

  Robustness on a valid single-cell sliver (an iteration effect, not a discretization one) is measured in `.claude/notes/schemes-log.md`, "Responsibility → `HessianCorrectedGradient`: slivers and the local Schur block".

  **`CorrectedGreenGauss` DIVERGES on a sliver under its default preconditioner, and the fix is an
  opt-in (2026-08-22).** `InverseVolume` is `1/V`, and `A_g`'s neglected coupling scales with face
  area while the volume does not, so as a cell flattens the preconditioner stops approximating the
  block at all. Sliver-cell error against a known analytic gradient, valid single-cell fixture:

  | volume ratio | `InverseCellVolume` (default) | `ExactCellBlock` |
  |---|---|---|
  | 1.9e+02 | 3.37e-02 | 3.49e-02 |
  | 1.9e+04 | 2.56e+02 | **1.73e-02** |
  | 1.9e+06 | 1.69e+10 | **1.74e-02** |
  | 1.9e+08 | **1.68e+18** | **1.74e-02** |

  Unbounded, against flat at the scheme's own discretization error. Clean meshes agree to four
  significant figures, so this is robustness and not accuracy. **The default stays
  `InverseCellVolume`** — ~3× cheaper at four sweeps (0.40 ms against 1.33 ms at 4096 cells) and
  correct on any mesh of reasonable quality; `ExactCellBlock` is the opt-in for a mesh with slivers or
  a case that diverges, and is documented for users in `docs/gradient_reconstruction.md`.
  ⚠️ Flooring `1/V` is **not** an adequate substitute: it converts an unbounded divergence into a
  bounded 3.7e+07, six orders worse than the block.

  **⚠️ THE OUTER PRECONDITIONER CAN BE BUILT FROM THE WRONG BLOCK, and `local_schur_block` — now the
  default — is the fix (2026-08-22).** The outer operator is the Schur complement
  `S = A_gg − A_gH A_HH⁻¹ A_Hg`, but the preconditioner takes `A_gg`'s per-cell diagonal block and
  ignores the elimination term entirely. On a well-shaped cell that term is a ~9 % perturbation and
  dropping it is harmless; on a flattened one the volume vanishes while the face couplings do not, so
  the neglected term becomes the **dominant** part of that cell's row and the sweep stops converging
  there. `local_schur_block=True` replaces each factor by its own per-cell block and contracts the
  three cell by cell — the approximation a pressure Schur usually gets.

  | mesh | `A_gg` block (default) | local Schur block |
  |---|---|---|
  | orthogonal | 8.230e-16 | 8.321e-16 |
  | planar-skew, perturb 0.3 | 1.344e-14 | **4.468e-15** |
  | warped, perturb 0.3 | 1.051e-14 | **9.123e-15** |
  | sliver 1.9e4 — cell / far | 8.05e-01 / 2.72e-07 | **1.42e-05 / 9.76e-11** |
  | sliver 1.9e6 — cell / far | 1.09e+00 / 4.94e-07 | **1.55e-02 / 1.40e-08** |

  **Current status:** `local_schur_block=True` is the default and `False` is the dominated control. On the 1.6M-cell reactor mesh the default moved True → False → True in one day (2026-08-22); what settled it is that solving the Hessian as six symmetric components (not the least-squares weighting, which was separated out 2026-08-23) removes the Schur block's divergence while the `A_gg` block's diverges; the divergent mode sits on ~1 cell (a spectral radius says nothing about whether its cause is whole-mesh); the Schur arm converges undamped and relaxation only slows it; a power iteration over `P⁻¹S` returns the largest-modulus eigenvalue, not the governing one. The full trail with tables is in `.claude/notes/schemes-log.md`, "Responsibility → `HessianCorrectedGradient`: slivers and the local Schur block".

  **THE HESSIAN'S BOUNDARY CLOSURE IS AN INJECTED STRATEGY, AND THE TWO OPTIONS GENUINELY TRADE
  (2026-08-23).** A boundary face has no neighbour to interpolate with, so its gradient is
  extrapolated from the owner and needs a first-order Hessian to carry it. `HessianBoundaryClosure`
  → `OwnerHessian` (default) / `AveragedInteriorHessian`, on `HessianCorrectedGradient
  (boundary_closure=…)`.

  | mesh | `OwnerHessian` (default) | `AveragedInteriorHessian` |
  |---|---|---|
  | tetrahedral, n=3, perturb 0.1 | **inf** | 1.70e-04 |
  | hex 3³, perturb 0.25 | **9.25e-16** | 3.28e-03 |
  | hex 8³, perturb 0.25 | **2.33e-15** | 4.03e-05 |
  | 2D, perturb 0.3 | **2.62e-15** | 4.70e-05 |

  And on the Hessian system's conditioning, which is the point of the second one:

  | mesh | `cond(A_HH)` owner | averaged | `cond(S)` owner | averaged |
  |---|---|---|---|---|
  | tetrahedral n=2 | **2.851e+17** | **8.115** | **4.040e+19** | **5.985** |
  | hex 3³ | 2.084e+01 | 4.362 | 1.995 | 1.788 |

  - **Neither dominates, which is why both are kept** rather than one being deleted as dominated.
    `OwnerHessian` carries the cell's own Hessian — for a quadratic that is the *exact* Hessian, so
    the reconstruction stays exact, which is the property this scheme costs 60× a corrected
    Green–Gauss reconstruction to have. `AveragedInteriorHessian` is Betchen & Straatman's Eq. (28)–(29)
    and is the only one that solves on a tetrahedral mesh.
  - **⚠️ BETCHEN'S AVERAGED CLOSURE GIVES UP EXACTNESS FOR QUADRATICS, and refinement does not
    recover it — but see the entry below: that is a property of HIS eligible set, not of
    averaging, and `AveragedNeighbourHessian` keeps the exactness**
    (3.3e-03 at 3³, 2.0e-03 at 5³, 4.0e-05 at 8³ — falling, but four orders short of the owner
    closure at any size). The cause is the closure's own empty case: where a cell has **no**
    boundary-clear neighbour — a corner, an edge, a mesh too coarse to have an interior — the average
    is zero and the extrapolation drops its curvature term entirely. That is consistent with the
    source, which reports second-*order* gradients (their `gamma_1` = 1.832) rather than exact ones.
    **So this is not a strictly better closure that we were simply missing; it is a different point
    on an accuracy/robustness trade, and defaulting it on would silently cost the scheme its
    headline property.**
  - **⚠️ IT MUST BE PASSED INTO THE FACE KERNELS, NOT COMPUTED INSIDE THEM.** The averaged closure
    reads a cell's *neighbours'* Hessians, and the per-cell blocks are recovered by probing those
    kernels with a **uniform** field — so a gather from neighbours returns the probe's own value and
    is indistinguishable from a diagonal term. Computed internally it would report neighbour coupling
    as diagonal: the "compose the blocks, not the operators" trap, one level further in. Each closure
    therefore declares its own diagonal contribution (`diagonal_probe`), which is what the
    extractions pass — `probe` for the owner closure, zeros for the averaged one, since a cell's own
    Hessian never appears in its own average. Pinned by
    `test_the_averaged_closure_contributes_nothing_to_a_cell_own_diagonal_block`, which checks that
    the prepared map really is non-zero on that probe while the declaration is zero — the two being
    equal is precisely the bug.
  - **The default is byte-unchanged**: `OwnerHessian.prepare` returns the identity and its
    `diagonal_probe` returns the probe, which is exactly what the kernels did before the seam existed.
  **⚠️⚠️ THE ACCURACY LOSS WAS NOT NECESSARY — averaging over ALL face neighbours is exact for a
  quadratic AND leaves the system solvable, which is better than the published closure on the axis
  that matters (measured 2026-08-23).** `AveragedNeighbourHessian`.

  **The principle, and it is worth stating because it makes the whole family predictable:
  a closure is exact for a quadratic exactly when its weights sum to ONE.** A quadratic's Hessian is
  constant, and any weighted average of a constant whose weights sum to one returns it unchanged. So
  `OwnerHessian` (weight 1 on self) is exact; Betchen's is exact wherever its eligible set is
  non-empty and **zero** where it is not, and that zero branch is the entire source of its error.
  Pinned directly rather than through a reconstruction, on a constant Hessian field, by
  `test_a_closure_reproduces_a_constant_hessian_exactly_iff_its_weights_sum_to_one`.

  | closure | `cond(A_HH)` tet | quadratic, converged | quadratic, shipped 20/10 |
  |---|---|---|---|
  | `OwnerHessian` (default) | **1.01e+18** | 6.65e-16 hex / *unsolvable* tet | 9.25e-16 hex |
  | **`AveragedNeighbourHessian`** | **8.39** | **5.95e-16 hex · 2.57e-15 tet · 9.36e-16 2D** | 4.48e-07 hex |
  | `AveragedInteriorHessian` (Betchen) | 8.39 | **3.28e-03 hex · 1.73e-04 tet** | 3.28e-03 hex |

  **Betchen's closure does not improve with a better solve** — 3.28e-03 at 20/10, at 60/20 and under
  an exact Krylov solve alike — which is what identifies it as a discretization error rather than an
  unconverged iteration. The neighbour average moves 4.5e-07 → 2.3e-11 → 5.9e-16 across those same
  three, i.e. it is exact and merely needs the sweeps.

  - **The restriction to boundary-clear neighbours is what costs the exactness, and it is not needed
    for solvability.** Both averaging closures reach `cond(A_HH)` 8.39 on tetrahedra; the wide one is
    four orders more accurate. What actually keeps the system solvable is that a cell's **own**
    Hessian stays out of its own boundary closure — not which neighbours are counted.
  - **⚠️ THE ELIGIBLE SET AND THE EMPTY CASE CANNOT BE MIXED FREELY.** With the *narrow* set, an
    owner fallback re-admits enough of the cell's own Hessian to make the tetrahedral system singular
    again (`cond(A_HH)` **1.7e+18**). So Betchen's zero fallback is load-bearing *for his eligible
    set*, and the wide set is what makes an owner fallback safe. The four combinations were measured;
    only two are viable and they are the two that ship.
  - **⚠️ IT COSTS SWEEPS AT FULL WEIGHT, and that is why it is not the default — but see the
    blend-weight entry below, which recovers almost all of it.** The closure adds neighbour coupling
    to the Hessian system, so that system converges more slowly. Calibrated to `tol = 1e-10`: outer
    falls 15 → 13 while **inner rises 8 → 23** on a perturbed hexahedral grid (9 → 20 in 2D), which is
    roughly `2.3x` the operator applies. At the shipped 20/10 it has not converged and reads 4.5e-07
    where the default reads 9.3e-16 — so **switching the default would silently degrade the shipped
    configuration**, and `calibrated` takes the closure precisely so the counts can follow it.
  - **⚠️ THE INNER SWEEP COST DOES NOT AMORTIZE WITH MESH SIZE — hypothesis raised and REFUTED the
    same hour.** The closure's value is only *read* at boundary faces, so it only couples
    boundary-owning cells' rows, and the obvious expectation is that the cost shrinks as that share
    falls. It does not. Calibrated to `tol = 1e-10` on perturbed hexahedral grids:

    | cells | boundary-owning | owner inner | averaged inner | ratio |
    |---|---|---|---|---|
    | 27 | 96 % | 8 | 23 | 2.88 |
    | 125 | 78 % | 10 | 23 | 2.30 |
    | 512 | 58 % | 9 | 23 | 2.56 |
    | 1728 | 42 % | 9 | 23 | 2.56 |

    **Pinned at 23 while the coupled fraction falls by more than half.** A sweep count is set by the
    operator's slowest mode, and that mode lives in the coupled rows however few of them there are —
    so "only a few rows are affected" is not an argument about cost, here or anywhere else in this
    file. **The reason the preconditioner cannot absorb it is structural:** the inner solve is
    preconditioned Richardson over a *per-cell block*, which is block Jacobi by construction, and
    neighbour coupling is exactly what such a block cannot represent.

  - **✅ BUT A SMALL BLEND WEIGHT RECOVERS ALMOST ALL OF IT, AND COSTS NO ACCURACY.**
    `AveragedNeighbourHessian(weight=w)` carries `(1-w)` of the cell's own Hessian plus `w` of the
    neighbour average. Both parts are partitions of unity, so the blend is one **at every weight** —
    hence exact for a quadratic at every weight, which is the property that makes the parameter safe
    to expose at all (measured 2.1e-15 on tetrahedra and 1.0e-15 on hexahedra across `w` from 1.0 to
    0.1). What the weight actually trades:

    | `weight` | `cond(A_HH)` tet | inner sweeps | quadratic at the shipped 20/10 |
    |---|---|---|---|
    | 1.0 | 8.4 | 23 | 5.1e-07 |
    | 0.4 | 35.7 | 15 | 2.1e-09 |
    | **0.2** | **142** | **11** | **9.8e-12** |
    | 0.1 | ~480 | — | **3.7e-14** |
    | 0.0 | **1.0e+18** | 10 | *unsolvable* |

    **A little coupling is enough to break the degeneracy.** At `w = 0.2` the tetrahedral system is
    comfortably solvable — 142 against double precision's ~1e16 — at 11 inner sweeps against the
    owner closure's 10, and the *fixed-sweep* reconstruction is 9.8e-12 rather than 5.1e-07. So the
    closure's cost is not intrinsic; it is what one pays for maximal decoupling, and most of the
    decoupling arrives long before `w = 1`.

    **✅ AND THE WEIGHT IS CALIBRATED FROM THE MESH, which removes the guess entirely.**
    `AveragedNeighbourHessian.calibrated(mesh, geometry)` measures the Hessian system's contraction
    rate at each candidate weight — the same estimator the sweep counts use — converts each to a
    sweep count, and takes the weight needing the fewest.

    **One objective, no threshold, and it is right at both ends.** The degeneracy is not a subtle
    signal: at `w = 0` on a tetrahedral mesh the measured rate is **15.4**, i.e. the iteration is
    *expansive*, so its count saturates at `cap` and loses to any weight that works. And on a mesh
    where the owner closure closes the system perfectly well, `w = 0` genuinely needs the fewest
    sweeps and is chosen — so the closure reduces to `OwnerHessian` and costs **nothing**, the
    `weight == 0` branch short-circuiting the averaging entirely rather than multiplying it by zero.
    Measured calibrated weights (0.10–0.20 on tetrahedra, 0.00 on hexahedra/2D) are in `.claude/notes/schemes-log.md`, "Responsibility → Hessian boundary closures".

  - **⚠️⚠️ A CALIBRATION MUST NOT DECIDE ANYTHING FROM A MEASUREMENT TAKEN ON A SINGULAR SYSTEM —
    caught by CI after passing locally (2026-08-23).** The weight ladder originally included `0`,
    where this closure *is* the owner closure and, on the meshes that need the closure at all, leaves
    the Hessian system numerically singular. The contraction rate there measured **15.4** on macOS
    arm64 — expansive, correctly saturating at `cap` and losing — and **below one** on CI's Linux
    runners, where it was then chosen as the cheapest option. Same mesh, same seed, same code: the
    rate of a singular system depends on how the platform's linear algebra inverts a near-singular
    block, and nothing built on it is reproducible.

    **The fix is to stop evaluating that point**, not to loosen the assertion: the ladder starts at
    `0.05`, and asking for no coupling at all means naming `OwnerHessian`, which is a choice rather
    than a measurement. It costs nothing — on a mesh needing no decoupling the ladder's smallest
    weight and zero take the same sweep count (10 on hex 5³).

    **Two lessons worth carrying past this closure.** A quantity that is only unstable *in the regime
    you are trying to detect* is the worst possible detector, and it will pass every test on the
    machine you wrote it on. And the assertion had to change shape too: pinning the chosen weight
    pins noise where several candidates sit within a sweep of each other, so the test now asserts the
    chosen closure **solves** — conditioning of the operator, which involves no inversion and is
    stable — rather than which value was chosen.

        **The shipped default is `0.2`**, which is where the trade sits on the meshes measured and is the
    value to use when calibration is not run. ⚠️ It remains one mesh's evidence: the conditioning
    scales roughly as `cond ~ w^-1.75` over the range tested, so a mesh an order more degenerate would
    be four orders worse at `w = 0.1`. That is what the calibration is for — prefer it to the
    default, and do not lower the default on the strength of that table.

  - **With this closure, pair it with `CoupledBlockSweep`, not a bigger inner count**, and judge `CoupledBlockSweep.calibrated` on the reconstruction when it reports its cap (a count equal to the cap is not a measurement). Measurements in `.claude/notes/schemes-log.md`, "Responsibility → Hessian boundary closures".

  - **What this changes about the earlier entry:** "the averaged closure gives up exactness" is true
    of *Betchen's* and **not** of the family. The choice on a mesh the default cannot solve is no
    longer accuracy-versus-solvability — it is sweeps-versus-solvability, which is a far better trade
    and the one to offer a user.
  - **Both averaging closures share one implementation** (`_inverse_distance_average`, taking the
    eligible set and the empty case), so the weighting, the fallback and the diagonal declaration are
    stated once. That matters here because the two differ in exactly two flags and would otherwise be
    a copy-paste pair whose difference is invisible.

  - **⚠️ BOTH CALIBRATION FACTORIES HAD TO TAKE IT, and the signature-parity test is what said so.**
    The closure changes the operator, so a count measured under one closure and returned on a scheme
    carrying another is calibrating the wrong system — the exact failure that factory exists to
    prevent. `HessianCorrectedGradient.calibrated` was the obvious one; `CoupledBlockSweep.calibrated`
    also builds `HessianCorrectedGradient._systems` to measure its coupled error operator, and would
    quietly have measured the default closure whatever the scheme ran. Measured difference on one
    mesh: **6 sweeps under the owner closure against 5 under the averaged one**, so this is not
    theoretical.
    ⚠️ **Exempting the keyword from the parity test would have hidden the drift it was added to
    catch**, since that exemption applies to every factory at once. The test now additionally asserts
    that the two Hessian-aware factories both offer it *and* offer it identically —
    `CorrectedGreenGauss` reconstructs no Hessian and is the only legitimate abstainer.

  **Under `OwnerHessian` the scheme does not solve on an all-tetrahedral mesh** (`cond(A_HH)` ~1e17, in both the nine- and six-component forms); the averaged closures take it to ~8. The diagnosis is in `.claude/notes/schemes-log.md`, "Responsibility → the all-tetrahedral `A_HH` singularity and the reactor divergence diagnosis".

  **Consequence for the test suite:** the fixture now carries a real test —
  `test_the_averaged_closure_makes_a_tetrahedral_hessian_system_solvable` pins both halves, that the
  owner closure leaves `cond(A_HH) > 1e15` there and the averaged one brings it under `1e3`, with a
  hexahedral control so the result is about the cell shape and not about one closure being generally
  better. **"The scheme works on tetrahedra" is true only with `AveragedInteriorHessian`, and then
  only to second order** — the default closure still cannot solve there, which is the trade the entry
  above describes.

  The reactor-mesh divergence diagnosis under the unsymmetrized Hessian (the rate, the refuted mechanisms, Betchen's relaxation, and how a default changed on 8³ meshes that could not see the failure) is in the same log section; it is history now that the Hessian is solved as six components.

  Better on **every** mesh measured — slightly on clean ones, 57000× on a sliver cell — for **~7 %
  forward time and ~17 % peak memory** (`dim` probes for `A_Hg` plus `dim²` for `A_gH`, a fixed
  prologue against ~220 applies at the shipped 20/10). Adjoint agrees with a finite difference to
  2.1e-09. **DEFAULT ON**: this scheme is chosen precisely for meshes a corrected Green-Gauss cannot
  handle, so its default preconditioner should be the one that survives them — and under the
  six-component Hessian it is also the only one of the two that converges there.
  `local_schur_block=False` recovers the historical `A_gg`-only block, which is now the arm that
  diverges on that mesh and is kept as the control the comparison needs rather than as a
  recommendation.

  `A_gH` takes one probe per Hessian unknown; probe sequentially, not under `vmap` (same wall clock, less peak memory). Measurements in `.claude/notes/schemes-log.md`, "Responsibility → `local_schur_block`: probe count and memory".
  **⚠️⚠️ TWO WRONG BUILDS OF THIS BLOCK BOTH LOOKED LIKE THE IDEA FAILING, and that is the lesson.**
  Both produced a plausible answer that was *slightly worse* on clean meshes — which reads as "this
  term does not matter" rather than as a bug, and closed the investigation twice:
  1. **Owner-side columns only.** `cell_diagonal_block` sums an owner-side *and* a neighbour-side
     column, because a cell contributes through faces it owns and faces it neighbours. Taking only
     the owner half drops roughly half of every cell's own diagonal — on every cell, not just bad
     ones, which is exactly why clean meshes regressed (8.2e-16 → 3.5e-12).
  2. **Composing the OPERATORS rather than the BLOCKS.** `gh_owner(C⁻¹(hg_owner(e_j)))` applies two
     scatters, so the second reads *neighbouring* cells' Hessians: the result is not a diagonal block
     at all but a block plus a ring of off-diagonal coupling. Extract the three blocks independently
     and contract them per cell.

  **The check that settles it is preconditioner-independent**: light one cell's degree of freedom,
  apply the outer operator, and read back that cell's own row — no neighbour can contribute, so it is
  the true `S` block whatever preconditioner is in force. Relative error of the extracted block
  against it, at a sliver: `A_gg` 1.06 → 8.23 as the cell flattens, local Schur a flat ~1e-3. Run that
  before trusting any future variant; the end-to-end error alone cannot distinguish a wrong block
  from a hard mesh.

  **⚠️⚠️ `CoupledBlockSweep` IS NOW THE DEFAULT SOLVE PATH (2026-08-23), and the flip cost three
  things that were not obvious from the measurement that justified it.** At the class count of 20 it
  matches the nested `20/10` pair's accuracy on every exactness fixture at **40 face passes against
  220** — 5.5x cheaper for the same answer (the count was 60 when this was written, at three passes
  per sweep; it was four then and is two now):

  The fixture table behind the flip is in `.claude/notes/schemes-log.md`, "Responsibility → `CoupledBlockSweep` as the default".

  - **⚠️ `narrow_gradient_sweeps` HAD TO LEARN ABOUT IT FIRST, and missing that would have failed
    SILENTLY.** It rewrote `SweptGradientSolve` nodes only, and the coupled sweep *replaces* both
    solvers rather than sitting beside them — so a defaulted scheme has no `SweptGradientSolve` left
    for it to find, and narrowing would have returned the tree unchanged while a caller believed the
    stencil was bounded. On `pitzDaily` that is the 1.75x probe-narrowing win, gone without a word.
    Exactly the shape already recorded for a Krylov outer solver, which narrowing genuinely cannot
    touch; this one it can, so it now does — carrying `relaxation` across, and pinned by a test that
    the narrowed copy differs in the count and in nothing else.
  - **⚠️ THE SILENT OVERRIDE, and four live call sites that proved why it mattered.** With the sweep
    on by default, any solver a caller passed never ran. That was not a hypothetical trap:
    `pitzdaily_gradient_ab/run_ab.py` (three sites, including the Betchen arm whose cost ratios this
    file quotes) and `uvreactor_openfoam/schur_block_diagnosis.py` (whose numbers were recorded the
    same day) would all have silently changed meaning. It was first patched with a guard that refused
    the contradictory pair; **the single-strategy field below removed the possibility instead**, which
    is the form to keep in mind — a guard is what you write when the type still lets the mistake be
    expressed.
  - **⚠️ `calibrated` WAS SIZING A PATH THE SCHEME WOULD NOT RUN.** It measured the nested pair and
    returned a scheme that would sweep instead — calibrated in name only. It now builds exactly one
    strategy, the one that will run.

  **✅ THE THREE SOLVE FIELDS ARE NOW ONE INJECTED STRATEGY (2026-08-23), which makes the conflict
  unrepresentable instead of guarded.** `HessianCorrectedGradient.hessian_solve: HessianSolve` →
  `CoupledBlockSweep` (default) / `NestedHessianSolve(solver, hessian_solver)` /
  `PackedSystemSolve(solver)`. Gone: `solver`, `hessian_solver`, `coupled_sweep`, `schur`, the guard
  that refused the contradictory pair, and the `_DEFAULT_*_SOLVER` constants the guard needed.

  **The defect it removes is not tidiness.** Held as separate fields, every path's settings were
  present at once and most were inert, so configuring the wrong one was ignored **in silence** — which
  is what happened the day the coupled sweep became the default and cost four call sites in this
  repository their meaning, two of them harnesses whose recorded numbers depend on those settings.
  A guard caught it afterwards; one field means it cannot be written down. `schur=False` folds in as
  a third strategy rather than a boolean crossed with a solver, which is what it always was.

  - **`_HessianSystems` gained `outer_preconditioner` and `dim`.** The coupled sweep needs the outer
    system's *preconditioner* and never applies its operator, whose construction takes an inner solve
    it would then discard — so reaching it through `outer` meant handing that call a strategy chosen
    only to be thrown away. Now `outer` is built *from* `outer_preconditioner`, so there is one place
    the block is formed.
  - **`narrow_gradient_sweeps` needed no new case for the nested path** — it recurses through
    `equinox.Module` fields, so the two swept solvers inside `NestedHessianSolve` are found the way
    they always were. Its explicit `CoupledBlockSweep` branch is still required, that being a count
    the recursion cannot see as a `SweptGradientSolve`. Both are pinned by one test.
  - **`calibrated` builds exactly one strategy**, the one that will run: `coupled=True` (default) the
    sweep, `coupled=False` the nested pair via `NestedHessianSolve.calibrated`, `schur=False` the
    packed system. There is no longer a second path on the returned scheme carrying counts nobody
    measured.
  - **Cost: 19 call sites**, mechanical, and each now states which path it means rather than relying
    on a default. Every measurement in this file taken on the nested path still describes it, because
    those sites now name it.

  **`CoupledBlockSweep` — sweep BOTH blocks instead of nesting a solve per apply (2026-08-22; the
  DEFAULT since 2026-08-23, see above).**
  The nested path re-converges the Hessian from zero once per outer sweep, throwing away what the
  previous outer sweep learned. Sweeping alternately keeps it:
  `h ← h + P_H⁻¹(A_Hg g − A_HH h)` then `g ← g + ω P_g⁻¹(b_g − A_gg g + A_gH h)`, Gauss–Seidel (the
  gradient update uses the Hessian just computed, which is what lets one Hessian sweep per gradient
  sweep converge at all). One sweep is **2** face-kernel passes against the nested `1 + inner` ≈ 11,
  each row evaluated once at the argument that yields its own defect rather than once per block and
  then subtracted — exact, because both rows are linear in `(g, H)` jointly.

  **Current status (profiled 2026-08-23, pitzDaily):** the sweeps are ~82 % of a reconstruction, so the sweep COUNT is the cost, not the work inside a sweep — calibrate it (`CoupledBlockSweep.calibrated`) rather than lowering the class default of 20; the sweep rate is a property of the boundary closure; XLA already shares the geometry-only prologue across fields; weigh any cache against its residency at reactor scale. Tables in `.claude/notes/schemes-log.md`, "Responsibility → where the time goes in a reconstruction".

  ⚠️ **A FACE-PASS COUNT IS A POOR PREDICTOR OF WALL CLOCK HERE, and this file quotes several.** The
  counts are exact and are fine for comparing arrangements that do genuinely different work (nested
  against coupled); they are misleading for judging a change that removes passes the compiler was
  already eliminating. Where a ratio in this file comes from counting passes, it says so.

  Calibrating to 1e-4 was confirmed on a full pitzDaily march (2.61x, bit-identical trajectory), and spending that lever promotes the prologue (the reason `bind` was built): same log section.

  **✅ `GradientScheme.bind(mesh, geometry)` IS BUILT (2026-08-23) — geometry-only reconstruction
  work hoisted out of the per-call path, bit-for-bit identical answers.** ⚠️ **`MultipleCorrectionGradient`
  takes an optional third argument, a `BoundaryLinearization`, and with it the binding is NO LONGER
  geometry-only.** Its corrections are probed through the operator the boundary conditions produce —
  see the ⚠️⚠️ note below for the probe rule, its two wrong predecessors and the measurements.
  **Both assemblers pass it**: `ResidualAssembler.build` evaluates it once, before any field exists, which is
  exact rather than approximate because both derivatives are bit-identical across states (pinned by
  `test_the_boundary_linearization_does_not_depend_on_the_state`). It is *not* independent of the
  **properties**: a convective condition blends `Gamma` into its face value, so a property that cannot be
  evaluated without a field makes `build` **raise** and name `boundary_linearization=` rather than linearize
  against the zero coefficient `boundary_values` falls back to — a silently wrong linearization is the defect
  this removes. `MomentumContinuity.build` binds once per solved field (see `.claude/rules/flow.md`). The base implementation is
  the **identity**, which is the honest answer for a single-pass sum or a swept solve whose
  preconditioner is a per-cell scalar; `HessianCorrectedGradient` overrides it to carry the outer
  preconditioner, threaded at the single place that is constructed (`_systems(..., prepared_outer=...)`)
  so all three `HessianSolve` strategies pick it up and none of them changed shape.

  **⚠️ THE ASSEMBLERS CALL IT, AND THAT PLACEMENT IS THE SAFETY PROPERTY, NOT A CONVENIENCE.**
  `ResidualAssembler.build` and `MomentumContinuity.build` bind the scheme they are handed, beside the
  face interpolation factors they already precompute. Binding at a *call site* instead would leave a
  bound scheme and a mesh as two separate things a caller could mispair — and a mispairing is silently
  wrong rather than slow (below). An assembler owns the geometry and the scheme together, so it cannot
  create that pairing. It also must **not** go in `__post_init__`: pytree unflattening runs at every
  jit boundary crossing, so the prologue would be rebuilt there rather than hoisted.
  ⚠️ **Both now bind against the CONDITIONS as well as the geometry, and the flow binds once per
  solved field (#467/#468/#469).** `ResidualAssembler.build` passes its own
  `boundary_gradient_weight`, evaluated at a zero state; `MomentumContinuity.build` carries
  `velocity_gradient_schemes` (one per component) and `pressure_gradient_scheme` because a patch
  treats the velocity and the pressure oppositely, so their weights are nonzero on disjoint patches.
  Both are two-phase builds — construct, then `dataclasses.replace` once the closures exist to read
  the weights from. `momentum.gradient_scheme` survives as the condition-free form for an initializer
  that re-binds it against a different equation's conditions; it is **not** what the flow residual
  applies. **`SSTTurbulence` carries the same shape since 2026-09-22** (`k_gradient_scheme` /
  `omega_gradient_scheme`): a wall prescribes `k` and lets `omega` extrapolate, so one binding cannot
  serve both, and its assembler is built inside each residual evaluation — which also means a scheme
  that cannot bind against a traced mesh only works there because the binding happens at build. See `.claude/rules/flow.md` for the measured numbers and the fixture trap.
  ⚠️⚠️ **THE PROBE GIVES EACH BASIS FIELD ITS OWN BOUNDARY DATA — AND TWO EARLIER RULES DID NOT.**
  `bind` takes a `BoundaryLinearization` (`value_weight` = ∂φ_f/∂φ_owner, `gradient_weight` =
  ∂φ_f/∂∇φ_owner, both read off the closures by `jax.jvp` at rest; `schemes/gradient.py`), and
  `_probe_corrections` drives a boundary face with `a(ψ_P + w·g₁) + (1−a)ψ_f + a(d·n)∂ₙψ(x_f)` —
  the condition's own construction fed **the probe's own value and normal derivative**. Every
  shipped condition builds its face value from the owner and a one-sided normal derivative (`a = 0`
  Dirichlet, `1` zero-gradient/Neumann, `1/(1+β)` Robin), so this is exact for them all, and with
  no linearization it reduces to the geometry-only build (`a = 0`, `w = 0`). The two rules it
  replaced, both wrong: *(i)* **geometry-only** — every face probed with the exact value — wrong for
  every derivative-type condition (2.0e-3 on the #467 fixture); *(ii)* **the condition's own data**
  (#468/#469: "the owner's value where the weight is nonzero") — which told every basis field its
  normal derivative on a zero-gradient face was **zero**, so the face contributed nothing to `M2`.
  Rule (ii) had two consequences, both measured. **Exactness needed fields satisfying their
  conditions IDENTICALLY**, not at the wall: 2.1 of the gradient on an 8×8 perturbed quad grid for
  a quadratic whose normal derivative vanishes on a zero-gradient wall but not one cell in, 7e-15
  under the current rule. **And a tetrahedron with two such faces was undetermined under every
  closure**, because `SkewCorrectedGradient` supplies the missing curvature from exactly that normal
  derivative: on `tetrahedral_grid_3d(3, perturb=0.25, seed=6)` with its whole boundary
  `ZeroGradient`, 18 two-face cells at `max|M2⁻¹|` ~8e15; on the 2462-cell tetrahedral duct the
  pressure binding reached **3.1e16 in 94 cells** and the laminar march from rest froze (`|R|`
  1.782e-4 on all 60 steps, every linear solve at the 120-cycle cap). Under the current rule every
  cell of both is determined (duct pressure peaks at 7.8e2) and the march moves again.
  ⚠️ **Rule (ii) also misclassified orthogonal faces**: it decided "follows the owner" from
  `gradient_weight != 0`, which is exactly zero on a face whose offset is purely normal, so a
  zero-gradient face on an orthogonal grid was probed as prescribed. `value_weight` does not have
  that blind spot, which is why it exists rather than being inferred from the gradient weight.
  **A geometry-only per-cell fallback was tried as a third repair tier (#475) and deleted**: under
  the current rule it was reached only with `OwnerGradient` and no fallback, where the geometry-only
  correction is just as singular, so it only ever added a warning falsely calling the cells well
  conditioned. The three repair warnings' shared once-per-process flag, which hid the duct's
  singular binding, is gone too: `_WARNED` is a set keyed per warning (`_warn_once`), pinned by
  `test_a_repair_report_does_not_silence_a_later_graver_one`.
  **Tests** (all mutation-verified): `test_every_condition_is_exact_for_a_quadratic_satisfying_it_at_the_wall`
  (Dirichlet, DirichletField, ZeroGradient, Neumann, Convective × both closures × skewed and
  orthogonal 8×8 quads, the condition on two adjacent walls so a corner cell carries two, plus the
  converse that the geometry-only binding is inexact for every non-prescribed type);
  `test_every_condition_is_exact_on_tetrahedra_with_two_boundary_faces`;
  `test_the_flow_is_exact_on_an_inlet_outlet_wall_duct`;
  `test_the_fallback_determines_the_corner_cells_under_their_own_conditions`. Breaking the probe
  five ways (the old rule, dropping the normal-derivative term, classifying by gradient weight,
  rounding the value weight to 0/1, dropping the prescribed term) and the flow's value weights two
  ways each turns at least one red; classifying by gradient weight is caught **only** by the
  orthogonal parametrization. ⚠️ The unit tier missed rule (ii) because every condition-aware
  fixture used a field satisfying its conditions identically and no cell with two gradient-type
  faces — test the at-the-wall case.
  **Still open:** the `SkewCorrectedGradient` march-loss table was measured with geometry-only
  binding and has not been repeated under this rule.
  ⚠️ Nothing about the numbers fails if the `bind` call is dropped from a factory — the residual just
  quietly goes back to rebuilding it per matvec — so it is pinned by
  `test_an_assembler_prepares_its_gradient_scheme_for_its_own_geometry`, and the identity default by
  `test_a_scheme_with_nothing_to_prepare_is_returned_unchanged` (asserting **identity**, since a
  scheme returning an equal copy would pass an equality check while defeating the point).

  The measured saving of `bind` (13 % of a six-field residual in 3D, 8 % in 2D) is in `.claude/notes/schemes-log.md`, "Responsibility → `GradientScheme.bind`".

  **⚠️ A STALE BINDING IS SILENTLY WRONG, NOT SLOW — this is why it is opt-in and not automatic.** The
  sweep runs a **fixed** count rather than to a tolerance, so the preconditioner determines the answer
  and not merely the rate. A cell-count mismatch raises; a *different* geometry at the same cell count
  cannot be detected. And binding outside a region differentiated with respect to **node positions**
  freezes the preconditioner into a constant, so the shape derivative comes back wrong rather than
  failing — differentiation with respect to the *field*, which is what a flow solve does, is
  unaffected and is pinned by a test comparing the bound and unbound gradients.

  **⚠️⚠️ THE CALIBRATOR MEASURED A PRECONDITIONER THE SCHEME DOES NOT RUN — fixed 2026-08-23, and it
  was worth 2.8x the sweep count on a skewed mesh.** `HessianCorrectedGradient.calibrated` took
  `local_schur_block`, forwarded it to `NestedHessianSolve.calibrated`, and **did not forward it to
  `CoupledBlockSweep.calibrated`** — which then measured the rate through `outer(...)`, whose
  `use_local_schur_block` defaults to `False`. So the count was derived from the cheaper
  preconditioner and spent under the shipped one. Measured, rate and the sweeps it implies at 1e-4:

  | mesh / closure | `local_schur_block=False` (what was measured) | `=True` (what runs) |
  |---|---|---|
  | pitzDaily 2D, `OwnerHessian` | 0.2263 → 7 | **0.1978 → 6** |
  | perturbed hex 3D, 35 % | 0.2570 → 7 | 0.2551 → 7 |
  | **perturbed tetrahedra, `AveragedNeighbourHessian`** | **0.8980 → 64** | **0.6620 → 23** |

  **It hides on well-shaped meshes and bites on exactly the ones this scheme exists for** — the two
  preconditioners are indistinguishable at 35 % hex perturbation and differ by 2.8x in cost on
  tetrahedra, which is why no fixture caught it. Pinned now by
  `test_the_coupled_sweep_calibrates_against_the_preconditioner_it_will_run_with`, on a tet mesh for
  that reason, asserting **both** directions so that forwarding a constant would not satisfy it.

  ⚠️ **`tools/sibling_builders.py` CANNOT SEE THIS PAIR, and that is its third blind spot.** The two
  factories are siblings by **contract** — both are `HessianSolve.calibrated` implementations reached
  from the same branch of one caller — but they construct *different* classes (`CoupledBlockSweep`
  against `SweptGradientSolve`), and the tool pairs builders by the class they construct. It reports
  `NestedHessianSolve.calibrated`'s `local_schur_block` as "only here" against an unrelated builder
  and never compares the pair that actually drifted. **A polymorphic factory family is invisible to
  it**, on top of the naming and `@classmethod` blind spots already recorded — so when the thing you
  are checking is one interface's factory implemented several ways, read the surfaces by hand.

  **Current status (UV reactor, 1.6M cells, measured 2026-08-23):** `local_schur_block=False` DIVERGES there (required, not an optimization); over-relaxation is refuted and the optimum is at or just below 1.0; `AveragedNeighbourHessian` is the faster closure there; sweep counts do not transfer between meshes — calibrate per mesh. Closed for the coupled sweep, do not re-propose: replacing it by a local least-squares fit (max error in skewed cells; least squares survives only as a preconditioner, unbuilt), sub-solving either block (the rate is inter-cell), polynomial acceleration / per-sweep relaxation schedules (bounded at ~1 sweep), coarse-grid / multigrid (the operator is second kind). The slow mode is extremely localized (~0.1 % of reactor cells). Literature notes (Betchen & Straatman, Nishikawa, Setzwein et al.) and all tables: `.claude/notes/schemes-log.md`, "Responsibility → the UV reactor, and directions closed for the coupled sweep".

  **✅ `MultipleCorrectionGradient` IS BUILT (2026-08-24) — the same exactness contract with NO
  system to solve.**
  ⚠️ **RECOMMENDED FOR HEXAHEDRAL AND POLYHEDRAL MESHES, NOT FOR TETRAHEDRA UNDER A PRESSURE COUPLING
  (decided 2026-09-23, closing the investigation).** On the tetrahedral duct its weights run ~2900x a
  damping reference and flip 24 Rhie--Chow damping diagonals under the pressure's own conditions
  (eigenvalue `+2.17`), and the coupled march then finds no admissible step from any seed tried. What
  was tried and did **not** fix it, so nobody re-tries them: both corner-cell boundary closures
  (`OwnerGradient`, the `SkewCorrectedGradient` repair — 15 orders of magnitude apart in `M2`
  conditioning, near-identical march failure), the first pass alone inside Rhie--Chow, ring and
  threshold withholding of the second pass, mixed per-cell schemes, and Betchen's scheme as a
  calibration source. The defect is the second pass's *weights*, which exactness does not pin down;
  the answer is a scheme that chooses them (`ProjectedStencilGradient`), not a repair of this one.
  This is recorded in the class docstring and on the docs page as a recommendation; **no default
  moved** — `DEFAULT_GRADIENT_SCHEME` is still this scheme, and the two shipped hexahedral cases are
  unaffected. #435 carries the history.
  `aquaflux/schemes/multiple_correction.py`, following Pont, Brenner, Cinnella,
  Maugars & Robinet (JCP 350, 2017); Setzwein, Ess & Gerlinger (JCP 446, 2021) is the vertex-centred
  adaptation and points back to Pont for the cell-centred correction matrices. Both papers are in
  the reference folder.

  **The construction.** The raw Green--Gauss sum handed a linear field of gradient `a` returns
  `M1 a`, not `a`. Recover `M1` by applying the sum to the coordinate fields; `D1 = M1⁻¹R` is then
  linear-exact. Apply `D1` twice for an inconsistent Hessian, recover `M2` as what that returns for
  each quadratic basis field, and `M2⁻¹` repairs it. `D1`'s own first-order error on a quadratic is
  a fixed linear function of the Hessian, so subtracting it lifts the gradient to second order.
  **Two face passes and three per-cell matrix products** — a fourth, and a per-cell inverse with it,
  whenever a caller supplies `boundary_gradient_weight`, which every residual assembler does — against
  the coupled sweep's 12--15 sweeps of two passes on the reactor.

  **Every correction matrix is PROBED, not derived** — the operators run on coordinate monomials,
  the same device `cell_diagonal_block` uses. So there are no geometric formulas to port, no volume
  moments, and the corrections cannot drift from the operators they correct.

  Measured (`validation/multiple_correction/exactness_probe.py`, and the JAX path reproduces it):

  | | quadrilateral | hexahedral | tetrahedral |
  |---|---|---|---|
  | gradient error on a quadratic | 1.9e-15 | 1.3e-15 | 2.2e-14 |
  | Hessian error | 3.8e-14 | 1.2e-14 | 3.0e-13 |
  | `cond(M1)` / `cond(M2)` | 2.2 / 2.9 | 2.0 / 2.9 | 4.8 / 6.9e2 |

  Compare `cond(A_HH)` = **1.01e18** for the coupled scheme on tetrahedra under `OwnerHessian`. The
  matrices this scheme inverts are small and local, which is *why* it needs no iteration — that part
  is structural and holds everywhere. ⚠️ **Their conditioning is NOT uniform, and the row above is its
  own fixture's.** Re-measured 2026-09-20 on `tetrahedral_grid_3d(3, perturb=0.25, seed=6)`:
  `cond(M1)` max 4.84 but `cond(M2)` **median 11**, max 3.6e17 at the corner cells an owner closure
  cannot determine, 118 once the fallback repairs them. On the real 2462-cell tetrahedral duct
  (`validation/tetrahedral_gradient_ab`): `cond(M1)` max **15**, `cond(M2)` median 7.9, p99 586 after
  repair. So quote conditioning per mesh; do not carry "well conditioned" as a property of the method. Order of accuracy on a smooth non-polynomial field reproduces Pont's Figs. 10 and 11 —
  gradient → 2, Hessian → 1 — and an uncorrected control fails by 12--97 % and 1.4--8.5x, which is
  what makes the exactness result mean something.

  **Exactly linear in the field, verified rather than assumed**: `jvp(t)` equals the map applied to
  `t` **bit for bit** (`0.0` difference), `f(0) = 0`, reverse mode matches a central difference to
  1e-8. So the tangent is an unrolled apply, not an implicit-function solve.

  **⚠️⚠️ THE BOUNDARY CLOSURE MUST REPRODUCE LINEAR FIELDS EXACTLY, and this is the non-obvious
  rule the whole scheme turns on.** The corrections are probed on the *quadratic* basis, so they
  absorb whatever a closure does to a quadratic — but an error made at *linear* order lands in a
  term those probes never see and nothing removes it. Measured, on tetrahedra:

  | closure | linear-exact | quadratic exactness | `cond(M2)` |
  |---|---|---|---|
  | `OwnerGradient` | yes | **fails** (7e-2) | 9.1e17 |
  | neighbour-averaged | yes | fails (8.7e-2) | 2.4e17 |
  | raw one-sided difference | **no** | fails (1.1e0, worse than owner) | — |
  | **`SkewCorrectedGradient`** | yes | **5.2e-14** | 6.9e2 |

  Two separate lessons there. `OwnerGradient` is linear-exact and still fails on tetrahedra, for a
  *dimensional* reason — four faces, one or two carrying no direction the owner's gradient has not
  already supplied, against six Hessian components. And the raw one-sided difference fails for the
  *other* reason, being linear-inexact on a skewed mesh; it uses more information than the owner
  closure and does worse. ⚠️ Note also that neighbour-averaging, the fix that rescued this scheme's
  own `A_HH`, makes things *worse* here — do not reason by analogy between the two schemes.

  **`non_orthogonal_correction` (`schemes/interpolation.py`) is the shared term that makes the
  default closure legal**, and `discretization/diffusion.py` now calls it too — it had the same
  formula inline, having always needed it to extrapolate the same derivative to the same place.
  ⚠️ That is the one home; a second copy in either caller is the defect.

  **Not yet done**: no A/B against the coupled scheme on a march, nothing measured at 1.6M cells or
  through snappyHexMesh's hanging-node interfaces, and the distributed path raises (it needs the
  reconstructed gradient halo-exchanged once between the two passes — a real gap, but not the
  structural bar the coupled scheme has).

  **✅ THE FIRST PASS'S `(M1 − B)⁻¹` IS THE ADJUGATE, NOT A BATCHED LAPACK LU — ~5× OFF THE pitzDaily
  RESIDUAL, ~2.5× OFF ITS JVP (2026-10-16, branch `claude/multicorr-adjugate-inverse`).** Assemblers
  always pass `boundary_gradient_weight`, so `_boundary_condition_first_pass` ran inside every residual
  and every `jvp`, and its `jnp.linalg.inv` compiled to one `lapack_dgetrf_ffi` + two `lapack_dtrsm_ffi`
  custom calls per reconstruction that XLA cannot fuse (counted in the compiled pitzDaily coupled
  residual at `main` `44ba609f`: **6 `getrf` + 12 `trsm`**, 0 after; an earlier count on 2026-10-16 read
  5 + 10 and was not reconciled) — the
  #648 problem, fixed then for compact/corrected only. It now calls `small_inverse` (see the #648
  entry). **`m1_inverse` at bind uses it too, deliberately**: with a zero weight the per-call path then
  inverts the identical matrix by the identical arithmetic, which is what keeps
  `test_an_orthogonal_mesh_is_bit_identical_to_ignoring_the_weight[multiple-correction]` and
  `test_a_prescribed_patch_is_left_alone_by_the_boundary_condition_weight` exact — with LAPACK at bind
  and the adjugate per call both went red by roundoff. `m2_inverse` (`n_sym × n_sym`, 6×6 in 3D, bind
  only, `cond(M2)` up to 6.9e2 on tetrahedra) stays `jnp.linalg.inv`. Pinned by
  `test_the_boundary_condition_first_pass_calls_no_library_factorization` (no `getrf` in the compiled
  reconstruction; red with the old call restored). Measured pitzDaily `case.yaml`, hybrid initial
  state, `eqx.filter_jit` of `coupled.residual` and of its `jax.jvp`, min of 9 warm calls, 4-core Linux
  container, jax 0.11.2, x64, CPU:

  | arm | residual | jvp | `getrf` in residual HLO | ‖R‖ |
  |---|---|---|---|---|
  | `jnp.linalg.inv` (before) | 29.8 ms | 45.0 ms | 6 | 2.884371116264e+02 |
  | `small_inverse` (after, two runs) | 5.3 / 6.4 ms | 17.2 / 19.3 ms | 0 | 2.884371116264e+02 |

  ‖R‖ and ‖J t‖ agree to all 13 printed digits. ⚠️ The `jvp` column is a standalone call, and on
  glibc each one pays ~11 000 page faults (its 42.8 MiB temporary buffer is above the 32 MiB adaptive
  mmap cap). The after arm is 9.8–11.1 ms with them removed, and the march never pays them
  (`discretization.md`, "Single-threaded kernels", 2026-10-17). Standalone on the same shapes (previous measurement,
  same configuration): `inv` 2.77 ms vs adjugate 0.08 ms at `(12225, 2, 2)`, 8.81 vs 0.94 ms at
  `(23040, 3, 3)`. bfs3d (3×3 blocks, six reconstructed fields) is unmeasured — its mesh is not in the
  repository. ⚠️ Every pitzDaily/bfs3d residual, JVP, probe or march timing in these files taken under
  `MultipleCorrectionGradient` before this predates it and was not re-run.

  On the shipped pitzDaily march the adjugate gain is ~2–5 %, not the per-call projection; and on pitzDaily the scheme marched faster than `CorrectedGreenGauss` at equal probe reach (2026-08-24): `.claude/notes/schemes-log.md`, "`MultipleCorrectionGradient` — measurements".

  **✅✅✅ AND ITS REAL PRIZE IS THE STENCIL REACH, WHICH THAT COMPARISON HELD FIXED — 32 % OF pitzDaily
  WALL CLOCK (measured 2026-09-08).** The table above ran both arms at probe reach 5, so it priced the
  reconstruction and not what the reconstruction *permits*. Each Richardson sweep couples a cell one ring
  further, so the swept residual reaches `sweeps + 1`; two face passes have no sweep count to push
  anything outward. Measured on the scalar-Laplace probe (`validation/gradient_stencil_reach.py`,
  extended with this arm), at every skewness tested:

  | scheme | reach | Jacobian mass beyond distance 3 | error vs the ANALYTIC gradient (skew 0.05 / 0.25 / 0.40) |
  |---|---|---|---|
  | swept, 4 (the shipped count) | **5** | 1.5e-7 … 1.5e-4 | 1.07e-2 / 1.91e-2 / 2.96e-2 |
  | exact Krylov solve of the same system | 7–11 | same | 1.07e-2 / 1.91e-2 / 3.02e-2 |
  | **`MultipleCorrectionGradient`** | **3** | **exactly 0.000e+00** | **7.11e-3 / 7.39e-3 / 7.77e-3** |

  Two findings, and the second was not expected. A probe at reach 3 recovers the two-pass **scalar**
  Jacobian **exactly** — nothing to fold, so the "vary reach and fill together or not at all" pairing rule
  has nothing to bite on. ⚠️ **Not the coupled RANS Jacobian** — see the reach-4 entry below the march
  table. And the corrected-Green–Gauss system's *own exact solution* is ~4x less accurate
  than the two-pass answer at high skew: **the sweeps converge faithfully to a worse answer**, and every
  sweep past ~2 buys no accuracy while pushing the stencil one more ring out. The two-pass error is also
  near skew-independent (7.11e-3 → 7.77e-3 over an 8x skew range) where corrected Green–Gauss degrades 3x.
  ⚠️ **On two-dimensional quadrilaterals.** On tetrahedra the second pass is not a small correction at
  all: measured 2026-09-20 on the 2462-cell duct, it is ~half the size of the gradient it corrects at the
  median *interior* cell, against machine-zero on an orthogonal hexahedral mesh of the same duct.

  The pitzDaily march at reach 3 against `CorrectedGreenGauss` at reach 5 (same root, 32 % less wall clock) is in the same log section.

  ⚠️ **THIS IS A SKEWED-MESH RESULT AND DOES NOT TRANSFER TO `bfs3d`.** That case is a rectilinear
  blockMesh, skew-free to round-off, so its sweeps are inert and it already floors at reach 3 — it sets
  no `stencil_reach` at all. What this changes is the standing warning that "`stencil_reach = 3` IS A
  PROPERTY OF SKEW-FREE MESHES, NOT OF THE DISCRETIZATION … the sibling gets 3 for free and that is luck,
  not physics": with a two-pass reconstruction, the **gradient's** reach becomes a property of the scheme
  on a genuinely skewed mesh too. pitzDaily has since adopted it (its `case.yaml` states
  `MultipleCorrectionGradient`, which `PITZ_GRADIENT` overrides only when set; `STENCIL_REACH` defaults to 3). So does `bfs3d` since 2026-10-13 (its hand-built `build_case` and its
  `case.yaml` both), and there the choice is NOT inert for the probe: under it the p, k and omega columns
  carry 1.1e-3 / 6.7e-5 / 3.9e-5 of their norm beyond reach 2 (against ~1e-16 under `CorrectedGreenGauss`),
  so its probe went from per-column `(3,3,3,3,2,2)` to a uniform reach of 3
  (`.claude/notes/solve-globalization-log.md`, "bfs3d takes pitzDaily's march settings").

  **⚠️ THE COUPLED pitzDaily RESIDUAL REACHES 4, NOT 3, UNDER THIS SCHEME — so the shipped probe at reach 3
  folds (measured 2026-10-07, `validation/pitzdaily_openfoam/probe_reach_check.py`).** The scalar figure
  above is the gradient's own; the coupled residual feeds the reconstructed **velocity** gradient into the
  eddy viscosity and the face viscosity, which spend one more ring. Per (row, column) block, relative error
  of the materialized matrix against the exact jvp on a random vector supported on that column, at the
  time-accurate OpenFOAM field mapped onto the mesh (jax 0.11.2, CPU):
  at **reach 3 (165 probes)** the `u` and `v` columns are inexact in every row (2.2e-08 to **5.5e-06**),
  `k` → `omega` is 5.9e-09, and the `p`, `k`-row and `omega` columns are at round-off; at **reach 4
  (265 probes)** every block is at round-off (worst 3.7e-16). An independent reach ladder on the same
  state agrees (165 probes 1.5e-7 global, 265 probes 3.9e-16). So "reach 3 becomes a property of the
  scheme" holds for a scalar and **not** for coupled RANS on a skewed mesh. Orthogonal `bfs3d` is
  unaffected (its second gradient pass vanishes). The case still ships reach 3 — whether the ~1e-6 fold
  costs Krylov cycles has not been marched; a per-column plan `(4,4,3,4,3)` would be exact at ~225 probes
  (`k` → `omega` makes `k` need 4; not measured as a plan). The 506 s / 417-cycle row above ran on the
  folded matrix.

  **✅✅ AND IT IS NOW THE LIBRARY DEFAULT — `schemes.DEFAULT_GRADIENT_SCHEME`, 2026-09-16 (#361).**
  `MultipleCorrectionGradient()` with its own defaults (`OwnerGradient`, no fallback), as a
  module-level singleton in `multiple_correction.py`. `MomentumContinuity.build` and
  `SSTTurbulence.build` take it when told nothing; both moved `gradient_scheme` from a **positional**
  parameter to a keyword-only one to have somewhere to put a default, which broke all 77 call sites
  at once and was mechanically converted. The evidence is the two tables above and nothing new was
  measured to justify the flip.

  **What the same change fixed is worth more than the flip: ONE RUN CAN NO LONGER USE TWO
  RECONSTRUCTIONS.** `hybrid_initialize` and `potential_flow` each substituted `CompactGreenGauss()`
  when handed nothing, and **every caller in the repository handed them nothing** — so every case
  built its initial condition on one discretization and solved on another, an omission behaving as a
  physics choice. They now read the scheme off the assembler they are initializing (`potential_flow`
  from `momentum.gradient_scheme`; `hybrid_initialize` from `turbulence.gradient_scheme` for k/omega
  and through `potential_flow` for the flow) and **take no `gradient_scheme` argument at all**, so
  there is nothing left to disagree with.

  Measured on pitzDaily (2026-09-16), reading the scheme off the assembler moves only the start, not the root: same log section.

  Pinned by four tests in `tests/unit/test_initialization.py`,
  each mutation-checked, and by a signature census
  (`test_no_initializer_names_a_gradient_scheme_of_its_own`) — which is the one that matches the
  defect's shape, since the drift was across call sites that each looked reasonable alone.

  ⚠️ **One sentinel, one meaning: `gradient_scheme=None` means NO reconstruction, everywhere.** It
  stays that on `ResidualAssembler.build` and `ScalarTransport.build`, where it is a checked state
  (refused the moment a flux operator reports `uses_gradient()`) and genuinely right for pure
  diffusion on an orthogonal mesh. Defaulting *those* to the multiple-correction scheme would build
  correction matrices — `(n, n_sym, n_sym)`, 440 MB at 1.6M cells — for a reconstruction nothing
  reads. The default therefore lives only on the two builders where a gradient is mandatory.

  ⚠️ **The default cannot run domain-decomposed, and that is now a property of the DEFAULT rather
  than of an opt-in** — it raises (see #134, which carries the halo-exchange gap). A partitioned run
  has to name `CorrectedGreenGauss` explicitly.

  ⚠️ **Nor is it right on a mesh with CORNER TETRAHEDRA, and neither is its documented escape
  hatch — #432 carries that.** `OwnerGradient` leaves such a cell's Hessian underdetermined (`bind`
  warns, naming the cells) and `SkewCorrectedGradient`, which supplies the missing direction, stalls
  pitzDaily dead — the whole chain is measured below. Every measurement here is on quadrilateral or
  hexahedral meshes, so the regime the default is *least* defensible in is the one with no case.

  **Current status of the `SkewCorrectedGradient` stall on pitzDaily (resolved 2026-08-25):** the arms part at the second inner Newton iteration of the target rung; the chain is reconstruction → velocity gradient at boundary-owning cells → near-wall `k` production, whose own `k`-derivative makes `J_kk` negative so the shift only just outweighs it → an unbounded upward `dk` → the `omega` fixation row slaves `omega` to `k` → log transport exponentiates it. The step length was set by the global k-positivity cap, and a per-entry projection made the skew closure march (that cap and its projection switch were both deleted 2026-10-14 for a per-cell `k` clip with an `omega` bound). Along the way `boundary_values_at` fixed a genuine double correction that was not the cause. Refuted, do not re-propose: the wall `omega` gradient, the correction matrices' conditioning, the double correction, a cap on the log-`omega` increment. The initializer re-bind costs ~0.014 s after the first. Full trail: `.claude/notes/schemes-log.md`, "`MultipleCorrectionGradient` — the initializer re-bind, and the `SkewCorrectedGradient` stall on pitzDaily".

  **✅✅ BUILT (2026-08-25, weight corrected since): `boundary_gradient_weight` — THE FIRST PASS AND THE
  CLOSURE READ THE FACE VALUE THE BOUNDARY CONDITION DEFINES ON A PATCH WHOSE VALUE FOLLOWS THE OWNER. Both
  closures become LINEAR-EXACT there.**

  The defect it repairs is upstream of every closure and was missed by every probe in this entry. A
  reconstruction is handed the field's boundary values, and on a gradient-type patch that value
  is the value evaluated at ZERO gradient, which on a skewed boundary cell is not the value the
  boundary condition defines (a zero-gradient face carries `phi_P + grad phi_P . (d - (d.n) n)`; the
  condition holds at every iterate, and that value depends on the gradient being reconstructed). The
  Green--Gauss sum then averages the interior field against a boundary that contradicts it, and the
  result is wrong at **linear** order, which is exactly the error the correction matrices (calibrated
  on quadratics) cannot remove.

  Measured on a linear field over an all-zero-gradient boundary, true Hessian identically zero:

  | rule | N | owner `\|H\|` | owner err | skew `\|H\|` | skew err |
  |---|---|---|---|---|---|
  | **Dirichlet control** | 4→32 | ~1e-15…1e-13 | **0.00 %** | ~1e-15…1e-13 | **0.00 %** |
  | boundary-condition values | 8 | 7.771e+00 | 57.14 % | 1.255e+01 | 61.54 % |
  | boundary-condition values | 16 | 1.554e+01 | 57.14 % | 2.511e+01 | 61.54 % |
  | boundary-condition values | 32 | 3.109e+01 | 57.14 % | 5.022e+01 | 61.54 % |

  **The Hessian DOUBLES with every refinement and the gradient error does not move at all.** The
  Dirichlet row is the control: with a prescribed value the reconstruction is exact, so this is the
  boundary *rule*, not the scheme. And note it is **not closure-specific** — the owner closure fails
  too, because the *first pass* reads boundary values before any closure is consulted.

  **The repair** (proposed by the project owner): feed the reconstruction the value the boundary
  condition defines, as a function of the owner gradient, and leave a prescribed value alone. Through
  the real assembler, same field and meshes (measured with the first form of the weight, the owner's
  full displacement `d`; the tangential weight below gives the same linear exactness, pinned by
  `test_a_field_satisfying_its_boundary_conditions_reconstructs_exactly`):

  | N | owner | skew |
  |---|---|---|
  | 8 | **1.864e-15** | **1.716e-15** |
  | 16 | **3.931e-15** | **3.483e-15** |
  | 32 | **7.617e-15** | **7.019e-15** |

  **It costs nothing.** Read literally the value is a fixed point (it needs the gradient it is
  reconstructing), and iterating it converges at a **mesh-independent 0.571 per pass** — eight or ten
  passes. But it is *affine* in the gradient: each such face contributes `(w . grad phi_P) A_f` to the
  sum, i.e. `B_P grad phi_P` for a per-cell matrix, so it moves to the other side and the pass is
  `(M1 - B) grad phi = raw` — one per-cell inverse, which is what `M1` already was. Verified against
  the 40-pass iteration: `1.6e-15` direct against `9.1e-13` iterated.

  **The seam is `boundary_gradient_weight`** = `d(boundary value)/d(grad phi_owner)` per face, shape
  `(n_faces, dim)`, **differentiated from the closures rather than declared** (one JVP per gradient
  component), so it cannot disagree with them and needs nothing new from a boundary condition: zero on
  a prescribed patch, the tangential offset on a zero-gradient or Neumann one, and what a Robin
  condition actually carries. Supplied by `ResidualAssembler._gradient` and by `MomentumContinuity`
  (per velocity **component** — a patch may treat them differently).

  ⚠️ **THE FIRST FORM OF THIS WEIGHT WAS WRONG ON TETRAHEDRA, and the name it had (`boundary_chain`,
  `d(value)/d(phi_owner)`, equal to the owner's full extrapolation `w = d`) is deleted.** It reads no
  information from the boundary condition: the face value simply follows whatever gradient the cell has,
  normal component included. On a cell whose interior faces do not span every direction -- a tetrahedron
  owning two such boundary faces has two interior faces for three gradient components -- `M1 - B` is then
  singular. Measured on the 2462-cell tetrahedral duct of `validation/tetrahedral_gradient_ab`:
  `cond(M1 - B)` up to `4e18` on every such cell and `1e16` on some owning one, against at most `2.5` and
  `8.3` with the condition's own weight. `inv` of a singular 3x3 returns rounding-dependent values, so
  the compiled and the eager residual disagreed (a `3e-1` gap in the `jvp` on the full scheme). After the
  fix compiled and eager agree to about `1e-16`. Hexahedral meshes never showed it: their boundary
  cells' interior faces span all three directions. (Until #313 this was the **only** boundary-consistency
  repair the flow block could take, because its closures accepted no gradient; they take one now, so
  it passes `boundary_values_at` as well.)

  Marched end to end on pitzDaily with the first-form weight, the answer did not move: `.claude/notes/schemes-log.md`, "`boundary_gradient_weight` — measurements".

  **✅ EXTENDED TO `CompactGreenGauss` AND `CorrectedGreenGauss` (2026-10-09, #648); STILL NOT
  `HessianCorrectedGradient`** (it ignores the weight; see the last bullet). `B_P` has one home, `schemes.boundary_gradient_block(weight,
  face_cells, geometry)` (`(n_cells, dim, dim)`, per volume, its docstring carrying the affine
  contract); `_boundary_condition_first_pass` calls it, **bit-identical** to before on quad, hex and
  tet fixtures under both closures. Compact solves `(I − B_P) g = raw/V` per cell. Corrected adds
  `− V B` to `A_g` (`_CorrectedTerms.boundary_block`, set by `terms(mesh, geometry, weight)`; `None`
  leaves the terms geometry-only, which is what `calibrated` and the contraction-rate tools measure),
  and **both `CellPreconditioner`s include it**: `InverseCellVolume` builds `CellBlockJacobi((I −
  B)⁻¹ / V)` when a block is present and plain `1/V` otherwise; `ExactCellBlock` subtracts it from the
  probed block. Measured (`tests/unit/test_gradient_boundary_block.py`, x64 CPU): a linear field
  satisfying zero-gradient / Neumann / Robin data on two walls of an 8×8 grid perturbed 0.3 is off by
  **0.15–0.17** of its gradient before and roundoff after (corrected, GMRES); same on columnwise 3D
  hexahedra, on tetrahedra with two flux faces, and for compact on a sheared parallelogram grid (0.30–
  0.34 before). Orthogonal mesh: **bit-identical** for every scheme (`w` is exactly zero there).
  - **The asymptotic sweep rate does not move, and that is structural**: for a cell owning one such
    face `B_P = s wᵀ/V` with `w ⟂ s`, so `B_P² = 0` — no eigenvalue. Gelfand rate, `InverseCellVolume`,
    before → after: quad 0.3 16² 0.1817 → 0.1816, quad 0.4 0.3699 → 0.3699, hex 0.3 6³ 0.1894 → 0.1894,
    tet n=3 0.6773 → 0.6804; `ExactCellBlock` likewise. **So a count calibrated on the geometry alone
    stays valid** and `calibrated` was left geometry-only.
  - **But a FIXED sweep count sees the transient, and corner cells are not nilpotent** — which is why
    the default preconditioner carries `B`. Sheared parallelograms (interior skew exactly zero, so
    `A_g = V(I − B)`), 16², all four walls zero-gradient, error vs the exact solve after 2/4/8 sweeps:
    aspect 50 (`max|B_P|` 10): plain `1/V` **3.9e-2 / 9.7e-3 / 6.0e-4**, `V(I − B)` roundoff from one
    sweep. With only the two horizontal walls zero-gradient (single-face cells, nilpotent `B`) plain
    `1/V` is exact in two sweeps. On randomly perturbed grids the interior skewness dominates and the two
    preconditioners agree to two figures.
  - **⚠️ THE FIRST VERSION COST 8–9× PER COMPACT RECONSTRUCTION ON EVERY MESH, ORTHOGONAL INCLUDED —
    and timed out slow shard 1 (PR #652).** Assemblers always pass the weight, so the block is built and
    inverted on every residual evaluation and every `jvp`, even where it is exactly zero. A batched
    `jnp.linalg.solve`/`inv` of the `(n, dim, dim)` blocks is a LAPACK call XLA cannot fuse: 5.8 ms
    against 0.43 ms for the adjugate at 13824 3×3 blocks (jax 0.10.2, x64, CPU, one core). Now
    `small_inverse` (closed-form 1/2/3-dim adjugate in `gradient.py`, imported by
    `multiple_correction.py` beside `boundary_gradient_block`, not in `schemes.__all__`) and an
    owner-only scatter (`FaceCellConnectivity.scatter_to_owner`) build it. ⚠️ **The fix stopped at these
    two schemes for a week: `MultipleCorrectionGradient`'s first pass kept a per-call
    `jnp.linalg.inv(M1 − B)` until 2026-10-16** — see the multiple-correction section's adjugate entry. Measured with/without the weight on orthogonal
    meshes, min of 15, warm: compact 2D 80×40 0.158/0.141 ms, 3D 24³ ~1.1–1.6×; corrected 2D ~1.2–1.3×,
    3D ~1.4× — noisy single runs in a shared container. **`InverseCellVolume` inverts `I − B` and then
    divides by `V`, in that order**: inverting `V(I − B)` directly broke bit-identity on orthogonal meshes
    (`V/(V·V)` is not `1/V` to the last bit, 1.8e-15). `_CorrectedTerms.boundary_block` therefore holds
    `B` per volume, and the operator and `ExactCellBlock` scale it by `V`.
  - **Gate C now holds with flux walls**: one Newton step reproduces a linear field to 1e-9 on a 25 %
    skewed grid with Neumann top/bottom (`test_gate_c_holds_with_flux_walls_on_a_skewed_mesh`); the
    control without the block misses by 1.1e-3.
  - ⚠️ **Every `CorrectedGreenGauss`/`CompactGreenGauss` measurement in these files that ran under
    derivative-type conditions on a skewed mesh predates this and was not re-run** — the tetrahedral
    duct's corrected **march** (laminar 13 steps / 5.4e-9), pitzDaily's corrected arms and its gradient
    A/B tables (near-orthogonal walls, so `w` is small there), bfs3d (skew-free to `1.9e-12`, so
    effectively unchanged). Measurements taken under Dirichlet values only are **unaffected** (`w = 0`):
    the duct's "misses a quadratic by 7x" (exact Dirichlet values) and its damping-operator rows
    (Dirichlet-zero values, eigenvalue `9.0e-4`) among them.
  - **`HessianCorrectedGradient` still ignores the weight** — `B` would enter the gradient equation's
    own block and must reach the bound outer preconditioner (`local_schur_block`, `bind`'s
    `prepared_outer`). Left open on #648 as its separable follow-up.

  ⚠️ **It reaches the first pass and the closure, not `M2`.** The correction matrices are still probed
  against exact face values, so a quadratic keeps a second-order inconsistency — measured at 1.6 % of
  the boundary-cell gradient at 8 cells across and falling as `h` (0.76 %, 0.37 % at 16 and 32),
  against the 58 % that does *not* fall without this.

  The flow-block `boundary_values_at` census (the defect #313 fixed) and the reactor measurement that both closures reconstruct a quadratic exactly there (`OwnerGradient` is right for the reactor) are in the same log section.

  **⚠️⚠️ A SINGULAR CORRECTION IS PLATFORM-DEPENDENT IN MAGNITUDE — large and finite on one machine,
  NON-FINITE on another (found by CI, 2026-08-24).** The same tetrahedral mesh gives `max |M2⁻¹|` of
  **1.61e16** under macOS Accelerate and **NaN** under the BLAS on the GitHub runners, and the
  reconstruction there is a 173 % error locally against NaN on CI. Neither is wrong — inverting a
  singular matrix has no defined answer — but it makes every `x > threshold` test of a *failure*
  wrong, because **every comparison against NaN is False**. Three tests here asserted the failure
  that way and passed locally while failing on CI; they now negate the success condition instead.

  **Two live comparisons had the same defect and are fixed:**
  - `_undetermined_cells` selected with `worst > limit`, so a NaN correction would **not** be
    detected and the repair would silently decline to fire on exactly the cells that need it.
    ⚠️ **The first fix, `~(worst <= limit)`, was STILL blind to it (found 2026-09-21), because
    `worst = max|M2⁻¹|` had already lost the NaN.** XLA's CPU max-reduction does not reliably propagate
    one: jax 0.10.2, macOS arm64, an all-NaN row reduces to NaN in a 4-cell array and to **-inf** in a
    1000-cell one, and to -inf on the 162-cell `tetrahedral_grid_3d(3, perturb=0.25, seed=6)` fixture —
    which then passes `worst <= limit` as determined. Now each entry is compared and the booleans
    reduced: `~jnp.all(|M2⁻¹| <= limit, axis=(1, 2))`. **A reduction that could see a NaN must never
    come before the comparison that is meant to catch it.** Pinned by
    `test_a_non_finite_correction_counts_as_undetermined_at_any_mesh_size`, whose 4- and 1000-cell arms
    fail the plain `>` form and the reduce-then-negate form respectively (mutation-checked).
    **It surfaced as a CI flake** (`assert 16 == 18` in
    `test_cells_a_condition_leaves_undetermined_keep_their_geometry_only_correction`, some runners and not
    others): that test thresholded `np.max(...) > limit` itself, so two NaN cells dropped out of its
    count. That test went with the third tier it pinned (#483); its successor,
    `test_the_fallback_determines_the_corner_cells_under_their_own_conditions`, takes the stuck set
    structurally (cells owning ≥2 boundary faces) and asserts `_undetermined_cells` flags exactly that
    set under the owner closure alone. Injecting NaN into two of the 18 inverses
    reproduces the CI count of 16 under the old test and fails the old production guard under the new one. ⚠️ Why only some
    runners produce a NaN was **not established** — a NaN from an exactly-zero pivot on some BLAS
    kernel paths is the consistent reading; the CI log records no CPU model.
  - `fastest_boundary_closure` compared `rate < best_rate`, so a closure whose rate came back
    non-finite lost *by accident* rather than on its merits. Non-finite now maps to infinity, making
    the ordering total.
  - ⚠️⚠️ **And the worse half, which the first CI fix did not catch: the singular closure's rate came
    back as `2.2e-308`, not as NaN.** `contraction_rate` deliberately maps an annihilated probe to
    `finfo.tiny` (a correction identically zero on the mesh genuinely has rate zero), and on a
    **singular** system the probe is annihilated by arithmetic instead — so the broken closure was
    reported as a **perfect** contraction and won. Same mesh, same closure: **8.64** under macOS
    Accelerate, **2.2e-308** under CI's BLAS. `ContractionRate` now reports `annihilated`, and the
    selector treats an annihilated candidate as unusable; neither shipped candidate can annihilate
    the probe legitimately, and the one case where that would be genuine makes them equivalent
    anyway. Pinned by simulating the floor rather than waiting for the platform that produces it —
    which is why it went unnoticed in the first place.

  **Consequence for tests on degenerate meshes (binding):** assert the property that survives the
  platform — "does not contract", "is not exact" — never the magnitude of a singular result. The
  numbers quoted in this file for singular cases are macOS numbers and are illustrative of *scale*
  only.

  **⚠️ THE FAILING SHAPE IS A TETRAHEDRON WITH *TWO OR MORE* BOUNDARY FACES — a corner or edge tet,
  not a tet at a boundary.** Resolved by boundary-face count on the synthetic tetrahedral fixture
  under `OwnerGradient`, which is what makes the reactor result explicable rather than surprising:

  | boundary faces | cells | `max \|M2⁻¹\|` | max relative error |
  |---|---|---|---|
  | 0 | 72 | 1.97e+00 | 4.46e-15 |
  | 1 | 72 | 4.48e+00 | 7.20e-15 |
  | **2** | **18** | **1.61e+16** | **1.73e+00** |

  Exactly those 18 cells exceed 1e-6, and no others. The reactor has **3,063 four-faced cells and
  every one has exactly ONE boundary face**, which is why it is unaffected; the synthetic fixture has
  18 with two, which is why it is the harsher test. A count of *boundary tetrahedra* therefore does
  not predict failure — the count of boundary faces *per* tetrahedron does.

  **✅ THE SCHEME NOW DETECTS THIS ITSELF AT BIND TIME — `_warn_if_underdetermined`.** `M2⁻¹` is
  already built there, and a healthy one is order unity (measured 2–7 across quadrilateral,
  hexahedral, and tetrahedral-under-skew); an underdetermined one runs to `1e16`. So the check is on
  the real quantity rather than on a cell-shape heuristic, and it discriminates on the **pair**:
  silent for quad/owner, hex/owner and tet/skew, and warns only for tet/owner — naming
  `SkewCorrectedGradient` as the way out. Once per process, and skipped under a tracer. Pinned by all
  four combinations, because a detector that fired on every tetrahedral mesh, or on every owner
  closure, would be useless.

  ⚠️ **WHETHER `SkewCorrectedGradient` STALLS A MARCH ON A MESH THAT NEEDS IT WAS UNTESTED FOR A WHILE;
  A CASE WITH THE RIGHT CELL SHAPE NOW EXISTS (`validation/tetrahedral_gradient_ab/`, #432) — see the
  dated entry below the `fallback` default discussion for what it found and what is still open.** The
  stall this section's earlier entries measure is on pitzDaily, which is 2D quadrilateral with **no
  tetrahedra at all**, so it never tested the regime the skew closure exists for on its own; the
  reactor march (`validation/uvreactor_openfoam/march.py`, GPU-scale) does not either, because its
  four-faced cells all have exactly one boundary face and `OwnerGradient` reconstructs it exactly
  (above). Both remain true as background — neither is the case that settles this.

  **✅✅ THE ACCURATE CLOSURE NO LONGER HAS TO BE CHOSEN FOR THE WHOLE MESH — `CellwiseFallback`
  (2026-08-24).** The two closures fail in opposite regimes and both regimes are **local**, so
  neither has to be global. `MultipleCorrectionGradient.bind` measures `M2⁻¹` per cell, and where the
  requested closure leaves it singular it rebuilds through
  `CellwiseFallback(cells, primary, secondary)` — the fallback on those cells' boundary faces, the
  primary everywhere else. `Corrections` now carries the **effective** closure, and
  `reconstruct` applies *that* rather than the requested one — otherwise the correction would be
  probed through one operator and applied through another, which is this module's own recorded trap.

  | mesh | cells repaired | gradient error | `max \|M2⁻¹\|` |
  |---|---|---|---|
  | tetrahedral | **18** | 1.31e-15 / 1.87e-14 / **3.91e-14** | 1.88e+01 |
  | hexahedral | 0 | — **bit-identical to owner** | 2.69e+00 |
  | quadrilateral | 0 | — **bit-identical to owner** | 2.53e+00 |
  | **pitzDaily (12225)** | **0** | — **bit-identical to owner**, no warning | 2.31e+00 |

  **This is what keeps the accurate answer from stalling the march ON A MESH THAT ASKS FOR THE
  REPAIR.** On pitzDaily the repair fires on **zero** cells and the reconstruction is bit-identical to
  the `OwnerGradient` arm measured at 689.8 s / 437 cycles / `x_r/h` 8.069 — so a case that asks for
  `fallback=SkewCorrectedGradient()` on this mesh never actually applies the stalling closure anywhere,
  and the march result transfers without re-running. Nothing is rebuilt when nothing is wrong, so a
  healthy mesh pays one comparison rather than a second correction build.

  **⚠️ `fallback` DEFAULTS TO `None`, NOT TO `SkewCorrectedGradient()` (corrected; it shipped the
  other way for a time).** The table above shows the repair is free wherever the mesh does not need
  it, but says nothing about whether it is *safe* wherever it does — and it is not, in general: the
  repair's only tool is `SkewCorrectedGradient`, and that closure reads a boundary value on every face
  it is installed on, which is exactly what the entry above measures destabilizing a coupled march at
  an unconverged iterate. Defaulting the fallback on installed that risk on every mesh with a corner
  tetrahedron the first time such a mesh reached a real march, with no case in this repository able to
  say whether it was safe there (no shipped mesh has both the cell shape and a march). With the default
  `None`, a mesh whose closure leaves cells underdetermined gets a warning naming the count and the
  opt-in (`fallback=SkewCorrectedGradient()`) instead of a silent repair — a caller decides with their
  own mesh and case in view, rather than the library deciding for every mesh in one direction. Nothing
  in the table above changes: it is still true that asking for the repair on pitzDaily costs nothing,
  which is the argument *for* asking when a mesh is known to need it.

  `CellwiseFallback` removes the need to apply the skew closure globally; it does not make it safe to (same log section).

  The tetrahedral duct (`validation/tetrahedral_gradient_ab/`, #432) confirmed the local repair at the `M2` level and could not get a coupled march started under either arm; that entry moved to `.claude/notes/schemes-435-tetrahedral-gradient-log.md`, "Moved from `schemes.md` (2026-10-17)".

  So: **the corner-cell defect and its local fix are as well-evidenced now as the M2 arithmetic can make
  them; whether the fix is safe under a real march remains open, blocked on #435.**

  ⚠️ **A full investigation into why the march still fails after this fix is logged in
  `.claude/notes/schemes-435-tetrahedral-gradient-log.md`** — a point-in-time snapshot (based on a
  commit before the per-field boundary binding of #467/#468/#469 and a `compare.py` harness fix
  unrelated to this scheme), not reconciled with `main` since. Read its own header before trusting
  anything in it as current.

  The Rhie–Chow damping sign probe (the second pass flips damping diagonals on tetrahedra) and `HessianCorrectedGradient` marching the duct are recorded in `.claude/notes/schemes-435-tetrahedral-gradient-log.md`, "Moved from `schemes.md` (2026-10-17)".

  ⚠️ **A design defect found by writing the fallback, and worth remembering: a defaulted `eqx.field`
  on an abstract base makes every subclass field defaulted too.** `reads_boundary_values` was
  declared that way and `CellwiseFallback` — the first closure needing *required* state — could not
  be written at all (`non-default argument 'cells' follows default argument`). It is a plain class
  attribute now, which subclasses override by assignment.

  **✅ CONSEQUENCE — `MultipleCorrectionGradient.boundary_closure` NOW DEFAULTS TO `OwnerGradient`
  (changed 2026-08-24).** The scheme is new and unreleased, so the default is set by what works:
  the owner closure marches every case here and is exact and better conditioned on quad/hex meshes.
  `SkewCorrectedGradient` remains the choice for a mesh with boundary **tetrahedra**, where the owner
  closure leaves the six Hessian components underdetermined (`cond(M2)` 9e17) — the trade is now
  explicit rather than hidden in a default. `tests/unit/test_multiple_correction.py` pairs each mesh
  with the closure that suits it, and `PITZ_AB_MULTICORR_CLOSURE` defaults to `owner`.

  **✅ RESOLVED (2026-08-24) BY PASSING THE GRADIENT IN, NOT BY A BETTER CLOSURE — `ImposedGradient`
  (`schemes/gradient.py`).** The diagnosis above is exactly right and the fix follows from it: a
  closure exists to supply a derivative nobody knows, and at a wall `omega` somebody does. It cannot
  be baked into a closure, because `omega_wall_gradient` reads `k` and `grad k` and so changes every
  residual evaluation.

  `ImposedGradient(cells, gradient)` is handed **to** `gradients()`. `GradientScheme.gradients` is
  now a template method that applies it to whatever `_reconstruct_gradient` returns — so every
  scheme honours it — and passes it down, so a scheme with somewhere earlier to put it can use it
  there. The multiple-correction scheme imposes at three points: on the first-order-exact gradient
  **before** the second pass differentiates it, on the boundary faces those cells own (in place of
  the closure), and on the returned gradient, since the second-order defect is a defect of the
  *reconstruction* and subtracting it from an imposed value would corrupt what was imposed.
  `turbulence/transport.py::_wall_omega_gradient` builds it; `closure_fields` passes it through
  `_field_gradient` → `ResidualAssembler.gradient(imposed=…)`.

  **This fixes both of the pre-existing problems the stall exposed** (neither introduced by this
  scheme, both affecting the shipped schemes equally):
  - `_imposed_wall_omega_gradient` (**gone — it is now `_wall_omega_gradient`, which RETURNS an
    `ImposedGradient` instead of applying one**) patched the gradient **returned** by `gradients()`, while
    the multiple-correction scheme builds its Hessian from its *internal* `g1` and returns
    `g2 = g1 − H2·h` — so the correction arrived **after** the quantity it should have informed. It
    now arrives before. Any scheme whose output depends on an internally reconstructed derivative had
    this problem.
  - `omega_wall_gradient`'s docstring records the magnitude nothing corrected: the reconstruction is
    ~0.26x the analytical one in the fixed cells. The imposed value is now what the Hessian pass and
    the wall faces see, rather than that quarter-sized estimate.

  ⚠️ **What it does NOT fix, and this is a real remaining gap:** the same docstring records that
  **the first interior ring reconstructs roughly twice too large**, and that ring is still overridden
  by nothing. Imposing a *gradient* cannot reach it — the ring's own first pass is a Green–Gauss sum
  over `omega` **values**, and the wall cells' values are unchanged. Fixing that would mean changing
  what `omega`'s wall boundary *value* means, which is a turbulence-model decision and not a gradient
  seam.

  ⚠️ **It reaches `ResidualAssembler.gradient` and NOT `ResidualAssembler.residual`** — the gradient
  a residual reconstructs for its own non-orthogonal correction is still unimposed, exactly as
  before. Deliberate, and pre-existing: the ω rows at those cells are the *fixation*, and the one
  path that reads their gradient is the diffusion `corr` on faces to interior neighbours, measured at
  **0.03 %** on pitzDaily (`.claude/rules/turbulence.md`). Threading it there is a bigger change (the
  imposition would have to be a property of the assembler, not an argument to one accessor) for a
  measured-negligible return.

  ⚠️ **Imposing a gradient puts the reconstruction off its own exactness contract at those cells and
  the cells that read them**, deliberately: the correction matrices are probed against the operator
  the scheme would otherwise apply (`_build_corrections` runs the closure, and knows nothing of a
  state-dependent imposition), and an imposed value is by construction not what that operator
  returns. The trade is the one the post-hoc wall-omega patch already took — a consistent operator
  around a value four times too small is worse than an inconsistent one around the right value.

  **`HessianCorrectedGradient` takes the same argument and applies it ONLY to its converged answer —
  a deliberate decision, not an omission.** Projecting the imposition onto the coupled sweep's
  gradient iterate would change the **fixed point** rather than the path to it, and that fixed point
  being the Schur solution of the gradient–Hessian system is the scheme's whole justification (see
  the `CoupledBlockSweep` entry). An imposed gradient would have to enter as a *constraint on that
  system*, which is real work and not something this seam licenses.

  **With nothing imposed every scheme is byte-identical**, pinned by
  `test_imposing_nothing_is_byte_identical_to_never_having_been_asked`. **With something imposed the
  reconstruction is AFFINE in the field, not linear** — the imposed values are an added constant, so
  `f(0) ≠ 0`. Its tangent is still the same unrolled apply and not an implicit-function solve, which
  is the half a differentiated solve depends on;
  `test_an_imposed_reconstruction_is_affine_with_the_same_linear_part` pins both halves, against the
  unimposed linearity test that pins the default path.

  Against a calibrated nested solve it is ~1.2× forward / 1.5× tangent on a skewed mesh at 5× the accuracy, slightly slower on an orthogonal one: `.claude/notes/schemes-log.md`, "`CoupledBlockSweep` — cost and boundary treatment".

  **The fixed point is provably the Schur solution**: the `h` update forces `A_HH h = A_Hg g`, and
  substituting leaves `S g = b_g`; both preconditioners are invertible so each step is an
  equivalence.

  - **The system stays gradient-sized.** No enlarged unknown, no solve on the packed `[g, H]` vector —
    `h` is an iterate of this sweep, not an unknown of a larger one. What it gives up against the
    nested path is `h`'s *transience*: it lives for one reconstruction rather than one apply
    (~115 MB at 1.6M cells). The reverse-mode tape should get **smaller**, holding one carried
    Hessian iterate per sweep instead of ten.
  - **⚠️ `h` STARTS AT ZERO ON EVERY CALL — a correctness requirement, not a style choice.** Carried
    between calls the reconstruction becomes history-dependent and stops being linear, which is the
    already-refuted warm-start from a previous step's answer. Verified: superposition 3.3e-16,
    `grad(0)` exactly zero, repeat calls bit-identical.
  - **Its `sweeps` is NOT the nested path's outer count** — `CoupledBlockSweep.calibrated(mesh,
    geometry)` measures it, sharing the calibration surface exactly (no scheme-specific keyword, so
    it joins the sibling-surface test rather than needing an exemption). Measured counts: 6 on an
    orthogonal grid, 7 at 20–30 % perturbation, 8 at 40 %.

  The 2026-08-21 boundary-treatment audit and the in-solve validation (`tests/integration/test_betchen_solve.py`) are in the same log section.
- **⚠️ THE INNER `A_HH` SOLVE NEVER NEEDED A KRYLOV METHOD — it needed a per-cell BLOCK preconditioner
  (measured 2026-08-21).** `GradientSolve` now takes an injected `GradientPreconditioner`
  (`InverseVolume` / `CellBlockJacobi`) in place of the bare `volume` array it used to stand in for,
  and `HessianCorrectedGradient` takes **two** solve strategies — `solver` (outer Schur) and
  `hessian_solver` (inner `A_HH`, default `SweptGradientSolve(sweeps=10, warn_tol=None)`) — because
  the two systems are not alike and sharing one strategy is what forced GMRES onto both.
  *Configuration for every number below: `perturbed_grid_2d` / `columnwise_perturbed_grid_3d` at the
  stated perturbation, x64, blocks and rates from a densely materialized operator.*
  - **The inverse-volume Richardson rate on `A_HH` is ρ = 0.500 on a PERFECTLY ORTHOGONAL mesh**, rising
    only to 0.63–0.70 at heavy skew. Read that first: the failure is **not** skewness, and no amount of
    mesh quality fixes it. It is the gradient and Hessian coupling to each other *within* a cell at the
    same order as the volume term, which `1/V` cannot represent at all. Against the exact per-cell block
    the same iteration is ρ = **0.000** (orthogonal), 0.055 (p=0.2), 0.127 (p=0.4) in 2D and 0.103 at
    3D p=0.3 — so a handful of fixed sweeps replaces the inner Krylov solve outright.
  - **The blocks are extracted EXACTLY with no graph colouring**, by `cell_diagonal_block` +
    `interpolation.blend_owner_neighbour`: reading a face's two sides from separate fields and zeroing
    one leaves each cell reading only its own value, so **one probe per component** (`dim`, not
    `n_colours × dim`) gives every cell's block at once. Both halves come from the *same* face kernel
    the operator is built from, so they cannot drift from it. Verified to 1e-13 against a densely
    materialized block.
  - **The OUTER Schur system was never the bottleneck** — `cond(S)` is 1.2–6.4 across mild-to-heavy skew,
    and `1/V` already gives ρ = 0.17–0.29. The block only helps at heavy skew (0.465 → 0.306 at p=0.4).
    It is used anyway (free, same machinery, never worse in measurement), but **do not go looking for
    the cost there**.
  - **`b_H` is identically zero** — the Hessian equation is a Green–Gauss sum of gradient components and
    carries no term in the field — so `rhs_g = b_g` and the old `a_gh(a_hh_inv(b_h))` was **an inner
    Krylov solve on a zero right-hand side, on every reconstruction**. Deleted.
  - **⚠️⚠️ PRECONDITION `GmresGradientSolve` ON THE RIGHT, NEVER THE LEFT — measured 2026-08-21, and
    getting it wrong broke a real case while every unit test stayed green.** These preconditioners scale
    by roughly the inverse cell volume, so on a real mesh `‖P⁻¹b‖` is ~1e6 × `‖b‖`. Left preconditioning
    (`P⁻¹A x = P⁻¹b`) hands the solver a residual measured in that norm, so its convergence **and
    stagnation** tests operate six orders from the problem's own scale — and lineax's stagnation
    detector then fires on a system it is about to solve. Measured on `pitzDaily`, reconstructing the
    **omega** gradient at the hybrid-initialized state (`‖rhs‖` 1.04e+03, gradient magnitude ~3e8):

    | arm on that exact system | result |
    |---|---|
    | unpreconditioned (the pre-2026-08-21 behaviour) | **7 steps, true relative residual 6.3e-17** |
    | left-preconditioned | **raises — stagnation** |
    | left-preconditioned, `stagnation_iters=100` | raises |
    | unpreconditioned at `rtol=1e-8, atol=0` | raises |

    Note the last row: a **looser** tolerance also raises, which is what proves this is the stagnation
    detector and not the convergence test. Right preconditioning (`A P⁻¹ y = b`, `x = P⁻¹y`) leaves the
    residual at the problem's scale and both tests see what they always saw; the whole coupled residual
    then evaluates, with the Krylov and swept outer solves agreeing to seven figures. **This is the same
    left→right correction `solve/linear.py` already made** — treat right as this package's convention.
    ⚠️ **No unit test caught it**: the test meshes carry O(1) fields and volumes, where the two norms are
    within a small factor. It surfaced only on a real case's first residual evaluation, which is the
    argument for smoke-testing a scheme against a real case before trusting it.
  - **⚠️ A DIAGNOSTIC-EMITTING SWEEP CANNOT LIVE INSIDE A TRANSPOSED OPERATOR, and the failure is a bare
    `AssertionError` from inside `lineax`.** `SweptGradientSolve`'s under-resolution check norms the
    residual, which is nonlinear; an outer Krylov solve forms its implicit-diff tangent via
    `jax.linear_transpose` of the operator, which rejects that. `stop_gradient` does **not** rescue it
    (measured). Hence `GradientSolve.requires_linear_operator` / `.emits_host_diagnostics` and an
    explanatory `ValueError`; the inner default carries `warn_tol=None`. A fixed-sweep *outer* solver
    transposes nothing and is unaffected.
  - The 2026-08-21 cost, sweep-count, swept-outer, `CellBlockJacobi`-on-`CorrectedGreenGauss` and full-coupled-system measurements are in `.claude/notes/schemes-log.md`, "The inner `A_HH` solve".

- **⚠️⚠️ THE BETCHEN RESIDUAL'S STENCIL IS ESSENTIALLY GLOBAL, WHICH BREAKS A COLOURED-PROBING
  PRECONDITIONER — measured 2026-08-21, and it is a bigger obstacle to using the scheme on a real case
  than its reconstruction cost is.** *Configuration: scalar Laplace, 10×10 randomly perturbed grid
  (`seed=1`, max cell-graph distance 18), all-Dirichlet, `dR/dφ` by `jacfwd`, mass measured as a share
  of `Σ|dR/dφ|` beyond a graph distance — the harness is `validation/gradient_stencil_reach.py`.*

  | scheme (perturb 0.2) | reach | beyond d=3 | beyond d=5 |
  |---|---|---|---|
  | `CorrectedGreenGauss` (swept 4, default) | 5 | 1.09e-05 | **0.000e+00** |
  | `CorrectedGreenGauss` (exact Krylov) | 9 | 1.09e-05 | 5.44e-09 |
  | `HessianCorrectedGradient` (default: Krylov outer, swept-10 inner) | **16** | 1.04e-03 | **2.90e-05** |
  | `HessianCorrectedGradient` (swept outer 40, swept-10 inner) | 16 | 1.04e-03 | 2.90e-05 |
  | `HessianCorrectedGradient` (swept outer 4, swept-2 inner) | 11 | 1.03e-03 | 2.66e-05 |

  - **The corrected gradient's mass beyond its own reach is EXACTLY zero**, which is precisely why
    `pitzdaily_openfoam`'s `STENCIL_REACH = 5` (`sweeps + 1`) probes an exact Jacobian. Betchen has no
    such cut-off: the gradient couples to the Hessian, which couples to the neighbours' Hessians, so
    the reconstruction at a cell depends on essentially the whole mesh.
  - **The magnitude is disqualifying against that case's own recorded sensitivity.** `pitzDaily` records
    that the ~**2e-07** left in the pressure column at reach 3 *breaks* the solve (300 matvecs, true
    residual 3.36) — because a colouring is collision-free only for its own pattern, so the far mass is
    **folded onto near entries** rather than dropped. Betchen leaves **2.9e-05** beyond reach 5, two
    orders larger than the amount already known to break it.
  - **⚠️ NARROWING DOES NOT BOUND IT, so do not reach for `narrow_gradient_sweeps` as the fix.** Outer 4
    with inner 2 still reaches 11 and still leaves 2.7e-05 beyond d=5 — the reach falls only from 16 to
    11 while the far mass barely moves. Worse, **`narrow_gradient_sweeps` cannot touch the default
    configuration at all**: it rewrites `SweptGradientSolve` nodes, and Betchen's default outer solver is
    `GmresGradientSolve`, so it returns the tree unchanged and a caller who believes it narrowed
    something gets a silently unbounded stencil. Treat a narrowed Betchen scheme as narrowed only if its
    outer solver is swept, and check the reach rather than assuming it.
  - **On nearly-orthogonal pitzDaily it does not bite**, and the reach that matters depends on the smoother family — do not carry a reach calibration across smoother families: `.claude/notes/schemes-log.md`, "The Betchen residual's stencil on pitzDaily".

- **Current status of the 2026-08-21 cost investigation:** a Krylov outer gradient solve costs ~5× a swept one on the `jvp` path (hence fixed sweeps throughout); the reconstruction is 12–20 % of a pitzDaily march step even for the cheap scheme; and the sweep count is the cost lever, with a contraction rate that varies ~50× across meshes — so it is calibrated per mesh (below) rather than guessed. Tables in `.claude/notes/schemes-log.md`, "Krylov outer cost, the first Betchen A/B, and the sweep-count lever".

- **BUILD-TIME SWEEP CALIBRATION IS BUILT (2026-08-21); `GradientScheme.bind` REMAINS PROPOSED.** The
  two were recorded together because two investigations arrived at the same place from different
  directions, but they turned out to be separable: calibration needs a concrete-geometry,
  once-per-case boundary, and a classmethod factory *is* one — it does not need `bind` to exist. What
  shipped, in `aquaflux/schemes/gradient.py`:
  - **`GradientSystem(preconditioner, operator, shape)`** — the triple a solve strategy consumes and
    the estimator measures, so a count is never calibrated against a different assembly than the one
    that runs. `CorrectedGreenGauss.system(terms)` returns it (and `gradients` now goes through it,
    which is where that system's choice of `InverseVolume` lives); `HessianCorrectedGradient._systems`
    returns its two, with the outer taking its inner solve **injected** so it is measured with the
    inner solve that will run inside it.
  - **`contraction_rate(system, *, iters=24, seed, norm) -> ContractionRate`** — the Gelfand estimate
    `(prod ‖M^k v‖/‖M^(k-1)v‖)^(1/iters)`, `M = I − P⁻¹A`, in ONE jitted `fori_loop` with one host
    sync. `ContractionRate.settling_ratio` is `rho(iters)/rho(iters/2)`, the settledness self-check;
    **measured 1.03–1.17** across orthogonal → 40 % perturbed and over both Betchen systems.
  - **`SweepCalibration(tol=1e-4, iters=24, floor=1, cap=64, seed=0)`** — a frozen dataclass, so
    `SweepCalibration.tol` at class level IS the default and both factory signatures read it rather
    than restating a literal. Validated in `__post_init__`, so an inconsistent one cannot exist.
    `.sweeps_for(rate)` is the pure conversion; `.sweeps(system)` measures then converts.
  - **`CorrectedGreenGauss.calibrated(mesh, geometry, ...)` and
    `HessianCorrectedGradient.calibrated(mesh, geometry, ..., schur=True)`**, both written against the
    one private tail `_calibrated_solver` — which is also the **only** place a calibrated
    `SweptGradientSolve` is constructed (an earlier draft had `SweepCalibration.solver` doing it too,
    i.e. two builders of one class, and it was deleted).
  - **Costs `iters` applies once.** A `k`-sweep reconstruction spends `k − 1` applies, so the default
    budget is about eight four-sweep reconstructions, mesh-size-independent (both sides linear).
  - ⚠️ **Traced geometry raises a named `ValueError` ("geometry is traced; calibrate outside the
    differentiated region and pass `sweeps=` explicitly")** rather than surfacing a
    `ConcretizationTypeError` from inside a logarithm. Checked up front on the preconditioner's
    leaves *and* caught at the host conversion. Correct rather than a compromise: an integer count has
    zero derivative almost everywhere, and the state and its adjoint use the same count either way.
  - **`validation/uvreactor_openfoam/gradient_sweep_calibration.py` is NOT subsumed and stays.** It
    walks the sweep ladder against a converged reference and measures the *reconstruction's* error;
    the estimator measures the *operator's* rate. That makes it the independent check on the
    estimator rather than a duplicate of it — which is what an unfalsifiable estimator would
    otherwise be missing.
  - **Not shipped: any change to a default.** `SweptGradientSolve.sweeps` is still 4 and Betchen's
    still 20/10. The factory is opt-in. Each case's own measured count is recorded in that case's
    README, beside the mesh it was measured on.

- **⚠️⚠️ THE CALIBRATION TOLERANCE IS AN L2 TOLERANCE AND THE WORST CELL EXCEEDS IT ABOUT HALF THE
  TIME — measured 2026-08-21, and it REVERSES the caution recorded above.** Configuration: shipped
  `CorrectedGreenGauss.calibrated(mesh, geometry, tol=1e-4)` (`InverseVolume`, Gelfand estimate at the
  default 24-apply budget), judged against the same system solved by `GmresGradientSolve`, over 26
  combinations — pitzDaily's `of_case` mesh plus 2D 20×20 grids at 5/10/20/30/40 % perturbation (two
  seeds each) and 3D 8³ columnwise grids at 15/30 %, two analytic fields each.

  | quantity | result |
  |---|---|
  | L2 relative gradient error ≤ `tol` | **26 of 26** |
  | worst single cell's relative error ≤ `tol` | **13 of 26** |
  | worst-cell ÷ L2 ratio | min 5.2× · median 11.8× · max 59.6× |

  The mechanism is that `ρ` is the iteration's dominant eigenvalue, which governs a norm and not the
  extremes: on pitzDaily the calibrated `k = 2` leaves L2 at 5.4e-06 (eighteen times better than
  asked) while the worst cell sits at 3.2e-04, i.e. **above** the tolerance. So the recorded worry —
  that a few bad cells would drag the count up and over-resolve everything else — is exactly backwards:
  the bulk is over-served and the bad cells are under-served. Ask for an L2 tolerance one to two orders
  tighter than the per-cell accuracy actually wanted. The docstrings state the norm for this reason.

- **The estimator needs NO safety margin, and it is a cancellation rather than luck — reproduced
  2026-08-21 on the shipped code.** 195 combinations (thirteen meshes × three fields × five tolerances
  from 1e-2 to 1e-8), configuration as above: **0 misses in the L2 norm**. The two errors are opposite
  and comparable — the Gelfand estimate approaches `ρ` from below (a random start's deficiency in the
  dominant eigendirection contributes `|c|^(1/iters)`, measured as a ~6 % deficit on a synthetic with
  a clean spectral gap), which alone would under-count, while the iteration's early transient reduces
  the error faster than `ρ^k`, which over-delivers by about as much. Do not add a margin "to be safe":
  it would double counts that already hit their target every time. ⚠️ This is the **L2** statement —
  see the entry above for the per-cell one.

- The per-case calibrated counts (pitzDaily `k = 2`, bfs3d `k = 1` at 1e-4), the pre-build `bind` proposal (since built, above) and the probe-narrowing marches (narrowing the probe copy beats lengthening the probe) are in `.claude/notes/schemes-log.md`, "Build-time sweep calibration, the `bind` proposal, and probe narrowing".

- **`SweptGradientSolve(sweeps)` — BUILT. The scalable `GradientSolve` strategy**, injected via
  `CorrectedGreenGauss(solver=SweptGradientSolve(n))` — **not a separate scheme** (same
  discretization as the GMRES path; only the `A_g⁻¹` apply differs). This is the efficient realization
  of absorbing the gradient into the flow system: `A_g` is geometry-only and constant, so its inverse
  can be applied far more cheaply than a fresh implicit-diff GMRES each call. Both solve strategies
  consume `CorrectedGreenGauss`'s reusable pieces — `terms(mesh,geom)` (geometry intermediates),
  `operator(terms)` (the constant matvec `A_g`), `rhs(terms,field,bvals)` (`B·φ`). It applies `A_g⁻¹`
  by a **fixed number of matrix-free preconditioned Richardson sweeps**
  `g ← g + P⁻¹(B·φ − A_g·g)` for the injected `GradientPreconditioner` `P` — `InverseVolume` here, so
  `P⁻¹ = V⁻¹` (converges because `V` dominates `A_g`), differentiated by unrolling the
  short static loop — **no dense matrix, no nested Krylov, no implicit-diff tangent solve**. Sweep
  count to machine precision is **mesh-independent** (⇒ genuinely `O(n)`): 12 sweeps at 0.1 skew, 16 at
  0.2, 24 at 0.3 (grows with skewness, not mesh size). Exact drop-in (3.8e-10 vs GMRES). **~5× faster
  than `GmresGradientSolve` at N=32** (per coupled Newton step 112 s → 23 s run, 96 s → 23 s compile —
  the compile collapse shows the nested Krylov + implicit-diff control flow *was* the blow-up).
  Validated in the coupled skewed cavity (`tests/integration/test_swept_gradient_flow.py`): converges,
  matches the GMRES solution, differentiable through the nonlinear solve. The Schur complement
  `∂R/∂x + (∂R/∂g)A_g⁻¹B` is still formed by AD; only `A_g⁻¹` changes from a nested solve to a cheap
  unrolled sparse apply. **The default `GradientSolve` for `CorrectedGreenGauss`** (every flow mesh,
  not only skewed ones) — cheap to differentiate inside a nonlinear Newton, where the `GmresGradientSolve`
  nested Krylov+implicit-diff alternative is impractical (see the `CorrectedGreenGauss` note above).
  - **The FIRST sweep's operator apply is peeled, exactly — `sweeps` sweeps cost `sweeps - 1` applies
    (2026-08-20).** The iteration starts at `x = 0`, where `rhs - A·0` is `rhs` outright, so that apply
    computed a known answer at full price. **Nothing downstream removed it:** the compiler folds the
    gathers against the zero constant but *not* the scatters, measured on the real scheme as 30 → 24
    scatter operations at the default `sweeps=4`. Worth ~9–17 % of the reconstruction on a 40 000-cell
    randomly-graded 2D grid, which is charged on **every residual evaluation and every Jacobian--vector
    product**, so it is inside every coupled Newton step. Bit-identical (`jnp.array_equal`), pinned two
    ways in `tests/unit/test_gradient.py` — an apply *count* per sweep count (which fails if the peel is
    reverted) and an equality against the unpeeled iteration written out (which fails if the arithmetic
    is ever rearranged). This is the same peel `_VCycleOps.smooth_zero` already carries (as did the since-deleted
    `Ilu0.sweep_from_zero`); the gradient solve was the one place in the tree still missing it.
    ⚠️ **It does NOT move the stencil reach**, which is what makes it safe next to the entry below: the
    peeled apply is the one against a zero vector, so it contributed no coupling. A `k`-sweep
    reconstruction still reads `k` cells out (it applies `A_g` `k-1` times on top of `B·φ`'s own ring),
    and the measured table below is unchanged. **A synthetic operator with random column indices put the
    same peel at 34 %**, twice the real figure — the locality caution recorded elsewhere in these files,
    reproduced exactly.

  - **⚠️ EACH SWEEP COUPLES ONE FURTHER RING, so the sweep count sets the RESIDUAL's stencil reach —
    and the shipped `sweeps=4` is inconsistent with the shipped `stencil_reach=3` on a skewed mesh
    (measured 2026-08-16, harness `validation/gradient_stencil_reach.py`).** `A_g` couples a cell to its
    face neighbours, so `k` sweeps make the reconstruction read `k` cells out; a residual built on it
    reads `k + 1` (a face flux gathers the gradient of the cells on both sides). Measured exactly, at
    every skewness, on scalar Laplace on a 12×12 randomly perturbed grid, all-Dirichlet:

    | sweeps | 1 | 2 | 3 | 4 | 8 | exact (GMRES) |
    |---|---|---|---|---|---|---|
    | reconstruction reach | 1 | 2 | 3 | 4 | 8 | to round-off |
    | scalar residual reach | 2 | 3 | 4 | 5 | 9 | 9–11 |

    **`sweeps=1` IS compact Green–Gauss** (`x₁ = V⁻¹Bφ`), correction and all reach included, so the
    useful range starts at 2. On the coupled RANS residual the base terms carry rings of their own —
    measured on a 6×6 skewed lid-driven cavity (first-order upwind, `DirectScalars`): reach **6** at
    `sweeps=4`, **4** at 2, i.e. `sweeps + 2` there. **So a reach is a property of the assembled case
    and must be measured, never derived from the sweep count.**
  - **⚠️ TRUNCATING THE SWEEPS DOES NOT REMOVE FAR COUPLING — it FOLDS it onto the last retained shell,
    and this is the finding that decides the design.** The mass beyond a given distance is set by the
    **mesh skewness**, not by the sweep count. At 25 % perturbation, `|dR/dφ|` beyond distance 2 is
    9.80e-4 at 2 sweeps, 1.004e-3 at 4, and 1.004e-3 at the exact solve — flat. Per-ring decay is
    ~1/37 at 25 % skew, ~1/6.6 at 40 %, ~1/240 at 5 %. Accuracy of the reconstruction against the exact
    solve, same runs: 3.6e-3 (2 sweeps) / 3.1e-5 (4) / 1.7e-8 (8) at 25 % skew; 1.1e-2 / 9.9e-4 / 2.5e-5
    at 40 %.
    **Consequence, and the reason "solve it exactly with GMRES instead" is the WRONG lever:** the exact
    solve is the sweep series run to round-off, so it has *strictly more* mass past any distance, not
    less — measured reach 9 at 25 % skew and 11 at 40 %, against 5 for `sweeps=4`. It also does not
    address why GMRES was rejected as the default (the nested implicit-diff tangent re-entered per jvp,
    ≈180× on pitzDaily), which a preconditioner inside that solve does not remove.
  - **`narrow_gradient_sweeps(tree, sweeps)` — BUILT (2026-08-16), the cap, for the PRECONDITIONER only.**
    Returns a copy of any tree (an assembled case, an assembler, a scheme) with every `SweptGradientSolve`
    in it capped; it only ever narrows (a solve already at or below the cap is returned by identity), and
    a `GmresGradientSolve` or `CompactGreenGauss` is untouched. It rebuilds each `equinox.Module` along
    the path with `dataclasses.replace` rather than `eqx.tree_at`, because `sweeps` is a **static** field
    and so lives in the treedef, not among the leaves — `tree_at` raises on it (`SweptGradientSolve` is an
    all-static Module, i.e. an *empty* pytree node, which `where` cannot locate). Every Module in
    `aquaflux` takes its fields as constructor arguments and none defines `__post_init__` / `__check_init__`,
    which is what makes `replace` faithful.
    **Why a cap rather than an exact solve:** a coloured probe recovers the Jacobian to a fixed distance
    and *folds* whatever lies beyond onto near entries, so probing a narrowed residual gives a matrix that
    is **exact for the residual it was taken from** — a stated approximation of the operator instead of a
    corrupted one. Consumed through `JacobianProbe(gradient_sweeps=…)` / the coupled builders'
    `probe_gradient_sweeps=`; see `.claude/rules/turbulence.md` and `.claude/rules/solve-direct-preconditioners.md`. **Default
    `None` everywhere is byte-identical.**
    The skew census showing the trap is live on pitzDaily (reach 5 needed at `sweeps=4`) is in `.claude/notes/schemes-log.md`, "`narrow_gradient_sweeps`: latent on bfs3d, live on pitzDaily".
  - **The `GradientSolve.solve(..., operator_hook=None)` distributed seam.** `operator_hook` is an
    optional transform applied to the unknown before **every operator apply**. `SweptGradientSolve`
    honours it — the Richardson sweeps form no global inner product, so a domain-decomposed residual
    can pass its ghost-cell exchange here to refresh the iterate's ghost rows each sweep, making the
    owned gradients serial-exact. `GmresGradientSolve` **raises** on a non-`None` `operator_hook` (its
    inner products span the whole local vector, double-counting ghost rows and unreduced across
    partitions), as does `HessianCorrectedGradient` (its nested Schur/`A_HH` solves read ghost
    gradients *and* Hessians the outer exchange does not refresh). This makes `SweptGradientSolve` the
    one gradient solve that runs under domain decomposition (the distributed non-orthogonal path; see
    `.claude/rules/parallel.md`).
- **Rejected alternative — dense LU of `A_g` (built, measured, removed; do not rebuild).** Factorizing
  the constant `A_g` once (dense, via `jit`-ed `jacfwd` + `lu_factor`) and applying `A_g⁻¹` by
  back-substitution is also exact, but dense ⇒ `O((n·dim)²)` per apply, so it is **strictly dominated by
  the swept solve at every mesh size** (measured run/step: N=12 0.27 vs 0.21 s, N=16 2.0 vs 0.83 s,
  N=24 24 vs 4.8 s) and crosses over to *slower than even the iterative baseline* by N=32. A scalable
  sparse LU in JAX needs host callbacks (off-GPU), so the matrix-free swept apply is the right sparse
  realization, not a factorization.

## Binding decisions
- **Physics and numerics are separate.** Scheme
  classes live here; operators in `discretization/` consume them via constructor
  injection. An operator never inlines a scheme choice.
- **Scheme classes are the DRY mechanism** (CLAUDE Principle 2): one scheme class defined
  once, injected into many operators/equations. Never copy a reconstruction into two
  operators.
- **Published bottleneck to respect:** Gauss gradients are not formally 2nd-order and
  cap accuracy on skewed grids — for *both* the primary and differentiated fields
  (DeGroot 2019). This is *why* the block is
  swappable. Keep the interface clean enough that upgrading it is a drop-in.

## Testability seam
Each scheme is tested by reconstructing an analytic field on a refined-mesh sequence and
asserting the measured order of accuracy — with **no physics involved** (the gradient's
exact oracle is `∇f` of a known `f`). Use `tests/support/meshes.py::perturbed_grid_2d` for
the refinement sequence; **measure error on interior cells only** (boundary cells reconstruct
at lower order and pollute the rate), and use **random** perturbation (not smooth) to
expose the true skewed-grid order (smooth perturbations cancel errors and flatter the
scheme). This harness is also the experiment that decides whether the implicit gradient
earns its Schur coupling.
