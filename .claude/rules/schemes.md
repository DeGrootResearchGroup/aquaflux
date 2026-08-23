---
paths:
  - "aquaflux/schemes/**"
---

# Rules — `aquaflux/schemes/` (first-class swappable numerics)

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

**⚠️ THIS ENTRY WAS ALREADY STALE WHEN FOUND (2026-08-20) — the diagnostic is gated, and has been for
longer than this file said.** It read "`SweptGradientSolve`'s `warn_tol` diagnostic fires
**unconditionally at `sweeps=1`**"; `solve` in fact skips it entirely when `sweeps <= 1`
(`... and self.sweeps > 1`), where it would carry no information rather than a little. What remains
true, and is the reason the entry is kept rather than deleted: the diagnostic measures the residual
from *before* the last update, so at one sweep that is the right-hand side itself and the ratio is
exactly 1 whatever the mesh — which is why gating it was the fix. **If you meet that warning in an old
log from a skew-free mesh, it was never evidence of non-orthogonality.**

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

**Two tests had to change lens, and neither was weakened — this is the instructive part.**
- `test_the_swept_solve_spends_one_apply_fewer_than_its_sweep_count` counted applies with a **Python
  counter inside the operator**. A scan body is traced **once** however many times it executes, so
  that counter reads `1` at every sweep count and the test would have passed whatever the peel did.
  It now reads the scan's **trip count out of the jaxpr**, which is what actually executes.
- `test_peeling_the_zero_apply_leaves_the_answer_BIT_identical` compared the peeled solve against an
  unpeeled **Python loop**. With one arm scanned it was comparing loop constructs rather than the
  peel, and failed by ~1 ulp. The reference is a scan too now, and bit-equality is restored — the peel
  itself is exact, since `A·0` is exactly zero.

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
`coupled_lu_continuation`, `coupled_amg_continuation`, `mass_flow_coupled_continuation`) →
`_coupled_step` → `ShiftedStep.jacobian_residual` → `_shifted_solve(jacobian_fn=…)`. It narrows the
gradient's sweeps in the copy of the residual the Krylov **operator** is differentiated from, via the
same `narrow_gradient_sweeps` the probe already uses. `None` everywhere is byte-identical. It is a
**third** thing to narrow and the three must not be conflated: `sweeps` is the residual's (the
discretization), `probe_gradient_sweeps` is what the preconditioner *materializes*, and this is what
the Krylov iteration *applies*.

**Per-matvec headroom, priced first** (pitzDaily 12225 cells, `state-00082`, `eqx.filter_jit`, warm,
min of 7, x64, compiled ILU(0) live):

| gradient sweeps | operator applies | `R` (ms) | `jvp(R)` (ms) |
|---|---|---|---|
| **4 (shipped)** | 3 | 3.85 | **8.27** |
| 3 | 2 | 2.97 | 6.71 |
| **2** | 1 | 1.97 | **4.63** |
| 1 | 0 | 1.33 | **3.72** |

A sweep costs ~0.9 ms in the residual and ~1.6 ms in the tangent, because **`jvp` spends two operator
applies per sweep — a primal and a tangent — against the residual's one**. (The residual is nonlinear
in the reconstructed gradient, through `nu_t` and the limiter, so the primal gradient cannot be
eliminated.) Hence the sweeps weigh nearly twice as much on the differentiated path as on the evaluated
one, and a march pays a tangent per Krylov iteration against a residual once per step.

**On a whole march (three arms back to back in one process, `simplesmooth` bundle, three Reynolds
rungs, `forward_rtol` 0.3, probe reach 5, residual held at swept-4):**

| Jacobian | steps | Krylov cycles | worst step | wall | `x_r/h` | field difference vs control |
|---|---|---|---|---|---|---|
| **full (exact) — control** | 71 | 404 | 14 | 713.3 s | 8.069 | — |
| **swept-2** | **71** | **404** | **14** | **644.0 s (−9.7 %)** | 8.069 | ≤ 1.2e-07 L2 |
| swept-1 | 71 | 410 | 15 | 644.4 s | 8.069 | ≤ 2.3e-06 L2 |

- **swept-2 is free: identical step count, identical cycle count, identical worst step, and the same
  root.** The step tables are equal column for column — `beta`, inner count, cycles, `alpha`, flags and
  residual to four figures — with only the wall-clock column moving. The whole 9.7 % is cheaper
  matrix-vector products.
- **THE KNEE IS AT 2, AND swept-1 IS PAST IT.** It costs 6 cycles (+1.5 %) and returns nothing in wall
  clock (644.4 s against 644.0), even though its tangent is a further 20 % cheaper per matvec. That is
  the exchange rate this idea lives or dies on, seen directly: below some accuracy the extra iterations
  eat the cheaper product. **Run the ladder one rung past where it stops helping** — the two endpoints
  alone would have read as "keep going".
- **Why 2 is exactly right here is predictable in advance, and that is the useful part.** This mesh's
  measured contraction rate is `rho = 5.07e-03`, so a swept-2 gradient departs from swept-4 by
  `~1.4e-05` relative and a swept-1 one by `~4.0e-03`. Against a linear solve stopping at 30 %, the
  first is invisible and the second is at the edge. **Size the cap from `contraction_rate`, not by
  feel** — and note the residual's own calibrated count on this mesh is also 2, so at `tol=1e-4` the
  two questions happen to give the same answer here; they will not in general.
- ⚠️ **One run per arm**, but the wall clock here is better anchored than usual. The **step and cycle
  counts are contention-immune and carry the verdict** — bit-identical trajectories are not a
  wall-clock claim — and the three arms ran back to back in one process. Beyond that, the control arm
  was run **twice**, in two separate invocations on the same machine, at **713.3 s and 720.1 s** with
  identical steps, cycles and reattachment: a **~1 % march-level repeatability**, which is what puts
  the 9.7 % comfortably outside it. That is a much finer instrument than this project's recorded ~15 %
  *per-application* noise floor, and the difference is worth knowing: a whole march averages away the
  per-application spread that a single-state probe is at the mercy of.
- ⚠️ **Measured on `simplesmooth`, not on the `petsc` bundle this case shipped until the same day** (see
  the case comment: `petsc` no longer marches it). Cycle counts do not transfer across preconditioner
  families.
- **The adjoint is untouched, by construction rather than by care.** `_implicit_solve_bwd` differentiates
  the residual it was handed at the converged state and never consults the forward step, so a cheaper
  forward operator cannot reach a gradient. See `.claude/rules/solve-globalization.md`.
- **Not measured: a genuinely skewed mesh, where `rho` is 0.14–0.26 rather than 5e-03.** There the
  residual needs many more sweeps and the *ratio* between the two counts should be much larger — which
  is where this lever should pay most, and where it is also most likely to start costing steps. Both
  shipped cases are near-orthogonal (`bfs3d` calibrates to `k = 1`), so neither can answer it.
  **The UV reactor is the wrong instrument for it** despite being the one skewed mesh in the tree: at
  1.6M cells a single march is far too expensive to walk a ladder on, and the ladder — not any single
  arm — is what carries the verdict here (the knee at 2 was only visible because 1 was also run).
  **What this needs is a SMALL skewed case**, on the order of the two existing ones, and the natural
  source is an automatically-generated mesh over a simple geometry rather than a perturbed grid: a
  synthetic perturbation makes `rho` a knob, where the question is what a real mesh generator produces.
  The strongest form is a **re-mesh of a geometry already validated on a block mesh**, so mesh quality
  is the only variable against a known answer and the metric stays the judge.
  **⚠️ BUT THE GEOMETRY MUST NOT BE AXIS-ALIGNED, AND `bfs3d` IS — so re-meshing IT would produce
  another orthogonal mesh.** An automatic hex mesher distorts cells only where it must **snap** to a
  surface the background mesh does not already conform to; a box with an axis-aligned step castellates
  and stops, leaving the background hexes intact. This project's own two meshes are the demonstration:
  `bfs3d` is a pure box and is skew-free to `1.9e-12`, while `pitzDaily` — the same class of geometry
  but with an **inclined lower wall and a contraction** — reaches `7.5e-02`. The non-alignment is where
  the skew comes from. So the candidate geometry needs a genuinely angled or curved surface (and
  refinement-level transitions, the other source, help), and `pitzDaily`'s own geometry is the better
  host for the idea than `bfs3d`'s for exactly this reason.
  **⚠️ Whatever is chosen, MEASURE THE MESH BEFORE BUILDING A CASE ON IT** — generate it, then run
  `contraction_rate` and a skew census. It costs minutes, and the failure it prevents is a case built
  around a mesh that turns out orthogonal, which is precisely why `bfs3d` cannot answer this question
  despite being the newer and better-tuned of the two.

**And the other half of the question, measured on the same case: the RESIDUAL's exactness buys nothing
here either — because the sweep series has already converged by two.** Same harness, `residual` group,
each arm's probing reach moved with its sweep count (`sweeps + 2`, so the probe is not the thing
degrading):

| residual | probe reach | steps | cycles | wall | `x_r/h` | field difference vs swept-4 |
|---|---|---|---|---|---|---|
| swept-2 | 4 | 71 | 391 | 545.8 s | 8.069 | ≤ 2.2e-06 L2 |
| **swept-4 (shipped)** | 5 | 71 | 404 | 720.1 s | 8.069 | — |
| swept-6 | 7 | 71 | 416 | **1059.1 s** | 8.069 | ≤ 7.1e-07 L2 |

**A 1.9× cost spread across the ladder, for a converged field that moves by ~1e-06 relative and a
reattachment length that does not move at all.** Read it as the sweep series having converged rather
than as accuracy being unnecessary: swept-2 and swept-6 each sit ~1e-06 from swept-4 *and from each
other*, which is what "all three are the same discretization" looks like. The direct answer to "what is
the exactness buying?" on this mesh is **nothing measurable** — and that is a property of `rho = 5.07e-03`,
not a general licence.
⚠️ **Do NOT read this as an argument to lower the shipped `sweeps=4`.** The same count is *insufficient*
at 30 % skew, the failure is silent (there is no residual test to trip), and the count also sets the
residual's Jacobian reach, which each case's probing reach is matched to — so moving it is a change to
the discretization that the case's reattachment result would have to be re-validated against. The
per-case answer is `CorrectedGreenGauss.calibrated`, which already gives **2** on this mesh.
⚠️ **The wall-clock column of this group mixes two changes** — the arms differ in probe reach as well as
in sweep count, deliberately, since a probe shorter than the residual's stencil would fold the far
coupling onto near entries and degrade the arm for the wrong reason. The **field** comparison is what
this group is for; the seconds are indicative.
**Contrast the two groups, because that is the finding.** Both cheapen the reconstruction and both leave
the answer where it was — but the residual group does it by *changing the discretization and being
lucky that this mesh does not care*, while the Jacobian group does it *without touching the
discretization at all*. Only the second is safe on a mesh nobody has calibrated.

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
- **`limiter.py` — BUILT.** `Limiter` (interface) → `VenkatakrishnanLimiter(k)`: a per-cell slope
  limiter `psi ∈ [0,1]` (smooth Venkatakrishnan 1993, `eps² = vol K³`), `limit(field, gradient,
  face_cells, geometry)`. Physics-free (verified in `tests/unit/test_limiter.py`), injected into
  `LimitedUpwind(limiter=…)` in `discretization/advection.py`, and evaluated only when that scheme
  runs (a diffusion-only or first-order solve never forms `psi`). See `.claude/rules/discretization.md`.
- **`gradient.py` — BUILT so far:** `GradientScheme` (interface) → `CompactGreenGauss`
  (one-shot `∇φ_P = (1/V_P) Σ φ_ip S_f`, linear-interpolated interior faces). Verified in
  `tests/unit/test_gradient.py`: linear-exact + 2nd-order on orthogonal grids;
  **inconsistent (order ~0) on irregular grids** — the deliberately-demonstrated
  Green–Gauss deficiency. Differentiable (`jax.grad` flows).
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

  **Robustness on degenerate cells (measured 2026-08-22, VALID single-cell fixture).** At a volume
  ratio of 1.9e+06 the shipped swept solve leaves **1.09e+00** on the sliver cell and **4.94e-07** in
  the far field, against a clean mesh's 8.6e-15 — so a single bad cell does cost the scheme its
  exactness globally. But this is an **iteration** effect and not a discretization one: an exact
  Krylov solve on the same mesh gives **1.79e-10** on the sliver and **2.41e-15** far. The assembled
  system holds the right answer and the fixed-sweep Richardson fails to find it, because the
  preconditioner is `A_gg`'s diagonal block while the operator is the Schur complement — and on a
  cell whose volume vanishes the neglected elimination term stops being a perturbation. Two guards
  are absent and worth knowing: `jnp.linalg.inv` on the per-cell blocks has no singularity check, and
  `interpolation_factor` guards `d → 0` on boundary faces only; neither has been observed to trigger.

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

  **⚠️⚠️ THE DEFAULT WENT `True` → `False` → `True` IN ONE DAY (2026-08-22). It is `True`. The middle
  step is kept because what moved it back is a change to the SCHEME, not a re-reading of the same
  evidence — with the Hessian solved as nine components the divergence below is real and reproducible;
  with six it is gone and the other arm diverges instead. Read the entries in order.**

  **⚠️⚠️ THE NINE-COMPONENT MEASUREMENT — IT DIVERGED ON A REAL
  MESH, AND THE REASON INVERTS THE OBVIOUS INTUITION.** On a 1.6M-cell snappyHexMesh reactor,
  measured in one process against a known analytic gradient
  (`validation/uvreactor_openfoam/schur_block_diagnosis.py`):

  | | median | p99 | max | cells > 100 % error |
  |---|---|---|---|---|
  | `A_gg` block | 1.137e-07 | **4.876e-05** | **7.011e-02** | 0 |
  | Schur block | **8.957e-08** | 7.398e-05 | **5.599e+18** | **4599 of 1 635 909** |

  A 27 % better median, a worse p99, and 0.28 % of the mesh diverging outright. ⚠️ **"A population,
  not an edge case, so no per-cell guard is proportionate" was the reading here and it is REFUTED**:
  the divergent mode's participation ratio is ~1 cell (below), so the population is one bad row's
  contamination spread by 20 sweeps, and a per-cell guard is exactly the right shape after all.

  **The failing cells are ORDINARY.** Four- and five-faced cells (tetrahedra and pyramids) against
  the mesh's median of six, volumes 0.24–0.42 of median (one is 9× *larger*), planarity 0.89–0.9996,
  closure ~1e-22. Not slivers, not warped, not invalid — every geometric hypothesis was refuted.

  **⚠️ AND THE SCHUR BLOCK IS THE MORE ACCURATE ONE. That is the finding.** Against the true local
  Schur block — obtained preconditioner-independently by lighting one cell's degree of freedom and
  reading back its own row:

  | cell | faces | `A_gg` vs truth | Schur vs truth | cond Schur | error |
  |---|---|---|---|---|---|
  | 1595711 | 4 | 3.284e-01 | **2.452e-02** | 2.6e+03 | 5.6e+18 |
  | 60355 | 5 | 2.535e-01 | **6.025e-02** | **6.37** | 1.4e+17 |

  **⚠️⚠️ SOLVING THE HESSIAN AS SIX COMPONENTS REMOVES THE DIVERGENCE ENTIRELY — AND SWAPS WHICH ARM
  DIVERGES (measured 2026-08-22).** Reproduce with
  `UV_MESH=<polyMesh> validation/run_case.sh validation/uvreactor_openfoam/schur_block_diagnosis.py`
  — the run logs are gitignored, so the harness and its defaults are the record. Same 1.6M-cell
  `snappyHexMesh` reactor mesh both times, same analytic quadratic, scheme defaults of 20 outer
  sweeps and `UV_INNER=12` on the Hessian, both preconditioner arms in one process. Judged against the analytic gradient:

  | Hessian | preconditioner | median | p99 | max | cells > 100 % |
  |---|---|---|---|---|---|
  | nine components | `A_gg` block | 1.137e-07 | 4.876e-05 | 7.011e-02 | 0 |
  | nine components | Schur block | 8.957e-08 | 7.398e-05 | **5.599e+18** | **4599** |
  | six components | `A_gg` block | 8.147e-08 | 4.918e-05 | **5.787e+05** | — |
  | **six components** | **Schur block** | **8.055e-08** | **4.320e-05** | **5.184e-03** | **0** |

  **The Schur block improves by twenty-one orders in the max and no longer diverges anywhere**, and
  the `A_gg` block — which was the safe arm — now blows up instead. Every diverging cell in both
  regimes is a four-faced cell at ~0.2 of the median volume, so it is the same population changing
  hands rather than a new one.

  **This is a controlled comparison, and that was checked rather than assumed.** Diffing
  `aquaflux/schemes/gradient.py` across every commit between the two runs, the only change to the
  reconstruction other than the symmetry reduction is `outer`'s `use_local_schur_block` default —
  which the harness passes explicitly on both arms, so it is inert here. The two intervening merges
  add opt-in paths (`CoupledBlockSweep`, the probe's gradient narrowing) that this harness does not
  select.

  **Consequence: `local_schur_block=False` is now the DOMINATED option, which inverts the default
  question rather than settling it.** The reason to prefer `A_gg`'s block was that the Schur block
  diverged on this mesh; it does not any more, and `A_gg`'s does. Note also that the under-resolution
  warning fires on the `A_gg` arm at 20 sweeps and not on the Schur arm — the fixed sweep count is
  visibly short for it, which is the same failure seen from the solver's side.

  **Why the representation should decide this at all**: the elimination term is
  `A_gH A_HH⁻¹ A_Hg`, and with the unsymmetrized Hessian `A_HH`'s per-cell block was neither symmetric
  nor definite, so the correction subtracted from `A_gg`'s block could be wrong-signed. The reduction
  makes that block `(A_P E)ᵀ(A_P E)` — SPD by construction — so the correction is a well-signed one.
  ⚠️ That is the mechanism the measurement is consistent with, **not** one this run isolates: the
  reduction also changes `S` itself, since a different reduction of an over-determined system picks a
  different solution. Do not report it as demonstrated.

  **THE RATE INVERTS WITH THE ARMS, AND PREDICTS BOTH MAGNITUDES.** `rho(I - P⁻¹S)`, 20 power
  iterations, half-budget estimate beside it:

  | Hessian | `A_gg` block | Schur block |
  |---|---|---|
  | nine components | 0.7148 ✓ | **6.2679 ✗** |
  | six components | **2.0397 ✗** (half 2.0199) | **0.3070 ✓** (half 0.2223) |

  `2.04²⁰ ~ 1.5e+06` against an observed max of 5.8e+05, and `6.27²⁰ ~ 1e+16` against 5.6e+18 — the
  rate accounts for the error magnitude in both regimes, which is what ties the spectral measurement
  to the reconstruction one rather than leaving them as two separate observations. The Schur block
  under six components has the best rate of the four. ⚠️ Its half-budget estimate is the least
  settled of them (0.2223 against 0.3070, a ratio of 1.38 where the others are within 1 %), so treat
  **0.31 as approximate** — unambiguously below one, not accurate to two figures.

  **⚠️⚠️ AND THE FAILURE IS LOCAL, NOT GLOBAL — this REFUTES the reading recorded here on
  2026-08-22, which was mine and which the measurement built to test it overturned.** The dominant
  mode's participation ratio:

  | block | effective cells | top 1 | top 10 |
  |---|---|---|---|
  | `A_gg` (the diverging arm) | **1.1 of 1 635 909** | **0.933** | 0.999 |
  | Schur | 16.9 | 0.091 | 0.688 |

  **Ninety-three per cent of the divergent mode's energy sits on a single cell.** So `rho > 1` was
  never evidence of a whole-mesh cause, a per-cell guard would work, and the earlier "the diverging
  cells are geometrically ordinary because they are merely where the dominant eigenvector has
  support, which is why no per-cell guard would help" had the inference backwards: a mode on one cell
  is exactly what a per-cell guard catches. The population of 4599 was the contamination of a tiny
  set spread by 20 sweeps, which is what the dilation test was added to check.

  **The general lesson, and it is the transferable one: a spectral radius is a whole-mesh scalar and
  says nothing about whether its cause is whole-mesh.** Reading locality out of it requires the
  eigenvector, which costs nothing extra once the power iteration is already running.

  **THE RELAXATION LADDER CONFIRMS THE SPECTRAL PICTURE INDEPENDENTLY, and rules relaxation OUT for
  this arm.** `rho(I - w P⁻¹S)` under the Schur block, against what the two-point model
  `max|1 - w·lambda|` over `lambda ∈ [0.693, 1.065]` predicts from the governing eigenvalue and the
  modulus bound:

  | `w` | 1.00 | 0.80 | 0.50 | 0.25 | 0.10 |
  |---|---|---|---|---|---|
  | measured | 0.3070 | 0.4597 | 0.6170 | 0.7863 | 0.9089 |
  | predicted | 0.3070 | 0.4456 | 0.6535 | 0.8267 | 0.9307 |

  Within 6 % at every rung, and **monotone increasing** — every amount of under-relaxation makes this
  arm strictly worse, which is what an all-positive spectrum requires and is the opposite of what a
  diverging arm needs. So Betchen & Straatman's relaxation requirement, which is stated for *their*
  block-Jacobi on an arbitrary grid, is **not** what this configuration wants: it converges undamped
  and damping only slows it. The prediction is a genuine one — the model was written down from the
  `w = 1` measurement alone, before the other four rungs were computed.

  The model also says the optimum is *above* one (`w = 2/(lambda_min + lambda_max) = 1.14`, giving
  `rho ~ 0.21` against 0.31). **Not measured, and not proposed**: a 30 % rate improvement bought by a
  knob calibrated from a two-point spectral estimate on one mesh is exactly the kind of tuning this
  project's own record shows sitting one grid step from a cliff.

  ⚠️ **Cost note for anyone re-running the ladder:** it used to build a fresh `jax.jit` per rung,
  closing `w` in as a constant — so every rung recompiled a program capturing 2.3 GB of constants at
  this mesh size, which dominated the twenty applies it then ran. `w` is now a traced argument and one
  compilation serves the whole ladder.

  **⚠️ THE SIGN TEST MEASURED THE WRONG EIGENVALUE, AND ITS PRINTED VERDICT IS THE OPPOSITE OF THE
  TRUE ONE.** The test was meant to split the two candidate fixes: a positive governing eigenvalue of
  `P⁻¹S` means some relaxation stabilizes the arm, a negative one means none does. It power-iterated
  over `P⁻¹S` and reported `|lambda_max| = 1.0612`, `Rayleigh +1.0597`, "relaxation can stabilize it"
  for `A_gg` — and `1.0649 / +1.0634` for the Schur arm, i.e. **nearly the same answer for an arm that
  diverges at `rho = 2.04` and one that converges at `rho = 0.31`.** Two arms whose behaviour differs
  by that much returning the same number is the tell.

  **A power iteration over `P⁻¹S` returns the largest-MODULUS eigenvalue. The sweep is governed by the
  eigenvalue FURTHEST FROM 1.** Those are different eigenvalues here, and the first is uninformative
  about the second: on this operator most cells have `P` nearly exact, so there is a large cluster of
  eigenvalues near `+1` — that cluster is what both iterations found, which is why both arms returned
  ~1.06.

  **Solved properly, `rho = |1 - lambda|` admits two candidates and the modulus bound kills one:**

  | arm | `rho` | candidates | `\|lambda\|_max` | survivor |
  |---|---|---|---|---|
  | `A_gg` | 2.0397 | `-1.0397` or `+3.0397` | 1.0612 | **`-1.0397`** (`+3.04` exceeds the bound) |
  | Schur | 0.3070 | `+0.6930` or `+1.3070` | 1.0649 | **`+0.6930`** (`+1.307` exceeds the bound) |

  So `A_gg`'s governing eigenvalue is **negative**, and `1 - w·lambda > 1` for every `w > 0`: **no
  positive relaxation stabilizes that arm** — the exact opposite of what the harness printed. Betchen
  & Straatman's own under-relaxation prescription would not have rescued it.

  **The instrument is fixed rather than annotated.** The governing eigenvalue is now read from the
  Rayleigh quotient of the `I - P⁻¹S` iteration already being run for `rho` — whose dominant mode *is*
  the governing one, and on which `I - P⁻¹S` acts as `1 - lambda` — so `lambda` comes out directly and
  free. The `P⁻¹S` iteration is kept, relabelled as the modulus bound it actually provides, because it
  is what discriminates the two candidates; when the two routes disagree, neither is trusted.

  ⚠️ **What made this catchable was a second measurement that had to agree and did not** — the same
  discipline that caught the `local_schur_block` reversal. A verdict from one power iteration, printed
  with a confident English gloss, would have been recorded as fact; it was already written into this
  file once as "relaxation can stabilize it" before the arithmetic was checked.

  None of this affects the six-component result, which needs no relaxation: the Schur block converges
  undamped at `rho = 0.31` with a governing eigenvalue of `+0.69`.

  **BOUNDARY PROXIMITY: the hardest cells ARE boundary cells — 12 of the 12 worst own a boundary
  face, against 23.1 % of the mesh doing so (2026-08-22).** Under the six-component Hessian nothing
  diverges, so the enrichment table over the diverging *set* is vacuous (0 of 0) and says nothing;
  what carries the finding is the residual ranking. Every one of the twelve worst-reconstructed cells
  sits at boundary hop **0**, where 2.8 of 12 would be the chance expectation — and every one is also
  four-faced. This scheme closes the Hessian at a boundary by taking the owner's, which Betchen &
  Straatman note can leave the system under-determined; their remedy — an inverse-distance average of
  the Hessian over interior neighbours not themselves adjacent to a boundary — is **not implemented
  here**, and this is the first evidence on a real mesh that it is the right place to look next.

  ⚠️ **Association, not mechanism.** A four-faced cell at a boundary is unusual in two ways at once
  and this cannot separate them: the boundary closure could be the cause, or low face count could be,
  or the two could simply co-occur because that is what `snappyHexMesh` produces where it cuts the
  geometry. The discriminating arm is to implement the averaged closure and re-measure the same
  ranking; a face-count-matched comparison of interior against boundary tetrahedra would separate
  them without writing any new closure, and is cheaper.

  ⚠️ **And the collapse the earlier regime showed is gone too**: `sigma_min` shrink now has a
  *minimum* of 5.4e-01 across the whole mesh (median 1.000, **zero** cells below 1e-1), against a
  minimum of 7.7e-03 and eight cells below 1e-1 under the unsymmetrized Hessian. The reduction did
  not merely re-rank the arms — it stopped the correction from nearly cancelling the block anywhere.

  **⚠️ IT IS THE SYMMETRY THAT FIXES THE REACTOR MESH, NOT THE LEAST-SQUARES WEIGHTING — measured
  2026-08-23, and it retires the SPD mechanism this file offered a day earlier.** The six-component
  change carried two things at once: solving the symmetric components, and removing the surplus
  equations by a weighting that makes the reduced block SPD. Separated by swapping the reduction for
  an unweighted projection (`contract_symmetric(residual)`, no `A_P` contraction, no row scaling) and
  re-running the same harness on the same 1.6M-cell mesh:

  | reduction | Schur median | p99 | max | Schur diverging | `A_gg` max | `A_gg` diverging |
  |---|---|---|---|---|---|---|
  | weighted (least squares) | 8.055e-08 | 4.320e-05 | 5.184e-03 | **0** | 5.787e+05 | — |
  | unweighted (projection) | 8.139e-08 | 4.467e-05 | 6.137e-03 | **0** | 7.095e+16 | 6680 |

  **Both fix it completely.** The unweighted block carries no definiteness guarantee whatever, so
  "the reduced block is SPD, therefore the elimination term is well-signed" — offered here as the
  mechanism — **is not what is doing the work.** What matters is that the Hessian is symmetric. Say
  that, and stop citing definiteness as the cause.

  **The weighting is kept anyway, and the reason is not the reactor.** It is the source's own
  formulation, it costs one contraction per apply, and it makes the block SPD — a property worth
  having even where it is not load-bearing. But its cost is real and shows up elsewhere: on a
  tetrahedral mesh under the owner boundary closure it squares an already-bad conditioning,
  `cond(S)` 4.184e+04 unweighted against **4.040e+19** weighted. That difference is moot once the
  averaged closure is in force (which fixes tetrahedra outright, `cond(S) ~ 6`), but it is the
  measurement to remember if a mesh ever turns up where the closure alone is not enough.

  ⚠️ **Note what the `A_gg` column also shows:** unweighted, that arm's worst cell is 7.095e+16
  against the weighted arm's 5.787e+05, and 6680 cells diverge. So the two reductions are not
  interchangeable for the arm that fails — they are interchangeable only for the one that works.

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
    Measured:

    | mesh | calibrated weight | inner sweeps |
    |---|---|---|
    | tetrahedral n=2 | 0.10 | 38 |
    | tetrahedral n=3 | 0.20 | 36 |
    | hex 5³, perturb 0.25 | **0.00** | 10 |
    | 2D, perturb 0.3 | **0.00** | 9 |

    Costs `iters` applies per candidate — about one reconstruction for the whole ladder, once, off
    the differentiated path.

    **The shipped default is `0.2`**, which is where the trade sits on the meshes measured and is the
    value to use when calibration is not run. ⚠️ It remains one mesh's evidence: the conditioning
    scales roughly as `cond ~ w^-1.75` over the range tested, so a mesh an order more degenerate would
    be four orders worse at `w = 0.1`. That is what the calibration is for — prefer it to the
    default, and do not lower the default on the strength of that table.

  - **✅ THE WAY TO IMPROVE THE INNER SOLVE IS NOT TO NEST IT — `CoupledBlockSweep` is worth ~12x
    here, far more than the 1.2--1.5x it was worth before this closure existed (2026-08-23).** The
    nested path re-converges the Hessian from zero on **every outer apply**, and this closure is
    precisely what makes that convergence expensive — so the arrangement that keeps the Hessian
    iterate between sweeps gains exactly where the closure costs. At matched accuracy (~1e-10),
    counting face-kernel passes as `outer x (1 + inner)` against the coupled sweep's `3 x sweeps`:

    | mesh | nested | coupled sweep | ratio |
    |---|---|---|---|
    | tetrahedral n=3, weight 0.2 | 1665 passes → 6.2e-11 | **~120 → 2.2e-10** | **~12x** |
    | hex 5³, perturb 0.25 | 165 passes → 1.9e-11 | **51 → 8.9e-13** | **~3x** |

    On the tetrahedral mesh the coupled sweep reaches **5.8e-15** at 192 passes, which the nested path
    does not approach at nine times the work. **So the pairing to recommend with this closure is the
    coupled sweep, not a bigger inner count.**

  - **⚠️ AND THE INNER COUNT THE CALIBRATION ASKS FOR IS LARGELY WASTED ON A HARD MESH.** Holding the
    outer count at 20 and sweeping the inner on the tetrahedral mesh, the reconstruction plateaus by
    **inner 8** (4.06e-06) and 36 buys nothing (3.48e-06) — the *outer* count is what limits the
    answer there, so 28 of the 36 calibrated inner sweeps are spent for a 13 % improvement. On the
    hexahedral mesh the same sweep keeps paying to inner 14. **The composition is the thing to
    calibrate, and calibrating each system separately to the same tolerance does not do it** — this
    file's older "three orders of margin" note is the same observation, and the margin is now measured
    at ~7e-4 attenuation on tetrahedra against ~1e-2 on hexahedra. ⚠️ **A blanket loosening factor is
    therefore NOT safe**: those differ by 14x, and the conservative end would over-resolve one mesh
    while under-resolving the other.

  - **⚠️ `CoupledBlockSweep.calibrated` SATURATES ITS CAP on the tetrahedral mesh, and over-counts by
    ~3x when it does not.** It reports 64 at the default cap, 128 at 128, and asks for **193** at 256
    — while 64 sweeps already reconstruct to **5.8e-15**. That is the conservatism this file already
    records structurally (the rate is measured on the packed `[g, h]` error, whose Hessian half
    dominates while the gradient's falls faster), quantified here on a mesh hard enough for it to
    matter. **A reported count equal to the cap is not a measurement** — raise the cap to find out
    what it wanted, and judge the sweep on the reconstruction rather than on the count.

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

  **⚠️⚠️ THE SCHEME DOES NOT SOLVE ON AN ALL-TETRAHEDRAL MESH UNDER THE DEFAULT BOUNDARY CLOSURE —
  `A_HH` IS NUMERICALLY SINGULAR THERE, AND THIS PREDATES THE SIX-COMPONENT CHANGE (measured
  2026-08-22). ✅ SOLVED 2026-08-23 by `AveragedInteriorHessian` (above), which takes `cond(A_HH)`
  from ~1e17 to ~8 — this entry is the diagnosis that led there, and the "not per-cell / not only the
  boundary" reasoning below is what pointed at the closure rather than at a guard.** `tests/support/meshes.py::
  tetrahedral_grid_3d` builds a conforming Kuhn subdivision of the unit cube (six tetrahedra per
  cube, every cell four-faced, closure ~1e-17, positive volumes, validated by
  `test_the_tetrahedral_fixture_is_a_valid_mesh_of_four_faced_cells`). On it, materialized densely:

  | mesh | formulation | `cond(A_HH)` | `cond(S)` |
  |---|---|---|---|
  | tet, 48 cells, perturb 0.1 | nine components | **4.696e+17** | 7.175e+02 |
  | tet, 48 cells, perturb 0.1 | six components | **2.851e+17** | **4.040e+19** |
  | hex, 3³, perturb 0.25 | nine components | 4.622e+00 | 1.995e+00 |
  | hex, 3³, perturb 0.25 | six components | 2.084e+01 | 1.995e+00 |

  Double precision carries ~1e16, so `A_HH` is **singular in both formulations** — the nine-component
  column is the pre-existing state and was checked by running the diagnostic against the parent
  commit's `gradient.py`, precisely so this could not be mis-attributed to the same day's change. A
  Krylov solve raises (lineax stagnation) and a swept solve returns `inf`, at every perturbation from
  0 to 0.25 and at every size from 48 to 750 cells.

  **What is NOT yet established, and the candidates are not equivalent:**
  - **It is NOT per-cell.** Every cell's diagonal block inverts; the null space is a property of the
    assembled operator. So a per-cell guard is the wrong shape here, unlike the `A_gg` divergence
    above.
  - **It is not only the boundary closure**, though that is the prime suspect: Betchen & Straatman
    introduce their averaged Hessian (Eq. 29) *explicitly* to stop the simple closure leaving the
    system under-determined, and this scheme implements the simple one. But raising the mesh from 75 %
    to 36 % boundary-touching cells does not rescue it, so if the closure is the cause its effect is
    not confined to the cells that touch a boundary.
  - **The six-component `cond(S)` is nine orders worse than the nine-component one** (4.0e+19 against
    7.2e+02) even though `A_HH` is comparably singular in both. The reading that fits: with the full
    tensor the null direction lies where `A_gH` cannot see it — the gradient equation contracts `H`
    against the face-curvature tensor, which is **symmetric**, so an antisymmetric null direction is
    annihilated and never reaches the Schur complement. Removing the antisymmetric components removes
    that shelter. ⚠️ Offered as the reading that fits, **not measured** — the null vector was not
    extracted.
  - **The least-squares weighting squares the conditioning** (`cond(NᵀN) = cond(N)²`), which is
    harmless where `C` is well conditioned (2.4 median on the reactor mesh) and is not where it is
    not. An unweighted projection would avoid the squaring at the cost of the SPD property that the
    reactor measurement above turns on. **Whether the reactor result needs the weighting or only the
    symmetry is UNSEPARATED** — both arrived in one change, and separating them is one more run.

  **Consequence for the test suite:** the fixture now carries a real test —
  `test_the_averaged_closure_makes_a_tetrahedral_hessian_system_solvable` pins both halves, that the
  owner closure leaves `cond(A_HH) > 1e15` there and the averaged one brings it under `1e3`, with a
  hexahedral control so the result is about the cell shape and not about one closure being generally
  better. **"The scheme works on tetrahedra" is true only with `AveragedInteriorHessian`, and then
  only to second order** — the default closure still cannot solve there, which is the trade the entry
  above describes.

  ⚠️ **THE FIXTURE GAP THAT LET THIS SHIP TWICE IS ONLY HALF CLOSED.** Both the original wrong
  default and this reversal were invisible to every synthetic mesh in the test suite, because those
  are perturbed hexahedral grids and every diverging cell here has **four** faces. A tetrahedral
  fixture now exists (`tetrahedral_grid_3d`) and is validated as a mesh — but the scheme does not
  *solve* on it (see the entry above), so it cannot yet carry a reconstruction test. **Until it can,
  a change to this preconditioner is not tested by a green fast gate — it is tested by this one case
  and nothing else.**

  **⚠️ THE RATE IS THE MECHANISM: `rho(I - P⁻¹S)` is 6.2679 under the Schur block and 0.7148 under
  `A_gg`'s.** The sweep is expansive by 6.3 per iteration, so twenty sweeps is `6.27²⁰ ~ 1e+16` --
  which is the observed 1e+16--1e+18 exactly. Half-budget estimates 5.42 and 0.58, so both are settled
  rather than transient.

  **⚠️⚠️ BUT `rho` IS A WHOLE-MESH NUMBER AND DOES NOT ESTABLISH A WHOLE-MESH CAUSE — an earlier
  version of this entry read it as "GLOBAL rather than per-cell" and concluded "NO PER-CELL FIX
  EXISTS". Neither follows.** A spectral radius above one is equally consistent with a mode spread
  over the mesh and with one pinned to a handful of rows, and nothing measured distinguishes them: a
  fixed sweep count propagates a bad row one cell per sweep, so **8 collapsed cells dilated 20 hops is
  a population of the same order as the 4599 that diverge**. The discriminators are the eigenvector's
  participation ratio, whether the diverging set is the dilation of the collapsed one, and the SIGN of
  the dominant eigenvalue of `P⁻¹S`; all three are now in
  `validation/uvreactor_openfoam/schur_block_diagnosis.py` and **none has been run** — the 1.6M-cell
  mesh is gitignored, was not kept, and the run-file did not record `UV_MESH`, so re-running needs
  `of_case/Allmesh` first.

  **⚠️ AND THE HARNESS THAT PRODUCED `rho` WAS DELETED IN THE SAME COMMIT THAT CITED IT.** The
  `sigma_min` and contraction-rate sections ran at 21:50 and the file committed at 22:05 did not
  contain them, so the headline finding was citable but not re-runnable — the precise state the
  measurement rule exists to prevent. Rebuilt 2026-08-22.

  **THE LEADING EXPLANATION IS THAT WE RUN THE ONE RELAXATION THE METHOD'S AUTHORS EXCLUDE.** Betchen
  and Straatman solve this reconstruction by block-Jacobi over the **coupled** `[g, h]` system, with
  the full per-cell block `A_P` (their Eq. 20) — and the gradient corner of `A_P⁻¹` **is** the local
  Schur complement `(A_gg,loc - A_gH,loc A_HH,loc⁻¹ A_Hg,loc)⁻¹`, by the standard block-inverse
  identity. So `local_schur_block=True` is their preconditioner and `A_gg`'s block is not: the latter
  is the `(1,1)` block of `A_P` itself rather than the inverse of the `(1,1)` block of `A_P⁻¹`, and
  the two differ by exactly the elimination term. Of their own iteration they state that **on an
  arbitrary grid a relaxation strictly below one is required for convergence at all**, and they run it
  at 0.8 (their `a = 0.2`; see the convention note in `SweptGradientSolve.solve`), converging in 33
  iterations. **Our outer solver runs undamped.** So the block that diverges is theirs run at a
  relaxation they exclude, and the block that survives is a weaker, more diagonally dominant stand-in
  that happens to contract undamped.

  **A SECOND DEVIATION WAS A CANDIDATE FOR THE SAME FAILURE — SINCE CLOSED (2026-08-22).** They solve
  the **six independent** Hessian components, reducing the over-determined nine-equation system by
  least squares (their Eq. 16--17), which makes their Hessian block `(A_P E)ᵀ(A_P E)` **symmetric
  positive definite by construction**; this scheme solved all `dim²` components unsymmetrized, so its
  block was neither. A non-definite `A_HH` can give the elimination term the wrong sign, and a
  negative eigenvalue of `P⁻¹S` is precisely the case **no** positive relaxation repairs
  (`1 - w·lambda > 1` for every `w > 0`), so the two candidate fixes are not interchangeable and the
  **sign of the dominant eigenvalue of `P⁻¹S`** is what separates them: positive says Betchen's
  relaxation is the fix, negative implicates the Hessian's representation.

  The representation is now Betchen's — see the six-components entry above — so that arm is settled
  and the block is SPD. What that change buys **on this failure** is a measurement, not an inference,
  and it is taken with the same harness on the same mesh.

  **⚠️ BETCHEN'S OWN "UNDER-DETERMINED" REMARK IS ABOUT NEITHER OF THESE — it is a BOUNDARY-closure
  degeneracy** (their Eq. 28--29): taking the boundary Hessian as the owner's leaves the system
  under-determined in some cases, their illustration being a grid one cell thick in a direction, where
  the second derivative across it is arbitrary. Their fix is an inverse-distance-averaged Hessian over
  interior neighbours not themselves adjacent to a boundary, which this scheme does not implement (see
  the boundary-treatment note above). It is **not** a statement that the per-cell Hessian block goes
  rank-deficient on low-face-count cells, and the measurement agrees: `cond C` is 1.17 mesh-wide
  median and, on four-faced cells, 2.45 median with a **maximum of 3.37** — the best-conditioned group
  on the mesh. **Whether the failing cells touch a boundary is unchecked**, and is the one route by
  which their remark could still bear on this.

  A limiter on the correction remains refuted as a *general* fix -- the `sigma_min` collapse it would
  target affects **8 cells of 1 635 909** (median shrink 1.000, only 3 below 1e-2) — but "8 cells
  cannot matter" is exactly the inference the dilation test exists to check, so do not treat it as
  closed either. A Krylov outer solve does not require `rho < 1`, which is the whole reason it is
  untroubled.

  The block that diverges is **13× closer to the truth**, and cell 60355 is **well conditioned** as
  well as accurate. So neither accuracy nor conditioning is the mechanism: **a stationary iteration
  does not want an accurate preconditioner, it wants a diagonally dominant one.** The correction
  *subtracts* from `A_gg`'s block, the diagonal shrinks, `P⁻¹` grows, and on a cell whose off-diagonal
  coupling is already comparable the iteration stops contracting. ⚠️ **This reading was marked
  "superseded by the global rate" and that marking was WRONG — the rate is this mechanism measured,
  not a competitor to it.** Loss of block diagonal dominance is *how* an iteration comes to have a
  rate above one, so the two are one finding at two levels of description; and it is corroborated
  independently by Betchen and Straatman requiring a relaxation below one for their own block-Jacobi
  on an arbitrary grid, which is what one does to an iteration not dominant enough to contract
  undamped. Three earlier mechanisms
  (near-singular `A_HH` from an under-determined Hessian on tetrahedra — refuted, `cond C` is 1.8 on
  the worst cells and 3.37 at worst over every four-faced cell; preconditioner/operator mismatch at
  low inner counts — refuted, invariant across inner 4–20; ill-conditioning generally — refuted by
  cell 60355) were each proposed and each rejected by measurement.

  **A Krylov outer solve is untroubled by the same preconditioner** (max 2.378e-02), spending
  iterations rather than diverging, because it does not require `rho < 1`. ⚠️ That was the *workaround*
  while the Hessian was unsymmetrized; under six components the fixed sweep converges too
  (`rho = 0.31`), so it is no longer a reason to reach for `GmresGradientSolve` here — and the fixed
  sweep remains far cheaper to differentiate.

  **⚠️ HOW THIS SHIPPED, because the process failure matters more than the bug.** It was defaulted on
  the strength of measurements taken **entirely on 8³ synthetic meshes**, which cannot contain the
  cells that break it — while "measure `local_schur_block` at 1.6M cells" was already an open item.
  A default was changed on evidence that structurally could not see the failure mode. The same run
  also exposed a probe-scheduling bug (below) whose justification was measured at 13824 cells, where
  its premise does not hold either. **Two independent faults in one feature, both from measuring on
  meshes too small to discriminate.**

  Better on **every** mesh measured — slightly on clean ones, 57000× on a sliver cell — for **~7 %
  forward time and ~17 % peak memory** (`dim` probes for `A_Hg` plus `dim²` for `A_gH`, a fixed
  prologue against ~220 applies at the shipped 20/10). Adjoint agrees with a finite difference to
  2.1e-09. **DEFAULT ON**: this scheme is chosen precisely for meshes a corrected Green-Gauss cannot
  handle, so its default preconditioner should be the one that survives them — and under the
  six-component Hessian it is also the only one of the two that converges there.
  `local_schur_block=False` recovers the historical `A_gg`-only block, which is now the arm that
  diverges on that mesh and is kept as the control the comparison needs rather than as a
  recommendation.

  - **`A_gH` needs one probe per Hessian unknown** — no Kronecker shortcut, because the gradient
    equation contracts **both** of the Hessian's indices (face curvature, each side's Hessian moment,
    the warp moment). Against the full tensor that was a rank-three block at `dim²` probes; against the
    six independent components it is a plain matrix at `n_sym` probes. ⚠️ The count below and the
    memory figures with it were measured at `dim² = 9`.
  - **Probe sequentially, not under `vmap`.** Same wall clock (248 ms against 247 at 13824 cells), but
    `vmap` holds all of the probes' face intermediates at once and those are the largest arrays in the
    scheme: peak 1.29 GB sequential against 1.39 GB vmapped, on a 1.10 GB baseline — a third of the
    feature's memory cost, for nothing.

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
  matches the nested `20/10` pair's accuracy on every exactness fixture at **60 face passes against
  220** — 3.7x cheaper for the same answer:

  | fixture | nested 20/10 | coupled 20 |
  |---|---|---|
  | 2D perturb 0.2 | 5.20e-15 | 5.15e-15 |
  | 2D perturb 0.3 | 2.42e-13 | 2.82e-13 |
  | 3D perturb 0.25 | 2.98e-15 | 2.43e-15 |

  ⚠️ **It is NOT uniformly better and the record should not be read as saying so** — it is 0.6x on an
  orthogonal mesh (slower, the nested outer solve being nearly trivial there), 1.4x at 30 % skew,
  1.8x at 40 %, and 3--12x with `AveragedNeighbourHessian`. The case for it is that it is better on
  the meshes this scheme exists for and a wash on the ones it does not, at consistently better
  accuracy.

  - **⚠️ `narrow_gradient_sweeps` HAD TO LEARN ABOUT IT FIRST, and missing that would have failed
    SILENTLY.** It rewrote `SweptGradientSolve` nodes only, and the coupled sweep *replaces* both
    solvers rather than sitting beside them — so a defaulted scheme has no `SweptGradientSolve` left
    for it to find, and narrowing would have returned the tree unchanged while a caller believed the
    stencil was bounded. On `pitzDaily` that is the 1.75x probe-narrowing win, gone without a word.
    Exactly the shape already recorded for a Krylov outer solver, which narrowing genuinely cannot
    touch; this one it can, so it now does — carrying `relaxation` across, and pinned by a test that
    the narrowed copy differs in the count and in nothing else.
  - **⚠️ THE SILENT OVERRIDE HAD TO BECOME LOUD, and four live call sites proved why.** With the sweep
    on by default, any `solver` or `hessian_solver` a caller passes never runs. That is not a
    hypothetical trap: `pitzdaily_gradient_ab/run_ab.py` (three sites, including the Betchen arm whose
    cost ratios this file quotes) and `uvreactor_openfoam/schur_block_diagnosis.py` (whose numbers
    were recorded the same day) would all have silently changed meaning. `gradients` now **refuses**
    the pair with a message naming the fix, and every site says `coupled_sweep=None` explicitly.
  - **⚠️ `calibrated` WAS SIZING A PATH THE SCHEME WOULD NOT RUN.** It measured the nested pair and
    returned a scheme that would sweep instead — calibrated in name only. It now sizes whichever path
    it returns: `coupled=True` (the default) measures the sweep, `coupled=False` the nested pair and
    sets `coupled_sweep=None`. A scheme carries one path or the other, never both.

  **⚠️ THE UNDERLYING SHAPE IS STILL WRONG, and the guard is a patch over it.** Three fields of which
  two are inert at any moment is what makes the conflict representable at all; **15 call sites** now
  have to state which path they mean. The structurally correct form is one injected strategy —
  `hessian_solve = CoupledBlockSweep(...)` or `NestedHessianSolve(outer, inner)` — under which the
  conflict cannot be written down and neither the guard nor the `coupled=` flag needs to exist. Not
  built: it re-touches every construction site again, and is a larger API change than the default flip
  it would tidy.

  **`CoupledBlockSweep` — sweep BOTH blocks instead of nesting a solve per apply (2026-08-22; the
  DEFAULT since 2026-08-23, see above).**
  The nested path re-converges the Hessian from zero once per outer sweep, throwing away what the
  previous outer sweep learned. Sweeping alternately keeps it:
  `h ← h + P_H⁻¹(A_Hg g − A_HH h)` then `g ← g + ω P_g⁻¹(b_g − A_gg g + A_gH h)`, Gauss–Seidel (the
  gradient update uses the Hessian just computed, which is what lets one Hessian sweep per gradient
  sweep converge at all). One sweep is ~3 face-kernel passes against the nested `1 + inner` ≈ 11.

  **⚠️ MEASURE IT AGAINST A CALIBRATED NESTED SOLVE, NOT THE SHIPPED DEFAULT.** Against `20/10` it
  looks like 2.0× forward and 3.1× on the tangent — but `20/10` is heavily over-provisioned on the
  meshes that comparison used, so most of that gap is the baseline's slack rather than this sweep's
  merit. Both calibrated at the shared default tolerance (8000 cells, wall clock):

  | mesh | arm | forward | jvp | error vs analytic |
  |---|---|---|---|---|
  | orthogonal | nested 5×1 | **24.5 ms** | 28.3 ms | 3.42e-06 |
  | orthogonal | coupled 6 | 27.9 ms | 28.2 ms | **5.04e-07** |
  | warped p=0.3 | nested 6×4 | 36.6 ms | 46.6 ms | 5.20e-06 |
  | warped p=0.3 | **coupled 7** | **30.7 ms** | **30.5 ms** | **1.09e-06** |

  So honestly: **1.2× forward and 1.5× on the tangent on a skewed mesh, at 5× the accuracy** — and
  slightly *slower* forward on an orthogonal one, where the nested outer solve is nearly trivial and
  there is little to save. In calibrated face-passes: 0.6× on orthogonal, 1.4× at 30 % perturbation,
  1.8× at 40 %. The advantage grows with non-orthogonality, which is the regime this scheme is for.

  **It beats the tolerance it is given, and that is structural.** The rate is measured on the packed
  `[g, h]` error because the blocks converge together, but the Hessian's error dominates that
  estimate while the gradient's falls faster — so the count is conservative for what is returned. A
  comparison at equal *tolerance* is therefore not a comparison at equal *accuracy*.

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

  **Boundary treatment: audited and correct.** Boundary cells reconstruct as exactly as interior ones
  (1.95e-15 vs 2.26e-15 median on a warped mesh). `f = 0` there, `skew` collapses to `d_own`, and the
  unconditionally-computed neighbour side is discarded *structurally* by `scatter` — its index points
  one row past the last cell and is sliced off — rather than by a mask. ⚠️ Betchen's own boundary
  caveat (his Eq. 29, an inverse-distance-averaged Hessian over interior neighbours) is **not**
  implemented; we use the simpler Eq. (28) form. He raises it for a 3D mesh one cell thick, which the
  empty-patch collapse turns into a genuine 2D mesh here, so it likely cannot arise — but that has not
  been proven.
  **Validated inside the solve** (`tests/integration/test_betchen_solve.py`): injected as the
  residual's gradient scheme it nests correctly (the outer Newton `jvp` differentiates through
  Betchen's Schur solve *and* its inner `A_HH` solve), converges in one Newton
  step, and is differentiable on a skewed mesh. **Finding:** in *pure diffusion* the gradient
  enters only as a small non-orthogonal correction, so the solved **field** order is set by the
  operator floor (~2nd) and matches `CorrectedGreenGauss`; Betchen's win shows in the
  **reconstructed gradient / flux** of the solved field (~4× smaller error, higher order on
  skewed, where corrected Gauss caps near 1st — the `ResidualAssembler.gradient(phi)` accessor).
  The scheme matters most where the gradient enters a face value at leading order — advection /
  Rhie–Chow.
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
  - **⚠️ THE `I_dim ⊗ C` BLOCK STRUCTURE IS GONE — it was a property of the UNSYMMETRIZED Hessian and
    the scheme now solves the six independent components (2026-08-22).** It held exactly (measured
    departure `0.000e+00` in 2D and 3D) because `H` entered its own equation only as `H·a` for per-face
    vectors `a`, contracting one index and leaving the other untouched — so the block stored as
    `dim×dim` rather than `dim²×dim²`, 72 B/cell against 648 in 3D. Imposing symmetry couples those two
    indices, so the reduced block is a dense `(n_sym, n_sym)`: **288 B/cell in 3D**, four times the old
    figure, while the unknown drops from 72 to 48. There is no such test any more; the property now
    pinned is that the reduced block is symmetric positive definite
    (`test_the_reduced_hessian_block_is_symmetric_positive_definite`). The unreduced `C` still exists
    and is still `I ⊗ C` — it is what weights the reduction — so the extraction below is unchanged.
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
  - **Cost, and the honest caveat.** On pitzDaily (12225 cells, skewness p99 0.016) the swept inner at the
    shipped 10 sweeps reconstructs **bit-identically** to the exact Krylov inner (`0.000e+00`) at
    **1.04 s against 4.25 s**; `CorrectedGreenGauss` is 0.15 s on the same field, so Betchen is ~7×
    that scheme and would be ~3.6× at a calibrated 4 sweeps. **Single runs on a shared machine — quote
    the structural claim (no nested Krylov, no implicit-diff tangent on the inner solve, cell-local
    preconditioner) rather than the seconds.**
  - **Choosing the sweep count is a MESH property and cannot be inferred from a test mesh.** Departure
    from exactness for a quadratic falls ~2.5 orders per 2 sweeps (3D hex, p=0.25): 4.8e-04 at 2,
    1.3e-06 at 4, 3.0e-09 at 6, 8.6e-12 at 8, machine precision at 10 = the default. A fixed sweep has
    no convergence test and the inner solver's warning is off by necessity, so an under-resolved mesh
    loses quadratic exactness **silently** — calibrate with
    `validation/uvreactor_openfoam/gradient_sweep_calibration.py`, which walks the ladder against a
    converged reference on the real mesh. On pitzDaily it says 4–6 sweeps suffice, i.e. the default is
    conservative there.
  - **A swept OUTER solve reaches the Krylov gradient to 1e-12** (60 sweeps, p=0.3), so an entirely
    Krylov-free, inner-product-free, unrolled-differentiable reconstruction is available —
    `HessianCorrectedGradient(solver=SweptGradientSolve(...))`. Not the default, and **not yet
    cost-compared on a march**.
  - **NOT re-enabled: the distributed path.** `HessianCorrectedGradient` still raises on
    `operator_hook`. The pieces are now much closer — the preconditioner is cell-local and a swept
    outer solve forms no global inner product — but the inner solve's own ghost exchange is not
    threaded, so this is an opportunity, not a claim.
  - **⚠️ NOT DONE: the blocks are rebuilt on every `gradients()` call.** They are geometry-only and
    could be built once per mesh, but `gradients()` builds all its terms inline (as
    `CorrectedGreenGauss` does), so `2·dim` extra applies are paid per reconstruction. Hoisting
    geometry-only reconstruction terms out of the residual is a shared refactor with
    `CorrectedGreenGauss.terms`, not a Betchen-local one.
  - **Measured but NOT acted on: `CellBlockJacobi` helps `CorrectedGreenGauss` too** — ρ 0.1365 → 0.0419
    at p=0.2 and 0.2568 → 0.0682 at p=0.3, a 3–4× better rate, which would make `sweeps=4` far more
    accurate on a skewed mesh or allow fewer sweeps. Its default is **unchanged** (`InverseVolume`);
    changing a shipped default is not something to do without being asked.
  - **REFUTED as the production route: solving the full coupled `[g, H]` system instead of eliminating.**
    With the exact per-cell 12×12 block it converges at ρ ≈ 0.38 for **one** operator apply per sweep,
    against the Schur route's ρ ≈ 0.13 for `k+2` applies — so per unit work the coupled route is ~2.5×
    cheaper (24–27 applies to 1e-10 against 60–108). It is nonetheless **not** the route: the eliminated
    system keeps the gradient as the only primary unknown, which is the point of the elimination and what
    lets it Schur-couple into the flow Newton at gradient size. Recorded because the cost argument is
    real and will be re-derived by anyone who measures it.

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
  - **⚠️⚠️ BUT IT DOES NOT BITE ON A NEARLY-ORTHOGONAL MESH, AND THE REACH THAT MATTERS IS A PROPERTY OF
    THE SMOOTHER FAMILY, NOT OF THE RESIDUAL'S STENCIL ALONE — measured on `pitzDaily` 2026-08-21. Read
    this before acting on the alarm above; the entry as first written over-weighted the obstacle.**
    - **Both gradient arms take the SAME cycle counts at the benchmark's own reach 5.** A capped march
      (6 steps, the case's shipped incomplete-LU bundle) gives corrected Green–Gauss `[2,2,2,3,3,6]` and
      the Hessian-corrected scheme `[2,2]` over the steps it reached — so the long stencil is not
      starving the probe here at all. The far-field mass scales with skewness, and this mesh's median
      skew is `2.2e-09` against the 5–20 % perturbed grids the table above was measured on.
    - **⚠️⚠️ THE SWEEP BELOW IS VOID — THE HARNESS PRINTED A REACH IT WAS NOT PROBING AT. Do not cite the
      "bit-identical cycles at 2, 3 and 5" table.** `run_ab.py`'s `solve_arm` derived a local `reach`
      from its argument, used it in the **banner**, and built the probe from the module-level `REACH`.
      So the three arms printed 2/3/5 and probed the same value each time — which explains
      bit-identical cycles far better than "the reach is inert" did. This is the validation rule *a
      setting the banner prints must be a setting that is in force* violated exactly as written, and it
      is the second time this same claim has failed for a see-the-variable reason (the first being that
      it was swept on an arm that could not feel it). **What survives** are the runs driven by the
      `PITZ_AB_REACH` environment variable, which sets the global the probe really read: the standard
      arm at reach 3 vs 5 (432 vs 439 cycles, 527 vs 711 s) and the Betchen arm (834 vs 511 cycles).
      Those are genuine, and they are what the entries below rest on. Fixed 2026-08-21.
    - **On the SIMPLE-smoothed field-split bundle the reach is inert, and the probe is 3.8× oversized.**
      Same case, standard arm, 6 steps of one Reynolds rung, leading inverse `simple_smoothed_inverse`
      (sweeps 2, pressure_sweeps 2, θ=0.25, no singletons, 5 levels, max_coarse 500, block splitting,
      ω=1.0) with a nodal trailing inverse:

      | probe reach | probes per refresh | plan build | march cycles |
      |---|---|---|---|
      | 2 | **100** | 0.27 s | `[4, 3, 3, 3, 3, 8]` |
      | 3 | **165** | 0.41 s | `[4, 3, 3, 3, 3, 8]` |
      | 5 (the benchmark's) | **380** | 0.86 s | `[4, 3, 3, 3, 3, 8]` |
      | 7 | 670 | 1.60 s | — |

      **Bit-identical cycles, at 3.8× the probes.** ⚠️ And the null is a real one: the knob demonstrably
      has teeth (100 → 670 probes, 0.27 → 1.60 s to build), which is the check that had to pass before
      a flat result could be read as "the reach buys nothing" rather than as "the reach was ignored".
    - **Why the two bundles differ, and it is the mechanism that generalizes:** `pitzDaily`'s reach-5
      calibration was measured against an **incomplete-LU** smoother, where the folded far entries
      become *pivots* — its record has reach 3 + fill 1 failing outright (300 matvecs, true residual
      3.36). A SIMPLE-smoothed hierarchy never eliminates the matrix; it forms an approximate Schur
      complement and applies V-cycles, whose rate is set by the strong near couplings, so a perturbed
      far entry degrades a rate instead of wrecking a factorization. **Do not carry a reach calibration
      across smoother families.**
    - **⚠️⚠️ AND IT IS ARM-SPECIFIC — MEASURED ON THE STANDARD GRADIENT ONLY, AND IT DOES NOT TRANSFER
      TO THE HESSIAN-CORRECTED ONE (2026-08-21).** The sweep above varied the reach with
      `CorrectedGreenGauss` in place, whose Jacobian carries *exactly zero* mass past its own reach — so
      it could not have detected a reach effect on a long-stencil reconstruction, and reading it as a
      property of the preconditioner was wrong. Measured on full `pitzdaily_gradient_ab` marches, the
      Betchen arm at outer/inner swept-5:

      | Betchen arm | cycles | per step | max | wall |
      |---|---|---|---|---|
      | probe reach 3 | 834 over 74 steps | 11.27 | 36 | 1715 s |
      | probe reach 5 | **511 over 72 steps** | **7.10** | **20** | **1604 s** |

      **A third fewer Krylov cycles**, and the cycle ratio against the standard arm falls from **1.93×
      to 1.18×** — i.e. at reach 3 the probe was genuinely under-resolving that arm's stencil and the
      cost difference was being mis-attributed to the reconstruction. ⚠️ **Note the wall clock barely
      moves** (1715 → 1604 s): the longer probe costs 380 residual evaluations per refresh against 165,
      which eats most of the cycle saving. So a longer reach here buys a *comparable measurement*, not a
      faster march — and the two are easy to confuse.
    - **⚠️ What this does NOT license.** It is six steps of the *first* Reynolds rung, on the standard
      gradient arm. This case's own history is that step-initial solves are the cheap ones and the hard
      operators appear mid-step and in retries, so this supports shortening the reach on this bundle
      pending a full march — not a default change on the validated benchmark, whose reach-5 pairing with
      its own smoother is untouched by this. The sweep's **wall-clock column is void** (the first arm
      paid JIT compilation); quote the probe counts and the cycles.
  - **Still open:** whether the shortened reach survives a full three-rung march, and whether the
    genuinely probe-free family (`coupled_continuation`'s block-diagonal SIMPLE + scalar AMGs, which
    takes no `stencil_reach` at all) is competitive on this case.

- **⚠️⚠️ THE KRYLOV OUTER SOLVE COSTS 5× THE SWEPT ONE ON THE PATH A MARCH ACTUALLY PAYS — profiled
  2026-08-21, and it is the single largest number in this subsystem.** *Configuration: `pitzDaily`
  (12225 cells, 2D), the case's own assembly, `eqx.filter_jit`-ed, warm, min of 7 reps, on an
  otherwise-idle machine. `fwd` is one reconstruction; `jvp` is `jax.jvp` of it, which is what every
  Krylov iteration of the flow solve costs.*

  | arm | fwd ms | ×base | jvp ms | ×base | **jvp/fwd** |
  |---|---|---|---|---|---|
  | `CorrectedGreenGauss` swept-4 (baseline) | 0.77 | 1.0× | 0.81 | 1.0× | 1.1× |
  | Betchen outer swept-**1** / inner swept-1 | 1.94 | 2.5× | 2.06 | 2.5× | 1.1× |
  | Betchen outer swept-10 / inner swept-10 | 25.50 | 33.3× | 25.55 | **31.5×** | **1.0×** |
  | Betchen outer swept-20 / inner swept-10 | 50.87 | 66.5× | 48.61 | 59.9× | 1.0× |
  | Betchen outer **GMRES** / inner swept-10 — *the default until 2026-08-21* | 71.11 | **92.9×** | 128.68 | **158.6×** | **1.8×** |

  - **The `jvp/fwd` column is the finding.** A fixed-sweep solve differentiates by unrolling and costs
    **1.0×** to differentiate; the Krylov solve differentiates by the implicit function theorem, which
    solves an *entire second Schur system* per JVP — each with its own 10-sweep inner solve inside every
    one of its iterations — and costs **1.8×** on top of an already 2.8× dearer forward pass. Net: the
    swept outer is **5× cheaper on the JVP path** (158.6× → 31.5×) and reaches the same gradient to
    1e-12. **✅ ACTED ON 2026-08-21: the outer default is now `SweptGradientSolve(sweeps=20)`.** The
    count is 20 rather than the 10 profiled here because 20 is what holds quadratic exactness on the
    meshes the exactness tests use — departure from an exactly-solved reconstruction is `1.7e-15`
    (2D, 20 % perturbed), `7.9e-14` (2D, 30 %) and `1.9e-13` (3D hex, 25 %), against `1.2e-08` /
    `7.4e-08` / `1.5e-07` at outer 10. So the swap is **equal-accuracy**: the Krylov default solved to
    `rtol` 1e-10, and anything below 20 would have quietly made the scheme *less* exact while changing
    the solver. At 20 the comparison is 66.5× forward / **59.9× jvp** against the Krylov 92.9× / 158.6×
    — **2.6× cheaper on the path a march pays, at the same accuracy.** No exactness test needed
    re-pinning; four tests that read the *default* rather than naming their arms did, which is its own
    lesson.
  - **Cost is LINEAR in outer sweeps** (10 → 20 doubles it), and the inner solve dominates each one, so
    the remaining lever is the inner count: 10 buys machine-precision quadratic exactness where 4 gives
    1.3e-06, far below any discretization error a flow solve has. Not yet measured as a pair.
  - **The fixed overhead is 2.5× the whole baseline reconstruction** — geometry, the two per-cell block
    builds, and the right-hand side. ⚠️ Read the `outer swept-1` rows correctly: at `sweeps=1` the swept
    solver returns `P⁻¹b` and **never applies the operator** (the peeled first sweep), so the inner solve
    is never invoked and those two rows are identical by construction, not by measurement. That is what
    makes them a clean isolation of the overhead. The blocks are geometry-only and rebuilt every call;
    hoisting them is the shared refactor with `CorrectedGreenGauss.terms`, and this prices it.
  - **On a whole residual the ratio is much smaller, because a residual is not only gradients:** the full
    coupled residual is **5.0 ms** (1.0 GB peak RSS) under corrected Green–Gauss and **93.2 ms** (2.2 GB)
    under Betchen outer-swept-10 — **18.6×**. Note also that five reconstructions at 0.77 ms is ~85 % of
    the baseline residual's 4.4–5.0 ms, so **the gradient is most of a residual on this case** — an
    intuition that it is a small percentage does not hold here.
  - **⚠️ Memory is a live constraint, not a footnote.** The Betchen residual peaks at 2.2 GB against
    1.0 GB at 12225 cells, and a profiling process was killed running three such arms in sequence. The
    swept path unrolls outer × inner sweeps onto the tape; at 1.6M cells that scales, and it should be
    checked before the scheme is pointed at a large mesh.
  - ⚠️ **Two earlier attempts at this measurement were void and neither announced itself:** one ran two
    copies of the profile concurrently (two background waiters fired on the same event), and the
    march-derived "10–15× per step" that prompted it rested on a *single* step-to-step delta per arm,
    with step 1 contaminated by setup and compilation. Timings on this machine are only meaningful when
    it is running one job.

- **WHAT THE BETCHEN GRADIENT ACTUALLY BUYS AND COSTS ON A REAL CASE — first end-to-end A/B,
  2026-08-21 (`validation/pitzdaily_gradient_ab`).**
  ⚠️ **EVERY `pitzdaily_gradient_ab` FIGURE IN THIS FILE WAS MEASURED UNDER `local_schur_block=False`,
  WHICH IS NO LONGER THE DEFAULT (moved 2026-08-22).** The case pins the two sweep counts but not the
  preconditioner, so it now runs the Schur block instead. On a mild all-hexahedral 2D mesh the two
  blocks agree closely on synthetic fixtures, so these numbers are **expected** to carry — but that is
  an expectation, not a measurement, and nothing here has been re-run. Treat the cost ratios and cycle
  counts below as provisional until one march is repeated, and pin the flag explicitly in any arm that
  is meant to reproduce them.
  Matched marches, both arms at probe reach 5,
  Betchen at outer/inner swept-5, everything else `pitzdaily_openfoam`'s own configuration:

  | | wall | cycles | steps | max cyc | `x_r/h` |
  |---|---|---|---|---|---|
  | `CorrectedGreenGauss` | 711.1 s | 439 | 71 | 15 | 8.069 |
  | `HessianCorrectedGradient` swept 5/5 | 1603.5 s | 511 | 72 | 20 | 8.069 |
  | ratio | **2.25×** | **1.16×** | — | — | same cell |

  - **It does not move the judged quantity.** Both reattach at `x_r/h` 8.069 (reference 7.741). ⚠️ Read
    that carefully: `reattachment_length` returns the **cell-centre** x of the last reversed-flow cell,
    and cells there are ~0.076 h wide — so "identical" means *the same cell*, i.e. agreement to ~1%,
    not agreement to three decimals. The fields do differ (1–2 % in L2, up to 12 % locally in `nu_t`);
    that difference simply does not reach the reattachment length on a mesh ~6° off orthogonal at
    worst. **This is the measurement that says the cheap scheme suffices at this mesh quality.**
  - **⚠️ THE FIRST COST NUMBER WAS 3.25× AND IT WAS WRONG — the arms were at different probe reaches.**
    At reach 3 the Betchen arm needed 1.93× the standard arm's cycles, and that got charged to the
    reconstruction. Matched, the cycle ratio is 1.16× and the wall ratio is the reconstruction. The
    lesson generalizes past this case: **when a scheme change also changes the operator the
    preconditioner sees, a wall-clock ratio is only a scheme comparison once the cycle counts match** —
    which is why the harness prints the cycle ratio next to the wall ratio and says what to do about it.
  - **The probe is a large share of a step, and this prices it:** the standard arm goes 527 s → 711 s
    (+35 %) from reach 3 to reach 5 at essentially unchanged cycles (432 → 439), i.e. ~184 s of pure
    probe cost for 380 residual evaluations per refresh instead of 165.
  - ⚠️ **One run per arm**, on a machine whose per-application noise floor is ~15 %. Step and cycle
    counts are contention-immune and are what to quote; the seconds are single samples.
  - **⚠️ AND THE REASON THIS MATTERS BEYOND BETCHEN: the reconstruction is 12–20 % of a march step even
    for the CHEAP scheme.** With the reconstruction's own jvp cost measured at 0.84 ms
    (`CorrectedGreenGauss`) against 9.30 ms (Betchen swept 5/5) — a factor `k` = 11.11 — a step ratio
    `S` pins the share directly, since `k·g + (1−g) = S` gives `g = (S−1)/(k−1)`:

    | basis | `S` | gradient share, corrected Gauss | under Betchen 5/5 |
    |---|---|---|---|
    | matched, both at probe reach 5 | 2.25× | **12.4 %** | 61 % |
    | standard at reach 3 (cheaper probe) | 3.04× | **20.2 %** | 74 % |

    The share moves with the *rest* of the step (a dearer probe dilutes it), which is why it is quoted
    as a range with its basis rather than as one number. Either way, an eighth to a fifth of every step
    spent reconstructing a term that enters the residual as a **correction** is the standing argument
    for attacking reconstruction cost directly — the geometry-only terms rebuilt per call
    (`CorrectedGreenGauss.terms`, and Betchen's two per-cell blocks, priced at 2.5× the whole baseline
    reconstruction) being the first place to look, since nothing in the current design lets them be
    cached across calls.

- **⚠️⚠️ THE SWEEP COUNT IS THE COST LEVER, AND THE FIXED DEFAULT OF 4 IS WRONG BY UP TO FIVE ORDERS IN
  BOTH DIRECTIONS — measured 2026-08-21.** Two independent investigations converged on this after both
  ruled out the thing that looked more promising.
  - **The contraction rate varies ~50× across meshes.** `ρ(I − P⁻¹A_g)` is **0.0053** on `pitzDaily`,
    against **0.14** at 20 % grid perturbation and **0.26** at 30 %. A single static `sweeps=4` is
    therefore *five orders over-resolved* on the first and under-resolved on the last. Cold relative
    error on pitzDaily at `k` = 1/2/3/4: `4.0e-3 / 1.4e-5 / 5.2e-8 / 1.6e-10`.
  - **On pitzDaily `sweeps=2` costs ONE operator apply** (the first sweep's apply is peeled) and leaves
    `1.4e-05` relative gradient error — three orders below the 1–2 % field difference between the two
    *schemes*. Corroborated independently: `schemes.md`'s own recorded reach-3 Jacobian error of
    1.99e-07 at `sweeps=4` is `ρ³` to one digit.
  - **It compounds, because the sweep count sets the residual's Jacobian reach** — but ⚠️ **NOT by
    `sweeps + 1` for every column, and a first version of this entry said `sweeps=2` collapses the reach
    from 5 to 3, which is WRONG.** Measured directly (coloured probe against the true jvp, per column
    field, at `hybrid_initialize` on pitzDaily):

    | sweeps | reach 3 (165 probes) | reach 4 (265) | reach 5 (380) |
    |---|---|---|---|
    | **4 (shipped)** | 2.45e-07 (**p column 4.0e-07**) | 4.66e-10 | **9.5e-16** ← needs reach 5 |
    | **2** | 1.39e-09 (p column 2.1e-16) | **2.6e-16** ← needs reach 4 | 2.6e-16 |
    | **1** | **2.6e-16** ← needs reach 3 | 2.6e-16 | 2.6e-16 |

    The **pressure** column does track `sweeps + 1` exactly (float64-exact at reach 5/3/2 for sweeps
    4/2/1) — it is the gradient-carried column. The **`u`/`v`** columns carry a further ring from the
    eddy viscosity's strain-rate dependence, which is *not* the gradient's, so they reach `sweeps + 2`.
    So `sweeps=2` needs reach **4**, worth ~14 % of the march rather than the ~35 % first claimed.
    (This run also independently reproduces this file's recorded 1.99e-07 at reach 3 — measured
    2.45e-07 — and its attribution to the pressure column.) At reach 3, `sweeps=2` leaves 1.4e-09 where
    `sweeps=4` leaves 2.45e-07, and 2e-07 is the value recorded as *breaking* the incomplete-LU bundle
    — so reach 3 may yet be safe at `sweeps=2`, but that needs a march and has not been run.
  - **The fix is to stop guessing it: estimate `ρ` once at case-build time** (a few power iterations on
    the geometry-only iteration matrix, which is constant) and set `sweeps = ceil(log(tol)/log(ρ))`.
    That gives the adaptivity a runtime convergence test would, at zero per-call cost, with no
    `lax.while_loop` (which JAX cannot reverse-differentiate) and no implicit-diff tangent — and it
    subsumes `validation/uvreactor_openfoam/gradient_sweep_calibration.py`, which answers the same
    question by hand. ⚠️ **The caution that stood here — that a global `ρ` would size the count for the
    worst cell and so over-resolve the bulk — is REFUTED, and the error runs the OTHER way; see the
    measured L2-versus-worst-cell entry below.**
  - ⚠️ **What this does NOT license:** lowering the shipped default without per-mesh calibration. The
    same count that is wasteful on pitzDaily is *insufficient* at 30 % skew, and a fixed sweep fails
    **silently** — there is no residual test to trip.

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
  - ⚠️ **`tools/sibling_builders.py` COULD NOT SEE THIS PAIR, and that was fixed in the same change.**
    It recognized only `build`/`create`/`make`/`from_*` as factory methods, and credited construction
    by capitalized-name convention — so a `@classmethod` returning `cls(...)` looked like it
    constructed *nothing* and dropped out of the report entirely. Not a quiet pair: **no pair at all**,
    which reads exactly like a clean tree. It now knows `calibrated` and credits `cls(...)` to its
    owning class, pinned by `test_it_reaches_classmethod_factories_that_return_cls`. With that, the
    pair reports with `schur` as the only difference — a genuine property of the scheme with two
    modes. A signature-parity test derives the shared surface from `dataclasses.fields(
    SweepCalibration)`, so adding a setting and wiring it into one factory fails there.
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

- **Per-case calibrated counts, measured on the meshes in this repository (2026-08-21):**

  | case mesh | cells / dim | `ρ` | `k` at 1e-4 | `k` at 1e-10 | ships |
  |---|---|---|---|---|---|
  | `pitzdaily_openfoam/of_case` | 12225 / 2D | 5.07e-03 | **2** | 5 | 4 |
  | `bfs3d` (read from `bfs3d_species/of_case`) | 23040 / 3D | **5.06e-15** | **1** | **1** | 4 |

  ⚠️ `bfs3d_openfoam/of_case/constant/polyMesh` is generated by running OpenFOAM and is **absent from a
  fresh checkout**, which is why the row above names the species case's copy — the same geometry (23040
  cells, 66368 interior faces, matching this file's own reach table), and the one actually in the
  repository. A harness that reads the flagship path will skip rather than measure.

  `bfs3d`'s mesh is a graded but orthogonal block, so the skewness correction is essentially absent and
  the first sweep is already at machine precision — the shipped `4` spends three operator applies per
  reconstruction, on every residual evaluation and every Jacobian--vector product, to no effect. It
  corroborates the reach table at the top of this file (`0 of 66368` faces above the skewness
  threshold). **Neither default was changed**: the sweep count sets the residual's Jacobian reach,
  which each case's probing reach is matched to, so moving it is a change to the discretization that
  the case's reattachment result would have to be re-validated against. Both numbers are recorded in
  their case READMEs beside the mesh.

- **PROPOSED, NOT BUILT — `GradientScheme.bind(mesh, geometry) -> BoundGradient`.**
  - **The seam.** `gradients()` takes `mesh` and `geometry` on every call and rebuilds every
    geometry-only term inside them. `bind` would split the *choice* (the injected, mesh-free
    `GradientScheme`) from the *derived product* (arrays, no mesh argument), exactly as `mesh.geometry()`
    already does and as `MomentumContinuity.build` already does for `interp_factor`/`normal_distance` —
    the gradient scheme is the one thing that opted out. `gradients()` then loses both parameters, which
    is the real payoff: today you can hand it a different mesh than the assembler holds and get a
    silently wrong answer. Three call sites change. ⚠️ **Do not sell it as performance** — XLA already
    CSEs the shared geometry, so it is worth ~2 % of a residual and ~0.4 % of a jvp; the Betchen prologue
    is the only piece large enough to matter (~4.7 % of a march step).
  - **⚠️ Binding OUTSIDE the differentiated region silently kills mesh-shape gradients** —
    `‖d(objective)/d(node_coords)‖ = 0.0`, no error. Binding inside is bit-identical to today. Not a new
    hazard (the same rule already governs `mesh.geometry()`), but now measured.
  - **Calibration was expected to need this seam and did not** — it shipped as a classmethod factory
    (above). `bind` would still be a natural home for it if it is ever built, but it is no longer a
    reason to build it.
  - **Betchen's own calibrated counts, measured:** outer `ρ` = 0.159 (pitzDaily) to 0.238 (30 %
    perturbed) — **barely moving with mesh quality**, which corroborates that the outer Schur system's
    difficulty is intra-cell gradient–Hessian coupling rather than skewness, and means the outer count is
    nearly mesh-independent. Inner `ρ` = 0.0074 to 0.091. At `tol=1e-4` that is outer **6**, inner **2**.
    ⚠️ Read against the shipped 20/10, which target ~1e-10 (the accuracy the Krylov default delivered),
    not 1e-4 — the two are answering different questions, and the shipped pair is not "20 against 6".
  - **The inner truncation sets a floor no number of outer sweeps removes** (pitzDaily: inner 1 → 4.7e-08,
    2 → 7.2e-11, 3 → 5.1e-13), but it is attenuated into the gradient by 1e-5 to 1e-3, so calibrating both
    at the same tolerance carries three orders of margin. Measured on two meshes only — check the
    composition per mesh rather than assuming it.
  - ⚠️ **Distributed:** the estimator norms the whole local vector, which double-counts ghost rows.
    Calibrate on the global mesh **before** partitioning, or supply an owned-only reduction.

- **✅ NARROWING THE PROBE BEATS LENGTHENING IT — measured 2026-08-21, and it is the largest single
  saving found on this case.** The long reach a long-stencil reconstruction seems to demand is paying
  to *tolerate folding*, not to capture coupling the preconditioner needs. Cap the gradient's sweeps
  **for the probe copy only** (`CoupledJacobianProbe.build(gradient_sweeps=…)` /
  `narrow_gradient_sweeps`) and the residual's stencil genuinely shortens, so the colouring is
  collision-free and the recovered matrix is **exact for the narrowed residual** instead of corrupted
  for the true one. Full `pitzdaily_gradient_ab` marches, Betchen arm at outer/inner swept-5:

  | probe | probes | standard | betchen | betchen/standard |
  |---|---|---|---|---|
  | reach 5, full sweeps | 380 | 711 s / 439 cyc | 1604 s / 511 cyc | 2.25× |
  | reach 3, full sweeps | 165 | 527 s / 432 cyc | 1715 s / 834 cyc | 3.25× |
  | **reach 3, `gradient_sweeps=1`** | **165** | **473 s / 421 cyc** | **918 s / 451 cyc** | **1.94×** |

  All six marches reach `x_r/h` 8.069.

  - **1.75× faster than the reach-5 arm, at 43 % of its probe cost, with FEWER cycles** (451 against
    511) and the identical root. The expectation was that narrowing would merely *recover* reach-5
    quality; it beats it. So the far entries were not helping the aggregation — a sparser, cleanly
    probed operator is both a better preconditioner and a cheaper one to build and apply.
  - **It is NOT a Betchen-specific fix — the standard scheme gains 1.50×** (711 → 473 s, cycles 439 →
    421). The benchmark next door ships reach 5 with full sweeps and is leaving that on the table.
    ⚠️ But do not assume it transfers: this case runs a SIMPLE-smoothed field split, and the benchmark
    runs an **incomplete-LU** smoother, which is the family folding hurts most (folded entries become
    pivots) *and* the family most exposed to the narrowed matrix being a cruder approximation. Both
    effects point opposite ways; it has to be measured there, not inferred.
  - **The matched cost of the Betchen scheme is 1.94×, and this is the trustworthy version of that
    number** — its cycle ratio is **1.07×**, against 1.16× at reach 5 and 1.93× at reach 3 unnarrowed.
    The figure has now read 3.25× / 2.25× / 1.94× across the session, and every correction came from
    removing a probe artifact rather than from anything about the reconstruction.
  - **The middle row is the cleanest demonstration of why the stencil matters.** Same probe, same mesh:
    the standard arm is untouched by the short reach (432 cycles, essentially its reach-5 value) while
    the Betchen arm degrades to 834. That is also why this went unnoticed — with the corrected gradient
    the folded mass on a mesh this mild is negligible, so the probe reach looks like a free parameter.
  - **The Krylov matvec keeps the exact jvp of the full residual**, so the converged state and its
    adjoint are untouched — only the preconditioner's materialize sees the narrowed copy. That is what
    makes this a free lunch rather than a discretization change.
  - **It is only reachable because the Betchen outer solver is now a fixed sweep.** A Krylov outer
    cannot be narrowed — `narrow_gradient_sweeps` rewrites `SweptGradientSolve` nodes and would return
    the tree unchanged, silently. At `sweeps=1` the swept solver returns `P⁻¹b` without ever applying
    its operator, so *both* Betchen systems collapse to a one-ring stencil.
  - **The floor is reach 3, not 1**: the coupled `(u,p)` Jacobian is intrinsically distance-2 (Rhie–Chow
    damping couples pressure to the neighbour-of-neighbour ring), and no narrowing removes that.
  - ⚠️ **One run per configuration.** Cycle and step counts are contention-immune and carry the verdict;
    the seconds are single samples. **Not yet measured:** the standard arm with the same narrowing (its
    reach-3 full-sweep probe already works, folding being negligible at this mesh's skew, so the gain
    there should be smaller); and whether this holds on a genuinely skewed mesh, where the folded mass
    is larger and the effect should be *stronger*.

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
    is ever rearranged). This is the same peel `_VCycleOps.smooth_zero` and `Ilu0.sweep_from_zero`
    already carry; the gradient solve was the one place in the tree still missing it.
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
    corrupted one. Consumed through `CoupledJacobianProbe(gradient_sweeps=…)` / the coupled builders'
    `probe_gradient_sweeps=`; see `.claude/rules/turbulence.md` and `.claude/rules/solve-direct-preconditioners.md`. **Default
    `None` everywhere is byte-identical.**
    ⚠️⚠️ **IT IS LATENT ON `bfs3d` AND LIVE ON pitzDaily — an earlier version of this entry said "latent
    on every case shipped today" and that is FALSE (corrected 2026-08-16).** The two shipped cases run
    *identical schemes* and differ only in the mesh, and only `bfs3d` is skew-free:

    | mesh | `|skew|/d` median | max | interior faces > 1e-6 |
    |---|---|---|---|
    | `bfs3d` | 7.0e-15 | 1.9e-12 | **0 of 66368** |
    | **pitzDaily** | 2.2e-09 | **7.5e-02** | **11567 of 24170** |

    pitzDaily's distribution is bimodal — most of it is a structured block at round-off, and the slanted
    lower wall and the contraction carry the tail, whose **maximum skew exceeds a 5 %-perturbed synthetic
    grid's**. Measured consequence (a `pitzDaily` session, at the `hybrid_initialize` seed):
    `jacobian_relative_error` against the true matrix-free jvp is **1.99e-07 at reach 3** and only reaches
    the float64 floor at **reach 5** (1.48e-15), where `bfs3d` is already at 2.34e-16 at reach 3 — and
    pitzDaily at `sweeps=1` floors at reach 3 exactly as `bfs3d` does, which is what ties the difference to
    the sweeps rather than to anything else. **So running pitzDaily at the shipped `stencil_reach=3` costs
    a real, measurable error today; the trap is not hypothetical.** The shortfall is carried almost entirely
    by the **pressure column**, which enters the residual only through gradients and so inherits the
    sweep-extended stencil undiluted — read `jacobian_relative_error` per (row field, column field), since
    one random vector under a global norm cannot see it.
    ⚠️ `validation/pitzdaily_openfoam/compare.py`'s own docstring still says this mesh is "only mildly
    non-orthogonal ... reaches the converged corrected-gradient to machine precision in the default few
    sweeps". Re-adjudicate that before quoting it.
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
