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

  **⚠️ THE OUTER PRECONDITIONER IS BUILT FROM THE WRONG BLOCK BY DEFAULT, and `local_schur_block`
  is the fix (2026-08-22).** The outer operator is the Schur complement
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

  Better on **every** mesh measured — slightly on clean ones, 57000× on a sliver cell — for **~7 %
  forward time and ~17 % peak memory** (`dim` probes for `A_Hg` plus `dim²` for `A_gH`, a fixed
  prologue against ~220 applies at the shipped 20/10). Adjoint agrees with a finite difference to
  2.1e-09. **DEFAULT ON since 2026-08-22**: this scheme is chosen precisely for meshes a
  corrected Green-Gauss cannot handle, so its default preconditioner should be the one that survives
  them; `local_schur_block=False` recovers the historical `A_gg`-only block.

  - **`A_gH` genuinely needs `dim²` probes** — no Kronecker shortcut. The gradient equation contracts
    **both** of the Hessian's indices (face curvature, each side's Hessian moment, the warp moment),
    so its block is a full rank-three tensor. `A_HH`'s is `I ⊗ C` and needs one.
  - **Probe sequentially, not under `vmap`.** Same wall clock (248 ms against 247 at 13824 cells), but
    `vmap` holds all `dim²` probes' face intermediates at once and those are the largest arrays in the
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

  **`CoupledBlockSweep` — sweep BOTH blocks instead of nesting a solve per apply (2026-08-22, opt-in).**
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
  - **The per-cell `A_HH` block is EXACTLY `I_dim ⊗ C` with `C` only `(dim, dim)`** — measured departure
    `0.000e+00` in 2D and 3D. `H` enters its own equation only as `H·a` for per-face vectors `a`, which
    contracts `H`'s second index and leaves the first untouched. So the block is stored and inverted as
    `dim×dim`, not `dim²×dim²`: **72 B/cell rather than 648 in 3D** (115 MB vs ~1 GB at 1.6M cells),
    which is the difference between usable and not at that scale. Pinned by
    `test_the_hessian_block_is_a_kronecker_product_so_it_is_stored_dim_by_dim`.
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
  2026-08-21 (`validation/pitzdaily_gradient_ab`).** Matched marches, both arms at probe reach 5,
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
