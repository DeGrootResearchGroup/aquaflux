---
paths:
  - "aquaflux/discretization/**"
---

# Rules — `aquaflux/discretization/` (the Layer-0 residual substrate)

> **Provenance boundary (binding).** This file cites the C++/Fortran precursors and the
> design notes to inform *your* understanding — that is its job, and why it loads into your
> context. Per the root `CLAUDE.md` **Comment Convention**, none of that provenance may
> reach the shipped surface (`.py` comments/docstrings, `docs/`): cite the *math*, never the
> reference code, the `.claude/` rules, the design notes, or the author's own papers.

The heart of the solver: assemble the discrete cell residual `R(state, params)` and
let AD produce the Jacobian/adjoint. Governed by the root `CLAUDE.md` Engineering
Principles.

## Responsibility
- The **gather → compute → scatter** engine: each `FaceFluxOperator` gathers the owner/neighbour
  fields it needs, computes its owner-outward face flux, and the assembler scatters the summed flux
  to owner (and, with sign flip, neighbour); boundary faces scatter to owner only. Neither gather
  nor scatter is open-coded — both compose `mesh.face_cells`
  (`aquaflux.mesh.FaceCellConnectivity`, the substrate-wide operator; see the mesh rule): operators
  gather by **direct indexing** with `face_cells.owner` / `face_cells.safe_neighbour`, and
  `ResidualAssembler._scatter` delegates to `face_cells.scatter_conservative`. So this module owns
  the *physics* (the flux operators), not the `segment_sum` mechanics.
  Reference: `reference-code-findings.md` §A.3 (C++ `FaceFluxAccumulator`).
- **No monolithic gathered-state bundle.** Operators are handed a lean shared
  `FaceContext` (`face_flux.py`) — `{face_cells, geometry, boundary_values, gradient, properties}`,
  only the cross-operator or expensive-to-form-once inputs (the reconstructed gradient is a solve, so
  it lives here, formed once; `properties` is the evaluated `{name: (n_cells,) array}` property map) —
  and **each operator gathers its own owner/neighbour fields** from it. `DiffusionFlux(coefficient=…)`
  names the property it reads (`context.properties[coefficient]`, default `"diffusivity"`;
  `"viscosity"` for momentum), so adding coefficients is a `PropertyModel` entry, never a context
  field — see `.claude/rules/properties.md`. `ResidualAssembler.build` takes a `PropertyModel` (not a
  raw `gamma` array).
  A diffusion-only solve therefore never forms an advection limiter, and each operator is
  self-describing about its inputs (what a declarative/DSL assembler consumes). The old fixed
  `FaceState` union-of-all-operators bundle + free `gather_face_state` are **deleted** — do not
  reintroduce a god-bundle that every operator must agree on.
- The per-operator closures for Milestone 0: **diffusion** (the DeGroot–Straatman
  non-orthogonal-corrected flux, `reference-code-findings.md` §D.1 / §B.6) and the
  **transient** term (BDF1 at step 1, BDF2 after).
- The `VolumeSource` seam (zero for pure diffusion, but wired) — this is where turbulence
  production/dissipation and aquakin reaction sources attach. **BUILT** (`source.py`): the
  `VolumeSource` ABC returns the *volume-integrated* source per cell (`∫_cell S dV`, production
  positive — it bakes in its own volume quadrature, as `DiffusionFlux` bakes in face `area`); the
  assembler subtracts each from the balance. The full nonlinear source is written into the residual
  (no Patankar `S_C`/`S_P` split — AD linearizes it, the limiter precedent).

## Status — BUILT (Milestone-0 Stage A)
- **`face_flux.py` — BUILT.** The face-flux contract, shared by every operator (so `diffusion.py`
  and `advection.py` depend on it, not on each other): `FaceFluxOperator` (the `face_flux(field,
  context)` strategy interface) + `FaceContext` (the shared per-face inputs; see Responsibility).
  `gamma` rides on the context as a per-cell **property** supplied by the assembler
  (consumed by the diffusion flux and the flux-type boundary closures) — this is the interim home
  for the eventual properties model, so keep it single-sourced on the assembler, not baked into
  `DiffusionFlux` as operator config.
- **`diffusion.py` — BUILT.** `DiffusionFlux` (a `FaceFluxOperator` that gathers phi/grad/gamma/x
  from the context). Implements the flux-*continuous* DeGroot–Straatman normal
  derivative: one-sided extrapolation from each cell centroid to the integration point, the
  common face value eliminated via `Gamma_P dphi/dn = Gamma_N dphi/dn`, giving
  `[(phi_N − phi_P) + corr_N − corr_P] / denom`, `denom = (D_P·n) − (Gamma_P/Gamma_N)(D_N·n)`.
  The `corr = grad·tangential(D)` terms and the `Gamma`-jump `denom` are written **into the
  residual**; AD linearizes them. **Note the construction differs from the over-relaxed
  area-vector split** (OpenFOAM/Jasak): it decomposes the *centroid-to-face* vector, uses
  each side's own cell gradient (no gradient interpolation), folds skewness into the same
  `corr` term, and handles coefficient jumps natively (conjugate-ready for `CellZones`).
  Verified (`test_diffusion.py`): orthogonal interior/boundary flux vs closed form, the
  correction is the difference `corr_N − corr_P` (cancels for equal gradients), Laplacian
  recovered at 2nd order, differentiable.
- **`residual.py` — BUILT.** `ResidualAssembler` (`equinox.Module`, built via `.build()` from an
  injected `BoundaryConditions({name: closure})` collection, which it binds to the mesh
  (`boundary.resolve(mesh.face_patches)`, off the jit path) and stores as a single `boundary` field).
  It reconstructs cell gradients once (injected `GradientScheme`, optional — `None` on
  orthogonal grids where the correction vanishes), evaluates the per-patch boundary closures,
  gathers owner/neighbour state, sums the injected `FaceFluxOperator`s, `segment_sum`-scatters
  (owner `+`, interior neighbour `−`), and adds the transient term. `R = accumulation −
  transport`. Verified: stub-operator scatter is conservative and correctly signed.
- **`transient.py` — BUILT.** `TransientTerm`: BDF1 at step 1 (static `first_step`), BDF2
  after; carries no physical coefficient. Verified against the closed BDF formulae.
- **`source.py` — BUILT.** `VolumeSource` (ABC, `source(field, context) -> (n_cells,)`): a
  volumetric term produced/consumed *in* the cell rather than across faces (reaction, turbulence
  production/dissipation). Returns the volume-integrated source (production positive; bakes in its
  own volume). Reads cell-oriented fields from the shared `FaceContext` (volume, gradient,
  properties) and holds any frozen coupling field as constructor state, like `AdvectionFlux.mass_flux`.
  `ResidualAssembler.build(..., source_operators=())` subtracts each; empty ⇒ unchanged. Verified
  (`test_source.py`): correct sign, summation, additive composition with the flux, volume
  integration, and differentiability in the field and the source's own coefficient.
- **`advection.py` — BUILT (upwind + limited 2nd order).** `AdvectionScheme` (interface) →
  `FirstOrderUpwind`, `LimitedUpwind`; `AdvectionFlux(mass_flux, scheme)` returns the
  owner-outward flux `mdot_f phi_f`. **The mass flux is always injected — the operator reads
  `mdot_f`, never builds it.** In the coupled flow it is the Rhie–Chow `mdot` (`flow/momentum.py`
  computes it via `interior_mass_flux` and feeds the *same* array to both the momentum convection
  and continuity — the consistency requirement, mirroring the Fortran `calcmdot` → `convectImpl`
  for U/V/W); a scalar transported by a solved flow must likewise reuse that flow's `mdot`, not
  rebuild it (rebuilding `(u·n)A` from cell velocities is non-conservative and violates discrete
  continuity). Because `MomentumContinuity._mass_flux` is private, exposing the converged `mdot`
  is the seam for the future scalar-transport coupling. `mdot = (u·n)A` for a *prescribed
  divergence-free* velocity is a **verification-only** helper and now lives in the test support
  (`tests/support/fields.py::face_mass_flux`), **not** the library — its divergence-free
  precondition makes it unsafe as a production operator (only uniform / stream-function fields
  qualify). `FirstOrderUpwind` is **linear** (affine residual → one Newton step) and monotone;
  verified against the 1-D advection–diffusion exponential
  (`tests/integration/test_advection_diffusion.py`). `LimitedUpwind` reconstructs
  `phi_f = phi_C + psi_C ∇φ_C·(x_f − x_C)` from the upwind cell `C`.
- **Slope limiter — BUILT, but lives in `schemes/` (`aquaflux/schemes/limiter.py`), not here.**
  `Limiter` (interface) → `VenkatakrishnanLimiter(k)` is physics-free reconstruction numerics
  (a **per-cell** slope limiter `psi ∈ [0,1]`, smooth Venkatakrishnan 1993, `eps² = vol K³`,
  matching `coeff.F90`), so it sits beside the gradient/interpolation schemes — keeping the
  `discretization → schemes` dependency one-way (a `schemes/` scheme could want limiting; it must
  not import *up* into `discretization`). The limiter is **held by `LimitedUpwind`**
  (`LimitedUpwind(limiter=…)`, in `advection.py`) and
  evaluated only when that scheme runs — a diffusion-only or first-order-advection solve never
  forms it (`psi` is not a shared/gathered field; `limiter=None` gives `psi = 1`, unlimited 2nd
  order; `psi = 0` = first order). Verified physics-free: `psi → 1` on smooth fields, `< 1` at a
  jump, in `[0,1]`, differentiable. **The limiter is the first genuinely nonlinear term** (stencil
  min/max + rational function of `phi`), so the residual now needs the IFT solve.
- **The AD-linearized-limiter result (`tests/integration/test_limited_advection.py`).** The
  reference **lags** the limiter (freezes `psi`, adds the limited term as an explicit RHS,
  `coeff.F90` line 326) and converges only *linearly*. Writing `psi(phi)` into the residual and
  letting AD linearize it puts the limiter in the Jacobian and recovers **quadratic** Newton
  convergence (measured: ~3 steps vs the lagged ~5), while staying differentiable (IFT) and
  giving 2nd-order accuracy — the "after" to the reference's "before". Boundedness: the smooth
  limiter *damps* over/undershoot (~halves it) rather than strictly eliminating it — the
  smoothness is what makes it AD-linearizable.
- **Sign convention (binding, matches the C++ `FaceFluxAccumulator`).** Every `FaceFluxOperator`
  returns the **owner-outward flux of the conserved quantity**; the residual is the finite-volume
  balance `R = accumulation + Σ scatter(outward flux)` (owner `+`, neighbour `−`). So advection
  returns `+mdot_f phi_f` and diffusion returns `−Γ(∇φ·n)A` (down-gradient, Fourier). `R` is
  invariant to this choice vs the earlier `−scatter` form, but the uniform outward-flux
  convention is what keeps multiple operators composable.
- **Gate C — PASSED (skewed mesh).** Injecting `CorrectedGreenGauss` into the residual folds
  the non-orthogonal correction into `R(φ)`; AD puts it in the Jacobian, so Newton is one step
  and linear-exact on a 25%-skewed grid (`tests/integration/test_skewed_diffusion.py`,
  `.claude/rules/solve.md`). The residual is **affine in φ** with the gradient scheme injected
  (the gradient solve and the correction are both linear in φ), which is *why* one step is
  exact.
- **Gradient-boundary circularity (documented, still open for non-Dirichlet skewed boundaries):**
  a non-Dirichlet boundary value depends on the owner gradient (`corr`), and the gradient
  reconstruction depends on boundary values — circular off-orthogonal. Resolved for now by
  feeding the gradient scheme a leading-order boundary value (its `corr` dropped, i.e. gradient
  = 0) while the *flux* uses the full boundary value at the reconstructed gradient. **Exact when
  boundary values are gradient-independent** — orthogonal grids (Gate A/B) and any all-Dirichlet
  problem (Gate C uses a `DirichletField` linear manufactured solution to stay exact). The
  fully-implicit boundary-gradient fold-in (needed for `ZeroGradient`/`Convective`/`Neumann`
  boundaries at 2nd order on *skewed* grids) couples the gradient scheme to the boundary
  closures and is the scoped follow-up — do not entangle scheme↔BC casually when it lands.

## Binding decisions
- **No hand-derived linearization. Ever.** The reference codes carry `coeff0`/`coeff1`
  (C++) and hand-assembled block coefficients (Fortran); **AD deletes all of it**
  (`reference-code-findings.md` §C.2, §D.3). Write the **full physical flux** as one
  residual term — including the non-orthogonal correction **in the residual**, not
  deferred to an explicit RHS as both references must do. AD then puts the correction
  *in the matrix*, giving a more accurate operator than either reference. This is a
  Milestone-0 deliverable, not incidental (`milestone-0-spec.md` §5).
- **Operators are strategy classes** (CLAUDE Principle 1): each is an `equinox.Module`
  implementing a common face-flux / volume-source `Protocol`, constructed with its
  injected schemes, coefficients, and the geometry it reads. Methods are side-effect-free
  (immutable Module) — required for both `jit`/`grad` and testability.
- **Operators are injected into the assembly engine, not hard-wired.** The residual
  assembler is constructed with a list of operator strategy objects; it does not import
  specific operators. A test can assemble a residual from a single stub operator. (A
  Layer-0 escape hatch may accept a raw closure per `dsl-design-note.md` §7.1, but
  built-in operators are strategy classes.)
- **System-first**: the residual is over the whole state vector with a shared DOF
  layout, never a per-equation matrix (briefing §7). Coupling is inferred by AD from
  which unknowns a term reads.

## Testability seam
- The scatter engine must be testable with a hand-made 2–3 cell mesh and a stub
  closure returning a known flux — assert the `segment_sum` result cell-by-cell.
- Each operator ships an **order-of-accuracy unit test** on an analytic field
  (CLAUDE Principle 1), independent of any solve.
- The AD Jacobian of the diffusion operator is checked against the C++ non-orthogonal
  diffusion calculator as a golden numerical oracle (briefing §11.3).

## One-source-of-truth watch
Face geometry comes from `aquaflux/mesh`; interpolation/gradient come from
`aquaflux/schemes`. This module **composes** them — it does not re-derive geometry or
inline a scheme (CLAUDE Principle 2).
