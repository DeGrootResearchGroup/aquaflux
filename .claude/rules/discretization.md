---
paths:
  - "aquaflux/discretization/**"
---

# Rules — `aquaflux/discretization/` (the Layer-0 residual substrate)

> **Provenance boundary (binding).** This file cites the C++/Fortran precursors to inform
> *your* understanding — that is its job, and why it loads into your
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
  `CellBalance` delegates its scatter to `face_cells.scatter_conservative`. So this module owns
  the *physics* (the flux operators), not the `segment_sum` mechanics (the role the C++
  `FaceFluxAccumulator` plays).
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
  non-orthogonal-corrected flux) and the **transient** term (BDF1 at step 1, BDF2 after).
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
  ⚠️ **There is no `gamma` field on the context — it was replaced by `properties` and this entry
  described the retired shape until 2026-08-16.** `FaceContext.properties` is a
  `Mapping[str, jnp.ndarray]` carrying the assembler's whole evaluated property map (density,
  viscosity, conductivity, …), and each operator reads the property it names. That is deliberately
  **one** context field however many properties exist, so adding a property never changes the
  context's shape — the reason the properties model landed here rather than widening the context.
  Keep it single-sourced on the assembler, not baked into `DiffusionFlux` as operator config.
- **`diffusion.py` — BUILT.** `DiffusionFlux` (a `FaceFluxOperator` that gathers phi/grad/gamma/x
  from the context). Its optional **`boundary_coefficient`** field (`(n_faces,)`, default `None`)
  overrides the owner-cell `Gamma` **on boundary faces only** — a surface whose effective transport
  coefficient differs from its owner cell's. `None` and every interior face are untouched, so
  it is behaviour-neutral wherever not supplied (pinned in `test_diffusion.py`). **Two live consumers,
  both in the adaptive wall treatment** (see `.claude/rules/turbulence.md`), plus the obvious future
  ones (contact resistance, surface film): the momentum wall-function eddy viscosity `mu + rho
  nu_t,wall`, and the k equation's wall-face diffusivity `(1-f)·gamma` faded to zero as the wall cell
  enters the log layer. That second one is worth noting as a *pattern*: a boundary flux that must
  vanish smoothly is expressed by fading the **coefficient**, not by making the boundary **face value**
  approach the owner value — the flux is identical, but a state-dependent face value contributes
  `d(phi_ip)/d(phi_P)` to the residual's linearization and can drive it past one (measured: the k
  version diverges), whereas the coefficient carries the same dependence at the *frozen* closure state.
  Implements the
  flux-*continuous* DeGroot–Straatman normal
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
  - **The `denom` formula and the operator-diagonal conductance are single-homed here (binding, #154).**
    `flux_continuous_denominator(dpn, dnn, Γ_P, Γ_N)` = `(D_P·n) − (Γ_P/Γ_N)(D_N·n)` is the one home of
    the coefficient-jump denominator (`DiffusionFlux` imports it), and
    `flux_continuous_conductance(Γ, geometry, face_cells)` = `Γ_P A / denom` is the per-face **conductance**
    — exactly `d(owner-outward flux)/d(phi_P)`, so scattering it to both incident cells reproduces this
    operator's diagonal, and on an orthogonal graded face it is the harmonic mean `2Γ_PΓ_N/(Γ_P+Γ_N)·A/h`.
    It is the shared definition every consumer of "the transport operator's viscous/diffusive diagonal"
    calls — the momentum `a_P` (`flow/rhie_chow.py`), the scalar pseudo-time shift and frozen AMG
    operators (`turbulence/preconditioner.py`, `flow/block_preconditioner.py`) — so none can drift from
    the residual flux. Do **not** re-derive a face conductance with an owner/neighbour *arithmetic*
    interpolation of `Γ`: it agrees only for constant `Γ` and over-counts a graded face by `(1+r)²/(4r)`
    (`r = Γ_N/Γ_P`), which is what made `a_P` disagree with the operator diagonal under a graded turbulent
    viscosity (issue #154; the `.claude/rules/flow.md` `a_P` note has the consequences).
- **`residual.py` — BUILT. Two objects: the CONTEXT half and the BALANCE half (binding — do not
  re-fuse them).** `ResidualAssembler` (`equinox.Module`, built via `.build()` from an
  injected `BoundaryConditions({name: closure})` collection, which it binds to the mesh
  (`boundary.resolve(mesh.face_patches)`, off the jit path) and stores as a single `boundary` field)
  builds the **context**: it reconstructs cell gradients once (injected `GradientScheme`, optional —
  `None` on orthogonal grids where the correction vanishes), evaluates the per-patch boundary
  closures, evaluates the `PropertyModel`, and packs a `FaceContext`. **`CellBalance`** then
  assembles the **balance** from that context: sum the injected `FaceFluxOperator`s,
  `segment_sum`-scatter (owner `+`, interior neighbour `−`), subtract the `VolumeSource`s, add the
  transient. `R = accumulation − transport`. The assembler holds a `CellBalance` (field `balance`)
  and delegates, so `build`/`residual` are unchanged for a scalar equation.
  - **`CellBalance` stores ONLY its operators** — `flux_operators`, `source_operators`, `transient`.
    The connectivity, geometry, boundary values, gradient and properties all arrive on the
    `FaceContext` it is handed, exactly as its operators gather from it. So it needs no mesh to
    construct and no `BoundaryConditions` to test (`test_cell_balance.py` hands it a hand-made
    context), and it does not duplicate the geometry leaves the assembler already holds.
  - **Why split:** the coupled flow needs the balance but *cannot* share the context step.
    `MomentumContinuity` reconstructs one velocity-gradient **tensor** shared across its components,
    from `FlowBoundary` closures that take **no gradient** (so the assembler's leading-order
    two-pass has nothing to resolve there) — it arrives already holding a context. It therefore
    drives a `CellBalance` per component directly. That is what let the momentum pressure term stop
    being hand-added and become an ordinary operator (`flow.PressureForce`); see
    `.claude/rules/flow.md`.
  - **⚠️ The flux-operator tuple order IS the summation order, and floating-point addition is not
    associative.** Reordering a balance's operators perturbs the residual in the last bits — which
    matters here because archived march trajectories are compared bit-for-bit. The momentum block's
    order is fixed at `(DiffusionFlux, PressureForce, AdvectionFlux)` to reproduce the arithmetic
    of the hand-assembled form it replaced; pinned by
    `test_cell_balance.py::test_operators_are_summed_in_tuple_order`.
  - **Verified, and the extraction was gated on BIT-IDENTITY rather than on tests passing** (2026-08-15,
    at the base `efc10e3`): the coupled `bfs3d` residual at `state-00067`, the momentum residual with
    and without the wall-function eddy viscosity, `mdot` and `a_P` — all bit-unchanged; plus 97 arrays
    over a 2D/3D × compact/corrected × Stokes/upwind/limited × pin-on/off sweep. Stub-operator scatter
    is conservative and correctly signed (`test_cell_balance.py`).
    ⚠️ **Method note for any future bit-identity comparison: include quantities the change should NOT
    touch.** The first run of that harness showed order-one differences in `mdot` and `a_P`, which this
    change cannot reach — the cause was `hash()` on strings being randomized per process, so the two
    runs drew different random states. A residual-only comparison would have read as a real regression.
  - **`residual(phi, *, gradient_hook=None)` seam.** `gradient_hook` is an optional transform
    `gradient -> gradient` overwriting ghost rows with the value their owning partition computed
    (identity when omitted). It exists so the **distributed** residual can correct ghost-cell
    gradients (a ghost's local stencil is incomplete, so its reconstructed gradient is wrong). It is
    used at **two depths**: `residual` applies it to the returned gradient *after* reconstruction (so
    the flux reads exchanged ghost gradients), **and** threads it into `_gradient →
    GradientScheme.gradients(operator_hook=…) → GradientSolve.solve(operator_hook=…)` so an
    *iterative* gradient scheme (`CorrectedGreenGauss` with `SweptGradientSolve`) refreshes the ghost
    rows of its solve unknown before each apply — its operator couples across partitions, so one
    exchange is not enough. A single-pass scheme ignores the deeper use; boundary values are unaffected
    either way, because every boundary face is owned by an interior cell of its own partition, so its
    closures read only owner-cell gradients the ghost exchange leaves untouched. (`SweptGradientSolve`
    honours `operator_hook`; the reduction-forming `GmresGradientSolve` and the nested-solve
    `HessianCorrectedGradient` **raise** — see `.claude/rules/parallel.md`.)
- **`transient.py` — BUILT.** `TransientTerm`: BDF1 at step 1 (static `first_step`), BDF2
  after; carries no physical coefficient. Verified against the closed BDF formulae.
- **`fixed_value.py` — BUILT (`FixedValueCells` + the injected `FixationRow`).** Replaces a chosen set
  of cells' residual rows with a strong algebraic constraint (a pinned reference pressure, the
  near-wall ω fixed to its analytical value) while every other cell keeps its balance. The target is a
  differentiable leaf. **How the constraint is written is an injected strategy, because it must match
  the variable being solved (binding).** `DifferenceRow` (`phi − target`, the default) is right when
  the solved unknown *is* `phi`; `LogRatioRow` (`log(phi/target)`) is its counterpart for a field
  solved in log form, where it equals `w − log target` and so is **exactly linear in the unknown with
  derivative 1 at any ratio**. The difference row under a log parametrization is instead exponential
  in the unknown: its correction `δw = target/phi − 1` overshoots to `phi·e^(r−1)` against a target
  ratio `r`, which both wrecks the step and (because the row then carries the scale of `phi`) lets a
  handful of fixation cells dominate the residual norm the whole march is judged by. Same root either
  way — only the path and the conditioning differ. The turbulence `ScalarVariableTransform` owns the
  choice (`fixation_row()`), since it is the only object that knows which variable is being stepped;
  do **not** branch on the transform inside the residual. Measured impact on the coupled RANS case:
  see `.claude/rules/turbulence.md`.
  - **A `FixationRow` also reports its own derivative — `jacobian_scale(phi, chain)` (binding).** Given
    the field's `chain = d(phi)/d(unknown)`, it returns `d(row)/d(unknown)`: `DifferenceRow` → `chain`;
    `LogRatioRow` → `chain/phi`, hence exactly **1** for a log-solved field. **Why the interface needs
    it:** a fixation row and a transport row of the same block can differ by orders of magnitude in the
    solved unknown, so anything rescaling the block *per row* must ask each row rather than apply the
    block-wide chain factor everywhere. Skipping this was a real regression — the coupled RANS
    preconditioner rescaled the near-wall ω fixation rows by `1/ω`, giving a `1e-5` eigenvalue cluster
    that stalled the Krylov solve (27× worse linear residual; see `.claude/rules/turbulence.md`). Pin
    any implementation against **AD of its own `row`**, which is what keeps the two from drifting.
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
  (C++) and hand-assembled block coefficients (Fortran); **AD deletes all of it**. Write
  the **full physical flux** as one residual term — including the non-orthogonal correction
  **in the residual**, not deferred to an explicit RHS as both references must do. AD then
  puts the correction *in the matrix*, giving a more accurate operator than either reference.
  This is a Milestone-0 deliverable, not incidental.
- **Operators are strategy classes** (CLAUDE Principle 1): each is an `equinox.Module`
  implementing a common face-flux / volume-source `Protocol`, constructed with its
  injected schemes, coefficients, and the geometry it reads. Methods are side-effect-free
  (immutable Module) — required for both `jit`/`grad` and testability.
- **Operators are injected into the assembly engine, not hard-wired.** The residual
  assembler is constructed with a list of operator strategy objects; it does not import
  specific operators. A test can assemble a residual from a single stub operator. (A
  Layer-0 escape hatch may accept a raw closure, but built-in operators are strategy
  classes.)
- **System-first**: the residual is over the whole state vector with a shared DOF
  layout, never a per-equation matrix. Coupling is inferred by AD from which unknowns a
  term reads.

## Testability seam
- The scatter engine must be testable with a hand-made 2–3 cell mesh and a stub
  closure returning a known flux — assert the `segment_sum` result cell-by-cell.
- Each operator ships an **order-of-accuracy unit test** on an analytic field
  (CLAUDE Principle 1), independent of any solve.
- The AD Jacobian of the diffusion operator is checked against the C++ non-orthogonal
  diffusion calculator as a golden numerical oracle.

## One-source-of-truth watch
Face geometry comes from `aquaflux/mesh`; interpolation/gradient come from
`aquaflux/schemes`. This module **composes** them — it does not re-derive geometry or
inline a scheme (CLAUDE Principle 2).
