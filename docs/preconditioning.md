# Preconditioning

Every Newton step solves a linear system, and the linear solve is where almost all of a
flow simulation's time goes. This page covers what `aquaflux` offers to make that solve
converge, how to choose between the options, and how to keep one healthy over a long run.

If you only want the short answer: build
{class}`~aquaflux.flow.BlockPreconditioner` for a pressure–velocity solve, use
{func}`~aquaflux.turbulence.coupled_amg_continuation` for a coupled flow-plus-turbulence
solve, and read the rest of this page when one of them stops converging.

```{note}
The flow examples below continue from `cavity`, the lid-driven-cavity assembler built in
[Steady-state solving](steady_state_solving.md), and from the `mesh` and `geometry` it was
built on; `assembler` stands for any {class}`~aquaflux.flow.MomentumContinuity`. The coupled
examples continue from a `coupled` system you have already assembled, and `reference_state`
is a representative flow to freeze the preconditioner at — usually the state you are starting
the march from.
```

## Why any of this is necessary

For a scalar diffusion problem an unpreconditioned Krylov method works, just slowly: the
iteration count grows roughly like `h^-1` as the mesh refines, so the cost per solve grows
faster than the number of cells.

The coupled pressure–velocity system is a different situation. Its Jacobian has the
saddle-point structure

```text
[ F  G ] [ δu ]   [ r_u ]
[ B  Ĉ ] [ δp ] = [ r_p ]
```

with `F` the momentum block, `G` the pressure gradient, `B` the divergence, and `Ĉ` the
pressure–pressure coupling that a collocated Rhie–Chow discretization carries to suppress
checkerboarding. This matrix is **indefinite** — it has eigenvalues of both signs — and a
generic Krylov method does not converge usefully on it at any mesh size. The cure is a
preconditioner that captures the **pressure Schur complement** `S = Ĉ - B F⁻¹ G`. With a
spectrally-equivalent approximation of `S`, the iteration count becomes mesh-independent
(Murphy, Golub & Wathen 2000: a block preconditioner with the exact Schur complement gives
Krylov convergence in a fixed, small number of iterations regardless of size).

So preconditioning here is not a tuning step you reach for when things are slow. For any
coupled flow solve it is what makes the solve work at all.

## What a preconditioner is in this package

Three properties hold for every preconditioner described below, and each one is something
you can rely on rather than an implementation detail.

**It is frozen, so it never changes the answer.** Every coefficient a preconditioner holds
is detached from automatic differentiation. It accelerates the Krylov iteration and
nothing else: it cannot move the converged state, and it cannot bias the gradient taken
through that state. The practical consequence is that a preconditioner built at one
operating point stays *valid* — never wrong, only less effective — at another, which is
what makes it reusable across a parameter sweep and refreshable mid-run without disturbing
the trajectory.

**It is a fixed linear map.** Each inner solve runs a fixed number of cycles or sweeps
rather than iterating to a tolerance. That is deliberate: a preconditioner whose work
varies with its input is a *different* operator on each application, which would require a
flexible Krylov method and would have no transpose. Because the map is fixed,
plain non-flexible GMRES suffices, and the adjoint's transpose comes from
`jax.linear_transpose` on the same code rather than from a separately written and
separately wrong transposed cycle.

**It is approximate on purpose.** The inner solves only have to make Newton progress, so
each step's linear solve is taken to a loose tolerance and the outer iteration cleans up
what is left. The single adjoint solve is the exception: it is taken tightly, because its
accuracy *is* the gradient's accuracy.

```{warning}
Build a preconditioner **outside** `jax.grad`, from concrete parameter values. Constructing
one inside a differentiated function captures JAX tracers in an object that is deliberately
not differentiated, and the solve then fails with a tracer-leak error. Since the
preconditioner cannot affect the gradient anyway, there is never a reason to build it
inside.
```

## Choosing one

| What you are solving | Use | Section |
| --- | --- | --- |
| A scalar transport or diffusion equation | {func}`~aquaflux.turbulence.scalar_transport_preconditioner` | [Scalar transport](#scalar-transport) |
| Pressure–velocity flow, on its own | {class}`~aquaflux.flow.BlockPreconditioner` | [Pressure–velocity flow](#pressurevelocity-flow) |
| Coupled flow and turbulence (`u, v, w, p, k, ω`) | {func}`~aquaflux.turbulence.coupled_amg_continuation` | [Coupled flow and turbulence](#coupled-flow-and-turbulence) |
| The same coupled system, on a moderate 2D mesh | {func}`~aquaflux.turbulence.coupled_lu_continuation` | [A complete factorization](#a-complete-factorization) |
| Your own linear system | {func}`~aquaflux.solve.solve_linear` | [Using one directly](#using-one-directly) |

## Scalar transport

A single transported scalar — a species concentration, temperature, a tracer, or one of
the turbulence variables — gives a nonsymmetric convection–diffusion matrix. The
preconditioner for it is an algebraic multigrid (AMG) V-cycle on a frozen linearization of
that operator, built by
{func}`~aquaflux.turbulence.scalar_transport_preconditioner`:

```python
import jax.numpy as jnp
from aquaflux.flow import volume_flux
from aquaflux.turbulence import scalar_transport_preconditioner

rho = 1.0                                               # the flow's constant density
flux = volume_flux(cavity.mass_flux(flow_state), rho)

precond = scalar_transport_preconditioner(
    mesh,
    geometry,
    jnp.full(mesh.n_cells, 1e-3),   # per-cell effective diffusivity
    flux,                           # per-face volumetric flux
    transport.residual(flux),       # phi -> residual, at that flux
    jnp.zeros(mesh.n_cells),        # the field to freeze the linearization at
)
```

`flow_state` is a converged flow and `transport` a
{class}`~aquaflux.transport.ScalarTransport` assembler. Note where the flux comes from: a
concentration rides the **volumetric** flux, so it is the flow's mass flux divided by density
({func}`~aquaflux.flow.volume_flux`), not the mass flux itself. That conversion takes the
**scalar** density, since a scalar written in kinematic form assumes a constant one. And
{meth}`ScalarTransport.residual <aquaflux.transport.ScalarTransport.residual>` takes that flux
and *returns* the `phi -> residual` closure, which is what gets frozen.

It returns a {class}`~aquaflux.turbulence.ScalarTransportPreconditioner`, which is a
`phi -> M` factory of the shape the solvers expect.

The `method` argument picks the hierarchy:

- `"twolevel"` (default) builds a {class}`~aquaflux.turbulence.ConvectionAmgPreconditioner`
  — a stable two-level nonsymmetric aggregation. Two levels by design: a
  coarse-of-coarse convection operator acquires modes that a single-factor smoother cannot
  damp.
- `"air"` builds an {class}`~aquaflux.turbulence.AirAmgPreconditioner`, the reduction-based
  local approximate ideal restriction (lAIR) method of Manteuffel, Ruge & Southworth
  (2018). It coarsens all the way down and stays mesh-independent, which is what the
  two-level method gives up.

Two smaller pieces belong here. `fixed_cells` names cells whose residual is a value
fixation rather than a transport balance — a prescribed near-wall `ω`, for instance. Those
rows are the identity, so they are detached from the aggregation to match the operator the
solve actually inverts; leaving them in gives the coarsening a stencil the matrix does not
have. And {class}`~aquaflux.turbulence.ScaledScalarPreconditioner` wraps any scalar
preconditioner with a fixed per-cell output scaling, which is how a solve in a transformed
variable (`log ω`, say) reuses a preconditioner built for the untransformed one.

## Pressure–velocity flow

{class}`~aquaflux.flow.BlockPreconditioner` is the preconditioner for the coupled
pressure–velocity saddle. It composes two inner solves — an approximate momentum inverse
and an approximate pressure-Schur inverse — and it is built from the flow assembler:

```python
from aquaflux.flow import BlockPreconditioner
from aquaflux.solve import DampedNewtonStep, ImplicitNewtonSolver

precond = BlockPreconditioner.build(cavity).factory()
solver = ImplicitNewtonSolver(
    max_steps=30,
    forward_step=DampedNewtonStep(preconditioner=precond),
)
```

`build` returns the preconditioner; `factory()` returns the `state -> M` callable the
forward step expects. Three arguments matter, and they are independent of one another.

### `velocity` — how the momentum block is inverted

| Value | What it builds |
| --- | --- |
| `"smoothed"` (default) | Smoothed-aggregation AMG on the **viscous** momentum operator. Mesh-independent, but blind to the cell Péclet number, so it bounds the Reynolds number the solve reaches. |
| `"convection"` | A two-level hierarchy on the frozen `viscous + first-order-upwind` operator, so it stays a good approximation as convection strengthens. |
| `"convection-air"` | The same convective linearization under an lAIR hierarchy — Péclet-robust *and* mesh-independent. |

Both convection-aware blocks freeze their linearization at a `reference_state`. Pass one
if you have a representative flow; otherwise the boundary conditions supply a uniform flow
at the fastest velocity any patch prescribes, so the frozen operator carries the operating
Péclet number without assuming a flow speed.

### `schur_scaling` — which Schur approximation

| Value | What it is |
| --- | --- |
| `"simple"` (default) | The classical SIMPLE Schur: a pressure Laplacian scaled by the momentum diagonal, `Ŝ ~ B diag(V/a_P) Bᵀ`. Degrades as convection strengthens, because `a_P` does. |
| `"msimple"` | The same Laplacian scaled instead by a frozen, velocity-independent mass diagonal `Q̂ = ρV/k`, giving a constant-coefficient pressure Poisson. Because it does not track the velocity, it does not degrade with convection — the variant that carries a flow-only solve past the Reynolds number at which the `a_P` Schur stalls. |
| `"lsc"` | The algebraic, nonuniform-mesh stabilized least-squares commutator of Elman, Howle, Shadid, Silvester & Tuminaro (2007), built from the momentum operator itself rather than from a diagonal. The *stabilized* variant is the relevant one, because a Rhie–Chow collocated discretization is equal-order stabilized. |

Both scaled Laplacians are near-Stokes approximations. Once a flow is strongly
convection-dominated, what limits the block is the *approximation* rather than how
accurately it is inverted — at which point raising `v_cycles` does not help, and on the
Schur it can hurt, because inverting the wrong operator more exactly is not progress.
`"lsc"` is markedly dearer per application (two multigrid solves plus three residual
linearizations, against one solve) and is stronger on an isolated flow saddle; it is
**not** a good choice inside a coupled flow–turbulence solve.

### `composition` — how the two solves are combined

The two inner solves can be arranged in more than one way, and the arrangement is a
separate choice from either solve. All three come from Klaij & Vuik (2013), whose
pressure-correction Schur is what the scaled Laplacians above assemble.

| Value | What it does | Cost per application |
| --- | --- | --- |
| `"triangular"` (default) | `δu = F⁻¹ r_u`, then `δp = Ŝ⁻¹(r_p - B δu)`. The lower block-triangular preconditioner: with exact inner solves the preconditioned operator is unipotent, the Murphy–Golub–Wathen structure a Krylov method resolves in two iterations. | 1 velocity solve, 1 Schur solve |
| `"simple"` | Their Algorithm 1: the above plus the closing velocity update `δu ← δu - F̃⁻¹ G δp`, which is the second factor of the block `LU`. | 1 velocity solve, 1 Schur solve |
| `"simpler"` | Their Algorithm 2: a **pressure prediction** before the velocity solve, so the momentum block is inverted against an already plausible pressure, then the velocity solve, then the correction. | 1 velocity solve, 2 Schur solves |

The paper's named methods are the two axes together: **SIMPLER** is
`schur_scaling="simple", composition="simpler"`, and **MSIMPLER** is
`schur_scaling="msimple", composition="simpler"`.

```python
precond = BlockPreconditioner.build(
    assembler,
    velocity="convection",
    schur_scaling="msimple",
    composition="simpler",     # MSIMPLER
).factory()
```

```{note}
Pair `composition="simpler"` with `schur_scaling="msimple"`. The prediction solves the
Schur against a velocity-derived right-hand side, which makes it much more sensitive than
the correction is to the Schur being a poor match for the operator it stands in for — and
on the `a_P`-scaled Schur that sensitivity is enough to stop the solve converging.
```

### The remaining arguments

`v_cycles` sets the multigrid cycles per inner apply; see the caution above before raising
it. `strength_threshold` (default `0`, meaning isotropic aggregation over the full graph)
aggregates only along **strong** connections when set — typically `0.25`. That is what
keeps the V-cycle contracting on a high-aspect-ratio or skewed mesh, where isotropic
aggregation coarsens across the stiff wall-normal direction. It is close to a no-op on a
low-aspect-ratio mesh. It makes the coarsening depend on the operator's *values* rather
than only its sparsity, so prefer it where the hierarchy is frozen — as this one is —
rather than repeatedly refreshed.

### Setting these from a continuation

You rarely build the preconditioner by hand. {func}`~aquaflux.flow.momentum_continuation` and
{func}`~aquaflux.flow.reused_flow_solve` forward any extra keyword straight to
{meth}`BlockPreconditioner.build <aquaflux.flow.BlockPreconditioner.build>`, so every option
above is settable where you already are:

```python
from aquaflux.flow import momentum_continuation
from aquaflux.solve import ImplicitNewtonSolver

continuation = momentum_continuation(
    assembler,
    beta0=2.0,                  # continuation's own argument
    velocity="convection",      # from here on, the preconditioner's
    schur_scaling="msimple",
    composition="simpler",
    strength_threshold=0.25,
)
solver = ImplicitNewtonSolver(max_steps=120, forward_step=continuation)
```

The continuation also hands the same preconditioner to the adjoint solve, so the gradient is
preconditioned without any further arrangement.

## Coupled flow and turbulence

Once the turbulence model is solved together with the flow, the system carries six fields
per cell (`u, v, w, p, k, ω`) and the block structure above no longer describes it. The
preconditioners for this system work from a **materialized** Jacobian: the coupled operator
is recovered by coloured probing into a sparse matrix, off the traced path, and a frozen
inverse of that matrix is applied as the preconditioner.

The entry point is a continuation builder rather than a bare preconditioner, because the
preconditioner and the pseudo-transient march that uses it are built together:

```python
from aquaflux.turbulence import coupled_amg_continuation, solve_coupled

continuation = coupled_amg_continuation(coupled, reference_state)
flow, k, omega = solve_coupled(coupled, continuation=continuation)
```

{func}`~aquaflux.turbulence.coupled_amg_continuation` builds a
{class}`~aquaflux.solve.MonolithicAmgPreconditioner`: a single algebraic-multigrid V-cycle
over the whole six-field matrix, equilibrated and reordered cell-major so that each cell's
six unknowns are adjacent. The arguments worth knowing:

- `amg_beta` — the pseudo-transient shift the preconditioner is built at. A shifted
  operator is easier to precondition, and a preconditioner built at a shift close to the
  march's own is the one that helps it.
- `stencil_reach` / `column_reach` — how far the probing recovers the Jacobian across the
  cell graph. The coupled flow Jacobian is intrinsically distance-2, because Rhie–Chow
  damping couples pressure to the neighbour-of-a-neighbour ring, so a reach of 3 is the
  working value. Shortening it does not merely drop small terms: a colouring is
  collision-free only for the pattern it was built at, so an under-reaching probe folds far
  couplings onto near entries instead.
- `smoother_fill_levels` / `smoother_sweeps` — the incomplete factorization used as the
  level smoother, and how many sweeps of it.
- `field_split` — see below.

### Splitting the fields

The flow saddle and the transported turbulence pair are different kinds of operator, and a
single hierarchy over both has to compromise. `field_split=True` instead builds a
{class}`~aquaflux.solve.FieldSplitAmgPreconditioner`, wrapping a
{class}`~aquaflux.solve.BlockTriangularFieldSplit`: one inverse for the leading
`[u, v, w, p]` group, another for the trailing `[k, ω]` group, and one retained coupling
block between them. Each group then gets an inverse suited to it, injected as
`leading_inverse` and `trailing_inverse`:

| Factory | Builds | Suited to |
| --- | --- | --- |
| {func}`~aquaflux.solve.native_saddle_inverse` | {class}`~aquaflux.solve.NativeSimpleInverse` — a multigrid over the saddle whose *level smoother* is a SIMPLE relaxation | the leading flow group |
| {func}`~aquaflux.solve.native_nodal_inverse` | {class}`~aquaflux.solve.NodalNativeInverse` — one hierarchy over the whole group, coarsening cells | either group |
| {func}`~aquaflux.solve.air_inverse` | a reduction-based (lAIR) hierarchy | the trailing transported group |
| {func}`~aquaflux.solve.host_ilu_inverse` | {class}`~aquaflux.solve.HostVCycleInverse` — the same coarsening, relaxed by an incomplete factorization | either group |

Note the relationship between the first of these and
{class}`~aquaflux.flow.BlockPreconditioner`, because it is easy to misread. Both are
SIMPLE-type. But `BlockPreconditioner` is **flat**: one application solves the velocity
block and the Schur and combines them, with no hierarchy over the saddle at all — the
multigrid lives *inside* it, on each sub-block.
{class}`~aquaflux.solve.NativeSimpleInverse` is the other arrangement: a genuine hierarchy
over the whole saddle, in which a SIMPLE relaxation is the smoother at each level and the
coarse grid carries the smooth global pressure mode. That mode is the one any SIMPLE-type
Schur approximates worst, which is why the arrangement matters.

{func}`~aquaflux.solve.build_block_triangular_field_split` builds the split directly if you
want one outside a continuation. {class}`~aquaflux.solve.FieldGroups` says where the
partition falls:

```python
from aquaflux.solve import (
    FieldGroups,
    build_block_triangular_field_split,
    native_nodal_inverse,
    native_saddle_inverse,
)

groups = FieldGroups(n_cells=mesh.n_cells, n_leading_fields=4, n_trailing_fields=2)
split = build_block_triangular_field_split(
    matrix,          # the assembled six-field Jacobian, as a scipy sparse matrix
    groups,
    leading_inverse=native_saddle_inverse(strength_threshold=0.25, max_levels=5),
    trailing_inverse=native_nodal_inverse(max_coarse=200),
)
```

`flow_first=True` (the default) solves the leading group first, which retains the
trailing-by-leading coupling and discards the other corner. Which corner is discarded is a
real choice rather than a symmetry: on a coupled flow–turbulence system the turbulence
equations depend on the flow far more strongly than the reverse, so keeping that direction
is the one to keep.

### A complete factorization

{func}`~aquaflux.turbulence.coupled_lu_continuation` builds a
{class}`~aquaflux.solve.MonolithicLuPreconditioner` instead — a **complete** sparse LU of
the coupled matrix. On a moderate 2D mesh this is the strongest option available and often
the fastest overall, because it converges the linear solve in very few iterations. It does
not scale: the factorization's fill grows quickly with mesh size, which is what the
multigrid path exists to avoid. Use it where the mesh is small enough to afford it, and as
a reference when you want to know how much of a slow solve is the preconditioner's fault.

## The pieces underneath

The hierarchies and factorizations above are public in their own right, and can be built
against any `scipy` sparse matrix.

| Builder | Hierarchy |
| --- | --- |
| {func}`~aquaflux.solve.build_smoothed_hierarchy` | Smoothed-aggregation AMG for a symmetric operator, returning a {class}`~aquaflux.solve.SmoothedHierarchy`. |
| {func}`~aquaflux.solve.build_convection_hierarchy` | Aggregation multigrid for a nonsymmetric convection–diffusion operator. |
| {func}`~aquaflux.solve.build_air_hierarchy` | Reduction-based lAIR, returning an {class}`~aquaflux.solve.AirHierarchy`. `block_size` runs the coarsening on the cell graph so it works on a multi-field block. |

Each has a matching apply — {func}`~aquaflux.solve.smoothed_multigrid_solve`,
{func}`~aquaflux.solve.convection_multigrid_solve`,
{func}`~aquaflux.solve.air_multigrid_solve` — that runs a fixed number of V-cycles, which
is what keeps the result a fixed linear map.

{func}`~aquaflux.solve.convection_diffusion_operator` assembles the frozen operator these
coarsen: a symmetric diffusive edge coupling, optionally plus first-order-upwind convection
at a reference flux. The first-order upwinding is the *preconditioner's* choice and not the
model's — whatever advection scheme the residual uses, this operator upwinds first order,
because that is what makes it an M-matrix an aggregation hierarchy can coarsen.

{class}`~aquaflux.solve.Ilu0` is a zero-fill incomplete factorization, refreshable in
place. How it orders its elimination is an injected strategy,
{class}`~aquaflux.solve.EliminationOrdering`, over a {class}`~aquaflux.solve.CellOrder` —
{class}`~aquaflux.solve.NaturalCells`, {class}`~aquaflux.solve.ReverseCuthillMcKeeCells` or
{class}`~aquaflux.solve.AscendingRowLengthCells`. That is a strategy rather than a knob
because at zero fill the ordering decides *which* couplings the factorization discards, and
on a coupled saddle that choice has taken a stationary sweep from amplifying the residual to
contracting it.

```{note}
`Ilu0` has a compiled kernel that must be built once per checkout with
`tools/build_ext.sh`. Without it the package still imports and runs, falling back to a pure
Python implementation with identical results but very different speed — so timings taken
in a fresh checkout are not comparable to timings taken in a built one.
`aquaflux.solve.ilu0.COMPILED` reports which one is live.
```

## Keeping it current

A frozen preconditioner drifts as the solution develops away from the state it was built
at. On a long march that drift eventually costs more than a rebuild would, so
{class}`~aquaflux.solve.RefreshPolicy` lets the solve rebuild it mid-run:

```python
from aquaflux.solve import CycleGrowthTrigger, RefreshPolicy
from aquaflux.turbulence import solve_coupled

flow, k, omega = solve_coupled(
    coupled,
    continuation=continuation,
    refresh=RefreshPolicy(trigger=CycleGrowthTrigger(growth=2.0), limit=4),
)
```

{class}`~aquaflux.solve.CycleGrowthTrigger` fires when the linear solve's cycle count grows
past a multiple of its early baseline — the direct symptom of a stale preconditioner. It
gates on the residual as well, because the cycle count also rises as the pseudo-transient
shift falls, and that rise is not staleness.

{func}`~aquaflux.turbulence.amg_beta_tracking_refresh` is the counterpart for the coupled
path. It re-preconditions **in place** as the march's shift moves, so the compiled solve is
reused rather than retraced, and it goes in as the policy's `precondition_step`:

```python
from aquaflux.solve import RefreshPolicy
from aquaflux.turbulence import amg_beta_tracking_refresh

refresh = RefreshPolicy(
    precondition_step=amg_beta_tracking_refresh(coupled, refresh_every=8, beta_rel_change=0.25)
)
```

It refreshes on a schedule, on shift drift, or when a single solve proves expensive. It reports what each rebuild cost through
{class}`~aquaflux.solve.RefreshTiming` — which branch ran, the total, and the parts.
{data}`~aquaflux.solve.NO_REFRESH` is the do-nothing policy, and the default.

Refreshing is safe for the same reason freezing is: the preconditioner is
`stop_gradient`-ed whatever state it is frozen at, so a refresh changes the forward Krylov
count and nothing else — not the converged state, not the gradient.

## Using one directly

{func}`~aquaflux.solve.solve_linear` takes a preconditioner for any matrix-free system:

```python
import jax
import jax.numpy as jnp
from aquaflux.solve import solve_linear

a = jnp.array([[4.0, 1.0], [1.0, 3.0]])
diagonal_inverse = 1.0 / jnp.diag(a)                     # a Jacobi preconditioner

x, cost = solve_linear(
    lambda v: a @ v,
    jnp.array([1.0, 2.0]),
    preconditioner=lambda r: diagonal_inverse * r,
)
```

`cost` is the solver's own iteration count, which is what you watch when tuning — see
[When it is not working](#when-it-is-not-working).

It preconditions on the **right** by default. That choice matters for more than symmetry:
with right preconditioning the residual the Krylov method measures and stops on is the
*true* residual of the original system. Under left preconditioning it stops on `M(Ax - b)`
instead, which is a different measure for every preconditioner — so two preconditioners
cannot be compared on it, and a solve can report success while the true residual is orders
away from the tolerance asked for. Pass `preconditioner_side="left"` only deliberately.

{func}`~aquaflux.solve.relative_residual_gmres` gives a GMRES that stops on a **global**
relative residual rather than `lineax`'s componentwise test. On a coupled system whose
right-hand side has near-zero entries, the componentwise test quietly becomes an absolute
demand and drives the solve orders of magnitude past the tolerance requested.

## Gradients

The preconditioner is part of the forward solve, and the reverse-mode gradient is a single
transpose linear solve at the converged state. That transpose solve wants preconditioning
too, and it gets it from the same object: because every preconditioner here is a fixed
linear map, `jax.linear_transpose` produces `Mᵀ` from the forward code exactly.

One case needs care. A few inverses can produce their transpose more cheaply than
`jax.linear_transpose` would — or are host objects that it cannot trace at all. Wrapping
such a factory in {class}`~aquaflux.solve.TransposedPreconditioner` marks its output as
*already* transposed, so the adjoint applies it as-is instead of transposing it a second
time.

The adjoint solve is taken to a tight tolerance where the forward solves are deliberately
loose, because the forward iteration corrects its own inexactness and the adjoint has
nothing after it to correct anything.

## When it is not working

Read the **cycle count**, not the wall clock. {func}`~aquaflux.solve.restart_cycles` strips
the fixed per-solve offset from a raw `lineax` iteration count, which is what makes small
counts readable: an offset means a solve that converges within a single restart cycle does
not report `1`.

From there:

- **The count climbs steadily over a march.** The preconditioner is going stale — add a
  {class}`~aquaflux.solve.RefreshPolicy`, or refresh more often.
- **The count is high from the first step.** The preconditioner is mismatched rather than
  stale. On a high-aspect-ratio mesh try `strength_threshold=0.25`; on a
  convection-dominated flow move the velocity block to `"convection"` and the Schur to
  `"msimple"`.
- **More inner cycles do not help, or make it worse.** The Schur *approximation* is the
  limit, not its inversion. Change the approximation, not its accuracy.
- **The solve converges but the Newton step is rejected.** That is globalization rather
  than preconditioning: see [Steady-state solving](steady_state_solving.md).
- **It converges on a small mesh and fails on a large one.** Suspect a two-level method
  where a fully coarsening one is needed — `"air"` for a transported scalar, or an lAIR
  block inverse in a field split.
