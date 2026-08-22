# Gradient reconstruction

A finite-volume solver stores one value per cell, but almost everything it computes needs a
**gradient**: the non-orthogonal correction in a diffusion flux, the reconstruction of a face
value for second-order advection, and the pressure gradient in a momentum balance. On a mesh
of perfect cubes any reasonable estimate will do. On a real mesh — graded, skewed, cut by a
CAD surface — the gradient scheme is often what sets the accuracy of the whole solve.

This page covers the three reconstructions `aquaflux` ships, how each one is solved, and how
to choose between them.

If you only want the short answer: use {class}`~aquaflux.schemes.CorrectedGreenGauss`, which
is the default everywhere, and read on if your mesh is skewed enough that you suspect the
gradient is the limiting error.

```{note}
The examples below use a `mesh` and its `geometry`, plus a cell field `phi` and the field's
values on the boundary faces. Every scheme has the same interface, so they are interchangeable
in any of the places a scheme is injected — a `ResidualAssembler`,
a {class}`~aquaflux.flow.MomentumContinuity`, a turbulence closure.
```

## The three schemes

```python
from aquaflux.schemes import (
    CompactGreenGauss,
    CorrectedGreenGauss,
    HessianCorrectedGradient,
)

gradient = CorrectedGreenGauss().gradients(phi, mesh, geometry, boundary_values)
```

### Compact Green–Gauss

{class}`~aquaflux.schemes.CompactGreenGauss` applies the divergence theorem directly:

```text
grad(phi)_P = (1 / V_P) * sum_faces  phi_ip * S_f
```

with the face value linearly interpolated between the two cells. It is one pass over the
faces, with no linear system, and it is second-order accurate on an orthogonal grid.

Its weakness is specific and worth knowing: the interpolation gives the value at the point
where the face centroid **projects onto the line joining the two cell centroids**, not at the
face centroid itself. On a skewed mesh those are different points, and the resulting error
does not shrink as the mesh refines — the scheme is *inconsistent* there, converging at
roughly zeroth order. Use it on near-orthogonal meshes, or where the gradient enters only as a
small correction.

### Corrected Green–Gauss

{class}`~aquaflux.schemes.CorrectedGreenGauss` closes exactly that gap. It adds the offset
from the projection foot to the true face centroid, evaluated with the gradient itself:

```text
phi_ip = (1-g) phi_P + g phi_N  +  [(1-g) grad(phi)_P + g grad(phi)_N] . D_g,ip
```

Because the correction uses the gradient being solved for, this is no longer a single pass —
it is a sparse linear system `A_g G = B phi` coupling each cell to its face neighbours. The
operator `A_g` depends only on the mesh, never on the field.

The payoff is that the reconstruction becomes **exact for linear fields on any mesh**, which
restores consistency on irregular grids. It remains capped near first order there for general
fields, because a Green–Gauss sum reproduces the mean of the face value, not its value at the
centroid, and that distinction survives the linear correction. This is the default scheme.

### Hessian-corrected (Betchen)

{class}`~aquaflux.schemes.HessianCorrectedGradient` implements the coupled gradient-and-Hessian
reconstruction of Betchen & Straatman (2010). Reconstructing the second derivative alongside
the gradient lets the face integral be exact for a quadratic, which lifts the gradient itself
to second order on any mesh:

```text
[ A_gg  A_gH ] [ g ]   [ b_g ]
[ A_Hg  A_HH ] [ H ] = [  0  ]
```

The Hessian is wanted only to correct the gradient, never as an output, so it is eliminated:
the system actually solved is `S g = b_g` with `S = A_gg - A_gH A_HH⁻¹ A_Hg`. Every block is
formed by automatic differentiation of the forward reconstruction rather than from
hand-derived coefficient matrices, so there is one statement of the discretization and no
second copy to drift.

It is exact for linear **and** quadratic fields on any mesh with planar faces. Where the
gradient enters a face value at leading order — advection, Rhie–Chow — that is the difference
between a scheme that caps near first order on a skewed mesh and one that does not.

```{warning}
A **warped** face breaks Green–Gauss exactness for a quadratic for every scheme in this
family, this one included, because the face integral itself is no longer exact. On a mesh
with badly non-planar faces the reconstruction is limited by the faces, not by the scheme;
{func}`~aquaflux.mesh.face_planarity` reports how close each face is to planar.
```

## How the system is solved

Two of the three schemes reduce to a linear system, and *how* that system is inverted is a
separate, injected choice — the discretization is identical either way. It matters because
the reconstruction runs inside every residual evaluation, so its cost is multiplied by
everything above it.

{class}`~aquaflux.schemes.SweptGradientSolve` applies a fixed number of preconditioned
Richardson sweeps:

```text
x <- x + P⁻¹ (b - A x)
```

There is no matrix, no nested Krylov solve, and no implicit-diff tangent: the loop has a
static length, so differentiating it is just unrolling. Its cost is linear in the mesh, which
is what makes it the default.

{class}`~aquaflux.schemes.GmresGradientSolve` solves the same system with matrix-free GMRES to
a requested tolerance, and is differentiated by implicit differentiation. It is exact to that
tolerance regardless of conditioning, at the cost of a nested Krylov solve inside every
reconstruction.

That cost is easy to underestimate, because it does not fall where you look for it. A flow solve
spends most of its time in Krylov iterations, and each one is a **Jacobian–vector product** through
the residual — so each one differentiates the reconstruction. A fixed sweep is differentiated by
unrolling, which costs about what the forward pass costs. Implicit differentiation instead solves a
*second* linear system per product. Measured on a coupled RANS case, per reconstruction against a
corrected Green–Gauss baseline of 1.0, at matched accuracy: a Krylov solve of the Hessian-corrected
system costs `93×` forward and `159×` as a Jacobian–vector product; a fixed sweep costs `67×` and
`60×`. The gap on the forward pass is modest; on the path the solver actually repeats, it is 2.6×.

This is why both schemes default to a fixed sweep, and why the Krylov strategy is kept for what it is
genuinely better at: a mesh skewed enough that the sweep count would grow impractically, and providing
an exact reference to calibrate a sweep count against.

```python
from aquaflux.schemes import CorrectedGreenGauss, GmresGradientSolve, SweptGradientSolve

CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=8))   # more sweeps for a skewed mesh
CorrectedGreenGauss(solver=GmresGradientSolve())           # exact, for a reference value

# The Hessian-corrected scheme has two systems and takes a strategy for each.
HessianCorrectedGradient(hessian_solver=GmresGradientSolve())   # exact inner, to calibrate against
```

### Preconditioning, and why it is a separate choice

Both solves take a {class}`~aquaflux.schemes.GradientPreconditioner` — an approximate inverse
of the operator's **per-cell diagonal block**. There are two, and the difference between them
is not a tuning knob:

{class}`~aquaflux.schemes.InverseVolume` scales each cell's residual by `1/V`. The corrected
gradient's operator is `A_g = V ⊙ I − (skewness coupling)`, so the volume really is most of
its diagonal block and this converges quickly.

{class}`~aquaflux.schemes.CellBlockJacobi` inverts each cell's own diagonal block exactly. The
gradient–Hessian system needs this, because there a cell's gradient and its Hessian couple to
*each other* at the same order as the volume term — something `1/V` cannot represent at all,
rather than merely approximate. The consequence is sharp: on a perfectly orthogonal mesh,
where there is no skewness whatsoever, inverse-volume Richardson on that system still has a
spectral radius of `0.5`, while inverting the cell block gives `0`. That is why the scheme
looked as though it needed a Krylov solve, and why it no longer does.

For the Hessian-corrected scheme there is a second question — *whose* block. After eliminating
the Hessian, the system being solved is the Schur complement
`S = A_gg − A_gH A_HH⁻¹ A_Hg`, so a preconditioner built from `A_gg`'s block alone omits the
elimination term. On a well-shaped cell that term is a small perturbation and omitting it costs
nothing; on a cell squashed nearly flat the volume vanishes while the face couplings do not, so
the omitted term becomes the *dominant* part of that cell's row and the sweep stops converging
there. `local_schur_block` (default `True`) builds the preconditioner from the Schur
complement's own block instead, which is better on every mesh measured — marginally on
well-shaped ones, by some four orders on a squashed cell — for roughly 7% of a reconstruction.
Set it to `False` for the slightly cheaper historical behaviour on a mesh of uniformly good
quality.

Each scheme supplies a sensible default, so this is not usually something you set — but for
the corrected Green–Gauss scheme it is worth knowing about, because on a poor-quality mesh the
default can fail outright.

### Choosing a preconditioner for a poor-quality mesh

{class}`~aquaflux.schemes.CorrectedGreenGauss` takes a `preconditioner`, and the choice is
between speed on a good mesh and robustness on a bad one:

```python
from aquaflux.schemes import CorrectedGreenGauss, ExactCellBlock

scheme = CorrectedGreenGauss()                                 # 1/V -- the default
scheme = CorrectedGreenGauss(preconditioner=ExactCellBlock())  # the true per-cell block
```

{class}`~aquaflux.schemes.InverseCellVolume` (the default) costs a reciprocal per cell and is
the right choice on a mesh of reasonable quality. Its accuracy rests on the volume being most
of the operator's diagonal block — and that assumption fails as a cell flattens, because the
neglected coupling scales with face **area** while the volume does not. On a mesh with a
near-degenerate cell, `1/V` stops approximating the block at all and the iteration diverges.

{class}`~aquaflux.schemes.ExactCellBlock` recovers the operator's true per-cell block by
probing it, and inverts that. It is insensitive to cell shape. Measured on a grid with one
cell progressively squashed flat, reconstructing a field whose gradient is known exactly:

| cell volume ratio | `InverseCellVolume` | `ExactCellBlock` |
| --- | --- | --- |
| 1.9 × 10² | 3.4 × 10⁻² | 3.5 × 10⁻² |
| 1.9 × 10⁴ | 2.6 × 10² | 1.7 × 10⁻² |
| 1.9 × 10⁶ | 1.7 × 10¹⁰ | 1.7 × 10⁻² |
| 1.9 × 10⁸ | 1.7 × 10¹⁸ | 1.7 × 10⁻² |

The default's error grows without bound with the volume ratio; the block holds the scheme's
own discretization error, flat, across six orders of magnitude. On a healthy mesh the two agree
to four significant figures, so this buys robustness and not accuracy.

It is not free: extracting the block costs a few extra operator applies, and applying it is a
small matrix product per cell rather than a scalar multiply — together roughly three times the
default's cost at four sweeps, though the extraction is a fixed prologue whose share falls as
the sweep count rises.

**When to reach for it.** If your mesh comes from an automatic mesher and has slivers, or if a
case diverges or produces implausible gradients in a small number of cells, switch to
`ExactCellBlock`. If your mesh is of good quality, the default is faster and just as accurate.
{func}`~aquaflux.mesh.quality.closed_cell_residual` and
{func}`~aquaflux.mesh.quality.face_planarity` will tell you which kind of mesh you have — and
note that a mesh whose cells do not *close* is invalid outright, which no preconditioner can
repair, since the whole discretization is the divergence theorem.

If you calibrate the sweep count (below), pass the preconditioner to
{meth}`~aquaflux.schemes.CorrectedGreenGauss.calibrated` as well: the count that reaches a
given accuracy belongs to the operator–preconditioner pairing it was measured on, and the
calibrated scheme carries the preconditioner with it for that reason.

## Choosing the sweep count

A fixed sweep count carries no convergence test, so it is worth knowing what sets it: the
**mesh's skewness, not its size**. That is what keeps the cost linear in the number of cells,
and it also means a count calibrated on a small test mesh does not transfer to a skewed one.

For the corrected gradient, the default of four sweeps reaches the exactly-solved gradient to
well within discretization error on mildly non-orthogonal meshes, and a strongly skewed mesh
wants more. For the Hessian-corrected scheme, the inner Hessian solve contracts geometrically
at a rate set by skewness alone: measured on a heavily skewed hexahedral grid, the
reconstruction's departure from exactness for a quadratic falls by about two and a half orders
per two sweeps, reaching the exactly-solved answer to machine precision at the default of ten.

The way to check a mesh you have not used before is to compare against an exact solve of the
same system, which is what `GmresGradientSolve` provides:

```python
approx = HessianCorrectedGradient().gradients(phi, mesh, geometry, boundary_values)
exact = HessianCorrectedGradient(
    hessian_solver=GmresGradientSolve()
).gradients(phi, mesh, geometry, boundary_values)
```

If those agree to well below your discretization error, the default is sufficient on that
mesh. A quadratic field is the sharpest probe, since the scheme is defined to reproduce it
exactly and any departure is the solve rather than the discretization.

```{warning}
Because the sweep is fixed, an under-resolved mesh loses that exactness **silently** — there
is no residual test to fail. Check it once per new mesh rather than assuming.
```

### Measuring the count instead of choosing it

The comparison above answers the question for one mesh by hand. The same question can be asked
of the mesh directly, because the answer is a property of it. The Richardson iteration started
from zero has error `M^k e_0` with `M = I - P^-1 A`, and both `A` and `P` are built from the
geometry alone — so its convergence rate `rho(M)` is fixed once the mesh is, and the count
reaching a relative error of `tol` is `ceil(log(tol) / log(rho))`.

{func}`~aquaflux.schemes.contraction_rate` measures that rate, and each scheme has a factory
that turns it into a scheme carrying the right count:

```python
from aquaflux.schemes import CorrectedGreenGauss, HessianCorrectedGradient

scheme = CorrectedGreenGauss.calibrated(mesh, geometry)              # tol=1e-4 by default
tight = CorrectedGreenGauss.calibrated(mesh, geometry, tol=1e-10)
betchen = HessianCorrectedGradient.calibrated(mesh, geometry)        # sizes both its systems
```

How far apart the answers are is the reason to ask. On an orthogonal mesh the skewness
correction vanishes, the system is diagonal, and **one** sweep is already exact; on a randomly
perturbed grid at 40 % the same tolerance wants **twelve**. A single count cannot be right for
both, and the one that is generous on the first is short on the second.

The estimate costs its apply budget once — about eight four-sweep reconstructions at the
default — against a saving on every reconstruction thereafter, and it reports
{attr}`~aquaflux.schemes.ContractionRate.settling_ratio` as a self-check that the budget was
long enough.

```{warning}
`tol` bounds the gradient's relative error in the **Euclidean norm over all cells**, not cell
by cell. Measured at the calibrated count across meshes from mildly to heavily non-orthogonal,
the worst single cell ran five to sixty times the norm figure and exceeded `tol` itself in
about half of them — the rate governs a norm, not the extremes, so a mesh whose skewness sits
in a few cells reaches the norm target with those cells still short of it. Ask for one to two
orders tighter than the per-cell accuracy you need.
```

```{note}
The count is a plain Python integer held in the solve strategy, so calibrate **outside** any
`jax.jit` or differentiated region and use the resulting scheme inside — calibrating under a
trace raises rather than silently doing something else. Under domain decomposition, calibrate
on the global mesh before partitioning.
```

## Choosing a scheme

| | mesh it suits | exact for | solve cost |
|---|---|---|---|
| {class}`~aquaflux.schemes.CompactGreenGauss` | near-orthogonal | linear, on orthogonal grids | one pass, no system |
| {class}`~aquaflux.schemes.CorrectedGreenGauss` | any | linear, on any mesh | a few sparse sweeps |
| {class}`~aquaflux.schemes.HessianCorrectedGradient` | skewed, where the gradient leads | linear and quadratic | an inner and an outer solve |

Two practical points beyond accuracy.

**The Hessian-corrected scheme has a much longer stencil.** Its gradient couples to the
Hessian, which couples to the neighbours' Hessians, so the reconstruction at a cell depends on
cells far across the mesh — where the corrected gradient's dependence stops a fixed number of
rings away, set by its sweep count. That is invisible to the solve itself, but it matters to
anything that assembles an approximate Jacobian over a bounded stencil: such an approximation
does not simply drop the couplings it cannot see. Preconditioners that build their own
operator from the mesh — the block preconditioners described in
[Preconditioning](preconditioning.md) — are unaffected, because they never try to represent
the gradient's coupling in the first place.

**Only the corrected gradient runs under domain decomposition.** Its Richardson sweeps form no
global inner product, so a partitioned solve can refresh ghost values once per sweep and get
owned gradients identical to a serial run. The Krylov solve and the Hessian-corrected scheme
raise rather than return a quietly wrong answer.

## A reference

Betchen, L. J. and Straatman, A. G. (2010). "An accurate gradient and Hessian reconstruction
method for cell-centered finite volume discretizations on general unstructured grids."
*International Journal for Numerical Methods in Fluids* 62(9), 945–962.
